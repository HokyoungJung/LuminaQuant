"""Volume-synchronized flow-toxicity primitives (VPIN / Bulk Volume Classification).

THEORY / PROVENANCE.  Easley, Lopez de Prado & O'Hara (2012, "Flow Toxicity
and Liquidity in a High-Frequency World," Review of Financial Studies)
introduce the Volume-synchronized Probability of INformed trading (VPIN):
rather than sampling in wall-clock time, traded volume is partitioned into
fixed-size BUCKETS (equal quantities of volume), each bucket's flow is
classified into buy- and sell-initiated volume, and VPIN is the trailing
average of the ABSOLUTE buy/sell imbalance across a rolling window of
buckets, ``mean(|V_buy - V_sell| / V)``.  A persistently high VPIN reflects
"toxic" order flow -- market makers are being adversely selected by one-sided
informed flow -- and the same paper documents VPIN spiking ahead of the 2010
Flash Crash: flow toxicity is a LEADING indicator of volatility events, not
merely a contemporaneous liquidity descriptor.

Classifying buy vs. sell volume normally requires tick-by-tick trade data
(the tick rule, or a quote-based classifier).  Easley, Lopez de Prado &
O'Hara (2012) also give the BULK VOLUME CLASSIFICATION (BVC) approximation,
which works from OHLCV bars alone: a bar's volume is split into a BUY
fraction ``Phi(dP / sigma)`` and a SELL fraction ``1 - Phi(dP / sigma)``,
where ``dP`` is the bar's price change, ``sigma`` is a trailing estimate of
the standard deviation of price changes, and ``Phi`` is the standard normal
CDF -- a bar whose price rose sharply relative to its own recent volatility
is assumed mostly buy-initiated, and vice versa.  This module implements BVC
(:func:`bulk_volume_buy_fraction`), the VPIN aggregation
(:func:`vpin_from_buckets`), and a small deterministic volume-bucket
accumulator (:func:`accumulate_volume_bucket`) so VPIN can be built from
OHLCV bars with no trade-level feed.

All functions are pure, dependency-light (``math`` only), and ``None``-safe
(never raise).  :func:`accumulate_volume_bucket` is the one function that
folds in rolling state, and it does so as an explicit pure transform over a
``state`` argument the caller supplies and receives back -- the CALLER owns
and persists that rolling state (e.g. as part of a strategy's per-symbol
state / ``get_state``/``set_state``); this module holds none of it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

_EPS = 1e-12


def bulk_volume_buy_fraction(price_change, sigma) -> float | None:
    """Return the BVC buy-side volume fraction ``Phi(price_change / sigma)``.

    Bulk Volume Classification (Easley, Lopez de Prado & O'Hara 2012): absent
    tick data, a bar's volume is assigned a BUY fraction equal to the
    standard normal CDF of the bar's price change normalized by a trailing
    estimate of price-change volatility (``sigma``).  ``Phi`` is evaluated
    exactly via ``math.erf`` (``Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))``), so
    the result is always in ``[0, 1]``: near 1 means the bar was almost
    entirely buy-initiated (a strong up-move relative to recent volatility),
    near 0 means almost entirely sell-initiated, and 0.5 is the neutral /
    no-move reading.

    Guards (never raises): returns ``None`` when ``sigma`` is non-positive,
    or when either input is non-finite or cannot be coerced to ``float`` --
    the classification is undefined without a valid volatility scale.

    Args:
        price_change: The bar's price change (e.g. ``close - prev_close``),
            in the SAME units as ``sigma``.
        sigma: A trailing standard deviation of price CHANGES (not returns).

    Returns:
        The BVC buy fraction in ``[0, 1]``, or ``None`` when unavailable.
    """
    try:
        dp = float(price_change)
        sig = float(sigma)
    except TypeError, ValueError:
        return None
    if not math.isfinite(dp) or not math.isfinite(sig) or sig <= 0.0:
        return None
    z = dp / sig
    if not math.isfinite(z):
        return None
    fraction = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.0, min(1.0, fraction))


def vpin_from_buckets(bucket_imbalances: Sequence[float] | Iterable[float]) -> float | None:
    """Return VPIN: the mean absolute normalized imbalance over trailing buckets.

    Easley, Lopez de Prado & O'Hara (2012): VPIN is the trailing average of
    ``|V_buy - V_sell| / V`` across a rolling window of EQUAL-SIZED volume
    buckets.  ``bucket_imbalances`` are the per-bucket SIGNED normalized
    imbalances ``(V_buy - V_sell) / V`` (each expected in ``[-1, 1]``, e.g.
    as produced by :func:`accumulate_volume_bucket`); this function returns
    the mean of their absolute values over whatever trailing sequence the
    caller passes in (the caller is responsible for windowing to the desired
    trailing bucket count).

    Guards (never raises): returns ``None`` on an empty input, or when no
    finite numeric values remain after filtering.

    Args:
        bucket_imbalances: Trailing per-bucket signed normalized imbalances.

    Returns:
        The VPIN estimate in ``[0, 1]``, or ``None`` when unavailable.
    """
    finite: list[float] = []
    for raw in bucket_imbalances:
        try:
            value = float(raw)
        except TypeError, ValueError:
            continue
        if math.isfinite(value):
            finite.append(value)
    if not finite:
        return None
    vpin = sum(abs(value) for value in finite) / float(len(finite))
    return vpin if math.isfinite(vpin) else None


def accumulate_volume_bucket(
    state: tuple[float, float] | None,
    *,
    buy_volume: float,
    sell_volume: float,
    bucket_size: float,
) -> tuple[tuple[float, float], list[float]]:
    """Fold one bar's classified volume into fixed-size volume buckets.

    Deterministic, dependency-light bucket accumulator for VPIN construction
    from OHLCV bars.  ``state`` is the ``(accumulated_buy, accumulated_sell)``
    carry from the PREVIOUS call for this symbol (``None`` to start a fresh
    accumulator).  This function is a pure transform over that explicit
    state -- it is the CALLER's responsibility to persist ``state`` across
    bars (e.g. as part of a strategy's per-symbol state /
    ``get_state``/``set_state``); this module holds no rolling state of its
    own.

    Each call adds ``buy_volume``/``sell_volume`` to the carry; whenever the
    accumulated total reaches ``bucket_size`` a bucket CLOSES and its signed
    normalized imbalance ``(buy_frac - sell_frac)`` (using the overall
    buy/sell split of whatever volume is in the accumulator at that instant)
    is appended to the returned list.  This is an explicit, honestly
    documented OHLCV-granularity approximation: with only bar-level volume
    (no trade-level sequencing) there is no way to know exactly which slice
    of a bar's volume falls in which bucket, so the bar's overall buy/sell
    split is applied uniformly across every bucket it contributes to.  A
    single unusually heavy bar can close MORE THAN ONE bucket; this is
    handled by looping, so the returned list may have zero, one, or several
    entries.

    Guards (never raises): a non-positive, non-finite, or non-numeric
    ``bucket_size`` leaves the accumulator unchanged and closes no buckets;
    non-finite or negative volume inputs are treated as zero.  Callers should
    read a persistently empty return list as "the bucket clock has not
    advanced," not as an error.

    Args:
        state: The prior ``(accumulated_buy, accumulated_sell)`` carry, or
            ``None`` to start a fresh accumulator.
        buy_volume: This bar's BUY-classified volume (non-negative).
        sell_volume: This bar's SELL-classified volume (non-negative).
        bucket_size: The fixed volume-per-bucket threshold.

    Returns:
        A tuple ``(new_state, completed_bucket_imbalances)``.
    """
    carry = state if isinstance(state, tuple) and len(state) == 2 else (0.0, 0.0)
    try:
        acc_buy = float(carry[0])
        acc_sell = float(carry[1])
    except TypeError, ValueError:
        acc_buy, acc_sell = 0.0, 0.0
    if not math.isfinite(acc_buy) or acc_buy < 0.0:
        acc_buy = 0.0
    if not math.isfinite(acc_sell) or acc_sell < 0.0:
        acc_sell = 0.0

    try:
        size = float(bucket_size)
    except TypeError, ValueError:
        size = 0.0
    if not math.isfinite(size) or size <= _EPS:
        return (acc_buy, acc_sell), []

    try:
        buy = float(buy_volume)
    except TypeError, ValueError:
        buy = 0.0
    try:
        sell = float(sell_volume)
    except TypeError, ValueError:
        sell = 0.0
    if not math.isfinite(buy) or buy < 0.0:
        buy = 0.0
    if not math.isfinite(sell) or sell < 0.0:
        sell = 0.0

    acc_buy += buy
    acc_sell += sell
    completed: list[float] = []
    while (acc_buy + acc_sell) >= size:
        total = acc_buy + acc_sell
        if total <= _EPS:
            break
        frac_buy = acc_buy / total
        frac_sell = acc_sell / total
        completed.append(max(-1.0, min(1.0, frac_buy - frac_sell)))
        acc_buy -= frac_buy * size
        acc_sell -= frac_sell * size
        # Numerical safety valve: clamp float-subtraction residue so the
        # accumulator never drifts (even slightly) below zero.
        acc_buy = max(0.0, acc_buy)
        acc_sell = max(0.0, acc_sell)
    return (acc_buy, acc_sell), completed


__all__ = [
    "accumulate_volume_bucket",
    "bulk_volume_buy_fraction",
    "vpin_from_buckets",
]
