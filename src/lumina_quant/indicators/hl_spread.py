"""Corwin-Schultz (2012) high-low bid-ask spread estimator (OHLC-only).

THEORY.  Corwin & Schultz (2012, JF 67(2)) recover the effective bid-ask
SPREAD of a security from daily HIGH and LOW prices alone.  The insight: over a
two-day (two-bar) window the observed high-low RANGE has two additive
components -- a VARIANCE component that scales (roughly) with the number of
periods, and a SPREAD component that does NOT.  Comparing the sum of two
single-bar squared log-ranges (``beta``) against the squared log-range of the
two bars COMBINED (``gamma``) isolates the spread: heavy bar-to-bar OVERLAP
(bid-ask bounce inside a stable band) inflates ``beta`` relative to ``gamma``
and yields a POSITIVE estimated spread, while clean directional diffusion (a
staircase where each bar extends the prior range) yields ``alpha < 0`` which is
clamped to zero.

Crucially the estimate depends on the OVERLAP geometry of adjacent bars, NOT
just on the per-bar range: two paths with an IDENTICAL per-bar ``ln(H/L)``
(hence identical Parkinson volatility) can produce a zero spread (clean
staircase) or a strictly positive spread (full overlap).  That is what makes
this a spread-stress gauge rather than a volatility proxy.

This module is the pure-Python (``math`` only) primitive; the strategy layer
z-scores the returned rolling-smoothed spread against its own trailing
distribution to detect spread-STRESS episodes.  No overnight-gap adjustment is
applied (the target book is 24/7 crypto, so the classic Corwin-Schultz
overnight correction does not apply).

Function:
- :func:`corwin_schultz_spread`: the rolling-mean Corwin-Schultz spread over the
  most recent ``smooth_window`` adjacent-bar pairs, or ``None`` when the input
  is too short or contains a degenerate bar (``H <= 0``, ``L <= 0``, ``H < L``,
  or a non-finite value).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

# 3 - 2*sqrt(2): the Corwin-Schultz normalisation constant (denominator of the
# alpha expression).  Precomputed once; strictly positive (~0.1715729).
_CS_DENOM = 3.0 - 2.0 * math.sqrt(2.0)


def _pair_spread(high_0: float, low_0: float, high_1: float, low_1: float) -> float | None:
    """Return the single-pair Corwin-Schultz spread ``S`` (clamped at 0), or ``None``.

    ``beta`` sums the two single-bar squared log-ranges; ``gamma`` is the
    squared log-range of the two bars combined.  ``S`` is the standard
    Corwin-Schultz transform ``2*(exp(alpha) - 1) / (1 + exp(alpha))`` with a
    negative ``alpha`` clamped to a zero spread.
    """
    ln_range_0 = math.log(high_0 / low_0)
    ln_range_1 = math.log(high_1 / low_1)
    beta = ln_range_0 * ln_range_0 + ln_range_1 * ln_range_1
    combined_high = high_0 if high_0 >= high_1 else high_1
    combined_low = low_0 if low_0 <= low_1 else low_1
    gamma_root = math.log(combined_high / combined_low)
    gamma = gamma_root * gamma_root
    alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / _CS_DENOM - math.sqrt(gamma / _CS_DENOM)
    if not math.isfinite(alpha):
        return None
    if alpha <= 0.0:
        # Negative alpha -> the standard Corwin-Schultz zero-spread clamp.
        return 0.0
    try:
        exp_alpha = math.exp(alpha)
    except OverflowError:
        # alpha -> +inf: S saturates at its 2.0 asymptote.
        return 2.0
    spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
    if not math.isfinite(spread):
        return None
    return max(0.0, spread)


def corwin_schultz_spread(
    highs: Iterable[float],
    lows: Iterable[float],
    *,
    smooth_window: int = 5,
) -> float | None:
    """Return the rolling-mean Corwin-Schultz spread over the recent bars.

    The estimator is computed for each adjacent pair among the trailing
    ``smooth_window + 1`` bars and averaged (the rolling smoothing recommended
    by Corwin & Schultz to reduce single-pair noise).  Only the trailing
    ``smooth_window + 1`` bars are inspected, so a stale degenerate bar far in
    the past does not poison a currently-clean window.

    Guards (never raises): requires at least ``smooth_window + 1`` aligned bars;
    returns ``None`` if any inspected bar is non-finite, has a non-positive high
    or low, or has ``high < low``; returns ``None`` on a non-finite result.

    Args:
        highs: Per-bar high prices (chronological).
        lows: Per-bar low prices (chronological, aligned to ``highs``).
        smooth_window: Number of adjacent-bar pairs to average (>= 1).

    Returns:
        The smoothed Corwin-Schultz spread estimate (>= 0.0), or ``None`` when
        unavailable.
    """
    window = max(1, int(smooth_window))
    high_list = list(highs)
    low_list = list(lows)
    depth = min(len(high_list), len(low_list))
    need = window + 1
    if depth < need:
        return None
    tail_high = high_list[-need:]
    tail_low = low_list[-need:]
    clean_high: list[float] = []
    clean_low: list[float] = []
    for raw_high, raw_low in zip(tail_high, tail_low, strict=False):
        try:
            high_value = float(raw_high)
            low_value = float(raw_low)
        except TypeError, ValueError:
            return None
        if not math.isfinite(high_value) or not math.isfinite(low_value):
            return None
        if high_value <= 0.0 or low_value <= 0.0 or high_value < low_value:
            return None
        clean_high.append(high_value)
        clean_low.append(low_value)
    spreads: list[float] = []
    for idx in range(len(clean_high) - 1):
        pair = _pair_spread(
            clean_high[idx],
            clean_low[idx],
            clean_high[idx + 1],
            clean_low[idx + 1],
        )
        if pair is None:
            return None
        spreads.append(pair)
    if not spreads:
        return None
    result = math.fsum(spreads) / float(len(spreads))
    return result if math.isfinite(result) else None


__all__ = [
    "corwin_schultz_spread",
]
