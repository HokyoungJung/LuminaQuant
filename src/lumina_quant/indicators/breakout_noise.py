"""Open-anchored volatility-breakout primitives (Larry Williams / systrader79 lineage).

Public-domain rules independently adapted from Korean retail-quant educational
material (systrader79 blog/books, widely reproduced crypto-bot variants):

* ``noise = 1 - |close - open| / (high - low)`` -- bar "noise" ratio; low noise
  means the bar travelled directionally (trend-friendly), high noise means the
  range was mostly wick.
* ``breakout level = today_open + K * (prev_high - prev_low)`` -- Williams range
  projection; ``K`` is often set adaptively to the trailing average noise ratio.
* ``moving-average score`` -- fraction of a set of trailing SMAs sitting below the
  latest close (0..1); systrader79's "market-timing score" used to scale exposure.
* ``range-based vol-target weight = target_vol / (prev_range / prev_close)`` --
  the crude but robust "volatility control" position weight from the same lineage.

Every function is a pure latest-value indicator (``float | None``), never raises,
and follows the ``indicators/`` convention of returning ``None`` on degenerate
input instead of guessing.  These are HYPOTHESIS primitives; nothing here claims
a reproduction of any author's actual live rules or results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .common import safe_float
from .moving_average import simple_moving_average

_DEFAULT_MA_SCORE_WINDOWS: tuple[int, ...] = (3, 5, 10, 20)


def bar_noise_ratio(open_price, high, low, close) -> float | None:
    """Return ``1 - |close-open| / (high-low)`` on ``[0, 1]``.

    ``None`` when the bar has no range or any input is non-finite.
    """
    o, h, lo, c = (safe_float(v) for v in (open_price, high, low, close))
    if o is None or h is None or lo is None or c is None:
        return None
    bar_range = h - lo
    if bar_range <= 0.0:
        return None
    ratio = 1.0 - abs(c - o) / bar_range
    return min(1.0, max(0.0, ratio))


def average_noise_ratio(opens, highs, lows, closes, *, period: int = 20) -> float | None:
    """Trailing mean of :func:`bar_noise_ratio` over the last ``period`` bars.

    Requires ``period`` complete bars in every series; bars with a zero range are
    skipped, and ``None`` is returned when fewer than half of the window
    contributed a valid ratio (so the average is not driven by a handful of bars).
    """
    period_i = max(1, int(period))
    series = [list(x) for x in (opens, highs, lows, closes)]
    if min(len(s) for s in series) < period_i:
        return None
    tail = [s[-period_i:] for s in series]
    ratios: list[float] = []
    for o, h, lo, c in zip(*tail):
        ratio = bar_noise_ratio(o, h, lo, c)
        if ratio is not None:
            ratios.append(ratio)
    if not ratios or len(ratios) * 2 < period_i:
        return None
    return sum(ratios) / float(len(ratios))


def volatility_breakout_levels(
    open_price, prev_high, prev_low, *, k: float
) -> tuple[float | None, float | None]:
    """Return ``(upper, lower)`` Williams-style breakout triggers.

    ``upper = open + k*range`` (long trigger), ``lower = open - k*range`` (short
    trigger) where ``range = prev_high - prev_low`` is the PREVIOUS session's
    range.  ``(None, None)`` on non-finite input, non-positive range or ``k<=0``.
    """
    o, h, lo = (safe_float(v) for v in (open_price, prev_high, prev_low))
    k_f = safe_float(k)
    if o is None or h is None or lo is None or k_f is None or k_f <= 0.0:
        return None, None
    prev_range = h - lo
    if prev_range <= 0.0:
        return None, None
    return o + k_f * prev_range, o - k_f * prev_range


def moving_average_score(
    closes, *, windows: Sequence[int] = _DEFAULT_MA_SCORE_WINDOWS
) -> float | None:
    """Fraction of trailing SMAs (``windows``) strictly below the latest close.

    Returns a value on ``[0, 1]``: 1.0 = price above every MA (full "risk-on"),
    0.0 = below every MA.  ``None`` if history is shorter than the longest window
    or ``windows`` is empty.
    """
    wins = sorted({max(1, int(w)) for w in windows})
    if not wins:
        return None
    series = [v for v in (safe_float(x) for x in list(closes)) if v is not None]
    if len(series) < wins[-1] or len(series) != len(list(closes)):
        return None
    last = series[-1]
    above = 0
    for w in wins:
        sma = simple_moving_average(series, w)
        if sma is None:
            return None
        if last > sma:
            above += 1
    return above / float(len(wins))


def range_volatility_target_weight(
    prev_high, prev_low, prev_close, *, target_vol: float, cap: float = 1.0
) -> float | None:
    """``min(cap, target_vol / (prev_range / prev_close))`` -- range-based vol control.

    ``target_vol`` and the realized proxy are both PER-BAR fractions (e.g. a 2%
    daily target against yesterday's ``(high-low)/close``).  ``None`` on
    non-finite input, non-positive range/close or non-positive ``target_vol``.
    """
    h, lo, c = (safe_float(v) for v in (prev_high, prev_low, prev_close))
    tv = safe_float(target_vol)
    cap_f = safe_float(cap)
    if h is None or lo is None or c is None or tv is None or tv <= 0.0 or c <= 0.0:
        return None
    realized = (h - lo) / c
    if realized <= 0.0 or not math.isfinite(realized):
        return None
    weight = tv / realized
    if cap_f is not None and cap_f > 0.0:
        weight = min(cap_f, weight)
    return max(0.0, weight)


__all__ = [
    "average_noise_ratio",
    "bar_noise_ratio",
    "moving_average_score",
    "range_volatility_target_weight",
    "volatility_breakout_levels",
]
