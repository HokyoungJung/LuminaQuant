"""Formulaic alpha operator primitives with tunable parameters."""

from __future__ import annotations

import math
from itertools import pairwise

from .rolling_stats import rolling_corr, sample_std


def _tail(values, window: int) -> list[float] | None:
    window_i = max(1, int(window))
    if len(values) < window_i:
        return None
    return [float(value) for value in list(values)[-window_i:]]


def delay(values, periods: int = 1) -> float | None:
    """Return value ``periods`` bars ago."""
    period_i = max(1, int(periods))
    if len(values) <= period_i:
        return None
    return float(list(values)[-1 - period_i])


def delta(values, periods: int = 1) -> float | None:
    """Return latest minus delayed value."""
    delayed = delay(values, periods)
    if delayed is None:
        return None
    latest = float(list(values)[-1])
    return latest - delayed


def ts_sum(values, window: int) -> float | None:
    """Return rolling sum over trailing window."""
    tail = _tail(values, window)
    if tail is None:
        return None
    return sum(tail)


def ts_product(values, window: int) -> float | None:
    """Return rolling product over trailing window."""
    tail = _tail(values, window)
    if tail is None:
        return None
    out = 1.0
    for value in tail:
        out *= value
    return out


def ts_min(values, window: int) -> float | None:
    """Return rolling minimum over trailing window."""
    tail = _tail(values, window)
    if tail is None:
        return None
    return min(tail)


def ts_max(values, window: int) -> float | None:
    """Return rolling maximum over trailing window."""
    tail = _tail(values, window)
    if tail is None:
        return None
    return max(tail)


def ts_argmin(values, window: int) -> float | None:
    """Return 1-indexed position of trailing minimum in window."""
    tail = _tail(values, window)
    if tail is None:
        return None
    index = min(range(len(tail)), key=lambda idx: tail[idx])
    return float(index + 1)


def ts_argmax(values, window: int) -> float | None:
    """Return 1-indexed position of trailing maximum in window."""
    tail = _tail(values, window)
    if tail is None:
        return None
    index = max(range(len(tail)), key=lambda idx: tail[idx])
    return float(index + 1)


def ts_stddev(values, window: int) -> float | None:
    """Return sample standard deviation over trailing window."""
    tail = _tail(values, window)
    if tail is None:
        return None
    return sample_std(tail)


def rank_pct(values, window: int = 20) -> float | None:
    """Return percentile rank of latest value in trailing window."""
    window_i = max(2, int(window))
    tail = _tail(values, window_i)
    if tail is None:
        return None
    latest = tail[-1]
    below = sum(1 for value in tail if value < latest)
    equal = sum(1 for value in tail if value == latest)
    return (below + 0.5 * equal) / float(window_i)


def ts_rank(values, window: int = 20) -> float | None:
    """Alias for trailing percentile rank."""
    return rank_pct(values, window=window)


def signed_power(value: float, power: float) -> float:
    """Return signed power ``sign(x) * |x|**p``."""
    value_f = float(value)
    p = float(power)
    if value_f == 0.0:
        return 0.0
    return math.copysign(abs(value_f) ** p, value_f)


def decay_linear(values, window: int) -> float | None:
    """Return linearly decayed weighted average over trailing window."""
    window_i = max(1, int(window))
    tail = _tail(values, window_i)
    if tail is None:
        return None
    denom = float(window_i * (window_i + 1) // 2)
    numer = 0.0
    for weight, value in enumerate(tail, start=1):
        numer += float(weight) * float(value)
    return numer / denom


def ts_correlation(x_values, y_values, window: int) -> float | None:
    """Return trailing correlation over aligned windows."""
    window_i = max(2, int(window))
    n = min(len(x_values), len(y_values))
    if n < window_i:
        return None
    x_tail = [float(value) for value in list(x_values)[-window_i:]]
    y_tail = [float(value) for value in list(y_values)[-window_i:]]
    return rolling_corr(x_tail, y_tail)


def ts_covariance(x_values, y_values, window: int) -> float | None:
    """Return trailing sample covariance over aligned windows."""
    window_i = max(2, int(window))
    n = min(len(x_values), len(y_values))
    if n < window_i:
        return None
    x_tail = [float(value) for value in list(x_values)[-window_i:]]
    y_tail = [float(value) for value in list(y_values)[-window_i:]]
    mean_x = sum(x_tail) / float(window_i)
    mean_y = sum(y_tail) / float(window_i)
    cov = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x_tail, y_tail, strict=False))
    cov /= float(window_i - 1)
    return cov if math.isfinite(cov) else None


def scale_latest(values, *, a: float = 1.0, window: int = 20) -> float | None:
    """Scale latest value by trailing absolute-sum normalization."""
    window_i = max(1, int(window))
    tail = _tail(values, window_i)
    if tail is None:
        return None
    denom = sum(abs(value) for value in tail)
    if denom <= 1e-12:
        return None
    return float(a) * tail[-1] / denom


def returns_from_close(closes) -> list[float]:
    """Return close-to-close simple returns."""
    closes_f = [float(value) for value in closes]
    out: list[float] = []
    for prev, curr in pairwise(closes_f):
        if abs(prev) <= 1e-12:
            out.append(0.0)
        else:
            out.append((curr / prev) - 1.0)
    return out


def adv(closes, volumes, window: int = 20) -> float | None:
    """Return average daily dollar volume over trailing window."""
    window_i = max(1, int(window))
    n = min(len(closes), len(volumes))
    if n < window_i:
        return None
    closes_tail = [float(value) for value in list(closes)[-window_i:]]
    volumes_tail = [max(0.0, float(value)) for value in list(volumes)[-window_i:]]
    dollars = [close * volume for close, volume in zip(closes_tail, volumes_tail, strict=False)]
    return sum(dollars) / float(window_i)
