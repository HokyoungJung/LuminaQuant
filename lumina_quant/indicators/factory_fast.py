"""High-throughput helper indicators for strategy-candidate research.

The functions in this module are intentionally lightweight and NumPy-first,
with optional Numba acceleration for hot rolling-window calculations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration backend
    njit = None

NUMBA_AVAILABLE = njit is not None


def _to_np(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _finite_tail(arr: np.ndarray, window: int) -> np.ndarray | None:
    window_i = max(2, int(window))
    if arr.size < window_i:
        return None
    tail = arr[-window_i:]
    if not np.all(np.isfinite(tail)):
        return None
    return tail


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _rolling_slope_latest_numba(arr: np.ndarray, window: int) -> float:  # pragma: no cover
        n = arr.shape[0]
        window_i = 2 if window < 2 else window
        if n < window_i:
            return np.nan

        start = n - window_i
        sum_y = 0.0
        sum_xy = 0.0
        sum_x = 0.0
        sum_x2 = 0.0
        x = 0.0
        for idx in range(start, n):
            y = arr[idx]
            if not np.isfinite(y):
                return np.nan
            sum_y += y
            sum_xy += x * y
            sum_x += x
            sum_x2 += x * x
            x += 1.0

        denom = (window_i * sum_x2) - (sum_x * sum_x)
        if abs(denom) <= 1e-12:
            return np.nan
        numer = (window_i * sum_xy) - (sum_x * sum_y)
        return numer / denom

    @njit(cache=True)
    def _rolling_std_latest_numba(arr: np.ndarray, window: int) -> float:  # pragma: no cover
        n = arr.shape[0]
        window_i = 2 if window < 2 else window
        if n < window_i:
            return np.nan
        start = n - window_i

        mean_value = 0.0
        count = 0.0
        for idx in range(start, n):
            value = arr[idx]
            if not np.isfinite(value):
                return np.nan
            mean_value += value
            count += 1.0
        mean_value /= count

        variance = 0.0
        for idx in range(start, n):
            diff = arr[idx] - mean_value
            variance += diff * diff
        variance /= max(1.0, count - 1.0)
        if variance <= 0.0:
            return np.nan
        return math.sqrt(variance)

else:

    def _rolling_slope_latest_numba(arr: np.ndarray, window: int) -> float:
        tail = _finite_tail(arr, window)
        if tail is None:
            return float("nan")
        x = np.arange(tail.size, dtype=np.float64)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(tail))
        numer = float(np.sum((x - x_mean) * (tail - y_mean)))
        denom = float(np.sum((x - x_mean) ** 2))
        if abs(denom) <= 1e-12:
            return float("nan")
        return numer / denom

    def _rolling_std_latest_numba(arr: np.ndarray, window: int) -> float:
        tail = _finite_tail(arr, window)
        if tail is None:
            return float("nan")
        std_value = float(np.std(tail, ddof=1))
        if not math.isfinite(std_value) or std_value <= 0.0:
            return float("nan")
        return std_value


def rolling_slope_latest(values: Sequence[float] | np.ndarray, *, window: int = 20) -> float | None:
    """Return linear-regression slope of the latest rolling window."""
    arr = _to_np(values)
    value = float(_rolling_slope_latest_numba(arr, int(window)))
    return value if math.isfinite(value) else None


def rolling_range_position_latest(
    highs: Sequence[float] | np.ndarray,
    lows: Sequence[float] | np.ndarray,
    closes: Sequence[float] | np.ndarray,
    *,
    window: int = 20,
) -> float | None:
    """Return normalized close position in trailing high/low range (0..1)."""
    high_arr = _to_np(highs)
    low_arr = _to_np(lows)
    close_arr = _to_np(closes)

    n = min(high_arr.size, low_arr.size, close_arr.size)
    window_i = max(2, int(window))
    if n < window_i:
        return None

    high_tail = high_arr[-window_i:]
    low_tail = low_arr[-window_i:]
    close_value = float(close_arr[-1])
    if (
        not np.all(np.isfinite(high_tail))
        or not np.all(np.isfinite(low_tail))
        or not math.isfinite(close_value)
    ):
        return None

    highest = float(np.max(high_tail))
    lowest = float(np.min(low_tail))
    span = highest - lowest
    if span <= 1e-12:
        return 0.5

    value = (close_value - lowest) / span
    if not math.isfinite(value):
        return None
    return min(1.0, max(0.0, float(value)))


def volatility_ratio_latest(
    values: Sequence[float] | np.ndarray,
    *,
    fast_window: int = 20,
    slow_window: int = 80,
) -> float | None:
    """Return fast/slow rolling volatility ratio (>1 = expansion, <1 = compression)."""
    arr = _to_np(values)
    fast_i = max(2, int(fast_window))
    slow_i = max(fast_i + 1, int(slow_window))
    if arr.size < slow_i:
        return None

    fast_std = float(_rolling_std_latest_numba(arr, fast_i))
    slow_std = float(_rolling_std_latest_numba(arr, slow_i))
    if not math.isfinite(fast_std) or not math.isfinite(slow_std) or slow_std <= 1e-12:
        return None
    ratio = fast_std / slow_std
    return float(ratio) if math.isfinite(ratio) and ratio > 0.0 else None


def composite_momentum_latest(
    values: Sequence[float] | np.ndarray,
    *,
    windows: Sequence[int] = (8, 21, 55),
    weights: Sequence[float] = (0.5, 0.3, 0.2),
) -> float | None:
    """Return weighted multi-horizon momentum score from close history."""
    arr = _to_np(values)
    if arr.size < 3:
        return None
    if not np.all(np.isfinite(arr)):
        return None

    win_list = [max(1, int(win)) for win in windows]
    weight_list = [float(weight) for weight in weights]
    if not win_list or len(win_list) != len(weight_list):
        return None

    latest = float(arr[-1])
    if latest <= 0.0:
        return None

    score = 0.0
    total_weight = 0.0
    for win, weight in zip(win_list, weight_list, strict=True):
        if arr.size <= win:
            continue
        base = float(arr[-1 - win])
        if base <= 0.0 or not math.isfinite(base):
            continue
        ret = math.log(latest / base)
        score += weight * ret
        total_weight += abs(weight)

    if total_weight <= 1e-12:
        return None
    value = score / total_weight
    return float(value) if math.isfinite(value) else None


__all__ = [
    "NUMBA_AVAILABLE",
    "composite_momentum_latest",
    "rolling_range_position_latest",
    "rolling_slope_latest",
    "volatility_ratio_latest",
]
