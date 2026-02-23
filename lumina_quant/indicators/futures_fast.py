"""High-performance futures indicator helpers.

These helpers are intentionally light-weight and NumPy-first so they can run
inside fast parameter-search loops. Optional Numba kernels are used when
available.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional runtime acceleration backend
    njit = None

NUMBA_AVAILABLE = njit is not None


def _to_np(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _as_float_or_none(value: float) -> float | None:
    out = float(value)
    if math.isfinite(out):
        return out
    return None


def _rolling_log_return_volatility_latest_python(closes: np.ndarray, window: int) -> float:
    if closes.size < window + 1:
        return float("nan")
    tail = closes[-(window + 1) :]
    if not np.all(np.isfinite(tail)) or np.any(tail <= 0.0):
        return float("nan")
    rets = np.log(tail[1:] / tail[:-1])
    if rets.size < 2:
        return float("nan")
    return float(np.std(rets, ddof=1))


def _normalized_true_range_latest_python(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int,
) -> float:
    n = min(highs.size, lows.size, closes.size)
    if n < window:
        return float("nan")
    h = highs[-window:]
    low_arr = lows[-window:]
    c = closes[-window:]
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(low_arr)) or not np.all(np.isfinite(c)):
        return float("nan")
    if np.any(c <= 0.0):
        return float("nan")
    tr_norm = (h - low_arr) / c
    if np.any(tr_norm < 0.0):
        return float("nan")
    return float(np.mean(tr_norm))


def _volume_shock_zscore_latest_python(volumes: np.ndarray, window: int) -> float:
    if volumes.size < window + 1:
        return float("nan")
    hist = volumes[-(window + 1) : -1]
    current = float(volumes[-1])
    if not np.all(np.isfinite(hist)) or not math.isfinite(current):
        return float("nan")
    std = float(np.std(hist, ddof=1))
    if std <= 0.0:
        return 0.0
    mean = float(np.mean(hist))
    return float((current - mean) / std)


def _trend_efficiency_latest_python(closes: np.ndarray, window: int) -> float:
    if closes.size < window + 1:
        return float("nan")
    tail = closes[-(window + 1) :]
    if not np.all(np.isfinite(tail)):
        return float("nan")
    net = abs(float(tail[-1] - tail[0]))
    path = float(np.sum(np.abs(np.diff(tail))))
    if path <= 0.0:
        return 0.0
    return float(net / path)


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _rolling_log_return_volatility_latest_numba(closes: np.ndarray, window: int) -> float:  # pragma: no cover
        n = closes.shape[0]
        if n < window + 1:
            return np.nan
        start = n - (window + 1)
        count = 0
        mean = 0.0
        for idx in range(start + 1, n):
            prev = closes[idx - 1]
            cur = closes[idx]
            if prev <= 0.0 or cur <= 0.0 or not np.isfinite(prev) or not np.isfinite(cur):
                return np.nan
            value = np.log(cur / prev)
            count += 1
            mean += (value - mean) / count

        if count < 2:
            return np.nan

        variance = 0.0
        for idx in range(start + 1, n):
            value = np.log(closes[idx] / closes[idx - 1])
            diff = value - mean
            variance += diff * diff
        return np.sqrt(variance / (count - 1))

    @njit(cache=True)
    def _normalized_true_range_latest_numba(  # pragma: no cover
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int,
    ) -> float:
        n = highs.shape[0]
        if lows.shape[0] < n:
            n = lows.shape[0]
        if closes.shape[0] < n:
            n = closes.shape[0]
        if n < window:
            return np.nan
        start = n - window
        accum = 0.0
        for idx in range(start, n):
            high = highs[idx]
            low = lows[idx]
            close = closes[idx]
            if not np.isfinite(high) or not np.isfinite(low) or not np.isfinite(close):
                return np.nan
            if close <= 0.0:
                return np.nan
            value = (high - low) / close
            if value < 0.0:
                return np.nan
            accum += value
        return accum / window

    @njit(cache=True)
    def _volume_shock_zscore_latest_numba(volumes: np.ndarray, window: int) -> float:  # pragma: no cover
        n = volumes.shape[0]
        if n < window + 1:
            return np.nan
        start = n - (window + 1)
        current = volumes[n - 1]
        if not np.isfinite(current):
            return np.nan
        mean = 0.0
        for idx in range(start, n - 1):
            value = volumes[idx]
            if not np.isfinite(value):
                return np.nan
            mean += value
        mean /= window

        var = 0.0
        for idx in range(start, n - 1):
            diff = volumes[idx] - mean
            var += diff * diff
        if window <= 1:
            return np.nan
        std = np.sqrt(var / (window - 1))
        if std <= 0.0:
            return 0.0
        return (current - mean) / std

    @njit(cache=True)
    def _trend_efficiency_latest_numba(closes: np.ndarray, window: int) -> float:  # pragma: no cover
        n = closes.shape[0]
        if n < window + 1:
            return np.nan
        start = n - (window + 1)
        first = closes[start]
        last = closes[n - 1]
        if not np.isfinite(first) or not np.isfinite(last):
            return np.nan
        net = abs(last - first)
        path = 0.0
        prev = first
        for idx in range(start + 1, n):
            cur = closes[idx]
            if not np.isfinite(cur):
                return np.nan
            path += abs(cur - prev)
            prev = cur
        if path <= 0.0:
            return 0.0
        return net / path

else:

    def _rolling_log_return_volatility_latest_numba(closes: np.ndarray, window: int) -> float:
        return _rolling_log_return_volatility_latest_python(closes, window)

    def _normalized_true_range_latest_numba(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int,
    ) -> float:
        return _normalized_true_range_latest_python(highs, lows, closes, window)

    def _volume_shock_zscore_latest_numba(volumes: np.ndarray, window: int) -> float:
        return _volume_shock_zscore_latest_python(volumes, window)

    def _trend_efficiency_latest_numba(closes: np.ndarray, window: int) -> float:
        return _trend_efficiency_latest_python(closes, window)


def rolling_log_return_volatility_latest(
    closes: Iterable[float],
    *,
    window: int = 64,
    annualization: float = 1.0,
) -> float | None:
    """Return latest rolling log-return volatility.

    Args:
        closes: Close-price series.
        window: Number of returns in the rolling window.
        annualization: Annualization multiplier. Use 1.0 to keep per-bar volatility.
    """
    window_i = max(2, int(window))
    arr = _to_np(closes)
    vol = _rolling_log_return_volatility_latest_numba(arr, window_i)
    out = _as_float_or_none(vol)
    if out is None:
        return None
    factor = max(0.0, float(annualization))
    if factor <= 0.0:
        return out
    return out * math.sqrt(factor)


def normalized_true_range_latest(
    highs: Iterable[float],
    lows: Iterable[float],
    closes: Iterable[float],
    *,
    window: int = 64,
) -> float | None:
    """Return latest mean normalized true range `(high-low)/close` over `window`."""
    window_i = max(2, int(window))
    out = _normalized_true_range_latest_numba(_to_np(highs), _to_np(lows), _to_np(closes), window_i)
    return _as_float_or_none(out)


def volume_shock_zscore_latest(volumes: Iterable[float], *, window: int = 64) -> float | None:
    """Return latest volume shock z-score vs the previous `window` bars."""
    window_i = max(2, int(window))
    out = _volume_shock_zscore_latest_numba(_to_np(volumes), window_i)
    return _as_float_or_none(out)


def trend_efficiency_latest(closes: Iterable[float], *, window: int = 64) -> float | None:
    """Return path efficiency ratio in `[0, 1]` for the latest `window` bars."""
    window_i = max(2, int(window))
    out = _trend_efficiency_latest_numba(_to_np(closes), window_i)
    parsed = _as_float_or_none(out)
    if parsed is None:
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


__all__ = [
    "NUMBA_AVAILABLE",
    "normalized_true_range_latest",
    "rolling_log_return_volatility_latest",
    "trend_efficiency_latest",
    "volume_shock_zscore_latest",
]
