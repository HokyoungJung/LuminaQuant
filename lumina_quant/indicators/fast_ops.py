"""Fast operator kernels for strategy-factory feature engineering."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        _ = (args, kwargs)

        def _decorator(fn):
            return fn

        return _decorator


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


@njit(cache=True)
def _rolling_zscore_kernel(values: np.ndarray, window: int, min_std: float) -> np.ndarray:
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2:
        return out

    for idx in range(window - 1, n):
        start = idx - window + 1
        valid = True
        mean = 0.0
        for j in range(start, idx + 1):
            value = values[j]
            if not np.isfinite(value):
                valid = False
                break
            mean += value
        if not valid:
            continue
        mean /= float(window)

        var = 0.0
        for j in range(start, idx + 1):
            delta = values[j] - mean
            var += delta * delta
        var /= float(max(1, window - 1))
        std = math.sqrt(var)
        if std <= min_std:
            continue
        out[idx] = (values[idx] - mean) / std

    return out


@njit(cache=True)
def _rolling_percentile_rank_kernel(values: np.ndarray, window: int) -> np.ndarray:
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2:
        return out

    for idx in range(window - 1, n):
        start = idx - window + 1
        current = values[idx]
        if not np.isfinite(current):
            continue

        below_or_equal = 0
        valid = 0
        for j in range(start, idx + 1):
            value = values[j]
            if not np.isfinite(value):
                continue
            valid += 1
            if value <= current:
                below_or_equal += 1

        if valid < 2:
            continue
        out[idx] = (below_or_equal - 1) / float(valid - 1)

    return out


def rolling_zscore(values: Iterable[float], *, window: int = 64, min_std: float = 1e-12) -> np.ndarray:
    """Return rolling z-score series using a Numba-friendly kernel."""
    arr = _as_float_array(values)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    return _rolling_zscore_kernel(arr, int(max(2, window)), float(min_std))


def rolling_percentile_rank(values: Iterable[float], *, window: int = 64) -> np.ndarray:
    """Return rolling percentile-rank series for the latest element in each window."""
    arr = _as_float_array(values)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    return _rolling_percentile_rank_kernel(arr, int(max(2, window)))


def rolling_zscore_latest(
    values: Iterable[float],
    *,
    window: int = 64,
    min_std: float = 1e-12,
) -> float | None:
    """Return latest finite z-score from rolling_zscore."""
    series = rolling_zscore(values, window=window, min_std=min_std)
    if series.size == 0:
        return None
    value = float(series[-1])
    return value if math.isfinite(value) else None


def rolling_percentile_rank_latest(values: Iterable[float], *, window: int = 64) -> float | None:
    """Return latest finite percentile rank from rolling_percentile_rank."""
    series = rolling_percentile_rank(values, window=window)
    if series.size == 0:
        return None
    value = float(series[-1])
    return value if math.isfinite(value) else None


def cross_sectional_rank(values: Iterable[float]) -> np.ndarray:
    """Return normalized cross-sectional rank in [0,1] while preserving NaNs."""
    arr = _as_float_array(values)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return out

    finite_idx = np.where(finite_mask)[0]
    finite_vals = arr[finite_mask]
    order = np.argsort(finite_vals)

    if finite_vals.size == 1:
        out[finite_idx[0]] = 1.0
        return out

    for rank, ord_idx in enumerate(order):
        original_idx = finite_idx[int(ord_idx)]
        out[original_idx] = rank / float(finite_vals.size - 1)
    return out


def volatility_adjusted_momentum(
    closes: Iterable[float],
    *,
    lookback: int = 32,
    volatility_window: int = 64,
    epsilon: float = 1e-12,
) -> float | None:
    """Return momentum normalized by realized volatility over recent returns."""
    arr = _as_float_array(closes)
    lb = int(max(2, lookback))
    vw = int(max(2, volatility_window))
    if arr.size < max(lb + 1, vw + 1):
        return None

    if np.any(arr <= 0.0):
        return None

    recent = arr[-(lb + 1) :]
    momentum = float(math.log(recent[-1] / recent[0]))

    returns = np.log(arr[1:] / arr[:-1])
    vol_tail = returns[-vw:]
    if vol_tail.size < 2 or not np.all(np.isfinite(vol_tail)):
        return None

    vol = float(np.std(vol_tail, ddof=1))
    if vol <= float(epsilon):
        return None

    score = momentum / vol
    return score if math.isfinite(score) else None


__all__ = [
    "NUMBA_AVAILABLE",
    "cross_sectional_rank",
    "rolling_percentile_rank",
    "rolling_percentile_rank_latest",
    "rolling_zscore",
    "rolling_zscore_latest",
    "volatility_adjusted_momentum",
]
