"""Numba-accelerated metric kernels used by optimization loops."""

from __future__ import annotations

import math

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


@njit(cache=True)
def _max_drawdown_numba(total_series: np.ndarray) -> float:
    peak = total_series[0]
    max_dd = 0.0
    for i in range(total_series.shape[0]):
        value = total_series[i]
        if value > peak:
            peak = value
        if peak > 0.0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


@njit(cache=True)
def evaluate_metrics_numba(
    total_series: np.ndarray, annual_periods: int
) -> tuple[float, float, float]:
    """Single-call evaluation kernel returning sharpe/cagr/max_drawdown."""
    n = total_series.shape[0]
    if n < 2:
        return -999.0, 0.0, 0.0

    returns = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        prev = total_series[i]
        nxt = total_series[i + 1]
        den = 1.0 if prev == 0.0 else prev
        returns[i] = (nxt - prev) / den

    mean_r = 0.0
    for i in range(returns.shape[0]):
        mean_r += returns[i]
    mean_r /= max(1, returns.shape[0])

    var_r = 0.0
    for i in range(returns.shape[0]):
        d = returns[i] - mean_r
        var_r += d * d
    if returns.shape[0] > 1:
        var_r /= returns.shape[0] - 1
    std_r = math.sqrt(var_r) if var_r > 0.0 else 0.0

    sharpe = -999.0
    if std_r > 0.0:
        sharpe = (mean_r / std_r) * math.sqrt(float(max(1, annual_periods)))

    initial = total_series[0]
    final = total_series[-1]
    if initial <= 0.0:
        cagr = 0.0
    else:
        years = float(n) / float(max(1, annual_periods))
        if years <= 0.0:
            cagr = 0.0
        else:
            cagr = (final / initial) ** (1.0 / years) - 1.0

    max_dd = _max_drawdown_numba(total_series)
    return sharpe, cagr, max_dd
