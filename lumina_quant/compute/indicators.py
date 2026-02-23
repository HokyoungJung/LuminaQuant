"""Indicator helpers for CPU execution."""

from __future__ import annotations

import numpy as np
import talib


def _to_numpy(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def compute_sma(values, period: int, backend: str = "cpu") -> np.ndarray:
    """Compute SMA with TA-Lib.

    The `backend` argument is kept for backward compatibility.
    """
    series = _to_numpy(values)
    _ = backend
    return talib.SMA(series, timeperiod=period)


def compute_rsi(values, period: int, backend: str = "cpu") -> np.ndarray:
    """Compute RSI with TA-Lib.

    The `backend` argument is kept for backward compatibility.
    """
    series = _to_numpy(values)
    _ = backend
    return talib.RSI(series, timeperiod=period)
