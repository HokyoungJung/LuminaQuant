"""Centralized numeric/rolling compute ops shared by indicators and strategies."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd


def _tail(values, window: int) -> list[float] | None:
    window_i = max(1, int(window))
    seq = [float(value) for value in list(values)]
    if len(seq) < window_i:
        return None
    return seq[-window_i:]


def delay(values, periods: int = 1) -> float | None:
    period_i = max(1, int(periods))
    seq = [float(value) for value in list(values)]
    if len(seq) <= period_i:
        return None
    return seq[-1 - period_i]


def delta(values, periods: int = 1) -> float | None:
    delayed = delay(values, periods)
    if delayed is None:
        return None
    latest = float(list(values)[-1])
    return latest - delayed


def ts_sum(values, window: int) -> float | None:
    tail = _tail(values, window)
    if tail is None:
        return None
    return float(sum(tail))


def ts_std(values, window: int) -> float | None:
    tail = _tail(values, max(2, int(window)))
    if tail is None:
        return None
    arr = np.asarray(tail, dtype=float)
    if arr.size < 2:
        return None
    out = float(np.std(arr, ddof=1))
    return out if math.isfinite(out) else None


def ts_rank(values, window: int = 20) -> float | None:
    tail = _tail(values, max(2, int(window)))
    if tail is None:
        return None
    latest = tail[-1]
    below = sum(1 for value in tail if value < latest)
    equal = sum(1 for value in tail if value == latest)
    return (below + 0.5 * equal) / float(len(tail))


def rolling_rank(values, window: int = 20) -> float | None:
    return ts_rank(values, window=window)


def ts_corr(left_values, right_values, window: int) -> float | None:
    window_i = max(2, int(window))
    n = min(len(left_values), len(right_values))
    if n < window_i:
        return None
    x = np.asarray(list(left_values)[-window_i:], dtype=float)
    y = np.asarray(list(right_values)[-window_i:], dtype=float)
    if x.size < 2 or y.size < 2:
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(corr):
        return None
    return max(-1.0, min(1.0, corr))


def ts_cov(left_values, right_values, window: int) -> float | None:
    window_i = max(2, int(window))
    n = min(len(left_values), len(right_values))
    if n < window_i:
        return None
    x = np.asarray(list(left_values)[-window_i:], dtype=float)
    y = np.asarray(list(right_values)[-window_i:], dtype=float)
    if x.size < 2 or y.size < 2:
        return None
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    return cov if math.isfinite(cov) else None


def signed_power(value: float, power: float) -> float:
    value_f = float(value)
    power_f = float(power)
    if value_f == 0.0:
        return 0.0
    return math.copysign(abs(value_f) ** power_f, value_f)


def decay_linear(values, window: int) -> float | None:
    tail = _tail(values, max(1, int(window)))
    if tail is None:
        return None
    weights = np.arange(1, len(tail) + 1, dtype=float)
    denom = float(weights.sum())
    if denom <= 1e-12:
        return None
    numer = float(np.dot(np.asarray(tail, dtype=float), weights))
    out = numer / denom
    return out if math.isfinite(out) else None


def where(condition: Any, left: Any, right: Any) -> Any:
    if condition is None:
        return right
    if isinstance(condition, (float, np.floating)) and math.isnan(float(condition)):
        return right
    return left if bool(condition) else right


def clip(value: float, lower: float, upper: float) -> float:
    low = float(lower)
    high = float(upper)
    if low > high:
        low, high = high, low
    return float(np.clip(float(value), low, high))


def returns_from_close(closes) -> list[float]:
    closes_f = [float(value) for value in closes]
    out: list[float] = []
    for prev, curr in pairwise(closes_f):
        if abs(prev) <= 1e-12:
            out.append(0.0)
        else:
            out.append((curr / prev) - 1.0)
    return out


def adv(closes, volumes, window: int = 20) -> float | None:
    window_i = max(1, int(window))
    n = min(len(closes), len(volumes))
    if n < window_i:
        return None
    closes_tail = [float(value) for value in list(closes)[-window_i:]]
    volumes_tail = [max(0.0, float(value)) for value in list(volumes)[-window_i:]]
    dollars = [close * volume for close, volume in zip(closes_tail, volumes_tail, strict=False)]
    return float(sum(dollars) / float(window_i))


def rolling_rank_series(series: pd.Series, window: int = 20) -> pd.Series:
    w = max(2, int(window))
    return series.rolling(w).apply(lambda a: pd.Series(a).rank(pct=True).iloc[-1], raw=False)


def ts_sum_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(max(1, int(window))).sum()


def ts_std_series(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(max(2, int(window))).std()


def ts_corr_series(left_series: pd.Series, right_series: pd.Series, window: int) -> pd.Series:
    return left_series.rolling(max(2, int(window))).corr(right_series)


def ts_cov_series(left_series: pd.Series, right_series: pd.Series, window: int) -> pd.Series:
    return left_series.rolling(max(2, int(window))).cov(right_series)


def decay_linear_series(series: pd.Series, window: int) -> pd.Series:
    w = max(1, int(window))
    weights = np.arange(1, w + 1, dtype=float)
    denom = float(weights.sum())
    return series.rolling(w).apply(lambda a: float(np.dot(a, weights) / denom), raw=True)


def where_series(cond: pd.Series, left: pd.Series, right: pd.Series) -> pd.Series:
    cond_clean = cond.where(cond.notna(), 0.0).astype(bool)
    return pd.Series(np.where(cond_clean, left, right), index=left.index, dtype=float)


def clip_series(values: pd.Series, lower: float, upper: float) -> pd.Series:
    low = float(lower)
    high = float(upper)
    if low > high:
        low, high = high, low
    return values.clip(lower=low, upper=high)


def adv_series(closes: pd.Series, volumes: pd.Series, window: int = 20) -> pd.Series:
    w = max(1, int(window))
    vol = volumes.clip(lower=0.0)
    dollars = closes * vol
    return dollars.rolling(w).mean()

