"""Alpha-zoo operator primitives for deterministic crypto/FX factor screens.

The operators intentionally mirror the Alpha191/Alpha158-style vocabulary
(rank, delay, delta, rolling moments, correlation, decay) without importing a
strategy or calendar concept.  They operate on pandas Series and accept an
optional grouping key, normally a symbol column, so research scripts can build
causal factor panels without lookahead.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd


def as_float_series(
    values: pd.Series | Sequence[float], *, index: pd.Index | None = None
) -> pd.Series:
    """Return a float Series, preserving ``index`` when supplied."""
    if isinstance(values, pd.Series):
        series = values.copy()
        if index is not None:
            series = series.reindex(index)
        return pd.to_numeric(series, errors="coerce").astype(float)
    return pd.Series(list(values), index=index, dtype=float)


def _groupby(values: pd.Series, by: pd.Series | Sequence[Hashable] | None):
    if by is None:
        return None
    return values.groupby(by, sort=False, group_keys=False)


def delay(
    values: pd.Series, periods: int = 1, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    """Causal delay/lag operator."""
    series = as_float_series(values)
    period_i = max(1, int(periods))
    grouped = _groupby(series, by)
    return series.shift(period_i) if grouped is None else grouped.shift(period_i)


def delta(
    values: pd.Series, periods: int = 1, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    """Latest value minus delayed value."""
    series = as_float_series(values)
    return series - delay(series, periods=periods, by=by)


def returns(
    values: pd.Series, periods: int = 1, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    """Simple return over ``periods`` bars."""
    series = as_float_series(values)
    lagged = delay(series, periods=periods, by=by)
    denom = lagged.where(lagged.abs() > 1e-12)
    return (series / denom) - 1.0


def rolling_sum(
    values: pd.Series, window: int, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    series = as_float_series(values)
    w = max(1, int(window))
    grouped = _groupby(series, by)
    if grouped is None:
        return series.rolling(w).sum()
    return grouped.rolling(w).sum().reset_index(level=0, drop=True)


def rolling_mean(
    values: pd.Series, window: int, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    series = as_float_series(values)
    w = max(1, int(window))
    grouped = _groupby(series, by)
    if grouped is None:
        return series.rolling(w).mean()
    return grouped.rolling(w).mean().reset_index(level=0, drop=True)


def rolling_std(
    values: pd.Series, window: int, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    series = as_float_series(values)
    w = max(2, int(window))
    grouped = _groupby(series, by)
    if grouped is None:
        return series.rolling(w).std()
    return grouped.rolling(w).std().reset_index(level=0, drop=True)


def rolling_min(
    values: pd.Series, window: int, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    series = as_float_series(values)
    w = max(1, int(window))
    grouped = _groupby(series, by)
    if grouped is None:
        return series.rolling(w).min()
    return grouped.rolling(w).min().reset_index(level=0, drop=True)


def rolling_max(
    values: pd.Series, window: int, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    series = as_float_series(values)
    w = max(1, int(window))
    grouped = _groupby(series, by)
    if grouped is None:
        return series.rolling(w).max()
    return grouped.rolling(w).max().reset_index(level=0, drop=True)


def rank(values: pd.Series, *, by: pd.Series | Sequence[Hashable] | None = None) -> pd.Series:
    """Percentile rank, cross-sectional when ``by`` is a timestamp key."""
    series = as_float_series(values)
    if by is None:
        return series.rank(pct=True)
    return series.groupby(by, sort=False).rank(pct=True)


def _latest_rank(window_values: np.ndarray) -> float:
    clean = window_values[np.isfinite(window_values)]
    if clean.size != window_values.size or clean.size == 0:
        return float("nan")
    latest = clean[-1]
    below = float(np.sum(clean < latest))
    equal = float(np.sum(clean == latest))
    return (below + 0.5 * equal) / float(clean.size)


def ts_rank(
    values: pd.Series, window: int, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    """Trailing percentile rank of the latest value."""
    series = as_float_series(values)
    w = max(2, int(window))
    grouped = _groupby(series, by)
    if grouped is None:
        return series.rolling(w).apply(_latest_rank, raw=True)
    return grouped.rolling(w).apply(_latest_rank, raw=True).reset_index(level=0, drop=True)


def corr(
    left: pd.Series,
    right: pd.Series,
    window: int,
    *,
    by: pd.Series | Sequence[Hashable] | None = None,
) -> pd.Series:
    """Trailing correlation over aligned windows."""
    left_s = as_float_series(left)
    right_s = as_float_series(right, index=left_s.index)
    w = max(2, int(window))
    if by is None:
        return left_s.rolling(w).corr(right_s)
    pieces: list[pd.Series] = []
    by_s = (
        pd.Series(list(by), index=left_s.index)
        if not isinstance(by, pd.Series)
        else by.reindex(left_s.index)
    )
    for idx in by_s.groupby(by_s, sort=False).groups.values():
        left_g = left_s.loc[idx]
        right_g = right_s.loc[idx]
        pieces.append(left_g.rolling(w).corr(right_g))
    if not pieces:
        return pd.Series(np.nan, index=left_s.index, dtype=float)
    return pd.concat(pieces).sort_index()


def cov(
    left: pd.Series,
    right: pd.Series,
    window: int,
    *,
    by: pd.Series | Sequence[Hashable] | None = None,
) -> pd.Series:
    """Trailing covariance over aligned windows."""
    left_s = as_float_series(left)
    right_s = as_float_series(right, index=left_s.index)
    w = max(2, int(window))
    if by is None:
        return left_s.rolling(w).cov(right_s)
    pieces: list[pd.Series] = []
    by_s = (
        pd.Series(list(by), index=left_s.index)
        if not isinstance(by, pd.Series)
        else by.reindex(left_s.index)
    )
    for idx in by_s.groupby(by_s, sort=False).groups.values():
        left_g = left_s.loc[idx]
        right_g = right_s.loc[idx]
        pieces.append(left_g.rolling(w).cov(right_g))
    if not pieces:
        return pd.Series(np.nan, index=left_s.index, dtype=float)
    return pd.concat(pieces).sort_index()


def decay_linear(
    values: pd.Series, window: int, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    """Linearly decayed trailing mean with greatest weight on the latest bar."""
    series = as_float_series(values)
    w = max(1, int(window))
    weights = np.arange(1, w + 1, dtype=float)
    denom = float(weights.sum())

    def _apply(arr: np.ndarray) -> float:
        if arr.size != w or not np.isfinite(arr).all():
            return float("nan")
        return float(np.dot(arr, weights) / denom)

    grouped = _groupby(series, by)
    if grouped is None:
        return series.rolling(w).apply(_apply, raw=True)
    return grouped.rolling(w).apply(_apply, raw=True).reset_index(level=0, drop=True)


def signed_power(values: pd.Series | float, power: float) -> pd.Series | float:
    """Return sign(x) * abs(x)**power."""
    if isinstance(values, pd.Series):
        series = as_float_series(values)
        return np.sign(series) * np.power(series.abs(), float(power))
    value = float(values)
    if not math.isfinite(value):
        return float("nan")
    return math.copysign(abs(value) ** float(power), value)


def zscore(
    values: pd.Series, window: int, *, by: pd.Series | Sequence[Hashable] | None = None
) -> pd.Series:
    """Rolling z-score using only trailing observations."""
    series = as_float_series(values)
    mu = rolling_mean(series, window, by=by)
    sigma = rolling_std(series, window, by=by)
    return (series - mu) / sigma.where(sigma.abs() > 1e-12)


def safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    """Divide while converting zero denominators to NaN."""
    num = as_float_series(numerator)
    den = (
        denominator
        if isinstance(denominator, pd.Series)
        else pd.Series(float(denominator), index=num.index)
    )
    den_s = as_float_series(den, index=num.index)
    return num / den_s.where(den_s.abs() > 1e-12)
