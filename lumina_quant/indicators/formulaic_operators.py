"""Formulaic alpha operator primitives with tunable parameters."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from lumina_quant.compute import ops as compute_ops


def _tail(values, window: int) -> list[float] | None:
    window_i = max(1, int(window))
    if len(values) < window_i:
        return None
    return [float(value) for value in list(values)[-window_i:]]


def delay(values, periods: int = 1) -> float | None:
    """Return value ``periods`` bars ago."""
    return compute_ops.delay(values, periods=periods)


def delta(values, periods: int = 1) -> float | None:
    """Return latest minus delayed value."""
    return compute_ops.delta(values, periods=periods)


def ts_sum(values, window: int) -> float | None:
    """Return rolling sum over trailing window."""
    return compute_ops.ts_sum(values, window=window)


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
    return compute_ops.ts_std(values, window=window)


def rank_pct(values, window: int = 20) -> float | None:
    """Return percentile rank of latest value in trailing window."""
    return compute_ops.rolling_rank(values, window=max(2, int(window)))


def ts_rank(values, window: int = 20) -> float | None:
    """Alias for trailing percentile rank."""
    return compute_ops.ts_rank(values, window=window)


def signed_power(value: float, power: float) -> float:
    """Return signed power ``sign(x) * |x|**p``."""
    return compute_ops.signed_power(value, power)


def decay_linear(values, window: int) -> float | None:
    """Return linearly decayed weighted average over trailing window."""
    return compute_ops.decay_linear(values, window=window)


def ts_correlation(x_values, y_values, window: int) -> float | None:
    """Return trailing correlation over aligned windows."""
    return compute_ops.ts_corr(x_values, y_values, window=window)


def ts_covariance(x_values, y_values, window: int) -> float | None:
    """Return trailing sample covariance over aligned windows."""
    return compute_ops.ts_cov(x_values, y_values, window=window)


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
    return compute_ops.returns_from_close(closes)


def adv(closes, volumes, window: int = 20) -> float | None:
    """Return average daily dollar volume over trailing window."""
    return compute_ops.adv(closes, volumes, window=window)


def _last_finite_value(series: pd.Series) -> float | None:
    for value in reversed(series.to_list()):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
    return None


def to_series(value, index: pd.Index) -> pd.Series:
    """Coerce scalar/sequence/Series into float Series aligned to ``index``."""
    if isinstance(value, pd.Series):
        return value.astype(float)
    if np.isscalar(value):
        try:
            scalar_arr = np.asarray(value, dtype=float)
            scalar = float(scalar_arr.reshape(-1)[0]) if scalar_arr.size > 0 else float("nan")
        except (TypeError, ValueError):
            scalar = float("nan")
        return pd.Series(scalar, index=index, dtype=float)
    arr = np.asarray(list(value), dtype=float)
    n = min(len(index), arr.size)
    out = np.full(len(index), np.nan, dtype=float)
    if n > 0:
        out[-n:] = arr[-n:]
    return pd.Series(out, index=index, dtype=float)


def as_window(value) -> int:
    """Return safe positive integer window from scalar/Series input."""
    if isinstance(value, pd.Series):
        value = _last_finite_value(value.dropna())
    if value is None:
        return 1
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return 1


def rank_series(values, *, index: pd.Index, window: int = 20) -> pd.Series:
    series = to_series(values, index)
    return compute_ops.rolling_rank_series(series, window=max(2, int(window)))


def ts_rank_series(values, window, *, index: pd.Index) -> pd.Series:
    return rank_series(values, index=index, window=as_window(window))


def ts_sum_series(values, window, *, index: pd.Index) -> pd.Series:
    return compute_ops.ts_sum_series(to_series(values, index), window=as_window(window))


def ts_stddev_series(values, window, *, index: pd.Index) -> pd.Series:
    return compute_ops.ts_std_series(to_series(values, index), window=max(2, as_window(window)))


def ts_corr_series(left, right, window, *, index: pd.Index) -> pd.Series:
    left_series = to_series(left, index)
    right_series = to_series(right, index)
    return compute_ops.ts_corr_series(
        left_series,
        right_series,
        window=max(2, as_window(window)),
    )


def ts_cov_series(left, right, window, *, index: pd.Index) -> pd.Series:
    left_series = to_series(left, index)
    right_series = to_series(right, index)
    return compute_ops.ts_cov_series(
        left_series,
        right_series,
        window=max(2, as_window(window)),
    )


def ts_min_series(values, window, *, index: pd.Index) -> pd.Series:
    return to_series(values, index).rolling(as_window(window)).min()


def ts_max_series(values, window, *, index: pd.Index) -> pd.Series:
    return to_series(values, index).rolling(as_window(window)).max()


def ts_product_series(values, window, *, index: pd.Index) -> pd.Series:
    return to_series(values, index).rolling(as_window(window)).apply(np.prod, raw=True)


def delay_series(values, period=1, *, index: pd.Index) -> pd.Series:
    return to_series(values, index).shift(as_window(period))


def delta_series(values, period=1, *, index: pd.Index) -> pd.Series:
    series = to_series(values, index)
    return series - series.shift(as_window(period))


def ts_argmax_series(values, window, *, index: pd.Index) -> pd.Series:
    series = to_series(values, index)
    return series.rolling(as_window(window)).apply(lambda a: float(np.argmax(a) + 1), raw=True)


def ts_argmin_series(values, window, *, index: pd.Index) -> pd.Series:
    series = to_series(values, index)
    return series.rolling(as_window(window)).apply(lambda a: float(np.argmin(a) + 1), raw=True)


def decay_linear_series(values, period=10, *, index: pd.Index) -> pd.Series:
    return compute_ops.decay_linear_series(to_series(values, index), window=as_window(period))


def scale_series(values, *, rank_window: int, index: pd.Index, a=1.0) -> pd.Series:
    series = to_series(values, index)
    denom = series.abs().rolling(max(2, int(rank_window))).sum()
    denom_arr = np.asarray(denom, dtype=float)
    denom_arr = np.where(denom_arr == 0.0, np.nan, denom_arr)
    return (float(a) * series).divide(pd.Series(denom_arr, index=index, dtype=float))


def signed_power_series(values, power, *, index: pd.Index) -> pd.Series:
    series = to_series(values, index)
    power_series = to_series(power, index)
    return np.sign(series) * np.power(np.abs(series), power_series)


def where_series(cond, left, right, *, index: pd.Index) -> pd.Series:
    cond_series = to_series(cond, index)
    left_series = to_series(left, index)
    right_series = to_series(right, index)
    return compute_ops.where_series(cond_series, left_series, right_series)


def indneutralize_series(values, group, *, index: pd.Index) -> pd.Series:
    series = to_series(values, index)
    group_series = to_series(group, index)
    frame = pd.DataFrame({"s": series, "g": group_series})
    centered = frame["s"] - frame.groupby("g")["s"].transform("mean")
    return centered.fillna(series)
