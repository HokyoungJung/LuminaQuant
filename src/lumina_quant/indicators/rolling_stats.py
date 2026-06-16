"""Reusable rolling-stat primitives for strategy composition."""

from __future__ import annotations

import math
from collections import deque
from statistics import mean


def sample_std(values) -> float | None:
    """Return sample standard deviation (ddof=1) or ``None`` if invalid."""
    count = len(values)
    if count < 2:
        return None
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / float(count - 1)
    if variance <= 0.0:
        return None
    return math.sqrt(variance)


def rolling_beta(x_values, y_values) -> float | None:
    """Return beta of x relative to y over aligned rolling samples."""
    count = min(len(x_values), len(y_values))
    if count < 2:
        return None
    x_tail = list(x_values)[-count:]
    y_tail = list(y_values)[-count:]

    mean_x = mean(x_tail)
    mean_y = mean(y_tail)
    var_y = sum((value - mean_y) ** 2 for value in y_tail) / float(count - 1)
    if var_y <= 1e-12:
        return None

    cov_xy = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x_tail, y_tail, strict=False))
    cov_xy /= float(count - 1)
    beta = cov_xy / var_y
    if not math.isfinite(beta):
        return None
    return max(-10.0, min(10.0, beta))


def _aligned_xy(x_values, y_values) -> tuple[list[float], list[float], float, float, float] | None:
    """Return aligned ``(x_tail, y_tail, mean_x, mean_y, var_x)`` or ``None``.

    Mirrors the alignment/guard numerics of :func:`rolling_beta`: trims both
    series to their common trailing length, requires at least two finite-friendly
    samples, and rejects degenerate ``x`` (the regressor) whose variance is zero.
    """
    count = min(len(x_values), len(y_values))
    if count < 2:
        return None
    x_tail = [float(value) for value in list(x_values)[-count:]]
    y_tail = [float(value) for value in list(y_values)[-count:]]
    if not all(math.isfinite(value) for value in x_tail):
        return None
    if not all(math.isfinite(value) for value in y_tail):
        return None

    mean_x = mean(x_tail)
    mean_y = mean(y_tail)
    var_x = sum((value - mean_x) ** 2 for value in x_tail) / float(count - 1)
    if var_x <= 1e-12:
        return None
    return x_tail, y_tail, mean_x, mean_y, var_x


def ts_regression_slope(x_values, y_values) -> float | None:
    """Return OLS slope of ``y`` regressed on ``x`` over aligned samples."""
    aligned = _aligned_xy(x_values, y_values)
    if aligned is None:
        return None
    x_tail, y_tail, mean_x, mean_y, var_x = aligned
    count = len(x_tail)
    cov_xy = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x_tail, y_tail, strict=False))
    cov_xy /= float(count - 1)
    slope = cov_xy / var_x
    if not math.isfinite(slope):
        return None
    return slope


def ts_regression_intercept(x_values, y_values) -> float | None:
    """Return OLS intercept of ``y`` regressed on ``x`` over aligned samples."""
    aligned = _aligned_xy(x_values, y_values)
    if aligned is None:
        return None
    x_tail, y_tail, mean_x, mean_y, var_x = aligned
    count = len(x_tail)
    cov_xy = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x_tail, y_tail, strict=False))
    cov_xy /= float(count - 1)
    slope = cov_xy / var_x
    intercept = mean_y - (slope * mean_x)
    if not math.isfinite(intercept):
        return None
    return intercept


def ts_regression_rsquared(x_values, y_values) -> float | None:
    """Return OLS coefficient of determination (R^2) over aligned samples."""
    aligned = _aligned_xy(x_values, y_values)
    if aligned is None:
        return None
    x_tail, y_tail, mean_x, mean_y, var_x = aligned
    count = len(x_tail)
    syy = sum((value - mean_y) ** 2 for value in y_tail)
    if syy <= 1e-12:
        return None
    cov_xy = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x_tail, y_tail, strict=False))
    cov_xy /= float(count - 1)
    slope = cov_xy / var_x
    sse = sum(
        (yv - (mean_y + slope * (xv - mean_x))) ** 2 for xv, yv in zip(x_tail, y_tail, strict=False)
    )
    rsquared = 1.0 - (sse / syy)
    if not math.isfinite(rsquared):
        return None
    return max(0.0, min(1.0, rsquared))


def hurst_exponent(values, *, min_lag: int = 2, max_lag: int = 20) -> float | None:
    """Return the Hurst exponent via a rescaled-range (R/S) log-log fit.

    Returns ``None`` on empty/degenerate/non-finite input or when the lag range
    is too narrow to fit a slope.
    """
    series = [float(value) for value in values]
    if not series or not all(math.isfinite(value) for value in series):
        return None
    count = len(series)

    low_lag = max(2, int(min_lag))
    high_lag = int(max_lag)
    # The R/S statistic needs at least two points per lag window.
    high_lag = min(high_lag, count - 1)
    if high_lag <= low_lag:
        return None

    log_lags: list[float] = []
    log_rs: list[float] = []
    for lag in range(low_lag, high_lag + 1):
        diffs = [series[i] - series[i - lag] for i in range(lag, count)]
        if len(diffs) < 2:
            continue
        mean_diff = mean(diffs)
        deviations = []
        running = 0.0
        for value in diffs:
            running += value - mean_diff
            deviations.append(running)
        rng = max(deviations) - min(deviations)
        variance = sum((value - mean_diff) ** 2 for value in diffs) / float(len(diffs) - 1)
        if variance <= 1e-12:
            continue
        std_diff = math.sqrt(variance)
        rescaled = rng / std_diff
        if rescaled <= 1e-12 or lag <= 0:
            continue
        log_lags.append(math.log(float(lag)))
        log_rs.append(math.log(rescaled))

    if len(log_lags) < 2:
        return None
    slope = ts_regression_slope(log_lags, log_rs)
    if slope is None or not math.isfinite(slope):
        return None
    return max(0.0, min(1.0, slope))


def ewma_volatility(values, *, half_life: int = 10, annualization=None) -> float | None:
    """Return EWMA volatility of returns with ``lambda = exp(-ln2/half_life)``.

    ``values`` are treated as a return series. Returns ``None`` on empty or
    non-finite input or a degenerate half-life.
    """
    series = [float(value) for value in values]
    if not series or not all(math.isfinite(value) for value in series):
        return None
    half_life_val = float(half_life)
    if half_life_val <= 0.0 or not math.isfinite(half_life_val):
        return None

    decay = math.exp(-math.log(2.0) / half_life_val)
    if not math.isfinite(decay) or not (0.0 < decay < 1.0):
        return None

    ewma_var = series[0] * series[0]
    for value in series[1:]:
        ewma_var = (decay * ewma_var) + ((1.0 - decay) * value * value)
    if ewma_var < 0.0 or not math.isfinite(ewma_var):
        return None

    vol = math.sqrt(ewma_var)
    if annualization is not None:
        factor = float(annualization)
        if not math.isfinite(factor) or factor <= 0.0:
            return None
        vol *= math.sqrt(factor)
    if not math.isfinite(vol):
        return None
    return vol


def rolling_quantile(values, *, window: int, q: float) -> float | None:
    """Return the ``q`` quantile over the trailing ``window`` samples.

    Uses linear interpolation between order statistics. Returns ``None`` on
    empty/non-finite input, a non-positive window, an out-of-range ``q``, or when
    fewer than one sample is available.
    """
    window_val = int(window)
    if window_val < 1:
        return None
    q_val = float(q)
    if not math.isfinite(q_val) or q_val < 0.0 or q_val > 1.0:
        return None

    tail = [float(value) for value in list(values)[-window_val:]]
    if not tail or not all(math.isfinite(value) for value in tail):
        return None

    ordered = sorted(tail)
    count = len(ordered)
    if count == 1:
        return ordered[0]

    position = q_val * (count - 1)
    lower_idx = math.floor(position)
    upper_idx = math.ceil(position)
    if lower_idx == upper_idx:
        result = ordered[lower_idx]
    else:
        frac = position - lower_idx
        result = ordered[lower_idx] + frac * (ordered[upper_idx] - ordered[lower_idx])
    if not math.isfinite(result):
        return None
    return result


def rolling_corr(x_values, y_values) -> float | None:
    """Return Pearson correlation over aligned rolling samples."""
    count = min(len(x_values), len(y_values))
    if count < 2:
        return None
    x_tail = list(x_values)[-count:]
    y_tail = list(y_values)[-count:]

    mean_x = mean(x_tail)
    mean_y = mean(y_tail)
    sxx = sum((value - mean_x) ** 2 for value in x_tail)
    syy = sum((value - mean_y) ** 2 for value in y_tail)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None

    sxy = sum((xv - mean_x) * (yv - mean_y) for xv, yv in zip(x_tail, y_tail, strict=False))
    corr = sxy / math.sqrt(sxx * syy)
    if not math.isfinite(corr):
        return None
    return max(-1.0, min(1.0, corr))


def recursive_least_squares_beta_update(
    beta: float,
    covariance: float,
    *,
    x_value: float,
    y_value: float,
    forgetting_factor: float = 0.985,
    covariance_floor: float = 1e-6,
    covariance_cap: float = 1e6,
) -> tuple[float, float] | None:
    """Update a scalar hedge ratio with an O(1) recursive least-squares step."""
    x_val = float(x_value)
    y_val = float(y_value)
    if not math.isfinite(x_val) or not math.isfinite(y_val) or abs(y_val) <= 1e-12:
        return None

    beta_val = float(beta)
    cov_val = max(float(covariance_floor), min(float(covariance_cap), float(covariance)))
    lambda_val = max(1e-6, min(0.999999, float(forgetting_factor)))

    gain_num = cov_val * y_val
    denom = lambda_val + (y_val * gain_num)
    if not math.isfinite(denom) or denom <= 1e-12:
        return None

    gain = gain_num / denom
    error = x_val - (beta_val * y_val)
    updated_beta = beta_val + (gain * error)
    updated_cov = (cov_val - (gain * y_val * cov_val)) / lambda_val
    if not math.isfinite(updated_beta) or not math.isfinite(updated_cov):
        return None
    updated_cov = max(float(covariance_floor), min(float(covariance_cap), float(updated_cov)))
    return float(updated_beta), float(updated_cov)


class RollingZScoreWindow:
    """Rolling z-score helper preserving O(1) aggregate updates."""

    def __init__(self, window: int):
        self.window = max(2, int(window))
        self.values = deque(maxlen=self.window)
        self.sum_value = 0.0
        self.sum_squares = 0.0

    def append(self, value: float) -> None:
        parsed = float(value)
        if len(self.values) == self.window:
            dropped = float(self.values[0])
            self.sum_value -= dropped
            self.sum_squares -= dropped * dropped
        self.values.append(parsed)
        self.sum_value += parsed
        self.sum_squares += parsed * parsed

    def zscore(self, value: float) -> float | None:
        count = len(self.values)
        if count < self.window:
            return None
        count_f = float(count)
        mean_value = self.sum_value / count_f
        variance = (self.sum_squares / count_f) - (mean_value * mean_value)
        if variance <= 1e-12:
            return None
        std_value = math.sqrt(variance)
        return (float(value) - mean_value) / std_value

    def to_state(self) -> dict:
        return {
            "values": list(self.values),
            "sum_value": float(self.sum_value),
            "sum_squares": float(self.sum_squares),
        }

    def load_state(self, values) -> None:
        self.values = deque(maxlen=self.window)
        self.sum_value = 0.0
        self.sum_squares = 0.0
        for value in values:
            self.append(float(value))
