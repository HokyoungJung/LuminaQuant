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
