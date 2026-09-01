"""Rolling moving-average primitives and tunable MA helpers."""

from __future__ import annotations

import math
from collections import deque


def _tail(values, window: int) -> list[float] | None:
    window_i = max(1, int(window))
    if len(values) < window_i:
        return None
    return [float(value) for value in list(values)[-window_i:]]


def simple_moving_average(values, window: int) -> float | None:
    """Return latest simple moving average over ``window`` samples."""
    tail = _tail(values, window)
    if tail is None:
        return None
    return sum(tail) / float(len(tail))


def weighted_moving_average(values, window: int) -> float | None:
    """Return latest linearly weighted moving average."""
    tail = _tail(values, window)
    if tail is None:
        return None
    n = len(tail)
    denom = float(n * (n + 1) // 2)
    weighted_sum = 0.0
    for index, value in enumerate(tail, start=1):
        weighted_sum += value * float(index)
    return weighted_sum / denom


def exponential_moving_average_series(
    values,
    window: int,
    *,
    smoothing: float = 2.0,
    seed_with_sma: bool = True,
) -> list[float] | None:
    """Return EMA series for all points after warmup."""
    values_list = [float(value) for value in values]
    window_i = max(1, int(window))
    if len(values_list) < window_i:
        return None

    alpha = float(smoothing) / float(window_i + 1)
    if not math.isfinite(alpha) or alpha <= 0.0:
        return None
    if alpha > 1.0:
        alpha = 1.0

    out: list[float] = []
    if seed_with_sma:
        ema = sum(values_list[:window_i]) / float(window_i)
        out.append(ema)
        start_index = window_i
    else:
        ema = values_list[0]
        out.append(ema)
        start_index = 1

    for value in values_list[start_index:]:
        ema = alpha * value + (1.0 - alpha) * ema
        out.append(ema)
    return out


def exponential_moving_average(
    values,
    window: int,
    *,
    smoothing: float = 2.0,
    seed_with_sma: bool = True,
) -> float | None:
    """Return latest EMA value."""
    series = exponential_moving_average_series(
        values,
        window,
        smoothing=smoothing,
        seed_with_sma=seed_with_sma,
    )
    if not series:
        return None
    return float(series[-1])


def double_exponential_moving_average(
    values,
    window: int,
    *,
    smoothing: float = 2.0,
) -> float | None:
    """Return latest DEMA value."""
    ema1 = exponential_moving_average_series(values, window, smoothing=smoothing)
    if not ema1:
        return None
    ema2 = exponential_moving_average_series(ema1, window, smoothing=smoothing)
    if not ema2:
        return None
    return 2.0 * float(ema1[-1]) - float(ema2[-1])


def triple_exponential_moving_average(
    values,
    window: int,
    *,
    smoothing: float = 2.0,
) -> float | None:
    """Return latest TEMA value."""
    ema1 = exponential_moving_average_series(values, window, smoothing=smoothing)
    if not ema1:
        return None
    ema2 = exponential_moving_average_series(ema1, window, smoothing=smoothing)
    if not ema2:
        return None
    ema3 = exponential_moving_average_series(ema2, window, smoothing=smoothing)
    if not ema3:
        return None
    return 3.0 * float(ema1[-1]) - 3.0 * float(ema2[-1]) + float(ema3[-1])


def hull_moving_average(values, window: int) -> float | None:
    """Return latest HMA value using WMA-based construction."""
    window_i = max(1, int(window))
    if len(values) < window_i:
        return None

    half_window = max(1, window_i // 2)
    sqrt_window = max(1, int(math.sqrt(window_i)))
    values_list = [float(value) for value in values]

    transformed: list[float] = []
    for idx in range(window_i - 1, len(values_list)):
        slice_values = values_list[: idx + 1]
        wma_half = weighted_moving_average(slice_values, half_window)
        wma_full = weighted_moving_average(slice_values, window_i)
        if wma_half is None or wma_full is None:
            continue
        transformed.append(2.0 * wma_half - wma_full)

    if len(transformed) < sqrt_window:
        return None
    return weighted_moving_average(transformed, sqrt_window)


class RollingMeanWindow:
    """O(1) rolling mean with explicit state serialization support."""

    def __init__(self, window: int):
        self.window = max(1, int(window))
        self.values: deque[float] = deque(maxlen=self.window)
        self.sum_value = 0.0

    def append(self, value: float) -> None:
        parsed = float(value)
        if len(self.values) == self.window:
            self.sum_value -= float(self.values[0])
        self.values.append(parsed)
        self.sum_value += parsed

    def mean(self) -> float | None:
        if len(self.values) < self.window:
            return None
        return self.sum_value / float(self.window)

    def to_state(self) -> dict:
        return {
            "values": list(self.values),
            "sum_value": float(self.sum_value),
        }

    def load_state(self, values) -> None:
        self.values = deque(maxlen=self.window)
        self.sum_value = 0.0
        for value in values:
            self.append(float(value))
