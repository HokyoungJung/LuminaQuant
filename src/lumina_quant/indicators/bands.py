"""Price-band and channel indicators with tunable parameters."""

from __future__ import annotations

from .atr import average_true_range, true_range
from .moving_average import exponential_moving_average, simple_moving_average
from .rolling_stats import sample_std


def bollinger_bands(
    values,
    *,
    window: int = 20,
    num_std: float = 2.0,
    use_sample_std: bool = True,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(middle, upper, lower)`` Bollinger Bands."""
    window_i = max(2, int(window))
    if len(values) < window_i:
        return None, None, None
    tail = [float(value) for value in list(values)[-window_i:]]
    middle = sum(tail) / float(window_i)

    if use_sample_std:
        std_value = sample_std(tail)
    else:
        variance = sum((value - middle) ** 2 for value in tail) / float(window_i)
        std_value = variance**0.5 if variance > 0.0 else None

    if std_value is None:
        return middle, None, None

    mult = float(num_std)
    return middle, middle + mult * std_value, middle - mult * std_value


def donchian_channel(
    highs,
    lows,
    *,
    window: int = 20,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(upper, middle, lower)`` Donchian channel."""
    window_i = max(1, int(window))
    if len(highs) < window_i or len(lows) < window_i:
        return None, None, None
    high_tail = [float(value) for value in list(highs)[-window_i:]]
    low_tail = [float(value) for value in list(lows)[-window_i:]]
    upper = max(high_tail)
    lower = min(low_tail)
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def keltner_channel(
    highs,
    lows,
    closes,
    *,
    window: int = 20,
    atr_window: int = 10,
    atr_multiplier: float = 2.0,
    use_ema_center: bool = True,
    ema_smoothing: float = 2.0,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(middle, upper, lower)`` Keltner channel."""
    if not highs or not lows or not closes:
        return None, None, None
    if len(highs) != len(lows) or len(highs) != len(closes):
        return None, None, None

    if use_ema_center:
        middle = exponential_moving_average(
            closes,
            max(1, int(window)),
            smoothing=float(ema_smoothing),
        )
    else:
        middle = simple_moving_average(closes, max(1, int(window)))
    if middle is None:
        return None, None, None

    tr_values: list[float] = []
    prev_close = None
    for high, low, close in zip(highs, lows, closes, strict=False):
        tr_values.append(true_range(float(high), float(low), prev_close))
        prev_close = float(close)

    atr_value = average_true_range(tr_values, max(1, int(atr_window)))
    if atr_value is None:
        return middle, None, None

    width = float(atr_multiplier) * atr_value
    return middle, middle + width, middle - width
