"""Momentum and spread helpers."""

from __future__ import annotations

import math


def cumulative_return(values, *, period: int | None = None) -> float | None:
    """Return cumulative return over the full series or trailing period."""
    values_f = [float(value) for value in values]
    if len(values_f) < 2:
        return None

    if period is None:
        start_idx = 0
    else:
        period_i = max(1, int(period))
        if len(values_f) <= period_i:
            return None
        start_idx = len(values_f) - period_i - 1

    start_value = values_f[start_idx]
    end_value = values_f[-1]
    if start_value <= 0.0:
        return None
    ret = (end_value / start_value) - 1.0
    return ret if math.isfinite(ret) else None


def momentum_return(latest: float, base: float) -> float | None:
    """Return simple momentum ``latest/base - 1`` with safety guards."""
    latest_f = float(latest)
    base_f = float(base)
    if base_f <= 0.0:
        return None
    momentum = (latest_f / base_f) - 1.0
    if not math.isfinite(momentum):
        return None
    return momentum


def momentum_spread(momentum_x: float, momentum_y: float) -> float:
    """Return spread between two momentum measurements."""
    return float(momentum_x) - float(momentum_y)


def lagged_momentum_return(
    values,
    *,
    lookback: int,
    skip_bars: int = 1,
) -> float | None:
    """Return lagged momentum while skipping the most-recent bars.

    This is useful for short-horizon crypto continuation effects where the
    predictive signal comes from the previous day's move rather than the
    in-progress bar.
    """
    values_f = [float(value) for value in values]
    lookback_i = max(1, int(lookback))
    skip_i = max(0, int(skip_bars))
    required = lookback_i + skip_i + 1
    if len(values_f) < required:
        return None

    latest_idx = len(values_f) - 1 - skip_i
    start_idx = latest_idx - lookback_i
    if start_idx < 0:
        return None

    base = values_f[start_idx]
    latest = values_f[latest_idx]
    if base <= 0.0:
        return None
    momentum = (latest / base) - 1.0
    return momentum if math.isfinite(momentum) else None


def rolling_mean_dollar_volume(
    close_values,
    volume_values,
    *,
    window: int,
) -> float | None:
    """Return trailing average dollar volume over ``window`` bars."""
    close_f = [float(value) for value in close_values]
    volume_f = [float(value) for value in volume_values]
    window_i = max(1, int(window))
    if len(close_f) < window_i or len(volume_f) < window_i:
        return None

    total = 0.0
    count = 0
    for close, volume in zip(close_f[-window_i:], volume_f[-window_i:], strict=True):
        if close <= 0.0 or volume < 0.0:
            continue
        total += close * volume
        count += 1
    if count <= 0:
        return None
    value = total / float(count)
    return value if math.isfinite(value) else None


def kaufman_efficiency_ratio(values, *, period: int = 10) -> float | None:
    """Return Kaufman Efficiency Ratio over trailing window."""
    period_i = max(1, int(period))
    values_f = [float(value) for value in values]
    if len(values_f) <= period_i:
        return None

    tail = values_f[-(period_i + 1) :]
    direction = abs(tail[-1] - tail[0])
    volatility = 0.0
    for idx in range(1, len(tail)):
        volatility += abs(tail[idx] - tail[idx - 1])
    if volatility <= 1e-12:
        return 0.0
    ratio = direction / volatility
    return ratio if math.isfinite(ratio) else None


def chande_momentum_oscillator(values, *, period: int = 14) -> float | None:
    """Return Chande Momentum Oscillator in ``[-100, 100]``."""
    period_i = max(1, int(period))
    values_f = [float(value) for value in values]
    if len(values_f) <= period_i:
        return None

    gains = 0.0
    losses = 0.0
    for idx in range(len(values_f) - period_i, len(values_f)):
        if idx <= 0:
            continue
        delta = values_f[idx] - values_f[idx - 1]
        if delta > 0.0:
            gains += delta
        elif delta < 0.0:
            losses += -delta

    denom = gains + losses
    if denom <= 1e-12:
        return 0.0
    cmo = 100.0 * ((gains - losses) / denom)
    return cmo if math.isfinite(cmo) else None


def detrended_price_oscillator(values, *, period: int = 20) -> float | None:
    """Return latest Detrended Price Oscillator (DPO)."""
    period_i = max(2, int(period))
    values_f = [float(value) for value in values]
    shift = (period_i // 2) + 1
    required = period_i + shift
    if len(values_f) < required:
        return None

    sma_window = values_f[-required:-shift]
    sma = sum(sma_window) / float(period_i)
    dpo = values_f[-1 - shift] - sma
    return dpo if math.isfinite(dpo) else None


def fisher_transform(values, *, period: int = 10, clamp: float = 0.999) -> float | None:
    """Return latest Fisher Transform value from normalized rolling price."""
    period_i = max(2, int(period))
    clamp_f = min(0.9999, max(0.5, float(clamp)))
    values_f = [float(value) for value in values]
    if len(values_f) < period_i:
        return None

    transformed: list[float] = []
    prev_x = 0.0
    for idx in range(period_i - 1, len(values_f)):
        window = values_f[idx - period_i + 1 : idx + 1]
        low_value = min(window)
        high_value = max(window)
        if high_value <= low_value:
            transformed.append(0.0)
            continue

        normalized = 2.0 * ((values_f[idx] - low_value) / (high_value - low_value) - 0.5)
        x = (0.33 * normalized) + (0.67 * prev_x)
        x = max(-clamp_f, min(clamp_f, x))
        fish = 0.5 * math.log((1.0 + x) / (1.0 - x))
        transformed.append(fish)
        prev_x = x

    if not transformed:
        return None
    value = transformed[-1]
    return value if math.isfinite(value) else None
