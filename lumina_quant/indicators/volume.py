"""Volume-derived indicators with tunable parameters."""

from __future__ import annotations

from .moving_average import exponential_moving_average, simple_moving_average
from .rolling_stats import rolling_corr


def on_balance_volume(closes, volumes) -> float | None:
    """Return latest OBV value."""
    n = min(len(closes), len(volumes))
    if n < 2:
        return None
    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]

    obv = 0.0
    for idx in range(1, n):
        if closes_f[idx] > closes_f[idx - 1]:
            obv += volumes_f[idx]
        elif closes_f[idx] < closes_f[idx - 1]:
            obv -= volumes_f[idx]
    return obv


def accumulation_distribution_line(highs, lows, closes, volumes) -> float | None:
    """Return latest Accumulation/Distribution (ADL) value."""
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n == 0:
        return None

    adl = 0.0
    for high, low, close, volume in zip(highs, lows, closes, volumes, strict=False):
        high_f = float(high)
        low_f = float(low)
        close_f = float(close)
        volume_f = max(0.0, float(volume))
        span = high_f - low_f
        if span <= 1e-12:
            continue
        money_flow_multiplier = ((close_f - low_f) - (high_f - close_f)) / span
        adl += money_flow_multiplier * volume_f
    return adl


def chaikin_money_flow(
    highs,
    lows,
    closes,
    volumes,
    *,
    period: int = 20,
) -> float | None:
    """Return latest Chaikin Money Flow value."""
    period_i = max(1, int(period))
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n < period_i:
        return None

    money_flow_volume: list[float] = []
    volume_values: list[float] = []
    for high, low, close, volume in zip(highs, lows, closes, volumes, strict=False):
        high_f = float(high)
        low_f = float(low)
        close_f = float(close)
        volume_f = max(0.0, float(volume))
        span = high_f - low_f
        if span <= 1e-12:
            multiplier = 0.0
        else:
            multiplier = ((close_f - low_f) - (high_f - close_f)) / span
        money_flow_volume.append(multiplier * volume_f)
        volume_values.append(volume_f)

    mfv_tail = money_flow_volume[-period_i:]
    vol_tail = volume_values[-period_i:]
    denom = sum(vol_tail)
    if denom <= 1e-12:
        return None
    return sum(mfv_tail) / denom


def volume_weighted_moving_average(closes, volumes, *, period: int = 20) -> float | None:
    """Return latest VWMA value."""
    period_i = max(1, int(period))
    n = min(len(closes), len(volumes))
    if n < period_i:
        return None

    close_tail = [float(value) for value in list(closes)[-period_i:]]
    volume_tail = [max(0.0, float(value)) for value in list(volumes)[-period_i:]]
    denom = sum(volume_tail)
    if denom <= 1e-12:
        return None
    num = sum(close * vol for close, vol in zip(close_tail, volume_tail, strict=False))
    return num / denom


def force_index(closes, volumes, *, ema_period: int = 13) -> float | None:
    """Return latest Force Index value (smoothed by EMA)."""
    n = min(len(closes), len(volumes))
    if n < 2:
        return None
    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]

    raw = [(closes_f[idx] - closes_f[idx - 1]) * volumes_f[idx] for idx in range(1, n)]
    if not raw:
        return None
    return exponential_moving_average(raw, max(1, int(ema_period)))


def ease_of_movement(
    highs,
    lows,
    volumes,
    *,
    divisor: float = 100000000.0,
    smoothing_period: int = 14,
) -> float | None:
    """Return latest Ease of Movement indicator."""
    n = min(len(highs), len(lows), len(volumes))
    if n < 2:
        return None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]
    scale = float(divisor)
    if abs(scale) <= 1e-12:
        return None

    emv_raw: list[float] = []
    for idx in range(1, n):
        distance_moved = ((highs_f[idx] + lows_f[idx]) / 2.0) - (
            (highs_f[idx - 1] + lows_f[idx - 1]) / 2.0
        )
        box_ratio_denom = volumes_f[idx] / scale
        span = highs_f[idx] - lows_f[idx]
        if abs(box_ratio_denom) <= 1e-12 or abs(span) <= 1e-12:
            emv_raw.append(0.0)
            continue
        box_ratio = box_ratio_denom / span
        emv_raw.append(distance_moved / box_ratio)

    period_i = max(1, int(smoothing_period))
    return simple_moving_average(emv_raw, period_i)


def chaikin_oscillator(
    highs,
    lows,
    closes,
    volumes,
    *,
    fast_period: int = 3,
    slow_period: int = 10,
    smoothing: float = 2.0,
) -> float | None:
    """Return Chaikin Oscillator (EMA fast - EMA slow of ADL)."""
    fast_i = max(1, int(fast_period))
    slow_i = max(fast_i + 1, int(slow_period))

    adl_series: list[float] = []
    running_adl = 0.0
    for high, low, close, volume in zip(highs, lows, closes, volumes, strict=False):
        high_f = float(high)
        low_f = float(low)
        close_f = float(close)
        volume_f = max(0.0, float(volume))
        span = high_f - low_f
        if span <= 1e-12:
            multiplier = 0.0
        else:
            multiplier = ((close_f - low_f) - (high_f - close_f)) / span
        running_adl += multiplier * volume_f
        adl_series.append(running_adl)

    if len(adl_series) < slow_i:
        return None

    ema_fast = exponential_moving_average(adl_series, fast_i, smoothing=smoothing)
    ema_slow = exponential_moving_average(adl_series, slow_i, smoothing=smoothing)
    if ema_fast is None or ema_slow is None:
        return None
    return ema_fast - ema_slow


def volume_price_trend(closes, volumes) -> float | None:
    """Return latest cumulative Volume Price Trend (VPT)."""
    n = min(len(closes), len(volumes))
    if n < 2:
        return None

    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]

    vpt = 0.0
    for idx in range(1, n):
        prev = closes_f[idx - 1]
        if abs(prev) <= 1e-12:
            continue
        vpt += volumes_f[idx] * ((closes_f[idx] - prev) / prev)
    return vpt


def positive_volume_index(closes, volumes, *, initial_value: float = 1000.0) -> float | None:
    """Return latest Positive Volume Index (PVI)."""
    n = min(len(closes), len(volumes))
    if n < 2:
        return None

    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]
    pvi = float(initial_value)

    for idx in range(1, n):
        prev_close = closes_f[idx - 1]
        if abs(prev_close) <= 1e-12:
            continue
        if volumes_f[idx] > volumes_f[idx - 1]:
            pvi *= 1.0 + ((closes_f[idx] - prev_close) / prev_close)
    return pvi


def negative_volume_index(closes, volumes, *, initial_value: float = 1000.0) -> float | None:
    """Return latest Negative Volume Index (NVI)."""
    n = min(len(closes), len(volumes))
    if n < 2:
        return None

    closes_f = [float(value) for value in closes][-n:]
    volumes_f = [max(0.0, float(value)) for value in volumes][-n:]
    nvi = float(initial_value)

    for idx in range(1, n):
        prev_close = closes_f[idx - 1]
        if abs(prev_close) <= 1e-12:
            continue
        if volumes_f[idx] < volumes_f[idx - 1]:
            nvi *= 1.0 + ((closes_f[idx] - prev_close) / prev_close)
    return nvi


def volume_oscillator(volumes, *, short_period: int = 14, long_period: int = 28) -> float | None:
    """Return percentage volume oscillator of short and long volume MAs."""
    short_i = max(1, int(short_period))
    long_i = max(short_i + 1, int(long_period))
    values = [max(0.0, float(value)) for value in volumes]
    if len(values) < long_i:
        return None

    short_ma = simple_moving_average(values, short_i)
    long_ma = simple_moving_average(values, long_i)
    if short_ma is None or long_ma is None or abs(long_ma) <= 1e-12:
        return None
    return ((short_ma - long_ma) / long_ma) * 100.0


def price_volume_correlation(closes, volumes, *, window: int = 20) -> float | None:
    """Return rolling correlation between close and volume."""
    window_i = max(2, int(window))
    closes_f = [float(value) for value in closes]
    volumes_f = [max(0.0, float(value)) for value in volumes]
    n = min(len(closes_f), len(volumes_f))
    if n < window_i:
        return None
    return rolling_corr(closes_f[-window_i:], volumes_f[-window_i:])
