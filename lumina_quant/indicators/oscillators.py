"""Oscillator indicators with fully tunable parameters."""

from __future__ import annotations

import math

from .moving_average import exponential_moving_average, simple_moving_average
from .rolling_stats import sample_std


def relative_strength_index(values, *, period: int = 14) -> float | None:
    """Return latest RSI computed from close values."""
    period_i = max(1, int(period))
    values_list = [float(value) for value in values]
    if len(values_list) < period_i + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for idx in range(1, len(values_list)):
        delta = values_list[idx] - values_list[idx - 1]
        gains.append(delta if delta > 0.0 else 0.0)
        losses.append(-delta if delta < 0.0 else 0.0)

    avg_gain = sum(gains[:period_i]) / float(period_i)
    avg_loss = sum(losses[:period_i]) / float(period_i)
    for gain, loss in zip(gains[period_i:], losses[period_i:], strict=False):
        avg_gain = ((avg_gain * float(period_i - 1)) + gain) / float(period_i)
        avg_loss = ((avg_loss * float(period_i - 1)) + loss) / float(period_i)

    if avg_loss <= 0.0:
        return 100.0 if avg_gain > 0.0 else 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rate_of_change(values, *, period: int = 12) -> float | None:
    """Return latest ROC in decimal form."""
    period_i = max(1, int(period))
    values_list = [float(value) for value in values]
    if len(values_list) <= period_i:
        return None
    base = values_list[-1 - period_i]
    if base <= 0.0:
        return None
    return (values_list[-1] / base) - 1.0


def commodity_channel_index(
    highs,
    lows,
    closes,
    *,
    period: int = 20,
    constant: float = 0.015,
) -> float | None:
    """Return latest CCI value."""
    period_i = max(2, int(period))
    if len(highs) < period_i or len(lows) < period_i or len(closes) < period_i:
        return None

    tp = [
        (float(high) + float(low) + float(close)) / 3.0
        for high, low, close in zip(highs, lows, closes, strict=False)
    ]
    tail = tp[-period_i:]
    tp_sma = sum(tail) / float(period_i)
    mean_dev = sum(abs(value - tp_sma) for value in tail) / float(period_i)
    if mean_dev <= 1e-12:
        return None
    return (tail[-1] - tp_sma) / (float(constant) * mean_dev)


def stochastic_oscillator(
    highs,
    lows,
    closes,
    *,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 1,
) -> tuple[float | None, float | None]:
    """Return latest ``(%K, %D)`` stochastic values."""
    k_period_i = max(1, int(k_period))
    d_period_i = max(1, int(d_period))
    smooth_k_i = max(1, int(smooth_k))

    if len(highs) < k_period_i or len(lows) < k_period_i or len(closes) < k_period_i:
        return None, None

    raw_k_values: list[float] = []
    highs_list = [float(value) for value in highs]
    lows_list = [float(value) for value in lows]
    closes_list = [float(value) for value in closes]

    for idx in range(k_period_i - 1, len(closes_list)):
        high_window = highs_list[idx - k_period_i + 1 : idx + 1]
        low_window = lows_list[idx - k_period_i + 1 : idx + 1]
        highest = max(high_window)
        lowest = min(low_window)
        if highest <= lowest:
            raw_k_values.append(50.0)
            continue
        raw_k_values.append(((closes_list[idx] - lowest) / (highest - lowest)) * 100.0)

    if len(raw_k_values) < smooth_k_i:
        return None, None

    smooth_k_values: list[float] = []
    for idx in range(smooth_k_i - 1, len(raw_k_values)):
        window = raw_k_values[idx - smooth_k_i + 1 : idx + 1]
        smooth_k_values.append(sum(window) / float(smooth_k_i))

    latest_k = smooth_k_values[-1]
    if len(smooth_k_values) < d_period_i:
        return latest_k, None
    latest_d = sum(smooth_k_values[-d_period_i:]) / float(d_period_i)
    return latest_k, latest_d


def williams_r(
    highs,
    lows,
    closes,
    *,
    period: int = 14,
) -> float | None:
    """Return latest Williams %R."""
    period_i = max(1, int(period))
    if len(highs) < period_i or len(lows) < period_i or len(closes) < period_i:
        return None

    high_window = [float(value) for value in list(highs)[-period_i:]]
    low_window = [float(value) for value in list(lows)[-period_i:]]
    highest = max(high_window)
    lowest = min(low_window)
    if highest <= lowest:
        return None
    close = float(list(closes)[-1])
    return -100.0 * ((highest - close) / (highest - lowest))


def money_flow_index(
    highs,
    lows,
    closes,
    volumes,
    *,
    period: int = 14,
) -> float | None:
    """Return latest MFI value."""
    period_i = max(1, int(period))
    if min(len(highs), len(lows), len(closes), len(volumes)) < period_i + 1:
        return None

    tp = [
        (float(high) + float(low) + float(close)) / 3.0
        for high, low, close in zip(highs, lows, closes, strict=False)
    ]
    rmf = [price * max(0.0, float(volume)) for price, volume in zip(tp, volumes, strict=False)]

    positive = 0.0
    negative = 0.0
    for idx in range(len(tp) - period_i, len(tp)):
        if idx <= 0:
            continue
        if tp[idx] > tp[idx - 1]:
            positive += rmf[idx]
        elif tp[idx] < tp[idx - 1]:
            negative += rmf[idx]

    if negative <= 1e-12:
        return 100.0 if positive > 0.0 else 50.0
    ratio = positive / negative
    return 100.0 - (100.0 / (1.0 + ratio))


def stochastic_rsi(
    values,
    *,
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[float | None, float | None]:
    """Return latest stochastic RSI ``(%K, %D)``."""
    rsi_period_i = max(1, int(rsi_period))
    stoch_period_i = max(1, int(stoch_period))
    smooth_k_i = max(1, int(smooth_k))
    smooth_d_i = max(1, int(smooth_d))

    values_list = [float(value) for value in values]
    if len(values_list) < rsi_period_i + stoch_period_i + smooth_k_i:
        return None, None

    rsi_series: list[float] = []
    for idx in range(rsi_period_i, len(values_list)):
        rsi_value = relative_strength_index(values_list[: idx + 1], period=rsi_period_i)
        if rsi_value is not None:
            rsi_series.append(rsi_value)

    if len(rsi_series) < stoch_period_i:
        return None, None

    stoch_values: list[float] = []
    for idx in range(stoch_period_i - 1, len(rsi_series)):
        window = rsi_series[idx - stoch_period_i + 1 : idx + 1]
        low_value = min(window)
        high_value = max(window)
        if high_value <= low_value:
            stoch_values.append(50.0)
        else:
            stoch_values.append(((rsi_series[idx] - low_value) / (high_value - low_value)) * 100.0)

    if len(stoch_values) < smooth_k_i:
        return None, None

    smooth_k_values: list[float] = []
    for idx in range(smooth_k_i - 1, len(stoch_values)):
        window = stoch_values[idx - smooth_k_i + 1 : idx + 1]
        smooth_k_values.append(sum(window) / float(smooth_k_i))

    latest_k = smooth_k_values[-1]
    if len(smooth_k_values) < smooth_d_i:
        return latest_k, None
    latest_d = sum(smooth_k_values[-smooth_d_i:]) / float(smooth_d_i)
    return latest_k, latest_d


def true_strength_index(
    values,
    *,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 7,
) -> tuple[float | None, float | None]:
    """Return latest ``(TSI, signal)`` values."""
    values_list = [float(value) for value in values]
    if len(values_list) < max(3, int(long_period) + int(short_period)):
        return None, None

    momentum = [values_list[idx] - values_list[idx - 1] for idx in range(1, len(values_list))]
    abs_momentum = [abs(value) for value in momentum]

    ema1_m = []
    ema1_abs = []
    for idx in range(1, len(values_list)):
        ema_m = exponential_moving_average(momentum[:idx], max(1, int(long_period)))
        ema_a = exponential_moving_average(abs_momentum[:idx], max(1, int(long_period)))
        if ema_m is None or ema_a is None:
            continue
        ema1_m.append(ema_m)
        ema1_abs.append(ema_a)

    if not ema1_m or not ema1_abs:
        return None, None

    ema2_m = exponential_moving_average(ema1_m, max(1, int(short_period)))
    ema2_abs = exponential_moving_average(ema1_abs, max(1, int(short_period)))
    if ema2_m is None or ema2_abs is None or abs(ema2_abs) <= 1e-12:
        return None, None

    tsi = 100.0 * (ema2_m / ema2_abs)

    tsi_series: list[float] = []
    for idx in range(1, len(ema1_m) + 1):
        ema2_m_i = exponential_moving_average(ema1_m[:idx], max(1, int(short_period)))
        ema2_abs_i = exponential_moving_average(ema1_abs[:idx], max(1, int(short_period)))
        if ema2_m_i is None or ema2_abs_i is None or abs(ema2_abs_i) <= 1e-12:
            continue
        tsi_series.append(100.0 * (ema2_m_i / ema2_abs_i))

    signal = (
        exponential_moving_average(tsi_series, max(1, int(signal_period))) if tsi_series else None
    )
    return tsi, signal


def zscore(values, *, window: int = 20, use_sample_std: bool = True) -> float | None:
    """Return latest z-score over rolling window."""
    window_i = max(2, int(window))
    if len(values) < window_i:
        return None
    tail = [float(value) for value in list(values)[-window_i:]]
    mean_value = sum(tail) / float(window_i)

    if use_sample_std:
        std_value = sample_std(tail)
    else:
        variance = sum((value - mean_value) ** 2 for value in tail) / float(window_i)
        std_value = math.sqrt(variance) if variance > 0.0 else None

    if std_value is None or std_value <= 1e-12:
        return None
    return (tail[-1] - mean_value) / std_value


def percentile_rank(values, *, window: int = 20) -> float | None:
    """Return percentile rank of latest value in trailing window."""
    window_i = max(2, int(window))
    if len(values) < window_i:
        return None
    tail = [float(value) for value in list(values)[-window_i:]]
    latest = tail[-1]
    below = sum(1 for value in tail if value < latest)
    equal = sum(1 for value in tail if value == latest)
    return (below + 0.5 * equal) / float(window_i)



def awesome_oscillator(highs, lows, *, fast_period: int = 5, slow_period: int = 34) -> float | None:
    """Return Awesome Oscillator using median price SMAs."""
    fast_i = max(1, int(fast_period))
    slow_i = max(fast_i + 1, int(slow_period))
    n = min(len(highs), len(lows))
    if n < slow_i:
        return None

    median_prices = [
        (float(high) + float(low)) / 2.0 for high, low in zip(highs, lows, strict=False)
    ]
    fast = simple_moving_average(median_prices, fast_i)
    slow = simple_moving_average(median_prices, slow_i)
    if fast is None or slow is None:
        return None
    return fast - slow


def ultimate_oscillator(
    highs,
    lows,
    closes,
    *,
    short_period: int = 7,
    medium_period: int = 14,
    long_period: int = 28,
    short_weight: float = 4.0,
    medium_weight: float = 2.0,
    long_weight: float = 1.0,
) -> float | None:
    """Return latest Ultimate Oscillator value."""
    short_i = max(1, int(short_period))
    medium_i = max(short_i + 1, int(medium_period))
    long_i = max(medium_i + 1, int(long_period))

    n = min(len(highs), len(lows), len(closes))
    if n < long_i + 1:
        return None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    buying_pressure: list[float] = []
    true_range_values: list[float] = []
    for idx in range(1, n):
        prev_close = closes_f[idx - 1]
        low_ref = min(lows_f[idx], prev_close)
        high_ref = max(highs_f[idx], prev_close)
        bp = closes_f[idx] - low_ref
        tr = high_ref - low_ref
        buying_pressure.append(bp)
        true_range_values.append(max(tr, 0.0))

    def _avg(period: int) -> float | None:
        bp_tail = buying_pressure[-period:]
        tr_tail = true_range_values[-period:]
        denom = sum(tr_tail)
        if denom <= 1e-12:
            return None
        return sum(bp_tail) / denom

    avg_short = _avg(short_i)
    avg_medium = _avg(medium_i)
    avg_long = _avg(long_i)
    if avg_short is None or avg_medium is None or avg_long is None:
        return None

    w1 = float(short_weight)
    w2 = float(medium_weight)
    w3 = float(long_weight)
    denom = w1 + w2 + w3
    if abs(denom) <= 1e-12:
        return None
    return 100.0 * ((w1 * avg_short + w2 * avg_medium + w3 * avg_long) / denom)
