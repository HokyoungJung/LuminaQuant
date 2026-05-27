"""Trend-following indicators with tunable parameters."""

from __future__ import annotations

from itertools import pairwise

from .atr import average_true_range, true_range
from .moving_average import exponential_moving_average


def moving_average_convergence_divergence(
    values,
    *,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    smoothing: float = 2.0,
) -> tuple[float | None, float | None, float | None]:
    """Return latest ``(macd, signal, histogram)`` tuple."""
    fast_i = max(1, int(fast_period))
    slow_i = max(fast_i + 1, int(slow_period))
    signal_i = max(1, int(signal_period))

    values_list = [float(value) for value in values]
    if len(values_list) < slow_i:
        return None, None, None

    macd_series: list[float] = []
    for idx in range(slow_i, len(values_list) + 1):
        frame = values_list[:idx]
        fast = exponential_moving_average(frame, fast_i, smoothing=smoothing)
        slow = exponential_moving_average(frame, slow_i, smoothing=smoothing)
        if fast is None or slow is None:
            continue
        macd_series.append(fast - slow)

    if not macd_series:
        return None, None, None
    macd = macd_series[-1]
    signal = exponential_moving_average(macd_series, signal_i, smoothing=smoothing)
    if signal is None:
        return macd, None, None
    return macd, signal, macd - signal


def average_directional_index(
    highs,
    lows,
    closes,
    *,
    period: int = 14,
) -> tuple[float | None, float | None, float | None]:
    """Return latest ``(adx, plus_di, minus_di)``."""
    period_i = max(2, int(period))
    n = min(len(highs), len(lows), len(closes))
    if n < period_i + 1:
        return None, None, None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    trs: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []

    for idx in range(1, n):
        up_move = highs_f[idx] - highs_f[idx - 1]
        down_move = lows_f[idx - 1] - lows_f[idx]
        plus_dm.append(up_move if up_move > down_move and up_move > 0.0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0.0 else 0.0)
        trs.append(true_range(highs_f[idx], lows_f[idx], closes_f[idx - 1]))

    if len(trs) < period_i:
        return None, None, None

    tr14 = sum(trs[:period_i])
    plus14 = sum(plus_dm[:period_i])
    minus14 = sum(minus_dm[:period_i])
    if tr14 <= 1e-12:
        return None, None, None

    dx_values: list[float] = []
    plus_di = 100.0 * (plus14 / tr14)
    minus_di = 100.0 * (minus14 / tr14)
    denom = plus_di + minus_di
    dx_values.append(0.0 if denom <= 1e-12 else 100.0 * abs(plus_di - minus_di) / denom)

    for idx in range(period_i, len(trs)):
        tr14 = tr14 - (tr14 / float(period_i)) + trs[idx]
        plus14 = plus14 - (plus14 / float(period_i)) + plus_dm[idx]
        minus14 = minus14 - (minus14 / float(period_i)) + minus_dm[idx]
        if tr14 <= 1e-12:
            continue
        plus_di = 100.0 * (plus14 / tr14)
        minus_di = 100.0 * (minus14 / tr14)
        denom = plus_di + minus_di
        dx_values.append(0.0 if denom <= 1e-12 else 100.0 * abs(plus_di - minus_di) / denom)

    if not dx_values:
        return None, None, None

    if len(dx_values) < period_i:
        adx = sum(dx_values) / float(len(dx_values))
    else:
        adx = sum(dx_values[:period_i]) / float(period_i)
        for dx_value in dx_values[period_i:]:
            adx = ((adx * float(period_i - 1)) + dx_value) / float(period_i)

    return adx, plus_di, minus_di


def aroon_indicator(
    highs,
    lows,
    *,
    period: int = 25,
) -> tuple[float | None, float | None, float | None]:
    """Return latest ``(aroon_up, aroon_down, oscillator)``."""
    period_i = max(2, int(period))
    if len(highs) < period_i or len(lows) < period_i:
        return None, None, None

    high_tail = [float(value) for value in list(highs)[-period_i:]]
    low_tail = [float(value) for value in list(lows)[-period_i:]]

    high_index = max(range(len(high_tail)), key=lambda idx: high_tail[idx])
    low_index = min(range(len(low_tail)), key=lambda idx: low_tail[idx])
    periods_since_high = period_i - 1 - high_index
    periods_since_low = period_i - 1 - low_index

    denom = float(period_i - 1)
    aroon_up = 100.0 * (denom - float(periods_since_high)) / denom
    aroon_down = 100.0 * (denom - float(periods_since_low)) / denom
    return aroon_up, aroon_down, aroon_up - aroon_down


def ichimoku_cloud(
    highs,
    lows,
    closes,
    *,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Return latest ``(tenkan, kijun, senkou_a, senkou_b, chikou)``."""
    n = min(len(highs), len(lows), len(closes))
    if n < max(int(tenkan_period), int(kijun_period), int(senkou_b_period)):
        return None, None, None, None, None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    tenkan_i = max(1, int(tenkan_period))
    kijun_i = max(1, int(kijun_period))
    span_b_i = max(1, int(senkou_b_period))

    tenkan_high = max(highs_f[-tenkan_i:])
    tenkan_low = min(lows_f[-tenkan_i:])
    tenkan = (tenkan_high + tenkan_low) / 2.0

    kijun_high = max(highs_f[-kijun_i:])
    kijun_low = min(lows_f[-kijun_i:])
    kijun = (kijun_high + kijun_low) / 2.0

    span_b_high = max(highs_f[-span_b_i:])
    span_b_low = min(lows_f[-span_b_i:])
    senkou_b = (span_b_high + span_b_low) / 2.0
    senkou_a = (tenkan + kijun) / 2.0
    chikou = closes_f[-1]
    return tenkan, kijun, senkou_a, senkou_b, chikou


def supertrend(
    highs,
    lows,
    closes,
    *,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> tuple[float | None, str | None, float | None, float | None]:
    """Return latest ``(supertrend_value, direction, upper_band, lower_band)``."""
    n = min(len(highs), len(lows), len(closes))
    atr_i = max(1, int(atr_period))
    if n < atr_i + 1:
        return None, None, None, None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    tr_values: list[float] = []
    for idx in range(1, n):
        tr_values.append(true_range(highs_f[idx], lows_f[idx], closes_f[idx - 1]))

    final_upper = None
    final_lower = None
    direction = "LONG"
    supertrend_value = None
    factor = float(multiplier)

    for idx in range(atr_i, n):
        atr_value = average_true_range(tr_values[:idx], atr_i)
        if atr_value is None:
            continue

        hl2 = (highs_f[idx] + lows_f[idx]) / 2.0
        basic_upper = hl2 + factor * atr_value
        basic_lower = hl2 - factor * atr_value

        if final_upper is None or closes_f[idx - 1] > final_upper:
            final_upper = basic_upper
        else:
            final_upper = min(basic_upper, final_upper)

        if final_lower is None or closes_f[idx - 1] < final_lower:
            final_lower = basic_lower
        else:
            final_lower = max(basic_lower, final_lower)

        if closes_f[idx] > final_upper:
            direction = "LONG"
        elif closes_f[idx] < final_lower:
            direction = "SHORT"

        supertrend_value = final_lower if direction == "LONG" else final_upper

    return supertrend_value, direction, final_upper, final_lower


def percentage_price_oscillator(
    values,
    *,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    smoothing: float = 2.0,
) -> tuple[float | None, float | None, float | None]:
    """Return latest ``(ppo, signal, histogram)`` tuple."""
    fast_i = max(1, int(fast_period))
    slow_i = max(fast_i + 1, int(slow_period))
    signal_i = max(1, int(signal_period))

    values_list = [float(value) for value in values]
    if len(values_list) < slow_i:
        return None, None, None

    ppo_series: list[float] = []
    for idx in range(slow_i, len(values_list) + 1):
        frame = values_list[:idx]
        fast = exponential_moving_average(frame, fast_i, smoothing=smoothing)
        slow = exponential_moving_average(frame, slow_i, smoothing=smoothing)
        if fast is None or slow is None or abs(slow) <= 1e-12:
            continue
        ppo_series.append(((fast - slow) / slow) * 100.0)

    if not ppo_series:
        return None, None, None
    ppo = ppo_series[-1]
    signal = exponential_moving_average(ppo_series, signal_i, smoothing=smoothing)
    if signal is None:
        return ppo, None, None
    return ppo, signal, ppo - signal


def triple_exponential_average_rate_of_change(
    values,
    *,
    period: int = 15,
    signal_period: int = 9,
    smoothing: float = 2.0,
) -> tuple[float | None, float | None]:
    """Return latest ``(trix, signal)`` tuple."""
    period_i = max(1, int(period))
    signal_i = max(1, int(signal_period))
    values_list = [float(value) for value in values]
    if len(values_list) < period_i + 1:
        return None, None

    ema1_series: list[float] = []
    ema2_series: list[float] = []
    ema3_series: list[float] = []

    for idx in range(period_i, len(values_list) + 1):
        frame = values_list[:idx]
        ema1 = exponential_moving_average(frame, period_i, smoothing=smoothing)
        if ema1 is None:
            continue
        ema1_series.append(ema1)

        ema2 = exponential_moving_average(ema1_series, period_i, smoothing=smoothing)
        if ema2 is None:
            continue
        ema2_series.append(ema2)

        ema3 = exponential_moving_average(ema2_series, period_i, smoothing=smoothing)
        if ema3 is None:
            continue
        ema3_series.append(ema3)

    if len(ema3_series) < 2:
        return None, None

    trix_series: list[float] = []
    for prev, curr in pairwise(ema3_series):
        if abs(prev) <= 1e-12:
            continue
        trix_series.append(((curr / prev) - 1.0) * 100.0)

    if not trix_series:
        return None, None

    trix = trix_series[-1]
    signal = exponential_moving_average(trix_series, signal_i, smoothing=smoothing)
    return trix, signal


def vortex_indicator(
    highs,
    lows,
    closes,
    *,
    period: int = 14,
) -> tuple[float | None, float | None]:
    """Return latest ``(vi_plus, vi_minus)`` Vortex Indicator values."""
    period_i = max(2, int(period))
    n = min(len(highs), len(lows), len(closes))
    if n < period_i + 1:
        return None, None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    vm_plus: list[float] = []
    vm_minus: list[float] = []
    tr_values: list[float] = []

    for idx in range(1, n):
        vm_plus.append(abs(highs_f[idx] - lows_f[idx - 1]))
        vm_minus.append(abs(lows_f[idx] - highs_f[idx - 1]))
        tr_values.append(true_range(highs_f[idx], lows_f[idx], closes_f[idx - 1]))

    vm_plus_sum = sum(vm_plus[-period_i:])
    vm_minus_sum = sum(vm_minus[-period_i:])
    tr_sum = sum(tr_values[-period_i:])
    if tr_sum <= 1e-12:
        return None, None

    return vm_plus_sum / tr_sum, vm_minus_sum / tr_sum


def linear_regression_slope(
    values,
    *,
    window: int = 20,
    normalize: bool = True,
) -> float | None:
    """Return latest linear-regression slope for trailing window."""
    window_i = max(2, int(window))
    values_list = [float(value) for value in values]
    if len(values_list) < window_i:
        return None

    y = values_list[-window_i:]
    x_mean = (window_i - 1) / 2.0
    y_mean = sum(y) / float(window_i)

    cov = 0.0
    var_x = 0.0
    for idx, y_val in enumerate(y):
        x_diff = float(idx) - x_mean
        y_diff = y_val - y_mean
        cov += x_diff * y_diff
        var_x += x_diff * x_diff

    if var_x <= 1e-12:
        return None
    slope = cov / var_x

    if not normalize:
        return slope
    if abs(y_mean) <= 1e-12:
        return None
    return slope / y_mean
