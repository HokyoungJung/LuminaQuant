"""Volatility and risk-oriented indicators with tunable settings."""

from __future__ import annotations

import math

from .atr import average_true_range, true_range
from .rolling_stats import sample_std


def log_returns(values) -> list[float]:
    """Return log-return series for positive input values."""
    values_f = [float(value) for value in values]
    out: list[float] = []
    for idx in range(1, len(values_f)):
        prev = values_f[idx - 1]
        curr = values_f[idx]
        if prev <= 0.0 or curr <= 0.0:
            continue
        out.append(math.log(curr / prev))
    return out


def historical_volatility(
    closes,
    *,
    window: int = 20,
    annualization: float = 252.0,
) -> float | None:
    """Return annualized historical volatility from log returns."""
    window_i = max(2, int(window))
    returns = log_returns(closes)
    if len(returns) < window_i:
        return None
    tail = returns[-window_i:]
    std_value = sample_std(tail)
    if std_value is None:
        return None
    return std_value * math.sqrt(float(annualization))


def atr_percent(
    highs,
    lows,
    closes,
    *,
    period: int = 14,
) -> float | None:
    """Return ATR as percentage of latest close."""
    period_i = max(1, int(period))
    n = min(len(highs), len(lows), len(closes))
    if n < period_i + 1:
        return None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    tr_values: list[float] = []
    for idx in range(1, n):
        tr_values.append(true_range(highs_f[idx], lows_f[idx], closes_f[idx - 1]))
    atr_value = average_true_range(tr_values, period_i)
    if atr_value is None:
        return None

    close = closes_f[-1]
    if close <= 0.0:
        return None
    return atr_value / close


def bollinger_bandwidth(
    closes,
    *,
    window: int = 20,
    num_std: float = 2.0,
) -> float | None:
    """Return Bollinger Bandwidth (upper-lower)/middle."""
    window_i = max(2, int(window))
    if len(closes) < window_i:
        return None

    tail = [float(value) for value in list(closes)[-window_i:]]
    middle = sum(tail) / float(window_i)
    if abs(middle) <= 1e-12:
        return None
    std_value = sample_std(tail)
    if std_value is None:
        return None

    upper = middle + float(num_std) * std_value
    lower = middle - float(num_std) * std_value
    return (upper - lower) / middle


def choppiness_index(
    highs,
    lows,
    closes,
    *,
    period: int = 14,
) -> float | None:
    """Return latest Choppiness Index."""
    period_i = max(2, int(period))
    n = min(len(highs), len(lows), len(closes))
    if n < period_i + 1:
        return None

    highs_f = [float(value) for value in highs][-n:]
    lows_f = [float(value) for value in lows][-n:]
    closes_f = [float(value) for value in closes][-n:]

    tr_values: list[float] = []
    for idx in range(1, n):
        tr_values.append(true_range(highs_f[idx], lows_f[idx], closes_f[idx - 1]))

    tr_tail = tr_values[-period_i:]
    tr_sum = sum(tr_tail)
    high_n = max(highs_f[-period_i:])
    low_n = min(lows_f[-period_i:])
    span = high_n - low_n
    if span <= 1e-12 or tr_sum <= 1e-12:
        return None

    return 100.0 * (math.log10(tr_sum / span) / math.log10(float(period_i)))


def ulcer_index(
    closes,
    *,
    window: int = 14,
) -> float | None:
    """Return Ulcer Index over rolling window."""
    window_i = max(2, int(window))
    if len(closes) < window_i:
        return None
    closes_f = [float(value) for value in list(closes)[-window_i:]]

    peak = closes_f[0]
    squared_drawdowns: list[float] = []
    for close in closes_f:
        peak = max(peak, close)
        if peak <= 1e-12:
            squared_drawdowns.append(0.0)
            continue
        drawdown_pct = ((close - peak) / peak) * 100.0
        squared_drawdowns.append(drawdown_pct * drawdown_pct)

    return math.sqrt(sum(squared_drawdowns) / float(len(squared_drawdowns)))


def downside_volatility(
    closes,
    *,
    window: int = 20,
    minimum_acceptable_return: float = 0.0,
    annualization: float = 252.0,
) -> float | None:
    """Return annualized downside volatility of rolling log returns."""
    window_i = max(2, int(window))
    mar = float(minimum_acceptable_return)
    returns = log_returns(closes)
    if len(returns) < window_i:
        return None

    tail = returns[-window_i:]
    downside = [min(0.0, value - mar) for value in tail]
    downside_variance = sum(value * value for value in downside) / float(window_i)
    if downside_variance <= 0.0:
        return 0.0
    return math.sqrt(downside_variance) * math.sqrt(float(annualization))


def parkinson_volatility(
    highs,
    lows,
    *,
    window: int = 20,
    annualization: float = 252.0,
) -> float | None:
    """Return annualized Parkinson volatility estimator."""
    window_i = max(2, int(window))
    n = min(len(highs), len(lows))
    if n < window_i:
        return None

    high_tail = [float(value) for value in list(highs)[-window_i:]]
    low_tail = [float(value) for value in list(lows)[-window_i:]]

    sum_sq = 0.0
    count = 0
    for high, low in zip(high_tail, low_tail, strict=False):
        if low <= 0.0 or high <= 0.0:
            continue
        ratio = high / low
        if ratio <= 0.0:
            continue
        log_range = math.log(ratio)
        sum_sq += log_range * log_range
        count += 1

    if count < 2:
        return None

    variance = sum_sq / (4.0 * float(count) * math.log(2.0))
    if variance <= 0.0:
        return None
    return math.sqrt(variance) * math.sqrt(float(annualization))


def max_drawdown(closes, *, window: int | None = None) -> float | None:
    """Return maximum drawdown ratio (negative or zero)."""
    closes_f = [float(value) for value in closes]
    if len(closes_f) < 2:
        return None

    if window is not None:
        window_i = max(2, int(window))
        if len(closes_f) < window_i:
            return None
        closes_f = closes_f[-window_i:]

    peak = closes_f[0]
    worst_dd = 0.0
    for close in closes_f:
        peak = max(peak, close)
        if peak <= 1e-12:
            continue
        drawdown = (close / peak) - 1.0
        if drawdown < worst_dd:
            worst_dd = drawdown
    return worst_dd


def value_at_risk(
    closes,
    *,
    window: int = 252,
    confidence: float = 0.95,
    use_log: bool = True,
) -> float | None:
    """Return historical Value at Risk as positive loss ratio."""
    window_i = max(5, int(window))
    conf = min(0.999, max(0.5, float(confidence)))
    if use_log:
        returns = log_returns(closes)
    else:
        closes_f = [float(value) for value in closes]
        returns = []
        for idx in range(1, len(closes_f)):
            prev = closes_f[idx - 1]
            if abs(prev) <= 1e-12:
                continue
            returns.append((closes_f[idx] / prev) - 1.0)

    if len(returns) < window_i:
        return None

    tail = sorted(returns[-window_i:])
    alpha = 1.0 - conf
    idx = max(0, min(len(tail) - 1, int(alpha * float(len(tail) - 1))))
    quantile = tail[idx]
    return max(0.0, -quantile)


def conditional_value_at_risk(
    closes,
    *,
    window: int = 252,
    confidence: float = 0.95,
    use_log: bool = True,
) -> float | None:
    """Return historical Conditional VaR (Expected Shortfall)."""
    window_i = max(5, int(window))
    conf = min(0.999, max(0.5, float(confidence)))
    if use_log:
        returns = log_returns(closes)
    else:
        closes_f = [float(value) for value in closes]
        returns = []
        for idx in range(1, len(closes_f)):
            prev = closes_f[idx - 1]
            if abs(prev) <= 1e-12:
                continue
            returns.append((closes_f[idx] / prev) - 1.0)

    if len(returns) < window_i:
        return None

    tail = sorted(returns[-window_i:])
    alpha = 1.0 - conf
    cutoff = max(1, int(alpha * float(len(tail))))
    worst = tail[:cutoff]
    if not worst:
        return None
    return max(0.0, -sum(worst) / float(len(worst)))


def rolling_sharpe_ratio(
    closes,
    *,
    window: int = 63,
    annualization: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float | None:
    """Return annualized rolling Sharpe ratio using log returns."""
    window_i = max(5, int(window))
    returns = log_returns(closes)
    if len(returns) < window_i:
        return None

    tail = returns[-window_i:]
    mean_return = sum(tail) / float(window_i)
    std_value = sample_std(tail)
    if std_value is None or std_value <= 1e-12:
        return None

    rf_daily = float(risk_free_rate) / float(annualization)
    excess = mean_return - rf_daily
    return (excess / std_value) * math.sqrt(float(annualization))


def rolling_sortino_ratio(
    closes,
    *,
    window: int = 63,
    annualization: float = 252.0,
    minimum_acceptable_return: float = 0.0,
) -> float | None:
    """Return annualized rolling Sortino ratio from log returns."""
    window_i = max(5, int(window))
    mar = float(minimum_acceptable_return) / float(annualization)
    returns = log_returns(closes)
    if len(returns) < window_i:
        return None

    tail = returns[-window_i:]
    mean_return = sum(tail) / float(window_i)
    downside = [min(0.0, value - mar) for value in tail]
    downside_var = sum(value * value for value in downside) / float(window_i)
    if downside_var <= 1e-12:
        return None
    downside_dev = math.sqrt(downside_var)
    excess = mean_return - mar
    return (excess / downside_dev) * math.sqrt(float(annualization))
