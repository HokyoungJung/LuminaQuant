import numpy as np


def create_alpha_beta(strategy_returns, benchmark_returns, periods=252):
    """Calculates Alpha and Beta.
    Beta = Cov(Ra, Rb) / Var(Rb)
    Alpha = R - Beta * Rb (Annualized)
    """
    # Ensure same length
    min_len = min(len(strategy_returns), len(benchmark_returns))
    s_ret = strategy_returns[:min_len]
    b_ret = benchmark_returns[:min_len]

    # Covariance Matrix
    matrix = np.cov(s_ret, b_ret)
    beta = matrix[0, 1] / matrix[1, 1]

    # Alpha (Annualized)
    # alpha = mean(s_ret) - beta * mean(b_ret)
    # Usually we compare annualized returns
    # But for daily alpha:
    alpha = np.mean(s_ret) - beta * np.mean(b_ret)
    alpha = alpha * periods

    return alpha, beta


def create_information_ratio(strategy_returns, benchmark_returns):
    """Calculates Information Ratio (Active Return / Tracking Error)."""
    min_len = min(len(strategy_returns), len(benchmark_returns))
    active_return = strategy_returns[:min_len] - benchmark_returns[:min_len]
    tracking_error = np.std(active_return)

    if tracking_error == 0:
        return 0.0
    return np.mean(active_return) / tracking_error * np.sqrt(252)


def create_cagr(final_value, initial_value, periods, annual_periods=252):
    """Calculate Compound Annual Growth Rate."""
    if initial_value == 0:
        return 0.0
    years = periods / annual_periods
    if years == 0:
        return 0.0
    return (final_value / initial_value) ** (1 / years) - 1


def create_annualized_volatility(returns, periods=252):
    """Calculate Annualized Volatility."""
    return np.std(returns) * np.sqrt(periods)


def create_sharpe_ratio(returns, periods=252, risk_free=0.0):
    """Create the Sharpe ratio for the strategy."""
    if np.std(returns) == 0:
        return 0.0
    return np.sqrt(periods) * (np.mean(returns) - risk_free) / np.std(returns)


def create_sortino_ratio(returns, periods=252, risk_free=0.0):
    """Create the Sortino ratio (Downside Risk only)."""
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0
    return np.sqrt(periods) * (np.mean(returns) - risk_free) / downside_std


def create_calmar_ratio(cagr, max_drawdown):
    """Create the Calmar ratio (CAGR / MaxDD)."""
    if max_drawdown == 0:
        return 0.0
    # MaxDD is usually positive in stats, but if passed as negative, handle it.
    return cagr / abs(max_drawdown)


def create_drawdowns(pnl):
    """Calculate the largest peak-to-trough drawdown of the PnL curve.
    pnl: list or numpy array of equity curve values
    """
    hwm = [0]
    drawdown = [0.0] * len(pnl)
    duration = [0] * len(pnl)

    for t in range(1, len(pnl)):
        cur_val = pnl[t]
        hwm.append(max(hwm[t - 1], cur_val))

        div = hwm[t]
        if div == 0:
            div = 1  # avoid div by zero

        dd = (hwm[t] - cur_val) / div
        drawdown[t] = dd
        duration[t] = 0 if dd == 0 else duration[t - 1] + 1

    return drawdown, max(duration)
