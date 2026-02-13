import numpy as np


def _finite_array(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def create_alpha_beta(strategy_returns, benchmark_returns, periods=252):
    """Calculates Alpha and Beta.
    Beta = Cov(Ra, Rb) / Var(Rb)
    Alpha = R - Beta * Rb (Annualized)
    """
    # Ensure same length
    min_len = min(len(strategy_returns), len(benchmark_returns))
    if min_len < 2:
        return 0.0, 0.0
    s_ret = _finite_array(strategy_returns[:min_len])
    b_ret = _finite_array(benchmark_returns[:min_len])
    min_len = min(len(s_ret), len(b_ret))
    if min_len < 2:
        return 0.0, 0.0
    s_ret = s_ret[:min_len]
    b_ret = b_ret[:min_len]

    # Covariance Matrix
    matrix = np.cov(s_ret, b_ret)
    denominator = matrix[1, 1]
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0, 0.0
    beta = matrix[0, 1] / denominator
    if not np.isfinite(beta):
        beta = 0.0

    # Alpha (Annualized)
    # alpha = mean(s_ret) - beta * mean(b_ret)
    # Usually we compare annualized returns
    # But for daily alpha:
    alpha = np.mean(s_ret) - beta * np.mean(b_ret)
    alpha = alpha * periods
    if not np.isfinite(alpha):
        alpha = 0.0

    return alpha, beta


def create_information_ratio(strategy_returns, benchmark_returns):
    """Calculates Information Ratio (Active Return / Tracking Error)."""
    min_len = min(len(strategy_returns), len(benchmark_returns))
    if min_len < 2:
        return 0.0
    s_ret = np.asarray(strategy_returns[:min_len], dtype=np.float64)
    b_ret = np.asarray(benchmark_returns[:min_len], dtype=np.float64)
    mask = np.isfinite(s_ret) & np.isfinite(b_ret)
    if not np.any(mask):
        return 0.0
    active_return = s_ret[mask] - b_ret[mask]
    tracking_error = np.std(active_return)

    if tracking_error == 0 or not np.isfinite(tracking_error):
        return 0.0
    ratio = np.mean(active_return) / tracking_error * np.sqrt(252)
    if not np.isfinite(ratio):
        return 0.0
    return ratio


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
    clean = _finite_array(returns)
    if clean.size == 0:
        return 0.0
    vol = np.std(clean) * np.sqrt(periods)
    if not np.isfinite(vol):
        return 0.0
    return vol


def create_sharpe_ratio(returns, periods=252, risk_free=0.0):
    """Create the Sharpe ratio for the strategy."""
    clean = _finite_array(returns)
    if clean.size == 0:
        return 0.0
    std = np.std(clean)
    if std == 0 or not np.isfinite(std):
        return 0.0
    sharpe = np.sqrt(periods) * (np.mean(clean) - risk_free) / std
    if not np.isfinite(sharpe):
        return 0.0
    return sharpe


def create_sortino_ratio(returns, periods=252, risk_free=0.0):
    """Create the Sortino ratio (Downside Risk only)."""
    clean = _finite_array(returns)
    if clean.size == 0:
        return 0.0
    downside_returns = clean[clean < 0]
    if downside_returns.size == 0:
        return 0.0
    downside_std = np.std(downside_returns)
    if downside_std == 0 or not np.isfinite(downside_std):
        return 0.0
    sortino = np.sqrt(periods) * (np.mean(clean) - risk_free) / downside_std
    if not np.isfinite(sortino):
        return 0.0
    return sortino


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
