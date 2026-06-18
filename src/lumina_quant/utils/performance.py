import numpy as np


def _finite_array(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _aligned_array(values, *, size: int, default: float = 0.0) -> np.ndarray:
    if np.isscalar(values):
        return np.full(int(size), float(values), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != int(size):
        raise ValueError("Aligned metric inputs must have matching lengths.")
    if arr.size == 0:
        return arr
    arr = arr.copy()
    arr[~np.isfinite(arr)] = float(default)
    return arr


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
    raw = np.asarray(returns, dtype=np.float64)
    if raw.size == 0:
        return 0.0
    rf = _aligned_array(risk_free, size=raw.size, default=0.0)
    mask = np.isfinite(raw) & np.isfinite(rf)
    clean = raw[mask]
    if clean.size == 0:
        return 0.0
    excess = clean - rf[mask]
    # ddof=1 (sample std) to match the canonical convention used everywhere else
    # (utils.risk_free, optimization.native_backend, research_metrics). size<=1
    # guard avoids a NaN/RuntimeWarning and falls through to the std==0 -> 0.0 path.
    std = np.std(clean, ddof=1) if clean.size > 1 else 0.0
    if std == 0 or not np.isfinite(std):
        return 0.0
    sharpe = np.sqrt(periods) * np.mean(excess) / std
    if not np.isfinite(sharpe):
        return 0.0
    return sharpe


def create_sortino_ratio(returns, periods=252, risk_free=0.0, target=None):
    """Create the Sortino ratio (Downside Risk only)."""
    raw = np.asarray(returns, dtype=np.float64)
    if raw.size == 0:
        return 0.0
    target_values = risk_free if target is None else target
    target_arr = _aligned_array(target_values, size=raw.size, default=0.0)
    mask = np.isfinite(raw) & np.isfinite(target_arr)
    clean = raw[mask]
    if clean.size == 0:
        return 0.0
    target_clean = target_arr[mask]
    downside_returns = clean[clean < target_clean] - target_clean[clean < target_clean]
    if downside_returns.size == 0:
        return 0.0
    # ddof=1 (sample std) to match the canonical Sharpe/Sortino convention.
    downside_std = np.std(downside_returns, ddof=1) if downside_returns.size > 1 else 0.0
    if downside_std == 0 or not np.isfinite(downside_std):
        return 0.0
    sortino = np.sqrt(periods) * np.mean(clean - target_clean) / downside_std
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
    values = np.asarray(pnl, dtype=np.float64)
    count = int(values.size)
    if count == 0:
        return [], 0

    drawdown = np.zeros(count, dtype=np.float64)
    if count == 1:
        return drawdown.tolist(), 0

    # Preserve legacy behavior exactly:
    # hwm[0] = 0 and for t>=1 hwm[t] = max(hwm[t-1], pnl[t]).
    hwm_tail = np.maximum.accumulate(values[1:])
    hwm_tail = np.maximum(hwm_tail, 0.0)
    denominator = np.where(hwm_tail == 0.0, 1.0, hwm_tail)
    drawdown[1:] = (hwm_tail - values[1:]) / denominator

    active = drawdown > 0.0
    idx = np.arange(count, dtype=np.int64)
    last_reset = np.maximum.accumulate(np.where(active, -1, idx))
    duration = np.where(active, idx - last_reset, 0)

    return drawdown.tolist(), int(duration.max())


class PerformanceMetrics:
    """OOP facade for performance metric calculations."""

    @staticmethod
    def alpha_beta(strategy_returns, benchmark_returns, periods=252) -> tuple[float, float]:
        return create_alpha_beta(strategy_returns, benchmark_returns, periods=periods)

    @staticmethod
    def information_ratio(strategy_returns, benchmark_returns) -> float:
        return create_information_ratio(strategy_returns, benchmark_returns)

    @staticmethod
    def cagr(final_value, initial_value, periods, annual_periods=252) -> float:
        return create_cagr(
            final_value,
            initial_value,
            periods,
            annual_periods=annual_periods,
        )

    @staticmethod
    def annualized_volatility(returns, periods=252) -> float:
        return create_annualized_volatility(returns, periods=periods)

    @staticmethod
    def sharpe_ratio(returns, periods=252, risk_free=0.0) -> float:
        return create_sharpe_ratio(returns, periods=periods, risk_free=risk_free)

    @staticmethod
    def sortino_ratio(returns, periods=252, risk_free=0.0, target=None) -> float:
        return create_sortino_ratio(returns, periods=periods, risk_free=risk_free, target=target)

    @staticmethod
    def calmar_ratio(cagr, max_drawdown) -> float:
        return create_calmar_ratio(cagr, max_drawdown)

    @staticmethod
    def drawdowns(pnl) -> tuple[list[float], int]:
        return create_drawdowns(pnl)
