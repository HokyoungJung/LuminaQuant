"""Metric/statistic helpers extracted from the monolithic research runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.utils.risk_free import (
    resolve_risk_free_config,
    sharpe_ratio as compute_sharpe_ratio,
    sortino_ratio as compute_sortino_ratio,
)


# Euler-Mascheroni constant used by the canonical expected-maximum-Sharpe estimator.
_EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (quantile) via Acklam's rational approximation.

    Accurate to roughly 1e-9 across (0, 1); used in place of scipy.stats.norm.ppf
    because scipy is not a dependency of this package.
    """
    if not math.isfinite(p):
        return 0.0
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def expected_max_sharpe(*, variance_across_trials: float, num_trials: int) -> float:
    """Canonical expected maximum Sharpe across ``num_trials`` independent trials.

    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio":

        E[max SR] = sqrt(Var(SR_across_trials)) *
                    ((1 - gamma) * Phi^-1(1 - 1/k) + gamma * Phi^-1(1 - 1/(k*e)))

    with ``gamma`` the Euler-Mascheroni constant. This is the multiple-testing
    benchmark Sharpe that an overfit search is expected to surpass by chance; it
    is NOT scaled by the series length (the previous proxy divided by sqrt(n),
    conflating the cross-trial penalty with per-period scaling).
    """
    k = float(max(1, int(num_trials)))
    if k <= 1.0:
        return 0.0
    var = max(0.0, float(variance_across_trials))
    if var <= 0.0:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / k)
    z2 = _norm_ppf(1.0 - 1.0 / (k * math.e))
    gamma = _EULER_MASCHERONI
    return math.sqrt(var) * ((1.0 - gamma) * z1 + gamma * z2)


def safe_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    out = float(np.std(values, ddof=1))
    if not math.isfinite(out):
        return 0.0
    return out


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    out = float(np.mean(values))
    if not math.isfinite(out):
        return 0.0
    return out


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = 1.0 - np.divide(equity, np.maximum(peaks, 1e-12))
    return float(np.max(drawdown)) if drawdown.size else 0.0


def rolling_sharpe_min(
    returns: np.ndarray,
    *,
    window: int = 64,
    periods_per_year: int = 365,
) -> float:
    if returns.size < max(8, window):
        return 0.0
    vals: list[float] = []
    for idx in range(window, returns.size + 1):
        tail = returns[idx - window : idx]
        mu = safe_mean(tail)
        sigma = safe_std(tail)
        if sigma <= 1e-12:
            continue
        vals.append((mu / sigma) * math.sqrt(periods_per_year))
    if not vals:
        return 0.0
    return float(min(vals))


def worst_month(returns: np.ndarray, *, bars_per_month: int) -> float:
    if returns.size == 0:
        return 0.0
    bars = max(4, int(bars_per_month))
    monthly: list[float] = []
    for idx in range(0, returns.size, bars):
        tail = returns[idx : idx + bars]
        if tail.size == 0:
            continue
        monthly.append(float(np.prod(1.0 + tail) - 1.0))
    if not monthly:
        return 0.0
    return float(min(monthly))


def _estimate_sr_variance_across_trials(sharpe: float, n: float) -> float:
    """Fallback estimate of Var(SR) across trials when none is supplied.

    With no cross-trial sample available, use the asymptotic sampling variance of
    a single Sharpe estimate, Var(SR) ~= (1 + SR^2 / 2) / (n - 1) (Lo, 2002).
    This is a conservative, defensible stand-in for the dispersion of trial
    Sharpes that keeps ``deflated_sharpe_ratio`` usable on a single return series.
    """
    denom = max(1.0, n - 1.0)
    return max(0.0, (1.0 + 0.5 * (sharpe**2)) / denom)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    num_trials: int = 1,
    variance_across_trials: float | None = None,
) -> float:
    """Canonical Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Steps:
      1. SR is the observed (per-period, non-annualized) Sharpe of ``returns``.
      2. SR0 = E[max SR] over ``num_trials`` independent trials, using
         ``variance_across_trials`` (the dispersion of trial Sharpes). When that
         dispersion is not supplied it is approximated from SR's own sampling
         variance (see ``_estimate_sr_variance_across_trials``).
      3. DSR = Phi( ((SR - SR0) * sqrt(n - 1)) /
                    sqrt(1 - skew*SR + ((kurt - 1)/4)*SR^2) ),
         with non-Gaussian (skew/kurtosis) adjustment to the SR sampling error.

    Returns a probability in [0, 1]: the confidence that the true SR exceeds the
    multiple-testing benchmark SR0.
    """
    if returns.size < 16:
        return 0.0
    mu = safe_mean(returns)
    sigma = safe_std(returns)
    if sigma <= 1e-12:
        return 0.0

    sharpe = mu / sigma
    n = float(max(2, returns.size))

    if variance_across_trials is None:
        var_sr = _estimate_sr_variance_across_trials(sharpe, n)
    else:
        var_sr = max(0.0, float(variance_across_trials))
    expected_max = expected_max_sharpe(
        variance_across_trials=var_sr,
        num_trials=num_trials,
    )

    centered = returns - mu
    m3 = float(np.mean(centered**3))
    m4 = float(np.mean(centered**4))
    skew = 0.0 if sigma <= 1e-12 else m3 / (sigma**3)
    kurt = 3.0 if sigma <= 1e-12 else m4 / (sigma**4)

    denom_term = 1.0 - (skew * sharpe) + (((kurt - 1.0) / 4.0) * (sharpe**2))
    denom_term = max(1e-8, denom_term)
    z = ((sharpe - expected_max) * math.sqrt(max(1.0, n - 1.0))) / math.sqrt(denom_term)

    return float(max(0.0, min(1.0, _norm_cdf(z))))


def approx_pbo(returns: np.ndarray) -> float:
    """Approximate probability of backtest overfitting from fold instability."""
    n = returns.size
    if n < 64:
        return 1.0
    folds = min(8, max(4, n // 32))
    fold_size = n // folds
    if fold_size <= 0:
        return 1.0

    failures = 0
    trials = 0
    for idx in range(folds):
        test_start = idx * fold_size
        test_end = n if idx == folds - 1 else (idx + 1) * fold_size
        test = returns[test_start:test_end]
        train = np.concatenate((returns[:test_start], returns[test_end:]))
        if train.size < 8 or test.size < 8:
            continue
        train_sharpe = 0.0
        test_sharpe = 0.0
        train_std = safe_std(train)
        test_std = safe_std(test)
        if train_std > 1e-12:
            train_sharpe = safe_mean(train) / train_std
        if test_std > 1e-12:
            test_sharpe = safe_mean(test) / test_std
        trials += 1
        if train_sharpe > 0.0 and test_sharpe <= 0.0:
            failures += 1
    if trials <= 0:
        return 1.0
    return float(failures / trials)


def fold_participation_stats(returns: np.ndarray) -> tuple[float, float, float]:
    """Return active-fold ratio, inactive-fold count, and failed-fold ratio."""
    n = returns.size
    if n < 64:
        return 0.0, 0.0, 1.0
    folds = min(8, max(4, n // 32))
    fold_size = n // folds
    if fold_size <= 0:
        return 0.0, 0.0, 1.0

    trials = 0
    active = 0
    inactive = 0
    failures = 0
    for idx in range(folds):
        test_start = idx * fold_size
        test_end = n if idx == folds - 1 else (idx + 1) * fold_size
        test = returns[test_start:test_end]
        train = np.concatenate((returns[:test_start], returns[test_end:]))
        if train.size < 8 or test.size < 8:
            continue
        train_std = safe_std(train)
        test_std = safe_std(test)
        train_sharpe = 0.0 if train_std <= 1e-12 else safe_mean(train) / train_std
        test_sharpe = 0.0 if test_std <= 1e-12 else safe_mean(test) / test_std
        test_active = bool(np.any(np.abs(test) > 1e-12))
        trials += 1
        if test_active:
            active += 1
        else:
            inactive += 1
        if train_sharpe > 0.0 and (not test_active or test_sharpe <= 0.0):
            failures += 1
    if trials <= 0:
        return 0.0, 0.0, 1.0
    return float(active / trials), float(inactive), float(failures / trials)


def spa_like_pvalue(
    returns: np.ndarray,
    *,
    bootstrap_rounds: int = 200,
    block_size: int | None = None,
    seed: int = 12345,
) -> float:
    """Stationary (circular) block-bootstrap reality-check p-value.

    A White (2000) Reality Check / Hansen (2005) SPA in the single-strategy case:
    test H0 that the strategy's expected studentized excess return is <= 0 against
    the data-snooping null, preserving serial dependence via a circular block
    bootstrap (Politis & Romano, 1994) rather than i.i.d. resampling.

    Procedure:
      1. Studentize the observed mean: t_obs = mean(r) / (std(r) / sqrt(n)).
      2. Centre the series, then draw ``bootstrap_rounds`` circular block-bootstrap
         resamples of the centred series (block length defaults to ~ n**(1/3),
         the standard rate for dependent data) and recompute the studentized
         statistic on each.
      3. p-value = fraction of bootstrap statistics >= the observed statistic.

    A non-positive observed mean returns 1.0 (no evidence against the null). Pure
    noise yields a p-value near uniform (~0.5 on average, far from significant).
    """
    n = returns.size
    if n < 16:
        return 1.0
    observed_mean = safe_mean(returns)
    if observed_mean <= 0.0:
        return 1.0
    sigma = safe_std(returns)
    if sigma <= 1e-12:
        return 1.0

    rounds = max(64, int(bootstrap_rounds))
    if block_size is None:
        block_len = max(1, round(n ** (1.0 / 3.0)))
    else:
        block_len = max(1, min(n, int(block_size)))
    sqrt_n = math.sqrt(float(n))

    centered = returns - observed_mean
    t_obs = observed_mean / (sigma / sqrt_n)

    rng = np.random.default_rng(seed)
    num_blocks = math.ceil(n / block_len)
    exceed = 0
    valid_rounds = 0
    for _ in range(rounds):
        starts = rng.integers(0, n, size=num_blocks)
        offsets = (starts[:, None] + np.arange(block_len)[None, :]) % n
        sample = centered[offsets.reshape(-1)[:n]]
        sample_mean = safe_mean(sample)
        sample_sigma = safe_std(sample)
        if sample_sigma <= 1e-12:
            continue
        valid_rounds += 1
        t_boot = sample_mean / (sample_sigma / sqrt_n)
        if t_boot >= t_obs:
            exceed += 1
    if valid_rounds == 0:
        return 1.0
    return float(exceed / valid_rounds)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    n = min(x.size, y.size)
    if n < 8:
        return 0.0
    xa = x[-n:]
    ya = y[-n:]
    xs = safe_std(xa)
    ys = safe_std(ya)
    if xs <= 1e-12 or ys <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(xa, ya)[0, 1])
    if not math.isfinite(corr):
        return 0.0
    return corr


@dataclass(frozen=True, slots=True)
class ComputedMetricPayload:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    turnover: float
    trade_count: float
    win_rate: float
    avg_trade: float
    exposure: float
    volatility: float
    stability: float
    rolling_sharpe_min: float
    worst_month: float
    benchmark_corr: float
    deflated_sharpe: float
    pbo: float
    active_fold_ratio: float
    inactive_fold_count: float
    failed_fold_ratio: float
    spa_pvalue: float


def empty_compute_metric_payload() -> ComputedMetricPayload:
    return ComputedMetricPayload(
        total_return=0.0,
        cagr=0.0,
        sharpe=0.0,
        sortino=0.0,
        calmar=0.0,
        max_drawdown=0.0,
        turnover=0.0,
        trade_count=0.0,
        win_rate=0.0,
        avg_trade=0.0,
        exposure=0.0,
        volatility=0.0,
        stability=0.0,
        rolling_sharpe_min=0.0,
        worst_month=0.0,
        benchmark_corr=0.0,
        deflated_sharpe=0.0,
        pbo=1.0,
        active_fold_ratio=0.0,
        inactive_fold_count=0.0,
        failed_fold_ratio=1.0,
        spa_pvalue=1.0,
    )


def resolve_compute_metric_payload(
    returns: np.ndarray,
    *,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int,
    num_trials: int,
    resolved_rf: Any,
) -> ComputedMetricPayload:
    total_return = float(np.prod(1.0 + returns) - 1.0)
    years = max(1.0 / float(periods_per_year), returns.size / float(periods_per_year))
    cagr = float(math.exp(math.log1p(max(-0.999999, total_return)) / years) - 1.0)

    sigma = safe_std(returns)
    sharpe = compute_sharpe_ratio(
        returns,
        periods_per_year=periods_per_year,
        risk_free_per_period=np.asarray(resolved_rf.periodic_rates, dtype=float),
    )
    sortino = compute_sortino_ratio(
        returns,
        periods_per_year=periods_per_year,
        target_per_period=np.asarray(resolved_rf.periodic_sortino_targets, dtype=float),
    )

    max_dd = max_drawdown(returns)
    calmar = 0.0 if max_dd <= 1e-12 else cagr / max_dd

    trade_count = float(np.sum(turnover > 1e-9))
    win_rate = float(np.sum(returns > 0.0) / returns.size)
    avg_trade = float(np.sum(returns) / max(1.0, trade_count))

    rolling_min = rolling_sharpe_min(
        returns,
        window=min(128, max(32, returns.size // 8)),
        periods_per_year=periods_per_year,
    )
    worst_month_value = worst_month(
        returns,
        bars_per_month=max(4, int(periods_per_year // 12)),
    )
    stability = 0.5 * max(-3.0, min(3.0, rolling_min)) + 0.5 * worst_month_value

    active_fold_ratio, inactive_fold_count, failed_fold_ratio = fold_participation_stats(returns)

    return ComputedMetricPayload(
        total_return=total_return,
        cagr=cagr,
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=float(max_dd),
        turnover=float(safe_mean(turnover)),
        trade_count=trade_count,
        win_rate=win_rate,
        avg_trade=avg_trade,
        exposure=float(safe_mean(np.abs(exposure))),
        volatility=float(sigma * math.sqrt(periods_per_year)),
        stability=float(stability),
        rolling_sharpe_min=float(rolling_min),
        worst_month=float(worst_month_value),
        benchmark_corr=float(correlation(returns, benchmark_returns)),
        deflated_sharpe=float(deflated_sharpe_ratio(returns, num_trials=num_trials)),
        pbo=float(approx_pbo(returns)),
        active_fold_ratio=float(active_fold_ratio),
        inactive_fold_count=float(inactive_fold_count),
        failed_fold_ratio=float(failed_fold_ratio),
        spa_pvalue=float(spa_like_pvalue(returns)),
    )


def compute_metric_summary(
    metric_payload: ComputedMetricPayload,
    *,
    resolved_rf: Any | None,
) -> dict[str, float]:
    risk_free_annual = 0.0 if resolved_rf is None else float(resolved_rf.annual_rate)
    risk_free_per_period = 0.0 if resolved_rf is None else float(resolved_rf.per_period_rate)
    sortino_target_annual = 0.0 if resolved_rf is None else float(resolved_rf.sortino_target_annual)
    sortino_target_per_period = (
        0.0 if resolved_rf is None else float(resolved_rf.sortino_target_per_period)
    )
    return {
        "return": metric_payload.total_return,
        "total_return": metric_payload.total_return,
        "cagr": metric_payload.cagr,
        "sharpe": metric_payload.sharpe,
        "sortino": metric_payload.sortino,
        "calmar": metric_payload.calmar,
        "mdd": metric_payload.max_drawdown,
        "max_drawdown": metric_payload.max_drawdown,
        "turnover": metric_payload.turnover,
        "trades": metric_payload.trade_count,
        "trade_count": metric_payload.trade_count,
        "win_rate": metric_payload.win_rate,
        "avg_trade": metric_payload.avg_trade,
        "exposure": metric_payload.exposure,
        "volatility": metric_payload.volatility,
        "stability": metric_payload.stability,
        "rolling_sharpe_min": metric_payload.rolling_sharpe_min,
        "worst_month": metric_payload.worst_month,
        "benchmark_corr": metric_payload.benchmark_corr,
        "deflated_sharpe": metric_payload.deflated_sharpe,
        "pbo": metric_payload.pbo,
        "active_fold_ratio": metric_payload.active_fold_ratio,
        "inactive_fold_count": metric_payload.inactive_fold_count,
        "failed_fold_ratio": metric_payload.failed_fold_ratio,
        "spa_pvalue": metric_payload.spa_pvalue,
        "risk_free_annual": risk_free_annual,
        "risk_free_per_period": risk_free_per_period,
        "sortino_target_annual": sortino_target_annual,
        "sortino_target_per_period": sortino_target_per_period,
    }


def compute_metrics(
    returns: np.ndarray,
    *,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int,
    num_trials: int,
    metric_config: Any | None = None,
    timestamps: np.ndarray | None = None,
) -> dict[str, float]:
    if returns.size == 0:
        return compute_metric_summary(
            empty_compute_metric_payload(),
            resolved_rf=None,
        )

    resolved_rf = resolve_risk_free_config(
        metric_config or get_default_runtime_config().backtest,
        periods_per_year=periods_per_year,
        timestamps=timestamps,
        size=int(returns.size),
    )
    metric_payload = resolve_compute_metric_payload(
        returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        num_trials=num_trials,
        resolved_rf=resolved_rf,
    )
    return compute_metric_summary(
        metric_payload,
        resolved_rf=resolved_rf,
    )


__all__ = [
    "ComputedMetricPayload",
    "approx_pbo",
    "compute_metric_summary",
    "compute_metrics",
    "correlation",
    "deflated_sharpe_ratio",
    "empty_compute_metric_payload",
    "expected_max_sharpe",
    "fold_participation_stats",
    "max_drawdown",
    "resolve_compute_metric_payload",
    "resolve_risk_free_config",
    "rolling_sharpe_min",
    "safe_mean",
    "safe_std",
    "spa_like_pvalue",
    "worst_month",
]
