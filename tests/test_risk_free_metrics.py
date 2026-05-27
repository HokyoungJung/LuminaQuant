from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lumina_quant.config import BacktestConfig
from lumina_quant.strategy_factory import research_runner as rr
from lumina_quant.utils.risk_free import annual_to_periodic_rate, resolve_risk_free_config

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_saved_incumbent_portfolio.py"
SPEC = importlib.util.spec_from_file_location("validate_saved_incumbent_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load validate_saved_incumbent_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


_EXPECTED_METRIC_KEYS = {
    "return",
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "calmar",
    "mdd",
    "max_drawdown",
    "turnover",
    "trades",
    "trade_count",
    "win_rate",
    "avg_trade",
    "exposure",
    "volatility",
    "stability",
    "rolling_sharpe_min",
    "worst_month",
    "benchmark_corr",
    "deflated_sharpe",
    "pbo",
    "active_fold_ratio",
    "inactive_fold_count",
    "failed_fold_ratio",
    "spa_pvalue",
    "risk_free_annual",
    "risk_free_per_period",
    "sortino_target_annual",
    "sortino_target_per_period",
}


def test_annual_to_periodic_rate_matches_compound_formula() -> None:
    annual = 0.05
    periods = 365
    expected = (1.0 + annual) ** (1.0 / periods) - 1.0
    assert math.isclose(float(annual_to_periodic_rate(annual, periods)), expected, rel_tol=1e-12)


def test_resolve_risk_free_config_uses_us_treasury_constant_defaults(monkeypatch) -> None:
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_MODE", "us_treasury_constant", raising=False)
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_TENOR", "3m", raising=False)
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_ANNUAL", 0.0475, raising=False)
    monkeypatch.setattr(BacktestConfig, "SORTINO_TARGET_MODE", "same_as_rf", raising=False)
    monkeypatch.setattr(BacktestConfig, "SORTINO_TARGET_ANNUAL", 0.0, raising=False)

    resolved = resolve_risk_free_config(BacktestConfig, periods_per_year=365)

    assert resolved.mode == "us_treasury_constant"
    assert resolved.tenor == "3m"
    assert math.isclose(resolved.annual_rate, 0.0475, rel_tol=1e-12)
    assert math.isclose(resolved.sortino_target_annual, 0.0475, rel_tol=1e-12)


def test_validator_and_research_runner_metrics_share_sharpe_sortino_semantics(monkeypatch) -> None:
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_MODE", "us_treasury_constant", raising=False)
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_TENOR", "3m", raising=False)
    monkeypatch.setattr(BacktestConfig, "RISK_FREE_ANNUAL", 0.03, raising=False)
    monkeypatch.setattr(BacktestConfig, "SORTINO_TARGET_MODE", "same_as_rf", raising=False)
    monkeypatch.setattr(BacktestConfig, "SORTINO_TARGET_ANNUAL", 0.0, raising=False)

    returns = np.asarray([0.01, -0.005, 0.004, 0.006, -0.002, 0.003], dtype=float)
    timestamps = np.asarray(
        [
            np.datetime64("2026-01-01"),
            np.datetime64("2026-01-02"),
            np.datetime64("2026-01-03"),
            np.datetime64("2026-01-04"),
            np.datetime64("2026-01-05"),
            np.datetime64("2026-01-06"),
        ],
        dtype="datetime64[ms]",
    )

    validator_metrics = MODULE._metrics(returns, periods_per_year=365, timestamps=timestamps)
    research_metrics = rr._compute_metrics(
        returns,
        turnover=np.zeros_like(returns),
        exposure=np.zeros_like(returns),
        benchmark_returns=np.zeros_like(returns),
        periods_per_year=365,
        num_trials=1,
        metric_config=BacktestConfig,
        timestamps=timestamps,
    )

    assert math.isclose(validator_metrics["sharpe"], research_metrics["sharpe"], rel_tol=1e-12)
    assert math.isclose(validator_metrics["sortino"], research_metrics["sortino"], rel_tol=1e-12)


def test_compute_metrics_returns_empty_payload_for_empty_returns() -> None:
    metrics = rr._compute_metrics(
        np.asarray([], dtype=float),
        turnover=np.asarray([], dtype=float),
        exposure=np.asarray([], dtype=float),
        benchmark_returns=np.asarray([], dtype=float),
        periods_per_year=365,
        num_trials=1,
    )

    assert set(metrics) == _EXPECTED_METRIC_KEYS
    assert metrics["return"] == 0.0
    assert metrics["total_return"] == metrics["return"]
    assert metrics["trades"] == metrics["trade_count"] == 0.0
    assert metrics["mdd"] == metrics["max_drawdown"] == 0.0
    assert metrics["pbo"] == 1.0
    assert metrics["risk_free_annual"] == 0.0
    assert metrics["risk_free_per_period"] == 0.0
    assert metrics["sortino_target_annual"] == 0.0
    assert metrics["sortino_target_per_period"] == 0.0


def test_compute_metrics_assembles_resolved_payload(monkeypatch) -> None:
    resolved_rf = SimpleNamespace(
        periodic_rates=np.asarray([0.0], dtype=float),
        periodic_sortino_targets=np.asarray([0.0], dtype=float),
        annual_rate=0.05,
        per_period_rate=0.001,
        sortino_target_annual=0.04,
        sortino_target_per_period=0.0008,
    )
    metric_payload = rr._ComputedMetricPayload(
        total_return=0.12,
        cagr=0.10,
        sharpe=1.5,
        sortino=1.7,
        calmar=0.9,
        max_drawdown=0.08,
        turnover=0.4,
        trade_count=12.0,
        win_rate=0.6,
        avg_trade=0.01,
        exposure=0.5,
        volatility=0.2,
        stability=0.3,
        rolling_sharpe_min=-0.4,
        worst_month=-0.1,
        benchmark_corr=0.25,
        deflated_sharpe=0.7,
        pbo=0.2,
        active_fold_ratio=0.75,
        inactive_fold_count=1.0,
        failed_fold_ratio=0.25,
        spa_pvalue=0.15,
    )

    monkeypatch.setattr(rr, "resolve_risk_free_config", lambda *args, **kwargs: resolved_rf)
    monkeypatch.setattr(
        rr, "_resolve_compute_metric_payload", lambda *args, **kwargs: metric_payload
    )

    metrics = rr._compute_metrics(
        np.asarray([0.01, -0.005], dtype=float),
        turnover=np.asarray([0.0, 1.0], dtype=float),
        exposure=np.asarray([1.0, -1.0], dtype=float),
        benchmark_returns=np.asarray([0.0, 0.0], dtype=float),
        periods_per_year=365,
        num_trials=2,
    )

    assert set(metrics) == _EXPECTED_METRIC_KEYS
    assert metrics["return"] == 0.12
    assert metrics["total_return"] == metrics["return"]
    assert metrics["trades"] == metrics["trade_count"] == 12.0
    assert metrics["mdd"] == metrics["max_drawdown"] == 0.08
    assert metrics["risk_free_annual"] == 0.05
    assert metrics["risk_free_per_period"] == 0.001
    assert metrics["sortino_target_annual"] == 0.04
    assert metrics["sortino_target_per_period"] == 0.0008
