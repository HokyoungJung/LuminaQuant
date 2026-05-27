from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from lumina_quant.services.portfolio import PortfolioPerformanceService
from lumina_quant.strategy_factory import research_runner
from lumina_quant.utils.risk_free import annual_to_periodic_rate

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_saved_incumbent_portfolio.py"
SPEC = importlib.util.spec_from_file_location("validate_saved_incumbent_portfolio", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load validate_saved_incumbent_portfolio module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _MetricConfig:
    RISK_FREE_MODE = "us_treasury_constant"
    RISK_FREE_TENOR = "3m"
    RISK_FREE_ANNUAL = 0.052
    RISK_FREE_RATE = 0.0
    RISK_FREE_SERIES_PATH = ""
    SORTINO_TARGET_MODE = "same_as_rf"
    SORTINO_TARGET_ANNUAL = 0.0
    ANNUAL_PERIODS = 365


def test_annual_to_periodic_rate_matches_compound_formula() -> None:
    expected = (1.0 + 0.052) ** (1.0 / 365.0) - 1.0
    assert annual_to_periodic_rate(0.052, 365) == pytest.approx(expected)


def test_sharpe_and_sortino_are_consistent_across_validation_research_and_summary_paths() -> None:
    returns = np.asarray([0.01, -0.02, 0.015, 0.005, -0.01, 0.012], dtype=float)
    timestamps = np.asarray(
        [
            np.datetime64("2026-03-01T00:00:00"),
            np.datetime64("2026-03-02T00:00:00"),
            np.datetime64("2026-03-03T00:00:00"),
            np.datetime64("2026-03-04T00:00:00"),
            np.datetime64("2026-03-05T00:00:00"),
            np.datetime64("2026-03-06T00:00:00"),
        ],
        dtype="datetime64[ms]",
    )

    validator_metrics = MODULE._metrics(
        returns,
        periods_per_year=365,
        timestamps=timestamps,
        metric_config=_MetricConfig,
    )
    research_metrics = research_runner._compute_metrics(
        returns,
        turnover=np.zeros_like(returns),
        exposure=np.zeros_like(returns),
        benchmark_returns=np.zeros_like(returns),
        periods_per_year=365,
        num_trials=1,
        metric_config=_MetricConfig,
        timestamps=timestamps,
    )

    total = [100.0]
    for ret in returns:
        total.append(total[-1] * (1.0 + float(ret)))
    equity_curve = pl.DataFrame(
        {
            "datetime": [
                "2026-03-01T00:00:00Z",
                "2026-03-02T00:00:00Z",
                "2026-03-03T00:00:00Z",
                "2026-03-04T00:00:00Z",
                "2026-03-05T00:00:00Z",
                "2026-03-06T00:00:00Z",
                "2026-03-07T00:00:00Z",
            ],
            "total": total,
            "returns": [0.0, *returns.tolist()],
            "benchmark_returns": [0.0] * 7,
            "benchmark_price": [100.0] * 7,
        }
    )
    summary = dict(
        PortfolioPerformanceService.build_summary_stats(
            equity_curve=equity_curve,
            config=_MetricConfig,
            total_funding_paid=0.0,
            liquidation_count=0,
        )
    )
    summary_sharpe = float(summary["Sharpe Ratio"])
    summary_sortino = float(summary["Sortino Ratio"])

    assert validator_metrics["risk_free_per_period"] == pytest.approx(
        research_metrics["risk_free_per_period"]
    )
    assert validator_metrics["sharpe"] == pytest.approx(research_metrics["sharpe"])
    assert validator_metrics["sortino"] == pytest.approx(research_metrics["sortino"])
    assert summary_sharpe == pytest.approx(validator_metrics["sharpe"], rel=1e-4)
    assert summary_sortino == pytest.approx(validator_metrics["sortino"], rel=1e-4)
