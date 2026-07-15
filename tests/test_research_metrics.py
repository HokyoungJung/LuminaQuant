from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lumina_quant.strategy_factory import research_metrics
from lumina_quant.strategy_factory import research_runner


def test_empty_compute_metric_payload_defaults_are_stable() -> None:
    payload = research_metrics.empty_compute_metric_payload()

    assert payload.total_return == 0.0
    assert payload.trade_count == 0.0
    assert payload.max_drawdown == 0.0
    assert payload.pbo == 1.0
    assert payload.spa_pvalue == 1.0


def test_max_drawdown_includes_initial_capital_peak() -> None:
    returns = np.asarray([-0.40, 0.10, 0.10, 0.10], dtype=float)

    assert research_metrics.max_drawdown(returns) == 0.40


def test_compute_metric_summary_exposes_alias_fields() -> None:
    payload = research_metrics.ComputedMetricPayload(
        total_return=0.12,
        cagr=0.1,
        sharpe=1.2,
        sortino=1.5,
        calmar=0.8,
        max_drawdown=0.15,
        turnover=0.3,
        trade_count=4.0,
        win_rate=0.5,
        avg_trade=0.02,
        exposure=0.4,
        volatility=0.25,
        stability=0.1,
        rolling_sharpe_min=-0.2,
        worst_month=-0.08,
        benchmark_corr=0.7,
        deflated_sharpe=0.6,
        pbo=0.1,
        active_fold_ratio=0.75,
        inactive_fold_count=1.0,
        failed_fold_ratio=0.25,
        spa_pvalue=0.2,
    )
    resolved_rf = SimpleNamespace(
        annual_rate=0.04,
        per_period_rate=0.001,
        sortino_target_annual=0.03,
        sortino_target_per_period=0.0008,
    )

    summary = research_metrics.compute_metric_summary(payload, resolved_rf=resolved_rf)

    assert summary["return"] == summary["total_return"] == 0.12
    assert summary["trades"] == summary["trade_count"] == 4.0
    assert summary["mdd"] == summary["max_drawdown"] == 0.15
    assert summary["active_fold_ratio"] == 0.75
    assert summary["inactive_fold_count"] == 1.0
    assert summary["failed_fold_ratio"] == 0.25
    assert summary["risk_free_annual"] == 0.04
    assert summary["sortino_target_per_period"] == 0.0008


def test_fold_participation_stats_flags_inactive_and_failed_folds() -> None:
    returns = np.concatenate(
        [
            np.full(32, 0.01, dtype=float),
            np.zeros(32, dtype=float),
            np.full(32, -0.01, dtype=float),
            np.zeros(32, dtype=float),
        ]
    )

    active_fold_ratio, inactive_fold_count, failed_fold_ratio = (
        research_metrics.fold_participation_stats(returns)
    )

    assert active_fold_ratio < 1.0
    assert inactive_fold_count >= 1.0
    assert failed_fold_ratio > 0.0


def test_research_metrics_compute_metrics_matches_runner_wrapper() -> None:
    returns = np.asarray([0.01, -0.02, 0.015, 0.005, -0.01, 0.012], dtype=float)
    turnover = np.asarray([0.0, 0.2, 0.1, 0.0, 0.3, 0.1], dtype=float)
    exposure = np.asarray([1.0, 1.0, -1.0, -1.0, 0.5, 0.5], dtype=float)
    benchmark_returns = np.asarray([0.0, -0.01, 0.01, 0.002, -0.004, 0.006], dtype=float)
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

    module_metrics = research_metrics.compute_metrics(
        returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark_returns,
        periods_per_year=365,
        num_trials=3,
        timestamps=timestamps,
    )
    runner_metrics = research_runner._compute_metrics(
        returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark_returns,
        periods_per_year=365,
        num_trials=3,
        timestamps=timestamps,
    )

    assert runner_metrics == module_metrics
