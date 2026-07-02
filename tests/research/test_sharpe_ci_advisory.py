"""Determinism + flat-payload non-interference for the advisory Sharpe-CI seam."""

from __future__ import annotations

import numpy as np
import pytest

from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.configuration.schema import SharpeConfidenceIntervalConfig
from lumina_quant.research import sharpe_ci as sc
from lumina_quant.strategy_factory import research_metrics


def _series() -> np.ndarray:
    rng = np.random.default_rng(20260701)
    return rng.normal(0.001, 0.01, size=256).astype(float)


def test_sharpe_ci_is_deterministic() -> None:
    series = _series()
    first = sc.compute_sharpe_confidence_interval(
        series, periods_per_year=365, bootstrap_rounds=500, seed=20260701
    )
    second = sc.compute_sharpe_confidence_interval(
        series, periods_per_year=365, bootstrap_rounds=500, seed=20260701
    )
    assert first == second
    assert first.lower <= first.point_estimate <= first.upper
    assert first.valid_rounds == 500
    assert first.block_size == 6


def test_short_or_degenerate_series_returns_degenerate_ci() -> None:
    ci = sc.compute_sharpe_confidence_interval(np.zeros(8), periods_per_year=365)
    assert ci.lower == ci.upper == ci.point_estimate == 0.0
    assert ci.valid_rounds == 0


def test_emit_gate_off_returns_none() -> None:
    cfg = SharpeConfidenceIntervalConfig()
    assert cfg.emit_enabled is False
    assert sc.maybe_emit_sharpe_ci_subobject(_series(), config=cfg) is None


def test_emit_gate_on_returns_nested_subobject() -> None:
    cfg = SharpeConfidenceIntervalConfig(emit_enabled=True, bootstrap_rounds=200)
    sub = sc.maybe_emit_sharpe_ci_subobject(_series(), config=cfg)
    assert isinstance(sub, dict)
    # Advisory CI is surfaced as a SEPARATE sub-object with its own keys.
    assert {"point_estimate", "lower", "upper", "confidence_level"} <= set(sub)


def test_flat_metric_payload_keyset_unaffected_by_import() -> None:
    """Importing / using the Sharpe-CI seam must not alter the flat payload."""
    rng = np.random.default_rng(7)
    size = 128
    returns = rng.normal(0.0005, 0.01, size=size).astype(float)
    turnover = np.abs(rng.normal(0.1, 0.05, size=size)).astype(float)
    exposure = rng.uniform(-1.0, 1.0, size=size).astype(float)
    benchmark = rng.normal(0.0002, 0.008, size=size).astype(float)
    summary = research_metrics.compute_metrics(
        returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark,
        periods_per_year=365,
        num_trials=1,
        metric_config=get_default_runtime_config().backtest,
    )
    # No Sharpe-CI keys leaked into the flat dict; every value stays a bare float.
    assert "sharpe_ci" not in summary
    assert "point_estimate" not in summary
    for value in summary.values():
        assert type(value) is float


def test_config_default_emit_is_off() -> None:
    rt = get_default_runtime_config()
    assert rt.research.sharpe_ci.emit_enabled is False


@pytest.mark.parametrize("level", [0.80, 0.90, 0.99])
def test_wider_confidence_widens_interval(level: float) -> None:
    series = _series()
    narrow = sc.compute_sharpe_confidence_interval(
        series, confidence_level=0.80, bootstrap_rounds=400, seed=1
    )
    wide = sc.compute_sharpe_confidence_interval(
        series, confidence_level=level, bootstrap_rounds=400, seed=1
    )
    assert (wide.upper - wide.lower) >= (narrow.upper - narrow.lower) - 1e-9
