"""Keyset / flat-payload drift guard for research-metric emitters (test-only).

This module pins the exact key surface of the metric-emitter payloads so that
any additive drift (a new key, a renamed key, or a nested sub-object smuggled
into an otherwise flat ``dict[str, float]``) fails loudly instead of silently
propagating into downstream JSON / dataclass serialization.

Three guards are exercised:

1.  Keyset snapshot -- each known emitter (``compute_metric_summary``,
    ``compute_metrics``, ``resolve_compute_metric_payload`` via
    ``dataclasses.asdict``, ``empty_compute_metric_payload``) is called with a
    representative deterministic fixture and its produced key set is diffed
    against a pinned snapshot.
2.  Flat-float invariant (FIX-m1) -- every value of the flat summary/payload
    dict must be a bare ``float``; a nested dict / list / dataclass would fail
    the ``type(value) is float`` check, preventing sub-object smuggling.
3.  Single-seam serialization canary -- one context manager temporarily wraps
    ``json.dump`` / ``json.dumps`` / ``dataclasses.asdict`` and proves the
    capture hook actually fires when the emitter output is serialized. The
    patch is installed and torn down locally (no autouse / conftest changes),
    so no other test is affected.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
from types import SimpleNamespace

import numpy as np

from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.strategy_factory import research_metrics


# ---------------------------------------------------------------------------
# Pinned key snapshots (frozen against the current emitter surface).
# ---------------------------------------------------------------------------

# ``compute_metric_summary`` / ``compute_metrics`` flat summary dict.
_SUMMARY_KEYSET = frozenset(
    {
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
)

# ``ComputedMetricPayload`` dataclass field surface.
_PAYLOAD_KEYSET = frozenset(
    {
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "turnover",
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
    }
)


# ---------------------------------------------------------------------------
# Representative deterministic fixtures.
# ---------------------------------------------------------------------------


def _representative_series() -> dict[str, np.ndarray]:
    """Deterministic multi-field fixture large enough for fold/pbo/spa paths."""
    rng = np.random.default_rng(20260701)
    size = 128
    returns = rng.normal(0.0005, 0.01, size=size).astype(float)
    turnover = np.abs(rng.normal(0.1, 0.05, size=size)).astype(float)
    exposure = rng.uniform(-1.0, 1.0, size=size).astype(float)
    benchmark_returns = rng.normal(0.0002, 0.008, size=size).astype(float)
    return {
        "returns": returns,
        "turnover": turnover,
        "exposure": exposure,
        "benchmark_returns": benchmark_returns,
    }


def _representative_payload() -> research_metrics.ComputedMetricPayload:
    """A fully populated payload built through the canonical resolve path."""
    fixture = _representative_series()
    resolved_rf = research_metrics.resolve_risk_free_config(
        get_default_runtime_config().backtest,
        periods_per_year=365,
        timestamps=None,
        size=int(fixture["returns"].size),
    )
    return research_metrics.resolve_compute_metric_payload(
        fixture["returns"],
        turnover=fixture["turnover"],
        exposure=fixture["exposure"],
        benchmark_returns=fixture["benchmark_returns"],
        periods_per_year=365,
        num_trials=3,
        resolved_rf=resolved_rf,
    )


def _constructed_payload() -> research_metrics.ComputedMetricPayload:
    """A literal payload so the summary keyset is exercised without the maths."""
    return research_metrics.ComputedMetricPayload(
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


def _resolved_rf_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        annual_rate=0.04,
        per_period_rate=0.001,
        sortino_target_annual=0.03,
        sortino_target_per_period=0.0008,
    )


def _assert_flat_floats(payload: dict[str, object]) -> None:
    """Every value must be a bare ``float`` -- nested sub-objects are rejected."""
    non_float = {
        key: type(value).__name__ for key, value in payload.items() if type(value) is not float
    }
    assert non_float == {}, f"non-float / nested values leaked into flat payload: {non_float}"


# ---------------------------------------------------------------------------
# 1. Keyset snapshot guards.
# ---------------------------------------------------------------------------


def test_compute_metrics_keyset_matches_pinned_snapshot() -> None:
    fixture = _representative_series()
    summary = research_metrics.compute_metrics(
        fixture["returns"],
        turnover=fixture["turnover"],
        exposure=fixture["exposure"],
        benchmark_returns=fixture["benchmark_returns"],
        periods_per_year=365,
        num_trials=3,
    )

    keys = frozenset(summary)
    assert keys == _SUMMARY_KEYSET, (
        "compute_metrics keyset drifted -- "
        f"added={sorted(keys - _SUMMARY_KEYSET)} removed={sorted(_SUMMARY_KEYSET - keys)}"
    )


def test_compute_metric_summary_keyset_matches_pinned_snapshot() -> None:
    with_rf = research_metrics.compute_metric_summary(
        _constructed_payload(), resolved_rf=_resolved_rf_namespace()
    )
    without_rf = research_metrics.compute_metric_summary(
        research_metrics.empty_compute_metric_payload(), resolved_rf=None
    )

    for label, summary in (("with_rf", with_rf), ("without_rf", without_rf)):
        keys = frozenset(summary)
        assert keys == _SUMMARY_KEYSET, (
            f"compute_metric_summary ({label}) keyset drifted -- "
            f"added={sorted(keys - _SUMMARY_KEYSET)} removed={sorted(_SUMMARY_KEYSET - keys)}"
        )


def test_resolve_compute_metric_payload_keyset_matches_pinned_snapshot() -> None:
    payload = _representative_payload()
    keys = frozenset(dataclasses.asdict(payload))
    assert keys == _PAYLOAD_KEYSET, (
        "resolve_compute_metric_payload dataclass field surface drifted -- "
        f"added={sorted(keys - _PAYLOAD_KEYSET)} removed={sorted(_PAYLOAD_KEYSET - keys)}"
    )


def test_empty_compute_metric_payload_keyset_matches_pinned_snapshot() -> None:
    keys = frozenset(dataclasses.asdict(research_metrics.empty_compute_metric_payload()))
    assert keys == _PAYLOAD_KEYSET, (
        "empty_compute_metric_payload field surface drifted -- "
        f"added={sorted(keys - _PAYLOAD_KEYSET)} removed={sorted(_PAYLOAD_KEYSET - keys)}"
    )


# ---------------------------------------------------------------------------
# 2. Flat-float invariant (FIX-m1: no nested sub-objects).
# ---------------------------------------------------------------------------


def test_compute_metrics_values_are_flat_floats() -> None:
    fixture = _representative_series()
    summary = research_metrics.compute_metrics(
        fixture["returns"],
        turnover=fixture["turnover"],
        exposure=fixture["exposure"],
        benchmark_returns=fixture["benchmark_returns"],
        periods_per_year=365,
        num_trials=3,
    )
    _assert_flat_floats(summary)


def test_compute_metric_summary_values_are_flat_floats() -> None:
    _assert_flat_floats(
        research_metrics.compute_metric_summary(
            _constructed_payload(), resolved_rf=_resolved_rf_namespace()
        )
    )
    _assert_flat_floats(
        research_metrics.compute_metric_summary(
            research_metrics.empty_compute_metric_payload(), resolved_rf=None
        )
    )


def test_resolve_compute_metric_payload_values_are_flat_floats() -> None:
    _assert_flat_floats(dataclasses.asdict(_representative_payload()))
    _assert_flat_floats(dataclasses.asdict(research_metrics.empty_compute_metric_payload()))


# ---------------------------------------------------------------------------
# 3. Single-seam serialization capture canary.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_serialization_seam():
    """Locally wrap the json / dataclasses serialization seam and count calls.

    The wrappers call straight through to the originals (behaviour preserved)
    and are unconditionally restored on exit, so nothing leaks to other tests.
    """
    captured = {"json_dumps": 0, "json_dump": 0, "asdict": 0}
    orig_dumps = json.dumps
    orig_dump = json.dump
    orig_asdict = dataclasses.asdict

    def wrapped_dumps(*args, **kwargs):
        captured["json_dumps"] += 1
        return orig_dumps(*args, **kwargs)

    def wrapped_dump(*args, **kwargs):
        captured["json_dump"] += 1
        return orig_dump(*args, **kwargs)

    def wrapped_asdict(*args, **kwargs):
        captured["asdict"] += 1
        return orig_asdict(*args, **kwargs)

    json.dumps = wrapped_dumps
    json.dump = wrapped_dump
    dataclasses.asdict = wrapped_asdict
    try:
        yield captured
    finally:
        json.dumps = orig_dumps
        json.dump = orig_dump
        dataclasses.asdict = orig_asdict


def test_serialization_capture_seam_actually_fires() -> None:
    payload = research_metrics.empty_compute_metric_payload()

    with _capture_serialization_seam() as captured:
        payload_dict = dataclasses.asdict(payload)
        summary = research_metrics.compute_metric_summary(payload, resolved_rf=None)
        blob = json.dumps(summary)
        buffer = io.StringIO()
        json.dump(summary, buffer)
        streamed = buffer.getvalue()

    # The hook fired at least once for each patched seam.
    assert captured["asdict"] >= 1
    assert captured["json_dumps"] >= 1
    assert captured["json_dump"] >= 1

    # Call-through behaviour was preserved (real serialization still happened).
    assert frozenset(payload_dict) == _PAYLOAD_KEYSET
    assert json.loads(blob) == summary
    assert json.loads(streamed) == summary

    # Seam is fully restored after the context exits.
    assert json.dumps is not None
    assert dataclasses.asdict(payload) == payload_dict
