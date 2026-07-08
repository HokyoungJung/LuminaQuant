"""Tests for the ``research_config`` activation seam on ``run_candidate_research``.

Context: the strict-research profile (``configs/profiles/research.yaml`` /
``backtest_cost_realistic.yaml``) turns on
``research.strict_selection_gate/use_lockbox_split/purge_embargo_bars/
single_correlation_discount/hac_inference/cscv_pbo`` on ``ResearchConfig``, but
nothing previously read those fields back out of the config object and threaded
them into ``run_candidate_research``'s own ``split`` / ``score_config`` kwargs
(the only places ``research_runner`` actually consults them). This file pins:

* ``research_config_to_overrides`` -- the pure, independently-testable mapping
  from a ``ResearchConfig``/``RuntimeConfig`` to the four override dicts.
* ``research_config=None`` (the default) is a strict no-op: the report is
  byte-identical to never having passed the parameter at all.
* ``research_config=<config with flags on>`` actually activates the lockbox
  split end-to-end (observable via the reported OOS window shrinking to the
  lockbox-carved half) and threads ``hac_inference`` into the per-candidate
  DSR call (``research_metrics.deflated_sharpe_ratio``, the function this
  pipeline actually invokes -- see NOTE below).
* Caller-provided ``split`` / ``score_config`` keys always win over the
  config-derived value.
* The specific bug the post-resolution-merge design avoids: merging the
  lockbox/purge override keys into a bare ``split`` mapping BEFORE
  ``_resolve_split_config`` runs would make an otherwise-empty mapping look
  like "exact dates supplied" and silently drop the rolling-default window.

NOTE on scope: the task that motivated this seam described the DSR kwargs as
flowing through ``lumina_quant.research.survivorship.deflated_sharpe_ratio``.
In this codebase that function (and ``research.alpha_search.run_alpha_search``,
which is its only caller) is a *separate* pipeline from
``run_candidate_research``/``research_runner`` -- neither is ever imported or
called from ``research_runner.py``/``research_entrypoints.py``/
``research_run_support.py`` (verified by grep: zero hits for "survivorship",
"cscv", "effective_trials" in those three files). The per-candidate DSR actually
invoked by ``run_candidate_research`` is
``lumina_quant.strategy_factory.research_metrics.deflated_sharpe_ratio``, a
distinct function with a ``hac_inference`` kwarg but no
``single_correlation_discount``. This suite therefore verifies ``hac_inference``
reaching *that* function (its real, live sink) and verifies
``single_correlation_discount`` / ``cscv_pbo`` are still faithfully surfaced by
the helper (for the separate alpha-search pipeline to consume) without
fabricating a call path that does not exist in this module.

NOTE on module resolution: some other test in this repository's wider suite
replaces ``sys.modules["lumina_quant.strategy_factory.research_runner"]`` with
a fresh module object during collection (observed empirically: a module-level
``research_runner`` name bound at THIS file's import time can end up distinct,
by identity, from ``sys.modules[...]`` by the time tests actually run). A
``monkeypatch.setattr`` on a stale module object silently has no effect on the
code path ``run_candidate_research`` actually executes. To stay robust
regardless of test order/selection, every monkeypatch target and constructed
value below is resolved fresh via ``importlib.import_module`` inside the test
body rather than through a module-level import.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from lumina_quant.configuration.schema import ResearchConfig, RuntimeConfig
from lumina_quant.strategy_factory import research_entrypoints, research_run_support

_START = datetime(2026, 1, 1, tzinfo=UTC)
_SPLIT = {
    "train_start": "2026-01-01",
    "train_end": "2026-01-31",
    "val_start": "2026-02-01",
    "val_end": "2026-02-28",
    "oos_start": "2026-03-01",
    "oos_end": "2026-03-16",
    "strategy_timeframe": "1d",
    "mode": "exact_dates",
}
_CANDIDATE = {
    "candidate_id": "seam-1",
    "name": "seam-1",
    "strategy_class": "CompositeTrendStrategy",
    "strategy_timeframe": "1d",
    "symbols": ["BTC/USDT"],
    "params": {},
}


def _rr():
    """Fresh, current ``research_runner`` module (see module docstring NOTE)."""
    return importlib.import_module("lumina_quant.strategy_factory.research_runner")


def _rm():
    """Fresh, current ``research_metrics`` module (see module docstring NOTE)."""
    return importlib.import_module("lumina_quant.strategy_factory.research_metrics")


def _bundle(rr, symbol: str, timestamps: list[datetime]):
    closes = np.linspace(100.0, 110.0, len(timestamps), dtype=float)
    return rr.SeriesBundle(
        symbol=symbol,
        timeframe="1d",
        datetime=np.asarray([np.datetime64(ts.replace(tzinfo=None), "ms") for ts in timestamps]),
        open=closes - 0.1,
        high=closes + 0.2,
        low=closes - 0.2,
        close=closes,
        volume=np.full(len(timestamps), 1_000.0, dtype=float),
    )


def _patch_synthetic_pipeline(monkeypatch) -> None:
    rr = _rr()
    timestamps = [_START + timedelta(days=offset) for offset in range(420)]
    bundle = _bundle(rr, "BTC/USDT", timestamps)

    def _mock_load_bundle_cache(*, symbols, timeframes, start_date=None, end_date=None, **kwargs):
        _ = symbols, timeframes, start_date, end_date, kwargs
        return {("BTC/USDT", "1d"): bundle}, {
            "parquet": ["BTC/USDT@1d"],
            "csv": [],
            "synthetic": [],
        }

    def _mock_load_feature_cache(*, symbols, start_date=None, end_date=None, **kwargs):
        _ = symbols, start_date, end_date, kwargs
        return {}

    def _mock_strategy_signal(candidate, *, aligned, symbols):
        _ = candidate, symbols
        length = len(aligned["datetime"])
        rng = np.random.default_rng(7)
        returns = 0.001 + 0.0006 * rng.standard_normal(length)
        turnover = np.full(length, 0.1, dtype=float)
        exposure = np.ones(length, dtype=float)
        return returns, turnover, exposure, {}

    monkeypatch.setattr(rr, "_load_bundle_cache", _mock_load_bundle_cache)
    monkeypatch.setattr(rr, "_load_feature_cache", _mock_load_feature_cache)
    monkeypatch.setattr(rr, "_strategy_signal", _mock_strategy_signal)


def _run(monkeypatch, *, research_config: Any = None, split: dict | None = None) -> dict:
    _patch_synthetic_pipeline(monkeypatch)
    return research_entrypoints.run_candidate_research(
        candidates=[dict(_CANDIDATE)],
        strategy_timeframes=["1d"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=8,
        split=dict(split if split is not None else _SPLIT),
        research_config=research_config,
    )


# ---------------------------------------------------------------------------
# research_config_to_overrides: pure-mapping unit tests.
# ---------------------------------------------------------------------------


def test_overrides_are_all_empty_for_none():
    overrides = research_run_support.research_config_to_overrides(None)
    assert overrides == {
        "split": {},
        "score_config_research": {},
        "deflation_kwargs": {},
        "robust_score_params": {},
    }


def test_overrides_shape_for_fully_enabled_research_config():
    cfg = ResearchConfig(
        strict_selection_gate=True,
        use_lockbox_split=True,
        purge_embargo_bars=5,
        single_correlation_discount=True,
        hac_inference=True,
        cscv_pbo=True,
    )
    overrides = research_run_support.research_config_to_overrides(cfg)
    assert overrides["split"] == {"use_lockbox_split": True, "purge_embargo_bars": 5}
    # The reject-gate floors are emitted at their (legacy no-op) defaults because
    # this fixture leaves them unset; only strict_selection_gate is flipped on.
    assert overrides["score_config_research"] == {
        "strict_selection_gate": True,
        "enforce_selection_reject_gate": False,
        "dsr_gate_floor": 0.0,
        "spa_gate_ceiling": 1.0,
        "pbo_gate_ceiling": 1.0,
        "max_cross_trial_pbo": 1.0,
        "cost_rate_multiplier": 1.0,
        "cost_rate_bps_override": None,
    }
    assert overrides["deflation_kwargs"] == {
        "single_correlation_discount": True,
        "hac_inference": True,
        "cscv_pbo": True,
    }
    assert overrides["robust_score_params"] == {
        "strict_selection_gate": True,
        "enforce_selection_reject_gate": False,
        "dsr_gate_floor": 0.0,
        "spa_gate_ceiling": 1.0,
        "pbo_gate_ceiling": 1.0,
    }


def test_overrides_unwrap_runtime_config():
    cfg = ResearchConfig(use_lockbox_split=True, purge_embargo_bars=2)
    via_runtime = research_run_support.research_config_to_overrides(RuntimeConfig(research=cfg))
    via_research = research_run_support.research_config_to_overrides(cfg)
    assert via_runtime == via_research


def test_overrides_all_flags_off_still_reports_false_not_empty():
    # A real (default) ResearchConfig has every field present -- just at its
    # legacy-off value -- so the helper reports explicit False/0, not an
    # absent key. Only ``research_config=None`` produces empty dicts.
    overrides = research_run_support.research_config_to_overrides(ResearchConfig())
    assert overrides["split"] == {"use_lockbox_split": False, "purge_embargo_bars": 0}
    assert overrides["score_config_research"] == {
        "strict_selection_gate": False,
        "enforce_selection_reject_gate": False,
        "dsr_gate_floor": 0.0,
        "spa_gate_ceiling": 1.0,
        "pbo_gate_ceiling": 1.0,
        "max_cross_trial_pbo": 1.0,
        "cost_rate_multiplier": 1.0,
        "cost_rate_bps_override": None,
    }
    assert overrides["deflation_kwargs"] == {
        "single_correlation_discount": False,
        "hac_inference": False,
        "cscv_pbo": False,
    }
    assert overrides["robust_score_params"] == {
        "strict_selection_gate": False,
        "enforce_selection_reject_gate": False,
        "dsr_gate_floor": 0.0,
        "spa_gate_ceiling": 1.0,
        "pbo_gate_ceiling": 1.0,
    }


# ---------------------------------------------------------------------------
# apply_split_overrides: the post-resolution-merge safety property.
# ---------------------------------------------------------------------------


def test_apply_split_overrides_noop_when_overrides_empty():
    resolved = _rr()._resolve_split_config(dict(_SPLIT), strategy_timeframe="1d")
    merged = research_run_support.apply_split_overrides(resolved, {})
    assert merged == resolved


def test_apply_split_overrides_preserves_rolling_default_when_split_is_none():
    """The critical bug this design avoids.

    Injecting ``use_lockbox_split`` into a bare mapping BEFORE calling
    ``_resolve_split_config`` would make an otherwise-empty ``split`` look like
    "exact dates supplied" to ``_resolve_split_config`` (any ``Mapping``, even
    one with no date fields, skips the rolling-window default), silently
    dropping the rolling 360/150/60-day window whenever a caller sets only the
    research-profile flags and no explicit dates. Applying the override AFTER
    resolution (this function's contract) sidesteps that: the rolling default
    is computed first from the untouched ``None``, and the flags are layered
    on top afterward.
    """
    rr = _rr()
    resolved_from_none = rr._resolve_split_config(None, strategy_timeframe="1d")
    assert resolved_from_none["mode"] == "rolling_default"
    assert resolved_from_none["train_start"] is not None

    merged = research_run_support.apply_split_overrides(
        resolved_from_none, {"use_lockbox_split": True, "purge_embargo_bars": 3}
    )
    assert merged["mode"] == "rolling_default"
    assert merged["train_start"] == resolved_from_none["train_start"]
    assert merged["oos_end"] == resolved_from_none["oos_end"]
    assert merged["use_lockbox_split"] is True
    assert merged["purge_embargo_bars"] == 3

    # Demonstrate the failure mode a naive pre-resolution merge would hit, so
    # the regression is pinned even if someone "simplifies" the call site back
    # to merging before resolution.
    naive_pre_merge = {"use_lockbox_split": True}
    broken = rr._resolve_split_config(naive_pre_merge, strategy_timeframe="1d")
    assert broken["mode"] == "exact_dates"
    assert broken["train_start"] is None


def test_apply_split_overrides_caller_value_wins():
    resolved = _rr()._resolve_split_config(
        {**_SPLIT, "use_lockbox_split": False}, strategy_timeframe="1d"
    )
    merged = research_run_support.apply_split_overrides(resolved, {"use_lockbox_split": True})
    assert merged["use_lockbox_split"] is False


# ---------------------------------------------------------------------------
# run_candidate_research(research_config=None) is byte-identical.
# ---------------------------------------------------------------------------


def test_run_candidate_research_default_omits_research_config_param(monkeypatch):
    """Calling without ``research_config`` at all == passing ``research_config=None``."""
    _patch_synthetic_pipeline(monkeypatch)
    kwargs = dict(
        candidates=[dict(_CANDIDATE)],
        strategy_timeframes=["1d"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=8,
        split=dict(_SPLIT),
    )
    omitted = research_entrypoints.run_candidate_research(**kwargs)
    _patch_synthetic_pipeline(monkeypatch)
    explicit_none = research_entrypoints.run_candidate_research(research_config=None, **kwargs)
    # ``generated_at`` is a wall-clock timestamp stamped fresh on every call
    # (unrelated to ``research_config``); everything else must match exactly.
    omitted.pop("generated_at")
    explicit_none.pop("generated_at")
    assert omitted == explicit_none


def test_run_candidate_research_none_derives_no_overrides(monkeypatch):
    report = _run(monkeypatch, research_config=None)
    assert "use_lockbox_split" not in report["split"]
    assert "purge_embargo_bars" not in report["split"]
    assert "research" not in report["scoring_config"]
    row = report["candidates"][0]
    assert len(row["return_streams"]["oos"]) == 16  # full OOS window, no lockbox carve


# ---------------------------------------------------------------------------
# Lockbox split actually activates end-to-end.
# ---------------------------------------------------------------------------


def test_run_candidate_research_lockbox_flag_activates(monkeypatch):
    report = _run(monkeypatch, research_config=ResearchConfig(use_lockbox_split=True))
    assert report["split"]["use_lockbox_split"] is True
    row = report["candidates"][0]
    # The reported OOS window carves its tail half into the lockbox, so the
    # observable OOS return-stream shrinks from the full 16-day window to the
    # front 8-day half; the remaining 8 days become the never-touched lockbox.
    assert len(row["return_streams"]["oos"]) == 8


def test_run_candidate_research_lockbox_creates_a_distinct_metric_segment(monkeypatch):
    """Direct spy proving a genuine 4th (lockbox) segment is computed."""
    rr = _rr()
    original = rr._evaluate_candidate_metric_payload
    captured: list[Any] = []

    def _spy(*args, **kwargs):
        payload = original(*args, **kwargs)
        captured.append(payload)
        return payload

    monkeypatch.setattr(rr, "_evaluate_candidate_metric_payload", _spy)
    _run(monkeypatch, research_config=ResearchConfig(use_lockbox_split=True))
    assert captured
    assert any(payload is not None and payload.lockbox_metrics is not None for payload in captured)


def test_run_candidate_research_lockbox_off_by_default_no_extra_segment(monkeypatch):
    rr = _rr()
    original = rr._evaluate_candidate_metric_payload
    captured: list[Any] = []

    def _spy(*args, **kwargs):
        payload = original(*args, **kwargs)
        captured.append(payload)
        return payload

    monkeypatch.setattr(rr, "_evaluate_candidate_metric_payload", _spy)
    _run(monkeypatch, research_config=None)
    assert captured
    assert all(payload is None or payload.lockbox_metrics is None for payload in captured)


# ---------------------------------------------------------------------------
# hac_inference threads into the per-candidate DSR call.
# ---------------------------------------------------------------------------


def test_run_candidate_research_hac_inference_reaches_deflated_sharpe_ratio(monkeypatch):
    rm = _rm()
    real_dsr = rm.deflated_sharpe_ratio
    captured: list[dict[str, Any]] = []

    def _spy_dsr(returns, **kwargs):
        captured.append(kwargs)
        return real_dsr(returns, **kwargs)

    monkeypatch.setattr(rm, "deflated_sharpe_ratio", _spy_dsr)
    _run(monkeypatch, research_config=ResearchConfig(hac_inference=True))
    assert captured
    assert any(call.get("hac_inference") is True for call in captured)


def test_run_candidate_research_hac_inference_off_by_default(monkeypatch):
    rm = _rm()
    real_dsr = rm.deflated_sharpe_ratio
    captured: list[dict[str, Any]] = []

    def _spy_dsr(returns, **kwargs):
        captured.append(kwargs)
        return real_dsr(returns, **kwargs)

    monkeypatch.setattr(rm, "deflated_sharpe_ratio", _spy_dsr)
    _run(monkeypatch, research_config=None)
    assert captured
    assert all(call.get("hac_inference", False) is False for call in captured)


# ---------------------------------------------------------------------------
# strict_selection_gate reaches score_config['research'] (already-existing
# _research_flag consumer; this pins that the NEW seam actually feeds it).
# ---------------------------------------------------------------------------


def test_run_candidate_research_strict_selection_gate_reaches_scoring_config(monkeypatch):
    report = _run(monkeypatch, research_config=ResearchConfig(strict_selection_gate=True))
    assert report["scoring_config"]["research"]["strict_selection_gate"] is True


# ---------------------------------------------------------------------------
# Caller-provided values always win over the config-derived override.
# ---------------------------------------------------------------------------


def test_run_candidate_research_caller_split_flag_wins_over_config(monkeypatch):
    report = _run(
        monkeypatch,
        research_config=ResearchConfig(use_lockbox_split=True),
        split={**_SPLIT, "use_lockbox_split": False},
    )
    assert report["split"]["use_lockbox_split"] is False
    row = report["candidates"][0]
    assert len(row["return_streams"]["oos"]) == 16


def test_run_candidate_research_caller_score_config_wins_over_config(monkeypatch):
    _patch_synthetic_pipeline(monkeypatch)
    report = research_entrypoints.run_candidate_research(
        candidates=[dict(_CANDIDATE)],
        strategy_timeframes=["1d"],
        symbol_universe=["BTC/USDT"],
        stage1_keep_ratio=1.0,
        max_candidates=8,
        split=dict(_SPLIT),
        score_config={"research": {"strict_selection_gate": False}},
        research_config=ResearchConfig(strict_selection_gate=True),
    )
    assert report["scoring_config"]["research"]["strict_selection_gate"] is False
