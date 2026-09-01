import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumina_quant.strategies.alpha_max_research_sleeves import (
    ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lumina_quant"
    / "strategies"
    / "artifact_portfolio_mode.py"
)
SPEC = importlib.util.spec_from_file_location("artifact_portfolio_mode_capsules", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _component(
    component_id: str,
    symbols=("BNB/USDT",),
    strategy_class="CapsuleChild",
    params=None,
):
    return MODULE.PortfolioModeComponent(
        component_id=component_id,
        label=component_id,
        strategy_class=strategy_class,
        symbols=tuple(symbols),
        params=dict(params or {}),
        weight=1.0,
        source="test",
    )


def _patch_definition(monkeypatch, components, *, receipts=()):
    definition = MODULE.PortfolioModeDefinition(
        portfolio_mode="hybrid_guarded_mode",
        components=tuple(components),
        cash_weight=0.0,
        source_artifacts={},
        artifact_read_receipts=receipts,
    )
    monkeypatch.setattr(MODULE, "resolve_portfolio_mode_definition", lambda _mode: definition)
    return definition


def _bars():
    return SimpleNamespace(
        symbol_list=["BNB/USDT", "ETH/USDT"],
        get_latest_bar_value=lambda *args, **kwargs: 100.0,
    )


class _CapsuleChild:
    instances = []

    def __init__(self, bars, events, **params):
        self.symbols = tuple(bars.symbol_list)
        self.events = events
        self.params = dict(params)
        self.state = {"symbols": list(self.symbols), "value": len(self.instances)}
        self.finalize_calls = []
        type(self).instances.append(self)

    def calculate_signals(self, event):
        _ = event

    def get_research_indicator_state(self):
        return dict(self.state)

    def set_research_indicator_state(self, capsule):
        self.state = dict(capsule)

    def get_native_finalization_rollback_state(self):
        return {
            "state": dict(self.state),
            "finalize_calls": list(self.finalize_calls),
        }

    def set_native_finalization_rollback_state(self, snapshot):
        self.state = dict(snapshot["state"])
        self.finalize_calls = list(snapshot["finalize_calls"])

    def finalize_completed_native_buckets(self, watermark):
        self.finalize_calls.append(watermark)
        return {"watermark": str(watermark), "symbols": list(self.symbols)}


class _NonCapableChild:
    def __init__(self, bars, events, **params):
        _ = bars, events, params

    def calculate_signals(self, event):
        _ = event


class _FailingRestoreChild(_CapsuleChild):
    fail_on_restore = False

    def set_research_indicator_state(self, capsule):
        self.state = dict(capsule)
        if self.fail_on_restore and "sentinel" not in capsule:
            raise RuntimeError("restore failed")


class _WarmupReadyChild(_CapsuleChild):
    def __init__(self, bars, events, **params):
        super().__init__(bars, events, **params)
        self.warmup_ready_calls = 0

    def validate_research_warmup_ready(self):
        self.warmup_ready_calls += 1


class _Release4hChild(_CapsuleChild):
    native_timeframe = "4h"
    native_timeframe_ms = 14_400_000

    def calculate_signals_context(self, context):
        self.state["release"] = {
            "time": context.event.time,
            "symbols": list(context.event.bars_1s),
            "lookup_id": id(context.feature_lookup),
            "completed": context.event.completed_native_bars,
        }


class _Release1dChild(_CapsuleChild):
    native_timeframe = "1d"
    native_timeframe_ms = 86_400_000

    def calculate_signals_context(self, context):
        self.state["release"] = {
            "time": context.event.time,
            "symbols": list(context.event.bars_1s),
            "lookup_id": id(context.feature_lookup),
            "completed": context.event.completed_native_bars,
        }


class _MutatingFinalizeChild(_CapsuleChild):
    def finalize_completed_native_buckets(self, watermark):
        self.finalize_calls.append(watermark)
        self.state["finalized"] = int(self.state.get("finalized", 0)) + 1
        self.events.put(("new-signal", tuple(self.symbols)))
        return 1


class _FailingMutatingFinalizeChild(_MutatingFinalizeChild):
    def finalize_completed_native_buckets(self, watermark):
        super().finalize_completed_native_buckets(watermark)
        raise RuntimeError("native finalization failed")


def _install_child_mapping(monkeypatch, mapping):
    def _resolve(name, default_name=None):
        _ = default_name
        return mapping[name]

    monkeypatch.setattr(MODULE, "resolve_strategy_class", _resolve)


def _strategy(monkeypatch, components, mapping=None, **kwargs):
    _CapsuleChild.instances = []
    _FailingRestoreChild.instances = []
    _FailingRestoreChild.fail_on_restore = False
    _patch_definition(monkeypatch, components)
    _install_child_mapping(monkeypatch, mapping or {"CapsuleChild": _CapsuleChild})
    return MODULE.ArtifactPortfolioModeStrategy(
        bars=_bars(),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode="hybrid_guarded_mode",
        **kwargs,
    )


def _refresh_outer_hash(capsule):
    capsule["sha256"] = MODULE._capsule_payload_sha256(capsule)
    return capsule


def _child_states(strategy):
    return [dict(child.state) for _component, child, _queue in strategy._children]


def _canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def test_portfolio_mode_decision_cadence_defaults_to_60_and_accepts_explicit_1(monkeypatch):
    default_strategy = _strategy(monkeypatch, [_component("comp-a")])
    assert default_strategy.decision_cadence_seconds == 60
    assert _CapsuleChild.instances[-1].params == {}

    explicit_strategy = _strategy(
        monkeypatch,
        [_component("comp-a")],
        decision_cadence_seconds=1,
    )
    assert explicit_strategy.decision_cadence_seconds == 1
    assert _CapsuleChild.instances[-1].params == {}

    with pytest.raises(ValueError, match="decision_cadence_seconds"):
        _strategy(monkeypatch, [_component("comp-a")], decision_cadence_seconds=0)


def test_completed_native_release_prevalidates_and_uses_manifest_order(monkeypatch) -> None:
    strategy = _strategy(
        monkeypatch,
        [
            _component("daily", ("ETH/USDT",), strategy_class="Release1d"),
            _component("carry", ("BNB/USDT",), strategy_class="Release4h"),
        ],
        mapping={"Release1d": _Release1dChild, "Release4h": _Release4hChild},
    )
    lookup = object()
    release_ms = int(datetime(2026, 1, 2, tzinfo=UTC).timestamp() * 1000)
    bar = (datetime(2026, 1, 1), 100.0, 101.0, 99.0, 100.5, 10.0)
    strategy.calculate_signals_completed_native_release(
        release_timestamp_ms=release_ms,
        bars_by_timeframe={
            "4h": {"BNB/USDT": bar},
            "1d": {"ETH/USDT": bar},
        },
        feature_lookup=lookup,
    )
    by_id = {
        component.component_id: child.state["release"]
        for component, child, _queue in strategy._children
    }
    assert by_id == {
        "daily": {
            "time": release_ms,
            "symbols": ["ETH/USDT"],
            "lookup_id": id(lookup),
            "completed": True,
        },
        "carry": {
            "time": release_ms,
            "symbols": ["BNB/USDT"],
            "lookup_id": id(lookup),
            "completed": True,
        },
    }

    before = _child_states(strategy)
    with pytest.raises(ValueError, match="symbols"):
        strategy.calculate_signals_completed_native_release(
            release_timestamp_ms=release_ms,
            bars_by_timeframe={
                "4h": {"ETH/USDT": bar},
                "1d": {"ETH/USDT": bar},
            },
            feature_lookup=lookup,
        )
    assert _child_states(strategy) == before


def test_portfolio_mode_rejects_duplicate_component_ids_only_for_capsule_mode(monkeypatch):
    strategy = _strategy(monkeypatch, [_component("dup"), _component("dup")])

    assert len(strategy._children) == 2
    with pytest.raises(ValueError, match="duplicate research indicator child ids"):
        strategy.get_research_indicator_state()


def test_research_indicator_capsule_is_canonical_and_restores_by_component_id(monkeypatch):
    strategy = _strategy(
        monkeypatch,
        [
            _component("b-child", ("ETH/USDT",)),
            _component("a-child", ("BNB/USDT",)),
        ],
    )
    capsule = strategy.get_research_indicator_state()

    assert capsule["kind"] == "artifact_portfolio_mode.research_indicator_state"
    assert [child["component_id"] for child in capsule["children"]] == [
        "a-child",
        "b-child",
    ]
    assert capsule["sha256"] == MODULE._capsule_payload_sha256(capsule)

    restored = _strategy(
        monkeypatch,
        [
            _component("b-child", ("ETH/USDT",)),
            _component("a-child", ("BNB/USDT",)),
        ],
    )
    restored.set_research_indicator_state(capsule)

    by_id = {component.component_id: child.state for component, child, _queue in restored._children}
    assert by_id["a-child"]["symbols"] == ["BNB/USDT"]
    assert by_id["b-child"]["symbols"] == ["ETH/USDT"]


def test_research_indicator_capsule_rejects_bad_child_sets_without_mutation(monkeypatch):
    strategy = _strategy(monkeypatch, [_component("a-child"), _component("b-child")])
    capsule = strategy.get_research_indicator_state()

    missing = dict(capsule)
    missing["children"] = capsule["children"][:1]
    _refresh_outer_hash(missing)

    extra = dict(capsule)
    extra["children"] = [*capsule["children"], dict(capsule["children"][0], component_id="x")]
    _refresh_outer_hash(extra)

    duplicate = dict(capsule)
    duplicate["children"] = [capsule["children"][0], dict(capsule["children"][0])]
    _refresh_outer_hash(duplicate)

    wrong_child_hash = dict(capsule)
    wrong_child_hash["children"] = [dict(item) for item in capsule["children"]]
    wrong_child_hash["children"][0]["sha256"] = "0" * 64
    _refresh_outer_hash(wrong_child_hash)

    for bad_capsule in (missing, extra, duplicate, wrong_child_hash, {"bad": "shape"}):
        before = _child_states(strategy)
        with pytest.raises(ValueError):
            strategy.set_research_indicator_state(bad_capsule)
        assert _child_states(strategy) == before


def test_research_indicator_capsule_rejects_non_capable_children(monkeypatch):
    strategy = _strategy(
        monkeypatch,
        [_component("non-capable", strategy_class="NonCapableChild")],
        mapping={"NonCapableChild": _NonCapableChild},
    )

    with pytest.raises(ValueError, match="not capable"):
        strategy.get_research_indicator_state()

    capsule = {
        "kind": "artifact_portfolio_mode.research_indicator_state",
        "schema_version": 1,
        "portfolio_mode": "hybrid_guarded_mode",
        "children": [
            {
                "component_id": "non-capable",
                "state": {},
                "sha256": MODULE._canonical_sha256({}),
            }
        ],
    }
    _refresh_outer_hash(capsule)
    with pytest.raises(ValueError, match="not capable"):
        strategy.set_research_indicator_state(capsule)


def test_research_indicator_restore_rolls_back_partial_child_mutation(monkeypatch):
    source = _strategy(
        monkeypatch,
        [
            _component("ok-child", strategy_class="CapsuleChild"),
            _component("fail-child", strategy_class="FailingRestoreChild"),
        ],
        mapping={"CapsuleChild": _CapsuleChild, "FailingRestoreChild": _FailingRestoreChild},
    )
    capsule = source.get_research_indicator_state()

    target = _strategy(
        monkeypatch,
        [
            _component("ok-child", strategy_class="CapsuleChild"),
            _component("fail-child", strategy_class="FailingRestoreChild"),
        ],
        mapping={"CapsuleChild": _CapsuleChild, "FailingRestoreChild": _FailingRestoreChild},
    )
    for component, child, _queue in target._children:
        child.state = {"sentinel": component.component_id}
    before = _child_states(target)
    assert before != [entry["state"] for entry in capsule["children"]]
    _FailingRestoreChild.fail_on_restore = True

    with pytest.raises(RuntimeError, match="restore failed"):
        target.set_research_indicator_state(capsule)

    assert _child_states(target) == before


def test_finalize_completed_native_buckets_fails_before_any_partial_child_mutation(monkeypatch):
    strategy = _strategy(
        monkeypatch,
        [
            _component("capable", strategy_class="CapsuleChild"),
            _component("legacy", strategy_class="NonCapableChild"),
        ],
        mapping={"CapsuleChild": _CapsuleChild, "NonCapableChild": _NonCapableChild},
    )

    capable_child = next(
        child
        for component, child, _queue in strategy._children
        if component.component_id == "capable"
    )
    with pytest.raises(ValueError, match="completed native bucket child is not capable: legacy"):
        strategy.finalize_completed_native_buckets("2026-07-01T00:00:00Z")
    assert capable_child.finalize_calls == []


@pytest.mark.parametrize(
    ("missing_method", "capability_name"),
    [
        (
            "get_native_finalization_rollback_state",
            "native finalization rollback getter",
        ),
        (
            "set_native_finalization_rollback_state",
            "native finalization rollback setter",
        ),
    ],
)
def test_finalize_completed_native_buckets_checks_rollback_capabilities_before_mutation(
    monkeypatch, missing_method, capability_name
):
    strategy = _strategy(monkeypatch, [_component("capable"), _component("legacy")])
    children = {component.component_id: child for component, child, _queue in strategy._children}
    monkeypatch.setattr(children["legacy"], missing_method, None)

    with pytest.raises(ValueError, match=f"{capability_name} child is not capable: legacy"):
        strategy.finalize_completed_native_buckets("2026-07-01T00:00:00Z")

    assert children["capable"].finalize_calls == []
    assert children["legacy"].finalize_calls == []


def test_finalize_completed_native_buckets_covers_exactly_all_child_ids(monkeypatch):
    strategy = _strategy(
        monkeypatch,
        [_component("b-child"), _component("a-child")],
    )

    result = strategy.finalize_completed_native_buckets("2026-07-01T00:00:00Z")

    assert sorted(result) == ["a-child", "b-child"]
    assert all(item["watermark"] == "2026-07-01T00:00:00Z" for item in result.values())


def test_finalize_completed_native_buckets_rejects_duplicate_ids_before_calls(monkeypatch):
    strategy = _strategy(monkeypatch, [_component("duplicate"), _component("duplicate")])

    with pytest.raises(ValueError, match="duplicate completed native bucket child ids"):
        strategy.finalize_completed_native_buckets("2026-07-01T00:00:00Z")
    assert all(child.finalize_calls == [] for _component, child, _queue in strategy._children)


def test_finalize_completed_native_buckets_rolls_back_child_state_and_queues(monkeypatch):
    strategy = _strategy(
        monkeypatch,
        [
            _component("first", strategy_class="MutatingFinalizeChild"),
            _component("second", strategy_class="FailingMutatingFinalizeChild"),
        ],
        mapping={
            "MutatingFinalizeChild": _MutatingFinalizeChild,
            "FailingMutatingFinalizeChild": _FailingMutatingFinalizeChild,
        },
    )
    before_states = _child_states(strategy)
    for component, _child, child_queue in strategy._children:
        child_queue.put(("existing-signal", component.component_id))

    with pytest.raises(RuntimeError, match="native finalization failed"):
        strategy.finalize_completed_native_buckets("2026-07-01T00:00:00Z")

    assert _child_states(strategy) == before_states
    assert [child_queue.drain() for _component, _child, child_queue in strategy._children] == [
        [("existing-signal", "first")],
        [("existing-signal", "second")],
    ]


def test_finalize_completed_native_buckets_rolls_back_all_near_high_state_byte_exact(
    monkeypatch,
):
    symbols = ("ADAUSDT", "BTCUSDT")
    strategy = _strategy(
        monkeypatch,
        [
            _component(
                "near-high",
                symbols=symbols,
                strategy_class="NearHighChild",
                params={
                    "admitted_symbols": symbols,
                    "high_lookback_bars": 20,
                    "min_history_bars": 5,
                    "vol_window": 2,
                    "quantile_pct": 0.5,
                    "rebalance_bars": 1,
                    "min_hold_bars": 0,
                    "min_symbols": 2,
                    "target_vol": 2.0,
                },
            ),
            _component("failing", strategy_class="FailingMutatingFinalizeChild"),
        ],
        mapping={
            "NearHighChild": ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy,
            "FailingMutatingFinalizeChild": _FailingMutatingFinalizeChild,
        },
    )
    children = {
        component.component_id: (child, child_queue)
        for component, child, child_queue in strategy._children
    }
    near_high, near_high_queue = children["near-high"]
    _failing, failing_queue = children["failing"]

    start = datetime(2026, 1, 1, tzinfo=UTC)
    historical_closes = {
        "ADAUSDT": (100.0, 110.0, 90.0, 120.0),
        "BTCUSDT": (100.0, 90.0, 110.0, 80.0),
    }

    def _bar(when, close, high):
        return (when, close, high, close * 0.99, close, 1_000.0)

    running_highs = dict.fromkeys(symbols, 0.0)
    for day in range(4):
        when = start + timedelta(days=day)
        bars = {}
        for symbol in symbols:
            close = historical_closes[symbol][day]
            running_highs[symbol] = max(running_highs[symbol], close)
            bars[symbol] = _bar(when, close, running_highs[symbol])
        near_high.calculate_signals_window(
            SimpleNamespace(
                type="MARKET_WINDOW",
                time=when,
                completed_native_bars=True,
                bars_1s={symbol: (bar,) for symbol, bar in bars.items()},
            )
        )
    assert near_high_queue.snapshot() == []

    complete_time = start + timedelta(days=4)
    forming_time = start + timedelta(days=5)
    complete_closes = {"ADAUSDT": 100.0, "BTCUSDT": 120.0}
    forming_closes = {"ADAUSDT": 119.0, "BTCUSDT": 82.0}

    class _NativeAggregator:
        def __init__(self):
            self.data = {}

        def get_bars(self, symbol, timeframe, lookback_bars=1, *, n=None):
            rows = self.data.get((symbol, timeframe), [])
            count = lookback_bars if n is None else n
            return rows[-int(count) :]

    aggregator = _NativeAggregator()
    for symbol in symbols:
        running_highs[symbol] = max(running_highs[symbol], complete_closes[symbol])
        completed = _bar(complete_time, complete_closes[symbol], running_highs[symbol])
        running_highs[symbol] = max(running_highs[symbol], forming_closes[symbol])
        forming = _bar(forming_time, forming_closes[symbol], running_highs[symbol])
        aggregator.data[(symbol, "1d")] = [completed, forming]

    # Bind the real aggregator without promoting the boundary bar yet.
    near_high.calculate_signals_window(
        SimpleNamespace(type="MARKET_WINDOW", time=complete_time, bars_1s={}),
        aggregator,
    )
    assert near_high._alpha_max_bound_aggregator is aggregator

    near_high._alpha_max_failed_native_keys["sentinel-key"] = "sentinel-failure"
    near_high._alpha_max_partial_bucket_error = "sentinel-partial"
    near_high_queue.put(("existing-signal", "near-high"))
    failing_queue.put(("existing-signal", "failing"))

    before_snapshot = near_high.get_native_finalization_rollback_state()
    before_state = before_snapshot["state"]
    before_state_bytes = _canonical_bytes(before_snapshot)
    before_queue_bytes = {
        component.component_id: _canonical_bytes(child_queue.snapshot())
        for component, _child, child_queue in strategy._children
    }
    observed = {}
    original_finalizer = near_high.finalize_completed_native_buckets

    def _recording_finalizer(watermark):
        result = original_finalizer(watermark)
        observed["result"] = result
        observed["state_bytes"] = _canonical_bytes(
            near_high.get_native_finalization_rollback_state()
        )
        observed["queue"] = near_high_queue.snapshot()
        return result

    monkeypatch.setattr(near_high, "finalize_completed_native_buckets", _recording_finalizer)

    with pytest.raises(RuntimeError, match="native finalization failed"):
        strategy.finalize_completed_native_buckets(forming_time)

    assert observed["result"] == 1
    assert observed["state_bytes"] != before_state_bytes
    assert len(observed["queue"]) > 1
    assert any(getattr(item, "type", None) == "SIGNAL" for item in observed["queue"])

    restored_snapshot = near_high.get_native_finalization_rollback_state()
    restored_state = restored_snapshot["state"]
    assert _canonical_bytes(restored_snapshot) == before_state_bytes
    chunk_key = near_high._chunk_state_key
    for state_key in (
        "barrier_pending",
        "barrier_closed",
        "failed_native_keys",
        "partial_bucket_error",
        "completed_native_keys",
        "completed_native_count_by_symbol",
        "last_completed_native_key_by_symbol",
    ):
        assert restored_state[chunk_key][state_key] == before_state[chunk_key][state_key]
    assert {key: value for key, value in restored_state.items() if key != chunk_key} == {
        key: value for key, value in before_state.items() if key != chunk_key
    }
    assert near_high._alpha_max_bound_aggregator is aggregator
    assert {
        component.component_id: _canonical_bytes(child_queue.snapshot())
        for component, _child, child_queue in strategy._children
    } == before_queue_bytes


def test_validate_research_warmup_ready_checks_all_children_before_calling_any(monkeypatch):
    strategy = _strategy(
        monkeypatch,
        [
            _component("ready", strategy_class="WarmupReadyChild"),
            _component("legacy", strategy_class="NonCapableChild"),
        ],
        mapping={
            "WarmupReadyChild": _WarmupReadyChild,
            "NonCapableChild": _NonCapableChild,
        },
    )
    ready_child = next(
        child
        for component, child, _queue in strategy._children
        if component.component_id == "ready"
    )

    with pytest.raises(ValueError, match="research warmup child is not capable: legacy"):
        strategy.validate_research_warmup_ready()
    assert ready_child.warmup_ready_calls == 0


def test_validate_research_warmup_ready_delegates_through_public_wrapper(monkeypatch):
    strategy = _strategy(
        monkeypatch,
        [
            _component("b-child", strategy_class="WarmupReadyChild"),
            _component("a-child", strategy_class="WarmupReadyChild"),
        ],
        mapping={"WarmupReadyChild": _WarmupReadyChild},
    )

    assert strategy.validate_research_warmup_ready() is None
    assert {
        component.component_id: child.warmup_ready_calls
        for component, child, _queue in strategy._children
    } == {"a-child": 1, "b-child": 1}


def test_validation_capsule_cannot_be_relabelled_as_prelock_final_refit(monkeypatch):
    _CapsuleChild.instances = []
    _patch_definition(monkeypatch, [_component("comp-a")])
    _install_child_mapping(monkeypatch, {"CapsuleChild": _CapsuleChild})
    validation_mode = "manifest:/sealed/validation_train_fit/row-a.json"
    final_refit_mode = "manifest:/sealed/prelock_final_refit/row-a.json"
    validation = MODULE.ArtifactPortfolioModeStrategy(
        bars=_bars(),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode=validation_mode,
    )
    validation_capsule = validation.get_research_indicator_state()
    final_refit = MODULE.ArtifactPortfolioModeStrategy(
        bars=_bars(),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode=final_refit_mode,
    )
    before = _child_states(final_refit)

    with pytest.raises(ValueError, match="portfolio_mode mismatch"):
        final_refit.set_research_indicator_state(validation_capsule)
    assert _child_states(final_refit) == before

    relabelled = dict(validation_capsule)
    relabelled["portfolio_mode"] = final_refit_mode
    with pytest.raises(ValueError, match="sha256 mismatch"):
        final_refit.set_research_indicator_state(relabelled)
    assert _child_states(final_refit) == before

    freshly_replayed = MODULE.ArtifactPortfolioModeStrategy(
        bars=_bars(),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode=final_refit_mode,
    ).get_research_indicator_state()
    final_refit.set_research_indicator_state(freshly_replayed)
    assert _child_states(final_refit) == [entry["state"] for entry in freshly_replayed["children"]]


def test_component_param_override_preserves_receipt_tuple_identity(monkeypatch):
    receipts = (object(),)
    definition = _patch_definition(monkeypatch, [_component("comp-a")], receipts=receipts)

    copied = MODULE._apply_component_param_overrides(
        definition,
        {"comp-a": {"rebalance_bars": 12}},
    )

    assert copied.artifact_read_receipts is receipts
