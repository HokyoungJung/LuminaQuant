import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def _component(component_id: str, symbols=("BNB/USDT",), strategy_class="CapsuleChild"):
    return MODULE.PortfolioModeComponent(
        component_id=component_id,
        label=component_id,
        strategy_class=strategy_class,
        symbols=tuple(symbols),
        params={},
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


def test_finalize_completed_native_buckets_forwards_to_capable_children_only(monkeypatch):
    strategy = _strategy(
        monkeypatch,
        [
            _component("capable", strategy_class="CapsuleChild"),
            _component("legacy", strategy_class="NonCapableChild"),
        ],
        mapping={"CapsuleChild": _CapsuleChild, "NonCapableChild": _NonCapableChild},
    )

    result = strategy.finalize_completed_native_buckets("2026-07-01T00:00:00Z")

    assert sorted(result) == ["capable"]
    assert result["capable"]["watermark"] == "2026-07-01T00:00:00Z"
    capable_child = next(
        child
        for component, child, _queue in strategy._children
        if component.component_id == "capable"
    )
    assert capable_child.finalize_calls == ["2026-07-01T00:00:00Z"]


def test_component_param_override_preserves_receipt_tuple_identity(monkeypatch):
    receipts = (object(),)
    definition = _patch_definition(monkeypatch, [_component("comp-a")], receipts=receipts)

    copied = MODULE._apply_component_param_overrides(
        definition,
        {"comp-a": {"rebalance_bars": 12}},
    )

    assert copied.artifact_read_receipts is receipts
