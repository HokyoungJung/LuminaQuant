from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies import artifact_portfolio_mode
from lumina_quant.strategies.aggressive_return_alpha_sleeves import (
    FundingHarvestCarryStrategy,
)
from lumina_quant.strategies.alpha_max_research_sleeves import (
    CANONICAL_ALPHA_MAX_COMPONENT_NODES,
    ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy,
    ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy,
    ResearchOnlyFourHourFundingHarvestCarryStrategy,
)
from lumina_quant.strategies.low_turnover_trend_alpha_sleeves import (
    LowTurnoverTrendPersistenceStrategy,
)
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_tier,
    resolve_strategy_class,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_NODES_PATH = REPO_ROOT / ".omx/plans/alpha-max-current-trial-nodes-v1.json"
ADAPTER_SOURCE_PATH = REPO_ROOT / "src/lumina_quant/strategies/alpha_max_research_sleeves.py"

ADAPTERS: dict[str, tuple[type[Any], type[Any], str]] = {
    "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy": (
        ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy,
        LowTurnoverTrendPersistenceStrategy,
        "1d",
    ),
    "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy": (
        ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy,
        CrossSectionalNearHighAnchoringStrategy,
        "1d",
    ),
    "ResearchOnlyFourHourFundingHarvestCarryStrategy": (
        ResearchOnlyFourHourFundingHarvestCarryStrategy,
        FundingHarvestCarryStrategy,
        "4h",
    ),
}


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def test_s01_s02_canonical_native_rows_are_complete_and_research_only() -> None:
    payload = json.loads(CURRENT_NODES_PATH.read_text(encoding="utf-8"))
    current = {
        str(node["implementation"]): node
        for node in payload["nodes"]
        if str(node.get("implementation", "")).startswith("ResearchOnly")
    }

    assert set(current) == set(ADAPTERS)
    for name, (adapter, original, timeframe) in ADAPTERS.items():
        node = current[name]
        assert issubclass(adapter, original)
        assert resolve_strategy_class(name) is adapter
        assert get_strategy_tier(name) == "research_only"
        assert get_default_strategy_params(name) == node["params"]
        assert CANONICAL_ALPHA_MAX_COMPONENT_NODES[name] == {
            "row_id": node["row_id"],
            "timeframe": timeframe,
            "candidate_symbols": tuple(node["symbols"]),
            "params": node["params"],
        }


def test_s09_mixed_native_portfolio_exposes_union_without_clock_conversion(
    monkeypatch,
) -> None:
    admitted = ("ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT")
    components = tuple(
        artifact_portfolio_mode.PortfolioModeComponent(
            component_id=f"native-{index}",
            label=name,
            strategy_class=name,
            symbols=admitted,
            params=adapter.canonical_component_params(),
            weight=1.0 / 3.0,
            source="alpha-max-native-contract-test",
        )
        for index, (name, (adapter, _original, _timeframe)) in enumerate(ADAPTERS.items(), start=1)
    )
    definition = artifact_portfolio_mode.PortfolioModeDefinition(
        portfolio_mode="alpha_max_native_union_test",
        components=components,
        cash_weight=0.0,
        source_artifacts={},
    )
    class_by_name = {name: adapter for name, (adapter, _original, _tf) in ADAPTERS.items()}

    monkeypatch.setattr(
        artifact_portfolio_mode,
        "resolve_portfolio_mode_definition",
        lambda _mode: definition,
    )
    monkeypatch.setattr(
        artifact_portfolio_mode,
        "resolve_strategy_class",
        lambda name, **_kwargs: class_by_name[name],
    )

    strategy = artifact_portfolio_mode.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(symbol_list=list(admitted)),
        events=_Queue(),
        portfolio_mode=definition.portfolio_mode,
        decision_cadence_seconds=1,
    )
    children = {component.strategy_class: child for component, child, _queue in strategy._children}

    assert strategy.decision_cadence_seconds == 1
    assert strategy.uses_timeframe_aggregator is True
    assert strategy.required_timeframes == ("1d", "4h")
    assert set(children) == set(ADAPTERS)
    for name, (_adapter, _original, timeframe) in ADAPTERS.items():
        child = children[name]
        assert child.required_native_timeframes == (timeframe,)
        assert child.required_timeframes == (timeframe,)
        assert tuple(child.symbol_list) == admitted


def test_s14_adapters_delegate_indicator_and_position_formulas_to_originals() -> None:
    tree = ast.parse(
        ADAPTER_SOURCE_PATH.read_text(encoding="utf-8"),
        filename=str(ADAPTER_SOURCE_PATH),
    )
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    forbidden_formula_overrides = {
        "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy": {
            "_process_symbol",
            "_horizon_agreement",
            "_efficiency",
            "_desired_signal",
            "_vol_scaled_allocation",
            "_decide",
            "_enter",
            "_emit_exit",
        },
        "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy": {
            "_update_symbol",
            "_score_and_select",
            "_inverse_vol_weights",
            "_age",
            "_evaluate",
            "_emit_targets",
        },
        "ResearchOnlyFourHourFundingHarvestCarryStrategy": {
            "_avg_funding",
            "_process_symbol",
            "_carry_direction",
            "_entry_decision",
            "_should_pyramid",
            "_vol_scaled_allocation",
            "_entry_metadata",
        },
    }
    expected_bases = {
        "ResearchOnlyDailyLowTurnoverTrendPersistenceStrategy": (
            "_NativeCompletedAdapterMixin",
            "LowTurnoverTrendPersistenceStrategy",
        ),
        "ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy": (
            "CrossSectionalNearHighAnchoringStrategy",
        ),
        "ResearchOnlyFourHourFundingHarvestCarryStrategy": (
            "_NativeCompletedAdapterMixin",
            "FundingHarvestCarryStrategy",
        ),
    }

    for name, forbidden in forbidden_formula_overrides.items():
        node = classes[name]
        methods = {
            item.name
            for item in node.body
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert tuple(ast.unparse(base) for base in node.bases) == expected_bases[name]
        assert methods.isdisjoint(forbidden)

    mixin = classes["_NativeCompletedAdapterMixin"]
    ingest = next(
        item
        for item in mixin.body
        if isinstance(item, ast.FunctionDef) and item.name == "_ingest_completed_native_bar"
    )
    near_high = classes["ResearchOnlyDailyCrossSectionalNearHighAnchoringStrategy"]
    barrier_flush = next(
        item
        for item in near_high.body
        if isinstance(item, ast.FunctionDef) and item.name == "_barrier_flush_if_complete"
    )
    assert "super().calculate_signals_window" in {
        ast.unparse(call.func) for call in ast.walk(ingest) if isinstance(call, ast.Call)
    }
    assert "super().calculate_signals_window" in {
        ast.unparse(call.func) for call in ast.walk(barrier_flush) if isinstance(call, ast.Call)
    }
