import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lumina_quant.core.events import MarketEvent, SignalEvent
from lumina_quant.live_selection import supports_live_portfolio_mode

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "lumina_quant"
    / "strategies"
    / "artifact_portfolio_mode.py"
)
SPEC = importlib.util.spec_from_file_location("artifact_portfolio_mode", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _patch_single_component(monkeypatch, child_cls: type) -> None:
    monkeypatch.setattr(
        MODULE,
        "resolve_portfolio_mode_definition",
        lambda portfolio_mode: MODULE.PortfolioModeDefinition(
            portfolio_mode=portfolio_mode,
            components=(
                MODULE.PortfolioModeComponent(
                    component_id="comp-a",
                    label="component-a",
                    strategy_class="MovingAverageCrossStrategy",
                    symbols=("BNB/USDT",),
                    params={},
                    weight=0.3,
                    source="test",
                ),
            ),
            cash_weight=0.7,
            source_artifacts={},
        ),
    )
    monkeypatch.setattr(MODULE, "resolve_strategy_class", lambda name, default_name=None: child_cls)


def test_portfolio_mode_does_not_propagate_child_timeframes_without_explicit_aggregator_use(
    monkeypatch,
) -> None:
    class _LegacyWindowChild:
        required_timeframes = ("1h",)

        def __init__(self, bars, events, **params):
            _ = bars, events, params

        def calculate_signals(self, event):
            _ = event

    _patch_single_component(monkeypatch, _LegacyWindowChild)

    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(
            symbol_list=["BNB/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0
        ),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode="hybrid_guarded_mode",
    )

    assert strategy.uses_timeframe_aggregator is False
    assert strategy.required_timeframes == ()


def test_portfolio_mode_propagates_child_timeframes_for_explicit_aggregator_use(
    monkeypatch,
) -> None:
    class _AggregatorChild:
        uses_timeframe_aggregator = True
        required_timeframes = ("20s", "1m")

        def __init__(self, bars, events, **params):
            _ = bars, events, params

        def calculate_signals(self, event):
            _ = event

    _patch_single_component(monkeypatch, _AggregatorChild)

    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(
            symbol_list=["BNB/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0
        ),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode="hybrid_guarded_mode",
    )

    assert strategy.uses_timeframe_aggregator is True
    assert strategy.required_timeframes == ("1m", "20s")


def test_portfolio_mode_applies_research_only_component_param_overrides(monkeypatch) -> None:
    observed_params = {}

    class _ParamChild:
        def __init__(self, bars, events, **params):
            _ = bars, events
            observed_params.update(params)

        def calculate_signals(self, event):
            _ = event

    _patch_single_component(monkeypatch, _ParamChild)

    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(
            symbol_list=["BNB/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0
        ),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode="hybrid_guarded_mode",
        component_param_overrides={
            "comp-a": {
                "rebalance_bars": 60,
                "gross_exposure": 0.01,
            }
        },
    )

    assert strategy.definition.components[0].params["rebalance_bars"] == 60
    assert strategy.definition.components[0].params["gross_exposure"] == 0.01
    assert observed_params["rebalance_bars"] == 60
    assert observed_params["gross_exposure"] == 0.01


def test_portfolio_mode_propagates_child_features_and_context(monkeypatch) -> None:
    class _ContextChild:
        required_features = ("taker_buy_quote_volume",)

        def __init__(self, bars, events, **params):
            _ = bars, params
            self.events = events

        def calculate_signals_context(self, context):
            assert context.feature_lookup == "feature-lookup"
            self.events.put(
                SignalEvent(
                    strategy_id="context-child",
                    symbol="BNB/USDT",
                    datetime=context.event.time,
                    signal_type="LONG",
                    strength=1.0,
                    metadata={"target_allocation": 0.1},
                )
            )

    _patch_single_component(monkeypatch, _ContextChild)

    events = []
    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(
            symbol_list=["BNB/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0
        ),
        events=SimpleNamespace(put=lambda item: events.append(item)),
        portfolio_mode="hybrid_guarded_mode",
    )
    strategy.calculate_signals_context(
        SimpleNamespace(
            event=MarketEvent(
                time="2026-04-17T00:00:00Z",
                symbol="BNB/USDT",
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1.0,
            ),
            aggregator=None,
            feature_lookup="feature-lookup",
        )
    )

    assert strategy.preferred_contract == "context"
    assert strategy.required_features == ("taker_buy_quote_volume",)
    assert len(events) == 1


def test_portfolio_mode_strategy_forwards_component_weighted_signals(monkeypatch) -> None:
    class _ChildStrategy:
        required_timeframes = ("1h",)

        def __init__(self, bars, events, **params):
            _ = bars, params
            self.events = events

        def calculate_signals(self, event):
            self.events.put(
                SignalEvent(
                    strategy_id="child",
                    symbol="BNB/USDT",
                    datetime=event.time,
                    signal_type="LONG",
                    strength=0.25,
                    metadata={
                        "target_allocation": 0.20,
                        "max_symbol_exposure_pct": 0.20,
                        "max_order_value": 500.0,
                    },
                )
            )

    _patch_single_component(monkeypatch, _ChildStrategy)

    events = []
    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(
            symbol_list=["BNB/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0
        ),
        events=SimpleNamespace(put=lambda item: events.append(item)),
        portfolio_mode="hybrid_guarded_mode",
    )
    strategy.calculate_signals(
        MarketEvent(
            time="2026-04-17T00:00:00Z",
            symbol="BNB/USDT",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
        )
    )

    assert len(events) == 1
    signal = events[0]
    assert signal.metadata["component_id"] == "comp-a"
    assert signal.metadata["target_allocation_scale"] == 0.3
    assert signal.metadata["child_target_allocation"] == 0.20
    assert signal.metadata["target_allocation"] == 0.06
    assert signal.metadata["max_symbol_exposure_pct"] == 0.06
    assert signal.metadata["max_order_value"] == 150.0
    assert signal.strength == 0.075
    assert signal.client_order_id.startswith("LQPM-") or signal.client_order_id.startswith(
        "comp-a-"
    )


def test_profit_portfolio_mode_caps_unbounded_child_signals(monkeypatch) -> None:
    class _UnboundedChildStrategy:
        def __init__(self, bars, events, **params):
            _ = bars, params
            self.events = events

        def calculate_signals(self, event):
            self.events.put(
                SignalEvent(
                    strategy_id="unbounded-child",
                    symbol="BNB/USDT",
                    datetime=event.time,
                    signal_type="LONG",
                    strength=1.0,
                    metadata={"strategy": "legacy_pair_child_without_sizing_metadata"},
                )
            )

    _patch_single_component(monkeypatch, _UnboundedChildStrategy)

    events = []
    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(
            symbol_list=["BNB/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0
        ),
        events=SimpleNamespace(put=lambda item: events.append(item)),
        portfolio_mode="profit_moonshot_balanced_mode",
    )
    strategy.calculate_signals(
        MarketEvent(
            time="2026-04-17T00:00:00Z",
            symbol="BNB/USDT",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
        )
    )

    assert len(events) == 1
    signal = events[0]
    assert signal.metadata["target_allocation"] == 0.006
    assert signal.metadata["max_symbol_exposure_pct"] == 0.006
    assert signal.metadata["max_order_value"] == 75.0
    assert signal.metadata["portfolio_mode_unbounded_child_target_allocation"] == 0.02
    assert signal.metadata["portfolio_mode_unbounded_child_max_order_value"] == 250.0


def test_derivatives_flow_squeeze_mode_resolves_new_alpha_components() -> None:
    definition = MODULE.resolve_portfolio_mode_definition("derivatives_flow_squeeze_mode")

    assert supports_live_portfolio_mode("derivatives_flow_squeeze_mode")
    assert definition.cash_weight == 0.0
    assert [component.component_id for component in definition.components] == [
        "dfse_top5_exhaustion_plus_flow",
        "dfse_fast_liquidation_reversal",
        "dfse_basis_flow_continuation",
    ]
    assert [component.weight for component in definition.components] == [0.55, 0.25, 0.2]
    assert {component.strategy_class for component in definition.components} == {
        "DerivativesFlowSqueezeStrategy"
    }
    assert "derivatives_flow_squeeze_manifest_path" in definition.source_artifacts


def test_profit_moonshot_derivatives_taker_flow_mode_uses_strict_raw_taker_replay() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_derivatives_taker_flow_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_derivatives_taker_flow_mode")
    assert [component.component_id for component in definition.components] == [
        "profit_moonshot_dfse_top3_taker_flow_continuation",
        "profit_moonshot_dfse_top3_liquidation_gap_probe",
    ]
    assert definition.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert {component.strategy_class for component in definition.components} == {
        "DerivativesFlowSqueezeStrategy"
    }
    assert all(
        component.params["allow_ohlcv_flow_proxy"] is False for component in definition.components
    )
    assert definition.components[0].params["enable_continuation"] is True
    assert definition.components[1].params["enable_exhaustion"] is True


def test_profit_moonshot_derivatives_sparse_mode_reduces_overtrading_without_exposure_increase() -> (
    None
):
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_derivatives_taker_flow_sparse_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_derivatives_taker_flow_sparse_mode")
    assert definition.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    component = definition.components[0]
    assert component.component_id == "profit_moonshot_dfse_top3_sparse_taker_flow"
    assert component.params["allow_ohlcv_flow_proxy"] is False
    assert component.params["evaluation_cadence_bars"] == 360
    assert component.params["flow_imbalance_min"] == 0.055
    assert component.params["target_allocation"] == 0.008


def test_profit_moonshot_leadlag_slow_diffusion_mode_uses_screened_btc_eth_candidate() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_leadlag_slow_diffusion_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_leadlag_slow_diffusion_mode")
    assert definition.symbols == ["BTC/USDT", "ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "CrossCryptoSlowDiffusionStrategy"
    assert component.component_id == "profit_moonshot_leadlag_btc_eth_2h_8h_slow_diffusion"
    assert component.params["leader_symbol"] == "BTC/USDT"
    assert component.params["target_symbol"] == "ETH/USDT"
    assert component.params["lag_bars"] == 2
    assert component.params["leader_abs_ret_min"] == 0.015
    assert component.params["max_hold_bars"] == 8
    assert component.params["target_allocation"] == 0.008


def test_profit_moonshot_leadlag_slow_diffusion_sol_eth_mode_uses_second_screen_survivor() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_leadlag_slow_diffusion_sol_eth_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_leadlag_slow_diffusion_sol_eth_mode")
    assert definition.symbols == ["SOL/USDT", "ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "CrossCryptoSlowDiffusionStrategy"
    assert component.component_id == "profit_moonshot_leadlag_sol_eth_1h_8h_slow_diffusion"
    assert component.params["leader_symbol"] == "SOL/USDT"
    assert component.params["target_symbol"] == "ETH/USDT"
    assert component.params["lag_bars"] == 1
    assert component.params["leader_abs_ret_min"] == 0.015
    assert component.params["max_hold_bars"] == 8
    assert component.params["target_allocation"] == 0.008


def test_profit_moonshot_leadlag_slow_diffusion_ensemble_splits_same_target_risk() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_leadlag_slow_diffusion_ensemble_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_leadlag_slow_diffusion_ensemble_mode")
    assert definition.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    assert [component.component_id for component in definition.components] == [
        "profit_moonshot_leadlag_btc_eth_2h_8h_slow_diffusion",
        "profit_moonshot_leadlag_sol_eth_1h_8h_slow_diffusion",
    ]
    assert [component.weight for component in definition.components] == [0.60, 0.40]
    assert {component.strategy_class for component in definition.components} == {
        "CrossCryptoSlowDiffusionStrategy"
    }
    assert [component.params["leader_symbol"] for component in definition.components] == [
        "BTC/USDT",
        "SOL/USDT",
    ]
    assert [component.params["lag_bars"] for component in definition.components] == [2, 1]
    assert all(
        component.params["target_symbol"] == "ETH/USDT" for component in definition.components
    )
    assert all(
        component.params["target_allocation"] == 0.008 for component in definition.components
    )
    assert (
        sum(
            component.weight * component.params["target_allocation"]
            for component in definition.components
        )
        == 0.008
    )


def test_profit_moonshot_hourly_shock_reversion_eth_mode_uses_stateful_screen_survivor() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_hourly_shock_reversion_eth_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_hourly_shock_reversion_eth_mode")
    assert definition.symbols == ["ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "HourlyShockReversionStrategy"
    assert component.component_id == "profit_moonshot_hourly_shock_reversion_eth_4h_48h_stop2"
    assert component.params["target_symbol"] == "ETH/USDT"
    assert component.params["lookback_bars"] == 4
    assert component.params["return_threshold"] == 0.006
    assert component.params["max_hold_bars"] == 48
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0
    assert component.params["stop_loss_pct"] == 0.02


def test_profit_moonshot_hourly_shock_reversion_eth_12h_mode_uses_second_survivor() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_hourly_shock_reversion_eth_12h_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_hourly_shock_reversion_eth_12h_mode")
    assert definition.symbols == ["ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "HourlyShockReversionStrategy"
    assert component.component_id == (
        "profit_moonshot_hourly_shock_reversion_eth_12h_72h_stop5_take10"
    )
    assert component.params["target_symbol"] == "ETH/USDT"
    assert component.params["lookback_bars"] == 12
    assert component.params["return_threshold"] == 0.01
    assert component.params["max_hold_bars"] == 72
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0
    assert component.params["stop_loss_pct"] == 0.05
    assert component.params["take_profit_pct"] == 0.10


def test_profit_moonshot_hourly_shock_reversion_dense_mode_lowers_threshold_only() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode")
    assert definition.symbols == ["ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "HourlyShockReversionStrategy"
    assert component.params["lookback_bars"] == 12
    assert component.params["return_threshold"] == 0.008
    assert component.params["max_hold_bars"] == 72
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0


def test_profit_moonshot_hourly_shock_reversion_funding_guard_mode_filters_hours() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode"
    )

    assert supports_live_portfolio_mode(
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode"
    )
    assert definition.symbols == ["ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "HourlyShockReversionStrategy"
    assert component.params["return_threshold"] == 0.008
    assert component.params["excluded_entry_hours_utc"] == "0,1,8,9,16,17"
    assert component.params["target_allocation"] == 0.008
    assert component.weight == 1.0


def test_profit_moonshot_hourly_shock_reversion_taker_flow_guard_mode_requires_features() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode"
    )

    assert not supports_live_portfolio_mode(
        "profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode"
    )
    assert definition.symbols == ["ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "HourlyShockReversionStrategy"
    assert component.params["return_threshold"] == 0.01
    assert component.params["flow_confirmation_lookback_bars"] == 1
    assert component.params["flow_imbalance_min"] == 0.10
    assert component.params["target_allocation"] == 0.008
    assert component.weight == 1.0
    strategy = MODULE.ArtifactPortfolioModeStrategy(
        bars=SimpleNamespace(
            symbol_list=["ETH/USDT"], get_latest_bar_value=lambda *args, **kwargs: 100.0
        ),
        events=SimpleNamespace(put=lambda item: None),
        portfolio_mode="profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode",
    )
    assert set(strategy.required_features) == {
        "taker_buy_base_volume",
        "taker_sell_base_volume",
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
    }


def test_profit_moonshot_hourly_shock_reversion_funding_taker_flow_guard_mode_combines_filters() -> (
    None
):
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode"
    )

    assert not supports_live_portfolio_mode(
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_taker_flow_guard_mode"
    )
    assert definition.symbols == ["ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "HourlyShockReversionStrategy"
    assert component.params["return_threshold"] == 0.008
    assert component.params["excluded_entry_hours_utc"] == "0,1,8,9,16,17"
    assert component.params["flow_confirmation_lookback_bars"] == 1
    assert component.params["flow_imbalance_min"] == 0.10
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0
    assert component.weight == 1.0


def test_profit_moonshot_hourly_shock_reversion_sol_regime_guard_mode_uses_replay_survivor() -> (
    None
):
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode"
    )

    assert not supports_live_portfolio_mode(
        "profit_moonshot_hourly_shock_reversion_eth_12h_sol_regime_guard_mode"
    )
    assert definition.symbols == ["ETH/USDT", "SOL/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "HourlyShockReversionStrategy"
    assert component.params["return_threshold"] == 0.01
    assert component.params["regime_symbol"] == "SOL/USDT"
    assert component.params["regime_lookback_bars"] == 24
    assert component.params["counterguard_return_threshold"] == 0.035
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0
    assert component.weight == 1.0


def test_profit_moonshot_precious_metal_pair_mode_includes_four_metals_with_caps() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_precious_metal_pair_aggressive_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_precious_metal_pair_aggressive_mode")
    assert definition.symbols == ["XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"]
    assert [component.strategy_class for component in definition.components] == [
        "TimeframePairZScoreReversionStrategy",
        "TimeframePairZScoreReversionStrategy",
    ]
    assert [component.symbols for component in definition.components] == [
        ("XAU/USDT", "XAG/USDT"),
        ("XPT/USDT", "XPD/USDT"),
    ]
    assert [component.weight for component in definition.components] == [0.65, 0.35]
    assert all(component.params["timeframe"] == "1h" for component in definition.components)
    assert all(
        component.params["target_allocation"] == 0.024 for component in definition.components
    )
    assert all(component.params["max_order_value"] == 350.0 for component in definition.components)
    assert (
        sum(
            component.weight * component.params["target_allocation"]
            for component in definition.components
        )
        == 0.024
    )


def test_profit_moonshot_filtered_shock_reversion_diversified_mode_keeps_gross_cap() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_filtered_shock_reversion_diversified_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_filtered_shock_reversion_diversified_mode")
    assert supports_live_portfolio_mode("profit_moonshot_taker_flow_exhaustion_eth_mode")
    assert supports_live_portfolio_mode("profit_moonshot_taker_flow_exhaustion_eth_reactive_mode")
    assert supports_live_portfolio_mode("profit_moonshot_taker_flow_exhaustion_eth_hold_mode")
    assert supports_live_portfolio_mode(
        "profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode"
    )
    assert definition.symbols == ["ETH/USDT", "SOL/USDT"]
    assert [component.strategy_class for component in definition.components] == [
        "HourlyShockReversionStrategy",
        "HourlyShockReversionStrategy",
    ]
    assert sum(component.weight for component in definition.components) == 1.0
    assert (
        sum(
            component.weight * component.params["target_allocation"]
            for component in definition.components
        )
        == 0.008
    )
    sol_component = next(
        item for item in definition.components if item.params["target_symbol"] == "SOL/USDT"
    )
    assert sol_component.params["entry_hours_utc"] == "2,3,4,10,11,12,18,19,20"


def test_profit_moonshot_taker_flow_exhaustion_eth_mode_uses_same_risk_cap() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_taker_flow_exhaustion_eth_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_taker_flow_exhaustion_eth_mode")
    assert definition.symbols == ["ETH/USDT"]
    component = definition.components[0]
    assert component.strategy_class == "TakerFlowExhaustionReversalStrategy"
    assert component.params["flow_imbalance_min"] == 0.14
    assert component.params["funding_abs_cap"] == 0.00015
    assert component.params["max_realized_volatility"] == 0.008
    assert component.params["entry_hours_utc"] == "13,14,15,16,17,18,19,20"
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0
    assert component.weight == 1.0


def test_resolve_portfolio_mode_definition_supports_recursive_allocator_sleeves(
    monkeypatch, tmp_path: Path
) -> None:
    def _write(path: Path, payload: dict) -> Path:
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    incumbent_path = _write(
        tmp_path / "incumbent.json",
        {
            "weights": [
                {
                    "candidate_id": "leaf_a",
                    "name": "leaf_a",
                    "strategy_class": "MovingAverageCrossStrategy",
                    "symbols": ["BTC/USDT"],
                    "weight": 0.42,
                    "weight_share": 0.6,
                },
                {
                    "candidate_id": "leaf_b",
                    "name": "leaf_b",
                    "strategy_class": "RsiStrategy",
                    "symbols": ["ETH/USDT"],
                    "weight": 0.28,
                    "weight_share": 0.4,
                },
            ],
            "cash_weight": 0.3,
        },
    )
    autoresearch_path = _write(
        tmp_path / "autoresearch.json",
        {
            "weights": [
                {
                    "candidate_id": "leaf_c",
                    "name": "leaf_c",
                    "strategy_class": "TopCapTimeSeriesMomentumStrategy",
                    "symbols": ["SOL/USDT"],
                    "weight": 1.0,
                }
            ]
        },
    )
    blend_path = _write(
        tmp_path / "blend.json",
        {
            "weights": [
                {"candidate_id": "incumbent_only", "name": "incumbent_only", "weight": 0.7},
                {"candidate_id": "autoresearch_55_45", "name": "autoresearch_55_45", "weight": 0.3},
            ]
        },
    )
    soft_path = _write(
        tmp_path / "soft.json",
        {
            "current_state": {
                "weights": {
                    "incumbent": 0.5,
                    "blend_85_15": 0.5,
                    "autoresearch_55_45": 0.0,
                }
            }
        },
    )
    three_way_path = _write(
        tmp_path / "three.json",
        {
            "current_state": {
                "weights": {
                    "incumbent": 0.0,
                    "blend_85_15": 1.0,
                    "autoresearch_55_45": 0.0,
                }
            }
        },
    )
    pair_path = _write(
        tmp_path / "pair.json",
        {
            "candidate_id": "leaf_pair",
            "name": "leaf_pair",
            "strategy_class": "PairSpreadZScoreStrategy",
            "symbols": ["BNB/USDT", "TRX/USDT"],
        },
    )
    state_vwap_pair_path = _write(
        tmp_path / "state_vwap_pair.json",
        {
            "candidate_id": "leaf_state_vwap_pair",
            "name": "leaf_state_vwap_pair",
            "strategy_class": "PairSpreadZScoreStrategy",
            "symbols": ["BNB/USDT", "TRX/USDT"],
            "params": {"signal_variant": "state_vwap"},
        },
    )
    wave2_pair_path = _write(
        tmp_path / "wave2_pair.json",
        {
            "candidate_id": "leaf_wave2_pair",
            "name": "leaf_wave2_pair",
            "strategy_class": "PairSpreadZScoreStrategy",
            "symbols": ["BNB/USDT", "TRX/USDT"],
            "params": {"entry_z": 2.2, "exit_z": 0.55},
        },
    )
    hybrid_path = _write(
        tmp_path / "hybrid.json",
        {
            "scenarios": {
                "refreshed_latest_tail": {
                    "final_allocation": {
                        "weights": {
                            "soft_three_way_regime": 0.4,
                            "balanced_overlay_80_20": 0.3,
                            "pair_tactical_mode": 0.1,
                        },
                        "cash_weight": 0.2,
                    }
                }
            }
        },
    )
    legacy_hybrid_path = _write(
        tmp_path / "legacy_hybrid.json",
        {
            "scenarios": {
                "refreshed_latest_tail": {
                    "final_allocation": {
                        "weights": {
                            "state_vwap_pair": 0.4,
                            "wave2_pair": 0.3,
                            "soft_three_way_regime": 0.2,
                        },
                        "cash_weight": 0.1,
                    }
                }
            }
        },
    )
    retuned_hybrid_path = _write(
        tmp_path / "retuned_hybrid.json",
        {
            "scenarios": {
                "refreshed_latest_tail": {
                    "final_allocation": {
                        "weights": {
                            "aggressive_realized_mode": 0.6,
                            "legacy_no_highvol_hybrid_mode": 0.4,
                        },
                        "cash_weight": 0.0,
                    }
                }
            }
        },
    )

    monkeypatch.setattr(MODULE, "REFRESHED_INCUMBENT_PATH", incumbent_path)
    monkeypatch.setattr(MODULE, "REFRESHED_AUTORESEARCH_55_45_PATH", autoresearch_path)
    monkeypatch.setattr(MODULE, "REFRESHED_BLEND_PATH", blend_path)
    monkeypatch.setattr(MODULE, "SOFT_THREE_WAY_ALLOCATOR_PATH", soft_path)
    monkeypatch.setattr(MODULE, "THREE_WAY_ALLOCATOR_PATH", three_way_path)
    monkeypatch.setattr(MODULE, "PAIR_TACTICAL_PATH", pair_path)
    monkeypatch.setattr(MODULE, "STATE_VWAP_PAIR_PATH", state_vwap_pair_path)
    monkeypatch.setattr(MODULE, "WAVE2_PAIR_PATH", wave2_pair_path)
    monkeypatch.setattr(MODULE, "HYBRID_PATH", hybrid_path)
    monkeypatch.setattr(MODULE, "LEGACY_NO_HIGHVOL_HYBRID_PATH", legacy_hybrid_path)
    monkeypatch.setattr(MODULE, "RETUNED_LIVE_PORTFOLIO_HYBRID_PATH", retuned_hybrid_path)
    monkeypatch.setattr(
        MODULE,
        "PRODUCTION_GUARDED_PATH",
        _write(
            tmp_path / "production_guarded.json",
            {
                "weights": [
                    {"candidate_id": "incumbent_only", "name": "incumbent_only", "weight": 0.4},
                    {"candidate_id": "blend_85_15", "name": "blend_85_15", "weight": 0.35},
                    {
                        "candidate_id": "autoresearch_55_45",
                        "name": "autoresearch_55_45",
                        "weight": 0.2,
                    },
                ],
                "cash_weight": 0.05,
            },
        ),
    )
    monkeypatch.setattr(MODULE, "STRICT_AUTORESEARCH_1X_PATH", autoresearch_path)

    defensive = MODULE.resolve_portfolio_mode_definition("defensive_overlay_mode")
    aggressive = MODULE.resolve_portfolio_mode_definition("aggressive_realized_mode")
    hybrid = MODULE.resolve_portfolio_mode_definition("hybrid_guarded_mode")
    legacy_hybrid = MODULE.resolve_portfolio_mode_definition("legacy_no_highvol_hybrid_mode")
    retuned_hybrid = MODULE.resolve_portfolio_mode_definition("retuned_live_portfolio_hybrid_mode")
    practical = MODULE.resolve_portfolio_mode_definition("strict_autoresearch_practical_mode")
    promoted = MODULE.resolve_portfolio_mode_definition("production_guarded_state_vwap_pair_mode")
    risk_off = MODULE.resolve_portfolio_mode_definition("risk_off_mode")

    defensive_weights = {item.component_id: round(item.weight, 6) for item in defensive.components}
    aggressive_weights = {
        item.component_id: round(item.weight, 6) for item in aggressive.components
    }
    hybrid_weights = {item.component_id: round(item.weight, 6) for item in hybrid.components}
    legacy_hybrid_weights = {
        item.component_id: round(item.weight, 6) for item in legacy_hybrid.components
    }
    retuned_hybrid_weights = {
        item.component_id: round(item.weight, 6) for item in retuned_hybrid.components
    }
    practical_weights = {item.component_id: round(item.weight, 6) for item in practical.components}
    promoted_weights = {item.component_id: round(item.weight, 6) for item in promoted.components}

    assert defensive_weights == {
        "leaf_a": 0.357,
        "leaf_b": 0.238,
        "leaf_c": 0.105,
        "leaf_pair": 0.3,
    }
    assert aggressive_weights == {
        "leaf_a": 0.42,
        "leaf_b": 0.28,
        "leaf_c": 0.3,
    }
    assert aggressive.cash_weight == 0.21
    assert hybrid_weights == {
        "leaf_a": 0.3264,
        "leaf_b": 0.2176,
        "leaf_c": 0.096,
        "leaf_pair": 0.16,
    }
    assert legacy_hybrid_weights == {
        "leaf_state_vwap_pair": 0.4,
        "leaf_wave2_pair": 0.3,
        "leaf_a": 0.102,
        "leaf_b": 0.068,
        "leaf_c": 0.03,
    }
    assert retuned_hybrid_weights == {
        "leaf_a": 0.2928,
        "leaf_b": 0.1952,
        "leaf_c": 0.192,
        "leaf_state_vwap_pair": 0.16,
        "leaf_wave2_pair": 0.12,
    }
    assert practical_weights == {
        "leaf_a": 0.3096,
        "leaf_b": 0.2064,
        "leaf_c": 0.444,
    }
    assert promoted_weights == {
        "leaf_a": 0.1548,
        "leaf_b": 0.1032,
        "leaf_c": 0.122,
        "leaf_state_vwap_pair": 0.25,
    }
    assert abs(hybrid.cash_weight - 0.3632) < 1e-12
    assert abs(legacy_hybrid.cash_weight - 0.151) < 1e-12
    assert abs(retuned_hybrid.cash_weight - 0.1864) < 1e-12
    assert abs(practical.cash_weight - 0.1948) < 1e-6
    assert abs(promoted.cash_weight - 0.4474) < 1e-6
    assert risk_off.cash_weight == 1.0
    assert risk_off.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TRX/USDT"]
    assert "legacy_no_highvol_hybrid_mode" in MODULE.supported_portfolio_modes()
    assert "retuned_live_portfolio_hybrid_mode" in MODULE.supported_portfolio_modes()
    assert "profit_reboot_panic_rebound_mode" in MODULE.supported_portfolio_modes()
    assert "profit_reboot_session_pair_carry_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_adaptive_momentum_120_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_adaptive_momentum_130_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_adaptive_momentum_140_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_adaptive_momentum_boost_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_adaptive_momentum_governed_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_adaptive_momentum_vol_target_mode" in MODULE.supported_portfolio_modes()
    assert (
        "profit_moonshot_adaptive_momentum_vol_target_132_mode"
        in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_adaptive_momentum_asym_dynamic_mode" in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_adaptive_momentum_volume_guard_mode" in MODULE.supported_portfolio_modes()
    )
    assert "profit_moonshot_momentum_hybrid_return_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_momentum_hybrid_safe_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_momentum_hybrid_core_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_ensemble_mode" in MODULE.supported_portfolio_modes()
    assert "profit_moonshot_derivatives_taker_flow_mode" in MODULE.supported_portfolio_modes()
    assert (
        "profit_moonshot_derivatives_taker_flow_sparse_mode" in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_leadlag_slow_diffusion_sol_eth_mode" in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_leadlag_slow_diffusion_ensemble_mode" in MODULE.supported_portfolio_modes()
    )
    assert "profit_moonshot_hourly_shock_reversion_eth_mode" in MODULE.supported_portfolio_modes()
    assert (
        "profit_moonshot_hourly_shock_reversion_eth_12h_mode" in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode"
        in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode"
        in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_hourly_shock_reversion_eth_12h_taker_flow_guard_mode"
        in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_filtered_shock_reversion_diversified_mode"
        in MODULE.supported_portfolio_modes()
    )
    assert "profit_moonshot_taker_flow_exhaustion_eth_mode" in MODULE.supported_portfolio_modes()
    assert (
        "profit_moonshot_taker_flow_exhaustion_eth_reactive_mode"
        in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_taker_flow_exhaustion_eth_hold_mode" in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode"
        in MODULE.supported_portfolio_modes()
    )
    assert (
        "profit_moonshot_precious_metal_pair_aggressive_mode" in MODULE.supported_portfolio_modes()
    )
    assert supports_live_portfolio_mode("legacy_no_highvol_hybrid_mode")
    assert supports_live_portfolio_mode("retuned_live_portfolio_hybrid_mode")
    assert supports_live_portfolio_mode("profit_reboot_panic_rebound_mode")
    assert supports_live_portfolio_mode("profit_reboot_session_pair_carry_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_120_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_130_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_140_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_boost_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_governed_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_vol_target_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_vol_target_132_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_asym_dynamic_mode")
    assert supports_live_portfolio_mode("profit_moonshot_adaptive_momentum_volume_guard_mode")
    assert supports_live_portfolio_mode("profit_moonshot_momentum_hybrid_return_mode")
    assert supports_live_portfolio_mode("profit_moonshot_momentum_hybrid_safe_mode")
    assert supports_live_portfolio_mode("profit_moonshot_momentum_hybrid_core_mode")
    assert supports_live_portfolio_mode("profit_moonshot_ensemble_mode")
    assert supports_live_portfolio_mode("profit_moonshot_derivatives_taker_flow_mode")
    assert supports_live_portfolio_mode("profit_moonshot_derivatives_taker_flow_sparse_mode")
    assert supports_live_portfolio_mode("profit_moonshot_leadlag_slow_diffusion_sol_eth_mode")
    assert supports_live_portfolio_mode("profit_moonshot_leadlag_slow_diffusion_ensemble_mode")
    assert supports_live_portfolio_mode("profit_moonshot_hourly_shock_reversion_eth_mode")
    assert supports_live_portfolio_mode("profit_moonshot_hourly_shock_reversion_eth_12h_mode")
    assert supports_live_portfolio_mode("profit_moonshot_hourly_shock_reversion_eth_12h_dense_mode")
    assert supports_live_portfolio_mode(
        "profit_moonshot_hourly_shock_reversion_eth_12h_funding_guard_mode"
    )
    assert supports_live_portfolio_mode("profit_moonshot_filtered_shock_reversion_diversified_mode")
    assert supports_live_portfolio_mode("profit_moonshot_precious_metal_pair_aggressive_mode")


def test_profit_reboot_synthetic_modes_resolve_new_strategy_families() -> None:
    panic = MODULE.resolve_portfolio_mode_definition("profit_reboot_panic_rebound_mode")
    pair = MODULE.resolve_portfolio_mode_definition("profit_reboot_session_pair_carry_mode")

    assert panic.components[0].strategy_class == "PanicReboundMeanReversionStrategy"
    assert panic.components[0].symbols == (
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "TRX/USDT",
    )
    assert pair.components[0].strategy_class == "SessionFilteredPairCarryStrategy"
    assert pair.components[0].symbols == ("BNB/USDT", "TRX/USDT")
    assert pair.components[0].params["allowed_session_utc_hours"]


def test_profit_moonshot_synthetic_modes_resolve_no_aggregator_strategy_families() -> None:
    boost = MODULE.resolve_portfolio_mode_definition("profit_moonshot_adaptive_momentum_boost_mode")
    ladder_120 = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_adaptive_momentum_120_mode"
    )
    ladder_130 = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_adaptive_momentum_130_mode"
    )
    ladder_140 = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_adaptive_momentum_140_mode"
    )
    governed = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_adaptive_momentum_governed_mode"
    )
    vol_target = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_adaptive_momentum_vol_target_mode"
    )
    vol_target_132 = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_adaptive_momentum_vol_target_132_mode"
    )
    asym_dynamic = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_adaptive_momentum_asym_dynamic_mode"
    )
    volume_guard = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_adaptive_momentum_volume_guard_mode"
    )
    hybrid_return = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_momentum_hybrid_return_mode"
    )
    hybrid_safe = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_momentum_hybrid_safe_mode"
    )
    hybrid_core = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_momentum_hybrid_core_mode"
    )
    trend = MODULE.resolve_portfolio_mode_definition("profit_moonshot_trend_mode")
    breakout = MODULE.resolve_portfolio_mode_definition("profit_moonshot_breakout_mode")
    reversion = MODULE.resolve_portfolio_mode_definition("profit_moonshot_reversion_mode")
    ensemble = MODULE.resolve_portfolio_mode_definition("profit_moonshot_ensemble_mode")

    assert boost.components[0].strategy_class == "AdaptiveRegimeMomentumStrategy"
    assert boost.components[0].params["gross_exposure"] == 0.0075
    assert boost.components[0].params["max_order_value"] == 300.0
    assert ladder_120.components[0].params["gross_exposure"] == 0.006
    assert ladder_120.components[0].params["max_order_value"] == 240.0
    assert ladder_130.components[0].params["gross_exposure"] == 0.0065
    assert ladder_130.components[0].params["max_order_value"] == 260.0
    assert ladder_140.components[0].params["gross_exposure"] == 0.007
    assert ladder_140.components[0].params["max_order_value"] == 280.0
    assert governed.components[0].params["max_realized_vol"] == 0.0035
    assert governed.components[0].params["broad_threshold"] == 0.0015
    assert vol_target.components[0].params["gross_exposure"] == 0.0075
    assert vol_target.components[0].params["volatility_target_per_bar"] == 0.00125
    assert vol_target.components[0].params["min_volatility_exposure_multiplier"] == 0.55
    assert vol_target.components[0].params["max_volatility_exposure_multiplier"] == 1.0
    assert vol_target_132.components[0].params["gross_exposure"] == 0.0075
    assert vol_target_132.components[0].params["volatility_target_per_bar"] == 0.00132
    assert vol_target_132.components[0].params["min_volatility_exposure_multiplier"] == 0.55
    assert vol_target_132.components[0].params["max_volatility_exposure_multiplier"] == 1.0
    assert asym_dynamic.components[0].params["short_exposure_multiplier"] == 0.35
    assert asym_dynamic.components[0].params["volume_weighted_broad"] is True
    assert asym_dynamic.components[0].params["volatility_trailing_multiplier"] == 7.0
    assert volume_guard.components[0].params["long_exposure_multiplier"] == 1.15
    assert volume_guard.components[0].params["short_exposure_multiplier"] == 0.25
    assert [component.component_id for component in hybrid_return.components] == [
        "profit_reboot_adaptive_momentum_boost",
        "profit_moonshot_adaptive_momentum_vol_target_132",
        "profit_moonshot_adaptive_momentum_governed",
    ]
    assert [component.weight for component in hybrid_return.components] == [0.6, 0.25, 0.15]
    assert sum(component.weight for component in hybrid_safe.components) == 1.0
    assert [component.weight for component in hybrid_core.components] == [0.4, 0.4, 0.15, 0.05]
    assert trend.components[0].strategy_class == "ProfitMoonshotTrendStrategy"
    assert breakout.components[0].strategy_class == "ProfitMoonshotBreakoutStrategy"
    assert reversion.components[0].strategy_class == "ProfitMoonshotReversionStrategy"
    assert {component.strategy_class for component in ensemble.components} == {
        "ProfitMoonshotTrendStrategy",
        "ProfitMoonshotBreakoutStrategy",
        "ProfitMoonshotReversionStrategy",
    }
    assert {
        component.strategy_class
        for component in MODULE.resolve_portfolio_mode_definition(
            "profit_moonshot_balanced_mode"
        ).components
    } == {
        "ProfitMoonshotTrendStrategy",
        "ProfitMoonshotBreakoutStrategy",
        "ProfitMoonshotReversionStrategy",
    }
    assert sum(component.weight for component in ensemble.components) == 1.0
    for definition in (
        boost,
        ladder_120,
        ladder_130,
        ladder_140,
        governed,
        hybrid_return,
        hybrid_safe,
        hybrid_core,
        asym_dynamic,
        volume_guard,
        trend,
        breakout,
        reversion,
        ensemble,
    ):
        strategy = MODULE.ArtifactPortfolioModeStrategy(
            bars=SimpleNamespace(
                symbol_list=definition.symbols,
                get_latest_bar_value=lambda *args, **kwargs: 100.0,
                get_latest_bar_datetime=lambda *args, **kwargs: "2026-01-01T00:00:00Z",
            ),
            events=SimpleNamespace(put=lambda item: None),
            portfolio_mode=definition.portfolio_mode,
        )
        assert strategy.uses_timeframe_aggregator is False
        assert strategy.required_timeframes == ()


def test_profit_moonshot_taker_flow_exhaustion_reactive_mode_evaluates_every_window() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_taker_flow_exhaustion_eth_reactive_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_taker_flow_exhaustion_eth_reactive_mode")
    component = definition.components[0]
    assert component.strategy_class == "TakerFlowExhaustionReversalStrategy"
    assert component.params["evaluation_cadence_bars"] == 1
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0


def test_profit_moonshot_taker_flow_exhaustion_hold_mode_widens_exits_only() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_taker_flow_exhaustion_eth_hold_mode"
    )

    assert supports_live_portfolio_mode("profit_moonshot_taker_flow_exhaustion_eth_hold_mode")
    component = definition.components[0]
    assert component.strategy_class == "TakerFlowExhaustionReversalStrategy"
    assert component.params["evaluation_cadence_bars"] == 1
    assert component.params["stop_loss_pct"] == 0.050
    assert component.params["take_profit_pct"] == 0.100
    assert component.params["trailing_exit_pct"] == 0.0
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0


def test_profit_moonshot_taker_flow_exhaustion_slow_momentum_mode_adds_cooldown() -> None:
    definition = MODULE.resolve_portfolio_mode_definition(
        "profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode"
    )

    assert supports_live_portfolio_mode(
        "profit_moonshot_taker_flow_exhaustion_eth_slow_momentum_mode"
    )
    component = definition.components[0]
    assert component.strategy_class == "TakerFlowExhaustionReversalStrategy"
    assert component.params["momentum_lookback_bars"] == 360
    assert component.params["cooldown_bars"] == 2160
    assert component.params["target_allocation"] == 0.008
    assert component.params["max_order_value"] == 175.0
