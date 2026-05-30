from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lumina_quant.strategies.alpha_zoo_optuna_hybrid_live import (
    DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT,
    AlphaZooOptunaHybridLiveStrategy,
    load_alpha_zoo_optuna_hybrid_live_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_69_ops_module():
    path = ROOT / "scripts" / "ops" / "write_alpha_zoo_69_asset_efficiency_repair_live_decision.py"
    spec = importlib.util.spec_from_file_location(
        "write_alpha_zoo_69_asset_efficiency_repair_live_decision", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Queue:
    def __init__(self) -> None:
        self.items: list[object] = []

    def put(self, item: object) -> None:
        self.items.append(item)


class _Bars:
    symbol_list: list[str] = []


class _DynamicAggregator:
    def get_bars(self, symbol: str, timeframe: str, n: int | None = None, lookback_bars: int = 1):
        count = int(n if n is not None else lookback_bars)
        count = max(count, 320)
        compact = str(symbol).replace("/", "").upper()
        seed = sum(ord(ch) for ch in f"{compact}:{timeframe}")
        base = 10.0 + float(seed % 200)
        slope = ((seed % 17) - 8) * 0.0004
        rows = []
        for idx in range(count):
            drift = 1.0 + slope * idx
            wave = 1.0 + 0.002 * ((idx + seed) % 11 - 5)
            close = max(0.01, base * drift * wave)
            rows.append((idx, close, close * 1.01, close * 0.99, close, 1000.0))
        return rows[-count:]


def test_69_asset_efficiency_config_reconstructs_live_universe_and_policies() -> None:
    config = load_alpha_zoo_optuna_hybrid_live_config(
        optuna_hybrid_artifact_path=DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT
    )

    assert config.selected_profile_id == "hybrid_v3_6_optuna_three_profile_blend"
    assert config.governance["artifact_kind"] == "alpha_zoo_69_asset_efficiency_repair_optuna"
    assert config.governance["paper_testnet_only"] is True
    assert config.governance["ready_for_real"] is False
    assert config.governance["real_money_execution"] is False
    assert config.governance["real_execution_allowed"] is False
    assert len(config.source_sleeves) == 18
    assert len(config.source_profiles) == 3
    assert len(config.watch_symbols) == 69
    assert {"BTC/USDT", "SOL/USDT", "XAG/USDT", "CRCL/USDT"}.issubset(set(config.watch_symbols))
    assert {sleeve.family for sleeve in config.source_sleeves} == {
        "cross_sectional_momentum_rank",
        "volatility_adjusted_trend_persistence",
    }
    assert {sleeve.timeframe for sleeve in config.source_sleeves} == {"30m", "1h", "2h", "4h"}
    assert config.final_profile_weights == pytest.approx(
        {
            "aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna": 0.1823047019724131,
            "balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 0.6162420536737696,
            "growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 0.1809761276157996,
        }
    )
    assert config.average_profile_weights == pytest.approx(
        {
            "aggressive_mdd30_gross10_69_asset_efficiency_repair_optuna": 0.31461001642771985,
            "balanced_mdd12_gross5_69_asset_efficiency_repair_optuna": 0.35079934376469424,
            "growth_mdd20_gross8_69_asset_efficiency_repair_optuna": 0.33459063980758585,
        }
    )
    assert config.governance["live_unfilled_order_policy"]["market_fallback_allowed"] is False
    assert config.governance["live_unfilled_order_policy"]["max_chase_attempts"] == 0
    assert config.governance["live_slippage_guard_policy"]["market_fallback_allowed"] is False
    assert config.governance["live_slippage_guard_policy"]["require_bbo_snapshot"] is True
    assert config.governance["live_slippage_guard_policy"]["max_bbo_spread_bps_at_submit"] == 4.0


def test_69_asset_efficiency_live_targets_use_final_weights_and_sleeve_multiplier() -> None:
    strategy = AlphaZooOptunaHybridLiveStrategy(
        _Bars(),
        _Queue(),
        optuna_hybrid_artifact_path=DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT,
    )

    total = sum(
        strategy.target_notional_fraction_for_sleeve(sleeve)
        for sleeve in strategy.config.source_sleeves
    )

    assert total == pytest.approx(
        strategy.config.governance["live_final_weight_gross_notional_fraction"]
    )
    assert total == pytest.approx(2.338903568470546)
    assert strategy.config.governance["historical_train_validation_gross_notional_fraction"] == (
        pytest.approx(2.504195680646255)
    )


def test_69_asset_efficiency_families_evaluate_without_btc_specific_fallback() -> None:
    strategy = AlphaZooOptunaHybridLiveStrategy(
        _Bars(),
        _Queue(),
        optuna_hybrid_artifact_path=DEFAULT_69_ASSET_EFFICIENCY_REPAIR_ARTIFACT,
    )
    aggregator = _DynamicAggregator()
    cache: dict[tuple[object, ...], object] = {}

    assert {item.family for item in strategy.config.source_sleeves}
    for family in {item.family for item in strategy.config.source_sleeves}:
        sleeve = next(item for item in strategy.config.source_sleeves if item.family == family)
        decision = strategy._evaluate_sleeve(aggregator, sleeve, cache)
        assert decision is not None, family
        assert decision.signal in {-1, 0, 1}
        assert decision.price > 0.0


def test_69_asset_efficiency_decision_payload_is_limit_only_and_no_chase() -> None:
    module = _load_69_ops_module()
    payload = module.build_69_asset_efficiency_repair_decision_payload()

    assert payload["selected_mode"] == "alpha_zoo_69_asset_efficiency_repair_optuna_hybrid"
    assert payload["strategy_name"] == "AlphaZooOptunaHybridLiveStrategy"
    assert payload["strategy_params"]["allow_real_money"] is False
    assert payload["paper_testnet_only"] is True
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["real_execution_allowed"] is False
    assert payload["limit_order_contract"]["default_order_type"] == "LMT"
    assert payload["limit_order_contract"]["allow_market_orders"] is False
    assert payload["limit_order_contract"]["limit_price_mode"] == "one_tick_worse"
    assert payload["unfilled_order_policy"]["market_fallback_allowed"] is False
    assert payload["unfilled_order_policy"]["max_chase_attempts"] == 0
    assert payload["slippage_guard_policy"]["market_fallback_allowed"] is False
    assert payload["slippage_guard_policy"]["require_bbo_snapshot"] is True
    assert (
        payload["slippage_guard_policy"]["on_missing_bbo_snapshot"]
        == "do_not_submit_no_market_fallback"
    )
    assert payload["slippage_guard_policy"]["on_pre_submit_breach"] == "do_not_submit"
    assert len(payload["symbols"]) == 69
    assert payload["strategy_params"]["selected_profile_id"] == (
        "hybrid_v3_6_optuna_three_profile_blend"
    )
    assert len(payload["asset_applicability_contract"]["selected_source_symbols"]) == 13
