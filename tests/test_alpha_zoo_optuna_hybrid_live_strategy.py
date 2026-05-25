from __future__ import annotations

import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from lumina_quant.core.events import MarketEvent
from lumina_quant.services.portfolio import PortfolioSizingService
from lumina_quant.live_selection import extract_live_decision_config
from lumina_quant.strategies.alpha_zoo_optuna_hybrid_live import (
    DEFAULT_INTEGER_PORTFOLIO_ARTIFACT,
    DEFAULT_OPTUNA_HYBRID_ARTIFACT,
    AlphaZooOptunaHybridLiveStrategy,
    completed_bars_only,
    debounced_state_signal,
    load_alpha_zoo_optuna_hybrid_live_config,
)
from lumina_quant.strategies.registry import get_strategy_tier, resolve_strategy_class

ROOT = Path(__file__).resolve().parents[1]


def _load_ops_module():
    path = ROOT / "scripts" / "ops" / "write_alpha_zoo_optuna_hybrid_live_decision.py"
    spec = importlib.util.spec_from_file_location("write_alpha_zoo_optuna_hybrid_live_decision", path)
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
    symbol_list = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "TRX/USDT"]


class _Aggregator:
    def __init__(self, bars: dict[tuple[str, str], list[tuple[object, float, float, float, float, float]]]) -> None:
        self._bars = bars

    def get_bars(self, symbol: str, timeframe: str, n: int | None = None, lookback_bars: int = 1):
        rows = list(self._bars.get((symbol, timeframe), []))
        if n is None:
            n = lookback_bars
        return rows[-int(n) :]


def _ohlcv_rows(closes: list[float]) -> list[tuple[object, float, float, float, float, float]]:
    rows = []
    for idx, close in enumerate(closes):
        rows.append((idx, close, close * 1.01, close * 0.99, close, 1000.0))
    return rows


def _bars_with_aliases(symbol: str, timeframe: str, rows: list[tuple[object, float, float, float, float, float]]):
    compact = symbol.replace("/", "")
    live = f"{compact[:-4]}/USDT" if compact.endswith("USDT") else compact
    return {(compact, timeframe): rows, (live, timeframe): rows}


def test_load_config_validates_frozen_artifacts_and_registers_strategy() -> None:
    config = load_alpha_zoo_optuna_hybrid_live_config()

    assert config.governance["paper_testnet_only"] is True
    assert config.governance["ready_for_real"] is False
    assert config.governance["real_money_execution"] is False
    assert config.governance["real_execution_allowed"] is False
    assert config.governance["research_primary_round_trip_cost_bps"] == 10.0
    assert len(config.source_sleeves) == 6
    assert {profile.profile_id for profile in config.source_profiles} == {
        "balanced_mdd12_gross5",
        "growth_mdd20_gross8",
        "aggressive_mdd30_gross10_shadow",
    }
    assert resolve_strategy_class("AlphaZooOptunaHybridLiveStrategy") is AlphaZooOptunaHybridLiveStrategy
    assert get_strategy_tier("AlphaZooOptunaHybridLiveStrategy") == "live_opt_in"
    assert {"1m", "5m", "1h", "2h", "4h"}.issubset(
        set(AlphaZooOptunaHybridLiveStrategy.required_timeframes)
    )


def test_fractional_source_integer_leverage_fails_closed(tmp_path: Path) -> None:
    mutated = json.loads((ROOT / DEFAULT_INTEGER_PORTFOLIO_ARTIFACT).read_text())
    mutated["profile_decision_rows"][0]["leverage_map"]["SOLUSDT"] = 2.5
    path = tmp_path / "integer.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(ValueError, match="source leverage"):
        load_alpha_zoo_optuna_hybrid_live_config(integer_portfolio_artifact_path=path)


def test_completed_bars_exclude_active_working_bar() -> None:
    completed = _ohlcv_rows([100.0, 99.0, 98.0])
    active = [(3, 100.0, 150.0, 50.0, 150.0, 1000.0)]
    aggregator = _Aggregator(_bars_with_aliases("SOLUSDT", "1h", completed + active))

    rows = completed_bars_only(aggregator, "SOLUSDT", "1h", 4)

    assert rows == completed
    assert rows[-1][4] == 98.0


def test_debounced_state_signal_honors_min_hold_and_cooldown() -> None:
    long_entry = pd.Series([False] * 8)
    long_exit = pd.Series([False] * 8)
    short_entry = pd.Series([True, False, False, False, False, True, False, False])
    short_exit = pd.Series([False, True, True, True, False, False, True, False])

    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side="short_only",
        min_hold_bars=3,
        cooldown_bars=1,
    )

    assert signal.tolist() == [-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0]


def test_strategy_emits_paper_testnet_short_signal_from_completed_1h_bars() -> None:
    completed_sol = _ohlcv_rows([100.0 - idx * 0.8 for idx in range(70)])
    intrabar_sol = _ohlcv_rows([100.0 - idx * 0.1 for idx in range(32)])
    active_sol = [(70, 55.0, 200.0, 55.0, 200.0, 1000.0)]
    completed_btc = _ohlcv_rows([100.0 for _ in range(70)])
    intrabar_btc = _ohlcv_rows([100.0 for _ in range(32)])
    active_btc = [(70, 100.0, 100.0, 100.0, 100.0, 1000.0)]
    bars = {}
    bars.update(_bars_with_aliases("SOLUSDT", "1h", completed_sol + active_sol))
    bars.update(_bars_with_aliases("SOLUSDT", "1m", [*intrabar_sol, active_sol[-1]]))
    bars.update(_bars_with_aliases("BTCUSDT", "1h", completed_btc + active_btc))
    bars.update(_bars_with_aliases("BTCUSDT", "1m", [*intrabar_btc, active_btc[-1]]))
    aggregator = _Aggregator(bars)
    queue = _Queue()
    strategy = AlphaZooOptunaHybridLiveStrategy(_Bars(), queue)

    strategy.calculate_signals_window(object(), aggregator)

    short_signals = [item for item in queue.items if getattr(item, "signal_type", "") == "SHORT"]
    assert short_signals, "expected at least one SOL short source sleeve to emit"
    signal = short_signals[0]
    assert signal.symbol == "SOL/USDT"
    assert signal.price != 200.0
    assert signal.strength > 0.0
    assert signal.metadata["paper_testnet_only"] is True
    assert signal.metadata["ready_for_real"] is False
    assert signal.metadata["real_money_execution"] is False
    assert signal.metadata["real_execution_allowed"] is False
    assert signal.metadata["round_trip_cost_bps"] == 10.0
    assert signal.metadata["target_notional_formula"] == "allocation_fraction*sum(profile_weight*integer_leverage)"
    assert signal.metadata["target_allocation"] == pytest.approx(
        signal.metadata["target_notional_fraction"]
    )
    assert signal.metadata["target_allocation_mode"] == "notional_fraction"
    assert signal.metadata["component_id"] == signal.metadata["source_model_id"]
    assert signal.metadata["intrabar_protection_enabled"] is True
    assert signal.metadata["intrabar_protection"]["source_timeframe"] == "1m"
    assert signal.metadata["intrabar_protection"]["stop_loss"] == pytest.approx(signal.stop_loss)
    assert signal.stop_loss is not None and signal.stop_loss > float(signal.price)
    assert "bbo_spread_bps_at_submit" in signal.metadata["microstructure_telemetry_required"]


def test_intrabar_guard_emits_component_exit_from_market_event() -> None:
    completed_sol = _ohlcv_rows([100.0 - idx * 0.8 for idx in range(70)])
    intrabar_sol = _ohlcv_rows([100.0 - idx * 0.1 for idx in range(32)])
    active_sol = [(70, 55.0, 200.0, 55.0, 200.0, 1000.0)]
    completed_btc = _ohlcv_rows([100.0 for _ in range(70)])
    active_btc = [(70, 100.0, 100.0, 100.0, 100.0, 1000.0)]
    bars = {}
    bars.update(_bars_with_aliases("SOLUSDT", "1h", completed_sol + active_sol))
    bars.update(_bars_with_aliases("SOLUSDT", "1m", [*intrabar_sol, active_sol[-1]]))
    bars.update(_bars_with_aliases("BTCUSDT", "1h", completed_btc + active_btc))
    aggregator = _Aggregator(bars)
    queue = _Queue()
    strategy = AlphaZooOptunaHybridLiveStrategy(_Bars(), queue)

    strategy.calculate_signals_window(object(), aggregator)
    short_entries = [item for item in queue.items if getattr(item, "signal_type", "") == "SHORT"]
    entry = short_entries[0]
    stop = float(entry.stop_loss)
    strategy.calculate_signals(
        MarketEvent(
            time="risk-tick",
            symbol="SOL/USDT",
            open=stop * 0.99,
            high=stop * 1.01,
            low=stop * 0.98,
            close=stop,
            volume=1000.0,
        )
    )

    exits = [item for item in queue.items if getattr(item, "signal_type", "") == "EXIT"]
    assert exits
    exit_signal = exits[-1]
    assert exit_signal.position_side == "SHORT"
    assert exit_signal.metadata["component_id"] in {
        item.metadata["component_id"] for item in short_entries
    }
    assert exit_signal.metadata["intrabar_exit_reason"] == "intrabar_stop_loss_or_trailing_short"


def test_intrabar_protection_is_generic_for_selected_eth_sol_trx_sleeves() -> None:
    selected_symbols = set()
    for sleeve in load_alpha_zoo_optuna_hybrid_live_config().source_sleeves:
        selected_symbols.add(sleeve.symbol)
        price = 0.1 if sleeve.symbol == "TRXUSDT" else 100.0
        intrabar_rows = _ohlcv_rows([price + idx * price * 0.001 for idx in range(32)])
        active_row = (32, price, price, price, price, 1000.0)
        aggregator = _Aggregator(_bars_with_aliases(sleeve.symbol, "1m", [*intrabar_rows, active_row]))
        queue = _Queue()
        strategy = AlphaZooOptunaHybridLiveStrategy(_Bars(), queue)
        signal_type = "SHORT" if sleeve.side == "short_only" else "LONG"
        decision = SimpleNamespace(price=price, completed_key=f"asset-check-{sleeve.symbol}")

        plan = strategy._build_intrabar_protection_plan(aggregator, sleeve, decision, signal_type)
        assert plan.enabled is True
        assert plan.source_timeframe == "1m"
        strategy._activate_intrabar_guard(sleeve, decision, signal_type, plan)
        stop = float(plan.stop_loss)
        if signal_type == "SHORT":
            high = stop * 1.01
            low = stop * 0.99
        else:
            high = stop * 1.01
            low = stop * 0.99
        strategy.calculate_signals(
            MarketEvent(
                time=f"risk-{sleeve.symbol}",
                symbol=sleeve.symbol,
                open=price,
                high=high,
                low=low,
                close=stop,
                volume=1000.0,
            )
        )
        exits = [item for item in queue.items if getattr(item, "signal_type", "") == "EXIT"]
        assert exits, sleeve.symbol
        assert exits[-1].metadata["component_id"] == sleeve.model_id

    assert selected_symbols == {"ETHUSDT", "SOLUSDT", "TRXUSDT"}


def test_target_notional_uses_final_weights_and_integer_leverage() -> None:
    queue = _Queue()
    strategy = AlphaZooOptunaHybridLiveStrategy(_Bars(), queue)
    sleeve = next(item for item in strategy.config.source_sleeves if item.symbol == "SOLUSDT")

    actual = strategy.target_notional_fraction_for_sleeve(sleeve)
    expected = sleeve.allocation_fraction * (
        0.07983098667432496 * 2 + 0.08067156944866473 * 4 + 0.5726993181554131 * 4
    )

    assert actual == pytest.approx(expected)


def test_signal_metadata_sizes_as_live_notional_fraction() -> None:
    completed_sol = _ohlcv_rows([100.0 - idx * 0.8 for idx in range(70)])
    active_sol = [(70, 55.0, 200.0, 55.0, 200.0, 1000.0)]
    completed_btc = _ohlcv_rows([100.0 for _ in range(70)])
    active_btc = [(70, 100.0, 100.0, 100.0, 100.0, 1000.0)]
    bars = {}
    bars.update(_bars_with_aliases("SOLUSDT", "1h", completed_sol + active_sol))
    bars.update(_bars_with_aliases("BTCUSDT", "1h", completed_btc + active_btc))
    queue = _Queue()
    strategy = AlphaZooOptunaHybridLiveStrategy(_Bars(), queue)

    strategy.calculate_signals_window(object(), _Aggregator(bars))

    signal = next(item for item in queue.items if getattr(item, "signal_type", "") == "SHORT")
    equity = 10_000.0
    quantity = PortfolioSizingService.risk_based_quantity(
        signal=signal,
        current_price=float(signal.price),
        equity=equity,
        risk_per_trade=0.001,
        default_stop_loss_pct=0.025,
        max_symbol_exposure_pct=5.0,
        target_allocation=0.0,
        max_order_value=0.0,
        target_allocation_mode="notional_fraction",
        leverage=12.0,
        max_order_notional_pct=5.0,
    )

    expected_notional = equity * float(signal.metadata["target_notional_fraction"])
    assert quantity * float(signal.price) == pytest.approx(expected_notional)


def test_ops_decision_payload_is_paper_testnet_only() -> None:
    module = _load_ops_module()
    payload = module.build_decision_payload()

    assert payload["strategy_name"] == "AlphaZooOptunaHybridLiveStrategy"
    assert payload["paper_testnet_only"] is True
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["real_execution_allowed"] is False
    assert payload["exchange"]["testnet"] is True
    assert payload["strategy_params"]["allow_real_money"] is False
    assert payload["research_primary_round_trip_cost_bps"] == 10.0
    assert payload["target_allocation"] == 0.0
    assert payload["target_allocation_mode"] == "notional_fraction"
    assert payload["sizing_mode"] == "notional_fraction"
    assert payload["risk_caps"]["max_order_value"] == 0.0
    assert payload["risk_caps"]["max_order_notional_pct"] > 1.18
    assert payload["risk_caps"]["max_symbol_exposure_pct"] > 1.35
    assert payload["risk_caps"]["max_total_notional_pct"] > 3.2
    assert any("no_exchange_paper_fill_telemetry" in item for item in payload["real_money_blockers"])
    assert any("backtest_cost_is_proxy" in item for item in payload["real_money_blockers"])
    assert any("Validation MDD" in item for item in payload["known_limitations"])
    assert "minimum 2 weeks paper/testnet observation before any real-money review" in payload[
        "paper_testnet_validation_requirements"
    ]
    assert payload["intrabar_protection_contract"]["enabled"] is True
    assert payload["intrabar_protection_contract"]["component_exit_key"] == (
        "SignalEvent.metadata.component_id"
    )
    assert payload["microstructure_telemetry_contract"]["required_fields"][0] == (
        "bbo_spread_bps_at_submit"
    )
    assert payload["paper_testnet_exchange_protection_contract"]["enabled"] is True
    assert payload["paper_testnet_exchange_protection_contract"]["endpoint"] == (
        "POST /fapi/v1/algoOrder"
    )
    assert payload["paper_testnet_exchange_protection_contract"]["real_money_policy"] == (
        "blocked_until_separate_exchange_side_order_telemetry_review"
    )
    assert payload["asset_applicability_contract"]["verified_symbols"] == [
        "ETHUSDT",
        "SOLUSDT",
        "TRXUSDT",
    ]
    runtime = extract_live_decision_config(payload)
    assert runtime["strategy_name"] == "AlphaZooOptunaHybridLiveStrategy"
    assert runtime["target_allocation"] == 0.0
    assert runtime["target_allocation_mode"] == "notional_fraction"
    assert runtime["max_total_margin_pct"] > 3.2


def test_adapter_rule_logic_has_no_calendar_rule_tokens() -> None:
    source = (
        (ROOT / "src/lumina_quant/strategies/alpha_zoo_optuna_hybrid_live.py").read_text()
        + (ROOT / "src/lumina_quant/alpha_zoo/optuna_hybrid_live_strategy.py").read_text()
    ).lower()
    forbidden = ["dayofweek", "weekday", "isocalendar", "month ==", "strftime", "timedelta"]
    assert not [token for token in forbidden if token in source]
