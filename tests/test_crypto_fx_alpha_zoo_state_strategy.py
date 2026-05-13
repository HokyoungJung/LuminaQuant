from __future__ import annotations

import queue
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lumina_quant.core.events import MarketBatchEvent, MarketEvent
from lumina_quant.strategies.crypto_fx_alpha_zoo_state import CryptoFxAlphaZooStateStrategy
from lumina_quant.strategies.registry import get_strategy_map, get_strategy_tier

_REPLAY_SPEC = importlib.util.spec_from_file_location("replay_crypto_fx_alpha_zoo_state", Path("scripts/research/replay_crypto_fx_alpha_zoo_state.py"))
_REPLAY_MODULE = importlib.util.module_from_spec(_REPLAY_SPEC)
assert _REPLAY_SPEC.loader is not None
sys.modules[_REPLAY_SPEC.name] = _REPLAY_MODULE
_REPLAY_SPEC.loader.exec_module(_REPLAY_MODULE)
_GridSpec = _REPLAY_MODULE._GridSpec
_liquidation_lanes = _REPLAY_MODULE._liquidation_lanes
replay_frame = _REPLAY_MODULE.replay_frame


@dataclass(slots=True)
class _Bars:
    symbol_list: list[str]


def _batch(ts: int, eth_mult: float = 1.0, sol_mult: float = 1.0) -> MarketBatchEvent:
    btc = 100.0 * (1.0 + 0.001 * ts)
    eth = 100.0 * (1.0 + 0.006 * ts) * eth_mult
    sol = 100.0 * (1.0 - 0.004 * ts) * sol_mult
    eurusd = 1.10 * (1.0 - 0.0001 * ts)
    gbpusd = 1.25 * (1.0 - 0.0001 * ts)
    audusd = 0.65 * (1.0 + 0.0002 * ts)
    usdjpy = 145.0 * (1.0 + 0.0001 * ts)
    bars = []
    for symbol, close, volume in (
        ("BTC/USDT", btc, 1000.0 + ts),
        ("ETH/USDT", eth, 1200.0 + 20.0 * ts),
        ("SOL/USDT", sol, 1100.0 + 5.0 * ts),
        ("EURUSD", eurusd, 0.0),
        ("GBPUSD", gbpusd, 0.0),
        ("AUDUSD", audusd, 0.0),
        ("USDJPY", usdjpy, 0.0),
    ):
        bars.append(MarketEvent(ts, symbol, close * 0.999, close * 1.002, close * 0.998, close, volume))
    return MarketBatchEvent(time=ts, bars=tuple(bars))


def _run_strategy(*, require_edge: bool = True, calibrated_edges: dict[str, float] | None = None):
    events: queue.Queue = queue.Queue()
    strategy = CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]),
        events,
        fast_lookback_bars=2,
        slow_lookback_bars=8,
        history_window=16,
        entry_threshold=0.15,
        exit_threshold=0.02,
        use_fx_filter=False,
        require_calibrated_edge=require_edge,
        calibrated_edges=calibrated_edges or {},
        max_longs=1,
        max_shorts=1,
    )
    for ts in range(30):
        strategy.calculate_signals(_batch(ts))
    signals = []
    while not events.empty():
        signals.append(events.get())
    return strategy, signals


def test_strategy_is_registered_live_opt_in_and_calendar_safe() -> None:
    strategy_map = get_strategy_map()
    assert strategy_map["CryptoFxAlphaZooStateStrategy"] is CryptoFxAlphaZooStateStrategy
    assert get_strategy_tier("CryptoFxAlphaZooStateStrategy") == "live_opt_in"
    assert CryptoFxAlphaZooStateStrategy.strategy_validity["calendar_primary"] is False
    assert CryptoFxAlphaZooStateStrategy.strategy_validity["locked_oos_role"] == "gate_report_only"


def test_strategy_requires_calibrated_edge_for_entries() -> None:
    _, blocked = _run_strategy(require_edge=True, calibrated_edges={})
    assert [signal for signal in blocked if signal.signal_type in {"LONG", "SHORT"}] == []

    _, allowed = _run_strategy(require_edge=True, calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0})
    assert any(signal.signal_type in {"LONG", "SHORT"} for signal in allowed)
    first_entry = next(signal for signal in allowed if signal.signal_type in {"LONG", "SHORT"})
    assert first_entry.symbol.endswith("USDT")
    assert first_entry.metadata["uses_locked_oos_for_selection"] is False
    assert first_entry.metadata["calibrated_lower_bound_edge_bps"] == 5.0


def test_strategy_state_roundtrip_preserves_signal_sequence() -> None:
    events_full: queue.Queue = queue.Queue()
    full = CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]),
        events_full,
        fast_lookback_bars=2,
        slow_lookback_bars=8,
        history_window=16,
        entry_threshold=0.15,
        use_fx_filter=False,
        calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0},
    )
    for ts in range(30):
        full.calculate_signals(_batch(ts))
    full_signals = [(item.datetime, item.symbol, item.signal_type) for item in list(events_full.queue)]

    events_split: queue.Queue = queue.Queue()
    split_a = CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]),
        events_split,
        fast_lookback_bars=2,
        slow_lookback_bars=8,
        history_window=16,
        entry_threshold=0.15,
        use_fx_filter=False,
        calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0},
    )
    for ts in range(15):
        split_a.calculate_signals(_batch(ts))
    state = split_a.get_state()
    split_b = CryptoFxAlphaZooStateStrategy(
        _Bars(["BTC/USDT", "ETH/USDT", "SOL/USDT", "EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]),
        events_split,
        fast_lookback_bars=2,
        slow_lookback_bars=8,
        history_window=16,
        entry_threshold=0.15,
        use_fx_filter=False,
        calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0},
    )
    split_b.set_state(state)
    for ts in range(15, 30):
        split_b.calculate_signals(_batch(ts))
    split_signals = [(item.datetime, item.symbol, item.signal_type) for item in list(events_split.queue)]
    assert split_signals == full_signals


def test_replay_grid_hides_locked_oos_until_after_train_validation_selection() -> None:
    rows = []
    for ts in range(60):
        timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=ts)
        split = "train" if ts < 30 else "validation" if ts < 45 else "locked_oos"
        for symbol, drift in (("BTC/USDT", 0.001), ("ETH/USDT", 0.004), ("SOL/USDT", -0.002)):
            close = 100.0 * (1.0 + drift * ts)
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 1000.0 + ts,
                    "split": split,
                }
            )
    payload = replay_frame(
        pd.DataFrame(rows),
        require_calibrated_edge=True,
        calibrated_edges={"default:LONG": 5.0, "default:SHORT": 5.0},
        strategy_params={"fast_lookback_bars": 2, "slow_lookback_bars": 8, "history_window": 16, "entry_threshold": 0.15},
        grid_specs=[
            _GridSpec("low_threshold", "unit", {"entry_threshold": 0.15}),
            _GridSpec("higher_threshold", "unit", {"entry_threshold": 0.30}),
        ],
    )
    grid = payload["candidate_selection_grid"]
    assert grid["uses_locked_oos_for_selection"] is False
    assert grid["locked_oos_calibration_record_count"] == 0
    for row in grid["rows"]:
        assert set(row["selection_metrics"]) == {"train", "validation"}
        assert row["locked_oos_metrics_visible_during_selection"] is False
        assert row["uses_locked_oos_for_selection"] is False
    assert payload["selection_provenance"]["candidate_freeze_before_locked_oos_gate"] is True


def test_revised_promotion_gate_treats_return_mdd_as_diagnostic_only() -> None:
    data = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "split"])
    trades = [
        {
            "symbol": "BTC/USDT",
            "side": "LONG",
            "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
            "exit_time": pd.Timestamp("2026-01-01T01:00:00Z"),
            "entry_price": 100.0,
            "entry_split": "locked_oos",
            "gross_return": gross_return,
        }
        for gross_return in (0.20, -0.05, 0.02)
    ]

    lanes = _liquidation_lanes(data, trades, allocation_fraction=1.0, max_leverage=1)
    promoted = lanes["strict_zero_liquidation_lane"]["promoted_candidate"]

    assert promoted["deployable_success"] is True
    assert promoted["performance_gates"]["oos_return_beats_current_base"] is True
    assert "oos_return_risk_beats_current_base" not in promoted["performance_gates"]
    assert promoted["performance_diagnostics"]["return_mdd_hurdle_required"] is False
    assert promoted["performance_diagnostics"]["oos_return_mdd_beats_current_base"] is False
