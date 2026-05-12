from __future__ import annotations

import queue
from dataclasses import dataclass

from lumina_quant.core.events import MarketBatchEvent, MarketEvent
from lumina_quant.strategies.crypto_fx_alpha_zoo_state import CryptoFxAlphaZooStateStrategy
from lumina_quant.strategies.registry import get_strategy_map, get_strategy_tier


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
