from __future__ import annotations

import queue
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from lumina_quant.core.events import MarketEvent, MarketWindowEvent
from lumina_quant.strategies.panic_rebound_mean_reversion import (
    PanicReboundMeanReversionStrategy,
)
from lumina_quant.strategies.profit_moonshot import (
    ProfitMoonshotBreakoutStrategy,
    ProfitMoonshotReversionStrategy,
    ProfitMoonshotTrendStrategy,
)
from lumina_quant.strategies.session_filtered_pair_carry import (
    SessionFilteredPairCarryStrategy,
)


@dataclass
class _BarStore:
    symbol_list: list[str]

    def __post_init__(self) -> None:
        self._latest: dict[str, dict[str, float | datetime]] = {
            symbol: {
                "time": datetime(2026, 1, 1, tzinfo=UTC),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 100.0,
            }
            for symbol in self.symbol_list
        }

    def set_bar(
        self,
        symbol: str,
        when: datetime,
        close_price: float,
        *,
        volume: float = 100.0,
    ) -> None:
        self._latest[symbol] = {
            "time": when,
            "open": close_price,
            "high": close_price * 1.002,
            "low": close_price * 0.998,
            "close": close_price,
            "volume": volume,
        }

    def get_latest_bar_value(self, symbol: str, value_type: str) -> float:
        return float(self._latest[symbol][value_type])

    def get_latest_bar_datetime(self, symbol: str) -> datetime:
        return self._latest[symbol]["time"]  # type: ignore[return-value]


def _drain(events: queue.Queue) -> list:
    out = []
    while not events.empty():
        out.append(events.get())
    return out


def _window_event(
    when: datetime,
    frame: dict[str, tuple[float, float]],
) -> MarketWindowEvent:
    return MarketWindowEvent(
        time=when,
        window_seconds=60,
        bars_1s={
            symbol: (
                (
                    when,
                    close,
                    close * 1.002,
                    close * 0.998,
                    close,
                    volume,
                ),
            )
            for symbol, (close, volume) in frame.items()
        },
    )


def _feed_window_frames(
    strategy,
    bars: _BarStore,
    frames: list[tuple[datetime, dict[str, tuple[float, float]]]],
) -> list:
    emitted = []
    for when, frame in frames:
        for symbol, (close, volume) in frame.items():
            bars.set_bar(symbol, when, close, volume=volume)
        strategy.calculate_signals_window(_window_event(when, frame), aggregator=None)
        emitted.extend(_drain(strategy.events))
    return emitted


def test_profit_moonshot_trend_uses_market_window_without_aggregator_and_emits_long_short() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    bars = _BarStore(symbols)
    events: queue.Queue = queue.Queue()
    strategy = ProfitMoonshotTrendStrategy(
        bars,
        events,
        lookback_bars=3,
        fast_lookback_bars=1,
        slow_lookback_bars=5,
        rebalance_bars=1,
        entry_threshold=0.001,
        max_longs=1,
        max_shorts=1,
        gross_exposure=0.60,
        max_order_value=400.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.12,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    frames = []
    for idx in range(7):
        frames.append(
            (
                start + timedelta(minutes=idx),
                {
                    "BTC/USDT": (100.0 + idx * 2.0, 100.0),
                    "ETH/USDT": (120.0 - idx * 1.8, 100.0),
                    "SOL/USDT": (80.0 + idx * 0.1, 100.0),
                },
            )
        )

    signals = _feed_window_frames(strategy, bars, frames)

    assert strategy.uses_timeframe_aggregator is False
    assert strategy.preferred_contract == "market_window"
    assert {(signal.symbol, signal.signal_type) for signal in signals} == {
        ("BTC/USDT", "LONG"),
        ("ETH/USDT", "SHORT"),
    }
    assert all(signal.metadata["target_allocation"] == 0.30 for signal in signals)
    assert all(signal.metadata["max_order_value"] == 400.0 for signal in signals)


def test_profit_moonshot_breakout_emits_expansion_signal_and_roundtrips_state() -> None:
    bars = _BarStore(["BTC/USDT"])
    events: queue.Queue = queue.Queue()
    params = {
        "lookback_bars": 4,
        "fast_lookback_bars": 2,
        "slow_lookback_bars": 5,
        "rebalance_bars": 1,
        "entry_threshold": 0.003,
        "breakout_buffer": 0.001,
        "squeeze_ratio_max": 10.0,
        "volume_z_min": -10.0,
        "max_longs": 1,
        "max_shorts": 0,
        "gross_exposure": 0.25,
        "max_order_value": 300.0,
    }
    strategy = ProfitMoonshotBreakoutStrategy(bars, events, **params)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    frames = [
        (start + timedelta(minutes=idx), {"BTC/USDT": (100.0 + (0.03 * (idx % 2)), 10.0)})
        for idx in range(6)
    ]
    frames.append((start + timedelta(minutes=6), {"BTC/USDT": (101.2, 18.0)}))

    signals = _feed_window_frames(strategy, bars, frames)
    snapshot = strategy.get_state()
    clone = ProfitMoonshotBreakoutStrategy(bars, queue.Queue(), **params)
    clone.set_state(snapshot)

    assert [signal.signal_type for signal in signals] == ["LONG"]
    assert signals[0].position_side == "LONG"
    assert signals[0].metadata["target_allocation_scale"] == 0.25
    assert signals[0].metadata["strategy"] == "ProfitMoonshotBreakoutStrategy"
    assert clone.get_state() == snapshot


def test_profit_moonshot_reversion_fades_volume_range_shock() -> None:
    bars = _BarStore(["BTC/USDT"])
    events: queue.Queue = queue.Queue()
    strategy = ProfitMoonshotReversionStrategy(
        bars,
        events,
        lookback_bars=8,
        fast_lookback_bars=2,
        slow_lookback_bars=10,
        rebalance_bars=1,
        entry_threshold=0.50,
        return_z_min=1.0,
        volume_z_min=0.0,
        range_z_min=0.0,
        max_longs=1,
        max_shorts=0,
        gross_exposure=0.20,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    frames = [
        (start + timedelta(minutes=idx), {"BTC/USDT": (100.0 + 0.05 * (idx % 2), 100.0)})
        for idx in range(9)
    ]
    frames.append((start + timedelta(minutes=9), {"BTC/USDT": (94.0, 900.0)}))

    signals = _feed_window_frames(strategy, bars, frames)

    assert [signal.signal_type for signal in signals] == ["LONG"]
    assert signals[0].metadata["strategy"] == "ProfitMoonshotReversionStrategy"
    assert signals[0].metadata["target_allocation"] == 0.20
    assert signals[0].stop_loss == 94.0 * 0.97
    assert signals[0].take_profit == 94.0 * 1.06


def test_panic_rebound_requires_later_confirmation_before_long_entry() -> None:
    bars = _BarStore(["BTC/USDT"])
    events: queue.Queue = queue.Queue()
    strategy = PanicReboundMeanReversionStrategy(
        bars,
        events,
        return_window=4,
        volume_window=4,
        vwap_window=3,
        shock_return_z=0.5,
        shock_return_pct=0.03,
        volume_z=0.5,
        confirmation_bars=3,
        min_rebound_pct=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_hold_bars=4,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    closes = [100.0, 101.0, 100.4, 101.2, 100.7, 101.0, 100.8, 100.9]
    for idx, close in enumerate(closes):
        when = start + timedelta(minutes=idx)
        bars.set_bar("BTC/USDT", when, close, volume=100.0 + idx * 3.0)
        strategy.calculate_signals(
            MarketEvent(when, "BTC/USDT", close, close, close, close, 100.0 + idx * 3.0)
        )
    assert _drain(events) == []

    shock_time = start + timedelta(minutes=len(closes))
    bars.set_bar("BTC/USDT", shock_time, 94.0, volume=500.0)
    strategy.calculate_signals(MarketEvent(shock_time, "BTC/USDT", 94.0, 94.2, 93.8, 94.0, 500.0))
    assert _drain(events) == []

    confirm_time = shock_time + timedelta(minutes=1)
    bars.set_bar("BTC/USDT", confirm_time, 96.0, volume=160.0)
    strategy.calculate_signals(MarketEvent(confirm_time, "BTC/USDT", 96.0, 96.2, 95.8, 96.0, 160.0))
    signals = _drain(events)

    assert [signal.signal_type for signal in signals] == ["LONG"]
    assert signals[0].metadata["reason"] == "confirmed_panic_rebound"
    assert signals[0].metadata["target_allocation"] > 0.0

    exit_time = confirm_time + timedelta(minutes=1)
    bars.set_bar("BTC/USDT", exit_time, 100.0, volume=120.0)
    strategy.calculate_signals(MarketEvent(exit_time, "BTC/USDT", 100.0, 100.2, 99.8, 100.0, 120.0))
    exit_signals = _drain(events)

    assert [signal.signal_type for signal in exit_signals] == ["EXIT"]
    assert exit_signals[0].metadata["reason"] == "take_profit"


def test_session_filtered_pair_carry_blocks_disallowed_entry_sessions() -> None:
    bars = _BarStore(["BNB/USDT", "TRX/USDT"])
    events: queue.Queue = queue.Queue()
    strategy = SessionFilteredPairCarryStrategy(
        bars,
        events,
        symbol_x="BNB/USDT",
        symbol_y="TRX/USDT",
        lookback_window=4,
        hedge_window=4,
        allowed_session_utc_hours="8,9",
        min_expected_move_pct=0.0,
    )

    assert strategy._entry_gate_passed(datetime(2026, 1, 1, 8, tzinfo=UTC), 2.5, 1.0)
    assert not strategy._entry_gate_passed(datetime(2026, 1, 1, 3, tzinfo=UTC), 2.5, 1.0)
