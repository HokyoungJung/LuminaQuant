"""Deterministic tests for PrevDayBoxQuartileReversionStrategy.

Synthetic 15-minute bars only (no randomness, no backtest).  Day 1 builds a box
of high=110 / low=90 with a flat 15m volume median of 10, so the quartiles are
q25=95, mid=100, q75=105.  Day 2 then exercises the long rebound, the one
signal-per-session cap, the take-profit and stop-loss exits, the volume filter,
the short mirror, the UTC day-end flatten, state round-trip, and registry
presence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.core.events import MarketEvent, MarketWindowEvent
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.strategies.prev_day_box_quartile_reversion import (
    PrevDayBoxQuartileReversionStrategy,
)

_SYMBOL = "BTC/USDT"
_DAY1 = datetime(2026, 1, 1, tzinfo=UTC)
_BOX_HIGH = 110.0
_BOX_LOW = 90.0
_Q25 = 95.0
_MID = 100.0
_Q75 = 105.0


class _Bars:
    """Minimal bars stub exposing ``symbol_list``."""

    def __init__(self, symbol: str = _SYMBOL) -> None:
        self.symbol_list = [symbol]


class _Events:
    """Capturing queue compatible with ``events.put(SignalEvent)``."""

    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)

    def drain(self) -> list[Any]:
        out = list(self.signals)
        self.signals.clear()
        return out


def _iso(day_offset: int, bar_index: int) -> str:
    stamp = _DAY1 + timedelta(days=day_offset, minutes=15 * bar_index)
    return stamp.isoformat().replace("+00:00", "Z")


def _bar(
    time: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float | None = 10.0,
) -> MarketEvent:
    return MarketEvent(
        time=time,
        symbol=_SYMBOL,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _day1_bars() -> list[MarketEvent]:
    """96 flat 15m bars whose extremes are exactly 110 / 90 with volume 10 each."""
    bars: list[MarketEvent] = []
    for index in range(96):
        high = _BOX_HIGH if index == 10 else 101.0
        low = _BOX_LOW if index == 20 else 99.0
        bars.append(_bar(_iso(0, index), 100.0, high, low, 100.0, 10.0))
    return bars


def _neutral_day2_open() -> MarketEvent:
    """First bar of day 2: triggers the rollover, touches no quartile."""
    return _bar(_iso(1, 0), 100.0, 100.5, 99.5, 100.0, 10.0)


def _long_setup_bar(bar_index: int = 1, volume: float | None = 20.0) -> MarketEvent:
    """Pierce q25=95 and close back above it with lower wick 1.5 >= body 0.5."""
    return _bar(_iso(1, bar_index), 95.5, 96.2, 94.0, 96.0, volume)


def _short_setup_bar(bar_index: int = 1, volume: float | None = 20.0) -> MarketEvent:
    """Pierce q75=105 and close back below it with upper wick 1.5 >= body 0.5."""
    return _bar(_iso(1, bar_index), 104.5, 106.0, 103.8, 104.0, volume)


def _build(**params: Any) -> tuple[PrevDayBoxQuartileReversionStrategy, _Events]:
    events = _Events()
    strategy = PrevDayBoxQuartileReversionStrategy(_Bars(), events, **params)
    return strategy, events


def _feed(strategy: PrevDayBoxQuartileReversionStrategy, bars: list[MarketEvent]) -> None:
    for bar in bars:
        strategy.calculate_signals(bar)


def _primed(**params: Any) -> tuple[PrevDayBoxQuartileReversionStrategy, _Events]:
    """Strategy after day 1 plus the day-2 rollover bar, with the queue drained."""
    strategy, events = _build(**params)
    _feed(strategy, _day1_bars())
    assert events.signals == []
    strategy.calculate_signals(_neutral_day2_open())
    assert events.drain() == []
    return strategy, events


def test_registry_exposes_strategy() -> None:
    assert (
        GLOBAL_REGISTRY.get("strategy", "PrevDayBoxQuartileReversionStrategy")
        is PrevDayBoxQuartileReversionStrategy
    )


def test_no_signal_before_a_box_exists() -> None:
    strategy, events = _build()
    _feed(strategy, _day1_bars())
    assert events.signals == []


def test_long_rebound_then_single_signal_then_take_profit() -> None:
    strategy, events = _primed()

    strategy.calculate_signals(_long_setup_bar())
    entries = events.drain()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.signal_type == "LONG"
    assert entry.stop_loss is None
    assert entry.take_profit is None
    assert entry.metadata["stop_price"] == _BOX_LOW
    assert entry.metadata["target_price"] == _MID
    assert entry.metadata["target_allocation"] == 0.25
    assert entry.metadata["max_order_value"] == 500.0
    assert entry.metadata["box_high"] == _BOX_HIGH
    assert entry.metadata["box_q25"] == _Q25
    assert entry.metadata["prev_median_volume"] == 10.0
    assert entry.metadata["reason"] == "lower_quartile_rebound"

    # A second qualifying setup in the same session must not signal again.
    strategy.calculate_signals(_long_setup_bar(bar_index=2))
    assert events.drain() == []

    strategy.calculate_signals(_bar(_iso(1, 3), 100.0, 100.8, 99.8, 100.5, 12.0))
    exits = events.drain()
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "take_profit"

    # Position is flat: further bars past the target emit nothing.
    strategy.calculate_signals(_bar(_iso(1, 4), 100.5, 101.5, 100.4, 101.0, 12.0))
    assert events.drain() == []


def test_long_stop_loss_exit_at_box_low() -> None:
    strategy, events = _primed()
    strategy.calculate_signals(_long_setup_bar())
    assert events.drain()[0].signal_type == "LONG"

    strategy.calculate_signals(_bar(_iso(1, 2), 94.0, 94.2, 89.0, 89.5, 30.0))
    exits = events.drain()
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "stop_loss"
    assert exits[0].metadata["side"] == "LONG"


def test_volume_filter_blocks_entry_at_or_below_median() -> None:
    strategy, events = _primed()
    strategy.calculate_signals(_long_setup_bar(volume=10.0))
    assert events.drain() == []

    # Missing volume is also treated as unconfirmed while require_volume is on.
    strategy.calculate_signals(_long_setup_bar(bar_index=2, volume=None))
    assert events.drain() == []

    # Disabling the filter lets the same setup through.
    relaxed, relaxed_events = _primed(require_volume=False)
    relaxed.calculate_signals(_long_setup_bar(volume=10.0))
    assert [signal.signal_type for signal in relaxed_events.drain()] == ["LONG"]


def test_open_position_is_flattened_at_the_utc_day_end() -> None:
    strategy, events = _primed()
    strategy.calculate_signals(_long_setup_bar())
    assert events.drain()[0].signal_type == "LONG"

    # Day 3 opens with the position still inside the box: flatten first.
    strategy.calculate_signals(_bar(_iso(2, 0), 97.0, 97.5, 96.5, 97.0, 10.0))
    exits = events.drain()
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "session_flat"


def test_short_mirror_from_the_upper_quartile() -> None:
    strategy, events = _primed()
    strategy.calculate_signals(_short_setup_bar())
    entries = events.drain()
    assert len(entries) == 1
    assert entries[0].signal_type == "SHORT"
    assert entries[0].stop_loss is None
    assert entries[0].take_profit is None
    assert entries[0].metadata["stop_price"] == _BOX_HIGH
    assert entries[0].metadata["target_price"] == _MID
    assert entries[0].metadata["reason"] == "upper_quartile_rejection"
    assert entries[0].metadata["box_q75"] == _Q75

    strategy.calculate_signals(_bar(_iso(1, 2), 102.0, 102.2, 99.0, 99.5, 12.0))
    exits = events.drain()
    assert len(exits) == 1
    assert exits[0].metadata["reason"] == "take_profit"
    assert exits[0].metadata["side"] == "SHORT"


def test_short_disabled_blocks_the_upper_quartile_setup() -> None:
    strategy, events = _primed(allow_short=False)
    strategy.calculate_signals(_short_setup_bar())
    assert events.drain() == []


def test_zero_target_allocation_refuses_to_emit_an_unsized_entry() -> None:
    """An allocation of 0 must block the entry, not ship a signal without one.

    ``_target_metadata`` omits ``target_allocation`` when it is not positive,
    so the portfolio would size the order off its own config default - the
    sleeve would place a trade it never sized.
    """
    strategy, events = _primed(target_allocation=0.0)
    strategy.calculate_signals(_long_setup_bar())
    assert events.drain() == []

    short_only, short_events = _primed(target_allocation=0.0)
    short_only.calculate_signals(_short_setup_bar())
    assert short_events.drain() == []

    # Control: the identical setups with a real allocation fire, and are sized.
    sized, sized_events = _primed(target_allocation=0.15)
    sized.calculate_signals(_long_setup_bar())
    signals = sized_events.drain()
    assert [signal.signal_type for signal in signals] == ["LONG"]
    assert signals[0].metadata["target_allocation"] == 0.15


def test_rejection_wick_must_be_at_least_the_body() -> None:
    """``wick >= body`` is load-bearing on both sides of the box.

    Each pair differs in exactly one dimension - the wick length - so the
    blocked bar satisfies every other clause of the setup.
    """
    # LONG: pierces q25=95 and closes back above it, but the lower wick is
    # 0.5 against a 1.5 body, so the candle never rejected the level.
    blocked, blocked_events = _primed()
    blocked.calculate_signals(_bar(_iso(1, 1), 94.5, 96.2, 94.0, 96.0, 20.0))
    assert blocked_events.drain() == []

    allowed, allowed_events = _primed()
    allowed.calculate_signals(_bar(_iso(1, 1), 94.5, 96.2, 92.0, 96.0, 20.0))
    assert [signal.signal_type for signal in allowed_events.drain()] == ["LONG"]

    # SHORT mirror: upper wick 1.0 against a 2.5 body.
    blocked_short, blocked_short_events = _primed()
    blocked_short.calculate_signals(_bar(_iso(1, 1), 104.5, 105.5, 103.0, 102.0, 20.0))
    assert blocked_short_events.drain() == []

    allowed_short, allowed_short_events = _primed()
    allowed_short.calculate_signals(_bar(_iso(1, 1), 104.5, 108.0, 103.0, 102.0, 20.0))
    assert [signal.signal_type for signal in allowed_short_events.drain()] == ["SHORT"]


def test_close_must_come_back_through_the_quartile() -> None:
    """Piercing the quartile is not enough - the bar has to close back inside.

    Each pair differs only in the close, so the wick/body and volume clauses
    are identical either way.
    """
    # LONG: low 91 is deep under q25=95 and the wick beats the body, but the
    # 94.9 close is still under the quartile: the flush has not been rejected.
    blocked, blocked_events = _primed()
    blocked.calculate_signals(_bar(_iso(1, 1), 94.0, 96.2, 91.0, 94.9, 20.0))
    assert blocked_events.drain() == []

    allowed, allowed_events = _primed()
    allowed.calculate_signals(_bar(_iso(1, 1), 94.0, 96.2, 91.0, 96.0, 20.0))
    assert [signal.signal_type for signal in allowed_events.drain()] == ["LONG"]

    # SHORT mirror: high 109 clears q75=105 but the 105.1 close stays above it.
    blocked_short, blocked_short_events = _primed()
    blocked_short.calculate_signals(_bar(_iso(1, 1), 106.0, 109.0, 103.8, 105.1, 20.0))
    assert blocked_short_events.drain() == []

    allowed_short, allowed_short_events = _primed()
    allowed_short.calculate_signals(_bar(_iso(1, 1), 106.0, 109.0, 103.8, 104.0, 20.0))
    assert [signal.signal_type for signal in allowed_short_events.drain()] == ["SHORT"]


def test_state_round_trip_preserves_behaviour() -> None:
    source, _ = _primed()
    restored, restored_events = _build()
    restored.set_state(source.get_state())

    baseline, baseline_events = _primed()
    for strategy, events in ((baseline, baseline_events), (restored, restored_events)):
        strategy.calculate_signals(_long_setup_bar())
        strategy.calculate_signals(_bar(_iso(1, 3), 100.0, 100.8, 99.8, 100.5, 12.0))

    def _fingerprint(signals: list[Any]) -> list[tuple[Any, ...]]:
        return [
            (
                signal.signal_type,
                signal.stop_loss,
                signal.take_profit,
                signal.metadata.get("reason"),
            )
            for signal in signals
        ]

    assert _fingerprint(restored_events.drain()) == _fingerprint(baseline_events.drain())


def test_state_round_trip_carries_an_open_position() -> None:
    source, source_events = _primed()
    source.calculate_signals(_long_setup_bar())
    assert source_events.drain()[0].signal_type == "LONG"

    restored, restored_events = _build()
    restored.set_state(source.get_state())
    restored.calculate_signals(_bar(_iso(1, 3), 100.0, 100.8, 99.8, 100.5, 12.0))
    exits = restored_events.drain()
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "take_profit"


def test_degenerate_bars_are_ignored() -> None:
    strategy, events = _primed()
    strategy.calculate_signals(_bar(_iso(1, 1), 95.5, 94.0, 96.2, 96.0, 20.0))
    assert events.drain() == []
    # Repeating an already-seen timestamp is deduped rather than re-signalled.
    setup = _long_setup_bar()
    strategy.calculate_signals(setup)
    assert len(events.drain()) == 1
    strategy.calculate_signals(setup)
    assert events.drain() == []


def test_market_window_path_produces_the_same_entry() -> None:
    strategy, events = _primed()
    first = (_iso(1, 1), 95.5, 96.2, 95.0, 95.0, 10.0)
    second = (_iso(1, 2), 95.0, 96.2, 94.0, 96.0, 10.0)
    strategy.calculate_signals(
        MarketWindowEvent(
            time=_iso(1, 2),
            window_seconds=900,
            bars_1s={_SYMBOL: (first, second)},
        )
    )
    entries = events.drain()
    assert len(entries) == 1
    assert entries[0].signal_type == "LONG"
    assert entries[0].stop_loss is None
    assert entries[0].take_profit is None
    assert entries[0].metadata["stop_price"] == _BOX_LOW
    assert entries[0].metadata["target_price"] == _MID
    assert entries[0].metadata["bar_volume"] == 20.0
