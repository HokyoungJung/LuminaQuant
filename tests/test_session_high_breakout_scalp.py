"""Tests for the session-high breakout scalp (research_only, ASCII only)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from queue import SimpleQueue

import pytest

from lumina_quant.core.events import MarketEvent
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.strategies.session_high_breakout_scalp import (
    SessionHighBreakoutScalpStrategy,
)

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


def _ts(second: int) -> str:
    return (_EPOCH + timedelta(seconds=second)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bar(
    symbol: str,
    second: int,
    close: float,
    volume: float | None,
    *,
    high: float | None = None,
    low: float | None = None,
) -> MarketEvent:
    return MarketEvent(
        time=_ts(second),
        symbol=symbol,
        open=close,
        high=close if high is None else high,
        low=close if low is None else low,
        close=close,
        volume=volume,
    )


def _drain(events: SimpleQueue) -> list:
    out = []
    while not events.empty():
        out.append(events.get())
    return out


def _build(symbols: list[str] | None = None, **params):
    events: SimpleQueue = SimpleQueue()
    strategy = SessionHighBreakoutScalpStrategy(_Bars(symbols or ["AAA"]), events, **params)
    return strategy, events


def _warmup(
    strategy: SessionHighBreakoutScalpStrategy,
    symbol: str = "AAA",
    *,
    bars: int = 60,
    start: int = 0,
    price: float = 100.0,
    volume: float = 1.0,
) -> int:
    """Feed ``bars`` flat bars; return the second of the next (free) slot."""
    for index in range(bars):
        strategy.calculate_signals(_bar(symbol, start + index, price, volume))
    return start + bars


# --------------------------------------------------------------------- entry


def test_breakout_with_volume_surge_emits_single_long() -> None:
    strategy, events = _build()
    nxt = _warmup(strategy)
    assert _drain(events) == []

    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    signals = _drain(events)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "LONG"
    assert signal.symbol == "AAA"
    assert signal.price == pytest.approx(100.2)
    assert signal.stop_loss == pytest.approx(100.2 * (1.0 - 0.007))
    assert signal.take_profit == pytest.approx(100.2 * (1.0 + 0.015))
    assert signal.metadata["target_allocation"] == pytest.approx(0.10)
    assert signal.metadata["max_order_value"] == pytest.approx(500.0)
    assert signal.metadata["session_high"] == pytest.approx(100.0)
    assert signal.metadata["breakout_level"] == pytest.approx(100.0 * (1.0 + 0.0005))
    assert signal.metadata["surge_ratio"] == pytest.approx(129.0 / 30.0)
    assert signal.metadata["minutes_since_open"] == pytest.approx(1.0)
    assert signal.metadata["entries_this_session"] == 1


def test_take_profit_exit_then_no_reentry_below_new_session_high() -> None:
    strategy, events = _build()
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    assert [signal.signal_type for signal in _drain(events)] == ["LONG"]

    # 100.2 * 1.015 = 101.703 -> a 101.8 close is the first take-profit touch.
    strategy.calculate_signals(_bar("AAA", nxt + 1, 101.8, 1.0, low=100.2))
    exits = _drain(events)
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "take_profit"
    assert exits[0].metadata["side"] == "LONG"
    assert exits[0].price == pytest.approx(101.8)

    # The session high is now 101.8, so a 100.3 close cannot re-arm the break.
    strategy.calculate_signals(_bar("AAA", nxt + 2, 100.3, 100.0, low=100.0))
    assert _drain(events) == []


def test_stop_loss_exit_fires_at_the_configured_level() -> None:
    strategy, events = _build()
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    assert [signal.signal_type for signal in _drain(events)] == ["LONG"]

    # 100.2 * 0.993 = 99.4986 -> a 99.4 close is below the stop.
    strategy.calculate_signals(_bar("AAA", nxt + 1, 99.4, 1.0, high=100.2))
    exits = _drain(events)
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "stop_loss"
    assert exits[0].metadata["entry_price"] == pytest.approx(100.2)
    assert exits[0].metadata["bars_held"] == 1


def test_time_stop_exit_after_max_hold_bars() -> None:
    strategy, events = _build(max_hold_bars=3)
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    assert [signal.signal_type for signal in _drain(events)] == ["LONG"]

    for offset in (1, 2):
        strategy.calculate_signals(_bar("AAA", nxt + offset, 100.25, 1.0, low=100.2))
        assert _drain(events) == []
    strategy.calculate_signals(_bar("AAA", nxt + 3, 100.25, 1.0, low=100.2))
    exits = _drain(events)
    assert len(exits) == 1
    assert exits[0].metadata["reason"] == "time_stop"
    assert exits[0].metadata["bars_held"] == 3


# ------------------------------------------------------------------- filters


def test_no_entry_outside_the_morning_window() -> None:
    start = 300 * 60  # minute 300 of the UTC session, past entry_end_minute 240

    def _run(**params) -> list:
        strategy, events = _build(**params)
        nxt = _warmup(strategy, start=start)
        strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
        return _drain(events)

    assert _run() == []
    # Control: the window is the only thing holding this setup back.
    opened = _run(entry_end_minute=1439)
    assert [signal.signal_type for signal in opened] == ["LONG"]
    assert opened[0].metadata["minutes_since_open"] == pytest.approx(301.0)


def test_no_entry_before_min_session_bars() -> None:
    def _run(**params) -> list:
        strategy, events = _build(**params)
        nxt = _warmup(strategy, bars=30)
        strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
        return _drain(events)

    assert _run() == []
    assert [signal.signal_type for signal in _run(min_session_bars=30)] == ["LONG"]


def test_no_entry_without_a_volume_surge() -> None:
    def _run(**params) -> list:
        strategy, events = _build(**params)
        nxt = _warmup(strategy)
        # Same break, but the bar volume is flat -> no order-flow proxy.
        strategy.calculate_signals(_bar("AAA", nxt, 100.2, 1.0, low=100.0))
        return _drain(events)

    assert _run() == []
    assert [signal.signal_type for signal in _run(surge_multiple=1.0)] == ["LONG"]


def test_missing_volume_is_ignored_without_raising() -> None:
    strategy, events = _build()
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, None, low=100.0))
    assert _drain(events) == []


def test_zero_target_allocation_refuses_to_emit_an_unsized_entry() -> None:
    """An allocation of 0 must block the entry, not ship a signal without one.

    ``_target_metadata`` omits ``target_allocation`` when it is not positive,
    and the portfolio then sizes the order off its own config default - i.e.
    the sleeve would place a trade it never sized.
    """
    strategy, events = _build(target_allocation=0.0)
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    assert _drain(events) == []

    short_only, short_events = _build(target_allocation=0.0, allow_short=True)
    short_next = _warmup(short_only)
    short_only.calculate_signals(_bar("AAA", short_next, 99.8, 100.0, high=100.0))
    assert _drain(short_events) == []

    # Control: the identical setup with a real allocation fires, and is sized.
    sized, sized_events = _build(target_allocation=0.05)
    sized_next = _warmup(sized)
    sized.calculate_signals(_bar("AAA", sized_next, 100.2, 100.0, low=100.0))
    signals = _drain(sized_events)
    assert [signal.signal_type for signal in signals] == ["LONG"]
    assert signals[0].metadata["target_allocation"] == pytest.approx(0.05)


def test_require_new_high_after_exit_blocks_the_snapback_reentry() -> None:
    def _run(require_new_high: bool) -> list:
        strategy, events = _build(require_new_high_after_exit=require_new_high)
        nxt = _warmup(strategy)
        strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
        strategy.calculate_signals(_bar("AAA", nxt + 1, 99.4, 1.0, high=100.2))
        assert [signal.signal_type for signal in _drain(events)] == ["LONG", "EXIT"]
        # 100.3 clears the session high 100.2 plus the buffer, but it is a
        # snap-back through the level we already paid for.
        strategy.calculate_signals(_bar("AAA", nxt + 2, 100.3, 100.0, low=99.4))
        return _drain(events)

    assert _run(True) == []
    resumed = _run(False)
    assert len(resumed) == 1
    assert resumed[0].signal_type == "LONG"
    assert resumed[0].metadata["entries_this_session"] == 2


def test_reentry_barrier_is_the_session_high_not_the_entry_price() -> None:
    """After a winner the barrier sits at the session high the trade ran to.

    Regression: the barrier used to be pinned at ``entry_price``.  The entry
    bar's own high is already at or above that price, so after a take-profit
    the block could never fire and the sleeve re-bought the level it had just
    sold out of.  With the barrier anchored on the session extreme the flag is
    only released once the session prints a genuinely new high.
    """
    strategy, events = _build()
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    # Take profit at 101.8: the session high is now 101.8 while the entry was
    # 100.2, which is exactly the gap the old barrier could not see.
    strategy.calculate_signals(_bar("AAA", nxt + 1, 101.8, 1.0, low=100.2))
    assert [signal.signal_type for signal in _drain(events)] == ["LONG", "EXIT"]
    stored = strategy.get_state()["symbol_state"]["AAA"]
    assert stored["reentry_high_barrier"] == pytest.approx(101.8)
    assert stored["session_high"] == pytest.approx(101.8)

    # A snap-back close under the barrier cannot re-arm the break, and neither
    # can a surge close ABOVE it while the session high itself has not moved -
    # the old entry-price barrier (100.2) let that second bar straight back in.
    strategy.calculate_signals(_bar("AAA", nxt + 2, 101.0, 100.0, low=100.0))
    strategy.calculate_signals(_bar("AAA", nxt + 3, 102.5, 100.0, low=101.0))
    assert _drain(events) == []

    # Once a bar prints a genuinely new session high, the next surge breakout
    # of that high is tradable again.
    strategy.calculate_signals(_bar("AAA", nxt + 4, 101.5, 1.0, high=102.9, low=101.0))
    strategy.calculate_signals(_bar("AAA", nxt + 5, 103.1, 100.0, low=101.5))
    resumed = _drain(events)
    assert [signal.signal_type for signal in resumed] == ["LONG"]
    assert resumed[0].metadata["entries_this_session"] == 2

    # Control: with the flag off the same tape re-enters on the 102.5 break.
    relaxed, relaxed_events = _build(require_new_high_after_exit=False)
    relaxed_next = _warmup(relaxed)
    relaxed.calculate_signals(_bar("AAA", relaxed_next, 100.2, 100.0, low=100.0))
    relaxed.calculate_signals(_bar("AAA", relaxed_next + 1, 101.8, 1.0, low=100.2))
    assert [signal.signal_type for signal in _drain(relaxed_events)] == ["LONG", "EXIT"]
    relaxed.calculate_signals(_bar("AAA", relaxed_next + 2, 101.0, 100.0, low=100.0))
    relaxed.calculate_signals(_bar("AAA", relaxed_next + 3, 102.5, 100.0, low=101.0))
    assert [signal.signal_type for signal in _drain(relaxed_events)] == ["LONG"]


def test_max_entries_per_session_is_respected() -> None:
    def _run(cap: int) -> list:
        strategy, events = _build(max_entries_per_session=cap, require_new_high_after_exit=False)
        nxt = _warmup(strategy)
        strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
        strategy.calculate_signals(_bar("AAA", nxt + 1, 101.8, 1.0, low=100.2))
        assert [signal.signal_type for signal in _drain(events)] == ["LONG", "EXIT"]
        strategy.calculate_signals(_bar("AAA", nxt + 2, 102.5, 100.0, low=101.8))
        return _drain(events)

    assert _run(1) == []
    second = _run(2)
    assert len(second) == 1
    assert second[0].signal_type == "LONG"
    assert second[0].metadata["entries_this_session"] == 2


def test_short_mirror_requires_allow_short() -> None:
    strategy, events = _build(allow_short=True)
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 99.8, 100.0, high=100.0))
    signals = _drain(events)
    assert len(signals) == 1
    assert signals[0].signal_type == "SHORT"
    assert signals[0].stop_loss == pytest.approx(99.8 * (1.0 + 0.007))
    assert signals[0].take_profit == pytest.approx(99.8 * (1.0 - 0.015))
    assert signals[0].metadata["session_low"] == pytest.approx(100.0)

    flat_strategy, flat_events = _build()
    flat_next = _warmup(flat_strategy)
    flat_strategy.calculate_signals(_bar("AAA", flat_next, 99.8, 100.0, high=100.0))
    assert _drain(flat_events) == []


def test_short_take_profit_and_stop_are_not_interchangeable() -> None:
    """Pin BOTH short exits: profit when price FALLS, stop when it RISES.

    Swapping the two comparisons in ``_manage_position`` leaves every other
    test green, so the direction of each leg is asserted explicitly here.
    """

    def _short(path: list[tuple[float, float, float]]) -> list:
        strategy, events = _build(allow_short=True)
        nxt = _warmup(strategy)
        strategy.calculate_signals(_bar("AAA", nxt, 99.8, 100.0, high=100.0))
        assert [signal.signal_type for signal in _drain(events)] == ["SHORT"]
        for offset, (close, high, low) in enumerate(path, start=1):
            strategy.calculate_signals(_bar("AAA", nxt + offset, close, 1.0, high=high, low=low))
        return _drain(events)

    # 99.8 * (1 - 0.015) = 98.303: 98.4 is still inside the band, 98.2 pays.
    profit = _short([(98.4, 99.8, 98.4), (98.2, 98.5, 98.2)])
    assert len(profit) == 1
    assert profit[0].signal_type == "EXIT"
    assert profit[0].metadata["reason"] == "take_profit"
    assert profit[0].metadata["side"] == "SHORT"
    assert profit[0].price == pytest.approx(98.2)
    assert profit[0].metadata["entry_price"] == pytest.approx(99.8)

    # 99.8 * (1 + 0.007) = 100.4986: 100.4 is inside the band, 100.6 stops out.
    stopped = _short([(100.4, 100.4, 99.8), (100.6, 100.6, 100.4)])
    assert len(stopped) == 1
    assert stopped[0].signal_type == "EXIT"
    assert stopped[0].metadata["reason"] == "stop_loss"
    assert stopped[0].metadata["side"] == "SHORT"
    assert stopped[0].price == pytest.approx(100.6)


# ------------------------------------------------------------------ universe


def _feed_universe_session(strategy, start: int) -> None:
    """One session of identical setups; AAA trades 2x the notional of BBB."""
    for index in range(60):
        strategy.calculate_signals(_bar("AAA", start + index, 200.0, 1.0))
        strategy.calculate_signals(_bar("BBB", start + index, 100.0, 1.0))
    strategy.calculate_signals(_bar("AAA", start + 60, 200.4, 100.0, low=200.0))
    strategy.calculate_signals(_bar("BBB", start + 60, 100.2, 100.0, low=100.0))


def test_turnover_universe_filter_keeps_only_the_top_symbol() -> None:
    strategy, events = _build(["AAA", "BBB"], max_symbols_by_turnover=1)

    # Session 1: nobody may trade yet - no previous session, so no turnover,
    # so no eligible symbol.  (Control below proves both would otherwise fire.)
    _feed_universe_session(strategy, 0)
    assert _drain(events) == []

    # Session 2 (next UTC day): identical setups, only the richer AAA is left.
    _feed_universe_session(strategy, 86_400)
    signals = _drain(events)
    assert len(signals) == 1
    assert signals[0].symbol == "AAA"
    assert signals[0].signal_type == "LONG"

    # Control: with the filter off, both symbols take the same breakout.
    unfiltered, unfiltered_events = _build(["AAA", "BBB"], max_symbols_by_turnover=0)
    _feed_universe_session(unfiltered, 0)
    assert sorted(signal.symbol for signal in _drain(unfiltered_events)) == ["AAA", "BBB"]


def test_session_rollover_flattens_an_open_position() -> None:
    strategy, events = _build()
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    assert [signal.signal_type for signal in _drain(events)] == ["LONG"]

    strategy.calculate_signals(_bar("AAA", 86_400, 100.25, 1.0))
    exits = _drain(events)
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "session_flat"


# --------------------------------------------------------------------- state


def test_state_round_trip_preserves_behaviour() -> None:
    warm, _ = _build()
    nxt = _warmup(warm)
    state = json.loads(json.dumps(warm.get_state()))

    restored, restored_events = _build()
    restored.set_state(state)

    original, original_events = _build()
    _warmup(original)
    _drain(original_events)

    breakout = 100.2
    original.calculate_signals(_bar("AAA", nxt, breakout, 100.0, low=100.0))
    restored.calculate_signals(_bar("AAA", nxt, breakout, 100.0, low=100.0))

    expected = _drain(original_events)
    actual = _drain(restored_events)
    assert len(expected) == len(actual) == 1
    for left, right in zip(expected, actual, strict=True):
        assert left.signal_type == right.signal_type
        assert left.price == pytest.approx(right.price)
        assert left.stop_loss == pytest.approx(right.stop_loss)
        assert left.take_profit == pytest.approx(right.take_profit)
        assert left.metadata == right.metadata


def test_state_round_trip_preserves_an_open_position() -> None:
    warm, warm_events = _build()
    nxt = _warmup(warm)
    warm.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    assert [signal.signal_type for signal in _drain(warm_events)] == ["LONG"]

    restored, restored_events = _build()
    restored.set_state(json.loads(json.dumps(warm.get_state())))
    restored.calculate_signals(_bar("AAA", nxt + 1, 101.8, 1.0, low=100.2))

    exits = _drain(restored_events)
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "take_profit"
    assert exits[0].metadata["entry_price"] == pytest.approx(100.2)


def test_duplicate_timestamps_are_ignored() -> None:
    strategy, events = _build()
    nxt = _warmup(strategy)
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    strategy.calculate_signals(_bar("AAA", nxt, 100.2, 100.0, low=100.0))
    assert len(_drain(events)) == 1


def test_registered_in_global_registry() -> None:
    assert (
        GLOBAL_REGISTRY.get("strategy", "SessionHighBreakoutScalpStrategy")
        is SessionHighBreakoutScalpStrategy
    )
