from __future__ import annotations

from queue import SimpleQueue
from typing import Any

import pytest

from lumina_quant.core.events import MarketEvent, MarketWindowEvent
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.strategies.turtle_unit_pyramiding import TurtleUnitPyramidingStrategy

SYMBOL = "BTC/USDT"

# Synthetic bars use a fixed half-range ``H`` around the close, so the true
# range of a bar is exactly ``2 * H`` whenever the close moves by at most ``H``
# and ``|close_move| + H`` otherwise.  That makes every lagged ATR ("N") below
# an exact, hand-checkable number.
HALF_RANGE = 5.0
FLAT_CLOSE = 100.0
FLAT_BARS = 60
FLAT_N = 2.0 * HALF_RANGE  # 10.0
BREAKOUT_TR = 11.0  # a +6.0 close move with HALF_RANGE 5.0


class _Bars:
    symbol_list = [SYMBOL]


def _bar(time: int, close: float, *, half_range: float = HALF_RANGE) -> MarketEvent:
    return MarketEvent(time, SYMBOL, close, close + half_range, close - half_range, close, 1.0)


def _window(time: int, close: float, *, half_range: float = HALF_RANGE) -> MarketWindowEvent:
    row = (time, close, close + half_range, close - half_range, close, 1.0)
    return MarketWindowEvent(time=time, window_seconds=86400, bars_1s={SYMBOL: (row,)})


def _drain(events: SimpleQueue) -> list[Any]:
    out: list[Any] = []
    while not events.empty():
        out.append(events.get_nowait())
    return out


def _lagged_n(breakout_bars: int) -> float:
    """Lagged 20-bar ATR after ``breakout_bars`` bars with TR 11 entered it."""
    flat = 20 - breakout_bars
    return (FLAT_N * flat + BREAKOUT_TR * breakout_bars) / 20.0


def _warmed(**params: Any) -> tuple[TurtleUnitPyramidingStrategy, SimpleQueue]:
    """Return a strategy fed ``FLAT_BARS`` identical bars (no signal yet)."""
    events: SimpleQueue = SimpleQueue()
    strategy = TurtleUnitPyramidingStrategy(_Bars(), events, **params)
    for index in range(FLAT_BARS):
        strategy.calculate_signals(_bar(index, FLAT_CLOSE))
    assert events.empty()
    return strategy, events


def _pyramided() -> tuple[TurtleUnitPyramidingStrategy, SimpleQueue, list[Any]]:
    """Entry at 106 plus three +6.0 adds, i.e. a full four-unit long stack."""
    strategy, events = _warmed()
    signals: list[Any] = []
    for time, close in ((60, 106.0), (61, 112.0), (62, 118.0), (63, 124.0)):
        strategy.calculate_signals(_bar(time, close))
        signals.extend(_drain(events))
    return strategy, events, signals


def test_no_signal_without_enough_history_or_with_zero_true_range() -> None:
    events: SimpleQueue = SimpleQueue()
    strategy = TurtleUnitPyramidingStrategy(_Bars(), events)
    # 54 bars is one short of the 55-bar entry channel: a breakout cannot fire.
    for index in range(54):
        strategy.calculate_signals(_bar(index, FLAT_CLOSE))
    strategy.calculate_signals(_bar(54, 130.0))
    assert events.empty()

    # Degenerate bars (high == low == close) give N == 0, so no unit is sizeable.
    flat_events: SimpleQueue = SimpleQueue()
    flat = TurtleUnitPyramidingStrategy(_Bars(), flat_events)
    for index in range(FLAT_BARS):
        flat.calculate_signals(_bar(index, FLAT_CLOSE, half_range=0.0))
    flat.calculate_signals(_bar(FLAT_BARS, 130.0, half_range=0.0))
    assert flat_events.empty()

    # Unusable payloads must never raise and never signal.
    bad_events: SimpleQueue = SimpleQueue()
    bad = TurtleUnitPyramidingStrategy(_Bars(), bad_events)
    bad.calculate_signals(MarketEvent(0, SYMBOL, None, None, None, None, None))
    bad.calculate_signals(MarketEvent(1, SYMBOL, 100.0, None, None, 100.0, 1.0))
    bad.calculate_signals(MarketWindowEvent(time=2, window_seconds=86400, bars_1s={}))
    assert bad_events.empty()


def test_unit_entry_sizes_from_atr_risk_and_anchors_a_two_n_stop() -> None:
    strategy, events = _warmed()
    strategy.calculate_signals(_bar(FLAT_BARS, 106.0))
    signals = _drain(events)

    assert len(signals) == 1
    entry = signals[0]
    assert entry.signal_type == "LONG"
    assert entry.strategy_id == "turtle_unit_pyramiding"
    # N is lagged: the breakout bar's own (larger) true range is excluded.
    assert entry.metadata["n_atr"] == pytest.approx(FLAT_N)
    assert entry.metadata["unit"] == 1
    assert entry.metadata["reason"] == "unit_entry"
    assert entry.metadata["target_allocation"] == pytest.approx(min(0.25, 0.01 * 106.0 / FLAT_N))
    assert entry.metadata["max_order_value"] == pytest.approx(500.0)
    assert entry.metadata["stop"] == pytest.approx(106.0 - 2.0 * FLAT_N)
    assert entry.stop_loss == pytest.approx(106.0 - 2.0 * FLAT_N)

    # A repeat of the same bar timestamp is deduped.
    strategy.calculate_signals(_bar(FLAT_BARS, 106.0))
    assert events.empty()


def test_unit_allocation_is_capped_by_max_unit_allocation() -> None:
    # Half-range 1.0 -> N == 2.0, so 1% risk would ask for 0.515 of equity.
    events: SimpleQueue = SimpleQueue()
    strategy = TurtleUnitPyramidingStrategy(_Bars(), events)
    for index in range(FLAT_BARS):
        strategy.calculate_signals(_bar(index, FLAT_CLOSE, half_range=1.0))
    strategy.calculate_signals(_bar(FLAT_BARS, 103.0, half_range=1.0))
    signals = _drain(events)

    assert len(signals) == 1
    assert signals[0].metadata["n_atr"] == pytest.approx(2.0)
    assert 0.01 * 103.0 / 2.0 > 0.25
    assert signals[0].metadata["target_allocation"] == pytest.approx(0.25)
    assert signals[0].stop_loss == pytest.approx(103.0 - 2.0 * 2.0)


def test_pyramid_adds_one_unit_per_half_n_up_to_max_units() -> None:
    strategy, events, signals = _pyramided()

    assert [signal.signal_type for signal in signals] == ["LONG"] * 4
    assert [signal.metadata["unit"] for signal in signals] == [1, 2, 3, 4]
    assert [signal.metadata["reason"] for signal in signals] == [
        "unit_entry",
        "pyramid_add",
        "pyramid_add",
        "pyramid_add",
    ]

    closes = (106.0, 112.0, 118.0, 124.0)
    expected_n = [_lagged_n(index) for index in range(4)]
    assert [signal.metadata["n_atr"] for signal in signals] == pytest.approx(expected_n)
    for signal, close, n_atr in zip(signals, closes, expected_n, strict=True):
        assert signal.metadata["target_allocation"] == pytest.approx(
            min(0.25, 0.01 * close / n_atr)
        )
        assert signal.metadata["stop"] == pytest.approx(close - 2.0 * n_atr)
        assert signal.stop_loss == pytest.approx(close - 2.0 * n_atr)

    # The stop ratchets up with every fill, always anchored on the LAST fill.
    stops = [signal.metadata["stop"] for signal in signals]
    assert stops == sorted(stops)
    assert stops[-1] == pytest.approx(103.7)

    # A fifth qualifying advance is refused: the stack is full at max_units.
    strategy.calculate_signals(_bar(64, 130.0))
    assert events.empty()


def test_advances_smaller_than_half_n_do_not_add_a_unit() -> None:
    """The add gate is ``+add_step_atr * N``, not "the market ticked up".

    Regression: every existing pyramid fixture advances a full +6.0 (> 0.5N), so
    a gate that added on ANY favourable tick passed them all.  Here the market
    grinds up 1.0 a bar -- three consecutive up-closes that must buy nothing --
    before one bar finally pays the half-N toll.
    """
    strategy, events = _warmed()
    strategy.calculate_signals(_bar(60, 106.0))
    entry = _drain(events)
    assert [signal.metadata["reason"] for signal in entry] == ["unit_entry"]
    state = strategy._state[SYMBOL]
    assert state.last_fill_price == pytest.approx(106.0)

    # Lagged N is 10.05 once the breakout bar's TR of 11 is in the window, so
    # the add threshold sits at 106.0 + 0.5 * 10.05 == 111.025.
    for time, close in ((61, 107.0), (62, 108.0), (63, 109.0)):
        strategy.calculate_signals(_bar(time, close))
        assert events.empty(), f"bar {time} rose but paid less than half an N"
        assert state.units == 1
        assert state.last_fill_price == pytest.approx(106.0), "no add means no new anchor"

    strategy.calculate_signals(_bar(64, 111.1))
    added = _drain(events)
    assert len(added) == 1, "the first bar past +0.5N adds exactly one unit"
    assert added[0].signal_type == "LONG"
    assert added[0].metadata["reason"] == "pyramid_add"
    assert added[0].metadata["unit"] == 2
    assert added[0].metadata["n_atr"] == pytest.approx(10.05)
    assert 111.1 >= 106.0 + 0.5 * 10.05
    assert state.last_fill_price == pytest.approx(111.1)


def test_unit_stop_exit_closes_the_whole_stack_once() -> None:
    strategy, events, _ = _pyramided()
    strategy.calculate_signals(_bar(64, 100.0))
    signals = _drain(events)

    assert len(signals) == 1
    assert signals[0].signal_type == "EXIT"
    assert signals[0].metadata["reason"] == "unit_stop"
    assert signals[0].metadata["unit"] == 4
    assert signals[0].metadata["stop"] == pytest.approx(103.7)
    assert strategy.get_state()["symbol_state"][SYMBOL]["units"] == 0
    assert strategy.get_state()["symbol_state"][SYMBOL]["mode"] == "OUT"


def test_exit_channel_exit_fires_and_precedes_a_direction_flip() -> None:
    long_only, long_events = _warmed(allow_short=False)
    long_only.calculate_signals(_bar(60, 106.0))
    assert _drain(long_events)[0].signal_type == "LONG"
    # 90.0 is above the 86.0 stop but below the 20-bar exit channel low of 95.0.
    long_only.calculate_signals(_bar(61, 90.0))
    exits = _drain(long_events)
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "exit_channel"

    # With shorts enabled the same bar is also a downside breakout, so the EXIT
    # must be emitted before the opposite-side entry.
    both, both_events = _warmed()
    both.calculate_signals(_bar(60, 106.0))
    _drain(both_events)
    both.calculate_signals(_bar(61, 90.0))
    flip = _drain(both_events)
    assert [signal.signal_type for signal in flip] == ["EXIT", "SHORT"]
    assert flip[0].metadata["reason"] == "exit_channel"
    assert flip[1].metadata["unit"] == 1


def test_short_entry_and_short_pyramid_mirror_the_long_side() -> None:
    strategy, events = _warmed()
    signals: list[Any] = []
    for time, close in ((60, 94.0), (61, 88.0), (62, 82.0)):
        strategy.calculate_signals(_bar(time, close))
        signals.extend(_drain(events))

    assert [signal.signal_type for signal in signals] == ["SHORT"] * 3
    assert [signal.metadata["unit"] for signal in signals] == [1, 2, 3]
    closes = (94.0, 88.0, 82.0)
    expected_n = [_lagged_n(index) for index in range(3)]
    for signal, close, n_atr in zip(signals, closes, expected_n, strict=True):
        assert signal.metadata["n_atr"] == pytest.approx(n_atr)
        assert signal.metadata["target_allocation"] == pytest.approx(
            min(0.25, 0.01 * close / n_atr)
        )
        assert signal.stop_loss == pytest.approx(close + 2.0 * n_atr)

    # allow_short=False refuses the same downside breakout outright.
    blocked, blocked_events = _warmed(allow_short=False)
    blocked.calculate_signals(_bar(60, 94.0))
    assert blocked_events.empty()


def test_short_exits_through_the_opposite_channel_before_its_stop() -> None:
    """A short leaves on a close above the lagged exit-channel HIGH.

    Regression: the only exit-channel coverage was long-side, so deleting the
    short branch left every test green while a short could only ever be closed
    by its 2N stop -- two N of extra giveback on a failed breakout.
    """
    strategy, events = _warmed()
    strategy.calculate_signals(_bar(60, 94.0))
    entry = _drain(events)
    assert [signal.signal_type for signal in entry] == ["SHORT"]
    # The 2N stop sits at 114.0, well above the bar that follows.
    assert entry[0].stop_loss == pytest.approx(94.0 + 2.0 * FLAT_N)

    # The lagged 20-bar high is 105.0 (nineteen flat bars plus the 99.0 of the
    # entry bar), so 106.0 clears the exit channel while the stop stays untouched.
    strategy.calculate_signals(_bar(61, 106.0))
    signals = _drain(events)
    assert signals, "the short must leave through its exit channel"
    # 106.0 also clears the 55-bar entry high of 105.0, so the covered short is
    # followed by the opposite-side entry -- the EXIT still has to come first.
    assert [signal.signal_type for signal in signals] == ["EXIT", "LONG"]
    assert signals[0].metadata["reason"] == "exit_channel"
    assert signals[0].metadata["unit"] == 1
    assert 94.0 + 2.0 * FLAT_N > 106.0, "the 2N stop was NOT what closed this trade"
    assert strategy._state[SYMBOL].mode == "LONG"

    # Same path with the exit channel out of reach: only the 2N stop can close
    # the short, and 106.0 is not enough to trigger it.
    wide, wide_events = _warmed(exit_lookback=20, allow_short=True, entry_lookback=55)
    wide.calculate_signals(_bar(60, 94.0))
    _drain(wide_events)
    wide.calculate_signals(_bar(61, 104.0))  # below the 105.0 channel high
    assert [signal.signal_type for signal in _drain(wide_events)] == []
    assert wide._state[SYMBOL].mode == "SHORT"


def test_state_round_trip_mid_pyramid_preserves_behaviour() -> None:
    strategy, events = _warmed()
    for time, close in ((60, 106.0), (61, 112.0)):
        strategy.calculate_signals(_bar(time, close))
    _drain(events)
    snapshot = strategy.get_state()
    assert snapshot["symbol_state"][SYMBOL]["units"] == 2
    assert snapshot["symbol_state"][SYMBOL]["mode"] == "LONG"

    restored_events: SimpleQueue = SimpleQueue()
    restored = TurtleUnitPyramidingStrategy(_Bars(), restored_events)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot

    strategy.calculate_signals(_bar(62, 118.0))
    restored.calculate_signals(_window(62, 118.0))
    original_add = _drain(events)
    restored_add = _drain(restored_events)
    assert len(original_add) == len(restored_add) == 1
    assert restored_add[0].signal_type == "LONG"
    assert restored_add[0].metadata == original_add[0].metadata
    assert restored_add[0].stop_loss == pytest.approx(original_add[0].stop_loss)
    assert restored.get_state() == strategy.get_state()


def test_prior_loser_skip_forfeits_exactly_one_breakout() -> None:
    path = ((60, 106.0), (61, 108.0), (62, 114.0), (63, 120.0))

    filtered, filtered_events = _warmed(require_prior_loser_skip=True, max_hold_bars=1)
    seen: list[tuple[int, str, str]] = []
    for time, close in path:
        filtered.calculate_signals(_bar(time, close))
        seen.extend(
            (time, signal.signal_type, str(signal.metadata.get("reason")))
            for signal in _drain(filtered_events)
        )
    # Bar 61 exits on max_hold ABOVE the entry (a winner), so the bar-62
    # breakout is skipped and only bar 63 re-enters.
    assert seen == [
        (60, "LONG", "unit_entry"),
        (61, "EXIT", "max_hold"),
        (63, "LONG", "unit_entry"),
    ]

    # Control: with the filter off the bar-62 breakout is taken.
    plain, plain_events = _warmed(require_prior_loser_skip=False, max_hold_bars=1)
    taken: list[tuple[int, str]] = []
    for time, close in path:
        plain.calculate_signals(_bar(time, close))
        taken.extend((time, signal.signal_type) for signal in _drain(plain_events))
    assert (62, "LONG") in taken


def test_strategy_is_registered_under_its_class_name() -> None:
    assert GLOBAL_REGISTRY.get("strategy", "TurtleUnitPyramidingStrategy") is (
        TurtleUnitPyramidingStrategy
    )
    schema = TurtleUnitPyramidingStrategy.get_param_schema()
    assert schema["entry_lookback"].default == 55
    assert schema["exit_lookback"].default == 20
    assert schema["atr_window"].default == 20
    assert schema["max_units"].default == 4
    assert schema["add_step_atr"].default == pytest.approx(0.5)
    assert schema["stop_atr_multiple"].default == pytest.approx(2.0)
    assert schema["unit_risk_pct"].default == pytest.approx(0.01)
    assert schema["max_unit_allocation"].default == pytest.approx(0.25)
    assert schema["max_unit_allocation"].tunable is False
    assert schema["max_order_value"].tunable is False
    # The public-rule parameters exist but default to the legacy behaviour.
    assert schema["channel_source"].default == "hl"
    assert schema["channel_source"].choices == ("hl", "close")
    assert schema["stop_loss_pct"].default == pytest.approx(0.0)
    assert schema["stop_loss_pct"].tunable is False
    assert schema["trend_ma_window"].default == 0
    assert schema["use_n_stop"].default is True
    assert TurtleUnitPyramidingStrategy.decision_cadence_seconds == 86400


# --------------------------------------------------------------------- preset
# The public "mul-tan-chan-bap" rule set: 20-day close new-high entry, 10-day
# close new-low exit, -3.5% fixed stop and a 120-day MA gate.  Every parameter
# below is off by default, so the block above still describes the sleeve.

PCT_STOP = 0.035
CLOSE_RULE = {
    "channel_source": "close",
    "entry_lookback": 20,
    "exit_lookback": 10,
    "allow_short": False,
}
MA_RULE = {"entry_lookback": 20, "allow_short": False, "trend_ma_window": 120}
PUBLIC_RULE = {
    "channel_source": "close",
    "entry_lookback": 20,
    "exit_lookback": 10,
    "stop_loss_pct": PCT_STOP,
    "trend_ma_window": 120,
    "max_units": 1,
    "use_n_stop": False,
    "allow_short": False,
}


def _regime_series(strategy: TurtleUnitPyramidingStrategy, plateau_close: float) -> None:
    """Feed 100 bars at ``plateau_close`` then 25 bars at ``FLAT_CLOSE``."""
    for index in range(100):
        strategy.calculate_signals(_bar(index, plateau_close))
    for index in range(100, 125):
        strategy.calculate_signals(_bar(index, FLAT_CLOSE))


def _reference_path(**params: Any) -> list[tuple[Any, ...]]:
    """Flatten the entry + three adds + 2N stop-out scenario for comparison."""
    strategy, events = _warmed(**params)
    out: list[tuple[Any, ...]] = []
    for time, close in ((60, 106.0), (61, 112.0), (62, 118.0), (63, 124.0), (64, 100.0)):
        strategy.calculate_signals(_bar(time, close))
        out.extend(
            (signal.signal_type, signal.price, signal.strength, signal.stop_loss, signal.metadata)
            for signal in _drain(events)
        )
    return out


def test_close_channel_ignores_wicks_and_exits_on_a_close_new_low() -> None:
    strategy, events = _warmed(**CLOSE_RULE)

    # Bar 60's HIGH (110) clears the 20-bar close-high of 100, but its CLOSE
    # does not, so a close-channel breakout must not fire on the wick.
    strategy.calculate_signals(_bar(60, FLAT_CLOSE, half_range=10.0))
    assert events.empty()

    # Bar 61 closes at 101, above the same 100 close-high, and does enter.
    strategy.calculate_signals(_bar(61, 101.0))
    entry = _drain(events)
    assert len(entry) == 1
    assert entry[0].signal_type == "LONG"
    assert entry[0].metadata["reason"] == "unit_entry"
    # Lagged N absorbs bar 60's 20-wide true range: (19 * 10 + 20) / 20.
    assert entry[0].metadata["n_atr"] == pytest.approx(10.5)
    assert entry[0].stop_loss == pytest.approx(101.0 - 2.0 * 10.5)

    # The 10-bar close-low is 100, so a 99 close leaves through the channel
    # while the 2N stop at 80.0 is still far away.
    strategy.calculate_signals(_bar(62, 99.0))
    exits = _drain(events)
    assert len(exits) == 1
    assert exits[0].signal_type == "EXIT"
    assert exits[0].metadata["reason"] == "exit_channel"

    # The high/low channel sees none of it: its 20-bar high is 110 (no entry)
    # and its 10-bar low is 90 (no exit).
    hl, hl_events = _warmed(entry_lookback=20, exit_lookback=10, allow_short=False)
    for time, close, half_range in (
        (60, FLAT_CLOSE, 10.0),
        (61, 101.0, HALF_RANGE),
        (62, 99.0, HALF_RANGE),
    ):
        hl.calculate_signals(_bar(time, close, half_range=half_range))
    assert hl_events.empty()


def test_pct_stop_exits_at_the_fixed_level_measured_from_the_entry() -> None:
    strategy, events = _warmed(use_n_stop=False, stop_loss_pct=PCT_STOP, max_units=1)
    strategy.calculate_signals(_bar(FLAT_BARS, 106.0))
    entry = _drain(events)
    level = 106.0 * (1.0 - PCT_STOP)

    assert len(entry) == 1
    assert entry[0].signal_type == "LONG"
    assert entry[0].stop_loss == pytest.approx(level)
    assert entry[0].metadata["stop"] == pytest.approx(level)
    assert entry[0].metadata["pct_stop"] == pytest.approx(level)
    # use_n_stop=False really removes the ATR stop, which would have been 86.0.
    assert strategy.get_state()["symbol_state"][SYMBOL]["stop_price"] is None

    # One tick above the level is not an exit; the level itself is.
    strategy.calculate_signals(_bar(61, level + 0.01))
    assert events.empty()
    strategy.calculate_signals(_bar(62, level))
    stopped = _drain(events)
    assert len(stopped) == 1
    assert stopped[0].signal_type == "EXIT"
    assert stopped[0].metadata["reason"] == "pct_stop"
    assert stopped[0].metadata["pct_stop"] == pytest.approx(level)
    assert stopped[0].metadata["entry_price"] == pytest.approx(106.0)


def test_both_stops_are_evaluated_and_the_closer_one_names_the_exit() -> None:
    # The 2N stop sits at 86.0 and the 3.5% stop at 102.29: the latter is
    # tighter, so it is what the entry advertises and what fires.
    tight, tight_events = _warmed(stop_loss_pct=PCT_STOP, max_units=1, allow_short=False)
    tight.calculate_signals(_bar(FLAT_BARS, 106.0))
    entry = _drain(tight_events)[0]
    level = 106.0 * (1.0 - PCT_STOP)
    assert entry.metadata["stop"] == pytest.approx(level)
    assert entry.stop_loss == pytest.approx(level)
    tight.calculate_signals(_bar(61, 102.0))
    stopped = _drain(tight_events)
    assert [signal.signal_type for signal in stopped] == ["EXIT"]
    assert stopped[0].metadata["reason"] == "pct_stop"

    # A 30% stop sits at 74.2, far below the 2N stop, so the N-stop wins both
    # the entry advertisement and the exit.
    loose, loose_events = _warmed(stop_loss_pct=0.30, max_units=1, allow_short=False)
    loose.calculate_signals(_bar(FLAT_BARS, 106.0))
    loose_entry = _drain(loose_events)[0]
    assert loose_entry.metadata["stop"] == pytest.approx(106.0 - 2.0 * FLAT_N)
    assert loose_entry.stop_loss == pytest.approx(106.0 - 2.0 * FLAT_N)
    assert loose_entry.metadata["pct_stop"] == pytest.approx(106.0 * 0.70)
    loose.calculate_signals(_bar(61, 85.0))
    loose_stopped = _drain(loose_events)
    assert [signal.signal_type for signal in loose_stopped] == ["EXIT"]
    assert loose_stopped[0].metadata["reason"] == "unit_stop"


def test_trend_ma_gate_blocks_breakouts_below_the_lagged_sma() -> None:
    # 95 stored bars at 200 and 25 at 100 give a lagged SMA120 of 179.1666...,
    # so the 110 breakout close is on the wrong side of the gate.
    gated_events: SimpleQueue = SimpleQueue()
    gated = TurtleUnitPyramidingStrategy(_Bars(), gated_events, **MA_RULE)
    _regime_series(gated, 200.0)
    assert gated_events.empty()
    gated.calculate_signals(_bar(125, 110.0))
    assert gated_events.empty()

    # The same path without the gate: the 20-bar channel high of 105 is broken.
    ungated_events: SimpleQueue = SimpleQueue()
    ungated = TurtleUnitPyramidingStrategy(
        _Bars(), ungated_events, entry_lookback=20, allow_short=False
    )
    _regime_series(ungated, 200.0)
    assert ungated_events.empty()
    ungated.calculate_signals(_bar(125, 110.0))
    assert [signal.signal_type for signal in _drain(ungated_events)] == ["LONG"]

    # Flat history puts SMA120 at 100.0, so the same close is above the gate.
    allowed_events: SimpleQueue = SimpleQueue()
    allowed = TurtleUnitPyramidingStrategy(_Bars(), allowed_events, **MA_RULE)
    _regime_series(allowed, FLAT_CLOSE)
    assert allowed_events.empty()
    allowed.calculate_signals(_bar(125, 110.0))
    assert [signal.signal_type for signal in _drain(allowed_events)] == ["LONG"]

    # Too little history for a 120-bar SMA: the breakout is refused outright.
    short_history, short_events = _warmed(**MA_RULE)
    short_history.calculate_signals(_bar(FLAT_BARS, 110.0))
    assert short_events.empty()


def test_public_rule_preset_round_trips_and_stops_at_the_fixed_percentage() -> None:
    events: SimpleQueue = SimpleQueue()
    strategy = TurtleUnitPyramidingStrategy(_Bars(), events, **PUBLIC_RULE)
    for index in range(125):
        strategy.calculate_signals(_bar(index, FLAT_CLOSE))
    assert events.empty()

    strategy.calculate_signals(_bar(125, 110.0))
    entry = _drain(events)
    level = 110.0 * (1.0 - PCT_STOP)
    assert len(entry) == 1
    assert entry[0].signal_type == "LONG"
    assert entry[0].metadata["unit"] == 1
    assert entry[0].metadata["target_allocation"] == pytest.approx(min(0.25, 0.01 * 110.0 / FLAT_N))
    assert entry[0].metadata["max_order_value"] == pytest.approx(500.0)
    assert entry[0].stop_loss == pytest.approx(level)

    snapshot = strategy.get_state()
    restored_events: SimpleQueue = SimpleQueue()
    restored = TurtleUnitPyramidingStrategy(_Bars(), restored_events, **PUBLIC_RULE)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot

    strategy.calculate_signals(_bar(126, 106.0))
    restored.calculate_signals(_window(126, 106.0))
    original = _drain(events)
    mirrored = _drain(restored_events)
    assert len(original) == len(mirrored) == 1
    assert original[0].signal_type == "EXIT"
    assert original[0].metadata["reason"] == "pct_stop"
    assert mirrored[0].metadata == original[0].metadata
    assert restored.get_state() == strategy.get_state()


def test_new_parameters_leave_the_default_signal_stream_untouched() -> None:
    default_path = _reference_path()
    explicit_path = _reference_path(
        channel_source="hl", stop_loss_pct=0.0, trend_ma_window=0, use_n_stop=True
    )
    assert default_path == explicit_path
    assert [row[0] for row in default_path] == ["LONG", "LONG", "LONG", "LONG", "EXIT"]
    # Nothing new leaks into the payload while the preset parameters are off.
    assert all("pct_stop" not in row[-1] for row in default_path)
    assert default_path[0][-1]["stop"] == pytest.approx(106.0 - 2.0 * FLAT_N)
    assert default_path[0][3] == pytest.approx(106.0 - 2.0 * FLAT_N)
    assert default_path[-1][-1]["reason"] == "unit_stop"
