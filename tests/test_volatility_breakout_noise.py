"""Deterministic tests for NoiseFilteredVolatilityBreakoutStrategy."""

from __future__ import annotations

from queue import SimpleQueue

import pytest

from lumina_quant.core.events import MarketBatchEvent, MarketEvent
from lumina_quant.strategies.volatility_breakout_noise import (
    NoiseFilteredVolatilityBreakoutStrategy,
)

# Synthetic session geometry.  Each session has range 8, |close-open| = 4 so the
# noise ratio is exactly 1 - 4/8 = 0.5, and the session high sits only +1 above
# the open so a K=0.5 trigger (open + 4) is never reached inside the history.
_STEP = 10.0
_BASE = 100.0


class _Bars:
    def __init__(self, symbols):
        self.symbol_list = list(symbols)


def _session_open(index: int) -> float:
    return _BASE + _STEP * index


def _quiet_session_bars(symbol: str, day: int, index: int, close_offset: float = -4.0):
    """Three hourly bars aggregating to open=O, high=O+1, low=O-7, close=O+offset."""
    o = _session_open(index)
    stamp = f"2026-01-{day:02d}T{{:02d}}:00:00Z"
    mid = o + close_offset / 2.0
    return [
        MarketEvent(stamp.format(0), symbol, o, o + 1.0, o, o + 0.25, 10.0),
        MarketEvent(stamp.format(1), symbol, o + 0.25, o + 0.25, o - 7.0, mid, 10.0),
        MarketEvent(
            stamp.format(2),
            symbol,
            mid,
            max(mid, o + close_offset),
            min(mid, o + close_offset) - 0.5,
            o + close_offset,
            10.0,
        ),
    ]


def _history(symbol: str, sessions: int, close_offset: float = -4.0):
    events = []
    for index in range(sessions):
        events.extend(_quiet_session_bars(symbol, index + 1, index, close_offset))
    return events


def _breakout_session_bars(symbol: str, day: int, index: int):
    """Bar 1 stays below the trigger, bar 2 crosses it, bar 3 crosses it again."""
    o = _session_open(index)
    stamp = f"2026-01-{day:02d}T{{:02d}}:00:00Z"
    return [
        MarketEvent(stamp.format(0), symbol, o, o + 1.0, o - 1.0, o, 10.0),
        MarketEvent(stamp.format(1), symbol, o, o + 5.0, o, o + 4.5, 10.0),
        MarketEvent(stamp.format(2), symbol, o + 4.5, o + 6.0, o + 4.0, o + 5.0, 10.0),
    ]


# Short-side geometry.  The quiet sessions above have range 8 and a +-4 band, so
# every one of them touches a trigger the moment shorts are enabled.  These
# sessions keep range 8 but noise 0.7 (|close-open| = 2.4), which widens the band
# to +-5.6 while the session itself only spans O+3 .. O-5 -- inside both triggers.
def _wide_band_session_bars(symbol: str, day: int, index: int):
    """Three hourly bars -> open=O, high=O+3, low=O-5, close=O-2.4 (noise 0.7)."""
    o = _session_open(index)
    stamp = f"2026-01-{day:02d}T{{:02d}}:00:00Z"
    return [
        MarketEvent(stamp.format(0), symbol, o, o + 3.0, o, o + 2.0, 10.0),
        MarketEvent(stamp.format(1), symbol, o + 2.0, o + 2.0, o - 5.0, o - 4.0, 10.0),
        MarketEvent(stamp.format(2), symbol, o - 4.0, o - 2.0, o - 4.5, o - 2.4, 10.0),
    ]


def _wide_band_history(symbol: str, sessions: int):
    events = []
    for index in range(sessions):
        events.extend(_wide_band_session_bars(symbol, index + 1, index))
    return events


def _breakdown_session_bars(symbol: str, day: int, index: int):
    """Bar 1 stays inside the band; bar 2 pierces the LOWER trigger only."""
    o = _session_open(index)
    stamp = f"2026-01-{day:02d}T{{:02d}}:00:00Z"
    return [
        MarketEvent(stamp.format(0), symbol, o, o + 1.0, o - 1.0, o, 10.0),
        MarketEvent(stamp.format(1), symbol, o, o + 0.5, o - 7.0, o - 6.5, 10.0),
        MarketEvent(stamp.format(2), symbol, o - 6.5, o - 6.0, o - 8.0, o - 7.0, 10.0),
    ]


def _drain(events):
    out = []
    while not events.empty():
        out.append(events.get())
    return out


def _feed(strategy, events, market_events):
    for event in market_events:
        strategy.calculate_signals(event)
    return _drain(events)


def test_noise_k_entry_sizing_single_entry_and_session_exit():
    events = SimpleQueue()
    strategy = NoiseFilteredVolatilityBreakoutStrategy(_Bars(["BTCUSDT"]), events)

    # 25 completed sessions (days 1..25), all with noise 0.5 and rising closes.
    assert _feed(strategy, events, _history("BTCUSDT", 25)) == []

    signals = _feed(strategy, events, _breakout_session_bars("BTCUSDT", 26, 25))
    assert [signal.signal_type for signal in signals] == ["LONG"]

    entry = signals[0]
    prev_open = _session_open(24)
    prev_high, prev_low, prev_close = prev_open + 1.0, prev_open - 7.0, prev_open - 4.0
    vol_weight = min(1.0, 0.02 / ((prev_high - prev_low) / prev_close))
    expected_allocation = 0.10 * 1.0 * vol_weight

    assert entry.symbol == "BTCUSDT"
    assert entry.metadata["target_allocation"] == pytest.approx(expected_allocation, rel=1e-12)
    assert entry.metadata["max_order_value"] == pytest.approx(500.0)
    assert entry.metadata["k"] == pytest.approx(0.5)
    assert entry.metadata["noise"] == pytest.approx(0.5)
    assert entry.metadata["ma_score"] == pytest.approx(1.0)
    assert entry.metadata["vol_weight"] == pytest.approx(vol_weight)
    assert entry.metadata["session"] == "2026-01-26"
    assert entry.strength == pytest.approx(expected_allocation, rel=1e-12)
    # price = max(trigger level, bar close); trigger = session open + 0.5 * prev range.
    assert entry.price == pytest.approx(max(_session_open(25) + 4.0, _session_open(25) + 4.5))
    assert entry.stop_loss is None
    assert vol_weight < 1.0

    # First bar of the next session flattens the carried position.
    exits = _feed(strategy, events, _breakout_session_bars("BTCUSDT", 27, 26)[:1])
    assert [signal.signal_type for signal in exits] == ["EXIT"]
    assert exits[0].metadata["reason"] == "session_time_cut"


def test_fixed_k_without_timing_or_vol_control_uses_base_allocation():
    events = SimpleQueue()
    strategy = NoiseFilteredVolatilityBreakoutStrategy(
        _Bars(["BTCUSDT"]),
        events,
        k_mode="fixed",
        k=0.5,
        use_ma_score=False,
        use_vol_target=False,
    )
    assert _feed(strategy, events, _history("BTCUSDT", 2)) == []

    signals = _feed(strategy, events, _breakout_session_bars("BTCUSDT", 3, 2))
    assert [signal.signal_type for signal in signals] == ["LONG"]
    assert signals[0].metadata["target_allocation"] == pytest.approx(0.10)
    assert signals[0].metadata["ma_score"] == pytest.approx(1.0)
    assert signals[0].metadata["vol_weight"] == pytest.approx(1.0)


def test_no_entry_when_history_shorter_than_noise_period():
    events = SimpleQueue()
    strategy = NoiseFilteredVolatilityBreakoutStrategy(
        _Bars(["BTCUSDT"]), events, use_ma_score=False
    )
    assert _feed(strategy, events, _history("BTCUSDT", 5)) == []
    # A genuine breakout bar still produces nothing: the noise-derived K is unavailable.
    assert _feed(strategy, events, _breakout_session_bars("BTCUSDT", 6, 5)) == []


def test_state_round_trip_mid_session_blocks_duplicate_entry():
    events = SimpleQueue()
    strategy = NoiseFilteredVolatilityBreakoutStrategy(_Bars(["BTCUSDT"]), events)
    _feed(strategy, events, _history("BTCUSDT", 25))

    breakout = _breakout_session_bars("BTCUSDT", 26, 25)
    assert [signal.signal_type for signal in _feed(strategy, events, breakout[:2])] == ["LONG"]

    restored_events = SimpleQueue()
    restored = NoiseFilteredVolatilityBreakoutStrategy(_Bars(["BTCUSDT"]), restored_events)
    restored.set_state(strategy.get_state())

    # The remaining bar of the same session crosses the trigger again -> still no entry.
    assert _feed(restored, restored_events, breakout[2:]) == []
    exits = _feed(restored, restored_events, _breakout_session_bars("BTCUSDT", 27, 26)[:1])
    assert [signal.signal_type for signal in exits] == ["EXIT"]


def test_restored_state_reproduces_the_next_session_entry_exactly():
    """A fresh instance fed only ``set_state`` must trade the next session identically.

    Regression: the existing round-trip test restores AFTER an entry, so it stays
    green even if ``set_state`` puts nothing back into the OHLC deques -- an empty
    history simply blocks the duplicate entry it was asserting was blocked.  Here
    the restore happens BEFORE the entry, so the deques have to survive.
    """
    events = SimpleQueue()
    strategy = NoiseFilteredVolatilityBreakoutStrategy(_Bars(["BTCUSDT"]), events)
    assert _feed(strategy, events, _history("BTCUSDT", 25)) == []

    snapshot = strategy.get_state()
    restored_events = SimpleQueue()
    restored = NoiseFilteredVolatilityBreakoutStrategy(_Bars(["BTCUSDT"]), restored_events)
    restored.set_state(snapshot)
    # The restored book carries the same completed-session history, not an empty one.
    assert restored.get_state() == snapshot
    assert len(restored._state["BTCUSDT"].closes) == 24

    breakout = _breakout_session_bars("BTCUSDT", 26, 25)
    original = _feed(strategy, events, breakout)
    mirrored = _feed(restored, restored_events, breakout)

    assert [signal.signal_type for signal in original] == ["LONG"]
    assert len(mirrored) == len(original)
    for left, right in zip(original, mirrored):
        assert left.signal_type == right.signal_type
        assert left.symbol == right.symbol
        assert left.metadata == right.metadata
        assert left.price == pytest.approx(right.price)
        assert left.strength == pytest.approx(right.strength)
    assert mirrored[0].metadata["target_allocation"] > 0.0
    assert restored.get_state() == strategy.get_state()


def test_allow_short_opens_on_the_lower_trigger():
    events = SimpleQueue()
    strategy = NoiseFilteredVolatilityBreakoutStrategy(_Bars(["BTCUSDT"]), events, allow_short=True)
    assert _feed(strategy, events, _wide_band_history("BTCUSDT", 25)) == []

    signals = _feed(strategy, events, _breakdown_session_bars("BTCUSDT", 26, 25))
    assert [signal.signal_type for signal in signals] == ["SHORT"]

    session_open = _session_open(25)
    # K resolves to the 0.7 noise ratio and the prior session's range is 8.
    assert signals[0].metadata["noise"] == pytest.approx(0.7)
    assert signals[0].metadata["k"] == pytest.approx(0.7)
    assert signals[0].metadata["lower"] == pytest.approx(session_open - 5.6)
    assert signals[0].metadata["upper"] == pytest.approx(session_open + 5.6)
    # price = min(trigger level, bar close); the bar closed through the trigger.
    assert signals[0].price == pytest.approx(session_open - 6.5)
    assert signals[0].stop_loss is None
    assert signals[0].metadata["target_allocation"] > 0.0
    assert strategy._state["BTCUSDT"].mode == "SHORT"

    # The default book is long-only: the same breakdown must produce nothing.
    long_only_events = SimpleQueue()
    long_only = NoiseFilteredVolatilityBreakoutStrategy(_Bars(["BTCUSDT"]), long_only_events)
    assert _feed(long_only, long_only_events, _wide_band_history("BTCUSDT", 25)) == []
    assert _feed(long_only, long_only_events, _breakdown_session_bars("BTCUSDT", 26, 25)) == []
    assert long_only._state["BTCUSDT"].mode == "OUT"


def test_stop_loss_pct_cuts_the_position_inside_the_session():
    events = SimpleQueue()
    strategy = NoiseFilteredVolatilityBreakoutStrategy(
        _Bars(["BTCUSDT"]), events, stop_loss_pct=0.01
    )
    assert _feed(strategy, events, _history("BTCUSDT", 25)) == []

    session_open = _session_open(25)
    breakout = _breakout_session_bars("BTCUSDT", 26, 25)
    entries = _feed(strategy, events, breakout[:2])
    assert [signal.signal_type for signal in entries] == ["LONG"]
    entry_price = session_open + 4.5
    assert entries[0].price == pytest.approx(entry_price)
    assert entries[0].stop_loss is None
    assert entries[0].take_profit is None
    assert entries[0].metadata["stop_loss"] == pytest.approx(entry_price * (1.0 - 0.01))

    stop_level = entry_price * (1.0 - 0.01)
    # A later bar of the SAME session that holds above the stop changes nothing.
    held = MarketEvent(
        "2026-01-26T02:00:00Z",
        "BTCUSDT",
        entry_price,
        entry_price + 0.5,
        stop_level + 0.5,
        stop_level + 1.0,
        10.0,
    )
    assert _feed(strategy, events, [held]) == []
    assert strategy._state["BTCUSDT"].mode == "LONG"

    # A bar closing under the stop cuts the position without waiting for the roll.
    stopped = MarketEvent(
        "2026-01-26T03:00:00Z",
        "BTCUSDT",
        stop_level + 1.0,
        stop_level + 1.0,
        stop_level - 2.0,
        stop_level - 1.0,
        10.0,
    )
    exits = _feed(strategy, events, [stopped])
    assert [signal.signal_type for signal in exits] == ["EXIT"]
    assert exits[0].metadata["reason"] == "stop_loss"
    assert exits[0].price == pytest.approx(stop_level - 1.0)
    assert strategy._state["BTCUSDT"].mode == "OUT"
    # One entry per session: the cut position must not re-open on a later cross.
    assert _feed(strategy, events, breakout[2:]) == []


def test_max_position_allocation_caps_the_entry_size():
    baseline_events = SimpleQueue()
    baseline = NoiseFilteredVolatilityBreakoutStrategy(_Bars(["BTCUSDT"]), baseline_events)
    _feed(baseline, baseline_events, _history("BTCUSDT", 25))
    uncapped = _feed(baseline, baseline_events, _breakout_session_bars("BTCUSDT", 26, 25))[0]

    capped_events = SimpleQueue()
    capped_strategy = NoiseFilteredVolatilityBreakoutStrategy(
        _Bars(["BTCUSDT"]), capped_events, max_position_allocation=0.03
    )
    _feed(capped_strategy, capped_events, _history("BTCUSDT", 25))
    capped = _feed(capped_strategy, capped_events, _breakout_session_bars("BTCUSDT", 26, 25))[0]

    # The cap really binds: the uncapped sizing asks for more than 0.03.
    assert uncapped.metadata["target_allocation"] > 0.03
    assert capped.metadata["target_allocation"] == pytest.approx(0.03)
    assert capped.strength == pytest.approx(0.03)
    assert capped.metadata["max_symbol_exposure_pct"] == pytest.approx(0.03)
    # Only the SIZE is capped: the trigger, K and vol weight are untouched.
    assert capped.price == pytest.approx(uncapped.price)
    assert capped.metadata["k"] == pytest.approx(uncapped.metadata["k"])
    assert capped.metadata["vol_weight"] == pytest.approx(uncapped.metadata["vol_weight"])


def test_cross_sectional_noise_filter_keeps_only_the_quietest_symbol():
    events = SimpleQueue()
    strategy = NoiseFilteredVolatilityBreakoutStrategy(
        _Bars(["NOISY", "QUIET"]),
        events,
        k_mode="fixed",
        k=0.5,
        noise_period=5,
        use_ma_score=False,
        use_vol_target=False,
        max_symbols_by_noise=1,
    )
    # NOISY: |close-open| = 4 over a range of 8 -> noise 0.5.
    # QUIET: |close-open| = 6.4 over a range of 8 -> noise 0.2 (ranks first).
    noisy = _history("NOISY", 6, close_offset=-4.0)
    quiet = _history("QUIET", 6, close_offset=-6.4)
    for left, right in zip(noisy, quiet):
        strategy.calculate_signals(left)
        strategy.calculate_signals(right)
    assert _drain(events) == []

    breakout = []
    for left, right in zip(
        _breakout_session_bars("NOISY", 7, 6), _breakout_session_bars("QUIET", 7, 6)
    ):
        breakout.extend([left, right])
    signals = _feed(strategy, events, breakout)
    assert [(signal.symbol, signal.signal_type) for signal in signals] == [("QUIET", "LONG")]


def test_noise_batch_ranking_and_signals_ignore_bar_order():
    seed_events = SimpleQueue()
    seed = NoiseFilteredVolatilityBreakoutStrategy(
        _Bars(["NOISY", "QUIET"]),
        seed_events,
        k_mode="fixed",
        k=0.5,
        noise_period=2,
        use_ma_score=False,
        use_vol_target=False,
        max_symbols_by_noise=1,
    )
    noisy = _history("NOISY", 2, close_offset=-4.0)
    quiet = _history("QUIET", 2, close_offset=-6.4)
    for left, right in zip(noisy, quiet, strict=True):
        seed.calculate_signals(left)
        seed.calculate_signals(right)
    assert _drain(seed_events) == []

    def run(reverse: bool):
        events = SimpleQueue()
        strategy = NoiseFilteredVolatilityBreakoutStrategy(
            _Bars(["NOISY", "QUIET"]),
            events,
            k_mode="fixed",
            k=0.5,
            noise_period=2,
            use_ma_score=False,
            use_vol_target=False,
            max_symbols_by_noise=1,
        )
        strategy.set_state(seed.get_state())
        first = [
            _breakout_session_bars("NOISY", 3, 2)[0],
            _breakout_session_bars("QUIET", 3, 2)[0],
        ]
        second = [
            _breakout_session_bars("NOISY", 3, 2)[1],
            _breakout_session_bars("QUIET", 3, 2)[1],
        ]
        if reverse:
            first.reverse()
            second.reverse()
        strategy.calculate_signals_batch(MarketBatchEvent(first[0].time, tuple(first)))
        allowed = strategy.get_state()["allowed_symbols"]
        strategy.calculate_signals_batch(MarketBatchEvent(second[0].time, tuple(second)))
        return allowed, [
            (signal.symbol, signal.signal_type, signal.price, signal.metadata)
            for signal in _drain(events)
        ]

    forward = run(False)
    backward = run(True)
    assert forward == backward
    assert forward[0] == ["QUIET"]
    assert [(symbol, signal_type) for symbol, signal_type, _, _ in forward[1]] == [
        ("QUIET", "LONG")
    ]


def test_strategy_is_registered():
    from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY

    assert (
        GLOBAL_REGISTRY.get("strategy", "NoiseFilteredVolatilityBreakoutStrategy")
        is NoiseFilteredVolatilityBreakoutStrategy
    )
