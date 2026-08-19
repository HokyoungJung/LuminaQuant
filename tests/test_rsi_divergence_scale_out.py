"""Deterministic tests for RsiDivergenceScaleOutStrategy (research_only sleeve)."""

from queue import SimpleQueue

import pytest

from lumina_quant.core.events import MarketEvent, MarketWindowEvent
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.indicators.rsi import IncrementalRsi
from lumina_quant.strategies.rsi_divergence_scale_out import RsiDivergenceScaleOutStrategy

SYMBOL = "BTC/USDT"
_HALF_RANGE = 0.5


class _Bars:
    symbol_list = [SYMBOL]


def _rsi_series(closes, period=11):
    """Same Wilder engine the strategy uses, so assertions cannot drift."""
    calc = IncrementalRsi(period)
    return [calc.update(close) for close in closes]


def _pivot_low_indices(closes):
    """Mirror of the strategy's 3-bar pivot rule applied to the synthetic lows."""
    lows = [close - _HALF_RANGE for close in closes]
    return [
        index
        for index in range(1, len(lows) - 1)
        if lows[index] < lows[index - 1] and lows[index] < lows[index + 1]
    ]


def _pivot_high_indices(closes):
    highs = [close + _HALF_RANGE for close in closes]
    return [
        index
        for index in range(1, len(highs) - 1)
        if highs[index] > highs[index - 1] and highs[index] > highs[index + 1]
    ]


def _bullish_divergence_closes(rally_bars=14, rebound=1.0, decline=2.0):
    """Rise, dump to a deeply oversold pivot, rebound, then a lower low at a higher RSI.

    ``rebound``/``decline`` set how deep the second leg cuts, i.e. where the
    newer RSI low lands: the defaults put it near 18.6 (inside the audited
    sub-20 zone), while ``rebound=1.2, decline=1.5`` lands it near 24.2.
    """
    closes = []
    price = 100.0
    for _ in range(14):  # 0..13 warm-up drift (no pivots)
        closes.append(price)
        price += 0.5
    for _ in range(12):  # 14..25 steep dump -> pivot low at 25, RSI ~10.6
        price -= 2.0
        closes.append(price)
    for _ in range(6):  # 26..31 rebound
        price += rebound
        closes.append(price)
    for _ in range(6):  # 32..37 second slide -> lower pivot low at 37
        price -= decline
        closes.append(price)
    for _ in range(rally_bars):  # 38.. rally that lifts RSI through the exit band
        price += 1.3
        closes.append(price)
    return [round(close, 4) for close in closes]


def _hidden_bullish_closes(mid_rally=14):
    """Steady uptrend, shallow pullback, strong leg up, then a sharper shallow pullback.

    The second pullback bottoms at a HIGHER price but a LOWER RSI: hidden bullish.
    ``mid_rally`` lengthens only the leg BETWEEN the two pullbacks, so the same
    divergence shape can be rebuilt at a wider pivot separation.
    """
    closes = []
    price = 100.0
    for _ in range(60):
        closes.append(price)
        price += 0.4
    for _ in range(3):
        price -= 0.8
        closes.append(price)
    for _ in range(mid_rally):
        price += 0.7
        closes.append(price)
    for _ in range(3):
        price -= 2.0
        closes.append(price)
    for _ in range(6):
        price += 1.0
        closes.append(price)
    return [round(close, 4) for close in closes]


def _tight_bullish_divergence_closes(gap=2):
    """The regular bullish shape with the two pivot lows exactly ``gap`` bars apart.

    Same dump / rebound / lower-low geometry as ``_bullish_divergence_closes``,
    only compressed: the rebound is ``gap - 1`` bars and the second leg one bar.
    """
    closes = []
    price = 100.0
    for _ in range(14):
        closes.append(price)
        price += 0.5
    for _ in range(12):
        price -= 2.0
        closes.append(price)
    for _ in range(gap - 1):
        price += 1.0
        closes.append(price)
    price -= 2.5
    closes.append(price)
    for _ in range(14):
        price += 1.3
        closes.append(price)
    return [round(close, 4) for close in closes]


def _feed(strategy, closes, volumes=None, start_index=0, opens=None):
    for offset, close in enumerate(closes):
        index = start_index + offset
        volume = 100.0 if volumes is None else volumes[index]
        # ``opens`` never leaves the [low, high] band below, so overriding it
        # changes only the bar's direction, never its pivots or its RSI.
        open_price = close if opens is None else opens[index]
        strategy.calculate_signals(
            MarketEvent(
                index,
                SYMBOL,
                open_price,
                close + _HALF_RANGE,
                close - _HALF_RANGE,
                close,
                volume,
            )
        )


def _drain(events):
    out = []
    while not events.empty():
        out.append(events.get())
    return out


def _run(closes, volumes=None, opens=None, **params):
    events = SimpleQueue()
    strategy = RsiDivergenceScaleOutStrategy(_Bars(), events, **params)
    _feed(strategy, closes, volumes, opens=opens)
    return _drain(events), strategy


def test_regular_bullish_divergence_entry_and_rsi_exit():
    closes = _bullish_divergence_closes()
    rsi = _rsi_series(closes)
    pivot_lows = _pivot_low_indices(closes)
    assert len(pivot_lows) == 2, pivot_lows
    prev_index, new_index = pivot_lows

    # Precondition, computed with the strategy's own RSI engine: the newer pivot
    # is a lower price low but a HIGHER RSI low, still inside the oversold band.
    assert closes[new_index] < closes[prev_index]
    assert rsi[new_index] > rsi[prev_index]
    assert rsi[new_index] <= 20.0  # the audited sub-20 oversold zone

    signals, _ = _run(closes)
    assert [signal.signal_type for signal in signals] == ["LONG", "EXIT", "EXIT"]

    entry = signals[0]
    # The pivot is only confirmed once its right-hand bar closes.
    assert entry.datetime == new_index + 1
    assert entry.metadata["divergence_type"] == "regular"
    assert entry.metadata["side"] == "LONG"
    assert entry.metadata["target_allocation"] == pytest.approx(0.10)
    assert entry.metadata["max_order_value"] == pytest.approx(500.0)
    assert entry.metadata["rsi_new"] == pytest.approx(rsi[new_index])
    assert entry.metadata["rsi_prev"] == pytest.approx(rsi[prev_index])
    assert entry.metadata["pivot_distance"] == new_index - prev_index
    assert entry.price == pytest.approx(closes[new_index + 1])
    assert entry.stop_loss is None
    assert entry.metadata["stop_price"] == pytest.approx(closes[new_index] - _HALF_RANGE)

    exit_signal = signals[1]
    assert exit_signal.metadata["reason"] == "stage1_rsi"
    assert exit_signal.metadata["side"] == "LONG"
    exit_index = int(exit_signal.datetime)
    assert rsi[exit_index] >= 45.0
    assert rsi[exit_index - 1] < 45.0  # first bar in the exit band, not a late fill


def test_mirrored_regular_bearish_divergence_short_and_exit():
    # Wilder RSI is antisymmetric under price reflection, so mirroring the long
    # fixture around 200 yields the overbought bearish case exactly.
    closes = [round(200.0 - close, 4) for close in _bullish_divergence_closes()]
    rsi = _rsi_series(closes)
    pivot_highs = _pivot_high_indices(closes)
    assert len(pivot_highs) == 2, pivot_highs
    prev_index, new_index = pivot_highs
    assert closes[new_index] > closes[prev_index]
    assert rsi[new_index] < rsi[prev_index]
    assert rsi[new_index] >= 80.0  # the audited above-80 overbought zone

    signals, _ = _run(closes)
    assert [signal.signal_type for signal in signals] == ["SHORT", "EXIT", "EXIT"]
    entry = signals[0]
    assert entry.datetime == new_index + 1
    assert entry.metadata["divergence_type"] == "regular"
    assert entry.metadata["side"] == "SHORT"
    assert entry.metadata["target_allocation"] == pytest.approx(0.10)
    assert entry.stop_loss is None
    assert entry.metadata["stop_price"] == pytest.approx(closes[new_index] + _HALF_RANGE)
    # Shorts mirror the staged levels: partial at RSI 55, remainder under 40.
    assert signals[1].metadata["reason"] == "stage1_rsi"
    assert signals[1].metadata["exit_fraction"] == pytest.approx(0.6)
    assert rsi[int(signals[1].datetime)] <= 55.0
    assert signals[2].metadata["reason"] == "stage2_rsi"
    assert "exit_fraction" not in signals[2].metadata
    assert rsi[int(signals[2].datetime)] <= 40.0


def test_allow_short_false_suppresses_the_bearish_leg():
    closes = [round(200.0 - close, 4) for close in _bullish_divergence_closes()]
    signals, _ = _run(closes, allow_short=False)
    assert signals == []


def test_hidden_bullish_divergence_requires_the_trend_filter():
    closes = _hidden_bullish_closes()
    signals, _ = _run(closes, allow_short=False)
    assert [signal.signal_type for signal in signals] == ["LONG", "EXIT", "EXIT"]
    entry = signals[0]
    assert entry.metadata["divergence_type"] == "hidden"
    # Higher price low paired with a lower RSI low.
    assert entry.metadata["pivot_price_new"] > entry.metadata["pivot_price_prev"]
    assert entry.metadata["rsi_new"] < entry.metadata["rsi_prev"]

    disabled, _ = _run(closes, allow_short=False, use_hidden=False)
    assert disabled == []


def test_no_signal_without_divergence():
    monotone = [round(100.0 + 0.4 * index, 4) for index in range(60)]
    signals, _ = _run(monotone)
    assert signals == []
    # A flat tape has no pivots and no RSI gradient either.
    flat, _ = _run([100.0] * 60)
    assert flat == []


def test_pivots_closer_than_min_pivot_distance_are_rejected():
    """A textbook divergence squeezed into 2 bars is noise, not a swing pair."""
    tight = _tight_bullish_divergence_closes(gap=2)
    prev_index, new_index = _pivot_low_indices(tight)
    rsi = _rsi_series(tight)
    # The setup is otherwise perfect: lower price low, higher RSI low, sub-20.
    assert new_index - prev_index == 2
    assert tight[new_index] < tight[prev_index]
    assert rsi[prev_index] < rsi[new_index] <= 20.0

    assert _run(tight)[0] == []  # default min_pivot_distance = 3
    relaxed, _ = _run(tight, min_pivot_distance=2)
    assert [signal.signal_type for signal in relaxed] == ["LONG", "EXIT", "EXIT"]
    assert relaxed[0].metadata["pivot_distance"] == 2

    # One bar wider and the same shape clears the default gate: the bound is
    # exclusive, so only the separation - not the geometry - was blocking.
    at_bound, _ = _run(_tight_bullish_divergence_closes(gap=3))
    assert [signal.signal_type for signal in at_bound] == ["LONG", "EXIT", "EXIT"]
    assert at_bound[0].metadata["pivot_distance"] == 3


def test_pivots_further_apart_than_max_pivot_distance_are_rejected():
    """The same hidden divergence, stretched past the 40-bar admissibility cap."""
    stretched = _hidden_bullish_closes(mid_rally=40)
    prev_index, new_index = _pivot_low_indices(stretched)
    assert new_index - prev_index == 43  # past the default max_pivot_distance 40

    assert _run(stretched, allow_short=False)[0] == []
    relaxed, _ = _run(stretched, allow_short=False, max_pivot_distance=43)
    assert [signal.signal_type for signal in relaxed] == ["LONG", "EXIT", "EXIT"]
    assert relaxed[0].metadata["divergence_type"] == "hidden"
    assert relaxed[0].metadata["pivot_distance"] == 43
    # One bar tighter than the pair and it is out of reach again.
    assert _run(stretched, allow_short=False, max_pivot_distance=42)[0] == []

    # Control: the identical shape inside the cap is taken at the defaults.
    inside, _ = _run(_hidden_bullish_closes(mid_rally=35), allow_short=False)
    assert [signal.signal_type for signal in inside] == ["LONG", "EXIT", "EXIT"]
    assert inside[0].metadata["pivot_distance"] == 38


def test_zero_target_allocation_refuses_to_emit_an_unsized_entry():
    """An allocation of 0 must block the entry, not ship a signal without one.

    ``_target_metadata`` omits ``target_allocation`` when it is not positive,
    so the portfolio would size the order off its own config default - the
    sleeve would place a trade it never sized.
    """
    closes = _bullish_divergence_closes()
    assert _run(closes, target_allocation=0.0)[0] == []

    mirrored = [round(200.0 - close, 4) for close in closes]
    assert _run(mirrored, target_allocation=0.0)[0] == []

    # Control: the identical tape with a real allocation fires, and is sized.
    sized, _ = _run(closes, target_allocation=0.03)
    assert [signal.signal_type for signal in sized] == ["LONG", "EXIT", "EXIT"]
    assert sized[0].metadata["target_allocation"] == pytest.approx(0.03)


def test_heavier_second_pivot_volume_invalidates_the_divergence():
    closes = _bullish_divergence_closes()
    new_index = _pivot_low_indices(closes)[1]
    volumes = [250.0 if index == new_index else 100.0 for index in range(len(closes))]
    blocked, _ = _run(closes, volumes=volumes)
    assert blocked == []
    # The same tape passes once the volume gate is switched off.
    allowed, _ = _run(closes, volumes=volumes, require_volume_confirmation=False)
    assert [signal.signal_type for signal in allowed] == ["LONG", "EXIT", "EXIT"]


def test_missing_volume_feed_does_not_block_the_signal():
    closes = _bullish_divergence_closes()
    events = SimpleQueue()
    strategy = RsiDivergenceScaleOutStrategy(_Bars(), events)
    for index, close in enumerate(closes):
        strategy.calculate_signals(
            MarketEvent(index, SYMBOL, close, close + _HALF_RANGE, close - _HALF_RANGE, close, None)
        )
    assert [signal.signal_type for signal in _drain(events)] == ["LONG", "EXIT", "EXIT"]


def test_pivot_stop_exit():
    closes = [*_bullish_divergence_closes(rally_bars=0), 78.3, 77.3, 76.2, 75.4]
    new_index = _pivot_low_indices(closes)[1]
    stop_level = closes[new_index] - _HALF_RANGE
    signals, _ = _run(closes)
    assert [signal.signal_type for signal in signals] == ["LONG", "EXIT"]
    exit_signal = signals[1]
    assert exit_signal.metadata["reason"] == "pivot_stop"
    assert exit_signal.metadata["stop_price"] == pytest.approx(stop_level)
    assert exit_signal.price < stop_level


def test_max_hold_exit():
    closes = _bullish_divergence_closes(rally_bars=0)
    closes += [round(78.3 + 0.01 * step, 4) for step in range(10)]
    signals, _ = _run(closes, max_hold_bars=4)
    assert [signal.signal_type for signal in signals] == ["LONG", "EXIT"]
    assert signals[1].metadata["reason"] == "max_hold"
    assert signals[1].metadata["bars_held"] == 4
    assert int(signals[1].datetime) - int(signals[0].datetime) == 4


def test_staged_exit_takes_a_partial_then_closes_the_remainder():
    closes = _bullish_divergence_closes()
    rsi = _rsi_series(closes)
    signals, strategy = _run(closes)
    assert [signal.signal_type for signal in signals] == ["LONG", "EXIT", "EXIT"]
    first, second = signals[1], signals[2]

    # Stage 1: a PARTIAL reduction on the first bar to reach the 40~50 band.
    assert first.metadata["reason"] == "stage1_rsi"
    assert first.metadata["exit_fraction"] == pytest.approx(0.6)
    assert rsi[int(first.datetime)] >= 45.0
    assert rsi[int(first.datetime) - 1] < 45.0

    # Nothing is emitted while RSI sits between the stages: the rest is held.
    held = range(int(first.datetime) + 1, int(second.datetime))
    assert len(held) >= 1, "fixture must leave bars between the two stages"
    assert all(45.0 <= rsi[index] < 60.0 for index in held)

    # Stage 2: the remainder leaves on a FULL exit carrying no fraction key, so
    # a consumer that ignores partial exits still flattens the position.
    assert second.metadata["reason"] == "stage2_rsi"
    assert "exit_fraction" not in second.metadata
    assert rsi[int(second.datetime)] >= 60.0
    assert rsi[int(second.datetime) - 1] < 60.0

    state = strategy.get_state()["symbol_state"][SYMBOL]
    assert state["mode"] == "OUT"
    assert state["exit_stage"] == 0


def test_first_exit_fraction_of_one_collapses_to_a_single_full_exit():
    closes = _bullish_divergence_closes()
    staged, _ = _run(closes)
    single, _ = _run(closes, first_exit_fraction=1.0)
    assert [signal.signal_type for signal in single] == ["LONG", "EXIT"]
    assert single[1].datetime == staged[1].datetime  # same first-stage bar
    assert single[1].metadata["reason"] == "stage1_rsi"
    assert "exit_fraction" not in single[1].metadata


def test_state_round_trip_in_stage_one_does_not_repeat_the_partial():
    closes = _bullish_divergence_closes()
    rsi = _rsi_series(closes)
    baseline, _ = _run(closes)
    split = int(baseline[1].datetime) + 1  # resume right after the stage-1 exit
    assert rsi[split] >= 45.0  # a stage-0 restore would fire a second partial here

    warm_events = SimpleQueue()
    warm = RsiDivergenceScaleOutStrategy(_Bars(), warm_events)
    _feed(warm, closes[:split])
    assert [signal.signal_type for signal in _drain(warm_events)] == ["LONG", "EXIT"]
    state = warm.get_state()
    assert state["symbol_state"][SYMBOL]["exit_stage"] == 1
    assert state["symbol_state"][SYMBOL]["mode"] == "LONG"  # remainder still open

    resumed_events = SimpleQueue()
    resumed = RsiDivergenceScaleOutStrategy(_Bars(), resumed_events)
    resumed.set_state(state)
    assert resumed.get_state()["symbol_state"][SYMBOL]["exit_stage"] == 1
    _feed(resumed, closes[split:], start_index=split)
    replayed = _drain(resumed_events)
    assert [signal.signal_type for signal in replayed] == ["EXIT"]
    assert replayed[0].metadata["reason"] == "stage2_rsi"
    assert replayed[0].datetime == baseline[2].datetime
    assert "exit_fraction" not in replayed[0].metadata


def test_repeated_timestamp_is_ignored():
    closes = _bullish_divergence_closes()
    events = SimpleQueue()
    strategy = RsiDivergenceScaleOutStrategy(_Bars(), events)
    for index, close in enumerate(closes):
        event = MarketEvent(
            index, SYMBOL, close, close + _HALF_RANGE, close - _HALF_RANGE, close, 100.0
        )
        strategy.calculate_signals(event)
        strategy.calculate_signals(event)  # duplicate delivery of the same bar
    assert [signal.signal_type for signal in _drain(events)] == ["LONG", "EXIT", "EXIT"]


def test_state_round_trip_preserves_behaviour():
    closes = _bullish_divergence_closes()
    split = 30
    baseline, _ = _run(closes)

    warm_events = SimpleQueue()
    warm = RsiDivergenceScaleOutStrategy(_Bars(), warm_events)
    _feed(warm, closes[:split])
    assert _drain(warm_events) == []
    snapshot = warm.get_state()

    resumed_events = SimpleQueue()
    resumed = RsiDivergenceScaleOutStrategy(_Bars(), resumed_events)
    resumed.set_state(snapshot)
    _feed(resumed, closes[split:], start_index=split)
    replayed = _drain(resumed_events)

    assert [(signal.datetime, signal.signal_type) for signal in replayed] == [
        (signal.datetime, signal.signal_type) for signal in baseline
    ]
    assert replayed[0].metadata == baseline[0].metadata
    assert replayed[0].stop_loss is baseline[0].stop_loss is None
    assert replayed[1].metadata == baseline[1].metadata


def test_state_round_trip_while_holding_a_position():
    closes = _bullish_divergence_closes()
    entry_index = _pivot_low_indices(closes)[1] + 1
    split = entry_index + 1
    baseline, _ = _run(closes)

    warm_events = SimpleQueue()
    warm = RsiDivergenceScaleOutStrategy(_Bars(), warm_events)
    _feed(warm, closes[:split])
    assert [signal.signal_type for signal in _drain(warm_events)] == ["LONG"]
    snapshot = warm.get_state()

    resumed_events = SimpleQueue()
    resumed = RsiDivergenceScaleOutStrategy(_Bars(), resumed_events)
    resumed.set_state(snapshot)
    # The open trade carries the pivot volumes the opposing-volume rule needs.
    assert resumed.get_state()["symbol_state"][SYMBOL]["pivot_volume_avg"] == pytest.approx(
        snapshot["symbol_state"][SYMBOL]["pivot_volume_avg"]
    )
    assert snapshot["symbol_state"][SYMBOL]["pivot_volume_avg"] == pytest.approx(100.0)
    _feed(resumed, closes[split:], start_index=split)
    replayed = _drain(resumed_events)
    assert [signal.signal_type for signal in replayed] == ["EXIT", "EXIT"]
    assert replayed[0].datetime == baseline[1].datetime
    assert replayed[0].metadata == baseline[1].metadata
    assert replayed[1].metadata == baseline[2].metadata


def test_market_window_contract_matches_the_market_event_path():
    closes = _bullish_divergence_closes()
    baseline, _ = _run(closes)

    events = SimpleQueue()
    strategy = RsiDivergenceScaleOutStrategy(_Bars(), events)
    for index, close in enumerate(closes):
        # Two 1s rows per decision window; the aggregate reproduces the same
        # high/low/close/volume the MarketEvent path saw.
        rows = (
            (2 * index + 1, close - 0.2, close + _HALF_RANGE, close - 0.4, close - 0.1, 50.0),
            (2 * index + 2, close - 0.1, close + 0.3, close - _HALF_RANGE, close, 50.0),
        )
        strategy.calculate_signals(
            MarketWindowEvent(
                time=2 * index + 2,
                window_seconds=600,
                bars_1s={SYMBOL: rows},
            )
        )
    windowed = _drain(events)
    assert [signal.signal_type for signal in windowed] == [
        signal.signal_type for signal in baseline
    ]
    assert windowed[0].metadata == baseline[0].metadata
    assert windowed[0].stop_loss is baseline[0].stop_loss is None
    assert windowed[1].metadata["reason"] == baseline[1].metadata["reason"]


def test_bad_bars_are_ignored_without_raising():
    events = SimpleQueue()
    strategy = RsiDivergenceScaleOutStrategy(_Bars(), events)
    strategy.calculate_signals(MarketEvent(0, SYMBOL, 100.0, None, None, None, 1.0))
    strategy.calculate_signals(MarketEvent(1, "ETH/USDT", 100.0, 100.5, 99.5, 100.0, 1.0))
    strategy.calculate_signals(object())
    assert _drain(events) == []


def test_default_thresholds_reject_a_divergence_that_is_only_mildly_oversold():
    """The pre-audit 25/75 bands would have taken this tape; 20/80 does not."""
    closes = _bullish_divergence_closes()
    new_index = _pivot_low_indices(closes)[1]
    # Same divergence geometry, shallower second leg: the newer RSI low now
    # sits between 20 and 25, so only the old bands would have accepted it.
    milder = _bullish_divergence_closes(rebound=1.2, decline=1.5)
    assert _pivot_low_indices(milder) == _pivot_low_indices(closes)
    assert _rsi_series(closes)[new_index] <= 20.0
    assert 20.0 < _rsi_series(milder)[new_index] <= 25.0

    assert _run(milder)[0] == []
    relaxed, _ = _run(milder, oversold=25.0)
    assert [signal.signal_type for signal in relaxed] == ["LONG", "EXIT", "EXIT"]


def test_htf_confirmation_blocks_entries_against_the_higher_timeframe():
    # Ten-minute bars grouped three at a time stand in for the 30m chart.
    htf = {"require_htf_confirmation": True, "htf_multiple": 3, "htf_ma_window": 10}

    # The regular-divergence tape is a dump: the HTF close sits under its MA,
    # so the long is refused even though the divergence itself is valid.
    dump = _bullish_divergence_closes()
    assert _run(dump, **htf)[0] == []
    assert [signal.signal_type for signal in _run(dump)[0]] == ["LONG", "EXIT", "EXIT"]

    # The hidden-divergence tape is an uptrend: the same filter lets it through.
    uptrend = _hidden_bullish_closes()
    confirmed, _ = _run(uptrend, allow_short=False, **htf)
    assert [signal.signal_type for signal in confirmed] == ["LONG", "EXIT", "EXIT"]
    assert confirmed[0].metadata["divergence_type"] == "hidden"

    # Not enough COMPLETED higher-timeframe groups -> no entry at all.
    assert _run(dump, require_htf_confirmation=True, htf_multiple=20, htf_ma_window=20)[0] == []


def test_htf_confirmation_gates_the_short_leg_symmetrically():
    downtrend = [round(250.0 - close, 4) for close in _hidden_bullish_closes()]
    unfiltered, _ = _run(downtrend)
    # Unfiltered, the book first takes a counter-trend long on this tape.
    assert unfiltered[0].signal_type == "LONG"

    filtered, _ = _run(downtrend, require_htf_confirmation=True, htf_multiple=3, htf_ma_window=10)
    # The counter-trend long is refused, leaving only the with-trend short.
    assert [signal.signal_type for signal in filtered] == ["SHORT", "EXIT", "EXIT"]
    assert int(filtered[0].datetime) > int(unfiltered[0].datetime)
    assert filtered[0].metadata["divergence_type"] == "hidden"
    assert filtered[1].metadata["exit_fraction"] == pytest.approx(0.6)
    assert "exit_fraction" not in filtered[2].metadata


def test_opposing_volume_bar_exits_the_position():
    closes = _bullish_divergence_closes()
    entry_index = _pivot_low_indices(closes)[1] + 1
    shock = entry_index + 1
    # One bearish bar (opens at its high, closes at its low) on 10x volume.
    opens = [
        round(close + _HALF_RANGE, 4) if index == shock else close
        for index, close in enumerate(closes)
    ]
    volumes = [1000.0 if index == shock else 100.0 for index in range(len(closes))]

    baseline, _ = _run(closes, volumes=volumes, opens=opens)
    assert [signal.signal_type for signal in baseline] == ["LONG", "EXIT", "EXIT"]
    assert baseline[1].metadata["reason"] == "stage1_rsi"  # rule is off by default

    stopped, _ = _run(closes, volumes=volumes, opens=opens, opposing_volume_multiple=2.0)
    assert [signal.signal_type for signal in stopped] == ["LONG", "EXIT"]
    assert stopped[0].datetime == entry_index
    assert stopped[1].datetime == shock
    assert stopped[1].metadata["reason"] == "opposing_volume"
    assert stopped[1].metadata["side"] == "LONG"
    # A risk exit closes everything, from either stage: no partial fraction.
    assert "exit_fraction" not in stopped[1].metadata

    # 1000 does not clear 20x the 100-lot pivot average, so nothing fires.
    quiet, _ = _run(closes, volumes=volumes, opens=opens, opposing_volume_multiple=20.0)
    assert quiet[1].metadata["reason"] == "stage1_rsi"

    # The same heavy print on a bar that agrees with the position is ignored.
    with_trend, _ = _run(closes, volumes=volumes, opposing_volume_multiple=2.0)
    assert with_trend[1].metadata["reason"] == "stage1_rsi"


def test_state_round_trip_preserves_htf_state():
    closes = _hidden_bullish_closes()
    htf = {
        "allow_short": False,
        "require_htf_confirmation": True,
        "htf_multiple": 3,
        "htf_ma_window": 10,
    }
    # Mid-group split, late enough that the tail alone cannot rebuild ten
    # completed HTF bars before the entry: the restored history is load-bearing.
    split = 71
    baseline, _ = _run(closes, **htf)
    assert [signal.signal_type for signal in baseline] == ["LONG", "EXIT", "EXIT"]

    warm_events = SimpleQueue()
    warm = RsiDivergenceScaleOutStrategy(_Bars(), warm_events, **htf)
    _feed(warm, closes[:split])
    assert _drain(warm_events) == []
    state = warm.get_state()
    snapshot = state["symbol_state"][SYMBOL]
    assert snapshot["htf_bar_count"] == split % 3  # an unfinished HTF bar
    assert len(snapshot["htf_closes"]) == 10  # capped at htf_ma_window

    resumed_events = SimpleQueue()
    resumed = RsiDivergenceScaleOutStrategy(_Bars(), resumed_events, **htf)
    resumed.set_state(state)
    restored = resumed.get_state()["symbol_state"][SYMBOL]
    assert restored["htf_closes"] == snapshot["htf_closes"]
    assert restored["htf_bar_count"] == snapshot["htf_bar_count"]
    _feed(resumed, closes[split:], start_index=split)
    replayed = _drain(resumed_events)
    assert [(signal.datetime, signal.signal_type) for signal in replayed] == [
        (signal.datetime, signal.signal_type) for signal in baseline
    ]
    assert replayed[0].metadata == baseline[0].metadata


def test_registered_in_global_registry():
    assert (
        GLOBAL_REGISTRY.get("strategy", "RsiDivergenceScaleOutStrategy")
        is RsiDivergenceScaleOutStrategy
    )
    assert RsiDivergenceScaleOutStrategy.decision_cadence_seconds == 600
    schema = RsiDivergenceScaleOutStrategy.get_param_schema()
    # Defaults mirror the audited public rule: 10m bars, RSI 11, sub-20 / above-80
    # divergence zones, first exit in the 40~50 band and the rest above 60.
    assert schema["rsi_period"].default == 11
    assert schema["oversold"].default == pytest.approx(20.0)
    assert schema["overbought"].default == pytest.approx(80.0)
    assert schema["exit_rsi_first"].default == pytest.approx(45.0)
    assert schema["exit_rsi_second"].default == pytest.approx(60.0)
    # "More than half" off at the first stage; the staged exit is not optional.
    assert schema["first_exit_fraction"].default == pytest.approx(0.6)
    assert schema["first_exit_fraction"].tunable is False
    assert "full_exit_at_second" not in schema
    # Both additions are off by default so the audited rule stays the baseline.
    assert schema["require_htf_confirmation"].default is False
    assert schema["opposing_volume_multiple"].default == pytest.approx(0.0)
    assert schema["htf_multiple"].default == 6  # 1h out of 10-minute bars
    assert schema["htf_ma_window"].default == 20
    assert schema["htf_multiple"].tunable is False
    assert schema["target_allocation"].tunable is False
    assert schema["max_order_value"].tunable is False
