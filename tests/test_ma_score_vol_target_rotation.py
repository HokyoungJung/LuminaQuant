"""Deterministic tests for MaScoreVolTargetRotationStrategy.

Synthetic daily closes are built from FIXED alternating log-returns (no
randomness), which gives every symbol a constant, exactly-known realized
volatility and a monotone price path, so the inverse-vol risk-parity split and
the moving-average score are both analytically predictable:

    AAA  log-returns alternate +0.010 / +0.002  -> strictly rising, vol v
    BBB  log-returns alternate -0.010 / -0.002  -> strictly falling, vol v
    CCC  log-returns alternate +0.020 / +0.004  -> strictly rising, vol 2v

Covers the first rebalance ranking, the hysteresis hold, a regime flip that both
promotes a new leader and flattens a faded one, the per-asset vol-target clamp,
the insufficient-history and thin-cross-section no-ops, MARKET vs MARKET_WINDOW
equivalence, get_state/set_state round-trip, adversarial input safety, and the
plugin-registry wiring.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.core.events import MarketEvent, MarketWindowEvent
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.indicators import moving_average_score
from lumina_quant.strategies.ma_score_vol_target_rotation import (
    MaScoreVolTargetRotationStrategy,
)

_UP_FAST = (0.010, 0.002)
_UP_LOUD = (0.020, 0.004)
_DOWN_FAST = (-0.010, -0.002)
_DOWN_HARD = (-0.012, -0.004)
_SYMBOLS = ["AAA", "BBB", "CCC"]


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _path(start: float, phases: list[tuple[tuple[float, ...], int]]) -> list[float]:
    """Compound ``start`` through fixed, cyclically repeated log-return phases."""
    out = [float(start)]
    for returns, count in phases:
        for step in range(count):
            out.append(out[-1] * math.exp(returns[step % len(returns)]))
    return out


def _stamp(index: int) -> str:
    return f"D{index:04d}"


def _feed_market(
    strategy: MaScoreVolTargetRotationStrategy,
    prices: dict[str, list[float]],
    *,
    start: int = 0,
    stop: int | None = None,
) -> None:
    symbols = list(prices)
    end = len(prices[symbols[0]]) if stop is None else stop
    for index in range(start, end):
        for symbol in symbols:
            close = prices[symbol][index]
            strategy.calculate_signals(
                MarketEvent(
                    time=_stamp(index),
                    symbol=symbol,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=1000.0,
                )
            )


def _feed_window(
    strategy: MaScoreVolTargetRotationStrategy,
    prices: dict[str, list[float]],
) -> None:
    symbols = list(prices)
    for index in range(len(prices[symbols[0]])):
        bars = {
            symbol: (
                # Row stamp deliberately DIFFERS from the window stamp: the
                # decision key must come from the window, not the last 1s row.
                (f"{_stamp(index)}T23:59:59", *(prices[symbol][index],) * 4, 1000.0),
            )
            for symbol in symbols
        }
        strategy.calculate_signals(
            MarketWindowEvent(time=_stamp(index), window_seconds=86400, bars_1s=bars)
        )


def _three_symbol_prices(bars: int) -> dict[str, list[float]]:
    return {
        "AAA": _path(100.0, [(_UP_FAST, bars - 1)]),
        "BBB": _path(100.0, [(_DOWN_FAST, bars - 1)]),
        "CCC": _path(100.0, [(_UP_LOUD, bars - 1)]),
    }


def _fingerprint(signals: list[Any]) -> list[tuple[Any, ...]]:
    return [
        (
            signal.signal_type,
            signal.symbol,
            signal.datetime,
            round(float(signal.metadata.get("target_allocation", 0.0)), 12),
            signal.metadata.get("reason"),
        )
        for signal in signals
    ]


def _build(symbols: list[str], **params: Any) -> tuple[MaScoreVolTargetRotationStrategy, _Queue]:
    queue = _Queue()
    return MaScoreVolTargetRotationStrategy(_Bars(symbols), queue, **params), queue


# --------------------------------------------------------------------------- #
# entry ranking
# --------------------------------------------------------------------------- #


def test_first_rebalance_ranks_by_score_and_inverse_vol() -> None:
    """Only the two rising names enter; the calmer one gets the bigger weight."""
    strategy, queue = _build(_SYMBOLS)
    _feed_market(strategy, _three_symbol_prices(30))

    signals = queue.items
    assert [signal.signal_type for signal in signals] == ["LONG", "LONG"]
    assert [signal.symbol for signal in signals] == ["AAA", "CCC"]
    # The falling name never trades: its MA score is 0, below ``min_score``.
    assert all(signal.symbol != "BBB" for signal in signals)

    by_symbol = {signal.symbol: signal for signal in signals}
    alloc_a = float(by_symbol["AAA"].metadata["target_allocation"])
    alloc_c = float(by_symbol["CCC"].metadata["target_allocation"])
    assert alloc_a > alloc_c > 0.0
    assert alloc_a <= strategy.max_weight
    assert alloc_c <= strategy.max_weight
    assert alloc_a + alloc_c <= strategy.gross_cap

    for signal in signals:
        assert signal.metadata["max_order_value"] == 500.0
        assert signal.metadata["ma_score"] == 1.0
        assert signal.metadata["vol"] > 0.0
        assert 0.0 < signal.metadata["rp_weight"] <= 1.0
        assert signal.metadata["vol_clamp"] == 1.0
        assert signal.price is not None

    # CCC's returns are exactly 2x AAA's in log space, so its realized vol is
    # exactly double and its inverse-vol share exactly half.
    assert math.isclose(
        by_symbol["CCC"].metadata["vol"], 2.0 * by_symbol["AAA"].metadata["vol"], rel_tol=1e-9
    )
    assert math.isclose(by_symbol["AAA"].metadata["rp_weight"], 0.4, rel_tol=1e-9)
    assert math.isclose(by_symbol["CCC"].metadata["rp_weight"], 0.2, rel_tol=1e-9)


def test_hysteresis_holds_when_the_cross_section_is_unchanged() -> None:
    """A second rebalance on an identical cross-section emits nothing."""
    strategy, queue = _build(_SYMBOLS)
    _feed_market(strategy, _three_symbol_prices(30))

    # 30 evaluations at ``rebalance_bars=5`` means rebalance ticks 25 and 30;
    # only the first one produced orders.
    assert strategy._tick == 30
    assert {signal.datetime for signal in queue.items} == {_stamp(24)}


# --------------------------------------------------------------------------- #
# regime flip
# --------------------------------------------------------------------------- #


def test_regime_flip_promotes_new_leader_and_flattens_faded_name() -> None:
    """BBB turning up earns a LONG; AAA rolling over is EXITed on the score floor."""
    prices = {
        "AAA": _path(100.0, [(_UP_FAST, 29), (_DOWN_HARD, 30)]),
        "BBB": _path(100.0, [(_DOWN_FAST, 29), (_UP_FAST, 30)]),
        "CCC": _path(100.0, [(_UP_LOUD, 59)]),
    }
    strategy, queue = _build(_SYMBOLS)
    _feed_market(strategy, prices)

    signals = queue.items
    long_bbb = [s for s in signals if s.symbol == "BBB" and s.signal_type == "LONG"]
    assert long_bbb, "BBB must earn an entry once its MA score clears min_score"
    first_bbb = long_bbb[0]
    assert first_bbb.metadata["ma_score"] >= strategy.min_score
    assert first_bbb.metadata["target_allocation"] > 0.0

    aaa = [s for s in signals if s.symbol == "AAA"]
    assert aaa[0].signal_type == "LONG"
    exit_aaa = [s for s in aaa if s.metadata.get("reason") == "score_below_floor"]
    assert exit_aaa, "AAA must be flattened once its MA score falls under min_score"
    assert exit_aaa[0].signal_type == "EXIT"
    assert exit_aaa[0].metadata["ma_score"] == 0.0
    # The flatten happens after the entry, and nothing re-enters AAA afterwards.
    flatten_at = signals.index(exit_aaa[0])
    assert flatten_at > signals.index(aaa[0])
    assert not [
        s for s in signals[flatten_at + 1 :] if s.symbol == "AAA" and s.signal_type == "LONG"
    ]


def test_resize_emits_exit_then_long_at_the_new_weight() -> None:
    """A live long that changes size round-trips (EXIT then LONG), never a bare add."""
    prices = {
        "AAA": _path(100.0, [(_UP_FAST, 29), (_DOWN_HARD, 30)]),
        "BBB": _path(100.0, [(_DOWN_FAST, 29), (_UP_FAST, 30)]),
        "CCC": _path(100.0, [(_UP_LOUD, 59)]),
    }
    strategy, queue = _build(_SYMBOLS)
    _feed_market(strategy, prices)

    resizes = [s for s in queue.items if s.metadata.get("reason") == "resize"]
    assert resizes, "the flip must resize at least one live position"
    for resize in resizes:
        assert resize.signal_type == "EXIT"
        position = queue.items.index(resize)
        follow_up = queue.items[position + 1]
        assert follow_up.signal_type == "LONG"
        assert follow_up.symbol == resize.symbol
        assert math.isclose(
            float(follow_up.metadata["target_allocation"]),
            float(resize.metadata["target_weight"]),
            rel_tol=1e-12,
        )
        assert (
            abs(float(resize.metadata["target_weight"]) - float(resize.metadata["previous_weight"]))
            >= strategy.min_weight_change
        )


# --------------------------------------------------------------------------- #
# vol target
# --------------------------------------------------------------------------- #


def test_vol_target_clamp_shrinks_the_loud_name_only() -> None:
    """A per-bar target below CCC's vol scales CCC down and leaves AAA alone."""
    prices = _three_symbol_prices(30)
    baseline, baseline_queue = _build(_SYMBOLS)
    _feed_market(baseline, prices)
    unclamped = {s.symbol: float(s.metadata["target_allocation"]) for s in baseline_queue.items}

    strategy, queue = _build(_SYMBOLS, target_vol_per_bar=0.005)
    _feed_market(strategy, prices)
    by_symbol = {signal.symbol: signal for signal in queue.items}

    assert set(by_symbol) == {"AAA", "CCC"}
    assert by_symbol["AAA"].metadata["vol_clamp"] == 1.0
    clamp_c = float(by_symbol["CCC"].metadata["vol_clamp"])
    assert 0.0 < clamp_c < 1.0
    assert math.isclose(clamp_c, 0.005 / float(by_symbol["CCC"].metadata["vol"]), rel_tol=1e-12)
    assert float(by_symbol["CCC"].metadata["target_allocation"]) < unclamped["CCC"]
    # AAA is already capped by ``max_weight`` in both runs, so it is untouched.
    assert float(by_symbol["AAA"].metadata["target_allocation"]) == unclamped["AAA"]


def test_target_vol_off_disables_the_clamp() -> None:
    """``target_vol_per_bar=0`` leaves the raw risk-parity weight in place."""
    strategy, queue = _build(_SYMBOLS, target_vol_per_bar=0.0, max_weight=1.0)
    _feed_market(strategy, _three_symbol_prices(30))
    by_symbol = {signal.symbol: signal for signal in queue.items}
    for symbol in ("AAA", "CCC"):
        assert by_symbol[symbol].metadata["vol_clamp"] == 1.0
        assert math.isclose(
            float(by_symbol[symbol].metadata["target_allocation"]),
            float(by_symbol[symbol].metadata["rp_weight"]),
            rel_tol=1e-12,
        )


# --------------------------------------------------------------------------- #
# no-op guards
# --------------------------------------------------------------------------- #


def test_insufficient_history_emits_nothing() -> None:
    strategy, queue = _build(_SYMBOLS)
    _feed_market(strategy, _three_symbol_prices(30), stop=12)
    assert queue.items == []


def test_thin_cross_section_is_skipped() -> None:
    """One eligible symbol cannot satisfy ``min_symbols=2``."""
    prices = {"AAA": _path(100.0, [(_UP_FAST, 29)])}
    strategy, queue = _build(["AAA"])
    _feed_market(strategy, prices)
    assert queue.items == []


def test_degenerate_input_never_raises() -> None:
    strategy, queue = _build(_SYMBOLS)
    strategy.calculate_signals(SimpleNamespace(type="FILL", symbol="AAA"))
    strategy.calculate_signals(
        MarketEvent(time="D0000", symbol="ZZZ", open=1, high=1, low=1, close=1, volume=1)
    )
    strategy.calculate_signals(
        SimpleNamespace(
            type="MARKET", time="D0000", symbol="AAA", open=1, high=1, low=1, close=None, volume=1
        )
    )
    for bad_close in (0.0, -5.0, float("nan")):
        strategy.calculate_signals(
            SimpleNamespace(
                type="MARKET",
                time="D0001",
                symbol="AAA",
                open=1,
                high=1,
                low=1,
                close=bad_close,
                volume=None,
            )
        )
    strategy.calculate_signals(MarketWindowEvent(time="D0002", window_seconds=86400, bars_1s={}))
    strategy.calculate_signals_window(
        MarketWindowEvent(time="D0003", window_seconds=86400, bars_1s={"AAA": ()})
    )
    assert queue.items == []
    assert list(strategy._state["AAA"].closes) == []


def test_flat_prices_stay_out() -> None:
    """Zero realized volatility makes a symbol ineligible instead of infinite-weight."""
    flat = [100.0] * 40
    strategy, queue = _build(_SYMBOLS)
    _feed_market(strategy, {symbol: list(flat) for symbol in _SYMBOLS})
    assert queue.items == []


# --------------------------------------------------------------------------- #
# no look-ahead: the bar being acted on never feeds the signal that trades it
# --------------------------------------------------------------------------- #


def test_current_bar_spike_does_not_flip_the_lagged_ma_score() -> None:
    """A gap-up on the decision bar must not buy itself.

    Look-ahead regression: ``_features`` scores ``closes[:-1]``.  AAA's lagged
    closes are a clean downtrend (score 0.0, under ``min_score``), while the bar
    being acted on gaps far above every moving average -- reading the current
    bar would score AAA 1.0 and open the position the spike created.
    """
    prices = {
        # bars 0..28 fall, bar 29 gaps to 400.
        "AAA": [*_path(100.0, [(_DOWN_FAST, 28)]), 400.0],
        "CCC": _path(100.0, [(_UP_LOUD, 29)]),
    }
    strategy, queue = _build(["AAA", "CCC"])
    _feed_market(strategy, prices)

    closes = list(strategy._state["AAA"].closes)
    assert closes[-1] == 400.0, "the spike bar reached the book"
    features = strategy._features(strategy._state["AAA"])
    assert features is not None
    lagged_score, _vol = features
    assert lagged_score == 0.0, "every LAGGED close sits below its moving averages"
    assert moving_average_score(closes, windows=strategy.ma_score_windows) == 1.0, (
        "reading the current bar WOULD score the spike as fully risk-on"
    )
    assert lagged_score < strategy.min_score
    assert not [signal for signal in queue.items if signal.symbol == "AAA"], (
        "the lagged rule keeps AAA flat through its own gap-up"
    )


def test_eligibility_requires_lagged_history_not_the_current_bar() -> None:
    """One close short of the lagged requirement is still one close short.

    At bar 20 the book holds 21 closes -- exactly the indicator requirement, but
    only 20 of them are LAGGED.  Counting the decision bar would let both names
    trade a bar early.
    """
    prices = {
        "AAA": _path(100.0, [(_UP_FAST, 21)]),
        "CCC": _path(100.0, [(_UP_LOUD, 21)]),
    }
    strategy, queue = _build(["AAA", "CCC"], rebalance_bars=1)

    _feed_market(strategy, prices, stop=21)  # bars 0..20
    assert list(strategy._state["AAA"].closes) == prices["AAA"][:21]
    assert strategy._features(strategy._state["AAA"]) is None
    assert queue.items == [], "20 lagged closes cannot satisfy a 21-close requirement"

    _feed_market(strategy, prices, start=21)  # bar 21
    assert [signal.datetime for signal in queue.items] == [_stamp(21), _stamp(21)]
    assert [signal.signal_type for signal in queue.items] == ["LONG", "LONG"]


# --------------------------------------------------------------------------- #
# contracts
# --------------------------------------------------------------------------- #


def test_window_and_market_paths_agree() -> None:
    prices = _three_symbol_prices(30)
    market, market_queue = _build(_SYMBOLS)
    _feed_market(market, prices)
    window, window_queue = _build(_SYMBOLS)
    _feed_window(window, prices)
    assert _fingerprint(window_queue.items) == _fingerprint(market_queue.items)


def test_state_round_trip_preserves_behaviour() -> None:
    prices = {
        "AAA": _path(100.0, [(_UP_FAST, 29), (_DOWN_HARD, 30)]),
        "BBB": _path(100.0, [(_DOWN_FAST, 29), (_UP_FAST, 30)]),
        "CCC": _path(100.0, [(_UP_LOUD, 59)]),
    }
    original, original_queue = _build(_SYMBOLS)
    _feed_market(original, prices, stop=30)
    warmup_signals = len(original_queue.items)
    assert warmup_signals > 0
    snapshot = json.loads(json.dumps(original.get_state()))

    restored, restored_queue = _build(_SYMBOLS)
    restored.set_state(snapshot)
    assert restored.get_state() == original.get_state()

    _feed_market(original, prices, start=30)
    _feed_market(restored, prices, start=30)
    assert _fingerprint(restored_queue.items) == _fingerprint(original_queue.items[warmup_signals:])
    assert len(restored_queue.items) > 0
    assert restored.get_state() == original.get_state()


def test_set_state_ignores_garbage() -> None:
    strategy, queue = _build(_SYMBOLS)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not-a-dict", "tick": "nope", "pending_count": -3})
    strategy.set_state(
        {
            "symbol_state": {
                "AAA": {"closes": ["x", None, 101.0], "weight": "bad", "last_time_key": 7},
                "UNKNOWN": {"closes": [1.0]},
            }
        }
    )
    assert list(strategy._state["AAA"].closes) == [101.0]
    assert strategy._state["AAA"].weight == 0.0
    assert strategy._tick == 0
    assert queue.items == []


def test_param_schema_and_class_contract() -> None:
    schema = MaScoreVolTargetRotationStrategy.get_param_schema()
    assert set(schema) == {
        "ma_score_windows",
        "vol_window",
        "rebalance_bars",
        "target_vol_per_bar",
        "gross_cap",
        "max_weight",
        "min_score",
        "min_weight_change",
        "min_symbols",
        "max_order_value",
    }
    assert schema["max_order_value"].tunable is False
    assert MaScoreVolTargetRotationStrategy.decision_cadence_seconds == 86400
    assert MaScoreVolTargetRotationStrategy.preferred_contract == "market_window"
    assert MaScoreVolTargetRotationStrategy.uses_timeframe_aggregator is False

    strategy, _ = _build(_SYMBOLS, ma_score_windows=" 20, 5 ,5, oops , -3 ")
    assert strategy.ma_score_windows == (5, 20)
    fallback, _ = _build(_SYMBOLS, ma_score_windows="nonsense")
    assert fallback.ma_score_windows == (3, 5, 10, 20)


def test_registered_in_plugin_registry() -> None:
    assert (
        GLOBAL_REGISTRY.get("strategy", "MaScoreVolTargetRotationStrategy")
        is MaScoreVolTargetRotationStrategy
    )
