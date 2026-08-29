"""Deterministic unit tests for the PCA residual s-score stat-arb book.

The fixture is a 6-name panel driven by one deterministic common factor.  Five
names carry small, fast idiosyncratic cycles; "ZZZ" carries a large one whose
cumulative residual sits far above its long-run mean early in the sample, which
is exactly the configuration that drives the Avellaneda-Lee s-score strongly
NEGATIVE (``s = -m / sigma_eq`` with the auxiliary process centred at the last
bar).  Expectations are derived by calling ``pca_residual_sscores`` on the same
panel the strategy builds, never by hard-coding s values.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from lumina_quant.core.events import MarketWindowEvent
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.indicators.stat_arb import pca_residual_sscores
from lumina_quant.strategies.pca_residual_stat_arb import PcaResidualStatArbStrategy
from lumina_quant.tuning import HyperParam

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE", "ZZZ"]
BETA = {"AAA": 1.0, "BBB": 0.9, "CCC": 1.1, "DDD": 1.05, "EEE": 0.95, "ZZZ": 1.0}
LOOKBACK = 60
BARS = 70
# First bar index at which the panel is complete (lookback + 1 closes).
FIRST_EVAL_BAR = LOOKBACK


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


def _factor(t: int) -> float:
    return 0.010 * math.sin(2.0 * math.pi * t / 23.0) + 0.004 * math.cos(2.0 * math.pi * t / 7.0)


def _idio(symbol: str, t: int) -> float:
    if symbol == "ZZZ":
        return 0.010 * math.sin(2.0 * math.pi * t / 20.0)
    k = SYMBOLS.index(symbol)
    return 0.0008 * math.sin(2.0 * math.pi * t / (17.0 + 3.0 * k) + 0.7 * k)


def _closes(symbols: list[str] = SYMBOLS, bars: int = BARS) -> dict[str, list[float]]:
    series = {symbol: [100.0] for symbol in symbols}
    for t in range(1, bars):
        for symbol in symbols:
            step = BETA[symbol] * _factor(t) + _idio(symbol, t)
            series[symbol].append(series[symbol][-1] * math.exp(step))
    return series


def _window(ts: int, closes: dict[str, float]) -> MarketWindowEvent:
    return MarketWindowEvent(
        time=ts,
        window_seconds=86400,
        bars_1s={
            symbol: ((int(ts), close, close, close, close, 1000.0),)
            for symbol, close in closes.items()
        },
    )


def _feed(
    strategy: PcaResidualStatArbStrategy,
    series: dict[str, list[float]],
    start: int,
    end: int,
) -> None:
    """Feed bar indices ``start..end`` inclusive as MARKET_WINDOW events."""
    for idx in range(start, end + 1):
        strategy.calculate_signals(
            _window(idx, {symbol: values[idx] for symbol, values in series.items()})
        )


def _expected_scores(
    series: dict[str, list[float]], end_bar: int, *, symbols: list[str] = SYMBOLS
) -> dict[str, float | None]:
    """s-scores from the indicator over the exact panel the strategy builds."""
    rows = [
        [math.log(series[symbol][i] / series[symbol][i - 1]) for symbol in symbols]
        for i in range(end_bar - LOOKBACK + 1, end_bar + 1)
    ]
    values = pca_residual_sscores(rows, n_factors=1, max_half_life_bars=None, min_rows=30)
    return dict(zip(symbols, values, strict=True))


def _signals(queue: _Queue, signal_type: str) -> dict[str, Any]:
    return {signal.symbol: signal for signal in queue.items if signal.signal_type == signal_type}


def _build(symbols: list[str] = SYMBOLS, **params: Any) -> tuple[Any, _Queue]:
    queue = _Queue()
    return PcaResidualStatArbStrategy(_Bars(symbols), queue, **params), queue


# --------------------------------------------------------------- registry


def test_registry_membership() -> None:
    assert GLOBAL_REGISTRY.get("strategy", "PcaResidualStatArbStrategy") is (
        PcaResidualStatArbStrategy
    )


def test_param_schema_shape() -> None:
    schema = PcaResidualStatArbStrategy.get_param_schema()
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "lookback_bars",
        "n_factors",
        "rebalance_bars",
        "min_symbols",
        "min_rows",
        "max_half_life_bars",
        "s_open",
        "s_close_long",
        "s_close_short",
        "s_stop",
        "max_longs",
        "max_shorts",
        "allow_short",
        "max_hold_bars",
        "none_tolerance_evals",
        "gross_cap",
        "max_position_allocation",
        "require_balanced",
        "max_order_value",
    ):
        assert required in schema
    assert schema["max_order_value"].tunable is False
    assert schema["max_position_allocation"].tunable is False
    assert PcaResidualStatArbStrategy.decision_cadence_seconds == 86400


# ------------------------------------------------------------ entry ladder


def test_first_evaluation_opens_long_on_the_depressed_residual() -> None:
    series = _closes()
    expected = _expected_scores(series, FIRST_EVAL_BAR)
    strategy, queue = _build()
    s_open = strategy.s_open

    # Fixture sanity, straight from the indicator: ZZZ is the only name whose
    # residual is far enough below its OU mean to trigger the long side.
    assert expected["ZZZ"] is not None
    assert expected["ZZZ"] < -s_open
    assert [sym for sym, value in expected.items() if value is not None and value < -s_open] == [
        "ZZZ"
    ]

    _feed(strategy, series, 0, FIRST_EVAL_BAR - 1)
    assert queue.items == []  # panel incomplete -> silent

    _feed(strategy, series, FIRST_EVAL_BAR, FIRST_EVAL_BAR)
    longs = _signals(queue, "LONG")
    shorts = _signals(queue, "SHORT")
    assert "ZZZ" in longs
    assert len(longs) <= strategy.max_longs
    assert len(shorts) <= strategy.max_shorts
    for symbol in longs:
        assert expected[symbol] is not None and expected[symbol] < -s_open
    for symbol in shorts:
        assert expected[symbol] is not None and expected[symbol] > s_open
    assert shorts  # the same panel puts at least one name above +s_open

    signal = longs["ZZZ"]
    assert signal.price == series["ZZZ"][FIRST_EVAL_BAR]
    metadata = signal.metadata
    # gross_cap 0.60 spread over max_longs + max_shorts = 6 slots.
    assert metadata["target_allocation"] == pytest.approx(0.60 / 6.0)
    assert metadata["max_order_value"] == 500.0
    assert metadata["max_symbol_exposure_pct"] == pytest.approx(0.60 / 6.0)
    assert metadata["strategy"] == "PcaResidualStatArbStrategy"
    assert metadata["reason"] == "open_long"
    assert metadata["n_factors"] == 1
    assert metadata["panel_size"] == len(SYMBOLS)
    assert metadata["s_score"] == expected["ZZZ"]


def test_allow_short_off_keeps_the_book_long_only() -> None:
    series = _closes()
    strategy, queue = _build(allow_short=False)
    _feed(strategy, series, 0, FIRST_EVAL_BAR)
    assert "ZZZ" in _signals(queue, "LONG")
    assert _signals(queue, "SHORT") == {}


def test_require_balanced_needs_matched_pairs() -> None:
    """With only one long candidate, pairing caps the new book at 1x1."""
    series = _closes()
    strategy, queue = _build(require_balanced=True)
    _feed(strategy, series, 0, FIRST_EVAL_BAR)
    longs = _signals(queue, "LONG")
    shorts = _signals(queue, "SHORT")
    assert "ZZZ" in longs
    assert len(longs) == len(shorts) == 1


# ------------------------------------------------- contested slot allocation
# The six-name fixture above never has more candidates than slots, so the
# ranking sorts in ``_open_positions`` are never actually exercised: reversing
# them (least-extreme s-score wins) left every test green.  The panel below adds
# a SECOND loud name so both books have to choose.

SLOT_SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "YYY", "ZZZ"]
# amplitude and phase of the shared 20-bar idiosyncratic cycle; the phase shift
# is what separates YYY's s-score from ZZZ's.
SLOT_LOUD = {"YYY": (0.008, 0.3), "ZZZ": (0.010, 0.0)}
SLOT_S_OPEN = 0.9


def _slot_closes(bars: int = BARS) -> dict[str, list[float]]:
    """Same common factor as ``_closes``, but with TWO large-idio names."""
    series = {symbol: [100.0] for symbol in SLOT_SYMBOLS}
    for t in range(1, bars):
        for symbol in SLOT_SYMBOLS:
            if symbol in SLOT_LOUD:
                amplitude, phase = SLOT_LOUD[symbol]
                idio = amplitude * math.sin(2.0 * math.pi * t / 20.0 + phase)
            else:
                idio = _idio(symbol, t)
            step = BETA.get(symbol, 1.0) * _factor(t) + idio
            series[symbol].append(series[symbol][-1] * math.exp(step))
    return series


def test_contested_long_slot_goes_to_the_most_negative_s_score() -> None:
    series = _slot_closes()
    expected = _expected_scores(series, FIRST_EVAL_BAR, symbols=SLOT_SYMBOLS)

    # Fixture sanity, straight from the indicator: exactly two names clear the
    # long threshold, and the winner is NOT the alphabetically first of them.
    candidates = sorted(
        symbol for symbol, value in expected.items() if value is not None and value < -SLOT_S_OPEN
    )
    assert candidates == ["YYY", "ZZZ"]
    assert expected["ZZZ"] < expected["YYY"] < -SLOT_S_OPEN

    strategy, queue = _build(SLOT_SYMBOLS, max_longs=1, allow_short=False, s_open=SLOT_S_OPEN)
    _feed(strategy, series, 0, FIRST_EVAL_BAR)

    longs = _signals(queue, "LONG")
    assert list(longs) == ["ZZZ"], "one slot, two candidates: the MORE negative one wins"
    assert longs["ZZZ"].metadata["s_score"] == expected["ZZZ"]
    assert strategy._state["ZZZ"].side == "LONG"
    assert strategy._state["YYY"].side == "OUT", "the runner-up must stay flat"


def test_contested_short_slot_goes_to_the_most_positive_s_score() -> None:
    series = _slot_closes()
    expected = _expected_scores(series, FIRST_EVAL_BAR, symbols=SLOT_SYMBOLS)

    candidates = sorted(
        symbol for symbol, value in expected.items() if value is not None and value > SLOT_S_OPEN
    )
    assert candidates == ["BBB", "CCC", "DDD"]
    assert expected["DDD"] > expected["CCC"] > expected["BBB"] > SLOT_S_OPEN

    strategy, queue = _build(SLOT_SYMBOLS, max_longs=0, max_shorts=1, s_open=SLOT_S_OPEN)
    _feed(strategy, series, 0, FIRST_EVAL_BAR)

    shorts = _signals(queue, "SHORT")
    assert list(shorts) == ["DDD"], "one slot, three candidates: the MOST positive one wins"
    assert shorts["DDD"].metadata["s_score"] == expected["DDD"]
    assert _signals(queue, "LONG") == {}, "max_longs=0 closes the long book"
    for runner_up in ("BBB", "CCC"):
        assert strategy._state[runner_up].side == "OUT"


# ------------------------------------------------------------- exit ladder


def test_long_exits_once_the_residual_reverts() -> None:
    series = _closes()
    strategy, queue = _build()
    _feed(strategy, series, 0, FIRST_EVAL_BAR)
    assert "ZZZ" in _signals(queue, "LONG")

    # Walk forward until the indicator says ZZZ has reverted past -s_close_long.
    exit_bar = None
    for bar in range(FIRST_EVAL_BAR + 1, BARS):
        value = _expected_scores(series, bar)["ZZZ"]
        if value is not None and value > -strategy.s_close_long:
            exit_bar = bar
            break
    assert exit_bar is not None
    # The bar before must still be inside the position (fixture sanity).
    prior = _expected_scores(series, exit_bar - 1)["ZZZ"]
    assert prior is not None and prior <= -strategy.s_close_long

    queue.items.clear()
    _feed(strategy, series, FIRST_EVAL_BAR + 1, exit_bar - 1)
    assert "ZZZ" not in _signals(queue, "EXIT")

    queue.items.clear()
    _feed(strategy, series, exit_bar, exit_bar)
    exits = _signals(queue, "EXIT")
    assert "ZZZ" in exits
    assert exits["ZZZ"].metadata["reason"] == "residual_reverted"
    assert exits["ZZZ"].price == series["ZZZ"][exit_bar]
    assert strategy._state["ZZZ"].side == "OUT"


def test_max_hold_bars_ages_the_position_out() -> None:
    series = _closes()
    # s_close_long=0.0 pushes the reversion exit out of reach (ZZZ stays below 0
    # for the next few bars), leaving the age-out as the only live rule.
    strategy, queue = _build(max_hold_bars=2, s_close_long=0.0)
    _feed(strategy, series, 0, FIRST_EVAL_BAR)
    assert "ZZZ" in _signals(queue, "LONG")

    queue.items.clear()
    _feed(strategy, series, FIRST_EVAL_BAR + 1, FIRST_EVAL_BAR + 1)
    assert _signals(queue, "EXIT") == {}  # aged to 1 of 2

    _feed(strategy, series, FIRST_EVAL_BAR + 2, FIRST_EVAL_BAR + 2)
    exits = _signals(queue, "EXIT")
    assert "ZZZ" in exits
    assert exits["ZZZ"].metadata["reason"] == "max_hold"
    assert exits["ZZZ"].metadata["bars_held"] == 2


def _with_installed_position(strategy: Any, symbol: str, side: str) -> None:
    """Install an open position without disturbing the ingested close history."""
    state = strategy.get_state()
    state["positions"] = {symbol: {"side": side, "bars_held": 0, "entry_s": 1.9}}
    strategy.set_state(state)


def test_s_stop_takes_precedence_over_the_reversion_target() -> None:
    series = _closes()
    # A SHORT on ZZZ is deliberately offside: its s-score is far NEGATIVE, which
    # trips both the short reversion target and an |s| >= s_stop disaster stop.
    stopped, stopped_queue = _build(s_stop=1.0)
    _feed(stopped, series, 0, FIRST_EVAL_BAR)
    _with_installed_position(stopped, "ZZZ", "SHORT")
    stopped_queue.items.clear()
    _feed(stopped, series, FIRST_EVAL_BAR + 1, FIRST_EVAL_BAR + 1)
    assert _signals(stopped_queue, "EXIT")["ZZZ"].metadata["reason"] == "s_stop"

    # Same setup with the stop disabled falls through to the reversion target.
    plain, plain_queue = _build()
    assert plain.s_stop == 0.0
    _feed(plain, series, 0, FIRST_EVAL_BAR)
    _with_installed_position(plain, "ZZZ", "SHORT")
    plain_queue.items.clear()
    _feed(plain, series, FIRST_EVAL_BAR + 1, FIRST_EVAL_BAR + 1)
    assert _signals(plain_queue, "EXIT")["ZZZ"].metadata["reason"] == "residual_reverted"


def test_unestimable_s_flattens_after_the_grace_period() -> None:
    # Flat prices make every s-score unestimable (zero residual variance).
    series = {symbol: [100.0] * BARS for symbol in SYMBOLS}
    strategy, queue = _build(none_tolerance_evals=2)
    _feed(strategy, series, 0, FIRST_EVAL_BAR)
    assert queue.items == []
    _with_installed_position(strategy, "ZZZ", "LONG")

    _feed(strategy, series, FIRST_EVAL_BAR + 1, FIRST_EVAL_BAR + 1)
    assert _signals(queue, "EXIT") == {}  # first unestimable evaluation: tolerated

    _feed(strategy, series, FIRST_EVAL_BAR + 2, FIRST_EVAL_BAR + 2)
    exits = _signals(queue, "EXIT")
    assert "ZZZ" in exits
    assert exits["ZZZ"].metadata["reason"] == "s_unestimable"
    assert exits["ZZZ"].metadata["s_score"] is None


def test_missing_panel_still_flattens_an_open_position() -> None:
    """A universe too narrow to model must not strand an inherited position."""
    symbols = SYMBOLS[:4]  # below min_symbols=5 -> the panel never builds
    series = _closes(symbols)
    strategy, queue = _build(symbols, none_tolerance_evals=1)
    _feed(strategy, series, 0, FIRST_EVAL_BAR)
    assert queue.items == []
    _with_installed_position(strategy, "AAA", "LONG")

    _feed(strategy, series, FIRST_EVAL_BAR + 1, FIRST_EVAL_BAR + 1)
    exits = _signals(queue, "EXIT")
    assert "AAA" in exits
    assert exits["AAA"].metadata["reason"] == "s_unestimable"
    assert exits["AAA"].metadata["panel_size"] == 0
    assert _signals(queue, "LONG") == {}


# ---------------------------------------------------------- degenerate input


def test_narrow_universe_emits_nothing() -> None:
    symbols = SYMBOLS[:4]  # below min_symbols=5
    series = _closes(symbols)
    strategy, queue = _build(symbols)
    _feed(strategy, series, 0, BARS - 1)
    assert queue.items == []


def test_flat_prices_emit_nothing() -> None:
    series = {symbol: [100.0] * BARS for symbol in SYMBOLS}
    strategy, queue = _build()
    _feed(strategy, series, 0, BARS - 1)
    assert queue.items == []


def test_repeated_time_key_is_deduped() -> None:
    series = _closes()
    strategy, queue = _build()
    _feed(strategy, series, 0, FIRST_EVAL_BAR)
    emitted = len(queue.items)
    assert emitted > 0
    # Replaying the same bar must not re-evaluate or re-emit.
    _feed(strategy, series, FIRST_EVAL_BAR, FIRST_EVAL_BAR)
    assert len(queue.items) == emitted


# ------------------------------------------------------------ state restore


def test_state_round_trip_preserves_behaviour() -> None:
    series = _closes()
    live, live_queue = _build()
    _feed(live, series, 0, FIRST_EVAL_BAR + 1)
    assert any(signal.signal_type == "LONG" for signal in live_queue.items)

    snapshot = live.get_state()
    assert snapshot["positions"]["ZZZ"]["side"] == "LONG"

    restored, restored_queue = _build()
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot

    live_queue.items.clear()
    for strategy in (live, restored):
        _feed(strategy, series, FIRST_EVAL_BAR + 2, BARS - 1)

    def _fingerprint(queue: _Queue) -> list[tuple[Any, ...]]:
        return [
            (signal.symbol, signal.signal_type, signal.price, sorted(signal.metadata.items()))
            for signal in queue.items
        ]

    assert _fingerprint(restored_queue) == _fingerprint(live_queue)
    assert restored_queue.items  # the continuation actually traded


def test_set_state_ignores_garbage() -> None:
    strategy, queue = _build()
    strategy.set_state({"positions": "nope", "closes": 3, "eval_count": "x"})
    strategy.set_state([])  # type: ignore[arg-type]
    series = _closes()
    _feed(strategy, series, 0, FIRST_EVAL_BAR)
    assert "ZZZ" in _signals(queue, "LONG")


def test_sparse_window_uses_only_one_exact_common_timestamp_panel() -> None:
    series = _closes()
    strategy, _queue = _build()
    _feed(strategy, series, 0, FIRST_EVAL_BAR - 1)
    partial = SYMBOLS[:-1]
    strategy.calculate_signals(
        _window(
            FIRST_EVAL_BAR,
            {symbol: series[symbol][FIRST_EVAL_BAR] for symbol in partial},
        )
    )
    symbols, rows = strategy._panel(FIRST_EVAL_BAR * 1000)
    assert symbols == sorted(partial)
    assert "ZZZ" not in symbols
    assert len(rows) == LOOKBACK
    assert all(len(row) == len(partial) for row in rows)


def test_panel_and_signals_are_symbol_order_invariant() -> None:
    series = _closes()
    forward, forward_queue = _build(SYMBOLS)
    reverse_symbols = list(reversed(SYMBOLS))
    reverse, reverse_queue = _build(reverse_symbols)
    reverse_series = {symbol: series[symbol] for symbol in reverse_symbols}
    _feed(forward, series, 0, FIRST_EVAL_BAR)
    _feed(reverse, reverse_series, 0, FIRST_EVAL_BAR)

    def fingerprint(queue: _Queue) -> list[tuple[str, str, float]]:
        return sorted(
            (signal.symbol, signal.signal_type, signal.metadata["s_score"])
            for signal in queue.items
        )

    assert fingerprint(reverse_queue) == fingerprint(forward_queue)


def test_unexpected_model_failure_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = _closes()
    strategy, _queue = _build()

    def broken_model(*_args: Any, **_kwargs: Any):
        raise RuntimeError("model defect")

    monkeypatch.setattr(
        "lumina_quant.strategies.pca_residual_stat_arb.pca_residual_sscores",
        broken_model,
    )
    with pytest.raises(RuntimeError, match="model defect"):
        _feed(strategy, series, 0, FIRST_EVAL_BAR)


def test_risk_exit_cannot_reopen_same_bar_and_cooldown_survives_restore() -> None:
    strategy, queue = _build(s_stop=1.0)
    _with_installed_position(strategy, "AAA", "LONG")
    strategy._state["AAA"].closes.append(100.0)
    strategy._state["AAA"].close_times_ms.append(1)
    strategy._scores = lambda _time: ({"AAA": 2.0}, len(SYMBOLS))  # type: ignore[method-assign]
    strategy._evaluate(1, 1)
    assert [(item.signal_type, item.symbol) for item in queue.items] == [("EXIT", "AAA")]
    assert strategy._state["AAA"].side == "OUT"

    snapshot = strategy.get_state()
    restored, restored_queue = _build(s_stop=1.0)
    restored.set_state(snapshot)
    restored._scores = lambda _time: ({"AAA": 2.0}, len(SYMBOLS))  # type: ignore[method-assign]
    restored._evaluate(2, 2)
    assert restored_queue.items == []
    restored._evaluate(3, 3)
    assert [(item.signal_type, item.symbol) for item in restored_queue.items] == [("SHORT", "AAA")]


def test_required_balance_reduces_total_held_book_after_one_sided_exit() -> None:
    strategy, queue = _build(require_balanced=True, s_stop=1.0)
    for symbol, side in (("AAA", "LONG"), ("BBB", "LONG"), ("CCC", "SHORT"), ("DDD", "SHORT")):
        strategy._state[symbol].side = side
        strategy._state[symbol].closes.append(100.0)
        strategy._state[symbol].close_times_ms.append(1)
    strategy._scores = lambda _time: (  # type: ignore[method-assign]
        {"AAA": -2.0, "BBB": -0.8, "CCC": 0.8, "DDD": 0.8},
        len(SYMBOLS),
    )
    strategy._evaluate(1, 1)
    exits = _signals(queue, "EXIT")
    assert exits["AAA"].metadata["reason"] == "s_stop"
    assert any(item.metadata["reason"] == "balance_reduction" for item in exits.values())
    assert sum(item.side == "LONG" for item in strategy._state.values()) == 1
    assert sum(item.side == "SHORT" for item in strategy._state.values()) == 1


def test_queue_failure_propagates_without_advancing_position_state() -> None:
    class FailingQueue:
        def put(self, _item: Any) -> None:
            raise RuntimeError("queue unavailable")

    strategy = PcaResidualStatArbStrategy(_Bars(SYMBOLS), FailingQueue(), s_stop=1.0)
    strategy._state["AAA"].side = "LONG"
    strategy._state["AAA"].closes.append(100.0)
    strategy._state["AAA"].close_times_ms.append(1)
    strategy._scores = lambda _time: ({"AAA": 2.0}, len(SYMBOLS))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="queue unavailable"):
        strategy._evaluate(1, 1)
    assert strategy._state["AAA"].side == "LONG"
