"""Deterministic tests for KalmanPairsStatArbStrategy.

The pair is built in LOG space, which is where the filter regresses::

    log_x = drift * i + osc(i)
    log_y = 2 * log_x + 5 + spread(i)

so the TRUE hedge the Kalman posterior has to find is ``beta = 2`` with
``alpha = 5``, and ``spread`` is the tradable deviation (sd ~2% in log terms).
The x leg carries a deterministic oscillation on purpose: with a flat x leg the
hedge is not identifiable (only ``beta*log_x + alpha`` is), so the filter could
sit on any slope/intercept pair and a beta assertion would be vacuous.

Randomness is confined to seeded ``numpy.random.default_rng`` draws that are
materialised into plain lists, so every series here is byte-reproducible.

Coverage: the SHIPPED DEFAULTS trading a cointegrated pair end to end, the
hedge-scaled x leg and its cap, the reversion / stop / coint-break / half-life /
max-hold exits, the ADF gate rejecting two independent random walks, the
half-life cap being derived from the Engle-Granger residual, the min_updates
warmup guard, degenerate input, both MARKET_WINDOW paths (aligned and gappy
per-symbol sub-times) and the get_state/set_state round-trip.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from lumina_quant.core.events import MarketEvent, MarketWindowEvent
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.indicators import KalmanHedgeState
from lumina_quant.strategies.kalman_pairs_stat_arb import KalmanPairsStatArbStrategy

_SYMBOL_Y = "YY/USDT"
_SYMBOL_X = "XX/USDT"

_TRUE_BETA = 2.0
_TRUE_ALPHA = 5.0
_X_DRIFT = 0.0006
_X_OSC_AMPLITUDE = 0.05
_X_OSC_PERIOD = 9.0


class _Bars:
    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbol_list = list(symbols if symbols is not None else [_SYMBOL_Y, _SYMBOL_X])


class _Events:
    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _SecondLegFailsEvents(_Events):
    def put(self, event: Any) -> None:
        if len(self.signals) == 1:
            raise RuntimeError("second leg queue failure")
        super().put(event)


# --------------------------------------------------------------------- series


def _log_x(index: int) -> float:
    """Smooth drift plus a small deterministic oscillation (~5% peak to trough)."""
    return _X_DRIFT * float(index) + _X_OSC_AMPLITUDE * math.sin(float(index) / _X_OSC_PERIOD)


def _pair_prices(spread: float, index: int) -> tuple[float, float]:
    log_x = _log_x(index)
    return math.exp(_TRUE_BETA * log_x + _TRUE_ALPHA + spread), math.exp(log_x)


def _ou_spread(
    count: int, *, phi: float = 0.85, scale: float = 0.010, seed: int = 7
) -> list[float]:
    """Seeded OU spread: stationary sd ``scale / sqrt(1 - phi^2)`` in log terms."""
    rng = np.random.default_rng(seed)
    level = 0.0
    out: list[float] = []
    for shock in rng.normal(0.0, scale, count):
        level = phi * level + float(shock)
        out.append(level)
    return out


def _cointegrated_pairs(count: int, *, start: int = 0, **kwargs: Any) -> list[tuple[float, float]]:
    """``count`` aligned ``(y, x)`` closes of the cointegrated pair from ``start``."""
    spreads = _ou_spread(start + count, **kwargs)[start:]
    return [_pair_prices(spread, start + offset) for offset, spread in enumerate(spreads)]


def _blowout_pairs(
    count: int, *, start: int = 100, base: float = 0.003
) -> list[tuple[float, float]]:
    """A pair whose spread detonates geometrically after ``start`` (never reverts)."""
    pairs: list[tuple[float, float]] = []
    for index in range(count):
        spread = 0.004 * math.sin(float(index) / 3.0)
        if index >= start:
            spread += base * 1.8 ** (index - start)
        pairs.append(_pair_prices(spread, index))
    return pairs


def _independent_walk_pairs(count: int, *, seed: int = 0) -> list[tuple[float, float]]:
    """Two INDEPENDENT log random walks: no cointegration exists to trade.

    Seed 0 is hard-coded because it is a seed whose walks the ADF gate rejects
    (the gate is what must block them; the same series trades with the gate off).
    """
    rng = np.random.default_rng(seed)
    log_y = math.log(100.0)
    log_x = math.log(50.0)
    pairs: list[tuple[float, float]] = []
    for shock_y, shock_x in zip(rng.normal(0.0, 0.01, count), rng.normal(0.0, 0.01, count)):
        log_y += float(shock_y)
        log_x += float(shock_x)
        pairs.append((math.exp(log_y), math.exp(log_x)))
    return pairs


# ---------------------------------------------------------------------- feeds


def _time_key(index: int) -> str:
    return f"T{index:05d}"


def _build(**params: Any) -> tuple[KalmanPairsStatArbStrategy, _Events]:
    """The strategy under its SHIPPED defaults; only the symbols are supplied."""
    events = _Events()
    strategy = KalmanPairsStatArbStrategy(
        _Bars(), events, symbol_y=_SYMBOL_Y, symbol_x=_SYMBOL_X, **params
    )
    return strategy, events


def _feed(
    strategy: Any,
    events: _Events,
    pairs: list[tuple[float, float]],
    *,
    start: int = 0,
) -> list[tuple[int, list[Any]]]:
    """Feed aligned MARKET bars and return ``(index, signals)`` for emitting bars."""
    batches: list[tuple[int, list[Any]]] = []
    for offset, (y_price, x_price) in enumerate(pairs):
        index = start + offset
        before = len(events.signals)
        for symbol, price in ((_SYMBOL_Y, y_price), (_SYMBOL_X, x_price)):
            strategy.calculate_signals(
                MarketEvent(
                    time=_time_key(index),
                    symbol=symbol,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=1.0,
                )
            )
        if len(events.signals) > before:
            batches.append((index, events.signals[before:]))
    return batches


def _feed_windows(
    strategy: Any,
    pairs: list[tuple[float, float]],
    *,
    sub_times: dict[str, str] | None = None,
) -> None:
    """Feed the same pair through MARKET_WINDOW events, optionally with gappy tapes."""
    for index, (y_price, x_price) in enumerate(pairs):
        rows = {}
        for symbol, price in ((_SYMBOL_Y, y_price), (_SYMBOL_X, x_price)):
            suffix = "" if sub_times is None else sub_times[symbol]
            rows[symbol] = ((f"{_time_key(index)}{suffix}", price, price, price, price, 1.0),)
        strategy.calculate_signals(
            MarketWindowEvent(time=_time_key(index), window_seconds=3600, bars_1s=rows)
        )


def _entries(batches: list[tuple[int, list[Any]]]) -> list[tuple[int, list[Any]]]:
    return [item for item in batches if item[1][0].signal_type in {"LONG", "SHORT"}]


def _exits(batches: list[tuple[int, list[Any]]]) -> list[tuple[int, list[Any]]]:
    return [item for item in batches if item[1][0].signal_type == "EXIT"]


def _fingerprint(events: _Events) -> list[tuple[str, str, dict[str, Any]]]:
    return [(signal.symbol, signal.signal_type, signal.metadata) for signal in events.signals]


# ---------------------------------------------------------------------- tests


def test_registry_exposes_kalman_pairs_stat_arb() -> None:
    assert GLOBAL_REGISTRY.get("strategy", "KalmanPairsStatArbStrategy") is (
        KalmanPairsStatArbStrategy
    )
    assert GLOBAL_REGISTRY.get_interface("strategy", "KalmanPairsStatArbStrategy") == "event_driven"


def test_shipped_defaults_trade_a_cointegrated_pair() -> None:
    """The defaults alone must be able to enter AND revert -- no overrides.

    ``kalman_obs_noise`` is an observation VARIANCE: at 1e-3 (sd ~3.2% in log
    terms) the standardized innovation of a real pair never reaches ``entry_z``,
    which made the shipped configuration inert.
    """
    schema = KalmanPairsStatArbStrategy.get_param_schema()
    assert schema["signal_mode"].default == "spread_z"
    assert schema["kalman_obs_noise"].default == 1e-5

    strategy, events = _build()
    assert strategy.signal_mode == "spread_z"
    assert strategy.require_cointegration is True

    batches = _feed(strategy, events, _cointegrated_pairs(1500))
    entries, exits = _entries(batches), _exits(batches)
    assert entries, "the shipped defaults never entered a cointegrated pair"

    _, first = entries[0]
    y_leg, x_leg = first
    assert (y_leg.symbol, x_leg.symbol) == (_SYMBOL_Y, _SYMBOL_X)
    assert {y_leg.signal_type, x_leg.signal_type} == {"LONG", "SHORT"}
    assert abs(y_leg.metadata["z"]) >= 2.0
    assert y_leg.metadata["signal_mode"] == "spread_z"

    assert all(len(batch) == 2 for _, batch in exits)
    assert any(batch[0].metadata["reason"] == "reversion" for _, batch in exits), (
        "a mean-reverting spread must produce a reversion exit"
    )


def test_entry_legs_are_hedge_scaled_and_beta_finds_the_true_ratio() -> None:
    strategy, events = _build()
    batches = _feed(strategy, events, _cointegrated_pairs(250))
    entries = _entries(batches)
    assert entries

    _, (y_leg, x_leg) = entries[0]
    beta = y_leg.metadata["beta"]
    # The filter must FIND the hedge the series was built with, not merely
    # report back whatever slope it happened to be carrying.
    assert beta == pytest.approx(_TRUE_BETA, abs=0.3)
    assert y_leg.metadata["target_allocation"] == 0.10
    assert x_leg.metadata["target_allocation"] == min(0.30, 0.10 * beta)
    assert x_leg.metadata["target_allocation"] > 0.10  # the hedge scales the x leg up
    assert x_leg.metadata["z"] == y_leg.metadata["z"]

    mode = "SHORT_SPREAD" if y_leg.signal_type == "SHORT" else "LONG_SPREAD"
    for leg, name in ((y_leg, "y"), (x_leg, "x")):
        assert leg.metadata["max_order_value"] == 500.0
        assert leg.metadata["mode"] == mode
        assert leg.metadata["leg"] == name
        assert leg.metadata["pair"] == f"{_SYMBOL_Y}|{_SYMBOL_X}"
        assert leg.metadata["signal_mode"] == "spread_z"

    exits = _exits(batches)
    assert exits
    assert "target_allocation" not in exits[0][1][0].metadata


def test_x_leg_allocation_is_capped_by_max_leg_allocation() -> None:
    strategy, events = _build(max_leg_allocation=0.05)
    batches = _feed(strategy, events, _cointegrated_pairs(250))
    entries = _entries(batches)
    assert entries
    _, (y_leg, x_leg) = entries[0]
    assert 0.10 * y_leg.metadata["beta"] > 0.05  # the cap is the binding constraint here
    assert x_leg.metadata["target_allocation"] == 0.05


def test_adverse_blowout_triggers_stop_exit() -> None:
    strategy, events = _build(require_cointegration=False)
    batches = _feed(strategy, events, _blowout_pairs(106))
    assert len(batches) >= 2

    _, entry = batches[0]
    assert [signal.signal_type for signal in entry] == ["SHORT", "LONG"]
    assert entry[0].metadata["z"] >= 2.0

    _, exit_batch = batches[1]
    assert {signal.signal_type for signal in exit_batch} == {"EXIT"}
    assert {signal.metadata["reason"] for signal in exit_batch} == {"stop"}
    # The stop fires on the ADVERSE side only: z never crossed back through zero.
    assert exit_batch[0].metadata["z"] >= 4.0


def test_uncalibrated_residual_adf_heuristic_blocks_two_independent_random_walks() -> None:
    """The heuristic reads the Engle-Granger residual, not the filter residual.

    The filter's own a-posteriori residual is white noise by construction, so an
    ADF test on it rejects a unit root even for these two independent walks.
    """
    walks = _independent_walk_pairs(400)

    gated, gated_events = _build(entry_z=1.5)
    _feed(gated, gated_events, walks)
    assert gated_events.signals == []
    # The gate must have EVALUATED and rejected -- not silently returned None.
    assert gated._adf_pass() is False

    # Same series, same z, same warmup: only the gate differs.
    ungated, ungated_events = _build(entry_z=1.5, require_cointegration=False)
    entries = _entries(_feed(ungated, ungated_events, walks))
    assert entries, "the ungated control must trade, or the gate proves nothing"
    assert entries[0][1][0].metadata["beta"] > 0.0  # min_beta is not what blocked the gated run
    assert entries[0][1][0].metadata["cointegration_gate"] == "uncalibrated_residual_adf_heuristic"
    assert "not a calibrated 5% Engle-Granger test" in (
        KalmanPairsStatArbStrategy.get_param_schema()["require_cointegration"].description
    )

    # ... and the genuinely cointegrated pair does pass that same gate.
    passing, passing_events = _build()
    gate: list[bool | None] = []
    passing_batches: list[tuple[int, list[Any]]] = []
    for index, pair in enumerate(_cointegrated_pairs(250)):
        passing_batches.extend(_feed(passing, passing_events, [pair], start=index))
        gate.append(passing._adf_pass())
    assert True in gate
    assert _entries(passing_batches)


def test_cointegration_break_exits_open_position() -> None:
    strategy, events = _build(
        entry_z=1.5,
        require_cointegration=False,
        exit_on_cointegration_break=True,
    )
    batches = _feed(strategy, events, _independent_walk_pairs(400))
    assert len(batches) >= 2
    _, exit_batch = batches[1]
    assert {signal.signal_type for signal in exit_batch} == {"EXIT"}
    assert {signal.metadata["reason"] for signal in exit_batch} == {"coint_break"}


def test_half_life_cap_is_derived_from_the_engle_granger_residual() -> None:
    """A white-noise residual would collapse the cap to a single bar."""
    strategy, events = _build()
    batches = _feed(strategy, events, _cointegrated_pairs(500))

    cap = strategy._half_life_cap()
    assert cap is not None
    assert cap > 1

    exits = _exits(batches)
    assert exits
    assert any(
        batch[0].metadata["reason"] == "reversion" and batch[0].metadata["bars_held"] > 1
        for _, batch in exits
    )


def test_max_hold_bars_is_the_outer_cap() -> None:
    strategy, events = _build(
        require_cointegration=False,
        half_life_multiple=0.0,
        exit_z=0.0,
        max_hold_bars=5,
    )
    slow = _cointegrated_pairs(200, phi=0.98, scale=0.004, seed=3)
    batches = _feed(strategy, events, slow)
    assert len(batches) >= 2
    entry_index, _ = batches[0]
    exit_index, exit_batch = batches[1]
    assert {signal.metadata["reason"] for signal in exit_batch} == {"max_hold"}
    assert exit_index - entry_index == 5
    assert exit_batch[0].metadata["bars_held"] == 5


def test_no_signal_before_min_updates() -> None:
    strategy, events = _build()
    batches = _feed(strategy, events, _cointegrated_pairs(250))
    assert batches
    assert min(index for index, _ in batches) >= 60

    blocked, blocked_events = _build(min_updates=10_000)
    _feed(blocked, blocked_events, _cointegrated_pairs(250))
    assert blocked_events.signals == []


def test_degenerate_input_stays_silent() -> None:
    strategy, events = _build(min_updates=2)

    # Only one leg ever prints: no aligned step can be formed.
    for index in range(80):
        price = math.exp(_log_x(index))
        strategy.calculate_signals(
            MarketEvent(
                time=_time_key(index),
                symbol=_SYMBOL_Y,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=1.0,
            )
        )
    assert events.signals == []

    # Non-positive / missing closes and unknown event types are ignored.
    strategy.calculate_signals(
        MarketEvent(
            time="T99999", symbol=_SYMBOL_X, open=1.0, high=1.0, low=1.0, close=0.0, volume=1.0
        )
    )
    strategy.calculate_signals(object())
    strategy.calculate_signals_window(
        MarketWindowEvent(time="T99999", window_seconds=3600, bars_1s={})
    )
    assert events.signals == []

    # A single-symbol universe leaves the pair unresolvable, so the sleeve is inert.
    lone_events = _Events()
    lone = KalmanPairsStatArbStrategy(_Bars([_SYMBOL_Y]), lone_events, min_updates=2)
    assert lone.enabled is False
    _feed(lone, lone_events, _cointegrated_pairs(120))
    assert lone_events.signals == []


def test_market_window_path_matches_market_event_path() -> None:
    pairs = _cointegrated_pairs(250)

    market, market_events = _build()
    _feed(market, market_events, pairs)

    window, window_events = _build()
    _feed_windows(window, pairs)

    assert market_events.signals
    assert _fingerprint(window_events) == _fingerprint(market_events)


def test_market_window_rejects_stale_native_leg_timestamps() -> None:
    """A window timestamp cannot overwrite a native completed-bar timestamp."""
    pairs = _cointegrated_pairs(250)

    gappy, gappy_events = _build()
    _feed_windows(gappy, pairs, sub_times={_SYMBOL_Y: ":59", _SYMBOL_X: ":07"})

    assert gappy._kalman is None
    assert gappy_events.signals == []
    assert gappy.get_state()["unmatched_leg_drops"] == len(pairs) * 2 - 1

    # A window with no timestamp of its own falls back to the row stamps.
    timeless, _ = _build()
    for index, (y_price, x_price) in enumerate(pairs[:120]):
        rows = {
            symbol: ((_time_key(index), price, price, price, price, 1.0),)
            for symbol, price in ((_SYMBOL_Y, y_price), (_SYMBOL_X, x_price))
        }
        timeless.calculate_signals(MarketWindowEvent(time=None, window_seconds=3600, bars_1s=rows))
    assert timeless._kalman is not None
    assert timeless._kalman.updates == 120


def test_stale_leg_is_dropped_before_a_matching_completed_bar_steps() -> None:
    strategy, _ = _build(min_updates=2)
    pair_0, pair_1 = _cointegrated_pairs(2)
    strategy.calculate_signals(
        MarketEvent("T00000", _SYMBOL_Y, pair_0[0], pair_0[0], pair_0[0], pair_0[0], 1.0)
    )
    strategy.calculate_signals(
        MarketEvent("T00001", _SYMBOL_X, pair_1[1], pair_1[1], pair_1[1], pair_1[1], 1.0)
    )
    assert strategy._kalman is None
    assert strategy.get_state()["unmatched_leg_drops"] == 1

    strategy.calculate_signals(
        MarketEvent("T00001", _SYMBOL_Y, pair_1[0], pair_1[0], pair_1[0], pair_1[0], 1.0)
    )
    assert strategy._kalman is not None
    assert strategy._kalman.updates == 1


def test_second_leg_queue_failure_does_not_commit_entry_state() -> None:
    events = _SecondLegFailsEvents()
    strategy = KalmanPairsStatArbStrategy(
        _Bars(),
        events,
        symbol_y=_SYMBOL_Y,
        symbol_x=_SYMBOL_X,
        min_updates=2,
        require_cointegration=False,
    )
    state = KalmanHedgeState(beta=1.0, alpha=0.0, p00=1.0, p01=0.0, p11=1.0, updates=2)

    strategy._maybe_enter("T00000", 100.0, 100.0, state, 2.0)

    assert len(events.signals) == 1
    assert strategy.get_state()["mode"] == "FLAT"
    assert strategy.get_state()["bars_held"] == 0
    assert strategy.get_state()["emission_failures"] == 1


def test_state_roundtrip_preserves_behaviour() -> None:
    reference, reference_events = _build()
    _feed(reference, reference_events, _cointegrated_pairs(400))

    warm, warm_events = _build()
    _feed(warm, warm_events, _cointegrated_pairs(200))
    snapshot = warm.get_state()
    assert snapshot["kalman"] is not None
    assert snapshot["mode"] in {"FLAT", "LONG_SPREAD", "SHORT_SPREAD"}
    assert len(snapshot["paired_history"]) == 92
    assert snapshot["pending_y"] == [_time_key(199), _cointegrated_pairs(1, start=199)[0][0]]

    resumed, resumed_events = _build()
    resumed.set_state(snapshot)
    _feed(resumed, resumed_events, _cointegrated_pairs(200, start=200), start=200)

    tail = _fingerprint(reference_events)[len(warm_events.signals) :]
    assert tail
    assert _fingerprint(resumed_events) == tail

    # An empty / malformed payload must not raise or corrupt the live filter.
    resumed.set_state({})
    resumed.set_state("not-a-dict")
    assert resumed.get_state()["kalman"] is not None

    # Corrupt paired history is rejected atomically; it cannot skew the legs.
    before = resumed.get_state()
    corrupt = dict(snapshot)
    corrupt["paired_history"] = [*snapshot["paired_history"][:-1], ["T00199", "bad", 0.0]]
    resumed.set_state(corrupt)
    assert resumed.get_state() == before

    corrupt_pending = dict(snapshot)
    corrupt_pending["pending_y"] = ["T00199", 0.0]
    resumed.set_state(corrupt_pending)
    assert resumed.get_state() == before
