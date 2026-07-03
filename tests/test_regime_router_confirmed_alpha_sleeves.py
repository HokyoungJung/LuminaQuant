"""Tests for RegimeRouterConfirmedRotationStrategy (Lane I2, CONDITIONAL).

This sleeve is a vol/cycle-CONFIRMED variant of the un-confirmed
``BullBearRegimeRotationStrategy``: it only flips into BULL/BEAR when a
GARCH conditional-vol read (or, when GARCH is unavailable, a spectral
cycle-phase fallback) concurs with the breadth+benchmark base vote.

The GATE this module must pass is the NON-REDUNDANCY test below: on an
identical multi-symbol chop fixture where breadth+benchmark alone vote
bear, the un-confirmed parent flips to a bear-short book while this sleeve
stays flat because its conditional-vol regime reads FALLING (not RISING),
i.e. the recent tape is not actually more turbulent than the calibration
history even though it is drifting down.  Companion tests prove the gate
is not simply dead (a genuine rising-vol decline DOES flip bear; a
sustained low-vol uptrend DOES flip bull) and exercise hysteresis,
never-raise safety, state roundtrip, and determinism.

All fixtures are synthetic and seeded via a hand-rolled LCG; no backtest is
run.  Imports the strategy class DIRECTLY (no @register in this lane) and
makes no registry/candidate assertions.
"""

from __future__ import annotations

import copy
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.bull_bear_regime_rotation import BullBearRegimeRotationStrategy
from lumina_quant.strategies.regime_router_confirmed_alpha_sleeves import (
    RegimeRouterConfirmedRotationStrategy,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)
_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]

_BASE_PARAMS: dict[str, Any] = {
    "momentum_lookback": 5,
    "trend_ma_window": 5,
    "signal_threshold": 0.01,
    "bull_breadth": 0.60,
    "bear_breadth": 0.60,
    "exit_breadth": 0.35,
    "benchmark_lookback": 5,
    "benchmark_bull_threshold": 0.005,
    "benchmark_bear_threshold": 0.005,
    "max_longs": 2,
    "max_shorts": 2,
    "rebalance_bars": 1,
    "stop_loss_pct": 0.20,
    "max_hold_bars": 10_000,
    "min_symbols": 5,
    "target_allocation": 0.60,
    "max_order_value": 1000.0,
}

_CONFIRM_PARAMS: dict[str, Any] = {
    "garch_window": 32,
    # Refit only once (during warmup, as soon as 32 obs are available); the
    # bear/chop/bull fixtures below are engineered around a FIXED calibration
    # so the story is "does the current tape confirm relative to history",
    # not "did the model adapt away the difference".
    "garch_refit_bars": 10_000,
    "confirm_bear_vol_ratio": 1.15,
    "confirm_bull_vol_ratio": 1.00,
    "cycle_window": 32,
    "cycle_min_period": 4,
    "cycle_max_period": 16,
    "min_cycle_purity": 0.15,
    "min_vol_size_scale": 0.35,
}

_WARM_SEED = 17
_WARM_NOISE = 0.008


class _BarsNoFeatures:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Events:
    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _Window:
    type = "MARKET_WINDOW"

    def __init__(self, time: str, prices: dict[str, float]) -> None:
        self.time = time
        self.bars_1s = {
            symbol: [
                {
                    "time": time,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 1_000_000.0,
                }
            ]
            for symbol, price in prices.items()
        }


class _EmptyWindow:
    type = "MARKET_WINDOW"
    bars_1s: dict[str, list[Any]] = {}
    time = _START.isoformat()


def _ts(i: int, *, step_minutes: int = 30) -> str:
    return (_START + timedelta(minutes=step_minutes * i)).isoformat()


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield (state / 2**31 - 0.5) * 2.0


def _parent(events: _Events) -> BullBearRegimeRotationStrategy:
    return BullBearRegimeRotationStrategy(_BarsNoFeatures(_SYMBOLS), events, **_BASE_PARAMS)


def _confirmed(events: _Events) -> RegimeRouterConfirmedRotationStrategy:
    return RegimeRouterConfirmedRotationStrategy(
        _BarsNoFeatures(_SYMBOLS), events, **_BASE_PARAMS, **_CONFIRM_PARAMS
    )


def _gen_walk(n: int, *, seed: int, drift: float, noise: float) -> list[dict[str, float]]:
    gen = _lcg(seed)
    prices = dict.fromkeys(_SYMBOLS, 100.0)
    out: list[dict[str, float]] = []
    for _ in range(n):
        for symbol in _SYMBOLS:
            prices[symbol] *= math.exp(drift + noise * next(gen))
        out.append(dict(prices))
    return out


def _gen_escalating(
    n: int, *, seed: int, drift: float, noise_start: float, noise_growth: float
) -> list[dict[str, float]]:
    gen = _lcg(seed)
    prices = dict.fromkeys(_SYMBOLS, 100.0)
    out: list[dict[str, float]] = []
    noise = noise_start
    for _ in range(n):
        for symbol in _SYMBOLS:
            prices[symbol] *= math.exp(drift + noise * next(gen))
        noise *= noise_growth
        out.append(dict(prices))
    return out


def _feed(strategy: Any, steps: list[dict[str, float]], *, start: int = 0) -> int:
    i = start
    for prices in steps:
        strategy.calculate_signals(_Window(_ts(i), prices))
        i += 1
    return i


def _entries(events: _Events) -> list[Any]:
    return [event for event in events.signals if event.signal_type in {"LONG", "SHORT"}]


def _warmup() -> list[dict[str, float]]:
    # Calibration window: moderate noise, no drift.  Long enough (> garch
    # fit minimum of 32 obs) to give the GARCH fit a stable long-run
    # variance baseline before any directional fixture is applied.
    return _gen_walk(40, seed=_WARM_SEED, drift=0.0, noise=_WARM_NOISE)


# --------------------------------------------------------------------------- #
# (a) THE GATE: deterministic non-redundancy vs the un-confirmed parent
# --------------------------------------------------------------------------- #
def test_non_redundant_vs_unconfirmed_parent_on_chop() -> None:
    """Identical chop input: parent flips bear-short, confirmed stays flat.

    The fixture drifts down at -0.6%/bar with noise at HALF the calibration
    amplitude -- breadth and the BTC benchmark clear the un-confirmed
    parent's bear vote decisively, but because the down-drift is gentle
    relative to the calibrated noise scale, the GARCH one-step variance
    forecast settles BELOW the long-run variance (a FALLING vol read, not
    RISING) -- so the confirmation-gated sleeve never concurs and never
    flips out of CHOP.
    """
    warmup = _warmup()
    chop = _gen_walk(25, seed=53, drift=-0.006, noise=_WARM_NOISE * 0.5)

    parent_events = _Events()
    parent = _parent(parent_events)
    _feed(parent, warmup + chop)
    parent_shorts = [e for e in parent_events.signals if e.signal_type == "SHORT"]
    assert parent_shorts, "expected the un-confirmed parent to flip bear-short on this chop tape"
    assert all((e.metadata or {}).get("regime") == "BEAR" for e in parent_shorts)
    assert parent.get_state()["regime"] == "BEAR"

    confirmed_events = _Events()
    confirmed = _confirmed(confirmed_events)
    regimes_seen: set[str] = set()
    for i, prices in enumerate(warmup + chop):
        confirmed.calculate_signals(_Window(_ts(i), prices))
        regimes_seen.add(confirmed.get_state()["regime"])
    assert _entries(confirmed_events) == [], "confirmed sleeve must not trade the unconfirmed vote"
    assert "BEAR" not in regimes_seen, f"confirmed sleeve must never flip bear here: {regimes_seen}"
    final_state = confirmed.get_state()
    assert final_state["regime"] == "CHOP"
    assert final_state["last_down_breadth"] >= _BASE_PARAMS["bear_breadth"], (
        "base bear vote must be genuinely cleared (breadth) for this to be a real "
        "confirmation-gate test, not a data-availability accident"
    )
    assert final_state["last_benchmark_return"] is not None
    assert final_state["last_benchmark_return"] <= -_BASE_PARAMS["benchmark_bear_threshold"], (
        "base bear vote must be genuinely cleared (benchmark) too"
    )
    assert final_state["last_vol_state"] != "RISING", final_state["last_vol_state"]


# --------------------------------------------------------------------------- #
# (b) sustained bull trend -> bull-long entries
# --------------------------------------------------------------------------- #
def test_sustained_bull_trend_enters_long_when_vol_confirms() -> None:
    warmup = _warmup()
    bull = _gen_walk(45, seed=57, drift=0.006, noise=_WARM_NOISE)

    events = _Events()
    strategy = _confirmed(events)
    _feed(strategy, warmup + bull)
    entries = _entries(events)
    assert entries, "expected LONG entries on a sustained, vol-confirmed uptrend"
    assert {e.signal_type for e in entries} == {"LONG"}
    assert all((e.metadata or {}).get("regime") == "BULL" for e in entries)
    state = strategy.get_state()
    assert state["regime"] == "BULL"
    assert state["last_vol_state"] == "FALLING", state["last_vol_state"]


# --------------------------------------------------------------------------- #
# (c) genuine bear (sustained decline + rising vol) -> proves the gate is
# not a dead gate: it CAN and DOES pass when vol genuinely confirms.
# --------------------------------------------------------------------------- #
def test_genuine_bear_with_rising_vol_enters_short() -> None:
    warmup = _warmup()
    bear = _gen_escalating(25, seed=54, drift=-0.006, noise_start=_WARM_NOISE, noise_growth=1.10)

    events = _Events()
    strategy = _confirmed(events)
    _feed(strategy, warmup + bear)
    entries = _entries(events)
    assert entries, "expected SHORT entries once vol genuinely confirms the bear vote"
    assert {e.signal_type for e in entries} == {"SHORT"}
    assert all((e.metadata or {}).get("regime") == "BEAR" for e in entries)
    state = strategy.get_state()
    assert state["regime"] == "BEAR"
    assert state["last_vol_state"] == "RISING", state["last_vol_state"]
    assert (state["last_vol_ratio"] or 0.0) >= _CONFIRM_PARAMS["confirm_bear_vol_ratio"]


# --------------------------------------------------------------------------- #
# (d) hysteresis: crossing the flip threshold briefly does not flip state
# --------------------------------------------------------------------------- #
def test_hysteresis_holds_bull_through_a_transient_breadth_dip() -> None:
    warmup = _warmup()
    bull = _gen_walk(45, seed=57, drift=0.006, noise=_WARM_NOISE)

    events = _Events()
    strategy = _confirmed(events)
    step = _feed(strategy, warmup + bull)
    pre_dip = strategy.get_state()
    assert pre_dip["regime"] == "BULL"
    assert pre_dip["last_up_breadth"] >= _BASE_PARAMS["bull_breadth"]

    # ONE bar: half the basket (incl. the benchmark) keeps rising, the other
    # half drops sharply.  Up-breadth falls to 0.5 -- BELOW the bull_breadth
    # (0.60) re-entry gate but ABOVE the looser exit_breadth (0.35) -- so the
    # sticky hysteresis branch, not a re-confirmed vote, must hold the state.
    last_prices = dict(bull[-1])
    dipped_symbols = {"ETH/USDT", "ADA/USDT", "XRP/USDT"}
    dip_prices = {
        symbol: (
            last_prices[symbol] * math.exp(-0.05)
            if symbol in dipped_symbols
            else last_prices[symbol] * math.exp(0.006 + _WARM_NOISE * 0.3)
        )
        for symbol in _SYMBOLS
    }
    strategy.calculate_signals(_Window(_ts(step), dip_prices))
    step += 1
    dip_state = strategy.get_state()
    assert dip_state["regime"] == "BULL", dip_state
    assert (
        _BASE_PARAMS["exit_breadth"] < dip_state["last_up_breadth"] < _BASE_PARAMS["bull_breadth"]
    )

    resume = _gen_walk(8, seed=91, drift=0.006, noise=_WARM_NOISE)
    base = dict(dip_prices)
    for row in resume:
        scaled = {symbol: base[symbol] * (row[symbol] / 100.0) for symbol in _SYMBOLS}
        strategy.calculate_signals(_Window(_ts(step), scaled))
        step += 1
        assert strategy.get_state()["regime"] == "BULL", "state flapped during resumed uptrend"


# --------------------------------------------------------------------------- #
# never-raise / graceful degradation
# --------------------------------------------------------------------------- #
def test_empty_window_and_degenerate_prices_are_safe() -> None:
    events = _Events()
    strategy = _confirmed(events)
    strategy.calculate_signals(_EmptyWindow())
    strategy.calculate_signals(_Window(_ts(0), dict.fromkeys(_SYMBOLS, 0.0)))
    assert events.signals == []


def test_below_min_symbols_never_raises_and_stays_inert() -> None:
    events = _Events()
    strategy = RegimeRouterConfirmedRotationStrategy(
        _BarsNoFeatures(_SYMBOLS[:3]),
        events,
        **{**_BASE_PARAMS, "min_symbols": 5},
        **_CONFIRM_PARAMS,
    )
    _feed(strategy, _gen_walk(60, seed=3, drift=0.01, noise=_WARM_NOISE))
    assert events.signals == []


def test_set_state_adversarial_payloads_never_raise() -> None:
    events = _Events()
    strategy = _confirmed(events)
    for payload in (
        None,
        {},
        {"regime": "not-a-regime", "tick": "abc"},
        {"last_up_breadth": float("nan"), "last_down_breadth": float("inf")},
        {"garch_omega": float("nan"), "garch_alpha": "x", "garch_beta": None},
        {"last_vol_state": 12345, "last_cycle_state": [1, 2, 3]},
        {"symbol_state": "garbage"},
        {"symbol_state": {"BTC/USDT": "garbage"}},
        {"symbol_state": {"BTC/USDT": {"closes": ["x", None, 1e200], "mode": 999}}},
    ):
        strategy.set_state(payload)  # must not raise
    # Strategy must remain usable afterward.
    _feed(strategy, _gen_walk(10, seed=5, drift=0.0, noise=_WARM_NOISE))


# --------------------------------------------------------------------------- #
# state roundtrip + determinism
# --------------------------------------------------------------------------- #
def test_state_roundtrip_preserves_regime_and_symbol_state() -> None:
    warmup = _warmup()
    bull = _gen_walk(45, seed=57, drift=0.006, noise=_WARM_NOISE)

    events = _Events()
    strategy = _confirmed(events)
    _feed(strategy, warmup + bull)
    snapshot = strategy.get_state()

    restored = _confirmed(_Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["regime"] == snapshot["regime"]
    assert again["tick"] == snapshot["tick"]
    assert again["garch_omega"] == snapshot["garch_omega"]
    assert again["garch_alpha"] == snapshot["garch_alpha"]
    assert again["garch_beta"] == snapshot["garch_beta"]
    assert again["last_vol_state"] == snapshot["last_vol_state"]
    assert (
        again["symbol_state"]["BTC/USDT"]["closes"]
        == snapshot["symbol_state"]["BTC/USDT"]["closes"]
    )


def test_determinism_same_input_gives_bit_identical_state() -> None:
    warmup = _warmup()
    chop = _gen_walk(25, seed=53, drift=-0.006, noise=_WARM_NOISE * 0.5)

    events_a = _Events()
    strategy_a = _confirmed(events_a)
    _feed(strategy_a, warmup + chop)

    events_b = _Events()
    strategy_b = _confirmed(events_b)
    _feed(strategy_b, warmup + chop)

    assert strategy_a.get_state() == strategy_b.get_state()
    assert len(events_a.signals) == len(events_b.signals)
    for left, right in zip(events_a.signals, events_b.signals, strict=True):
        assert left.signal_type == right.signal_type
        assert left.symbol == right.symbol
        assert left.metadata == right.metadata


def test_get_state_snapshot_is_independent_copy() -> None:
    # Guard against accidental aliasing between successive get_state() calls.
    warmup = _warmup()
    events = _Events()
    strategy = _confirmed(events)
    _feed(strategy, warmup)
    first = strategy.get_state()
    mutated = copy.deepcopy(first)
    mutated["symbol_state"]["BTC/USDT"]["closes"].append(999999.0)
    second = strategy.get_state()
    assert (
        second["symbol_state"]["BTC/USDT"]["closes"]
        != mutated["symbol_state"]["BTC/USDT"]["closes"]
    )
