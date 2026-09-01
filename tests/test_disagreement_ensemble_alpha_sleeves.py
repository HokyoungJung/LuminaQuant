"""Deterministic tests for the disagreement-gated ensemble sleeve (Lane M1).

Imports ``DisagreementGatedEnsembleStrategy`` DIRECTLY -- this lane ships
without the ``@register`` decorator (live-safety: registration + the
research_only tier hint are applied atomically in a later, separate wave), so
these tests deliberately carry NO registry/tier assertions and NO
candidate-wiring tests.

Covers: a clean consensus uptrend enters LONG with composite/disagreement/
weights metadata present; the mirror clean consensus downtrend enters SHORT;
constructing the four components into conflict (equally strong trend vs.
reversion, so the disagreement coefficient exceeds the gate) blocks entry; a
flat series blocks entry (the disagreement coefficient is undefined at a
near-zero consensus mean, treated conservatively as blocked); no-history/
empty-window/degenerate-close inputs never raise; get_state/set_state
roundtrip equality; adversarial set_state payloads (strings/NaN/lists/wrong
types) never raise; and determinism (two identical runs over an LCG-seeded
noisy series produce bit-identical signal streams and state).  No backtest is
run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.disagreement_ensemble_alpha_sleeves import (
    DisagreementGatedEnsembleStrategy,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)


class _BarsNoFeatures:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Events:
    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _Market:
    type = "MARKET"

    def __init__(
        self,
        time: str,
        symbol: str,
        close: float,
        *,
        high: float | None = None,
        low: float | None = None,
        open_: float | None = None,
    ) -> None:
        self.time = time
        self.open = open_ if open_ is not None else close
        self.high = high if high is not None else close * 1.0005
        self.low = low if low is not None else close * 0.9995
        self.close = close
        self.symbol = symbol
        self.volume = 1_000.0


def _ts(i: int, *, step_minutes: int = 30) -> str:
    return (_START + timedelta(minutes=step_minutes * i)).isoformat()


def _entries(events: _Events) -> list[Any]:
    return [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield (state / 2**31 - 0.5) * 2.0


def _ensemble_strategy(
    bars: Any, events: _Events, *, reversion_scale: float = 40.0
) -> DisagreementGatedEnsembleStrategy:
    # Windows/scales chosen so a clean, steady trend (all four components
    # saturating toward the same sign) sits below the disagreement gate, while
    # sharpening the reversion component (smaller ``reversion_scale``) against
    # the SAME trend pushes disagreement above the gate -- the "equally
    # strong trend vs. reversion" conflict construction.
    return DisagreementGatedEnsembleStrategy(
        bars,
        events,
        tsmom_window=5,
        tsmom_lookback=2,
        tsmom_scale=0.003,
        reversion_window=40,
        reversion_scale=reversion_scale,
        donchian_window=40,
        efficiency_period=40,
        error_window=60,
        return_deadband=0.0,
        disagreement_gate=1.0,
        entry_band=0.15,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _uptrend_prices(n: int, *, growth: float = 1.004) -> list[float]:
    price = 100.0
    prices = []
    for _ in range(n):
        price *= growth
        prices.append(price)
    return prices


def _downtrend_prices(n: int, *, decay: float = 0.996) -> list[float]:
    price = 100.0
    prices = []
    for _ in range(n):
        price *= decay
        prices.append(price)
    return prices


def _feed(strategy: DisagreementGatedEnsembleStrategy, symbol: str, prices: list[float]) -> None:
    for i, price in enumerate(prices):
        rng = price * 0.0005
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )


# --------------------------------------------------------------------------- #
# Consensus -> entry
# --------------------------------------------------------------------------- #
def test_consensus_uptrend_enters_long() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _uptrend_prices(60))
    entries = _entries(events)
    assert entries, "expected a LONG entry once the four components reach consensus"
    assert entries[0].signal_type == "LONG", entries[0].signal_type
    metadata = entries[0].metadata or {}
    assert metadata.get("composite") is not None and metadata["composite"] > 0.0, metadata
    assert metadata.get("disagreement") is not None, metadata
    weights = metadata.get("weights")
    assert weights, metadata
    assert abs(sum(weights.values()) - 1.0) < 1e-9, weights
    assert metadata.get("component_scores"), metadata


def test_consensus_downtrend_enters_short() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _downtrend_prices(60))
    entries = _entries(events)
    assert entries, "expected a SHORT entry once the four components reach consensus"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type
    metadata = entries[0].metadata or {}
    assert metadata.get("composite") is not None and metadata["composite"] < 0.0, metadata
    weights = metadata.get("weights")
    assert weights and abs(sum(weights.values()) - 1.0) < 1e-9, weights


# --------------------------------------------------------------------------- #
# Conflict -> gated (no entry)
# --------------------------------------------------------------------------- #
def test_conflicting_components_block_entry() -> None:
    # Same clean uptrend as the consensus case, but a sharpened reversion
    # component (small ``reversion_scale``) now opposes the trend components
    # with comparable strength, pushing the disagreement coefficient above the
    # gate for every bar -- no entry should ever fire.
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), events, reversion_scale=2.0)
    _feed(strategy, symbol, _uptrend_prices(60))
    assert _entries(events) == []


def test_conflicting_components_block_entry_downtrend() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), events, reversion_scale=2.0)
    _feed(strategy, symbol, _downtrend_prices(60))
    assert _entries(events) == []


def test_flat_series_no_entry() -> None:
    symbol = "TRX/USDT"
    events = _Events()
    strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), events)
    # A constant price drives all four component scores to exactly 0 -> the
    # disagreement coefficient's mean is 0 -> undefined (None) -> blocked.
    for i in range(60):
        strategy.calculate_signals(_Market(_ts(i), symbol, 100.0, high=100.05, low=99.95))
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# None / empty-window / degenerate-close safety
# --------------------------------------------------------------------------- #
def test_no_history_empty_window_degenerate_close_never_raise() -> None:
    symbol = "ADA/USDT"
    events = _Events()
    strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), events)

    class _EmptyWindow:
        type = "MARKET_WINDOW"
        bars_1s: list[Any] = []

    empty = _EmptyWindow()
    empty.symbol = symbol
    strategy.calculate_signals(empty)  # must not raise
    strategy.calculate_signals(_Market(_ts(0), symbol, 0.0))  # degenerate close, no raise
    strategy.calculate_signals(_Market(_ts(1), symbol, -5.0))  # negative close, no raise
    for i in range(3):  # far fewer bars than any component window
        strategy.calculate_signals(_Market(_ts(2 + i), symbol, 100.0 + i))
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# State roundtrip + adversarial set_state
# --------------------------------------------------------------------------- #
def test_state_roundtrip() -> None:
    symbol = "XRP/USDT"
    events = _Events()
    strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), events)
    _feed(strategy, symbol, _uptrend_prices(50))
    snapshot = strategy.get_state()
    restored = _ensemble_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["ensemble_scores"][symbol] == snapshot["ensemble_scores"][symbol]
    assert again["ensemble_history"][symbol] == snapshot["ensemble_history"][symbol]
    assert again["symbol_state"][symbol] == snapshot["symbol_state"][symbol]


def test_adversarial_set_state_never_raises() -> None:
    symbol = "BTC/USDT"
    strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), _Events())
    strategy.set_state(None)
    strategy.set_state({})
    strategy.set_state({"ensemble_scores": "garbage", "ensemble_history": 123})
    strategy.set_state({"ensemble_scores": {symbol: "abc"}, "ensemble_history": {symbol: "nope"}})
    strategy.set_state(
        {
            "ensemble_scores": {symbol: [float("nan"), "x", None, 1.0]},
            "ensemble_history": {symbol: {"realized": "nan", "predicted": "not-a-list"}},
        }
    )
    strategy.set_state(
        {
            "ensemble_scores": {symbol: [1.0, 2.0]},  # wrong length -> ignored
            "ensemble_history": {
                symbol: {
                    "realized": [1.0, float("nan"), "x", None],
                    "predicted": [[1.0, "y"], None, 5, [0.1]],
                }
            },
        }
    )
    strategy.set_state({"ensemble_scores": {"NOPE/USDT": [1.0, 2.0, 3.0, 4.0]}})
    strategy.set_state({"symbol_state": {symbol: {"closes": [float("nan"), "x", None]}}})


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_two_runs_identical() -> None:
    symbol = "DOGE/USDT"
    rng = _lcg(20260703)
    price = 100.0
    prices: list[float] = []
    for _ in range(90):
        drift = 0.0025 + 0.01 * next(rng)
        price = max(1.0, price * (1.0 + drift))
        prices.append(price)

    def _run() -> tuple[list[Any], dict[str, Any]]:
        events = _Events()
        strategy = _ensemble_strategy(_BarsNoFeatures([symbol]), events)
        _feed(strategy, symbol, prices)
        signals = [(e.signal_type, e.symbol, e.price, e.metadata) for e in events.signals]
        return signals, strategy.get_state()

    signals_a, state_a = _run()
    signals_b, state_b = _run()
    assert signals_a == signals_b
    assert state_a == state_b
