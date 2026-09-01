"""Deterministic tests for the regime-adaptive disagreement variant (Lane MR2).

Imports ``RegimeAdaptiveDisagreementEnsembleStrategy`` DIRECTLY (this lane ships
without ``@register`` -- registration and the research_only tier hint are applied
atomically in a later, separate wave), so these tests carry NO registry/tier
assertions and NO candidate-wiring tests.

The load-bearing tests are the DIVERGENCE GATE (both directions, versus the base
``DisagreementGatedEnsembleStrategy`` imported directly): a high-dispersion input
on which the fixed-gate base BLOCKS while the variant's regime-WIDENED gate
ADMITS, and a calm-regime input on which the base ADMITS while the variant's
regime-TIGHTENED gate BLOCKS -- divergent action on identical input, both
directions.  If either direction fails to diverge the lane is redundant and must
be DROPPED.  A ``gate_sensitivity=0`` control proves the divergence is caused by
the regime gate (the variant then reproduces the base exactly on both inputs).

Also covers: monotone gate response to the dispersion-regime measure, run-twice
determinism, get_state/set_state roundtrip + adversarial set_state, and
never-raise degenerate inputs.  No backtest is run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

from lumina_quant.strategies.disagreement_ensemble_alpha_sleeves import (
    DisagreementGatedEnsembleStrategy,
)
from lumina_quant.strategies.regime_adaptive_disagreement_alpha_sleeves import (
    RegimeAdaptiveDisagreementEnsembleStrategy,
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
    ) -> None:
        self.time = time
        self.open = close
        self.high = high if high is not None else close * 1.0005
        self.low = low if low is not None else close * 0.9995
        self.close = close
        self.symbol = symbol
        self.volume = 1_000.0


def _ts(i: int, *, step_minutes: int = 30) -> str:
    return (_START + timedelta(minutes=step_minutes * i)).isoformat()


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield (state / 2**31 - 0.5) * 2.0


def _entries(events: _Events) -> list[Any]:
    return [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]


# Shared base sleeve params; the variant additionally takes the regime params.
_BASE_PARAMS: dict[str, Any] = {
    "tsmom_window": 5,
    "tsmom_lookback": 2,
    "tsmom_scale": 0.003,
    "error_window": 60,
    "return_deadband": 0.0,
    "disagreement_gate": 1.0,
    "entry_band": 0.05,
    "trail_atr_mult": 4.0,
    "atr_period": 5,
    "max_adds": 0,
    "vol_window": 10,
    "target_vol": 0.0,
    "max_hold_bars": 100000,
    "allow_short": True,
    "add_alloc_fraction": 0.0,
}
_REGIME_PARAMS: dict[str, Any] = {
    "regime_fast_window": 8,
    "regime_slow_window": 60,
    "gate_sensitivity": 2.0,
    "min_gate_factor": 0.05,
    "max_gate_factor": 12.0,
}


def _base(events: _Events, symbol: str, **over: Any) -> DisagreementGatedEnsembleStrategy:
    params = {**_BASE_PARAMS, **over}
    return DisagreementGatedEnsembleStrategy(_BarsNoFeatures([symbol]), events, **params)


def _variant(
    events: _Events, symbol: str, **over: Any
) -> RegimeAdaptiveDisagreementEnsembleStrategy:
    params = {**_BASE_PARAMS, **_REGIME_PARAMS, **over}
    return RegimeAdaptiveDisagreementEnsembleStrategy(_BarsNoFeatures([symbol]), events, **params)


def _feed(strategy: Any, symbol: str, series: list[tuple[float, float]]) -> None:
    for i, (price, rng) in enumerate(series):
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )


# --------------------------------------------------------------------------- #
# Divergence-gate input constructions (deterministic; no RNG)
# --------------------------------------------------------------------------- #
def _turbulent_series() -> list[tuple[float, float]]:
    """Calm low-vol uptrend, then a high-vol upward burst (turbulent regime).

    The persistently sharp reversion component keeps the cross-component
    disagreement above the fixed gate on EVERY bar (the fixed-gate base blocks
    throughout); the terminal high-vol burst lifts the fast/slow realized-vol
    ratio well above 1, widening the variant's gate above that disagreement so
    the variant admits during the burst.
    """
    series: list[tuple[float, float]] = []
    price = 100.0
    for _ in range(90):  # calm low-vol uptrend -> low slow-window vol baseline
        price *= 1.004
        series.append((price, price * 0.0005))
    for j in range(12):  # high-vol upward burst -> fast-window vol spikes
        price *= 1.004 * (1.0 + (0.05 if j % 2 == 0 else -0.03))
        series.append((price, price * 0.03))
    return series


def _turbulent_overrides() -> dict[str, Any]:
    # Sharp reversion (reversion_scale=3.0) keeps disagreement above the fixed
    # gate on every bar so the base never admits.
    return {
        "reversion_scale": 3.0,
        "reversion_window": 40,
        "donchian_window": 40,
        "efficiency_period": 40,
    }


def _calm_series() -> list[tuple[float, float]]:
    """High-vol directionless chop, then a clean calm uptrend (calm regime).

    The symmetric high-vol chop keeps disagreement above the gate for BOTH
    sleeves (both blocked in the prefix) while building an elevated slow-window
    vol baseline; the terminal calm uptrend drops the fast/slow vol ratio well
    below 1.  In that calm tail the base admits (disagreement below the fixed
    gate, composite beyond the entry band) but the variant's regime-TIGHTENED
    gate blocks the same bar.
    """
    series: list[tuple[float, float]] = []
    price = 100.0
    amp = 0.06
    for j in range(60):  # exact mean-reverting high-vol chop (net ~0 drift)
        price *= (1.0 + amp) if j % 2 == 0 else (1.0 / (1.0 + amp))
        series.append((price, price * amp * 0.5))
    for _ in range(30):  # calm uptrend tail -> fast-window vol collapses
        price *= 1.004
        series.append((price, price * 0.0005))
    return series


def _calm_overrides() -> dict[str, Any]:
    return {
        "reversion_scale": 40.0,
        "reversion_window": 20,
        "donchian_window": 20,
        "efficiency_period": 20,
    }


# --------------------------------------------------------------------------- #
# DIVERGENCE GATE (both directions vs base M1) -- fail => DROP the lane
# --------------------------------------------------------------------------- #
def test_high_dispersion_regime_variant_admits_base_blocks() -> None:
    symbol = "BTC/USDT"
    series = _turbulent_series()
    over = _turbulent_overrides()

    base_events = _Events()
    _feed(_base(base_events, symbol, **over), symbol, series)
    variant_events = _Events()
    _feed(_variant(variant_events, symbol, **over), symbol, series)

    base_entries = _entries(base_events)
    variant_entries = _entries(variant_events)

    # Base's fixed gate blocks the entire persistently-disagreeing series.
    assert base_entries == [], (
        f"expected fixed-gate base to BLOCK; got {[e.signal_type for e in base_entries]}"
    )
    # The variant's turbulence-widened gate admits during the burst -> divergent.
    assert variant_entries, (
        "expected the regime-widened variant to ADMIT (divergence-fail => DROP lane)"
    )
    assert variant_entries[0].signal_type == "LONG", variant_entries[0].signal_type
    metadata = variant_entries[0].metadata or {}
    assert metadata.get("regime_ratio") is not None and metadata["regime_ratio"] > 1.0, metadata
    assert metadata.get("effective_gate") is not None and metadata["effective_gate"] > 1.0, metadata


def test_calm_regime_variant_blocks_base_admits() -> None:
    symbol = "ETH/USDT"
    series = _calm_series()
    over = _calm_overrides()

    base_events = _Events()
    _feed(_base(base_events, symbol, **over), symbol, series)
    variant_events = _Events()
    _feed(_variant(variant_events, symbol, **over), symbol, series)

    base_entries = _entries(base_events)
    variant_entries = _entries(variant_events)

    # Base admits (and trades) once the calm tail brings disagreement below the
    # fixed gate with a strong composite.
    assert base_entries, "expected the fixed-gate base to ADMIT in the calm tail"
    assert base_entries[0].signal_type == "LONG", base_entries[0].signal_type
    # The variant's calm-tightened gate blocks that same trade -> divergent.
    assert variant_entries == [], (
        "expected the regime-tightened variant to BLOCK "
        f"(divergence-fail => DROP lane); got {[e.signal_type for e in variant_entries]}"
    )


def test_neutral_sensitivity_reproduces_base_on_both_inputs() -> None:
    # gate_sensitivity=0 pins the multiplier to 1.0 for every regime ratio, so
    # the variant must reproduce the fixed-gate base EXACTLY -- proving the
    # divergence above is caused by the regime gate, not any other difference.
    for series, over in (
        (_turbulent_series(), _turbulent_overrides()),
        (_calm_series(), _calm_overrides()),
    ):
        symbol = "SOL/USDT"
        base_events = _Events()
        _feed(_base(base_events, symbol, **over), symbol, series)
        variant_events = _Events()
        _feed(_variant(variant_events, symbol, gate_sensitivity=0.0, **over), symbol, series)
        base_stream = [(e.signal_type, e.symbol, e.price) for e in base_events.signals]
        variant_stream = [(e.signal_type, e.symbol, e.price) for e in variant_events.signals]
        assert base_stream == variant_stream


# --------------------------------------------------------------------------- #
# Gate-adaptivity unit test: gate responds monotonically to the regime measure
# --------------------------------------------------------------------------- #
def test_gate_multiplier_monotonic_in_regime() -> None:
    strategy = _variant(
        _Events(), "BTC/USDT", gate_sensitivity=2.0, min_gate_factor=0.05, max_gate_factor=12.0
    )
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 6.0, 20.0]
    multipliers = [strategy._regime_gate_multiplier(r) for r in ratios]
    # Monotone non-decreasing in the regime ratio.
    for lo, hi in pairwise(multipliers):
        assert hi >= lo, (lo, hi)
    # Base-identical at ratio 1.0 and when the ratio is unavailable.
    assert strategy._regime_gate_multiplier(1.0) == 1.0
    assert strategy._regime_gate_multiplier(None) == 1.0
    assert strategy._regime_gate_multiplier(float("nan")) == 1.0
    # Turbulent widens above the base gate; calm tightens below it.
    assert strategy._regime_gate_multiplier(1.5) > 1.0
    assert strategy._regime_gate_multiplier(0.5) < 1.0
    # Strictly increasing across the unclamped middle band.
    assert strategy._regime_gate_multiplier(1.5) < strategy._regime_gate_multiplier(2.0)
    # Effective gate scales the base gate by the multiplier.
    assert strategy.disagreement_gate == 1.0


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_determinism_two_runs_identical() -> None:
    symbol = "DOGE/USDT"
    rng = _lcg(20260709)
    price = 100.0
    series: list[tuple[float, float]] = []
    for _ in range(120):
        drift = 0.002 + 0.02 * next(rng)
        price = max(1.0, price * (1.0 + drift))
        series.append((price, price * (0.001 + 0.02 * abs(next(rng)))))

    def _run() -> tuple[list[Any], dict[str, Any]]:
        events = _Events()
        strategy = _variant(events, symbol)
        _feed(strategy, symbol, series)
        signals = [(e.signal_type, e.symbol, e.price, e.metadata) for e in events.signals]
        return signals, strategy.get_state()

    signals_a, state_a = _run()
    signals_b, state_b = _run()
    assert signals_a == signals_b
    assert state_a == state_b


# --------------------------------------------------------------------------- #
# State roundtrip + adversarial set_state
# --------------------------------------------------------------------------- #
def test_state_roundtrip() -> None:
    symbol = "XRP/USDT"
    events = _Events()
    strategy = _variant(events, symbol)
    _feed(strategy, symbol, _turbulent_series())
    snapshot = strategy.get_state()
    restored = _variant(_Events(), symbol)
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["ensemble_scores"][symbol] == snapshot["ensemble_scores"][symbol]
    assert again["ensemble_history"][symbol] == snapshot["ensemble_history"][symbol]
    assert again["symbol_state"][symbol] == snapshot["symbol_state"][symbol]


def test_adversarial_set_state_never_raises() -> None:
    symbol = "BTC/USDT"
    strategy = _variant(_Events(), symbol)
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
            "ensemble_scores": {symbol: [1.0, 2.0]},
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
# None / empty-window / degenerate-close safety
# --------------------------------------------------------------------------- #
def test_degenerate_inputs_never_raise() -> None:
    symbol = "ADA/USDT"
    events = _Events()
    strategy = _variant(events, symbol)

    class _EmptyWindow:
        type = "MARKET_WINDOW"
        bars_1s: list[Any] = []

    empty = _EmptyWindow()
    empty.symbol = symbol
    strategy.calculate_signals(empty)  # must not raise
    strategy.calculate_signals(_Market(_ts(0), symbol, 0.0))  # degenerate close
    strategy.calculate_signals(_Market(_ts(1), symbol, -5.0))  # negative close
    for i in range(3):  # far fewer bars than any component/regime window
        strategy.calculate_signals(_Market(_ts(2 + i), symbol, 100.0 + i))
    assert _entries(events) == []
    # Regime ratio and effective gate stay well-defined (never-raise) on short history.
    assert strategy._regime_ratio(symbol) is None
    assert strategy._effective_disagreement_gate(symbol) == strategy.disagreement_gate
