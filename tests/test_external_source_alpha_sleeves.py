"""Deterministic tests for the external-source absorption rider sleeves.

Covers registry presence of the three new classes; that
``SpectralCycleRiderStrategy`` ENTERS on a clean synthetic price cycle (SHORT
just past the crest, LONG just past the trough) and does NOT enter on a flat
series; that ``AdfGatedReversionRiderStrategy`` ENTERS on a seeded
mean-reverting (AR) log price and does NOT enter on a residual-free
deterministic trend (the ADF gate blocks); that
``GarchInnovationRiderStrategy`` ENTERS on a conditional-scale shock after a
calm seeded window, mirrors SHORT on a negative shock, and is SUPPRESSED by a
tight jump-exhaustion gate; graceful no-history / empty-window / degenerate
close behaviour; get_state/set_state roundtrips; and candidate wiring (all
single-asset, allowed timeframes, cadence >= 1800 from metadata).  All
fixtures are synthetic and seeded via a hand-rolled LCG.  No backtest is run.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.garch_innovation_alpha_sleeves import GarchInnovationRiderStrategy
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategies.spectral_cycle_alpha_sleeves import SpectralCycleRiderStrategy
from lumina_quant.strategies.stationarity_reversion_alpha_sleeves import (
    AdfGatedReversionRiderStrategy,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_STRATEGIES = (
    "SpectralCycleRiderStrategy",
    "AdfGatedReversionRiderStrategy",
    "GarchInnovationRiderStrategy",
)

_ALLOWED_TIMEFRAMES = {"30m", "1h", "4h", "1d"}
_FORBIDDEN_TIMEFRAMES = {"1s", "1m", "5m", "15m"}

_CRYPTO_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "ADA/USDT",
    "XRP/USDT",
]

_START = datetime(2026, 1, 1, tzinfo=UTC)
_TWO_PI = 2.0 * math.pi


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


def _entries(events: _Events) -> list[Any]:
    return [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield (state / 2**31 - 0.5) * 2.0


# --------------------------------------------------------------------------- #
# Registry presence
# --------------------------------------------------------------------------- #
def test_new_sleeves_registered() -> None:
    names = set(get_strategy_names())
    for cls in NEW_STRATEGIES:
        assert cls in names, cls
        assert get_strategy_param_schema(cls), cls
        assert get_default_strategy_params(cls), cls


# --------------------------------------------------------------------------- #
# SpectralCycleRider
# --------------------------------------------------------------------------- #
def _spectral_strategy(bars: Any, events: _Events) -> SpectralCycleRiderStrategy:
    return SpectralCycleRiderStrategy(
        bars,
        events,
        cycle_window=64,
        min_period=8,
        max_period=24,
        min_purity=0.15,
        entry_phase_band=0.20,
        min_amplitude_frac=0.0,
        trail_atr_mult=6.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _cycle_price(i: int, *, period: float = 16.0, shift: float = 0.0) -> float:
    return 100.0 * math.exp(0.03 * math.cos(_TWO_PI * (i - shift) / period))


def test_spectral_enters_short_after_crest() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _spectral_strategy(_BarsNoFeatures([symbol]), events)
    # cos starting at the crest: the first phase-band hit is just PAST the
    # crest, so the first entry must be a SHORT into the downswing.
    for i in range(200):
        strategy.calculate_signals(_Market(_ts(i), symbol, _cycle_price(i)))
    entries = _entries(events)
    assert entries, "expected entries on a clean 16-bar cycle"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type
    metadata = entries[0].metadata or {}
    assert metadata.get("cycle_period") == 16.0, metadata
    assert (metadata.get("cycle_purity") or 0.0) >= 0.15, metadata


def test_spectral_enters_long_after_trough() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _spectral_strategy(_BarsNoFeatures([symbol]), events)
    # Shift by half a period so the window ends just past a TROUGH instead.
    for i in range(200):
        strategy.calculate_signals(_Market(_ts(i), symbol, _cycle_price(i, shift=8.0)))
    entries = _entries(events)
    assert entries, "expected entries on the shifted cycle"
    assert entries[0].signal_type == "LONG", entries[0].signal_type


def test_spectral_no_entry_on_flat_series() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _spectral_strategy(_BarsNoFeatures([symbol]), events)
    for i in range(200):
        strategy.calculate_signals(_Market(_ts(i), symbol, 100.0))
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# AdfGatedReversionRider
# --------------------------------------------------------------------------- #
def _adf_strategy(bars: Any, events: _Events) -> AdfGatedReversionRiderStrategy:
    return AdfGatedReversionRiderStrategy(
        bars,
        events,
        stat_window=64,
        adf_lags=1,
        adf_significance="5%",
        zscore_entry=1.2,
        max_half_life_bars=32,
        trail_atr_mult=6.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def test_adf_enters_on_mean_reverting_series() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _adf_strategy(_BarsNoFeatures([symbol]), events)
    gen = _lcg(11)
    x = 0.0
    for i in range(300):
        x = 0.7 * x + 0.02 * next(gen)
        strategy.calculate_signals(_Market(_ts(i), symbol, 100.0 * math.exp(x)))
    entries = _entries(events)
    assert entries, "expected reversion entries on a stationary AR log price"
    metadata = entries[0].metadata or {}
    critical = -2.86
    assert (metadata.get("adf_t_stat") or 0.0) < critical, metadata
    z = metadata.get("reversion_z")
    assert z is not None and abs(z) >= 1.2, metadata
    # The fade direction opposes the z-score sign.
    expected = "LONG" if z <= -1.2 else "SHORT"
    assert entries[0].signal_type == expected, (entries[0].signal_type, z)


def test_adf_gate_blocks_deterministic_trend() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _adf_strategy(_BarsNoFeatures([symbol]), events)
    # A residual-free exponential trend: the ADF regression is degenerate and
    # the half-life undefined, so the stationarity gate must block every bar
    # even though the z-score itself runs to extremes.
    price = 100.0
    for i in range(300):
        price *= 1.003
        strategy.calculate_signals(_Market(_ts(i), symbol, price))
    assert _entries(events) == []


def test_adf_no_history_graceful() -> None:
    symbol = "TRX/USDT"
    events = _Events()
    strategy = _adf_strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for i in range(10):  # < stat_window
        price *= 1.002
        strategy.calculate_signals(_Market(_ts(i), symbol, price))
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# GarchInnovationRider
# --------------------------------------------------------------------------- #
def _garch_strategy(
    bars: Any, events: _Events, *, jump_mult: float = 50.0
) -> GarchInnovationRiderStrategy:
    return GarchInnovationRiderStrategy(
        bars,
        events,
        garch_window=64,
        refit_every=8,
        band_k=2.0,
        jump_window=64,
        jump_quantile=0.99,
        jump_mult=jump_mult,
        trail_atr_mult=6.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _feed_calm_then_shock(
    strategy: GarchInnovationRiderStrategy, symbol: str, *, shock: float
) -> None:
    gen = _lcg(31)
    price = 100.0
    for i in range(120):
        price *= math.exp(0.002 * next(gen))
        strategy.calculate_signals(_Market(_ts(i), symbol, price))
    price *= math.exp(shock)
    strategy.calculate_signals(_Market(_ts(120), symbol, price))


def test_garch_enters_long_on_conditional_shock() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _garch_strategy(_BarsNoFeatures([symbol]), events)
    _feed_calm_then_shock(strategy, symbol, shock=0.02)
    entries = _entries(events)
    assert entries, "expected a LONG entry on a positive conditional-scale shock"
    assert entries[-1].signal_type == "LONG", entries[-1].signal_type
    metadata = entries[-1].metadata or {}
    z = metadata.get("garch_innovation_z")
    sigma = metadata.get("garch_sigma_forecast")
    assert z is not None and z >= 2.0, metadata
    assert sigma, metadata
    # Metadata self-consistency: the emitted sigma must be the one that
    # STANDARDIZED the entry return, so z * sigma reproduces the log return.
    assert abs(z * sigma - 0.02) < 1e-9, metadata


def test_garch_enters_short_on_negative_shock() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _garch_strategy(_BarsNoFeatures([symbol]), events)
    _feed_calm_then_shock(strategy, symbol, shock=-0.02)
    entries = _entries(events)
    assert entries, "expected a SHORT entry on a negative conditional-scale shock"
    assert entries[-1].signal_type == "SHORT", entries[-1].signal_type


def test_garch_jump_gate_suppresses_blowoff() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _garch_strategy(_BarsNoFeatures([symbol]), events, jump_mult=1.0)
    # Same calm-then-shock tape, but the tight jump gate flags the shock bar as
    # a blow-off print (far beyond the trailing |return| quantile) -> no entry.
    _feed_calm_then_shock(strategy, symbol, shock=0.02)
    assert _entries(events) == []


def test_garch_no_history_graceful() -> None:
    symbol = "TRX/USDT"
    events = _Events()
    strategy = _garch_strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for i in range(5):  # < garch fit minimum
        price *= 1.004
        strategy.calculate_signals(_Market(_ts(i), symbol, price))
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# None / empty-window safety + state roundtrips
# --------------------------------------------------------------------------- #
def test_sleeves_none_and_empty_window_safe() -> None:
    symbol = "BTC/USDT"
    for factory in (_spectral_strategy, _adf_strategy, _garch_strategy):
        events = _Events()
        strategy = factory(_BarsNoFeatures([symbol]), events)

        class _EmptyWindow:
            type = "MARKET_WINDOW"
            bars_1s: list[Any] = []

        empty = _EmptyWindow()
        empty.symbol = symbol
        strategy.calculate_signals(empty)  # must not raise
        strategy.calculate_signals(_Market(_ts(0), symbol, 0.0))  # degenerate close
        assert _entries(events) == []


def test_spectral_state_roundtrip() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _spectral_strategy(_BarsNoFeatures([symbol]), events)
    for i in range(80):
        strategy.calculate_signals(_Market(_ts(i), symbol, _cycle_price(i)))
    snapshot = strategy.get_state()
    restored = _spectral_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["symbol_state"][symbol]["closes"] == snapshot["symbol_state"][symbol]["closes"]


def test_garch_set_state_adversarial_payloads_never_raise() -> None:
    symbol = "BTC/USDT"
    for payload in (
        {"bars_since_refit": "abc"},
        {"bars_since_refit": float("nan")},
        {"bars_since_refit": float("inf")},
        {"bars_since_refit": [1, 2]},
        {"returns": ["x", None, 1e200], "params": "garbage"},
        {"params": [1.0], "sigma_sq_forecast": "nope", "last_z": object()},
    ):
        strategy = _garch_strategy(_BarsNoFeatures([symbol]), _Events())
        strategy.set_state({"garch_state": {symbol: payload}})  # must not raise
        # The very next bars must also survive whatever was restored.
        price = 100.0
        for i in range(3):
            price *= 1.001
            strategy.calculate_signals(_Market(_ts(i), symbol, price))


def test_garch_state_roundtrip() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _garch_strategy(_BarsNoFeatures([symbol]), events)
    gen = _lcg(7)
    price = 100.0
    for i in range(100):
        price *= math.exp(0.003 * next(gen))
        strategy.calculate_signals(_Market(_ts(i), symbol, price))
    snapshot = strategy.get_state()
    restored = _garch_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["garch_state"][symbol] == snapshot["garch_state"][symbol]


# --------------------------------------------------------------------------- #
# Candidate-admission wiring
# --------------------------------------------------------------------------- #
def _candidates_for(strategy_class: str) -> list[Any]:
    timeframes = ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    candidates = build_binance_futures_candidates(timeframes=timeframes, symbols=_CRYPTO_UNIVERSE)
    return [c for c in candidates if c.strategy_class == strategy_class]


def test_new_sleeves_wired_single_asset_and_only_ge_30m() -> None:
    for strategy_class in NEW_STRATEGIES:
        rows = _candidates_for(strategy_class)
        assert rows, f"expected candidates for {strategy_class}"
        tfs = {c.timeframe for c in rows}
        assert tfs <= _ALLOWED_TIMEFRAMES, (strategy_class, tfs)
        assert not (tfs & _FORBIDDEN_TIMEFRAMES), (strategy_class, tfs)
        for c in rows:
            assert candidate_mix_type(c.to_dict()) == "single", c.name
            assert len(c.symbols) == 1, c.symbols
            assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata
