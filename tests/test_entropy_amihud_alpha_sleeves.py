"""Deterministic tests for the permutation-entropy + Amihud-illiquidity sleeves.

Covers registry presence; the permutation-entropy helper (low for a monotonic
path, high for a scrambled one); that ``PermutationEntropyTrendRiderStrategy``
ENTERS on a structured (low-entropy) uptrend and is SUPPRESSED on a high-entropy
path even when it nets upward; that ``AmihudIlliquidityMomentumRiderStrategy``
ENTERS when the Amihud illiquidity is elevated vs its rolling median (+ trend) and
is SUPPRESSED when it is not; candidate wiring (both single-asset, allowed
timeframes, cadence >= 1800 from metadata); None/empty-window safety; and
get_state/set_state roundtrip.  No backtest is run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.entropy_amihud_alpha_sleeves import (
    AmihudIlliquidityMomentumRiderStrategy,
    PermutationEntropyTrendRiderStrategy,
    _permutation_entropy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_STRATEGIES = (
    "PermutationEntropyTrendRiderStrategy",
    "AmihudIlliquidityMomentumRiderStrategy",
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
        volume: float = 1_000_000.0,
    ) -> None:
        self.time = time
        self.open = close
        self.high = high if high is not None else close * 1.0005
        self.low = low if low is not None else close * 0.9995
        self.close = close
        self.symbol = symbol
        self.volume = volume


def _ts(i: int, *, step_minutes: int = 30) -> str:
    return (_START + timedelta(minutes=step_minutes * i)).isoformat()


def _entries(events: _Events) -> list[Any]:
    return [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]


# --------------------------------------------------------------------------- #
# Registry presence + permutation-entropy helper
# --------------------------------------------------------------------------- #
def test_new_sleeves_registered() -> None:
    names = set(get_strategy_names())
    for cls in NEW_STRATEGIES:
        assert cls in names, cls
        assert get_strategy_param_schema(cls), cls
        assert get_default_strategy_params(cls), cls


def test_permutation_entropy_helper() -> None:
    monotonic = [float(i) for i in range(64)]
    pe_low = _permutation_entropy(monotonic, dim=3)
    assert pe_low is not None and pe_low < 0.1, pe_low
    # A scrambled, order-diverse path has high normalized entropy.
    scramble = [0.0, 3.0, 1.0, 4.0, 2.0, 5.0, 1.5, 4.5, 0.5, 3.5, 2.5, 5.5] * 6
    pe_high = _permutation_entropy(scramble, dim=3)
    assert pe_high is not None and pe_high > 0.7, pe_high
    assert _permutation_entropy([1.0, 2.0], dim=3) is None  # too few points


# --------------------------------------------------------------------------- #
# PermutationEntropyTrendRider
# --------------------------------------------------------------------------- #
def _pe_strategy(bars: Any, events: _Events, *, pe_threshold: float = 0.85) -> Any:
    return PermutationEntropyTrendRiderStrategy(
        bars,
        events,
        pe_dim=3,
        pe_window=48,
        pe_threshold=pe_threshold,
        trend_lookback=10,
        trend_ma_window=10,
        min_trend_roc=0.0,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def test_pe_enters_on_structured_uptrend() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _pe_strategy(_BarsNoFeatures([symbol]), events)
    # A smooth, monotonic uptrend has near-zero permutation entropy (one dominant
    # ordinal pattern) and a positive trend -> a LONG ride is armed.
    price = 100.0
    for i in range(80):
        price *= 1.003
        rng = price * 0.0005
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )
    entries = _entries(events)
    assert entries, "expected a LONG entry on a low-entropy structured uptrend"
    assert entries[0].signal_type == "LONG", entries[0].signal_type
    pe = (entries[0].metadata or {}).get("permutation_entropy")
    assert pe is not None and pe < 0.85, entries[0].metadata


def test_pe_suppressed_on_high_entropy_path() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    # Low threshold so only a genuinely structured regime qualifies; a scrambled
    # (high-entropy) path that still nets upward must NOT trigger an entry.
    strategy = _pe_strategy(_BarsNoFeatures([symbol]), events, pe_threshold=0.50)
    zig = [0.0, 0.05, -0.05, 0.03, -0.04, 0.06, -0.03, 0.04, -0.06, 0.02]
    price = 100.0
    for i in range(120):
        drift = 1.0015  # mild net upward drift
        price *= drift
        wobble = 1.0 + zig[i % len(zig)]
        c = price * wobble
        rng = c * 0.0005
        strategy.calculate_signals(_Market(_ts(i), symbol, c, high=c + rng, low=c - rng))
    assert _entries(events) == [], _entries(events)


# --------------------------------------------------------------------------- #
# AmihudIlliquidityMomentumRider
# --------------------------------------------------------------------------- #
def _amihud_strategy(bars: Any, events: _Events) -> AmihudIlliquidityMomentumRiderStrategy:
    return AmihudIlliquidityMomentumRiderStrategy(
        bars,
        events,
        amihud_window=10,
        amihud_history_window=24,
        illiquidity_rel=1.0,
        trend_lookback=10,
        trend_ma_window=10,
        min_trend_roc=0.0,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _feed_amihud(strategy: Any, symbol: str, vols: list[float]) -> None:
    """Feed a steady uptrend with a per-bar volume schedule (drives Amihud)."""
    price = 100.0
    for i, vol in enumerate(vols):
        price *= 1.004
        rng = price * 0.0005
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng, volume=vol)
        )


def test_amihud_enters_when_illiquidity_elevated() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _amihud_strategy(_BarsNoFeatures([symbol]), events)
    # High-volume (low-Amihud) warmup, then a low-volume (high-Amihud) stretch:
    # current illiquidity sits well above the rolling median -> gate passes, trend
    # up -> LONG.
    _feed_amihud(strategy, symbol, [1_000_000.0] * 30 + [10_000.0] * 14)
    entries = _entries(events)
    assert entries, "expected a LONG entry in the elevated-illiquidity regime"
    assert entries[0].signal_type == "LONG", entries[0].signal_type


def test_amihud_suppressed_when_not_illiquid() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _amihud_strategy(_BarsNoFeatures([symbol]), events)
    # Reverse schedule: low-volume (high-Amihud) warmup, then high-volume
    # (low-Amihud) recent bars -> current illiquidity is BELOW the rolling median
    # -> gate suppresses the entry despite the same uptrend a plain rider would buy.
    _feed_amihud(strategy, symbol, [10_000.0] * 30 + [1_000_000.0] * 14)
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# None / empty-window safety + state roundtrip
# --------------------------------------------------------------------------- #
def test_sleeves_none_and_empty_window_safe() -> None:
    symbol = "BTC/USDT"
    for factory in (_pe_strategy, _amihud_strategy):
        events = _Events()
        strategy = factory(_BarsNoFeatures([symbol]), events)

        class _EmptyWindow:
            type = "MARKET_WINDOW"
            bars_1s: list[Any] = []

        empty = _EmptyWindow()
        empty.symbol = symbol
        strategy.calculate_signals(empty)  # must not raise
        strategy.calculate_signals(_Market(_ts(0), symbol, 0.0))  # degenerate close, no raise
        assert _entries(events) == []


def test_amihud_state_roundtrip() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _amihud_strategy(_BarsNoFeatures([symbol]), events)
    _feed_amihud(strategy, symbol, [1_000_000.0] * 20 + [50_000.0] * 20)
    snapshot = strategy.get_state()
    restored = _amihud_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert (
        again["amihud_state"][symbol]["amihud_hist"]
        == snapshot["amihud_state"][symbol]["amihud_hist"]
    )
    assert again["amihud_state"][symbol]["volumes"] == snapshot["amihud_state"][symbol]["volumes"]


# --------------------------------------------------------------------------- #
# Candidate-admission wiring
# --------------------------------------------------------------------------- #
def _candidates_for(strategy_class: str) -> list[Any]:
    timeframes = ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    candidates = build_binance_futures_candidates(timeframes=timeframes, symbols=_CRYPTO_UNIVERSE)
    return [c for c in candidates if c.strategy_class == strategy_class]


def test_pe_wired_single_asset_and_only_ge_30m() -> None:
    rows = _candidates_for("PermutationEntropyTrendRiderStrategy")
    assert rows, "expected permutation-entropy candidates"
    tfs = {c.timeframe for c in rows}
    assert tfs <= _ALLOWED_TIMEFRAMES, tfs
    assert not (tfs & _FORBIDDEN_TIMEFRAMES), tfs
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata


def test_amihud_wired_single_asset_and_only_ge_30m() -> None:
    rows = _candidates_for("AmihudIlliquidityMomentumRiderStrategy")
    assert rows, "expected amihud-illiquidity candidates"
    tfs = {c.timeframe for c in rows}
    assert tfs <= _ALLOWED_TIMEFRAMES, tfs
    assert not (tfs & _FORBIDDEN_TIMEFRAMES), tfs
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata
