"""Deterministic tests for the Kalman-trend + realized-semivariance sleeves.

Covers registry presence of both new classes; that ``KalmanTrendRiderStrategy``
ENTERS a LONG ride on a clean uptrend (the filtered slope becomes significant)
and does NOT enter on a flat series (no slope); that
``RealizedSemivarianceTrendRiderStrategy`` ENTERS LONG when the upside/downside
semivariance asymmetry is positive AND the trend agrees, and does NOT enter on a
symmetric, trendless oscillation; graceful no-history behaviour; candidate
wiring (both single-asset, allowed timeframes, cadence >= 1800 from metadata);
None/empty-window safety; and get_state/set_state roundtrip.  No backtest is run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.kalman_semivar_alpha_sleeves import (
    KalmanTrendRiderStrategy,
    RealizedSemivarianceTrendRiderStrategy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_STRATEGIES = (
    "KalmanTrendRiderStrategy",
    "RealizedSemivarianceTrendRiderStrategy",
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
# KalmanTrendRider
# --------------------------------------------------------------------------- #
def _kalman_strategy(bars: Any, events: _Events) -> KalmanTrendRiderStrategy:
    return KalmanTrendRiderStrategy(
        bars,
        events,
        obs_noise=0.0001,
        level_process_noise=2e-5,
        slope_process_noise=2e-7,
        slope_t=1.0,
        min_slope_frac=0.0,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def test_kalman_enters_on_clean_uptrend() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _kalman_strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for i in range(120):
        price *= 1.003
        rng = price * 0.001
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )
    entries = _entries(events)
    assert entries, "expected a LONG entry once the Kalman slope is significant"
    assert entries[0].signal_type == "LONG", entries[0].signal_type
    assert (entries[0].metadata or {}).get("kalman_slope") is not None, entries[0].metadata


def test_kalman_enters_short_on_clean_downtrend() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _kalman_strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for i in range(120):
        price *= 0.997
        rng = price * 0.001
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )
    entries = _entries(events)
    assert entries, "expected a SHORT entry once the Kalman slope is significantly negative"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type


def test_kalman_no_entry_on_flat_series() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _kalman_strategy(_BarsNoFeatures([symbol]), events)
    # A constant price has no drift -> the filtered slope stays ~0 -> no entry.
    for i in range(120):
        strategy.calculate_signals(_Market(_ts(i), symbol, 100.0, high=100.05, low=99.95))
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# RealizedSemivarianceTrendRider
# --------------------------------------------------------------------------- #
def _semivar_strategy(bars: Any, events: _Events) -> RealizedSemivarianceTrendRiderStrategy:
    return RealizedSemivarianceTrendRiderStrategy(
        bars,
        events,
        semivar_window=20,
        semivar_threshold=0.20,
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


def test_semivar_enters_on_positive_asymmetry_uptrend() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _semivar_strategy(_BarsNoFeatures([symbol]), events)
    # A steady uptrend has only positive returns -> RS-=0 -> SJ=+1 >= threshold,
    # and the trend is up, so a LONG ride is armed.
    price = 100.0
    for i in range(60):
        price *= 1.004
        rng = price * 0.001
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )
    entries = _entries(events)
    assert entries, "expected a LONG entry on positive semivariance asymmetry + uptrend"
    assert entries[0].signal_type == "LONG", entries[0].signal_type
    sj = (entries[0].metadata or {}).get("signed_semivariance_asymmetry")
    assert sj is not None and sj > 0.0, entries[0].metadata


def test_semivar_enters_short_on_negative_asymmetry_downtrend() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _semivar_strategy(_BarsNoFeatures([symbol]), events)
    # A steady downtrend has only negative returns -> RS+=0 -> SJ=-1 <= -threshold,
    # and the trend is down, so a SHORT ride is armed (the mirror of the LONG case).
    price = 100.0
    for i in range(60):
        price *= 0.996
        rng = price * 0.001
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )
    entries = _entries(events)
    assert entries, "expected a SHORT entry on negative semivariance asymmetry + downtrend"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type
    sj = (entries[0].metadata or {}).get("signed_semivariance_asymmetry")
    assert sj is not None and sj < 0.0, entries[0].metadata


def test_semivar_no_entry_on_symmetric_trendless() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _semivar_strategy(_BarsNoFeatures([symbol]), events)
    # Symmetric oscillation: equal up/down moves -> SJ~0 (< threshold) and no
    # net trend -> no entry on either gate.
    base = 100.0
    for i in range(80):
        price = base * (1.01 if i % 2 == 0 else 1.0 / 1.01)
        rng = price * 0.001
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )
    assert _entries(events) == []


def test_semivar_no_history_graceful() -> None:
    symbol = "TRX/USDT"
    events = _Events()
    strategy = _semivar_strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for i in range(5):  # < semivar_window
        price *= 1.004
        strategy.calculate_signals(_Market(_ts(i), symbol, price))
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# None / empty-window safety + state roundtrip
# --------------------------------------------------------------------------- #
def test_sleeves_none_and_empty_window_safe() -> None:
    symbol = "BTC/USDT"
    for factory in (_kalman_strategy, _semivar_strategy):
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


def test_kalman_state_roundtrip() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _kalman_strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for i in range(40):
        price *= 1.003
        strategy.calculate_signals(_Market(_ts(i), symbol, price))
    snapshot = strategy.get_state()
    restored = _kalman_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["kalman_state"][symbol] == snapshot["kalman_state"][symbol]


def test_semivar_state_roundtrip() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _semivar_strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for i in range(40):
        price *= 1.004
        strategy.calculate_signals(_Market(_ts(i), symbol, price))
    snapshot = strategy.get_state()
    restored = _semivar_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["semivar_returns"][symbol] == snapshot["semivar_returns"][symbol]


# --------------------------------------------------------------------------- #
# Candidate-admission wiring
# --------------------------------------------------------------------------- #
def _candidates_for(strategy_class: str) -> list[Any]:
    timeframes = ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    candidates = build_binance_futures_candidates(timeframes=timeframes, symbols=_CRYPTO_UNIVERSE)
    return [c for c in candidates if c.strategy_class == strategy_class]


def test_kalman_wired_single_asset_and_only_ge_30m() -> None:
    rows = _candidates_for("KalmanTrendRiderStrategy")
    assert rows, "expected kalman-trend candidates"
    tfs = {c.timeframe for c in rows}
    assert tfs <= _ALLOWED_TIMEFRAMES, tfs
    assert not (tfs & _FORBIDDEN_TIMEFRAMES), tfs
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata


def test_semivar_wired_single_asset_and_only_ge_30m() -> None:
    rows = _candidates_for("RealizedSemivarianceTrendRiderStrategy")
    assert rows, "expected realized-semivariance candidates"
    tfs = {c.timeframe for c in rows}
    assert tfs <= _ALLOWED_TIMEFRAMES, tfs
    assert not (tfs & _FORBIDDEN_TIMEFRAMES), tfs
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata
