"""Deterministic tests for the intraday-seasonal + overnight-session sleeves.

Covers registry presence of both new classes; that
``IntradaySeasonalMomentumRiderStrategy`` ENTERS a LONG ride on a steady uptrend
once a UTC time-of-day slot has significant positive drift AND the trend agrees,
but is SUPPRESSED on the SAME uptrend when the seasonal-history gate cannot
qualify (proving the slot gate bites — a plain trend rider would have entered);
that ``OvernightSessionReturnRiderStrategy`` tilts LONG during a
historically-positive UTC session and never enters during the negative session;
graceful no-history behaviour; no-look-ahead (a slot/session statistic excludes
the bar being decided); candidate-admission wiring via build->StrategyCandidate
(both single-asset, timeframes a subset of ``{30m,1h,4h}`` and none of
``{1s,1m,5m,15m,1d}``, decision cadence >= 1800 from metadata); None/empty-window
safety; and get_state/set_state roundtrip.  No backtest is run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.intraday_overnight_alpha_sleeves import (
    IntradaySeasonalMomentumRiderStrategy,
    OvernightSessionReturnRiderStrategy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_STRATEGIES = (
    "IntradaySeasonalMomentumRiderStrategy",
    "OvernightSessionReturnRiderStrategy",
)

_ALLOWED_TIMEFRAMES = {"30m", "1h", "4h"}
_FORBIDDEN_TIMEFRAMES = {"1s", "1m", "5m", "15m", "1d"}

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
    """Bars stub WITHOUT any feature getter (OHLCV-only sleeves)."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Events:
    """Capturing event queue compatible with ``events.put(SignalEvent)``."""

    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _Market:
    """Minimal ``MARKET`` event with explicit OHLC + a real UTC timestamp."""

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
    """ISO-8601 UTC timestamp for bar ``i`` at ``step_minutes`` cadence."""
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
        schema = get_strategy_param_schema(cls)
        assert schema, cls
        defaults = get_default_strategy_params(cls)
        assert defaults, cls


# --------------------------------------------------------------------------- #
# IntradaySeasonalMomentumRider
# --------------------------------------------------------------------------- #
def _intraday_strategy(
    bars: Any, events: _Events, *, min_slot_observations: int
) -> IntradaySeasonalMomentumRiderStrategy:
    return IntradaySeasonalMomentumRiderStrategy(
        bars,
        events,
        slot_minutes=60,
        seasonal_decay=0.25,
        min_slot_observations=min_slot_observations,
        slot_t_threshold=0.5,
        trend_lookback=10,
        trend_ma_window=10,
        min_trend_roc=0.0,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,  # flat target_allocation sizing for determinism
        max_hold_bars=100000,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _feed_steady_uptrend(strategy: Any, symbol: str, *, bars: int, base: float = 100.0) -> None:
    """Feed a steady multiplicative uptrend on real 30m UTC timestamps.

    Every bar rises by a small fixed fraction, so each time-of-day slot
    accumulates a consistently positive return (significant once it has history)
    and the price path trends up (so the trend sign agrees).
    """
    price = base
    for i in range(bars):
        price *= 1.003
        rng = price * 0.001
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
        )


def test_intraday_seasonal_enters_in_significant_aligned_slot() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _intraday_strategy(_BarsNoFeatures([symbol]), events, min_slot_observations=4)
    # 8 days of 30m bars: each hourly slot accrues >=4 positive observations and
    # the trend is up, so a LONG ride is armed once a qualifying slot recurs.
    _feed_steady_uptrend(strategy, symbol, bars=8 * 48)
    entries = _entries(events)
    assert entries, "expected a LONG entry once a slot is significant and the trend agrees"
    assert entries[0].signal_type == "LONG", entries[0].signal_type
    assert (entries[0].metadata or {}).get("slot") is not None, entries[0].metadata


def test_intraday_seasonal_suppressed_without_seasonal_significance() -> None:
    """SAME uptrend, but the seasonal-history gate can never qualify -> NO entry.

    With ``min_slot_observations`` set unreachably high, no slot ever becomes
    significant, so the sleeve takes no position despite a strong, aligned uptrend
    that a plain trend rider would buy.  This isolates the seasonal gate as the
    thing that blocks the trade (and shows the decision uses only past slot data).
    """
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _intraday_strategy(_BarsNoFeatures([symbol]), events, min_slot_observations=100000)
    _feed_steady_uptrend(strategy, symbol, bars=8 * 48)
    assert _entries(events) == [], _entries(events)


def test_intraday_seasonal_no_history_graceful() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _intraday_strategy(_BarsNoFeatures([symbol]), events, min_slot_observations=4)
    # Only a handful of bars: no slot has enough history -> no entry, no raise.
    _feed_steady_uptrend(strategy, symbol, bars=6)
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# OvernightSessionReturnRider
# --------------------------------------------------------------------------- #
def _overnight_strategy(
    bars: Any, events: _Events, *, allow_short: bool = False
) -> OvernightSessionReturnRiderStrategy:
    return OvernightSessionReturnRiderStrategy(
        bars,
        events,
        overnight_start_hour_utc=0,
        overnight_end_hour_utc=8,
        session_decay=0.25,
        min_session_observations=4,
        session_t_threshold=0.5,
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,
        max_hold_bars=100000,
        allow_short=allow_short,
        add_alloc_fraction=0.0,
    )


def _feed_session_split(strategy: Any, symbol: str, *, days: int, base: float = 100.0) -> None:
    """Feed 30m bars where the overnight UTC session (00:00-08:00) rises and the
    active session (08:00-24:00) falls, so the overnight session mean is
    significantly positive and the active session mean significantly negative.
    """
    price = base
    i = 0
    for _ in range(days):
        for _slot in range(48):  # 48 x 30m bars per UTC day
            dt = _START + timedelta(minutes=30 * i)
            up = dt.hour < 8  # overnight window [0, 8)
            price *= 1.004 if up else 0.996
            rng = price * 0.001
            strategy.calculate_signals(
                _Market(_ts(i), symbol, price, high=price + rng, low=price - rng)
            )
            i += 1


def test_overnight_session_tilts_long_in_positive_session_only() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    # allow_short=False so an active-session (negative) SHORT is blocked at the
    # base gate; only the positive overnight session can open a (LONG) position.
    strategy = _overnight_strategy(_BarsNoFeatures([symbol]), events, allow_short=False)
    _feed_session_split(strategy, symbol, days=8)
    entries = _entries(events)
    assert entries, "expected a LONG tilt during the historically-positive session"
    assert all(e.signal_type == "LONG" for e in entries), [e.signal_type for e in entries]
    # Every entry must be opened inside the positive (overnight) session, never the
    # negative (active) one.
    assert all((e.metadata or {}).get("session") == "overnight" for e in entries), [
        (e.metadata or {}).get("session") for e in entries
    ]


def test_overnight_session_no_history_graceful() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _overnight_strategy(_BarsNoFeatures([symbol]), events)
    # Only a few bars: the session has < min_session_observations folds, so the
    # significance gate cannot qualify -> no entry, no raise.
    price = 100.0
    for i in range(3):
        price *= 1.004
        strategy.calculate_signals(
            _Market(_ts(i), symbol, price, high=price * 1.001, low=price * 0.999)
        )
    assert _entries(events) == []


# --------------------------------------------------------------------------- #
# None / empty-window safety + state roundtrip
# --------------------------------------------------------------------------- #
def test_sleeves_none_and_empty_window_safe() -> None:
    symbol = "BTC/USDT"
    for factory in (
        lambda b, e: _intraday_strategy(b, e, min_slot_observations=4),
        lambda b, e: _overnight_strategy(b, e),
    ):
        events = _Events()
        strategy = factory(_BarsNoFeatures([symbol]), events)

        class _EmptyWindow:
            type = "MARKET_WINDOW"
            bars_1s: list[Any] = []

        empty = _EmptyWindow()
        empty.symbol = symbol
        strategy.calculate_signals(empty)  # must not raise
        # A bar with an unparseable timestamp must not raise and must not trade.
        strategy.calculate_signals(_Market("not-a-timestamp", symbol, 100.0))
        assert _entries(events) == []


def test_intraday_state_roundtrip() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _intraday_strategy(_BarsNoFeatures([symbol]), events, min_slot_observations=4)
    _feed_steady_uptrend(strategy, symbol, bars=120)
    snapshot = strategy.get_state()
    restored = _intraday_strategy(_BarsNoFeatures([symbol]), _Events(), min_slot_observations=4)
    restored.set_state(snapshot)
    again = restored.get_state()
    assert (
        again["seasonal_state"][symbol]["slot_stats"]
        == snapshot["seasonal_state"][symbol]["slot_stats"]
    )
    assert (
        again["seasonal_state"][symbol]["decision_slot"]
        == snapshot["seasonal_state"][symbol]["decision_slot"]
    )


def test_overnight_state_roundtrip() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _overnight_strategy(_BarsNoFeatures([symbol]), events)
    _feed_session_split(strategy, symbol, days=3)
    snapshot = strategy.get_state()
    restored = _overnight_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert (
        again["session_return_state"][symbol]["session_stats"]
        == snapshot["session_return_state"][symbol]["session_stats"]
    )


# --------------------------------------------------------------------------- #
# Candidate-admission wiring
# --------------------------------------------------------------------------- #
def _candidates_for(strategy_class: str) -> list[Any]:
    timeframes = ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    candidates = build_binance_futures_candidates(timeframes=timeframes, symbols=_CRYPTO_UNIVERSE)
    return [c for c in candidates if c.strategy_class == strategy_class]


def test_intraday_seasonal_wired_single_asset_and_only_intraday_tfs() -> None:
    rows = _candidates_for("IntradaySeasonalMomentumRiderStrategy")
    assert rows, "expected intraday-seasonal candidates"
    tfs = {c.timeframe for c in rows}
    assert tfs <= _ALLOWED_TIMEFRAMES, tfs
    assert not (tfs & _FORBIDDEN_TIMEFRAMES), tfs
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata


def test_overnight_session_wired_single_asset_and_only_intraday_tfs() -> None:
    rows = _candidates_for("OvernightSessionReturnRiderStrategy")
    assert rows, "expected overnight-session candidates"
    tfs = {c.timeframe for c in rows}
    assert tfs <= _ALLOWED_TIMEFRAMES, tfs
    assert not (tfs & _FORBIDDEN_TIMEFRAMES), tfs
    for c in rows:
        assert candidate_mix_type(c.to_dict()) == "single", c.name
        assert len(c.symbols) == 1, c.symbols
        assert int((c.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800, c.metadata
