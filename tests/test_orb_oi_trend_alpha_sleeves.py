"""Deterministic tests for the ORB + open-interest-trend-confirmation sleeves.

Covers registry presence of both new classes; dry-emission behaviour for
``OpeningRangeBreakoutRiderStrategy`` (does NOT enter during the opening-range
accumulation window, ENTERS LONG on an upside break AFTER the range is set, and
the opening range RESETS on a new UTC day so the next break is measured against
the NEW day's range, not the old one); ``OpenInterestTrendConfirmationRiderStrategy``
(ENTERS long on trend-up + RISING OI, does NOT enter on trend-up + FALLING OI =
short-covering suppression, and gracefully no-enters when OI data is absent);
candidate-admission wiring via build->StrategyCandidate (both single-asset,
timeframes a subset of ``{30m,1h,4h,1d}`` and none of ``{1s,1m,5m,15m}``,
decision cadence >= 1800 from metadata, OI sleeve perp-gated); None/empty-window
safety; and get_state/set_state roundtrip.  No backtest is run.
"""

from __future__ import annotations

from typing import Any

from lumina_quant.strategies.orb_oi_trend_alpha_sleeves import (
    OpeningRangeBreakoutRiderStrategy,
    OpenInterestTrendConfirmationRiderStrategy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory import candidate_library
from lumina_quant.strategy_factory.candidate_library import _CandidateBuildContext
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_STRATEGIES = (
    "OpeningRangeBreakoutRiderStrategy",
    "OpenInterestTrendConfirmationRiderStrategy",
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


class _Bars:
    """Minimal bars stub: ``symbol_list`` plus optional latched feature values.

    ``feature_values`` maps ``(symbol, field)`` to a constant float so the OI
    sleeve's ``get_latest_feature_value`` cascade can read a deterministic
    ``open_interest``.  When a per-call OI override is needed, pass a callable
    ``oi_series`` instead.
    """

    def __init__(
        self,
        symbols: list[str],
        feature_values: dict[tuple[str, str], float] | None = None,
        oi_series: Any = None,
    ) -> None:
        self.symbol_list = list(symbols)
        self._feature_values = dict(feature_values or {})
        self._oi_series = oi_series

    def get_latest_feature_value(self, symbol: str, field_name: str) -> float | None:
        if self._oi_series is not None and str(field_name) == "open_interest":
            return self._oi_series(str(symbol))
        return self._feature_values.get((str(symbol), str(field_name)))


class _BarsNoFeatures:
    """Bars stub WITHOUT any feature getter (open-interest data fully absent)."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Events:
    """Capturing event queue compatible with ``events.put(SignalEvent)``."""

    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _Market:
    """Minimal ``MARKET`` event with explicit OHLC fields used by the sleeves."""

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
        self.symbol = symbol
        self.open = open_ if open_ is not None else close
        self.high = high if high is not None else close * 1.0005
        self.low = low if low is not None else close * 0.9995
        self.close = close
        self.volume = 1_000.0


def _entries(events: _Events) -> list[Any]:
    return [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]


# --------------------------------------------------------------------------- #
# Registry presence
# --------------------------------------------------------------------------- #
def test_new_strategies_register_after_lazy_discovery() -> None:
    discovered = set(get_strategy_names())
    assert sum(1 for name in NEW_STRATEGIES if name in discovered) == 2
    for strategy_name in NEW_STRATEGIES:
        assert strategy_name in discovered, strategy_name
        schema = get_strategy_param_schema(strategy_name)
        assert schema, strategy_name
        defaults = get_default_strategy_params(strategy_name)
        assert defaults, strategy_name
        assert set(defaults).issubset(schema)


# --------------------------------------------------------------------------- #
# OpeningRangeBreakoutRider dry-emission
# --------------------------------------------------------------------------- #
def _orb_strategy(bars: Any, events: _Events) -> OpeningRangeBreakoutRiderStrategy:
    return OpeningRangeBreakoutRiderStrategy(
        bars,
        events,
        opening_range_bars=3,
        session_start_minute_utc=0,
        buffer_atr_mult=0.0,  # no buffer so a clean break above the range fires
        trail_atr_mult=4.0,
        atr_period=5,
        max_adds=0,
        vol_window=10,
        target_vol=0.0,  # flat target_allocation sizing for determinism
        max_hold_bars=500,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _ts(day: int, hour: int) -> str:
    """Build a UTC ISO timestamp for ``2026-06-{day}T{hour}:00:00Z``."""
    return f"2026-06-{day:02d}T{hour:02d}:00:00+00:00"


def test_orb_does_not_enter_during_opening_range_window() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = _orb_strategy(_BarsNoFeatures([symbol]), events)
    # First 3 bars (opening_range_bars=3) of UTC day 15 build the range: bars at
    # hours 0,1,2 accumulate the high/low band [99, 101]. No trade may fire while
    # the range is still accumulating (session_bars <= opening_range_bars).
    strategy.calculate_signals(_Market(_ts(15, 0), symbol, 100.0, high=100.5, low=99.5))
    strategy.calculate_signals(_Market(_ts(15, 1), symbol, 100.5, high=101.0, low=100.0))
    strategy.calculate_signals(_Market(_ts(15, 2), symbol, 99.5, high=100.0, low=99.0))
    assert _entries(events) == [], _entries(events)


def test_orb_enters_long_on_upside_break_after_range_set() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _orb_strategy(_BarsNoFeatures([symbol]), events)
    # Accumulate the opening range over the first 3 bars: high band ~101.0.
    strategy.calculate_signals(_Market(_ts(15, 0), symbol, 100.0, high=100.5, low=99.5))
    strategy.calculate_signals(_Market(_ts(15, 1), symbol, 100.5, high=101.0, low=100.0))
    strategy.calculate_signals(_Market(_ts(15, 2), symbol, 99.5, high=100.0, low=99.0))
    assert _entries(events) == [], "no entry while the range accumulates"
    # 4th bar (range now established): a close ABOVE the opening-range high (101.0)
    # arms the one-shot breakout -> LONG.
    strategy.calculate_signals(_Market(_ts(15, 3), symbol, 102.0, high=102.5, low=101.0))
    entries = _entries(events)
    assert entries, "expected a LONG break above the opening-range high"
    first = entries[0]
    assert first.signal_type == "LONG", first.signal_type
    assert (first.metadata or {}).get("target_allocation", 0.0) > 0.0, first.metadata
    assert (first.metadata or {}).get("opening_range_high") is not None, first.metadata


def test_orb_range_resets_on_new_utc_day() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _orb_strategy(_BarsNoFeatures([symbol]), events)
    # Day 15: a HIGH opening range around 200. Then break above 201 -> LONG, then
    # let the trailing stop flatten the position before the new day so day 16 can
    # arm a fresh break.
    strategy.calculate_signals(_Market(_ts(15, 0), symbol, 200.0, high=200.5, low=199.5))
    strategy.calculate_signals(_Market(_ts(15, 1), symbol, 200.5, high=201.0, low=200.0))
    strategy.calculate_signals(_Market(_ts(15, 2), symbol, 199.5, high=200.0, low=199.0))
    strategy.calculate_signals(_Market(_ts(15, 3), symbol, 203.0, high=203.5, low=202.0))
    day15_entries = _entries(events)
    assert day15_entries and day15_entries[0].signal_type == "LONG", day15_entries
    # Flatten day-15 long via a hard down move that breaches the trailing stop.
    strategy.calculate_signals(_Market(_ts(15, 4), symbol, 150.0, high=151.0, low=149.0))

    # Day 16 opens a MUCH LOWER session (~100). The opening range MUST reset to the
    # new day's first-k-bars extremes; a close at 102 is BELOW the OLD day-15 high
    # (201) yet ABOVE the NEW day-16 range high (~101), so the reset is what lets a
    # fresh LONG arm. If the range had NOT reset, no day-16 LONG could fire (102 <
    # the stale 201 level).
    before = len(_entries(events))
    strategy.calculate_signals(_Market(_ts(16, 0), symbol, 100.0, high=100.5, low=99.5))
    strategy.calculate_signals(_Market(_ts(16, 1), symbol, 100.5, high=101.0, low=100.0))
    strategy.calculate_signals(_Market(_ts(16, 2), symbol, 99.5, high=100.0, low=99.0))
    # No entry while the day-16 range accumulates.
    assert len(_entries(events)) == before, "no day-16 entry during accumulation"
    strategy.calculate_signals(_Market(_ts(16, 3), symbol, 102.0, high=102.5, low=101.0))
    day16_entries = _entries(events)[before:]
    assert day16_entries, "expected a fresh LONG vs the NEW day-16 range (reset)"
    first16 = day16_entries[0]
    assert first16.signal_type == "LONG", first16.signal_type
    # The break is relative to the NEW day's range (high ~101), not the stale ~201.
    assert float((first16.metadata or {}).get("opening_range_high")) < 150.0, first16.metadata
    assert (first16.metadata or {}).get("session_key") == "2026-06-16", first16.metadata


# --------------------------------------------------------------------------- #
# OpenInterestTrendConfirmationRider dry-emission
# --------------------------------------------------------------------------- #
def _oi_strategy(bars: Any, events: _Events) -> OpenInterestTrendConfirmationRiderStrategy:
    return OpenInterestTrendConfirmationRiderStrategy(
        bars,
        events,
        trend_lookback=24,
        trend_ma_window=24,
        min_trend_roc=0.0,
        oi_lookback=4,
        oi_rise_threshold=0.0,
        trail_atr_mult=4.0,
        atr_period=10,
        max_adds=0,
        vol_window=24,
        target_vol=0.0,  # flat target_allocation sizing for determinism
        max_hold_bars=500,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _feed_uptrend_with_oi(
    strategy: Any, symbol: str, oi_state: dict[str, float], n: int, oi_step: float
) -> None:
    price = 100.0
    for idx in range(n):
        price *= 1.01
        oi_state[symbol] = oi_state.get(symbol, 1_000_000.0) + oi_step
        rng = price * 0.002
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )


def test_oi_trend_enters_long_on_trend_up_and_rising_oi() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    oi_state: dict[str, float] = {}
    bars = _Bars([symbol], oi_series=lambda s: oi_state.get(s))
    strategy = _oi_strategy(bars, events)
    # Trend UP and open interest RISING each bar (fresh longs committing) -> LONG.
    _feed_uptrend_with_oi(strategy, symbol, oi_state, n=60, oi_step=5_000.0)
    entries = _entries(events)
    assert entries, "expected a LONG entry (trend up + rising OI confirms)"
    first = entries[0]
    assert first.signal_type == "LONG", first.signal_type
    assert (first.metadata or {}).get("target_allocation", 0.0) > 0.0, first.metadata
    assert (first.metadata or {}).get("oi_rising") is True, first.metadata
    assert int((first.metadata or {}).get("trend_sign", 0)) == 1, first.metadata


def test_oi_trend_suppresses_entry_on_trend_up_but_falling_oi() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    oi_state: dict[str, float] = {}
    bars = _Bars([symbol], oi_series=lambda s: oi_state.get(s))
    strategy = _oi_strategy(bars, events)
    # Trend UP but open interest FALLING each bar (short-covering / unwind): the
    # OI gate must suppress the entry despite the clean uptrend.
    _feed_uptrend_with_oi(strategy, symbol, oi_state, n=60, oi_step=-5_000.0)
    assert _entries(events) == [], _entries(events)


def test_oi_trend_degrades_gracefully_without_oi_data() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    # No feature getter at all: open interest is fully absent -> no entries.
    strategy = _oi_strategy(_BarsNoFeatures([symbol]), events)
    price = 100.0
    for idx in range(60):
        price *= 1.01
        rng = price * 0.002
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )
    assert _entries(events) == [], _entries(events)


# --------------------------------------------------------------------------- #
# None / empty-window safety
# --------------------------------------------------------------------------- #
def test_none_and_empty_window_safety() -> None:
    symbol = "BTC/USDT"

    class _EmptyWindow:
        type = "MARKET_WINDOW"
        time = "t0"
        bars_1s: dict[str, Any] = {}

    for builder in (_orb_strategy, _oi_strategy):
        events = _Events()
        strategy = builder(_Bars([symbol]), events)
        # Empty window event, then a degenerate sub-min-price close, then an
        # unlisted symbol, then a bar with no parseable timestamp: never raise.
        strategy.calculate_signals(_EmptyWindow())
        strategy.calculate_signals(_Market("t1", symbol, 0.0))
        strategy.calculate_signals(_Market("t2", "UNLISTED/USDT", 100.0))
        strategy.calculate_signals(_Market("not-a-timestamp", symbol, 100.0))
        assert events.signals == [], events.signals


# --------------------------------------------------------------------------- #
# get_state / set_state roundtrip
# --------------------------------------------------------------------------- #
def test_orb_state_roundtrip() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = _orb_strategy(_BarsNoFeatures([symbol]), events)
    strategy.calculate_signals(_Market(_ts(15, 0), symbol, 100.0, high=100.5, low=99.5))
    strategy.calculate_signals(_Market(_ts(15, 1), symbol, 100.5, high=101.0, low=100.0))
    strategy.calculate_signals(_Market(_ts(15, 2), symbol, 99.5, high=100.0, low=99.0))
    strategy.calculate_signals(_Market(_ts(15, 3), symbol, 102.0, high=102.5, low=101.0))
    snapshot = strategy.get_state()

    restored = _orb_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["symbol_state"][symbol]["closes"] == snapshot["symbol_state"][symbol]["closes"]
    assert again["symbol_state"][symbol]["mode"] == snapshot["symbol_state"][symbol]["mode"]
    assert again["session_state"][symbol] == snapshot["session_state"][symbol]


def test_oi_state_roundtrip() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    oi_state: dict[str, float] = {}
    bars = _Bars([symbol], oi_series=lambda s: oi_state.get(s))
    strategy = _oi_strategy(bars, events)
    _feed_uptrend_with_oi(strategy, symbol, oi_state, n=40, oi_step=5_000.0)
    snapshot = strategy.get_state()

    restored = _oi_strategy(_Bars([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["symbol_state"][symbol]["closes"] == snapshot["symbol_state"][symbol]["closes"]
    assert again["symbol_state"][symbol]["mode"] == snapshot["symbol_state"][symbol]["mode"]
    assert again["open_interest"][symbol] == snapshot["open_interest"][symbol]


# --------------------------------------------------------------------------- #
# Candidate wiring + admission
# --------------------------------------------------------------------------- #
def _build_rows_with_perp() -> list[Any]:
    """Build candidates with perp-support forced on (OI sleeve is perp-gated)."""
    original = candidate_library._has_perp_support_data
    try:
        candidate_library._has_perp_support_data = lambda: True
        ctx = _CandidateBuildContext(
            normalized_timeframes=("1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"),
            normalized_symbols=tuple(_CRYPTO_UNIVERSE),
        )
        ctx.perp_support_data_available = True
        return list(ctx.build())
    finally:
        candidate_library._has_perp_support_data = original


def test_orb_breakout_wired_single_asset_and_only_ge_30m() -> None:
    rows = [
        r
        for r in _build_rows_with_perp()
        if r.strategy_class == "OpeningRangeBreakoutRiderStrategy"
    ]
    assert rows, "opening-range-breakout builder emitted no candidates"
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert len(row.symbols) == 1, row.name
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert "single_asset" in row.tags
        assert "opening_range" in row.tags
        assert row.metadata["timeframe"] == row.timeframe
        assert int(row.metadata["decision_cadence_seconds"]) >= 1800, row.name


def test_oi_trend_wired_single_asset_perp_gated_and_only_ge_30m() -> None:
    rows = [
        r
        for r in _build_rows_with_perp()
        if r.strategy_class == "OpenInterestTrendConfirmationRiderStrategy"
    ]
    assert rows, "open-interest-trend builder emitted no candidates"
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert len(row.symbols) == 1, row.name
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert "single_asset" in row.tags
        assert "open_interest" in row.tags
        assert row.metadata["timeframe"] == row.timeframe
        assert bool(row.metadata.get("data_dependent")) is True, row.name
        assert int(row.metadata["decision_cadence_seconds"]) >= 1800, row.name


def test_oi_trend_perp_gate_suppresses_when_no_perp_support() -> None:
    """With perp support OFF, the perp-gated OI sleeve must emit nothing."""
    original = candidate_library._has_perp_support_data
    try:
        candidate_library._has_perp_support_data = lambda: False
        ctx = _CandidateBuildContext(
            normalized_timeframes=("30m", "1h", "4h", "1d"),
            normalized_symbols=tuple(_CRYPTO_UNIVERSE),
        )
        ctx.perp_support_data_available = False
        rows = [
            r
            for r in ctx.build()
            if r.strategy_class == "OpenInterestTrendConfirmationRiderStrategy"
        ]
        assert rows == [], rows
    finally:
        candidate_library._has_perp_support_data = original


def test_new_builders_only_emit_allowed_timeframes() -> None:
    rows = build_binance_futures_candidates(
        timeframes=["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        symbols=_CRYPTO_UNIVERSE,
    )
    for row in rows:
        if row.strategy_class in NEW_STRATEGIES:
            assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
            assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
