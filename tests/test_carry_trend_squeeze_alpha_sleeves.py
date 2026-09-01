"""Deterministic tests for the carry-trend-confluence + squeeze-breakout sleeves.

Covers registry presence of both new classes; dry-emission behaviour for
``CarryTrendConfluenceRiderStrategy`` (ENTERS long when the trend is up AND
funding is favorable for a long; does NOT enter when the trend is up but funding
is hostile = carry veto, and does NOT enter when funding is favorable but the
trend is down); ``VolatilitySqueezeBreakoutRiderStrategy`` (ENTERS on a
volatility expansion + range break that FOLLOWS a low-vol squeeze, and does NOT
enter while still inside the low-vol squeeze); candidate-admission wiring via
build->StrategyCandidate (both single-asset, timeframes a subset of
``{30m,1h,4h,1d}`` and none of ``{1s,1m,5m,15m}``, decision cadence >= 1800 from
metadata); None/empty-window safety; and get_state/set_state roundtrip.  No
backtest is run.
"""

from __future__ import annotations

from typing import Any

from lumina_quant.strategies.carry_trend_squeeze_alpha_sleeves import (
    CarryTrendConfluenceRiderStrategy,
    VolatilitySqueezeBreakoutRiderStrategy,
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
    "CarryTrendConfluenceRiderStrategy",
    "VolatilitySqueezeBreakoutRiderStrategy",
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

    ``feature_values`` maps ``(symbol, field)`` to a constant float so the
    carry sleeve's ``get_latest_feature_value`` cascade can read a deterministic
    ``funding_rate``.  When the map is empty the funding read returns ``None`` so
    the sleeve degrades gracefully (no entries).
    """

    def __init__(
        self,
        symbols: list[str],
        feature_values: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.symbol_list = list(symbols)
        self._feature_values = dict(feature_values or {})

    def get_latest_feature_value(self, symbol: str, field_name: str) -> float | None:
        return self._feature_values.get((str(symbol), str(field_name)))


class _BarsNoFeatures:
    """Bars stub WITHOUT any feature getter (funding data fully absent)."""

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
# CarryTrendConfluenceRider dry-emission
# --------------------------------------------------------------------------- #
def _carry_strategy(bars: Any, events: _Events) -> CarryTrendConfluenceRiderStrategy:
    return CarryTrendConfluenceRiderStrategy(
        bars,
        events,
        trend_lookback=24,
        trend_ma_window=24,
        min_trend_roc=0.0,
        funding_window=4,
        long_carry_funding=0.0,  # LONG requires funding <= 0 (long perp paid).
        short_carry_funding=0.0,  # SHORT requires funding >= 0 (short perp paid).
        trail_atr_mult=4.0,
        atr_period=10,
        max_adds=0,
        vol_window=24,
        target_vol=0.0,  # flat target_allocation sizing for determinism
        max_hold_bars=500,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _feed_uptrend(strategy: Any, symbol: str, n: int = 60, start: float = 100.0) -> float:
    price = start
    for idx in range(n):
        price *= 1.01
        rng = price * 0.002
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )
    return price


def _feed_downtrend(strategy: Any, symbol: str, n: int = 60, start: float = 100.0) -> float:
    price = start
    for idx in range(n):
        price *= 0.99
        rng = price * 0.002
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )
    return price


def test_carry_trend_enters_long_when_trend_up_and_funding_favorable() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    # Favorable carry for a long: funding is NEGATIVE (long perp receives funding).
    bars = _Bars([symbol], feature_values={(symbol, "funding_rate"): -0.0002})
    strategy = _carry_strategy(bars, events)
    _feed_uptrend(strategy, symbol)
    entries = _entries(events)
    assert entries, "expected a LONG confluence entry (trend up + favorable carry)"
    first = entries[0]
    assert first.signal_type == "LONG", first.signal_type
    assert (first.metadata or {}).get("target_allocation", 0.0) > 0.0, first.metadata
    assert (first.metadata or {}).get("avg_funding_rate") is not None, first.metadata
    assert int((first.metadata or {}).get("trend_sign", 0)) == 1, first.metadata


def test_carry_trend_vetoes_long_when_funding_hostile() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    # Trend is UP but funding is strongly POSITIVE (a long would bleed carry):
    # the carry veto must block the long entry.
    bars = _Bars([symbol], feature_values={(symbol, "funding_rate"): 0.0003})
    strategy = _carry_strategy(bars, events)
    _feed_uptrend(strategy, symbol)
    assert _entries(events) == [], _entries(events)


def test_carry_trend_does_not_enter_when_trend_down_but_funding_favorable_for_long() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    # Funding is negative (favorable for a LONG) but the trend is DOWN, so the
    # LONG confluence fails; and SHORT needs funding >= short_carry_funding (>=0),
    # which negative funding does not satisfy -> no entry either way.
    bars = _Bars([symbol], feature_values={(symbol, "funding_rate"): -0.0002})
    strategy = _carry_strategy(bars, events)
    _feed_downtrend(strategy, symbol)
    assert _entries(events) == [], _entries(events)


def test_carry_trend_shorts_when_trend_down_and_funding_high_positive() -> None:
    symbol = "XRP/USDT"
    events = _Events()
    # Trend DOWN and funding strongly POSITIVE (a short perp receives funding):
    # the confluence opens a SHORT.
    bars = _Bars([symbol], feature_values={(symbol, "funding_rate"): 0.0003})
    strategy = _carry_strategy(bars, events)
    _feed_downtrend(strategy, symbol)
    entries = _entries(events)
    assert entries, "expected a SHORT confluence entry (trend down + short receives carry)"
    assert entries[0].signal_type == "SHORT", entries[0].signal_type


def test_carry_trend_degrades_gracefully_without_funding_data() -> None:
    symbol = "ADA/USDT"
    events = _Events()
    # No feature getter at all: funding is fully absent -> no entries.
    strategy = _carry_strategy(_BarsNoFeatures([symbol]), events)
    _feed_uptrend(strategy, symbol)
    assert _entries(events) == [], _entries(events)


# --------------------------------------------------------------------------- #
# VolatilitySqueezeBreakoutRider dry-emission
# --------------------------------------------------------------------------- #
def _squeeze_strategy(
    bars: Any, events: _Events, *, require_bb_in_kc: bool = False
) -> VolatilitySqueezeBreakoutRiderStrategy:
    return VolatilitySqueezeBreakoutRiderStrategy(
        bars,
        events,
        bandwidth_window=10,
        bandwidth_num_std=2.0,
        squeeze_percentile_window=40,
        squeeze_percentile=0.30,
        expansion_mult=1.3,
        breakout_window=10,
        require_bb_in_kc=require_bb_in_kc,
        keltner_window=10,
        keltner_atr_window=10,
        keltner_atr_mult=1.5,
        trail_atr_mult=4.0,
        atr_period=10,
        max_adds=0,
        vol_window=24,
        target_vol=0.0,  # flat target_allocation sizing for determinism
        max_hold_bars=500,
        allow_short=True,
        add_alloc_fraction=0.0,
    )


def _feed_squeeze(strategy: Any, symbol: str, base: float, idx0: int = 0) -> tuple[int, float]:
    """Feed a higher-vol warmup then a tightening coil (a squeeze regime).

    A higher-vol regime first populates the percentile window with elevated
    Bollinger-Bandwidth values, then a progressively TIGHTER coil drives the
    latest bandwidth to a multi-bar percentile low (the contraction state) — a
    perfectly periodic wiggle would instead hold a constant bandwidth (rank
    ~0.5) and never register as a squeeze.  Returns ``(next_idx, last_close)``.
    """
    idx = idx0
    price = base
    # Phase 0: higher-vol "normal" regime so the later coil ranks low.
    for k in range(30):
        amp = 0.03
        price = base * (1.0 + (amp if k % 2 else -amp))
        rng = price * 0.015
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )
        idx += 1
    # Phase 1: progressively tighter coil so the latest bandwidth is the minimum.
    for k in range(20):
        amp = 0.004 / (1.0 + 0.3 * k)
        price = base * (1.0 + (amp if k % 2 else -amp))
        rng = price * 0.0002
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )
        idx += 1
    return idx, price


def test_squeeze_breakout_does_not_enter_inside_low_vol_squeeze() -> None:
    symbol = "BNB/USDT"
    events = _Events()
    strategy = _squeeze_strategy(_BarsNoFeatures([symbol]), events)
    # A warmup + tightening coil only: the contraction regime is active but there
    # is no expansion / range break, so the breakout ride must NOT enter.
    _feed_squeeze(strategy, symbol, base=100.0)
    assert _entries(events) == [], _entries(events)


def test_squeeze_breakout_enters_on_expansion_after_contraction() -> None:
    symbol = "TRX/USDT"
    events = _Events()
    strategy = _squeeze_strategy(_BarsNoFeatures([symbol]), events)
    # Phase 1: warmup + tightening coil so the bandwidth sits at a percentile low.
    idx, price = _feed_squeeze(strategy, symbol, base=100.0)
    assert _entries(events) == [], "no entry expected while still squeezed"
    # Phase 2: a sharp UPSIDE expansion + range break out of the coil. The first
    # large bar both expands the bandwidth and closes above the prior N-bar high.
    breakout_moves = [0.06, 0.04, 0.04, 0.03]
    for mv in breakout_moves:
        price *= 1.0 + mv
        rng = price * 0.01
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )
        idx += 1
    entries = _entries(events)
    assert entries, "expected a breakout entry on expansion after the squeeze"
    first = entries[0]
    assert first.signal_type == "LONG", first.signal_type
    assert (first.metadata or {}).get("target_allocation", 0.0) > 0.0, first.metadata
    assert (first.metadata or {}).get("squeeze_bandwidth") is not None, first.metadata


def test_squeeze_keltner_gate_changes_behavior() -> None:
    """require_bb_in_kc routes the squeeze latch through the Keltner condition.

    A percentile-low bandwidth coil that would otherwise latch a squeeze must be
    VETOED when the optional TTM-squeeze gate (Bollinger band inside the Keltner
    channel) is on and the band is outside the channel — proving the four
    keltner_* / require_bb_in_kc params are consumed, not dead.
    """
    symbol = "BNB/USDT"
    # Gate OFF (default): the tightening coil latches a squeeze on its own.
    off = _squeeze_strategy(_BarsNoFeatures([symbol]), _Events())
    _feed_squeeze(off, symbol, base=100.0)
    assert off._prev_squeeze[symbol] is True

    # Gate ON, Keltner condition forced to FAIL: the same coil must NOT register
    # as a squeeze (the latch now honors the gate).
    on = _squeeze_strategy(_BarsNoFeatures([symbol]), _Events(), require_bb_in_kc=True)
    on._bb_inside_kc = lambda item: False  # type: ignore[method-assign]
    _feed_squeeze(on, symbol, base=100.0)
    assert on._prev_squeeze[symbol] is False

    # Gate ON, Keltner condition satisfied: the latch matches the gate-off path,
    # so only the Keltner result flips the squeeze (the percentile gate is intact).
    ok = _squeeze_strategy(_BarsNoFeatures([symbol]), _Events(), require_bb_in_kc=True)
    ok._bb_inside_kc = lambda item: True  # type: ignore[method-assign]
    _feed_squeeze(ok, symbol, base=100.0)
    assert ok._prev_squeeze[symbol] is True


# --------------------------------------------------------------------------- #
# None / empty-window safety
# --------------------------------------------------------------------------- #
def test_none_and_empty_window_safety() -> None:
    symbol = "BTC/USDT"

    class _EmptyWindow:
        type = "MARKET_WINDOW"
        time = "t0"
        bars_1s: dict[str, Any] = {}

    for builder in (_carry_strategy, _squeeze_strategy):
        events = _Events()
        strategy = builder(_Bars([symbol]), events)
        # Empty window event, then a degenerate sub-min-price close: never raise.
        strategy.calculate_signals(_EmptyWindow())
        strategy.calculate_signals(_Market("t1", symbol, 0.0))
        strategy.calculate_signals(_Market("t2", "UNLISTED/USDT", 100.0))
        assert events.signals == [], events.signals


# --------------------------------------------------------------------------- #
# get_state / set_state roundtrip
# --------------------------------------------------------------------------- #
def test_carry_state_roundtrip() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    bars = _Bars([symbol], feature_values={(symbol, "funding_rate"): -0.0002})
    strategy = _carry_strategy(bars, events)
    _feed_uptrend(strategy, symbol, n=40)
    snapshot = strategy.get_state()

    restored = _carry_strategy(_Bars([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["symbol_state"][symbol]["closes"] == snapshot["symbol_state"][symbol]["closes"]
    assert again["symbol_state"][symbol]["mode"] == snapshot["symbol_state"][symbol]["mode"]
    assert again["funding"][symbol] == snapshot["funding"][symbol]


def test_squeeze_state_roundtrip() -> None:
    symbol = "SOL/USDT"
    events = _Events()
    strategy = _squeeze_strategy(_BarsNoFeatures([symbol]), events)
    _feed_squeeze(strategy, symbol, base=100.0)
    snapshot = strategy.get_state()

    restored = _squeeze_strategy(_BarsNoFeatures([symbol]), _Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert (
        again["squeeze_state"][symbol]["bandwidths"]
        == snapshot["squeeze_state"][symbol]["bandwidths"]
    )
    assert (
        again["squeeze_state"][symbol]["prev_squeeze"]
        == snapshot["squeeze_state"][symbol]["prev_squeeze"]
    )
    assert again["symbol_state"][symbol]["closes"] == snapshot["symbol_state"][symbol]["closes"]


# --------------------------------------------------------------------------- #
# Candidate wiring + admission
# --------------------------------------------------------------------------- #
def _build_rows_with_perp() -> list[Any]:
    """Build candidates with perp-support forced on (carry sleeve is perp-gated)."""
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


def test_carry_trend_confluence_wired_single_asset_and_only_ge_30m() -> None:
    rows = [
        r
        for r in _build_rows_with_perp()
        if r.strategy_class == "CarryTrendConfluenceRiderStrategy"
    ]
    assert rows, "carry-trend-confluence builder emitted no candidates"
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert len(row.symbols) == 1, row.name
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert "single_asset" in row.tags
        assert "confluence" in row.tags
        assert row.metadata["timeframe"] == row.timeframe
        assert int(row.metadata["decision_cadence_seconds"]) >= 1800, row.name


def test_squeeze_breakout_wired_single_asset_and_only_ge_30m() -> None:
    rows = [
        r
        for r in _build_rows_with_perp()
        if r.strategy_class == "VolatilitySqueezeBreakoutRiderStrategy"
    ]
    assert rows, "squeeze-breakout builder emitted no candidates"
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert len(row.symbols) == 1, row.name
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert "single_asset" in row.tags
        assert "volatility_squeeze" in row.tags
        assert row.metadata["timeframe"] == row.timeframe
        assert int(row.metadata["decision_cadence_seconds"]) >= 1800, row.name


def test_new_builders_only_emit_allowed_timeframes() -> None:
    rows = build_binance_futures_candidates(
        timeframes=["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        symbols=_CRYPTO_UNIVERSE,
    )
    for row in rows:
        if row.strategy_class in NEW_STRATEGIES:
            assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
            assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
