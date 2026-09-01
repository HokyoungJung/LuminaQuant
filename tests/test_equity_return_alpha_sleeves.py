"""Deterministic tests for the directional EQUITY/ETF return alpha sleeves.

Covers registry presence of the new classes; dry-emission ride behaviour for the
reused single-name equity trend rider and the leveraged-ETF trend timer (enters
only above SMA200 + golden cross, rides with a trailing stop, exits below the
trend filter, never shorts); dual-momentum-defensive rotation into the defensive
leg on a risk-off (benchmark-down) series and into the strongest equity ETF on a
risk-on series; and candidate-wiring assertions over a mixed universe — each new
builder targets EQUITY/ETF (not crypto) symbols, single-asset for S-EQ1/EQ2 and
cross_sectional-admissible for S-EQ3, with timeframes restricted to
``{30m,1h,4h,1d}`` — plus the crypto-rider de-leak (crypto riders no longer emit
candidates for equity/ETF symbols). No backtest is run.
"""

from __future__ import annotations

from typing import Any

from lumina_quant.strategies.equity_return_alpha_sleeves import (
    DualMomentumDefensiveRotationStrategy,
    LeveragedTrendTimingRiderStrategy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_EQUITY_STRATEGIES = (
    "LeveragedTrendTimingRiderStrategy",
    "DualMomentumDefensiveRotationStrategy",
)

_ALLOWED_TIMEFRAMES = {"30m", "1h", "4h", "1d"}
_FORBIDDEN_TIMEFRAMES = {"1s", "1m", "5m", "15m"}

# A mixed universe spanning crypto, equity, ETF/index, leveraged ETF, defensive.
_CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
_EQUITY_SYMBOLS = [
    "NVDA/USDT",
    "TSLA/USDT",
    "PLTR/USDT",
    "MSTR/USDT",
    "COIN/USDT",
    "AMD/USDT",
    "AAPL/USDT",
]
_ROTATION_ETF_SYMBOLS = [
    "SPY/USDT",
    "QQQ/USDT",
    "IWM/USDT",
    "EWY/USDT",
    "EWJ/USDT",
    "EWT/USDT",
    "EWZ/USDT",
    "STXX/USDT",
]
_LEVERAGED_ETF_SYMBOLS = ["SOXL/USDT", "URNM/USDT"]
_DEFENSIVE_SYMBOLS = ["XAU/USDT", "XLE/USDT", "UVXY/USDT"]
_EQUITY_ETF_DEFENSIVE = set(
    _EQUITY_SYMBOLS + _ROTATION_ETF_SYMBOLS + _LEVERAGED_ETF_SYMBOLS + _DEFENSIVE_SYMBOLS
)
_MIXED_UNIVERSE = (
    _CRYPTO_SYMBOLS
    + _EQUITY_SYMBOLS
    + _ROTATION_ETF_SYMBOLS
    + _LEVERAGED_ETF_SYMBOLS
    + ["XAG/USDT"]
    + _DEFENSIVE_SYMBOLS
)


class _Bars:
    """Minimal bars stub exposing only ``symbol_list``."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Events:
    """Capturing event queue compatible with ``events.put(SignalEvent)``."""

    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _Market:
    """Minimal ``MARKET`` event with OHLC fields used by the sleeves."""

    type = "MARKET"

    def __init__(
        self,
        time: str,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> None:
        self.time = time
        self.symbol = symbol
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = 1_000.0


def _ohlc(prev_close: float, close: float) -> tuple[float, float, float, float]:
    """Derive a plausible OHLC bar from the prior and current close."""
    high = max(prev_close, close) * 1.004
    low = min(prev_close, close) * 0.996
    return prev_close, high, low, close


def _leveraged_trend_series() -> list[float]:
    """Seed a low SMA baseline, then a sustained uptrend, then a crash.

    The long low/choppy lead-in keeps the 200-bar SMA well below the eventual
    uptrend so the regime gate + golden cross only turn on once the trend leg is
    well established, then a sharp crash breaches the trailing stop / trend filter.
    """
    closes: list[float] = []
    price = 100.0
    for _ in range(40):
        closes.append(price)
    for idx in range(220):
        price *= 1.0 + (0.001 if idx % 2 else -0.001)
        closes.append(price)
    for idx in range(120):
        pull = -0.004 if idx % 6 == 5 else 0.0
        price *= 1.0 + 0.015 + pull
        closes.append(price)
    for _ in range(30):
        price *= 0.94
        closes.append(price)
    return closes


def _run_single_asset(strategy: Any, symbol: str, closes: list[float]) -> dict[str, Any]:
    """Feed ``closes`` bar-by-bar and summarize the emitted ride behaviour."""
    events: _Events = strategy.events
    prev = closes[0]
    summary: dict[str, Any] = {
        "entries": 0,
        "pyramids": 0,
        "longs": 0,
        "shorts": 0,
        "trailing_stop_exits": 0,
        "trend_filter_exits": 0,
        "max_position_run": 0,
        "entry_direction": None,
    }
    run = 0
    for idx, close in enumerate(closes):
        open_, high, low, close_value = _ohlc(prev, close)
        prev = close
        before = len(events.signals)
        strategy.calculate_signals(_Market(f"t{idx:04d}", symbol, open_, high, low, close_value))
        for event in events.signals[before:]:
            metadata = event.metadata or {}
            signal_type = event.signal_type
            if signal_type == "LONG":
                summary["longs"] += 1
                if metadata.get("action") == "pyramid_add":
                    summary["pyramids"] += 1
                elif metadata.get("action") == "entry":
                    summary["entries"] += 1
                    if summary["entry_direction"] is None:
                        summary["entry_direction"] = "LONG"
            elif signal_type == "SHORT":
                summary["shorts"] += 1
                if summary["entry_direction"] is None:
                    summary["entry_direction"] = "SHORT"
            elif signal_type == "EXIT":
                if metadata.get("reason") == "trailing_stop":
                    summary["trailing_stop_exits"] += 1
                if metadata.get("reason") == "trend_filter_lost":
                    summary["trend_filter_exits"] += 1
        item = strategy._state[symbol]
        if item.mode in {"LONG", "SHORT"}:
            run += 1
            summary["max_position_run"] = max(summary["max_position_run"], run)
        else:
            run = 0
    return summary


def _build_geometric_series(rate: float, n: int = 260, start: float = 100.0) -> list[float]:
    out = [start]
    for _ in range(1, n):
        out.append(out[-1] * (1.0 + rate))
    return out


def _drive_rotation(
    strategy: Any, symbols: list[str], series: dict[str, list[float]], n: int
) -> dict[str, str]:
    """Feed all symbols bar-by-bar; return the final non-flat holdings."""

    class _Tick(_Market):
        def __init__(self, time: str, symbol: str, close: float) -> None:
            super().__init__(time, symbol, close, close * 1.001, close * 0.999, close)

    for idx in range(n):
        for symbol in symbols:
            strategy.calculate_signals(_Tick(f"t{idx:04d}", symbol, series[symbol][idx]))
    return {
        symbol: strategy._state[symbol].mode
        for symbol in symbols
        if strategy._state[symbol].mode != "OUT"
    }


# --------------------------------------------------------------------------- #
# Registry presence
# --------------------------------------------------------------------------- #
def test_new_equity_strategies_register_after_lazy_discovery() -> None:
    discovered = set(get_strategy_names())
    for strategy_name in NEW_EQUITY_STRATEGIES:
        assert strategy_name in discovered, strategy_name
        schema = get_strategy_param_schema(strategy_name)
        assert schema, strategy_name
        defaults = get_default_strategy_params(strategy_name)
        assert defaults, strategy_name
        assert set(defaults).issubset(schema)


def test_leveraged_trend_timer_schema_is_long_only_with_decay_controls() -> None:
    schema = LeveragedTrendTimingRiderStrategy.get_param_schema()
    # Long-only: allow_short default False and grid only False.
    assert schema["allow_short"].default is False
    assert list(schema["allow_short"].grid or []) == [False]
    # Trend-filter + decay-aware sizing controls present.
    for key in (
        "regime_sma_bars",
        "fast_sma_bars",
        "slow_sma_bars",
        "decay_vol_ref",
        "decay_floor",
    ):
        assert key in schema, key
    # Trailing-stop ride lever + meaningful allocation present.
    assert schema["trail_atr_mult"].default >= 0.1
    assert schema["target_allocation"].default >= 0.10


# --------------------------------------------------------------------------- #
# S-EQ1 — single-name equity trend rider (reused class, long-only ride)
# --------------------------------------------------------------------------- #
def test_equity_single_name_rider_rides_long_winner_without_shorting() -> None:
    # Reuse the AdaptiveTrendRiderStrategy class on an equity-named symbol with the
    # equity-tuned long-only params from the S-EQ1 builder.
    from lumina_quant.strategies.return_rider_alpha_sleeves import AdaptiveTrendRiderStrategy

    symbol = "NVDA/USDT"
    closes = _leveraged_trend_series()
    strategy = AdaptiveTrendRiderStrategy(
        _Bars([symbol]),
        _Events(),
        kama_period=20,
        kama_slow=40,
        min_efficiency=0.35,
        trail_atr_mult=3.5,
        max_adds=3,
        target_vol=0.025,
        max_hold_bars=252,
        allow_short=False,
    )
    summary = _run_single_asset(strategy, symbol, closes)
    assert summary["entries"] >= 1, summary
    assert summary["entry_direction"] == "LONG", summary
    assert summary["shorts"] == 0, summary
    assert summary["max_position_run"] > 10, summary


# --------------------------------------------------------------------------- #
# S-EQ2 — leveraged-ETF trend timer
# --------------------------------------------------------------------------- #
def test_leveraged_trend_timer_enters_above_trend_rides_and_exits_below() -> None:
    symbol = "SOXL/USDT"
    closes = _leveraged_trend_series()
    strategy = LeveragedTrendTimingRiderStrategy(
        _Bars([symbol]),
        _Events(),
        regime_sma_bars=200,
        fast_sma_bars=50,
        slow_sma_bars=200,
        confirm_bars=3,
        max_hold_bars=400,
    )
    summary = _run_single_asset(strategy, symbol, closes)
    # Enters LONG only (never shorts) once above SMA200 + golden cross.
    assert summary["entries"] >= 1, summary
    assert summary["entry_direction"] == "LONG", summary
    assert summary["shorts"] == 0, summary
    # Rides the move for many bars with the trailing stop.
    assert summary["max_position_run"] > 10, summary
    # Exits once the trend filter is lost or the trailing stop is breached.
    assert (summary["trend_filter_exits"] + summary["trailing_stop_exits"]) >= 1, summary


def test_leveraged_trend_timer_stays_flat_without_trend() -> None:
    # A pure choppy/sideways series never satisfies the regime + golden cross gate.
    symbol = "URNM/USDT"
    closes: list[float] = []
    price = 100.0
    for idx in range(360):
        price *= 1.0 + (0.002 if idx % 2 else -0.002)
        closes.append(price)
    strategy = LeveragedTrendTimingRiderStrategy(
        _Bars([symbol]),
        _Events(),
        regime_sma_bars=200,
        fast_sma_bars=50,
        slow_sma_bars=200,
        confirm_bars=3,
    )
    summary = _run_single_asset(strategy, symbol, closes)
    assert summary["entries"] == 0, summary
    assert summary["shorts"] == 0, summary


def test_leveraged_trend_timer_decay_penalty_shrinks_high_vol_allocation() -> None:
    strategy = LeveragedTrendTimingRiderStrategy(
        _Bars(["SOXL/USDT"]),
        _Events(),
        target_vol=0.0,  # isolate the decay penalty from the vol-target scaling
        target_allocation=0.30,
        vol_window=20,
        decay_vol_ref=0.02,
        decay_floor=0.25,
    )
    calm = _build_geometric_series(0.0005, n=60)
    wild = [100.0]
    for idx in range(1, 60):
        wild.append(wild[-1] * (1.0 + (0.08 if idx % 2 else -0.075)))
    calm_alloc = strategy._vol_scaled_allocation(calm)
    wild_alloc = strategy._vol_scaled_allocation(wild)
    # High realized volatility must shrink the allocation via the decay penalty.
    assert wild_alloc < calm_alloc, (calm_alloc, wild_alloc)
    assert wild_alloc >= 0.0


# --------------------------------------------------------------------------- #
# S-EQ3 — dual-momentum with defensive rotation
# --------------------------------------------------------------------------- #
def test_dual_momentum_defensive_rotates_into_defensive_on_risk_off() -> None:
    rotation = ["SPY/USDT", "QQQ/USDT", "IWM/USDT", "EWY/USDT"]
    defensive = ["XAU/USDT", "XLE/USDT", "UVXY/USDT"]
    symbols = rotation + defensive
    n = 260
    series = {symbol: _build_geometric_series(-0.004, n=n) for symbol in rotation}
    series["XAU/USDT"] = _build_geometric_series(0.005, n=n)
    series["XLE/USDT"] = _build_geometric_series(0.001, n=n)
    series["UVXY/USDT"] = _build_geometric_series(-0.002, n=n)
    strategy = DualMomentumDefensiveRotationStrategy(
        _Bars(symbols),
        _Events(),
        benchmark_symbol="SPY/USDT",
        defensive_symbols=",".join(defensive),
        rebalance_bars=21,
        sma_bars=200,
        absolute_lookback_bars=12,
        min_symbols=4,
        max_holdings=3,
    )
    holdings = _drive_rotation(strategy, symbols, series, n)
    assert holdings, "expected a non-flat defensive holding on risk-off"
    assert all(symbol in defensive for symbol in holdings), holdings
    assert not any(symbol in rotation for symbol in holdings), holdings
    # All holdings are LONG (long defensive, not flat, not short).
    assert all(mode == "LONG" for mode in holdings.values()), holdings


def test_dual_momentum_defensive_holds_strongest_equity_on_risk_on() -> None:
    rotation = ["SPY/USDT", "QQQ/USDT", "IWM/USDT", "EWY/USDT"]
    defensive = ["XAU/USDT", "XLE/USDT", "UVXY/USDT"]
    symbols = rotation + defensive
    n = 260
    rates = {
        "SPY/USDT": 0.004,
        "QQQ/USDT": 0.008,
        "IWM/USDT": 0.003,
        "EWY/USDT": 0.002,
        "XAU/USDT": 0.001,
        "XLE/USDT": 0.0005,
        "UVXY/USDT": -0.003,
    }
    series = {symbol: _build_geometric_series(rates[symbol], n=n) for symbol in symbols}
    strategy = DualMomentumDefensiveRotationStrategy(
        _Bars(symbols),
        _Events(),
        benchmark_symbol="SPY/USDT",
        defensive_symbols=",".join(defensive),
        rebalance_bars=21,
        sma_bars=200,
        absolute_lookback_bars=12,
        min_symbols=4,
        max_holdings=3,
    )
    holdings = _drive_rotation(strategy, symbols, series, n)
    assert "QQQ/USDT" in holdings, holdings  # strongest equity ETF is held
    assert not any(symbol in defensive for symbol in holdings), holdings
    assert all(mode == "LONG" for mode in holdings.values()), holdings


# --------------------------------------------------------------------------- #
# Candidate wiring + de-leak
# --------------------------------------------------------------------------- #
def _mixed_rows() -> list[Any]:
    return build_binance_futures_candidates(
        timeframes=["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        symbols=_MIXED_UNIVERSE,
    )


def test_equity_single_name_rider_targets_equity_single_asset_only() -> None:
    rows = [r for r in _mixed_rows() if r.name.startswith("equity_single_name_trend_rider_")]
    assert rows, "S-EQ1 builder emitted no candidates"
    targeted: set[str] = set()
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert len(row.symbols) == 1, row.name
        assert row.strategy_class == "AdaptiveTrendRiderStrategy", row.name
        assert row.family == "trend", (row.name, row.family)
        assert row.timeframe in {"4h", "1d"}, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, row.timeframe
        assert "equity" in row.tags and "return_rider" in row.tags, row.tags
        assert row.params["allow_short"] is False, row.name
        targeted.update(row.symbols)
    assert targeted, targeted
    assert targeted.issubset(set(_EQUITY_SYMBOLS)), targeted
    assert targeted.isdisjoint(set(_CRYPTO_SYMBOLS)), targeted


def test_leveraged_trend_timer_targets_leveraged_etf_single_asset_only() -> None:
    rows = [r for r in _mixed_rows() if r.strategy_class == "LeveragedTrendTimingRiderStrategy"]
    assert rows, "S-EQ2 builder emitted no candidates"
    targeted: set[str] = set()
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert len(row.symbols) == 1, row.name
        assert row.family == "trend", (row.name, row.family)
        assert row.timeframe in {"4h", "1d"}, (row.name, row.timeframe)
        assert "leveraged_etf" in row.tags, row.tags
        assert row.params["allow_short"] is False, row.name
        targeted.update(row.symbols)
    assert targeted.issubset(set(_LEVERAGED_ETF_SYMBOLS)), targeted
    assert targeted.isdisjoint(set(_CRYPTO_SYMBOLS)), targeted


def test_dual_momentum_defensive_targets_index_basket_with_defensive_leg() -> None:
    rows = [r for r in _mixed_rows() if r.strategy_class == "DualMomentumDefensiveRotationStrategy"]
    assert rows, "S-EQ3 builder emitted no candidates"
    admission = {"cross_sectional", "carry", "momentum"}
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "multi", row.name
        assert row.family == "cross_sectional", (row.name, row.family)
        assert row.timeframe == "1d", (row.name, row.timeframe)
        assert admission.issubset(set(row.tags)), row.tags
        assert "defensive" in row.tags, row.tags
        symbol_set = set(row.symbols)
        # No crypto leaks into the rotation basket.
        assert symbol_set.isdisjoint(set(_CRYPTO_SYMBOLS)), symbol_set
        # The defensive leg is wired into the basket and the params.
        assert symbol_set & set(_DEFENSIVE_SYMBOLS), symbol_set
        defensive_param = set(str(row.params.get("defensive_symbols", "")).split(","))
        assert defensive_param & set(_DEFENSIVE_SYMBOLS), row.params


def test_new_equity_builders_only_emit_allowed_timeframes() -> None:
    new_classes = {
        "LeveragedTrendTimingRiderStrategy",
        "DualMomentumDefensiveRotationStrategy",
    }
    for row in _mixed_rows():
        if row.strategy_class in new_classes or row.name.startswith(
            "equity_single_name_trend_rider_"
        ):
            assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
            assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)


def test_crypto_riders_do_not_leak_equity_or_etf_symbols() -> None:
    # The three crypto rider builders are identified by candidate-name prefix
    # (S-EQ1 deliberately reuses AdaptiveTrendRiderStrategy on equity symbols, so a
    # by-class check would be ambiguous).
    crypto_prefixes = (
        "adaptive_trend_rider_",
        "volatility_breakout_rider_",
        "acceleration_rider_",
    )
    rows = _mixed_rows()
    crypto_rows = [r for r in rows if r.name.startswith(crypto_prefixes)]
    assert crypto_rows, "expected crypto rider candidates on the mixed universe"
    crypto_targeted: set[str] = set()
    for row in crypto_rows:
        crypto_targeted.update(row.symbols)
        # De-leak: no crypto rider may target any equity/ETF/defensive perp.
        assert set(row.symbols).isdisjoint(_EQUITY_ETF_DEFENSIVE), (row.name, row.symbols)
        assert "crypto" in row.tags, row.tags
    # Crypto riders still cover the crypto symbols.
    assert crypto_targeted.issubset(set(_CRYPTO_SYMBOLS)), crypto_targeted
    assert crypto_targeted, crypto_targeted
