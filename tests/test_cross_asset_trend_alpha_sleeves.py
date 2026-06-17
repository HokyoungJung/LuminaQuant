"""Deterministic tests for the cross-asset trend + intermarket lead-lag sleeves.

Covers registry presence of both new classes; dry-emission behaviour for
``CrossAssetDiversifiedTrendStrategy`` (longs up-trenders, shorts down-trenders
across a mixed synthetic universe, inverse-vol weights the higher-vol name
smaller, emits >=1 non-EXIT with ``target_allocation>0``, graceful skip under
``min_symbols``); ``IntermarketLeadLagContinuationStrategy`` (enters the follower
in the leader's direction after a leader move and NOT before, exits on
catch-up); and candidate-admission wiring via build->to_dict (both
cross_sectional-admissible, timeframes a subset of ``{30m,1h,4h,1d}``).  No
backtest is run.
"""

from __future__ import annotations

from typing import Any

from lumina_quant.strategies.cross_asset_trend_alpha_sleeves import (
    CrossAssetDiversifiedTrendStrategy,
    IntermarketLeadLagContinuationStrategy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_CROSS_ASSET_STRATEGIES = (
    "CrossAssetDiversifiedTrendStrategy",
    "IntermarketLeadLagContinuationStrategy",
)

_ALLOWED_TIMEFRAMES = {"30m", "1h", "4h", "1d"}
_FORBIDDEN_TIMEFRAMES = {"1s", "1m", "5m", "15m"}

_MIXED_UNIVERSE = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
    "SPY/USDT",
    "QQQ/USDT",
    "NVDA/USDT",
    "AMD/USDT",
    "CL/USDT",
    "BZ/USDT",
    "COPPER/USDT",
    "XLE/USDT",
]


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

    def __init__(self, time: str, symbol: str, close: float) -> None:
        self.time = time
        self.symbol = symbol
        self.open = close
        self.high = close * 1.001
        self.low = close * 0.999
        self.close = close
        self.volume = 1_000.0


# --------------------------------------------------------------------------- #
# Registry presence
# --------------------------------------------------------------------------- #
def test_new_cross_asset_strategies_register_after_lazy_discovery() -> None:
    discovered = set(get_strategy_names())
    for strategy_name in NEW_CROSS_ASSET_STRATEGIES:
        assert strategy_name in discovered, strategy_name
        schema = get_strategy_param_schema(strategy_name)
        assert schema, strategy_name
        defaults = get_default_strategy_params(strategy_name)
        assert defaults, strategy_name
        assert set(defaults).issubset(schema)


# --------------------------------------------------------------------------- #
# CrossAssetDiversifiedTrend dry-emission
# --------------------------------------------------------------------------- #
def _run_cross_asset(strategy: Any, series: dict[str, list[float]], n: int) -> dict[str, Any]:
    events: _Events = strategy.events
    for idx in range(n):
        for symbol, closes in series.items():
            strategy.calculate_signals(_Market(f"t{idx:04d}", symbol, closes[idx]))
    last: dict[str, tuple[str, float | None, float | None]] = {}
    for event in events.signals:
        meta = event.metadata or {}
        last[event.symbol] = (
            event.signal_type,
            meta.get("inverse_vol_weight"),
            meta.get("target_allocation"),
        )
    return last


def _trend_series(rate: float, jitter: float, n: int, start: float = 100.0) -> list[float]:
    out = [start]
    for idx in range(1, n):
        out.append(out[-1] * (1.0 + rate + (jitter if idx % 2 else -jitter)))
    return out


def test_cross_asset_trend_longs_up_shorts_down_and_inverse_vol_weights() -> None:
    n = 130
    series = {
        "UP_LOWVOL": _trend_series(0.010, 0.0005, n),  # up, low realized vol
        "UP_HIGHVOL": _trend_series(0.010, 0.040, n),  # up, high realized vol
        "DOWN": _trend_series(-0.010, 0.001, n),  # down-trender
        "FLAT": _trend_series(0.000, 0.001, n),  # below min_trend gate
        "UP2": _trend_series(0.009, 0.0006, n),
        "DOWN2": _trend_series(-0.009, 0.001, n),
        "FLAT2": _trend_series(0.000, 0.0008, n),
    }
    events = _Events()
    strategy = CrossAssetDiversifiedTrendStrategy(
        _Bars(list(series)),
        events,
        trend_lookback=20,
        vol_window=20,
        min_trend=0.02,
        rebalance_bars=1,
        target_vol=0.05,
        rebalance_band=0.0,
        confirm_sma_slope=False,
        allow_short=True,
        min_symbols=4,
    )
    last = _run_cross_asset(strategy, series, n)
    longs = {sym for sym, row in last.items() if row[0] == "LONG"}
    shorts = {sym for sym, row in last.items() if row[0] == "SHORT"}

    # Up-trenders go long, down-trenders go short.
    assert longs & {"UP_LOWVOL", "UP2"}, last
    assert shorts & {"DOWN", "DOWN2"}, last
    # The flat names never clear the min_trend gate and stay flat.
    assert "FLAT" not in longs and "FLAT" not in shorts, last

    # Inverse-vol risk parity: the lower-vol up-trender gets the larger weight.
    if last.get("UP_LOWVOL", (None,))[0] == "LONG" and last.get("UP_HIGHVOL", (None,))[0] == "LONG":
        assert last["UP_LOWVOL"][1] > last["UP_HIGHVOL"][1], last

    # At least one non-EXIT emission carries a positive target_allocation.
    non_exit_alloc = [
        e
        for e in events.signals
        if e.signal_type != "EXIT" and (e.metadata or {}).get("target_allocation", 0.0) > 0.0
    ]
    assert non_exit_alloc, "expected >=1 non-EXIT signal with target_allocation>0"


def test_cross_asset_trend_band_does_not_churn_held_names() -> None:
    # Regression: rebalance_band>0 must NOT EXIT->re-ENTER a same-side held name
    # every bar. All names are strong persistent trenders within max_positions, so
    # once entered none should be dropped with reason='rebalance_removed', and each
    # symbol should enter only a handful of times (not once per bar).
    n = 130
    series = {
        "UPA": _trend_series(0.012, 0.0006, n),
        "UPB": _trend_series(0.011, 0.0008, n),
        "UPC": _trend_series(0.010, 0.0010, n),
        "DNA": _trend_series(-0.012, 0.0006, n),
        "DNB": _trend_series(-0.011, 0.0008, n),
        "DNC": _trend_series(-0.010, 0.0010, n),
    }
    events = _Events()
    strategy = CrossAssetDiversifiedTrendStrategy(
        _Bars(list(series)),
        events,
        trend_lookback=20,
        vol_window=20,
        min_trend=0.02,
        rebalance_bars=1,
        target_vol=0.05,
        rebalance_band=0.25,  # the previously-buggy banded path
        confirm_sma_slope=False,
        allow_short=True,
        max_positions=6,
        max_hold_bars=2000,
        stop_loss_pct=0.9,
        min_symbols=4,
    )
    _run_cross_asset(strategy, series, n)
    removed = [
        e
        for e in events.signals
        if e.signal_type == "EXIT" and (e.metadata or {}).get("reason") == "rebalance_removed"
    ]
    entries_per_symbol: dict[str, int] = {}
    for e in events.signals:
        if e.signal_type in {"LONG", "SHORT"}:
            entries_per_symbol[e.symbol] = entries_per_symbol.get(e.symbol, 0) + 1
    assert removed == [], (
        f"banded rebalance churned held names: {len(removed)} rebalance_removed EXITs"
    )
    assert max(entries_per_symbol.values(), default=0) <= 3, entries_per_symbol


def test_cross_asset_trend_skips_below_min_symbols() -> None:
    n = 90
    series = {
        "A": _trend_series(0.010, 0.0005, n),
        "B": _trend_series(0.010, 0.0005, n),
        "C": _trend_series(-0.010, 0.001, n),
    }
    events = _Events()
    strategy = CrossAssetDiversifiedTrendStrategy(
        _Bars(list(series)),
        events,
        trend_lookback=20,
        vol_window=20,
        min_trend=0.02,
        rebalance_bars=1,
        min_symbols=6,
        confirm_sma_slope=False,
    )
    _run_cross_asset(strategy, series, n)
    # Universe (3) < min_symbols (6): graceful skip, no emissions.
    assert events.signals == []


# --------------------------------------------------------------------------- #
# IntermarketLeadLagContinuation dry-emission
# --------------------------------------------------------------------------- #
def test_intermarket_lead_lag_enters_follower_after_leader_and_exits_on_catch_up() -> None:
    leader = "CL/USDT"
    follower = "XLE/USDT"
    events = _Events()
    strategy = IntermarketLeadLagContinuationStrategy(
        _Bars([leader, follower]),
        events,
        pair_spec="CL/USDT>XLE/USDT",
        lead_lookback=5,
        follow_lookback=3,
        z_window=30,
        entry_z=1.2,
        min_leader_move=0.01,
        max_follower_move=0.01,
        catch_up_fraction=0.6,
        allow_short=True,
        stop_loss_pct=0.20,
        max_hold_bars=50,
    )

    cl = 100.0
    xle = 50.0
    # Phase 1: both roughly flat to seed the leader-return z-window.
    for idx in range(40):
        step = 0.0005 if idx % 2 else -0.0005
        cl *= 1.0 + step
        xle *= 1.0 + step
        strategy.calculate_signals(_Market(f"t{idx:04d}", leader, cl))
        strategy.calculate_signals(_Market(f"t{idx:04d}", follower, xle))
    # No entry may precede the leader impulse.
    pre_entries = [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]
    assert pre_entries == [], pre_entries

    # Phase 2: the leader jumps hard while the follower stays flat (no chase).
    for idx in range(40, 48):
        cl *= 1.03
        xle *= 1.0005
        strategy.calculate_signals(_Market(f"t{idx:04d}", leader, cl))
        strategy.calculate_signals(_Market(f"t{idx:04d}", follower, xle))
    entries = [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]
    assert entries, "expected a follower entry after the leader move"
    first = entries[0]
    assert first.symbol == follower, first.symbol
    assert first.signal_type == "LONG", first.signal_type
    assert (first.metadata or {}).get("reason") == "leader_continuation", first.metadata
    assert (first.metadata or {}).get("target_allocation", 0.0) > 0.0, first.metadata

    entry_count = len(events.signals)
    # Phase 3: the follower catches up (delayed reaction plays out) -> exit.
    for idx in range(48, 60):
        cl *= 1.0005
        xle *= 1.03
        strategy.calculate_signals(_Market(f"t{idx:04d}", leader, cl))
        strategy.calculate_signals(_Market(f"t{idx:04d}", follower, xle))
    exits = [e for e in events.signals[entry_count:] if e.signal_type == "EXIT"]
    assert exits, "expected an exit once the follower catches up"
    assert any((e.metadata or {}).get("reason") == "follower_caught_up" for e in exits), exits


# --------------------------------------------------------------------------- #
# Candidate wiring + admission
# --------------------------------------------------------------------------- #
def _mixed_rows() -> list[Any]:
    return build_binance_futures_candidates(
        timeframes=["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        symbols=_MIXED_UNIVERSE,
    )


def test_cross_asset_trend_wired_cross_sectional_admissible() -> None:
    rows = [r for r in _mixed_rows() if r.strategy_class == "CrossAssetDiversifiedTrendStrategy"]
    assert rows, "cross-asset trend builder emitted no candidates"
    admission = {"cross_sectional", "carry", "momentum"}
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "multi", row.name
        assert row.family == "cross_sectional", (row.name, row.family)
        assert admission.issubset(set(row.tags)), row.tags
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert row.metadata["timeframe"] == row.timeframe
        # The whole normalized universe is targeted (diversified cross-asset book).
        assert len(row.symbols) >= 6, row.name


def test_intermarket_lead_lag_wired_cross_sectional_admissible() -> None:
    rows = [
        r for r in _mixed_rows() if r.strategy_class == "IntermarketLeadLagContinuationStrategy"
    ]
    assert rows, "intermarket lead-lag builder emitted no candidates"
    admission = {"cross_sectional", "carry", "momentum"}
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "multi", row.name
        assert row.family == "cross_sectional", (row.name, row.family)
        assert admission.issubset(set(row.tags)), row.tags
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        # Only the commodity leader + energy ETF follower legs are targeted.
        symbol_set = set(row.symbols)
        assert "XLE/USDT" in symbol_set, symbol_set
        assert symbol_set & {"CL/USDT", "BZ/USDT"}, symbol_set
        spec = str(row.params.get("pair_spec", ""))
        assert ">" in spec and "XLE/USDT" in spec, spec


def test_new_cross_asset_builders_only_emit_allowed_timeframes() -> None:
    for row in _mixed_rows():
        if row.strategy_class in NEW_CROSS_ASSET_STRATEGIES:
            assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
            assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
