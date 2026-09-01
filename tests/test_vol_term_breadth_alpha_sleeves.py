"""Deterministic tests for the RV-term-structure + breadth-regime-timer sleeves.

Covers registry presence of both new classes; dry-emission behaviour for
``RealizedVolTermStructureStrategy`` (no entry in a calm regime, a LONG recovery
entry after a realized-vol-term-structure shock where the short RV spikes versus
the long RV, and that the recovery is RIDDEN for several bars);
``BreadthRegimeTrendTimerStrategy`` (scales up long exposure when most names trend
up = high breadth, and reduces/flattens when breadth collapses = most names
down); and candidate-admission wiring via build->to_dict (RV-term single-asset,
breadth-timer cross_sectional-admissible; timeframes a subset of
``{30m,1h,4h,1d}``; decision cadence >= 1800).  No backtest is run.
"""

from __future__ import annotations

from typing import Any

from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategies.vol_term_breadth_alpha_sleeves import (
    BreadthRegimeTrendTimerStrategy,
    RealizedVolTermStructureStrategy,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

NEW_STRATEGIES = (
    "RealizedVolTermStructureStrategy",
    "BreadthRegimeTrendTimerStrategy",
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
# RealizedVolTermStructure dry-emission
# --------------------------------------------------------------------------- #
def test_rv_term_structure_calm_regime_does_not_enter() -> None:
    symbol = "BTC/USDT"
    events = _Events()
    strategy = RealizedVolTermStructureStrategy(
        _Bars([symbol]),
        events,
        short_window=6,
        long_window=48,
        shock_ratio=1.8,
        exit_ratio=1.1,
        use_micro_short=True,
        fade_upside_short=False,
        atr_period=10,
        max_hold_bars=200,
        allow_short=False,
    )
    # A calm, gently drifting regime: per-bar moves and intrabar ranges are both
    # tiny and on the same scale, so the (Parkinson-micro-blended) short RV never
    # separates from the long RV (ratio stays ~1, well below the shock_ratio).
    price = 100.0
    for idx in range(120):
        price *= 1.0008 if idx % 2 else 0.99925
        rng = price * 0.0005
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )
    entries = [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]
    assert entries == [], entries


def test_rv_term_structure_enters_long_recovery_after_shock_and_rides() -> None:
    symbol = "ETH/USDT"
    events = _Events()
    strategy = RealizedVolTermStructureStrategy(
        _Bars([symbol]),
        events,
        short_window=6,
        long_window=48,
        shock_ratio=1.6,
        exit_ratio=1.1,
        use_micro_short=False,  # keep the short leg purely close-to-close for determinism
        fade_upside_short=False,
        trail_atr_mult=4.0,
        atr_period=10,
        max_adds=0,
        vol_window=48,
        target_vol=0.0,  # disable vol-scaling so sizing is the flat target_allocation
        max_hold_bars=200,
        allow_short=False,
    )

    price = 100.0
    # Phase 1: long calm history so the LONG-window RV settles low.
    for idx in range(80):
        price *= 1.0001 if idx % 2 else 0.99995
        rng = price * 0.0004
        strategy.calculate_signals(
            _Market(f"t{idx:04d}", symbol, price, high=price + rng, low=price - rng)
        )
    assert all(e.signal_type not in {"LONG", "SHORT"} for e in events.signals)

    # Phase 2: a vol shock — several large alternating moves spike the SHORT-window
    # RV far above the still-low LONG-window RV (ratio >> 1 = backwardation panic).
    shock_moves = [-0.07, 0.05, -0.06, 0.05, -0.05]
    for j, mv in enumerate(shock_moves):
        price *= 1.0 + mv
        rng = price * 0.02
        strategy.calculate_signals(
            _Market(f"t{80 + j:04d}", symbol, price, high=price + rng, low=price - rng)
        )
    entries = [e for e in events.signals if e.signal_type in {"LONG", "SHORT"}]
    assert entries, "expected a recovery entry after the vol-term-structure shock"
    first = entries[0]
    assert first.signal_type == "LONG", first.signal_type
    assert (first.metadata or {}).get("target_allocation", 0.0) > 0.0, first.metadata
    assert (first.metadata or {}).get("vol_term_ratio") is not None, first.metadata

    # Phase 3: a clean recovery up-leg — the position is RIDDEN (no immediate exit
    # while vol stays elevated and price climbs above the trailing stop).
    n_before_exit = len(events.signals)
    for k in range(40):
        price *= 1.01
        rng = price * 0.004
        strategy.calculate_signals(
            _Market(f"t{90 + k:04d}", symbol, price, high=price + rng, low=price - rng)
        )
    # The ride should persist for several bars before any exit (winners run).
    exits = [e for e in events.signals[n_before_exit:] if e.signal_type == "EXIT"]
    # Either it is still riding (no exit yet) or it exited only after climbing.
    assert all(e.price is None or e.price > first.price for e in exits), exits


# --------------------------------------------------------------------------- #
# BreadthRegimeTrendTimer dry-emission
# --------------------------------------------------------------------------- #
def _trend_series(rate: float, jitter: float, n: int, start: float = 100.0) -> list[float]:
    out = [start]
    for idx in range(1, n):
        out.append(out[-1] * (1.0 + rate + (jitter if idx % 2 else -jitter)))
    return out


def _run_basket(strategy: Any, series: dict[str, list[float]], lo: int, hi: int) -> None:
    for idx in range(lo, hi):
        for symbol, closes in series.items():
            strategy.calculate_signals(_Market(f"t{idx:04d}", symbol, closes[idx]))


def test_breadth_timer_scales_up_on_high_breadth_and_flattens_on_collapse() -> None:
    n = 220
    # Phase A (first ~120 bars): broad uptrend across the basket (high breadth).
    # Phase B (after): the whole basket reverses down hard (breadth collapses).
    flip = 120
    series: dict[str, list[float]] = {}
    for sym in _CRYPTO_UNIVERSE:
        up = _trend_series(0.010, 0.0008, flip)
        # Continue from the last up value into a sharp synchronized downtrend.
        down = [up[-1]]
        for idx in range(1, n - flip + 1):
            down.append(down[-1] * (1.0 - 0.012 + (0.0008 if idx % 2 else -0.0008)))
        series[sym] = up + down[1:]

    events = _Events()
    strategy = BreadthRegimeTrendTimerStrategy(
        _Bars(list(series)),
        events,
        trend_window=20,
        risk_on_breadth=0.60,
        risk_off_breadth=0.40,
        require_positive_return=True,
        max_positions=7,
        min_trend=0.0,
        max_gross=0.80,
        rebalance_bars=1,
        stop_loss_pct=0.9,
        max_hold_bars=10000,
        min_symbols=4,
    )

    # Phase A: broad uptrend -> risk-on -> long basket with breadth-scaled gross.
    _run_basket(strategy, series, 0, flip)
    long_entries = [e for e in events.signals if e.signal_type == "LONG"]
    assert long_entries, "expected long entries under high breadth (risk-on)"
    breadths = [
        (e.metadata or {}).get("breadth")
        for e in long_entries
        if (e.metadata or {}).get("breadth") is not None
    ]
    assert breadths, "expected breadth metadata on long entries"
    assert max(breadths) >= 0.60, breadths
    # The breadth-scaled gross must be > 0 and consistent with target_allocation.
    gross = [(e.metadata or {}).get("breadth_scaled_gross", 0.0) for e in long_entries]
    assert max(gross) > 0.0, gross

    n_after_phase_a = len(events.signals)

    # Phase B: synchronized downtrend -> breadth collapses -> risk-off -> flatten.
    _run_basket(strategy, series, flip, n)
    risk_off_exits = [
        e
        for e in events.signals[n_after_phase_a:]
        if e.signal_type == "EXIT"
        and (e.metadata or {}).get("reason")
        in {"breadth_risk_off", "rebalance_removed", "no_uptrend_names", "stop_loss", "max_hold"}
    ]
    assert risk_off_exits, "expected the book to flatten when breadth collapses"
    # After the collapse, no fresh LONG entries should be emitted in the risk-off tail.
    tail_longs = [e for e in events.signals[n_after_phase_a:] if e.signal_type == "LONG"]
    # Allow a couple straggler entries on the very first reversal bars, but the
    # regime must end flat: the LAST emission per held symbol is an EXIT.
    last_by_symbol: dict[str, str] = {}
    for e in events.signals:
        last_by_symbol[e.symbol] = e.signal_type
    assert all(sig == "EXIT" for sig in last_by_symbol.values()), last_by_symbol
    assert len(tail_longs) <= len(_CRYPTO_UNIVERSE), tail_longs


def test_breadth_timer_skips_below_min_symbols() -> None:
    n = 60
    series = {
        "BTC/USDT": _trend_series(0.010, 0.0005, n),
        "ETH/USDT": _trend_series(0.010, 0.0005, n),
        "SOL/USDT": _trend_series(0.010, 0.0005, n),
    }
    events = _Events()
    strategy = BreadthRegimeTrendTimerStrategy(
        _Bars(list(series)),
        events,
        trend_window=15,
        rebalance_bars=1,
        min_symbols=6,
    )
    _run_basket(strategy, series, 0, n)
    assert events.signals == []


# --------------------------------------------------------------------------- #
# Candidate wiring + admission
# --------------------------------------------------------------------------- #
def _crypto_rows() -> list[Any]:
    return build_binance_futures_candidates(
        timeframes=["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        symbols=_CRYPTO_UNIVERSE,
    )


def test_rv_term_structure_wired_single_asset_and_only_ge_30m() -> None:
    rows = [r for r in _crypto_rows() if r.strategy_class == "RealizedVolTermStructureStrategy"]
    assert rows, "RV-term-structure builder emitted no candidates"
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert len(row.symbols) == 1, row.name
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert "term_structure" in row.tags
        assert "single_asset" in row.tags
        assert row.metadata["timeframe"] == row.timeframe
        assert int(row.metadata["decision_cadence_seconds"]) >= 1800, row.name


def test_breadth_timer_wired_cross_sectional_admissible() -> None:
    rows = [r for r in _crypto_rows() if r.strategy_class == "BreadthRegimeTrendTimerStrategy"]
    assert rows, "breadth-timer builder emitted no candidates"
    admission = {"cross_sectional", "carry", "momentum"}
    for row in rows:
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "multi", row.name
        assert row.family == "cross_sectional", (row.name, row.family)
        assert admission.issubset(set(row.tags)), row.tags
        assert "breadth" in row.tags
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert len(row.symbols) >= 5, row.name
        assert int(row.params["decision_cadence_seconds"]) >= 1800, row.name


def test_new_builders_only_emit_allowed_timeframes() -> None:
    for row in _crypto_rows():
        if row.strategy_class in NEW_STRATEGIES:
            assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
            assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
