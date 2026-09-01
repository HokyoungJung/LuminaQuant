"""Deterministic unit tests for the selection-aware cross-sectional sleeves.

These tests exercise the two sleeves that compose the multi-factor target-pool
selector with a directional signal:

- registry presence for both classes;
- dry-emission with a synthetic universe wide enough that the selector yields a
  non-empty pool and each sleeve emits at least one non-EXIT signal carrying a
  positive ``target_allocation``;
- the selection gate: when only a handful of names are tradeable (the rest are
  dead-flat / illiquid), the sleeve only trades the screened ones;
- a graceful skip when the screened pool is smaller than ``min_symbols``;
- momentum-vs-reversion directionality on a constructed divergent universe;
- candidate-admission safety for both sleeves.

No backtest is run; everything is fed through the local event/bar contract.
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.market_window_contract import build_market_window_event
from lumina_quant.strategies.registry import get_strategy_map
from lumina_quant.strategies.selection_aware_alpha_sleeves import (
    SelectionGatedMomentumStrategy,
    SelectionGatedReversionStrategy,
)
from lumina_quant.strategy_factory.selection import (
    _allowlisted_portfolio_native_multi_asset_candidate,
    candidate_mix_type,
)


@dataclass(slots=True)
class _Bars:
    symbol_list: list[str]


def _window(ts: int, closes: dict[str, float], volumes: dict[str, float]) -> Any:
    bars_1s = {
        symbol: (
            (
                int(ts),
                float(close) * 0.999,
                float(close) * 1.01,
                float(close) * 0.99,
                float(close),
                float(volumes.get(symbol, 1000.0)),
            ),
        )
        for symbol, close in closes.items()
    }
    return build_market_window_event(
        time=ts,
        window_seconds=3600,
        bars_1s=bars_1s,
        event_time_watermark_ms=int(ts),
        commit_id=None,
        lag_ms=0,
        is_stale=False,
        emit_metrics=False,
    )


def _drain(events: queue.Queue) -> list[Any]:
    signals: list[Any] = []
    while not events.empty():
        signals.append(events.get())
    return signals


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


# ---------------------------------------------------------------------------
# (a) registry presence
# ---------------------------------------------------------------------------
def test_selection_aware_sleeves_are_registered() -> None:
    strategy_map = get_strategy_map()
    assert strategy_map["SelectionGatedMomentumStrategy"] is SelectionGatedMomentumStrategy
    assert strategy_map["SelectionGatedReversionStrategy"] is SelectionGatedReversionStrategy


# ---------------------------------------------------------------------------
# (b) dry-emission: non-empty pool -> at least one allocated non-EXIT signal
# ---------------------------------------------------------------------------
def _diverse_universe() -> list[str]:
    return [
        "AAA/USDT",
        "BBB/USDT",
        "CCC/USDT",
        "DDD/USDT",
        "EEE/USDT",
        "FFF/USDT",
    ]


def _run_momentum(symbols: list[str], n: int = 40) -> tuple[Any, list[Any]]:
    events: queue.Queue = queue.Queue()
    strat = SelectionGatedMomentumStrategy(
        _Bars(list(symbols)),
        events,
        momentum_lookback_bars=4,
        vol_window=4,
        rebalance_bars=4,
        history_window=40,
        pool_size=6,
        min_symbols=4,
        allow_short=False,
    )
    for ts in range(n):
        closes = {}
        volumes = {}
        for idx, symbol in enumerate(symbols):
            drift = (idx - 2) * 0.012
            closes[symbol] = 100.0 * (1.0 + drift) ** ts + math.sin(ts * 0.3 + idx) * 0.4
            volumes[symbol] = 1000.0 + 25.0 * ts + 7.0 * idx
        strat.calculate_signals(_window(ts, closes, volumes))
    return strat, _drain(events)


def test_momentum_dry_emission_yields_allocated_signal() -> None:
    _strat, signals = _run_momentum(_diverse_universe())
    non_exit = _non_exit(signals)
    assert non_exit, "momentum sleeve emitted no non-EXIT signals"
    allocated = [
        sig for sig in non_exit if float((sig.metadata or {}).get("target_allocation", 0.0)) > 0.0
    ]
    assert allocated, "no non-EXIT signal carried target_allocation>0"
    assert all((sig.metadata or {}).get("screened") is True for sig in allocated)


def test_reversion_dry_emission_yields_allocated_signal() -> None:
    symbols = _diverse_universe()
    events: queue.Queue = queue.Queue()
    strat = SelectionGatedReversionStrategy(
        _Bars(list(symbols)),
        events,
        reversion_lookback_bars=3,
        vol_window=4,
        rebalance_bars=4,
        history_window=40,
        pool_size=6,
        min_symbols=4,
        allow_short=False,
    )
    for ts in range(40):
        closes = {}
        volumes = {}
        for idx, symbol in enumerate(symbols):
            # Alternating zig-zag so a recent move exists to fade.
            base = 100.0 + idx * 2.0
            closes[symbol] = base * (1.0 + 0.02 * math.sin(ts * 0.9 + idx))
            volumes[symbol] = 1000.0 + 25.0 * ts + 7.0 * idx
        strat.calculate_signals(_window(ts, closes, volumes))
    signals = _drain(events)
    non_exit = _non_exit(signals)
    assert non_exit, "reversion sleeve emitted no non-EXIT signals"
    allocated = [
        sig for sig in non_exit if float((sig.metadata or {}).get("target_allocation", 0.0)) > 0.0
    ]
    assert allocated, "no non-EXIT signal carried target_allocation>0"


# ---------------------------------------------------------------------------
# (c) selection gate: only screened names trade; dead-flat/illiquid excluded
# ---------------------------------------------------------------------------
def test_selection_gate_only_trades_screened_names() -> None:
    tradeable = ["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT", "EEE/USDT"]
    dead = ["DEAD1/USDT", "DEAD2/USDT", "DEAD3/USDT"]
    symbols = [*tradeable, *dead]
    events: queue.Queue = queue.Queue()
    strat = SelectionGatedMomentumStrategy(
        _Bars(list(symbols)),
        events,
        momentum_lookback_bars=4,
        vol_window=4,
        rebalance_bars=4,
        history_window=40,
        pool_size=12,
        min_symbols=4,
        allow_short=False,
    )
    for ts in range(40):
        closes = {}
        volumes = {}
        for idx, symbol in enumerate(tradeable):
            closes[symbol] = 100.0 * (1.0 + 0.01 * (idx + 1)) ** ts + math.sin(ts * 0.3 + idx) * 0.3
            volumes[symbol] = 5000.0 + 30.0 * ts
        for symbol in dead:
            # Dead-flat and illiquid: zero volume, zero volatility -> gated out.
            closes[symbol] = 50.0
            volumes[symbol] = 0.0
        strat.calculate_signals(_window(ts, closes, volumes))
    signals = _drain(events)
    traded = {sig.symbol for sig in _non_exit(signals)}
    assert traded, "no symbol traded"
    assert traded.issubset(set(tradeable)), traded
    assert not (traded & set(dead)), traded


# ---------------------------------------------------------------------------
# (d) graceful skip when the screened pool is below min_symbols
# ---------------------------------------------------------------------------
def test_graceful_skip_when_pool_below_min_symbols() -> None:
    # Only two names ever carry usable data; the rest are dead-flat/illiquid.
    live = ["AAA/USDT", "BBB/USDT"]
    dead = ["DEAD1/USDT", "DEAD2/USDT", "DEAD3/USDT", "DEAD4/USDT"]
    symbols = [*live, *dead]
    events: queue.Queue = queue.Queue()
    strat = SelectionGatedMomentumStrategy(
        _Bars(list(symbols)),
        events,
        momentum_lookback_bars=4,
        vol_window=4,
        rebalance_bars=4,
        history_window=40,
        pool_size=12,
        min_symbols=4,
        allow_short=False,
    )
    for ts in range(40):
        closes = {}
        volumes = {}
        for idx, symbol in enumerate(live):
            closes[symbol] = 100.0 * (1.0 + 0.01 * (idx + 1)) ** ts
            volumes[symbol] = 5000.0 + 30.0 * ts
        for symbol in dead:
            closes[symbol] = 50.0
            volumes[symbol] = 0.0
        strat.calculate_signals(_window(ts, closes, volumes))
    signals = _drain(events)
    # Pool of tradeable names (<= 2) is below min_symbols (4): no entries open.
    assert not _non_exit(signals), [s.symbol for s in _non_exit(signals)]


# ---------------------------------------------------------------------------
# (e) momentum vs reversion directionality on a divergent universe
# ---------------------------------------------------------------------------
def _final_side_by_symbol(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def test_momentum_and_reversion_are_directionally_opposite() -> None:
    symbols = ["WIN/USDT", "MID1/USDT", "MID2/USDT", "MID3/USDT", "LOSE/USDT"]

    def feed(strat: Any, events: queue.Queue) -> dict[str, str]:
        for ts in range(40):
            closes = {}
            volumes = {}
            # Persistent monotone divergence: WIN trends up, LOSE trends down.
            ramp = {
                "WIN/USDT": 0.020,
                "MID1/USDT": 0.004,
                "MID2/USDT": 0.000,
                "MID3/USDT": -0.004,
                "LOSE/USDT": -0.020,
            }
            for idx, symbol in enumerate(symbols):
                # Persistent drift plus a small jitter so realized volatility is
                # finite (a pure geometric ramp has ~zero vol and is filtered).
                trend = 100.0 * (1.0 + ramp[symbol]) ** ts
                jitter = 1.0 + 0.004 * math.sin(ts * 0.7 + idx)
                closes[symbol] = trend * jitter
                volumes[symbol] = 5000.0 + 30.0 * ts
            strat.calculate_signals(_window(ts, closes, volumes))
        return _final_side_by_symbol(_drain(events))

    mom_events: queue.Queue = queue.Queue()
    mom = SelectionGatedMomentumStrategy(
        _Bars(list(symbols)),
        mom_events,
        momentum_lookback_bars=6,
        vol_window=4,
        rebalance_bars=4,
        history_window=40,
        pool_size=8,
        min_symbols=4,
        top_fraction=0.20,
        allow_short=True,
    )
    rev_events: queue.Queue = queue.Queue()
    rev = SelectionGatedReversionStrategy(
        _Bars(list(symbols)),
        rev_events,
        reversion_lookback_bars=6,
        vol_window=4,
        rebalance_bars=4,
        history_window=40,
        pool_size=8,
        min_symbols=4,
        top_fraction=0.20,
        allow_short=True,
    )

    mom_side = feed(mom, mom_events)
    rev_side = feed(rev, rev_events)

    # Momentum chases the trend: the steady winner is long, the loser is short.
    assert mom_side.get("WIN/USDT") == "LONG", mom_side
    assert mom_side.get("LOSE/USDT") == "SHORT", mom_side
    # Reversion fades the trend: opposite sign on the same extreme names.
    assert rev_side.get("WIN/USDT") == "SHORT", rev_side
    assert rev_side.get("LOSE/USDT") == "LONG", rev_side
    # And the two sleeves disagree on at least the trend extremes.
    assert mom_side.get("WIN/USDT") != rev_side.get("WIN/USDT")
    assert mom_side.get("LOSE/USDT") != rev_side.get("LOSE/USDT")


# ---------------------------------------------------------------------------
# (f) candidate-admission: built rows survive the default shortlist policy
# ---------------------------------------------------------------------------
def test_candidate_admission_for_selection_aware_sleeves(monkeypatch) -> None:
    from lumina_quant.strategy_factory import build_binance_futures_candidates, candidate_library

    monkeypatch.setattr(candidate_library, "_has_perp_support_data", lambda: True)

    symbols = [
        "BTC/USDT",
        "ETH/USDT",
        "BNB/USDT",
        "SOL/USDT",
        "TRX/USDT",
        "XRP/USDT",
        "ADA/USDT",
        "DOGE/USDT",
    ]
    rows = build_binance_futures_candidates(
        timeframes=["15m", "30m", "1h", "4h", "1d"],
        symbols=symbols,
    )

    required_multi_tags = {"cross_sectional", "carry", "momentum"}
    for strategy_class, builder_name in (
        ("SelectionGatedMomentumStrategy", "_build_selection_gated_momentum_candidates"),
        ("SelectionGatedReversionStrategy", "_build_selection_gated_reversion_candidates"),
    ):
        assert hasattr(candidate_library, builder_name), builder_name
        matches = [row for row in rows if row.strategy_class == strategy_class]
        assert matches, strategy_class
        for row in matches:
            payload = row.to_dict()
            mix = candidate_mix_type(payload)
            assert mix == "single" or _allowlisted_portfolio_native_multi_asset_candidate(
                payload
            ), (
                strategy_class,
                mix,
            )
            if len(row.symbols) >= 3:
                assert row.family == "cross_sectional"
                assert required_multi_tags.issubset(set(row.tags))
            assert row.metadata["timeframe"] == row.timeframe
