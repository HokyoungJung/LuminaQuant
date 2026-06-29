"""Tests for BullBearRegimeRotationStrategy.

Covers registry discovery, deterministic bull/bear regime entries, neutral
flattening, None/empty safety, and state roundtrip.  These tests do not run a
backtest; they validate the event-driven contract before data-machine WF runs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from lumina_quant.strategies.bull_bear_regime_rotation import BullBearRegimeRotationStrategy
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_metadata,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

_START = datetime(2026, 1, 1, tzinfo=UTC)
_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT"]


class _BarsNoFeatures:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Events:
    def __init__(self) -> None:
        self.signals: list[Any] = []

    def put(self, event: Any) -> None:
        self.signals.append(event)


class _Window:
    type = "MARKET_WINDOW"

    def __init__(self, time: str, prices: dict[str, float]) -> None:
        self.time = time
        self.bars_1s = {
            symbol: [
                {
                    "time": time,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 1_000_000.0,
                }
            ]
            for symbol, price in prices.items()
        }


class _EmptyWindow:
    type = "MARKET_WINDOW"
    bars_1s: dict[str, list[Any]] = {}
    time = _START.isoformat()


def _ts(i: int, *, step_minutes: int = 30) -> str:
    return (_START + timedelta(minutes=step_minutes * i)).isoformat()


def _strategy(events: _Events) -> BullBearRegimeRotationStrategy:
    return BullBearRegimeRotationStrategy(
        _BarsNoFeatures(_SYMBOLS),
        events,
        momentum_lookback=5,
        trend_ma_window=5,
        signal_threshold=0.01,
        bull_breadth=0.60,
        bear_breadth=0.60,
        exit_breadth=0.35,
        benchmark_lookback=5,
        benchmark_bull_threshold=0.005,
        benchmark_bear_threshold=0.005,
        max_longs=2,
        max_shorts=2,
        rebalance_bars=1,
        stop_loss_pct=0.20,
        max_hold_bars=10_000,
        min_symbols=5,
        target_allocation=0.60,
        max_order_value=1000.0,
    )


def _feed_windows(strategy: BullBearRegimeRotationStrategy, prices_by_step: list[dict[str, float]]) -> None:
    for i, prices in enumerate(prices_by_step):
        strategy.calculate_signals(_Window(_ts(i), prices))


def _trend_path(mults: dict[str, float], *, steps: int = 12, start: float = 100.0) -> list[dict[str, float]]:
    prices = dict.fromkeys(_SYMBOLS, start)
    out: list[dict[str, float]] = []
    for _ in range(steps):
        for symbol in _SYMBOLS:
            prices[symbol] *= mults[symbol]
        out.append(dict(prices))
    return out


def _entries(events: _Events) -> list[Any]:
    return [event for event in events.signals if event.signal_type in {"LONG", "SHORT"}]


def test_bull_bear_regime_rotation_registered_as_research_only() -> None:
    assert "BullBearRegimeRotationStrategy" in set(get_strategy_names())
    assert get_strategy_param_schema("BullBearRegimeRotationStrategy")
    assert get_default_strategy_params("BullBearRegimeRotationStrategy")
    assert get_strategy_metadata("BullBearRegimeRotationStrategy")["tier"] == "research_only"


def test_bull_regime_enters_strongest_long_names() -> None:
    events = _Events()
    strategy = _strategy(events)
    mults = {
        "BTC/USDT": 1.010,
        "ETH/USDT": 1.012,
        "BNB/USDT": 1.014,
        "SOL/USDT": 1.008,
        "ADA/USDT": 1.007,
        "XRP/USDT": 1.006,
    }
    _feed_windows(strategy, _trend_path(mults))
    entries = _entries(events)
    assert entries, "expected LONG entries in broad bull regime"
    assert {event.signal_type for event in entries} == {"LONG"}
    assert len({event.symbol for event in entries}) <= 2
    assert {event.symbol for event in entries} <= {"BNB/USDT", "ETH/USDT", "BTC/USDT"}
    assert all((event.metadata or {}).get("regime") == "BULL" for event in entries)


def test_bear_regime_enters_weakest_short_names() -> None:
    events = _Events()
    strategy = _strategy(events)
    mults = {
        "BTC/USDT": 0.990,
        "ETH/USDT": 0.988,
        "BNB/USDT": 0.986,
        "SOL/USDT": 0.992,
        "ADA/USDT": 0.993,
        "XRP/USDT": 0.994,
    }
    _feed_windows(strategy, _trend_path(mults))
    entries = _entries(events)
    assert entries, "expected SHORT entries in broad bear regime"
    assert {event.signal_type for event in entries} == {"SHORT"}
    assert len({event.symbol for event in entries}) <= 2
    assert {event.symbol for event in entries} <= {"BNB/USDT", "ETH/USDT", "BTC/USDT"}
    assert all((event.metadata or {}).get("regime") == "BEAR" for event in entries)


def test_neutral_regime_flattens_open_positions() -> None:
    events = _Events()
    strategy = _strategy(events)
    _feed_windows(
        strategy,
        _trend_path(dict.fromkeys(_SYMBOLS, 1.012), steps=10),
    )
    assert _entries(events), "bull setup should open positions first"
    # Flat/oscillating tape long enough to erase the momentum window and close the
    # breadth gate.  The router should emit EXITs instead of holding stale beta.
    prices = dict.fromkeys(_SYMBOLS, 120.0)
    neutral_steps = []
    for i in range(8):
        bump = 1.0005 if i % 2 == 0 else 0.9995
        for symbol in _SYMBOLS:
            prices[symbol] *= bump
        neutral_steps.append(dict(prices))
    _feed_windows(strategy, neutral_steps)
    exits = [event for event in events.signals if event.signal_type == "EXIT"]
    assert exits, "expected neutral regime to flatten stale long book"
    assert strategy.get_state()["regime"] == "NEUTRAL"


def test_empty_window_and_degenerate_prices_are_safe() -> None:
    events = _Events()
    strategy = _strategy(events)
    strategy.calculate_signals(_EmptyWindow())
    strategy.calculate_signals(_Window(_ts(0), dict.fromkeys(_SYMBOLS, 0.0)))
    assert events.signals == []


def test_state_roundtrip_preserves_regime_and_symbol_state() -> None:
    events = _Events()
    strategy = _strategy(events)
    _feed_windows(strategy, _trend_path(dict.fromkeys(_SYMBOLS, 1.01), steps=8))
    snapshot = strategy.get_state()
    restored = _strategy(_Events())
    restored.set_state(snapshot)
    again = restored.get_state()
    assert again["regime"] == snapshot["regime"]
    assert again["tick"] == snapshot["tick"]
    assert again["symbol_state"]["BTC/USDT"]["closes"] == snapshot["symbol_state"]["BTC/USDT"]["closes"]


def test_candidate_builder_wires_basket_only_ge_30m() -> None:
    rows = [
        row
        for row in build_binance_futures_candidates(
            timeframes=["15m", "30m", "1h", "4h", "1d"],
            symbols=_SYMBOLS,
        )
        if row.strategy_class == "BullBearRegimeRotationStrategy"
    ]
    assert rows, "expected bull/bear regime rotation candidates on a real basket"
    assert {row.timeframe for row in rows} == {"30m", "1h", "4h", "1d"}
    for row in rows:
        assert row.family == "cross_sectional"
        assert candidate_mix_type(row.to_dict()) == "multi"
        assert len(row.symbols) >= 5
        assert row.params["allow_short"] is True
        assert int((row.metadata or {}).get("decision_cadence_seconds", 0)) >= 1800
