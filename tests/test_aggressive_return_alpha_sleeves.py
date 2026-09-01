"""Deterministic tests for the aggressive per-symbol return-maximizing sleeves.

These tests exercise registry presence; the ride/pyramid/exit dry-emission
behavior on synthetic trending series; the multi-horizon AGREEMENT entry gate; the
buy-the-dip pullback entry timing; the directional funding-carry harvest (long on
persistent negative funding, short on persistent positive funding) fed via a stub
``get_latest_feature_value``; trailing-stop exits; and candidate-admission wiring
(single-asset, >=30m only).  No backtest is run.
"""

from __future__ import annotations

from typing import Any

import pytest

from lumina_quant.strategies.aggressive_return_alpha_sleeves import (
    FundingHarvestCarryStrategy,
    MultiTimeframeTrendEnsembleStrategy,
    PullbackTrendContinuationStrategy,
)
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory import candidate_library
from lumina_quant.strategy_factory.selection import candidate_mix_type

AGGRESSIVE_STRATEGIES = (
    "MultiTimeframeTrendEnsembleStrategy",
    "PullbackTrendContinuationStrategy",
    "FundingHarvestCarryStrategy",
)

RIDE_CLASSES = (
    MultiTimeframeTrendEnsembleStrategy,
    PullbackTrendContinuationStrategy,
)

_SYMBOL = "BTC/USDT"
_ALLOWED_TIMEFRAMES = {"30m", "1h", "4h", "1d"}
_FORBIDDEN_TIMEFRAMES = {"1s", "1m", "5m", "15m"}


class _Bars:
    """Minimal bars stub exposing ``symbol_list`` (no feature data)."""

    def __init__(self, symbol: str) -> None:
        self.symbol_list = [symbol]


class _FundingBars:
    """Bars stub serving a programmable per-bar funding rate via the feature getter."""

    def __init__(self, symbol: str) -> None:
        self.symbol_list = [symbol]
        self._funding: dict[str, float] = {symbol: 0.0}

    def set_funding(self, symbol: str, value: float) -> None:
        self._funding[symbol] = float(value)

    def get_latest_feature_value(self, symbol: str, field: str) -> float | None:
        if field == "funding_rate":
            return self._funding.get(symbol)
        return None


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


def _sustained_trend(
    n_trend: int = 220,
    *,
    up: bool = True,
    n_calm: int = 60,
    dip_every: int = 8,
    dip: float = 0.03,
    drift: float = 0.012,
) -> list[float]:
    """Calm range then a sustained drift with periodic genuine in-trend pullbacks.

    The calm lead-in keeps the baseline ATR low; the trend leg drifts steadily but
    every ``dip_every`` bars takes a real counter-trend step (so the short-horizon
    ROC actually turns against the trend), which exercises both the buy-the-dip
    pullback entry and the trailing-stop ride through noise.
    """
    closes: list[float] = []
    price = 100.0
    for _ in range(max(0, int(n_calm))):
        closes.append(price)
    for idx in range(int(n_trend)):
        if idx % dip_every == dip_every - 1:
            price *= (1.0 - dip) if up else (1.0 + dip)
        else:
            price *= (1.0 + drift) if up else (1.0 - drift)
        closes.append(price)
    return closes


def _trend_then_reverse(
    n_trend: int = 110,
    n_reverse: int = 40,
    *,
    up: bool = True,
    n_calm: int = 60,
) -> list[float]:
    """Calm range, sustained trend, then a sharp reversal that breaches the stop."""
    closes = _sustained_trend(n_trend, up=up, n_calm=n_calm)
    price = closes[-1]
    reverse = 0.95 if up else 1.05
    for _ in range(int(n_reverse)):
        price *= reverse
        closes.append(price)
    return closes


def _ohlc(prev_close: float, close: float) -> tuple[float, float, float, float]:
    """Derive a plausible OHLC bar from the prior and current close."""
    high = max(prev_close, close) * 1.004
    low = min(prev_close, close) * 0.996
    return prev_close, high, low, close


def _run(
    strategy: Any,
    closes: list[float],
    *,
    funding: list[float] | None = None,
) -> dict[str, Any]:
    """Feed ``closes`` bar-by-bar; optionally update funding before each bar."""
    events: _Events = strategy.events
    bars = strategy.bars
    prev = closes[0]
    summary: dict[str, Any] = {
        "entries": 0,
        "pyramids": 0,
        "nonexit_alloc": 0,
        "longs": 0,
        "shorts": 0,
        "trailing_stop_exits": 0,
        "max_position_run": 0,
        "entry_bar": None,
        "entry_price": None,
        "first_exit_bar": None,
        "entry_direction": None,
        "max_close": closes[0],
    }
    run = 0
    for idx, close in enumerate(closes):
        if funding is not None and hasattr(bars, "set_funding"):
            bars.set_funding(_SYMBOL, funding[idx])
        open_, high, low, close_value = _ohlc(prev, close)
        prev = close
        summary["max_close"] = max(summary["max_close"], close_value)
        before = len(events.signals)
        strategy.calculate_signals(_Market(f"t{idx:04d}", _SYMBOL, open_, high, low, close_value))
        for event in events.signals[before:]:
            metadata = event.metadata or {}
            signal_type = event.signal_type
            if signal_type in {"LONG", "SHORT"}:
                if signal_type == "LONG":
                    summary["longs"] += 1
                else:
                    summary["shorts"] += 1
                if (metadata.get("target_allocation") or 0.0) > 0.0:
                    summary["nonexit_alloc"] += 1
                if metadata.get("action") == "pyramid_add":
                    summary["pyramids"] += 1
                elif metadata.get("action") == "entry":
                    summary["entries"] += 1
                    if summary["entry_bar"] is None:
                        summary["entry_bar"] = idx
                        summary["entry_direction"] = signal_type
                        summary["entry_price"] = close_value
            elif signal_type == "EXIT":
                if metadata.get("reason") == "trailing_stop":
                    summary["trailing_stop_exits"] += 1
                if summary["first_exit_bar"] is None:
                    summary["first_exit_bar"] = idx
        item = strategy._state[_SYMBOL]
        if item.mode in {"LONG", "SHORT"}:
            run += 1
            summary["max_position_run"] = max(summary["max_position_run"], run)
        else:
            run = 0
    return summary


def test_aggressive_strategies_register_after_lazy_discovery() -> None:
    discovered = set(get_strategy_names())
    assert sum(1 for name in AGGRESSIVE_STRATEGIES if name in discovered) == 3
    for strategy_name in AGGRESSIVE_STRATEGIES:
        assert strategy_name in discovered, strategy_name
        schema = get_strategy_param_schema(strategy_name)
        assert schema, strategy_name
        defaults = get_default_strategy_params(strategy_name)
        assert defaults, strategy_name
        assert set(defaults).issubset(schema)
        # Trailing-stop ride lever, pyramiding, and TopCap-scale allocation present.
        assert "trail_atr_mult" in schema
        assert "max_adds" in schema
        assert "target_allocation" in schema
        assert schema["target_allocation"].default >= 0.10


@pytest.mark.parametrize("strategy_cls", RIDE_CLASSES)
def test_ride_winner_with_pyramiding(strategy_cls: type) -> None:
    closes = _sustained_trend(up=True)
    # ``max_adds=5`` so a single clean ride emits >=5 non-EXIT signals (1 entry +
    # several pyramid adds) carrying positive target_allocation.
    strategy = strategy_cls(_Bars(_SYMBOL), _Events(), max_adds=5)
    summary = _run(strategy, closes)
    # Opens a position and rides it across many bars.
    assert summary["entries"] >= 1, summary
    assert summary["max_position_run"] > 10, summary
    # Pyramids at least once on continuation.
    assert summary["pyramids"] >= 1, summary
    # Emits many non-EXIT signals carrying positive target_allocation.
    assert summary["nonexit_alloc"] >= 5, summary
    # The first entry on a sustained uptrend is LONG.
    assert summary["entry_direction"] == "LONG", summary


@pytest.mark.parametrize("strategy_cls", RIDE_CLASSES)
def test_short_on_downtrend(strategy_cls: type) -> None:
    closes = _sustained_trend(up=False)
    strategy = strategy_cls(_Bars(_SYMBOL), _Events())
    summary = _run(strategy, closes)
    assert summary["entries"] >= 1, summary
    assert summary["entry_direction"] == "SHORT", summary
    assert summary["shorts"] >= 1, summary


@pytest.mark.parametrize("strategy_cls", RIDE_CLASSES)
def test_exits_on_trailing_stop_breach(strategy_cls: type) -> None:
    closes = _trend_then_reverse(n_trend=90, n_reverse=25, up=True)
    strategy = strategy_cls(_Bars(_SYMBOL), _Events())
    summary = _run(strategy, closes)
    assert summary["entries"] >= 1, summary
    assert summary["trailing_stop_exits"] >= 1, summary
    assert summary["entry_bar"] is not None and summary["first_exit_bar"] is not None
    assert summary["first_exit_bar"] - summary["entry_bar"] > 5, summary


def _ramp(start: float, n: int, step_pct: float) -> list[float]:
    """Append ``n`` multiplicative steps of ``step_pct`` starting from ``start``."""
    out: list[float] = []
    price = start
    for _ in range(n):
        price *= 1.0 + step_pct
        out.append(price)
    return out


def test_multi_tf_no_entry_when_horizons_disagree() -> None:
    # Deterministic disagreement: build a closes window where the SHORT horizon is
    # up, the MEDIUM horizon is down, and the LONG horizon is up — no 2/3 majority
    # in a single direction beyond the 3/3 gate, so the ensemble must NOT enter.
    strategy = MultiTimeframeTrendEnsembleStrategy(
        _Bars(_SYMBOL),
        _Events(),
        short_lookback=4,
        mid_lookback=12,
        long_lookback=36,
        align_threshold=3,
    )
    item = strategy._state[_SYMBOL]
    # long-up base, then a medium-down leg, then a short-up rebound: the three
    # lookbacks land on opposing signs so alignment never reaches 3/3.
    closes = (
        _ramp(100.0, 40, 0.01)  # long-horizon uptrend base
        + _ramp(148.0, 16, -0.012)  # medium-horizon pulls down
        + _ramp(122.0, 4, 0.015)  # short-horizon rebounds up
    )
    for value in closes:
        item.closes.append(value)
    net, agree, _avg = strategy._alignment(list(item.closes))
    assert agree < 3, (net, agree)
    assert strategy._entry_decision(item) == "", (net, agree)

    # On a genuinely aligned sustained trend the same 3/3 gate DOES enter LONG.
    aligned = MultiTimeframeTrendEnsembleStrategy(_Bars(_SYMBOL), _Events(), align_threshold=3)
    aligned_summary = _run(aligned, _sustained_trend(up=True))
    assert aligned_summary["entries"] >= 1, aligned_summary
    assert aligned_summary["entry_direction"] == "LONG", aligned_summary


def test_pullback_enters_on_dip_not_at_breakout_high() -> None:
    closes = _sustained_trend(up=True)
    strategy = PullbackTrendContinuationStrategy(_Bars(_SYMBOL), _Events())
    summary = _run(strategy, closes)
    assert summary["entries"] >= 1, summary
    assert summary["entry_direction"] == "LONG", summary
    # The buy-the-dip entry price is strictly below the running high seen so far
    # (i.e. it did not chase the breakout high — it bought a pullback).
    assert summary["entry_price"] is not None and summary["entry_bar"] is not None
    high_through_entry = max(closes[: summary["entry_bar"] + 1])
    assert summary["entry_price"] < high_through_entry, summary


def _funding_signal(closes: list[float], rate: float) -> list[float]:
    return [rate] * len(closes)


def test_funding_harvest_long_on_persistent_negative_funding() -> None:
    # Persistent negative funding (longs paid) + rising price -> LONG, ride, pyramid.
    closes = _sustained_trend(up=True)
    funding = _funding_signal(closes, -0.0010)
    strategy = FundingHarvestCarryStrategy(_FundingBars(_SYMBOL), _Events(), max_adds=5)
    summary = _run(strategy, closes, funding=funding)
    assert summary["entries"] >= 1, summary
    assert summary["entry_direction"] == "LONG", summary
    assert summary["max_position_run"] > 10, summary
    assert summary["pyramids"] >= 1, summary
    assert summary["nonexit_alloc"] >= 5, summary


def test_funding_harvest_short_on_persistent_positive_funding() -> None:
    # Persistent positive funding (shorts paid) + falling price -> SHORT.
    closes = _sustained_trend(up=False)
    funding = _funding_signal(closes, 0.0010)
    strategy = FundingHarvestCarryStrategy(_FundingBars(_SYMBOL), _Events())
    summary = _run(strategy, closes, funding=funding)
    assert summary["entries"] >= 1, summary
    assert summary["entry_direction"] == "SHORT", summary
    assert summary["shorts"] >= 1, summary


def test_funding_harvest_exits_on_trailing_stop_breach() -> None:
    # Negative funding LONG ridden up, then a sharp crash breaches the trail.
    closes = _trend_then_reverse(n_trend=90, n_reverse=25, up=True)
    funding = _funding_signal(closes, -0.0010)
    strategy = FundingHarvestCarryStrategy(_FundingBars(_SYMBOL), _Events())
    summary = _run(strategy, closes, funding=funding)
    assert summary["entries"] >= 1, summary
    assert summary["entry_direction"] == "LONG", summary
    assert summary["trailing_stop_exits"] >= 1, summary


def test_funding_harvest_no_fight_blocks_short_into_rising_price() -> None:
    # Positive funding (wants SHORT) turns on only once a strong uptrend is already
    # established -> the trend-no-fight guard blocks the short while price rises.
    closes = _sustained_trend(up=True)
    funding = [0.0 if idx < 120 else 0.0010 for idx in range(len(closes))]
    strategy = FundingHarvestCarryStrategy(
        _FundingBars(_SYMBOL), _Events(), no_fight_roc=0.02, allow_short=True
    )
    summary = _run(strategy, closes, funding=funding)
    assert summary["shorts"] == 0, summary


def test_funding_harvest_state_roundtrip_is_stable() -> None:
    closes = _sustained_trend(up=True)
    funding = _funding_signal(closes, -0.0010)
    strategy = FundingHarvestCarryStrategy(_FundingBars(_SYMBOL), _Events())
    _run(strategy, closes, funding=funding)
    state = strategy.get_state()
    restored = FundingHarvestCarryStrategy(_FundingBars(_SYMBOL), _Events())
    restored.set_state(state)
    assert restored.get_state() == state


def test_aggressive_candidates_single_asset_and_only_ge_30m(monkeypatch) -> None:
    # Force the perp-feature gate so the funding-harvest builder wires too.
    monkeypatch.setattr(candidate_library, "_has_perp_support_data", lambda: True)
    rows = build_binance_futures_candidates(
        timeframes=["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    counts = {
        name: sum(1 for row in rows if row.strategy_class == name) for name in AGGRESSIVE_STRATEGIES
    }
    assert all(count > 0 for count in counts.values()), counts

    for row in rows:
        if row.strategy_class not in AGGRESSIVE_STRATEGIES:
            continue
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert len(row.symbols) == 1, row.name
        assert row.family in {"trend", "carry"}, (row.name, row.family)
        assert "return_rider" in row.tags
        assert "trailing_stop" in row.tags
        assert row.metadata["timeframe"] == row.timeframe


def test_funding_harvest_builder_gated_on_perp_support(monkeypatch) -> None:
    # Without perp feature data the funding-harvest builder wires nothing.
    monkeypatch.setattr(candidate_library, "_has_perp_support_data", lambda: False)
    rows = build_binance_futures_candidates(
        timeframes=["30m", "1h", "4h", "1d"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    assert not any(row.strategy_class == "FundingHarvestCarryStrategy" for row in rows)
    # The non-feature-gated sleeves still wire.
    assert any(row.strategy_class == "MultiTimeframeTrendEnsembleStrategy" for row in rows)
    assert any(row.strategy_class == "PullbackTrendContinuationStrategy" for row in rows)
