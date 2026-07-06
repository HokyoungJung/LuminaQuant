"""Deterministic tests for the per-symbol return-rider alpha sleeves.

These tests exercise registry presence, the ride/pyramid/exit dry-emission
behavior on synthetic trending series, directionality, trailing-stop exits, and
candidate-admission wiring (single-asset, >=30m only).  No backtest is run.
"""

from __future__ import annotations

import math
import random
import struct
from typing import Any

import pytest

from lumina_quant.indicators.alpha_features import finite_floats
from lumina_quant.indicators.momentum import kaufman_efficiency_ratio
from lumina_quant.strategies.registry import (
    get_default_strategy_params,
    get_strategy_names,
    get_strategy_param_schema,
)
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    AccelerationRiderStrategy,
    AdaptiveTrendRiderStrategy,
    VolatilityBreakoutRiderStrategy,
    _kama,
)
from lumina_quant.strategy_factory import build_binance_futures_candidates
from lumina_quant.strategy_factory.selection import candidate_mix_type

RIDER_STRATEGIES = (
    "AdaptiveTrendRiderStrategy",
    "VolatilityBreakoutRiderStrategy",
    "AccelerationRiderStrategy",
)

RIDER_CLASSES = (
    AdaptiveTrendRiderStrategy,
    VolatilityBreakoutRiderStrategy,
    AccelerationRiderStrategy,
)

_SYMBOL = "BTC/USDT"
_ALLOWED_TIMEFRAMES = {"30m", "1h", "4h", "1d"}
_FORBIDDEN_TIMEFRAMES = {"1s", "1m", "5m", "15m"}


class _Bars:
    """Minimal bars stub exposing only ``symbol_list``."""

    def __init__(self, symbol: str) -> None:
        self.symbol_list = [symbol]


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


def _trend_then_reverse(
    n_trend: int = 110,
    n_reverse: int = 60,
    *,
    up: bool = True,
    n_calm: int = 60,
) -> list[float]:
    """Build a calm range, a sustained directional trend, then a sharp reversal.

    The calm lead-in keeps the baseline ATR low so the directional trend leg reads
    as a genuine volatility-expansion breakout (in either direction); the trend leg
    drifts steadily with periodic pullbacks so the trailing stop is forced to hold
    through noise; the reversal leg moves sharply the other way so the trailing
    stop is eventually breached.
    """
    closes: list[float] = []
    price = 100.0
    # Flat closes (a nonzero ATR is still synthesized from the H/L band in
    # ``_ohlc``) so the calm period keeps the baseline ATR low and produces no
    # premature trend/breakout/acceleration entry before the directional leg.
    for _ in range(max(0, int(n_calm))):
        closes.append(price)
    drift = 0.012 if up else -0.012
    for idx in range(n_trend):
        pull = (-0.005 if up else 0.005) if idx % 6 == 5 else 0.0
        price *= 1.0 + drift + pull
        closes.append(price)
    reverse = 0.95 if up else 1.05
    for _ in range(n_reverse):
        price *= reverse
        closes.append(price)
    return closes


def _ohlc(prev_close: float, close: float) -> tuple[float, float, float, float]:
    """Derive a plausible OHLC bar from the prior and current close."""
    high = max(prev_close, close) * 1.004
    low = min(prev_close, close) * 0.996
    return prev_close, high, low, close


def _run(strategy: Any, closes: list[float]) -> dict[str, Any]:
    """Feed ``closes`` bar-by-bar and summarize the emitted ride behavior."""
    events: _Events = strategy.events
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
        "first_exit_bar": None,
        "entry_direction": None,
    }
    run = 0
    for idx, close in enumerate(closes):
        open_, high, low, close_value = _ohlc(prev, close)
        prev = close
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


def test_return_rider_strategies_register_after_lazy_discovery() -> None:
    discovered = set(get_strategy_names())
    assert sum(1 for name in RIDER_STRATEGIES if name in discovered) == 3
    for strategy_name in RIDER_STRATEGIES:
        assert strategy_name in discovered, strategy_name
        schema = get_strategy_param_schema(strategy_name)
        assert schema, strategy_name
        defaults = get_default_strategy_params(strategy_name)
        assert defaults, strategy_name
        assert set(defaults).issubset(schema)
        # Trailing-stop ride lever and meaningful TopCap-scale allocation present.
        assert "trail_atr_mult" in schema
        assert "max_adds" in schema
        assert "target_allocation" in schema
        assert schema["target_allocation"].default >= 0.10


@pytest.mark.parametrize("strategy_cls", RIDER_CLASSES)
def test_return_rider_rides_winner_with_pyramiding(strategy_cls: type) -> None:
    closes = _trend_then_reverse(up=True)
    strategy = strategy_cls(_Bars(_SYMBOL), _Events())
    summary = _run(strategy, closes)

    # Opens a position and rides it across many bars (trailing stop holds through
    # the periodic pullbacks rather than getting shaken out immediately).
    assert summary["entries"] >= 1, summary
    assert summary["max_position_run"] > 10, summary
    # Pyramids at least once on continuation.
    assert summary["pyramids"] >= 1, summary
    # Emits many non-EXIT signals carrying positive target_allocation.
    assert summary["nonexit_alloc"] >= 5, summary
    # The first entry on a sustained uptrend is LONG.
    assert summary["entry_direction"] == "LONG", summary


@pytest.mark.parametrize("strategy_cls", RIDER_CLASSES)
def test_return_rider_short_on_downtrend(strategy_cls: type) -> None:
    closes = _trend_then_reverse(up=False)
    strategy = strategy_cls(_Bars(_SYMBOL), _Events())
    summary = _run(strategy, closes)
    assert summary["entries"] >= 1, summary
    assert summary["entry_direction"] == "SHORT", summary
    assert summary["shorts"] >= 1, summary


@pytest.mark.parametrize("strategy_cls", RIDER_CLASSES)
def test_return_rider_exits_on_trailing_stop_breach(strategy_cls: type) -> None:
    # Uptrend then sharp crash: the trailing stop must hold during the trend and
    # only fire once the reversal breaches the trailed level.
    closes = _trend_then_reverse(n_trend=90, n_reverse=25, up=True)
    strategy = strategy_cls(_Bars(_SYMBOL), _Events())
    summary = _run(strategy, closes)
    assert summary["entries"] >= 1, summary
    assert summary["trailing_stop_exits"] >= 1, summary
    # The position was ridden for a meaningful number of bars before the stop.
    assert summary["entry_bar"] is not None and summary["first_exit_bar"] is not None
    assert summary["first_exit_bar"] - summary["entry_bar"] > 5, summary


def test_return_rider_state_roundtrip_is_stable() -> None:
    closes = _trend_then_reverse(up=True)
    strategy = AdaptiveTrendRiderStrategy(_Bars(_SYMBOL), _Events())
    _run(strategy, closes)
    state = strategy.get_state()

    restored = AdaptiveTrendRiderStrategy(_Bars(_SYMBOL), _Events())
    restored.set_state(state)
    assert restored.get_state() == state


def test_return_rider_candidates_single_asset_and_only_ge_30m() -> None:
    rows = build_binance_futures_candidates(
        timeframes=["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
    )
    counts = {
        name: sum(1 for row in rows if row.strategy_class == name) for name in RIDER_STRATEGIES
    }
    assert all(count > 0 for count in counts.values()), counts

    for row in rows:
        if row.strategy_class not in RIDER_STRATEGIES:
            continue
        payload = row.to_dict()
        assert candidate_mix_type(payload) == "single", row.name
        assert row.timeframe in _ALLOWED_TIMEFRAMES, (row.name, row.timeframe)
        assert row.timeframe not in _FORBIDDEN_TIMEFRAMES, (row.name, row.timeframe)
        assert len(row.symbols) == 1, row.name
        assert row.family in {"trend", "breakout", "momentum"}, (row.name, row.family)
        assert "return_rider" in row.tags
        assert "trailing_stop" in row.tags
        assert row.metadata["timeframe"] == row.timeframe


# ---------------------------------------------------------------------------
# Audit X1 (2026-07-06): the KAMA slice rewrite and the kaufman-efficiency-ratio
# tail-before-float-convert change MUST be bit-identical to the pre-fix code.
# The legacy oracles below reproduce the exact pre-fix expressions so any future
# refactor that perturbs the summation order / slice window trips these asserts.
# ---------------------------------------------------------------------------


def _kaufman_efficiency_ratio_legacy(values: Any, *, period: int = 10) -> float | None:
    """Pre-fix kaufman_efficiency_ratio: float-convert the full history first."""
    period_i = max(1, int(period))
    values_f = [float(value) for value in values]
    if len(values_f) <= period_i:
        return None
    tail = values_f[-(period_i + 1) :]
    direction = abs(tail[-1] - tail[0])
    volatility = 0.0
    for idx in range(1, len(tail)):
        volatility += abs(tail[idx] - tail[idx - 1])
    if volatility <= 1e-12:
        return 0.0
    ratio = direction / volatility
    return ratio if math.isfinite(ratio) else None


def _kama_legacy(
    values: list[float], *, period: int, fast: int, slow: int
) -> float | None:
    """Pre-fix O(n^2)-sliced KAMA using the legacy ``vals[: idx + 1][-tail:]`` window."""
    period_i = max(1, int(period))
    fast_i = max(1, int(fast))
    slow_i = max(fast_i + 1, int(slow))
    vals = finite_floats(values)
    if len(vals) <= period_i:
        return None
    fast_sc = 2.0 / (float(fast_i) + 1.0)
    slow_sc = 2.0 / (float(slow_i) + 1.0)
    kama = vals[0]
    for idx in range(1, len(vals)):
        window = vals[: idx + 1][-(period_i + 1) :]  # legacy slice expression
        er = _kaufman_efficiency_ratio_legacy(window, period=period_i)
        er_value = 0.0 if er is None else max(0.0, min(1.0, float(er)))
        smoothing = (er_value * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = kama + smoothing * (vals[idx] - kama)
    if not isinstance(kama, float):
        kama = float(kama)
    return kama if kama == kama else None  # NaN guard


def _bits(value: float | None) -> str:
    """Return an exact bit-pattern key so float equality is byte-for-byte."""
    return "None" if value is None else struct.pack(">d", value).hex()


def test_kaufman_efficiency_ratio_tail_slice_is_bit_identical() -> None:
    rng = random.Random(4242)
    checked = 0
    for _ in range(60):
        n = rng.randint(1, 300)
        vals = [rng.uniform(-500.0, 500.0) for _ in range(n)]
        period = rng.randint(1, 40)
        assert _bits(kaufman_efficiency_ratio(vals, period=period)) == _bits(
            _kaufman_efficiency_ratio_legacy(vals, period=period)
        ), (n, period)
        checked += 1
    assert checked >= 50
    # Tuple inputs must also match (isinstance short-circuit path).
    tup = tuple(rng.uniform(-10.0, 10.0) for _ in range(50))
    assert _bits(kaufman_efficiency_ratio(tup, period=7)) == _bits(
        _kaufman_efficiency_ratio_legacy(tup, period=7)
    )
    # Boundary lengths around the period guard (<=period -> None).
    for period in (1, 5, 20):
        for length in (period - 1, period, period + 1, period + 2):
            if length < 0:
                continue
            seq = [float(i) * 1.5 - 3.0 for i in range(length)]
            assert _bits(kaufman_efficiency_ratio(seq, period=period)) == _bits(
                _kaufman_efficiency_ratio_legacy(seq, period=period)
            ), (length, period)


def test_kama_slice_rewrite_is_bit_identical() -> None:
    rng = random.Random(20260706)
    checked = 0
    for _ in range(60):
        n = rng.randint(2, 400)
        vals = [rng.uniform(-1000.0, 1000.0) for _ in range(n)]
        period = rng.randint(1, 30)
        fast = rng.randint(1, 10)
        slow = rng.randint(fast + 1, 40)
        assert _bits(_kama(vals, period=period, fast=fast, slow=slow)) == _bits(
            _kama_legacy(vals, period=period, fast=fast, slow=slow)
        ), (n, period, fast, slow)
        checked += 1
    assert checked >= 50
    # Short-history and flat series edge cases.
    assert _kama([1.0], period=5, fast=2, slow=10) is None
    for series in ([5.0] * 40, list(range(60)), [3.0, 3.0, 9.0, 1.0, 7.0, 2.0]):
        seq = [float(x) for x in series]
        assert _bits(_kama(seq, period=4, fast=2, slow=12)) == _bits(
            _kama_legacy(seq, period=4, fast=2, slow=12)
        ), series


def test_kama_window_slice_matches_legacy_expression() -> None:
    """Directly prove the two slice expressions select the identical subsequence."""
    vals = [float(i) for i in range(50)]
    for period_i in range(1, 20):
        for idx in range(1, len(vals)):
            assert (
                vals[max(0, idx - period_i) : idx + 1]
                == vals[: idx + 1][-(period_i + 1) :]
            ), (period_i, idx)
