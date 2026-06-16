"""Deterministic unit tests for the cross-sectional anomaly alpha sleeves.

These tests exercise the four decorrelated anomaly sleeves entirely through the
local event/bar contract (no backtest, no data fetch):

- registry presence for all four classes;
- dry-emission: each sleeve emits at least one non-EXIT signal carrying a
  positive ``target_allocation`` on a constructed divergent universe;
- per-factor directionality: low idio-vol ranked long / high shorted; high-skew
  (lottery) shorted; clean high-efficiency uptrend long; the dispersion gate only
  fires in a high-dispersion regime and then fades the extreme movers;
- graceful skip when fewer than ``min_symbols`` names carry usable history;
- candidate-admission via build -> to_dict -> candidate_mix_type / allowlist.
"""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass
from typing import Any

from lumina_quant.core.market_window_contract import build_market_window_event
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    DispersionConditionedReversionStrategy,
    IdiosyncraticVolatilityStrategy,
    LotterySkewnessStrategy,
    TrendEfficiencyMomentumStrategy,
    _skewness,
)
from lumina_quant.strategies.registry import get_strategy_map
from lumina_quant.strategy_factory.selection import (
    _allowlisted_portfolio_native_multi_asset_candidate,
    candidate_mix_type,
)


ANOMALY_STRATEGIES = (
    "IdiosyncraticVolatilityStrategy",
    "LotterySkewnessStrategy",
    "TrendEfficiencyMomentumStrategy",
    "DispersionConditionedReversionStrategy",
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


def _allocated(signals: list[Any]) -> list[Any]:
    return [
        sig
        for sig in _non_exit(signals)
        if float((sig.metadata or {}).get("target_allocation", 0.0)) > 0.0
    ]


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


# ---------------------------------------------------------------------------
# pure helper
# ---------------------------------------------------------------------------
def test_skewness_helper_is_none_safe_and_signed() -> None:
    assert _skewness([1.0, 2.0]) is None  # < 3 samples
    assert _skewness([5.0, 5.0, 5.0, 5.0]) is None  # zero dispersion
    assert _skewness([float("nan"), float("inf"), 1.0]) is None  # only one finite
    right = _skewness([0.0, 0.0, 0.0, 0.0, 10.0])
    left = _skewness([0.0, -10.0, 0.0, 0.0, 0.0])
    assert right is not None and right > 0.0
    assert left is not None and left < 0.0


# ---------------------------------------------------------------------------
# (a) registry presence (4)
# ---------------------------------------------------------------------------
def test_anomaly_sleeves_are_registered() -> None:
    strategy_map = get_strategy_map()
    assert strategy_map["IdiosyncraticVolatilityStrategy"] is IdiosyncraticVolatilityStrategy
    assert strategy_map["LotterySkewnessStrategy"] is LotterySkewnessStrategy
    assert strategy_map["TrendEfficiencyMomentumStrategy"] is TrendEfficiencyMomentumStrategy
    assert (
        strategy_map["DispersionConditionedReversionStrategy"]
        is DispersionConditionedReversionStrategy
    )


# ---------------------------------------------------------------------------
# (b)+(c) dry-emission + per-factor directionality
# ---------------------------------------------------------------------------
def _run_idiosyncratic_volatility() -> tuple[Any, list[Any]]:
    # Benchmark carries REAL return variance and each symbol return is
    # beta*bench_ret + idiosyncratic noise. BBB has a HIGH beta (so high TOTAL
    # volatility) but LOW idiosyncratic noise; a total-volatility ranking would
    # short it, whereas correct benchmark residualization must not. The
    # benchmark uses a period-2 square wave and the idio noise a period-4 wave,
    # which are (near-)orthogonal so rolling_beta recovers the true loading.
    symbols = ["BTC/USDT", "AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT", "EEE/USDT"]
    betas = {
        "AAA/USDT": 0.3,
        "BBB/USDT": 1.6,
        "CCC/USDT": 1.0,
        "DDD/USDT": 1.0,
        "EEE/USDT": 1.0,
    }
    idio_amp = {
        "AAA/USDT": 0.0008,
        "BBB/USDT": 0.0010,
        "CCC/USDT": 0.0080,
        "DDD/USDT": 0.0160,
        "EEE/USDT": 0.0300,
    }
    events: queue.Queue = queue.Queue()
    strat = IdiosyncraticVolatilityStrategy(
        _Bars(list(symbols)),
        events,
        benchmark_symbol="BTC/USDT",
        beta_window=32,
        vol_window=20,
        rebalance_bars=5,
        quantile_pct=0.25,
        min_symbols=4,
        allow_short=True,
    )
    bench_price = 100.0
    prices = dict.fromkeys(symbols[1:], 100.0)
    for ts in range(160):
        bench_ret = 0.01 if ts % 2 == 0 else -0.01  # period-2, real variance
        bench_price *= 1.0 + bench_ret
        closes = {"BTC/USDT": bench_price}
        volumes = {"BTC/USDT": 1000.0}
        idio_sign = 1.0 if ts % 4 < 2 else -1.0  # period-4, orthogonal to bench
        for symbol in symbols[1:]:
            sym_ret = betas[symbol] * bench_ret + idio_amp[symbol] * idio_sign
            prices[symbol] *= 1.0 + sym_ret
            closes[symbol] = prices[symbol]
            volumes[symbol] = 1000.0
        strat.calculate_signals(_window(ts, closes, volumes))
    return strat, _drain(events)


def test_idiosyncratic_volatility_dry_emission_and_directionality() -> None:
    _strat, signals = _run_idiosyncratic_volatility()
    assert _allocated(signals), "idio-vol sleeve emitted no allocated non-EXIT signal"
    side = _final_side(signals)
    # Low idiosyncratic vol (AAA) is long; high idiosyncratic vol (EEE) is short.
    assert side.get("AAA/USDT") == "LONG", side
    assert side.get("EEE/USDT") == "SHORT", side
    # BBB has high beta (high TOTAL vol) but low idio vol: residualization must
    # remove the systematic component, so it is never shorted as "high vol".
    assert side.get("BBB/USDT") != "SHORT", side


def _run_lottery_skewness() -> tuple[Any, list[Any]]:
    symbols = ["LOTTO/USDT", "S1/USDT", "S2/USDT", "S3/USDT", "STEADY/USDT"]
    events: queue.Queue = queue.Queue()
    strat = LotterySkewnessStrategy(
        _Bars(list(symbols)),
        events,
        skew_window=30,
        max_window=10,
        rebalance_bars=5,
        quantile_pct=0.25,
        min_symbols=4,
        allow_short=True,
    )
    for ts in range(80):
        closes: dict[str, float] = {}
        volumes: dict[str, float] = {}
        for idx, symbol in enumerate(symbols):
            if symbol == "LOTTO/USDT":
                # Periodic upside spikes -> high positive skew + large MAX.
                spike = 0.15 if ts % 17 == 0 else 1.0
                closes[symbol] = 100.0 * (1.0 + 0.001 * ts) * (1.0 + spike if ts % 17 == 0 else 1.0)
            elif symbol == "STEADY/USDT":
                closes[symbol] = 100.0 * (1.002) ** ts * (1.0 + 0.001 * math.sin(ts * 0.5))
            else:
                closes[symbol] = 100.0 * (1.001) ** ts * (1.0 + 0.010 * math.sin(ts * 0.7 + idx))
            volumes[symbol] = 1000.0
        strat.calculate_signals(_window(ts, closes, volumes))
    return strat, _drain(events)


def test_lottery_skewness_dry_emission_and_directionality() -> None:
    _strat, signals = _run_lottery_skewness()
    assert _allocated(signals), "lottery sleeve emitted no allocated non-EXIT signal"
    side = _final_side(signals)
    # High-lottery name shorted; smooth low-skew name long.
    assert side.get("LOTTO/USDT") == "SHORT", side
    assert side.get("STEADY/USDT") == "LONG", side


def _run_trend_efficiency() -> tuple[Any, list[Any]]:
    symbols = ["CLEAN/USDT", "C1/USDT", "C2/USDT", "CHOPPY/USDT", "DOWN/USDT"]
    events: queue.Queue = queue.Queue()
    strat = TrendEfficiencyMomentumStrategy(
        _Bars(list(symbols)),
        events,
        efficiency_period=15,
        trend_lookback_bars=15,
        rebalance_bars=5,
        quantile_pct=0.25,
        signal_threshold=0.05,
        min_symbols=4,
        allow_short=True,
    )
    for ts in range(80):
        closes: dict[str, float] = {}
        volumes: dict[str, float] = {}
        for idx, symbol in enumerate(symbols):
            if symbol == "CLEAN/USDT":
                closes[symbol] = 100.0 * (1.010) ** ts  # smooth clean uptrend
            elif symbol == "DOWN/USDT":
                closes[symbol] = 100.0 * (0.990) ** ts  # clean downtrend
            elif symbol == "CHOPPY/USDT":
                closes[symbol] = 100.0 * (1.003) ** ts * (1.0 + 0.05 * math.sin(ts * 2.1))
            else:
                closes[symbol] = 100.0 * (1.002) ** ts * (1.0 + 0.02 * math.sin(ts * 1.3 + idx))
            volumes[symbol] = 1000.0
        strat.calculate_signals(_window(ts, closes, volumes))
    return strat, _drain(events)


def test_trend_efficiency_dry_emission_and_directionality() -> None:
    _strat, signals = _run_trend_efficiency()
    assert _allocated(signals), "trend-efficiency sleeve emitted no allocated non-EXIT signal"
    side = _final_side(signals)
    # Clean high-efficiency uptrend is long; clean downtrend is short.
    assert side.get("CLEAN/USDT") == "LONG", side
    assert side.get("DOWN/USDT") == "SHORT", side


def test_dispersion_gate_only_fires_in_high_dispersion_and_fades_extremes() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]
    events: queue.Queue = queue.Queue()
    strat = DispersionConditionedReversionStrategy(
        _Bars(list(symbols)),
        events,
        reversion_lookback_bars=4,
        dispersion_threshold=0.03,
        rebalance_bars=2,
        quantile_pct=0.25,
        min_symbols=4,
        allow_short=True,
    )
    # Phase 1: all symbols move together -> near-zero cross-sectional dispersion.
    for ts in range(40):
        closes = dict.fromkeys(symbols, 100.0 * 1.002**ts)
        volumes = dict.fromkeys(symbols, 1000.0)
        strat.calculate_signals(_window(ts, closes, volumes))
    phase1_signals = _drain(events)
    assert not _non_exit(phase1_signals), [s.symbol for s in _non_exit(phase1_signals)]

    # Phase 2: wide cross-sectional dispersion -> gate opens, fade the extremes.
    phase2_events: queue.Queue = queue.Queue()
    strat.events = phase2_events
    ramp = {
        "A/USDT": 0.06,
        "B/USDT": 0.02,
        "C/USDT": 0.0,
        "D/USDT": -0.02,
        "E/USDT": -0.06,
    }
    for ts in range(40, 90):
        closes = {symbol: 100.0 * (1.0 + ramp[symbol]) ** (ts - 39) for symbol in symbols}
        volumes = dict.fromkeys(symbols, 1000.0)
        strat.calculate_signals(_window(ts, closes, volumes))
    phase2_signals = _drain(phase2_events)
    assert _allocated(phase2_signals), "dispersion gate emitted nothing in high-dispersion regime"
    side = _final_side(phase2_signals)
    # Fade extremes: top mover (A) shorted, bottom mover (E) bought.
    assert side.get("A/USDT") == "SHORT", side
    assert side.get("E/USDT") == "LONG", side


# ---------------------------------------------------------------------------
# (d) graceful skip when fewer than min_symbols carry usable history
# ---------------------------------------------------------------------------
def test_graceful_skip_below_min_symbols() -> None:
    # Only three names ever carry usable closes; min_symbols default is 4.
    live = ["AAA/USDT", "BBB/USDT", "CCC/USDT"]
    events: queue.Queue = queue.Queue()
    strat = TrendEfficiencyMomentumStrategy(
        _Bars(list(live)),
        events,
        efficiency_period=10,
        trend_lookback_bars=10,
        rebalance_bars=4,
        min_symbols=4,
        allow_short=True,
    )
    for ts in range(60):
        closes = {symbol: 100.0 * (1.0 + 0.01 * (idx + 1)) ** ts for idx, symbol in enumerate(live)}
        volumes = dict.fromkeys(live, 1000.0)
        strat.calculate_signals(_window(ts, closes, volumes))
    signals = _drain(events)
    assert not _non_exit(signals), [s.symbol for s in _non_exit(signals)]


def test_graceful_skip_when_no_symbols() -> None:
    # Empty universe must not raise and must emit nothing.
    events: queue.Queue = queue.Queue()
    strat = IdiosyncraticVolatilityStrategy(_Bars([]), events)
    strat.calculate_signals(_window(0, {}, {}))
    assert _drain(events) == []


# ---------------------------------------------------------------------------
# (e) candidate-admission: built rows survive the default shortlist policy
# ---------------------------------------------------------------------------
def test_candidate_admission_for_anomaly_sleeves(monkeypatch) -> None:
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
    builders = {
        "IdiosyncraticVolatilityStrategy": "_build_idiosyncratic_volatility_candidates",
        "LotterySkewnessStrategy": "_build_lottery_skewness_candidates",
        "TrendEfficiencyMomentumStrategy": "_build_trend_efficiency_momentum_candidates",
        "DispersionConditionedReversionStrategy": (
            "_build_dispersion_conditioned_reversion_candidates"
        ),
    }
    for strategy_class, builder_name in builders.items():
        assert hasattr(candidate_library, builder_name), builder_name
        matches = [row for row in rows if row.strategy_class == strategy_class]
        assert matches, strategy_class
        for row in matches:
            payload = row.to_dict()
            mix = candidate_mix_type(payload)
            assert mix == "single" or _allowlisted_portfolio_native_multi_asset_candidate(
                payload
            ), (strategy_class, mix)
            if len(row.symbols) >= 3:
                assert row.family == "cross_sectional"
                assert required_multi_tags.issubset(set(row.tags))
                assert float(row.params["rebalance_band"]) <= 2.5
            assert row.metadata["timeframe"] == row.timeframe
