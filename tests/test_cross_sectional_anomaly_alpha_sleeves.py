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
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from lumina_quant.core.market_window_contract import build_market_window_event
from lumina_quant.indicators.rolling_stats import rolling_skewness
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

_DAY_MS = 86_400_000
_UTC_START_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z


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


def test_skewness_is_the_original_plain_sum_recipe_not_the_fsum_alias() -> None:
    # REGRESSION (W3, 2026-08-20): this registered module's ``_skewness`` was
    # briefly aliased to the canonical ``rolling_stats.rolling_skewness``, whose
    # ``math.fsum`` moment accumulation drifts from the original plain-``sum``
    # recipe by one ULP on some inputs — enough to flip LotterySkewness
    # cross-sectional rank ties.  Pin the ORIGINAL numerics: the fixture below
    # is a return-scale vector where the two recipes provably disagree in the
    # last bit, so re-aliasing flips this exact-equality golden.
    assert _skewness is not rolling_skewness
    fixture = [
        -0.088907,
        0.047501,
        -0.035967,
        -0.077349,
        -0.005959,
        -0.072556,
        -0.032015,
        -0.080339,
        0.073846,
        0.045776,
        -0.05739,
        -0.083633,
    ]
    # Golden from the original recipe (statistics.mean + plain-sum moments).
    golden = 0.69033172674867
    assert _skewness(fixture) == golden
    # Witness that the fixture separates the recipes: the fsum canonical copy
    # lands one ULP away, so the golden above uniquely pins the plain-sum path.
    canonical = rolling_skewness(fixture)
    assert canonical is not None and canonical != golden


def test_anomaly_panel_admission_is_payload_order_invariant_and_rejects_duplicates() -> None:
    symbols = ["BTC/USDT", "AAA/USDT", "BBB/USDT", "CCC/USDT"]
    params = dict(beta_window=4, vol_window=2, rebalance_bars=99, min_symbols=2)
    left = IdiosyncraticVolatilityStrategy(_Bars(symbols), queue.Queue(), **params)
    right = IdiosyncraticVolatilityStrategy(_Bars(symbols), queue.Queue(), **params)
    closes = {symbol: 100.0 + index for index, symbol in enumerate(symbols)}
    volumes = dict.fromkeys(symbols, 1_000.0)
    left.calculate_signals(_window(_UTC_START_MS, closes, volumes))
    right.calculate_signals(_window(_UTC_START_MS, dict(reversed(closes.items())), volumes))
    assert left.get_state() == right.get_state()

    before = left.get_state()
    duplicate_row = _window(_UTC_START_MS + _DAY_MS, closes, volumes)
    duplicate_row.bars_1s["AAA/USDT"] *= 2
    left.calculate_signals(duplicate_row)
    assert left.get_state() == before

    duplicate_symbols = IdiosyncraticVolatilityStrategy(
        _Bars([*symbols, "AAA/USDT"]), queue.Queue(), **params
    )
    duplicate_symbols.calculate_signals(_window(_UTC_START_MS, closes, volumes))
    assert duplicate_symbols.get_state()["tick"] == 0


def test_idiosyncratic_returns_abstain_on_a_gapped_common_benchmark_grid() -> None:
    symbols = ["BTC/USDT", "AAA/USDT", "BBB/USDT", "CCC/USDT"]
    strategy = IdiosyncraticVolatilityStrategy(
        _Bars(symbols),
        queue.Queue(),
        beta_window=4,
        vol_window=2,
        rebalance_bars=99,
        min_symbols=2,
    )
    closes = {symbol: 100.0 + index for index, symbol in enumerate(symbols)}
    volumes = dict.fromkeys(symbols, 1_000.0)
    for offset in (0, _DAY_MS, 2 * _DAY_MS, 4 * _DAY_MS, 5 * _DAY_MS):
        strategy.calculate_signals(_window(_UTC_START_MS + offset, closes, volumes))
    assert strategy._synchronized_returns("AAA/USDT", "BTC/USDT", 4) is None


def test_idiosyncratic_uses_one_common_tail_for_distinct_beta_and_vol_horizons() -> None:
    symbols = ["BTC/USDT", "AAA/USDT", "BBB/USDT"]
    closes = dict.fromkeys(symbols, 100.0)
    volumes = dict.fromkeys(symbols, 1_000.0)
    for beta_window, vol_window in ((4, 2), (4, 6)):
        strategy = IdiosyncraticVolatilityStrategy(
            _Bars(symbols),
            queue.Queue(),
            beta_window=beta_window,
            vol_window=vol_window,
            rebalance_bars=99,
            min_symbols=2,
        )
        for index in range(max(beta_window, vol_window) + 1):
            for symbol in symbols:
                closes[symbol] *= 1.0 + (0.01 if index % 2 else -0.005)
            strategy.calculate_signals(_window(_UTC_START_MS + index * _DAY_MS, closes, volumes))
        aligned = strategy._synchronized_returns(
            "AAA/USDT", "BTC/USDT", max(beta_window, vol_window)
        )
        assert aligned is not None
        assert all(len(series) == max(beta_window, vol_window) for series in aligned)


def test_idiosyncratic_equal_scores_abstain() -> None:
    symbols = ["BTC/USDT", "AAA/USDT", "BBB/USDT"]
    events: queue.Queue = queue.Queue()
    strategy = IdiosyncraticVolatilityStrategy(
        _Bars(symbols), events, beta_window=4, vol_window=4, rebalance_bars=1, min_symbols=2
    )
    closes = dict.fromkeys(symbols, 100.0)
    volumes = dict.fromkeys(symbols, 1_000.0)
    for index in range(5):
        for symbol in symbols:
            closes[symbol] *= 1.01 if index % 2 else 0.995
        strategy.calculate_signals(_window(_UTC_START_MS + index * _DAY_MS, closes, volumes))
    assert _drain(events) == []


def test_lottery_rejects_gapped_daily_max_tail_and_bad_restore_without_mutation() -> None:
    symbols = ["AAA/USDT", "BBB/USDT"]
    strategy = LotterySkewnessStrategy(
        _Bars(symbols), queue.Queue(), skew_window=3, max_window=2, rebalance_bars=99, min_symbols=2
    )
    assert (
        strategy._lottery_score(
            [100.0, 101.0, 99.0, 102.0],
            completed_daily_closes=[
                (str(_UTC_START_MS), 100.0),
                (str(_UTC_START_MS + 2 * _DAY_MS), 101.0),
                (str(_UTC_START_MS + 3 * _DAY_MS), 102.0),
            ],
        )
        is None
    )

    closes = dict.fromkeys(symbols, 100.0)
    volumes = dict.fromkeys(symbols, 1_000.0)
    for index in range(4):
        for symbol in symbols:
            closes[symbol] *= 1.0 + (0.01 if index % 2 else -0.005)
        strategy.calculate_signals(_window(_UTC_START_MS + index * _DAY_MS, closes, volumes))
    checkpoint = strategy.get_state()
    malformed = deepcopy(checkpoint)
    del malformed["completed_daily_closes"]
    strategy.set_state(malformed)
    assert strategy.get_state() == checkpoint
    partial = deepcopy(checkpoint)
    partial["close_times"]["AAA/USDT"].pop()
    strategy.set_state(partial)
    assert strategy.get_state() == checkpoint
    shifted = deepcopy(checkpoint)
    shifted["close_times"]["AAA/USDT"].pop(0)
    shifted["symbol_state"]["AAA/USDT"]["closes"].pop(0)
    shifted["symbol_state"]["AAA/USDT"]["volumes"].pop(0)
    strategy.set_state(shifted)
    assert strategy.get_state() == checkpoint
    forged_open = deepcopy(checkpoint)
    forged_open["open_daily_closes"]["AAA/USDT"][1] += 1.0
    strategy.set_state(forged_open)
    assert strategy.get_state() == checkpoint

    empty = LotterySkewnessStrategy(_Bars(symbols), queue.Queue(), max_window=2, min_symbols=2)
    empty_checkpoint = empty.get_state()
    empty.set_state(empty_checkpoint)
    assert empty.get_state() == empty_checkpoint


def test_cross_state_rejects_mismatched_same_tail_grids_atomically() -> None:
    symbols = ["AAA/USDT", "BBB/USDT"]
    strategy = LotterySkewnessStrategy(
        _Bars(symbols), queue.Queue(), skew_window=3, max_window=2, rebalance_bars=99, min_symbols=2
    )
    closes = dict.fromkeys(symbols, 100.0)
    volumes = dict.fromkeys(symbols, 1_000.0)
    for index in range(4):
        for symbol in symbols:
            closes[symbol] *= 1.01 if index % 2 else 0.995
        strategy.calculate_signals(_window(_UTC_START_MS + index * _DAY_MS, closes, volumes))

    checkpoint = strategy.get_state()
    forged = deepcopy(checkpoint)
    # Keep AAA's complete, strictly increasing tail valid on its own while
    # moving an interior timestamp off the cross-symbol synchronized grid.
    forged["close_times"]["AAA/USDT"][1] = str(_UTC_START_MS + _DAY_MS + 3_600_000)
    strategy.set_state(forged)
    assert strategy.get_state() == checkpoint


def test_lottery_partial_checkpoint_roundtrips_and_forged_open_close_is_atomic() -> None:
    symbols = ["AAA/USDT", "BBB/USDT"]
    strategy = LotterySkewnessStrategy(
        _Bars(symbols), queue.Queue(), skew_window=3, max_window=2, rebalance_bars=99, min_symbols=2
    )
    closes = dict.fromkeys(symbols, 100.0)
    volumes = dict.fromkeys(symbols, 1_000.0)
    strategy.calculate_signals(_window(_UTC_START_MS, closes, volumes))

    checkpoint = strategy.get_state()
    restored = LotterySkewnessStrategy(
        _Bars(symbols), queue.Queue(), skew_window=3, max_window=2, rebalance_bars=99, min_symbols=2
    )
    restored.set_state(checkpoint)
    assert restored.get_state() == checkpoint

    forged = deepcopy(checkpoint)
    forged["open_daily_closes"]["AAA/USDT"][1] *= 1.01
    restored.set_state(forged)
    assert restored.get_state() == checkpoint


def test_lottery_checkpoint_restores_exactly_and_continues() -> None:
    symbols = ["AAA/USDT", "BBB/USDT"]
    params = dict(skew_window=3, max_window=2, rebalance_bars=99, min_symbols=2)
    original = LotterySkewnessStrategy(_Bars(symbols), queue.Queue(), **params)
    closes = dict.fromkeys(symbols, 100.0)
    volumes = dict.fromkeys(symbols, 1_000.0)
    for index in range(4):
        for symbol in symbols:
            closes[symbol] *= 1.0 + (0.01 if index % 2 else -0.005)
        original.calculate_signals(_window(_UTC_START_MS + index * _DAY_MS, closes, volumes))
    restored = LotterySkewnessStrategy(_Bars(symbols), queue.Queue(), **params)
    restored.set_state(original.get_state())
    assert restored.get_state() == original.get_state()
    for symbol in symbols:
        closes[symbol] *= 1.01
    event = _window(_UTC_START_MS + 4 * _DAY_MS, closes, volumes)
    original.calculate_signals(event)
    restored.calculate_signals(event)
    assert restored.get_state() == original.get_state()


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
    # Low residual volatility is long; high residual volatility is short.
    # BBB's high total volatility is systematic and disappears after beta
    # residualization, so it wins the low-idiosyncratic-volatility rank.
    assert side.get("BBB/USDT") == "LONG", side
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
    steady_price = 100.0
    for ts in range(80):
        closes: dict[str, float] = {}
        volumes: dict[str, float] = {}
        for idx, symbol in enumerate(symbols):
            if symbol == "LOTTO/USDT":
                # Frequent upside spikes keep a large daily MAX in the trailing
                # completed-day window at every late rebalance.
                spike = 0.15 if ts % 7 == 0 else 1.0
                closes[symbol] = 100.0 * (1.0 + 0.001 * ts) * (1.0 + spike if ts % 7 == 0 else 1.0)
            elif symbol == "STEADY/USDT":
                # Small ordinary gains punctuated by rare losses make this the
                # low-lottery control under daily MAX normalization.
                if ts:
                    steady_price *= 0.99 if ts % 6 == 0 else 1.001
                closes[symbol] = steady_price
            else:
                closes[symbol] = 100.0 * (1.001) ** ts * (1.0 + 0.010 * math.sin(ts * 0.7 + idx))
            volumes[symbol] = 1000.0
        # MAX is defined over completed UTC-day returns, not intraday bars.
        strat.calculate_signals(_window(_UTC_START_MS + ts * _DAY_MS, closes, volumes))
    return strat, _drain(events)


def test_lottery_skewness_dry_emission_and_directionality() -> None:
    _strat, signals = _run_lottery_skewness()
    assert _allocated(signals), "lottery sleeve emitted no allocated non-EXIT signal"
    side = _final_side(signals)
    # High-lottery name is repeatedly shorted; its final stop/rebalance exit
    # does not invalidate the directionality of those admitted entries.
    assert any(
        signal.symbol == "LOTTO/USDT" and str(signal.signal_type).upper() == "SHORT"
        for signal in _allocated(signals)
    )
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


def test_partial_skewed_windows_and_individual_events_cannot_drive_decisions() -> None:
    """Cross-sectional decisions require one complete, exact-time panel."""
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT"]
    events: queue.Queue = queue.Queue()
    strat = TrendEfficiencyMomentumStrategy(
        _Bars(list(symbols)),
        events,
        efficiency_period=6,
        trend_lookback_bars=6,
        rebalance_bars=1,
        min_symbols=4,
        allow_short=True,
    )
    for ts in range(12):
        closes = {
            symbol: 100.0 * (1.0 + 0.003 * (idx + 1)) ** ts for idx, symbol in enumerate(symbols)
        }
        strat.calculate_signals(_window(_UTC_START_MS + ts * _DAY_MS, closes, {}))
    _drain(events)
    baseline = strat.get_state()

    # A missing name rejects the batch without aging positions or appending a
    # subset of the panel.
    partial_closes = {
        symbol: 100.0 * (1.0 + 0.003 * (idx + 1)) ** 12 for idx, symbol in enumerate(symbols[:-1])
    }
    partial_time = _UTC_START_MS + 12 * _DAY_MS
    strat.calculate_signals(_window(partial_time, partial_closes, {}))
    assert _drain(events) == []
    assert strat.get_state() == baseline

    # All names alone are insufficient: every bar must carry the event time.
    skewed_closes = {
        symbol: 100.0 * (1.0 + 0.003 * (idx + 1)) ** 13 for idx, symbol in enumerate(symbols)
    }
    skewed_time = _UTC_START_MS + 13 * _DAY_MS
    skewed = _window(skewed_time, skewed_closes, {})
    close = skewed_closes["D/USDT"]
    skewed.bars_1s["D/USDT"] = (
        (skewed_time - 1_000, close * 0.999, close * 1.01, close * 0.99, close, 1000.0),
    )
    strat.calculate_signals(skewed)
    assert _drain(events) == []
    assert strat.get_state() == baseline

    # Individual MARKET events cannot establish a common-time panel and must
    # not mutate cross-sectional history.
    for symbol in symbols:
        close = skewed_closes[symbol]
        strat.calculate_signals(
            SimpleNamespace(
                type="MARKET",
                symbol=symbol,
                time=skewed_time,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1000.0,
            )
        )
    assert _drain(events) == []
    assert strat.get_state() == baseline


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
