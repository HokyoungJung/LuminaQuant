"""Deterministic tests for TrendGatedResidualMomentumStrategy (W2-1).

Direct class import only (no ``@register`` on this lane, so no registry/tier
assertions).  The BUILD GATE runs the REAL incumbents through their full
decision paths on ONE hand-built synthetic panel and asserts materially
different EMITTED actions (anti-strawman rule): the new sleeve versus
``ResidualMomentumRotationStrategy`` (point residual, no skip),
``ResidualEquityMomentumStrategy(benchmark=BTC)`` (the true BHM nearest
neighbour, unconditional), ``TopCapTimeSeriesMomentumStrategy`` (raw momentum),
and ``StationarityGatedResidualReversionStrategy`` (the ADF-gate complement).

The synthetic closes are built from fixed arithmetic paths (no RNG): each
symbol's log return is ``beta * bench_return + idiosyncratic`` where the
idiosyncratic component is a deterministic drift plus a tiny alternating
oscillation (so residual-level ADF is well-defined) -- benchmark ``B`` falls
over the formation window, so a hidden idiosyncratic winner has a NEGATIVE raw
formation return but a POSITIVE residual one.

Dates start on a Monday and the panel length makes the LAST bar the first day
of a fresh ISO week, so the weekly decision clock fires cleanly on the final
completed bar.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.adaptive_crypto_alpha_sleeves import ResidualMomentumRotationStrategy
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import ResidualEquityMomentumStrategy
from lumina_quant.strategies.residual_momentum_alpha_sleeves import (
    TrendGatedResidualMomentumStrategy,
)
from lumina_quant.strategies.residual_reversion_alpha_sleeves import (
    StationarityGatedResidualReversionStrategy,
)
from lumina_quant.strategies.topcap_tsmom import TopCapTimeSeriesMomentumStrategy
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module) -- used only for the filler jitter in
# the hygiene tests, never to shape the load-bearing divergence paths.
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #

_N = 197  # last index 196 == 28*7: a Monday when BASE is a Monday -> fresh week
_BASE = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday
_TS = [(_BASE + timedelta(days=t)).isoformat() for t in range(_N)]
_BENCH = "BTC/USDT"
_SYMS = [_BENCH, "S1", "S2", "S3", "F1", "F2", "F3", "F4"]
# The divergence panel's fixture ADF threshold: cleanly separates the stationary
# oscillator S2 (adf ~= -4) from the trending residuals S1/S3/fillers (adf > -1.7).
_ADF_MIN_T = -2.0


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _PanelBars:
    """Bars stub for the MARKET-event incumbents (TopCap) with a per-symbol cursor."""

    def __init__(self, symbols: list[str], series: dict[str, list[float]]) -> None:
        self.symbol_list = list(symbols)
        self._series = series
        self._cursor = dict.fromkeys(symbols, -1)

    def advance(self, symbol: str, idx: int) -> None:
        self._cursor[symbol] = idx

    def get_latest_bar_datetime(self, symbol: str) -> Any:
        idx = self._cursor.get(symbol, -1)
        return _TS[idx] if idx >= 0 else None

    def get_latest_bar_value(self, symbol: str, field: str) -> float | None:
        idx = self._cursor.get(symbol, -1)
        if idx < 0:
            return None
        close = self._series[symbol][idx]
        return {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1000.0,
        }.get(field)


def _prices_from_logret(logret: list[float], p0: float = 100.0) -> list[float]:
    out = [p0]
    for r in logret:
        out.append(out[-1] * math.exp(r))
    return out


def _bench_logret(n: int = _N) -> list[float]:
    # Mild downtrend + period-4 square oscillation (mean-zero over 4 -> clean
    # drift, non-degenerate variance for the beta estimate).
    return [-0.0036 + 0.01 * (1.0 if (t // 2) % 2 == 0 else -1.0) for t in range(n - 1)]


def _idio(drift: float, *, amp: float = 0.002, n: int = _N) -> list[float]:
    return [drift + amp * (1.0 if t % 2 == 0 else -1.0) for t in range(n - 1)]


def _combine(beta: float, bench: list[float], idio: list[float]) -> list[float]:
    return [beta * b + i for b, i in zip(bench, idio, strict=False)]


def _divergence_series() -> dict[str, list[float]]:
    """Return the shared closes panel exercised by every incumbent."""
    bench = _bench_logret()
    # S1 "hidden idio winner": beta 1.5, +20% idio over the 56-bar formation ->
    # raw formation return 1.5*(-20%)+20% = -10% < 0, residual momentum > 0.
    s1 = _combine(1.5, bench, _idio(0.20 / 56))
    f1 = _combine(1.0, bench, _idio(0.10 / 56))
    f2 = _combine(1.0, bench, _idio(0.05 / 56))
    f3 = _combine(1.0, bench, _idio(-0.05 / 56))
    f4 = _combine(1.0, bench, _idio(-0.08 / 56))
    # S3 "spike-then-drift": -12% drift over the formation-ex-skip window plus a
    # +15% jump SPREAD across the 7 skip-week bars (so the point-residual
    # incumbent goes long while this sleeve, which excludes the skip, goes short).
    s3_idio = _idio(-0.12 / 56)
    for k in range(7):
        s3_idio[_N - 2 - k] += 0.15 / 7.0
    s3 = _combine(1.0, bench, s3_idio)
    # S2 "stationary oscillator": beta 1.0 residual triangle wave (period 8),
    # phased so the last bar sits at a positive peak -> ADF strongly negative.
    s2_level = []
    for t in range(_N):
        phase = t % 8
        val = phase if phase <= 4 else 8 - phase
        s2_level.append(0.05 * (val / 4.0))
    s2 = [1.0 * bench[t] + (s2_level[t + 1] - s2_level[t]) for t in range(_N - 1)]
    return {
        _BENCH: _prices_from_logret(bench),
        "S1": _prices_from_logret(s1),
        "S2": _prices_from_logret(s2),
        "S3": _prices_from_logret(s3),
        "F1": _prices_from_logret(f1),
        "F2": _prices_from_logret(f2),
        "F3": _prices_from_logret(f3),
        "F4": _prices_from_logret(f4),
    }


def _window_event(series: dict[str, list[float]], t: int) -> SimpleNamespace:
    bars_1s = {}
    for symbol in _SYMS:
        close = series[symbol][t]
        bars_1s[symbol] = [
            {
                "time": _TS[t],
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            }
        ]
    return SimpleNamespace(type="MARKET_WINDOW", time=_TS[t], bars_1s=bars_1s)


def _feed_window(strategy: Any, series: dict[str, list[float]]) -> None:
    for t in range(_N):
        strategy.calculate_signals(_window_event(series, t))


def _feed_market(strategy: Any, bars: _PanelBars, series: dict[str, list[float]]) -> None:
    for t in range(_N):
        for symbol in _SYMS:
            bars.advance(symbol, t)
            strategy.calculate_signals(
                SimpleNamespace(
                    type="MARKET",
                    symbol=symbol,
                    time=_TS[t],
                    open=series[symbol][t],
                    high=series[symbol][t] * 1.001,
                    low=series[symbol][t] * 0.999,
                    close=series[symbol][t],
                    volume=1000.0,
                )
            )


def _final_sides(items: list[Any]) -> dict[str, str]:
    sides: dict[str, str] = {}
    for sig in items:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            sides[sig.symbol] = kind
        elif kind == "EXIT":
            sides.pop(sig.symbol, None)
    return sides


def _new_sleeve(**overrides: Any) -> TrendGatedResidualMomentumStrategy:
    params: dict[str, Any] = dict(
        benchmark_symbol=_BENCH,
        beta_window_bars=90,
        formation_weeks=8,
        skip_weeks=1,
        bars_per_week=7,
        adf_window_bars=90,
        adf_nonstationarity_min_t=_ADF_MIN_T,
        quantile_pct=0.25,
        min_symbols=5,
        residual_vol_window_bars=56,
        min_hold_decisions=4,
        cooldown_decisions=1,
        min_history_bars=120,
        allow_short=True,
        hedge_benchmark=False,
        max_gross=1.0,
        min_price=0.0,
    )
    params.update(overrides)
    return TrendGatedResidualMomentumStrategy(_Bars(_SYMS), _Queue(), **params)


# --------------------------------------------------------------------------- #
# BUILD GATE
# --------------------------------------------------------------------------- #


def test_build_gate_new_sleeve_book_longs_hidden_winner_shorts_drifter() -> None:
    """Candidate-ACTS: both book sides non-empty; LONG S1, SHORT S3 on the panel."""
    series = _divergence_series()
    sleeve = _new_sleeve()
    _feed_window(sleeve, series)
    sides = _final_sides(sleeve.events.items)
    # Candidate-ACTS: both sides of the long-short book are non-empty.
    assert "LONG" in sides.values()
    assert "SHORT" in sides.values()
    assert sides.get("S1") == "LONG", sides  # hidden idio winner (raw return < 0)
    assert sides.get("S3") == "SHORT", sides  # spike-then-drift, skip excluded


def test_build_gate_s1_stage1_premises() -> None:
    """Stage-1: S1 raw formation return < 0 while residual momentum > 0 and non-stationary."""
    series = _divergence_series()
    sleeve = _new_sleeve()
    _feed_window(sleeve, series)
    closes_s1 = list(sleeve._state["S1"].closes)
    closes_b = list(sleeve._state[_BENCH].closes)
    result = sleeve._asset_score(closes_s1, closes_b)
    assert result is not None  # S1 is tradeable (residual non-stationary)
    assert result.residual_cumret > 0.0  # positive residual momentum -> LONG
    assert result.beta > 1.2  # levered to the benchmark (~1.5)
    assert result.adf_t > _ADF_MIN_T  # residual level FAILS stationarity (trending)
    # Raw (un-purged) formation return of S1 is NEGATIVE (benchmark dragged it down).
    formation = sleeve.formation_bars + sleeve.skip_bars
    raw_ret = math.log(closes_s1[-1 - sleeve.skip_bars] / closes_s1[-1 - formation])
    assert raw_ret < 0.0, raw_ret


def test_build_gate_vs_topcap_raw_momentum_diverges_on_s1() -> None:
    """S1(b): TopCap (raw momentum) is flat/short S1 while the new sleeve is LONG."""
    series = _divergence_series()
    sleeve = _new_sleeve()
    _feed_window(sleeve, series)
    assert _final_sides(sleeve.events.items).get("S1") == "LONG"

    bars = _PanelBars(_SYMS, series)
    topcap = TopCapTimeSeriesMomentumStrategy(
        bars,
        _Queue(),
        lookback_bars=63,
        rebalance_bars=1,
        signal_threshold=0.02,
        max_longs=6,
        max_shorts=5,
        min_price=0.0,
        btc_regime_ma=48,
        btc_symbol=_BENCH,
    )
    _feed_market(topcap, bars, series)
    top_sides = _final_sides(topcap.events.items)
    # Incumbent-LIVE: TopCap emits a definite non-empty book on the panel.
    assert top_sides, "TopCap must be live (emit a non-empty book) on the panel"
    # Divergent action: TopCap never LONGs the hidden idiosyncratic winner.
    assert top_sides.get("S1") != "LONG", top_sides


def test_build_gate_vs_reversion_incumbent_adf_gate_complement() -> None:
    """S1(c)/S2: the reversion incumbent ABSTAINS on the trending residuals and
    TRADES the stationary oscillator -- the exact complement of this sleeve."""
    series = _divergence_series()
    sleeve = _new_sleeve()
    _feed_window(sleeve, series)
    sleeve_sides = _final_sides(sleeve.events.items)

    rev = StationarityGatedResidualReversionStrategy(
        _Bars(_SYMS),
        _Queue(),
        benchmark_symbol=_BENCH,
        lookback_bars=5,
        beta_window=60,
        vol_window=20,
        adf_window=90,
        adf_reject_threshold=-2.86,
        entry_score=0.1,
        rebalance_bars=1,
        min_hold_bars=0,
        max_longs=4,
        max_shorts=4,
        allow_short=True,
        hedge_benchmark=False,
        min_price=0.0,
    )
    _feed_window(rev, series)
    rev_sides = _final_sides(rev.events.items)
    # Incumbent-LIVE: the reversion sleeve trades the STATIONARY residual (S2).
    assert rev_sides.get("S2") in {"LONG", "SHORT"}, rev_sides
    # Complementarity: the reversion sleeve ABSTAINS on the trending residuals
    # (S1, S3) that this sleeve trades; this sleeve ABSTAINS on the stationary
    # S2 that the reversion sleeve trades.
    assert "S1" not in rev_sides and "S3" not in rev_sides, rev_sides
    assert sleeve_sides.get("S1") == "LONG" and sleeve_sides.get("S3") == "SHORT"
    assert "S2" not in sleeve_sides, sleeve_sides


def test_build_gate_vs_residual_equity_momentum_isolates_adf_gate() -> None:
    """S2 decisive: the true BHM nearest neighbour (ResidualEquityMomentum,
    unconditional) TRADES the stationary S2 while this sleeve ABSTAINS -- the ADF
    anti-gate is the sole differentiator."""
    series = _divergence_series()
    sleeve = _new_sleeve()
    _feed_window(sleeve, series)
    # This sleeve abstains on S2 (residual level is stationary -> ADF gate closed).
    assert (
        sleeve._asset_score(list(sleeve._state["S2"].closes), list(sleeve._state[_BENCH].closes))
        is None
    )

    resm = ResidualEquityMomentumStrategy(
        _Bars(_SYMS),
        _Queue(),
        benchmark_symbol=_BENCH,
        lookback_bars=4,
        skip_bars=0,
        beta_window=8,
        rebalance_bars=1,
        signal_threshold=0.0,
        quintile_pct=0.5,
        min_symbols=4,
        allow_short=True,
        min_price=0.0,
    )
    _feed_window(resm, series)
    resm_sides = _final_sides(resm.events.items)
    # Incumbent-LIVE + divergent action: the unconditional BHM neighbour TRADES S2.
    assert resm_sides, "ResidualEquityMomentum must be live on the panel"
    assert resm_sides.get("S2") in {"LONG", "SHORT"}, resm_sides
    assert "S2" not in _final_sides(sleeve.events.items)


def test_build_gate_vs_residual_momentum_rotation_skip_divergence_on_s3() -> None:
    """S3: the point-residual incumbent (no skip) goes LONG S3 on the skip-week
    spike while this sleeve, which EXCLUDES the skip window, goes SHORT."""
    series = _divergence_series()
    sleeve = _new_sleeve()
    _feed_window(sleeve, series)
    assert _final_sides(sleeve.events.items).get("S3") == "SHORT"

    rmr = ResidualMomentumRotationStrategy(
        _Bars(_SYMS),
        _Queue(),
        benchmark_symbol=_BENCH,
        momentum_lookback_bars=7,
        beta_window=90,
        vol_window=30,
        rebalance_bars=1,
        signal_threshold=0.0,
        max_longs=3,
        max_shorts=3,
        allow_short=True,
        min_price=0.0,
    )
    _feed_window(rmr, series)
    rmr_sides = _final_sides(rmr.events.items)
    # Incumbent-LIVE + divergent action: RMR (point residual) LONGs S3.
    assert rmr_sides, "ResidualMomentumRotation must be live on the panel"
    assert rmr_sides.get("S3") == "LONG", rmr_sides
    assert _final_sides(sleeve.events.items).get("S3") == "SHORT"


# --------------------------------------------------------------------------- #
# min-hold property (the proven C1 turnover rescue encoded as a test)
# --------------------------------------------------------------------------- #


def _minhold_series() -> dict[str, list[float]]:
    """S1 idio stays mid-rank, spikes UP for one week (enters LONG at decision k),
    then reverses hard DOWN so its rank flips at decision k+1."""
    bench = _bench_logret()

    def s1_idio() -> list[float]:
        out = []
        for t in range(_N - 1):
            if t < 176:
                drift = 0.0  # mid-rank: not in the long book
            elif t < 183:
                drift = 0.03  # one-week up spike -> top rank -> LONG entry
            else:
                drift = -0.04  # hard reversal -> bottom rank -> would flip SHORT
            out.append(drift + 0.002 * (1.0 if t % 2 == 0 else -1.0))
        return out

    return {
        _BENCH: _prices_from_logret(bench),
        "S1": _prices_from_logret(_combine(1.0, bench, s1_idio())),
        "S2": _prices_from_logret(_combine(1.0, bench, _idio(0.02 / 56))),
        "S3": _prices_from_logret(_combine(1.0, bench, _idio(-0.02 / 56))),
        "F1": _prices_from_logret(_combine(1.0, bench, _idio(0.04 / 56))),
        "F2": _prices_from_logret(_combine(1.0, bench, _idio(0.01 / 56))),
        "F3": _prices_from_logret(_combine(1.0, bench, _idio(-0.01 / 56))),
        "F4": _prices_from_logret(_combine(1.0, bench, _idio(-0.04 / 56))),
    }


def _s1_directional_stream(items: list[Any]) -> list[str]:
    return [str(sig.signal_type).upper() for sig in items if sig.symbol == "S1"]


def test_min_hold_suppresses_rank_flip_and_reference_min_hold_flips() -> None:
    """With min_hold>=4 a would-be flip inside the hold window is suppressed; a
    min_hold=1 reference config releases the flip (the C1 rescue as a test)."""
    series = _minhold_series()
    common = dict(
        formation_weeks=1, skip_weeks=0, adf_nonstationarity_min_t=-1.5, cooldown_decisions=0
    )

    held = _new_sleeve(min_hold_decisions=6, **common)
    _feed_window(held, series)
    held_stream = _s1_directional_stream(held.events.items)
    # S1 enters LONG and is never flipped to SHORT while the min-hold is active.
    assert "LONG" in held_stream, held_stream
    assert "SHORT" not in held_stream, held_stream

    flips = _new_sleeve(min_hold_decisions=1, **common)
    _feed_window(flips, series)
    flip_stream = _s1_directional_stream(flips.events.items)
    # The reference min_hold=1 config releases the position: S1 does flip to SHORT.
    assert "LONG" in flip_stream and "SHORT" in flip_stream, flip_stream


# --------------------------------------------------------------------------- #
# lane invariants
# --------------------------------------------------------------------------- #


def test_run_twice_bit_identical() -> None:
    series = _divergence_series()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        sleeve = _new_sleeve()
        _feed_window(sleeve, series)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in sleeve.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal on the divergence panel"


def test_state_roundtrip_lossless() -> None:
    series = _divergence_series()
    sleeve = _new_sleeve()
    _feed_window(sleeve, series)
    snapshot = sleeve.get_state()
    restored = _new_sleeve()
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    for symbol in _SYMS:
        assert list(restored._state[symbol].closes) == list(sleeve._state[symbol].closes)
        assert restored._state[symbol].mode == sleeve._state[symbol].mode
        assert restored._state[symbol].bars_held == sleeve._state[symbol].bars_held
    assert restored._last_decision_week == sleeve._last_decision_week
    assert restored._tick == sleeve._tick


def test_adversarial_set_state_never_raises() -> None:
    sleeve = _new_sleeve()
    sleeve.set_state(None)  # type: ignore[arg-type]
    sleeve.set_state("not a dict")  # type: ignore[arg-type]
    sleeve.set_state(12345)  # type: ignore[arg-type]
    sleeve.set_state([])  # type: ignore[arg-type]
    sleeve.set_state({"symbol_state": "not a dict"})
    sleeve.set_state({"symbol_state": {"S1": "not a dict either"}})
    sleeve.set_state({"symbol_state": {"S1": {"closes": 12345}}})
    sleeve.set_state({"symbol_state": {"S1": {"closes": {"nested": "dict"}}}})
    sleeve.set_state(
        {
            "last_decision_week": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "bars_since_exit": [1, 2],
                    "last_bar_key": 123,
                    "score": [1, 2, 3],
                    "beta_exposure": "nope",
                }
                for symbol in _SYMS
            },
        }
    )
    for item in sleeve._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
    # The sleeve still functions afterward on a fresh feed.
    _feed_window(sleeve, _divergence_series())


def test_never_raise_on_degenerate_input() -> None:
    sleeve = _new_sleeve()
    sleeve.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    sleeve.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    sleeve.calculate_signals(SimpleNamespace(type="MARKET", symbol="S1", time=_TS[0], close=0.0))
    sleeve.calculate_signals(
        SimpleNamespace(type="MARKET", symbol="S1", time=_TS[1], close=float("nan"))
    )
    sleeve.calculate_signals(
        SimpleNamespace(type="MARKET", symbol="S1", time=_TS[2], close=float("inf"))
    )
    sleeve.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=_TS[3], bars_1s={}))
    sleeve.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=_TS[4], bars_1s={"S1": []}))
    assert [sig for sig in sleeve.events.items if str(sig.signal_type).upper() != "EXIT"] == []


def test_self_skip_below_min_symbols() -> None:
    # Only three symbols -> below min_symbols; the book must emit nothing.
    syms = [_BENCH, "S1", "S2"]
    sleeve = TrendGatedResidualMomentumStrategy(
        _Bars(syms),
        _Queue(),
        benchmark_symbol=_BENCH,
        min_symbols=5,
        min_history_bars=20,
        bars_per_week=7,
        min_price=0.0,
    )
    series = _divergence_series()
    for t in range(_N):
        bars_1s = {
            s: [
                {
                    "time": _TS[t],
                    "open": series[s][t],
                    "high": series[s][t],
                    "low": series[s][t],
                    "close": series[s][t],
                    "volume": 1000.0,
                }
            ]
            for s in syms
        }
        sleeve.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=_TS[t], bars_1s=bars_1s)
        )
    assert [s for s in sleeve.events.items if str(s.signal_type).upper() != "EXIT"] == []


def test_hedge_benchmark_leg_emits_and_never_raises() -> None:
    series = _divergence_series()
    sleeve = _new_sleeve(hedge_benchmark=True)
    _feed_window(sleeve, series)
    # The benchmark carries an offsetting hedge leg when the book has net beta.
    bench_signals = [sig for sig in sleeve.events.items if sig.symbol == _BENCH]
    assert bench_signals, "hedge_benchmark=True must emit a benchmark leg"


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = TrendGatedResidualMomentumStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "benchmark_symbol",
        "beta_window_bars",
        "formation_weeks",
        "skip_weeks",
        "adf_window_bars",
        "adf_nonstationarity_min_t",
        "quantile_pct",
        "min_symbols",
        "residual_vol_window_bars",
        "min_hold_decisions",
        "cooldown_decisions",
        "min_history_bars",
        "allow_short",
        "hedge_benchmark",
        "max_gross",
    ):
        assert required in schema


def test_decision_cadence_at_least_30m() -> None:
    assert TrendGatedResidualMomentumStrategy.decision_cadence_seconds >= 1800
