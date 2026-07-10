"""Deterministic build-gate + hygiene tests for the spread-stress reversal lane.

Direct class import only (no ``@register`` on this lane).  The four author-time
BUILD GATES (T1-T4) are binding keep/drop decisions: each instantiates the REAL
reversal incumbents and drives their full decision path on identical synthetic
bars, asserting a materially DIVERGENT emitted action (never merely a different
internal score).  All randomness is a small seeded LCG (no ``random`` module),
so every run is bit-for-bit reproducible.

- T1  INDICATOR ISOLATION: the Corwin-Schultz spread separates spread-stress
      from Parkinson volatility (staircase vs oscillation, identical per-bar
      ``ln(H/L)``), and the strategy gate opens only on the stressed path.
- T2  vs LiquidityShockReversionStrategy: episode-not-shock -- a gated cumulative
      drift with flat volume and small per-bar returns fires here while the
      shock incumbent's per-bar conjunction stays silent (positive control:
      the same incumbent DOES fire on a genuine shock).
- T3  vs HourlyShockReversionStrategy: an isolated price shock in a calm
      (low-spread) regime is faded by the shock incumbent but this gate stays
      CLOSED (abstain).
- T4  vs StationarityGatedResidualReversionStrategy: a lockstep asset+benchmark
      crash has ~0 residual (that sleeve abstains) yet a stressed spread +
      negative trailing return fires LONG here; conversely an idiosyncratic
      residual dislocation in a calm-spread market trades the residual sleeve
      while this gate abstains.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.core.events import MarketEvent
from lumina_quant.indicators.hl_spread import corwin_schultz_spread
from lumina_quant.strategies.external_alpha_sleeves import LiquidityShockReversionStrategy
from lumina_quant.strategies.hourly_shock_reversion import HourlyShockReversionStrategy
from lumina_quant.strategies.residual_reversion_alpha_sleeves import (
    StationarityGatedResidualReversionStrategy,
)
from lumina_quant.strategies.spread_stress_reversion_alpha_sleeves import (
    _SPREAD_STRESS_REVERSION_SLICE,
    SpreadStressLiquidityReversionStrategy,
)
from lumina_quant.timeframe_aggregator import TimeframeAggregator
from lumina_quant.tuning import HyperParam

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


# --------------------------------------------------------------------------- #
# feed harness
# --------------------------------------------------------------------------- #

# OHLCV bar tuple: (open, high, low, close, volume).
_Bar = tuple[float, float, float, float, float]


def _market_event(symbol: str, t: datetime, bar: _Bar) -> SimpleNamespace:
    o, h, low, c, v = bar
    return SimpleNamespace(
        type="MARKET", time=t, symbol=symbol, open=o, high=h, low=low, close=c, volume=v
    )


def _feed_single(strategy: Any, symbol: str, bars: list[_Bar], t0: int = 0) -> None:
    for i, bar in enumerate(bars):
        strategy.calculate_signals(_market_event(symbol, _START + timedelta(days=t0 + i), bar))


def _window_event(t: datetime, rows: dict[str, _Bar]) -> SimpleNamespace:
    bars_1s = {
        sym: [{"time": t, "open": o, "high": h, "low": low, "close": c, "volume": v}]
        for sym, (o, h, low, c, v) in rows.items()
    }
    return SimpleNamespace(type="MARKET_WINDOW", time=t, bars_1s=bars_1s)


def _feed_window(strategy: Any, series: dict[str, list[_Bar]], symbols: list[str]) -> None:
    n = len(series[symbols[0]])
    for i in range(n):
        rows = {sym: series[sym][i] for sym in symbols}
        strategy.calculate_signals_window(_window_event(_START + timedelta(days=i), rows), None)


def _entries(strategy: Any) -> list[Any]:
    return [s for s in strategy.events.items if s.signal_type in {"LONG", "SHORT"}]


def _entries_for(strategy: Any, symbol: str) -> list[Any]:
    return [s for s in _entries(strategy) if s.symbol == symbol]


# --------------------------------------------------------------------------- #
# deterministic bar generators
# --------------------------------------------------------------------------- #


def _calm_bars(n: int, gen, *, base: float = 100.0, half: float = 0.003) -> list[_Bar]:
    """Fixed-band calm bars: constant [base*(1-h), base*(1+h)] high/low (full
    overlap -> CONSTANT Corwin-Schultz spread, so the z-gate never opens) with a
    mild LCG close jitter inside the band for non-degenerate realized vol."""
    hi = base * (1.0 + half)
    lo = base * (1.0 - half)
    bars: list[_Bar] = []
    for _ in range(n):
        close = base * (1.0 + (next(gen) - 0.5) * 0.003)
        bars.append((close, hi, lo, close, 1000.0))
    return bars


def _stress_down_bars(
    n: int,
    start_price: float,
    *,
    band_half: float = 0.05,
    per_bar: float = 0.008,
    volume: float = 1000.0,
) -> list[_Bar]:
    """Wide OVERLAPPING ranges (spread stress) with the close drifting DOWN."""
    bars: list[_Bar] = []
    close = start_price
    for _ in range(n):
        close *= 1.0 - per_bar
        bars.append((close, close * (1.0 + band_half), close * (1.0 - band_half), close, volume))
    return bars


_CAND_KW: dict[str, Any] = dict(
    cs_smooth_window=3,
    z_window_bars=20,
    z_entry=2.0,
    fade_lookback_bars=3,
    min_hold_bars=2,
    max_hold_bars=6,
    cooldown_bars=3,
    vol_window_bars=8,
    allow_short=True,
    min_history_bars=25,
    min_dollar_volume=0.0,
    target_vol=0.0,
    stop_loss_pct=0.0,
    take_profit_pct=0.0,
    base_allocation=0.02,
    min_price=0.01,
)

_K = math.log(1.05)
_OSCILLATION_SPREAD = 2.0 * (math.exp(_K) - 1.0) / (1.0 + math.exp(_K))


def _candidate(symbols: list[str], **overrides: Any) -> SpreadStressLiquidityReversionStrategy:
    kwargs = dict(_CAND_KW, **overrides)
    return SpreadStressLiquidityReversionStrategy(_Bars(symbols), _Queue(), **kwargs)


# --------------------------------------------------------------------------- #
# T1 -- INDICATOR ISOLATION: spread stress is not Parkinson volatility
# --------------------------------------------------------------------------- #


def test_t1_indicator_isolates_spread_from_parkinson_volatility() -> None:
    # Stage-1 premise: two paths with IDENTICAL per-bar ln(H/L) = k (equal
    # Parkinson vol) yield a ZERO (clean staircase) vs a strictly-positive
    # (full overlap) Corwin-Schultz spread.
    stair_h = [105.0 * (1.05**i) for i in range(6)]
    stair_l = [100.0 * (1.05**i) for i in range(6)]
    osc_h = [105.0] * 6
    osc_l = [100.0] * 6
    for hi, lo in list(zip(stair_h, stair_l, strict=True)) + list(zip(osc_h, osc_l, strict=True)):
        assert math.isclose(math.log(hi / lo), _K, rel_tol=1e-12)
    assert corwin_schultz_spread(stair_h, stair_l, smooth_window=5) == 0.0
    osc_spread = corwin_schultz_spread(osc_h, osc_l, smooth_window=5)
    assert osc_spread is not None and math.isclose(osc_spread, _OSCILLATION_SPREAD, rel_tol=1e-8)


def test_t1_gate_opens_only_on_the_stressed_path() -> None:
    gen = _lcg_stream(1)
    calm = _calm_bars(30, gen)
    stressed = _stress_down_bars(12, calm[-1][3])

    # Stressed feed: the OHLC-spread z clears the gate and a NEGATIVE trailing
    # return is faded -> LONG, with the stress z carried on the signal.
    stress_cand = _candidate(["A/USDT"])
    _feed_single(stress_cand, "A/USDT", calm + stressed)
    stress_entries = _entries(stress_cand)
    assert stress_entries, "stressed path must open the gate"
    assert all(sig.signal_type == "LONG" for sig in stress_entries)
    meta = stress_entries[0].metadata or {}
    assert (
        meta.get("spread_stress_z") is not None and meta["spread_stress_z"] >= _CAND_KW["z_entry"]
    )
    assert meta.get("fade_return") is not None and meta["fade_return"] < 0.0

    # Calm-only feed: constant band -> constant spread -> the gate never opens.
    calm_gen = _lcg_stream(1)
    calm_cand = _candidate(["A/USDT"])
    _feed_single(calm_cand, "A/USDT", _calm_bars(42, calm_gen))
    assert _entries(calm_cand) == []


# --------------------------------------------------------------------------- #
# T2 -- vs LiquidityShockReversionStrategy: episode, not per-bar shock
# --------------------------------------------------------------------------- #

_LSR_KW: dict[str, Any] = dict(
    volume_window=20,
    range_window=20,
    volume_shock_z=2.0,
    range_shock_z=1.5,
    return_shock_pct=0.012,
    stop_loss_pct=0.0,
    min_price=0.01,
)


def test_t2_episode_fires_where_shock_conjunction_stays_silent() -> None:
    gen = _lcg_stream(7)
    calm = _calm_bars(60, gen)
    # 8 stress bars: wide overlapping ranges, FLAT volume, per-bar |ret|=0.8%
    # (< the incumbent's 1.2% return_shock_pct), cumulative drift ~ -6%.
    stressed = _stress_down_bars(8, calm[-1][3], band_half=0.05, per_bar=0.008, volume=1000.0)

    incumbent = LiquidityShockReversionStrategy(_Bars(["A/USDT"]), _Queue(), **_LSR_KW)
    _feed_single(incumbent, "A/USDT", calm + stressed)
    # Its conjunctive trigger (`abs(ret) < return_shock_pct OR vol_z <
    # volume_shock_z OR ...`) fails every bar -> it emits nothing.
    assert _entries(incumbent) == []

    candidate = _candidate(["A/USDT"])
    _feed_single(candidate, "A/USDT", calm + stressed)
    cand_entries = _entries(candidate)
    assert cand_entries and all(sig.signal_type == "LONG" for sig in cand_entries)

    # Incumbent-LIVE positive control: the SAME incumbent DOES fire on a genuine
    # single-bar shock (|ret|=2%, 15x volume, wide range) -> not vacuously mute.
    ctrl_gen = _lcg_stream(9)
    ctrl_calm = _calm_bars(25, ctrl_gen)
    shock_close = ctrl_calm[-1][3] * 0.98
    shock_bar: _Bar = (shock_close, shock_close * 1.05, shock_close * 0.95, shock_close, 15000.0)
    control = LiquidityShockReversionStrategy(_Bars(["A/USDT"]), _Queue(), **_LSR_KW)
    _feed_single(control, "A/USDT", [*ctrl_calm, shock_bar])
    assert [sig.signal_type for sig in _entries(control)] == ["LONG"]


# --------------------------------------------------------------------------- #
# T3 -- vs HourlyShockReversionStrategy: calm-regime shock -> gate closed
# --------------------------------------------------------------------------- #


def test_t3_calm_regime_shock_faded_by_incumbent_but_gate_stays_closed() -> None:
    # Incumbent-LIVE: HourlyShockReversion fades a completed -2% 1h bar.
    queue = _Queue()
    incumbent = HourlyShockReversionStrategy(
        _Bars(["ETH/USDT"]),
        queue,
        lookback_bars=1,
        return_threshold=0.006,
        max_hold_bars=48,
        stop_loss_pct=0.0,
    )
    aggregator = TimeframeAggregator(timeframes=["1h"], lookbacks={"1h": 16})
    for offset, close in enumerate([100.0, 100.0, 100.0, 98.0, 98.0]):
        event = MarketEvent(
            time=_START + timedelta(hours=offset),
            symbol="ETH/USDT",
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0,
        )
        aggregator.update_from_1s_batch("ETH/USDT", [event])
        incumbent.calculate_signals_window(event, aggregator)
    incumbent_entries = _entries(incumbent)
    assert [sig.signal_type for sig in incumbent_entries] == ["LONG"]
    assert incumbent_entries[0].metadata["shock_return"] <= -0.006

    # Candidate: the SAME -2% close move inside a calm (low-spread) band leaves
    # the gate CLOSED -> abstain (no bolted-on shock leg).
    gen = _lcg_stream(3)
    calm = _calm_bars(35, gen, base=100.0)
    dropped = _calm_bars(6, gen, base=98.0)  # -2% close move, still a calm spread band
    candidate = _candidate(["A/USDT"])
    _feed_single(candidate, "A/USDT", calm + dropped)
    assert _entries(candidate) == []


# --------------------------------------------------------------------------- #
# T4 -- vs StationarityGatedResidualReversionStrategy: lockstep vs idiosyncratic
# --------------------------------------------------------------------------- #

_SGR_KW: dict[str, Any] = dict(
    benchmark_symbol="BTC/USDT",
    lookback_bars=3,
    beta_window=8,
    vol_window=5,
    adf_window=9,
    adf_lags=0,
    adf_reject_threshold=-2.0,
    entry_score=0.1,
    rebalance_bars=1,
    min_hold_bars=0,
    max_hold_bars=50,
    min_dollar_volume=0.0,
    min_price=0.01,
)

_SYMS = ["A/USDT", "BTC/USDT"]


def _lockstep_series(n: int) -> dict[str, list[_Bar]]:
    """A and BTC move with IDENTICAL returns (BTC at 2x the level -> residual
    ~0), calm ranges then a wide-overlapping-range lockstep -5% crash."""
    a: list[_Bar] = []
    b: list[_Bar] = []
    for i in range(n):
        if i < n - 8:
            price_a = 100.0 * (1.0 + 0.0005 * i)
            wide = 0.0001
        else:
            j = i - (n - 8)
            price_a = 100.0 * (1.0 + 0.0005 * (n - 9)) * (0.99 ** (j + 1))
            wide = 0.05
        a.append((price_a, price_a * (1 + wide), price_a * (1 - wide), price_a, 1000.0))
        price_b = 2.0 * price_a
        b.append((price_b, price_b * (1 + wide), price_b * (1 - wide), price_b, 1000.0))
    return {"A/USDT": a, "BTC/USDT": b}


def _idiosyncratic_series(n: int) -> dict[str, list[_Bar]]:
    """BTC ~flat; A oscillates +-3% each bar (a strongly mean-reverting, hence
    ADF-stationary, idiosyncratic residual) with NARROW per-bar ranges."""
    a: list[_Bar] = []
    b: list[_Bar] = []
    gen = _lcg_stream(11)
    for i in range(n):
        price_b = 100.0 * (1.0 + (next(gen) - 0.5) * 0.001)
        price_a = 100.0 * (1.0 + (0.03 if i % 2 == 0 else -0.03))
        a.append((price_a, price_a * 1.0008, price_a * 0.9992, price_a, 1000.0))
        b.append((price_b, price_b * 1.0008, price_b * 0.9992, price_b, 1000.0))
    return {"A/USDT": a, "BTC/USDT": b}


def test_t4_lockstep_crash_fires_where_residual_sleeve_abstains() -> None:
    series = _lockstep_series(30)

    incumbent = StationarityGatedResidualReversionStrategy(_Bars(_SYMS), _Queue(), **_SGR_KW)
    _feed_window(incumbent, series, _SYMS)
    # Lockstep -> beta-hedged residual ~0 -> the residual sleeve never trades A.
    assert _entries_for(incumbent, "A/USDT") == []

    candidate = _candidate(_SYMS)
    _feed_window(candidate, series, _SYMS)
    a_entries = _entries_for(candidate, "A/USDT")
    assert a_entries and all(sig.signal_type == "LONG" for sig in a_entries)
    assert (a_entries[0].metadata or {}).get("spread_stress_z", 0.0) >= _CAND_KW["z_entry"]


def test_t4_idiosyncratic_dislocation_trades_residual_sleeve_but_gate_abstains() -> None:
    series = _idiosyncratic_series(30)

    # Incumbent-LIVE control: the residual sleeve DOES trade the stationary
    # idiosyncratic dislocation.
    incumbent = StationarityGatedResidualReversionStrategy(_Bars(_SYMS), _Queue(), **_SGR_KW)
    _feed_window(incumbent, series, _SYMS)
    assert _entries_for(incumbent, "A/USDT"), "residual sleeve must trade the idio dislocation"

    # Candidate: calm per-bar ranges -> spread gate closed -> abstain, even
    # though the close swings +-3% each bar.
    candidate = _candidate(_SYMS)
    _feed_window(candidate, series, _SYMS)
    assert _entries(candidate) == []


# --------------------------------------------------------------------------- #
# hygiene: determinism / state roundtrip / adversarial / degenerate / schema
# --------------------------------------------------------------------------- #


def _stress_scenario() -> tuple[list[str], list[_Bar]]:
    gen = _lcg_stream(1)
    calm = _calm_bars(30, gen)
    return ["A/USDT"], calm + _stress_down_bars(12, calm[-1][3])


def test_determinism_two_runs_identical_signals() -> None:
    symbols, bars = _stress_scenario()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = _candidate(symbols)
        _feed_single(strategy, symbols[0], bars)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    assert first, "expected at least one signal in the stress scenario"
    assert first == _run()


def test_state_roundtrip_lossless() -> None:
    symbols, bars = _stress_scenario()
    strategy = _candidate(symbols)
    _feed_single(strategy, symbols[0], bars)
    snapshot = strategy.get_state()

    restored = _candidate(symbols)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    assert restored._tick == strategy._tick
    for symbol in symbols:
        assert restored._state[symbol].mode == strategy._state[symbol].mode
        assert list(restored._state[symbol].spreads) == list(strategy._state[symbol].spreads)


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT"]
    strategy = _candidate(symbols)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "nope"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"highs": 12345}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "highs": ["x", float("nan"), 1.0],
                    "lows": {"bad": "type"},
                    "closes": [float("inf"), 2.0, None],
                    "volumes": "nope",
                    "spreads": [-1.0, "bad", 0.01],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "cooldown_remaining": -4,
                    "in_episode": "yes",
                    "episode_entered": None,
                    "last_time_key": 123,
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
    # Still functional afterward.
    _, bars = _stress_scenario()
    _feed_single(strategy, "A/USDT", bars)


def test_degenerate_inputs_never_raise() -> None:
    strategy = _candidate(["Z/USDT"])
    # H < L, non-positive, NaN, inf closes, missing volume.
    strategy.calculate_signals(_market_event("Z/USDT", _START, (100.0, 90.0, 110.0, 0.0, None)))
    strategy.calculate_signals(
        _market_event("Z/USDT", _START + timedelta(days=1), (-5.0, -5.0, -5.0, -5.0, 1.0))
    )
    strategy.calculate_signals(
        _market_event(
            "Z/USDT",
            _START + timedelta(days=2),
            (float("nan"), float("nan"), float("nan"), float("nan"), 1.0),
        )
    )
    strategy.calculate_signals(
        _market_event(
            "Z/USDT",
            _START + timedelta(days=3),
            (float("inf"), float("inf"), float("inf"), float("inf"), 1.0),
        )
    )
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    assert _entries(strategy) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = SpreadStressLiquidityReversionStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "cs_smooth_window",
        "z_window_bars",
        "z_entry",
        "fade_lookback_bars",
        "min_hold_bars",
        "max_hold_bars",
        "cooldown_bars",
        "one_entry_per_episode",
        "vol_window_bars",
        "min_history_bars",
    ):
        assert required in schema


def test_slice_multi_timeframe_keys_and_bounds() -> None:
    """Pin the 1d/4h/1h slice: mirrored variants + keys, scaled cells in-bounds.

    Also pins the load-bearing COST decision for this episodic fade lane:
    ``z_entry`` is RAISED (not held) at finer tf so the wall-clock episode rate
    does not explode, and the hold/cooldown clocks scale x6/x24 so wall-clock
    hold durations are preserved.
    """
    slice_dict = _SPREAD_STRESS_REVERSION_SLICE
    assert set(slice_dict) == {"1d", "4h", "1h"}
    counts = {tf: len(cells) for tf, cells in slice_dict.items()}
    assert len(set(counts.values())) == 1, counts
    base = {cell["variant"]: cell for cell in slice_dict["1d"]}
    schema = SpreadStressLiquidityReversionStrategy.get_param_schema()
    for tf, cells in slice_dict.items():
        assert tuple(c["variant"] for c in cells) == tuple(base), (tf, counts)
        for cell in cells:
            assert set(cell) == set(base[cell["variant"]]), (tf, cell["variant"])
            for key, value in cell.items():
                if key == "variant" or isinstance(value, bool):
                    continue
                hp = schema[key]
                if hp.low is not None:
                    assert value >= hp.low, (tf, key, value)
                if hp.high is not None:
                    assert value <= hp.high, (tf, key, value)
    # Hold/cooldown clocks scale wall-clock (x6 at 4h, x24 at 1h).
    for factor, tf in ((6, "4h"), (24, "1h")):
        for cell in slice_dict[tf]:
            b = base[cell["variant"]]
            for clock in ("min_hold_bars", "max_hold_bars", "cooldown_bars"):
                assert cell[clock] == factor * b[clock], (tf, clock)
            # Episode-frequency guard: the stress threshold is raised sub-daily.
            assert cell["z_entry"] > b["z_entry"], (tf, cell["variant"])


# --------------------------------------------------------------------------- #
# vol-target horizon regression (worker-vt2)
#
# The entry size scalar is ``min(1, target_vol / vol)`` where ``vol`` is a
# PER-BAR realized vol; it MUST be annualized via sqrt(bars_per_year) from
# observed bar spacing first, else the Moreira-Muir clamp is inert.  NOTE the
# sleeve default is ``target_vol=0.0`` (throttle off), so these tests set it > 0.
# --------------------------------------------------------------------------- #

_VT_SYM = "A/USDT"
_HOUR_EPOCHS = [1_700_000_000.0 + i * 3600.0 for i in range(12)]  # 1h spacing


def _vt_fill(item: Any, factor_up: float, factor_dn: float) -> None:
    """Fill a per-symbol state with an oscillating, net-declining close path.

    Net decline (each up/down pair < 1) makes ``fade_return`` negative -> LONG;
    the oscillation magnitude sets the per-bar realized vol the throttle reads.
    """
    price = 100.0
    for i in range(26):  # >= min_history_bars (25) so _liquid passes
        price *= factor_up if i % 2 == 0 else factor_dn
        item.closes.append(price)
        item.volumes.append(1000.0)


def _vt_entry_signal(strat: Any, symbol: str) -> Any:
    sigs = [s for s in strat.events.items if str(s.signal_type).upper() in {"LONG", "SHORT"}]
    return sigs[-1] if sigs else None


def test_vol_target_throttle_active_on_hourly_high_vol() -> None:
    strat = _candidate([_VT_SYM], target_vol=0.20)
    _vt_fill(strat._state[_VT_SYM], 1.04, 0.95)  # large per-bar vol
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    strat._maybe_enter(_VT_SYM, strat._state[_VT_SYM], _START, 3.0)
    sig = _vt_entry_signal(strat, _VT_SYM)
    assert sig is not None, strat.events.items
    assert sig.metadata["inverse_vol_scalar"] < 1.0
    assert sig.metadata["target_allocation"] < strat.base_allocation


def test_vol_target_passthrough_without_bar_spacing() -> None:
    strat = _candidate([_VT_SYM], target_vol=0.20)  # empty _recent_times
    _vt_fill(strat._state[_VT_SYM], 1.04, 0.95)
    strat._maybe_enter(_VT_SYM, strat._state[_VT_SYM], _START, 3.0)
    sig = _vt_entry_signal(strat, _VT_SYM)
    assert sig is not None, strat.events.items
    assert sig.metadata["inverse_vol_scalar"] == 1.0  # pass-through, not throttled
    assert sig.metadata["target_allocation"] == strat.base_allocation


def test_vol_target_calm_leaves_size_unthrottled() -> None:
    strat = _candidate([_VT_SYM], target_vol=0.20)
    _vt_fill(strat._state[_VT_SYM], 1.001, 0.9985)  # mild vol -> annualized < target
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    strat._maybe_enter(_VT_SYM, strat._state[_VT_SYM], _START, 3.0)
    sig = _vt_entry_signal(strat, _VT_SYM)
    assert sig is not None, strat.events.items
    assert sig.metadata["inverse_vol_scalar"] == 1.0


def test_vol_target_size_scalar_deterministic() -> None:
    scalars = []
    for _ in range(2):
        strat = _candidate([_VT_SYM], target_vol=0.20)
        _vt_fill(strat._state[_VT_SYM], 1.04, 0.95)
        for epoch in _HOUR_EPOCHS:
            strat._recent_times.append(epoch)
        strat._maybe_enter(_VT_SYM, strat._state[_VT_SYM], _START, 3.0)
        scalars.append(_vt_entry_signal(strat, _VT_SYM).metadata["inverse_vol_scalar"])
    assert scalars[0] == scalars[1]


def test_vol_target_epochs_tracked_from_datetime_feed() -> None:
    strat = _candidate([_VT_SYM])
    for idx in range(6):
        epoch = 1_700_000_000.0 + idx * 3600.0  # numeric epoch -> parsed to a datetime
        row = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}
        strat.calculate_signals_window(
            SimpleNamespace(
                type="MARKET_WINDOW",
                time=epoch,
                bars_1s={_VT_SYM: [dict(row, time=epoch)]},
            ),
            None,
        )
    times = list(strat._recent_times)
    assert len(times) >= 5, times
    gaps = [round(times[i + 1] - times[i]) for i in range(len(times) - 1)]
    assert gaps and all(gap == 3600 for gap in gaps), gaps


def test_vol_target_recent_times_survive_state_roundtrip() -> None:
    strat = _candidate([_VT_SYM])
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    restored = _candidate([_VT_SYM])
    restored.set_state(strat.get_state())
    assert list(restored._recent_times) == list(_HOUR_EPOCHS)
