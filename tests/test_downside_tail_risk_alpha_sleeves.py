"""Deterministic build-gate + hygiene tests for the downside-tail-risk sleeve.

Direct class import only (no ``@register`` on this lane).  The build gate RUNS
the real incumbents on hand-built synthetic panels and asserts materially
different emitted actions:

  * Leg 1 (vs ``IdiosyncraticVolatilityStrategy``): on an equal-total-vol panel
    the candidate LONGs the worst left-tail name and SHORTs the mildest, while
    the idio-vol incumbent (proven live on a control panel) ranks by VOL -- its
    long pick is a low-vol helper, not the worst-tail name, and it never longs
    the candidate's top pick.
  * Leg 2 (vs ``LotterySkewnessStrategy``): a fat-both-tails name is LONGed by
    the candidate (worst left tail) and SHORTed by the lottery incumbent (high
    MAX) -- opposite sides on identical input.
  * Leg 3a (vs ``TailIndexRegimeRiderStrategy``): a trendless panel makes the
    rider's trend gate return no regime (asserted at ``_trend_sign``/
    ``_regime_direction``) so it emits nothing, while the candidate emits a full
    XS book; a trending control fixture proves the rider live.
  * Leg 3b (SHAPE vs LEVEL): two symbols whose loss tails are scalar multiples
    share an identical Hill exponent (the rider's statistic is indifferent) but
    different expected shortfall, and the candidate ranks them apart.
  * Min-hold rescue + hygiene: a would-be flip inside min-hold is suppressed;
    determinism, state roundtrip, adversarial set_state, self-skip, never-raise.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.indicators.tail_index import hill_tail_index
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    IdiosyncraticVolatilityStrategy,
    LotterySkewnessStrategy,
)
from lumina_quant.strategies.downside_tail_risk_alpha_sleeves import (
    _DOWNSIDE_TAIL_RISK_SLICE,
    DownsideTailRiskPremiumStrategy,
    _average_ranks,
    _expected_shortfall,
    _log_returns,
    _rank_residual,
)
from lumina_quant.strategies.tail_index_alpha_sleeves import (
    TailIndexRegimeRiderStrategy,
    _trend_sign,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _window_event(idx: int, rows: dict[str, dict[str, Any]]) -> SimpleNamespace:
    time = f"t{idx:05d}"
    return SimpleNamespace(
        type="MARKET_WINDOW",
        time=time,
        bars_1s={sym: [dict(row, time=time)] for sym, row in rows.items()},
    )


def _feed(strategy: Any, series: dict[str, list[dict[str, Any]]]) -> None:
    n = len(next(iter(series.values())))
    for idx in range(n):
        strategy.calculate_signals(_window_event(idx, {sym: series[sym][idx] for sym in series}))


def _final_side(items: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in items:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def _non_exit(items: list[Any]) -> list[Any]:
    return [sig for sig in items if str(sig.signal_type).upper() != "EXIT"]


# --------------------------------------------------------------------------- #
# return / price builders
# --------------------------------------------------------------------------- #


def _rescale(returns: list[float], *, target: float = 0.01) -> list[float]:
    sigma = sample_std(returns)
    assert sigma is not None
    return [value / sigma * target for value in returns]


def _closes_from(returns: list[float], p0: float = 100.0) -> list[float]:
    out = [p0]
    price = p0
    for value in returns:
        price *= math.exp(value)
        out.append(price)
    return out


def _bars(closes: list[float]) -> list[dict[str, Any]]:
    return [{"open": c, "high": c, "low": c, "close": c, "volume": 1000.0} for c in closes]


def _rep(values: list[float], n: int) -> list[float]:
    return [values[i % len(values)] for i in range(n)]


def _bars_from(returns: list[float]) -> list[dict[str, Any]]:
    return _bars(_closes_from(returns))


_W = 60


def _window_returns(bars: list[dict[str, Any]]) -> list[float]:
    return _log_returns([b["close"] for b in bars])[-_W:]


# --------------------------------------------------------------------------- #
# Leg 1: worst-tail vs idiosyncratic volatility
# --------------------------------------------------------------------------- #

_A_RAW = _rep([+0.5, -0.5, +0.4, -0.4], _W)
_B_RAW = [-6.0 if i % 20 == 19 else +0.35 for i in range(_W)]  # fat LEFT tail
_C_RAW = [-x for x in _B_RAW]  # fat RIGHT tail (mildest left)
_E_RAW = [
    (-6.0 if i % 20 == 19 else (+6.0 if i % 20 == 9 else (+0.5 if i % 2 == 0 else -0.5)))
    for i in range(_W)
]
_BTC_RAW = _rep([+0.45, -0.55, +0.55, -0.45], _W)

_CANDIDATE_KWARGS: dict[str, Any] = dict(
    es_window=_W,
    tail_q=0.05,
    vol_neutralize=True,
    min_history_bars=_W,
    quantile_pct=0.15,
    rebalance_bars=1,
    min_hold_periods=0,
    hysteresis_band=0.0,
    min_symbols=5,
    allow_short=True,
    target_vol=0.0,
    stop_loss_pct=0.0,
    min_price=0.01,
)


def _panel1() -> dict[str, list[dict[str, Any]]]:
    return {
        "BTC/USDT": _bars_from(_rescale(_BTC_RAW)),
        "A/USDT": _bars_from(_rescale(_A_RAW)),
        "B/USDT": _bars_from(_rescale(_B_RAW)),
        "C/USDT": _bars_from(_rescale(_C_RAW)),
        "E/USDT": _bars_from(_rescale(_E_RAW)),
        "LOWVOL/USDT": _bars_from(_rescale(_A_RAW, target=0.002)),
        "HIGHVOL/USDT": _bars_from(_rescale(_A_RAW, target=0.05)),
    }


def test_leg1_worst_tail_vs_idiosyncratic_volatility() -> None:
    series = _panel1()
    arch = ["A/USDT", "B/USDT", "C/USDT", "E/USDT", "BTC/USDT"]
    stds = {s: sample_std(_window_returns(series[s])) for s in arch}
    es = {s: _expected_shortfall(_window_returns(series[s]), 0.05) for s in arch}

    # Stage-1 premises: the archetypes have equal total vol (within 1e-9), B has
    # the worst signed tail and C the mildest.
    assert max(stds.values()) - min(stds.values()) < 1e-9  # type: ignore[operator]
    assert es["B/USDT"] == min(es.values())
    assert es["C/USDT"] == max(es.values())

    syms = list(series)
    candidate = DownsideTailRiskPremiumStrategy(_Bars(syms), _Queue(), **_CANDIDATE_KWARGS)
    _feed(candidate, series)
    cand = _final_side(candidate.events.items)
    # Candidate deterministically LONGs the worst tail and SHORTs the mildest.
    assert cand.get("B/USDT") == "LONG"
    assert cand.get("C/USDT") == "SHORT"

    idio = IdiosyncraticVolatilityStrategy(
        _Bars(syms),
        _Queue(),
        benchmark_symbol="BTC/USDT",
        beta_window=_W,
        vol_window=_W // 2,
        rebalance_bars=1,
        quantile_pct=0.20,
        min_symbols=5,
        allow_short=True,
        stop_loss_pct=0.0,
        max_hold_bars=100,
        min_price=0.01,
    )
    _feed(idio, series)
    idio_side = _final_side(idio.events.items)
    # Divergent action: idio-vol ranks by VOLATILITY -- it LONGs the low-vol
    # helper, NOT the candidate's worst-tail pick B, which it never longs.
    assert idio_side.get("LOWVOL/USDT") == "LONG"
    assert idio_side.get("B/USDT") != "LONG"


def test_leg1_idio_vol_live_on_unequal_vol_control() -> None:
    # Incumbent-LIVE control: with genuine vol dispersion idio-vol emits a
    # definite book, proving the flat behaviour on the equal-vol panel is
    # mechanism (vol-blind to tail shape), not a dead harness.
    series: dict[str, list[dict[str, Any]]] = {}
    base = _rep([+0.5, -0.5, +0.3, -0.7, +0.6, -0.4], _W)
    for sym, scale in (
        ("BTC/USDT", 1.0),
        ("H1/USDT", 3.0),
        ("H2/USDT", 2.5),
        ("L1/USDT", 0.3),
        ("L2/USDT", 0.4),
        ("M1/USDT", 1.0),
        ("M2/USDT", 1.2),
    ):
        series[sym] = _bars_from([scale * x for x in _rescale(base)])
    idio = IdiosyncraticVolatilityStrategy(
        _Bars(list(series)),
        _Queue(),
        benchmark_symbol="BTC/USDT",
        beta_window=_W,
        vol_window=_W // 2,
        rebalance_bars=1,
        quantile_pct=0.20,
        min_symbols=5,
        allow_short=True,
        stop_loss_pct=0.0,
        max_hold_bars=100,
        min_price=0.01,
    )
    _feed(idio, series)
    assert _final_side(idio.events.items)


# --------------------------------------------------------------------------- #
# Leg 2: worst left tail vs lottery / MAX
# --------------------------------------------------------------------------- #


def test_leg2_worst_left_tail_vs_lottery_opposite_side() -> None:
    series: dict[str, list[dict[str, Any]]] = {"E/USDT": _bars_from(_rescale(_E_RAW))}
    for j in range(5):
        series[f"G{j}/USDT"] = _bars_from(
            _rescale(_rep([+0.5 * (1 + 0.1 * j), -0.5 * (1 + 0.1 * j), +0.4, -0.4], _W))
        )
    syms = list(series)

    candidate = DownsideTailRiskPremiumStrategy(_Bars(syms), _Queue(), **_CANDIDATE_KWARGS)
    _feed(candidate, series)
    cand = _final_side(candidate.events.items)

    lottery = LotterySkewnessStrategy(
        _Bars(syms),
        _Queue(),
        skew_window=_W,
        max_window=_W // 2,
        rebalance_bars=1,
        quantile_pct=0.20,
        min_symbols=5,
        allow_short=True,
        stop_loss_pct=0.0,
        max_hold_bars=100,
        min_price=0.01,
    )
    _feed(lottery, series)
    lot = _final_side(lottery.events.items)

    assert cand and lot
    # Opposite sides on E: worst left tail (LONG) vs high MAX / lottery (SHORT).
    assert cand.get("E/USDT") == "LONG"
    assert lot.get("E/USDT") == "SHORT"


# --------------------------------------------------------------------------- #
# Leg 3a: trend gate silence vs candidate XS book
# --------------------------------------------------------------------------- #

_TW = 30
_OSC_N = 80


def _osc_closes(dipmag: float, dipevery: int = 15) -> list[float]:
    out: list[float] = []
    for i in range(_OSC_N):
        close = 100.0 + (0.05 if i % 2 == 0 else -0.05)
        if dipevery and i % dipevery == dipevery - 1:
            close = 100.0 * (1.0 - dipmag)
        out.append(close)
    return out


def _rider(symbols: list[str], **overrides: Any) -> TailIndexRegimeRiderStrategy:
    params = dict(
        tail_window=_TW,
        recent_window=10,
        k_short=4,
        k_long=8,
        fatten_ratio=0.85,
        thin_ratio=1.15,
        trend_lookback=20,
        min_trend_roc=0.15,
        allow_short=True,
        min_price=0.01,
    )
    params.update(overrides)
    return TailIndexRegimeRiderStrategy(_Bars(symbols), _Queue(), **params)


def test_leg3a_trend_gate_silence_vs_candidate_book() -> None:
    series = {
        f"O{j}/USDT": [
            {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1000.0}
            for c in _osc_closes(d)
        ]
        for j, d in enumerate([0.10, 0.04, 0.02, 0.06, 0.08])
    }
    syms = list(series)

    candidate = DownsideTailRiskPremiumStrategy(
        _Bars(syms),
        _Queue(),
        es_window=_TW,
        tail_q=0.10,
        vol_neutralize=True,
        min_history_bars=_TW,
        quantile_pct=0.20,
        rebalance_bars=1,
        min_hold_periods=0,
        hysteresis_band=0.0,
        min_symbols=5,
        allow_short=True,
        target_vol=0.0,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(candidate, series)
    # Candidate emits a full XS long-short book on the trendless panel.
    cand = _final_side(candidate.events.items)
    assert any(v == "LONG" for v in cand.values())
    assert any(v == "SHORT" for v in cand.values())

    rider = _rider(syms)
    _feed(rider, series)
    # WHY the rider is silent: the trend gate reads neutral for every symbol, so
    # ``_regime_direction`` returns '' and no entry is emitted.
    for sym in syms:
        item = rider._state[sym]
        assert _trend_sign(list(item.closes), trend_lookback=20, min_trend_roc=0.15) == 0
        assert rider._regime_direction(sym, item) == ""
    assert _final_side(rider.events.items) == {}


def test_leg3a_rider_live_on_trending_control() -> None:
    closes: list[float] = []
    price = 100.0
    for i in range(_OSC_N):
        price *= (1.0 - 0.004) if i < _OSC_N - 15 else (1.0 - 0.05)
        closes.append(price)
    series = {
        "CTRL/USDT": [
            {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1000.0}
            for c in closes
        ]
    }
    rider = _rider(["CTRL/USDT"])
    _feed(rider, series)
    item = rider._state["CTRL/USDT"]
    # A confirmed downtrend with a fattening recent loss tail arms a SHORT.
    assert _trend_sign(list(item.closes), trend_lookback=20, min_trend_roc=0.15) == -1
    assert rider._regime_direction("CTRL/USDT", item) == "SHORT"
    assert _final_side(rider.events.items).get("CTRL/USDT") == "SHORT"


# --------------------------------------------------------------------------- #
# Leg 3b: SHAPE (Hill) invariance vs LEVEL (ES) separation
# --------------------------------------------------------------------------- #


def test_leg3b_shape_invariant_but_level_separates() -> None:
    x_raw = [-(0.02 + 0.004 * ((i // 3) % 5)) if i % 3 == 2 else +0.006 for i in range(_W)]
    y_raw = [2 * x for x in x_raw]  # loss tail is a scalar multiple of x's
    series: dict[str, list[dict[str, Any]]] = {
        "X/USDT": _bars_from(x_raw),
        "Y/USDT": _bars_from(y_raw),
    }
    for j in range(4):
        series[f"G{j}/USDT"] = _bars_from(_rep([+0.004, -0.004, +0.003, -0.003], _W))
    syms = list(series)

    # Candidate (raw ES level) ranks the deeper-tailed Y strictly above X.
    candidate = DownsideTailRiskPremiumStrategy(
        _Bars(syms),
        _Queue(),
        es_window=_W,
        tail_q=0.10,
        vol_neutralize=False,
        min_history_bars=_W,
        quantile_pct=0.15,
        rebalance_bars=1,
        min_hold_periods=0,
        hysteresis_band=0.0,
        min_symbols=5,
        allow_short=True,
        target_vol=0.0,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(candidate, series)
    scores, _sigmas, _metas = candidate._residual_scores()
    assert scores["X/USDT"] < scores["Y/USDT"]

    # The rider's Hill statistic (loss-tail SHAPE) is identical for the two --
    # scale-invariant on a scalar-multiple tail -- so it is indifferent.
    rider = _rider(syms, tail_window=_TW, recent_window=12, k_short=3, k_long=5, min_trend_roc=0.0)
    _feed(rider, series)
    _hs_x, hb_x, _r_x = rider._hill_metrics("X/USDT")
    _hs_y, hb_y, _r_y = rider._hill_metrics("Y/USDT")
    assert hb_x is not None and hb_y is not None
    assert abs(hb_x - hb_y) < 1e-9


# --------------------------------------------------------------------------- #
# Min-hold rescue + hygiene
# --------------------------------------------------------------------------- #


def _swap_universe() -> dict[str, list[dict[str, Any]]]:
    phase1, phase2 = 45, 25
    xx = [(-0.06 if i % 10 == 9 else +0.006) for i in range(phase1)] + [+0.006] * phase2
    yy = [+0.006] * phase1 + [(-0.06 if i % 10 == 9 else +0.006) for i in range(phase2)]
    series: dict[str, list[dict[str, Any]]] = {
        "XX/USDT": _bars_from(xx),
        "YY/USDT": _bars_from(yy),
    }
    for j in range(4):
        series[f"H{j}/USDT"] = _bars_from(_rep([+0.004, -0.004], phase1 + phase2))
    return series


def test_min_hold_suppresses_flip() -> None:
    series = _swap_universe()
    syms = list(series)
    base = dict(
        es_window=30,
        tail_q=0.10,
        vol_neutralize=True,
        min_history_bars=30,
        quantile_pct=0.20,
        rebalance_bars=1,
        hysteresis_band=0.0,
        min_symbols=5,
        allow_short=True,
        target_vol=0.0,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    free = DownsideTailRiskPremiumStrategy(_Bars(syms), _Queue(), **dict(base, min_hold_periods=0))
    _feed(free, series)
    held = DownsideTailRiskPremiumStrategy(
        _Bars(syms), _Queue(), **dict(base, min_hold_periods=100)
    )
    _feed(held, series)
    # XX starts LONG (worst tail in phase 1); once its tail rolls out of the
    # window the free book releases it while the long min-hold keeps it LONG.
    assert _final_side(held.events.items).get("XX/USDT") == "LONG"
    assert _final_side(free.events.items).get("XX/USDT") != "LONG"


def test_rank_helpers_tie_and_residual() -> None:
    # Tie-aware ranks collapse a constant axis so vol-neutralization is a no-op.
    assert _average_ranks([1.0, 1.0, 1.0, 1.0]) == [1.5, 1.5, 1.5, 1.5]
    assert _rank_residual([0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 5.0, 5.0]) == [-1.5, -0.5, 0.5, 1.5]
    es = _expected_shortfall([-0.1, -0.05, 0.0, 0.02], 0.5)
    assert es is not None and abs(es - (-0.075)) < 1e-12


def test_determinism_two_runs_identical_signals() -> None:
    series = _panel1()
    syms = list(series)

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strat = DownsideTailRiskPremiumStrategy(_Bars(syms), _Queue(), **_CANDIDATE_KWARGS)
        _feed(strat, series)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strat.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal"


def test_state_roundtrip_lossless() -> None:
    series = _panel1()
    syms = list(series)
    strat = DownsideTailRiskPremiumStrategy(_Bars(syms), _Queue(), **_CANDIDATE_KWARGS)
    _feed(strat, series)
    snapshot = strat.get_state()
    restored = DownsideTailRiskPremiumStrategy(_Bars(syms), _Queue(), **_CANDIDATE_KWARGS)
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot


def test_adversarial_set_state_never_raises() -> None:
    syms = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]
    strat = DownsideTailRiskPremiumStrategy(_Bars(syms), _Queue(), **_CANDIDATE_KWARGS)
    strat.set_state(None)  # type: ignore[arg-type]
    strat.set_state("nope")  # type: ignore[arg-type]
    strat.set_state({"symbol_state": "not a dict"})
    strat.set_state({"symbol_state": {"A/USDT": 123}})
    strat.set_state(
        {
            "tick": "bad",
            "symbol_state": {
                sym: {
                    "closes": ["x", float("nan"), 1.0],
                    "volumes": {"bad": "type"},
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "last_time_key": 5,
                }
                for sym in syms
            },
        }
    )
    for item in strat._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


def test_degenerate_input_never_raises() -> None:
    strat = DownsideTailRiskPremiumStrategy(_Bars(["Z/USDT"]), _Queue(), **_CANDIDATE_KWARGS)
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t0", close=0.0))
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t1", close=-1.0))
    strat.calculate_signals(
        SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t2", close=float("nan"))
    )
    strat.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t3", bars_1s={}))
    strat.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    assert _non_exit(strat.events.items) == []


def test_self_skip_below_min_symbols() -> None:
    series = {s: _bars_from(_rescale(_A_RAW)) for s in ("A/USDT", "B/USDT", "C/USDT")}
    strat = DownsideTailRiskPremiumStrategy(
        _Bars(list(series)), _Queue(), **dict(_CANDIDATE_KWARGS, min_symbols=5)
    )
    _feed(strat, series)
    assert _non_exit(strat.events.items) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = DownsideTailRiskPremiumStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "es_window",
        "tail_q",
        "vol_neutralize",
        "quantile_pct",
        "min_hold_periods",
        "hysteresis_band",
        "min_symbols",
    ):
        assert required in schema


def test_slice_timeframe_expansion_scales_bar_windows() -> None:
    """4h/1h cells mirror the 1d variant: ES / min-history windows + the bar-count
    weekly rebalance clock scale x6/x24, while the tail fraction / min-hold /
    hysteresis stay timeframe-invariant."""
    slice_ = _DOWNSIDE_TAIL_RISK_SLICE
    assert set(slice_) == {"4h", "1h", "1d"}
    variants = {tf: tuple(cell["variant"] for cell in cells) for tf, cells in slice_.items()}
    assert variants["4h"] == variants["1h"] == variants["1d"]
    by = {tf: {cell["variant"]: cell for cell in cells} for tf, cells in slice_.items()}
    for variant in variants["1d"]:
        d, h4, h1 = by["1d"][variant], by["4h"][variant], by["1h"][variant]
        for key in ("es_window", "min_history_bars"):
            assert h4[key] == d[key] * 6, (variant, key)
            assert h1[key] == d[key] * 24, (variant, key)
        assert (d["rebalance_bars"], h4["rebalance_bars"], h1["rebalance_bars"]) == (7, 42, 168)
        for key in (
            "tail_q",
            "min_hold_periods",
            "hysteresis_band",
            "quantile_pct",
            "target_gross_exposure",
            "target_vol",
            "stop_loss_pct",
        ):
            assert h4[key] == d[key] == h1[key], (variant, key)


# --------------------------------------------------------------------------- #
# vol-target horizon regression (worker-vt2)
#
# The ``target_vol`` scalar compares an annual-scale target (0.20) against the
# inverse-vol-weighted PER-BAR portfolio vol; it MUST annualize the per-bar
# estimate via sqrt(bars_per_year) inferred from observed bar spacing, else the
# Moreira-Muir clamp is inert.  The risk-parity WEIGHTS are horizon-free
# (normalized inverse-vol cancels), so only the SCALAR is annualized.
# --------------------------------------------------------------------------- #

_VT_SYMS = ["A", "B", "C", "D"]
_HOUR_EPOCHS = [1_700_000_000.0 + i * 3600.0 for i in range(12)]  # 1h spacing


def _vt_sleeve() -> DownsideTailRiskPremiumStrategy:
    return DownsideTailRiskPremiumStrategy(_Bars(_VT_SYMS), _Queue())


def _vt_targets() -> dict[str, Any]:
    return {sym: ("LONG", 1.0, {}) for sym in _VT_SYMS}


def test_vol_target_throttle_active_on_hourly_high_vol() -> None:
    strat = _vt_sleeve()
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    sigmas = dict.fromkeys(_VT_SYMS, 0.05)  # per-bar 5% vol
    _weights, scalar = strat._inverse_vol_weights(_vt_targets(), sigmas)
    assert scalar < 0.2, scalar
    portfolio_vol_ann = annualize_per_bar_vol(0.05, bars_per_year_from_spacing(list(_HOUR_EPOCHS)))
    assert scalar == min(1.0, 0.20 / portfolio_vol_ann)


def test_vol_target_passthrough_without_bar_spacing() -> None:
    strat = _vt_sleeve()  # no observed times -> spacing unknown -> pass-through
    sigmas = dict.fromkeys(_VT_SYMS, 0.05)
    _weights, scalar = strat._inverse_vol_weights(_vt_targets(), sigmas)
    assert scalar == 1.0


def test_vol_target_calm_leaves_scalar_unthrottled() -> None:
    strat = _vt_sleeve()
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    calm = dict.fromkeys(_VT_SYMS, 0.0005)  # annualized ~= 0.047 < 0.20
    _weights, scalar = strat._inverse_vol_weights(_vt_targets(), calm)
    assert scalar == 1.0


def test_vol_target_scalar_deterministic() -> None:
    strat = _vt_sleeve()
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    sigmas = dict.fromkeys(_VT_SYMS, 0.05)
    _w1, s1 = strat._inverse_vol_weights(_vt_targets(), sigmas)
    _w2, s2 = strat._inverse_vol_weights(_vt_targets(), sigmas)
    assert s1 == s2


def test_vol_target_epochs_tracked_from_datetime_feed() -> None:
    syms = ["A", "B", "C", "D", "E"]  # >= min_symbols (5) so _rebalance runs
    strat = DownsideTailRiskPremiumStrategy(_Bars(syms), _Queue())
    for idx in range(6):
        epoch = 1_700_000_000.0 + idx * 3600.0  # numeric epoch -> parsed to a datetime
        rows = {sym: {"close": 100.0 + idx, "volume": 1000.0} for sym in syms}
        strat.calculate_signals(
            SimpleNamespace(
                type="MARKET_WINDOW",
                time=epoch,
                bars_1s={sym: [dict(row, time=epoch)] for sym, row in rows.items()},
            )
        )
    times = list(strat._recent_times)
    assert len(times) >= 5, times
    gaps = [round(times[i + 1] - times[i]) for i in range(len(times) - 1)]
    assert gaps and all(gap == 3600 for gap in gaps), gaps


def test_vol_target_recent_times_survive_state_roundtrip() -> None:
    strat = _vt_sleeve()
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    restored = _vt_sleeve()
    restored.set_state(strat.get_state())
    assert list(restored._recent_times) == list(_HOUR_EPOCHS)
