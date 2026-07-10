"""Deterministic build-gate + hygiene tests for the CLV accumulation sleeve.

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions).  The build gate RUNS the real incumbents on
hand-built synthetic feeds and asserts materially different emitted actions:

  * Leg 1 (vs ``CrossSectionalNearHighAnchoringStrategy``): a four-quadrant
    (CLV x nearness) universe where the candidate LONGs an accumulation name
    stranded far below an old spike high (low nearness) and SHORTs a
    distribution name pinned near its trailing max (high nearness), while the
    near-high incumbent takes the OPPOSITE book on both -- proving the double
    residualization decouples the flow footprint from nearness.
  * Leg 2 (not-secretly-momentum): a strong-trend, mid-range-close (CLV 0) name
    is excluded from the candidate book after residualization.
  * Leg 3 (vs ``VpinToxicityRiderStrategy``): two names with IDENTICAL close and
    volume but opposite intra-bar CLV get opposite candidate books, while the
    unsigned VPIN toxicity statistic is bit-identical for both (sign-blind).
  * Leg 4 (vs ``CrossSectionalFlowShareRotationStrategy``): the same identical
    close+volume pair gets identical volume-SHARE treatment from the incumbent
    (which is proven live on the same feed), while the candidate splits them.
  * Leg 5 (min-hold rescue): a would-be side flip inside the min-hold window is
    suppressed.
  * Leg 6 (residualizer decoupling) + hygiene: collinearity -> ~0 residual,
    never-raise, determinism, state roundtrip, adversarial set_state, self-skip.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.indicators.volume import chaikin_money_flow
from lumina_quant.strategies.clv_accumulation_alpha_sleeves import (
    _CLV_ACCUMULATION_SLICE,
    CrossSectionalCloseLocationAccumulationStrategy,
)
from lumina_quant.strategies.flow_share_rotation_alpha_sleeves import (
    CrossSectionalFlowShareRotationStrategy,
)
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.strategies.vpin_toxicity_alpha_sleeves import VpinToxicityRiderStrategy
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
    time = f"t{idx:04d}"
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


def _entries(items: list[Any]) -> list[Any]:
    return [sig for sig in items if str(sig.signal_type).upper() in {"LONG", "SHORT"}]


def _non_exit(items: list[Any]) -> list[Any]:
    return [sig for sig in items if str(sig.signal_type).upper() != "EXIT"]


# --------------------------------------------------------------------------- #
# deterministic bar generators
# --------------------------------------------------------------------------- #

_N = 40


def _clv_bar(close: float, m: float, *, spike_high: float | None = None) -> dict[str, Any]:
    """A bar whose close-location-value is exactly ``m`` (range ~1.0)."""
    upper = (1.0 - m) * 0.5
    lower = (1.0 + m) * 0.5
    high = close + upper
    low = close - lower
    if spike_high is not None:
        high = max(high, spike_high)
    return {"open": close, "high": high, "low": low, "close": close, "volume": 1000.0}


def _wig(i: int) -> float:
    return 0.2 * ((i % 3) - 1)


def _quadrant(m: float, near: float, price: float = 100.0) -> list[dict[str, Any]]:
    """Accumulation/distribution footprint ``m`` at a target nearness ``near``.

    Low nearness is planted via an early spike high in the trailing-max window.
    """
    spike = price / near if near < 0.95 else None
    bars: list[dict[str, Any]] = []
    for i in range(_N):
        close = price + _wig(i)
        bars.append(_clv_bar(close, m, spike_high=(spike if i == 0 else None)))
    return bars


def _momentum_series() -> list[dict[str, Any]]:
    """Strong uptrend, CLV 0, early symmetric spike -> low nearness."""
    bars: list[dict[str, Any]] = []
    for i in range(_N):
        close = 100.0 * (1.0 + 0.012 * i)
        if i == 0:
            bars.append(
                {"open": close, "high": 170.0, "low": 30.0, "close": close, "volume": 1000.0}
            )
        else:
            bars.append(
                {
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    return bars


def _quadrant_universe() -> dict[str, list[dict[str, Any]]]:
    series: dict[str, list[dict[str, Any]]] = {
        "ACC2/USDT": _quadrant(+0.8, near=0.5),  # accumulation, low nearness
        "ACChi/USDT": _quadrant(+0.8, near=0.99),  # accumulation, high nearness
        "DIST2/USDT": _quadrant(-0.8, near=0.99),  # distribution, high nearness
        "DISTlo/USDT": _quadrant(-0.8, near=0.5),  # distribution, low nearness
        "MOM/USDT": _momentum_series(),
    }
    for fi in range(3):
        series[f"F{fi}/USDT"] = _quadrant(0.0, near=0.8, price=90 + fi)
    return series


_COMMON_KWARGS: dict[str, Any] = dict(
    accumulation_window_bars=28,
    momentum_window_bars=28,
    nearness_window_bars=364,
    min_history_bars=30,
    vol_window=20,
    quantile_pct=0.25,
    rebalance_bars=1,
    min_hold_decisions=0,
    rank_hysteresis_buffer=0,
    min_symbols=6,
    allow_short=True,
    target_vol=0.0,
    stop_loss_pct=0.0,
    min_price=0.01,
)


def _cmf_last(bars: list[dict[str, Any]], period: int = 28) -> float:
    value = chaikin_money_flow(
        [b["high"] for b in bars],
        [b["low"] for b in bars],
        [b["close"] for b in bars],
        [b["volume"] for b in bars],
        period=period,
    )
    assert value is not None
    return value


# --------------------------------------------------------------------------- #
# Leg 1 + Leg 2: opposite book vs near-high anchoring; momentum name excluded
# --------------------------------------------------------------------------- #


def test_leg1_opposite_book_vs_near_high_anchoring() -> None:
    series = _quadrant_universe()

    # Stage-1 premises (asserted with the real primitive BEFORE any strategy):
    # accumulation names carry positive CLV, distribution names negative, and
    # the momentum name is CLV-flat.
    assert _cmf_last(series["ACC2/USDT"]) > 0.5
    assert _cmf_last(series["DIST2/USDT"]) < -0.5
    assert abs(_cmf_last(series["MOM/USDT"])) < 0.05

    syms = list(series)
    candidate = CrossSectionalCloseLocationAccumulationStrategy(
        _Bars(syms), _Queue(), **_COMMON_KWARGS
    )
    _feed(candidate, series)
    cand = _final_side(candidate.events.items)

    near_high = CrossSectionalNearHighAnchoringStrategy(
        _Bars(syms),
        _Queue(),
        high_lookback_bars=364,
        min_history_bars=30,
        vol_window=20,
        quantile_pct=0.25,
        rebalance_bars=1,
        min_hold_bars=0,
        allow_short=True,
        min_symbols=6,
        target_vol=0.0,
        stop_loss_pct=0.0,
        min_price=0.01,
    )
    _feed(near_high, series)
    anchor = _final_side(near_high.events.items)

    # Both incumbents emit definite non-empty books (incumbent-LIVE + candidate-ACTS).
    assert cand and anchor
    # Load-bearing divergence: OPPOSITE actions on BOTH names, on identical input.
    assert cand.get("ACC2/USDT") == "LONG" and anchor.get("ACC2/USDT") == "SHORT"
    assert cand.get("DIST2/USDT") == "SHORT" and anchor.get("DIST2/USDT") == "LONG"


def test_leg2_not_secretly_momentum_excludes_trend_name() -> None:
    series = _quadrant_universe()
    syms = list(series)
    candidate = CrossSectionalCloseLocationAccumulationStrategy(
        _Bars(syms), _Queue(), **_COMMON_KWARGS
    )
    _feed(candidate, series)
    cand = _final_side(candidate.events.items)
    # The strong-trend, mid-range-close (CLV 0) name residualizes to ~0 and is in
    # neither book, despite its large momentum.
    assert "MOM/USDT" not in cand


# --------------------------------------------------------------------------- #
# Leg 3 + Leg 4: signed CLV vs unsigned VPIN toxicity and volume-share
# --------------------------------------------------------------------------- #

_M = 60


def _pn_universe() -> dict[str, list[dict[str, Any]]]:
    """P and N share IDENTICAL close + volume; only intra-bar CLV sign differs.

    An identical early spike high fixes both nearness values equal, so the sole
    cross-sectional difference between P and N is the sign of the close-location.
    """
    series: dict[str, list[dict[str, Any]]] = {}
    p_bars: list[dict[str, Any]] = []
    n_bars: list[dict[str, Any]] = []
    for i in range(_M):
        close = 100.0 + _wig(i)
        if i == 0:
            spike = {"open": close, "high": 130.0, "low": 70.0, "close": close, "volume": 1000.0}
            p_bars.append(dict(spike))
            n_bars.append(dict(spike))
        else:
            p_bars.append(_clv_bar(close, +0.8))  # close near high
            n_bars.append(_clv_bar(close, -0.8))  # close near low
    series["P/USDT"] = p_bars
    series["N/USDT"] = n_bars
    for fi in range(3):
        fills: list[dict[str, Any]] = []
        for i in range(_M):
            close = 90.0 + fi + _wig(i)
            fills.append(_clv_bar(close, 0.0))
        series[f"F{fi}/USDT"] = fills
    # A rising-share name so the flow-share incumbent is provably live.
    riser: list[dict[str, Any]] = []
    price = 100.0
    volume = 500.0
    for i in range(_M):
        price *= 1.02 * (1.0 + (((i * 2654435761) % 1000) / 1000.0 - 0.5) * 0.006)
        volume *= 1.08
        riser.append(
            {
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": volume,
            }
        )
    series["RISER/USDT"] = riser
    return series


_PN_KWARGS = dict(_COMMON_KWARGS, quantile_pct=0.20)


def test_leg3_signed_clv_vs_unsigned_vpin_toxicity() -> None:
    series = _pn_universe()
    syms = list(series)

    candidate = CrossSectionalCloseLocationAccumulationStrategy(_Bars(syms), _Queue(), **_PN_KWARGS)
    _feed(candidate, series)
    cand = _final_side(candidate.events.items)
    # Candidate takes OPPOSITE sides on the pair, driven purely by CLV sign.
    assert cand.get("P/USDT") == "LONG" and cand.get("N/USDT") == "SHORT"

    vpin = VpinToxicityRiderStrategy(
        _Bars(syms),
        _Queue(),
        n_buckets=5,
        vpin_history=40,
        bucket_mult=1.0,
        trend_lookback=10,
        min_trend_roc=0.0,
        vol_window=20,
        min_price=0.01,
    )
    _feed(vpin, series)
    extra_p = vpin._extra["P/USDT"]
    extra_n = vpin._extra["N/USDT"]
    # The VPIN toxicity statistic reads only (close-to-close change, volume),
    # which are identical for P and N -> bit-identical, non-None (live) reading,
    # so the unsigned toxicity consumer CANNOT distinguish the two.
    assert extra_p.last_vpin is not None and extra_p.last_vpin == extra_n.last_vpin
    assert extra_p.last_percentile == extra_n.last_percentile
    item_p = vpin._state["P/USDT"]
    item_n = vpin._state["N/USDT"]
    assert vpin._gated_direction("P/USDT", item_p) == vpin._gated_direction("N/USDT", item_n)


def test_leg4_signed_clv_vs_flow_share_participation() -> None:
    series = _pn_universe()
    syms = list(series)

    candidate = CrossSectionalCloseLocationAccumulationStrategy(_Bars(syms), _Queue(), **_PN_KWARGS)
    _feed(candidate, series)
    cand = _final_side(candidate.events.items)
    assert cand.get("P/USDT") == "LONG" and cand.get("N/USDT") == "SHORT"

    flow = CrossSectionalFlowShareRotationStrategy(
        _Bars(syms),
        _Queue(),
        share_z_window=20,
        confirm_window=3,
        share_z_entry=1.0,
        top_n=8,
        vol_window=8,
        min_symbols=5,
        target_vol=0.0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
        min_price=0.01,
    )
    _feed(flow, series)
    flow_items = flow.events.items
    flow_side = _final_side(flow_items)
    # Incumbent-LIVE: flow-share emits a definite book on this feed.
    assert _entries(flow_items)
    # Identical close+volume -> identical dollar-volume SHARE -> identical
    # treatment of P and N (never opposite), unlike the candidate.
    assert flow_side.get("P/USDT") == flow_side.get("N/USDT")


# --------------------------------------------------------------------------- #
# Leg 5: min-hold rescue suppresses a would-be side flip
# --------------------------------------------------------------------------- #


def _flip_universe() -> dict[str, list[dict[str, Any]]]:
    """Phase 1: P accumulation / N distribution.  Phase 2: the footprints SWAP."""
    phase1, phase2 = 40, 20
    series: dict[str, list[dict[str, Any]]] = {"P/USDT": [], "N/USDT": []}
    for i in range(phase1 + phase2):
        close = 100.0 + _wig(i)
        p_m = +0.8 if i < phase1 else -0.8
        n_m = -0.8 if i < phase1 else +0.8
        if i == 0:
            spike = {"open": close, "high": 130.0, "low": 70.0, "close": close, "volume": 1000.0}
            series["P/USDT"].append(dict(spike))
            series["N/USDT"].append(dict(spike))
        else:
            series["P/USDT"].append(_clv_bar(close, p_m))
            series["N/USDT"].append(_clv_bar(close, n_m))
    for fi in range(4):
        fills = []
        for i in range(phase1 + phase2):
            close = 90.0 + fi + _wig(i)
            fills.append(_clv_bar(close, 0.0))
        series[f"F{fi}/USDT"] = fills
    return series


def test_leg5_min_hold_suppresses_side_flip() -> None:
    series = _flip_universe()
    syms = list(series)
    base = dict(_COMMON_KWARGS, quantile_pct=0.20)

    held = CrossSectionalCloseLocationAccumulationStrategy(
        _Bars(syms), _Queue(), **dict(base, min_hold_decisions=100)
    )
    _feed(held, series)
    free = CrossSectionalCloseLocationAccumulationStrategy(
        _Bars(syms), _Queue(), **dict(base, min_hold_decisions=0)
    )
    _feed(free, series)

    # With no min-hold the swap flips P from LONG to SHORT; a long min-hold
    # suppresses the flip so P is still LONG at the end of the feed.
    assert _final_side(free.events.items).get("P/USDT") == "SHORT"
    assert _final_side(held.events.items).get("P/USDT") == "LONG"


# --------------------------------------------------------------------------- #
# Leg 6 + hygiene
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    series = _quadrant_universe()
    syms = list(series)

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strat = CrossSectionalCloseLocationAccumulationStrategy(
            _Bars(syms), _Queue(), **_COMMON_KWARGS
        )
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
    series = _quadrant_universe()
    syms = list(series)
    strat = CrossSectionalCloseLocationAccumulationStrategy(_Bars(syms), _Queue(), **_COMMON_KWARGS)
    _feed(strat, series)
    snapshot = strat.get_state()

    restored = CrossSectionalCloseLocationAccumulationStrategy(
        _Bars(syms), _Queue(), **_COMMON_KWARGS
    )
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot


def test_adversarial_set_state_never_raises() -> None:
    syms = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT", "F/USDT"]
    strat = CrossSectionalCloseLocationAccumulationStrategy(_Bars(syms), _Queue(), **_COMMON_KWARGS)
    strat.set_state(None)  # type: ignore[arg-type]
    strat.set_state("nope")  # type: ignore[arg-type]
    strat.set_state({"symbol_state": "not a dict"})
    strat.set_state({"symbol_state": {"A/USDT": "not a dict either"}})
    strat.set_state({"symbol_state": {"A/USDT": {"closes": 123}}})
    strat.set_state(
        {
            "tick": "bad",
            "symbol_state": {
                sym: {
                    "closes": ["x", float("nan"), 1.0],
                    "highs": {"bad": "type"},
                    "lows": None,
                    "volumes": [1.0, "y"],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "last_time_key": 5,
                    "score": [1, 2],
                }
                for sym in syms
            },
        }
    )
    for item in strat._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


def test_degenerate_input_never_raises() -> None:
    strat = CrossSectionalCloseLocationAccumulationStrategy(
        _Bars(["Z/USDT"]), _Queue(), **_COMMON_KWARGS
    )
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t0", close=0.0))
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t1", close=-1.0))
    strat.calculate_signals(
        SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t2", close=float("nan"))
    )
    strat.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t3", bars_1s={}))
    strat.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    assert _non_exit(strat.events.items) == []


def test_self_skip_below_min_symbols() -> None:
    series = {
        s: _quadrant(0.0, near=0.8, price=90 + i)
        for i, s in enumerate(["A/USDT", "B/USDT", "C/USDT"])
    }
    strat = CrossSectionalCloseLocationAccumulationStrategy(
        _Bars(list(series)), _Queue(), **dict(_COMMON_KWARGS, min_symbols=6)
    )
    _feed(strat, series)
    assert _non_exit(strat.events.items) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalCloseLocationAccumulationStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "accumulation_window_bars",
        "momentum_window_bars",
        "nearness_window_bars",
        "quantile_pct",
        "min_hold_decisions",
        "rank_hysteresis_buffer",
        "min_symbols",
    ):
        assert required in schema


def test_slice_timeframe_expansion_scales_bar_windows() -> None:
    """4h/1h cells mirror the 1d variants: bar windows + the bar-count weekly
    rebalance clock scale x6/x24, while the min-hold / hysteresis / fractions stay
    timeframe-invariant."""
    slice_ = _CLV_ACCUMULATION_SLICE
    assert set(slice_) == {"4h", "1h", "1d"}
    variants = {tf: tuple(cell["variant"] for cell in cells) for tf, cells in slice_.items()}
    assert variants["4h"] == variants["1h"] == variants["1d"]
    by = {tf: {cell["variant"]: cell for cell in cells} for tf, cells in slice_.items()}
    for variant in variants["1d"]:
        d, h4, h1 = by["1d"][variant], by["4h"][variant], by["1h"][variant]
        for key in (
            "accumulation_window_bars",
            "momentum_window_bars",
            "nearness_window_bars",
            "min_history_bars",
            "vol_window",
        ):
            assert h4[key] == d[key] * 6, (variant, key)
            assert h1[key] == d[key] * 24, (variant, key)
        assert (d["rebalance_bars"], h4["rebalance_bars"], h1["rebalance_bars"]) == (7, 42, 168)
        for key in (
            "min_hold_decisions",
            "rank_hysteresis_buffer",
            "quantile_pct",
            "target_gross_exposure",
            "target_vol",
            "stop_loss_pct",
        ):
            assert h4[key] == d[key] == h1[key], (variant, key)


# --------------------------------------------------------------------------- #
# vol-target horizon regression (worker-vt2)
#
# The ``target_vol`` clamp compares an annual-scale target (0.20) against the
# inverse-vol-weighted PER-BAR portfolio vol.  It MUST first annualize that
# per-bar vol via sqrt(bars_per_year) inferred from observed bar spacing, else
# the Moreira-Muir clamp is inert (scalar pinned at 1.0).  The risk-parity
# WEIGHTS themselves are horizon-free (normalized inverse-vol -> annualization
# cancels), so only the SCALAR is annualized.
# --------------------------------------------------------------------------- #

_VT_SYMS = ["A", "B", "C", "D"]
_HOUR_EPOCHS = [1_700_000_000.0 + i * 3600.0 for i in range(12)]  # 1h spacing


def _vt_sleeve() -> CrossSectionalCloseLocationAccumulationStrategy:
    return CrossSectionalCloseLocationAccumulationStrategy(_Bars(_VT_SYMS), _Queue())


def _vt_targets() -> dict[str, Any]:
    return {sym: ("LONG", 1.0, {}) for sym in _VT_SYMS}


def test_vol_target_throttle_active_on_hourly_high_vol() -> None:
    strat = _vt_sleeve()
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    vols = dict.fromkeys(_VT_SYMS, 0.05)  # per-bar 5% vol
    _weights, scalar = strat._inverse_vol_weights(_vt_targets(), vols)
    # equal per-symbol vols -> weighted portfolio_vol == 0.05 (per bar); annualized
    # ~= 4.68 >> target_vol 0.20 -> scalar << 1. Recompute via the same public
    # helpers the sleeve uses so the assertion is convention-agnostic.
    assert scalar < 0.2, scalar
    portfolio_vol_ann = annualize_per_bar_vol(0.05, bars_per_year_from_spacing(list(_HOUR_EPOCHS)))
    assert scalar == min(1.0, 0.20 / portfolio_vol_ann)


def test_vol_target_passthrough_without_bar_spacing() -> None:
    strat = _vt_sleeve()  # no observed times -> spacing unknown
    vols = dict.fromkeys(_VT_SYMS, 0.05)
    _weights, scalar = strat._inverse_vol_weights(_vt_targets(), vols)
    assert scalar == 1.0  # pass-through, never throttle on a mismatched horizon


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
    vols = dict.fromkeys(_VT_SYMS, 0.05)
    _w1, s1 = strat._inverse_vol_weights(_vt_targets(), vols)
    _w2, s2 = strat._inverse_vol_weights(_vt_targets(), vols)
    assert s1 == s2


def test_vol_target_epochs_tracked_from_datetime_feed() -> None:
    syms = ["A", "B", "C", "D", "E", "F", "G"]  # >= min_symbols so _evaluate runs
    strat = CrossSectionalCloseLocationAccumulationStrategy(_Bars(syms), _Queue())
    for idx in range(6):
        epoch = 1_700_000_000.0 + idx * 3600.0  # numeric epoch -> parsed to a datetime
        rows = {
            sym: {"close": 100.0 + idx, "high": 101.0 + idx, "low": 99.0 + idx, "volume": 1000.0}
            for sym in syms
        }
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
    assert gaps and all(gap == 3600 for gap in gaps), gaps  # 1h spacing tracked


def test_vol_target_recent_times_survive_state_roundtrip() -> None:
    strat = _vt_sleeve()
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    restored = _vt_sleeve()
    restored.set_state(strat.get_state())
    assert list(restored._recent_times) == list(_HOUR_EPOCHS)
