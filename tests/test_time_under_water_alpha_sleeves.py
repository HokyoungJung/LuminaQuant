"""Deterministic tests for CrossSectionalTimeUnderWaterStrategy (W3-3, conditional).

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the W3 integration wave).

The load-bearing tests are the BUILD GATE legs.  On a single synthetic
cross-section a STALE drawdown (peak 350 bars ago) and a FRESH drawdown (peak 21
bars ago) share IDENTICAL depth (-30%), so:
  - the near-high incumbent scores them identically (nearness 0.70 both),
  - the long-run overreaction incumbent ties them at formation return 0 (its
    21-bar skip band swallows the entire fresh crash),
  - the capital-gains-overhang incumbent leaves STALE mid-rank (overhang ~0),
yet the duration transform SPLITS them (long the stale stagnation, short the
fresh drawdown).  Leg B proves volume-invariance vs the CGO incumbent, Leg C the
underwater eligibility structure.  Failure of Leg A, B, or C means the leaf is
behaviorally spanned and the lane DROPS -- the gate is never weakened.

No ``random`` module is used: fixture paths are deterministic piecewise lines
(with a tiny parity sawtooth only where realised vol must stay non-degenerate).
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize
from lumina_quant.indicators.reference_price import (
    capital_gains_overhang,
    grinblatt_han_reference_price,
)
from lumina_quant.strategies.cgo_disposition_alpha_sleeves import (
    CrossSectionalCapitalGainsOverhangStrategy,
)
from lumina_quant.strategies.longrun_overreaction_alpha_sleeves import (
    LongRunOverreactionReversalStrategy,
)
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.strategies.time_under_water_alpha_sleeves import (
    CrossSectionalTimeUnderWaterStrategy,
    _time_under_water,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# harness (daily ISO-date bars; MARKET_WINDOW feed keeps the cross-section
# coherent per bar)
# --------------------------------------------------------------------------- #


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _ts(idx: int) -> str:
    return (datetime.date(2024, 1, 1) + datetime.timedelta(days=idx)).isoformat()


def _lin(p0: float, p1: float, k: int) -> list[float]:
    return [p0 + (p1 - p0) * i / (k - 1) for i in range(k)] if k > 1 else [p1]


Series = dict[str, list[float]]
Vols = dict[str, list[float]]


def _feed(
    strategy: Any, symbols: list[str], series: Series, n: int, vols: Vols | None = None
) -> None:
    for idx in range(n):
        bars_1s: dict[str, list[dict[str, Any]]] = {}
        for symbol in symbols:
            closes = series[symbol]
            if idx >= len(closes):
                continue
            close = closes[idx]
            volume = vols[symbol][idx] if vols and symbol in vols else 1000.0
            bars_1s[symbol] = [
                {
                    "time": _ts(idx),
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": volume,
                }
            ]
        strategy.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=_ts(idx), bars_1s=bars_1s)
        )


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def _entries(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if sig.signal_type in {"LONG", "SHORT"}]


# --------------------------------------------------------------------------- #
# BUILD-GATE cross-section (Leg A + Leg C): STALE vs FRESH share depth -30%.
#
#   STALE : 100 (0-50), decline to 70 (51-80), 70 thereafter -> TUW=350.
#   FRESH : 100 (0-379), crash to 70 (380-400) -> TUW=21, entire crash inside the
#           long-run incumbent's 21-bar skip band.
#   EXWIN : +200% winner (at its high, depth ~0) -> the long-run incumbent's short.
#   EXLOSE: -65% extreme loser -> the long-run incumbent's long.
#   HIGH1/2: pinned to a NEW high (TUW=0, depth 0) -> near-high top longs, and
#           structurally OUTSIDE the candidate's underwater universe.
#   LOW1  : nearness 0.35 -> near-high short.
#   UW1..4: neutral underwater fillers spanning intermediate TUW/depth.
# --------------------------------------------------------------------------- #

_N = 401


def _saw(idx: int) -> float:
    return 0.8 if idx % 2 else -0.8


def _build_gate_series() -> Series:
    stale = [100.0] * 51 + _lin(100.0, 70.0, 31)[1:] + [70.0] * (381 - 81)
    stale += [70.0 + _saw(i) for i in range(381, 400)] + [70.0]
    fresh = [100.0] * 380 + _lin(100.0, 70.0, 22)[1:]
    exwin = _lin(100.0, 300.0, 381) + [300.0 + _saw(i) for i in range(381, 401)]
    exlose = [100.0] * 201 + _lin(100.0, 35.0, 201)[1:]
    high1 = _lin(100.0, 150.0, _N)
    high2 = _lin(100.0, 140.0, _N)
    low1 = [100.0] * 121 + _lin(100.0, 35.0, 281)[1:]

    def uw(peak_bar: int, final: float) -> list[float]:
        return ([100.0] * (peak_bar + 1) + _lin(100.0, final, _N - peak_bar)[1:])[:_N]

    series: Series = {
        "STALE": stale[:_N],
        "FRESH": fresh[:_N],
        "EXWIN": exwin[:_N],
        "EXLOSE": exlose[:_N],
        "HIGH1": high1[:_N],
        "HIGH2": high2[:_N],
        "LOW1": low1[:_N],
        "UW1": uw(150, 78.0),
        "UW2": uw(220, 74.0),
        "UW3": uw(280, 82.0),
        "UW4": uw(320, 68.0),
    }
    for symbol, closes in series.items():
        assert len(closes) == _N, (symbol, len(closes))
    return series


_GATE_SYMBOLS = [
    "STALE",
    "FRESH",
    "EXWIN",
    "EXLOSE",
    "HIGH1",
    "HIGH2",
    "LOW1",
    "UW1",
    "UW2",
    "UW3",
    "UW4",
]

_CAND_KWARGS: dict[str, Any] = dict(
    lookback_bars=_N,
    min_history_bars=60,
    depth_enter=-0.12,
    depth_exit=-0.08,
    depth_floor=-0.85,
    score_mode="duration",
    vol_window=20,
    quantile_pct=0.25,
    rebalance_bars=1,
    min_hold_decisions=0,
    cooldown_decisions=0,
    rank_hysteresis_buffer=0,
    residualize=True,
    allow_short=True,
    min_symbols=5,
    target_gross_exposure=1.0,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)

_NEARHIGH_KWARGS: dict[str, Any] = dict(
    high_lookback_bars=364,
    min_history_bars=60,
    vol_window=20,
    quantile_pct=0.25,
    rebalance_bars=1,
    min_hold_bars=0,
    allow_short=True,
    min_symbols=5,
    target_gross_exposure=1.0,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)

_LRO_KWARGS: dict[str, Any] = dict(
    formation_bars=126,
    skip_bars=21,
    z_min=1.0,
    max_universe=12,
    rebalance_bars=1,
    min_hold_bars=0,
    quantile_pct=0.25,
    min_symbols=5,
    allow_short=True,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)

_CGO_KWARGS: dict[str, Any] = dict(
    window_bars=56,
    skip_recent=1,
    vol_window=30,
    quantile_pct=0.25,
    rebalance_bars=1,
    min_hold_bars=0,
    min_history_bars=70,
    allow_short=True,
    min_symbols=5,
    target_gross_exposure=1.0,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)


# --------------------------------------------------------------------------- #
# Stage-1: the fixture premise asserted with the real primitives.
# --------------------------------------------------------------------------- #


def test_leg0_stage1_fixture_primitives() -> None:
    series = _build_gate_series()
    tuw_stale, depth_stale, _ = _time_under_water(series["STALE"])
    tuw_fresh, depth_fresh, _ = _time_under_water(series["FRESH"])
    assert tuw_stale == 350
    assert tuw_fresh == 21
    assert abs(depth_stale - depth_fresh) < 1e-9
    assert abs(depth_stale - (-0.30)) < 1e-9

    # Near-high nearness (over the incumbent's 364-bar window) is IDENTICAL.
    near_stale = series["STALE"][-1] / max(series["STALE"][37:])
    near_fresh = series["FRESH"][-1] / max(series["FRESH"][37:])
    assert abs(near_stale - near_fresh) < 1e-9
    assert abs(near_stale - 0.70) < 1e-9

    # HIGH1 is at a new high: TUW 0, depth 0 -> outside the underwater universe.
    tuw_high, depth_high, _ = _time_under_water(series["HIGH1"])
    assert tuw_high == 0
    assert abs(depth_high) < 1e-9


# --------------------------------------------------------------------------- #
# LEG A: the duration transform SPLITS the depth-tied pair that all three
# incumbents cannot separate.
# --------------------------------------------------------------------------- #


def test_legA_candidate_splits_stale_and_fresh() -> None:
    series = _build_gate_series()
    candidate = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    _feed(candidate, _GATE_SYMBOLS, series, _N)
    side = _final_side(candidate.events.items)
    assert side.get("STALE") == "LONG", side
    assert side.get("FRESH") == "SHORT", side
    assert any(v == "LONG" for v in side.values())
    assert any(v == "SHORT" for v in side.values())


def test_legA_diverges_from_near_high_anchoring() -> None:
    series = _build_gate_series()
    incumbent = CrossSectionalNearHighAnchoringStrategy(
        _Bars(_GATE_SYMBOLS), _Queue(), **_NEARHIGH_KWARGS
    )
    _feed(incumbent, _GATE_SYMBOLS, series, _N)
    inc_side = _final_side(incumbent.events.items)
    # Live incumbent, but STALE and FRESH (nearness 0.70 each) are mid-rank in
    # NEITHER extreme -- the near-high axis cannot see the duration split.
    assert inc_side, "near-high incumbent must emit a non-empty book"
    assert "STALE" not in inc_side
    assert "FRESH" not in inc_side


def test_legA_diverges_from_longrun_overreaction() -> None:
    series = _build_gate_series()
    lro = LongRunOverreactionReversalStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_LRO_KWARGS)
    _feed(lro, _GATE_SYMBOLS, series, _N)
    lro_side = _final_side(lro.events.items)

    # The incumbent's OWN formation primitive returns EQUAL (0) for the pair --
    # the WHY of its blindness (the 21-bar skip swallows the fresh crash).
    form_stale = lro._formation("STALE")
    form_fresh = lro._formation("FRESH")
    assert form_stale is not None and form_fresh is not None
    assert abs(form_stale[0]) < 1e-12
    assert abs(form_fresh[0]) < 1e-12

    # Live incumbent; both are z-gate excluded from its book; the candidate splits.
    assert lro_side, "LRO must emit a non-empty book"
    assert "STALE" not in lro_side
    assert "FRESH" not in lro_side


def test_legA_diverges_from_capital_gains_overhang() -> None:
    series = _build_gate_series()
    cgo = CrossSectionalCapitalGainsOverhangStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CGO_KWARGS)
    _feed(cgo, _GATE_SYMBOLS, series, _N)
    cgo_side = _final_side(cgo.events.items)

    # CGO's own statistic leaves STALE mid-rank (overhang ~0: its recent holder
    # cohort bought AT 70), so the disposition axis does not long it.
    ref = grinblatt_han_reference_price(series["STALE"], [1000.0] * _N, 56, skip_recent=1)
    overhang = capital_gains_overhang(series["STALE"][-1], ref)
    assert overhang is not None and abs(overhang) < 1e-2
    assert cgo_side, "CGO must emit a non-empty book"
    assert cgo_side.get("STALE") != "LONG"

    candidate = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    _feed(candidate, _GATE_SYMBOLS, series, _N)
    assert _final_side(candidate.events.items).get("STALE") == "LONG"


# --------------------------------------------------------------------------- #
# LEG C: the underwater eligibility structure -- an at-high name is structurally
# OUTSIDE the candidate universe while it is the near-high incumbent's top long.
# --------------------------------------------------------------------------- #


def test_legC_at_high_name_outside_candidate_universe() -> None:
    series = _build_gate_series()
    candidate = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    _feed(candidate, _GATE_SYMBOLS, series, _N)
    duration_frac, _depth, _rec, _vols, _metas = candidate._eligible_features()
    assert "HIGH1" not in duration_frac
    assert "HIGH2" not in duration_frac
    assert "EXWIN" not in duration_frac
    # It never appears in the candidate's book on either side.
    assert all(sig.symbol != "HIGH1" for sig in _entries(candidate.events.items))

    incumbent = CrossSectionalNearHighAnchoringStrategy(
        _Bars(_GATE_SYMBOLS), _Queue(), **_NEARHIGH_KWARGS
    )
    _feed(incumbent, _GATE_SYMBOLS, series, _N)
    inc_side = _final_side(incumbent.events.items)
    assert inc_side.get("HIGH1") == "LONG"


# --------------------------------------------------------------------------- #
# LEG B: volume-invariance vs CGO -- an identical-price / different-volume pair
# is TIED by the candidate (volume-blind) and SEPARATED by the disposition
# incumbent (order-sensitive Grinblatt-Han reference price).
# --------------------------------------------------------------------------- #


def _build_legb() -> tuple[list[str], Series, Vols]:
    n = 221
    path = [100.0] * 31 + [100.0 - 24.0 * (i + 1) / 190.0 for i in range(190)]
    path = path[:n]
    vol_early = [200.0] * n
    for i in range(165, 176):
        vol_early[i] = 6000.0  # heavy volume at the HIGHER price
    vol_late = [200.0] * n
    for i in range(205, 216):
        vol_late[i] = 6000.0  # heavy volume at the LOWER price

    def mk(peak: int, final: float) -> list[float]:
        return ([100.0] * (peak + 1) + _lin(100.0, final, n - peak)[1:])[:n]

    symbols = ["VOLEARLY", "VOLLATE", "DEEP1", "DEEP2", "SHAL1", "SHAL2"]
    series: Series = {
        "VOLEARLY": path,
        "VOLLATE": list(path),
        "DEEP1": mk(20, 45.0),
        "DEEP2": mk(40, 50.0),
        "SHAL1": mk(120, 88.0),
        "SHAL2": mk(150, 90.0),
    }
    vols: Vols = {"VOLEARLY": vol_early, "VOLLATE": vol_late}
    return symbols, series, vols


def test_legB_volume_invariance_vs_cgo() -> None:
    symbols, series, vols = _build_legb()
    n = len(series["VOLEARLY"])

    # The candidate is volume-blind: identical closes -> identical duration/depth.
    candidate = CrossSectionalTimeUnderWaterStrategy(
        _Bars(symbols),
        _Queue(),
        **dict(_CAND_KWARGS, lookback_bars=n, depth_floor=-0.99),
    )
    _feed(candidate, symbols, series, n, vols=vols)
    duration_frac, depth_by, _rec, _vols, _metas = candidate._eligible_features()
    assert "VOLEARLY" in duration_frac and "VOLLATE" in duration_frac
    assert abs(duration_frac["VOLEARLY"] - duration_frac["VOLLATE"]) < 1e-12
    assert abs(depth_by["VOLEARLY"] - depth_by["VOLLATE"]) < 1e-12

    # The Grinblatt-Han reference price (and hence CGO's overhang) DIFFERS.
    ref_early = grinblatt_han_reference_price(
        series["VOLEARLY"], vols["VOLEARLY"], 56, skip_recent=1
    )
    ref_late = grinblatt_han_reference_price(series["VOLLATE"], vols["VOLLATE"], 56, skip_recent=1)
    assert ref_early is not None and ref_late is not None
    assert abs(ref_early - ref_late) > 1.0

    # The real CGO class rank-separates the pair the candidate tied.
    cgo = CrossSectionalCapitalGainsOverhangStrategy(
        _Bars(symbols), _Queue(), **dict(_CGO_KWARGS, vol_window=20)
    )
    _feed(cgo, symbols, series, n, vols=vols)
    cgo_side = _final_side(cgo.events.items)
    assert cgo_side.get("VOLLATE") == "LONG"
    assert cgo_side.get("VOLEARLY") != "LONG"


# --------------------------------------------------------------------------- #
# LEG D: the depth residualizer decoupler.
# --------------------------------------------------------------------------- #


def test_legD_residualizer_collinear_duration_vanishes() -> None:
    depth_z = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5]
    duration_z = [1.0 - 0.5 * z for z in depth_z]  # exactly collinear with depth
    residual = cross_sectional_residualize(duration_z, [depth_z])
    assert residual is not None
    assert max(abs(value) for value in residual) < 1e-9


def test_legD_strategy_abstains_without_orthogonal_dispersion() -> None:
    # Six identical underwater paths -> zero cross-sectional duration dispersion
    # -> the score collapses -> the sleeve abstains (never-raise).
    symbols = [f"S{i}" for i in range(6)]
    path = [100.0] * 40 + _lin(100.0, 75.0, 41)[1:] + [75.0] * 20
    series: Series = {symbol: list(path) for symbol in symbols}
    strategy = CrossSectionalTimeUnderWaterStrategy(
        _Bars(symbols),
        _Queue(),
        **dict(_CAND_KWARGS, lookback_bars=len(path), min_history_bars=30, vol_window=5),
    )
    _feed(strategy, symbols, series, len(path))
    assert strategy.events.items == []


# --------------------------------------------------------------------------- #
# LEG E: min-hold suppresses an inside-window side flip (the C1 rescue as a test).
# --------------------------------------------------------------------------- #


def _build_lege() -> tuple[list[str], Series]:
    n = 84
    flip = (
        [100.0]
        + _lin(100.0, 70.0, 8)[1:]
        + [70.0] * (45 - 8)
        + _lin(70.0, 108.0, 17)[1:]
        + _lin(108.0, 76.0, 24)[1:]
    )
    flip = flip[:n]
    while len(flip) < n:
        flip.append(76.0)

    def mid(peak_bar: int, final: float, seed: int) -> list[float]:
        base = [100.0] * (peak_bar + 1) + _lin(100.0, final, n - peak_bar)[1:]
        return [x * (1.0 + ((seed * 7 + i * 3) % 5 - 2) * 0.0008) for i, x in enumerate(base[:n])]

    symbols = ["FLIP", "M1", "M2", "M3", "M4", "M5"]
    series: Series = {
        "FLIP": flip,
        "M1": mid(16, 72.0, 1),
        "M2": mid(18, 74.0, 2),
        "M3": mid(20, 71.0, 3),
        "M4": mid(22, 73.0, 4),
        "M5": mid(19, 70.0, 5),
    }
    for closes in series.values():
        assert len(closes) == n
    return symbols, series


def test_legE_min_hold_suppresses_flip() -> None:
    symbols, series = _build_lege()
    n = len(series["FLIP"])
    common = dict(
        lookback_bars=n,
        min_history_bars=20,
        depth_enter=-0.12,
        depth_exit=-0.08,
        depth_floor=-0.99,
        score_mode="duration",
        vol_window=5,
        quantile_pct=0.25,
        rebalance_bars=7,
        cooldown_decisions=0,
        rank_hysteresis_buffer=0,
        residualize=False,
        allow_short=True,
        min_symbols=5,
        target_gross_exposure=1.0,
        target_vol=0.0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
        min_price=0.01,
    )
    held = CrossSectionalTimeUnderWaterStrategy(
        _Bars(symbols), _Queue(), **dict(common, min_hold_decisions=1000)
    )
    _feed(held, symbols, series, n)
    held_kinds = [str(sig.signal_type).upper() for sig in held.events.items if sig.symbol == "FLIP"]
    assert "LONG" in held_kinds
    assert "SHORT" not in held_kinds  # min-hold suppressed the flip

    flips = CrossSectionalTimeUnderWaterStrategy(
        _Bars(symbols), _Queue(), **dict(common, min_hold_decisions=0)
    )
    _feed(flips, symbols, series, n)
    flip_kinds = [
        str(sig.signal_type).upper() for sig in flips.events.items if sig.symbol == "FLIP"
    ]
    assert "LONG" in flip_kinds
    assert "SHORT" in flip_kinds  # min_hold=0 reference DOES flip


def test_min_hold_reduces_turnover() -> None:
    """RPT design property: the hard min-hold materially cuts side-changing signals."""
    series = _build_gate_series()

    def _churn(min_hold: int) -> int:
        strategy = CrossSectionalTimeUnderWaterStrategy(
            _Bars(_GATE_SYMBOLS),
            _Queue(),
            **dict(_CAND_KWARGS, rebalance_bars=7, min_hold_decisions=min_hold),
        )
        _feed(strategy, _GATE_SYMBOLS, series, _N)
        return len([sig for sig in strategy.events.items if str(sig.signal_type).upper() != "EXIT"])

    assert _churn(8) <= _churn(0)


# --------------------------------------------------------------------------- #
# score_mode="duration_recovery" still produces a book; residualize ablation.
# --------------------------------------------------------------------------- #


def test_recovery_score_mode_emits_book() -> None:
    series = _build_gate_series()
    strategy = CrossSectionalTimeUnderWaterStrategy(
        _Bars(_GATE_SYMBOLS),
        _Queue(),
        **dict(_CAND_KWARGS, score_mode="duration_recovery", recovery_window=28),
    )
    _feed(strategy, _GATE_SYMBOLS, series, _N)
    side = _final_side(strategy.events.items)
    assert side, side
    assert any(v == "LONG" for v in side.values())
    assert any(v == "SHORT" for v in side.values())


def test_residualize_ablation_cell_still_splits_pair() -> None:
    series = _build_gate_series()
    strategy = CrossSectionalTimeUnderWaterStrategy(
        _Bars(_GATE_SYMBOLS), _Queue(), **dict(_CAND_KWARGS, residualize=False)
    )
    _feed(strategy, _GATE_SYMBOLS, series, _N)
    side = _final_side(strategy.events.items)
    assert side.get("STALE") == "LONG", side
    assert side.get("FRESH") == "SHORT", side


# --------------------------------------------------------------------------- #
# Determinism + state roundtrip + resumed behavior.
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    series = _build_gate_series()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = CrossSectionalTimeUnderWaterStrategy(
            _Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS
        )
        _feed(strategy, _GATE_SYMBOLS, series, _N)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


def test_state_roundtrip_lossless() -> None:
    series = _build_gate_series()
    strategy = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    _feed(strategy, _GATE_SYMBOLS, series, _N)
    snapshot = strategy.get_state()

    restored = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again == snapshot
    for symbol in _GATE_SYMBOLS:
        r = restored._state[symbol]
        o = strategy._state[symbol]
        assert list(r.closes) == list(o.closes)
        assert list(r.volumes) == list(o.volumes)
        assert r.mode == o.mode
        assert r.bars_held == o.bars_held
        assert r.cooldown == o.cooldown
    assert restored._tick == strategy._tick


def test_restored_state_reproduces_book() -> None:
    series = _build_gate_series()
    split = _N - 3
    full = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    _feed(full, _GATE_SYMBOLS, series, _N)

    warm = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    _feed(warm, _GATE_SYMBOLS, series, split)
    resumed = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    resumed.set_state(warm.get_state())
    for idx in range(split, _N):
        bars_1s = {
            sym: [
                {
                    "time": _ts(idx),
                    "open": series[sym][idx],
                    "high": series[sym][idx],
                    "low": series[sym][idx],
                    "close": series[sym][idx],
                    "volume": 1000.0,
                }
            ]
            for sym in _GATE_SYMBOLS
        }
        resumed.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=_ts(idx), bars_1s=bars_1s)
        )
    for symbol in _GATE_SYMBOLS:
        assert resumed._state[symbol].mode == full._state[symbol].mode, symbol
    assert full._state["STALE"].mode == "LONG"
    assert full._state["FRESH"].mode == "SHORT"


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A", "B", "C", "D", "E", "F"]
    strategy = CrossSectionalTimeUnderWaterStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)

    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A": "not a dict either"}})
    strategy.set_state({"symbol_state": {"A": {"closes": 12345}}})
    strategy.set_state({"symbol_state": {"A": {"volumes": {"nested": "dict"}}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "volumes": {"unexpected": "type"},
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "cooldown": "nope",
                    "last_time_key": 123,
                    "score": [1, 2, 3],
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


# --------------------------------------------------------------------------- #
# Never-raise on degenerate input + self-skip.
# --------------------------------------------------------------------------- #


def _market_event(symbol: str, idx: int, close: Any) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=_ts(idx),
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


def test_degenerate_market_events_never_raise() -> None:
    strategy = CrossSectionalTimeUnderWaterStrategy(_Bars(["Z"]), _Queue(), **_CAND_KWARGS)
    strategy.calculate_signals(_market_event("Z", 0, 0.0))
    strategy.calculate_signals(_market_event("Z", 1, -5.0))
    strategy.calculate_signals(_market_event("Z", 2, float("nan")))
    strategy.calculate_signals(_market_event("Z", 3, float("inf")))
    strategy.calculate_signals(_market_event("Z", 4, None))
    assert strategy.events.items == []


def test_empty_and_unknown_events_never_raise() -> None:
    strategy = CrossSectionalTimeUnderWaterStrategy(_Bars(["Z", "Y"]), _Queue(), **_CAND_KWARGS)
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", bars_1s={}, time="t0"))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="UNKNOWN", close=None))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z", close=None))
    assert strategy.events.items == []


def test_self_skip_below_min_symbols() -> None:
    series = _build_gate_series()
    symbols = ["STALE", "FRESH"]  # min_symbols is 5
    sub = {s: series[s] for s in symbols}
    strategy = CrossSectionalTimeUnderWaterStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed(strategy, symbols, sub, _N)
    assert strategy.events.items == []


def test_self_skip_when_history_too_short() -> None:
    series = _build_gate_series()
    strategy = CrossSectionalTimeUnderWaterStrategy(_Bars(_GATE_SYMBOLS), _Queue(), **_CAND_KWARGS)
    _feed(strategy, _GATE_SYMBOLS, series, 12)  # far below min_history_bars=60
    assert strategy.events.items == []


# --------------------------------------------------------------------------- #
# schema sanity (not a registry/tier/candidate-wiring assertion)
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalTimeUnderWaterStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "lookback_bars",
        "min_history_bars",
        "depth_enter",
        "depth_exit",
        "depth_floor",
        "score_mode",
        "recovery_window",
        "vol_window",
        "quantile_pct",
        "rebalance_bars",
        "min_hold_decisions",
        "cooldown_decisions",
        "rank_hysteresis_buffer",
        "residualize",
        "allow_short",
        "min_symbols",
        "min_dollar_volume",
        "target_gross_exposure",
        "target_vol",
        "stop_loss_pct",
        "max_hold_bars",
        "base_allocation",
        "max_symbol_exposure_pct",
        "max_order_value",
        "min_price",
    ):
        assert required in schema
    for cap in ("base_allocation", "max_symbol_exposure_pct", "max_order_value"):
        assert schema[cap].tunable is False
