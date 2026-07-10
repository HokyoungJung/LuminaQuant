"""Deterministic tests for CrossSectionalNearLowRecoveryStrategy (W3-2, core).

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the W3 integration wave).

The load-bearing tests are the BUILD GATE legs: on a single synthetic
cross-section where a V-shaped recovery (VSHAPE) and a monotone grind into a
fresh low (GRINDER) have IDENTICAL nearness-to-high, the near-low sleeve must
SPLIT the pair (long the aged recovery, short the fresh low) where both the real
``CrossSectionalNearHighAnchoringStrategy`` (a level statistic) and the real
``LongRunOverreactionReversalStrategy`` (a path-blind formation return) tie them
on the SAME side.  If that divergence cannot be produced with an honest
implementation the leaf is behaviorally spanned and must be dropped -- the gate
is never weakened to pass.

No ``random`` module is used: the fixture paths are deterministic piecewise
lines with genuine curvature (so realised vol is non-degenerate), and the one
hygiene jitter is a small seeded LCG.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize
from lumina_quant.strategies.longrun_overreaction_alpha_sleeves import (
    LongRunOverreactionReversalStrategy,
)
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.strategies.near_low_recovery_alpha_sleeves import (
    CrossSectionalNearLowRecoveryStrategy,
    _low_recency,
    _rebound_log,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module) -- only for a hygiene vol jitter.
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def _jitter(prices: list[float], seed: int, amp: float = 0.0012) -> list[float]:
    gen = _lcg_stream(seed)
    return [price * (1.0 + (next(gen) - 0.5) * 2.0 * amp) for price in prices]


# --------------------------------------------------------------------------- #
# harness (MARKET_WINDOW feed keeps the cross-section coherent per bar)
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
    return f"2026-01-01T{idx // 60:02d}:{idx % 60:02d}:00Z"


def _pw(points: list[tuple[int, float]], n: int) -> list[float]:
    """Piecewise-linear close path hitting ``(bar, price)`` knots, length ``n``."""
    out: list[float] = []
    for t in range(n):
        placed = False
        for i in range(len(points) - 1):
            t0, p0 = points[i]
            t1, p1 = points[i + 1]
            if t0 <= t <= t1:
                out.append(p1 if t1 == t0 else p0 + (p1 - p0) * (t - t0) / (t1 - t0))
                placed = True
                break
        if not placed:
            out.append(points[-1][1])
    return out


# series[symbol] -> (closes, lows, highs)
Series = dict[str, tuple[list[float], list[float], list[float]]]


def _feed_window(
    strategy: Any,
    symbols: list[str],
    series: Series,
    n: int,
    first_bar: dict[str, int] | None = None,
) -> None:
    first_bar = first_bar or {}
    for idx in range(n):
        bars_1s: dict[str, list[dict[str, Any]]] = {}
        for symbol in symbols:
            start = first_bar.get(symbol, 0)
            if idx < start:
                continue
            local = idx - start
            closes, lows, highs = series[symbol]
            if local >= len(closes):
                continue
            bars_1s[symbol] = [
                {
                    "time": _ts(idx),
                    "open": closes[local],
                    "high": highs[local],
                    "low": lows[local],
                    "close": closes[local],
                    "volume": 1000.0,
                }
            ]
        strategy.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=_ts(idx), bars_1s=bars_1s)
        )


def _flat(closes: list[float]) -> tuple[list[float], list[float], list[float]]:
    """Package a close path with low=close, high=close."""
    return closes, list(closes), list(closes)


def _entries(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if sig.signal_type in {"LONG", "SHORT"}]


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


# --------------------------------------------------------------------------- #
# BUILD-GATE cross-section: nearness-to-HIGH ties VSHAPE and GRINDER; the low
# order-statistic + argmin recency SPLITS them.
#
#   VSHAPE  : early high 160, crashes to 55 @ bar 30, recovers to 88 (never
#             re-touching) -> nearness 0.55, big AGED rebound.
#   GRINDER : same early high 160, monotone grind to a FRESH low 88 printed at
#             the final bar -> nearness 0.55 (IDENTICAL), zero rebound/recency.
#   NEARHIGH: pinned near its trailing high (nearness ~0.99), oscillating.
#   MID1..3 : mid-nearness fillers giving the XS regression breadth.
# --------------------------------------------------------------------------- #

_N = 100
# LRO formation knots (formation_bars=40, skip_bars=5 -> recent=close[94], base=close[54]).
_GRINDER = _pw([(0, 160.0), (99, 88.0)], _N)
_C54 = _GRINDER[54]
_C94 = _GRINDER[94]
_VSHAPE = _pw([(0, 160.0), (5, 160.0), (30, 55.0), (54, _C54), (94, _C94), (99, 88.0)], _N)
_NEARHIGH = _pw([(0, 100.0), (40, 157.0), (99, 157.0)], _N)
_NEARHIGH = [c + (1.5 if i % 2 else -1.5) if i >= 40 else c for i, c in enumerate(_NEARHIGH)]
_MID1 = _pw([(0, 120.0), (25, 100.0), (50, 118.0), (75, 102.0), (99, 110.0)], _N)
_MID2 = _pw([(0, 130.0), (30, 108.0), (60, 125.0), (99, 112.0)], _N)
_MID3 = _pw([(0, 115.0), (20, 98.0), (55, 112.0), (85, 100.0), (99, 106.0)], _N)


def _high_with_early_peak(closes: list[float], peak_bars: range, peak: float) -> list[float]:
    return [max(peak, closes[i]) if i in peak_bars else closes[i] for i in range(len(closes))]


def _gate_universe() -> tuple[list[str], Series]:
    symbols = [
        "VSHAPE/USDT",
        "GRINDER/USDT",
        "NEARHIGH/USDT",
        "MID1/USDT",
        "MID2/USDT",
        "MID3/USDT",
    ]
    series: Series = {
        # low=close (crash low printed by the close itself); explicit early high 160.
        "VSHAPE/USDT": (_VSHAPE, list(_VSHAPE), _high_with_early_peak(_VSHAPE, range(0, 6), 160.0)),
        "GRINDER/USDT": (
            _GRINDER,
            list(_GRINDER),
            _high_with_early_peak(_GRINDER, range(0, 1), 160.0),
        ),
        "NEARHIGH/USDT": (
            _NEARHIGH,
            list(_NEARHIGH),
            _high_with_early_peak(_NEARHIGH, range(40, _N), 160.0),
        ),
        "MID1/USDT": _flat(_MID1),
        "MID2/USDT": _flat(_MID2),
        "MID3/USDT": _flat(_MID3),
    }
    return symbols, series


_CAND_KWARGS: dict[str, Any] = dict(
    low_lookback_bars=100,
    min_history_bars=60,
    vol_window=20,
    quantile_pct=0.4,  # 6 eligible -> n_side = 2
    rebalance_bars=1,
    min_hold_bars=0,
    residualize=True,
    allow_short=True,
    min_symbols=5,
    target_gross_exposure=1.0,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)

_INCUMBENT_KWARGS: dict[str, Any] = dict(
    high_lookback_bars=100,
    min_history_bars=60,
    vol_window=20,
    quantile_pct=0.4,  # n_side = 2 -> BOTH VSHAPE and GRINDER share the short set
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
    formation_bars=40,
    skip_bars=5,
    z_min=0.5,
    max_universe=12,
    rebalance_bars=1,
    min_hold_bars=0,
    quantile_pct=0.5,  # max_side = 2 -> both tied losers land on the SAME (long) side
    min_symbols=5,
    allow_short=True,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)


# --------------------------------------------------------------------------- #
# LEG 0 (stage-1): the fixture's statistical premise, asserted with the real
# module primitives BEFORE any strategy runs.
# --------------------------------------------------------------------------- #


def test_leg0_stage1_fixture_primitives() -> None:
    # Nearness-to-high is IDENTICAL for the V-shape and the grinder (both 0.55).
    near_vshape = _VSHAPE[-1] / max(_high_with_early_peak(_VSHAPE, range(0, 6), 160.0))
    near_grinder = _GRINDER[-1] / max(_high_with_early_peak(_GRINDER, range(0, 1), 160.0))
    assert abs(near_vshape - near_grinder) < 1e-9
    assert abs(near_vshape - 0.55) < 1e-9

    eff = _N
    reb_v = _rebound_log(_VSHAPE[-1], min(_VSHAPE[-eff:]))
    reb_g = _rebound_log(_GRINDER[-1], min(_GRINDER[-eff:]))
    assert reb_v is not None and reb_g is not None
    # The V-shape carries a large AGED rebound; the grinder is at a fresh low.
    assert reb_v > 0.4
    assert abs(reb_v - math.log(88.0 / 55.0)) < 1e-9
    assert reb_g < 0.01

    rec_v = _low_recency(_VSHAPE[-eff:], eff)
    rec_g = _low_recency(_GRINDER[-eff:], eff)
    assert rec_v is not None and rec_g is not None
    assert rec_v > 0.5  # low is old (capitulation ~69% of the window ago)
    assert rec_g < 0.05  # fresh low printed at the final bar


# --------------------------------------------------------------------------- #
# LEG 2 (incumbent-LIVE + same-side tie) + LEG 3 (candidate-ACTS + divergent):
# the near-high incumbent shorts BOTH VSHAPE and GRINDER (tied) and longs
# NEARHIGH; the candidate LONGS VSHAPE and SHORTS GRINDER.
# --------------------------------------------------------------------------- #


def test_leg2_leg3_diverges_from_near_high_anchoring() -> None:
    symbols, series = _gate_universe()

    candidate = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed_window(candidate, symbols, series, _N)
    cand_side = _final_side(candidate.events.items)

    incumbent = CrossSectionalNearHighAnchoringStrategy(
        _Bars(symbols), _Queue(), **_INCUMBENT_KWARGS
    )
    _feed_window(incumbent, symbols, series, _N)
    inc_side = _final_side(incumbent.events.items)

    # LEG 2: the incumbent is LIVE (non-empty book) and TIES the pair on the SAME
    # (short) side -- nearness-to-high cannot see the low-side split.
    assert inc_side, "near-high incumbent must emit a non-empty book"
    assert inc_side.get("VSHAPE/USDT") == "SHORT", inc_side
    assert inc_side.get("GRINDER/USDT") == "SHORT", inc_side
    assert inc_side.get("NEARHIGH/USDT") == "LONG", inc_side

    # LEG 3: the candidate ACTS on both book sides and SPLITS the tied pair --
    # long the aged recovery, short the fresh-low grinder.  Opposite-signed on
    # VSHAPE vs the incumbent's short; opposite sides within the tied pair.
    assert cand_side.get("VSHAPE/USDT") == "LONG", cand_side
    assert cand_side.get("GRINDER/USDT") == "SHORT", cand_side
    assert any(v == "LONG" for v in cand_side.values())
    assert any(v == "SHORT" for v in cand_side.values())

    # The residualizer CANNOT erase the split: VSHAPE and GRINDER carry the SAME
    # nearness regressor value, so the composite difference survives residualizing.
    vshape_long = [
        sig
        for sig in _entries(candidate.events.items)
        if sig.symbol == "VSHAPE/USDT" and sig.signal_type == "LONG"
    ]
    assert vshape_long
    meta = vshape_long[-1].metadata or {}
    grinder_short = [
        sig
        for sig in _entries(candidate.events.items)
        if sig.symbol == "GRINDER/USDT" and sig.signal_type == "SHORT"
    ]
    assert grinder_short
    gmeta = grinder_short[-1].metadata or {}
    assert abs(float(meta["nearness_z"]) - float(gmeta["nearness_z"])) < 1e-9
    assert float(meta["residual"]) > float(gmeta["residual"])
    assert float(meta.get("target_allocation", 0.0)) > 0.0


# --------------------------------------------------------------------------- #
# LEG 4 vs LongRunOverreactionReversalStrategy: equal path-blind formation
# returns -> same side there; the candidate takes OPPOSITE sides.
# --------------------------------------------------------------------------- #


def test_leg4_diverges_from_longrun_overreaction() -> None:
    symbols, series = _gate_universe()

    lro = LongRunOverreactionReversalStrategy(_Bars(symbols), _Queue(), **_LRO_KWARGS)
    _feed_window(lro, symbols, series, _N)
    lro_side = _final_side(lro.events.items)

    # Stage-1: the incumbent's OWN formation primitive returns EQUAL values for
    # the pair (it is blind to the low geometry) -- the WHY of its same-side tie.
    form_v = lro._formation("VSHAPE/USDT")
    form_g = lro._formation("GRINDER/USDT")
    assert form_v is not None and form_g is not None
    assert abs(form_v[0] - form_g[0]) < 1e-6

    # Incumbent LIVE and puts the tied pair on the SAME side (both extreme losers
    # -> both LONG); it never splits them.
    assert lro_side, "LRO must emit a non-empty book"
    assert lro_side.get("VSHAPE/USDT") == lro_side.get("GRINDER/USDT")
    assert lro_side.get("VSHAPE/USDT") in {"LONG", "SHORT"}
    # It never splits the pair onto opposite sides.
    assert {lro_side.get("VSHAPE/USDT"), lro_side.get("GRINDER/USDT")} != {"LONG", "SHORT"}

    # The candidate SPLITS the identical-formation pair onto opposite sides.
    candidate = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed_window(candidate, symbols, series, _N)
    cand_side = _final_side(candidate.events.items)
    assert cand_side.get("VSHAPE/USDT") == "LONG"
    assert cand_side.get("GRINDER/USDT") == "SHORT"


# --------------------------------------------------------------------------- #
# LEG 5: the residualizer decoupler.  A composite exactly collinear with the
# nearness regressor residualizes to ~0; a dispersion-free cross-section makes
# the strategy abstain.
# --------------------------------------------------------------------------- #


def test_leg5_residualizer_collinear_composite_vanishes() -> None:
    nearness_z = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5]
    composite = [3.0 + 2.0 * z for z in nearness_z]  # exactly collinear (a + b*z)
    residual = cross_sectional_residualize(composite, [nearness_z])
    assert residual is not None
    assert max(abs(value) for value in residual) < 1e-9


def test_leg5_strategy_abstains_without_orthogonal_dispersion() -> None:
    # Six identical monotone-recovery paths: zero cross-sectional dispersion in
    # rebound/recency -> composite collapses -> the sleeve abstains (never-raise).
    symbols = [f"S{i}/USDT" for i in range(6)]
    path = _pw([(0, 120.0), (30, 80.0), (99, 110.0)], _N)
    series: Series = {symbol: _flat(list(path)) for symbol in symbols}
    strategy = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed_window(strategy, symbols, series, _N)
    assert strategy.events.items == []


# --------------------------------------------------------------------------- #
# LEG 6: min-hold suppresses an inside-window side flip (the C1 rescue as a test).
# --------------------------------------------------------------------------- #


def test_min_hold_suppresses_flip_inside_hold_window() -> None:
    # FLIP is the unambiguous top-residual name early (a big aged rebound off an
    # old low) and the unambiguous bottom-residual name late (it prints a fresh
    # low).  With a min-hold longer than the run the early LONG must persist.
    n = 60
    flip = _pw([(0, 120.0), (6, 60.0), (30, 118.0), (50, 116.0), (59, 55.0)], n)

    # Steady fillers whose low is old and rebound moderate throughout.
    def _mid(seed: int) -> tuple[list[float], list[float], list[float]]:
        base = _pw([(0, 118.0), (8, 96.0), (59, 108.0)], n)
        return _flat(_jitter(base, seed=seed))

    symbols = ["FLIP/USDT", "M1/USDT", "M2/USDT", "M3/USDT", "M4/USDT", "M5/USDT"]
    series: Series = {
        "FLIP/USDT": _flat(flip),
        "M1/USDT": _mid(4101),
        "M2/USDT": _mid(4102),
        "M3/USDT": _mid(4103),
        "M4/USDT": _mid(4104),
        "M5/USDT": _mid(4105),
    }
    common = dict(
        low_lookback_bars=n,
        min_history_bars=20,
        vol_window=5,
        quantile_pct=0.34,
        rebalance_bars=1,
        residualize=False,  # rank on the raw composite so the flip geometry is direct
        allow_short=True,
        min_symbols=5,
        target_gross_exposure=1.0,
        target_vol=0.0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
        min_price=0.01,
    )

    held = CrossSectionalNearLowRecoveryStrategy(
        _Bars(symbols), _Queue(), **dict(common, min_hold_bars=n)
    )
    _feed_window(held, symbols, series, n)
    held_kinds = [
        str(sig.signal_type).upper() for sig in held.events.items if sig.symbol == "FLIP/USDT"
    ]
    assert "LONG" in held_kinds
    assert "SHORT" not in held_kinds  # min-hold suppressed the flip

    flips = CrossSectionalNearLowRecoveryStrategy(
        _Bars(symbols), _Queue(), **dict(common, min_hold_bars=0)
    )
    _feed_window(flips, symbols, series, n)
    flip_kinds = [
        str(sig.signal_type).upper() for sig in flips.events.items if sig.symbol == "FLIP/USDT"
    ]
    assert "LONG" in flip_kinds
    assert "SHORT" in flip_kinds  # min_hold=0 reference DOES flip


# --------------------------------------------------------------------------- #
# residualize ablation cell: the {True, False} sweep both trade the split.
# --------------------------------------------------------------------------- #


def test_residualize_ablation_cell_still_splits_pair() -> None:
    symbols, series = _gate_universe()
    strategy = CrossSectionalNearLowRecoveryStrategy(
        _Bars(symbols), _Queue(), **dict(_CAND_KWARGS, residualize=False)
    )
    _feed_window(strategy, symbols, series, _N)
    side = _final_side(strategy.events.items)
    assert side.get("VSHAPE/USDT") == "LONG", side
    assert side.get("GRINDER/USDT") == "SHORT", side


# --------------------------------------------------------------------------- #
# Determinism: two identical runs -> bit-identical signal stream.
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical_signals() -> None:
    symbols, series = _gate_universe()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
        _feed_window(strategy, symbols, series, _N)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal in this scenario"


# --------------------------------------------------------------------------- #
# State roundtrip + adversarial set_state + resumed behavior.
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    symbols, series = _gate_universe()
    strategy = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed_window(strategy, symbols, series, _N)
    snapshot = strategy.get_state()

    restored = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    restored.set_state(snapshot)
    again = restored.get_state()

    assert again == snapshot
    for symbol in symbols:
        r = restored._state[symbol]
        o = strategy._state[symbol]
        assert list(r.closes) == list(o.closes)
        assert list(r.lows) == list(o.lows)
        assert list(r.highs) == list(o.highs)
        assert r.mode == o.mode
        assert r.bars_held == o.bars_held
    assert restored._tick == strategy._tick
    assert restored._last_eval_time_key == strategy._last_eval_time_key


def test_restored_state_reproduces_book() -> None:
    symbols, series = _gate_universe()
    split = _N - 3

    full = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed_window(full, symbols, series, _N)

    warm = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed_window(warm, symbols, series, split)
    resumed = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    resumed.set_state(warm.get_state())
    for idx in range(split, _N):
        bars_1s = {
            sym: [
                {
                    "time": _ts(idx),
                    "open": series[sym][0][idx],
                    "high": series[sym][2][idx],
                    "low": series[sym][1][idx],
                    "close": series[sym][0][idx],
                    "volume": 1000.0,
                }
            ]
            for sym in symbols
        }
        resumed.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=_ts(idx), bars_1s=bars_1s)
        )

    for symbol in symbols:
        assert resumed._state[symbol].mode == full._state[symbol].mode, symbol
    assert full._state["VSHAPE/USDT"].mode == "LONG"
    assert full._state["GRINDER/USDT"].mode == "SHORT"


def test_adversarial_set_state_never_raises() -> None:
    symbols = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT", "F/USDT"]
    strategy = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)

    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state([])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "not a dict"})
    strategy.set_state({"symbol_state": {"A/USDT": "not a dict either"}})
    strategy.set_state({"symbol_state": {"A/USDT": {"closes": 12345}}})
    strategy.set_state({"symbol_state": {"A/USDT": {"lows": {"nested": "dict"}}}})
    strategy.set_state(
        {
            "last_eval_time_key": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", "y", float("nan"), float("inf"), 12.5, None],
                    "lows": {"unexpected": "type"},
                    "highs": 999,
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
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
    strategy = CrossSectionalNearLowRecoveryStrategy(_Bars(["Z/USDT"]), _Queue(), **_CAND_KWARGS)
    strategy.calculate_signals(_market_event("Z/USDT", 0, 0.0))
    strategy.calculate_signals(_market_event("Z/USDT", 1, -5.0))
    strategy.calculate_signals(_market_event("Z/USDT", 2, float("nan")))
    strategy.calculate_signals(_market_event("Z/USDT", 3, float("inf")))
    strategy.calculate_signals(_market_event("Z/USDT", 4, None))
    assert strategy.events.items == []


def test_empty_and_unknown_events_never_raise() -> None:
    strategy = CrossSectionalNearLowRecoveryStrategy(
        _Bars(["Z/USDT", "Y/USDT"]), _Queue(), **_CAND_KWARGS
    )
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", bars_1s={}, time="t0"))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="UNKNOWN/USDT", close=None))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z/USDT", close=None))
    assert strategy.events.items == []


def test_self_skip_below_min_symbols() -> None:
    symbols = ["VSHAPE/USDT", "GRINDER/USDT"]  # min_symbols is 5
    _, full = _gate_universe()
    series: Series = {s: full[s] for s in symbols}
    strategy = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed_window(strategy, symbols, series, _N)
    assert strategy.events.items == []


def test_self_skip_when_history_too_short() -> None:
    symbols, series = _gate_universe()
    strategy = CrossSectionalNearLowRecoveryStrategy(_Bars(symbols), _Queue(), **_CAND_KWARGS)
    _feed_window(strategy, symbols, series, 8)  # far below min_history_bars=60
    assert strategy.events.items == []


# --------------------------------------------------------------------------- #
# Breadth: a young alt (< full lookback but >= min-history) is admitted via the
# max_available window; a below-min-history symbol is skipped without raising.
# --------------------------------------------------------------------------- #


def test_breadth_young_alt_admitted_via_max_available() -> None:
    n = 40
    symbols = [
        "YOUNG/USDT",
        "TOOSHORT/USDT",
        "F1/USDT",
        "F2/USDT",
        "F3/USDT",
        "F4/USDT",
    ]
    # YOUNG: 20 bars from bar 20, a big aged rebound off an early low -> top book.
    young = _pw([(0, 100.0), (4, 60.0), (19, 108.0)], 20)
    tooshort = _pw([(0, 100.0), (7, 105.0)], 8)
    series: Series = {
        "YOUNG/USDT": _flat(young),
        "TOOSHORT/USDT": _flat(tooshort),
        "F1/USDT": _flat(_jitter(_pw([(0, 118.0), (10, 96.0), (39, 108.0)], n), 5001)),
        "F2/USDT": _flat(_jitter(_pw([(0, 120.0), (12, 98.0), (39, 110.0)], n), 5002)),
        "F3/USDT": _flat(_jitter(_pw([(0, 116.0), (9, 95.0), (39, 106.0)], n), 5003)),
        "F4/USDT": _flat(_jitter(_pw([(0, 122.0), (14, 99.0), (39, 112.0)], n), 5004)),
    }
    first_bar = {"YOUNG/USDT": 20, "TOOSHORT/USDT": 32}
    strategy = CrossSectionalNearLowRecoveryStrategy(
        _Bars(symbols),
        _Queue(),
        low_lookback_bars=40,
        min_history_bars=12,
        vol_window=5,
        quantile_pct=0.25,
        rebalance_bars=1,
        min_hold_bars=0,
        residualize=False,
        allow_short=True,
        min_symbols=4,
        target_gross_exposure=1.0,
        target_vol=0.0,
        stop_loss_pct=0.0,
        max_hold_bars=0,
        min_price=0.01,
    )
    _feed_window(strategy, symbols, series, n, first_bar=first_bar)
    young_entries = [sig for sig in _entries(strategy.events.items) if sig.symbol == "YOUNG/USDT"]
    assert young_entries, "young alt should have been admitted into the ranking"
    meta = young_entries[-1].metadata or {}
    assert meta.get("full_lookback") is False, meta
    assert 12 <= meta.get("lookback_used") < 40, meta
    # The below-min-history symbol is never scored, never a signal, no raise.
    assert all(sig.symbol != "TOOSHORT/USDT" for sig in strategy.events.items)


# --------------------------------------------------------------------------- #
# schema sanity (not a registry/tier/candidate-wiring assertion)
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalNearLowRecoveryStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "low_lookback_bars",
        "min_history_bars",
        "vol_window",
        "quantile_pct",
        "rebalance_bars",
        "min_hold_bars",
        "residualize",
        "allow_short",
        "min_symbols",
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


def test_slice_multi_timeframe_cells_pinned() -> None:
    """4h/1h cells mirror the 1d variants and scale the bar clocks x6 / x24."""
    from lumina_quant.strategies.near_low_recovery_alpha_sleeves import (
        _NEAR_LOW_RECOVERY_SLICE as sl,
    )

    assert {"1d", "4h", "1h"} <= set(sl)
    base = tuple(cell["variant"] for cell in sl["1d"])
    for tf in ("4h", "1h"):
        assert tuple(cell["variant"] for cell in sl[tf]) == base
    assert sl["4h"][0]["low_lookback_bars"] == 2184
    assert sl["4h"][0]["rebalance_bars"] == 42
    assert sl["4h"][0]["vol_window"] == 120
    assert sl["1h"][0]["low_lookback_bars"] == 8736
    assert sl["1h"][0]["rebalance_bars"] == 168
    assert sl["1h"][0]["vol_window"] == 480
    # Unit-free knobs stay fixed across timeframes.
    for tf in ("1d", "4h", "1h"):
        assert sl[tf][0]["quantile_pct"] == 0.25
        assert sl[tf][0]["min_symbols"] == 6


# --------------------------------------------------------------------------- #
# vol-target horizon fix (Class-B throttle): regression.
# --------------------------------------------------------------------------- #


def test_vol_target_throttle_annualizes_per_bar_vol() -> None:
    """The Class-B throttle annualizes the PER-BAR portfolio vol by the median
    observed bar spacing (shared ``_annualize_per_bar_vol``) before comparing it to
    ``target_vol``: a hurricane per-1h vol now DE-RISKS, a calm one does not, and
    an unknown cadence stays at 1.0.
    """
    symbols = ["A/USDT", "B/USDT"]
    strategy = CrossSectionalNearLowRecoveryStrategy(
        _Bars(symbols), _Queue(), **dict(_CAND_KWARGS, target_vol=0.20)
    )
    base = 1_700_000_000.0
    for i in range(12):
        strategy._recent_times.append(base + i * 3600.0)  # clean 1h cadence
    targets = {"A/USDT": ("LONG", 1.0, {}), "B/USDT": ("SHORT", -1.0, {})}
    # Hurricane per-1h vol -> annualizes far above 0.20 -> throttle engages.
    _weights, hot = strategy._inverse_vol_weights(targets, {"A/USDT": 0.02, "B/USDT": 0.026})
    assert hot < 1.0, hot
    # Calm per-1h vol -> annualizes below 0.20 -> no de-risk.
    _weights, calm = strategy._inverse_vol_weights(targets, {"A/USDT": 0.0002, "B/USDT": 0.00026})
    assert calm == 1.0, calm
    # Determinism: same inputs -> identical scalar.
    _weights, hot2 = strategy._inverse_vol_weights(targets, {"A/USDT": 0.02, "B/USDT": 0.026})
    assert hot2 == hot
    # Unknown cadence (no observed bars) -> conservative unity scalar.
    fresh = CrossSectionalNearLowRecoveryStrategy(
        _Bars(symbols), _Queue(), **dict(_CAND_KWARGS, target_vol=0.20)
    )
    _weights, unknown = fresh._inverse_vol_weights(targets, {"A/USDT": 0.02, "B/USDT": 0.026})
    assert unknown == 1.0, unknown
