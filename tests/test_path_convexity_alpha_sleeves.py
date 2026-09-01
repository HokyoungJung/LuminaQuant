"""Deterministic build-gate + hygiene tests for the path-convexity XS sleeve.

Direct class import only (no ``@register`` on this lane).  The build gate proves
the orthonormal-curvature book is behaviorally distinct from the occupied
trailing-return-level trend axis by RUNNING the real incumbents on hand-built
fixtures:

- a decelerating winner (level up, curvature down) is ``ProfitMoonshotTrend``'s
  LONG and this sleeve's SHORT -- opposite action; ``AccelerationRider`` abstains
  with pinned roc reasons;
- an accelerating base (level still down, curvature up) is NOT longed by the
  moonshot (level negative) while this sleeve LONGs it -- divergent; the rider
  abstains again;
- pure linear-drift fillers (zero curvature) are the moonshot's level longs but
  this sleeve EXCLUDES them -- the basis-orthogonality guarantee.

Plus the property test: inputs differing only by a linear trend produce the
identical convexity score (zero first-order loading).  Randomness for fillers is
a small seeded LCG (no ``random`` module); every run is bit-for-bit reproducible.
"""

from __future__ import annotations

import datetime
import math
from collections import deque
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.log_price_regression import orthonormal_path_convexity
from lumina_quant.indicators.oscillators import rate_of_change
from lumina_quant.strategies.path_convexity_alpha_sleeves import (
    _PATH_CONVEXITY_SLICE,
    CrossSectionalPathConvexityStrategy,
)
from lumina_quant.strategies.profit_moonshot import ProfitMoonshotTrendStrategy
from lumina_quant.strategies.return_rider_alpha_sleeves import (
    AccelerationRiderStrategy,
    _RiderState,
)
from lumina_quant.tuning import HyperParam

_T = 200
_ENTRY_THRESHOLD = 0.012  # ProfitMoonshotTrend default entry_threshold
_START = datetime.date(2025, 1, 1)


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


def _feed(
    strategy: Any,
    series: dict[str, list[float]],
    *,
    step_days: int = 1,
    start: datetime.date = _START,
    via_window_method: bool = False,
) -> None:
    n = len(next(iter(series.values())))
    for idx in range(n):
        stamp = (start + datetime.timedelta(days=step_days * idx)).isoformat() + "T00:00:00Z"
        bars_1s = {
            symbol: [
                {
                    "time": stamp,
                    "open": values[idx],
                    "high": values[idx],
                    "low": values[idx],
                    "close": values[idx],
                    "volume": 1000.0,
                }
            ]
            for symbol, values in series.items()
        }
        event = SimpleNamespace(type="MARKET_WINDOW", time=stamp, bars_1s=bars_1s)
        if via_window_method:
            strategy.calculate_signals_window(event, None)
        else:
            strategy.calculate_signals(event)


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


# --------------------------------------------------------------------------- #
# deterministic fixtures
# --------------------------------------------------------------------------- #


def _decel(zz: float = 0.0015) -> list[float]:
    """Concave-INCREASING throughout: level up, curvature down, roc>0 falling."""
    return [math.exp(0.010 * t - 1.0e-5 * t * t + zz * ((-1) ** t)) for t in range(_T)]


def _accel(zz: float = 0.0015) -> list[float]:
    """Convex-DECREASING throughout: level down, curvature up, roc<0 rising."""
    return [math.exp(1.0e-5 * t * (t - 2 * 205) + zz * ((-1) ** t)) for t in range(_T)]


def _linear_filler(drift: float) -> list[float]:
    """Pure linear log-drift + small zigzag: zero curvature, nonzero vol."""
    return [math.exp(drift * t + 0.01 * ((-1) ** t)) for t in range(_T)]


def _fillers() -> dict[str, list[float]]:
    return {f"F{i}/USDT": _linear_filler(d) for i, d in enumerate((0.002, 0.004, 0.006, 0.008))}


def _universe() -> dict[str, list[float]]:
    return {"DECEL/USDT": _decel(), "ACCEL/USDT": _accel(), **_fillers()}


# --------------------------------------------------------------------------- #
# stage-1 premises
# --------------------------------------------------------------------------- #


def test_stage1_premises() -> None:
    universe = _universe()
    c2 = {sym: orthonormal_path_convexity(vals, window=56) for sym, vals in universe.items()}
    # DECEL is the cross-section curvature MINIMUM, ACCEL the MAXIMUM, fillers ~0.
    assert c2["DECEL/USDT"] == min(c2.values()) and c2["DECEL/USDT"] < 0.0
    assert c2["ACCEL/USDT"] == max(c2.values()) and c2["ACCEL/USDT"] > 0.0
    for sym in _fillers():
        assert abs(c2[sym]) < 1.0

    decel, accel = universe["DECEL/USDT"], universe["ACCEL/USDT"]
    # DECEL: roc>0 and falling (decelerating rise).
    assert rate_of_change(decel, period=8) > 0.0
    assert rate_of_change(decel, period=8) < rate_of_change(decel[:-1], period=8)
    # ACCEL: roc<0 and rising (decelerating decline).
    assert rate_of_change(accel, period=8) < 0.0
    assert rate_of_change(accel, period=8) > rate_of_change(accel[:-1], period=8)


# --------------------------------------------------------------------------- #
# BUILD GATE -- moonshot (level) vs candidate (curvature): opposite action
# --------------------------------------------------------------------------- #


def test_gate_vs_profit_moonshot_trend() -> None:
    universe = _universe()

    moonshot = ProfitMoonshotTrendStrategy(
        _Bars(list(universe)), _Queue(), rebalance_bars=1, max_longs=6, max_shorts=6
    )
    _feed(moonshot, universe, via_window_method=True)
    ms_side = _final_side(moonshot.events.items)
    _targets, centered = moonshot._targets()

    candidate = CrossSectionalPathConvexityStrategy(_Bars(list(universe)), _Queue())
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)

    # Incumbent-LIVE: the moonshot trades a definite book (level-driven).
    assert ms_side, "moonshot emitted no targets"
    # DECEL: moonshot LONGs it (centered >= entry_threshold), candidate SHORTS it.
    assert centered["DECEL/USDT"] >= _ENTRY_THRESHOLD
    assert ms_side.get("DECEL/USDT") == "LONG"
    assert cand_side.get("DECEL/USDT") == "SHORT"
    # ACCEL: moonshot does NOT long it (centered < entry_threshold), candidate LONGs.
    assert centered["ACCEL/USDT"] < _ENTRY_THRESHOLD
    assert ms_side.get("ACCEL/USDT") != "LONG"
    assert cand_side.get("ACCEL/USDT") == "LONG"
    # Orthogonality: linear-drift fillers are moonshot longs but candidate excludes.
    for sym in _fillers():
        assert centered[sym] >= _ENTRY_THRESHOLD
        assert ms_side.get(sym) == "LONG"
        assert sym not in cand_side


def test_gate_vs_acceleration_rider_abstains() -> None:
    universe = _universe()
    rider = AccelerationRiderStrategy(_Bars(list(universe)), _Queue(), roc_period=8)

    decel, accel = universe["DECEL/USDT"], universe["ACCEL/USDT"]

    # DECEL: roc>0 blocks the SHORT gate, roc<prev_roc blocks the LONG gate.
    roc_d = rate_of_change(decel, period=8)
    prev_d = rate_of_change(decel[:-1], period=8)
    assert roc_d > 0.0 and roc_d < prev_d
    state_d = _RiderState(
        opens=deque(), highs=deque(), lows=deque(), closes=deque(decel), prev_roc=prev_d
    )
    assert rider._entry_decision(state_d) == ""

    # ACCEL: roc<0 blocks the LONG gate, roc>prev_roc blocks the SHORT gate.
    roc_a = rate_of_change(accel, period=8)
    prev_a = rate_of_change(accel[:-1], period=8)
    assert roc_a < 0.0 and roc_a > prev_a
    state_a = _RiderState(
        opens=deque(), highs=deque(), lows=deque(), closes=deque(accel), prev_roc=prev_a
    )
    assert rider._entry_decision(state_a) == ""

    # Candidate ACTS on both (both book sides non-empty) on the same fixture.
    candidate = CrossSectionalPathConvexityStrategy(_Bars(list(universe)), _Queue())
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)
    assert cand_side.get("DECEL/USDT") == "SHORT"
    assert cand_side.get("ACCEL/USDT") == "LONG"


# --------------------------------------------------------------------------- #
# orthogonality-to-momentum property (strategy layer)
# --------------------------------------------------------------------------- #


def test_convexity_score_invariant_to_linear_trend() -> None:
    # Two symbols with the SAME curvature but different linear trends must land on
    # the same book side; the sleeve carries no first-order loading.
    curved = [math.exp(2.0e-5 * t * t + 0.0015 * ((-1) ** t)) for t in range(_T)]  # convex
    curved_ramped = [curved[t] * math.exp(0.02 * t) for t in range(_T)]  # + steep linear up
    universe = {
        "CVXA/USDT": curved,
        "CVXB/USDT": curved_ramped,
        **{f"F{i}/USDT": _linear_filler(d) for i, d in enumerate((0.001, -0.001, 0.0))},
    }
    # Indicator-level: identical convexity score despite the added linear trend.
    s_a = orthonormal_path_convexity(curved, window=56)
    s_b = orthonormal_path_convexity(curved_ramped, window=56)
    assert s_a is not None and s_b is not None and abs(s_a - s_b) < 1e-9

    candidate = CrossSectionalPathConvexityStrategy(
        _Bars(list(universe)), _Queue(), min_symbols=5, quantile=0.40
    )
    _feed(candidate, universe)
    side = _final_side(candidate.events.items)
    # Both convex twins share the long tail (same curvature) even though CVXB has a
    # much stronger trailing return.
    assert side.get("CVXA/USDT") == "LONG"
    assert side.get("CVXB/USDT") == "LONG"


# --------------------------------------------------------------------------- #
# min-hold rescue (with a load-bearing contrast)
# --------------------------------------------------------------------------- #


def _lcg_filler(seed: int, n: int) -> list[float]:
    state = (seed * 2654435761) & 0xFFFFFFFF
    out: list[float] = []
    logp = 0.0
    for t in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        u = state / float(0x7FFFFFFF) - 0.5
        logp += 0.001 * seed + 0.01 * ((-1) ** t) + 0.002 * u
        out.append(math.exp(logp))
    return out


def _run_min_hold(min_hold: int) -> tuple[str, list[str]]:
    window = 12
    n = window + 1
    # P: increasing-CONVEX (top curvature) then a crash on the last bar so the
    # trailing window flips CONCAVE (bottom curvature).
    logs = [0.02 * t + 0.001 * t * t for t in range(window)]
    logs.append(logs[-1] - 0.5)
    p = [math.exp(v) for v in logs]
    universe = {"P/USDT": p, **{f"F{i}/USDT": _lcg_filler(i + 1, n) for i in range(4)}}
    strategy = CrossSectionalPathConvexityStrategy(
        _Bars(list(universe)),
        _Queue(),
        window_bars=window,
        min_symbols=5,
        min_hold_decisions=min_hold,
        vol_window=6,
        quantile=0.20,
        hysteresis_exit_pct=0.40,
    )
    _feed(strategy, universe, step_days=7, start=datetime.date(2025, 1, 6))
    p_signals = [sig.signal_type for sig in strategy.events.items if sig.symbol == "P/USDT"]
    return strategy._state["P/USDT"].mode, p_signals


def test_min_hold_suppresses_flip_inside_window() -> None:
    mode_hold, sig_hold = _run_min_hold(4)
    assert mode_hold == "LONG"
    assert sig_hold == ["LONG"]  # entered once, flip inside min-hold suppressed
    mode_free, sig_free = _run_min_hold(1)
    assert mode_free == "OUT"
    assert "EXIT" in sig_free  # contrast: guard off -> the flip exits


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical() -> None:
    universe = _universe()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = CrossSectionalPathConvexityStrategy(_Bars(list(universe)), _Queue())
        _feed(strategy, universe)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strategy.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal"


def test_state_roundtrip_lossless() -> None:
    universe = _universe()
    strategy = CrossSectionalPathConvexityStrategy(_Bars(list(universe)), _Queue())
    _feed(strategy, universe)
    snapshot = strategy.get_state()

    restored = CrossSectionalPathConvexityStrategy(_Bars(list(universe)), _Queue())
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    for symbol in universe:
        assert list(restored._state[symbol].closes) == list(strategy._state[symbol].closes)
        assert restored._state[symbol].mode == strategy._state[symbol].mode
        assert restored._state[symbol].bars_held == strategy._state[symbol].bars_held


def test_adversarial_set_state_never_raises() -> None:
    symbols = [f"S{i}/USDT" for i in range(5)]
    strategy = CrossSectionalPathConvexityStrategy(_Bars(symbols), _Queue())
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state([1, 2, 3])  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "nope"})
    strategy.set_state({"symbol_state": {"S0/USDT": 42}})
    strategy.set_state(
        {
            "last_eval_week": 999,
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), float("inf"), 12.5, None],
                    "mode": object(),
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "last_time_key": 123,
                    "score": {"nested": 1},
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}


def test_degenerate_inputs_never_raise() -> None:
    strategy = CrossSectionalPathConvexityStrategy(_Bars(["Z/USDT"]), _Queue())
    stamp = "2026-01-01T00:00:00Z"
    for close in (0.0, -5.0, float("nan"), float("inf")):
        strategy.calculate_signals(
            SimpleNamespace(
                type="MARKET",
                time=stamp,
                symbol="Z/USDT",
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1000.0,
            )
        )
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=stamp, bars_1s={}))
    assert strategy.events.items == []


def test_self_skip_below_min_symbols() -> None:
    universe = {"DECEL/USDT": _decel(), "ACCEL/USDT": _accel(), "F0/USDT": _linear_filler(0.003)}
    strategy = CrossSectionalPathConvexityStrategy(_Bars(list(universe)), _Queue(), min_symbols=5)
    _feed(strategy, universe)
    assert _non_exit(strategy.events.items) == []


def test_self_skip_history_too_short() -> None:
    short = {f"F{i}/USDT": _linear_filler(0.001 * (i + 1))[:20] for i in range(5)}
    strategy = CrossSectionalPathConvexityStrategy(_Bars(list(short)), _Queue(), window_bars=56)
    _feed(strategy, short)
    assert _non_exit(strategy.events.items) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalPathConvexityStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "window_bars",
        "quantile",
        "hysteresis_exit_pct",
        "min_hold_decisions",
        "vol_floor",
        "min_symbols",
        "allow_short",
        "target_gross_exposure",
    ):
        assert required in schema


def test_slice_timeframe_expansion_scales_bar_windows() -> None:
    """4h/1h cells mirror the 1d variants: the curvature / vol bar windows scale
    x6/x24, while the weekly min-hold, quantile, and hysteresis-exit band stay
    timeframe-invariant."""
    slice_ = _PATH_CONVEXITY_SLICE
    assert set(slice_) == {"4h", "1h", "1d"}
    variants = {tf: tuple(cell["variant"] for cell in cells) for tf, cells in slice_.items()}
    assert variants["4h"] == variants["1h"] == variants["1d"]
    by = {tf: {cell["variant"]: cell for cell in cells} for tf, cells in slice_.items()}
    for variant in variants["1d"]:
        d, h4, h1 = by["1d"][variant], by["4h"][variant], by["1h"][variant]
        for key in ("window_bars", "vol_window"):
            assert h4[key] == d[key] * 6, (variant, key)
            assert h1[key] == d[key] * 24, (variant, key)
        for key in (
            "quantile",
            "hysteresis_exit_pct",
            "min_hold_decisions",
            "target_gross_exposure",
        ):
            assert h4[key] == d[key] == h1[key], (variant, key)
