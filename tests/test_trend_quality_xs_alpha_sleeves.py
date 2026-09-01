"""Deterministic build-gate + hygiene tests for the trend-quality XS sleeve.

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions here).  The build gate proves the signed-R^2
trend-quality book is behaviorally distinct from the occupied KER path-efficiency
axis by RUNNING the real incumbents on hand-built fixtures:

- vs ``LowTurnoverTrendPersistenceStrategy`` on a zigzag-but-net-directional path
  the incumbent's efficiency gate keeps it flat (with a positive-control SHORT on
  a clean trend) while this sleeve LONGs it;
- vs ``TrendEfficiencyMomentumStrategy`` a flat-then-jump path is the incumbent's
  top long while this sleeve EXCLUDES it (low fit quality) and instead longs the
  smooth zigzag -- opposite top-of-book on identical input.

Any pseudo-randomness for the filler symbols is a small seeded LCG (no ``random``
module), so every run is bit-for-bit reproducible.
"""

from __future__ import annotations

import datetime
import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.log_price_regression import signed_trend_quality
from lumina_quant.indicators.momentum import kaufman_efficiency_ratio
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    TrendEfficiencyMomentumStrategy,
)
from lumina_quant.strategies.low_turnover_trend_alpha_sleeves import (
    LowTurnoverTrendPersistenceStrategy,
)
from lumina_quant.strategies.trend_quality_xs_alpha_sleeves import (
    _TREND_QUALITY_XS_SLICE,
    CrossSectionalRegressionTrendQualityStrategy,
)
from lumina_quant.tuning import HyperParam

_T = 140
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


def _date(idx: int, *, step_days: int = 1) -> str:
    return (_START + datetime.timedelta(days=step_days * idx)).isoformat() + "T00:00:00Z"


def _feed(
    strategy: Any,
    series: dict[str, list[float]],
    *,
    step_days: int = 1,
    start: datetime.date = _START,
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
        strategy.calculate_signals(
            SimpleNamespace(type="MARKET_WINDOW", time=stamp, bars_1s=bars_1s)
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


def _non_exit(signals: list[Any]) -> list[Any]:
    return [sig for sig in signals if str(sig.signal_type).upper() != "EXIT"]


# --------------------------------------------------------------------------- #
# deterministic fixtures
# --------------------------------------------------------------------------- #


def _x_zigzag() -> list[float]:
    """Zigzag-but-net-directional: KER collapses, signed-R^2 stays high."""
    return [math.exp(0.01 * t + 0.05 * ((-1) ** t)) for t in range(_T)]


def _y_smooth_down() -> list[float]:
    """Smooth (near-linear) downtrend: KER=1, signed-R^2 ~ -1, nonzero vol."""
    logs = [math.log(100.0)]
    for t in range(1, _T):
        logs.append(logs[-1] + (-0.010 if t % 2 == 1 else -0.014))
    return [math.exp(v) for v in logs]


def _z_flat_then_jump() -> list[float]:
    """Near-flat then a two-bar jump: KER saturates, signed-R^2 collapses."""
    out: list[float] = []
    for t in range(_T):
        if t < _T - 2:
            out.append(100.0 * (1.0 + 0.0004 * ((-1) ** t)))
        elif t == _T - 2:
            out.append(115.0)
        else:
            out.append(130.0)
    return out


def _chop(seed: int, n: int = _T) -> list[float]:
    """Choppy, low-fit-quality filler (LCG-seeded, nonzero vol, near-zero trend)."""
    state = (seed * 2654435761) & 0xFFFFFFFF
    out: list[float] = []
    logp = math.log(100.0)
    for t in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        u = state / float(0x7FFFFFFF) - 0.5
        logp += 0.035 * ((-1) ** t) + 0.006 * u
        out.append(math.exp(logp))
    return out


def _fillers(n: int = _T) -> dict[str, list[float]]:
    return {f"F{i}/USDT": _chop(i + 1, n) for i in range(3)}


# --------------------------------------------------------------------------- #
# stage-1 premises (fixture statistics asserted with the real primitives)
# --------------------------------------------------------------------------- #


def test_stage1_premises() -> None:
    x, y, z = _x_zigzag(), _y_smooth_down(), _z_flat_then_jump()
    assert kaufman_efficiency_ratio(x, period=20) < 0.30
    assert signed_trend_quality(x, window=56)[1] > 0.8  # r2
    assert signed_trend_quality(x, window=56)[0] > 0.8  # signed_r2 positive
    assert abs(kaufman_efficiency_ratio(y, period=20) - 1.0) < 1e-9
    assert signed_trend_quality(y, window=56)[0] < -0.9  # smooth downtrend
    assert kaufman_efficiency_ratio(z, period=20) >= 0.30  # jump saturates KER
    assert signed_trend_quality(z, window=56)[1] < 0.30  # low fit quality
    for values in _fillers().values():
        assert signed_trend_quality(values, window=56)[1] < 0.30  # non-tradeable fillers


# --------------------------------------------------------------------------- #
# BUILD GATE -- TEST 1: vs LowTurnoverTrendPersistenceStrategy
# --------------------------------------------------------------------------- #


def test_gate1_vs_low_turnover_trend_persistence() -> None:
    universe = {"X/USDT": _x_zigzag(), "Y/USDT": _y_smooth_down(), **_fillers()}

    incumbent = LowTurnoverTrendPersistenceStrategy(_Bars(list(universe)), _Queue())
    _feed(incumbent, universe)
    inc_side = _final_side(incumbent.events.items)

    candidate = CrossSectionalRegressionTrendQualityStrategy(_Bars(list(universe)), _Queue())
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)

    # Incumbent-LIVE positive control: the smooth trend Y is traded (SHORT).
    assert inc_side.get("Y/USDT") == "SHORT", inc_side
    # Incumbent is flat on the zigzag X, and the WHY is its efficiency gate with
    # the trend direction agreeing (horizon +1) -- pinned directly.
    assert "X/USDT" not in inc_side
    item_x = incumbent._state["X/USDT"]
    assert incumbent._horizon_agreement(list(item_x.closes)) == 1
    assert incumbent._efficiency(list(item_x.closes)) < incumbent.min_efficiency
    assert incumbent._desired_signal(item_x) == 0

    # Candidate ranks X top-long and emits LONG X (divergent action on X) while
    # matching the incumbent's SHORT on Y.
    assert cand_side.get("X/USDT") == "LONG", cand_side
    assert cand_side.get("Y/USDT") == "SHORT", cand_side


# --------------------------------------------------------------------------- #
# BUILD GATE -- TEST 2: vs TrendEfficiencyMomentumStrategy (XS ranking inversion)
# --------------------------------------------------------------------------- #


def test_gate2_vs_trend_efficiency_momentum() -> None:
    universe = {"X/USDT": _x_zigzag(), "Z/USDT": _z_flat_then_jump(), **_fillers()}

    incumbent = TrendEfficiencyMomentumStrategy(
        _Bars(list(universe)), _Queue(), rebalance_bars=1, min_symbols=4
    )
    _feed(incumbent, universe)
    inc_side = _final_side(incumbent.events.items)

    candidate = CrossSectionalRegressionTrendQualityStrategy(_Bars(list(universe)), _Queue())
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)

    # Incumbent's top long is the flat-then-jump Z (KER saturates) -- live.
    assert inc_side.get("Z/USDT") == "LONG", inc_side
    # Candidate's top long is the smooth zigzag X, and Z is EXCLUDED (fit quality
    # below floor) -- opposite top-of-book on identical input.
    assert cand_side.get("X/USDT") == "LONG", cand_side
    assert "Z/USDT" not in cand_side, cand_side


# --------------------------------------------------------------------------- #
# min-hold rescue (with a load-bearing contrast)
# --------------------------------------------------------------------------- #


def _min_hold_universe() -> tuple[dict[str, list[float]], int]:
    window = 12
    n = window + 1
    # P: smooth uptrend (top signed-R^2) then a crash on the last bar so its
    # trailing window flips to a bottom/low-quality reading.
    logs = [0.03 * t + 0.006 * ((-1) ** t) for t in range(window)]
    logs.append(logs[-1] - 0.9)
    p = [math.exp(v) for v in logs]
    universe = {"P/USDT": p, **{f"F{i}/USDT": _chop(i + 1, n) for i in range(4)}}
    return universe, window


def _run_min_hold(min_hold: int) -> tuple[str, list[str]]:
    universe, window = _min_hold_universe()
    strategy = CrossSectionalRegressionTrendQualityStrategy(
        _Bars(list(universe)),
        _Queue(),
        formation_bars=window,
        min_symbols=5,
        min_hold_weeks=min_hold,
        vol_window=6,
        quantile_entry_pct=0.20,
        quantile_exit_pct=0.40,
        min_quality=0.30,
    )
    # Weekly-spaced bars: every bar is a distinct ISO week -> a decision each bar.
    _feed(strategy, universe, step_days=7, start=datetime.date(2025, 1, 6))
    p_signals = [sig.signal_type for sig in strategy.events.items if sig.symbol == "P/USDT"]
    return strategy._state["P/USDT"].mode, p_signals


def test_min_hold_suppresses_flip_inside_window() -> None:
    # min_hold=4: the rank flip arrives while held (bars_held < 4) -> suppressed.
    mode_hold, sig_hold = _run_min_hold(4)
    assert mode_hold == "LONG"
    assert sig_hold == ["LONG"]  # entered once, never exited/flipped
    # Contrast (anti-vacuous): with min_hold=1 the guard is off and P exits.
    mode_free, sig_free = _run_min_hold(1)
    assert mode_free == "OUT"
    assert "EXIT" in sig_free


# --------------------------------------------------------------------------- #
# score_mode == "t_stat" still trades the same-signed book
# --------------------------------------------------------------------------- #


def test_t_stat_mode_trades() -> None:
    universe = {"X/USDT": _x_zigzag(), "Y/USDT": _y_smooth_down(), **_fillers()}
    candidate = CrossSectionalRegressionTrendQualityStrategy(
        _Bars(list(universe)), _Queue(), score_mode="t_stat"
    )
    _feed(candidate, universe)
    side = _final_side(candidate.events.items)
    assert side.get("X/USDT") == "LONG"
    assert side.get("Y/USDT") == "SHORT"


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical() -> None:
    universe = {"X/USDT": _x_zigzag(), "Y/USDT": _y_smooth_down(), **_fillers()}

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strategy = CrossSectionalRegressionTrendQualityStrategy(_Bars(list(universe)), _Queue())
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
    universe = {"X/USDT": _x_zigzag(), "Y/USDT": _y_smooth_down(), **_fillers()}
    strategy = CrossSectionalRegressionTrendQualityStrategy(_Bars(list(universe)), _Queue())
    _feed(strategy, universe)
    snapshot = strategy.get_state()

    restored = CrossSectionalRegressionTrendQualityStrategy(_Bars(list(universe)), _Queue())
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    for symbol in universe:
        assert list(restored._state[symbol].closes) == list(strategy._state[symbol].closes)
        assert restored._state[symbol].mode == strategy._state[symbol].mode
        assert restored._state[symbol].bars_held == strategy._state[symbol].bars_held


def test_adversarial_set_state_never_raises() -> None:
    symbols = [f"S{i}/USDT" for i in range(5)]
    strategy = CrossSectionalRegressionTrendQualityStrategy(_Bars(symbols), _Queue())
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "nope"})
    strategy.set_state({"symbol_state": {"S0/USDT": "not a dict"}})
    strategy.set_state({"symbol_state": {"S0/USDT": {"closes": 999}}})
    strategy.set_state(
        {
            "last_eval_week": None,
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), float("inf"), 12.5, None],
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


def test_degenerate_inputs_never_raise() -> None:
    strategy = CrossSectionalRegressionTrendQualityStrategy(_Bars(["Z/USDT"]), _Queue())
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
    universe = {"X/USDT": _x_zigzag(), "Y/USDT": _y_smooth_down(), "F0/USDT": _chop(1)}
    strategy = CrossSectionalRegressionTrendQualityStrategy(
        _Bars(list(universe)), _Queue(), min_symbols=5
    )
    _feed(strategy, universe)
    assert _non_exit(strategy.events.items) == []


def test_self_skip_history_too_short() -> None:
    short = {f"F{i}/USDT": _chop(i + 1, 20) for i in range(5)}
    strategy = CrossSectionalRegressionTrendQualityStrategy(
        _Bars(list(short)), _Queue(), formation_bars=56
    )
    _feed(strategy, short)
    assert _non_exit(strategy.events.items) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalRegressionTrendQualityStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "formation_bars",
        "min_quality",
        "score_mode",
        "quantile_entry_pct",
        "quantile_exit_pct",
        "min_hold_weeks",
        "min_symbols",
        "allow_short",
        "target_gross_exposure",
    ):
        assert required in schema


def test_slice_timeframe_expansion_scales_bar_windows() -> None:
    """4h/1h cells mirror the 1d variants: the formation / vol bar windows scale
    x6/x24 (formation capped at the schema high 4096), while the weekly min-hold,
    quantile bands, score mode, and quality floor stay timeframe-invariant."""
    slice_ = _TREND_QUALITY_XS_SLICE
    assert set(slice_) == {"4h", "1h", "1d"}
    variants = {tf: tuple(cell["variant"] for cell in cells) for tf, cells in slice_.items()}
    assert variants["4h"] == variants["1h"] == variants["1d"]
    by = {tf: {cell["variant"]: cell for cell in cells} for tf, cells in slice_.items()}
    for variant in variants["1d"]:
        d, h4, h1 = by["1d"][variant], by["4h"][variant], by["1h"][variant]
        # OLS formation window scales x6/x24 but is pinned to the schema high (4096).
        assert h4["formation_bars"] == min(d["formation_bars"] * 6, 4096), variant
        assert h1["formation_bars"] == min(d["formation_bars"] * 24, 4096), variant
        assert h4["vol_window"] == d["vol_window"] * 6, variant
        assert h1["vol_window"] == d["vol_window"] * 24, variant
        for key in (
            "min_quality",
            "score_mode",
            "quantile_entry_pct",
            "quantile_exit_pct",
            "min_hold_weeks",
            "target_gross_exposure",
        ):
            assert h4[key] == d[key] == h1[key], (variant, key)
