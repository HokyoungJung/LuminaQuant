"""Deterministic build-gate + hygiene tests for the information-discreteness sleeve.

Direct class import only (no ``@register`` on this lane, so no registry / tier /
candidate-wiring assertions here -- those land with the integration wave).

The BUILD GATE proves the magnitude-BLIND sign-census conditioning is
behaviorally distinct from the momentum / efficiency / fit incumbents by RUNNING
the real incumbents on hand-built fixtures whose primitive statistics are pinned
in stage-1:

- JUMPY: flat drift + a single large jump 18 bars back -- the momentum /
  efficiency / signed-R^2 incumbents all treat it as a clean strong trend and
  LONG it, while the sign census reads it as DISCRETE (ID ~ +0.96) and EXCLUDES
  it from both books.
- SMOOTH: a clean continuous uptrend (ID = -1) -- longed by both.
- CONT: a zigzag-but-net-directional CONTINUOUS winner (ID ~ -0.61) that the
  efficiency incumbents keep flat / exclude (KER collapses) while the candidate
  LONGS it; its mirror CONT_LOSER is SHORTED.

Gating legs (KEEP/DROP): LEG 1 (LowTurnoverTrendPersistence), LEG 2
(TrendEfficiencyMomentum), LEG 4 (CrossSectionalRegressionTrendQuality).  LEG 3
(SelectionGatedMomentum) and LEG 5 (short leg) are supporting.

All pseudo-randomness is a small seeded LCG (no ``random`` module); every run is
bit-for-bit reproducible.  The candidate is driven with test-scaled
``continuity_pct`` / ``quantile_pct`` so the intended book forms on the small
fixture (the data-PC owns the pre-registered param sweep).
"""

from __future__ import annotations

import datetime
import math
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.alpha_features import (
    realized_volatility,
    simple_return,
    volatility_ratio,
)
from lumina_quant.indicators.information_discreteness import information_discreteness
from lumina_quant.indicators.momentum import kaufman_efficiency_ratio
from lumina_quant.indicators.rolling_stats import ts_regression_rsquared
from lumina_quant.indicators.trend import average_directional_index
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    TrendEfficiencyMomentumStrategy,
)
from lumina_quant.strategies.information_discreteness_alpha_sleeves import (
    InformationDiscretenessMomentumStrategy,
)
from lumina_quant.strategies.low_turnover_trend_alpha_sleeves import (
    LowTurnoverTrendPersistenceStrategy,
)
from lumina_quant.strategies.selection_aware_alpha_sleeves import (
    SelectionGatedMomentumStrategy,
)
from lumina_quant.strategies.trend_quality_xs_alpha_sleeves import (
    CrossSectionalRegressionTrendQualityStrategy,
)
from lumina_quant.tuning import HyperParam

_T = 120
_FORMATION = 56
_SKIP = 7


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


def _feed(strategy: Any, series: dict[str, list[float]], *, start_week: int = 0) -> None:
    n = len(next(iter(series.values())))
    base = datetime.date(2025, 1, 6)
    for idx in range(n):
        stamp = (base + datetime.timedelta(weeks=start_week + idx)).isoformat() + "T00:00:00Z"
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


def _compound(returns: list[float], p0: float = 100.0) -> list[float]:
    path = [p0]
    for value in returns:
        path.append(path[-1] * (1.0 + value))
    return path[1:]


def _jumpy(*, jump_back: int = 18, drift: float = -0.0005, jump: float = 0.18) -> list[float]:
    """Flat drift punctuated by a single large jump ``jump_back`` bars from the end.

    Discrete-information winner: KER / ADX saturate and signed-R^2 is moderate,
    so the momentum / efficiency / fit incumbents treat it as a clean trend --
    while the sign census reads ID ~ +0.96 (discrete) and excludes it.
    """
    returns = [drift] * _T
    returns[_T - jump_back] = jump
    return _compound(returns)


def _smooth() -> list[float]:
    """All-positive days (ID = -1, KER = 1, R^2 = 1) with nonzero vol."""
    return _compound([(0.0022 if i % 2 == 0 else 0.0027) for i in range(_T)])


def _cont() -> list[float]:
    """Zigzag-but-net-directional CONTINUOUS winner (ID ~ -0.61, KER collapses)."""
    block = [0.012, 0.012, 0.012, 0.012, -0.04]
    return _compound([block[i % 5] for i in range(_T)])


def _cont_loser() -> list[float]:
    """Mirror of the continuous winner (ID ~ -0.61, negative PRET)."""
    block = [-0.012, -0.012, -0.012, -0.012, 0.04]
    return _compound([block[i % 5] for i in range(_T)])


def _chop(seed: int, drift: float, n: int = _T) -> list[float]:
    """Choppy filler: low fit quality (R^2 < 0.5), high vol, ID ~ 0."""
    state = (seed * 2654435761) & 0xFFFFFFFF
    out: list[float] = []
    logp = math.log(100.0)
    for _ in range(n):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        u = state / float(0x7FFFFFFF) - 0.5
        logp += drift + 0.045 * ((-1) ** len(out)) + 0.010 * u
        out.append(math.exp(logp))
    return out


_FILLER_DRIFTS = (0.0006, 0.0009, 0.0007, 0.0011, -0.0003, -0.0006)


def _universe() -> dict[str, list[float]]:
    universe = {
        "JUMPY/USDT": _jumpy(),
        "SMOOTH/USDT": _smooth(),
        "CONT/USDT": _cont(),
        "CONT_LOSER/USDT": _cont_loser(),
    }
    for idx, drift in enumerate(_FILLER_DRIFTS):
        universe[f"F{idx}/USDT"] = _chop(101 + idx, drift)
    return universe


def _candidate(symbols: list[str], **overrides: Any) -> InformationDiscretenessMomentumStrategy:
    kwargs: dict[str, Any] = dict(
        formation_bars=_FORMATION,
        skip_bars=_SKIP,
        min_history_bars=70,
        min_symbols=6,
        continuity_pct=0.80,
        id_max=0.0,
        quantile_pct=0.34,
        vol_window=30,
        min_hold_decisions=1,
        cooldown_decisions=0,
    )
    kwargs.update(overrides)
    return InformationDiscretenessMomentumStrategy(_Bars(symbols), _Queue(), **kwargs)


def _r2_logclose(closes: list[float], window: int) -> float:
    tail = closes[-window:]
    xs = list(range(len(tail)))
    ys = [math.log(value) for value in tail]
    return ts_regression_rsquared(xs, ys)


# --------------------------------------------------------------------------- #
# LEG 0 -- stage-1 premises (real primitives, before any strategy runs)
# --------------------------------------------------------------------------- #


def test_stage1_premises() -> None:
    jumpy, smooth, cont, cont_loser = _jumpy(), _smooth(), _cont(), _cont_loser()

    # JUMPY: efficiency / trend gates saturate; sign census reads discrete.
    assert abs(kaufman_efficiency_ratio(jumpy, period=20) - 0.885) < 0.02
    adx_j, _, _ = average_directional_index(jumpy, jumpy, jumpy, period=14)
    assert adx_j > 20.0
    assert volatility_ratio(jumpy, fast_window=16, slow_window=64) < 1.5
    assert abs(information_discreteness(jumpy, formation_bars=56, skip_bars=7) - 0.964) < 0.01
    assert 0.5 < _r2_logclose(jumpy, 56) < 0.7
    for lookback in (28, 56, 84):
        assert simple_return(jumpy, lookback=lookback) > 0.0

    # SMOOTH: clean continuous uptrend, ID = -1 exactly.
    assert abs(kaufman_efficiency_ratio(smooth, period=20) - 1.0) < 1e-6
    assert abs(information_discreteness(smooth, formation_bars=56, skip_bars=7) + 1.0) < 1e-9
    assert _r2_logclose(smooth, 56) > 0.99

    # CONT: continuous winner, KER collapses (choppy) but ID stays negative.
    assert kaufman_efficiency_ratio(cont, period=20) < 0.30
    assert abs(information_discreteness(cont, formation_bars=56, skip_bars=7) + 0.607) < 0.01
    adx_c, _, _ = average_directional_index(cont, cont, cont, period=14)
    assert adx_c < 20.0
    assert _r2_logclose(cont, 56) >= 0.30  # tradeable in the fit incumbent

    # CONT_LOSER: mirror sign census, negative PRET.
    assert abs(information_discreteness(cont_loser, formation_bars=56, skip_bars=7) + 0.607) < 0.01


# --------------------------------------------------------------------------- #
# LEG 1 -- vs LowTurnoverTrendPersistence (GATING)
# --------------------------------------------------------------------------- #


def test_gate_leg1_vs_low_turnover_trend_persistence() -> None:
    universe = _universe()
    incumbent = LowTurnoverTrendPersistenceStrategy(_Bars(list(universe)), _Queue())
    _feed(incumbent, universe)
    inc_side = _final_side(incumbent.events.items)

    # Incumbent-LIVE control: it ENTERS LONG the discrete JUMPY with each of its
    # four gates asserted passing at source values.
    assert inc_side.get("JUMPY/USDT") == "LONG", inc_side
    item_j = incumbent._state["JUMPY/USDT"]
    assert incumbent._horizon_agreement(list(item_j.closes)) == 1
    assert incumbent._efficiency(list(item_j.closes)) >= incumbent.min_efficiency
    adx_j, _, _ = average_directional_index(
        list(item_j.highs), list(item_j.lows), list(item_j.closes), period=incumbent.adx_period
    )
    assert adx_j >= incumbent.adx_min
    assert incumbent._desired_signal(item_j) == 1
    # On CONT the incumbent is flat, and the WHY is its efficiency + ADX gates.
    assert "CONT/USDT" not in inc_side
    item_c = incumbent._state["CONT/USDT"]
    assert incumbent._efficiency(list(item_c.closes)) < incumbent.min_efficiency
    adx_c, _, _ = average_directional_index(
        list(item_c.highs), list(item_c.lows), list(item_c.closes), period=incumbent.adx_period
    )
    assert adx_c < incumbent.adx_min
    assert incumbent._desired_signal(item_c) == 0

    candidate = _candidate(list(universe))
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)
    # Divergent action BOTH directions: candidate EXCLUDES JUMPY, LONGS CONT.
    assert "JUMPY/USDT" not in cand_side, cand_side
    assert cand_side.get("CONT/USDT") == "LONG", cand_side


# --------------------------------------------------------------------------- #
# LEG 2 -- vs TrendEfficiencyMomentum (GATING)
# --------------------------------------------------------------------------- #


def test_gate_leg2_vs_trend_efficiency_momentum() -> None:
    universe = _universe()
    incumbent = TrendEfficiencyMomentumStrategy(
        _Bars(list(universe)), _Queue(), rebalance_bars=1, min_symbols=4
    )
    _feed(incumbent, universe)
    inc_side = _final_side(incumbent.events.items)
    # Incumbent's long book CONTAINS the discrete JUMPY (KER * +trend) and
    # EXCLUDES CONT (KER below its long threshold).
    assert inc_side.get("JUMPY/USDT") == "LONG", inc_side
    assert "CONT/USDT" not in inc_side, inc_side

    candidate = _candidate(list(universe))
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)
    # Opposite membership: candidate longs CONT and SMOOTH and excludes JUMPY.
    assert cand_side.get("CONT/USDT") == "LONG", cand_side
    assert cand_side.get("SMOOTH/USDT") == "LONG", cand_side
    assert "JUMPY/USDT" not in cand_side, cand_side


# --------------------------------------------------------------------------- #
# LEG 3 -- vs SelectionGatedMomentum (supporting)
# --------------------------------------------------------------------------- #


def test_gate_leg3_vs_selection_gated_momentum() -> None:
    universe = _universe()
    incumbent = SelectionGatedMomentumStrategy(
        _Bars(list(universe)),
        _Queue(),
        rebalance_bars=1,
        min_symbols=4,
        pool_size=20,
        allow_short=True,
    )
    _feed(incumbent, universe)
    inc_side = _final_side(incumbent.events.items)
    # Magnitude-driven mom/vol puts the discrete JUMPY in its long book.
    assert inc_side.get("JUMPY/USDT") == "LONG", inc_side

    candidate = _candidate(list(universe))
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)
    assert "JUMPY/USDT" not in cand_side, cand_side
    assert cand_side.get("CONT/USDT") == "LONG", cand_side


# --------------------------------------------------------------------------- #
# LEG 4 -- vs CrossSectionalRegressionTrendQuality (GATING; DGW sibling)
# --------------------------------------------------------------------------- #


def test_gate_leg4_vs_trend_quality() -> None:
    universe = _universe()
    incumbent = CrossSectionalRegressionTrendQualityStrategy(
        _Bars(list(universe)),
        _Queue(),
        formation_bars=56,
        min_symbols=5,
        min_hold_weeks=1,
        quantile_entry_pct=0.34,
        quantile_exit_pct=0.5,
    )
    _feed(incumbent, universe)
    inc_side = _final_side(incumbent.events.items)
    # The signed-R^2 sibling longs BOTH SMOOTH and JUMPY (fit quality above floor,
    # positive slope) -- the same-pair split proving the sign census is not
    # signed-R^2.
    assert inc_side.get("SMOOTH/USDT") == "LONG", inc_side
    assert inc_side.get("JUMPY/USDT") == "LONG", inc_side

    candidate = _candidate(list(universe))
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)
    # Candidate longs SMOOTH and EXCLUDES JUMPY (the split).
    assert cand_side.get("SMOOTH/USDT") == "LONG", cand_side
    assert "JUMPY/USDT" not in cand_side, cand_side
    # Agreement case honestly disclosed: both long the continuous winner CONT.
    assert inc_side.get("CONT/USDT") == "LONG", inc_side
    assert cand_side.get("CONT/USDT") == "LONG", cand_side


# --------------------------------------------------------------------------- #
# LEG 5 -- both book sides non-empty (short the continuous loser)
# --------------------------------------------------------------------------- #


def test_gate_leg5_shorts_continuous_loser() -> None:
    universe = _universe()
    candidate = _candidate(list(universe))
    _feed(candidate, universe)
    cand_side = _final_side(candidate.events.items)
    assert cand_side.get("CONT_LOSER/USDT") == "SHORT", cand_side
    assert any(mode == "LONG" for mode in cand_side.values()), cand_side


# --------------------------------------------------------------------------- #
# LEG 6 -- min-hold suppresses a flip inside the window (C1 rescue as a test)
# --------------------------------------------------------------------------- #


def _min_hold_universe() -> dict[str, list[float]]:
    n = 110
    switch = 60
    winner = [(0.004 if i < switch else -0.006) for i in range(n)]
    universe = {
        "P/USDT": _compound([winner[i] + (0.0003 if i % 2 else -0.0002) for i in range(n)]),
        "LOSER/USDT": _compound([-0.004 + (0.0004 if i % 2 else -0.0003) for i in range(n)]),
    }
    for idx in range(6):
        universe[f"W{idx}/USDT"] = _compound(
            [(0.0015 + 0.0002 * idx) + (0.0004 if i % 2 else -0.0003) for i in range(n)]
        )
    return universe


def _run_min_hold(min_hold: int, max_hold: int) -> tuple[str, list[str]]:
    universe = _min_hold_universe()
    strategy = _candidate(
        list(universe),
        formation_bars=28,
        skip_bars=7,
        min_history_bars=40,
        continuity_pct=0.9,
        id_max=0.5,
        vol_window=10,
        min_hold_decisions=min_hold,
        max_hold_decisions=max_hold,
    )
    _feed(strategy, universe)
    p_signals = [sig.signal_type for sig in strategy.events.items if sig.symbol == "P/USDT"]
    return strategy._state["P/USDT"].mode, p_signals


def test_min_hold_suppresses_flip_inside_window() -> None:
    mode_hold, sig_hold = _run_min_hold(500, 3000)
    assert mode_hold == "LONG", (mode_hold, sig_hold)
    assert sig_hold == ["LONG"], sig_hold
    mode_free, sig_free = _run_min_hold(1, 52)
    assert mode_free != "LONG", (mode_free, sig_free)
    assert "EXIT" in sig_free, sig_free


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def test_determinism_two_runs_identical() -> None:
    universe = _universe()

    def _run() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        candidate = _candidate(list(universe))
        _feed(candidate, universe)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in candidate.events.items
        ]

    first = _run()
    second = _run()
    assert first == second
    assert first, "expected at least one signal"


def test_state_roundtrip_lossless() -> None:
    universe = _universe()
    candidate = _candidate(list(universe))
    _feed(candidate, universe)
    snapshot = candidate.get_state()

    restored = _candidate(list(universe))
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot
    for symbol in universe:
        assert list(restored._state[symbol].closes) == list(candidate._state[symbol].closes)
        assert restored._state[symbol].mode == candidate._state[symbol].mode
        assert restored._state[symbol].bars_held == candidate._state[symbol].bars_held
        assert restored._state[symbol].bars_since_exit == candidate._state[symbol].bars_since_exit
    assert restored._tick == candidate._tick
    assert restored._last_decision_week == candidate._last_decision_week


def test_adversarial_set_state_never_raises() -> None:
    symbols = [f"S{i}/USDT" for i in range(6)]
    strategy = _candidate(symbols)
    strategy.set_state(None)  # type: ignore[arg-type]
    strategy.set_state("not a dict")  # type: ignore[arg-type]
    strategy.set_state(12345)  # type: ignore[arg-type]
    strategy.set_state({"symbol_state": "nope"})
    strategy.set_state({"symbol_state": {"S0/USDT": "not a dict"}})
    strategy.set_state({"symbol_state": {"S0/USDT": {"closes": 999}}})
    strategy.set_state(
        {
            "last_decision_week": None,
            "tick": "not-an-int",
            "symbol_state": {
                symbol: {
                    "closes": ["x", float("nan"), float("inf"), 12.5, None],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "bars_since_exit": None,
                    "last_bar_key": 123,
                    "score": [1, 2, 3],
                }
                for symbol in symbols
            },
        }
    )
    for item in strategy._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
        assert item.bars_held >= 0
        assert item.bars_since_exit >= 0


def test_degenerate_inputs_never_raise() -> None:
    symbols = [f"S{i}/USDT" for i in range(6)]
    strategy = _candidate(symbols)
    strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t0", bars_1s={}))
    strategy.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strategy.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    base = datetime.date(2025, 1, 6)
    for idx, value in enumerate((0.0, -5.0, float("nan"), float("inf"), 100.0)):
        stamp = (base + datetime.timedelta(weeks=idx)).isoformat() + "T00:00:00Z"
        bars = {
            "S0/USDT": [
                {
                    "time": stamp,
                    "open": value,
                    "high": value,
                    "low": value,
                    "close": value,
                    "volume": 1000.0,
                }
            ]
        }
        strategy.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=stamp, bars_1s=bars))
    assert _non_exit(strategy.events.items) == []


def test_self_skip_below_min_symbols() -> None:
    universe = _universe()
    symbols = list(universe)[:4]
    prices = {symbol: universe[symbol] for symbol in symbols}
    strategy = _candidate(symbols, min_symbols=6)
    _feed(strategy, prices)
    assert _non_exit(strategy.events.items) == []


def test_self_skip_history_too_short() -> None:
    universe = _universe()
    prices = {symbol: values[:40] for symbol, values in universe.items()}
    strategy = _candidate(list(universe))
    _feed(strategy, prices)
    assert _non_exit(strategy.events.items) == []


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = InformationDiscretenessMomentumStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "formation_bars",
        "skip_bars",
        "continuity_pct",
        "id_max",
        "quantile_pct",
        "vol_window",
        "min_symbols",
        "min_hold_decisions",
        "cooldown_decisions",
        "rank_hysteresis_buffer",
        "allow_short",
        "target_gross_exposure",
    ):
        assert required in schema
    for cap in ("zero_eps", "base_allocation", "max_order_value"):
        assert schema[cap].tunable is False
