"""Deterministic build-gate + hygiene tests for the momentum echo-minus-recent
z-spread sleeve (W3-10, ``CrossSectionalIntermediateEchoMomentumStrategy``).

Direct class import only (no ``@register`` on this lane).  The BUILD GATE drives
the REAL momentum-family incumbents through their FULL decision paths at their
HONEST ECHO-EQUIVALENT nearest-neighbour cells (per the recorded refute fix, NOT
a strawman parameterization) and asserts materially different EMITTED actions:

* LEG A -- ``CrossSectionalEquityMomentumStrategy(lookback=35, skip=49)`` is the
  echo-window transform; it provably TIES the EQECHO pair (identical echo paths)
  while the candidate's SECOND z-window separates them (opposite sides).
* LEG B -- ``LongRunOverreactionReversalStrategy`` SHORTS the extreme formation
  winner ECHO_MONSTER while the candidate LONGS it (high echo, flat recent).
* LEG C -- ``TrendGatedResidualMomentumStrategy`` at its defaults LONGS the
  recent-window sprinter the candidate SHORTS; at the echo-equivalent cell
  (formation_weeks=5, skip_weeks=7) its momentum score TIES the EQECHO pair (no
  second-window subtraction in its parameter space) while the candidate separates.
* LEG D -- ``CrossSectionalNearHighAnchoringStrategy`` puts the recovered-then-
  faded ECHO_STALE in its low-nearness SHORT tail while the candidate LONGS it.
* LEG E -- a recent-window perturbation strictly LOWERS the candidate's score
  while leaving the LEG-A incumbent cell's score unchanged (machine-proving the
  second window is load-bearing).

Benchmark BTC is flat (residual == raw for the residual-momentum incumbent).
Panel length makes the LAST bar the first day of a fresh ISO week so the weekly
decision clock fires cleanly on the final completed bar.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.alpha_features import realized_volatility
from lumina_quant.indicators.stationarity import adf_t_statistic
from lumina_quant.strategies.equity_xs_factor_alpha_sleeves import (
    CrossSectionalEquityMomentumStrategy,
)
from lumina_quant.strategies.longrun_overreaction_alpha_sleeves import (
    LongRunOverreactionReversalStrategy,
)
from lumina_quant.strategies.momentum_echo_alpha_sleeves import (
    CrossSectionalIntermediateEchoMomentumStrategy,
    _xs_zscores,
)
from lumina_quant.strategies.near_high_anchoring_alpha_sleeves import (
    CrossSectionalNearHighAnchoringStrategy,
)
from lumina_quant.strategies.residual_momentum_alpha_sleeves import (
    TrendGatedResidualMomentumStrategy,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #

_N = 204  # last index 203 == 29*7 -> Monday -> the weekly clock fires with full data
_BASE = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday
_TS = [(_BASE + timedelta(days=t)).isoformat() for t in range(_N)]
_BENCH = "BTC/USDT"

# candidate offsets (n=204): echo window close[154]/close[119]; recent window
# close[189]/close[161]; reversal band 189..203.
_ECHO_A, _ECHO_B = _N - 85, _N - 50
_REC_A, _REC_B = _N - 43, _N - 15


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def _jit(t: int) -> float:
    # tiny alternating log-return jitter, IDENTICAL across all symbols -> a common
    # multiplicative factor that leaves rankings, z-scores, and the EQECHO ties
    # intact while giving every vol window non-zero variance.
    return 0.0008 * (1.0 if t % 2 == 0 else -1.0)


def _ramp(lr: list[float], a: int, b: int, total_ret: float) -> None:
    if b <= a:
        return
    g = math.log(1.0 + total_ret) / float(b - a)
    for t in range(a, b):
        lr[t] += g


def _closes(segments: list[tuple[int, int, float]]) -> list[float]:
    lr = [0.0] * (_N - 1)
    for a, b, r in segments:
        _ramp(lr, a, b, r)
    for t in range(_N - 1):
        lr[t] += _jit(t)
    out = [100.0]
    for r in lr:
        out.append(out[-1] * math.exp(r))
    return out


def _panel() -> dict[str, list[float]]:
    return {
        _BENCH: _closes([]),
        # +40% echo, then a decline through the recent window (recent < 0)
        "ECHO_STALE": _closes([(_ECHO_A, _ECHO_B, 0.40), (_ECHO_B, _REC_B, -0.35)]),
        # +100% entirely inside the echo window, flat after (recent ~ 0)
        "ECHO_MONSTER": _closes([(_ECHO_A, _ECHO_B, 1.00)]),
        # EQECHO pair: IDENTICAL pre-echo drop + IDENTICAL +40% echo (so the echo-
        # equivalent incumbent cells tie them), mirrored recent windows.  The pre-
        # echo drop keeps their NET long-run formation off the winner tail so the
        # LongRunOverreaction extreme winner is UNAMBIGUOUSLY ECHO_MONSTER.
        "EQECHO_RECUP": _closes(
            [(_N - 99, _ECHO_A, -0.40), (_ECHO_A, _ECHO_B, 0.40), (_REC_A, _REC_B, 0.30)]
        ),
        "EQECHO_RECDN": _closes(
            [(_N - 99, _ECHO_A, -0.40), (_ECHO_A, _ECHO_B, 0.40), (_REC_A, _REC_B, -0.30)]
        ),
        # flat echo, +60% recent
        "RECENT_SPRINTER": _closes([(_REC_A, _REC_B, 0.60)]),
        # -35% inside the echo window
        "LOSER_STALE": _closes([(_ECHO_A, _ECHO_B, -0.35)]),
        "FILL1": _closes([(20, 180, 0.06)]),
        "FILL2": _closes([(20, 180, -0.05)]),
        "FILL3": _closes([(60, 140, 0.03)]),
    }


_SYMS = list(_panel())


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _window_event(panel: dict[str, list[float]], t: int, symbols: list[str]) -> SimpleNamespace:
    bars_1s = {}
    for sym in symbols:
        close = panel[sym][t]
        bars_1s[sym] = [
            {
                "time": _TS[t],
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
            }
        ]
    return SimpleNamespace(type="MARKET_WINDOW", time=_TS[t], bars_1s=bars_1s)


def _feed(strategy: Any, panel: dict[str, list[float]], symbols: list[str], *, n: int = _N) -> None:
    for t in range(n):
        strategy.calculate_signals(_window_event(panel, t, symbols))


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def _candidate(
    symbols: list[str], **overrides: Any
) -> CrossSectionalIntermediateEchoMomentumStrategy:
    kwargs: dict[str, Any] = dict(
        echo_start_weeks=12,
        echo_end_weeks=7,
        recent_start_weeks=6,
        recent_end_weeks=2,
        quantile_entry_pct=0.30,
        quantile_exit_pct=0.45,
        min_hold_decisions=4,
        cooldown_decisions=1,
        min_symbols=5,
        min_history_bars=_N,
        allow_short=True,
        target_gross_exposure=1.0,
        min_price=0.01,
    )
    kwargs.update(overrides)
    return CrossSectionalIntermediateEchoMomentumStrategy(_Bars(symbols), _Queue(), **kwargs)


def _eqmom_echo_cell(symbols: list[str]) -> CrossSectionalEquityMomentumStrategy:
    # the honest ECHO-EQUIVALENT nearest neighbour: mom == close[-50]/close[-85].
    return CrossSectionalEquityMomentumStrategy(
        _Bars(symbols),
        _Queue(),
        lookback_bars=35,
        skip_bars=49,
        rebalance_bars=7,
        vol_window=10,
        regime_sma_bars=8,
        quintile_pct=0.25,
        min_symbols=4,
        allow_short=True,
        stop_loss_pct=0.0,
        max_hold_bars=100000,
        min_price=0.01,
    )


def _eqmom_score(eqmom: CrossSectionalEquityMomentumStrategy, closes: list[float]) -> float:
    """Reproduce the incumbent's exact per-symbol score (mom / realized-vol)."""
    recent = closes[-(eqmom.skip_bars + 1)]
    base = closes[-(eqmom.lookback_bars + eqmom.skip_bars + 1)]
    vol = realized_volatility(closes, window=eqmom.vol_window)
    return (recent / base - 1.0) / max(vol or 0.0, 1e-12)


def _candidate_spreads(panel: dict[str, list[float]]) -> dict[str, float]:
    cand = _candidate(list(panel))
    raw = {s: cand.raw_windows(panel[s]) for s in panel}
    raw = {s: w for s, w in raw.items() if w is not None}
    z_echo = _xs_zscores({s: w.r_echo for s, w in raw.items()})
    z_recent = _xs_zscores({s: w.r_recent for s, w in raw.items()})
    return {s: z_echo[s] - z_recent[s] for s in raw}


# --------------------------------------------------------------------------- #
# (0) helper + stage-1 premises
# --------------------------------------------------------------------------- #


def test_xs_zscore_helper() -> None:
    z = _xs_zscores({"a": 0.4, "b": 0.0, "c": -0.4})
    assert abs(z["a"] - 1.0) < 1e-12 and abs(z["c"] + 1.0) < 1e-12
    assert _xs_zscores({"a": 5.0, "b": 5.0}) == {"a": 0.0, "b": 0.0}  # degenerate -> 0


def test_stage1_window_premises() -> None:
    panel = _panel()
    cand = _candidate(_SYMS)
    w_stale = cand.raw_windows(panel["ECHO_STALE"])
    w_monster = cand.raw_windows(panel["ECHO_MONSTER"])
    w_up = cand.raw_windows(panel["EQECHO_RECUP"])
    w_dn = cand.raw_windows(panel["EQECHO_RECDN"])
    w_sprint = cand.raw_windows(panel["RECENT_SPRINTER"])
    # hand-computed echo / recent returns
    assert abs(w_stale.r_echo - 0.40) < 5e-3 and w_stale.r_recent < -0.25
    assert abs(w_monster.r_echo - 1.00) < 5e-3 and abs(w_monster.r_recent) < 5e-3
    assert abs(w_sprint.r_echo) < 5e-3 and abs(w_sprint.r_recent - 0.60) < 5e-3
    # the EQECHO pair's echo returns are equal (identical echo paths)
    assert abs(w_up.r_echo - w_dn.r_echo) < 1e-12
    assert w_up.r_recent > 0.25 and w_dn.r_recent < -0.25
    # RECENT_SPRINTER's (flat-bench) residual level is non-stationary -> passes
    # the residual-momentum ANTI-gate (so LEG C's incumbent can trade it)
    levels = [math.log(c) for c in panel["RECENT_SPRINTER"][-90:]]
    assert adf_t_statistic(levels, lags=0) > -1.5


# --------------------------------------------------------------------------- #
# (LEG A) echo-equivalent CrossSectionalEquityMomentum ties the EQECHO pair
# --------------------------------------------------------------------------- #


def test_leg_a_echo_equivalent_ties_pair_candidate_separates() -> None:
    panel = _panel()
    eqmom = _eqmom_echo_cell(_SYMS)
    # the incumbent's window sees ONLY the identical echo paths -> tied scores
    s_up = _eqmom_score(eqmom, panel["EQECHO_RECUP"])
    s_dn = _eqmom_score(eqmom, panel["EQECHO_RECDN"])
    assert abs(s_up - s_dn) < 1e-9
    _feed(eqmom, panel, _SYMS)
    eqmom_book = _final_side(eqmom.events.items)
    assert eqmom_book.get("ECHO_MONSTER") == "LONG"  # LIVE control
    # the candidate's SECOND z-window separates the pair the incumbent ties
    cand = _candidate(_SYMS)
    _feed(cand, panel, _SYMS)
    book = _final_side(cand.events.items)
    assert book.get("EQECHO_RECDN") == "LONG"
    assert book.get("EQECHO_RECUP") == "SHORT"
    spreads = _candidate_spreads(panel)
    assert spreads["EQECHO_RECDN"] - spreads["EQECHO_RECUP"] > 1.0  # materially different


# --------------------------------------------------------------------------- #
# (LEG B) LongRunOverreaction shorts the extreme winner the candidate longs
# --------------------------------------------------------------------------- #


def test_leg_b_longrun_shorts_monster_candidate_longs() -> None:
    panel = _panel()
    lro = LongRunOverreactionReversalStrategy(
        _Bars(_SYMS),
        _Queue(),
        formation_bars=84,
        skip_bars=14,
        z_min=1.0,
        rebalance_bars=1,
        quantile_pct=0.50,
        min_symbols=4,
        allow_short=True,
        min_hold_bars=0,
        min_price=0.01,
    )
    _feed(lro, panel, _SYMS)
    assert _final_side(lro.events.items).get("ECHO_MONSTER") == "SHORT"
    cand = _candidate(_SYMS)
    _feed(cand, panel, _SYMS)
    assert _final_side(cand.events.items).get("ECHO_MONSTER") == "LONG"


# --------------------------------------------------------------------------- #
# (LEG C) TrendGatedResidualMomentum divergence at BOTH cells
# --------------------------------------------------------------------------- #


def test_leg_c_residual_momentum_defaults_and_echo_equivalent() -> None:
    panel = _panel()
    tgrm = TrendGatedResidualMomentumStrategy(
        _Bars(_SYMS),
        _Queue(),
        benchmark_symbol=_BENCH,
        formation_weeks=8,
        skip_weeks=1,
        bars_per_week=7,
        adf_nonstationarity_min_t=-1.5,
        quantile_pct=0.50,
        min_symbols=3,
        min_hold_decisions=0,
        cooldown_decisions=0,
        min_history_bars=120,
        allow_short=True,
    )
    _feed(tgrm, panel, _SYMS)
    assert _final_side(tgrm.events.items).get("RECENT_SPRINTER") == "LONG"
    cand = _candidate(_SYMS)
    _feed(cand, panel, _SYMS)
    assert _final_side(cand.events.items).get("RECENT_SPRINTER") == "SHORT"
    # echo-equivalent cell: its momentum SCORE cannot subtract a second window, so
    # it ties the EQECHO pair (ADF admission gate disabled to isolate the score).
    tgrm_echo = TrendGatedResidualMomentumStrategy(
        _Bars(_SYMS),
        _Queue(),
        benchmark_symbol=_BENCH,
        formation_weeks=5,
        skip_weeks=7,
        bars_per_week=7,
        adf_nonstationarity_min_t=-1000.0,
        quantile_pct=0.25,
        min_symbols=4,
        min_hold_decisions=0,
        cooldown_decisions=0,
        min_history_bars=120,
        allow_short=True,
    )
    s_up = tgrm_echo._asset_score(panel["EQECHO_RECUP"], panel[_BENCH])
    s_dn = tgrm_echo._asset_score(panel["EQECHO_RECDN"], panel[_BENCH])
    assert s_up is not None and s_dn is not None
    assert abs(s_up.score - s_dn.score) < 1e-6


# --------------------------------------------------------------------------- #
# (LEG D) NearHighAnchoring shorts the faded stale name the candidate longs
# --------------------------------------------------------------------------- #


def test_leg_d_near_high_shorts_echo_stale_candidate_longs() -> None:
    panel = _panel()
    nha = CrossSectionalNearHighAnchoringStrategy(
        _Bars(_SYMS),
        _Queue(),
        high_lookback_bars=100,
        min_history_bars=60,
        vol_window=20,
        rebalance_bars=1,
        quantile_pct=0.30,
        min_hold_bars=0,
        min_symbols=4,
        allow_short=True,
        min_price=0.01,
    )
    _feed(nha, panel, _SYMS)
    assert _final_side(nha.events.items).get("ECHO_STALE") == "SHORT"
    cand = _candidate(_SYMS)
    _feed(cand, panel, _SYMS)
    assert _final_side(cand.events.items).get("ECHO_STALE") == "LONG"


# --------------------------------------------------------------------------- #
# (LEG E) sign-sensitivity: the second window is load-bearing
# --------------------------------------------------------------------------- #


def test_leg_e_recent_window_is_load_bearing() -> None:
    panel = _panel()
    base_spreads = _candidate_spreads(panel)
    # perturb ONLY the recent-window closes of EQECHO_RECDN upward (+10%)
    up = dict(panel)
    recdn_up = list(panel["EQECHO_RECDN"])
    for t in range(_REC_B, _N):
        recdn_up[t] *= 1.10
    up["EQECHO_RECDN"] = recdn_up
    up_spreads = _candidate_spreads(up)
    # the candidate's score STRICTLY decreases (monotone in -z_recent)
    assert up_spreads["EQECHO_RECDN"] < base_spreads["EQECHO_RECDN"] - 1e-6
    # the LEG-A incumbent cell's score is unchanged (recent bars outside its window)
    eqmom = _eqmom_echo_cell(_SYMS)
    assert abs(_eqmom_score(eqmom, panel["EQECHO_RECDN"]) - _eqmom_score(eqmom, recdn_up)) < 1e-6


# --------------------------------------------------------------------------- #
# (LEG F) hard min-hold + hygiene
# --------------------------------------------------------------------------- #


def test_min_hold_reference_config_trades_at_least_as_often() -> None:
    panel = _panel()
    held = _candidate(_SYMS, min_history_bars=170, min_hold_decisions=4)
    _feed(held, panel, _SYMS)
    ref = _candidate(_SYMS, min_history_bars=170, min_hold_decisions=0, cooldown_decisions=0)
    _feed(ref, panel, _SYMS)
    held_changes = [s for s in held.events.items if s.symbol == "ECHO_MONSTER"]
    ref_changes = [s for s in ref.events.items if s.symbol == "ECHO_MONSTER"]
    assert len(ref_changes) >= len(held_changes)


def test_run_twice_bit_identical() -> None:
    panel = _panel()
    a = _candidate(_SYMS)
    _feed(a, panel, _SYMS)
    b = _candidate(_SYMS)
    _feed(b, panel, _SYMS)
    sig_a = [(s.symbol, s.signal_type, round(float(s.strength), 12)) for s in a.events.items]
    sig_b = [(s.symbol, s.signal_type, round(float(s.strength), 12)) for s in b.events.items]
    assert sig_a == sig_b


def test_state_roundtrip_mid_position() -> None:
    panel = _panel()
    a = _candidate(_SYMS, min_history_bars=140)
    _feed(a, panel, _SYMS, n=175)
    snap = a.get_state()
    b = _candidate(_SYMS, min_history_bars=140)
    b.set_state(snap)
    assert b.get_state() == snap


def test_adversarial_set_state_never_raises() -> None:
    cand = _candidate(_SYMS)
    for garbage in (
        None,
        [],
        "x",
        42,
        {"symbol_state": "nope"},
        {"symbol_state": {"ECHO_MONSTER": None}},
        {"symbol_state": {"ECHO_MONSTER": {"closes": "bad", "bars_held": -3}}},
        {"tick": "NaN", "last_decision_week": 9},
    ):
        cand.set_state(garbage)


def test_degenerate_input_never_raises() -> None:
    syms = [_BENCH, "A", "B", "C", "D", "E"]
    cand = _candidate(syms)
    bad_rows = {
        _BENCH: [{"time": _TS[0], "close": 100.0, "high": 100.0, "low": 100.0, "volume": 0.0}],
        "A": [{"time": _TS[0], "close": None, "high": None, "low": None, "volume": None}],
        "B": [{"time": _TS[0], "close": float("nan"), "high": 1.0, "low": 1.0, "volume": 1.0}],
        "C": [{"time": _TS[0], "close": -5.0, "high": 1.0, "low": 9.0, "volume": 1.0}],
        "D": [],
        "E": [{"time": _TS[0], "close": float("inf"), "high": 1.0, "low": 1.0, "volume": 1.0}],
    }
    cand.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=_TS[0], bars_1s=bad_rows))
    cand.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=_TS[1], bars_1s={}))


def test_sub_min_symbols_self_skip() -> None:
    syms = [_BENCH, "ONLY1", "ONLY2"]  # below min_symbols=5
    panel = {s: _closes([(20, 180, 0.05)]) for s in syms}
    cand = _candidate(syms)
    _feed(cand, panel, syms)
    assert _final_side(cand.events.items) == {}


def test_param_schema_is_snake_case() -> None:
    schema = CrossSectionalIntermediateEchoMomentumStrategy.get_param_schema()
    assert schema
    for name, hp in schema.items():
        assert name == name.lower() and " " not in name
        assert isinstance(hp, HyperParam)
