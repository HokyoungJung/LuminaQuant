"""Deterministic build-gate + hygiene tests for the behavioral-value XS sleeves
(``SalienceTheoryValueStrategy`` + ``ProspectTheoryValueStrategy``).

Data-free, seed-free (fixed alternating jitter, no RNG).  The two pre-registered
mechanical orthogonality locks from the proposals are included:

(a) ST co-timing permutation lock -- moving the market-wide day so it co-times
    with the target's spike day changes the target's ST characteristic while
    its own return path (hence momentum and MAX, the lottery-side controls) is
    bit-identical.
(b) PT equal-skew-equal-MAX different-TK fixture -- two return vectors with
    exactly equal skewness (both symmetric -> 0) and exactly equal MAX get
    different TK values (the loss-aversion kink + probability weighting react
    to mid-distribution loss mass that skew/MAX ignore).

Sign-lock panels: the ST panel puts one IDENTICAL +6% common-factor day in
every symbol (r == m -> zero salience despite the largest magnitude, so every
symbol carries the SAME MAX) and gives the two salience symbols solo +/-5%
spikes on quiet days with a compensating drift that pins their 21d momentum
mid-pack -- the fade must therefore come from the co-timing distortion alone.
The PT panel places jackpots/crashes OUTSIDE the trailing 21d momentum/MAX
regressor windows, so the TK spread is orthogonal to both regressors by
construction.  Panel lengths end on a Monday so the ISO-week decision clock
fires on the final completed bar.
"""

from __future__ import annotations

from copy import deepcopy
import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.behavioral_value import prospect_theory_value
from lumina_quant.strategies.behavioral_value_xs_alpha_sleeves import (
    _PROSPECT_VALUE_SLICE,
    _SALIENCE_VALUE_SLICE,
    _SUGGESTED_CANDIDATE_TAGS,
    ProspectTheoryValueStrategy,
    SalienceTheoryValueStrategy,
    _salience_value,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #

_BASE = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday
_N_ST = 36  # last index 35 == 5*7 -> Monday -> weekly clock fires with full data
_N_PT = 99  # last index 98 == 14*7 -> Monday


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _ts(t: int) -> str:
    return (_BASE + timedelta(days=t)).isoformat()


def _window_event(panel: dict[str, list[float]], t: int) -> SimpleNamespace:
    bars_1s = {}
    for sym, closes in panel.items():
        close = closes[t]
        bars_1s[sym] = [
            {
                "time": _ts(t),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
            }
        ]
    return SimpleNamespace(type="MARKET_WINDOW", time=_ts(t), bars_1s=bars_1s)


def _feed(strategy: Any, panel: dict[str, list[float]], *, n: int) -> None:
    for t in range(n):
        strategy.calculate_signals(_window_event(panel, t))


def _final_side(signals: list[Any]) -> dict[str, str]:
    side: dict[str, str] = {}
    for sig in signals:
        kind = str(sig.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            side[sig.symbol] = kind
        elif kind == "EXIT":
            side.pop(sig.symbol, None)
    return side


def _closes_from_returns(rets: list[float]) -> list[float]:
    out = [100.0]
    for r in rets:
        out.append(out[-1] * (1.0 + r))
    return out


def _jit(t: int, phase: int) -> float:
    return 0.001 * (1.0 if (t + phase) % 2 == 0 else -1.0)


def _skewness(values: list[float]) -> float:
    n = len(values)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    std = var**0.5
    return sum((v - mu) ** 3 for v in values) / n / std**3


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #

_ST_SYMS = ["FILL1", "FILL2", "FILL3", "FILL4", "SAL_UP", "SAL_DOWN", "MOM_UP", "MOM_DN"]
_ST_DRIFT = {"MOM_UP": 0.008, "MOM_DN": -0.008, "SAL_UP": -0.005, "SAL_DOWN": 0.005}
_ST_SPIKES = {
    "SAL_UP": {24: 0.05, 30: 0.05},
    "SAL_DOWN": {25: -0.05, 31: -0.05},
}


def _st_panel(*, common_day: int = 20, frozen: dict[str, list[float]] | None = None):
    """ST panel; ``common_day`` carries an IDENTICAL +6% return for everyone.

    ``frozen`` pins a symbol's close path verbatim (used by the permutation
    lock to keep the target's own returns bit-identical between panels).
    """
    panel: dict[str, list[float]] = {}
    for i, sym in enumerate(_ST_SYMS):
        if frozen is not None and sym in frozen:
            panel[sym] = list(frozen[sym])
            continue
        rets = []
        for t in range(_N_ST - 1):
            if t == common_day:
                rets.append(0.06)
                continue
            r = _jit(t, i) + _ST_DRIFT.get(sym, 0.0)
            r += _ST_SPIKES.get(sym, {}).get(t, 0.0)
            rets.append(r)
        panel[sym] = _closes_from_returns(rets)
    return panel


_PT_SYMS = ["FILL1", "FILL2", "FILL3", "FILL4", "TK_HIGH", "TK_LOW", "MOM_UP", "MOM_DN"]


def _pt_panel() -> dict[str, list[float]]:
    panel: dict[str, list[float]] = {}
    for i, sym in enumerate(_PT_SYMS):
        rets = []
        for t in range(_N_PT - 1):
            r = _jit(t, i)
            if sym == "TK_HIGH":
                # Lottery-like: jackpots OUTSIDE the 21d momentum/MAX windows.
                r = 0.001 + _jit(t, i)
                if t in (20, 40, 60):
                    r = 0.12
            elif sym == "TK_LOW":
                r = -0.001 + _jit(t, i)
                if t in (22, 42, 62):
                    r = -0.12
            elif sym == "MOM_UP":
                r = 0.004 + _jit(t, i)
            elif sym == "MOM_DN":
                r = -0.004 + _jit(t, i)
            elif sym == "FILL3":
                if t == 85:
                    r = 0.06  # MAX control INSIDE the 21d window
            elif sym == "FILL4":
                if t == 87:
                    r = -0.06
            rets.append(r)
        panel[sym] = _closes_from_returns(rets)
    return panel


def _st_candidate(symbols: list[str], **overrides: Any) -> SalienceTheoryValueStrategy:
    kwargs: dict[str, Any] = dict(min_symbols=5, target_gross_exposure=1.0, min_price=0.01)
    kwargs.update(overrides)
    return SalienceTheoryValueStrategy(_Bars(symbols), _Queue(), **kwargs)


def _pt_candidate(symbols: list[str], **overrides: Any) -> ProspectTheoryValueStrategy:
    kwargs: dict[str, Any] = dict(min_symbols=5, target_gross_exposure=1.0, min_price=0.01)
    kwargs.update(overrides)
    return ProspectTheoryValueStrategy(_Bars(symbols), _Queue(), **kwargs)


# --------------------------------------------------------------------------- #
# sign-locks
# --------------------------------------------------------------------------- #


def test_st_sign_lock_fades_salient_cotiming() -> None:
    panel = _st_panel()
    cand = _st_candidate(_ST_SYMS)
    _feed(cand, panel, n=_N_ST)
    book = _final_side(cand.events.items)
    assert book.get("SAL_UP") == "SHORT"  # salient-upside co-timing -> overbought
    assert book.get("SAL_DOWN") == "LONG"
    scored = cand._score_symbols()
    # The fade is decisive: SAL extremes carry the most extreme residual scores.
    ordered = sorted(scored, key=lambda s: scored[s][0])
    assert ordered[0] == "SAL_UP" and ordered[-1] == "SAL_DOWN"
    # Every symbol has the SAME MAX (the identical common-factor day), so the
    # fade cannot be a MAX artifact.
    max_values = {round(scored[s][1]["max_daily_return"], 12) for s in scored}
    assert len(max_values) == 1


def test_st_entry_metadata_stamps_intended_hold() -> None:
    panel = _st_panel()
    cand = _st_candidate(_ST_SYMS)
    _feed(cand, panel, n=_N_ST)
    entries = [s for s in cand.events.items if str(s.signal_type).upper() in {"LONG", "SHORT"}]
    assert entries
    for sig in entries:
        assert sig.metadata.get("intended_hold_seconds") == 14 * 86400  # ~2 weeks
        assert sig.metadata.get("target_allocation", 0.0) > 0.0


def test_st_co_timing_permutation_lock() -> None:
    """Lock (a): permuted market-day alignment moves ST; momentum/MAX frozen."""
    panel_a = _st_panel(common_day=20)
    # Panel B: the common-factor day moves onto SAL_UP's spike day (24) for the
    # REST of the universe; SAL_UP's own close path is pinned verbatim.
    panel_b = _st_panel(common_day=24, frozen={"SAL_UP": panel_a["SAL_UP"]})
    assert panel_b["SAL_UP"] == panel_a["SAL_UP"]

    cand_a = _st_candidate(_ST_SYMS)
    _feed(cand_a, panel_a, n=_N_ST)
    meta_a = cand_a._score_symbols()["SAL_UP"][1]

    cand_b = _st_candidate(_ST_SYMS)
    _feed(cand_b, panel_b, n=_N_ST)
    meta_b = cand_b._score_symbols()["SAL_UP"][1]

    # ST is a co-timing functional: it moves when only the alignment changed...
    assert abs(meta_a["behavioral_value"] - meta_b["behavioral_value"]) > 1e-4
    # ...while the alignment-invariant lottery-side controls are bit-identical.
    assert meta_a["momentum_return"] == meta_b["momentum_return"]
    assert meta_a["max_daily_return"] == meta_b["max_daily_return"]


def test_pt_sign_lock_fades_high_tk() -> None:
    panel = _pt_panel()
    cand = _pt_candidate(_PT_SYMS)
    _feed(cand, panel, n=_N_PT)
    book = _final_side(cand.events.items)
    assert book.get("TK_HIGH") == "SHORT"  # PT-attractive distribution -> overbought
    assert book.get("TK_LOW") == "LONG"
    scored = cand._score_symbols()
    ordered = sorted(scored, key=lambda s: scored[s][0])
    assert ordered[0] == "TK_HIGH" and ordered[-1] == "TK_LOW"
    entries = [s for s in cand.events.items if str(s.signal_type).upper() in {"LONG", "SHORT"}]
    assert entries
    for sig in entries:
        assert sig.metadata.get("intended_hold_seconds") == 21 * 86400  # ~3 weeks


def test_pt_equal_skew_equal_max_different_tk_fixture() -> None:
    """Lock (b): identical skew, identical MAX, different TK (mid-loss mass)."""
    x = ([0.05, -0.05, 0.01, -0.01] * 22) + [0.05, -0.05]  # 90 outcomes
    y = ([0.05, -0.05, 0.03, -0.03] * 22) + [0.05, -0.05]
    # Both symmetric -> skew exactly 0; MAX identical by construction.
    assert _skewness(x) == 0.0 and _skewness(y) == 0.0
    assert max(x) == max(y) == 0.05
    tk_x = prospect_theory_value(x)
    tk_y = prospect_theory_value(y)
    assert tk_x is not None and tk_y is not None
    assert abs(tk_x - tk_y) > 1e-4  # the full-distribution functional separates


def test_salience_uses_the_same_probability_mass_for_distortion_and_value() -> None:
    returns = [0.20, -0.02]
    market = [0.00, 0.00]
    weighted = _salience_value(returns, market, theta=0.1, delta=0.7, probability_mass=[0.9, 0.1])
    uniform = _salience_value(returns, market, theta=0.1, delta=0.7, probability_mass=[0.5, 0.5])
    assert weighted is not None and uniform is not None
    assert weighted != uniform


def test_required_exit_is_not_blocked_by_min_hold_deadband() -> None:
    cand = _st_candidate(_ST_SYMS, min_hold_daily_decisions=5)
    item = cand._state["SAL_UP"]
    item.mode = "LONG"
    item.days_held = 0
    item.pending_close = 100.0
    cand._emit_book({"SAL_UP": "OUT"}, {}, {}, _ts(1))
    assert item.mode == "OUT"
    assert any(
        sig.symbol == "SAL_UP" and str(sig.signal_type).upper() == "EXIT"
        for sig in cand.events.items
    )


# --------------------------------------------------------------------------- #
# hygiene (both classes)
# --------------------------------------------------------------------------- #


def test_zero_weight_no_emit() -> None:
    st = _st_candidate(_ST_SYMS, target_gross_exposure=0.0)
    _feed(st, _st_panel(), n=_N_ST)
    assert st.events.items == []
    pt = _pt_candidate(_PT_SYMS, target_gross_exposure=0.0)
    _feed(pt, _pt_panel(), n=_N_PT)
    assert pt.events.items == []


def test_two_run_determinism() -> None:
    for build, panel, n in (
        (_st_candidate, _st_panel(), _N_ST),
        (_pt_candidate, _pt_panel(), _N_PT),
    ):
        syms = list(panel)
        a = build(syms)
        _feed(a, panel, n=n)
        b = build(syms)
        _feed(b, panel, n=n)
        sig_a = [
            (s.symbol, s.signal_type, float(s.strength), dict(s.metadata)) for s in a.events.items
        ]
        sig_b = [
            (s.symbol, s.signal_type, float(s.strength), dict(s.metadata)) for s in b.events.items
        ]
        assert sig_a == sig_b


def test_state_roundtrip_mid_position() -> None:
    for build, panel, n_partial in (
        (_st_candidate, _st_panel(), 31),
        (_pt_candidate, _pt_panel(), 95),
    ):
        syms = list(panel)
        a = build(syms)
        _feed(a, panel, n=n_partial)
        snap = a.get_state()
        b = build(syms)
        b.set_state(snap)
        assert b.get_state() == snap


def test_state_roundtrip_resumes_identically() -> None:
    panel = _st_panel()
    full = _st_candidate(_ST_SYMS)
    _feed(full, panel, n=_N_ST)
    partial = _st_candidate(_ST_SYMS)
    _feed(partial, panel, n=30)
    resumed = _st_candidate(_ST_SYMS)
    resumed.set_state(partial.get_state())
    for t in range(30, _N_ST):
        resumed.calculate_signals(_window_event(panel, t))
    tail = [(s.symbol, s.signal_type) for s in resumed.events.items]
    full_all = [(s.symbol, s.signal_type) for s in full.events.items]
    assert full_all[-len(tail) :] == tail if tail else True
    assert resumed.get_state() == full.get_state()


def test_only_strictly_newer_complete_windows_advance_xs_state() -> None:
    """Stale, partial, and single-name callbacks cannot advance clocks or data."""
    panel = _st_panel()
    cand = _st_candidate(_ST_SYMS)
    cand.calculate_signals(_window_event(panel, 0))
    cand.calculate_signals(_window_event(panel, 1))
    after_two_windows = cand.get_state()

    for symbol in reversed(_ST_SYMS):
        close = panel[symbol][2]
        cand.calculate_signals(
            SimpleNamespace(type="MARKET", symbol=symbol, time=_ts(2), close=close)
        )
    assert cand.get_state() == after_two_windows

    partial = _window_event(panel, 2)
    partial.bars_1s.pop(_ST_SYMS[-1])
    cand.calculate_signals(partial)
    cand.calculate_signals(_window_event(panel, 0))
    assert cand.get_state() == after_two_windows

    cand.calculate_signals(_window_event(panel, 2))
    after_newer_window = cand.get_state()
    cand.calculate_signals(_window_event(panel, 1))
    partial_newer = _window_event(panel, 3)
    partial_newer.bars_1s.pop(_ST_SYMS[0])
    cand.calculate_signals(partial_newer)
    assert cand.get_state() == after_newer_window


def test_single_name_order_permutations_do_not_affect_window_continuation() -> None:
    """Only the aligned window, not callback order, determines XS state."""
    panel = _st_panel()
    expected = _st_candidate(_ST_SYMS)
    observed = _st_candidate(_ST_SYMS)
    for t in range(3):
        expected.calculate_signals(_window_event(panel, t))
        for symbol in reversed(_ST_SYMS) if t % 2 else _ST_SYMS:
            observed.calculate_signals(
                SimpleNamespace(
                    type="MARKET",
                    symbol=symbol,
                    time=_ts(t),
                    close=panel[symbol][t],
                )
            )
        event = _window_event(panel, t)
        event.bars_1s = dict(reversed(list(event.bars_1s.items())))
        observed.calculate_signals(event)
    assert observed.get_state() == expected.get_state()


def test_adversarial_set_state_never_raises() -> None:
    for cand in (_st_candidate(_ST_SYMS), _pt_candidate(_PT_SYMS)):
        for garbage in (
            None,
            [],
            "x",
            42,
            {"symbol_state": "nope"},
            {"symbol_state": {"SAL_UP": None, "TK_HIGH": None}},
            {"symbol_state": {"SAL_UP": {"daily_closes": "bad", "days_held": -3}}},
            {"symbol_state": {"TK_HIGH": {"pending_close": "NaN", "mode": 7}}},
            {"tick": "NaN", "last_decision_week": 9, "last_day_key": None},
        ):
            cand.set_state(garbage)


def test_degenerate_input_never_raises() -> None:
    syms = ["A", "B", "C", "D", "E", "F"]
    for cand in (_st_candidate(syms), _pt_candidate(syms)):
        bad_rows = {
            "A": [{"time": _ts(0), "close": None, "high": None, "low": None, "volume": None}],
            "B": [{"time": _ts(0), "close": float("nan"), "high": 1.0, "low": 1.0, "volume": 1.0}],
            "C": [{"time": _ts(0), "close": -5.0, "high": 1.0, "low": 9.0, "volume": 1.0}],
            "D": [],
            "E": [{"time": _ts(0), "close": float("inf"), "high": 1.0, "low": 1.0, "volume": 1.0}],
            "F": [{"time": _ts(0), "close": 100.0, "high": 100.0, "low": 100.0, "volume": 0.0}],
        }
        cand.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=_ts(0), bars_1s=bad_rows))
        cand.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=_ts(1), bars_1s={}))
        cand.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time=None, bars_1s=bad_rows))
        # Single-symbol MARKET path with adversarial payloads.
        cand.calculate_signals(SimpleNamespace(type="MARKET", symbol="A", time=_ts(2), close=None))
        cand.calculate_signals(
            SimpleNamespace(type="MARKET", symbol="F", time=_ts(2), close=101.0, volume=None)
        )
        cand.calculate_signals(SimpleNamespace(type="OTHER"))
        assert cand.events.items == []


def test_interior_utc_day_gap_excludes_symbol_from_panel() -> None:
    """SLEEVE-STATE-05 regression: the formation panel is CALENDAR-aligned.

    A symbol missing ONE interior UTC day (its formation window would be
    shifted vs the market) is excluded from the panel even though it is fresh
    and long enough; the gap-free symbols' ST/TK characteristics are exactly
    the calendar-aligned computation (bit-identical to a run on the gap-free
    universe alone).
    """
    for build, base_panel, n, gap_day in (
        (_st_candidate, _st_panel(), _N_ST, 28),
        (_pt_candidate, _pt_panel(), _N_PT, 50),
    ):
        syms = list(base_panel)
        gappy = "GAPPY"
        panel = dict(base_panel)
        panel[gappy] = _closes_from_returns(
            [0.002 * (1.0 if t % 3 == 0 else -1.0) for t in range(n - 1)]
        )
        cand = build([*syms, gappy])
        _feed(cand, panel, n=n)
        item = cand._state[gappy]
        missing_day = _ts(gap_day)[:10]
        missing_index = list(item.daily_days).index(missing_day)
        del item.daily_days[missing_index]
        del item.daily_closes[missing_index]
        # The gap symbol is fresh and long enough -- ONLY the calendar check
        # can exclude it (the exclusion is not staleness / short history).
        assert item.last_committed_day == cand._last_committed_day
        assert len(item.daily_closes) >= cand._required_days
        formation = cand._formation_panel()
        assert gappy not in formation
        assert set(formation) == set(syms)
        # Gap-free symbols match the calendar-aligned computation exactly.
        ref = build(syms)
        _feed(ref, base_panel, n=n)
        scored_gap = cand._score_symbols()
        scored_ref = ref._score_symbols()
        assert set(scored_gap) == set(scored_ref) == set(syms)
        for sym in syms:
            assert scored_gap[sym][0] == scored_ref[sym][0]
            assert scored_gap[sym][1]["behavioral_value"] == scored_ref[sym][1]["behavioral_value"]


def test_gapped_common_calendar_cannot_form_or_trade() -> None:
    """A missing global UTC day cannot be repaired by a later window."""
    panel = _st_panel()
    cand = _st_candidate(_ST_SYMS)
    for t in range(_N_ST):
        if t != 25:
            cand.calculate_signals(_window_event(panel, t))
    assert cand._formation_panel() == {}
    assert cand._score_symbols() == {}
    assert cand.events.items == []


def test_malformed_window_clock_is_an_atomic_noop() -> None:
    cand = _st_candidate(_ST_SYMS)
    cand.calculate_signals(_window_event(_st_panel(), 0))
    before = cand.get_state()
    bad = _window_event(_st_panel(), 1)
    bad.time = "not-a-clock"
    bad.bars_1s["SAL_UP"][0]["time"] = "also-not-a-clock"
    cand.calculate_signals(bad)
    assert cand.get_state() == before


def test_zero_residual_dispersion_cannot_trade_on_symbol_ties() -> None:
    syms = ["A", "B", "C", "D", "E", "F"]
    closes = _closes_from_returns([0.002 if t % 2 else -0.001 for t in range(_N_ST - 1)])
    cand = _st_candidate(syms)
    _feed(cand, {symbol: list(closes) for symbol in syms}, n=_N_ST)
    assert cand._score_symbols() == {}
    assert cand.events.items == []


def test_partial_or_malformed_checkpoint_is_an_atomic_noop() -> None:
    cand = _st_candidate(_ST_SYMS)
    _feed(cand, _st_panel(), n=31)
    before = cand.get_state()
    partial = cand.get_state()
    partial["symbol_state"].pop("SAL_UP")
    cand.set_state(partial)
    assert cand.get_state() == before
    malformed = cand.get_state()
    malformed["last_day_key"] = "2024-13-99"
    cand.set_state(malformed)
    assert cand.get_state() == before
    malformed = cand.get_state()
    malformed["symbol_state"]["SAL_UP"]["pending_day"] = "tomorrow"
    cand.set_state(malformed)
    assert cand.get_state() == before


def test_checkpoint_requires_exact_schema_and_coherent_clocks() -> None:
    cand = _st_candidate(_ST_SYMS)
    panel = _st_panel()
    _feed(cand, panel, n=31)
    before = cand.get_state()

    missing_global = deepcopy(before)
    missing_global.pop("last_window_key")
    cand.set_state(missing_global)
    assert cand.get_state() == before

    empty_global = deepcopy(before)
    empty_global["last_decision_week"] = ""
    cand.set_state(empty_global)
    assert cand.get_state() == before

    missing_symbol = deepcopy(before)
    missing_symbol["symbol_state"]["SAL_UP"].pop("pending_close")
    cand.set_state(missing_symbol)
    assert cand.get_state() == before

    extra_global = deepcopy(before)
    extra_global["forged"] = True
    cand.set_state(extra_global)
    assert cand.get_state() == before

    extra_symbol = deepcopy(before)
    extra_symbol["symbol_state"]["SAL_UP"]["forged"] = True
    cand.set_state(extra_symbol)
    assert cand.get_state() == before

    clock_mismatch = deepcopy(before)
    clock_mismatch["symbol_state"]["SAL_UP"]["last_bar_key"] = _ts(29)
    cand.set_state(clock_mismatch)
    assert cand.get_state() == before

    coerced_close = deepcopy(before)
    coerced_close["symbol_state"]["SAL_UP"]["daily_closes"][0] = 100
    cand.set_state(coerced_close)
    assert cand.get_state() == before


def test_checkpoint_same_week_replay_and_continuation_are_exact() -> None:
    panel = _st_panel()
    original = _st_candidate(_ST_SYMS)
    _feed(original, panel, n=31)
    checkpoint = original.get_state()

    resumed = _st_candidate(_ST_SYMS)
    resumed.set_state(deepcopy(checkpoint))
    assert resumed.get_state() == checkpoint

    resumed.calculate_signals(_window_event(panel, 30))
    assert resumed.get_state() == checkpoint
    assert resumed.events.items == []

    original_event_count = len(original.events.items)
    original.calculate_signals(_window_event(panel, 31))
    resumed.calculate_signals(_window_event(panel, 31))
    assert resumed._tick == checkpoint["tick"]
    assert resumed.get_state() == original.get_state()

    for t in range(32, _N_ST):
        original.calculate_signals(_window_event(panel, t))
        resumed.calculate_signals(_window_event(panel, t))
    assert resumed.get_state() == original.get_state()
    assert [(signal.symbol, signal.signal_type) for signal in resumed.events.items] == [
        (signal.symbol, signal.signal_type)
        for signal in original.events.items[original_event_count:]
    ]


def test_state_roundtrip_preserves_committed_day_keys() -> None:
    """The parallel committed-day-key deque serializes losslessly."""
    cand = _st_candidate(_ST_SYMS)
    _feed(cand, _st_panel(), n=31)
    snap = cand.get_state()
    payload = snap["symbol_state"]["SAL_UP"]
    assert len(payload["daily_days"]) == len(payload["daily_closes"])
    assert payload["daily_days"][-1] == cand._last_committed_day
    restored = _st_candidate(_ST_SYMS)
    restored.set_state(snap)
    assert list(restored._state["SAL_UP"].daily_days) == payload["daily_days"]
    assert restored.get_state() == snap


def test_min_symbols_self_skip() -> None:
    syms = ["ONLY1", "ONLY2", "ONLY3"]  # below min_symbols=5
    panel = {
        s: _closes_from_returns([0.002 * ((i + t) % 3 - 1) for t in range(_N_ST - 1)])
        for i, s in enumerate(syms)
    }
    cand = _st_candidate(syms)
    _feed(cand, panel, n=_N_ST)
    assert cand.events.items == []


def test_short_history_self_skip() -> None:
    st = _st_candidate(_ST_SYMS)
    _feed(st, _st_panel(), n=15)  # < 22 committed daily closes required
    assert st.events.items == []
    pt = _pt_candidate(_PT_SYMS)
    _feed(pt, _pt_panel(), n=60)  # < 91 committed daily closes required
    assert pt.events.items == []


def test_min_hold_reference_config_trades_at_least_as_often() -> None:
    panel = _st_panel()
    held = _st_candidate(_ST_SYMS, min_hold_daily_decisions=5)
    _feed(held, panel, n=_N_ST)
    ref = _st_candidate(_ST_SYMS, min_hold_daily_decisions=0)
    _feed(ref, panel, n=_N_ST)
    assert len(ref.events.items) >= len(held.events.items)


# --------------------------------------------------------------------------- #
# schema / exports
# --------------------------------------------------------------------------- #


def test_param_schema_is_snake_case_and_canonical_defaults() -> None:
    for cls in (SalienceTheoryValueStrategy, ProspectTheoryValueStrategy):
        schema = cls.get_param_schema()
        assert schema
        for name, hp in schema.items():
            assert name == name.lower() and " " not in name
            assert isinstance(hp, HyperParam)
    st_schema = SalienceTheoryValueStrategy.get_param_schema()
    assert st_schema["theta"].default == 0.1 and st_schema["delta"].default == 0.7
    assert st_schema["formation_days"].default == 21
    pt_schema = ProspectTheoryValueStrategy.get_param_schema()
    assert pt_schema["alpha"].default == 0.88 and pt_schema["lam"].default == 2.25
    assert pt_schema["gamma_gain"].default == 0.61 and pt_schema["gamma_loss"].default == 0.69
    assert pt_schema["formation_days"].default == 90
    for schema in (st_schema, pt_schema):
        assert schema["quantile_entry_pct"].default == 0.20
        assert schema["quantile_exit_pct"].default == 0.35
        assert schema["min_hold_daily_decisions"].default == 5


def test_slice_exports_pinned() -> None:
    assert "cross_sectional" in _SUGGESTED_CANDIDATE_TAGS
    for sl, keys in (
        (_SALIENCE_VALUE_SLICE, ("theta", "delta")),
        (_PROSPECT_VALUE_SLICE, ("alpha", "lam")),
    ):
        assert {"1d", "4h", "1h"} <= set(sl)
        for tf in ("1d", "4h", "1h"):
            cells = sl[tf]
            assert 2 <= len(cells) <= 4
            # UTC-day sampling makes the cells timeframe INVARIANT.
            assert cells == sl["1d"]
            for cell in cells:
                for key in keys:
                    assert key in cell
                assert cell["quantile_entry_pct"] == 0.20
                assert cell["quantile_exit_pct"] == 0.35
    assert _SALIENCE_VALUE_SLICE["1d"][0]["theta"] == 0.1
    assert _SALIENCE_VALUE_SLICE["1d"][0]["delta"] == 0.7
    assert _PROSPECT_VALUE_SLICE["1d"][0]["alpha"] == 0.88
    assert _PROSPECT_VALUE_SLICE["1d"][0]["lam"] == 2.25


def test_intended_hold_constants() -> None:
    assert SalienceTheoryValueStrategy.intended_hold_seconds == 14 * 86400
    assert ProspectTheoryValueStrategy.intended_hold_seconds == 21 * 86400
    assert not math.isclose(
        SalienceTheoryValueStrategy.intended_hold_seconds,
        ProspectTheoryValueStrategy.intended_hold_seconds,
    )
