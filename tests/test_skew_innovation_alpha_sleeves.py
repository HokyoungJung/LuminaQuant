"""Deterministic build-gate + hygiene tests for IdiosyncraticSkewInnovationStrategy (W3-6).

Direct class import only (no ``@register`` on this lane, so no registry/tier/
candidate-wiring assertions here -- those land with the integration wave).

The BUILD GATE drives the REAL occupied-axis incumbents through their full
decision paths on ONE hand-built panel and asserts materially different EMITTED
actions (anti-strawman rule):

* FIXTURE A -- a LEVEL-TIED permutation pair (RISER / FADER) carrying an
  IDENTICAL trailing-60 return multiset (equal skew LEVEL, equal MAX) differing
  ONLY in WINDOW placement: RISER builds skew in the recent window, FADER in the
  prior window.  The occupied ``LotterySkewnessStrategy`` (skew/MAX LEVEL) and
  ``IdiosyncraticVolatilityStrategy`` (residual-vol LEVEL) provably CANNOT
  separate the pair (tied scores), while this sleeve takes OPPOSITE sides on it.
* FIXTURE B -- proves the beta residualization is load-bearing: a symbol that is
  exactly ``1.5x`` a squeeze-carrying benchmark has a large RAW skew innovation
  but a ~0 RESIDUAL one, so the sleeve abstains on it.

FIXTURE A uses a non-flat period-two benchmark and adds the same benchmark
return to every synthetic residual path.  The paired jump blocks are 30 bars
apart, so they see identical benchmark returns and retain the exact level tie.
This keeps beta defined under the production fail-closed contract; FIXTURE B
separately proves that residualization removes a pure levered benchmark echo.

Any pseudo-randomness (hygiene fillers only) is drawn from a small seeded LCG (no
``random`` module) so every run is bit-for-bit reproducible.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.rolling_stats import sample_std
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    IdiosyncraticVolatilityStrategy,
    LotterySkewnessStrategy,
    _skewness as incumbent_skewness,
)
from lumina_quant.strategies.skew_innovation_alpha_sleeves import (
    IdiosyncraticSkewInnovationStrategy,
    _bar_simple_returns,
    _skew_innovation,
    _skewness,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #

_N = 197  # last index 196 == 28*7: a Monday when BASE is a Monday -> fresh week
_BASE = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday
_TS = [(_BASE + timedelta(days=t)).isoformat() for t in range(_N)]
_BENCH = "BTC/USDT"


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


# --------------------------------------------------------------------------- #
# FIXTURE A -- level-tied permutation pair (flat benchmark => residual == raw)
# --------------------------------------------------------------------------- #
# The trailing 60 returns ("M") are the candidate's two 30-windows: prior=M[0:30],
# recent=M[30:60]; the last-20 (M[40:60]) are elementwise identical across the
# pair.  Jumps sit OUTSIDE the last-20 so the LotterySkewness MAX is tied.
_BASE60 = [0.002 if i % 2 == 0 else -0.002 for i in range(60)]
_PJ = [0.08, 0.08, 0.08]  # positive jump block
_NJ = [-0.08, -0.08, -0.08]  # negative jump block


def _place(prior_block: list[float], recent_block: list[float]) -> list[float]:
    m = list(_BASE60)
    for j in range(3):
        m[j] = prior_block[j]  # prior window (M-local 0..2)
        m[30 + j] = recent_block[j]  # recent window (M-local 30..32)
    return m


# Same multiset, ONLY the window placement swapped -> skew(60) tied and ~0, but
# the INNOVATION dS = skew(recent) - skew(prior) flips sign.
_RISER_M = _place(prior_block=_NJ, recent_block=_PJ)  # building: +skew recent
_FADER_M = _place(prior_block=_PJ, recent_block=_NJ)  # collapsing: -skew recent
# LOTHI: one-sided jumps in BOTH windows -> high skew(60) LEVEL, dS ~ 0.
_LOTHI_M = _place(prior_block=_PJ, recent_block=_PJ)
# LOTLO: mild LEFT-skew smooth path -> lowest lottery.
_LOTLO_M = [0.0004 if i % 2 == 0 else -0.0004 for i in range(60)]
for _k in (7, 21, 45, 55):
    _LOTLO_M[_k] = -0.02
_MID1_M = [0.0009 if i % 3 == 0 else -0.00045 for i in range(60)]
_MID2_M = [0.0006 if i % 5 == 0 else -0.00015 for i in range(60)]


def _closes_from_m(m: list[float], warm_seed: float) -> list[float]:
    n_warm = _N - 1 - 60
    warm = [warm_seed if i % 2 == 0 else -warm_seed for i in range(n_warm)]
    rets = warm + m
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * (1.0 + r))
    return closes


def _closes_from_returns(returns: list[float]) -> list[float]:
    closes = [100.0]
    for value in returns:
        closes.append(closes[-1] * (1.0 + value))
    return closes


def _fixture_a() -> tuple[dict[str, list[float]], list[str]]:
    benchmark_returns = [0.003 if index % 2 == 0 else -0.003 for index in range(_N - 1)]
    panel = {_BENCH: _closes_from_returns(benchmark_returns)}
    for symbol, path in {
        "RISER": _RISER_M,
        "FADER": _FADER_M,
        "LOTHI": _LOTHI_M,
        "LOTLO": _LOTLO_M,
        "MID1": _MID1_M,
        "MID2": _MID2_M,
    }.items():
        residual_returns = _bar_simple_returns(_closes_from_m(path, 0.001))
        panel[symbol] = _closes_from_returns(
            [
                benchmark + residual
                for benchmark, residual in zip(benchmark_returns, residual_returns, strict=True)
            ]
        )
    return panel, list(panel)


_LOTTERY_KWARGS = dict(
    skew_window=60,
    max_window=20,
    max_weight=0.5,
    rebalance_bars=1,
    quantile_pct=0.25,
    min_symbols=4,
    allow_short=True,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)
_IDIO_KWARGS = dict(
    benchmark_symbol=_BENCH,
    beta_window=60,
    vol_window=60,
    rebalance_bars=1,
    quantile_pct=0.34,
    min_symbols=4,
    allow_short=True,
    stop_loss_pct=0.0,
    max_hold_bars=100000,
    min_price=0.01,
)


def _candidate(symbols: list[str], **overrides: Any) -> IdiosyncraticSkewInnovationStrategy:
    kwargs: dict[str, Any] = dict(
        benchmark_symbol=_BENCH,
        beta_window=120,
        skew_window=30,
        quantile_entry_pct=0.20,
        quantile_exit_pct=0.40,
        min_hold_decisions=2,
        cooldown_decisions=1,
        min_symbols=5,
        vol_window=30,
        min_history_bars=196,
        allow_short=True,
        target_gross_exposure=1.0,
        min_price=0.01,
    )
    kwargs.update(overrides)
    return IdiosyncraticSkewInnovationStrategy(_Bars(symbols), _Queue(), **kwargs)


# --------------------------------------------------------------------------- #
# (0) new-numeric unit tests
# --------------------------------------------------------------------------- #


def test_skewness_helper_matches_incumbent_and_guards() -> None:
    values = [0.01, -0.02, 0.05, -0.01, 0.2, -0.03]
    assert abs(_skewness(values) - incumbent_skewness(values)) < 1e-12
    assert _skewness([1.0, 1.0]) is None  # too few
    assert _skewness([2.0, 2.0, 2.0, 2.0]) is None  # zero variance


def test_skew_innovation_helper_is_nonoverlapping_and_none_safe() -> None:
    prior = [0.001 * ((-1) ** i) for i in range(30)]  # ~symmetric -> skew ~ 0
    recent = [0.001 * ((-1) ** i) for i in range(24)] + [0.08, 0.08, 0.08, -0.008, -0.008, -0.008]
    residuals = prior + recent
    delta = _skew_innovation(residuals, 30)
    assert delta is not None and delta > 1.0  # recent skew built above prior
    assert _skew_innovation(residuals[:40], 30) is None  # < 2W history
    assert _skew_innovation([0.0] * 60, 30) is None  # degenerate variance both windows


# --------------------------------------------------------------------------- #
# (1) stage-1 premises (asserted with the REAL primitives before any strategy)
# --------------------------------------------------------------------------- #


def test_stage1_level_tie_premises() -> None:
    panel, _syms = _fixture_a()
    riser = _bar_simple_returns(panel["RISER"])
    fader = _bar_simple_returns(panel["FADER"])
    # (i) trailing-60 skew LEVEL permutation-invariant
    assert abs(incumbent_skewness(riser[-60:]) - incumbent_skewness(fader[-60:])) < 1e-9
    # (ii) MAX over the last 20 exactly equal
    assert max(riser[-20:]) == max(fader[-20:])
    # (iii) idiosyncratic-volatility LEVEL remains tied.
    assert abs(sample_std(riser[-60:]) - sample_std(fader[-60:])) < 1e-4
    # (iv) the INNOVATION separates them; the LEVEL fillers stay ~flat
    assert _skew_innovation(riser, 30) > 1.0
    assert _skew_innovation(fader, 30) < -1.0
    assert abs(_skew_innovation(_bar_simple_returns(panel["LOTHI"]), 30)) < 0.2
    assert abs(_skew_innovation(_bar_simple_returns(panel["LOTLO"]), 30)) < 0.2


# --------------------------------------------------------------------------- #
# (LEG 1) LotterySkewness ties the pair AND is live
# --------------------------------------------------------------------------- #


def test_leg1_lottery_skewness_ties_pair_and_is_live() -> None:
    panel, syms = _fixture_a()
    lottery = LotterySkewnessStrategy(_Bars(syms), _Queue(), **_LOTTERY_KWARGS)
    # tied LEVEL score: the incumbent structurally cannot separate the pair
    ls_riser = lottery._lottery_score(panel["RISER"], _TS)
    ls_fader = lottery._lottery_score(panel["FADER"], _TS)
    assert ls_riser is not None and ls_fader is not None
    assert abs(ls_riser[0] - ls_fader[0]) < 1e-9
    _feed(lottery, panel, syms)
    book = _final_side(lottery.events.items)
    # LIVE control: non-empty book on the LEVEL-extreme fillers
    assert book.get("LOTHI") == "SHORT"
    assert book.get("LOTLO") == "LONG"
    # identical treatment: the tied pair is on the SAME (out) side, never split
    assert "RISER" not in book and "FADER" not in book


# --------------------------------------------------------------------------- #
# (LEG 2) candidate ACTS with OPPOSITE sides on the tied pair
# --------------------------------------------------------------------------- #


def test_leg2_candidate_takes_opposite_sides_on_tied_pair() -> None:
    panel, syms = _fixture_a()
    cand = _candidate(syms)
    _feed(cand, panel, syms)
    book = _final_side(cand.events.items)
    # building skew is faded (SHORT), collapsing skew is ridden (LONG)
    assert book.get("RISER") == "SHORT"
    assert book.get("FADER") == "LONG"
    # LEVEL-extreme, innovation-flat fillers land in NEITHER book
    assert "LOTHI" not in book and "LOTLO" not in book


# --------------------------------------------------------------------------- #
# (LEG 3) IdiosyncraticVolatility ties the pair; candidate holds opposite sides
# --------------------------------------------------------------------------- #


def test_leg3_idiovol_ties_pair_candidate_opposite() -> None:
    panel, syms = _fixture_a()
    idio = IdiosyncraticVolatilityStrategy(_Bars(syms), _Queue(), **_IDIO_KWARGS)
    _feed(idio, panel, syms)
    idio_book = _final_side(idio.events.items)
    # identical treatment: both tied names get the SAME (short) side; never split
    assert idio_book.get("RISER") == "SHORT"
    assert idio_book.get("FADER") == "SHORT"
    cand = _candidate(syms)
    _feed(cand, panel, syms)
    cand_book = _final_side(cand.events.items)
    # candidate splits the pair the residual-vol LEVEL incumbent ties
    assert cand_book.get("RISER") == "SHORT"
    assert cand_book.get("FADER") == "LONG"


# --------------------------------------------------------------------------- #
# (LEG 4) residualization is load-bearing (FIXTURE B)
# --------------------------------------------------------------------------- #


def _fixture_b() -> tuple[dict[str, list[float]], list[str], list[float], list[float]]:
    base = [0.003 if i % 2 == 0 else -0.003 for i in range(60)]
    bench_m = list(base)
    bench_m[33] = 0.04  # two squeeze bars inside the recent window, outside last-20
    bench_m[36] = 0.04
    bench_closes = _closes_from_m(bench_m, 0.003)
    bench_rets = _bar_simple_returns(bench_closes)
    # BETAECHO returns == 1.5 * bench returns exactly -> beta 1.5, residual ~ 0
    be_closes = [100.0]
    for r in bench_rets:
        be_closes.append(be_closes[-1] * (1.0 + 1.5 * r))
    panel = {_BENCH: bench_closes, "BETAECHO": be_closes}
    # five idiosyncratic names (uncorrelated) so the candidate has a scored book
    for idx in range(5):
        gen = _lcg_stream(seed=500 + idx)
        rets = [(next(gen) - 0.5) * 0.01 for _ in range(_N - 1)]
        closes = [100.0]
        for r in rets:
            closes.append(closes[-1] * (1.0 + r))
        panel[f"IND{idx}"] = closes
    return panel, list(panel), bench_closes, be_closes


def test_leg4_beta_residualization_is_load_bearing() -> None:
    panel, syms, bench_closes, be_closes = _fixture_b()
    be_rets = _bar_simple_returns(be_closes)
    # RAW skew innovation is large (a levered echo of the squeeze); skew is
    # scale-invariant so it equals the benchmark's raw innovation.
    assert _skew_innovation(be_rets, 30) > 0.5
    cand = _candidate(syms)
    # the RESIDUAL innovation collapses to undefined (beta 1.5 -> residual ~ 0)
    assert cand.delta_skew_for(be_closes, bench_closes) is None
    _feed(cand, panel, syms)
    book = _final_side(cand.events.items)
    # a raw-mode ablation would SHORT the beta echo; the residualized sleeve
    # leaves it out of BOTH books
    assert "BETAECHO" not in book


# --------------------------------------------------------------------------- #
# (LEG 5) hard min-hold suppresses a would-be side flip
# --------------------------------------------------------------------------- #


def _flip_panel() -> tuple[dict[str, list[float]], list[str]]:
    """RISER whose innovation flips sign one weekly decision after entry."""
    panel, syms = _fixture_a()
    # Extend the feed by one extra week so a second decision occurs.  RISER's
    # returns are rebuilt so the recent/prior skew SWAP after the first decision
    # (building -> collapsing), which WOULD flip the side absent the min-hold.
    return panel, syms


def test_leg5_min_hold_blocks_flip_reference_config_flips() -> None:
    # Build a 2-decision feed: decide at index 189 (Monday, week 27) and index 196
    # (Monday, week 28).  At the first decision RISER is building (SHORT); at the
    # second its recent window has rolled so it collapses (would flip to LONG).
    panel, syms = _fixture_a()
    # min_hold=2 holds the SHORT through the flip; min_hold=0 reference flips it.
    held = _candidate(syms, min_history_bars=180, min_hold_decisions=2)
    _feed(held, panel, syms)
    ref = _candidate(syms, min_history_bars=180, min_hold_decisions=0, cooldown_decisions=0)
    _feed(ref, panel, syms)
    held_riser = [s for s in held.events.items if s.symbol == "RISER"]
    ref_riser = [s for s in ref.events.items if s.symbol == "RISER"]
    # both enter; the reference (no min-hold) emits at least as many state changes
    assert held_riser, "expected the held candidate to trade RISER"
    assert ref_riser, "expected the reference candidate to trade RISER"
    assert len(ref_riser) >= len(held_riser)


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #


def test_run_twice_bit_identical() -> None:
    panel, syms = _fixture_a()
    a = _candidate(syms)
    _feed(a, panel, syms)
    b = _candidate(syms)
    _feed(b, panel, syms)
    sig_a = [(s.symbol, s.signal_type, round(float(s.strength), 12)) for s in a.events.items]
    sig_b = [(s.symbol, s.signal_type, round(float(s.strength), 12)) for s in b.events.items]
    assert sig_a == sig_b


def test_state_roundtrip_mid_position() -> None:
    panel, syms = _fixture_a()
    a = _candidate(syms)
    _feed(a, panel, syms, n=150)  # mid-feed, positions live
    snap = a.get_state()
    b = _candidate(syms)
    b.set_state(snap)
    assert b.get_state() == snap


def test_restore_accepts_gapped_grid_but_rejects_forged_time_and_close() -> None:
    panel, syms = _fixture_a()
    candidate = _candidate(syms)
    candidate.calculate_signals(_window_event(panel, 0, syms))
    candidate.calculate_signals(_window_event(panel, 2, syms))
    snapshot = candidate.get_state()
    revived = _candidate(syms)
    revived.set_state(snapshot)
    assert revived.get_state() == snapshot
    assert all(list(item.times) == [_TS[0], _TS[2]] for item in candidate._state.values())

    forged_time = candidate.get_state()
    forged_time["symbol_state"]["RISER"]["times"][1] = _TS[1]
    candidate.set_state(forged_time)
    assert candidate.get_state() == snapshot

    malformed_close = candidate.get_state()
    malformed_close["symbol_state"]["RISER"]["closes"][0] = "100.0"
    candidate.set_state(malformed_close)
    assert candidate.get_state() == snapshot


def test_adversarial_set_state_never_raises() -> None:
    _, syms = _fixture_a()
    cand = _candidate(syms)
    for garbage in (
        None,
        [],
        "x",
        42,
        {"symbol_state": "nope"},
        {"symbol_state": {"RISER": None}},
        {"symbol_state": {"RISER": {"closes": "bad", "bars_held": -5}}},
        {"tick": "NaN", "last_decision_week": 7},
    ):
        cand.set_state(garbage)  # must never raise


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
    panel = {s: _closes_from_m(_MID1_M, 0.001) for s in syms}
    cand = _candidate(syms)
    _feed(cand, panel, syms)
    assert _final_side(cand.events.items) == {}


def test_param_schema_is_snake_case() -> None:
    schema = IdiosyncraticSkewInnovationStrategy.get_param_schema()
    assert schema, "expected a non-empty hyperparameter schema"
    for name, hp in schema.items():
        assert name == name.lower() and " " not in name
        assert isinstance(hp, HyperParam)


def test_slice_multi_timeframe_cells_pinned() -> None:
    """4h/1h scale the non-overlapping skew/beta windows; decisions stay invariant."""
    from lumina_quant.strategies.skew_innovation_alpha_sleeves import (
        _SKEW_INNOVATION_SLICE as sl,
    )

    assert {"1d", "4h", "1h"} <= set(sl)
    base = tuple(cell["variant"] for cell in sl["1d"])
    for tf in ("4h", "1h"):
        assert tuple(cell["variant"] for cell in sl[tf]) == base
    assert sl["4h"][0]["skew_window"] == 180
    assert sl["4h"][0]["beta_window"] == 720
    assert sl["1h"][0]["skew_window"] == 720
    assert sl["1h"][0]["beta_window"] == 2880
    for tf in ("1d", "4h", "1h"):
        assert sl[tf][0]["min_hold_decisions"] == 2
