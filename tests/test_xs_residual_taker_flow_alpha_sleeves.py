"""Deterministic build-gate + hygiene tests for the residual taker-flow sleeve.

Data-free, synthetic-panel tests (no registry/tier/candidate-wiring assertions;
the orchestrator covers registration).  Covered legs:

  * Sign-lock: a synthetic panel where flat-price names carry strong POSITIVE
    net taker flow ends LONG and strong NEGATIVE net flow ends SHORT (the
    pre-registered continuation sign; a reversal-shaped result fails here).
  * Residual-degeneracy guard: a dispersion-free flow cross-section emits
    NOTHING (the in-code leg of the documented data-PC gate).
  * Rank hysteresis: a held long whose residual decays to mid-rank is retained
    inside a wide hysteresis band and exited without one.
  * Hard min-hold: a would-be side flip inside the min-hold window is
    suppressed.
  * Never-raise on None/degenerate inputs (missing taker features, nan closes,
    empty windows, foreign event types).
  * Zero-weight no-emit (v5 zero-alloc resize trap) + intended_hold_seconds
    stamping on sized entries.
  * State roundtrip (lossless) + adversarial set_state never raises.
  * Two-run determinism.
  * Min-symbols / short-history self-skip.
  * Vol-target throttle annualization via ``indicators/annualization``.
  * Slice invariants: pre-registered mechanism constants fixed across variants.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)
from lumina_quant.strategies.xs_residual_taker_flow_alpha_sleeves import (
    _SUGGESTED_CANDIDATE_TAGS,
    _XS_RESIDUAL_TAKER_FLOW_SLICE,
    CrossSectionalResidualTakerFlowStrategy,
)
from lumina_quant.tuning import HyperParam

# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class _Bars:
    """Bars stub: ``symbol_list`` plus a mutable per-bar feature map.

    ``features`` maps ``(symbol, field)`` to the CURRENT bar's value (or is
    missing -> None), so the sleeve's ``get_latest_feature_value`` cascade reads
    a deterministic per-bar taker print.  The feed loop mutates it before each
    event.
    """

    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)
        self.features: dict[tuple[str, str], float] = {}

    def get_latest_feature_value(self, symbol: str, field_name: str) -> float | None:
        return self.features.get((str(symbol), str(field_name)))


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


def _feed(
    strategy: CrossSectionalResidualTakerFlowStrategy,
    bars: _Bars,
    series: dict[str, list[dict[str, Any]]],
    flows: dict[str, list[tuple[float, float] | None]],
) -> None:
    """Feed bar ``idx`` for every symbol, latching that bar's taker features."""
    n = len(next(iter(series.values())))
    for idx in range(n):
        bars.features = {}
        for sym in series:
            print_ = flows.get(sym, [None] * n)[idx]
            if print_ is not None:
                buy, sell = print_
                bars.features[(sym, "taker_buy_quote_volume")] = float(buy)
                bars.features[(sym, "taker_sell_quote_volume")] = float(sell)
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
# deterministic panel generators (inline LCG, never the random module)
# --------------------------------------------------------------------------- #

_N = 40


def _lcg(seed: int) -> Any:
    state = seed & 0x7FFFFFFF

    def _next() -> float:
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state / float(0x7FFFFFFF)

    return _next


def _wig(i: int) -> float:
    """Period-3 wiggle: keeps realized vol positive, 12-bar return exactly 0."""
    return 0.2 * ((i % 3) - 1)


def _flat_bars(price: float, n: int = _N) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    for i in range(n):
        close = price + _wig(i)
        bars.append(
            {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 10.0}
        )
    return bars


def _const_flow(buy: float, sell: float, n: int = _N) -> list[tuple[float, float] | None]:
    return [(buy, sell)] * n


def _panel() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[tuple[float, float] | None]]]:
    """10 flat-price symbols; only the taker imbalance differs cross-sectionally.

    ACC* absorb heavy net taker BUYING with a flat price (informed accumulation),
    DIST* heavy net SELLING; fillers carry small distinct imbalances so the
    cross-section has dispersion.  Quintile 0.20 of 10 -> 2 names per side.
    """
    series: dict[str, list[dict[str, Any]]] = {}
    flows: dict[str, list[tuple[float, float] | None]] = {}
    series["ACC0/USDT"] = _flat_bars(100.0)
    series["ACC1/USDT"] = _flat_bars(101.0)
    series["DIST0/USDT"] = _flat_bars(102.0)
    series["DIST1/USDT"] = _flat_bars(103.0)
    flows["ACC0/USDT"] = _const_flow(800.0, 200.0)
    flows["ACC1/USDT"] = _const_flow(780.0, 220.0)
    flows["DIST0/USDT"] = _const_flow(200.0, 800.0)
    flows["DIST1/USDT"] = _const_flow(220.0, 780.0)
    rng = _lcg(20260820)
    for fi in range(6):
        sym = f"F{fi}/USDT"
        series[sym] = _flat_bars(90.0 + fi)
        tilt = (rng() - 0.5) * 40.0  # small, deterministic, distinct imbalance
        flows[sym] = _const_flow(500.0 + tilt, 500.0 - tilt)
    return series, flows


_COMMON_KWARGS: dict[str, Any] = dict(
    formation_window_bars=12,
    decision_interval_bars=1,
    rebalance_decisions=1,
    min_hold_decisions=0,
    rank_hysteresis_z=0.0,
    quantile_pct=0.20,
    vol_window=8,
    min_symbols=6,
    allow_short=True,
    target_vol=0.0,
    stop_loss_pct=0.0,
    max_hold_bars=0,
    min_price=0.01,
)


def _run(
    series: dict[str, list[dict[str, Any]]],
    flows: dict[str, list[tuple[float, float] | None]],
    **overrides: Any,
) -> CrossSectionalResidualTakerFlowStrategy:
    syms = list(series)
    bars = _Bars(syms)
    strat = CrossSectionalResidualTakerFlowStrategy(
        bars, _Queue(), **dict(_COMMON_KWARGS, **overrides)
    )
    _feed(strat, bars, series, flows)
    return strat


# --------------------------------------------------------------------------- #
# sign-lock (pre-registered continuation sign)
# --------------------------------------------------------------------------- #


def test_sign_lock_absorbed_buying_flat_price_goes_long() -> None:
    series, flows = _panel()
    strat = _run(series, flows)
    side = _final_side(strat.events.items)
    # Continuation sign is PRE-REGISTERED: absorbed net taker buying with a
    # flat price -> LONG; absorbed selling -> SHORT.  A reversal-shaped
    # implementation fails here and dies honestly.
    assert side.get("ACC0/USDT") == "LONG"
    assert side.get("ACC1/USDT") == "LONG"
    assert side.get("DIST0/USDT") == "SHORT"
    assert side.get("DIST1/USDT") == "SHORT"
    # Quintile 0.20 of 10 names = exactly 2 per side; fillers stay out.
    assert all(not sym.startswith("F") for sym in side)


def test_entry_metadata_carries_residual_and_intended_hold() -> None:
    series, flows = _panel()
    strat = _run(series, flows)
    entries = _non_exit(strat.events.items)
    assert entries
    for sig in entries:
        meta = dict(sig.metadata or {})
        assert float(meta["target_allocation"]) > 0.0
        assert float(meta["intended_hold_seconds"]) == 604800.0
        assert "flow_residual_z" in meta and "taker_flow" in meta


# --------------------------------------------------------------------------- #
# residual-degeneracy guard (in-code leg of the documented data-PC gate)
# --------------------------------------------------------------------------- #


def test_degenerate_flow_cross_section_emits_nothing() -> None:
    series, _ = _panel()
    # Identical imbalance for every symbol -> flow z degenerates to all-zero ->
    # residual carries no cross-sectional dispersion -> the guard emits NOTHING.
    flows = {sym: _const_flow(600.0, 400.0) for sym in series}
    strat = _run(series, flows)
    assert _non_exit(strat.events.items) == []


# --------------------------------------------------------------------------- #
# rank hysteresis (0.5z band, tested with band on vs off)
# --------------------------------------------------------------------------- #


def _decay_panel() -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, list[tuple[float, float] | None]]
]:
    """ACC0 is top-flow in phase 1, decays to mid-rank in phase 2; NEW takes over."""
    phase1, phase2 = 30, 30
    n = phase1 + phase2
    series: dict[str, list[dict[str, Any]]] = {}
    flows: dict[str, list[tuple[float, float] | None]] = {}
    series["ACC0/USDT"] = _flat_bars(100.0, n)
    flows["ACC0/USDT"] = [(800.0, 200.0)] * phase1 + [(505.0, 495.0)] * phase2
    series["NEW/USDT"] = _flat_bars(101.0, n)
    flows["NEW/USDT"] = [(500.0, 500.0)] * phase1 + [(900.0, 100.0)] * phase2
    series["DIST0/USDT"] = _flat_bars(102.0, n)
    flows["DIST0/USDT"] = [(200.0, 800.0)] * n
    rng = _lcg(97)
    for fi in range(5):
        sym = f"F{fi}/USDT"
        series[sym] = _flat_bars(90.0 + fi, n)
        tilt = (rng() - 0.5) * 20.0
        flows[sym] = [(500.0 + tilt, 500.0 - tilt)] * n
    return series, flows


def test_rank_hysteresis_band_retains_decayed_holding() -> None:
    series, flows = _decay_panel()
    banded = _run(series, flows, rank_hysteresis_z=50.0)
    unbanded = _run(series, flows, rank_hysteresis_z=0.0)
    # With a wide band the decayed ACC0 long is retained; with no band it is
    # exited once it leaves the top quintile.
    assert _final_side(banded.events.items).get("ACC0/USDT") == "LONG"
    assert "ACC0/USDT" not in _final_side(unbanded.events.items)
    # The band never blocks fresh top-rank entries.
    assert _final_side(banded.events.items).get("NEW/USDT") == "LONG"


# --------------------------------------------------------------------------- #
# hard min-hold (the one proven turnover rescue, applied ex-ante)
# --------------------------------------------------------------------------- #


def _flip_panel() -> tuple[
    dict[str, list[dict[str, Any]]], dict[str, list[tuple[float, float] | None]]
]:
    """Phase 1: P accumulation / Q distribution.  Phase 2: the footprints SWAP."""
    phase1, phase2 = 30, 30
    n = phase1 + phase2
    series: dict[str, list[dict[str, Any]]] = {}
    flows: dict[str, list[tuple[float, float] | None]] = {}
    series["P/USDT"] = _flat_bars(100.0, n)
    flows["P/USDT"] = [(800.0, 200.0)] * phase1 + [(200.0, 800.0)] * phase2
    series["Q/USDT"] = _flat_bars(101.0, n)
    flows["Q/USDT"] = [(200.0, 800.0)] * phase1 + [(800.0, 200.0)] * phase2
    rng = _lcg(4242)
    for fi in range(6):
        sym = f"F{fi}/USDT"
        series[sym] = _flat_bars(90.0 + fi, n)
        tilt = (rng() - 0.5) * 20.0
        flows[sym] = [(500.0 + tilt, 500.0 - tilt)] * n
    return series, flows


def test_min_hold_suppresses_side_flip() -> None:
    series, flows = _flip_panel()
    free = _run(series, flows, min_hold_decisions=0)
    held = _run(series, flows, min_hold_decisions=1000)
    # With no min-hold the swap flips P from LONG to SHORT; a long min-hold
    # suppresses the flip so P is still LONG at the end of the feed.
    assert _final_side(free.events.items).get("P/USDT") == "SHORT"
    assert _final_side(held.events.items).get("P/USDT") == "LONG"


# --------------------------------------------------------------------------- #
# never-raise on None / degenerate inputs
# --------------------------------------------------------------------------- #


def test_missing_taker_features_yields_no_entries_and_never_raises() -> None:
    series, _ = _panel()
    flows: dict[str, list[tuple[float, float] | None]] = {sym: [None] * _N for sym in series}
    strat = _run(series, flows)
    assert _non_exit(strat.events.items) == []


def test_partial_feature_coverage_only_covered_symbols_enter() -> None:
    series, flows = _panel()
    # Kill the taker prints for one accumulation name entirely: all-or-nothing
    # capture drops its bars, so it contributes NO entries (never a stale rank).
    flows = dict(flows)
    flows["ACC0/USDT"] = [None] * _N
    strat = _run(series, flows)
    side = _final_side(strat.events.items)
    assert "ACC0/USDT" not in side
    assert side.get("ACC1/USDT") == "LONG"
    assert side.get("DIST0/USDT") == "SHORT"


def test_degenerate_input_never_raises() -> None:
    bars = _Bars(["Z/USDT"])
    strat = CrossSectionalResidualTakerFlowStrategy(bars, _Queue(), **_COMMON_KWARGS)
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t0", close=0.0))
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t1", close=-1.0))
    strat.calculate_signals(
        SimpleNamespace(type="MARKET", symbol="Z/USDT", time="t2", close=float("nan"))
    )
    strat.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t3", bars_1s={}))
    strat.calculate_signals(SimpleNamespace(type="MARKET_WINDOW", time="t4", bars_1s=None))
    strat.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    assert _non_exit(strat.events.items) == []


def test_zero_turnover_bars_never_raise_and_never_rank() -> None:
    series, _ = _panel()
    flows = {sym: _const_flow(0.0, 0.0) for sym in series}
    strat = _run(series, flows)
    assert _non_exit(strat.events.items) == []


# --------------------------------------------------------------------------- #
# v5 zero-alloc gate: a computed alloc of 0 must NOT emit a LONG/SHORT --
# ``_target_metadata`` omits ``target_allocation`` at alloc 0 and the engine
# would resize the entry to its DEFAULT allocation (an unsized, un-vol-gated
# bet).
# --------------------------------------------------------------------------- #


def test_zero_alloc_entry_skipped_not_default_sized() -> None:
    symbols = ["FLIP/USDT", "FRESH/USDT", "N0/USDT", "N1/USDT", "N2/USDT", "N3/USDT"]
    strat = CrossSectionalResidualTakerFlowStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    flip = strat._state["FLIP/USDT"]
    flip.mode = "LONG"
    flip.entry_price = 100.0
    flip.decisions_held = 10_000  # clear the min-hold so the side flip is live
    targets = {
        "FLIP/USDT": ("SHORT", -1.0, {}),
        "FRESH/USDT": ("LONG", 1.0, {}),
    }
    # empty weights -> alloc == 0 for every targeted symbol
    strat._emit_targets(targets, {}, 1.0, "2026-01-01T00:00:00Z")
    kinds = [(sig.symbol, str(sig.signal_type).upper()) for sig in strat.events.items]
    assert not [sym for sym, kind in kinds if kind in {"LONG", "SHORT"}], kinds
    # the side-flip EXIT still fired and FLIP is now flat (state matches it)
    assert ("FLIP/USDT", "EXIT") in kinds
    assert strat._state["FLIP/USDT"].mode == "OUT"
    assert strat._state["FLIP/USDT"].entry_price is None
    assert strat._state["FLIP/USDT"].decisions_held == 0

    # a positive inverse-vol weight DOES emit a sized entry carrying a strictly
    # positive ``target_allocation`` and the intended-hold stamp.
    sized = CrossSectionalResidualTakerFlowStrategy(_Bars(symbols), _Queue(), **_COMMON_KWARGS)
    sized._emit_targets(
        {"FRESH/USDT": ("LONG", 1.0, {})}, {"FRESH/USDT": 0.5}, 1.0, "2026-01-01T00:00:00Z"
    )
    entries = _non_exit(sized.events.items)
    assert entries, "a positive inverse-vol weight must emit a sized entry"
    for sig in entries:
        meta = dict(sig.metadata or {})
        assert float(meta.get("target_allocation", 0.0)) > 0.0
        assert float(meta.get("intended_hold_seconds", 0.0)) == 604800.0


# --------------------------------------------------------------------------- #
# state roundtrip / adversarial set_state / determinism / self-skip
# --------------------------------------------------------------------------- #


def test_state_roundtrip_lossless() -> None:
    series, flows = _panel()
    strat = _run(series, flows)
    snapshot = strat.get_state()
    restored = CrossSectionalResidualTakerFlowStrategy(
        _Bars(list(series)), _Queue(), **_COMMON_KWARGS
    )
    restored.set_state(snapshot)
    assert restored.get_state() == snapshot


def test_adversarial_set_state_never_raises() -> None:
    syms = ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT", "F/USDT"]
    strat = CrossSectionalResidualTakerFlowStrategy(_Bars(syms), _Queue(), **_COMMON_KWARGS)
    strat.set_state(None)  # type: ignore[arg-type]
    strat.set_state("nope")  # type: ignore[arg-type]
    strat.set_state({"symbol_state": "not a dict"})
    strat.set_state({"symbol_state": {"A/USDT": "not a dict either"}})
    strat.set_state({"symbol_state": {"A/USDT": {"closes": 123}}})
    strat.set_state(
        {
            "tick": "bad",
            "decision_count": -3.7,
            "recent_times": {"bad": "type"},
            "symbol_state": {
                sym: {
                    "closes": ["x", float("nan"), 1.0],
                    "net_flows": {"bad": "type"},
                    "turnovers": [-5.0, "y", 2.0],
                    "mode": 999,
                    "entry_price": "abc",
                    "bars_held": "oops",
                    "decisions_held": None,
                    "last_time_key": 5,
                    "score": [1, 2],
                }
                for sym in syms
            },
        }
    )
    for item in strat._state.values():
        assert item.mode in {"OUT", "LONG", "SHORT"}
        assert all(value >= 0.0 for value in item.turnovers)


def test_determinism_two_runs_identical_signals() -> None:
    series, flows = _panel()

    def _signals() -> list[tuple[str, str, float | None, dict[str, Any]]]:
        strat = _run(series, flows)
        return [
            (sig.symbol, sig.signal_type, sig.strength, dict(sig.metadata or {}))
            for sig in strat.events.items
        ]

    first = _signals()
    second = _signals()
    assert first == second
    assert first, "expected at least one signal"


def test_self_skip_below_min_symbols() -> None:
    series, flows = _panel()
    keep = list(series)[:3]
    strat = _run(
        {sym: series[sym] for sym in keep},
        {sym: flows[sym] for sym in keep},
        min_symbols=6,
    )
    assert _non_exit(strat.events.items) == []


def test_self_skip_on_short_history() -> None:
    series, flows = _panel()
    short_series = {sym: rows[:8] for sym, rows in series.items()}  # < formation 12
    short_flows = {sym: rows[:8] for sym, rows in flows.items()}
    strat = _run(short_series, short_flows)
    assert _non_exit(strat.events.items) == []


# --------------------------------------------------------------------------- #
# two-level rebalance clock (decision ticks x rebalance decisions)
# --------------------------------------------------------------------------- #


def test_rebalance_clock_gates_entries_to_decision_multiples() -> None:
    """With ``decision_interval_bars=5`` and ``rebalance_decisions=2`` the ranker
    only runs every 10th bar (tick % 10 == 0), i.e. the weekly clock is the
    product of the two pre-registered cadences."""
    series, flows = _panel()
    strat = _run(series, flows, decision_interval_bars=5, rebalance_decisions=2)
    entries = _non_exit(strat.events.items)
    assert entries
    for sig in entries:
        tick = int(str(sig.datetime)[1:]) + 1  # bar idx -> 1-based tick
        assert tick % 10 == 0, (sig.symbol, sig.datetime)


# --------------------------------------------------------------------------- #
# vol-target horizon regression (annualization via indicators/annualization)
# --------------------------------------------------------------------------- #

_VT_SYMS = ["A", "B", "C", "D"]
_HOUR_EPOCHS = [1_700_000_000.0 + i * 3600.0 for i in range(12)]  # 1h spacing


def _vt_sleeve() -> CrossSectionalResidualTakerFlowStrategy:
    return CrossSectionalResidualTakerFlowStrategy(_Bars(_VT_SYMS), _Queue())


def _vt_targets() -> dict[str, Any]:
    return {sym: ("LONG", 1.0, {}) for sym in _VT_SYMS}


def test_vol_target_throttle_active_on_hourly_high_vol() -> None:
    strat = _vt_sleeve()
    for epoch in _HOUR_EPOCHS:
        strat._recent_times.append(epoch)
    vols = dict.fromkeys(_VT_SYMS, 0.05)  # per-bar 5% vol
    _weights, scalar = strat._inverse_vol_weights(_vt_targets(), vols)
    assert scalar < 0.2, scalar
    portfolio_vol_ann = annualize_per_bar_vol(0.05, bars_per_year_from_spacing(list(_HOUR_EPOCHS)))
    assert scalar == min(1.0, 0.20 / portfolio_vol_ann)


def test_vol_target_passthrough_without_bar_spacing() -> None:
    strat = _vt_sleeve()  # no observed times -> spacing unknown
    vols = dict.fromkeys(_VT_SYMS, 0.05)
    _weights, scalar = strat._inverse_vol_weights(_vt_targets(), vols)
    assert scalar == 1.0  # pass-through, never throttle on a mismatched horizon


# --------------------------------------------------------------------------- #
# schema + slice pre-registration invariants
# --------------------------------------------------------------------------- #


def test_schema_keys_snake_case_and_hyperparam() -> None:
    schema = CrossSectionalResidualTakerFlowStrategy.get_param_schema()
    assert schema
    for key, value in schema.items():
        assert key == key.lower()
        assert " " not in key
        assert isinstance(value, HyperParam)
    for required in (
        "formation_window_bars",
        "decision_interval_bars",
        "rebalance_decisions",
        "min_hold_decisions",
        "rank_hysteresis_z",
        "quantile_pct",
        "min_symbols",
        "intended_hold_seconds",
    ):
        assert required in schema


def test_required_features_declared() -> None:
    assert CrossSectionalResidualTakerFlowStrategy.required_features == (
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
    )


def test_slice_mechanism_constants_pre_registered_and_invariant() -> None:
    """Every variant/timeframe carries the SAME pre-registered mechanism
    constants (quintile 0.20, 0.5z hysteresis, weekly rebalance = 7 daily
    decisions, hard min-hold 7, ~7-day intended hold); only bar-denominated
    windows scale with the timeframe and only risk plumbing differs by variant."""
    slice_ = _XS_RESIDUAL_TAKER_FLOW_SLICE
    assert set(slice_) == {"1h", "4h"}
    for timeframe, cells in slice_.items():
        assert 2 <= len(cells) <= 4
        for cell in cells:
            assert cell["quantile_pct"] == 0.20, timeframe
            assert cell["rank_hysteresis_z"] == 0.5, timeframe
            assert cell["rebalance_decisions"] == 7, timeframe
            assert cell["min_hold_decisions"] == 7, timeframe
            assert cell["intended_hold_seconds"] == 604800.0, timeframe
            # daily decision tick x formation ~5 days, in the timeframe's bars
            if timeframe == "1h":
                assert cell["formation_window_bars"] == 120
                assert cell["decision_interval_bars"] == 24
            else:
                assert cell["formation_window_bars"] == 30
                assert cell["decision_interval_bars"] == 6
    assert "cross_sectional" in _SUGGESTED_CANDIDATE_TAGS


# --------------------------------------------------------------------------- #
# XS-STOP-BRACKET-DESYNC regression: entries carry NO engine intrabar bracket
# (stop_loss=None); the close-confirmed stop in _age is the ONLY stop.
# --------------------------------------------------------------------------- #


def test_entries_carry_no_bracket_and_close_confirmed_stop_still_fires() -> None:
    phase1, phase2 = 30, 4
    n = phase1 + phase2
    series: dict[str, list[dict[str, Any]]] = {}
    flows: dict[str, list[tuple[float, float] | None]] = {}
    # ACC0 crashes ~20% in phase 2 (entry ~100 -> 80), breaching a 10% stop.
    series["ACC0/USDT"] = _flat_bars(100.0, phase1) + _flat_bars(80.0, phase2)
    flows["ACC0/USDT"] = _const_flow(800.0, 200.0, n)
    series["ACC1/USDT"] = _flat_bars(101.0, n)
    flows["ACC1/USDT"] = _const_flow(780.0, 220.0, n)
    series["DIST0/USDT"] = _flat_bars(102.0, n)
    flows["DIST0/USDT"] = _const_flow(200.0, 800.0, n)
    series["DIST1/USDT"] = _flat_bars(103.0, n)
    flows["DIST1/USDT"] = _const_flow(220.0, 780.0, n)
    rng = _lcg(777)
    for fi in range(6):
        sym = f"F{fi}/USDT"
        series[sym] = _flat_bars(90.0 + fi, n)
        tilt = (rng() - 0.5) * 40.0
        flows[sym] = _const_flow(500.0 + tilt, 500.0 - tilt, n)
    strat = _run(series, flows, stop_loss_pct=0.10)
    entries = _non_exit(strat.events.items)
    assert entries, "expected sized entries with stop_loss_pct > 0"
    for sig in entries:
        assert sig.stop_loss is None, (sig.symbol, sig.stop_loss)
    # The close-confirmed stop (aging path) still protects the breached long.
    stop_exits = [
        sig
        for sig in strat.events.items
        if str(sig.signal_type).upper() == "EXIT"
        and dict(sig.metadata or {}).get("reason") == "stop_loss"
    ]
    assert any(sig.symbol == "ACC0/USDT" for sig in stop_exits), stop_exits


# --------------------------------------------------------------------------- #
# WARMUP-GHOST-BOOK regression: on_warmup_end drops ghost positions (the
# engine suppressed warmup signals) while preserving the data deques, and the
# next live rebalance can enter normally.
# --------------------------------------------------------------------------- #


def test_on_warmup_end_flattens_ghost_book_and_preserves_deques() -> None:
    series, flows = _panel()
    bars = _Bars(list(series))
    strat = CrossSectionalResidualTakerFlowStrategy(bars, _Queue(), **_COMMON_KWARGS)
    _feed(strat, bars, series, flows)  # warmup region: entries are ghosts
    assert _non_exit(strat.events.items), "expected ghost entries during warmup"
    assert any(item.mode != "OUT" for item in strat._state.values())
    deques_before = {
        sym: (list(item.closes), list(item.net_flows), list(item.turnovers))
        for sym, item in strat._state.items()
    }
    strat.on_warmup_end()
    for sym, item in strat._state.items():
        assert item.mode == "OUT"
        assert item.entry_price is None
        assert item.bars_held == 0
        assert item.decisions_held == 0
        assert item.score is None
        assert (list(item.closes), list(item.net_flows), list(item.turnovers)) == deques_before[
            sym
        ], sym
    # Next live rebalance re-enters normally from the preserved history.
    signals_before = len(strat.events.items)
    live_series = {sym: rows[:3] for sym, rows in series.items()}
    live_flows = {sym: rows[:3] for sym, rows in flows.items()}
    # continue the bar-key clock past the warmup feed
    n_live = 3
    for idx in range(n_live):
        bars.features = {}
        for sym in live_series:
            print_ = live_flows[sym][idx]
            if print_ is not None:
                buy, sell = print_
                bars.features[(sym, "taker_buy_quote_volume")] = float(buy)
                bars.features[(sym, "taker_sell_quote_volume")] = float(sell)
        strat.calculate_signals(
            _window_event(_N + idx, {sym: live_series[sym][idx] for sym in live_series})
        )
    live_entries = _non_exit(strat.events.items[signals_before:])
    assert live_entries, "post-warmup live rebalance must be able to enter"
    assert _final_side(strat.events.items).get("ACC0/USDT") == "LONG"


# --------------------------------------------------------------------------- #
# STALE-SYMBOL freshness regression: a frozen feed is excluded from the
# cross-sectional ranks and receives no new entries while others advance.
# --------------------------------------------------------------------------- #


def test_frozen_symbol_excluded_from_ranks_and_gets_no_new_entries() -> None:
    phase1, phase2 = 30, 20
    series, flows = _panel()
    bars = _Bars(list(series))
    strat = CrossSectionalResidualTakerFlowStrategy(bars, _Queue(), **_COMMON_KWARGS)
    # Phase 1: everyone feeds; top-flow ACC0 enters LONG.
    for idx in range(phase1):
        bars.features = {}
        for sym in series:
            buy, sell = flows[sym][idx]  # type: ignore[misc]
            bars.features[(sym, "taker_buy_quote_volume")] = float(buy)
            bars.features[(sym, "taker_sell_quote_volume")] = float(sell)
        strat.calculate_signals(_window_event(idx, {sym: series[sym][idx] for sym in series}))
    assert _final_side(strat.events.items).get("ACC0/USDT") == "LONG"
    # Phase 2: ACC0's feed FREEZES (no bars); every other symbol advances and
    # F0 ramps its buy flow so a FRESH name displaces into the long quintile.
    for idx in range(phase1, phase1 + phase2):
        bars.features = {}
        rows: dict[str, dict[str, Any]] = {}
        for sym in series:
            if sym == "ACC0/USDT":
                continue
            if sym == "F0/USDT":
                buy, sell = 900.0, 100.0
            else:
                buy, sell = flows[sym][idx % phase1]  # type: ignore[misc]
            bars.features[(sym, "taker_buy_quote_volume")] = float(buy)
            bars.features[(sym, "taker_sell_quote_volume")] = float(sell)
            rows[sym] = series[sym][idx % phase1]
        strat.calculate_signals(_window_event(idx, rows))
    phase2_signals = [sig for sig in strat.events.items if int(str(sig.datetime)[1:]) >= phase1]
    # The advancing names still rank and trade (the panel is alive)...
    assert _non_exit(phase2_signals), "advancing symbols must still receive entries"
    # ...but the frozen symbol is excluded from ranks: no new entries at all.
    assert not [sig for sig in _non_exit(phase2_signals) if sig.symbol == "ACC0/USDT"], (
        "frozen feed must not receive new entries"
    )
    assert "ACC0/USDT" not in _final_side(strat.events.items)


# --------------------------------------------------------------------------- #
# SLEEVE-STATE-06 sibling regression: a degenerate panel (empty targets) for
# one decision must NOT flush a mature book via rank_lapsed exits.
# --------------------------------------------------------------------------- #


def test_degenerate_panel_decision_does_not_flush_mature_book() -> None:
    series, flows = _panel()
    bars = _Bars(list(series))
    strat = CrossSectionalResidualTakerFlowStrategy(bars, _Queue(), **_COMMON_KWARGS)
    _feed(strat, bars, series, flows)
    book_before = {sym: item.mode for sym, item in strat._state.items() if item.mode != "OUT"}
    assert book_before, "expected a mature book after the panel feed"
    signals_before = len(strat.events.items)
    # Universe-wide feature outage for one decision: only F0 prints taker
    # features, so the fresh panel collapses below min_symbols and targets are
    # empty -- the guard returns instead of emitting rank_lapsed exits.
    bars.features = {
        ("F0/USDT", "taker_buy_quote_volume"): 500.0,
        ("F0/USDT", "taker_sell_quote_volume"): 500.0,
    }
    strat.calculate_signals(_window_event(_N, {"F0/USDT": series["F0/USDT"][0]}))
    new_signals = strat.events.items[signals_before:]
    assert new_signals == [], new_signals
    book_after = {sym: item.mode for sym, item in strat._state.items() if item.mode != "OUT"}
    assert book_after == book_before
    # Direct degenerate-decision leg (pins the guard itself, independent of the
    # freshness gate): a decision whose ranker returns EMPTY targets must not
    # flush the mature book via rank_lapsed exits.
    strat._score_and_select = lambda: ({}, {})  # type: ignore[method-assign]
    strat._evaluate("t9999")
    assert strat.events.items[signals_before:] == []
    book_final = {sym: item.mode for sym, item in strat._state.items() if item.mode != "OUT"}
    assert book_final == book_before
