"""Deterministic BUILD-GATE tests for OpenInterestGrowthPressureStrategy.

Direct class import (registry/tier/candidate-wiring assertions land with the
orchestrator's integration wave, not here).  The fixture is a hand-built,
closed-form stream of daily UTC bars (no RNG anywhere): one MARKET event per
symbol per day at 00:00 UTC starting Monday 2025-01-06, with ``open_interest``
and ``mark_price`` carried as event attributes (tier-1 of the shared
None-tolerant feature cascade).

Panel design for the sign-lock: every symbol shares the SAME mildly-oscillating
price path (identical momentum and identical realized vol -> the momentum
regressor is cross-sectionally degenerate and momentum can explain nothing),
while the OI paths diverge: RISER's OI grows steadily, FALLER's OI drains, and
the fillers hold flat OI.  Pre-registered POSITIVE continuation -> RISER must be
LONG and FALLER must be SHORT (no flip).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.oi_growth_pressure_alpha_sleeves import (
    DELTA_OI_WINDOW_DAYS,
    DELTA_OI_WINDOW_DAYS_FALLBACK,
    INTENDED_HOLD_SECONDS,
    OpenInterestGrowthPressureStrategy,
    _DayRecord,
    _OI_GROWTH_PRESSURE_SLICE,
)

_START = datetime(2025, 1, 6, tzinfo=UTC)  # Monday
_N_DAYS = 28

RISER = "BTC/USDT"
FALLER = "ETH/USDT"
FILLERS = ["SOL/USDT", "TSLAUSDT", "NVDAUSDT", "QQQUSDT"]
_ALL_SYMBOLS = [RISER, FALLER, *FILLERS]


def _price(day: int) -> float:
    """Shared closed-form price path: mild oscillation, identical per symbol.

    Exactly 7-day periodic (``day % 7``), so EVERY 7d momentum window is
    bit-zero regardless of the one-day offset between the symbol that triggers
    the weekly evaluation and the symbols that have not yet printed that bar --
    the momentum regressor is degenerate by construction and the residual is
    the pure OI-pressure axis.
    """
    return 100.0 * (1.0 + 0.002 * math.sin(float(day % 7)))


def _open_interest(symbol: str, day: int) -> float | None:
    if symbol == RISER:
        return 1_000.0 + 150.0 * day
    if symbol == FALLER:
        return 6_000.0 - 150.0 * day
    # Distinct flat levels so nothing ties bit-exactly.
    return 2_000.0 + 10.0 * FILLERS.index(symbol)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _event(
    symbol: str,
    moment: datetime,
    close: float,
    *,
    open_interest: float | None,
    mark_price: float | None,
    volume: float = 1_000.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=moment.isoformat().replace("+00:00", "Z"),
        symbol=symbol,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=volume,
        open_interest=open_interest,
        mark_price=mark_price,
    )


def _feed_panel(
    strategy: Any,
    symbols: list[str],
    *,
    n_days: int = _N_DAYS,
    start_day: int = 0,
    oi_fn: Any = _open_interest,
    price_fn: Any = None,
    mark_from_price: bool = False,
) -> None:
    for day in range(start_day, n_days):
        moment = _START + timedelta(days=day)
        for symbol in symbols:
            close = price_fn(symbol, day) if price_fn is not None else _price(day)
            oi = oi_fn(symbol, day)
            mark = close if mark_from_price else 100.0
            strategy.calculate_signals(
                _event(symbol, moment, close, open_interest=oi, mark_price=mark)
            )


def _final_sides(items: list[Any]) -> dict[str, str]:
    sides: dict[str, str] = {}
    for signal in items:
        kind = str(signal.signal_type).upper()
        if kind in {"LONG", "SHORT"}:
            sides[signal.symbol] = kind
        elif kind == "EXIT":
            sides.pop(signal.symbol, None)
    return sides


def _make_candidate(symbols: list[str] | None = None, **overrides: Any) -> Any:
    params: dict[str, Any] = dict(
        delta_oi_window_days=DELTA_OI_WINDOW_DAYS,
        quantile_pct=0.20,
        min_symbols=5,
        min_history_days=DELTA_OI_WINDOW_DAYS + 1,
        min_oi_coverage=0.80,
        min_hold_decisions=2,
        rank_hysteresis_buffer=1,
        vol_window=5,
        allow_short=True,
    )
    params.update(overrides)
    bars = SimpleNamespace(symbol_list=list(symbols if symbols is not None else _ALL_SYMBOLS))
    return OpenInterestGrowthPressureStrategy(bars, _Queue(), **params)


# --------------------------------------------------------------------------- #
# pre-registration constants are declared, not tuned
# --------------------------------------------------------------------------- #


def test_pre_registered_constants_and_required_features() -> None:
    assert DELTA_OI_WINDOW_DAYS == 7
    assert DELTA_OI_WINDOW_DAYS_FALLBACK == 14  # declared fallback, module constant
    assert INTENDED_HOLD_SECONDS == 14 * 86_400  # ~2 weeks
    assert OpenInterestGrowthPressureStrategy.required_features == (
        "open_interest",
        "mark_price",
    )


# --------------------------------------------------------------------------- #
# sign-lock: POSITIVE continuation on the synthetic feature panel (no flip)
# --------------------------------------------------------------------------- #


def test_sign_lock_rising_oi_long_falling_oi_short() -> None:
    candidate = _make_candidate()
    _feed_panel(candidate, _ALL_SYMBOLS)
    sides = _final_sides(candidate.events.items)
    assert sides.get(RISER) == "LONG"
    assert sides.get(FALLER) == "SHORT"
    # Entries stamp the ex-ante ~2-week intended hold.
    entries = [
        signal
        for signal in candidate.events.items
        if str(signal.signal_type).upper() in {"LONG", "SHORT"}
    ]
    assert entries
    for signal in entries:
        metadata = signal.metadata or {}
        assert metadata.get("intended_hold_seconds") == INTENDED_HOLD_SECONDS
        assert float(metadata.get("target_allocation", 0.0)) > 0.0


def test_residual_scores_rank_riser_above_faller() -> None:
    candidate = _make_candidate()
    _feed_panel(candidate, _ALL_SYMBOLS)
    residual, vols, metas = candidate._residual_scores()
    assert residual, "the OI-covered panel must produce residual scores"
    assert residual[RISER] > 0.0 > residual[FALLER]
    assert residual[RISER] == max(residual.values())
    assert residual[FALLER] == min(residual.values())
    assert set(vols) == set(residual)
    for meta in metas.values():
        assert meta["oi_coverage"] >= 0.80


# --------------------------------------------------------------------------- #
# momentum-residualization lock: pure-momentum panel with flat OI emits nothing
# --------------------------------------------------------------------------- #


def test_pure_momentum_panel_with_flat_oi_notional_emits_nothing() -> None:
    def _drift_price(symbol: str, day: int) -> float:
        drift = 0.001 * (1 + _ALL_SYMBOLS.index(symbol))
        return 100.0 * math.exp(drift * day) * (1.0 + 0.001 * math.sin(day / 2.0))

    def _flat_oi(_symbol: str, _day: int) -> float:
        return 5_000.0

    candidate = _make_candidate()
    # mark_price held constant too, so OI NOTIONAL is bit-flat while price trends.
    _feed_panel(candidate, _ALL_SYMBOLS, oi_fn=_flat_oi, price_fn=_drift_price)
    entries = [
        signal
        for signal in candidate.events.items
        if str(signal.signal_type).upper() in {"LONG", "SHORT"}
    ]
    assert not entries, "flat OI notional must never trade, whatever momentum does"


# --------------------------------------------------------------------------- #
# in-window coverage floor -> deterministic exclusion
# --------------------------------------------------------------------------- #


def test_sparse_oi_symbol_is_excluded_from_the_rank() -> None:
    candidate = _make_candidate(min_symbols=2)
    window_len = DELTA_OI_WINDOW_DAYS + 1
    # Hand-seed finalized day records: DENSE has full OI coverage, SPARSE has
    # OI on only 3 of 8 window days (coverage 0.375 < 0.80 floor).
    dense, sparse, third = _ALL_SYMBOLS[0], _ALL_SYMBOLS[1], _ALL_SYMBOLS[2]
    for day in range(window_len + 2):
        close = _price(day)
        candidate._state[dense].days.append(
            _DayRecord(oi_notional=1_000.0 + 50.0 * day, dollar_volume=100_000.0, close=close)
        )
        candidate._state[sparse].days.append(
            _DayRecord(
                oi_notional=(2_000.0 + 40.0 * day) if day % 3 == 0 else None,
                dollar_volume=100_000.0,
                close=close,
            )
        )
        candidate._state[third].days.append(
            _DayRecord(oi_notional=3_000.0 - 20.0 * day, dollar_volume=100_000.0, close=close)
        )
    for symbol in (dense, sparse, third):
        for day in range(8):
            candidate._state[symbol].closes.append(_price(day))
    residual, _vols, _metas = candidate._residual_scores()
    assert dense in residual
    assert third in residual
    assert sparse not in residual  # deterministic coverage-floor exclusion


# --------------------------------------------------------------------------- #
# never-raise on None / degenerate / adversarial inputs
# --------------------------------------------------------------------------- #


def test_degenerate_events_never_raise() -> None:
    candidate = _make_candidate()
    for event in (
        SimpleNamespace(type="MARKET", time=None, symbol=RISER, close=None),
        SimpleNamespace(type="MARKET", time="not-a-date", symbol=RISER, close=100.0),
        SimpleNamespace(
            type="MARKET",
            time="2025-01-06T00:00:00Z",
            symbol=RISER,
            close=float("nan"),
            open_interest=float("inf"),
            mark_price=-1.0,
        ),
        SimpleNamespace(
            type="MARKET",
            time="2025-01-06T00:00:00Z",
            symbol=RISER,
            close=100.0,
            open_interest={"bad": "payload"},
            mark_price="junk",
            volume=None,
        ),
        SimpleNamespace(type="MARKET_WINDOW", time="2025-01-06T00:00:00Z", bars_1s={}),
        SimpleNamespace(type="OTHER"),
    ):
        candidate.calculate_signals(event)  # must never raise


def test_missing_features_degrade_to_exclusion_not_raise() -> None:
    def _no_oi(_symbol: str, _day: int) -> float | None:
        return None

    candidate = _make_candidate()
    _feed_panel(candidate, _ALL_SYMBOLS, oi_fn=_no_oi)
    entries = [
        signal
        for signal in candidate.events.items
        if str(signal.signal_type).upper() in {"LONG", "SHORT"}
    ]
    assert not entries  # every symbol below the coverage floor -> whole book flat


def test_adversarial_set_state_never_raises() -> None:
    candidate = _make_candidate()
    for junk in (
        None,
        {},
        [],
        {"last_eval_week": "bad", "symbol_state": 5},
        {"symbol_state": {RISER: {"closes": ["x", None], "days": [[1], "y", 3]}}},
        {"symbol_state": {RISER: {"mode": 123, "decisions_held": -9, "cur_close": "z"}}},
        {"symbol_state": {RISER: {"days": [[None, "bad", None], [1.0, 2.0, 3.0]]}}},
        {"last_eval_week": [1], "symbol_state": {"UNKNOWN": {"closes": [1.0]}}},
        {"recent_times": "nope", "symbol_state": {RISER: {"cur_oi_notional": object()}}},
    ):
        candidate.set_state(junk)  # must never raise


# --------------------------------------------------------------------------- #
# state roundtrip (lossless)
# --------------------------------------------------------------------------- #


def test_state_roundtrip_is_lossless_mid_position() -> None:
    candidate = _make_candidate()
    _feed_panel(candidate, _ALL_SYMBOLS)
    snapshot = candidate.get_state()
    revived = _make_candidate()
    revived.set_state(snapshot)
    assert revived.get_state() == snapshot


# --------------------------------------------------------------------------- #
# v5 zero-alloc gate: zero/negative computed allocation emits NOTHING
# --------------------------------------------------------------------------- #


def test_zero_alloc_entry_skipped_not_default_sized() -> None:
    flip_sym, fresh_sym = _ALL_SYMBOLS[0], _ALL_SYMBOLS[1]
    strat = _make_candidate()
    flip = strat._state[flip_sym]
    flip.mode = "LONG"
    flip.entry_price = 100.0
    flip.decisions_held = 10_000  # clear the min-hold so the side flip is live
    flip.closes.append(100.0)
    targets = {
        flip_sym: ("SHORT", -1.0, {}),
        fresh_sym: ("LONG", 1.0, {}),
    }
    # empty weights -> alloc == 0 for every targeted symbol
    strat._emit_targets(targets, {}, 1.0, "2026-01-01T00:00:00Z")
    kinds = [(sig.symbol, str(sig.signal_type).upper()) for sig in strat.events.items]
    assert not [sym for sym, kind in kinds if kind in {"LONG", "SHORT"}], kinds
    # the side-flip EXIT still fired and the flipped symbol is now flat
    assert (flip_sym, "EXIT") in kinds
    assert strat._state[flip_sym].mode == "OUT"
    assert strat._state[flip_sym].entry_price is None

    # a positive inverse-vol weight DOES emit a sized entry with intended hold
    sized = _make_candidate()
    sized._emit_targets(
        {fresh_sym: ("LONG", 1.0, {})}, {fresh_sym: 0.5}, 1.0, "2026-01-01T00:00:00Z"
    )
    entries = [
        sig for sig in sized.events.items if str(sig.signal_type).upper() in {"LONG", "SHORT"}
    ]
    assert entries
    for sig in entries:
        metadata = sig.metadata or {}
        assert float(metadata.get("target_allocation", 0.0)) > 0.0
        assert metadata.get("intended_hold_seconds") == INTENDED_HOLD_SECONDS


# --------------------------------------------------------------------------- #
# min-hold suppression
# --------------------------------------------------------------------------- #


def test_min_hold_suppresses_flip_until_the_hold_clears() -> None:
    def _seed(min_hold: int) -> tuple[Any, Any]:
        strategy = _make_candidate(min_hold_decisions=min_hold)
        item = strategy._state[RISER]
        item.mode = "LONG"
        item.entry_price = 100.0
        item.decisions_held = 0
        item.closes.append(100.0)
        return strategy, item

    targets = {RISER: ("SHORT", -1.5, {})}
    weights = {RISER: 1.0}

    held, held_item = _seed(min_hold=2)
    held._emit_targets(targets, weights, 1.0, "2026-01-01T00:00:00Z")
    assert held_item.mode == "LONG"  # flip suppressed inside the hold window

    flips, flip_item = _seed(min_hold=0)
    flips._emit_targets(targets, weights, 1.0, "2026-01-01T00:00:00Z")
    assert flip_item.mode == "SHORT"  # min_hold=0 reference flips immediately


# --------------------------------------------------------------------------- #
# two-run determinism
# --------------------------------------------------------------------------- #


def test_run_twice_bit_identical() -> None:
    first = _make_candidate()
    _feed_panel(first, _ALL_SYMBOLS)
    second = _make_candidate()
    _feed_panel(second, _ALL_SYMBOLS)
    a = [(s.symbol, s.signal_type, round(float(s.strength), 12)) for s in first.events.items]
    b = [(s.symbol, s.signal_type, round(float(s.strength), 12)) for s in second.events.items]
    assert a == b
    assert a, "the determinism check must compare a non-empty signal stream"


# --------------------------------------------------------------------------- #
# min-symbols / short-history self-skip
# --------------------------------------------------------------------------- #


def test_self_skips_below_min_symbols() -> None:
    candidate = _make_candidate(symbols=[RISER, FALLER], min_symbols=5)
    _feed_panel(candidate, [RISER, FALLER])
    assert not [
        s for s in candidate.events.items if str(s.signal_type).upper() in {"LONG", "SHORT"}
    ]


def test_self_skips_on_short_history() -> None:
    candidate = _make_candidate()
    # 8 days spans one ISO-week boundary but finalizes only 7 day records --
    # below min_history_days (8), so the first weekly evaluation must abstain.
    _feed_panel(candidate, _ALL_SYMBOLS, n_days=8)
    assert not [
        s for s in candidate.events.items if str(s.signal_type).upper() in {"LONG", "SHORT"}
    ]


# --------------------------------------------------------------------------- #
# weekly ISO clock (not per-bar churn)
# --------------------------------------------------------------------------- #


def test_min_hold_re_pinned_to_two_weekly_decisions() -> None:
    """OI-MINHOLD-UNIT-MISMATCH: 2 weekly decisions ~= 2-3 weeks, matching the
    ~2-week INTENDED_HOLD_SECONDS and the registered 1-4 week horizon."""
    schema = OpenInterestGrowthPressureStrategy.get_param_schema()
    assert schema["min_hold_decisions"].default == 2
    for variants in _OI_GROWTH_PRESSURE_SLICE.values():
        for variant in variants:
            assert variant["min_hold_decisions"] == 2, variant["variant"]
    assert INTENDED_HOLD_SECONDS == 14 * 86_400  # NOT re-pinned by the fix


# --------------------------------------------------------------------------- #
# SLEEVE-STATE-06: universe-wide coverage failure must not flush a mature book
# --------------------------------------------------------------------------- #


def test_universe_wide_oi_outage_week_does_not_flush_mature_book() -> None:
    """One ISO week of universe-wide ``open_interest=None`` empties the panel;
    the sleeve must return WITHOUT rank_lapsed exits, holding the mature book."""

    def _oi_with_outage(symbol: str, day: int) -> float | None:
        return None if day >= _N_DAYS else _open_interest(symbol, day)

    candidate = _make_candidate()
    # Days 0..27 build the book; days 28..34 are the None-OI ISO week; the
    # day-35 bar (Monday W07) triggers the post-outage weekly evaluation.
    _feed_panel(candidate, _ALL_SYMBOLS, n_days=_N_DAYS + 8, oi_fn=_oi_with_outage)
    sides = _final_sides(candidate.events.items)
    assert sides.get(RISER) == "LONG"
    assert sides.get(FALLER) == "SHORT"
    assert candidate._state[RISER].mode == "LONG"
    assert candidate._state[FALLER].mode == "SHORT"
    exits = [s for s in candidate.events.items if str(s.signal_type).upper() == "EXIT"]
    assert not exits, "an empty residual panel must never emit rank_lapsed exits"


# --------------------------------------------------------------------------- #
# freshness gate: a stale symbol (dead feed) is excluded from the rank
# --------------------------------------------------------------------------- #


def test_stale_symbol_is_excluded_from_the_rank() -> None:
    """A symbol whose last committed day predates the strategy-level latest
    committed day (feed died mid-history) must be excluded from the panel even
    though its own (old) formation window has full OI coverage."""
    stale = FILLERS[0]
    fresh = [symbol for symbol in _ALL_SYMBOLS if symbol != stale]
    candidate = _make_candidate()
    for day in range(_N_DAYS):
        moment = _START + timedelta(days=day)
        for symbol in _ALL_SYMBOLS:
            if symbol == stale and day >= 13:
                continue  # the stale feed dies after day 12
            close = _price(day)
            candidate.calculate_signals(
                _event(
                    symbol,
                    moment,
                    close,
                    open_interest=_open_interest(symbol, day),
                    mark_price=100.0,
                )
            )
    stale_state = candidate._state[stale]
    # The stale symbol WOULD be panel-eligible on history + coverage alone --
    # only the freshness gate keeps its old window out of the rank.
    window_len = DELTA_OI_WINDOW_DAYS + 1
    assert len(stale_state.days) >= candidate.min_history_days
    assert all(record.oi_notional is not None for record in list(stale_state.days)[-window_len:])
    assert stale_state.last_committed_day < candidate._last_committed_day
    residual, vols, _metas = candidate._residual_scores()
    assert stale not in residual
    assert stale not in vols
    assert set(residual) == set(fresh)
    # No entry was ever emitted for the stale symbol either.
    assert not [
        s
        for s in candidate.events.items
        if s.symbol == stale and str(s.signal_type).upper() in {"LONG", "SHORT"}
    ]


# --------------------------------------------------------------------------- #
# WARMUP-GHOST-BOOK: on_warmup_end flattens the ghost book, keeps history
# --------------------------------------------------------------------------- #


def test_on_warmup_end_flattens_ghost_book_and_preserves_history() -> None:
    candidate = _make_candidate()
    _feed_panel(candidate, _ALL_SYMBOLS)
    assert candidate._state[RISER].mode == "LONG"  # a warmup-built ghost book
    before = {
        symbol: (
            list(item.closes),
            [[r.oi_notional, r.dollar_volume, r.close] for r in item.days],
            item.cur_date,
            item.cur_oi_notional,
            item.cur_dollar_volume,
            item.cur_close,
            item.last_committed_day,
        )
        for symbol, item in candidate._state.items()
    }
    hook = getattr(candidate, "on_warmup_end", None)
    assert callable(hook)  # engine fires it via getattr + callable
    hook()
    for symbol, item in candidate._state.items():
        assert item.mode == "OUT"
        assert item.entry_price is None
        assert item.decisions_held == 0
        assert item.score is None
        assert (
            list(item.closes),
            [[r.oi_notional, r.dollar_volume, r.close] for r in item.days],
            item.cur_date,
            item.cur_oi_notional,
            item.cur_dollar_volume,
            item.cur_close,
            item.last_committed_day,
        ) == before[symbol], symbol  # OI/close history fully preserved
    # The preserved history keeps working: the next weekly clock re-enters.
    marker = len(candidate.events.items)
    _feed_panel(candidate, _ALL_SYMBOLS, n_days=_N_DAYS + 8, start_day=_N_DAYS)
    fresh_entries = {
        signal.symbol: str(signal.signal_type).upper()
        for signal in candidate.events.items[marker:]
        if str(signal.signal_type).upper() in {"LONG", "SHORT"}
    }
    assert fresh_entries.get(RISER) == "LONG"
    assert fresh_entries.get(FALLER) == "SHORT"


def test_decision_clock_is_weekly_iso_not_every_bar() -> None:
    candidate = _make_candidate()
    calls = {"n": 0}
    original = candidate._evaluate

    def _counting(event_time: Any) -> None:
        calls["n"] += 1
        original(event_time)

    candidate._evaluate = _counting  # type: ignore[method-assign]
    _feed_panel(candidate, _ALL_SYMBOLS)
    weeks = len({(_START + timedelta(days=day)).isocalendar()[:2] for day in range(_N_DAYS)})
    assert calls["n"] == weeks - 1  # one decision per NEW ISO week, seed week excluded
    assert calls["n"] < _N_DAYS  # far below the per-bar cadence
