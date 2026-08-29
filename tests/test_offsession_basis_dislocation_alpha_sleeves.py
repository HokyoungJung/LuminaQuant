"""Deterministic BUILD-GATE tests for OffSessionBasisDislocationStrategy.

Direct class import only (tier hints / candidate wiring land with the
integration wave, so there are no registry-resolution assertions here).

The fixture is a hand-built 254-hour stream of UTC-stamped 1h bars from Monday
2025-01-06 00:00 across ten TradFi legs plus two crypto legs.  Every bar
carries ``mark_price == close``; ``index_price == close`` at all times EXCEPT
Friday 2025-01-10 20:00 through Monday 2025-01-13 13:00, where the index is
FROZEN at its Friday-20:00 value (the stale off-session anchor).  Over the
weekend one leg (PUMP) drifts +2% and one (DUMP) drifts -2% while the index is
frozen, so their accumulated off-session basis dislocations are ~+200bps /
~-200bps against eight noise-only fillers: the Monday pre-reopen decision must
SHORT the pumped leg and LONG the dumped leg (sign-lock), and the fixed 72h
max exit must flatten both on Thursday.

Weeknights have ``mark == index`` exactly, so every pre-weekend decision sees
an all-zero score vector (degenerate cross-section -> no entries): calendar
time alone never generates a position.

Noise comes from a seeded inline LCG (no ``random`` module).  ASCII only.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from lumina_quant.strategies.offsession_basis_dislocation_alpha_sleeves import (
    _DEFAULT_TRADFI_UNIVERSE,
    _OFFSESSION_BASIS_DISLOCATION_SLICE,
    _SUGGESTED_CANDIDATE_TAGS,
    OffSessionBasisDislocationStrategy,
)

# --------------------------------------------------------------------------- #
# fixture parameters
# --------------------------------------------------------------------------- #

_START = datetime(2025, 1, 6, tzinfo=UTC)  # Monday
_N_STEPS = 254  # through Thursday 2025-01-16 13:00
_FREEZE_START = 116  # Friday 2025-01-10 20:00 UTC (index goes stale)
_FREEZE_END = 181  # Monday 2025-01-13 13:00 UTC (last stale bar)
_WEEKEND_DRIFT_STEPS = frozenset(range(120, 168))  # Sat 00:00 .. Sun 23:00

PUMP = "TSLAUSDT"
DUMP = "NVDAUSDT"
FILLERS = [
    "AAPLUSDT",
    "MSFTUSDT",
    "AMZNUSDT",
    "METAUSDT",
    "GOOGLUSDT",
    "TSMUSDT",
    "AMDUSDT",
    "ORCLUSDT",
]
_TRADFI = [PUMP, DUMP, *FILLERS]
_CRYPTO = ["BTC/USDT", "ETHUSDT"]
_ALL_SYMBOLS = [*_TRADFI, *_CRYPTO]

_MONDAY_DECISION = datetime(2025, 1, 13, 12, tzinfo=UTC)


def _lcg_stream(seed: int) -> Any:
    """Seeded inline LCG in [0, 1) -- the only noise source in this file."""
    state = (seed * 2654435761 + 12345) & 0x7FFFFFFF

    def _next() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / float(0x7FFFFFFF)

    return _next


def _build_panel() -> tuple[list[datetime], dict[str, list[float]], dict[str, list[float]]]:
    times = [_START + timedelta(hours=step) for step in range(_N_STEPS)]
    closes: dict[str, list[float]] = {}
    indexes: dict[str, list[float]] = {}
    for order, symbol in enumerate(_ALL_SYMBOLS):
        rand = _lcg_stream(97 + order * 31)
        price = 100.0 + 3.0 * order
        series: list[float] = []
        for step in range(_N_STEPS):
            ret = 0.0002 * (rand() - 0.5)  # +/-1bp hourly wiggle
            if symbol == PUMP and step in _WEEKEND_DRIFT_STEPS:
                ret += 0.02 / 48.0  # +2% cumulative over the frozen weekend
            elif symbol == DUMP and step in _WEEKEND_DRIFT_STEPS:
                ret -= 0.02 / 48.0
            price *= math.exp(ret)
            series.append(price)
        closes[symbol] = series
        idx_series = list(series)
        frozen = series[_FREEZE_START]
        for step in range(_FREEZE_START, min(_FREEZE_END + 1, _N_STEPS)):
            idx_series[step] = frozen
        indexes[symbol] = idx_series
    return times, closes, indexes


_TIMES, _CLOSES, _INDEXES = _build_panel()


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Bars:
    """Feature-store stand-in for complete MARKET_WINDOW observations."""

    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)
        self.features: dict[tuple[str, str], float] = {}

    def get_latest_feature_value(self, symbol: str, field: str) -> float | None:
        return self.features.get((symbol, field))


def _event(symbol: str, moment: datetime, close: float, index: float) -> SimpleNamespace:
    return SimpleNamespace(
        type="MARKET",
        time=moment.isoformat().replace("+00:00", "Z"),
        symbol=symbol,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=1000.0,
        mark_price=close,
        index_price=index,
    )


def _window(
    candidate: Any,
    moment: datetime,
    symbols: list[str],
    *,
    row_times: dict[str, datetime] | None = None,
) -> SimpleNamespace:
    """Build one aligned atomic panel and populate its per-symbol features."""
    bars = candidate.bars
    assert isinstance(bars, _Bars)
    rows: dict[str, list[dict[str, Any]]] = {}
    step = _TIMES.index(moment)
    for symbol in symbols:
        row_time = (row_times or {}).get(symbol, moment)
        close = _CLOSES[symbol][step]
        index = _INDEXES[symbol][step]
        bars.features[(symbol, "mark_price")] = close
        bars.features[(symbol, "index_price")] = index
        rows[symbol] = [
            {
                "time": row_time.isoformat().replace("+00:00", "Z"),
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1000.0,
            }
        ]
    return SimpleNamespace(
        type="MARKET_WINDOW",
        time=moment.isoformat().replace("+00:00", "Z"),
        bars_1s=rows,
    )


def _feed(strategy: Any, symbols: list[str], end_step: int = _N_STEPS) -> None:
    for step in range(min(end_step, _N_STEPS)):
        moment = _TIMES[step]
        strategy.calculate_signals(_window(strategy, moment, symbols))


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
        tradfi_symbols=tuple(_TRADFI),
        entry_z=2.0,
        quantile_pct=0.20,
        min_symbols=4,
        min_off_observations=3,
        min_abs_dislocation_bps=10.0,
        min_hold_hours=36,
        max_hold_hours=72,
        vol_window=48,
        allow_short=True,
    )
    params.update(overrides)
    bars = _Bars(list(symbols if symbols is not None else _ALL_SYMBOLS))
    return OffSessionBasisDislocationStrategy(bars, _Queue(), **params)


def _seed_eligible(strategy: Any, accum_by_symbol: dict[str, float]) -> None:
    """Inject eligible per-symbol state (off_obs, dislocation, vol-able closes)."""
    for symbol, accum in accum_by_symbol.items():
        item = strategy._state[symbol]
        item.off_obs = 5
        item.off_accum_bps = float(accum)
        rand = _lcg_stream(7 + sum(ord(ch) for ch in symbol))
        price = 100.0
        for _ in range(60):
            price *= math.exp(0.0004 * (rand() - 0.5))
            item.closes.append(price)


# --------------------------------------------------------------------------- #
# stage 1: fixture properties + mirrored session classifier
# --------------------------------------------------------------------------- #


def test_stage1_classifier_mirrors_tugofwar_boundary_convention() -> None:
    candidate = _make_candidate()
    # The DST-ambiguous boundary hours {start-1, start, end} = {13, 14, 20}.
    assert candidate._ambiguous_hours == frozenset({13, 14, 20})
    # Last decision bar before cash reopen: the last non-ambiguous pre-open hour.
    assert candidate._decision_hour_utc == 12
    assert candidate._classify_hour(datetime(2025, 1, 11, 15, tzinfo=UTC)) == "OFF"  # Saturday
    assert candidate._classify_hour(datetime(2025, 1, 8, 13, tzinfo=UTC)) == "AMBIGUOUS"
    assert candidate._classify_hour(datetime(2025, 1, 8, 14, tzinfo=UTC)) == "AMBIGUOUS"
    assert candidate._classify_hour(datetime(2025, 1, 8, 20, tzinfo=UTC)) == "AMBIGUOUS"
    assert candidate._classify_hour(datetime(2025, 1, 8, 15, tzinfo=UTC)) == "CASH"
    assert candidate._classify_hour(datetime(2025, 1, 8, 12, tzinfo=UTC)) == "OFF"


def test_stage1_weekend_dislocation_accumulates_in_state() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS, end_step=180)  # through Monday 11:00
    pump = candidate._state[PUMP]
    dump = candidate._state[DUMP]
    assert pump.off_accum_bps > 150.0, pump.off_accum_bps
    assert dump.off_accum_bps < -150.0, dump.off_accum_bps
    for symbol in FILLERS:
        assert abs(candidate._state[symbol].off_accum_bps) < 10.0
    # Weekend bars accrued: off_obs is well past the eligibility floor.
    assert pump.off_obs >= 40


def test_partial_or_skewed_windows_fail_closed_until_aligned_panel_arrives() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS, end_step=180)  # through Monday 11:00
    pump = candidate._state[PUMP]
    closes_before = list(pump.closes)
    decision_before = candidate._last_decision_date

    # Missing one configured TradFi leg at the decision timestamp is not a
    # usable cross-section; it cannot age state, score, or emit an order.
    candidate.calculate_signals(_window(candidate, _TIMES[180], _TRADFI[:-1]))
    assert candidate.events.items == []
    assert candidate._last_decision_date == decision_before
    assert list(pump.closes) == closes_before

    # Every leg must also carry the event's exact timestamp; cached/skewed
    # rows fail before mutating the panel.
    candidate.calculate_signals(
        _window(candidate, _TIMES[180], _TRADFI, row_times={PUMP: _TIMES[179]})
    )
    assert candidate.events.items == []
    assert candidate._last_decision_date == decision_before
    assert list(pump.closes) == closes_before

    candidate.calculate_signals(_window(candidate, _TIMES[180], _TRADFI))
    sides = _final_sides(candidate.events.items)
    assert sides == {PUMP: "SHORT", DUMP: "LONG"}


# --------------------------------------------------------------------------- #
# sign-lock: SHORT the pumped-off-anchor leg, LONG the dumped leg
# --------------------------------------------------------------------------- #


def test_sign_lock_shorts_pumped_longs_dumped_at_pre_reopen_decision() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS, end_step=182)  # through Monday 13:00
    sides = _final_sides(candidate.events.items)
    assert sides.get(PUMP) == "SHORT"
    assert sides.get(DUMP) == "LONG"
    for symbol in FILLERS:
        assert symbol not in sides  # |z| ~ 0 fillers never enter
    for symbol in _CRYPTO:
        assert symbol not in candidate.symbol_list  # universe-filtered out
        assert all(signal.symbol != symbol for signal in candidate.events.items)


def test_entry_metadata_stamps_intended_hold_and_allocation() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS, end_step=182)
    entries = [
        signal
        for signal in candidate.events.items
        if str(signal.signal_type).upper() in {"LONG", "SHORT"}
    ]
    assert entries
    for signal in entries:
        metadata = dict(signal.metadata or {})
        assert float(metadata.get("intended_hold_seconds", 0.0)) >= 36 * 3600.0
        assert float(metadata.get("intended_hold_seconds", 0.0)) == 72 * 3600.0
        assert float(metadata.get("target_allocation", 0.0)) > 0.0
        assert float(metadata.get("max_order_value", 0.0)) > 0.0
        assert "dislocation_z" in metadata
        assert "off_basis_accum_bps" in metadata


def test_no_entry_on_calm_weekdays_calendar_alone_never_trades() -> None:
    # Every pre-weekend decision (mark == index -> all-zero scores) stays flat:
    # the session calendar only defines the measurement window.
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS, end_step=116)  # Mon-Fri incl. 5 decisions
    assert not [
        s for s in candidate.events.items if str(s.signal_type).upper() in {"LONG", "SHORT"}
    ]


def test_individual_market_events_learn_but_never_rebalance() -> None:
    candidate = _make_candidate()
    for symbol in _TRADFI:
        candidate.calculate_signals(
            _event(symbol, _TIMES[180], _CLOSES[symbol][180], _INDEXES[symbol][180])
        )
    assert candidate._state[PUMP].closes
    assert candidate._last_decision_date is None
    assert candidate.events.items == []


# --------------------------------------------------------------------------- #
# decision clock: once per weekday at the pre-reopen hour, never on weekends
# --------------------------------------------------------------------------- #


def test_decision_fires_once_per_weekday_pre_reopen_bar() -> None:
    candidate = _make_candidate()
    calls: list[Any] = []
    original = candidate._evaluate

    def _counting(event_time: Any) -> None:
        calls.append(event_time)
        original(event_time)

    candidate._evaluate = _counting  # type: ignore[method-assign]
    _feed(candidate, _ALL_SYMBOLS)
    weekday_noon_bars = [moment for moment in _TIMES if moment.weekday() < 5 and moment.hour == 12]
    assert len(calls) == len(weekday_noon_bars) == 9
    # Sat/Sun noon bars exist in the tape but never trigger a decision.
    fired_days = {str(t)[:10] for t in calls}
    assert "2025-01-11" not in fired_days and "2025-01-12" not in fired_days


# --------------------------------------------------------------------------- #
# holding discipline: hard 36h min-hold, fixed 72h max exit
# --------------------------------------------------------------------------- #


def test_min_hold_suppresses_side_flip_until_36h() -> None:
    candidate = _make_candidate()
    item = candidate._state[PUMP]
    item.closes.append(100.0)
    item.mode = "LONG"
    item.entry_price = 100.0
    item.entry_epoch = _MONDAY_DECISION.timestamp()
    targets = {PUMP: ("SHORT", -2.5, {})}
    weights = {PUMP: 1.0}
    # +1h: inside the 36h window -> the flip is suppressed, nothing is emitted.
    candidate._emit_targets(targets, weights, 1.0, "2025-01-13T13:00:00Z")
    assert item.mode == "LONG"
    assert not candidate.events.items
    # +37h: past the min-hold -> close-confirmed EXIT then the flipped entry.
    candidate._emit_targets(targets, weights, 1.0, "2025-01-15T01:00:00Z")
    kinds = [str(s.signal_type).upper() for s in candidate.events.items]
    assert kinds == ["EXIT", "SHORT"]
    assert item.mode == "SHORT"
    assert item.entry_epoch == datetime(2025, 1, 15, 1, tzinfo=UTC).timestamp()


def test_fixed_72h_max_exit_flattens_the_book_close_confirmed() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS)  # through Thursday 13:00
    exits = [
        s
        for s in candidate.events.items
        if str(s.signal_type).upper() == "EXIT"
        and dict(s.metadata or {}).get("reason") == "max_hold"
    ]
    assert {s.symbol for s in exits} == {PUMP, DUMP}
    # Exit fires on the first completed bar at/past the 72h ceiling (Thu 12:00).
    for signal in exits:
        exited = datetime.fromisoformat(str(signal.datetime).replace("Z", "+00:00"))
        age = (exited - _MONDAY_DECISION).total_seconds()
        assert age >= 72 * 3600.0
        assert age <= 73 * 3600.0
    assert candidate._state[PUMP].mode == "OUT"
    assert candidate._state[DUMP].mode == "OUT"
    assert _final_sides(candidate.events.items) == {}


# --------------------------------------------------------------------------- #
# entry gates: |z| > 2 quintile gate and the absolute-dislocation noise floor
# --------------------------------------------------------------------------- #


def test_no_entry_without_extreme_z() -> None:
    # A linear dislocation ramp has max |z| ~ 1.49 < 2 -> no targets at all,
    # even though every score clears the absolute noise floor.
    candidate = _make_candidate()
    _seed_eligible(candidate, {symbol: 30.0 * i for i, symbol in enumerate(_TRADFI)})
    targets, _vols = candidate._score_and_select()
    assert targets == {}


def test_noise_floor_blocks_tiny_dislocations_even_at_high_z() -> None:
    # Two symmetric +/-5bp outliers give |z| ~ 2.12 but sit below the 10bp
    # pre-registered microstructure floor -> no targets.
    accum = dict.fromkeys(_TRADFI, 0.0)
    accum[PUMP] = 5.0
    accum[DUMP] = -5.0
    candidate = _make_candidate()
    _seed_eligible(candidate, accum)
    targets, _vols = candidate._score_and_select()
    assert targets == {}
    # Control: the same shape at +/-200bp clears both gates with the fade sign.
    control = _make_candidate()
    _seed_eligible(control, {**dict.fromkeys(_TRADFI, 0.0), PUMP: 200.0, DUMP: -200.0})
    targets, vols = control._score_and_select()
    assert set(targets) == {PUMP, DUMP}
    assert targets[PUMP][0] == "SHORT"
    assert targets[DUMP][0] == "LONG"
    assert vols[PUMP] > 0.0


# --------------------------------------------------------------------------- #
# v5 zero-alloc gate: computed alloc 0 emits NOTHING
# --------------------------------------------------------------------------- #


def test_zero_alloc_entry_skipped_not_default_sized() -> None:
    candidate = _make_candidate()
    flip = candidate._state[PUMP]
    flip.closes.append(100.0)
    flip.mode = "LONG"
    flip.entry_price = 100.0
    flip.entry_epoch = _MONDAY_DECISION.timestamp() - 200 * 3600.0  # hold long cleared
    targets = {
        PUMP: ("SHORT", -2.5, {}),
        DUMP: ("LONG", 2.5, {}),
    }
    # Empty weights -> alloc == 0 for every targeted symbol -> no entries.
    candidate._emit_targets(targets, {}, 1.0, _MONDAY_DECISION.isoformat())
    kinds = [(s.symbol, str(s.signal_type).upper()) for s in candidate.events.items]
    assert not [sym for sym, kind in kinds if kind in {"LONG", "SHORT"}], kinds
    # The side-flip EXIT still fired and the state matches it (flat, no ghost).
    assert (PUMP, "EXIT") in kinds
    assert candidate._state[PUMP].mode == "OUT"
    assert candidate._state[PUMP].entry_epoch is None

    # A positive weight DOES emit a sized entry with a strictly positive alloc.
    sized = _make_candidate()
    sized._state[DUMP].closes.append(100.0)
    sized._emit_targets({DUMP: ("LONG", 2.5, {})}, {DUMP: 0.5}, 1.0, _MONDAY_DECISION.isoformat())
    entries = [s for s in sized.events.items if str(s.signal_type).upper() in {"LONG", "SHORT"}]
    assert entries
    assert all(float((s.metadata or {}).get("target_allocation", 0.0)) > 0.0 for s in entries)


# --------------------------------------------------------------------------- #
# universe / self-skip discipline
# --------------------------------------------------------------------------- #


def test_crypto_only_universe_self_skips_deterministically() -> None:
    crypto = ["BTC/USDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
    candidate = _make_candidate(symbols=crypto)
    assert candidate.symbol_list == []
    for step in (0, 1, 2):
        for symbol in crypto[:2]:
            candidate.calculate_signals(
                _event(symbol, _TIMES[step], _CLOSES[symbol][step], _INDEXES[symbol][step])
            )
    assert not candidate.events.items


def test_empty_tradfi_symbols_means_use_all_configured() -> None:
    candidate = _make_candidate(symbols=_TRADFI, tradfi_symbols=())
    assert candidate.symbol_list == _TRADFI


def test_min_symbols_floor_clamps_to_four_and_small_books_skip() -> None:
    assert _make_candidate(min_symbols=2).min_symbols == 4
    # Three eligible TradFi legs with huge dislocations still self-skip (< 4).
    trio = _TRADFI[:3]
    candidate = _make_candidate(symbols=trio)
    _seed_eligible(candidate, {trio[0]: 300.0, trio[1]: -300.0, trio[2]: 0.0})
    targets, vols = candidate._score_and_select()
    assert targets == {} and vols == {}


def test_short_history_self_skips_via_off_obs_and_vol_floors() -> None:
    candidate = _make_candidate()
    _seed_eligible(candidate, {PUMP: 500.0})
    candidate._state[PUMP].off_obs = 2  # below min_off_observations=3
    scores, _vols = candidate._dislocation_scores()
    assert PUMP not in scores
    # Too few closes for the vol window -> realized vol is None -> excluded.
    fresh = _make_candidate()
    item = fresh._state[DUMP]
    item.off_obs = 5
    item.off_accum_bps = 500.0
    for i in range(10):
        item.closes.append(100.0 + float(i))
    scores, _vols = fresh._dislocation_scores()
    assert DUMP not in scores


# --------------------------------------------------------------------------- #
# never-raise hygiene + state roundtrip + determinism
# --------------------------------------------------------------------------- #


def test_degenerate_and_feature_missing_events_never_raise() -> None:
    candidate = _make_candidate(symbols=_TRADFI)
    for event in (
        SimpleNamespace(type="MARKET", time=None, symbol=PUMP, close=None),
        SimpleNamespace(type="MARKET", time="not-a-date", symbol=PUMP, close=100.0),
        SimpleNamespace(
            type="MARKET", time="2025-01-06T00:00:00Z", symbol=PUMP, close=float("nan")
        ),
        # No mark/index attributes at all (8h feature staleness): basis becomes
        # None, the anchor is invalidated, and nothing raises.
        SimpleNamespace(type="MARKET", time="2025-01-06T01:00:00Z", symbol=PUMP, close=100.0),
        SimpleNamespace(
            type="MARKET",
            time="2025-01-06T02:00:00Z",
            symbol=PUMP,
            close=100.0,
            mark_price="garbage",
            index_price=-5.0,
        ),
        SimpleNamespace(type="MARKET_WINDOW", time="2025-01-06T03:00:00Z", bars_1s={}),
        SimpleNamespace(type="OTHER"),
    ):
        candidate.calculate_signals(event)  # must never raise
    assert candidate._state[PUMP].prev_basis_bps is None


def test_stale_feature_gap_is_dropped_not_misattributed() -> None:
    candidate = _make_candidate(symbols=_TRADFI)
    item = candidate._state[PUMP]
    moment = datetime(2025, 1, 11, 0, tzinfo=UTC)  # Saturday, fully OFF
    # Valid bar seeds the anchor; a feature outage invalidates it; the next
    # valid bar re-seeds WITHOUT accruing the 100bp jump across the outage.
    candidate.calculate_signals(_event(PUMP, moment, 100.0, 100.0))
    candidate.calculate_signals(
        SimpleNamespace(
            type="MARKET",
            time=(moment + timedelta(hours=1)).isoformat(),
            symbol=PUMP,
            close=100.5,
        )
    )
    assert item.prev_basis_bps is None
    candidate.calculate_signals(_event(PUMP, moment + timedelta(hours=2), 101.0, 100.0))
    assert item.prev_basis_bps is not None
    assert abs(item.off_accum_bps) < 1e-9


def test_state_roundtrip_is_lossless_mid_position() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS, end_step=183)  # in-position, post-decision
    assert candidate._state[PUMP].mode == "SHORT"
    snapshot = candidate.get_state()
    revived = _make_candidate()
    revived.set_state(snapshot)
    assert revived.get_state() == snapshot


def test_adversarial_set_state_never_raises() -> None:
    candidate = _make_candidate()
    for junk in (
        None,
        {},
        [],
        {"last_decision_date": 123, "symbol_state": 5},
        {"recent_times": "nope", "symbol_state": {PUMP: {"closes": ["x", None]}}},
        {"symbol_state": {PUMP: {"mode": 123, "off_obs": -9, "prev_basis_bps": "z"}}},
        {"symbol_state": {"UNKNOWN": {"closes": [1.0]}, PUMP: "not-a-dict"}},
        {"symbol_state": {PUMP: {"entry_epoch": "bad", "off_accum_bps": float("nan")}}},
    ):
        candidate.set_state(junk)  # must never raise
    # The instance still functions after the abuse.
    _feed(candidate, _ALL_SYMBOLS, end_step=5)


def test_two_runs_are_bit_identical() -> None:
    first = _make_candidate()
    _feed(first, _ALL_SYMBOLS)
    second = _make_candidate()
    _feed(second, _ALL_SYMBOLS)

    def _trace(strategy: Any) -> list[tuple[str, str, float, float]]:
        return [
            (
                s.symbol,
                str(s.signal_type).upper(),
                round(float(s.strength), 12),
                round(float((s.metadata or {}).get("target_allocation", 0.0)), 12),
            )
            for s in strategy.events.items
        ]

    assert _trace(first) == _trace(second)
    assert _trace(first)  # the fixture produces a non-empty tape


def test_window_row_order_does_not_change_the_signal_tape() -> None:
    first = _make_candidate()
    _feed(first, _ALL_SYMBOLS)
    second = _make_candidate()
    _feed(second, list(reversed(_ALL_SYMBOLS)))

    def _trace(strategy: Any) -> list[tuple[str, str, float, float]]:
        return [
            (
                s.symbol,
                str(s.signal_type).upper(),
                round(float(s.strength), 12),
                round(float((s.metadata or {}).get("target_allocation", 0.0)), 12),
            )
            for s in strategy.events.items
        ]

    assert _trace(first) == _trace(second)
    assert _trace(first)


# --------------------------------------------------------------------------- #
# SLEEVE-STATE-02 regressions: panel-wide aging + halted-symbol staleness gate
# --------------------------------------------------------------------------- #


def test_max_hold_exit_requires_a_complete_fresh_panel() -> None:
    # A partial panel cannot age a halted leg out: exits are close-confirmed by
    # a complete same-timestamp window.  Once the leg has a fresh quote in a
    # complete panel, max-hold closes it at its prior known price.
    candidate = _make_candidate()
    item = candidate._state[PUMP]
    item.closes.append(123.0)
    item.mode = "SHORT"
    item.entry_price = 123.0
    item.entry_epoch = _MONDAY_DECISION.timestamp()
    later = _MONDAY_DECISION + timedelta(hours=73)
    candidate.calculate_signals(_window(candidate, later, [DUMP]))
    assert not candidate.events.items
    assert item.mode == "SHORT"
    candidate.calculate_signals(_window(candidate, later, _TRADFI))
    exits = [s for s in candidate.events.items if str(s.signal_type).upper() == "EXIT"]
    assert [s.symbol for s in exits] == [PUMP]
    assert dict(exits[0].metadata or {}).get("reason") == "max_hold"
    assert exits[0].price == 123.0  # last known close of the halted leg
    assert item.mode == "OUT"
    assert item.entry_price is None and item.entry_epoch is None and item.score is None

    # None-safety: a held symbol with NO known close still exits (price=None).
    ghost = _make_candidate()
    bare = ghost._state[PUMP]
    bare.mode = "LONG"
    bare.entry_epoch = _MONDAY_DECISION.timestamp()
    ghost.calculate_signals(_window(ghost, later, _TRADFI))
    exits = [s for s in ghost.events.items if str(s.signal_type).upper() == "EXIT"]
    assert [s.symbol for s in exits] == [PUMP]
    assert exits[0].price is None
    assert bare.mode == "OUT"

    # Inside the 72h window nothing ages out (no spurious panel-wide exits).
    early = _make_candidate()
    held = early._state[PUMP]
    held.closes.append(123.0)
    held.mode = "SHORT"
    held.entry_epoch = _MONDAY_DECISION.timestamp()
    early.calculate_signals(_window(early, _MONDAY_DECISION + timedelta(hours=71), _TRADFI))
    assert not [s for s in early.events.items if str(s.signal_type).upper() == "EXIT"]
    assert held.mode == "SHORT"


def test_halted_leg_fails_closed_instead_of_rebalancing_stale_panel() -> None:
    # PUMP halts at Monday 00:00 with a huge frozen weekend accumulator.  The
    # incomplete Monday panel cannot drive a decision from the remaining legs.
    candidate = _make_candidate()
    halt_at = 168  # Monday 2025-01-13 00:00 UTC: PUMP prints nothing from here
    for step in range(182):  # through Monday 13:00
        moment = _TIMES[step]
        symbols = _TRADFI if step < halt_at else [symbol for symbol in _TRADFI if symbol != PUMP]
        candidate.calculate_signals(_window(candidate, moment, symbols))
    assert candidate._state[PUMP].off_accum_bps > 150.0  # frozen photograph
    assert _final_sides(candidate.events.items) == {}


def test_stale_last_bar_excluded_from_scores_fresh_passes() -> None:
    # Fix 1b (unit): with hourly spacing inferred from _recent_times, a leg
    # whose last accepted bar is >3h older than the decision is dropped from
    # the score panel; fresh legs and the no-decision-time path pass through.
    candidate = _make_candidate()
    _seed_eligible(candidate, dict.fromkeys(_TRADFI, 0.0))
    candidate._state[PUMP].off_accum_bps = 200.0
    decision_epoch = _MONDAY_DECISION.timestamp()
    for back in range(8, 0, -1):
        candidate._recent_times.append(decision_epoch - back * 3600.0)
    for symbol in _TRADFI:
        candidate._state[symbol].last_bar_epoch = decision_epoch
    candidate._state[PUMP].last_bar_epoch = decision_epoch - 4 * 3600.0  # > 3x 1h
    scores, _vols = candidate._dislocation_scores(decision_epoch)
    assert PUMP not in scores
    assert set(scores) == set(_TRADFI) - {PUMP}
    # Exactly-3x staleness is NOT excluded (strict > boundary).
    candidate._state[PUMP].last_bar_epoch = decision_epoch - 3 * 3600.0
    scores, _vols = candidate._dislocation_scores(decision_epoch)
    assert PUMP in scores
    # No decision time -> pass-through (seeded direct-call paths keep working).
    candidate._state[PUMP].last_bar_epoch = decision_epoch - 100 * 3600.0
    scores, _vols = candidate._dislocation_scores()
    assert PUMP in scores


# --------------------------------------------------------------------------- #
# WARMUP-GHOST-BOOK regression: on_warmup_end flattens positions only
# --------------------------------------------------------------------------- #


def test_on_warmup_end_flattens_ghost_book_preserves_accumulators() -> None:
    candidate = _make_candidate()
    _feed(candidate, _ALL_SYMBOLS, end_step=183)  # in-position, post-decision
    item = candidate._state[PUMP]
    assert item.mode == "SHORT"
    closes_before = list(item.closes)
    accum_before = item.off_accum_bps
    obs_before = item.off_obs
    anchor_before = item.prev_basis_bps
    key_before = item.last_time_key
    epoch_before = item.last_bar_epoch
    recent_before = list(candidate._recent_times)
    decision_before = candidate._last_decision_date
    events_before = len(candidate.events.items)

    candidate.on_warmup_end()

    for state in candidate._state.values():
        assert state.mode == "OUT"
        assert state.entry_price is None
        assert state.entry_epoch is None
        assert state.score is None
    # Accumulators / deques / clocks survive verbatim.
    assert list(item.closes) == closes_before
    assert item.off_accum_bps == accum_before
    assert item.off_obs == obs_before
    assert item.prev_basis_bps == anchor_before
    assert item.last_time_key == key_before
    assert item.last_bar_epoch == epoch_before
    assert list(candidate._recent_times) == recent_before
    assert candidate._last_decision_date == decision_before
    # The ghost book is dropped silently: no EXIT is emitted for it.
    assert len(candidate.events.items) == events_before
    # The engine discovers the hook via getattr + callable.
    assert callable(getattr(candidate, "on_warmup_end", None))


# --------------------------------------------------------------------------- #
# candidate-wiring exports (module-level slice + tags)
# --------------------------------------------------------------------------- #


def test_slice_and_tags_exports_are_pre_registered_constants() -> None:
    assert "cross_sectional" in _SUGGESTED_CANDIDATE_TAGS
    assert set(_OFFSESSION_BASIS_DISLOCATION_SLICE) == {"1h"}
    variants = _OFFSESSION_BASIS_DISLOCATION_SLICE["1h"]
    assert 2 <= len(variants) <= 4
    assert _DEFAULT_TRADFI_UNIVERSE  # candidate wiring passes the TradFi set
    for variant in variants:
        assert variant["entry_z"] == 2.0
        assert variant["min_hold_hours"] == 36
        assert variant["max_hold_hours"] == 72
        assert tuple(variant["tradfi_symbols"]) == _DEFAULT_TRADFI_UNIVERSE
        built = OffSessionBasisDislocationStrategy(
            SimpleNamespace(symbol_list=list(_TRADFI)), _Queue(), **variant
        )
        assert built.min_symbols >= 4
        assert built.symbol_list == _TRADFI
