"""Deterministic data-free tests for the basis-funding gap convergence sleeve.

Covers the FADE sign-lock on a synthetic panel (rich basis vs standing funding
=> SHORT; discount => LONG); the 8h funding-boundary cadence floor; the hard
min-hold before the gap-sign-flip exit; the 72h-style max-hold exit; the
``intended_hold_seconds`` metadata stamp (>= 2 funding intervals, L-D guard
composition); zero-weight no-emit (v5 zero-alloc resize trap); min-symbols and
short-history self-skips; never-raise on garbage events and adversarial
``set_state``; lossless state roundtrip; and two-run determinism.  No backtest
is run; no data files are read; randomness is a seeded inline LCG only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from lumina_quant.strategies.basis_funding_gap_alpha_sleeves import (
    _BASIS_FUNDING_GAP_SLICE,
    _SUGGESTED_CANDIDATE_TAGS,
    BasisFundingGapConvergenceStrategy,
)

_SYMBOLS = [f"{c * 3}/USDT" for c in "ABCDEFGH"]
_RICH = "HHH/USDT"
_START = datetime(2026, 1, 1, tzinfo=UTC)
_INTERVAL_SECONDS = 28800


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield state / 2**31


def _ts(i: int) -> str:
    return (_START + timedelta(seconds=_INTERVAL_SECONDS * i)).isoformat()


def _interval_of(raw: Any) -> int:
    dt = datetime.fromisoformat(str(raw))
    return int((dt - _START).total_seconds() // _INTERVAL_SECONDS)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


class _Bars:
    def __init__(self, symbols: list[str]):
        self.symbol_list = list(symbols)

    def get_latest_feature_value(self, symbol: str, field: str) -> float | None:
        return None

    def get_latest_bar_value(self, symbol: str, field: str) -> float | None:
        return None


def _close(base: float, t: int) -> float:
    zig = 1.0 if t % 2 == 0 else -1.0
    return base * (1.0 + 0.004 * zig)


def _event(
    symbol: str,
    t: int,
    *,
    close: float,
    mark: float | None,
    index: float | None = 100.0,
    funding: float | None = 0.0001,
) -> SimpleNamespace:
    kwargs: dict[str, Any] = {
        "type": "MARKET",
        "time": _ts(t),
        "symbol": symbol,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000.0,
    }
    if funding is not None:
        kwargs["funding_rate"] = funding
    if mark is not None:
        kwargs["mark_price"] = mark
    if index is not None:
        kwargs["index_price"] = index
    return SimpleNamespace(**kwargs)


def _default_mark(symbol: str, t: int) -> float:
    """Flat cross-section: gaps span -1..+6 bps, no |z|>2 outlier anywhere."""
    _ = t
    return 100.0 + 0.01 * _SYMBOLS.index(symbol)


def _feed(
    strategy: BasisFundingGapConvergenceStrategy,
    n: int,
    mark_fn: Any,
    *,
    symbols: list[str] | None = None,
) -> None:
    roster = symbols if symbols is not None else _SYMBOLS
    for t in range(n):
        for i, symbol in enumerate(roster):
            strategy.calculate_signals(
                _event(symbol, t, close=_close(100.0 + i, t), mark=mark_fn(symbol, t))
            )


def _make(queue: _Queue, **extra: Any) -> BasisFundingGapConvergenceStrategy:
    params: dict[str, Any] = {"min_symbols": 8, "vol_window": 4}
    params.update(extra)
    return BasisFundingGapConvergenceStrategy(_Bars(_SYMBOLS), queue, **params)


def _rich_mark(symbol: str, t: int) -> float:
    """One rich-basis outlier (+99bp gap) among a -1..+5bp cross-section."""
    if symbol == _RICH:
        return 101.0
    return 100.0 + 0.01 * _SYMBOLS.index(symbol)


def _discount_mark(symbol: str, t: int) -> float:
    """One discount outlier (-101bp gap) among a -1..+5bp cross-section."""
    if symbol == _RICH:
        return 99.0
    return 100.0 + 0.01 * _SYMBOLS.index(symbol)


def _entries(queue: _Queue) -> list[Any]:
    return [s for s in queue.items if s.signal_type in {"LONG", "SHORT"}]


def _exits(queue: _Queue) -> list[Any]:
    return [s for s in queue.items if s.signal_type == "EXIT"]


# --------------------------------------------------------------------------- #
# Funding-boundary cadence discipline
# --------------------------------------------------------------------------- #
def test_cadence_floor_is_the_8h_funding_interval() -> None:
    assert BasisFundingGapConvergenceStrategy.decision_cadence_seconds == 28800
    strat = _make(_Queue())
    assert strat.decision_cadence_seconds >= 28800


# --------------------------------------------------------------------------- #
# Sign-lock: FADE the gap
# --------------------------------------------------------------------------- #
def test_rich_basis_outlier_is_faded_short() -> None:
    queue = _Queue()
    strat = _make(queue)
    _feed(strat, 10, _rich_mark)
    entries = _entries(queue)
    assert entries, "expected the rich-basis outlier to trigger an entry"
    assert {s.symbol for s in entries} == {_RICH}
    assert all(s.signal_type == "SHORT" for s in entries)
    meta = entries[0].metadata
    assert meta["gap_z"] > 2.0
    assert meta["gap_bps"] == pytest.approx(99.0)
    # Close-confirmed exits only: no engine intrabar brackets on the signal.
    assert entries[0].stop_loss is None and entries[0].take_profit is None


def test_discount_outlier_is_faded_long() -> None:
    queue = _Queue()
    strat = _make(queue)
    _feed(strat, 10, _discount_mark)
    entries = _entries(queue)
    assert entries, "expected the discount outlier to trigger an entry"
    assert {s.symbol for s in entries} == {_RICH}
    assert all(s.signal_type == "LONG" for s in entries)
    assert entries[0].metadata["gap_z"] < -2.0


def test_no_dislocation_no_entries() -> None:
    queue = _Queue()
    strat = _make(queue)
    _feed(strat, 10, _default_mark)  # gaps -1..+5bp, no |z|>2 outlier
    assert queue.items == []


def test_allow_short_false_suppresses_the_rich_fade() -> None:
    queue = _Queue()
    strat = _make(queue, allow_short=False)
    _feed(strat, 10, _rich_mark)
    assert _entries(queue) == []


# --------------------------------------------------------------------------- #
# Metadata: intended_hold_seconds (L-D guard composition) + allocation caps
# --------------------------------------------------------------------------- #
def test_entry_metadata_stamps_intended_hold_of_at_least_two_intervals() -> None:
    queue = _Queue()
    strat = _make(queue)
    _feed(strat, 10, _rich_mark)
    entries = _entries(queue)
    assert entries
    for sig in entries:
        held = int(sig.metadata["intended_hold_seconds"])
        assert held == strat.min_hold_intervals * 28800
        assert held >= 2 * 28800
        assert float(sig.metadata["target_allocation"]) > 0.0
        assert sig.metadata["max_symbol_exposure_pct"] <= strat.max_symbol_exposure_pct + 1e-12


# --------------------------------------------------------------------------- #
# Hard min-hold + gap-sign-flip exit
# --------------------------------------------------------------------------- #
def test_min_hold_blocks_flip_exit_then_flip_exits() -> None:
    queue = _Queue()
    strat = _make(queue, min_hold_intervals=2, max_hold_intervals=9)

    def mark_fn(symbol: str, t: int) -> float:
        if symbol == _RICH:
            # Rich through interval 4 (entry at the interval-5 evaluation,
            # which still sees the interval-4 gap), then the gap sign flips to
            # a small discount (-1bp: sign flip WITHOUT a fresh |z|>2 signal).
            return 101.0 if t < 5 else 100.0
        return 100.0 + 0.01 * _SYMBOLS.index(symbol)

    _feed(strat, 12, mark_fn)
    entries = _entries(queue)
    exits = _exits(queue)
    assert len(entries) == 1 and entries[0].symbol == _RICH
    assert len(exits) == 1 and exits[0].symbol == _RICH
    assert exits[0].metadata["reason"] == "gap_sign_flip"
    held_intervals = _interval_of(exits[0].datetime) - _interval_of(entries[0].datetime)
    # The flip was observable one interval after entry, but the HARD min-hold
    # forces the position to be held the full two funding intervals first.
    assert held_intervals == 2


# --------------------------------------------------------------------------- #
# Max-hold exit
# --------------------------------------------------------------------------- #
def test_max_hold_exits_a_persistent_dislocation() -> None:
    queue = _Queue()
    strat = _make(queue, min_hold_intervals=2, max_hold_intervals=3)
    _feed(strat, 14, _rich_mark)  # gap never flips
    entries = _entries(queue)
    exits = _exits(queue)
    assert entries and exits
    assert exits[0].metadata["reason"] == "max_hold"
    held_intervals = _interval_of(exits[0].datetime) - _interval_of(entries[0].datetime)
    assert held_intervals == 3
    # The dislocation persists after the max-hold flatten, so the sleeve
    # re-fades it: every entry is a SHORT on the same rich outlier.
    assert {s.symbol for s in entries} == {_RICH}
    assert all(s.signal_type == "SHORT" for s in entries)


# --------------------------------------------------------------------------- #
# Zero-weight => emit NOTHING (v5 zero-alloc resize trap)
# --------------------------------------------------------------------------- #
def test_zero_weight_candidate_emits_nothing() -> None:
    strat = _make(_Queue())
    strat._emit_entries({_RICH: ("SHORT", 3.0)}, {}, 1.0, _ts(0))
    strat._emit_entries({_RICH: ("SHORT", 3.0)}, {_RICH: 0.0}, 1.0, _ts(0))
    assert strat.events.items == []
    assert strat._state[_RICH].mode == "OUT"


def test_zero_gross_exposure_emits_nothing_end_to_end() -> None:
    queue = _Queue()
    strat = _make(queue, target_gross_exposure=0.0)
    _feed(strat, 10, _rich_mark)
    assert queue.items == []


# --------------------------------------------------------------------------- #
# Self-skips: min-symbols and short history
# --------------------------------------------------------------------------- #
def test_min_symbols_self_skip_small_universe() -> None:
    queue = _Queue()
    small = _SYMBOLS[:3]
    strat = BasisFundingGapConvergenceStrategy(_Bars(small), queue, min_symbols=6, vol_window=4)
    _feed(strat, 10, _rich_mark, symbols=small)
    assert queue.items == []


def test_min_symbols_self_skip_when_features_missing() -> None:
    queue = _Queue()
    strat = _make(queue, min_symbols=6)
    # Only 3 symbols ever print mark_price => fewer than min_symbols gaps.
    for t in range(10):
        for i, symbol in enumerate(_SYMBOLS):
            mark = _rich_mark(symbol, t) if i < 3 else None
            strat.calculate_signals(_event(symbol, t, close=_close(100.0 + i, t), mark=mark))
    assert queue.items == []


def test_short_history_self_skip() -> None:
    queue = _Queue()
    strat = _make(queue)  # vol_window=4 needs 5 closes per symbol
    _feed(strat, 3, _rich_mark)
    assert queue.items == []


# --------------------------------------------------------------------------- #
# Never-raise: garbage events and adversarial set_state
# --------------------------------------------------------------------------- #
def test_never_raises_on_garbage_events() -> None:
    strat = _make(_Queue())
    strat.calculate_signals(SimpleNamespace(type="HEARTBEAT"))
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol="ZZZ/USDT", close=None))
    strat.calculate_signals(SimpleNamespace(type="MARKET", symbol=_RICH, close=None, time=None))
    strat.calculate_signals(
        SimpleNamespace(type="MARKET", symbol=_RICH, close=0.0, time=_ts(0))
    )  # below min_price
    # Features present but degenerate index => gap None => bar dropped.
    strat.calculate_signals(_event(_RICH, 1, close=100.0, mark=101.0, index=0.0, funding=0.0001))
    # Missing funding => gap None => bar dropped.
    strat.calculate_signals(_event(_RICH, 2, close=100.0, mark=101.0, funding=None))

    class _EmptyWindow:
        type = "MARKET_WINDOW"
        bars_1s: dict[str, Any] = {}

    strat.calculate_signals(_EmptyWindow())
    assert strat.events.items == []


def test_adversarial_set_state_never_raises() -> None:
    for payload in (
        None,
        [],
        "garbage",
        {"symbol_state": "nope"},
        {"symbol_state": {"ZZZ/USDT": {"closes": [1.0]}}},
        {
            "tick": "x",
            "recent_times": ["y", None, 1.0],
            "symbol_state": {
                _RICH: {
                    "closes": ["x", None, 101.0],
                    "gaps": {"not": "a list"},
                    "mode": "SIDEWAYS",
                    "entry_price": "n/a",
                    "entry_gap_sign": "z",
                    "intervals_held": -7,
                    "last_time_key": 123,
                }
            },
        },
    ):
        strat = _make(_Queue())
        strat.set_state(payload)  # type: ignore[arg-type]
        item = strat._state[_RICH]
        assert item.mode in {"OUT", "LONG", "SHORT"}
        assert item.entry_gap_sign in {-1, 0, 1}
        assert item.intervals_held >= 0


# --------------------------------------------------------------------------- #
# State roundtrip
# --------------------------------------------------------------------------- #
def test_state_roundtrip_lossless() -> None:
    strat = _make(_Queue())
    _feed(strat, 8, _rich_mark)
    item = strat._state[_RICH]
    assert item.mode == "SHORT"  # entered by the fixture
    state = strat.get_state()

    restored = _make(_Queue())
    restored.set_state(state)
    for symbol in _SYMBOLS:
        a, b = strat._state[symbol], restored._state[symbol]
        assert list(a.closes) == list(b.closes)
        assert list(a.gaps) == list(b.gaps)
        assert a.mode == b.mode
        assert a.entry_price == b.entry_price
        assert a.entry_gap_sign == b.entry_gap_sign
        assert a.intervals_held == b.intervals_held
        assert a.last_time_key == b.last_time_key
    assert restored._tick == strat._tick
    assert restored._last_eval_time_key == strat._last_eval_time_key
    assert list(restored._recent_times) == list(strat._recent_times)
    # Roundtrip is idempotent (a second export matches the first).
    assert restored.get_state() == state


# --------------------------------------------------------------------------- #
# Two-run determinism
# --------------------------------------------------------------------------- #
def test_two_run_determinism_on_noisy_panel() -> None:
    def run() -> list[tuple[str, str, Any, tuple[tuple[str, Any], ...]]]:
        gen = _lcg(20260820)
        noise = {(s, t): (next(gen) - 0.5) * 0.02 for t in range(16) for s in _SYMBOLS}

        def mark_fn(symbol: str, t: int) -> float:
            base = 101.0 if (symbol == _RICH and t < 9) else (100.0 + 0.01 * _SYMBOLS.index(symbol))
            return base + noise[(symbol, t)]

        queue = _Queue()
        strat = _make(queue, min_hold_intervals=2, max_hold_intervals=5)
        _feed(strat, 16, mark_fn)
        return [
            (s.symbol, s.signal_type, s.datetime, tuple(sorted((s.metadata or {}).items())))
            for s in queue.items
        ]

    first = run()
    second = run()
    assert first == second
    assert first, "fixture should emit at least one signal"


# --------------------------------------------------------------------------- #
# Candidate-wiring exports (hints only; the integrator wires candidates)
# --------------------------------------------------------------------------- #
def test_slice_exports_are_pre_registered_constants() -> None:
    assert "cross_sectional" in _SUGGESTED_CANDIDATE_TAGS
    assert "carry" in _SUGGESTED_CANDIDATE_TAGS
    schema = BasisFundingGapConvergenceStrategy.get_param_schema()
    total_variants = 0
    for timeframe, variants in _BASIS_FUNDING_GAP_SLICE.items():
        assert timeframe in {"30m", "1h", "4h", "1d"}
        for variant in variants:
            total_variants += 1
            for key, value in variant.items():
                if key == "variant":
                    continue
                assert key in schema, key
                # Pre-registered constants only -- scalars, never grids.
                assert isinstance(value, (int, float, bool)), (key, value)
            assert int(variant["min_hold_intervals"]) >= 2
            assert int(variant["max_hold_intervals"]) * 8 <= 72
    assert 2 <= total_variants <= 4
