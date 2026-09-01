"""Zero-allocation emission gate for VolManagedMomentumCrashGateStrategy (v5 fix).

A selection whose vol-scaled allocation collapses to 0 (crash gate with
``stress_reduce=0`` or a vol-collapsed leverage of 0) must NOT emit an entry:
``_target_metadata`` omits ``target_allocation`` when it is not positive and
the engine falls back to its DEFAULT sizing -- silently resizing a 0% target
into a real position.  These tests pin the selection-time gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from lumina_quant.strategies.external_alpha_sleeves import (
    VolManagedMomentumCrashGateStrategy,
)

_SYMBOLS = ["BTC/USDT", "AAA/USDT", "BBB/USDT"]
_T = datetime(2026, 1, 1, tzinfo=UTC)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def _make(**overrides: Any) -> VolManagedMomentumCrashGateStrategy:
    params: dict[str, Any] = dict(
        allow_short=True,
        max_longs=2,
        max_shorts=2,
        signal_threshold=0.1,
        rebalance_bars=1,
    )
    params.update(overrides)
    return VolManagedMomentumCrashGateStrategy(_Bars(_SYMBOLS), _Queue(), **params)


def _entries(strategy: Any) -> list[Any]:
    return [s for s in strategy.events.items if s.signal_type in {"LONG", "SHORT"}]


def test_crash_zero_stress_short_book_emits_nothing() -> None:
    # crash=True clears longs; stress_reduce=0 makes base_alloc 0 for the
    # surviving shorts.  Pre-fix this emitted SHORT entries whose metadata
    # lacked ``target_allocation`` (engine default sizing).
    strat = _make(stress_reduce=0.0)
    strat._score_symbol = lambda symbol: (-2.0, 0.02, 1.0)
    strat._crash_gate = lambda: (True, 0.0, {"stress_multiplier": 0.0})
    strat._rebalance(_T)
    assert not _entries(strat)


def test_zero_leverage_symbol_skipped_sized_symbols_still_enter() -> None:
    strat = _make()

    def _score(symbol: str) -> tuple[float, float, float]:
        if symbol == "AAA/USDT":
            return (2.0, 0.02, 0.0)  # vol-collapsed leverage -> alloc 0
        return (2.0, 0.02, 1.0)

    strat._score_symbol = _score
    strat._crash_gate = lambda: (False, 1.0, {"stress_multiplier": 1.0})
    strat._rebalance(_T)
    entries = _entries(strat)
    entered = {s.symbol for s in entries}
    assert "AAA/USDT" not in entered, "zero-alloc selection must not emit an entry"
    assert entered, "positively sized selections must still enter"
    for signal in entries:
        assert float(signal.metadata.get("target_allocation", 0.0)) > 0.0


def test_entries_always_carry_positive_target_allocation_metadata() -> None:
    strat = _make(stress_reduce=0.35)
    strat._score_symbol = lambda symbol: (2.0, 0.02, 1.0)
    strat._crash_gate = lambda: (True, 0.35, {"stress_multiplier": 0.35})
    strat._rebalance(_T)
    # crash=True clears the long book entirely -> nothing to size.
    assert not _entries(strat)

    strat2 = _make(stress_reduce=0.35)
    strat2._score_symbol = lambda symbol: (-2.0, 0.02, 1.0)
    strat2._crash_gate = lambda: (True, 0.35, {"stress_multiplier": 0.35})
    strat2._rebalance(_T)
    entries = _entries(strat2)
    assert entries, "stress-reduced (but nonzero) short book must still enter"
    for signal in entries:
        assert float(signal.metadata.get("target_allocation", 0.0)) > 0.0
