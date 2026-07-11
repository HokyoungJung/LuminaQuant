"""Regression test for the shared OLS-basis cross-sectional book engine.

``ols_basis_xs_common.OLSBasisCrossSectionalBook`` is an inert abstract base
(no ``@register``, ``_score_symbol`` unimplemented), so its shared
``_emit_targets`` gate is exercised through a concrete subclass -- the
regression-trend-quality lane, which adds only a scoring hook.

v5 zero-alloc gate: an entry whose inverse-vol weight is 0 must NOT be emitted.
``_target_metadata`` drops ``target_allocation`` at alloc 0, and the engine
would then resize the position to its DEFAULT allocation (an unsized,
un-vol-gated bet).  The book computes inverse-vol weights INTERNALLY, so an
empty ``vols`` map forces every weight (and thus every alloc) to 0.

All inputs are hand-built and deterministic (no ``random`` module).
"""

from __future__ import annotations

from typing import Any

from lumina_quant.strategies.trend_quality_xs_alpha_sleeves import (
    CrossSectionalRegressionTrendQualityStrategy,
)


class _Bars:
    def __init__(self, symbols: list[str]) -> None:
        self.symbol_list = list(symbols)


class _Queue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, item: Any) -> None:
        self.items.append(item)


def test_zero_alloc_entry_skipped_not_default_sized() -> None:
    symbols = ["FLIP/USDT", "FRESH/USDT", "N0/USDT", "N1/USDT", "N2/USDT", "N3/USDT"]
    strat = CrossSectionalRegressionTrendQualityStrategy(_Bars(symbols), _Queue())
    flip = strat._state["FLIP/USDT"]
    flip.mode = "LONG"
    flip.entry_price = 100.0
    flip.bars_held = 10_000
    desired = {"FLIP/USDT": "SHORT", "FRESH/USDT": "LONG"}
    # empty vols -> internal inverse-vol weights are all 0 -> alloc == 0
    strat._emit_targets(desired, {}, "2026-01-01T00:00:00Z")
    kinds = [(sig.symbol, str(sig.signal_type).upper()) for sig in strat.events.items]
    assert not [sym for sym, kind in kinds if kind in {"LONG", "SHORT"}], kinds
    # the side-flip EXIT still fired and FLIP is now flat (state matches the exit)
    assert ("FLIP/USDT", "EXIT") in kinds
    assert strat._state["FLIP/USDT"].mode == "OUT"
    assert strat._state["FLIP/USDT"].entry_price is None

    # a positive inverse-vol weight (from a real vol) DOES emit a sized entry
    # carrying a strictly positive ``target_allocation``.
    sized = CrossSectionalRegressionTrendQualityStrategy(_Bars(symbols), _Queue())
    sized._emit_targets({"FRESH/USDT": "LONG"}, {"FRESH/USDT": 0.1}, "2026-01-01T00:00:00Z")
    entries = [
        sig for sig in sized.events.items if str(sig.signal_type).upper() in {"LONG", "SHORT"}
    ]
    assert entries, "a positive inverse-vol weight must emit a sized entry"
    assert all(float((sig.metadata or {}).get("target_allocation", 0.0)) > 0.0 for sig in entries)
