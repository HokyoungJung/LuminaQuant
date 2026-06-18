"""Snapshot test for the candidate_identity implementations in ``strategy_factory``.

* ``selection.candidate_identity`` -- now INCLUDES ``strategy_class`` in the hash
  payload (collision fixed), so two candidates that share
  name/timeframe/symbols/params but differ in ``strategy_class`` get DISTINCT
  identities. It emits a 20-char digest.
* ``research_run_support._candidate_identity`` -- INCLUDES ``strategy_class`` in
  the payload, so the same pair does NOT collide. It emits a 16-char digest.
* ``research_runner._candidate_identity`` -- delegates to the
  ``research_run_support`` implementation, so it matches that behavior.

``selection`` and the research path remain independent identity functions for
different subsystems (different payload key names + digest widths, 20 vs 16); they
are intentionally NOT unified, but BOTH now include ``strategy_class``.
"""

from __future__ import annotations

from lumina_quant.strategy_factory.research_run_support import (
    _candidate_identity as research_run_support_identity,
)
from lumina_quant.strategy_factory.research_runner import (
    _candidate_identity as research_runner_identity,
)
from lumina_quant.strategy_factory.selection import candidate_identity


def _candidate(strategy_class: str) -> dict[str, object]:
    """Two of these differ ONLY in ``strategy_class``."""
    return {
        "name": "alpha",
        "strategy_class": strategy_class,
        "timeframe": "1h",
        "symbols": ["BTCUSDT"],
        "params": {"x": 1},
    }


def test_selection_identity_distinguishes_strategy_class() -> None:
    # FIXED: selection now includes strategy_class, so same-name candidates with
    # a different strategy_class get DISTINCT identities (no more collision).
    foo = _candidate("FooStrategy")
    bar = _candidate("BarStrategy")

    foo_id = candidate_identity(foo)
    bar_id = candidate_identity(bar)

    assert foo_id != bar_id, "selection.candidate_identity must distinguish strategy_class"
    # Pin the exact post-fix snapshot digests + width (20 chars).
    assert foo_id == "0e14b331a91177aebb22"
    assert bar_id == "1a107601dd86fff1b563"
    assert len(foo_id) == 20
    # Identical candidates remain stable.
    assert candidate_identity(_candidate("FooStrategy")) == foo_id


def test_research_run_support_identity_does_not_collide() -> None:
    # CURRENT BEHAVIOR: research_run_support includes strategy_class, so the
    # same pair does NOT collide.
    foo_id = research_run_support_identity(_candidate("FooStrategy"))
    bar_id = research_run_support_identity(_candidate("BarStrategy"))

    assert foo_id != bar_id, "PIN: research_run_support identity does NOT collide"
    assert len(foo_id) == 16
    assert len(bar_id) == 16


def test_research_runner_identity_matches_run_support() -> None:
    # CURRENT BEHAVIOR: research_runner delegates to research_run_support, so it
    # shares the non-colliding 16-char behavior and produces identical digests.
    foo = _candidate("FooStrategy")
    bar = _candidate("BarStrategy")

    runner_foo = research_runner_identity(foo)
    runner_bar = research_runner_identity(bar)

    assert runner_foo != runner_bar, "PIN: research_runner identity does NOT collide"
    assert runner_foo == research_run_support_identity(foo)
    assert runner_bar == research_run_support_identity(bar)


def test_selection_diverges_from_research_implementations() -> None:
    # selection (20-char) and the research path (16-char) are intentionally
    # independent identity functions for different subsystems; both now include
    # strategy_class, but they use different payload keys/widths so their digests
    # differ for the same input. Lock that they stay distinct (not accidentally unified).
    foo = _candidate("FooStrategy")

    selection_id = candidate_identity(foo)
    research_id = research_run_support_identity(foo)

    assert len(selection_id) == 20
    assert len(research_id) == 16
    assert selection_id != research_id
