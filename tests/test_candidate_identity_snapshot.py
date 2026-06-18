"""Snapshot test pinning the CURRENT (divergent) candidate_identity behavior.

There are three candidate_identity implementations inside ``strategy_factory``:

* ``selection.candidate_identity`` -- OMITS ``strategy_class`` from the hash
  payload, so two candidates that share name/timeframe/symbols/params but differ
  ONLY in ``strategy_class`` produce the SAME identity (a hash collision). It
  also emits a 20-char digest.
* ``research_run_support._candidate_identity`` -- INCLUDES ``strategy_class`` in
  the payload, so the same pair does NOT collide. It emits a 16-char digest.
* ``research_runner._candidate_identity`` -- delegates to the
  ``research_run_support`` implementation, so it matches that behavior.

NOTE: the ``selection`` collision is the documented CURRENT behavior to be fixed
in the candidate-identity unification pass. This test deliberately PINS the
collision (it does NOT assert the desired post-fix behavior) so the divergent
surface is locked before the refactor. When unification lands, update the
``selection`` assertions here to assert non-collision.
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


def test_selection_identity_collides_on_strategy_class() -> None:
    # CURRENT BEHAVIOR (bug to be fixed in unification pass): selection omits
    # strategy_class, so same-name candidates with different strategy_class
    # collide to the SAME identity.
    foo = _candidate("FooStrategy")
    bar = _candidate("BarStrategy")

    foo_id = candidate_identity(foo)
    bar_id = candidate_identity(bar)

    assert foo_id == bar_id, "PIN: selection.candidate_identity currently COLLIDES"
    # Pin the exact snapshot digest + width (20 chars).
    assert foo_id == "b6f316cc63fdd9c8aac4"
    assert len(foo_id) == 20


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
    # Lock the cross-implementation divergence: selection (20-char, colliding)
    # is distinct from the research path (16-char, non-colliding) for the SAME
    # input candidate. This is the surface the unification pass must collapse.
    foo = _candidate("FooStrategy")

    selection_id = candidate_identity(foo)
    research_id = research_run_support_identity(foo)

    assert len(selection_id) == 20
    assert len(research_id) == 16
    assert selection_id != research_id
