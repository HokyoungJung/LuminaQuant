"""Engine-side wiring of the anti-overfit selection gate in the monthly-refit
walk-forward runner (``scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward``).

These pin the config-gated seam without running the (data-dependent) walk-forward:
the OFF-path is a strict identity no-op (byte-identical admit boundary), basket
dedup collapses identical-symbol clones, the reject floors resolve from CLI and
from a strict RuntimeConfig, and the reject axis stays inert until the upstream
per-candidate DSR/SPA/PBO metric plumbing lands (``reject_ready`` TODO).
"""

from __future__ import annotations

import argparse
import importlib

import pytest

ENGINE = "scripts.research.run_alpha_zoo_69_asset_monthly_refit_walkforward"


@pytest.fixture
def eng():
    return importlib.import_module(ENGINE)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        config=None,
        enforce_selection_reject_gate=False,
        dsr_gate_floor=None,
        spa_gate_ceiling=None,
        pbo_gate_ceiling=None,
        dedupe_baskets=False,
        max_per_symbol_basket=None,
        max_per_lineage=None,
        max_per_family_basket=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _rows():
    clone = {
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "validation": {"deflated_sharpe": 0.05, "spa_pvalue": 0.4, "pbo": 0.6},
    }
    return [
        {"candidate_id": "a", "strategy_class": "X", "family": "f", **clone},
        {"candidate_id": "b", "strategy_class": "Y", "family": "g", **clone},
        {"candidate_id": "c", "strategy_class": "Z", "family": "h", **clone},
        {
            "candidate_id": "d",
            "strategy_class": "Q",
            "family": "i",
            "symbols": ["DOGE/USDT"],
            "validation": {"deflated_sharpe": 0.99, "spa_pvalue": 0.01, "pbo": 0.1},
        },
    ]


def test_gate_off_by_default_is_identity_passthrough(eng):
    eng._configure_selection_gate(_args())
    rows = _rows()
    out = eng._apply_selection_gate_rows(rows, mode="val")
    assert eng._SELECTION_GATE_STATE["enabled"] is False
    assert [r["candidate_id"] for r in out] == ["a", "b", "c", "d"]
    # Same objects, same order: strict identity no-op (byte-identical admit).
    assert all(a is b for a, b in zip(out, rows, strict=True))


def test_dedupe_flag_collapses_identical_symbol_basket_clones(eng):
    eng._configure_selection_gate(_args(dedupe_baskets=True))
    state = eng._SELECTION_GATE_STATE
    assert state["enabled"] is True
    assert (state["max_per_symbol_basket"], state["max_per_lineage"]) == (2, 1)
    out = eng._apply_selection_gate_rows(_rows(), mode="val")
    # The 3-member identical-basket clone cluster (a/b/c) collapses to 2; d kept.
    assert [r["candidate_id"] for r in out] == ["a", "b", "d"]


def test_enforce_flag_reject_axis_is_inert_until_metrics_land(eng):
    """--enforce-selection-reject-gate resolves the floors but does NOT over-reject
    today's DSR-less engine rows: ``reject_ready`` is False, so the reject axis is
    a documented no-op while basket dedup (if requested) still applies."""
    eng._configure_selection_gate(_args(enforce_selection_reject_gate=True))
    state = eng._SELECTION_GATE_STATE
    assert state["reject_ready"] is False
    assert state["robust_score_params"] is not None  # resolved + ready for the hook
    out = eng._apply_selection_gate_rows(_rows(), mode="val")
    assert [r["candidate_id"] for r in out] == ["a", "b", "c", "d"]


def test_strict_profile_config_resolves_reject_floors(eng):
    eng._configure_selection_gate(_args(config="configs/profiles/research.yaml"))
    state = eng._SELECTION_GATE_STATE
    assert state["enabled"] is True
    assert state["robust_score_params"] == {
        "enforce_selection_reject_gate": True,
        "dsr_gate_floor": 0.90,
        "spa_gate_ceiling": 0.05,
        "pbo_gate_ceiling": 0.50,
    }
    # Reset the module holder so test order cannot leak an armed gate.
    eng._configure_selection_gate(_args())
    assert eng._SELECTION_GATE_STATE["enabled"] is False


def test_strict_profile_config_arms_basket_dedup_caps(eng):
    """Loading the strict profile via --config now ARMS basket dedup: the 2/1/1
    caps come from research.yaml (not just the reject floors), so a strict-profile
    run actually drops clones -- the [HIGH] review fix."""
    eng._configure_selection_gate(_args(config="configs/profiles/research.yaml"))
    state = eng._SELECTION_GATE_STATE
    assert (
        state["max_per_symbol_basket"],
        state["max_per_lineage"],
        state["max_per_family_basket"],
    ) == (2, 1, 1)
    # And dedup actually fires end-to-end on the clone cluster.
    out = eng._apply_selection_gate_rows(_rows(), mode="val")
    assert [r["candidate_id"] for r in out] == ["a", "b", "d"]
    eng._configure_selection_gate(_args())  # reset
    assert eng._SELECTION_GATE_STATE["enabled"] is False


def test_explicit_cli_cap_overrides_config_cap(eng):
    """An explicit --max-per-symbol-basket wins over the strict-profile config cap
    (resolution order: CLI > --dedupe-baskets default > config > OFF)."""
    eng._configure_selection_gate(
        _args(config="configs/profiles/research.yaml", max_per_symbol_basket=3)
    )
    assert eng._SELECTION_GATE_STATE["max_per_symbol_basket"] == 3
    # lineage/family still resolve from the config.
    assert eng._SELECTION_GATE_STATE["max_per_lineage"] == 1
    eng._configure_selection_gate(_args())  # reset


def test_explicit_cli_floor_overrides_config(eng):
    eng._configure_selection_gate(
        _args(config="configs/profiles/research.yaml", dsr_gate_floor=0.75)
    )
    assert eng._SELECTION_GATE_STATE["robust_score_params"]["dsr_gate_floor"] == 0.75
    eng._configure_selection_gate(_args())  # reset
