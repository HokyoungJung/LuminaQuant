from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_integer_leverage_hybrid_decision.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_integer_leverage_hybrid_decision", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _dates() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2025-01-01 00:00:00",
                "2025-01-01 01:00:00",
                "2026-01-01 00:00:00",
                "2026-01-01 01:00:00",
                "2026-04-01 00:00:00",
                "2026-04-01 01:00:00",
            ]
        )
    )


def _stream(profile_id: str, returns: list[float]) -> object:
    index = _dates()
    return MODULE.ProfileStream(
        profile_id=profile_id,
        candidate_tier="test",
        leverage_map={profile_id: 1},
        gross_notional_fraction=1.0,
        asset_gross_notional_fraction={profile_id: 1.0},
        selected_model_ids=(profile_id,),
        returns=pd.Series(returns, index=index, dtype=float),
        turnover_by_split={"train": 1.0, "validation": 1.0, "locked_oos": 1.0},
        trade_events_by_split={"train": 10, "validation": 10, "locked_oos": 10},
        liquidation_count_by_split={"train": 0, "validation": 0, "locked_oos": 0},
    )


def test_weight_grid_uses_all_three_profiles_with_minimum_weight() -> None:
    weights = list(MODULE._iter_weight_grid(["balanced", "growth", "aggressive"], step=0.05, min_weight=0.10))

    assert weights
    assert all(sum(row.values()) == pytest.approx(1.0) for row in weights)
    assert all(all(value >= 0.10 for value in row.values()) for row in weights)
    assert {"balanced", "growth", "aggressive"} == set(weights[0])


def test_hybrid_selection_ignores_locked_oos_when_freezing_weights() -> None:
    streams = [
        _stream("balanced", [0.30, 0.00, 0.10, 0.00, -0.90, 0.00]),
        _stream("growth", [0.40, 0.00, 0.15, 0.00, 0.01, 0.00]),
        _stream("aggressive", [0.50, 0.00, 0.20, 0.00, 0.02, 0.00]),
    ]
    selected, _ = MODULE.select_hybrid_row(streams)
    original_weights = selected["weights"]

    changed_oos = [
        _stream("balanced", [0.30, 0.00, 0.10, 0.00, 0.90, 0.00]),
        _stream("growth", [0.40, 0.00, 0.15, 0.00, -0.90, 0.00]),
        _stream("aggressive", [0.50, 0.00, 0.20, 0.00, -0.90, 0.00]),
    ]
    selected_changed, _ = MODULE.select_hybrid_row(changed_oos)

    assert selected_changed["weights"] == original_weights


def test_weighted_hybrid_row_keeps_real_money_disabled_and_cost_threshold() -> None:
    streams = [
        _stream("balanced", [0.30, 0.00, 0.10, 0.00, 0.01, 0.00]),
        _stream("growth", [0.40, 0.00, 0.15, 0.00, 0.01, 0.00]),
        _stream("aggressive", [0.50, 0.00, 0.20, 0.00, 0.01, 0.00]),
    ]
    row = MODULE._weighted_hybrid_row(
        streams,
        {"balanced": 0.2, "growth": 0.4, "aggressive": 0.4},
    )

    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False
    assert row["train_return_per_turnover_proxy_bps"] > MODULE.ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS
    assert MODULE.ilp.PRIMARY_ROUND_TRIP_COST_BPS == 10.0


def test_profile_corr_matrix_returns_finite_diagonal() -> None:
    streams = [
        _stream("balanced", [0.30, 0.00, 0.10, 0.00, 0.01, 0.00]),
        _stream("growth", [0.40, 0.00, 0.15, 0.00, 0.01, 0.00]),
        _stream("aggressive", [0.50, 0.00, 0.20, 0.00, 0.01, 0.00]),
    ]

    corr = MODULE._profile_corr_matrix(streams, split="train_validation")

    assert np.isfinite(corr["balanced"]["balanced"])
    assert corr["balanced"]["balanced"] == pytest.approx(1.0)
