from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py"
SPEC = importlib.util.spec_from_file_location("run_profit_moonshot_hybrid_v35_v36_fixed_inputs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fixed_input_universe_has_no_nested_hybrid_sources() -> None:
    assert MODULE.FIXED_INPUT_ORDER == ("A0", "P0", "E0", "S1", "S2", "S3", "S4")


def test_train_val_score_does_not_accept_locked_oos() -> None:
    assert list(inspect.signature(MODULE._train_val_score).parameters) == ["train", "val"]


def test_v36_only_refreshes_default_candidate_from_v35_core_params() -> None:
    # Candidate 0 is the warmup/default leader, but candidate 1 becomes the
    # rolling-score leader later.  V3.5 must keep default_idx=0; V3.6 should
    # dynamically refresh only the default candidate, not weight/boost/cap knobs.
    returns = np.array(
        [[0.010, -0.010]] * 5
        + [[-0.010, 0.020]] * 30,
        dtype=float,
    )
    params = MODULE.HybridParams(
        bias_correction_alpha=0.5,
        bias_combine_ratio=0.0,
        max_single_weight=0.80,
        mape_window=3,
        bias_window=3,
        short_vol_window=2,
    )
    learned = MODULE.LearnedParams(
        high_vol_threshold=99.0,
        default_idx=0,
        high_vol_best_idx=0,
        default_weight_ratio=0.40,
        high_vol_weight_boost=0.12,
        cv_score=0.0,
    )

    _, allocations_v35 = MODULE._portfolio_returns_for_params(
        returns,
        params=params,
        learned=learned,
        version="v3_5",
        start_idx=0,
    )
    _, allocations_v36 = MODULE._portfolio_returns_for_params(
        returns,
        params=params,
        learned=learned,
        version="v3_6",
        start_idx=0,
    )

    assert allocations_v35[-1]["default_idx"] == 0
    assert allocations_v36[-1]["default_idx"] == 1
    for row in allocations_v36:
        assert row["adaptive_weight_ratio"] == learned.default_weight_ratio
        assert row["adaptive_high_vol_boost"] == learned.high_vol_weight_boost
        assert row["adaptive_max_single_weight"] == params.max_single_weight


def test_integrated_margin_replay_flags_account_liquidation_pressure() -> None:
    timestamps = np.array([1, 2, 3], dtype=np.int64)
    split_masks = {
        "train": np.array([True, False, False]),
        "validation": np.array([False, True, False]),
        "locked_oos": np.array([False, False, True]),
    }
    candidate_returns = np.array([[0.01], [0.01], [-1.10]], dtype=float)
    allocations = [
        {"index": 0, "exposure": 1.0, "weights": [1.0]},
        {"index": 1, "exposure": 1.0, "weights": [1.0]},
        {"index": 2, "exposure": 1.0, "weights": [1.0]},
    ]

    replay = MODULE._integrated_margin_replay(
        timestamps=timestamps,
        split_masks=split_masks,
        candidate_returns=candidate_returns,
        portfolio_returns=candidate_returns[:, 0],
        allocations=allocations,
        streams=[{"label": "A0", "leverage": 6.0, "allocation_fraction": 0.10}],
    )

    locked = replay["split_status"]["locked_oos"]
    assert replay["uses_locked_oos_for_selection"] is False
    assert locked["liquidation_count"] == 1
    assert replay["component_threshold_breach_counts_diagnostic"]["A0"] == 1
    assert locked["minimum_margin_buffer"] <= 0.0


def test_live_policy_can_promote_after_safe_integrated_margin_replay() -> None:
    returns = np.full((6, 1), 0.02, dtype=float)
    split_masks = {
        "train": np.array([True, True, False, False, False, False]),
        "validation": np.array([False, False, True, True, False, False]),
        "locked_oos": np.array([False, False, False, False, True, True]),
    }
    result = {
        "version": "v3_5",
        "params": MODULE.HybridParams(mape_window=2, bias_window=2, short_vol_window=2).__dict__,
        "learned_params": MODULE.LearnedParams(
            high_vol_threshold=99.0,
            default_idx=0,
            high_vol_best_idx=0,
            default_weight_ratio=1.0,
            high_vol_weight_boost=0.0,
            cv_score=0.0,
        ).__dict__,
        "splits": {
            "train": {"total_return": 0.10, "max_drawdown": 0.01, "sharpe": 1.0, "sortino": 1.0, "smart_sortino": 1.0, "calmar": 1.0},
            "validation": {"total_return": 0.10, "max_drawdown": 0.01, "sharpe": 1.0, "sortino": 1.0, "smart_sortino": 1.0, "calmar": 1.0},
            "locked_oos": {"total_return": 0.10, "max_drawdown": 0.01, "sharpe": 1.0, "sortino": 1.0, "smart_sortino": 1.0, "calmar": 1.0},
        },
        "train_val_score": 1.0,
        "train_val_gate": True,
        "allocations": [],
        "final_weights": [1.0],
        "portfolio_returns": returns[:, 0],
    }

    annotated = MODULE._annotate_live_policy(
        result,
        labels=["A0"],
        returns=returns,
        split_masks=split_masks,
        timestamps=np.arange(6, dtype=np.int64),
        streams=[{"label": "A0", "leverage": 6.0, "allocation_fraction": 0.10}],
    )

    assert annotated["margin_replay_available"] is True
    assert annotated["deployable_success"] is True
    assert annotated["rejection_reasons"] == []
    assert annotated["splits"]["locked_oos"]["liquidation_count"] == 0
    assert annotated["splits"]["locked_oos"]["minimum_margin_buffer"] > 0.0
