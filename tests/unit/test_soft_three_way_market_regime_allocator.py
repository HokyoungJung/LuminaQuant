import importlib.util
import sys
from pathlib import Path

import pandas as pd

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_soft_three_way_market_regime_allocator.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_soft_three_way_market_regime_allocator", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_target_weights_fall_back_to_blend_when_signal_is_weak() -> None:
    params = MODULE.SoftAllocatorParams(
        min_confidence=0.2,
        min_signal_score=0.001,
        confidence_scale=0.5,
        score_scale=0.01,
        blend_floor=0.2,
        alpha=0.5,
        max_daily_turnover=0.2,
    )
    weights = MODULE._target_weights(
        {
            "confidence": 0.05,
            "max_signal_score": 0.0001,
            "incumbent_score": 0.0,
            "autoresearch_score": 0.0,
        },
        params=params,
    )
    assert weights == {"incumbent": 0.0, "blend_85_15": 1.0, "autoresearch_55_45": 0.0}


def test_apply_smoothing_respects_turnover_cap() -> None:
    params = MODULE.SoftAllocatorParams(
        min_confidence=0.0,
        min_signal_score=0.0,
        confidence_scale=0.5,
        score_scale=0.01,
        blend_floor=0.0,
        alpha=1.0,
        max_daily_turnover=0.1,
    )
    previous = {"incumbent": 0.0, "blend_85_15": 1.0, "autoresearch_55_45": 0.0}
    target = {"incumbent": 1.0, "blend_85_15": 0.0, "autoresearch_55_45": 0.0}
    weights, turnover = MODULE._apply_smoothing(previous=previous, target=target, params=params)
    assert turnover <= 0.1000001
    assert weights["blend_85_15"] > 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_run_soft_allocator_computes_weighted_returns() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01", tz="UTC"),
                "split_group": "train",
                "incumbent": 0.01,
                "blend_85_15": 0.005,
                "autoresearch_55_45": 0.0,
                "favored_group": "mixed",
                "confidence": 0.0,
                "incumbent_score": 0.0,
                "autoresearch_score": 0.0,
                "max_signal_score": 0.0,
                "active_rules": [],
            },
            {
                "date": pd.Timestamp("2025-01-02", tz="UTC"),
                "split_group": "train",
                "incumbent": 0.02,
                "blend_85_15": 0.01,
                "autoresearch_55_45": -0.01,
                "favored_group": "incumbent",
                "confidence": 1.0,
                "incumbent_score": 0.08,
                "autoresearch_score": 0.0,
                "max_signal_score": 0.08,
                "active_rules": [],
            },
        ]
    )
    params = MODULE.SoftAllocatorParams(
        min_confidence=0.0,
        min_signal_score=0.0,
        confidence_scale=1.0,
        score_scale=0.08,
        blend_floor=0.0,
        alpha=1.0,
        max_daily_turnover=1.0,
    )
    result = MODULE._run_soft_allocator(panel=frame, params=params)
    state_frame = result["state_frame"]
    assert state_frame.loc[0, "return"] == 0.005
    assert abs(state_frame.loc[1, "return"] - 0.02) < 1e-12
    assert state_frame.loc[1, "state"] == "incumbent"
