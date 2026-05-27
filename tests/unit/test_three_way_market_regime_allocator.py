import importlib.util
import sys
from pathlib import Path

import pandas as pd

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_three_way_market_regime_allocator.py"
)
SPEC = importlib.util.spec_from_file_location("run_three_way_market_regime_allocator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_raw_target_state_routes_to_expected_group() -> None:
    signal = {
        "favored_group": "incumbent",
        "confidence": 1.0,
        "max_signal_score": 0.05,
    }
    params = MODULE.AllocatorParams(
        min_confidence=0.2, min_signal_score=0.001, confirmation_days=1, min_hold_days=1
    )
    assert MODULE._raw_target_state(signal, params=params) == "incumbent"

    signal = {
        "favored_group": "autoresearch",
        "confidence": 0.9,
        "max_signal_score": 0.01,
    }
    assert MODULE._raw_target_state(signal, params=params) == "autoresearch_55_45"

    weak = {
        "favored_group": "autoresearch",
        "confidence": 0.1,
        "max_signal_score": 0.0005,
    }
    assert MODULE._raw_target_state(weak, params=params) == "blend_85_15"


def test_run_allocator_honors_confirmation_and_hold_days() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01", tz="UTC"),
                "split_group": "train",
                "incumbent": 0.01,
                "blend_85_15": 0.005,
                "autoresearch_55_45": 0.0,
                "favored_group": "incumbent",
                "confidence": 1.0,
                "incumbent_score": 0.05,
                "autoresearch_score": 0.0,
                "max_signal_score": 0.05,
                "active_rules": [],
            },
            {
                "date": pd.Timestamp("2025-01-02", tz="UTC"),
                "split_group": "train",
                "incumbent": 0.02,
                "blend_85_15": 0.005,
                "autoresearch_55_45": -0.01,
                "favored_group": "incumbent",
                "confidence": 1.0,
                "incumbent_score": 0.05,
                "autoresearch_score": 0.0,
                "max_signal_score": 0.05,
                "active_rules": [],
            },
            {
                "date": pd.Timestamp("2025-01-03", tz="UTC"),
                "split_group": "train",
                "incumbent": -0.02,
                "blend_85_15": 0.001,
                "autoresearch_55_45": 0.03,
                "favored_group": "autoresearch",
                "confidence": 1.0,
                "incumbent_score": 0.0,
                "autoresearch_score": 0.01,
                "max_signal_score": 0.01,
                "active_rules": [],
            },
            {
                "date": pd.Timestamp("2025-01-04", tz="UTC"),
                "split_group": "train",
                "incumbent": -0.02,
                "blend_85_15": 0.001,
                "autoresearch_55_45": 0.03,
                "favored_group": "autoresearch",
                "confidence": 1.0,
                "incumbent_score": 0.0,
                "autoresearch_score": 0.01,
                "max_signal_score": 0.01,
                "active_rules": [],
            },
        ]
    )
    params = MODULE.AllocatorParams(
        min_confidence=0.0, min_signal_score=0.0, confirmation_days=2, min_hold_days=1
    )
    result = MODULE._run_allocator(panel=frame, params=params)
    states = result["state_frame"]["state"].tolist()
    assert states == ["blend_85_15", "incumbent", "incumbent", "autoresearch_55_45"]


def test_run_allocator_can_enter_autoresearch_faster_than_default() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01", tz="UTC"),
                "split_group": "train",
                "incumbent": 0.0,
                "blend_85_15": 0.0,
                "autoresearch_55_45": 0.0,
                "favored_group": "incumbent",
                "confidence": 1.0,
                "incumbent_score": 0.05,
                "autoresearch_score": 0.0,
                "max_signal_score": 0.05,
                "active_rules": [],
            },
            {
                "date": pd.Timestamp("2025-01-02", tz="UTC"),
                "split_group": "train",
                "incumbent": 0.0,
                "blend_85_15": 0.0,
                "autoresearch_55_45": 0.0,
                "favored_group": "incumbent",
                "confidence": 1.0,
                "incumbent_score": 0.05,
                "autoresearch_score": 0.0,
                "max_signal_score": 0.05,
                "active_rules": [],
            },
            {
                "date": pd.Timestamp("2025-01-03", tz="UTC"),
                "split_group": "train",
                "incumbent": 0.0,
                "blend_85_15": 0.0,
                "autoresearch_55_45": 0.0,
                "favored_group": "incumbent",
                "confidence": 1.0,
                "incumbent_score": 0.05,
                "autoresearch_score": 0.0,
                "max_signal_score": 0.05,
                "active_rules": [],
            },
            {
                "date": pd.Timestamp("2025-01-04", tz="UTC"),
                "split_group": "train",
                "incumbent": -0.02,
                "blend_85_15": 0.0,
                "autoresearch_55_45": 0.03,
                "favored_group": "autoresearch",
                "confidence": 1.0,
                "incumbent_score": 0.0,
                "autoresearch_score": 0.01,
                "max_signal_score": 0.01,
                "active_rules": [],
            },
        ]
    )
    symmetric = MODULE.AllocatorParams(
        min_confidence=0.0, min_signal_score=0.0, confirmation_days=2, min_hold_days=2
    )
    asymmetric = MODULE.AllocatorParams(
        min_confidence=0.0,
        min_signal_score=0.0,
        confirmation_days=2,
        min_hold_days=2,
        enter_autoresearch_confirmation_days=1,
        enter_autoresearch_min_hold_days=1,
    )

    symmetric_states = MODULE._run_allocator(panel=frame, params=symmetric)["state_frame"][
        "state"
    ].tolist()
    asymmetric_states = MODULE._run_allocator(panel=frame, params=asymmetric)["state_frame"][
        "state"
    ].tolist()

    assert symmetric_states == ["blend_85_15", "blend_85_15", "incumbent", "incumbent"]
    assert asymmetric_states == ["blend_85_15", "blend_85_15", "incumbent", "autoresearch_55_45"]
