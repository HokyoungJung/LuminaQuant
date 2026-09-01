from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_long_only_reversal_guarded_study.py"
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_long_only_reversal_guarded_study", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_long_only_reversal_guarded_study_keeps_family_shadow_only(tmp_path: Path) -> None:
    payload = MODULE.build_payload(MODULE.parse_args(["--output-dir", str(tmp_path)]))

    assert payload["artifact_kind"] == "alpha_zoo_long_only_reversal_guarded_study"
    assert payload["research_primary_round_trip_cost_bps"] == pytest.approx(10.0)
    assert payload["ready_for_paper"] is False
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["paper_execution_allowed"] is False
    assert payload["shadow_observation_allowed"] is True

    assert payload["selection_policy"]["uses_locked_oos_for_selection"] is False
    assert payload["selection_policy"]["locked_oos_role"] == "post_freeze_gate_report_only"

    summary = payload["guarded_study_summary"]
    assert summary["target_family_model_count"] == 43
    assert summary["strict_paper_guard_pass_count"] == 0
    assert summary["primary_10bps_promotion_gate_pass_count"] == 0
    assert summary["max_validation_trade_event_count"] == 18
    assert summary["max_locked_oos_trade_event_count"] == 13
    assert summary["max_train_validation_return_ratio"] < 0.5

    leader = payload["guarded_candidates"][0]
    assert leader["model_id"] == (
        "fresh_tv10_filter_family_crypto_residual_reversal_abs_score_ge_1p5_"
        "alpha_zoo_high_confidence_long_only_8p0x_0p2alloc"
    )
    assert leader["validation_return"] > 0.14
    assert leader["train_return"] > 0.05
    assert leader["locked_oos_return"] > 0.0
    assert leader["locked_oos_liquidation_count"] == pytest.approx(0.0)
    assert leader["validation_trade_event_count"] == 18
    assert leader["locked_oos_trade_event_count"] == 13
    assert leader["train_validation_guard_pass"] is False
    assert leader["locked_oos_report_gate_pass"] is False
    assert leader["paper_promotion_guard_pass"] is False
    assert "validation_trade_event_count_18_below_30" in leader["guard_fail_reasons"]
    assert "locked_oos_trade_event_count_13_below_20" in leader["guard_fail_reasons"]
    assert "primary_10bps_promotion_gate_failed" in leader["guard_fail_reasons"]
    assert "train_total_return_not_above_validation" in leader["primary_10bps_gate_reasons"]

    assert payload["decision"]["paper_lanes_to_keep_running"] == [
        MODULE.paper_preflight.ACTIVE_MODEL_ID,
        MODULE.paper_preflight.BALANCED_MODEL_ID,
        "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc",
        "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc",
    ]

    for key in (
        "latest_json",
        "timestamped_json",
        "latest_markdown",
        "guarded_candidates_csv",
        "guarded_references_csv",
        "artifact_generation_validation_log",
    ):
        assert Path(payload["output_paths"][key]).exists()
