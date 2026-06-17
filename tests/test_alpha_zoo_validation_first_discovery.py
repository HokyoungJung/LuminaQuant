from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_validation_first_discovery.py"
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_validation_first_discovery", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _split_metrics(*, model_id: str, locked_oos_return: float) -> dict[str, dict[str, object]]:
    return {
        "train": {
            "model_id": model_id,
            "total_return": 0.2,
            "max_drawdown": 0.1,
            "sharpe": 1.0,
        },
        "validation": {
            "model_id": model_id,
            "total_return": 0.05,
            "max_drawdown": 0.02,
            "sharpe": 0.8,
        },
        "locked_oos": {
            "model_id": model_id,
            "total_return": locked_oos_return,
            "max_drawdown": 0.03,
            "sharpe": 0.7,
        },
    }


def test_validation_rank_key_ignores_locked_oos_metrics() -> None:
    high_oos = _split_metrics(model_id="same", locked_oos_return=1.0)
    low_oos = _split_metrics(model_id="same", locked_oos_return=-1.0)

    assert MODULE._validation_rank_key(high_oos) == MODULE._validation_rank_key(low_oos)


def test_gate_reasons_preserves_semicolon_tokens() -> None:
    assert MODULE._gate_reasons({"primary_10bps_promotion_gate_reasons": "a;b;c"}) == [
        "a",
        "b",
        "c",
    ]
    assert MODULE._gate_reasons({"primary_10bps_promotion_gate_reasons": ["x", "y"]}) == [
        "x",
        "y",
    ]


def test_build_validation_first_discovery_from_frozen_10bps_sources(tmp_path: Path) -> None:
    payload = MODULE.build_payload(
        MODULE.parse_args(
            [
                "--output-dir",
                str(tmp_path),
            ]
        )
    )

    assert payload["artifact_kind"] == "alpha_zoo_validation_first_discovery"
    assert payload["research_primary_round_trip_cost_bps"] == pytest.approx(10.0)
    # Paper readiness was intentionally demoted (see "Demote weak validation Alpha
    # Zoo exposure before paper trials"); the frozen 10bps evidence now yields
    # ready_for_paper=False, so this asserts the current demoted gate state.
    assert payload["ready_for_paper"] is False
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["selection_policy"]["selection_inputs"] == ["train", "validation"]
    assert payload["selection_policy"]["uses_locked_oos_for_selection"] is False

    selected = {row["role"]: row for row in payload["selected_paper_candidates"]}
    leader = selected["validation_return_leader"]
    efficient = selected["validation_efficiency_reference"]
    assert leader["model_id"] == (
        "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc"
    )
    assert leader["validation_return"] > 0.005
    assert leader["preflight"]["status"]["ready_for_paper"] is False
    assert leader["preflight"]["status"]["ready_for_real"] is False
    assert leader["paper_equivalent_sizing"]["expected_replay_notional"] == pytest.approx(10_000.0)
    assert efficient["validation_mdd"] < leader["validation_mdd"]
    assert efficient["preflight"]["status"]["ready_for_paper"] is False
    assert efficient["preflight"]["status"]["ready_for_real"] is False

    quarantine = payload["high_validation_quarantine"]
    assert quarantine[0]["validation_return"] > 0.20
    assert quarantine[0]["locked_oos_return"] < 0.0
    assert "locked_oos_return_non_positive" in quarantine[0]["gate_reasons"]
    ceiling = payload["new_strategy_findings"]["validation_ceiling_audit"]
    assert ceiling["max_live_gate_validation_return"] == leader["validation_return"]
    assert (
        ceiling["candidate_count_with_validation_gt_1pct_and_positive_locked_oos_zero_liquidation"]
        == 0
    )

    for profile in payload["selection_profiles"].values():
        assert profile["uses_locked_oos_for_selection"] is False
        assert profile["locked_oos_role"] == "gate_report_only_after_validation_first_freeze"

    assert Path(payload["output_paths"]["latest_json"]).exists()
    assert Path(payload["output_paths"]["latest_markdown"]).exists()
    assert Path(payload["output_paths"]["monitoring_contract_json"]).exists()
    assert Path(payload["output_paths"]["monitoring_contract_csv"]).exists()
    assert "validation_return_leader" in Path(payload["output_paths"]["selected_csv"]).read_text(
        encoding="utf-8"
    )
