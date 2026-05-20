from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_four_lane_shadow_discovery.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_four_lane_shadow_discovery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_four_lane_shadow_discovery_builds_paper_only_bundle(tmp_path: Path) -> None:
    payload = MODULE.build_payload(MODULE.parse_args(["--output-dir", str(tmp_path)]))

    assert payload["artifact_kind"] == "alpha_zoo_four_lane_shadow_discovery"
    assert payload["research_primary_round_trip_cost_bps"] == pytest.approx(10.0)
    assert payload["ready_for_paper"] is True
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False

    lanes = {row["role"]: row for row in payload["four_lane_paper_candidates"]}
    assert set(lanes) == {
        "active",
        "balanced",
        "validation_return_leader",
        "validation_efficiency_reference",
    }
    assert lanes["active"]["target_notional_fraction_of_equity"] == pytest.approx(1.4)
    assert lanes["balanced"]["target_notional_fraction_of_equity"] == pytest.approx(1.05)
    assert lanes["validation_return_leader"]["target_notional_fraction_of_equity"] == pytest.approx(1.0)
    assert lanes["validation_efficiency_reference"]["target_notional_fraction_of_equity"] == pytest.approx(0.7)
    assert all(row["notional_parity_passed"] for row in lanes.values())
    assert all(row["ready_for_paper"] for row in lanes.values())
    assert not any(row["ready_for_real"] for row in lanes.values())

    conservative = payload["shadow_discovery"]["conservative_exit_rescue_hypotheses"]
    assert conservative[0]["validation_return"] > 0.20
    assert conservative[0]["locked_oos_return"] < 0.0
    assert conservative[0]["shadow_status"] == "shadow_only_locked_oos_negative"
    assert payload["strategy_findings"]["conservative_exit_top_validation_locked_oos_positive_count"] == 0

    side_family = payload["shadow_discovery"]["side_family_threshold_hypotheses"]
    assert side_family[0]["validation_return"] > 0.05
    assert side_family[0]["locked_oos_return"] < 0.0
    assert payload["strategy_findings"]["side_family_positive_oos_zero_liq_in_top_shadow_count"] == 0

    quality = payload["shadow_discovery"]["quality_single_pair_surface"]
    assert quality["top_live_quality_candidates"][0]["model_id"] == (
        "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc"
    )
    assert "quality_single_pair" in quality["finding"]

    monitoring = payload["monitoring_contract"]
    assert monitoring["ready_for_real"] is False
    assert monitoring["real_money_execution"] is False
    assert len(monitoring["profile_rows"]) == 4

    for key in (
        "latest_json",
        "timestamped_json",
        "latest_markdown",
        "four_lane_csv",
        "shadow_csv",
        "monitoring_contract_json",
        "monitoring_contract_csv",
        "artifact_generation_validation_log",
    ):
        assert Path(payload["output_paths"][key]).exists()
