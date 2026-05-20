from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_expanded_filter_shadow_selection.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_expanded_filter_shadow_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_expanded_filter_shadow_selection_finds_positive_oos_shadow_family(tmp_path: Path) -> None:
    payload = MODULE.build_payload(MODULE.parse_args(["--output-dir", str(tmp_path)]))

    assert payload["artifact_kind"] == "alpha_zoo_expanded_filter_shadow_selection"
    assert payload["research_primary_round_trip_cost_bps"] == pytest.approx(10.0)
    assert payload["ready_for_paper"] is False
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False

    summary = payload["expanded_retune_summary"]
    assert summary["model_count"] == 976
    assert summary["live_promotable_count"] == 56
    assert summary["positive_oos_shadow_candidate_count"] >= 20
    assert summary["conservative_exit_positive_oos_count"] == 0

    leader = payload["positive_oos_shadow_candidates"][0]
    assert leader["model_id"] == (
        "fresh_tv10_filter_family_crypto_residual_reversal_abs_score_ge_1p5_"
        "alpha_zoo_high_confidence_long_only_8p0x_0p2alloc"
    )
    assert leader["validation_return"] > 0.14
    assert leader["locked_oos_return"] > 0.0
    assert leader["live_promotable_10bps"] is False
    assert leader["ready_for_paper"] is False
    assert "train_total_return_not_above_validation" in leader["gate_reasons"]
    assert payload["decision"]["paper_execution_allowed"] is False
    assert payload["decision"]["shadow_observation_allowed"] is True

    for key in (
        "latest_json",
        "timestamped_json",
        "latest_markdown",
        "positive_oos_shadow_csv",
        "conservative_exit_positive_oos_csv",
        "artifact_generation_validation_log",
    ):
        assert Path(payload["output_paths"][key]).exists()
