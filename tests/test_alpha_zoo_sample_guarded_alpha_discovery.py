from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_sample_guarded_alpha_discovery.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_sample_guarded_alpha_discovery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _model_ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row["model_id"]) for row in rows]


def test_sample_guarded_discovery_builds_no_promotion_shadow_bundle(tmp_path: Path) -> None:
    payload = MODULE.build_payload(MODULE.parse_args(["--output-dir", str(tmp_path)]))

    assert payload["artifact_kind"] == "alpha_zoo_sample_guarded_alpha_discovery"
    assert payload["research_primary_round_trip_cost_bps"] == pytest.approx(10.0)
    assert payload["ready_for_paper"] is False
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["paper_execution_allowed"] is False
    assert payload["decision"]["status"] == "no_new_paper_promotion_shadow_shortlist"

    thresholds = payload["promotion_thresholds"]
    assert thresholds["min_train_trade_event_count"] == 80
    assert thresholds["min_validation_trade_event_count"] == 30
    assert thresholds["min_locked_oos_trade_event_count_report_gate"] == 20
    assert thresholds["min_validation_return"] == pytest.approx(0.02)
    assert thresholds["min_train_validation_return_ratio"] == pytest.approx(0.50)
    assert thresholds["max_validation_mdd"] == pytest.approx(0.12)

    policy = payload["selection_policy"]
    assert policy["candidate_freeze_inputs"] == ["train", "validation"]
    assert policy["locked_oos_role"] == "gate_report_only_after_train_validation_profile_freeze"
    for key in (
        "uses_locked_oos_for_discovery",
        "uses_locked_oos_for_selection",
        "uses_locked_oos_for_objective",
        "uses_locked_oos_for_pruning",
        "uses_locked_oos_for_parameter_fitting",
        "uses_locked_oos_for_correlation",
    ):
        assert policy[key] is False

    summary = payload["sample_guarded_summary"]
    assert summary["candidate_count"] == 976
    assert summary["paper_candidate_count"] == 0
    assert summary["shadow_only_thin_sample_count"] > 0
    assert summary["historical_oos_bucket_quarantined_count"] == 20
    assert summary["calendar_quarantined_count"] == 0

    assert set(payload["selection_profiles"]) == {
        "validation_strength_v1",
        "train_validation_robustness_v1",
        "cost_efficiency_v1",
    }
    for profile in payload["selection_profiles"].values():
        assert profile["selection_inputs"] == ["train", "validation"]
        assert profile["uses_locked_oos_for_selection"] is False
        assert profile["uses_locked_oos_for_objective"] is False

    assert set(payload["profile_rankings"]) == set(payload["selection_profiles"])
    assert all(payload["profile_rankings"][profile_id] for profile_id in payload["selection_profiles"])

    grid = payload["grid_coverage"]
    assert "LONG" in grid["side_values_in_selected_metric_surface"]
    assert "crypto_residual_reversal" in grid["factor_families_in_selected_metric_surface"]
    assert 1.5 in grid["abs_factor_score_min_values_in_selected_metric_surface"]
    assert grid["calendar_quarantine_count"] == 0
    assert "no symbol-filtered variant survived" in grid["symbol_grid_note"]

    prior = payload["prior_shadow_findings"]
    assert prior["long_only_crypto_residual_reversal_model_count"] == 43
    assert prior["long_only_crypto_residual_reversal_paper_candidate_count"] == 0
    assert prior["keep_long_only_crypto_residual_reversal_shadow_only"] is True

    long_only_leader = next(
        row
        for row in payload["sample_guarded_candidates"]
        if row["model_id"]
        == "fresh_tv10_filter_family_crypto_residual_reversal_abs_score_ge_1p5_"
        "alpha_zoo_high_confidence_long_only_8p0x_0p2alloc"
    )
    assert long_only_leader["status"] == "shadow_only_thin_sample"
    assert long_only_leader["validation_trade_event_count"] == 18
    assert long_only_leader["locked_oos_trade_event_count"] == 13
    assert long_only_leader["ready_for_paper"] is False
    assert long_only_leader["ready_for_real"] is False
    assert long_only_leader["real_money_execution"] is False
    assert "validation_trade_event_count_18_below_30" in long_only_leader["rejection_reasons"]
    assert "locked_oos_trade_event_count_13_below_20" in long_only_leader["rejection_reasons"]

    assert len(payload["baseline_paper_lanes"]) == 4
    baseline_ids = {row["model_id"] for row in payload["baseline_paper_lanes"]}
    assert MODULE.paper_preflight.ACTIVE_MODEL_ID in baseline_ids
    assert MODULE.paper_preflight.BALANCED_MODEL_ID in baseline_ids
    baseline_contract = {
        row["role"]: (
            row["model_id"],
            row["leverage"],
            row["allocation_fraction"],
            row["expected_replay_notional_for_10000_equity"],
            row["live_notional_for_10000_equity"],
        )
        for row in payload["baseline_paper_lanes"]
    }
    assert baseline_contract == {
        "active": (
            MODULE.paper_preflight.ACTIVE_MODEL_ID,
            7.0,
            0.20,
            pytest.approx(14_000.0),
            pytest.approx(14_000.0),
        ),
        "balanced": (
            MODULE.paper_preflight.BALANCED_MODEL_ID,
            6.0,
            0.175,
            pytest.approx(10_500.0),
            pytest.approx(10_500.0),
        ),
        "validation_return_leader": (
            "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc",
            5.0,
            0.20,
            pytest.approx(10_000.0),
            pytest.approx(10_000.0),
        ),
        "validation_efficiency_reference": (
            "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc",
            4.0,
            0.175,
            pytest.approx(7_000.0),
            pytest.approx(7_000.0),
        ),
    }
    assert all(row["ready_for_real"] is False for row in payload["baseline_paper_lanes"])
    assert all(row["real_money_execution"] is False for row in payload["baseline_paper_lanes"])

    decisions = payload["paper_candidate_decisions"]
    assert decisions
    assert {row["decision"] for row in decisions} == {"no_promotion"}
    assert all(row["rejection_reasons"] for row in decisions)
    assert all(row["ready_for_real"] is False for row in decisions)
    assert all(row["real_money_execution"] is False for row in decisions)

    costs = payload["cost_sensitivity"]
    assert costs
    assert {row["round_trip_cost_bps"] for row in costs} == {5.0, 10.0, 15.0, 20.0}
    assert all(row["may_reduce_promotion_cost"] is False for row in costs)
    assert any(row["metric_source"] == "expanded_retune_primary_10bps" for row in costs)
    assert any(row["metric_source"] == "not_replayed_in_sample_guarded_runner" for row in costs)

    memory = payload["memory_summary"]
    assert memory["pass_under_8gb"] is True
    assert memory["peak_rss_mib"] < 8192.0

    for key in (
        "latest_json",
        "timestamped_json",
        "latest_markdown",
        "sample_guarded_candidates_csv",
        "paper_candidate_decisions_csv",
        "shadow_hypotheses_csv",
        "cost_sensitivity_csv",
        "artifact_generation_validation_log",
    ):
        assert Path(payload["output_paths"][key]).exists()

    candidates_csv = Path(payload["output_paths"]["sample_guarded_candidates_csv"])
    with candidates_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        assert next(reader)[:4] == ["selection_rank", "status", "decision", "model_id"]
        assert next(reader)

    decisions_csv = Path(payload["output_paths"]["paper_candidate_decisions_csv"])
    with decisions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_decision = next(reader)
    assert first_decision["decision"] == "no_promotion"
    assert first_decision["rejection_reasons"]

    markdown = Path(payload["output_paths"]["latest_markdown"]).read_text(encoding="utf-8")
    assert "no_new_paper_promotion_shadow_shortlist" in markdown
    assert "ready_for_real=false" in markdown

    generation_log = Path(payload["output_paths"]["artifact_generation_validation_log"]).read_text(
        encoding="utf-8"
    )
    for needle in (
        "primary_round_trip_cost_bps=10.0",
        "paper_candidate_count=0",
        "ready_for_real=false",
        "real_money_execution=false",
        "uses_locked_oos_for_selection=false",
        "memory_guard_status=pass",
    ):
        assert needle in generation_log


def test_calendar_primary_rows_are_quarantined_before_ranking() -> None:
    def row(split: str) -> dict[str, object]:
        return {
            "split": split,
            "model_id": "calendar_model",
            "candidate_name": "alpha_zoo_quality_single_pair",
            "model_kind": "synthetic",
            "role": "test",
            "variant_name": "calendar_test",
            "leverage": 5.0,
            "allocation_fraction": 0.2,
            "calendar_primary": split == "validation",
            "candidate_universe_uses_locked_oos_bucket": False,
            "primary_10bps_promotion_gate_pass": True,
            "live_promotable_10bps": True,
            "total_return": 0.05,
            "max_drawdown": 0.05,
            "sharpe": 1.0,
            "sortino": 1.0,
            "smart_sortino": 1.0,
            "calmar": 1.0,
            "trade_event_count": 100,
            "liquidation_count": 0,
            "account_wipeout_count": 0,
        }

    summary = MODULE._candidate_summary(
        "calendar_model",
        {"train": row("train"), "validation": row("validation"), "locked_oos": row("locked_oos")},
    )

    assert summary["calendar_quarantined"] is True
    assert summary["selection_eligible"] is False
    assert summary["status"] == "reject_or_quarantine"
    assert summary["ready_for_paper"] is False
    assert summary["ready_for_real"] is False
    assert summary["real_money_execution"] is False
    assert "calendar_primary_or_calendar_rule_quarantine" in summary["rejection_reasons"]


def test_sample_guarded_rankings_ignore_locked_oos_values() -> None:
    base_rows = [
        {
            "model_id": "alpha",
            "validation_return": 0.04,
            "validation_sharpe": 1.1,
            "validation_sortino": 1.2,
            "validation_calmar": 0.9,
            "validation_mdd": 0.04,
            "train_return": 0.06,
            "train_validation_return_ratio": 1.5,
            "train_trade_event_count": 100,
            "validation_trade_event_count": 40,
            "target_notional_fraction_of_equity": 0.7,
            "locked_oos_return": -0.99,
            "locked_oos_trade_event_count": 1,
            "locked_oos_liquidation_count": 9,
        },
        {
            "model_id": "beta",
            "validation_return": 0.03,
            "validation_sharpe": 1.0,
            "validation_sortino": 1.1,
            "validation_calmar": 0.8,
            "validation_mdd": 0.05,
            "train_return": 0.05,
            "train_validation_return_ratio": 1.4,
            "train_trade_event_count": 120,
            "validation_trade_event_count": 45,
            "target_notional_fraction_of_equity": 0.6,
            "locked_oos_return": 0.99,
            "locked_oos_trade_event_count": 500,
            "locked_oos_liquidation_count": 0,
        },
    ]
    mutated_rows = [dict(row) for row in base_rows]
    mutated_rows[0].update(
        locked_oos_return=100.0,
        locked_oos_trade_event_count=10_000,
        locked_oos_liquidation_count=0,
    )
    mutated_rows[1].update(
        locked_oos_return=-100.0,
        locked_oos_trade_event_count=0,
        locked_oos_liquidation_count=99,
    )

    assert _model_ids(MODULE._all_ranked_candidates(base_rows)) == _model_ids(
        MODULE._all_ranked_candidates(mutated_rows)
    )
    for profile_id in (
        "validation_strength_v1",
        "train_validation_robustness_v1",
        "cost_efficiency_v1",
    ):
        assert _model_ids(MODULE._rank_candidates(base_rows, profile_id, 2)) == _model_ids(
            MODULE._rank_candidates(mutated_rows, profile_id, 2)
        )


def test_non_10bps_source_retune_is_rejected(tmp_path: Path) -> None:
    non_10bps = tmp_path / "non_10bps_retune.json"
    non_10bps.write_text(
        '{"artifact_kind":"alpha_zoo_10bps_full_retune",'
        '"round_trip_slippage_fee_bps_primary":5.0,'
        '"real_money_execution":false}\n',
        encoding="utf-8",
    )

    args = MODULE.parse_args(["--expanded-retune-json", str(non_10bps), "--output-dir", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="requires a 10bps expanded retune artifact"):
        MODULE.build_payload(args)
