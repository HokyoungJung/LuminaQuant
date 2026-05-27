from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "validate_profit_moonshot_pass_under_8gb.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_profit_moonshot_pass_under_8gb", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _passing_candidate_payload() -> dict:
    return {
        "research_success_candidate": True,
        "metrics": {
            "train_monthlyized_return": 0.030,
            "validation_monthlyized_return": 0.110,
            "raw_train_monthlyized_return": 0.012,
            "raw_val_monthlyized_return": 0.030,
            "train_max_drawdown": 0.050,
            "validation_max_drawdown": 0.040,
            "train_sharpe": 2.0,
            "train_sortino": 2.0,
            "train_calmar": 4.0,
            "validation_sharpe": 4.5,
            "validation_sortino": 5.0,
            "validation_calmar": 35.0,
            "leverage": 2.0,
            "sleeve_count": 4,
            "locked_oos_monthlyized_return": 0.024,
            "locked_oos_total_return": 0.08,
            "locked_oos_max_drawdown": 0.008,
            "locked_oos_sharpe": 2.5,
            "locked_oos_sortino": 3.8,
            "locked_oos_smart_sortino": 3.2,
            "locked_oos_calmar": 10.0,
        },
        "liquidation_aware_validation": {
            "splits": {
                "train": {
                    "liquidation_count": 0,
                    "minimum_margin_buffer": 1000.0,
                    "minimum_margin_ratio": 3.0,
                },
                "validation": {
                    "liquidation_count": 0,
                    "minimum_margin_buffer": 1000.0,
                    "minimum_margin_ratio": 3.0,
                },
                "oos": {
                    "liquidation_count": 0,
                    "minimum_margin_buffer": 1000.0,
                    "minimum_margin_ratio": 3.0,
                },
            }
        },
    }


def test_validator_fails_running_result_with_missing_evidence(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write_json(result_path, {"status": "running", "passed": False})

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    failed = {check["name"] for check in payload["checks"] if not check["passed"]}
    assert "declared_pass_status" in failed
    assert "passing_candidate_artifact_exists" in failed
    assert "candidate_return_quality_contract" in failed
    assert "rss_under_8gib_evidence" in failed
    assert "local_tests_evidence" in failed
    assert "ci_success_evidence" in failed


def test_validator_passes_with_candidate_rss_tests_ci_and_push(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    time_log = tmp_path / "time.log"
    result_path = tmp_path / "result.json"
    _write_json(candidate_path, _passing_candidate_payload())
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    time_log.write_text("Maximum resident set size (kbytes): 65536\n", encoding="utf-8")
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": True,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json", "time.log"],
            "test_evidence": [{"command": "pytest", "passed": True}],
            "ci_evidence": [{"workflow": "ci", "conclusion": "success"}],
            "git_evidence": [{"remote": "private/main", "pushed": True}],
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is True
    assert all(check["passed"] for check in payload["checks"])
    assert [item["under_8gib"] for item in payload["rss_evidence"]] == [True, True]


def test_validator_rejects_low_monthly_return_candidate_even_with_old_pass_label(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    source_artifact = tmp_path / "source.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    _write_json(
        candidate_path,
        {
            "name": "old_low_return_pass",
            "research_success_candidate": True,
            "source_artifact": "source.json",
        },
    )
    _write_json(
        source_artifact,
        {
            "best_success_candidate": {
                "name": "old_low_return_pass",
                "splits": {
                    "train": {"metrics": {"cagr": 0.055, "total_return": 0.055}},
                    "val": {"metrics": {"cagr": 0.275, "total_return": 0.04}},
                    "oos": {
                        "metrics": {
                            "cagr": 0.076,
                            "total_return": 0.0134,
                            "max_drawdown": 0.0018,
                            "sharpe": 5.4,
                            "sortino": 6.7,
                            "calmar": 42.0,
                        }
                    },
                },
            }
        },
    )
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    quality_check = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    assert quality_check["passed"] is False
    details = json.loads(quality_check["detail"])
    assert details["train_monthlyized_return"] < validator.MIN_STABLE_MONTHLY_RETURN
    assert details["locked_oos_monthlyized_return"] < validator.MIN_STABLE_MONTHLY_RETURN


def test_validator_rejects_candidate_that_beats_old_champion_but_not_current_base(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    candidate = _passing_candidate_payload()
    candidate["metrics"]["locked_oos_total_return"] = validator.CURRENT_BASE_OOS_RETURN - 0.001
    _write_json(candidate_path, candidate)
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    quality_check = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    assert quality_check["passed"] is False
    details = json.loads(quality_check["detail"])
    assert details["current_champion_oos_return"] == validator.CURRENT_CHAMPION_OOS_RETURN
    assert details["current_base_oos_return"] == validator.CURRENT_BASE_OOS_RETURN
    assert details["locked_oos_total_return"] == pytest.approx(
        validator.CURRENT_BASE_OOS_RETURN - 0.001
    )


def test_validator_rejects_floor_fit_candidate_with_weak_raw_train(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    candidate = _passing_candidate_payload()
    candidate["metrics"]["train_monthlyized_return"] = 0.030
    candidate["metrics"]["raw_train_monthlyized_return"] = (
        validator.MIN_RAW_TRAIN_MONTHLY_RETURN - 0.001
    )
    _write_json(candidate_path, candidate)
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    quality_check = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    details = json.loads(quality_check["detail"])
    assert quality_check["passed"] is False
    assert details["train_monthlyized_return"] >= validator.MIN_BUFFERED_TRAIN_MONTHLY_RETURN
    assert details["raw_train_monthlyized_return"] < validator.MIN_RAW_TRAIN_MONTHLY_RETURN


def test_validator_rejects_fractional_leverage_candidate(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    candidate = _passing_candidate_payload()
    candidate["metrics"]["leverage"] = 2.5
    _write_json(candidate_path, candidate)
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    quality_check = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    details = json.loads(quality_check["detail"])
    assert quality_check["passed"] is False
    assert details["leverage"] == pytest.approx(2.5)


def test_validator_rejects_positive_liquidation_even_when_tolerance_metadata_allows_it(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    candidate = _passing_candidate_payload()
    candidate["liquidation_aware_validation"] = {
        "liquidation_tolerance": {"allowed_total_liquidations": 1},
        "splits": {
            "train": {"liquidation_count": 0, "minimum_margin_buffer": 1000.0},
            "validation": {"liquidation_count": 1, "minimum_margin_buffer": 1000.0},
            "oos": {"liquidation_count": 0, "minimum_margin_buffer": 1000.0},
        },
    }
    _write_json(candidate_path, candidate)
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    quality_check = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    details = json.loads(quality_check["detail"])
    assert quality_check["passed"] is False
    assert details["liquidation_count"] == 1
    assert details["allowed_total_liquidations"] == 1


def test_validator_rejects_promoted_candidate_with_missing_liquidation_margin_evidence(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    candidate = _passing_candidate_payload()
    candidate.pop("liquidation_aware_validation")
    _write_json(candidate_path, candidate)
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    quality_check = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    details = json.loads(quality_check["detail"])
    assert quality_check["passed"] is False
    assert details["liquidation_count"] is None
    assert details["minimum_margin_buffer"] is None


def test_validator_accepts_no_improvement_current_base_retention(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    candidate = {
        "labels": {
            "current_base_retained": True,
            "no_improvement_current_base_retained": True,
        },
        "promotion_status": "no_improvement_current_base_retained",
        "metrics": {
            "train_monthlyized_return": validator.CURRENT_BASE_TRAIN_MONTHLY_RETURN,
            "validation_monthlyized_return": validator.CURRENT_BASE_VAL_MONTHLY_RETURN,
            "train_max_drawdown": validator.CURRENT_BASE_TRAIN_MDD,
            "validation_max_drawdown": validator.CURRENT_BASE_VAL_MDD,
            "train_sharpe": validator.CURRENT_BASE_TRAIN_SHARPE,
            "train_sortino": validator.CURRENT_BASE_TRAIN_SORTINO,
            "train_calmar": validator.CURRENT_BASE_TRAIN_CALMAR,
            "validation_sharpe": validator.CURRENT_BASE_VAL_SHARPE,
            "validation_sortino": validator.CURRENT_BASE_VAL_SORTINO,
            "validation_calmar": validator.CURRENT_BASE_VAL_CALMAR,
            "leverage": validator.CURRENT_BASE_LEVERAGE,
            "sleeve_count": validator.CURRENT_BASE_SLEEVE_COUNT,
            "locked_oos_monthlyized_return": 0.030883445250837083,
            "locked_oos_total_return": validator.CURRENT_BASE_OOS_RETURN,
            "locked_oos_max_drawdown": validator.CURRENT_BASE_OOS_MDD,
            "locked_oos_sharpe": 5.653697867183353,
            "locked_oos_sortino": 7.396117977603192,
            "locked_oos_smart_sortino": 7.153592515941036,
            "locked_oos_calmar": 53.73501485217365,
        },
    }
    _write_json(candidate_path, candidate)
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "selection_outcome": "no_improvement_current_base_retained",
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is True
    quality_check = next(
        check for check in payload["checks"] if check["name"] == "candidate_return_quality_contract"
    )
    details = json.loads(quality_check["detail"])
    assert details["no_improvement_base_retained"] is True
    assert details["train_val_stability_score"] == pytest.approx(
        validator.CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE
    )


def test_validator_requires_mutex_evidence_when_heavy_run_is_declared(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    _write_json(candidate_path, _passing_candidate_payload())
    _write_json(rss_summary, {"peak_rss_bytes": 512 * 1024 * 1024})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
            "requires_heavy_mutex_evidence": True,
        },
    )

    missing = validator.validate(result_path, repo_root=tmp_path)
    mutex_check = next(
        check for check in missing["checks"] if check["name"] == "heavy_run_mutex_evidence"
    )
    assert mutex_check["passed"] is False

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["heavy_mutex_evidence"] = [
        {
            "lock_path": ".omx/locks/profit_moonshot_heavy.lock",
            "exclusive": True,
            "status": "completed",
            "overlap_check": "passed",
        }
    ]
    _write_json(result_path, payload)

    present = validator.validate(result_path, repo_root=tmp_path)
    mutex_check = next(
        check for check in present["checks"] if check["name"] == "heavy_run_mutex_evidence"
    )
    assert mutex_check["passed"] is True


def test_validator_rejects_over_budget_rss_evidence(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    rss_summary = tmp_path / "rss_summary.json"
    result_path = tmp_path / "result.json"
    _write_json(candidate_path, _passing_candidate_payload())
    _write_json(rss_summary, {"peak_rss_bytes": validator.EIGHT_GIB_BYTES + 1})
    _write_json(
        result_path,
        {
            "status": "passed",
            "passed": True,
            "source_changed": False,
            "passing_candidate_artifact": "candidate.json",
            "rss_under_8gb_logs": ["rss_summary.json"],
            "tests_passed": True,
            "ci_passed": True,
        },
    )

    payload = validator.validate(result_path, repo_root=tmp_path)

    assert payload["passed"] is False
    rss_check = next(
        check for check in payload["checks"] if check["name"] == "rss_under_8gib_evidence"
    )
    assert rss_check["passed"] is False
