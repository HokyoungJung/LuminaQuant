from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSERTION_PATH = ROOT / "scripts" / "research" / "assert_alpha_zoo_10bps_full_retune_artifact.py"
SPEC = importlib.util.spec_from_file_location(
    "assert_alpha_zoo_10bps_full_retune_artifact", ASSERTION_PATH
)
assert SPEC and SPEC.loader
ASSERTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ASSERTION
SPEC.loader.exec_module(ASSERTION)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tv_only_policy() -> dict[str, object]:
    return {
        "objective_inputs": ["train", "validation"],
        "selection_inputs": ["train", "validation"],
        "optimization_input_splits": ["train", "validation"],
        "parameter_fit_inputs": ["train", "validation"],
        "pruning_inputs": ["train", "validation"],
    }


def _selection_profile(*, selected_model_id: str, formula: str, consequence: str) -> dict[str, object]:
    return {
        **_tv_only_policy(),
        "score_formula_inputs": ["train", "validation"],
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "score_formula": formula,
        "selected_model_id": selected_model_id,
        "risk_profile_consequence": consequence,
    }


def _low_correlation_policy() -> dict[str, object]:
    return {
        **_tv_only_policy(),
        "correlation_inputs": ["train", "validation"],
        "correlation_split_inputs": ["train", "validation"],
        "candidate_freeze_inputs": ["train", "validation"],
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "uses_locked_oos_for_correlation": False,
        "uses_locked_oos_for_discovery": False,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
        "reference_model_id": ASSERTION.EXPECTED_HIGHER_RISK_MODEL_ID,
    }


def _payload(
    *, winner: str | None = ASSERTION.EXPECTED_HIGHER_RISK_MODEL_ID
) -> dict[str, object]:
    evidence = {
        "diagnostic_only": True,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT"],
    }
    higher_risk_model = ASSERTION.EXPECTED_HIGHER_RISK_MODEL_ID
    balanced_model = ASSERTION.EXPECTED_BALANCED_REFERENCE_MODEL_ID
    return {
        "real_money_execution": False,
        "round_trip_slippage_fee_bps_primary": 10.0,
        "live_promotable_10bps_model_id": winner,
        "active_selection_profile": ASSERTION.ACTIVE_SELECTION_PROFILE,
        "balanced_reference_10bps_model_id": balanced_model,
        "selection_profiles": {
            ASSERTION.BALANCED_SELECTION_PROFILE: _selection_profile(
                selected_model_id=balanced_model,
                formula="8.0*validation_total_return + 0.5*train_total_return - validation_drawdown_penalty",
                consequence="Balanced reference preserves lower leverage and allocation.",
            ),
            ASSERTION.ACTIVE_SELECTION_PROFILE: _selection_profile(
                selected_model_id=higher_risk_model,
                formula="6.0*train_total_return + 3.0*validation_total_return - validation_drawdown_penalty",
                consequence="Higher-risk final accepts 7x/0.20 drawdown risk after 10bps gates.",
            ),
        },
        "split_manifest": {
            "split_contract": ASSERTION.EXPECTED_SPLIT_CONTRACT,
            "timestamp_index_hash": ASSERTION.EXPECTED_TIMESTAMP_INDEX_HASH,
        },
        "memory_summary": {
            "limit_mib": 8192.0,
            "peak_rss_mib": 512.0,
            "pass_under_8gb": True,
            "guard_status": "pass",
            "pass_fail_reason": "peak RSS under 8192 MiB",
        },
        "locked_oos_contamination_audit": {
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
            "objective_inputs": ["train", "validation"],
            "selection_inputs": ["train", "validation"],
            "parameter_fit_inputs": ["train", "validation"],
            "pruning_inputs": ["train", "validation"],
        },
        "selection_policy": {
            **_tv_only_policy(),
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
        },
        "low_correlation_discovery": {
            "reference_model_id": higher_risk_model,
            "discovery_policy": _low_correlation_policy(),
        },
        "execution_cost_evidence": evidence,
    }


def _metric_rows(
    *,
    model_id: str = ASSERTION.EXPECTED_HIGHER_RISK_MODEL_ID,
    cost_bps: float = 10.0,
    promotable: bool = True,
    overrides: dict[tuple[str, str], object] | None = None,
    metadata: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    split_values = {
        "train": {
            "total_return": 0.30,
            "max_drawdown": 0.05,
            "sharpe": 3.0,
            "sortino": 4.0,
            "smart_sortino": 3.5,
            "calmar": 6.0,
        },
        "validation": {
            "total_return": 0.12,
            "max_drawdown": 0.04,
            "sharpe": 1.5,
            "sortino": 2.0,
            "smart_sortino": 1.8,
            "calmar": 3.0,
        },
        "locked_oos": {
            "total_return": 0.06,
            "max_drawdown": 0.03,
            "sharpe": 0.9,
            "sortino": 1.2,
            "smart_sortino": 1.1,
            "calmar": 2.0,
        },
    }
    for (split, key), value in (overrides or {}).items():
        split_values[split][key] = value
    rows: list[dict[str, object]] = []
    for split, values in split_values.items():
        rows.append(
            {
                "model_id": model_id,
                "round_trip_slippage_fee_bps": cost_bps,
                "split": split,
                "liquidation_count": 0,
                "account_wipeout_count": 0,
                "minimum_margin_buffer": 100.0,
                "candidate_universe_uses_locked_oos_bucket": False,
                "regenerated_train_validation_only": True,
                "promotability_scope": "live_candidate",
                "live_promotable_10bps": promotable,
                **values,
                **(metadata or {}),
            }
        )
    return rows


def _default_metric_rows() -> list[dict[str, object]]:
    return [
        *_metric_rows(model_id=ASSERTION.EXPECTED_HIGHER_RISK_MODEL_ID, promotable=True),
        *_metric_rows(
            model_id=ASSERTION.EXPECTED_BALANCED_REFERENCE_MODEL_ID,
            promotable=False,
            metadata={"role": "balanced_reference"},
        ),
    ]


def _variant_rows(
    *, calendar_primary: bool = False, params: dict[str, object] | None = None
) -> list[dict[str, object]]:
    return [
        {
            "model_id": "candidate_a",
            "calendar_primary": calendar_primary,
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
            "params_json": json.dumps(params or {"entry_threshold": 1.25, "max_hold_bars": 36}),
        }
    ]


def _low_correlation_json() -> dict[str, object]:
    return {
        "reference_model_id": ASSERTION.EXPECTED_HIGHER_RISK_MODEL_ID,
        "selection_profile": ASSERTION.ACTIVE_SELECTION_PROFILE,
        "reference_profile": ASSERTION.ACTIVE_SELECTION_PROFILE,
        "discovery_policy": _low_correlation_policy(),
        "summary": {
            "row_count": 2,
            "deployable_pass_count": 1,
            "research_only_locked_oos_fail_count": 1,
        },
    }


def _low_correlation_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_model_id": "low_corr_deployable",
            "candidate_family": "carry_cross_section",
            "variant_name": "base",
            "candidate_variant_name": "base",
            "train_validation_correlation_to_reference": 0.21,
            "correlation_train_validation": 0.21,
            "correlation_train_validation_abs": 0.21,
            "selection_correlation_split_inputs": "train;validation",
            "correlation_inputs": "train;validation",
            "selection_inputs": "train;validation",
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_correlation": False,
            "locked_oos_gate_pass": True,
            "locked_oos_gate_reasons": "",
            "deployability_label": "deployable_10bps_gate_pass",
            "train_total_return": 0.11,
            "validation_total_return": 0.05,
            "locked_oos_total_return": 0.03,
        },
        {
            "candidate_model_id": "low_corr_research_only",
            "candidate_family": "mean_reversion_tail",
            "variant_name": "base",
            "candidate_variant_name": "base",
            "train_validation_correlation_to_reference": -0.13,
            "correlation_train_validation": -0.13,
            "correlation_train_validation_abs": 0.13,
            "selection_correlation_split_inputs": "train;validation",
            "correlation_inputs": "train;validation",
            "selection_inputs": "train;validation",
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_correlation": False,
            "locked_oos_gate_pass": False,
            "locked_oos_gate_reasons": "locked_oos_sharpe_non_positive",
            "deployability_label": "research_only_locked_oos_gate_fail",
            "train_total_return": 0.14,
            "validation_total_return": 0.06,
            "locked_oos_total_return": -0.01,
        },
    ]


def _write_artifact(
    root: Path,
    *,
    payload: dict[str, object] | None = None,
    metric_rows: list[dict[str, object]] | None = None,
    variant_rows: list[dict[str, object]] | None = None,
    low_correlation_json: dict[str, object] | None = None,
    low_correlation_rows: list[dict[str, object]] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = _payload() if payload is None else payload
    _write_json(root / "alpha_zoo_10bps_full_retune_latest.json", payload)
    _write_json(
        root / "execution_cost_evidence_latest.json", dict(payload["execution_cost_evidence"])
    )
    _write_json(
        root / ASSERTION.LOW_CORRELATION_DISCOVERY_JSON,
        low_correlation_json or _low_correlation_json(),
    )
    _write_json(root / "tuned_seed_selection_latest.json", {"selected": ["candidate_a"]})
    _write_csv(root / "tuned_seed_selection_latest.csv", [{"model_id": "candidate_a", "rank": 1}])
    _write_csv(root / "candidate_model_metrics_latest.csv", metric_rows or _default_metric_rows())
    _write_csv(root / "candidate_variant_inventory_latest.csv", variant_rows or _variant_rows())
    _write_csv(
        root / ASSERTION.LOW_CORRELATION_DISCOVERY_CSV,
        low_correlation_rows or _low_correlation_rows(),
    )


def test_artifact_assertion_accepts_locked_oos_report_only_contract(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    summary = ASSERTION.validate_artifact(tmp_path)

    assert summary == {
        "artifact_dir": str(tmp_path),
        "models": 2,
        "metric_rows": 6,
        "low_correlation_rows": 2,
        "promotable": [ASSERTION.EXPECTED_HIGHER_RISK_MODEL_ID],
    }


@pytest.mark.parametrize(
    ("payload_update", "message"),
    [
        (
            {"locked_oos_contamination_audit": {"uses_locked_oos_for_selection": True}},
            "uses_locked_oos_for_selection",
        ),
        (
            {
                "selection_policy": {
                    "optimization_input_splits": ["train", "validation", "locked_oos"]
                }
            },
            "optimization_input_splits",
        ),
    ],
)
def test_artifact_assertion_rejects_locked_oos_selection_inputs(
    tmp_path: Path,
    payload_update: dict[str, object],
    message: str,
) -> None:
    payload = _payload()
    for key, value in payload_update.items():
        payload[key] = {**dict(payload[key]), **dict(value)}
    _write_artifact(tmp_path, payload=payload)

    with pytest.raises(ASSERTION.ArtifactAssertionError, match=message):
        ASSERTION.validate_artifact(tmp_path)


def test_artifact_assertion_requires_active_higher_risk_profile(tmp_path: Path) -> None:
    payload = _payload()
    payload["active_selection_profile"] = ASSERTION.BALANCED_SELECTION_PROFILE
    _write_artifact(tmp_path, payload=payload)

    with pytest.raises(ASSERTION.ArtifactAssertionError, match="active_selection_profile"):
        ASSERTION.validate_artifact(tmp_path)


def test_artifact_assertion_rejects_locked_oos_profile_inputs(tmp_path: Path) -> None:
    payload = _payload()
    profiles = dict(payload["selection_profiles"])
    higher = dict(profiles[ASSERTION.ACTIVE_SELECTION_PROFILE])
    higher["selection_inputs"] = ["train", "locked_oos"]
    profiles[ASSERTION.ACTIVE_SELECTION_PROFILE] = higher
    payload["selection_profiles"] = profiles
    _write_artifact(tmp_path, payload=payload)

    with pytest.raises(ASSERTION.ArtifactAssertionError, match="selection_inputs"):
        ASSERTION.validate_artifact(tmp_path)


def test_artifact_assertion_rejects_locked_oos_discovery_inputs(tmp_path: Path) -> None:
    discovery = _low_correlation_json()
    policy = dict(discovery["discovery_policy"])
    policy["correlation_inputs"] = ["train", "validation", "locked_oos"]
    discovery["discovery_policy"] = policy
    _write_artifact(tmp_path, low_correlation_json=discovery)

    with pytest.raises(ASSERTION.ArtifactAssertionError, match="correlation_inputs"):
        ASSERTION.validate_artifact(tmp_path)


def test_artifact_assertion_requires_low_correlation_label_consistency(tmp_path: Path) -> None:
    rows = _low_correlation_rows()
    rows[1]["deployability_label"] = "deployable_10bps_gate_pass"
    _write_artifact(tmp_path, low_correlation_rows=rows)

    with pytest.raises(ASSERTION.ArtifactAssertionError, match="research-only"):
        ASSERTION.validate_artifact(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({("validation", "total_return"): 0.0}, "validation_total_return_non_positive"),
        ({("locked_oos", "sharpe"): 0.0}, "locked_oos_sharpe_non_positive"),
        ({("train", "total_return"): 0.05}, "train_total_return_not_gt_validation"),
        ({("locked_oos", "max_drawdown"): 0.30}, "locked_oos_mdd_gt_25pct"),
        (
            {("locked_oos", "minimum_margin_buffer"): 0.0},
            "locked_oos_minimum_margin_buffer_non_positive",
        ),
        ({("validation", "liquidation_count"): 1}, "validation_liquidation_count_positive"),
    ],
)
def test_artifact_assertion_reports_exact_promotion_gate_failures(
    overrides: dict[tuple[str, str], object],
    expected_reason: str,
) -> None:
    failures = ASSERTION._promotion_gate_failures_for_model(_metric_rows(overrides=overrides))

    assert expected_reason in failures


def test_artifact_assertion_requires_exact_10bps_train_validation_locked_oos_rows(
    tmp_path: Path,
) -> None:
    missing_locked_oos = [row for row in _metric_rows() if row["split"] != "locked_oos"]
    _write_artifact(tmp_path, metric_rows=missing_locked_oos)

    with pytest.raises(ASSERTION.ArtifactAssertionError, match="10bps splits"):
        ASSERTION.validate_artifact(tmp_path)


def test_artifact_assertion_rejects_shadow_only_without_fresh_train_validation_model(
    tmp_path: Path,
) -> None:
    _write_artifact(
        tmp_path,
        payload=_payload(winner=None),
        metric_rows=_metric_rows(
            promotable=False,
            metadata={
                "candidate_universe_uses_locked_oos_bucket": True,
                "regenerated_train_validation_only": False,
                "promotability_scope": "shadow_only",
            },
        ),
    )

    with pytest.raises(ASSERTION.ArtifactAssertionError, match="fresh train\\+validation-only"):
        ASSERTION.validate_artifact(tmp_path)


def test_artifact_assertion_rejects_non_10bps_promotion_rows(tmp_path: Path) -> None:
    _write_artifact(tmp_path, metric_rows=_metric_rows(cost_bps=5.0))

    with pytest.raises(ASSERTION.ArtifactAssertionError, match="must use exactly 10bps"):
        ASSERTION.validate_artifact(tmp_path)


def test_artifact_assertion_rejects_calendar_primary_or_calendar_param_keys(
    tmp_path: Path,
) -> None:
    _write_artifact(tmp_path, variant_rows=_variant_rows(calendar_primary=True))
    with pytest.raises(ASSERTION.ArtifactAssertionError, match="calendar_primary=true"):
        ASSERTION.validate_artifact(tmp_path)

    other = tmp_path / "calendar-param"
    _write_artifact(other, variant_rows=_variant_rows(params={"month_filter": [1, 2]}))
    with pytest.raises(ASSERTION.ArtifactAssertionError, match="calendar/date parameter"):
        ASSERTION.validate_artifact(other)
