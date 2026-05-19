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


def _payload(*, winner: str | None = "candidate_a") -> dict[str, object]:
    evidence = {
        "diagnostic_only": True,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT"],
    }
    return {
        "real_money_execution": False,
        "round_trip_slippage_fee_bps_primary": 10.0,
        "live_promotable_10bps_model_id": winner,
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
        },
        "selection_policy": {
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "optimization_input_splits": ["train", "validation"],
        },
        "execution_cost_evidence": evidence,
    }


def _metric_rows(
    *,
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
                "model_id": "candidate_a",
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


def _write_artifact(
    root: Path,
    *,
    payload: dict[str, object] | None = None,
    metric_rows: list[dict[str, object]] | None = None,
    variant_rows: list[dict[str, object]] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = _payload() if payload is None else payload
    _write_json(root / "alpha_zoo_10bps_full_retune_latest.json", payload)
    _write_json(
        root / "execution_cost_evidence_latest.json", dict(payload["execution_cost_evidence"])
    )
    _write_json(root / "tuned_seed_selection_latest.json", {"selected": ["candidate_a"]})
    _write_csv(root / "tuned_seed_selection_latest.csv", [{"model_id": "candidate_a", "rank": 1}])
    _write_csv(root / "candidate_model_metrics_latest.csv", metric_rows or _metric_rows())
    _write_csv(root / "candidate_variant_inventory_latest.csv", variant_rows or _variant_rows())


def test_artifact_assertion_accepts_locked_oos_report_only_contract(tmp_path: Path) -> None:
    _write_artifact(tmp_path)

    summary = ASSERTION.validate_artifact(tmp_path)

    assert summary == {
        "artifact_dir": str(tmp_path),
        "models": 1,
        "metric_rows": 3,
        "promotable": ["candidate_a"],
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
