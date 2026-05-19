#!/usr/bin/env python3
"""Assert Alpha Zoo 10bps full-retune artifacts obey locked-OOS gate/report rules."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PRIMARY_COST_BPS = 10.0
MEMORY_LIMIT_MIB = 8192.0
MAX_LOCKED_OOS_MDD = 0.25
TARGET_ARTIFACT_DIR = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_10bps_full_retune_20260519"
)
SPLIT_ORDER = ("train", "validation", "locked_oos")
PROMOTION_METRICS = ("total_return", "sharpe", "sortino", "smart_sortino", "calmar")
POSITIVE_SPLIT_METRICS = ("total_return", "sharpe", "sortino", "smart_sortino", "calmar")
LOCKED_OOS_FLAGS = (
    "uses_locked_oos_for_objective",
    "uses_locked_oos_for_selection",
    "uses_locked_oos_for_pruning",
    "uses_locked_oos_for_parameter_fitting",
)
SELECTION_INPUT_KEYS = (
    "objective_inputs",
    "selection_inputs",
    "optimization_input_splits",
    "parameter_fit_inputs",
    "pruning_inputs",
    "hybrid_selection_inputs",
    "fit_splits",
)
CALENDAR_KEY_TOKENS = ("calendar", "date", "month", "day")
REQUIRED_COST_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT"}
EXPECTED_TIMESTAMP_INDEX_HASH = "b973165bc1057f3aaa08ea637b73a45df3e84fdb7d1337b1637233d205696bb0"
EXPECTED_SPLIT_CONTRACT = {
    "train": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:00:00Z"},
    "validation": {"start": "2026-01-01T00:00:00Z", "end": "2026-03-31T23:00:00Z"},
    "locked_oos": {"start": "2026-04-01T00:00:00Z", "end": "2026-05-17T10:00:00Z"},
}
REQUIRED_FILES = (
    "alpha_zoo_10bps_full_retune_latest.json",
    "candidate_model_metrics_latest.csv",
    "candidate_variant_inventory_latest.csv",
    "tuned_seed_selection_latest.csv",
    "tuned_seed_selection_latest.json",
    "execution_cost_evidence_latest.json",
)


class ArtifactAssertionError(AssertionError):
    """Raised when the Alpha Zoo 10bps artifact violates the report-only gate."""


def _fail(message: str) -> None:
    raise ArtifactAssertionError(message)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_close(left: Any, right: float) -> bool:
    return math.isclose(_safe_float(left, float("nan")), float(right), rel_tol=0.0, abs_tol=1e-12)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail(f"missing required file: {path}")
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, Mapping):
        _fail(f"JSON root must be an object: {path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except FileNotFoundError:
        _fail(f"missing required file: {path}")


def _require_files(root: Path) -> None:
    for filename in REQUIRED_FILES:
        path = root / filename
        if not path.is_file():
            _fail(f"missing required file: {path}")


def _validate_real_money_and_cost(payload: Mapping[str, Any]) -> None:
    if payload.get("real_money_execution") is not False:
        _fail("real_money_execution must be boolean false")
    if not _is_close(payload.get("round_trip_slippage_fee_bps_primary"), PRIMARY_COST_BPS):
        _fail("round_trip_slippage_fee_bps_primary must be exactly 10.0")


def _validate_split_manifest(payload: Mapping[str, Any]) -> None:
    manifest = _as_mapping(payload.get("split_manifest"))
    contract = _as_mapping(manifest.get("split_contract"))
    for split, expected in EXPECTED_SPLIT_CONTRACT.items():
        actual = _as_mapping(contract.get(split))
        for boundary, expected_value in expected.items():
            if actual.get(boundary) != expected_value:
                _fail(f"split_contract.{split}.{boundary} drifted: {actual.get(boundary)!r}")
    if manifest.get("timestamp_index_hash") != EXPECTED_TIMESTAMP_INDEX_HASH:
        _fail("split_manifest.timestamp_index_hash drifted from locked latest-split hash")


def _validate_memory_summary(payload: Mapping[str, Any]) -> None:
    memory = _as_mapping(payload.get("memory_summary"))
    if not _is_close(memory.get("limit_mib"), MEMORY_LIMIT_MIB):
        _fail("memory_summary.limit_mib must be 8192.0")
    peak = _safe_float(memory.get("peak_rss_mib"), float("nan"))
    if not math.isfinite(peak):
        _fail("memory_summary.peak_rss_mib must be present and finite")
    if peak >= MEMORY_LIMIT_MIB:
        _fail("memory_summary.peak_rss_mib must be below 8192.0")
    if memory.get("pass_under_8gb") is not True:
        _fail("memory_summary.pass_under_8gb must be true")
    for key in ("guard_status", "pass_fail_reason"):
        if key not in memory:
            _fail(f"memory_summary.{key} is required")


def _input_values_include_locked_oos(values: Any) -> bool:
    if values is None:
        return False
    if isinstance(values, str):
        return values.strip().lower() in {"locked_oos", "oos"}
    if isinstance(values, Sequence):
        return any(_input_values_include_locked_oos(item) for item in values)
    return False


def _validate_selection_inputs(container: Mapping[str, Any], *, prefix: str) -> None:
    for key in SELECTION_INPUT_KEYS:
        if _input_values_include_locked_oos(container.get(key)):
            _fail(f"{prefix}.{key} must not include locked_oos/oos")


def _validate_locked_oos_audit(payload: Mapping[str, Any]) -> None:
    audit = _as_mapping(payload.get("locked_oos_contamination_audit"))
    if not audit:
        _fail("locked_oos_contamination_audit is required")
    for key in LOCKED_OOS_FLAGS:
        if audit.get(key) is not False:
            _fail(f"locked_oos_contamination_audit.{key} must be false")
    role = str(audit.get("locked_oos_role") or "")
    if "gate" not in role or "report" not in role:
        _fail("locked_oos_contamination_audit.locked_oos_role must be gate/report-only")
    _validate_selection_inputs(audit, prefix="locked_oos_contamination_audit")
    selection_policy = _as_mapping(payload.get("selection_policy"))
    _validate_selection_inputs(selection_policy, prefix="selection_policy")
    for key in LOCKED_OOS_FLAGS:
        if key in selection_policy and selection_policy.get(key) is not False:
            _fail(f"selection_policy.{key} must be false")


def _validate_cost_evidence(
    payload: Mapping[str, Any],
    execution_cost_evidence: Mapping[str, Any],
) -> None:
    payload_evidence = _as_mapping(payload.get("execution_cost_evidence"))
    if not payload_evidence:
        _fail("payload.execution_cost_evidence is required")
    for label, evidence in (
        ("payload.execution_cost_evidence", payload_evidence),
        ("execution_cost_evidence_latest.json", execution_cost_evidence),
    ):
        if evidence.get("diagnostic_only") is not True:
            _fail(f"{label}.diagnostic_only must be true")
        symbols = {str(item) for item in list(evidence.get("symbols") or [])}
        missing = sorted(REQUIRED_COST_SYMBOLS - symbols)
        if missing:
            _fail(f"{label}.symbols missing {missing}")


def _param_key_paths(value: Any, *, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(key_path)
            paths.extend(_param_key_paths(item, prefix=key_path))
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_param_key_paths(item, prefix=f"{prefix}[{index}]"))
        return paths
    return []


def _json_param_keys(row: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for field in ("variant_params_json", "strategy_params_json", "params_json", "params"):
        raw = row.get(field)
        if raw in {None, ""}:
            continue
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        paths.extend(_param_key_paths(parsed))
    return paths


def _validate_variant_inventory(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        _fail("candidate_variant_inventory_latest.csv must contain at least one row")
    for index, row in enumerate(rows, start=1):
        if "calendar_primary" not in row:
            _fail(f"variant row {index} missing calendar_primary=false marker")
        if _is_true(row.get("calendar_primary")):
            _fail(f"variant row {index} is calendar_primary=true")
        if _is_true(row.get("uses_locked_oos_for_selection")):
            _fail(f"variant row {index} uses locked_oos for selection")
        role = str(row.get("locked_oos_role") or "")
        if role and ("gate" not in role or "report" not in role):
            _fail(f"variant row {index} locked_oos_role must be gate/report-only")
        bad_keys = [
            path
            for path in _json_param_keys(row)
            if any(token in path.lower() for token in CALENDAR_KEY_TOKENS)
        ]
        if bad_keys:
            _fail(f"variant row {index} contains calendar/date parameter keys: {bad_keys}")


def _rows_at_primary_cost(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row for row in rows if _is_close(row.get("round_trip_slippage_fee_bps"), PRIMARY_COST_BPS)
    ]


def _group_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in _rows_at_primary_cost(rows):
        model_id = str(row.get("model_id") or "")
        if not model_id:
            _fail("10bps metric row missing model_id")
        grouped.setdefault(model_id, []).append(row)
    return grouped


def _promotion_requested(row: Mapping[str, Any]) -> bool:
    return any(
        _is_true(row.get(key))
        for key in ("live_promotable_10bps", "live_promotable", "promotion_gate_pass")
    )


def _metric(row: Mapping[str, Any], key: str) -> float:
    return _safe_float(row.get(key), float("nan"))


def _promotion_gate_failures_for_model(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    by_split = {str(row.get("split")): row for row in rows}
    failures: list[str] = []
    if set(by_split) != set(SPLIT_ORDER):
        return ["missing_required_splits"]
    for row in rows:
        if not _is_close(row.get("round_trip_slippage_fee_bps"), PRIMARY_COST_BPS):
            failures.append("promotion_cost_not_10bps")
        if _is_true(row.get("candidate_universe_uses_locked_oos_bucket")):
            failures.append("candidate_universe_uses_locked_oos_bucket")
        if str(row.get("promotability_scope") or "").strip().lower() == "shadow_only":
            failures.append("promotability_scope_shadow_only")
    train = by_split["train"]
    validation = by_split["validation"]
    locked_oos = by_split["locked_oos"]
    for split_name, row in (("validation", validation), ("locked_oos", locked_oos)):
        for metric in POSITIVE_SPLIT_METRICS:
            if _metric(row, metric) <= 0.0:
                failures.append(f"{split_name}_{metric}_non_positive")
    for metric in PROMOTION_METRICS:
        train_value = _metric(train, metric)
        if train_value <= _metric(validation, metric):
            failures.append(f"train_{metric}_not_gt_validation")
        if train_value <= _metric(locked_oos, metric):
            failures.append(f"train_{metric}_not_gt_locked_oos")
    if _metric(locked_oos, "max_drawdown") > MAX_LOCKED_OOS_MDD:
        failures.append("locked_oos_mdd_gt_25pct")
    for split_name, row in by_split.items():
        if int(_safe_float(row.get("account_wipeout_count"), 0.0)) != 0:
            failures.append(f"{split_name}_account_wipeout_count_positive")
        if int(_safe_float(row.get("liquidation_count"), 0.0)) != 0:
            failures.append(f"{split_name}_liquidation_count_positive")
        if "minimum_margin_buffer" not in row or _metric(row, "minimum_margin_buffer") <= 0.0:
            failures.append(f"{split_name}_minimum_margin_buffer_non_positive")
    return sorted(set(failures))


def _validate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    if not rows:
        _fail("candidate_model_metrics_latest.csv must contain rows")
    for row in rows:
        if _promotion_requested(row) and not _is_close(
            row.get("round_trip_slippage_fee_bps"), PRIMARY_COST_BPS
        ):
            _fail(f"promotion row for {row.get('model_id')} must use exactly 10bps")
    grouped = _group_metric_rows(rows)
    if not grouped:
        _fail("candidate_model_metrics_latest.csv must include 10bps rows")
    for model_id, model_rows in grouped.items():
        splits = {str(row.get("split")) for row in model_rows}
        if splits != set(SPLIT_ORDER):
            _fail(
                f"model {model_id} 10bps splits must be exactly {SPLIT_ORDER}, got {sorted(splits)}"
            )
        for row in model_rows:
            for metric in (
                "total_return",
                "max_drawdown",
                "sharpe",
                "sortino",
                "smart_sortino",
                "calmar",
            ):
                if not math.isfinite(_metric(row, metric)):
                    _fail(f"model {model_id} split {row.get('split')} missing finite {metric}")
    return grouped


def _validate_promotions(
    payload: Mapping[str, Any], grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    promotable_ids = {
        str(row.get("model_id"))
        for rows in grouped_rows.values()
        for row in rows
        if _promotion_requested(row)
    }
    payload_winner = payload.get("live_promotable_10bps_model_id")
    if payload_winner not in {None, "", False}:
        promotable_ids.add(str(payload_winner))
    for model_id in sorted(promotable_ids):
        rows = grouped_rows.get(model_id)
        if rows is None:
            _fail(f"live_promotable_10bps_model_id missing from 10bps metric rows: {model_id}")
        failures = _promotion_gate_failures_for_model(rows)
        if failures:
            _fail(f"model {model_id} cannot be live_promotable_10bps: {failures}")
    return sorted(promotable_ids)


def validate_artifact(root: str | Path = TARGET_ARTIFACT_DIR) -> dict[str, Any]:
    root = Path(root)
    _require_files(root)
    payload = _load_json(root / "alpha_zoo_10bps_full_retune_latest.json")
    cost_evidence = _load_json(root / "execution_cost_evidence_latest.json")
    metric_rows = _load_csv(root / "candidate_model_metrics_latest.csv")
    variant_rows = _load_csv(root / "candidate_variant_inventory_latest.csv")
    tuned_seed_json = _load_json(root / "tuned_seed_selection_latest.json")
    tuned_seed_rows = _load_csv(root / "tuned_seed_selection_latest.csv")

    if not tuned_seed_json:
        _fail("tuned_seed_selection_latest.json must not be empty")
    if not tuned_seed_rows:
        _fail("tuned_seed_selection_latest.csv must not be empty")
    _validate_real_money_and_cost(payload)
    _validate_split_manifest(payload)
    _validate_memory_summary(payload)
    _validate_locked_oos_audit(payload)
    _validate_cost_evidence(payload, cost_evidence)
    _validate_variant_inventory(variant_rows)
    grouped_rows = _validate_metric_rows(metric_rows)
    promotable_ids = _validate_promotions(payload, grouped_rows)
    return {
        "artifact_dir": str(root),
        "models": len(grouped_rows),
        "metric_rows": len(metric_rows),
        "promotable": promotable_ids,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", nargs="?", default=str(TARGET_ARTIFACT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = validate_artifact(Path(args.artifact_dir))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
