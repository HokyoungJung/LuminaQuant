#!/usr/bin/env python3
"""Calibrate Crypto/FX Alpha Zoo candidate edges from train/validation outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lumina_quant.research.candidate_outcome_ledger import CandidateOutcomeLedger
from lumina_quant.research.edge_calibration import calibrate_edge_buckets

TRAIN_VALIDATION_SPLITS = {"train", "validation"}
LOCKED_OOS_SPLITS = {"locked_oos", "oos"}


def _is_train_validation(record: dict[str, Any]) -> bool:
    return str(record.get("split")) in TRAIN_VALIDATION_SPLITS


def _edge_map_from_calibrations(calibrations: dict[str, Any]) -> dict[str, float]:
    """Build strategy edge keys from allowed/downsize calibration buckets."""
    values: dict[str, list[float]] = {}
    for payload in calibrations.values():
        decision = dict(payload.get("decision") or {})
        action = str(decision.get("action") or "")
        if action not in {"allow", "downsize"}:
            continue
        lower = float(decision.get("lower_confidence_edge_bps") or 0.0)
        if lower <= 0.0:
            continue
        bucket = [str(item) for item in list(payload.get("bucket") or [])]
        if len(bucket) >= 3:
            symbol = bucket[2]
            side = bucket[1]
            values.setdefault(f"{symbol}:{side}", []).append(lower)
        if len(bucket) >= 2:
            side = bucket[1]
            values.setdefault(f"default:{side}", []).append(lower)
    return {key: max(items) for key, items in sorted(values.items()) if items}


def build_calibration_payload(
    records: list[dict[str, Any]],
    *,
    ledger_summary: dict[str, Any],
    bucket_fields: tuple[str, ...],
    parent_fields: tuple[str, ...],
    min_bucket_n: int,
    confidence_z: float = 1.64,
    min_lower_edge_bps: float = 0.0,
    max_tail_loss_bps: float = 250.0,
) -> dict[str, Any]:
    input_split_counts = Counter(str(row.get("split")) for row in records)
    calibration_records = [row for row in records if _is_train_validation(row)]
    calibration_split_counts = Counter(str(row.get("split")) for row in calibration_records)
    calibrated = calibrate_edge_buckets(
        calibration_records,
        bucket_fields=bucket_fields,
        parent_fields=parent_fields,
        min_bucket_n=max(1, int(min_bucket_n)),
        confidence_z=float(confidence_z),
        min_lower_edge_bps=float(min_lower_edge_bps),
        max_tail_loss_bps=float(max_tail_loss_bps),
    )
    calibration_dict = {"|".join(key): value.to_dict() for key, value in calibrated.items()}
    input_locked_oos = sum(input_split_counts.get(split, 0) for split in LOCKED_OOS_SPLITS)
    calibration_locked_oos = sum(calibration_split_counts.get(split, 0) for split in LOCKED_OOS_SPLITS)
    return {
        "artifact_kind": "crypto_fx_alpha_zoo_edge_calibration",
        "selection_policy": "train_validation_only_locked_oos_report_only",
        "calibration_policy": "physical_train_validation_record_filter_before_bucket_estimation",
        "calibration_splits": sorted(TRAIN_VALIDATION_SPLITS),
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_calibration": False,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
        "input_record_count": len(records),
        "input_split_counts": dict(input_split_counts),
        "input_locked_oos_record_count": input_locked_oos,
        "calibration_record_count": len(calibration_records),
        "train_validation_calibration_record_count": len(calibration_records),
        "calibration_split_counts": dict(calibration_split_counts),
        "locked_oos_calibration_record_count": calibration_locked_oos,
        "excluded_locked_oos_record_count": input_locked_oos,
        "ledger_summary": ledger_summary,
        "bucket_fields": list(bucket_fields),
        "parent_fields": list(parent_fields),
        "min_bucket_n": max(1, int(min_bucket_n)),
        "calibrations": calibration_dict,
        "calibrated_edges_for_strategy": _edge_map_from_calibrations(calibration_dict),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="JSONL ledger from candidate_outcome_ledger")
    parser.add_argument("--output", default="var/reports/crypto_fx_alpha_zoo_v0/edge_calibration_latest.json")
    parser.add_argument("--bucket-fields", default="candidate_id,side,symbol,regime_bucket,volatility_bucket,factor_bucket")
    parser.add_argument("--parent-fields", default="candidate_id,side")
    parser.add_argument("--min-bucket-n", type=int, default=30)
    parser.add_argument("--confidence-z", type=float, default=1.64)
    parser.add_argument("--min-lower-edge-bps", type=float, default=0.0)
    parser.add_argument("--max-tail-loss-bps", type=float, default=250.0)
    args = parser.parse_args()

    ledger = CandidateOutcomeLedger(args.ledger)
    records = ledger.read_all()
    bucket_fields = tuple(item.strip() for item in args.bucket_fields.split(",") if item.strip())
    parent_fields = tuple(item.strip() for item in args.parent_fields.split(",") if item.strip())
    payload = build_calibration_payload(
        records,
        ledger_summary=ledger.summary(),
        bucket_fields=bucket_fields,
        parent_fields=parent_fields,
        min_bucket_n=max(1, int(args.min_bucket_n)),
        confidence_z=float(args.confidence_z),
        min_lower_edge_bps=float(args.min_lower_edge_bps),
        max_tail_loss_bps=float(args.max_tail_loss_bps),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
