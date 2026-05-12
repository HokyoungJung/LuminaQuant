#!/usr/bin/env python3
"""Calibrate Crypto/FX Alpha Zoo candidate edges from an outcome ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumina_quant.research.candidate_outcome_ledger import CandidateOutcomeLedger
from lumina_quant.research.edge_calibration import calibrate_edge_buckets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="JSONL ledger from candidate_outcome_ledger")
    parser.add_argument("--output", default="var/reports/crypto_fx_alpha_zoo_v0/edge_calibration_latest.json")
    parser.add_argument("--bucket-fields", default="candidate_id,side,symbol")
    parser.add_argument("--parent-fields", default="candidate_id,side")
    parser.add_argument("--min-bucket-n", type=int, default=30)
    args = parser.parse_args()

    ledger = CandidateOutcomeLedger(args.ledger)
    records = ledger.read_all()
    bucket_fields = tuple(item.strip() for item in args.bucket_fields.split(",") if item.strip())
    parent_fields = tuple(item.strip() for item in args.parent_fields.split(",") if item.strip())
    calibrated = calibrate_edge_buckets(
        records,
        bucket_fields=bucket_fields,
        parent_fields=parent_fields,
        min_bucket_n=max(1, int(args.min_bucket_n)),
    )
    payload = {
        "artifact_kind": "crypto_fx_alpha_zoo_edge_calibration",
        "selection_policy": "train_validation_only_locked_oos_report_only",
        "uses_locked_oos_for_selection": False,
        "ledger_summary": ledger.summary(),
        "calibrations": {"|".join(key): value.to_dict() for key, value in calibrated.items()},
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
