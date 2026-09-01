#!/usr/bin/env python3
"""Validate a LuminaQuant alpha-discovery run skeleton or decision bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_INIT_FILES = [
    "run_manifest.json",
    "candidate_registry.json",
    "experiment_design.json",
    "quality_gate_receipt.json",
    "run_plan.md",
]
REQUIRED_DECISION_FILES = [*REQUIRED_INIT_FILES, "scoreboard.json", "decision.json"]
FORBIDDEN_TRUE_KEYS = {"real_money_execution", "allow_real_money", "ready_for_real"}
REQUIRED_FORBIDDEN_DATA = {"locked_current_oos", "future_data"}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON {path}: {exc}") from exc


def walk(obj: Any, path: str = "$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k, v
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["init", "decision"], default="init")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required = REQUIRED_DECISION_FILES if args.mode == "decision" else REQUIRED_INIT_FILES

    if not run_dir.exists():
        errors.append(f"run_dir does not exist: {run_dir}")
    for name in required:
        if not (run_dir / name).exists():
            errors.append(f"missing required file: {name}")

    json_docs: dict[str, Any] = {}
    for path in sorted(run_dir.glob("*.json")):
        try:
            json_docs[path.name] = load_json(path)
        except ValueError as exc:
            errors.append(str(exc))

    for file_name, doc in json_docs.items():
        for p, k, v in walk(doc):
            if k in FORBIDDEN_TRUE_KEYS and v is not False:
                errors.append(f"{file_name}:{p} must be false, got {v!r}")

    registry = json_docs.get("candidate_registry.json")
    if registry:
        candidates = registry.get("candidates", [])
        if not candidates:
            errors.append("candidate_registry.json has no candidates")
        for idx, cand in enumerate(candidates):
            cid = cand.get("candidate_id", f"index:{idx}")
            forbidden = set(cand.get("forbidden_data", []))
            if not REQUIRED_FORBIDDEN_DATA.issubset(forbidden):
                errors.append(
                    f"candidate {cid} missing forbidden_data entries {sorted(REQUIRED_FORBIDDEN_DATA - forbidden)}"
                )
            if not cand.get("hypothesis") or not cand.get("mechanism"):
                errors.append(f"candidate {cid} missing hypothesis or mechanism")
            if cand.get("decision") not in {
                "pending",
                "reject",
                "shadow-watch",
                "promotion",
                "no-promotion",
            }:
                warnings.append(
                    f"candidate {cid} has nonstandard decision {cand.get('decision')!r}"
                )

    design = json_docs.get("experiment_design.json")
    if design:
        locked = design.get("windows", {}).get("locked_current_oos", {})
        forbidden_for = set(locked.get("forbidden_for", []))
        required_forbidden_for = {
            "threshold_selection",
            "sleeve_selection",
            "weight_selection",
            "tie_breaks",
        }
        if not required_forbidden_for.issubset(forbidden_for):
            errors.append(
                "experiment_design locked_current_oos does not forbid all selection operations"
            )
        if design.get("cost_stress_bps") != [10, 15, 20]:
            warnings.append("experiment_design cost_stress_bps should be [10, 15, 20]")

    result = {
        "run_dir": str(run_dir),
        "mode": args.mode,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
