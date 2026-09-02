#!/usr/bin/env python3
"""Run Alpha-Max data verification, backtests, and observability as one pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "alpha_max_backtest_pipeline.v1"
STAGES = (
    "canonical_data_verification",
    "prelock_validation",
    "historical_report_only",
    "validation_observability",
    "historical_observability",
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _absolute(value: object, *, field: str, must_exist: bool = True) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{field} must be an absolute nonsymlink path")
    if must_exist and not path.exists():
        raise ValueError(f"{field} does not exist: {path}")
    return path


def load_plan(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    plan = json.loads(raw)
    if type(plan) is not dict or plan.get("schema_version") != SCHEMA:
        raise ValueError("alpha_max_backtest_pipeline_plan_invalid")
    if plan.get("exchange") != "binance" or plan.get("order_routing_enabled") is not False:
        raise ValueError("alpha_max_backtest_pipeline_safety_invalid")
    phase_roots = plan.get("phase_roots")
    required = {
        "warmup_raw",
        "warmup_feature",
        "train_raw",
        "train_feature",
        "purge_raw",
        "purge_feature",
        "validation_raw",
        "validation_feature",
        "embargo_raw",
        "embargo_feature",
        "historical_evaluation_raw",
        "historical_evaluation_feature",
    }
    if type(phase_roots) is not dict or set(phase_roots) != required:
        raise ValueError("alpha_max_backtest_pipeline_phase_roots_invalid")
    for field in ("contract_manifest", "canonical_db", "config", "prior_trial_blob"):
        _absolute(plan.get(field), field=field)
    for field, value in phase_roots.items():
        _absolute(value, field=f"phase_roots.{field}")
    _absolute(plan.get("run_root"), field="run_root", must_exist=False)
    return plan


def build_commands(
    plan: Mapping[str, Any], *, python: Path, repository: Path
) -> list[tuple[str, list[str]]]:
    phase = plan["phase_roots"]
    run_root = Path(str(plan["run_root"]))
    prelock = run_root / "prelock"
    historical = run_root / "historical"
    return [
        (
            "canonical_data_verification",
            [
                str(python),
                str(repository / "scripts/research/verify_alpha_max_canonical_pipeline.py"),
                "--contract",
                str(plan["contract_manifest"]),
                "--db",
                str(plan["canonical_db"]),
                "--output",
                str(run_root / "canonical_pipeline_verification.json"),
            ],
        ),
        (
            "prelock_validation",
            [
                str(python),
                str(repository / "scripts/research/run_alpha_max_prelock.py"),
                "--config",
                str(plan["config"]),
                "--contract-manifest",
                str(plan["contract_manifest"]),
                "--prior-trial-blob",
                str(plan["prior_trial_blob"]),
                "--exchange",
                "binance",
                "--output-root",
                str(prelock),
                "--checkpoint-root",
                str(run_root / "checkpoints/prelock"),
                "--warmup-raw-root",
                str(phase["warmup_raw"]),
                "--warmup-feature-root",
                str(phase["warmup_feature"]),
                "--train-raw-root",
                str(phase["train_raw"]),
                "--train-feature-root",
                str(phase["train_feature"]),
                "--purge-raw-root",
                str(phase["purge_raw"]),
                "--purge-feature-root",
                str(phase["purge_feature"]),
                "--validation-raw-root",
                str(phase["validation_raw"]),
                "--validation-feature-root",
                str(phase["validation_feature"]),
                "--embargo-raw-root",
                str(phase["embargo_raw"]),
                "--embargo-feature-root",
                str(phase["embargo_feature"]),
            ],
        ),
        (
            "historical_report_only",
            [
                str(python),
                str(repository / "scripts/research/run_alpha_max_historical_evaluation.py"),
                "--sealed-prelock-directory",
                str(prelock),
                "--embargo-feature-root",
                str(phase["embargo_feature"]),
                "--historical-evaluation-raw-root",
                str(phase["historical_evaluation_raw"]),
                "--historical-evaluation-feature-root",
                str(phase["historical_evaluation_feature"]),
                "--exchange",
                "binance",
                "--output-root",
                str(historical),
                "--checkpoint-root",
                str(run_root / "checkpoints/historical"),
            ],
        ),
        (
            "validation_observability",
            [
                str(python),
                str(repository / "scripts/research/export_alpha_max_observability.py"),
                "--bundle-root",
                str(prelock),
                "--domain",
                "validation",
                "--manifest-root",
                str(prelock),
                "--output",
                str(run_root / "observability/validation.json"),
            ],
        ),
        (
            "historical_observability",
            [
                str(python),
                str(repository / "scripts/research/export_alpha_max_observability.py"),
                "--bundle-root",
                str(historical),
                "--domain",
                "historical_exposed_evaluation",
                "--manifest-root",
                str(prelock),
                "--output",
                str(run_root / "observability/historical.json"),
            ],
        ),
    ]


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = _canonical_bytes(payload)
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_pipeline(plan_path: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    plan = load_plan(plan_path)
    run_root = _absolute(plan["run_root"], field="run_root", must_exist=False)
    if run_root.exists():
        raise ValueError("alpha_max_backtest_pipeline_run_root_exists")
    run_root.mkdir(parents=True, mode=0o700)
    (run_root / "logs").mkdir(mode=0o700)
    (run_root / "observability").mkdir(mode=0o700)
    commands = build_commands(
        plan,
        python=Path(sys.executable).absolute(),
        repository=repository,
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("LQ_")}
    environment["PYTHONPATH"] = str(repository / "src")
    receipt: dict[str, Any] = {
        "artifact_kind": "alpha_max_backtest_pipeline_receipt.v1",
        "schema_version": SCHEMA,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path.read_bytes()),
        "repository": str(repository),
        "order_routing_enabled": False,
        "stages": [],
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    receipt_path = run_root / "pipeline_receipt.json"
    for stage, argv in commands:
        stdout_path = run_root / "logs" / f"{stage}.stdout.log"
        stderr_path = run_root / "logs" / f"{stage}.stderr.log"
        started = datetime.now(UTC)
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(
                argv,
                cwd=repository,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        stage_receipt = {
            "stage": stage,
            "argv": argv,
            "return_code": completed.returncode,
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "stdout_path": str(stdout_path),
            "stdout_sha256": _sha256(stdout_path.read_bytes()),
            "stderr_path": str(stderr_path),
            "stderr_sha256": _sha256(stderr_path.read_bytes()),
        }
        receipt["stages"].append(stage_receipt)
        if completed.returncode != 0:
            receipt["status"] = "failed"
            receipt["failed_stage"] = stage
            receipt["completed_at_utc"] = datetime.now(UTC).isoformat()
            _write_atomic(receipt_path, receipt)
            raise RuntimeError(f"alpha_max_backtest_pipeline_stage_failed:{stage}")
        _write_atomic(receipt_path, receipt)
    receipt["status"] = "complete"
    receipt["completed_at_utc"] = datetime.now(UTC).isoformat()
    _write_atomic(receipt_path, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = run_pipeline(args.plan.resolve())
    print(json.dumps({"status": receipt["status"], "stages": STAGES}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
