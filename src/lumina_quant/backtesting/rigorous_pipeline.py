"""Restartable orchestration for strategy-agnostic rigorous backtests.

The pipeline never copies market data. Every stage receives the same canonical
read-only data root and publishes only result files beneath one run root.
Specialized research programs remain adapters; this module owns ordering,
resource bounds, lineage, and safe resume semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "lumina_quant.rigorous_backtest_pipeline.v1"
STAGE_ORDER = (
    "coarse_screen",
    "event_driven_walk_forward",
    "validation_selection",
    "execution_model_tick_validation",
    "report_only_evaluation",
)
_ALLOWED_ENVIRONMENT = frozenset({"LQ_CONFIG_PATH", "LQ_RAW_FIRST_BACKEND"})


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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_receipt(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return {"path": str(path), "bytes": byte_count, "sha256": digest.hexdigest()}


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _resolved_nonsymlink(
    value: object,
    *,
    field: str,
    base: Path,
    must_exist: bool = True,
    directory: bool | None = None,
) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = base / path
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        raise ValueError(f"{field} must be a nonsymlink path")
    if must_exist and not path.exists():
        raise ValueError(f"{field} does not exist: {path}")
    if must_exist and directory is True and not path.is_dir():
        raise ValueError(f"{field} must be a directory")
    if must_exist and directory is False and not path.is_file():
        raise ValueError(f"{field} must be a regular file")
    return path


def _expand(token: str, *, data_root: Path, run_root: Path, repository: Path) -> str:
    return (
        token.replace("${DATA_ROOT}", str(data_root))
        .replace("${RUN_ROOT}", str(run_root))
        .replace("${REPOSITORY}", str(repository))
    )


def _validate_stage(
    value: object,
    *,
    expected_name: str,
    repository: Path,
    data_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "name",
        "script",
        "arguments",
        "inputs",
        "outputs",
        "environment",
        "accepted_return_codes",
    }:
        raise ValueError(f"rigorous_pipeline_stage_invalid:{expected_name}")
    if value["name"] != expected_name:
        raise ValueError("rigorous_pipeline_stage_order_invalid")
    script = Path(
        _expand(
            str(value["script"]),
            data_root=data_root,
            run_root=run_root,
            repository=repository,
        )
    )
    script = _resolved_nonsymlink(
        script,
        field=f"{expected_name}.script",
        base=repository,
        directory=False,
    )
    scripts_root = (repository / "scripts").resolve()
    if not _inside(script.resolve(), scripts_root):
        raise ValueError(f"{expected_name}.script must be beneath repository scripts/")
    arguments = value["arguments"]
    inputs = value["inputs"]
    outputs = value["outputs"]
    environment = value["environment"]
    accepted_return_codes = value["accepted_return_codes"]
    if type(arguments) is not list or not all(type(item) is str for item in arguments):
        raise ValueError(f"{expected_name}.arguments must be a string list")
    if type(inputs) is not list or not all(type(item) is str for item in inputs):
        raise ValueError(f"{expected_name}.inputs must be a string list")
    if type(outputs) is not list or not outputs or not all(type(item) is str for item in outputs):
        raise ValueError(f"{expected_name}.outputs must be a non-empty string list")
    if type(environment) is not dict or not set(environment) <= _ALLOWED_ENVIRONMENT:
        raise ValueError(f"{expected_name}.environment contains unsupported keys")
    if not all(type(key) is str and type(item) is str for key, item in environment.items()):
        raise ValueError(f"{expected_name}.environment must contain strings")
    if (
        type(accepted_return_codes) is not list
        or not accepted_return_codes
        or not all(type(code) is int and 0 <= code <= 255 for code in accepted_return_codes)
        or len(set(accepted_return_codes)) != len(accepted_return_codes)
    ):
        raise ValueError(f"{expected_name}.accepted_return_codes is invalid")
    expanded_inputs = [
        _resolved_nonsymlink(
            _expand(item, data_root=data_root, run_root=run_root, repository=repository),
            field=f"{expected_name}.inputs",
            base=repository,
            must_exist=False,
            directory=False,
        )
        for item in inputs
    ]
    if any(not path.exists() and not _inside(path, run_root) for path in expanded_inputs):
        raise ValueError(f"{expected_name}.inputs must exist or be produced beneath run_root")
    expanded_outputs = [
        _resolved_nonsymlink(
            _expand(item, data_root=data_root, run_root=run_root, repository=repository),
            field=f"{expected_name}.outputs",
            base=repository,
            must_exist=False,
        )
        for item in outputs
    ]
    if any(not _inside(path, run_root) or _inside(path, data_root) for path in expanded_outputs):
        raise ValueError(f"{expected_name}.outputs must be nonsymlink paths beneath run_root")
    return {
        "name": expected_name,
        "script": script,
        "arguments": [
            _expand(item, data_root=data_root, run_root=run_root, repository=repository)
            for item in arguments
        ],
        "inputs": expanded_inputs,
        "outputs": expanded_outputs,
        "environment": dict(environment),
        "accepted_return_codes": tuple(accepted_return_codes),
    }


def load_plan(path: Path, *, repository: Path) -> dict[str, Any]:
    plan = json.loads(path.read_bytes())
    if type(plan) is not dict or set(plan) != {
        "schema_version",
        "data_root",
        "data_receipt",
        "run_root",
        "order_routing_enabled",
        "memory_max_bytes",
        "stages",
    }:
        raise ValueError("rigorous_backtest_pipeline_plan_invalid")
    if plan["schema_version"] != SCHEMA or plan["order_routing_enabled"] is not False:
        raise ValueError("rigorous_backtest_pipeline_safety_invalid")
    memory_max = plan["memory_max_bytes"]
    if type(memory_max) is not int or not 512 * 1024**2 <= memory_max <= 7 * 1024**3:
        raise ValueError("rigorous_backtest_pipeline_memory_limit_invalid")
    repository = repository.resolve()
    data_root = _resolved_nonsymlink(
        plan["data_root"], field="data_root", base=repository, directory=True
    )
    data_receipt = _resolved_nonsymlink(
        plan["data_receipt"],
        field="data_receipt",
        base=repository,
        directory=False,
    )
    run_root = _resolved_nonsymlink(
        plan["run_root"],
        field="run_root",
        base=repository,
        must_exist=False,
    )
    if not _inside(data_root.resolve(), repository):
        raise ValueError("data_root must be inside repository")
    if not _inside(data_receipt.resolve(), repository):
        raise ValueError("data_receipt must be inside repository")
    allowed_run_parent = (repository / "var" / "backtests").resolve()
    if not _inside(run_root, allowed_run_parent) or _inside(run_root, data_root):
        raise ValueError("run_root must be beneath repository var/backtests")
    stages = plan["stages"]
    if type(stages) is not list or len(stages) != len(STAGE_ORDER):
        raise ValueError("rigorous_backtest_pipeline_stages_invalid")
    validated = [
        _validate_stage(
            value,
            expected_name=name,
            repository=repository,
            data_root=data_root,
            run_root=run_root,
        )
        for name, value in zip(STAGE_ORDER, stages, strict=True)
    ]
    return {
        **plan,
        "data_root": data_root,
        "data_receipt": data_receipt,
        "run_root": run_root,
        "stages": validated,
    }


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = _canonical_bytes(value)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _stage_fingerprint(
    stage: Mapping[str, Any], *, data_receipt: Path, git_head: str
) -> tuple[str, list[dict[str, object]]]:
    inputs = [_file_receipt(path) for path in stage["inputs"]]
    fingerprint = _sha256(
        _canonical_bytes(
            {
                "name": stage["name"],
                "script": _file_receipt(stage["script"]),
                "arguments": stage["arguments"],
                "environment": stage["environment"],
                "accepted_return_codes": stage["accepted_return_codes"],
                "inputs": inputs,
                "data_receipt": _file_receipt(data_receipt),
                "git_head": git_head,
            }
        )
    )
    return fingerprint, inputs


def _completed_stage_matches(
    receipt_path: Path,
    *,
    fingerprint: str,
    outputs: Sequence[Path],
) -> bool:
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return False
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except OSError, ValueError, json.JSONDecodeError:
        return False
    if receipt.get("status") != "complete" or receipt.get("fingerprint") != fingerprint:
        return False
    expected = receipt.get("outputs")
    if type(expected) is not list or len(expected) != len(outputs):
        return False
    return all(
        path.is_file() and _file_receipt(path) == item for path, item in zip(outputs, expected)
    )


def _limit_address_space(memory_max_bytes: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (memory_max_bytes, memory_max_bytes))


def run_pipeline(
    plan_path: Path,
    *,
    repository: Path,
    resume: bool = False,
) -> dict[str, Any]:
    repository = repository.resolve()
    plan = load_plan(plan_path, repository=repository)
    run_root: Path = plan["run_root"]
    if run_root.exists() and not resume:
        raise ValueError("rigorous_backtest_pipeline_run_root_exists")
    run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (run_root / "logs").mkdir(exist_ok=True, mode=0o700)
    (run_root / "receipts").mkdir(exist_ok=True, mode=0o700)
    git_head = _git_head(repository)
    stage_rows: list[dict[str, object]] = []
    for stage in plan["stages"]:
        fingerprint, inputs = _stage_fingerprint(
            stage,
            data_receipt=plan["data_receipt"],
            git_head=git_head,
        )
        stage_receipt = run_root / "receipts" / f"{stage['name']}.json"
        if resume and _completed_stage_matches(
            stage_receipt,
            fingerprint=fingerprint,
            outputs=stage["outputs"],
        ):
            stage_rows.append(
                {"name": stage["name"], "status": "reused", "fingerprint": fingerprint}
            )
            continue
        for output in stage["outputs"]:
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() or output.is_symlink():
                raise ValueError(f"stale stage output blocks safe execution: {output}")
        stdout_path = run_root / "logs" / f"{stage['name']}.stdout.log"
        stderr_path = run_root / "logs" / f"{stage['name']}.stderr.log"
        if stdout_path.exists() or stderr_path.exists():
            raise ValueError(f"stale stage log blocks safe execution: {stage['name']}")
        argv = [sys.executable, str(stage["script"]), *stage["arguments"]]
        environment = {key: value for key, value in os.environ.items() if not key.startswith("LQ_")}
        environment.update(stage["environment"])
        environment["PYTHONPATH"] = str(repository / "src")
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
                preexec_fn=lambda: _limit_address_space(plan["memory_max_bytes"]),
            )
        outputs = [_file_receipt(path) for path in stage["outputs"] if path.is_file()]
        status = (
            "complete"
            if completed.returncode in stage["accepted_return_codes"]
            and len(outputs) == len(stage["outputs"])
            else "failed"
        )
        row = {
            "artifact_kind": "lumina_quant.rigorous_backtest_stage_receipt.v1",
            "name": stage["name"],
            "status": status,
            "fingerprint": fingerprint,
            "argv": argv,
            "inputs": inputs,
            "outputs": outputs,
            "stdout": _file_receipt(stdout_path),
            "stderr": _file_receipt(stderr_path),
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "return_code": completed.returncode,
            "order_routing_enabled": False,
        }
        _atomic_write(stage_receipt, row)
        stage_rows.append(row)
        if status != "complete":
            raise RuntimeError(f"rigorous_backtest_pipeline_stage_failed:{stage['name']}")
    receipt = {
        "artifact_kind": "lumina_quant.rigorous_backtest_pipeline_receipt.v1",
        "schema_version": SCHEMA,
        "status": "complete",
        "plan": _file_receipt(plan_path),
        "data_root": str(plan["data_root"]),
        "data_receipt": _file_receipt(plan["data_receipt"]),
        "run_root": str(run_root),
        "git_head": git_head,
        "memory_max_bytes": plan["memory_max_bytes"],
        "order_routing_enabled": False,
        "stages": stage_rows,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write(run_root / "pipeline_receipt.json", receipt)
    return receipt
