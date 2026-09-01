"""Research/live-parity run cards with fail-closed reality gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_REALITY_GATES: tuple[str, ...] = (
    "gate_backtest_live_parity",
    "gate_cost_funding_realism",
    "gate_no_oos_leakage",
    "gate_data_integrity_fail_closed",
    "gate_performance_budget",
    "gate_no_real_money_run",
)
SAFE_NON_REAL_EXECUTION_MODES: frozenset[str] = frozenset(
    {
        "advisory",
        "backtest",
        "dry_run",
        "dry-run",
        "paper",
        "research",
        "shadow",
        "simulation",
        "sim",
        "testnet",
    }
)


@dataclass(frozen=True, slots=True)
class RunCard:
    """Immutable JSON-safe provenance record for a research or paper/live-shadow run."""

    run_id: str
    generated_at: str
    execution_mode: str
    strategy_name: str
    config_hash: str
    candidate_hash: str
    data_hash: str
    artifact_hashes: Mapping[str, str]
    source_refs: tuple[str, ...]
    git_head: str | None
    cost_model: Mapping[str, Any]
    funding_model: Mapping[str, Any]
    data_integrity: Mapping[str, Any]
    selection_policy: Mapping[str, Any]
    parity_checks: Mapping[str, Any]
    performance_budget: Mapping[str, Any]
    reality_gates: Mapping[str, bool]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_kind"] = "lumina_research_run_card"
        payload["run_card_hash"] = stable_payload_hash(
            {key: value for key, value in payload.items() if key != "run_card_hash"}
        )
        return payload


class RunCardRealityGateError(ValueError):
    """Raised when a run card violates a hard reality/parity gate."""

    def __init__(self, failed_gates: Sequence[str]) -> None:
        self.failed_gates = tuple(failed_gates)
        super().__init__("run_card_reality_gates_failed:" + ",".join(self.failed_gates))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError("nonfinite_run_card_value")
        return out
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return str(value)


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    """Serialize a payload deterministically and reject NaN/Infinity."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    )


def stable_payload_hash(payload: Mapping[str, Any] | Sequence[Any] | str | None) -> str:
    """Return a SHA-256 hash for JSON-like provenance payloads."""
    if isinstance(payload, str):
        encoded = payload.encode("utf-8")
    elif payload is None:
        encoded = b"null"
    else:
        encoded = stable_json_dumps(
            payload if isinstance(payload, Mapping) else {"items": payload}
        ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, str | int]:
    """Return a resolved path plus byte/hash identity for a runtime input."""
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


@contextmanager
def atomic_output_path(path: str | Path) -> Iterator[Path]:
    """Yield a same-directory staging path and replace the target only on success."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Atomically replace a text artifact."""
    target = Path(path)
    with atomic_output_path(target) as temporary:
        temporary.write_text(text, encoding="utf-8")
    return target


def runtime_provenance(
    *,
    repo_root: str | Path | None = None,
    packages: Sequence[str] = (),
    source_files: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Capture the small runtime identity needed to reproduce a research artifact."""
    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()

    def git(*args: str) -> str | None:
        try:
            process = subprocess.run(
                ["git", *args],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        return process.stdout.strip() if process.returncode == 0 else None

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    lock = root / "uv.lock"
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "executable": file_identity(sys.executable),
        },
        "packages": versions,
        "uv_lock": file_identity(lock) if lock.is_file() else None,
        "git": {
            "head": git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "status_sha256": None
            if status is None
            else hashlib.sha256(status.encode()).hexdigest(),
        },
        "source_files": {str(Path(path).resolve()): file_identity(path) for path in source_files},
    }


def _git_head() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def _artifact_hashes(artifacts: Mapping[str, Any] | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, value in dict(artifacts or {}).items():
        key = str(name)
        if isinstance(value, (str, Path)) and Path(value).exists():
            hashes[key] = file_sha256(value)
        elif isinstance(value, Mapping):
            hashes[key] = stable_payload_hash(value)
        else:
            hashes[key] = stable_payload_hash({"value": value})
    return hashes


def _explicit_cost_model(cost_model: Mapping[str, Any]) -> bool:
    if not cost_model:
        return False
    numeric_keys = ("round_trip_bps", "fee_bps", "slippage_bps", "spread_bps")
    return any(
        key in cost_model and _finite_number(cost_model[key]) is not None for key in numeric_keys
    )


def _explicit_funding_model(funding_model: Mapping[str, Any]) -> bool:
    if not funding_model:
        return False
    if funding_model.get("not_applicable") is True and funding_model.get("reason"):
        return True
    numeric_keys = ("funding_rate_bps", "funding_cost_bps", "borrow_bps", "carry_bps")
    return any(
        key in funding_model and _finite_number(funding_model[key]) is not None
        for key in numeric_keys
    )


def _finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except TypeError, ValueError:
        return None
    return out if math.isfinite(out) else None


def _explicit_false(mapping: Mapping[str, Any], key: str) -> bool:
    return key in mapping and mapping.get(key) is False


def build_reality_gates(
    *,
    execution_mode: str,
    cost_model: Mapping[str, Any],
    funding_model: Mapping[str, Any],
    data_integrity: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
    parity_checks: Mapping[str, Any],
    performance_budget: Mapping[str, Any],
) -> dict[str, bool]:
    """Evaluate hard gates selected for this milestone."""
    mode = str(execution_mode or "").strip().lower()
    perf_regression = _finite_number(performance_budget.get("max_regression_ratio"))
    perf_observed = _finite_number(performance_budget.get("observed_regression_ratio"))
    return {
        "gate_backtest_live_parity": bool(parity_checks.get("passed") is True),
        "gate_cost_funding_realism": _explicit_cost_model(cost_model)
        and _explicit_funding_model(funding_model),
        "gate_no_oos_leakage": _explicit_false(selection_policy, "uses_locked_oos_for_selection"),
        "gate_data_integrity_fail_closed": bool(data_integrity.get("passed") is True),
        "gate_performance_budget": perf_observed is not None
        and perf_regression is not None
        and perf_observed <= perf_regression,
        "gate_no_real_money_run": mode in SAFE_NON_REAL_EXECUTION_MODES,
    }


def assert_reality_gates_pass(payload: Mapping[str, Any]) -> None:
    """Raise when any hard gate is false."""
    gates = dict(payload.get("reality_gates") or {})
    failed = [
        key for key in REQUIRED_REALITY_GATES if key not in gates or gates.get(key) is not True
    ]
    failed.extend(
        key
        for key, value in sorted(gates.items())
        if key not in REQUIRED_REALITY_GATES and value is not True
    )
    if failed:
        raise RunCardRealityGateError(failed)


def build_research_run_card(
    *,
    run_id: str,
    execution_mode: str,
    strategy_name: str,
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    source_refs: Sequence[str],
    cost_model: Mapping[str, Any],
    funding_model: Mapping[str, Any],
    data_integrity: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
    parity_checks: Mapping[str, Any],
    performance_budget: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
    generated_at: str | None = None,
) -> RunCard:
    """Build a provenance run card and evaluate the selected hard gates."""
    refs = tuple(str(ref) for ref in source_refs if str(ref).strip())
    if not refs:
        raise ValueError("run_card_source_refs_missing")
    gates = build_reality_gates(
        execution_mode=execution_mode,
        cost_model=cost_model,
        funding_model=funding_model,
        data_integrity=data_integrity,
        selection_policy=selection_policy,
        parity_checks=parity_checks,
        performance_budget=performance_budget,
    )
    return RunCard(
        run_id=str(run_id),
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        execution_mode=str(execution_mode),
        strategy_name=str(strategy_name),
        config_hash=stable_payload_hash(config),
        candidate_hash=stable_payload_hash(candidate),
        data_hash=stable_payload_hash(data_manifest),
        artifact_hashes=_artifact_hashes(artifacts),
        source_refs=refs,
        git_head=_git_head(),
        cost_model=dict(cost_model),
        funding_model=dict(funding_model),
        data_integrity=dict(data_integrity),
        selection_policy=dict(selection_policy),
        parity_checks=dict(parity_checks),
        performance_budget=dict(performance_budget),
        reality_gates=gates,
        notes=tuple(str(note) for note in notes),
    )


def write_run_card(path: str | Path, run_card: RunCard | Mapping[str, Any]) -> Path:
    """Write a run card as strict deterministic JSON."""
    payload = run_card.to_dict() if isinstance(run_card, RunCard) else dict(run_card)
    assert_reality_gates_pass(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    return target


__all__ = [
    "REQUIRED_REALITY_GATES",
    "SAFE_NON_REAL_EXECUTION_MODES",
    "RunCard",
    "RunCardRealityGateError",
    "assert_reality_gates_pass",
    "build_reality_gates",
    "build_research_run_card",
    "file_sha256",
    "stable_json_dumps",
    "stable_payload_hash",
    "write_run_card",
]
