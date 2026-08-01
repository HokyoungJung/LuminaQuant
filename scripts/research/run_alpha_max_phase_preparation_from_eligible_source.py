#!/usr/bin/env python3
"""Run the frozen Alpha-Max phase preparer only from a verified eligible source."""

from __future__ import annotations
from contextlib import ExitStack
import argparse
import errno
import fcntl
import hashlib
import io
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
from typing import Any

import polars as pl

CONTRACT_SHA256 = "ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220"
ACQUIRER_SHA256 = "d3c674ecf28c5869eab43f9903b4479185b36faca108919868c2f2c31662db70"
EVIDENCE_SHA256 = "214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719"
PREPARER_SHA256 = "ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac"
MAX_CAPTURE_BYTES = 65_536
FICLONE = 0x40049409
SNAPSHOT_SCHEMA = "alpha_max_phase_preparation_source_snapshot.v1"
PHASE_INTERVALS = (
    ("warmup", "2022-12-31T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("train", "2024-01-01T00:00:00Z", "2025-06-01T00:00:00Z"),
    ("purge", "2025-06-01T00:00:00Z", "2025-06-08T00:00:00Z"),
    ("validation", "2025-06-08T00:00:00Z", "2025-08-31T00:00:00Z"),
    ("embargo", "2025-08-31T00:00:00Z", "2025-09-07T00:00:00Z"),
    ("historical_exposed_evaluation", "2025-09-07T00:00:00Z", "2026-07-01T00:00:00Z"),
)


class PreparationError(ValueError):
    """Raised when the eligibility-to-preparation boundary is unsafe."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def argv_sha256(argv: list[str]) -> str:
    return sha256(canonical_bytes(argv))


def _absolute_clean_path(value: Path, label: str) -> Path:
    if not value.is_absolute() or ".." in value.parts:
        raise PreparationError(f"{label}_must_be_clean_absolute")
    return Path(os.path.normpath(os.fspath(value)))


def _assert_no_symlink_components(path: Path, label: str, *, include_leaf: bool) -> None:
    parts = path.parts
    current = Path(parts[0])
    limit = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:limit]:
        current /= part
        try:
            item = current.lstat()
        except OSError as exc:
            raise PreparationError(f"{label}_component_missing") from exc
        if stat.S_ISLNK(item.st_mode):
            raise PreparationError(f"{label}_symlink_component")


def _regular_file(path: Path, label: str) -> os.stat_result:
    _assert_no_symlink_components(path, label, include_leaf=True)
    try:
        item = path.lstat()
    except OSError as exc:
        raise PreparationError(f"{label}_missing") from exc
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        raise PreparationError(f"{label}_unsafe_file")
    return item


def _open_verified_file(path: Path, label: str) -> tuple[int, os.stat_result]:
    if not path.is_absolute():
        raise PreparationError(f"{label}_must_be_clean_absolute")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor
        )
    except OSError as exc:
        os.close(descriptor)
        raise PreparationError(f"{label}_missing") from exc
    os.close(descriptor)
    item = os.fstat(file_descriptor)
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        os.close(file_descriptor)
        raise PreparationError(f"{label}_unsafe_file")
    return file_descriptor, item


def _stable_file_identity(item: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        item.st_dev,
        item.st_ino,
        stat.S_IFMT(item.st_mode),
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _assert_stable_file(descriptor: int, before: os.stat_result, label: str) -> None:
    after = os.fstat(descriptor)
    if _stable_file_identity(after) != _stable_file_identity(before):
        raise PreparationError(f"{label}_changed_during_read")


MAX_METADATA_BYTES = 16 << 20


def _verified_file_bytes(path: Path, label: str) -> bytes:
    descriptor, before = _open_verified_file(path, label)
    try:
        if before.st_size > MAX_METADATA_BYTES:
            raise PreparationError(f"{label}_too_large")
        chunks = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        _assert_stable_file(descriptor, before, label)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _directory_identity(path: Path, label: str) -> dict[str, int]:
    _assert_no_symlink_components(path, label, include_leaf=True)
    try:
        item = path.lstat()
    except OSError as exc:
        raise PreparationError(f"{label}_missing") from exc
    if not stat.S_ISDIR(item.st_mode):
        raise PreparationError(f"{label}_not_directory")
    return {"st_dev": item.st_dev, "st_ino": item.st_ino}


def _open_authenticated_root(path: Path, label: str) -> tuple[int, dict[str, int]]:
    """Open and identify an authentication root while retaining descriptor ownership."""
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        item = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise PreparationError(f"{label}_namespace_unavailable") from exc
    return descriptor, {"st_dev": item.st_dev, "st_ino": item.st_ino}


def _assert_directory_identity(path: Path, expected: dict[str, int], label: str) -> None:
    if _directory_identity(path, label) != expected:
        raise PreparationError(f"{label}_replaced")


def file_sha256(path: Path, label: str) -> str:
    descriptor, before = _open_verified_file(path, label)
    digest = hashlib.sha256()
    try:
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        _assert_stable_file(descriptor, before, label)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _open_pinned_file(path: Path, label: str) -> tuple[int, os.stat_result, str]:
    """Keep the verified object open through all child invocations."""
    descriptor, before = _open_verified_file(path, label)
    digest = hashlib.sha256()
    try:
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        _assert_stable_file(descriptor, before, label)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, before, digest.hexdigest()
    except Exception:
        os.close(descriptor)
        raise


def _pinned_fd_path(descriptor: int) -> str:
    path = f"/proc/self/fd/{descriptor}"
    if not os.path.exists(path):
        raise PreparationError("pinned_descriptor_path_unavailable")
    return path


def _assert_pinned_files_unchanged(pinned: dict[str, tuple[int, os.stat_result, str]]) -> None:
    for label, (descriptor, before, _) in pinned.items():
        _assert_stable_file(descriptor, before, label)


def _is_ancestor_or_same(first: Path, second: Path) -> bool:
    return first == second or first in second.parents


def _reject_overlap(paths: dict[str, Path], forbidden_roots: list[Path]) -> None:
    checked = list(paths.items())
    for index, (left_name, left) in enumerate(checked):
        for right_name, right in checked[index + 1 :]:
            if _is_ancestor_or_same(left, right) or _is_ancestor_or_same(right, left):
                raise PreparationError(f"path_overlap:{left_name}:{right_name}")
    for forbidden in forbidden_roots:
        for name, path in checked:
            if _is_ancestor_or_same(forbidden, path) or _is_ancestor_or_same(path, forbidden):
                raise PreparationError(f"forbidden_root_overlap:{name}")


def _capture(
    command: list[str], label: str, *, pass_fds: tuple[int, ...] = ()
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, capture_output=True, check=False, pass_fds=pass_fds)
    stdout = result.stdout if isinstance(result.stdout, bytes) else str(result.stdout).encode()
    stderr = result.stderr if isinstance(result.stderr, bytes) else str(result.stderr).encode()
    if len(stdout) > MAX_CAPTURE_BYTES or len(stderr) > MAX_CAPTURE_BYTES:
        raise PreparationError(f"{label}_output_too_large")
    return subprocess.CompletedProcess(command, result.returncode, stdout, stderr)


def _regular_bytes(path: Path, label: str) -> bytes:
    return _verified_file_bytes(path, label)


def _canonical_artifact_map(source_root: Path, source_report: Path) -> tuple[str, dict[str, str]]:
    manifest_path = source_report / "source_manifest.json"
    raw = _regular_bytes(manifest_path, "source_manifest")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError("source_manifest_invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "alpha_max_official_source_manifest.v5"
        or not isinstance(manifest.get("artifacts"), list)
    ):
        raise PreparationError("source_manifest_invalid")
    artifacts: dict[str, str] = {}
    for entry in manifest["artifacts"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise PreparationError("source_manifest_artifact_invalid")
        relative, expected = entry["path"], entry["sha256"]
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise PreparationError("source_manifest_artifact_invalid")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or len(path.parts) < 2
            or any(part in ("", ".", "..") for part in path.parts)
            or path.parts[0] not in {"output", "report"}
        ):
            raise PreparationError("source_manifest_artifact_path_invalid")
        if relative in artifacts:
            raise PreparationError("source_manifest_artifact_duplicate")
        artifact_root = source_root if path.parts[0] == "output" else source_report
        actual = file_sha256(artifact_root.joinpath(*path.parts[1:]), "source_manifest_artifact")
        if actual != expected:
            raise PreparationError("source_manifest_artifact_hash_mismatch")
        artifacts[relative] = expected
    return sha256(canonical_bytes(artifacts)), artifacts


def _eligibility_snapshot(source_root: Path, source_report: Path) -> dict[str, Any]:
    artifact_map_sha, _ = _canonical_artifact_map(source_root, source_report)
    return {
        "source_root_identity": _directory_identity(source_root, "source_root"),
        "source_report_identity": _directory_identity(source_report, "source_report"),
        "source_eligible_receipt_sha256": file_sha256(
            source_report / "source_eligible_receipt.json", "source_eligible_receipt"
        ),
        "source_manifest_sha256": file_sha256(
            source_report / "source_manifest.json", "source_manifest"
        ),
        "acquisition_journal_sha256": file_sha256(
            source_report / "acquisition.journal.jsonl", "acquisition_journal"
        ),
        "plan_sha256": file_sha256(source_report / "plan.json", "plan"),
        "source_owner_sha256": file_sha256(source_root / ".alpha_max_owner.json", "source_owner"),
        "report_owner_sha256": file_sha256(source_report / ".alpha_max_owner.json", "report_owner"),
        "source_manifest_artifact_map_sha256": artifact_map_sha,
    }


def _require_unchanged(
    before: dict[str, Any], source_root: Path, source_report: Path
) -> dict[str, Any]:
    after = _eligibility_snapshot(source_root, source_report)
    if after != before:
        raise PreparationError("eligible_source_changed")
    return after


SIDE_CAR_SUFFIX = ".alpha_max_phase_preparation"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquirer", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--forbidden-root", required=True, action="append", type=Path)
    parser.add_argument("--contract-manifest", required=True, type=Path)
    parser.add_argument("--availability-evidence", required=True, type=Path)
    parser.add_argument("--preparer", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def _sidecars(output_root: Path) -> dict[str, Path]:
    base = output_root.parent / f".{output_root.name}{SIDE_CAR_SUFFIX}"
    snapshot = Path(f"{base}.source-snapshot")
    return {
        "descriptor_stage": Path(f"{base}.invocation.stage.json"),
        "descriptor": Path(f"{base}.invocation.json"),
        "receipt_stage": Path(f"{base}.handoff.stage.json"),
        "receipt": Path(f"{base}.handoff.json"),
        "snapshot": snapshot,
        "inputs": Path(f"{base}.invocation-inputs"),
        "lock": Path(f"{base}.lock"),
        "snapshot_manifest_stage": snapshot / ".snapshot-manifest.stage.json",
        "snapshot_manifest": snapshot / "snapshot-manifest.json",
        "snapshot_complete_stage": snapshot / ".complete.stage.json",
        "snapshot_complete": snapshot / ".complete.json",
    }


def _open_invocation_lock(path: Path) -> tuple[int, dict[str, int]]:
    _private_sidecar_parent(path, "invocation_lock")
    try:
        descriptor = _open_no_follow(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            descriptor = _open_no_follow(path, os.O_RDWR)
        except OSError as exc:
            raise PreparationError("invocation_lock_unsafe_file") from exc
    except OSError as exc:
        raise PreparationError("invocation_lock_create_failed") from exc
    item = os.fstat(descriptor)
    if (
        not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != os.geteuid()
        or stat.S_IMODE(item.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise PreparationError("invocation_lock_unsafe_file")
    return descriptor, {"st_dev": item.st_dev, "st_ino": item.st_ino}


def _canonical_json_file(path: Path, label: str) -> Any:
    raw = _regular_bytes(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{label}_invalid_json") from exc
    if raw != canonical_bytes(value):
        raise PreparationError(f"{label}_not_canonical")
    return value


def _private_sidecar_parent(path: Path, label: str) -> os.stat_result:
    _assert_no_symlink_components(path, label, include_leaf=False)
    try:
        item = path.parent.lstat()
    except OSError as exc:
        raise PreparationError(f"{label}_parent_missing") from exc
    if (
        not stat.S_ISDIR(item.st_mode)
        or item.st_uid != os.getuid()
        or stat.S_IMODE(item.st_mode) not in {0o700, 0o555}
    ):
        raise PreparationError(f"{label}_parent_unsafe")
    return item


def _sidecar_exists(path: Path, label: str) -> bool:
    _private_sidecar_parent(path, label)
    _assert_no_symlink_components(path, label, include_leaf=False)
    try:
        item = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PreparationError(f"{label}_unreadable") from exc
    if stat.S_ISLNK(item.st_mode):
        raise PreparationError(f"{label}_unsafe_file")
    return True


def _fsync_directory(path: Path, label: str) -> None:
    _private_sidecar_parent(path / ".sidecar-parent-check", label)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise PreparationError(f"{label}_directory_sync_failed") from exc


def _fsync_any_directory(path: Path, label: str) -> None:
    _directory_identity(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise PreparationError(f"{label}_directory_sync_failed") from exc


def _write_or_resume(stage: Path, payload: bytes, label: str) -> None:
    _private_sidecar_parent(stage, label)
    if _sidecar_exists(stage, f"{label}_stage"):
        current = _regular_bytes(stage, f"{label}_stage")
        if len(current) > len(payload) or payload[: len(current)] != current:
            raise PreparationError(f"{label}_stage_diverged")
        flags = os.O_WRONLY | os.O_APPEND
        suffix = payload[len(current) :]
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        suffix = payload
    try:
        descriptor = os.open(stage, flags, 0o600)
        try:
            remaining = memoryview(suffix)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("sidecar_write_failed")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _regular_file(stage, f"{label}_stage")
        _fsync_directory(stage.parent, label)
    except OSError as exc:
        raise PreparationError(f"{label}_publish_failed") from exc


def _persist_immutable(stage: Path, final: Path, value: Any, label: str) -> None:
    payload = canonical_bytes(value)
    final_exists = _sidecar_exists(final, label)
    recovering = _sidecar_exists(stage, f"{label}_stage")
    if final_exists:
        if recovering:
            raise PreparationError(f"{label}_stage_conflict")
        if _regular_bytes(final, label) != payload:
            raise PreparationError(f"{label}_diverged")
        return
    _write_or_resume(stage, payload, label)
    try:
        os.replace(stage, final)
        _regular_file(final, label)
        _fsync_directory(final.parent, label)
    except OSError as exc:
        raise PreparationError(f"{label}_{'recovery' if recovering else 'publish'}_failed") from exc


def _pinned_bytes(descriptor: int, before: os.stat_result, label: str) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while block := os.read(descriptor, 1 << 20):
        chunks.append(block)
    _assert_stable_file(descriptor, before, label)
    return b"".join(chunks)


def _materialize_invocation_inputs(
    root: Path, pinned: dict[str, tuple[int, os.stat_result, str]]
) -> dict[str, Path]:
    """Atomically publish child inputs copied from the descriptors that were pinned."""
    _create_snapshot_root(root)
    files = {
        "acquirer": root / "acquirer.py",
        "contract_manifest": root / "contract_manifest.json",
        "availability_evidence": root / "availability_evidence.json",
        "preparer": root / "preparer.py",
    }
    expected = {
        label: {
            "sha256": pinned[label][2],
            "byte_count": pinned[label][1].st_size,
            "path": path.name,
        }
        for label, path in files.items()
    }
    proof = root / ".complete.json"
    completion = {"schema": "alpha_max_phase_preparation_invocation_inputs.v1", "files": expected}

    for label, path in files.items():
        payload = _pinned_bytes(pinned[label][0], pinned[label][1], label)
        if sha256(payload) != pinned[label][2]:
            raise PreparationError(f"{label}_changed_during_read")
        stage = path.with_name(f".{path.name}.stage")
        destination_exists = _sidecar_exists(path, f"invocation_input_{label}")
        stage_exists = _sidecar_exists(stage, f"invocation_input_{label}_stage")
        if destination_exists and stage_exists:
            _assert_no_symlink_components(path, f"invocation_input_{label}", include_leaf=True)
            _assert_no_symlink_components(
                stage, f"invocation_input_{label}_stage", include_leaf=True
            )
            destination = os.lstat(path)
            staged = os.lstat(stage)
            if (
                not stat.S_ISREG(destination.st_mode)
                or not stat.S_ISREG(staged.st_mode)
                or destination.st_uid != os.geteuid()
                or staged.st_uid != os.geteuid()
                or stat.S_IMODE(destination.st_mode) != 0o444
                or stat.S_IMODE(staged.st_mode) != 0o444
                or destination.st_size != pinned[label][1].st_size
                or staged.st_size != pinned[label][1].st_size
                or destination.st_nlink != 2
                or staged.st_nlink != 2
                or destination.st_dev != staged.st_dev
                or destination.st_ino != staged.st_ino
            ):
                raise PreparationError("invocation_inputs_stage_conflict")
            os.unlink(stage)
            _fsync_directory(root, f"invocation_input_{label}")
            destination = _regular_file(path, f"invocation_input_{label}")
            if file_sha256(path, f"invocation_input_{label}") != pinned[label][2]:
                raise PreparationError("invocation_inputs_digest_invalid")
            continue
        if destination_exists:
            destination = _regular_file(path, f"invocation_input_{label}")
            if (
                stat.S_IMODE(destination.st_mode) != 0o444
                or destination.st_size != pinned[label][1].st_size
                or file_sha256(path, f"invocation_input_{label}") != pinned[label][2]
            ):
                raise PreparationError("invocation_inputs_digest_invalid")
            continue
        if stage_exists:
            staged = _regular_file(stage, f"invocation_input_{label}_stage")
            if staged.st_size > len(payload):
                raise PreparationError("invocation_inputs_stage_conflict")
            if staged.st_size == len(payload):
                if file_sha256(stage, f"invocation_input_{label}_stage") != pinned[label][2]:
                    raise PreparationError("invocation_inputs_digest_invalid")
                if stat.S_IMODE(staged.st_mode) != 0o444:
                    os.chmod(stage, 0o444)
            else:
                _write_or_resume(stage, payload, f"invocation_input_{label}")
                os.chmod(stage, 0o444)
        else:
            _write_or_resume(stage, payload, f"invocation_input_{label}")
            os.chmod(stage, 0o444)
        try:
            os.link(stage, path, follow_symlinks=False)
            _fsync_directory(root, f"invocation_input_{label}")
            os.unlink(stage)
            _fsync_directory(root, f"invocation_input_{label}")
        except FileExistsError:
            raise PreparationError("invocation_inputs_stage_conflict") from None
        except OSError as exc:
            raise PreparationError("invocation_inputs_publish_failed") from exc

    if _sidecar_exists(proof, "invocation_inputs_complete"):
        if _canonical_json_file(proof, "invocation_inputs_complete") != completion:
            raise PreparationError("invocation_inputs_complete_diverged")
        if stat.S_IMODE(_regular_file(proof, "invocation_inputs_complete").st_mode) == 0o600:
            os.chmod(proof, 0o444)
            _fsync_directory(root, "invocation_inputs_complete")
    else:
        _persist_immutable(
            proof.with_name(".complete.stage.json"), proof, completion, "invocation_inputs_complete"
        )
        os.chmod(proof, 0o444)
        _fsync_directory(root, "invocation_inputs_complete")

    root_fd = _open_snapshot_directory(root.parent, (root.name,), "invocation_inputs")
    try:
        children_fd = os.dup(root_fd)
        try:
            actual_names = set()
            with os.scandir(children_fd) as children:
                for child in children:
                    item = child.stat(follow_symlinks=False)
                    if (
                        not stat.S_ISREG(item.st_mode)
                        or item.st_nlink != 1
                        or stat.S_IMODE(item.st_mode) != 0o444
                    ):
                        raise PreparationError("invocation_inputs_inventory_invalid")
                    actual_names.add(child.name)
        finally:
            os.close(children_fd)
    finally:
        os.close(root_fd)
    if actual_names != {path.name for path in files.values()} | {proof.name}:
        raise PreparationError("invocation_inputs_inventory_invalid")
    for label, path in files.items():
        if file_sha256(path, f"invocation_input_{label}") != pinned[label][2]:
            raise PreparationError("invocation_inputs_digest_invalid")
    return files


def _open_materialized_inputs(
    files: dict[str, Path], expected: dict[str, tuple[int, os.stat_result, str]]
) -> dict[str, tuple[int, os.stat_result, str]]:
    with ExitStack() as cleanup:
        materialized = {}
        for label, path in files.items():
            materialized[label] = _open_pinned_file(path, f"materialized_{label}")
            cleanup.callback(os.close, materialized[label][0])
        for label, (_, _, digest) in materialized.items():
            if digest != expected[label][2]:
                raise PreparationError(f"materialized_{label}_digest_invalid")
        cleanup.pop_all()
        return materialized


def _assert_materialized_inputs_unchanged(
    files: dict[str, Path], pinned: dict[str, tuple[int, os.stat_result, str]]
) -> None:
    _assert_pinned_files_unchanged(pinned)
    for label, path in files.items():
        descriptor, item = _open_verified_file(path, f"materialized_{label}")
        try:
            before = pinned[label][1]
            if _stable_file_identity(item) != _stable_file_identity(before):
                raise PreparationError(f"materialized_{label}_namespace_changed")
        finally:
            os.close(descriptor)


def _snapshot_entries(source_root: Path, source_report: Path) -> list[dict[str, Any]]:
    manifest = _canonical_json_file(source_report / "source_manifest.json", "source_manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), list):
        raise PreparationError("source_manifest_invalid")
    entries: list[dict[str, Any]] = []
    for artifact in manifest["artifacts"]:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise PreparationError("source_manifest_artifact_invalid")
        relative, digest = artifact["path"], artifact["sha256"]
        path = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath(".")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or path.is_absolute()
            or path.as_posix() != relative
            or len(path.parts) < 3
            or path.parts[:2] not in {("output", "market_ohlcv_1s"), ("output", "feature_points")}
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            continue
        source = source_root.joinpath(*path.parts[1:])
        item = _regular_file(source, "snapshot_source")
        if file_sha256(source, "snapshot_source") != digest:
            raise PreparationError("snapshot_source_manifest_mismatch")
        entries.append(
            {
                "source_relative_path": "/".join(path.parts[1:]),
                "sha256": digest,
                "byte_count": item.st_size,
            }
        )
    if not entries or len({entry["source_relative_path"] for entry in entries}) != len(entries):
        raise PreparationError("snapshot_inventory_invalid")
    return sorted(entries, key=lambda entry: entry["source_relative_path"])


def _snapshot_relative_parts(value: str) -> tuple[str, ...]:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise PreparationError("snapshot_entry_relative_path_invalid")
    return path.parts


def _snapshot_inventory(
    snapshot: Path, entries: list[dict[str, Any]], *, finalize_modes: bool
) -> None:
    """Authenticate the complete snapshot namespace without following components."""
    expected_files = {entry["source_relative_path"] for entry in entries} | {
        "snapshot-manifest.json",
        ".complete.json",
    }
    expected_directories = {""}
    for relative_path in expected_files:
        parts = _snapshot_relative_parts(relative_path)
        expected_directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    actual_files: set[str] = set()
    actual_directories: set[str] = {""}

    def walk(directory_fd: int, relative_path: str) -> None:
        children_fd = os.dup(directory_fd)
        try:
            with os.scandir(children_fd) as children:
                for child in children:
                    child_relative = (
                        f"{relative_path}/{child.name}" if relative_path else child.name
                    )
                    observed = child.stat(follow_symlinks=False)
                    if observed.st_uid != os.geteuid():
                        raise PreparationError("snapshot_inventory_diverged")
                    if stat.S_ISDIR(observed.st_mode):
                        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
                    elif stat.S_ISREG(observed.st_mode):
                        if observed.st_nlink != 1:
                            raise PreparationError("snapshot_inventory_diverged")
                        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    else:
                        raise PreparationError("snapshot_inventory_diverged")
                    try:
                        child_fd = os.open(child.name, flags, dir_fd=directory_fd)
                    except OSError as exc:
                        raise PreparationError("snapshot_inventory_diverged") from exc
                    try:
                        item = os.fstat(child_fd)
                        if _stable_file_identity(item) != _stable_file_identity(observed):
                            raise PreparationError("snapshot_inventory_diverged")
                        if stat.S_ISDIR(item.st_mode):
                            if child_relative not in expected_directories or stat.S_IMODE(
                                item.st_mode
                            ) not in {
                                0o700,
                                0o555,
                            }:
                                raise PreparationError("snapshot_completed_mode_invalid")
                            actual_directories.add(child_relative)
                            walk(child_fd, child_relative)
                            if finalize_modes and stat.S_IMODE(os.fstat(child_fd).st_mode) != 0o555:
                                os.fchmod(child_fd, 0o555)
                                os.fsync(child_fd)
                        elif stat.S_ISREG(item.st_mode):
                            mode = stat.S_IMODE(item.st_mode)
                            if (
                                child_relative not in expected_files
                                or item.st_nlink != 1
                                or mode not in ({0o600, 0o444} if finalize_modes else {0o444})
                            ):
                                raise PreparationError("snapshot_completed_mode_invalid")
                            if finalize_modes and mode == 0o600:
                                os.fchmod(child_fd, 0o444)
                                os.fsync(child_fd)
                            actual_files.add(child_relative)
                        else:
                            raise PreparationError("snapshot_inventory_diverged")
                    finally:
                        os.close(child_fd)
        finally:
            os.close(children_fd)

    root_fd = _open_snapshot_directory(snapshot.parent, (snapshot.name,), "snapshot_root")
    try:
        root_item = os.fstat(root_fd)
        if root_item.st_uid != os.geteuid() or stat.S_IMODE(root_item.st_mode) not in {
            0o700,
            0o555,
        }:
            raise PreparationError("snapshot_completed_mode_invalid")
        walk(root_fd, "")
        if actual_files != expected_files or actual_directories != expected_directories:
            raise PreparationError("snapshot_inventory_diverged")
        if finalize_modes and stat.S_IMODE(os.fstat(root_fd).st_mode) != 0o555:
            os.fchmod(root_fd, 0o555)
            os.fsync(root_fd)
    except OSError as exc:
        raise PreparationError("snapshot_inventory_diverged") from exc
    finally:
        os.close(root_fd)
    if finalize_modes:
        _fsync_directory(snapshot.parent, "snapshot_parent")


def _open_snapshot_directory(snapshot: Path, relative_parts: tuple[str, ...], label: str) -> int:
    """Open or create clean snapshot-relative directories without symlink traversal."""
    if not relative_parts or any(part in ("", ".", "..") for part in relative_parts):
        raise PreparationError(f"{label}_invalid_relative_path")
    if not snapshot.is_absolute():
        raise PreparationError(f"{label}_root_unsafe")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in snapshot.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        os.close(descriptor)
        raise PreparationError(f"{label}_root_unsafe") from exc
    try:
        root_item = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(root_item.st_mode)
            or root_item.st_uid != os.geteuid()
            or stat.S_IMODE(root_item.st_mode) not in {0o700, 0o555}
        ):
            raise PreparationError(f"{label}_root_unsafe")
        for part in relative_parts:
            try:
                next_descriptor = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                    next_descriptor = os.open(
                        part,
                        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise PreparationError(f"{label}_create_failed") from exc
            except OSError as exc:
                raise PreparationError(f"{label}_unsafe_component") from exc
            item = os.fstat(next_descriptor)
            if (
                not stat.S_ISDIR(item.st_mode)
                or item.st_uid != os.geteuid()
                or stat.S_IMODE(item.st_mode) not in {0o700, 0o555}
            ):
                os.close(next_descriptor)
                raise PreparationError(f"{label}_unsafe_component")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _snapshot_target_parent(target: Path, entry: dict[str, Any]) -> int:
    relative_parts = _snapshot_relative_parts(entry["source_relative_path"])
    if target.name != relative_parts[-1]:
        raise PreparationError("snapshot_target_relative_path_invalid")
    snapshot = target
    for _ in relative_parts:
        snapshot = snapshot.parent
    if snapshot.joinpath(*relative_parts) != target:
        raise PreparationError("snapshot_target_relative_path_invalid")
    return _open_snapshot_directory(snapshot, relative_parts[:-1], "snapshot_target_parent")


def _create_snapshot_root(snapshot: Path) -> None:
    _private_sidecar_parent(snapshot, "snapshot_root")
    if not snapshot.parent.is_absolute():
        raise PreparationError("snapshot_create_failed")
    parent_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in snapshot.parent.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = next_descriptor
    except OSError as exc:
        os.close(parent_fd)
        raise PreparationError("snapshot_create_failed") from exc
    try:
        try:
            os.mkdir(snapshot.name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise PreparationError("snapshot_create_failed") from exc
    finally:
        os.close(parent_fd)
    snapshot_fd = _open_snapshot_directory(snapshot.parent, (snapshot.name,), "snapshot_root")
    os.close(snapshot_fd)
    _fsync_directory(snapshot.parent, "snapshot_parent")


def _open_no_follow(path: Path, flags: int, mode: int = 0o600) -> int:
    return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), mode)


def _stream_prefix_matches(source_fd: int, stage: Path, prefix_size: int) -> None:
    """Compare an authenticated prefix without routing data through metadata reads."""
    stage_fd, stage_before = _open_verified_file(stage, "snapshot_entry_stage")
    try:
        if stage_before.st_size != prefix_size:
            raise PreparationError("snapshot_stage_diverged")
        remaining = prefix_size
        while remaining:
            size = min(1 << 20, remaining)
            if os.read(source_fd, size) != os.read(stage_fd, size):
                raise PreparationError("snapshot_stage_diverged")
            remaining -= size
        _assert_stable_file(stage_fd, stage_before, "snapshot_entry_stage")
    finally:
        os.close(stage_fd)


def _copy_snapshot_entry(source: Path, stage: Path, entry: dict[str, Any]) -> None:
    if _sidecar_exists(stage, "snapshot_entry_stage"):
        stage_item = _regular_file(stage, "snapshot_entry_stage")
        current_size = stage_item.st_size
    else:
        current_size = 0
    if current_size > entry["byte_count"]:
        raise PreparationError("snapshot_stage_diverged")
    try:
        source_fd, source_item = _open_verified_file(source, "snapshot_source")
        if source_item.st_size != entry["byte_count"]:
            os.close(source_fd)
            raise PreparationError("snapshot_source_changed_during_copy")
        try:
            if current_size:
                _stream_prefix_matches(source_fd, stage, current_size)
            target_fd = _open_no_follow(stage, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                remaining = entry["byte_count"] - current_size
                while remaining:
                    block = os.read(source_fd, min(1 << 20, remaining))
                    if not block:
                        raise PreparationError("snapshot_source_changed_during_copy")
                    view = memoryview(block)
                    while view:
                        written = os.write(target_fd, view)
                        if written <= 0:
                            raise OSError("snapshot_copy_write_failed")
                        view = view[written:]
                    remaining -= len(block)
                if os.read(source_fd, 1):
                    raise PreparationError("snapshot_source_changed_during_copy")
                _assert_stable_file(source_fd, source_item, "snapshot_source")
                os.fsync(target_fd)
            finally:
                os.close(target_fd)
        finally:
            os.close(source_fd)
        stage_item = _regular_file(stage, "snapshot_entry_stage")
        if (
            stage_item.st_size != entry["byte_count"]
            or file_sha256(stage, "snapshot_entry_stage") != entry["sha256"]
        ):
            raise PreparationError("snapshot_source_changed_during_copy")
        _fsync_directory(stage.parent, "snapshot_entry_stage")
    except PreparationError:
        raise
    except OSError as exc:
        raise PreparationError("snapshot_copy_failed") from exc


def _publish_snapshot_stage(stage: Path, target: Path, entry: dict[str, Any]) -> None:
    stage_item = _regular_file(stage, "snapshot_entry_stage")
    if (
        stage_item.st_size != entry["byte_count"]
        or file_sha256(stage, "snapshot_entry_stage") != entry["sha256"]
    ):
        raise PreparationError("snapshot_stage_diverged")
    try:
        os.chmod(stage, 0o444)
        descriptor = _open_no_follow(stage, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(stage, target, follow_symlinks=False)
        linked_stage = os.stat(stage, follow_symlinks=False)
        linked_target = os.stat(target, follow_symlinks=False)
        if (
            linked_stage.st_nlink != 2
            or linked_target.st_nlink != 2
            or linked_stage.st_dev != linked_target.st_dev
            or linked_stage.st_ino != linked_target.st_ino
        ):
            raise PreparationError("snapshot_publish_identity_invalid")
        _fsync_directory(target.parent, "snapshot_entry_publish")
        os.unlink(stage)
        _fsync_directory(target.parent, "snapshot_entry_publish")
        _regular_file(target, "snapshot_clone")
    except FileExistsError as exc:
        raise PreparationError("snapshot_target_already_exists") from exc
    except OSError as exc:
        raise PreparationError("snapshot_publish_failed") from exc


def _recover_published_snapshot_stage(stage: Path, target: Path, entry: dict[str, Any]) -> bool:
    if not _sidecar_exists(stage, "snapshot_entry_stage"):
        return False
    stage_item = _regular_file(stage, "snapshot_entry_stage")
    if (
        stage_item.st_size != entry["byte_count"]
        or file_sha256(stage, "snapshot_entry_stage") != entry["sha256"]
    ):
        return False
    if stat.S_IMODE(stage_item.st_mode) == 0o444 and not target.exists():
        _publish_snapshot_stage(stage, target, entry)
        return True
    if not target.exists():
        return False
    target_item = _regular_file(target, "snapshot_clone")
    if (
        stage_item.st_nlink != 2
        or target_item.st_nlink != 2
        or stage_item.st_dev != target_item.st_dev
        or stage_item.st_ino != target_item.st_ino
    ):
        raise PreparationError("snapshot_publish_recovery_invalid")
    os.unlink(stage)
    _fsync_directory(target.parent, "snapshot_entry_publish")
    return True


def _snapshot_entry(source_root: Path, target: Path, entry: dict[str, Any]) -> None:
    source = source_root / entry["source_relative_path"]
    stage = target.with_name(f".{target.name}.snapshot-stage")
    target_parent = _snapshot_target_parent(target, entry)
    os.close(target_parent)
    if _recover_published_snapshot_stage(stage, target, entry):
        pass
    if target.exists():
        item = _regular_file(target, "snapshot_clone")
        if (
            item.st_size != entry["byte_count"]
            or file_sha256(target, "snapshot_clone") != entry["sha256"]
        ):
            raise PreparationError("snapshot_entry_diverged")
        return
    if _sidecar_exists(stage, "snapshot_entry_stage"):
        _copy_snapshot_entry(source, stage, entry)
        _publish_snapshot_stage(stage, target, entry)
    else:
        try:
            source_fd, source_item = _open_verified_file(source, "snapshot_source")
            try:
                if source_item.st_size != entry["byte_count"]:
                    raise PreparationError("snapshot_source_manifest_mismatch")
                stage_fd = _open_no_follow(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    fcntl.ioctl(stage_fd, FICLONE, source_fd)
                    os.fsync(stage_fd)
                finally:
                    os.close(stage_fd)
            finally:
                os.close(source_fd)
        except OSError as exc:
            if exc.errno not in {errno.EOPNOTSUPP, errno.ENOTSUP, errno.EXDEV, errno.ENOTTY}:
                raise PreparationError("snapshot_reflink_failed") from exc
            _copy_snapshot_entry(source, stage, entry)
        _publish_snapshot_stage(stage, target, entry)
    source_item = _regular_file(source, "snapshot_source")
    target_item = _regular_file(target, "snapshot_clone")
    if (
        source_item.st_ino == target_item.st_ino
        or source_item.st_size != entry["byte_count"]
        or target_item.st_size != entry["byte_count"]
        or file_sha256(target, "snapshot_clone") != entry["sha256"]
    ):
        raise PreparationError("snapshot_clone_invalid")


def _build_snapshot(
    snapshot: Path, source_root: Path, source_report: Path, descriptor_sha: str
) -> tuple[dict[str, Any], Path, Path]:
    entries = _snapshot_entries(source_root, source_report)
    value = {
        "schema": SNAPSHOT_SCHEMA,
        "descriptor_sha256": descriptor_sha,
        "source_manifest_sha256": file_sha256(
            source_report / "source_manifest.json", "source_manifest"
        ),
        "entries": entries,
    }
    _create_snapshot_root(snapshot)
    marker = snapshot / ".complete.json"
    complete = marker.exists()
    if complete:
        if _canonical_json_file(snapshot / "snapshot-manifest.json", "snapshot_manifest") != value:
            raise PreparationError("snapshot_manifest_diverged")
        if _canonical_json_file(marker, "snapshot_complete") != {
            "schema": SNAPSHOT_SCHEMA,
            "snapshot_manifest_sha256": sha256(canonical_bytes(value)),
        }:
            raise PreparationError("snapshot_complete_diverged")
        _snapshot_inventory(snapshot, entries, finalize_modes=True)
        for entry in entries:
            _snapshot_entry(source_root, snapshot / entry["source_relative_path"], entry)
        return value, snapshot / "market_ohlcv_1s", snapshot / "feature_points"
    if snapshot.stat().st_mode & 0o077:
        raise PreparationError("snapshot_resume_unsafe")
    snapshot_fd = _open_snapshot_directory(snapshot.parent, (snapshot.name,), "snapshot_root")
    os.close(snapshot_fd)
    for root in ("market_ohlcv_1s", "feature_points"):
        root_fd = _open_snapshot_directory(snapshot, (root,), "snapshot_target_parent")
        os.close(root_fd)
    for entry in entries:
        _snapshot_entry(source_root, snapshot / entry["source_relative_path"], entry)
    _persist_immutable(
        snapshot / ".snapshot-manifest.stage.json",
        snapshot / "snapshot-manifest.json",
        value,
        "snapshot_manifest",
    )
    _persist_immutable(
        snapshot / ".complete.stage.json",
        marker,
        {"schema": SNAPSHOT_SCHEMA, "snapshot_manifest_sha256": sha256(canonical_bytes(value))},
        "snapshot_complete",
    )
    _snapshot_inventory(snapshot, entries, finalize_modes=True)
    return value, snapshot / "market_ohlcv_1s", snapshot / "feature_points"


def _relative_output_path(value: Any) -> str:
    if not isinstance(value, str):
        raise PreparationError("preparation_manifest_entry_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or value == "preparation_manifest.json"
    ):
        raise PreparationError("preparation_manifest_entry_path_invalid")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise PreparationError(f"{label}_invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PreparationError(f"{label}_invalid") from exc
    if parsed.tzinfo != UTC or parsed.isoformat().replace("+00:00", "Z") != value:
        raise PreparationError(f"{label}_invalid")
    return parsed


def _read_pinned_parquet(descriptor: int, before: os.stat_result, label: str) -> pl.DataFrame:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        with io.FileIO(os.dup(descriptor), "rb", closefd=True) as stream:
            frame = pl.read_parquet(stream)
        _assert_stable_file(descriptor, before, label)
        return frame
    except PreparationError:
        raise
    except Exception as exc:
        raise PreparationError(f"{label}_invalid_parquet") from exc


def _read_verified_parquet(path: Path, label: str) -> pl.DataFrame:
    descriptor, before = _open_verified_file(path, label)
    try:
        return _read_pinned_parquet(descriptor, before, label)
    finally:
        os.close(descriptor)


def _pinned_file_sha256(descriptor: int, before: os.stat_result, label: str) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while block := os.read(descriptor, 1 << 20):
        digest.update(block)
    _assert_stable_file(descriptor, before, label)
    return digest.hexdigest()


def _open_pinned_relative_file(
    root_fd: int, relative_path: str, label: str
) -> tuple[int, os.stat_result]:
    descriptor = os.dup(root_fd)
    try:
        for component in _snapshot_relative_parts(relative_path):
            next_descriptor = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        os.close(descriptor)
        raise PreparationError(f"{label}_missing") from exc
    item = os.fstat(descriptor)
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        os.close(descriptor)
        raise PreparationError(f"{label}_unsafe_file")
    return descriptor, item


def _expected_output_layout(
    contract: dict[str, Any], snapshot: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = contract.get("records")
    if (
        contract.get("schema_version") != "alpha_max_contract_manifest.v2"
        or contract.get("exchange") != "binance"
        or not isinstance(records, list)
        or len(records) != 10
    ):
        raise PreparationError("contract_manifest_semantic_invalid")
    source_map = {
        item.get("source_relative_path"): item
        for item in snapshot.get("entries", [])
        if isinstance(item, dict)
    }
    if len(source_map) != len(snapshot.get("entries", [])):
        raise PreparationError("snapshot_inventory_invalid")
    availability = {
        kind: {"availability_start_by_symbol": {}, "availability_end_by_symbol": {}}
        for kind in ("raw", "feature")
    }
    expected: list[dict[str, Any]] = []
    symbols: set[str] = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or type(record.get("symbol")) is not str
            or record["symbol"] in symbols
            or record["symbol"]
            not in {
                "ADAUSDT",
                "AVAXUSDT",
                "BNBUSDT",
                "BTCUSDT",
                "DOGEUSDT",
                "ETHUSDT",
                "SOLUSDT",
                "TONUSDT",
                "TRXUSDT",
                "XRPUSDT",
            }
        ):
            raise PreparationError("contract_manifest_semantic_invalid")
        symbol = record["symbol"]
        symbols.add(symbol)
        for kind, prefix, is_month in (("raw", "raw", True), ("feature", "feature", False)):
            start = _parse_utc(
                record.get(f"{prefix}_availability_start_utc"), "contract_availability"
            )
            end = _parse_utc(record.get(f"{prefix}_availability_end_utc"), "contract_availability")
            if start >= end:
                raise PreparationError("contract_availability_invalid")
            availability[kind]["availability_start_by_symbol"][symbol] = record[
                f"{prefix}_availability_start_utc"
            ]
            availability[kind]["availability_end_by_symbol"][symbol] = record[
                f"{prefix}_availability_end_utc"
            ]
            for phase_id, phase_start_text, phase_end_text in PHASE_INTERVALS:
                effective_start, effective_end = (
                    max(start, _parse_utc(phase_start_text, "phase")),
                    min(end, _parse_utc(phase_end_text, "phase")),
                )
                cursor = (
                    effective_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    if is_month
                    else effective_start.replace(hour=0, minute=0, second=0, microsecond=0)
                )
                while cursor < effective_end:
                    following = (
                        (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
                        if is_month
                        else cursor + timedelta(days=1)
                    )
                    owned_start, owned_end = (
                        max(effective_start, cursor),
                        min(effective_end, following),
                    )
                    if is_month:
                        source_path = f"market_ohlcv_1s/binance/{symbol}/{cursor:%Y-%m}.parquet"
                        output_path = f"{phase_id}/raw/market_ohlcv_1s/binance/{symbol}/{cursor:%Y-%m}.parquet"
                    else:
                        prefix_path = f"feature_points/exchange=binance/symbol={symbol}/date={cursor:%Y-%m-%d}/"
                        matches = sorted(
                            key
                            for key in source_map
                            if isinstance(key, str)
                            and key.startswith(prefix_path)
                            and key.endswith(".parquet")
                        )
                        if len(matches) != 1:
                            raise PreparationError("snapshot_feature_partition_inventory_invalid")
                        source_path = matches[0]
                        output_path = f"{phase_id}/feature/feature_points/exchange=binance/symbol={symbol}/date={cursor:%Y-%m-%d}/part-0.parquet"
                    source = source_map.get(source_path)
                    if not isinstance(source, dict):
                        raise PreparationError("snapshot_source_mapping_invalid")
                    expected.append(
                        {
                            "phase_id": phase_id,
                            "root_kind": kind,
                            "symbol": symbol,
                            "owned_start_utc": owned_start.isoformat().replace("+00:00", "Z"),
                            "owned_end_utc": owned_end.isoformat().replace("+00:00", "Z"),
                            "output_relative_path": output_path,
                            "source_relative_path": source_path,
                            "source_sha256": source.get("sha256"),
                            "source_byte_count": source.get("byte_count"),
                        }
                    )
                    cursor = following
    if {entry["source_relative_path"] for entry in expected} != set(source_map):
        raise PreparationError("snapshot_source_inventory_mismatch")
    return sorted(expected, key=lambda entry: entry["output_relative_path"]), availability


def _authenticate_output_inner(
    output_root: Path,
    contract: dict[str, Any],
    contract_sha: str,
    snapshot: dict[str, Any],
    snapshot_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    cleanup = ExitStack()
    try:
        root_fd, output_root_identity = _open_authenticated_root(output_root, "output_root")
        cleanup.callback(os.close, root_fd)
        snapshot_fd, snapshot_root_identity = _open_authenticated_root(
            snapshot_root, "snapshot_root"
        )
        cleanup.callback(os.close, snapshot_fd)
        manifest_descriptor, manifest_item = _open_pinned_relative_file(
            root_fd, "preparation_manifest.json", "preparation_manifest"
        )
        cleanup.callback(os.close, manifest_descriptor)
        manifest_bytes = _pinned_bytes(manifest_descriptor, manifest_item, "preparation_manifest")
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreparationError("preparation_manifest_invalid_json") from exc
        if manifest_bytes != canonical_bytes(manifest):
            raise PreparationError("preparation_manifest_not_canonical")
        manifest_sha = sha256(manifest_bytes)
        required_fields = {
            "availability",
            "availability_sha256_by_root_kind",
            "contract_manifest_schema_version",
            "contract_manifest_sha256",
            "exchange",
            "file_count",
            "files",
            "phase_intervals",
            "schema_version",
            "symbols",
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != required_fields
            or manifest.get("schema_version") != "alpha_max_phase_root_preparation_manifest.v1"
            or type(manifest.get("file_count")) is not int
            or not isinstance(manifest.get("files"), list)
            or manifest["file_count"] != len(manifest["files"])
        ):
            raise PreparationError("preparation_manifest_invalid")
        layout, availability = _expected_output_layout(contract, snapshot)
        symbols = [record["symbol"] for record in contract["records"]]
        phases = [
            {"phase_id": phase_id, "start_utc": start, "end_utc": end}
            for phase_id, start, end in PHASE_INTERVALS
        ]
        if (
            manifest.get("contract_manifest_sha256") != contract_sha
            or manifest.get("contract_manifest_schema_version") != contract.get("schema_version")
            or manifest.get("exchange") != "binance"
            or manifest.get("symbols") != symbols
            or manifest.get("phase_intervals") != phases
            or manifest.get("availability") != availability
            or manifest.get("availability_sha256_by_root_kind")
            != {kind: sha256(canonical_bytes(availability[kind])) for kind in ("raw", "feature")}
        ):
            raise PreparationError("preparation_manifest_semantic_mismatch")
        required_entry_fields = {
            "output_byte_count",
            "output_relative_path",
            "output_row_count",
            "output_sha256",
            "owned_end_utc",
            "owned_start_utc",
            "phase_id",
            "root_kind",
            "source_byte_count",
            "source_relative_path",
            "source_sha256",
            "symbol",
        }
        expected_by_path = {entry["output_relative_path"]: entry for entry in layout}
        if (
            len(expected_by_path) != len(layout)
            or len(manifest["files"]) != len(layout)
            or [
                entry.get("output_relative_path") if isinstance(entry, dict) else None
                for entry in manifest["files"]
            ]
            != sorted(expected_by_path)
        ):
            raise PreparationError("preparation_manifest_entry_set_mismatch")
        manifest_by_path: dict[str, dict[str, Any]] = {}
        for entry in manifest["files"]:
            if not isinstance(entry, dict) or set(entry) != required_entry_fields:
                raise PreparationError("preparation_manifest_entry_invalid")
            relative = _relative_output_path(entry["output_relative_path"])
            expected = expected_by_path.get(relative)
            if expected is None or any(
                entry.get(field) != expected[field]
                for field in (
                    "phase_id",
                    "root_kind",
                    "symbol",
                    "owned_start_utc",
                    "owned_end_utc",
                    "source_relative_path",
                    "source_sha256",
                    "source_byte_count",
                )
            ):
                raise PreparationError("preparation_manifest_semantic_mismatch")
            if (
                type(entry["output_byte_count"]) is not int
                or entry["output_byte_count"] < 0
                or type(entry["output_row_count"]) is not int
                or entry["output_row_count"] < 0
                or type(entry["output_sha256"]) is not str
                or len(entry["output_sha256"]) != 64
            ):
                raise PreparationError("preparation_manifest_entry_invalid")
            manifest_by_path[relative] = entry
        actual: set[str] = set()
        directories: set[str] = set()
        pinned_outputs: dict[str, tuple[int, os.stat_result]] = {}
    except Exception:
        cleanup.close()
        raise
    try:

        def walk(directory_fd: int, relative_path: str) -> None:
            children_fd = os.dup(directory_fd)
            try:
                with os.scandir(children_fd) as children:
                    for child in children:
                        child_relative = (
                            f"{relative_path}/{child.name}" if relative_path else child.name
                        )
                        observed = child.stat(follow_symlinks=False)
                        if stat.S_ISLNK(observed.st_mode):
                            raise PreparationError("output_tree_symlink")
                        if stat.S_ISDIR(observed.st_mode):
                            flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
                        elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
                            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                        else:
                            raise PreparationError("output_tree_unsafe_file")
                        try:
                            child_fd = os.open(child.name, flags, dir_fd=directory_fd)
                        except OSError as exc:
                            raise PreparationError("output_tree_directory_changed") from exc
                        item = os.fstat(child_fd)
                        if _stable_file_identity(item) != _stable_file_identity(observed):
                            os.close(child_fd)
                            raise PreparationError("output_tree_directory_changed")
                        if stat.S_ISDIR(item.st_mode):
                            try:
                                directories.add(child_relative)
                                walk(child_fd, child_relative)
                            finally:
                                os.close(child_fd)
                        elif child_relative != "preparation_manifest.json":
                            actual.add(child_relative)
                            pinned_outputs[child_relative] = (child_fd, item)
                            cleanup.callback(os.close, child_fd)
                        else:
                            os.close(child_fd)
            finally:
                os.close(children_fd)

        walk(root_fd, "")
        expected_directories: set[str] = set()
        for relative in expected_by_path:
            parent = PurePosixPath(relative).parent
            while parent != PurePosixPath("."):
                expected_directories.add(parent.as_posix())
                parent = parent.parent
        if actual != set(expected_by_path) or directories != expected_directories:
            raise PreparationError("preparation_manifest_mismatch")
        snapshot_manifest_descriptor, snapshot_manifest_item = _open_pinned_relative_file(
            snapshot_fd, "snapshot-manifest.json", "snapshot_manifest"
        )
        cleanup.callback(os.close, snapshot_manifest_descriptor)
        if _pinned_bytes(
            snapshot_manifest_descriptor, snapshot_manifest_item, "snapshot_manifest"
        ) != canonical_bytes(snapshot):
            raise PreparationError("snapshot_manifest_diverged")
        pinned_sources: dict[str, tuple[int, os.stat_result]] = {}
        for entry in layout:
            relative = entry["source_relative_path"]
            if relative not in pinned_sources:
                pinned_sources[relative] = _open_pinned_relative_file(
                    snapshot_fd, relative, "snapshot_parquet"
                )
                cleanup.callback(os.close, pinned_sources[relative][0])
        source_declarations: dict[str, tuple[int, str]] = {}
        for entry in layout:
            relative = entry["source_relative_path"]
            declaration = (entry["source_byte_count"], entry["source_sha256"])
            if (
                type(declaration[0]) is not int
                or declaration[0] < 0
                or type(declaration[1]) is not str
                or len(declaration[1]) != 64
            ):
                raise PreparationError("snapshot_inventory_invalid")
            existing = source_declarations.setdefault(relative, declaration)
            if existing != declaration:
                raise PreparationError("snapshot_layout_declaration_conflict")
        for relative, (source_descriptor, source_item) in pinned_sources.items():
            expected_size, expected_sha = source_declarations[relative]
            if (
                source_item.st_size != expected_size
                or _pinned_file_sha256(source_descriptor, source_item, "snapshot_parquet")
                != expected_sha
            ):
                raise PreparationError("snapshot_parquet_mismatch")
        for relative in sorted(actual):
            entry = manifest_by_path[relative]
            output_descriptor, output_item = pinned_outputs[relative]
            source_descriptor, source_item = pinned_sources[entry["source_relative_path"]]
            if (
                output_item.st_size != entry["output_byte_count"]
                or _pinned_file_sha256(output_descriptor, output_item, "output_parquet")
                != entry["output_sha256"]
            ):
                raise PreparationError("preparation_manifest_mismatch")
            source_frame = _read_pinned_parquet(source_descriptor, source_item, "snapshot_parquet")
            output_frame = _read_pinned_parquet(output_descriptor, output_item, "output_parquet")
            start = _parse_utc(entry["owned_start_utc"], "owned_interval")
            end = _parse_utc(entry["owned_end_utc"], "owned_interval")
            if entry["root_kind"] == "raw":
                required = {"datetime", "open", "high", "low", "close", "volume"}
                if (
                    not required.issubset(source_frame.schema)
                    or source_frame.schema["datetime"] != pl.Datetime("ms")
                    or output_frame.schema != source_frame.schema
                ):
                    raise PreparationError("output_raw_schema_invalid")
                start_ms = int(start.timestamp() * 1000)
                end_ms = int(end.timestamp() * 1000)
                expected_frame = source_frame.filter(
                    (pl.col("datetime").dt.epoch("ms") >= start_ms)
                    & (pl.col("datetime").dt.epoch("ms") < end_ms)
                )
                timestamps = output_frame.get_column("datetime").dt.epoch("ms").to_list()
                if timestamps != list(
                    range(int(start.timestamp() * 1000), int(end.timestamp() * 1000), 1000)
                ):
                    raise PreparationError("output_raw_grid_invalid")
                numeric = output_frame.select(
                    [
                        pl.col(name).cast(pl.Float64)
                        for name in ("open", "high", "low", "close", "volume")
                    ]
                )
                if any(
                    series.null_count() or not bool(series.is_finite().all()) for series in numeric
                ):
                    raise PreparationError("output_raw_values_invalid")
                if bool(
                    output_frame.select(
                        (
                            (pl.col("open") <= 0)
                            | (pl.col("high") <= 0)
                            | (pl.col("low") <= 0)
                            | (pl.col("close") <= 0)
                            | (pl.col("volume") < 0)
                        ).any()
                    ).item()
                ):
                    raise PreparationError("output_raw_values_invalid")
                if not output_frame.filter(
                    (pl.col("high") < pl.col("open"))
                    | (pl.col("high") < pl.col("close"))
                    | (pl.col("low") > pl.col("open"))
                    | (pl.col("low") > pl.col("close"))
                    | (pl.col("high") < pl.col("low"))
                ).is_empty():
                    raise PreparationError("output_raw_values_invalid")
            else:
                required = [
                    "timestamp_ms",
                    "source_timestamp_ms",
                    "exchange",
                    "symbol",
                    "funding_rate",
                ]
                if output_frame.columns != required:
                    raise PreparationError("output_feature_schema_invalid")
                if (
                    "timestamp_ms" not in source_frame.schema
                    or "funding_rate" not in source_frame.schema
                ):
                    raise PreparationError("output_feature_source_schema_invalid")
                if "exchange" in source_frame.schema and (
                    source_frame.get_column("exchange").null_count()
                    or {
                        str(value).lower()
                        for value in source_frame.get_column("exchange").to_list()
                    }
                    != {"binance"}
                ):
                    raise PreparationError("output_feature_source_exchange_invalid")
                if "symbol" in source_frame.schema and (
                    source_frame.get_column("symbol").null_count()
                    or {str(value).upper() for value in source_frame.get_column("symbol").to_list()}
                    != {entry["symbol"]}
                ):
                    raise PreparationError("output_feature_source_symbol_invalid")
                interval = 14_400_000 if entry["symbol"] == "TONUSDT" else 28_800_000
                valid = source_frame.filter(
                    pl.col("funding_rate").is_not_null() & pl.col("funding_rate").is_finite()
                )
                source_times = valid.get_column("timestamp_ms").cast(pl.Int64)
                canonical = (source_times // interval) * interval
                if any(
                    source - settlement < 0 or source - settlement > 1000
                    for source, settlement in zip(
                        source_times.to_list(), canonical.to_list(), strict=True
                    )
                ):
                    raise PreparationError("output_feature_jitter_invalid")
                expected_frame = (
                    valid.with_columns(
                        [
                            pl.col("timestamp_ms").cast(pl.Int64).alias("source_timestamp_ms"),
                            ((pl.col("timestamp_ms").cast(pl.Int64) // interval) * interval).alias(
                                "timestamp_ms"
                            ),
                            pl.lit("binance").alias("exchange"),
                            pl.lit(entry["symbol"]).alias("symbol"),
                            pl.col("funding_rate").cast(pl.Float64),
                        ]
                    )
                    .filter(
                        (pl.col("timestamp_ms") >= int(start.timestamp() * 1000))
                        & (pl.col("timestamp_ms") < int(end.timestamp() * 1000))
                    )
                    .select(required)
                )
                expected_grid = list(
                    range(
                        ((int(start.timestamp() * 1000) + interval - 1) // interval) * interval,
                        int(end.timestamp() * 1000),
                        interval,
                    )
                )
                if (
                    output_frame.get_column("timestamp_ms").to_list() != expected_grid
                    or output_frame.get_column("exchange").to_list()
                    != ["binance"] * len(expected_grid)
                    or output_frame.get_column("symbol").to_list()
                    != [entry["symbol"]] * len(expected_grid)
                ):
                    raise PreparationError("output_feature_grid_invalid")
            if output_frame.height != entry["output_row_count"] or not output_frame.equals(
                expected_frame
            ):
                raise PreparationError("output_parquet_content_mismatch")
        return (
            {
                "file_count": len(layout),
                "output_root": os.fspath(output_root),
                "preparation_manifest_sha256": manifest_sha,
            },
            {
                "output_root": output_root_identity,
                "snapshot_root": snapshot_root_identity,
                "output_generation": {
                    "preparation_manifest.json": _stable_file_identity(manifest_item),
                    **{
                        relative: _stable_file_identity(item)
                        for relative, (_, item) in pinned_outputs.items()
                    },
                },
                "snapshot_generation": {
                    "snapshot-manifest.json": _stable_file_identity(snapshot_manifest_item),
                    **{
                        relative: _stable_file_identity(item)
                        for relative, (_, item) in pinned_sources.items()
                    },
                },
            },
        )
    finally:
        cleanup.close()


def _authenticate_output_with_provenance(
    output_root: Path,
    contract: dict[str, Any] | Path,
    contract_sha: str | dict[str, Any],
    snapshot: dict[str, Any] | Path,
    snapshot_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    if isinstance(contract, Path):
        if (
            not isinstance(contract_sha, dict)
            or not isinstance(snapshot, Path)
            or snapshot_root is not None
        ):
            raise PreparationError("contract_manifest_invalid")
        snapshot_root = snapshot
        snapshot = contract_sha
        contract_sha = file_sha256(contract, "contract_manifest")
        contract = _canonical_json_file(contract, "contract_manifest")
    if not isinstance(contract_sha, str) or not isinstance(snapshot, dict) or snapshot_root is None:
        raise PreparationError("contract_manifest_invalid")
    try:
        return _authenticate_output_inner(
            output_root, contract, contract_sha, snapshot, snapshot_root
        )
    except pl.exceptions.PolarsError as exc:
        raise PreparationError("output_parquet_semantic_invalid") from exc


def _authenticate_output(
    output_root: Path,
    contract: dict[str, Any] | Path,
    contract_sha: str | dict[str, Any],
    snapshot: dict[str, Any] | Path,
    snapshot_root: Path | None = None,
) -> dict[str, Any]:
    preparer_value, _ = _authenticate_output_with_provenance(
        output_root, contract, contract_sha, snapshot, snapshot_root
    )
    return preparer_value


def _open_generation_root(
    path: Path, expected: dict[str, int], label: str
) -> tuple[int, dict[str, int]]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise PreparationError(f"{label}_namespace_unavailable") from exc
    try:
        item = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise PreparationError(f"{label}_namespace_unavailable") from exc
    identity = {"st_dev": item.st_dev, "st_ino": item.st_ino}
    if identity != expected:
        os.close(descriptor)
        raise PreparationError(f"{label}_replaced")
    return descriptor, identity


def _open_generation_file(
    root_fd: int, relative_path: str, expected_sha: str, expected_size: int, label: str
) -> tuple[int, os.stat_result]:
    descriptor: int | None = None
    try:
        descriptor, item = _open_pinned_relative_file(root_fd, relative_path, label)
        if (
            item.st_size != expected_size
            or _pinned_file_sha256(descriptor, item, label) != expected_sha
        ):
            raise PreparationError(f"{label}_mismatch")
        return descriptor, item
    except PreparationError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise PreparationError(f"{label}_namespace_unavailable") from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _authenticate_public_generation(
    output_root: Path,
    snapshot_root: Path,
    preparer_value: dict[str, Any],
    snapshot_value: dict[str, Any],
    authenticated_roots: dict[str, dict[str, int]],
    cleanup: ExitStack,
) -> dict[str, Any]:
    """Pin every receipt-visible public object until final namespace rewalk."""
    output_fd, _ = _open_generation_root(
        output_root, authenticated_roots["output_root"], "output_root"
    )
    cleanup.callback(os.close, output_fd)
    snapshot_fd, _ = _open_generation_root(
        snapshot_root, authenticated_roots["snapshot_root"], "snapshot_root"
    )
    cleanup.callback(os.close, snapshot_fd)

    manifest_fd, manifest_item = _open_pinned_relative_file(
        output_fd, "preparation_manifest.json", "output_manifest"
    )
    cleanup.callback(os.close, manifest_fd)
    manifest_bytes = _pinned_bytes(manifest_fd, manifest_item, "output_manifest")
    if sha256(manifest_bytes) != preparer_value["preparation_manifest_sha256"]:
        raise PreparationError("output_manifest_mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError("output_manifest_invalid_json") from exc
    if manifest_bytes != canonical_bytes(manifest) or not isinstance(manifest.get("files"), list):
        raise PreparationError("output_manifest_invalid")

    snapshot_manifest_fd, snapshot_manifest_item = _open_pinned_relative_file(
        snapshot_fd, "snapshot-manifest.json", "snapshot_manifest"
    )
    cleanup.callback(os.close, snapshot_manifest_fd)
    snapshot_manifest_bytes = _pinned_bytes(
        snapshot_manifest_fd, snapshot_manifest_item, "snapshot_manifest"
    )
    if snapshot_manifest_bytes != canonical_bytes(snapshot_value):
        raise PreparationError("snapshot_manifest_diverged")
    complete_fd, complete_item = _open_pinned_relative_file(
        snapshot_fd, ".complete.json", "snapshot_complete"
    )
    cleanup.callback(os.close, complete_fd)
    if _pinned_bytes(complete_fd, complete_item, "snapshot_complete") != canonical_bytes(
        {
            "schema": SNAPSHOT_SCHEMA,
            "snapshot_manifest_sha256": sha256(canonical_bytes(snapshot_value)),
        }
    ):
        raise PreparationError("snapshot_complete_diverged")
    _snapshot_inventory(snapshot_root, snapshot_value["entries"], finalize_modes=False)

    output_files: dict[str, tuple[int, os.stat_result, str, int]] = {
        "preparation_manifest.json": (
            manifest_fd,
            manifest_item,
            preparer_value["preparation_manifest_sha256"],
            manifest_item.st_size,
        )
    }
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise PreparationError("output_manifest_invalid")
        relative = _relative_output_path(entry.get("output_relative_path"))
        digest, size = entry.get("output_sha256"), entry.get("output_byte_count")
        if not isinstance(digest, str) or type(size) is not int:
            raise PreparationError("output_manifest_invalid")
        descriptor, item = _open_generation_file(
            output_fd, relative, digest, size, "output_parquet"
        )
        cleanup.callback(os.close, descriptor)
        output_files[relative] = (descriptor, item, digest, size)

    snapshot_files: dict[str, tuple[int, os.stat_result, str, int]] = {
        "snapshot-manifest.json": (
            snapshot_manifest_fd,
            snapshot_manifest_item,
            sha256(canonical_bytes(snapshot_value)),
            snapshot_manifest_item.st_size,
        ),
        ".complete.json": (
            complete_fd,
            complete_item,
            sha256(
                canonical_bytes(
                    {
                        "schema": SNAPSHOT_SCHEMA,
                        "snapshot_manifest_sha256": sha256(canonical_bytes(snapshot_value)),
                    }
                )
            ),
            complete_item.st_size,
        ),
    }
    for entry in snapshot_value["entries"]:
        relative, digest, size = (
            entry["source_relative_path"],
            entry["sha256"],
            entry["byte_count"],
        )
        descriptor, item = _open_generation_file(
            snapshot_fd, relative, digest, size, "snapshot_parquet"
        )
        cleanup.callback(os.close, descriptor)
        snapshot_files[relative] = (descriptor, item, digest, size)
    for relative, (_, item, _, _) in output_files.items():
        if authenticated_roots["output_generation"].get(relative) != _stable_file_identity(item):
            raise PreparationError("output_generation_replaced")
    for relative, (_, item, _, _) in snapshot_files.items():
        expected = authenticated_roots["snapshot_generation"].get(relative)
        if expected is not None and expected != _stable_file_identity(item):
            raise PreparationError("snapshot_generation_replaced")
    return {
        "output_fd": output_fd,
        "snapshot_fd": snapshot_fd,
        "output_files": output_files,
        "snapshot_files": snapshot_files,
    }


def _assert_generation_inventory(
    root_fd: int, files: dict[str, tuple[int, os.stat_result, str, int]], label: str
) -> None:
    expected_files = set(files)
    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files: set[str] = set()
    actual_directories: set[str] = set()

    def walk(directory_fd: int, relative_path: str) -> None:
        children_fd = os.dup(directory_fd)
        try:
            with os.scandir(children_fd) as children:
                for child in children:
                    child_relative = (
                        f"{relative_path}/{child.name}" if relative_path else child.name
                    )
                    observed = child.stat(follow_symlinks=False)
                    if stat.S_ISDIR(observed.st_mode):
                        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
                    elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
                        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                    else:
                        raise PreparationError(f"{label}_generation_inventory_mismatch")
                    child_fd = os.open(child.name, flags, dir_fd=directory_fd)
                    try:
                        if _stable_file_identity(os.fstat(child_fd)) != _stable_file_identity(
                            observed
                        ):
                            raise PreparationError(f"{label}_generation_inventory_mismatch")
                        if stat.S_ISDIR(observed.st_mode):
                            actual_directories.add(child_relative)
                            walk(child_fd, child_relative)
                        else:
                            actual_files.add(child_relative)
                    finally:
                        os.close(child_fd)
        finally:
            os.close(children_fd)

    try:
        walk(root_fd, "")
    except PreparationError:
        raise
    except OSError as exc:
        raise PreparationError(f"{label}_generation_inventory_unavailable") from exc
    if actual_files != expected_files or actual_directories != expected_directories:
        raise PreparationError(f"{label}_generation_inventory_mismatch")


def _assert_public_generation_unchanged(
    output_root: Path, snapshot_root: Path, snapshot_value: dict[str, Any], token: dict[str, Any]
) -> None:
    """Rewalk public names from pinned roots; path replacement cannot satisfy this token."""
    for label, descriptor in (
        ("output_root", token["output_fd"]),
        ("snapshot_root", token["snapshot_fd"]),
    ):
        try:
            os.fstat(descriptor)
        except OSError as exc:
            raise PreparationError(f"{label}_namespace_unavailable") from exc
    _snapshot_inventory(snapshot_root, snapshot_value["entries"], finalize_modes=False)
    for root_path, root_fd, files, label in (
        (snapshot_root, token["snapshot_fd"], token["snapshot_files"], "snapshot"),
        (output_root, token["output_fd"], token["output_files"], "output"),
    ):
        try:
            namespace_fd = os.open(
                root_path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
            )
        except OSError as exc:
            raise PreparationError(f"{label}_root_namespace_unavailable") from exc
        try:
            try:
                if _stable_file_identity(os.fstat(namespace_fd)) != _stable_file_identity(
                    os.fstat(root_fd)
                ):
                    raise PreparationError(f"{label}_root_replaced")
            except PreparationError:
                raise
            except OSError as exc:
                raise PreparationError(f"{label}_root_namespace_unavailable") from exc
            for relative, (descriptor, before, digest, size) in files.items():
                try:
                    _assert_stable_file(descriptor, before, f"{label}_generation")
                    reopened, reopened_item = _open_generation_file(
                        namespace_fd, relative, digest, size, f"{label}_generation"
                    )
                except PreparationError:
                    raise
                except OSError as exc:
                    raise PreparationError(f"{label}_generation_namespace_unavailable") from exc
                try:
                    if _stable_file_identity(reopened_item) != _stable_file_identity(before):
                        raise PreparationError(f"{label}_generation_mismatch")
                finally:
                    os.close(reopened)
            _assert_generation_inventory(namespace_fd, files, label)
        finally:
            os.close(namespace_fd)


def _output_exists(output_root: Path) -> bool:
    _assert_no_symlink_components(output_root, "output_root", include_leaf=False)
    try:
        item = output_root.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise PreparationError("output_root_unreadable") from exc
    if stat.S_ISLNK(item.st_mode):
        raise PreparationError("output_root_symlink")
    return True


def _main_with_lock_inner(
    args: argparse.Namespace, lock_identity: dict[str, int], cleanup: ExitStack
) -> int:
    acquirer = _absolute_clean_path(args.acquirer, "acquirer")
    source_root = _absolute_clean_path(args.source_root, "source_root")
    source_report = _absolute_clean_path(args.source_report, "source_report")
    contract = _absolute_clean_path(args.contract_manifest, "contract_manifest")
    evidence = _absolute_clean_path(args.availability_evidence, "availability_evidence")
    preparer = _absolute_clean_path(args.preparer, "preparer")
    output_root = _absolute_clean_path(args.output_root, "output_root")
    forbidden_roots = [
        _absolute_clean_path(value, "forbidden_root") for value in args.forbidden_root
    ]
    sidecars = _sidecars(output_root)
    paths = {
        "source_root": source_root,
        "source_report": source_report,
        "contract_manifest": contract,
        "availability_evidence": evidence,
        "acquirer": acquirer,
        "preparer": preparer,
        "output_root": output_root,
        "sidecar_descriptor_stage": sidecars["descriptor_stage"],
        "sidecar_descriptor": sidecars["descriptor"],
        "sidecar_receipt_stage": sidecars["receipt_stage"],
        "sidecar_receipt": sidecars["receipt"],
        "sidecar_snapshot": sidecars["snapshot"],
        "sidecar_inputs": sidecars["inputs"],
        "sidecar_lock": sidecars["lock"],
    }
    _reject_overlap(paths, forbidden_roots)
    _regular_file(acquirer, "acquirer")
    _regular_file(contract, "contract_manifest")
    _regular_file(evidence, "availability_evidence")
    _regular_file(preparer, "preparer")
    output_exists = _output_exists(output_root)
    raw_root = source_root / "market_ohlcv_1s"
    feature_root = source_root / "feature_points"
    _directory_identity(raw_root, "raw_root")
    _directory_identity(feature_root, "feature_root")

    pinned = {}
    for label, path in {
        "acquirer": acquirer,
        "contract_manifest": contract,
        "availability_evidence": evidence,
        "preparer": preparer,
    }.items():
        pinned[label] = _open_pinned_file(path, label)
        cleanup.callback(os.close, pinned[label][0])
    acquirer_sha = pinned["acquirer"][2]
    contract_sha = pinned["contract_manifest"][2]
    evidence_sha = pinned["availability_evidence"][2]
    preparer_sha = pinned["preparer"][2]
    wrapper_sha = file_sha256(Path(__file__), "wrapper")
    if acquirer_sha != ACQUIRER_SHA256:
        raise PreparationError("acquirer_sha256_not_approved")
    if contract_sha != CONTRACT_SHA256:
        raise PreparationError("contract_sha256_not_approved")
    if evidence_sha != EVIDENCE_SHA256:
        raise PreparationError("availability_evidence_sha256_not_approved")
    if preparer_sha != PREPARER_SHA256:
        raise PreparationError("preparer_sha256_not_approved")

    before = _eligibility_snapshot(source_root, source_report)
    verifier_argv = [
        sys.executable,
        os.fspath(acquirer),
        "--contract-manifest",
        os.fspath(contract),
        "--availability-evidence",
        os.fspath(evidence),
        "--output-root",
        os.fspath(source_root),
        "--report-dir",
        os.fspath(source_report),
    ]
    for forbidden in forbidden_roots:
        verifier_argv.extend(("--forbidden-root", os.fspath(forbidden)))
    verifier_argv.append("--verify-eligible")
    snapshot_raw_root = sidecars["snapshot"] / "market_ohlcv_1s"
    snapshot_feature_root = sidecars["snapshot"] / "feature_points"
    preparer_argv = [
        sys.executable,
        os.fspath(preparer),
        "--raw-root",
        os.fspath(snapshot_raw_root),
        "--feature-root",
        os.fspath(snapshot_feature_root),
        "--contract-manifest",
        os.fspath(contract),
        "--output-root",
        os.fspath(output_root),
    ]
    inputs = _materialize_invocation_inputs(sidecars["inputs"], pinned)
    materialized = _open_materialized_inputs(inputs, pinned)
    for descriptor, _, _ in materialized.values():
        cleanup.callback(os.close, descriptor)
    contract_bytes = _pinned_bytes(
        materialized["contract_manifest"][0],
        materialized["contract_manifest"][1],
        "materialized_contract_manifest",
    )
    try:
        contract_value = json.loads(contract_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError("materialized_contract_manifest_invalid_json") from exc
    if contract_bytes != canonical_bytes(contract_value):
        raise PreparationError("materialized_contract_manifest_not_canonical")
    verifier_exec_argv = verifier_argv.copy()
    verifier_exec_argv[1] = _pinned_fd_path(materialized["acquirer"][0])
    verifier_exec_argv[3] = os.fspath(inputs["contract_manifest"])
    verifier_exec_argv[5] = os.fspath(inputs["availability_evidence"])
    verifier_exec_argv.extend(("--verifier-code-fd", str(materialized["acquirer"][0])))
    preparer_exec_argv = preparer_argv.copy()
    preparer_exec_argv[1] = _pinned_fd_path(materialized["preparer"][0])
    preparer_exec_argv[7] = os.fspath(inputs["contract_manifest"])
    pinned_fds = tuple(value[0] for value in materialized.values())
    for label in (
        "descriptor_stage",
        "descriptor",
        "receipt_stage",
        "receipt",
        "snapshot",
        "inputs",
        "lock",
    ):
        _private_sidecar_parent(sidecars[label], label)
    descriptor = {
        "schema": "alpha_max_phase_preparation_invocation.v1",
        "paths": {
            "acquirer": os.fspath(acquirer),
            "source_root": os.fspath(source_root),
            "source_report": os.fspath(source_report),
            "contract_manifest": os.fspath(contract),
            "availability_evidence": os.fspath(evidence),
            "preparer": os.fspath(preparer),
            "output_root": os.fspath(output_root),
            "raw_root": os.fspath(snapshot_raw_root),
            "feature_root": os.fspath(snapshot_feature_root),
            "invocation_descriptor": os.fspath(sidecars["descriptor"]),
            "invocation_descriptor_stage": os.fspath(sidecars["descriptor_stage"]),
            "handoff_receipt": os.fspath(sidecars["receipt"]),
            "handoff_receipt_stage": os.fspath(sidecars["receipt_stage"]),
            "source_snapshot": os.fspath(sidecars["snapshot"]),
            "source_snapshot_manifest": os.fspath(sidecars["snapshot_manifest"]),
            "source_snapshot_complete": os.fspath(sidecars["snapshot_complete"]),
            "invocation_inputs": os.fspath(sidecars["inputs"]),
            "invocation_input_acquirer": os.fspath(inputs["acquirer"]),
            "invocation_input_contract_manifest": os.fspath(inputs["contract_manifest"]),
            "invocation_input_availability_evidence": os.fspath(inputs["availability_evidence"]),
            "invocation_input_preparer": os.fspath(inputs["preparer"]),
            "invocation_lock": os.fspath(sidecars["lock"]),
        },
        "forbidden_roots": [os.fspath(root) for root in forbidden_roots],
        "frozen_sha256": {
            "acquirer": acquirer_sha,
            "contract_manifest": contract_sha,
            "availability_evidence": evidence_sha,
            "preparer": preparer_sha,
            "wrapper": wrapper_sha,
        },
        "invocation_lock_identity": lock_identity,
        "source_eligibility_snapshot": before,
        "verifier_argv": verifier_argv,
        "preparer_argv": preparer_argv,
    }
    descriptor_exists = _sidecar_exists(sidecars["descriptor"], "invocation_descriptor")
    descriptor_stage_exists = _sidecar_exists(
        sidecars["descriptor_stage"], "invocation_descriptor_stage"
    )
    if output_exists and not descriptor_exists and not descriptor_stage_exists:
        raise PreparationError("output_root_without_invocation_descriptor")
    _persist_immutable(
        sidecars["descriptor_stage"],
        sidecars["descriptor"],
        descriptor,
        "invocation_descriptor",
    )
    lock_fd, _ = _open_verified_file(sidecars["descriptor"], "invocation_descriptor")
    cleanup.callback(os.close, lock_fd)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        output_exists = _output_exists(output_root)
        _assert_materialized_inputs_unchanged(inputs, materialized)
        _assert_pinned_files_unchanged(pinned)
        verifier = _capture(verifier_exec_argv, "verifier", pass_fds=pinned_fds)
        _assert_materialized_inputs_unchanged(inputs, materialized)
        _assert_pinned_files_unchanged(pinned)
        if verifier.returncode != 0:
            raise PreparationError("eligible_source_verification_failed")
        verified = _require_unchanged(before, source_root, source_report)
        if verified != descriptor["source_eligibility_snapshot"]:
            raise PreparationError("invocation_descriptor_source_snapshot_diverged")
        snapshot_value, snapshot_raw_root, snapshot_feature_root = _build_snapshot(
            sidecars["snapshot"], source_root, source_report, sha256(canonical_bytes(descriptor))
        )
        if _require_unchanged(verified, source_root, source_report) != verified:
            raise PreparationError("eligible_source_changed_after_snapshot")
        if output_exists:
            preparer_value, authenticated_roots = _authenticate_output_with_provenance(
                output_root, contract_value, contract_sha, snapshot_value, sidecars["snapshot"]
            )
        else:
            if _sidecar_exists(sidecars["receipt"], "handoff_receipt") or _sidecar_exists(
                sidecars["receipt_stage"], "handoff_receipt_stage"
            ):
                raise PreparationError("handoff_receipt_without_output_root")
            _require_unchanged(verified, source_root, source_report)
            _assert_materialized_inputs_unchanged(inputs, materialized)
            _assert_pinned_files_unchanged(pinned)
            preparer_result = _capture(preparer_exec_argv, "preparer", pass_fds=pinned_fds)
            _assert_materialized_inputs_unchanged(inputs, materialized)
            _assert_pinned_files_unchanged(pinned)
            if preparer_result.returncode != 0:
                raise PreparationError("phase_preparer_failed")
            _require_unchanged(verified, source_root, source_report)
            try:
                declared = json.loads(preparer_result.stdout)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PreparationError("phase_preparer_stdout_invalid_json") from exc
            preparer_value, authenticated_roots = _authenticate_output_with_provenance(
                output_root, contract_value, contract_sha, snapshot_value, sidecars["snapshot"]
            )
            if declared != preparer_value:
                raise PreparationError("phase_preparer_stdout_mismatch")
        receipt = {
            "schema": "alpha_max_phase_preparation_eligible_source_receipt.v2",
            "invocation_descriptor_sha256": sha256(canonical_bytes(descriptor)),
            "source_eligibility_snapshot": verified,
            "verifier_argv_sha256": argv_sha256(verifier_argv),
            "preparer_argv_sha256": argv_sha256(preparer_argv),
            "preparer_result": preparer_value,
            "output_root_identity": authenticated_roots["output_root"],
            "source_snapshot_manifest_sha256": sha256(canonical_bytes(snapshot_value)),
            "source_snapshot_identity": authenticated_roots["snapshot_root"],
            "output_manifest_sha256": preparer_value["preparation_manifest_sha256"],
        }
        _assert_materialized_inputs_unchanged(inputs, materialized)
        _assert_pinned_files_unchanged(pinned)
        _assert_directory_identity(output_root, authenticated_roots["output_root"], "output_root")
        _assert_directory_identity(
            sidecars["snapshot"], authenticated_roots["snapshot_root"], "snapshot_root"
        )
        generation_token = _authenticate_public_generation(
            output_root,
            sidecars["snapshot"],
            preparer_value,
            snapshot_value,
            authenticated_roots,
            cleanup,
        )
        _assert_public_generation_unchanged(
            output_root, sidecars["snapshot"], snapshot_value, generation_token
        )
        _persist_immutable(
            sidecars["receipt_stage"], sidecars["receipt"], receipt, "handoff_receipt"
        )
        print(canonical_bytes(receipt).decode(), end="")
        return 0
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _main_with_lock(args: argparse.Namespace, lock_identity: dict[str, int]) -> int:
    with ExitStack() as cleanup:
        return _main_with_lock_inner(args, lock_identity, cleanup)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    acquirer = _absolute_clean_path(args.acquirer, "acquirer")
    source_root = _absolute_clean_path(args.source_root, "source_root")
    source_report = _absolute_clean_path(args.source_report, "source_report")
    contract = _absolute_clean_path(args.contract_manifest, "contract_manifest")
    evidence = _absolute_clean_path(args.availability_evidence, "availability_evidence")
    preparer = _absolute_clean_path(args.preparer, "preparer")
    output_root = _absolute_clean_path(args.output_root, "output_root")
    forbidden_roots = [
        _absolute_clean_path(value, "forbidden_root") for value in args.forbidden_root
    ]
    sidecars = _sidecars(output_root)
    _reject_overlap(
        {
            "source_root": source_root,
            "source_report": source_report,
            "contract_manifest": contract,
            "availability_evidence": evidence,
            "acquirer": acquirer,
            "preparer": preparer,
            "output_root": output_root,
            "sidecar_descriptor_stage": sidecars["descriptor_stage"],
            "sidecar_descriptor": sidecars["descriptor"],
            "sidecar_receipt_stage": sidecars["receipt_stage"],
            "sidecar_receipt": sidecars["receipt"],
            "sidecar_snapshot": sidecars["snapshot"],
            "sidecar_inputs": sidecars["inputs"],
            "sidecar_lock": sidecars["lock"],
        },
        forbidden_roots,
    )
    lock_fd, lock_identity = _open_invocation_lock(sidecars["lock"])
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current_fd, current_identity = _open_invocation_lock(_sidecars(output_root)["lock"])
        try:
            if current_identity != lock_identity:
                raise PreparationError("invocation_lock_replaced")
        finally:
            os.close(current_fd)
        return _main_with_lock(args, lock_identity)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreparationError as exc:
        print(f"run_alpha_max_phase_preparation_from_eligible_source: {exc}", file=sys.stderr)
        raise SystemExit(2)
