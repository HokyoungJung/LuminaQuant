#!/usr/bin/env python3
"""Publish an authenticated Alpha-Max source into the shared parquet store only."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
import base64
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import polars as pl
import subprocess

from lumina_quant.alpha_max_terminal_policy import (
    ALPHA_MAX_PUBLICATION_OBSERVER_READY_SCHEMA,
    ALPHA_MAX_PUBLICATION_OBSERVER_READY_UNSIGNED_FIELDS,
    TerminalPolicyError,
    alpha_max_canonical_inventory_records,
    is_alpha_max_coordination_lock as _is_coordination_lock,
    verify_message,
    verify_signed_receipt,
)
from lumina_quant.market_data import MarketDataRepository
from lumina_quant.storage.parquet import ParquetMarketDataRepository

_STORAGE = {
    "host_reserve_path": "/mnt/c",
    "host_reserve_bytes": 21_474_836_480,
    "max_live_archives": 1,
    "archive_retention": "retired_after_double_derivation",
}
_MAX_JSON = 64 * 1024 * 1024
_CONFLICT_AUTH_MAX_JSON = 4 * 1024 * 1024
_PUBLICATION_CONTROL_RESERVE_BYTES = 64 * 1024 * 1024
_PARQUET_ENCODING_RESERVE_BYTES = 64 * 1024 * 1024
_HEX = set("0123456789abcdef")
_CONFLICT_AUTHORIZATION_DOMAIN = b"luminaquant.alpha_max.canonical_conflict_authorization.v2\0"
_CONFLICT_RUN_ID = "482f9e03e246eda50641d06d81dcf17084799e7815656361bb62663dd1f149ea"
_CURRENT_APPROVAL_LEAF = "current-state-approval-v13.json"
_ACQUISITION_REQUEST_ID = "4d55958bf9387a63f1ce77f38e7e063909a550fce66aff873fc1d3b85851d152"
_PHASE_REQUEST_ID = "2e1d21418d6e1a24ffbc9ebcc6f042ab5a9a5743f0512a11704f8a43782a765d"
_ONE_TOUCH_REQUEST_ID = "bd12f06acde927fa1284d92b1efe2e1cbd4606a7cf1f1f9fb1a0cab933b8e514"
_PUBLICATION_STAGE_ENVELOPE_DOMAIN = b"luminaquant.alpha_max.publication_stage_envelope.v1\0"
_PUBLICATION_STAGE_ENVELOPE_SCHEMA = "alpha_max_publication_stage_envelope.v1"
_APPROVAL_OWNER_SESSION_PATH = ".gjc/_session-019fad7d-536a-7000-b794-52ccaa961746/"
_APPROVAL_OWNER_EXCLUDE = f":(exclude){_APPROVAL_OWNER_SESSION_PATH}**"
_APPROVAL_IGNORED_PATHS = (
    "src",
    "scripts",
    "apps",
    ":(exclude)apps/dashboard_web/.next/**",
    ":(exclude)apps/dashboard_web/node_modules/**",
)
_APPROVAL_RUNTIME_ROOTS = {
    "current_python": Path("/home/hoky/Quants-agent/LuminaQuant/.venv-g056v8-current"),
    "accepted_python": Path(
        "/home/hoky/Quants-agent/LuminaQuant-alpha-max-fresh-20260718/.venv-g056v8-accepted"
    ),
    "base_python": Path("/home/hoky/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu"),
}
_APPROVAL_RECOVERY_ROOT = Path("/home/hoky/quants-recovery-runs")
_APPROVAL_EXECUTION_ALIAS = Path(f"/mnt/wsl/luminaquant-alpha-max-execution-{_CONFLICT_RUN_ID}")

_APPROVAL_ACCEPTED_ALPHA_COMMIT = "391000b40717386765bfa39bd212d91c2e3be794"
_APPROVAL_BASELINE_ANCESTOR = "629d91e5d4aac26911af65a4a5e15ebdcbded30f"
_APPROVAL_ACCEPTED_SOURCE_ROOT = Path(
    "/home/hoky/Quants-agent/LuminaQuant-alpha-max-fresh-20260718"
)
_APPROVAL_ABSENT_RECOVERY_ARTIFACTS = {
    "control_root": f"/home/hoky/quants-recovery-runs/g056v8-controls-{_CONFLICT_RUN_ID}",
    "key_root": f"/home/hoky/quants-recovery-runs/g056v8-keys-{_CONFLICT_RUN_ID}",
    "evidence_root": f"/home/hoky/quants-recovery-runs/g056v8-acquisition-evidence-{_CONFLICT_RUN_ID}",
    "telemetry_root": f"/home/hoky/quants-recovery-runs/g056v8-telemetry-{_CONFLICT_RUN_ID}",
    "output_parent": f"/home/hoky/quants-recovery-runs/g056v8-acquisition-output-{_CONFLICT_RUN_ID}",
    "stage_results_parent": f"/home/hoky/quants-recovery-runs/g056v8-stage-results-{_CONFLICT_RUN_ID}",
}
_PREOPEN_ROLLBACK_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "kind",
        "request_id",
        "approval_sha256",
        "activation_intent_sha256",
        "observer_ready_sha256",
        "failure_reason",
        "failure_evidence_leaf",
        "failure_evidence_sha256",
        "candidate_identity",
        "predecessor_identity",
        "swap_identity",
    }
)


class PublicationError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _absolute(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise PublicationError(f"{label} must be a clean absolute path")
    return Path(os.path.normpath(path))


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
    )


def _private_regular(
    path: Path, label: str
) -> tuple[int, tuple[int, int, int, int, int, int, int, int, int]]:
    try:
        named = os.lstat(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise PublicationError(f"{label} is unavailable") from exc
    info = os.fstat(fd)
    ident = _identity(info)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or _identity(named) != ident:
        os.close(fd)
        raise PublicationError(f"{label} is not a private regular file")
    return fd, ident


def _revalidate(
    path: Path, fd: int, expected: tuple[int, int, int, int, int, int, int, int, int], label: str
) -> None:
    try:
        named = os.lstat(path)
        actual = _identity(os.fstat(fd))
    except OSError as exc:
        raise PublicationError(f"{label} disappeared") from exc
    if actual != expected or _identity(named) != expected:
        raise PublicationError(f"{label} identity drift")


def _regular_bytes(path: Path, label: str, limit: int = _MAX_JSON) -> bytes:
    fd, identity = _private_regular(path, label)
    try:
        if identity[3] > limit:
            raise PublicationError(f"{label} exceeds bounded read limit")
        data = bytearray()
        while len(data) < identity[3]:
            chunk = os.read(fd, min(65536, identity[3] - len(data)))
            if not chunk:
                raise PublicationError(f"{label} was truncated")
            data.extend(chunk)
        _revalidate(path, fd, identity, label)
        return bytes(data)
    finally:
        os.close(fd)


def _conflict_public_key(path: Path) -> bytes:
    label = "conflict authority public key"
    fd, identity = _private_regular(path, label)
    try:
        if (
            identity[3] != 32
            or stat.S_IMODE(identity[2]) not in {0o400, 0o444}
            or identity[7] != os.getuid()
        ):
            raise PublicationError(
                "conflict authority public key must be an owned 32-byte 0400 or 0444 file"
            )
        data = bytearray()
        while len(data) < 32:
            chunk = os.read(fd, 32 - len(data))
            if not chunk:
                raise PublicationError("conflict authority public key was truncated")
            data.extend(chunk)
        _revalidate(path, fd, identity, label)
        return bytes(data)
    finally:
        os.close(fd)


def _publication_private_key(fd: int) -> Ed25519PrivateKey:
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size != 32:
            raise PublicationError("publisher key fd must reference a raw 32-byte regular file")
        data = bytearray()
        while len(data) < 32:
            chunk = os.read(fd, 32 - len(data))
            if not chunk:
                raise PublicationError("publisher key fd was truncated")
            data.extend(chunk)
        if os.read(fd, 1):
            raise PublicationError("publisher key fd has trailing bytes")
        return Ed25519PrivateKey.from_private_bytes(bytes(data))
    except OSError as exc:
        raise PublicationError("publisher key fd is unavailable") from exc


def _stage_envelope(
    private_key: Ed25519PrivateKey, kind: str, message: dict[str, Any]
) -> dict[str, Any]:
    if kind not in {"activation", "open_window", "replay"}:
        raise PublicationError("publication stage kind is invalid")
    inner = {**message, "kind": kind}
    public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    unsigned = {
        "schema": _PUBLICATION_STAGE_ENVELOPE_SCHEMA,
        "kind": kind,
        "authority_key_id": hashlib.sha256(public).hexdigest(),
        "message": inner,
    }
    return {
        **unsigned,
        "signature": base64.b64encode(
            private_key.sign(_PUBLICATION_STAGE_ENVELOPE_DOMAIN + canonical_bytes(unsigned))
        ).decode("ascii"),
    }


def _verify_stage_envelope(
    path: Path, publisher_key: Ed25519PrivateKey, expected_kind: str
) -> dict[str, Any]:
    envelope, _ = _json(path, "publication stage envelope")
    public = publisher_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    if (
        set(envelope) != {"schema", "kind", "authority_key_id", "message", "signature"}
        or envelope.get("schema") != _PUBLICATION_STAGE_ENVELOPE_SCHEMA
        or envelope.get("kind") != expected_kind
        or envelope.get("authority_key_id") != hashlib.sha256(public).hexdigest()
        or not isinstance(envelope.get("message"), dict)
        or envelope["message"].get("kind") != expected_kind
        or not isinstance(envelope.get("signature"), str)
    ):
        raise PublicationError("publication stage envelope binding is invalid")
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(
            base64.b64decode(envelope["signature"], validate=True),
            _PUBLICATION_STAGE_ENVELOPE_DOMAIN
            + canonical_bytes(
                {key: value for key, value in envelope.items() if key != "signature"}
            ),
        )
    except (ValueError, InvalidSignature) as exc:
        raise PublicationError("publication stage envelope signature is invalid") from exc
    return envelope["message"]


def _manager_terminal_or_observer_failure(control: Path) -> None:
    names = (
        "window-decision-intent.json",
        "W7_ROLLING_BACK.json",
        "W8_FINALIZING.json",
        "rollback-exchange-fsynced.json",
        "predecessor-cleanup-manifest.json",
        "predecessor-quarantined.json",
        "predecessor-cleanup-fsynced.json",
        "rolled-back.json",
        "finalized.json",
        "completed.json",
        "observer-failure.json",
        "observer-failure-intent.json",
        "failure-intent.json",
    )
    if any((control / name).exists() for name in names):
        raise PublicationError("manager terminal or observer failure prevents publication replay")


def _preopen_failure_evidence(
    control: Path,
    observer_public_key: Path,
    *,
    request_id: str,
    approval_sha256: str,
    observer_ready_sha256: str,
) -> tuple[str, str] | None:
    for name in (
        "observer-failure.json",
        "observer-failure-intent.json",
        "observer-terminal-fail.json",
    ):
        path = control / name
        if not path.exists():
            continue
        envelope, raw = _json(path, "observer failure evidence")
        public = _conflict_public_key(observer_public_key)
        unsigned = {key: value for key, value in envelope.items() if key != "signature"}
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(
                base64.b64decode(envelope.get("signature", ""), validate=True),
                _PUBLICATION_STAGE_ENVELOPE_DOMAIN + canonical_bytes(unsigned),
            )
        except (TypeError, ValueError, InvalidSignature) as exc:
            raise PublicationError("observer failure evidence signature is invalid") from exc
        message = envelope.get("message")
        if (
            set(envelope) != {"schema", "kind", "authority_key_id", "message", "signature"}
            or envelope.get("schema") != _PUBLICATION_STAGE_ENVELOPE_SCHEMA
            or envelope.get("kind") != "observer"
            or envelope.get("authority_key_id") != hashlib.sha256(public).hexdigest()
            or not isinstance(message, dict)
            or message.get("kind") != "observer_failure"
            or message.get("outcome") != "FAIL"
            or message.get("run_id") != _CONFLICT_RUN_ID
            or message.get("request_id") != request_id
            or message.get("approval_leaf") != _CURRENT_APPROVAL_LEAF
            or message.get("approval_sha256") != approval_sha256
            or not isinstance(message.get("frozen_observation"), dict)
        ):
            raise PublicationError("observer failure evidence binding is invalid")
        return name, hashlib.sha256(raw).hexdigest()
    return None


def _validate_preopen_rollback_intent(
    rollback: dict[str, Any],
    *,
    control: Path,
    observer_public_key: Path,
    request_id: str,
    approval_sha256: str,
    activation_intent_sha256: str,
    candidate_identity: list[Any],
    predecessor_identity: list[Any],
    swap_identity: list[Any],
    observer_ready_sha256: str | None,
) -> None:
    expected = {
        "fields": _PREOPEN_ROLLBACK_FIELDS,
        "schema": "alpha_max_publication_pre_open_rollback.v2",
        "phase": "rollback_intent",
        "kind": "activation",
        "request_id": request_id,
        "approval_sha256": approval_sha256,
        "activation_intent_sha256": activation_intent_sha256,
        "observer_ready_sha256": observer_ready_sha256,
        "candidate_identity": candidate_identity,
        "predecessor_identity": predecessor_identity,
        "swap_identity": swap_identity,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if (set(rollback) if field == "fields" else rollback.get(field)) != value
    ]
    if mismatches:
        raise PublicationError(
            "pre-open rollback intent binding is invalid: " + ", ".join(mismatches)
        )
    leaf = rollback["failure_evidence_leaf"]
    digest = rollback["failure_evidence_sha256"]
    if (leaf is None) != (digest is None) or (
        leaf is not None and (not isinstance(leaf, str) or not isinstance(digest, str))
    ):
        raise PublicationError("pre-open rollback failure evidence binding is invalid")
    if leaf is not None and (control / leaf).exists():
        verified = _preopen_failure_evidence(
            control,
            observer_public_key,
            request_id=request_id,
            approval_sha256=approval_sha256,
            observer_ready_sha256=observer_ready_sha256,
        )
        if verified != (leaf, digest):
            raise PublicationError("pre-open rollback failure evidence changed")


def _json(path: Path, label: str, *, limit: int = _MAX_JSON) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes(path, label, limit=limit)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise PublicationError(f"{label} is not canonical JSON")
    return value, raw


def _process_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        ticks = int(raw.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError, UnicodeError) as exc:
        raise PublicationError("observer process identity is unavailable") from exc
    if ticks <= 0:
        raise PublicationError("observer process start time is invalid")
    return ticks


def _observer_process_alive(ready: dict[str, Any]) -> None:
    pid = ready.get("observer_pid")
    uid = ready.get("observer_uid")
    if type(pid) is not int or pid <= 0 or type(uid) is not int or uid != os.getuid():
        raise PublicationError("observer process binding is invalid")
    try:
        process = os.stat(f"/proc/{pid}", follow_symlinks=False)
    except OSError as exc:
        raise PublicationError("bound observer process is not alive") from exc
    if process.st_uid != uid or _process_start_ticks(pid) != ready.get("observer_start_ticks"):
        raise PublicationError("bound observer process identity changed")


def _observer_ready(
    path: Path,
    public_key_path: Path,
    *,
    request_id: str,
    approval_sha256: str,
    canonical_root: Path,
) -> tuple[dict[str, Any], str]:
    value, raw = _json(path, "publication observer readiness")
    public_bytes = _conflict_public_key(public_key_path)
    public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
    try:
        unsigned = verify_message("publication_observer_ready", value, public_key)
    except TerminalPolicyError as exc:
        raise PublicationError(str(exc)) from exc
    if (
        set(unsigned) != ALPHA_MAX_PUBLICATION_OBSERVER_READY_UNSIGNED_FIELDS
        or unsigned.get("schema") != ALPHA_MAX_PUBLICATION_OBSERVER_READY_SCHEMA
        or unsigned.get("kind") != "publication_observer_ready"
        or unsigned.get("run_id") != _CONFLICT_RUN_ID
        or unsigned.get("request_id") != request_id
        or unsigned.get("approval_leaf") != _CURRENT_APPROVAL_LEAF
        or unsigned.get("approval_sha256") != approval_sha256
        or unsigned.get("canonical_root") != str(canonical_root)
        or unsigned.get("observer_key_id") != hashlib.sha256(public_bytes).hexdigest()
        or unsigned.get("observer_uid") != os.getuid()
        or not isinstance(unsigned.get("query_spec_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", unsigned["query_spec_sha256"]) is None
        or not isinstance(unsigned.get("old_identity"), list)
        or len(unsigned["old_identity"]) != 3
        or unsigned["old_identity"][2] != "directory"
        or not isinstance(unsigned.get("old_inventory_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", unsigned["old_inventory_sha256"]) is None
    ):
        raise PublicationError("publication observer readiness binding is invalid")
    ready = {
        **unsigned,
        "observer_signature_b64": value["observer_signature_b64"],
    }
    return ready, hashlib.sha256(raw).hexdigest()


def _approval_canonical(value: Any) -> bytes:
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


def _approval_record(value: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(value).hexdigest(), "byte_count": len(value)}


def _valid_approval_record(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"sha256", "byte_count"}
        and isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
        and isinstance(value["byte_count"], int)
        and not isinstance(value["byte_count"], bool)
        and value["byte_count"] >= 0
    )


def _approval_git(root: Path, *arguments: str) -> bytes:
    if arguments and arguments[0] == "diff":
        arguments = ("diff", "--no-ext-diff", *arguments[1:])
    return subprocess.run(
        (
            "/usr/bin/git",
            "--no-optional-locks",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(root),
            *arguments,
        ),
        check=True,
        capture_output=True,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    ).stdout


def _approval_source_git(root: Path, *arguments: str) -> bytes:
    return _approval_git(root, *arguments, "--", ".", _APPROVAL_OWNER_EXCLUDE)


def _approval_inventory_entry(root: Path, name: str) -> dict[str, Any]:
    if not name or name.startswith("/") or ".." in Path(name).parts:
        raise PublicationError("current-state approval inventory path is unsafe")
    path = root / name
    before = os.lstat(path)
    if stat.S_ISREG(before.st_mode):
        if (
            before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            raise PublicationError("current-state approval inventory file is unsafe")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(descriptor)
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(descriptor)
            named_after = os.lstat(path)
        finally:
            os.close(descriptor)
        if (
            _identity(before) != _identity(opened)
            or _identity(opened) != _identity(after)
            or _identity(after) != _identity(named_after)
            or size != after.st_size
        ):
            raise PublicationError("current-state approval inventory file identity drift")
        kind = "regular"
        content_size = size
        sha256 = digest.hexdigest()
        info = after
    elif stat.S_ISLNK(before.st_mode):
        if before.st_nlink != 1 or before.st_uid != os.getuid():
            raise PublicationError("current-state approval inventory symlink is unsafe")
        target = os.readlink(path)
        info = os.lstat(path)
        content = os.fsencode(target)
        if _identity(before) != _identity(info) or len(content) != info.st_size:
            raise PublicationError("current-state approval inventory symlink identity drift")
        kind = "symlink"
        content_size = len(content)
        sha256 = hashlib.sha256(content).hexdigest()
    else:
        raise PublicationError("current-state approval inventory entry is unsafe")
    return {
        "path": name,
        "type": kind,
        "mode": stat.S_IMODE(info.st_mode),
        "size": content_size,
        "sha256": sha256,
    }


def _approval_inventory(root: Path, names: list[bytes]) -> bytes:
    records = [_approval_inventory_entry(root, raw.decode("utf-8")) for raw in names]
    return _approval_canonical(records)


def _approval_source_inventory(root: Path) -> bytes:
    names = _approval_source_git(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    ).split(b"\0")[:-1]
    return _approval_inventory(root, names)


def _approval_ignored_source_inventory(root: Path) -> bytes:
    names = _approval_git(
        root,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        *_APPROVAL_IGNORED_PATHS,
    ).split(b"\0")[:-1]
    return _approval_inventory(root, names)


def _approval_runtime_inventory(root: Path) -> bytes:
    names: list[bytes] = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        relative = Path(directory).relative_to(root)
        for name in (*subdirectories, *filenames):
            path = Path(directory) / name
            info = os.lstat(path)
            if stat.S_ISDIR(info.st_mode):
                continue
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                raise PublicationError("current-state approval runtime entry is unsafe")
            names.append(os.fsencode(str(relative / name)))
    return _approval_inventory(root, sorted(names))


def _approval_execution_alias() -> dict[str, Any]:
    before = os.lstat(_APPROVAL_EXECUTION_ALIAS)
    if not stat.S_ISLNK(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
        raise PublicationError("current-state execution alias is unsafe")
    target = os.readlink(_APPROVAL_EXECUTION_ALIAS)
    after = os.lstat(_APPROVAL_EXECUTION_ALIAS)
    if (
        target != str(_APPROVAL_RECOVERY_ROOT)
        or _identity(before) != _identity(after)
        or len(os.fsencode(target)) != after.st_size
    ):
        raise PublicationError("current-state execution alias identity drift")
    return {
        "path": str(_APPROVAL_EXECUTION_ALIAS),
        "target": target,
        "st_dev": after.st_dev,
        "st_ino": after.st_ino,
        "st_uid": after.st_uid,
        "st_gid": after.st_gid,
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": after.st_nlink,
    }


def _approval(path: Path) -> tuple[dict[str, Any], str]:
    if path.name != _CURRENT_APPROVAL_LEAF:
        raise PublicationError("current-state approval leaf is invalid")
    approval, raw = _json(path, "current-state approval")
    required = {
        "schema",
        "repository_root",
        "head",
        "accepted_alpha_commit",
        "baseline_ancestor",
        "verdict",
        "porcelain",
        "commit_overlay",
        "worktree_overlay",
        "source_inventory",
        "ignored_source_inventory",
        "runtime_inventories",
        "execution_alias",
        "run_id",
        "request_ids",
        "absent_recovery_artifacts",
        "accepted_source_state",
    }
    repository_root = Path(__file__).resolve().parents[2]
    records = (
        "porcelain",
        "commit_overlay",
        "worktree_overlay",
        "source_inventory",
        "ignored_source_inventory",
    )
    accepted_state_fields = {
        "root",
        "head",
        "porcelain",
        "source_inventory",
        "ignored_source_inventory",
    }
    if (
        set(approval) != required
        or approval.get("schema") != "alpha_max_v8_current_state_approval.v3"
        or approval.get("repository_root") != str(repository_root)
        or approval.get("run_id") != _CONFLICT_RUN_ID
        or approval.get("request_ids")
        != {
            "acquisition": _ACQUISITION_REQUEST_ID,
            "phase_preparation": _PHASE_REQUEST_ID,
            "one_touch": _ONE_TOUCH_REQUEST_ID,
        }
        or approval.get("verdict") != "PASS_REVIEWED_OVERLAY"
        or any(not _valid_approval_record(approval.get(name)) for name in records)
        or not isinstance(approval.get("runtime_inventories"), dict)
        or set(approval["runtime_inventories"]) != set(_APPROVAL_RUNTIME_ROOTS)
        or any(
            not _valid_approval_record(value) for value in approval["runtime_inventories"].values()
        )
        or approval.get("accepted_alpha_commit") != _APPROVAL_ACCEPTED_ALPHA_COMMIT
        or approval.get("baseline_ancestor") != _APPROVAL_BASELINE_ANCESTOR
        or approval.get("absent_recovery_artifacts") != _APPROVAL_ABSENT_RECOVERY_ARTIFACTS
        or not isinstance(approval.get("accepted_source_state"), dict)
        or set(approval["accepted_source_state"]) != accepted_state_fields
        or approval["accepted_source_state"].get("root") != str(_APPROVAL_ACCEPTED_SOURCE_ROOT)
        or approval["accepted_source_state"].get("head") != _APPROVAL_ACCEPTED_ALPHA_COMMIT
        or any(
            not _valid_approval_record(approval["accepted_source_state"].get(name))
            for name in ("porcelain", "source_inventory", "ignored_source_inventory")
        )
    ):
        raise PublicationError("current-state approval binding is invalid")
    try:
        actual = {
            "head": _approval_git(repository_root, "rev-parse", "HEAD").decode().strip(),
            "porcelain": _approval_record(
                _approval_source_git(repository_root, "status", "--porcelain=v1", "-z")
            ),
            "commit_overlay": _approval_record(
                _approval_source_git(
                    repository_root,
                    "diff",
                    "--binary",
                    f"{approval['accepted_alpha_commit']}..HEAD",
                )
            ),
            "worktree_overlay": _approval_record(
                _approval_source_git(repository_root, "diff", "--binary", "HEAD")
            ),
            "source_inventory": _approval_record(_approval_source_inventory(repository_root)),
            "ignored_source_inventory": _approval_record(
                _approval_ignored_source_inventory(repository_root)
            ),
            "runtime_inventories": {
                name: _approval_record(_approval_runtime_inventory(root))
                for name, root in _APPROVAL_RUNTIME_ROOTS.items()
            },
            "execution_alias": _approval_execution_alias(),
            "accepted_source_state": {
                "root": str(_APPROVAL_ACCEPTED_SOURCE_ROOT),
                "head": _approval_git(_APPROVAL_ACCEPTED_SOURCE_ROOT, "rev-parse", "HEAD")
                .decode()
                .strip(),
                "porcelain": _approval_record(
                    _approval_source_git(
                        _APPROVAL_ACCEPTED_SOURCE_ROOT, "status", "--porcelain=v1", "-z"
                    )
                ),
                "source_inventory": _approval_record(
                    _approval_source_inventory(_APPROVAL_ACCEPTED_SOURCE_ROOT)
                ),
                "ignored_source_inventory": _approval_record(
                    _approval_ignored_source_inventory(_APPROVAL_ACCEPTED_SOURCE_ROOT)
                ),
            },
        }
    except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        raise PublicationError("current-state approval repository readback failed") from exc
    if any(approval.get(name) != value for name, value in actual.items()):
        raise PublicationError("current-state approval repository binding drift")
    return approval, hashlib.sha256(raw).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise PublicationError(f"{label} must be a lowercase SHA-256")
    return value


def _inside(root: Path, relative: str, label: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or any(p in {"", "."} for p in pure.parts)
    ):
        raise PublicationError(f"{label} escapes its authenticated root")
    current = root
    try:
        root_info = os.lstat(root)
    except OSError as exc:
        raise PublicationError(f"{label} root missing") from exc
    if not stat.S_ISDIR(root_info.st_mode) or root_info.st_nlink < 1:
        raise PublicationError(f"{label} root unsafe")
    for component in pure.parts[:-1]:
        current = current / component
        info = os.lstat(current)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise PublicationError(f"{label} has unsafe parent")
    return current / pure.parts[-1]


def _authorized_ohlcv_relative(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    path = PurePosixPath(value)
    return (
        len(path.parts) == 4
        and not path.is_absolute()
        and ".." not in path.parts
        and all(component not in {"", "."} for component in path.parts)
        and path.parts[:2] == ("market_ohlcv_1s", "binance")
        and path.parts[3].endswith(".parquet")
    )


def _receipt_path(report: Path, relative: str) -> Path:
    return _inside(
        report,
        "partitions/" + hashlib.sha256(relative.encode()).hexdigest() + ".json",
        "partition receipt",
    )


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("short receipt write")
        offset += written


def _ensure_private_directory(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise PublicationError("private directory has no existing ancestor")
        cursor = parent
    ancestor = os.lstat(cursor)
    if not stat.S_ISDIR(ancestor.st_mode):
        raise PublicationError("private directory ancestor is unsafe")
    for directory in reversed(missing):
        os.mkdir(directory, 0o700)
        _fsync_dir(directory.parent)
    info = os.lstat(path)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise PublicationError("private directory identity is unsafe")


def _receipt_write(fd: int, payload: bytes) -> None:
    _write_all(fd, payload)


def _receipt_fsync(fd: int) -> None:
    os.fsync(fd)


def _receipt_install_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _remove_receipt_temp_remnants(path: Path) -> None:
    remnants = sorted(path.parent.glob(f".{path.name}.tmp-*"))
    if len(remnants) > 64:
        raise PublicationError("too many receipt temp remnants")
    removed = False
    for remnant in remnants:
        info = os.lstat(remnant)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > _MAX_JSON
            or info.st_nlink != 1
        ):
            raise PublicationError("publication receipt temp remnant is unsafe")
        remnant.unlink()
        removed = True
    if removed:
        _fsync_dir(path.parent)


def _write_noreplace(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    _ensure_private_directory(path.parent)
    _remove_receipt_temp_remnants(path)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        _receipt_write(fd, payload)
        _receipt_fsync(fd)
    except BaseException:
        os.close(fd)
        raise
    else:
        os.close(fd)
    try:
        _receipt_install_noreplace(temporary, path)
    except FileExistsError:
        existing, raw = _json(path, "publication receipt")
        if existing != value or raw != payload:
            raise PublicationError("publication receipt conflict")
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    _fsync_dir(path.parent)
    if _regular_bytes(path, "publication receipt") != payload:
        raise PublicationError("publication receipt readback mismatch")


def _artifact(item: Any, expected_path: str) -> tuple[str, int]:
    if (
        not isinstance(item, dict)
        or set(item) != {"kind", "path", "sha256", "byte_count"}
        or item.get("path") != expected_path
    ):
        raise PublicationError("terminal validated artifact shape is invalid")
    return _sha(item.get("sha256"), "terminal artifact hash"), _positive_int(
        item.get("byte_count"), "terminal artifact byte count"
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicationError(f"{label} is invalid")
    return value


def _terminal(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, dict[str, tuple[str, int]], str]:
    verified = verify_signed_receipt(args.terminal_receipt, args.authority_public_key)
    message = verified.message
    results = message.get("target_results")
    if (
        message.get("scope") != "acquisition"
        or message.get("terminal_state", {}).get("kind") != "SUCCEEDED"
        or not isinstance(results, list)
        or len(results) != 2
    ):
        raise PublicationError("terminal receipt does not authorize successful acquisition")
    if any(not isinstance(item, dict) or item.get("return_code") != 0 for item in results):
        raise PublicationError("terminal receipt has unsuccessful results")
    expected = {
        "source_eligible_receipt": str(args.source_report / "source_eligible_receipt.json"),
        "source_manifest": str(args.source_report / "source_manifest.json"),
        "source_journal": str(args.source_report / "acquisition.journal.jsonl"),
    }
    command_bindings: list[dict[str, tuple[str, int]]] = []
    for command in results:
        artifacts = command.get("validated_artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 3:
            raise PublicationError("terminal command source artifacts are not exact")
        bound: dict[str, tuple[str, int]] = {}
        for kind, path in expected.items():
            matches = [
                item for item in artifacts if isinstance(item, dict) and item.get("kind") == kind
            ]
            if len(matches) != 1:
                raise PublicationError("terminal command source artifacts are not exact")
            bound[kind] = _artifact(matches[0], path)
        command_bindings.append(bound)
    if command_bindings[0] != command_bindings[1]:
        raise PublicationError("terminal command artifact bindings diverge")
    return message, verified.key_id, command_bindings[1], verified.receipt_sha256


def _conflict_controls(
    args: argparse.Namespace,
    *,
    terminal: dict[str, Any],
    approval_sha256: str | None,
    terminal_sha256: str,
    request_id: str,
    key_id: str,
    predecessor_inventory: list[dict[str, Any]],
) -> dict[str, str] | None:
    values = (
        args.conflict_authorization_receipt,
        args.conflict_authority_public_key,
        args.fresh_acquisition_audit_receipt,
        args.wal_transition_receipt,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise PublicationError(
            "conflict authorization, fresh audit, and WAL transition controls must be paired"
        )
    fresh, fresh_bytes = _json(
        args.fresh_acquisition_audit_receipt,
        "fresh acquisition audit receipt",
        limit=_MAX_JSON,
    )
    wal, wal_bytes = _json(
        args.wal_transition_receipt,
        "WAL transition receipt",
        limit=_MAX_JSON,
    )
    for path, label in (
        (args.fresh_acquisition_audit_receipt, "fresh acquisition audit receipt"),
        (args.wal_transition_receipt, "WAL transition receipt"),
    ):
        info = os.lstat(path)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise PublicationError(f"{label} is not private")
    telemetry = fresh.get("composite_telemetry")
    if (
        fresh.get("schema") != "luminaquant_fresh_acquisition_audit.v1"
        or fresh.get("run_id") != _CONFLICT_RUN_ID
        or fresh.get("request_id") != request_id
        or fresh.get("sealed") is not True
        or fresh.get("outcome") != "pass"
        or fresh.get("terminal_receipt_sha256") != terminal_sha256
        or fresh.get("authority_key_id") != key_id
        or fresh.get("source_root") != str(args.source_root)
        or fresh.get("source_report") != str(args.source_report)
        or fresh.get("approval_sha256") != approval_sha256
        or fresh.get("signed_terminal_request_sha256") != terminal.get("request_sha256")
        or not isinstance(telemetry, dict)
        or not isinstance(telemetry.get("sha256"), str)
    ):
        raise PublicationError("fresh acquisition audit receipt binding is invalid")
    telemetry_digest = _sha(telemetry["sha256"], "fresh composite telemetry digest")
    if (
        wal.get("schema") != "luminaquant.canonical_wal_transition.v1"
        or wal.get("mode") != "execute"
        or wal.get("compaction_complete") is not True
        or wal.get("run_id") != _CONFLICT_RUN_ID
        or wal.get("request_id") != request_id
        or wal.get("canonical_root") != str(args.canonical_root)
        or wal.get("post_transition_inventory") != predecessor_inventory
        or wal.get("approval_sha256") != approval_sha256
        or wal.get("post_transition_inventory_sha256") != _inventory_digest(predecessor_inventory)
    ):
        raise PublicationError("WAL transition receipt binding is invalid")
    return {
        "fresh_acquisition_audit_receipt_sha256": hashlib.sha256(fresh_bytes).hexdigest(),
        "approval_sha256": approval_sha256,
        "acquisition_run_id": _CONFLICT_RUN_ID,
        "composite_telemetry_sha256": telemetry_digest,
        "wal_transition_receipt_sha256": hashlib.sha256(wal_bytes).hexdigest(),
        "wal_post_transition_inventory_sha256": wal["post_transition_inventory_sha256"],
        "signed_terminal_request_sha256": fresh["signed_terminal_request_sha256"],
    }


def _conflict_authorization(
    args: argparse.Namespace,
    *,
    terminal_sha256: str,
    request_id: str,
    hashes: dict[str, str],
    predecessor: Path,
    predecessor_inventory: list[dict[str, Any]],
    parts: list[dict[str, Any]],
    controls: dict[str, str] | None = None,
    terminal_public_key: bytes = b"",
    allow_predecessor_ctime_transition: bool = False,
) -> dict[str, Any] | None:
    receipt_path = args.conflict_authorization_receipt
    key_path = args.conflict_authority_public_key
    if receipt_path is None and key_path is None and controls is None:
        return None
    if receipt_path is None or key_path is None or controls is None:
        raise PublicationError("conflict authorization controls must be complete")
    receipt, receipt_bytes = _json(
        receipt_path,
        "conflict authorization receipt",
        limit=_CONFLICT_AUTH_MAX_JSON,
    )
    receipt_info = os.lstat(receipt_path)
    if receipt_info.st_uid != os.getuid() or stat.S_IMODE(receipt_info.st_mode) != 0o600:
        raise PublicationError("conflict authorization receipt is not private")
    raw_key = _conflict_public_key(key_path)
    if raw_key != terminal_public_key:
        raise PublicationError("conflict authority must equal terminal authority")
    try:
        authority = Ed25519PublicKey.from_public_bytes(raw_key)
    except ValueError as exc:
        raise PublicationError("conflict authority public key is invalid") from exc
    outer_fields = {"schema", "type", "authority_key_id", "message", "signature"}
    if (
        set(receipt) != outer_fields
        or receipt["schema"] != "alpha_max_canonical_conflict_authorization_receipt.v2"
        or receipt["type"] != "canonical_conflict_authorization"
        or receipt["authority_key_id"] != hashlib.sha256(raw_key).hexdigest()
        or not isinstance(receipt["message"], dict)
        or not isinstance(receipt["signature"], str)
    ):
        raise PublicationError("conflict authorization receipt schema is invalid")
    unsigned = {key: value for key, value in receipt.items() if key != "signature"}
    try:
        authority.verify(
            base64.b64decode(receipt["signature"], validate=True),
            _CONFLICT_AUTHORIZATION_DOMAIN + canonical_bytes(unsigned),
        )
    except (ValueError, InvalidSignature) as exc:
        raise PublicationError("conflict authorization signature is invalid") from exc
    message = receipt["message"]
    message_fields = {
        "schema",
        "scope",
        "decision",
        "canonical_root",
        "acquisition_request_id",
        "terminal_receipt_sha256",
        "source_manifest_sha256",
        "source_eligible_receipt_sha256",
        "predecessor_path",
        "predecessor_identity",
        "predecessor_inventory_sha256",
        "fresh_acquisition_audit_receipt_sha256",
        "acquisition_run_id",
        "composite_telemetry_sha256",
        "wal_transition_receipt_sha256",
        "wal_post_transition_inventory_sha256",
        "signed_terminal_request_sha256",
        "approval_sha256",
        "entries",
    }
    if (
        set(message) != message_fields
        or message["schema"] != "alpha_max_canonical_conflict_authorization_message.v2"
        or message["scope"] != "canonical_conflict_reconciliation"
        or message["decision"] != "approve_exact_effects"
        or message["canonical_root"] != str(args.canonical_root)
        or message["acquisition_request_id"] != request_id
        or message["terminal_receipt_sha256"] != terminal_sha256
        or message["source_manifest_sha256"] != hashes["manifest"]
        or message["source_eligible_receipt_sha256"] != hashes["eligible"]
        or message["predecessor_path"] != str(predecessor)
        or message["predecessor_identity"]
        != [os.stat(predecessor).st_dev, os.stat(predecessor).st_ino]
        or message["predecessor_inventory_sha256"] != _inventory_digest(predecessor_inventory)
        or any(message.get(field) != value for field, value in controls.items())
        or not isinstance(message["entries"], list)
    ):
        raise PublicationError("conflict authorization message binding is invalid")
    eligible = {item["relative"]: item for item in parts}
    entry_fields = {
        "relative",
        "source_sha256",
        "source_byte_count",
        "source_row_count",
        "provenance_receipt_sha256",
        "predecessor_identity",
        "predecessor_sha256",
        "predecessor_byte_count",
        "predecessor_row_count",
        "effects",
    }
    entries: dict[str, dict[str, Any]] = {}
    for entry in message["entries"]:
        if not isinstance(entry, dict) or set(entry) != entry_fields:
            raise PublicationError("conflict authorization entry schema is invalid")
        relative = entry["relative"]
        if (
            not _authorized_ohlcv_relative(relative)
            or relative in entries
            or relative not in eligible
        ):
            raise PublicationError("conflict authorization entry path is invalid")
        part = eligible[relative]
        if (
            entry["source_sha256"] != part["sha256"]
            or entry["source_byte_count"] != os.stat(part["source"]).st_size
            or entry["source_row_count"] != part["rows"]
            or entry["provenance_receipt_sha256"] != part["provenance"]
            or any(
                isinstance(entry[field], bool)
                or not isinstance(entry[field], int)
                or entry[field] < 0
                for field in (
                    "source_byte_count",
                    "source_row_count",
                    "predecessor_byte_count",
                    "predecessor_row_count",
                )
            )
            or not isinstance(entry["predecessor_identity"], list)
            or len(entry["predecessor_identity"]) != 9
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in entry["predecessor_identity"]
            )
            or not isinstance(entry["effects"], dict)
            or set(entry["effects"])
            != {
                "conflict_rows",
                "conflict_sha256",
                "canonical_only_rows",
                "canonical_only_sha256",
                "source_only_rows",
                "source_only_sha256",
            }
            or any(
                isinstance(entry["effects"][field], bool)
                or not isinstance(entry["effects"][field], int)
                or entry["effects"][field] < 0
                for field in ("conflict_rows", "canonical_only_rows", "source_only_rows")
            )
            or entry["effects"]["conflict_rows"] == 0
        ):
            raise PublicationError("conflict authorization source binding is invalid")
        for field in (
            "source_sha256",
            "provenance_receipt_sha256",
            "predecessor_sha256",
        ):
            _sha(entry[field], f"conflict authorization {field}")
        for field in (
            "conflict_sha256",
            "canonical_only_sha256",
            "source_only_sha256",
        ):
            _sha(entry["effects"][field], f"conflict authorization effects {field}")
        target = _inside(predecessor, relative, "authorization predecessor")
        size, digest = _file_sha(target)
        target_info = os.stat(target)
        actual_identity = list(_identity(target_info))
        expected_identity = entry["predecessor_identity"]
        identity_matches = expected_identity == actual_identity or (
            allow_predecessor_ctime_transition
            and expected_identity[:5] == actual_identity[:5]
            and actual_identity[5] >= expected_identity[5]
            and expected_identity[6:] == actual_identity[6:]
        )
        if (
            not identity_matches
            or entry["predecessor_sha256"] != digest
            or entry["predecessor_byte_count"] != size
            or entry["predecessor_row_count"] != _parquet_rows(target)
        ):
            raise PublicationError("conflict authorization predecessor binding is invalid")
        entries[relative] = entry
    if not entries or list(entries) != sorted(entries):
        raise PublicationError(
            "conflict authorization entries must be nonempty and uniquely sorted"
        )
    return {
        "receipt": receipt,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "key_id": receipt["authority_key_id"],
        "public_key": raw_key,
        "message": message,
        "message_sha256": hashlib.sha256(canonical_bytes(message)).hexdigest(),
        "entries": entries,
    }


def _replay_conflict_authorization(
    args: argparse.Namespace,
    *,
    final: dict[str, Any],
    terminal_sha256: str,
    request_id: str,
    hashes: dict[str, str],
    parts: list[dict[str, Any]],
    terminal: dict[str, Any] | None = None,
    key_id: str = "",
    terminal_public_key: bytes = b"",
) -> None:
    paired = args.conflict_authorization_receipt is not None
    is_signed_publication = final.get("schema") in {
        "alpha_max_canonical_publication_receipt.v3",
        "alpha_max_canonical_publication_receipt.v4",
    }
    if not is_signed_publication:
        if paired:
            raise PublicationError("V2 publication cannot accept conflict authorization")
        return
    if not paired:
        raise PublicationError(
            "signed publication replay requires conflict authorization receipt and public key"
        )
    predecessor = Path(final["predecessor_path"])
    controls = _conflict_controls(
        args,
        terminal=terminal or {},
        approval_sha256=final.get("approval_sha256"),
        terminal_sha256=terminal_sha256,
        request_id=request_id,
        key_id=key_id,
        predecessor_inventory=final["old_inventory"],
    )
    if controls is None:
        raise PublicationError(
            "signed publication replay requires fresh audit and WAL transition controls"
        )
    predecessor_available = (
        predecessor.exists()
        and not predecessor.is_symlink()
        and [os.stat(predecessor).st_dev, os.stat(predecessor).st_ino]
        == final["predecessor_identity"]
    )
    if not predecessor_available:
        receipt, receipt_bytes = _json(
            args.conflict_authorization_receipt,
            "conflict authorization receipt",
            limit=_CONFLICT_AUTH_MAX_JSON,
        )
        receipt_info = os.lstat(args.conflict_authorization_receipt)
        raw_key = _conflict_public_key(args.conflict_authority_public_key)
        if raw_key != _conflict_public_key(args.authority_public_key):
            raise PublicationError("replay conflict authority must equal terminal authority")
        unsigned = {key: value for key, value in receipt.items() if key != "signature"}
        try:
            Ed25519PublicKey.from_public_bytes(raw_key).verify(
                base64.b64decode(receipt["signature"], validate=True),
                _CONFLICT_AUTHORIZATION_DOMAIN + canonical_bytes(unsigned),
            )
        except (KeyError, TypeError, ValueError, InvalidSignature) as exc:
            raise PublicationError("replay conflict authorization signature is invalid") from exc
        message = receipt.get("message")
        message_entries = message.get("entries") if isinstance(message, dict) else None
        eligible = {item["relative"]: item for item in parts}
        entry_fields = {
            "relative",
            "source_sha256",
            "source_byte_count",
            "source_row_count",
            "provenance_receipt_sha256",
            "predecessor_identity",
            "predecessor_sha256",
            "predecessor_byte_count",
            "predecessor_row_count",
            "effects",
        }
        entries_valid = isinstance(message_entries, list) and all(
            isinstance(entry, dict)
            and set(entry) == entry_fields
            and _authorized_ohlcv_relative(entry.get("relative"))
            and entry["relative"] in eligible
            and entry["source_sha256"] == eligible[entry["relative"]]["sha256"]
            and entry["source_byte_count"] == os.stat(eligible[entry["relative"]]["source"]).st_size
            and entry["source_row_count"] == eligible[entry["relative"]]["rows"]
            and entry["provenance_receipt_sha256"] == eligible[entry["relative"]]["provenance"]
            and all(
                type(entry[field]) is int and entry[field] >= 0
                for field in (
                    "source_byte_count",
                    "source_row_count",
                    "predecessor_byte_count",
                    "predecessor_row_count",
                )
            )
            and isinstance(entry["predecessor_identity"], list)
            and len(entry["predecessor_identity"]) == 9
            and all(type(value) is int for value in entry["predecessor_identity"])
            and all(
                isinstance(entry[field], str)
                and re.fullmatch(r"[0-9a-f]{64}", entry[field]) is not None
                for field in (
                    "source_sha256",
                    "provenance_receipt_sha256",
                    "predecessor_sha256",
                )
            )
            and isinstance(entry["effects"], dict)
            and set(entry["effects"])
            == {
                "conflict_rows",
                "conflict_sha256",
                "canonical_only_rows",
                "canonical_only_sha256",
                "source_only_rows",
                "source_only_sha256",
            }
            and all(
                type(entry["effects"][field]) is int and entry["effects"][field] >= 0
                for field in ("conflict_rows", "canonical_only_rows", "source_only_rows")
            )
            and entry["effects"]["conflict_rows"] > 0
            and all(
                isinstance(entry["effects"][field], str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["effects"][field]) is not None
                for field in (
                    "conflict_sha256",
                    "canonical_only_sha256",
                    "source_only_sha256",
                )
            )
            for entry in message_entries or []
        )
        entry_relatives = [entry["relative"] for entry in message_entries] if entries_valid else []
        if (
            receipt_info.st_uid != os.getuid()
            or stat.S_IMODE(receipt_info.st_mode) != 0o600
            or set(receipt) != {"schema", "type", "authority_key_id", "message", "signature"}
            or receipt.get("schema") != "alpha_max_canonical_conflict_authorization_receipt.v2"
            or receipt.get("type") != "canonical_conflict_authorization"
            or receipt.get("authority_key_id") != hashlib.sha256(raw_key).hexdigest()
            or not isinstance(message, dict)
            or set(message)
            != {
                "schema",
                "scope",
                "decision",
                "canonical_root",
                "acquisition_request_id",
                "terminal_receipt_sha256",
                "source_manifest_sha256",
                "source_eligible_receipt_sha256",
                "predecessor_path",
                "predecessor_identity",
                "predecessor_inventory_sha256",
                "fresh_acquisition_audit_receipt_sha256",
                "acquisition_run_id",
                "composite_telemetry_sha256",
                "wal_transition_receipt_sha256",
                "wal_post_transition_inventory_sha256",
                "signed_terminal_request_sha256",
                "approval_sha256",
                "entries",
            }
            or message.get("schema") != "alpha_max_canonical_conflict_authorization_message.v2"
            or message.get("scope") != "canonical_conflict_reconciliation"
            or message.get("decision") != "approve_exact_effects"
            or message.get("canonical_root") != str(args.canonical_root)
            or message.get("acquisition_request_id") != request_id
            or message.get("terminal_receipt_sha256") != terminal_sha256
            or message.get("source_manifest_sha256") != hashes["manifest"]
            or message.get("source_eligible_receipt_sha256") != hashes["eligible"]
            or message.get("predecessor_path") != str(predecessor)
            or message.get("predecessor_identity") != final["predecessor_identity"]
            or message.get("predecessor_inventory_sha256") != final["old_inventory_sha256"]
            or not entries_valid
            or not entry_relatives
            or entry_relatives != sorted(set(entry_relatives))
            or hashlib.sha256(receipt_bytes).hexdigest()
            != final["conflict_authorization_receipt_sha256"]
            or receipt["authority_key_id"] != final["conflict_authority_key_id"]
            or any(message.get(field) != controls[field] for field in controls)
            or any(final.get(field) != controls[field] for field in controls)
            or receipt["authority_key_id"] != key_id
            or any(
                message.get(field) != final.get(field)
                for field in (
                    "fresh_acquisition_audit_receipt_sha256",
                    "acquisition_run_id",
                    "composite_telemetry_sha256",
                    "wal_transition_receipt_sha256",
                    "wal_post_transition_inventory_sha256",
                    "signed_terminal_request_sha256",
                    "approval_sha256",
                )
            )
            or hashlib.sha256(canonical_bytes(message)).hexdigest()
            != final["conflict_authorization_message_sha256"]
            or message["predecessor_inventory_sha256"]
            != final["conflict_authorization_predecessor_inventory_sha256"]
        ):
            raise PublicationError("replay conflict authorization binding mismatch")
        return
    authorization = _conflict_authorization(
        args,
        terminal_sha256=terminal_sha256,
        request_id=request_id,
        hashes=hashes,
        predecessor=predecessor,
        predecessor_inventory=final["old_inventory"],
        controls=controls,
        terminal_public_key=terminal_public_key,
        parts=parts,
        allow_predecessor_ctime_transition=True,
    )
    if authorization is None or (
        authorization["receipt_sha256"] != final["conflict_authorization_receipt_sha256"]
        or authorization["key_id"] != final["conflict_authority_key_id"]
        or authorization["message_sha256"] != final["conflict_authorization_message_sha256"]
        or authorization["message"]["predecessor_inventory_sha256"]
        != final["conflict_authorization_predecessor_inventory_sha256"]
        or any(authorization["message"].get(field) != controls[field] for field in controls)
        or any(final.get(field) != controls[field] for field in controls)
    ):
        raise PublicationError("replay conflict authorization binding mismatch")


def _file_binding(path: Path, label: str, expected: tuple[str, int]) -> bytes:
    raw = _regular_bytes(path, label)
    if len(raw) != expected[1] or hashlib.sha256(raw).hexdigest() != expected[0]:
        raise PublicationError(f"{label} does not match terminal authorization")
    return raw


def _partitions(
    source: Path, report: Path, bound: dict[str, tuple[str, int]]
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    manifest, manifest_bytes = _json(report / "source_manifest.json", "source manifest")
    receipt, receipt_bytes = _json(report / "source_eligible_receipt.json", "eligible receipt")
    journal = _file_binding(
        report / "acquisition.journal.jsonl", "source journal", bound["source_journal"]
    )
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != bound["source_manifest"][0]
        or len(manifest_bytes) != bound["source_manifest"][1]
        or hashlib.sha256(receipt_bytes).hexdigest() != bound["source_eligible_receipt"][0]
        or len(receipt_bytes) != bound["source_eligible_receipt"][1]
    ):
        raise PublicationError("terminal source artifact binding mismatch")
    if (
        manifest.get("schema") != "alpha_max_official_source_manifest.v5"
        or receipt.get("schema") != "alpha_max_official_source_receipt.v4"
        or receipt.get("source_eligible") is not True
    ):
        raise PublicationError("source eligibility schema is invalid")
    hashes = {
        "manifest": hashlib.sha256(manifest_bytes).hexdigest(),
        "eligible": hashlib.sha256(receipt_bytes).hexdigest(),
        "journal": hashlib.sha256(journal).hexdigest(),
    }
    if (
        receipt.get("source_manifest_sha256") != hashes["manifest"]
        or receipt.get("acquisition_journal_sha256") != hashes["journal"]
    ):
        raise PublicationError("eligible receipt chain mismatch")
    if (
        manifest.get("storage_contract") != _STORAGE
        or receipt.get("storage_contract") != _STORAGE
        or not isinstance(manifest.get("archive_evidence_sha256"), str)
        or manifest.get("archive_evidence_sha256") != receipt.get("archive_evidence_sha256")
    ):
        raise PublicationError("source storage contract or archive evidence is invalid")
    contract, contract_bytes = _json(
        report / "provenance" / "contract_manifest.json", "contract manifest"
    )
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    if (
        manifest.get("contract_sha256") != contract_sha
        or contract.get("schema_version") != "alpha_max_contract_manifest.v2"
        or contract.get("exchange") != "binance"
    ):
        raise PublicationError("contract manifest binding invalid")
    records = contract.get("records")
    if not isinstance(records, list) or not records:
        raise PublicationError("contract listing records invalid")
    listing: list[dict[str, Any]] = []
    symbols: set[str] = set()
    required = {
        "symbol",
        "market_type",
        "quote_asset",
        "margin_asset",
        "settle_asset",
        "linear",
        "inverse",
        "contract_multiplier",
        "volume_unit",
        "raw_availability_start_utc",
        "raw_availability_end_utc",
        "feature_availability_start_utc",
        "feature_availability_end_utc",
    }
    for item in records:
        if (
            not isinstance(item, dict)
            or set(item) != required
            or not isinstance(item["symbol"], str)
            or item["symbol"] in symbols
        ):
            raise PublicationError("contract listing record invalid")
        text_fields = required - {"linear", "inverse", "contract_multiplier"}
        if (
            any(not isinstance(item[k], str) or not item[k] for k in text_fields)
            or not isinstance(item["linear"], bool)
            or not isinstance(item["inverse"], bool)
            or isinstance(item["contract_multiplier"], bool)
            or not isinstance(item["contract_multiplier"], (int, float))
        ):
            raise PublicationError("contract listing record invalid")
        symbols.add(item["symbol"])
        listing.append({k: item[k] for k in sorted(required)})
    if [x["symbol"] for x in listing] != sorted(symbols):
        raise PublicationError("contract listing records must be uniquely ordered")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PublicationError("manifest artifacts invalid")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item["path"], str)
            or not isinstance(item["sha256"], str)
        ):
            raise PublicationError("manifest artifact shape is invalid")
        path = item["path"]
        _sha(item["sha256"], "manifest artifact hash")
        if path in seen:
            raise PublicationError("duplicate manifest artifact")
        seen.add(path)
        if not path.startswith("output/"):
            continue
        relative = path[7:]
        parts = PurePosixPath(relative).parts
        ohlcv = (
            len(parts) == 4
            and parts[:2] == ("market_ohlcv_1s", "binance")
            and parts[3].endswith(".parquet")
        )
        funding = (
            len(parts) == 5
            and parts[:2] == ("feature_points", "exchange=binance")
            and parts[2].startswith("symbol=")
            and parts[3].startswith("date=")
            and parts[4] == "funding.parquet"
        )
        if not (ohlcv or funding):
            raise PublicationError("manifest output artifact has an unrecognized shape")
        symbol = parts[2] if ohlcv else parts[2][7:]
        if symbol not in symbols:
            raise PublicationError("manifest output symbol is absent from contract listing")
        partition, partition_bytes = _json(_receipt_path(report, relative), "partition receipt")
        rows = _positive_int(partition.get("rows"), "partition row count")
        if rows <= 0:
            raise PublicationError("partition row count must be positive")
        if (
            partition.get("schema") != "alpha_max_partition_receipt.v2"
            or partition.get("path") != relative
            or partition.get("output_sha256") != item["sha256"]
        ):
            raise PublicationError("partition receipt mismatch")
        output.append(
            {
                "relative": relative,
                "source": _inside(source, relative, "source output"),
                "sha256": item["sha256"],
                "rows": rows,
                "provenance": hashlib.sha256(partition_bytes).hexdigest(),
            }
        )
    contract_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.get("path") == "report/provenance/contract_manifest.json"
    ]
    if len(contract_artifacts) != 1 or contract_artifacts[0].get("sha256") != contract_sha:
        raise PublicationError("contract manifest is not authenticated by source manifest")
    if not output:
        raise PublicationError("no eligible partitions")
    return (
        sorted(output, key=lambda x: x["relative"]),
        {**hashes, "contract": contract_sha},
        listing,
    )


def _hash_source(
    path: Path, expected: str
) -> tuple[int, tuple[int, int, int, int, int, int, int, int, int]]:
    fd, identity = _private_regular(path, "source output")
    try:
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        _revalidate(path, fd, identity, "source output")
        if digest.hexdigest() != expected:
            raise PublicationError("source output hash mismatch")
        return identity[3], identity
    finally:
        os.close(fd)


def _pin_source(
    source: Path,
    destination: Path,
    expected_sha256: str,
) -> tuple[int, tuple[int, int, int, int, int, int, int, int, int]]:
    source_fd, source_identity = _private_regular(source, "source output")
    _ensure_private_directory(destination.parent)
    try:
        try:
            destination_fd = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o400,
            )
        except FileExistsError:
            size, digest = _file_sha(destination)
            if size != source_identity[3] or digest != expected_sha256:
                raise PublicationError("immutable source pin conflicts")
        else:
            digest = hashlib.sha256()
            try:
                while chunk := os.read(source_fd, 1024 * 1024):
                    digest.update(chunk)
                    _write_all(destination_fd, chunk)
                os.fchmod(destination_fd, 0o400)
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
            if digest.hexdigest() != expected_sha256:
                destination.unlink()
                _fsync_dir(destination.parent)
                raise PublicationError("source output hash mismatch while pinning")
            _fsync_dir(destination.parent)
        _revalidate(source, source_fd, source_identity, "source output")
        pinned_fd, pinned_identity = _private_regular(destination, "immutable source pin")
        try:
            if (
                pinned_identity[3] != source_identity[3]
                or stat.S_IMODE(pinned_identity[2]) != 0o400
                or _file_sha(destination) != (source_identity[3], expected_sha256)
            ):
                raise PublicationError("immutable source pin readback mismatch")
        finally:
            os.close(pinned_fd)
        return source_identity[3], source_identity
    finally:
        os.close(source_fd)


def _file_sha(path: Path) -> tuple[int, str]:
    fd, identity = _private_regular(path, "published target")
    try:
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        _revalidate(path, fd, identity, "published target")
        return identity[3], digest.hexdigest()
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _inventory(root: Path, *, require_private: bool = True) -> list[dict[str, Any]]:
    """Return a complete, identity-bound, symlink-free tree inventory."""
    records: list[dict[str, Any]] = []

    def visit(current: Path, relative: str) -> None:
        for entry in sorted(os.scandir(current), key=lambda item: item.name):
            child = current / entry.name
            name = f"{relative}/{entry.name}" if relative else entry.name
            info = entry.stat(follow_symlinks=False)
            common = {
                "path": name,
                "mode": stat.S_IMODE(info.st_mode),
                "dev": info.st_dev,
                "ino": info.st_ino,
                "ctime_ns": info.st_ctime_ns,
            }
            if stat.S_ISDIR(info.st_mode):
                records.append({**common, "kind": "dir"})
                visit(child, name)
            elif stat.S_ISREG(info.st_mode):
                if require_private and info.st_nlink != 1:
                    raise PublicationError(f"canonical root contains hardlinked file: {name}")
                records.append(
                    {
                        **common,
                        "kind": "file",
                        "size": info.st_size,
                        "mtime_ns": info.st_mtime_ns,
                        "nlink": info.st_nlink,
                    }
                )
            else:
                raise PublicationError(f"canonical root contains unsafe artifact: {name}")

    info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode):
        raise PublicationError("canonical root is not a directory")
    visit(root, "")
    return records


def _inventory_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def _data_inventory_digest(root: Path) -> str:
    """Digest immutable candidate data while excluding publication controls."""
    return _inventory_digest(alpha_max_canonical_inventory_records(root))


def _canonical_inventory(root: Path) -> dict[str, Any]:
    """Bind every immutable data-bearing dataset, including untouched partitions."""
    records = alpha_max_canonical_inventory_records(root)
    datasets: dict[str, dict[str, int]] = {}
    for record in records:
        if record["kind"] == "file":
            dataset = PurePosixPath(record["path"]).parts[0]
            totals = datasets.setdefault(dataset, {"files": 0, "bytes": 0, "rows": 0})
            totals["files"] += 1
            totals["bytes"] += record["size"]
            if record["path"].endswith(".parquet"):
                totals["rows"] += _parquet_rows(root / record["path"])
    return {
        "inventory": records,
        "inventory_sha256": _inventory_digest(records),
        "datasets": [{"dataset": name, **datasets[name]} for name in sorted(datasets)],
        "files": sum(item["files"] for item in datasets.values()),
        "bytes": sum(item["bytes"] for item in datasets.values()),
        "rows": sum(item["rows"] for item in datasets.values()),
    }


def _merged_target_effects(
    predecessor: Path | None, source: Path, target: Path, *, funding: bool
) -> dict[str, int]:
    """Derive key-union effects independently of the repository write path."""
    key = "timestamp_ms" if funding else "datetime"
    incoming = pl.read_parquet(source)
    existing = (
        pl.read_parquet(predecessor)
        if predecessor is not None and predecessor.exists()
        else pl.DataFrame(schema={key: incoming.schema[key]})
    )
    candidate = pl.read_parquet(target)
    if (
        incoming.select(key).n_unique() != incoming.height
        or existing.select(key).n_unique() != existing.height
    ):
        raise PublicationError("partition input keys are not unique")
    overlap = existing.join(incoming, on=key, how="inner", suffix="_source")
    if funding:
        conflict = overlap.filter(
            pl.col("funding_rate").is_not_null()
            & pl.col("funding_rate_source").is_not_null()
            & (pl.col("funding_rate") != pl.col("funding_rate_source"))
        ).height
    else:
        conflict = overlap.filter(
            pl.any_horizontal(
                [
                    pl.col(column) != pl.col(f"{column}_source")
                    for column in ("open", "high", "low", "close", "volume")
                ]
            )
        ).height
    source_only = incoming.join(existing.select(key), on=key, how="anti").height
    canonical_only = existing.join(incoming.select(key), on=key, how="anti").height
    equal = overlap.height - conflict
    effects = {
        "predecessor_rows": existing.height,
        "source_rows": incoming.height,
        "equal_rows": equal,
        "conflict_rows": conflict,
        "canonical_only_rows": canonical_only,
        "source_only_rows": source_only,
        "target_rows": candidate.height,
    }
    if (
        effects["target_rows"] != effects["source_rows"] + effects["canonical_only_rows"]
        or effects["target_rows"] != effects["predecessor_rows"] + effects["source_only_rows"]
    ):
        raise PublicationError("merged target key-union accounting mismatch")
    return effects


def _clone_root(old: Path, candidate: Path, expected: list[dict[str, Any]]) -> None:
    """Hardlink-clone data, omit transient locks, and prove both bound trees."""
    candidate.mkdir(mode=0o700)
    _fsync_dir(candidate.parent)

    def clone(source: Path, destination: Path) -> None:
        for entry in sorted(os.scandir(source), key=lambda item: item.name):
            src, dst = source / entry.name, destination / entry.name
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                dst.mkdir(mode=stat.S_IMODE(info.st_mode))
                _fsync_dir(destination)
                clone(src, dst)
                _fsync_dir(dst)
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise PublicationError(f"canonical root contains hardlinked file: {src}")
                if not _is_coordination_lock(str(src.relative_to(old))):
                    os.link(src, dst, follow_symlinks=False)
            else:
                raise PublicationError(f"canonical root contains unsafe artifact: {src}")

    clone(old, candidate)
    _fsync_dir(candidate)
    old_after = _inventory(old, require_private=False)
    candidate_after = _inventory(candidate, require_private=False)
    expected_by_path = {record["path"]: record for record in expected}
    expected_candidate_paths = {
        path
        for path, record in expected_by_path.items()
        if record["kind"] == "dir" or not _is_coordination_lock(path)
    }
    if set(expected_by_path) != {
        record["path"] for record in old_after
    } or expected_candidate_paths != {record["path"] for record in candidate_after}:
        raise PublicationError("clone tree has missing or extra entries")
    for record in old_after:
        bound = expected_by_path[record["path"]]
        if record["kind"] != bound["kind"] or (record["dev"], record["ino"]) != (
            bound["dev"],
            bound["ino"],
        ):
            raise PublicationError("canonical root changed while cloning")
        if record["kind"] == "file":
            expected_nlink = (
                bound["nlink"] if _is_coordination_lock(record["path"]) else bound["nlink"] + 1
            )
            if record["nlink"] != expected_nlink:
                raise PublicationError("canonical root hardlink transition changed while cloning")
    for record in candidate_after:
        bound = expected_by_path[record["path"]]
        if record["kind"] != bound["kind"] or record["mode"] != bound["mode"]:
            raise PublicationError("candidate tree metadata changed while cloning")
        if record["kind"] == "file":
            if (record["dev"], record["ino"]) != (bound["dev"], bound["ino"]):
                raise PublicationError("candidate hardlink does not share source inode")
            if record["nlink"] != bound["nlink"] + 1:
                raise PublicationError("candidate hardlink transition is invalid")


def _rename_exchange(first: Path, second: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(first), -100, os.fsencode(second), 2) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(first))


def _remove_bound_tree(
    root: Path, identity: tuple[int, int], expected: list[dict[str, Any]]
) -> None:
    info = os.lstat(root)
    if (info.st_dev, info.st_ino) != identity or not stat.S_ISDIR(info.st_mode):
        raise PublicationError("retired root identity changed")
    actual = _inventory(root, require_private=False)
    bound = {item["path"]: item for item in expected}
    actual_paths = {item["path"] for item in actual}
    if not actual_paths.issubset(bound):
        raise PublicationError("retired root has unbound entries")
    for item in actual:
        prior = bound[item["path"]]
        if (
            item["kind"] != prior["kind"]
            or item["mode"] != prior["mode"]
            or (item["dev"], item["ino"]) != (prior["dev"], prior["ino"])
            or (
                item["kind"] == "file"
                and (item["size"] != prior["size"] or item["mtime_ns"] != prior["mtime_ns"])
            )
        ):
            raise PublicationError("retired root entry identity changed")

    def remove(current: Path) -> None:
        for entry in os.scandir(current):
            path = current / entry.name
            entry_info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(entry_info.st_mode):
                remove(path)
            elif stat.S_ISREG(entry_info.st_mode):
                path.unlink()
            else:
                raise PublicationError("retired root contains unsafe artifact")
        current.rmdir()

    remove(root)


def _active_root(logical: Path, generations: Path) -> Path:
    info = os.lstat(logical)
    if stat.S_ISDIR(info.st_mode):
        return logical
    if not stat.S_ISLNK(info.st_mode):
        raise PublicationError("canonical root must be a directory or generation symlink")
    target = os.readlink(logical)
    parts = Path(target).parts
    if (
        os.path.isabs(target)
        or len(parts) != 2
        or parts[0] != generations.name
        or parts[1] in {"", ".", ".."}
    ):
        raise PublicationError("canonical root symlink is untrusted")
    physical = logical.parent / target
    if physical.parent != generations or not stat.S_ISDIR(os.lstat(physical).st_mode):
        raise PublicationError("canonical root symlink target is untrusted")
    return physical


def _topology_identity(path: Path) -> list[Any]:
    """Stable identity for directory/symlink exchange topology."""
    info = os.lstat(path)
    if stat.S_ISDIR(info.st_mode):
        return [info.st_dev, info.st_ino, "directory"]
    if stat.S_ISLNK(info.st_mode):
        return [info.st_dev, info.st_ino, "symlink", os.readlink(path)]
    raise PublicationError("exchange topology path is unsafe")


def _window_bindings(
    *,
    request_id: str,
    key_id: str,
    terminal_sha256: str,
    candidate: Path,
    final: dict[str, Any],
    swap: dict[str, Any],
    temporary: Path,
) -> dict[str, Any]:
    predecessor = _retained_predecessor(Path(final["canonical_root"]), final)
    canonical_root = Path(final["canonical_root"])
    swap_bytes = canonical_bytes(swap)
    return {
        "request_id": request_id,
        "run_id": _CONFLICT_RUN_ID,
        "acquisition_request_id": _ACQUISITION_REQUEST_ID,
        "approval_leaf": _CURRENT_APPROVAL_LEAF,
        "approval_sha256": final.get("approval_sha256"),
        "authority_key_id": key_id,
        "terminal_receipt_sha256": terminal_sha256,
        "observer_key_id": final.get("observer_key_id"),
        "observer_ready_sha256": final.get("observer_ready_sha256"),
        "observer_query_spec_sha256": final.get("observer_query_spec_sha256"),
        "candidate": str(candidate),
        "candidate_leaf": candidate.name,
        "predecessor": str(predecessor),
        "swap": temporary.name,
        "swap_receipt_sha256": hashlib.sha256(swap_bytes).hexdigest(),
        "swap_temporary_path": str(temporary),
        "candidate_identity": _topology_identity(candidate),
        "predecessor_identity": _topology_identity(predecessor),
        "swap_identity": _topology_identity(temporary),
        "pre_exchange_predecessor_identity": [
            final["predecessor_identity"][0],
            final["predecessor_identity"][1],
            "directory",
        ],
        "post_exchange_candidate_identity": _topology_identity(candidate),
        "post_exchange_predecessor_identity": _topology_identity(predecessor),
        "canonical_logical_root_identity": _topology_identity(canonical_root),
        "canonical_resolved_root": str(
            _active_root(
                canonical_root, canonical_root.parent / f".{canonical_root.name}.generations"
            )
        ),
        "candidate_inventory_sha256": _canonical_inventory(candidate)["inventory_sha256"],
        "predecessor_inventory_sha256": _canonical_inventory(predecessor)["inventory_sha256"],
    }


def _retained_predecessor(canonical_root: Path, final: dict[str, Any]) -> Path:
    expected = final["predecessor_identity"]
    predecessor = Path(final["predecessor_path"])
    if predecessor.exists() and not predecessor.is_symlink():
        info = os.lstat(predecessor)
        if [info.st_dev, info.st_ino] == expected:
            return predecessor
    swap_path = (
        canonical_root.parent
        / f".{canonical_root.name}.transactions"
        / final["request_id"]
        / "swap.json"
    )
    if swap_path.exists():
        swap, _ = _json(swap_path, "swap receipt")
        temporary = canonical_root.parent / swap.get("temporary", "")
        if temporary.exists() and not temporary.is_symlink():
            info = os.lstat(temporary)
            if [info.st_dev, info.st_ino] == expected:
                return temporary
    raise PublicationError("retained predecessor identity is unavailable")


def _write_open_window_receipts(
    *,
    control: Path,
    publisher_key: Ed25519PrivateKey,
    request_id: str,
    key_id: str,
    terminal_sha256: str,
    candidate: Path,
    final: dict[str, Any],
    swap: dict[str, Any],
    temporary: Path,
    replay: bool,
) -> None:
    bindings = _window_bindings(
        request_id=request_id,
        key_id=key_id,
        terminal_sha256=terminal_sha256,
        candidate=candidate,
        final=final,
        swap=swap,
        temporary=temporary,
    )
    _write_noreplace(
        control / "activated.json",
        _stage_envelope(
            publisher_key,
            "activation",
            {"schema": "alpha_max_publication_activation.v1", "phase": "activated", **bindings},
        ),
    )
    _write_noreplace(
        control / "rollback-window-open.json",
        _stage_envelope(
            publisher_key,
            "open_window",
            {"schema": "alpha_max_publication_rollback_window.v1", "phase": "open", **bindings},
        ),
    )
    if replay:
        _write_noreplace(
            control / "replay-verified.json",
            _stage_envelope(
                publisher_key,
                "replay",
                {"schema": "alpha_max_publication_replay.v1", "phase": "pass", **bindings},
            ),
        )


def _parquet_rows(path: Path) -> int:
    try:
        return int(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item())
    except Exception as exc:
        raise PublicationError(f"published target is not readable parquet: {path}") from exc


def _parquet_materialized_bytes(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += int(pl.read_parquet(path).estimated_size())
        except Exception as exc:
            raise PublicationError(
                f"capacity audit cannot materialize parquet metadata: {path}"
            ) from exc
    return total


def _host_free_bytes() -> int:
    usage = os.statvfs(_STORAGE["host_reserve_path"])
    return usage.f_bavail * usage.f_frsize


def _publication_admission(
    candidate: Path, item: dict[str, Any], source_bytes: int
) -> dict[str, int]:
    """Bound one merge without crediting retained old, staging, or rollback data."""
    path = PurePosixPath(item["relative"])
    existing_paths: list[Path]
    if path.parts[0] == "market_ohlcv_1s":
        target = candidate.joinpath(*path.parts)
        existing_paths = [target] if target.exists() else []
    else:
        date_root = candidate.joinpath(*path.parts[:-1])
        existing_paths = sorted(entry for entry in date_root.glob("*.parquet") if entry.is_file())
    existing_bytes = sum(entry.stat().st_size for entry in existing_paths)
    materialized_inputs_bytes = _parquet_materialized_bytes([*existing_paths, Path(item["source"])])
    merged_output_upper_bound = (
        max(existing_bytes + source_bytes, materialized_inputs_bytes)
        + _PARQUET_ENCODING_RESERVE_BYTES
    )
    required_increment = (
        source_bytes + merged_output_upper_bound + _PUBLICATION_CONTROL_RESERVE_BYTES
    )
    free_bytes = _host_free_bytes()
    required_free_bytes = _STORAGE["host_reserve_bytes"] + required_increment
    admission = {
        "free_bytes": free_bytes,
        "host_reserve_bytes": _STORAGE["host_reserve_bytes"],
        "source_staging_bytes_preserved": source_bytes,
        "immutable_source_pin_bytes": source_bytes,
        "existing_target_bytes_retained": existing_bytes,
        "materialized_inputs_bytes": materialized_inputs_bytes,
        "parquet_encoding_reserve_bytes": _PARQUET_ENCODING_RESERVE_BYTES,
        "temporary_merged_output_upper_bound": merged_output_upper_bound,
        "enforced_output_quota_bytes": merged_output_upper_bound,
        "control_fsync_rollback_reserve_bytes": _PUBLICATION_CONTROL_RESERVE_BYTES,
        "required_increment_bytes": required_increment,
        "required_free_bytes": required_free_bytes,
    }
    if free_bytes <= required_free_bytes:
        raise PublicationError("host reserve cannot guarantee publication peak")
    return admission


def _validate_prepared_commit(
    candidate: Path,
    final: dict[str, Any],
    *,
    request_id: str,
    terminal_sha256: str,
    key_id: str,
    hashes: dict[str, str],
    listing: list[dict[str, Any]],
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_final_fields = {
        "schema",
        "request_id",
        "terminal_receipt_sha256",
        "authority_key_id",
        "source_manifest_sha256",
        "source_eligible_receipt_sha256",
        "source_journal_sha256",
        "contract_manifest_sha256",
        "listing_records",
        "listing_metadata_sha256",
        "canonical_root",
        "predecessor_identity",
        "predecessor_path",
        "old_inventory",
        "old_inventory_sha256",
        "candidate_data_inventory_sha256",
        "detached_predecessor_paths",
        "capacity_admissions",
        "partitions",
        "ohlcv_partition_count",
        "funding_partition_count",
        "listing_count",
        "rows",
    }
    is_v4 = True
    if final.get("schema") != "alpha_max_canonical_publication_receipt.v4":
        raise PublicationError("prepared commit must use v4 controls")
    if is_v4:
        expected_final_fields |= {
            "conflict_authorization_receipt_sha256",
            "conflict_authority_key_id",
            "conflict_authorization_message_sha256",
            "conflict_authorization_predecessor_inventory_sha256",
            "authorized_conflict_partition_count",
            "authorized_replaced_row_count",
            "fresh_acquisition_audit_receipt_sha256",
            "acquisition_run_id",
            "composite_telemetry_sha256",
            "wal_transition_receipt_sha256",
            "wal_post_transition_inventory_sha256",
            "signed_terminal_request_sha256",
            "approval_sha256",
            "observer_key_id",
            "observer_ready_sha256",
            "observer_query_spec_sha256",
            "source_input",
            "canonical_before",
            "canonical_candidate",
        }
    if set(final) != expected_final_fields:
        raise PublicationError("prepared commit schema mismatch")
    if (
        final.get("request_id") != request_id
        or final.get("terminal_receipt_sha256") != terminal_sha256
        or final.get("authority_key_id") != key_id
        or final.get("source_manifest_sha256") != hashes["manifest"]
        or final.get("source_eligible_receipt_sha256") != hashes["eligible"]
        or final.get("source_journal_sha256") != hashes["journal"]
        or final.get("contract_manifest_sha256") != hashes["contract"]
        or final.get("listing_records") != listing
        or not isinstance(final.get("listing_metadata_sha256"), str)
        or final.get("canonical_root") is None
        or not isinstance(final.get("predecessor_path"), str)
        or not isinstance(final.get("predecessor_identity"), list)
        or len(final["predecessor_identity"]) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in final["predecessor_identity"]
        )
        or not isinstance(final.get("old_inventory"), list)
        or (
            is_v4
            and (
                not isinstance(final.get("approval_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", final["approval_sha256"]) is None
            )
        )
        or final.get("old_inventory_sha256") != _inventory_digest(final["old_inventory"])
    ):
        raise PublicationError("prepared commit binding mismatch")
    canonical_root = Path(final["canonical_root"])
    predecessor = Path(final["predecessor_path"])
    if (
        not canonical_root.is_absolute()
        or not predecessor.is_absolute()
        or candidate != canonical_root.parent / f".{canonical_root.name}.generations" / request_id
        or not (predecessor == canonical_root or predecessor.parent == candidate.parent)
    ):
        raise PublicationError("prepared generation path binding mismatch")
    if is_v4:
        ready, ready_raw = _json(
            canonical_root.parent
            / f".{canonical_root.name}.transactions"
            / request_id
            / "observer-ready.json",
            "publication observer readiness",
        )
        if (
            final.get("observer_key_id") != ready.get("observer_key_id")
            or final.get("observer_ready_sha256") != hashlib.sha256(ready_raw).hexdigest()
            or final.get("observer_query_spec_sha256") != ready.get("query_spec_sha256")
        ):
            raise PublicationError("prepared observer readiness binding mismatch")
    predecessor = _retained_predecessor(canonical_root, final)
    listing_value, listing_bytes = _json(
        candidate / ".alpha_max_publication" / request_id / "listing_records.json",
        "canonical listing metadata",
    )
    if (
        listing_value
        != {
            "schema": "alpha_max_canonical_listing.v1",
            "request_id": request_id,
            "records": listing,
        }
        or hashlib.sha256(listing_bytes).hexdigest() != final["listing_metadata_sha256"]
    ):
        raise PublicationError("prepared listing metadata binding mismatch")
    published = final.get("partitions")
    if not isinstance(published, list) or any(not isinstance(item, dict) for item in published):
        raise PublicationError("prepared commit partition inventory is invalid")
    if (
        final["ohlcv_partition_count"]
        != sum(item.get("relative", "").startswith("market_ohlcv_1s/") for item in published)
        or final["funding_partition_count"]
        != sum(item.get("relative", "").startswith("feature_points/") for item in published)
        or final["listing_count"] != len(listing)
        or final["rows"] != sum(item.get("rows", -1) for item in published)
    ):
        raise PublicationError("prepared commit summary mismatch")
    detached = final.get("detached_predecessor_paths")
    if (
        not isinstance(detached, list)
        or detached != sorted(set(detached))
        or any(not isinstance(path, str) or not path for path in detached)
    ):
        raise PublicationError("prepared predecessor detachment inventory is invalid")
    admissions = final.get("capacity_admissions")
    if not isinstance(admissions, list) or len(admissions) != len(parts):
        raise PublicationError("prepared capacity admission inventory is invalid")
    for admission, part in zip(admissions, parts):
        if (
            not isinstance(admission, dict)
            or admission.get("relative") != part["relative"]
            or any(
                isinstance(admission.get(field), bool)
                or not isinstance(admission.get(field), int)
                or admission[field] < 0
                for field in (
                    "free_bytes",
                    "host_reserve_bytes",
                    "source_staging_bytes_preserved",
                    "immutable_source_pin_bytes",
                    "existing_target_bytes_retained",
                    "materialized_inputs_bytes",
                    "parquet_encoding_reserve_bytes",
                    "temporary_merged_output_upper_bound",
                    "enforced_output_quota_bytes",
                    "control_fsync_rollback_reserve_bytes",
                    "required_increment_bytes",
                    "required_free_bytes",
                )
            )
            or admission["host_reserve_bytes"] != _STORAGE["host_reserve_bytes"]
            or admission["source_staging_bytes_preserved"] != part["source"].stat().st_size
            or admission["immutable_source_pin_bytes"]
            != admission["source_staging_bytes_preserved"]
            or admission["parquet_encoding_reserve_bytes"] != _PARQUET_ENCODING_RESERVE_BYTES
            or admission["control_fsync_rollback_reserve_bytes"]
            != _PUBLICATION_CONTROL_RESERVE_BYTES
            or admission["temporary_merged_output_upper_bound"]
            != max(
                admission["existing_target_bytes_retained"]
                + admission["source_staging_bytes_preserved"],
                admission["materialized_inputs_bytes"],
            )
            + admission["parquet_encoding_reserve_bytes"]
            or admission["enforced_output_quota_bytes"]
            != admission["temporary_merged_output_upper_bound"]
            or admission["required_increment_bytes"]
            != admission["immutable_source_pin_bytes"]
            + admission["temporary_merged_output_upper_bound"]
            + admission["control_fsync_rollback_reserve_bytes"]
            or admission["required_free_bytes"]
            != admission["host_reserve_bytes"] + admission["required_increment_bytes"]
            or admission["free_bytes"] <= admission["required_free_bytes"]
        ):
            raise PublicationError("prepared capacity admission binding mismatch")
    expected = {item["relative"]: item for item in parts}
    if set(expected) != {item.get("relative") for item in published if isinstance(item, dict)}:
        raise PublicationError("prepared commit partition inventory mismatch")
    for record in published:
        expected_record_fields = {
            "relative",
            "target",
            "source_sha256",
            "source_identity",
            "immutable_source_pin",
            "immutable_source_pin_identity",
            "authorized_predecessor_detachments",
            "target_sha256",
            "bytes",
            "target_bytes",
            "rows",
            "target_rows",
            "provenance_receipt_sha256",
        }
        if is_v4:
            expected_record_fields |= {
                "publication_mode",
                "conflict_effects",
                "merged_targets",
            }
        if set(record) != expected_record_fields:
            raise PublicationError("prepared target record shape is invalid")
        source = expected[record["relative"]]
        current_source_bytes, current_source_identity = _hash_source(
            source["source"], source["sha256"]
        )
        if (
            record["source_sha256"] != source["sha256"]
            or record["source_identity"] != list(current_source_identity)
            or current_source_bytes != record["bytes"]
            or record["rows"] != source["rows"]
            or record["provenance_receipt_sha256"] != source["provenance"]
        ):
            raise PublicationError("prepared target source binding mismatch")
        expected_pin = (
            canonical_root.parent
            / f".{canonical_root.name}.transactions"
            / request_id
            / "pinned"
            / (hashlib.sha256(record["relative"].encode()).hexdigest() + ".parquet")
        )
        pin_bytes, pin_identity = _hash_source(expected_pin, source["sha256"])
        if (
            record["immutable_source_pin"] != str(expected_pin)
            or record["immutable_source_pin_identity"] != list(pin_identity)
            or pin_bytes != record["bytes"]
        ):
            raise PublicationError("prepared immutable source pin binding mismatch")
        if record["authorized_predecessor_detachments"] != _authorized_predecessor_detachments(
            record["target"],
            final["old_inventory"],
        ):
            raise PublicationError("prepared predecessor authorization mismatch")
        target = _inside(candidate, record["target"], "prepared target")
        size, digest = _file_sha(target)
        if (
            size != record["target_bytes"]
            or digest != record["target_sha256"]
            or _parquet_rows(target) != record["target_rows"]
        ):
            raise PublicationError("prepared target readback mismatch")
        partition_receipt, _ = _json(
            candidate
            / ".alpha_max_publication"
            / request_id
            / "partitions"
            / (hashlib.sha256(record["relative"].encode()).hexdigest() + ".json"),
            "prepared partition receipt",
        )
        if partition_receipt != record:
            raise PublicationError("prepared partition receipt binding mismatch")
        if is_v4 and (
            record["publication_mode"] not in {"strict_merge", "authorized_reconciliation"}
            or (
                record["publication_mode"] == "strict_merge"
                and record["conflict_effects"] is not None
            )
            or (
                record["publication_mode"] == "authorized_reconciliation"
                and not isinstance(record["conflict_effects"], dict)
            )
            or not isinstance(record["merged_targets"], dict)
        ):
            raise PublicationError("prepared conflict partition binding is invalid")
    canonical_receipt, _ = _json(
        candidate / ".alpha_max_publication" / request_id / "canonical_publication_receipt.json",
        "canonical publication receipt",
    )
    if canonical_receipt != final:
        raise PublicationError("canonical publication receipt binding mismatch")
    if _data_inventory_digest(candidate) != final.get("candidate_data_inventory_sha256"):
        raise PublicationError("prepared candidate data inventory digest mismatch")
    if is_v4:
        source_input = {
            "partitions": len(parts),
            "rows": sum(part["rows"] for part in parts),
            "bytes": sum(record["bytes"] for record in published),
        }
        if final.get("source_input") != source_input:
            raise PublicationError("source input accounting mismatch")
        if final.get("canonical_before") != _canonical_inventory(predecessor):
            raise PublicationError("canonical-before inventory mismatch")
        if final.get("canonical_candidate") != _canonical_inventory(candidate):
            raise PublicationError("canonical-candidate inventory mismatch")
        for record in published:
            predecessor_target = predecessor / record["target"]
            effects = _merged_target_effects(
                predecessor_target if predecessor_target.exists() else None,
                Path(record["immutable_source_pin"]),
                _inside(candidate, record["target"], "prepared target"),
                funding=record["relative"].startswith("feature_points/"),
            )
            if record["merged_targets"] != effects:
                raise PublicationError("merged target accounting mismatch")
    if is_v4:
        publication = candidate / ".alpha_max_publication" / request_id
        receipt, receipt_bytes = _json(
            publication / "conflict_authorization_receipt.json",
            "prepared conflict authorization receipt",
        )
        message, _ = _json(
            publication / "conflict_authorization_message.json",
            "prepared conflict authorization message",
        )
        message_fields = {
            "schema",
            "scope",
            "decision",
            "canonical_root",
            "acquisition_request_id",
            "terminal_receipt_sha256",
            "source_manifest_sha256",
            "source_eligible_receipt_sha256",
            "predecessor_path",
            "predecessor_identity",
            "predecessor_inventory_sha256",
            "fresh_acquisition_audit_receipt_sha256",
            "acquisition_run_id",
            "composite_telemetry_sha256",
            "wal_transition_receipt_sha256",
            "wal_post_transition_inventory_sha256",
            "signed_terminal_request_sha256",
            "approval_sha256",
            "entries",
        }
        entry_fields = {
            "relative",
            "source_sha256",
            "source_byte_count",
            "source_row_count",
            "provenance_receipt_sha256",
            "predecessor_identity",
            "predecessor_sha256",
            "predecessor_byte_count",
            "predecessor_row_count",
            "effects",
        }
        effect_fields = {
            "conflict_rows",
            "conflict_sha256",
            "canonical_only_rows",
            "canonical_only_sha256",
            "source_only_rows",
            "source_only_sha256",
        }
        message_entries = message.get("entries")
        if not isinstance(message_entries, list) or any(
            not isinstance(entry, dict)
            or set(entry) != entry_fields
            or not _authorized_ohlcv_relative(entry.get("relative"))
            or any(
                type(entry[field]) is not int or entry[field] < 0
                for field in (
                    "source_byte_count",
                    "source_row_count",
                    "predecessor_byte_count",
                    "predecessor_row_count",
                )
            )
            or not isinstance(entry["predecessor_identity"], list)
            or len(entry["predecessor_identity"]) != 9
            or any(type(value) is not int for value in entry["predecessor_identity"])
            or any(
                not isinstance(entry[field], str)
                or re.fullmatch(r"[0-9a-f]{64}", entry[field]) is None
                for field in (
                    "source_sha256",
                    "provenance_receipt_sha256",
                    "predecessor_sha256",
                )
            )
            or not isinstance(entry.get("effects"), dict)
            or set(entry["effects"]) != effect_fields
            or any(
                isinstance(entry["effects"][field], bool)
                or not isinstance(entry["effects"][field], int)
                or entry["effects"][field] < 0
                for field in ("conflict_rows", "canonical_only_rows", "source_only_rows")
            )
            or entry["effects"]["conflict_rows"] == 0
            or any(
                not isinstance(entry["effects"][field], str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["effects"][field]) is None
                for field in (
                    "conflict_sha256",
                    "canonical_only_sha256",
                    "source_only_sha256",
                )
            )
            for entry in message_entries
        ):
            raise PublicationError("prepared conflict authorization binding mismatch")
        authorization_entries = {entry["relative"]: entry for entry in message_entries}
        authorized_records = {
            record["relative"]: record
            for record in published
            if record.get("publication_mode") == "authorized_reconciliation"
        }
        if (
            set(receipt) != {"schema", "type", "authority_key_id", "message", "signature"}
            or receipt.get("schema") != "alpha_max_canonical_conflict_authorization_receipt.v2"
            or receipt.get("type") != "canonical_conflict_authorization"
            or set(message) != message_fields
            or message.get("schema") != "alpha_max_canonical_conflict_authorization_message.v2"
            or message.get("scope") != "canonical_conflict_reconciliation"
            or message.get("decision") != "approve_exact_effects"
            or message.get("canonical_root") != final["canonical_root"]
            or message.get("acquisition_request_id") != request_id
            or message.get("terminal_receipt_sha256") != terminal_sha256
            or message.get("source_manifest_sha256") != hashes["manifest"]
            or message.get("source_eligible_receipt_sha256") != hashes["eligible"]
            or message.get("predecessor_path") != final["predecessor_path"]
            or message.get("predecessor_identity") != final["predecessor_identity"]
            or message.get("predecessor_inventory_sha256") != final["old_inventory_sha256"]
            or len(authorization_entries) != len(message_entries)
            or list(authorization_entries) != sorted(authorization_entries)
            or set(authorization_entries) != set(authorized_records)
            or any(
                record["source_sha256"] != authorization_entries[relative]["source_sha256"]
                or record["bytes"] != authorization_entries[relative]["source_byte_count"]
                or record["rows"] != authorization_entries[relative]["source_row_count"]
                or record["provenance_receipt_sha256"]
                != authorization_entries[relative]["provenance_receipt_sha256"]
                or record["conflict_effects"] != authorization_entries[relative]["effects"]
                for relative, record in authorized_records.items()
            )
            or receipt.get("message") != message
            or hashlib.sha256(receipt_bytes).hexdigest()
            != final["conflict_authorization_receipt_sha256"]
            or hashlib.sha256(canonical_bytes(message)).hexdigest()
            != final["conflict_authorization_message_sha256"]
            or receipt.get("authority_key_id") != final["conflict_authority_key_id"]
            or final["conflict_authority_key_id"] != final["authority_key_id"]
            or any(
                message.get(field) != final.get(field)
                for field in (
                    "fresh_acquisition_audit_receipt_sha256",
                    "acquisition_run_id",
                    "composite_telemetry_sha256",
                    "wal_transition_receipt_sha256",
                    "wal_post_transition_inventory_sha256",
                    "signed_terminal_request_sha256",
                    "approval_sha256",
                )
            )
            or message["predecessor_inventory_sha256"]
            != final["conflict_authorization_predecessor_inventory_sha256"]
            or final["authorized_conflict_partition_count"] != len(authorization_entries)
            or final["authorized_replaced_row_count"]
            != sum(entry["effects"]["conflict_rows"] for entry in authorization_entries.values())
        ):
            raise PublicationError("prepared conflict authorization binding mismatch")
    return published


def _authorized_predecessor_detachments(
    target: str,
    old_inventory: list[dict[str, Any]],
) -> list[str]:
    target_path = PurePosixPath(target)
    allowed: set[str] = set()
    if target_path.parts and target_path.parts[0] == "market_ohlcv_1s":
        allowed.update(
            {
                target,
                str(target_path.with_suffix(".seal.json")),
                str(target_path.with_suffix(".pending.json")),
            }
        )
    elif target_path.parts and target_path.parts[0] == "feature_points":
        for item in old_inventory:
            path = PurePosixPath(item["path"])
            if (
                item["kind"] == "file"
                and path.parent == target_path.parent
                and (path.suffix == ".parquet" or path.name == "alpha_max_official_funding_seal.v1")
            ):
                allowed.add(item["path"])
    return sorted(
        path
        for path in allowed
        if any(item["kind"] == "file" and item["path"] == path for item in old_inventory)
    )


def _validate_cloned_predecessor(
    old: Path,
    candidate: Path,
    old_inventory: list[dict[str, Any]],
    published: list[dict[str, Any]],
) -> list[str]:
    """Prove every normal hardlink break instead of assuming one global nlink."""
    actual = _inventory(old, require_private=False)
    expected = {record["path"]: record for record in old_inventory}
    candidate_entries = {
        record["path"]: record for record in _inventory(candidate, require_private=False)
    }
    authorized_detachments = {
        path
        for published_record in published
        for path in published_record["authorized_predecessor_detachments"]
    }
    if set(expected) != {record["path"] for record in actual}:
        raise PublicationError("prepared predecessor inventory changed")

    detached: list[str] = []
    for record in actual:
        prior = expected[record["path"]]
        if (
            record["kind"] != prior["kind"]
            or (record["dev"], record["ino"]) != (prior["dev"], prior["ino"])
            or record["mode"] != prior["mode"]
        ):
            raise PublicationError("prepared predecessor identity changed")
        if record["kind"] != "file":
            continue
        if record["size"] != prior["size"] or record["mtime_ns"] != prior["mtime_ns"]:
            raise PublicationError("prepared predecessor metadata changed")
        candidate_record = candidate_entries.get(record["path"])
        if record["nlink"] == prior["nlink"] + 1:
            if (
                candidate_record is None
                or candidate_record["kind"] != "file"
                or (candidate_record["dev"], candidate_record["ino"])
                != (prior["dev"], prior["ino"])
                or candidate_record["nlink"] != record["nlink"]
            ):
                raise PublicationError("prepared predecessor shared link changed")
            continue
        if record["nlink"] != prior["nlink"]:
            raise PublicationError("prepared predecessor link count changed")
        coordination_lock = _is_coordination_lock(record["path"])
        if record["path"] not in authorized_detachments and not coordination_lock:
            raise PublicationError("prepared predecessor link break is unauthenticated")
        if candidate_record is not None and (
            candidate_record["kind"] != "file"
            or candidate_record["nlink"] != 1
            or (candidate_record["dev"], candidate_record["ino"]) == (prior["dev"], prior["ino"])
        ):
            raise PublicationError("prepared candidate is not inode-independent")
        detached.append(record["path"])
    return sorted(detached)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "source-report",
        "terminal-receipt",
        "authority-public-key",
        "canonical-root",
    ):
        parser.add_argument("--" + name, required=True)
    for name in (
        "conflict-authorization-receipt",
        "conflict-authority-public-key",
        "fresh-acquisition-audit-receipt",
        "wal-transition-receipt",
        "current-state-approval",
        "observer-ready-receipt",
        "observer-public-key",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--publisher-key-fd", type=int, required=True)
    args = parser.parse_args(argv)
    for name in (
        "source_root",
        "source_report",
        "terminal_receipt",
        "authority_public_key",
        "canonical_root",
    ):
        setattr(args, name, _absolute(getattr(args, name), name))
    for name in (
        "conflict_authorization_receipt",
        "conflict_authority_public_key",
        "fresh_acquisition_audit_receipt",
        "wal_transition_receipt",
        "current_state_approval",
        "observer_ready_receipt",
        "observer_public_key",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, _absolute(value, name))
    _, approval_sha256 = _approval(args.current_state_approval)
    publisher_key = _publication_private_key(args.publisher_key_fd)
    terminal, key_id, bound, terminal_sha256 = _terminal(args)
    terminal_public_key = _conflict_public_key(args.authority_public_key)
    publisher_public = publisher_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    if hashlib.sha256(publisher_public).hexdigest() == key_id:
        raise PublicationError("publisher key must be distinct from terminal authority")
    if hashlib.sha256(terminal_public_key).hexdigest() != key_id:
        raise PublicationError("terminal authority public key binding is invalid")
    request_id = _sha(terminal.get("request_id"), "terminal request identity")
    if request_id != _ACQUISITION_REQUEST_ID:
        raise PublicationError(
            "terminal request identity is not the authorized acquisition request"
        )
    parts, hashes, listing = _partitions(args.source_root, args.source_report, bound)
    generations = args.canonical_root.parent / f".{args.canonical_root.name}.generations"
    control = args.canonical_root.parent / f".{args.canonical_root.name}.transactions" / request_id
    candidate = generations / request_id
    if args.observer_ready_receipt != control / "observer-ready.json":
        raise PublicationError("publication observer readiness path is invalid")
    observer_ready, observer_ready_sha256 = _observer_ready(
        args.observer_ready_receipt,
        args.observer_public_key,
        request_id=request_id,
        approval_sha256=approval_sha256,
        canonical_root=args.canonical_root,
    )
    if (
        observer_ready is not None
        and len(
            {
                key_id,
                hashlib.sha256(publisher_public).hexdigest(),
                observer_ready["observer_key_id"],
            }
        )
        != 3
    ):
        raise PublicationError("terminal authority, publisher, and observer keys must be distinct")
    lock_repo = ParquetMarketDataRepository(args.canonical_root)
    with lock_repo.generation_lock(exclusive=True, allow_incomplete_bootstrap=True):
        _ensure_private_directory(generations)
        _ensure_private_directory(control)
        try:
            active = _active_root(args.canonical_root, generations)
        except FileNotFoundError:
            active = None
        if (
            observer_ready is not None
            and active != candidate
            and not (control / "pre-open-rollback-intent.json").exists()
        ):
            _observer_process_alive(observer_ready)
            if (
                active is None
                or _topology_identity(active) != observer_ready["old_identity"]
                or _canonical_inventory(active)["inventory_sha256"]
                != observer_ready["old_inventory_sha256"]
            ):
                raise PublicationError("active predecessor mismatches observer readiness")
        if candidate.exists() and active != candidate:
            rollback_path = control / "pre-open-rollback-intent.json"
            if rollback_path.exists():
                rollback = _verify_stage_envelope(rollback_path, publisher_key, "activation")
                activation_intent = _verify_stage_envelope(
                    control / "activation-intent.json", publisher_key, "activation"
                )
                swap, swap_raw = _json(control / "swap.json", "pre-open rollback swap receipt")
                temporary_leaf = swap.get("temporary")
                if (
                    activation_intent.get("schema") != "alpha_max_publication_activation_intent.v1"
                    or activation_intent.get("phase") != "activation_intent"
                    or activation_intent.get("request_id") != request_id
                    or activation_intent.get("approval_sha256") != approval_sha256
                    or activation_intent.get("candidate") != str(candidate)
                    or set(swap) != {"request_id", "phase", "candidate", "temporary", "mode"}
                    or swap.get("request_id") != request_id
                    or swap.get("phase") != "swap_ready"
                    or swap.get("candidate") != candidate.name
                    or swap.get("mode") != "exchange"
                    or not isinstance(temporary_leaf, str)
                    or Path(temporary_leaf).name != temporary_leaf
                    or temporary_leaf in {"", ".", ".."}
                    or hashlib.sha256(swap_raw).hexdigest()
                    != activation_intent.get("swap_payload_sha256")
                ):
                    raise PublicationError("pre-open rollback activation binding is invalid")
                temporary = args.canonical_root.parent / temporary_leaf
                if activation_intent.get("swap") != temporary_leaf or activation_intent.get(
                    "swap_temporary_path"
                ) != str(temporary):
                    raise PublicationError("pre-open rollback swap path binding is invalid")
                if not os.path.lexists(args.canonical_root) or not os.path.lexists(temporary):
                    raise PublicationError("pre-open rollback topology is unavailable")
                restored_identity = _topology_identity(args.canonical_root)
                candidate_identity = _topology_identity(candidate)
                temporary_identity = _topology_identity(temporary)
                if (
                    restored_identity != activation_intent.get("expected_old_identity")
                    or candidate_identity != activation_intent.get("expected_candidate_identity")
                    or temporary_identity != activation_intent.get("expected_swap_identity")
                ):
                    raise PublicationError("pre-open rollback topology binding is invalid")
                _validate_preopen_rollback_intent(
                    rollback,
                    control=control,
                    observer_public_key=args.observer_public_key,
                    request_id=request_id,
                    approval_sha256=approval_sha256,
                    activation_intent_sha256=hashlib.sha256(
                        _regular_bytes(control / "activation-intent.json", "activation intent")
                    ).hexdigest(),
                    candidate_identity=candidate_identity,
                    predecessor_identity=restored_identity,
                    swap_identity=restored_identity,
                    observer_ready_sha256=observer_ready_sha256,
                )
                _fsync_dir(args.canonical_root.parent)
                if (
                    _topology_identity(args.canonical_root) != restored_identity
                    or _topology_identity(candidate) != candidate_identity
                    or _topology_identity(temporary) != temporary_identity
                ):
                    raise PublicationError("pre-open rollback topology changed")
                _write_noreplace(
                    control / "pre-open-rollback.json",
                    _stage_envelope(
                        publisher_key, "activation", {**rollback, "phase": "rolled_back"}
                    ),
                )
                raise PublicationError("pre-open observer failure was rolled back")
            _manager_terminal_or_observer_failure(control)

        if active == candidate:
            activation_intent_path = control / "activation-intent.json"
            intent = _verify_stage_envelope(activation_intent_path, publisher_key, "activation")
            activation_intent_sha256 = hashlib.sha256(
                _regular_bytes(activation_intent_path, "activation intent")
            ).hexdigest()
            swap, _ = _json(control / "swap.json", "active swap receipt")
            temporary = args.canonical_root.parent / swap["temporary"]
            if (
                intent.get("schema") != "alpha_max_publication_activation_intent.v1"
                or intent.get("phase") != "activation_intent"
                or intent.get("request_id") != request_id
                or intent.get("expected_candidate_identity") != _topology_identity(candidate)
                or intent.get("expected_swap_identity") != _topology_identity(args.canonical_root)
                or intent.get("expected_old_identity") != _topology_identity(temporary)
            ):
                raise PublicationError("active candidate activation intent binding is invalid")
            if not (candidate / "commit.json").exists() or not (control / "prepared.json").exists():
                swap, _ = _json(control / "swap.json", "unready active swap receipt")
                temporary = args.canonical_root.parent / swap["temporary"]
                if not os.path.lexists(temporary):
                    raise PublicationError("unready active generation has no bound predecessor")
                _rename_exchange(temporary, args.canonical_root)
                _fsync_dir(args.canonical_root.parent)
                restored = _active_root(args.canonical_root, generations)
                if os.path.lexists(restored / ".bootstrap-incomplete"):
                    args.canonical_root.unlink()
                    _fsync_dir(args.canonical_root.parent)
                _write_noreplace(
                    control / "rollback.json",
                    {"request_id": request_id, "phase": "rollback"},
                )
                raise PublicationError("unready active generation was rolled back")
            final, _ = _json(candidate / "commit.json", "active commit")
            if (
                final.get("schema") != "alpha_max_canonical_publication_receipt.v4"
                or final.get("approval_sha256") != approval_sha256
            ):
                raise PublicationError("active publication v4 approval binding changed")
            _replay_conflict_authorization(
                args,
                final=final,
                terminal_sha256=terminal_sha256,
                request_id=request_id,
                hashes=hashes,
                parts=parts,
                terminal=terminal,
                key_id=key_id,
                terminal_public_key=terminal_public_key,
            )
            _validate_prepared_commit(
                candidate,
                final,
                request_id=request_id,
                terminal_sha256=terminal_sha256,
                key_id=key_id,
                hashes=hashes,
                listing=listing,
                parts=parts,
            )
            failure = _preopen_failure_evidence(
                control,
                args.observer_public_key,
                request_id=request_id,
                approval_sha256=approval_sha256,
                observer_ready_sha256=observer_ready_sha256,
            )
            observer_unavailable = False
            if observer_ready is not None:
                try:
                    _observer_process_alive(observer_ready)
                except PublicationError:
                    observer_unavailable = True
            rollback_intent_path = control / "pre-open-rollback-intent.json"
            rollback_intent = None
            if rollback_intent_path.exists():
                rollback_intent = _verify_stage_envelope(
                    rollback_intent_path, publisher_key, "activation"
                )
                _validate_preopen_rollback_intent(
                    rollback_intent,
                    control=control,
                    observer_public_key=args.observer_public_key,
                    request_id=request_id,
                    approval_sha256=approval_sha256,
                    activation_intent_sha256=activation_intent_sha256,
                    candidate_identity=intent["expected_candidate_identity"],
                    predecessor_identity=intent["expected_old_identity"],
                    swap_identity=_topology_identity(temporary),
                    observer_ready_sha256=observer_ready_sha256,
                )
            elif not (control / "replay-verified.json").exists() and (
                failure is not None or observer_unavailable
            ):
                failure_reason = (
                    "observer_failure_evidence"
                    if failure is not None
                    else "observer_process_unavailable"
                )
                rollback_intent = {
                    "schema": "alpha_max_publication_pre_open_rollback.v2",
                    "phase": "rollback_intent",
                    "request_id": request_id,
                    "approval_sha256": approval_sha256,
                    "activation_intent_sha256": activation_intent_sha256,
                    "observer_ready_sha256": observer_ready_sha256,
                    "failure_reason": failure_reason,
                    "failure_evidence_leaf": failure[0] if failure is not None else None,
                    "failure_evidence_sha256": failure[1] if failure is not None else None,
                    "candidate_identity": intent["expected_candidate_identity"],
                    "predecessor_identity": intent["expected_old_identity"],
                    "swap_identity": _topology_identity(temporary),
                }
                _write_noreplace(
                    rollback_intent_path,
                    _stage_envelope(publisher_key, "activation", rollback_intent),
                )
            if rollback_intent is not None:
                if not os.path.lexists(temporary):
                    raise PublicationError("pre-open recovery predecessor is unavailable")
                _rename_exchange(temporary, args.canonical_root)
                _fsync_dir(args.canonical_root.parent)
                if _topology_identity(args.canonical_root) != intent["expected_old_identity"]:
                    raise PublicationError("pre-open recovery predecessor identity changed")
                _fsync_dir(args.canonical_root.parent)
                if (
                    _topology_identity(args.canonical_root)
                    != rollback_intent["predecessor_identity"]
                ):
                    raise PublicationError("pre-open recovery predecessor identity changed")
                _write_noreplace(
                    control / "pre-open-rollback.json",
                    _stage_envelope(
                        publisher_key, "activation", {**rollback_intent, "phase": "rolled_back"}
                    ),
                )
                raise PublicationError("pre-open observer failure was rolled back")
            swap, _ = _json(control / "swap.json", "swap receipt")
            temporary = args.canonical_root.parent / swap["temporary"]
            if observer_ready is not None:
                _observer_process_alive(observer_ready)
            _write_open_window_receipts(
                control=control,
                publisher_key=publisher_key,
                request_id=request_id,
                key_id=key_id,
                terminal_sha256=terminal_sha256,
                candidate=candidate,
                final=final,
                swap=swap,
                temporary=temporary,
                replay=True,
            )
            return 0

        if candidate.exists():
            if not (candidate / "commit.json").exists():
                raise PublicationError("incomplete candidate requires bounded operator recovery")
            final, _ = _json(candidate / "commit.json", "prepared commit")
            if (
                final.get("schema") != "alpha_max_canonical_publication_receipt.v4"
                or final.get("approval_sha256") != approval_sha256
            ):
                raise PublicationError("prepared publication v4 approval binding changed")
            canonical_receipt_path = (
                candidate
                / ".alpha_max_publication"
                / request_id
                / "canonical_publication_receipt.json"
            )
            if not canonical_receipt_path.exists():
                _write_noreplace(canonical_receipt_path, final)
            _replay_conflict_authorization(
                args,
                final=final,
                terminal_sha256=terminal_sha256,
                request_id=request_id,
                hashes=hashes,
                parts=parts,
                terminal=terminal,
                key_id=key_id,
                terminal_public_key=terminal_public_key,
            )
            published = _validate_prepared_commit(
                candidate,
                final,
                request_id=request_id,
                terminal_sha256=terminal_sha256,
                key_id=key_id,
                hashes=hashes,
                listing=listing,
                parts=parts,
            )
            if not (control / "prepared.json").exists():
                _write_noreplace(
                    control / "prepared.json",
                    {
                        "request_id": request_id,
                        "phase": "prepared",
                        "candidate": candidate.name,
                    },
                )
            old_inventory = final["old_inventory"]
            predecessor = Path(final["predecessor_path"])
            if active is not None and active != predecessor:
                raise PublicationError("prepared predecessor generation changed")
            if not predecessor.exists() or predecessor.is_symlink():
                raise PublicationError("prepared predecessor is missing")
            old = predecessor
            detached = _validate_cloned_predecessor(old, candidate, old_inventory, published)
            if detached != final.get("detached_predecessor_paths"):
                raise PublicationError("prepared predecessor detachment binding mismatch")
        else:
            root_missing = active is None
            if root_missing:
                old = generations / f".bootstrap-{uuid.uuid4().hex}"
                old.mkdir(mode=0o700)
                _fsync_dir(generations)
                _write_noreplace(
                    old / ".bootstrap-incomplete",
                    {
                        "schema": "alpha_max_incomplete_bootstrap.v1",
                        "request_id": request_id,
                    },
                )
            else:
                old = active
            old_identity = (os.stat(old).st_dev, os.stat(old).st_ino)
            old_inventory = _inventory(old)
            controls = _conflict_controls(
                args,
                terminal=terminal,
                approval_sha256=approval_sha256,
                terminal_sha256=terminal_sha256,
                request_id=request_id,
                key_id=key_id,
                predecessor_inventory=old_inventory,
            )
            authorization = _conflict_authorization(
                args,
                terminal_sha256=terminal_sha256,
                request_id=request_id,
                hashes=hashes,
                predecessor=old,
                predecessor_inventory=old_inventory,
                parts=parts,
                controls=controls,
                terminal_public_key=terminal_public_key,
            )
            if authorization is None or authorization["key_id"] != key_id:
                raise PublicationError("complete v4 conflict authorization is required")
            _write_noreplace(
                control / "cloning.json",
                {"request_id": request_id, "phase": "cloning"},
            )
            _clone_root(old, candidate, old_inventory)
            _write_noreplace(
                control / "cloned.json",
                {
                    "request_id": request_id,
                    "phase": "cloned",
                    "candidate": candidate.name,
                    "old_inventory_sha256": _inventory_digest(old_inventory),
                },
            )
            if authorization is not None:
                publication = candidate / ".alpha_max_publication" / request_id
                _write_noreplace(
                    publication / "conflict_authorization_receipt.json", authorization["receipt"]
                )
                _write_noreplace(
                    publication / "conflict_authorization_message.json", authorization["message"]
                )
            repo = ParquetMarketDataRepository(candidate)
            feature_repo = MarketDataRepository(str(candidate))
            published = []
            capacity_admissions: list[dict[str, Any]] = []
            for item in parts:
                byte_count, before = _hash_source(item["source"], item["sha256"])
                admission = _publication_admission(candidate, item, byte_count)
                capacity_admissions.append({"relative": item["relative"], **admission})
                pin = (
                    control
                    / "pinned"
                    / (hashlib.sha256(item["relative"].encode()).hexdigest() + ".parquet")
                )
                pin_count, pin_source_identity = _pin_source(
                    item["source"],
                    pin,
                    item["sha256"],
                )
                if pin_count != byte_count or pin_source_identity != before:
                    raise PublicationError("source identity changed before immutable pin")
                pinned_count, pinned_identity = _hash_source(pin, item["sha256"])
                if pinned_count != byte_count:
                    raise PublicationError("immutable source pin size mismatch")
                path = PurePosixPath(item["relative"])
                authorized_entry = (
                    authorization["entries"].get(item["relative"])
                    if authorization is not None
                    else None
                )
                if path.parts[0] == "market_ohlcv_1s" and authorized_entry is not None:
                    target = repo.reconcile_authorized_signed_month_into_candidate(
                        exchange="binance",
                        symbol=path.parts[2],
                        month=path.parts[3][:-8],
                        source=pin,
                        expected_sha256=item["sha256"],
                        expected_byte_count=byte_count,
                        expected_row_count=item["rows"],
                        provenance_receipt_sha256=item["provenance"],
                        conflict_authorization_receipt=authorization["receipt"],
                        conflict_authority_public_key=authorization["public_key"],
                        authorization_entry=authorized_entry,
                        expected_run_id=controls["acquisition_run_id"],
                        expected_approval_sha256=approval_sha256,
                        max_output_bytes=admission["enforced_output_quota_bytes"],
                    )
                elif path.parts[0] == "market_ohlcv_1s":
                    target = repo.merge_signed_month_into_candidate(
                        exchange="binance",
                        symbol=path.parts[2],
                        month=path.parts[3][:-8],
                        source=pin,
                        expected_sha256=item["sha256"],
                        expected_byte_count=byte_count,
                        expected_row_count=item["rows"],
                        provenance_receipt_sha256=item["provenance"],
                        max_output_bytes=admission["enforced_output_quota_bytes"],
                    )
                else:
                    target = feature_repo.publish_official_funding_day(
                        exchange="binance",
                        symbol=path.parts[2][7:],
                        day=path.parts[3][5:],
                        source=pin,
                        expected_sha256=item["sha256"],
                        expected_byte_count=byte_count,
                        expected_row_count=item["rows"],
                        provenance_receipt_sha256=item["provenance"],
                        max_output_bytes=admission["enforced_output_quota_bytes"],
                    )
                after_count, after = _hash_source(item["source"], item["sha256"])
                if after_count != byte_count or after != before:
                    raise PublicationError("source output identity or digest changed during merge")
                source_info = os.lstat(pin)
                target_info = os.lstat(target)
                if (source_info.st_dev, source_info.st_ino) == (
                    target_info.st_dev,
                    target_info.st_ino,
                ):
                    raise PublicationError("published target shares the staging source inode")
                if _host_free_bytes() <= _STORAGE["host_reserve_bytes"]:
                    raise PublicationError("host reserve was violated during source copy")
                target_size, target_sha = _file_sha(target)
                record = {
                    "relative": item["relative"],
                    "target": str(target.relative_to(candidate)),
                    "source_sha256": item["sha256"],
                    "source_identity": list(before),
                    "immutable_source_pin": str(pin),
                    "immutable_source_pin_identity": list(pinned_identity),
                    "authorized_predecessor_detachments": _authorized_predecessor_detachments(
                        str(target.relative_to(candidate)),
                        old_inventory,
                    ),
                    "target_sha256": target_sha,
                    "bytes": byte_count,
                    "target_bytes": target_size,
                    "rows": item["rows"],
                    "target_rows": _parquet_rows(target),
                    "provenance_receipt_sha256": item["provenance"],
                }
                if authorization is not None:
                    record["publication_mode"] = (
                        "authorized_reconciliation"
                        if authorized_entry is not None
                        else "strict_merge"
                    )
                    record["conflict_effects"] = (
                        authorized_entry["effects"] if authorized_entry is not None else None
                    )
                    record["merged_targets"] = _merged_target_effects(
                        old / str(target.relative_to(candidate)),
                        pin,
                        target,
                        funding=path.parts[0] == "feature_points",
                    )
                _write_noreplace(
                    candidate
                    / ".alpha_max_publication"
                    / request_id
                    / "partitions"
                    / (hashlib.sha256(item["relative"].encode()).hexdigest() + ".json"),
                    record,
                )
                published.append(record)
            listing_metadata = {
                "schema": "alpha_max_canonical_listing.v1",
                "request_id": request_id,
                "records": listing,
            }
            listing_path = (
                candidate / ".alpha_max_publication" / request_id / "listing_records.json"
            )
            _write_noreplace(listing_path, listing_metadata)
            listing_metadata_sha256 = hashlib.sha256(canonical_bytes(listing_metadata)).hexdigest()
            detached_predecessor_paths = _validate_cloned_predecessor(
                old, candidate, old_inventory, published
            )
            candidate_data_digest = _data_inventory_digest(candidate)
            if authorization is not None and len(authorization["entries"]) != sum(
                record["publication_mode"] == "authorized_reconciliation" for record in published
            ):
                raise PublicationError("conflict authorization entry was not consumed exactly once")
            final = {
                "schema": "alpha_max_canonical_publication_receipt.v4",
                "request_id": request_id,
                "terminal_receipt_sha256": terminal_sha256,
                "authority_key_id": key_id,
                "source_manifest_sha256": hashes["manifest"],
                "source_eligible_receipt_sha256": hashes["eligible"],
                "source_journal_sha256": hashes["journal"],
                "contract_manifest_sha256": hashes["contract"],
                "listing_records": listing,
                "listing_metadata_sha256": listing_metadata_sha256,
                "canonical_root": str(args.canonical_root),
                "predecessor_identity": list(old_identity),
                "predecessor_path": str(old),
                "old_inventory": old_inventory,
                "old_inventory_sha256": _inventory_digest(old_inventory),
                "candidate_data_inventory_sha256": candidate_data_digest,
                "detached_predecessor_paths": detached_predecessor_paths,
                "capacity_admissions": capacity_admissions,
                "partitions": published,
                "ohlcv_partition_count": sum(
                    item["relative"].startswith("market_ohlcv_1s/") for item in published
                ),
                "funding_partition_count": sum(
                    item["relative"].startswith("feature_points/") for item in published
                ),
                "listing_count": len(listing),
                "rows": sum(item["rows"] for item in published),
            }
            if authorization is not None:
                if observer_ready is None or observer_ready_sha256 is None:
                    raise PublicationError("signed publication is missing observer readiness")
                final.update(
                    {
                        "conflict_authorization_receipt_sha256": authorization["receipt_sha256"],
                        "conflict_authority_key_id": authorization["key_id"],
                        "conflict_authorization_message_sha256": authorization["message_sha256"],
                        "conflict_authorization_predecessor_inventory_sha256": authorization[
                            "message"
                        ]["predecessor_inventory_sha256"],
                        "authorized_conflict_partition_count": len(authorization["entries"]),
                        "authorized_replaced_row_count": sum(
                            entry["effects"]["conflict_rows"]
                            for entry in authorization["entries"].values()
                        ),
                        "observer_key_id": observer_ready["observer_key_id"],
                        "observer_ready_sha256": observer_ready_sha256,
                        "observer_query_spec_sha256": observer_ready["query_spec_sha256"],
                        "source_input": {
                            "partitions": len(parts),
                            "rows": sum(item["rows"] for item in parts),
                            "bytes": sum(item["bytes"] for item in published),
                        },
                        "canonical_before": _canonical_inventory(old),
                        "canonical_candidate": _canonical_inventory(candidate),
                        **{
                            field: authorization["message"][field]
                            for field in (
                                "fresh_acquisition_audit_receipt_sha256",
                                "acquisition_run_id",
                                "composite_telemetry_sha256",
                                "wal_transition_receipt_sha256",
                                "wal_post_transition_inventory_sha256",
                                "signed_terminal_request_sha256",
                                "approval_sha256",
                            )
                        },
                    }
                )
            _write_noreplace(candidate / "commit.json", final)
            _write_noreplace(
                candidate
                / ".alpha_max_publication"
                / request_id
                / "canonical_publication_receipt.json",
                final,
            )
            _write_noreplace(
                control / "prepared.json",
                {
                    "request_id": request_id,
                    "phase": "prepared",
                    "candidate": candidate.name,
                },
            )

        _validate_prepared_commit(
            candidate,
            final,
            request_id=request_id,
            terminal_sha256=terminal_sha256,
            key_id=key_id,
            hashes=hashes,
            listing=listing,
            parts=parts,
        )
        predecessor = Path(final["predecessor_path"])
        predecessor_info = os.lstat(predecessor)
        if (
            not stat.S_ISDIR(predecessor_info.st_mode)
            or [predecessor_info.st_dev, predecessor_info.st_ino] != final["predecessor_identity"]
        ):
            raise PublicationError("pre-exchange predecessor identity changed")
        if active is not None and active != predecessor:
            raise PublicationError("pre-exchange predecessor generation changed")
        if (
            _host_free_bytes()
            <= _STORAGE["host_reserve_bytes"] + _PUBLICATION_CONTROL_RESERVE_BYTES
        ):
            raise PublicationError("host reserve cannot guarantee exchange and rollback")
        swap_path = control / "swap.json"
        if swap_path.exists():
            swap, _ = _json(swap_path, "swap receipt")
            temporary = args.canonical_root.parent / swap["temporary"]
            mode = swap["mode"]
            if mode != "exchange":
                raise PublicationError("prepared swap mode is invalid")
            if not os.path.lexists(temporary):
                os.symlink(
                    f".{args.canonical_root.name}.generations/{request_id}",
                    temporary,
                )
                _fsync_dir(temporary.parent)
            if not temporary.is_symlink():
                raise PublicationError("prepared swap path is invalid")
        else:
            temporary = (
                args.canonical_root.parent / f".{args.canonical_root.name}.{uuid.uuid4().hex}.swap"
            )
            os.symlink(
                f".{args.canonical_root.name}.generations/{request_id}",
                temporary,
            )
            swap = {
                "request_id": request_id,
                "phase": "swap_ready",
                "candidate": candidate.name,
                "temporary": temporary.name,
                "mode": "exchange",
            }
            _write_noreplace(swap_path, swap)
            _fsync_dir(args.canonical_root.parent)
        mode = swap["mode"]
        if os.readlink(temporary) != f"{generations.name}/{request_id}":
            raise PublicationError("prepared swap target changed")
        if active is None:
            if predecessor.parent != generations or not predecessor.is_dir():
                raise PublicationError("bootstrap predecessor is untrusted")
            os.symlink(f"{generations.name}/{predecessor.name}", args.canonical_root)
            _fsync_dir(args.canonical_root.parent)
            active = _active_root(args.canonical_root, generations)
        if active != predecessor:
            raise PublicationError("pre-exchange predecessor generation changed")
        if mode != "exchange":
            raise PublicationError("prepared swap mode is invalid")
        _observer_process_alive(observer_ready)
        intent_bindings = _window_bindings(
            request_id=request_id,
            key_id=key_id,
            terminal_sha256=terminal_sha256,
            candidate=candidate,
            final=final,
            swap=swap,
            temporary=temporary,
        )
        _write_noreplace(
            control / "activation-intent.json",
            _stage_envelope(
                publisher_key,
                "activation",
                {
                    "schema": "alpha_max_publication_activation_intent.v1",
                    "phase": "activation_intent",
                    "expected_old_identity": _topology_identity(args.canonical_root),
                    "expected_candidate_identity": _topology_identity(candidate),
                    "expected_swap_identity": _topology_identity(temporary),
                    "commit_payload_sha256": hashlib.sha256(canonical_bytes(final)).hexdigest(),
                    "swap_payload_sha256": hashlib.sha256(canonical_bytes(swap)).hexdigest(),
                    **intent_bindings,
                },
            ),
        )
        _rename_exchange(temporary, args.canonical_root)
        _fsync_dir(args.canonical_root.parent)
        try:
            activated = _active_root(args.canonical_root, generations)
            if activated != candidate:
                raise PublicationError("activated generation target mismatch")
            _validate_prepared_commit(
                candidate,
                final,
                request_id=request_id,
                terminal_sha256=terminal_sha256,
                key_id=key_id,
                hashes=hashes,
                listing=listing,
                parts=parts,
            )
        except Exception:
            if temporary.exists() or temporary.is_symlink():
                _rename_exchange(temporary, args.canonical_root)
            _fsync_dir(args.canonical_root.parent)
            if os.path.lexists(predecessor / ".bootstrap-incomplete"):
                if args.canonical_root.is_symlink():
                    args.canonical_root.unlink()
                    _fsync_dir(args.canonical_root.parent)
                if temporary.is_symlink():
                    temporary.unlink()
                    _fsync_dir(temporary.parent)
            _write_noreplace(
                control / "rollback.json",
                {"request_id": request_id, "phase": "rollback"},
            )
            _write_noreplace(
                control / "failure.json",
                {"request_id": request_id, "phase": "failure"},
            )
            raise
        swap, _ = _json(control / "swap.json", "swap receipt")
        _write_open_window_receipts(
            control=control,
            publisher_key=publisher_key,
            request_id=request_id,
            key_id=key_id,
            terminal_sha256=terminal_sha256,
            candidate=candidate,
            final=final,
            swap=swap,
            temporary=temporary,
            replay=False,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PublicationError, ValueError, OSError) as exc:
        raise SystemExit(str(exc))
