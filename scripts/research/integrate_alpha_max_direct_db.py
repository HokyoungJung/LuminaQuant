#!/usr/bin/env python3
"""Activate a freshly verified Alpha-Max source as one canonical DB generation."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from lumina_quant.market_data import MarketDataRepository as FeatureMarketDataRepository  # noqa: E402

from lumina_quant.storage.parquet import ParquetMarketDataRepository  # noqa: E402
from lumina_quant.storage.wal.binary import BinaryWAL  # noqa: E402
from scripts.research import acquire_alpha_max_official_source as acquisition  # noqa: E402
from scripts.research.audit_alpha_max_canonical_contract import (  # noqa: E402
    CONTRACT_SHA256,
    RAW_SCHEMA,
    audit_contract,
    audit_raw_partition,
    load_contract,
    month_bounds,
    month_starts,
    utc_ms,
)

FICLONE = 0x40049409
FUNDING_FEATURE_COLUMNS = (
    "funding_rate",
    "funding_mark_price",
    "funding_fee_rate",
    "funding_fee_quote_per_unit",
)
EXPECTED_RAW_PARTITIONS = 415
EXPECTED_RAW_ROWS = 1_066_681_730
EXPECTED_FUNDING_ROWS = 39_569
EXPECTED_FUNDING_PARTITIONS = 12_347
CANDIDATE_ID = re.compile(r"[a-z0-9][a-z0-9-]{7,79}")
CANDIDATE_OWNER_FILENAME = ".alpha-max-g003-candidate-owner.json"
PREPARATION_INTENT_FILENAME = "preparation_intent.json"
PREPARATION_READY_FILENAME = "preparation_ready.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while chunk := os.read(fd, 1024 * 1024):
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def inode_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(stat.S_IFMT(info.st_mode)),
        int(info.st_uid),
        int(info.st_gid),
    )


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def directory_identity(path: Path) -> dict[str, int]:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"not a directory: {path}")
    return {
        "dev": int(info.st_dev),
        "ino": int(info.st_ino),
        "mode": int(stat.S_IMODE(info.st_mode)),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
    }


def directory_owner_identity(path: Path) -> dict[str, int]:
    identity = directory_identity(path)
    identity.pop("nlink")
    return identity


def stable_regular(path: Path) -> os.stat_result:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"unsafe regular file: {path}")
    return info


def atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    canonical: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = (
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if canonical
            else json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        )
        + "\n"
    ).encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(errno.EIO, "short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_filesystem(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syncfs = libc.syncfs
        syncfs.argtypes = [ctypes.c_int]
        syncfs.restype = ctypes.c_int
        if syncfs(fd) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    finally:
        os.close(fd)


def tree_file_bytes(root: Path) -> int:
    total = 0
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += int(info.st_size)
    return total


def capacity_audit(
    canonical_root: Path,
    source_root: Path,
    *,
    reserve_bytes: int,
) -> dict[str, int | bool]:
    usage = shutil.disk_usage(canonical_root.parent)
    canonical_bytes = tree_file_bytes(canonical_root)
    source_bytes = tree_file_bytes(source_root)
    required_free = reserve_bytes + canonical_bytes + source_bytes
    return {
        "free_bytes": int(usage.free),
        "reserve_bytes": reserve_bytes,
        "canonical_file_bytes": canonical_bytes,
        "source_file_bytes": source_bytes,
        "required_free_bytes": required_free,
        "passes": int(usage.free) >= required_free,
    }


def write_frame_atomic(frame: pl.DataFrame, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = 0o600
    if target.exists() or target.is_symlink():
        mode = stat.S_IMODE(stable_regular(target).st_mode)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.g003.tmp")
    try:
        frame.write_parquet(temporary)
        os.chmod(temporary, mode)
        fsync_file(temporary)
        output_sha256 = file_sha256(temporary)
        output_bytes = stable_regular(temporary).st_size
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            fsync_directory(temporary.parent)
    if file_sha256(target) != output_sha256:
        raise ValueError(f"activated frame digest mismatch: {target}")
    return {
        "target": str(target),
        "sha256": output_sha256,
        "byte_count": int(output_bytes),
        "copy_mode": "rewrite",
    }


def _copy_file_payload(source_fd: int, target_fd: int) -> str:
    try:
        fcntl.ioctl(target_fd, FICLONE, source_fd)
        return "reflink"
    except OSError as exc:
        if exc.errno not in {
            errno.EINVAL,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
            errno.EXDEV,
        }:
            raise
    os.lseek(source_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        view = memoryview(chunk)
        while view:
            written = os.write(target_fd, view)
            if written <= 0:
                raise OSError(errno.EIO, "short write")
            view = view[written:]
    return "copy"


def clone_file_atomic(source: Path, target: Path) -> dict[str, Any]:
    source_info = stable_regular(source)
    source_digest = file_sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.g003.tmp")
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    target_fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        stat.S_IMODE(source_info.st_mode),
    )
    try:
        mode = _copy_file_payload(source_fd, target_fd)
        os.fsync(target_fd)
    finally:
        os.close(target_fd)
        os.close(source_fd)
    try:
        copied_info = stable_regular(temporary)
        if copied_info.st_size != source_info.st_size or file_sha256(temporary) != source_digest:
            raise ValueError(f"copied file differs from source: {source}")
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            fsync_directory(temporary.parent)
    target_info = stable_regular(target)
    if target_info.st_size != source_info.st_size or file_sha256(target) != source_digest:
        raise ValueError(f"activated file differs from source: {target}")
    return {
        "source": str(source),
        "target": str(target),
        "sha256": source_digest,
        "byte_count": int(source_info.st_size),
        "copy_mode": mode,
    }


def _load_preparation_intent(
    args: argparse.Namespace,
    *,
    candidate_root: Path,
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    path = args.receipt_dir / PREPARATION_INTENT_FILENAME
    payload = ParquetMarketDataRepository._read_canonical_json(path)
    expected_keys = {
        "artifact_kind",
        "schema",
        "phase",
        "candidate_id",
        "candidate_nonce",
        "canonical_root",
        "candidate_root",
        "canonical_before",
        "contract_sha256",
        "source",
        "created_at_utc",
    }
    if (
        set(payload) != expected_keys
        or payload["artifact_kind"] != "alpha_max_direct_db_preparation_intent"
        or payload["schema"] != "alpha_max_direct_db_preparation_intent.v1"
        or payload["phase"] != "building"
        or payload["candidate_id"] != args.candidate_id
        or not re.fullmatch(r"[0-9a-f]{32}", str(payload["candidate_nonce"]))
        or payload["canonical_root"] != str(args.canonical_root)
        or payload["candidate_root"] != str(candidate_root)
        or payload["contract_sha256"] != CONTRACT_SHA256
        or payload["source"] != source_receipt
        or not isinstance(payload["canonical_before"], dict)
        or not isinstance(payload["created_at_utc"], str)
        or not payload["created_at_utc"]
    ):
        raise ValueError("durable preparation intent is invalid")
    return payload


def _create_preparation_intent(
    args: argparse.Namespace,
    *,
    candidate_root: Path,
    canonical_before: dict[str, int],
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    payload = {
        "artifact_kind": "alpha_max_direct_db_preparation_intent",
        "schema": "alpha_max_direct_db_preparation_intent.v1",
        "phase": "building",
        "candidate_id": args.candidate_id,
        "candidate_nonce": uuid.uuid4().hex,
        "canonical_root": str(args.canonical_root),
        "candidate_root": str(candidate_root),
        "canonical_before": canonical_before,
        "contract_sha256": CONTRACT_SHA256,
        "source": source_receipt,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(
        args.receipt_dir / PREPARATION_INTENT_FILENAME,
        payload,
        canonical=True,
    )
    return payload


def _write_candidate_owner(
    location: Path,
    *,
    preparation: dict[str, Any],
    preparation_intent_path: Path,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": "alpha_max_direct_db_candidate_owner",
        "schema": "alpha_max_direct_db_candidate_owner.v1",
        "candidate_id": preparation["candidate_id"],
        "candidate_nonce": preparation["candidate_nonce"],
        "canonical_root": preparation["canonical_root"],
        "candidate_root": preparation["candidate_root"],
        "canonical_before": preparation["canonical_before"],
        "contract_sha256": preparation["contract_sha256"],
        "source": preparation["source"],
        "preparation_intent_sha256": file_sha256(preparation_intent_path),
        "candidate_identity": directory_owner_identity(location),
        "created_at_utc": preparation["created_at_utc"],
    }
    atomic_json(location / CANDIDATE_OWNER_FILENAME, payload, canonical=True)
    return payload


def _validate_candidate_owner(
    location: Path,
    *,
    candidate_root: Path,
    preparation: dict[str, Any],
    preparation_intent_path: Path,
) -> dict[str, Any]:
    marker = location / CANDIDATE_OWNER_FILENAME
    payload = ParquetMarketDataRepository._read_canonical_json(marker)
    expected_keys = {
        "artifact_kind",
        "schema",
        "candidate_id",
        "candidate_nonce",
        "canonical_root",
        "candidate_root",
        "canonical_before",
        "contract_sha256",
        "source",
        "preparation_intent_sha256",
        "candidate_identity",
        "created_at_utc",
    }
    if (
        set(payload) != expected_keys
        or payload["artifact_kind"] != "alpha_max_direct_db_candidate_owner"
        or payload["schema"] != "alpha_max_direct_db_candidate_owner.v1"
        or payload["candidate_id"] != preparation["candidate_id"]
        or payload["candidate_nonce"] != preparation["candidate_nonce"]
        or payload["canonical_root"] != preparation["canonical_root"]
        or payload["candidate_root"] != str(candidate_root)
        or payload["canonical_before"] != preparation["canonical_before"]
        or payload["contract_sha256"] != preparation["contract_sha256"]
        or payload["source"] != preparation["source"]
        or payload["preparation_intent_sha256"] != file_sha256(preparation_intent_path)
        or payload["candidate_identity"] != directory_owner_identity(location)
        or payload["created_at_utc"] != preparation["created_at_utc"]
    ):
        raise ValueError("candidate owner marker does not match preparation intent")
    return payload


def _remove_candidate_owner(
    location: Path,
    *,
    candidate_root: Path,
    preparation: dict[str, Any],
    preparation_intent_path: Path,
    required: bool,
) -> dict[str, Any] | None:
    marker = location / CANDIDATE_OWNER_FILENAME
    if not marker.exists() and not marker.is_symlink():
        if required:
            raise ValueError("candidate owner marker is missing")
        return None
    payload = _validate_candidate_owner(
        location,
        candidate_root=candidate_root,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    marker.unlink()
    fsync_directory(location)
    return payload


def rename_noreplace(source: Path, target: Path) -> None:
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
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def rename_noreplace_at(
    source_dir_fd: int,
    source_name: str,
    target_dir_fd: int,
    target_name: str,
) -> None:
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
    if (
        renameat2(
            source_dir_fd,
            os.fsencode(source_name),
            target_dir_fd,
            os.fsencode(target_name),
            1,
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _create_owned_candidate(
    candidate: Path,
    *,
    preparation: dict[str, Any],
    preparation_intent_path: Path,
) -> None:
    if candidate.exists() or candidate.is_symlink():
        raise ValueError("candidate root must be absent")
    temporary = candidate.with_name(f"{candidate.name}.{uuid.uuid4().hex}.g003-preparing")
    temporary.mkdir(mode=0o700)
    fsync_directory(temporary.parent)
    _write_candidate_owner(
        temporary,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    fsync_directory(temporary)
    rename_noreplace(temporary, candidate)
    fsync_directory(candidate.parent)
    _validate_candidate_owner(
        candidate,
        candidate_root=candidate,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )


def _quarantine_owned_candidate(
    candidate: Path,
    *,
    preparation: dict[str, Any],
    preparation_intent_path: Path,
) -> tuple[Path, dict[str, int], str]:
    expected_identity = directory_identity(candidate)
    marker = candidate / CANDIDATE_OWNER_FILENAME
    stable_regular(marker)
    expected_marker_sha256 = file_sha256(marker)
    _validate_candidate_owner(
        candidate,
        candidate_root=candidate,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    quarantine = candidate.with_name(f"{candidate.name}.{uuid.uuid4().hex}.g003-quarantine")
    rename_noreplace(candidate, quarantine)
    fsync_directory(candidate.parent)
    try:
        if directory_identity(quarantine) != expected_identity:
            raise ValueError("quarantined candidate identity changed")
        _validate_candidate_owner(
            quarantine,
            candidate_root=candidate,
            preparation=preparation,
            preparation_intent_path=preparation_intent_path,
        )
        stable_regular(quarantine / CANDIDATE_OWNER_FILENAME)
        if file_sha256(quarantine / CANDIDATE_OWNER_FILENAME) != expected_marker_sha256:
            raise ValueError("quarantined candidate owner marker changed")
    except BaseException as exc:
        raise ValueError(f"untrusted quarantine preserved at {quarantine}") from exc
    return quarantine, expected_identity, expected_marker_sha256


def _retire_directory_contents(
    directory_fd: int,
    *,
    preserve: frozenset[str] = frozenset(),
) -> None:
    for entry in list(os.scandir(directory_fd)):
        name = entry.name
        if name in preserve:
            continue
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        tombstone = f".g003-retire-{uuid.uuid4().hex}"
        rename_noreplace_at(directory_fd, name, directory_fd, tombstone)
        moved = os.stat(tombstone, dir_fd=directory_fd, follow_symlinks=False)
        if inode_identity(moved) != inode_identity(before):
            raise ValueError(f"quarantine entry changed while moved: {name}")
        if stat.S_ISDIR(moved.st_mode):
            child_fd = os.open(
                tombstone,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            try:
                if inode_identity(os.fstat(child_fd)) != inode_identity(moved):
                    raise ValueError(f"quarantine directory changed while opened: {name}")
                _retire_directory_contents(child_fd)
                current = os.stat(
                    tombstone,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if inode_identity(current) != inode_identity(moved):
                    raise ValueError(f"quarantine directory changed before removal: {name}")
                os.rmdir(tombstone, dir_fd=directory_fd)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(moved.st_mode) or stat.S_ISLNK(moved.st_mode):
            current = os.stat(
                tombstone,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if inode_identity(current) != inode_identity(moved):
                raise ValueError(f"quarantine file changed before removal: {name}")
            os.unlink(tombstone, dir_fd=directory_fd)
        else:
            raise ValueError(f"unsupported quarantine entry preserved: {name}")
    os.fsync(directory_fd)


def _retire_owned_quarantine(
    quarantine: Path,
    *,
    expected_identity: dict[str, int],
    expected_marker_sha256: str,
) -> None:
    directory_fd = os.open(
        quarantine,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        opened = os.fstat(directory_fd)
        if {
            "dev": int(opened.st_dev),
            "ino": int(opened.st_ino),
            "mode": int(stat.S_IMODE(opened.st_mode)),
            "uid": int(opened.st_uid),
            "gid": int(opened.st_gid),
            "nlink": int(opened.st_nlink),
        } != expected_identity:
            raise ValueError("quarantine changed before descriptor pin")
        marker_fd = os.open(
            CANDIDATE_OWNER_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            marker_info = os.fstat(marker_fd)
            if (
                not stat.S_ISREG(marker_info.st_mode)
                or marker_info.st_nlink != 1
                or file_sha256_fd(marker_fd) != expected_marker_sha256
            ):
                raise ValueError("quarantine owner marker changed before retirement")
        finally:
            os.close(marker_fd)
        _retire_directory_contents(
            directory_fd,
            preserve=frozenset({CANDIDATE_OWNER_FILENAME}),
        )
        marker_fd = os.open(
            CANDIDATE_OWNER_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        try:
            if file_sha256_fd(marker_fd) != expected_marker_sha256:
                raise ValueError("quarantine owner marker changed during retirement")
        finally:
            os.close(marker_fd)
        current = os.lstat(quarantine)
        if inode_identity(current) != inode_identity(opened):
            raise ValueError("quarantine pathname changed during retirement")
    finally:
        os.close(directory_fd)
    fsync_directory(quarantine.parent)


def _supports_tree_reflink(candidate: Path) -> tuple[bool, str]:
    """Probe same-filesystem reflink support through pinned owned descriptors."""
    directory_fd = os.open(
        candidate,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    source_fd = target_fd = -1
    probe = f".g003-reflink-probe-{uuid.uuid4().hex}"
    probe_identity: tuple[int, int] | None = None
    supported = False
    diagnostic = ""
    try:
        source_fd = os.open(
            CANDIDATE_OWNER_FILENAME,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        source_info = os.fstat(source_fd)
        if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
            raise ValueError("unsafe reflink probe source")
        target_fd = os.open(
            probe,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        target_info = os.fstat(target_fd)
        probe_identity = (target_info.st_dev, target_info.st_ino)
        try:
            fcntl.ioctl(target_fd, FICLONE, source_fd)
            os.fsync(target_fd)
            cloned = os.fstat(target_fd)
            if cloned.st_size != source_info.st_size or file_sha256_fd(target_fd) != file_sha256_fd(
                source_fd
            ):
                raise ValueError("reflink probe content mismatch")
            supported = True
        except OSError as exc:
            if exc.errno not in {
                errno.EINVAL,
                errno.ENOSYS,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            diagnostic = f"{type(exc).__name__}: {exc}"
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        if source_fd >= 0:
            os.close(source_fd)
        if probe_identity is not None:
            named = os.stat(probe, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(named.st_mode)
                or named.st_nlink != 1
                or (named.st_dev, named.st_ino) != probe_identity
            ):
                raise ValueError("unsafe reflink probe output")
            os.unlink(probe, dir_fd=directory_fd)
            os.fsync(directory_fd)
        os.close(directory_fd)
    return supported, diagnostic


def _candidate_copy_command(source: Path, candidate: Path, reflink: str) -> list[str]:
    entries = sorted(source.iterdir(), key=lambda path: path.name)
    if not entries:
        raise ValueError("canonical root is empty")
    if any(path.name == CANDIDATE_OWNER_FILENAME for path in entries):
        raise ValueError("canonical root contains a reserved candidate owner marker")
    return [
        "cp",
        "-a",
        f"--reflink={reflink}",
        "--",
        *(str(path) for path in entries),
        str(candidate),
    ]


def clone_canonical_tree(
    source: Path,
    candidate: Path,
    *,
    preparation: dict[str, Any],
    preparation_intent_path: Path,
) -> str:
    if candidate.exists() or candidate.is_symlink():
        raise ValueError("candidate root must be absent")
    _create_owned_candidate(
        candidate,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    reflink_supported, reflink_diagnostic = _supports_tree_reflink(candidate)
    command = _candidate_copy_command(source, candidate, "always")
    if reflink_supported:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    else:
        result = subprocess.CompletedProcess(command, 1, "", reflink_diagnostic)
    if result.returncode == 0:
        _validate_candidate_owner(
            candidate,
            candidate_root=candidate,
            preparation=preparation,
            preparation_intent_path=preparation_intent_path,
        )
        fsync_directory(candidate)
        return "reflink"
    if reflink_supported:
        quarantine, quarantine_identity, quarantine_marker_sha256 = _quarantine_owned_candidate(
            candidate,
            preparation=preparation,
            preparation_intent_path=preparation_intent_path,
        )
        _retire_owned_quarantine(
            quarantine,
            expected_identity=quarantine_identity,
            expected_marker_sha256=quarantine_marker_sha256,
        )
        _create_owned_candidate(
            candidate,
            preparation=preparation,
            preparation_intent_path=preparation_intent_path,
        )
    fallback = subprocess.run(
        _candidate_copy_command(source, candidate, "auto"),
        check=False,
        capture_output=True,
        text=True,
    )
    if fallback.returncode != 0:
        raise RuntimeError(
            "candidate clone failed: "
            + (result.stderr.strip() or result.stdout.strip())
            + "; fallback: "
            + (fallback.stderr.strip() or fallback.stdout.strip())
        )
    _validate_candidate_owner(
        candidate,
        candidate_root=candidate,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    fsync_directory(candidate)
    return "copy-fallback"


def _discard_interrupted_preparation(
    args: argparse.Namespace,
    *,
    candidate_root: Path,
    canonical_before: dict[str, int],
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    preparation_intent_path = args.receipt_dir / PREPARATION_INTENT_FILENAME
    if (
        not preparation_intent_path.exists()
        or preparation_intent_path.is_symlink()
        or not preparation_intent_path.is_file()
    ):
        raise ValueError("candidate root lacks a proven preparation intent")
    stable_regular(preparation_intent_path)
    owner_path = candidate_root / CANDIDATE_OWNER_FILENAME
    if not owner_path.exists() or owner_path.is_symlink():
        raise ValueError("candidate root lacks a proven owner marker")
    preparation = _load_preparation_intent(
        args,
        candidate_root=candidate_root,
        source_receipt=source_receipt,
    )
    if preparation["canonical_before"] != canonical_before:
        raise ValueError("interrupted preparation canonical identity changed")
    owner = _validate_candidate_owner(
        candidate_root,
        candidate_root=candidate_root,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    stale_artifacts: dict[str, dict[str, Any]] = {}
    for name in ("candidate_inventory.json", PREPARATION_READY_FILENAME):
        path = args.receipt_dir / name
        if not path.exists() and not path.is_symlink():
            continue
        info = stable_regular(path)
        stale_artifacts[name] = {
            "sha256": file_sha256(path),
            "byte_count": int(info.st_size),
        }
    quarantine, removed_identity, owner_sha256 = _quarantine_owned_candidate(
        candidate_root,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    _retire_owned_quarantine(
        quarantine,
        expected_identity=removed_identity,
        expected_marker_sha256=owner_sha256,
    )
    for name in stale_artifacts:
        path = args.receipt_dir / name
        stable_regular(path)
        path.unlink()
    if stale_artifacts:
        fsync_directory(args.receipt_dir)
    recovery = {
        "artifact_kind": "alpha_max_direct_db_preparation_recovery",
        "schema": "alpha_max_direct_db_preparation_recovery.v1",
        "candidate_id": args.candidate_id,
        "candidate_root": str(candidate_root),
        "quarantine_path": str(quarantine),
        "quarantine_retained": True,
        "candidate_nonce": preparation["candidate_nonce"],
        "canonical_identity": canonical_before,
        "removed_candidate_identity": removed_identity,
        "preparation_intent_sha256": file_sha256(preparation_intent_path),
        "owner_marker_sha256": owner_sha256,
        "owner": owner,
        "removed_stale_artifacts": stale_artifacts,
        "rebuild_required": True,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(
        args.receipt_dir / "preparation_recovery.json",
        recovery,
        canonical=True,
    )
    return recovery


def _write_preparation_ready(
    args: argparse.Namespace,
    *,
    candidate_root: Path,
    candidate_identity: dict[str, int],
    preparation: dict[str, Any],
    prepared: dict[str, Any],
    candidate_audit_path: Path,
) -> dict[str, Any]:
    preparation_intent_path = args.receipt_dir / PREPARATION_INTENT_FILENAME
    owner_path = candidate_root / CANDIDATE_OWNER_FILENAME
    stable_regular(owner_path)
    _validate_candidate_owner(
        candidate_root,
        candidate_root=candidate_root,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    payload = {
        "artifact_kind": "alpha_max_direct_db_preparation_ready",
        "schema": "alpha_max_direct_db_preparation_ready.v1",
        "candidate_id": args.candidate_id,
        "candidate_nonce": preparation["candidate_nonce"],
        "canonical_root": str(args.canonical_root),
        "candidate_root": str(candidate_root),
        "candidate_identity": candidate_identity,
        "contract_sha256": CONTRACT_SHA256,
        "source": preparation["source"],
        "preparation_intent_sha256": file_sha256(preparation_intent_path),
        "owner_marker_sha256": file_sha256(owner_path),
        "candidate_audit_sha256": file_sha256(candidate_audit_path),
        "prepared": {
            "raw_partition_count": prepared["raw_partition_count"],
            "funding_partition_count": prepared["funding_partition_count"],
            "raw_seal_count": prepared["raw_seal_count"],
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(
        args.receipt_dir / PREPARATION_READY_FILENAME,
        payload,
        canonical=True,
    )
    return payload


def _load_preparation_ready(
    args: argparse.Namespace,
    *,
    candidate_root: Path,
    source_receipt: dict[str, str],
    preparation: dict[str, Any],
    candidate_audit_path: Path,
) -> dict[str, Any]:
    path = args.receipt_dir / PREPARATION_READY_FILENAME
    payload = ParquetMarketDataRepository._read_canonical_json(path)
    expected_keys = {
        "artifact_kind",
        "schema",
        "candidate_id",
        "candidate_nonce",
        "canonical_root",
        "candidate_root",
        "candidate_identity",
        "contract_sha256",
        "source",
        "preparation_intent_sha256",
        "owner_marker_sha256",
        "candidate_audit_sha256",
        "prepared",
        "created_at_utc",
    }
    prepared = payload.get("prepared")
    if (
        set(payload) != expected_keys
        or payload["artifact_kind"] != "alpha_max_direct_db_preparation_ready"
        or payload["schema"] != "alpha_max_direct_db_preparation_ready.v1"
        or payload["candidate_id"] != args.candidate_id
        or payload["candidate_nonce"] != preparation["candidate_nonce"]
        or payload["canonical_root"] != str(args.canonical_root)
        or payload["candidate_root"] != str(candidate_root)
        or not isinstance(payload["candidate_identity"], dict)
        or payload["contract_sha256"] != CONTRACT_SHA256
        or payload["source"] != source_receipt
        or payload["preparation_intent_sha256"]
        != file_sha256(args.receipt_dir / PREPARATION_INTENT_FILENAME)
        or payload["candidate_audit_sha256"] != file_sha256(candidate_audit_path)
        or not isinstance(payload["owner_marker_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", payload["owner_marker_sha256"])
        or prepared
        != {
            "raw_partition_count": EXPECTED_RAW_PARTITIONS,
            "funding_partition_count": EXPECTED_FUNDING_PARTITIONS,
            "raw_seal_count": EXPECTED_RAW_PARTITIONS,
        }
        or not isinstance(payload["created_at_utc"], str)
        or not payload["created_at_utc"]
    ):
        raise ValueError("durable preparation-ready receipt is invalid")
    return payload


def merge_raw_contract_partition(
    source: Path,
    target: Path,
    *,
    symbol: str,
    month: str,
    start_ms: int,
    end_ms: int,
    nominal_start_ms: int,
    nominal_end_ms: int,
) -> dict[str, Any]:
    partial_partition = start_ms != nominal_start_ms or end_ms != nominal_end_ms
    if not partial_partition:
        receipt = clone_file_atomic(source, target)
        receipt.update({"preserved_rows": 0, "source_rows": (end_ms - start_ms) // 1000})
        return receipt

    source_audit = audit_raw_partition(
        source,
        symbol=symbol,
        month=month,
        start_ms=start_ms,
        end_ms=end_ms,
        deep=True,
    )
    if (
        source_audit["status"] != "complete"
        or source_audit["outside_contract_rows"] != 0
        or source_audit["total_rows"] != source_audit["expected_rows"]
    ):
        raise ValueError(f"partial source partition is not exact: {source}")

    if not target.exists() and not target.is_symlink():
        receipt = clone_file_atomic(source, target)
        receipt.update({"preserved_rows": 0, "source_rows": source_audit["actual_rows"]})
        return receipt

    stable_regular(target)
    existing = pl.read_parquet(target)
    if existing.schema != RAW_SCHEMA:
        raise ValueError(f"existing raw partition schema mismatch: {target}")
    timestamp_ms = pl.col("datetime").dt.epoch("ms")
    preserved = existing.filter((timestamp_ms < start_ms) | (timestamp_ms >= end_ms))
    source_frame = pl.read_parquet(source)
    merged = pl.concat([preserved, source_frame], how="vertical").sort("datetime")
    receipt = write_frame_atomic(merged, target)
    post_audit = audit_raw_partition(
        target,
        symbol=symbol,
        month=month,
        start_ms=start_ms,
        end_ms=end_ms,
        deep=True,
    )
    if (
        post_audit["status"] != "complete"
        or post_audit["outside_contract_rows"] != preserved.height
        or post_audit["total_rows"] != preserved.height + source_frame.height
    ):
        raise ValueError(f"merged raw partition failed verification: {target}")
    receipt.update(
        {
            "source": str(source),
            "source_sha256": file_sha256(source),
            "source_rows": source_frame.height,
            "preserved_rows": preserved.height,
        }
    )
    return receipt


def publish_funding_contract_file(
    source: Path,
    *,
    relative: Path,
    source_report: Path,
    repository: FeatureMarketDataRepository,
) -> dict[str, Any]:
    source_info = stable_regular(source)
    source_sha256 = file_sha256(source)
    relative_text = relative.as_posix()
    provenance_path = acquisition.partition_path(source_report, relative_text)
    partition_receipt = json.loads(provenance_path.read_text())
    if (
        partition_receipt.get("schema") != "alpha_max_partition_receipt.v2"
        or partition_receipt.get("path") != relative_text
        or partition_receipt.get("output_sha256") != source_sha256
        or not isinstance(partition_receipt.get("rows"), int)
        or partition_receipt["rows"] <= 0
    ):
        raise ValueError(f"funding partition receipt mismatch: {source}")
    path_parts = relative.parts
    symbol = path_parts[2].removeprefix("symbol=")
    day = path_parts[3].removeprefix("date=")
    provenance_sha256 = file_sha256(provenance_path)
    target = repository.publish_official_funding_day(
        exchange="binance",
        symbol=symbol,
        day=day,
        source=source,
        expected_sha256=source_sha256,
        expected_byte_count=int(source_info.st_size),
        expected_row_count=int(partition_receipt["rows"]),
        provenance_receipt_sha256=provenance_sha256,
    )
    target_info = stable_regular(target)
    return {
        "source": str(source),
        "source_sha256": source_sha256,
        "source_rows": int(partition_receipt["rows"]),
        "source_byte_count": int(source_info.st_size),
        "provenance_receipt": str(provenance_path),
        "provenance_receipt_sha256": provenance_sha256,
        "target": str(target),
        "target_sha256": file_sha256(target),
        "target_byte_count": int(target_info.st_size),
        "copy_mode": "canonical-feature-merge",
    }


def filter_wal_contract(
    candidate_root: Path,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, int]:
    wal_path = candidate_root / "market_ohlcv_1s" / "binance" / symbol / "wal.bin"
    if not wal_path.exists():
        return {"records_before": 0, "records_removed": 0, "records_after": 0}
    info = stable_regular(wal_path)
    records = list(BinaryWAL(wal_path, auto_repair=False).iter_all())
    if len(records) * 64 != info.st_size:
        raise ValueError(f"WAL contains invalid or partial records: {wal_path}")
    kept = [record for record in records if not start_ms <= record.ts_ms < end_ms]
    temporary = wal_path.with_name(f".{wal_path.name}.{uuid.uuid4().hex}.g003.tmp")
    try:
        rewritten = BinaryWAL(temporary, fsync_every_n_batches=1, auto_repair=False)
        rewritten.append(kept)
        rewritten.force_fsync()
        if temporary.stat().st_size != len(kept) * 64:
            raise ValueError("rewritten WAL byte count is invalid")
        os.replace(temporary, wal_path)
        fsync_directory(wal_path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            fsync_directory(temporary.parent)
    meta_path = wal_path.parent / "compaction.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["wal_offset"] = 0
        meta["updated_at"] = datetime.now(UTC).isoformat()
        meta["g003_contract_rows_removed_from_wal"] = len(records) - len(kept)
        atomic_json(meta_path, meta)
    return {
        "records_before": len(records),
        "records_removed": len(records) - len(kept),
        "records_after": len(kept),
    }


def scrub_funding_features(
    candidate_root: Path,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, int]:
    root = candidate_root / "feature_points" / "exchange=binance" / f"symbol={symbol}"
    files_seen = files_rewritten = values_cleared = 0
    for path in sorted(root.glob("date=*/*.parquet")) if root.exists() else []:
        if path.name == "funding.parquet":
            continue
        files_seen += 1
        schema = pl.read_parquet_schema(path)
        columns = [name for name in FUNDING_FEATURE_COLUMNS if name in schema]
        if not columns or "timestamp_ms" not in schema:
            continue
        frame = pl.read_parquet(path)
        in_window = (pl.col("timestamp_ms") >= start_ms) & (pl.col("timestamp_ms") < end_ms)
        counts = frame.select(
            [(in_window & pl.col(name).is_not_null()).sum().alias(name) for name in columns]
        ).row(0)
        count = sum(int(value) for value in counts)
        if count == 0:
            continue
        rewritten = frame.with_columns(
            [pl.when(in_window).then(None).otherwise(pl.col(name)).alias(name) for name in columns]
        )
        write_frame_atomic(rewritten, path)
        files_rewritten += 1
        values_cleared += count
    return {
        "files_seen": files_seen,
        "files_rewritten": files_rewritten,
        "values_cleared": values_cleared,
    }


def rename_exchange(first: Path, second: Path) -> None:
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
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def raw_contract_partitions(contract: dict[str, Any]):
    for record in contract["records"]:
        symbol = str(record["symbol"])
        contract_start = utc_ms(record["raw_availability_start_utc"])
        contract_end = utc_ms(record["raw_availability_end_utc"])
        for month in month_starts(contract_start, contract_end):
            label = month.strftime("%Y-%m")
            nominal_start, nominal_end = month_bounds(month)
            yield {
                "symbol": symbol,
                "month": label,
                "start_ms": max(contract_start, nominal_start),
                "end_ms": min(contract_end, nominal_end),
                "nominal_start_ms": nominal_start,
                "nominal_end_ms": nominal_end,
            }


def prepare_candidate(
    *,
    contract: dict[str, Any],
    source_root: Path,
    source_report: Path,
    canonical_root: Path,
    candidate_root: Path,
    preparation: dict[str, Any],
    preparation_intent_path: Path,
) -> dict[str, Any]:
    clone_mode = clone_canonical_tree(
        canonical_root,
        candidate_root,
        preparation=preparation,
        preparation_intent_path=preparation_intent_path,
    )
    feature_repository = FeatureMarketDataRepository(str(candidate_root))
    raw_receipts = []
    wal_receipts = []
    funding_scrubs = []
    records_by_symbol = {str(record["symbol"]): record for record in contract["records"]}
    for record in contract["records"]:
        symbol = str(record["symbol"])
        wal_receipts.append(
            {
                "symbol": symbol,
                **filter_wal_contract(
                    candidate_root,
                    symbol=symbol,
                    start_ms=utc_ms(record["raw_availability_start_utc"]),
                    end_ms=utc_ms(record["raw_availability_end_utc"]),
                ),
            }
        )
        funding_scrubs.append(
            {
                "symbol": symbol,
                **scrub_funding_features(
                    candidate_root,
                    symbol=symbol,
                    start_ms=utc_ms(record["feature_availability_start_utc"]),
                    end_ms=utc_ms(record["feature_availability_end_utc"]),
                ),
            }
        )
    for item in raw_contract_partitions(contract):
        relative = Path("market_ohlcv_1s") / "binance" / item["symbol"] / f"{item['month']}.parquet"
        for suffix in (".pending.json", ".seal.json"):
            control = (candidate_root / relative).with_suffix(suffix)
            if control.exists() or control.is_symlink():
                raise ValueError(f"managed partition control already exists: {control}")
        receipt = merge_raw_contract_partition(
            source_root / relative,
            candidate_root / relative,
            symbol=item["symbol"],
            month=item["month"],
            start_ms=item["start_ms"],
            end_ms=item["end_ms"],
            nominal_start_ms=item["nominal_start_ms"],
            nominal_end_ms=item["nominal_end_ms"],
        )
        receipt.update(item)
        raw_receipts.append(receipt)
    funding_receipts = []
    source_feature_root = source_root / "feature_points" / "exchange=binance"
    for source in sorted(source_feature_root.glob("symbol=*/date=*/funding.parquet")):
        relative = source.relative_to(source_root)
        symbol = relative.parts[2].removeprefix("symbol=")
        if symbol not in records_by_symbol:
            raise ValueError(f"unexpected source funding symbol: {symbol}")
        funding_receipts.append(
            publish_funding_contract_file(
                source,
                relative=relative,
                source_report=source_report,
                repository=feature_repository,
            )
        )
    return {
        "clone_mode": clone_mode,
        "raw_partition_count": len(raw_receipts),
        "raw_partitions": raw_receipts,
        "funding_partition_count": len(funding_receipts),
        "funding_partitions": funding_receipts,
        "wal_filters": wal_receipts,
        "funding_scrubs": funding_scrubs,
    }


def seal_candidate_raw_partitions(
    *,
    prepared: dict[str, Any],
    candidate_audit: dict[str, Any],
    candidate_root: Path,
    source_report: Path,
) -> list[dict[str, Any]]:
    audited = {(item["symbol"], item["month"]): item for item in candidate_audit["raw"]["items"]}
    seals: list[dict[str, Any]] = []
    for receipt in prepared["raw_partitions"]:
        symbol = str(receipt["symbol"])
        month = str(receipt["month"])
        item = audited[(symbol, month)]
        if item["status"] != "complete" or any((item["deep_failures"] or {}).values()):
            raise ValueError(f"cannot seal raw partition without a deep audit: {symbol} {month}")
        target = Path(receipt["target"])
        source = Path(receipt["source"])
        relative = target.relative_to(candidate_root)
        provenance_path = acquisition.partition_path(source_report, relative.as_posix())
        provenance = json.loads(provenance_path.read_text())
        source_sha256 = file_sha256(source)
        target_sha256 = file_sha256(target)
        target_info = stable_regular(target)
        if (
            provenance.get("schema") != "alpha_max_partition_receipt.v2"
            or provenance.get("path") != relative.as_posix()
            or provenance.get("output_sha256") != source_sha256
            or provenance.get("rows") != receipt["source_rows"]
            or target_sha256 != receipt["sha256"]
            or target_info.st_size != receipt["byte_count"]
            or item["total_rows"] <= 0
        ):
            raise ValueError(f"raw partition provenance mismatch: {symbol} {month}")
        pending_path = target.with_suffix(".pending.json")
        seal_path = target.with_suffix(".seal.json")
        if (
            pending_path.exists()
            or pending_path.is_symlink()
            or seal_path.exists()
            or seal_path.is_symlink()
        ):
            raise ValueError(f"raw partition seal path is preseeded: {symbol} {month}")
        payload = {
            "schema": "alpha_max_canonical_partition_seal.v1",
            "relative_partition_path": relative.as_posix(),
            "sha256": target_sha256,
            "byte_count": int(target_info.st_size),
            "row_count": int(item["total_rows"]),
            "month": month,
            "exchange": "binance",
            "symbol": symbol,
            "provenance_receipt_sha256": file_sha256(provenance_path),
            "status": "sealed",
        }
        atomic_json(seal_path, payload, canonical=True)
        seals.append(
            {
                "symbol": symbol,
                "month": month,
                "path": str(seal_path),
                "sha256": file_sha256(seal_path),
                "partition_sha256": target_sha256,
                "row_count": int(item["total_rows"]),
                "provenance_receipt": str(provenance_path),
                "provenance_receipt_sha256": payload["provenance_receipt_sha256"],
            }
        )
    return seals


def _audit_is_exact(payload: dict[str, Any], *, require_deep: bool) -> bool:
    accepted_status = {"complete"} if require_deep else {"complete", "inventory-complete"}
    return (
        payload["status"] in accepted_status
        and payload["audit_mode"] == ("deep" if require_deep else "inventory")
        and payload["raw"]["target_partitions"] == EXPECTED_RAW_PARTITIONS
        and payload["raw"]["complete_partitions"] == EXPECTED_RAW_PARTITIONS
        and payload["raw"]["target_rows"] == EXPECTED_RAW_ROWS
        and payload["raw"]["complete_rows"] == EXPECTED_RAW_ROWS
        and payload["funding"]["target_rows"] == EXPECTED_FUNDING_ROWS
        and payload["funding"]["complete_rows"] == EXPECTED_FUNDING_ROWS
        and payload["funding"]["missing_rows"] == 0
        and payload["funding"]["extra_rows_in_window"] == 0
        and payload["funding"]["duplicate_rows_in_window"] == 0
        and payload["funding"]["jitter_violation_rows"] == 0
        and payload["funding"]["error_count"] == 0
    )


def _activate_candidate_transaction(
    args: argparse.Namespace,
    *,
    candidate_root: Path,
    canonical_before: dict[str, int],
    candidate_identity: dict[str, int],
    source_receipt: dict[str, str],
    capacity: dict[str, Any],
    free_after_prepare: int,
    candidate_audit_path: Path,
    intent_path: Path,
) -> dict[str, Any]:
    canonical_now = directory_identity(args.canonical_root)
    candidate_now = directory_identity(candidate_root)
    pre_exchange = canonical_now == canonical_before and candidate_now == candidate_identity
    post_exchange = canonical_now == candidate_identity and candidate_now == canonical_before
    if not pre_exchange and not post_exchange:
        raise ValueError("activation directories do not match the durable intent")
    for root in (args.canonical_root, candidate_root):
        marker = root / CANDIDATE_OWNER_FILENAME
        if marker.exists() or marker.is_symlink():
            raise ValueError("candidate owner marker must be removed before activation")

    exchanged = post_exchange
    rollback_complete = False
    try:
        if not exchanged:
            rename_exchange(args.canonical_root, candidate_root)
            exchanged = True
        fsync_directory(args.canonical_root.parent)
        sync_filesystem(args.canonical_root.parent)
        if (
            directory_identity(args.canonical_root) != candidate_identity
            or directory_identity(candidate_root) != canonical_before
        ):
            raise ValueError("post-exchange directory identity mismatch")
        active_audit = audit_contract(
            contract_path=args.contract,
            db_path=args.canonical_root,
            deep=False,
        )
        if not _audit_is_exact(active_audit, require_deep=False):
            raise ValueError("active canonical contract audit failed")
        active_audit_path = args.receipt_dir / "active_inventory.json"
        atomic_json(active_audit_path, active_audit)
        activation = {
            "artifact_kind": "alpha_max_direct_db_activation_receipt",
            "candidate_id": args.candidate_id,
            "contract_sha256": CONTRACT_SHA256,
            "canonical_root": str(args.canonical_root),
            "old_generation_root": str(candidate_root),
            "canonical_before": canonical_before,
            "canonical_after": directory_identity(args.canonical_root),
            "old_generation_after": directory_identity(candidate_root),
            "source": source_receipt,
            "capacity": capacity,
            "free_after_prepare": free_after_prepare,
            "intent_sha256": file_sha256(intent_path),
            "candidate_audit_sha256": file_sha256(candidate_audit_path),
            "active_audit_sha256": file_sha256(active_audit_path),
            "raw_partitions": active_audit["raw"]["complete_partitions"],
            "raw_rows": active_audit["raw"]["complete_rows"],
            "funding_rows": active_audit["funding"]["complete_rows"],
            "activated_at_utc": datetime.now(UTC).isoformat(),
            "atomic_exchange": True,
            "order_routing": False,
        }
        atomic_json(args.receipt_dir / "activation_receipt.json", activation)
        return activation
    except BaseException as exc:
        if (
            exchanged
            and args.canonical_root.exists()
            and candidate_root.exists()
            and directory_identity(args.canonical_root) == candidate_identity
        ):
            rename_exchange(args.canonical_root, candidate_root)
            fsync_directory(args.canonical_root.parent)
            sync_filesystem(args.canonical_root.parent)
            rollback_complete = (
                directory_identity(args.canonical_root) == canonical_before
                and directory_identity(candidate_root) == candidate_identity
            )
        activation_receipt_path = args.receipt_dir / "activation_receipt.json"
        if rollback_complete and activation_receipt_path.exists():
            activation_receipt_path.unlink()
            fsync_directory(activation_receipt_path.parent)
        atomic_json(
            args.receipt_dir / "activation_failure.json",
            {
                "artifact_kind": "alpha_max_direct_db_activation_failure",
                "candidate_id": args.candidate_id,
                "error": repr(exc),
                "exchanged": exchanged,
                "rollback_complete": rollback_complete,
                "canonical_identity": directory_identity(args.canonical_root),
                "candidate_identity": directory_identity(candidate_root),
                "recorded_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        if exchanged and not rollback_complete:
            raise RuntimeError("canonical rollback failed") from exc
        raise


def verify_raw_seal_snapshot(root: Path, audit: dict[str, Any]) -> None:
    if audit["raw"]["complete_partitions"] != EXPECTED_RAW_PARTITIONS:
        raise ValueError("raw seal verification requires the complete contract")
    for item in audit["raw"]["items"]:
        relative = Path("market_ohlcv_1s") / "binance" / item["symbol"] / f"{item['month']}.parquet"
        target = root / relative
        seal = target.with_suffix(".seal.json")
        pending = target.with_suffix(".pending.json")
        if pending.exists() or pending.is_symlink():
            raise ValueError(f"pending raw seal remains: {relative}")
        target_info = stable_regular(target)
        seal_info = stable_regular(seal)
        payload = ParquetMarketDataRepository._read_canonical_json(seal)
        expected_keys = {
            "schema",
            "relative_partition_path",
            "sha256",
            "byte_count",
            "row_count",
            "month",
            "exchange",
            "symbol",
            "provenance_receipt_sha256",
            "status",
        }
        if (
            set(payload) != expected_keys
            or payload["schema"] != "alpha_max_canonical_partition_seal.v1"
            or payload["relative_partition_path"] != relative.as_posix()
            or payload["sha256"] != file_sha256(target)
            or payload["byte_count"] != target_info.st_size
            or payload["row_count"] != item["total_rows"]
            or payload["month"] != item["month"]
            or payload["exchange"] != "binance"
            or payload["symbol"] != item["symbol"]
            or not isinstance(payload["provenance_receipt_sha256"], str)
            or len(payload["provenance_receipt_sha256"]) != 64
            or payload["status"] != "sealed"
            or seal_info.st_size <= 0
        ):
            raise ValueError(f"raw seal verification failed: {relative}")


def _existing_activation_receipt(
    args: argparse.Namespace,
    *,
    candidate_root: Path,
    source_receipt: dict[str, str],
) -> dict[str, Any] | None:
    path = args.receipt_dir / "activation_receipt.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    intent_path = args.receipt_dir / "activation_intent.json"
    if (
        value.get("artifact_kind") != "alpha_max_direct_db_activation_receipt"
        or value.get("candidate_id") != args.candidate_id
        or value.get("contract_sha256") != CONTRACT_SHA256
        or value.get("canonical_root") != str(args.canonical_root)
        or value.get("old_generation_root") != str(candidate_root)
        or value.get("source") != source_receipt
        or value.get("intent_sha256") != file_sha256(intent_path)
        or value.get("raw_partitions") != EXPECTED_RAW_PARTITIONS
        or value.get("raw_rows") != EXPECTED_RAW_ROWS
        or value.get("funding_rows") != EXPECTED_FUNDING_ROWS
        or value.get("canonical_after") != directory_identity(args.canonical_root)
        or value.get("old_generation_after") != directory_identity(candidate_root)
        or value.get("atomic_exchange") is not True
        or value.get("order_routing") is not False
    ):
        raise ValueError("existing activation receipt does not match current state")
    return value


def _recover_existing_candidate(
    args: argparse.Namespace,
    *,
    candidate_root: Path,
    source_receipt: dict[str, str],
) -> dict[str, Any]:
    existing = _existing_activation_receipt(
        args,
        candidate_root=candidate_root,
        source_receipt=source_receipt,
    )
    if existing is not None:
        return existing

    intent_path = args.receipt_dir / "activation_intent.json"
    preparation_intent_path = args.receipt_dir / PREPARATION_INTENT_FILENAME
    preparation_ready_path = args.receipt_dir / PREPARATION_READY_FILENAME
    candidate_audit_path = args.receipt_dir / "candidate_inventory.json"
    if (
        not intent_path.is_file()
        or not preparation_intent_path.is_file()
        or not preparation_ready_path.is_file()
        or not candidate_audit_path.is_file()
    ):
        raise ValueError("candidate root is preseeded without a complete durable intent")
    preparation = _load_preparation_intent(
        args,
        candidate_root=candidate_root,
        source_receipt=source_receipt,
    )
    preparation_ready = _load_preparation_ready(
        args,
        candidate_root=candidate_root,
        source_receipt=source_receipt,
        preparation=preparation,
        candidate_audit_path=candidate_audit_path,
    )
    intent = json.loads(intent_path.read_text())
    candidate_audit = json.loads(candidate_audit_path.read_text())
    canonical_before = intent.get("canonical_before")
    candidate_identity = intent.get("candidate_before")
    if (
        intent.get("artifact_kind") != "alpha_max_direct_db_activation_intent"
        or intent.get("candidate_id") != args.candidate_id
        or intent.get("candidate_nonce") != preparation["candidate_nonce"]
        or intent.get("canonical_root") != str(args.canonical_root)
        or intent.get("candidate_root") != str(candidate_root)
        or intent.get("contract_sha256") != CONTRACT_SHA256
        or intent.get("source") != source_receipt
        or intent.get("preparation_intent_sha256") != file_sha256(preparation_intent_path)
        or intent.get("preparation_ready_sha256") != file_sha256(preparation_ready_path)
        or intent.get("candidate_audit_sha256") != file_sha256(candidate_audit_path)
        or not isinstance(canonical_before, dict)
        or canonical_before != preparation["canonical_before"]
        or not isinstance(candidate_identity, dict)
        or candidate_identity != preparation_ready["candidate_identity"]
        or not isinstance(intent.get("capacity"), dict)
        or not isinstance(intent.get("free_after_prepare"), int)
        or not isinstance(intent.get("prepared"), dict)
        or intent["prepared"].get("raw_partition_count") != EXPECTED_RAW_PARTITIONS
        or intent["prepared"].get("funding_partition_count") != EXPECTED_FUNDING_PARTITIONS
        or intent["prepared"].get("raw_seal_count") != EXPECTED_RAW_PARTITIONS
        or not _audit_is_exact(candidate_audit, require_deep=True)
    ):
        raise ValueError("durable activation intent is invalid")

    canonical_now = directory_identity(args.canonical_root)
    candidate_now = directory_identity(candidate_root)
    pre_exchange = canonical_now == canonical_before and candidate_now == candidate_identity
    post_exchange = canonical_now == candidate_identity and candidate_now == canonical_before
    if not pre_exchange and not post_exchange:
        raise ValueError("interrupted activation identities are irreconcilable")
    candidate_location = args.canonical_root if post_exchange else candidate_root
    owner_path = candidate_location / CANDIDATE_OWNER_FILENAME
    if owner_path.exists() or owner_path.is_symlink():
        stable_regular(owner_path)
        if file_sha256(owner_path) != preparation_ready["owner_marker_sha256"]:
            raise ValueError("candidate owner marker changed after preparation")
        _remove_candidate_owner(
            candidate_location,
            candidate_root=candidate_root,
            preparation=preparation,
            preparation_intent_path=preparation_intent_path,
            required=True,
        )
        sync_filesystem(candidate_location)
    try:
        recovered_audit = audit_contract(
            contract_path=args.contract,
            db_path=candidate_location,
            deep=True,
        )
        if not _audit_is_exact(recovered_audit, require_deep=True):
            raise ValueError("interrupted candidate deep audit failed")
        verify_raw_seal_snapshot(candidate_location, recovered_audit)
    except BaseException as exc:
        rollback_complete = False
        if post_exchange:
            rename_exchange(args.canonical_root, candidate_root)
            fsync_directory(args.canonical_root.parent)
            sync_filesystem(args.canonical_root.parent)
            rollback_complete = (
                directory_identity(args.canonical_root) == canonical_before
                and directory_identity(candidate_root) == candidate_identity
            )
        atomic_json(
            args.receipt_dir / "activation_recovery_failure.json",
            {
                "artifact_kind": "alpha_max_direct_db_activation_recovery_failure",
                "candidate_id": args.candidate_id,
                "error": repr(exc),
                "post_exchange": post_exchange,
                "rollback_complete": rollback_complete,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        if post_exchange and not rollback_complete:
            raise RuntimeError("interrupted activation rollback failed") from exc
        raise

    return _activate_candidate_transaction(
        args,
        candidate_root=candidate_root,
        canonical_before=canonical_before,
        candidate_identity=candidate_identity,
        source_receipt=source_receipt,
        capacity=intent["capacity"],
        free_after_prepare=intent["free_after_prepare"],
        candidate_audit_path=candidate_audit_path,
        intent_path=intent_path,
    )


def _integrate_verified(
    args: argparse.Namespace,
    *,
    contract: dict[str, Any],
    candidate_root: Path,
) -> dict[str, Any]:
    eligible_path = args.source_report / "source_eligible_receipt.json"
    source_manifest = args.source_report / "source_manifest.json"
    source_receipt = {
        "eligible_path": str(eligible_path),
        "eligible_sha256": file_sha256(eligible_path),
        "manifest_path": str(source_manifest),
        "manifest_sha256": file_sha256(source_manifest),
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    canonical_resolved = args.canonical_root.resolve(strict=True)
    source_resolved = args.source_root.resolve(strict=True)
    receipt_resolved = args.receipt_dir.resolve(strict=True)
    for first, second in (
        (canonical_resolved, source_resolved),
        (canonical_resolved, receipt_resolved),
        (source_resolved, receipt_resolved),
    ):
        if first == second or first in second.parents or second in first.parents:
            raise ValueError("canonical, source, and receipt roots must be disjoint")

    repository = ParquetMarketDataRepository(args.canonical_root)
    with repository.generation_lock(exclusive=True, timeout_seconds=600):
        canonical_before = directory_identity(args.canonical_root)
        source_identity = directory_identity(args.source_root)
        if canonical_before["dev"] != source_identity["dev"]:
            raise ValueError("source and canonical roots must share one filesystem")
        if candidate_root.is_symlink():
            raise ValueError("candidate root must not be a symlink")
        activation_intent_path = args.receipt_dir / "activation_intent.json"
        activation_receipt_path = args.receipt_dir / "activation_receipt.json"
        preparation_intent_path = args.receipt_dir / PREPARATION_INTENT_FILENAME
        preparation_ready_path = args.receipt_dir / PREPARATION_READY_FILENAME
        candidate_audit_path = args.receipt_dir / "candidate_inventory.json"
        activation_artifacts_exist = (
            activation_intent_path.exists()
            or activation_intent_path.is_symlink()
            or activation_receipt_path.exists()
            or activation_receipt_path.is_symlink()
        )
        recovered_preparation = False
        if candidate_root.exists():
            if activation_artifacts_exist:
                return _recover_existing_candidate(
                    args,
                    candidate_root=candidate_root,
                    source_receipt=source_receipt,
                )
            _discard_interrupted_preparation(
                args,
                candidate_root=candidate_root,
                canonical_before=canonical_before,
                source_receipt=source_receipt,
            )
            recovered_preparation = True
        if activation_artifacts_exist:
            raise ValueError("activation artifacts exist without their candidate root")
        for stale in (preparation_ready_path, candidate_audit_path):
            if stale.exists() or stale.is_symlink():
                raise ValueError("preparation artifacts exist without their candidate root")
        if recovered_preparation or (
            not preparation_intent_path.exists() and not preparation_intent_path.is_symlink()
        ):
            preparation = _create_preparation_intent(
                args,
                candidate_root=candidate_root,
                canonical_before=canonical_before,
                source_receipt=source_receipt,
            )
        else:
            preparation = _load_preparation_intent(
                args,
                candidate_root=candidate_root,
                source_receipt=source_receipt,
            )
            if preparation["canonical_before"] != canonical_before:
                raise ValueError("preparation intent canonical identity changed")
        capacity = capacity_audit(
            args.canonical_root,
            args.source_root,
            reserve_bytes=args.reserve_bytes,
        )
        if not capacity["passes"]:
            raise ValueError("insufficient free space for conservative candidate fallback")
        prepared = prepare_candidate(
            contract=contract,
            source_root=args.source_root,
            canonical_root=args.canonical_root,
            candidate_root=candidate_root,
            source_report=args.source_report,
            preparation=preparation,
            preparation_intent_path=preparation_intent_path,
        )
        if (
            prepared["raw_partition_count"] != EXPECTED_RAW_PARTITIONS
            or prepared["funding_partition_count"] != EXPECTED_FUNDING_PARTITIONS
        ):
            raise ValueError("prepared partition count does not match the approved contract")
        sync_filesystem(candidate_root)
        free_after_prepare = int(shutil.disk_usage(args.canonical_root.parent).free)
        if free_after_prepare < args.reserve_bytes:
            raise ValueError("post-prepare free space violates reserve")
        candidate_identity = directory_identity(candidate_root)
        candidate_audit = audit_contract(
            contract_path=args.contract,
            db_path=candidate_root,
            deep=True,
        )
        if not _audit_is_exact(candidate_audit, require_deep=True):
            raise ValueError("candidate contract audit failed")
        raw_seals = seal_candidate_raw_partitions(
            prepared=prepared,
            candidate_audit=candidate_audit,
            candidate_root=candidate_root,
            source_report=args.source_report,
        )
        prepared["raw_seal_count"] = len(raw_seals)
        prepared["raw_seals"] = raw_seals
        if len(raw_seals) != EXPECTED_RAW_PARTITIONS:
            raise ValueError("raw seal count does not match the approved contract")
        verify_raw_seal_snapshot(candidate_root, candidate_audit)
        sync_filesystem(candidate_root)
        atomic_json(candidate_audit_path, candidate_audit)
        preparation_ready = _write_preparation_ready(
            args,
            candidate_root=candidate_root,
            candidate_identity=candidate_identity,
            preparation=preparation,
            prepared=prepared,
            candidate_audit_path=candidate_audit_path,
        )
        intent = {
            "artifact_kind": "alpha_max_direct_db_activation_intent",
            "candidate_id": args.candidate_id,
            "candidate_nonce": preparation["candidate_nonce"],
            "canonical_root": str(args.canonical_root),
            "candidate_root": str(candidate_root),
            "canonical_before": canonical_before,
            "candidate_before": candidate_identity,
            "contract_sha256": CONTRACT_SHA256,
            "source": source_receipt,
            "capacity": capacity,
            "free_after_prepare": free_after_prepare,
            "prepared": prepared,
            "preparation_intent_sha256": file_sha256(preparation_intent_path),
            "preparation_ready_sha256": file_sha256(preparation_ready_path),
            "candidate_audit_sha256": file_sha256(candidate_audit_path),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        intent_path = args.receipt_dir / "activation_intent.json"
        atomic_json(intent_path, intent)
        owner_path = candidate_root / CANDIDATE_OWNER_FILENAME
        stable_regular(owner_path)
        if file_sha256(owner_path) != preparation_ready["owner_marker_sha256"]:
            raise ValueError("candidate owner marker changed before activation")
        _remove_candidate_owner(
            candidate_root,
            candidate_root=candidate_root,
            preparation=preparation,
            preparation_intent_path=preparation_intent_path,
            required=True,
        )
        sync_filesystem(candidate_root)
        return _activate_candidate_transaction(
            args,
            candidate_root=candidate_root,
            canonical_before=canonical_before,
            candidate_identity=candidate_identity,
            source_receipt=source_receipt,
            capacity=capacity,
            free_after_prepare=free_after_prepare,
            candidate_audit_path=candidate_audit_path,
            intent_path=intent_path,
        )


def integrate(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.contract,
        args.source_root,
        args.source_report,
        args.canonical_root,
        args.receipt_dir,
    ):
        if not path.is_absolute():
            raise ValueError("all paths must be absolute")
    if args.reserve_bytes < 0:
        raise ValueError("reserve bytes must be nonnegative")
    if not CANDIDATE_ID.fullmatch(args.candidate_id):
        raise ValueError("candidate ID is invalid")
    candidate_root = args.canonical_root.parent / (
        f".{args.canonical_root.name}.g003-{args.candidate_id}"
    )
    contract = load_contract(args.contract)
    contracts = acquisition.load_contract(args.contract)
    with acquisition.owned_run_lock(args.source_report, exclusive=True):
        acquisition.verify_eligible(args.source_root, args.source_report, contracts)
        return _integrate_verified(
            args,
            contract=contract,
            candidate_root=candidate_root,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--reserve-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "artifact_kind": "alpha_max_direct_db_activation_plan",
                    "contract_sha256": CONTRACT_SHA256,
                    "source_root": str(args.source_root),
                    "canonical_root": str(args.canonical_root),
                    "candidate_id": args.candidate_id,
                    "execute": False,
                },
                sort_keys=True,
            )
        )
        return 0
    result = integrate(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
