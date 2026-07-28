#!/usr/bin/env python3
"""Publish an authenticated Alpha-Max source into the shared parquet store only."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
import polars as pl

from lumina_quant.alpha_max_terminal_policy import verify_signed_receipt
from lumina_quant.market_data import MarketDataRepository
from lumina_quant.storage.parquet import ParquetMarketDataRepository

_STORAGE = {
    "host_reserve_path": "/mnt/c",
    "host_reserve_bytes": 21_474_836_480,
    "max_live_archives": 1,
    "archive_retention": "retired_after_double_derivation",
}
_MAX_JSON = 64 * 1024 * 1024
_HEX = set("0123456789abcdef")


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


def _json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise PublicationError(f"{label} is not canonical JSON")
    return value, raw


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


def _write_noreplace(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        existing, raw = _json(path, "publication receipt")
        if existing != value or raw != payload:
            raise PublicationError("publication receipt conflict")
        return
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
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


def _terminal(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, tuple[str, int]]]:
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
    return message, verified.key_id, command_bindings[1]


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
    """Digest stable candidate identities while excluding transaction evidence."""
    records = []
    for record in _inventory(root, require_private=False):
        if record["path"] == "commit.json" or record["path"].startswith(".alpha_max_publication/"):
            continue
        stable = {key: value for key, value in record.items() if key not in {"ctime_ns", "nlink"}}
        records.append(stable)
    return _inventory_digest(records)


def _clone_root(old: Path, candidate: Path, expected: list[dict[str, Any]]) -> None:
    """Hardlink-clone old and prove both trees against the bound preclone records."""
    candidate.mkdir(mode=0o700)

    def clone(source: Path, destination: Path) -> None:
        for entry in sorted(os.scandir(source), key=lambda item: item.name):
            src, dst = source / entry.name, destination / entry.name
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                dst.mkdir(mode=stat.S_IMODE(info.st_mode))
                clone(src, dst)
                _fsync_dir(dst)
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise PublicationError(f"canonical root contains hardlinked file: {src}")
                os.link(src, dst, follow_symlinks=False)
            else:
                raise PublicationError(f"canonical root contains unsafe artifact: {src}")

    clone(old, candidate)
    _fsync_dir(candidate)
    old_after = _inventory(old, require_private=False)
    candidate_after = _inventory(candidate, require_private=False)
    expected_by_path = {record["path"]: record for record in expected}
    if set(expected_by_path) != {record["path"] for record in old_after} or set(
        expected_by_path
    ) != {record["path"] for record in candidate_after}:
        raise PublicationError("clone tree has missing or extra entries")
    for record in old_after:
        bound = expected_by_path[record["path"]]
        if record["kind"] != bound["kind"] or (record["dev"], record["ino"]) != (
            bound["dev"],
            bound["ino"],
        ):
            raise PublicationError("canonical root changed while cloning")
        if record["kind"] == "file" and record["nlink"] != bound["nlink"] + 1:
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
    if set(bound) != {item["path"] for item in actual}:
        raise PublicationError("retired root has missing or extra entries")
    for item in actual:
        prior = bound[item["path"]]
        if item["kind"] != prior["kind"] or (item["dev"], item["ino"]) != (
            prior["dev"],
            prior["ino"],
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


def _parquet_rows(path: Path) -> int:
    try:
        return int(pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item())
    except Exception as exc:
        raise PublicationError(f"published target is not readable parquet: {path}") from exc


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
    if (
        final.get("schema") != "alpha_max_canonical_publication_receipt.v2"
        or final.get("request_id") != request_id
        or final.get("terminal_receipt_sha256") != terminal_sha256
        or final.get("authority_key_id") != key_id
        or final.get("source_manifest_sha256") != hashes["manifest"]
        or final.get("source_eligible_receipt_sha256") != hashes["eligible"]
        or final.get("source_journal_sha256") != hashes["journal"]
        or final.get("contract_manifest_sha256") != hashes["contract"]
        or final.get("listing_records") != listing
        or final.get("canonical_root") is None
        or not isinstance(final.get("old_inventory"), list)
        or final.get("old_inventory_sha256") != _inventory_digest(final["old_inventory"])
    ):
        raise PublicationError("prepared commit binding mismatch")
    published = final.get("partitions")
    if not isinstance(published, list):
        raise PublicationError("prepared commit partition inventory is invalid")
    expected = {item["relative"]: item for item in parts}
    if set(expected) != {item.get("relative") for item in published if isinstance(item, dict)}:
        raise PublicationError("prepared commit partition inventory mismatch")
    for record in published:
        if set(record) != {
            "relative",
            "target",
            "source_sha256",
            "target_sha256",
            "bytes",
            "target_bytes",
            "rows",
            "target_rows",
            "provenance_receipt_sha256",
        }:
            raise PublicationError("prepared target record shape is invalid")
        source = expected[record["relative"]]
        if (
            record["source_sha256"] != source["sha256"]
            or record["rows"] != source["rows"]
            or record["provenance_receipt_sha256"] != source["provenance"]
        ):
            raise PublicationError("prepared target source binding mismatch")
        target = _inside(candidate, record["target"], "prepared target")
        size, digest = _file_sha(target)
        if (
            size != record["target_bytes"]
            or digest != record["target_sha256"]
            or _parquet_rows(target) != record["target_rows"]
        ):
            raise PublicationError("prepared target readback mismatch")
    if _data_inventory_digest(candidate) != final.get("candidate_data_inventory_sha256"):
        raise PublicationError("prepared candidate data inventory digest mismatch")
    return published


def _validate_cloned_predecessor(old: Path, old_inventory: list[dict[str, Any]]) -> None:
    actual = _inventory(old, require_private=False)
    expected = {record["path"]: record for record in old_inventory}
    if set(expected) != {record["path"] for record in actual}:
        raise PublicationError("prepared predecessor inventory changed")
    for record in actual:
        prior = expected[record["path"]]
        if record["kind"] != prior["kind"] or (record["dev"], record["ino"]) != (
            prior["dev"],
            prior["ino"],
        ):
            raise PublicationError("prepared predecessor identity changed")
        if record["kind"] == "file" and record["nlink"] != prior["nlink"] + 1:
            raise PublicationError("prepared predecessor link count changed")


def _cleanup_predecessor(
    *,
    temporary: Path,
    final: dict[str, Any],
) -> None:
    old_identity_value = final.get("predecessor_identity")
    old_inventory = final.get("old_inventory")
    predecessor_value = final.get("predecessor_path")
    if (
        not isinstance(old_identity_value, list)
        or len(old_identity_value) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in old_identity_value
        )
        or not isinstance(old_inventory, list)
        or not isinstance(predecessor_value, str)
    ):
        raise PublicationError("prepared predecessor binding is invalid")
    old_identity = (old_identity_value[0], old_identity_value[1])
    predecessor = Path(predecessor_value)
    if temporary.exists() and not temporary.is_symlink():
        retired = temporary
    elif temporary.is_symlink():
        target = os.readlink(temporary)
        resolved = temporary.parent / target
        if resolved != predecessor:
            raise PublicationError("retired generation symlink changed")
        temporary.unlink()
        _fsync_dir(temporary.parent)
        retired = predecessor
    elif predecessor.exists() and not predecessor.is_symlink():
        retired = predecessor
    else:
        return
    _remove_bound_tree(retired, old_identity, old_inventory)
    _fsync_dir(retired.parent)


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
    args = parser.parse_args(argv)
    for name in (
        "source_root",
        "source_report",
        "terminal_receipt",
        "authority_public_key",
        "canonical_root",
    ):
        setattr(args, name, _absolute(getattr(args, name), name))
    terminal, key_id, bound = _terminal(args)
    parts, hashes, listing = _partitions(args.source_root, args.source_report, bound)
    request_id = terminal.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise PublicationError("terminal request identity is invalid")
    terminal_sha256 = hashlib.sha256(
        _regular_bytes(args.terminal_receipt, "terminal receipt")
    ).hexdigest()
    generations = args.canonical_root.parent / f".{args.canonical_root.name}.generations"
    control = args.canonical_root.parent / f".{args.canonical_root.name}.transactions" / request_id
    candidate = generations / request_id
    lock_repo = ParquetMarketDataRepository(args.canonical_root)
    with lock_repo.generation_lock(exclusive=True):
        generations.mkdir(mode=0o700, exist_ok=True)
        try:
            active = _active_root(args.canonical_root, generations)
        except FileNotFoundError:
            active = None

        if active == candidate:
            final, _ = _json(candidate / "commit.json", "active commit")
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
            swap, _ = _json(control / "swap.json", "swap receipt")
            temporary = args.canonical_root.parent / swap["temporary"]
            _cleanup_predecessor(temporary=temporary, final=final)
            _inventory(candidate)
            _write_noreplace(
                control / "completed.json",
                {"request_id": request_id, "phase": "completed"},
            )
            return 0

        if candidate.exists():
            if not (candidate / "commit.json").exists() or not (control / "prepared.json").exists():
                raise PublicationError("incomplete candidate requires bounded operator recovery")
            final, _ = _json(candidate / "commit.json", "prepared commit")
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
            if active is None:
                raise PublicationError("prepared predecessor is missing")
            old = active
            old_inventory = final["old_inventory"]
            _validate_cloned_predecessor(old, old_inventory)
        else:
            root_missing = active is None
            if root_missing:
                old = generations / f".bootstrap-{uuid.uuid4().hex}"
                old.mkdir(mode=0o700)
            else:
                old = active
            old_identity = (os.stat(old).st_dev, os.stat(old).st_ino)
            old_inventory = _inventory(old)
            control.mkdir(parents=True, mode=0o700)
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
            repo = ParquetMarketDataRepository(candidate)
            feature_repo = MarketDataRepository(str(candidate))
            published = []
            for item in parts:
                byte_count, before = _hash_source(item["source"], item["sha256"])
                free = (
                    os.statvfs(_STORAGE["host_reserve_path"]).f_bavail
                    * os.statvfs(_STORAGE["host_reserve_path"]).f_frsize
                )
                if free < _STORAGE["host_reserve_bytes"] + byte_count:
                    raise PublicationError("host reserve would be violated before source copy")
                path = PurePosixPath(item["relative"])
                if path.parts[0] == "market_ohlcv_1s":
                    target = repo.merge_signed_month_into_candidate(
                        exchange="binance",
                        symbol=path.parts[2],
                        month=path.parts[3][:-8],
                        source=item["source"],
                        expected_sha256=item["sha256"],
                        expected_byte_count=byte_count,
                        expected_row_count=item["rows"],
                        provenance_receipt_sha256=item["provenance"],
                    )
                else:
                    target = feature_repo.publish_official_funding_day(
                        exchange="binance",
                        symbol=path.parts[2][7:],
                        day=path.parts[3][5:],
                        source=item["source"],
                        expected_sha256=item["sha256"],
                        expected_byte_count=byte_count,
                        expected_row_count=item["rows"],
                        provenance_receipt_sha256=item["provenance"],
                    )
                fd, _ = _private_regular(item["source"], "source output")
                try:
                    _revalidate(item["source"], fd, before, "source output")
                finally:
                    os.close(fd)
                if (
                    os.statvfs(_STORAGE["host_reserve_path"]).f_bavail
                    * os.statvfs(_STORAGE["host_reserve_path"]).f_frsize
                    < _STORAGE["host_reserve_bytes"]
                ):
                    raise PublicationError("host reserve was violated during source copy")
                target_size, target_sha = _file_sha(target)
                record = {
                    "relative": item["relative"],
                    "target": str(target.relative_to(candidate)),
                    "source_sha256": item["sha256"],
                    "target_sha256": target_sha,
                    "bytes": byte_count,
                    "target_bytes": target_size,
                    "rows": item["rows"],
                    "target_rows": _parquet_rows(target),
                    "provenance_receipt_sha256": item["provenance"],
                }
                _write_noreplace(
                    candidate
                    / ".alpha_max_publication"
                    / request_id
                    / "partitions"
                    / (hashlib.sha256(item["relative"].encode()).hexdigest() + ".json"),
                    record,
                )
                published.append(record)
            candidate_data_digest = _data_inventory_digest(candidate)
            final = {
                "schema": "alpha_max_canonical_publication_receipt.v2",
                "request_id": request_id,
                "terminal_receipt_sha256": terminal_sha256,
                "authority_key_id": key_id,
                "source_manifest_sha256": hashes["manifest"],
                "source_eligible_receipt_sha256": hashes["eligible"],
                "source_journal_sha256": hashes["journal"],
                "contract_manifest_sha256": hashes["contract"],
                "listing_records": listing,
                "canonical_root": str(args.canonical_root),
                "predecessor_identity": list(old_identity),
                "predecessor_path": str(old),
                "old_inventory": old_inventory,
                "old_inventory_sha256": _inventory_digest(old_inventory),
                "candidate_data_inventory_sha256": candidate_data_digest,
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
        swap_path = control / "swap.json"
        if swap_path.exists():
            swap, _ = _json(swap_path, "swap receipt")
            temporary = args.canonical_root.parent / swap["temporary"]
            mode = swap["mode"]
            if not (temporary.exists() or temporary.is_symlink()):
                raise PublicationError("prepared swap path is missing")
        else:
            temporary = (
                args.canonical_root.parent / f".{args.canonical_root.name}.{uuid.uuid4().hex}.swap"
            )
            os.symlink(
                f".{args.canonical_root.name}.generations/{request_id}",
                temporary,
            )
            mode = "replace" if active is None else "exchange"
            _write_noreplace(
                swap_path,
                {
                    "request_id": request_id,
                    "phase": "swap_ready",
                    "candidate": candidate.name,
                    "temporary": temporary.name,
                    "mode": mode,
                },
            )
            _fsync_dir(args.canonical_root.parent)

        if mode == "replace":
            if args.canonical_root.exists() or args.canonical_root.is_symlink():
                raise PublicationError("bootstrap canonical root appeared")
            os.replace(temporary, args.canonical_root)
        elif mode == "exchange":
            _rename_exchange(temporary, args.canonical_root)
        else:
            raise PublicationError("prepared swap mode is invalid")
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
            if mode == "exchange" and (temporary.exists() or temporary.is_symlink()):
                _rename_exchange(temporary, args.canonical_root)
            elif mode == "replace" and args.canonical_root.is_symlink():
                args.canonical_root.unlink()
            _fsync_dir(args.canonical_root.parent)
            _write_noreplace(
                control / "rollback.json",
                {"request_id": request_id, "phase": "rollback"},
            )
            _write_noreplace(
                control / "failure.json",
                {"request_id": request_id, "phase": "failure"},
            )
            raise
        _write_noreplace(
            control / "activated.json",
            {
                "request_id": request_id,
                "phase": "activated",
                "candidate": candidate.name,
            },
        )
        _cleanup_predecessor(temporary=temporary, final=final)
        _inventory(candidate)
        _write_noreplace(
            control / "completed.json",
            {"request_id": request_id, "phase": "completed"},
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PublicationError, ValueError, OSError) as exc:
        raise SystemExit(str(exc))
