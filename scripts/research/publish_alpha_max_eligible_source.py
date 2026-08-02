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
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
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
_CONFLICT_AUTH_MAX_JSON = 4 * 1024 * 1024
_PUBLICATION_CONTROL_RESERVE_BYTES = 64 * 1024 * 1024
_PARQUET_ENCODING_RESERVE_BYTES = 64 * 1024 * 1024
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


def _json(path: Path, label: str, *, limit: int = _MAX_JSON) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes(path, label, limit=limit)
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


def _write_noreplace(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    _ensure_private_directory(path.parent)
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


def _conflict_authorization(
    args: argparse.Namespace,
    *,
    terminal_sha256: str,
    request_id: str,
    hashes: dict[str, str],
    predecessor: Path,
    predecessor_inventory: list[dict[str, Any]],
    parts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    receipt_path = args.conflict_authorization_receipt
    key_path = args.conflict_authority_public_key
    if receipt_path is None and key_path is None:
        return None
    if receipt_path is None or key_path is None:
        raise PublicationError("conflict authorization receipt and public key must be paired")
    receipt, receipt_bytes = _json(
        receipt_path,
        "conflict authorization receipt",
        limit=_CONFLICT_AUTH_MAX_JSON,
    )
    receipt_info = os.lstat(receipt_path)
    if receipt_info.st_uid != os.getuid() or stat.S_IMODE(receipt_info.st_mode) != 0o600:
        raise PublicationError("conflict authorization receipt is not private")
    raw_key = _conflict_public_key(key_path)
    try:
        authority = Ed25519PublicKey.from_public_bytes(raw_key)
    except ValueError as exc:
        raise PublicationError("conflict authority public key is invalid") from exc
    outer_fields = {"schema", "type", "authority_key_id", "message", "signature"}
    if (
        set(receipt) != outer_fields
        or receipt["schema"] != "alpha_max_canonical_conflict_authorization_receipt.v1"
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
            b"luminaquant.alpha_max.canonical_conflict_authorization.v1\0"
            + canonical_bytes(unsigned),
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
        "entries",
    }
    if (
        set(message) != message_fields
        or message["schema"] != "alpha_max_canonical_conflict_authorization_message.v1"
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
        if (
            entry["predecessor_identity"] != list(_identity(target_info))
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
) -> None:
    paired = args.conflict_authorization_receipt is not None
    is_v3 = final.get("schema") == "alpha_max_canonical_publication_receipt.v3"
    if not is_v3:
        if paired:
            raise PublicationError("V2 publication cannot accept conflict authorization")
        return
    if not paired:
        raise PublicationError("V3 replay requires conflict authorization receipt and public key")
    predecessor = Path(final["predecessor_path"])
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
        unsigned = {key: value for key, value in receipt.items() if key != "signature"}
        try:
            Ed25519PublicKey.from_public_bytes(raw_key).verify(
                base64.b64decode(receipt["signature"], validate=True),
                b"luminaquant.alpha_max.canonical_conflict_authorization.v1\0"
                + canonical_bytes(unsigned),
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
            or receipt.get("schema") != "alpha_max_canonical_conflict_authorization_receipt.v1"
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
                "entries",
            }
            or message.get("schema") != "alpha_max_canonical_conflict_authorization_message.v1"
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
        parts=parts,
    )
    if authorization is None or (
        authorization["receipt_sha256"] != final["conflict_authorization_receipt_sha256"]
        or authorization["key_id"] != final["conflict_authority_key_id"]
        or authorization["message_sha256"] != final["conflict_authorization_message_sha256"]
        or authorization["message"]["predecessor_inventory_sha256"]
        != final["conflict_authorization_predecessor_inventory_sha256"]
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
    """Digest stable candidate identities while excluding transaction evidence."""
    records = []
    for record in _inventory(root, require_private=False):
        if record["path"] == "commit.json" or record["path"].endswith(
            "/canonical_publication_receipt.json"
        ):
            continue
        stable = {key: value for key, value in record.items() if key not in {"ctime_ns", "nlink"}}
        records.append(stable)
    return _inventory_digest(records)


def _is_coordination_lock(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if parts == (".bootstrap-incomplete",):
        return True
    if (
        len(parts) == 4
        and parts[0] == "market_ohlcv_1s"
        and re.fullmatch(r"\d{4}-\d{2}\.lock", parts[3])
    ):
        return True
    if (
        len(parts) == 4
        and parts[0] == "market_data_raw_aggtrades"
        and parts[3] == ".raw-stream.lock"
    ):
        return True
    if (
        len(parts) == 5
        and parts[0] == "market_data_raw_aggtrades"
        and parts[3].startswith("date=")
        and parts[4] == ".raw-partition.lock"
    ):
        return True
    return (
        len(parts) == 5
        and parts[0] == "feature_points"
        and parts[1].startswith("exchange=")
        and parts[2].startswith("symbol=")
        and parts[3].startswith("date=")
        and parts[4] == ".writer.lock"
    )


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
    if free_bytes < required_free_bytes:
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
    allow_active_mutation: bool = False,
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
    is_v3 = final.get("schema") == "alpha_max_canonical_publication_receipt.v3"
    if is_v3:
        expected_final_fields |= {
            "conflict_authorization_receipt_sha256",
            "conflict_authority_key_id",
            "conflict_authorization_message_sha256",
            "conflict_authorization_predecessor_inventory_sha256",
            "authorized_conflict_partition_count",
            "authorized_replaced_row_count",
        }
    if set(final) != expected_final_fields:
        raise PublicationError("prepared commit schema mismatch")
    if (
        final.get("schema")
        not in {
            "alpha_max_canonical_publication_receipt.v2",
            "alpha_max_canonical_publication_receipt.v3",
        }
        or final.get("request_id") != request_id
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
            or admission["free_bytes"] < admission["required_free_bytes"]
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
        if is_v3:
            expected_record_fields |= {"publication_mode", "conflict_effects"}
        if set(record) != expected_record_fields:
            raise PublicationError("prepared target record shape is invalid")
        source = expected[record["relative"]]
        if allow_active_mutation:
            current_source_bytes = record["bytes"]
            current_source_identity = tuple(record["source_identity"])
        else:
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
        if not allow_active_mutation:
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
        if is_v3 and (
            record["publication_mode"] not in {"strict_merge", "authorized_reconciliation"}
            or (
                record["publication_mode"] == "strict_merge"
                and record["conflict_effects"] is not None
            )
            or (
                record["publication_mode"] == "authorized_reconciliation"
                and not isinstance(record["conflict_effects"], dict)
            )
        ):
            raise PublicationError("prepared conflict partition binding is invalid")
    canonical_receipt, _ = _json(
        candidate / ".alpha_max_publication" / request_id / "canonical_publication_receipt.json",
        "canonical publication receipt",
    )
    if canonical_receipt != final:
        raise PublicationError("canonical publication receipt binding mismatch")
    if not allow_active_mutation and _data_inventory_digest(candidate) != final.get(
        "candidate_data_inventory_sha256"
    ):
        raise PublicationError("prepared candidate data inventory digest mismatch")
    if is_v3:
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
            or receipt.get("schema") != "alpha_max_canonical_conflict_authorization_receipt.v1"
            or receipt.get("type") != "canonical_conflict_authorization"
            or set(message) != message_fields
            or message.get("schema") != "alpha_max_canonical_conflict_authorization_message.v1"
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
            or final["conflict_authority_key_id"] == final["authority_key_id"]
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
    parser.add_argument("--conflict-authorization-receipt")
    parser.add_argument("--conflict-authority-public-key")
    args = parser.parse_args(argv)
    for name in (
        "source_root",
        "source_report",
        "terminal_receipt",
        "authority_public_key",
        "canonical_root",
    ):
        setattr(args, name, _absolute(getattr(args, name), name))
    for name in ("conflict_authorization_receipt", "conflict_authority_public_key"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, _absolute(value, name))
    if (args.conflict_authorization_receipt is None) != (
        args.conflict_authority_public_key is None
    ):
        raise PublicationError("conflict authorization receipt and public key must be paired")
    terminal, key_id, bound = _terminal(args)
    parts, hashes, listing = _partitions(args.source_root, args.source_report, bound)
    request_id = _sha(terminal.get("request_id"), "terminal request identity")
    terminal_sha256 = hashlib.sha256(
        _regular_bytes(args.terminal_receipt, "terminal receipt")
    ).hexdigest()
    generations = args.canonical_root.parent / f".{args.canonical_root.name}.generations"
    control = args.canonical_root.parent / f".{args.canonical_root.name}.transactions" / request_id
    candidate = generations / request_id
    lock_repo = ParquetMarketDataRepository(args.canonical_root)
    with lock_repo.generation_lock(exclusive=True, allow_incomplete_bootstrap=True):
        _ensure_private_directory(generations)
        _ensure_private_directory(control)
        try:
            active = _active_root(args.canonical_root, generations)
        except FileNotFoundError:
            active = None

        if active == candidate:
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
            _replay_conflict_authorization(
                args,
                final=final,
                terminal_sha256=terminal_sha256,
                request_id=request_id,
                hashes=hashes,
                parts=parts,
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
                allow_active_mutation=True,
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
            if not (candidate / "commit.json").exists():
                raise PublicationError("incomplete candidate requires bounded operator recovery")
            final, _ = _json(candidate / "commit.json", "prepared commit")
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
            authorization = _conflict_authorization(
                args,
                terminal_sha256=terminal_sha256,
                request_id=request_id,
                hashes=hashes,
                predecessor=old,
                predecessor_inventory=old_inventory,
                parts=parts,
            )
            if authorization is not None and authorization["key_id"] == key_id:
                raise PublicationError(
                    "conflict authorization must use a distinct repair authority"
                )
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
                if _host_free_bytes() < _STORAGE["host_reserve_bytes"]:
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
                "schema": (
                    "alpha_max_canonical_publication_receipt.v3"
                    if authorization is not None
                    else "alpha_max_canonical_publication_receipt.v2"
                ),
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
        if _host_free_bytes() < _STORAGE["host_reserve_bytes"] + _PUBLICATION_CONTROL_RESERVE_BYTES:
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
            mode = "exchange"
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
