#!/usr/bin/env python3
"""Observe one canonical generation exchange through the production reader facade.

The observer must start before the publisher. It first pins and signs the complete
old view, then remains alive while the publisher prepares and atomically exchanges
the generation. A PASS is possible only when that same process later reads the
bound new view after publisher replay.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from lumina_quant.alpha_max_terminal_policy import (
    ALPHA_MAX_PUBLICATION_OBSERVER_READY_SCHEMA,
    ALPHA_MAX_PUBLICATION_OBSERVER_READY_UNSIGNED_FIELDS,
    ALPHA_MAX_PUBLICATION_RECEIPT_FIELDS,
    ALPHA_MAX_PUBLICATION_WINDOW_BINDING_FIELDS,
    TerminalPolicyError,
    alpha_max_canonical_inventory_records,
    alpha_max_canonical_inventory_sha256,
    canonical_bytes,
    public_key_id,
    sign_message,
    verify_message,
)
from lumina_quant.market_data import MarketDataRepository
from lumina_quant.storage.parquet.ohlcv_repo import ParquetMarketDataRepository

RUN_ID = "482f9e03e246eda50641d06d81dcf17084799e7815656361bb62663dd1f149ea"
APPROVAL_LEAF = "current-state-approval-v13.json"
DOMAIN = b"luminaquant.alpha_max.publication_stage_envelope.v1\0"
MAX_JSON = 4 * 1024 * 1024
ACQUISITION_REQUEST_ID = "4d55958bf9387a63f1ce77f38e7e063909a550fce66aff873fc1d3b85851d152"
STAGE_ENVELOPE_SCHEMA = "alpha_max_publication_stage_envelope.v1"


class ObserverError(ValueError):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        named = os.lstat(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ObserverError(f"{label} is unavailable") from exc
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > MAX_JSON
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ObserverError(f"{label} is unsafe")
        chunks = bytearray()
        while len(chunks) < opened.st_size:
            block = os.read(fd, min(65536, opened.st_size - len(chunks)))
            if not block:
                raise ObserverError(f"{label} was truncated")
            chunks.extend(block)
        after = os.fstat(fd)
        current = os.lstat(path)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ObserverError(f"{label} changed while read")
    finally:
        os.close(fd)
    raw = bytes(chunks)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObserverError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise ObserverError(f"{label} is not canonical JSON")
    return value, raw


def _physical_identity(path: Path) -> list[Any]:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise ObserverError("active generation is not a physical directory")
    return [info.st_dev, info.st_ino, "directory"]


def _process_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = raw.rsplit(")", 1)[1].split()
        ticks = int(tail[19])
    except (OSError, IndexError, ValueError, UnicodeError) as exc:
        raise ObserverError("observer process identity is unavailable") from exc
    if ticks <= 0:
        raise ObserverError("observer process start time is invalid")
    return ticks


def inventory(root: Path) -> str:
    try:
        return alpha_max_canonical_inventory_sha256(root)
    except TerminalPolicyError as exc:
        raise ObserverError(str(exc)) from exc


def representatives(root: Path, maximum: int = 8) -> list[dict[str, Any]]:
    try:
        records = alpha_max_canonical_inventory_records(root)
    except TerminalPolicyError as exc:
        raise ObserverError(str(exc)) from exc
    values: list[dict[str, Any]] = []
    for record in records:
        if record["kind"] != "file":
            continue
        path = root / record["path"]
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise ObserverError("representative is unavailable") from exc
        try:
            before = os.fstat(fd)
            sample = os.read(fd, 4096)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (
            not sample
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (before.st_dev, before.st_ino) != (record["dev"], record["ino"])
        ):
            raise ObserverError("representative changed or returned an empty read")
        values.append(
            {
                "path": record["path"],
                "sha256": digest(sample),
                "bytes": len(sample),
            }
        )
        if len(values) == maximum:
            break
    if not values:
        raise ObserverError("active generation is empty; no representative reads")
    return values


def _verify_representatives(root: Path, expected: list[dict[str, Any]]) -> None:
    if not expected or len(expected) > 8:
        raise ObserverError("old generation drift sentinel is invalid")
    for representative in expected:
        if set(representative) != {"path", "sha256", "bytes"}:
            raise ObserverError("old generation drift sentinel is invalid")
        relative = representative["path"]
        expected_digest = representative["sha256"]
        expected_bytes = representative["bytes"]
        if (
            not isinstance(relative, str)
            or not isinstance(expected_digest, str)
            or not isinstance(expected_bytes, int)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not 0 < expected_bytes <= 4096
        ):
            raise ObserverError("old generation drift sentinel is invalid")
        try:
            fd = os.open(root / relative, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise ObserverError("old generation representative is unavailable") from exc
        try:
            before = os.fstat(fd)
            content = os.read(fd, 4096)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(content) != expected_bytes
            or digest(content) != expected_digest
        ):
            raise ObserverError("old generation changed after readiness")


def sample_physical(root: Path, sequence: int) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "timestamp_ns": time.time_ns(),
        "generation": root.name,
        "identity": _physical_identity(root),
        "inventory_sha256": inventory(root),
        "representatives": representatives(root),
    }


def sample(canonical_root: Path, sequence: int, expected_inventory: str) -> dict[str, Any]:
    """Compatibility helper used by focused tests and operator diagnostics."""
    physical = (
        canonical_root.resolve(strict=True) if canonical_root.is_symlink() else canonical_root
    )
    observation = sample_physical(physical, sequence)
    if observation["inventory_sha256"] != expected_inventory:
        raise ObserverError("active generation inventory mismatches expectation")
    return observation


def facade_digest(repository: MarketDataRepository, query: dict[str, Any]) -> str:
    one_second = repository.load_ohlcv(
        exchange=query["exchange"],
        symbol=query["symbol"],
        timeframe="1s",
        start_date=query["start"],
        end_date=query["end"],
    )
    aggregate = repository.load_ohlcv(
        exchange=query["exchange"],
        symbol=query["symbol"],
        timeframe=query["aggregate_timeframe"],
        start_date=query["start"],
        end_date=query["end"],
    )
    features = repository.load_futures_feature_points(
        exchange=query["exchange"],
        symbol=query["symbol"],
        start_date=query["start"],
        end_date=query["end"],
    )
    feature_name = query["feature_name"]
    if (
        one_second.is_empty()
        or aggregate.is_empty()
        or features.is_empty()
        or feature_name not in features.columns
    ):
        raise ObserverError("public facade returned an empty or incomplete required frame")
    feature_view = features.select("timestamp_ms", feature_name)
    return digest(
        canonical_bytes(
            {
                "one_second": json.loads(one_second.write_json()),
                "aggregate": json.loads(aggregate.write_json()),
                "feature": json.loads(feature_view.write_json()),
            }
        )
    )


def _fsync_parent(path: Path) -> None:
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _remove_receipt_temp_remnants(path: Path) -> None:
    prefix = f".{path.name}.tmp-"
    try:
        remnants = tuple(path.parent.iterdir())
    except FileNotFoundError:
        return
    for remnant in remnants:
        if not remnant.name.startswith(prefix):
            continue
        try:
            info = os.lstat(remnant)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            raise ObserverError(f"{path.name} has an unsafe temporary remnant")
        try:
            os.unlink(remnant)
        except FileNotFoundError:
            continue


def _validate_existing_receipt(path: Path, value: dict[str, Any], payload: bytes) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ObserverError(f"{path.name} is not a private regular receipt")
    existing, raw = _read_json(path, path.name)
    if existing != value or raw != payload:
        raise ObserverError(f"{path.name} conflicts")
    return True


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "renameat2 unavailable") from exc
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
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code), str(source))


def write_noreplace(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    _remove_receipt_temp_remnants(path)
    if _validate_existing_receipt(path, value, payload):
        _fsync_parent(path)
        return

    temp = path.parent / f".{path.name}.tmp-{os.urandom(16).hex()}"
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise ObserverError(f"cannot create temporary {path.name} receipt") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ObserverError("short observer receipt write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)

    try:
        _rename_noreplace(temp, path)
    except FileExistsError:
        if not _validate_existing_receipt(path, value, payload):
            raise ObserverError(f"{path.name} installation disappeared")
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass

    _fsync_parent(path)


def _private_key(private_key_fd: int) -> Ed25519PrivateKey:
    raw = os.pread(private_key_fd, 33, 0)
    if len(raw) != 32:
        raise ObserverError("observer credential fd is not a raw Ed25519 seed")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _publisher_public_key(path: Path) -> Ed25519PublicKey:
    try:
        named = os.lstat(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ObserverError("publisher public key is unavailable") from exc
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != 32
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ObserverError("publisher public key is unsafe")
        raw = os.read(fd, 33)
        if len(raw) != 32 or os.read(fd, 1):
            raise ObserverError("publisher public key is invalid")
        current = os.lstat(path)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ObserverError("publisher public key changed while read")
    finally:
        os.close(fd)
    try:
        return Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        raise ObserverError("publisher public key is invalid") from exc


def _publisher_envelope(
    value: dict[str, Any],
    raw: bytes,
    *,
    expected_kind: str,
    expected_schema: str,
    public_key: Ed25519PublicKey,
    key_id: str,
    name: str,
) -> dict[str, Any]:
    if set(value) != {"schema", "kind", "authority_key_id", "message", "signature"}:
        raise ObserverError(f"{name} envelope fields are invalid")
    if (
        value.get("schema") != STAGE_ENVELOPE_SCHEMA
        or value.get("kind") != expected_kind
        or value.get("authority_key_id") != key_id
        or not isinstance(value.get("message"), dict)
        or not isinstance(value.get("signature"), str)
    ):
        raise ObserverError(f"{name} envelope is invalid")
    unsigned = {key: item for key, item in value.items() if key != "signature"}
    try:
        signature = base64.b64decode(value["signature"], validate=True)
        public_key.verify(signature, DOMAIN + canonical_bytes(unsigned))
    except Exception as exc:
        raise ObserverError(f"{name} signature is invalid") from exc
    message = value["message"]
    if (
        set(message) != ALPHA_MAX_PUBLICATION_RECEIPT_FIELDS | {"kind"}
        or message.get("schema") != expected_schema
        or message.get("kind") != expected_kind
        or canonical_bytes(value) != raw
    ):
        raise ObserverError(f"{name} message schema is invalid")
    return message


def sign_terminal(private_key_fd: int, message: dict[str, Any]) -> dict[str, Any]:
    key = _private_key(private_key_fd)
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    unsigned = {
        "schema": "alpha_max_publication_stage_envelope.v1",
        "kind": "observer",
        "authority_key_id": digest(public),
        "message": message,
    }
    return {
        **unsigned,
        "signature": base64.b64encode(key.sign(DOMAIN + canonical_bytes(unsigned))).decode("ascii"),
    }


def _query(value: str) -> dict[str, Any]:
    try:
        query = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ObserverError("query spec is invalid JSON") from exc
    fields = {
        "exchange",
        "symbol",
        "start",
        "end",
        "aggregate_timeframe",
        "feature_name",
        "old_result_sha256",
        "new_result_sha256",
    }
    if not isinstance(query, dict) or set(query) != fields:
        raise ObserverError("query spec has unexpected fields")
    if not all(isinstance(query[name], str) and query[name] for name in fields):
        raise ObserverError("query spec contains an invalid value")
    if any(
        len(query[name]) != 64
        or any(character not in "0123456789abcdef" for character in query[name])
        for name in ("old_result_sha256", "new_result_sha256")
    ):
        raise ObserverError("query result digest is invalid")
    return query


def _ready_unsigned(
    *,
    args: argparse.Namespace,
    query_sha256: str,
    key_id: str,
    observation: dict[str, Any],
    loader_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": ALPHA_MAX_PUBLICATION_OBSERVER_READY_SCHEMA,
        "kind": "publication_observer_ready",
        "run_id": RUN_ID,
        "request_id": args.request_id,
        "approval_leaf": APPROVAL_LEAF,
        "approval_sha256": args.approval_sha256,
        "canonical_root": str(Path(args.canonical_root).absolute()),
        "query_spec_sha256": query_sha256,
        "observer_key_id": key_id,
        "observer_pid": os.getpid(),
        "observer_uid": os.getuid(),
        "observer_start_ticks": _process_start_ticks(os.getpid()),
        "observed_ns": observation["timestamp_ns"],
        "old_identity": observation["identity"],
        "old_inventory_sha256": observation["inventory_sha256"],
        "old_loader_sha256": loader_sha256,
        "old_observation": observation,
    }


def _validate_publisher_receipts(
    transaction: Path,
    ready: dict[str, Any],
    publisher_key: Ed25519PublicKey,
    *,
    require_replay: bool,
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    values: list[dict[str, Any]] = []
    raws: list[bytes] = []
    names = ["activated.json", "rollback-window-open.json"]
    schemas = [
        "alpha_max_publication_activation.v1",
        "alpha_max_publication_rollback_window.v1",
    ]
    kinds = ["activation", "open_window"]
    if require_replay:
        names.append("replay-verified.json")
        schemas.append("alpha_max_publication_replay.v1")
        kinds.append("replay")
    key_bytes = publisher_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    for name, schema, kind in zip(names, schemas, kinds, strict=True):
        value, raw = _read_json(transaction / name, name)
        values.append(
            _publisher_envelope(
                value,
                raw,
                expected_kind=kind,
                expected_schema=schema,
                public_key=publisher_key,
                key_id=digest(key_bytes),
                name=name,
            )
        )
        raws.append(raw)
    binding_fields = ALPHA_MAX_PUBLICATION_WINDOW_BINDING_FIELDS
    if any(
        any(value.get(field) != values[0].get(field) for field in binding_fields)
        for value in values[1:]
    ):
        raise ObserverError("publisher success receipts have mismatched bindings")
    activated = values[0]
    if (
        activated["request_id"] != ready["request_id"]
        or activated["run_id"] != ready["run_id"]
        or activated["approval_leaf"] != ready["approval_leaf"]
        or activated["approval_sha256"] != ready["approval_sha256"]
        or activated["observer_key_id"] != ready["observer_key_id"]
        or activated["observer_ready_sha256"] != digest(canonical_bytes(ready))
        or activated["observer_query_spec_sha256"] != ready["query_spec_sha256"]
        or activated["predecessor_identity"] != ready["old_identity"]
        or activated["predecessor_inventory_sha256"] != ready["old_inventory_sha256"]
    ):
        raise ObserverError("publisher receipts do not bind observer readiness")
    return activated, raws[0], raws[1], raws[2] if require_replay else b""


def _validate_paths(canonical: Path, transaction: Path, request_id: str) -> None:
    if (
        not canonical.is_absolute()
        or not transaction.is_absolute()
        or len(request_id) != 64
        or any(character not in "0123456789abcdef" for character in request_id)
        or request_id != ACQUISITION_REQUEST_ID
        or transaction != canonical.parent / f".{canonical.name}.transactions" / request_id
    ):
        raise ObserverError("observer path or request binding is invalid")


def _persist_failure_markers(
    transaction: Path,
    private_key_fd: int,
    *,
    request_id: str,
    approval_sha256: str,
    frozen_observation: dict[str, Any],
    state: str,
    error: str,
) -> None:
    message = {
        "kind": "observer_failure",
        "outcome": "FAIL",
        "run_id": RUN_ID,
        "request_id": request_id,
        "approval_leaf": APPROVAL_LEAF,
        "approval_sha256": approval_sha256,
        "state": state,
        "frozen_observation": frozen_observation,
        "error": error,
        "timestamp_ns": time.time_ns(),
    }
    receipt = sign_terminal(private_key_fd, message)
    write_noreplace(transaction / "observer-failure-intent.json", receipt)
    write_noreplace(transaction / "observer-failure.json", receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--transaction-root", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--publisher-public-key", required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--observer-key-fd", type=int, required=True)
    parser.add_argument("--query-spec", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or args.interval_seconds <= 0:
        raise ObserverError("sampling arguments are invalid")

    canonical = Path(args.canonical_root).absolute()
    transaction = Path(args.transaction_root).absolute()
    _validate_paths(canonical, transaction, args.request_id)
    query = _query(args.query_spec)
    query_sha256 = digest(canonical_bytes(query))
    key = _private_key(args.observer_key_fd)
    publisher_key = _publisher_public_key(Path(args.publisher_public_key).absolute())
    key_id = public_key_id(key.public_key())
    publisher_raw = publisher_key.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    if publisher_raw == key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ):
        raise ObserverError("observer key must be distinct from publisher key")
    repo = ParquetMarketDataRepository(canonical)
    facade = MarketDataRepository(str(canonical))
    deadline = time.monotonic() + args.timeout_seconds

    ready_path = transaction / "observer-ready.json"
    if ready_path.exists():
        ready, _ = _read_json(ready_path, "observer readiness")
        try:
            unsigned = verify_message("publication_observer_ready", ready, key.public_key())
        except TerminalPolicyError as exc:
            raise ObserverError(str(exc)) from exc
        if set(unsigned) != ALPHA_MAX_PUBLICATION_OBSERVER_READY_UNSIGNED_FIELDS:
            raise ObserverError("observer readiness schema is invalid")
        if unsigned["observer_pid"] != os.getpid() or unsigned[
            "observer_start_ticks"
        ] != _process_start_ticks(os.getpid()):
            raise ObserverError("observer readiness belongs to a prior process")
        ready = {**unsigned, "observer_signature_b64": ready["observer_signature_b64"]}
    else:
        with repo.generation_lock(exclusive=False) as physical:
            observation = sample_physical(physical, 0)
            loader_sha256 = facade_digest(facade, query)
            observation["public_loader_sha256"] = loader_sha256
        if loader_sha256 != query["old_result_sha256"]:
            raise ObserverError("old public loader digest mismatches the query contract")
        unsigned = _ready_unsigned(
            args=args,
            query_sha256=query_sha256,
            key_id=key_id,
            observation=observation,
            loader_sha256=loader_sha256,
        )
        ready = sign_message("publication_observer_ready", unsigned, key)
        write_noreplace(ready_path, ready)

    if (
        set(ready)
        != ALPHA_MAX_PUBLICATION_OBSERVER_READY_UNSIGNED_FIELDS | {"observer_signature_b64"}
        or ready["schema"] != ALPHA_MAX_PUBLICATION_OBSERVER_READY_SCHEMA
        or ready["kind"] != "publication_observer_ready"
        or ready["run_id"] != RUN_ID
        or ready["request_id"] != args.request_id
        or ready["approval_leaf"] != APPROVAL_LEAF
        or ready["approval_sha256"] != args.approval_sha256
        or ready["canonical_root"] != str(canonical)
        or ready["query_spec_sha256"] != query_sha256
        or ready["observer_key_id"] != key_id
        or ready["observer_uid"] != os.getuid()
    ):
        raise ObserverError("observer readiness binding is invalid")

    observations = [ready["old_observation"]]
    outcome = "PASS"
    error: str | None = None
    activated: dict[str, Any] | None = None
    activated_raw = open_raw = replay_raw = b""
    frozen_observation: dict[str, Any] | None = None
    try:
        while time.monotonic() < deadline:
            remaining = max(0.001, deadline - time.monotonic())
            with repo.generation_lock(
                exclusive=False,
                timeout_seconds=remaining,
                allow_incomplete_bootstrap=True,
            ) as physical:
                current_identity = _physical_identity(physical)
                if current_identity == ready["old_identity"]:
                    _verify_representatives(physical, ready["old_observation"]["representatives"])
                else:
                    replay_exists = (transaction / "replay-verified.json").exists()
                    try:
                        frozen_observation = sample_physical(physical, len(observations))
                        activated, activated_raw, open_raw, replay_raw = (
                            _validate_publisher_receipts(
                                transaction,
                                ready,
                                publisher_key,
                                require_replay=replay_exists,
                            )
                        )
                        if (
                            frozen_observation["identity"] != activated["candidate_identity"]
                            or frozen_observation["inventory_sha256"]
                            != activated["candidate_inventory_sha256"]
                            or str(physical) != activated["canonical_resolved_root"]
                        ):
                            raise ObserverError("new generation identity or inventory mismatches")
                        loader_sha256 = facade_digest(facade, query)
                        if loader_sha256 != query["new_result_sha256"]:
                            raise ObserverError(
                                "new public loader digest mismatches query contract"
                            )
                        frozen_observation["public_loader_sha256"] = loader_sha256
                        if replay_exists:
                            observations.append(frozen_observation)
                            break
                    except (OSError, RuntimeError, TerminalPolicyError, ObserverError) as exc:
                        if frozen_observation is not None:
                            _persist_failure_markers(
                                transaction,
                                args.observer_key_fd,
                                request_id=args.request_id,
                                approval_sha256=args.approval_sha256,
                                frozen_observation=frozen_observation,
                                state="W2_REPLAY_OK" if replay_exists else "W1_ACTIVATED_OPEN",
                                error=str(exc),
                            )
                        raise
            time.sleep(args.interval_seconds)
        else:
            if frozen_observation is not None:
                with repo.generation_lock(
                    exclusive=False,
                    timeout_seconds=max(0.001, deadline - time.monotonic()),
                    allow_incomplete_bootstrap=True,
                ) as physical:
                    frozen_observation = sample_physical(physical, len(observations))
                    replay_exists = (transaction / "replay-verified.json").exists()
                    if replay_exists:
                        activated, activated_raw, open_raw, replay_raw = (
                            _validate_publisher_receipts(
                                transaction, ready, publisher_key, require_replay=True
                            )
                        )
                    else:
                        _persist_failure_markers(
                            transaction,
                            args.observer_key_fd,
                            request_id=args.request_id,
                            approval_sha256=args.approval_sha256,
                            frozen_observation=frozen_observation,
                            state="W1_ACTIVATED_OPEN",
                            error="observer timed out before complete old/new exchange",
                        )
            raise ObserverError("observer timed out before complete old/new exchange")
    except (OSError, RuntimeError, TerminalPolicyError, ObserverError) as exc:
        outcome = "FAIL"
        error = str(exc)

    evidence = {
        "schema": "alpha_max_canonical_exchange_observations.v2",
        "run_id": RUN_ID,
        "request_id": args.request_id,
        "approval_sha256": args.approval_sha256,
        "query_spec_sha256": query_sha256,
        "observer_ready_sha256": digest(canonical_bytes(ready)),
        "activated_receipt_sha256": digest(activated_raw) if activated_raw else None,
        "open_receipt_sha256": digest(open_raw) if open_raw else None,
        "replay_receipt_sha256": digest(replay_raw) if replay_raw else None,
        "old_before_new": bool(
            outcome == "PASS"
            and len(observations) == 2
            and observations[0]["identity"] == ready["old_identity"]
            and activated is not None
            and observations[1]["identity"] == activated["candidate_identity"]
        ),
        "outcome": outcome,
        "observations": observations,
        "error": error,
        "frozen_observation": frozen_observation,
    }
    evidence_path = transaction / (
        "observer-observations-pass.json"
        if outcome == "PASS"
        else "observer-observations-fail.json"
    )
    write_noreplace(evidence_path, evidence)

    if frozen_observation is None:
        active_identity = ready["old_identity"]
        active_inventory = ready["old_inventory_sha256"]
    else:
        active_identity = frozen_observation["identity"]
        active_inventory = frozen_observation["inventory_sha256"]
    if activated is None:
        identities = {
            "active": active_identity,
            "candidate": active_identity,
            "predecessor": ready["old_identity"],
            "swap": active_identity,
        }
        inventories = {
            "active": active_inventory,
            "candidate": active_inventory,
            "predecessor": ready["old_inventory_sha256"],
        }
    else:
        identities = {
            "active": active_identity,
            "candidate": activated["candidate_identity"],
            "predecessor": activated["predecessor_identity"],
            "swap": activated["swap_identity"],
        }
        inventories = {
            "active": active_inventory,
            "candidate": activated["candidate_inventory_sha256"],
            "predecessor": activated["predecessor_inventory_sha256"],
        }
    state = (
        "W2_REPLAY_OK"
        if replay_raw
        else "W1_ACTIVATED_OPEN"
        if activated is not None or frozen_observation is not None
        else "W0_PRE_ACTIVATION"
    )
    message = {
        "kind": "observer",
        "outcome": outcome,
        "run_id": RUN_ID,
        "request_id": args.request_id,
        "approval_leaf": APPROVAL_LEAF,
        "approval_sha256": args.approval_sha256,
        "state": state,
        "stage": "observer",
        "timestamp_ns": time.time_ns(),
        "identities": identities,
        "inventories": inventories,
        "evidence_sha256": digest(canonical_bytes(evidence)),
    }
    if outcome == "FAIL":
        message["failure_reason"] = "observer_failed"
        message["failure_payload_sha256"] = message["evidence_sha256"]
    receipt = sign_terminal(args.observer_key_fd, message)
    write_noreplace(
        transaction
        / ("observer-terminal-pass.json" if outcome == "PASS" else "observer-terminal-fail.json"),
        receipt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ObserverError, OSError, ValueError) as exc:
        raise SystemExit(str(exc))
