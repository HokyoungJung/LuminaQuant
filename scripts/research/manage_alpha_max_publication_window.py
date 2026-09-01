#!/usr/bin/env python3
"""Inspect, admit finalization, roll back, or finalize one authenticated Alpha-Max window."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from lumina_quant.alpha_max_terminal_policy import (
    ALPHA_MAX_PUBLICATION_RECEIPT_FIELDS,
    ALPHA_MAX_PUBLICATION_WINDOW_BINDING_FIELDS,
    TerminalPolicyError,
    alpha_max_canonical_inventory_sha256,
    canonical_bytes,
    validate_w10_canonical_finalize_bundle,
    verify_signed_receipt,
)
from lumina_quant.storage.parquet.ohlcv_repo import ParquetMarketDataRepository

RUN_ID = "7359f1a8bc9f658d6778e152de1664199f517d069c30f4fae9866b76ddca3de4"
ACQUISITION_REQUEST = "7420631e32720073838a1b838b5e9a6df37dee9ba14583487e801a9a21bf736f"
APPROVAL_LEAF = "current-state-approval-v14.json"
CAPACITY_PATH = "/mnt/c"
HOST_RESERVE_BYTES = 21_474_836_480
MAX_JSON = 64 * 1024 * 1024
MAX_ACTION_AGE_NS = 300 * 1_000_000_000
MAX_GUARD_AGE_NS = 120 * 1_000_000_000
MAX_CLOCK_SKEW_NS = 5 * 1_000_000_000
STAGE_DOMAIN = b"luminaquant.alpha_max.publication_stage_envelope.v1\0"
ACTION_DOMAIN = b"luminaquant.alpha_max.publication_action_envelope.v1\0"
W10_FINALIZE_BUNDLE_DOMAIN = b"luminaquant.alpha_max.w10_canonical_finalize_bundle.v2\0"
W10_FINALIZE_BUNDLE_LEAF = "canonical-finalize-bundle.json"
W10_FINALIZE_RECEIPTS = (
    "decision-localization.json",
    "window-decision-intent.json",
    "W8_FINALIZING.json",
    "predecessor-cleanup-manifest.json",
    "predecessor-quarantined.json",
    "predecessor-cleanup-fsynced.json",
    "completed.json",
    "finalized.json",
)
ENVELOPE_FIELDS = frozenset({"schema", "kind", "authority_key_id", "message", "signature"})
STAGE_MESSAGE_FIELDS = frozenset(
    {
        "kind",
        "outcome",
        "run_id",
        "request_id",
        "approval_leaf",
        "approval_sha256",
        "state",
        "stage",
        "timestamp_ns",
        "identities",
        "inventories",
        "evidence_sha256",
    }
)
GUARD_FIELDS = frozenset({"capacity_path", "free_bytes", "required_bytes", "sample_sequence"})
FAILURE_STAGE_FIELDS = frozenset({"failure_reason", "failure_payload_sha256"})
STAGE_KINDS = frozenset(
    {
        "guard_capacity",
        "unit_exit",
        "cgroup",
        "replay",
        "observer",
        "postverify",
        "deterministic_backtest",
        "final_audit",
        "integrity",
    }
)
PUBLISHER_STAGE_KINDS = ("activation", "open_window", "replay")
PUBLISHER_PASS_PHASES = ("activated", "open", "pass")
READY_FIELDS = frozenset(
    {
        "kind",
        "run_id",
        "request_id",
        "approval_leaf",
        "approval_sha256",
        "state",
        "stage",
        "final_audit_sha256",
        "guard_receipt_sha256",
        "previous_guard_receipt_sha256",
        "previous_guard_sample_sequence",
        "guard_sample_sequence",
        "guard_no_gap",
        "capacity_path",
        "free_bytes",
        "required_bytes",
        "readiness_timestamp_ns",
        "context_sha256",
        "identities",
        "inventories",
    }
)
ACTION_COMMON_FIELDS = frozenset(
    {
        "kind",
        "run_id",
        "request_id",
        "approval_leaf",
        "approval_sha256",
        "state",
        "issued_ns",
        "expires_ns",
        "context_sha256",
        "identities",
        "inventories",
    }
)
ACTION_EXTRA_FIELDS = {
    "inspection": frozenset(
        {
            "next_pass",
            "unit_exit_sha256",
            "cgroup_sha256",
            "guard_receipt_sha256",
        }
    ),
    "rollback": frozenset({"reason", "failure_evidence_kind", "failure_evidence_sha256"}),
    "finalize": frozenset(),
}
PASS_NAMES = (
    "activated.json",
    "rollback-window-open.json",
    "replay-verified.json",
    "observer-terminal-pass.json",
    "postverify-pass.json",
    "deterministic-backtest-pass.json",
    "final-audit-pass.json",
    "ready-to-finalize.json",
)
PASS_SCHEMAS = (
    "alpha_max_publication_activation.v1",
    "alpha_max_publication_rollback_window.v1",
    "alpha_max_publication_replay.v1",
)
STATES = (
    "W0_PRE_ACTIVATION",
    "W1_ACTIVATED_OPEN",
    "W2_REPLAY_OK",
    "W3_OBSERVER_OK",
    "W4_POSTVERIFY_OK",
    "W5_BACKTEST_OK",
    "W6_READY_TO_FINALIZE",
)
PASS_STAGE_KINDS = {
    3: ("observer", "W2_REPLAY_OK"),
    4: ("postverify", "W3_OBSERVER_OK"),
    5: ("deterministic_backtest", "W4_POSTVERIFY_OK"),
    6: ("final_audit", "W5_BACKTEST_OK"),
}
REASONS = {
    "W1_ACTIVATED_OPEN": {
        "replay_not_started_after_activation",
        "replay_failed",
        "replay_crashed",
        "immediate_postactivation_integrity_failure",
    },
    "W2_REPLAY_OK": {
        "observer_failed",
        "observer_lost",
        "observer_gap",
        "observer_mixed_or_empty_read",
        "observer_crashed",
    },
    "W3_OBSERVER_OK": {"postverify_failed", "postverify_crashed"},
    "W4_POSTVERIFY_OK": {
        "deterministic_backtest_failed",
        "deterministic_backtest_nondeterministic",
        "deterministic_backtest_crashed",
    },
    "W5_BACKTEST_OK": {
        "final_audit_inventory_failed",
        "final_audit_loader_failed",
        "final_audit_lock_failed",
        "final_audit_capacity_failed",
        "final_audit_crashed",
    },
    "W6_READY_TO_FINALIZE": {"late_infrastructure_failure"},
}
REASON_EVIDENCE_KINDS = {
    "replay_not_started_after_activation": {"inspection"},
    "replay_failed": {"replay"},
    "replay_crashed": {"unit_exit", "inspection"},
    "immediate_postactivation_integrity_failure": {"integrity"},
    "observer_failed": {"observer"},
    "observer_lost": {"unit_exit", "inspection"},
    "observer_gap": {"observer"},
    "observer_mixed_or_empty_read": {"observer"},
    "observer_crashed": {"unit_exit", "inspection"},
    "postverify_failed": {"postverify"},
    "postverify_crashed": {"unit_exit", "inspection"},
    "deterministic_backtest_failed": {"deterministic_backtest"},
    "deterministic_backtest_nondeterministic": {"deterministic_backtest"},
    "deterministic_backtest_crashed": {"unit_exit", "inspection"},
    "final_audit_inventory_failed": {"final_audit"},
    "final_audit_loader_failed": {"final_audit"},
    "final_audit_lock_failed": {"final_audit"},
    "final_audit_capacity_failed": {"final_audit", "guard_capacity"},
    "final_audit_crashed": {"unit_exit", "inspection"},
    "late_infrastructure_failure": {"final_audit", "guard_capacity", "integrity"},
}
STATE_STAGE = {
    "W1_ACTIVATED_OPEN": "replay",
    "W2_REPLAY_OK": "observer",
    "W3_OBSERVER_OK": "postverify",
    "W4_POSTVERIFY_OK": "deterministic_backtest",
    "W5_BACKTEST_OK": "final_audit",
    "W6_READY_TO_FINALIZE": "decision_admission",
}
PREOPEN_ROLLBACK_FIELDS = frozenset(
    {
        "schema",
        "phase",
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
INSPECTION_STATES = frozenset(STATE_STAGE) - {"W6_READY_TO_FINALIZE"}


class WindowError(ValueError):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WindowError(f"{label} is not a SHA-256 digest")
    return value


def _typed_identity(value: Any, label: str) -> list[Any]:
    if (
        not isinstance(value, list)
        or len(value) not in {3, 4}
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[0] < 0
        or value[1] < 0
        or value[2] not in {"directory", "symlink"}
        or (value[2] == "directory" and len(value) != 3)
        or (
            value[2] == "symlink"
            and (len(value) != 4 or not isinstance(value[3], str) or not value[3])
        )
    ):
        raise WindowError(f"{label} identity is invalid")
    return value


def identity(path: Path) -> list[Any]:
    info = os.lstat(path)
    if stat.S_ISDIR(info.st_mode):
        return [info.st_dev, info.st_ino, "directory"]
    if stat.S_ISLNK(info.st_mode):
        return [info.st_dev, info.st_ino, "symlink", os.readlink(path)]
    raise WindowError("topology identity is neither directory nor symlink")


def inventory(path: Path) -> str:
    try:
        return alpha_max_canonical_inventory_sha256(path)
    except TerminalPolicyError as exc:
        raise WindowError(str(exc)) from exc


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path: Path, label: str, *, limit: int = MAX_JSON) -> tuple[dict[str, Any], bytes]:
    try:
        named = os.lstat(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise WindowError(f"{label} is unavailable") from exc
    try:
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > limit
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise WindowError(f"{label} is unsafe")
        payload = bytearray()
        while len(payload) < opened.st_size:
            block = os.read(fd, min(65536, opened.st_size - len(payload)))
            if not block:
                raise WindowError(f"{label} was truncated")
            payload.extend(block)
        after = os.fstat(fd)
        current = os.lstat(path)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise WindowError(f"{label} changed while read")
    finally:
        os.close(fd)
    raw = bytes(payload)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WindowError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise WindowError(f"{label} is not canonical JSON")
    return value, raw


def _receipt_write(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise WindowError("short receipt write")
        view = view[written:]


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
        raise WindowError("too many receipt temp remnants")
    removed = False
    for remnant in remnants:
        info = os.lstat(remnant)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size > MAX_JSON
        ):
            raise WindowError("receipt temp remnant is unsafe")
        remnant.unlink()
        removed = True
    if removed:
        _fsync_dir(path.parent)


def write_noreplace(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        old, raw = read_json(path, path.name)
        if old != value or raw != payload:
            raise WindowError(f"{path.name} conflicts")
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    _fsync_dir(path.parent)


def _decision_localization(
    root: Path, *, action: str | None = None
) -> tuple[dict[str, Any], bytes] | None:
    path = root / "decision-localization.json"
    if not path.exists():
        return None
    value, raw = read_json(path, "decision localization bundle")
    if (
        set(value)
        != {
            "schema",
            "action",
            "authorization_sha256",
            "authorization_envelope",
            "guard_receipt_sha256",
            "guard_envelope",
            "failure_evidence_sha256",
            "failure_evidence",
        }
        or value.get("schema") != "alpha_max_window_decision_localization.v1"
        or value.get("action") not in {"rollback", "finalize"}
        or (action is not None and value["action"] != action)
        or not isinstance(value.get("authorization_envelope"), dict)
        or not isinstance(value.get("guard_envelope"), dict)
    ):
        raise WindowError("decision localization bundle is invalid")
    _digest(value.get("authorization_sha256"), "localized authorization")
    _digest(value.get("guard_receipt_sha256"), "localized guard")
    if sha(canonical_bytes(value["authorization_envelope"])) != value["authorization_sha256"]:
        raise WindowError("localized authorization digest changed")
    if sha(canonical_bytes(value["guard_envelope"])) != value["guard_receipt_sha256"]:
        raise WindowError("localized guard digest changed")
    if value["action"] == "rollback":
        if not isinstance(value.get("failure_evidence"), dict):
            raise WindowError("localized rollback evidence is invalid")
        _digest(value.get("failure_evidence_sha256"), "localized failure evidence")
        if sha(canonical_bytes(value["failure_evidence"])) != value["failure_evidence_sha256"]:
            raise WindowError("localized failure evidence digest changed")
    elif (
        value.get("failure_evidence") is not None
        or value.get("failure_evidence_sha256") is not None
    ):
        raise WindowError("localized finalize evidence is invalid")
    return value, raw


def _public_key(path: Path, label: str) -> tuple[Ed25519PublicKey, str]:
    try:
        named = os.lstat(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise WindowError(f"{label} is unavailable") from exc
    try:
        opened = os.fstat(fd)
        raw = os.read(fd, 33)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) not in {0o400, 0o444}
            or len(raw) != 32
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise WindowError(f"{label} is unsafe")
    finally:
        os.close(fd)
    return Ed25519PublicKey.from_public_bytes(raw), sha(raw)


def _private_key(fd: int) -> Ed25519PrivateKey:
    raw = os.pread(fd, 33, 0)
    if len(raw) != 32:
        raise WindowError("authority credential is not a raw Ed25519 seed")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _seal(
    *,
    schema: str,
    domain: bytes,
    private_key_fd: int,
    message: dict[str, Any],
) -> dict[str, Any]:
    private = _private_key(private_key_fd)
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    unsigned = {
        "schema": schema,
        "kind": message["kind"],
        "authority_key_id": sha(public),
        "message": message,
    }
    return {
        **unsigned,
        "signature": base64.b64encode(private.sign(domain + canonical_bytes(unsigned))).decode(
            "ascii"
        ),
    }


def _w10_finalize_bundle(
    root: Path,
    context: dict[str, Any],
    authorization_sha256: str,
    private_key_fd: int,
) -> dict[str, Any]:
    private = _private_key(private_key_fd)
    authority_raw = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    if (
        sha(authority_raw)
        != _public_key(Path(context["_authority_public_key"]), "terminal authority public key")[1]
    ):
        raise WindowError("finalize authority credential does not match pinned public key")
    receipts: dict[str, str] = {}
    for leaf in W10_FINALIZE_RECEIPTS:
        _, raw = read_json(root / leaf, f"W10 transaction receipt {leaf}")
        receipts[leaf] = sha(raw)
    active = Path(context["_canonical_root"]).resolve(strict=True)
    message = {
        "run_id": RUN_ID,
        "acquisition_request_id": ACQUISITION_REQUEST,
        "transaction_root": str(root),
        "approval_sha256": context["approval_sha256"],
        "canonical_identity": {
            "logical_root": identity(Path(context["_canonical_root"])),
            "active_generation": identity(active),
        },
        "state": "W10_FINALIZED",
        "finalize_authorization_sha256": authorization_sha256,
        "finalize_context_sha256": _context_hash(context),
        "transaction_receipt_sha256s": receipts,
    }
    unsigned = {
        "schema": "alpha_max_canonical_finalize_bundle.v2",
        "authority_key_id": sha(authority_raw),
        "message": message,
    }
    return {
        **unsigned,
        "signature": base64.b64encode(
            private.sign(W10_FINALIZE_BUNDLE_DOMAIN + canonical_bytes(unsigned))
        ).decode("ascii"),
    }


def _validate_w10_finalize_bundle(
    root: Path,
    context: dict[str, Any],
    authorization_sha256: str,
) -> None:
    public, _ = _public_key(Path(context["_authority_public_key"]), "terminal authority public key")
    public_raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    try:
        validate_w10_canonical_finalize_bundle(
            root / W10_FINALIZE_BUNDLE_LEAF,
            authority_public_key_b64=base64.b64encode(public_raw).decode("ascii"),
            run_id=RUN_ID,
            acquisition_request_id=ACQUISITION_REQUEST,
            approval_sha256=context["approval_sha256"],
            canonical_identity={
                "logical_root": identity(Path(context["_canonical_root"])),
                "active_generation": identity(
                    Path(context["_canonical_root"]).resolve(strict=True)
                ),
            },
            finalize_authorization_sha256=authorization_sha256,
            finalize_context_sha256=_context_hash(context),
        )
    except TerminalPolicyError as exc:
        raise WindowError(str(exc)) from exc


def _stage_message(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") not in STAGE_KINDS:
        raise WindowError("stage message kind is invalid")
    expected = STAGE_MESSAGE_FIELDS | (
        GUARD_FIELDS if value["kind"] == "guard_capacity" else frozenset()
    )
    if value.get("outcome") == "FAIL":
        expected |= FAILURE_STAGE_FIELDS
    if set(value) != expected or value.get("outcome") not in {"PASS", "FAIL"}:
        raise WindowError("stage message schema is invalid")
    if (
        value.get("run_id") != RUN_ID
        or value.get("approval_leaf") != APPROVAL_LEAF
        or type(value.get("timestamp_ns")) is not int
        or value["timestamp_ns"] <= 0
        or not isinstance(value.get("identities"), dict)
        or not isinstance(value.get("inventories"), dict)
    ):
        raise WindowError("stage message binding is invalid")
    _digest(value.get("approval_sha256"), "stage approval")
    if value["outcome"] == "FAIL":
        _digest(value.get("failure_payload_sha256"), "failure payload")
    _digest(value.get("evidence_sha256"), "stage evidence")
    if value["kind"] == "guard_capacity" and (
        value.get("capacity_path") != CAPACITY_PATH
        or type(value.get("free_bytes")) is not int
        or type(value.get("required_bytes")) is not int
        or type(value.get("sample_sequence")) is not int
        or value["free_bytes"] < 0
        or value["required_bytes"] < HOST_RESERVE_BYTES
        or value["sample_sequence"] < 0
    ):
        raise WindowError("guard capacity schema is invalid")
    return value


def seal_stage_envelope(private_key_fd: int, message: dict[str, Any]) -> dict[str, Any]:
    _stage_message(message)
    return _seal(
        schema="alpha_max_publication_stage_envelope.v1",
        domain=STAGE_DOMAIN,
        private_key_fd=private_key_fd,
        message=message,
    )


def _action_message(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") not in ACTION_EXTRA_FIELDS:
        raise WindowError("action message kind is invalid")
    if set(value) != ACTION_COMMON_FIELDS | ACTION_EXTRA_FIELDS[value["kind"]]:
        raise WindowError("action message schema is invalid")
    if (
        value.get("run_id") != RUN_ID
        or value.get("approval_leaf") != APPROVAL_LEAF
        or type(value.get("issued_ns")) is not int
        or type(value.get("expires_ns")) is not int
        or value["issued_ns"] <= 0
        or value["expires_ns"] < value["issued_ns"]
        or value["expires_ns"] - value["issued_ns"] > MAX_ACTION_AGE_NS
        or not isinstance(value.get("identities"), dict)
        or not isinstance(value.get("inventories"), dict)
    ):
        raise WindowError("action message binding is invalid")
    _digest(value.get("approval_sha256"), "action approval")
    _digest(value.get("context_sha256"), "action context")
    for field in ACTION_EXTRA_FIELDS[value["kind"]]:
        if field.endswith("_sha256"):
            _digest(value.get(field), field)
    if value["kind"] == "rollback" and (
        value.get("reason") not in REASON_EVIDENCE_KINDS
        or value.get("failure_evidence_kind") not in REASON_EVIDENCE_KINDS[value["reason"]]
    ):
        raise WindowError("rollback evidence kind is invalid")
    return value


def seal_action_envelope(private_key_fd: int, message: dict[str, Any]) -> dict[str, Any]:
    _action_message(message)
    return _seal(
        schema="alpha_max_publication_action_envelope.v1",
        domain=ACTION_DOMAIN,
        private_key_fd=private_key_fd,
        message=message,
    )


def _verify_envelope(
    path: Path,
    public_key_path: Path,
    *,
    schema: str,
    domain: bytes,
    kind: str,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    envelope, raw = read_json(path, label)
    if (
        set(envelope) != ENVELOPE_FIELDS
        or envelope.get("schema") != schema
        or envelope.get("kind") != kind
        or not isinstance(envelope.get("message"), dict)
    ):
        raise WindowError(f"{label} envelope schema is invalid")
    public, key_id = _public_key(public_key_path, f"{label} public key")
    if envelope.get("authority_key_id") != key_id:
        raise WindowError(f"{label} key binding is invalid")
    unsigned = {field: envelope[field] for field in ENVELOPE_FIELDS - {"signature"}}
    try:
        public.verify(
            base64.b64decode(envelope["signature"], validate=True),
            domain + canonical_bytes(unsigned),
        )
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise WindowError(f"{label} signature is invalid") from exc
    return envelope["message"], raw


def verify_stage_envelope(
    path: Path,
    public_key_path: Path,
    *,
    kind: str,
    request_id: str,
    approval_sha256: str,
    state_name: str | None = None,
) -> dict[str, Any]:
    message, _ = _verify_envelope(
        path,
        public_key_path,
        schema="alpha_max_publication_stage_envelope.v1",
        domain=STAGE_DOMAIN,
        kind=kind,
        label="stage evidence",
    )
    _stage_message(message)
    if (
        message["kind"] != kind
        or message["request_id"] != request_id
        or message["approval_sha256"] != approval_sha256
        or (state_name is not None and message["state"] != state_name)
    ):
        raise WindowError("stage envelope contextual binding is invalid")
    return message


def verify_action_envelope(
    path: Path,
    public_key_path: Path,
    *,
    kind: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    message, raw = _verify_envelope(
        path,
        public_key_path,
        schema="alpha_max_publication_action_envelope.v1",
        domain=ACTION_DOMAIN,
        kind=kind,
        label="action authorization",
    )
    _action_message(message)
    envelope, _ = read_json(path, "action authorization")
    return message, sha(raw), envelope


def _verify_action_envelope_value(
    envelope: dict[str, Any], public_key_path: Path, *, kind: str
) -> tuple[dict[str, Any], str]:
    if (
        set(envelope) != ENVELOPE_FIELDS
        or envelope.get("schema") != "alpha_max_publication_action_envelope.v1"
        or envelope.get("kind") != kind
        or not isinstance(envelope.get("message"), dict)
    ):
        raise WindowError("localized action authorization envelope is invalid")
    public, key_id = _public_key(public_key_path, "action authorization public key")
    if envelope.get("authority_key_id") != key_id:
        raise WindowError("localized action authorization key binding is invalid")
    unsigned = {field: envelope[field] for field in ENVELOPE_FIELDS - {"signature"}}
    try:
        public.verify(
            base64.b64decode(envelope["signature"], validate=True),
            ACTION_DOMAIN + canonical_bytes(unsigned),
        )
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise WindowError("localized action authorization signature is invalid") from exc
    message = envelope["message"]
    _action_message(message)
    return message, sha(canonical_bytes(envelope))


def _verify_stage_envelope_value(
    envelope: dict[str, Any],
    public_key_path: Path,
    *,
    kind: str,
    request_id: str,
    approval_sha256: str,
    state_name: str,
) -> tuple[dict[str, Any], str]:
    if (
        set(envelope) != ENVELOPE_FIELDS
        or envelope.get("schema") != "alpha_max_publication_stage_envelope.v1"
        or envelope.get("kind") != kind
        or not isinstance(envelope.get("message"), dict)
    ):
        raise WindowError("localized guard envelope is invalid")
    public, key_id = _public_key(public_key_path, "guard capacity public key")
    if envelope.get("authority_key_id") != key_id:
        raise WindowError("localized guard key binding is invalid")
    unsigned = {field: envelope[field] for field in ENVELOPE_FIELDS - {"signature"}}
    try:
        public.verify(
            base64.b64decode(envelope["signature"], validate=True),
            STAGE_DOMAIN + canonical_bytes(unsigned),
        )
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise WindowError("localized guard signature is invalid") from exc
    message = envelope["message"]
    _stage_message(message)
    if (
        message["kind"] != kind
        or message["request_id"] != request_id
        or message["approval_sha256"] != approval_sha256
        or message["state"] != state_name
    ):
        raise WindowError("localized guard contextual binding is invalid")
    return message, sha(canonical_bytes(envelope))


def _verify_publisher_receipt(path: Path, public_key_path: Path, *, kind: str) -> dict[str, Any]:
    if kind not in PUBLISHER_STAGE_KINDS:
        raise WindowError("publisher receipt kind is invalid")
    message, _ = _verify_envelope(
        path,
        public_key_path,
        schema="alpha_max_publication_stage_envelope.v1",
        domain=STAGE_DOMAIN,
        kind=kind,
        label="publisher PASS receipt",
    )
    if (
        set(message) != ALPHA_MAX_PUBLICATION_RECEIPT_FIELDS | {"kind"}
        or message.get("kind") != kind
    ):
        raise WindowError("publisher PASS receipt message is invalid")
    return message


def _verify_ready_to_finalize(root: Path, context: dict[str, Any]) -> dict[str, Any]:
    message, _ = _verify_envelope(
        root / "ready-to-finalize.json",
        Path(context["_authority_public_key"]),
        schema="alpha_max_publication_stage_envelope.v1",
        domain=STAGE_DOMAIN,
        kind="ready_to_finalize",
        label="ready-to-finalize receipt",
    )
    if (
        set(message) != READY_FIELDS
        or message["kind"] != "ready_to_finalize"
        or message["run_id"] != RUN_ID
        or message["request_id"] != context["request_id"]
        or message["approval_leaf"] != APPROVAL_LEAF
        or message["approval_sha256"] != context["approval_sha256"]
        or message["state"] != "W5_BACKTEST_OK"
        or message["stage"] != "decision_admission"
        or message["context_sha256"] != _context_hash(context)
        or message["identities"] != _context_identities(context)
        or message["inventories"] != _context_inventories(context)
        or message["guard_no_gap"] is not True
        or message["capacity_path"] != CAPACITY_PATH
        or any(
            type(message[field]) is not int
            for field in (
                "free_bytes",
                "required_bytes",
                "guard_sample_sequence",
                "previous_guard_sample_sequence",
                "readiness_timestamp_ns",
            )
        )
        or message["free_bytes"] <= message["required_bytes"]
        or message["required_bytes"] < HOST_RESERVE_BYTES
        or message["guard_sample_sequence"] <= message["previous_guard_sample_sequence"]
    ):
        raise WindowError("ready-to-finalize receipt binding is invalid")
    final_audit, final_raw = read_json(root / "final-audit-pass.json", "final audit")
    if sha(final_raw) != message["final_audit_sha256"]:
        raise WindowError("ready-to-finalize final-audit digest is invalid")
    for name, digest, sequence in (
        (
            "ready-to-finalize-guard.json",
            message["guard_receipt_sha256"],
            message["guard_sample_sequence"],
        ),
        (
            "ready-to-finalize-previous-guard.json",
            message["previous_guard_receipt_sha256"],
            message["previous_guard_sample_sequence"],
        ),
    ):
        guard, raw = read_json(root / name, "ready-to-finalize guard")
        verified = verify_stage_envelope(
            root / name,
            Path(context["_authority_public_key"]),
            kind="guard_capacity",
            request_id=context["request_id"],
            approval_sha256=context["approval_sha256"],
            state_name="W5_BACKTEST_OK",
        )
        if (
            guard.get("message") != verified
            or sha(raw) != digest
            or verified["outcome"] != "PASS"
            or verified["sample_sequence"] != sequence
            or verified["identities"] != _context_identities(context)
            or verified["inventories"] != _context_inventories(context)
            or verified["free_bytes"] <= verified["required_bytes"]
        ):
            raise WindowError("ready-to-finalize guard continuity is invalid")
    if not isinstance(final_audit.get("message"), dict) or message[
        "readiness_timestamp_ns"
    ] < final_audit["message"].get("timestamp_ns", 0):
        raise WindowError("ready-to-finalize predates final audit")
    return message


def transaction_root(canonical: Path, request_id: str) -> Path:
    _digest(request_id, "request id")
    return canonical.parent / f".{canonical.name}.transactions" / request_id


def _context_hash(context: dict[str, Any]) -> str:
    return sha(
        canonical_bytes(
            {field: context[field] for field in sorted(ALPHA_MAX_PUBLICATION_WINDOW_BINDING_FIELDS)}
        )
    )


def _context_identities(context: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "active": context["candidate_identity"],
        "candidate": context["candidate_identity"],
        "predecessor": context["predecessor_identity"],
        "swap": context["swap_identity"],
    }


def _context_inventories(context: dict[str, Any]) -> dict[str, str]:
    return {
        "active": context["candidate_inventory_sha256"],
        "candidate": context["candidate_inventory_sha256"],
        "predecessor": context["predecessor_inventory_sha256"],
    }


def _validate_context_paths(context: dict[str, Any], canonical: Path) -> None:
    candidate = Path(context["candidate"])
    predecessor = Path(context["predecessor"])
    swap = Path(context["swap_temporary_path"])
    generations = canonical.parent / f".{canonical.name}.generations"
    if (
        not candidate.is_absolute()
        or candidate != generations / context["candidate_leaf"]
        or context["candidate_leaf"] != context["request_id"]
        or not predecessor.is_absolute()
        or predecessor == candidate
        or not swap.is_absolute()
        or swap != canonical.parent / context["swap"]
        or swap.name != context["swap"]
        or (predecessor != swap and predecessor.parent != generations)
        or context["canonical_resolved_root"] != str(candidate)
    ):
        raise WindowError("publisher generation paths are invalid")
    for field in (
        "candidate_identity",
        "predecessor_identity",
        "swap_identity",
        "pre_exchange_predecessor_identity",
        "post_exchange_candidate_identity",
        "post_exchange_predecessor_identity",
        "canonical_logical_root_identity",
    ):
        _typed_identity(context[field], field)
    if (
        context["candidate_identity"] != context["post_exchange_candidate_identity"]
        or context["predecessor_identity"] != context["post_exchange_predecessor_identity"]
        or context["predecessor_identity"] != context["pre_exchange_predecessor_identity"]
        or context["canonical_logical_root_identity"][2] != "symlink"
        or context["canonical_logical_root_identity"][3]
        != f".{canonical.name}.generations/{context['candidate_leaf']}"
    ):
        raise WindowError("publisher topology fields are inconsistent")
    for field in (
        "approval_sha256",
        "terminal_receipt_sha256",
        "observer_key_id",
        "observer_ready_sha256",
        "observer_query_spec_sha256",
        "swap_receipt_sha256",
        "candidate_inventory_sha256",
        "predecessor_inventory_sha256",
    ):
        _digest(context[field], field)


def pass_context(
    root: Path,
    canonical: Path,
    terminal_receipt: Path,
    authority_key: Path,
    observer_key: Path,
    publisher_key: Path,
) -> dict[str, Any]:
    terminal = verify_signed_receipt(terminal_receipt, authority_key)
    results = terminal.message.get("target_results")
    if (
        terminal.message.get("scope") != "acquisition"
        or terminal.message.get("request_id") != ACQUISITION_REQUEST
        or terminal.message.get("terminal_state", {}).get("kind") != "SUCCEEDED"
        or not isinstance(results, list)
        or len(results) != 2
        or any(not isinstance(item, dict) or item.get("return_code") != 0 for item in results)
    ):
        raise WindowError("terminal receipt does not authorize successful acquisition")
    _, authority_key_id = _public_key(authority_key, "terminal authority public key")
    _, observer_key_id = _public_key(observer_key, "observer public key")
    _, publisher_key_id = _public_key(publisher_key, "publisher public key")
    if len({authority_key_id, observer_key_id, publisher_key_id}) != 3:
        raise WindowError("terminal authority, publisher, and observer keys must be distinct")
    if authority_key_id != terminal.key_id:
        raise WindowError("terminal authority public key binding is invalid")
    value = _verify_publisher_receipt(
        root / "activated.json",
        publisher_key,
        kind="activation",
    )
    if (
        set(value) != ALPHA_MAX_PUBLICATION_RECEIPT_FIELDS | {"kind"}
        or value.get("schema") != PASS_SCHEMAS[0]
        or value.get("phase") != "activated"
        or value.get("request_id") != root.name
        or value.get("run_id") != RUN_ID
        or value.get("acquisition_request_id") != ACQUISITION_REQUEST
        or value.get("approval_leaf") != APPROVAL_LEAF
        or value.get("terminal_receipt_sha256") != terminal.receipt_sha256
        or value.get("authority_key_id") != terminal.key_id
    ):
        raise WindowError("activation receipt authority binding is invalid")
    if observer_key_id != value.get("observer_key_id"):
        raise WindowError("observer public key binding is invalid")
    _validate_context_paths(value, canonical)
    return {
        **value,
        "_authority_public_key": str(authority_key),
        "_observer_public_key": str(observer_key),
        "_publisher_public_key": str(publisher_key),
        "_canonical_root": str(canonical),
    }


def _success_state(root: Path, context: dict[str, Any] | None = None) -> str:
    present = [name for name in PASS_NAMES if (root / name).exists()]
    prefix = 0
    for index, name in enumerate(PASS_NAMES):
        path = root / name
        if not path.exists():
            break
        if context is not None:
            if index < 3:
                value = _verify_publisher_receipt(
                    path,
                    Path(context["_publisher_public_key"]),
                    kind=PUBLISHER_STAGE_KINDS[index],
                )
                if (
                    value.get("schema") != PASS_SCHEMAS[index]
                    or value.get("phase") != PUBLISHER_PASS_PHASES[index]
                    or any(
                        value.get(field) != context.get(field)
                        for field in ALPHA_MAX_PUBLICATION_WINDOW_BINDING_FIELDS
                    )
                ):
                    raise WindowError("publisher PASS receipt binding is invalid")
            elif index == 7:
                _verify_ready_to_finalize(root, context)
            else:
                kind, stage_state = PASS_STAGE_KINDS[index]
                key = Path(
                    context[
                        "_observer_public_key" if kind == "observer" else "_authority_public_key"
                    ]
                )
                message = verify_stage_envelope(
                    path,
                    key,
                    kind=kind,
                    request_id=context["request_id"],
                    approval_sha256=context["approval_sha256"],
                    state_name=stage_state,
                )
                if (
                    message["outcome"] != "PASS"
                    or message["identities"] != _context_identities(context)
                    or message["inventories"] != _context_inventories(context)
                ):
                    raise WindowError("stage PASS topology binding is invalid")
                if kind == "observer":
                    evidence, raw = read_json(
                        root / "observer-observations-pass.json",
                        "observer evidence",
                    )
                    if (
                        message["evidence_sha256"] != sha(raw)
                        or evidence.get("schema") != "alpha_max_canonical_exchange_observations.v2"
                        or evidence.get("outcome") != "PASS"
                        or evidence.get("old_before_new") is not True
                        or evidence.get("observer_ready_sha256") != context["observer_ready_sha256"]
                    ):
                        raise WindowError("observer evidence binding is invalid")
        prefix += 1
    if any(name not in PASS_NAMES[:prefix] for name in present):
        raise WindowError("success receipts are not contiguous")
    if prefix == 1:
        raise WindowError("activation and rollback-window receipts are incomplete")
    if prefix == 0:
        return STATES[0]
    if prefix == len(PASS_NAMES):
        return STATES[-1]
    if prefix == len(PASS_NAMES) - 1:
        return STATES[-2]
    return STATES[prefix - 1]


def _decision_receipt(
    root: Path,
    context: dict[str, Any] | None,
    success_state: str,
) -> tuple[str, dict[str, Any]] | None:
    intent_path = root / "window-decision-intent.json"
    rollback_journal = root / "W7_ROLLING_BACK.json"
    finalize_journal = root / "W8_FINALIZING.json"
    rolled_back = root / "rolled-back.json"
    finalized = root / "finalized.json"
    completed = root / "completed.json"
    rollback_exchange = root / "rollback-exchange-fsynced.json"
    finalize_artifacts = (
        root / "predecessor-cleanup-manifest.json",
        root / "predecessor-quarantined.json",
        root / "predecessor-cleanup-fsynced.json",
    )
    decision_paths = (
        intent_path,
        rollback_journal,
        finalize_journal,
        rollback_exchange,
        *finalize_artifacts,
        rolled_back,
        finalized,
        completed,
    )
    if not intent_path.exists():
        if any(path.exists() for path in decision_paths[1:]):
            raise WindowError("publication decision artifacts are orphaned")
        return None

    intent, _ = read_json(intent_path, "decision intent")
    if (
        set(intent)
        != {
            "schema",
            "action",
            "state",
            "authorization_leaf",
            "authorization_sha256",
            "context_sha256",
            "failure_evidence_leaf",
            "failure_evidence_sha256",
            "localization_sha256",
        }
        or intent.get("schema") != "alpha_max_window_decision_intent.v4"
        or intent.get("action") not in {"rollback", "finalize"}
        or intent.get("state") != success_state
        or intent.get("authorization_leaf") != "decision-localization.json"
        or (
            intent.get("action") == "rollback"
            and intent.get("failure_evidence_leaf") != "decision-localization.json"
        )
        or (intent.get("action") == "finalize" and intent.get("failure_evidence_leaf") is not None)
    ):
        raise WindowError("publication decision intent is invalid")
    _digest(intent.get("authorization_sha256"), "decision authorization")
    _digest(intent.get("context_sha256"), "decision context")
    _digest(intent.get("localization_sha256"), "decision localization")
    if intent["action"] == "rollback":
        _digest(intent.get("failure_evidence_sha256"), "decision failure evidence")
    localized = _decision_localization(root, action=intent["action"])
    if localized is None:
        raise WindowError("decision localization is missing")
    bundle, bundle_raw = localized
    if (
        sha(bundle_raw) != intent["localization_sha256"]
        or bundle["authorization_sha256"] != intent["authorization_sha256"]
        or bundle["failure_evidence_sha256"] != intent["failure_evidence_sha256"]
    ):
        raise WindowError("decision localization conflicts")
    if context is not None:
        guard, guard_sha256 = _verify_stage_envelope_value(
            bundle["guard_envelope"],
            Path(context["_authority_public_key"]),
            kind="guard_capacity",
            request_id=context["request_id"],
            approval_sha256=context["approval_sha256"],
            state_name=intent["state"],
        )
        if (
            guard_sha256 != bundle["guard_receipt_sha256"]
            or guard["outcome"] != "PASS"
            or guard["identities"] != _context_identities(context)
            or guard["inventories"] != _context_inventories(context)
        ):
            raise WindowError("localized guard binding changed")
    authorization = bundle["authorization_envelope"]["message"]
    if context is not None:
        authorization, authorization_sha256 = _verify_action_envelope_value(
            bundle["authorization_envelope"],
            Path(context["_authority_public_key"]),
            kind=intent["action"],
        )
        if authorization_sha256 != intent["authorization_sha256"]:
            raise WindowError("decision authorization bundle signature changed")
        _verify_action_context(authorization, context, success_state)
    if intent["action"] == "rollback":
        if bundle["failure_evidence_sha256"] != authorization["failure_evidence_sha256"]:
            raise WindowError("decision failure bundle conflicts")
    elif (
        intent["state"] != "W6_READY_TO_FINALIZE"
        or intent.get("failure_evidence_sha256") is not None
    ):
        raise WindowError("finalize decision intent is invalid")
    if context is not None and intent["context_sha256"] != _context_hash(context):
        raise WindowError("publication decision context changed")

    journal_path = rollback_journal if intent["action"] == "rollback" else finalize_journal
    opposite_journal = finalize_journal if intent["action"] == "rollback" else rollback_journal
    if opposite_journal.exists():
        raise WindowError("opposite publication decision journal exists")
    if journal_path.exists():
        journal, _ = read_json(journal_path, "decision journal")
        if journal != {
            "schema": "alpha_max_window_action_journal.v1",
            "action": intent["action"],
            "authorization_sha256": intent["authorization_sha256"],
            "context_sha256": intent["context_sha256"],
            "phase": "intent-fsynced",
        }:
            raise WindowError("publication decision journal conflicts")
    if rollback_exchange.exists():
        if intent["action"] != "rollback" or not journal_path.exists():
            raise WindowError("rollback exchange receipt is orphaned")
        exchange, _ = read_json(rollback_exchange, "rollback exchange receipt")
        if exchange != {
            "schema": "alpha_max_window_rollback_exchange.v1",
            "authorization_sha256": intent["authorization_sha256"],
            "context_sha256": intent["context_sha256"],
        }:
            raise WindowError("rollback exchange receipt conflicts")
    if any(path.exists() for path in finalize_artifacts) and intent["action"] != "finalize":
        raise WindowError("rollback decision has finalize cleanup artifacts")

    if rolled_back.exists() and finalized.exists():
        raise WindowError("opposite publication terminals coexist")
    terminal_path = rolled_back if intent["action"] == "rollback" else finalized
    opposite_terminal = finalized if intent["action"] == "rollback" else rolled_back
    if opposite_terminal.exists():
        raise WindowError("opposite publication terminal exists")
    if completed.exists():
        if intent["action"] != "finalize" or not journal_path.exists():
            raise WindowError("publication completion receipt is orphaned")
        completion, _ = read_json(completed, "publication completion receipt")
        if completion != {
            "schema": "alpha_max_publication_completed.v1",
            "request_id": context["request_id"]
            if context is not None
            else completion.get("request_id"),
            "phase": "completed",
            "authorization_sha256": intent["authorization_sha256"],
            "context_sha256": intent["context_sha256"],
        }:
            raise WindowError("publication completion receipt conflicts")
    if terminal_path.exists():
        if not journal_path.exists():
            raise WindowError("publication terminal has no decision journal")
        terminal, _ = read_json(terminal_path, "terminal receipt")
        terminal_state = "W9_ROLLED_BACK" if intent["action"] == "rollback" else "W10_FINALIZED"
        if terminal != {
            "schema": "alpha_max_window_terminal.v2",
            "action": intent["action"],
            "state": terminal_state,
            "authorization_sha256": intent["authorization_sha256"],
            "context_sha256": intent["context_sha256"],
        }:
            raise WindowError("publication terminal receipt conflicts")
        if intent["action"] == "finalize" and not completed.exists():
            raise WindowError("finalize terminal has no completion receipt")
        if intent["action"] == "finalize":
            if context is None:
                raise WindowError("W10 bundle requires authenticated context")
            if not (root / W10_FINALIZE_BUNDLE_LEAF).exists():
                return "W8_FINALIZING", intent
            _validate_w10_finalize_bundle(root, context, intent["authorization_sha256"])
        return terminal_state, intent
    return ("W7_ROLLING_BACK" if intent["action"] == "rollback" else "W8_FINALIZING"), intent


def state(root: Path, context: dict[str, Any] | None = None) -> str:
    success_state = _success_state(root, context)
    decision = _decision_receipt(root, context, success_state)
    return success_state if decision is None else decision[0]


def _verify_action_context(message: dict[str, Any], context: dict[str, Any], current: str) -> None:
    if (
        message["run_id"] != RUN_ID
        or message["request_id"] != context["request_id"]
        or message["approval_leaf"] != APPROVAL_LEAF
        or message["approval_sha256"] != context["approval_sha256"]
        or message["state"] != current
        or message["context_sha256"] != _context_hash(context)
        or message["identities"] != _context_identities(context)
        or message["inventories"] != _context_inventories(context)
    ):
        raise WindowError("action authorization context is invalid")


def _verify_guard(
    path: Path,
    context: dict[str, Any],
    current: str,
) -> tuple[dict[str, Any], bytes]:
    message = verify_stage_envelope(
        path,
        Path(context["_authority_public_key"]),
        kind="guard_capacity",
        request_id=context["request_id"],
        approval_sha256=context["approval_sha256"],
        state_name=current,
    )
    _, raw = read_json(path, "guard capacity receipt")
    now = time.time_ns()
    if (
        message["outcome"] != "PASS"
        or message["identities"] != _context_identities(context)
        or message["inventories"] != _context_inventories(context)
        or message["free_bytes"] <= message["required_bytes"]
        or message["timestamp_ns"] > now + MAX_CLOCK_SKEW_NS
        or now - message["timestamp_ns"] > MAX_GUARD_AGE_NS
    ):
        raise WindowError("guard capacity PASS is stale or underbound")
    stats = os.statvfs(message["capacity_path"])
    current_free = stats.f_bavail * stats.f_frsize
    if current_free <= message["required_bytes"]:
        raise WindowError("current host capacity is not strictly above admission")
    return message, raw


def admit_finalize(args: argparse.Namespace, root: Path, context: dict[str, Any]) -> int:
    if args.authority_key_fd is None or not args.guard_receipt or not args.previous_guard_receipt:
        raise WindowError("finalize admission requires authority key and two guard receipts")
    previous_path = Path(args.previous_guard_receipt)
    guard_path = Path(args.guard_receipt)
    previous, previous_raw = _verify_guard(previous_path, context, "W5_BACKTEST_OK")
    guard, guard_raw = _verify_guard(guard_path, context, "W5_BACKTEST_OK")
    if (
        guard["sample_sequence"] != previous["sample_sequence"] + 1
        or guard["timestamp_ns"] <= previous["timestamp_ns"]
        or guard["capacity_path"] != previous["capacity_path"]
        or guard["required_bytes"] != previous["required_bytes"]
    ):
        raise WindowError("finalize guard samples are not continuous")

    ready_path = root / "ready-to-finalize.json"
    if ready_path.exists():
        ready = _verify_ready_to_finalize(root, context)
        if ready["guard_receipt_sha256"] != sha(guard_raw) or ready[
            "previous_guard_receipt_sha256"
        ] != sha(previous_raw):
            raise WindowError("ready-to-finalize receipt conflicts with guard inputs")
        print(json.dumps(ready, sort_keys=True))
        return 0

    if (
        _success_state(root, context) != "W5_BACKTEST_OK"
        or not (root / "final-audit-pass.json").exists()
    ):
        raise WindowError("finalize admission requires the complete final-audit prefix")
    final_audit, final_raw = read_json(root / "final-audit-pass.json", "final audit")
    if not isinstance(final_audit.get("message"), dict):
        raise WindowError("final audit envelope is invalid")

    private = _private_key(args.authority_key_fd)
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    if sha(public) != context["authority_key_id"]:
        raise WindowError("finalize admission key is not the terminal authority")
    message = {
        "kind": "ready_to_finalize",
        "run_id": RUN_ID,
        "request_id": context["request_id"],
        "approval_leaf": APPROVAL_LEAF,
        "approval_sha256": context["approval_sha256"],
        "state": "W5_BACKTEST_OK",
        "stage": "decision_admission",
        "final_audit_sha256": sha(final_raw),
        "guard_receipt_sha256": sha(guard_raw),
        "previous_guard_receipt_sha256": sha(previous_raw),
        "previous_guard_sample_sequence": previous["sample_sequence"],
        "guard_sample_sequence": guard["sample_sequence"],
        "guard_no_gap": True,
        "capacity_path": guard["capacity_path"],
        "free_bytes": guard["free_bytes"],
        "required_bytes": guard["required_bytes"],
        "readiness_timestamp_ns": max(
            time.time_ns(), guard["timestamp_ns"], final_audit["message"]["timestamp_ns"]
        ),
        "context_sha256": _context_hash(context),
        "identities": _context_identities(context),
        "inventories": _context_inventories(context),
    }
    envelope = _seal(
        schema="alpha_max_publication_stage_envelope.v1",
        domain=STAGE_DOMAIN,
        private_key_fd=args.authority_key_fd,
        message=message,
    )
    previous_envelope, _ = read_json(previous_path, "previous guard capacity receipt")
    guard_envelope, _ = read_json(guard_path, "guard capacity receipt")
    write_noreplace(root / "ready-to-finalize-previous-guard.json", previous_envelope)
    write_noreplace(root / "ready-to-finalize-guard.json", guard_envelope)
    write_noreplace(ready_path, envelope)
    ready = _verify_ready_to_finalize(root, context)
    print(json.dumps(ready, sort_keys=True))
    return 0


def _verify_open_topology(
    context: dict[str, Any], canonical: Path, *, phase: str
) -> tuple[Path, Path, Path]:
    candidate = Path(context["candidate"])
    predecessor = Path(context["predecessor"])
    swap = Path(context["swap_temporary_path"])
    if identity(candidate) != context["candidate_identity"]:
        raise WindowError("candidate topology identity changed")
    if inventory(candidate) != context["candidate_inventory_sha256"]:
        raise WindowError("candidate inventory changed")
    if phase == "before":
        if (
            identity(canonical) != context["canonical_logical_root_identity"]
            or canonical.resolve(strict=True) != candidate
            or identity(swap) != context["swap_identity"]
            or identity(predecessor) != context["predecessor_identity"]
            or inventory(predecessor) != context["predecessor_inventory_sha256"]
        ):
            raise WindowError("open-window topology binding changed")
    elif phase == "after-rollback":
        if (
            identity(canonical) != context["swap_identity"]
            or identity(swap) != context["canonical_logical_root_identity"]
        ):
            raise WindowError("rollback exchange topology changed")
        active = canonical.resolve(strict=True) if canonical.is_symlink() else canonical
        if (
            identity(active) != context["predecessor_identity"]
            or inventory(active) != context["predecessor_inventory_sha256"]
        ):
            raise WindowError("rolled-back predecessor binding changed")
    elif phase == "finalize":
        if (
            identity(canonical) != context["canonical_logical_root_identity"]
            or canonical.resolve(strict=True) != candidate
        ):
            raise WindowError("finalized candidate topology changed")
    else:
        raise WindowError("unknown topology verification phase")
    return candidate, predecessor, swap


def _next_pass(current: str) -> str:
    index = STATES.index(current)
    return (
        PASS_NAMES[index + 1]
        if current != "W6_READY_TO_FINALIZE"
        else "window-decision-intent.json"
    )


def inspect(args: argparse.Namespace, root: Path, context: dict[str, Any]) -> int:
    current = state(root, context)
    if not args.for_rollback:
        print(current)
        return 0
    if current not in INSPECTION_STATES:
        raise WindowError("current state has no inspection-backed rollback")
    message, authorization_sha256, envelope = verify_action_envelope(
        Path(args.authorization),
        Path(context["_authority_public_key"]),
        kind="inspection",
    )
    _verify_action_context(message, context, current)
    now = time.time_ns()
    if not (message["issued_ns"] - MAX_CLOCK_SKEW_NS <= now <= message["expires_ns"]):
        raise WindowError("inspection authorization is not current")
    expected_next = _next_pass(current)
    if message["next_pass"] != expected_next or (root / expected_next).exists():
        raise WindowError("inspection next-PASS assertion is invalid")
    unit_path = Path(args.unit_exit_evidence)
    cgroup_path = Path(args.cgroup_evidence)
    guard_path = Path(args.guard_receipt)
    unit = verify_stage_envelope(
        unit_path,
        Path(context["_authority_public_key"]),
        kind="unit_exit",
        request_id=context["request_id"],
        approval_sha256=context["approval_sha256"],
        state_name=current,
    )
    cgroup = verify_stage_envelope(
        cgroup_path,
        Path(context["_authority_public_key"]),
        kind="cgroup",
        request_id=context["request_id"],
        approval_sha256=context["approval_sha256"],
        state_name=current,
    )
    _, unit_raw = read_json(unit_path, "unit exit evidence")
    _, cgroup_raw = read_json(cgroup_path, "cgroup evidence")
    _, guard_raw = _verify_guard(guard_path, context, current)
    expected_stage = STATE_STAGE[current]
    if (
        unit["outcome"] != "FAIL"
        or cgroup["outcome"] != "PASS"
        or unit["stage"] != expected_stage
        or cgroup["stage"] != expected_stage
        or any(
            item["identities"] != _context_identities(context)
            or item["inventories"] != _context_inventories(context)
            for item in (unit, cgroup)
        )
        or message["unit_exit_sha256"] != sha(unit_raw)
        or message["cgroup_sha256"] != sha(cgroup_raw)
        or message["guard_receipt_sha256"] != sha(guard_raw)
    ):
        raise WindowError("inspection evidence binding is invalid")
    _verify_open_topology(context, Path(context["_canonical_root"]), phase="before")
    authorization_name = f"inspection-authorization-{current}.json"
    write_noreplace(root / authorization_name, envelope)
    for source, name in (
        (unit_path, f"inspection-unit-exit-{current}.json"),
        (cgroup_path, f"inspection-cgroup-{current}.json"),
        (guard_path, f"inspection-guard-{current}.json"),
    ):
        copied, _ = read_json(source, "inspection source evidence")
        write_noreplace(root / name, copied)
    receipt = {
        "schema": "alpha_max_window_inspection.v4",
        "state": current,
        "next_pass": expected_next,
        "context_sha256": _context_hash(context),
        "authorization_sha256": authorization_sha256,
        "authorization_leaf": authorization_name,
        "unit_exit_sha256": sha(unit_raw),
        "cgroup_sha256": sha(cgroup_raw),
        "guard_receipt_sha256": sha(guard_raw),
        "identities": _context_identities(context),
        "inventories": _context_inventories(context),
    }
    write_noreplace(root / f"recovery-inspection-{current}.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _verify_inspection_evidence(
    path: Path,
    root: Path,
    context: dict[str, Any],
    current: str,
) -> int:
    receipt, _ = read_json(path, "rollback inspection evidence")
    expected_fields = {
        "schema",
        "state",
        "next_pass",
        "context_sha256",
        "authorization_sha256",
        "authorization_leaf",
        "unit_exit_sha256",
        "cgroup_sha256",
        "guard_receipt_sha256",
        "identities",
        "inventories",
    }
    authorization_leaf = f"inspection-authorization-{current}.json"
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != "alpha_max_window_inspection.v4"
        or receipt.get("state") != current
        or receipt.get("next_pass") != _next_pass(current)
        or receipt.get("context_sha256") != _context_hash(context)
        or receipt.get("authorization_leaf") != authorization_leaf
        or receipt.get("identities") != _context_identities(context)
        or receipt.get("inventories") != _context_inventories(context)
        or (root / receipt["next_pass"]).exists()
    ):
        raise WindowError("rollback inspection evidence is invalid")
    authorization_path = root / authorization_leaf
    authorization, raw = read_json(authorization_path, "inspection authorization copy")
    message, digest_value, verified = verify_action_envelope(
        authorization_path,
        Path(context["_authority_public_key"]),
        kind="inspection",
    )
    if (
        authorization != verified
        or digest_value != receipt["authorization_sha256"]
        or sha(raw) != receipt["authorization_sha256"]
        or message["next_pass"] != receipt["next_pass"]
        or message["unit_exit_sha256"] != receipt["unit_exit_sha256"]
        or message["cgroup_sha256"] != receipt["cgroup_sha256"]
        or message["guard_receipt_sha256"] != receipt["guard_receipt_sha256"]
    ):
        raise WindowError("inspection authorization copy is invalid")
    _verify_action_context(message, context, current)
    for path_name, kind, digest, outcome in (
        (f"inspection-unit-exit-{current}.json", "unit_exit", receipt["unit_exit_sha256"], "FAIL"),
        (f"inspection-cgroup-{current}.json", "cgroup", receipt["cgroup_sha256"], "PASS"),
    ):
        evidence_path = root / path_name
        evidence = verify_stage_envelope(
            evidence_path,
            Path(context["_authority_public_key"]),
            kind=kind,
            request_id=context["request_id"],
            approval_sha256=context["approval_sha256"],
            state_name=current,
        )
        _, raw = read_json(evidence_path, "inspection signed evidence")
        if (
            sha(raw) != digest
            or evidence["outcome"] != outcome
            or evidence["identities"] != _context_identities(context)
            or evidence["inventories"] != _context_inventories(context)
        ):
            raise WindowError("inspection signed evidence changed")
    _, guard_raw = _verify_guard(root / f"inspection-guard-{current}.json", context, current)
    if sha(guard_raw) != receipt["guard_receipt_sha256"]:
        raise WindowError("inspection guard evidence changed")
    return message["issued_ns"]


def _verify_failure_evidence(
    args: argparse.Namespace,
    root: Path,
    context: dict[str, Any],
    current: str,
    authorization: dict[str, Any],
) -> None:
    path = Path(args.failure_evidence)
    _, raw = read_json(path, "rollback failure evidence")
    if sha(raw) != authorization["failure_evidence_sha256"]:
        raise WindowError("rollback failure evidence digest is invalid")
    kind = authorization["failure_evidence_kind"]
    reason = authorization["reason"]
    if kind not in REASON_EVIDENCE_KINDS[reason]:
        raise WindowError("rollback reason does not accept this evidence kind")
    if kind == "inspection":
        evidence_timestamp = _verify_inspection_evidence(path, root, context, current)
    else:
        key = Path(
            context["_observer_public_key" if kind == "observer" else "_authority_public_key"]
        )
        message = verify_stage_envelope(
            path,
            key,
            kind=kind,
            request_id=context["request_id"],
            approval_sha256=context["approval_sha256"],
            state_name=current,
        )
        if (
            message["outcome"] != "FAIL"
            or message["stage"] != STATE_STAGE[current]
            or message.get("failure_reason") != reason
            or message.get("failure_payload_sha256") != message["evidence_sha256"]
            or message["identities"] != _context_identities(context)
            or message["inventories"] != _context_inventories(context)
        ):
            raise WindowError("stage failure evidence is underbound")
        evidence_timestamp = message["timestamp_ns"]
    now = time.time_ns()
    if (
        evidence_timestamp > now + MAX_CLOCK_SKEW_NS
        or evidence_timestamp > authorization["issued_ns"]
    ):
        raise WindowError("rollback failure evidence is not contemporaneous")
    if current == "W6_READY_TO_FINALIZE":
        readiness = _verify_ready_to_finalize(root, context)
        if evidence_timestamp <= readiness["readiness_timestamp_ns"]:
            raise WindowError("late infrastructure evidence is not newer than readiness")


def _rename_exchange(left: Path, right: Path) -> None:
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
    if renameat2(-100, os.fsencode(left), -100, os.fsencode(right), 2) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(left))


def _rename_noreplace(source: Path, destination: Path) -> None:
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
        raise OSError(code, os.strerror(code), str(source))


def _full_inventory(root: Path) -> list[dict[str, Any]]:
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise WindowError("cleanup root is not a physical directory")
    records: list[dict[str, Any]] = []

    def visit(current: Path, relative: str) -> None:
        for entry in sorted(os.scandir(current), key=lambda item: item.name):
            info = entry.stat(follow_symlinks=False)
            name = f"{relative}/{entry.name}" if relative else entry.name
            common = {
                "path": name,
                "mode": stat.S_IMODE(info.st_mode),
                "dev": info.st_dev,
                "ino": info.st_ino,
            }
            if stat.S_ISDIR(info.st_mode):
                records.append({**common, "kind": "dir"})
                visit(current / entry.name, name)
            elif stat.S_ISREG(info.st_mode):
                records.append(
                    {
                        **common,
                        "kind": "file",
                        "size": info.st_size,
                        "mtime_ns": info.st_mtime_ns,
                    }
                )
            else:
                raise WindowError("cleanup root contains an unsafe artifact")

    visit(root, "")
    return records


def _cleanup_manifest(
    root: Path,
    context: dict[str, Any],
    authorization_sha256: str,
) -> dict[str, Any]:
    if inventory(root) != context["predecessor_inventory_sha256"]:
        raise WindowError("cleanup manifest does not match predecessor inventory")
    records = _full_inventory(root)
    return {
        "schema": "alpha_max_predecessor_cleanup_manifest.v1",
        "request_id": context["request_id"],
        "authorization_sha256": authorization_sha256,
        "predecessor": context["predecessor"],
        "predecessor_identity": context["predecessor_identity"],
        "stable_inventory_sha256": context["predecessor_inventory_sha256"],
        "entries": records,
    }


def _verify_manifest_subset(root: Path, manifest: dict[str, Any]) -> None:
    expected = {entry["path"]: entry for entry in manifest["entries"]}
    actual = _full_inventory(root)
    if any(expected.get(entry["path"]) != entry for entry in actual):
        raise WindowError("cleanup remainder is not an authenticated subset")


def _remove_manifest_subset(root: Path, manifest: dict[str, Any]) -> None:
    if not root.exists():
        return
    _verify_manifest_subset(root, manifest)
    actual = _full_inventory(root)
    for entry in sorted(
        actual, key=lambda item: (item["kind"] == "dir", item["path"]), reverse=True
    ):
        if entry["kind"] != "file":
            continue
        path = root / entry["path"]
        if path.exists():
            current = os.lstat(path)
            if (
                not stat.S_ISREG(current.st_mode)
                or [current.st_dev, current.st_ino] != [entry["dev"], entry["ino"]]
                or current.st_size != entry["size"]
                or current.st_mtime_ns != entry["mtime_ns"]
            ):
                raise WindowError("cleanup file identity changed")
            path.unlink()
    directories = [entry for entry in actual if entry["kind"] == "dir"]
    for entry in sorted(
        directories,
        key=lambda item: (len(Path(item["path"]).parts), item["path"]),
        reverse=True,
    ):
        path = root / entry["path"]
        if path.exists():
            current = os.lstat(path)
            if not stat.S_ISDIR(current.st_mode) or [current.st_dev, current.st_ino] != [
                entry["dev"],
                entry["ino"],
            ]:
                raise WindowError("cleanup directory identity changed")
            path.rmdir()
    root.rmdir()
    _fsync_dir(root.parent)


def _finalize_cleanup(
    root: Path,
    context: dict[str, Any],
    authorization_sha256: str,
) -> None:
    predecessor = Path(context["predecessor"])
    swap = Path(context["swap_temporary_path"])
    quarantine = root / "retired-predecessor"
    manifest_path = root / "predecessor-cleanup-manifest.json"
    quarantine_receipt_path = root / "predecessor-quarantined.json"
    quarantine_receipt = {
        "schema": "alpha_max_predecessor_quarantine.v1",
        "authorization_sha256": authorization_sha256,
        "identity": context["predecessor_identity"],
        "inventory_sha256": context["predecessor_inventory_sha256"],
    }
    if manifest_path.exists():
        manifest, _ = read_json(manifest_path, "predecessor cleanup manifest")
        if (
            manifest.get("authorization_sha256") != authorization_sha256
            or manifest.get("predecessor_identity") != context["predecessor_identity"]
            or manifest.get("stable_inventory_sha256") != context["predecessor_inventory_sha256"]
        ):
            raise WindowError("predecessor cleanup manifest conflicts")
    else:
        if not predecessor.exists():
            raise WindowError("predecessor disappeared before cleanup was journaled")
        manifest = _cleanup_manifest(predecessor, context, authorization_sha256)
        write_noreplace(manifest_path, manifest)
    if predecessor.exists():
        if identity(predecessor) != context["predecessor_identity"]:
            raise WindowError("predecessor identity changed before quarantine")
        _verify_manifest_subset(predecessor, manifest)
        if quarantine.exists():
            raise WindowError("predecessor and quarantine both exist")
        _rename_noreplace(predecessor, quarantine)
    elif not quarantine.exists() and not quarantine_receipt_path.exists():
        raise WindowError("predecessor quarantine is missing")
    _fsync_dir(predecessor.parent)
    if quarantine.parent != predecessor.parent:
        _fsync_dir(quarantine.parent)
    if quarantine.exists():
        if identity(quarantine) != context["predecessor_identity"]:
            raise WindowError("quarantined predecessor identity changed")
        write_noreplace(quarantine_receipt_path, quarantine_receipt)
    recorded_quarantine, _ = read_json(quarantine_receipt_path, "predecessor quarantine")
    if recorded_quarantine != quarantine_receipt:
        raise WindowError("predecessor quarantine receipt conflicts")
    if swap != predecessor:
        if quarantine.exists():
            _verify_manifest_subset(quarantine, manifest)
        if os.path.lexists(swap):
            if identity(swap) != context["swap_identity"] or not swap.is_symlink():
                raise WindowError("swap topology changed before finalize cleanup")
            swap.unlink()
        _fsync_dir(swap.parent)
    if quarantine.exists():
        _remove_manifest_subset(quarantine, manifest)
    _fsync_dir(quarantine.parent)
    write_noreplace(
        root / "predecessor-cleanup-fsynced.json",
        {
            "schema": "alpha_max_predecessor_cleanup.v1",
            "authorization_sha256": authorization_sha256,
            "phase": "removal-fsynced",
        },
    )


def decide(
    args: argparse.Namespace,
    root: Path,
    action: str,
    context: dict[str, Any],
) -> int:
    if action == "finalize" and args.authority_key_fd is None:
        raise WindowError("finalize requires an authority key")
    success_current = _success_state(root, context)
    full_current = state(root, context)
    intent_path = root / "window-decision-intent.json"
    stored_intent = None
    localized = _decision_localization(root)
    if localized is not None and localized[0]["action"] != action:
        raise WindowError("opposite publication localization is immutable")
    if intent_path.exists():
        stored_intent, _ = read_json(intent_path, "decision intent")
        if stored_intent.get("action") != action:
            raise WindowError("opposite publication decision is immutable")
    if localized is not None:
        bundle, bundle_raw = localized
        authorization, authorization_sha256 = _verify_action_envelope_value(
            bundle["authorization_envelope"],
            Path(context["_authority_public_key"]),
            kind=action,
        )
        if authorization_sha256 != bundle["authorization_sha256"]:
            raise WindowError("localized authorization binding changed")
        envelope = bundle["authorization_envelope"]
    else:
        if not args.authorization:
            raise WindowError("action authorization is required")
        authorization, authorization_sha256, envelope = verify_action_envelope(
            Path(args.authorization), Path(context["_authority_public_key"]), kind=action
        )
        bundle_raw = b""
    if localized is not None:
        localized_guard, localized_guard_sha256 = _verify_stage_envelope_value(
            bundle["guard_envelope"],
            Path(context["_authority_public_key"]),
            kind="guard_capacity",
            request_id=context["request_id"],
            approval_sha256=context["approval_sha256"],
            state_name=success_current,
        )
        if (
            localized_guard_sha256 != bundle["guard_receipt_sha256"]
            or localized_guard["outcome"] != "PASS"
            or localized_guard["identities"] != _context_identities(context)
            or localized_guard["inventories"] != _context_inventories(context)
        ):
            raise WindowError("localized guard binding changed")
    else:
        guard_path_value = getattr(args, "guard_receipt", None) or context.get(
            "_decision_guard_receipt"
        )
        if not guard_path_value:
            raise WindowError("guard receipt is required")
        _, guard_raw = _verify_guard(Path(guard_path_value), context, success_current)
        guard_envelope, _ = read_json(Path(guard_path_value), "guard capacity receipt")
        localized_guard_sha256 = sha(guard_raw)
    _verify_action_context(authorization, context, success_current)
    if args.authorization and localized is not None and os.path.lexists(args.authorization):
        _, supplied_sha256, _ = verify_action_envelope(
            Path(args.authorization), Path(context["_authority_public_key"]), kind=action
        )
        if supplied_sha256 != authorization_sha256:
            raise WindowError("changed action authorization conflicts with decision intent")
    if (
        getattr(args, "guard_receipt", None)
        and localized is not None
        and os.path.lexists(args.guard_receipt)
    ):
        _, supplied_guard_sha256 = _verify_stage_envelope_value(
            read_json(Path(args.guard_receipt), "supplied guard receipt")[0],
            Path(context["_authority_public_key"]),
            kind="guard_capacity",
            request_id=context["request_id"],
            approval_sha256=context["approval_sha256"],
            state_name=success_current,
        )
        if supplied_guard_sha256 != localized_guard_sha256:
            raise WindowError("changed guard receipt conflicts with decision intent")
    if (
        action == "rollback"
        and args.failure_evidence
        and localized is not None
        and os.path.lexists(args.failure_evidence)
    ):
        _, supplied_failure_raw = read_json(
            Path(args.failure_evidence), "supplied rollback failure evidence"
        )
        if sha(supplied_failure_raw) != localized[0]["failure_evidence_sha256"]:
            raise WindowError("changed failure evidence conflicts with decision intent")
    intent = {
        "schema": "alpha_max_window_decision_intent.v4",
        "action": action,
        "state": success_current,
        "authorization_leaf": "decision-localization.json",
        "authorization_sha256": authorization_sha256,
        "context_sha256": _context_hash(context),
        "failure_evidence_leaf": "decision-localization.json" if action == "rollback" else None,
        "failure_evidence_sha256": authorization.get("failure_evidence_sha256"),
        "localization_sha256": sha(bundle_raw) if localized is not None else None,
    }
    terminal_state = "W9_ROLLED_BACK" if action == "rollback" else "W10_FINALIZED"
    if full_current in {"W9_ROLLED_BACK", "W10_FINALIZED"}:
        if full_current != terminal_state:
            raise WindowError("opposite publication terminal is immutable")
        terminal_path = root / ("rolled-back.json" if action == "rollback" else "finalized.json")
        terminal, _ = read_json(terminal_path, "terminal receipt")
        phase = "after-rollback" if action == "rollback" else "finalize"
        _verify_open_topology(context, Path(context["_canonical_root"]), phase=phase)
        if terminal != {
            "schema": "alpha_max_window_terminal.v2",
            "action": action,
            "state": terminal_state,
            "authorization_sha256": authorization_sha256,
            "context_sha256": _context_hash(context),
        }:
            raise WindowError("terminal action is immutable")
        if action == "finalize":
            expected_bundle = _w10_finalize_bundle(
                root, context, authorization_sha256, args.authority_key_fd
            )
            stored_bundle, _ = read_json(root / W10_FINALIZE_BUNDLE_LEAF, "W10 finalize bundle")
            if stored_bundle != expected_bundle:
                raise WindowError("W10 finalize bundle is immutable")
            _validate_w10_finalize_bundle(root, context, authorization_sha256)
        print(json.dumps(terminal, sort_keys=True))
        return 0

    progress_state = "W7_ROLLING_BACK" if action == "rollback" else "W8_FINALIZING"
    if full_current in {"W7_ROLLING_BACK", "W8_FINALIZING"} and full_current != progress_state:
        raise WindowError("opposite publication decision is immutable")
    if intent_path.exists():
        old, _ = read_json(intent_path, "decision intent")
        if old != intent or full_current != progress_state:
            raise WindowError("conflicting or changed decision intent")
    else:
        if full_current != success_current:
            raise WindowError("publication decision state is invalid")
        if action == "finalize" and success_current != "W6_READY_TO_FINALIZE":
            raise WindowError("finalize requires W6")
        if action == "rollback":
            if authorization["reason"] not in REASONS.get(success_current, set()):
                raise WindowError("rollback reason is invalid in the current state")
            if localized is None:
                if not args.failure_evidence:
                    raise WindowError("rollback failure evidence is required")
                _verify_failure_evidence(args, root, context, success_current, authorization)
        now = time.time_ns()
        if localized is None and not (
            authorization["issued_ns"] - MAX_CLOCK_SKEW_NS <= now <= authorization["expires_ns"]
        ):
            raise WindowError("action authorization is not current")
        if localized is None:
            failure = None
            failure_sha256 = None
            if action == "rollback":
                failure, failure_raw = read_json(
                    Path(args.failure_evidence), "rollback failure evidence"
                )
                failure_sha256 = sha(failure_raw)
            bundle = {
                "schema": "alpha_max_window_decision_localization.v1",
                "action": action,
                "authorization_sha256": authorization_sha256,
                "authorization_envelope": envelope,
                "guard_receipt_sha256": localized_guard_sha256,
                "guard_envelope": guard_envelope,
                "failure_evidence_sha256": failure_sha256,
                "failure_evidence": failure,
            }
            write_noreplace(root / "decision-localization.json", bundle)
            localized = _decision_localization(root, action=action)
            if localized is None:
                raise WindowError("decision localization disappeared")
            bundle, bundle_raw = localized
            if bundle["authorization_sha256"] != authorization_sha256 or bundle[
                "failure_evidence_sha256"
            ] != authorization.get("failure_evidence_sha256"):
                raise WindowError("decision localization conflicts")
            intent["localization_sha256"] = sha(bundle_raw)
    active = Path(context["_canonical_root"]).resolve(strict=True)
    phase = "before"
    if action == "rollback" and identity(active) == context["predecessor_identity"]:
        phase = "after-rollback"
    elif action == "finalize" and (
        not Path(context["predecessor"]).exists()
        or (root / "predecessor-cleanup-manifest.json").exists()
    ):
        phase = "finalize"
    _verify_open_topology(context, Path(context["_canonical_root"]), phase=phase)
    write_noreplace(intent_path, intent)
    journal_name = "W7_ROLLING_BACK.json" if action == "rollback" else "W8_FINALIZING.json"
    write_noreplace(
        root / journal_name,
        {
            "schema": "alpha_max_window_action_journal.v1",
            "action": action,
            "authorization_sha256": authorization_sha256,
            "context_sha256": _context_hash(context),
            "phase": "intent-fsynced",
        },
    )
    canonical = Path(context["_canonical_root"])
    if action == "rollback":
        _, _, swap = (
            Path(context["candidate"]),
            Path(context["predecessor"]),
            Path(context["swap_temporary_path"]),
        )
        active = canonical.resolve(strict=True)
        if identity(active) == context["candidate_identity"]:
            _rename_exchange(canonical, swap)
        elif identity(active) != context["predecessor_identity"]:
            raise WindowError("rollback active generation is unauthorized")
        _fsync_dir(canonical.parent)
        _verify_open_topology(context, canonical, phase="after-rollback")
        write_noreplace(
            root / "rollback-exchange-fsynced.json",
            {
                "schema": "alpha_max_window_rollback_exchange.v1",
                "authorization_sha256": authorization_sha256,
                "context_sha256": _context_hash(context),
            },
        )
        terminal = {
            "schema": "alpha_max_window_terminal.v2",
            "action": "rollback",
            "state": "W9_ROLLED_BACK",
            "authorization_sha256": authorization_sha256,
            "context_sha256": _context_hash(context),
        }
        write_noreplace(root / "rolled-back.json", terminal)
    else:
        _finalize_cleanup(root, context, authorization_sha256)
        _verify_open_topology(context, canonical, phase="finalize")
        write_noreplace(
            root / "completed.json",
            {
                "schema": "alpha_max_publication_completed.v1",
                "request_id": context["request_id"],
                "phase": "completed",
                "authorization_sha256": authorization_sha256,
                "context_sha256": _context_hash(context),
            },
        )
        terminal = {
            "schema": "alpha_max_window_terminal.v2",
            "action": "finalize",
            "state": "W10_FINALIZED",
            "authorization_sha256": authorization_sha256,
            "context_sha256": _context_hash(context),
        }
        write_noreplace(root / "finalized.json", terminal)
        write_noreplace(
            root / W10_FINALIZE_BUNDLE_LEAF,
            _w10_finalize_bundle(root, context, authorization_sha256, args.authority_key_fd),
        )
        _validate_w10_finalize_bundle(root, context, authorization_sha256)
    print(json.dumps(terminal, sort_keys=True))
    return 0


def _preexchange_recovery_state(root: Path, canonical: Path, publisher_key: Path) -> bool:
    intent_path = root / "activation-intent.json"
    if not intent_path.exists():
        return False
    intent, _ = _verify_envelope(
        intent_path,
        publisher_key,
        schema="alpha_max_publication_stage_envelope.v1",
        domain=STAGE_DOMAIN,
        kind="activation",
        label="pre-exchange activation intent",
    )
    candidate = intent.get("candidate")
    if (
        intent.get("request_id") != root.name
        or intent.get("run_id") != RUN_ID
        or not isinstance(candidate, str)
        or not candidate
    ):
        raise WindowError("pre-exchange activation intent binding is invalid")
    return canonical.exists() and canonical.resolve(strict=True) == Path(candidate)


def _preopen_rollback_state(root: Path, publisher_key: Path) -> bool:
    intent_path = root / "pre-open-rollback-intent.json"
    receipt_path = root / "pre-open-rollback.json"
    if not intent_path.exists():
        return False
    intent, _ = _verify_envelope(
        intent_path,
        publisher_key,
        schema="alpha_max_publication_stage_envelope.v1",
        domain=STAGE_DOMAIN,
        kind="activation",
        label="pre-open rollback intent",
    )
    if (
        set(intent) != PREOPEN_ROLLBACK_FIELDS
        or intent.get("schema") != "alpha_max_publication_pre_open_rollback.v2"
        or intent.get("phase") != "rollback_intent"
        or intent.get("request_id") != root.name
    ):
        raise WindowError("pre-open rollback intent binding is invalid")
    if receipt_path.exists():
        receipt, _ = _verify_envelope(
            receipt_path,
            publisher_key,
            schema="alpha_max_publication_stage_envelope.v1",
            domain=STAGE_DOMAIN,
            kind="activation",
            label="pre-open rollback receipt",
        )
        if receipt != {**intent, "phase": "rolled_back"}:
            raise WindowError("pre-open rollback receipt conflicts")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "admit-finalize", "rollback", "finalize"))
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--authorization")
    parser.add_argument("--authority-public-key", required=True)
    parser.add_argument("--observer-public-key", required=True)
    parser.add_argument("--publisher-public-key", required=True)
    parser.add_argument("--authority-key-fd", type=int)
    parser.add_argument("--previous-guard-receipt")
    parser.add_argument("--terminal-receipt", required=True)
    parser.add_argument("--guard-receipt")
    parser.add_argument("--unit-exit-evidence")
    parser.add_argument("--cgroup-evidence")
    parser.add_argument("--failure-evidence")
    parser.add_argument("--for-rollback", action="store_true")
    args = parser.parse_args(argv)
    canonical = Path(args.canonical_root).absolute()
    root = transaction_root(canonical, args.request_id)
    if not root.is_dir() or root.is_symlink():
        raise WindowError("transaction root is unavailable")
    if _preopen_rollback_state(root, Path(args.publisher_public_key)):
        if args.command == "inspect" and not args.for_rollback:
            print("W0_PREOPEN_ROLLBACK")
            return 0
        raise WindowError("pre-open rollback state cannot authorize an action")
    if not (root / "activated.json").exists() and _preexchange_recovery_state(
        root, canonical, Path(args.publisher_public_key)
    ):
        if args.command == "inspect" and not args.for_rollback:
            print("W0_PREEXCHANGE_RECOVERY")
            return 0
        raise WindowError("pre-exchange recovery state cannot authorize an action")
    context = pass_context(
        root,
        canonical,
        Path(args.terminal_receipt),
        Path(args.authority_public_key),
        Path(args.observer_public_key),
        Path(args.publisher_public_key),
    )
    repo = ParquetMarketDataRepository(canonical)
    with repo.generation_lock(exclusive=True):
        if args.command == "admit-finalize":
            return admit_finalize(args, root, context)
        if args.command == "inspect":
            if args.for_rollback and not all(
                (
                    args.authorization,
                    args.guard_receipt,
                    args.unit_exit_evidence,
                    args.cgroup_evidence,
                )
            ):
                raise WindowError("rollback inspection evidence is incomplete")
            return inspect(args, root, context)
        if (
            not args.authorization
            and not (root / "window-decision-intent.json").exists()
            and _decision_localization(root) is None
        ):
            raise WindowError("action authorization is required")
        full_current = state(root, context)
        expected_terminal = "W9_ROLLED_BACK" if args.command == "rollback" else "W10_FINALIZED"
        if full_current in {"W9_ROLLED_BACK", "W10_FINALIZED"}:
            if full_current != expected_terminal:
                raise WindowError("opposite publication terminal is immutable")
            return decide(args, root, args.command, context)
        expected_progress = "W7_ROLLING_BACK" if args.command == "rollback" else "W8_FINALIZING"
        if full_current in {"W7_ROLLING_BACK", "W8_FINALIZING"} and (
            full_current != expected_progress
        ):
            raise WindowError("opposite publication decision is immutable")
        if (
            not (root / "window-decision-intent.json").exists()
            and _decision_localization(root) is None
        ):
            if not args.guard_receipt:
                raise WindowError("guard receipt is required")
            success_current = _success_state(root, context)
            guard, guard_raw = _verify_guard(Path(args.guard_receipt), context, success_current)
            if success_current == "W6_READY_TO_FINALIZE":
                readiness = _verify_ready_to_finalize(root, context)
                if sha(guard_raw) != readiness["guard_receipt_sha256"] and (
                    guard["sample_sequence"] <= readiness["guard_sample_sequence"]
                    or guard["sample_sequence"] != readiness["guard_sample_sequence"] + 1
                ):
                    raise WindowError(
                        "finalize guard is neither readiness guard nor continuous newer guard"
                    )
        if (
            args.command == "rollback"
            and not args.failure_evidence
            and full_current != "W7_ROLLING_BACK"
        ):
            raise WindowError("rollback failure evidence is required")
        return decide(args, root, args.command, context)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WindowError, TerminalPolicyError, OSError, ValueError) as exc:
        raise SystemExit(str(exc))
