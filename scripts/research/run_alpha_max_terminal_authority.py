#!/usr/bin/env python3
"""Authenticated terminal authority; deliberately contains no launch capability."""

from __future__ import annotations

import argparse
import base64
import hashlib
import fcntl
import os
import socket
import stat
import struct
import sys
import time
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from lumina_quant import alpha_max_terminal_policy as policy

SCHEMA = policy.WIRE_SCHEMA
CHALLENGE = set(policy.CHALLENGE_FIELDS)
PROOF = set(policy.OBSERVER_PROOF_FIELDS)
AUTHORIZATION = set(policy.AUTHORIZATION_FIELDS)
CLEARANCE = set(policy.COMMAND_CLEARANCE_FIELDS)
RECEIPT = set(policy.TERMINAL_RECEIPT_FIELDS)
PENDING_NAME = ".terminal-observer.pending"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _exact(value: Mapping[str, Any], fields: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("unexpected message fields")


def _ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(") ", 1)[1].split()[19])


def peer_credentials(connection: socket.socket) -> tuple[int, int, int]:
    pid, uid, _gid = struct.unpack(
        "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    )
    return pid, uid, _ticks(pid)


def _absolute_cli(path: Path) -> None:
    policy.validate_lexical_control_path(path)


def _context(
    args: argparse.Namespace,
    *,
    require_outputs_absent: bool,
) -> tuple[Any, Any, Any, tuple[tuple[str, ...], ...]]:
    paths = [
        args.policy,
        args.checkpoint,
        args.envelope,
        args.request,
        args.private_key,
        args.evidence_root,
    ]
    if hasattr(args, "socket"):
        paths.append(args.socket)
    for path in paths:
        _absolute_cli(path)
    terminal_policy = policy.load_policy(args.policy)
    checkpoint = policy.load_checkpoint(args.checkpoint, terminal_policy)
    envelope = policy.load_envelope(args.envelope, terminal_policy, checkpoint)
    request = policy.load_request(
        args.request,
        scope=args.scope,
        policy=terminal_policy,
        checkpoint=checkpoint,
        envelope=envelope,
    )
    if args.evidence_root != Path(request.evidence_root.path) or (
        hasattr(args, "socket") and args.socket != Path(request.authority_socket)
    ):
        raise ValueError("authority selector does not match request")
    policy.validate_prelaunch(
        envelope,
        request,
        require_outputs_absent=require_outputs_absent,
    )
    return checkpoint, envelope, request, policy.derive_scope_commands(envelope, request)


def _observer(envelope: Any, scope: str) -> tuple[Any, Any]:
    binding = next((item for item in envelope.observer_keys if item.scope == scope), None)
    if binding is None:
        raise ValueError("observer scope is not bound")
    return policy.public_key_from_b64(binding.public_key_b64), binding


def _bundle(request: Any, commands: tuple[tuple[str, ...], ...]) -> str:
    return policy.command_bundle_sha256(request, commands)


def _challenge(
    private: Any, checkpoint: Any, envelope: Any, request: Any, commands: Any
) -> dict[str, Any]:
    return policy.sign_message(
        "challenge",
        {
            "schema": SCHEMA,
            "type": "challenge",
            "authority_key_id": policy.public_key_id(private.public_key()),
            "scope": request.scope,
            "request_id": request.request_id,
            "checkpoint_pin_sha256": checkpoint.sha256,
            "envelope_sha256": envelope.sha256,
            "request_sha256": request.sha256,
            "command_bundle_sha256": _bundle(request, commands),
            "nonce_b64": base64.b64encode(os.urandom(32)).decode("ascii"),
            "issued_utc": _utc(),
        },
        private,
    )


def _raise_with_cleanup(
    primary: BaseException, label: str, cleanup_errors: list[BaseException]
) -> None:
    if cleanup_errors:
        cleanup = BaseExceptionGroup(f"{label} cleanup failed", cleanup_errors)
        raise BaseExceptionGroup(
            f"{label} failed and cleanup failed", [primary, cleanup]
        ) from primary
    raise primary


def _read_claim(root_fd: int, request: Any) -> tuple[dict[str, Any], bytes]:
    fd = os.open(
        request.publication.claim,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise ValueError("unsafe terminal claim")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > 65536:
                raise ValueError("terminal claim is too large")
            chunks.append(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or byte_count != after.st_size:
            raise ValueError("terminal claim changed while read")
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        try:
            os.close(fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(primary, "terminal claim read", cleanup_errors)
    else:
        os.close(fd)
    data = b"".join(chunks)
    return policy.parse_canonical_object(data, "claim"), data


def _validate_proof(
    message: Mapping[str, Any],
    challenge: Mapping[str, Any],
    envelope: Any,
    request: Any,
    peer: tuple[int, int, int],
    root_fd: int,
) -> str:
    _exact(message, PROOF)
    public, binding = _observer(envelope, request.scope)
    policy.verify_message("observer_proof", message, public)
    for key in (
        "schema",
        "authority_key_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
        "nonce_b64",
    ):
        if message[key] != challenge[key]:
            raise ValueError("observer proof binding mismatch")
    if message["type"] != "observer_proof" or message["observer_key_id"] != binding.key_id:
        raise ValueError("observer proof identity mismatch")
    if (message["observer_pid"], message["observer_uid"], message["observer_start_ticks"]) != peer:
        raise ValueError("observer peer credentials mismatch")
    if message["observer_source_sha256"] != envelope.observer_source_sha256:
        raise ValueError("observer source binding mismatch")
    policy.validate_sha256(message["claim_sha256"], "observer proof claim digest")
    policy.validate_sha256(message["observer_source_sha256"], "observer proof source digest")
    claim, claim_bytes = _read_claim(root_fd, request)
    _exact(claim, set(policy.CLAIM_FIELDS))
    if (
        _sha(claim_bytes) != message["claim_sha256"]
        or claim["request_id"] != request.request_id
        or claim["scope"] != request.scope
        or claim["checkpoint_pin_sha256"] != challenge["checkpoint_pin_sha256"]
        or claim["evidence_root"] != policy.plain(request.evidence_root)
        or (claim["observer_pid"], claim["observer_uid"], claim["observer_start_ticks"]) != peer
        or claim["schema"] != policy.CLAIM_SCHEMA
    ):
        raise ValueError("observer claim binding mismatch")
    return message["claim_sha256"]


def _authorization(
    private: Any,
    checkpoint: Any,
    envelope: Any,
    request: Any,
    commands: Any,
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    epoch = int(time.time())
    return policy.sign_message(
        "authorization",
        {
            "schema": SCHEMA,
            "type": "authorization",
            "authority_key_id": policy.public_key_id(private.public_key()),
            "authorization_id": _sha(os.urandom(32)),
            "scope": request.scope,
            "request_id": request.request_id,
            "checkpoint_pin_sha256": checkpoint.sha256,
            "envelope_sha256": envelope.sha256,
            "request_sha256": request.sha256,
            "command_bundle_sha256": _bundle(request, commands),
            "claim_sha256": proof["claim_sha256"],
            "observer_key_id": proof["observer_key_id"],
            "observer_pid": proof["observer_pid"],
            "observer_uid": proof["observer_uid"],
            "observer_start_ticks": proof["observer_start_ticks"],
            "observer_source_sha256": proof["observer_source_sha256"],
            "not_before_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch)),
            "expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch + 300)),
        },
        private,
    )


def _authorization_window(authorization: Mapping[str, Any]) -> tuple[int, int]:
    return policy.authorization_epoch_window(authorization)


def _validate_launch_intent_window(
    event: Mapping[str, Any],
    authorization: Mapping[str, Any],
    prior_clearance: Mapping[str, Any] | None,
    received_before: int,
    received_after: int,
) -> None:
    if event.get("event") != "launch_intent":
        return
    observed_at = int(
        policy.parse_utc_second(
            event.get("observed_utc"), "launch intent observation time"
        ).timestamp()
    )
    if event.get("command_index") == 0:
        not_before, expires = _authorization_window(authorization)
        if (
            received_before <= received_after
            and not_before <= observed_at < expires
            and not_before <= received_before < expires
            and not_before <= received_after < expires
        ):
            return
        raise ValueError("launch intent is outside authorization window")
    if (
        not isinstance(prior_clearance, Mapping)
        or event.get("prior_clearance") != prior_clearance
        or event.get("command_index") != prior_clearance.get("next_command_index")
    ):
        raise ValueError("launch intent clearance mismatch")
    issued_at = int(
        policy.parse_utc_second(
            prior_clearance.get("issued_utc"), "command clearance issued time"
        ).timestamp()
    )
    expires_at = issued_at + 60
    if (
        received_before <= received_after
        and issued_at <= observed_at <= expires_at
        and issued_at <= received_before <= expires_at
        and issued_at <= received_after <= expires_at
    ):
        return
    raise ValueError("launch intent is outside clearance window")


def _validate_stored_authorization(
    authorization: Mapping[str, Any],
    checkpoint: Any,
    envelope: Any,
    request: Any,
    commands: Any,
    root_fd: int,
) -> None:
    _exact(authorization, AUTHORIZATION)
    authority = policy.public_key_from_b64(envelope.authority_key.public_key_b64)
    policy.verify_message("authorization", authorization, authority)
    _observer_public, observer_binding = _observer(envelope, request.scope)
    expected = {
        "schema": SCHEMA,
        "type": "authorization",
        "authority_key_id": envelope.authority_key.key_id,
        "scope": request.scope,
        "request_id": request.request_id,
        "checkpoint_pin_sha256": checkpoint.sha256,
        "envelope_sha256": envelope.sha256,
        "request_sha256": request.sha256,
        "command_bundle_sha256": _bundle(request, commands),
        "observer_key_id": observer_binding.key_id,
        "observer_uid": os.getuid(),
        "observer_source_sha256": envelope.observer_source_sha256,
    }
    if any(authorization[key] != value for key, value in expected.items()):
        raise ValueError("authorization binding mismatch")
    policy.validate_sha256(authorization["authorization_id"], "authorization id")
    policy.validate_sha256(authorization["claim_sha256"], "authorization claim digest")
    if (
        type(authorization["observer_pid"]) is not int
        or authorization["observer_pid"] <= 0
        or type(authorization["observer_start_ticks"]) is not int
        or authorization["observer_start_ticks"] <= 0
    ):
        raise ValueError("invalid authorization observer identity")
    _authorization_window(authorization)
    claim, claim_bytes = _read_claim(root_fd, request)
    _exact(claim, set(policy.CLAIM_FIELDS))
    if (
        _sha(claim_bytes) != authorization["claim_sha256"]
        or claim["schema"] != policy.CLAIM_SCHEMA
        or claim["request_id"] != request.request_id
        or claim["scope"] != request.scope
        or claim["checkpoint_pin_sha256"] != checkpoint.sha256
        or claim["evidence_root"] != policy.plain(request.evidence_root)
        or claim["observer_pid"] != authorization["observer_pid"]
        or claim["observer_uid"] != authorization["observer_uid"]
        or claim["observer_start_ticks"] != authorization["observer_start_ticks"]
    ):
        raise ValueError("authorization claim mismatch")


def _clearance(
    private: Any,
    authorization: Mapping[str, Any],
    request: Any,
    completed: int,
    evidence: Any,
) -> dict[str, Any]:
    return policy.sign_message(
        "command_clearance",
        {
            "schema": SCHEMA,
            "type": "command_clearance",
            "authority_key_id": policy.public_key_id(private.public_key()),
            "authorization_id": authorization["authorization_id"],
            "scope": request.scope,
            "request_id": request.request_id,
            "completed_command_index": completed,
            "next_command_index": completed + 1,
            "validated_artifact_snapshot_sha256": evidence.snapshot_sha256,
            "issued_utc": _utc(),
        },
        private,
    )


def _open_root(request: Any, *, exclusive: bool = False) -> int:
    root_fd = policy.open_directory_fd(request.evidence_root.path, "evidence root")
    try:
        info = os.fstat(root_fd)
        expected = (
            request.evidence_root.st_dev,
            request.evidence_root.st_ino,
            request.evidence_root.st_uid,
            request.evidence_root.st_gid,
            request.evidence_root.mode,
        )
        actual = (
            info.st_dev,
            info.st_ino,
            info.st_uid,
            info.st_gid,
            stat.S_IMODE(info.st_mode),
        )
        if actual != expected:
            raise ValueError("evidence root identity drift")
        try:
            fcntl.flock(
                root_fd,
                (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise ValueError("terminal evidence root is active") from exc
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        try:
            os.close(root_fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(primary, "evidence root open", cleanup_errors)
    return root_fd


def _read_at_identity(
    root_fd: int, name: str, label: str, *, limit: int = 64 * 1024 * 1024
) -> tuple[bytes, tuple[int, int]]:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise ValueError(f"unsafe {label}")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > limit:
                raise ValueError(f"{label} is too large")
            chunks.append(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or byte_count != after.st_size:
            raise ValueError(f"{label} changed while read")
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        try:
            os.close(fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(primary, f"{label} read", cleanup_errors)
    else:
        os.close(fd)
    return b"".join(chunks), (before.st_dev, before.st_ino)


def _read_at(root_fd: int, name: str, label: str, *, limit: int = 64 * 1024 * 1024) -> bytes:
    data, _identity = _read_at_identity(root_fd, name, label, limit=limit)
    return data


def _cleanup_failed_new_publication(
    root_fd: int,
    name: str,
    fd: int,
    identity: tuple[int, int] | None,
    *,
    fd_open: bool,
) -> list[BaseException]:
    errors: list[BaseException] = []
    if identity is None and fd_open:
        try:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
        except BaseException as error:
            errors.append(error)
    if fd_open:
        try:
            os.close(fd)
        except BaseException as error:
            errors.append(error)
    if identity is None:
        errors.append(ValueError("cannot verify failed publication identity"))
    else:
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != identity or not stat.S_ISREG(info.st_mode):
                raise ValueError("failed publication identity changed")
            os.unlink(name, dir_fd=root_fd)
        except BaseException as error:
            errors.append(error)
    try:
        os.fsync(root_fd)
    except BaseException as error:
        errors.append(error)
    return errors


def _write_new(root_fd: int, name: str, data: bytes) -> None:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=root_fd,
    )
    identity: tuple[int, int] | None = None
    fd_open = True
    try:
        info = os.fstat(fd)
        identity = (info.st_dev, info.st_ino)
        os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("unsafe new publication")
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short receipt write")
            view = view[written:]
        os.fsync(fd)
        fd_open = False
        os.close(fd)
        os.fsync(root_fd)
    except BaseException as primary:
        cleanup_errors = _cleanup_failed_new_publication(
            root_fd, name, fd, identity, fd_open=fd_open
        )
        _raise_with_cleanup(primary, "new publication", cleanup_errors)


def _validate_journal_fd(fd: int, expected_identity: tuple[int, int]) -> None:
    info = os.fstat(fd)
    if (
        (info.st_dev, info.st_ino) != expected_identity
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise ValueError("unsafe terminal journal")


def _append_at(
    root_fd: int,
    name: str,
    data: bytes,
    expected_identity: tuple[int, int] | None,
) -> tuple[int, int]:
    if expected_identity is None:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=root_fd,
        )
    else:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=root_fd,
        )
    try:
        if expected_identity is None:
            info = os.fstat(fd)
            expected_identity = (info.st_dev, info.st_ino)
        _validate_journal_fd(fd, expected_identity)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short journal write")
            view = view[written:]
        os.fsync(fd)
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        try:
            os.close(fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(primary, "terminal journal append", cleanup_errors)
    else:
        try:
            os.close(fd)
        except BaseException as primary:
            cleanup_errors: list[BaseException] = []
            try:
                os.fsync(root_fd)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            _raise_with_cleanup(primary, "terminal journal append", cleanup_errors)
        os.fsync(root_fd)
    return expected_identity


def _truncate_at(root_fd: int, name: str, size: int, expected_identity: tuple[int, int]) -> None:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_fd,
    )
    try:
        _validate_journal_fd(fd, expected_identity)
        os.ftruncate(fd, size)
        os.fsync(fd)
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        try:
            os.close(fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(primary, "terminal journal truncate", cleanup_errors)
    else:
        os.close(fd)


def _unlink_at(root_fd: int, name: str, expected_identity: tuple[int, int]) -> None:
    info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if (
        (info.st_dev, info.st_ino) != expected_identity
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise ValueError("terminal journal identity changed")
    os.unlink(name, dir_fd=root_fd)
    os.fsync(root_fd)


def _bind_server(root_fd: int, name: str) -> tuple[socket.socket, tuple[int, int]]:
    try:
        os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("authority socket already exists")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    identity: tuple[int, int] | None = None
    bound = False
    try:
        old_umask = os.umask(0o177)
        try:
            server.bind(f"/proc/self/fd/{root_fd}/{name}")
            bound = True
        finally:
            os.umask(old_umask)
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        identity = (info.st_dev, info.st_ino)
        if (
            not stat.S_ISSOCK(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("unsafe authority socket")
        os.fsync(root_fd)
        server.listen(1)
        return server, identity
    except BaseException as bind_error:
        cleanup_errors: list[BaseException] = []
        try:
            server.close()
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if bound and identity is None:
            cleanup_errors.append(ValueError("bound authority socket identity is unknown"))
        if identity is not None:
            try:
                _remove_socket(root_fd, name, identity)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(bind_error, "authority socket bind", cleanup_errors)


def _remove_socket(root_fd: int, name: str, identity: tuple[int, int]) -> None:
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (info.st_dev, info.st_ino) != identity or not stat.S_ISSOCK(info.st_mode):
        raise ValueError("authority socket identity changed")
    os.unlink(name, dir_fd=root_fd)
    os.fsync(root_fd)


def _receipt(
    private: Any,
    checkpoint: Any,
    envelope: Any,
    request: Any,
    authorization: Mapping[str, Any],
    events: list[dict[str, Any]],
    terminal_state: dict[str, Any],
    root_fd: int,
) -> dict[str, Any]:
    journal_bytes = _read_at(
        root_fd,
        request.publication.journal,
        "terminal journal",
    )
    if _journal_records(journal_bytes) != [authorization, *events]:
        raise ValueError("terminal journal does not match receipt records")
    results = policy.validate_scope_artifacts(
        envelope, request, tuple(events), allow_incomplete=True
    )
    return policy.sign_message(
        "terminal_receipt",
        {
            "schema": SCHEMA,
            "type": "terminal_receipt",
            "authority_key_id": policy.public_key_id(private.public_key()),
            "authorization_id": authorization["authorization_id"],
            "scope": request.scope,
            "request_id": request.request_id,
            "checkpoint_pin_sha256": checkpoint.sha256,
            "envelope_sha256": envelope.sha256,
            "request_sha256": request.sha256,
            "claim_sha256": authorization["claim_sha256"],
            "observer_key_id": authorization["observer_key_id"],
            "observer_pid": authorization["observer_pid"],
            "observer_start_ticks": authorization["observer_start_ticks"],
            "command_bundle_sha256": authorization["command_bundle_sha256"],
            "events_sha256": _sha(policy.canonical_bytes(events)),
            "journal_sha256": _sha(journal_bytes),
            "prerequisites": [policy.plain(item) for item in request.prerequisites],
            "target_results": [policy.plain(item) for item in results],
            "terminal_state": terminal_state,
            "publication": policy.plain(request.publication),
            "created_utc": _utc(),
        },
        private,
    )


def serve(args: argparse.Namespace) -> int:
    checkpoint, envelope, request, commands = _context(
        args,
        require_outputs_absent=True,
    )
    private = policy.secure_private_key(args.private_key)
    if policy.public_key_id(private.public_key()) != envelope.authority_key.key_id:
        raise ValueError("authority key mismatch")
    root_fd = _open_root(request)
    socket_name = Path(request.authority_socket).name
    server: socket.socket | None = None
    socket_identity: tuple[int, int] | None = None
    connection: socket.socket | None = None
    try:
        policy.validate_prelaunch(
            envelope,
            request,
            require_outputs_absent=True,
        )
        server, socket_identity = _bind_server(root_fd, socket_name)
        connection, _ = server.accept()
        if connection is not None:
            peer = peer_credentials(connection)
            challenge = _challenge(private, checkpoint, envelope, request, commands)
            policy.send_packet(connection, challenge)
            proof = policy.receive_packet(connection)
            _validate_proof(proof, challenge, envelope, request, peer, root_fd)
            authorization = _authorization(
                private,
                checkpoint,
                envelope,
                request,
                commands,
                proof,
            )
            policy.send_packet(connection, authorization)
            events: list[dict[str, Any]] = []
            prior_clearance: dict[str, Any] | None = None
            evidence: list[Any] = []
            while True:
                received_before = int(time.time())
                event = policy.receive_packet(connection)
                received_after = int(time.time())
                if event.get("authorization_id") != authorization["authorization_id"]:
                    raise ValueError("event authorization mismatch")
                _validate_launch_intent_window(
                    event,
                    authorization,
                    prior_clearance,
                    received_before,
                    received_after,
                )
                candidate = (*events, event)
                policy.validate_scope_artifacts(
                    envelope,
                    request,
                    candidate,
                    allow_incomplete=True,
                )
                events.append(event)
                if event["event"] == "start_failed":
                    state: dict[str, Any] | None = {
                        "kind": "START_FAILED",
                        "command_index": event["command_index"],
                        "errno": event["errno"],
                    }
                elif event["event"] == "child_exited" and event["return_code"] != 0:
                    state = {
                        "kind": "FAILED",
                        "failed_command_index": event["command_index"],
                    }
                else:
                    state = None
                if state is not None:
                    receipt = _receipt(
                        private,
                        checkpoint,
                        envelope,
                        request,
                        authorization,
                        events,
                        state,
                        root_fd,
                    )
                    _write_new(
                        root_fd,
                        request.publication.receipt,
                        policy.canonical_bytes(receipt),
                    )
                    policy.send_packet(connection, receipt)
                    return 2
                if event["event"] == "child_exited":
                    index = event["command_index"]
                    policy.validate_prelaunch(
                        envelope,
                        request,
                        require_outputs_absent=False,
                    )
                    checked = policy.validate_completed_command(
                        envelope,
                        request,
                        index,
                        evidence[-1] if evidence else None,
                    )
                    evidence.append(checked)
                    if index + 1 < len(commands):
                        clearance = _clearance(
                            private,
                            authorization,
                            request,
                            index,
                            checked,
                        )
                        policy.send_packet(connection, clearance)
                        prior_clearance = clearance
                    else:
                        receipt = _receipt(
                            private,
                            checkpoint,
                            envelope,
                            request,
                            authorization,
                            events,
                            {"kind": "SUCCEEDED"},
                            root_fd,
                        )
                        _write_new(
                            root_fd,
                            request.publication.receipt,
                            policy.canonical_bytes(receipt),
                        )
                        policy.send_packet(connection, receipt)
                        return 0
    finally:
        cleanup_errors: list[BaseException] = []
        if connection is not None:
            try:
                connection.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if server is not None:
            try:
                server.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if socket_identity is not None:
            try:
                _remove_socket(root_fd, socket_name, socket_identity)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            os.close(root_fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        primary_error = sys.exception()
        if primary_error is not None:
            if cleanup_errors:
                cleanup_error = BaseExceptionGroup(
                    "authority server cleanup failed", cleanup_errors
                )
                raise BaseExceptionGroup(
                    "authority server failed and cleanup failed",
                    [primary_error, cleanup_error],
                ) from primary_error
        elif cleanup_errors:
            raise BaseExceptionGroup("authority server cleanup failed", cleanup_errors)


def _journal_records(data: bytes) -> list[dict[str, Any]]:
    if not data:
        raise ValueError("terminal journal is empty")
    return [
        policy.parse_canonical_object(line, "journal record")
        for line in data.splitlines(keepends=True)
    ]


def _reconcile_pending(
    root_fd: int,
    checkpoint: Any,
    envelope: Any,
    request: Any,
    commands: Any,
) -> list[dict[str, Any]]:
    pending_identity: tuple[int, int] | None
    try:
        pending, pending_identity = _read_at_identity(
            root_fd, PENDING_NAME, "terminal pending record"
        )
    except FileNotFoundError:
        pending = None
        pending_identity = None
        pending_record = None
    else:
        pending_record = policy.parse_canonical_object(pending, "pending journal record")

    journal_identity: tuple[int, int] | None
    try:
        journal_bytes, journal_identity = _read_at_identity(
            root_fd,
            request.publication.journal,
            "terminal journal",
        )
    except FileNotFoundError:
        journal_bytes = b""
        journal_identity = None

    boundary = journal_bytes.rfind(b"\n") + 1
    complete_journal = journal_bytes[:boundary]
    trailing = journal_bytes[boundary:]
    if trailing and (pending is None or trailing == pending or not pending.startswith(trailing)):
        raise ValueError("terminal journal tail does not match pending authorization")

    records = _journal_records(complete_journal) if complete_journal else []
    if (
        len(records) > 1
        or (records and records[0].get("type") != "authorization")
        or any("event" in record for record in records)
        or (
            pending_record is not None
            and (pending_record.get("type") != "authorization" or "event" in pending_record)
        )
    ):
        raise ValueError("recovery cannot prove authorization-only history")

    authorization = records[0] if records else pending_record
    if authorization is not None:
        _validate_stored_authorization(
            authorization,
            checkpoint,
            envelope,
            request,
            commands,
            root_fd,
        )

    if pending is None:
        if trailing:
            raise ValueError("terminal journal has an incomplete record")
        return records

    if pending_identity is None or pending_record is None:
        raise ValueError("pending authorization identity is missing")
    fully_appended = bool(records)
    if fully_appended and (trailing or journal_bytes != pending or pending_record != records[0]):
        raise ValueError("pending authorization does not match terminal journal")

    current_pending, current_pending_identity = _read_at_identity(
        root_fd, PENDING_NAME, "terminal pending record"
    )
    if current_pending != pending or current_pending_identity != pending_identity:
        raise ValueError("pending journal identity changed")
    if journal_identity is not None:
        current_journal, current_journal_identity = _read_at_identity(
            root_fd,
            request.publication.journal,
            "terminal journal",
        )
        if current_journal != journal_bytes or current_journal_identity != journal_identity:
            raise ValueError("terminal journal changed before recovery mutation")

    if fully_appended:
        _unlink_at(root_fd, PENDING_NAME, pending_identity)
        return records
    if trailing:
        if journal_identity is None:
            raise ValueError("terminal journal identity is missing")
        _truncate_at(root_fd, request.publication.journal, boundary, journal_identity)
    journal_identity = _append_at(
        root_fd,
        request.publication.journal,
        pending,
        journal_identity,
    )
    repaired_journal, repaired_identity = _read_at_identity(
        root_fd,
        request.publication.journal,
        "terminal journal",
    )
    current_pending, current_pending_identity = _read_at_identity(
        root_fd, PENDING_NAME, "terminal pending record"
    )
    if (
        repaired_journal != pending
        or repaired_identity != journal_identity
        or current_pending != pending
        or current_pending_identity != pending_identity
    ):
        raise ValueError("recovered authorization changed before pending unlink")
    _unlink_at(root_fd, PENDING_NAME, pending_identity)
    return [pending_record]


def recover(args: argparse.Namespace) -> int:
    checkpoint, envelope, request, commands = _context(
        args,
        require_outputs_absent=False,
    )
    private = policy.secure_private_key(args.private_key)
    if policy.public_key_id(private.public_key()) != envelope.authority_key.key_id:
        raise ValueError("authority key mismatch")
    root_fd = _open_root(request, exclusive=True)
    try:
        policy.validate_prelaunch(
            envelope,
            request,
            require_outputs_absent=False,
        )
        records = _reconcile_pending(
            root_fd,
            checkpoint,
            envelope,
            request,
            commands,
        )
        if len(records) != 1:
            raise ValueError("recovery requires exactly one authenticated authorization")
        authorization = records[0]
        _validate_stored_authorization(
            authorization,
            checkpoint,
            envelope,
            request,
            commands,
            root_fd,
        )
        events: list[dict[str, Any]] = []
        policy.validate_scope_artifacts(
            envelope,
            request,
            tuple(events),
            allow_incomplete=True,
        )
        state = {
            "kind": "UNAUTHENTICATED_TERMINAL",
            "last_authenticated_sequence": 0,
        }
        receipt = _receipt(
            private,
            checkpoint,
            envelope,
            request,
            authorization,
            events,
            state,
            root_fd,
        )
        if (
            receipt["terminal_state"] != state
            or receipt["events_sha256"] != _sha(policy.canonical_bytes(events))
            or receipt["target_results"] != []
        ):
            raise ValueError("authorization-only recovery receipt mismatch")
        _write_new(
            root_fd,
            request.publication.receipt,
            policy.canonical_bytes(receipt),
        )
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        try:
            os.close(root_fd)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _raise_with_cleanup(primary, "recovery", cleanup_errors)
    else:
        os.close(root_fd)
    return 2


def recovery_state(events: list[dict[str, Any]], command_count: int) -> dict[str, Any]:
    """Classify authenticated crash prefixes without launching or advancing work."""
    if not events:
        return {"kind": "UNAUTHENTICATED_TERMINAL", "last_authenticated_sequence": 0}
    terminal = {event["command_index"]: event for event in events}
    for index in range(command_count):
        event = terminal.get(index)
        if event is None:
            return {
                "kind": "UNAUTHENTICATED_TERMINAL",
                "last_authenticated_sequence": events[-1]["sequence"],
            }
        if event["event"] == "start_failed":
            return {"kind": "START_FAILED", "command_index": index, "errno": event["errno"]}
        if event["event"] == "child_exited":
            if event["return_code"] != 0:
                return {"kind": "FAILED", "failed_command_index": index}
            continue
        if event["event"] == "launch_intent":
            return {"kind": "START_UNKNOWN", "command_index": index}
        if event["event"] == "child_started":
            return {
                "kind": "OBSERVER_LOST",
                "command_index": index,
                "child_pid": event["child_pid"],
                "child_start_ticks": event["child_start_ticks"],
            }
        return {
            "kind": "UNAUTHENTICATED_TERMINAL",
            "last_authenticated_sequence": events[-1]["sequence"],
        }
    return {"kind": "SUCCEEDED"}


def verify_receipt(args: argparse.Namespace) -> int:
    for path in (
        args.policy,
        args.checkpoint,
        args.envelope,
        args.request,
        args.public_key,
        args.receipt,
    ):
        _absolute_cli(path)
    terminal_policy = policy.load_policy(args.policy)
    checkpoint = policy.load_checkpoint(args.checkpoint, terminal_policy)
    envelope = policy.load_envelope(args.envelope, terminal_policy, checkpoint)
    request = policy.load_request(
        args.request,
        scope=args.scope,
        policy=terminal_policy,
        checkpoint=checkpoint,
        envelope=envelope,
    )
    policy.validate_prelaunch(
        envelope,
        request,
        require_outputs_absent=False,
    )
    commands = policy.derive_scope_commands(envelope, request)
    expected_receipt = Path(request.evidence_root.path) / request.publication.receipt
    if args.receipt != expected_receipt:
        raise ValueError("terminal receipt context mismatch")
    _observer_public, observer_binding = _observer(envelope, request.scope)
    verified = policy.verify_signed_receipt(args.receipt, args.public_key)
    message = verified.message
    authorization = verified.authorization
    _exact(message, RECEIPT)
    _exact(authorization, AUTHORIZATION)
    context = {
        "authority_key_id": envelope.authority_key.key_id,
        "scope": request.scope,
        "request_id": request.request_id,
        "checkpoint_pin_sha256": checkpoint.sha256,
        "envelope_sha256": envelope.sha256,
        "request_sha256": request.sha256,
        "command_bundle_sha256": _bundle(request, commands),
        "observer_key_id": observer_binding.key_id,
    }
    if (
        verified.key_id != envelope.authority_key.key_id
        or any(message[key] != value for key, value in context.items())
        or any(authorization[key] != value for key, value in context.items())
        or authorization["observer_source_sha256"] != envelope.observer_source_sha256
        or message["authorization_id"] != authorization["authorization_id"]
        or message["claim_sha256"] != authorization["claim_sha256"]
        or message["publication"] != policy.plain(request.publication)
        or message["prerequisites"] != [policy.plain(item) for item in request.prerequisites]
        or args.receipt != expected_receipt
    ):
        raise ValueError("terminal receipt context mismatch")
    results = policy.validate_scope_artifacts(
        envelope, request, verified.events, allow_incomplete=True
    )
    if message["target_results"] != policy.plain(results) or message[
        "terminal_state"
    ] != recovery_state(list(verified.events), len(commands)):
        raise ValueError("terminal receipt result mismatch")
    return 0


def _add_context_arguments(
    item: argparse.ArgumentParser,
    *,
    private_key: bool,
    socket_path: bool,
    evidence_root: bool,
) -> None:
    names = ["policy", "checkpoint", "envelope", "request"]
    if private_key:
        names.append("private_key")
    if socket_path:
        names.append("socket")
    if evidence_root:
        names.append("evidence_root")
    for name in names:
        item.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    item.add_argument(
        "--scope",
        required=True,
        choices=("acquisition", "phase_preparation", "one_touch"),
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="action", required=True)
    serve_parser = sub.add_parser("serve")
    _add_context_arguments(
        serve_parser,
        private_key=True,
        socket_path=True,
        evidence_root=True,
    )
    recover_parser = sub.add_parser("recover")
    _add_context_arguments(
        recover_parser,
        private_key=True,
        socket_path=False,
        evidence_root=True,
    )
    verify_parser = sub.add_parser("verify")
    _add_context_arguments(
        verify_parser,
        private_key=False,
        socket_path=False,
        evidence_root=False,
    )
    verify_parser.add_argument("--receipt", required=True, type=Path)
    verify_parser.add_argument("--public-key", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"serve": serve, "recover": recover, "verify": verify_receipt}[args.action](args)
    except (OSError, ValueError, policy.TerminalPolicyError) as exc:
        print(f"terminal authority: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
