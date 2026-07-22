#!/usr/bin/env python3
"""Authenticated terminal authority; deliberately contains no launch capability."""

from __future__ import annotations

import argparse
import calendar
import base64
import hashlib
import os
import socket
import stat
import struct
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
    finally:
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
    for key in ("claim_sha256", "observer_source_sha256"):
        if not isinstance(message[key], str) or len(message[key]) != 64:
            raise ValueError("invalid observer proof digest")
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
    now = _utc()
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
            "not_before_utc": now,
            "expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 300)),
        },
        private,
    )


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
    for key in ("authorization_id", "claim_sha256"):
        try:
            if len(authorization[key]) != 64:
                raise ValueError
            bytes.fromhex(authorization[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid authorization {key}") from exc
    if (
        type(authorization["observer_pid"]) is not int
        or authorization["observer_pid"] <= 0
        or type(authorization["observer_start_ticks"]) is not int
        or authorization["observer_start_ticks"] <= 0
    ):
        raise ValueError("invalid authorization observer identity")
    try:
        not_before = calendar.timegm(
            time.strptime(authorization["not_before_utc"], "%Y-%m-%dT%H:%M:%SZ")
        )
        expires = calendar.timegm(time.strptime(authorization["expires_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid authorization window") from exc
    if not not_before < expires or expires - not_before > 300:
        raise ValueError("invalid authorization window")
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


def _open_root(request: Any) -> int:
    root_fd = policy.open_directory_fd(request.evidence_root.path, "evidence root")
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
        os.close(root_fd)
        raise ValueError("evidence root identity drift")
    return root_fd


def _read_at(root_fd: int, name: str, label: str, *, limit: int = 64 * 1024 * 1024) -> bytes:
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
    finally:
        os.close(fd)
    return b"".join(chunks)


def _write_new(root_fd: int, name: str, data: bytes) -> None:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=root_fd,
    )
    try:
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
    finally:
        os.close(fd)
    os.fsync(root_fd)


def _append_at(root_fd: int, name: str, data: bytes) -> None:
    try:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=root_fd,
        )
    except FileNotFoundError:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=root_fd,
        )
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("unsafe terminal journal")
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short journal write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(root_fd)


def _truncate_at(root_fd: int, name: str, size: int) -> None:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_fd,
    )
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("unsafe terminal journal")
        os.ftruncate(fd, size)
        os.fsync(fd)
    finally:
        os.close(fd)


def _unlink_at(root_fd: int, name: str) -> None:
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
    try:
        server.bind(f"/proc/self/fd/{root_fd}/{name}")
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        identity = (info.st_dev, info.st_ino)
        os.chmod(name, 0o600, dir_fd=root_fd, follow_symlinks=False)
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            not stat.S_ISSOCK(info.st_mode)
            or (info.st_dev, info.st_ino) != identity
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("unsafe authority socket")
        os.fsync(root_fd)
        server.listen(1)
        return server, identity
    except BaseException:
        server.close()
        if identity is not None:
            _remove_socket(root_fd, name, identity)
        raise


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
    try:
        server, socket_identity = _bind_server(root_fd, socket_name)
        connection, _ = server.accept()
        with connection:
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
            evidence: list[Any] = []
            while True:
                event = policy.receive_packet(connection)
                if event.get("authorization_id") != authorization["authorization_id"]:
                    raise ValueError("event authorization mismatch")
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
                        policy.send_packet(
                            connection,
                            _clearance(
                                private,
                                authorization,
                                request,
                                index,
                                checked,
                            ),
                        )
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
        if server is not None:
            server.close()
        if socket_identity is not None:
            _remove_socket(root_fd, socket_name, socket_identity)
        os.close(root_fd)


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
    try:
        pending = _read_at(root_fd, PENDING_NAME, "terminal pending record")
    except FileNotFoundError:
        pending = None
    if pending is not None:
        pending_record = policy.parse_canonical_object(pending, "pending journal record")
    try:
        journal_bytes = _read_at(
            root_fd,
            request.publication.journal,
            "terminal journal",
        )
    except FileNotFoundError:
        journal_bytes = b""
    if pending is None:
        return _journal_records(journal_bytes) if journal_bytes else []

    boundary = journal_bytes.rfind(b"\n") + 1
    complete_journal = journal_bytes[:boundary]
    trailing = journal_bytes[boundary:]
    if trailing and (trailing == pending or not pending.startswith(trailing)):
        raise ValueError("terminal journal tail does not match pending record")
    records = _journal_records(complete_journal) if complete_journal else []
    if not trailing and journal_bytes.endswith(pending):
        _unlink_at(root_fd, PENDING_NAME)
        return records
    if not records:
        _validate_stored_authorization(
            pending_record,
            checkpoint,
            envelope,
            request,
            commands,
            root_fd,
        )
    else:
        authorization = records[0]
        _validate_stored_authorization(
            authorization,
            checkpoint,
            envelope,
            request,
            commands,
            root_fd,
        )
        events = records[1:]
        if pending_record.get("authorization_id") != authorization["authorization_id"]:
            raise ValueError("pending event authorization mismatch")
        candidate = (*events, pending_record)
        policy.validate_scope_artifacts(
            envelope,
            request,
            candidate,
            allow_incomplete=True,
        )
    if trailing:
        _truncate_at(root_fd, request.publication.journal, boundary)
    _append_at(root_fd, request.publication.journal, pending)
    _unlink_at(root_fd, PENDING_NAME)
    return _journal_records(_read_at(root_fd, request.publication.journal, "terminal journal"))


def recover(args: argparse.Namespace) -> int:
    checkpoint, envelope, request, commands = _context(
        args,
        require_outputs_absent=False,
    )
    private = policy.secure_private_key(args.private_key)
    if policy.public_key_id(private.public_key()) != envelope.authority_key.key_id:
        raise ValueError("authority key mismatch")
    root_fd = _open_root(request)
    try:
        records = _reconcile_pending(
            root_fd,
            checkpoint,
            envelope,
            request,
            commands,
        )
        if not records:
            raise ValueError("missing authenticated authorization")
        authorization = records[0]
        _validate_stored_authorization(
            authorization,
            checkpoint,
            envelope,
            request,
            commands,
            root_fd,
        )
        events = records[1:]
        if any(
            event.get("authorization_id") != authorization["authorization_id"] for event in events
        ):
            raise ValueError("journal event authorization mismatch")
        policy.validate_scope_artifacts(
            envelope,
            request,
            tuple(events),
            allow_incomplete=True,
        )
        state = recovery_state(events, len(commands))
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
    finally:
        os.close(root_fd)
    return 0 if state["kind"] == "SUCCEEDED" else 2


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
    verified = policy.verify_signed_receipt(args.receipt, args.public_key)
    message = verified.message
    if (
        verified.key_id != envelope.authority_key.key_id
        or message["scope"] != request.scope
        or message["request_id"] != request.request_id
        or message["checkpoint_pin_sha256"] != checkpoint.sha256
        or message["envelope_sha256"] != envelope.sha256
        or message["request_sha256"] != request.sha256
        or args.receipt != Path(request.evidence_root.path) / request.publication.receipt
    ):
        raise ValueError("terminal receipt context mismatch")
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
