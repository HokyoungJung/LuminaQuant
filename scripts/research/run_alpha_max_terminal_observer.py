#!/usr/bin/env python3
"""The sole process launcher for Alpha-Max terminal scopes."""

from __future__ import annotations

import calendar
import argparse
import base64
import errno
import hashlib
import os
import socket
import stat
import subprocess
import time
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from lumina_quant import alpha_max_terminal_policy as policy

SCHEMA = policy.WIRE_SCHEMA
PENDING_NAME = ".terminal-observer.pending"
CHALLENGE = set(policy.CHALLENGE_FIELDS)
PROOF = set(policy.OBSERVER_PROOF_FIELDS)
AUTHORIZATION = set(policy.AUTHORIZATION_FIELDS)
CLEARANCE = set(policy.COMMAND_CLEARANCE_FIELDS)
RECEIPT = set(policy.TERMINAL_RECEIPT_FIELDS)


class ObserverError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _exact(value: Mapping[str, Any], fields: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ObserverError("unexpected message fields")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short write")
        view = view[count:]


def _open_root(path: Path, identity: Any) -> int:
    fd = policy.open_directory_fd(path, "evidence root")
    info = os.fstat(fd)
    expected = (identity.st_dev, identity.st_ino, identity.st_uid, identity.st_gid, identity.mode)
    actual = (info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
    if actual != expected:
        os.close(fd)
        raise ObserverError("evidence root identity drift")
    return fd


def _relative(root: Path, path: str) -> str:
    if not isinstance(path, str) or Path(path).is_absolute() or Path(path).name != path:
        raise ObserverError("publication must be an evidence-root leaf")
    return path


def _new(root_fd: int, name: str, data: bytes, mode: int = 0o600) -> None:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode,
        dir_fd=root_fd,
    )
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) != mode
        ):
            raise ObserverError("unsafe new publication")
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(root_fd)


def _append(root_fd: int, record: dict[str, Any], journal: str) -> None:
    data = policy.canonical_bytes(record)
    _new(root_fd, PENDING_NAME, data)
    fd = None
    try:
        try:
            fd = os.open(
                journal,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=root_fd,
            )
        except FileExistsError:
            fd = os.open(
                journal,
                os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=root_fd,
            )
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.getuid()
            or info.st_gid != os.getgid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ObserverError("unsafe terminal journal")
        _write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.unlink(PENDING_NAME, dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        if fd is not None:
            os.close(fd)


def _claim(root_fd: int, name: str, checkpoint: Any, request: Any) -> str:
    data = policy.canonical_bytes(
        {
            "schema": policy.CLAIM_SCHEMA,
            "request_id": request.request_id,
            "scope": request.scope,
            "checkpoint_pin_sha256": checkpoint.sha256,
            "evidence_root": policy.plain(request.evidence_root),
            "observer_pid": os.getpid(),
            "observer_uid": os.getuid(),
            "observer_start_ticks": _ticks(os.getpid()),
            "created_utc": _utc(),
        }
    )
    _exact(policy.parse_canonical_object(data, "claim"), set(policy.CLAIM_FIELDS))
    _new(root_fd, name, data)
    return _sha(data)


def _absent(root_fd: int, names: Sequence[str]) -> None:
    for name in names:
        try:
            os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise ObserverError("pre-existing publication")


def _ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(") ", 1)[1].split()[19])


def _identity(fd: int, name: str) -> dict[str, int | str | bool]:
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or os.isatty(fd)
    ):
        raise ObserverError("unsafe child log")
    return {
        "path": name,
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
        "isatty": False,
    }


def _event(
    private: Any,
    authorization: Mapping[str, Any],
    preflight: Any,
    sequence: int,
    kind: str,
    prior_clearance: dict[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    return policy.sign_message(
        "process_event",
        {
            "schema": SCHEMA,
            "type": "process_event",
            "event": kind,
            "authorization_id": authorization["authorization_id"],
            "sequence": sequence,
            "command_index": preflight.command_index,
            "argv_sha256": preflight.argv_sha256,
            "environment_sha256": preflight.environment_sha256,
            "prior_clearance": prior_clearance,
            "observed_utc": _utc(),
            **extra,
        },
        private,
    )


def _source_digest() -> str:
    fd = os.open(__file__, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ObserverError("observer source is unsafe")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            byte_count += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or byte_count != after.st_size:
            raise ObserverError("observer source changed while read")
    finally:
        os.close(fd)
    return digest.hexdigest()


def _absolute_cli(path: Path) -> None:
    policy.validate_lexical_control_path(path)


def _prepare(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, tuple[Any, ...], int, Any]:
    paths = (
        args.policy,
        args.checkpoint,
        args.envelope,
        args.request,
        args.authority_socket,
        args.observer_private_key,
        args.evidence_root,
    )
    for item in paths:
        _absolute_cli(item)
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
    policy.validate_prelaunch(envelope, request)
    if args.evidence_root != Path(request.evidence_root.path):
        raise ObserverError("evidence root selector mismatch")
    if args.authority_socket != Path(request.authority_socket):
        raise ObserverError("authority socket selector mismatch")
    key = policy.secure_private_key(args.observer_private_key)
    binding = next((item for item in envelope.observer_keys if item.scope == request.scope), None)
    if binding is None or policy.public_key_id(key.public_key()) != binding.key_id:
        raise ObserverError("observer key mismatch")
    root_fd = _open_root(args.evidence_root, request.evidence_root)
    return (
        checkpoint,
        envelope,
        request,
        key,
        policy.derive_command_preflight(envelope, request),
        root_fd,
        binding,
    )


def _read_leaf(
    root_fd: int,
    name: str,
    label: str,
    *,
    limit: int = 64 * 1024 * 1024,
) -> bytes:
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
            raise ObserverError(f"unsafe {label}")
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > limit:
                raise ObserverError(f"{label} is too large")
            chunks.append(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or byte_count != after.st_size:
            raise ObserverError(f"{label} changed while read")
    finally:
        os.close(fd)
    return b"".join(chunks)


def _validate_challenge(
    message: Mapping[str, Any],
    authority: Any,
    checkpoint: Any,
    envelope: Any,
    request: Any,
    commands: Any,
) -> None:
    _exact(message, CHALLENGE)
    policy.verify_message("challenge", message, authority)
    expected = {
        "schema": SCHEMA,
        "type": "challenge",
        "authority_key_id": policy.public_key_id(authority),
        "scope": request.scope,
        "request_id": request.request_id,
        "checkpoint_pin_sha256": checkpoint.sha256,
        "envelope_sha256": envelope.sha256,
        "request_sha256": request.sha256,
        "command_bundle_sha256": policy.command_bundle_sha256(request, commands),
    }
    if any(message[key] != value for key, value in expected.items()):
        raise ObserverError("challenge binding mismatch")
    try:
        nonce = base64.b64decode(message["nonce_b64"], validate=True)
        issued = calendar.timegm(time.strptime(message["issued_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError) as exc:
        raise ObserverError("invalid challenge freshness") from exc
    if len(nonce) != 32 or not 0 <= time.time() - issued <= 60:
        raise ObserverError("invalid challenge freshness")


def _validate_authorization(
    message: Mapping[str, Any],
    authority: Any,
    challenge: Mapping[str, Any],
    claim: str,
    binding: Any,
    source: str,
) -> None:
    _exact(message, AUTHORIZATION)
    policy.verify_message("authorization", message, authority)
    for key in (
        "authority_key_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
    ):
        if message[key] != challenge[key]:
            raise ObserverError("authorization binding mismatch")
    try:
        if len(message["authorization_id"]) != 64:
            raise ValueError
        bytes.fromhex(message["authorization_id"])
    except (TypeError, ValueError) as exc:
        raise ObserverError("invalid authorization id") from exc
    if (
        message["schema"] != SCHEMA
        or message["type"] != "authorization"
        or message["claim_sha256"] != claim
        or message["observer_key_id"] != binding.key_id
        or message["observer_pid"] != os.getpid()
        or message["observer_uid"] != os.getuid()
        or message["observer_start_ticks"] != _ticks(os.getpid())
        or message["observer_source_sha256"] != source
    ):
        raise ObserverError("authorization observer mismatch")
    try:
        not_before = calendar.timegm(time.strptime(message["not_before_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        expires = calendar.timegm(time.strptime(message["expires_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError) as exc:
        raise ObserverError("invalid authorization window") from exc
    now = time.time()
    if not not_before < expires or not_before > now or expires < now or expires - not_before > 300:
        raise ObserverError("authorization window is not currently bounded")


def _validate_receipt(
    message: Mapping[str, Any],
    authority: Any,
    authorization: Mapping[str, Any],
    checkpoint: Any,
    envelope: Any,
    request: Any,
    events: list[dict[str, Any]],
    terminal_kind: str,
    root_fd: int,
) -> None:
    _exact(message, RECEIPT)
    policy.verify_message("terminal_receipt", message, authority)
    policy.validate_terminal_state(message["terminal_state"])
    expected = {
        "schema": SCHEMA,
        "type": "terminal_receipt",
        "authority_key_id": policy.public_key_id(authority),
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
        "prerequisites": [policy.plain(item) for item in request.prerequisites],
        "publication": policy.plain(request.publication),
    }
    journal_bytes = _read_leaf(
        root_fd,
        request.publication.journal,
        "terminal journal",
    )
    if (
        any(message[key] != value for key, value in expected.items())
        or message["journal_sha256"] != _sha(journal_bytes)
        or message["terminal_state"]["kind"] != terminal_kind
        or not isinstance(message["target_results"], list)
    ):
        raise ObserverError("receipt binding mismatch")
    receipt_bytes = _read_leaf(
        root_fd,
        request.publication.receipt,
        "terminal receipt",
    )
    if receipt_bytes != policy.canonical_bytes(message):
        raise ObserverError("terminal receipt readback mismatch")


def _digest_log(fd: int) -> tuple[str, int]:
    os.fsync(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        part = os.read(fd, 65536)
        if not part:
            break
        byte_count += len(part)
        digest.update(part)
    return digest.hexdigest(), byte_count


def _validate_clearance(
    message: Mapping[str, Any],
    authority: Any,
    authorization: Mapping[str, Any],
    request: Any,
    completed_index: int,
    evidence: Any,
) -> None:
    _exact(message, CLEARANCE)
    policy.verify_message("command_clearance", message, authority)
    expected = {
        "schema": SCHEMA,
        "type": "command_clearance",
        "authority_key_id": policy.public_key_id(authority),
        "authorization_id": authorization["authorization_id"],
        "scope": request.scope,
        "request_id": request.request_id,
        "completed_command_index": completed_index,
        "next_command_index": completed_index + 1,
        "validated_artifact_snapshot_sha256": evidence.snapshot_sha256,
    }
    if any(message[key] != value for key, value in expected.items()):
        raise ObserverError("invalid command clearance")
    try:
        issued = calendar.timegm(time.strptime(message["issued_utc"], "%Y-%m-%dT%H:%M:%SZ"))
    except (TypeError, ValueError) as exc:
        raise ObserverError("invalid command clearance timestamp") from exc
    if not 0 <= time.time() - issued <= 60:
        raise ObserverError("stale command clearance")


def run(args: argparse.Namespace) -> int:
    checkpoint, envelope, request, private, preflights, root_fd, binding = _prepare(args)
    root = Path(request.evidence_root.path)
    claim_name = _relative(root, request.publication.claim)
    journal = _relative(root, request.publication.journal)
    stdout_names = tuple(_relative(root, item) for item in request.publication.stdout)
    stderr_names = tuple(_relative(root, item) for item in request.publication.stderr)
    receipt_name = _relative(root, request.publication.receipt)
    source = _source_digest()
    if source != envelope.observer_source_sha256:
        raise ObserverError("observer source binding mismatch")
    try:
        _absent(
            root_fd,
            (
                claim_name,
                journal,
                receipt_name,
                PENDING_NAME,
                *stdout_names,
                *stderr_names,
            ),
        )
        claim = _claim(root_fd, claim_name, checkpoint, request)
        authority = policy.public_key_from_b64(envelope.authority_key.public_key_b64)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            connection.connect(args.authority_socket)
            challenge = policy.receive_packet(connection)
            _validate_challenge(
                challenge,
                authority,
                checkpoint,
                envelope,
                request,
                tuple(item.argv for item in preflights),
            )
            proof = policy.sign_message(
                "observer_proof",
                {
                    "schema": SCHEMA,
                    "type": "observer_proof",
                    "authority_key_id": challenge["authority_key_id"],
                    "scope": request.scope,
                    "request_id": request.request_id,
                    "checkpoint_pin_sha256": checkpoint.sha256,
                    "envelope_sha256": envelope.sha256,
                    "request_sha256": request.sha256,
                    "command_bundle_sha256": challenge["command_bundle_sha256"],
                    "nonce_b64": challenge["nonce_b64"],
                    "observer_key_id": binding.key_id,
                    "observer_pid": os.getpid(),
                    "observer_uid": os.getuid(),
                    "observer_start_ticks": _ticks(os.getpid()),
                    "observer_source_sha256": source,
                    "claim_sha256": claim,
                },
                private,
            )
            policy.send_packet(connection, proof)
            authorization = policy.receive_packet(connection)
            _validate_authorization(
                authorization,
                authority,
                challenge,
                claim,
                binding,
                source,
            )
            _append(root_fd, authorization, journal)
            events: list[dict[str, Any]] = []
            previous = None
            prior_clearance = None
            sequence = 0
            for index, preflight in enumerate(preflights):
                policy.validate_prelaunch(
                    envelope,
                    request,
                    require_outputs_absent=index == 0,
                )
                policy.validate_command_semantics(
                    envelope,
                    request,
                    index,
                    preflight.argv,
                    request.environment,
                )
                intent = _event(
                    private,
                    authorization,
                    preflight,
                    sequence,
                    "launch_intent",
                    prior_clearance,
                )
                sequence += 1
                _append(root_fd, intent, journal)
                events.append(intent)
                policy.send_packet(connection, intent)

                stdout_fd: int | None = None
                stderr_fd: int | None = None
                try:
                    try:
                        stdout_fd = os.open(
                            stdout_names[index],
                            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                            0o600,
                            dir_fd=root_fd,
                        )
                        stderr_fd = os.open(
                            stderr_names[index],
                            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                            0o600,
                            dir_fd=root_fd,
                        )
                        before_out = _identity(stdout_fd, stdout_names[index])
                        before_err = _identity(stderr_fd, stderr_names[index])
                        environment = policy.plain(request.environment)
                        child = subprocess.Popen(
                            preflight.argv,
                            cwd=request.repository_root.path,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout_fd,
                            stderr=stderr_fd,
                            close_fds=True,
                            pass_fds=(),
                        )
                    except (OSError, ObserverError, policy.TerminalPolicyError) as exc:
                        error_number = exc.errno if isinstance(exc, OSError) else errno.EIO
                        failed = _event(
                            private,
                            authorization,
                            preflight,
                            sequence,
                            "start_failed",
                            prior_clearance,
                            errno=error_number or errno.EIO,
                            error_name=type(exc).__name__,
                        )
                        sequence += 1
                        _append(root_fd, failed, journal)
                        events.append(failed)
                        policy.send_packet(connection, failed)
                        receipt = policy.receive_packet(connection)
                        _validate_receipt(
                            receipt,
                            authority,
                            authorization,
                            checkpoint,
                            envelope,
                            request,
                            events,
                            "START_FAILED",
                            root_fd,
                        )
                        return 2

                    started = _event(
                        private,
                        authorization,
                        preflight,
                        sequence,
                        "child_started",
                        prior_clearance,
                        child_pid=child.pid,
                        child_start_ticks=_ticks(child.pid),
                        stdin_identity={"kind": "DEVNULL", "isatty": False},
                        stdout=before_out,
                        stderr=before_err,
                    )
                    sequence += 1
                    _append(root_fd, started, journal)
                    events.append(started)
                    policy.send_packet(connection, started)
                    code = child.wait()
                    stdout_sha256, stdout_byte_count = _digest_log(stdout_fd)
                    stderr_sha256, stderr_byte_count = _digest_log(stderr_fd)
                    exited = _event(
                        private,
                        authorization,
                        preflight,
                        sequence,
                        "child_exited",
                        prior_clearance,
                        child_pid=child.pid,
                        child_start_ticks=started["child_start_ticks"],
                        stdin_identity={"kind": "DEVNULL", "isatty": False},
                        return_code=code,
                        stdout=_identity(stdout_fd, stdout_names[index]),
                        stderr=_identity(stderr_fd, stderr_names[index]),
                        stdout_sha256=stdout_sha256,
                        stdout_byte_count=stdout_byte_count,
                        stderr_sha256=stderr_sha256,
                        stderr_byte_count=stderr_byte_count,
                    )
                    sequence += 1
                    _append(root_fd, exited, journal)
                    events.append(exited)
                    policy.send_packet(connection, exited)
                finally:
                    if stdout_fd is not None:
                        os.close(stdout_fd)
                    if stderr_fd is not None:
                        os.close(stderr_fd)

                if code != 0:
                    receipt = policy.receive_packet(connection)
                    _validate_receipt(
                        receipt,
                        authority,
                        authorization,
                        checkpoint,
                        envelope,
                        request,
                        events,
                        "FAILED",
                        root_fd,
                    )
                    return 2
                policy.validate_prelaunch(
                    envelope,
                    request,
                    require_outputs_absent=False,
                )
                checked = policy.validate_completed_command(
                    envelope,
                    request,
                    index,
                    previous,
                )
                previous = checked
                if index + 1 < len(preflights):
                    clearance = policy.receive_packet(connection)
                    _validate_clearance(
                        clearance,
                        authority,
                        authorization,
                        request,
                        index,
                        checked,
                    )
                    prior_clearance = clearance
            receipt = policy.receive_packet(connection)
            _validate_receipt(
                receipt,
                authority,
                authorization,
                checkpoint,
                envelope,
                request,
                events,
                "SUCCEEDED",
                root_fd,
            )
            return 0
        finally:
            connection.close()
    finally:
        os.close(root_fd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "policy",
        "checkpoint",
        "envelope",
        "request",
        "authority_socket",
        "observer_private_key",
        "evidence_root",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument(
        "--scope", required=True, choices=("acquisition", "phase_preparation", "one_touch")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (OSError, ValueError, policy.TerminalPolicyError) as exc:
        print(f"terminal observer: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
