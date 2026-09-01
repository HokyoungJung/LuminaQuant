#!/usr/bin/env python3
"""The sole process launcher for Alpha-Max terminal scopes."""

from __future__ import annotations

import signal
import argparse
import base64
import errno
import hashlib
import fcntl
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from contextlib import contextmanager
from collections.abc import Callable
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


def _raise_cleanup(
    primary: BaseException | None, cleanup_errors: list[BaseException], label: str
) -> None:
    if primary is not None:
        if cleanup_errors:
            cleanup = BaseExceptionGroup(f"{label} cleanup failed", cleanup_errors)
            raise BaseExceptionGroup(
                f"{label} failed and cleanup failed", [primary, cleanup]
            ) from primary
        raise primary
    if cleanup_errors:
        raise BaseExceptionGroup(f"{label} cleanup failed", cleanup_errors)


@contextmanager
def _owned(label: str, closers: list[Callable[[], None]]):
    primary: BaseException | None = None
    try:
        yield
    except BaseException as error:
        primary = error
    cleanup_errors: list[BaseException] = []
    for close in closers:
        try:
            close()
        except BaseException as error:
            cleanup_errors.append(error)
    _raise_cleanup(primary, cleanup_errors, label)


def _remove_created(root_fd: int, name: str, identity: tuple[int, int]) -> list[BaseException]:
    cleanup_errors: list[BaseException] = []
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (info.st_dev, info.st_ino) != identity:
            raise ObserverError(f"created publication identity changed during cleanup: {name}")
        os.unlink(name, dir_fd=root_fd)
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        os.fsync(root_fd)
    except BaseException as error:
        cleanup_errors.append(error)
    return cleanup_errors


def _close_fd(fd: int) -> None:
    os.close(fd)


def _close_all(label: str, closers: Sequence[Callable[[], None]]) -> None:
    cleanup_errors: list[BaseException] = []
    for close in closers:
        try:
            close()
        except BaseException as error:
            cleanup_errors.append(error)
    _raise_cleanup(sys.exception(), cleanup_errors, label)


def _open_root(path: Path, identity: Any) -> int:
    fd = policy.open_directory_fd(path, "evidence root")
    primary: BaseException | None = None
    try:
        info = os.fstat(fd)
        expected = (
            identity.st_dev,
            identity.st_ino,
            identity.st_uid,
            identity.st_gid,
            identity.mode,
        )
        actual = (info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
        if actual != expected:
            raise ObserverError("evidence root identity drift")
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ObserverError("terminal evidence root is under recovery") from exc
    except BaseException as error:
        primary = error
    if primary is not None:
        cleanup_errors: list[BaseException] = []
        try:
            os.close(fd)
        except BaseException as error:
            cleanup_errors.append(error)
        _raise_cleanup(primary, cleanup_errors, "evidence root open")
    return fd


def _relative(root: Path, path: str) -> str:
    if not isinstance(path, str) or Path(path).is_absolute() or Path(path).name != path:
        raise ObserverError("publication must be an evidence-root leaf")
    return path


def _new(root_fd: int, name: str, data: bytes, mode: int = 0o600) -> tuple[int, int]:
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode,
        dir_fd=root_fd,
    )
    created: tuple[int, int] | None = None
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        try:
            info = os.fstat(fd)
        except BaseException as error:
            primary = error
            try:
                info = os.fstat(fd)
            except BaseException as recovery:
                primary = BaseExceptionGroup(
                    "new publication identity recovery failed", [error, recovery]
                )
            else:
                created = (info.st_dev, info.st_ino)
        else:
            created = (info.st_dev, info.st_ino)
        if primary is None:
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
    except BaseException as error:
        primary = error
    try:
        os.close(fd)
    except BaseException as error:
        if primary is None:
            primary = error
        else:
            cleanup_errors.append(error)
    if primary is None:
        try:
            os.fsync(root_fd)
        except BaseException as error:
            primary = error
    if primary is not None:
        if created is not None:
            cleanup_errors.extend(_remove_created(root_fd, name, created))
        else:
            try:
                os.fsync(root_fd)
            except BaseException as error:
                cleanup_errors.append(error)
        _raise_cleanup(primary, cleanup_errors, "new publication")
    return created


def _safe_journal(info: os.stat_result, identity: tuple[int, int] | None = None) -> tuple[int, int]:
    actual = (info.st_dev, info.st_ino)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or (identity is not None and actual != identity)
    ):
        raise ObserverError("unsafe terminal journal")
    return actual


def _journal_path_identity(root_fd: int, journal: str, expected: tuple[int, int]) -> None:
    _safe_journal(
        os.stat(journal, dir_fd=root_fd, follow_symlinks=False),
        expected,
    )


def _append(
    root_fd: int,
    record: dict[str, Any],
    journal: str,
    journal_identity: tuple[int, int] | None,
) -> tuple[int, int]:
    data = policy.canonical_bytes(record)
    pending_identity = _new(root_fd, PENDING_NAME, data)
    fd: int | None = None
    closers: list[Callable[[], None]] = []
    with _owned("terminal journal append", closers):
        if journal_identity is None:
            fd = os.open(
                journal,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=root_fd,
            )
        else:
            fd = os.open(
                journal,
                os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=root_fd,
            )
        closers.append(lambda: _close_fd(fd))
        actual_identity = _safe_journal(os.fstat(fd), journal_identity)
        _journal_path_identity(root_fd, journal, actual_identity)
        _write_all(fd, data)
        os.fsync(fd)
        _journal_path_identity(root_fd, journal, actual_identity)
        retirement_errors = _remove_created(root_fd, PENDING_NAME, pending_identity)
        if retirement_errors:
            raise BaseExceptionGroup(
                f"pending publication retirement failed: {actual_identity}",
                retirement_errors,
            )
    return actual_identity


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


def _log_path_identity(root_fd: int, name: str, expected: dict[str, int | str | bool]) -> None:
    info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if (info.st_dev, info.st_ino) != (expected["st_dev"], expected["st_ino"]):
        raise ObserverError("child log identity drift")
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise ObserverError("unsafe child log")


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
    digest = hashlib.sha256()
    primary: BaseException | None = None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ObserverError("observer source is unsafe")
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
    except BaseException as error:
        primary = error
    cleanup_errors: list[BaseException] = []
    try:
        os.close(fd)
    except BaseException as error:
        cleanup_errors.append(error)
    _raise_cleanup(primary, cleanup_errors, "observer source digest")
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
    try:
        policy.validate_prelaunch(envelope, request)
    except BaseException:
        _close_all("observer root", [lambda: _close_fd(root_fd)])
        raise
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
    expected_identity: tuple[int, int] | None = None,
) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root_fd)
    chunks: list[bytes] = []
    primary: BaseException | None = None
    try:
        before = os.fstat(fd)
        identity = (before.st_dev, before.st_ino)
        path_before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_size,
            path_before.st_mtime_ns,
        ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            raise ObserverError(f"{label} changed while read")
        _safe_journal(path_before, identity)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise ObserverError(f"unsafe {label}")
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
        path_after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if (path_after.st_dev, path_after.st_ino, path_after.st_size, path_after.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_size,
            path_before.st_mtime_ns,
        ) != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        ):
            raise ObserverError(f"{label} changed while read")
        _safe_journal(path_after, identity)
    except BaseException as error:
        primary = error
    cleanup_errors: list[BaseException] = []
    try:
        os.close(fd)
    except BaseException as error:
        cleanup_errors.append(error)
    _raise_cleanup(primary, cleanup_errors, label)
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
        issued = policy.parse_utc_second(message["issued_utc"], "challenge issued_utc").timestamp()
    except (TypeError, ValueError, policy.TerminalPolicyError) as exc:
        raise ObserverError("invalid challenge freshness") from exc
    if len(nonce) != 32 or not 0 <= time.time() - issued <= 60:
        raise ObserverError("invalid challenge freshness")


def _authorization_bounds(message: Mapping[str, Any]) -> tuple[int, int]:
    try:
        return policy.authorization_epoch_window(message)
    except policy.TerminalPolicyError as exc:
        raise ObserverError("authorization window is not currently bounded") from exc


def _authorization_is_current(bounds: tuple[int, int]) -> None:
    not_before, expires = bounds
    now = time.time()
    if not not_before <= now < expires:
        raise ObserverError("authorization window is not currently bounded")


def _validate_authorization(
    message: Mapping[str, Any],
    authority: Any,
    challenge: Mapping[str, Any],
    claim: str,
    binding: Any,
    source: str,
) -> tuple[int, int]:
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
        policy.validate_sha256(message["authorization_id"], "authorization id")
    except policy.TerminalPolicyError as exc:
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
    bounds = _authorization_bounds(message)
    _authorization_is_current(bounds)
    return bounds


def _validate_receipt(
    message: Mapping[str, Any],
    authority: Any,
    authorization: Mapping[str, Any],
    checkpoint: Any,
    envelope: Any,
    request: Any,
    events: list[dict[str, Any]],
    terminal_state: Mapping[str, Any],
    root_fd: int,
    journal_identity: tuple[int, int],
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
        expected_identity=journal_identity,
    )
    target_results = policy.plain(
        policy._validate_scope_artifacts_at(
            envelope,
            request,
            events,
            None,
            root_fd,
            allow_incomplete=True,
        )
    )
    if (
        any(message[key] != value for key, value in expected.items())
        or message["journal_sha256"] != _sha(journal_bytes)
        or message["terminal_state"] != terminal_state
        or message["target_results"] != target_results
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


def _digest_bound_log(
    root_fd: int, name: str, fd: int, expected: dict[str, int | str | bool]
) -> tuple[str, int]:
    _log_path_identity(root_fd, name, expected)
    before = os.fstat(fd)
    path_before = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    expected_inode = (expected["st_dev"], expected["st_ino"])
    snapshot = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if (
        (before.st_dev, before.st_ino) != expected_inode
        or (path_before.st_dev, path_before.st_ino) != expected_inode
        or snapshot
        != (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_size,
            path_before.st_mtime_ns,
        )
    ):
        raise ObserverError("child log identity drift")
    digest, byte_count = _digest_log(fd)
    after = os.fstat(fd)
    path_after = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    if (
        snapshot != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or snapshot
        != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
        )
        or byte_count != after.st_size
        or _identity(fd, name) != expected
    ):
        raise ObserverError("child log changed while read")
    return digest, byte_count


def _proc_stat(pid: int) -> tuple[int, int, int]:
    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(") ", 1)[1]
        fields = tail.split()
        start_ticks, pgrp, session = int(fields[19]), int(fields[2]), int(fields[3])
    except OSError:
        raise
    except (IndexError, ValueError, UnicodeError) as exc:
        raise ObserverError(f"malformed child stat: {pid}") from exc
    if start_ticks <= 0:
        raise ObserverError(f"malformed child stat: {pid}")
    return start_ticks, pgrp, session


class _ChildLifecycle:
    """Own one direct child from Popen through its single reap."""

    def __init__(self, child: subprocess.Popen[Any]) -> None:
        self.child = child
        self.pid = child.pid
        self.start_ticks: int | None = None
        self.state = "OWNED_UNBOUND"
        self.wait_status: Any = None

    def bind(self) -> int:
        start_ticks, pgrp, session = _proc_stat(self.pid)
        if pgrp != self.pid or session != self.pid:
            raise ObserverError("child session identity mismatch")
        self.start_ticks = start_ticks
        self.state = "OWNED_BOUND"
        return start_ticks

    def _verify_bound(self) -> None:
        if self.state != "OWNED_BOUND":
            return
        start_ticks, pgrp, session = _proc_stat(self.pid)
        if (start_ticks, pgrp, session) != (self.start_ticks, self.pid, self.pid):
            raise ObserverError("child session identity drift")

    def scan_members(self) -> list[int]:
        try:
            entries = sorted(int(entry) for entry in os.listdir("/proc") if entry.isdigit())
        except OSError as exc:
            raise ObserverError("process scan failed") from exc

        members: list[int] = []
        for pid in entries:
            try:
                _start_ticks, pgrp, session = _proc_stat(pid)
            except OSError as exc:
                if exc.errno in (errno.ENOENT, errno.ESRCH):
                    continue
                raise ObserverError(f"process scan failed: {pid}") from exc
            if pid != self.pid and pgrp == self.pid and session == self.pid:
                members.append(pid)
        return members

    def observe_exit(self, *, nohang: bool = False) -> bool:
        if self.state in ("EXITED_UNREAPED", "REAPED"):
            return True
        flags = os.WEXITED | os.WNOWAIT | (os.WNOHANG if nohang else 0)
        status = os.waitid(os.P_PID, self.pid, flags)
        if status is None:
            return False
        self.wait_status = status
        self.state = "EXITED_UNREAPED"
        return True

    def _quiesce(self, deadline: float | None = None) -> bool:
        empty_scans = 0
        while empty_scans < 2:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            if self.scan_members():
                empty_scans = 0
            else:
                empty_scans += 1
            if empty_scans < 2:
                time.sleep(0.05)
        return True

    def await_exit_and_drain(self) -> None:
        self.observe_exit()
        for scan in range(2):
            if self.scan_members():
                raise ObserverError("child process group outlived leader")
            if scan == 0:
                time.sleep(0.05)

    def final_empty_scan(self) -> None:
        if self.scan_members():
            raise ObserverError("child process group outlived leader")

    def _validate_wait_status(self, code: int) -> None:
        if self.wait_status is None:
            return
        status = self.wait_status.si_status
        expected = status if self.wait_status.si_code == os.CLD_EXITED else -status
        if code != expected:
            raise ObserverError("child exit status mismatch")

    def reap(self) -> int:
        if self.state != "EXITED_UNREAPED":
            raise ObserverError("child was not observed exited")
        code = self.child.wait()
        self.state = "REAPED"
        self._validate_wait_status(code)
        return code

    def _signal_owned(
        self,
        signum: int,
        phase: str,
        errors: list[BaseException],
    ) -> None:
        if self.state == "REAPED":
            errors.append(ObserverError(f"{phase} signal refused after child reap"))
            return
        try:
            os.killpg(self.pid, signum)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            errors.append(
                ObserverError(f"{phase} cleanup failed for process group {self.pid}: {exc}")
            )

    def _wait_for_exit_and_drain(
        self, phase: str, deadline: float, errors: list[BaseException]
    ) -> bool:
        waitid_ok = True
        while time.monotonic() < deadline:
            if self.state != "EXITED_UNREAPED" and waitid_ok:
                try:
                    self.observe_exit(nohang=True)
                except BaseException as exc:
                    errors.append(ObserverError(f"{phase} cleanup waitid failed: {exc}"))
                    waitid_ok = False
            if self.state == "EXITED_UNREAPED":
                try:
                    drained = self._quiesce(deadline)
                except BaseException as exc:
                    errors.append(
                        ObserverError(
                            f"{phase} cleanup drain failed for process group {self.pid}: {exc}"
                        )
                    )
                    return False
                if not drained:
                    if phase == "KILL":
                        errors.append(
                            ObserverError(
                                f"KILL cleanup drain timed out for process group {self.pid}"
                            )
                        )
                    return False
                try:
                    self.reap()
                except BaseException as exc:
                    errors.append(ObserverError(f"{phase} cleanup reap failed: {exc}"))
                    return False
                return True
            time.sleep(0.05)
        if phase == "KILL":
            errors.append(
                ObserverError(f"KILL cleanup wait timed out for process group {self.pid}")
            )
        return False

    def cleanup(self, primary: BaseException) -> None:
        errors: list[BaseException] = []
        if self.state == "REAPED":
            _raise_cleanup(primary, errors, "child lifecycle")
            return

        if self.state == "OWNED_BOUND":
            try:
                self._verify_bound()
            except BaseException as exc:
                errors.append(exc)

        for signum, phase in (
            (signal.SIGTERM, "TERM"),
            (signal.SIGKILL, "KILL"),
        ):
            if self.state == "REAPED":
                break
            self._signal_owned(signum, phase, errors)
            if self._wait_for_exit_and_drain(phase, time.monotonic() + 5, errors):
                break
        if self.state not in ("EXITED_UNREAPED", "REAPED"):
            try:
                code = self.child.wait(timeout=5)
                self.state = "REAPED"
                self._validate_wait_status(code)
            except BaseException as exc:
                errors.append(ObserverError(f"final cleanup reap failed: {exc}"))
        _raise_cleanup(primary, errors, "child lifecycle")


def _own_child(child: subprocess.Popen[Any]) -> _ChildLifecycle:
    try:
        return _ChildLifecycle(child)
    except BaseException as primary:
        errors: list[BaseException] = []
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as exc:
            errors.append(ObserverError(f"child adoption group cleanup failed: {exc}"))
        try:
            child.wait()
        except BaseException as exc:
            errors.append(ObserverError(f"child adoption reap failed: {exc}"))
        _raise_cleanup(primary, errors, "child adoption")
        raise AssertionError("unreachable")


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
        issued = policy.parse_utc_second(
            message["issued_utc"], "command clearance issued_utc"
        ).timestamp()
    except policy.TerminalPolicyError as exc:
        raise ObserverError("invalid command clearance timestamp") from exc
    if not 0 <= time.time() - issued <= 60:
        raise ObserverError("stale command clearance")


def _validate_launch_authority(
    command_index: int,
    authorization_bounds: tuple[int, int],
    prior_clearance: Mapping[str, Any] | None,
    authority: Any,
    authorization: Mapping[str, Any],
    request: Any,
    previous: Any,
) -> None:
    if command_index == 0:
        _authorization_is_current(authorization_bounds)
        return
    if prior_clearance is None or previous is None:
        raise ObserverError("missing prior command clearance")
    _validate_clearance(
        prior_clearance,
        authority,
        authorization,
        request,
        command_index - 1,
        previous,
    )


def run(args: argparse.Namespace) -> int:
    checkpoint, envelope, request, private, preflights, root_fd, binding = _prepare(args)
    try:
        root = Path(request.evidence_root.path)
        claim_name = _relative(root, request.publication.claim)
        journal = _relative(root, request.publication.journal)
        stdout_names = tuple(_relative(root, item) for item in request.publication.stdout)
        stderr_names = tuple(_relative(root, item) for item in request.publication.stderr)
        receipt_name = _relative(root, request.publication.receipt)
        source = _source_digest()
        if source != envelope.observer_source_sha256:
            raise ObserverError("observer source binding mismatch")
    except BaseException:
        _close_all("observer root", [lambda: _close_fd(root_fd)])
        raise
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
            connection.connect(f"/proc/self/fd/{root_fd}/{Path(args.authority_socket).name}")
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
            authorization_bounds = _validate_authorization(
                authorization,
                authority,
                challenge,
                claim,
                binding,
                source,
            )
            journal_identity = _append(root_fd, authorization, journal, None)
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
                _validate_launch_authority(
                    index,
                    authorization_bounds,
                    prior_clearance,
                    authority,
                    authorization,
                    request,
                    previous,
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
                _append(root_fd, intent, journal, journal_identity)
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
                        _log_path_identity(root_fd, stdout_names[index], before_out)
                        _log_path_identity(root_fd, stderr_names[index], before_err)
                        environment = policy.plain(request.environment)
                        _validate_launch_authority(
                            index,
                            authorization_bounds,
                            prior_clearance,
                            authority,
                            authorization,
                            request,
                            previous,
                        )
                        child = subprocess.Popen(
                            preflight.argv,
                            cwd=request.repository_root.path,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=stdout_fd,
                            stderr=stderr_fd,
                            close_fds=True,
                            pass_fds=(),
                            start_new_session=True,
                        )
                        lifecycle = _own_child(child)
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
                        _append(root_fd, failed, journal, journal_identity)
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
                            {
                                "kind": "START_FAILED",
                                "command_index": index,
                                "errno": error_number or errno.EIO,
                            },
                            root_fd,
                            journal_identity,
                        )
                        return 2

                    try:
                        start_ticks = lifecycle.bind()
                        started = _event(
                            private,
                            authorization,
                            preflight,
                            sequence,
                            "child_started",
                            prior_clearance,
                            child_pid=child.pid,
                            child_start_ticks=start_ticks,
                            stdin_identity={"kind": "DEVNULL", "isatty": False},
                            stdout=before_out,
                            stderr=before_err,
                        )
                        sequence += 1
                        _append(root_fd, started, journal, journal_identity)
                        events.append(started)
                        policy.send_packet(connection, started)
                        lifecycle.await_exit_and_drain()
                        stdout_sha256, stdout_byte_count = _digest_bound_log(
                            root_fd, stdout_names[index], stdout_fd, before_out
                        )
                        stderr_sha256, stderr_byte_count = _digest_bound_log(
                            root_fd, stderr_names[index], stderr_fd, before_err
                        )
                        lifecycle.final_empty_scan()
                        code = lifecycle.reap()
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
                        _append(root_fd, exited, journal, journal_identity)
                        events.append(exited)
                        policy.send_packet(connection, exited)
                    except BaseException as exc:
                        lifecycle.cleanup(exc)
                finally:
                    _close_all(
                        "child log",
                        [
                            *(
                                [lambda stdout_fd=stdout_fd: _close_fd(stdout_fd)]
                                if stdout_fd is not None
                                else []
                            ),
                            *(
                                [lambda stderr_fd=stderr_fd: _close_fd(stderr_fd)]
                                if stderr_fd is not None
                                else []
                            ),
                        ],
                    )

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
                        {"kind": "FAILED", "failed_command_index": index},
                        root_fd,
                        journal_identity,
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
                {"kind": "SUCCEEDED"},
                root_fd,
                journal_identity,
            )
            return 0
        finally:
            _close_all("authority connection", [connection.close])
    finally:
        _close_all("observer root", [lambda: _close_fd(root_fd)])


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
