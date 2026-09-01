"""Runtime/security tests for the terminal observer's narrow local boundary."""

from __future__ import annotations

import ast

import base64
import json
import importlib.util
import os
import stat
import sys
import time
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
OBSERVER_PATH = ROOT / "scripts/research/run_alpha_max_terminal_observer.py"
AUTHORITY_PATH = ROOT / "scripts/research/run_alpha_max_terminal_authority.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def observer():
    return load(OBSERVER_PATH, "alpha_max_terminal_observer_test")


def identity(path: Path, policy):
    info = path.stat()
    return policy.DirectoryIdentity(
        str(path), info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)
    )


class _CompletedLifecycle:
    def __init__(self, child):
        self.child = child

    def bind(self):
        return 7

    def await_exit_and_drain(self):
        return None

    def final_empty_scan(self):
        return None

    def reap(self):
        return self.child.wait()

    def cleanup(self, primary):
        raise primary


def test_parser_requires_bound_authority_socket_and_full_evidence_root(observer):
    args = observer.parse_args(
        [
            "--scope",
            "acquisition",
            "--policy",
            "/policy",
            "--checkpoint",
            "/checkpoint",
            "--envelope",
            "/envelope",
            "--request",
            "/request",
            "--authority-socket",
            "/socket",
            "--observer-private-key",
            "/key",
            "--evidence-root",
            "/evidence",
        ]
    )
    assert args.authority_socket == Path("/socket")
    assert args.evidence_root == Path("/evidence")
    assert not hasattr(args, "evidence_root_fd")


def test_descriptor_relative_socket_connects_below_long_evidence_root(observer, tmp_path):
    root = tmp_path / ("e" * 80)
    root.mkdir()
    full_path = root / "terminal-authority.sock"
    assert len(os.fsencode(full_path)) > 107
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    server = observer.socket.socket(observer.socket.AF_UNIX, observer.socket.SOCK_SEQPACKET)
    client = observer.socket.socket(observer.socket.AF_UNIX, observer.socket.SOCK_SEQPACKET)
    accepted = None
    try:
        descriptor_path = f"/proc/self/fd/{root_fd}/{full_path.name}"
        assert len(os.fsencode(descriptor_path)) < 108
        server.bind(descriptor_path)
        server.listen(1)
        client.connect(descriptor_path)
        accepted, _address = server.accept()
    finally:
        client.close()
        if accepted is not None:
            accepted.close()
        server.close()
        try:
            os.unlink(full_path.name, dir_fd=root_fd)
        except FileNotFoundError:
            pass
        os.close(root_fd)


def test_child_process_contract_keeps_exact_argv_environment_and_session(observer):
    tree = ast.parse(OBSERVER_PATH.read_text(encoding="utf-8"))
    run = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    popen_calls = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "subprocess.Popen"
    ]
    assert len(popen_calls) == 1
    call = popen_calls[0]
    assert [ast.unparse(argument) for argument in call.args] == ["preflight.argv"]
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords == {
        "cwd": "request.repository_root.path",
        "env": "environment",
        "stdin": "subprocess.DEVNULL",
        "stdout": "stdout_fd",
        "stderr": "stderr_fd",
        "close_fds": "True",
        "pass_fds": "()",
        "start_new_session": "True",
    }
    assert any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and ast.unparse(node).startswith("environment = policy.plain(request.environment)")
        for node in ast.walk(run)
    )

    environment = observer.policy.plain(
        observer.policy.Environment(
            "/evidence",
            "C.UTF-8",
            "C.UTF-8",
            "/usr/bin:/bin",
            "0",
            "1",
            "1",
            "UTC",
        )
    )
    assert set(environment) == {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "TZ",
    }


def test_observer_shared_root_lock_rejects_active_recovery(observer, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root_identity = identity(root, observer.policy)
    exclusive_fd = observer.policy.open_directory_fd(root, "evidence root")
    try:
        observer.fcntl.flock(
            exclusive_fd,
            observer.fcntl.LOCK_EX | observer.fcntl.LOCK_NB,
        )
        with pytest.raises(
            observer.ObserverError,
            match="terminal evidence root is under recovery",
        ):
            observer._open_root(root, root_identity)
    finally:
        os.close(exclusive_fd)

    shared_fd = observer._open_root(root, root_identity)
    os.close(shared_fd)


def test_integrated_claim_is_durable_before_authority_or_process_actions(observer, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    root_identity = identity(root, observer.policy)
    fd = observer._open_root(root, root_identity)
    checkpoint = SimpleNamespace(sha256="1" * 64)
    request = SimpleNamespace(
        request_id="3" * 64, scope="acquisition", sha256="4" * 64, evidence_root=root_identity
    )
    try:
        digest = observer._claim(fd, "claim", checkpoint, request)
        claim = observer.policy.parse_canonical_object((root / "claim").read_bytes(), "claim")
        assert digest == observer._sha((root / "claim").read_bytes())
        assert claim["evidence_root"] == observer.policy.plain(root_identity)
        assert set(claim) == set(observer.policy.CLAIM_FIELDS)
        with pytest.raises(FileExistsError):
            observer._claim(fd, "claim", checkpoint, request)
    finally:
        os.close(fd)
    (root / "claim").unlink()
    authority_key = Ed25519PrivateKey.generate()
    authority_raw = authority_key.public_key().public_bytes(
        observer.policy.serialization.Encoding.Raw,
        observer.policy.serialization.PublicFormat.Raw,
    )
    observer_key = Ed25519PrivateKey.generate()
    publication = observer.policy.PublicationPaths(
        "claim", "journal", ("stdout",), ("stderr",), "receipt"
    )
    orchestration_request = SimpleNamespace(
        scope="acquisition",
        request_id="5" * 64,
        environment={"HOME": "/empty"},
        sha256="6" * 64,
        evidence_root=root_identity,
        repository_root=SimpleNamespace(path=str(root)),
        publication=publication,
    )
    envelope = SimpleNamespace(
        authority_key=SimpleNamespace(public_key_b64=base64.b64encode(authority_raw).decode()),
        observer_source_sha256="7" * 64,
        sha256="8" * 64,
    )
    preflight = SimpleNamespace(
        command_index=0,
        argv=("not-run",),
        argv_sha256="9" * 64,
        environment_sha256="a" * 64,
    )
    challenge = {
        "authority_key_id": observer.policy.public_key_id(authority_key.public_key()),
        "command_bundle_sha256": "b" * 64,
        "nonce_b64": base64.b64encode(b"n" * 32).decode(),
    }
    authorization = {"authorization_id": "c" * 64}
    received = iter((challenge, authorization, {"type": "terminal_receipt"}))
    events: list[str] = []
    original_fsync = os.fsync

    def fsync_claim(fd):
        inode = os.fstat(fd).st_ino
        if inode == root.stat().st_ino:
            events.append("root_fsync")
        else:
            events.append("claim_fsync")
        original_fsync(fd)

    long_socket = "/" + ("e" * 160) + "/terminal-authority.sock"

    class FakeConnection:
        def connect(self, path):
            assert path.startswith("/proc/self/fd/")
            assert path.endswith("/terminal-authority.sock")
            assert len(os.fsencode(path)) < 108
            require_claim("authority_connect")

        def close(self):
            events.append("authority_close")

    def require_claim(event: str) -> None:
        assert events[:2] == ["claim_fsync", "root_fsync"], (
            f"{event} occurred before claim durability"
        )
        events.append(event)

    def append(_root_fd, _record, _journal, _journal_identity):
        require_claim("journal_append")

    original_open = os.open

    def open_log(name, *args, **kwargs):
        if name in {"stdout", "stderr"}:
            require_claim(f"log_create:{name}")
        return original_open(name, *args, **kwargs)

    def launch(*_args, **_kwargs):
        require_claim("child_launch")
        raise OSError("popen unavailable")

    def send_packet(_connection, item):
        require_claim(
            "publication:proof"
            if item["type"] == "observer_proof"
            else f"publication:{item['event']}"
        )

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                checkpoint,
                envelope,
                orchestration_request,
                observer_key,
                (preflight,),
                os.open(root, os.O_RDONLY | os.O_DIRECTORY),
                SimpleNamespace(key_id="e" * 64),
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value=envelope.observer_source_sha256),
        mock.patch.object(observer.os, "fsync", side_effect=fsync_claim),
        mock.patch.object(observer, "_append", side_effect=append),
        mock.patch.object(observer, "_ticks", return_value=7),
        mock.patch.object(
            observer.socket,
            "socket",
            side_effect=lambda *_: (require_claim("authority_socket"), FakeConnection())[1],
        ),
        mock.patch.object(observer.policy, "receive_packet", side_effect=lambda _: next(received)),
        mock.patch.object(observer.policy, "send_packet", side_effect=send_packet),
        mock.patch.object(observer.policy, "validate_prelaunch"),
        mock.patch.object(observer.policy, "validate_command_semantics"),
        mock.patch.object(observer, "_validate_challenge"),
        mock.patch.object(observer, "_validate_authorization", return_value=(0, 2_000_000_000)),
        mock.patch.object(observer, "_validate_receipt"),
        mock.patch.object(observer.os, "open", side_effect=open_log),
        mock.patch.object(observer.subprocess, "Popen", side_effect=launch),
    ):
        assert observer.run(SimpleNamespace(authority_socket=long_socket)) == 2

    assert events == [
        "claim_fsync",
        "root_fsync",
        "authority_socket",
        "authority_connect",
        "publication:proof",
        "journal_append",
        "journal_append",
        "publication:launch_intent",
        "log_create:stdout",
        "log_create:stderr",
        "child_launch",
        "journal_append",
        "publication:start_failed",
        "authority_close",
    ]


def test_publication_create_is_exclusive_nofollow_mode_owner_and_pending_crash_survives(
    observer, tmp_path
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        observer._new(fd, "receipt", b"receipt\n")
        created = (root / "receipt").stat()
        assert stat.S_IMODE(created.st_mode) == 0o600 and created.st_uid == os.getuid()
        with pytest.raises(FileExistsError):
            observer._new(fd, "receipt", b"replacement\n")
        (root / "link").symlink_to(root / "receipt")
        with pytest.raises(FileExistsError):
            observer._new(fd, "link", b"x\n")
        pending = observer.policy.canonical_bytes({"sequence": 0})
        original_open = os.open

        def fail_journal_open(name, *args, **kwargs):
            if name == "journal":
                raise OSError("journal unavailable")
            return original_open(name, *args, **kwargs)

        with (
            mock.patch.object(observer.os, "open", side_effect=fail_journal_open),
            pytest.raises(OSError, match="journal unavailable"),
        ):
            observer._append(fd, {"sequence": 0}, "journal", None)
        assert (root / observer.PENDING_NAME).read_bytes() == pending
        assert not (root / "journal").exists()
    finally:
        os.close(fd)


def test_journal_identity_rejects_create_races_replacements_and_receipt_readback_drift(
    observer, tmp_path
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        attacker = b"attacker-bytes\n"
        (root / "journal").write_bytes(attacker)
        (root / "journal").chmod(0o600)
        with pytest.raises(FileExistsError):
            observer._append(fd, {"sequence": 0}, "journal", None)
        assert (root / "journal").read_bytes() == attacker
        (root / observer.PENDING_NAME).unlink()

        (root / "journal").unlink()
        journal_identity = observer._append(fd, {"sequence": 0}, "journal", None)
        original = (root / "journal").read_bytes()
        (root / "journal").unlink()
        (root / "journal").write_bytes(attacker)
        (root / "journal").chmod(0o600)
        with pytest.raises(observer.ObserverError, match="unsafe terminal journal"):
            observer._append(fd, {"sequence": 1}, "journal", journal_identity)
        assert (root / "journal").read_bytes() == attacker
        assert original != attacker

        with pytest.raises(observer.ObserverError, match="unsafe terminal journal"):
            observer._read_leaf(
                fd,
                "journal",
                "terminal journal",
                expected_identity=journal_identity,
            )
        assert (root / "journal").read_bytes() == attacker
    finally:
        os.close(fd)


def test_authorization_expiry_boundaries_are_checked_at_each_launch_point(observer):
    bounds = (100, 200)
    for now in (100, 199):
        with mock.patch.object(observer.time, "time", return_value=now):
            observer._authorization_is_current(bounds)
    for now in (99, 200, 201):
        with (
            mock.patch.object(observer.time, "time", return_value=now),
            pytest.raises(observer.ObserverError, match="authorization window"),
        ):
            observer._authorization_is_current(bounds)


def test_later_launch_authority_requires_exact_clearance_and_completed_evidence(observer):
    clearance = {"signed": "clearance"}
    authorization = {"authorization_id": "a" * 64}
    request = SimpleNamespace()
    evidence = SimpleNamespace(snapshot_sha256="b" * 64)

    with mock.patch.object(observer, "_validate_clearance") as validate_clearance:
        observer._validate_launch_authority(
            1, (0, 300), clearance, "authority", authorization, request, evidence
        )
        observer._validate_launch_authority(
            1, (0, 300), clearance, "authority", authorization, request, evidence
        )

    assert validate_clearance.call_args_list == [
        mock.call(clearance, "authority", authorization, request, 0, evidence),
        mock.call(clearance, "authority", authorization, request, 0, evidence),
    ]
    with pytest.raises(observer.ObserverError, match="missing prior command clearance"):
        observer._validate_launch_authority(
            1, (0, 300), None, "authority", authorization, request, evidence
        )
    with pytest.raises(observer.ObserverError, match="missing prior command clearance"):
        observer._validate_launch_authority(
            1, (0, 300), clearance, "authority", authorization, request, None
        )


def test_command_zero_launch_authority_remains_authorization_window_bound(observer):
    with mock.patch.object(observer, "_authorization_is_current") as authorization_is_current:
        observer._validate_launch_authority(0, (100, 400), None, None, {}, None, None)

    authorization_is_current.assert_called_once_with((100, 400))


def test_expiry_after_intent_emits_start_failed_without_launch(observer, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    authority_key = Ed25519PrivateKey.generate()
    authority_raw = authority_key.public_key().public_bytes(
        observer.policy.serialization.Encoding.Raw,
        observer.policy.serialization.PublicFormat.Raw,
    )
    authorization = {"authorization_id": "a" * 64, "claim_sha256": "b" * 64}
    request = SimpleNamespace(
        scope="acquisition",
        request_id="c" * 64,
        environment={"HOME": "/empty"},
        sha256="d" * 64,
        evidence_root=identity(root, observer.policy),
        repository_root=SimpleNamespace(path=str(root)),
        publication=observer.policy.PublicationPaths(
            "claim", "journal", ("stdout",), ("stderr",), "receipt"
        ),
    )
    envelope = SimpleNamespace(
        authority_key=SimpleNamespace(public_key_b64=base64.b64encode(authority_raw).decode()),
        observer_source_sha256="e" * 64,
        sha256="f" * 64,
    )
    preflight = SimpleNamespace(
        command_index=0,
        argv=("not-run",),
        argv_sha256="0" * 64,
        environment_sha256="1" * 64,
    )
    private = Ed25519PrivateKey.generate()
    sent: list[dict[str, object]] = []
    challenge = {
        "authority_key_id": observer.policy.public_key_id(authority_key.public_key()),
        "command_bundle_sha256": "4" * 64,
        "nonce_b64": base64.b64encode(b"n" * 32).decode(),
    }

    class FakeConnection:
        def connect(self, _path):
            return None

        def close(self):
            return None

    def append(_root_fd, record, _journal, journal_identity):
        sent.append(record)
        return (1, 2) if journal_identity is None else journal_identity

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                SimpleNamespace(sha256="2" * 64),
                envelope,
                request,
                private,
                (preflight,),
                os.open(root, os.O_RDONLY | os.O_DIRECTORY),
                SimpleNamespace(key_id="3" * 64),
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value=envelope.observer_source_sha256),
        mock.patch.object(observer, "_absent"),
        mock.patch.object(observer, "_claim", return_value=authorization["claim_sha256"]),
        mock.patch.object(observer.socket, "socket", return_value=FakeConnection()),
        mock.patch.object(
            observer.policy,
            "receive_packet",
            side_effect=iter((challenge, authorization, {})),
        ),
        mock.patch.object(observer.policy, "send_packet"),
        mock.patch.object(observer.policy, "validate_prelaunch"),
        mock.patch.object(observer.policy, "validate_command_semantics"),
        mock.patch.object(observer, "_validate_challenge"),
        mock.patch.object(observer, "_validate_authorization", return_value=(0, 1)),
        mock.patch.object(
            observer,
            "_authorization_is_current",
            side_effect=(
                None,
                observer.ObserverError("authorization window is not currently bounded"),
            ),
        ),
        mock.patch.object(observer, "_append", side_effect=append),
        mock.patch.object(observer, "_validate_receipt"),
        mock.patch.object(observer.subprocess, "Popen") as popen,
    ):
        assert observer.run(SimpleNamespace(authority_socket="/unused")) == 2

    popen.assert_not_called()
    events = [event for event in sent if "event" in event]
    assert [event["event"] for event in events] == ["launch_intent", "start_failed"]
    observer.policy.verify_message("process_event", events[-1], private.public_key())


@pytest.mark.parametrize("failure", ("intent_append", "log_open", "popen"))
def test_launch_failures_never_create_unauthenticated_or_relaunched_children(
    observer, tmp_path, failure
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    authority_key = Ed25519PrivateKey.generate()
    authority_raw = authority_key.public_key().public_bytes(
        observer.policy.serialization.Encoding.Raw,
        observer.policy.serialization.PublicFormat.Raw,
    )
    observer_key = Ed25519PrivateKey.generate()
    authorization = {"authorization_id": "a" * 64, "claim_sha256": "b" * 64}
    request = SimpleNamespace(
        scope="acquisition",
        request_id="c" * 64,
        environment={"HOME": "/empty"},
        sha256="d" * 64,
        evidence_root=identity(root, observer.policy),
        repository_root=SimpleNamespace(path=str(root)),
        publication=observer.policy.PublicationPaths(
            "claim", "journal", ("stdout",), ("stderr",), "receipt"
        ),
    )
    envelope = SimpleNamespace(
        authority_key=SimpleNamespace(public_key_b64=base64.b64encode(authority_raw).decode()),
        observer_source_sha256="e" * 64,
        sha256="3" * 64,
    )
    private = observer_key
    preflight = SimpleNamespace(
        command_index=0,
        argv=("not-run",),
        argv_sha256="f" * 64,
        environment_sha256="0" * 64,
    )
    challenge = {
        "authority_key_id": observer.policy.public_key_id(authority_key.public_key()),
        "command_bundle_sha256": "4" * 64,
        "nonce_b64": base64.b64encode(b"n" * 32).decode(),
    }
    received = iter((challenge, authorization, {"type": "terminal_receipt"}))
    sent: list[dict[str, object]] = []

    class FakeConnection:
        def connect(self, _path):
            return None

        def close(self):
            return None

    def append(_root_fd, record, _journal, journal_identity):
        if failure == "intent_append" and record.get("event") == "launch_intent":
            raise OSError("intent journal unavailable")
        return (1, 2) if journal_identity is None else journal_identity

    def validate_receipt(
        _receipt,
        _authority,
        _authorization,
        _checkpoint,
        _envelope,
        _request,
        events,
        terminal_state,
        _root_fd,
        _journal_identity,
    ):
        assert failure != "intent_append"
        assert terminal_state == {"kind": "START_FAILED", "command_index": 0, "errno": 5}
        assert [event["event"] for event in events] == ["launch_intent", "start_failed"]
        for event in events:
            observer.policy.verify_message("process_event", event, private.public_key())

    original_open = os.open

    def open_log(name, *args, **kwargs):
        if failure == "log_open" and name == "stdout":
            raise OSError("stdout unavailable")
        return original_open(name, *args, **kwargs)

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                SimpleNamespace(sha256="1" * 64),
                envelope,
                request,
                private,
                (preflight,),
                os.open(root, os.O_RDONLY | os.O_DIRECTORY),
                SimpleNamespace(key_id="2" * 64),
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value="e" * 64),
        mock.patch.object(observer, "_absent"),
        mock.patch.object(observer, "_claim", return_value=authorization["claim_sha256"]),
        mock.patch.object(observer, "_append", side_effect=append),
        mock.patch.object(observer, "_ticks", return_value=7),
        mock.patch.object(observer.socket, "socket", return_value=FakeConnection()),
        mock.patch.object(observer.policy, "receive_packet", side_effect=lambda _: next(received)),
        mock.patch.object(
            observer.policy, "send_packet", side_effect=lambda _, item: sent.append(item)
        ),
        mock.patch.object(observer.policy, "validate_prelaunch"),
        mock.patch.object(observer.policy, "validate_command_semantics"),
        mock.patch.object(observer, "_validate_challenge"),
        mock.patch.object(observer, "_validate_authorization", return_value=(0, 2_000_000_000)),
        mock.patch.object(observer, "_validate_receipt", side_effect=validate_receipt) as receipt,
        mock.patch.object(observer.os, "open", side_effect=open_log),
        mock.patch.object(
            observer.subprocess,
            "Popen",
            side_effect=OSError("popen unavailable") if failure == "popen" else None,
        ) as popen,
    ):
        if failure == "intent_append":
            with pytest.raises(OSError, match="intent journal unavailable"):
                observer.run(SimpleNamespace(authority_socket="/unused"))
        else:
            assert observer.run(SimpleNamespace(authority_socket="/unused")) == 2

    events = [item for item in sent if "event" in item]
    if failure == "intent_append":
        assert [event["event"] for event in events] == []
        receipt.assert_not_called()
        popen.assert_not_called()
        assert not (root / "stdout").exists()
        assert not (root / "stderr").exists()
    else:
        assert [event["event"] for event in events] == ["launch_intent", "start_failed"]
        receipt.assert_called_once()
        assert popen.call_count == (0 if failure == "log_open" else 1)
        if failure == "log_open":
            assert not (root / "stdout").exists()
            assert not (root / "stderr").exists()


@pytest.mark.parametrize(
    "path",
    (
        "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source",
        "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source/child",
        "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc",
        "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc/child",
    ),
)
def test_observer_rejects_quarantined_cli_paths_lexically_without_opening(observer, path):
    with pytest.raises(observer.policy.TerminalPolicyError):
        observer._absolute_cli(Path(path))


def test_observer_rejects_symlinked_evidence_root_ancestor(observer, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    info = root.stat()
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    expected = observer.policy.DirectoryIdentity(
        str(linked / "evidence"),
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        stat.S_IMODE(info.st_mode),
    )
    with pytest.raises(observer.policy.TerminalPolicyError):
        observer._open_root(linked / "evidence", expected)


def test_challenge_and_authorization_reject_freshness_replay_and_identity_binding(observer):
    authority_key = Ed25519PrivateKey.generate()
    observer_key = Ed25519PrivateKey.generate()
    authority_id = observer.policy.public_key_id(authority_key.public_key())
    binding = SimpleNamespace(key_id=observer.policy.public_key_id(observer_key.public_key()))
    checkpoint = SimpleNamespace(sha256="1" * 64)
    envelope = SimpleNamespace(sha256="2" * 64)
    request = SimpleNamespace(
        scope="acquisition", request_id="3" * 64, sha256="4" * 64, environment={"HOME": "/empty"}
    )
    commands = (("/bin/true",),)
    challenge_unsigned = {
        "schema": observer.SCHEMA,
        "type": "challenge",
        "authority_key_id": authority_id,
        "scope": request.scope,
        "request_id": request.request_id,
        "checkpoint_pin_sha256": checkpoint.sha256,
        "envelope_sha256": envelope.sha256,
        "request_sha256": request.sha256,
        "command_bundle_sha256": observer.policy.command_bundle_sha256(request, commands),
        "nonce_b64": base64.b64encode(b"n" * 32).decode(),
        "issued_utc": observer._utc(),
    }
    challenge = observer.policy.sign_message("challenge", challenge_unsigned, authority_key)
    observer._validate_challenge(
        challenge, authority_key.public_key(), checkpoint, envelope, request, commands
    )
    stale = dict(challenge, issued_utc="2020-01-01T00:00:00Z")
    stale = observer.policy.sign_message(
        "challenge",
        {key: value for key, value in stale.items() if key != "authority_signature_b64"},
        authority_key,
    )
    with pytest.raises(observer.ObserverError):
        observer._validate_challenge(
            stale, authority_key.public_key(), checkpoint, envelope, request, commands
        )
    replay = observer.policy.sign_message(
        "challenge",
        {
            key: value
            for key, value in dict(challenge, request_id="f" * 64).items()
            if key != "authority_signature_b64"
        },
        authority_key,
    )
    with pytest.raises(observer.ObserverError):
        observer._validate_challenge(
            replay, authority_key.public_key(), checkpoint, envelope, request, commands
        )
    authorization_unsigned = {
        "schema": observer.SCHEMA,
        "type": "authorization",
        "authority_key_id": authority_id,
        "authorization_id": "a" * 64,
        "scope": request.scope,
        "request_id": request.request_id,
        "checkpoint_pin_sha256": checkpoint.sha256,
        "envelope_sha256": envelope.sha256,
        "request_sha256": request.sha256,
        "command_bundle_sha256": challenge["command_bundle_sha256"],
        "claim_sha256": "c" * 64,
        "observer_key_id": binding.key_id,
        "observer_pid": os.getpid(),
        "observer_uid": os.getuid(),
        "observer_start_ticks": observer._ticks(os.getpid()),
        "observer_source_sha256": "d" * 64,
        "not_before_utc": "2026-07-22T00:00:00Z",
        "expires_utc": "2026-07-22T00:04:00Z",
    }
    authorization = observer.policy.sign_message(
        "authorization", authorization_unsigned, authority_key
    )
    with mock.patch.object(observer.time, "time", return_value=1784678520):
        observer._validate_authorization(
            authorization, authority_key.public_key(), challenge, "c" * 64, binding, "d" * 64
        )
    wrong_uid = observer.policy.sign_message(
        "authorization",
        {
            key: value
            for key, value in dict(authorization, observer_uid=os.getuid() + 1).items()
            if key != "authority_signature_b64"
        },
        authority_key,
    )
    with pytest.raises(observer.ObserverError):
        observer._validate_authorization(
            wrong_uid, authority_key.public_key(), challenge, "c" * 64, binding, "d" * 64
        )


@pytest.mark.parametrize(
    "clearance_mutation",
    (
        None,
        "expires_after_intent",
        "signature",
        "authority_domain",
        "index",
        "snapshot",
        "timestamp",
        "authorization_id",
        "scope",
        "request_id",
        "completed_command_index",
        "extra_field",
        "missing_field",
        "future_issued_utc",
    ),
)
def test_mocked_two_command_trace_revalidates_fresh_clearance_after_authorization_expiry(
    observer, tmp_path, clearance_mutation
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    root_identity = identity(root, observer.policy)
    authority_key = Ed25519PrivateKey.generate()
    authority_public = authority_key.public_key()
    authority_raw = authority_public.public_bytes(
        observer.policy.serialization.Encoding.Raw,
        observer.policy.serialization.PublicFormat.Raw,
    )
    authorization = {
        "authorization_id": "a" * 64,
        "claim_sha256": "c" * 64,
        "observer_key_id": "b" * 64,
        "observer_pid": 1,
        "observer_start_ticks": 2,
        "command_bundle_sha256": "d" * 64,
    }
    checked = SimpleNamespace(snapshot_sha256="e" * 64)
    request = SimpleNamespace(
        scope="acquisition",
        request_id="f" * 64,
        environment={"HOME": "/empty"},
        sha256="6" * 64,
        evidence_root=root_identity,
        repository_root=SimpleNamespace(path=str(root)),
        prerequisites=(),
        publication=observer.policy.PublicationPaths(
            "claim",
            "journal",
            ("child-0.stdout", "child-1.stdout"),
            ("child-0.stderr", "child-1.stderr"),
            "receipt",
        ),
    )
    clearance_unsigned = {
        "schema": observer.SCHEMA,
        "type": "command_clearance",
        "authority_key_id": observer.policy.public_key_id(authority_public),
        "authorization_id": authorization["authorization_id"],
        "scope": request.scope,
        "request_id": request.request_id,
        "completed_command_index": 0,
        "next_command_index": 1,
        "validated_artifact_snapshot_sha256": checked.snapshot_sha256,
        "issued_utc": "1970-01-01T00:06:40Z",
    }
    clearance_key = authority_key
    if clearance_mutation == "signature":
        clearance_key = Ed25519PrivateKey.generate()
    elif clearance_mutation == "authority_domain":
        clearance_unsigned["authority_key_id"] = "0" * 64
    elif clearance_mutation == "index":
        clearance_unsigned["next_command_index"] = 2
    elif clearance_mutation == "snapshot":
        clearance_unsigned["validated_artifact_snapshot_sha256"] = "0" * 64
    elif clearance_mutation == "timestamp":
        clearance_unsigned["issued_utc"] = "2020-01-01T00:00:00Z"
    elif clearance_mutation == "authorization_id":
        clearance_unsigned["authorization_id"] = "0" * 64
    elif clearance_mutation == "scope":
        clearance_unsigned["scope"] = "phase_preparation"
    elif clearance_mutation == "request_id":
        clearance_unsigned["request_id"] = "0" * 64
    elif clearance_mutation == "completed_command_index":
        clearance_unsigned["completed_command_index"] = 1
    elif clearance_mutation == "extra_field":
        clearance_unsigned["extra"] = "unexpected"
    elif clearance_mutation == "missing_field":
        del clearance_unsigned["scope"]
    elif clearance_mutation == "future_issued_utc":
        clearance_unsigned["issued_utc"] = "1970-01-01T00:06:41Z"
    clearance = observer.policy.sign_message("command_clearance", clearance_unsigned, clearance_key)
    preflights = (
        SimpleNamespace(
            command_index=0,
            argv=("mock-command-0",),
            argv_sha256="0" * 64,
            environment_sha256="1" * 64,
        ),
        SimpleNamespace(
            command_index=1,
            argv=("mock-command-1",),
            argv_sha256="2" * 64,
            environment_sha256="3" * 64,
        ),
    )
    envelope = SimpleNamespace(
        authority_key=SimpleNamespace(public_key_b64=base64.b64encode(authority_raw).decode()),
        observer_source_sha256="4" * 64,
        sha256="7" * 64,
    )
    challenge = {
        "authority_key_id": observer.policy.public_key_id(authority_public),
        "command_bundle_sha256": authorization["command_bundle_sha256"],
        "nonce_b64": base64.b64encode(b"n" * 32).decode(),
    }
    received = iter((challenge, authorization, clearance, {"type": "terminal_receipt"}))
    trace_events = []
    persisted_events = []
    sent_events = []
    order = []
    private = Ed25519PrivateKey.generate()

    class FakeConnection:
        def connect(self, _path):
            return None

        def close(self):
            return None

    class FakeChild:
        pid = 123

        def wait(self):
            return 0

    def receive_packet(_connection):
        return next(received)

    def validate_receipt(
        _receipt,
        _authority,
        _authorization,
        _checkpoint,
        _envelope,
        _request,
        events,
        _terminal_state,
        _root_fd,
        _journal_identity,
    ):
        trace_events.extend(events)

    def validate_challenge(*_args):
        order.append("challenge")

    def validate_authorization(*_args):
        order.append("authorization")
        return 0, 300

    def launch(*_args, **_kwargs):
        order.append("popen")
        return FakeChild()

    def append(*args):
        persisted_events.append(args[1])
        return args[3] or (1, 2)

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                SimpleNamespace(sha256="5" * 64),
                envelope,
                request,
                private,
                preflights,
                os.open(root, os.O_RDONLY | os.O_DIRECTORY),
                SimpleNamespace(key_id=authorization["observer_key_id"]),
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value="4" * 64),
        mock.patch.object(observer, "_absent"),
        mock.patch.object(observer, "_claim", return_value=authorization["claim_sha256"]),
        mock.patch.object(observer, "_append", side_effect=append),
        mock.patch.object(observer, "_ticks", return_value=7),
        mock.patch.object(observer.socket, "socket", return_value=FakeConnection()),
        mock.patch.object(observer.policy, "receive_packet", side_effect=receive_packet),
        mock.patch.object(
            observer.policy,
            "send_packet",
            side_effect=lambda _connection, message: sent_events.append(message),
        ),
        mock.patch.object(observer.policy, "validate_prelaunch"),
        mock.patch.object(observer.policy, "validate_command_semantics"),
        mock.patch.object(observer.policy, "validate_completed_command", return_value=checked),
        mock.patch.object(observer, "_validate_challenge", side_effect=validate_challenge),
        mock.patch.object(observer, "_validate_authorization", side_effect=validate_authorization),
        mock.patch.object(observer, "_validate_receipt", side_effect=validate_receipt),
        mock.patch.object(observer.subprocess, "Popen", side_effect=launch) as popen,
        mock.patch.object(observer, "_ChildLifecycle", _CompletedLifecycle),
        mock.patch.object(
            observer.time,
            "time",
            side_effect=(
                (100, 100, 400, 400, 461)
                if clearance_mutation == "expires_after_intent"
                else (100, 100, 400, 401, 402)
            ),
        ),
    ):
        if clearance_mutation is None:
            assert observer.run(SimpleNamespace(authority_socket="/unused")) == 0
        elif clearance_mutation == "expires_after_intent":
            assert observer.run(SimpleNamespace(authority_socket="/unused")) == 2
        else:
            with pytest.raises((observer.ObserverError, observer.policy.TerminalPolicyError)):
                observer.run(SimpleNamespace(authority_socket="/unused"))

    if clearance_mutation not in (None, "expires_after_intent"):
        assert popen.call_count == 1
        assert not any(
            event.get("command_index") == 1 and event.get("event") == "launch_intent"
            for event in sent_events
        )
        return
    if clearance_mutation == "expires_after_intent":
        assert popen.call_count == 1
        command_one = [event for event in trace_events if event["command_index"] == 1]
        assert [event["event"] for event in command_one] == [
            "launch_intent",
            "start_failed",
        ]
        assert [
            event["event"] for event in persisted_events if event.get("command_index") == 1
        ] == ["launch_intent", "start_failed"]
        assert [event["event"] for event in sent_events if event.get("command_index") == 1] == [
            "launch_intent",
            "start_failed",
        ]
        assert all(event["prior_clearance"] is clearance for event in command_one)
        observer.policy.verify_message("process_event", command_one[-1], private.public_key())
        return
    assert order[:3] == ["challenge", "authorization", "popen"]

    command_one = [event for event in trace_events if event["command_index"] == 1]
    assert [event["event"] for event in command_one] == [
        "launch_intent",
        "child_started",
        "child_exited",
    ]
    assert all(event["prior_clearance"] is clearance for event in command_one)


@pytest.mark.parametrize("failure", ("challenge", "authorization"))
def test_run_rejects_validation_failures_before_popen(observer, tmp_path, failure):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    authority_key = Ed25519PrivateKey.generate()
    authority_public = authority_key.public_key()
    authority_raw = authority_public.public_bytes(
        observer.policy.serialization.Encoding.Raw,
        observer.policy.serialization.PublicFormat.Raw,
    )
    request = SimpleNamespace(
        scope="acquisition",
        request_id="f" * 64,
        environment={"HOME": "/empty"},
        sha256="6" * 64,
        evidence_root=identity(root, observer.policy),
        repository_root=SimpleNamespace(path=str(root)),
        publication=observer.policy.PublicationPaths("claim", "journal", (), (), "receipt"),
    )
    envelope = SimpleNamespace(
        authority_key=SimpleNamespace(public_key_b64=base64.b64encode(authority_raw).decode()),
        observer_source_sha256="4" * 64,
        sha256="3" * 64,
    )
    challenge = {
        "authority_key_id": observer.policy.public_key_id(authority_public),
        "command_bundle_sha256": "d" * 64,
        "nonce_b64": base64.b64encode(b"n" * 32).decode(),
    }
    packets = (challenge,) if failure == "challenge" else (challenge, {})
    received = iter(packets)

    class FakeConnection:
        def connect(self, _path):
            return None

        def close(self):
            return None

    def reject(*_args):
        raise observer.ObserverError(f"invalid {failure}")

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                SimpleNamespace(sha256="5" * 64),
                envelope,
                request,
                Ed25519PrivateKey.generate(),
                (),
                os.open(root, os.O_RDONLY | os.O_DIRECTORY),
                SimpleNamespace(key_id="b" * 64),
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value="4" * 64),
        mock.patch.object(observer, "_absent"),
        mock.patch.object(observer, "_claim", return_value="c" * 64),
        mock.patch.object(observer, "_ticks", return_value=7),
        mock.patch.object(observer.socket, "socket", return_value=FakeConnection()),
        mock.patch.object(observer.policy, "receive_packet", side_effect=lambda _: next(received)),
        mock.patch.object(observer.policy, "send_packet"),
        mock.patch.object(
            observer,
            "_validate_challenge",
            side_effect=reject if failure == "challenge" else None,
        ),
        mock.patch.object(
            observer,
            "_validate_authorization",
            side_effect=reject if failure == "authorization" else None,
        ),
        mock.patch.object(observer.subprocess, "Popen") as popen,
        pytest.raises(observer.ObserverError, match=f"invalid {failure}"),
    ):
        observer.run(SimpleNamespace(authority_socket="/unused"))

    popen.assert_not_called()


def _receipt_context(observer, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    observer._new(fd, "journal", b"journal\n")
    authority_key = Ed25519PrivateKey.generate()
    authorization = {
        "authorization_id": "a" * 64,
        "claim_sha256": "b" * 64,
        "observer_key_id": "c" * 64,
        "observer_pid": 1,
        "observer_start_ticks": 2,
        "command_bundle_sha256": "d" * 64,
    }
    checkpoint = SimpleNamespace(sha256="e" * 64)
    envelope = SimpleNamespace(sha256="f" * 64)
    publication = observer.policy.PublicationPaths("claim", "journal", (), (), "receipt")
    request = SimpleNamespace(
        scope="acquisition",
        request_id="0" * 64,
        sha256="1" * 64,
        prerequisites=(),
        publication=publication,
    )
    return fd, root, authority_key, authorization, checkpoint, envelope, request


def _signed_receipt(
    observer,
    authority_key,
    authorization,
    checkpoint,
    envelope,
    request,
    events_sha256,
    journal_sha256,
    *,
    target_results=None,
    terminal_state=None,
):
    message = {
        "schema": observer.SCHEMA,
        "type": "terminal_receipt",
        "authority_key_id": observer.policy.public_key_id(authority_key.public_key()),
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
        "events_sha256": events_sha256,
        "journal_sha256": journal_sha256,
        "prerequisites": [observer.policy.plain(item) for item in request.prerequisites],
        "target_results": [] if target_results is None else target_results,
        "terminal_state": (
            {"kind": "START_FAILED", "command_index": 0, "errno": 5}
            if terminal_state is None
            else terminal_state
        ),
        "publication": observer.policy.plain(request.publication),
        "created_utc": observer._utc(),
    }
    return observer.policy.sign_message("terminal_receipt", message, authority_key)


def _validate_signed_receipt(
    observer,
    signed,
    authority_key,
    authorization,
    checkpoint,
    envelope,
    request,
    fd,
    expected_terminal_state,
    expected_results,
):
    journal_info = os.stat(request.publication.journal, dir_fd=fd, follow_symlinks=False)
    journal_identity = (journal_info.st_dev, journal_info.st_ino)
    with mock.patch.object(
        observer.policy, "_validate_scope_artifacts_at", return_value=expected_results
    ) as validate_scope:
        observer._validate_receipt(
            signed,
            authority_key.public_key(),
            authorization,
            checkpoint,
            envelope,
            request,
            [],
            expected_terminal_state,
            fd,
            journal_identity,
        )
    validate_scope.assert_called_once_with(envelope, request, [], None, fd, allow_incomplete=True)


def test_receipt_accepts_matching_authority_signed_canonical_readback(observer, tmp_path):
    fd, root, authority_key, authorization, checkpoint, envelope, request = _receipt_context(
        observer, tmp_path
    )
    try:
        events = []
        signed = _signed_receipt(
            observer,
            authority_key,
            authorization,
            checkpoint,
            envelope,
            request,
            observer._sha(observer.policy.canonical_bytes(events)),
            observer._sha(b"journal\n"),
        )
        canonical = observer.policy.canonical_bytes(signed)
        observer._new(fd, "receipt", canonical)
        assert (root / "receipt").read_bytes() == canonical
        _validate_signed_receipt(
            observer,
            signed,
            authority_key,
            authorization,
            checkpoint,
            envelope,
            request,
            fd,
            {"kind": "START_FAILED", "command_index": 0, "errno": 5},
            [],
        )
    finally:
        os.close(fd)


def test_receipt_readback_mismatch_is_enforced(observer, tmp_path):
    fd, _root, authority_key, authorization, checkpoint, envelope, request = _receipt_context(
        observer, tmp_path
    )
    try:
        signed = _signed_receipt(
            observer,
            authority_key,
            authorization,
            checkpoint,
            envelope,
            request,
            observer._sha(observer.policy.canonical_bytes([])),
            observer._sha(b"journal\n"),
        )
        observer._new(fd, "receipt", b"wrong\n")
        with pytest.raises(observer.ObserverError, match="readback mismatch"):
            _validate_signed_receipt(
                observer,
                signed,
                authority_key,
                authorization,
                checkpoint,
                envelope,
                request,
                fd,
                {"kind": "START_FAILED", "command_index": 0, "errno": 5},
                [],
            )
    finally:
        os.close(fd)


def test_receipt_rejects_independently_signed_events_digest_tamper(observer, tmp_path):
    fd, root, authority_key, authorization, checkpoint, envelope, request = _receipt_context(
        observer, tmp_path
    )
    try:
        signed = _signed_receipt(
            observer,
            authority_key,
            authorization,
            checkpoint,
            envelope,
            request,
            "9" * 64,
            observer._sha(b"journal\n"),
        )
        observer._new(fd, "receipt", observer.policy.canonical_bytes(signed))
        assert (root / "receipt").read_bytes() == observer.policy.canonical_bytes(signed)
        with pytest.raises(observer.ObserverError, match="receipt binding mismatch"):
            _validate_signed_receipt(
                observer,
                signed,
                authority_key,
                authorization,
                checkpoint,
                envelope,
                request,
                fd,
                {"kind": "START_FAILED", "command_index": 0, "errno": 5},
                [],
            )
    finally:
        os.close(fd)


def test_receipt_rejects_independently_signed_journal_digest_tamper(observer, tmp_path):
    fd, root, authority_key, authorization, checkpoint, envelope, request = _receipt_context(
        observer, tmp_path
    )
    try:
        signed = _signed_receipt(
            observer,
            authority_key,
            authorization,
            checkpoint,
            envelope,
            request,
            observer._sha(observer.policy.canonical_bytes([])),
            "8" * 64,
        )
        observer._new(fd, "receipt", observer.policy.canonical_bytes(signed))
        assert (root / "receipt").read_bytes() == observer.policy.canonical_bytes(signed)
        with pytest.raises(observer.ObserverError, match="receipt binding mismatch"):
            _validate_signed_receipt(
                observer,
                signed,
                authority_key,
                authorization,
                checkpoint,
                envelope,
                request,
                fd,
                {"kind": "START_FAILED", "command_index": 0, "errno": 5},
                [],
            )
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    ("mutation", "resign", "match"),
    (
        ("signature", False, "invalid terminal_receipt signature"),
        ("terminal_state", True, "receipt binding mismatch"),
        ("target_results_order", True, "receipt binding mismatch"),
        ("target_results_content", True, "receipt binding mismatch"),
    ),
)
def test_receipt_rejects_corruption_and_resigned_terminal_contract_mismatches(
    observer, tmp_path, mutation, resign, match
):
    fd, _root, authority_key, authorization, checkpoint, envelope, request = _receipt_context(
        observer, tmp_path
    )
    expected_terminal_state = {"kind": "SUCCEEDED"}
    expected_results = [
        {
            "command_index": index,
            "argv_sha256": "a" * 64,
            "environment_sha256": "b" * 64,
            "return_code": 0,
            "stdout": {
                "kind": "stdout",
                "path": str(tmp_path / f"stdout-{index}"),
                "sha256": "c" * 64,
                "byte_count": 0,
                "st_dev": 1,
                "st_ino": index + 1,
                "mode": 0o600,
                "nlink": 1,
            },
            "stderr": {
                "kind": "stderr",
                "path": str(tmp_path / f"stderr-{index}"),
                "sha256": "d" * 64,
                "byte_count": 0,
                "st_dev": 1,
                "st_ino": index + 3,
                "mode": 0o600,
                "nlink": 1,
            },
            "validated_artifacts": [],
            "sealed_artifacts": [],
            "completed_utc": "2026-07-24T00:00:00Z",
        }
        for index in range(2)
    ]
    try:
        signed = _signed_receipt(
            observer,
            authority_key,
            authorization,
            checkpoint,
            envelope,
            request,
            observer._sha(observer.policy.canonical_bytes([])),
            observer._sha(b"journal\n"),
            target_results=expected_results,
            terminal_state=expected_terminal_state,
        )
        if mutation == "signature":
            signed = dict(signed, authorization_id="0" * 64)
        else:
            unsigned = {
                key: value for key, value in signed.items() if key != "authority_signature_b64"
            }
            if mutation == "terminal_state":
                unsigned["terminal_state"] = {
                    "kind": "START_FAILED",
                    "command_index": 0,
                    "errno": 5,
                }
            elif mutation == "target_results_order":
                unsigned["target_results"] = list(reversed(expected_results))
            else:
                unsigned["target_results"] = [
                    {**expected_results[0], "return_code": 1},
                    expected_results[1],
                ]
            signed = observer.policy.sign_message("terminal_receipt", unsigned, authority_key)
        observer._new(fd, "receipt", observer.policy.canonical_bytes(signed))
        with pytest.raises(
            (observer.ObserverError, observer.policy.TerminalPolicyError), match=match
        ):
            _validate_signed_receipt(
                observer,
                signed,
                authority_key,
                authorization,
                checkpoint,
                envelope,
                request,
                fd,
                expected_terminal_state,
                expected_results,
            )
        if resign:
            observer.policy.verify_message("terminal_receipt", signed, authority_key.public_key())
    finally:
        os.close(fd)


def test_source_binding_failure_after_prepare_closes_root_before_authority(observer, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    request = SimpleNamespace(
        evidence_root=SimpleNamespace(path=str(root)),
        publication=observer.policy.PublicationPaths("claim", "journal", (), (), "receipt"),
    )
    closed: list[int] = []
    original_close = os.close

    def close(fd):
        closed.append(fd)
        original_close(fd)

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                None,
                SimpleNamespace(observer_source_sha256="wrong"),
                request,
                None,
                (),
                root_fd,
                None,
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value="actual"),
        mock.patch.object(observer.os, "close", side_effect=close),
        mock.patch.object(observer.socket, "socket") as socket,
        pytest.raises(observer.ObserverError, match="source binding mismatch"),
    ):
        observer.run(SimpleNamespace(authority_socket="/unused"))

    assert closed == [root_fd]
    socket.assert_not_called()


def test_close_all_attempts_every_close_and_preserves_all_failures(observer):
    attempted: list[str] = []

    def fail(name):
        def close():
            attempted.append(name)
            raise OSError(f"{name} close failed")

        return close

    with pytest.raises(BaseExceptionGroup) as raised:
        observer._close_all(
            "observer cleanup", [fail("stdout"), fail("stderr"), fail("connection"), fail("root")]
        )

    assert attempted == ["stdout", "stderr", "connection", "root"]
    assert [str(error) for error in raised.value.exceptions] == [
        "stdout close failed",
        "stderr close failed",
        "connection close failed",
        "root close failed",
    ]


def test_new_fsync_failure_removes_owned_publication(observer, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    original_fsync = os.fsync

    def fail_claim_fsync(fd):
        if os.fstat(fd).st_ino != root.stat().st_ino:
            raise OSError("claim fsync failed")
        original_fsync(fd)

    try:
        with (
            mock.patch.object(observer.os, "fsync", side_effect=fail_claim_fsync),
            pytest.raises(OSError, match="claim fsync failed"),
        ):
            observer._new(root_fd, "claim", b"claim")
        assert not (root / "claim").exists()
    finally:
        os.close(root_fd)


@pytest.mark.parametrize(
    ("failure", "expected_primary"),
    (("claim_file", "claim file fsync failed"), ("evidence_root", "evidence root fsync failed")),
)
def test_claim_durability_failures_stop_run_before_authority_or_publication(
    observer, tmp_path, failure, expected_primary
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    root_inode = os.fstat(root_fd).st_ino
    authority_key = Ed25519PrivateKey.generate()
    authority_raw = authority_key.public_key().public_bytes(
        observer.policy.serialization.Encoding.Raw,
        observer.policy.serialization.PublicFormat.Raw,
    )
    request = SimpleNamespace(
        scope="acquisition",
        request_id="a" * 64,
        environment={"HOME": "/empty"},
        sha256="b" * 64,
        evidence_root=identity(root, observer.policy),
        repository_root=SimpleNamespace(path=str(root)),
        prerequisites=(),
        publication=observer.policy.PublicationPaths("claim", "journal", (), (), "receipt"),
    )
    envelope = SimpleNamespace(
        authority_key=SimpleNamespace(public_key_b64=base64.b64encode(authority_raw).decode()),
        observer_source_sha256="c" * 64,
        sha256="d" * 64,
    )
    fsync_calls: list[str] = []

    def fsync(fd):
        kind = "root" if os.fstat(fd).st_ino == root_inode else "claim"
        fsync_calls.append(kind)
        if (failure == "claim_file" and kind == "claim") or (
            failure == "evidence_root" and kind == "root" and fsync_calls.count("root") == 1
        ):
            raise OSError(expected_primary)
        if kind == "root":
            raise OSError("claim cleanup root fsync failed")

    def messages(error):
        if isinstance(error, BaseExceptionGroup):
            return [message for item in error.exceptions for message in messages(item)]
        return [str(error)]

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                SimpleNamespace(sha256="e" * 64),
                envelope,
                request,
                Ed25519PrivateKey.generate(),
                (),
                root_fd,
                SimpleNamespace(key_id="f" * 64),
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value=envelope.observer_source_sha256),
        mock.patch.object(observer.os, "fsync", side_effect=fsync),
        mock.patch.object(observer.socket, "socket") as authority_socket,
        mock.patch.object(observer.subprocess, "Popen") as child,
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        observer.run(SimpleNamespace(authority_socket="/unused"))

    assert ("claim" if failure == "claim_file" else "root") in fsync_calls
    assert expected_primary in messages(raised.value)
    assert "claim cleanup root fsync failed" in messages(raised.value)
    authority_socket.assert_not_called()
    child.assert_not_called()
    assert not any(
        (root / name).exists()
        for name in ("claim", "journal", "receipt", ".terminal-observer.pending")
    )


def test_final_receipt_validation_keeps_live_descriptors_and_aggregates_close_failures(
    observer, tmp_path
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    root_identity = identity(root, observer.policy)
    authority_key = Ed25519PrivateKey.generate()
    authority_raw = authority_key.public_key().public_bytes(
        observer.policy.serialization.Encoding.Raw,
        observer.policy.serialization.PublicFormat.Raw,
    )
    private = Ed25519PrivateKey.generate()
    authorization = {
        "authorization_id": "a" * 64,
        "claim_sha256": "b" * 64,
        "observer_key_id": "c" * 64,
        "observer_pid": os.getpid(),
        "observer_start_ticks": 7,
        "command_bundle_sha256": "d" * 64,
    }
    request = SimpleNamespace(
        scope="acquisition",
        request_id="e" * 64,
        environment={"HOME": "/empty"},
        sha256="f" * 64,
        evidence_root=root_identity,
        repository_root=SimpleNamespace(path=str(root)),
        prerequisites=(),
        publication=observer.policy.PublicationPaths(
            "claim", "journal", ("stdout",), ("stderr",), "receipt"
        ),
    )
    envelope = SimpleNamespace(
        authority_key=SimpleNamespace(public_key_b64=base64.b64encode(authority_raw).decode()),
        observer_source_sha256="0" * 64,
        sha256="1" * 64,
    )
    challenge = {
        "type": "challenge",
        "authority_key_id": observer.policy.public_key_id(authority_key.public_key()),
        "command_bundle_sha256": authorization["command_bundle_sha256"],
        "nonce_b64": base64.b64encode(b"n" * 32).decode(),
    }
    preflight = SimpleNamespace(
        command_index=0,
        argv=("not-run",),
        argv_sha256="2" * 64,
        environment_sha256="3" * 64,
    )
    close_counts: dict[tuple[int, int], int] = {}
    close_paths: list[str] = []
    publication_calls: list[str] = []
    original_close = os.close
    root_key = (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino)

    class Connection:
        close_calls = 0

        def connect(self, _path):
            return None

        def close(self):
            self.close_calls += 1
            raise OSError("authority close failed")

    class Child:
        pid = os.getpid()

        def wait(self):
            return 0

    connection = Connection()
    authorization = {**authorization, "type": "authorization"}
    packets = iter((challenge, authorization, {"type": "terminal_receipt"}))
    validation_calls: list[str] = []
    received_types: list[str] = []

    def receive_packet(_connection):
        packet = next(packets)
        received_types.append(packet["type"])
        if packet["type"] == "terminal_receipt":
            publication_calls.append("receipt")
            observer._new(root_fd, "receipt", observer.policy.canonical_bytes(packet))
        return packet

    def close(fd):
        info = os.fstat(fd)
        key = (info.st_dev, info.st_ino)
        close_counts[key] = close_counts.get(key, 0) + 1
        close_paths.append(os.readlink(f"/proc/self/fd/{fd}"))
        original_close(fd)
        if key == root_key:
            raise OSError("root close failed")

    def reject_final_receipt(*args):
        assert args[-2] == root_fd
        assert publication_calls == ["receipt"]
        assert all(
            (root / name).exists() for name in ("claim", "journal", "stdout", "stderr", "receipt")
        )
        raise observer.ObserverError("final receipt validation failed")

    def validate_challenge(*_args):
        validation_calls.append("challenge")

    def validate_authorization(*_args):
        validation_calls.append("authorization")
        return 0, 2_000_000_000

    def messages(error):
        if isinstance(error, BaseExceptionGroup):
            return [message for item in error.exceptions for message in messages(item)]
        return [str(error)]

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                SimpleNamespace(sha256="4" * 64),
                envelope,
                request,
                private,
                (preflight,),
                root_fd,
                SimpleNamespace(key_id=authorization["observer_key_id"]),
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value=envelope.observer_source_sha256),
        mock.patch.object(observer, "_ticks", return_value=7),
        mock.patch.object(observer.socket, "socket", return_value=connection),
        mock.patch.object(observer.policy, "receive_packet", side_effect=receive_packet),
        mock.patch.object(observer.policy, "send_packet"),
        mock.patch.object(observer.policy, "validate_prelaunch"),
        mock.patch.object(observer.policy, "validate_command_semantics"),
        mock.patch.object(observer.policy, "validate_completed_command"),
        mock.patch.object(observer, "_validate_challenge", side_effect=validate_challenge),
        mock.patch.object(observer, "_validate_authorization", side_effect=validate_authorization),
        mock.patch.object(
            observer, "_validate_receipt", side_effect=reject_final_receipt
        ) as validate_receipt,
        mock.patch.object(observer.subprocess, "Popen", return_value=Child()) as popen,
        mock.patch.object(observer, "_ChildLifecycle", _CompletedLifecycle),
        mock.patch.object(observer.os, "close", side_effect=close),
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        observer.run(SimpleNamespace(authority_socket="/unused"))

    assert connection.close_calls == 1
    assert publication_calls == ["receipt"]
    validate_receipt.assert_called_once()
    popen.assert_called_once()
    assert validation_calls == ["challenge", "authorization"]
    assert received_types == ["challenge", "authorization", "terminal_receipt"]
    assert close_paths.count(str(root / "claim")) == 1
    assert close_paths.count(str(root / "journal")) == 4
    assert all(close_paths.count(str(root / name)) == 1 for name in ("stdout", "stderr", "receipt"))
    assert close_counts[root_key] == 1
    assert {
        "final receipt validation failed",
        "authority close failed",
        "root close failed",
    } <= set(messages(raised.value))


def test_post_popen_bind_fault_terminates_and_reaps_original_process_group(observer):
    child = observer.subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        start_new_session=True,
    )
    lifecycle = observer._ChildLifecycle(child)
    real_proc_stat = observer._proc_stat
    initial = True

    def fail_initial_stat(pid):
        nonlocal initial
        if initial and pid == child.pid:
            initial = False
            raise observer.ObserverError("stat fault")
        return real_proc_stat(pid)

    try:
        with (
            mock.patch.object(observer, "_proc_stat", side_effect=fail_initial_stat),
            pytest.raises(observer.ObserverError, match="stat fault") as raised,
        ):
            lifecycle.bind()
        with pytest.raises(observer.ObserverError, match="stat fault"):
            lifecycle.cleanup(raised.value)

        assert lifecycle.state == "REAPED"
        assert child.poll() == -observer.signal.SIGTERM
    finally:
        if child.poll() is None:
            os.killpg(child.pid, observer.signal.SIGKILL)
            child.wait()


def test_lifecycle_drains_real_descendant_writer_and_term_ignoring_process(observer, tmp_path):
    written = tmp_path / "descendant-output"
    descendant = (
        "import pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(0.1); pathlib.Path(sys.argv[1]).write_text('done')"
    )
    leader = (
        "import subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}, {str(written)!r}]); "
        "raise SystemExit(child.wait())"
    )
    child = observer.subprocess.Popen((sys.executable, "-c", leader), start_new_session=True)
    lifecycle = observer._ChildLifecycle(child)
    try:
        assert lifecycle.bind() > 0
        lifecycle.await_exit_and_drain()
        lifecycle.final_empty_scan()
        assert lifecycle.reap() == 0
        assert lifecycle.state == "REAPED"
        assert written.read_text() == "done"
    finally:
        if child.poll() is None:
            os.killpg(child.pid, observer.signal.SIGKILL)
            child.wait()


def test_lifecycle_reaps_normal_nonzero_exit_once(observer):
    child = observer.subprocess.Popen(
        (sys.executable, "-c", "raise SystemExit(3)"), start_new_session=True
    )
    lifecycle = observer._ChildLifecycle(child)
    try:
        assert lifecycle.bind() > 0
        lifecycle.await_exit_and_drain()
        lifecycle.final_empty_scan()
        assert lifecycle.reap() == 3
        assert lifecycle.state == "REAPED"
        with pytest.raises(observer.ObserverError, match="later fault"):
            lifecycle.cleanup(observer.ObserverError("later fault"))
    finally:
        if child.poll() is None:
            os.killpg(child.pid, observer.signal.SIGKILL)
            child.wait()


def test_contradictory_initial_identity_preserves_primary_and_signals_original_group(observer):
    child = observer.subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        start_new_session=True,
    )
    lifecycle = observer._ChildLifecycle(child)
    real_killpg = observer.os.killpg
    try:
        with (
            mock.patch.object(
                observer,
                "_proc_stat",
                return_value=(1, child.pid + 1, child.pid + 1),
            ),
            pytest.raises(
                observer.ObserverError,
                match="child session identity mismatch",
            ) as raised,
        ):
            lifecycle.bind()
        with (
            mock.patch.object(observer.os, "killpg", wraps=real_killpg) as killpg,
            pytest.raises(
                observer.ObserverError,
                match="child session identity mismatch",
            ),
        ):
            lifecycle.cleanup(raised.value)
        assert killpg.call_args_list[0].args == (child.pid, observer.signal.SIGTERM)
        assert lifecycle.state == "REAPED"
        assert child.returncode == -observer.signal.SIGTERM
    finally:
        if child.poll() is None:
            os.killpg(child.pid, observer.signal.SIGKILL)
            child.wait()


def test_waitid_failure_preserves_errors_without_direct_signal(observer):
    class Child:
        pid = 987_654

        def __init__(self):
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            return -observer.signal.SIGKILL

    child = Child()
    lifecycle = observer._ChildLifecycle(child)
    primary = observer.ObserverError("body fault")
    ticks = iter((0.0, 1.0, 6.0, 10.0, 11.0, 16.0))

    with (
        mock.patch.object(
            observer.os,
            "waitid",
            side_effect=OSError("waitid fault"),
        ),
        mock.patch.object(observer.os, "killpg") as killpg,
        mock.patch.object(observer.os, "kill") as kill_direct,
        mock.patch.object(observer.time, "monotonic", side_effect=lambda: next(ticks)),
        mock.patch.object(observer.time, "sleep"),
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        lifecycle.cleanup(primary)

    assert raised.value.exceptions[0] is primary
    messages = [str(error) for error in raised.value.exceptions[1].exceptions]
    assert any("TERM cleanup waitid failed" in message for message in messages)
    assert any("KILL cleanup waitid failed" in message for message in messages)
    assert any("KILL cleanup wait timed out" in message for message in messages)
    assert [call.args[1] for call in killpg.call_args_list] == [
        observer.signal.SIGTERM,
        observer.signal.SIGKILL,
    ]
    kill_direct.assert_not_called()
    assert child.wait_calls == 1
    assert lifecycle.state == "REAPED"


def test_reap_state_is_truthful_before_status_mismatch_and_cleanup_never_signals(
    observer,
):
    child = mock.Mock()
    child.pid = 101
    child.wait.return_value = 0
    lifecycle = observer._ChildLifecycle(child)
    lifecycle.state = "EXITED_UNREAPED"
    lifecycle.wait_status = SimpleNamespace(
        si_status=1,
        si_code=observer.os.CLD_EXITED,
    )

    with pytest.raises(observer.ObserverError, match="child exit status mismatch"):
        lifecycle.reap()
    assert lifecycle.state == "REAPED"
    child.wait.assert_called_once()

    with (
        mock.patch.object(observer.os, "killpg") as killpg,
        mock.patch.object(observer.os, "kill") as kill_direct,
        pytest.raises(observer.ObserverError, match="later fault"),
    ):
        lifecycle.cleanup(observer.ObserverError("later fault"))
    killpg.assert_not_called()
    kill_direct.assert_not_called()


def test_child_adoption_failure_kills_group_and_reaps(observer):
    child = mock.Mock()
    child.pid = 202
    with (
        mock.patch.object(
            observer,
            "_ChildLifecycle",
            side_effect=observer.ObserverError("adoption fault"),
        ),
        mock.patch.object(observer.os, "killpg") as killpg,
        pytest.raises(observer.ObserverError, match="adoption fault"),
    ):
        observer._own_child(child)

    killpg.assert_called_once_with(child.pid, observer.signal.SIGKILL)
    child.wait.assert_called_once()
    child.kill.assert_not_called()


def test_cleanup_kills_real_term_ignoring_process_group(observer, tmp_path):
    ready = tmp_path / "ready"
    descendant = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    )
    leader = (
        "import pathlib, signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    child = observer.subprocess.Popen(
        (sys.executable, "-c", leader, str(ready)),
        start_new_session=True,
    )
    lifecycle = observer._ChildLifecycle(child)
    try:
        for _attempt in range(200):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists()
        assert lifecycle.bind() > 0
        primary = observer.ObserverError("cleanup trigger")
        with pytest.raises(observer.ObserverError, match="cleanup trigger"):
            lifecycle.cleanup(primary)

        assert lifecycle.state == "REAPED"
        assert child.returncode == -observer.signal.SIGKILL
    finally:
        if child.poll() is None:
            os.killpg(child.pid, observer.signal.SIGKILL)
            child.wait()


def test_read_leaf_rejects_path_replacement_without_expected_identity(observer, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    original = root / "receipt"
    original.write_bytes(b"original")
    original.chmod(0o600)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    replacement = root / "replacement"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o600)
    real_open = observer.os.open

    def replace_after_open(name, flags, *args, **kwargs):
        opened = real_open(name, flags, *args, **kwargs)
        if name == "receipt":
            os.replace(replacement, original)
        return opened

    try:
        with (
            mock.patch.object(observer.os, "open", side_effect=replace_after_open),
            pytest.raises(observer.ObserverError, match="changed while read"),
        ):
            observer._read_leaf(fd, "receipt", "terminal receipt")
    finally:
        os.close(fd)


@pytest.mark.parametrize("name", ("stdout", "stderr"))
@pytest.mark.parametrize(
    "mutation", ("same_bytes", "different_bytes", "append", "truncate", "metadata")
)
def test_retained_log_descriptor_rejects_path_and_metadata_drift(
    observer, tmp_path, name, mutation
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    path = root / name
    original = b"retained-log\n"
    path.write_bytes(original)
    path.chmod(0o600)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    log_fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    expected = observer._identity(log_fd, name)
    real_read = observer.os.read
    mutated = False

    def mutate_after_read(fd, size):
        nonlocal mutated
        result = real_read(fd, size)
        if fd == log_fd and not mutated:
            mutated = True
            if mutation in {"same_bytes", "different_bytes"}:
                replacement = root / f"{name}.replacement"
                replacement.write_bytes(original if mutation == "same_bytes" else b"attacker-log\n")
                replacement.chmod(0o600)
                os.replace(replacement, path)
            elif mutation == "append":
                with path.open("ab") as changed:
                    changed.write(b"append")
            elif mutation == "truncate":
                with path.open("r+b") as changed:
                    changed.truncate(1)
            else:
                info = path.stat()
                os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1))
        return result

    try:
        with (
            mock.patch.object(observer.os, "read", side_effect=mutate_after_read),
            pytest.raises(observer.ObserverError, match="child log changed while read"),
        ):
            observer._digest_bound_log(root_fd, name, log_fd, expected)
        assert mutated
    finally:
        os.close(log_fd)
        os.close(root_fd)


@pytest.mark.parametrize("replacement_bytes", (b"original", b"replacement"))
def test_read_leaf_rejects_same_or_different_byte_receipt_path_replacement(
    observer, tmp_path, replacement_bytes
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    receipt = root / "receipt"
    receipt.write_bytes(b"original")
    receipt.chmod(0o600)
    replacement = root / "replacement"
    replacement.write_bytes(replacement_bytes)
    replacement.chmod(0o600)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    real_open = observer.os.open

    def replace_after_open(name, flags, *args, **kwargs):
        opened = real_open(name, flags, *args, **kwargs)
        if name == "receipt":
            os.replace(replacement, receipt)
        return opened

    try:
        with (
            mock.patch.object(observer.os, "open", side_effect=replace_after_open),
            pytest.raises(observer.ObserverError, match="changed while read"),
        ):
            observer._read_leaf(root_fd, "receipt", "terminal receipt")
    finally:
        os.close(root_fd)


def test_cleanup_pins_unreaped_leader_then_escalates_and_reaps_original_group(observer, tmp_path):
    ready = tmp_path / "descendant-ready"
    descendant = (
        "import pathlib, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
    )
    leader = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {descendant!r}, sys.argv[1]]); "
        "raise SystemExit(0)"
    )
    child = observer.subprocess.Popen(
        (sys.executable, "-c", leader, str(ready)), start_new_session=True
    )
    lifecycle = observer._ChildLifecycle(child)
    real_killpg = observer.os.killpg
    signals: list[int] = []

    def kill_group(pid, signum):
        signals.append(signum)
        return real_killpg(pid, signum)

    try:
        for _attempt in range(200):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.read_text() == "ready"
        assert lifecycle.bind() > 0
        for _attempt in range(200):
            if lifecycle.observe_exit(nohang=True):
                break
            time.sleep(0.01)
        assert lifecycle.state == "EXITED_UNREAPED"
        assert lifecycle.scan_members()
        with (
            pytest.raises(
                observer.ObserverError, match="child process group outlived leader"
            ) as outlived,
            mock.patch.object(observer.os, "killpg", side_effect=kill_group),
        ):
            lifecycle.await_exit_and_drain()
        with (
            mock.patch.object(observer.os, "killpg", side_effect=kill_group),
            pytest.raises(observer.ObserverError, match="child process group outlived leader"),
        ):
            lifecycle.cleanup(outlived.value)
        assert signals == [observer.signal.SIGTERM, observer.signal.SIGKILL]
        assert lifecycle.state == "REAPED"
        assert child.returncode == 0
        with pytest.raises(observer.ObserverError, match="later fault"):
            lifecycle.cleanup(observer.ObserverError("later fault"))
        assert signals == [observer.signal.SIGTERM, observer.signal.SIGKILL]
    finally:
        if child.poll() is None:
            os.killpg(child.pid, observer.signal.SIGKILL)
            child.wait()


def test_scan_members_treats_enoent_as_local_and_rejects_hard_errors(observer):
    child = SimpleNamespace(pid=50)
    lifecycle = observer._ChildLifecycle(child)

    def local_stat(pid):
        if pid == 51:
            raise OSError(observer.errno.ENOENT, "gone")
        return (1, 50, 50)

    with (
        mock.patch.object(observer.os, "listdir", return_value=["52", "51", "50"]),
        mock.patch.object(observer, "_proc_stat", side_effect=local_stat),
    ):
        assert lifecycle.scan_members() == [52]

    with (
        mock.patch.object(observer.os, "listdir", side_effect=PermissionError("denied")),
        pytest.raises(observer.ObserverError, match="process scan failed"),
    ):
        lifecycle.scan_members()


def test_quiesce_requires_two_separated_empty_scans_and_resets_for_member(observer):
    lifecycle = observer._ChildLifecycle(SimpleNamespace(pid=50))
    scans = mock.Mock(side_effect=[[], [51], [], []])

    with (
        mock.patch.object(lifecycle, "scan_members", scans),
        mock.patch.object(observer.time, "sleep") as sleep,
    ):
        assert lifecycle._quiesce() is True

    assert scans.call_count == 4
    assert sleep.call_args_list == [
        mock.call(0.05),
        mock.call(0.05),
        mock.call(0.05),
    ]


def test_cleanup_escalates_when_term_drain_times_out_and_reports_kill_timeout(observer):
    child = mock.Mock()
    child.pid = 50
    lifecycle = observer._ChildLifecycle(child)
    lifecycle.state = "EXITED_UNREAPED"

    with (
        mock.patch.object(lifecycle, "_quiesce", side_effect=[False, False]),
        mock.patch.object(observer.os, "killpg") as killpg,
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        lifecycle.cleanup(observer.ObserverError("primary"))

    assert [call.args for call in killpg.call_args_list] == [
        (50, observer.signal.SIGTERM),
        (50, observer.signal.SIGKILL),
    ]
    assert [str(error) for error in raised.value.exceptions[1].exceptions] == [
        "KILL cleanup drain timed out for process group 50",
    ]
    child.wait.assert_not_called()


def test_cleanup_aggregate_order_is_primary_then_lifecycle_phases(observer):
    child = mock.Mock()
    child.pid = 50
    lifecycle = observer._ChildLifecycle(child)
    lifecycle.state = "OWNED_BOUND"
    lifecycle.start_ticks = 1

    with (
        mock.patch.object(observer, "_proc_stat", return_value=(2, 50, 50)),
        mock.patch.object(observer.os, "killpg", side_effect=[OSError("term"), OSError("kill")]),
        mock.patch.object(lifecycle, "_wait_for_exit_and_drain", return_value=False),
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        lifecycle.cleanup(observer.ObserverError("primary"))

    assert [str(error) for error in raised.value.exceptions[1].exceptions] == [
        "child session identity drift",
        "TERM cleanup failed for process group 50: term",
        "KILL cleanup failed for process group 50: kill",
    ]


def test_bound_identity_drift_keeps_cleanup_on_original_group(observer):
    child = mock.Mock()
    child.pid = 50
    lifecycle = observer._ChildLifecycle(child)
    lifecycle.state = "OWNED_BOUND"
    lifecycle.start_ticks = 1

    with (
        mock.patch.object(observer, "_proc_stat", return_value=(2, 99, 99)),
        mock.patch.object(observer.os, "killpg") as killpg,
        mock.patch.object(lifecycle, "_wait_for_exit_and_drain", return_value=True),
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        lifecycle.cleanup(observer.ObserverError("primary"))

    assert [call.args for call in killpg.call_args_list] == [(50, observer.signal.SIGTERM)]
    assert [str(error) for error in raised.value.exceptions[1].exceptions] == [
        "child session identity drift"
    ]


def test_offline_signed_authority_observer_exchange_preserves_local_child_contract(
    observer, tmp_path
):
    authority = load(AUTHORITY_PATH, "alpha_max_terminal_authority_live_fixture")
    authority_tests = load(
        ROOT / "tests/test_run_alpha_max_terminal_authority.py",
        "alpha_max_terminal_authority_live_fixture_support",
    )
    external = load(
        ROOT / "tests/test_alpha_max_terminal_external_envelope.py",
        "alpha_max_terminal_external_live_fixture",
    )
    (
        policy,
        checkpoint,
        envelope,
        request,
        _authority_private,
        observer_private,
        paths,
    ) = authority_tests._runtime_recovery_context(authority, tmp_path)

    # The external fixture supplies the complete ten-symbol acquisition outputs.  Move
    # them under alternate leaves of the request's already-bound output parent; their
    # owner declarations name the requested leaves, so the atomic rename below restores
    # the exact owner-path and parent-identity contract before validation.
    _phase_envelope, _phase_request, fixture = external._phase_records_fixture(
        tmp_path / "external-acquisition"
    )
    policy_path, checkpoint_path, envelope_path, request_path, authority_private_path = paths
    bound = policy_path.parent / "bound"
    for role, fixture_name in (
        ("acquirer", "acquirer.py"),
        ("contract_manifest", "contract.json"),
        ("availability_evidence", "availability.json"),
    ):
        materialized = bound / role
        materialized.write_bytes(fixture[fixture_name].read_bytes())
        materialized.chmod(0o600)
    policy_value = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_value["pins"].update(
        acquirer_sha256=observer._sha((bound / "acquirer").read_bytes()),
        contract_sha256=observer._sha((bound / "contract_manifest").read_bytes()),
        availability_sha256=observer._sha((bound / "availability_evidence").read_bytes()),
    )
    external.write_canonical(policy_path, policy_value)
    policy = authority.policy.load_policy(policy_path)
    envelope_value = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope_value["policy_sha256"] = policy.source_sha256
    file_by_role = {item["role"]: item for item in envelope_value["files"]}
    file_by_role["policy_json"]["file"] = authority_tests._runtime_file(policy_path)
    for role in ("acquirer", "contract_manifest", "availability_evidence"):
        file_by_role[role]["file"] = authority_tests._runtime_file(bound / role)
    external.write_canonical(envelope_path, envelope_value)
    checkpoint_value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_value.update(
        **policy.pins,
        authority_manifest_sha256=observer._sha(envelope_path.read_bytes()),
    )
    external.write_canonical(checkpoint_path, checkpoint_value)
    checkpoint = authority.policy.load_checkpoint(checkpoint_path, policy)
    envelope = authority.policy.load_envelope(envelope_path, policy, checkpoint)
    request_value = json.loads(request_path.read_text(encoding="utf-8"))
    checkpoint_prerequisite = authority_tests._runtime_file(checkpoint_path)
    request_value["prerequisites"][0] = {
        "kind": "checkpoint_pin",
        **{
            key: value
            for key, value in checkpoint_prerequisite.items()
            if key not in {"st_uid", "st_gid"}
        },
    }
    request_value.update(
        checkpoint_pin_sha256=checkpoint.sha256,
        **{
            role: file_by_role[role]["file"]
            for role in ("acquirer", "contract_manifest", "availability_evidence")
        },
    )
    external.write_canonical(request_path, request_value)
    request = authority.policy.load_request(
        request_path,
        scope="acquisition",
        policy=policy,
        checkpoint=checkpoint,
        envelope=envelope,
    )
    output_parent = Path(request.records.source_root.parent.path)
    source_target = Path(request.records.source_root.path)
    report_target = Path(request.records.report_root.path)
    source_stage = output_parent / "source.staged"
    report_stage = output_parent / "report.staged"
    fixture["source"].parent.chmod(0o700)
    fixture["source"].chmod(0o755)
    fixture["report"].chmod(0o755)
    fixture["source"].rename(source_stage)
    fixture["report"].rename(report_stage)
    owner_parent = output_parent.stat()
    for owner_path in (
        source_stage / ".alpha_max_owner.json",
        report_stage / ".alpha_max_owner.json",
    ):
        owner_path.parent.chmod(0o755)
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        owner.update(
            output_path=str(source_target),
            report_path=str(report_target),
            output_parent_identity=[owner_parent.st_dev, owner_parent.st_ino],
            report_parent_identity=[owner_parent.st_dev, owner_parent.st_ino],
            output_identity=[source_stage.stat().st_dev, source_stage.stat().st_ino],
            report_identity=[report_stage.stat().st_dev, report_stage.stat().st_ino],
        )
        external.write_canonical(owner_path, owner)
        owner_path.parent.chmod(0o555)

    observer_private_path = tmp_path / "observer.private"
    observer_private_path.write_bytes(
        observer_private.private_bytes(
            observer.policy.serialization.Encoding.Raw,
            observer.policy.serialization.PrivateFormat.Raw,
            observer.policy.serialization.NoEncryption(),
        )
    )
    observer_private_path.chmod(0o400)
    authority_public_path = tmp_path / "authority.public"
    authority_public_path.write_bytes(
        _authority_private.public_key().public_bytes(
            observer.policy.serialization.Encoding.Raw,
            observer.policy.serialization.PublicFormat.Raw,
        )
    )
    authority_public_path.chmod(0o444)

    authority_args = authority.parser().parse_args(
        [
            "serve",
            "--policy",
            str(policy_path),
            "--checkpoint",
            str(checkpoint_path),
            "--envelope",
            str(envelope_path),
            "--request",
            str(request_path),
            "--private-key",
            str(authority_private_path),
            "--socket",
            request.authority_socket,
            "--evidence-root",
            request.evidence_root.path,
            "--scope",
            "acquisition",
        ]
    )
    observer_args = observer.parse_args(
        [
            "--policy",
            str(policy_path),
            "--checkpoint",
            str(checkpoint_path),
            "--envelope",
            str(envelope_path),
            "--request",
            str(request_path),
            "--authority-socket",
            request.authority_socket,
            "--observer-private-key",
            str(observer_private_path),
            "--evidence-root",
            request.evidence_root.path,
            "--scope",
            "acquisition",
        ]
    )
    server_result: list[int] = []
    server_error: list[BaseException] = []

    def serve() -> None:
        try:
            server_result.append(authority.serve(authority_args))
        except BaseException as exc:
            server_error.append(exc)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    socket_path = Path(request.authority_socket)
    deadline = time.monotonic() + 5
    while not socket_path.exists() and not server_error and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists(), server_error

    expected_commands = observer.policy.derive_scope_commands(envelope, request)
    popen_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    real_popen = observer.subprocess.Popen

    def popen(argv: object, **kwargs: object):
        index = len(popen_calls)
        actual_argv = tuple(argv)
        popen_calls.append((actual_argv, kwargs))
        assert actual_argv == expected_commands[index]
        assert kwargs == {
            "cwd": request.repository_root.path,
            "env": observer.policy.plain(request.environment),
            "stdin": observer.subprocess.DEVNULL,
            "stdout": kwargs["stdout"],
            "stderr": kwargs["stderr"],
            "close_fds": True,
            "pass_fds": (),
            "start_new_session": True,
        }
        assert set(kwargs["env"]) == {
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "PYTHONHASHSEED",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
            "TZ",
        }
        if index == 0:
            assert not source_target.exists() and not report_target.exists()
            source_stage.rename(source_target)
            report_stage.rename(report_target)
        return real_popen(
            (
                sys.executable,
                "-c",
                "import os; os.write(1, b'live-fixture-child\\n')",
            ),
            **kwargs,
        )

    try:
        with mock.patch.object(observer.subprocess, "Popen", side_effect=popen):
            assert observer.run(observer_args) == 0
    finally:
        server.join(timeout=5)
    assert not server.is_alive()
    assert not server_error
    assert server_result == [0]
    assert not socket_path.exists()

    root = Path(request.evidence_root.path)
    claim = root / request.publication.claim
    journal = root / request.publication.journal
    receipt_path = root / request.publication.receipt
    assert claim.exists()
    assert not (root / observer.PENDING_NAME).exists()
    assert claim.stat().st_mtime_ns <= journal.stat().st_mtime_ns
    journal_records = [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    assert [record.get("event", record["type"]) for record in journal_records] == [
        "authorization",
        "launch_intent",
        "child_started",
        "child_exited",
        "launch_intent",
        "child_started",
        "child_exited",
    ]
    assert len(popen_calls) == 2
    assert all(
        (root / name).read_text(encoding="utf-8") == "live-fixture-child\n"
        for name in request.publication.stdout
    )
    assert all((root / name).read_bytes() == b"" for name in request.publication.stderr)

    verified = observer.policy.verify_signed_receipt(receipt_path, authority_public_path)
    assert verified.authorization["claim_sha256"] == observer._sha(claim.read_bytes())
    results = observer.policy.validate_scope_artifacts(envelope, request, verified.events)
    assert len(results) == 2
    assert verified.message["target_results"] == observer.policy.plain(results)
    assert all(result.validated_artifacts for result in results)
