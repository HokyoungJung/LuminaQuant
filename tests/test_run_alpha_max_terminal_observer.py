"""Runtime/security tests for the terminal observer's narrow local boundary."""

from __future__ import annotations

import base64
import importlib.util
import os
import stat
import sys
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


def test_root_identity_claim_is_first_durable_mutation_and_contains_full_identity(
    observer, tmp_path
):
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
    source = OBSERVER_PATH.read_text(encoding="utf-8")
    assert source.index("claim = _claim(") < source.index("connection = socket.socket")


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
            observer._append(fd, {"sequence": 0}, "journal")
        assert (root / observer.PENDING_NAME).read_bytes() == pending
        assert not (root / "journal").exists()
    finally:
        os.close(fd)


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

    def append(_root_fd, record, _journal):
        if failure == "intent_append" and record.get("event") == "launch_intent":
            raise OSError("intent journal unavailable")

    def validate_receipt(
        _receipt,
        _authority,
        _authorization,
        _checkpoint,
        _envelope,
        _request,
        events,
        terminal_kind,
        _root_fd,
    ):
        assert failure != "intent_append"
        assert terminal_kind == "START_FAILED"
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
        mock.patch.object(observer, "_validate_authorization"),
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
    (None, "signature", "authority_domain", "index", "snapshot", "timestamp"),
)
def test_mocked_two_command_trace_requires_validated_clearance_before_command_one(
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
        "issued_utc": observer._utc(),
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
    order = []

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
        _kind,
        _root_fd,
    ):
        trace_events.extend(events)

    def validate_challenge(*_args):
        order.append("challenge")

    def validate_authorization(*_args):
        order.append("authorization")

    def launch(*_args, **_kwargs):
        order.append("popen")
        return FakeChild()

    with (
        mock.patch.object(
            observer,
            "_prepare",
            return_value=(
                SimpleNamespace(sha256="5" * 64),
                envelope,
                request,
                Ed25519PrivateKey.generate(),
                preflights,
                os.open(root, os.O_RDONLY | os.O_DIRECTORY),
                SimpleNamespace(key_id=authorization["observer_key_id"]),
            ),
        ),
        mock.patch.object(observer, "_source_digest", return_value="4" * 64),
        mock.patch.object(observer, "_absent"),
        mock.patch.object(observer, "_claim", return_value=authorization["claim_sha256"]),
        mock.patch.object(observer, "_append"),
        mock.patch.object(observer, "_ticks", return_value=7),
        mock.patch.object(observer.socket, "socket", return_value=FakeConnection()),
        mock.patch.object(observer.policy, "receive_packet", side_effect=receive_packet),
        mock.patch.object(observer.policy, "send_packet"),
        mock.patch.object(observer.policy, "validate_prelaunch"),
        mock.patch.object(observer.policy, "validate_command_semantics"),
        mock.patch.object(observer.policy, "validate_completed_command", return_value=checked),
        mock.patch.object(observer, "_validate_challenge", side_effect=validate_challenge),
        mock.patch.object(observer, "_validate_authorization", side_effect=validate_authorization),
        mock.patch.object(observer, "_validate_receipt", side_effect=validate_receipt),
        mock.patch.object(observer.subprocess, "Popen", side_effect=launch) as popen,
    ):
        if clearance_mutation is None:
            assert observer.run(SimpleNamespace(authority_socket="/unused")) == 0
        else:
            with pytest.raises((observer.ObserverError, observer.policy.TerminalPolicyError)):
                observer.run(SimpleNamespace(authority_socket="/unused"))

    if clearance_mutation is not None:
        assert popen.call_count == 1
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
        "prerequisites": [],
        "target_results": [],
        "terminal_state": {"kind": "SUCCEEDED"},
        "publication": observer.policy.plain(request.publication),
        "created_utc": observer._utc(),
    }
    return observer.policy.sign_message("terminal_receipt", message, authority_key)


def _validate_signed_receipt(
    observer, signed, authority_key, authorization, checkpoint, envelope, request, fd
):
    observer._validate_receipt(
        signed,
        authority_key.public_key(),
        authorization,
        checkpoint,
        envelope,
        request,
        [],
        "SUCCEEDED",
        fd,
    )


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
        observer._validate_receipt(
            signed,
            authority_key.public_key(),
            authorization,
            checkpoint,
            envelope,
            request,
            events,
            "SUCCEEDED",
            fd,
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
                observer, signed, authority_key, authorization, checkpoint, envelope, request, fd
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
                observer, signed, authority_key, authorization, checkpoint, envelope, request, fd
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
                observer, signed, authority_key, authorization, checkpoint, envelope, request, fd
            )
    finally:
        os.close(fd)
