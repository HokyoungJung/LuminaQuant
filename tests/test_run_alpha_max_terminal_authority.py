from __future__ import annotations

import ast
import argparse
import base64
import importlib.util
import os
import socket
import stat
import sys
from pathlib import Path
from unittest import mock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
AUTHORITY_PATH = ROOT / "scripts/research/run_alpha_max_terminal_authority.py"
KEYS_PATH = ROOT / "scripts/research/create_alpha_max_terminal_keys.py"
OBSERVER_PATH = ROOT / "scripts/research/run_alpha_max_terminal_observer.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def authority():
    return load(AUTHORITY_PATH, "alpha_max_terminal_authority_test")


def _process_creation_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".", 1)[0]
                aliases[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def qualified(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            parent = qualified(node.value)
            return f"{parent}.{node.attr}" if parent else None
        return None

    calls = {
        target
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        if (target := qualified(node.func)) is not None
    }

    def creates_process(target: str) -> bool:
        if target.startswith(("subprocess.", "multiprocessing.")):
            return True
        if not target.startswith("os."):
            return False
        operation = target.removeprefix("os.")
        return operation in {"popen", "startfile", "system"} or operation.startswith(
            ("fork", "exec", "spawn", "posix_spawn")
        )

    return {target for target in calls if creates_process(target)}


def _recovery_request(journal: str = "journal"):
    return type(
        "Request",
        (),
        {"publication": type("Publication", (), {"journal": journal})()},
    )()


def _receipt_request(journal: str = "journal"):
    return type(
        "Request",
        (),
        {
            "publication": type("Publication", (), {"journal": journal})(),
            "prerequisites": (),
            "scope": "acquisition",
            "request_id": "request",
            "sha256": "request",
        },
    )()


def test_seqpacket_exact_framing_rejects_length_noncanonical_duplicate_and_ancillary(authority):
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    read_fd, write_fd = os.pipe()
    try:
        cases = (
            b"\x00\x00\x00\x04{}\n",  # declared length does not match body
            b"\x00\x00\x00\x02{}",  # canonical JSON must include its LF
            b'\x00\x00\x00\x0e{"a":1,"a":2}\n',  # duplicate key
        )
        for packet in cases:
            right.send(packet)
            with pytest.raises(authority.policy.TerminalPolicyError):
                authority.policy.receive_packet(left)
        body = authority.policy.canonical_bytes({"a": 1})
        right.sendmsg(
            [len(body).to_bytes(4, "big") + body],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, write_fd.to_bytes(4, sys.byteorder))],
        )
        with pytest.raises(authority.policy.TerminalPolicyError):
            authority.policy.receive_packet(left)
    finally:
        os.close(read_fd)
        os.close(write_fd)
        left.close()
        right.close()


def test_signed_wire_domains_require_exact_fields_and_cannot_be_rebound(authority):
    private = Ed25519PrivateKey.generate()
    unsigned = {
        "schema": authority.SCHEMA,
        "type": "challenge",
        "authority_key_id": authority.policy.public_key_id(private.public_key()),
        "scope": "acquisition",
        "request_id": "0" * 64,
        "checkpoint_pin_sha256": "1" * 64,
        "envelope_sha256": "2" * 64,
        "request_sha256": "3" * 64,
        "command_bundle_sha256": "4" * 64,
        "nonce_b64": base64.b64encode(b"x" * 32).decode(),
        "issued_utc": "2026-07-22T00:00:00Z",
    }
    signed = authority.policy.sign_message("challenge", unsigned, private)
    assert set(signed) == authority.CHALLENGE
    authority.policy.verify_message("challenge", signed, private.public_key())
    with pytest.raises(authority.policy.TerminalPolicyError):
        authority.policy.verify_message("authorization", signed, private.public_key())
    rebound = dict(signed, type="authorization")
    with pytest.raises(authority.policy.TerminalPolicyError):
        authority.policy.verify_message("challenge", rebound, private.public_key())
    with pytest.raises(ValueError):
        authority._exact(dict(signed, extra=True), authority.CHALLENGE)


def test_authority_cli_selectors_are_fixed_and_only_observer_owns_process_launch(authority):
    serve = authority.parser().parse_args(
        [
            "serve",
            "--scope",
            "acquisition",
            "--policy",
            "/p",
            "--checkpoint",
            "/c",
            "--envelope",
            "/e",
            "--request",
            "/r",
            "--private-key",
            "/k",
            "--socket",
            "/s",
            "--evidence-root",
            "/root",
        ]
    )
    assert serve.socket == Path("/s") and serve.evidence_root == Path("/root")
    assert _process_creation_calls(AUTHORITY_PATH) == set()
    assert _process_creation_calls(OBSERVER_PATH) == {"subprocess.Popen"}


def test_peer_credentials_are_current_process_bound(authority):
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        pid, uid, start_ticks = authority.peer_credentials(left)
        assert pid == os.getpid()
        assert uid == os.getuid()
        assert start_ticks == authority._ticks(pid)
    finally:
        left.close()
        right.close()


def test_evidence_root_identity_and_publications_fail_closed(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    info = root.stat()
    identity = authority.policy.DirectoryIdentity(
        str(root), info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)
    )
    request = type("Request", (), {"evidence_root": identity})()
    fd = authority._open_root(request)
    os.close(fd)
    drifted = authority.policy.DirectoryIdentity(
        str(root), info.st_dev, info.st_ino, info.st_uid, info.st_gid, 0o755
    )
    with pytest.raises(ValueError):
        authority._open_root(type("Request", (), {"evidence_root": drifted})())
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, "receipt", b"receipt\n")
        with pytest.raises(FileExistsError):
            authority._write_new(fd, "receipt", b"replacement\n")
        (root / "journal-link").symlink_to(root / "receipt")
        with pytest.raises(OSError):
            authority._append_at(fd, "journal-link", b"event\n")
    finally:
        os.close(fd)


def test_new_receipt_refuses_unsafe_identity_before_writing(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    unsafe = type(
        "Info",
        (),
        {
            "st_mode": stat.S_IFREG | 0o644,
            "st_nlink": 1,
            "st_uid": os.getuid(),
            "st_gid": os.getgid(),
        },
    )()
    try:
        with (
            mock.patch.object(authority.os, "fstat", return_value=unsafe),
            mock.patch.object(authority.os, "write") as write,
            pytest.raises(ValueError, match="unsafe new publication"),
        ):
            authority._write_new(fd, "receipt", b"receipt\n")
        write.assert_not_called()
        assert (root / "receipt").read_bytes() == b""
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "path",
    (
        "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source",
        "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source/child",
        "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc",
        "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc/child",
    ),
)
def test_authority_rejects_quarantined_cli_paths_lexically_without_opening(authority, path):
    with pytest.raises(authority.policy.TerminalPolicyError):
        authority._absolute_cli(Path(path))


def test_authority_rejects_symlinked_evidence_root_ancestor(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    info = root.stat()
    request = type(
        "Request",
        (),
        {
            "evidence_root": authority.policy.DirectoryIdentity(
                str(tmp_path / "linked" / "evidence"),
                info.st_dev,
                info.st_ino,
                info.st_uid,
                info.st_gid,
                stat.S_IMODE(info.st_mode),
            )
        },
    )()
    (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(authority.policy.TerminalPolicyError):
        authority._open_root(request)


def test_recovery_preserves_pending_until_durable_append_and_never_serves(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        pending = authority.policy.canonical_bytes({"event": "launch_intent", "sequence": 0})
        authority._write_new(fd, authority.PENDING_NAME, pending)
        with (
            mock.patch.object(authority, "_validate_stored_authorization"),
            mock.patch.object(authority, "_append_at", side_effect=OSError("disk full")),
            pytest.raises(OSError),
        ):
            authority._reconcile_pending(
                fd,
                None,
                None,
                type(
                    "Request",
                    (),
                    {"publication": type("Publication", (), {"journal": "journal"})()},
                )(),
                None,
            )
        assert (root / authority.PENDING_NAME).read_bytes() == pending
    finally:
        os.close(fd)
    tree = ast.parse(AUTHORITY_PATH.read_text(encoding="utf-8"))
    recover = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "recover"
    )
    assert not any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == "serve"
        for node in ast.walk(recover)
    )


def test_reconcile_pending_repairs_only_strict_prefix_tails_durably(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    authorization = {"authorization_id": "authorization"}
    pending_record = {"authorization_id": "authorization", "event": "launch_intent"}
    authorization_bytes = authority.policy.canonical_bytes(authorization)
    pending = authority.policy.canonical_bytes(pending_record)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, "journal", authorization_bytes + pending[:9])
        authority._write_new(fd, authority.PENDING_NAME, pending)
        with (
            mock.patch.object(authority, "_validate_stored_authorization"),
            mock.patch.object(authority.policy, "validate_scope_artifacts"),
            mock.patch.object(authority.os, "fsync", wraps=os.fsync) as fsync,
        ):
            records = authority._reconcile_pending(fd, None, None, _recovery_request(), None)
        assert records == [authorization, pending_record]
        assert (root / "journal").read_bytes() == authorization_bytes + pending
        assert not (root / authority.PENDING_NAME).exists()
        assert fsync.call_count >= 4
    finally:
        os.close(fd)


def test_reconcile_pending_only_appends_validated_record(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    pending_record = {"authorization_id": "authorization"}
    pending = authority.policy.canonical_bytes(pending_record)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, authority.PENDING_NAME, pending)
        with mock.patch.object(authority, "_validate_stored_authorization") as validate:
            assert authority._reconcile_pending(fd, None, None, _recovery_request(), None) == [
                pending_record
            ]
        validate.assert_called_once()
        assert (root / "journal").read_bytes() == pending
        assert not (root / authority.PENDING_NAME).exists()
    finally:
        os.close(fd)


def test_reconcile_pending_fully_appended_record_is_idempotent(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    authorization = {"authorization_id": "authorization"}
    pending_record = {"authorization_id": "authorization", "event": "launch_intent"}
    pending = authority.policy.canonical_bytes(pending_record)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(
            fd, "journal", authority.policy.canonical_bytes(authorization) + pending
        )
        authority._write_new(fd, authority.PENDING_NAME, pending)
        assert authority._reconcile_pending(fd, None, None, _recovery_request(), None) == [
            authorization,
            pending_record,
        ]
        assert not (root / authority.PENDING_NAME).exists()
        assert (root / "journal").read_bytes() == authority.policy.canonical_bytes(
            authorization
        ) + pending
    finally:
        os.close(fd)


def test_reconcile_pending_preserves_nonprefix_tail_and_pending(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    pending = authority.policy.canonical_bytes({"authorization_id": "authorization"})
    journal = authority.policy.canonical_bytes({"authorization_id": "authorization"}) + b"wrong"
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, "journal", journal)
        authority._write_new(fd, authority.PENDING_NAME, pending)
        with pytest.raises(ValueError, match="tail"):
            authority._reconcile_pending(fd, None, None, _recovery_request(), None)
        assert (root / "journal").read_bytes() == journal
        assert (root / authority.PENDING_NAME).read_bytes() == pending
    finally:
        os.close(fd)


def test_reconcile_pending_preserves_pending_when_complete_journal_is_malformed(
    authority, tmp_path
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    pending = authority.policy.canonical_bytes({"authorization_id": "authorization"})
    journal = b'{"authorization_id":}\n'
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, "journal", journal)
        authority._write_new(fd, authority.PENDING_NAME, pending)
        with pytest.raises(authority.policy.TerminalPolicyError):
            authority._reconcile_pending(fd, None, None, _recovery_request(), None)
        assert (root / "journal").read_bytes() == journal
        assert (root / authority.PENDING_NAME).read_bytes() == pending
    finally:
        os.close(fd)


def test_reconcile_pending_validates_pending_before_journal(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, "journal", b'{"journal":}\n')
        authority._write_new(fd, authority.PENDING_NAME, b'{"pending":}\n')
        with pytest.raises(authority.policy.TerminalPolicyError, match="pending journal record"):
            authority._reconcile_pending(fd, None, None, _recovery_request(), None)
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "journal_records",
    [
        [{"event": "child_exited", "sequence": 1}, {"authorization_id": "authorization"}],
        [
            {"authorization_id": "authorization"},
            {"event": "child_exited", "sequence": 1},
            {"event": "child_exited", "sequence": 1},
        ],
        [{"authorization_id": "authorization"}, {"event": "substituted", "sequence": 1}],
        [
            {"authorization_id": "authorization"},
            {"event": "child_exited", "sequence": 1},
            {"event": "trailing", "sequence": 2},
        ],
    ],
)
def test_receipt_rejects_journal_records_not_exactly_equal(authority, tmp_path, journal_records):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    authorization = {
        "authorization_id": "authorization",
        "claim_sha256": "claim",
        "observer_key_id": "observer",
        "observer_pid": 1,
        "observer_start_ticks": 2,
        "command_bundle_sha256": "bundle",
    }
    events = [{"event": "child_exited", "sequence": 1}]
    try:
        authority._write_new(
            fd,
            "journal",
            b"".join(authority.policy.canonical_bytes(record) for record in journal_records),
        )
        with pytest.raises(ValueError, match="does not match"):
            authority._receipt(
                Ed25519PrivateKey.generate(),
                type("Checkpoint", (), {"sha256": "checkpoint"})(),
                object(),
                _receipt_request(),
                authorization,
                events,
                {"kind": "SUCCEEDED"},
                fd,
            )
    finally:
        os.close(fd)


def test_receipt_accepts_exact_authorization_and_event_journal(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    authorization = {
        "authorization_id": "authorization",
        "claim_sha256": "claim",
        "observer_key_id": "observer",
        "observer_pid": 1,
        "observer_start_ticks": 2,
        "command_bundle_sha256": "bundle",
    }
    events = [{"event": "child_exited", "sequence": 1}]
    journal = b"".join(
        authority.policy.canonical_bytes(record) for record in [authorization, *events]
    )
    try:
        authority._write_new(fd, "journal", journal)
        with (
            mock.patch.object(authority.policy, "validate_scope_artifacts", return_value=()),
            mock.patch.object(
                authority.policy,
                "sign_message",
                side_effect=lambda _domain, message, _key: message,
            ),
        ):
            receipt = authority._receipt(
                Ed25519PrivateKey.generate(),
                type("Checkpoint", (), {"sha256": "checkpoint"})(),
                type("Envelope", (), {"sha256": "envelope"})(),
                _receipt_request(),
                authorization,
                events,
                {"kind": "SUCCEEDED"},
                fd,
            )
        assert receipt["journal_sha256"] == authority._sha(journal)
    finally:
        os.close(fd)


def test_recovery_state_classifies_all_nonlaunching_terminal_prefixes(authority):
    assert authority.recovery_state([], 1)["kind"] == "UNAUTHENTICATED_TERMINAL"
    assert (
        authority.recovery_state([{"command_index": 0, "event": "start_failed", "errno": 5}], 1)[
            "kind"
        ]
        == "START_FAILED"
    )
    assert (
        authority.recovery_state([{"command_index": 0, "event": "launch_intent"}], 1)["kind"]
        == "START_UNKNOWN"
    )
    assert (
        authority.recovery_state(
            [
                {
                    "command_index": 0,
                    "event": "child_started",
                    "child_pid": 1,
                    "child_start_ticks": 2,
                }
            ],
            1,
        )["kind"]
        == "OBSERVER_LOST"
    )
    assert (
        authority.recovery_state(
            [{"command_index": 0, "event": "child_exited", "return_code": 9}], 1
        )["kind"]
        == "FAILED"
    )
    assert (
        authority.recovery_state(
            [{"command_index": 0, "event": "child_exited", "return_code": 0}], 1
        )["kind"]
        == "SUCCEEDED"
    )


@pytest.mark.parametrize(
    ("events", "expected"),
    (
        (
            [{"command_index": 0, "event": "child_exited", "return_code": 0, "sequence": 2}],
            "UNAUTHENTICATED_TERMINAL",
        ),
        (
            [
                {"command_index": 0, "event": "child_exited", "return_code": 0, "sequence": 2},
                {"command_index": 1, "event": "launch_intent", "sequence": 3},
            ],
            "START_UNKNOWN",
        ),
        (
            [
                {"command_index": 0, "event": "child_exited", "return_code": 0, "sequence": 2},
                {
                    "command_index": 1,
                    "event": "child_started",
                    "child_pid": 9,
                    "child_start_ticks": 10,
                    "sequence": 4,
                },
            ],
            "OBSERVER_LOST",
        ),
        (
            [
                {"command_index": 0, "event": "child_exited", "return_code": 0, "sequence": 2},
                {"command_index": 1, "event": "child_exited", "return_code": 1, "sequence": 5},
            ],
            "FAILED",
        ),
        (
            [
                {"command_index": 0, "event": "child_exited", "return_code": 0, "sequence": 2},
                {"command_index": 1, "event": "child_exited", "return_code": 0, "sequence": 5},
            ],
            "SUCCEEDED",
        ),
    ),
)
def test_two_command_recovery_requires_both_successful_exits(authority, events, expected):
    assert authority.recovery_state(events, 2)["kind"] == expected


def test_key_creator_is_empty_0700_all_or_none_and_emits_raw_public_only(tmp_path):
    keys = load(KEYS_PATH, "alpha_max_terminal_keys_test")
    insecure = tmp_path / "insecure"
    insecure.mkdir()
    insecure.chmod(0o755)
    with pytest.raises(ValueError):
        keys.create_keys(insecure)
    root = tmp_path / "keys"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    summary = keys.create_keys(root)
    expected = {f"{role}.{kind}" for role in keys.KEY_NAMES for kind in ("private", "public")}
    assert {item.name for item in root.iterdir()} == expected
    assert all(
        stat.S_IMODE(item.stat().st_mode) == 0o400 and len(item.read_bytes()) == 32
        for item in root.iterdir()
    )
    assert set(summary) == set(keys.KEY_NAMES)
    assert all(
        set(value) == {"key_id", "public_key_b64", "public_key_sha256"}
        for value in summary.values()
    )
    assert all(
        base64.b64decode(value["public_key_b64"], validate=True)
        == (root / f"{name}.public").read_bytes()
        for name, value in summary.items()
    )
    rollback = tmp_path / "rollback"
    rollback.mkdir(mode=0o700)
    rollback.chmod(0o700)
    original = keys._create_file
    calls = 0

    def fail_second_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("write failed")
        return original(*args, **kwargs)

    with (
        mock.patch.object(keys, "_create_file", side_effect=fail_second_create),
        pytest.raises(OSError),
    ):
        keys.create_keys(rollback)
    assert calls == 2
    assert list(rollback.iterdir()) == []


def test_key_creator_removes_current_registered_file_after_write_failure(tmp_path):
    keys = load(KEYS_PATH, "alpha_max_terminal_keys_write_failure_test")
    root = tmp_path / "keys"
    root.mkdir(mode=0o700)
    root.chmod(0o700)

    with (
        mock.patch.object(keys, "_write_all", side_effect=OSError("partial write failed")),
        pytest.raises(OSError, match="partial write failed"),
    ):
        keys.create_keys(root)

    assert list(root.iterdir()) == []


def test_key_creator_reports_replacement_inode_without_deleting_it(tmp_path):
    keys = load(KEYS_PATH, "alpha_max_terminal_keys_replacement_test")
    root = tmp_path / "keys"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    replacement = root / "authority.private"

    def replace_current_file_then_fail(_fd, _data):
        replacement.unlink()
        replacement.write_bytes(b"replacement")
        raise OSError("write failed after replacement")

    with (
        mock.patch.object(keys, "_write_all", side_effect=replace_current_file_then_fail),
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        keys.create_keys(root)

    creation_error, cleanup_error = raised.value.exceptions
    assert isinstance(creation_error, OSError)
    assert str(creation_error) == "write failed after replacement"
    assert raised.value.__cause__ is creation_error
    assert isinstance(cleanup_error, BaseExceptionGroup)
    assert isinstance(cleanup_error.exceptions[0], ValueError)
    assert "identity changed during cleanup: authority.private" in str(cleanup_error.exceptions[0])
    assert replacement.read_bytes() == b"replacement"
    assert [item.name for item in root.iterdir()] == ["authority.private"]


def test_key_cleanup_aggregates_stat_unlink_and_fsync_failures(tmp_path):
    keys = load(KEYS_PATH, "alpha_max_terminal_keys_cleanup_failures_test")
    root = tmp_path / "keys"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    safe_info = type("Info", (), {"st_dev": 1, "st_ino": 1})()
    try:
        with (
            mock.patch.object(keys.os, "stat", side_effect=(OSError("stat failed"), safe_info)),
            mock.patch.object(keys.os, "unlink", side_effect=OSError("unlink failed")),
            mock.patch.object(keys.os, "fsync", side_effect=OSError("fsync failed")),
            pytest.raises(BaseExceptionGroup) as raised,
        ):
            keys._cleanup(root_fd, [("safe", 1, 1), ("missing", 2, 2)])
    finally:
        os.close(root_fd)

    assert [str(error) for error in raised.value.exceptions] == [
        "stat failed",
        "unlink failed",
        "fsync failed",
    ]


@pytest.mark.parametrize(
    "path",
    (
        "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source",
        "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source/child",
        "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc",
        "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc/child",
    ),
)
def test_key_creator_rejects_quarantined_roots_lexically_without_opening(path):
    keys = load(KEYS_PATH, "alpha_max_terminal_keys_quarantined_test")
    with pytest.raises(argparse.ArgumentTypeError, match="forbidden root"):
        keys._absolute(path)
    with (
        mock.patch.object(keys.policy.os, "open") as open_directory,
        mock.patch.object(keys.os, "fstat") as fstat,
        pytest.raises(keys.policy.TerminalPolicyError, match="forbidden root"),
    ):
        keys.create_keys(Path(path))
    open_directory.assert_not_called()
    fstat.assert_not_called()


def test_key_creator_rejects_symlinked_root_ancestor(tmp_path):
    keys = load(KEYS_PATH, "alpha_max_terminal_keys_ancestor_symlink_test")
    root = tmp_path / "keys"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(keys.policy.TerminalPolicyError, match="cannot open key root"):
        keys.create_keys(linked / "keys")

    assert list(root.iterdir()) == []
