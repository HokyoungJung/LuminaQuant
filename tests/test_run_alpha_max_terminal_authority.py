from __future__ import annotations

import ast
import hashlib
import argparse
import base64
import importlib.util
import inspect
import json
import os
import socket
import stat
import sys
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
AUTHORITY_PATH = ROOT / "scripts/research/run_alpha_max_terminal_authority.py"
KEYS_PATH = ROOT / "scripts/research/create_alpha_max_terminal_keys.py"
OBSERVER_PATH = ROOT / "scripts/research/run_alpha_max_terminal_observer.py"
EXTERNAL_ENVELOPE_TEST_PATH = ROOT / "tests/test_alpha_max_terminal_external_envelope.py"
POLICY_PATH = ROOT / "configs/research/alpha_max_terminal_authority_policy_v1.json"
ACCEPTED_ALPHA_COMMIT = "391000b40717386765bfa39bd212d91c2e3be794"


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
    shared_fd = authority._open_root(request)
    try:
        with pytest.raises(ValueError, match="terminal evidence root is active"):
            authority._open_root(request, exclusive=True)
    finally:
        os.close(shared_fd)
    exclusive_fd = authority._open_root(request, exclusive=True)
    os.close(exclusive_fd)
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
            authority._append_at(fd, "journal-link", b"event\n", None)
        journal_identity = authority._append_at(fd, "journal", b"event\n", None)
        replacement = root / "journal-replacement"
        replacement.write_bytes(b"replacement\n")
        replacement.chmod(0o600)
        os.replace(replacement, root / "journal")
        with pytest.raises(ValueError, match="unsafe terminal journal"):
            authority._append_at(fd, "journal", b"event\n", journal_identity)
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
            "st_dev": 1,
            "st_ino": 2,
        },
    )()
    try:
        with (
            mock.patch.object(authority.os, "fstat", return_value=unsafe),
            mock.patch.object(authority.os, "write") as write,
            pytest.raises(BaseExceptionGroup) as raised,
        ):
            authority._write_new(fd, "receipt", b"receipt\n")
        assert isinstance(raised.value.exceptions[0], ValueError)
        write.assert_not_called()
        assert (root / "receipt").read_bytes() == b""
    finally:
        os.close(fd)


def test_read_at_preserves_leaf_close_failure_with_primary(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    (root / "claim").write_bytes(b"claim")
    try:
        with (
            mock.patch.object(authority.os, "fstat", side_effect=ValueError("leaf failed")),
            mock.patch.object(authority.os, "close", side_effect=OSError("leaf close failed")),
            pytest.raises(BaseExceptionGroup) as raised,
        ):
            authority._read_at(root_fd, "claim", "claim")
    finally:
        os.close(root_fd)

    primary, cleanup = raised.value.exceptions
    assert str(primary) == "leaf failed"
    assert [str(error) for error in cleanup.exceptions] == ["leaf close failed"]


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


def test_verify_rejects_out_of_context_receipt_before_signature_read(authority):
    checkpoint = type("Checkpoint", (), {})()
    envelope = type("Envelope", (), {})()
    request = type(
        "Request",
        (),
        {
            "evidence_root": type("Root", (), {"path": "/evidence"})(),
            "publication": type("Publication", (), {"receipt": "receipt"})(),
        },
    )()
    args = argparse.Namespace(
        policy=Path("/policy"),
        checkpoint=Path("/checkpoint"),
        envelope=Path("/envelope"),
        request=Path("/request"),
        public_key=Path("/key"),
        receipt=Path("/other/receipt"),
        scope="acquisition",
    )
    with (
        mock.patch.object(authority.policy, "load_policy", return_value=object()),
        mock.patch.object(authority.policy, "load_checkpoint", return_value=checkpoint),
        mock.patch.object(authority.policy, "load_envelope", return_value=envelope),
        mock.patch.object(authority.policy, "load_request", return_value=request),
        mock.patch.object(authority.policy, "validate_prelaunch"),
        mock.patch.object(authority.policy, "derive_scope_commands"),
        mock.patch.object(authority.policy, "verify_signed_receipt") as verify,
        pytest.raises(ValueError, match="receipt context mismatch"),
    ):
        authority.verify_receipt(args)
    verify.assert_not_called()


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


def test_bind_server_preserves_primary_and_socket_removal_failures(authority):
    server = mock.Mock()
    server.listen.side_effect = OSError("listen failed")
    socket_info = type(
        "Info",
        (),
        {
            "st_mode": stat.S_IFSOCK | 0o600,
            "st_dev": 1,
            "st_ino": 2,
            "st_uid": os.getuid(),
            "st_gid": os.getgid(),
        },
    )()
    with (
        mock.patch.object(authority.socket, "socket", return_value=server),
        mock.patch.object(authority.os, "stat", side_effect=(FileNotFoundError(), socket_info)),
        mock.patch.object(authority.os, "chmod") as chmod,
        mock.patch.object(authority.os, "fsync"),
        mock.patch.object(
            authority, "_remove_socket", side_effect=OSError("remove failed")
        ) as remove,
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        authority._bind_server(73, "authority.sock")

    primary_error, cleanup_error = raised.value.exceptions
    assert str(primary_error) == "listen failed"
    assert raised.value.__cause__ is primary_error
    assert isinstance(cleanup_error, BaseExceptionGroup)
    assert [str(error) for error in cleanup_error.exceptions] == ["remove failed"]
    server.close.assert_called_once_with()
    chmod.assert_not_called()
    remove.assert_called_once_with(73, "authority.sock", (1, 2))


def test_bind_server_cleans_up_after_post_bind_identity_failure(authority):
    server = mock.Mock()
    with (
        mock.patch.object(authority.socket, "socket", return_value=server),
        mock.patch.object(
            authority.os,
            "stat",
            side_effect=(FileNotFoundError(), OSError("post-bind stat failed")),
        ),
        mock.patch.object(authority.os, "chmod") as chmod,
        mock.patch.object(authority, "_remove_socket") as remove,
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        authority._bind_server(73, "authority.sock")

    primary_error, cleanup_error = raised.value.exceptions
    assert str(primary_error) == "post-bind stat failed"
    assert [str(error) for error in cleanup_error.exceptions] == [
        "bound authority socket identity is unknown"
    ]
    chmod.assert_not_called()
    remove.assert_not_called()
    server.close.assert_called_once_with()


def test_bind_server_does_not_mutate_or_remove_replacement_after_identity_lookup_failure(
    authority, tmp_path
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    socket_name = "authority.sock"
    socket_path = root / socket_name
    replacement = root / "replacement"
    original_stat = authority.os.stat
    calls = 0

    def fail_first_post_bind_lookup(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            socket_path.unlink()
            replacement.write_bytes(b"replacement")
            os.replace(replacement, socket_path)
            raise OSError("post-bind lookup failed")
        return original_stat(*args, **kwargs)

    try:
        with (
            mock.patch.object(authority.os, "stat", side_effect=fail_first_post_bind_lookup),
            mock.patch.object(authority.os, "chmod") as chmod,
            mock.patch.object(authority, "_remove_socket") as remove,
            pytest.raises(BaseExceptionGroup) as raised,
        ):
            authority._bind_server(root_fd, socket_name)
    finally:
        os.close(root_fd)

    primary_error, cleanup_error = raised.value.exceptions
    assert str(primary_error) == "post-bind lookup failed"
    assert [str(error) for error in cleanup_error.exceptions] == [
        "bound authority socket identity is unknown"
    ]
    assert socket_path.read_bytes() == b"replacement"
    chmod.assert_not_called()
    remove.assert_not_called()


def test_authorization_has_exact_five_minute_window_and_command_zero_boundary(authority):
    private = Ed25519PrivateKey.generate()
    checkpoint = type("Checkpoint", (), {"sha256": "c" * 64})()
    envelope = type("Envelope", (), {"sha256": "e" * 64})()
    request = type(
        "Request", (), {"scope": "acquisition", "request_id": "r" * 64, "sha256": "q" * 64}
    )()
    proof = {
        "claim_sha256": "l" * 64,
        "observer_key_id": "observer",
        "observer_pid": 1,
        "observer_uid": os.getuid(),
        "observer_start_ticks": 2,
        "observer_source_sha256": "s" * 64,
    }
    epoch = _launch_intent_epoch(authority, "2026-07-22T00:00:00Z")

    with (
        mock.patch.object(authority.time, "time", return_value=epoch),
        mock.patch.object(authority, "_bundle", return_value="b" * 64),
    ):
        authorization = authority._authorization(
            private, checkpoint, envelope, request, (("command",),), proof
        )

    authority.policy.verify_message("authorization", authorization, private.public_key())
    not_before, expires = authority._authorization_window(authorization)
    assert expires - not_before == 300
    for offset, accepted in ((299, True), (300, False)):
        timestamp = authority.time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", authority.time.gmtime(not_before + offset)
        )
        event = {"event": "launch_intent", "command_index": 0, "observed_utc": timestamp}
        if accepted:
            authority._validate_launch_intent_window(
                event, authorization, None, not_before + offset, not_before + offset
            )
        else:
            with pytest.raises(ValueError, match="outside authorization window"):
                authority._validate_launch_intent_window(
                    event, authorization, None, not_before + offset, not_before + offset
                )


@pytest.mark.parametrize(
    ("observed_utc", "received_before_utc", "received_after_utc", "valid"),
    (
        ("2026-07-21T23:59:59Z", "2026-07-22T00:00:00Z", "2026-07-22T00:00:00Z", False),
        ("2026-07-22T00:00:00Z", "2026-07-22T00:00:00Z", "2026-07-22T00:00:00Z", True),
        ("2026-07-22T00:00:00Z", "2026-07-21T23:59:59Z", "2026-07-22T00:00:00Z", False),
        ("2026-07-22T00:00:00Z", "2026-07-22T00:00:00Z", "2026-07-21T23:59:59Z", False),
        ("2026-07-22T00:00:00Z", "2026-07-21T23:59:59Z", "2026-07-22T00:00:01Z", False),
        ("2026-07-22T00:03:59Z", "2026-07-22T00:03:59Z", "2026-07-22T00:03:59Z", True),
        ("2026-07-22T00:03:59Z", "2026-07-22T00:03:59Z", "2026-07-22T00:03:58Z", False),
        ("2026-07-22T00:04:00Z", "2026-07-22T00:03:59Z", "2026-07-22T00:03:59Z", False),
        ("2026-07-22T00:03:59Z", "2026-07-22T00:04:00Z", "2026-07-22T00:04:00Z", False),
        ("2026-07-22T00:03:59Z", "2026-07-22T00:03:59Z", "2026-07-22T00:04:00Z", False),
    ),
)
def test_launch_intent_requires_observation_and_receive_bracket_within_signed_window(
    authority, observed_utc, received_before_utc, received_after_utc, valid
):
    authorization = {
        "not_before_utc": "2026-07-22T00:00:00Z",
        "expires_utc": "2026-07-22T00:04:00Z",
    }
    event = {"event": "launch_intent", "command_index": 0, "observed_utc": observed_utc}
    received_before = int(
        authority.policy.parse_utc_second(
            received_before_utc, "test receive-before time"
        ).timestamp()
    )
    received_after = int(
        authority.policy.parse_utc_second(received_after_utc, "test receive-after time").timestamp()
    )

    if valid:
        authority._validate_launch_intent_window(
            event, authorization, None, received_before, received_after
        )
    else:
        with pytest.raises(ValueError, match="outside authorization window"):
            authority._validate_launch_intent_window(
                event, authorization, None, received_before, received_after
            )


def _launch_intent_epoch(authority, value):
    return int(authority.policy.parse_utc_second(value, "test launch intent time").timestamp())


def test_command_one_uses_fresh_exact_clearance_after_authorization_expires(authority):
    authorization = {
        "not_before_utc": "2026-07-22T00:00:00Z",
        "expires_utc": "2026-07-22T00:05:00Z",
    }
    clearance = {
        "issued_utc": "2026-07-22T00:10:00Z",
        "next_command_index": 1,
        "authority_signature_b64": "fresh-signature",
    }
    event = {
        "event": "launch_intent",
        "command_index": 1,
        "prior_clearance": clearance,
        "observed_utc": "2026-07-22T00:10:30Z",
    }

    authority._validate_launch_intent_window(
        event,
        authorization,
        clearance,
        _launch_intent_epoch(authority, "2026-07-22T00:10:30Z"),
        _launch_intent_epoch(authority, "2026-07-22T00:10:31Z"),
    )


def test_command_one_rejects_future_clearance(authority):
    clearance = {
        "issued_utc": "2026-07-22T00:10:01Z",
        "next_command_index": 1,
        "authority_signature_b64": "future-signature",
    }
    event = {
        "event": "launch_intent",
        "command_index": 1,
        "prior_clearance": clearance,
        "observed_utc": "2026-07-22T00:10:00Z",
    }

    with pytest.raises(ValueError, match="outside clearance window"):
        authority._validate_launch_intent_window(
            event,
            {"not_before_utc": "2026-07-22T00:00:00Z", "expires_utc": "2026-07-22T00:05:00Z"},
            clearance,
            _launch_intent_epoch(authority, "2026-07-22T00:10:00Z"),
            _launch_intent_epoch(authority, "2026-07-22T00:10:00Z"),
        )


@pytest.mark.parametrize(
    (
        "event_clearance",
        "command_index",
        "observed_utc",
        "received_before_utc",
        "received_after_utc",
        "error",
    ),
    (
        (
            None,
            1,
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:31Z",
            "clearance mismatch",
        ),
        (
            {
                "issued_utc": "2026-07-22T00:10:00Z",
                "next_command_index": 1,
                "authority_signature_b64": "different-signature",
            },
            1,
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:31Z",
            "clearance mismatch",
        ),
        (
            {
                "issued_utc": "2026-07-22T00:09:00Z",
                "next_command_index": 1,
                "authority_signature_b64": "replayed-signature",
            },
            1,
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:31Z",
            "clearance mismatch",
        ),
        (
            {
                "issued_utc": "2026-07-22T00:10:00Z",
                "next_command_index": 1,
                "authority_signature_b64": "fresh-signature",
            },
            2,
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:31Z",
            "clearance mismatch",
        ),
        (
            {
                "issued_utc": "2026-07-22T00:10:00Z",
                "next_command_index": 1,
                "authority_signature_b64": "fresh-signature",
            },
            1,
            "2026-07-22T00:11:01Z",
            "2026-07-22T00:11:01Z",
            "2026-07-22T00:11:01Z",
            "outside clearance window",
        ),
        (
            {
                "issued_utc": "2026-07-22T00:10:00Z",
                "next_command_index": 1,
                "authority_signature_b64": "fresh-signature",
            },
            1,
            "2026-07-22T00:09:59Z",
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:31Z",
            "outside clearance window",
        ),
        (
            {
                "issued_utc": "2026-07-22T00:10:00Z",
                "next_command_index": 1,
                "authority_signature_b64": "fresh-signature",
            },
            1,
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:09:59Z",
            "2026-07-22T00:10:31Z",
            "outside clearance window",
        ),
        (
            {
                "issued_utc": "2026-07-22T00:10:00Z",
                "next_command_index": 1,
                "authority_signature_b64": "fresh-signature",
            },
            1,
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:11:01Z",
            "outside clearance window",
        ),
        (
            {
                "issued_utc": "2026-07-22T00:10:00Z",
                "next_command_index": 1,
                "authority_signature_b64": "fresh-signature",
            },
            1,
            "2026-07-22T00:10:30Z",
            "2026-07-22T00:10:31Z",
            "2026-07-22T00:10:30Z",
            "outside clearance window",
        ),
    ),
)
def test_command_one_rejects_invalid_clearance_or_receive_bracket(
    authority,
    event_clearance,
    command_index,
    observed_utc,
    received_before_utc,
    received_after_utc,
    error,
):
    clearance = {
        "issued_utc": "2026-07-22T00:10:00Z",
        "next_command_index": 1,
        "authority_signature_b64": "fresh-signature",
    }
    event = {
        "event": "launch_intent",
        "command_index": command_index,
        "prior_clearance": event_clearance,
        "observed_utc": observed_utc,
    }

    with pytest.raises(ValueError, match=error):
        authority._validate_launch_intent_window(
            event,
            {"not_before_utc": "2026-07-22T00:00:00Z", "expires_utc": "2026-07-22T00:05:00Z"},
            clearance,
            _launch_intent_epoch(authority, received_before_utc),
            _launch_intent_epoch(authority, received_after_utc),
        )


def test_serve_rejects_expired_launch_intent_before_artifact_validation(authority):
    server = mock.Mock()
    connection = mock.Mock()
    server.accept.return_value = (connection, None)
    private = mock.Mock()
    private.public_key.return_value = object()
    request = type(
        "Request",
        (),
        {
            "authority_socket": "/evidence/authority.sock",
            "publication": type("Publication", (), {"receipt": "receipt"})(),
        },
    )()
    envelope = type(
        "Envelope", (), {"authority_key": type("AuthorityKey", (), {"key_id": "authority"})()}
    )()
    authorization = {
        "authorization_id": "authorization",
        "not_before_utc": "2026-07-22T00:00:00Z",
        "expires_utc": "2026-07-22T00:04:00Z",
    }
    event = {
        "event": "launch_intent",
        "authorization_id": "authorization",
        "command_index": 0,
        "observed_utc": "2026-07-22T00:03:59Z",
    }
    expires_at = int(
        authority.policy.parse_utc_second(
            authorization["expires_utc"], "test authorization expiry"
        ).timestamp()
    )

    with (
        mock.patch.object(authority, "_context", return_value=(object(), envelope, request, ())),
        mock.patch.object(authority.policy, "secure_private_key", return_value=private),
        mock.patch.object(authority.policy, "public_key_id", return_value="authority"),
        mock.patch.object(authority, "_open_root", return_value=73),
        mock.patch.object(authority.policy, "validate_prelaunch"),
        mock.patch.object(authority, "_bind_server", return_value=(server, (1, 2))),
        mock.patch.object(authority, "peer_credentials", return_value=(1, 2, 3)),
        mock.patch.object(authority, "_challenge", return_value={"challenge": True}),
        mock.patch.object(authority, "_validate_proof"),
        mock.patch.object(authority, "_authorization", return_value=authorization),
        mock.patch.object(authority.policy, "receive_packet", side_effect=({}, event)),
        mock.patch.object(authority.policy, "send_packet"),
        mock.patch.object(authority.policy, "validate_scope_artifacts") as validate_artifacts,
        mock.patch.object(authority, "_remove_socket"),
        mock.patch.object(authority.os, "close"),
        mock.patch.object(
            authority.time, "time", side_effect=(expires_at - 1, expires_at)
        ) as sampled_time,
        pytest.raises(ValueError, match="outside authorization window"),
    ):
        authority.serve(argparse.Namespace(private_key=Path("/key")))

    validate_artifacts.assert_not_called()
    assert sampled_time.call_args_list == [mock.call(), mock.call()]


def test_serve_rejects_replayed_command_one_clearance_before_further_progress(authority, tmp_path):
    (
        checkpoint,
        envelope,
        request,
        commands,
        private,
        observer_private,
        _authorization,
        root,
        root_fd,
    ) = _signed_recovery_history(authority, tmp_path, intent_history=False)
    assert len(commands) > 1
    stdout_path = root / request.publication.stdout[0]
    stderr_path = root / request.publication.stderr[0]
    stdout_path.write_bytes(b"stdout\n")
    stderr_path.write_bytes(b"stderr\n")
    stdout_path.chmod(0o600)
    stderr_path.chmod(0o600)

    def stream(path):
        info = path.stat()
        return {
            "path": path.name,
            "st_dev": info.st_dev,
            "st_ino": info.st_ino,
            "st_uid": info.st_uid,
            "st_gid": info.st_gid,
            "mode": stat.S_IMODE(info.st_mode),
            "nlink": info.st_nlink,
            "isatty": False,
        }

    stdout = stream(stdout_path)
    stderr = stream(stderr_path)
    authorization = {
        "authorization_id": "a" * 64,
        "not_before_utc": "2026-07-22T00:00:00Z",
        "expires_utc": "2026-07-22T00:05:00Z",
    }
    observed_utc = "2026-07-22T00:00:00Z"
    epoch = _launch_intent_epoch(authority, observed_utc)
    clearance = None
    clearance_function = authority._clearance
    command_one_sequence = 3

    def event(sequence, kind, command_index, prior_clearance, **extra):
        return authority.policy.sign_message(
            "process_event",
            {
                "schema": authority.SCHEMA,
                "type": "process_event",
                "event": kind,
                "authorization_id": authorization["authorization_id"],
                "sequence": sequence,
                "command_index": command_index,
                "argv_sha256": hashlib.sha256(
                    authority.policy.canonical_bytes(commands[command_index])
                ).hexdigest(),
                "environment_sha256": hashlib.sha256(
                    authority.policy.canonical_bytes(request.environment)
                ).hexdigest(),
                "prior_clearance": prior_clearance,
                "observed_utc": observed_utc,
                **extra,
            },
            observer_private,
        )

    child = {
        "child_pid": 1,
        "child_start_ticks": 1,
        "stdin_identity": {"kind": "DEVNULL", "isatty": False},
        "stdout": stdout,
        "stderr": stderr,
    }
    received = iter(
        (
            {"proof": True},
            event(0, "launch_intent", 0, None),
            event(1, "child_started", 0, None, **child),
            event(
                2,
                "child_exited",
                0,
                None,
                **child,
                return_code=0,
                stdout_sha256=hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
                stdout_byte_count=stdout_path.stat().st_size,
                stderr_sha256=hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
                stderr_byte_count=stderr_path.stat().st_size,
            ),
        )
    )

    def receive_packet(_connection):
        nonlocal clearance, command_one_sequence
        try:
            return next(received)
        except StopIteration:
            assert clearance is not None
            sequence = command_one_sequence
            command_one_sequence += 1
            return event(sequence, "launch_intent", 1, clearance)

    def issue_clearance(*args):
        nonlocal clearance
        clearance = clearance_function(*args)
        return clearance

    server = mock.Mock()
    connection = mock.Mock()
    server.accept.return_value = (connection, None)
    evidence = type(
        "Evidence",
        (),
        {
            "snapshot_sha256": "b" * 64,
            "validated_artifacts": (),
            "sealed_artifacts": (),
        },
    )()
    receipt = mock.Mock()
    with (
        mock.patch.object(
            authority, "_context", return_value=(checkpoint, envelope, request, commands)
        ),
        mock.patch.object(authority.policy, "secure_private_key", return_value=private),
        mock.patch.object(authority.policy, "validate_prelaunch"),
        mock.patch.object(authority, "_open_root", return_value=root_fd),
        mock.patch.object(authority, "_bind_server", return_value=(server, (1, 2))),
        mock.patch.object(authority, "peer_credentials", return_value=(1, os.getuid(), 3)),
        mock.patch.object(authority, "_challenge", return_value={"challenge": True}),
        mock.patch.object(authority, "_validate_proof", return_value="claim"),
        mock.patch.object(authority, "_authorization", return_value=authorization),
        mock.patch.object(authority.policy, "receive_packet", side_effect=receive_packet),
        mock.patch.object(authority.policy, "send_packet") as send_packet,
        mock.patch.object(
            authority.policy, "validate_completed_command", return_value=evidence
        ) as validate_completed,
        mock.patch.object(authority, "_clearance", side_effect=issue_clearance) as issue,
        mock.patch.object(authority, "_receipt", receipt),
        mock.patch.object(authority, "_write_new") as write_new,
        mock.patch.object(authority, "_remove_socket"),
        mock.patch.object(authority.time, "time", return_value=epoch),
        mock.patch.object(authority, "_utc", return_value=observed_utc),
        pytest.raises(
            authority.policy.TerminalPolicyError, match="replayed or out-of-order launch intent"
        ),
    ):
        authority.serve(argparse.Namespace(private_key=Path("/key")))

    assert validate_completed.call_count == 3
    assert all(
        call.args == (envelope, request, 0, None) for call in validate_completed.call_args_list
    )
    issue.assert_called_once()
    receipt.assert_not_called()
    write_new.assert_not_called()
    assert send_packet.call_count == 3


def test_write_new_removes_failed_publication_without_deleting_replacement(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    receipt = root / "receipt"

    def replace_then_fail(_fd, _data):
        receipt.unlink()
        receipt.write_bytes(b"replacement")
        raise OSError("write failed")

    try:
        with (
            mock.patch.object(authority.os, "write", side_effect=replace_then_fail),
            pytest.raises(BaseExceptionGroup) as raised,
        ):
            authority._write_new(fd, "receipt", b"receipt\n")
    finally:
        os.close(fd)

    assert str(raised.value.exceptions[0]) == "write failed"
    assert receipt.read_bytes() == b"replacement"


def test_serve_preserves_primary_and_cleanup_failures_and_closes_root_fd(authority):
    server = mock.Mock()
    server.accept.side_effect = OSError("accept failed")
    server.close.side_effect = OSError("server close failed")
    private = mock.Mock()
    private.public_key.return_value = object()
    request = type(
        "Request",
        (),
        {
            "authority_socket": "/evidence/authority.sock",
            "publication": type("Publication", (), {"receipt": "receipt"})(),
        },
    )()
    envelope = type(
        "Envelope", (), {"authority_key": type("AuthorityKey", (), {"key_id": "authority"})()}
    )()
    closed_fds: list[int] = []
    with (
        mock.patch.object(authority, "_context", return_value=(object(), envelope, request, ())),
        mock.patch.object(authority.policy, "secure_private_key", return_value=private),
        mock.patch.object(authority.policy, "public_key_id", return_value="authority"),
        mock.patch.object(authority, "_open_root", return_value=73),
        mock.patch.object(authority.policy, "validate_prelaunch"),
        mock.patch.object(authority, "_bind_server", return_value=(server, (1, 2))),
        mock.patch.object(authority, "_remove_socket", side_effect=OSError("remove failed")),
        mock.patch.object(authority.os, "close", side_effect=closed_fds.append),
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        authority.serve(argparse.Namespace(private_key=Path("/key")))

    primary_error, cleanup_error = raised.value.exceptions
    assert str(primary_error) == "accept failed"
    assert raised.value.__cause__ is primary_error
    assert isinstance(cleanup_error, BaseExceptionGroup)
    assert [str(error) for error in cleanup_error.exceptions] == [
        "server close failed",
        "remove failed",
    ]
    assert closed_fds == [73]


def test_serve_preserves_body_connection_and_all_cleanup_failures(authority):
    server = mock.Mock()
    connection = mock.Mock()
    server.accept.return_value = (connection, None)
    connection.close.side_effect = OSError("connection close failed")
    server.close.side_effect = OSError("server close failed")
    private = mock.Mock()
    private.public_key.return_value = object()
    request = type(
        "Request",
        (),
        {
            "authority_socket": "/evidence/authority.sock",
            "publication": type("Publication", (), {"receipt": "receipt"})(),
        },
    )()
    envelope = type(
        "Envelope", (), {"authority_key": type("AuthorityKey", (), {"key_id": "authority"})()}
    )()
    with (
        mock.patch.object(authority, "_context", return_value=(object(), envelope, request, ())),
        mock.patch.object(authority.policy, "secure_private_key", return_value=private),
        mock.patch.object(authority.policy, "public_key_id", return_value="authority"),
        mock.patch.object(authority, "_open_root", return_value=73),
        mock.patch.object(authority.policy, "validate_prelaunch"),
        mock.patch.object(authority, "_bind_server", return_value=(server, (1, 2))),
        mock.patch.object(authority, "peer_credentials", side_effect=ValueError("body failed")),
        mock.patch.object(authority, "_remove_socket", side_effect=OSError("remove failed")),
        mock.patch.object(authority.os, "close", side_effect=OSError("root close failed")),
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        authority.serve(argparse.Namespace(private_key=Path("/key")))

    primary, cleanup = raised.value.exceptions
    assert str(primary) == "body failed"
    assert [str(error) for error in cleanup.exceptions] == [
        "connection close failed",
        "server close failed",
        "remove failed",
        "root close failed",
    ]


def test_recover_authorization_only_never_acquires_serving_or_launch_capabilities(authority):
    private = mock.Mock()
    private.public_key.return_value = object()
    request = type(
        "Request",
        (),
        {"publication": type("Publication", (), {"receipt": "receipt"})()},
    )()
    envelope = type(
        "Envelope", (), {"authority_key": type("AuthorityKey", (), {"key_id": "authority"})()}
    )()
    authorization = {"authorization_id": "authorization"}
    expected_state = {
        "kind": "UNAUTHENTICATED_TERMINAL",
        "last_authenticated_sequence": 0,
    }
    expected_receipt = {
        "terminal_state": expected_state,
        "events_sha256": authority._sha(authority.policy.canonical_bytes([])),
        "target_results": [],
    }
    with (
        mock.patch.object(
            authority, "_context", return_value=(object(), envelope, request, (("command",),))
        ),
        mock.patch.object(authority.policy, "secure_private_key", return_value=private),
        mock.patch.object(authority.policy, "public_key_id", return_value="authority"),
        mock.patch.object(authority, "_open_root", return_value=73),
        mock.patch.object(authority.policy, "validate_prelaunch"),
        mock.patch.object(authority, "_reconcile_pending", return_value=[authorization]),
        mock.patch.object(authority, "_validate_stored_authorization"),
        mock.patch.object(authority.policy, "validate_scope_artifacts"),
        mock.patch.object(authority, "_receipt", return_value=expected_receipt),
        mock.patch.object(authority, "_write_new") as write_new,
        mock.patch.object(authority.os, "close") as close_root,
        mock.patch.object(authority, "serve", side_effect=AssertionError("serve called")) as serve,
        mock.patch.object(
            authority, "_bind_server", side_effect=AssertionError("bind/listen called")
        ) as bind_server,
        mock.patch.object(
            authority.socket, "socket", side_effect=AssertionError("socket/accept called")
        ) as socket_factory,
        mock.patch.object(
            authority.os, "system", side_effect=AssertionError("system called")
        ) as system,
        mock.patch.object(
            authority.os, "execve", side_effect=AssertionError("exec called")
        ) as execve,
        mock.patch.object(
            authority.os, "posix_spawn", side_effect=AssertionError("spawn called")
        ) as posix_spawn,
    ):
        assert authority.recover(argparse.Namespace(private_key=Path("/key"))) == 2

    close_root.assert_called_once_with(73)
    write_new.assert_called_once_with(
        73,
        "receipt",
        authority.policy.canonical_bytes(expected_receipt),
    )
    serve.assert_not_called()
    bind_server.assert_not_called()
    socket_factory.assert_not_called()
    system.assert_not_called()
    execve.assert_not_called()
    posix_spawn.assert_not_called()


def test_recovery_preserves_pending_until_durable_append_and_never_serves(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        pending = authority.policy.canonical_bytes(
            {"authorization_id": "authorization", "type": "authorization"}
        )
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


def test_reconcile_pending_rejects_replacement_before_journal_append(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    pending_record = {"authorization_id": "authorization", "type": "authorization"}
    pending = authority.policy.canonical_bytes(pending_record)
    replacement = root / "replacement"
    try:
        authority._write_new(fd, authority.PENDING_NAME, pending)

        def replace_pending(*_args):
            replacement.write_bytes(b"attacker\n")
            replacement.chmod(0o600)
            os.replace(replacement, root / authority.PENDING_NAME)

        with (
            mock.patch.object(
                authority, "_validate_stored_authorization", side_effect=replace_pending
            ) as validate,
            pytest.raises(ValueError, match="pending journal identity changed"),
        ):
            authority._reconcile_pending(fd, None, None, _recovery_request(), None)

        validate.assert_called_once()
        assert not (root / "journal").exists()
        assert not (root / "receipt").exists()
        assert (root / authority.PENDING_NAME).read_bytes() == b"attacker\n"
    finally:
        os.close(fd)


def test_reconcile_pending_repairs_only_strict_prefix_tails_durably(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    authorization = {
        "authorization_id": "authorization",
        "type": "authorization",
    }
    pending = authority.policy.canonical_bytes(authorization)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, "journal", pending[:9])
        authority._write_new(fd, authority.PENDING_NAME, pending)
        with (
            mock.patch.object(authority, "_validate_stored_authorization"),
            mock.patch.object(authority.os, "fsync", wraps=os.fsync) as fsync,
        ):
            records = authority._reconcile_pending(fd, None, None, _recovery_request(), None)
        assert records == [authorization]
        assert (root / "journal").read_bytes() == pending
        assert not (root / authority.PENDING_NAME).exists()
        assert fsync.call_count >= 4
    finally:
        os.close(fd)


def test_reconcile_pending_only_appends_validated_record(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    pending_record = {"authorization_id": "authorization", "type": "authorization"}
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


def test_reconcile_pending_fully_appended_authorization_is_idempotent(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    authorization = {
        "authorization_id": "authorization",
        "type": "authorization",
    }
    pending = authority.policy.canonical_bytes(authorization)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, "journal", pending)
        authority._write_new(fd, authority.PENDING_NAME, pending)
        with mock.patch.object(authority, "_validate_stored_authorization") as validate:
            assert authority._reconcile_pending(fd, None, None, _recovery_request(), None) == [
                authorization
            ]
        validate.assert_called_once()
        assert not (root / authority.PENDING_NAME).exists()
        assert (root / "journal").read_bytes() == pending
    finally:
        os.close(fd)


@pytest.mark.parametrize("strict_prefix", (False, True))
def test_reconcile_pending_rejects_intent_history_without_mutating_publications(
    authority, tmp_path, strict_prefix
):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    authorization = authority.policy.canonical_bytes(
        {"authorization_id": "authorization", "type": "authorization"}
    )
    pending = authority.policy.canonical_bytes(
        {"authorization_id": "authorization", "event": "launch_intent", "type": "process_event"}
    )
    journal = authorization + (pending[:9] if strict_prefix else pending)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        authority._write_new(fd, "journal", journal)
        authority._write_new(fd, authority.PENDING_NAME, pending)
        before = {
            name: ((root / name).read_bytes(), (root / name).stat().st_ino)
            for name in ("journal", authority.PENDING_NAME)
        }
        request = _recovery_request()
        with pytest.raises(ValueError, match="authorization-only history"):
            authority._reconcile_pending(fd, None, None, request, None)
        after = {
            name: ((root / name).read_bytes(), (root / name).stat().st_ino)
            for name in ("journal", authority.PENDING_NAME)
        }
        assert after == before
        assert not (root / "receipt").exists()
    finally:
        os.close(fd)


def test_reconcile_pending_preserves_nonprefix_tail_and_pending(authority, tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    pending = authority.policy.canonical_bytes(
        {"authorization_id": "authorization", "type": "authorization"}
    )
    journal = (
        authority.policy.canonical_bytes(
            {"authorization_id": "authorization", "type": "authorization"}
        )
        + b"wrong"
    )
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
    pending = authority.policy.canonical_bytes(
        {"authorization_id": "authorization", "type": "authorization"}
    )
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


def _runtime_file(path: Path) -> dict[str, int | str]:
    info = path.stat()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": info.st_size,
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
    }


def _runtime_directory(path: Path) -> dict[str, int | str]:
    info = path.stat()
    return {
        "path": str(path),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
    }


def _accepted_blob(path: str, expected_sha256: str) -> bytes:
    data = subprocess.check_output(
        ("git", "show", f"{ACCEPTED_ALPHA_COMMIT}:{path}"),
        cwd=ROOT,
    )
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise AssertionError(f"accepted blob digest mismatch: {path}")
    return data


def _runtime_recovery_context(authority, tmp_path):
    """Build a canonical recovery context from current controls and pinned local objects."""
    external = load(EXTERNAL_ENVELOPE_TEST_PATH, "alpha_max_terminal_external_fixture")
    bound = tmp_path / "bound"
    bound.mkdir(mode=0o700)
    alignment = bound / "alignment_receipt"
    alignment.write_bytes(b"runtime alignment receipt\n")
    alignment.chmod(0o600)
    policy_value = json.loads(POLICY_PATH.read_text())
    policy_value["pins"]["alignment_receipt_sha256"] = hashlib.sha256(
        alignment.read_bytes()
    ).hexdigest()
    policy_path = tmp_path / "policy.json"
    external.write_canonical(policy_path, policy_value)
    policy = authority.policy.load_policy(policy_path)
    pin_sources = {
        "runbook": "docs/research_note/alpha_max_data_pc_runbook_20260711.md",
        "alpha_uv_lock": "uv.lock",
        "portfolio": "configs/research/alpha_max_portfolio_20260711_listing_aware.json",
        "contract_manifest": "configs/research/alpha_max_contract_manifest_20260711_listing_aware.json",
        "availability_evidence": "configs/research/alpha_max_official_availability_evidence_20260711.json",
        "preparer": "scripts/research/prepare_alpha_max_phase_roots.py",
        "prelock_script": "scripts/research/run_alpha_max_prelock.py",
        "historical_script": "scripts/research/run_alpha_max_historical_evaluation.py",
        "process_boundary": "src/lumina_quant/alpha_max_process_boundary.py",
    }
    current_sources = {
        "policy_json": policy_path,
        "policy_module": ROOT / "src/lumina_quant/alpha_max_terminal_policy.py",
        "authority_script": AUTHORITY_PATH,
        "observer_script": OBSERVER_PATH,
        "key_creator": KEYS_PATH,
        "acquirer": ROOT / "scripts/research/acquire_alpha_max_official_source.py",
        "phase_wrapper": (
            ROOT / "scripts/research/run_alpha_max_phase_preparation_from_eligible_source.py"
        ),
    }
    pin_names = {
        "alpha_uv_lock": "uv_lock_sha256",
        "alignment_receipt": "alignment_receipt_sha256",
        "contract_manifest": "contract_sha256",
        "availability_evidence": "availability_sha256",
        "prelock_script": "prelock_sha256",
        "historical_script": "historical_sha256",
    }
    materialized: dict[str, Path] = {"alignment_receipt": alignment}
    for role, source in pin_sources.items():
        path = bound / role
        path.write_bytes(_accepted_blob(source, policy.pins[pin_names.get(role, f"{role}_sha256")]))
        path.chmod(0o600)
        materialized[role] = path
    value = external._envelope_value(tmp_path, policy)
    authority_private = Ed25519PrivateKey.generate()
    observer_private = Ed25519PrivateKey.generate()
    authority_raw = authority_private.public_key().public_bytes(
        authority.policy.serialization.Encoding.Raw, authority.policy.serialization.PublicFormat.Raw
    )
    authority_key_id = authority.policy.public_key_id(authority_private.public_key())
    value["authority_key"] = {
        "key_id": authority_key_id,
        "public_key_b64": base64.b64encode(authority_raw).decode("ascii"),
        "public_key_sha256": authority_key_id,
    }
    observer_keys = []
    for scope in external.TEST_SCOPES:
        private = observer_private if scope == "acquisition" else Ed25519PrivateKey.generate()
        public = private.public_key()
        raw = public.public_bytes(
            authority.policy.serialization.Encoding.Raw,
            authority.policy.serialization.PublicFormat.Raw,
        )
        key_id = authority.policy.public_key_id(public)
        observer_keys.append(
            {
                "scope": scope,
                "key_id": key_id,
                "public_key_b64": base64.b64encode(raw).decode("ascii"),
                "public_key_sha256": key_id,
            }
        )
    value["observer_keys"] = observer_keys
    for item in value["files"]:
        role = item["role"]
        item["file"] = _runtime_file(
            materialized[role] if role in materialized else current_sources[role]
        )
    for repository in value["repositories"]:
        receipt = tmp_path / f"{repository['role']}.receipt"
        receipt.write_bytes(b"clean\n")
        receipt.chmod(0o600)
        repository["root"] = _runtime_directory(Path(repository["root"]["path"]))
        repository["clean_receipt"] = _runtime_file(receipt)
    for item in value["interpreters"]:
        role = item["role"]
        interpreter = bound / f"{role}.python"
        interpreter.write_bytes(Path(sys.executable).read_bytes())
        interpreter.chmod(0o700)
        freeze = bound / f"{role}.freeze"
        freeze.write_bytes((ROOT / "uv.lock").read_bytes())
        freeze.chmod(0o600)
        item["file"] = _runtime_file(interpreter)
        item["package_freeze"] = _runtime_file(freeze)
    envelope_path = tmp_path / "envelope.json"
    external.write_canonical(envelope_path, value)
    checkpoint_value = {
        "schema": "alpha_max_terminal_checkpoint.v1",
        "accepted_alpha_commit": policy.accepted_alpha_commit,
        "baseline_ancestor": policy.baseline_ancestor,
        **policy.pins,
        "authority_manifest_sha256": hashlib.sha256(envelope_path.read_bytes()).hexdigest(),
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    external.write_canonical(checkpoint_path, checkpoint_value)
    checkpoint = authority.policy.load_checkpoint(checkpoint_path, policy)
    envelope = authority.policy.load_envelope(envelope_path, policy, checkpoint)
    request_value = external._request_value(tmp_path, checkpoint, envelope, "acquisition")
    request_path = tmp_path / "request.json"
    external.write_canonical(request_path, request_value)
    request = authority.policy.load_request(
        request_path, scope="acquisition", policy=policy, checkpoint=checkpoint, envelope=envelope
    )
    private_path = tmp_path / "authority.private"
    private_path.write_bytes(
        authority_private.private_bytes(
            authority.policy.serialization.Encoding.Raw,
            authority.policy.serialization.PrivateFormat.Raw,
            authority.policy.serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o400)
    return (
        policy,
        checkpoint,
        envelope,
        request,
        authority_private,
        observer_private,
        (
            policy_path,
            checkpoint_path,
            envelope_path,
            request_path,
            private_path,
        ),
    )


@pytest.mark.parametrize("tamper", (None, "signature", "context", "journal"))
def test_runtime_recovery_uses_canonical_context_and_prelaunch(authority, tmp_path, tamper):
    _policy, checkpoint, envelope, request, private, _observer_private, paths = (
        _runtime_recovery_context(authority, tmp_path)
    )
    policy = authority.policy
    claim = {
        "schema": policy.CLAIM_SCHEMA,
        "request_id": request.request_id,
        "scope": request.scope,
        "checkpoint_pin_sha256": checkpoint.sha256,
        "evidence_root": policy.plain(request.evidence_root),
        "observer_pid": os.getpid(),
        "observer_uid": os.getuid(),
        "observer_start_ticks": authority._ticks(os.getpid()),
        "created_utc": "2026-07-22T00:00:00Z",
    }
    claim_bytes = policy.canonical_bytes(claim)
    root = Path(request.evidence_root.path)
    (root / request.publication.claim).write_bytes(claim_bytes)
    (root / request.publication.claim).chmod(0o600)
    commands = policy.derive_scope_commands(envelope, request)
    authorization = policy.sign_message(
        "authorization",
        {
            "schema": authority.SCHEMA,
            "type": "authorization",
            "authority_key_id": envelope.authority_key.key_id,
            "authorization_id": "9" * 64,
            "scope": request.scope,
            "request_id": request.request_id,
            "checkpoint_pin_sha256": checkpoint.sha256,
            "envelope_sha256": envelope.sha256,
            "request_sha256": request.sha256,
            "command_bundle_sha256": policy.command_bundle_sha256(request, commands),
            "claim_sha256": hashlib.sha256(claim_bytes).hexdigest(),
            "observer_key_id": envelope.observer_key("acquisition").key_id,
            "observer_pid": claim["observer_pid"],
            "observer_uid": claim["observer_uid"],
            "observer_start_ticks": claim["observer_start_ticks"],
            "observer_source_sha256": next(
                item.file.sha256 for item in envelope.files if item.role == "observer_script"
            ),
            "not_before_utc": "2026-07-22T00:00:00Z",
            "expires_utc": "2026-07-22T00:04:00Z",
        },
        private,
    )
    (root / request.publication.journal).write_bytes(policy.canonical_bytes(authorization))
    (root / request.publication.journal).chmod(0o600)
    args = authority.parser().parse_args(
        [
            "recover",
            "--policy",
            str(paths[0]),
            "--checkpoint",
            str(paths[1]),
            "--envelope",
            str(paths[2]),
            "--request",
            str(paths[3]),
            "--private-key",
            str(paths[4]),
            "--evidence-root",
            request.evidence_root.path,
            "--scope",
            "acquisition",
        ]
    )
    target = tmp_path / "target-canary"
    target.write_bytes(b"must not execute")
    before = target.stat().st_mtime_ns, target.read_bytes()
    with mock.patch.object(
        subprocess, "Popen", side_effect=AssertionError("recovery attempted process creation")
    ) as popen:
        assert authority.recover(args) == 2
    popen.assert_not_called()
    receipt_path = root / request.publication.receipt
    public_path = tmp_path / "authority.public"
    public_path.write_bytes(
        private.public_key().public_bytes(
            authority.policy.serialization.Encoding.Raw,
            authority.policy.serialization.PublicFormat.Raw,
        )
    )
    public_path.chmod(0o444)
    verified = policy.verify_signed_receipt(receipt_path, public_path)
    assert verified.message["terminal_state"] == {
        "kind": "UNAUTHENTICATED_TERMINAL",
        "last_authenticated_sequence": 0,
    }
    assert verified.message["target_results"] == []
    assert verified.events == ()
    assert (target.stat().st_mtime_ns, target.read_bytes()) == before
    assert not any(
        (root / name).exists()
        for name in (*request.publication.stdout, *request.publication.stderr)
    )
    assert not Path(request.records.source_root.path).exists()
    assert not Path(request.records.report_root.path).exists()
    if tamper is None:
        return
    receipt = policy.parse_canonical_object(receipt_path.read_bytes(), "terminal receipt")
    if tamper == "signature":
        receipt["authority_signature_b64"] = base64.b64encode(b"\0" * 64).decode("ascii")
        receipt_path.write_bytes(policy.canonical_bytes(receipt))
    elif tamper == "context":
        unsigned = {
            key: value for key, value in receipt.items() if key != "authority_signature_b64"
        }
        unsigned["request_id"] = "f" * 64
        receipt_path.write_bytes(
            policy.canonical_bytes(policy.sign_message("terminal_receipt", unsigned, private))
        )
    else:
        journal_path = root / request.publication.journal
        journal_path.write_bytes(
            journal_path.read_bytes() + policy.canonical_bytes({"tampered": True})
        )
    with pytest.raises(policy.TerminalPolicyError):
        policy.verify_signed_receipt(receipt_path, public_path)


def _signed_recovery_history(authority, tmp_path, *, intent_history=True):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    root_info = root.stat()
    root_identity = authority.policy.DirectoryIdentity(
        str(root),
        root_info.st_dev,
        root_info.st_ino,
        root_info.st_uid,
        root_info.st_gid,
        stat.S_IMODE(root_info.st_mode),
    )
    authority_private = Ed25519PrivateKey.generate()
    observer_private = Ed25519PrivateKey.generate()
    authority_raw = authority_private.public_key().public_bytes(
        authority.policy.serialization.Encoding.Raw,
        authority.policy.serialization.PublicFormat.Raw,
    )
    observer_raw = observer_private.public_key().public_bytes(
        authority.policy.serialization.Encoding.Raw,
        authority.policy.serialization.PublicFormat.Raw,
    )
    authority_key_id = authority.policy.public_key_id(authority_private.public_key())
    observer_key_id = authority.policy.public_key_id(observer_private.public_key())
    observer_source_sha256 = "d" * 64
    file_identity = authority.policy.FileIdentity(
        "/fixture",
        "e" * 64,
        1,
        1,
        1,
        os.getuid(),
        os.getgid(),
        0o600,
        1,
    )
    envelope = authority.policy.LaunchEnvelope(
        "fixture",
        "f" * 64,
        "0" * 40,
        "1" * 40,
        "2" * 40,
        (),
        (
            authority.policy.FileBinding(
                "observer_script",
                authority.policy.FileIdentity(
                    "/observer", observer_source_sha256, 1, 1, 2, os.getuid(), os.getgid(), 0o600, 1
                ),
            ),
        ),
        (),
        authority.policy.KeyBinding(
            authority_key_id,
            base64.b64encode(authority_raw).decode(),
            authority_key_id,
        ),
        (
            authority.policy.ObserverKeyBinding(
                "acquisition",
                observer_key_id,
                base64.b64encode(observer_raw).decode(),
                observer_key_id,
            ),
        ),
        (),
        ("acquisition",),
    )
    publication = authority.policy.PublicationPaths(
        "prelaunch.claim.json",
        "terminal-observer.journal.jsonl",
        ("child-0.stdout.log", "child-1.stdout.log"),
        ("child-0.stderr.log", "child-1.stderr.log"),
        "terminal-authority.receipt.json",
    )
    prerequisites = []
    for kind, name in (
        ("checkpoint_pin", "checkpoint-pin.json"),
        ("alignment_receipt", "alignment-receipt.json"),
    ):
        path = tmp_path / name
        path.write_bytes(f"{kind}\n".encode())
        path.chmod(0o444)
        info = path.stat()
        prerequisites.append(
            authority.policy.ValidatedArtifact(
                kind,
                str(path),
                hashlib.sha256(path.read_bytes()).hexdigest(),
                info.st_size,
                info.st_dev,
                info.st_ino,
                stat.S_IMODE(info.st_mode),
                info.st_nlink,
            )
        )
    request = authority.policy.ScopeRequest(
        "fixture",
        "a" * 64,
        "acquisition",
        "b" * 64,
        file_identity,
        root_identity,
        root_identity,
        str(root / "authority.sock"),
        authority.policy.Environment("/empty", "C", "C", "/usr/bin", "0", "1", "1", "UTC"),
        ("/forbidden-a", "/forbidden-b"),
        publication,
        tuple(prerequisites),
        authority.policy.AcquisitionRecords(
            file_identity,
            file_identity,
            file_identity,
            authority.policy.AbsentOutput(str(root / "output"), root_identity, "output", True),
            authority.policy.AbsentOutput(str(root / "report"), root_identity, "report", True),
        ),
    )
    checkpoint = type("Checkpoint", (), {"sha256": "c" * 64})()
    claim = {
        "schema": authority.policy.CLAIM_SCHEMA,
        "request_id": request.request_id,
        "scope": request.scope,
        "checkpoint_pin_sha256": checkpoint.sha256,
        "evidence_root": authority.policy.plain(root_identity),
        "observer_pid": 17,
        "observer_uid": os.getuid(),
        "observer_start_ticks": 19,
        "created_utc": "2026-07-22T00:00:00Z",
    }
    claim_bytes = authority.policy.canonical_bytes(claim)
    (root / publication.claim).write_bytes(claim_bytes)
    (root / publication.claim).chmod(0o600)
    commands = authority.policy.derive_scope_commands(envelope, request)
    authorization = authority.policy.sign_message(
        "authorization",
        {
            "schema": authority.SCHEMA,
            "type": "authorization",
            "authority_key_id": authority_key_id,
            "authorization_id": "9" * 64,
            "scope": request.scope,
            "request_id": request.request_id,
            "checkpoint_pin_sha256": checkpoint.sha256,
            "envelope_sha256": envelope.sha256,
            "request_sha256": request.sha256,
            "command_bundle_sha256": authority.policy.command_bundle_sha256(request, commands),
            "claim_sha256": hashlib.sha256(claim_bytes).hexdigest(),
            "observer_key_id": observer_key_id,
            "observer_pid": 17,
            "observer_uid": os.getuid(),
            "observer_start_ticks": 19,
            "observer_source_sha256": observer_source_sha256,
            "not_before_utc": "2026-07-22T00:00:00Z",
            "expires_utc": "2026-07-22T00:04:00Z",
        },
        authority_private,
    )

    def event(sequence, kind, **extra):
        return authority.policy.sign_message(
            "process_event",
            {
                "schema": authority.SCHEMA,
                "type": "process_event",
                "event": kind,
                "authorization_id": authorization["authorization_id"],
                "sequence": sequence,
                "command_index": 0,
                "argv_sha256": hashlib.sha256(
                    authority.policy.canonical_bytes(commands[0])
                ).hexdigest(),
                "environment_sha256": hashlib.sha256(
                    authority.policy.canonical_bytes(request.environment)
                ).hexdigest(),
                "prior_clearance": None,
                "observed_utc": "2026-07-22T00:00:01Z",
                **extra,
            },
            observer_private,
        )

    if intent_history:
        intent = event(0, "launch_intent")
        start_failed = event(1, "start_failed", errno=5, error_name="EIO")
        (root / publication.journal).write_bytes(
            authority.policy.canonical_bytes(authorization)
            + authority.policy.canonical_bytes(intent)
        )
        (root / authority.PENDING_NAME).write_bytes(authority.policy.canonical_bytes(start_failed))
        (root / authority.PENDING_NAME).chmod(0o600)
    else:
        (root / publication.journal).write_bytes(authority.policy.canonical_bytes(authorization))
    (root / publication.journal).chmod(0o600)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    return (
        checkpoint,
        envelope,
        request,
        commands,
        authority_private,
        observer_private,
        authorization,
        root,
        root_fd,
    )


def test_recover_publishes_exact_signed_authorization_only_receipt(authority, tmp_path):
    (
        checkpoint,
        envelope,
        request,
        commands,
        private,
        _observer_private,
        authorization,
        root,
        root_fd,
    ) = _signed_recovery_history(authority, tmp_path, intent_history=False)
    target = tmp_path / "target"
    target.write_bytes(b"target canary")
    target_before = (
        target.read_bytes(),
        target.stat().st_ino,
        target.stat().st_mtime_ns,
    )
    os.close(root_fd)
    write_new = authority._write_new
    lock_checked = False

    def publish_under_recovery_lock(fd, name, data):
        nonlocal lock_checked
        with pytest.raises(ValueError, match="terminal evidence root is active"):
            authority._open_root(request)
        lock_checked = True
        write_new(fd, name, data)

    with (
        mock.patch.object(
            authority,
            "_context",
            return_value=(checkpoint, envelope, request, commands),
        ),
        mock.patch.object(
            authority.policy,
            "secure_private_key",
            return_value=private,
        ),
        mock.patch.object(
            authority,
            "_write_new",
            side_effect=publish_under_recovery_lock,
        ),
        mock.patch.object(authority.policy, "validate_prelaunch"),
    ):
        assert authority.recover(argparse.Namespace(private_key=Path("/key"))) == 2
    assert lock_checked

    receipt = authority.policy.parse_canonical_object(
        (root / request.publication.receipt).read_bytes(),
        "terminal receipt",
    )
    authority.policy.verify_message(
        "terminal_receipt",
        receipt,
        private.public_key(),
    )
    public_key_path = tmp_path / "authority-public.raw"
    public_key_path.write_bytes(
        private.public_key().public_bytes(
            authority.policy.serialization.Encoding.Raw,
            authority.policy.serialization.PublicFormat.Raw,
        )
    )
    public_key_path.chmod(0o444)
    assert tuple(inspect.signature(authority.policy.verify_signed_receipt).parameters) == (
        "path",
        "public_key_path",
    )
    verified = authority.policy.verify_signed_receipt(
        root / request.publication.receipt,
        public_key_path,
    )
    assert verified.message == receipt
    assert receipt["authorization_id"] == authorization["authorization_id"]
    assert receipt["terminal_state"] == {
        "kind": "UNAUTHENTICATED_TERMINAL",
        "last_authenticated_sequence": 0,
    }
    assert receipt["events_sha256"] == authority._sha(authority.policy.canonical_bytes([]))
    assert receipt["target_results"] == []
    assert (root / request.publication.journal).read_bytes() == (
        authority.policy.canonical_bytes(authorization)
    )
    assert not (root / authority.PENDING_NAME).exists()
    assert (
        target.read_bytes(),
        target.stat().st_ino,
        target.stat().st_mtime_ns,
    ) == target_before
    assert not any((root / name).exists() for name in ("stdout", "stderr", "output", "report"))


def test_reconcile_pending_rejects_runtime_signed_history_injected_after_validation(
    authority, tmp_path
):
    (
        checkpoint,
        envelope,
        request,
        commands,
        _private,
        observer_private,
        authorization,
        root,
        root_fd,
    ) = _signed_recovery_history(authority, tmp_path, intent_history=False)
    authorization_bytes = authority.policy.canonical_bytes(authorization)
    authority._write_new(root_fd, authority.PENDING_NAME, authorization_bytes)
    intent = authority.policy.sign_message(
        "process_event",
        {
            "schema": authority.SCHEMA,
            "type": "process_event",
            "event": "launch_intent",
            "authorization_id": authorization["authorization_id"],
            "sequence": 0,
            "command_index": 0,
            "argv_sha256": hashlib.sha256(
                authority.policy.canonical_bytes(commands[0])
            ).hexdigest(),
            "environment_sha256": hashlib.sha256(
                authority.policy.canonical_bytes(request.environment)
            ).hexdigest(),
            "prior_clearance": None,
            "observed_utc": "2026-07-22T00:00:01Z",
        },
        observer_private,
    )
    injected = authorization_bytes + authority.policy.canonical_bytes(intent)
    replacement = root / "journal-replacement"
    replacement.write_bytes(injected)
    replacement.chmod(0o600)
    validate = authority._validate_stored_authorization

    def inject_after_validation(*args):
        validate(*args)
        os.replace(replacement, root / request.publication.journal)

    try:
        with (
            mock.patch.object(
                authority,
                "_validate_stored_authorization",
                side_effect=inject_after_validation,
            ),
            pytest.raises(
                ValueError,
                match="terminal journal changed before recovery mutation",
            ),
        ):
            authority._reconcile_pending(
                root_fd,
                checkpoint,
                envelope,
                request,
                commands,
            )
        assert (root / request.publication.journal).read_bytes() == injected
        assert (root / authority.PENDING_NAME).read_bytes() == authorization_bytes
        assert not (root / request.publication.receipt).exists()
    finally:
        os.close(root_fd)


def test_recover_rejects_signed_intent_history_without_mutating_it(authority, tmp_path):
    (
        checkpoint,
        envelope,
        request,
        commands,
        private,
        _observer_private,
        _authorization,
        root,
        root_fd,
    ) = _signed_recovery_history(authority, tmp_path)
    target = tmp_path / "target"
    target.write_bytes(b"target canary")
    for name in ("stdout", "stderr", "output", "report"):
        path = root / name
        path.write_bytes(f"{name} canary".encode())
        path.chmod(0o600)
    canaries = (
        target,
        root / request.publication.journal,
        root / authority.PENDING_NAME,
        root / "stdout",
        root / "stderr",
        root / "output",
        root / "report",
    )
    before = {
        str(path): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        for path in canaries
    }
    root_before = (root.stat().st_ino, root.stat().st_mtime_ns)
    with (
        mock.patch.object(
            authority, "_context", return_value=(checkpoint, envelope, request, commands)
        ),
        mock.patch.object(authority.policy, "secure_private_key", return_value=private),
        mock.patch.object(authority, "_open_root", return_value=root_fd),
        mock.patch.object(authority.policy, "validate_prelaunch"),
        mock.patch.object(authority, "serve", side_effect=AssertionError("serve called")) as serve,
        mock.patch.object(
            authority, "_bind_server", side_effect=AssertionError("bind/listen called")
        ) as bind_server,
        mock.patch.object(
            authority.socket, "socket", side_effect=AssertionError("socket called")
        ) as socket_factory,
        mock.patch.object(
            authority.os, "system", side_effect=AssertionError("system called")
        ) as system,
        mock.patch.object(
            authority.os, "execve", side_effect=AssertionError("exec called")
        ) as execve,
        mock.patch.object(
            authority.os, "posix_spawn", side_effect=AssertionError("spawn called")
        ) as posix_spawn,
        mock.patch.object(authority, "subprocess", create=True) as subprocess,
        pytest.raises(ValueError, match="authorization-only history"),
    ):
        subprocess.Popen.side_effect = AssertionError("Popen called")
        authority.recover(argparse.Namespace(private_key=Path("/key")))

    after = {
        str(path): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        for path in canaries
    }
    assert after == before
    assert (root.stat().st_ino, root.stat().st_mtime_ns) == root_before
    assert not (root / "receipt").exists()
    serve.assert_not_called()
    bind_server.assert_not_called()
    socket_factory.assert_not_called()
    system.assert_not_called()
    execve.assert_not_called()
    posix_spawn.assert_not_called()
    subprocess.Popen.assert_not_called()


@pytest.mark.parametrize("injection", ("final_validation", "final_publication"))
def test_final_receipt_failures_preserve_all_owned_cleanup_failures(authority, injection):
    server = mock.Mock()
    connection = mock.Mock()
    server.accept.return_value = (connection, None)
    connection.close.side_effect = OSError("connection close failed")
    server.close.side_effect = OSError("server close failed")
    private = mock.Mock()
    private.public_key.return_value = object()
    request = type(
        "Request",
        (),
        {
            "authority_socket": "/evidence/authority.sock",
            "publication": type("Publication", (), {"receipt": "receipt"})(),
        },
    )()
    envelope = type(
        "Envelope", (), {"authority_key": type("AuthorityKey", (), {"key_id": "authority"})()}
    )()
    event = {
        "event": "start_failed",
        "command_index": 0,
        "errno": 5,
        "authorization_id": "authorization",
    }
    receipt = mock.Mock(
        side_effect=OSError("final validation failed")
        if injection == "final_validation"
        else {"receipt": "final"}
    )
    publish = mock.Mock(
        side_effect=OSError("final publication failed")
        if injection == "final_publication"
        else None
    )
    connection.recvmsg.side_effect = (
        (authority.policy.packet_bytes({"proof": True}), [], 0, None),
        (authority.policy.packet_bytes(event), [], 0, None),
    )
    with (
        mock.patch.object(
            authority, "_context", return_value=(object(), envelope, request, (("cmd",),))
        ),
        mock.patch.object(authority.policy, "secure_private_key", return_value=private),
        mock.patch.object(authority.policy, "public_key_id", return_value="authority"),
        mock.patch.object(authority, "_open_root", return_value=73),
        mock.patch.object(authority.policy, "validate_prelaunch"),
        mock.patch.object(authority, "_bind_server", return_value=(server, (1, 2))),
        mock.patch.object(authority, "peer_credentials", return_value=(1, 2, 3)),
        mock.patch.object(authority, "_challenge", return_value={"challenge": True}),
        mock.patch.object(authority, "_validate_proof", return_value="claim"),
        mock.patch.object(
            authority, "_authorization", return_value={"authorization_id": "authorization"}
        ),
        mock.patch.object(authority.policy, "send_packet") as send_packet,
        mock.patch.object(authority.policy, "validate_scope_artifacts", return_value=()),
        mock.patch.object(authority, "_receipt", receipt),
        mock.patch.object(authority, "_write_new", publish),
        mock.patch.object(
            authority, "_remove_socket", side_effect=OSError("unlink failed")
        ) as unlink,
        mock.patch.object(
            authority.os, "close", side_effect=OSError("root close failed")
        ) as close_root,
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        authority.serve(argparse.Namespace(private_key=Path("/key")))

    primary, cleanup = raised.value.exceptions
    assert str(primary) == (
        "final validation failed" if injection == "final_validation" else "final publication failed"
    )
    assert [str(error) for error in cleanup.exceptions] == [
        "connection close failed",
        "server close failed",
        "unlink failed",
        "root close failed",
    ]
    receipt.assert_called_once()
    send_packet.assert_has_calls(
        (
            mock.call(connection, {"challenge": True}),
            mock.call(connection, {"authorization_id": "authorization"}),
        )
    )
    assert send_packet.call_count == 2
    assert connection.recvmsg.call_count == 2
    if injection == "final_publication":
        publish.assert_called_once()
    else:
        publish.assert_not_called()
    connection.close.assert_called_once()
    server.close.assert_called_once()
    unlink.assert_called_once_with(73, "authority.sock", (1, 2))
    close_root.assert_called_once_with(73)


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
    key_cleanup_error = cleanup_error.exceptions[0]
    assert isinstance(key_cleanup_error, BaseExceptionGroup)
    replacement_error = key_cleanup_error.exceptions[0]
    assert isinstance(replacement_error, ValueError)
    assert "identity changed during cleanup: authority.private" in str(replacement_error)
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


def test_key_creator_recovers_first_fstat_identity_for_cleanup(tmp_path):
    keys = load(KEYS_PATH, "alpha_max_terminal_keys_first_fstat_test")
    root = tmp_path / "keys"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    created: list[tuple[str, int, int]] = []
    original_fstat = keys.os.fstat
    calls = 0

    def fail_first_fstat(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("first fstat failed")
        return original_fstat(fd)

    try:
        with (
            mock.patch.object(keys.os, "fstat", side_effect=fail_first_fstat),
            pytest.raises(OSError, match="first fstat failed"),
        ):
            keys._create_file(root_fd, "authority.private", b"x", created)
        assert created and created[0][0] == "authority.private"
        keys._cleanup(root_fd, created)
    finally:
        os.close(root_fd)

    assert list(root.iterdir()) == []


def test_key_creator_aggregates_creation_cleanup_and_root_close_failures(tmp_path):
    keys = load(KEYS_PATH, "alpha_max_terminal_keys_root_close_test")
    root = tmp_path / "keys"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    root_fd = 73
    with (
        mock.patch.object(keys, "_open_secure_root", return_value=root_fd) as open_root,
        mock.patch.object(keys, "_preflight"),
        mock.patch.object(keys, "_create_file", side_effect=OSError("creation failed")),
        mock.patch.object(keys, "_cleanup", side_effect=OSError("cleanup failed")),
        mock.patch.object(keys.os, "close", side_effect=OSError("root close failed")) as close,
        pytest.raises(BaseExceptionGroup) as raised,
    ):
        keys.create_keys(root)
    open_root.assert_called_once_with(root)
    close.assert_called_once_with(root_fd)

    primary, cleanup = raised.value.exceptions
    assert str(primary) == "creation failed"
    assert [str(error) for error in cleanup.exceptions] == ["cleanup failed", "root close failed"]


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
