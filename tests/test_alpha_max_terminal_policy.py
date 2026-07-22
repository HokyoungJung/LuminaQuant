from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

import lumina_quant.alpha_max_terminal_policy as terminal_policy
from lumina_quant.alpha_max_terminal_policy import (
    TerminalPolicyError,
    canonical_bytes,
    load_policy,
    packet_bytes,
    parse_canonical_object,
    receive_packet,
    sign_message,
    signing_preimage,
    verify_message,
)


ROOT = Path(__file__).parents[1]
POLICY = ROOT / "configs/research/alpha_max_terminal_authority_policy_v1.json"
EXPECTED_V3_PINS = {
    "acquirer_sha256": "b440d79899a4ed60e18decfcd8bc2656d2de012189f03572a8be65f90cd24978",
    "alignment_receipt_sha256": "8687b52180502a11de9fbe317a19d00bb4492c464b3bf33d4eda2437683ca812",
    "availability_sha256": "214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719",
    "contract_sha256": "ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220",
    "historical_sha256": "951290033c7efd9b59ba5418e38d96fbdcf3885211915b29010b79ae545f3fb0",
    "phase_wrapper_sha256": "054163d23e8d2f1446b225e281472bcc563ac76f06aa47552cc5f3953b7c4dd9",
    "portfolio_sha256": "2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c",
    "prelock_sha256": "838d633ae34d44443dad4990a79f4d8caa95f7102ffe2a649ed341b1bed16ad0",
    "preparer_sha256": "ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac",
    "process_boundary_sha256": "f95e8e0d356ca36063a415a7b37919e72d9d1f47af7d2c447e228546fddfb94c",
    "runbook_sha256": "249694fb1513354d61f67552f5c1b9175382f3c2bf9f271ee64dc0358d3c663f",
    "uv_lock_sha256": "59d9de230be950761736c24e04af3456e229cf4aa077536167fb7e650a71c339",
}


def test_policy_has_accepted_head_and_full_v3_pins() -> None:
    policy = load_policy(POLICY)
    assert policy.accepted_alpha_commit == "391000b40717386765bfa39bd212d91c2e3be794"
    assert policy.pins == EXPECTED_V3_PINS
    assert POLICY.read_bytes() == canonical_bytes(json.loads(POLICY.read_text()))
    assert policy.source_sha256 == hashlib.sha256(POLICY.read_bytes()).hexdigest()


def test_canonical_bytes_and_signing_preimage_are_domain_separated() -> None:
    assert canonical_bytes({"b": 1, "a": "x"}) == b'{"a":"x","b":1}\n'
    assert (
        signing_preimage("receipt", {"a": 1})
        == b'luminaquant.alpha_max.terminal.v1/receipt\x00{"a":1}\n'
    )


def test_policy_rejects_noncanonical_json(tmp_path: Path) -> None:
    value = json.loads(POLICY.read_text())
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value, indent=2))
    with pytest.raises(TerminalPolicyError, match="non-canonical"):
        load_policy(path)


class _PacketSocket:
    def __init__(self, packet: bytes) -> None:
        self.packet = packet

    def recvmsg(self, size: int) -> tuple[bytes, list[object], int, object]:
        self.requested_size = size
        return self.packet, [], 0, None


def test_packet_size_boundary_is_enforced_for_construction_and_receive() -> None:
    payload = "x" * (terminal_policy.MAX_PACKET_BYTES - len(canonical_bytes({"payload": ""})))
    assert len(canonical_bytes({"payload": payload})) == terminal_policy.MAX_PACKET_BYTES

    packet = packet_bytes({"payload": payload})
    assert len(packet) == terminal_policy.MAX_PACKET_BYTES + 4
    socket = _PacketSocket(packet)
    assert receive_packet(socket) == {"payload": payload}
    assert socket.requested_size == terminal_policy.MAX_PACKET_BYTES + 5

    too_large = b"x" * (terminal_policy.MAX_PACKET_BYTES + 1)
    with pytest.raises(TerminalPolicyError, match="packet exceeds maximum size"):
        packet_bytes({"payload": too_large.decode("ascii")})
    with pytest.raises(TerminalPolicyError, match="invalid packet framing"):
        receive_packet(
            _PacketSocket((terminal_policy.MAX_PACKET_BYTES + 1).to_bytes(4, "big") + too_large)
        )


def test_packet_framing_and_canonical_json_reject_replay_shaped_input() -> None:
    body = canonical_bytes({"sequence": 0, "type": "challenge"})
    assert receive_packet(_PacketSocket(packet_bytes({"type": "challenge", "sequence": 0}))) == {
        "sequence": 0,
        "type": "challenge",
    }
    with pytest.raises(TerminalPolicyError, match="packet framing"):
        receive_packet(_PacketSocket(b"\x00\x00\x00\x01" + body))
    with pytest.raises(TerminalPolicyError, match="non-canonical"):
        parse_canonical_object(b'{"type":"challenge", "sequence":0}\n', "packet")
    with pytest.raises(TerminalPolicyError, match="duplicate key"):
        parse_canonical_object(b'{"sequence":0,"sequence":1}\n', "packet")


def test_signatures_are_bound_to_the_message_domain_and_unsigned_shape() -> None:
    private = terminal_policy.Ed25519PrivateKey.generate()
    message = sign_message("challenge", {"sequence": 0}, private)
    verify_message("challenge", message, private.public_key())
    with pytest.raises(TerminalPolicyError, match="invalid authorization signature"):
        verify_message("authorization", message, private.public_key())
    with pytest.raises(TerminalPolicyError, match="signature present"):
        sign_message("challenge", message, private)


def test_signed_failure_receipt_round_trips_real_claim_journal_and_artifacts(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    evidence.chmod(0o700)

    def create_file(name: str, payload: bytes, mode: int) -> Path:
        path = evidence / name
        path.write_bytes(payload)
        path.chmod(mode)
        return path

    def artifact(path: Path, kind: str) -> dict[str, object]:
        info = path.stat()
        payload = path.read_bytes()
        return {
            "kind": kind,
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "st_dev": info.st_dev,
            "st_ino": info.st_ino,
            "mode": stat.S_IMODE(info.st_mode),
            "nlink": info.st_nlink,
        }

    authority_private = terminal_policy.Ed25519PrivateKey.generate()
    authority_public = authority_private.public_key().public_bytes(
        terminal_policy.serialization.Encoding.Raw,
        terminal_policy.serialization.PublicFormat.Raw,
    )
    public_key = create_file("authority.public", authority_public, 0o444)
    authority_key_id = hashlib.sha256(authority_public).hexdigest()

    checkpoint = create_file("checkpoint.json", b"checkpoint\n", 0o600)
    alignment = create_file("alignment.json", b"alignment\n", 0o600)
    root_info = evidence.stat()
    request_id = "1" * 64
    checkpoint_sha256 = "2" * 64
    envelope_sha256 = "3" * 64
    request_sha256 = "4" * 64
    command_bundle_sha256 = "5" * 64
    observer_key_id = "6" * 64
    observer_pid = os.getpid()
    observer_start_ticks = 1
    claim = {
        "schema": terminal_policy.CLAIM_SCHEMA,
        "request_id": request_id,
        "scope": "acquisition",
        "checkpoint_pin_sha256": checkpoint_sha256,
        "evidence_root": {
            "path": str(evidence),
            "st_dev": root_info.st_dev,
            "st_ino": root_info.st_ino,
            "st_uid": root_info.st_uid,
            "st_gid": root_info.st_gid,
            "mode": stat.S_IMODE(root_info.st_mode),
        },
        "observer_pid": observer_pid,
        "observer_uid": os.getuid(),
        "observer_start_ticks": observer_start_ticks,
        "created_utc": "2026-07-22T00:00:00Z",
    }
    claim_path = create_file("prelaunch.claim.json", canonical_bytes(claim), 0o600)
    claim_sha256 = hashlib.sha256(claim_path.read_bytes()).hexdigest()
    authorization = sign_message(
        "authorization",
        {
            "schema": terminal_policy.WIRE_SCHEMA,
            "type": "authorization",
            "authority_key_id": authority_key_id,
            "authorization_id": "7" * 64,
            "scope": "acquisition",
            "request_id": request_id,
            "checkpoint_pin_sha256": checkpoint_sha256,
            "envelope_sha256": envelope_sha256,
            "request_sha256": request_sha256,
            "command_bundle_sha256": command_bundle_sha256,
            "claim_sha256": claim_sha256,
            "observer_key_id": observer_key_id,
            "observer_pid": observer_pid,
            "observer_uid": os.getuid(),
            "observer_start_ticks": observer_start_ticks,
            "observer_source_sha256": "8" * 64,
            "not_before_utc": "2026-07-22T00:00:00Z",
            "expires_utc": "2026-07-22T00:04:00Z",
        },
        authority_private,
    )
    journal = create_file(
        "terminal-observer.journal.jsonl",
        canonical_bytes(authorization),
        0o600,
    )
    publication = {
        "claim": claim_path.name,
        "journal": journal.name,
        "stdout": ["child-0.stdout.log", "child-1.stdout.log"],
        "stderr": ["child-0.stderr.log", "child-1.stderr.log"],
        "receipt": "terminal-authority.receipt.json",
    }
    receipt = sign_message(
        "terminal_receipt",
        {
            "schema": terminal_policy.WIRE_SCHEMA,
            "type": "terminal_receipt",
            "authority_key_id": authority_key_id,
            "authorization_id": authorization["authorization_id"],
            "scope": "acquisition",
            "request_id": request_id,
            "checkpoint_pin_sha256": checkpoint_sha256,
            "envelope_sha256": envelope_sha256,
            "request_sha256": request_sha256,
            "claim_sha256": claim_sha256,
            "observer_key_id": observer_key_id,
            "observer_pid": observer_pid,
            "observer_start_ticks": observer_start_ticks,
            "command_bundle_sha256": command_bundle_sha256,
            "events_sha256": hashlib.sha256(canonical_bytes([])).hexdigest(),
            "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
            "prerequisites": [
                artifact(checkpoint, "checkpoint_pin"),
                artifact(alignment, "alignment_receipt"),
            ],
            "target_results": [],
            "terminal_state": {
                "kind": "UNAUTHENTICATED_TERMINAL",
                "last_authenticated_sequence": 0,
            },
            "publication": publication,
            "created_utc": "2026-07-22T00:04:01Z",
        },
        authority_private,
    )
    receipt_path = create_file(publication["receipt"], canonical_bytes(receipt), 0o600)

    verified = terminal_policy.verify_signed_receipt(receipt_path, public_key)
    assert verified.message == receipt
    assert verified.key_id == authority_key_id

    journal.write_bytes(journal.read_bytes() + canonical_bytes({"tampered": True}))
    with pytest.raises(TerminalPolicyError, match="journal hash"):
        terminal_policy.verify_signed_receipt(receipt_path, public_key)


def _signed_receipt_case(tmp_path: Path, terminal_state: dict[str, object]) -> dict[str, object]:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    evidence.chmod(0o700)

    def create(name: str, payload: bytes, mode: int = 0o600) -> Path:
        path = evidence / name
        path.write_bytes(payload)
        path.chmod(mode)
        return path

    def artifact(path: Path, kind: str) -> dict[str, object]:
        info = path.stat()
        payload = path.read_bytes()
        return {
            "kind": kind,
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "st_dev": info.st_dev,
            "st_ino": info.st_ino,
            "mode": stat.S_IMODE(info.st_mode),
            "nlink": info.st_nlink,
        }

    private = terminal_policy.Ed25519PrivateKey.generate()
    public_bytes = private.public_key().public_bytes(
        terminal_policy.serialization.Encoding.Raw,
        terminal_policy.serialization.PublicFormat.Raw,
    )
    public_key = create("authority.public", public_bytes, 0o444)
    key_id = hashlib.sha256(public_bytes).hexdigest()
    root_info = evidence.stat()
    request_id = "1" * 64
    checkpoint_sha256 = "2" * 64
    claim = {
        "schema": terminal_policy.CLAIM_SCHEMA,
        "request_id": request_id,
        "scope": "acquisition",
        "checkpoint_pin_sha256": checkpoint_sha256,
        "evidence_root": {
            "path": str(evidence),
            "st_dev": root_info.st_dev,
            "st_ino": root_info.st_ino,
            "st_uid": root_info.st_uid,
            "st_gid": root_info.st_gid,
            "mode": stat.S_IMODE(root_info.st_mode),
        },
        "observer_pid": os.getpid(),
        "observer_uid": os.getuid(),
        "observer_start_ticks": 1,
        "created_utc": "2026-07-22T00:00:00Z",
    }
    claim_path = create("prelaunch.claim.json", canonical_bytes(claim))
    claim_sha256 = hashlib.sha256(claim_path.read_bytes()).hexdigest()
    authorization = sign_message(
        "authorization",
        {
            "schema": terminal_policy.WIRE_SCHEMA,
            "type": "authorization",
            "authority_key_id": key_id,
            "authorization_id": "3" * 64,
            "scope": "acquisition",
            "request_id": request_id,
            "checkpoint_pin_sha256": checkpoint_sha256,
            "envelope_sha256": "4" * 64,
            "request_sha256": "5" * 64,
            "command_bundle_sha256": "6" * 64,
            "claim_sha256": claim_sha256,
            "observer_key_id": "7" * 64,
            "observer_pid": os.getpid(),
            "observer_uid": os.getuid(),
            "observer_start_ticks": 1,
            "observer_source_sha256": "8" * 64,
            "not_before_utc": "2026-07-22T00:00:00Z",
            "expires_utc": "2026-07-22T00:04:00Z",
        },
        private,
    )
    journal = create("terminal-observer.journal.jsonl", canonical_bytes(authorization))
    publication = {
        "claim": claim_path.name,
        "journal": journal.name,
        "stdout": ["child-0.stdout.log", "child-1.stdout.log"],
        "stderr": ["child-0.stderr.log", "child-1.stderr.log"],
        "receipt": "terminal-authority.receipt.json",
    }

    def result(index: int, return_code: int) -> dict[str, object]:
        stdout = create(publication["stdout"][index], b"stdout\n")
        stderr = create(publication["stderr"][index], b"stderr\n")
        validated: list[dict[str, object]] = []
        if return_code == 0:
            for kind in ("source_eligible_receipt", "source_manifest", "source_journal"):
                validated.append(artifact(create(f"{index}.{kind}", b"artifact\n"), kind))
        return {
            "command_index": index,
            "argv_sha256": "9" * 64,
            "environment_sha256": "a" * 64,
            "return_code": return_code,
            "stdout": artifact(stdout, "stdout"),
            "stderr": artifact(stderr, "stderr"),
            "validated_artifacts": validated,
            "sealed_artifacts": [],
            "completed_utc": "2026-07-22T00:04:00Z",
        }

    kind = terminal_state["kind"]
    if kind == "SUCCEEDED":
        results = [result(0, 0), result(1, 0)]
    elif kind == "FAILED":
        results = [result(0, 0), result(1, 1)]
    elif kind in {"START_FAILED", "START_UNKNOWN", "OBSERVER_LOST"}:
        results = [result(0, 0)]
    else:
        results = []
    receipt = {
        "schema": terminal_policy.WIRE_SCHEMA,
        "type": "terminal_receipt",
        "authority_key_id": key_id,
        "authorization_id": authorization["authorization_id"],
        "scope": "acquisition",
        "request_id": request_id,
        "checkpoint_pin_sha256": checkpoint_sha256,
        "envelope_sha256": "4" * 64,
        "request_sha256": "5" * 64,
        "claim_sha256": claim_sha256,
        "observer_key_id": "7" * 64,
        "observer_pid": os.getpid(),
        "observer_start_ticks": 1,
        "command_bundle_sha256": "6" * 64,
        "events_sha256": hashlib.sha256(canonical_bytes([])).hexdigest(),
        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "prerequisites": [
            artifact(create("checkpoint.json", b"checkpoint\n"), "checkpoint_pin"),
            artifact(create("alignment.json", b"alignment\n"), "alignment_receipt"),
        ],
        "target_results": results,
        "terminal_state": terminal_state,
        "publication": publication,
        "created_utc": "2026-07-22T00:04:01Z",
    }
    receipt_path = evidence / publication["receipt"]
    return {
        "private": private,
        "public_key": public_key,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


def _write_signed_receipt(case: dict[str, object]) -> None:
    receipt = dict(case["receipt"])
    receipt.pop("authority_signature_b64", None)
    signed = sign_message("terminal_receipt", receipt, case["private"])
    case["receipt"] = signed
    case["receipt_path"].write_bytes(canonical_bytes(signed))


@pytest.mark.parametrize(
    "terminal_state",
    [
        {"kind": "SUCCEEDED"},
        {"kind": "FAILED", "failed_command_index": 1},
        {"kind": "START_FAILED", "command_index": 1, "errno": 1},
        {"kind": "START_UNKNOWN", "command_index": 1},
        {
            "kind": "OBSERVER_LOST",
            "command_index": 1,
            "child_pid": os.getpid(),
            "child_start_ticks": 1,
        },
    ],
)
def test_signed_receipt_accepts_each_authenticated_terminal_shape(
    tmp_path: Path, terminal_state: dict[str, object]
) -> None:
    case = _signed_receipt_case(tmp_path, terminal_state)
    _write_signed_receipt(case)

    verified = terminal_policy.verify_signed_receipt(case["receipt_path"], case["public_key"])

    assert verified.message == case["receipt"]


@pytest.mark.parametrize(
    ("terminal_state", "return_code", "match"),
    [
        ({"kind": "SUCCEEDED"}, 1, "successful receipt has incomplete results"),
        ({"kind": "FAILED", "failed_command_index": 1}, 0, "failed receipt result mismatch"),
        (
            {"kind": "START_FAILED", "command_index": 1, "errno": 1},
            1,
            "incomplete receipt result mismatch",
        ),
        (
            {"kind": "START_UNKNOWN", "command_index": 1},
            1,
            "incomplete receipt result mismatch",
        ),
        (
            {
                "kind": "OBSERVER_LOST",
                "command_index": 1,
                "child_pid": os.getpid(),
                "child_start_ticks": 1,
            },
            1,
            "incomplete receipt result mismatch",
        ),
    ],
)
def test_signed_receipt_rejects_terminal_result_near_misses(
    tmp_path: Path, terminal_state: dict[str, object], return_code: int, match: str
) -> None:
    case = _signed_receipt_case(tmp_path, terminal_state)
    case["receipt"]["target_results"][-1]["return_code"] = return_code
    _write_signed_receipt(case)

    with pytest.raises(TerminalPolicyError, match=match):
        terminal_policy.verify_signed_receipt(case["receipt_path"], case["public_key"])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda receipt: receipt.__setitem__(
                "target_results", list(reversed(receipt["target_results"]))
            ),
            "terminal result order mismatch",
        ),
        (
            lambda receipt: receipt.__setitem__("target_results", receipt["target_results"][:-1]),
            "successful receipt has incomplete results",
        ),
        (
            lambda receipt: receipt["target_results"][1].__setitem__("return_code", 1),
            "successful receipt has incomplete results",
        ),
        (
            lambda receipt: receipt["target_results"][0]["stdout"].__setitem__(
                "path", receipt["target_results"][0]["stderr"]["path"]
            ),
            "terminal result log path mismatch",
        ),
        (
            lambda receipt: receipt["target_results"][0]["validated_artifacts"][0].__setitem__(
                "kind", "wrong_kind"
            ),
            "terminal result artifact contract mismatch",
        ),
    ],
)
def test_signed_success_receipt_rejects_result_contract_near_misses(
    tmp_path: Path, mutation: object, match: str
) -> None:
    case = _signed_receipt_case(tmp_path, {"kind": "SUCCEEDED"})
    mutation(case["receipt"])
    _write_signed_receipt(case)

    with pytest.raises(TerminalPolicyError, match=match):
        terminal_policy.verify_signed_receipt(case["receipt_path"], case["public_key"])


def test_signed_success_receipt_rejects_log_mode_and_identity_drift(tmp_path: Path) -> None:
    case = _signed_receipt_case(tmp_path, {"kind": "SUCCEEDED"})
    stdout = Path(case["receipt"]["target_results"][0]["stdout"]["path"])
    stdout.chmod(0o644)
    _write_signed_receipt(case)

    with pytest.raises(TerminalPolicyError, match="identity drift"):
        terminal_policy.verify_signed_receipt(case["receipt_path"], case["public_key"])

    info = stdout.stat()
    case["receipt"]["target_results"][0]["stdout"].update(
        {
            "mode": stat.S_IMODE(info.st_mode),
            "st_dev": info.st_dev,
            "st_ino": info.st_ino,
            "nlink": info.st_nlink,
        }
    )
    _write_signed_receipt(case)
    with pytest.raises(TerminalPolicyError, match="terminal result log mode mismatch"):
        terminal_policy.verify_signed_receipt(case["receipt_path"], case["public_key"])
