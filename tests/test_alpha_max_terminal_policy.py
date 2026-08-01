from __future__ import annotations

import hashlib
import inspect
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
    authorization_epoch_window,
    parse_utc_second,
    validate_sha256,
)


ROOT = Path(__file__).parents[1]
POLICY = ROOT / "configs/research/alpha_max_terminal_authority_policy_v1.json"
EXPECTED_V3_PINS = {
    "acquirer_sha256": "d3c674ecf28c5869eab43f9903b4479185b36faca108919868c2f2c31662db70",
    "alignment_receipt_sha256": "8687b52180502a11de9fbe317a19d00bb4492c464b3bf33d4eda2437683ca812",
    "availability_sha256": "214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719",
    "contract_sha256": "ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220",
    "historical_sha256": "951290033c7efd9b59ba5418e38d96fbdcf3885211915b29010b79ae545f3fb0",
    "phase_wrapper_sha256": "0db198af0b743df0bdb6d3700ed8f0bc53cc28373846ba890e1ac70edc287ce1",
    "portfolio_sha256": "2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c",
    "prelock_sha256": "838d633ae34d44443dad4990a79f4d8caa95f7102ffe2a649ed341b1bed16ad0",
    "preparer_sha256": "ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac",
    "process_boundary_sha256": "f95e8e0d356ca36063a415a7b37919e72d9d1f47af7d2c447e228546fddfb94c",
    "runbook_sha256": "249694fb1513354d61f67552f5c1b9175382f3c2bf9f271ee64dc0358d3c663f",
    "uv_lock_sha256": "59d9de230be950761736c24e04af3456e229cf4aa077536167fb7e650a71c339",
}
TEST_SCOPE_CONTRACT = {
    "acquisition": {
        "prerequisites": ("checkpoint_pin", "alignment_receipt"),
        "commands": (
            ("source_eligible_receipt", "source_manifest", "source_journal"),
            ("source_eligible_receipt", "source_manifest", "source_journal"),
        ),
    },
    "phase_preparation": {
        "prerequisites": (
            "checkpoint_pin",
            "alignment_receipt",
            "source_eligible_receipt",
            "source_manifest",
            "source_journal",
        ),
        "commands": (("phase_handoff_receipt", "preparation_manifest"),),
    },
    "one_touch": {
        "prerequisites": (
            "checkpoint_pin",
            "alignment_receipt",
            "phase_handoff_receipt",
            "preparation_manifest",
        ),
        "commands": (
            (
                "prelock_readback",
                "prelock_observability",
                "prelock_inventory_before",
                "input_inventory_before",
                "prelock_bundle",
            ),
            (
                "historical_readback",
                "historical_observability",
                "prelock_inventory_after",
                "input_inventory_after",
                "historical_bundle",
            ),
        ),
    },
}
TEST_SCOPES = tuple(TEST_SCOPE_CONTRACT)
ARTIFACT_KINDS = tuple(
    dict.fromkeys(
        kind
        for contract in TEST_SCOPE_CONTRACT.values()
        for command in contract["commands"]
        for kind in command
    )
)


def test_policy_has_accepted_head_and_full_v3_pins() -> None:
    policy = load_policy(POLICY)
    assert policy.accepted_alpha_commit == "391000b40717386765bfa39bd212d91c2e3be794"
    assert policy.baseline_ancestor == "629d91e5d4aac26911af65a4a5e15ebdcbded30f"
    assert policy.scope_order == TEST_SCOPES
    assert policy.pins == EXPECTED_V3_PINS
    assert POLICY.read_bytes() == canonical_bytes(json.loads(POLICY.read_text()))
    assert policy.source_sha256 == hashlib.sha256(POLICY.read_bytes()).hexdigest()


def test_frozen_acquirer_wrapper_and_lock_bytes_are_pinned() -> None:
    frozen = {
        ROOT / "scripts/research/acquire_alpha_max_official_source.py": EXPECTED_V3_PINS[
            "acquirer_sha256"
        ],
        ROOT
        / "scripts/research/run_alpha_max_phase_preparation_from_eligible_source.py": EXPECTED_V3_PINS[
            "phase_wrapper_sha256"
        ],
        ROOT / "uv.lock": "603d057f5c520b1864944ea2ab131d2ac8af0dce065bdde0a2bac854f238a92a",
    }
    for path, expected_sha256 in frozen.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def test_public_wire_domain_shapes_and_verifier_signature_are_frozen() -> None:
    assert terminal_policy.WIRE_SCHEMA == "alpha-max-terminal-authority/v3"
    assert terminal_policy.CLAIM_SCHEMA == "alpha_max_terminal_claim.v1"
    signature = inspect.signature(terminal_policy.verify_signed_receipt)
    assert tuple(signature.parameters) == ("path", "public_key_path")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
    assert terminal_policy.MESSAGE_SIGNATURE_FIELDS == {
        "challenge": "authority_signature_b64",
        "observer_proof": "observer_signature_b64",
        "authorization": "authority_signature_b64",
        "command_clearance": "authority_signature_b64",
        "process_event": "observer_signature_b64",
        "terminal_receipt": "authority_signature_b64",
    }
    assert {
        "schema",
        "type",
        "authority_key_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
        "nonce_b64",
        "issued_utc",
        "authority_signature_b64",
    } == terminal_policy.CHALLENGE_FIELDS
    assert {
        "schema",
        "type",
        "authority_key_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
        "nonce_b64",
        "observer_key_id",
        "observer_pid",
        "observer_uid",
        "observer_start_ticks",
        "observer_source_sha256",
        "claim_sha256",
        "observer_signature_b64",
    } == terminal_policy.OBSERVER_PROOF_FIELDS
    assert {
        "schema",
        "type",
        "authority_key_id",
        "authorization_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "command_bundle_sha256",
        "claim_sha256",
        "observer_key_id",
        "observer_pid",
        "observer_uid",
        "observer_start_ticks",
        "observer_source_sha256",
        "not_before_utc",
        "expires_utc",
        "authority_signature_b64",
    } == terminal_policy.AUTHORIZATION_FIELDS
    assert {
        "schema",
        "type",
        "authority_key_id",
        "authorization_id",
        "scope",
        "request_id",
        "completed_command_index",
        "next_command_index",
        "validated_artifact_snapshot_sha256",
        "issued_utc",
        "authority_signature_b64",
    } == terminal_policy.COMMAND_CLEARANCE_FIELDS
    assert {
        "schema",
        "type",
        "authority_key_id",
        "authorization_id",
        "scope",
        "request_id",
        "checkpoint_pin_sha256",
        "envelope_sha256",
        "request_sha256",
        "claim_sha256",
        "observer_key_id",
        "observer_pid",
        "observer_start_ticks",
        "command_bundle_sha256",
        "events_sha256",
        "journal_sha256",
        "prerequisites",
        "target_results",
        "terminal_state",
        "publication",
        "created_utc",
        "authority_signature_b64",
    } == terminal_policy.TERMINAL_RECEIPT_FIELDS
    assert {
        "schema",
        "request_id",
        "scope",
        "checkpoint_pin_sha256",
        "evidence_root",
        "observer_pid",
        "observer_uid",
        "observer_start_ticks",
        "created_utc",
    } == terminal_policy.CLAIM_FIELDS


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("O_NOFOLLOW", None),
        ("O_CLOEXEC", None),
        ("O_NOFOLLOW", 0),
        ("O_CLOEXEC", 0),
    ],
)
def test_open_absolute_requires_secure_open_capabilities(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: int | None
) -> None:
    if value is None:
        monkeypatch.delattr(terminal_policy.os, attribute, raising=False)
    else:
        monkeypatch.setattr(terminal_policy.os, attribute, value)
    monkeypatch.setattr(
        terminal_policy.os,
        "open",
        lambda *_args, **_kwargs: pytest.fail("open must not be attempted"),
    )

    with pytest.raises(TerminalPolicyError, match="required secure open flags unavailable"):
        terminal_policy._open_absolute("/unused", os.O_RDONLY, "unused")


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
    receipt_bytes = canonical_bytes(receipt)
    receipt_path = create_file(publication["receipt"], receipt_bytes, 0o600)

    verified = terminal_policy.verify_signed_receipt(receipt_path, public_key)
    assert receipt_path.read_bytes() == receipt_bytes
    assert verified.message == receipt
    assert verified.key_id == authority_key_id
    assert verified.authorization == authorization
    assert verified.events == ()

    journal.write_bytes(journal.read_bytes() + canonical_bytes({"tampered": True}))
    with pytest.raises(TerminalPolicyError, match="journal hash"):
        terminal_policy.verify_signed_receipt(receipt_path, public_key)


def _signed_receipt_case(
    tmp_path: Path, terminal_state: dict[str, object], scope: str = "acquisition"
) -> dict[str, object]:
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

    def sealed_artifact(path: Path, kind: str) -> dict[str, object]:
        value = artifact(path, kind)
        value.update(
            {
                "sealed_payload_sha256": "b" * 64,
                "canonical_inventory_sha256": "c" * 64,
                "readback_sha256": "d" * 64,
            }
        )
        return value

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
    command_count = len(TEST_SCOPE_CONTRACT[scope]["commands"])
    claim = {
        "schema": terminal_policy.CLAIM_SCHEMA,
        "request_id": request_id,
        "scope": scope,
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
            "scope": scope,
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
        "stdout": [f"child-{index}.stdout.log" for index in range(command_count)],
        "stderr": [f"child-{index}.stderr.log" for index in range(command_count)],
        "receipt": "terminal-authority.receipt.json",
    }

    def result(index: int, return_code: int) -> dict[str, object]:
        stdout = create(publication["stdout"][index], b"stdout\n")
        stderr = create(publication["stderr"][index], b"stderr\n")
        contract = TEST_SCOPE_CONTRACT[scope]["commands"][index]
        validated_kinds = tuple(kind for kind in contract if not kind.endswith("_bundle"))
        sealed_kinds = tuple(kind for kind in contract if kind.endswith("_bundle"))
        validated = [
            artifact(create(f"{index}.{artifact_kind}", b"artifact\n"), artifact_kind)
            for artifact_kind in validated_kinds
        ]
        sealed = [
            sealed_artifact(create(f"{index}.{artifact_kind}", b"sealed artifact\n"), artifact_kind)
            for artifact_kind in sealed_kinds
        ]
        if return_code != 0:
            validated = []
            sealed = []
        return {
            "command_index": index,
            "argv_sha256": "9" * 64,
            "environment_sha256": "a" * 64,
            "return_code": return_code,
            "stdout": artifact(stdout, "stdout"),
            "stderr": artifact(stderr, "stderr"),
            "validated_artifacts": validated,
            "sealed_artifacts": sealed,
            "completed_utc": "2026-07-22T00:04:00Z",
        }

    kind = terminal_state["kind"]
    if kind == "SUCCEEDED":
        results = [result(index, 0) for index in range(command_count)]
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
        "scope": scope,
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
            artifact(create(f"{kind}.json", b"prerequisite\n"), kind)
            for kind in TEST_SCOPE_CONTRACT[scope]["prerequisites"]
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


@pytest.mark.parametrize("scope", TEST_SCOPES)
def test_signed_success_receipt_accepts_each_scope(tmp_path: Path, scope: str) -> None:
    case = _signed_receipt_case(tmp_path, {"kind": "SUCCEEDED"}, scope)
    _write_signed_receipt(case)

    verified = terminal_policy.verify_signed_receipt(case["receipt_path"], case["public_key"])

    assert verified.message == case["receipt"]


@pytest.mark.parametrize("scope", TEST_SCOPES)
@pytest.mark.parametrize("mutation", ("missing", "wrong", "reordered"))
def test_signed_receipt_rejects_each_scope_prerequisite_contract(
    tmp_path: Path, scope: str, mutation: str
) -> None:
    case = _signed_receipt_case(tmp_path, {"kind": "SUCCEEDED"}, scope)
    prerequisites = case["receipt"]["prerequisites"]
    if mutation == "missing":
        prerequisites.pop()
    elif mutation == "wrong":
        prerequisites[0]["kind"] = "wrong_kind"
    else:
        prerequisites.reverse()
    _write_signed_receipt(case)

    with pytest.raises(TerminalPolicyError, match="invalid terminal receipt collections"):
        terminal_policy.verify_signed_receipt(case["receipt_path"], case["public_key"])


@pytest.mark.parametrize(
    ("scope", "command_index", "artifact_field", "mutation"),
    [
        (scope, index, field, mutation)
        for scope, contract in TEST_SCOPE_CONTRACT.items()
        for index, command in enumerate(contract["commands"])
        for field, artifacts in (
            (
                "validated_artifacts",
                tuple(kind for kind in command if not kind.endswith("_bundle")),
            ),
            ("sealed_artifacts", tuple(kind for kind in command if kind.endswith("_bundle"))),
        )
        if artifacts
        for mutation in ("missing", "wrong", "reordered")
        if mutation != "reordered" or len(artifacts) >= 2
    ],
)
def test_signed_success_receipt_rejects_every_literal_artifact_contract(
    tmp_path: Path, scope: str, command_index: int, artifact_field: str, mutation: str
) -> None:
    case = _signed_receipt_case(tmp_path, {"kind": "SUCCEEDED"}, scope)
    artifacts = case["receipt"]["target_results"][command_index][artifact_field]
    if mutation == "missing":
        artifacts.pop()
    elif mutation == "wrong":
        existing = {artifact["kind"] for artifact in artifacts}
        artifacts[0]["kind"] = next(kind for kind in ARTIFACT_KINDS if kind not in existing)
    else:
        artifacts.reverse()
    _write_signed_receipt(case)

    with pytest.raises(TerminalPolicyError, match="terminal result artifact contract mismatch"):
        terminal_policy.verify_signed_receipt(case["receipt_path"], case["public_key"])


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


class _StringSubclass(str):
    pass


class _IntegerSubclass(int):
    pass


@pytest.mark.parametrize(
    ("validator", "value", "label"),
    (
        pytest.param("digest", "A" * 64, "digest", id="digest-uppercase"),
        pytest.param("digest", " " + ("a" * 64), "digest", id="digest-leading-whitespace"),
        pytest.param("digest", ("a" * 64) + " ", "digest", id="digest-trailing-whitespace"),
        pytest.param(
            "digest", ("a" * 32) + " " + ("a" * 31), "digest", id="digest-internal-whitespace"
        ),
        pytest.param("digest", b"a" * 64, "digest", id="digest-bytes"),
        pytest.param("digest", _StringSubclass("a" * 64), "digest", id="digest-string-subclass"),
        pytest.param("utc", " 2026-07-24T01:02:03Z", "time", id="utc-leading-whitespace"),
        pytest.param("utc", "2026-07-24T01:02:03Z ", "time", id="utc-trailing-whitespace"),
        pytest.param("utc", "2026-07-24T01:02:03 Z", "time", id="utc-internal-whitespace"),
        pytest.param("utc", "2026-07-24T01:02:60Z", "time", id="utc-leap-second"),
        pytest.param("utc", "2026-07-24T01:02:03+00:00", "time", id="utc-offset"),
        pytest.param("utc", "2026-07-24T01:02:03.000000Z", "time", id="utc-fractional-second"),
        pytest.param("utc", "2026-07-24T01:02:03z", "time", id="utc-lowercase-z"),
        pytest.param("utc", "2026-02-29T01:02:03Z", "time", id="utc-normalization-like-date"),
        pytest.param("utc", "2026-07-24T24:02:03Z", "time", id="utc-normalization-like-time"),
        pytest.param("utc", "2026-7-24T01:02:03Z", "time", id="utc-unpadded-date"),
        pytest.param(
            "utc", _StringSubclass("2026-07-24T01:02:03Z"), "time", id="utc-string-subclass"
        ),
        pytest.param("utc", True, "time", id="utc-bool"),
        pytest.param("utc", 0, "time", id="utc-int"),
        pytest.param("utc", _IntegerSubclass(0), "time", id="utc-int-subclass"),
        pytest.param("utc", 0.0, "time", id="utc-float"),
    ),
)
def test_shared_digest_and_utc_grammar_is_exact(validator: str, value: object, label: str) -> None:
    digest = "a" * 64
    assert validate_sha256(digest, "digest") == digest
    assert (
        parse_utc_second("2026-07-24T01:02:03Z", "time").isoformat() == "2026-07-24T01:02:03+00:00"
    )

    with pytest.raises(TerminalPolicyError, match=f"invalid {label}"):
        if validator == "digest":
            validate_sha256(value, label)
        else:
            parse_utc_second(value, label)


@pytest.mark.parametrize(
    "value",
    (
        pytest.param(True, id="window-bool"),
        pytest.param(0, id="window-int"),
        pytest.param(_IntegerSubclass(0), id="window-int-subclass"),
        pytest.param(0.0, id="window-float"),
    ),
)
def test_authorization_epoch_window_rejects_nonmapping_inputs(value: object) -> None:
    with pytest.raises(TerminalPolicyError, match="invalid authorization window"):
        authorization_epoch_window(value)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        pytest.param(
            "not_before_utc",
            _StringSubclass("2026-07-24T01:00:00Z"),
            id="window-not-before-string-subclass",
        ),
        pytest.param("not_before_utc", True, id="window-not-before-bool"),
        pytest.param("not_before_utc", 0, id="window-not-before-int"),
        pytest.param("not_before_utc", _IntegerSubclass(0), id="window-not-before-int-subclass"),
        pytest.param("not_before_utc", 0.0, id="window-not-before-float"),
    ),
)
def test_authorization_epoch_window_rejects_structurally_invalid_fields(
    field: str, value: object
) -> None:
    window: dict[str, object] = {
        "not_before_utc": "2026-07-24T01:00:00Z",
        "expires_utc": "2026-07-24T01:05:00Z",
    }
    window[field] = value

    with pytest.raises(TerminalPolicyError, match="invalid authorization not-before"):
        authorization_epoch_window(window)


@pytest.mark.parametrize(
    "expires",
    (
        pytest.param("2026-07-24T01:00:00Z", id="window-empty"),
        pytest.param("2026-07-24T01:05:01Z", id="window-over-five-minutes"),
    ),
)
def test_authorization_epoch_window_is_structural_and_bounded(expires: str) -> None:
    assert authorization_epoch_window(
        {"not_before_utc": "2000-01-01T00:00:00Z", "expires_utc": "2000-01-01T00:05:00Z"}
    ) == (946684800, 946685100)

    with pytest.raises(TerminalPolicyError, match="invalid authorization window"):
        authorization_epoch_window(
            {"not_before_utc": "2026-07-24T01:00:00Z", "expires_utc": expires}
        )
