from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from dataclasses import replace
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import lumina_quant.alpha_max_terminal_policy as terminal_policy
from lumina_quant.alpha_max_terminal_policy import (
    TerminalPolicyError,
    canonical_bytes,
    derive_command_preflight,
    derive_scope_commands,
    load_checkpoint,
    load_envelope,
    load_policy,
    load_request,
    validate_command_semantics,
    validate_prelaunch,
    validate_lexical_control_path,
)


ROOT = Path(__file__).parents[1]
POLICY_PATH = ROOT / "configs/research/alpha_max_terminal_authority_policy_v1.json"

TEST_SCOPES = ("acquisition", "phase_preparation", "one_touch")


def write_canonical(path: Path, value: dict) -> None:
    path.write_bytes(canonical_bytes(value))


def test_descriptor_tree_walk_rejects_symlinked_root_without_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "must-not-read").write_text("sentinel")
    output = tmp_path / "output"
    output.symlink_to(sentinel, target_is_directory=True)
    monkeypatch.setattr(Path, "resolve", lambda *_args, **_kwargs: pytest.fail("resolve"))
    monkeypatch.setattr(Path, "rglob", lambda *_args, **_kwargs: pytest.fail("rglob"))
    monkeypatch.setattr(Path, "iterdir", lambda *_args, **_kwargs: pytest.fail("iterdir"))

    with pytest.raises(TerminalPolicyError, match="cannot open"):
        terminal_policy._safe_tree_files(output, "output")


def test_descriptor_tree_walk_remains_anchored_after_descendant_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "output"
    nested = root / "nested"
    nested.mkdir(parents=True)
    root.chmod(0o700)
    (nested / "evidence.json").write_text("{}")
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()
    (sentinel / "must-not-read").write_text("sentinel")
    sentinel_directory_identity = (sentinel.stat().st_dev, sentinel.stat().st_ino)
    sentinel_file = sentinel / "must-not-read"
    sentinel_file_identity = (sentinel_file.stat().st_dev, sentinel_file.stat().st_ino)
    original_open_child = terminal_policy._open_child_fd
    original_open = os.open
    original_stat = os.stat
    original_read = os.read
    replaced = False

    def assert_not_sentinel(fd: int) -> None:
        identity = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        if identity in {sentinel_directory_identity, sentinel_file_identity}:
            pytest.fail("sentinel identity was accessed")

    def guarded_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is not None:
            assert_not_sentinel(dir_fd)
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        assert_not_sentinel(descriptor)
        return descriptor

    def guarded_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if dir_fd is not None:
            assert_not_sentinel(dir_fd)
        result = original_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if (result.st_dev, result.st_ino) in {
            sentinel_directory_identity,
            sentinel_file_identity,
        }:
            pytest.fail("sentinel identity was accessed")
        return result

    def guarded_read(fd: int, length: int) -> bytes:
        assert_not_sentinel(fd)
        return original_read(fd, length)

    def replace_after_open(parent_fd: int, name: str, flags: int, label: str) -> int:
        nonlocal replaced
        descriptor = original_open_child(parent_fd, name, flags, label)
        if name == "nested" and not replaced:
            replaced = True
            nested.rename(root / "nested-replaced")
            nested.symlink_to(sentinel, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(os, "stat", guarded_stat)
    monkeypatch.setattr(os, "read", guarded_read)
    monkeypatch.setattr(terminal_policy, "_open_child_fd", replace_after_open)

    assert terminal_policy._safe_tree_files(root, "output") == {"nested/evidence.json"}
    assert replaced


def test_envelope_rejects_missing_ordered_file_roles(tmp_path: Path) -> None:
    policy, _checkpoint, value = _loaded_envelope(tmp_path)
    path = tmp_path / "envelope.json"
    for mutate in (
        lambda item: item["files"].pop(),
        lambda item: item["files"].reverse(),
    ):
        candidate = copy.deepcopy(value)
        mutate(candidate)
        checkpoint = _checkpoint_for(path, policy, candidate)
        with pytest.raises(TerminalPolicyError, match="invalid envelope file roles"):
            load_envelope(path, policy, checkpoint)


def test_envelope_rejects_duplicate_authority_and_observer_key_ids(tmp_path: Path) -> None:
    policy, _checkpoint, value = _loaded_envelope(tmp_path)
    path = tmp_path / "envelope.json"
    candidate = copy.deepcopy(value)
    candidate["observer_keys"][0].update(candidate["authority_key"])
    candidate["observer_keys"][0]["scope"] = TEST_SCOPES[0]
    checkpoint = _checkpoint_for(path, policy, candidate)

    with pytest.raises(TerminalPolicyError, match="duplicate authority or observer key id"):
        load_envelope(path, policy, checkpoint)


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _file(path: Path, digest: str | None = None) -> dict[str, int | str]:
    return {
        "path": str(path),
        "sha256": _digest(str(path)) if digest is None else digest,
        "byte_count": 1,
        "st_dev": 1,
        "st_ino": abs(hash(str(path))) % 1_000_000 + 1,
        "st_uid": os.getuid(),
        "st_gid": os.getgid(),
        "mode": 0o600,
        "nlink": 1,
    }


def _directory(path: Path) -> dict[str, int | str]:
    info = path.stat()
    return {
        "path": str(path),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": info.st_mode & 0o777,
    }


def _envelope_value(tmp_path: Path, policy: object) -> dict:
    key = b"k" * 32
    key_id = hashlib.sha256(key).hexdigest()
    encoded = base64.b64encode(key).decode()
    observer_keys = []
    for index, scope in enumerate(TEST_SCOPES, start=1):
        observer_key = bytes([index]) * 32
        observer_key_id = hashlib.sha256(observer_key).hexdigest()
        observer_keys.append(
            {
                "scope": scope,
                "key_id": observer_key_id,
                "public_key_b64": base64.b64encode(observer_key).decode(),
                "public_key_sha256": observer_key_id,
            }
        )
    pin_roles = {
        "runbook": "runbook_sha256",
        "alpha_uv_lock": "uv_lock_sha256",
        "alignment_receipt": "alignment_receipt_sha256",
        "portfolio": "portfolio_sha256",
        "contract_manifest": "contract_sha256",
        "availability_evidence": "availability_sha256",
        "preparer": "preparer_sha256",
        "prelock_script": "prelock_sha256",
        "historical_script": "historical_sha256",
        "process_boundary": "process_boundary_sha256",
        "acquirer": "acquirer_sha256",
        "phase_wrapper": "phase_wrapper_sha256",
    }
    roles = (
        "policy_json",
        "policy_module",
        "authority_script",
        "observer_script",
        "key_creator",
        "acquirer",
        "phase_wrapper",
        "runbook",
        "alpha_uv_lock",
        "alignment_receipt",
        "portfolio",
        "contract_manifest",
        "availability_evidence",
        "preparer",
        "prelock_script",
        "historical_script",
        "process_boundary",
    )
    current, accepted = tmp_path / "current", tmp_path / "accepted"
    current.mkdir()
    accepted.mkdir()
    return {
        "schema": "alpha_max_terminal_launch_envelope.v3",
        "policy_sha256": policy.source_sha256,
        "current_head": "1" * 40,
        "accepted_alpha_commit": policy.accepted_alpha_commit,
        "baseline_ancestor": policy.baseline_ancestor,
        "repositories": [
            {
                "role": "current_repository",
                "root": _directory(current),
                "head": "1" * 40,
                "clean_receipt": _file(tmp_path / "receipt-current"),
            },
            {
                "role": "accepted_alpha_repository",
                "root": _directory(accepted),
                "head": policy.accepted_alpha_commit,
                "clean_receipt": _file(tmp_path / "receipt-accepted"),
            },
        ],
        "files": [
            {
                "role": role,
                "file": _file(
                    tmp_path / "bound" / role,
                    policy.pins[pin_roles[role]] if role in pin_roles else None,
                ),
            }
            for role in roles
        ],
        "interpreters": [
            {
                "role": "current_python",
                "file": _file(tmp_path / "python-current"),
                "package_freeze": _file(tmp_path / "freeze-current"),
            },
            {
                "role": "accepted_alpha_python",
                "file": _file(tmp_path / "python-accepted"),
                "package_freeze": _file(tmp_path / "freeze-accepted"),
            },
        ],
        "authority_key": {"key_id": key_id, "public_key_b64": encoded, "public_key_sha256": key_id},
        "observer_keys": observer_keys,
        "forbidden_roots": list(terminal_policy._FORBIDDEN_ROOTS),
        "scope_order": ["acquisition", "phase_preparation", "one_touch"],
    }


def _loaded_envelope(tmp_path: Path) -> tuple[object, object, dict]:
    policy = load_policy(POLICY_PATH)
    value = _envelope_value(tmp_path, policy)
    path = tmp_path / "envelope.json"
    write_canonical(path, value)
    checkpoint_value = {
        "schema": "alpha_max_terminal_checkpoint.v1",
        "accepted_alpha_commit": policy.accepted_alpha_commit,
        "baseline_ancestor": policy.baseline_ancestor,
        **policy.pins,
        "authority_manifest_sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(),
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    write_canonical(checkpoint_path, checkpoint_value)
    return policy, load_checkpoint(checkpoint_path, policy), value


def _checkpoint_for(path: Path, policy: object, envelope: dict) -> object:
    write_canonical(path, envelope)
    value = {
        "schema": "alpha_max_terminal_checkpoint.v1",
        "accepted_alpha_commit": policy.accepted_alpha_commit,
        "baseline_ancestor": policy.baseline_ancestor,
        **policy.pins,
        "authority_manifest_sha256": hashlib.sha256(canonical_bytes(envelope)).hexdigest(),
    }
    checkpoint_path = path.with_name("candidate-checkpoint.json")
    write_canonical(checkpoint_path, value)
    return load_checkpoint(checkpoint_path, policy)


def _request_value(tmp_path: Path, checkpoint: object, envelope: object, scope: str) -> dict:
    evidence, parent = tmp_path / f"evidence-{scope}", tmp_path / f"outputs-{scope}"
    evidence.mkdir()
    parent.mkdir()
    evidence.chmod(0o700)
    parent.chmod(0o700)
    files = {item.role: item.file for item in envelope.files}
    root = envelope.repositories[1 if scope == "one_touch" else 0].root
    interpreter = envelope.interpreters[1 if scope == "one_touch" else 0].file

    def absent(leaf: str) -> dict:
        return {
            "path": str(parent / leaf),
            "parent": _directory(parent),
            "leaf": leaf,
            "must_be_absent": True,
        }

    if scope == "acquisition":
        records = {
            "acquirer": files["acquirer"],
            "contract_manifest": files["contract_manifest"],
            "availability_evidence": files["availability_evidence"],
            "source_root": absent("source"),
            "report_root": absent("report"),
        }
        kinds = ("checkpoint_pin", "alignment_receipt")
    elif scope == "phase_preparation":
        source, report = tmp_path / "source", tmp_path / "report"
        source.mkdir()
        report.mkdir()
        records = {
            "phase_wrapper": files["phase_wrapper"],
            "acquirer": files["acquirer"],
            "source_root": _directory(source),
            "source_report": _directory(report),
            "contract_manifest": files["contract_manifest"],
            "availability_evidence": files["availability_evidence"],
            "preparer": files["preparer"],
            "phase_output": absent("phase"),
        }
        kinds = (
            "checkpoint_pin",
            "alignment_receipt",
            "source_eligible_receipt",
            "source_manifest",
            "source_journal",
        )
    else:
        phase = tmp_path / "phase"
        phase.mkdir()
        records = {
            "portfolio": files["portfolio"],
            "contract_manifest": files["contract_manifest"],
            "prelock_script": files["prelock_script"],
            "historical_script": files["historical_script"],
            "phase_output": _directory(phase),
            "prelock_output": absent("prelock"),
            "historical_output": absent("historical"),
        }
        kinds = (
            "checkpoint_pin",
            "alignment_receipt",
            "phase_handoff_receipt",
            "preparation_manifest",
        )
    prerequisites = []
    for kind in kinds:
        if kind == "checkpoint_pin":
            item = terminal_policy._plain(checkpoint.source_identity)
        elif kind == "alignment_receipt":
            item = terminal_policy._plain(files["alignment_receipt"])
        elif kind.startswith("source_"):
            suffix = {
                "source_eligible_receipt": "source_eligible_receipt.json",
                "source_manifest": "source_manifest.json",
                "source_journal": "acquisition.journal.jsonl",
            }[kind]
            item = _file(Path(records["source_report"]["path"]) / suffix)
        elif kind == "phase_handoff_receipt":
            phase_path = Path(records["phase_output"]["path"])
            item = _file(
                phase_path.parent / f".{phase_path.name}.alpha_max_phase_preparation.handoff.json"
            )
        else:
            item = _file(Path(records["phase_output"]["path"]) / "preparation_manifest.json")
        prerequisites.append(
            {
                "kind": kind,
                **{key: value for key, value in item.items() if key not in {"st_uid", "st_gid"}},
            }
        )
    count = 1 if scope == "phase_preparation" else 2
    return {
        "schema": f"alpha_max_terminal_request.{scope}.v1",
        "request_id": "a" * 64,
        "scope": scope,
        "checkpoint_pin_sha256": checkpoint.sha256,
        "interpreter": terminal_policy._plain(interpreter),
        "repository_root": terminal_policy._plain(root),
        "evidence_root": _directory(evidence),
        "authority_socket": str(evidence / "terminal-authority.sock"),
        "environment": {
            "HOME": str(evidence),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TZ": "UTC",
        },
        "forbidden_roots": list(envelope.forbidden_roots),
        "publication": {
            "claim": "prelaunch.claim.json",
            "journal": "terminal-observer.journal.jsonl",
            "stdout": [f"child-{i}.stdout.log" for i in range(count)],
            "stderr": [f"child-{i}.stderr.log" for i in range(count)],
            "receipt": "terminal-authority.receipt.json",
        },
        "prerequisites": prerequisites,
        **records,
    }


def test_checkpoint_binds_exact_canonical_envelope_bytes_and_roles(tmp_path: Path) -> None:
    policy, checkpoint, value = _loaded_envelope(tmp_path)
    path = tmp_path / "envelope.json"
    assert (
        load_envelope(path, policy, checkpoint).sha256
        == hashlib.sha256(canonical_bytes(value)).hexdigest()
    )
    changed = copy.deepcopy(value)
    changed["current_head"] = "2" * 40
    write_canonical(path, changed)
    with pytest.raises(TerminalPolicyError, match="authority manifest"):
        load_envelope(path, policy, checkpoint)
    for mutate, error in (
        (lambda item: item["files"].append(copy.deepcopy(item["files"][0])), "file roles"),
        (lambda item: item["files"][0].__setitem__("extra", True), "file roles"),
        (lambda item: item["repositories"].reverse(), "repositories"),
        (
            lambda item: item["interpreters"].append(copy.deepcopy(item["interpreters"][0])),
            "interpreters",
        ),
    ):
        candidate = copy.deepcopy(value)
        mutate(candidate)
        candidate_checkpoint = _checkpoint_for(path, policy, candidate)
        with pytest.raises(TerminalPolicyError, match=error):
            load_envelope(path, policy, candidate_checkpoint)


def test_envelope_binds_policy_bytes_and_repository_roles(tmp_path: Path) -> None:
    policy, _checkpoint, value = _loaded_envelope(tmp_path)
    path = tmp_path / "envelope.json"
    for mutate, error in (
        (
            lambda item: item.__setitem__("policy_sha256", _digest("other-policy")),
            "policy mismatch",
        ),
        (
            lambda item: item["repositories"][0].__setitem__("head", policy.accepted_alpha_commit),
            "head binding",
        ),
        (lambda item: item["repositories"][1].__setitem__("head", "2" * 40), "head binding"),
    ):
        candidate = copy.deepcopy(value)
        mutate(candidate)
        checkpoint = _checkpoint_for(path, policy, candidate)
        with pytest.raises(TerminalPolicyError, match=error):
            load_envelope(path, policy, checkpoint)


def test_envelope_rejects_forbidden_protected_path_before_opening(tmp_path: Path) -> None:
    policy, _checkpoint, value = _loaded_envelope(tmp_path)
    value["repositories"][0]["root"]["path"] = terminal_policy._FORBIDDEN_ROOTS[0]
    path = tmp_path / "envelope.json"
    checkpoint = _checkpoint_for(path, policy, value)
    with pytest.raises(TerminalPolicyError, match="forbidden envelope path"):
        load_envelope(path, policy, checkpoint)


@pytest.mark.parametrize("pin", sorted(load_policy(POLICY_PATH).pins))
def test_checkpoint_rejects_every_policy_pin_mismatch(tmp_path: Path, pin: str) -> None:
    policy, _checkpoint, _value = _loaded_envelope(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.json"
    candidate = json.loads(checkpoint_path.read_text())
    candidate[pin] = _digest(f"mismatch:{pin}")
    write_canonical(checkpoint_path, candidate)
    with pytest.raises(TerminalPolicyError, match=f"checkpoint {pin} mismatch"):
        load_checkpoint(checkpoint_path, policy)


@pytest.mark.parametrize(
    ("role", "pin"),
    (
        ("runbook", "runbook_sha256"),
        ("alpha_uv_lock", "uv_lock_sha256"),
        ("alignment_receipt", "alignment_receipt_sha256"),
        ("portfolio", "portfolio_sha256"),
        ("contract_manifest", "contract_sha256"),
        ("availability_evidence", "availability_sha256"),
        ("preparer", "preparer_sha256"),
        ("prelock_script", "prelock_sha256"),
        ("historical_script", "historical_sha256"),
        ("process_boundary", "process_boundary_sha256"),
        ("acquirer", "acquirer_sha256"),
        ("phase_wrapper", "phase_wrapper_sha256"),
    ),
)
def test_envelope_rejects_every_mapped_file_pin_mismatch(
    tmp_path: Path, role: str, pin: str
) -> None:
    policy, _checkpoint, value = _loaded_envelope(tmp_path)
    candidate = copy.deepcopy(value)
    next(item for item in candidate["files"] if item["role"] == role)["file"]["sha256"] = _digest(
        f"mismatch:{role}"
    )
    path = tmp_path / "envelope.json"
    checkpoint = _checkpoint_for(path, policy, candidate)
    with pytest.raises(TerminalPolicyError, match="envelope file pin mismatch"):
        load_envelope(path, policy, checkpoint)


def test_request_binds_checkpoint_prerequisite_to_loaded_source(tmp_path: Path) -> None:
    policy, checkpoint, _value = _loaded_envelope(tmp_path)
    envelope = load_envelope(tmp_path / "envelope.json", policy, checkpoint)
    request = _request_value(tmp_path, checkpoint, envelope, "one_touch")
    request["prerequisites"][0]["path"] = str(tmp_path / "different-checkpoint.json")
    path = tmp_path / "request.json"
    write_canonical(path, request)

    with pytest.raises(TerminalPolicyError, match="checkpoint_pin binding mismatch"):
        load_request(
            path, scope="one_touch", policy=policy, checkpoint=checkpoint, envelope=envelope
        )
    request["prerequisites"][0]["path"] = checkpoint.source_path
    request["prerequisites"][0]["byte_count"] += 1
    write_canonical(path, request)

    with pytest.raises(TerminalPolicyError, match="checkpoint prerequisite identity mismatch"):
        load_request(
            path, scope="one_touch", policy=policy, checkpoint=checkpoint, envelope=envelope
        )


def test_request_rejects_every_forbidden_prerequisite_without_opening_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy, _checkpoint, value = _loaded_envelope(tmp_path)
    envelope_path = tmp_path / "envelope.json"
    checkpoint = _checkpoint_for(envelope_path, policy, value)
    envelope = load_envelope(envelope_path, policy, checkpoint)
    original_regular_file = terminal_policy._regular_file

    def reject_forbidden_open(path: Path | str, *args: object, **kwargs: object) -> object:
        target = str(path)
        assert all(
            target != root and not target.startswith(root + "/")
            for root in envelope.forbidden_roots
        )
        return original_regular_file(path, *args, **kwargs)

    monkeypatch.setattr(terminal_policy, "_regular_file", reject_forbidden_open)
    request_path = tmp_path / "request.json"
    for scope in ("acquisition", "phase_preparation", "one_touch"):
        request = _request_value(tmp_path, checkpoint, envelope, scope)
        for index in range(len(request["prerequisites"])):
            for root in envelope.forbidden_roots:
                for path in (root, f"{root}/nested/prerequisite"):
                    candidate = copy.deepcopy(request)
                    candidate["prerequisites"][index]["path"] = path
                    write_canonical(request_path, candidate)
                    with pytest.raises(TerminalPolicyError, match="forbidden prerequisite path"):
                        load_request(
                            request_path,
                            scope=scope,
                            policy=policy,
                            checkpoint=checkpoint,
                            envelope=envelope,
                        )


def test_one_touch_command_rejects_mismatched_records(tmp_path: Path) -> None:
    policy, checkpoint, _value = _loaded_envelope(tmp_path)
    envelope = load_envelope(tmp_path / "envelope.json", policy, checkpoint)
    one_touch_path = tmp_path / "one-touch.json"
    acquisition_path = tmp_path / "acquisition.json"
    write_canonical(one_touch_path, _request_value(tmp_path, checkpoint, envelope, "one_touch"))
    write_canonical(acquisition_path, _request_value(tmp_path, checkpoint, envelope, "acquisition"))
    one_touch = load_request(
        one_touch_path, scope="one_touch", policy=policy, checkpoint=checkpoint, envelope=envelope
    )
    acquisition = load_request(
        acquisition_path,
        scope="acquisition",
        policy=policy,
        checkpoint=checkpoint,
        envelope=envelope,
    )

    with pytest.raises(TerminalPolicyError, match="scope records do not match scope"):
        derive_scope_commands(envelope, replace(one_touch, records=acquisition.records))


def test_absent_output_revalidates_parent_identity_and_does_not_follow_links(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o700)
    identity = terminal_policy.DirectoryIdentity(**_directory(parent))
    output = terminal_policy.AbsentOutput(str(parent / "output"), identity, "output", True)
    terminal_policy._verify_absent(output)
    os.symlink(tmp_path / "target", parent / "output")
    with pytest.raises(TerminalPolicyError, match="output is not absent"):
        terminal_policy._verify_absent(output)
    drifted = terminal_policy.DirectoryIdentity(
        identity.path,
        identity.st_dev,
        identity.st_ino + 1,
        identity.st_uid,
        identity.st_gid,
        identity.mode,
    )
    with pytest.raises(TerminalPolicyError, match="identity drift"):
        terminal_policy._verify_absent(
            terminal_policy.AbsentOutput(str(parent / "different"), drifted, "different", True)
        )


def test_request_rejects_identity_environment_publication_and_root_confusion(
    tmp_path: Path,
) -> None:
    policy, checkpoint, _value = _loaded_envelope(tmp_path)
    envelope = load_envelope(tmp_path / "envelope.json", policy, checkpoint)
    request = _request_value(tmp_path, checkpoint, envelope, "acquisition")
    path = tmp_path / "request.json"
    for mutate, error in (
        (lambda item: item.__setitem__("request_id", "bad"), "request mismatch"),
        (lambda item: item["environment"].__setitem__("EXTRA", "1"), "environment"),
        (lambda item: item["publication"]["stdout"].pop(), "invalid publication"),
        (
            lambda item: item["forbidden_roots"].__setitem__(0, item["repository_root"]["path"]),
            "forbidden roots mismatch",
        ),
        (
            lambda item: item["source_root"].__setitem__("path", str(tmp_path / "elsewhere")),
            "invalid absent output",
        ),
    ):
        candidate = copy.deepcopy(request)
        mutate(candidate)
        write_canonical(path, candidate)
        with pytest.raises(TerminalPolicyError, match=error):
            load_request(
                path, scope="acquisition", policy=policy, checkpoint=checkpoint, envelope=envelope
            )


def test_scope_command_topology_and_phase_path_shapes_are_fixed(tmp_path: Path) -> None:
    policy, checkpoint, _value = _loaded_envelope(tmp_path)
    envelope = load_envelope(tmp_path / "envelope.json", policy, checkpoint)
    requests = {}
    for scope in ("acquisition", "phase_preparation", "one_touch"):
        path = tmp_path / f"{scope}.json"
        write_canonical(path, _request_value(tmp_path, checkpoint, envelope, scope))
        requests[scope] = load_request(
            path, scope=scope, policy=policy, checkpoint=checkpoint, envelope=envelope
        )
    acquisition = requests["acquisition"]
    phase = requests["phase_preparation"]
    one_touch = requests["one_touch"]
    acquisition_records = acquisition.records
    phase_records = phase.records
    one_touch_records = one_touch.records
    assert tuple(
        derive_scope_commands(envelope, requests[scope])
        for scope in ("acquisition", "phase_preparation", "one_touch")
    ) == (
        (
            (
                acquisition.interpreter.path,
                acquisition_records.acquirer.path,
                "--contract-manifest",
                acquisition_records.contract_manifest.path,
                "--availability-evidence",
                acquisition_records.availability_evidence.path,
                "--output-root",
                acquisition_records.source_root.path,
                "--report-dir",
                acquisition_records.report_root.path,
                "--forbidden-root",
                acquisition.forbidden_roots[0],
                "--forbidden-root",
                acquisition.forbidden_roots[1],
                "--execute",
                "--validate-complete",
            ),
            (
                acquisition.interpreter.path,
                acquisition_records.acquirer.path,
                "--contract-manifest",
                acquisition_records.contract_manifest.path,
                "--availability-evidence",
                acquisition_records.availability_evidence.path,
                "--output-root",
                acquisition_records.source_root.path,
                "--report-dir",
                acquisition_records.report_root.path,
                "--forbidden-root",
                acquisition.forbidden_roots[0],
                "--forbidden-root",
                acquisition.forbidden_roots[1],
                "--verify-eligible",
            ),
        ),
        (
            (
                phase.interpreter.path,
                phase_records.phase_wrapper.path,
                "--acquirer",
                phase_records.acquirer.path,
                "--source-root",
                phase_records.source_root.path,
                "--source-report",
                phase_records.source_report.path,
                "--forbidden-root",
                phase.forbidden_roots[0],
                "--forbidden-root",
                phase.forbidden_roots[1],
                "--contract-manifest",
                phase_records.contract_manifest.path,
                "--availability-evidence",
                phase_records.availability_evidence.path,
                "--preparer",
                phase_records.preparer.path,
                "--output-root",
                phase_records.phase_output.path,
            ),
        ),
        (
            (
                one_touch.interpreter.path,
                one_touch_records.prelock_script.path,
                "--config",
                one_touch_records.portfolio.path,
                "--contract-manifest",
                one_touch_records.contract_manifest.path,
                "--exchange",
                "binance",
                "--output-root",
                one_touch_records.prelock_output.path,
                "--warmup-raw-root",
                f"{one_touch_records.phase_output.path}/warmup/raw",
                "--warmup-feature-root",
                f"{one_touch_records.phase_output.path}/warmup/feature",
                "--train-raw-root",
                f"{one_touch_records.phase_output.path}/train/raw",
                "--train-feature-root",
                f"{one_touch_records.phase_output.path}/train/feature",
                "--purge-raw-root",
                f"{one_touch_records.phase_output.path}/purge/raw",
                "--purge-feature-root",
                f"{one_touch_records.phase_output.path}/purge/feature",
                "--validation-raw-root",
                f"{one_touch_records.phase_output.path}/validation/raw",
                "--validation-feature-root",
                f"{one_touch_records.phase_output.path}/validation/feature",
                "--embargo-raw-root",
                f"{one_touch_records.phase_output.path}/embargo/raw",
                "--embargo-feature-root",
                f"{one_touch_records.phase_output.path}/embargo/feature",
            ),
            (
                one_touch.interpreter.path,
                one_touch_records.historical_script.path,
                "--sealed-prelock-directory",
                one_touch_records.prelock_output.path,
                "--embargo-feature-root",
                f"{one_touch_records.phase_output.path}/embargo/feature",
                "--historical-evaluation-raw-root",
                f"{one_touch_records.phase_output.path}/historical_exposed_evaluation/raw",
                "--historical-evaluation-feature-root",
                f"{one_touch_records.phase_output.path}/historical_exposed_evaluation/feature",
                "--exchange",
                "binance",
                "--output-root",
                one_touch_records.historical_output.path,
            ),
        ),
    )


def test_strict_result_and_terminal_parsers_reject_extra_and_unsafe_shapes() -> None:
    artifact = {
        key: value
        for key, value in _file(Path("/tmp") / "artifact").items()
        if key not in {"st_uid", "st_gid"}
    }
    result = {
        "command_index": 0,
        "argv_sha256": _digest("argv"),
        "environment_sha256": _digest("env"),
        "return_code": 0,
        "stdout": {"kind": "stdout", **artifact},
        "stderr": {"kind": "stderr", **artifact},
        "validated_artifacts": [],
        "sealed_artifacts": [],
        "completed_utc": "2026-07-22T00:00:00Z",
        "extra": True,
    }
    with pytest.raises(TerminalPolicyError, match="target result has unexpected fields"):
        terminal_policy._parse_target_result(result)
    with pytest.raises(TerminalPolicyError, match="invalid artifact identity"):
        terminal_policy._parse_validated({**artifact, "kind": "unsafe", "nlink": 2})
    with pytest.raises(TerminalPolicyError, match="validated artifact has unexpected fields"):
        terminal_policy._parse_validated({**artifact, "kind": "prerequisite", "extra": True})
    with pytest.raises(TerminalPolicyError, match="unexpected fields"):
        terminal_policy._terminal_state({"kind": "SUCCEEDED", "failed_command_index": 0})


def test_control_paths_and_envelope_forbidden_roots_are_exactly_lexical(tmp_path: Path) -> None:
    for root in terminal_policy._FORBIDDEN_ROOTS:
        for candidate in (root, f"{root}/nested/control.json"):
            with pytest.raises(TerminalPolicyError, match="forbidden root"):
                validate_lexical_control_path(candidate)
    assert validate_lexical_control_path(tmp_path / "control.json") == str(
        tmp_path / "control.json"
    )

    policy, _checkpoint, value = _loaded_envelope(tmp_path)
    value["forbidden_roots"].reverse()
    path = tmp_path / "envelope.json"
    checkpoint = _checkpoint_for(path, policy, value)
    with pytest.raises(TerminalPolicyError, match="invalid forbidden roots"):
        load_envelope(path, policy, checkpoint)


def test_command_semantics_rejects_all_single_argument_changes_for_every_scope(
    tmp_path: Path,
) -> None:
    policy, checkpoint, _value = _loaded_envelope(tmp_path)
    envelope = load_envelope(tmp_path / "envelope.json", policy, checkpoint)
    for scope in TEST_SCOPES:
        request_path = tmp_path / f"{scope}.json"
        write_canonical(request_path, _request_value(tmp_path, checkpoint, envelope, scope))
        request = load_request(
            request_path, scope=scope, policy=policy, checkpoint=checkpoint, envelope=envelope
        )
        preflight = derive_command_preflight(envelope, request)
        for expected in preflight:
            assert (
                validate_command_semantics(
                    envelope, request, expected.command_index, expected.argv, request.environment
                )
                == expected
            )
            for position in range(len(expected.argv)):
                substituted = list(expected.argv)
                substituted[position] = f"tampered-{position}"
                with pytest.raises(TerminalPolicyError, match="command semantics mismatch"):
                    validate_command_semantics(
                        envelope,
                        request,
                        expected.command_index,
                        tuple(substituted),
                        request.environment,
                    )
            with pytest.raises(TerminalPolicyError, match="command semantics mismatch"):
                validate_command_semantics(
                    envelope,
                    request,
                    expected.command_index,
                    tuple(reversed(expected.argv)),
                    request.environment,
                )
            with pytest.raises(TerminalPolicyError, match="command semantics mismatch"):
                validate_command_semantics(
                    envelope,
                    request,
                    expected.command_index,
                    (*expected.argv, "--injected"),
                    request.environment,
                )
            with pytest.raises(TerminalPolicyError, match="command semantics mismatch"):
                validate_command_semantics(
                    envelope,
                    request,
                    expected.command_index,
                    expected.argv,
                    replace(request.environment, TZ="UTC+1"),
                )
        for index in (-1, len(preflight), "0"):
            with pytest.raises(TerminalPolicyError, match="invalid command index"):
                validate_command_semantics(
                    envelope, request, index, preflight[0].argv, request.environment
                )


def test_descriptor_opens_reject_temp_symlinked_ancestors(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    source = target / "file"
    source.write_bytes(b"safe")
    linked = tmp_path / "linked"
    os.symlink(target, linked)
    with pytest.raises(TerminalPolicyError, match="cannot open"):
        terminal_policy._regular_file(linked / "file", "symlinked file")
    with pytest.raises(TerminalPolicyError, match="cannot open"):
        terminal_policy._read_directory(
            terminal_policy.DirectoryIdentity(**_directory(linked)), "symlinked directory"
        )


def test_public_directory_fd_rejects_lexical_and_symlinked_roots(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    descriptor = terminal_policy.open_directory_fd(directory, "directory")
    try:
        assert stat.S_ISDIR(os.fstat(descriptor).st_mode)
    finally:
        os.close(descriptor)
    linked = tmp_path / "linked"
    os.symlink(directory, linked)
    with pytest.raises(TerminalPolicyError, match="cannot open"):
        terminal_policy.open_directory_fd(linked, "linked")
    for root in terminal_policy._FORBIDDEN_ROOTS:
        with pytest.raises(TerminalPolicyError, match="forbidden"):
            terminal_policy.open_directory_fd(root, "forbidden")


@pytest.mark.parametrize(
    "drift",
    ("content", "inode", "mode", "link_count", "receipt", "interpreter", "package_freeze"),
)
def test_validate_prelaunch_reads_real_bound_identities_and_rejects_drift(
    tmp_path: Path, drift: str
) -> None:
    def file_identity(name: str) -> terminal_policy.FileIdentity:
        path = tmp_path / "files" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        path.chmod(0o600)
        info = path.stat()
        return terminal_policy.FileIdentity(
            str(path),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            info.st_size,
            info.st_dev,
            info.st_ino,
            info.st_uid,
            info.st_gid,
            stat.S_IMODE(info.st_mode),
            info.st_nlink,
        )

    def directory_identity(name: str) -> terminal_policy.DirectoryIdentity:
        path = tmp_path / name
        path.mkdir()
        path.chmod(0o700)
        return terminal_policy.DirectoryIdentity(**_directory(path))

    current = directory_identity("current")
    accepted = directory_identity("accepted")
    evidence = directory_identity("evidence")
    outputs = directory_identity("outputs")
    receipts = (file_identity("receipt-current"), file_identity("receipt-accepted"))
    files = tuple(
        terminal_policy.FileBinding(role, file_identity(role))
        for role in terminal_policy._FILE_ROLES
    )
    current_python, current_freeze = (
        file_identity("current-python"),
        file_identity("current-freeze"),
    )
    accepted_python, accepted_freeze = (
        file_identity("accepted-python"),
        file_identity("accepted-freeze"),
    )
    interpreters = (
        terminal_policy.InterpreterBinding("current_python", current_python, current_freeze),
        terminal_policy.InterpreterBinding(
            "accepted_alpha_python", accepted_python, accepted_freeze
        ),
    )
    envelope = terminal_policy.LaunchEnvelope(
        "alpha_max_terminal_launch_envelope.v3",
        _digest("policy"),
        "1" * 40,
        "2" * 40,
        "3" * 40,
        (
            terminal_policy.RepositoryBinding("current_repository", current, "1" * 40, receipts[0]),
            terminal_policy.RepositoryBinding(
                "accepted_alpha_repository", accepted, "2" * 40, receipts[1]
            ),
        ),
        files,
        interpreters,
        terminal_policy.KeyBinding(
            _digest("key"), base64.b64encode(b"k" * 32).decode(), _digest("key")
        ),
        (),
        terminal_policy._FORBIDDEN_ROOTS,
        TEST_SCOPES,
    )
    bound = {item.role: item.file for item in files}

    def absent(leaf: str) -> terminal_policy.AbsentOutput:
        return terminal_policy.AbsentOutput(str(tmp_path / "outputs" / leaf), outputs, leaf, True)

    request = terminal_policy.ScopeRequest(
        "alpha_max_terminal_request.acquisition.v1",
        _digest("request"),
        "acquisition",
        _digest("checkpoint"),
        current_python,
        current,
        evidence,
        str(tmp_path / "evidence" / "terminal-authority.sock"),
        terminal_policy.Environment(
            str(tmp_path / "evidence"), "C.UTF-8", "C.UTF-8", "/usr/bin:/bin", "0", "1", "1", "UTC"
        ),
        terminal_policy._FORBIDDEN_ROOTS,
        terminal_policy.PublicationPaths("claim", "journal", "stdout", "stderr", "receipt"),
        (),
        terminal_policy.AcquisitionRecords(
            bound["acquirer"],
            bound["contract_manifest"],
            bound["availability_evidence"],
            absent("source"),
            absent("report"),
        ),
    )
    assert validate_prelaunch(envelope, request).files
    targets = {
        "content": bound["acquirer"].path,
        "inode": bound["contract_manifest"].path,
        "mode": bound["availability_evidence"].path,
        "link_count": bound["preparer"].path,
        "receipt": receipts[0].path,
        "interpreter": current_python.path,
        "package_freeze": current_freeze.path,
    }
    target = Path(targets[drift])
    if drift == "inode":
        replacement = target.with_name(f"{target.name}.replacement")
        replacement.write_bytes(target.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, target)
    elif drift == "mode":
        target.chmod(0o644)
    elif drift == "link_count":
        os.link(target, target.with_name(f"{target.name}.link"))
    else:
        target.write_bytes(b"drift")
    with pytest.raises(TerminalPolicyError):
        validate_prelaunch(envelope, request)


def _sealed_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical(path, value)
    path.chmod(0o444)


def _semantic_bundle(tmp_path: Path, kind: str) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / f"{kind}-{len(tuple(tmp_path.iterdir()))}"
    champion = "candidate-a"
    bucket = {"buckets": ["tiny"]}
    _sealed_write(root / "admission/train_liquidity_buckets.json", bucket)
    domain = "validation" if kind == "prelock_bundle" else "historical_exposed_evaluation"
    folds = 12 if kind == "prelock_bundle" else 10
    matrix = {
        "artifact_kind": "alpha_max_matrix_statuses.v1",
        "domain": domain,
        "engine_cell_count": 68,
        "physical_fold_run_count": 816 if kind == "prelock_bundle" else 680,
        "status_count": 84,
        "statuses": [{"engine_constructed": index < 68} for index in range(84)],
    }
    diagnostic = {
        "artifact_kind": "alpha_max_trend_liquidity_falsifier.v1",
        "bucket_contribution_usdt": {},
        "domain": domain,
        "fold_run_sha256s": [_digest(f"{kind}-{index}") for index in range(folds)],
        "nominal_cost_bps": 30,
        "rejection_reasons": [],
        "report_only": True,
        "row_id": "component_trend_1x",
        "selection_influence": False,
        "status": "complete",
        "symbol_contribution_usdt": {},
        "total_contribution_usdt": 0,
        "train_liquidity_buckets": bucket,
        "train_liquidity_buckets_sha256": hashlib.sha256(canonical_bytes(bucket)).hexdigest(),
    }
    selection = {
        "artifact_kind": (
            "alpha_max_prelock_selection.v2"
            if kind == "prelock_bundle"
            else "alpha_max_historical_report_ranking.v2"
        ),
        "decisions": [{} for _ in range(17)],
        "historical_evaluation_leader": champion,
        "prelock_champion": champion,
        "ranked_candidate_ids": [champion],
        "role": "prelock_selection" if kind == "prelock_bundle" else "historical_report",
        "scaling_attributions": [{}, {}],
        "selected_candidate_id": champion,
    }
    terminal = {
        "confirmation_status": "confirmed",
        "historical_evaluation_leader": champion,
        "historical_exposure_status": "report_only",
        "incumbent_comparison_status": "not_run",
        "leader_differs_from_prelock_champion": False,
        "prelock_champion": champion,
        "requires_fresh_confirmation": False,
        "selected_candidate_id": champion,
        "terminal_outcome": "complete",
    }
    if kind == "prelock_bundle":
        result = {
            "artifact_kind": "alpha_max_prelock_process_result.v1",
            "engine_cell_count": 68,
            "failure_reasons": [],
            "physical_fold_run_count": 816,
            "prelock_champion": champion,
            "selected_candidate_id": champion,
            "status": "complete",
            "terminal_outcome": "complete",
        }
        _sealed_write(root / "run/prelock_result.json", result)
        _sealed_write(root / "selection/prelock.json", selection)
        _sealed_write(root / "terminal/prelock.json", terminal)
        diagnostic_path = "diagnostics/validation/trend_liquidity_falsifier.json"
        seal = {"prelock_champion": champion}
    else:
        result = {
            "artifact_kind": "alpha_max_historical_process_result.v1",
            "confirmation_status": "confirmed",
            "engine_cell_count": 68,
            "failure_reasons": [],
            "historical_evaluation_leader": champion,
            "historical_exposure_status": "report_only",
            "physical_fold_run_count": 680,
            "prelock_champion": champion,
            "requires_fresh_confirmation": False,
            "selected_candidate_id": champion,
            "status": "complete_report_only",
            "terminal_outcome": "complete",
        }
        _sealed_write(root / "report/historical_result.json", result)
        _sealed_write(root / "selection/historical_ranking.json", selection)
        _sealed_write(root / "terminal/historical.json", terminal)
        diagnostic_path = "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json"
        seal = {}
    _sealed_write(root / "status/matrix.json", matrix)
    _sealed_write(root / diagnostic_path, diagnostic)
    return root, seal


def _semantic_readback(root: Path, kind: str, seal: dict) -> str:
    root_fd = terminal_policy.open_directory_fd(root, f"{kind} root")
    try:
        tree = terminal_policy._walk_tree_at(root_fd, kind)
        return terminal_policy._semantic_bundle_readback(root_fd, str(root), kind, seal, dict(tree))
    finally:
        os.close(root_fd)


def test_semantic_bundle_readback_accepts_and_rejects_cross_bindings(tmp_path: Path) -> None:
    for kind in ("prelock_bundle", "historical_bundle"):
        root, seal = _semantic_bundle(tmp_path, kind)
        assert _semantic_readback(root, kind, seal)
        result = root / (
            "run/prelock_result.json"
            if kind == "prelock_bundle"
            else "report/historical_result.json"
        )
        value = json.loads(result.read_text())
        result.chmod(0o644)
        value["selected_candidate_id"] = "other"
        write_canonical(result, value)
        result.chmod(0o444)
        with pytest.raises(TerminalPolicyError, match="outcome/readback mismatch"):
            _semantic_readback(root, kind, seal)
    root, seal = _semantic_bundle(tmp_path, "prelock_bundle")
    matrix = root / "status/matrix.json"
    value = json.loads(matrix.read_text())
    matrix.chmod(0o644)
    value["statuses"] = value["statuses"][:-1]
    write_canonical(matrix, value)
    matrix.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="matrix mismatch"):
        _semantic_readback(root, "prelock_bundle", seal)
    root, seal = _semantic_bundle(tmp_path, "historical_bundle")
    diagnostic = root / "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json"
    value = json.loads(diagnostic.read_text())
    diagnostic.chmod(0o644)
    value["train_liquidity_buckets_sha256"] = _digest("wrong")
    write_canonical(diagnostic, value)
    diagnostic.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="observability mismatch"):
        _semantic_readback(root, "historical_bundle", seal)
    root, seal = _semantic_bundle(tmp_path, "prelock_bundle")
    terminal = root / "terminal/prelock.json"
    value = json.loads(terminal.read_text())
    terminal.chmod(0o644)
    value["terminal_outcome"] = "other"
    write_canonical(terminal, value)
    terminal.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="outcome/readback mismatch"):
        _semantic_readback(root, "prelock_bundle", seal)
    root, seal = _semantic_bundle(tmp_path, "historical_bundle")
    matrix = root / "status/matrix.json"
    value = json.loads(matrix.read_text())
    matrix.chmod(0o644)
    value["physical_fold_run_count"] = 1
    write_canonical(matrix, value)
    matrix.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="matrix mismatch"):
        _semantic_readback(root, "historical_bundle", seal)


def _sealed_bundle(tmp_path: Path, kind: str) -> Path:
    root, semantic_seal = _semantic_bundle(
        tmp_path / f"sealed-{kind}-{len(list(tmp_path.iterdir()))}", kind
    )
    required = (
        (
            "admission/train.json",
            "admission/train_computation.json",
            "allocation/train_fit.json",
            "allocation/train_validation_refit.json",
            "inputs/config.json",
            "inputs/contract_manifest.json",
            "inputs/prior_trial_inventory.json",
            "trial/ledger.json",
        )
        if kind == "prelock_bundle"
        else ("binding/prelock_seal.json",)
    )
    for relative in required:
        _sealed_write(root / relative, {"path": relative})
    groups = (
        (
            ("manifests/validation_train_fit", 17),
            ("manifests/prelock_final_refit", 17),
            ("capsules/validation_train_fit/a", 204),
            ("capsules/prelock_final_refit/a", 17),
            ("evidence/validation/cells/a", 68),
            ("evidence/validation/rows", 816),
        )
        if kind == "prelock_bundle"
        else (
            ("capsules/prelock_final_refit/a", 153),
            ("evidence/historical_exposed_evaluation/cells/a", 68),
            ("evidence/historical_exposed_evaluation/rows", 680),
        )
    )
    for parent, count in groups:
        for index in range(count):
            _sealed_write(root / parent / f"{index}.json", {"i": index})
    entries = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "byte_count": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file()
    ]
    entries.sort(key=lambda entry: entry["relative_path"])
    if kind == "prelock_bundle":
        seal = {
            "artifact_count": len(entries),
            "artifact_kind": "alpha_max_immutable_prelock_seal.v1",
            "artifacts": entries,
            "historical_evaluation_inputs_included": False,
            "immutable": True,
            "inventory_sha256": hashlib.sha256(canonical_bytes(entries)).hexdigest(),
            "prelock_champion": semantic_seal["prelock_champion"],
            "selected_candidate_id": semantic_seal["prelock_champion"],
        }
    else:
        seal = {
            "artifact_kind": "alpha_max_append_only_historical_package.v1",
            "completion_id": "historical_exposed_evaluation",
            "historical_artifacts": entries,
            "immutable": True,
            "prelock_seal_sha256": _digest("prelock-seal"),
            "prelock_snapshot_sha256": _digest("prelock-snapshot"),
        }
    _sealed_write(root / "SEALED.json", seal)
    for path in (root, *(item for item in root.rglob("*") if item.is_dir())):
        path.chmod(0o555)
    return root


def _refresh_sealed_inventory(root: Path, kind: str) -> None:
    sealed = root / "SEALED.json"
    root.chmod(0o755)
    sealed.chmod(0o644)
    value = json.loads(sealed.read_text())
    entries = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "byte_count": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file() and path.name != "SEALED.json"
    ]
    entries.sort(key=lambda entry: entry["relative_path"])
    key = "artifacts" if kind == "prelock_bundle" else "historical_artifacts"
    value[key] = entries
    if kind == "prelock_bundle":
        value["artifact_count"] = len(entries)
        value["inventory_sha256"] = hashlib.sha256(canonical_bytes(entries)).hexdigest()
    write_canonical(sealed, value)
    sealed.chmod(0o444)


def _sealed_tree(root: Path, kind: str, inventory_key: str) -> terminal_policy.SealedArtifact:
    root_fd = terminal_policy.open_directory_fd(root, f"{kind} root")
    try:
        return terminal_policy._sealed_tree_at(root_fd, str(root), kind, inventory_key)[0]
    finally:
        os.close(root_fd)


def test_sealed_tree_accepts_and_rejects_inventory_cardinality_and_seal_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for kind, key in (
        ("prelock_bundle", "artifacts"),
        ("historical_bundle", "historical_artifacts"),
    ):
        root = _sealed_bundle(tmp_path, kind)
        assert _sealed_tree(root, kind, key).kind == kind
    root = _sealed_bundle(tmp_path, "prelock_bundle")
    monkeypatch.setattr(
        terminal_policy,
        "_semantic_bundle_readback",
        lambda root_fd, root_path, kind, seal, enumerated: "readback",
    )
    extra = root / "extra.json"
    root.chmod(0o755)
    _sealed_write(extra, {})
    with pytest.raises(TerminalPolicyError, match="inventory does not cover tree"):
        _sealed_tree(root, "prelock_bundle", "artifacts")
    extra.unlink()
    sealed = root / "SEALED.json"
    sealed.chmod(0o644)
    value = json.loads(sealed.read_text())
    value["artifact_count"] -= 1
    write_canonical(sealed, value)
    sealed.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="seal binding mismatch"):
        _sealed_tree(root, "prelock_bundle", "artifacts")
    root = _sealed_bundle(tmp_path, "prelock_bundle")
    sealed = root / "SEALED.json"
    root.chmod(0o755)
    sealed.chmod(0o644)
    value = json.loads(sealed.read_text())
    value["artifacts"].pop()
    write_canonical(sealed, value)
    sealed.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="inventory does not cover tree"):
        _sealed_tree(root, "prelock_bundle", "artifacts")
    root = _sealed_bundle(tmp_path, "prelock_bundle")
    sealed = root / "SEALED.json"
    root.chmod(0o755)
    sealed.chmod(0o644)
    value = json.loads(sealed.read_text())
    value["inventory_sha256"] = _digest("wrong")
    write_canonical(sealed, value)
    sealed.chmod(0o444)
    root.chmod(0o555)
    with pytest.raises(TerminalPolicyError, match="inventory hash mismatch"):
        _sealed_tree(root, "prelock_bundle", "artifacts")
    root = _sealed_bundle(tmp_path, "prelock_bundle")
    row = root / "evidence/validation/rows/0.json"
    row.parent.chmod(0o755)
    row.unlink()
    _refresh_sealed_inventory(root, "prelock_bundle")
    with pytest.raises(TerminalPolicyError, match="cardinality mismatch"):
        _sealed_tree(root, "prelock_bundle", "artifacts")
    root = _sealed_bundle(tmp_path, "historical_bundle")
    root.chmod(0o755)
    (root / "admission/train_liquidity_buckets.json").chmod(0o644)
    with pytest.raises(TerminalPolicyError, match="inventory identity drift"):
        _sealed_tree(root, "historical_bundle", "historical_artifacts")
    (root / "admission/train_liquidity_buckets.json").chmod(0o444)
    sealed = root / "SEALED.json"
    sealed.chmod(0o644)
    value = json.loads(sealed.read_text())
    value["completion_id"] = "wrong"
    write_canonical(sealed, value)
    sealed.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="historical seal binding mismatch"):
        _sealed_tree(root, "historical_bundle", "historical_artifacts")


def _identity(path: Path) -> terminal_policy.FileIdentity:
    info = path.stat()
    return terminal_policy.FileIdentity(
        str(path),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        info.st_size,
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
    )


def _prerequisite(kind: str, path: Path) -> terminal_policy.PrerequisiteRecord:
    identity = _identity(path)
    return terminal_policy.PrerequisiteRecord(
        kind,
        identity.path,
        identity.sha256,
        identity.byte_count,
        identity.st_dev,
        identity.st_ino,
        identity.mode,
        identity.nlink,
    )


def _directory_identity(path: Path) -> terminal_policy.DirectoryIdentity:
    info = path.stat()
    return terminal_policy.DirectoryIdentity(
        str(path), info.st_dev, info.st_ino, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)
    )


def _one_touch_phase_fixture(
    tmp_path: Path,
) -> tuple[object, object, Path, Path, Path]:
    _phase_envelope, phase_request, phase_files = _phase_records_fixture(tmp_path / "prepared")
    phase_records = phase_request.records
    assert isinstance(phase_records, terminal_policy.PhaseRecords)
    phase = phase_files["output"]
    handoff_path = phase_files["handoff"]
    manifest_path = phase / "preparation_manifest.json"

    parent = tmp_path / "outputs"
    parent.mkdir()
    portfolio = tmp_path / "portfolio.json"
    prelock_script = tmp_path / "prelock.py"
    historical_script = tmp_path / "historical.py"
    for path in (portfolio, prelock_script, historical_script):
        path.write_text(path.name)

    records = terminal_policy.OneTouchRecords(
        _identity(portfolio),
        phase_records.contract_manifest,
        _identity(prelock_script),
        _identity(historical_script),
        _directory_identity(phase),
        terminal_policy.AbsentOutput(
            str(parent / "prelock"),
            _directory_identity(parent),
            "prelock",
            True,
        ),
        terminal_policy.AbsentOutput(
            str(parent / "historical"),
            _directory_identity(parent),
            "historical",
            True,
        ),
    )
    request = SimpleNamespace(
        scope="one_touch",
        records=records,
        interpreter=phase_request.interpreter,
        forbidden_roots=phase_request.forbidden_roots,
        prerequisites=(
            _prerequisite("phase_handoff_receipt", handoff_path),
            _prerequisite("preparation_manifest", manifest_path),
        ),
    )
    return (
        SimpleNamespace(scope_order=("one_touch",)),
        request,
        phase,
        handoff_path,
        parent,
    )


def _bind_historical_prelock(prelock: Path, historical: Path) -> None:
    historical.chmod(0o755)
    for path in (
        historical / "binding/prelock_seal.json",
        historical / "admission/train_liquidity_buckets.json",
        historical / "SEALED.json",
    ):
        path.parent.chmod(0o755)
        path.chmod(0o644)
    (historical / "binding/prelock_seal.json").write_bytes((prelock / "SEALED.json").read_bytes())
    (historical / "admission/train_liquidity_buckets.json").write_bytes(
        (prelock / "admission/train_liquidity_buckets.json").read_bytes()
    )
    seal_path = historical / "SEALED.json"
    seal = json.loads(seal_path.read_text())
    seal["prelock_seal_sha256"] = hashlib.sha256((prelock / "SEALED.json").read_bytes()).hexdigest()
    seal["prelock_snapshot_sha256"] = terminal_policy._snapshot_digest(prelock)
    write_canonical(seal_path, seal)
    _refresh_sealed_inventory(historical, "historical_bundle")
    for path in (
        historical / "binding/prelock_seal.json",
        historical / "admission/train_liquidity_buckets.json",
        historical / "SEALED.json",
    ):
        path.chmod(0o444)
    for path in (historical, *(item for item in historical.rglob("*") if item.is_dir())):
        path.chmod(0o555)


def test_one_touch_completed_commands_bind_real_seals_and_reject_replacements(
    tmp_path: Path,
) -> None:
    envelope, request, _phase, _handoff, parent = _one_touch_phase_fixture(tmp_path)
    prelock = _sealed_bundle(parent, "prelock_bundle")
    historical = _sealed_bundle(parent, "historical_bundle")
    prelock.chmod(0o755)
    historical.chmod(0o755)
    prelock.rename(parent / "prelock")
    historical.rename(parent / "historical")
    prelock, historical = parent / "prelock", parent / "historical"
    prelock.chmod(0o555)
    _bind_historical_prelock(prelock, historical)

    command_zero = terminal_policy.validate_completed_command(envelope, request, 0)
    command_one = terminal_policy.validate_completed_command(envelope, request, 1, command_zero)
    assert command_zero.sealed_artifacts[0].kind == "prelock_bundle"
    assert command_one.sealed_artifacts[0].kind == "historical_bundle"

    seal = prelock / "SEALED.json"
    replacement = prelock / "SEALED.replacement"
    prelock.chmod(0o755)
    replacement.write_bytes(seal.read_bytes())
    replacement.chmod(0o444)
    os.replace(replacement, seal)
    prelock.chmod(0o555)
    with pytest.raises(TerminalPolicyError, match="prelock sealed receipt changed"):
        terminal_policy.validate_completed_command(envelope, request, 1, command_zero)
    prelock.chmod(0o755)

    prelock.rename(parent / "moved-prelock")
    with pytest.raises(TerminalPolicyError):
        terminal_policy.validate_completed_command(envelope, request, 1, command_zero)


def test_one_touch_preparation_manifest_and_handoff_are_real_and_fail_closed(
    tmp_path: Path,
) -> None:
    _envelope, request, phase, handoff_path, _parent = _one_touch_phase_fixture(tmp_path)
    assert terminal_policy._validate_preparation_manifest(request)[0]

    handoff_path.chmod(0o644)
    handoff = json.loads(handoff_path.read_text())
    handoff["preparer_result"]["output_root"] = str(phase / "wrong")
    write_canonical(handoff_path, handoff)
    handoff_path.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="phase handoff receipt mismatch"):
        terminal_policy._validate_preparation_manifest(request)
    _envelope, request, _phase, handoff_path, _parent = _one_touch_phase_fixture(
        tmp_path / "wrong-output-identity"
    )
    assert terminal_policy._validate_preparation_manifest(request)[0]
    handoff_path.chmod(0o644)
    handoff = json.loads(handoff_path.read_text())
    handoff["output_root_identity"]["st_ino"] += 1
    write_canonical(handoff_path, handoff)
    handoff_path.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="phase handoff receipt mismatch"):
        terminal_policy._validate_preparation_manifest(request)

    _envelope, request, _phase, handoff_path, _parent = _one_touch_phase_fixture(
        tmp_path / "digest-mismatch"
    )
    assert terminal_policy._validate_preparation_manifest(request)[0]
    handoff_path.chmod(0o644)
    handoff = json.loads(handoff_path.read_text())
    handoff["output_manifest_sha256"] = _digest("wrong-manifest")
    write_canonical(handoff_path, handoff)
    handoff_path.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="phase handoff receipt mismatch"):
        terminal_policy._validate_preparation_manifest(request)

    _envelope, request, _phase, _handoff, _parent = _one_touch_phase_fixture(
        tmp_path / "prerequisite-drift"
    )
    assert terminal_policy._validate_preparation_manifest(request)[0]
    request.prerequisites = (
        request.prerequisites[0],
        replace(request.prerequisites[1], st_ino=request.prerequisites[1].st_ino + 1),
    )
    with pytest.raises(TerminalPolicyError, match="preparation_manifest prerequisite drift"):
        terminal_policy._validate_preparation_manifest(request)


_FIXTURE_SYMBOLS = (
    "ADAUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "TONUSDT",
    "TRXUSDT",
    "XRPUSDT",
)
_FIXTURE_PHASE_INTERVALS = (
    ("warmup", "2022-12-31T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("train", "2024-01-01T00:00:00Z", "2025-06-01T00:00:00Z"),
    ("purge", "2025-06-01T00:00:00Z", "2025-06-08T00:00:00Z"),
    ("validation", "2025-06-08T00:00:00Z", "2025-08-31T00:00:00Z"),
    ("embargo", "2025-08-31T00:00:00Z", "2025-09-07T00:00:00Z"),
    (
        "historical_exposed_evaluation",
        "2025-09-07T00:00:00Z",
        "2026-07-01T00:00:00Z",
    ),
)
_FIXTURE_PHASES = tuple(interval[0] for interval in _FIXTURE_PHASE_INTERVALS)


def test_fixture_protocol_constants_match_the_approved_policy_contract() -> None:
    assert _FIXTURE_SYMBOLS == terminal_policy._SYMBOLS
    assert _FIXTURE_PHASE_INTERVALS == terminal_policy._PHASE_INTERVALS


def _fixture_write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _fixture_official_receipt(
    path: Path, requested_url: str, query: dict[str, str] | None = None
) -> str:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    _fixture_write_canonical(
        path.with_name(path.name + ".receipt.json"),
        {
            "schema": "official_request_receipt.v1",
            "requested_url": requested_url,
            "final_url": requested_url,
            "final_host": requested_url.split("/")[2].split("?")[0],
            "query": query or {},
            "retrieved_at_utc": "2025-06-09T00:00:00Z",
            "byte_count": len(payload),
            "sha256": digest,
        },
    )
    return digest


def _fixture_partition_path(relative: str) -> str:
    return "partitions/" + hashlib.sha256(relative.encode()).hexdigest() + ".json"


def _fixture_write_partition(
    report: Path,
    relative: str,
    source_sha256: str,
    output_sha256: str,
    rows: int,
    start_ms: int,
    end_ms: int,
    code_sha256: str,
    page_hashes: list[str],
    input_carry_close: float | None = None,
    output_carry_close: float | None = None,
) -> None:
    _fixture_write_canonical(
        report / _fixture_partition_path(relative),
        {
            "schema": "alpha_max_partition_receipt.v2",
            "path": relative,
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
            "rows": rows,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "input_carry_close": input_carry_close,
            "output_carry_close": output_carry_close,
            "derivation_version": "alpha-max-binance-ohlcv-v4",
            "code_sha256": code_sha256,
            "page_hashes": page_hashes,
        },
    )


def _phase_records_fixture(
    tmp_path: Path, *, ton_owned_settlements: int | None = None
) -> tuple[object, object, dict[str, Path]]:
    parent = tmp_path / "phase-parent"
    source, report, output = parent / "source", parent / "report", parent / "phase"
    parent.mkdir(parents=True)
    source.mkdir()
    report.mkdir()
    output.mkdir()
    files = {}
    for name in (
        "phase_wrapper.py",
        "acquirer.py",
        "contract.json",
        "availability.json",
        "preparer.py",
        "python",
        "checkpoint.json",
        "alignment.json",
    ):
        path = tmp_path / name
        path.write_text(name)
        files[name] = path
    if ton_owned_settlements is not None and ton_owned_settlements not in (999, 1000):
        raise ValueError("TON fixture requires 999 or 1000 owned settlements")
    start_utc, end_utc = "2025-06-07T00:00:00Z", "2025-06-09T00:00:00Z"
    start_ms, end_ms = 1_749_254_400_000, 1_749_427_200_000
    ton_feature_end_utc = (
        end_utc
        if ton_owned_settlements is None
        else ("2025-11-20T12:00:00Z" if ton_owned_settlements == 999 else "2025-11-20T16:00:00Z")
    )
    contract = {
        "schema_version": "alpha_max_contract_manifest.v2",
        "exchange": "binance",
        "records": [
            {
                "symbol": symbol,
                "market_type": "perpetual",
                "linear": True,
                "inverse": False,
                "quote_asset": "USDT",
                "margin_asset": "USDT",
                "settle_asset": "USDT",
                "volume_unit": "base_asset",
                "contract_multiplier": 1.0,
                "raw_availability_start_utc": start_utc,
                "raw_availability_end_utc": end_utc,
                "feature_availability_start_utc": start_utc,
                "feature_availability_end_utc": (
                    ton_feature_end_utc if symbol == "TONUSDT" else end_utc
                ),
            }
            for symbol in _FIXTURE_SYMBOLS
        ],
    }
    availability = {
        "raw": {
            "availability_start_by_symbol": dict.fromkeys(_FIXTURE_SYMBOLS, start_utc),
            "availability_end_by_symbol": dict.fromkeys(_FIXTURE_SYMBOLS, end_utc),
        },
        "feature": {
            "availability_start_by_symbol": dict.fromkeys(_FIXTURE_SYMBOLS, start_utc),
            "availability_end_by_symbol": {
                symbol: ton_feature_end_utc if symbol == "TONUSDT" else end_utc
                for symbol in _FIXTURE_SYMBOLS
            },
        },
    }
    _fixture_write_canonical(files["contract.json"], contract)
    _fixture_write_canonical(files["availability.json"], availability)
    contract_digest = hashlib.sha256(files["contract.json"].read_bytes()).hexdigest()
    availability_digest = hashlib.sha256(files["availability.json"].read_bytes()).hexdigest()
    plan = {
        "schema": "alpha_max_official_acquisition_plan.v4",
        "source_eligible": False,
        "symbols": list(_FIXTURE_SYMBOLS),
        "months": [],
        "contract_sha256": contract_digest,
        "availability_evidence_sha256": availability_digest,
        "storage_contract": {
            "host_reserve_path": "/mnt/c",
            "host_reserve_bytes": 21_474_836_480,
            "max_live_archives": 1,
            "archive_retention": "retired_after_double_derivation",
        },
    }
    _fixture_write_canonical(report / "plan.json", plan)
    run_id = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    owner = {
        "schema": "alpha_max_owned_roots.v2",
        "run_id": run_id,
        "output_path": str(source),
        "report_path": str(report),
        "output_parent_identity": [parent.stat().st_dev, parent.stat().st_ino],
        "report_parent_identity": [parent.stat().st_dev, parent.stat().st_ino],
        "output_identity": [source.stat().st_dev, source.stat().st_ino],
        "report_identity": [report.stat().st_dev, report.stat().st_ino],
        "uid": os.getuid(),
        "contract_sha256": contract_digest,
        "availability_evidence_sha256": availability_digest,
        "derivation_version": "alpha-max-binance-ohlcv-v4",
        "code_sha256": hashlib.sha256(files["acquirer.py"].read_bytes()).hexdigest(),
    }
    _fixture_write_canonical(source / ".alpha_max_owner.json", owner)
    _fixture_write_canonical(report / ".alpha_max_owner.json", owner)
    provenance = report / "provenance"
    _fixture_write_canonical(provenance / "contract_manifest.json", contract)
    _fixture_write_canonical(provenance / "availability_evidence.json", availability)
    exchange_path = provenance / "exchangeInfo.json"
    exchange_path.write_bytes(
        b'{ "symbols" : ['
        + b",".join(
            f'{{"symbol":"{symbol}"}}'.encode()
            for symbol in _FIXTURE_SYMBOLS
            if symbol != "TONUSDT"
        )
        + b"] }\n"
    )
    exchange_digest = _fixture_official_receipt(
        exchange_path, "https://fapi.binance.com/fapi/v1/exchangeInfo"
    )
    journal = report / "acquisition.journal.jsonl"
    journal.write_bytes(b'{"event":"acquired","run":"bounded-fixture"}\n')
    output_inventory: list[str] = []
    required_report = {
        "provenance/contract_manifest.json",
        "provenance/availability_evidence.json",
        "provenance/exchangeInfo.json",
        "provenance/exchangeInfo.json.receipt.json",
        "acquisition.journal.jsonl",
    }
    storage_contract = plan["storage_contract"]
    raw_total = funding_total = 0
    for symbol in _FIXTURE_SYMBOLS:
        raw_relative = f"market_ohlcv_1s/binance/{symbol}/2025-06.parquet"
        raw_output = source / raw_relative
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_bytes(f"raw:{symbol}:2025-06".encode())
        archive_relative = f"provenance/archives/{symbol}/{symbol}-aggTrades-2025-06.zip"
        archive = report / archive_relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(f"archive:{symbol}:2025-06".encode())
        archive_url = (
            "https://data.binance.vision/data/futures/um/monthly/aggTrades/"
            f"{symbol}/{symbol}-aggTrades-2025-06.zip"
        )
        _fixture_official_receipt(archive, archive_url)
        archive_receipt_path = archive.with_name(archive.name + ".receipt.json")
        archive_receipt_sha = hashlib.sha256(archive_receipt_path.read_bytes()).hexdigest()
        archive_receipt = json.loads(archive_receipt_path.read_text())
        archive.unlink()
        checksum = archive.with_name(archive.name + ".CHECKSUM")
        checksum.write_text(f"{archive_receipt['sha256']}  {archive.name}\n")
        checksum_payload_sha = _fixture_official_receipt(checksum, archive_url + ".CHECKSUM")
        checksum_receipt_sha = hashlib.sha256(
            checksum.with_name(checksum.name + ".receipt.json").read_bytes()
        ).hexdigest()
        _fixture_write_partition(
            report,
            raw_relative,
            archive_receipt["sha256"],
            hashlib.sha256(raw_output.read_bytes()).hexdigest(),
            (end_ms - start_ms) // 1000,
            start_ms,
            end_ms,
            owner["code_sha256"],
            [checksum_payload_sha, archive_receipt["sha256"]],
            output_carry_close=1.0,
        )
        partition = json.loads((report / _fixture_partition_path(raw_relative)).read_text())
        evidence_root = provenance / "archive-evidence" / symbol
        evidence_prefix = evidence_root / "2025-06"
        derivation = {
            "schema": "alpha_max_archive_derivation_receipt.v1",
            "output_path": raw_relative,
            "output_sha256": partition["output_sha256"],
            "output_byte_count": raw_output.stat().st_size,
            "rows": partition["rows"],
            "start_ms": start_ms,
            "end_ms": end_ms,
            "input_carry_close": None,
            "output_carry_close": 1.0,
            "archive_url": archive_url,
            "archive_member": f"{symbol}-aggTrades-2025-06.csv",
            "archive_sha256": archive_receipt["sha256"],
            "archive_byte_count": archive_receipt["byte_count"],
            "archive_request_receipt_sha256": archive_receipt_sha,
            "checksum_payload_sha256": partition["source_sha256"],
            "checksum_request_receipt_sha256": checksum_receipt_sha,
            "partition_receipt_sha256": hashlib.sha256(canonical_bytes(partition)).hexdigest(),
            "prior_derivation_receipt_sha256": None,
            "derivation_version": "alpha-max-binance-ohlcv-v4",
            "code_sha256": owner["code_sha256"],
        }
        intent = {
            "schema": "alpha_max_archive_retirement_intent.v1",
            "derivation_receipt_sha256": hashlib.sha256(canonical_bytes(derivation)).hexdigest(),
            "partition_receipt_sha256": derivation["partition_receipt_sha256"],
            "archive_request_receipt_sha256": archive_receipt_sha,
            "archive_relative_path": archive_relative,
            "archive_sha256": archive_receipt["sha256"],
            "archive_byte_count": archive_receipt["byte_count"],
            "output_path": raw_relative,
            "output_sha256": partition["output_sha256"],
        }
        deletion = {
            "schema": "alpha_max_archive_deletion_receipt.v1",
            "retirement_intent_sha256": hashlib.sha256(canonical_bytes(intent)).hexdigest(),
            "derivation_receipt_sha256": intent["derivation_receipt_sha256"],
            "archive_relative_path": archive_relative,
            "archive_sha256": archive_receipt["sha256"],
            "archive_byte_count": archive_receipt["byte_count"],
            "archive_absent": True,
        }
        _fixture_write_canonical(evidence_prefix.with_suffix(".derivation.json"), derivation)
        _fixture_write_canonical(evidence_prefix.with_suffix(".retirement-intent.json"), intent)
        _fixture_write_canonical(evidence_prefix.with_suffix(".deletion.json"), deletion)
        output_inventory.append(raw_relative)
        required_report.update(
            {
                archive_relative + ".receipt.json",
                archive_relative + ".CHECKSUM",
                archive_relative + ".CHECKSUM.receipt.json",
                _fixture_partition_path(raw_relative),
                f"provenance/archive-evidence/{symbol}/2025-06.derivation.json",
                f"provenance/archive-evidence/{symbol}/2025-06.retirement-intent.json",
                f"provenance/archive-evidence/{symbol}/2025-06.deletion.json",
            }
        )
        interval = 14_400_000 if symbol == "TONUSDT" else 28_800_000
        feature_end_ms = (
            start_ms
            + (ton_owned_settlements if ton_owned_settlements is not None else 12) * interval
            if symbol == "TONUSDT"
            else end_ms
        )
        cursor = start_ms - 2 * interval if symbol == "TONUSDT" else start_ms
        settlements = list(range(start_ms, feature_end_ms, interval))
        funding_rows = (
            [{"symbol": symbol, "fundingTime": cursor, "fundingRate": "0.0001"}]
            if symbol == "TONUSDT"
            else []
        ) + [
            {"symbol": symbol, "fundingTime": settlement, "fundingRate": "0.0001"}
            for settlement in settlements
        ]
        page_hashes = []
        number = 0
        while cursor < feature_end_ms:
            number += 1
            query = {
                "symbol": symbol,
                "startTime": str(cursor),
                "endTime": str(feature_end_ms - 1),
                "limit": "1000",
            }
            page_rows = [row for row in funding_rows if row["fundingTime"] >= cursor][:1000]
            if symbol == "TONUSDT" and ton_owned_settlements is not None:
                assert len(page_rows) in (0, 1, 1000)
            page_relative = f"provenance/funding_pages/{symbol}/{number:06d}.json"
            page = report / page_relative
            _fixture_write_canonical(page, page_rows)
            page_sha = _fixture_official_receipt(
                page,
                "https://fapi.binance.com/fapi/v1/fundingRate?"
                + "&".join(f"{key}={value}" for key, value in query.items()),
                query,
            )
            page_hashes.append(page_sha)
            required_report.update({page_relative, page_relative + ".receipt.json"})
            if len(page_rows) < 1000:
                break
            next_cursor = page_rows[-1]["fundingTime"] + 1
            assert next_cursor == max(row["fundingTime"] for row in page_rows) + 1
            cursor = next_cursor
        normalized = [
            {
                "timestamp_ms": settlement,
                "source_timestamp_ms": settlement,
                "exchange": "binance",
                "symbol": symbol,
                "funding_rate": 0.0001,
            }
            for settlement in settlements
        ]
        for day_ms in range(start_ms, feature_end_ms, 86_400_000):
            owned = [
                row for row in normalized if day_ms <= row["timestamp_ms"] < day_ms + 86_400_000
            ]
            day = datetime.fromtimestamp(day_ms / 1000, UTC).strftime("%Y-%m-%d")
            funding_relative = (
                f"feature_points/exchange=binance/symbol={symbol}/date={day}/funding.parquet"
            )
            funding_output = source / funding_relative
            funding_output.parent.mkdir(parents=True, exist_ok=True)
            funding_output.write_bytes(f"funding:{symbol}:{day}".encode())
            _fixture_write_partition(
                report,
                funding_relative,
                hashlib.sha256(canonical_bytes(owned)).hexdigest(),
                hashlib.sha256(funding_output.read_bytes()).hexdigest(),
                len(owned),
                day_ms,
                day_ms + 86_400_000,
                owner["code_sha256"],
                page_hashes,
            )
            output_inventory.append(funding_relative)
            required_report.add(_fixture_partition_path(funding_relative))
            funding_total += len(owned)
        raw_total += (end_ms - start_ms) // 1000
    output_inventory.sort()
    artifact_paths = [
        *(f"output/{relative}" for relative in output_inventory),
        *(f"report/{relative}" for relative in sorted(required_report)),
    ]
    artifacts = [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                (
                    (source if relative.startswith("output/") else report)
                    / relative.split("/", 1)[1]
                ).read_bytes()
            ).hexdigest(),
        }
        for relative in artifact_paths
    ]
    archive_evidence_sha256 = hashlib.sha256(
        canonical_bytes(
            sorted(
                (
                    artifact
                    for artifact in artifacts
                    if artifact["path"].startswith("report/provenance/archive-evidence/")
                ),
                key=lambda artifact: artifact["path"],
            )
        )
    ).hexdigest()
    manifest = {
        "schema": "alpha_max_official_source_manifest.v5",
        "contract_sha256": contract_digest,
        "availability_evidence_sha256": availability_digest,
        "derivation_version": "alpha-max-binance-ohlcv-v4",
        "storage_contract": storage_contract,
        "archive_evidence_sha256": archive_evidence_sha256,
        "artifacts": artifacts,
    }
    manifest_path = report / "source_manifest.json"
    _fixture_write_canonical(manifest_path, manifest)
    receipt_path = report / "source_eligible_receipt.json"
    receipt = {
        "schema": "alpha_max_official_source_receipt.v4",
        "source_eligible": True,
        "raw_rows": raw_total,
        "funding_rows": funding_total,
        "contract_sha256": contract_digest,
        "availability_evidence_sha256": availability_digest,
        "derivation_version": "alpha-max-binance-ohlcv-v4",
        "code_sha256": owner["code_sha256"],
        "storage_contract": storage_contract,
        "archive_evidence_sha256": archive_evidence_sha256,
        "exchange_info_sha256": exchange_digest,
        "inventory_sha256": hashlib.sha256(canonical_bytes(output_inventory)).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "acquisition_journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
    }
    _fixture_write_canonical(receipt_path, receipt)

    def parse_utc(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)

    def format_utc(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    def intersection(
        left_start: datetime,
        left_end: datetime,
        right_start: datetime,
        right_end: datetime,
    ) -> tuple[datetime, datetime] | None:
        start, end = max(left_start, right_start), min(left_end, right_end)
        return (start, end) if start < end else None

    contract_by_symbol = {record["symbol"]: record for record in contract["records"]}
    phase_entries: list[dict[str, object]] = []
    output_directories = {output}
    for phase_name in _FIXTURE_PHASES:
        phase_directory = output / phase_name
        phase_directory.mkdir()
        output_directories.add(phase_directory)
    for relative in output_inventory:
        parts = relative.split("/")
        if relative.startswith("market_ohlcv_1s/"):
            root_kind, symbol = "raw", parts[2]
            month = parts[-1].removesuffix(".parquet")
            year, month_number = (int(value) for value in month.split("-"))
            source_start = datetime(year, month_number, 1, tzinfo=UTC)
            source_end = datetime(
                year + (month_number == 12),
                1 if month_number == 12 else month_number + 1,
                1,
                tzinfo=UTC,
            )
        else:
            root_kind = "feature"
            symbol = next(
                part.removeprefix("symbol=") for part in parts if part.startswith("symbol=")
            )
            date = next(part.removeprefix("date=") for part in parts if part.startswith("date="))
            source_start = datetime.fromisoformat(date).replace(tzinfo=UTC)
            source_end = source_start + timedelta(days=1)
        bounded = intersection(
            source_start,
            source_end,
            parse_utc(contract_by_symbol[symbol][f"{root_kind}_availability_start_utc"]),
            parse_utc(contract_by_symbol[symbol][f"{root_kind}_availability_end_utc"]),
        )
        assert bounded is not None
        bounded = intersection(
            *bounded,
            parse_utc(availability[root_kind]["availability_start_by_symbol"][symbol]),
            parse_utc(availability[root_kind]["availability_end_by_symbol"][symbol]),
        )
        assert bounded is not None
        source_path = source / relative
        source_bytes = source_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        for phase_name, phase_start, phase_end in _FIXTURE_PHASE_INTERVALS:
            owned = intersection(
                *bounded,
                parse_utc(phase_start),
                parse_utc(phase_end),
            )
            if owned is None:
                continue
            owned_start, owned_end = owned
            output_relative = (
                f"{phase_name}/raw/{relative}"
                if root_kind == "raw"
                else (f"{phase_name}/feature/{Path(relative).parent.as_posix()}/part-0.parquet")
            )
            output_path = output / output_relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_directory = output_path.parent
            while output_directory != output:
                output_directories.add(output_directory)
                output_directory = output_directory.parent
            output_bytes = canonical_bytes(
                {
                    "owned_end_utc": format_utc(owned_end),
                    "owned_start_utc": format_utc(owned_start),
                    "phase_id": phase_name,
                    "root_kind": root_kind,
                    "source_relative_path": relative,
                    "symbol": symbol,
                }
            )
            output_path.write_bytes(output_bytes)
            output_path.chmod(0o444)
            phase_entries.append(
                {
                    "phase_id": phase_name,
                    "root_kind": root_kind,
                    "symbol": symbol,
                    "owned_start_utc": format_utc(owned_start),
                    "owned_end_utc": format_utc(owned_end),
                    "source_relative_path": relative,
                    "source_sha256": source_sha256,
                    "source_byte_count": len(source_bytes),
                    "output_relative_path": output_relative,
                    "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                    "output_byte_count": len(output_bytes),
                    "output_row_count": max(1, int((owned_end - owned_start).total_seconds())),
                }
            )
    phase_entries.sort(key=lambda entry: str(entry["output_relative_path"]))
    assert phase_entries
    preparation_manifest = {
        "availability": availability,
        "availability_sha256_by_root_kind": {
            kind: hashlib.sha256(canonical_bytes(value)).hexdigest()
            for kind, value in availability.items()
        },
        "contract_manifest_schema_version": "alpha_max_contract_manifest.v2",
        "contract_manifest_sha256": contract_digest,
        "exchange": "binance",
        "file_count": len(phase_entries),
        "files": phase_entries,
        "phase_intervals": [
            {"phase_id": name, "start_utc": start, "end_utc": end}
            for name, start, end in _FIXTURE_PHASE_INTERVALS
        ],
        "schema_version": "alpha_max_phase_root_preparation_manifest.v1",
        "symbols": list(_FIXTURE_SYMBOLS),
    }
    preparation_path = output / "preparation_manifest.json"
    _sealed_write(preparation_path, preparation_manifest)

    prefix = ".phase.alpha_max_phase_preparation"
    lock_path = parent / f"{prefix}.lock"
    lock_path.write_text("locked")
    lock_path.chmod(0o600)
    source_snapshot = parent / f"{prefix}.source-snapshot"
    source_snapshot.mkdir()
    descriptor_path = parent / f"{prefix}.invocation.json"
    handoff_path = parent / f"{prefix}.handoff.json"
    invocation_inputs = parent / f"{prefix}.invocation-inputs"
    source_snapshot_data = {
        "source_root_identity": {"st_dev": source.stat().st_dev, "st_ino": source.stat().st_ino},
        "source_report_identity": {"st_dev": report.stat().st_dev, "st_ino": report.stat().st_ino},
        "source_eligible_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "acquisition_journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256((report / "plan.json").read_bytes()).hexdigest(),
        "source_owner_sha256": hashlib.sha256(
            (source / ".alpha_max_owner.json").read_bytes()
        ).hexdigest(),
        "report_owner_sha256": hashlib.sha256(
            (report / ".alpha_max_owner.json").read_bytes()
        ).hexdigest(),
        "source_manifest_artifact_map_sha256": hashlib.sha256(
            canonical_bytes(
                dict(sorted((entry["path"], entry["sha256"]) for entry in manifest["artifacts"]))
            )
        ).hexdigest(),
    }
    interpreter = _identity(files["python"])
    records = terminal_policy.PhaseRecords(
        _identity(files["phase_wrapper.py"]),
        _identity(files["acquirer.py"]),
        _directory_identity(source),
        _directory_identity(report),
        _identity(files["contract.json"]),
        _identity(files["availability.json"]),
        _identity(files["preparer.py"]),
        terminal_policy.AbsentOutput(str(output), _directory_identity(parent), "phase", True),
    )
    request = SimpleNamespace(
        scope="phase_preparation",
        records=records,
        interpreter=interpreter,
        forbidden_roots=("/forbidden-a", "/forbidden-b"),
        prerequisites=(
            _prerequisite("checkpoint_pin", files["checkpoint.json"]),
            _prerequisite("alignment_receipt", files["alignment.json"]),
            _prerequisite("source_eligible_receipt", receipt_path),
            _prerequisite("source_manifest", manifest_path),
            _prerequisite("source_journal", journal),
        ),
    )
    paths = {
        "acquirer": records.acquirer.path,
        "source_root": records.source_root.path,
        "source_report": records.source_report.path,
        "contract_manifest": records.contract_manifest.path,
        "availability_evidence": records.availability_evidence.path,
        "preparer": records.preparer.path,
        "output_root": records.phase_output.path,
        "raw_root": str(source_snapshot / "market_ohlcv_1s"),
        "feature_root": str(source_snapshot / "feature_points"),
        "invocation_descriptor": str(descriptor_path),
        "invocation_descriptor_stage": str(parent / f"{prefix}.invocation.stage.json"),
        "handoff_receipt": str(handoff_path),
        "handoff_receipt_stage": str(parent / f"{prefix}.handoff.stage.json"),
        "source_snapshot": str(source_snapshot),
        "source_snapshot_manifest": str(source_snapshot / "snapshot-manifest.json"),
        "source_snapshot_complete": str(source_snapshot / ".complete.json"),
        "invocation_inputs": str(invocation_inputs),
        "invocation_input_acquirer": str(invocation_inputs / "acquirer.py"),
        "invocation_input_contract_manifest": str(invocation_inputs / "contract_manifest.json"),
        "invocation_input_availability_evidence": str(
            invocation_inputs / "availability_evidence.json"
        ),
        "invocation_input_preparer": str(invocation_inputs / "preparer.py"),
        "invocation_lock": str(lock_path),
    }
    lock_info = lock_path.stat()
    verifier_argv = [
        interpreter.path,
        records.acquirer.path,
        "--contract-manifest",
        records.contract_manifest.path,
        "--availability-evidence",
        records.availability_evidence.path,
        "--output-root",
        records.source_root.path,
        "--report-dir",
        records.source_report.path,
        "--forbidden-root",
        "/forbidden-a",
        "--forbidden-root",
        "/forbidden-b",
        "--verify-eligible",
    ]
    preparer_argv = [
        interpreter.path,
        records.preparer.path,
        "--raw-root",
        paths["raw_root"],
        "--feature-root",
        paths["feature_root"],
        "--contract-manifest",
        records.contract_manifest.path,
        "--output-root",
        records.phase_output.path,
    ]
    descriptor = {
        "schema": "alpha_max_phase_preparation_invocation.v1",
        "paths": paths,
        "forbidden_roots": list(request.forbidden_roots),
        "frozen_sha256": {
            "acquirer": records.acquirer.sha256,
            "contract_manifest": records.contract_manifest.sha256,
            "availability_evidence": records.availability_evidence.sha256,
            "preparer": records.preparer.sha256,
            "wrapper": records.phase_wrapper.sha256,
        },
        "invocation_lock_identity": [
            lock_info.st_dev,
            lock_info.st_ino,
            stat.S_IFMT(lock_info.st_mode),
            lock_info.st_nlink,
            lock_info.st_size,
            lock_info.st_mtime_ns,
            lock_info.st_ctime_ns,
        ],
        "source_eligibility_snapshot": source_snapshot_data,
        "verifier_argv": verifier_argv,
        "preparer_argv": preparer_argv,
    }
    _sealed_write(descriptor_path, descriptor)
    snapshot_entries: list[dict[str, object]] = []
    snapshot_directories = {source_snapshot}
    for relative in output_inventory:
        source_path = source / relative
        snapshot_path = source_snapshot / relative
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_directory = snapshot_path.parent
        while snapshot_directory != source_snapshot:
            snapshot_directories.add(snapshot_directory)
            snapshot_directory = snapshot_directory.parent
        snapshot_bytes = source_path.read_bytes()
        snapshot_path.write_bytes(snapshot_bytes)
        snapshot_path.chmod(0o444)
        snapshot_entries.append(
            {
                "source_relative_path": relative,
                "sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                "byte_count": len(snapshot_bytes),
            }
        )
    snapshot_entries.sort(key=lambda entry: str(entry["source_relative_path"]))
    snapshot_manifest = {
        "schema": "alpha_max_phase_preparation_source_snapshot.v1",
        "descriptor_sha256": hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": source_snapshot_data["source_manifest_sha256"],
        "entries": snapshot_entries,
    }
    _sealed_write(source_snapshot / "snapshot-manifest.json", snapshot_manifest)
    _sealed_write(
        source_snapshot / ".complete.json",
        {
            "schema": "alpha_max_phase_preparation_source_snapshot.v1",
            "snapshot_manifest_sha256": hashlib.sha256(
                (source_snapshot / "snapshot-manifest.json").read_bytes()
            ).hexdigest(),
        },
    )
    for directory in sorted(snapshot_directories, key=lambda path: len(path.parts), reverse=True):
        directory.chmod(0o555)
    for directory in (
        source,
        report,
        provenance,
        *sorted(output_directories, key=lambda path: len(path.parts), reverse=True),
    ):
        directory.chmod(0o555)
    records = replace(
        records,
        source_root=_directory_identity(source),
        source_report=_directory_identity(report),
    )
    request.records = records
    handoff = {
        "schema": "alpha_max_phase_preparation_eligible_source_receipt.v2",
        "invocation_descriptor_sha256": hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
        "source_eligibility_snapshot": source_snapshot_data,
        "verifier_argv_sha256": hashlib.sha256(canonical_bytes(verifier_argv)).hexdigest(),
        "preparer_argv_sha256": hashlib.sha256(canonical_bytes(preparer_argv)).hexdigest(),
        "preparer_result": {
            "file_count": len(phase_entries),
            "output_root": str(output),
            "preparation_manifest_sha256": hashlib.sha256(
                preparation_path.read_bytes()
            ).hexdigest(),
        },
        "output_root_identity": {"st_dev": output.stat().st_dev, "st_ino": output.stat().st_ino},
        "source_snapshot_manifest_sha256": hashlib.sha256(
            (source_snapshot / "snapshot-manifest.json").read_bytes()
        ).hexdigest(),
        "source_snapshot_identity": {
            "st_dev": source_snapshot.stat().st_dev,
            "st_ino": source_snapshot.stat().st_ino,
        },
        "output_manifest_sha256": hashlib.sha256(preparation_path.read_bytes()).hexdigest(),
    }
    _sealed_write(handoff_path, handoff)
    prerequisites = (
        *request.prerequisites,
        _prerequisite("phase_handoff_receipt", handoff_path),
        _prerequisite("preparation_manifest", preparation_path),
    )
    repository = tmp_path / "repository"
    evidence = tmp_path / "evidence"
    repository.mkdir()
    evidence.mkdir()
    request = terminal_policy.ScopeRequest(
        "alpha_max_terminal_request.phase_preparation.v1",
        _digest("phase-request"),
        "phase_preparation",
        _digest("checkpoint"),
        interpreter,
        _directory_identity(repository),
        _directory_identity(evidence),
        str(evidence / "terminal-authority.sock"),
        terminal_policy.Environment(
            str(evidence),
            "C.UTF-8",
            "C.UTF-8",
            "/usr/bin:/bin",
            "0",
            "1",
            "1",
            "UTC",
        ),
        ("/forbidden-a", "/forbidden-b"),
        terminal_policy.PublicationPaths(
            str(evidence / "claim.json"),
            str(evidence / "journal.jsonl"),
            (str(evidence / "stdout.log"),),
            (str(evidence / "stderr.log"),),
            str(evidence / "receipt.json"),
        ),
        prerequisites,
        records,
    )
    files.update(
        source=source,
        report=report,
        output=output,
        receipt=receipt_path,
        manifest=manifest_path,
        journal=journal,
        plan=report / "plan.json",
        source_owner=source / ".alpha_max_owner.json",
        report_owner=report / ".alpha_max_owner.json",
        handoff=handoff_path,
    )
    return SimpleNamespace(scope_order=("phase_preparation",)), request, files


def _rewrite_sealed(path: Path, value: dict | bytes) -> None:
    path.parent.chmod(0o755)
    path.chmod(0o644)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        write_canonical(path, value)
    path.chmod(0o444)
    path.parent.chmod(0o555)


def _rewrite_leaf(path: Path, value: dict) -> None:
    path.chmod(0o644)
    write_canonical(path, value)
    path.chmod(0o444)


def test_phase_records_real_acquisition_predecessor_rejects_authenticated_mutations(
    tmp_path: Path,
) -> None:
    _envelope, request, _files = _phase_records_fixture(tmp_path / "baseline")
    artifacts, _snapshots = terminal_policy._validate_preparation_manifest(request)
    assert [artifact.kind for artifact in artifacts] == [
        "phase_handoff_receipt",
        "preparation_manifest",
    ]
    assert [artifact.path for artifact in artifacts] == [
        str(_files["handoff"]),
        str(_files["output"] / "preparation_manifest.json"),
    ]
    assert _snapshots == (terminal_policy._snapshot_digest(_files["output"]),)

    cases = (
        (
            "receipt",
            "source acquisition coverage mismatch",
            lambda paths: _rewrite_sealed(
                paths["receipt"],
                {**json.loads(paths["receipt"].read_text()), "source_eligible": False},
            ),
        ),
        (
            "plan",
            "source acquisition plan mismatch",
            lambda paths: _rewrite_sealed(
                paths["plan"], {**json.loads(paths["plan"].read_text()), "months": ["2020-01"]}
            ),
        ),
        (
            "owners",
            "source ownership binding mismatch",
            lambda paths: [
                _rewrite_sealed(
                    path,
                    {
                        **json.loads(path.read_text()),
                        "report_path": str(paths["report"] / "changed"),
                    },
                )
                for path in (paths["source_owner"], paths["report_owner"])
            ],
        ),
        (
            "phase-cross-binding",
            "phase source eligibility snapshot mismatch",
            lambda paths: _rewrite_leaf(
                paths["handoff"],
                {
                    **json.loads(paths["handoff"].read_text()),
                    "source_eligibility_snapshot": {
                        **json.loads(paths["handoff"].read_text())["source_eligibility_snapshot"],
                        "source_owner_sha256": _digest("wrong-phase-owner"),
                    },
                },
            ),
        ),
        (
            "manifest",
            "source official report coverage mismatch",
            lambda paths: _rewrite_sealed(
                paths["manifest"],
                {
                    **json.loads(paths["manifest"].read_text()),
                    "artifacts": [
                        {
                            **json.loads(paths["manifest"].read_text())["artifacts"][0],
                            "sha256": _digest("wrong"),
                        }
                    ],
                },
            ),
        ),
        (
            "journal",
            "source official report coverage mismatch",
            lambda paths: _rewrite_sealed(paths["journal"], b'{"event":"changed"}\n'),
        ),
    )
    mutated_paths = (
        "receipt",
        "plan",
        "source_owner",
        "report_owner",
        "manifest",
        "journal",
        "handoff",
    )
    for name, message, mutate in cases:
        _envelope, candidate, candidate_files = _phase_records_fixture(tmp_path / name)
        before = {key: candidate_files[key].read_bytes() for key in mutated_paths}
        mutate(candidate_files)
        assert any(candidate_files[key].read_bytes() != value for key, value in before.items()), (
            name
        )
        with pytest.raises(TerminalPolicyError, match=message):
            terminal_policy._validate_preparation_manifest(candidate)

    _envelope, candidate, candidate_files = _phase_records_fixture(tmp_path / "inode")
    replacement = candidate_files["receipt"].with_suffix(".replacement")
    candidate_files["report"].chmod(0o755)
    replacement.write_bytes(candidate_files["receipt"].read_bytes())
    replacement.chmod(0o444)
    os.replace(replacement, candidate_files["receipt"])
    candidate_files["report"].chmod(0o555)
    receipt_prerequisite = next(
        item for item in candidate.prerequisites if item.kind == "source_eligible_receipt"
    )
    inode_replaced = candidate_files["receipt"].stat().st_ino != receipt_prerequisite.st_ino
    assert inode_replaced
    with pytest.raises(TerminalPolicyError, match="source_eligible_receipt prerequisite drift"):
        terminal_policy._validate_preparation_manifest(candidate)


def _adversarial_write(path: Path, value: dict | list | bytes) -> None:
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
    path.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        if path.exists():
            path.chmod(0o644)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            write_canonical(path, value)
        path.chmod(0o444)
    finally:
        path.parent.chmod(parent_mode)


def _adversarial_remove(path: Path) -> None:
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
    path.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        path.chmod(0o644)
        path.unlink()
    finally:
        path.parent.chmod(parent_mode)


def _adversarial_add_file(path: Path, payload: bytes = b"extra") -> None:
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
    path.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        path.write_bytes(payload)
        path.chmod(0o444)
    finally:
        path.parent.chmod(parent_mode)


def _adversarial_snapshot(paths: dict[str, Path]) -> Path:
    return paths["handoff"].with_name(".phase.alpha_max_phase_preparation.source-snapshot")


def _adversarial_receipt(path: Path) -> None:
    receipt = json.loads(path.with_name(path.name + ".receipt.json").read_text())
    receipt.update(
        byte_count=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    _adversarial_write(path.with_name(path.name + ".receipt.json"), receipt)


def _adversarial_official_receipt(paths: dict[str, Path], relative: str, **updates: object) -> None:
    receipt_path = paths["report"] / f"{relative}.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(updates)
    _adversarial_write(receipt_path, receipt)


def _adversarial_add_directory(path: Path) -> None:
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
    path.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        path.mkdir()
        path.chmod(0o555)
    finally:
        path.parent.chmod(parent_mode)


def _adversarial_add_symlink(path: Path, target: Path) -> None:
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
    path.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        path.symlink_to(target)
    finally:
        path.parent.chmod(parent_mode)


def _adversarial_add_hard_link(path: Path, target: Path) -> None:
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
    path.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        os.link(target, path)
    finally:
        path.parent.chmod(parent_mode)


def _adversarial_add_official_page(paths: dict[str, Path], relative: str) -> None:
    page = paths["report"] / relative
    parent_mode = stat.S_IMODE(page.parent.stat().st_mode)
    page.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        page.write_bytes(b"[]")
        page.chmod(0o444)
        _fixture_official_receipt(page, "https://fapi.binance.com/fapi/v1/fundingRate")
    finally:
        page.parent.chmod(parent_mode)


def _adversarial_add_funding_page(
    paths: dict[str, Path], number: int, rows: list[dict[str, object]], query: dict[str, str]
) -> None:
    page = paths["report"] / f"provenance/funding_pages/TONUSDT/{number:06d}.json"
    parent_mode = stat.S_IMODE(page.parent.stat().st_mode)
    page.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        _fixture_write_canonical(page, rows)
        page.chmod(0o444)
        _fixture_official_receipt(
            page,
            "https://fapi.binance.com/fapi/v1/fundingRate?"
            + "&".join(f"{key}={value}" for key, value in query.items()),
            query,
        )
        page.with_name(page.name + ".receipt.json").chmod(0o444)
    finally:
        page.parent.chmod(parent_mode)


def _adversarial_rename(path: Path, replacement: Path) -> None:
    parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
    path.parent.chmod(parent_mode | stat.S_IWUSR)
    try:
        os.replace(path, replacement)
    finally:
        path.parent.chmod(parent_mode)


def _adversarial_reseal_manifest_and_receipt(paths: dict[str, Path]) -> None:
    manifest = json.loads(paths["manifest"].read_text())
    for artifact in manifest["artifacts"]:
        root = paths["source"] if artifact["path"].startswith("output/") else paths["report"]
        artifact["sha256"] = hashlib.sha256(
            (root / artifact["path"].split("/", 1)[1]).read_bytes()
        ).hexdigest()
    _adversarial_write(paths["manifest"], manifest)
    receipt = json.loads(paths["receipt"].read_text())
    inventory = sorted(
        artifact["path"].split("/", 1)[1]
        for artifact in manifest["artifacts"]
        if artifact["path"].startswith("output/")
    )
    receipt.update(
        exchange_info_sha256=hashlib.sha256(
            (paths["report"] / "provenance/exchangeInfo.json").read_bytes()
        ).hexdigest(),
        inventory_sha256=hashlib.sha256(canonical_bytes(inventory)).hexdigest(),
        source_manifest_sha256=hashlib.sha256(paths["manifest"].read_bytes()).hexdigest(),
        acquisition_journal_sha256=hashlib.sha256(paths["journal"].read_bytes()).hexdigest(),
    )
    _adversarial_write(paths["receipt"], receipt)


def _adversarial_reseal_handoff(paths: dict[str, Path]) -> None:
    handoff = json.loads(paths["handoff"].read_text())
    output = paths["output"]
    snapshot = paths["handoff"].with_name(".phase.alpha_max_phase_preparation.source-snapshot")
    manifest = output / "preparation_manifest.json"
    snapshot_manifest = snapshot / "snapshot-manifest.json"
    handoff.update(
        output_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_snapshot_manifest_sha256=hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest(),
        source_snapshot_identity={
            "st_dev": snapshot.stat().st_dev,
            "st_ino": snapshot.stat().st_ino,
        },
    )
    handoff["preparer_result"]["preparation_manifest_sha256"] = handoff["output_manifest_sha256"]
    _adversarial_write(paths["handoff"], handoff)


def _adversarial_partition(paths: dict[str, Path], prefix: str) -> Path:
    return next(
        entry
        for entry in (paths["report"] / "partitions").iterdir()
        if json.loads(entry.read_text())["path"].startswith(prefix)
    )


def _adversarial_acquisition_records(
    records: terminal_policy.PhaseRecords,
) -> terminal_policy.AcquisitionRecords:
    parent = _directory_identity(Path(records.source_root.path).parent)
    return terminal_policy.AcquisitionRecords(
        records.acquirer,
        records.contract_manifest,
        records.availability_evidence,
        terminal_policy.AbsentOutput(
            records.source_root.path,
            parent,
            Path(records.source_root.path).name,
            True,
        ),
        terminal_policy.AbsentOutput(
            records.source_report.path,
            parent,
            Path(records.source_report.path).name,
            True,
        ),
    )


@pytest.mark.parametrize(("owned", "terminal_rows"), ((999, 0), (1000, 1)))
def test_phase_records_ton_full_page_pagination_controls(
    tmp_path: Path, owned: int, terminal_rows: int
) -> None:
    _envelope, request, paths = _phase_records_fixture(tmp_path, ton_owned_settlements=owned)
    pages = [
        page
        for page in sorted((paths["report"] / "provenance/funding_pages/TONUSDT").glob("*.json"))
        if not page.name.endswith(".receipt.json")
    ]
    assert [page.name for page in pages] == ["000001.json", "000002.json"]
    assert [
        page.name
        for page in sorted(
            (paths["report"] / "provenance/funding_pages/TONUSDT").glob("*.receipt.json")
        )
    ] == ["000001.json.receipt.json", "000002.json.receipt.json"]
    assert [len(json.loads(page.read_text())) for page in pages] == [1000, terminal_rows]

    expected_end = 1_749_254_400_000 + owned * 14_400_000 - 1
    cursor = 1_749_225_600_000
    page_hashes = []
    for page in pages:
        rows = json.loads(page.read_text())
        receipt = json.loads(page.with_name(page.name + ".receipt.json").read_text())
        assert receipt["query"] == {
            "symbol": "TONUSDT",
            "startTime": str(cursor),
            "endTime": str(expected_end),
            "limit": "1000",
        }
        page_hashes.append(hashlib.sha256(page.read_bytes()).hexdigest())
        if len(rows) == 1000:
            cursor = rows[-1]["fundingTime"] + 1

    ton_partitions = [
        json.loads(path.read_text())
        for path in (paths["report"] / "partitions").iterdir()
        if json.loads(path.read_text())["path"].startswith(
            "feature_points/exchange=binance/symbol=TONUSDT/"
        )
    ]
    assert ton_partitions
    assert all(partition["page_hashes"] == page_hashes for partition in ton_partitions)
    phase_manifest = json.loads((paths["output"] / "preparation_manifest.json").read_text())
    ton_feature_entries = [
        entry
        for entry in phase_manifest["files"]
        if entry["symbol"] == "TONUSDT" and entry["root_kind"] == "feature"
    ]
    assert max(entry["owned_end_utc"] for entry in ton_feature_entries) == (
        "2025-11-20T12:00:00Z" if owned == 999 else "2025-11-20T16:00:00Z"
    )
    assert terminal_policy._validate_acquisition(_adversarial_acquisition_records(request.records))
    assert terminal_policy._validate_preparation_manifest(request)[0]


@pytest.mark.parametrize(
    ("name", "message"),
    (
        ("missing-continuation", "official payload was not safely enumerated"),
        ("renamed-continuation", "official payload was not safely enumerated"),
        ("cursor-drift", "source official report coverage mismatch"),
        ("early-termination", "source official report coverage mismatch"),
        ("missing-matching-receipt", "official request receipt was not safely enumerated"),
        ("post-terminal-page", "source official report coverage mismatch"),
    ),
)
def test_phase_records_ton_pagination_control_negatives(
    tmp_path: Path, name: str, message: str
) -> None:
    _envelope, request, paths = _phase_records_fixture(tmp_path, ton_owned_settlements=999)
    pages = paths["report"] / "provenance/funding_pages/TONUSDT"
    continuation = pages / "000002.json"
    continuation_receipt = continuation.with_name(continuation.name + ".receipt.json")

    if name == "missing-continuation":
        _adversarial_remove(continuation)
        _adversarial_remove(continuation_receipt)
    elif name == "renamed-continuation":
        _adversarial_rename(continuation, pages / "000003.json")
        _adversarial_rename(continuation_receipt, pages / "000003.json.receipt.json")
    elif name == "cursor-drift":
        receipt = json.loads(continuation_receipt.read_text())
        _adversarial_official_receipt(
            paths,
            "provenance/funding_pages/TONUSDT/000002.json",
            query={**receipt["query"], "startTime": str(int(receipt["query"]["startTime"]) + 1)},
        )
        _adversarial_reseal_manifest_and_receipt(paths)
    elif name == "early-termination":
        first = pages / "000001.json"
        _adversarial_write(first, json.loads(first.read_text())[:-1])
        _adversarial_receipt(first)
        _adversarial_reseal_manifest_and_receipt(paths)
    elif name == "missing-matching-receipt":
        _adversarial_remove(continuation_receipt)
    else:
        terminal_receipt = json.loads(continuation_receipt.read_text())
        _adversarial_add_funding_page(paths, 3, [], terminal_receipt["query"])
        _adversarial_reseal_manifest_and_receipt(paths)

    with pytest.raises(TerminalPolicyError, match=message):
        terminal_policy._validate_acquisition(_adversarial_acquisition_records(request.records))


@pytest.mark.parametrize(
    ("name", "message", "mutate"),
    (
        (
            "owner-only-fake-totals",
            "raw output was not safely enumerated",
            lambda paths: _adversarial_write(
                paths["receipt"],
                {
                    **json.loads(paths["receipt"].read_text()),
                    "raw_rows": 1,
                    "funding_rows": 1,
                    "inventory_sha256": _digest("fake-inventory"),
                },
            ),
        ),
        (
            "missing-source-output",
            "raw output was not safely enumerated",
            lambda paths: _adversarial_remove(
                paths["source"] / "market_ohlcv_1s/binance/ADAUSDT/2025-06.parquet"
            ),
        ),
        (
            "extra-source-output",
            "source acquisition coverage mismatch",
            lambda paths: (
                paths["source"].chmod(0o755),
                (paths["source"] / "unexpected.parquet").write_bytes(b"extra"),
                (paths["source"] / "unexpected.parquet").chmod(0o444),
                paths["source"].chmod(0o555),
                _adversarial_reseal_manifest_and_receipt(paths),
            ),
        ),
        (
            "missing-report-leaf",
            "official payload was not safely enumerated",
            lambda paths: _adversarial_remove(paths["report"] / "provenance/exchangeInfo.json"),
        ),
        (
            "extra-report-leaf",
            "source official report coverage mismatch",
            lambda paths: (
                paths["report"].chmod(0o755),
                (paths["report"] / "extra.json").write_bytes(b"{}"),
                (paths["report"] / "extra.json").chmod(0o444),
                paths["report"].chmod(0o555),
                _adversarial_reseal_manifest_and_receipt(paths),
            ),
        ),
        (
            "extra-empty-report-directory",
            "source official report coverage mismatch",
            lambda paths: (
                paths["report"].chmod(0o755),
                (paths["report"] / "empty").mkdir(),
                (paths["report"] / "empty").chmod(0o555),
                paths["report"].chmod(0o555),
            ),
        ),
        (
            "official-redirect",
            "source official report coverage mismatch",
            lambda paths: _adversarial_write(
                paths["report"] / "provenance/exchangeInfo.json.receipt.json",
                {
                    **json.loads(
                        (paths["report"] / "provenance/exchangeInfo.json.receipt.json").read_text()
                    ),
                    "final_url": "https://redirect.invalid/exchangeInfo",
                },
            ),
        ),
        (
            "official-non-string-query",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json.receipt.json",
                    {
                        **json.loads(
                            (
                                paths["report"]
                                / "provenance/funding_pages/ADAUSDT/000001.json.receipt.json"
                            ).read_text()
                        ),
                        "query": {
                            **json.loads(
                                (
                                    paths["report"]
                                    / "provenance/funding_pages/ADAUSDT/000001.json.receipt.json"
                                ).read_text()
                            )["query"],
                            "limit": 1000,
                        },
                    },
                ),
            ),
        ),
        (
            "official-duplicate-key",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/exchangeInfo.json",
                    b'{"symbols":[],"symbols":[]}',
                ),
                _adversarial_receipt(paths["report"] / "provenance/exchangeInfo.json"),
            ),
        ),
        (
            "official-nan",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/exchangeInfo.json",
                    b'{"symbols":NaN}',
                ),
                _adversarial_receipt(paths["report"] / "provenance/exchangeInfo.json"),
            ),
        ),
        (
            "funding-first-gap",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json",
                    [
                        {
                            **json.loads(
                                (
                                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                                ).read_text()
                            )[0],
                            "fundingTime": 1_749_283_200_000,
                        }
                    ],
                ),
                _adversarial_receipt(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                ),
            ),
        ),
        (
            "funding-before-current-cursor",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json",
                    [
                        {
                            **json.loads(
                                (
                                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                                ).read_text()
                            )[0],
                            "fundingTime": 1_749_254_399_999,
                        }
                    ],
                ),
                _adversarial_receipt(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                ),
            ),
        ),
        (
            "funding-at-feature-end",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json",
                    [
                        {
                            **json.loads(
                                (
                                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                                ).read_text()
                            )[0],
                            "fundingTime": 1_749_427_200_000,
                        }
                    ],
                ),
                _adversarial_receipt(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                ),
            ),
        ),
        (
            "funding-after-feature-end",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json",
                    [
                        {
                            **json.loads(
                                (
                                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                                ).read_text()
                            )[0],
                            "fundingTime": 1_749_427_200_001,
                        }
                    ],
                ),
                _adversarial_receipt(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                ),
            ),
        ),
        (
            "funding-post-terminal-page",
            "source official report coverage mismatch",
            lambda paths: (
                (paths["report"] / "provenance/funding_pages/ADAUSDT").chmod(0o755),
                (paths["report"] / "provenance/funding_pages/ADAUSDT/000002.json").write_text("[]"),
                (paths["report"] / "provenance/funding_pages/ADAUSDT/000002.json").chmod(0o444),
                _fixture_official_receipt(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000002.json",
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                ),
                (paths["report"] / "provenance/funding_pages/ADAUSDT").chmod(0o555),
            ),
        ),
        (
            "funding-1001-rows",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json",
                    [
                        {
                            "symbol": "ADAUSDT",
                            "fundingTime": 1_749_254_400_000 + index,
                            "fundingRate": "0.0001",
                        }
                        for index in range(1001)
                    ],
                ),
                _adversarial_receipt(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json"
                ),
            ),
        ),
        (
            "ton-proof-missing",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/funding_pages/TONUSDT/000001.json",
                    json.loads(
                        (
                            paths["report"] / "provenance/funding_pages/TONUSDT/000001.json"
                        ).read_text()
                    )[1:],
                ),
                _adversarial_receipt(
                    paths["report"] / "provenance/funding_pages/TONUSDT/000001.json"
                ),
            ),
        ),
        (
            "funding-partition-source-sha",
            "source partition receipt mismatch",
            lambda paths: _adversarial_write(
                _adversarial_partition(paths, "feature_points/"),
                {
                    **json.loads(_adversarial_partition(paths, "feature_points/").read_text()),
                    "source_sha256": _digest("wrong-funding-source"),
                },
            ),
        ),
        (
            "raw-carry-chain",
            "source partition receipt mismatch",
            lambda paths: _adversarial_write(
                _adversarial_partition(paths, "market_ohlcv_1s/"),
                {
                    **json.loads(_adversarial_partition(paths, "market_ohlcv_1s/").read_text()),
                    "input_carry_close": 7.0,
                },
            ),
        ),
        (
            "raw-output-carry-type",
            "source partition receipt mismatch",
            lambda paths: _adversarial_write(
                _adversarial_partition(paths, "market_ohlcv_1s/"),
                {
                    **json.loads(_adversarial_partition(paths, "market_ohlcv_1s/").read_text()),
                    "output_carry_close": "9.0",
                },
            ),
        ),
        (
            "partition-path-and-code",
            "source partition receipt mismatch",
            lambda paths: _adversarial_write(
                _adversarial_partition(paths, "market_ohlcv_1s/"),
                {
                    **json.loads(_adversarial_partition(paths, "market_ohlcv_1s/").read_text()),
                    "path": "market_ohlcv_1s/binance/ADAUSDT/wrong.parquet",
                    "code_sha256": _digest("wrong-code"),
                },
            ),
        ),
        (
            "ton-forbidden-floor-one",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["report"] / "provenance/funding_pages/TONUSDT/000001.json",
                    [
                        *json.loads(
                            (
                                paths["report"] / "provenance/funding_pages/TONUSDT/000001.json"
                            ).read_text()
                        ),
                        {
                            "symbol": "TONUSDT",
                            "fundingTime": 1_749_240_000_000,
                            "fundingRate": "0.0001",
                        },
                    ],
                ),
                _adversarial_receipt(
                    paths["report"] / "provenance/funding_pages/TONUSDT/000001.json"
                ),
            ),
        ),
        (
            "partition-derivation-v4",
            "source derivation version mismatch",
            lambda paths: _adversarial_write(
                _adversarial_partition(paths, "market_ohlcv_1s/"),
                {
                    **json.loads(_adversarial_partition(paths, "market_ohlcv_1s/").read_text()),
                    "derivation_version": "alpha-max-binance-ohlcv-v3",
                },
            ),
        ),
        (
            "funding-page-gap",
            "source official report coverage mismatch",
            lambda paths: _adversarial_add_official_page(
                paths, "provenance/funding_pages/ADAUSDT/000003.json"
            ),
        ),
        (
            "funding-wrong-first-page-name",
            "official payload was not safely enumerated",
            lambda paths: (
                _adversarial_rename(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json",
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000002.json",
                ),
                _adversarial_rename(
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000001.json.receipt.json",
                    paths["report"] / "provenance/funding_pages/ADAUSDT/000002.json.receipt.json",
                ),
            ),
        ),
        (
            "funding-next-page-cursor",
            "source official report coverage mismatch",
            lambda paths: _adversarial_official_receipt(
                paths,
                "provenance/funding_pages/ADAUSDT/000001.json",
                query={
                    "symbol": "ADAUSDT",
                    "startTime": "1749254400001",
                    "endTime": "1749427199999",
                    "limit": "1000",
                },
            ),
        ),
        (
            "official-wrong-host",
            "source official report coverage mismatch",
            lambda paths: _adversarial_official_receipt(
                paths,
                "provenance/exchangeInfo.json",
                final_host="example.invalid",
            ),
        ),
        (
            "official-wrong-path",
            "source official report coverage mismatch",
            lambda paths: _adversarial_official_receipt(
                paths,
                "provenance/exchangeInfo.json",
                requested_url="https://fapi.binance.com/fapi/v1/wrong",
                final_url="https://fapi.binance.com/fapi/v1/wrong",
            ),
        ),
        (
            "official-wrong-byte-count",
            "source official report coverage mismatch",
            lambda paths: _adversarial_official_receipt(
                paths, "provenance/exchangeInfo.json", byte_count=0
            ),
        ),
        (
            "official-wrong-payload-digest",
            "source official report coverage mismatch",
            lambda paths: _adversarial_official_receipt(
                paths, "provenance/exchangeInfo.json", sha256=_digest("wrong-payload")
            ),
        ),
        (
            "source-owner-derivation-v4",
            "source derivation version mismatch",
            lambda paths: _adversarial_write(
                paths["source_owner"],
                {
                    **json.loads(paths["source_owner"].read_text()),
                    "derivation_version": "alpha-max-binance-ohlcv-v3",
                },
            ),
        ),
        (
            "report-owner-derivation-v4",
            "source ownership binding mismatch",
            lambda paths: _adversarial_write(
                paths["report_owner"],
                {
                    **json.loads(paths["report_owner"].read_text()),
                    "derivation_version": "alpha-max-binance-ohlcv-v3",
                },
            ),
        ),
        (
            "manifest-derivation-v4",
            "source derivation version mismatch",
            lambda paths: _adversarial_write(
                paths["manifest"],
                {
                    **json.loads(paths["manifest"].read_text()),
                    "derivation_version": "alpha-max-binance-ohlcv-v3",
                },
            ),
        ),
        (
            "eligible-receipt-derivation-v4",
            "source derivation version mismatch",
            lambda paths: _adversarial_write(
                paths["receipt"],
                {
                    **json.loads(paths["receipt"].read_text()),
                    "derivation_version": "alpha-max-binance-ohlcv-v3",
                },
            ),
        ),
        (
            "report-basename-prefix-confusion",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_add_directory(paths["report"] / "provenance/plan.json.shadow"),
                _adversarial_add_file(
                    paths["report"] / "provenance/plan.json.shadow/unexpected.json"
                ),
            ),
        ),
        (
            "manifest-class-reorder",
            "source official report coverage mismatch",
            lambda paths: (
                _adversarial_write(
                    paths["manifest"],
                    {
                        **json.loads(paths["manifest"].read_text()),
                        "artifacts": list(
                            reversed(json.loads(paths["manifest"].read_text())["artifacts"])
                        ),
                    },
                ),
                _adversarial_reseal_manifest_and_receipt(paths),
            ),
        ),
    ),
)
def test_phase_records_acquisition_adversarial_matrix(
    tmp_path: Path, name: str, message: str, mutate: object
) -> None:
    _envelope, request, paths = _phase_records_fixture(tmp_path / name)
    if name == "owner-only-fake-totals":
        for root, directories, filenames in os.walk(paths["source"], topdown=False):
            directory = Path(root)
            directory.chmod(0o755)
            for filename in filenames:
                leaf = directory / filename
                if leaf.name != ".alpha_max_owner.json":
                    leaf.chmod(0o644)
                    leaf.unlink()
            for child in directories:
                child_path = directory / child
                child_path.chmod(0o755)
                child_path.rmdir()
            directory.chmod(0o555)
    mutate(paths)
    with pytest.raises(TerminalPolicyError, match=message):
        terminal_policy._validate_acquisition(_adversarial_acquisition_records(request.records))


@pytest.mark.parametrize(
    "case",
    (
        "old-plan",
        "old-manifest",
        "old-receipt",
        "intent-tamper",
        "deletion-tamper",
        "retained-zip",
        "retained-zip-symlink",
        "retained-zip-hardlink",
        "missing-evidence",
        "archive-receipt-tamper",
        "derivation-chain-tamper",
        "storage-contract-tamper",
        "evidence-digest-tamper",
    ),
)
def test_compact_archive_proof_adversarial_envelope(tmp_path: Path, case: str) -> None:
    _envelope, request, paths = _phase_records_fixture(tmp_path / case)
    report = paths["report"]
    symbol = _FIXTURE_SYMBOLS[0]
    archive = report / f"provenance/archives/{symbol}/{symbol}-aggTrades-2025-06.zip"
    evidence = report / f"provenance/archive-evidence/{symbol}/2025-06"
    if case == "old-plan":
        _adversarial_write(
            paths["plan"],
            {
                **json.loads(paths["plan"].read_text()),
                "schema": "alpha_max_official_acquisition_plan.v3",
            },
        )
    elif case == "old-manifest":
        _adversarial_write(
            paths["manifest"],
            {
                **json.loads(paths["manifest"].read_text()),
                "schema": "alpha_max_official_source_manifest.v4",
            },
        )
    elif case == "old-receipt":
        _adversarial_write(
            paths["receipt"],
            {
                **json.loads(paths["receipt"].read_text()),
                "schema": "alpha_max_official_source_receipt.v3",
            },
        )
    elif case == "retained-zip":
        _adversarial_add_file(archive)
    elif case == "retained-zip-symlink":
        _adversarial_add_symlink(archive, paths["receipt"])
    elif case == "retained-zip-hardlink":
        _adversarial_add_hard_link(archive, paths["receipt"])
    elif case == "missing-evidence":
        _adversarial_remove(evidence.with_suffix(".deletion.json"))
    elif case == "archive-receipt-tamper":
        receipt = archive.with_name(archive.name + ".receipt.json")
        _adversarial_write(receipt, {**json.loads(receipt.read_text()), "byte_count": 0})
    elif case == "derivation-chain-tamper":
        derivation = evidence.with_suffix(".derivation.json")
        _adversarial_write(
            derivation,
            {
                **json.loads(derivation.read_text()),
                "prior_derivation_receipt_sha256": _digest("wrong-prior"),
            },
        )
    elif case == "intent-tamper":
        intent = evidence.with_suffix(".retirement-intent.json")
        _adversarial_write(intent, {**json.loads(intent.read_text()), "archive_absent": True})
    elif case == "deletion-tamper":
        deletion = evidence.with_suffix(".deletion.json")
        _adversarial_write(deletion, {**json.loads(deletion.read_text()), "archive_absent": False})
    elif case == "storage-contract-tamper":
        _adversarial_write(
            paths["receipt"],
            {
                **json.loads(paths["receipt"].read_text()),
                "storage_contract": {
                    **json.loads(paths["receipt"].read_text())["storage_contract"],
                    "max_live_archives": 2,
                },
            },
        )
    else:
        _adversarial_write(
            paths["manifest"],
            {
                **json.loads(paths["manifest"].read_text()),
                "archive_evidence_sha256": _digest("wrong-evidence"),
            },
        )
    with pytest.raises(TerminalPolicyError):
        terminal_policy._validate_acquisition(_adversarial_acquisition_records(request.records))


def test_phase_records_noncanonical_official_json_is_accepted(
    tmp_path: Path,
) -> None:
    _envelope, request, paths = _phase_records_fixture(tmp_path)
    payload = paths["report"] / "provenance/exchangeInfo.json"
    _adversarial_write(
        payload,
        json.dumps(json.loads(payload.read_text()), indent=1).encode(),
    )
    _adversarial_receipt(payload)
    _adversarial_reseal_manifest_and_receipt(paths)
    assert terminal_policy._validate_acquisition(_adversarial_acquisition_records(request.records))


@pytest.mark.parametrize(
    ("name", "message", "mutate"),
    (
        (
            "six-empty-roots",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "file_count": 0,
                    "files": [],
                },
            ),
        ),
        (
            "missing-entry",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "file_count": len(
                        json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                            "files"
                        ]
                    )
                    - 1,
                    "files": json.loads(
                        (paths["output"] / "preparation_manifest.json").read_text()
                    )["files"][1:],
                },
            ),
        ),
        (
            "duplicate-entry",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "file_count": len(
                        json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                            "files"
                        ]
                    )
                    + 1,
                    "files": [
                        *json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                            "files"
                        ],
                        json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                            "files"
                        ][0],
                    ],
                },
            ),
        ),
        (
            "wrong-clipped-bound",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "files": [
                        {
                            **entry,
                            "owned_start_utc": "2025-06-06T00:00:00Z",
                        }
                        if index == 0
                        else entry
                        for index, entry in enumerate(
                            json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                                "files"
                            ]
                        )
                    ],
                },
            ),
        ),
        (
            "extra-empty-output-directory",
            "preparation manifest entry set mismatch",
            lambda paths: (
                paths["output"].chmod(0o755),
                (paths["output"] / "residual").mkdir(),
                (paths["output"] / "residual").chmod(0o555),
                paths["output"].chmod(0o555),
            ),
        ),
        (
            "missing-snapshot-clone",
            "was not safely enumerated",
            lambda paths: _adversarial_remove(
                _adversarial_snapshot(paths)
                / json.loads((_adversarial_snapshot(paths) / "snapshot-manifest.json").read_text())[
                    "entries"
                ][0]["source_relative_path"]
            ),
        ),
        (
            "extra-snapshot-clone",
            "phase snapshot inventory mismatch",
            lambda paths: _adversarial_add_file(_adversarial_snapshot(paths) / "extra.parquet"),
        ),
        (
            "handoff-snapshot-list",
            "phase handoff source snapshot identity",
            lambda paths: _adversarial_write(
                paths["handoff"],
                {
                    **json.loads(paths["handoff"].read_text()),
                    "source_snapshot_identity": [1, 2],
                },
            ),
        ),
        (
            "handoff-snapshot-extra-key",
            "phase handoff source snapshot identity",
            lambda paths: _adversarial_write(
                paths["handoff"],
                {
                    **json.loads(paths["handoff"].read_text()),
                    "source_snapshot_identity": {
                        **json.loads(paths["handoff"].read_text())["source_snapshot_identity"],
                        "extra": 1,
                    },
                },
            ),
        ),
        (
            "handoff-snapshot-missing-key",
            "phase handoff source snapshot identity",
            lambda paths: _adversarial_write(
                paths["handoff"],
                {
                    **json.loads(paths["handoff"].read_text()),
                    "source_snapshot_identity": {
                        "st_dev": json.loads(paths["handoff"].read_text())[
                            "source_snapshot_identity"
                        ]["st_dev"],
                    },
                },
            ),
        ),
        (
            "handoff-snapshot-bool",
            "phase handoff source snapshot identity",
            lambda paths: _adversarial_write(
                paths["handoff"],
                {
                    **json.loads(paths["handoff"].read_text()),
                    "source_snapshot_identity": {"st_dev": True, "st_ino": 1},
                },
            ),
        ),
        (
            "handoff-snapshot-negative",
            "phase handoff source snapshot identity",
            lambda paths: _adversarial_write(
                paths["handoff"],
                {
                    **json.loads(paths["handoff"].read_text()),
                    "source_snapshot_identity": {"st_dev": -1, "st_ino": -1},
                },
            ),
        ),
        (
            "handoff-snapshot-wrong-inode",
            "phase handoff source snapshot identity",
            lambda paths: _adversarial_write(
                paths["handoff"],
                {
                    **json.loads(paths["handoff"].read_text()),
                    "source_snapshot_identity": {
                        **json.loads(paths["handoff"].read_text())["source_snapshot_identity"],
                        "st_ino": 0,
                    },
                },
            ),
        ),
        (
            "out-of-order-manifest-entries",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "files": list(
                        reversed(
                            json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                                "files"
                            ]
                        )
                    ),
                },
            ),
        ),
        (
            "extra-manifest-entry",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "file_count": len(
                        json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                            "files"
                        ]
                    )
                    + 1,
                    "files": [
                        *json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                            "files"
                        ],
                        {
                            **json.loads(
                                (paths["output"] / "preparation_manifest.json").read_text()
                            )["files"][0],
                            "output_relative_path": "purge/raw/unmatched.parquet",
                        },
                    ],
                },
            ),
        ),
        (
            "source-output-mapping-collision",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "files": [
                        {
                            **entry,
                            "output_relative_path": json.loads(
                                (paths["output"] / "preparation_manifest.json").read_text()
                            )["files"][0]["output_relative_path"],
                        }
                        if index == 1
                        else entry
                        for index, entry in enumerate(
                            json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                                "files"
                            ]
                        )
                    ],
                },
            ),
        ),
        (
            "unmatched-source-mapping",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "files": [
                        {
                            **entry,
                            "source_relative_path": "market_ohlcv_1s/binance/ADAUSDT/2099-01.parquet",
                        }
                        if index == 0
                        else entry
                        for index, entry in enumerate(
                            json.loads((paths["output"] / "preparation_manifest.json").read_text())[
                                "files"
                            ]
                        )
                    ],
                },
            ),
        ),
        (
            "unused-selected-source",
            "preparation manifest entry set mismatch",
            lambda paths: _adversarial_write(
                paths["output"] / "preparation_manifest.json",
                {
                    **json.loads((paths["output"] / "preparation_manifest.json").read_text()),
                    "files": [
                        entry
                        for entry in json.loads(
                            (paths["output"] / "preparation_manifest.json").read_text()
                        )["files"]
                        if entry["source_relative_path"]
                        != "market_ohlcv_1s/binance/ADAUSDT/2025-06.parquet"
                    ],
                    "file_count": len(
                        [
                            entry
                            for entry in json.loads(
                                (paths["output"] / "preparation_manifest.json").read_text()
                            )["files"]
                            if entry["source_relative_path"]
                            != "market_ohlcv_1s/binance/ADAUSDT/2025-06.parquet"
                        ]
                    ),
                },
            ),
        ),
        (
            "prepared-output-symlink",
            "phase output tree contains a symlink",
            lambda paths: _adversarial_add_symlink(
                paths["output"] / "linked.parquet",
                paths["output"] / "preparation_manifest.json",
            ),
        ),
        (
            "prepared-output-hard-link",
            "preparation manifest was not safely enumerated",
            lambda paths: _adversarial_add_hard_link(
                paths["output"] / "linked.parquet",
                paths["output"] / "preparation_manifest.json",
            ),
        ),
        (
            "prepared-required-directory-mode",
            "phase output directory is not immutable",
            lambda paths: (paths["output"] / "warmup").chmod(0o755),
        ),
        (
            "snapshot-report-residual",
            "phase snapshot inventory mismatch",
            lambda paths: (
                _adversarial_add_directory(_adversarial_snapshot(paths) / "report"),
                _adversarial_add_file(_adversarial_snapshot(paths) / "report/residual.json"),
            ),
        ),
        (
            "snapshot-stage-residual",
            "phase snapshot inventory mismatch",
            lambda paths: _adversarial_add_file(
                _adversarial_snapshot(paths) / "snapshot-manifest.stage.json"
            ),
        ),
        (
            "snapshot-extra-directory",
            "phase snapshot inventory mismatch",
            lambda paths: _adversarial_add_directory(_adversarial_snapshot(paths) / "residual"),
        ),
        (
            "snapshot-symlink",
            "phase source snapshot tree contains a symlink",
            lambda paths: _adversarial_add_symlink(
                _adversarial_snapshot(paths) / "linked.parquet",
                _adversarial_snapshot(paths) / "snapshot-manifest.json",
            ),
        ),
        (
            "snapshot-hard-link",
            "was not safely enumerated",
            lambda paths: _adversarial_add_hard_link(
                _adversarial_snapshot(paths) / "linked.parquet",
                _adversarial_snapshot(paths) / "snapshot-manifest.json",
            ),
        ),
    ),
)
def test_phase_records_a02_snapshot_adversarial_matrix(
    tmp_path: Path, name: str, message: str, mutate: object
) -> None:
    _envelope, request, paths = _phase_records_fixture(tmp_path / name)
    mutate(paths)
    if "manifest" in name or "output" in name:
        _adversarial_reseal_handoff(paths)
    try:
        with pytest.raises(TerminalPolicyError, match=message):
            terminal_policy._validate_preparation_manifest(request)
    finally:
        if name in {"snapshot-symlink", "snapshot-hard-link"}:
            added = _adversarial_snapshot(paths) / "linked.parquet"
            added.parent.chmod(0o755)
            try:
                added.unlink(missing_ok=True)
            finally:
                added.parent.chmod(0o555)
            for directory, _children, _files in os.walk(tmp_path / name):
                Path(directory).chmod(0o755)


def test_phase_records_rejects_self_consistently_resealed_divergent_snapshot_clone(
    tmp_path: Path,
) -> None:
    _envelope, request, paths = _phase_records_fixture(tmp_path)
    snapshot = _adversarial_snapshot(paths)
    manifest_path = snapshot / "snapshot-manifest.json"
    snapshot_manifest = json.loads(manifest_path.read_text())
    entry = snapshot_manifest["entries"][0]
    clone = snapshot / entry["source_relative_path"]
    divergent = clone.read_bytes()[::-1]
    assert len(divergent) == clone.stat().st_size
    _adversarial_write(clone, divergent)
    entry["sha256"] = hashlib.sha256(divergent).hexdigest()
    _adversarial_write(manifest_path, snapshot_manifest)
    _adversarial_write(
        snapshot / ".complete.json",
        {
            "schema": "alpha_max_phase_preparation_source_snapshot.v1",
            "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    )
    _adversarial_reseal_handoff(paths)

    with pytest.raises(TerminalPolicyError, match="phase snapshot inventory mismatch"):
        terminal_policy._validate_preparation_manifest(request)
