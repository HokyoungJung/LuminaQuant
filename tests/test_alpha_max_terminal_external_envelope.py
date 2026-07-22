from __future__ import annotations

import base64
import copy
import hashlib
import json
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


def write_canonical(path: Path, value: dict) -> None:
    path.write_bytes(canonical_bytes(value))


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
        "observer_keys": [
            {
                "scope": scope,
                "key_id": key_id,
                "public_key_b64": encoded,
                "public_key_sha256": key_id,
            }
            for scope in ("acquisition", "phase_preparation", "one_touch")
        ],
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
    for scope in terminal_policy._SCOPES:
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
        terminal_policy._SCOPES,
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
    root = tmp_path / kind
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


def test_semantic_bundle_readback_accepts_and_rejects_cross_bindings(tmp_path: Path) -> None:
    for kind in ("prelock_bundle", "historical_bundle"):
        root, seal = _semantic_bundle(tmp_path, kind)
        assert terminal_policy._semantic_bundle_readback(root, kind, seal)
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
            terminal_policy._semantic_bundle_readback(root, kind, seal)
    root, seal = _semantic_bundle(tmp_path, "prelock_bundle")
    matrix = root / "status/matrix.json"
    value = json.loads(matrix.read_text())
    matrix.chmod(0o644)
    value["statuses"] = value["statuses"][:-1]
    write_canonical(matrix, value)
    matrix.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="matrix mismatch"):
        terminal_policy._semantic_bundle_readback(root, "prelock_bundle", seal)
    root, seal = _semantic_bundle(tmp_path, "historical_bundle")
    diagnostic = root / "diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json"
    value = json.loads(diagnostic.read_text())
    diagnostic.chmod(0o644)
    value["train_liquidity_buckets_sha256"] = _digest("wrong")
    write_canonical(diagnostic, value)
    diagnostic.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="observability mismatch"):
        terminal_policy._semantic_bundle_readback(root, "historical_bundle", seal)
    root, seal = _semantic_bundle(tmp_path, "prelock_bundle")
    terminal = root / "terminal/prelock.json"
    value = json.loads(terminal.read_text())
    terminal.chmod(0o644)
    value["terminal_outcome"] = "other"
    write_canonical(terminal, value)
    terminal.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="outcome/readback mismatch"):
        terminal_policy._semantic_bundle_readback(root, "prelock_bundle", seal)
    root, seal = _semantic_bundle(tmp_path, "historical_bundle")
    matrix = root / "status/matrix.json"
    value = json.loads(matrix.read_text())
    matrix.chmod(0o644)
    value["physical_fold_run_count"] = 1
    write_canonical(matrix, value)
    matrix.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="matrix mismatch"):
        terminal_policy._semantic_bundle_readback(root, "historical_bundle", seal)


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


def test_sealed_tree_accepts_and_rejects_inventory_cardinality_and_seal_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for kind, key in (
        ("prelock_bundle", "artifacts"),
        ("historical_bundle", "historical_artifacts"),
    ):
        root = _sealed_bundle(tmp_path, kind)
        assert terminal_policy._sealed_tree(str(root), kind, key).kind == kind
    root = _sealed_bundle(tmp_path, "prelock_bundle")
    monkeypatch.setattr(terminal_policy, "_semantic_bundle_readback", lambda *_: "readback")
    extra = root / "extra.json"
    root.chmod(0o755)
    _sealed_write(extra, {})
    with pytest.raises(TerminalPolicyError, match="inventory does not cover tree"):
        terminal_policy._sealed_tree(str(root), "prelock_bundle", "artifacts")
    extra.unlink()
    sealed = root / "SEALED.json"
    sealed.chmod(0o644)
    value = json.loads(sealed.read_text())
    value["artifact_count"] -= 1
    write_canonical(sealed, value)
    sealed.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="seal binding mismatch"):
        terminal_policy._sealed_tree(str(root), "prelock_bundle", "artifacts")
    root = _sealed_bundle(tmp_path, "prelock_bundle")
    sealed = root / "SEALED.json"
    root.chmod(0o755)
    sealed.chmod(0o644)
    value = json.loads(sealed.read_text())
    value["artifacts"].pop()
    write_canonical(sealed, value)
    sealed.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="inventory does not cover tree"):
        terminal_policy._sealed_tree(str(root), "prelock_bundle", "artifacts")
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
        terminal_policy._sealed_tree(str(root), "prelock_bundle", "artifacts")
    root = _sealed_bundle(tmp_path, "prelock_bundle")
    row = root / "evidence/validation/rows/0.json"
    row.parent.chmod(0o755)
    row.unlink()
    _refresh_sealed_inventory(root, "prelock_bundle")
    with pytest.raises(TerminalPolicyError, match="cardinality mismatch"):
        terminal_policy._sealed_tree(str(root), "prelock_bundle", "artifacts")
    root = _sealed_bundle(tmp_path, "historical_bundle")
    root.chmod(0o755)
    (root / "admission/train_liquidity_buckets.json").chmod(0o644)
    with pytest.raises(TerminalPolicyError, match="inventory identity drift"):
        terminal_policy._sealed_tree(str(root), "historical_bundle", "historical_artifacts")
    (root / "admission/train_liquidity_buckets.json").chmod(0o444)
    sealed = root / "SEALED.json"
    sealed.chmod(0o644)
    value = json.loads(sealed.read_text())
    value["completion_id"] = "wrong"
    write_canonical(sealed, value)
    sealed.chmod(0o444)
    with pytest.raises(TerminalPolicyError, match="historical seal binding mismatch"):
        terminal_policy._sealed_tree(str(root), "historical_bundle", "historical_artifacts")


def test_one_touch_second_command_rejects_prelock_and_input_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prelock, historical = tmp_path / "prelock", tmp_path / "historical"
    for root in (prelock, historical):
        root.mkdir()
    _sealed_write(prelock / "SEALED.json", {"seal": "before"})
    _sealed_write(
        historical / "SEALED.json",
        {
            "prelock_seal_sha256": _digest("wrong-seal"),
            "prelock_snapshot_sha256": _digest("wrong-snapshot"),
        },
    )
    _sealed_write(historical / "binding/prelock_seal.json", {"seal": "after"})
    _sealed_write(prelock / "admission/train_liquidity_buckets.json", {"input": "before"})
    _sealed_write(historical / "admission/train_liquidity_buckets.json", {"input": "after"})
    records = terminal_policy.OneTouchRecords(
        None,
        None,
        None,
        None,
        SimpleNamespace(path=str(tmp_path / "phase")),
        SimpleNamespace(path=str(prelock)),
        SimpleNamespace(path=str(historical)),
    )
    request = SimpleNamespace(records=records)
    sealed = SimpleNamespace(sha256=_digest("seal"))
    monkeypatch.setattr(terminal_policy, "_sealed_tree", lambda root, *_: sealed)
    with pytest.raises(TerminalPolicyError, match="prelock binding mismatch"):
        terminal_policy._one_touch_second_command_artifacts(request, 1)
    monkeypatch.setattr(
        terminal_policy,
        "_canonical_object",
        lambda path: (
            {
                "prelock_seal_sha256": sealed.sha256,
                "prelock_snapshot_sha256": terminal_policy._snapshot_digest(prelock),
            }
            if path.name == "SEALED.json" and path.parent == historical
            else terminal_policy.json.loads(path.read_text())
        ),
    )
    with pytest.raises(TerminalPolicyError, match="immutable input comparison failed"):
        terminal_policy._one_touch_second_command_artifacts(request, 1)


def test_completed_command_rejects_one_touch_chain_and_input_snapshot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase, prelock, historical = tmp_path / "phase", tmp_path / "prelock", tmp_path / "historical"
    for root in (phase, prelock, historical):
        root.mkdir()
    records = terminal_policy.OneTouchRecords(
        None,
        None,
        None,
        None,
        SimpleNamespace(path=str(phase)),
        SimpleNamespace(path=str(prelock)),
        SimpleNamespace(path=str(historical)),
    )
    request = SimpleNamespace(scope="one_touch", records=records)
    monkeypatch.setattr(terminal_policy, "derive_scope_commands", lambda *_: ("first", "second"))
    with pytest.raises(TerminalPolicyError, match="command evidence chain is invalid"):
        terminal_policy.validate_completed_command(SimpleNamespace(), request, 1)
    monkeypatch.setattr(terminal_policy, "_validate_preparation_manifest", lambda *_: None)
    monkeypatch.setattr(terminal_policy, "_one_touch_second_command_artifacts", lambda *_: ((), ()))
    prior = terminal_policy.CommandEvidence(0, "verified", "prior", ("drift", "drift"), (), ())
    with pytest.raises(TerminalPolicyError, match="changed authenticated phase inputs"):
        terminal_policy.validate_completed_command(SimpleNamespace(), request, 1, prior)
