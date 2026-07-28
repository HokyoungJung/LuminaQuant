"""Focused behavioural tests for the non-launching v8 acquisition builder."""

from __future__ import annotations

import argparse
import builtins
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/research/create_alpha_max_v8_acquisition_controls.py"


def _module():
    spec = importlib.util.spec_from_file_location("alpha_max_v8_controls", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_json_rejects_duplicates_nonfinite_and_noncanonical(tmp_path: Path):
    module = _module()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"a":1,"a":2}\n')
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"a":NaN}\n')
    whitespace = tmp_path / "whitespace.json"
    whitespace.write_bytes(b'{"a": 1}\n')
    for path in (duplicate, nonfinite, whitespace):
        with pytest.raises(ValueError):
            module._load_canonical(path)
    assert module._canonical({"b": 1, "a": "x"}) == b'{"a":"x","b":1}\n'


def test_paths_are_lexically_rejected_before_filesystem_access():
    module = _module()
    for value in (
        "relative",
        "/tmp/../tmp",
        "/tmp/v7-control",
        "/tmp/v6-control",
        "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source/x",
        "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc/x",
    ):
        with pytest.raises(ValueError):
            module._absolute(value)


def test_stable_regular_read_rejects_symlink_and_accepts_exact_file(tmp_path: Path):
    module = _module()
    target = tmp_path / "target"
    target.write_bytes(b"content")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        module._read_regular(link)
    info, content = module._read_regular(target)
    assert content == b"content"
    assert info.st_size == len(content)


def test_freeze_executes_pinned_fd_and_rejects_replaced_pathname(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    interpreter = tmp_path / "python"
    replacement = tmp_path / "replacement"
    replacement_marker = tmp_path / "replacement-executed"
    shutil.copyfile(sys.executable, interpreter)
    interpreter.chmod(0o500)
    expected = module._file(interpreter)
    replacement.write_text(f"#!/bin/sh\nprintf replacement > {replacement_marker!s}\nexit 1\n")
    replacement.chmod(0o500)
    real_run = module.subprocess.run
    calls = 0

    def replace_pathname(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["executable"].startswith("/proc/self/fd/")
        assert kwargs["pass_fds"]
        os.replace(replacement, interpreter)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", replace_pathname)
    with pytest.raises(ValueError, match="interpreter identity drift"):
        module._freeze(expected)
    assert calls == 1
    assert not replacement_marker.exists()


def test_durable_writer_handles_partial_writes_and_preserves_failed_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    original_write = module.os.write
    calls = 0

    def partial(fd, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        return original_write(fd, data[:1])

    monkeypatch.setattr(module.os, "write", partial)
    with pytest.raises(OSError):
        module._write_bytes_new(tmp_path / "failure.bin", b"long value")
    assert (tmp_path / "failure.bin").read_bytes() == b"l"


def test_durable_writer_preserves_preexisting_leaf(tmp_path: Path):
    module = _module()
    path = tmp_path / "existing.bin"
    path.write_bytes(b"original")
    before = (path.stat().st_dev, path.stat().st_ino, path.read_bytes())
    with pytest.raises(FileExistsError):
        module._write_bytes_new(path, b"replacement")
    assert (path.stat().st_dev, path.stat().st_ino, path.read_bytes()) == before


def test_complete_publication_is_atomic_and_no_replace(tmp_path: Path):
    module = _module()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    first = {
        "schema": "alpha_max_v8_acquisition_complete.v3",
        "launch_performed": False,
        "manifest_sha256": "a" * 64,
    }
    identity = module._publish_complete(control, first)
    complete = control / "COMPLETE.json"
    pending = control / ".COMPLETE.json.pending"
    assert module._load_canonical(complete) == first
    assert module._file(complete) == identity
    assert not pending.exists()

    second = {**first, "manifest_sha256": "b" * 64}
    before = (complete.stat().st_dev, complete.stat().st_ino, complete.read_bytes())
    with pytest.raises(FileExistsError):
        module._publish_complete(control, second)
    assert (complete.stat().st_dev, complete.stat().st_ino, complete.read_bytes()) == before
    assert module._load_canonical(pending) == second


def test_complete_prepublication_failure_never_creates_final_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    real_write = module._write_new

    def fail_pending(path: Path, value: object):
        if path.name == ".COMPLETE.json.pending":
            path.write_bytes(b"{")
            path.chmod(0o600)
            raise OSError("injected pending write failure")
        return real_write(path, value)

    monkeypatch.setattr(module, "_write_new", fail_pending)
    with pytest.raises(OSError, match="pending write failure"):
        module._publish_complete(
            control,
            {
                "schema": "alpha_max_v8_acquisition_complete.v3",
                "launch_performed": False,
                "manifest_sha256": "a" * 64,
            },
        )
    assert not (control / "COMPLETE.json").exists()
    assert (control / ".COMPLETE.json.pending").read_bytes() == b"{"


def test_quarantine_truthfully_records_present_completion_marker(tmp_path: Path):
    module = _module()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    (control / "COMPLETE.json").write_bytes(b"{")
    module._quarantine(
        control,
        {"control_root": module._directory(control, private=True)},
        RuntimeError("injected terminal publication failure"),
    )
    failure = module._load_canonical(control / "FAILED.json")
    assert failure["complete_absent"] is False


def test_key_bindings_use_one_snapshot_and_reject_late_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    root = tmp_path / "keys"
    root.mkdir(mode=0o700)
    originals = {}
    for name in ("authority", *module._SCOPES):
        for kind in ("private", "public"):
            data = f"{name}-{kind}".encode().ljust(32, b"_")
            path = root / f"{name}.{kind}"
            path.write_bytes(data)
            path.chmod(0o400)
            originals[(name, kind)] = data
    replacement = tmp_path / "replacement.public"
    replacement.write_bytes(b"replacement-public".ljust(32, b"_"))
    replacement.chmod(0o400)
    real_read = module._read_regular
    replaced = False

    def replace_after_snapshot(path: Path):
        nonlocal replaced
        info, data = real_read(path)
        if Path(path).name == "authority.public" and not replaced:
            replaced = True
            os.replace(replacement, path)
        return info, data

    monkeypatch.setattr(module, "_read_regular", replace_after_snapshot)
    authority, _, summary, key_files = module._key_bindings(root)
    assert authority["key_id"] == module._sha(originals[("authority", "public")])
    assert summary["keys"][0]["public"] == key_files[0]["public"]
    with pytest.raises(ValueError, match="terminal key identity drift"):
        module._revalidate_key_files(key_files)


def test_inventory_includes_untracked_content_type_mode_and_size(tmp_path: Path, monkeypatch):
    module = _module()
    tracked = tmp_path / "tracked"
    untracked = tmp_path / "untracked"
    tracked.write_bytes(b"a")
    untracked.write_bytes(b"bc")
    monkeypatch.setattr(module, "_git", lambda root, *args: b"tracked\0untracked\0")
    inventory = json.loads(module._inventory(tmp_path))
    assert inventory == [
        {
            "mode": os.stat(tracked).st_mode & 0o7777,
            "path": "tracked",
            "sha256": module._sha(b"a"),
            "size": 1,
            "type": "regular",
        },
        {
            "mode": os.stat(untracked).st_mode & 0o7777,
            "path": "untracked",
            "sha256": module._sha(b"bc"),
            "size": 2,
            "type": "regular",
        },
    ]


def test_supported_systemd_plan_contract_is_fixed():
    module = _module()
    unit = module._unit(
        "test",
        ["python", "worker", "--private-key", "%d/authority.private"],
        {"HOME": "/safe"},
        {"high": 1, "max": 2, "swap": 3},
        ["capture"],
        read_paths=["/control"],
        write_paths=["/evidence", "/telemetry"],
        inaccessible_paths=["/keys"],
        load_credential="authority.private:/keys/authority.private",
    )
    service = unit["Service"]
    assert (
        service["MemoryHigh"] == 1 and service["MemoryMax"] == 2 and service["MemorySwapMax"] == 3
    )
    assert service["OOMPolicy"] == "kill"
    assert "CGroupExpectation" not in unit
    assert set(unit) == module._UNIT_DIRECTIVES
    assert set(service) <= module._SERVICE_DIRECTIVES
    assert "MemoryOOMGroup" not in service and "ExpectedMemoryOOMGroup" not in service
    assert service["ProtectSystem"] == "strict" and service["ProtectHome"] == "read-only"
    assert service["ReadWritePaths"] == ["/evidence", "/telemetry"]
    assert service["InaccessiblePaths"] == ["/keys"]
    assert service["LoadCredential"] == ["authority.private:/keys/authority.private"]
    assert service["IPAddressDeny"] == "any" and service["RestrictAddressFamilies"] == ["AF_UNIX"]
    service["ReadOnlyPaths"] = ["/evidence"]
    with pytest.raises(ValueError, match="paths overlap"):
        module._validate_unit(unit)
    service["ReadOnlyPaths"] = ["/control"]
    service["LoadCredential"] = []
    with pytest.raises(ValueError, match="exactly one credential"):
        module._validate_unit(unit)
    service["LoadCredential"] = ["authority.private:/keys/authority.private"]
    service["ReadOnlyPaths"] = ["/keys"]
    with pytest.raises(ValueError, match="key root must not be readable or writable"):
        module._validate_unit(unit)
    service["ReadOnlyPaths"] = ["/control"]
    service["LoadCredential"] = ["authority.private:/other/authority.private"]
    with pytest.raises(ValueError, match="directly under the inaccessible key root"):
        module._validate_unit(unit)
    service["LoadCredential"] = ["authority.private:/keys/sibling.private"]
    with pytest.raises(ValueError, match="credential name does not match its source"):
        module._validate_unit(unit)
    service["LoadCredential"] = ["authority.private:/keys/authority.private"]
    service["ExecStart"][-1] = "/keys/authority.private"
    with pytest.raises(ValueError, match="does not match unit role argv"):
        module._validate_unit(unit)
    service["ExecStart"][-1] = "%d/authority.private"
    service["LoadCredential"] = "authority.private:/keys/authority.private"
    module._validate_unit(unit)
    observer = module._unit(
        "observer",
        ["python", "observer", "--observer-private-key", "%d/acquisition.private"],
        {"HOME": "/safe"},
        {"high": 1, "max": 2, "swap": 3},
        ["capture"],
        read_paths=["/control"],
        write_paths=["/evidence", "/telemetry", "/output"],
        inaccessible_paths=["/keys"],
        load_credential="acquisition.private:/keys/acquisition.private",
        observer=True,
    )["Service"]
    assert "IPAddressDeny" not in observer
    assert observer["RestrictAddressFamilies"] == ["AF_UNIX", "AF_INET", "AF_INET6"]
    telemetry = module._unit(
        "telemetry",
        ["python", "monitor", "--authority-public-key", "%d/authority.public"],
        {"HOME": "/safe"},
        {"high": 1, "max": 2, "swap": 3},
        None,
        read_paths=["/control"],
        write_paths=["/telemetry"],
        inaccessible_paths=["/keys"],
        load_credential="authority.public:/keys/authority.public",
    )["Service"]
    assert "ExecStopPost" not in telemetry


def test_import_is_stdlib_only_and_policy_constants_are_local():
    module = _module()
    assert not hasattr(module, "policy")
    assert module._SCOPES == ("acquisition", "phase_preparation", "one_touch")
    assert module._FILE_ROLES[0] == "policy_json"
    assert module.G067_IDENTITY["checksum_sha256"] != module.G067_IDENTITY["archive_sha256"]


def test_authenticated_policy_and_key_creator_use_only_captured_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    policy = tmp_path / "policy.py"
    policy.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Value:\n    value: int\n"
        "source_sha256='verified-policy'\n"
        "_SCOPES=('acquisition','phase_preparation','one_touch')\n"
        f"_FILE_ROLES={module._FILE_ROLES!r}\n"
        f"_FORBIDDEN_ROOTS={module._FORBIDDEN_ROOTS!r}\n"
    )
    sys.modules.pop("alpha_max_v8_verified_policy", None)
    loaded = module._load_policy(policy, module._file(policy))
    assert loaded.Value(1).value == 1
    assert sys.modules["alpha_max_v8_verified_policy"] is loaded
    with pytest.raises(ValueError, match="already registered"):
        module._load_policy(policy, module._file(policy))

    ambient = tmp_path / "ambient" / "lumina_quant"
    ambient.mkdir(parents=True)
    malicious_marker = tmp_path / "ambient-policy-executed"
    (ambient / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(malicious_marker)!r}).write_text('bad')\n"
    )
    monkeypatch.syspath_prepend(str(ambient.parent))
    monkeypatch.delitem(sys.modules, "lumina_quant", raising=False)
    monkeypatch.delitem(sys.modules, "lumina_quant.alpha_max_terminal_policy", raising=False)
    creator_file = tmp_path / "creator.py"
    creator_file.write_text(
        "from lumina_quant import alpha_max_terminal_policy as policy\n"
        "def create_keys(root):\n"
        "    (root / 'created').write_text(policy.source_sha256)\n"
    )
    sys.modules.pop("alpha_max_v8_verified_key_creator", None)
    creator = module._load_key_creator(creator_file, module._file(creator_file), loaded)
    creator(tmp_path)
    assert (tmp_path / "created").read_text() == "verified-policy"
    assert not malicious_marker.exists()
    assert "lumina_quant" not in sys.modules
    assert "lumina_quant.alpha_max_terminal_policy" not in sys.modules
    sys.modules.pop("alpha_max_v8_verified_key_creator", None)
    sys.modules.pop("alpha_max_v8_verified_policy", None)


def test_authenticated_key_creator_rejects_ambient_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    creator_file = tmp_path / "creator.py"
    creator_file.write_text(
        "from lumina_quant import alpha_max_terminal_policy as policy\n"
        "def create_keys(root):\n    pass\n"
    )
    monkeypatch.setitem(sys.modules, "lumina_quant", SimpleNamespace())
    with pytest.raises(ValueError, match="dependency already registered"):
        module._load_key_creator(creator_file, module._file(creator_file), SimpleNamespace())
    assert "alpha_max_v8_verified_key_creator" not in sys.modules


def test_authenticated_module_is_removed_after_execution_failure(tmp_path: Path):
    module = _module()
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('boom')\n")
    sys.modules.pop("alpha_max_v8_verified_policy", None)
    with pytest.raises(RuntimeError, match="boom"):
        module._load_policy(broken, module._file(broken))
    assert "alpha_max_v8_verified_policy" not in sys.modules


def test_policy_semantic_rejection_removes_authenticated_alias(tmp_path: Path):
    module = _module()
    policy = tmp_path / "bad-policy.py"
    policy.write_text(
        "_SCOPES=()\n"
        f"_FILE_ROLES={module._FILE_ROLES!r}\n"
        f"_FORBIDDEN_ROOTS={module._FORBIDDEN_ROOTS!r}\n"
    )
    sys.modules.pop("alpha_max_v8_verified_policy", None)
    with pytest.raises(ValueError, match="constants mismatch"):
        module._load_policy(policy, module._file(policy))
    assert "alpha_max_v8_verified_policy" not in sys.modules


def test_authenticated_loader_executes_only_captured_bytes_and_rejects_late_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    policy = tmp_path / "policy.py"
    approved_marker, malicious_marker = tmp_path / "approved", tmp_path / "malicious"
    policy.write_text(
        "from dataclasses import dataclass\n"
        "from pathlib import Path\n"
        f"Path({str(approved_marker)!r}).write_text('approved')\n"
        "@dataclass\nclass Value:\n    value: int\n"
        "_SCOPES=('acquisition','phase_preparation','one_touch')\n"
        f"_FILE_ROLES={module._FILE_ROLES!r}\n"
        f"_FORBIDDEN_ROOTS={module._FORBIDDEN_ROOTS!r}\n"
    )
    expected = module._file(policy)
    malicious = (
        "from pathlib import Path\n"
        f"Path({str(malicious_marker)!r}).write_text('malicious')\n"
        "_SCOPES=()\n_FILE_ROLES=()\n_FORBIDDEN_ROOTS=()\n"
    )
    original_compile = builtins.compile

    def replace_after_capture(source, filename, mode, *args, **kwargs):
        policy.write_text(malicious)
        return original_compile(source, filename, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "compile", replace_after_capture)
    sys.modules.pop("alpha_max_v8_verified_policy", None)
    with pytest.raises(ValueError, match="identity drift"):
        module._load_policy(policy, expected)
    assert approved_marker.read_text() == "approved"
    assert not malicious_marker.exists()
    assert "alpha_max_v8_verified_policy" not in sys.modules


def test_root_topology_and_quarantine_contract(tmp_path: Path):
    module = _module()
    run_id = "a" * 64
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    expected = {
        "control_root": recovery / f"g056v8-controls-{run_id}",
        "key_root": recovery / f"g056v8-keys-{run_id}",
        "evidence_root": recovery / f"g056v8-acquisition-evidence-{run_id}",
        "telemetry_root": recovery / f"g056v8-telemetry-{run_id}",
        "output_parent": recovery / f"g056v8-acquisition-output-{run_id}",
    }
    for root in expected.values():
        module._create_root(root)
    identities = {name: module._directory(root, private=True) for name, root in expected.items()}
    module._quarantine(
        expected["control_root"], identities, RuntimeError("injected post-key failure")
    )
    assert (expected["control_root"] / "FAILED.json").exists()
    assert not (expected["control_root"] / "COMPLETE.json").exists()
    failure = module._load_canonical(expected["control_root"] / "FAILED.json")
    assert set(failure) == {"schema", "error", "created_roots", "complete_absent"}
    assert failure["complete_absent"] is True
    assert not (expected["control_root"] / "COMPLETE.json").exists()


def test_build_assembles_private_artifacts_without_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    current, accepted = inputs / "current", inputs / "accepted"
    current.mkdir()
    accepted.mkdir()
    for root in (current, accepted):
        (root / "tracked").write_bytes(b"tracked")
    alignment = inputs / "alignment.json"
    alignment.write_bytes(b"alignment")
    telemetry_script = inputs / "telemetry.py"
    telemetry_script.write_text("raise AssertionError('must not execute')\n")
    current_python = inputs / "current-python"
    accepted_python = inputs / "accepted-python"
    for interpreter in (current_python, accepted_python):
        interpreter.write_bytes(b"synthetic interpreter")
    role_root = inputs / "roles"
    role_root.mkdir()
    role_pins = {}
    for role in module.ROLE_PINS:
        name = f"{role}.py"
        (role_root / name).write_text(f"# {role}\n")
        role_pins[role] = (str(role_root), name, None)
    g067_approval, current_approval = inputs / "g067.json", inputs / "current.json"
    g067_approval.write_text("{}\n")
    current_approval.write_text("{}\n")
    monkeypatch.setattr(module, "RECOVERY_ROOT", recovery)
    monkeypatch.setattr(module, "CURRENT", str(current))
    monkeypatch.setattr(module, "ACCEPTED", str(accepted))
    monkeypatch.setattr(module, "ALIGNMENT", str(alignment))
    monkeypatch.setattr(module, "ALIGNMENT_SHA256", module._sha(b"alignment"))
    monkeypatch.setattr(module, "G067_APPROVAL", str(g067_approval))
    monkeypatch.setattr(module, "CURRENT_APPROVAL", str(current_approval))
    monkeypatch.setattr(module, "ROLE_PINS", role_pins)

    def fake_git(root: Path, *args: str) -> bytes:
        if args == ("rev-parse", "HEAD"):
            return module.ACCEPTED_COMMIT.encode() + b"\n"
        if args == ("ls-files", "-z", "--cached", "--others", "--exclude-standard"):
            return b"tracked\0"
        return b""

    monkeypatch.setattr(module, "_git", fake_git)
    monkeypatch.setattr(module, "_state", lambda root, approval: module.ACCEPTED_COMMIT)

    def fake_approval(path: Path, fields: set[str], schema: str) -> dict[str, object]:
        if schema == "alpha_max_g067_sol_archive_approval.v1":
            return {"schema": schema, **module.G067_IDENTITY, "approved_utc": "now"}
        return {
            field: (
                schema
                if field == "schema"
                else str(current)
                if field == "repository_root"
                else module.ACCEPTED_COMMIT
                if field in {"head", "accepted_alpha_commit"}
                else module.BASELINE
                if field == "baseline_ancestor"
                else "PASS_REVIEWED_OVERLAY"
                if field == "verdict"
                else {}
            )
            for field in fields
        }

    monkeypatch.setattr(module, "_approval", fake_approval)
    monkeypatch.setattr(
        module,
        "_freeze",
        lambda interpreter: {
            "schema": "alpha_max_v8_interpreter_package_freeze.v1",
            "interpreter_sha256": interpreter["sha256"],
            "implementation": "cpython",
            "python_version": "3.12.0",
            "cache_tag": "cpython-312",
            "packages": [],
        },
    )

    def executable_pin(name: str, path: Path, freeze: dict[str, object] | None = None):
        identity = module._file(path)
        pin = {
            "path": str(path),
            "sha256": identity["sha256"],
            "byte_count": identity["byte_count"],
            "mode": identity["mode"],
        }
        if freeze is not None:
            pin["package_freeze_sha256"] = module._sha(module._canonical(freeze))
        return pin

    module.EXECUTABLE_PINS = {
        "current_python": executable_pin(
            "current_python", current_python, module._freeze(module._file(current_python))
        ),
        "accepted_python": executable_pin(
            "accepted_python", accepted_python, module._freeze(module._file(accepted_python))
        ),
        "telemetry_script": executable_pin("telemetry_script", telemetry_script),
    }

    def create_keys(root: Path) -> None:
        for name in ("authority", *module._SCOPES):
            for kind in ("private", "public"):
                path = root / f"{name}.{kind}"
                path.write_bytes(f"{name}-{kind}".encode().ljust(32, b"_"))
                path.chmod(0o400)

    monkeypatch.setattr(module, "_load_key_creator", lambda path, identity, policy: create_keys)
    policy = SimpleNamespace(source_sha256="policy-source")

    def load_checkpoint(path, loaded_policy):
        return SimpleNamespace(sha256=module._file(path)["sha256"])

    def commands(envelope, request):
        common = [
            str(current_python),
            str(role_root / "acquirer.py"),
            "--contract-manifest",
            str(role_root / "contract_manifest.py"),
            "--availability-evidence",
            str(role_root / "availability_evidence.py"),
            "--output-root",
            request.value["source_root"]["path"],
            "--report-dir",
            request.value["report_root"]["path"],
            "--forbidden-root",
            module._FORBIDDEN_ROOTS[0],
            "--forbidden-root",
            module._FORBIDDEN_ROOTS[1],
        ]
        return [(*common, "--execute", "--validate-complete"), (*common, "--verify-eligible")]

    policy.load_policy = lambda path: policy
    policy.load_checkpoint = load_checkpoint
    policy.load_envelope = lambda path, loaded_policy, checkpoint: SimpleNamespace()
    policy.load_request = lambda path, **kwargs: SimpleNamespace(value=module._load_canonical(path))
    policy.derive_scope_commands = commands
    monkeypatch.setattr(module, "_load_policy", lambda path, identity: policy)
    launches: list[object] = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: launches.append((args, kwargs)),
    )

    def make_args(run_id: str) -> argparse.Namespace:
        return argparse.Namespace(
            control_root=str(recovery / f"g056v8-controls-{run_id}"),
            key_root=str(recovery / f"g056v8-keys-{run_id}"),
            evidence_root=str(recovery / f"g056v8-acquisition-evidence-{run_id}"),
            telemetry_root=str(recovery / f"g056v8-telemetry-{run_id}"),
            output_parent=str(recovery / f"g056v8-acquisition-output-{run_id}"),
            telemetry_script=str(telemetry_script),
            g067_approval=str(g067_approval),
            current_approval=str(current_approval),
            current_python=str(current_python),
            accepted_python=str(accepted_python),
            run_id=run_id,
            request_id="b" * 64 if run_id == "a" * 64 else "d" * 64,
        )

    args = make_args("a" * 64)
    artifacts = module.build(args)
    control = Path(args.control_root)
    for path in artifacts.values():
        assert Path(path).is_file()
    for name in (
        "policy.json",
        "checkpoint.json",
        "envelope.json",
        "acquisition-request.json",
        "launch-plan.json",
        "manifest.json",
        "COMPLETE.json",
    ):
        assert (control / name).is_file()
    documents = {
        name: module._load_canonical(control / name)
        for name in (
            "policy.json",
            "checkpoint.json",
            "envelope.json",
            "acquisition-request.json",
            "manifest.json",
        )
    }
    assert documents["policy.json"]["schema"] == "alpha_max_terminal_authority_policy.v3"
    assert documents["checkpoint.json"]["schema"] == "alpha_max_terminal_checkpoint.v1"
    assert documents["envelope.json"]["schema"] == "alpha_max_terminal_launch_envelope.v3"
    assert documents["acquisition-request.json"]["scope"] == "acquisition"
    assert documents["manifest.json"]["launch_performed"] is False
    assert not (Path(args.output_parent) / "source").exists()
    assert not (Path(args.output_parent) / "report").exists()
    plan = module._load_canonical(control / "launch-plan.json")
    assert plan["ordering"] == [
        f"alpha-max-v8-authority-{args.run_id}.service",
        f"alpha-max-v8-telemetry-{args.run_id}.service",
        f"alpha-max-v8-observer-{args.run_id}.service",
    ]
    assert plan["cgroup_contract"] == {"oom_policy_kill_implies_memory_oom_group": 1}
    assert plan["telemetry_contract"]["monitor"][-1] == "536870912"
    assert "86400" in plan["telemetry_contract"]["monitor"]
    authority = plan["systemd_units"]["authority"]["Service"]
    observer = plan["systemd_units"]["observer"]["Service"]
    telemetry = plan["systemd_units"]["telemetry"]["Service"]
    for unit in plan["systemd_units"].values():
        module._validate_unit({key: value for key, value in unit.items() if key != "name"})
        assert not (set(unit["Service"]["ReadOnlyPaths"]) & set(unit["Service"]["ReadWritePaths"]))
    assert (authority["MemoryHigh"], authority["MemoryMax"], authority["MemorySwapMax"]) == (
        268435456,
        536870912,
        67108864,
    )
    assert (observer["MemoryHigh"], observer["MemoryMax"], observer["MemorySwapMax"]) == (
        2147483648,
        3221225472,
        536870912,
    )
    assert (telemetry["MemoryHigh"], telemetry["MemoryMax"], telemetry["MemorySwapMax"]) == (
        67108864,
        134217728,
        33554432,
    )
    assert authority["IPAddressDeny"] == "any"
    assert "IPAddressDeny" not in observer
    assert authority["ExecStopPost"] == plan["telemetry_contract"]["authority_capture"]
    assert observer["ExecStopPost"] == plan["telemetry_contract"]["observer_capture"]
    key_root = Path(args.key_root)
    assert all(
        str(key_root) not in service[path_key]
        for service in (authority, observer, telemetry)
        for path_key in ("ReadOnlyPaths", "ReadWritePaths")
    )
    assert authority["InaccessiblePaths"] == [str(key_root)]
    assert observer["InaccessiblePaths"] == [str(key_root)]
    assert telemetry["InaccessiblePaths"] == [str(key_root)]
    assert authority["LoadCredential"] == [f"authority.private:{key_root / 'authority.private'}"]
    assert observer["LoadCredential"] == [f"acquisition.private:{key_root / 'acquisition.private'}"]
    assert telemetry["LoadCredential"] == [f"authority.public:{key_root / 'authority.public'}"]
    assert authority["ExecStart"][authority["ExecStart"].index("--private-key") + 1] == (
        "%d/authority.private"
    )
    assert (
        observer["ExecStart"][observer["ExecStart"].index("--observer-private-key") + 1]
        == "%d/acquisition.private"
    )
    assert (
        telemetry["ExecStart"][telemetry["ExecStart"].index("--authority-public-key") + 1]
        == "%d/authority.public"
    )
    assert documents["manifest.json"]["executable_inputs"] == plan["executable_inputs"]
    assert (
        plan["executable_inputs"]["current_python"]["package_freeze_sha256"]
        == (module.EXECUTABLE_PINS["current_python"]["package_freeze_sha256"])
    )
    assert (
        plan["executable_inputs"]["accepted_python"]["package_freeze_sha256"]
        == (module.EXECUTABLE_PINS["accepted_python"]["package_freeze_sha256"])
    )
    assert launches == []
    substituted = make_args("e" * 64)
    replacement = inputs / "replacement-python"
    replacement.write_bytes(b"substituted interpreter")
    substituted.current_python = str(replacement)
    with pytest.raises(ValueError, match="executable path mismatch"):
        module.build(substituted)
    for root in (
        substituted.control_root,
        substituted.key_root,
        substituted.evidence_root,
        substituted.telemetry_root,
        substituted.output_parent,
    ):
        assert not Path(root).exists()
    original_freeze_pin = module.EXECUTABLE_PINS["current_python"]["package_freeze_sha256"]
    module.EXECUTABLE_PINS["current_python"]["package_freeze_sha256"] = "0" * 64
    freeze_substituted = make_args("f" * 64)
    with pytest.raises(ValueError, match="interpreter freeze pin mismatch"):
        module.build(freeze_substituted)
    module.EXECUTABLE_PINS["current_python"]["package_freeze_sha256"] = original_freeze_pin
    for root in (
        freeze_substituted.control_root,
        freeze_substituted.key_root,
        freeze_substituted.evidence_root,
        freeze_substituted.telemetry_root,
        freeze_substituted.output_parent,
    ):
        assert not Path(root).exists()

    original_write = module._write_new

    def late_failure(path: Path, value: object):
        if path.name == "manifest.json":
            raise RuntimeError("late injected failure")
        return original_write(path, value)

    monkeypatch.setattr(module, "_write_new", late_failure)
    failed_args = make_args("c" * 64)
    with pytest.raises(RuntimeError, match="late injected failure"):
        module.build(failed_args)
    failed_control = Path(failed_args.control_root)
    assert (failed_control / "FAILED.json").is_file()
    assert not (failed_control / "COMPLETE.json").exists()


def test_quarantine_failure_preserves_primary_and_quarantine_errors(tmp_path: Path, monkeypatch):
    module = _module()
    primary = RuntimeError("primary build failure")
    with pytest.raises(RuntimeError) as rethrown:
        module._quarantine_or_reraise(None, {}, primary)
    assert rethrown.value is primary

    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    identity = {"control_root": module._directory(control, private=True)}
    original_write = module._write_new

    def failed_quarantine_write(path: Path, value: object):
        if path.name == "FAILED.json":
            raise OSError("injected quarantine write failure")
        return original_write(path, value)

    monkeypatch.setattr(module, "_write_new", failed_quarantine_write)
    with pytest.raises(BaseExceptionGroup) as raised:
        module._quarantine_or_reraise(control, identity, primary)
    assert raised.value.exceptions[0] is primary
    assert isinstance(raised.value.exceptions[1], OSError)
