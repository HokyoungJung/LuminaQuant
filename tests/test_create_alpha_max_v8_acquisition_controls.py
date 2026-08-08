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
PROBE_SCRIPT = (
    Path(__file__).parents[1] / "scripts/research/probe_alpha_max_v8_systemd_credential.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("alpha_max_v8_controls", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe_module():
    spec = importlib.util.spec_from_file_location("alpha_max_v8_probe", PROBE_SCRIPT)
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
    for name in ("authority", "publication", *module._SCOPES):
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
    assert summary["publication_key"]["key_id"] == module._sha(originals[("publication", "public")])
    assert summary["keys"][0]["public"] == key_files[0]["public"]
    with pytest.raises(ValueError, match="terminal key identity drift"):
        module._revalidate_key_files(key_files)


def test_inventory_includes_untracked_content_type_mode_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    versioned = tmp_path / "mode_v6"
    versioned.mkdir()
    tracked = versioned / "tracked"
    untracked = tmp_path / "untracked"
    link = tmp_path / "link"
    tracked.write_bytes(b"a")
    untracked.write_bytes(b"bc")
    link.symlink_to("mode_v6/tracked")
    monkeypatch.setattr(
        module,
        "_git",
        lambda root, *args: b"mode_v6/tracked\0untracked\0link\0",
    )
    inventory = json.loads(module._inventory(tmp_path))
    assert inventory == [
        {
            "mode": os.stat(tracked).st_mode & 0o7777,
            "path": "mode_v6/tracked",
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
        {
            "mode": os.lstat(link).st_mode & 0o7777,
            "path": "link",
            "sha256": module._sha(b"mode_v6/tracked"),
            "size": len(b"mode_v6/tracked"),
            "type": "symlink",
        },
    ]


def test_runtime_inventory_binds_regular_and_symlink_mutations(tmp_path: Path):
    module = _module()
    runtime = tmp_path / "venv"
    executable = runtime / "bin" / "python"
    package = runtime / "lib" / "package.py"
    executable.parent.mkdir(parents=True)
    package.parent.mkdir()
    executable.write_bytes(b"python")
    package.write_bytes(b"first")
    link = runtime / "bin" / "package-link"
    link.symlink_to("../lib/package.py")
    identity = {"path": str(executable)}
    initial = module._runtime_inventory(identity)
    package.write_bytes(b"second")
    assert module._runtime_inventory(identity) != initial
    package.write_bytes(b"first")
    link.unlink()
    link.symlink_to("../lib/other.py")
    with pytest.raises(ValueError, match="runtime symlink target is missing"):
        module._runtime_inventory(identity)


def test_ignored_importable_source_is_bound_by_approval_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = _module()
    ignored = tmp_path / "src" / "ignored_import.py"
    ignored.parent.mkdir()
    ignored.write_text("value = 1\n")
    monkeypatch.setattr(
        module,
        "_git",
        lambda _root, *args: b"src/ignored_import.py\0" if "--ignored" in args else b"",
    )
    initial = module._ignored_source_inventory(tmp_path)
    ignored.write_text("value = 2\n")
    assert module._ignored_source_inventory(tmp_path) != initial
    assert module._only_owner_session_runtime_changes(tmp_path)


def test_source_approval_excludes_only_owner_session_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "source.py"
    runtime = repo / module._OWNER_SESSION_RUNTIME_PATH / "state.json"
    runtime.parent.mkdir(parents=True)
    source.write_text("approved = True\n")
    runtime.write_text('{"state":"initial"}\n')

    def git(*args: str) -> str:
        return module.subprocess.run(
            ("git", *args),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.com")
    git("add", ".")
    git("commit", "-m", "initial")
    head = git("rev-parse", "HEAD")

    recovery = tmp_path / "recovery"
    recovery.mkdir()
    approval_path = recovery / module.CURRENT_APPROVAL_LEAF
    monkeypatch.setattr(module, "CURRENT", str(repo))
    monkeypatch.setattr(module, "RECOVERY_ROOT", recovery)
    monkeypatch.setattr(module, "CURRENT_APPROVAL", str(approval_path))
    monkeypatch.setattr(module, "ACCEPTED_COMMIT", head)
    monkeypatch.setattr(module, "BASELINE", head)
    execution_alias = tmp_path / "execution-alias"
    execution_alias.symlink_to(recovery, target_is_directory=True)
    monkeypatch.setattr(module, "EXECUTION_ALIAS_ROOT", execution_alias)
    monkeypatch.setattr(
        module,
        "_preflight_executables",
        lambda _paths: {
            "current_python": {"path": "/runtime/current/bin/python"},
            "accepted_python": {"path": "/runtime/accepted/bin/python"},
            "base_python": {"path": "/runtime/base/bin/python"},
            "telemetry_script": {"path": "/runtime/telemetry"},
        },
    )
    monkeypatch.setattr(module, "_runtime_inventory", lambda _interpreter: b"[]\n")
    request_ids = {
        "acquisition": "a" * 64,
        "phase_preparation": "b" * 64,
        "one_touch": "c" * 64,
    }
    approval = module._create_current_approval(
        approval_path,
        root=repo,
        run_id="d" * 64,
        request_ids=request_ids,
        absent_paths={},
    )

    runtime.write_text('{"state":"active"}\n')
    assert module._only_owner_session_runtime_changes(repo)
    assert (
        module._load_current_approval(
            approval_path,
            run_id="d" * 64,
            request_ids=request_ids,
            absent_paths={},
        )
        == approval
    )

    source.write_text("approved = False\n")
    assert not module._only_owner_session_runtime_changes(repo)
    with pytest.raises(ValueError, match="current approval mismatch"):
        module._load_current_approval(
            approval_path,
            run_id="d" * 64,
            request_ids=request_ids,
            absent_paths={},
        )

    source.write_text("approved = True\n")
    (repo / "untracked.py").write_text("unapproved = True\n")
    assert not module._only_owner_session_runtime_changes(repo)
    with pytest.raises(ValueError, match="current approval mismatch"):
        module._load_current_approval(
            approval_path,
            run_id="d" * 64,
            request_ids=request_ids,
            absent_paths={},
        )


def test_supported_systemd_plan_contract_is_fixed():
    module = _module()
    module.RECOVERY_ROOT = Path("/")
    module.EXECUTION_ALIAS_ROOT = Path("/execution-alias")
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
    assert service["UMask"] == "0077"
    assert "CGroupExpectation" not in unit
    assert set(unit) == module._UNIT_DIRECTIVES
    assert set(service) <= module._SERVICE_DIRECTIVES
    assert "MemoryOOMGroup" not in service and "ExpectedMemoryOOMGroup" not in service
    assert service["ProtectSystem"] == "strict" and service["ProtectHome"] == "tmpfs"
    assert service["BindPaths"] == [
        "/execution-alias/evidence:/evidence",
        "/execution-alias/telemetry:/telemetry",
    ]
    assert service["InaccessiblePaths"] == ["-/keys"]
    assert service["LoadCredential"] == ["authority.private:/keys/authority.private"]
    assert service["IPAddressDeny"] == "any" and service["RestrictAddressFamilies"] == ["AF_UNIX"]
    service["BindReadOnlyPaths"] = ["/evidence"]
    with pytest.raises(ValueError, match="writable binds overlap"):
        module._validate_unit(unit)
    service["BindReadOnlyPaths"] = ["/control"]
    service["LoadCredential"] = []
    with pytest.raises(ValueError, match="exactly one credential"):
        module._validate_unit(unit)
    service["LoadCredential"] = ["authority.private:/keys/authority.private"]
    service["BindReadOnlyPaths"] = ["/keys"]
    with pytest.raises(ValueError, match="key root must not be readable or writable"):
        module._validate_unit(unit)
    service["BindReadOnlyPaths"] = ["/control"]
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
    rendered = module._render_systemd_unit(unit)
    assert rendered == module._render_systemd_unit(unit)
    assert b"\nUMask=0077\n" in rendered
    assert b'"%d/authority.private"' in rendered
    assert b"%%d" not in rendered
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
    observer["RestrictAddressFamilies"] = ["AF_UNIX", "AF_INET"]
    with pytest.raises(ValueError, match="must allow only Unix and Internet"):
        module._validate_unit(
            {
                "Description": "observer",
                "After": ["network-online.target"],
                "Wants": ["network-online.target"],
                "Service": observer,
            }
        )
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


def test_persisted_probe_renders_and_consumes_role_specific_credential_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _module()
    probe = _probe_module()
    module.RECOVERY_ROOT = Path("/recovery")
    module.EXECUTION_ALIAS_ROOT = Path("/execution-alias")
    request = tmp_path / "request.json"
    approval = tmp_path / "approval.json"
    request.write_text("{}", encoding="utf-8")
    approval.write_text("{}", encoding="utf-8")
    roles = (
        ("authority.private", "--private-key", False),
        ("authority.public", "--authority-public-key", False),
        ("acquisition.private", "--observer-private-key", True),
        ("phase_preparation.private", "--observer-private-key", False),
        ("one_touch.private", "--observer-private-key", False),
    )
    for credential, flag, observer in roles:
        original = module._unit(
            "probe",
            ["worker", flag, f"%d/{credential}"],
            {"HOME": "/safe"},
            {"high": 1, "max": 2, "swap": 3},
            None,
            read_paths=["/control"],
            write_paths=["/recovery/evidence"],
            inaccessible_paths=["/keys"],
            load_credential=f"{credential}:/keys/{credential}",
            observer=observer,
        )
        credential_file = tmp_path / credential
        credential_file.write_bytes(b"credential")
        marker = tmp_path / f"{credential}.marker"
        release = tmp_path / f"{credential}.release"
        release.write_text("", encoding="utf-8")
        command = probe._persisted_probe_execstart(
            module,
            credential,
            f"%d/{credential}",
            module._sha(b"credential"),
            marker,
            release,
            "acquisition",
            "probe",
            str(request),
            "0" * 64,
            approval,
        )
        original["Service"]["ExecStart"] = command
        module._validate_unit(original)
        rendered = module._render_systemd_unit(original)
        assert flag in command
        assert "--credential" not in command
        assert flag.encode() in rendered

        argv = ["probe", *command[5:]]
        credential_index = argv.index(f"%d/{credential}")
        argv[credential_index] = str(credential_file)
        monkeypatch.setattr(sys, "argv", argv)
        exec(probe._PERSISTED_PROBE_CODE, {})
        assert json.loads(marker.read_text(encoding="utf-8"))["credential_path"] == str(
            credential_file
        )

        original["Service"]["ExecStart"][original["Service"]["ExecStart"].index(flag)] = (
            "--credential"
        )
        with pytest.raises(ValueError, match="does not match unit role argv"):
            module._validate_unit(original)
        wrong_flag = "--private-key" if flag != "--private-key" else "--authority-public-key"
        original["Service"]["ExecStart"][original["Service"]["ExecStart"].index("--credential")] = (
            wrong_flag
        )
        with pytest.raises(ValueError, match="does not match unit role argv"):
            module._validate_unit(original)
        monkeypatch.setattr(
            sys,
            "argv",
            ["probe", wrong_flag, str(credential_file), "--credential-name", credential],
        )
        with pytest.raises(ValueError):
            exec(probe._PERSISTED_PROBE_CODE, {})


def test_writable_bind_rejects_every_non_normal_path_component():
    module = _module()
    module.RECOVERY_ROOT = Path("/recovery")
    module.EXECUTION_ALIAS_ROOT = Path("/execution-alias")
    unit = module._unit(
        "test",
        ["python", "worker", "--private-key", "%d/authority.private"],
        {"HOME": "/safe"},
        {"high": 1, "max": 2, "swap": 3},
        None,
        read_paths=["/control"],
        write_paths=["/recovery/evidence"],
        inaccessible_paths=["/keys"],
        load_credential="authority.private:/keys/authority.private",
    )
    for source, destination in (
        ("/execution-alias/..", "/recovery/.."),
        ("/execution-alias/evidence/.", "/recovery/evidence/."),
        (
            "/execution-alias/evidence/../telemetry",
            "/recovery/evidence/../telemetry",
        ),
    ):
        unit["Service"]["BindPaths"] = [f"{source}:{destination}"]
        with pytest.raises(ValueError, match="canonically absolute"):
            module._validate_unit(unit)


def test_execstart_mount_wrapper_binds_verified_scoped_admission(tmp_path: Path):
    module = _module()
    destination = tmp_path / "destination"
    destination.mkdir(mode=0o700)
    wrapped = module._wrap_execstart(
        ["/bin/true"],
        [str(destination)],
        receipt="/control/admissions/launch-admission-acquisition.json",
        unit_name="alpha-max-v8-acquisition-authority.service",
        authority_public_b64="a" * 44,
    )
    assert wrapped[:5] == ["/usr/bin/python3", "-I", "-S", "-c", module._MOUNT_IDENTITY_CODE]
    config = json.loads(wrapped[5])
    assert config["receipt"] == "/control/admissions/launch-admission-acquisition.json"
    assert config["scope"] == "acquisition"
    assert config["unit_name"] == "alpha-max-v8-acquisition-authority.service"
    assert "control_artifacts" in module._MOUNT_IDENTITY_SOURCE
    assert "pkeyutl" in module._MOUNT_IDENTITY_SOURCE
    assert "assert artifacts" in module._MOUNT_IDENTITY_SOURCE
    assert "signed_payload_fd = os.memfd_create" in module._MOUNT_IDENTITY_SOURCE
    assert 'f"/proc/self/fd/{signed_payload_fd}"' in module._MOUNT_IDENTITY_SOURCE
    assert "os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC" in module._MOUNT_IDENTITY_SOURCE
    assert "def stat_tuple(info):" in module._MOUNT_IDENTITY_SOURCE
    assert "info.st_mtime_ns" in module._MOUNT_IDENTITY_SOURCE
    assert "info.st_ctime_ns" in module._MOUNT_IDENTITY_SOURCE
    assert "def check_snapshot(path, fd, expected, digest):" in module._MOUNT_IDENTITY_SOURCE
    assert "for fd in fds" in module._MOUNT_IDENTITY_SOURCE
    assert "\n" not in module._MOUNT_IDENTITY_CODE
    compile(module._MOUNT_IDENTITY_CODE, "<final-namespace-bootstrap>", "exec")


def test_final_wrapper_rechecks_retained_fd_after_same_size_hardlink_mutation(tmp_path: Path):
    module = _module()
    namespace: dict[str, object] = {}
    exec(module._MOUNT_IDENTITY_SOURCE.split("fds = []", 1)[0], namespace)
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"same-size")
    artifact.chmod(0o600)
    namespace["fds"] = []
    fd, expected, digest, _data = namespace["snapshot"](str(artifact))
    (tmp_path / "artifact-link").hardlink_to(artifact)
    with pytest.raises(AssertionError, match="retained fd drift"):
        namespace["check_snapshot"](str(artifact), fd, expected, digest)
    os.close(fd)


def test_approval_and_persisted_topology_bind_accepted_and_control_artifacts():
    probe = _probe_module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"accepted_source_state": _accepted_source_state()' in source
    assert 'approval["accepted_source_state"] != _accepted_source_state()' in source
    assert 'approval.get("accepted_source_state") != accepted_actual' in source
    assert "control_artifacts" in probe.verify_persisted_topology.__code__.co_consts
    assert "control_artifacts" in probe._execute_persisted_topology.__code__.co_consts
    assert "signed control artifact mismatch" in source
    assert "receipt=prestart[5]" in source
    assert "unit_name=prestart[7]" in source
    assert "authority_public_b64=prestart[9]" in source
    probe_source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert "production ExecStart lacks final namespace verifier" in probe_source
    assert 'receipt=wrapper_config["receipt"]' in probe_source


def test_unit_bind_contract_rejects_broad_roots_and_binds_absolute_commands():
    module = _module()
    module.RECOVERY_ROOT = Path("/")
    module.EXECUTION_ALIAS_ROOT = Path("/execution-alias")
    unit = module._unit(
        "test",
        ["/runtime/current/bin/python", "/code/worker.py", "--private-key", "%d/authority.private"],
        {"HOME": "/safe"},
        {"high": 1, "max": 2, "swap": 3},
        None,
        read_paths=["/control"],
        write_paths=["/evidence"],
        inaccessible_paths=["/keys"],
        load_credential="authority.private:/keys/authority.private",
    )
    service = unit["Service"]
    assert "/runtime/current/bin/python" in service["BindReadOnlyPaths"]
    assert "/code/worker.py" in service["BindReadOnlyPaths"]
    service["BindReadOnlyPaths"].append(module.CURRENT)
    with pytest.raises(ValueError, match="broad home path"):
        module._validate_unit(unit)
    service["BindReadOnlyPaths"].remove(module.CURRENT)
    service["BindReadOnlyPaths"].append(module.ACCEPTED)
    with pytest.raises(ValueError, match="broad home path"):
        module._validate_unit(unit)


def test_non_network_unit_shapes_are_unix_only_and_ip_denied():
    module = _module()
    module.RECOVERY_ROOT = Path("/")
    module.EXECUTION_ALIAS_ROOT = Path("/execution-alias")
    for credential, argument in (
        ("authority.private", "--private-key"),
        ("authority.public", "--authority-public-key"),
        ("phase_preparation.private", "--observer-private-key"),
        ("one_touch.private", "--observer-private-key"),
    ):
        unit = module._unit(
            "non-network",
            ["worker", argument, f"%d/{credential}"],
            {"HOME": "/safe"},
            {"high": 1, "max": 2, "swap": 3},
            None,
            read_paths=["/control"],
            write_paths=["/evidence"],
            inaccessible_paths=["/keys"],
            load_credential=f"{credential}:/keys/{credential}",
        )
        assert unit["Service"]["RestrictAddressFamilies"] == ["AF_UNIX"]
        assert unit["Service"]["IPAddressDeny"] == "any"


def test_embedded_admission_sources_are_independently_compilable():
    module = _module()
    probe = _probe_module()
    compile(module._ADMISSION_PRESTART_CODE, "<admission-prestart>", "exec")
    compile(probe._ADMISSION_SIGN_CODE, "<admission-signer>", "exec")


def test_execution_alias_identity_and_writable_mapping_are_exact(tmp_path: Path):
    module = _module()
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    alias = tmp_path / "execution-alias"
    alias.symlink_to(recovery, target_is_directory=True)
    module.RECOVERY_ROOT = recovery
    module.EXECUTION_ALIAS_ROOT = alias

    identity = module._execution_alias()
    assert identity["path"] == str(alias)
    assert identity["target"] == str(recovery)
    unit = module._unit(
        "alias mapping",
        ["/usr/bin/python3", "--private-key", "%d/authority.private"],
        {"HOME": "/tmp"},
        {"high": 1, "max": 2, "swap": 0},
        None,
        read_paths=["/usr/bin/python3"],
        write_paths=[str(recovery / "evidence")],
        inaccessible_paths=[str(tmp_path / "keys")],
        load_credential=f"authority.private:{tmp_path / 'keys' / 'authority.private'}",
    )
    expected = f"{alias / 'evidence'}:{recovery / 'evidence'}"
    assert unit["Service"]["BindPaths"] == [expected]
    module._validate_unit(unit)

    unit["Service"]["BindPaths"] = [f"{alias / 'other'}:{recovery / 'evidence'}"]
    with pytest.raises(ValueError, match="mapping mismatch"):
        module._validate_unit(unit)

    alias.unlink()
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    alias.symlink_to(wrong, target_is_directory=True)
    with pytest.raises(ValueError, match="target mismatch"):
        module._execution_alias()


def test_admission_signer_returns_exact_signature_over_abstract_socket(tmp_path: Path):
    probe = _probe_module()
    private_path = tmp_path / "authority.private"
    payload_path = tmp_path / "payload.json"
    private_path.write_bytes(
        bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    )
    private_path.chmod(0o400)
    payload = b'{"schema":"signer-selftest.v1"}\n'
    payload_path.write_bytes(payload)
    payload_path.chmod(0o600)
    public = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    token = f"luminaquant-pytest-{os.getpid()}-{os.urandom(8).hex()}"
    listener = probe.socket.socket(probe.socket.AF_UNIX, probe.socket.SOCK_STREAM)
    listener.settimeout(30)
    listener.bind("\0" + token)
    listener.listen(1)
    process = probe.subprocess.Popen(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            probe._ADMISSION_SIGN_CODE,
            "--private-key",
            str(private_path),
            "--payload",
            str(payload_path),
            "--socket",
            token,
        ],
        stdout=probe.subprocess.PIPE,
        stderr=probe.subprocess.PIPE,
    )
    try:
        connection, _ = listener.accept()
        with connection:
            chunks = []
            total = 0
            while total <= 64:
                chunk = connection.recv(65 - total)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
        signature = b"".join(chunks)
        stdout, stderr = process.communicate(timeout=30)
    finally:
        listener.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
    assert process.returncode == 0, stderr.decode()
    assert stdout == b""
    assert len(signature) == 64
    probe.Ed25519PublicKey.from_public_bytes(public).verify(signature, payload)
    assert "--output" not in probe._ADMISSION_SIGN_SOURCE


def test_scoped_admission_and_output_capabilities_are_explicit():
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert module._SCOPES == ("acquisition", "phase_preparation", "one_touch")
    assert 'f"launch-admission-{scope}.json"' in source
    assert 'phase_parent = stage_results / "phase_preparation"' in source
    assert 'prelock_parent = stage_results / "prelock"' in source
    assert 'historical_parent = stage_results / "historical"' in source
    assert "else [str(prelock_parent), str(historical_parent)]" in source


def test_credential_binding_requires_exact_owned_source_identity(tmp_path: Path):
    module = _module()
    module.RECOVERY_ROOT = tmp_path
    module.EXECUTION_ALIAS_ROOT = Path("/execution-alias")
    key_root = tmp_path / "keys"
    key_root.mkdir(mode=0o700)
    private = key_root / "authority.private"
    public = key_root / "authority.public"
    private.write_bytes(b"p" * 32)
    public.write_bytes(b"q" * 32)
    private.chmod(0o400)
    public.chmod(0o400)
    key_files = [
        {
            "name": "authority",
            "private": module._file(private),
            "public": module._file(public),
        }
    ]
    unit = module._unit(
        "probe",
        ["python", "worker", "--private-key", "%d/authority.private"],
        {"HOME": "/safe"},
        {"high": 1, "max": 2, "swap": 3},
        None,
        read_paths=[str(tmp_path / "control")],
        write_paths=[str(tmp_path / "evidence")],
        inaccessible_paths=[str(key_root)],
        load_credential=f"authority.private:{private}",
    )
    binding = module._credential_binding(unit, key_files)
    assert binding == {
        "name": "authority.private",
        "source": key_files[0]["private"],
        "target": "%d/authority.private",
    }

    private.chmod(0o600)
    with pytest.raises(ValueError, match="identity drift"):
        module._credential_binding(unit, key_files)
    private.chmod(0o400)
    sibling = key_root / "linked.private"
    os.link(private, sibling)
    with pytest.raises(ValueError, match=r"unsafe file|identity drift"):
        module._credential_binding(unit, key_files)


def test_import_is_stdlib_only_and_policy_constants_are_local():
    module = _module()
    assert not hasattr(module, "policy")
    assert module._SCOPES == ("acquisition", "phase_preparation", "one_touch")
    assert module._FILE_ROLES[0] == "policy_json"
    assert module.CURRENT_APPROVAL_LEAF == "current-state-approval-v8.json"


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
    execution_alias = tmp_path / "execution-alias"
    execution_alias.symlink_to(recovery, target_is_directory=True)
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
    current_python = current / ".venv" / "bin" / "python"
    accepted_python = accepted / ".venv" / "bin" / "python"
    base_python = inputs / "base-runtime" / "bin" / "python"
    for interpreter in (current_python, accepted_python, base_python):
        interpreter.parent.mkdir(parents=True)
        interpreter.write_bytes(b"synthetic interpreter")
    role_root = inputs / "roles"
    role_root.mkdir()
    role_pins = {}
    for role in module.ROLE_PINS:
        name = f"{role}.py"
        (role_root / name).write_text(f"# {role}\n")
        role_pins[role] = (str(role_root), name, None)
    current_approval = recovery / module.CURRENT_APPROVAL_LEAF
    monkeypatch.setattr(module, "RECOVERY_ROOT", recovery)
    monkeypatch.setattr(module, "EXECUTION_ALIAS_ROOT", execution_alias)
    monkeypatch.setattr(module, "CURRENT", str(current))
    monkeypatch.setattr(module, "ACCEPTED", str(accepted))
    monkeypatch.setattr(module, "ALIGNMENT", str(alignment))
    monkeypatch.setattr(module, "ALIGNMENT_SHA256", module._sha(b"alignment"))
    monkeypatch.setattr(module, "CURRENT_APPROVAL", str(current_approval))
    monkeypatch.setattr(module, "ROLE_PINS", role_pins)
    current_approval.write_bytes(b'{"approval":"fixture"}\n')
    monkeypatch.setattr(
        module,
        "_load_current_approval",
        lambda path, **kwargs: {
            "head": module.ACCEPTED_COMMIT,
            "repository_root": str(current),
            "accepted_alpha_commit": module.ACCEPTED_COMMIT,
            "baseline_ancestor": module.BASELINE,
            "verdict": "PASS_REVIEWED_OVERLAY",
        },
    )

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
        "base_python": executable_pin("base_python", base_python),
        "telemetry_script": executable_pin("telemetry_script", telemetry_script),
    }

    def create_keys(root: Path) -> None:
        for name in ("authority", "publication", *module._SCOPES):
            for kind in ("private", "public"):
                path = root / f"{name}.{kind}"
                path.write_bytes(f"{name}-{kind}".encode().ljust(32, b"_"))
                path.chmod(0o400)

    monkeypatch.setattr(module, "_load_key_creator", lambda path, identity, policy: create_keys)
    policy = SimpleNamespace(source_sha256="policy-source")
    policy.scope_contract = lambda scope: (
        {
            "acquisition": ("checkpoint_pin", "alignment_receipt"),
            "phase_preparation": (
                "checkpoint_pin",
                "alignment_receipt",
                "source_eligible_receipt",
                "source_manifest",
                "source_journal",
                "canonical_finalize_receipt",
            ),
            "one_touch": (
                "checkpoint_pin",
                "alignment_receipt",
                "phase_handoff_receipt",
                "preparation_manifest",
            ),
        }[scope],
        (),
    )

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
            current_approval=str(current_approval),
            current_python=str(current_python),
            accepted_python=str(accepted_python),
            run_id=run_id,
            request_id=module.ACQUISITION_REQUEST_ID,
            phase_request_id=module.PHASE_PREPARATION_REQUEST_ID,
            one_touch_request_id=module.ONE_TOUCH_REQUEST_ID,
        )

    args = make_args(module.RUN_ID)
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
    prerequisite_fields = {
        "kind",
        "path",
        "sha256",
        "byte_count",
        "st_dev",
        "st_ino",
        "mode",
        "nlink",
    }
    assert all(
        set(item) == prerequisite_fields
        for item in documents["acquisition-request.json"]["prerequisites"]
    )
    assert documents["manifest.json"]["launch_performed"] is False
    assert documents["manifest.json"]["execution_alias"] == module._execution_alias()
    assert not (Path(args.output_parent) / "source").exists()
    assert not (Path(args.output_parent) / "report").exists()
    plan = module._load_canonical(control / "launch-plan.json")
    assert plan["scope_topology"]["execution_alias"] == module._execution_alias()
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
        service = unit["Service"]
        module._validate_unit({key: value for key, value in unit.items() if key != "name"})
        destinations = [binding.split(":", 1)[1] for binding in service["BindPaths"]]
        assert not any(
            left == right or left.startswith(right + "/") or right.startswith(left + "/")
            for left in service["BindReadOnlyPaths"]
            for right in destinations
        )
        assert service["ProtectHome"] == "tmpfs"
        assert service["WorkingDirectory"] == "/"
    assert (authority["MemoryHigh"], authority["MemoryMax"], authority["MemorySwapMax"]) == (
        402653184,
        536870912,
        268435456,
    )
    monitor = plan["telemetry_contract"]["monitor"]
    assert monitor[monitor.index("--authority-memory-max") + 1] == str(authority["MemoryMax"])
    assert monitor[monitor.index("--authority-swap-max") + 1] == str(authority["MemorySwapMax"])
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
    for wrapped, capture in (
        (authority["ExecStopPost"], plan["telemetry_contract"]["authority_capture"]),
        (observer["ExecStopPost"], plan["telemetry_contract"]["observer_capture"]),
    ):
        assert wrapped[:5] == ["/usr/bin/python3", "-I", "-S", "-c", module._MOUNT_IDENTITY_CODE]
        assert wrapped[6:] == capture
    key_root = Path(args.key_root)
    assert all(
        str(key_root) not in service[path_key]
        for service in (authority, observer, telemetry)
        for path_key in ("BindReadOnlyPaths", "BindPaths")
    )
    assert authority["InaccessiblePaths"] == [f"-{key_root}"]
    assert observer["InaccessiblePaths"] == [f"-{key_root}"]
    assert telemetry["InaccessiblePaths"] == [f"-{key_root}"]
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
    rendered_units = plan["rendered_systemd_units"]
    assert documents["manifest.json"]["rendered_systemd_units"] == rendered_units
    for role, item in rendered_units.items():
        unit = {key: value for key, value in plan["systemd_units"][role].items() if key != "name"}
        unit_path = Path(item["file"]["path"])
        assert unit_path.read_bytes() == module._render_systemd_unit(unit)
        assert module._file(unit_path) == item["file"]
        assert item["credential"]["target"].startswith("%d/")
        assert (
            item["credential"]["source"]["sha256"]
            == module._file(Path(item["credential"]["source"]["path"]))["sha256"]
        )
        assert b"%%d" not in unit_path.read_bytes()
        rendered = unit_path.read_bytes()
        assert module._MOUNT_IDENTITY_CODE.encode() in rendered
        assert b"launch-admission-acquisition.json" in rendered
        assert b"\\n" not in module._MOUNT_IDENTITY_CODE.encode()
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
    with pytest.raises(ValueError, match="fresh recovery identity mismatch"):
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
    with pytest.raises(ValueError, match="fresh recovery identity mismatch"):
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

    failed_args = make_args("c" * 64)
    with pytest.raises(ValueError, match="fresh recovery identity mismatch"):
        module.build(failed_args)
    assert not Path(failed_args.control_root).exists()


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


def test_recovery_epoch_identifiers_and_approval_leaf_are_exact():
    module = _module()
    assert module.CURRENT_APPROVAL_LEAF == "current-state-approval-v8.json"
    assert module.RUN_ID == "41ed4e09bd7b4af1793d3138e5d55d22d217d5fd4bd9477493d885e42fae1602"
    assert {
        "acquisition": module.ACQUISITION_REQUEST_ID,
        "phase_preparation": module.PHASE_PREPARATION_REQUEST_ID,
        "one_touch": module.ONE_TOUCH_REQUEST_ID,
    } == {
        "acquisition": "3377aca77e4edead9454051883d015c15389ffb0da06d63ec9e76ee2573252ec",
        "phase_preparation": "a89b7872357feaee131c9bdbf4d78312ad86e7d4da90c7d2441d2d9cd2b43bac",
        "one_touch": "30f6eafb4f0023c602c54bab8719ce105b2c28d07c7511cb5d9db42936f49d5c",
    }


def test_approval_only_does_not_create_declared_control_roots(tmp_path: Path, monkeypatch):
    module = _module()
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    execution_alias = tmp_path / "execution-alias"
    execution_alias.symlink_to(recovery, target_is_directory=True)
    current = tmp_path / "current"
    current.mkdir()
    monkeypatch.setattr(module, "RECOVERY_ROOT", recovery)
    monkeypatch.setattr(module, "EXECUTION_ALIAS_ROOT", execution_alias)
    monkeypatch.setattr(module, "CURRENT", str(current))
    monkeypatch.setattr(module, "_git", lambda *_args: b"0" * 40 + b"\n")
    monkeypatch.setattr(module, "_only_owner_session_runtime_changes", lambda _root: True)
    monkeypatch.setattr(module, "_inventory", lambda _root: b"[]\n")
    monkeypatch.setattr(
        module,
        "_preflight_executables",
        lambda _paths: {
            "current_python": {"path": "/runtime/current/bin/python"},
            "accepted_python": {"path": "/runtime/accepted/bin/python"},
            "base_python": {"path": "/runtime/base/bin/python"},
            "telemetry_script": {"path": "/runtime/telemetry"},
        },
    )
    monkeypatch.setattr(module, "_runtime_inventory", lambda _interpreter: b"[]\n")
    monkeypatch.setattr(module, "_ignored_source_inventory", lambda _root: b"[]\n", raising=False)
    args = argparse.Namespace(
        control_root=str(recovery / f"g056v8-controls-{module.RUN_ID}"),
        key_root=str(recovery / f"g056v8-keys-{module.RUN_ID}"),
        evidence_root=str(recovery / f"g056v8-acquisition-evidence-{module.RUN_ID}"),
        telemetry_root=str(recovery / f"g056v8-telemetry-{module.RUN_ID}"),
        output_parent=str(recovery / f"g056v8-acquisition-output-{module.RUN_ID}"),
        current_approval=str(recovery / module.CURRENT_APPROVAL_LEAF),
        run_id=module.RUN_ID,
        request_id=module.ACQUISITION_REQUEST_ID,
        phase_request_id=module.PHASE_PREPARATION_REQUEST_ID,
        one_touch_request_id=module.ONE_TOUCH_REQUEST_ID,
    )
    module.create_approval(args)
    assert (recovery / module.CURRENT_APPROVAL_LEAF).is_file()
    assert not any(
        Path(getattr(args, name)).exists()
        for name in ("control_root", "key_root", "evidence_root", "telemetry_root", "output_parent")
    )


def test_stage_modes_are_explicit_and_require_predecessor_arguments():
    parser_source = SCRIPT.read_text(encoding="utf-8")
    assert "--build-phase-preparation" in parser_source
    assert "--build-one-touch" in parser_source
    assert "--canonical-finalize-receipt" in parser_source
    assert "deferred_until_authenticated_acquisition_receipts" in parser_source
    assert "deferred_until_authenticated_phase_receipts" in parser_source


def test_staged_scope_contract_declares_authority_observer_and_telemetry_units():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "alpha-max-v8-{scope}-telemetry-{args.run_id}.service" in source
    assert '"telemetry": {' in source
    assert "authority_capture" in source
    assert "observer_capture" in source
    assert "canonical-finalize-receipt" in source
    assert '"unit_definitions": {' in source
    assert "alpha_max_v8_terminal_stage_complete.v1" in source
    assert "policy.validate_w10_canonical_finalize_bundle(" in source
    assert "authority_public_key_b64=envelope.authority_key.public_key_b64" in source
    assert "run_id=args.run_id" in source
    assert "acquisition_request_id=ACQUISITION_REQUEST_ID" in source
    assert 'approval_sha256=_file(approval_path)["sha256"]' in source
    assert "canonical_finalize_identity = _file(canonical_finalize_receipt)" in source
    assert "canonical finalize receipt identity drift" in source


def test_persisted_probe_parses_real_cgroup_v2_whitespace_counters():
    probe = _probe_module()
    assert probe._cgroup_properties("high 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n") == {
        "high": "0",
        "max": "0",
        "oom": "0",
        "oom_kill": "0",
        "oom_group_kill": "0",
    }
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert '"memory.high"' in source
    assert '"memory.oom.group"' in source
    assert '"cgroup.events"' in source
    assert '"oom_group_kill"' in source


def test_persisted_probe_stage_selection_allows_acquisition_only_and_rejects_missing_stage(
    tmp_path: Path,
):
    probe = _probe_module()
    control = tmp_path / "control"
    control.mkdir()
    assert probe._selected_stages(control, "acquisition") == ("acquisition",)
    assert probe._selected_stages(control, "all-ready") == ("acquisition",)
    with pytest.raises(RuntimeError, match="missing persisted phase_preparation stage"):
        probe._selected_stages(control, "phase_preparation")
    (control / "phase_preparation-manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="incomplete persisted phase_preparation stage"):
        probe._selected_stages(control, "all-ready")


def test_persisted_probe_scope_cleanup_and_live_property_contract_is_explicit():
    source = PROBE_SCRIPT.read_text(encoding="utf-8")
    assert "--scope" in source
    assert 'default="acquisition"' in source
    assert "all-ready cannot publish" in source
    assert "production_links" in source
    assert "_PRODUCTION_PROPERTIES" in source
    assert "production_properties" in source
    assert "if link.is_symlink()" in source
    assert "probe_omits_production_prestart" in source
