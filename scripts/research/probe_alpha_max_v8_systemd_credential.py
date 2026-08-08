#!/usr/bin/env python3
"""Exercise the production static-unit credential path without network access."""

from __future__ import annotations

import base64
import copy
import argparse
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import socket
import time
from typing import Any
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_BUILDER = Path(__file__).with_name("create_alpha_max_v8_acquisition_controls.py")
_PROBE_CODE = (
    "import hashlib,json,os,sys;"
    "credential=sys.argv[sys.argv.index('--private-key')+1];"
    "expected=sys.argv[sys.argv.index('--expected-sha256')+1];"
    "marker=sys.argv[sys.argv.index('--marker')+1];"
    "data=open(credential,'rb').read();"
    "actual=hashlib.sha256(data).hexdigest();"
    "assert actual==expected;"
    "payload=json.dumps({'credential_path':credential,'credential_sha256':actual,"
    "'entered_execstart':True},sort_keys=True,separators=(',',':')).encode()+b'\\n';"
    "fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);"
    "os.write(fd,payload);os.fsync(fd);os.close(fd)"
)
_PERSISTED_PROBE_CODE = (
    "import hashlib,json,os,stat,sys,time;"
    "credential_name=sys.argv[sys.argv.index('--credential-name')+1];"
    "credential_flag={'authority.private':'--private-key','authority.public':'--authority-public-key',"
    "'acquisition.private':'--observer-private-key','phase_preparation.private':'--observer-private-key',"
    "'one_touch.private':'--observer-private-key'}[credential_name];"
    "credential=sys.argv[sys.argv.index(credential_flag)+1];"
    "expected=sys.argv[sys.argv.index('--expected-sha256')+1];"
    "marker=sys.argv[sys.argv.index('--marker')+1];"
    "release=sys.argv[sys.argv.index('--release')+1];"
    "scope=sys.argv[sys.argv.index('--scope')+1];"
    "role=sys.argv[sys.argv.index('--role')+1];"
    "request=sys.argv[sys.argv.index('--request')+1];"
    "key_id=sys.argv[sys.argv.index('--key-id')+1];"
    "approval=sys.argv[sys.argv.index('--approval')+1];"
    "data=open(credential,'rb').read();info=os.stat(credential);"
    "assert hashlib.sha256(data).hexdigest()==expected;"
    "assert stat.S_ISREG(info.st_mode) and info.st_size==len(data);"
    "assert os.path.isfile(request) and os.path.isfile(approval) and len(key_id)==64;"
    "payload=json.dumps({'credential_path':credential,'credential_sha256':expected,"
    "'credential_mode':stat.S_IMODE(info.st_mode),'key_id':key_id,'scope':scope,'role':role,"
    "'request':request,'approval':approval},sort_keys=True,separators=(',',':')).encode()+b'\\n';"
    "fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);"
    "os.write(fd,payload);os.fsync(fd);os.close(fd);"
    "deadline=time.monotonic()+30;"
    'exec("while not os.path.exists(release):\\n assert time.monotonic()<deadline\\n time.sleep(.01)")'
)
_ADMISSION_SIGN_SOURCE = r"""
import fcntl
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys


def read_regular(path):
    path = Path(path)
    before = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise ValueError(f"unsafe signer input: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink", "st_size")
    if any(getattr(before, field) != getattr(opened, field) for field in fields) or any(
        getattr(opened, field) != getattr(after, field) for field in fields
    ):
        raise ValueError(f"unstable signer input: {path}")
    data = b"".join(chunks)
    if len(data) != opened.st_size:
        raise ValueError(f"short signer input: {path}")
    return data


key_path = sys.argv[sys.argv.index("--private-key") + 1]
payload_path = sys.argv[sys.argv.index("--payload") + 1]
socket_token = sys.argv[sys.argv.index("--socket") + 1]
key = read_regular(key_path)
payload = read_regular(payload_path)
if len(key) != 32:
    raise ValueError("invalid raw Ed25519 private key")
pkcs8 = bytes.fromhex("302e020100300506032b657004220420") + key


def sealed_memfd(name, data):
    descriptor = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short memfd write")
            view = view[written:]
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(
            descriptor,
            fcntl.F_ADD_SEALS,
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE,
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


key_descriptor = sealed_memfd("alpha-max-ed25519", pkcs8)
payload_descriptor = sealed_memfd("alpha-max-admission-payload", payload)
try:
    result = subprocess.run(
        (
            "/usr/bin/openssl",
            "pkeyutl",
            "-sign",
            "-inkey",
            f"/proc/self/fd/{key_descriptor}",
            "-keyform",
            "DER",
            "-rawin",
            "-in",
            f"/proc/self/fd/{payload_descriptor}",
        ),
        check=True,
        capture_output=True,
        pass_fds=(key_descriptor, payload_descriptor),
    )
finally:
    os.close(payload_descriptor)
    os.close(key_descriptor)
signature = result.stdout
if len(signature) != 64:
    raise ValueError("invalid Ed25519 signature length")
channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    channel.connect("\0" + socket_token)
    channel.sendall(signature)
    channel.shutdown(socket.SHUT_WR)
finally:
    channel.close()
"""
_ADMISSION_SIGN_CODE = (
    "import base64;exec(compile(base64.b64decode("
    + repr(base64.b64encode(_ADMISSION_SIGN_SOURCE.encode()).decode())
    + "),'<alpha-max-admission-signer>','exec'))"
)


def _controls():
    spec = importlib.util.spec_from_file_location("alpha_max_v8_probe_controls", _BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the production control builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*argv: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=check, capture_output=True, text=True, timeout=30)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _write_new(path: Path, payload: bytes, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _properties(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            values[name] = value
    return values


def _ip_address_deny_all(value: str | None) -> bool:
    return value == "any" or set((value or "").split()) == {"0.0.0.0/0", "::/0"}


def _cgroup_properties(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 2:
            values[fields[0]] = fields[1]
    return values


_PRODUCTION_PROPERTIES = (
    "FragmentPath",
    "ExecStart",
    "ExecStartPre",
    "ExecStopPost",
    "Environment",
    "WorkingDirectory",
    "UMask",
    "NoNewPrivileges",
    "PrivateTmp",
    "PrivateDevices",
    "ProtectSystem",
    "ProtectHome",
    "BindReadOnlyPaths",
    "BindPaths",
    "InaccessiblePaths",
    "LoadCredential",
    "ProtectKernelTunables",
    "ProtectKernelModules",
    "ProtectControlGroups",
    "RestrictAddressFamilies",
    "IPAddressDeny",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
    "OOMPolicy",
    "TimeoutStartUSec",
    "TimeoutStopUSec",
)


def _production_properties(unit_name: str) -> dict[str, str]:
    command = ["systemctl", "--user", "show", unit_name]
    for name in _PRODUCTION_PROPERTIES:
        command.extend(("-p", name))
    return _properties(_run(*command).stdout)


def _selected_stages(control_root: Path, scope: str) -> tuple[str, ...]:
    if scope not in {"acquisition", "phase_preparation", "one_touch", "all-ready"}:
        raise ValueError("invalid persisted probe scope")
    available = ["acquisition"]
    for candidate in ("phase_preparation", "one_touch"):
        stage = control_root / f"{candidate}-manifest.json"
        complete = control_root / f"{candidate}.COMPLETE.json"
        if stage.exists() != complete.exists():
            raise RuntimeError(f"incomplete persisted {candidate} stage")
        if stage.exists():
            available.append(candidate)
    if "one_touch" in available and "phase_preparation" not in available:
        raise RuntimeError("one_touch stage exists without phase_preparation stage")
    if scope == "all-ready":
        return tuple(available)
    required = {
        "acquisition": ("acquisition",),
        "phase_preparation": ("acquisition", "phase_preparation"),
        "one_touch": ("acquisition", "phase_preparation", "one_touch"),
    }[scope]
    if any(candidate not in available for candidate in required):
        raise RuntimeError(f"missing persisted {scope} stage")
    return (scope,)


def verify_persisted_topology(
    control_root: Path, key_root: Path, approval: Path, scope: str = "all-ready"
) -> dict[str, Any]:
    """Fail closed on requested persisted stage artifacts before probing."""
    selected = _selected_stages(control_root, scope)
    required = (
        selected
        if scope == "all-ready"
        else {
            "acquisition": ("acquisition",),
            "phase_preparation": ("acquisition", "phase_preparation"),
            "one_touch": ("acquisition", "phase_preparation", "one_touch"),
        }[scope]
    )
    controls = _controls()
    expected_paths = {
        "control_root": control_root,
        "key_root": key_root,
        "evidence_root": control_root.parent / f"g056v8-acquisition-evidence-{controls.RUN_ID}",
        "telemetry_root": control_root.parent / f"g056v8-telemetry-{controls.RUN_ID}",
        "output_parent": control_root.parent / f"g056v8-acquisition-output-{controls.RUN_ID}",
        "stage_results_parent": control_root.parent / f"g056v8-stage-results-{controls.RUN_ID}",
    }
    controls._load_current_approval(
        approval,
        run_id=controls.RUN_ID,
        request_ids={
            "acquisition": controls.ACQUISITION_REQUEST_ID,
            "phase_preparation": controls.PHASE_PREPARATION_REQUEST_ID,
            "one_touch": controls.ONE_TOUCH_REQUEST_ID,
        },
        absent_paths=expected_paths,
        require_absent=False,
    )
    manifest = controls._load_canonical(control_root / "manifest.json")
    complete = controls._load_canonical(control_root / "COMPLETE.json")
    if (
        complete.get("manifest_sha256") != controls._file(control_root / "manifest.json")["sha256"]
        or manifest.get("approval") != controls._file(approval)
        or manifest.get("roots")
        != {
            name: controls._directory(path, private=True)
            for name, path in expected_paths.items()
            if name != "stage_results_parent"
        }
        or manifest.get("admission_root")
        != controls._directory(control_root / "admissions", private=True)
        or manifest.get("execution_alias") != controls._execution_alias()
    ):
        raise RuntimeError("acquisition manifest or approval binding mismatch")
    _authority, _observers, _summary, key_files = controls._key_bindings(key_root)
    controls._revalidate_key_files(key_files)
    stages: dict[str, dict[str, Any]] = {}
    for candidate in required:
        if candidate == "acquisition":
            continue
        stage = control_root / f"{candidate}-manifest.json"
        marker = control_root / f"{candidate}.COMPLETE.json"
        value = controls._load_canonical(stage)
        if (
            value.get("scope") != candidate
            or value.get("approval") != controls._file(approval)
            or controls._load_canonical(marker).get("manifest_sha256")
            != controls._file(stage)["sha256"]
        ):
            raise RuntimeError(f"{candidate} stage binding mismatch")
        stages[candidate] = value
    verified: list[dict[str, Any]] = []
    acquisition_plan_path = Path(manifest["launch_plan"]["path"])
    acquisition_plan = controls._load_canonical(acquisition_plan_path)
    if (
        controls._file(acquisition_plan_path) != manifest["launch_plan"]
        or acquisition_plan.get("scope_topology", {}).get("execution_alias")
        != controls._execution_alias()
    ):
        raise RuntimeError("acquisition launch plan or execution alias binding mismatch")
    if "acquisition" in selected:
        for role, rendered in acquisition_plan["rendered_systemd_units"].items():
            item = acquisition_plan["systemd_units"][role]
            unit = {key: value for key, value in item.items() if key != "name"}
            if (
                controls._file(Path(rendered["file"]["path"])) != rendered["file"]
                or controls._render_systemd_unit(unit)
                != Path(rendered["file"]["path"]).read_bytes()
                or controls._credential_binding(unit, key_files) != rendered["credential"]
            ):
                raise RuntimeError("acquisition rendered unit or credential mismatch")
            verified.append(
                {
                    "scope": "acquisition",
                    "role": role,
                    "unit": rendered["file"],
                    "credential": rendered["credential"],
                    "definition": unit,
                    "production_unit": item["name"],
                }
            )
    for candidate in selected:
        if candidate == "acquisition":
            continue
        stage = stages[candidate]
        for role, item in stage["units"].items():
            definition = stage["unit_definitions"].get(role)
            unit_path = Path(item["file"]["path"])
            rendered = unit_path.read_bytes()
            if (
                not isinstance(definition, dict)
                or controls._file(unit_path) != item["file"]
                or controls._render_systemd_unit(definition) != rendered
            ):
                raise RuntimeError("staged rendered unit mismatch")
            credential = item["credential"]
            source = credential["source"]
            if (
                controls._file(Path(source["path"])) != source
                or controls._credential_binding(definition, key_files) != credential
                or credential["target"] not in rendered.decode()
            ):
                raise RuntimeError("staged credential binding mismatch")
            verified.append(
                {
                    "scope": candidate,
                    "role": role,
                    "unit": item["file"],
                    "credential": credential,
                    "definition": definition,
                    "production_unit": item["name"],
                }
            )
    artifacts = [
        controls._file(control_root / "manifest.json"),
        controls._file(control_root / "COMPLETE.json"),
        controls._file(acquisition_plan_path),
        controls._file(Path(manifest["request"]["path"])),
        *[item["unit"] for item in verified],
    ]
    for candidate in selected:
        if candidate != "acquisition":
            artifacts.extend(
                (
                    controls._file(control_root / f"{candidate}-request.json"),
                    controls._file(control_root / f"{candidate}-manifest.json"),
                    controls._file(control_root / f"{candidate}.COMPLETE.json"),
                )
            )
    return {
        "schema": "alpha_max_v8_persisted_unit_probe.v1",
        "verdict": "PASS",
        "requested_scope": scope,
        "stages": list(selected),
        "units": verified,
        "control_artifacts": artifacts,
    }


def _persisted_probe_execstart(
    controls: Any,
    credential_name: str,
    credential_target: str,
    expected_sha256: str,
    marker: Path,
    release: Path,
    scope: str,
    role: str,
    request: str,
    key_id: str,
    approval: Path,
) -> list[str]:
    return [
        controls.EXECUTABLE_PINS["current_python"]["path"],
        "-I",
        "-S",
        "-c",
        _PERSISTED_PROBE_CODE,
        controls._credential_argument(credential_name),
        credential_target,
        "--credential-name",
        credential_name,
        "--expected-sha256",
        expected_sha256,
        "--marker",
        str(marker),
        "--release",
        str(release),
        "--scope",
        scope,
        "--role",
        role,
        "--request",
        request,
        "--key-id",
        key_id,
        "--approval",
        str(approval),
    ]


def _execute_persisted_topology(
    control_root: Path,
    key_root: Path,
    approval: Path,
    evidence_root: Path,
    scope: str = "all-ready",
) -> dict[str, Any]:
    """Run a safe command substitution under requested persisted unit directives."""
    if scope == "all-ready":
        raise ValueError("all-ready cannot publish a receipt unused by a single production scope")
    verified = verify_persisted_topology(control_root, key_root, approval, scope)
    requested_scope = scope
    selected = tuple(verified["stages"])
    controls = _controls()
    authority, observers, _summary, _key_files = controls._key_bindings(key_root)
    key_ids = {
        "authority": authority["key_id"],
        **{item["scope"]: item["key_id"] for item in observers},
    }
    expected_evidence = control_root.parent / f"g056v8-acquisition-evidence-{controls.RUN_ID}"
    manifest = controls._load_canonical(control_root / "manifest.json")
    admission_root = control_root / "admissions"
    probe_admission_root = admission_root / "probe"
    admission_paths = [
        admission_root / f"launch-admission-{requested_scope}{suffix}"
        for suffix in (".payload.json", ".signature", ".json")
    ]
    audit_path = evidence_root / f"launch-admission-{requested_scope}.audit.json"
    if (
        evidence_root != expected_evidence
        or not evidence_root.is_absolute()
        or evidence_root.is_symlink()
        or not evidence_root.is_dir()
        or manifest.get("admission_root") != controls._directory(admission_root, private=True)
        or manifest.get("execution_alias") != controls._execution_alias()
        or any(path.exists() or path.is_symlink() for path in [*admission_paths, audit_path])
        or probe_admission_root.exists()
        or probe_admission_root.is_symlink()
    ):
        raise ValueError(
            "persisted probe roots must be the declared fresh production evidence and admission paths"
        )
    plan = controls._load_canonical(Path(manifest["launch_plan"]["path"]))
    definitions: list[tuple[str, str, dict[str, Any], dict[str, Any], str, str]] = []
    if "acquisition" in selected:
        for role, rendered in plan["rendered_systemd_units"].items():
            item = plan["systemd_units"][role]
            definitions.append(
                (
                    "acquisition",
                    role,
                    {key: value for key, value in item.items() if key != "name"},
                    rendered["credential"],
                    str(control_root / "acquisition-request.json"),
                    item["name"],
                )
            )
    for candidate in selected:
        if candidate == "acquisition":
            continue
        stage = controls._load_canonical(control_root / f"{candidate}-manifest.json")
        for role, item in stage["units"].items():
            definitions.append(
                (
                    candidate,
                    role,
                    stage["unit_definitions"][role],
                    item["credential"],
                    stage["request"]["path"],
                    item["name"],
                )
            )
    unit_directory = Path.home() / ".config/systemd/user"
    unit_directory.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    production_links: list[Path] = []
    production_properties: dict[str, dict[str, str]] = {}
    try:
        for entry in verified["units"]:
            unit_file = Path(entry["unit"]["path"])
            link = unit_directory / unit_file.name
            if link.exists() or link.is_symlink():
                raise RuntimeError(f"production unit link already exists: {link}")
            if controls._file(unit_file) != entry["unit"]:
                raise RuntimeError("production unit identity drift before link")
            os.symlink(unit_file, link)
            production_links.append(link)
        _run("systemctl", "--user", "daemon-reload")
        for entry in verified["units"]:
            unit_file = Path(entry["unit"]["path"])
            _run("systemd-analyze", "--user", "verify", str(unit_file))
            properties = _production_properties(unit_file.name)
            if (
                properties.get("FragmentPath")
                not in {str(unit_file), str(unit_directory / unit_file.name)}
                or not properties.get("ExecStart")
                or not properties.get("ExecStartPre")
                or properties.get("WorkingDirectory") != "/"
                or properties.get("UMask") != "0077"
                or properties.get("MemoryHigh") != str(entry["definition"]["Service"]["MemoryHigh"])
                or properties.get("MemoryMax") != str(entry["definition"]["Service"]["MemoryMax"])
                or properties.get("MemorySwapMax")
                != str(entry["definition"]["Service"]["MemorySwapMax"])
                or properties.get("OOMPolicy") != entry["definition"]["Service"]["OOMPolicy"]
            ):
                raise RuntimeError("production unit static readback mismatch")
            production_properties[unit_file.name] = properties
    except BaseException:
        for link in production_links:
            if link.is_symlink():
                link.unlink()
        _run("systemctl", "--user", "daemon-reload", check=False)
        raise

    def admission_snapshot(
        topology: dict[str, Any], properties: dict[str, dict[str, str]]
    ) -> dict[str, Any]:
        runtime_inputs = controls._preflight_executables(
            {
                "current_python": Path(controls.EXECUTABLE_PINS["current_python"]["path"]),
                "accepted_python": Path(controls.EXECUTABLE_PINS["accepted_python"]["path"]),
                "telemetry_script": Path(controls.EXECUTABLE_PINS["telemetry_script"]["path"]),
            }
        )
        accepted_root = Path(controls.ACCEPTED)
        return {
            "approval": controls._file(approval),
            "complete": controls._file(control_root / "COMPLETE.json"),
            "manifest": controls._file(control_root / "manifest.json"),
            "requested_scope": requested_scope,
            "stages": list(selected),
            "topology": topology,
            "control_artifacts": topology["control_artifacts"],
            "source_inventory": controls._record(controls._inventory(Path(controls.CURRENT))),
            "ignored_source_inventory": controls._record(
                controls._ignored_source_inventory(Path(controls.CURRENT))
            ),
            "accepted_source_state": {
                "root": controls.ACCEPTED,
                "head": controls._git(accepted_root, "rev-parse", "HEAD").decode().strip(),
                "porcelain": controls._record(
                    controls._source_git(accepted_root, "status", "--porcelain=v1", "-z")
                ),
                "source_inventory": controls._record(controls._inventory(accepted_root)),
                "ignored_source_inventory": controls._record(
                    controls._ignored_source_inventory(accepted_root)
                ),
            },
            "runtime_inventories": {
                name: controls._record(controls._runtime_inventory(runtime_inputs[name]))
                for name in controls._RUNTIME_NAMES
            },
            "execution_alias": controls._execution_alias(),
            "production_properties": properties,
            "runtime_bindings": {
                name: {
                    "interpreter": runtime_inputs[name],
                    "root": str(Path(controls.EXECUTABLE_PINS[name]["path"]).parent.parent),
                }
                for name in controls._RUNTIME_NAMES
            },
        }

    common_admission = admission_snapshot(verified, production_properties)
    controls._create_root(probe_admission_root)
    late_bound_receipt = evidence_root / "terminal-authority.receipt.json"
    late_bound_consumers = [
        {"scope": definition_scope, "role": role}
        for definition_scope, role, original, _credential, _request, _name in definitions
        if str(late_bound_receipt)
        in original["Service"].get("BindReadOnlyPaths", [])
    ]
    probe_only_control_artifacts: list[dict[str, Any]] = []
    probe_placeholder = (
        probe_admission_root / "late-bound-terminal-authority.placeholder.json"
    )
    if late_bound_consumers:
        if late_bound_receipt.exists() or late_bound_receipt.is_symlink():
            raise RuntimeError("late-bound production receipt exists before persisted probe")
        _write_new(
            probe_placeholder,
            _canonical(
                {
                    "schema": "alpha_max_v8_probe_late_bound_placeholder.v1",
                    "production_path": str(late_bound_receipt),
                    "consumers": late_bound_consumers,
                    "requested_scope": requested_scope,
                    "run_id": controls.RUN_ID,
                }
            ),
            0o400,
        )
        probe_only_control_artifacts.append(controls._file(probe_placeholder))
    late_bound_probe = {
        "production_receipt": {
            "path": str(late_bound_receipt),
            "state": "absent",
        },
        "consumers": late_bound_consumers,
        "placeholder": (
            probe_only_control_artifacts[0] if probe_only_control_artifacts else None
        ),
    }
    probe_payload = {
        "schema": "alpha_max_v8_persisted_probe_preflight_admission.v1",
        "verdict": "PROBE_ONLY",
        **common_admission,
        "control_artifacts": [
            *common_admission["control_artifacts"],
            *probe_only_control_artifacts,
        ],
        "late_bound_probe": late_bound_probe,
        "units": [],
    }
    signed_probe_admission = _sign_launch_admission(
        controls,
        probe_admission_root,
        evidence_root,
        key_root,
        authority,
        probe_payload,
        requested_scope,
    )
    if signed_probe_admission.get("payload") != probe_payload:
        raise RuntimeError("probe launch-admission signer readback mismatch")
    probe_receipt = probe_admission_root / f"launch-admission-{requested_scope}.json"
    probe_artifacts = {
        "payload": controls._file(
            probe_admission_root / f"launch-admission-{requested_scope}.payload.json"
        ),
        "signature": controls._file(
            probe_admission_root / f"launch-admission-{requested_scope}.signature"
        ),
        "envelope": controls._file(probe_receipt),
    }
    for index, (scope, role, original, credential, request, production_name) in enumerate(
        definitions
    ):
        active = _properties(
            _run(
                "systemctl", "--user", "show", production_name, "-p", "ActiveState", check=False
            ).stdout
        )
        if active.get("ActiveState") not in {"inactive", "failed", ""}:
            raise RuntimeError(f"production unit is active: {production_name}")
        source, credential_name = credential["source"], credential["name"]
        marker = evidence_root / f"{index:02d}-{scope}-{role}.json"
        release = evidence_root / f"{index:02d}-{scope}-{role}.release"
        unit_name = f"luminaquant-persisted-{index}-{secrets.token_hex(12)}.service"
        substituted = copy.deepcopy(original)
        service = substituted["Service"]
        production_read_paths = list(service.get("BindReadOnlyPaths", []))
        probe_read_paths = list(production_read_paths)
        readonly_overlays: list[dict[str, Any]] = []
        if str(late_bound_receipt) in probe_read_paths:
            if probe_read_paths.count(str(late_bound_receipt)) != 1:
                raise RuntimeError("late-bound production receipt binding is ambiguous")
            if len(probe_only_control_artifacts) != 1:
                raise RuntimeError("late-bound probe placeholder is unavailable")
            probe_read_paths[probe_read_paths.index(str(late_bound_receipt))] = str(
                probe_placeholder
            )
            service["BindReadOnlyPaths"] = probe_read_paths
            readonly_overlays.append(
                {
                    "production_path": str(late_bound_receipt),
                    "probe_placeholder": probe_only_control_artifacts[0],
                }
            )
        production_wrapper = original["Service"]["ExecStart"]
        if production_wrapper[:5] != [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            controls._MOUNT_IDENTITY_CODE,
        ]:
            raise RuntimeError("production ExecStart lacks final namespace verifier")
        wrapper_config = json.loads(production_wrapper[5])
        stop = service.get("ExecStopPost")
        if stop is not None:
            if (
                stop[:5] != production_wrapper[:5]
                or len(stop) <= 6
                or json.loads(stop[5]) != wrapper_config
            ):
                raise RuntimeError("production ExecStopPost lacks final namespace verifier")
            stop_argv = list(stop[6:])
            try:
                expected_unit_index = stop_argv.index("--expected-unit") + 1
            except ValueError as error:
                raise RuntimeError("production ExecStopPost lacks expected unit binding") from error
            if (
                expected_unit_index >= len(stop_argv)
                or stop_argv[expected_unit_index] != production_name
            ):
                raise RuntimeError("production ExecStopPost unit binding mismatch")
            stop_argv[expected_unit_index] = unit_name
            service["ExecStopPost"] = controls._wrap_execstart(
                stop_argv,
                [binding.split(":", 1)[1] for binding in service.get("BindPaths", [])],
                receipt=str(probe_receipt),
                unit_name=wrapper_config["unit_name"],
                authority_public_b64=wrapper_config["authority_public_b64"],
            )
        probe_prestart = service.pop("ExecStartPre", None)
        if probe_prestart != original["Service"].get("ExecStartPre"):
            raise RuntimeError("probe omitted a nonexact production prestart")
        probe_argv = _persisted_probe_execstart(
            controls,
            credential_name,
            credential["target"],
            source["sha256"],
            marker,
            release,
            scope,
            role,
            request,
            key_ids[
                "authority"
                if credential_name.startswith("authority.")
                else credential_name.removesuffix(".private")
            ],
            approval,
        )
        service["ExecStart"] = controls._wrap_execstart(
            probe_argv,
            [binding.split(":", 1)[1] for binding in service.get("BindPaths", [])],
            receipt=str(probe_receipt),
            unit_name=wrapper_config["unit_name"],
            authority_public_b64=wrapper_config["authority_public_b64"],
        )
        controls._validate_unit(substituted)
        if (
            service.get("BindReadOnlyPaths") != probe_read_paths
            or any(
                service.get(name) != value
                for name, value in original["Service"].items()
                if name
                not in {
                    "ExecStart",
                    "ExecStopPost",
                    "ExecStartPre",
                    "BindReadOnlyPaths",
                }
            )
        ):
            raise RuntimeError("probe directive drift")
        probe_unit = controls._render_systemd_unit(substituted)
        network_deny_overlay = True
        probe_unit += (
            b"RestrictAddressFamilies=\nRestrictAddressFamilies=AF_UNIX\nIPAddressDeny=any\n"
        )
        transient, link = evidence_root / unit_name, unit_directory / unit_name
        _write_new(transient, probe_unit)
        properties: dict[str, str] = {}
        cgroup_evidence: dict[str, str] = {}
        try:
            os.symlink(transient, link)
            _run("systemctl", "--user", "daemon-reload")
            _run("systemd-analyze", "--user", "verify", str(transient))
            _run("systemctl", "--user", "start", unit_name)
            deadline = time.monotonic() + 10.0
            while not marker.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("safe substituted ExecStart did not publish marker")
                time.sleep(0.01)
            properties = _properties(
                _run(
                    "systemctl",
                    "--user",
                    "show",
                    unit_name,
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                    "-p",
                    "ActiveState",
                    "-p",
                    "MemoryHigh",
                    "-p",
                    "MemoryMax",
                    "-p",
                    "MemorySwapMax",
                    "-p",
                    "OOMPolicy",
                    "-p",
                    "LoadCredential",
                    "-p",
                    "ExecStart",
                    "-p",
                    "RestrictAddressFamilies",
                    "-p",
                    "IPAddressDeny",
                    "-p",
                    "ControlGroup",
                    check=False,
                ).stdout
            )
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
            if (
                properties.get("ActiveState") != "active"
                or marker_value.get("credential_sha256") != source["sha256"]
                or marker_value.get("scope") != scope
                or marker_value.get("role") != role
                or marker_value.get("credential_mode") != 0o400
                or marker_value.get("request") != request
                or marker_value.get("approval") != str(approval)
                or marker_value.get("key_id")
                != key_ids[
                    "authority"
                    if credential_name.startswith("authority.")
                    else credential_name.removesuffix(".private")
                ]
                or not marker_value.get("credential_path", "").endswith("/" + credential_name)
                or marker_value.get("credential_path") == source["path"]
                or properties.get("RestrictAddressFamilies") != "AF_UNIX"
                or not _ip_address_deny_all(properties.get("IPAddressDeny"))
            ):
                raise RuntimeError("safe substituted unit readback mismatch")
            control_group = properties.get("ControlGroup", "")
            cgroup = Path("/sys/fs/cgroup") / control_group.lstrip("/")
            required_cgroup_files = {
                "cgroup.events": cgroup / "cgroup.events",
                "memory.events": cgroup / "memory.events",
                "memory.high": cgroup / "memory.high",
                "memory.max": cgroup / "memory.max",
                "memory.swap.max": cgroup / "memory.swap.max",
                "memory.oom.group": cgroup / "memory.oom.group",
            }
            if not control_group or any(
                not path.is_file() for path in required_cgroup_files.values()
            ):
                raise RuntimeError("safe substituted unit has no live cgroup evidence")
            cgroup_evidence = {
                name: path.read_text(encoding="utf-8")
                for name, path in required_cgroup_files.items()
            }
            memory_events = _cgroup_properties(cgroup_evidence["memory.events"])
            cgroup_events = _cgroup_properties(cgroup_evidence["cgroup.events"])
            required_events = ("high", "max", "oom", "oom_kill", "oom_group_kill")
            if (
                cgroup_evidence["memory.high"].strip() != properties.get("MemoryHigh")
                or cgroup_evidence["memory.max"].strip() != properties.get("MemoryMax")
                or cgroup_evidence["memory.swap.max"].strip() != properties.get("MemorySwapMax")
                or cgroup_evidence["memory.oom.group"].strip() != "1"
                or cgroup_events.get("populated") != "1"
                or any(
                    name not in memory_events
                    or not memory_events[name].isdigit()
                    or int(memory_events[name]) < 0
                    for name in required_events
                )
                or memory_events["oom"] != "0"
                or memory_events["oom_kill"] != "0"
                or memory_events["oom_group_kill"] != "0"
            ):
                raise RuntimeError("safe substituted unit cgroup limits or OOM evidence mismatch")
            _write_new(release, b"release\n")
            deadline = time.monotonic() + 10.0
            while True:
                terminal = _properties(
                    _run(
                        "systemctl",
                        "--user",
                        "show",
                        unit_name,
                        "-p",
                        "Result",
                        "-p",
                        "ExecMainStatus",
                        "-p",
                        "ActiveState",
                        check=False,
                    ).stdout
                )
                if terminal.get("ActiveState") in {"inactive", "failed"}:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("safe substituted ExecStart did not terminate")
                time.sleep(0.01)
            if terminal.get("Result") != "success" or terminal.get("ExecMainStatus") != "0":
                raise RuntimeError("safe substituted ExecStart terminal result mismatch")
            cgroup_evidence["terminal_result"] = _canonical(terminal).decode("utf-8")
        finally:
            active_error = sys.exception()
            cleanup_errors: list[BaseException] = []
            for command in (
                ("systemctl", "--user", "stop", unit_name),
                ("systemctl", "--user", "reset-failed", unit_name),
            ):
                try:
                    _run(*command, check=False)
                except BaseException as error:
                    cleanup_errors.append(error)
            try:
                if link.exists() or link.is_symlink():
                    if not link.is_symlink():
                        raise RuntimeError("transient probe unit link changed type")
                    link.unlink()
            except BaseException as error:
                cleanup_errors.append(error)
            try:
                _run("systemctl", "--user", "daemon-reload", check=False)
            except BaseException as error:
                cleanup_errors.append(error)
            if cleanup_errors:
                failures = [
                    *([active_error] if active_error is not None else []),
                    *cleanup_errors,
                ]
                raise BaseExceptionGroup(
                    "safe substituted unit cleanup failed", failures
                ) from None
        receipts.append(
            {
                "scope": scope,
                "role": role,
                "production_unit": production_name,
                "credential": source,
                "properties": properties,
                "cgroup_evidence": cgroup_evidence,
                "production_prestart": probe_prestart,
                "probe_omits_production_prestart": True,
                "network_deny_overlay": network_deny_overlay,
                "readonly_path_overlays": readonly_overlays,
                "probe_unit_sha256": hashlib.sha256(probe_unit).hexdigest(),
            }
        )
    final_verified = verify_persisted_topology(control_root, key_root, approval, requested_scope)
    final_properties = {
        entry["unit"]["path"].rsplit("/", 1)[-1]: _production_properties(
            entry["unit"]["path"].rsplit("/", 1)[-1]
        )
        for entry in final_verified["units"]
    }
    final_common_admission = admission_snapshot(final_verified, final_properties)
    if final_verified != verified or final_common_admission != common_admission:
        raise RuntimeError("launch-admission state drifted during persisted probes")
    final_probe_control_artifacts = [
        controls._file(Path(item["path"])) for item in probe_only_control_artifacts
    ]
    final_late_bound_probe = {
        "production_receipt": {
            "path": str(late_bound_receipt),
            "state": "absent",
        },
        "consumers": late_bound_consumers,
        "placeholder": (
            final_probe_control_artifacts[0] if final_probe_control_artifacts else None
        ),
    }
    if (
        late_bound_receipt.exists()
        or late_bound_receipt.is_symlink()
        or final_probe_control_artifacts != probe_only_control_artifacts
        or final_late_bound_probe != late_bound_probe
    ):
        raise RuntimeError("late-bound probe state drifted during persisted probes")
    aggregate = {
        "schema": "alpha_max_v8_persisted_probe_launch_admission.v1",
        "verdict": "PASS",
        **final_common_admission,
        "control_artifacts": [
            *final_common_admission["control_artifacts"],
            *probe_artifacts.values(),
            *probe_only_control_artifacts,
        ],
        "units": receipts,
        "probe_admission": probe_artifacts,
        "late_bound_probe": late_bound_probe,
    }
    _write_new(audit_path, _canonical(aggregate), 0o600)
    signed = _sign_launch_admission(
        controls,
        admission_root,
        evidence_root,
        key_root,
        authority,
        aggregate,
        requested_scope,
    )
    return signed


def execute_persisted_topology(
    control_root: Path,
    key_root: Path,
    approval: Path,
    evidence_root: Path,
    scope: str = "acquisition",
) -> dict[str, Any]:
    try:
        return _execute_persisted_topology(control_root, key_root, approval, evidence_root, scope)
    except BaseException:
        unit_directory = Path.home() / ".config/systemd/user"
        systemd_root = control_root / "systemd"
        if systemd_root.is_dir():
            for link in unit_directory.iterdir() if unit_directory.is_dir() else ():
                if link.is_symlink() and link.resolve().parent == systemd_root:
                    link.unlink()
            _run("systemctl", "--user", "daemon-reload", check=False)
        raise


def _sign_launch_admission(
    controls,
    admission_root: Path,
    evidence_root: Path,
    key_root: Path,
    authority: dict[str, Any],
    payload: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    payload_path = admission_root / f"launch-admission-{scope}.payload.json"
    signature_path = admission_root / f"launch-admission-{scope}.signature"
    envelope_path = admission_root / f"launch-admission-{scope}.json"
    _write_new(payload_path, _canonical(payload), 0o600)
    socket_token = f"luminaquant-alpha-max-admission-{secrets.token_hex(12)}"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.settimeout(30.0)
    listener.bind("\0" + socket_token)
    listener.listen(1)
    unit = controls._unit(
        "LuminaQuant authority launch-admission signer",
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            _ADMISSION_SIGN_CODE,
            "--private-key",
            "%d/authority.private",
            "--payload",
            str(payload_path),
            "--socket",
            socket_token,
        ],
        {"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"},
        {"high": 67_108_864, "max": 134_217_728, "swap": 33_554_432},
        None,
        read_paths=[str(payload_path)],
        write_paths=[],
        inaccessible_paths=[str(key_root)],
        load_credential=f"authority.private:{key_root / 'authority.private'}",
        observer=False,
    )
    unit_name = f"luminaquant-launch-admission-{secrets.token_hex(12)}.service"
    unit_path = evidence_root / unit_name
    unit_link = Path.home() / ".config/systemd/user" / unit_name
    _write_new(unit_path, controls._render_systemd_unit(unit))
    try:
        try:
            unit_link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(unit_path, unit_link)
            _run("systemctl", "--user", "daemon-reload")
            _run("systemd-analyze", "--user", "verify", str(unit_path))
            _run("systemctl", "--user", "start", unit_name)
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(30.0)
                chunks: list[bytes] = []
                total = 0
                while total <= 64:
                    chunk = connection.recv(65 - total)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
            signature = b"".join(chunks)
            if len(signature) != 64:
                raise RuntimeError("authority launch-admission signature length is invalid")
            deadline = time.monotonic() + 10.0
            while True:
                shown = _properties(
                    _run(
                        "systemctl",
                        "--user",
                        "show",
                        unit_name,
                        "-p",
                        "Result",
                        "-p",
                        "ExecMainStatus",
                        "-p",
                        "ActiveState",
                        "-p",
                        "RestrictAddressFamilies",
                        "-p",
                        "IPAddressDeny",
                        "-p",
                        "LoadCredential",
                        "-p",
                        "BindPaths",
                        check=False,
                    ).stdout
                )
                if shown.get("ActiveState") in {"inactive", "failed"}:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("authority launch-admission signer did not terminate")
                time.sleep(0.01)
            if (
                shown.get("Result") != "success"
                or shown.get("ExecMainStatus") != "0"
                or shown.get("RestrictAddressFamilies") != "AF_UNIX"
                or not _ip_address_deny_all(shown.get("IPAddressDeny"))
                or shown.get("BindPaths", "")
                or shown.get("LoadCredential")
                not in {"[unprintable]", f"authority.private:{key_root / 'authority.private'}"}
            ):
                raise RuntimeError("authority launch-admission signer readback failed")
        finally:
            _run("systemctl", "--user", "stop", unit_name, check=False)
            _run("systemctl", "--user", "reset-failed", unit_name, check=False)
            if unit_link.is_symlink():
                unit_link.unlink()
            _run("systemctl", "--user", "daemon-reload", check=False)
    finally:
        listener.close()
    public = base64.b64decode(authority["public_key_b64"], validate=True)
    Ed25519PublicKey.from_public_bytes(public).verify(
        signature, controls._read_regular(payload_path)[1]
    )
    _write_new(signature_path, signature, 0o600)
    envelope = {
        "schema": "alpha_max_v8_signed_launch_admission.v1",
        "payload": payload,
        "signature_b64": base64.b64encode(signature).decode("ascii"),
        "authority_key_id": authority["key_id"],
    }
    _write_new(envelope_path, _canonical(envelope), 0o600)
    return envelope


def _probe_one(evidence_root: Path, credential_name: str) -> dict[str, Any]:
    controls = _controls()
    if not evidence_root.is_absolute() or evidence_root.exists() or evidence_root.is_symlink():
        raise ValueError("evidence root must be a fresh absolute path")
    evidence_root.mkdir(mode=0o700)
    parent_fd = os.open(evidence_root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)

    token = secrets.token_hex(12)
    unit_name = f"luminaquant-credential-probe-{credential_name}-{token}.service"
    key_root = evidence_root.parent / f".{evidence_root.name}.{token}.keys"
    key_root.mkdir(mode=0o700)
    source = key_root / f"{credential_name}.private"
    public = key_root / f"{credential_name}.public"
    secret = secrets.token_bytes(32)
    _write_new(source, secret, 0o400)
    _write_new(public, secrets.token_bytes(32), 0o400)
    expected_sha256 = hashlib.sha256(secret).hexdigest()
    marker = evidence_root / "execstart-marker.json"
    unit_path = evidence_root / unit_name
    unit_link = Path.home() / ".config/systemd/user" / unit_name
    receipt_path = evidence_root / "terminal-probe.json"
    started_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    unit = controls._unit(
        "LuminaQuant non-network credential delivery probe",
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            _PROBE_CODE,
            "--private-key",
            f"%d/{credential_name}.private",
            "--expected-sha256",
            expected_sha256,
            "--marker",
            str(marker),
        ],
        {"HOME": str(evidence_root)},
        {"high": 67_108_864, "max": 134_217_728, "swap": 33_554_432},
        None,
        read_paths=[],
        write_paths=[str(evidence_root)],
        inaccessible_paths=[str(key_root)],
        load_credential=f"{credential_name}.private:{source}",
        observer=False,
    )
    key_files = [
        {
            "name": credential_name,
            "private": controls._file(source),
            "public": controls._file(public),
        }
    ]
    binding = controls._credential_binding(unit, key_files)
    rendered = controls._render_systemd_unit(unit)
    _write_new(unit_path, rendered)
    unit_identity = controls._file(unit_path)
    reload_unit_identity: dict[str, Any] | None = None
    terminal_unit_identity: dict[str, Any] | None = None

    error = ""
    properties: dict[str, str] = {}
    verify_stdout = ""
    verify_stderr = ""
    status_output = ""
    journal_output = ""
    cleanup_errors: list[str] = []
    try:
        unit_link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(unit_path, unit_link)
        _run("systemctl", "--user", "daemon-reload")
        reload_unit_identity = controls._file(unit_path)
        if reload_unit_identity != unit_identity or os.readlink(unit_link) != str(unit_path):
            raise RuntimeError("systemd unit identity changed during reload")
        verified = _run("systemd-analyze", "--user", "verify", str(unit_path))
        verify_stdout, verify_stderr = verified.stdout, verified.stderr
        _run("systemctl", "--user", "start", unit_name)
        deadline = time.monotonic() + 10.0
        while not marker.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("ExecStart marker was not durably published")
            time.sleep(0.01)
        while True:
            shown = _run(
                "systemctl",
                "--user",
                "show",
                unit_name,
                "-p",
                "Result",
                "-p",
                "ExecMainCode",
                "-p",
                "ExecMainStatus",
                "-p",
                "ActiveState",
                "-p",
                "SubState",
                "-p",
                "UMask",
                "-p",
                "MemoryHigh",
                "-p",
                "MemoryMax",
                "-p",
                "MemorySwapMax",
                "-p",
                "OOMPolicy",
                "-p",
                "FragmentPath",
                "-p",
                "ExecStart",
                "-p",
                "LoadCredential",
                "-p",
                "BindPaths",
            )
            properties = _properties(shown.stdout)
            if properties.get("ActiveState") in {"inactive", "failed"}:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("credential probe unit did not reach a terminal state")
            time.sleep(0.01)
        marker_value = json.loads(marker.read_text(encoding="utf-8"))
        terminal_unit_identity = controls._file(unit_path)
        fragment_path = properties.get("FragmentPath", "")
        exec_start = properties.get("ExecStart", "")
        load_credential = properties.get("LoadCredential", "")
        credential_path = marker_value.get("credential_path")
        if (
            marker_value.get("entered_execstart") is not True
            or marker_value.get("credential_sha256") != expected_sha256
            or not isinstance(credential_path, str)
            or not credential_path.endswith(f"/{credential_name}.private")
            or credential_path == str(source)
            or properties.get("Result") != "success"
            or properties.get("ExecMainCode") not in {"0", "1"}
            or properties.get("ExecMainStatus") != "0"
            or properties.get("ActiveState") != "inactive"
            or properties.get("UMask") != "0077"
            or terminal_unit_identity != unit_identity
            or fragment_path not in {str(unit_link), str(unit_path)}
            or "/usr/bin/python3" not in exec_start
            or credential_path not in exec_start
            or expected_sha256 not in exec_start
            or str(marker) not in exec_start
            or load_credential not in {"[unprintable]", f"{credential_name}.private:{source}"}
            or str(source) in exec_start
            or f"LoadCredential={credential_name}.private:{source}" not in rendered.decode()
            or properties.get("BindPaths") != unit["Service"]["BindPaths"][0] + ":rbind"
            or f"%d/{credential_name}.private" not in rendered.decode()
            or properties.get("MemoryHigh") != "67108864"
            or properties.get("MemoryMax") != "134217728"
            or properties.get("MemorySwapMax") != "33554432"
            or properties.get("OOMPolicy") != "kill"
        ):
            raise RuntimeError("live credential or ExecStart readback mismatch")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        status_output = _run(
            "systemctl", "--user", "status", unit_name, "--no-pager", "--full", check=False
        ).stdout
        journal_output = _run(
            "journalctl",
            "--user-unit",
            unit_name,
            "--no-pager",
            "--lines=50",
            "--output=short-iso",
            check=False,
        ).stdout
    finally:
        for command in (
            ("systemctl", "--user", "stop", unit_name),
            ("systemctl", "--user", "reset-failed", unit_name),
        ):
            try:
                _run(*command, check=False)
            except BaseException as exc:
                cleanup_errors.append(f"{' '.join(command)}: {exc}")
        try:
            if unit_link.is_symlink():
                unit_link.unlink()
        except BaseException as exc:
            cleanup_errors.append(f"unit link cleanup: {exc}")
        try:
            _run("systemctl", "--user", "daemon-reload", check=False)
        except BaseException as exc:
            cleanup_errors.append(f"daemon-reload cleanup: {exc}")
        try:
            shutil.rmtree(key_root)
        except BaseException as exc:
            cleanup_errors.append(f"credential cleanup: {exc}")

    verdict = "PASS" if not error and not cleanup_errors else "FAIL"
    receipt = {
        "schema": "alpha_max_v8_systemd_credential_probe.v1",
        "verdict": verdict,
        "started_utc": started_utc,
        "completed_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "network_allowed": False,
        "unit_name": unit_name,
        "unit": unit_identity,
        "unit_after_reload": reload_unit_identity,
        "unit_after_execution": terminal_unit_identity,
        "credential_binding": binding,
        "expected_credential_sha256": expected_sha256,
        "properties": properties,
        "execstart_marker": json.loads(marker.read_text(encoding="utf-8"))
        if marker.exists()
        else None,
        "systemd_analyze_stdout": verify_stdout,
        "systemd_analyze_stderr": verify_stderr,
        "failure": error,
        "status_output": status_output[-65_536:],
        "journal_output": journal_output[-65_536:],
        "cleanup_errors": cleanup_errors,
        "unit_link_absent": not unit_link.exists() and not unit_link.is_symlink(),
        "credential_root_absent": not key_root.exists(),
    }
    _write_new(receipt_path, _canonical(receipt), 0o600)
    if verdict != "PASS":
        raise RuntimeError(f"systemd credential probe failed; evidence: {receipt_path}")
    return receipt


def probe(evidence_root: Path) -> dict[str, Any]:
    if not evidence_root.is_absolute() or evidence_root.exists() or evidence_root.is_symlink():
        raise ValueError("evidence root must be a fresh absolute path")
    evidence_root.mkdir(mode=0o700)
    parent_fd = os.open(evidence_root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    receipts = {
        credential: _probe_one(evidence_root / credential, credential)
        for credential in ("authority", "acquisition", "phase_preparation", "one_touch")
    }
    receipt = {
        "schema": "alpha_max_v8_systemd_credential_probe.v2",
        "verdict": "PASS",
        "network_allowed": False,
        "scopes": receipts,
    }
    _write_new(evidence_root / "all-scopes-probe.json", _canonical(receipt), 0o600)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--key-root", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument(
        "--scope",
        choices=("acquisition", "phase_preparation", "one_touch", "all-ready"),
        default="acquisition",
    )
    args = parser.parse_args(argv)
    if args.control_root or args.key_root or args.approval:
        if not (args.control_root and args.key_root and args.approval and args.evidence_root):
            parser.error(
                "persisted execution requires control root, key root, approval, and evidence root"
            )
        execute_persisted_topology(
            args.control_root, args.key_root, args.approval, args.evidence_root, args.scope
        )
    elif args.evidence_root is not None:
        probe(args.evidence_root)
    else:
        parser.error("provide persisted topology or evidence root")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
