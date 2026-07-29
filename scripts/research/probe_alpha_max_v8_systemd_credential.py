#!/usr/bin/env python3
"""Exercise the production static-unit credential path without network access."""

from __future__ import annotations

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
import time
from typing import Any

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


def probe(evidence_root: Path) -> dict[str, Any]:
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
    unit_name = f"luminaquant-credential-probe-{token}.service"
    key_root = evidence_root.parent / f".{evidence_root.name}.{token}.keys"
    key_root.mkdir(mode=0o700)
    source = key_root / "authority.private"
    public = key_root / "authority.public"
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
            sys.executable,
            "-I",
            "-S",
            "-c",
            _PROBE_CODE,
            "--private-key",
            "%d/authority.private",
            "--expected-sha256",
            expected_sha256,
            "--marker",
            str(marker),
        ],
        {"HOME": str(evidence_root)},
        {"high": 67_108_864, "max": 134_217_728, "swap": 33_554_432},
        None,
        read_paths=[controls.CURRENT],
        write_paths=[str(evidence_root)],
        inaccessible_paths=[str(key_root)],
        load_credential=f"authority.private:{source}",
    )
    key_files = [
        {
            "name": "authority",
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
            or not credential_path.endswith("/authority.private")
            or credential_path == str(source)
            or properties.get("Result") != "success"
            or properties.get("ExecMainCode") not in {"0", "1"}
            or properties.get("ExecMainStatus") != "0"
            or properties.get("ActiveState") != "inactive"
            or properties.get("UMask") != "0077"
            or terminal_unit_identity != unit_identity
            or fragment_path not in {str(unit_link), str(unit_path)}
            or str(sys.executable) not in exec_start
            or credential_path not in exec_start
            or expected_sha256 not in exec_start
            or str(marker) not in exec_start
            or load_credential not in {"[unprintable]", f"authority.private:{source}"}
            or str(source) in exec_start
            or f"LoadCredential=authority.private:{source}" not in rendered.decode()
            or "%d/authority.private" not in rendered.decode()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    args = parser.parse_args(argv)
    probe(args.evidence_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
