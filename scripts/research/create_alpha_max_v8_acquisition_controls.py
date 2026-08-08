#!/usr/bin/env python3
"""Build, but never execute, the Alpha-Max v8 acquisition control package."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import importlib.machinery
import json
import os
import re
import stat
import subprocess
from pathlib import Path
import sys
import unicodedata
import types
from typing import Any

FileIdentity = dict[str, Any]

_FORBIDDEN_ROOTS = (
    "/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source",
    "/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc",
)
_SCOPES = ("acquisition", "phase_preparation", "one_touch")
_FILE_ROLES = (
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

CURRENT = "/home/hoky/Quants-agent/LuminaQuant"
ACCEPTED = "/home/hoky/Quants-agent/LuminaQuant-alpha-max-fresh-20260718"
ACCEPTED_COMMIT = "391000b40717386765bfa39bd212d91c2e3be794"
BASELINE = "629d91e5d4aac26911af65a4a5e15ebdcbded30f"
RECOVERY_ROOT = Path("/home/hoky/quants-recovery-runs")
CURRENT_APPROVAL_LEAF = "current-state-approval-v9.json"
RUN_ID = "0ea09388c6a3b52a722727e00135191c10a09b7652aaa3c31918a19f5ccea5db"
EXECUTION_ALIAS_ROOT = Path(f"/mnt/wsl/luminaquant-alpha-max-execution-{RUN_ID}")
ACQUISITION_REQUEST_ID = "7d11aa88513b3e2b649fac0dc517f8e13fab246b58d9c244f138d0a919a600f1"
PHASE_PREPARATION_REQUEST_ID = "b4c2f5e8c954dc6aff5c757780c7aef8ff0ca1378fdd368e8fc8ea29491832ec"
ONE_TOUCH_REQUEST_ID = "ec426deceeed57c9b22ac1772f3afe03f7ac6262a8028502cb843eb7e4e599bb"
CURRENT_APPROVAL = str(RECOVERY_ROOT / CURRENT_APPROVAL_LEAF)
_OWNER_SESSION_RUNTIME_PATH = ".gjc/_session-019fad7d-536a-7000-b794-52ccaa961746/"
_OWNER_SESSION_RUNTIME_EXCLUDE = f":(exclude){_OWNER_SESSION_RUNTIME_PATH}**"
_IGNORED_SOURCE_PATHS = (
    "src",
    "scripts",
    "apps",
    ":(exclude)apps/dashboard_web/.next/**",
    ":(exclude)apps/dashboard_web/node_modules/**",
)
ALIGNMENT = (
    "/home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-rev515-alignment-receipt-v5.json"
)
ALIGNMENT_SHA256 = "8687b52180502a11de9fbe317a19d00bb4492c464b3bf33d4eda2437683ca812"
AUTHORITY_MEMORY_HIGH = 402653184
AUTHORITY_MEMORY_MAX = 536870912
AUTHORITY_SWAP_MAX = 268435456
HEX = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_ARGUMENTS = {
    "authority.private": "--private-key",
    "authority.public": "--authority-public-key",
    "acquisition.private": "--observer-private-key",
    "phase_preparation.private": "--observer-private-key",
    "one_touch.private": "--observer-private-key",
}


def _credential_argument(name: str) -> str:
    try:
        return _CREDENTIAL_ARGUMENTS[name]
    except KeyError:
        _fail("systemd credential name has no role argument")


_RUNTIME_NAMES = ("current_python", "accepted_python", "base_python")
EXECUTABLE_PINS = {
    "current_python": {
        "path": "/home/hoky/Quants-agent/LuminaQuant/.venv-g056v8-current/bin/python-g056v8-current",
        "sha256": "a1512f9a07029c4a9b02a1bb63bbd156d36b0dcb26f49cb7f5ee175f19b222da",
        "byte_count": 32299584,
        "mode": 0o555,
        "package_freeze_sha256": "3b8e4d900ddfc1bf05d65ff4fcf1eb6a04709dcc684ad8f4a49fe3bd4bba9724",
    },
    "accepted_python": {
        "path": "/home/hoky/Quants-agent/LuminaQuant-alpha-max-fresh-20260718/.venv-g056v8-accepted/bin/python-g056v8-accepted",
        "sha256": "a1512f9a07029c4a9b02a1bb63bbd156d36b0dcb26f49cb7f5ee175f19b222da",
        "byte_count": 32299584,
        "mode": 0o555,
        "package_freeze_sha256": "df09a5a1d4d1ab657d6a11d28eaf00cea06df4d9e28c0ef81ec5382257d6abf6",
    },
    "base_python": {
        "path": "/home/hoky/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14",
        "sha256": "a1512f9a07029c4a9b02a1bb63bbd156d36b0dcb26f49cb7f5ee175f19b222da",
        "byte_count": 32299584,
        "mode": 0o755,
    },
    "telemetry_script": {
        "path": "/home/hoky/Quants-agent/LuminaQuant/scripts/research/monitor_alpha_max_v8_resources.py",
        "sha256": "5d3e7eedea70102c6aa182e153139131bdbcd3ffc904499184aecc55eee54d4f",
        "byte_count": 28491,
        "mode": 0o600,
    },
}
ROLE_PINS = {
    "runbook": (
        ACCEPTED,
        "docs/research_note/alpha_max_data_pc_runbook_20260711.md",
        "249694fb1513354d61f67552f5c1b9175382f3c2bf9f271ee64dc0358d3c663f",
    ),
    "alpha_uv_lock": (
        ACCEPTED,
        "uv.lock",
        "59d9de230be950761736c24e04af3456e229cf4aa077536167fb7e650a71c339",
    ),
    "portfolio": (
        ACCEPTED,
        "configs/research/alpha_max_portfolio_20260711_listing_aware.json",
        "2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c",
    ),
    "contract_manifest": (
        ACCEPTED,
        "configs/research/alpha_max_contract_manifest_20260711_listing_aware.json",
        "ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220",
    ),
    "availability_evidence": (
        ACCEPTED,
        "configs/research/alpha_max_official_availability_evidence_20260711.json",
        "214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719",
    ),
    "preparer": (
        ACCEPTED,
        "scripts/research/prepare_alpha_max_phase_roots.py",
        "ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac",
    ),
    "prelock_script": (
        ACCEPTED,
        "scripts/research/run_alpha_max_prelock.py",
        "838d633ae34d44443dad4990a79f4d8caa95f7102ffe2a649ed341b1bed16ad0",
    ),
    "historical_script": (
        ACCEPTED,
        "scripts/research/run_alpha_max_historical_evaluation.py",
        "951290033c7efd9b59ba5418e38d96fbdcf3885211915b29010b79ae545f3fb0",
    ),
    "process_boundary": (
        ACCEPTED,
        "src/lumina_quant/alpha_max_process_boundary.py",
        "f95e8e0d356ca36063a415a7b37919e72d9d1f47af7d2c447e228546fddfb94c",
    ),
    "policy_module": (CURRENT, "src/lumina_quant/alpha_max_terminal_policy.py", None),
    "authority_script": (CURRENT, "scripts/research/run_alpha_max_terminal_authority.py", None),
    "observer_script": (CURRENT, "scripts/research/run_alpha_max_terminal_observer.py", None),
    "key_creator": (CURRENT, "scripts/research/create_alpha_max_terminal_keys.py", None),
    "acquirer": (
        CURRENT,
        "scripts/research/acquire_alpha_max_official_source.py",
        "d3c674ecf28c5869eab43f9903b4479185b36faca108919868c2f2c31662db70",
    ),
    "phase_wrapper": (
        CURRENT,
        "scripts/research/run_alpha_max_phase_preparation_from_eligible_source.py",
        "0db198af0b743df0bdb6d3700ed8f0bc53cc28373846ba890e1ac70edc287ce1",
    ),
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def _json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def _validated_absolute(value: str, *, reject_legacy_name: bool) -> Path:
    if not isinstance(value, str):
        _fail("path is not a string")
    path = Path(value)
    if not path.is_absolute() or str(path) != value or "/../" in value or value.endswith("/."):
        _fail("path is not canonically absolute")
    # This is lexical and occurs before every filesystem operation.
    if any(value == root or value.startswith(root + "/") for root in _FORBIDDEN_ROOTS) or (
        reject_legacy_name and re.search(r"(^|[-_/])v[67]($|[-_/])", value, re.I)
    ):
        _fail("legacy or forbidden path")
    return path


def _absolute(value: str) -> Path:
    return _validated_absolute(value, reject_legacy_name=True)


def _inventory_absolute(value: str) -> Path:
    # Git may track unrelated versioned research names; exact forbidden roots remain rejected.
    return _validated_absolute(value, reject_legacy_name=False)


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns, a.st_ctime_ns) == (
        b.st_dev,
        b.st_ino,
        b.st_size,
        b.st_mtime_ns,
        b.st_ctime_ns,
    )


def _read_all(fd: int, size: int) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(fd, min(1 << 20, remaining))
        if not chunk:
            _fail("short read")
        parts.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        _fail("file grew while read")
    return b"".join(parts)


def _open_directory(path: Path, *, inventory_entry: bool = False) -> int:
    path = (_inventory_absolute if inventory_entry else _absolute)(str(path))
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_regular(path: Path, *, inventory_entry: bool = False) -> tuple[os.stat_result, bytes]:
    path = (_inventory_absolute if inventory_entry else _absolute)(str(path))
    parent_fd = _open_directory(path.parent, inventory_entry=inventory_entry)
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            _fail(f"unsafe file: {path}")
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
        try:
            opened = os.fstat(fd)
            data = _read_all(fd, opened.st_size)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    if not _same(before, opened) or not _same(opened, after):
        _fail(f"unstable file: {path}")
    return opened, data


def _read_symlink(path: Path, *, inventory_entry: bool = False) -> tuple[os.stat_result, bytes]:
    path = (_inventory_absolute if inventory_entry else _absolute)(str(path))
    parent_fd = _open_directory(path.parent, inventory_entry=inventory_entry)
    try:
        before = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISLNK(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
            _fail(f"unsafe symlink: {path}")
        target = os.readlink(path.name, dir_fd=parent_fd)
        after = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(parent_fd)
    data = os.fsencode(target)
    if not _same(before, after) or len(data) != after.st_size:
        _fail(f"unstable symlink: {path}")
    return after, data


def _execution_alias() -> dict[str, Any]:
    info, raw_target = _read_symlink(EXECUTION_ALIAS_ROOT)
    target = os.fsdecode(raw_target)
    if target != str(RECOVERY_ROOT):
        _fail("execution alias target mismatch")
    return {
        "path": str(EXECUTION_ALIAS_ROOT),
        "target": target,
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
    }


def _directory(path: Path, *, private: bool = False) -> dict[str, Any]:
    fd = _open_directory(path)
    try:
        info = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink < 2
        or stat.S_IMODE(info.st_mode) & 0o022
        or (private and stat.S_IMODE(info.st_mode) != 0o700)
    ):
        _fail(f"unsafe directory: {path}")
    return {
        "path": str(path),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
    }


def _identity(path: Path, info: os.stat_result, data: bytes) -> FileIdentity:
    return {
        "path": str(path),
        "sha256": _sha(data),
        "byte_count": len(data),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
    }


def _file(path: Path) -> FileIdentity:
    info, data = _read_regular(path)
    return _identity(path, info, data)


def _prerequisite(kind: str, item: FileIdentity) -> dict[str, Any]:
    fields = ("path", "sha256", "byte_count", "st_dev", "st_ino", "mode", "nlink")
    return {"kind": kind, **{field: item[field] for field in fields}}


def _absent(path: Path) -> dict[str, Any]:
    path = _absolute(str(path))
    parent = _directory(path.parent)
    parent_fd = _open_directory(path.parent)
    try:
        os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return {"path": str(path), "parent": parent, "leaf": path.name, "must_be_absent": True}
    finally:
        os.close(parent_fd)
    _fail(f"output exists: {path}")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short write")
        view = view[count:]


def _write_new(path: Path, value: Any) -> dict[str, Any]:
    return _write_bytes_new(path, _canonical(value))


def _load_canonical(path: Path) -> dict[str, Any]:
    _, raw = _read_regular(path)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=lambda pairs: _no_duplicates(pairs),
            parse_constant=_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid canonical JSON: {path}") from exc
    if not isinstance(value, dict) or _canonical(value) != raw:
        _fail(f"noncanonical JSON: {path}")
    return value


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        _fail("duplicate JSON key")
    return result


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        (
            "/usr/bin/git",
            "--no-optional-locks",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(root),
            *args,
        ),
        check=True,
        capture_output=True,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    ).stdout


def _source_git(root: Path, *args: str) -> bytes:
    if args and args[0] == "diff":
        args = ("diff", "--no-ext-diff", *args[1:])
    return _git(root, *args, "--", ".", _OWNER_SESSION_RUNTIME_EXCLUDE)


def _inventory(root: Path) -> bytes:
    names = _source_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(
        b"\0"
    )[:-1]
    return _inventory_paths(root, names)


def _ignored_source_inventory(root: Path) -> bytes:
    """Bind ignored source/runtime bytes except explicit non-production web build trees."""
    names = _git(
        root,
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        *_IGNORED_SOURCE_PATHS,
    ).split(b"\0")[:-1]
    return _inventory_paths(root, names)


def _inventory_paths(root: Path, names: list[bytes]) -> bytes:
    records = []
    for raw in names:
        name = raw.decode("utf-8")
        if not name or name.startswith("/") or ".." in Path(name).parts:
            _fail("unsafe inventory path")
        path = root / name
        info = os.stat(path, follow_symlinks=False)
        if stat.S_ISREG(info.st_mode):
            info, content = _read_regular(path, inventory_entry=True)
            kind = "regular"
        elif stat.S_ISLNK(info.st_mode):
            info, content = _read_symlink(path, inventory_entry=True)
            kind = "symlink"
        else:
            _fail(f"unsupported inventory entry: {path}")
        records.append(
            {
                "path": name,
                "type": kind,
                "mode": stat.S_IMODE(info.st_mode),
                "size": len(content),
                "sha256": _sha(content),
            }
        )
    return _canonical(records)


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _validate_runtime_extensions(root: Path) -> None:
    runtime_roots = tuple(
        Path(EXECUTABLE_PINS[name]["path"]).parent.parent.resolve() for name in _RUNTIME_NAMES
    )
    repository_roots = (
        (Path(CURRENT, "src").resolve(),)
        if root.resolve() == runtime_roots[0]
        else (Path(ACCEPTED, "src").resolve(),)
        if root.resolve() == runtime_roots[1]
        else ()
    )
    for path in root.rglob("*"):
        if path.is_symlink():
            try:
                target = path.resolve(strict=True)
            except OSError:
                _fail(f"runtime symlink target is missing: {path}")
            if not _within(target, (root.resolve(), *runtime_roots)):
                _fail(f"runtime symlink escapes inventoried roots: {path}")
        if path.suffix == ".pth" and path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("import "):
                    if not re.fullmatch(r"import [A-Za-z_][A-Za-z0-9_]*", line):
                        _fail(f"unsafe runtime import hook: {path}")
                    module = line.removeprefix("import ")
                    if not any(root.rglob(f"{module}.py")):
                        _fail(f"runtime import hook module is not inventoried: {path}")
                    continue
                target = Path(line)
                if not target.is_absolute():
                    target = path.parent / target
                try:
                    target = target.resolve(strict=True)
                except OSError:
                    _fail(f"runtime .pth target is missing: {path}")
                if not _within(target, (root.resolve(), *runtime_roots, *repository_roots)):
                    _fail(f"runtime .pth target escapes inventoried roots: {path}")


def _runtime_inventory(interpreter: FileIdentity) -> bytes:
    """Bind every executable/importable entry below the pinned virtualenv root."""
    executable = _absolute(interpreter["path"])
    root = executable.parent.parent
    _validate_runtime_extensions(root)
    names: list[bytes] = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        relative = Path(directory).relative_to(root)
        for name in [*subdirectories, *filenames]:
            path = Path(directory) / name
            info = os.stat(path, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                continue
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                _fail(f"unsupported runtime entry: {path}")
            names.append(os.fsencode(str(relative / name)))
    return _inventory_paths(root, sorted(names))


def _only_owner_session_runtime_changes(root: Path) -> bool:
    """Approval admits only the declared owner-session runtime files in porcelain state."""
    allowed = _OWNER_SESSION_RUNTIME_PATH.encode()
    records = _git(root, "status", "--porcelain=v1", "-z").split(b"\0")[:-1]
    return all(
        len(record) >= 4 and record[2:3] == b" " and record[3:].startswith(allowed)
        for record in records
    )


def _record(raw: bytes) -> dict[str, Any]:
    return {"sha256": _sha(raw), "byte_count": len(raw)}


def _accepted_source_state() -> dict[str, Any]:
    root = Path(ACCEPTED)
    return {
        "root": ACCEPTED,
        "head": _git(root, "rev-parse", "HEAD").decode().strip(),
        "porcelain": _record(_source_git(root, "status", "--porcelain=v1", "-z")),
        "source_inventory": _record(_inventory(root)),
        "ignored_source_inventory": _record(_ignored_source_inventory(root)),
    }


_MOUNT_IDENTITY_SOURCE = r"""
import base64
import hashlib
import json
import os
import stat
import subprocess
import sys


def stat_tuple(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def read_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    parts = []
    while chunk := os.read(fd, 1 << 20):
        parts.append(chunk)
    return b"".join(parts)


def snapshot(path):
    before = os.lstat(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    fds.append(fd)
    opened = os.fstat(fd)
    expected = stat_tuple(opened)
    if not stat.S_ISREG(opened.st_mode) or stat_tuple(before) != expected:
        raise AssertionError("unstable final namespace input")
    data = read_fd(fd)
    if stat_tuple(os.fstat(fd)) != expected:
        raise AssertionError("final namespace input drift")
    return fd, expected, hashlib.sha256(data).hexdigest(), data


def check_snapshot(path, fd, expected, digest):
    if stat_tuple(os.fstat(fd)) != expected:
        raise AssertionError("final namespace retained fd drift")
    if hashlib.sha256(read_fd(fd)).hexdigest() != digest:
        raise AssertionError("final namespace retained fd hash drift")
    if stat_tuple(os.lstat(path)) != expected:
        raise AssertionError("final namespace path drift")


fds = []
try:
    config = json.loads(sys.argv[1])
    mounts, argv = config["mounts"], sys.argv[2:]
    assert argv and os.path.isabs(argv[0])
    actual = [
        {"path": path, "st_dev": info.st_dev, "st_ino": info.st_ino, "type": stat.S_IFMT(info.st_mode)}
        for path in (item["path"] for item in mounts)
        for info in [os.stat(path)]
    ]
    assert actual == mounts
    receipt = config["receipt"]
    payload_path = receipt.removesuffix(".json") + ".payload.json"
    receipt_fd, receipt_stat, receipt_digest, receipt_raw = snapshot(receipt)
    payload_fd, payload_stat, payload_digest, payload_raw = snapshot(payload_path)
    envelope, payload = json.loads(receipt_raw), json.loads(payload_raw)
    assert envelope["schema"] == "alpha_max_v8_signed_launch_admission.v1"
    assert envelope["payload"] == payload
    assert payload_raw == (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    public = base64.b64decode(config["authority_public_b64"], validate=True)
    signature = base64.b64decode(envelope["signature_b64"], validate=True)
    assert (
        len(public) == 32
        and len(signature) == 64
        and envelope["authority_key_id"] == hashlib.sha256(public).hexdigest()
    )
    public_fd = os.memfd_create("authority", os.MFD_CLOEXEC)
    signature_fd = os.memfd_create("signature", os.MFD_CLOEXEC)
    signed_payload_fd = os.memfd_create("payload", os.MFD_CLOEXEC)
    fds.extend((public_fd, signature_fd, signed_payload_fd))
    os.write(public_fd, b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00" + public)
    os.write(signature_fd, signature)
    os.write(signed_payload_fd, payload_raw)
    subprocess.run(
        (
            "/usr/bin/openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            f"/proc/self/fd/{public_fd}",
            "-keyform",
            "DER",
            "-rawin",
            "-in",
            f"/proc/self/fd/{signed_payload_fd}",
            "-sigfile",
            f"/proc/self/fd/{signature_fd}",
        ),
        check=True,
        capture_output=True,
        pass_fds=(public_fd, signature_fd, signed_payload_fd),
    )
    scope = receipt.rsplit("/", 1)[-1].removeprefix("launch-admission-").removesuffix(".json")
    assert scope == config["scope"] == payload["requested_scope"]
    matches = [
        item
        for item in payload["topology"]["units"]
        if item["production_unit"] == config["unit_name"]
    ]
    assert len(matches) == 1 and matches[0]["scope"] == scope
    artifacts = payload["control_artifacts"]
    assert artifacts
    artifact_snapshots = []
    for item in artifacts:
        fd, expected, digest, _data = snapshot(item["path"])
        assert (
            expected
            == (
                item["st_dev"],
                item["st_ino"],
                item["st_uid"],
                item["st_gid"],
                stat.S_IFREG,
                item["mode"],
                item["nlink"],
                item["byte_count"],
                os.fstat(fd).st_mtime_ns,
                os.fstat(fd).st_ctime_ns,
            )
            and digest == item["sha256"]
        )
        artifact_snapshots.append((item["path"], fd, expected, digest))
    check_snapshot(receipt, receipt_fd, receipt_stat, receipt_digest)
    check_snapshot(payload_path, payload_fd, payload_stat, payload_digest)
    for path, fd, expected, digest in artifact_snapshots:
        check_snapshot(path, fd, expected, digest)
    os.execv(argv[0], argv)
finally:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass
"""
_MOUNT_IDENTITY_B64 = base64.b64encode(_MOUNT_IDENTITY_SOURCE.encode()).decode()
_MOUNT_IDENTITY_CODE = (
    "import base64;exec(compile(base64.b64decode("
    + repr(_MOUNT_IDENTITY_B64)
    + "),'<alpha-max-final-namespace>','exec'))"
)


def _writable_mount_identities(write_paths: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": path,
            "st_dev": info.st_dev,
            "st_ino": info.st_ino,
            "type": stat.S_IFMT(info.st_mode),
        }
        for path in write_paths
        for info in [os.stat(path)]
    ]


def _wrap_execstart(
    argv: list[str],
    write_paths: list[str],
    *,
    receipt: str,
    unit_name: str,
    authority_public_b64: str,
) -> list[str]:
    if not argv:
        _fail("systemd ExecStart is empty")
    if not os.path.isabs(argv[0]):
        return argv
    scope = Path(receipt).name.removeprefix("launch-admission-").removesuffix(".json")
    if not scope or not unit_name:
        _fail("invalid final namespace admission binding")
    return [
        "/usr/bin/python3",
        "-I",
        "-S",
        "-c",
        _MOUNT_IDENTITY_CODE,
        json.dumps(
            {
                "mounts": _writable_mount_identities(write_paths),
                "receipt": receipt,
                "scope": scope,
                "unit_name": unit_name,
                "authority_public_b64": authority_public_b64,
            },
            separators=(",", ":"),
        ),
        *argv,
    ]


def _state(root: Path, approval: dict[str, Any]) -> str:
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    executables = _preflight_executables(
        {
            "current_python": Path(EXECUTABLE_PINS["current_python"]["path"]),
            "accepted_python": Path(EXECUTABLE_PINS["accepted_python"]["path"]),
            "telemetry_script": Path(EXECUTABLE_PINS["telemetry_script"]["path"]),
        }
    )
    actual = {
        "head": head,
        "porcelain": _record(_source_git(root, "status", "--porcelain=v1", "-z")),
        "commit_overlay": _record(
            _source_git(root, "diff", "--binary", f"{ACCEPTED_COMMIT}..HEAD")
        ),
        "worktree_overlay": _record(_source_git(root, "diff", "--binary", "HEAD")),
        "source_inventory": _record(_inventory(root)),
        "ignored_source_inventory": _record(_ignored_source_inventory(root)),
        "runtime_inventories": {
            name: _record(_runtime_inventory(executables[name])) for name in _RUNTIME_NAMES
        },
        "execution_alias": _execution_alias(),
    }
    for key, record in actual.items():
        if key == "head":
            if approval.get(key) != record:
                _fail("current approval HEAD mismatch")
        elif approval.get(key) != record:
            _fail(f"current approval mismatch: {key}")
    return head


def _approval(path: Path, fields: set[str], schema: str) -> dict[str, Any]:
    value = _load_canonical(path)
    if set(value) != fields or value.get("schema") != schema:
        _fail("approval schema mismatch")
    return value


def _create_current_approval(
    path: Path,
    *,
    root: Path,
    run_id: str,
    request_ids: dict[str, str],
    absent_paths: dict[str, Path],
) -> dict[str, Any]:
    """Publish the sole source-bound approval before any recovery artifact exists."""
    if path != RECOVERY_ROOT / CURRENT_APPROVAL_LEAF:
        _fail("approval path mismatch")
    _absent(path)
    for candidate in absent_paths.values():
        _absent(candidate)
    approval = {
        "schema": "alpha_max_v8_current_state_approval.v3",
        "repository_root": CURRENT,
        "head": _git(root, "rev-parse", "HEAD").decode().strip(),
        "accepted_alpha_commit": ACCEPTED_COMMIT,
        "baseline_ancestor": BASELINE,
        "verdict": "PASS_REVIEWED_OVERLAY",
        "porcelain": _record(_source_git(root, "status", "--porcelain=v1", "-z")),
        "commit_overlay": _record(
            _source_git(root, "diff", "--binary", f"{ACCEPTED_COMMIT}..HEAD")
        ),
        "worktree_overlay": _record(_source_git(root, "diff", "--binary", "HEAD")),
        "source_inventory": _record(_inventory(root)),
        "ignored_source_inventory": _record(_ignored_source_inventory(root)),
        "accepted_source_state": _accepted_source_state(),
        "runtime_inventories": {
            name: _record(
                _runtime_inventory(
                    _preflight_executables(
                        {
                            "current_python": Path(EXECUTABLE_PINS["current_python"]["path"]),
                            "accepted_python": Path(EXECUTABLE_PINS["accepted_python"]["path"]),
                            "telemetry_script": Path(EXECUTABLE_PINS["telemetry_script"]["path"]),
                        }
                    )[name]
                )
            )
            for name in _RUNTIME_NAMES
        },
        "execution_alias": _execution_alias(),
        "run_id": run_id,
        "request_ids": request_ids,
        "absent_recovery_artifacts": {
            name: str(candidate) for name, candidate in absent_paths.items()
        },
    }
    _write_new(path, approval)
    return approval


def _load_current_approval(
    path: Path,
    *,
    run_id: str,
    request_ids: dict[str, str],
    absent_paths: dict[str, Path],
    require_absent: bool = True,
) -> dict[str, Any]:
    approval = _approval(
        path,
        {
            "schema",
            "repository_root",
            "head",
            "accepted_alpha_commit",
            "baseline_ancestor",
            "verdict",
            "porcelain",
            "commit_overlay",
            "worktree_overlay",
            "source_inventory",
            "ignored_source_inventory",
            "runtime_inventories",
            "accepted_source_state",
            "execution_alias",
            "run_id",
            "request_ids",
            "absent_recovery_artifacts",
        },
        "alpha_max_v8_current_state_approval.v3",
    )
    if (
        approval["run_id"] != run_id
        or approval["request_ids"] != request_ids
        or approval["absent_recovery_artifacts"]
        != {name: str(candidate) for name, candidate in absent_paths.items()}
        or approval["repository_root"] != CURRENT
        or approval["accepted_alpha_commit"] != ACCEPTED_COMMIT
        or approval["baseline_ancestor"] != BASELINE
        or approval["verdict"] != "PASS_REVIEWED_OVERLAY"
        or approval["execution_alias"] != _execution_alias()
        or approval["accepted_source_state"] != _accepted_source_state()
    ):
        _fail("current approval binding mismatch")
    if require_absent:
        for candidate in absent_paths.values():
            _absent(candidate)
    _state(Path(CURRENT), approval)
    return approval


def _create_root(path: Path) -> None:
    path = _absolute(str(path))
    _directory(path.parent)
    parent_fd = _open_directory(path.parent)
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileExistsError:
        _fail(f"fresh root exists: {path}")
    finally:
        os.close(parent_fd)
    _directory(path, private=True)


def _matches_file_identity(info: os.stat_result, data: bytes, expected: FileIdentity) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_dev == expected["st_dev"]
        and info.st_ino == expected["st_ino"]
        and info.st_uid == expected["st_uid"]
        and info.st_gid == expected["st_gid"]
        and stat.S_IMODE(info.st_mode) == expected["mode"]
        and info.st_nlink == expected["nlink"]
        and len(data) == expected["byte_count"]
        and _sha(data) == expected["sha256"]
    )


def _freeze(interpreter: FileIdentity) -> dict[str, Any]:
    path = _absolute(interpreter["path"])
    parent_fd = _open_directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        opened = os.fstat(fd)
        data = _read_all(fd, opened.st_size)
        after_read = os.fstat(fd)
        if not _same(opened, after_read) or not _matches_file_identity(opened, data, interpreter):
            _fail("interpreter identity drift")
        os.lseek(fd, 0, os.SEEK_SET)
        program = (
            "import importlib.metadata as m,json,platform,sys;"
            "print(json.dumps({'schema':'alpha_max_v8_interpreter_package_freeze.v1',"
            "'interpreter_sha256':None,'implementation':sys.implementation.name,"
            "'python_version':platform.python_version(),'cache_tag':sys.implementation.cache_tag,"
            "'packages':[{'name':d.metadata['Name'],'version':d.version} for d in m.distributions() if d.metadata['Name']]},"
            "sort_keys=True,separators=(',',':')))"
        )
        result = subprocess.run(
            (str(path), "-I", "-c", program),
            executable=f"/proc/self/fd/{fd}",
            pass_fds=(fd,),
            capture_output=True,
            text=True,
        )
        after_execution = os.fstat(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        executed_data = _read_all(fd, after_execution.st_size)
        if (
            not _same(opened, after_execution)
            or not _matches_file_identity(after_execution, executed_data, interpreter)
            or _file(path) != interpreter
        ):
            _fail("interpreter identity drift")
    finally:
        os.close(fd)
    if result.returncode or result.stderr or len(result.stdout.encode()) > (4 << 20):
        _fail("interpreter freeze command failed")
    try:
        value = json.loads(result.stdout, object_pairs_hook=_no_duplicates)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid interpreter freeze") from exc
    if (
        set(value)
        != {
            "schema",
            "interpreter_sha256",
            "implementation",
            "python_version",
            "cache_tag",
            "packages",
        }
        or value["schema"] != "alpha_max_v8_interpreter_package_freeze.v1"
    ):
        _fail("invalid interpreter freeze schema")
    packages = value["packages"]
    if not isinstance(packages, list) or any(
        not isinstance(p, dict)
        or set(p) != {"name", "version"}
        or not isinstance(p["name"], str)
        or not isinstance(p["version"], str)
        for p in packages
    ):
        _fail("invalid interpreter package freeze")
    unique_packages: dict[str, dict[str, str]] = {}
    for package in packages:
        normalized = re.sub(
            r"[-_.]+",
            "-",
            unicodedata.normalize("NFKC", package["name"]).lower(),
        )
        if not normalized:
            _fail("noncanonical interpreter package freeze")
        existing = unique_packages.get(normalized)
        if existing is not None and existing != package:
            _fail("conflicting duplicate interpreter package")
        unique_packages[normalized] = package
    value["packages"] = sorted(
        unique_packages.values(),
        key=lambda p: (
            re.sub(r"[-_.]+", "-", unicodedata.normalize("NFKC", p["name"]).lower()),
            p["name"],
            p["version"],
        ),
    )
    value["interpreter_sha256"] = interpreter["sha256"]
    return value


def _preflight_executables(paths: dict[str, Path]) -> dict[str, FileIdentity]:
    identities: dict[str, FileIdentity] = {}
    for name, pin in EXECUTABLE_PINS.items():
        path = paths.get(name, Path(pin["path"]))
        if path != Path(pin["path"]):
            _fail(f"executable path mismatch: {name}")
        identity = _file(path)
        if (
            identity["sha256"] != pin["sha256"]
            or identity["byte_count"] != pin["byte_count"]
            or identity["mode"] != pin["mode"]
            or identity["nlink"] != 1
            or identity["st_uid"] != os.getuid()
        ):
            _fail(f"executable identity mismatch: {name}")
        identities[name] = identity
    return identities


def _validated_freeze(
    name: str, interpreter: FileIdentity, freeze: dict[str, Any]
) -> dict[str, Any]:
    if (
        freeze.get("interpreter_sha256") != interpreter["sha256"]
        or _sha(_canonical(freeze)) != EXECUTABLE_PINS[name]["package_freeze_sha256"]
    ):
        _fail(f"interpreter freeze pin mismatch: {name}")
    return freeze


def _key_bindings(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    expected = {
        f"{name}.{kind}"
        for name in ("authority", "publication", *_SCOPES)
        for kind in ("private", "public")
    }
    fd = _open_directory(root)
    try:
        if set(os.listdir(fd)) != expected:
            _fail("key root inventory mismatch")
    finally:
        os.close(fd)
    bindings = []
    public_files = []
    key_files = []
    for name in ("authority", "publication", *_SCOPES):
        private = _file(root / f"{name}.private")
        public_path = root / f"{name}.public"
        public_info, public = _read_regular(public_path)
        public_file = _identity(public_path, public_info, public)
        if (
            private["mode"],
            private["st_uid"],
            private["st_gid"],
            private["nlink"],
            private["byte_count"],
        ) != (0o400, os.getuid(), os.getgid(), 1, 32) or (
            public_file["mode"],
            public_file["st_uid"],
            public_file["st_gid"],
            public_file["nlink"],
            public_file["byte_count"],
        ) != (0o400, os.getuid(), os.getgid(), 1, 32):
            _fail("unsafe terminal key")
        digest = _sha(public)
        bindings.append(
            {
                "key_id": digest,
                "public_key_b64": base64.b64encode(public).decode("ascii"),
                "public_key_sha256": digest,
            }
        )
        public_files.append({"name": name, "public": public_file, "key_id": digest})
        key_files.append({"name": name, "private": private, "public": public_file})
    if len({item["key_id"] for item in bindings}) != 5:
        _fail("duplicate keys")
    summary = {
        "schema": "alpha_max_v8_public_key_summary.v1",
        "keys": public_files,
        "publication_key": bindings[1],
    }
    return (
        bindings[0],
        [{"scope": scope, **item} for scope, item in zip(_SCOPES, bindings[2:])],
        summary,
        key_files,
    )


def _revalidate_key_files(key_files: list[dict[str, Any]]) -> None:
    for item in key_files:
        for kind in ("private", "public"):
            identity = item[kind]
            if _file(Path(identity["path"])) != identity:
                _fail(f"terminal key identity drift: {item['name']}.{kind}")


def _load_authenticated(path: Path, expected: FileIdentity, name: str):
    if name in sys.modules:
        _fail(f"authenticated module already registered: {name}")
    info, source = _read_regular(path)
    actual: FileIdentity = {
        "path": str(path),
        "sha256": _sha(source),
        "byte_count": len(source),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
    }
    if actual != expected:
        _fail("authenticated module identity mismatch")
    spec = importlib.machinery.ModuleSpec(name, loader=None, origin=str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__spec__ = spec
    sys.modules[name] = module
    try:
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        if _file(path) != expected:
            _fail("authenticated module identity drift")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_key_creator(path: Path, expected: FileIdentity, policy):
    package_name = "lumina_quant"
    policy_name = f"{package_name}.alpha_max_terminal_policy"
    module_name = "alpha_max_v8_verified_key_creator"
    if package_name in sys.modules or policy_name in sys.modules:
        _fail("authenticated key-creator dependency already registered")
    package_spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    package = types.ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = []
    package.__spec__ = package_spec
    package.alpha_max_terminal_policy = policy
    sys.modules[package_name] = package
    sys.modules[policy_name] = policy
    try:
        module = _load_authenticated(path, expected, module_name)
        creator = getattr(module, "create_keys", None)
        if (
            module.__dict__.get("policy") is not policy
            or not callable(creator)
            or sys.modules.get(package_name) is not package
            or sys.modules.get(policy_name) is not policy
        ):
            _fail("authenticated key-creator dependency drift")
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        policy_entry = sys.modules.pop(policy_name, None)
        package_entry = sys.modules.pop(package_name, None)
    if policy_entry is not policy or package_entry is not package:
        sys.modules.pop(module_name, None)
        _fail("authenticated key-creator dependency drift")
    return creator


_ADMISSION_ISOLATED_SOURCE = r"""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


def canonical(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def pairs(values):
    result = dict(values)
    if len(result) != len(values):
        raise ValueError("duplicate JSON key")
    return result


def invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def read_regular(path):
    path = Path(path)
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise ValueError(f"unsafe regular file: {path}")
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
        raise ValueError(f"unstable regular file: {path}")
    data = b"".join(chunks)
    if len(data) != opened.st_size:
        raise ValueError(f"short regular file read: {path}")
    return opened, data


def file_identity(path):
    info, data = read_regular(path)
    return {
        "path": str(Path(path)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
    }


def symlink_identity(path, expected_target):
    path = Path(path)
    before = os.lstat(path)
    if not stat.S_ISLNK(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
        raise ValueError(f"unsafe execution alias: {path}")
    target = os.readlink(path)
    after = os.lstat(path)
    fields = (
        "st_dev",
        "st_ino",
        "st_uid",
        "st_gid",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        target != expected_target
        or any(getattr(before, field) != getattr(after, field) for field in fields)
        or len(os.fsencode(target)) != after.st_size
    ):
        raise ValueError("execution alias identity mismatch")
    return {
        "path": str(path),
        "target": target,
        "st_dev": after.st_dev,
        "st_ino": after.st_ino,
        "st_uid": after.st_uid,
        "st_gid": after.st_gid,
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": after.st_nlink,
    }


def load(path):
    _, raw = read_regular(path)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict) or canonical(value) != raw:
        raise ValueError(f"noncanonical JSON: {path}")
    return value


def record(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}


def git(root, *arguments):
    if arguments and arguments[0] == "diff":
        arguments = ("diff", "--no-ext-diff", *arguments[1:])
    return subprocess.run(
        (
            "/usr/bin/git",
            "--no-optional-locks",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(root),
            *arguments,
        ),
        check=True,
        capture_output=True,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    ).stdout


def inventory_entry(root, name):
    if not name or name.startswith("/") or ".." in Path(name).parts:
        raise ValueError("unsafe inventory path")
    path = root / name
    before = os.lstat(path)
    if stat.S_ISREG(before.st_mode):
        info, data = read_regular(path)
        kind = "regular"
    elif stat.S_ISLNK(before.st_mode):
        if before.st_nlink != 1 or before.st_uid != os.getuid():
            raise ValueError(f"unsafe inventory symlink: {path}")
        target = os.readlink(path)
        info = os.lstat(path)
        fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink", "st_size")
        if any(getattr(before, field) != getattr(info, field) for field in fields):
            raise ValueError(f"unstable inventory symlink: {path}")
        data = os.fsencode(target)
        if len(data) != info.st_size:
            raise ValueError(f"invalid inventory symlink size: {path}")
        kind = "symlink"
    else:
        raise ValueError(f"unsupported inventory entry: {path}")
    return {
        "path": name,
        "type": kind,
        "mode": stat.S_IMODE(info.st_mode),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def inventory(root, names):
    return canonical([inventory_entry(root, os.fsdecode(name)) for name in names])


def runtime_inventory(root):
    root = Path(root)
    runtime_roots = (
        Path(current_runtime).resolve(),
        Path(accepted_runtime).resolve(),
        Path(base_runtime).resolve(),
    )
    repository_roots = (
        (Path(current, "src").resolve(),)
        if root.resolve() == runtime_roots[0]
        else (Path(accepted, "src").resolve(),)
        if root.resolve() == runtime_roots[1]
        else ()
    )
    allowed = (root.resolve(), *runtime_roots, *repository_roots)
    def within(path):
        return any(path == candidate or candidate in path.parents for candidate in allowed)
    names = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        relative = Path(directory).relative_to(root)
        for name in (*subdirectories, *filenames):
            path = Path(directory) / name
            info = os.lstat(path)
            if stat.S_ISDIR(info.st_mode):
                continue
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                raise ValueError(f"unsupported runtime entry: {path}")
            if stat.S_ISLNK(info.st_mode) and not within(path.resolve(strict=True)):
                raise ValueError(f"runtime symlink escapes inventoried roots: {path}")
            if path.suffix == ".pth":
                for raw in read_regular(path)[1].decode().splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("import "):
                        if not __import__("re").fullmatch(r"import [A-Za-z_][A-Za-z0-9_]*", line):
                            raise ValueError(f"unsafe runtime import hook: {path}")
                        if not any(root.rglob(line[7:] + ".py")):
                            raise ValueError(f"runtime import hook module is not inventoried: {path}")
                    else:
                        target = Path(line)
                        target = (path.parent / target if not target.is_absolute() else target).resolve(
                            strict=True
                        )
                        if not within(target):
                            raise ValueError(f"runtime .pth target escapes inventoried roots: {path}")
            names.append(os.fsencode(str(relative / name)))
    return inventory(root, sorted(names))


(
    approval_path,
    payload_path,
    credential_name,
    credential_source,
    credential_json,
    current,
    accepted,
    execution_alias,
    recovery_root,
    current_runtime,
    accepted_runtime,
    base_runtime,
    accepted_commit,
    owner_exclude,
    *ignored_source_paths,
) = sys.argv[1:]
payload = load(payload_path)
approval = load(approval_path)
current_root = Path(current)
source_paths = ("--", ".", owner_exclude)
actual = {
    "head": git(current_root, "rev-parse", "HEAD").decode().strip(),
    "porcelain": record(git(current_root, "status", "--porcelain=v1", "-z", *source_paths)),
    "commit_overlay": record(
        git(current_root, "diff", "--binary", f"{accepted_commit}..HEAD", *source_paths)
    ),
    "worktree_overlay": record(
        git(current_root, "diff", "--binary", "HEAD", *source_paths)
    ),
    "source_inventory": record(
        inventory(
            current_root,
            git(
                current_root,
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                *source_paths,
            ).split(b"\0")[:-1],
        )
    ),
    "ignored_source_inventory": record(
        inventory(
            current_root,
            git(
                current_root,
                "ls-files",
                "-z",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                *ignored_source_paths,
            ).split(b"\0")[:-1],
        )
    ),
}
for name, value in actual.items():
    if approval.get(name) != value:
        raise ValueError(f"approval source state mismatch: {name}")
if approval.get("accepted_alpha_commit") != accepted_commit:
    raise ValueError("approval accepted commit mismatch")
accepted_root = Path(accepted)
accepted_actual = {
    "root": accepted,
    "head": git(accepted_root, "rev-parse", "HEAD").decode().strip(),
    "porcelain": record(
        git(accepted_root, "status", "--porcelain=v1", "-z", *source_paths)
    ),
    "source_inventory": record(
        inventory(
            accepted_root,
            git(
                accepted_root,
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                *source_paths,
            ).split(b"\0")[:-1],
        )
    ),
    "ignored_source_inventory": record(
        inventory(
            accepted_root,
            git(
                accepted_root,
                "ls-files",
                "-z",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                *ignored_source_paths,
            ).split(b"\0")[:-1],
        )
    ),
}
if (
    accepted_actual["head"] != accepted_commit
    or approval.get("accepted_source_state") != accepted_actual
    or payload.get("accepted_source_state") != accepted_actual
):
    raise ValueError("signed accepted source state mismatch")
observed_alias = symlink_identity(execution_alias, recovery_root)
if approval.get("execution_alias") != observed_alias or payload.get(
    "execution_alias"
) != observed_alias:
    raise ValueError("signed execution alias mismatch")
runtime_roots = {
    "current_python": Path(current_runtime),
    "accepted_python": Path(accepted_runtime),
    "base_python": Path(base_runtime),
}
for name, root in runtime_roots.items():
    binding = payload["runtime_bindings"][name]
    if binding["root"] != str(root) or file_identity(binding["interpreter"]["path"]) != binding[
        "interpreter"
    ]:
        raise ValueError(f"runtime binding mismatch: {name}")
    observed = record(runtime_inventory(root))
    if observed != approval["runtime_inventories"][name] or observed != payload[
        "runtime_inventories"
    ][name]:
        raise ValueError(f"runtime inventory mismatch: {name}")
if payload["source_inventory"] != approval["source_inventory"] or payload[
    "ignored_source_inventory"
] != approval["ignored_source_inventory"]:
    raise ValueError("signed source inventory mismatch")
expected_credential = json.loads(
    credential_json, object_pairs_hook=pairs, parse_constant=invalid_constant
)
if (
    expected_credential["name"] != credential_name
    or expected_credential["source"]["path"] != credential_source
    or file_identity(credential_source) != expected_credential["source"]
):
    raise ValueError("credential source identity mismatch")
credential_path = Path(os.environ["CREDENTIALS_DIRECTORY"]) / credential_name
credential_info, credential_bytes = read_regular(credential_path)
if (
    credential_info.st_size != expected_credential["source"]["byte_count"]
    or hashlib.sha256(credential_bytes).hexdigest()
    != expected_credential["source"]["sha256"]
):
    raise ValueError("delivered credential mismatch")
"""

_ADMISSION_PRESTART_SOURCE = r"""
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile


def canonical(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def pairs(values):
    result = dict(values)
    if len(result) != len(values):
        raise ValueError("duplicate JSON key")
    return result


def invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def read_regular(path):
    path = Path(path)
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise ValueError(f"unsafe regular file: {path}")
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
        raise ValueError(f"unstable regular file: {path}")
    data = b"".join(chunks)
    if len(data) != opened.st_size:
        raise ValueError(f"short regular file read: {path}")
    return opened, data


def file_identity(path):
    info, data = read_regular(path)
    return {
        "path": str(Path(path)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "st_uid": info.st_uid,
        "st_gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
    }


def load(path):
    _, raw = read_regular(path)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict) or canonical(value) != raw:
        raise ValueError(f"noncanonical JSON: {path}")
    return value, raw


(
    approval_path,
    receipt_path,
    key_root,
    unit_name,
    unit_file,
    authority_public_b64,
    current,
    accepted,
    execution_alias,
    recovery_root,
    current_runtime,
    accepted_runtime,
    base_runtime,
    owner_exclude,
    accepted_commit,
    isolated_source_b64,
    *ignored_source_paths,
) = sys.argv[1:]
receipt = Path(receipt_path)
payload_path = receipt.with_suffix(".payload.json")
envelope, _ = load(receipt)
detached_payload, detached_bytes = load(payload_path)
payload = envelope["payload"]
if payload != detached_payload or canonical(payload) != detached_bytes:
    raise ValueError("detached launch-admission payload mismatch")
public = base64.b64decode(authority_public_b64, validate=True)
signature = base64.b64decode(envelope["signature_b64"], validate=True)
if (
    len(public) != 32
    or len(signature) != 64
    or envelope["schema"] != "alpha_max_v8_signed_launch_admission.v1"
    or envelope["authority_key_id"] != hashlib.sha256(public).hexdigest()
):
    raise ValueError("launch-admission envelope mismatch")
der = b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00" + public
with tempfile.TemporaryDirectory() as directory:
    public_path = Path(directory) / "authority.der"
    signature_path = Path(directory) / "admission.signature"
    public_path.write_bytes(der)
    signature_path.write_bytes(signature)
    subprocess.run(
        (
            "/usr/bin/openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            str(public_path),
            "-keyform",
            "DER",
            "-rawin",
            "-in",
            str(payload_path),
            "-sigfile",
            str(signature_path),
        ),
        check=True,
        capture_output=True,
    )
scope = receipt.name.removeprefix("launch-admission-").removesuffix(".json")
if payload["requested_scope"] != scope:
    raise ValueError("launch-admission scope mismatch")
if (
    file_identity(approval_path) != payload["approval"]
    or file_identity(payload["complete"]["path"]) != payload["complete"]
    or file_identity(payload["manifest"]["path"]) != payload["manifest"]
):
    raise ValueError("launch-admission control identity mismatch")
for artifact in payload.get("control_artifacts", []):
    if not isinstance(artifact, dict) or file_identity(artifact.get("path", "")) != artifact:
        raise ValueError("launch-admission signed control artifact mismatch")
matching = [
    item
    for item in payload["topology"]["units"]
    if item["production_unit"] == unit_name and item["unit"]["path"] == unit_file
]
if len(matching) != 1:
    raise ValueError("launch-admission production unit mismatch")
entry = matching[0]
if entry["scope"] != scope or Path(unit_file).name != unit_name:
    raise ValueError("launch-admission topology mismatch")
if file_identity(unit_file) != entry["unit"]:
    raise ValueError("launch-admission unit identity drift")
credential = entry["credential"]
credential_source = credential["source"]["path"]
if Path(credential_source).parent != Path(key_root) or credential["target"] != (
    f"%d/{credential['name']}"
):
    raise ValueError("launch-admission credential topology mismatch")
transient_name = (
    unit_name.removesuffix(".service")
    + f"-admission-verify-{os.getpid()}.service"
)
inner_code = (
    "import base64,sys;"
    "exec(compile(base64.b64decode(sys.argv[1]),"
    "'<alpha-max-admission-isolated>','exec'))"
)
command = [
    "/usr/bin/systemd-run",
    "--user",
    "--wait",
    "--collect",
    "--quiet",
    f"--unit={transient_name}",
    "--property=Type=exec",
    "--property=UMask=0077",
    "--property=NoNewPrivileges=yes",
    "--property=PrivateTmp=yes",
    "--property=PrivateDevices=yes",
    "--property=ProtectSystem=strict",
    "--property=ProtectHome=tmpfs",
    "--property=ProtectKernelTunables=yes",
    "--property=ProtectKernelModules=yes",
    "--property=ProtectControlGroups=yes",
    "--property=RestrictAddressFamilies=AF_UNIX",
    "--property=IPAddressDeny=any",
    "--property=MemoryHigh=536870912",
    "--property=MemoryMax=1073741824",
    "--property=MemorySwapMax=0",
    "--property=OOMPolicy=kill",
    "--property=RuntimeMaxSec=1800s",
    "--property=WorkingDirectory=/",
    "--property=Environment=HOME=/tmp",
    "--property=Environment=PYTHONDONTWRITEBYTECODE=1",
]
for path in (
    current,
    accepted,
    current_runtime,
    accepted_runtime,
    base_runtime,
    str(Path(unit_file).parent.parent),
    approval_path,
    credential_source,
):
    command.append(f"--property=BindReadOnlyPaths={path}")
command.extend(
    (
        f"--property=LoadCredential={credential['name']}:{credential_source}",
        "/usr/bin/python3",
        "-I",
        "-c",
        inner_code,
        isolated_source_b64,
        approval_path,
        str(payload_path),
        credential["name"],
        credential_source,
        canonical(credential).decode().strip(),
        current,
        accepted,
        execution_alias,
        recovery_root,
        current_runtime,
        accepted_runtime,
        base_runtime,
        accepted_commit,
        owner_exclude,
        *ignored_source_paths,
    )
)
subprocess.run(command, check=True)
if (
    file_identity(unit_file) != entry["unit"]
    or file_identity(approval_path) != payload["approval"]
    or load(payload_path)[1] != detached_bytes
):
    raise ValueError("launch-admission post-verification identity drift")
property_names = (
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
show = ["/usr/bin/systemctl", "--user", "show", unit_name]
for name in property_names:
    show.extend(("-p", name))
live = {}
for line in subprocess.run(show, check=True, capture_output=True, text=True).stdout.splitlines():
    name, separator, value = line.partition("=")
    if separator:
        live[name] = value
if live != payload["production_properties"][unit_name]:
    raise ValueError("launch-admission production property drift")
"""

_ADMISSION_ISOLATED_B64 = base64.b64encode(_ADMISSION_ISOLATED_SOURCE.encode()).decode()
_ADMISSION_PRESTART_B64 = base64.b64encode(_ADMISSION_PRESTART_SOURCE.encode()).decode()
_ADMISSION_PRESTART_CODE = (
    "import base64;exec(compile(base64.b64decode("
    + repr(_ADMISSION_PRESTART_B64)
    + "),'<alpha-max-admission-prestart>','exec'))"
)


def _admission_prestart(
    interpreter: str,
    approval: Path,
    receipt: Path,
    key_root: Path,
    unit_name: str,
    unit_file: Path,
    authority_public_b64: str,
) -> list[str]:
    return [
        "/usr/bin/python3",
        "-I",
        "-c",
        _ADMISSION_PRESTART_CODE,
        str(approval),
        str(receipt),
        str(key_root),
        unit_name,
        str(unit_file),
        authority_public_b64,
        CURRENT,
        ACCEPTED,
        str(EXECUTION_ALIAS_ROOT),
        str(RECOVERY_ROOT),
        str(Path(EXECUTABLE_PINS["current_python"]["path"]).parent.parent),
        str(Path(EXECUTABLE_PINS["accepted_python"]["path"]).parent.parent),
        str(Path(EXECUTABLE_PINS["base_python"]["path"]).parent.parent),
        _OWNER_SESSION_RUNTIME_EXCLUDE,
        ACCEPTED_COMMIT,
        _ADMISSION_ISOLATED_B64,
        *_IGNORED_SOURCE_PATHS,
    ]


def _command_read_paths(
    commands: list[list[str]],
    *,
    write_paths: list[str],
    inaccessible_paths: list[str],
) -> list[str]:
    """Expose only exact absolute command inputs hidden by ProtectHome=tmpfs."""
    excluded_roots = [*write_paths, *inaccessible_paths, *_FORBIDDEN_ROOTS]
    result: list[str] = []
    for command in commands:
        if not isinstance(command, list) or not command:
            _fail("invalid policy-derived command")
        for argument in command:
            if not isinstance(argument, str):
                _fail("invalid policy-derived command argument")
            if not argument.startswith("/"):
                continue
            if any(argument == root or argument.startswith(root + "/") for root in excluded_roots):
                continue
            if argument in {CURRENT, ACCEPTED}:
                _fail("policy-derived command exposes a broad repository root")
            _inventory_absolute(argument)
            result.append(argument)
    return list(dict.fromkeys(result))


def _write_bind_paths(write_paths: list[str]) -> list[str]:
    bindings: list[str] = []
    seen: set[Path] = set()
    for raw_destination in write_paths:
        destination = _absolute(raw_destination)
        try:
            relative = destination.relative_to(RECOVERY_ROOT)
        except ValueError:
            _fail("systemd writable destination escapes the recovery root")
        if relative == Path(".") or destination in seen:
            _fail("systemd writable destinations must be distinct recovery subpaths")
        seen.add(destination)
        source = EXECUTION_ALIAS_ROOT / relative
        bindings.append(f"{source}:{destination}")
    return bindings


def _unit(
    description: str,
    argv: list[str],
    env: dict[str, str],
    limits: dict[str, int],
    stop: list[str] | None,
    *,
    read_paths: list[str],
    write_paths: list[str],
    inaccessible_paths: list[str],
    load_credential: str,
    observer: bool = False,
    prestart: list[str] | None = None,
) -> dict[str, Any]:
    argv_read_paths = _command_read_paths(
        [argv, *([stop] if stop is not None else [])],
        write_paths=write_paths,
        inaccessible_paths=inaccessible_paths,
    )
    if prestart is not None:
        if len(prestart) < 5 or not prestart[4].startswith("/"):
            _fail("production prestart lacks an absolute approval")
        argv_read_paths.extend(
            (
                prestart[4],
                str(Path(EXECUTABLE_PINS["current_python"]["path"]).parent.parent),
                str(Path(EXECUTABLE_PINS["accepted_python"]["path"]).parent.parent),
                str(Path(EXECUTABLE_PINS["base_python"]["path"]).parent.parent),
                str(Path(CURRENT) / "src"),
                str(Path(ACCEPTED) / "src"),
            )
        )
    read_paths = list(dict.fromkeys([*read_paths, *argv_read_paths]))
    bind_paths = _write_bind_paths(write_paths)
    service = {
        "UMask": "0077",
        "Type": "exec",
        "ExecStart": (
            _wrap_execstart(
                argv,
                write_paths,
                receipt=prestart[5],
                unit_name=prestart[7],
                authority_public_b64=prestart[9],
            )
            if prestart is not None
            else argv
        ),
        "Environment": [f"{k}={v}" for k, v in env.items()],
        "WorkingDirectory": "/",
        "NoNewPrivileges": True,
        "PrivateTmp": True,
        "PrivateDevices": True,
        "ProtectSystem": "strict",
        "ProtectHome": "tmpfs",
        "BindReadOnlyPaths": read_paths,
        "InaccessiblePaths": [f"-{path}" for path in inaccessible_paths],
        "LoadCredential": [load_credential],
        "BindPaths": bind_paths,
        "ProtectKernelTunables": True,
        "ProtectKernelModules": True,
        "ProtectControlGroups": True,
        "RestrictAddressFamilies": (
            ["AF_UNIX", "AF_INET", "AF_INET6"] if observer else ["AF_UNIX"]
        ),
        "MemoryHigh": limits["high"],
        "MemoryMax": limits["max"],
        "MemorySwapMax": limits["swap"],
        "OOMPolicy": "kill",
        "TimeoutStartSec": "1800s",
        "TimeoutStopSec": "120s",
    }
    if not bind_paths:
        service.pop("BindPaths")
    if prestart is not None:
        service["ExecStartPre"] = prestart
    if not observer:
        service["IPAddressDeny"] = "any"
    if stop is not None:
        service["ExecStopPost"] = (
            _wrap_execstart(
                stop,
                write_paths,
                receipt=prestart[5],
                unit_name=prestart[7],
                authority_public_b64=prestart[9],
            )
            if prestart is not None
            else stop
        )
    return {
        "Description": description,
        "After": ["network-online.target"],
        "Wants": ["network-online.target"],
        "Service": service,
    }


def _load_policy(path: Path, expected: FileIdentity):
    name = "alpha_max_v8_verified_policy"
    module = _load_authenticated(path, expected, name)
    try:
        if (
            tuple(module._SCOPES) != _SCOPES
            or tuple(module._FILE_ROLES) != _FILE_ROLES
            or tuple(module._FORBIDDEN_ROOTS) != _FORBIDDEN_ROOTS
        ):
            _fail("authenticated policy constants mismatch")
    except BaseException:
        if sys.modules.get(name) is module:
            sys.modules.pop(name)
        raise
    return module


def _write_bytes_new(path: Path, data: bytes) -> dict[str, Any]:
    """Exclusively create, durably write, and return a byte artifact."""
    path = _absolute(str(path))
    parent_fd = _open_directory(path.parent)
    fd = -1
    try:
        fd = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
        _write_all(fd, data)
        os.fsync(fd)
        written = os.fstat(fd)
        identity = _identity(path, written, data)
        os.close(fd)
        fd = -1
        os.fsync(parent_fd)
        if _file(path) != identity:
            _fail("created artifact identity drift")
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        if fd >= 0:
            try:
                os.close(fd)
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            os.close(parent_fd)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "artifact creation failed and descriptor cleanup failed",
                [primary, *cleanup_errors],
            ) from primary
        raise
    os.close(parent_fd)
    return identity


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    if not source or not destination or "/" in source or "/" in destination:
        _fail("invalid atomic publication leaf")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("renameat2 is required for no-replace publication") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            directory_fd,
            os.fsencode(source),
            directory_fd,
            os.fsencode(destination),
            1,
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _publish_complete(control: Path, value: dict[str, Any]) -> FileIdentity:
    pending = control / ".COMPLETE.json.pending"
    complete = control / "COMPLETE.json"
    identity = _write_new(pending, value)
    if _file(pending) != identity:
        _fail("pending completion identity drift")
    directory_fd = _open_directory(control)
    try:
        _rename_noreplace(directory_fd, pending.name, complete.name)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    published = {**identity, "path": str(complete)}
    if _file(complete) != published:
        _fail("published completion identity drift")
    return published


_UNIT_RENDER_ORDER = ("Description", "After", "Wants")
_SERVICE_RENDER_ORDER = (
    "Type",
    "UMask",
    "ExecStart",
    "ExecStartPre",
    "ExecStopPost",
    "Environment",
    "WorkingDirectory",
    "NoNewPrivileges",
    "PrivateTmp",
    "PrivateDevices",
    "ProtectSystem",
    "ProtectHome",
    "BindReadOnlyPaths",
    "InaccessiblePaths",
    "LoadCredential",
    "BindPaths",
    "ProtectKernelTunables",
    "ProtectKernelModules",
    "ProtectControlGroups",
    "RestrictAddressFamilies",
    "IPAddressDeny",
    "MemoryHigh",
    "MemoryMax",
    "MemorySwapMax",
    "OOMPolicy",
    "TimeoutStartSec",
    "TimeoutStopSec",
)


def _systemd_word(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail("systemd unit contains an invalid word")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _render_systemd_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if name == "UMask":
        if value != "0077":
            _fail("systemd UMask must be 0077")
        return value
    if isinstance(value, str):
        _systemd_word(value)
        if name != "Description" and any(character.isspace() for character in value):
            _fail(f"systemd directive requires one atom: {name}")
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        if name in {"ExecStart", "ExecStartPre", "ExecStopPost", "Environment"}:
            return " ".join(_systemd_word(item) for item in value)
        if any(any(character.isspace() for character in item) for item in value):
            _fail(f"systemd directive contains a non-atomic list value: {name}")
        return " ".join(value)
    _fail(f"unsupported systemd directive value: {name}")


def _render_systemd_unit(unit: dict[str, Any]) -> bytes:
    """Render the validated production unit without a shell or a second contract."""
    _validate_unit(unit)
    if set(unit) != set(_UNIT_RENDER_ORDER) | {"Service"}:
        _fail("unsupported systemd unit directive")
    service = unit["Service"]
    if set(service) - set(_SERVICE_RENDER_ORDER):
        _fail("unsupported systemd service directive")
    lines = ["[Unit]"]
    for name in _UNIT_RENDER_ORDER:
        lines.append(f"{name}={_render_systemd_value(name, unit[name])}")
    lines.extend(("", "[Service]"))
    for name in _SERVICE_RENDER_ORDER:
        if name in service:
            lines.append(f"{name}={_render_systemd_value(name, service[name])}")
    return ("\n".join(lines) + "\n").encode()


def _credential_binding(unit: dict[str, Any], key_files: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind one stable source credential to the `%d` target used by ExecStart."""
    _validate_unit(unit)
    service = unit["Service"]
    credential = service["LoadCredential"][0]
    name, _, source_value = credential.partition(":")
    source = Path(source_value)
    identities = {
        item[kind]["path"]: item[kind] for item in key_files for kind in ("private", "public")
    }
    expected = identities.get(str(source))
    if expected is None:
        _fail("systemd credential source is not an approved terminal key")
    actual = _file(source)
    if actual != expected:
        _fail("systemd credential source identity drift")
    if (
        actual["st_uid"],
        actual["st_gid"],
        actual["mode"],
        actual["nlink"],
    ) != (os.getuid(), os.getgid(), 0o400, 1):
        _fail("systemd credential source permissions are unsafe")
    target = f"%d/{name}"
    if service["ExecStart"].count(target) != 1:
        _fail("systemd credential target is not exact")
    return {"name": name, "source": actual, "target": target}


_SERVICE_DIRECTIVES = frozenset(
    {
        "Type",
        "UMask",
        "ExecStart",
        "ExecStartPre",
        "Environment",
        "WorkingDirectory",
        "NoNewPrivileges",
        "PrivateTmp",
        "PrivateDevices",
        "ProtectSystem",
        "ProtectHome",
        "BindReadOnlyPaths",
        "InaccessiblePaths",
        "LoadCredential",
        "BindPaths",
        "ProtectKernelTunables",
        "ProtectKernelModules",
        "ProtectControlGroups",
        "RestrictAddressFamilies",
        "MemoryHigh",
        "MemoryMax",
        "MemorySwapMax",
        "OOMPolicy",
        "TimeoutStartSec",
        "TimeoutStopSec",
        "ExecStopPost",
        "IPAddressDeny",
    }
)
_UNIT_DIRECTIVES = frozenset({"Description", "After", "Wants", "Service"})


def _validate_unit(unit: dict[str, Any]) -> None:
    if set(unit) != _UNIT_DIRECTIVES or not isinstance(unit.get("Service"), dict):
        _fail("unsupported systemd unit directive")
    unknown = set(unit["Service"]) - _SERVICE_DIRECTIVES
    if unknown:
        _fail(f"unsupported systemd service directive: {sorted(unknown)!r}")
    service = unit["Service"]
    if service.get("UMask") != "0077":
        _fail("systemd UMask must be 0077")
    if not isinstance(service.get("BindReadOnlyPaths"), list) or not isinstance(
        service.get("InaccessiblePaths"), list
    ):
        _fail("invalid systemd path directives")
    bind_paths = service.get("BindPaths", [])
    if (
        not isinstance(bind_paths, list)
        or ("BindPaths" in service and not bind_paths)
        or any(not isinstance(path, str) for path in bind_paths)
    ):
        _fail("invalid systemd writable bind paths")
    load_credential = service["LoadCredential"]
    if isinstance(load_credential, str):
        credentials = [load_credential]
    elif isinstance(load_credential, list):
        credentials = load_credential
    else:
        _fail("systemd LoadCredential must be a string or list")
    if len(credentials) != 1 or not isinstance(credentials[0], str):
        _fail("systemd unit must load exactly one credential")
    name, separator, source = credentials[0].partition(":")
    if not separator or not name or not source or ":" in source:
        _fail("invalid systemd LoadCredential")
    if any(
        not isinstance(path, str)
        for path in (*service["BindReadOnlyPaths"], *service["InaccessiblePaths"])
    ):
        _fail("invalid systemd path directives")
    read_paths = set(service["BindReadOnlyPaths"])
    if len(read_paths) != len(service["BindReadOnlyPaths"]):
        _fail("duplicate systemd read-only bind")
    write_paths: set[str] = set()
    bind_sources: set[str] = set()
    for binding in bind_paths:
        if binding.count(":") != 1:
            _fail("systemd writable bind must map one source to one destination")
        raw_source, raw_destination = binding.split(":")
        source_path = Path(raw_source)
        destination = Path(raw_destination)
        if (
            not source_path.is_absolute()
            or not destination.is_absolute()
            or str(source_path) != raw_source
            or str(destination) != raw_destination
            or os.path.normpath(raw_source) != raw_source
            or os.path.normpath(raw_destination) != raw_destination
            or any(component in {".", ".."} for component in source_path.parts)
            or any(component in {".", ".."} for component in destination.parts)
        ):
            _fail("systemd writable bind paths must be canonically absolute")
        try:
            source_relative = source_path.relative_to(EXECUTION_ALIAS_ROOT)
            destination_relative = destination.relative_to(RECOVERY_ROOT)
        except ValueError:
            _fail("systemd writable bind escapes its execution capability")
        if (
            source_relative == Path(".")
            or source_relative != destination_relative
            or raw_source in bind_sources
            or raw_destination in write_paths
        ):
            _fail("systemd writable bind mapping mismatch")
        bind_sources.add(raw_source)
        write_paths.add(raw_destination)
    raw_inaccessible_paths = set(service["InaccessiblePaths"])
    if len(raw_inaccessible_paths) != 1 or any(
        not path.startswith("-/") for path in raw_inaccessible_paths
    ):
        _fail("systemd inaccessible paths must contain only the optional key root")
    inaccessible_paths = {path[1:] for path in raw_inaccessible_paths}
    key_root = next(iter(inaccessible_paths))
    if (
        service.get("ProtectHome") != "tmpfs"
        or service.get("WorkingDirectory") != "/"
        or any(not path.startswith("/") for path in read_paths | write_paths)
        or any(
            path in {"/home", str(Path.home()), CURRENT, ACCEPTED}
            or any(path == root or root.startswith(path + "/") for root in (CURRENT, ACCEPTED))
            or any(
                path == root or path.startswith(root + "/") or root.startswith(path + "/")
                for root in _FORBIDDEN_ROOTS
            )
            for path in read_paths | write_paths
        )
    ):
        _fail("systemd unit exposes a broad home path")
    source_path = Path(source)
    if not source_path.is_absolute() or source_path.parent != Path(key_root):
        _fail("systemd credential source must be directly under the inaccessible key root")
    if source_path.name != name:
        _fail("systemd credential name does not match its source")
    argument = _credential_argument(name)
    argv = service["ExecStart"]
    if (
        not isinstance(argv, list)
        or argv.count(argument) != 1
        or argv[argv.index(argument) + 1 : argv.index(argument) + 2] != [f"%d/{name}"]
    ):
        _fail("systemd credential does not match unit role argv")
    if key_root in read_paths or key_root in write_paths:
        _fail("systemd key root must not be readable or writable")
    if any(
        left == right or left.startswith(right + "/") or right.startswith(left + "/")
        for left in read_paths
        for right in write_paths
    ):
        _fail("systemd read-only and writable binds overlap")
    if any(
        left == right or left.startswith(right + "/") or right.startswith(left + "/")
        for left in read_paths | write_paths
        for right in inaccessible_paths
    ):
        _fail("systemd accessible and inaccessible paths overlap")
    production = any(
        "run_alpha_max_terminal_" in argument or "monitor_alpha_max_v8_resources.py" in argument
        for argument in argv
    )
    prestart = service.get("ExecStartPre")
    if production and (
        not isinstance(prestart, list)
        or len(prestart) < 10
        or not all(isinstance(item, str) for item in prestart)
        or prestart[3] != _ADMISSION_PRESTART_CODE
        or not any(
            re.fullmatch(
                r".*/launch-admission-(acquisition|phase_preparation|one_touch)\.json", item
            )
            for item in prestart
        )
    ):
        _fail("production unit lacks an exact launch-admission prestart contract")
    if name == "acquisition.private":
        if (
            service.get("RestrictAddressFamilies") != ["AF_UNIX", "AF_INET", "AF_INET6"]
            or "IPAddressDeny" in service
        ):
            _fail("acquisition observer must allow only Unix and Internet socket families")
    elif (
        service.get("RestrictAddressFamilies") != ["AF_UNIX"]
        or service.get("IPAddressDeny") != "any"
    ):
        _fail("non-acquisition systemd unit must deny all network families")
    if any(not path.startswith("/") for path in write_paths):
        _fail("systemd writable paths must be absolute")


def _repository_evidence(
    root: Path, prefix: str, control: Path
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    raw = {
        "porcelain": _source_git(root, "status", "--porcelain=v1", "-z"),
        "commit_overlay": _source_git(root, "diff", "--binary", f"{ACCEPTED_COMMIT}..HEAD"),
        "worktree_overlay": _source_git(root, "diff", "--binary", "HEAD"),
        "source_inventory": _inventory(root),
    }
    evidence = {
        key: _write_bytes_new(control / f"{prefix}-{key}.bin", value) for key, value in raw.items()
    }
    return head, raw, evidence


def _quarantine(
    control: Path | None, roots: dict[str, dict[str, Any]], error: BaseException
) -> None:
    if control is None or not control.exists():
        return
    owned = {}
    for name, identity in roots.items():
        info = _directory(Path(identity["path"]), private=True)
        if (info["st_dev"], info["st_ino"]) == (identity["st_dev"], identity["st_ino"]):
            owned[name] = info
    complete_fd = _open_directory(control)
    try:
        try:
            os.stat("COMPLETE.json", dir_fd=complete_fd, follow_symlinks=False)
        except FileNotFoundError:
            complete_absent = True
        else:
            complete_absent = False
    finally:
        os.close(complete_fd)
    failed = control / "FAILED.json"
    if not failed.exists():
        _write_new(
            failed,
            {
                "schema": "alpha_max_v8_acquisition_failed.v1",
                "error": type(error).__name__,
                "created_roots": owned,
                "complete_absent": complete_absent,
            },
        )


def _quarantine_or_reraise(
    control: Path | None, roots: dict[str, dict[str, Any]], primary: BaseException
) -> None:
    try:
        _quarantine(control, roots, primary)
    except BaseException as quarantine_error:
        raise BaseExceptionGroup(
            "build failed and quarantine failed", [primary, quarantine_error]
        ) from primary
    raise primary


def create_approval(args: argparse.Namespace) -> dict[str, str]:
    """Create only the reviewed source-bound approval; never create a control artifact."""
    paths = {
        name: _absolute(getattr(args, name))
        for name in (
            "control_root",
            "key_root",
            "evidence_root",
            "telemetry_root",
            "output_parent",
            "current_approval",
        )
    }
    paths["stage_results_parent"] = RECOVERY_ROOT / f"g056v8-stage-results-{args.run_id}"
    request_ids = {
        "acquisition": args.request_id,
        "phase_preparation": args.phase_request_id,
        "one_touch": args.one_touch_request_id,
    }
    expected = {
        "control_root": f"g056v8-controls-{args.run_id}",
        "key_root": f"g056v8-keys-{args.run_id}",
        "evidence_root": f"g056v8-acquisition-evidence-{args.run_id}",
        "telemetry_root": f"g056v8-telemetry-{args.run_id}",
        "output_parent": f"g056v8-acquisition-output-{args.run_id}",
        "stage_results_parent": f"g056v8-stage-results-{args.run_id}",
    }
    if (
        paths["current_approval"] != RECOVERY_ROOT / CURRENT_APPROVAL_LEAF
        or args.run_id != RUN_ID
        or request_ids
        != {
            "acquisition": ACQUISITION_REQUEST_ID,
            "phase_preparation": PHASE_PREPARATION_REQUEST_ID,
            "one_touch": ONE_TOUCH_REQUEST_ID,
        }
        or any(
            paths[name].parent != RECOVERY_ROOT or paths[name].name != leaf
            for name, leaf in expected.items()
        )
    ):
        _fail("fresh recovery approval identity mismatch")
    if not _only_owner_session_runtime_changes(Path(CURRENT)):
        _fail("current recovery source contains unapproved dirty or untracked files")
    approval = _create_current_approval(
        paths["current_approval"],
        root=Path(CURRENT),
        run_id=args.run_id,
        request_ids=request_ids,
        absent_paths={name: paths[name] for name in expected},
    )
    return {"approval": approval["schema"], "path": str(paths["current_approval"])}


def build(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        name: _absolute(getattr(args, name))
        for name in (
            "control_root",
            "key_root",
            "evidence_root",
            "telemetry_root",
            "output_parent",
            "telemetry_script",
            "current_approval",
            "current_python",
            "accepted_python",
        )
    }
    paths["stage_results_parent"] = RECOVERY_ROOT / f"g056v8-stage-results-{args.run_id}"
    if paths["current_approval"] != RECOVERY_ROOT / CURRENT_APPROVAL_LEAF:
        _fail("approval path mismatch")
    request_ids = {
        "acquisition": args.request_id,
        "phase_preparation": args.phase_request_id,
        "one_touch": args.one_touch_request_id,
    }
    if (
        args.run_id != RUN_ID
        or request_ids
        != {
            "acquisition": ACQUISITION_REQUEST_ID,
            "phase_preparation": PHASE_PREPARATION_REQUEST_ID,
            "one_touch": ONE_TOUCH_REQUEST_ID,
        }
        or len(set(request_ids.values())) != len(_SCOPES)
        or any(not HEX.fullmatch(value) for value in (args.run_id, *request_ids.values()))
    ):
        _fail("fresh recovery identity mismatch")
    recovery = paths["output_parent"].parent
    if recovery != RECOVERY_ROOT:
        _fail("unexpected recovery root")
    expected = {
        "control_root": f"g056v8-controls-{args.run_id}",
        "key_root": f"g056v8-keys-{args.run_id}",
        "evidence_root": f"g056v8-acquisition-evidence-{args.run_id}",
        "telemetry_root": f"g056v8-telemetry-{args.run_id}",
        "output_parent": f"g056v8-acquisition-output-{args.run_id}",
        "stage_results_parent": f"g056v8-stage-results-{args.run_id}",
    }
    if (
        any(
            paths[name].parent != recovery or paths[name].name != leaf
            for name, leaf in expected.items()
        )
        or len({paths[name] for name in expected}) != 6
    ):
        _fail("invalid v8 root topology")
    paths["admission_root"] = paths["control_root"] / "admissions"
    _directory(recovery)
    approval = _load_current_approval(
        paths["current_approval"],
        run_id=args.run_id,
        request_ids=request_ids,
        absent_paths={name: paths[name] for name in expected},
    )
    current_head = approval["head"]
    if _git(Path(ACCEPTED), "rev-parse", "HEAD").decode().strip() != ACCEPTED_COMMIT or _git(
        Path(ACCEPTED), "status", "--porcelain=v1", "-z"
    ):
        _fail("accepted checkout is not exact and clean")
    alignment = _file(Path(ALIGNMENT))
    if alignment["sha256"] != ALIGNMENT_SHA256:
        _fail("alignment digest mismatch")
    files = [
        {"role": role, "file": _file(Path(base) / relative)}
        for role, (base, relative, pin) in ROLE_PINS.items()
    ]
    for item in files:
        if ROLE_PINS[item["role"]][2] and item["file"]["sha256"] != ROLE_PINS[item["role"]][2]:
            _fail(f"role pin mismatch: {item['role']}")
    executable_inputs = _preflight_executables(paths)
    freezes_data = [
        (
            freeze_role,
            executable_inputs[pin_name],
            _validated_freeze(
                pin_name,
                executable_inputs[pin_name],
                _freeze(executable_inputs[pin_name]),
            ),
        )
        for freeze_role, pin_name in (
            ("current_python", "current_python"),
            ("accepted_alpha_python", "accepted_python"),
        )
    ]
    telemetry_preflight = executable_inputs["telemetry_script"]
    role_files = {item["role"]: item["file"] for item in files}
    policy = _load_policy(Path(role_files["policy_module"]["path"]), role_files["policy_module"])
    creator = _load_key_creator(
        Path(role_files["key_creator"]["path"]), role_files["key_creator"], policy
    )
    created: dict[str, dict[str, Any]] = {}
    try:
        for name in (
            "control_root",
            "key_root",
            "evidence_root",
            "telemetry_root",
            "output_parent",
        ):
            _create_root(paths[name])
            created[name] = _directory(paths[name], private=True)
        _create_root(paths["admission_root"])
        source, report = paths["output_parent"] / "source", paths["output_parent"] / "report"
        source_absence, report_absence = _absent(source), _absent(report)
        creator(paths["key_root"])
        authority, observers, key_summary, key_files = _key_bindings(paths["key_root"])
        if args.run_id in {
            authority["key_id"],
            *(item["key_id"] for item in observers),
        } or args.request_id in {authority["key_id"], *(item["key_id"] for item in observers)}:
            _fail("run and request IDs must differ from generated key IDs")
        key_summary_file = _write_new(
            paths["control_root"] / "public-key-summary.json", key_summary
        )
        policy_value = {
            "schema": "alpha_max_terminal_authority_policy.v3",
            "accepted_alpha_commit": ACCEPTED_COMMIT,
            "baseline_ancestor": BASELINE,
            "pins": {
                "runbook_sha256": ROLE_PINS["runbook"][2],
                "uv_lock_sha256": ROLE_PINS["alpha_uv_lock"][2],
                "alignment_receipt_sha256": ALIGNMENT_SHA256,
                "portfolio_sha256": ROLE_PINS["portfolio"][2],
                "contract_sha256": ROLE_PINS["contract_manifest"][2],
                "availability_sha256": ROLE_PINS["availability_evidence"][2],
                "preparer_sha256": ROLE_PINS["preparer"][2],
                "prelock_sha256": ROLE_PINS["prelock_script"][2],
                "historical_sha256": ROLE_PINS["historical_script"][2],
                "process_boundary_sha256": ROLE_PINS["process_boundary"][2],
                "acquirer_sha256": ROLE_PINS["acquirer"][2],
                "phase_wrapper_sha256": ROLE_PINS["phase_wrapper"][2],
            },
            "scope_order": list(_SCOPES),
        }
        policy_path = paths["control_root"] / "policy.json"
        _write_new(policy_path, policy_value)
        loaded_policy = policy.load_policy(policy_path)
        by_role = {item["role"]: item for item in files}
        by_role["policy_json"] = {"role": "policy_json", "file": _file(policy_path)}
        by_role["alignment_receipt"] = {"role": "alignment_receipt", "file": alignment}
        files = [by_role[role] for role in _FILE_ROLES]
        freezes = []
        for role, interpreter, freeze in freezes_data:
            freeze_path = paths["control_root"] / f"{role}.freeze.json"
            _write_new(freeze_path, freeze)
            freezes.append(
                {"role": role, "file": interpreter, "package_freeze": _file(freeze_path)}
            )
        chead, craw, cevidence = _repository_evidence(
            Path(CURRENT), "current", paths["control_root"]
        )
        ahead, araw, aevidence = _repository_evidence(
            Path(ACCEPTED), "accepted", paths["control_root"]
        )
        if (
            chead != current_head
            or craw["porcelain"] != _source_git(Path(CURRENT), "status", "--porcelain=v1", "-z")
            or ahead != ACCEPTED_COMMIT
            or araw["porcelain"]
        ):
            _fail("repository evidence changed")
        current_receipt = _write_new(
            paths["control_root"] / "current-state-receipt.json",
            {
                "schema": "alpha_max_v8_repository_state_receipt.v3",
                "repository_root": _directory(Path(CURRENT)),
                "head": chead,
                "accepted_alpha_commit": ACCEPTED_COMMIT,
                "baseline_ancestor": BASELINE,
                "verdict": "PASS_REVIEWED_OVERLAY",
                "approval": _file(paths["current_approval"]),
                "evidence": cevidence,
            },
        )
        accepted_receipt = _write_new(
            paths["control_root"] / "accepted-state-receipt.json",
            {
                "schema": "alpha_max_v8_repository_state_receipt.v3",
                "repository_root": _directory(Path(ACCEPTED)),
                "head": ahead,
                "accepted_alpha_commit": ACCEPTED_COMMIT,
                "baseline_ancestor": BASELINE,
                "verdict": "PASS_CLEAN",
                "evidence": aevidence,
            },
        )
        envelope = {
            "schema": "alpha_max_terminal_launch_envelope.v3",
            "policy_sha256": loaded_policy.source_sha256,
            "current_head": current_head,
            "accepted_alpha_commit": ACCEPTED_COMMIT,
            "baseline_ancestor": BASELINE,
            "repositories": [
                {
                    "role": "current_repository",
                    "root": _directory(Path(CURRENT)),
                    "head": current_head,
                    "clean_receipt": current_receipt,
                },
                {
                    "role": "accepted_alpha_repository",
                    "root": _directory(Path(ACCEPTED)),
                    "head": ACCEPTED_COMMIT,
                    "clean_receipt": accepted_receipt,
                },
            ],
            "files": files,
            "interpreters": freezes,
            "authority_key": authority,
            "observer_keys": observers,
            "forbidden_roots": list(_FORBIDDEN_ROOTS),
            "scope_order": list(_SCOPES),
        }
        checkpoint = {
            "schema": "alpha_max_terminal_checkpoint.v1",
            "accepted_alpha_commit": ACCEPTED_COMMIT,
            "baseline_ancestor": BASELINE,
            **policy_value["pins"],
            "authority_manifest_sha256": _sha(_canonical(envelope)),
        }
        checkpoint_path = paths["control_root"] / "checkpoint.json"
        _write_new(checkpoint_path, checkpoint)
        loaded_checkpoint = policy.load_checkpoint(checkpoint_path, loaded_policy)
        envelope_path = paths["control_root"] / "envelope.json"
        _write_new(envelope_path, envelope)
        loaded_envelope = policy.load_envelope(envelope_path, loaded_policy, loaded_checkpoint)
        request = {
            "schema": "alpha_max_terminal_request.acquisition.v1",
            "request_id": args.request_id,
            "scope": "acquisition",
            "checkpoint_pin_sha256": loaded_checkpoint.sha256,
            "interpreter": freezes[0]["file"],
            "repository_root": _directory(Path(CURRENT)),
            "evidence_root": _directory(paths["evidence_root"], private=True),
            "authority_socket": str(paths["evidence_root"] / "terminal-authority.sock"),
            "environment": {
                "HOME": str(paths["evidence_root"]),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TZ": "UTC",
            },
            "forbidden_roots": list(_FORBIDDEN_ROOTS),
            "publication": {
                "claim": "prelaunch.claim.json",
                "journal": "terminal-observer.journal.jsonl",
                "stdout": ["child-0.stdout.log", "child-1.stdout.log"],
                "stderr": ["child-0.stderr.log", "child-1.stderr.log"],
                "receipt": "terminal-authority.receipt.json",
            },
            "prerequisites": [
                _prerequisite("checkpoint_pin", _file(checkpoint_path)),
                _prerequisite("alignment_receipt", alignment),
            ],
            "acquirer": by_role["acquirer"]["file"],
            "contract_manifest": by_role["contract_manifest"]["file"],
            "availability_evidence": by_role["availability_evidence"]["file"],
            "source_root": source_absence,
            "report_root": report_absence,
        }
        request_path = paths["control_root"] / "acquisition-request.json"
        _write_new(request_path, request)
        loaded_request = policy.load_request(
            request_path,
            scope="acquisition",
            policy=loaded_policy,
            checkpoint=loaded_checkpoint,
            envelope=loaded_envelope,
        )
        commands = [
            list(item) for item in policy.derive_scope_commands(loaded_envelope, loaded_request)
        ]
        common = [
            str(paths["current_python"]),
            by_role["acquirer"]["file"]["path"],
            "--contract-manifest",
            by_role["contract_manifest"]["file"]["path"],
            "--availability-evidence",
            by_role["availability_evidence"]["file"]["path"],
            "--output-root",
            str(source),
            "--report-dir",
            str(report),
            "--forbidden-root",
            _FORBIDDEN_ROOTS[0],
            "--forbidden-root",
            _FORBIDDEN_ROOTS[1],
        ]
        if commands != [
            [*common, "--execute", "--validate-complete"],
            [*common, "--verify-eligible"],
        ]:
            _fail("native acquisition argv mismatch")
        telemetry = telemetry_preflight
        env = request["environment"]
        authority_name, observer_name, telemetry_name = (
            f"alpha-max-v8-authority-{args.run_id}.service",
            f"alpha-max-v8-observer-{args.run_id}.service",
            f"alpha-max-v8-telemetry-{args.run_id}.service",
        )
        authority_capture, observer_capture = (
            paths["telemetry_root"] / "authority-terminal.json",
            paths["telemetry_root"] / "observer-terminal.json",
        )
        scope_contracts = {scope: policy.scope_contract(scope) for scope in _SCOPES}
        authority_stop = [
            str(paths["current_python"]),
            str(paths["telemetry_script"]),
            "capture",
            "--expected-unit",
            authority_name,
            "--output",
            str(authority_capture),
        ]
        observer_stop = [
            str(paths["current_python"]),
            str(paths["telemetry_script"]),
            "capture",
            "--expected-unit",
            observer_name,
            "--output",
            str(observer_capture),
        ]
        monitor = [
            str(paths["current_python"]),
            str(paths["telemetry_script"]),
            "monitor",
            "--authority-unit",
            authority_name,
            "--observer-unit",
            observer_name,
            "--evidence-root",
            str(paths["telemetry_root"]),
            "--authority-terminal",
            str(authority_capture),
            "--observer-terminal",
            str(observer_capture),
            "--signed-terminal-receipt",
            str(paths["evidence_root"] / "terminal-authority.receipt.json"),
            "--authority-public-key",
            "%d/authority.public",
            "--request",
            str(request_path),
            "--output",
            str(paths["telemetry_root"] / "monitor.json"),
            "--interval-seconds",
            "5",
            "--timeout-seconds",
            "86400",
            "--authority-memory-max",
            str(AUTHORITY_MEMORY_MAX),
            "--authority-swap-max",
            str(AUTHORITY_SWAP_MAX),
            "--observer-memory-max",
            "3221225472",
            "--observer-swap-max",
            "536870912",
        ]
        authority_argv = [
            str(paths["current_python"]),
            by_role["authority_script"]["file"]["path"],
            "serve",
            "--policy",
            str(policy_path),
            "--checkpoint",
            str(checkpoint_path),
            "--envelope",
            str(envelope_path),
            "--request",
            str(request_path),
            "--private-key",
            "%d/authority.private",
            "--socket",
            str(paths["evidence_root"] / "terminal-authority.sock"),
            "--evidence-root",
            str(paths["evidence_root"]),
            "--scope",
            "acquisition",
        ]
        observer_argv = [
            str(paths["current_python"]),
            by_role["observer_script"]["file"]["path"],
            "--policy",
            str(policy_path),
            "--checkpoint",
            str(checkpoint_path),
            "--envelope",
            str(envelope_path),
            "--request",
            str(request_path),
            "--authority-socket",
            str(paths["evidence_root"] / "terminal-authority.sock"),
            "--observer-private-key",
            "%d/acquisition.private",
            "--evidence-root",
            str(paths["evidence_root"]),
            "--scope",
            "acquisition",
        ]
        plan = paths["control_root"] / "launch-plan.json"
        systemd_root = paths["control_root"] / "systemd"
        admission_receipt = paths["admission_root"] / "launch-admission-acquisition.json"

        def admission_prestart(name: str) -> list[str]:
            return _admission_prestart(
                str(paths["current_python"]),
                paths["current_approval"],
                admission_receipt,
                paths["key_root"],
                name,
                systemd_root / name,
                authority["public_key_b64"],
            )

        observer_write_paths = [
            str(paths["evidence_root"]),
            str(paths["telemetry_root"]),
            str(paths["output_parent"]),
        ]
        plan_value = {
            "schema": "alpha_max_v8_acquisition_launch_plan.v3",
            "launch_performed": False,
            "launch_eligible_only_with_complete": True,
            "launch_admission": {
                "receipt": str(admission_receipt),
                "schema": "alpha_max_v8_signed_launch_admission.v1",
                "requirements": [
                    "approval_digest",
                    "COMPLETE_manifest",
                    "production_unit_and_credential_identities",
                    "live_properties_and_cgroup_evidence",
                    "fresh_source_runtime_and_rendered_unit_readback",
                ],
            },
            "cgroup_contract": {
                "oom_policy_kill_implies_memory_oom_group": 1,
            },
            "scope_topology": {
                "approval": _file(paths["current_approval"]),
                "execution_alias": _execution_alias(),
                "run_id": args.run_id,
                "terminal_contracts": {
                    scope: {
                        "prerequisites": list(contract[0]),
                        "results": [
                            {"validated": list(validated), "sealed": list(sealed)}
                            for validated, sealed in contract[1]
                        ],
                    }
                    for scope, contract in scope_contracts.items()
                },
                "authority": {
                    "key": authority,
                    "retirement": "recovery_epoch_close",
                    "scopes": list(_SCOPES),
                },
                "scopes": {
                    "acquisition": {
                        "request_id": request_ids["acquisition"],
                        "request": _file(request_path),
                        "observer_key": next(
                            item for item in observers if item["scope"] == "acquisition"
                        ),
                        "observer_retirement": "after_acquisition_and_fresh_audit_readback",
                        "prerequisites": ["checkpoint_pin", "alignment_receipt"],
                        "results": [
                            ["source_eligible_receipt", "source_manifest", "source_journal"],
                            ["source_eligible_receipt", "source_manifest", "source_journal"],
                        ],
                    },
                    "phase_preparation": {
                        "request_id": request_ids["phase_preparation"],
                        "state": "deferred_until_authenticated_acquisition_receipts",
                        "observer_key": next(
                            item for item in observers if item["scope"] == "phase_preparation"
                        ),
                        "observer_retirement": "after_phase_terminal_and_readback",
                        "prerequisites": [
                            "checkpoint_pin",
                            "alignment_receipt",
                            "source_eligible_receipt",
                            "source_manifest",
                            "source_journal",
                        ],
                        "results": [["phase_handoff_receipt", "preparation_manifest"]],
                    },
                    "one_touch": {
                        "request_id": request_ids["one_touch"],
                        "state": "deferred_until_authenticated_phase_receipts",
                        "observer_key": next(
                            item for item in observers if item["scope"] == "one_touch"
                        ),
                        "observer_retirement": "after_one_touch_terminal_and_readback",
                        "prerequisites": [
                            "checkpoint_pin",
                            "alignment_receipt",
                            "phase_handoff_receipt",
                            "preparation_manifest",
                        ],
                        "results": [
                            [
                                "prelock_readback",
                                "prelock_observability",
                                "prelock_inventory_before",
                                "input_inventory_before",
                                "prelock_bundle",
                            ],
                            [
                                "historical_readback",
                                "historical_observability",
                                "prelock_inventory_after",
                                "input_inventory_after",
                                "historical_bundle",
                            ],
                        ],
                    },
                },
            },
            "telemetry_root": _directory(paths["telemetry_root"], private=True),
            "telemetry_script": telemetry,
            "executable_inputs": {
                name: {
                    "file": identity,
                    **(
                        {"package_freeze_sha256": EXECUTABLE_PINS[name]["package_freeze_sha256"]}
                        if "package_freeze_sha256" in EXECUTABLE_PINS[name]
                        else {}
                    ),
                }
                for name, identity in executable_inputs.items()
            },
            "acquisition_commands": commands,
            "systemd_units": {
                "authority": {
                    "name": authority_name,
                    **_unit(
                        "Alpha-Max v8 authority",
                        authority_argv,
                        env,
                        {
                            "high": AUTHORITY_MEMORY_HIGH,
                            "max": AUTHORITY_MEMORY_MAX,
                            "swap": AUTHORITY_SWAP_MAX,
                        },
                        authority_stop,
                        read_paths=[str(paths["control_root"])],
                        write_paths=[str(paths["evidence_root"]), str(paths["telemetry_root"])],
                        inaccessible_paths=[str(paths["key_root"])],
                        load_credential=(
                            f"authority.private:{paths['key_root'] / 'authority.private'}"
                        ),
                        prestart=admission_prestart(authority_name),
                    ),
                },
                "telemetry": {
                    "name": telemetry_name,
                    **_unit(
                        "Alpha-Max v8 telemetry",
                        monitor,
                        env,
                        {"high": 67108864, "max": 134217728, "swap": 33554432},
                        None,
                        read_paths=[
                            str(paths["control_root"]),
                            str(paths["evidence_root"]),
                        ],
                        write_paths=[str(paths["telemetry_root"])],
                        inaccessible_paths=[str(paths["key_root"])],
                        load_credential=(
                            f"authority.public:{paths['key_root'] / 'authority.public'}"
                        ),
                        prestart=admission_prestart(telemetry_name),
                    ),
                },
                "observer": {
                    "name": observer_name,
                    **_unit(
                        "Alpha-Max v8 observer",
                        observer_argv,
                        env,
                        {"high": 2147483648, "max": 3221225472, "swap": 536870912},
                        observer_stop,
                        read_paths=[
                            str(paths["control_root"]),
                            *_command_read_paths(
                                commands,
                                write_paths=observer_write_paths,
                                inaccessible_paths=[str(paths["key_root"])],
                            ),
                        ],
                        write_paths=observer_write_paths,
                        inaccessible_paths=[str(paths["key_root"])],
                        load_credential=(
                            f"acquisition.private:{paths['key_root'] / 'acquisition.private'}"
                        ),
                        observer=True,
                        prestart=admission_prestart(observer_name),
                    ),
                },
            },
            "ordering": [authority_name, telemetry_name, observer_name],
            "telemetry_contract": {
                "authority_capture": authority_stop,
                "observer_capture": observer_stop,
                "monitor": monitor,
            },
        }
        _create_root(systemd_root)
        rendered_units: dict[str, dict[str, Any]] = {}
        for role, item in plan_value["systemd_units"].items():
            unit = {key: value for key, value in item.items() if key != "name"}
            _validate_unit(unit)
            unit_file = _write_bytes_new(systemd_root / item["name"], _render_systemd_unit(unit))
            rendered_units[role] = {
                "name": item["name"],
                "file": unit_file,
                "credential": _credential_binding(unit, key_files),
            }
        plan_value["rendered_systemd_units"] = rendered_units
        _write_new(plan, plan_value)
        manifest = paths["control_root"] / "manifest.json"
        _write_new(
            manifest,
            {
                "schema": "alpha_max_v8_acquisition_manifest.v3",
                "launch_performed": False,
                "roots": {
                    name: _directory(paths[name], private=True)
                    for name in expected
                    if name != "stage_results_parent"
                },
                "admission_root": _directory(paths["admission_root"], private=True),
                "execution_alias": _execution_alias(),
                "source_absence": source_absence,
                "report_absence": report_absence,
                "policy": _file(policy_path),
                "checkpoint": _file(checkpoint_path),
                "envelope": _file(envelope_path),
                "request": _file(request_path),
                "launch_plan": _file(plan),
                "approval": _file(paths["current_approval"]),
                "repository_receipts": [current_receipt, accepted_receipt],
                "package_freezes": freezes,
                "executable_inputs": {
                    name: {
                        "file": identity,
                        **(
                            {
                                "package_freeze_sha256": EXECUTABLE_PINS[name][
                                    "package_freeze_sha256"
                                ]
                            }
                            if "package_freeze_sha256" in EXECUTABLE_PINS[name]
                            else {}
                        ),
                    }
                    for name, identity in executable_inputs.items()
                },
                "public_key_summary": key_summary_file,
                "telemetry": telemetry,
                "rendered_systemd_units": rendered_units,
            },
        )
        if (
            _state(Path(CURRENT), approval) != current_head
            or _git(Path(ACCEPTED), "rev-parse", "HEAD").decode().strip() != ACCEPTED_COMMIT
            or _git(Path(ACCEPTED), "status", "--porcelain=v1", "-z")
            or _file(Path(ALIGNMENT))["sha256"] != ALIGNMENT_SHA256
        ):
            _fail("repository or input revalidation failed")
        for artifact in (
            policy_path,
            checkpoint_path,
            envelope_path,
            request_path,
            plan,
            manifest,
            key_summary_file["path"],
            *[item["file"]["path"] for item in rendered_units.values()],
            *[item["package_freeze"]["path"] for item in freezes],
            current_receipt["path"],
            accepted_receipt["path"],
        ):
            _file(Path(artifact))
        manifest_value, plan_value, request_value = (
            _load_canonical(manifest),
            _load_canonical(plan),
            _load_canonical(request_path),
        )
        if (
            manifest_value["roots"]
            != {
                name: _directory(paths[name], private=True)
                for name in expected
                if name != "stage_results_parent"
            }
            or manifest_value["admission_root"] != _directory(paths["admission_root"], private=True)
            or manifest_value["execution_alias"] != _execution_alias()
            or plan_value["scope_topology"]["execution_alias"] != _execution_alias()
            or manifest_value["public_key_summary"] != key_summary_file
            or manifest_value["repository_receipts"] != [current_receipt, accepted_receipt]
            or manifest_value["executable_inputs"] != plan_value["executable_inputs"]
            or manifest_value["rendered_systemd_units"] != plan_value["rendered_systemd_units"]
            or plan_value["rendered_systemd_units"] != rendered_units
            or plan_value["executable_inputs"]
            != {
                name: {
                    "file": identity,
                    **(
                        {"package_freeze_sha256": EXECUTABLE_PINS[name]["package_freeze_sha256"]}
                        if "package_freeze_sha256" in EXECUTABLE_PINS[name]
                        else {}
                    ),
                }
                for name, identity in executable_inputs.items()
            }
        ):
            _fail("manifest binding revalidation failed")
        if (
            _load_canonical(Path(key_summary_file["path"])) != key_summary
            or _load_canonical(Path(current_receipt["path"]))["verdict"] != "PASS_REVIEWED_OVERLAY"
            or _load_canonical(Path(accepted_receipt["path"]))["verdict"] != "PASS_CLEAN"
        ):
            _fail("receipt or public key summary revalidation failed")
        for item in freezes:
            freeze = _load_canonical(Path(item["package_freeze"]["path"]))
            if (
                freeze.get("schema") != "alpha_max_v8_interpreter_package_freeze.v1"
                or freeze.get("interpreter_sha256") != item["file"]["sha256"]
            ):
                _fail("freeze revalidation failed")
        if not (
            plan_value["launch_performed"] is False
            and plan_value["launch_eligible_only_with_complete"] is True
            and request_value["source_root"] == source_absence
            and request_value["report_root"] == report_absence
        ):
            _fail("launch or absence revalidation failed")
        for item in files:
            if _file(Path(item["file"]["path"])) != item["file"]:
                _fail("role identity revalidation failed")
        for _, interpreter, _ in freezes_data:
            if _file(Path(interpreter["path"])) != interpreter:
                _fail("interpreter identity revalidation failed")
        if _file(paths["telemetry_script"]) != telemetry_preflight:
            _fail("telemetry identity revalidation failed")
        _revalidate_key_files(key_files)
        for role, item in plan_value["systemd_units"].items():
            unit = {key: value for key, value in item.items() if key != "name"}
            rendered = rendered_units[role]
            if (
                _file(Path(rendered["file"]["path"])) != rendered["file"]
                or _sha(_render_systemd_unit(unit)) != rendered["file"]["sha256"]
                or _credential_binding(unit, key_files) != rendered["credential"]
            ):
                _fail("rendered systemd unit binding revalidation failed")
        policy.load_policy(policy_path)
        policy.load_checkpoint(checkpoint_path, loaded_policy)
        policy.load_envelope(envelope_path, loaded_policy, loaded_checkpoint)
        policy.load_request(
            request_path,
            scope="acquisition",
            policy=loaded_policy,
            checkpoint=loaded_checkpoint,
            envelope=loaded_envelope,
        )
        _absent(source)
        _absent(report)
        result = {
            "policy": str(policy_path),
            "approval": str(paths["current_approval"]),
            "checkpoint": str(checkpoint_path),
            "envelope": str(envelope_path),
            "request": str(request_path),
            "launch_plan": str(plan),
            "manifest": str(manifest),
        }
        complete = {
            "schema": "alpha_max_v8_acquisition_complete.v3",
            "launch_performed": False,
            "manifest_sha256": _file(manifest)["sha256"],
        }
        _publish_complete(paths["control_root"], complete)
        return result
    except BaseException as error:
        _quarantine_or_reraise(
            paths["control_root"] if "control_root" in created else None, created, error
        )


def _build_stage(args: argparse.Namespace, scope: str) -> dict[str, str]:
    """Append one exact terminal scope after its authenticated predecessor exists."""
    if scope not in {"phase_preparation", "one_touch"}:
        _fail("invalid staged scope")
    control, keys, output = (
        _absolute(args.control_root),
        _absolute(args.key_root),
        _absolute(args.output_parent),
    )
    approval_path = _absolute(args.current_approval)
    if approval_path != RECOVERY_ROOT / CURRENT_APPROVAL_LEAF or output.parent != RECOVERY_ROOT:
        _fail("staged approval or recovery root mismatch")
    request_ids = {
        "acquisition": args.request_id,
        "phase_preparation": args.phase_request_id,
        "one_touch": args.one_touch_request_id,
    }
    expected = {
        "control_root": f"g056v8-controls-{args.run_id}",
        "key_root": f"g056v8-keys-{args.run_id}",
        "evidence_root": f"g056v8-acquisition-evidence-{args.run_id}",
        "telemetry_root": f"g056v8-telemetry-{args.run_id}",
        "output_parent": f"g056v8-acquisition-output-{args.run_id}",
        "stage_results_parent": f"g056v8-stage-results-{args.run_id}",
    }
    paths = {
        "control_root": control,
        "key_root": keys,
        "evidence_root": _absolute(args.evidence_root),
        "telemetry_root": _absolute(args.telemetry_root),
        "output_parent": output,
        "stage_results_parent": RECOVERY_ROOT / f"g056v8-stage-results-{args.run_id}",
    }
    paths["admission_root"] = control / "admissions"
    if (
        args.run_id != RUN_ID
        or request_ids
        != {
            "acquisition": ACQUISITION_REQUEST_ID,
            "phase_preparation": PHASE_PREPARATION_REQUEST_ID,
            "one_touch": ONE_TOUCH_REQUEST_ID,
        }
        or any(
            paths[name].parent != RECOVERY_ROOT or paths[name].name != leaf
            for name, leaf in expected.items()
        )
    ):
        _fail("staged identity mismatch")
    approval = _load_current_approval(
        approval_path,
        run_id=args.run_id,
        request_ids=request_ids,
        absent_paths={name: paths[name] for name in expected},
        require_absent=False,
    )
    manifest = _load_canonical(control / "manifest.json")
    complete = _load_canonical(control / "COMPLETE.json")
    if (
        complete.get("manifest_sha256") != _file(control / "manifest.json")["sha256"]
        or manifest.get("approval") != _file(approval_path)
        or manifest.get("roots")
        != {
            name: _directory(paths[name], private=True)
            for name in expected
            if name != "stage_results_parent"
        }
        or manifest.get("admission_root") != _directory(paths["admission_root"], private=True)
        or manifest.get("execution_alias") != _execution_alias()
        or _state(Path(CURRENT), approval) != approval["head"]
    ):
        _fail("acquisition control readback mismatch")
    authority, observers, _summary, key_files = _key_bindings(keys)
    _revalidate_key_files(key_files)
    policy = _load_policy(Path(manifest["policy"]["path"]), manifest["policy"])
    checkpoint = policy.load_checkpoint(Path(manifest["checkpoint"]["path"]), policy)
    envelope = policy.load_envelope(Path(manifest["envelope"]["path"]), policy, checkpoint)
    if authority["key_id"] != envelope.authority_key.key_id:
        _fail("authority retention mismatch")
    observer = next(item for item in observers if item["scope"] == scope)
    canonical_finalize_receipt = None
    canonical_finalize_identity: FileIdentity | None = None
    if scope == "phase_preparation":
        canonical_finalize_receipt = _absolute(args.canonical_finalize_receipt)
        canonical_finalize_identity = _file(canonical_finalize_receipt)
        policy.validate_w10_canonical_finalize_bundle(
            canonical_finalize_receipt,
            authority_public_key_b64=envelope.authority_key.public_key_b64,
            run_id=args.run_id,
            acquisition_request_id=ACQUISITION_REQUEST_ID,
            approval_sha256=_file(approval_path)["sha256"],
        )
        if _file(canonical_finalize_receipt) != canonical_finalize_identity:
            _fail("canonical finalize receipt identity drift")
    scope_evidence = paths["evidence_root"] / scope
    _absent(scope_evidence)
    _create_root(scope_evidence)
    source = output / "source"
    report = output / "report"
    prereq_paths = (
        [
            checkpoint.source_path,
            envelope.file("alignment_receipt").path,
            str(report / "source_eligible_receipt.json"),
            str(report / "source_manifest.json"),
            str(report / "acquisition.journal.jsonl"),
            str(canonical_finalize_receipt),
        ]
        if scope == "phase_preparation"
        else [
            checkpoint.source_path,
            envelope.file("alignment_receipt").path,
            str(
                _absolute(args.phase_output).parent
                / f".{_absolute(args.phase_output).name}.alpha_max_phase_preparation.handoff.json"
            ),
            str(_absolute(args.phase_output) / "preparation_manifest.json"),
        ]
    )
    prerequisite_kinds, _results = policy.scope_contract(scope)
    prerequisites = [
        _prerequisite(
            kind,
            (
                canonical_finalize_identity
                if kind == "canonical_finalize_receipt" and canonical_finalize_identity is not None
                else _file(Path(path))
            ),
        )
        for kind, path in zip(prerequisite_kinds, prereq_paths)
    ]
    environment = {
        "HOME": str(scope_evidence),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "UTC",
    }
    by_role = {
        item["role"]: item["file"]
        for item in _load_canonical(Path(manifest["envelope"]["path"]))["files"]
    }
    phase_output = _absolute(args.phase_output)
    stage_results = paths["stage_results_parent"]
    phase_parent = stage_results / "phase_preparation"
    if scope == "phase_preparation":
        _absent(stage_results)
        _create_root(stage_results)
        _create_root(phase_parent)
        if phase_output != phase_parent / "result":
            _fail("phase output must be the exact dedicated stage-results child")
        _absent(phase_output)
        request = {
            "schema": "alpha_max_terminal_request.phase_preparation.v1",
            "request_id": request_ids[scope],
            "scope": scope,
            "checkpoint_pin_sha256": checkpoint.sha256,
            "interpreter": _load_canonical(Path(manifest["envelope"]["path"]))["interpreters"][0][
                "file"
            ],
            "repository_root": _load_canonical(Path(manifest["envelope"]["path"]))["repositories"][
                0
            ]["root"],
            "evidence_root": _directory(scope_evidence, private=True),
            "authority_socket": str(scope_evidence / "terminal-authority.sock"),
            "environment": environment,
            "forbidden_roots": list(_FORBIDDEN_ROOTS),
            "publication": {
                "claim": "prelaunch.claim.json",
                "journal": "terminal-observer.journal.jsonl",
                "stdout": ["child-0.stdout.log"],
                "stderr": ["child-0.stderr.log"],
                "receipt": "terminal-authority.receipt.json",
            },
            "prerequisites": prerequisites,
            "phase_wrapper": by_role["phase_wrapper"],
            "acquirer": by_role["acquirer"],
            "source_root": _directory(source, private=True),
            "source_report": _directory(report, private=True),
            "contract_manifest": by_role["contract_manifest"],
            "availability_evidence": by_role["availability_evidence"],
            "preparer": by_role["preparer"],
            "phase_output": _absent(phase_output),
        }
    else:
        prelock, historical = _absolute(args.prelock_output), _absolute(args.historical_output)
        prelock_parent = stage_results / "prelock"
        historical_parent = stage_results / "historical"
        if (
            not stage_results.is_dir()
            or phase_output != phase_parent / "result"
            or prelock != prelock_parent / "result"
            or historical != historical_parent / "result"
        ):
            _fail("one-touch stage result topology mismatch")
        _directory(stage_results, private=True)
        _directory(phase_parent, private=True)
        _directory(phase_output, private=True)
        _create_root(prelock_parent)
        _create_root(historical_parent)
        _absent(prelock)
        _absent(historical)
        request = {
            "schema": "alpha_max_terminal_request.one_touch.v1",
            "request_id": request_ids[scope],
            "scope": scope,
            "checkpoint_pin_sha256": checkpoint.sha256,
            "interpreter": _load_canonical(Path(manifest["envelope"]["path"]))["interpreters"][1][
                "file"
            ],
            "repository_root": _load_canonical(Path(manifest["envelope"]["path"]))["repositories"][
                1
            ]["root"],
            "evidence_root": _directory(scope_evidence, private=True),
            "authority_socket": str(scope_evidence / "terminal-authority.sock"),
            "environment": environment,
            "forbidden_roots": list(_FORBIDDEN_ROOTS),
            "publication": {
                "claim": "prelaunch.claim.json",
                "journal": "terminal-observer.journal.jsonl",
                "stdout": ["child-0.stdout.log", "child-1.stdout.log"],
                "stderr": ["child-0.stderr.log", "child-1.stderr.log"],
                "receipt": "terminal-authority.receipt.json",
            },
            "prerequisites": prerequisites,
            "portfolio": by_role["portfolio"],
            "contract_manifest": by_role["contract_manifest"],
            "prelock_script": by_role["prelock_script"],
            "historical_script": by_role["historical_script"],
            "phase_output": _directory(phase_output, private=True),
            "prelock_output": _absent(prelock),
            "historical_output": _absent(historical),
        }
    request_path = control / f"{scope}-request.json"
    _write_new(request_path, request)
    loaded = policy.load_request(
        request_path, scope=scope, policy=policy, checkpoint=checkpoint, envelope=envelope
    )
    commands = [list(command) for command in policy.derive_scope_commands(envelope, loaded)]
    systemd = control / "systemd"
    authority_name = f"alpha-max-v8-{scope}-authority-{args.run_id}.service"
    observer_name = f"alpha-max-v8-{scope}-observer-{args.run_id}.service"
    telemetry_name = f"alpha-max-v8-{scope}-telemetry-{args.run_id}.service"
    authority_capture = paths["telemetry_root"] / f"{scope}-authority-terminal.json"
    observer_capture = paths["telemetry_root"] / f"{scope}-observer-terminal.json"
    authority_stop = [
        _load_canonical(Path(manifest["envelope"]["path"]))["interpreters"][0]["file"]["path"],
        manifest["telemetry"]["path"],
        "capture",
        "--expected-unit",
        authority_name,
        "--output",
        str(authority_capture),
    ]
    observer_stop = [
        _load_canonical(Path(manifest["envelope"]["path"]))["interpreters"][0]["file"]["path"],
        manifest["telemetry"]["path"],
        "capture",
        "--expected-unit",
        observer_name,
        "--output",
        str(observer_capture),
    ]
    observer_argv = [
        _load_canonical(Path(manifest["envelope"]["path"]))["interpreters"][0]["file"]["path"],
        by_role["observer_script"]["path"],
        "--policy",
        manifest["policy"]["path"],
        "--checkpoint",
        manifest["checkpoint"]["path"],
        "--envelope",
        manifest["envelope"]["path"],
        "--request",
        str(request_path),
        "--authority-socket",
        str(scope_evidence / "terminal-authority.sock"),
        "--observer-private-key",
        f"%d/{scope}.private",
        "--evidence-root",
        str(scope_evidence),
        "--scope",
        scope,
    ]
    admission_receipt = paths["admission_root"] / f"launch-admission-{scope}.json"

    def staged_prestart(name: str) -> list[str]:
        return _admission_prestart(
            _load_canonical(Path(manifest["envelope"]["path"]))["interpreters"][0]["file"]["path"],
            approval_path,
            admission_receipt,
            keys,
            name,
            systemd / name,
            authority["public_key_b64"],
        )

    stage_write_paths = [
        str(scope_evidence),
        str(paths["telemetry_root"]),
        *(
            [str(phase_parent)]
            if scope == "phase_preparation"
            else [str(prelock_parent), str(historical_parent)]
        ),
    ]
    unit = _unit(
        f"Alpha-Max v8 {scope} observer",
        observer_argv,
        environment,
        {"high": 2147483648, "max": 3221225472, "swap": 536870912},
        observer_stop,
        read_paths=[
            str(control),
            *_command_read_paths(
                commands,
                write_paths=stage_write_paths,
                inaccessible_paths=[str(keys)],
            ),
        ],
        write_paths=stage_write_paths,
        inaccessible_paths=[str(keys)],
        load_credential=f"{scope}.private:{keys / (scope + '.private')}",
        observer=scope == "acquisition",
        prestart=staged_prestart(observer_name),
    )
    _validate_unit(unit)
    authority_argv = [
        _load_canonical(Path(manifest["envelope"]["path"]))["interpreters"][0]["file"]["path"],
        by_role["authority_script"]["path"],
        "serve",
        "--policy",
        manifest["policy"]["path"],
        "--checkpoint",
        manifest["checkpoint"]["path"],
        "--envelope",
        manifest["envelope"]["path"],
        "--request",
        str(request_path),
        "--private-key",
        "%d/authority.private",
        "--socket",
        str(scope_evidence / "terminal-authority.sock"),
        "--evidence-root",
        str(scope_evidence),
        "--scope",
        scope,
    ]
    authority_unit = _unit(
        f"Alpha-Max v8 {scope} authority",
        authority_argv,
        environment,
        {"high": AUTHORITY_MEMORY_HIGH, "max": AUTHORITY_MEMORY_MAX, "swap": AUTHORITY_SWAP_MAX},
        authority_stop,
        read_paths=[str(control)],
        write_paths=[str(scope_evidence), str(paths["telemetry_root"])],
        inaccessible_paths=[str(keys)],
        load_credential=f"authority.private:{keys / 'authority.private'}",
        prestart=staged_prestart(authority_name),
    )
    _validate_unit(authority_unit)
    authority_file = _write_bytes_new(
        systemd / f"alpha-max-v8-{scope}-authority-{args.run_id}.service",
        _render_systemd_unit(authority_unit),
    )
    authority_binding = _credential_binding(authority_unit, key_files)
    telemetry_argv = [
        _load_canonical(Path(manifest["envelope"]["path"]))["interpreters"][0]["file"]["path"],
        manifest["telemetry"]["path"],
        "monitor",
        "--authority-unit",
        authority_name,
        "--observer-unit",
        observer_name,
        "--evidence-root",
        str(paths["telemetry_root"]),
        "--authority-terminal",
        str(authority_capture),
        "--observer-terminal",
        str(observer_capture),
        "--signed-terminal-receipt",
        str(scope_evidence / "terminal-authority.receipt.json"),
        "--authority-public-key",
        "%d/authority.public",
        "--request",
        str(request_path),
        "--output",
        str(paths["telemetry_root"] / f"{scope}-monitor.json"),
        "--interval-seconds",
        "5",
        "--timeout-seconds",
        "86400",
        "--authority-memory-max",
        str(AUTHORITY_MEMORY_MAX),
        "--authority-swap-max",
        str(AUTHORITY_SWAP_MAX),
        "--observer-memory-max",
        "3221225472",
        "--observer-swap-max",
        "536870912",
    ]
    telemetry_unit = _unit(
        f"Alpha-Max v8 {scope} telemetry",
        telemetry_argv,
        environment,
        {"high": 67108864, "max": 134217728, "swap": 33554432},
        None,
        read_paths=[str(control), str(scope_evidence)],
        write_paths=[str(paths["telemetry_root"])],
        inaccessible_paths=[str(keys)],
        load_credential=f"authority.public:{keys / 'authority.public'}",
        prestart=staged_prestart(telemetry_name),
    )
    _validate_unit(telemetry_unit)
    telemetry_file = _write_bytes_new(
        systemd / f"alpha-max-v8-{scope}-telemetry-{args.run_id}.service",
        _render_systemd_unit(telemetry_unit),
    )
    telemetry_binding = _credential_binding(telemetry_unit, key_files)
    unit_file = _write_bytes_new(
        systemd / f"alpha-max-v8-{scope}-observer-{args.run_id}.service", _render_systemd_unit(unit)
    )
    binding = _credential_binding(unit, key_files)
    stage_manifest = {
        "schema": "alpha_max_v8_terminal_stage.v1",
        "scope": scope,
        "approval": _file(approval_path),
        "acquisition_manifest": _file(control / "manifest.json"),
        "request": _file(request_path),
        "commands": commands,
        "observer": observer,
        "authority": authority,
        "unit_definitions": {
            "authority": authority_unit,
            "telemetry": telemetry_unit,
            "observer": unit,
        },
        "observer_retirement": "after_phase_terminal_and_readback"
        if scope == "phase_preparation"
        else "after_one_touch_terminal_and_readback",
        "authority_retirement": "recovery_epoch_close",
        "units": {
            "authority": {
                "name": authority_name,
                "file": authority_file,
                "credential": authority_binding,
            },
            "telemetry": {
                "name": telemetry_name,
                "file": telemetry_file,
                "credential": telemetry_binding,
                "authority_capture": str(authority_capture),
                "observer_capture": str(observer_capture),
                "terminal_receipt": str(scope_evidence / "terminal-authority.receipt.json"),
            },
            "observer": {"name": observer_name, "file": unit_file, "credential": binding},
        },
    }
    stage_path = control / f"{scope}-manifest.json"
    _write_new(stage_path, stage_manifest)
    _write_new(
        control / f"{scope}.COMPLETE.json",
        {
            "schema": "alpha_max_v8_terminal_stage_complete.v1",
            "scope": scope,
            "manifest_sha256": _file(stage_path)["sha256"],
        },
    )
    return {"request": str(request_path), "manifest": str(stage_path), "unit": str(unit_file)}


def build_stage(args: argparse.Namespace, scope: str) -> dict[str, str]:
    """Fail closed by quarantining the staged control root on every stage failure."""
    try:
        return _build_stage(args, scope)
    except BaseException as error:
        try:
            control = _absolute(args.control_root)
            if control.exists() and not (control / f"{scope}.COMPLETE.json").exists():
                _write_new(
                    control / f"{scope}.FAILED.json",
                    {
                        "schema": "alpha_max_v8_terminal_stage_failed.v1",
                        "scope": scope,
                        "error": type(error).__name__,
                    },
                )
        except BaseException as quarantine_error:
            raise BaseExceptionGroup(
                "stage build failed and quarantine failed", [error, quarantine_error]
            ) from error
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "control_root",
        "key_root",
        "evidence_root",
        "telemetry_root",
        "output_parent",
        "current_approval",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    for name in ("telemetry_script", "current_python", "accepted_python"):
        parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--phase-request-id", required=True)
    parser.add_argument("--one-touch-request-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--approval-only", action="store_true")
    mode.add_argument("--build-acquisition", action="store_true")
    mode.add_argument("--build-phase-preparation", action="store_true")
    mode.add_argument("--build-one-touch", action="store_true")
    parser.add_argument("--phase-output")
    parser.add_argument("--prelock-output")
    parser.add_argument("--historical-output")
    parser.add_argument("--canonical-finalize-receipt")
    args = parser.parse_args(argv)
    if args.build_acquisition and any(
        getattr(args, name) is None
        for name in ("telemetry_script", "current_python", "accepted_python")
    ):
        parser.error("acquisition build requires telemetry script and both interpreters")
    if args.build_phase_preparation and (
        args.phase_output is None or args.canonical_finalize_receipt is None
    ):
        parser.error("phase build requires --phase-output and --canonical-finalize-receipt")
    if args.build_one_touch and any(
        getattr(args, name) is None
        for name in ("phase_output", "prelock_output", "historical_output")
    ):
        parser.error("one-touch build requires phase, prelock, and historical outputs")
    result = (
        create_approval(args)
        if args.approval_only
        else build(args)
        if args.build_acquisition
        else build_stage(args, "phase_preparation")
        if args.build_phase_preparation
        else build_stage(args, "one_touch")
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
