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
G067_APPROVAL = (
    "/home/hoky/quants-recovery-runs/g065-oom-safety-20260726/g067-solusdt-202311-approval.json"
)
CURRENT_APPROVAL = (
    "/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/"
    "current-state-approval-v6.json"
)
ALIGNMENT = (
    "/home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-rev515-alignment-receipt-v5.json"
)
ALIGNMENT_SHA256 = "8687b52180502a11de9fbe317a19d00bb4492c464b3bf33d4eda2437683ca812"
HEX = re.compile(r"^[0-9a-f]{64}$")
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
    "telemetry_script": {
        "path": "/home/hoky/Quants-agent/LuminaQuant/scripts/research/monitor_alpha_max_v8_resources.py",
        "sha256": "5d3e7eedea70102c6aa182e153139131bdbcd3ffc904499184aecc55eee54d4f",
        "byte_count": 28491,
        "mode": 0o600,
    },
}
G067_IDENTITY = {
    "scientific_verdict": "APPROVED_EXACT_IDENTITY",
    "symbol": "SOLUSDT",
    "official_archive_url": "https://data.binance.vision/data/futures/um/monthly/aggTrades/SOLUSDT/SOLUSDT-aggTrades-2023-11.zip",
    "official_checksum_url": "https://data.binance.vision/data/futures/um/monthly/aggTrades/SOLUSDT/SOLUSDT-aggTrades-2023-11.zip.CHECKSUM",
    "archive_sha256": "188c3145ecaab1cf546318c293fb4fef0e320a6dc05b14eea013a46209ebbd73",
    "archive_byte_count": 535864305,
    "checksum_sha256": "d1a92cf7d5775d5edd1960d75091c06af72955c99fb806dca4ccf670af983f9d",
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
    return subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True).stdout


def _inventory(root: Path) -> bytes:
    names = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")[
        :-1
    ]
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


def _record(raw: bytes) -> dict[str, Any]:
    return {"sha256": _sha(raw), "byte_count": len(raw)}


def _state(root: Path, approval: dict[str, Any]) -> str:
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    actual = {
        "head": head,
        "porcelain": _record(_git(root, "status", "--porcelain=v1", "-z")),
        "commit_overlay": _record(_git(root, "diff", "--binary", f"{ACCEPTED_COMMIT}..HEAD")),
        "worktree_overlay": _record(_git(root, "diff", "--binary", "HEAD")),
        "source_inventory": _record(_inventory(root)),
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
        path = paths[name]
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
        f"{name}.{kind}" for name in ("authority", *_SCOPES) for kind in ("private", "public")
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
    for name in ("authority", *_SCOPES):
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
    if len({item["key_id"] for item in bindings}) != 4:
        _fail("duplicate keys")
    summary = {"schema": "alpha_max_v8_public_key_summary.v1", "keys": public_files}
    return (
        bindings[0],
        [{"scope": scope, **item} for scope, item in zip(_SCOPES, bindings[1:])],
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
) -> dict[str, Any]:
    service = {
        "UMask": "0077",
        "Type": "exec",
        "ExecStart": argv,
        "Environment": [f"{k}={v}" for k, v in env.items()],
        "WorkingDirectory": CURRENT,
        "NoNewPrivileges": True,
        "PrivateTmp": True,
        "PrivateDevices": True,
        "ProtectSystem": "strict",
        "ProtectHome": "read-only",
        "ReadOnlyPaths": read_paths,
        "InaccessiblePaths": inaccessible_paths,
        "LoadCredential": [load_credential],
        "ReadWritePaths": write_paths,
        "ProtectKernelTunables": True,
        "ProtectKernelModules": True,
        "ProtectControlGroups": True,
        "RestrictAddressFamilies": ["AF_UNIX", "AF_INET", "AF_INET6"] if observer else ["AF_UNIX"],
        "MemoryHigh": limits["high"],
        "MemoryMax": limits["max"],
        "MemorySwapMax": limits["swap"],
        "OOMPolicy": "kill",
        "TimeoutStartSec": "1800s",
        "TimeoutStopSec": "120s",
    }
    if stop is not None:
        service["ExecStopPost"] = stop
    if not observer:
        service["IPAddressDeny"] = "any"
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
    "ExecStopPost",
    "Environment",
    "WorkingDirectory",
    "NoNewPrivileges",
    "PrivateTmp",
    "PrivateDevices",
    "ProtectSystem",
    "ProtectHome",
    "ReadOnlyPaths",
    "InaccessiblePaths",
    "LoadCredential",
    "ReadWritePaths",
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
        if name in {"ExecStart", "ExecStopPost", "Environment"}:
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
        "Environment",
        "WorkingDirectory",
        "NoNewPrivileges",
        "PrivateTmp",
        "PrivateDevices",
        "ProtectSystem",
        "ProtectHome",
        "ReadOnlyPaths",
        "InaccessiblePaths",
        "LoadCredential",
        "ReadWritePaths",
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
    path_directives = ("ReadOnlyPaths", "ReadWritePaths", "InaccessiblePaths")
    if any(not isinstance(service[key], list) for key in path_directives):
        _fail("invalid systemd path directives")
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
        for path in (
            *service["ReadOnlyPaths"],
            *service["ReadWritePaths"],
            *service["InaccessiblePaths"],
        )
    ):
        _fail("invalid systemd path directives")
    read_paths = set(service["ReadOnlyPaths"])
    write_paths = set(service["ReadWritePaths"])
    inaccessible_paths = set(service["InaccessiblePaths"])
    if len(service["InaccessiblePaths"]) != 1:
        _fail("systemd inaccessible paths must contain only the key root")
    key_root = service["InaccessiblePaths"][0]
    source_path = Path(source)
    if not source_path.is_absolute() or source_path.parent != Path(key_root):
        _fail("systemd credential source must be directly under the inaccessible key root")
    expected_arguments = {
        "authority.private": "--private-key",
        "acquisition.private": "--observer-private-key",
        "authority.public": "--authority-public-key",
    }
    if name not in expected_arguments or source_path.name != name:
        _fail("systemd credential name does not match its source")
    argv = service["ExecStart"]
    argument = expected_arguments[name]
    if (
        not isinstance(argv, list)
        or argv.count(argument) != 1
        or argv[argv.index(argument) + 1 : argv.index(argument) + 2] != [f"%d/{name}"]
    ):
        _fail("systemd credential does not match unit role argv")
    if key_root in read_paths or key_root in write_paths:
        _fail("systemd key root must not be readable or writable")
    if read_paths & write_paths:
        _fail("systemd read-only and read-write paths overlap")
    if (read_paths | write_paths) & inaccessible_paths:
        _fail("systemd accessible and inaccessible paths overlap")


def _repository_evidence(
    root: Path, prefix: str, control: Path
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    raw = {
        "porcelain": _git(root, "status", "--porcelain=v1", "-z"),
        "commit_overlay": _git(root, "diff", "--binary", f"{ACCEPTED_COMMIT}..HEAD"),
        "worktree_overlay": _git(root, "diff", "--binary", "HEAD"),
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
            "g067_approval",
            "current_approval",
            "current_python",
            "accepted_python",
        )
    }
    if paths["g067_approval"] != Path(G067_APPROVAL) or paths["current_approval"] != Path(
        CURRENT_APPROVAL
    ):
        _fail("approval path mismatch")
    if (
        not HEX.fullmatch(args.run_id)
        or not HEX.fullmatch(args.request_id)
        or args.run_id == args.request_id
    ):
        _fail("run and request IDs must be distinct lowercase SHA-256 values")
    recovery = paths["output_parent"].parent
    if recovery != RECOVERY_ROOT:
        _fail("unexpected recovery root")
    expected = {
        "control_root": f"g056v8-controls-{args.run_id}",
        "key_root": f"g056v8-keys-{args.run_id}",
        "evidence_root": f"g056v8-acquisition-evidence-{args.run_id}",
        "telemetry_root": f"g056v8-telemetry-{args.run_id}",
        "output_parent": f"g056v8-acquisition-output-{args.run_id}",
    }
    if (
        any(
            paths[name].parent != recovery or paths[name].name != leaf
            for name, leaf in expected.items()
        )
        or len({paths[name] for name in expected}) != 5
    ):
        _fail("invalid v8 root topology")
    _directory(recovery)
    for name in expected:
        _absent(paths[name])
    g067 = _approval(
        paths["g067_approval"],
        {"schema", *G067_IDENTITY, "approved_utc"},
        "alpha_max_g067_sol_archive_approval.v1",
    )
    if any(g067[key] != value for key, value in G067_IDENTITY.items()):
        _fail("G067 identity mismatch")
    fields = {
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
        "approved_utc",
    }
    approval = _approval(
        paths["current_approval"], fields, "alpha_max_v8_current_state_approval.v2"
    )
    if (
        approval["repository_root"] != CURRENT
        or approval["accepted_alpha_commit"] != ACCEPTED_COMMIT
        or approval["baseline_ancestor"] != BASELINE
        or approval["verdict"] != "PASS_REVIEWED_OVERLAY"
    ):
        _fail("current approval binding mismatch")
    current_head = _state(Path(CURRENT), approval)
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
            or craw["porcelain"] != _git(Path(CURRENT), "status", "--porcelain=v1", "-z")
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
            "536870912",
            "--authority-swap-max",
            "67108864",
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
        plan_value = {
            "schema": "alpha_max_v8_acquisition_launch_plan.v3",
            "launch_performed": False,
            "launch_eligible_only_with_complete": True,
            "cgroup_contract": {
                "oom_policy_kill_implies_memory_oom_group": 1,
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
                        {"high": 268435456, "max": 536870912, "swap": 67108864},
                        authority_stop,
                        read_paths=[str(paths["control_root"])],
                        write_paths=[str(paths["evidence_root"]), str(paths["telemetry_root"])],
                        inaccessible_paths=[str(paths["key_root"])],
                        load_credential=(
                            f"authority.private:{paths['key_root'] / 'authority.private'}"
                        ),
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
                        read_paths=[str(paths["control_root"])],
                        write_paths=[
                            str(paths["evidence_root"]),
                            str(paths["telemetry_root"]),
                            str(paths["output_parent"]),
                        ],
                        inaccessible_paths=[str(paths["key_root"])],
                        load_credential=(
                            f"acquisition.private:{paths['key_root'] / 'acquisition.private'}"
                        ),
                        observer=True,
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
        systemd_root = paths["control_root"] / "systemd"
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
                "roots": {name: _directory(paths[name], private=True) for name in expected},
                "source_absence": source_absence,
                "report_absence": report_absence,
                "policy": _file(policy_path),
                "checkpoint": _file(checkpoint_path),
                "envelope": _file(envelope_path),
                "request": _file(request_path),
                "launch_plan": _file(plan),
                "g067_approval": _file(paths["g067_approval"]),
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
            != {name: _directory(paths[name], private=True) for name in expected}
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "control_root",
        "key_root",
        "evidence_root",
        "telemetry_root",
        "output_parent",
        "telemetry_script",
        "g067_approval",
        "current_approval",
        "current_python",
        "accepted_python",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-id", required=True)
    print(json.dumps(build(parser.parse_args(argv)), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
