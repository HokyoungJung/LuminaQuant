#!/usr/bin/env python3
"""Fail-closed, read-only cgroup terminal evidence for Alpha-Max v8."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import PurePosixPath

from lumina_quant.alpha_max_terminal_policy import TerminalPolicyError, verify_signed_receipt

CAPTURE_SCHEMA = "alpha_max_v8_terminal_cgroup.v1"
SAMPLE_SCHEMA = "alpha_max_v8_resource_sample.v2"
FINAL_SCHEMA = "alpha_max_v8_resource_receipt.v1"
_EVENTS = ("low", "high", "max", "oom", "oom_kill", "oom_group_kill")
_REQUIRED_COUNTERS = ("current", "peak", "swap_current", "swap_peak")
_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]*\.service\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _fail(message: str) -> None:
    raise ValueError(message)


def _unit(value: str) -> str:
    if type(value) is not str or not _UNIT.fullmatch(value):
        _fail("unit must be a strict .service leaf")
    return value


def _absolute(value: str) -> str:
    if type(value) is not str or not value.startswith("/") or "//" in value:
        _fail("path must have canonical absolute spelling")
    if (
        str(PurePosixPath(value)) != value
        or "/./" in value
        or "/../" in value
        or value.endswith(("/.", "/.."))
    ):
        _fail("path must have canonical absolute spelling")
    return value


def _parts(relative: str) -> tuple[str, ...]:
    if type(relative) is not str or not relative or relative.startswith("/"):
        _fail("unsafe relative path")
    result = tuple(relative.split("/"))
    if any(part in ("", ".", "..") for part in result):
        _fail("unsafe relative path")
    return result


def _open_absolute(path: str, flags: int, label: str) -> int:
    path = _absolute(path)
    root = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in _parts(path.lstrip("/"))[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root
            )
            os.close(root)
            root = child
        leaf = _parts(path.lstrip("/"))[-1]
        return os.open(leaf, flags | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise FileNotFoundError(path) from exc
        _fail(f"cannot open {label}: {exc.strerror}")
    finally:
        os.close(root)


def _open_dir(path: str, label: str) -> int:
    return _open_absolute(path, os.O_RDONLY | os.O_DIRECTORY, label)


def _open_child(parent: int, name: str, flags: int, label: str) -> int:
    if "/" in name or name in ("", ".", ".."):
        _fail(f"unsafe {label} leaf")
    try:
        return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise FileNotFoundError(name) from exc
        _fail(f"cannot open {label}: {exc.strerror}")


def _read_all(fd: int, limit: int = 64 * 1024 * 1024) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        try:
            chunk = os.read(fd, 1024 * 1024)
        except InterruptedError:
            continue
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            _fail("file exceeds limit")
        chunks.append(chunk)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0:
            _fail("short write")
        view = view[written:]


def _json(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def _parse_json(data: bytes, label: str) -> dict:
    def duplicate(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains duplicate key")
            result[key] = value
        return result

    def nonfinite(_value):
        _fail(f"{label} contains non-finite number")

    try:
        value = json.loads(data, object_pairs_hook=duplicate, parse_constant=nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"malformed {label}: {exc}")
    if not isinstance(value, dict) or _json(value) != data:
        _fail(f"non-canonical {label}")
    return value


def _read_regular_at(parent: int, leaf: str, label: str, mode: int | None = None) -> bytes:
    fd = _open_child(parent, leaf, os.O_RDONLY, label)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or (mode is not None and stat.S_IMODE(before.st_mode) != mode)
        ):
            _fail(f"unsafe {label}")
        data = _read_all(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (
        before.st_dev,
        before.st_ino,
        before.st_uid,
        before.st_gid,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_uid,
        after.st_gid,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail(f"{label} changed while read")
    return data


def _safe_directory(path: str, mode: int) -> tuple[int, os.stat_result]:
    fd = _open_dir(path, "directory")
    info = os.fstat(fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or stat.S_IMODE(info.st_mode) != mode
        or info.st_nlink < 2
    ):
        os.close(fd)
        _fail("unsafe evidence directory")
    return fd, info


def _fsync(fd: int) -> None:
    while True:
        try:
            os.fsync(fd)
            return
        except InterruptedError:
            continue


def _new_output(root_fd: int, leaf: str, mode: int) -> int:
    if "/" in leaf or leaf in ("", ".", ".."):
        _fail("output must be a direct child")
    try:
        return os.open(
            leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            mode,
            dir_fd=root_fd,
        )
    except OSError as exc:
        raise (
            FileExistsError(leaf)
            if exc.errno == errno.EEXIST
            else ValueError(f"cannot create output: {exc.strerror}")
        ) from exc


def _append(fd: int, value: dict) -> None:
    _write_all(fd, _json(value))
    _fsync(fd)


def _relative_from_proc(unit: str, proc_cgroup: str) -> str:
    fd = _open_absolute(proc_cgroup, os.O_RDONLY, "proc cgroup")
    try:
        data = _read_all(fd, 1024 * 1024)
    finally:
        os.close(fd)
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError:
        _fail("malformed proc cgroup")
    values = [
        line.split(":", 2)[2]
        for line in text.splitlines()
        if len(line.split(":", 2)) == 3 and line.split(":", 2)[:2] == ["0", ""]
    ]
    if len(values) != 1 or not values[0].startswith("/"):
        _fail("missing unified cgroup")
    relative = values[0].lstrip("/")
    _parts(relative)
    if not relative.endswith("/" + unit) and relative != unit:
        _fail("process is not in expected unit cgroup")
    return relative


def _derived_relative(unit: str, uid: int | None = None) -> str:
    uid = os.getuid() if uid is None else uid
    return f"user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{_unit(unit)}"


def _cgroup_dir(root: str, relative: str) -> int:
    root_fd = _open_dir(root, "cgroup root")
    try:
        for part in _parts(relative):
            child = _open_child(root_fd, part, os.O_RDONLY | os.O_DIRECTORY, "cgroup component")
            os.close(root_fd)
            root_fd = child
        info = os.fstat(root_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 2:
            _fail("unsafe cgroup directory")
        return root_fd
    except BaseException:
        os.close(root_fd)
        raise


def _cg_text(directory: int, leaf: str) -> str:
    fd = _open_child(directory, leaf, os.O_RDONLY, "cgroup metric")
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink < 1:
            _fail("unsafe cgroup metric")
        data = _read_all(fd, 1024 * 1024)
    finally:
        os.close(fd)
    try:
        return data.decode("utf-8", "strict")
    except UnicodeDecodeError:
        _fail("malformed cgroup metric")


def _parse_numbers(text: str, required=()) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0].rstrip(":") in values:
            _fail("malformed cgroup metric")
        try:
            value = int(fields[1])
        except ValueError:
            _fail("non-numeric cgroup metric")
        if value < 0:
            _fail("negative cgroup metric")
        values[fields[0].rstrip(":")] = value
    if any(key not in values for key in required):
        _fail("missing cgroup metric")
    return values


def _metrics(relative: str, cgroup_root: str = "/sys/fs/cgroup") -> dict:
    fd = _cgroup_dir(cgroup_root, relative)
    try:
        try:
            events = _parse_numbers(_cg_text(fd, "memory.events"), _EVENTS)
            if _cg_text(fd, "memory.oom.group").strip() != "1":
                _fail("memory.oom.group must be 1")
            values = {
                name: _parse_scalar(_cg_text(fd, "memory." + name.replace("_", ".")))
                for name in _REQUIRED_COUNTERS
            }
        except FileNotFoundError as exc:
            _fail(f"missing cgroup metric: {exc}")
        return {**values, "oom_group": 1, "events": {name: events[name] for name in _EVENTS}}
    finally:
        os.close(fd)


def _parse_scalar(text: str) -> int:
    try:
        value = int(text.strip())
    except ValueError:
        _fail("non-numeric cgroup metric")
    if value < 0:
        _fail("negative cgroup metric")
    return value


def _terminal(
    parent_fd: int, leaf: str, unit: str, relative: str, limits: tuple[int, int]
) -> tuple[dict, str]:
    raw = _read_regular_at(parent_fd, leaf, "terminal capture", 0o400)
    value = _parse_json(raw, "terminal capture")
    required = {
        "schema",
        "unit",
        "cgroup_relative_path",
        "oom_group",
        *_REQUIRED_COUNTERS,
        "events",
        "service_result",
        "exit_code",
        "exit_status",
    }
    if (
        set(value) != required
        or value["schema"] != CAPTURE_SCHEMA
        or value["unit"] != unit
        or value["cgroup_relative_path"] != relative
        or value["oom_group"] != 1
        or not isinstance(value["events"], dict)
    ):
        _fail("wrong terminal capture")
    if set(value["events"]) != set(_EVENTS) or any(
        type(value["events"][key]) is not int or value["events"][key] < 0 for key in _EVENTS
    ):
        _fail("incomplete terminal counters")
    if any(type(value[key]) is not int or value[key] < 0 for key in _REQUIRED_COUNTERS):
        _fail("incomplete terminal counters")
    if (value["service_result"], value["exit_code"], value["exit_status"]) != (
        "success",
        "exited",
        "0",
    ):
        _fail("terminal service did not exit successfully")
    if (
        any(value["events"][key] != 0 for key in ("max", "oom", "oom_kill", "oom_group_kill"))
        or value["peak"] >= limits[0]
        or value["swap_peak"] >= limits[1]
    ):
        _fail("terminal resource limit failure")
    return value, hashlib.sha256(raw).hexdigest()


def _request(path: str) -> tuple[dict, str]:
    data = _read_path_regular(path, "request", 0o600)
    value = _parse_json(data, "request")
    if (
        value.get("schema") != "alpha_max_terminal_request.acquisition.v1"
        or value.get("scope") != "acquisition"
        or not isinstance(value.get("request_id"), str)
        or not _DIGEST.fullmatch(value["request_id"])
    ):
        _fail("invalid acquisition request")
    return value, hashlib.sha256(data).hexdigest()


def _bound_receipt_path(request: dict) -> str:
    evidence_root = request.get("evidence_root")
    publication = request.get("publication")
    if (
        not isinstance(evidence_root, dict)
        or set(evidence_root) != {"path", "st_dev", "st_ino", "st_uid", "st_gid", "mode"}
        or not isinstance(publication, dict)
        or type(evidence_root["path"]) is not str
        or type(publication.get("receipt")) is not str
    ):
        _fail("request lacks acquisition receipt publication")
    root = _absolute(evidence_root["path"])
    receipt = publication["receipt"]
    if "/" in receipt or receipt in ("", ".", ".."):
        _fail("unsafe acquisition receipt leaf")
    return root + "/" + receipt


def _native_receipt(
    receipt_path: str,
    key_path: str,
    request: dict,
    request_sha: str,
    key_identity: tuple[int, ...] | None = None,
    key_digest: str | None = None,
) -> tuple[dict, str]:
    raw = _read_path_regular(receipt_path, "signed terminal receipt")
    if key_identity is not None and _authority_public_key(key_path) != (
        key_identity,
        key_digest,
    ):
        _fail("authority public key changed since preflight")
    try:
        verified = verify_signed_receipt(receipt_path, key_path)
    except (TerminalPolicyError, OSError, ValueError) as exc:
        _fail(f"native signed-terminal-receipt validation failed: {exc}")
    if key_identity is not None and _authority_public_key(key_path) != (
        key_identity,
        key_digest,
    ):
        _fail("authority public key changed during native verification")
    if key_identity is not None and verified.key_id != key_digest:
        _fail("native signed-terminal-receipt authority key mismatch")
    value = verified.message
    if _json(value) != raw:
        _fail("signed terminal receipt canonical message mismatch")
    if (
        value.get("scope") != "acquisition"
        or value.get("request_id") != request["request_id"]
        or value.get("request_sha256") != request_sha
        or value.get("terminal_state") != {"kind": "SUCCEEDED"}
    ):
        _fail("signed terminal receipt request binding mismatch")
    return value, hashlib.sha256(raw).hexdigest()


def _read_path_regular(path: str, label: str, mode: int = 0o400) -> bytes:
    fd = _open_absolute(path, os.O_RDONLY, label)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or stat.S_IMODE(before.st_mode) != mode
        ):
            _fail(f"unsafe {label}")
        data = _read_all(fd)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_uid,
            before.st_gid,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_uid,
            after.st_gid,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(f"{label} changed while read")
        return data
    finally:
        os.close(fd)


def _authority_public_key(path: str) -> tuple[tuple[int, ...], str]:
    try:
        fd = _open_absolute(path, os.O_RDONLY, "authority public key")
    except FileNotFoundError:
        _fail("missing authority public key")
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_gid != os.getgid()
            or stat.S_IMODE(before.st_mode) != 0o400
        ):
            _fail("unsafe authority public key")
        data = _read_all(fd, 33)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_uid,
        before.st_gid,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_uid,
        after.st_gid,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _fail("authority public key changed while read")
    if len(data) != 32:
        _fail("unsafe authority public key")
    return identity, hashlib.sha256(data).hexdigest()


def capture(args) -> int:
    unit = _unit(args.expected_unit)
    output = _absolute(args.output)
    parent, _info = _safe_directory(os.path.dirname(output), 0o700)
    try:
        relative = _relative_from_proc(unit, args.proc_cgroup)
        data = _metrics(relative, args.cgroup_root)
        result = {
            name: os.environ.get(name) for name in ("SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS")
        }
        if any(value is None for value in result.values()):
            _fail("capture requires ExecStopPost result environment")
        fd = _new_output(parent, os.path.basename(output), 0o400)
        try:
            _write_all(
                fd,
                _json(
                    {
                        "schema": CAPTURE_SCHEMA,
                        "unit": unit,
                        "cgroup_relative_path": relative,
                        **data,
                        "service_result": result["SERVICE_RESULT"],
                        "exit_code": result["EXIT_CODE"],
                        "exit_status": result["EXIT_STATUS"],
                    }
                ),
            )
            _fsync(fd)
        finally:
            os.close(fd)
        _fsync(parent)
    finally:
        os.close(parent)
    return 0


def monitor(args, *, clock=time.monotonic, sleep=time.sleep) -> int:
    authority, observer = _unit(args.authority_unit), _unit(args.observer_unit)
    if authority == observer:
        _fail("authority and observer units must differ")
    root_path, output_path = _absolute(args.evidence_root), _absolute(args.output)
    root, _root_info = _safe_directory(root_path, 0o700)
    try:
        paths = {
            "authority": _absolute(args.authority_terminal),
            "observer": _absolute(args.observer_terminal),
            "output": output_path,
        }
        if any(os.path.dirname(path) != root_path for path in paths.values()):
            _fail("telemetry evidence must be direct children of its private root")
        leaves = {name: os.path.basename(path) for name, path in paths.items()}
        if len(set(leaves.values())) != len(leaves):
            _fail("telemetry evidence leaves alias")
        request, request_sha = _request(_absolute(args.request))
        signed_path = _absolute(args.signed_terminal_receipt)
        if signed_path.startswith(root_path + "/"):
            _fail("signed terminal receipt must be outside telemetry root")
        if signed_path != _bound_receipt_path(request):
            _fail("signed terminal receipt is not the request-bound acquisition publication")
        limits = {
            authority: (args.authority_memory_max, args.authority_swap_max),
            observer: (args.observer_memory_max, args.observer_swap_max),
        }
        relatives = {authority: _derived_relative(authority), observer: _derived_relative(observer)}
        fd = _new_output(root, leaves["output"], 0o600)
        final: dict | None = None
        try:
            key_path = ""
            key_identity = None
            key_digest = None
            key_problem = None
            try:
                key_path = _absolute(args.authority_public_key)
                key_identity, key_digest = _authority_public_key(key_path)
            except Exception as exc:
                key_problem = str(exc)
            deadline = clock() + args.timeout_seconds
            while True:
                sample = {
                    "schema": SAMPLE_SCHEMA,
                    "authority_unit": authority,
                    "observer_unit": observer,
                    "units": {},
                }
                if key_problem is not None:
                    sample["terminal_state"] = "invalid"
                    sample["reason"] = key_problem
                    _append(fd, sample)
                    final = {
                        "schema": FINAL_SCHEMA,
                        "outcome": "failure",
                        "reason": key_problem,
                    }
                    break
                live_problem = None
                gone = True
                for unit in (authority, observer):
                    try:
                        metrics = _metrics(relatives[unit], args.cgroup_root)
                    except FileNotFoundError:
                        sample["units"][unit] = {"state": "absent_unit_cgroup"}
                    except Exception as exc:
                        sample["units"][unit] = {"state": "invalid", "reason": str(exc)}
                        live_problem = str(exc)
                        gone = False
                    else:
                        sample["units"][unit] = {"state": "present", **metrics}
                        gone = False
                        if (
                            metrics["events"]["max"]
                            or metrics["events"]["oom"]
                            or metrics["events"]["oom_kill"]
                            or metrics["events"]["oom_group_kill"]
                            or metrics["peak"] >= limits[unit][0]
                            or metrics["swap_peak"] >= limits[unit][1]
                        ):
                            live_problem = "live resource limit failure"
                terminal_error = None
                terminal_hashes = {}
                signed_hash = None
                try:
                    _authority_terminal, terminal_hashes[authority] = _terminal(
                        root,
                        leaves["authority"],
                        authority,
                        relatives[authority],
                        limits[authority],
                    )
                    _observer_terminal, terminal_hashes[observer] = _terminal(
                        root, leaves["observer"], observer, relatives[observer], limits[observer]
                    )
                    _receipt, signed_hash = _native_receipt(
                        signed_path,
                        key_path,
                        request,
                        request_sha,
                        key_identity,
                        key_digest,
                    )
                except FileNotFoundError:
                    sample["terminal_state"] = "pending"
                except Exception as exc:
                    terminal_error = str(exc)
                    sample["terminal_state"] = "invalid"
                else:
                    sample["terminal_state"] = "valid"
                _append(fd, sample)
                if live_problem or terminal_error:
                    final = {
                        "schema": FINAL_SCHEMA,
                        "outcome": "failure",
                        "reason": live_problem or terminal_error,
                    }
                    break
                if gone and signed_hash is not None:
                    final = {
                        "schema": FINAL_SCHEMA,
                        "outcome": "success",
                        "authority_unit": authority,
                        "observer_unit": observer,
                        "request_id": request["request_id"],
                        "request_sha256": request_sha,
                        "authority_cgroup_relative_path": relatives[authority],
                        "observer_cgroup_relative_path": relatives[observer],
                        "authority_terminal_path": paths["authority"],
                        "authority_terminal_sha256": terminal_hashes[authority],
                        "observer_terminal_path": paths["observer"],
                        "observer_terminal_sha256": terminal_hashes[observer],
                        "signed_terminal_receipt_path": signed_path,
                        "signed_terminal_receipt_sha256": signed_hash,
                    }
                    break
                if clock() >= deadline:
                    final = {
                        "schema": FINAL_SCHEMA,
                        "outcome": "timeout",
                        "reason": "deadline exceeded",
                    }
                    break
                sleep(args.interval_seconds)
            _append(fd, final)
        except BaseException as exc:
            final = {
                "schema": FINAL_SCHEMA,
                "outcome": "failure",
                "reason": f"unexpected monitor exception: {type(exc).__name__}: {exc}",
            }
            try:
                _append(fd, final)
            except BaseException as finalization_error:
                raise BaseExceptionGroup(
                    "monitor emergency finalization failed", [exc, finalization_error]
                ) from exc
        finally:
            try:
                try:
                    os.fchmod(fd, 0o400)
                    _fsync(fd)
                finally:
                    os.close(fd)
            finally:
                _fsync(root)
        return 0 if final["outcome"] == "success" else 1
    finally:
        os.close(root)


def _positive(value: str, maximum: int) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= number <= maximum:
        raise argparse.ArgumentTypeError("out of range")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--expected-unit", required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.set_defaults(
        proc_cgroup="/proc/self/cgroup", cgroup_root="/sys/fs/cgroup", handler=capture
    )
    monitor_parser = commands.add_parser("monitor")
    for name in (
        "authority_unit",
        "observer_unit",
        "evidence_root",
        "authority_terminal",
        "observer_terminal",
        "signed_terminal_receipt",
        "authority_public_key",
        "request",
        "output",
    ):
        monitor_parser.add_argument("--" + name.replace("_", "-"), required=True)
    monitor_parser.add_argument(
        "--interval-seconds", required=True, type=lambda value: _positive(value, 60)
    )
    monitor_parser.add_argument(
        "--timeout-seconds", required=True, type=lambda value: _positive(value, 86400)
    )
    for name in (
        "authority_memory_max",
        "authority_swap_max",
        "observer_memory_max",
        "observer_swap_max",
    ):
        monitor_parser.add_argument(
            "--" + name.replace("_", "-"),
            required=True,
            type=lambda value: _positive(value, 2**63 - 1),
        )
    monitor_parser.set_defaults(cgroup_root="/sys/fs/cgroup", handler=monitor)
    return result


def main(argv=None) -> int:
    try:
        args = parser().parse_args(argv)
        return args.handler(args)
    except (ValueError, OSError) as exc:
        print("resource evidence error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
