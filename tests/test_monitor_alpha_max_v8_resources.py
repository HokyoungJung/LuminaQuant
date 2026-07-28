import ast
import argparse
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/research/monitor_alpha_max_v8_resources.py"
spec = importlib.util.spec_from_file_location("v8_resources", SCRIPT)
v8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8)


def put(path, text, mode=0o600):
    if path.exists():
        path.chmod(0o600)
    path.write_text(text)
    path.chmod(mode)


def cgroup(
    root,
    relative,
    events="low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
    group="1\n",
    peak="10\n",
    swap_peak="5\n",
):
    target = root / relative
    target.mkdir(parents=True, exist_ok=True)
    for name, content in {
        "memory.events": events,
        "memory.oom.group": group,
        "memory.current": "1\n",
        "memory.peak": peak,
        "memory.swap.current": "2\n",
        "memory.swap.peak": swap_peak,
    }.items():
        put(target / name, content)
    return target


def capture_receipt(path, unit, **extra):
    value = {
        "schema": v8.CAPTURE_SCHEMA,
        "unit": unit,
        "cgroup_relative_path": v8._derived_relative(unit),
        "oom_group": 1,
        "current": 1,
        "peak": 10,
        "swap_current": 1,
        "swap_peak": 5,
        "events": dict.fromkeys(v8._EVENTS, 0),
        "service_result": "success",
        "exit_code": "exited",
        "exit_status": "0",
    }
    value.update(extra)
    put(path, v8._json(value).decode(), 0o400)


def request(path, acquisition_root):
    value = {
        "schema": "alpha_max_terminal_request.acquisition.v1",
        "scope": "acquisition",
        "request_id": "a" * 64,
        "evidence_root": {
            "path": str(acquisition_root),
            "st_dev": 1,
            "st_ino": 1,
            "st_uid": os.getuid(),
            "st_gid": os.getgid(),
            "mode": 0o700,
        },
        "publication": {"receipt": "terminal-authority.receipt.json"},
    }
    put(path, v8._json(value).decode(), 0o600)
    return value


def test_canonical_json_rejects_duplicates_nonfinite_and_noncanonical():
    for data in (b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{ "x":1}\n', b"[]\n"):
        with pytest.raises(ValueError):
            v8._parse_json(data, "test")
    assert v8._parse_json(b'{"x":1}\n', "test") == {"x": 1}


def test_absolute_and_component_access_reject_traversal_and_ancestor_link(tmp_path):
    for path in ("relative", "/tmp//x", "/tmp/./x", "/tmp/a/../x"):
        with pytest.raises(ValueError):
            v8._absolute(path)
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        v8._open_dir(str(tmp_path / "link"), "link")


def test_metrics_complete_distinguishes_missing_unit_and_missing_leaf(tmp_path):
    root = tmp_path / "cg"
    root.mkdir()
    cgroup(root, "u.service")
    assert v8._metrics("u.service", str(root))["peak"] == 10
    (root / "u.service" / "memory.peak").unlink()
    with pytest.raises(ValueError, match="cgroup metric"):
        v8._metrics("u.service", str(root))
    with pytest.raises(FileNotFoundError):
        v8._metrics("gone.service", str(root))


def test_metrics_reject_oom_group_events_and_limit_conditions(tmp_path):
    root = tmp_path / "cg"
    root.mkdir()
    target = cgroup(root, "u.service", group="0\n")
    with pytest.raises(ValueError):
        v8._metrics("u.service", str(root))
    cgroup(root, "u.service", events="low 0\nhigh 0\nmax 1\noom 0\noom_kill 0\noom_group_kill 0\n")
    assert v8._metrics("u.service", str(root))["events"]["max"] == 1
    (root / "link.service").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        v8._metrics("link.service", str(root))


def test_terminal_requires_exact_success_complete_counters_and_strict_limits(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    capture_receipt(root / "a", "a.service")
    fd = os.open(root, os.O_RDONLY)
    try:
        assert (
            v8._terminal(fd, "a", "a.service", v8._derived_relative("a.service"), (11, 6))[0][
                "unit"
            ]
            == "a.service"
        )
    finally:
        os.close(fd)
    capture_receipt(root / "a", "a.service", peak=11)
    fd = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(ValueError):
            v8._terminal(fd, "a", "a.service", v8._derived_relative("a.service"), (11, 6))
    finally:
        os.close(fd)
    capture_receipt(root / "a", "a.service", service_result="failed")
    fd = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(ValueError):
            v8._terminal(fd, "a", "a.service", v8._derived_relative("a.service"), (12, 6))
    finally:
        os.close(fd)
    capture_receipt(root / "a", "a.service", cgroup_relative_path="other.service")
    fd = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="wrong terminal capture"):
            v8._terminal(fd, "a", "a.service", v8._derived_relative("a.service"), (12, 6))
    finally:
        os.close(fd)
    capture_receipt(root / "a", "a.service", oom_group=0)
    fd = os.open(root, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="wrong terminal capture"):
            v8._terminal(fd, "a", "a.service", v8._derived_relative("a.service"), (12, 6))
    finally:
        os.close(fd)


def test_capture_binds_actual_proc_cgroup_and_is_durable(tmp_path, monkeypatch):
    root = tmp_path / "cg"
    root.mkdir()
    cgroup(root, "u.service")
    proc = tmp_path / "proc"
    put(proc, "0::/u.service\n")
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    monkeypatch.setenv("SERVICE_RESULT", "success")
    monkeypatch.setenv("EXIT_CODE", "exited")
    monkeypatch.setenv("EXIT_STATUS", "0")
    out = evidence / "terminal.json"
    assert (
        v8.capture(
            argparse.Namespace(
                expected_unit="u.service",
                output=str(out),
                proc_cgroup=str(proc),
                cgroup_root=str(root),
            )
        )
        == 0
    )
    receipt = v8._parse_json(out.read_bytes(), "receipt")
    assert receipt["unit"] == "u.service"
    assert receipt["cgroup_relative_path"] == "u.service"
    assert receipt["oom_group"] == 1
    assert stat.S_IMODE(out.stat().st_mode) == 0o400


def test_native_receipt_adapter_binds_request_and_rejects_tamper(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt"
    key = tmp_path / "key"
    put(key, "k", 0o400)
    req = {"request_id": "a" * 64}
    sha = "b" * 64

    class Verified:
        message = {
            "scope": "acquisition",
            "request_id": req["request_id"],
            "request_sha256": sha,
            "terminal_state": {"kind": "SUCCEEDED"},
        }

    put(receipt, v8._json(Verified.message).decode(), 0o400)
    monkeypatch.setattr(v8, "verify_signed_receipt", lambda path, public: Verified())
    value, digest = v8._native_receipt(str(receipt), str(key), req, sha)
    assert (
        value["scope"] == "acquisition"
        and digest == hashlib.sha256(receipt.read_bytes()).hexdigest()
    )
    Verified.message = {**Verified.message, "replacement": True}
    with pytest.raises(ValueError, match="canonical message mismatch"):
        v8._native_receipt(str(receipt), str(key), req, sha)


def test_native_receipt_adapter_uses_real_native_verifier(tmp_path):
    policy_test = Path(__file__).with_name("test_alpha_max_terminal_policy.py")
    policy_spec = importlib.util.spec_from_file_location("native_policy_cases", policy_test)
    policy_cases = importlib.util.module_from_spec(policy_spec)
    policy_spec.loader.exec_module(policy_cases)
    case = policy_cases._signed_receipt_case(tmp_path, {"kind": "SUCCEEDED"})
    request_id = case["receipt"]["request_id"]
    request_sha = case["receipt"]["request_sha256"]
    policy_cases._write_signed_receipt(case)
    case["receipt_path"].chmod(0o400)
    case["public_key"].chmod(0o400)

    value, digest = v8._native_receipt(
        str(case["receipt_path"]), str(case["public_key"]), {"request_id": request_id}, request_sha
    )

    assert value == case["receipt"]
    assert digest == hashlib.sha256(case["receipt_path"].read_bytes()).hexdigest()


def test_monitor_missing_signed_receipt_remains_pending(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    acquisition_root = tmp_path / "acquisition"
    acquisition_root.mkdir(mode=0o700)
    request(root / "request", acquisition_root)
    put(root / "key", "k" * 32, 0o400)
    capture_receipt(root / "authority", "authority.service")
    capture_receipt(root / "observer", "observer.service")
    ticks = iter((0, 2))

    assert (
        v8.monitor(
            _monitor_args(root, cgroup_root, acquisition_root),
            clock=lambda: next(ticks),
            sleep=lambda _: None,
        )
        == 1
    )

    lines = [json.loads(line) for line in (root / "stream").read_text().splitlines()]
    assert lines[0]["terminal_state"] == "pending"
    assert lines[-1]["outcome"] == "timeout"


def test_monitor_missing_authority_key_is_immediate_durable_failure(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    acquisition_root = tmp_path / "acquisition"
    acquisition_root.mkdir(mode=0o700)
    request_path = root / "request"
    request(request_path, acquisition_root)
    capture_receipt(root / "authority", "authority.service")
    capture_receipt(root / "observer", "observer.service")
    message = {
        "scope": "acquisition",
        "request_id": "a" * 64,
        "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
        "terminal_state": {"kind": "SUCCEEDED"},
    }
    put(acquisition_root / "terminal-authority.receipt.json", v8._json(message).decode(), 0o400)
    calls = []

    def native_verifier(receipt_path, key_path):
        calls.append((receipt_path, key_path))
        raise v8.TerminalPolicyError("authority public key missing")

    monkeypatch.setattr(v8, "verify_signed_receipt", native_verifier)
    assert v8.monitor(_monitor_args(root, cgroup_root, acquisition_root), clock=lambda: 0) == 1

    lines = [json.loads(line) for line in (root / "stream").read_text().splitlines()]
    assert not calls and lines[0]["terminal_state"] == "invalid"
    assert lines[-1]["outcome"] == "failure"
    assert stat.S_IMODE((root / "stream").stat().st_mode) == 0o400


def test_monitor_missing_key_and_dynamic_evidence_fails_without_sleep(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    acquisition_root = tmp_path / "acquisition"
    acquisition_root.mkdir(mode=0o700)
    request(root / "request", acquisition_root)

    assert (
        v8.monitor(
            _monitor_args(root, cgroup_root, acquisition_root),
            clock=lambda: 0,
            sleep=lambda _: pytest.fail("missing key must not sleep"),
        )
        == 1
    )

    lines = [json.loads(line) for line in (root / "stream").read_text().splitlines()]
    assert lines[0]["terminal_state"] == "invalid"
    assert "authority public key" in lines[0]["reason"]
    assert lines[-1]["outcome"] == "failure"
    assert "authority public key" in lines[-1]["reason"]
    assert stat.S_IMODE((root / "stream").stat().st_mode) == 0o400


def _monitor_args(root, cg, acquisition_root, **overrides):
    values = dict(
        authority_unit="authority.service",
        observer_unit="observer.service",
        evidence_root=str(root),
        authority_terminal=str(root / "authority"),
        observer_terminal=str(root / "observer"),
        signed_terminal_receipt=str(acquisition_root / "terminal-authority.receipt.json"),
        authority_public_key=str(root / "key"),
        request=str(root / "request"),
        output=str(root / "stream"),
        interval_seconds=1,
        timeout_seconds=1,
        authority_memory_max=11,
        authority_swap_max=6,
        observer_memory_max=11,
        observer_swap_max=6,
        cgroup_root=str(cg),
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_monitor_pending_then_success_hashes_all_terminal_evidence(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    cg = tmp_path / "cg"
    cg.mkdir()
    authority_cgroup = cgroup(cg, v8._derived_relative("authority.service"))
    observer_cgroup = cgroup(cg, v8._derived_relative("observer.service"))
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir(mode=0o700)
    request_path = root / "request"
    request(request_path, acquisition)
    key = root / "key"
    put(key, "k" * 32, 0o400)
    signed = acquisition / "terminal-authority.receipt.json"
    snapshots = {}

    class Verified:
        message = {
            "scope": "acquisition",
            "request_id": "a" * 64,
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "terminal_state": {"kind": "SUCCEEDED"},
        }
        key_id = hashlib.sha256(key.read_bytes()).hexdigest()

    def identity(path):
        info = path.stat()
        return info.st_dev, info.st_ino, path.read_bytes() if path.is_file() else None

    def finish_units(_interval):
        for target in (authority_cgroup, observer_cgroup):
            for metric in target.iterdir():
                metric.unlink()
            target.rmdir()
        capture_receipt(root / "authority", "authority.service")
        capture_receipt(root / "observer", "observer.service")
        put(signed, v8._json(Verified.message).decode(), 0o400)
        for path in (request_path, key, cg, root / "authority", root / "observer", signed):
            snapshots[path] = identity(path)

    monkeypatch.setattr(v8, "verify_signed_receipt", lambda *_: Verified())
    assert (
        v8.monitor(_monitor_args(root, cg, acquisition), clock=lambda: 0, sleep=finish_units) == 0
    )
    lines = [json.loads(line) for line in (root / "stream").read_text().splitlines()]
    assert [line["terminal_state"] for line in lines[:-1]] == ["pending", "valid"]
    assert lines[-1]["outcome"] == "success"
    assert lines[-1]["signed_terminal_receipt_path"] == str(signed)
    assert set(lines[-1]) >= {
        "authority_terminal_sha256",
        "observer_terminal_sha256",
        "signed_terminal_receipt_sha256",
    }
    assert snapshots == {path: identity(path) for path in snapshots}


def test_monitor_pending_leaves_and_malformed_terminal_are_durable(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    cg = tmp_path / "cg"
    cg.mkdir()
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir(mode=0o700)
    request(root / "request", acquisition)
    put(root / "key", "k" * 32, 0o400)
    ticks = iter((0, 2))
    assert (
        v8.monitor(
            _monitor_args(root, cg, acquisition), clock=lambda: next(ticks), sleep=lambda _: None
        )
        == 1
    )
    lines = [json.loads(line) for line in (root / "stream").read_text().splitlines()]
    assert lines[0]["terminal_state"] == "pending" and lines[-1]["outcome"] == "timeout"
    root2 = tmp_path / "evidence2"
    root2.mkdir(mode=0o700)
    acquisition2 = tmp_path / "acquisition2"
    acquisition2.mkdir(mode=0o700)
    request(root2 / "request", acquisition2)
    put(root2 / "key", "k" * 32, 0o400)
    put(root2 / "authority", "bad\n", 0o400)
    capture_receipt(root2 / "observer", "observer.service")
    put(acquisition2 / "terminal-authority.receipt.json", "{}\n", 0o400)
    tracked = (
        root2 / "request",
        root2 / "key",
        cg,
        root2 / "authority",
        root2 / "observer",
        acquisition2 / "terminal-authority.receipt.json",
    )
    before = {
        path: (
            path.stat().st_dev,
            path.stat().st_ino,
            path.read_bytes() if path.is_file() else None,
        )
        for path in tracked
    }
    assert (
        v8.monitor(_monitor_args(root2, cg, acquisition2), clock=lambda: 0, sleep=lambda _: None)
        == 1
    )
    malformed = [json.loads(line) for line in (root2 / "stream").read_text().splitlines()]
    assert malformed[0]["terminal_state"] == "invalid"
    assert malformed[0]["reason"] == malformed[-1]["reason"]
    assert malformed[-1]["outcome"] == "failure"
    after = {
        path: (
            path.stat().st_dev,
            path.stat().st_ino,
            path.read_bytes() if path.is_file() else None,
        )
        for path in tracked
    }
    assert after == before


def test_monitor_rejects_aliases_request_mode_and_cli_includes_native_validator_inputs(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    cg = tmp_path / "cg"
    cg.mkdir()
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir(mode=0o700)
    request(root / "request", acquisition)
    assert stat.S_IMODE((root / "request").stat().st_mode) == 0o600
    args = _monitor_args(root, cg, acquisition, observer_terminal=str(root / "authority"))
    with pytest.raises(ValueError, match="alias"):
        v8.monitor(args, clock=lambda: 0, sleep=lambda _: None)
    with pytest.raises(ValueError, match="outside telemetry"):
        v8.monitor(
            _monitor_args(root, cg, acquisition, signed_terminal_receipt=str(root / "signed")),
            clock=lambda: 0,
            sleep=lambda _: None,
        )
    with pytest.raises(ValueError, match="request-bound"):
        v8.monitor(
            _monitor_args(
                root, cg, acquisition, signed_terminal_receipt=str(acquisition / "other.json")
            ),
            clock=lambda: 0,
            sleep=lambda _: None,
        )
    parsed = v8.parser().parse_args(
        [
            "monitor",
            "--authority-unit",
            "a.service",
            "--observer-unit",
            "b.service",
            "--evidence-root",
            "/x",
            "--authority-terminal",
            "/x/a",
            "--observer-terminal",
            "/x/b",
            "--signed-terminal-receipt",
            "/x/s",
            "--authority-public-key",
            "/x/key",
            "--request",
            "/x/request",
            "--output",
            "/x/out",
            "--interval-seconds",
            "1",
            "--timeout-seconds",
            "1",
            "--authority-memory-max",
            "1",
            "--authority-swap-max",
            "1",
            "--observer-memory-max",
            "1",
            "--observer-swap-max",
            "1",
        ]
    )
    assert parsed.command == "monitor"
    tree = ast.parse(SCRIPT.read_text())
    denied_imports = {
        "ftplib",
        "http",
        "requests",
        "socket",
        "subprocess",
        "telnetlib",
        "urllib",
    }
    denied_calls = {
        "call",
        "check_call",
        "check_output",
        "connect",
        "popen",
        "request",
        "run",
        "socket",
        "system",
        "systemctl",
        "urlopen",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in denied_imports for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in denied_imports
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in denied_calls
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in denied_calls


def test_monitor_missing_metric_leaf_is_fatal_not_absent(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir(mode=0o700)
    cg = tmp_path / "cg"
    cg.mkdir()
    relative = v8._derived_relative("authority.service")
    cgroup(cg, relative)
    (cg / relative / "memory.peak").unlink()
    request(root / "request", acquisition)
    put(root / "key", "k" * 32, 0o400)
    assert (
        v8.monitor(_monitor_args(root, cg, acquisition), clock=lambda: 0, sleep=lambda _: None) == 1
    )
    lines = [json.loads(line) for line in (root / "stream").read_text().splitlines()]
    assert lines[0]["units"]["authority.service"]["state"] == "invalid"
    assert lines[-1]["outcome"] == "failure"


def test_monitor_unexpected_exception_finalizes_sealed_failure(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir(mode=0o700)
    cg = tmp_path / "cg"
    cg.mkdir()
    request(root / "request", acquisition)
    put(root / "key", "k" * 32, 0o400)
    append = v8._append

    def raise_after_sample(fd, value):
        append(fd, value)
        if value["schema"] == v8.SAMPLE_SCHEMA:
            raise RuntimeError("injected")

    monkeypatch.setattr(v8, "_append", raise_after_sample)
    assert (
        v8.monitor(_monitor_args(root, cg, acquisition), clock=lambda: 0, sleep=lambda _: None) == 1
    )
    lines = [json.loads(line) for line in (root / "stream").read_text().splitlines()]
    assert lines[-1]["outcome"] == "failure"
    assert "unexpected monitor exception" in lines[-1]["reason"]
    assert stat.S_IMODE((root / "stream").stat().st_mode) == 0o400


def test_monitor_emergency_finalization_preserves_double_append_failure(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    acquisition = tmp_path / "acquisition"
    acquisition.mkdir(mode=0o700)
    cg = tmp_path / "cg"
    cg.mkdir()
    request(root / "request", acquisition)
    put(root / "key", "k" * 32, 0o400)

    def fail_append(_fd, value):
        if value["schema"] == v8.SAMPLE_SCHEMA:
            raise RuntimeError("sample append failed")
        raise OSError("emergency append failed")

    monkeypatch.setattr(v8, "_append", fail_append)
    with pytest.raises(BaseExceptionGroup) as raised:
        v8.monitor(_monitor_args(root, cg, acquisition), clock=lambda: 0, sleep=lambda _: None)

    errors = raised.value.exceptions
    assert isinstance(errors[0], RuntimeError)
    assert isinstance(errors[1], OSError)
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert stat.S_IMODE((root / "stream").stat().st_mode) == 0o400
