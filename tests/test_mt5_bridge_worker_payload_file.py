"""End-to-end regression test for the MT5 bridge worker `--payload-file` contract.

Round-2 hardening moved MT5 credentials off argv into a private 0600 file and made
the exchange pass `--payload-file <path>`.  The worker (`scripts/mt5_bridge_worker.py`)
must accept and read that flag; if it does not, argparse rejects the unknown option
with exit code 2 and EVERY bridge action fails.  These tests invoke the REAL worker
as a subprocess (no mock can mask an argparse rejection).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "scripts" / "mt5_bridge_worker.py"


def _run_worker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(WORKER), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_worker_accepts_payload_file_flag_not_argparse_rejected():
    # argparse rejection would be exit code 2; the worker must accept --payload-file.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"login": 123, "password": "secret", "server": "Broker"}, fh)
        secret_path = fh.name
    try:
        proc = _run_worker(
            "--action",
            "definitely_unsupported_action",
            "--payload",
            "{}",
            "--payload-file",
            secret_path,
        )
    finally:
        Path(secret_path).unlink(missing_ok=True)

    assert proc.returncode != 2, f"argparse rejected --payload-file: {proc.stderr}"
    assert proc.returncode == 0, proc.stderr
    # Reached action dispatch (past arg parsing + file load + merge) => Unsupported action.
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert "Unsupported action" in str(payload.get("error"))


def test_worker_reports_invalid_payload_file():
    # Proves the worker actually opens/reads the file path it is given.
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write("{ this is : not valid json")
        bad_path = fh.name
    try:
        proc = _run_worker(
            "--action",
            "connect",
            "--payload",
            "{}",
            "--payload-file",
            bad_path,
        )
    finally:
        Path(bad_path).unlink(missing_ok=True)

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert "Invalid payload file" in str(payload.get("error"))


def test_worker_without_payload_file_still_works():
    # Backward compatibility: omitting --payload-file must not break arg parsing.
    proc = _run_worker("--action", "definitely_unsupported_action", "--payload", "{}")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert "Unsupported action" in str(payload.get("error"))
