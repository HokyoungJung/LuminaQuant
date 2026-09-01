from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_EXPORT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "ci" / "export_market_window_gate_metrics.py"
)
_EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_market_window_gate_metrics", _EXPORT_PATH
)
if _EXPORT_SPEC is None or _EXPORT_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_EXPORT_PATH}")
export_script = importlib.util.module_from_spec(_EXPORT_SPEC)
_EXPORT_SPEC.loader.exec_module(export_script)

_CHECK_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "ci" / "check_market_window_rollout_gates.py"
)
_CHECK_SPEC = importlib.util.spec_from_file_location(
    "check_market_window_rollout_gates", _CHECK_PATH
)
if _CHECK_SPEC is None or _CHECK_SPEC.loader is None:
    raise RuntimeError(f"Failed to load module from {_CHECK_PATH}")
check_script = importlib.util.module_from_spec(_CHECK_SPEC)
_CHECK_SPEC.loader.exec_module(check_script)


def _write_metrics(path: Path) -> None:
    rows = [
        {
            "timestamp_ms": 1_700_000_000_000,
            "payload_bytes": 20_000,
            "queue_lag_ms": 100,
            "parity_v2_enabled": False,
            "fail_fast_incident": False,
        },
        {
            "timestamp_ms": 1_700_000_000_500,
            "payload_bytes": 30_000,
            "queue_lag_ms": 110,
            "parity_v2_enabled": False,
            "fail_fast_incident": False,
        },
        {
            "timestamp_ms": 1_700_000_001_000,
            "payload_bytes": 60_000,
            "queue_lag_ms": 105,
            "parity_v2_enabled": True,
            "fail_fast_incident": False,
        },
        {
            "timestamp_ms": 1_700_000_001_500,
            "payload_bytes": 80_000,
            "queue_lag_ms": 108,
            "parity_v2_enabled": True,
            "fail_fast_incident": False,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_rollout_gate_metrics_export_and_check(tmp_path, monkeypatch):
    metrics_path = tmp_path / "market_window_metrics.ndjson"
    baseline_path = tmp_path / "baseline.json"
    canary_path = tmp_path / "canary.json"
    _write_metrics(metrics_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_market_window_gate_metrics.py",
            "--input",
            str(metrics_path),
            "--output",
            str(baseline_path),
            "--window-hours",
            "24",
            "--require-flag",
            "false",
        ],
    )
    assert export_script.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_market_window_gate_metrics.py",
            "--input",
            str(metrics_path),
            "--output",
            str(canary_path),
            "--window-hours",
            "24",
            "--require-flag",
            "true",
        ],
    )
    assert export_script.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_market_window_rollout_gates.py",
            "--baseline",
            str(baseline_path),
            "--canary",
            str(canary_path),
            "--max-p95-payload-bytes",
            "131072",
            "--max-queue-lag-increase-pct",
            "5",
            "--max-fail-fast-incidents",
            "0",
        ],
    )
    assert check_script.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_market_window_rollout_gates.py",
            "--baseline",
            str(baseline_path),
            "--canary",
            str(canary_path),
            "--max-p95-payload-bytes",
            "50000",
            "--max-queue-lag-increase-pct",
            "1",
            "--max-fail-fast-incidents",
            "0",
        ],
    )
    assert check_script.main() == 1
