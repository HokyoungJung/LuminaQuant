from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "verify_polars_gpu_runtime.py"
_SPEC = importlib.util.spec_from_file_location("verify_polars_gpu_runtime_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {_SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = MODULE
_SPEC.loader.exec_module(MODULE)


def test_parser_defaults():
    args = MODULE.build_parser().parse_args([])
    assert args.require_gpu is False
    assert args.mode == "auto"
    assert args.device == ""
    assert args.min_vram_gb == 0.0
    assert args.rows == 4096


def test_run_check_skips_cleanly_without_gpu_when_not_required(monkeypatch):
    monkeypatch.setattr(MODULE, "detect_nvidia_gpu", lambda: (False, "nvidia-smi not found"))
    monkeypatch.setattr(MODULE.pl, "GPUEngine", object(), raising=False)

    rc, payload = MODULE.run_check(
        require_gpu=False,
        mode="auto",
        device=None,
        min_vram_gb=0.0,
        rows=128,
    )

    assert rc == 0
    assert payload["status"] == "skipped"


def test_run_check_fails_without_gpu_when_required(monkeypatch):
    monkeypatch.setattr(MODULE, "detect_nvidia_gpu", lambda: (True, "detected 1 GPU"))
    monkeypatch.setattr(MODULE, "polars_gpu_available", lambda **kwargs: (False, "gpu smoke failed"))
    monkeypatch.setattr(MODULE.pl, "GPUEngine", object(), raising=False)

    rc, payload = MODULE.run_check(
        require_gpu=True,
        mode="forced-gpu",
        device=0,
        min_vram_gb=0.0,
        rows=128,
    )

    assert rc == 1
    assert payload["status"] == "failed"
    assert payload["reason"] == "gpu smoke failed"


def test_run_check_passes_when_gpu_probe_and_strict_query_succeed(monkeypatch):
    monkeypatch.setattr(MODULE, "detect_nvidia_gpu", lambda: (True, "detected 1 GPU"))
    monkeypatch.setattr(MODULE, "polars_gpu_available", lambda **kwargs: (True, "gpu smoke passed"))
    monkeypatch.setattr(
        MODULE,
        "resolve_compute_engine",
        lambda **kwargs: MODULE.ComputeEngine(
            requested_mode="forced-gpu",
            resolved_engine="gpu",
            device=0,
            verbose=False,
            reason="gpu smoke passed",
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_run_strict_gpu_query",
        lambda **kwargs: {
            "gpu_rows": 32,
            "cpu_rows": 32,
            "matches_cpu": True,
            "buckets": 32,
        },
    )
    monkeypatch.setattr(MODULE.pl, "GPUEngine", object(), raising=False)

    rc, payload = MODULE.run_check(
        require_gpu=True,
        mode="forced-gpu",
        device=0,
        min_vram_gb=0.0,
        rows=128,
    )

    assert rc == 0
    assert payload["status"] == "passed"
    assert payload["resolved_engine"] == "gpu"


def test_main_writes_output_json(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        MODULE,
        "run_check",
        lambda **kwargs: (
            0,
            {
                "status": "passed",
                "reason": "ok",
            },
        ),
    )

    output_path = tmp_path / "gpu-runtime.json"
    rc = MODULE.main(["--output-json", str(output_path)])

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert '"status": "passed"' in capsys.readouterr().out
