from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_install.py"
_SPEC = importlib.util.spec_from_file_location("verify_install_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {_SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = MODULE
_SPEC.loader.exec_module(MODULE)


def test_verify_install_syncs_gpu_extra_and_runs_gpu_contract(monkeypatch):
    commands: list[list[str]] = []

    monkeypatch.setattr(MODULE, "run", lambda cmd: commands.append(list(cmd)))
    monkeypatch.setattr(MODULE, "run_optional", lambda cmd: None)
    monkeypatch.setattr(MODULE.platform, "platform", lambda: "TestOS")
    monkeypatch.setattr(MODULE.shutil, "which", lambda name: None)

    MODULE.main()

    assert commands[0] == [
        "uv",
        "sync",
        "--extra",
        "optimize",
        "--extra",
        "dev",
        "--extra",
        "live",
        "--extra",
        "gpu",
    ]
    assert [
        "uv",
        "run",
        "python",
        "scripts/ci/verify_polars_gpu_runtime.py",
        "--output-json",
        "reports/benchmarks/verify_install_gpu_contract.json",
    ] in commands


def test_verify_install_runs_strict_gpu_check_when_nvidia_smi_exists(monkeypatch):
    commands: list[list[str]] = []
    optional_commands: list[list[str]] = []

    monkeypatch.setattr(MODULE, "run", lambda cmd: commands.append(list(cmd)))
    monkeypatch.setattr(
        MODULE,
        "run_optional",
        lambda cmd: optional_commands.append(list(cmd)) or type("Result", (), {"returncode": 0})(),
    )
    monkeypatch.setattr(MODULE.platform, "platform", lambda: "TestOS")
    monkeypatch.setattr(MODULE.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    MODULE.main()

    assert optional_commands == [
        [
            "uv",
            "run",
            "python",
            "scripts/ci/verify_polars_gpu_runtime.py",
            "--require-gpu",
            "--mode",
            "forced-gpu",
            "--output-json",
            "reports/benchmarks/verify_install_gpu_runtime.json",
        ]
    ]
