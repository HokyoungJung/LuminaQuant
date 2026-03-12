from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_exact_window_suite.py"
    spec = importlib.util.spec_from_file_location("run_exact_window_suite_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_exact_window_suite module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_run_exact_window_suite_script_delegates_to_cli(monkeypatch):
    captured: dict[str, object] = {}

    def _stub_main(argv=None):
        captured["argv"] = list(argv or [])
        return 7

    monkeypatch.setattr(MODULE, "exact_window_main", _stub_main)
    rc = MODULE.main(["--emit-memory-baseline", "--output-dir", "var/reports/exact_window_backtests"])
    assert rc == 7
    assert captured["argv"] == [
        "--emit-memory-baseline",
        "--output-dir",
        "var/reports/exact_window_backtests",
    ]
