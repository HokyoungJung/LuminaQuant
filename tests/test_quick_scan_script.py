from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "scripts" / "quick_scan.py"
    spec = importlib.util.spec_from_file_location("quick_scan_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load quick_scan module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_quick_profile_plan_uses_curated_tests():
    plan = MODULE._command_plan("quick", 1, False)
    assert plan[0] == ["uv", "run", "ruff", "check", "."]
    assert "tests/test_phase1_research_script.py" in plan[1]
    assert "tests/test_data_sync.py" in plan[1]


def test_full_profile_plan_uses_full_pytest_and_build_optional():
    plan = MODULE._command_plan("full", 2, True)
    assert plan[1] == ["uv", "run", "pytest", "-q", "--maxfail", "2"]
    assert plan[2] == ["uv", "build"]
