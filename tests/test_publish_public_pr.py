from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "publish_public_pr.py"
_SPEC = importlib.util.spec_from_file_location("publish_public_pr_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {_SCRIPT_PATH}")
publish_public_pr = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = publish_public_pr
_SPEC.loader.exec_module(publish_public_pr)


def test_sensitive_path_detection_blocks_private_trees():
    assert publish_public_pr.is_sensitive_path("lumina_quant/strategies/rsi_strategy.py")
    assert publish_public_pr.is_sensitive_path("strategies/rsi_strategy.py")
    assert publish_public_pr.is_sensitive_path("lumina_quant/indicators/formulaic_alpha.py")
    assert publish_public_pr.is_sensitive_path("reports/benchmarks/latest.json")
    assert publish_public_pr.is_sensitive_path(".env")


def test_sensitive_path_detection_allows_public_runtime_paths():
    assert not publish_public_pr.is_sensitive_path("run_backtest.py")
    assert not publish_public_pr.is_sensitive_path("lumina_quant/backtesting/chunked_runner.py")
    assert not publish_public_pr.is_sensitive_path("docs/RUNBOOK_1Y_1S_LOCAL.md")


def test_default_branch_name_uses_prefix():
    branch = publish_public_pr._default_branch_name("public-sync")
    assert branch.startswith("public-sync-")
