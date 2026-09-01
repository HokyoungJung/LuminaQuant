from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from lumina_quant.core.memory_budget import DEFAULT_EXECUTION_MEMORY_POLICY

_POLICY = DEFAULT_EXECUTION_MEMORY_POLICY


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "research" / "run_llm_alpha_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_llm_alpha_pipeline_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_preserve_8gb_memory_budget():
    module = _load_module()
    args = module.build_parser().parse_args([])

    assert args.total_memory_cap_gib == _POLICY.total_memory_cap_gib
    assert args.heavy_run_cap_gib == _POLICY.heavy_run_cap_gib
    assert args.report_root.endswith("var/reports/exact_window_backtests")


def test_main_passes_default_memory_budget_to_manifest_writer(monkeypatch, tmp_path: Path, capsys):
    module = _load_module()
    captured: dict[str, object] = {}

    def _fake_writer(**kwargs):
        captured.update(kwargs)
        return {
            "json_path": str(tmp_path / "alpha_research_pipeline_latest.json"),
            "md_path": str(tmp_path / "alpha_research_pipeline_latest.md"),
            "family_count": 1,
            "metric_count": 2,
        }

    monkeypatch.setattr(module, "write_alpha_research_pipeline_manifest", _fake_writer)

    rc = module.main([])

    assert rc == 0
    assert captured["total_memory_cap_gib"] == _POLICY.total_memory_cap_gib
    assert captured["heavy_run_cap_gib"] == _POLICY.heavy_run_cap_gib
    assert str(captured["report_root"]).endswith("var/reports/exact_window_backtests")
    assert '"family_count": 1' in capsys.readouterr().out
