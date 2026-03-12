from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "apps" / "dashboard" / "services" / "exact_window.py"
    spec = importlib.util.spec_from_file_location("dashboard_exact_window_loader", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load dashboard exact-window loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_load_exact_window_bundle_reads_latest_pointer(tmp_path: Path):
    run_root = tmp_path / "run-123"
    run_root.mkdir(parents=True)
    (tmp_path / "latest.json").write_text(json.dumps({"run_id": "run-123"}), encoding="utf-8")
    (run_root / "exact_window_suite_summary_latest.json").write_text(
        json.dumps({"generated_at": "2026-03-08T00:00:00Z"}),
        encoding="utf-8",
    )
    (run_root / "exact_window_candidate_details_latest.json").write_text(
        json.dumps([{"candidate_id": "c1"}]),
        encoding="utf-8",
    )
    (run_root / "exact_window_fail_analysis_latest.json").write_text(
        json.dumps({"counts_by_rejection_reason": []}),
        encoding="utf-8",
    )
    (run_root / "exact_window_memory_evidence_latest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (tmp_path / "exact_window_backtest_registry_latest.json").write_text(
        json.dumps({"entries": [{"run_id": "run-123", "status": "completed", "requested_timeframes": ["1m", "5m"]}]}),
        encoding="utf-8",
    )
    (tmp_path / "exact_window_decision_latest.json").write_text(
        json.dumps({"total_evaluated": 336, "promoted_total": 0}),
        encoding="utf-8",
    )
    (tmp_path / "pipeline").mkdir(parents=True)
    (tmp_path / "pipeline" / "alpha_research_pipeline_latest.json").write_text(
        json.dumps({"label": "pipeline", "families": [{"family_id": "pairs"}]}),
        encoding="utf-8",
    )
    (tmp_path / "exact_window_backtest_registry_recovered_latest.json").write_text(
        json.dumps({"entries": [{"run_id": "recovered-run", "status": "completed"}]}),
        encoding="utf-8",
    )

    payload = MODULE.load_exact_window_bundle(tmp_path)
    assert payload["run_root"] == str(run_root.resolve())
    assert payload["summary"]["generated_at"] == "2026-03-08T00:00:00Z"
    assert payload["details"][0]["candidate_id"] == "c1"
    assert payload["fail_analysis"]["counts_by_rejection_reason"] == []
    assert payload["memory_evidence"]["status"] == "completed"
    assert payload["decision"]["total_evaluated"] == 336
    assert payload["paths"]["decision"].endswith("exact_window_decision_latest.json")
    assert payload["paths"]["pipeline_manifest"].endswith("alpha_research_pipeline_latest.json")
    assert payload["pipeline_manifest"]["families"][0]["family_id"] == "pairs"
    assert payload["paths"]["registry"].endswith("exact_window_backtest_registry_latest.json")
    assert payload["paths"]["recovered_registry"].endswith("exact_window_backtest_registry_recovered_latest.json")
    assert payload["registry"][0]["run_id"] == "run-123"
    assert payload["recovered_registry"][0]["run_id"] == "recovered-run"


def test_load_exact_window_bundle_prefers_root_level_latest_aliases(tmp_path: Path):
    run_root = tmp_path / "run-123"
    run_root.mkdir(parents=True)
    (tmp_path / "latest.json").write_text(json.dumps({"run_id": "run-123"}), encoding="utf-8")
    (run_root / "exact_window_suite_summary_latest.json").write_text(
        json.dumps({"generated_at": "2026-03-08T00:00:00Z"}),
        encoding="utf-8",
    )
    (run_root / "exact_window_candidate_details_latest.json").write_text(
        json.dumps([{"candidate_id": "nested"}]),
        encoding="utf-8",
    )
    (run_root / "exact_window_fail_analysis_latest.json").write_text(
        json.dumps({"counts_by_rejection_reason": [{"rejection_reason": "nested", "count": 1}]}),
        encoding="utf-8",
    )
    (run_root / "exact_window_memory_evidence_latest.json").write_text(
        json.dumps({"status": "nested"}),
        encoding="utf-8",
    )
    (tmp_path / "exact_window_suite_summary_latest.json").write_text(
        json.dumps({"generated_at": "2026-03-09T00:00:00Z"}),
        encoding="utf-8",
    )
    (tmp_path / "exact_window_candidate_details_latest.json").write_text(
        json.dumps([{"candidate_id": "root"}]),
        encoding="utf-8",
    )
    (tmp_path / "exact_window_fail_analysis_latest.json").write_text(
        json.dumps({"counts_by_rejection_reason": [{"rejection_reason": "root", "count": 2}]}),
        encoding="utf-8",
    )
    (tmp_path / "exact_window_memory_evidence_latest.json").write_text(
        json.dumps({"status": "root"}),
        encoding="utf-8",
    )
    (tmp_path / "exact_window_backtest_registry_latest.json").write_text(
        json.dumps({"entries": [{"run_id": "run-123", "status": "completed", "requested_timeframes": ["1m", "5m"]}]}),
        encoding="utf-8",
    )

    payload = MODULE.load_exact_window_bundle(tmp_path)
    assert payload["summary"]["generated_at"] == "2026-03-09T00:00:00Z"
    assert payload["details"][0]["candidate_id"] == "root"
    assert payload["fail_analysis"]["counts_by_rejection_reason"][0]["rejection_reason"] == "root"
    assert payload["memory_evidence"]["status"] == "root"
    assert payload["registry"][0]["run_id"] == "run-123"


def test_load_exact_window_bundle_records_followup_parse_warnings(tmp_path: Path):
    (tmp_path / "followup_status").mkdir(parents=True)
    (tmp_path / "followup_status" / "broken.json").write_text("{", encoding="utf-8")

    payload = MODULE.load_exact_window_bundle(tmp_path)
    assert payload["warnings"]
    assert "broken.json" in payload["warnings"][0]


def test_exact_window_suite_module_exposes_embeddable_renderer():
    root = Path(__file__).resolve().parents[1]
    source = (root / "apps" / "dashboard" / "exact_window_suite.py").read_text(
        encoding="utf-8"
    )

    assert "def render_exact_window_dashboard" in source
    assert 'if __name__ == "__main__":' in source


def test_main_dashboard_exposes_exact_window_view_toggle():
    root = Path(__file__).resolve().parents[1]
    source = (root / "apps" / "dashboard" / "app.py").read_text(encoding="utf-8")

    assert "Dashboard View" in source
    assert "Exact-Window Suite" in source
    assert "render_exact_window_dashboard(standalone=False)" in source
