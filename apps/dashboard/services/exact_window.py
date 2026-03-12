"""Dashboard loader helpers for exact-window report bundles."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
for candidate in (ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

_reporting = importlib.import_module("lumina_quant.eval.exact_window_reporting")
resolve_exact_window_artifact_paths = _reporting.resolve_exact_window_artifact_paths
resolve_backtest_registry = _reporting.resolve_backtest_registry

_decision = importlib.import_module("lumina_quant.eval.exact_window_decision")
load_exact_window_decision_artifact = _decision.load_exact_window_decision_artifact
resolve_exact_window_decision_paths = _decision.resolve_exact_window_decision_paths

DEFAULT_REPORT_ROOT = ROOT / "var" / "reports" / "exact_window_backtests"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_exact_window_bundle(root: str | Path | None = None) -> dict[str, Any]:
    resolved_root = Path(root).resolve() if root is not None else DEFAULT_REPORT_ROOT.resolve()
    paths = resolve_exact_window_artifact_paths(resolved_root)
    decision_paths = resolve_exact_window_decision_paths(resolved_root)
    followup_status_root = resolved_root / "followup_status"
    pipeline_manifest_path = resolved_root / "pipeline" / "alpha_research_pipeline_latest.json"
    recovered_registry_path = resolved_root / "exact_window_backtest_registry_recovered_latest.json"
    archive_json_path = followup_status_root / "backtest_log_archive_latest.json"
    warnings: list[str] = []
    payload: dict[str, Any] = {
        "paths": {
            key: str(value) if isinstance(value, Path) else None
            for key, value in paths.items()
            if key not in {"root", "run_root", "latest_pointer"}
        },
        "root": str(paths["root"]),
        "run_root": str(paths["run_root"]),
        "latest_pointer": str(paths["latest_pointer"]) if paths.get("latest_pointer") else None,
        "decision_paths": {key: str(value) for key, value in decision_paths.items() if key != "root"},
    }
    payload["registry"] = resolve_backtest_registry(resolved_root)
    payload["paths"]["decision"] = str(decision_paths["json_latest"])
    payload["paths"]["decision_md"] = str(decision_paths["md_latest"])
    payload["paths"]["registry"] = str(resolved_root / "exact_window_backtest_registry_latest.json")
    payload["paths"]["recovered_registry"] = str(recovered_registry_path)
    payload["paths"]["archive_json"] = str(archive_json_path)
    payload["paths"]["pipeline_manifest"] = str(pipeline_manifest_path)
    for key in ("summary", "details", "fail_analysis", "memory_evidence"):
        candidate = paths.get(key)
        payload[key] = _read_json(candidate) if isinstance(candidate, Path) and candidate.exists() else None
    payload["pipeline_manifest"] = (
        _read_json(pipeline_manifest_path) if pipeline_manifest_path.exists() else None
    )
    payload["recovered_registry"] = []
    if recovered_registry_path.exists():
        recovered_payload = _read_json(recovered_registry_path)
        payload["recovered_registry"] = list(recovered_payload.get("entries") or [])
    payload["archive_payload"] = _read_json(archive_json_path) if archive_json_path.exists() else None
    payload["decision"] = load_exact_window_decision_artifact(resolved_root)
    payload["followup_status_root"] = str(followup_status_root)
    payload["followup_status"] = {}
    payload["followup_details"] = {}
    payload["followup_summaries"] = {}
    if followup_status_root.exists():
        for candidate in sorted(followup_status_root.glob("*.json")):
            try:
                stage_payload = _read_json(candidate)
                payload["followup_status"][candidate.stem] = stage_payload
                if isinstance(stage_payload, dict):
                    summary_path = Path(str(stage_payload.get("summary_path") or "")).resolve() if stage_payload.get("summary_path") else None
                    details_path = Path(str(stage_payload.get("details_path") or "")).resolve() if stage_payload.get("details_path") else None
                    if summary_path and summary_path.exists():
                        payload["followup_summaries"][candidate.stem] = _read_json(summary_path)
                    if details_path and details_path.exists():
                        payload["followup_details"][candidate.stem] = _read_json(details_path)
            except Exception as exc:
                warnings.append(f"Failed to load follow-up artifact {candidate.name}: {exc}")
    payload["queue_status"] = payload["followup_status"].get("queue_latest")
    payload["next_iteration"] = payload["followup_status"].get("next_iteration_latest")
    payload["warnings"] = warnings
    return payload


__all__ = [
    "DEFAULT_REPORT_ROOT",
    "load_exact_window_bundle",
    "load_exact_window_decision_artifact",
    "resolve_backtest_registry",
    "resolve_exact_window_decision_paths",
]
