from __future__ import annotations

import json
from pathlib import Path

from scripts.research import write_tradfi_external_alpha_baseline_provenance as module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_build_payload_hashes_artifacts_and_records_baselines(tmp_path: Path) -> None:
    wf_path = _write_json(
        tmp_path / "wf.json",
        {
            "data_coverage": {
                "global_earliest_utc": "2025-01-01T00:00:00",
                "global_latest_utc": "2026-06-13T08:30:00",
                "source": "direct_1m_ohlcv_resampled_to_30m_plus",
            },
            "folds": [{"fold_id": "2025-09"}, {"fold_id": "2025-10"}],
            "universe": {
                "requested_symbol_count": 3,
                "loaded_symbol_count": 3,
                "missing_symbol_count": 0,
                "symbols": ["SPYUSDT", "BTCUSDT", "AAPLUSDT"],
                "missing_symbols": [],
            },
        },
    )
    summary_path = _write_json(
        tmp_path / "summary.json",
        {
            "baseline_candidates_recorded": [
                {"label": "relaxed_efficiency:hybrid_v3_5"},
                {"label": "fixed_relaxed_dynamic_blend:relaxed60_dynamic40"},
                {"label": "dynamic_conviction_switch:t0.90_risk_capped_fallback"},
            ],
            "best_clean_mdd15": {
                "candidate_label": "dynamic_conviction_switch:mdd15",
                "clean_promotion_eligible": True,
                "compounded_oos_return_pct": 12.9684,
                "fold_count": 10,
                "hard_stop_promotable": False,
                "latest_oos_return_pct": 0.0,
                "max_oos_mdd_pct": 10.0802,
                "positive_oos_folds": 3,
            },
            "current_best_clean": {
                "candidate_label": "dynamic_conviction_switch:clean",
                "clean_promotion_eligible": True,
                "compounded_oos_return_pct": 34.3886,
                "fold_count": 10,
                "hard_stop_promotable": False,
                "latest_oos_return_pct": 0.0,
                "max_oos_mdd_pct": 27.6901,
                "positive_oos_folds": 3,
            },
            "current_best_demoted_shadow": {
                "candidate_label": "codex_lagged_leaf_router_grid:shadow",
                "clean_promotion_eligible": False,
                "compounded_oos_return_pct": 79.4211,
                "fold_count": 10,
                "hard_stop_promotable": False,
                "latest_oos_return_pct": 12.5609,
                "max_oos_mdd_pct": 27.6901,
                "non_clean_reasons": ["post_oos_research_variant"],
                "positive_oos_folds": 4,
            },
            "data_coverage": {
                "fold_count": 10,
                "latest_available_data": "2026-06-13T08:30:00",
                "loaded_symbol_count": 3,
                "missing_symbol_count": 0,
                "requested_symbol_count": 3,
            },
            "real_money_status": {"hard_block": True},
        },
    )
    prd = tmp_path / "prd.md"
    spec = tmp_path / "test-spec.md"
    prd.write_text("prd", encoding="utf-8")
    spec.write_text("spec", encoding="utf-8")

    payload = module.build_payload(
        baseline_wf_json=wf_path,
        baseline_summary_json=summary_path,
        prd_path=prd,
        test_spec_path=spec,
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["missing_required_artifacts"] == []
    assert payload["source_artifacts"]["baseline_wf_json"]["sha256"] == module._sha256_file(wf_path)
    assert payload["source_artifacts"]["prd"]["sha256"] == module._sha256_file(prd)
    assert payload["data_coverage"]["latest_available_data_utc"] == "2026-06-13T08:30:00"
    assert payload["universe"]["symbols"] == ["AAPLUSDT", "BTCUSDT", "SPYUSDT"]
    assert (
        payload["current_baseline_labels"][0]["candidate_label"]
        == "dynamic_conviction_switch:clean"
    )
    assert (
        payload["current_baseline_labels"][2]["readiness_label"]
        == "shadow_freeze_only_requires_fresh_forward"
    )
    assert payload["external_source_registry"]["validation"]["valid"] is True
    assert payload["policy"]["real_money_execution"] is False


def test_source_registry_validation_rejects_cycle_allowed_credentialed_source() -> None:
    registry = module.build_external_source_registry(generated_at_utc="2026-06-13T00:00:00Z")
    assert registry["validation"]["valid"] is True
    assert "fred_api_key_docs_excluded" in registry["validation"]["excluded_source_ids"]

    bad = dict(registry)
    sources = [dict(source) for source in registry["sources"]]
    sources[-1]["cycle_allowed"] = True
    bad["sources"] = sources

    validation = module.validate_source_registry(bad)

    assert validation["valid"] is False
    assert (
        validation["violations"][0]["reason"] == "cycle_allowed_source_requires_disallowed_access"
    )


def test_write_outputs_persists_snapshot_registry_and_hash(tmp_path: Path) -> None:
    wf_path = _write_json(
        tmp_path / "wf.json",
        {
            "data_coverage": {"global_latest_utc": "2026-06-13T08:30:00"},
            "folds": [],
            "universe": {"symbols": ["BTCUSDT"], "missing_symbols": []},
        },
    )
    summary_path = _write_json(
        tmp_path / "summary.json",
        {
            "current_best_clean": {"candidate_label": "clean", "clean_promotion_eligible": True},
            "best_clean_mdd15": {"candidate_label": "mdd15", "clean_promotion_eligible": True},
            "current_best_demoted_shadow": {
                "candidate_label": "shadow",
                "clean_promotion_eligible": False,
            },
            "data_coverage": {"latest_available_data": "2026-06-13T08:30:00"},
        },
    )
    prd = tmp_path / "prd.md"
    spec = tmp_path / "test-spec.md"
    prd.write_text("prd", encoding="utf-8")
    spec.write_text("spec", encoding="utf-8")
    payload = module.build_payload(
        baseline_wf_json=wf_path,
        baseline_summary_json=summary_path,
        prd_path=prd,
        test_spec_path=spec,
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    outputs = module.write_outputs(payload, output_dir=tmp_path / "out")

    snapshot = Path(outputs["baseline_evidence_snapshot_json"])
    registry = Path(outputs["external_source_registry_json"])
    sha = Path(outputs["baseline_evidence_snapshot_sha256"])
    assert snapshot.exists()
    assert registry.exists()
    assert sha.read_text(encoding="utf-8").split()[0] == module._sha256_file(snapshot)
    saved = json.loads(snapshot.read_text(encoding="utf-8"))
    assert "source_registry_payload" not in saved
    assert saved["external_source_registry"]["source_count"] >= 5
