from __future__ import annotations

import json
from pathlib import Path

from scripts.research import write_alpha_zoo_69_asset_strategy_baseline_readiness as module


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_build_payload_classifies_user_designated_baselines(tmp_path: Path) -> None:
    exact = _write(
        tmp_path / "exact.json",
        {
            "aggregate_rankings": [
                {
                    "candidate_label": "relaxed_efficiency:hybrid_v3_5",
                    "family": "relaxed_efficiency",
                    "fold_count": 10,
                    "clean_promotion_eligible": True,
                    "compounded_oos_return": 1.5603,
                    "max_oos_mdd": 0.1975,
                    "positive_oos_folds": 5,
                },
                {
                    "candidate_label": "fixed_relaxed_dynamic_blend:relaxed60_dynamic40",
                    "family": "fixed_relaxed_dynamic_blend",
                    "fold_count": 10,
                    "clean_promotion_eligible": False,
                    "post_oos_research_variant": True,
                    "requires_fresh_forward_shadow": True,
                    "nested_hybrid_dependency": True,
                    "compounded_oos_return": 1.1175,
                    "max_oos_mdd": 0.1619,
                    "positive_oos_folds": 6,
                },
            ]
        },
    )
    final = _write(
        tmp_path / "final.json",
        {
            "comparison_rows": [
                {
                    "candidate_label": "dynamic_conviction_switch:t0.90_risk_capped_fallback",
                    "family": "dynamic_conviction_switch",
                    "fold_count": 10,
                    "clean_promotion_eligible": True,
                    "compounded_oos_return": 0.5338,
                    "max_oos_mdd": 0.188,
                    "positive_oos_folds": 5,
                }
            ]
        },
    )
    full = _write(
        tmp_path / "full.json",
        {
            "aggregate_rankings": [
                {
                    "candidate_label": "dynamic_conviction_switch:t0.90_risk_capped_fallback",
                    "clean_promotion_eligible": True,
                    "compounded_oos_return": 0.0861,
                    "max_oos_mdd": 0.1008,
                }
            ]
        },
    )

    payload = module.build_payload(
        source_paths={
            "exact_blend_20260603": exact,
            "best_strategy_final_20260601": final,
            "no_nested_full_eval_20260604": full,
        },
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["global_verdict"]["ready_for_real"] is False
    by_label = {item["label"]: item for item in payload["candidate_assessments"]}
    assert (
        by_label["relaxed_efficiency:hybrid_v3_5"]["status"]
        == "historical_clean_reference_shadow_only"
    )
    assert (
        by_label["fixed_relaxed_dynamic_blend:relaxed60_dynamic40"]["status"]
        == "diagnostic_shadow_only_rebuild_leaf_first"
    )
    assert (
        by_label["fixed_relaxed_dynamic_blend:relaxed60_dynamic40"]["primary_metrics"][
            "clean_promotion_eligible"
        ]
        is False
    )
    assert (
        by_label["dynamic_conviction_switch:t0.90_risk_capped_fallback"]["status"]
        == "paper_control_shadow_only"
    )
    assert (
        by_label["dynamic_conviction_switch:t0.90_risk_capped_fallback"]["primary_metrics"][
            "_source"
        ]
        == "best_strategy_final_20260601"
    )


def test_leaf_rebuild_command_freezes_source_universe_symbols(tmp_path: Path) -> None:
    exact = _write(
        tmp_path / "exact.json",
        {
            "universe": {
                "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            },
            "aggregate_rankings": [],
        },
    )

    payload = module.build_payload(
        source_paths={"exact_blend_20260603": exact},
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["freeze_universe"]["symbol_count"] == 3
    leaf_command = payload["recommended_next_commands"][1]
    assert "--symbols BTCUSDT,ETHUSDT,SOLUSDT" in leaf_command


def test_build_payload_summarizes_leaf_rebuild_and_preflight(tmp_path: Path) -> None:
    leaf = _write(
        tmp_path / "leaf.json",
        {
            "generated_at_utc": "2026-06-13T06:32:09Z",
            "data_coverage": {
                "global_latest_utc": "2026-06-06T08:30:00",
                "requested_symbol_count": 69,
                "loaded_symbol_count": 69,
            },
            "folds": [{"fold_id": "2025-09"}],
            "aggregate_rankings": [
                {
                    "candidate_label": "relaxed_efficiency:hybrid_v3_5",
                    "clean_promotion_eligible": True,
                    "compounded_oos_return": -0.2949,
                    "max_oos_mdd": 0.144,
                    "hard_stop_promotable": False,
                },
            ],
            "clean_promotion_rankings": [
                {
                    "candidate_label": "dynamic_conviction_switch:t0.90_risk_capped_fallback_val_mdd30_scaled",
                    "clean_promotion_eligible": True,
                    "compounded_oos_return": 0.5329,
                    "max_oos_mdd": 0.2769,
                    "hard_stop_promotable": False,
                }
            ],
        },
    )
    preflight = _write(
        tmp_path / "preflight.json",
        {
            "generated_at": "2026-06-13T06:32:30+00:00",
            "recommended_action": "block_until_preflight_gaps_closed",
            "status": {"ready_for_real": False, "ready_for_paper": False},
            "checks": {
                "decision_allows_live_start": False,
                "refresh_is_stale": True,
                "testnet": True,
                "artifact_real_money_veto": True,
                "artifact_post_oos_research_variant": True,
                "artifact_requires_fresh_forward_shadow": True,
                "artifact_clean_promotion_eligible": False,
            },
        },
    )

    payload = module.build_payload(
        source_paths={
            "leaf_rebuild_shadow_20260613": leaf,
            "live_readiness_preflight_20260613": preflight,
            "missing_provenance_boundary": tmp_path / "missing.json",
        },
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["baseline_labels"] == list(module.BASELINE_LABELS)
    assert payload["missing_json_sources"] == ["missing_provenance_boundary"]
    source_manifest = payload["source_manifest"]
    assert set(source_manifest) == {
        "leaf_rebuild_shadow_20260613",
        "live_readiness_preflight_20260613",
        "missing_provenance_boundary",
    }
    assert source_manifest["leaf_rebuild_shadow_20260613"]["exists"] is True
    assert source_manifest["leaf_rebuild_shadow_20260613"]["sha256"]
    assert source_manifest["leaf_rebuild_shadow_20260613"]["size_bytes"] > 0
    assert source_manifest["live_readiness_preflight_20260613"]["exists"] is True
    assert source_manifest["live_readiness_preflight_20260613"]["sha256"]
    assert source_manifest["live_readiness_preflight_20260613"]["size_bytes"] > 0
    assert source_manifest["missing_provenance_boundary"]["exists"] is False
    assert source_manifest["missing_provenance_boundary"]["sha256"] is None
    assert source_manifest["missing_provenance_boundary"]["size_bytes"] is None

    assert payload["latest_leaf_rebuild_shadow"]["loaded_symbol_count"] == 69
    assert (
        payload["latest_leaf_rebuild_shadow"]["baseline_rows"]["relaxed_efficiency:hybrid_v3_5"][
            "compounded_oos_return"
        ]
        == -0.2949
    )
    assert (
        payload["live_readiness_preflight"]["recommended_action"]
        == "block_until_preflight_gaps_closed"
    )
    key_checks = payload["live_readiness_preflight"]["key_checks"]
    assert key_checks["artifact_real_money_veto"] is True
    assert key_checks["artifact_post_oos_research_variant"] is True
    assert key_checks["artifact_requires_fresh_forward_shadow"] is True
    assert key_checks["artifact_clean_promotion_eligible"] is False

    markdown = module.render_markdown(payload)
    assert "artifact_requires_fresh_forward_shadow" in markdown


def test_render_markdown_keeps_real_money_blocked(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "source.json",
        {
            "aggregate_rankings": [
                {
                    "candidate_label": "relaxed_efficiency:hybrid_v3_5",
                    "clean_promotion_eligible": True,
                    "compounded_oos_return": 1.0,
                    "max_oos_mdd": 0.2,
                }
            ]
        },
    )
    payload = module.build_payload(
        source_paths={"exact_blend_20260603": source},
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    markdown = module.render_markdown(payload)

    assert "Real-money: **blocked**" in markdown
    assert "`relaxed_efficiency:hybrid_v3_5`" in markdown
    assert "fresh_forward_shadow" in markdown
