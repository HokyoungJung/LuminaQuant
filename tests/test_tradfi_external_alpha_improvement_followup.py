from __future__ import annotations

import json
from pathlib import Path

from scripts.research import write_tradfi_external_alpha_improvement_followup as module


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _summary_payload() -> dict:
    return {
        "clean_section": {
            "best_clean": {
                "candidate_label": "dynamic_conviction_switch:clean",
                "family": "dynamic_conviction_switch",
                "compounded_oos_return": 0.3439,
                "max_oos_mdd": 0.2769,
                "positive_oos_folds": 3,
                "hard_stop_promotable": False,
            },
            "best_clean_under_15pct_mdd": {
                "candidate_label": "dynamic_conviction_switch:clean_mdd15",
                "family": "dynamic_conviction_switch",
                "compounded_oos_return": 0.1297,
                "max_oos_mdd": 0.1008,
                "positive_oos_folds": 3,
                "hard_stop_promotable": False,
            },
        },
        "new_external_family_section": {
            "best_new_clean": {
                "candidate_label": "tradfi_intraday_session_v1:weak",
                "family": "tradfi_intraday_session_v1",
                "compounded_oos_return": 0.0027,
                "max_oos_mdd": 0.0027,
                "positive_oos_folds": 1,
            }
        },
        "moonshot_shadow_section": {
            "best_demoted_shadow_or_post_oos": {
                "candidate_label": "codex_lagged_leaf_router_grid:moonshot",
                "family": "lagged_shadow_leaf_router",
                "compounded_oos_return": 0.7942,
                "max_oos_mdd": 0.2769,
                "positive_oos_folds": 4,
                "non_clean_reasons": ["post_oos_research_variant", "requires_fresh_forward_shadow"],
            }
        },
    }


def _fast_selector_payload() -> dict:
    return {
        "aggregate_rankings": [
            {
                "candidate_label": "row_level_leaf_selector:validation_calmar_mdd20",
                "compounded_oos_return": -0.1887,
                "max_oos_mdd": 0.2662,
                "positive_oos_folds": 4,
                "non_clean_reasons": ["post_oos_research_variant"],
            }
        ],
        "clean_promotion_rankings": [
            {
                "candidate_label": "dynamic_conviction_switch:clean",
                "compounded_oos_return": 0.3439,
                "max_oos_mdd": 0.2769,
                "positive_oos_folds": 3,
            }
        ],
        "row_level_leaf_selector_report": {"enabled": True, "selector_row_count": 40},
    }


def _new_alpha_payload() -> dict:
    return {
        "aggregate": {
            "compounded_oos_return": -0.0854,
            "max_oos_mdd": 0.0998,
            "positive_oos_folds": 2,
            "fold_count": 6,
        },
        "candidate_row_count_total": 32832,
        "fold_count": 6,
        "enabled_families": ["cross_asset_lead_lag_momentum", "cross_sectional"],
        "selection_policy": {"authority": "train_validation_only"},
    }


def _raw_probe_payload() -> dict:
    return {
        "candidate_count": 2688,
        "data_coverage": {"global_latest_utc": "2026-06-13T08:30:00"},
        "top_static_post_hoc": [
            {
                "candidate_label": "fast_leadlag_revert_QQQUSDT_l26_top3_lev10",
                "compounded_oos_return": 13.2450,
                "max_oos_mdd": 0.5427,
                "positive_oos_folds": 3,
                "validity": "invalid_static_post_hoc_upper_bound_not_clean",
            }
        ],
        "train_validation_selector": {
            "compounded_oos_return": -0.9121,
            "max_oos_mdd": 0.8707,
            "positive_oos_folds": 1,
            "validity": "failed_train_validation_selector_not_promotable",
        },
    }


def test_followup_records_no_clean_improvement_and_freeze_candidate(tmp_path: Path) -> None:
    payload = module.build_payload(
        summary_json=_write(tmp_path / "summary.json", _summary_payload()),
        wf_json=_write(tmp_path / "wf.json", {"artifact_kind": "wf"}),
        fast_selector_json=_write(tmp_path / "fast.json", _fast_selector_payload()),
        new_alpha_json=_write(tmp_path / "new.json", _new_alpha_payload()),
        raw_probe_json=_write(tmp_path / "raw.json", _raw_probe_payload()),
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["decision"]["clean_performance_improvement_found"] is False
    assert payload["decision"]["promotable_improvement_found"] is False
    assert payload["decision"]["real_money_execution"] is False
    assert payload["freeze_candidate"]["candidate_label"] == "codex_lagged_leaf_router_grid:moonshot"
    assert "fresh_forward_shadow_only" in payload["freeze_candidate"]["allowed_usage"]
    attempts = {attempt["name"]: attempt for attempt in payload["attempts"]}
    assert attempts["raw_tradfi_leadlag_moonshot_probe"]["top_static_post_hoc"][
        "compounded_oos_return"
    ] == 13.2450
    assert attempts["raw_tradfi_leadlag_moonshot_probe"]["promotable"] is False
    markdown = module.render_markdown(payload)
    assert "no_clean_performance_improvement_found" in markdown
    assert "codex_lagged_leaf_router_grid:moonshot" in markdown


def test_missing_required_sources_fail_closed_without_enabling_execution(tmp_path: Path) -> None:
    corrupt = tmp_path / "summary.json"
    corrupt.write_text("{not-json", encoding="utf-8")

    payload = module.build_payload(
        summary_json=corrupt,
        wf_json=tmp_path / "missing-wf.json",
        fast_selector_json=tmp_path / "missing-fast.json",
        new_alpha_json=tmp_path / "missing-new.json",
        raw_probe_json=tmp_path / "missing-raw.json",
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["source_manifest"]["summary_json"]["valid"] is False
    assert payload["source_manifest"]["wf_json"]["valid"] is False
    assert payload["decision"]["clean_performance_improvement_found"] is False
    assert payload["decision"]["real_money_execution"] is False
    assert set(payload["decision"]["source_failures"]) == {
        "summary_json",
        "wf_json",
        "fast_selector_json",
        "new_alpha_json",
    }
    assert payload["decision"]["raw_probe_missing"] is True


def test_write_outputs_records_hashable_sources(tmp_path: Path) -> None:
    payload = module.build_payload(
        summary_json=_write(tmp_path / "summary.json", _summary_payload()),
        wf_json=_write(tmp_path / "wf.json", {"artifact_kind": "wf"}),
        fast_selector_json=_write(tmp_path / "fast.json", _fast_selector_payload()),
        new_alpha_json=_write(tmp_path / "new.json", _new_alpha_payload()),
        raw_probe_json=_write(tmp_path / "raw.json", _raw_probe_payload()),
        generated_at_utc="2026-06-13T00:00:00Z",
    )
    output_json = tmp_path / "followup.json"
    output_md = tmp_path / "followup.md"

    module.write_outputs(payload, output_json=output_json, output_md=output_md)

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["source_manifest"]["summary_json"]["sha256"]
    assert output_md.read_text(encoding="utf-8").startswith(
        "# TradFi external-alpha improvement follow-up"
    )
