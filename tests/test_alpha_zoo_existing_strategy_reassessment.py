from __future__ import annotations

import json

from scripts.research import write_alpha_zoo_existing_strategy_reassessment as module


def test_build_reassessment_payload_enumerates_registry_rows_without_promotion() -> None:
    payload = module.build_reassessment_payload(
        current_top_payload={
            "core_selection_set": [
                {
                    "role": "best_clean_paper_baseline",
                    "model": "dynamic_conviction_switch:t0.85",
                    "clean": True,
                    "oos_comp_pct": 34.39,
                    "max_oos_mdd_pct": 27.69,
                    "status": "paper_baseline_only_not_real_money",
                }
            ]
        },
        strategy_names=("MeanReversionStdStrategy", "MissingStrategy"),
    )

    assert payload["artifact_kind"] == "alpha_zoo_existing_strategy_reassessment_smoke_manifest"
    assert payload["selection_inputs"] == ["registry_metadata", "train_validation_future_smoke_only"]
    assert payload["locked_oos_policy"] == "current_known_oos_is_report_context_only_not_selection"
    assert payload["strategy_count"] == 1
    assert payload["skipped_count"] == 1
    row = payload["strategy_rows"][0]
    assert row["strategy_name"] == "MeanReversionStdStrategy"
    assert row["runnable_registry_entry"] is True
    assert row["full_wf_promotion_eligible"] is False
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False
    assert "requires_bounded_smoke_metrics" in row["rejection_reasons"]
    assert payload["survivor_list"] == []
    assert payload["full_wf_promotion_list"] == []


def test_reassessment_markdown_exposes_audit_and_control_evidence() -> None:
    payload = module.build_reassessment_payload(
        current_top_payload={
            "core_selection_set": [
                {
                    "role": "risk_trimmed_shadow",
                    "model": "codex_lagged_leaf_router_grid:fallback_mdd20_cap2",
                    "clean": False,
                    "oos_comp_pct": 64.42,
                    "max_oos_mdd_pct": 18.46,
                    "status": "preferred_shadow_watch_if_drawdown_matters",
                }
            ]
        },
        strategy_names=("MeanReversionStdStrategy",),
    )

    markdown = module.render_markdown(payload)

    assert "## Strategy audit rows" in markdown
    assert "`MeanReversionStdStrategy`" in markdown
    assert "`not_promoted_requires_smoke_and_full_wf`" in markdown
    assert "## Current benchmark/control evidence" in markdown
    assert "risk_trimmed_shadow" in markdown
    assert "full-WF promotion list: `[]`" in markdown


def test_write_outputs_round_trips_json_and_markdown(tmp_path) -> None:
    payload = module.build_reassessment_payload(strategy_names=("MeanReversionStdStrategy",))

    paths = module.write_outputs(payload, output_dir=tmp_path, stem="existing_strategy_reassessment_test")

    loaded = json.loads((tmp_path / "existing_strategy_reassessment_test.json").read_text())
    markdown = (tmp_path / "existing_strategy_reassessment_test.md").read_text()
    assert paths["json"].endswith("existing_strategy_reassessment_test.json")
    assert loaded["strategy_count"] == 1
    assert "Existing strategy reassessment smoke manifest" in markdown
