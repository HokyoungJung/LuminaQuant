from __future__ import annotations

import json
from pathlib import Path

from scripts.research import write_tradfi_external_alpha_real_money_path as module


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _summary_payload(
    *,
    real_ready: bool = False,
    shadow_allowed: bool = False,
    improved: bool = False,
    hard_stop_promotable: bool = False,
    execution_telemetry: dict | None = None,
) -> dict:
    return {
        "decision": {
            "new_external_family_improved_clean_best": improved,
            "paper_or_shadow_start_allowed": shadow_allowed,
            "real_money_ready": real_ready,
            "recommended_action": "block",
            "main_reason": "new external families did not beat clean best",
        },
        "run_coverage": {"data_latest_utc": "2026-06-13T08:30:00"},
        "schema_checks": {
            "has_clean_section": True,
            "has_demoted_section": True,
            "cost_stress_bps": [10, 15, 20],
            "new_family_labels_seen": ["tradfi_intraday_session_v1"],
        },
        "clean_section": {
            "best_clean": {
                "candidate_label": "dynamic_conviction_switch:x",
                "compounded_oos_return": 0.34,
                "max_oos_mdd": 0.27,
                "positive_oos_folds": 3,
                "latest_oos_return": 0.0,
                "hard_stop_promotable": hard_stop_promotable,
            },
            "best_clean_under_15pct_mdd": {
                "candidate_label": "dynamic_conviction_switch:y",
                "compounded_oos_return": 0.12,
                "max_oos_mdd": 0.10,
                "positive_oos_folds": 3,
                "latest_oos_return": 0.0,
                "hard_stop_promotable": False,
            },
        },
        "new_external_family_section": {
            "best_new_clean": {
                "candidate_label": "tradfi_intraday_session_v1:open_impulse_close_top10_mdd15",
                "compounded_oos_return": 0.0027,
                "max_oos_mdd": 0.0026,
                "positive_oos_folds": 1,
                "latest_oos_return": -0.001,
                "hard_stop_promotable": False,
            },
            "best_new_diagnostic": None,
        },
        "moonshot_shadow_section": {
            "best_demoted_shadow_or_post_oos": {
                "candidate_label": "lagged_shadow_leaf_router:x",
                "compounded_oos_return": 0.79,
                "max_oos_mdd": 0.27,
                "positive_oos_folds": 4,
                "latest_oos_return": 0.12,
                "hard_stop_promotable": False,
            }
        },
        "baseline_sections": {},
        "external_evidence_urls": {},
        "execution_telemetry": execution_telemetry or {},
    }


def test_real_money_path_blocks_all_execution_when_wf_and_preflight_block(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary_payload())
    wf = _write(tmp_path / "wf.json", {"artifact_kind": "wf"})
    preflight = _write(
        tmp_path / "preflight.json",
        {
            "recommended_action": "block_until_preflight_gaps_closed",
            "status": {
                "ready_for_paper": False,
                "ready_for_shadow": False,
                "ready_for_real": False,
            },
            "checks": {"refresh_is_stale": True, "decision_allows_live_start": False},
        },
    )

    payload = module.build_payload(
        summary_json=summary,
        wf_json=wf,
        live_preflight_json=preflight,
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["global_verdict"]["real_money_execution"] is False
    assert payload["global_verdict"]["paper_trading_start"] is False
    assert payload["global_verdict"]["shadow_trading_start"] is False
    assert payload["blocking_gates"]
    assert "summary_allows_real_money" in payload["blocking_gates"]
    assert "live_preflight_paper_or_shadow" in payload["blocking_gates"]
    markdown = module.render_markdown(payload)
    assert "Real-money: **blocked**" in markdown
    assert "tradfi_intraday_session_v1" in markdown


def test_missing_live_preflight_fails_closed(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary_payload())
    wf = _write(tmp_path / "wf.json", {"artifact_kind": "wf"})

    payload = module.build_payload(
        summary_json=summary,
        wf_json=wf,
        live_preflight_json=tmp_path / "missing.json",
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["live_readiness_preflight"]["exists"] is False
    assert payload["source_manifest"]["live_preflight_json"]["valid"] is False
    assert payload["global_verdict"]["allowed_start_modes"] == []
    assert "live_preflight_source_valid" in payload["blocking_gates"]
    assert "live_preflight_real_money" in payload["blocking_gates"]


def test_missing_or_corrupt_required_sources_write_disabled_artifact(tmp_path: Path) -> None:
    corrupt_summary = tmp_path / "corrupt-summary.json"
    corrupt_summary.write_text("{not-json", encoding="utf-8")
    missing_wf = tmp_path / "missing-wf.json"
    preflight = _write(
        tmp_path / "preflight.json",
        {
            "recommended_action": "paper_run_allowed",
            "status": {"ready_for_paper": True, "ready_for_shadow": False, "ready_for_real": False},
            "checks": {"refresh_is_stale": False, "decision_allows_live_start": True},
        },
    )

    payload = module.build_payload(
        summary_json=corrupt_summary,
        wf_json=missing_wf,
        live_preflight_json=preflight,
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["source_manifest"]["summary_json"]["exists"] is True
    assert payload["source_manifest"]["summary_json"]["valid"] is False
    assert "JSONDecodeError" in payload["source_manifest"]["summary_json"]["error"]
    assert payload["source_manifest"]["wf_json"]["exists"] is False
    assert payload["source_manifest"]["wf_json"]["valid"] is False
    assert payload["global_verdict"]["real_money_execution"] is False
    assert payload["global_verdict"]["paper_trading_start"] is False
    assert payload["global_verdict"]["shadow_trading_start"] is False
    assert {"summary_source_valid", "wf_source_valid"}.issubset(payload["blocking_gates"])


def test_all_gates_passed_can_express_real_money_allow_state(tmp_path: Path) -> None:
    summary = _write(
        tmp_path / "summary.json",
        _summary_payload(
            real_ready=True,
            shadow_allowed=True,
            improved=True,
            hard_stop_promotable=True,
            execution_telemetry={
                "paper_testnet_telemetry_passed": True,
                "mean_round_trip_cost_bps": 7.5,
                "p95_round_trip_cost_bps": 12.0,
                "unexplained_reconciliation_gaps": 0,
            },
        ),
    )
    wf = _write(tmp_path / "wf.json", {"artifact_kind": "wf"})
    preflight = _write(
        tmp_path / "preflight.json",
        {
            "recommended_action": "real_run_allowed",
            "status": {
                "ready_for_paper": True,
                "ready_for_shadow": True,
                "ready_for_real": True,
                "ready_for_full": True,
            },
            "checks": {"refresh_is_stale": False, "decision_allows_live_start": True},
        },
    )

    payload = module.build_payload(
        summary_json=summary,
        wf_json=wf,
        live_preflight_json=preflight,
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert payload["blocking_gates"] == []
    assert payload["global_verdict"]["paper_trading_start"] is True
    assert payload["global_verdict"]["shadow_trading_start"] is True
    assert payload["global_verdict"]["real_money_execution"] is True
    assert payload["global_verdict"]["allowed_start_modes"] == ["paper", "shadow", "real"]
    assert payload["global_verdict"]["decision"] == "real_money_allowed_after_all_gates_passed"
    by_stage = {stage["name"]: stage["status"] for stage in payload["real_money_path"]}
    assert by_stage["canary_real_money"] == "allowed"
    markdown = module.render_markdown(payload)
    assert "Real-money: **allowed**" in markdown
    assert "Paper start: **allowed**" in markdown
    assert "Shadow start: **allowed**" in markdown


def test_malformed_execution_telemetry_fails_closed_without_crashing(tmp_path: Path) -> None:
    summary = _write(
        tmp_path / "summary.json",
        _summary_payload(
            real_ready=True,
            shadow_allowed=True,
            improved=True,
            hard_stop_promotable=True,
            execution_telemetry={
                "paper_testnet_telemetry_passed": True,
                "mean_round_trip_cost_bps": 7.5,
                "p95_round_trip_cost_bps": 12.0,
                "unexplained_reconciliation_gaps": "n/a",
            },
        ),
    )
    wf = _write(tmp_path / "wf.json", {"artifact_kind": "wf"})
    preflight = _write(
        tmp_path / "preflight.json",
        {
            "recommended_action": "real_run_allowed",
            "status": {
                "ready_for_paper": True,
                "ready_for_shadow": True,
                "ready_for_real": True,
            },
            "checks": {"refresh_is_stale": False, "decision_allows_live_start": True},
        },
    )

    payload = module.build_payload(
        summary_json=summary,
        wf_json=wf,
        live_preflight_json=preflight,
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    assert "execution_telemetry" in payload["blocking_gates"]
    assert payload["global_verdict"]["real_money_execution"] is False


def test_write_outputs_records_hashable_sources(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary_payload())
    wf = _write(tmp_path / "wf.json", {"artifact_kind": "wf"})
    preflight = _write(
        tmp_path / "preflight.json",
        {"status": {"ready_for_real": False}, "checks": {}, "recommended_action": "block"},
    )
    payload = module.build_payload(
        summary_json=summary,
        wf_json=wf,
        live_preflight_json=preflight,
        generated_at_utc="2026-06-13T00:00:00Z",
    )

    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    module.write_outputs(payload, output_json=output_json, output_md=output_md)

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["source_manifest"]["summary_json"]["sha256"]
    assert output_md.read_text(encoding="utf-8").startswith("# TradFi external-alpha")
