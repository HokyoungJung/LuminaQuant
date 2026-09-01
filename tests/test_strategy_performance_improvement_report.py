from __future__ import annotations

import json
from pathlib import Path

from scripts.research import build_strategy_performance_improvement_report as report


def test_build_report_normalizes_required_wf_fields(tmp_path: Path) -> None:
    source = tmp_path / "wf.json"
    source.write_text(
        json.dumps(
            {
                "completed_at_utc": "2026-07-07T00:00:00Z",
                "trial_policy": {"source_symbol_workers": 2},
                "runner_peak_rss_mib": 123.5,
                "folds": [{"fold_id": "2026-06"}],
                "fold_summaries": [{"fold_id": "2026-06"}],
                "fold_candidate_rows": [
                    {
                        "fold_id": "2026-06",
                        "candidate_label": "clean_leaf",
                        "uses_locked_oos_for_selection": False,
                        "real_money_execution": False,
                        "current_fold_oos_used_for_weighting": False,
                        "same_month_self_feeding": False,
                    }
                ],
                "universe": {
                    "requested_symbol_count": 2,
                    "loaded_symbol_count": 2,
                    "missing_symbol_count": 0,
                    "missing_symbols": [],
                },
                "data_coverage": {
                    "global_latest_utc": "2026-07-06T23:30:00",
                    "missing_symbols": [],
                },
                "timeframe_coverage": {
                    "30m": {
                        "symbols_with_rows": 2,
                        "symbols_without_rows": 0,
                        "total_rows": 20,
                        "latest": "2026-07-06T23:30:00",
                    }
                },
                "dynamic_self_feed_audit": {"no_same_month_dynamic_self_feeding": True},
                "bridge_protocol_audit": {
                    "post_oos_expansion_for_same_protocol": False,
                    "current_fold_oos_used_for_bridge_weighting": False,
                    "same_month_dynamic_self_feeding": False,
                },
                "aggregate_rankings": [{"candidate_label": "clean_leaf"}],
                "clean_promotion_rankings": [{"candidate_label": "clean_leaf"}],
            }
        ),
        "utf-8",
    )
    command_log = tmp_path / "command_log.md"
    command_log.write_text("# log\n", "utf-8")
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    payload = report.build_report(
        source_json=source,
        output_json=output_json,
        output_md=output_md,
        command_log_path=command_log,
    )

    assert output_json.exists()
    assert output_md.exists()
    assert payload["command_log_path"] == str(command_log.resolve())
    assert payload["worker_count"] == 2
    assert payload["chunk_sizes"]["fold_count"] == 1
    assert payload["chunk_sizes"]["rows_by_fold"] == {"2026-06": 1}
    assert payload["peak_rss_mb"] == 123.5
    assert payload["peak_rss_source"] == "runner_peak_rss_mib_or_blocker_peak_rss_mb"
    assert "backend" in payload["native_backend_status"]
    assert payload["data_coverage_counts"]["loaded_symbol_count"] == 2
    assert payload["leak_checks"]["pass"] is True
    assert payload["full_universe_claim_status"] == (
        "claimed_loaded_all_requested_symbols_completed_walkforward"
    )


def test_build_report_persists_source_missing_blocker(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    payload = report.build_report(
        source_json=missing,
        output_json=output_json,
        output_md=output_md,
        command_log_path=tmp_path / "command_log.md",
        worker_count=3,
    )

    assert output_json.exists()
    assert output_md.exists()
    assert payload["source_status"] == "missing"
    assert payload["worker_count"] == 3
    assert payload["leak_checks"]["status"] == "not_evaluable_source_json_missing"
    assert payload["full_universe_claim_status"] == "not_claimed_source_json_missing"


def test_build_report_preserves_loaded_no_data_blocker(tmp_path: Path) -> None:
    source = tmp_path / "wf_blocked.json"
    source.write_text(
        json.dumps(
            {
                "status": "blocked",
                "pass_fail_decision": "FAIL_BLOCKED_NO_DATA",
                "blocker": (
                    "data/market_parquet has no direct 1m-derived bars; runner raised "
                    "ValueError: no direct 1m-derived bars loaded for any symbol/timeframe."
                ),
                "full_universe_claim_status": "blocked",
                "peak_rss_mb": 190.85,
                "data_coverage_counts": {
                    "basis": "runner announced symbols=110 then loaded no direct 1m bars",
                    "loaded": 0,
                    "missing": 110,
                    "train_eligible": 0,
                    "monitor_only": 0,
                },
                "leak_checks": {
                    "basis": "blocked before selection",
                    "uses_locked_oos_for_selection": False,
                    "uses_locked_oos_for_weighting": False,
                },
                "chunk_sizes": {"checkpoint_interval": 1, "checkpoint_markdown_interval": 0},
                "selected_variants": [],
            }
        ),
        "utf-8",
    )
    payload = report.build_report(
        source_json=source,
        output_json=tmp_path / "report.json",
        output_md=tmp_path / "report.md",
        command_log_path=tmp_path / "command_log.md",
        worker_count=2,
    )

    assert payload["source_status"] == "loaded"
    assert payload["source_blocker"].startswith("data/market_parquet has no direct 1m")
    assert payload["peak_rss_mb"] == 190.85
    assert payload["full_universe_claim_status"] == ("blocked_not_claimed_missing_direct_1m_bars")
    assert payload["data_coverage_counts"] == {
        "source": "blocked_walkforward_payload",
        "basis": "runner announced symbols=110 then loaded no direct 1m bars",
        "requested_symbol_count": 110,
        "loaded_symbol_count": 0,
        "missing_symbol_count": 110,
        "train_eligible_symbol_count": 0,
        "monitor_only_symbol_count": 0,
        "global_latest_utc": None,
        "timeframes": {},
    }
    assert payload["leak_checks"]["status"] == ("not_evaluable_blocked_missing_direct_1m_bars")
    assert payload["leak_checks"]["pass"] is False
    assert payload["chunk_sizes"]["fold_candidate_row_count"] == 0


def test_build_report_does_not_relabel_generic_blocker_as_direct_1m(tmp_path: Path) -> None:
    source = tmp_path / "wf_blocked_generic.json"
    source.write_text(
        json.dumps(
            {
                "status": "blocked",
                "pass_fail_decision": "FAIL_TIMEOUT",
                "blocker": "walk-forward timed out before all folds completed",
                "universe": {
                    "requested_symbol_count": 10,
                    "loaded_symbol_count": 5,
                    "missing_symbol_count": 5,
                },
                "fold_candidate_rows": [],
            }
        ),
        "utf-8",
    )

    payload = report.build_report(
        source_json=source,
        output_json=tmp_path / "report.json",
        output_md=tmp_path / "report.md",
        command_log_path=tmp_path / "command_log.md",
    )

    assert payload["source_status"] == "loaded"
    assert payload["source_blocker"] == "walk-forward timed out before all folds completed"
    assert payload["full_universe_claim_status"] == "blocked_not_claimed_other"
    assert payload["data_coverage_counts"]["requested_symbol_count"] == 10
    assert payload["data_coverage_counts"]["loaded_symbol_count"] == 5
    assert payload["data_coverage_counts"]["missing_symbol_count"] == 5
    assert payload["leak_checks"]["status"] == "not_evaluable_blocked"
    assert payload["leak_checks"]["pass"] is False
