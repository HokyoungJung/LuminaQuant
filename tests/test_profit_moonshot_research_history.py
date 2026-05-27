from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "write_profit_moonshot_research_history.py"
SPEC = importlib.util.spec_from_file_location("write_profit_moonshot_research_history", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


REQUIRED_TOP_LEVEL = {
    "strategy_chronology",
    "git_commit_history",
    "source_history_inventory",
    "source_search_ledger",
    "decision_log",
    "invalidity_lessons",
    "future_session_instructions",
    "generation_metadata",
}

REQUIRED_STRATEGY_FIELDS = {
    "research_date",
    "strategy_family",
    "artifact_paths",
    "hypothesis",
    "primary_signal_type",
    "state_variables_or_features",
    "universe",
    "timeframe",
    "split_periods",
    "implementation_files",
    "train_metrics",
    "validation_metrics",
    "oos_metrics",
    "leverage_status",
    "liquidation_status",
    "source_ledger_refs",
    "advantages",
    "disadvantages_or_risks",
    "final_decision",
    "rejection_or_promotion_reason",
}

REQUIRED_LEDGER_FIELDS = {
    "research_date",
    "source_type",
    "query_or_title",
    "normalized_key",
    "path_or_url",
    "content_summary",
    "what_was_used",
    "associated_strategy_families",
    "decision_impact",
    "staleness_policy",
    "recheck_before_use",
    "do_not_repeat_note",
}


def test_research_history_payload_schema_and_duplicate_ledger_collapse(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    plans = tmp_path / ".omx" / "plans"
    reports = tmp_path / "var" / "reports" / "profit_moonshot_20260501"
    docs.mkdir(parents=True)
    plans.mkdir(parents=True)
    reports.mkdir(parents=True)
    repeated_text = "Binance funding open interest taker flow liquidation docs were consulted."
    (docs / "session_handoff_20260509_profit_moonshot_integer_leverage.md").write_text(
        repeated_text + "\nInteger leverage only.\n", encoding="utf-8"
    )
    (plans / "profit_moonshot_dynamic_restart_research_history_20260510.md").write_text(
        repeated_text + "\nCalendar-primary invalidation and dynamic restart.\n", encoding="utf-8"
    )
    (reports / "profit_moonshot_strategy_validity_audit_result_20260510.md").write_text(
        "calendar-primary invalid; source ledger required; locked-OOS report-only.\n",
        encoding="utf-8",
    )

    payload = MODULE.build_research_history_payload(
        roots=[docs, plans, reports],
        generated_at_utc="2026-05-10T12:00:00Z",
        include_git_history=False,
    )

    assert REQUIRED_TOP_LEVEL.issubset(payload)
    assert payload["strategy_chronology"]
    assert payload["source_history_inventory"]
    assert payload["source_search_ledger"]
    assert REQUIRED_STRATEGY_FIELDS.issubset(payload["strategy_chronology"][0])
    assert all(REQUIRED_LEDGER_FIELDS.issubset(entry) for entry in payload["source_search_ledger"])
    assert all(
        item.get("ledger_refs")
        or (item.get("not_reconstructable") and item.get("not_reconstructable_reason"))
        for item in payload["source_history_inventory"]
    )

    external_keys = [entry["normalized_key"] for entry in payload["source_search_ledger"]]
    assert external_keys.count("external_reference:binance_funding_oi_taker_liquidation_docs") == 1
    binance_entry = next(
        entry
        for entry in payload["source_search_ledger"]
        if entry["normalized_key"] == "external_reference:binance_funding_oi_taker_liquidation_docs"
    )
    assert "2026-05-09" in binance_entry["research_date"]
    assert "2026-05-10" in binance_entry["research_date"]
    assert binance_entry["content_summary"] != binance_entry["what_was_used"]
    assert binance_entry["do_not_repeat_note"]


def test_research_history_writer_emits_markdown_and_json(tmp_path: Path) -> None:
    source_root = tmp_path / "docs"
    source_root.mkdir()
    (source_root / "profit_moonshot_next_hypotheses_20260508.md").write_text(
        "dynamic momentum residual funding OI research handoff\n", encoding="utf-8"
    )
    output_doc = tmp_path / "docs" / "profit_moonshot_research_history_20260510.md"
    report_dir = tmp_path / "var" / "reports" / "profit_moonshot_20260501" / "research_history"

    payload = MODULE.build_research_history_payload(
        roots=[source_root],
        generated_at_utc="2026-05-10T12:00:00Z",
        include_git_history=False,
    )
    paths = MODULE.write_research_history_outputs(
        payload, docs_path=output_doc, report_dir=report_dir
    )

    assert paths["docs_markdown"] == output_doc
    assert output_doc.exists()
    assert (report_dir / "profit_moonshot_research_history_latest.md").exists()
    json_payload = json.loads(
        (report_dir / "profit_moonshot_research_history_latest.json").read_text(encoding="utf-8")
    )
    assert json_payload["generation_metadata"]["schema_version"] == MODULE.SCHEMA_VERSION
    assert json_payload["generation_metadata"]["date_range"] == "2026-03-01..2026-05-10"
    assert "source/search ledger" in output_doc.read_text(encoding="utf-8").lower()


def test_git_commit_history_discovers_march_research_and_filters_runtime(monkeypatch) -> None:
    research_sha = "a" * 40
    runtime_sha = "b" * 40
    raw = (
        f"{research_sha}\x1f2026-03-14\x1fTune portfolio walk-forward research lane\x1f"
        "Validated candidate ensemble context.\x1e"
        f"{runtime_sha}\x1f2026-03-14\x1fomx(team): auto-checkpoint worker-1 [1]\x1f"
        "runtime scaffold\x1e"
    )

    class Result:
        returncode = 0
        stdout = raw

    def fake_run(*args, **kwargs):
        return Result()

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    items = MODULE.discover_git_commit_history()

    assert [item["commit_hash"] for item in items] == [research_sha]
    assert items[0]["research_date"] == "2026-03-14"
    assert items[0]["source_type"] == "git_commit"
    assert "dynamic_portfolio_allocator" in items[0]["associated_strategy_families"]
