from __future__ import annotations

import json
from pathlib import Path

from lumina_quant.eval.exact_window_reporting import (
    build_fail_analysis,
    resolve_backtest_registry,
    resolve_exact_window_artifact_paths,
    sync_exact_window_latest_aliases,
    upsert_backtest_registry,
    write_fail_analysis_bundle,
    write_memory_evidence_bundle,
)


def _sample_summary() -> dict[str, object]:
    return {
        "generated_at": "2026-03-08T09:02:35Z",
        "windows": {"actual_max_timestamp": "2026-03-07T10:00:00+00:00"},
        "execution_profile": {"requested_timeframes": ["1m", "5m", "1h"]},
        "promoted_count": 1,
        "evaluated_count": 3,
        "best_per_strategy": [
            {
                "candidate_id": "promoted-1",
                "strategy_class": "CarryStrategy",
                "strategy_timeframe": "1h",
                "promoted": True,
            }
        ],
    }


def _sample_details() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "cand-1",
            "name": "mr-1m",
            "family": "mean_reversion",
            "strategy_class": "MeanReversionStrategy",
            "strategy_timeframe": "1m",
            "oos": {"trade_count": 3, "mdd": 0.11, "sharpe": -0.4},
            "hurdle_fields": {
                "train": {"pass": True},
                "val": {"pass": True},
                "oos": {"pass": False},
            },
            "hard_reject_reasons": {"oos_sharpe": -0.4, "trade_count": 3.0},
            "metadata": {},
        },
        {
            "candidate_id": "cand-2",
            "name": "trend-5m",
            "family": "trend",
            "strategy_class": "TrendStrategy",
            "strategy_timeframe": "5m",
            "oos": {"trade_count": 12, "mdd": 0.28, "sharpe": 0.2},
            "hurdle_fields": {
                "train": {"pass": True},
                "val": {"pass": True},
                "oos": {"pass": False},
            },
            "hard_reject_reasons": {"pbo": 0.75},
            "metadata": {"rss_guard_triggered": True},
        },
        {
            "candidate_id": "promoted-1",
            "name": "carry-1h",
            "family": "carry",
            "strategy_class": "CarryStrategy",
            "strategy_timeframe": "1h",
            "oos": {"trade_count": 24, "mdd": 0.05, "sharpe": 1.3},
            "hurdle_fields": {
                "train": {"pass": True},
                "val": {"pass": True},
                "oos": {"pass": True},
            },
            "hard_reject_reasons": {},
            "metadata": {},
        },
    ]


def test_build_fail_analysis_groups_reasons_and_next_steps():
    payload = build_fail_analysis(_sample_summary(), _sample_details())

    reason_counts = {row["rejection_reason"]: row["count"] for row in payload["counts_by_rejection_reason"]}
    assert reason_counts["oos_sharpe"] == 1
    assert reason_counts["trade_count"] == 1
    assert reason_counts["pbo"] == 1
    assert reason_counts["rss_guard"] == 1
    assert reason_counts["promoted"] == 1
    assert payload["strategy_next_steps"]
    assert payload["strategy_next_steps"][0]["proposal"]


def test_write_fail_analysis_bundle_persists_json_and_markdown(tmp_path: Path):
    summary = _sample_summary()
    details = _sample_details()
    bundle = write_fail_analysis_bundle(output_dir=tmp_path, summary=summary, details=details)

    assert bundle["json_latest"].exists()
    assert bundle["md_latest"].exists()
    latest_payload = json.loads(bundle["json_latest"].read_text(encoding="utf-8"))
    assert latest_payload["promoted_count"] == 1
    assert "Exact-Window Fail Analysis" in bundle["md_latest"].read_text(encoding="utf-8")


def test_write_memory_evidence_bundle_includes_summary_context(tmp_path: Path):
    bundle = write_memory_evidence_bundle(
        output_dir=tmp_path,
        memory_summary={
            "status": "baseline_probe",
            "peak_rss_mib": 123.0,
            "budget_mib": 512.0,
            "soft_limit_mib": 307.2,
            "hard_limit_mib": 409.6,
            "rss_log_path": str(tmp_path / "rss.jsonl"),
        },
        summary=_sample_summary(),
    )

    payload = json.loads(bundle["json_latest"].read_text(encoding="utf-8"))
    assert payload["status"] == "baseline_probe"
    assert payload["windows"]["actual_max_timestamp"] == "2026-03-07T10:00:00+00:00"
    assert "Exact-Window Memory Evidence" in bundle["md_latest"].read_text(encoding="utf-8")


def test_sync_exact_window_latest_aliases_materializes_root_level_latest_files(tmp_path: Path):
    run_root = tmp_path / "resume_5m" / "5m"
    run_root.mkdir(parents=True)
    (tmp_path / "latest.json").write_text(
        json.dumps({"run_dir": str(run_root)}),
        encoding="utf-8",
    )
    (run_root / "exact_window_suite_summary_latest.json").write_text(
        json.dumps({"generated_at": "2026-03-09T09:00:00Z"}),
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
    (run_root / "exact_window_rss_latest.jsonl").write_text("sample\n", encoding="utf-8")

    synced = sync_exact_window_latest_aliases(tmp_path)
    assert synced["summary"] == (tmp_path / "exact_window_suite_summary_latest.json").resolve()
    assert synced["details"] == (tmp_path / "exact_window_candidate_details_latest.json").resolve()
    assert synced["fail_analysis"] == (tmp_path / "exact_window_fail_analysis_latest.json").resolve()
    assert synced["memory_evidence"] == (tmp_path / "exact_window_memory_evidence_latest.json").resolve()
    assert synced["rss_log"] == (tmp_path / "exact_window_rss_latest.jsonl").resolve()

    resolved = resolve_exact_window_artifact_paths(tmp_path)
    assert resolved["summary"] == (tmp_path / "exact_window_suite_summary_latest.json").resolve()
    assert json.loads(Path(resolved["summary"]).read_text(encoding="utf-8"))["generated_at"] == "2026-03-09T09:00:00Z"


def test_backtest_registry_upsert_and_resolve_dedupes(tmp_path: Path):
    output_root = tmp_path
    (output_root / "latest.json").write_text(json.dumps({"run_dir": str(output_root / "run-1" / "5m")}))
    (output_root / "run-1" / "5m").mkdir(parents=True)
    first = upsert_backtest_registry(
        output_root,
        run_id="run-1",
        batch_id="5m",
        status="completed",
        run_signature="sig",
        manifest_path=str(output_root / "manifest-1.json"),
        summary_path=str(output_root / "run-1" / "5m" / "exact_window_suite_summary_latest.json"),
        details_path=str(output_root / "run-1" / "5m" / "exact_window_candidate_details_latest.json"),
        fail_analysis_path=None,
        memory_evidence_path=None,
        requested_timeframes=["5m"],
        requested_symbols=["BTC/USDT"],
        allow_metals=False,
        batch_payload={"chunk_days": 14, "window_profile": "default"},
    )
    assert first["entry_count"] == 1
    second = upsert_backtest_registry(
        output_root,
        run_id="run-1",
        batch_id="5m",
        status="aborted_rss_guard",
        run_signature="sig",
        manifest_path=str(output_root / "manifest-1.json"),
        summary_path=str(output_root / "run-1" / "5m" / "exact_window_suite_summary_latest.json"),
        details_path=str(output_root / "run-1" / "5m" / "exact_window_candidate_details_latest.json"),
        fail_analysis_path=None,
        memory_evidence_path=None,
        requested_timeframes=["5m"],
        requested_symbols=["BTC/USDT"],
        allow_metals=False,
        batch_payload={"chunk_days": 14, "window_profile": "default"},
    )
    assert second["entry_count"] == 1
    registry = resolve_backtest_registry(output_root)
    assert isinstance(registry, list)
    assert len(registry) == 1
    assert registry[0]["status"] == "aborted_rss_guard"
    assert registry[0]["run_signature"] == "sig"
