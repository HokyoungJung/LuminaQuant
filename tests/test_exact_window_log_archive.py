from __future__ import annotations

import json
from pathlib import Path

from lumina_quant.eval.exact_window_log_archive import (
    CANONICAL_REGISTRY_LATEST,
    RECOVERED_REGISTRY_LATEST,
    SIGNATURE_REGISTRY_LATEST,
    scan_exact_window_logs,
    write_exact_window_canonical_registry,
    write_exact_window_log_archive,
)


def test_scan_exact_window_logs_extracts_run_entries(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "exact_window_sample.log").write_text(
        """[INFO] header
{
  "status": "completed",
  "run_id": "sample_run",
  "manifest_path": "/tmp/manifest.json",
  "summary_latest": "/tmp/summary.json",
  "details_latest": "/tmp/details.json",
  "fail_analysis_latest": "/tmp/fail.json",
  "memory_evidence_latest": "/tmp/memory.json",
  "allow_metals": true,
  "promoted_count": 1
}
\tCommand being timed: "uv run lq exact-window --output-dir var/reports/exact_window_backtests/sample --run-id sample_run --timeframes 4h 1d --symbols BTC/USDT XAG/USDT --chunk-days 14 --allow-metals"
\tMaximum resident set size (kbytes): 2048000
""",
        encoding="utf-8",
    )

    entries = scan_exact_window_logs(log_dir)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["run_id"] == "sample_run"
    assert entry["requested_timeframes"] == ["4h", "1d"]
    assert entry["requested_symbols"] == ["BTC/USDT", "XAG/USDT"]
    assert entry["peak_rss_mib"] == 2000.0
    assert entry["allow_metals"] is True


def test_write_exact_window_log_archive_creates_recovered_registry_and_markdown(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "lq-exact-sample.log").write_text(
        """{
  "status": "completed",
  "run_id": "sample_run",
  "manifest_path": "/tmp/manifest.json",
  "summary_latest": "/tmp/summary.json",
  "details_latest": "/tmp/details.json",
  "allow_metals": false
}
\tCommand being timed: "uv run lq exact-window --output-dir var/reports/exact_window_backtests/sample --run-id sample_run --timeframes 30m --symbols BTC/USDT ETH/USDT --chunk-days 4"
\tMaximum resident set size (kbytes): 1024000
""",
        encoding="utf-8",
    )

    result = write_exact_window_log_archive(log_dir=log_dir, report_root=tmp_path / "reports")
    registry = json.loads(Path(result["recovered_registry_path"]).read_text(encoding="utf-8"))
    assert registry["entry_count"] == 1
    assert registry["entries"][0]["run_id"] == "sample_run"
    assert Path(result["archive_md"]).exists()


def test_write_exact_window_log_archive_does_not_overwrite_canonical_registry(tmp_path: Path):
    report_root = tmp_path / "reports"
    report_root.mkdir(parents=True)
    canonical_registry_path = report_root / CANONICAL_REGISTRY_LATEST
    canonical_payload = {
        "schema_version": "1.0",
        "entry_count": 1,
        "entries": [{"run_id": "canonical_run", "run_signature": "sig-1", "status": "completed"}],
    }
    canonical_registry_path.write_text(json.dumps(canonical_payload), encoding="utf-8")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "lq-exact-sample.log").write_text(
        """{
  "status": "completed",
  "run_id": "recovered_run",
  "manifest_path": "/tmp/manifest.json"
}
""",
        encoding="utf-8",
    )

    write_exact_window_log_archive(log_dir=log_dir, report_root=report_root)
    assert json.loads(canonical_registry_path.read_text(encoding="utf-8")) == canonical_payload
    recovered_payload = json.loads((report_root / RECOVERED_REGISTRY_LATEST).read_text(encoding="utf-8"))
    assert recovered_payload["entries"][0]["run_id"] == "recovered_run"


def test_write_exact_window_canonical_registry_rebuilds_from_signature_registry(tmp_path: Path):
    report_root = tmp_path / "reports"
    report_root.mkdir(parents=True)
    manifest_path = report_root / "run-1" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "run_signature": "sig-1",
                "batch_id": "15m-30m",
                "requested_timeframes": ["15m", "30m"],
                "requested_symbols": ["BTC/USDT", "ETH/USDT"],
                "allow_metals": False,
                "chunk_days": 6,
                "window_profile": "default",
                "promoted_count": 2,
                "evaluated_count": 15,
                "requested_oos_end_exclusive": "2026-03-09T00:00:00+00:00",
                "train_start": "2025-01-01T00:00:00+00:00",
                "val_start": "2026-01-01T00:00:00+00:00",
                "oos_start": "2026-02-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    memory_path = report_root / "run-1" / "memory.json"
    memory_path.write_text(json.dumps({"peak_rss_mib": 1234.5}), encoding="utf-8")
    (report_root / SIGNATURE_REGISTRY_LATEST).write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "signature": "sig-1",
                "status": "completed",
                "batch_id": "15m-30m",
                "manifest_path": str(manifest_path),
                "memory_evidence_path": str(memory_path),
                "summary_path": str(report_root / "summary.json"),
                "details_path": str(report_root / "details.json"),
                "completed_at_utc": "2026-03-09T13:50:41.240030+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = write_exact_window_canonical_registry(report_root=report_root)
    registry = json.loads(Path(result["registry_path"]).read_text(encoding="utf-8"))
    assert registry["entry_count"] == 1
    entry = registry["entries"][0]
    assert entry["run_id"] == "run-1"
    assert entry["run_signature"] == "sig-1"
    assert entry["requested_timeframes"] == ["15m", "30m"]
    assert entry["peak_rss_mib"] == 1234.5
