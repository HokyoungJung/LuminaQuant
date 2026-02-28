from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_8gb_baseline.py"
_SPEC = importlib.util.spec_from_file_location("verify_8gb_baseline_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {_SCRIPT_PATH}")
verify_8gb_baseline = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verify_8gb_baseline
_SPEC.loader.exec_module(verify_8gb_baseline)


def test_parser_accepts_benchmark_aliases():
    parser = verify_8gb_baseline._build_parser()
    args_a = parser.parse_args(["--benchmark", "reports/benchmarks/a.json"])
    args_b = parser.parse_args(["--benchmark-json", "reports/benchmarks/b.json"])
    assert args_a.benchmark_json == "reports/benchmarks/a.json"
    assert args_b.benchmark_json == "reports/benchmarks/b.json"


def test_main_passes_with_time_log_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_8gb_baseline, "PROJECT_ROOT", tmp_path)

    benchmark = tmp_path / "bench.json"
    benchmark.write_text(
        json.dumps(
            {
                "iterations": 1,
                "median_seconds": 1.25,
                "median_bars_per_sec": 1100.0,
                "max_peak_rss_mb": 900.0,
            }
        ),
        encoding="utf-8",
    )
    time_log = tmp_path / "time.log"
    time_log.write_text("Maximum resident set size (kbytes): 2048000\n", encoding="utf-8")

    rc = verify_8gb_baseline.main(
        [
            "--benchmark",
            str(benchmark),
            "--time-log",
            str(time_log),
            "--skip-dmesg",
            "--allow-missing-oom-sources",
            "--disk-path",
            str(tmp_path),
            "--disk-budget-gib",
            "1.0",
            "--output",
            str(tmp_path / "gate.json"),
        ]
    )
    assert rc == 0


def test_main_fails_when_rss_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_8gb_baseline, "PROJECT_ROOT", tmp_path)

    benchmark = tmp_path / "bench.json"
    benchmark.write_text(
        json.dumps(
            {
                "iterations": 1,
                "median_seconds": 1.25,
                "median_bars_per_sec": 1000.0,
                "max_peak_rss_mb": 600.0,
            }
        ),
        encoding="utf-8",
    )
    rc = verify_8gb_baseline.main(
        [
            "--benchmark",
            str(benchmark),
            "--rss-mib",
            "9000",
            "--skip-dmesg",
            "--allow-missing-oom-sources",
            "--disk-path",
            str(tmp_path),
            "--disk-budget-gib",
            "1.0",
            "--output",
            str(tmp_path / "gate.json"),
        ]
    )
    assert rc == 1


def test_main_passes_when_missing_rss_source_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_8gb_baseline, "PROJECT_ROOT", tmp_path)

    benchmark = tmp_path / "bench.json"
    benchmark.write_text(
        json.dumps(
            {
                "iterations": 1,
                "median_seconds": 0.2,
                "median_bars_per_sec": 10000.0,
                "max_peak_rss_mb": None,
                "samples": [{"peak_rss_mb": None}],
            }
        ),
        encoding="utf-8",
    )

    rc = verify_8gb_baseline.main(
        [
            "--benchmark",
            str(benchmark),
            "--allow-missing-rss-source",
            "--skip-dmesg",
            "--allow-missing-oom-sources",
            "--disk-path",
            str(tmp_path),
            "--disk-budget-gib",
            "1.0",
            "--output",
            str(tmp_path / "gate.json"),
        ]
    )
    assert rc == 0


def test_main_accepts_skipped_benchmark_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_8gb_baseline, "PROJECT_ROOT", tmp_path)

    benchmark = tmp_path / "bench.json"
    benchmark.write_text(
        json.dumps(
            {
                "skipped": True,
                "reason": "strategies package unavailable in this distribution",
            }
        ),
        encoding="utf-8",
    )
    time_log = tmp_path / "time.log"
    time_log.write_text("Maximum resident set size (kbytes): 102400\n", encoding="utf-8")

    rc = verify_8gb_baseline.main(
        [
            "--benchmark",
            str(benchmark),
            "--time-log",
            str(time_log),
            "--skip-dmesg",
            "--allow-missing-oom-sources",
            "--disk-path",
            str(tmp_path),
            "--disk-budget-gib",
            "1.0",
            "--output",
            str(tmp_path / "gate.json"),
        ]
    )
    assert rc == 0
