"""Verify 8GB baseline readiness using benchmark, RSS, OOM, and disk checks."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIME_RSS_PATTERN = re.compile(
    r"Maximum resident set size \(kbytes\):\s*(\d+)",
    flags=re.IGNORECASE,
)
OOM_PATTERN = re.compile(
    r"(killed process|out of memory|oom-killer|oom kill|memory cgroup out of memory)",
    flags=re.IGNORECASE,
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _peak_rss_from_benchmark(payload: dict[str, Any]) -> float | None:
    direct = _as_float(payload.get("max_peak_rss_mb"))
    if direct is not None:
        return direct

    candidates: list[float] = []
    for sample in payload.get("samples", []):
        if not isinstance(sample, dict):
            continue
        value = _as_float(sample.get("peak_rss_mb"))
        if value is not None:
            candidates.append(value)

    if not candidates:
        return None
    return max(candidates)


def _peak_rss_from_time_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = [float(group) for group in TIME_RSS_PATTERN.findall(text)]
    if not matches:
        return None

    rss_kib = max(matches)
    return rss_kib / 1024.0


def _load_benchmark(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Benchmark payload must be a JSON object: {path}"
        raise ValueError(msg)
    return payload


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _find_oom_hits(text: str, *, limit: int = 5) -> list[str]:
    hits: list[str] = []
    for line in text.splitlines():
        if OOM_PATTERN.search(line):
            hits.append(line.strip())
            if len(hits) >= limit:
                break
    return hits


def _collect_log_files(explicit_logs: list[str]) -> list[Path]:
    if explicit_logs:
        return [_resolve_repo_path(raw) for raw in explicit_logs]

    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        return []
    return sorted(path for path in logs_dir.glob("*.log") if path.is_file())


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        try:
            total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _bytes_to_gib(value: int) -> float:
    return float(value) / (1024.0**3)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify 8GB baseline gates.")
    parser.add_argument(
        "--benchmark-json",
        "--benchmark",
        dest="benchmark_json",
        default="reports/benchmarks/ci_smoke.json",
        help="Benchmark JSON output path.",
    )
    parser.add_argument(
        "--rss-log",
        "--time-log",
        dest="rss_log",
        default="",
        help="Optional /usr/bin/time -v output log path.",
    )
    parser.add_argument(
        "--rss-mib",
        type=float,
        default=None,
        help="Explicit RSS value in MiB. Highest RSS source is enforced.",
    )
    parser.add_argument(
        "--rss-limit-gib",
        type=float,
        default=7.2,
        help="RSS limit in GiB (default: 7.2).",
    )
    parser.add_argument(
        "--allow-missing-rss-source",
        action="store_true",
        help="Do not fail when RSS sources are unavailable (fallback mode).",
    )
    parser.add_argument(
        "--oom-log",
        action="append",
        default=[],
        help="Optional explicit log path to scan for OOM signatures (repeatable).",
    )
    parser.add_argument(
        "--skip-dmesg",
        action="store_true",
        help="Skip dmesg OOM scan (useful in restricted CI environments).",
    )
    parser.add_argument(
        "--allow-missing-oom-sources",
        action="store_true",
        help="Do not fail if no OOM sources were available to scan.",
    )
    parser.add_argument(
        "--disk-path",
        action="append",
        dest="disk_paths",
        default=[],
        help="Disk path to include in snapshot (repeatable). Defaults to data/logs/reports.",
    )
    parser.add_argument(
        "--disk-budget-gib",
        type=float,
        default=30.0,
        help="Combined disk budget for tracked paths in GiB (default: 30).",
    )
    parser.add_argument(
        "--output",
        default="reports/benchmarks/8gb_baseline_gate.json",
        help="Output JSON path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    benchmark_path = _resolve_repo_path(args.benchmark_json)
    output_path = _resolve_repo_path(args.output)
    rss_log_path = _resolve_repo_path(args.rss_log) if args.rss_log else None
    disk_paths = args.disk_paths or ["data", "logs", "reports"]

    checks: dict[str, dict[str, Any]] = {}

    try:
        benchmark_payload = _load_benchmark(benchmark_path)
        if bool(benchmark_payload.get("skipped")):
            checks["benchmark_parse"] = {
                "status": "PASS",
                "detail": (
                    f"Benchmark reported skipped mode: "
                    f"{benchmark_payload.get('reason', 'no reason provided')}"
                ),
            }
        else:
            missing_keys = [
                key
                for key in ("iterations", "median_seconds", "median_bars_per_sec")
                if key not in benchmark_payload
            ]
            if missing_keys:
                checks["benchmark_parse"] = {
                    "status": "FAIL",
                    "detail": f"Missing benchmark keys: {', '.join(missing_keys)}",
                }
            else:
                checks["benchmark_parse"] = {
                    "status": "PASS",
                    "detail": (
                        f"Loaded {benchmark_path} (iterations={benchmark_payload.get('iterations')}, "
                        f"median_seconds={benchmark_payload.get('median_seconds')}, "
                        f"median_bars_per_sec={benchmark_payload.get('median_bars_per_sec')})"
                    ),
                }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        benchmark_payload = {}
        checks["benchmark_parse"] = {
            "status": "FAIL",
            "detail": f"Unable to parse benchmark JSON: {exc}",
        }

    rss_candidates: list[tuple[str, float]] = []
    if args.rss_mib is not None:
        rss_candidates.append(("--rss-mib", float(args.rss_mib)))

    if rss_log_path is not None:
        parsed_rss = _peak_rss_from_time_log(rss_log_path)
        if parsed_rss is not None:
            rss_candidates.append((f"rss_log:{rss_log_path}", parsed_rss))

    bench_rss = _peak_rss_from_benchmark(benchmark_payload)
    if bench_rss is not None:
        rss_candidates.append((f"benchmark:{benchmark_path}", bench_rss))

    rss_limit_mib = float(args.rss_limit_gib) * 1024.0
    if not rss_candidates:
        if args.allow_missing_rss_source:
            checks["rss_budget"] = {
                "status": "PASS",
                "detail": (
                    "No RSS source available; skipped due --allow-missing-rss-source. "
                    "Use --rss-log or --rss-mib for strict enforcement."
                ),
            }
        else:
            checks["rss_budget"] = {
                "status": "FAIL",
                "detail": "No RSS source available (provide --rss-mib, --rss-log, or benchmark max_peak_rss_mb).",
            }
    else:
        rss_source, rss_peak_mib = max(rss_candidates, key=lambda item: item[1])
        if rss_peak_mib < rss_limit_mib:
            checks["rss_budget"] = {
                "status": "PASS",
                "detail": (
                    f"peak_rss={rss_peak_mib:.2f}MiB < limit={rss_limit_mib:.2f}MiB "
                    f"(source={rss_source})"
                ),
            }
        else:
            checks["rss_budget"] = {
                "status": "FAIL",
                "detail": (
                    f"peak_rss={rss_peak_mib:.2f}MiB >= limit={rss_limit_mib:.2f}MiB "
                    f"(source={rss_source})"
                ),
            }

    oom_hits: list[dict[str, Any]] = []
    source_count = 0

    for log_path in _collect_log_files(args.oom_log):
        if not log_path.exists():
            continue
        source_count += 1
        text = log_path.read_text(encoding="utf-8", errors="replace")
        hits = _find_oom_hits(text)
        if hits:
            oom_hits.append({"source": str(log_path), "hits": hits})

    if not args.skip_dmesg:
        proc = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            source_count += 1
            hits = _find_oom_hits(proc.stdout)
            if hits:
                oom_hits.append({"source": "dmesg", "hits": hits})
        elif not args.allow_missing_oom_sources:
            checks["oom_evidence"] = {
                "status": "FAIL",
                "detail": (
                    "dmesg scan failed. Use --skip-dmesg in restricted environments or "
                    "--allow-missing-oom-sources."
                ),
            }

    if "oom_evidence" not in checks:
        if oom_hits:
            first_hit = oom_hits[0]["hits"][0]
            checks["oom_evidence"] = {
                "status": "FAIL",
                "detail": f"Detected OOM signatures (first hit: {first_hit})",
                "hits": oom_hits,
            }
        elif source_count == 0 and not args.allow_missing_oom_sources:
            checks["oom_evidence"] = {
                "status": "FAIL",
                "detail": "No OOM evidence source available (logs or dmesg).",
            }
        else:
            checks["oom_evidence"] = {
                "status": "PASS",
                "detail": "No OOM signatures detected.",
            }

    disk_snapshot: list[dict[str, Any]] = []
    total_bytes = 0
    for raw_path in disk_paths:
        resolved = _resolve_repo_path(raw_path)
        size_bytes = _size_bytes(resolved)
        total_bytes += size_bytes
        disk_snapshot.append(
            {
                "path": str(raw_path),
                "resolved": str(resolved),
                "exists": resolved.exists(),
                "size_bytes": size_bytes,
                "size_gib": round(_bytes_to_gib(size_bytes), 6),
            }
        )

    total_gib = _bytes_to_gib(total_bytes)
    if total_gib <= float(args.disk_budget_gib):
        checks["disk_budget_snapshot"] = {
            "status": "PASS",
            "detail": f"disk_total={total_gib:.3f}GiB <= budget={float(args.disk_budget_gib):.3f}GiB",
            "snapshot": disk_snapshot,
        }
    else:
        checks["disk_budget_snapshot"] = {
            "status": "FAIL",
            "detail": f"disk_total={total_gib:.3f}GiB > budget={float(args.disk_budget_gib):.3f}GiB",
            "snapshot": disk_snapshot,
        }

    failed = [name for name, check in checks.items() if check["status"] == "FAIL"]
    overall_status = "PASS" if not failed else "FAIL"

    payload: dict[str, Any] = {
        "overall_status": overall_status,
        "benchmark_json": str(benchmark_path),
        "rss_limit_gib": float(args.rss_limit_gib),
        "disk_budget_gib": float(args.disk_budget_gib),
        "checks": checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name in ("benchmark_parse", "rss_budget", "oom_evidence", "disk_budget_snapshot"):
        check = checks[name]
        print(f"[{check['status']}] {name}: {check['detail']}")
    print(f"Report written to {output_path}")

    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
