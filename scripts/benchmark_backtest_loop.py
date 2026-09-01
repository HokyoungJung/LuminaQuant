"""Thin wrapper benchmark for the backtest loop used in CI/regression gates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (PROJECT_ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import benchmark_backtest as backtest_benchmark  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LuminaQuant backtest loop.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols override.")
    parser.add_argument("--strategy", default=None, help="Strategy class name override.")
    parser.add_argument("--iters", type=int, default=1, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument(
        "--record-history",
        action="store_true",
        help="Keep full portfolio/trade history during benchmark run.",
    )
    parser.add_argument(
        "--output",
        default="reports/benchmarks/backtest_loop.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--compare-to",
        default="",
        help="Optional path to previous benchmark JSON snapshot for delta report.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not backtest_benchmark.STRATEGY_MAP:
        payload = {
            "skipped": True,
            "reason": "strategies package unavailable in this distribution",
        }
    else:
        summary = backtest_benchmark.build_benchmark_summary(args)
        payload = asdict(summary)
        previous = backtest_benchmark._load_snapshot(args.compare_to)
        if previous is not None:
            payload["comparison"] = backtest_benchmark._build_comparison(summary, previous)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark snapshot: {output}")


if __name__ == "__main__":
    main()
