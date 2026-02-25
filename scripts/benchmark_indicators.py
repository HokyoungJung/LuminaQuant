"""Benchmark selected indicator alpha primitives on synthetic OHLCV."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumina_quant.indicators import (  # noqa: E402
    alpha_001,
    alpha_005,
    alpha_011,
    alpha_025,
    alpha_101,
)
from scripts.benchmark_formulaic_pipeline import _synthetic_ohlcv  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark indicator alpha primitives.")
    parser.add_argument("--rows", type=int, default=6000, help="Synthetic OHLCV row count.")
    parser.add_argument("--iters", type=int, default=500, help="Iteration count.")
    parser.add_argument(
        "--output",
        default="reports/benchmarks/indicator_primitives.json",
        help="Output JSON path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = _synthetic_ohlcv(args.rows)
    opens = payload["opens"]
    highs = payload["highs"]
    lows = payload["lows"]
    closes = payload["closes"]
    volumes = payload["volumes"]
    vwaps = payload["vwaps"]

    samples: list[float] = []
    for _ in range(max(1, int(args.iters))):
        started = perf_counter()
        alpha_001(closes)
        alpha_005(opens, closes, vwaps)
        alpha_011(closes, vwaps, volumes)
        alpha_025(highs, closes, volumes, vwaps)
        alpha_101(opens, highs, lows, closes)
        samples.append(perf_counter() - started)

    median_seconds = statistics.median(samples)
    mean_seconds = statistics.fmean(samples)
    throughput = 5.0 / max(median_seconds, 1e-12)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(closes),
        "iters": int(args.iters),
        "median_seconds": float(median_seconds),
        "mean_seconds": float(mean_seconds),
        "calls_per_second": float(throughput),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark snapshot: {output}")


if __name__ == "__main__":
    main()
