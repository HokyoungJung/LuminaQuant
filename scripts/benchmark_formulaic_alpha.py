"""Benchmark compiled Alpha101 formula evaluation throughput."""

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

from lumina_quant.indicators.formulaic_alpha import compute_alpha101  # noqa: E402
from scripts.benchmark_formulaic_pipeline import _synthetic_ohlcv  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Alpha101 compiled formula evaluator.")
    parser.add_argument("--rows", type=int, default=6000, help="Synthetic OHLCV row count.")
    parser.add_argument("--iters", type=int, default=10, help="Iteration count.")
    parser.add_argument("--alpha-start", type=int, default=1, help="First Alpha id (inclusive).")
    parser.add_argument("--alpha-end", type=int, default=101, help="Last Alpha id (inclusive).")
    parser.add_argument("--backend", choices=("auto", "numpy", "polars"), default="auto")
    parser.add_argument(
        "--output",
        default="reports/benchmarks/formulaic_alpha.json",
        help="Output JSON path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = _synthetic_ohlcv(args.rows)
    alpha_start = max(1, int(args.alpha_start))
    alpha_end = min(101, int(args.alpha_end))
    alpha_ids = list(range(alpha_start, alpha_end + 1))
    samples: list[float] = []

    for _ in range(max(1, int(args.iters))):
        started = perf_counter()
        for alpha_id in alpha_ids:
            compute_alpha101(
                alpha_id,
                opens=payload["opens"],
                highs=payload["highs"],
                lows=payload["lows"],
                closes=payload["closes"],
                volumes=payload["volumes"],
                vwaps=payload["vwaps"],
                vector_backend=args.backend,
            )
        samples.append(perf_counter() - started)

    median_seconds = statistics.median(samples)
    mean_seconds = statistics.fmean(samples)
    throughput = float(len(alpha_ids)) / max(median_seconds, 1e-12)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(payload["closes"]),
        "iters": int(args.iters),
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "backend": args.backend,
        "median_seconds": float(median_seconds),
        "mean_seconds": float(mean_seconds),
        "alphas_per_second": float(throughput),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark snapshot: {output}")


if __name__ == "__main__":
    main()
