"""Benchmark indicator, Alpha101 formulaic, and backtest-loop workloads."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (PROJECT_ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lumina_quant.indicators import (  # noqa: E402
    alpha_001,
    alpha_005,
    alpha_011,
    alpha_025,
    alpha_101,
)
from lumina_quant.indicators.formulaic_alpha import compute_alpha101  # noqa: E402

import benchmark_backtest as backtest_benchmark  # noqa: E402


@dataclass(slots=True)
class SectionSummary:
    name: str
    iterations: int
    median_seconds: float
    mean_seconds: float
    throughput: float
    extra: dict[str, float | int | None]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark formulaic alpha execution pipeline.")
    parser.add_argument("--rows", type=int, default=6000, help="Synthetic OHLCV row count.")
    parser.add_argument("--indicator-iters", type=int, default=250, help="Indicator workload iterations.")
    parser.add_argument("--formula-iters", type=int, default=10, help="Formulaic workload iterations.")
    parser.add_argument("--alpha-start", type=int, default=1, help="First Alpha id (inclusive).")
    parser.add_argument("--alpha-end", type=int, default=101, help="Last Alpha id (inclusive).")
    parser.add_argument("--backend", choices=("auto", "numpy", "polars"), default="auto")
    parser.add_argument("--config", default="config.yaml", help="Backtest config path.")
    parser.add_argument("--strategy", default=None, help="Backtest strategy override.")
    parser.add_argument("--symbols", default="", help="Backtest symbols override.")
    parser.add_argument("--backtest-iters", type=int, default=1, help="Backtest benchmark iterations.")
    parser.add_argument("--backtest-warmup", type=int, default=0, help="Backtest warmup iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for backtest benchmark.")
    parser.add_argument(
        "--output",
        default="reports/benchmarks/formulaic_pipeline.json",
        help="Output JSON path.",
    )
    return parser


def _synthetic_ohlcv(rows: int) -> dict[str, list[float]]:
    count = max(300, int(rows))
    idx = np.arange(count, dtype=float)
    trend = 100.0 + (0.03 * idx)
    cycle = np.sin(idx / 37.0) * 0.8
    closes = trend + cycle
    opens = closes - 0.15
    highs = closes + 0.45
    lows = closes - 0.55
    vwaps = (highs + lows + closes) / 3.0
    volumes = 1500.0 + (idx * 3.5) + (np.cos(idx / 19.0) * 40.0)
    return {
        "opens": opens.tolist(),
        "highs": highs.tolist(),
        "lows": lows.tolist(),
        "closes": closes.tolist(),
        "volumes": volumes.tolist(),
        "vwaps": vwaps.tolist(),
    }


def _run_timed(iterations: int, workload) -> list[float]:
    samples: list[float] = []
    for _ in range(max(1, int(iterations))):
        started = perf_counter()
        workload()
        samples.append(perf_counter() - started)
    return samples


def _summarize(
    *,
    name: str,
    iterations: int,
    samples: list[float],
    units_per_iter: float,
    extra: dict[str, float | int | None] | None = None,
) -> SectionSummary:
    median_seconds = statistics.median(samples)
    mean_seconds = statistics.fmean(samples)
    throughput = units_per_iter / max(median_seconds, 1e-12)
    return SectionSummary(
        name=name,
        iterations=int(iterations),
        median_seconds=float(median_seconds),
        mean_seconds=float(mean_seconds),
        throughput=float(throughput),
        extra=dict(extra or {}),
    )


def main() -> None:
    args = _build_parser().parse_args()
    payload = _synthetic_ohlcv(args.rows)
    opens = payload["opens"]
    highs = payload["highs"]
    lows = payload["lows"]
    closes = payload["closes"]
    volumes = payload["volumes"]
    vwaps = payload["vwaps"]

    alpha_start = max(1, int(args.alpha_start))
    alpha_end = min(101, int(args.alpha_end))
    alpha_ids = list(range(alpha_start, alpha_end + 1))

    indicator_samples = _run_timed(
        args.indicator_iters,
        lambda: (
            alpha_001(closes),
            alpha_005(opens, closes, vwaps),
            alpha_011(closes, vwaps, volumes),
            alpha_025(highs, closes, volumes, vwaps),
            alpha_101(opens, highs, lows, closes),
        ),
    )
    indicator_summary = _summarize(
        name="indicators",
        iterations=args.indicator_iters,
        samples=indicator_samples,
        units_per_iter=5.0,
        extra={"rows": len(closes)},
    )

    def formulaic_workload() -> None:
        for alpha_id in alpha_ids:
            compute_alpha101(
                alpha_id,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                vwaps=vwaps,
                vector_backend=args.backend,
            )

    formulaic_samples = _run_timed(args.formula_iters, formulaic_workload)
    formulaic_summary = _summarize(
        name="formulaic_alpha101",
        iterations=args.formula_iters,
        samples=formulaic_samples,
        units_per_iter=float(len(alpha_ids)),
        extra={"alpha_start": alpha_start, "alpha_end": alpha_end, "backend": args.backend},
    )

    backtest_section: SectionSummary
    if backtest_benchmark.STRATEGY_MAP:
        bt_args = argparse.Namespace(
            config=args.config,
            symbols=args.symbols,
            strategy=args.strategy,
            iters=max(1, int(args.backtest_iters)),
            warmup=max(0, int(args.backtest_warmup)),
            seed=int(args.seed),
            record_history=False,
            output=args.output,
            compare_to="",
        )
        started = perf_counter()
        bt_summary = backtest_benchmark.build_benchmark_summary(bt_args)
        elapsed = perf_counter() - started
        backtest_section = SectionSummary(
            name="backtest_loop",
            iterations=int(bt_summary.iterations),
            median_seconds=float(bt_summary.median_seconds),
            mean_seconds=float(bt_summary.mean_seconds),
            throughput=float(bt_summary.median_bars_per_sec),
            extra={
                "elapsed_seconds": float(elapsed),
                "bars_per_sec": float(bt_summary.median_bars_per_sec),
            },
        )
    else:
        backtest_section = SectionSummary(
            name="backtest_loop",
            iterations=0,
            median_seconds=math.nan,
            mean_seconds=math.nan,
            throughput=0.0,
            extra={"skipped": 1, "reason": "strategies package unavailable"},
        )

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(closes),
        "sections": [
            asdict(indicator_summary),
            asdict(formulaic_summary),
            asdict(backtest_section),
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark snapshot: {output_path}")


if __name__ == "__main__":
    main()
