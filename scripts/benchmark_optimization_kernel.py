"""Benchmark frozen-dataset metric kernel throughput.

Reports eval/sec for repeated metric evaluation calls.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from lumina_quant.optimization.fast_eval import evaluate_metrics_numba


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark optimization metric kernel throughput")
    parser.add_argument("--bars", type=int, default=50_000)
    parser.add_argument("--evals", type=int, default=5_000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.001, size=max(2, int(args.bars))).astype(np.float64)
    totals = (1.0 + returns).cumprod() * 10_000.0

    evaluate_metrics_numba(totals, 252)

    start = time.perf_counter()
    for _ in range(max(1, int(args.evals))):
        evaluate_metrics_numba(totals, 252)
    elapsed = max(1e-9, time.perf_counter() - start)
    eps = float(args.evals) / elapsed

    print("benchmark_optimization_kernel")
    print(f"bars={int(args.bars)} evals={int(args.evals)}")
    print(f"elapsed_sec={elapsed:.6f}")
    print(f"evals_per_sec={eps:.2f}")


if __name__ == "__main__":
    main()
