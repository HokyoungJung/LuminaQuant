#!/usr/bin/env python3
"""Benchmark the monthly-refit period-metric hot path.

The full 69-asset walk-forward runner spends a lot of wall time repeatedly
asking for train/validation/OOS metrics on the same candidate return streams
while ranking selectors, hybrids, gates, and final report rows.  This benchmark
compares the pre-cache behavior (sort + boolean mask on every call) against the
current cached/searchsorted implementation on the same synthetic workload.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_monthly_refit_walkforward as runner  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402


def _legacy_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 365.0 * 24.0 * 2.0
    diffs = np.diff(index.view("int64")) / 1_000_000_000.0
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 365.0 * 24.0 * 2.0
    seconds = float(np.median(positive))
    return 365.0 * 24.0 * 60.0 * 60.0 / seconds


def _legacy_period_metrics(
    returns: pd.Series, window: tuple[pd.Timestamp, pd.Timestamp]
) -> dict[str, Any]:
    sorted_returns = returns.sort_index()
    idx = pd.DatetimeIndex(sorted_returns.index)
    mask = (idx >= window[0]) & (idx <= window[1])
    values = sorted_returns.to_numpy(dtype=float)[mask]
    period_index = idx[mask]
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    downside = values[values < 0.0]
    down_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    annual = _legacy_periods_per_year(period_index)
    total = float(np.prod(1.0 + values) - 1.0) if values.size else 0.0
    mdd = float(broad69.max_drawdown(values)) if values.size else 0.0
    return {
        "bar_count": int(values.size),
        "total_return": total,
        "mdd": mdd,
        "sharpe": mean / std * math.sqrt(annual) if std > 0.0 else 0.0,
        "sortino": mean / down_std * math.sqrt(annual) if down_std > 0.0 else 0.0,
        "calmar": total / mdd if mdd > 0.0 else 0.0,
    }


def _make_workload(
    *, candidates: int, bars: int, seed: int
) -> tuple[list[pd.Series], tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=max(8, int(bars)), freq="30min")
    series: list[pd.Series] = []
    for idx in range(max(1, int(candidates))):
        values = rng.normal(0.00005 + idx * 1e-8, 0.0015, size=len(index)).astype(float)
        series.append(pd.Series(values, index=index, dtype=float))
    windows = (
        (index[0], index[min(len(index) - 1, int(bars * 0.35))]),
        (
            index[min(len(index) - 1, int(bars * 0.35) + 1)],
            index[min(len(index) - 1, int(bars * 0.55))],
        ),
        (index[min(len(index) - 1, int(bars * 0.55) + 1)], index[-1]),
    )
    return series, windows


def _run_metrics_loop(
    fn,
    series: list[pd.Series],
    windows: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
    *,
    loops: int,
) -> tuple[float, float]:
    checksum = 0.0
    started = time.perf_counter()
    for _ in range(max(1, int(loops))):
        for returns in series:
            # Repeat train/validation metrics to mimic selector scoring,
            # downstream eligibility checks, and final row evaluation.
            for window in (windows[0], windows[1], windows[1], windows[2], windows[1]):
                row = fn(returns, window)
                checksum += float(row["total_return"]) + float(row["mdd"])
    return max(1e-12, time.perf_counter() - started), checksum


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark monthly-refit metric hot path")
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--bars", type=int, default=3072)
    parser.add_argument("--loops", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    args = parser.parse_args()

    series, windows = _make_workload(candidates=args.candidates, bars=args.bars, seed=args.seed)
    runner._clear_period_metric_caches()

    # Numerical equivalence on representative windows before timing.
    for returns in series[:3]:
        for window in windows:
            legacy = _legacy_period_metrics(returns, window)
            optimized = runner._period_metrics(returns, window)
            for field in ("bar_count", "total_return", "mdd", "sharpe", "sortino", "calmar"):
                if not math.isclose(
                    float(legacy[field]), float(optimized[field]), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise SystemExit(
                        f"metric mismatch field={field}: legacy={legacy[field]} optimized={optimized[field]}"
                    )

    runner._clear_period_metric_caches()
    legacy_elapsed, legacy_checksum = _run_metrics_loop(
        _legacy_period_metrics, series, windows, loops=args.loops
    )
    runner._clear_period_metric_caches()
    optimized_elapsed, optimized_checksum = _run_metrics_loop(
        runner._period_metrics, series, windows, loops=args.loops
    )
    speedup = legacy_elapsed / optimized_elapsed

    print("benchmark_monthly_refit_eval_hotpath")
    print(f"candidates={args.candidates} bars={args.bars} loops={args.loops}")
    print(f"legacy_elapsed_sec={legacy_elapsed:.6f}")
    print(f"optimized_elapsed_sec={optimized_elapsed:.6f}")
    print(f"speedup={speedup:.3f}x")
    print(f"legacy_checksum={legacy_checksum:.12f}")
    print(f"optimized_checksum={optimized_checksum:.12f}")
    if speedup < float(args.min_speedup):
        raise SystemExit(f"FAIL speedup {speedup:.3f}x < {float(args.min_speedup):.3f}x")
    print("PASS")


if __name__ == "__main__":
    main()
