"""Compare Python/Numba/Native metric backends for speed and parity."""

from __future__ import annotations

import argparse
import importlib
import os
import time
from collections.abc import Callable
from typing import cast

import numpy as np
from lumina_quant.optimization.fast_eval import NUMBA_AVAILABLE, evaluate_metrics_numba


def _evaluate_python(total_series: np.ndarray, annual_periods: int) -> tuple[float, float, float]:
    if total_series.size < 2:
        return -999.0, 0.0, 0.0

    prev_total = total_series[:-1]
    next_total = total_series[1:]
    returns = np.divide(
        next_total - prev_total,
        np.where(prev_total == 0.0, 1.0, prev_total),
        dtype=np.float64,
    )
    mean_r = float(np.mean(returns)) if returns.size > 0 else 0.0
    std_r = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = -999.0
    if std_r > 0.0:
        sharpe = (mean_r / std_r) * np.sqrt(float(max(1, annual_periods)))

    initial = float(total_series[0])
    final = float(total_series[-1])
    years = float(total_series.size) / float(max(1, annual_periods))
    if initial <= 0.0 or years <= 0.0:
        cagr = 0.0
    else:
        cagr = (final / initial) ** (1.0 / years) - 1.0

    peak = np.maximum.accumulate(total_series)
    dd = np.divide(peak - total_series, np.where(peak == 0.0, 1.0, peak), dtype=np.float64)
    mdd = float(np.max(dd)) if dd.size > 0 else 0.0
    return float(sharpe), float(cagr), float(mdd)


def _bench(
    fn: Callable[[np.ndarray, int], tuple[float, float, float]],
    arr: np.ndarray,
    annual_periods: int,
    loops: int,
) -> tuple[float, tuple[float, float, float]]:
    loops_i = max(1, int(loops))
    out = fn(arr, annual_periods)
    start = time.perf_counter()
    for _ in range(loops_i):
        out = fn(arr, annual_periods)
    elapsed = max(1e-9, time.perf_counter() - start)
    out_tuple = cast(tuple[float, float, float], out)
    return float(loops_i) / elapsed, (
        float(out_tuple[0]),
        float(out_tuple[1]),
        float(out_tuple[2]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Python/Numba/Native metric backends")
    parser.add_argument("--bars", type=int, default=50_000)
    parser.add_argument("--evals", type=int, default=5_000)
    parser.add_argument("--dll", default="", help="Optional path to native metrics DLL")
    args = parser.parse_args()

    if str(args.dll).strip():
        os.environ["LQ_NATIVE_METRICS_DLL"] = str(args.dll).strip()
        os.environ["LQ_NATIVE_AUTO_SELECT"] = "0"

    native_backend = importlib.import_module("lumina_quant.optimization.native_backend")
    native_backend = importlib.reload(native_backend)

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.001, size=max(2, int(args.bars))).astype(np.float64)
    totals = (1.0 + returns).cumprod() * 10_000.0

    py_speed, py_out = _bench(_evaluate_python, totals, 252, int(args.evals))
    nb_speed = 0.0
    nb_out = py_out
    if NUMBA_AVAILABLE:
        evaluate_metrics_numba(totals, 252)
        nb_speed, nb_out = _bench(evaluate_metrics_numba, totals, 252, int(args.evals))

    native_fn = native_backend.evaluate_metrics_backend
    native_speed, native_out = _bench(native_fn, totals, 252, int(args.evals))

    print("benchmark_native_compare")
    print(f"native_backend_name={native_backend.NATIVE_BACKEND_NAME}")
    print(f"python_eval_per_sec={py_speed:.2f}")
    print(f"numba_available={NUMBA_AVAILABLE}")
    print(f"numba_eval_per_sec={nb_speed:.2f}")
    print(f"native_eval_per_sec={native_speed:.2f}")
    print(
        "python_vs_native_abs_diff="
        f"({abs(py_out[0] - native_out[0]):.12f}, {abs(py_out[1] - native_out[1]):.12f}, {abs(py_out[2] - native_out[2]):.12f})"
    )
    print(
        "numba_vs_native_abs_diff="
        f"({abs(nb_out[0] - native_out[0]):.12f}, {abs(nb_out[1] - native_out[1]):.12f}, {abs(nb_out[2] - native_out[2]):.12f})"
    )


if __name__ == "__main__":
    main()
