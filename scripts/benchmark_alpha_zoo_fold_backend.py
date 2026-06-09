#!/usr/bin/env python3
"""Benchmark Alpha Zoo fold symbol simulation backends.

This is intentionally small and deterministic: it compares the legacy
DataFrame-based simulator with the clean-new-alpha runner's cached array/native
path while asserting numerical parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.alpha_zoo import native_alpha_fold_backend  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_clean_new_alpha_discovery as clean_new  # noqa: E402


def _synthetic_case(rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260609)
    log_returns = rng.normal(0.00002, 0.002, size=rows)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    spread = np.maximum(0.001, np.abs(rng.normal(0.002, 0.001, size=rows)))
    signal_raw = rng.choice(
        [-1.0, 0.0, 1.0],
        size=max(1, int(np.ceil(rows / 24))),
        p=[0.2, 0.55, 0.25],
    )
    signal = np.repeat(signal_raw, 24)[:rows].astype(float)
    bars = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=rows, freq="h"),
            "open": close,
            "high": close * (1.0 + spread),
            "low": close * (1.0 - spread),
            "close": close,
            "volume": rng.uniform(1000.0, 2000.0, size=rows),
        }
    )
    return bars, signal


def _time_loop(fn, iterations: int) -> float:
    start = perf_counter()
    for _ in range(iterations):
        fn()
    return perf_counter() - start


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="auto", choices=("auto", "python", "rust"))
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a shorter deterministic benchmark for local evaluator loops.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = 8000 if args.quick else max(1000, int(args.rows))
    iterations = 200 if args.quick else max(10, int(args.iterations))
    bars, signal = _synthetic_case(rows)

    reference = broad69.simulate_symbol(
        bars,
        signal,
        integer_leverage=3,
        allocation_fraction=0.1,
        round_trip_cost_bps=10.0,
    )
    candidate = clean_new._simulate_symbol(
        bars,
        signal,
        integer_leverage=3,
        allocation_fraction=0.1,
        round_trip_cost_bps=10.0,
        simulation_backend=args.backend,
    )
    max_return_abs_diff = float(np.max(np.abs(reference.returns - candidate.returns)))
    parity = (
        max_return_abs_diff <= 1e-12
        and reference.liquidation_flags.tolist() == candidate.liquidation_flags.tolist()
        and reference.account_wipeout_flags.tolist() == candidate.account_wipeout_flags.tolist()
    )

    # Warm both paths before timing.
    broad69.simulate_symbol(
        bars,
        signal,
        integer_leverage=3,
        allocation_fraction=0.1,
        round_trip_cost_bps=10.0,
    )
    clean_new._simulate_symbol(
        bars,
        signal,
        integer_leverage=3,
        allocation_fraction=0.1,
        round_trip_cost_bps=10.0,
        simulation_backend=args.backend,
    )

    baseline_seconds = _time_loop(
        lambda: broad69.simulate_symbol(
            bars,
            signal,
            integer_leverage=3,
            allocation_fraction=0.1,
            round_trip_cost_bps=10.0,
        ),
        iterations,
    )
    backend_seconds = _time_loop(
        lambda: clean_new._simulate_symbol(
            bars,
            signal,
            integer_leverage=3,
            allocation_fraction=0.1,
            round_trip_cost_bps=10.0,
            simulation_backend=args.backend,
        ),
        iterations,
    )
    datetimes = bars["datetime"]
    train = (datetimes.iloc[0], datetimes.iloc[rows // 3])
    validation = (datetimes.iloc[rows // 3 + 1], datetimes.iloc[(rows * 2) // 3])
    locked_oos = (datetimes.iloc[(rows * 2) // 3 + 1], datetimes.iloc[-1])
    base = clean_new._candidate_base(
        family="benchmark",
        model_parts=("benchmark", "1h", "BTCUSDT", "lev3"),
        symbol="BTCUSDT",
        timeframe="1h",
        side="long_short",
        lookback=24,
        threshold=0.0,
        exit_threshold=0.0,
        min_hold=4,
        leverage=3,
        allocation_fraction=0.1,
    )

    def reference_finalize() -> dict[str, object]:
        row = broad69.finalize_candidate(
            base,
            reference,
            datetimes,
            timeframe="1h",
            windows=broad69.SplitWindows(train=train, validation=validation),
        )
        locked_mask = clean_new._window_mask(datetimes, locked_oos)
        locked = broad69.split_metrics(
            reference.returns[locked_mask],
            reference.position[locked_mask],
            reference.liquidation_flags[locked_mask],
            reference.account_wipeout_flags[locked_mask],
            timeframe="1h",
        )
        row["locked_oos_return_report_only"] = locked["total_return"]
        return row

    fast_row = clean_new._finalize_row(
        base=base,
        sim=candidate,
        datetimes=datetimes,
        timeframe="1h",
        train=train,
        validation=validation,
        locked_oos=locked_oos,
    )
    reference_row = reference_finalize()
    finalize_parity = all(
        np.isclose(
            float(fast_row.get(key) or 0.0),
            float(reference_row.get(key) or 0.0),
            rtol=0.0,
            atol=1e-12,
        )
        for key in (
            "train_return",
            "train_mdd",
            "validation_return",
            "validation_mdd",
            "train_validation_score",
            "locked_oos_return_report_only",
        )
    )
    legacy_finalize_seconds = _time_loop(reference_finalize, iterations)
    fast_finalize_seconds = _time_loop(
        lambda: clean_new._finalize_row(
            base=base,
            sim=candidate,
            datetimes=datetimes,
            timeframe="1h",
            train=train,
            validation=validation,
            locked_oos=locked_oos,
        ),
        iterations,
    )
    finalize_speedup = legacy_finalize_seconds / max(fast_finalize_seconds, 1e-12)
    speedup = baseline_seconds / max(backend_seconds, 1e-12)
    diagnostics = native_alpha_fold_backend.alpha_fold_backend_diagnostics(args.backend)
    payload = {
        "rows": rows,
        "iterations": iterations,
        "backend": diagnostics,
        "legacy_dataframe_seconds": baseline_seconds,
        "cached_native_seconds": backend_seconds,
        "legacy_finalize_seconds": legacy_finalize_seconds,
        "fast_finalize_seconds": fast_finalize_seconds,
        "speedup": speedup,
        "finalize_speedup": finalize_speedup,
        "max_return_abs_diff": max_return_abs_diff,
        "parity": parity,
        "finalize_parity": finalize_parity,
        "pass": bool(
            parity
            and finalize_parity
            and speedup >= float(args.min_speedup)
            and finalize_speedup >= float(args.min_speedup)
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
