#!/usr/bin/env python3
"""Benchmark raw aggTrades -> canonical 1s OHLCV backends.

The public API stays Python (`raw_aggtrades_to_1s_frame`).  This script only
proves whether the optional Rust raw-first kernel is worth using underneath it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.data.native_raw_first_backend import (
    native_backend_available,
    raw_first_backend_diagnostics,
)
from lumina_quant.data.raw_first_lineage import raw_aggtrades_to_1s_frame

DEFAULT_START_MS = 1_710_000_000_000
FLOAT_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class BackendTiming:
    """Single backend timing result."""

    backend: str
    output_bars: int
    evals: int
    elapsed_seconds: float
    seconds_per_eval: float
    input_trades_per_second: float


@dataclass(frozen=True)
class ParityResult:
    """Python/Rust output parity check."""

    checked: bool
    passed: bool
    max_abs_diff: float
    reason: str | None = None


@dataclass(frozen=True)
class RawFirstBenchmarkResult:
    """Machine-readable benchmark payload."""

    status: str
    trades: int
    seconds: int
    evals: int
    rust_available: bool
    auto_backend: dict[str, Any]
    python: BackendTiming | None
    rust: BackendTiming | None
    rust_speedup_vs_python: float | None
    parity: ParityResult
    peak_memory_note: str


def generate_synthetic_aggtrades(
    *, trades: int, seconds: int, start_ms: int = DEFAULT_START_MS, seed: int = 20260527
) -> pl.DataFrame:
    """Generate deterministic, Binance-like aggTrade rows with millisecond timestamps."""
    if trades <= 0:
        raise ValueError("trades must be positive")
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    rng = np.random.default_rng(seed)
    timestamp_offsets = np.sort(rng.integers(0, seconds * 1000, size=trades, dtype=np.int64))
    prices = 100.0 + np.cumsum(rng.normal(0.0, 0.02, size=trades))
    prices = np.maximum(prices, 1.0)
    quantities = rng.lognormal(mean=-2.0, sigma=0.6, size=trades)

    return pl.DataFrame(
        {
            "agg_trade_id": np.arange(trades, dtype=np.int64),
            "timestamp_ms": start_ms + timestamp_offsets,
            "price": prices,
            "quantity": quantities,
            "is_buyer_maker": rng.random(trades) < 0.5,
        }
    )


def _time_backend(
    raw: pl.DataFrame,
    *,
    backend: str,
    seconds: int,
    evals: int,
    start_ms: int = DEFAULT_START_MS,
) -> tuple[BackendTiming, pl.DataFrame]:
    kwargs = {
        "source": "rawfirst-benchmark",
        "range_start_ms": start_ms,
        "range_end_ms": start_ms + seconds * 1000,
        "complete_through_ms": start_ms + seconds * 1000 - 1,
        "previous_close": 100.0,
        "backend": backend,
    }
    # Warm once to pay import/ctypes/JIT-like setup outside the timed loop.
    output = raw_aggtrades_to_1s_frame(raw, **kwargs)

    started = time.perf_counter()
    for _ in range(evals):
        output = raw_aggtrades_to_1s_frame(raw, **kwargs)
    elapsed = time.perf_counter() - started
    return (
        BackendTiming(
            backend=backend,
            output_bars=output.height,
            evals=evals,
            elapsed_seconds=elapsed,
            seconds_per_eval=elapsed / evals,
            input_trades_per_second=(raw.height * evals) / elapsed if elapsed > 0 else 0.0,
        ),
        output,
    )


def _check_parity(
    python_frame: pl.DataFrame | None, rust_frame: pl.DataFrame | None
) -> ParityResult:
    if python_frame is None or rust_frame is None:
        return ParityResult(checked=False, passed=False, max_abs_diff=0.0, reason="missing backend")
    if python_frame.shape != rust_frame.shape:
        return ParityResult(
            checked=True,
            passed=False,
            max_abs_diff=float("inf"),
            reason=f"shape mismatch: python={python_frame.shape} rust={rust_frame.shape}",
        )
    if python_frame.get_column("datetime").to_list() != rust_frame.get_column("datetime").to_list():
        return ParityResult(
            checked=True,
            passed=False,
            max_abs_diff=float("inf"),
            reason="datetime mismatch",
        )

    max_abs_diff = 0.0
    for column in FLOAT_COLUMNS:
        diff = (python_frame.get_column(column) - rust_frame.get_column(column)).abs().max()
        max_abs_diff = max(max_abs_diff, float(diff or 0.0))
    return ParityResult(checked=True, passed=True, max_abs_diff=max_abs_diff)


def run_benchmark(
    *,
    trades: int,
    seconds: int,
    evals: int,
    backend: str,
    require_rust: bool = False,
    require_speedup: bool = False,
    min_speedup: float = 1.0,
) -> RawFirstBenchmarkResult:
    """Run the benchmark and return a serializable result payload."""
    if evals <= 0:
        raise ValueError("evals must be positive")
    auto_backend = raw_first_backend_diagnostics("auto")
    rust_available = native_backend_available()
    rust_diag = raw_first_backend_diagnostics("rust")
    if require_rust and not rust_available:
        return RawFirstBenchmarkResult(
            status="rust_unavailable",
            trades=trades,
            seconds=seconds,
            evals=evals,
            rust_available=False,
            auto_backend=auto_backend,
            python=None,
            rust=None,
            rust_speedup_vs_python=None,
            parity=ParityResult(
                checked=False,
                passed=False,
                max_abs_diff=0.0,
                reason=str(rust_diag.get("native_load_error") or "rust backend unavailable"),
            ),
            peak_memory_note="synthetic single-frame benchmark; keep trades bounded for <8GB sessions",
        )

    raw = generate_synthetic_aggtrades(trades=trades, seconds=seconds)
    python_timing: BackendTiming | None = None
    rust_timing: BackendTiming | None = None
    python_frame: pl.DataFrame | None = None
    rust_frame: pl.DataFrame | None = None

    if backend in {"python", "both"}:
        python_timing, python_frame = _time_backend(
            raw, backend="python", seconds=seconds, evals=evals
        )
    if backend in {"rust", "both"} and rust_available:
        rust_timing, rust_frame = _time_backend(raw, backend="rust", seconds=seconds, evals=evals)

    parity = (
        _check_parity(python_frame, rust_frame)
        if backend == "both"
        else ParityResult(
            checked=False,
            passed=True,
            max_abs_diff=0.0,
            reason="single-backend run",
        )
    )
    speedup = None
    if python_timing is not None and rust_timing is not None:
        speedup = python_timing.seconds_per_eval / rust_timing.seconds_per_eval

    status = "pass"
    if backend == "both" and rust_available and not parity.passed:
        status = "parity_failed"
    elif require_speedup and (speedup is None or speedup < min_speedup):
        status = "speedup_failed"
    elif backend in {"rust", "both"} and not rust_available:
        status = "rust_unavailable"

    return RawFirstBenchmarkResult(
        status=status,
        trades=trades,
        seconds=seconds,
        evals=evals,
        rust_available=rust_available,
        auto_backend=auto_backend,
        python=python_timing,
        rust=rust_timing,
        rust_speedup_vs_python=speedup,
        parity=parity,
        peak_memory_note="synthetic single-frame benchmark; keep trades bounded for <8GB sessions",
    )


def _to_jsonable(result: RawFirstBenchmarkResult) -> dict[str, Any]:
    payload = asdict(result)
    # JSON has no representation for infinity; retain a readable failure marker instead.
    if payload["parity"]["max_abs_diff"] == float("inf"):
        payload["parity"]["max_abs_diff"] = "inf"
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=int, default=200_000, help="synthetic aggTrade rows")
    parser.add_argument("--seconds", type=int, default=60_000, help="1s buckets to materialize")
    parser.add_argument("--evals", type=int, default=3, help="timed evaluations per backend")
    parser.add_argument(
        "--backend",
        choices=("python", "rust", "both"),
        default="both",
        help="backend set to benchmark",
    )
    parser.add_argument(
        "--require-rust", action="store_true", help="fail if Rust backend is unavailable"
    )
    parser.add_argument(
        "--require-speedup",
        action="store_true",
        help="fail if Rust speedup is below --min-speedup",
    )
    parser.add_argument(
        "--min-speedup", type=float, default=1.0, help="minimum Rust/Python speedup"
    )
    parser.add_argument("--output-json", type=Path, help="write JSON result to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_benchmark(
        trades=args.trades,
        seconds=args.seconds,
        evals=args.evals,
        backend=args.backend,
        require_rust=args.require_rust,
        require_speedup=args.require_speedup,
        min_speedup=args.min_speedup,
    )
    payload = _to_jsonable(result)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    rust_missing_but_optional = result.status == "rust_unavailable" and not (
        args.require_rust or args.require_speedup
    )
    return 0 if result.status == "pass" or rust_missing_but_optional else 2


if __name__ == "__main__":
    raise SystemExit(main())
