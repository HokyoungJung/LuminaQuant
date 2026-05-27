#!/usr/bin/env python3
"""Benchmark Python vs Rust live-signal state-machine kernels."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Iterator

import numpy as np
import pandas as pd

from lumina_quant.alpha_zoo import native_live_signal_backend as native_backend
from lumina_quant.alpha_zoo.optuna_hybrid_signals import (
    debounced_state_signal,
    trailing_state_signal,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "var"
    / "reports"
    / "native_acceleration_20260527"
    / "live_signal_backend_benchmark_latest.json"
)


@dataclass(frozen=True)
class SyntheticInputs:
    close: np.ndarray
    atr: np.ndarray
    long_entry: np.ndarray
    long_exit: np.ndarray
    short_entry: np.ndarray
    short_exit: np.ndarray


@dataclass(frozen=True)
class BackendTiming:
    backend: str
    debounced_seconds_per_eval: float
    trailing_seconds_per_eval: float
    total_seconds_per_eval: float


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    debounced_exact: bool
    trailing_exact: bool
    max_abs_diff: float
    tolerance: float
    reason: str | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    generated_at_utc: str
    status: str
    rows: int
    evals: int
    python: BackendTiming | None
    rust: BackendTiming | None
    speedup_total: float | None
    parity: ParityResult
    diagnostics: dict[str, object]


@contextmanager
def _backend_mode(mode: str) -> Iterator[None]:
    previous = os.environ.get(native_backend.LIVE_SIGNAL_BACKEND_ENV)
    os.environ[native_backend.LIVE_SIGNAL_BACKEND_ENV] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(native_backend.LIVE_SIGNAL_BACKEND_ENV, None)
        else:
            os.environ[native_backend.LIVE_SIGNAL_BACKEND_ENV] = previous


def generate_synthetic_inputs(*, rows: int) -> SyntheticInputs:
    rng = np.random.default_rng(20260527)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.22, size=int(rows))).astype(np.float64)
    atr = np.maximum(0.05, rng.lognormal(mean=-2.7, sigma=0.35, size=int(rows))).astype(np.float64)
    long_entry = rng.random(int(rows)) < 0.055
    long_exit = rng.random(int(rows)) < 0.045
    short_entry = rng.random(int(rows)) < 0.050
    short_exit = rng.random(int(rows)) < 0.040
    return SyntheticInputs(
        close=close,
        atr=atr,
        long_entry=long_entry,
        long_exit=long_exit,
        short_entry=short_entry,
        short_exit=short_exit,
    )


def _as_series(values: np.ndarray) -> pd.Series:
    return pd.Series(values)


def _evaluate_once(inputs: SyntheticInputs) -> tuple[np.ndarray, np.ndarray]:
    debounced = debounced_state_signal(
        _as_series(inputs.long_entry),
        _as_series(inputs.long_exit),
        _as_series(inputs.short_entry),
        _as_series(inputs.short_exit),
        side="long_short",
        min_hold_bars=5,
        cooldown_bars=3,
    )
    trailing = trailing_state_signal(
        _as_series(inputs.close),
        _as_series(inputs.long_entry),
        _as_series(inputs.short_entry),
        _as_series(inputs.long_exit),
        _as_series(inputs.short_exit),
        _as_series(inputs.atr),
        side="long_short",
        min_hold_bars=4,
        cooldown_bars=2,
        trail_atr_mult=2.25,
    )
    return debounced, trailing


def _time_backend(
    inputs: SyntheticInputs, *, mode: str, evals: int
) -> tuple[BackendTiming, tuple[np.ndarray, np.ndarray]]:
    eval_count = max(1, int(evals))
    with _backend_mode(mode):
        first = _evaluate_once(inputs)
        started = time.perf_counter()
        result = first
        for _ in range(eval_count):
            result = _evaluate_once(inputs)
        elapsed = time.perf_counter() - started
    total_per_eval = elapsed / float(eval_count)

    # Measure each kernel separately for attribution without replacing the total timing.
    with _backend_mode(mode):
        started = time.perf_counter()
        for _ in range(eval_count):
            _ = debounced_state_signal(
                _as_series(inputs.long_entry),
                _as_series(inputs.long_exit),
                _as_series(inputs.short_entry),
                _as_series(inputs.short_exit),
                side="long_short",
                min_hold_bars=5,
                cooldown_bars=3,
            )
        debounced_per_eval = (time.perf_counter() - started) / float(eval_count)
        started = time.perf_counter()
        for _ in range(eval_count):
            _ = trailing_state_signal(
                _as_series(inputs.close),
                _as_series(inputs.long_entry),
                _as_series(inputs.short_entry),
                _as_series(inputs.long_exit),
                _as_series(inputs.short_exit),
                _as_series(inputs.atr),
                side="long_short",
                min_hold_bars=4,
                cooldown_bars=2,
                trail_atr_mult=2.25,
            )
        trailing_per_eval = (time.perf_counter() - started) / float(eval_count)

    return (
        BackendTiming(
            backend=mode,
            debounced_seconds_per_eval=float(debounced_per_eval),
            trailing_seconds_per_eval=float(trailing_per_eval),
            total_seconds_per_eval=float(total_per_eval),
        ),
        result,
    )


def _parity(
    py_result: tuple[np.ndarray, np.ndarray] | None,
    rust_result: tuple[np.ndarray, np.ndarray] | None,
    *,
    reason: str | None = None,
) -> ParityResult:
    if py_result is None or rust_result is None:
        return ParityResult(
            passed=False,
            debounced_exact=False,
            trailing_exact=False,
            max_abs_diff=float("inf"),
            tolerance=0.0,
            reason=reason,
        )
    debounced_exact = bool(np.array_equal(py_result[0], rust_result[0]))
    trailing_exact = bool(np.array_equal(py_result[1], rust_result[1]))
    max_abs_diff = float(
        max(
            np.max(np.abs(py_result[0] - rust_result[0])) if len(py_result[0]) else 0.0,
            np.max(np.abs(py_result[1] - rust_result[1])) if len(py_result[1]) else 0.0,
        )
    )
    return ParityResult(
        passed=debounced_exact and trailing_exact,
        debounced_exact=debounced_exact,
        trailing_exact=trailing_exact,
        max_abs_diff=max_abs_diff,
        tolerance=0.0,
        reason=None if debounced_exact and trailing_exact else "state_signal_mismatch",
    )


def run_benchmark(
    *,
    rows: int,
    evals: int,
    backend: str,
    require_rust: bool,
) -> BenchmarkResult:
    inputs = generate_synthetic_inputs(rows=max(1, int(rows)))
    diagnostics = native_backend.live_signal_backend_diagnostics("auto")
    if require_rust and not native_backend.native_backend_available():
        return BenchmarkResult(
            generated_at_utc=datetime.now(UTC).isoformat(),
            status="rust_unavailable",
            rows=int(rows),
            evals=int(evals),
            python=None,
            rust=None,
            speedup_total=None,
            parity=_parity(
                None, None, reason=str(diagnostics.get("native_load_error") or "rust unavailable")
            ),
            diagnostics=diagnostics,
        )

    py_timing: BackendTiming | None = None
    rust_timing: BackendTiming | None = None
    py_result: tuple[np.ndarray, np.ndarray] | None = None
    rust_result: tuple[np.ndarray, np.ndarray] | None = None

    if backend in {"both", "python"}:
        py_timing, py_result = _time_backend(inputs, mode="python", evals=evals)
    if backend in {"both", "rust"}:
        if native_backend.native_backend_available():
            rust_timing, rust_result = _time_backend(inputs, mode="rust", evals=evals)
        elif require_rust:
            return BenchmarkResult(
                generated_at_utc=datetime.now(UTC).isoformat(),
                status="rust_unavailable",
                rows=int(rows),
                evals=int(evals),
                python=py_timing,
                rust=None,
                speedup_total=None,
                parity=_parity(
                    py_result,
                    None,
                    reason=str(diagnostics.get("native_load_error") or "rust unavailable"),
                ),
                diagnostics=diagnostics,
            )

    if backend == "rust" and rust_result is not None and py_result is None:
        py_timing, py_result = _time_backend(inputs, mode="python", evals=1)
    if (
        backend == "python"
        and py_result is not None
        and rust_result is None
        and native_backend.native_backend_available()
    ):
        rust_timing, rust_result = _time_backend(inputs, mode="rust", evals=1)

    parity = _parity(py_result, rust_result)
    speedup_total = None
    if (
        py_timing is not None
        and rust_timing is not None
        and rust_timing.total_seconds_per_eval > 0.0
    ):
        speedup_total = float(py_timing.total_seconds_per_eval / rust_timing.total_seconds_per_eval)
    return BenchmarkResult(
        generated_at_utc=datetime.now(UTC).isoformat(),
        status="pass" if parity.passed else "fail",
        rows=int(rows),
        evals=int(evals),
        python=py_timing,
        rust=rust_timing,
        speedup_total=speedup_total,
        parity=parity,
        diagnostics=native_backend.live_signal_backend_diagnostics("auto"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000, help="Synthetic live rows")
    parser.add_argument("--evals", type=int, default=20, help="Evaluations per backend")
    parser.add_argument(
        "--backend",
        choices=["both", "python", "rust"],
        default="both",
        help="Backend timing mode",
    )
    parser.add_argument("--require-rust", action="store_true", help="Fail/skip if Rust is missing")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_benchmark(
        rows=args.rows,
        evals=args.evals,
        backend=args.backend,
        require_rust=bool(args.require_rust),
    )
    payload = asdict(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_rust and result.status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
