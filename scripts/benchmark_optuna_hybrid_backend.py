#!/usr/bin/env python3
"""Benchmark Alpha Zoo Optuna hybrid portfolio backends.

The public research API stays Python.  This script proves whether the optional
Rust portfolio kernel is worth using underneath the v3.5/v3.6 Optuna runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.alpha_zoo import native_hybrid_optuna_backend as native_backend  # noqa: E402

PARITY_TOL = 1e-8


@dataclass(frozen=True)
class BackendTiming:
    backend: str
    rows: int
    columns: int
    evals: int
    elapsed_seconds: float
    seconds_per_eval: float
    rows_per_second: float


@dataclass(frozen=True)
class ParityResult:
    checked: bool
    passed: bool
    max_abs_return_diff: float
    max_abs_weight_diff: float
    reason: str | None = None


@dataclass(frozen=True)
class OptunaHybridBenchmarkResult:
    status: str
    rows: int
    columns: int
    evals: int
    version: str
    rust_available: bool
    backend_diagnostics: dict[str, Any]
    python: BackendTiming | None
    rust: BackendTiming | None
    rust_speedup_vs_python: float | None
    parity: ParityResult
    peak_memory_note: str


def _load_runner_module() -> Any:
    path = (
        REPO_ROOT
        / "scripts"
        / "research"
        / "run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py"
    )
    spec = importlib.util.spec_from_file_location("alpha_zoo_optuna_hybrid_bench_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load runner module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_synthetic_returns(*, rows: int, columns: int, seed: int = 20260527) -> np.ndarray:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if columns <= 0:
        raise ValueError("columns must be positive")
    rng = np.random.default_rng(seed)
    base = rng.normal(0.00005, 0.002, size=(rows, columns)).astype(np.float64)
    # Add mild sleeve diversity so adaptive/default/boost branches are exercised.
    for col in range(columns):
        base[:, col] += (col - (columns / 2.0)) * 0.00001
    return base


def _params(module: Any) -> tuple[Any, Any]:
    params = module.HybridParams()
    learned = module.LearnedParams(
        high_vol_threshold=0.001,
        default_idx=0,
        high_vol_best_idx=min(1, 2),
        default_weight_ratio=0.5,
        high_vol_weight_boost=0.1,
        cv_score=0.0,
    )
    return params, learned


def _set_backend_env(backend: str) -> str | None:
    previous = os.environ.get(native_backend.HYBRID_OPTUNA_BACKEND_ENV)
    os.environ[native_backend.HYBRID_OPTUNA_BACKEND_ENV] = backend
    return previous


def _restore_backend_env(previous: str | None) -> None:
    if previous is None:
        os.environ.pop(native_backend.HYBRID_OPTUNA_BACKEND_ENV, None)
    else:
        os.environ[native_backend.HYBRID_OPTUNA_BACKEND_ENV] = previous


def _time_backend(
    module: Any,
    returns: np.ndarray,
    *,
    backend: str,
    version: str,
    evals: int,
) -> tuple[BackendTiming, np.ndarray, np.ndarray]:
    params, learned = _params(module)
    previous = _set_backend_env(backend)
    try:
        portfolio, weights, _ = module._portfolio_returns_for_params(
            returns,
            params=params,
            learned=learned,
            version=version,
            start_idx=0,
            allocation_stride=24,
        )
        started = time.perf_counter()
        for _ in range(max(1, int(evals))):
            portfolio, weights, _ = module._portfolio_returns_for_params(
                returns,
                params=params,
                learned=learned,
                version=version,
                start_idx=0,
                allocation_stride=24,
            )
        elapsed = max(1e-9, time.perf_counter() - started)
    finally:
        _restore_backend_env(previous)
    evals_i = max(1, int(evals))
    return (
        BackendTiming(
            backend=backend,
            rows=int(returns.shape[0]),
            columns=int(returns.shape[1]),
            evals=evals_i,
            elapsed_seconds=float(elapsed),
            seconds_per_eval=float(elapsed / evals_i),
            rows_per_second=float((returns.shape[0] * evals_i) / elapsed),
        ),
        np.asarray(portfolio, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _check_parity(
    python_returns: np.ndarray | None,
    python_weights: np.ndarray | None,
    rust_returns: np.ndarray | None,
    rust_weights: np.ndarray | None,
) -> ParityResult:
    if (
        python_returns is None
        or python_weights is None
        or rust_returns is None
        or rust_weights is None
    ):
        return ParityResult(False, False, 0.0, 0.0, "missing backend")
    if python_returns.shape != rust_returns.shape or python_weights.shape != rust_weights.shape:
        return ParityResult(
            True,
            False,
            float("inf"),
            float("inf"),
            f"shape mismatch returns={python_returns.shape}/{rust_returns.shape} "
            f"weights={python_weights.shape}/{rust_weights.shape}",
        )
    return_diff = float(np.max(np.abs(python_returns - rust_returns)))
    weight_diff = float(np.max(np.abs(python_weights - rust_weights)))
    passed = return_diff <= PARITY_TOL and weight_diff <= PARITY_TOL
    return ParityResult(
        True,
        passed,
        return_diff,
        weight_diff,
        None if passed else f"diff>{PARITY_TOL:g}",
    )


def run_benchmark(
    *,
    rows: int,
    columns: int,
    evals: int,
    version: str,
    backend: str,
    require_rust: bool = False,
    require_speedup: bool = False,
    min_speedup: float = 1.0,
) -> OptunaHybridBenchmarkResult:
    if evals <= 0:
        raise ValueError("evals must be positive")
    if version not in {"v3_5", "v3_6"}:
        raise ValueError("version must be one of: v3_5, v3_6")
    module = _load_runner_module()
    rust_available = native_backend.native_backend_available()
    diagnostics = native_backend.hybrid_optuna_backend_diagnostics("rust")
    if require_rust and not rust_available:
        return OptunaHybridBenchmarkResult(
            status="rust_unavailable",
            rows=rows,
            columns=columns,
            evals=evals,
            version=version,
            rust_available=False,
            backend_diagnostics=diagnostics,
            python=None,
            rust=None,
            rust_speedup_vs_python=None,
            parity=ParityResult(False, False, 0.0, 0.0, diagnostics.get("native_load_error")),
            peak_memory_note="synthetic matrix benchmark; keep rows bounded for <8GB sessions",
        )

    returns = generate_synthetic_returns(rows=rows, columns=columns)
    python_timing = None
    rust_timing = None
    python_returns = None
    python_weights = None
    rust_returns = None
    rust_weights = None

    if backend in {"python", "both"}:
        python_timing, python_returns, python_weights = _time_backend(
            module, returns, backend="python", version=version, evals=evals
        )
    if backend in {"rust", "both"} and rust_available:
        rust_timing, rust_returns, rust_weights = _time_backend(
            module, returns, backend="rust", version=version, evals=evals
        )

    parity = (
        _check_parity(python_returns, python_weights, rust_returns, rust_weights)
        if backend == "both"
        else ParityResult(False, True, 0.0, 0.0, "single-backend run")
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

    return OptunaHybridBenchmarkResult(
        status=status,
        rows=rows,
        columns=columns,
        evals=evals,
        version=version,
        rust_available=rust_available,
        backend_diagnostics=diagnostics,
        python=python_timing,
        rust=rust_timing,
        rust_speedup_vs_python=speedup,
        parity=parity,
        peak_memory_note="synthetic matrix benchmark; keep rows bounded for <8GB sessions",
    )


def _jsonable(result: OptunaHybridBenchmarkResult) -> dict[str, Any]:
    payload = asdict(result)
    for key in ("max_abs_return_diff", "max_abs_weight_diff"):
        if payload["parity"].get(key) == float("inf"):
            payload["parity"][key] = "inf"
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000, help="synthetic return rows")
    parser.add_argument("--columns", type=int, default=3, help="source profile columns")
    parser.add_argument("--evals", type=int, default=10, help="timed evaluations per backend")
    parser.add_argument("--version", choices=("v3_5", "v3_6"), default="v3_5")
    parser.add_argument("--backend", choices=("python", "rust", "both"), default="both")
    parser.add_argument("--require-rust", action="store_true")
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_benchmark(
        rows=int(args.rows),
        columns=int(args.columns),
        evals=int(args.evals),
        version=str(args.version),
        backend=str(args.backend),
        require_rust=bool(args.require_rust),
        require_speedup=bool(args.require_speedup),
        min_speedup=float(args.min_speedup),
    )
    payload = _jsonable(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        _write_json(Path(args.output), payload)
    if result.status not in {"pass", "rust_unavailable"}:
        raise SystemExit(1)
    if bool(args.require_rust) and result.status == "rust_unavailable":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
