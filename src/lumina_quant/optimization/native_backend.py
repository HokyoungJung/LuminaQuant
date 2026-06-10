"""Native metric backend selection with automatic fastest-path defaulting.

Phase 2: pyo3 migration — lumina_quant._compute (pyo3).
Fallback chain: pyo3 → numba → python.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import cast

import numpy as np
from lumina_quant.optimization.constants import (
    DEFAULT_ANNUAL_PERIODS,
    METRIC_FALLBACK_SHARPE,
    METRIC_FALLBACK_TRIPLE,
    NATIVE_AUTO_SELECT_ENV,
    NATIVE_BACKEND_ENV,
    NATIVE_BENCH_DEFAULT_LOOPS,
    NATIVE_BENCH_DEFAULT_TOL,
    NATIVE_BENCH_LOOPS_ENV,
    NATIVE_BENCH_MIN_ELAPSED_SECONDS,
    NATIVE_BENCH_MIN_LOOPS,
    NATIVE_BENCH_MIN_TOL,
    NATIVE_BENCH_RANDOM_SEED,
    NATIVE_BENCH_RETURNS_STD,
    NATIVE_BENCH_SAMPLE_SIZE,
    NATIVE_BENCH_STARTING_CAPITAL,
    NATIVE_MIN_SPEEDUP_ENV,
    NATIVE_MODE_AUTO,
    NATIVE_MODE_NATIVE,
    NATIVE_MODE_NUMBA,
    NATIVE_MODE_PYTHON,
    NATIVE_SELECTION_TOL_ENV,
)
from lumina_quant.optimization.fast_eval import NUMBA_AVAILABLE, evaluate_metrics_numba

# ── pyo3 binding ──────────────────────────────────────────────────────────────
_PYO3_FN: Callable[..., tuple[float, float, float]] | None = None
_PYO3_LOAD_ERROR: str = ""

try:
    from lumina_quant._compute import evaluate_metrics as _pyo3_evaluate_metrics  # type: ignore[attr-defined]

    _PYO3_FN = _pyo3_evaluate_metrics
except Exception as _exc:
    _PYO3_LOAD_ERROR = str(_exc)

# ── module state ──────────────────────────────────────────────────────────────
_BACKEND_MODE = NATIVE_MODE_NUMBA if NUMBA_AVAILABLE else NATIVE_MODE_PYTHON
NATIVE_BACKEND_NAME = _BACKEND_MODE


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return float(raw)
    except Exception:
        return float(default)


def _evaluate_pyo3(
    total_series: np.ndarray,
    annual_periods: int,
) -> tuple[float, float, float] | None:
    if _PYO3_FN is None:
        return None
    arr = np.ascontiguousarray(total_series, dtype=np.float64)
    try:
        result = _PYO3_FN(arr, int(annual_periods))
        return float(result[0]), float(result[1]), float(result[2])
    except Exception:
        return None


def _evaluate_python(total_series: np.ndarray, annual_periods: int) -> tuple[float, float, float]:
    if total_series.size < 2:
        return METRIC_FALLBACK_TRIPLE

    prev_total = total_series[:-1]
    next_total = total_series[1:]
    returns = np.divide(
        next_total - prev_total,
        np.where(prev_total == 0.0, 1.0, prev_total),
        dtype=np.float64,
    )
    if returns.size > 0 and not np.isfinite(returns[0]):
        returns[0] = 0.0

    mean_r = float(np.mean(returns)) if returns.size > 0 else 0.0
    std_r = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if std_r > 0.0:
        sharpe = mean_r / std_r * np.sqrt(float(max(1, annual_periods)))
    else:
        sharpe = METRIC_FALLBACK_SHARPE

    initial = float(total_series[0])
    final = float(total_series[-1])
    years = float(total_series.size) / float(max(1, annual_periods))
    if initial <= 0.0 or years <= 0.0:
        cagr = 0.0
    else:
        cagr = (final / initial) ** (1.0 / years) - 1.0

    peaks = np.maximum.accumulate(total_series)
    dd = np.divide(peaks - total_series, np.where(peaks == 0.0, 1.0, peaks), dtype=np.float64)
    max_dd = float(np.max(dd)) if dd.size > 0 else 0.0
    return float(sharpe), float(cagr), float(max_dd)


def _evaluate_numba_or_python(
    total_series: np.ndarray,
    annual_periods: int,
) -> tuple[float, float, float]:
    if NUMBA_AVAILABLE:
        numba_out = evaluate_metrics_numba(total_series, annual_periods)
        return float(numba_out[0]), float(numba_out[1]), float(numba_out[2])
    return _evaluate_python(total_series, annual_periods)


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
    elapsed = max(NATIVE_BENCH_MIN_ELAPSED_SECONDS, time.perf_counter() - start)
    out_tuple = cast(tuple[float, float, float], out)
    return float(loops_i) / elapsed, (
        float(out_tuple[0]),
        float(out_tuple[1]),
        float(out_tuple[2]),
    )


def _outputs_close(
    lhs: tuple[float, float, float], rhs: tuple[float, float, float], tol: float
) -> bool:
    return (
        abs(float(lhs[0]) - float(rhs[0])) <= tol
        and abs(float(lhs[1]) - float(rhs[1])) <= tol
        and abs(float(lhs[2]) - float(rhs[2])) <= tol
    )


def _select_fastest_backend() -> None:
    global _BACKEND_MODE, NATIVE_BACKEND_NAME

    fallback_name = NATIVE_MODE_NUMBA if NUMBA_AVAILABLE else NATIVE_MODE_PYTHON
    fallback_fn = _evaluate_numba_or_python
    _BACKEND_MODE = fallback_name
    NATIVE_BACKEND_NAME = fallback_name

    mode_override = str(os.getenv(NATIVE_BACKEND_ENV, NATIVE_MODE_AUTO)).strip().lower()
    if mode_override == NATIVE_MODE_PYTHON:
        _BACKEND_MODE = NATIVE_MODE_PYTHON
        NATIVE_BACKEND_NAME = NATIVE_MODE_PYTHON
        return
    if mode_override == NATIVE_MODE_NUMBA and NUMBA_AVAILABLE:
        _BACKEND_MODE = NATIVE_MODE_NUMBA
        NATIVE_BACKEND_NAME = NATIVE_MODE_NUMBA
        return

    if _PYO3_FN is None:
        return

    if mode_override == NATIVE_MODE_NATIVE:
        _BACKEND_MODE = NATIVE_MODE_NATIVE
        NATIVE_BACKEND_NAME = f"{NATIVE_MODE_NATIVE}:pyo3"
        return

    auto_select = _env_bool(NATIVE_AUTO_SELECT_ENV, True)
    min_gain = max(0.0, _env_float(NATIVE_MIN_SPEEDUP_ENV, 0.0))
    loops = max(
        NATIVE_BENCH_MIN_LOOPS, _env_int(NATIVE_BENCH_LOOPS_ENV, NATIVE_BENCH_DEFAULT_LOOPS)
    )
    tol = max(NATIVE_BENCH_MIN_TOL, _env_float(NATIVE_SELECTION_TOL_ENV, NATIVE_BENCH_DEFAULT_TOL))

    rng = np.random.default_rng(NATIVE_BENCH_RANDOM_SEED)
    returns = rng.normal(0.0, NATIVE_BENCH_RETURNS_STD, size=NATIVE_BENCH_SAMPLE_SIZE).astype(
        np.float64
    )
    series = (1.0 + returns).cumprod() * NATIVE_BENCH_STARTING_CAPITAL

    fallback_speed, fallback_out = _bench(fallback_fn, series, DEFAULT_ANNUAL_PERIODS, loops)

    def _call_pyo3(arr: np.ndarray, periods: int) -> tuple[float, float, float]:
        out = _evaluate_pyo3(arr, periods)
        return out if out is not None else METRIC_FALLBACK_TRIPLE

    pyo3_speed, pyo3_out = _bench(_call_pyo3, series, DEFAULT_ANNUAL_PERIODS, loops)
    if not _outputs_close(pyo3_out, fallback_out, tol):
        return

    if not auto_select:
        _BACKEND_MODE = NATIVE_MODE_NATIVE
        NATIVE_BACKEND_NAME = f"{NATIVE_MODE_NATIVE}:pyo3"
        return

    threshold = fallback_speed * (1.0 + min_gain)
    if pyo3_speed > threshold:
        _BACKEND_MODE = NATIVE_MODE_NATIVE
        NATIVE_BACKEND_NAME = f"{NATIVE_MODE_NATIVE}:pyo3"


_select_fastest_backend()


def evaluate_metrics_backend(
    total_series: np.ndarray,
    annual_periods: int,
) -> tuple[float, float, float]:
    """Evaluate metrics via selected fastest backend with safe fallback."""
    if _BACKEND_MODE == NATIVE_MODE_NATIVE and _PYO3_FN is not None:
        native_out = _evaluate_pyo3(total_series, annual_periods)
        if native_out is not None:
            return native_out
    return _evaluate_numba_or_python(total_series, annual_periods)


def backend_selection_details() -> dict[str, str]:
    """Return selected backend metadata for diagnostics and tests."""
    return {
        "backend": str(NATIVE_BACKEND_NAME),
        "mode": str(_BACKEND_MODE),
        "pyo3_available": str(_PYO3_FN is not None),
        "pyo3_load_error": str(_PYO3_LOAD_ERROR) if _PYO3_LOAD_ERROR else "",
    }
