"""Optional pyo3 backend for Alpha Zoo fold-level symbol simulation.

Phase 2: pyo3 migration — lumina_quant._compute (pyo3).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

ALPHA_FOLD_BACKEND_AUTO = "auto"
ALPHA_FOLD_BACKEND_PYTHON = "python"
ALPHA_FOLD_BACKEND_RUST = "rust"
ALPHA_FOLD_BACKEND_ENV = "LQ_ALPHA_FOLD_BACKEND"
ALPHA_FOLD_DLL_ENV = "LQ_ALPHA_FOLD_DLL"  # kept for env-var backward compat (ignored)
_VALID_BACKENDS = {
    ALPHA_FOLD_BACKEND_AUTO,
    ALPHA_FOLD_BACKEND_PYTHON,
    ALPHA_FOLD_BACKEND_RUST,
}

# ── pyo3 binding ──────────────────────────────────────────────────────────────
_PYO3_FN: Any = None
_PYO3_LOAD_ERROR: str = ""

try:
    from lumina_quant._compute import simulate_symbol_fold as _pyo3_simulate_symbol_fold  # type: ignore[attr-defined]

    _PYO3_FN = _pyo3_simulate_symbol_fold
except Exception as _exc:
    _PYO3_LOAD_ERROR = str(_exc)


def normalize_alpha_fold_backend(value: str | None = None) -> str:
    token = str(value or os.getenv(ALPHA_FOLD_BACKEND_ENV, ALPHA_FOLD_BACKEND_AUTO))
    normalized = token.strip().lower() or ALPHA_FOLD_BACKEND_AUTO
    if normalized not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported alpha-fold backend: {value!r}")
    return normalized


def load_alpha_fold_native_library() -> Any | None:
    """Return pyo3 function handle if available (API preserved for callers)."""
    return _PYO3_FN


def native_backend_available() -> bool:
    return _PYO3_FN is not None


def alpha_fold_backend_diagnostics(requested: str | None = None) -> dict[str, Any]:
    mode = normalize_alpha_fold_backend(requested)
    available = native_backend_available()
    resolved = (
        ALPHA_FOLD_BACKEND_RUST
        if mode != ALPHA_FOLD_BACKEND_PYTHON and available
        else ALPHA_FOLD_BACKEND_PYTHON
    )
    return {
        "requested_backend": mode,
        "resolved_backend": resolved,
        "native_available": available,
        "native_library_path": None,
        "native_load_error": _PYO3_LOAD_ERROR or None,
    }


def _as_float64(values: Any, *, name: str, expected_len: int | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if expected_len is not None and int(array.shape[0]) != int(expected_len):
        raise ValueError(f"{name} length mismatch: {array.shape[0]} != {expected_len}")
    return array


def _simulate_symbol_arrays_python(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    signal: np.ndarray,
    *,
    integer_leverage: int,
    allocation_fraction: float,
    round_trip_cost_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    next_return = np.r_[np.diff(close) / np.maximum(close[:-1], 1e-12), 0.0]
    notional = float(integer_leverage) * float(allocation_fraction)
    transition = np.abs(np.diff(np.r_[0.0, signal]))
    costs = (round_trip_cost_bps / 10000.0) * notional * transition / 2.0
    returns = signal * notional * next_return - costs
    long_liq = (signal > 0.0) & (
        ((low / np.maximum(close, 1e-12)) - 1.0) * int(integer_leverage) <= -0.95
    )
    short_liq = (signal < 0.0) & (
        ((high / np.maximum(close, 1e-12)) - 1.0) * int(integer_leverage) >= 0.95
    )
    liquidation = long_liq | short_liq
    equity = np.cumprod(1.0 + returns)
    return returns, liquidation, equity <= 0.0


def simulate_symbol_arrays(
    close: Any,
    high: Any,
    low: Any,
    signal: Any,
    *,
    integer_leverage: int,
    allocation_fraction: float,
    round_trip_cost_bps: float,
    backend: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close_arr = _as_float64(close, name="close")
    high_arr = _as_float64(high, name="high", expected_len=int(close_arr.shape[0]))
    low_arr = _as_float64(low, name="low", expected_len=int(close_arr.shape[0]))
    signal_arr = _as_float64(signal, name="signal", expected_len=int(close_arr.shape[0]))
    mode = normalize_alpha_fold_backend(backend)

    if mode != ALPHA_FOLD_BACKEND_PYTHON and _PYO3_FN is not None:
        try:
            returns, liquidation, account_wipeout = _PYO3_FN(
                close_arr,
                high_arr,
                low_arr,
                signal_arr,
                int(integer_leverage),
                float(allocation_fraction),
                float(round_trip_cost_bps),
            )
            return returns, liquidation.astype(bool), account_wipeout.astype(bool)
        except Exception:
            if mode == ALPHA_FOLD_BACKEND_RUST:
                raise
    elif mode == ALPHA_FOLD_BACKEND_RUST and _PYO3_FN is None:
        raise RuntimeError("Rust alpha-fold backend requested but unavailable")

    return _simulate_symbol_arrays_python(
        close_arr,
        high_arr,
        low_arr,
        signal_arr,
        integer_leverage=integer_leverage,
        allocation_fraction=allocation_fraction,
        round_trip_cost_bps=round_trip_cost_bps,
    )


__all__ = [
    "ALPHA_FOLD_BACKEND_AUTO",
    "ALPHA_FOLD_BACKEND_ENV",
    "ALPHA_FOLD_BACKEND_PYTHON",
    "ALPHA_FOLD_BACKEND_RUST",
    "ALPHA_FOLD_DLL_ENV",
    "alpha_fold_backend_diagnostics",
    "load_alpha_fold_native_library",
    "native_backend_available",
    "normalize_alpha_fold_backend",
    "simulate_symbol_arrays",
]
