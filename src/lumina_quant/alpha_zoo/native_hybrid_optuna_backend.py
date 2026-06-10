"""Optional pyo3 backend for Alpha Zoo Optuna hybrid portfolio loops.

Phase 2: pyo3 migration — lumina_quant._compute (pyo3).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

HYBRID_OPTUNA_BACKEND_AUTO = "auto"
HYBRID_OPTUNA_BACKEND_PYTHON = "python"
HYBRID_OPTUNA_BACKEND_RUST = "rust"
HYBRID_OPTUNA_BACKEND_ENV = "LQ_HYBRID_OPTUNA_BACKEND"
HYBRID_OPTUNA_DLL_ENV = "LQ_HYBRID_OPTUNA_DLL"  # kept for env-var compat (ignored)
_VALID_BACKENDS = {
    HYBRID_OPTUNA_BACKEND_AUTO,
    HYBRID_OPTUNA_BACKEND_PYTHON,
    HYBRID_OPTUNA_BACKEND_RUST,
}

# ── pyo3 binding ──────────────────────────────────────────────────────────────
_PYO3_FN: Any = None
_PYO3_LOAD_ERROR: str = ""

try:
    from lumina_quant._compute import (  # type: ignore[attr-defined]
        evaluate_hybrid_optuna_portfolio as _pyo3_evaluate_hybrid,
    )

    _PYO3_FN = _pyo3_evaluate_hybrid
except Exception as _exc:
    _PYO3_LOAD_ERROR = str(_exc)


def normalize_hybrid_optuna_backend(value: str | None = None) -> str:
    token = str(value or os.getenv(HYBRID_OPTUNA_BACKEND_ENV, HYBRID_OPTUNA_BACKEND_AUTO))
    normalized = token.strip().lower() or HYBRID_OPTUNA_BACKEND_AUTO
    if normalized not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported hybrid Optuna backend: {value!r}")
    return normalized


def load_hybrid_optuna_native_library() -> Any | None:
    """Return pyo3 function handle if available (API preserved for callers)."""
    return _PYO3_FN


def native_backend_available() -> bool:
    return _PYO3_FN is not None


def hybrid_optuna_backend_diagnostics(requested: str | None = None) -> dict[str, Any]:
    mode = normalize_hybrid_optuna_backend(requested)
    available = native_backend_available()
    resolved = (
        HYBRID_OPTUNA_BACKEND_RUST
        if mode != HYBRID_OPTUNA_BACKEND_PYTHON and available
        else HYBRID_OPTUNA_BACKEND_PYTHON
    )
    return {
        "requested_backend": mode,
        "resolved_backend": resolved,
        "native_available": available,
        "native_library_path": None,
        "native_load_error": _PYO3_LOAD_ERROR or None,
    }


def evaluate_hybrid_portfolio_native(
    returns: np.ndarray,
    *,
    version: str,
    start_idx: int,
    mape_window: int,
    bias_window: int,
    short_vol_window: int,
    bias_correction_alpha: float,
    bias_combine_ratio: float,
    max_single_weight: float,
    default_idx: int,
    high_vol_best_idx: int,
    default_weight_ratio: float,
    high_vol_threshold: float,
    high_vol_weight_boost: float,
    backend: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return native portfolio, exposed weights, raw weights, and diagnostics arrays."""
    mode = normalize_hybrid_optuna_backend(backend)
    if mode == HYBRID_OPTUNA_BACKEND_PYTHON:
        return None

    if _PYO3_FN is None:
        if mode == HYBRID_OPTUNA_BACKEND_RUST:
            raise RuntimeError("Rust hybrid Optuna backend requested but unavailable")
        return None

    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None

    version_code = 36 if str(version) == "v3_6" else 35 if str(version) == "v3_5" else 0
    if version_code == 0:
        return None
    cols = int(arr.shape[1])
    default_idx_i = int(default_idx)
    if default_idx_i < 0 or default_idx_i >= cols:
        return None

    contiguous = np.ascontiguousarray(arr, dtype=np.float64)
    try:
        portfolio, exposed_weights, raw_weights, default_idx_out, high_vol_feature, exposure = (
            _PYO3_FN(
                contiguous,
                int(version_code),
                int(max(0, int(start_idx))),
                int(max(1, int(mape_window))),
                int(max(1, int(bias_window))),
                int(max(1, int(short_vol_window))),
                float(bias_correction_alpha),
                float(bias_combine_ratio),
                float(max_single_weight),
                int(default_idx_i),
                int(max(0, int(high_vol_best_idx))),
                float(default_weight_ratio),
                float(high_vol_threshold),
                float(high_vol_weight_boost),
            )
        )
        return portfolio, exposed_weights, raw_weights, default_idx_out, high_vol_feature, exposure
    except Exception:
        if mode == HYBRID_OPTUNA_BACKEND_RUST:
            raise
        return None


__all__ = [
    "HYBRID_OPTUNA_BACKEND_AUTO",
    "HYBRID_OPTUNA_BACKEND_ENV",
    "HYBRID_OPTUNA_BACKEND_PYTHON",
    "HYBRID_OPTUNA_BACKEND_RUST",
    "HYBRID_OPTUNA_DLL_ENV",
    "evaluate_hybrid_portfolio_native",
    "hybrid_optuna_backend_diagnostics",
    "load_hybrid_optuna_native_library",
    "native_backend_available",
    "normalize_hybrid_optuna_backend",
]
