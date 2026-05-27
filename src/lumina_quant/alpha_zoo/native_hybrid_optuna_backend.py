"""Optional Rust backend for Alpha Zoo Optuna hybrid portfolio loops."""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

HYBRID_OPTUNA_BACKEND_AUTO = "auto"
HYBRID_OPTUNA_BACKEND_PYTHON = "python"
HYBRID_OPTUNA_BACKEND_RUST = "rust"
HYBRID_OPTUNA_BACKEND_ENV = "LQ_HYBRID_OPTUNA_BACKEND"
HYBRID_OPTUNA_DLL_ENV = "LQ_HYBRID_OPTUNA_DLL"
_VALID_BACKENDS = {
    HYBRID_OPTUNA_BACKEND_AUTO,
    HYBRID_OPTUNA_BACKEND_PYTHON,
    HYBRID_OPTUNA_BACKEND_RUST,
}

_NATIVE_FN: Any = None
_NATIVE_HANDLE: Any = None
_NATIVE_DLL = ""
_NATIVE_LOAD_ERROR = ""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _native_lib_filename(stem: str) -> str:
    system_name = platform.system().lower()
    if system_name == "windows":
        return f"{stem}.dll"
    if system_name == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def normalize_hybrid_optuna_backend(value: str | None = None) -> str:
    token = str(value or os.getenv(HYBRID_OPTUNA_BACKEND_ENV, HYBRID_OPTUNA_BACKEND_AUTO))
    normalized = token.strip().lower() or HYBRID_OPTUNA_BACKEND_AUTO
    if normalized not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported hybrid Optuna backend: {value!r}")
    return normalized


def _discover_dll_candidates() -> list[str]:
    explicit = str(os.getenv(HYBRID_OPTUNA_DLL_ENV, "")).strip()
    if explicit:
        return [explicit]
    return [
        str(
            _project_root()
            / "native"
            / "rust_hybrid_optuna"
            / "target"
            / "release"
            / _native_lib_filename("lumina_hybrid_optuna")
        )
    ]


def load_hybrid_optuna_native_library() -> Any | None:
    global _NATIVE_DLL, _NATIVE_HANDLE, _NATIVE_LOAD_ERROR
    if _NATIVE_HANDLE is not None:
        return _NATIVE_HANDLE
    last_error = ""
    for dll_path in _discover_dll_candidates():
        if not dll_path or not os.path.exists(dll_path):
            last_error = (
                f"native library missing at {dll_path}"
                if dll_path
                else "native library path missing"
            )
            continue
        try:
            handle = ctypes.CDLL(dll_path)
        except Exception as exc:
            last_error = f"failed to load native library {dll_path}: {exc}"
            continue
        _NATIVE_HANDLE = handle
        _NATIVE_DLL = dll_path
        _NATIVE_LOAD_ERROR = ""
        return handle
    _NATIVE_LOAD_ERROR = last_error or "native library unavailable"
    return None


def _load_native_function() -> Any | None:
    global _NATIVE_FN
    if _NATIVE_FN is not None:
        return _NATIVE_FN
    handle = load_hybrid_optuna_native_library()
    if handle is None:
        return None
    try:
        fn = handle.evaluate_hybrid_optuna_portfolio
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        fn.restype = ctypes.c_int32
    except Exception as exc:
        _NATIVE_LOAD_ERROR = f"failed to bind native hybrid function: {exc}"
        return None
    _NATIVE_FN = fn
    return fn


def native_backend_available() -> bool:
    return _load_native_function() is not None


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
        "native_library_path": _NATIVE_DLL or None,
        "native_load_error": _NATIVE_LOAD_ERROR or None,
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
    """Return native portfolio, exposed weights, raw weights, and diagnostics arrays.

    The Rust path is deliberately conservative: it only accepts finite 2-D float64
    matrices.  Any unsupported input falls back to the Python implementation in
    auto mode so research/live reproducibility is preserved.
    """
    mode = normalize_hybrid_optuna_backend(backend)
    if mode == HYBRID_OPTUNA_BACKEND_PYTHON:
        return None

    fn = _load_native_function()
    if fn is None:
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
    rows = int(contiguous.shape[0])
    exposed_weights = np.zeros((rows, cols), dtype=np.float64)
    raw_weights = np.zeros((rows, cols), dtype=np.float64)
    portfolio = np.zeros(rows, dtype=np.float64)
    default_idx_out = np.full(rows, -1, dtype=np.int64)
    high_vol_feature = np.zeros(rows, dtype=np.float64)
    exposure = np.zeros(rows, dtype=np.float64)

    status = int(
        fn(
            contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(rows),
            ctypes.c_size_t(cols),
            ctypes.c_int32(version_code),
            ctypes.c_size_t(max(0, int(start_idx))),
            ctypes.c_size_t(max(1, int(mape_window))),
            ctypes.c_size_t(max(1, int(bias_window))),
            ctypes.c_size_t(max(1, int(short_vol_window))),
            ctypes.c_double(float(bias_correction_alpha)),
            ctypes.c_double(float(bias_combine_ratio)),
            ctypes.c_double(float(max_single_weight)),
            ctypes.c_size_t(default_idx_i),
            ctypes.c_size_t(max(0, int(high_vol_best_idx))),
            ctypes.c_double(float(default_weight_ratio)),
            ctypes.c_double(float(high_vol_threshold)),
            ctypes.c_double(float(high_vol_weight_boost)),
            portfolio.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            exposed_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            raw_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            default_idx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            high_vol_feature.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            exposure.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if status != 0:
        if mode == HYBRID_OPTUNA_BACKEND_RUST:
            raise RuntimeError(f"Rust hybrid Optuna backend failed with status {status}")
        return None
    return portfolio, exposed_weights, raw_weights, default_idx_out, high_vol_feature, exposure


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
