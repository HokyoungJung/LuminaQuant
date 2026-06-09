"""Optional Rust backend for Alpha Zoo fold-level symbol simulation.

The clean-new-alpha discovery runner evaluates many candidate signals against
identical OHLCV arrays.  This module keeps the public semantics identical to the
existing Python/NumPy simulator while allowing the pure array loop to run in a
small Rust cdylib when available.
"""

from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

ALPHA_FOLD_BACKEND_AUTO = "auto"
ALPHA_FOLD_BACKEND_PYTHON = "python"
ALPHA_FOLD_BACKEND_RUST = "rust"
ALPHA_FOLD_BACKEND_ENV = "LQ_ALPHA_FOLD_BACKEND"
ALPHA_FOLD_DLL_ENV = "LQ_ALPHA_FOLD_DLL"
_VALID_BACKENDS = {
    ALPHA_FOLD_BACKEND_AUTO,
    ALPHA_FOLD_BACKEND_PYTHON,
    ALPHA_FOLD_BACKEND_RUST,
}

_NATIVE_HANDLE: Any = None
_NATIVE_FN: Any = None
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


def normalize_alpha_fold_backend(value: str | None = None) -> str:
    token = str(value or os.getenv(ALPHA_FOLD_BACKEND_ENV, ALPHA_FOLD_BACKEND_AUTO))
    normalized = token.strip().lower() or ALPHA_FOLD_BACKEND_AUTO
    if normalized not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported alpha-fold backend: {value!r}")
    return normalized


def _discover_dll_candidates() -> list[str]:
    explicit = str(os.getenv(ALPHA_FOLD_DLL_ENV, "")).strip()
    if explicit:
        return [explicit]
    return [
        str(
            _project_root()
            / "native"
            / "rust_alpha_fold"
            / "target"
            / "release"
            / _native_lib_filename("lumina_alpha_fold")
        )
    ]


def load_alpha_fold_native_library() -> Any | None:
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
        except Exception as exc:  # pragma: no cover - platform loader detail
            last_error = f"failed to load native library {dll_path}: {exc}"
            continue
        _NATIVE_HANDLE = handle
        _NATIVE_DLL = dll_path
        _NATIVE_LOAD_ERROR = ""
        return handle
    _NATIVE_LOAD_ERROR = last_error or "native library unavailable"
    return None


def _load_native_function() -> Any | None:
    global _NATIVE_FN, _NATIVE_LOAD_ERROR
    if _NATIVE_FN is not None:
        return _NATIVE_FN
    handle = load_alpha_fold_native_library()
    if handle is None:
        return None
    try:
        fn = handle.simulate_symbol_fold_native
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        fn.restype = ctypes.c_int32
    except Exception as exc:  # pragma: no cover - corrupt/wrong library
        _NATIVE_LOAD_ERROR = f"failed to bind native alpha-fold function: {exc}"
        return None
    _NATIVE_FN = fn
    return fn


def native_backend_available() -> bool:
    return _load_native_function() is not None


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
        "native_library_path": _NATIVE_DLL or None,
        "native_load_error": _NATIVE_LOAD_ERROR or None,
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

    if mode != ALPHA_FOLD_BACKEND_PYTHON:
        fn = _load_native_function()
        if fn is None:
            if mode == ALPHA_FOLD_BACKEND_RUST:
                raise RuntimeError("Rust alpha-fold backend requested but unavailable")
        else:
            returns = np.zeros(int(close_arr.shape[0]), dtype=np.float64)
            liquidation = np.zeros(int(close_arr.shape[0]), dtype=np.uint8)
            account_wipeout = np.zeros(int(close_arr.shape[0]), dtype=np.uint8)
            status = int(
                fn(
                    close_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    high_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    low_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    signal_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_size_t(int(close_arr.shape[0])),
                    ctypes.c_int32(int(integer_leverage)),
                    ctypes.c_double(float(allocation_fraction)),
                    ctypes.c_double(float(round_trip_cost_bps)),
                    returns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    liquidation.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    account_wipeout.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                )
            )
            if status == 0:
                return returns, liquidation.astype(bool), account_wipeout.astype(bool)
            if mode == ALPHA_FOLD_BACKEND_RUST:
                raise RuntimeError(f"Rust alpha-fold backend failed with status {status}")

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
