"""Optional Rust backend for live Alpha Zoo state-signal kernels.

The public API remains Python.  Rust is an internal acceleration path for
pure deterministic state machines used by the paper/testnet live strategy.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

LIVE_SIGNAL_BACKEND_AUTO = "auto"
LIVE_SIGNAL_BACKEND_PYTHON = "python"
LIVE_SIGNAL_BACKEND_RUST = "rust"
LIVE_SIGNAL_BACKEND_ENV = "LQ_LIVE_SIGNAL_BACKEND"
LIVE_SIGNAL_BACKEND_DLL_ENV = "LQ_LIVE_SIGNAL_BACKEND_DLL"
_VALID_BACKENDS = {
    LIVE_SIGNAL_BACKEND_AUTO,
    LIVE_SIGNAL_BACKEND_PYTHON,
    LIVE_SIGNAL_BACKEND_RUST,
}
_SIDE_MODES = {
    "long_only": 0,
    "short_only": 1,
    "long_short": 2,
}
_SIDE_MODE_NONE = 3

_NATIVE_HANDLE: Any = None
_DEBOUNCED_FN: Any = None
_TRAILING_FN: Any = None
_NATIVE_DLL = ""
_NATIVE_LOAD_ERROR = ""
_AUTO_FALLBACK_WARNED: set[str] = set()
_LOGGER = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _native_lib_filename(stem: str) -> str:
    system_name = platform.system().lower()
    if system_name == "windows":
        return f"{stem}.dll"
    if system_name == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def normalize_live_signal_backend(value: str | None = None) -> str:
    token = (
        str(value or os.getenv(LIVE_SIGNAL_BACKEND_ENV, LIVE_SIGNAL_BACKEND_AUTO)).strip().lower()
    )
    normalized = token or LIVE_SIGNAL_BACKEND_AUTO
    if normalized not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported live-signal backend: {value!r}")
    return normalized


def _discover_dll_candidates() -> list[str]:
    explicit = str(os.getenv(LIVE_SIGNAL_BACKEND_DLL_ENV, "")).strip()
    if explicit:
        return [explicit]
    root = _project_root()
    return [
        str(
            root
            / "native"
            / "rust_live_signals"
            / "target"
            / "release"
            / _native_lib_filename("lumina_live_signals")
        )
    ]


def _warn_auto_fallback_once(reason: str) -> None:
    message = str(reason or "").strip()
    if not message or message in _AUTO_FALLBACK_WARNED:
        return
    _AUTO_FALLBACK_WARNED.add(message)
    _LOGGER.warning("%s", message)


def load_live_signal_native_library() -> Any | None:
    global _NATIVE_HANDLE, _NATIVE_DLL, _NATIVE_LOAD_ERROR
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


def _load_debounced_function() -> Any | None:
    global _DEBOUNCED_FN
    if _DEBOUNCED_FN is not None:
        return _DEBOUNCED_FN
    handle = load_live_signal_native_library()
    if handle is None:
        return None
    try:
        fn = handle.debounced_state_signal_native
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
        ]
        fn.restype = ctypes.c_int32
    except Exception:  # pragma: no cover - corrupt/wrong library
        return None
    _DEBOUNCED_FN = fn
    return fn


def _load_trailing_function() -> Any | None:
    global _TRAILING_FN
    if _TRAILING_FN is not None:
        return _TRAILING_FN
    handle = load_live_signal_native_library()
    if handle is None:
        return None
    try:
        fn = handle.trailing_state_signal_native
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        fn.restype = ctypes.c_int32
    except Exception:  # pragma: no cover - corrupt/wrong library
        return None
    _TRAILING_FN = fn
    return fn


def native_backend_available() -> bool:
    return _load_debounced_function() is not None and _load_trailing_function() is not None


def live_signal_backend_diagnostics(requested: str | None = None) -> dict[str, Any]:
    mode = normalize_live_signal_backend(requested)
    description = describe_live_signal_backend(mode)
    resolved_backend = LIVE_SIGNAL_BACKEND_PYTHON
    if description.startswith(f"{LIVE_SIGNAL_BACKEND_RUST}:"):
        resolved_backend = LIVE_SIGNAL_BACKEND_RUST
    return {
        "requested_backend": mode,
        "resolved_backend": resolved_backend,
        "description": description,
        "native_available": resolved_backend == LIVE_SIGNAL_BACKEND_RUST,
        "native_library_path": _NATIVE_DLL or None,
        "native_load_error": _NATIVE_LOAD_ERROR or None,
        "auto_fallback_warning_count": len(_AUTO_FALLBACK_WARNED),
        "auto_fallback_warning_reasons": sorted(_AUTO_FALLBACK_WARNED),
    }


def describe_live_signal_backend(requested: str | None = None) -> str:
    mode = normalize_live_signal_backend(requested)
    if mode == LIVE_SIGNAL_BACKEND_PYTHON:
        return LIVE_SIGNAL_BACKEND_PYTHON
    if not native_backend_available():
        return (
            LIVE_SIGNAL_BACKEND_PYTHON
            if mode == LIVE_SIGNAL_BACKEND_AUTO
            else f"{LIVE_SIGNAL_BACKEND_RUST}:unavailable"
        )
    return f"{LIVE_SIGNAL_BACKEND_RUST}:{_NATIVE_DLL}"


def _side_mode(side: str) -> int:
    return _SIDE_MODES.get(str(side or "").strip().lower(), _SIDE_MODE_NONE)


def _as_bool_u8(values: Any, *, name: str, expected_len: int | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.bool_), dtype=np.uint8)
    if expected_len is not None and int(array.shape[0]) != int(expected_len):
        raise ValueError(f"{name} length mismatch: {array.shape[0]} != {expected_len}")
    return array


def _as_float64(values: Any, *, name: str, expected_len: int | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64), dtype=np.float64)
    if expected_len is not None and int(array.shape[0]) != int(expected_len):
        raise ValueError(f"{name} length mismatch: {array.shape[0]} != {expected_len}")
    return array


def _ensure_len_fits_int32(length: int) -> None:
    if int(length) > np.iinfo(np.int32).max:
        raise ValueError(f"live-signal array too large for native backend: {length}")


def evaluate_debounced_state_native(
    long_entry: Any,
    long_exit: Any,
    short_entry: Any,
    short_exit: Any,
    *,
    side: str,
    min_hold_bars: int,
    cooldown_bars: int,
    backend: str | None = None,
) -> np.ndarray | None:
    mode = normalize_live_signal_backend(backend)
    if mode == LIVE_SIGNAL_BACKEND_PYTHON:
        return None

    fn = _load_debounced_function()
    if fn is None:
        if mode == LIVE_SIGNAL_BACKEND_RUST:
            raise RuntimeError(
                "Rust live-signal backend requested but native library is unavailable"
            )
        _warn_auto_fallback_once(
            "Rust live-signal backend unavailable in auto mode; falling back to Python"
            + (f" ({_NATIVE_LOAD_ERROR})" if _NATIVE_LOAD_ERROR else "")
        )
        return None

    long_entry_values = _as_bool_u8(long_entry, name="long_entry")
    length = int(long_entry_values.shape[0])
    _ensure_len_fits_int32(length)
    long_exit_values = _as_bool_u8(long_exit, name="long_exit", expected_len=length)
    short_entry_values = _as_bool_u8(short_entry, name="short_entry", expected_len=length)
    short_exit_values = _as_bool_u8(short_exit, name="short_exit", expected_len=length)
    out = np.zeros(length, dtype=np.float64)
    if length == 0:
        return out

    status = int(
        fn(
            long_entry_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            long_exit_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            short_entry_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            short_exit_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(length),
            int(_side_mode(side)),
            int(min_hold_bars),
            int(cooldown_bars),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if status != 0:
        if mode == LIVE_SIGNAL_BACKEND_AUTO:
            _warn_auto_fallback_once(
                f"Rust live-signal debounced kernel returned status={status}; falling back to Python"
            )
            return None
        raise RuntimeError(f"Rust live-signal debounced kernel failed with status={status}")
    return out


def evaluate_trailing_state_native(
    close: Any,
    long_entry: Any,
    short_entry: Any,
    long_exit: Any,
    short_exit: Any,
    atr: Any,
    *,
    side: str,
    min_hold_bars: int,
    cooldown_bars: int,
    trail_atr_mult: float,
    backend: str | None = None,
) -> np.ndarray | None:
    mode = normalize_live_signal_backend(backend)
    if mode == LIVE_SIGNAL_BACKEND_PYTHON:
        return None

    fn = _load_trailing_function()
    if fn is None:
        if mode == LIVE_SIGNAL_BACKEND_RUST:
            raise RuntimeError(
                "Rust live-signal backend requested but native library is unavailable"
            )
        _warn_auto_fallback_once(
            "Rust live-signal backend unavailable in auto mode; falling back to Python"
            + (f" ({_NATIVE_LOAD_ERROR})" if _NATIVE_LOAD_ERROR else "")
        )
        return None

    close_values = _as_float64(close, name="close")
    length = int(close_values.shape[0])
    _ensure_len_fits_int32(length)
    long_entry_values = _as_bool_u8(long_entry, name="long_entry", expected_len=length)
    short_entry_values = _as_bool_u8(short_entry, name="short_entry", expected_len=length)
    long_exit_values = _as_bool_u8(long_exit, name="long_exit", expected_len=length)
    short_exit_values = _as_bool_u8(short_exit, name="short_exit", expected_len=length)
    atr_values = _as_float64(atr, name="atr", expected_len=length)
    out = np.zeros(length, dtype=np.float64)
    if length == 0:
        return out

    status = int(
        fn(
            close_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            long_entry_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            short_entry_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            long_exit_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            short_exit_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            atr_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(length),
            int(_side_mode(side)),
            int(min_hold_bars),
            int(cooldown_bars),
            float(trail_atr_mult),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )
    if status != 0:
        if mode == LIVE_SIGNAL_BACKEND_AUTO:
            _warn_auto_fallback_once(
                f"Rust live-signal trailing kernel returned status={status}; falling back to Python"
            )
            return None
        raise RuntimeError(f"Rust live-signal trailing kernel failed with status={status}")
    return out


__all__ = [
    "LIVE_SIGNAL_BACKEND_AUTO",
    "LIVE_SIGNAL_BACKEND_DLL_ENV",
    "LIVE_SIGNAL_BACKEND_ENV",
    "LIVE_SIGNAL_BACKEND_PYTHON",
    "LIVE_SIGNAL_BACKEND_RUST",
    "describe_live_signal_backend",
    "evaluate_debounced_state_native",
    "evaluate_trailing_state_native",
    "live_signal_backend_diagnostics",
    "load_live_signal_native_library",
    "native_backend_available",
    "normalize_live_signal_backend",
]
