"""Optional pyo3 backend for live Alpha Zoo state-signal kernels.

Phase 2: pyo3 migration — lumina_quant._compute (pyo3).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

LIVE_SIGNAL_BACKEND_AUTO = "auto"
LIVE_SIGNAL_BACKEND_PYTHON = "python"
LIVE_SIGNAL_BACKEND_RUST = "rust"
LIVE_SIGNAL_BACKEND_ENV = "LQ_LIVE_SIGNAL_BACKEND"
LIVE_SIGNAL_BACKEND_DLL_ENV = "LQ_LIVE_SIGNAL_BACKEND_DLL"  # kept for env-var compat
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

_AUTO_FALLBACK_WARNED: set[str] = set()
_LOGGER = logging.getLogger(__name__)

# ── pyo3 bindings ─────────────────────────────────────────────────────────────
_PYO3_DEBOUNCED_FN: Any = None
_PYO3_TRAILING_FN: Any = None
_PYO3_LOAD_ERROR: str = ""

try:
    from lumina_quant._compute import (  # type: ignore[attr-defined]
        debounced_state_signal as _pyo3_debounced,
        trailing_state_signal as _pyo3_trailing,
    )

    _PYO3_DEBOUNCED_FN = _pyo3_debounced
    _PYO3_TRAILING_FN = _pyo3_trailing
except Exception as _exc:
    _PYO3_LOAD_ERROR = str(_exc)


def normalize_live_signal_backend(value: str | None = None) -> str:
    token = (
        str(value or os.getenv(LIVE_SIGNAL_BACKEND_ENV, LIVE_SIGNAL_BACKEND_AUTO)).strip().lower()
    )
    normalized = token or LIVE_SIGNAL_BACKEND_AUTO
    if normalized not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported live-signal backend: {value!r}")
    return normalized


def _warn_auto_fallback_once(reason: str) -> None:
    message = str(reason or "").strip()
    if not message or message in _AUTO_FALLBACK_WARNED:
        return
    _AUTO_FALLBACK_WARNED.add(message)
    _LOGGER.warning("%s", message)


def load_live_signal_native_library() -> Any | None:
    """Return pyo3 debounced function if available (API preserved for callers)."""
    return _PYO3_DEBOUNCED_FN


def native_backend_available() -> bool:
    return _PYO3_DEBOUNCED_FN is not None and _PYO3_TRAILING_FN is not None


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
        "native_library_path": None,
        "native_load_error": _PYO3_LOAD_ERROR or None,
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
    return f"{LIVE_SIGNAL_BACKEND_RUST}:pyo3"


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

    if _PYO3_DEBOUNCED_FN is None:
        if mode == LIVE_SIGNAL_BACKEND_RUST:
            raise RuntimeError(
                "Rust live-signal backend requested but native library is unavailable"
            )
        _warn_auto_fallback_once(
            "Rust live-signal backend unavailable in auto mode; falling back to Python"
            + (f" ({_PYO3_LOAD_ERROR})" if _PYO3_LOAD_ERROR else "")
        )
        return None

    long_entry_values = _as_bool_u8(long_entry, name="long_entry")
    length = int(long_entry_values.shape[0])
    _ensure_len_fits_int32(length)
    long_exit_values = _as_bool_u8(long_exit, name="long_exit", expected_len=length)
    short_entry_values = _as_bool_u8(short_entry, name="short_entry", expected_len=length)
    short_exit_values = _as_bool_u8(short_exit, name="short_exit", expected_len=length)
    if length == 0:
        return np.zeros(0, dtype=np.float64)

    try:
        return _PYO3_DEBOUNCED_FN(
            long_entry_values,
            long_exit_values,
            short_entry_values,
            short_exit_values,
            int(length),
            int(_side_mode(side)),
            int(min_hold_bars),
            int(cooldown_bars),
        )
    except Exception as exc:
        if mode == LIVE_SIGNAL_BACKEND_AUTO:
            _warn_auto_fallback_once(
                f"Rust live-signal debounced kernel error; falling back to Python ({exc})"
            )
            return None
        raise RuntimeError(f"Rust live-signal debounced kernel failed: {exc}") from exc


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

    if _PYO3_TRAILING_FN is None:
        if mode == LIVE_SIGNAL_BACKEND_RUST:
            raise RuntimeError(
                "Rust live-signal backend requested but native library is unavailable"
            )
        _warn_auto_fallback_once(
            "Rust live-signal backend unavailable in auto mode; falling back to Python"
            + (f" ({_PYO3_LOAD_ERROR})" if _PYO3_LOAD_ERROR else "")
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
    if length == 0:
        return np.zeros(0, dtype=np.float64)

    try:
        return _PYO3_TRAILING_FN(
            close_values,
            long_entry_values,
            short_entry_values,
            long_exit_values,
            short_exit_values,
            atr_values,
            int(length),
            int(_side_mode(side)),
            int(min_hold_bars),
            int(cooldown_bars),
            float(trail_atr_mult),
        )
    except Exception as exc:
        if mode == LIVE_SIGNAL_BACKEND_AUTO:
            _warn_auto_fallback_once(
                f"Rust live-signal trailing kernel error; falling back to Python ({exc})"
            )
            return None
        raise RuntimeError(f"Rust live-signal trailing kernel failed: {exc}") from exc


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
