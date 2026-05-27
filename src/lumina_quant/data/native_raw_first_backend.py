"""Optional native backend for raw aggTrades -> 1s OHLCV aggregation."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

RAW_FIRST_BACKEND_AUTO = "auto"
RAW_FIRST_BACKEND_PYTHON = "python"
RAW_FIRST_BACKEND_RUST = "rust"
RAW_FIRST_BACKEND_ENV = "LQ_RAW_FIRST_BACKEND"
RAW_FIRST_BACKEND_DLL_ENV = "LQ_RAW_FIRST_BACKEND_DLL"
_VALID_BACKENDS = {
    RAW_FIRST_BACKEND_AUTO,
    RAW_FIRST_BACKEND_PYTHON,
    RAW_FIRST_BACKEND_RUST,
}
_NATIVE_FN: Any = None
_NATIVE_HANDLE: Any = None
_NATIVE_DLL = ""
_NATIVE_LOAD_ERROR = ""
_AUTO_FALLBACK_WARNED: set[str] = set()
_LOGGER = logging.getLogger(__name__)


def _empty_ohlcv_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "datetime": pl.Datetime(time_unit="ms"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _native_lib_filename(stem: str) -> str:
    system_name = platform.system().lower()
    if system_name == "windows":
        return f"{stem}.dll"
    if system_name == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def normalize_raw_first_backend(value: str | None = None) -> str:
    token = str(value or os.getenv(RAW_FIRST_BACKEND_ENV, RAW_FIRST_BACKEND_AUTO)).strip().lower()
    normalized = token or RAW_FIRST_BACKEND_AUTO
    if normalized not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported raw-first backend: {value!r}")
    return normalized


def _discover_dll_candidates() -> list[str]:
    explicit = str(os.getenv(RAW_FIRST_BACKEND_DLL_ENV, "")).strip()
    if explicit:
        return [explicit]
    root = _project_root()
    return [
        str(
            root
            / "native"
            / "rust_rawfirst"
            / "target"
            / "release"
            / _native_lib_filename("lumina_rawfirst")
        )
    ]


def _warn_auto_fallback_once(reason: str) -> None:
    message = str(reason or "").strip()
    if not message or message in _AUTO_FALLBACK_WARNED:
        return
    _AUTO_FALLBACK_WARNED.add(message)
    _LOGGER.warning("%s", message)


def load_rawfirst_native_library() -> Any | None:
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
    handle = load_rawfirst_native_library()
    if handle is None:
        return None
    try:
        fn = handle.aggregate_raw_aggtrades_to_1s
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.c_longlong,
            ctypes.c_int32,
            ctypes.c_longlong,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.c_int32,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        fn.restype = ctypes.c_int32
    except Exception:
        return None
    _NATIVE_FN = fn
    return fn


def native_backend_available() -> bool:
    return _load_native_function() is not None


def raw_first_backend_diagnostics(requested: str | None = None) -> dict[str, Any]:
    mode = normalize_raw_first_backend(requested)
    description = describe_raw_first_backend(mode)
    resolved_backend = RAW_FIRST_BACKEND_PYTHON
    if description.startswith(f"{RAW_FIRST_BACKEND_RUST}:"):
        resolved_backend = RAW_FIRST_BACKEND_RUST
    return {
        "requested_backend": mode,
        "resolved_backend": resolved_backend,
        "description": description,
        "native_library_path": _NATIVE_DLL or None,
        "native_load_error": _NATIVE_LOAD_ERROR or None,
        "auto_fallback_warning_count": len(_AUTO_FALLBACK_WARNED),
        "auto_fallback_warning_reasons": sorted(_AUTO_FALLBACK_WARNED),
    }


def describe_raw_first_backend(requested: str | None = None) -> str:
    mode = normalize_raw_first_backend(requested)
    if mode == RAW_FIRST_BACKEND_PYTHON:
        return RAW_FIRST_BACKEND_PYTHON
    fn = _load_native_function()
    if fn is None:
        return (
            RAW_FIRST_BACKEND_PYTHON
            if mode == RAW_FIRST_BACKEND_AUTO
            else f"{RAW_FIRST_BACKEND_RUST}:unavailable"
        )
    return f"{RAW_FIRST_BACKEND_RUST}:{_NATIVE_DLL}"


def aggregate_raw_aggtrades_to_1s_native(
    raw: pl.DataFrame,
    *,
    range_start_ms: int | None,
    range_end_ms: int | None,
    previous_close: float | None,
    complete_through_ms: int,
    backend: str | None = None,
) -> pl.DataFrame | None:
    mode = normalize_raw_first_backend(backend)
    if mode == RAW_FIRST_BACKEND_PYTHON:
        return None

    fn = _load_native_function()
    if fn is None:
        if mode == RAW_FIRST_BACKEND_RUST:
            raise RuntimeError("Rust raw-first backend requested but native library is unavailable")
        _warn_auto_fallback_once(
            "Rust raw-first backend unavailable in auto mode; falling back to Python"
            + (f" ({_NATIVE_LOAD_ERROR})" if _NATIVE_LOAD_ERROR else "")
        )
        return None

    if raw.is_empty():
        return _empty_ohlcv_frame()

    ts_series = raw.get_column("timestamp_ms").cast(pl.Int64)
    price_series = raw.get_column("price").cast(pl.Float64)
    quantity_series = raw.get_column("quantity").cast(pl.Float64)
    timestamps = np.ascontiguousarray(ts_series.to_numpy(), dtype=np.int64)
    prices = np.ascontiguousarray(price_series.to_numpy(), dtype=np.float64)
    quantities = np.ascontiguousarray(quantity_series.to_numpy(), dtype=np.float64)

    first_trade_second = (int(timestamps[0]) // 1000) * 1000
    last_complete_second = (((int(complete_through_ms) + 1) // 1000) * 1000) - 1000
    start_candidate = (
        (int(range_start_ms) // 1000) * 1000
        if range_start_ms is not None
        else int(first_trade_second)
    )
    end_candidate = int(last_complete_second)
    if range_end_ms is not None:
        end_candidate = min(int(end_candidate), (int(range_end_ms) // 1000) * 1000)
    if end_candidate < start_candidate:
        return _empty_ohlcv_frame()

    output_capacity = max(0, ((int(end_candidate) - int(start_candidate)) // 1000) + 1)
    if output_capacity <= 0:
        return _empty_ohlcv_frame()

    out_timestamps = np.empty(output_capacity, dtype=np.int64)
    out_open = np.empty(output_capacity, dtype=np.float64)
    out_high = np.empty(output_capacity, dtype=np.float64)
    out_low = np.empty(output_capacity, dtype=np.float64)
    out_close = np.empty(output_capacity, dtype=np.float64)
    out_volume = np.empty(output_capacity, dtype=np.float64)
    out_len = ctypes.c_int32(0)

    status = int(
        fn(
            timestamps.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            quantities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(timestamps.shape[0]),
            int(range_start_ms or 0),
            1 if range_start_ms is not None else 0,
            int(range_end_ms or 0),
            1 if range_end_ms is not None else 0,
            float(previous_close or 0.0),
            1 if previous_close is not None else 0,
            int(complete_through_ms),
            out_timestamps.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            out_open.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_high.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_low.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_close.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_volume.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(output_capacity),
            ctypes.byref(out_len),
        )
    )
    if status != 0:
        if mode == RAW_FIRST_BACKEND_AUTO:
            _warn_auto_fallback_once(
                f"Rust raw-first backend returned status={status} in auto mode; falling back to Python"
            )
            return None
        raise RuntimeError(f"Rust raw-first backend failed with status={status}")

    row_count = int(out_len.value)
    if row_count <= 0:
        return _empty_ohlcv_frame()

    return pl.DataFrame(
        {
            "datetime": pl.Series("datetime", out_timestamps[:row_count], dtype=pl.Int64).cast(
                pl.Datetime(time_unit="ms")
            ),
            "open": pl.Series("open", out_open[:row_count], dtype=pl.Float64),
            "high": pl.Series("high", out_high[:row_count], dtype=pl.Float64),
            "low": pl.Series("low", out_low[:row_count], dtype=pl.Float64),
            "close": pl.Series("close", out_close[:row_count], dtype=pl.Float64),
            "volume": pl.Series("volume", out_volume[:row_count], dtype=pl.Float64),
        }
    )


__all__ = [
    "RAW_FIRST_BACKEND_AUTO",
    "RAW_FIRST_BACKEND_ENV",
    "RAW_FIRST_BACKEND_PYTHON",
    "RAW_FIRST_BACKEND_RUST",
    "aggregate_raw_aggtrades_to_1s_native",
    "describe_raw_first_backend",
    "load_rawfirst_native_library",
    "native_backend_available",
    "normalize_raw_first_backend",
    "raw_first_backend_diagnostics",
]
