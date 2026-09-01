"""Optional pyo3 backend for raw aggTrades -> 1s OHLCV aggregation.

Phase 2: pyo3 migration — lumina_quant._compute (pyo3).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import polars as pl

RAW_FIRST_BACKEND_AUTO = "auto"
RAW_FIRST_BACKEND_PYTHON = "python"
RAW_FIRST_BACKEND_RUST = "rust"
RAW_FIRST_BACKEND_ENV = "LQ_RAW_FIRST_BACKEND"
RAW_FIRST_BACKEND_DLL_ENV = "LQ_RAW_FIRST_BACKEND_DLL"  # kept for env-var compat (ignored)
_VALID_BACKENDS = {
    RAW_FIRST_BACKEND_AUTO,
    RAW_FIRST_BACKEND_PYTHON,
    RAW_FIRST_BACKEND_RUST,
}
_AUTO_FALLBACK_WARNED: set[str] = set()
_LOGGER = logging.getLogger(__name__)

# ── pyo3 binding ──────────────────────────────────────────────────────────────
_PYO3_FN: Any = None
_PYO3_LOAD_ERROR: str = ""

try:
    from lumina_quant._compute import (  # type: ignore[attr-defined]
        aggregate_raw_aggtrades_to_1s as _pyo3_aggregate,
    )

    _PYO3_FN = _pyo3_aggregate
except Exception as _exc:
    _PYO3_LOAD_ERROR = str(_exc)

if _PYO3_FN is not None:
    from lumina_quant._native_kernel_version import check_native_kernel_version

    check_native_kernel_version()


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


def normalize_raw_first_backend(value: str | None = None) -> str:
    token = str(value or os.getenv(RAW_FIRST_BACKEND_ENV, RAW_FIRST_BACKEND_AUTO)).strip().lower()
    normalized = token or RAW_FIRST_BACKEND_AUTO
    if normalized not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported raw-first backend: {value!r}")
    return normalized


def _warn_auto_fallback_once(reason: str) -> None:
    message = str(reason or "").strip()
    if not message or message in _AUTO_FALLBACK_WARNED:
        return
    _AUTO_FALLBACK_WARNED.add(message)
    _LOGGER.warning("%s", message)


def _load_native_function() -> Any | None:
    """Return pyo3 function if available. Patchable seam for tests."""
    return _PYO3_FN


def load_rawfirst_native_library() -> Any | None:
    """Return pyo3 function if available (API preserved for callers)."""
    return _PYO3_FN


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
        "native_library_path": None,
        "native_load_error": _PYO3_LOAD_ERROR or None,
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
    return f"{RAW_FIRST_BACKEND_RUST}:pyo3"


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
            + (f" ({_PYO3_LOAD_ERROR})" if _PYO3_LOAD_ERROR else "")
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

    try:
        (
            out_timestamps,
            out_open,
            out_high,
            out_low,
            out_close,
            out_volume,
        ) = fn(
            timestamps,
            prices,
            quantities,
            int(range_start_ms or 0),
            1 if range_start_ms is not None else 0,
            int(range_end_ms or 0),
            1 if range_end_ms is not None else 0,
            float(previous_close or 0.0),
            1 if previous_close is not None else 0,
            int(complete_through_ms),
        )
    except Exception as exc:
        if mode == RAW_FIRST_BACKEND_AUTO:
            _warn_auto_fallback_once(
                f"Rust raw-first backend error in auto mode; falling back to Python ({exc})"
            )
            return None
        raise RuntimeError(f"Rust raw-first backend failed: {exc}") from exc

    row_count = int(out_timestamps.shape[0])
    if row_count <= 0:
        return _empty_ohlcv_frame()

    return pl.DataFrame(
        {
            "datetime": pl.Series("datetime", out_timestamps, dtype=pl.Int64).cast(
                pl.Datetime(time_unit="ms")
            ),
            "open": pl.Series("open", out_open, dtype=pl.Float64),
            "high": pl.Series("high", out_high, dtype=pl.Float64),
            "low": pl.Series("low", out_low, dtype=pl.Float64),
            "close": pl.Series("close", out_close, dtype=pl.Float64),
            "volume": pl.Series("volume", out_volume, dtype=pl.Float64),
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
