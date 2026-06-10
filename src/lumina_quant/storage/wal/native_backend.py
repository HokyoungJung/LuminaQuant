"""Optional pyo3 helpers for WAL append acceleration.

Phase 2: pyo3 migration — lumina_quant._compute (pyo3).
The import of load_rawfirst_native_library is removed; this module calls
the pyo3 append_ohlcv_1s_wal binding directly.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl

# ── pyo3 binding ──────────────────────────────────────────────────────────────
_PYO3_WAL_FN: Any = None
_PYO3_LOAD_ERROR: str = ""

try:
    from lumina_quant._compute import (  # type: ignore[attr-defined]
        append_ohlcv_1s_wal as _pyo3_append_wal,
    )
    _PYO3_WAL_FN = _pyo3_append_wal
except Exception as _exc:
    _PYO3_LOAD_ERROR = str(_exc)


def native_wal_append_available() -> bool:
    return _PYO3_WAL_FN is not None


def append_ohlcv_frame_native(
    wal_path: str | os.PathLike[str],
    frame: pl.DataFrame,
    *,
    fsync_after_write: bool,
) -> int | None:
    if _PYO3_WAL_FN is None:
        return None
    if frame.is_empty():
        return 0

    prepared = frame.select(["datetime", "open", "high", "low", "close", "volume"]).with_columns(
        [
            pl.col("datetime").dt.epoch("ms").cast(pl.Int64).alias("timestamp_ms"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]
    )

    timestamps = np.ascontiguousarray(
        prepared.get_column("timestamp_ms").to_numpy(), dtype=np.int64
    )
    opens = np.ascontiguousarray(prepared.get_column("open").to_numpy(), dtype=np.float64)
    highs = np.ascontiguousarray(prepared.get_column("high").to_numpy(), dtype=np.float64)
    lows = np.ascontiguousarray(prepared.get_column("low").to_numpy(), dtype=np.float64)
    closes = np.ascontiguousarray(prepared.get_column("close").to_numpy(), dtype=np.float64)
    volumes = np.ascontiguousarray(prepared.get_column("volume").to_numpy(), dtype=np.float64)

    try:
        written = int(
            _PYO3_WAL_FN(
                os.fspath(wal_path),
                timestamps,
                opens,
                highs,
                lows,
                closes,
                volumes,
                bool(fsync_after_write),
            )
        )
        return written
    except Exception as exc:
        raise RuntimeError(f"Native WAL append failed: {exc}") from exc


__all__ = [
    "append_ohlcv_frame_native",
    "native_wal_append_available",
]
