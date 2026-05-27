"""Optional native helpers for WAL append acceleration."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.data.native_raw_first_backend import load_rawfirst_native_library

_NATIVE_WAL_FN: Any = None


def _load_native_wal_function() -> Any | None:
    global _NATIVE_WAL_FN
    if _NATIVE_WAL_FN is not None:
        return _NATIVE_WAL_FN

    handle = load_rawfirst_native_library()
    if handle is None:
        return None

    try:
        fn = handle.append_ohlcv_1s_wal
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        fn.restype = ctypes.c_int32
    except Exception:
        return None

    _NATIVE_WAL_FN = fn
    return fn


def native_wal_append_available() -> bool:
    return _load_native_wal_function() is not None


def append_ohlcv_frame_native(
    wal_path: str | os.PathLike[str] | Path,
    frame: pl.DataFrame,
    *,
    fsync_after_write: bool,
) -> int | None:
    fn = _load_native_wal_function()
    if fn is None:
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

    output_len = ctypes.c_int32(0)
    status = int(
        fn(
            ctypes.c_char_p(os.fsencode(os.fspath(wal_path))),
            timestamps.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
            opens.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            highs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            lows.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            closes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            volumes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(prepared.height),
            1 if fsync_after_write else 0,
            ctypes.byref(output_len),
        )
    )
    if status != 0:
        raise RuntimeError(f"Native WAL append failed with status={status}")
    return int(output_len.value)


__all__ = [
    "append_ohlcv_frame_native",
    "native_wal_append_available",
]
