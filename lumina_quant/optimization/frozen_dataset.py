"""Frozen dataset build stage for optimization and sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from lumina_quant.optimization.constants import (
    FROZEN_DEFAULT_ROLLING_WINDOW,
    FROZEN_DEFAULT_SPLIT_KEY,
    TIMESTAMP_MEDIUM_SCALE_TO_NS,
    TIMESTAMP_MEDIUM_THRESHOLD,
    TIMESTAMP_SMALL_SCALE_TO_NS,
    TIMESTAMP_SMALL_THRESHOLD,
)


@dataclass(slots=True)
class FrozenDataset:
    symbol_index: dict[str, tuple[int, int]]
    timestamp_ns: np.ndarray
    close: np.ndarray
    returns: np.ndarray
    roll_mean_32: np.ndarray
    roll_std_32: np.ndarray
    split_ranges: dict[str, tuple[int, int]]


def _as_ns_epoch(value) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp() * 1_000_000_000)
    if isinstance(value, (int, float)):
        raw = int(value)
        if abs(raw) < TIMESTAMP_SMALL_THRESHOLD:
            return raw * TIMESTAMP_SMALL_SCALE_TO_NS
        if abs(raw) < TIMESTAMP_MEDIUM_THRESHOLD:
            return raw * TIMESTAMP_MEDIUM_SCALE_TO_NS
        return raw
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return int(parsed.timestamp() * 1_000_000_000)
    except Exception:
        return 0


def _build_split_ranges(timestamp_ns: np.ndarray, split: dict | None) -> dict[str, tuple[int, int]]:
    if not split:
        return {FROZEN_DEFAULT_SPLIT_KEY: (0, int(timestamp_ns.shape[0]))}

    out: dict[str, tuple[int, int]] = {}
    boundaries = {
        "train": (split.get("train_start"), split.get("train_end")),
        "val": (split.get("val_start"), split.get("val_end")),
        "test": (split.get("test_start"), split.get("test_end")),
    }

    for key, (start_dt, end_dt) in boundaries.items():
        if start_dt is None or end_dt is None:
            continue
        start_ns = _as_ns_epoch(start_dt)
        end_ns = _as_ns_epoch(end_dt)
        lo = int(np.searchsorted(timestamp_ns, start_ns, side="left"))
        hi = int(np.searchsorted(timestamp_ns, end_ns, side="right"))
        if hi < lo:
            hi = lo
        out[key] = (lo, hi)
    if not out:
        out[FROZEN_DEFAULT_SPLIT_KEY] = (0, int(timestamp_ns.shape[0]))
    return out


def build_frozen_dataset(
    data_dict: dict[str, pl.DataFrame],
    split: dict | None = None,
    *,
    rolling_window: int = FROZEN_DEFAULT_ROLLING_WINDOW,
) -> FrozenDataset:
    """Run one-time ETL and produce contiguous arrays reused in all trials."""
    rolling_window_i = max(1, int(rolling_window))
    if not data_dict:
        empty = np.asarray([], dtype=np.float64)
        return FrozenDataset(
            symbol_index={},
            timestamp_ns=np.asarray([], dtype=np.int64),
            close=empty,
            returns=empty,
            roll_mean_32=empty,
            roll_std_32=empty,
            split_ranges={FROZEN_DEFAULT_SPLIT_KEY: (0, 0)},
        )

    frames: list[pl.DataFrame] = []
    for symbol, frame in data_dict.items():
        if frame is None or frame.height == 0:
            continue
        normalized = frame.select(["datetime", "close"]).with_columns(
            [pl.lit(str(symbol)).alias("symbol")]
        )
        frames.append(normalized)

    if not frames:
        raise ValueError("No usable frames found to build frozen dataset")

    merged = pl.concat(frames, how="vertical_relaxed").sort(["symbol", "datetime"])
    merged = merged.with_columns(
        [
            pl.col("datetime")
            .cast(pl.Datetime("ns"))
            .dt.epoch(time_unit="ns")
            .cast(pl.Int64)
            .alias("timestamp_ns"),
            pl.col("close").cast(pl.Float64),
        ]
    )
    merged = merged.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0)
            .fill_null(0.0)
            .alias("returns"),
            pl.col("close")
            .rolling_mean(window_size=rolling_window_i)
            .over("symbol")
            .fill_null(0.0)
            .alias("roll_mean_32"),
            pl.col("close")
            .rolling_std(window_size=rolling_window_i)
            .over("symbol")
            .fill_null(0.0)
            .alias("roll_std_32"),
        ]
    )

    symbols = merged["symbol"].to_list()
    symbol_index: dict[str, tuple[int, int]] = {}
    if symbols:
        current = symbols[0]
        start = 0
        for idx in range(1, len(symbols)):
            if symbols[idx] != current:
                symbol_index[str(current)] = (start, idx)
                current = symbols[idx]
                start = idx
        symbol_index[str(current)] = (start, len(symbols))

    timestamp_ns = merged["timestamp_ns"].to_numpy().astype(np.int64, copy=False)
    split_ranges = _build_split_ranges(timestamp_ns, split)

    return FrozenDataset(
        symbol_index=symbol_index,
        timestamp_ns=timestamp_ns,
        close=merged["close"].to_numpy().astype(np.float64, copy=False),
        returns=merged["returns"].to_numpy().astype(np.float64, copy=False),
        roll_mean_32=merged["roll_mean_32"].to_numpy().astype(np.float64, copy=False),
        roll_std_32=merged["roll_std_32"].to_numpy().astype(np.float64, copy=False),
        split_ranges=split_ranges,
    )
