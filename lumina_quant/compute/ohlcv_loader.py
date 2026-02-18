"""Reusable OHLCV frame loader and normalizer.

This module centralizes CSV->Polars ingestion so backtest and optimization
paths share the same column projection/filter/sort behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def has_required_ohlcv_columns(
    frame: pl.DataFrame,
    columns: tuple[str, ...] = REQUIRED_OHLCV_COLUMNS,
) -> bool:
    """Return True when all required OHLCV columns exist."""
    return all(column in frame.columns for column in columns)


@dataclass(slots=True)
class OHLCVFrameLoader:
    """Load and normalize canonical OHLCV frames."""

    start_date: Any = None
    end_date: Any = None
    columns: tuple[str, ...] = REQUIRED_OHLCV_COLUMNS

    def normalize(self, frame: pl.DataFrame | None) -> pl.DataFrame | None:
        """Select canonical columns and apply optional date filtering.

        Returns None when the incoming frame is missing required columns.
        """
        if frame is None:
            return None
        if not has_required_ohlcv_columns(frame, self.columns):
            return None

        out = frame.select(list(self.columns))
        if self.start_date is not None:
            out = out.filter(pl.col("datetime") >= self.start_date)
        if self.end_date is not None:
            out = out.filter(pl.col("datetime") <= self.end_date)
        return out.sort("datetime")

    def load_csv(self, csv_path: str) -> pl.DataFrame | None:
        """Load OHLCV CSV with lazy pushdown first, eager fallback on failure."""
        try:
            lazy_frame = pl.scan_csv(csv_path, try_parse_dates=True).select(list(self.columns))
            if self.start_date is not None:
                lazy_frame = lazy_frame.filter(pl.col("datetime") >= self.start_date)
            if self.end_date is not None:
                lazy_frame = lazy_frame.filter(pl.col("datetime") <= self.end_date)
            frame = lazy_frame.collect(engine="streaming")
            return frame.sort("datetime")
        except Exception:
            pass

        try:
            eager = pl.read_csv(csv_path, try_parse_dates=True)
        except Exception:
            return None
        return self.normalize(eager)


def normalize_ohlcv_frame(
    frame: pl.DataFrame | None,
    *,
    start_date: Any = None,
    end_date: Any = None,
) -> pl.DataFrame | None:
    """Functional helper for one-off frame normalization."""
    loader = OHLCVFrameLoader(start_date=start_date, end_date=end_date)
    return loader.normalize(frame)


def load_csv_ohlcv(
    csv_path: str,
    *,
    start_date: Any = None,
    end_date: Any = None,
) -> pl.DataFrame | None:
    """Functional helper for one-off OHLCV CSV loading."""
    loader = OHLCVFrameLoader(start_date=start_date, end_date=end_date)
    return loader.load_csv(csv_path)
