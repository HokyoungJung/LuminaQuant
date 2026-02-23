"""Reusable OHLCV frame loader and normalizer.

This module centralizes CSV->Polars ingestion so backtest and optimization
paths share the same column projection/filter/sort behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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

    @staticmethod
    def _coerce_bound(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            numeric = int(value)
            if abs(numeric) < 100_000_000_000:
                numeric *= 1000
            dt = datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
        else:
            text = str(value).strip()
            if not text:
                return None
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            return dt.astimezone(UTC).replace(tzinfo=None)
        return dt

    @staticmethod
    def _normalize_datetime_column(frame: pl.DataFrame) -> pl.DataFrame:
        dtype = frame.schema.get("datetime")
        time_zone = getattr(dtype, "time_zone", None)
        if time_zone is None:
            return frame
        return frame.with_columns(
            pl.col("datetime").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )

    def normalize(self, frame: pl.DataFrame | None) -> pl.DataFrame | None:
        """Select canonical columns and apply optional date filtering.

        Returns None when the incoming frame is missing required columns.
        """
        if frame is None:
            return None
        if not has_required_ohlcv_columns(frame, self.columns):
            return None

        out = self._normalize_datetime_column(frame.select(list(self.columns)))
        start_bound = self._coerce_bound(self.start_date)
        end_bound = self._coerce_bound(self.end_date)
        if start_bound is not None:
            out = out.filter(pl.col("datetime") >= start_bound)
        if end_bound is not None:
            out = out.filter(pl.col("datetime") <= end_bound)
        return out.sort("datetime")

    def load_csv(self, csv_path: str) -> pl.DataFrame | None:
        """Load OHLCV CSV with lazy pushdown first, eager fallback on failure."""
        try:
            lazy_frame = pl.scan_csv(csv_path, try_parse_dates=True).select(list(self.columns))
            start_bound = self._coerce_bound(self.start_date)
            end_bound = self._coerce_bound(self.end_date)
            if start_bound is not None:
                lazy_frame = lazy_frame.filter(pl.col("datetime") >= start_bound)
            if end_bound is not None:
                lazy_frame = lazy_frame.filter(pl.col("datetime") <= end_bound)
            frame = lazy_frame.collect(engine="streaming")
            return self._normalize_datetime_column(frame).sort("datetime")
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
