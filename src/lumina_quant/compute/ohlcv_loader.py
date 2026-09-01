"""Reusable OHLCV frame loader and normalizer.

This module centralizes CSV->Polars ingestion so backtest and optimization
paths share the same column projection/filter/sort behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from lumina_quant.compute.ohlcv_validation import assert_valid_ohlcv_frame

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
    validation: str = "none"
    allow_eager_fallback: bool = False

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
            if self._strict_validation:
                assert_valid_ohlcv_frame(frame, context="ohlcv_normalize")
            return None
        if not has_required_ohlcv_columns(frame, self.columns):
            if self._strict_validation:
                assert_valid_ohlcv_frame(
                    frame,
                    context="ohlcv_normalize",
                    required_columns=self.columns,
                )
            return None

        out = self._normalize_datetime_column(frame.select(list(self.columns)))
        if self._strict_validation:
            assert_valid_ohlcv_frame(
                out,
                context="ohlcv_normalize_pre_sort",
                required_columns=self.columns,
            )
        start_bound = self._coerce_bound(self.start_date)
        end_bound = self._coerce_bound(self.end_date)
        if start_bound is not None:
            out = out.filter(pl.col("datetime") >= start_bound)
        if end_bound is not None:
            out = out.filter(pl.col("datetime") <= end_bound)
        out = out.sort("datetime")
        if self._strict_validation:
            assert_valid_ohlcv_frame(
                out,
                context="ohlcv_normalize",
                required_columns=self.columns,
            )
        return out

    @property
    def _strict_validation(self) -> bool:
        return str(self.validation or "").strip().lower() in {
            "strict",
            "fail_closed",
            "fail-closed",
        }

    def load_csv(self, csv_path: str) -> pl.DataFrame | None:
        """Load OHLCV CSV with lazy pushdown first.

        Strict validation is fail-closed by default: lazy/streaming ingestion
        failures are raised instead of silently switching engines.  Eager
        fallback remains an explicit compatibility opt-in for non-production
        ingestion environments.
        """
        if self._strict_validation:
            try:
                frame = (
                    pl.scan_csv(csv_path, try_parse_dates=True)
                    .select(list(self.columns))
                    .collect(engine="streaming")
                )
            except pl.exceptions.PolarsError:
                if not self.allow_eager_fallback:
                    raise
                frame = pl.read_csv(csv_path, try_parse_dates=True)
            return self.normalize(frame)

        try:
            lazy_frame = pl.scan_csv(csv_path, try_parse_dates=True).select(list(self.columns))
            start_bound = self._coerce_bound(self.start_date)
            end_bound = self._coerce_bound(self.end_date)
            if start_bound is not None:
                lazy_frame = lazy_frame.filter(pl.col("datetime") >= start_bound)
            if end_bound is not None:
                lazy_frame = lazy_frame.filter(pl.col("datetime") <= end_bound)
            frame = lazy_frame.collect(engine="streaming")
            out = self._normalize_datetime_column(frame).sort("datetime")
            if self._strict_validation:
                assert_valid_ohlcv_frame(
                    out,
                    context=f"ohlcv_csv:{csv_path}",
                    required_columns=self.columns,
                )
            return out
        except Exception:
            if self._strict_validation and not self.allow_eager_fallback:
                raise

        try:
            eager = pl.read_csv(csv_path, try_parse_dates=True)
        except Exception:
            if self._strict_validation:
                raise
            return None
        return self.normalize(eager)


def normalize_ohlcv_frame(
    frame: pl.DataFrame | None,
    *,
    start_date: Any = None,
    end_date: Any = None,
    validation: str = "none",
) -> pl.DataFrame | None:
    """Functional helper for one-off frame normalization."""
    loader = OHLCVFrameLoader(start_date=start_date, end_date=end_date, validation=validation)
    return loader.normalize(frame)


def load_csv_ohlcv(
    csv_path: str,
    *,
    start_date: Any = None,
    end_date: Any = None,
    validation: str = "none",
) -> pl.DataFrame | None:
    """Functional helper for one-off OHLCV CSV loading."""
    loader = OHLCVFrameLoader(start_date=start_date, end_date=end_date, validation=validation)
    return loader.load_csv(csv_path)
