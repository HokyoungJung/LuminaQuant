"""Parquet-backed market-data repository for local-only storage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.market_data import (
    normalize_symbol,
    normalize_timeframe_token,
    timeframe_to_milliseconds,
)

_DEFAULT_SCHEMA: dict[str, pl.DataType] = {
    "datetime": pl.Datetime(time_unit="ms"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


@dataclass(slots=True)
class CompactionResult:
    """Compaction metadata for a single parquet partition."""

    partition: str
    files_before: int
    files_after: int
    rows_before: int
    rows_after: int


class ParquetMarketDataRepository:
    """Store and query OHLCV bars in partitioned parquet files."""

    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)

    @staticmethod
    def _normalize_exchange(exchange: str) -> str:
        return str(exchange).strip().lower()

    @staticmethod
    def _normalize_symbol_token(symbol: str) -> str:
        return normalize_symbol(symbol).replace("/", "")

    @staticmethod
    def _empty_ohlcv_frame() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "datetime": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            schema=_DEFAULT_SCHEMA,
        )

    def _series_path(self, *, exchange: str, symbol: str, timeframe: str) -> Path:
        tf = normalize_timeframe_token(timeframe)
        return (
            self.root_path
            / f"exchange={self._normalize_exchange(exchange)}"
            / f"symbol={self._normalize_symbol_token(symbol)}"
            / f"timeframe={tf}"
        )

    def _date_partition_path(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: date,
    ) -> Path:
        base = self._series_path(exchange=exchange, symbol=symbol, timeframe=timeframe)
        return base / f"date={partition_date.isoformat()}"

    @staticmethod
    def _coerce_datetime_expr(expr: pl.Expr) -> pl.Expr:
        parsed = expr.cast(pl.Utf8, strict=False).str.to_datetime(
            strict=False,
            time_zone="UTC",
        )
        as_datetime = expr.cast(pl.Datetime(time_zone="UTC"), strict=False)
        return (
            pl.coalesce([as_datetime, parsed])
            .dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None)
            .cast(pl.Datetime(time_unit="ms"))
        )

    @staticmethod
    def _ensure_ohlcv_frame(rows: pl.DataFrame | list[dict[str, Any]]) -> pl.DataFrame:
        if isinstance(rows, pl.DataFrame):
            frame = rows
        else:
            frame = pl.DataFrame(rows)

        if frame.is_empty():
            return pl.DataFrame(
                {
                    "datetime": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                },
                schema=_DEFAULT_SCHEMA,
            )

        required = ["datetime", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"OHLCV rows missing columns: {missing}")

        casted = frame.select(required).with_columns(
            [
                ParquetMarketDataRepository._coerce_datetime_expr(pl.col("datetime")).alias("datetime"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        return casted.drop_nulls(subset=["datetime"]).sort("datetime")

    def upsert_1s(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: pl.DataFrame | list[dict[str, Any]],
    ) -> int:
        """Append OHLCV 1s rows into date-partitioned parquet files."""
        frame = self._ensure_ohlcv_frame(rows)
        if frame.is_empty():
            return 0

        dated = frame.with_columns(pl.col("datetime").dt.date().alias("partition_date"))
        partitions = dated.partition_by("partition_date", maintain_order=True)

        total_rows = 0
        for partition in partitions:
            partition_date = partition["partition_date"][0]
            if partition_date is None:
                continue
            partition_path = self._date_partition_path(
                exchange=exchange,
                symbol=symbol,
                timeframe="1s",
                partition_date=partition_date,
            )
            partition_path.mkdir(parents=True, exist_ok=True)
            filename = f"part-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}.parquet"
            out_path = partition_path / filename
            partition.drop("partition_date").write_parquet(
                out_path,
                compression="zstd",
                statistics=True,
            )
            total_rows += partition.height

        return total_rows

    def _scan_1s(
        self,
        *,
        exchange: str,
        symbol: str,
    ) -> pl.LazyFrame:
        series_path = self._series_path(exchange=exchange, symbol=symbol, timeframe="1s")
        if not any(series_path.glob("date=*/*.parquet")):
            return self._empty_ohlcv_frame().lazy()
        glob_pattern = str(series_path / "date=*" / "*.parquet")
        return pl.scan_parquet(glob_pattern)

    @staticmethod
    def _filter_range(
        lazy_frame: pl.LazyFrame,
        *,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.LazyFrame:
        out = lazy_frame
        if start_date is not None:
            out = out.filter(pl.col("datetime") >= start_date)
        if end_date is not None:
            out = out.filter(pl.col("datetime") <= end_date)
        return out

    @staticmethod
    def _collect_lazy(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
        """Collect helper that prefers compute_engine adapter when available."""
        try:
            from lumina_quant.compute_engine import resolve_compute_engine

            engine = resolve_compute_engine()
            collect = getattr(engine, "collect", None)
            if callable(collect):
                return collect(lazy_frame)
        except Exception:
            pass
        return lazy_frame.collect(engine="streaming")

    def load_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        """Load OHLCV for timeframe using 1s parquet base with bucket groupby resample."""
        timeframe_token = normalize_timeframe_token(timeframe)
        source = self._filter_range(
            self._scan_1s(exchange=exchange, symbol=symbol),
            start_date=start_date,
            end_date=end_date,
        ).sort("datetime")

        if timeframe_token == "1s":
            return self._collect_lazy(source)

        tf_ms = int(timeframe_to_milliseconds(timeframe_token))
        bucketed = source.with_columns(
            pl.col("datetime").dt.epoch("ms").alias("timestamp_ms")
        ).with_columns(((pl.col("timestamp_ms") // tf_ms) * tf_ms).alias("bucket_ms"))

        # GPU-friendly aggregation: scalar expressions only, no UDF/group_by_dynamic.
        aggregated = (
            bucketed.group_by("bucket_ms")
            .agg(
                [
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                ]
            )
            .sort("bucket_ms")
            .with_columns(pl.from_epoch("bucket_ms", time_unit="ms").alias("datetime"))
            .select(["datetime", "open", "high", "low", "close", "volume"])
        )
        return self._collect_lazy(aggregated)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
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
    def _iter_chunks(
        *,
        start: datetime,
        end: datetime,
        chunk_days: int,
    ) -> list[tuple[datetime, datetime]]:
        if chunk_days <= 0:
            return [(start, end)]
        windows: list[tuple[datetime, datetime]] = []
        cursor = start
        delta = timedelta(days=chunk_days)
        while cursor <= end:
            chunk_end = min(end, cursor + delta - timedelta(microseconds=1))
            windows.append((cursor, chunk_end))
            cursor = chunk_end + timedelta(microseconds=1)
        return windows

    def load_ohlcv_chunked(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
        chunk_days: int = 7,
        warmup_bars: int = 0,
    ) -> pl.DataFrame:
        """Load timeframe bars via bounded weekly chunks with warmup continuity."""
        start_dt = self._coerce_datetime(start_date)
        end_dt = self._coerce_datetime(end_date)
        if start_dt is None or end_dt is None or start_dt > end_dt:
            return self.load_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

        timeframe_token = normalize_timeframe_token(timeframe)
        tf_ms = int(timeframe_to_milliseconds(timeframe_token))
        warmup_ms = max(0, int(warmup_bars)) * max(1, tf_ms)
        frames: list[pl.DataFrame] = []
        for chunk_start, chunk_end in self._iter_chunks(
            start=start_dt,
            end=end_dt,
            chunk_days=max(1, int(chunk_days)),
        ):
            query_start = chunk_start - timedelta(milliseconds=warmup_ms) if warmup_ms > 0 else chunk_start
            chunk = self.load_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe_token,
                start_date=query_start,
                end_date=chunk_end,
            )
            if chunk.is_empty():
                continue
            trimmed = chunk.filter(pl.col("datetime") >= chunk_start)
            if not trimmed.is_empty():
                frames.append(trimmed)

        if not frames:
            return self._empty_ohlcv_frame()

        return (
            pl.concat(frames, how="vertical")
            .sort("datetime")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
        )

    def compact_partition(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_date: str | date,
        timeframe: str = "1s",
        remove_sources: bool = True,
    ) -> CompactionResult:
        """Compact one partition into a single deduplicated parquet file."""
        if isinstance(partition_date, str):
            resolved_date = date.fromisoformat(partition_date)
        else:
            resolved_date = partition_date

        partition_path = self._date_partition_path(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            partition_date=resolved_date,
        )
        files = sorted(partition_path.glob("*.parquet"))
        if not files:
            return CompactionResult(str(partition_path), 0, 0, 0, 0)

        frames = [pl.read_parquet(path) for path in files]
        rows_before = int(sum(frame.height for frame in frames))

        merged = (
            pl.concat(frames, how="vertical")
            .sort("datetime")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
        )

        compacted_path = partition_path / f"compact-{resolved_date.isoformat()}.parquet"
        tmp_path = compacted_path.with_suffix(".tmp.parquet")
        merged.write_parquet(tmp_path, compression="zstd", statistics=True)
        tmp_path.replace(compacted_path)

        if remove_sources:
            for path in files:
                if path != compacted_path and path.exists():
                    path.unlink()

        files_after = len(list(partition_path.glob("*.parquet")))
        return CompactionResult(
            partition=str(partition_path),
            files_before=len(files),
            files_after=files_after,
            rows_before=rows_before,
            rows_after=int(merged.height),
        )

    def compact_all(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str = "1s",
        remove_sources: bool = True,
    ) -> list[CompactionResult]:
        """Compact every partition under one exchange/symbol/timeframe series."""
        series_path = self._series_path(exchange=exchange, symbol=symbol, timeframe=timeframe)
        results: list[CompactionResult] = []
        for partition_path in sorted(series_path.glob("date=*")):
            date_token = partition_path.name.split("date=", maxsplit=1)[-1]
            if not date_token:
                continue
            results.append(
                self.compact_partition(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    partition_date=date_token,
                    remove_sources=remove_sources,
                )
            )
        return results


def is_parquet_market_data_store(path: str, *, backend: str | None = None) -> bool:
    """Heuristic detector for parquet-backed market storage path."""
    backend_token = str(backend or "").strip().lower()
    if backend_token in {"parquet", "local", "postgres_parquet", "parquet_postgres"}:
        return True
    raw = str(path or "").strip()
    if not raw:
        return False
    if raw.lower().endswith(".parquet"):
        return True
    if "parquet" in Path(raw).name.lower():
        return True
    root = Path(raw)
    if not root.exists():
        return False
    return any(root.glob("exchange=*/symbol=*/timeframe=*/date=*/*.parquet"))


def load_data_dict_from_parquet(
    root_path: str,
    *,
    exchange: str,
    symbol_list: list[str],
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
    chunk_days: int = 7,
    warmup_bars: int = 0,
) -> dict[str, pl.DataFrame]:
    """Load OHLCV frames by symbol from partitioned parquet storage."""
    repo = ParquetMarketDataRepository(root_path)
    out: dict[str, pl.DataFrame] = {}
    for symbol in symbol_list:
        frame = repo.load_ohlcv_chunked(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            chunk_days=chunk_days,
            warmup_bars=warmup_bars,
        )
        if not frame.is_empty():
            out[str(symbol)] = frame
    return out
