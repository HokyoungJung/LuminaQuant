"""Monthly-parquet + custom-WAL market-data repository."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.storage.wal_binary import BinaryWAL, WALRecord
from lumina_quant.symbols import canonical_symbol

_DEFAULT_SCHEMA: dict[str, pl.DataType] = {
    "datetime": pl.Datetime(time_unit="ms"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
_KNOWN_QUOTES = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
_TIMEFRAME_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
    "M": 2_592_000_000,
}
_TIMEFRAME_PATTERN = re.compile(r"^([1-9][0-9]*)([smhdwM])$")


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format into BASE/QUOTE uppercase."""
    return canonical_symbol(symbol)


def normalize_timeframe_token(timeframe: str) -> str:
    """Normalize timeframe token while preserving month/minute semantics."""
    raw = str(timeframe or "").strip()
    if not raw:
        raise ValueError("Timeframe cannot be empty")
    if len(raw) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    value = raw[:-1].strip()
    unit_raw = raw[-1]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"Invalid timeframe value: {timeframe}")

    if unit_raw == "M":
        unit = "M"
    else:
        unit = unit_raw.lower()
    token = f"{int(value)}{unit}"
    if _TIMEFRAME_PATTERN.fullmatch(token) is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return token


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert timeframe token like 1m/1h/1d into milliseconds."""
    token = normalize_timeframe_token(timeframe)
    if len(token) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    unit = token[-1]
    value = int(token[:-1])
    if value <= 0:
        raise ValueError(f"Invalid timeframe value: {timeframe}")
    unit_ms = _TIMEFRAME_UNIT_MS.get(unit)
    if unit_ms is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return value * unit_ms


@dataclass(slots=True)
class CompactionResult:
    """Compaction metadata for a single monthly parquet file."""

    partition: str
    files_before: int
    files_after: int
    rows_before: int
    rows_after: int


class ParquetMarketDataRepository:
    """Store and query OHLCV bars in monthly parquet + custom WAL layout.

    Layout:
    - monthly parquet: <root>/market_ohlcv_1s/<exchange>/<symbol>/<YYYY-MM>.parquet
    - wal:             <root>/market_ohlcv_1s/<exchange>/<symbol>/wal.bin
    """

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
    def _datetime_to_ms(value: datetime | None) -> int | None:
        if value is None:
            return None
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(dt.astimezone(UTC).timestamp() * 1000)

    @staticmethod
    def _ms_to_datetime(ts_ms: int) -> datetime:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=UTC).replace(tzinfo=None)

    @staticmethod
    def _month_token(dt: datetime) -> str:
        return f"{dt.year:04d}-{dt.month:02d}"

    @staticmethod
    def _month_token_from_ms(ts_ms: int) -> str:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=UTC).strftime("%Y-%m")

    @staticmethod
    def _iter_month_tokens(start: datetime, end: datetime) -> list[str]:
        cursor = datetime(start.year, start.month, 1)
        stop = datetime(end.year, end.month, 1)
        out: list[str] = []
        while cursor <= stop:
            out.append(f"{cursor.year:04d}-{cursor.month:02d}")
            if cursor.month == 12:
                cursor = datetime(cursor.year + 1, 1, 1)
            else:
                cursor = datetime(cursor.year, cursor.month + 1, 1)
        return out

    def _symbol_root(self, *, exchange: str, symbol: str) -> Path:
        return (
            self.root_path
            / "market_ohlcv_1s"
            / self._normalize_exchange(exchange)
            / self._normalize_symbol_token(symbol)
        )

    def _monthly_path(self, *, exchange: str, symbol: str, month_token: str) -> Path:
        return self._symbol_root(exchange=exchange, symbol=symbol) / f"{month_token}.parquet"

    def _wal_path(self, *, exchange: str, symbol: str) -> Path:
        return self._symbol_root(exchange=exchange, symbol=symbol) / "wal.bin"

    def _meta_path(self, *, exchange: str, symbol: str) -> Path:
        return self._symbol_root(exchange=exchange, symbol=symbol) / "compaction.meta.json"

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
    def _ensure_ohlcv_frame(rows: pl.DataFrame | list[dict[str, Any]] | list[tuple[Any, ...]]) -> pl.DataFrame:
        if isinstance(rows, pl.DataFrame):
            frame = rows
        else:
            frame = pl.DataFrame(rows)

        if frame.is_empty():
            return ParquetMarketDataRepository._empty_ohlcv_frame()

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

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as fh:
            os.fsync(fh.fileno())

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except Exception:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _env_int(*, names: list[str], default: int) -> int:
        for name in names:
            raw = os.getenv(str(name), "").strip()
            if not raw:
                continue
            try:
                return int(raw)
            except Exception:
                continue
        return int(default)

    @staticmethod
    def _env_bool(*, names: list[str], default: bool) -> bool:
        for name in names:
            raw = os.getenv(str(name), "").strip().lower()
            if not raw:
                continue
            if raw in {"1", "true", "yes", "on"}:
                return True
            if raw in {"0", "false", "no", "off"}:
                return False
        return bool(default)

    def _resolve_wal_controls(self) -> tuple[int, bool, int]:
        wal_max_bytes = self._env_int(
            names=["LQ_WAL_MAX_BYTES", "LQ__STORAGE__WAL_MAX_BYTES"],
            default=268435456,
        )
        compact_on_threshold = self._env_bool(
            names=["LQ_WAL_COMPACT_ON_THRESHOLD", "LQ__STORAGE__WAL_COMPACT_ON_THRESHOLD"],
            default=True,
        )
        compaction_interval = self._env_int(
            names=[
                "LQ_WAL_COMPACTION_INTERVAL_SEC",
                "LQ_WAL_COMPACTION_INTERVAL_SECONDS",
                "LQ__STORAGE__WAL_COMPACTION_INTERVAL_SEC",
                "LQ__STORAGE__WAL_COMPACTION_INTERVAL_SECONDS",
            ],
            default=3600,
        )
        return max(0, int(wal_max_bytes)), bool(compact_on_threshold), max(0, int(compaction_interval))

    @staticmethod
    def _parse_iso_utc(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _enforce_wal_growth_controls(self, *, exchange: str, symbol: str) -> None:
        wal_max_bytes, compact_on_threshold, compaction_interval_seconds = self._resolve_wal_controls()
        if wal_max_bytes <= 0:
            return

        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        if not wal_path.exists():
            return
        wal_size = int(wal_path.stat().st_size)
        if wal_size <= int(wal_max_bytes):
            return

        now = datetime.now(tz=UTC)
        meta = self._read_meta(exchange=exchange, symbol=symbol)
        meta["wal_compaction_required"] = True
        meta["last_wal_over_limit_detected_at"] = now.isoformat()
        last_attempt = self._parse_iso_utc(meta.get("last_compaction_attempt_at"))
        elapsed = None if last_attempt is None else max(0.0, (now - last_attempt).total_seconds())
        if not bool(compact_on_threshold):
            last_warning = self._parse_iso_utc(meta.get("last_wal_over_limit_warning_at"))
            warn_elapsed = None if last_warning is None else (now - last_warning).total_seconds()
            if warn_elapsed is None or warn_elapsed >= 60.0:
                print(
                    f"[WARN] WAL size {wal_size} bytes exceeds limit {wal_max_bytes} for {exchange}:{symbol}. "
                    "wal_compact_on_threshold=false, manual compaction required "
                    "(scripts/compact_wal_to_monthly_parquet.py)."
                )
                meta["last_wal_over_limit_warning_at"] = now.isoformat()
            self._write_meta(exchange=exchange, symbol=symbol, payload=meta)
            return

        can_compact = (
            compaction_interval_seconds <= 0
            or elapsed is None
            or float(elapsed) >= float(compaction_interval_seconds)
        )
        if can_compact:
            print(
                f"[WARN] WAL size {wal_size} bytes exceeds limit {wal_max_bytes} "
                f"for {exchange}:{symbol}. Triggering compaction."
            )
            try:
                self.compact_wal_to_monthly_parquet(
                    exchange=exchange,
                    symbol=symbol,
                    remove_sources=True,
                )
                meta = self._read_meta(exchange=exchange, symbol=symbol)
                meta["wal_compaction_required"] = False
                meta["last_wal_compaction_resolved_at"] = now.isoformat()
                self._write_meta(exchange=exchange, symbol=symbol, payload=meta)
            except Exception as exc:
                print(
                    f"[WARN] WAL compaction trigger failed for {exchange}:{symbol}: {exc}. "
                    "Continuing without blocking writes."
                )
                self._write_meta(exchange=exchange, symbol=symbol, payload=meta)
            return

        last_warning = self._parse_iso_utc(meta.get("last_wal_over_limit_warning_at"))
        warn_elapsed = None if last_warning is None else (now - last_warning).total_seconds()
        if warn_elapsed is None or warn_elapsed >= 60.0:
            wait_seconds = max(0, int(compaction_interval_seconds - int(elapsed or 0)))
            print(
                f"[WARN] WAL size {wal_size} bytes exceeds limit {wal_max_bytes} for {exchange}:{symbol} "
                f"but compaction interval not reached; retry in ~{wait_seconds}s."
            )
            meta["last_wal_over_limit_warning_at"] = now.isoformat()
        self._write_meta(exchange=exchange, symbol=symbol, payload=meta)

    def _read_meta(self, *, exchange: str, symbol: str) -> dict[str, Any]:
        path = self._meta_path(exchange=exchange, symbol=symbol)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_meta(self, *, exchange: str, symbol: str, payload: dict[str, Any]) -> None:
        path = self._meta_path(exchange=exchange, symbol=symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._fsync_file(tmp)
        tmp.replace(path)
        self._fsync_dir(path.parent)

    def _monthly_files_for_range(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[Path]:
        symbol_root = self._symbol_root(exchange=exchange, symbol=symbol)
        all_monthly = sorted(symbol_root.glob("????-??.parquet"))
        if not all_monthly:
            return []

        if start_date is None and end_date is None:
            return all_monthly

        if start_date is None:
            start_date = datetime(1970, 1, 1)
        if end_date is None:
            end_date = datetime(3000, 1, 1)
        months = set(self._iter_month_tokens(start_date, end_date))
        return [path for path in all_monthly if path.stem in months]

    def _load_monthly_frame(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.DataFrame:
        files = self._monthly_files_for_range(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if not files:
            return self._empty_ohlcv_frame()

        lazy = pl.scan_parquet([str(path) for path in files]).select(
            ["datetime", "open", "high", "low", "close", "volume"]
        )
        if start_date is not None:
            lazy = lazy.filter(pl.col("datetime") >= start_date)
        if end_date is not None:
            lazy = lazy.filter(pl.col("datetime") <= end_date)

        out = self._collect_lazy(lazy)
        if out.is_empty():
            return self._empty_ohlcv_frame()
        return out.sort("datetime")

    def _load_wal_frame(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.DataFrame:
        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        if not wal_path.exists():
            return self._empty_ohlcv_frame()

        wal = BinaryWAL(wal_path, auto_repair=True)
        start_ms = self._datetime_to_ms(start_date)
        end_ms = self._datetime_to_ms(end_date)
        records = list(wal.iter_range(start_ms, end_ms))
        if not records:
            return self._empty_ohlcv_frame()

        return pl.DataFrame(
            {
                "datetime": [self._ms_to_datetime(item.ts_ms) for item in records],
                "open": [item.open for item in records],
                "high": [item.high for item in records],
                "low": [item.low for item in records],
                "close": [item.close for item in records],
                "volume": [item.volume for item in records],
                "_seq": list(range(len(records))),
            }
        ).with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))

    def _merge_monthly_and_wal(
        self,
        *,
        monthly: pl.DataFrame,
        wal: pl.DataFrame,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        merge_cols = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "_source_priority",
            "_seq",
        ]

        if not monthly.is_empty():
            frames.append(
                monthly.with_columns(
                    [
                        pl.lit(0).cast(pl.Int8).alias("_source_priority"),
                        pl.int_range(pl.len()).alias("_seq"),
                    ]
                ).select(merge_cols)
            )
        if not wal.is_empty():
            frames.append(
                wal.with_columns(pl.lit(1).cast(pl.Int8).alias("_source_priority")).select(merge_cols)
            )

        if not frames:
            return self._empty_ohlcv_frame()

        merged = (
            pl.concat(frames, how="vertical_relaxed")
            .sort(["datetime", "_source_priority", "_seq"])
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
            .drop(["_source_priority", "_seq"], strict=False)
        )
        if merged.is_empty():
            return self._empty_ohlcv_frame()
        return merged.select(["datetime", "open", "high", "low", "close", "volume"])

    def upsert_1s(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: pl.DataFrame | list[dict[str, Any]] | list[tuple[Any, ...]],
    ) -> int:
        """Append OHLCV 1s rows into custom binary WAL."""
        frame = self._ensure_ohlcv_frame(rows)
        if frame.is_empty():
            return 0

        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        fsync_n = max(1, int(os.getenv("LQ_WAL_FSYNC_EVERY_N_BATCHES", "1") or "1"))
        wal = BinaryWAL(wal_path, fsync_every_n_batches=fsync_n, auto_repair=True)

        records = [
            WALRecord(
                ts_ms=self._datetime_to_ms(item[0]) or 0,
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
            )
            for item in frame.iter_rows(named=False)
        ]
        appended = int(wal.append(records))
        self._enforce_wal_growth_controls(exchange=exchange, symbol=symbol)
        return appended

    def _load_ohlcv_1s_merged(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        start_dt = self._coerce_datetime(start_date)
        end_dt = self._coerce_datetime(end_date)

        monthly = self._load_monthly_frame(
            exchange=exchange,
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
        )
        wal = self._load_wal_frame(
            exchange=exchange,
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
        )
        return self._merge_monthly_and_wal(monthly=monthly, wal=wal)

    def load_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        """Load OHLCV using monthly parquet + WAL merge and bucket resampling."""
        timeframe_token = normalize_timeframe_token(timeframe)
        merged_1s = self._load_ohlcv_1s_merged(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if merged_1s.is_empty():
            return self._empty_ohlcv_frame()

        if timeframe_token == "1s":
            return merged_1s

        tf_ms = int(timeframe_to_milliseconds(timeframe_token))
        source = merged_1s.lazy().with_columns(
            pl.col("datetime").dt.epoch("ms").alias("timestamp_ms")
        ).with_columns(((pl.col("timestamp_ms") // tf_ms) * tf_ms).alias("bucket_ms"))

        # GPU-friendly aggregation: scalar expressions only, no UDF/group_by_dynamic.
        aggregated = (
            source.group_by("bucket_ms")
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
        """Load timeframe bars by chunk-days windows with optional warmup overlap."""
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
            query_start = (
                chunk_start - timedelta(milliseconds=warmup_ms) if warmup_ms > 0 else chunk_start
            )
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

    def compact_wal_to_monthly_parquet(
        self,
        *,
        exchange: str,
        symbol: str,
        remove_sources: bool = True,
    ) -> list[CompactionResult]:
        """Compact WAL records into monthly parquet files atomically."""
        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        if not wal_path.exists():
            return []

        wal = BinaryWAL(wal_path, auto_repair=True)
        wal_size = wal.size_bytes()

        meta = self._read_meta(exchange=exchange, symbol=symbol)
        offset = int(meta.get("wal_offset", 0) or 0)
        if offset < 0 or offset > wal_size:
            offset = 0

        records = list(wal.iter_records_from_offset(offset))
        if not records:
            return []

        by_month: dict[str, list[WALRecord]] = {}
        for record in records:
            by_month.setdefault(self._month_token_from_ms(record.ts_ms), []).append(record)

        results: list[CompactionResult] = []
        for month_token in sorted(by_month):
            monthly_path = self._monthly_path(exchange=exchange, symbol=symbol, month_token=month_token)
            monthly_path.parent.mkdir(parents=True, exist_ok=True)

            existing = pl.read_parquet(monthly_path) if monthly_path.exists() else self._empty_ohlcv_frame()
            incoming_rows = by_month[month_token]
            incoming = pl.DataFrame(
                {
                    "datetime": [self._ms_to_datetime(item.ts_ms) for item in incoming_rows],
                    "open": [item.open for item in incoming_rows],
                    "high": [item.high for item in incoming_rows],
                    "low": [item.low for item in incoming_rows],
                    "close": [item.close for item in incoming_rows],
                    "volume": [item.volume for item in incoming_rows],
                    "_seq": list(range(len(incoming_rows))),
                }
            ).with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))

            merged = self._merge_monthly_and_wal(monthly=existing, wal=incoming)

            tmp_path = monthly_path.with_suffix(".tmp.parquet")
            merged.write_parquet(tmp_path, compression="zstd", statistics=True)
            self._fsync_file(tmp_path)
            tmp_path.replace(monthly_path)
            self._fsync_dir(monthly_path.parent)

            results.append(
                CompactionResult(
                    partition=str(monthly_path),
                    files_before=1 if existing.height > 0 else 0,
                    files_after=1,
                    rows_before=int(existing.height + incoming.height),
                    rows_after=int(merged.height),
                )
            )

        if remove_sources:
            wal.truncate()
            next_offset = 0
        else:
            next_offset = wal.size_bytes()

        self._write_meta(
            exchange=exchange,
            symbol=symbol,
            payload={
                **meta,
                "wal_offset": int(next_offset),
                "updated_at": datetime.now(tz=UTC).isoformat(),
                "last_compaction_attempt_at": datetime.now(tz=UTC).isoformat(),
                "compacted_rows": len(records),
                "remove_sources": bool(remove_sources),
            },
        )
        return results

    def compact_partition(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_date: str | date,
        timeframe: str = "1s",
        remove_sources: bool = True,
    ) -> CompactionResult:
        """Compatibility wrapper: compact WAL and return the requested month summary."""
        _ = timeframe
        if isinstance(partition_date, str):
            resolved = date.fromisoformat(partition_date)
        else:
            resolved = partition_date
        month_token = f"{resolved.year:04d}-{resolved.month:02d}"
        results = self.compact_wal_to_monthly_parquet(
            exchange=exchange,
            symbol=symbol,
            remove_sources=remove_sources,
        )
        for result in results:
            if Path(result.partition).stem == month_token:
                return result
        monthly_path = self._monthly_path(exchange=exchange, symbol=symbol, month_token=month_token)
        return CompactionResult(str(monthly_path), 0, int(monthly_path.exists()), 0, 0)

    def compact_all(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str = "1s",
        remove_sources: bool = True,
    ) -> list[CompactionResult]:
        """Compact every WAL-backed month for one symbol."""
        if normalize_timeframe_token(timeframe) != "1s":
            return []
        return self.compact_wal_to_monthly_parquet(
            exchange=exchange,
            symbol=symbol,
            remove_sources=remove_sources,
        )

    def get_symbol_time_range(
        self,
        *,
        exchange: str,
        symbol: str,
    ) -> tuple[datetime | None, datetime | None]:
        """Return min/max datetime across monthly parquet + WAL for one symbol."""
        monthly_files = self._monthly_files_for_range(
            exchange=exchange,
            symbol=symbol,
            start_date=None,
            end_date=None,
        )

        min_dt: datetime | None = None
        max_dt: datetime | None = None

        if monthly_files:
            try:
                first = (
                    pl.scan_parquet(str(monthly_files[0]))
                    .select(pl.col("datetime").min().alias("min_dt"))
                    .collect()
                )
                last = (
                    pl.scan_parquet(str(monthly_files[-1]))
                    .select(pl.col("datetime").max().alias("max_dt"))
                    .collect()
                )
                left = first["min_dt"][0]
                right = last["max_dt"][0]
                if left is not None:
                    min_dt = left
                if right is not None:
                    max_dt = right
            except Exception:
                pass

        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        if wal_path.exists():
            wal = BinaryWAL(wal_path, auto_repair=True)
            first_wal: WALRecord | None = None
            last_wal: WALRecord | None = None
            for record in wal.iter_all():
                if first_wal is None:
                    first_wal = record
                last_wal = record
            if first_wal is not None:
                first_dt = self._ms_to_datetime(first_wal.ts_ms)
                min_dt = first_dt if min_dt is None else min(min_dt, first_dt)
            if last_wal is not None:
                last_dt = self._ms_to_datetime(last_wal.ts_ms)
                max_dt = last_dt if max_dt is None else max(max_dt, last_dt)

        return min_dt, max_dt


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
    return bool(
        any(root.glob("market_ohlcv_1s/*/*/*.parquet"))
        or any(root.glob("market_ohlcv_1s/*/*/wal.bin"))
        or any(root.glob("exchange=*/symbol=*/timeframe=*/date=*/*.parquet"))
    )


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
    """Load OHLCV frames by symbol from monthly parquet+WAL storage."""
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
