"""Monthly-parquet + custom-WAL market-data repository."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.backtesting.cli_contract import (
    RawFirstDataMissingError,
    RawFirstManifestInvalidError,
    RawFirstStaleWindowError,
    normalize_data_mode,
)
from lumina_quant.data.raw_first_lineage import resample_1s_frame
from lumina_quant.data.timeframe import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.eval.exact_window_runtime import HeavyRunActiveError, HeavyRunLock
from lumina_quant.storage.wal.binary import BinaryWAL, WALRecord
from lumina_quant.storage.wal.native_backend import append_ohlcv_frame_native
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
_OHLCV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
_MANIFEST_REQUIRED_FIELDS = (
    "manifest_version",
    "commit_id",
    "symbol",
    "timeframe",
    "partition",
    "window_start_ms",
    "window_end_ms",
    "event_time_watermark_ms",
    "source_checkpoint_start",
    "source_checkpoint_end",
    "row_count",
    "canonical_row_checksum",
    "data_files",
    "created_at_utc",
    "producer",
    "status",
)
_RAW_AGGTRADES_SCHEMA: dict[str, pl.DataType] = {
    "agg_trade_id": pl.Int64,
    "timestamp_ms": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "is_buyer_maker": pl.Boolean,
}
_RAW_AGGTRADES_REQUIRED_COLUMNS = tuple(_RAW_AGGTRADES_SCHEMA.keys())
_MATERIALIZED_REQUIRED_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")
_MATERIALIZED_REQUIRED_MANIFEST_FIELDS = (
    "manifest_version",
    "commit_id",
    "symbol",
    "timeframe",
    "partition",
    "window_start_ms",
    "window_end_ms",
    "event_time_watermark_ms",
    "source_checkpoint_start",
    "source_checkpoint_end",
    "row_count",
    "canonical_row_checksum",
    "data_files",
    "created_at_utc",
    "producer",
    "status",
)
_RAW_PART_PATTERN = re.compile(r"^part-(\d+)\.parquet$")


class RawPartitionBusyError(RuntimeError):
    """Raised when a raw partition cannot acquire its writer lock in time."""


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format into BASE/QUOTE uppercase."""
    return canonical_symbol(symbol)


# ``normalize_timeframe_token`` / ``timeframe_to_milliseconds`` are imported from
# ``lumina_quant.data.timeframe`` (the single canonical timeframe util) and
# re-exported here for the existing in-module callers and ``storage.parquet``
# package re-exports.


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
    def _partition_date_token(partition_date: str | date) -> str:
        if isinstance(partition_date, date):
            return partition_date.strftime("%Y-%m-%d")
        return str(partition_date).strip()

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

    def _raw_symbol_root(self, *, exchange: str, symbol: str) -> Path:
        return (
            self.root_path
            / "market_data_raw_aggtrades"
            / self._normalize_exchange(exchange)
            / self._normalize_symbol_token(symbol)
        )

    def raw_partition_path(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_date: str | date,
    ) -> Path:
        return (
            self._raw_partition_path(
                exchange=exchange,
                symbol=symbol,
                partition_date=partition_date,
            )
            / "part-0000.parquet"
        )

    def raw_checkpoint_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_checkpoint_path(exchange=exchange, symbol=symbol)

    def raw_wal_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_wal_path(exchange=exchange, symbol=symbol)

    def _materialized_symbol_root(self, *, exchange: str, symbol: str) -> Path:
        return (
            self.root_path
            / "market_data_materialized"
            / self._normalize_exchange(exchange)
            / self._normalize_symbol_token(symbol)
        )

    def materialized_partition_root(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str | date,
    ) -> Path:
        return self._materialized_date_root(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            partition_date=partition_date,
        )

    def materialized_manifest_path(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str | date,
    ) -> Path:
        return self._materialized_manifest_path(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            partition_date=partition_date,
        )

    @staticmethod
    def _date_token_from_ms(ts_ms: int) -> str:
        return datetime.fromtimestamp(float(int(ts_ms)) / 1000.0, tz=UTC).strftime("%Y-%m-%d")

    @staticmethod
    def _format_checksum_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return format(float(value), ".12g")
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.astimezone(UTC).isoformat()
        return str(value)

    @classmethod
    def canonical_row_checksum(cls, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return sha256(b"").hexdigest()
        required = [column for column in _OHLCV_COLUMNS if column in frame.columns]
        if not required:
            return sha256(b"").hexdigest()
        ordered = frame.select(required).sort("datetime")
        digest = sha256()
        for row in ordered.iter_rows(named=True):
            line = "|".join(cls._format_checksum_value(row.get(col)) for col in required)
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _validate_manifest_payload(
        self,
        *,
        manifest: dict[str, Any],
        exchange: str,
        symbol: str,
        timeframe: str,
        manifest_path: Path,
    ) -> dict[str, Any]:
        missing = [
            field for field in _MANIFEST_REQUIRED_FIELDS if manifest.get(field) in {None, ""}
        ]
        if missing:
            raise RawFirstManifestInvalidError(
                f"Manifest missing required fields {missing} at {manifest_path}."
            )

        if str(manifest.get("status", "")).strip().lower() != "committed":
            raise RawFirstDataMissingError(f"Manifest status is not committed at {manifest_path}.")

        expected_exchange = self._normalize_exchange(exchange)
        expected_symbol = self._normalize_symbol_token(symbol)
        if self._normalize_symbol_token(str(manifest.get("symbol", ""))) != expected_symbol:
            raise RawFirstManifestInvalidError(
                f"Manifest symbol mismatch at {manifest_path}: expected {symbol}."
            )
        if normalize_timeframe_token(
            str(manifest.get("timeframe", ""))
        ) != normalize_timeframe_token(timeframe):
            raise RawFirstManifestInvalidError(
                f"Manifest timeframe mismatch at {manifest_path}: expected {timeframe}."
            )
        partition = str(manifest.get("partition", "")).strip()
        if (
            partition
            and f"/{expected_exchange}/" not in partition
            and f"\\{expected_exchange}\\" not in partition
            and expected_exchange not in partition
        ):
            # Best-effort exchange sanity check while keeping backward compatibility.
            raise RawFirstManifestInvalidError(
                f"Manifest partition exchange mismatch at {manifest_path}: expected {expected_exchange}."
            )
        try:
            int(manifest.get("row_count", 0))
            int(manifest.get("window_start_ms", 0))
            int(manifest.get("window_end_ms", 0))
            int(manifest.get("event_time_watermark_ms", 0))
            int(manifest.get("source_checkpoint_start", 0))
            int(manifest.get("source_checkpoint_end", 0))
        except Exception as exc:
            raise RawFirstManifestInvalidError(
                f"Manifest numeric fields invalid at {manifest_path}: {exc}"
            ) from exc

        data_files = manifest.get("data_files")
        if not isinstance(data_files, list) or not data_files:
            raise RawFirstManifestInvalidError(
                f"Manifest data_files must be a non-empty list at {manifest_path}."
            )
        for item in data_files:
            token = str(item or "").strip()
            if not token:
                raise RawFirstManifestInvalidError(
                    f"Manifest data_files contains empty entry at {manifest_path}."
                )

        return manifest

    @staticmethod
    def _normalize_ohlcv_frame(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return ParquetMarketDataRepository._empty_ohlcv_frame()
        missing = [column for column in _OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise RawFirstManifestInvalidError(
                f"Materialized frame missing OHLCV columns: {missing}"
            )
        return (
            frame.select(_OHLCV_COLUMNS)
            .with_columns(
                [
                    ParquetMarketDataRepository._coerce_datetime_expr(pl.col("datetime")).alias(
                        "datetime"
                    ),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )
            .drop_nulls(subset=["datetime"])
            .sort("datetime")
        )

    def _raw_partition_path(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_date: str | date,
    ) -> Path:
        return (
            self._raw_symbol_root(
                exchange=exchange,
                symbol=symbol,
            )
            / f"date={self._partition_date_token(partition_date)}"
        )

    def _raw_wal_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_symbol_root(exchange=exchange, symbol=symbol) / "wal.bin"

    def _raw_checkpoint_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_symbol_root(exchange=exchange, symbol=symbol) / "checkpoint.json"

    def _raw_meta_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_symbol_root(exchange=exchange, symbol=symbol) / "compaction.meta.json"

    @staticmethod
    def _raw_part_paths(partition_root: Path) -> list[Path]:
        return sorted(
            path
            for path in partition_root.glob("part-*.parquet")
            if _RAW_PART_PATTERN.fullmatch(path.name) is not None
        )

    @staticmethod
    def _next_raw_part_path(partition_root: Path) -> Path:
        highest = -1
        for path in ParquetMarketDataRepository._raw_part_paths(partition_root):
            match = _RAW_PART_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            highest = max(highest, int(match.group(1)))
        next_index = highest + 1
        return partition_root / f"part-{next_index:04d}.parquet"

    @staticmethod
    def _raw_tail_key(paths: list[Path]) -> tuple[int, int] | None:
        latest: tuple[int, int] | None = None
        for path in paths:
            try:
                tail = (
                    pl.scan_parquet(str(path))
                    .select(["timestamp_ms", "agg_trade_id"])
                    .sort(["timestamp_ms", "agg_trade_id"], descending=[True, True])
                    .limit(1)
                    .collect()
                )
            except Exception:
                return None
            if tail.is_empty():
                continue
            key = (int(tail["timestamp_ms"][0]), int(tail["agg_trade_id"][0]))
            latest = key if latest is None else max(latest, key)
        return latest

    @staticmethod
    def _rename_corrupt_raw_part(path: Path) -> None:
        corrupt_path = path.with_name(
            f"{path.stem}.corrupt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}{path.suffix}"
        )
        path.replace(corrupt_path)
        ParquetMarketDataRepository._fsync_dir(path.parent)

    @staticmethod
    def _raw_checkpoint_key(payload: dict[str, Any]) -> tuple[int, int] | None:
        try:
            ts_ms = int(payload.get("last_timestamp_ms", 0) or 0)
            trade_id = int(payload.get("last_trade_id", -1) or -1)
        except Exception:
            return None
        if ts_ms <= 0:
            return None
        return (ts_ms, trade_id)

    def _materialized_date_root(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str | date,
    ) -> Path:
        token = normalize_timeframe_token(timeframe)
        return (
            self._materialized_symbol_root(exchange=exchange, symbol=symbol)
            / f"timeframe={token}"
            / f"date={self._partition_date_token(partition_date)}"
        )

    def _materialized_manifest_path(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str | date,
    ) -> Path:
        return (
            self._materialized_date_root(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                partition_date=partition_date,
            )
            / "manifest.json"
        )

    def _materialized_commit_dir(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str,
        commit_id: str,
    ) -> Path:
        return (
            self._materialized_date_root(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                partition_date=partition_date,
            )
            / f"commit={commit_id}"
        )

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
    def _ensure_ohlcv_frame(
        rows: pl.DataFrame | list[dict[str, Any]] | list[tuple[Any, ...]],
    ) -> pl.DataFrame:
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
                ParquetMarketDataRepository._coerce_datetime_expr(pl.col("datetime")).alias(
                    "datetime"
                ),
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
    def _partition_date_from_ms(timestamp_ms: int) -> str:
        dt = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _canonical_materialized_checksum(frame: pl.DataFrame) -> str:
        return ParquetMarketDataRepository.canonical_row_checksum(frame)

    @staticmethod
    def _empty_raw_aggtrades_frame() -> pl.DataFrame:
        return pl.DataFrame(
            {name: [] for name in _RAW_AGGTRADES_SCHEMA},
            schema=_RAW_AGGTRADES_SCHEMA,
        )

    @staticmethod
    def _normalize_loaded_raw_aggtrades_frame(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return ParquetMarketDataRepository._empty_raw_aggtrades_frame()
        return (
            frame.select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS))
            .with_columns(
                [
                    pl.col("agg_trade_id").cast(pl.Int64),
                    pl.col("timestamp_ms").cast(pl.Int64),
                    pl.col("price").cast(pl.Float64),
                    pl.col("quantity").cast(pl.Float64),
                    pl.col("is_buyer_maker").cast(pl.Boolean),
                ]
            )
            .filter(pl.col("timestamp_ms").is_not_null())
            .sort(["timestamp_ms", "agg_trade_id"])
            .unique(subset=["timestamp_ms", "agg_trade_id"], keep="last")
            .sort(["timestamp_ms", "agg_trade_id"])
        )

    @staticmethod
    def _ensure_raw_aggtrades_frame(
        rows: pl.DataFrame | list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> pl.DataFrame:
        frame = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows or [])
        if frame.is_empty():
            return ParquetMarketDataRepository._empty_raw_aggtrades_frame()

        missing = [
            column for column in _RAW_AGGTRADES_REQUIRED_COLUMNS if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"Raw aggTrades rows missing columns: {missing}")

        return (
            frame.select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS))
            .with_columns(
                [
                    pl.col("agg_trade_id").cast(pl.Int64),
                    pl.col("timestamp_ms").cast(pl.Int64),
                    pl.col("price").cast(pl.Float64),
                    pl.col("quantity").cast(pl.Float64),
                    pl.col("is_buyer_maker").cast(pl.Boolean),
                ]
            )
            .filter(pl.col("timestamp_ms").is_not_null())
            .sort(["timestamp_ms", "agg_trade_id"])
        )

    def read_raw_checkpoint(self, *, exchange: str, symbol: str) -> dict[str, Any]:
        checkpoint_path = self._raw_checkpoint_path(exchange=exchange, symbol=symbol)
        if not checkpoint_path.exists():
            return {}
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    def write_raw_checkpoint(
        self,
        *,
        exchange: str,
        symbol: str,
        payload: dict[str, Any],
    ) -> None:
        checkpoint_path = self._raw_checkpoint_path(exchange=exchange, symbol=symbol)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._fsync_file(checkpoint_path)
        self._fsync_dir(checkpoint_path.parent)

    def append_raw_wal_record(
        self,
        *,
        exchange: str,
        symbol: str,
        payload: dict[str, Any],
    ) -> None:
        wal_path = self._raw_wal_path(exchange=exchange, symbol=symbol)
        wal_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(dict(payload or {}), ensure_ascii=False)
        with wal_path.open("ab") as fh:
            fh.write(encoded.encode("utf-8"))
            fh.write(b"\n")
            fh.flush()
            os.fsync(fh.fileno())

    def append_raw_aggtrades(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: pl.DataFrame | list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> int:
        frame = self._ensure_raw_aggtrades_frame(rows)
        if frame.is_empty():
            return 0
        checkpoint_key = self._raw_checkpoint_key(
            self.read_raw_checkpoint(exchange=exchange, symbol=symbol)
        )

        stamped = frame.with_columns(
            (pl.col("timestamp_ms") // 1000).cast(pl.Int64).alias("_ts_sec")
        ).with_columns(
            pl.from_epoch(pl.col("_ts_sec"), time_unit="s")
            .dt.strftime("%Y-%m-%d")
            .alias("_partition_date")
        )

        total_rows = 0
        for partition_date, partition in stamped.partition_by(
            "_partition_date", as_dict=True
        ).items():
            part_key = str(
                partition_date[0] if isinstance(partition_date, tuple) else partition_date
            )
            payload = (
                partition.select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS))
                .sort(["timestamp_ms", "agg_trade_id"])
                .unique(subset=["timestamp_ms", "agg_trade_id"], keep="last")
                .sort(["timestamp_ms", "agg_trade_id"])
            )
            if payload.is_empty():
                continue
            part_path = self._raw_partition_path(
                exchange=exchange,
                symbol=symbol,
                partition_date=part_key,
            )
            part_path.mkdir(parents=True, exist_ok=True)
            partition_lock = self._acquire_raw_partition_lock(
                exchange=exchange,
                symbol=symbol,
                partition_root=part_path,
            )
            try:
                existing_paths = self._raw_part_paths(part_path)
                incoming_first_key = (
                    int(payload["timestamp_ms"][0]),
                    int(payload["agg_trade_id"][0]),
                )
                existing_tail_key = (
                    checkpoint_key
                    if checkpoint_key is not None and existing_paths
                    else (self._raw_tail_key(existing_paths) if existing_paths else None)
                )
                append_only = not existing_paths or (
                    existing_tail_key is not None and incoming_first_key > existing_tail_key
                )

                if append_only:
                    output_path = (
                        part_path / "part-0000.parquet"
                        if not existing_paths
                        else self._next_raw_part_path(part_path)
                    )
                    payload.write_parquet(output_path, compression="zstd", statistics=True)
                    self._fsync_file(output_path)
                    self._fsync_dir(output_path.parent)
                    total_rows += int(payload.height)
                    self._enforce_raw_partition_controls(
                        exchange=exchange,
                        symbol=symbol,
                        partition_root=part_path,
                    )
                    continue

                existing_frames: list[pl.DataFrame] = []
                for existing_path in existing_paths:
                    try:
                        loaded = pl.read_parquet(existing_path).select(
                            list(_RAW_AGGTRADES_REQUIRED_COLUMNS)
                        )
                    except BaseException:
                        self._rename_corrupt_raw_part(existing_path)
                        continue
                    if not loaded.is_empty():
                        existing_frames.append(loaded)

                merged = (
                    pl.concat([*existing_frames, payload], how="vertical_relaxed")
                    .sort(["timestamp_ms", "agg_trade_id"])
                    .unique(subset=["timestamp_ms", "agg_trade_id"], keep="last")
                    .sort(["timestamp_ms", "agg_trade_id"])
                )
                total_rows += int(merged.height)

                output_path = part_path / "part-0000.parquet"
                tmp_output_path = output_path.with_suffix(".tmp.parquet")
                merged.write_parquet(tmp_output_path, compression="zstd", statistics=True)
                self._fsync_file(tmp_output_path)
                tmp_output_path.replace(output_path)
                for existing_path in existing_paths:
                    if existing_path == output_path:
                        continue
                    if existing_path.exists():
                        existing_path.unlink()
                self._fsync_dir(output_path.parent)
                self._enforce_raw_partition_controls(
                    exchange=exchange,
                    symbol=symbol,
                    partition_root=part_path,
                )
            finally:
                partition_lock.release()

        return int(frame.height)

    def load_raw_aggtrades(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        start_dt = self._coerce_datetime(start_date)
        end_dt = self._coerce_datetime(end_date)
        if end_dt is not None and start_dt is not None and end_dt < start_dt:
            return self._empty_raw_aggtrades_frame()

        root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
        if not root.exists():
            return self._empty_raw_aggtrades_frame()

        candidates = sorted(
            path
            for path in root.glob("date=*/part-*.parquet")
            if _RAW_PART_PATTERN.fullmatch(path.name) is not None
        )
        if not candidates:
            return self._empty_raw_aggtrades_frame()

        start_ms = self._datetime_to_ms(start_dt)
        end_ms = self._datetime_to_ms(end_dt)
        filtered_paths: list[Path] = []
        for path in candidates:
            try:
                partition_token = path.parent.name.replace("date=", "", 1)
                partition_dt = datetime.fromisoformat(partition_token)
            except Exception:
                partition_dt = None
            if (
                start_dt is not None
                and partition_dt is not None
                and partition_dt.date() < start_dt.date()
            ):
                continue
            if (
                end_dt is not None
                and partition_dt is not None
                and partition_dt.date() > end_dt.date()
            ):
                continue
            filtered_paths.append(path)

        if not filtered_paths:
            return self._empty_raw_aggtrades_frame()

        try:
            lazy = pl.scan_parquet([str(path) for path in filtered_paths]).select(
                list(_RAW_AGGTRADES_REQUIRED_COLUMNS)
            )
            if start_dt is not None:
                lazy = lazy.filter(pl.col("timestamp_ms") >= int(start_ms or 0))
            if end_dt is not None:
                lazy = lazy.filter(pl.col("timestamp_ms") <= int(end_ms or 0))
            return self._normalize_loaded_raw_aggtrades_frame(self._collect_lazy(lazy))
        except Exception:
            pass

        frames: list[pl.DataFrame] = []
        for path in filtered_paths:
            try:
                loaded = pl.read_parquet(path).select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS))
            except Exception:
                continue
            if loaded.is_empty():
                continue
            frames.append(loaded)

        if not frames:
            return self._empty_raw_aggtrades_frame()

        merged = self._normalize_loaded_raw_aggtrades_frame(
            pl.concat(frames, how="vertical_relaxed")
        )
        if start_dt is not None:
            merged = merged.filter(pl.col("timestamp_ms") >= int(start_ms or 0))
        if end_dt is not None:
            merged = merged.filter(pl.col("timestamp_ms") <= int(end_ms or 0))
        return merged.select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS))

    def _iter_materialized_manifest_paths(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> list[Path]:
        root = (
            self._materialized_symbol_root(exchange=exchange, symbol=symbol)
            / f"timeframe={normalize_timeframe_token(timeframe)}"
        )
        if not root.exists():
            return []
        return sorted(root.glob("date=*/manifest.json"))

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}

    @staticmethod
    def _normalize_manifest_data_files(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        out: list[str] = []
        for item in value:
            token = str(item or "").strip()
            if token:
                out.append(token)
        return out

    def _validate_manifest_payload(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
        exchange: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        staleness_threshold_seconds: int | None = None,
    ) -> None:
        missing = [name for name in _MATERIALIZED_REQUIRED_MANIFEST_FIELDS if name not in manifest]
        if missing:
            raise RawFirstManifestInvalidError(
                f"Manifest missing required fields {missing}: {manifest_path}"
            )
        if str(manifest.get("status", "")).strip().lower() != "committed":
            raise RawFirstManifestInvalidError(
                f"Manifest status must be committed: {manifest_path}"
            )

        data_files = self._normalize_manifest_data_files(manifest.get("data_files"))
        if not data_files:
            raise RawFirstManifestInvalidError(f"Manifest data_files is empty: {manifest_path}")

        if exchange is not None:
            expected_exchange = self._normalize_exchange(exchange)
            partition = str(manifest.get("partition", "")).strip()
            if partition and expected_exchange not in partition:
                raise RawFirstManifestInvalidError(
                    f"Manifest exchange mismatch expected={expected_exchange}: {manifest_path}"
                )

        if symbol is not None:
            expected_symbol = self._normalize_symbol_token(symbol)
            actual_symbol = self._normalize_symbol_token(str(manifest.get("symbol", "")))
            if expected_symbol != actual_symbol:
                raise RawFirstManifestInvalidError(
                    f"Manifest symbol mismatch expected={symbol}: {manifest_path}"
                )

        if timeframe is not None:
            expected_tf = normalize_timeframe_token(timeframe)
            actual_tf = normalize_timeframe_token(str(manifest.get("timeframe", "")))
            if expected_tf != actual_tf:
                raise RawFirstManifestInvalidError(
                    f"Manifest timeframe mismatch expected={expected_tf}: {manifest_path}"
                )

        try:
            row_count = int(manifest.get("row_count", 0))
        except Exception as exc:
            raise RawFirstManifestInvalidError(
                f"Manifest row_count must be int: {manifest_path}"
            ) from exc
        if row_count < 0:
            raise RawFirstManifestInvalidError(f"Manifest row_count must be >= 0: {manifest_path}")

        if staleness_threshold_seconds is None:
            return
        try:
            watermark_ms = int(manifest.get("event_time_watermark_ms", 0))
        except Exception as exc:
            raise RawFirstManifestInvalidError(
                f"Manifest event_time_watermark_ms invalid: {manifest_path}"
            ) from exc
        if watermark_ms <= 0:
            raise RawFirstManifestInvalidError(
                f"Manifest watermark must be positive: {manifest_path}"
            )
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        lag_ms = max(0, now_ms - watermark_ms)
        if lag_ms > int(staleness_threshold_seconds) * 1000:
            raise RawFirstStaleWindowError(
                f"Manifest stale window detected lag_ms={lag_ms} threshold_s={staleness_threshold_seconds} "
                f"path={manifest_path}",
                symbol=str(manifest.get("symbol", "")),
                timeframe=str(manifest.get("timeframe", "")),
                lag_ms=int(lag_ms),
                commit_id=str(manifest.get("commit_id", "")),
            )

    def _load_committed_manifest_frame(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
    ) -> pl.DataFrame:
        data_files = self._normalize_manifest_data_files(manifest.get("data_files"))
        date_root = manifest_path.parent
        frames: list[pl.DataFrame] = []
        for file_token in data_files:
            data_path = Path(file_token)
            if not data_path.is_absolute():
                data_path = date_root / file_token
            if not data_path.exists():
                raise RawFirstManifestInvalidError(
                    f"Manifest referenced data file does not exist: {data_path}"
                )
            loaded = pl.read_parquet(data_path)
            missing = [
                name for name in _MATERIALIZED_REQUIRED_COLUMNS if name not in loaded.columns
            ]
            if missing:
                raise RawFirstManifestInvalidError(
                    f"Manifest data file missing columns {missing}: {data_path}"
                )
            frames.append(loaded.select(list(_MATERIALIZED_REQUIRED_COLUMNS)))

        if not frames:
            return self._empty_ohlcv_frame()

        merged = (
            pl.concat(frames, how="vertical_relaxed")
            .with_columns(
                [
                    self._coerce_datetime_expr(pl.col("datetime")).alias("datetime"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )
            .drop_nulls(subset=["datetime"])
            .sort("datetime")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
        )

        row_count = int(manifest.get("row_count", 0))
        if int(merged.height) != row_count:
            raise RawFirstManifestInvalidError(
                f"Manifest row_count mismatch expected={row_count} actual={merged.height} path={manifest_path}"
            )

        expected_checksum = str(manifest.get("canonical_row_checksum", "")).strip()
        actual_checksum = self._canonical_materialized_checksum(merged)
        if expected_checksum != actual_checksum:
            raise RawFirstManifestInvalidError(
                f"Manifest checksum mismatch expected={expected_checksum} actual={actual_checksum} "
                f"path={manifest_path}"
            )
        return merged

    def write_materialized_manifest(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str,
        payload: dict[str, Any],
    ) -> Path:
        manifest_path = self._materialized_manifest_path(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            partition_date=partition_date,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(dict(payload or {}), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._fsync_file(tmp_path)
        tmp_path.replace(manifest_path)
        self._fsync_dir(manifest_path.parent)
        return manifest_path

    def read_latest_materialized_manifest(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any] | None:
        token = normalize_timeframe_token(timeframe)
        manifests = self._iter_materialized_manifest_paths(
            exchange=exchange,
            symbol=symbol,
            timeframe=token,
        )
        latest_payload: dict[str, Any] | None = None
        latest_key: tuple[int, int, str, str] | None = None

        for manifest_path in manifests:
            try:
                payload = self._read_json_file(manifest_path)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            try:
                checkpoint_end = int(payload.get("source_checkpoint_end", 0) or 0)
            except Exception:
                checkpoint_end = 0
            try:
                watermark_ms = int(payload.get("event_time_watermark_ms", 0) or 0)
            except Exception:
                watermark_ms = 0
            commit_id = str(payload.get("commit_id", "") or "")
            key = (
                int(checkpoint_end),
                int(watermark_ms),
                commit_id,
                str(manifest_path),
            )
            if latest_key is None or key > latest_key:
                latest_key = key
                latest_payload = dict(payload)

        return latest_payload

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
        return (
            max(0, int(wal_max_bytes)),
            bool(compact_on_threshold),
            max(0, int(compaction_interval)),
        )

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
        wal_max_bytes, compact_on_threshold, compaction_interval_seconds = (
            self._resolve_wal_controls()
        )
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

    def _read_raw_meta(self, *, exchange: str, symbol: str) -> dict[str, Any]:
        path = self._raw_meta_path(exchange=exchange, symbol=symbol)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_raw_meta(self, *, exchange: str, symbol: str, payload: dict[str, Any]) -> None:
        path = self._raw_meta_path(exchange=exchange, symbol=symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._fsync_file(tmp)
        tmp.replace(path)
        self._fsync_dir(path.parent)

    def _resolve_raw_partition_lock_controls(self) -> tuple[float, float]:
        timeout_raw = str(
            os.getenv("LQ_RAW_PARTITION_LOCK_TIMEOUT_SECONDS", "")
            or os.getenv("LQ__STORAGE__RAW_PARTITION_LOCK_TIMEOUT_SECONDS", "")
        ).strip()
        poll_raw = str(
            os.getenv("LQ_RAW_PARTITION_LOCK_POLL_SECONDS", "")
            or os.getenv("LQ__STORAGE__RAW_PARTITION_LOCK_POLL_SECONDS", "")
        ).strip()
        try:
            timeout_seconds = max(0.1, float(timeout_raw)) if timeout_raw else 10.0
        except ValueError:
            timeout_seconds = 10.0
        try:
            poll_seconds = max(0.01, float(poll_raw)) if poll_raw else 0.05
        except ValueError:
            poll_seconds = 0.05
        return float(timeout_seconds), float(poll_seconds)

    def _acquire_raw_partition_lock(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_root: Path,
    ) -> HeavyRunLock:
        timeout_seconds, poll_seconds = self._resolve_raw_partition_lock_controls()
        deadline = time.monotonic() + float(timeout_seconds)
        active_payload: dict[str, Any] = {}
        lock_path = partition_root / ".raw-partition.lock"
        while True:
            try:
                return HeavyRunLock.acquire(
                    lock_path=lock_path,
                    label="raw_partition",
                    metadata={
                        "exchange": self._normalize_exchange(exchange),
                        "symbol": self._normalize_symbol_token(symbol),
                        "partition": partition_root.name,
                    },
                )
            except HeavyRunActiveError as exc:
                active_payload = dict(exc.active_payload)
                if time.monotonic() >= deadline:
                    raise RawPartitionBusyError(
                        "Raw partition is busy; timed out waiting for writer/compaction lease "
                        f"partition={partition_root} active={active_payload}"
                    ) from exc
                time.sleep(float(poll_seconds))

    def _resolve_raw_part_controls(self) -> tuple[int, bool]:
        raw_max_parts = self._env_int(
            names=["LQ_RAW_PARTITION_MAX_PARTS", "LQ__STORAGE__RAW_PARTITION_MAX_PARTS"],
            default=8,
        )
        compact_on_threshold = self._env_bool(
            names=["LQ_RAW_COMPACT_ON_THRESHOLD", "LQ__STORAGE__RAW_COMPACT_ON_THRESHOLD"],
            default=True,
        )
        return max(0, int(raw_max_parts)), bool(compact_on_threshold)

    def _compact_raw_partition(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_root: Path,
    ) -> int:
        part_paths = self._raw_part_paths(partition_root)
        if not part_paths:
            return 0

        frames: list[pl.DataFrame] = []
        for path in part_paths:
            try:
                loaded = pl.read_parquet(path).select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS))
            except Exception:
                self._rename_corrupt_raw_part(path)
                continue
            if not loaded.is_empty():
                frames.append(loaded)

        if not frames:
            return 0

        merged = self._normalize_loaded_raw_aggtrades_frame(
            pl.concat(frames, how="vertical_relaxed")
        )
        output_path = partition_root / "part-0000.parquet"
        tmp_output_path = output_path.with_suffix(".tmp.parquet")
        merged.write_parquet(tmp_output_path, compression="zstd", statistics=True)
        self._fsync_file(tmp_output_path)
        tmp_output_path.replace(output_path)
        for path in part_paths:
            if path == output_path:
                continue
            if path.exists():
                path.unlink()
        self._fsync_dir(output_path.parent)
        return len([path for path in part_paths if path.exists()])

    def _enforce_raw_partition_controls(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_root: Path,
    ) -> None:
        max_parts, compact_on_threshold = self._resolve_raw_part_controls()
        if max_parts <= 0:
            return

        part_paths = self._raw_part_paths(partition_root)
        if len(part_paths) <= max_parts:
            return

        now = datetime.now(tz=UTC)
        meta = self._read_raw_meta(exchange=exchange, symbol=symbol)
        meta["raw_compaction_required"] = True
        meta["last_raw_over_limit_detected_at"] = now.isoformat()
        meta["last_raw_over_limit_partition"] = partition_root.name
        meta["last_raw_part_count"] = len(part_paths)

        if not compact_on_threshold:
            last_warning = self._parse_iso_utc(meta.get("last_raw_over_limit_warning_at"))
            warn_elapsed = None if last_warning is None else (now - last_warning).total_seconds()
            if warn_elapsed is None or warn_elapsed >= 60.0:
                print(
                    f"[WARN] Raw partition {partition_root} has {len(part_paths)} parts (limit {max_parts}); "
                    "raw_compact_on_threshold=false, manual compaction required."
                )
                meta["last_raw_over_limit_warning_at"] = now.isoformat()
            self._write_raw_meta(exchange=exchange, symbol=symbol, payload=meta)
            return

        self._compact_raw_partition(exchange=exchange, symbol=symbol, partition_root=partition_root)
        meta = self._read_raw_meta(exchange=exchange, symbol=symbol)
        meta["raw_compaction_required"] = False
        meta["last_raw_compaction_resolved_at"] = now.isoformat()
        meta["last_raw_compaction_partition"] = partition_root.name
        self._write_raw_meta(exchange=exchange, symbol=symbol, payload=meta)

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

        # Read paths only need the valid prefix. Avoid eagerly repairing/truncating the
        # WAL on every load because large healthy WALs can spend most of the batch budget
        # re-validating bytes that will be decoded again immediately afterwards.
        wal = BinaryWAL(wal_path, auto_repair=False)
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
                wal.with_columns(pl.lit(1).cast(pl.Int8).alias("_source_priority")).select(
                    merge_cols
                )
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

        native_appended = append_ohlcv_frame_native(
            wal.path,
            frame,
            fsync_after_write=fsync_n <= 1,
        )
        if native_appended is not None:
            appended = int(native_appended)
        else:
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
        source = (
            merged_1s.lazy()
            .with_columns(pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"))
            .with_columns(((pl.col("timestamp_ms") // tf_ms) * tf_ms).alias("bucket_ms"))
        )

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
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        progress_context: Mapping[str, Any] | None = None,
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
        chunk_ranges = list(
            self._iter_chunks(
                start=start_dt,
                end=end_dt,
                chunk_days=max(1, int(chunk_days)),
            )
        )
        progress_base = dict(progress_context or {})
        progress_base.setdefault("symbol", str(symbol))
        progress_base.setdefault("timeframe", str(timeframe_token))
        chunk_count = len(chunk_ranges)
        for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            chunk_started_at = time.perf_counter()
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
            trimmed = (
                chunk.filter(pl.col("datetime") >= chunk_start) if not chunk.is_empty() else chunk
            )
            trimmed_row_count = int(trimmed.height) if trimmed is not None else 0
            if progress_callback is not None:
                progress_callback(
                    "resource_bundle_symbol_window_loaded",
                    {
                        **progress_base,
                        "unit_kind": "chunk",
                        "unit_index": chunk_index,
                        "unit_count": chunk_count,
                        "window_start": chunk_start.isoformat(),
                        "window_end": chunk_end.isoformat(),
                        "query_start": query_start.isoformat(),
                        "row_count": trimmed_row_count,
                        "elapsed_seconds": round(
                            max(0.0, time.perf_counter() - chunk_started_at),
                            6,
                        ),
                    },
                )
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

    def _iter_committed_manifest_paths(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[Path]:
        root = (
            self._materialized_symbol_root(exchange=exchange, symbol=symbol)
            / f"timeframe={normalize_timeframe_token(timeframe)}"
        )
        if not root.exists():
            return []

        manifests = sorted(root.glob("date=*/manifest.json"))
        if start_date is None and end_date is None:
            return manifests

        if start_date is None:
            start_date = datetime(1970, 1, 1)
        if end_date is None:
            end_date = datetime(3000, 1, 1)
        start_day = start_date.date()
        end_day = end_date.date()

        filtered: list[Path] = []
        for path in manifests:
            token = str(path.parent.name).strip()
            if not token.startswith("date="):
                continue
            try:
                part_date = date.fromisoformat(token.split("=", 1)[1])
            except Exception:
                continue
            if start_day <= part_date <= end_day:
                filtered.append(path)
        return filtered

    def _resolve_manifest_data_paths(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
    ) -> list[Path]:
        out: list[Path] = []
        for item in list(manifest.get("data_files") or []):
            raw = str(item or "").strip()
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (manifest_path.parent / candidate).resolve()
            if not candidate.exists():
                raise RawFirstDataMissingError(
                    f"Manifest referenced data file missing: {candidate} (manifest={manifest_path})"
                )
            out.append(candidate)
        if not out:
            raise RawFirstDataMissingError(f"Manifest has no readable data files: {manifest_path}")
        return out

    def _load_manifest_json(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RawFirstManifestInvalidError(
                f"Failed to parse manifest JSON at {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RawFirstManifestInvalidError(f"Manifest payload must be an object at {path}.")
        return payload

    def load_committed_ohlcv_chunked(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
        chunk_days: int = 7,
        warmup_bars: int = 0,
        staleness_threshold_seconds: int | None = None,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        progress_context: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        token = normalize_timeframe_token(timeframe)
        start_dt = self._coerce_datetime(start_date)
        end_dt = self._coerce_datetime(end_date)
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            return self._empty_ohlcv_frame()

        query_start = start_dt
        if start_dt is not None and int(warmup_bars) > 0:
            tf_ms = int(timeframe_to_milliseconds(token))
            query_start = start_dt - timedelta(milliseconds=max(0, int(warmup_bars)) * tf_ms)

        manifests = self._iter_materialized_manifest_paths(
            exchange=exchange,
            symbol=symbol,
            timeframe=token,
        )
        if query_start is not None or end_dt is not None:
            bounded: list[Path] = []
            lower = query_start.date() if query_start is not None else date(1970, 1, 1)
            upper = end_dt.date() if end_dt is not None else date(3000, 1, 1)
            for path in manifests:
                parent = str(path.parent.name)
                if not parent.startswith("date="):
                    continue
                try:
                    part_date = date.fromisoformat(parent.split("=", 1)[1])
                except Exception:
                    continue
                if lower <= part_date <= upper:
                    bounded.append(path)
            manifests = bounded

        if not manifests:
            if token != "1s":
                source_1s = self.load_committed_ohlcv_chunked(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe="1s",
                    start_date=query_start,
                    end_date=end_dt,
                    chunk_days=chunk_days,
                    warmup_bars=warmup_bars,
                    staleness_threshold_seconds=staleness_threshold_seconds,
                )
                if source_1s.is_empty():
                    raise RawFirstDataMissingError(
                        f"No committed 1s rows available to rebuild {exchange}:{symbol}:{token}."
                    )
                latest_1s_ms = int(
                    source_1s.select(pl.col("datetime").dt.epoch("ms").max().alias("ts")).item()
                    or 0
                )
                rebuilt = resample_1s_frame(
                    source_1s,
                    timeframe=token,
                    complete_through_ms=int(latest_1s_ms) + 999,
                ).sort("datetime")
                if end_dt is not None:
                    rebuilt = rebuilt.filter(pl.col("datetime") <= end_dt)
                if start_dt is not None and int(warmup_bars) <= 0:
                    rebuilt = rebuilt.filter(pl.col("datetime") >= start_dt)
                if rebuilt.is_empty():
                    raise RawFirstDataMissingError(
                        f"Committed 1s rows exist but could not rebuild {exchange}:{symbol}:{token}."
                    )
                return rebuilt
            raise RawFirstDataMissingError(
                f"No committed manifests found for {exchange}:{symbol}:{token}."
            )

        frames: list[pl.DataFrame] = []
        newest_watermark_ms: int | None = None
        newest_commit_id: str | None = None
        progress_base = dict(progress_context or {})
        progress_base.setdefault("symbol", str(symbol))
        progress_base.setdefault("timeframe", str(token))
        manifest_count = len(manifests)
        for manifest_index, manifest_path in enumerate(manifests, start=1):
            manifest_started_at = time.perf_counter()
            manifest = self._read_json_file(manifest_path)
            self._validate_manifest_payload(
                manifest=manifest,
                manifest_path=manifest_path,
                exchange=exchange,
                symbol=symbol,
                timeframe=token,
                staleness_threshold_seconds=staleness_threshold_seconds,
            )
            frame = self._load_committed_manifest_frame(
                manifest=manifest,
                manifest_path=manifest_path,
            )
            if progress_callback is not None:
                progress_callback(
                    "resource_bundle_symbol_window_loaded",
                    {
                        **progress_base,
                        "unit_kind": "manifest",
                        "unit_index": manifest_index,
                        "unit_count": manifest_count,
                        "partition": str(manifest_path.parent.name),
                        "row_count": int(frame.height),
                        "elapsed_seconds": round(
                            max(0.0, time.perf_counter() - manifest_started_at),
                            6,
                        ),
                    },
                )
            if not frame.is_empty():
                frames.append(frame)
            try:
                watermark_ms = int(manifest.get("event_time_watermark_ms", 0))
            except Exception:
                watermark_ms = 0
            if watermark_ms > 0 and (
                newest_watermark_ms is None or watermark_ms >= newest_watermark_ms
            ):
                newest_watermark_ms = int(watermark_ms)
                newest_commit_id = str(manifest.get("commit_id", "") or "")

        if not frames:
            raise RawFirstDataMissingError(
                f"Committed manifests found but no rows for {exchange}:{symbol}:{token}."
            )

        merged = (
            pl.concat(frames, how="vertical_relaxed")
            .sort("datetime")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
        )
        if query_start is not None:
            merged = merged.filter(pl.col("datetime") >= query_start)
        if end_dt is not None:
            merged = merged.filter(pl.col("datetime") <= end_dt)
        if start_dt is not None and int(warmup_bars) <= 0:
            merged = merged.filter(pl.col("datetime") >= start_dt)

        if merged.is_empty():
            raise RawFirstDataMissingError(
                f"No committed OHLCV rows in range for {exchange}:{symbol}:{token}."
            )

        if (
            staleness_threshold_seconds is not None
            and int(staleness_threshold_seconds) > 0
            and newest_watermark_ms is not None
        ):
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            lag_ms = max(0, int(now_ms - newest_watermark_ms))
            if lag_ms > int(staleness_threshold_seconds) * 1000:
                raise RawFirstStaleWindowError(
                    "Committed window stale for "
                    f"{exchange}:{symbol}:{token}: lag_ms={lag_ms} threshold_ms={int(staleness_threshold_seconds) * 1000}.",
                    symbol=str(symbol),
                    timeframe=str(token),
                    lag_ms=int(lag_ms),
                    commit_id=newest_commit_id,
                )

        return merged.select(_OHLCV_COLUMNS)

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
            monthly_path = self._monthly_path(
                exchange=exchange, symbol=symbol, month_token=month_token
            )
            monthly_path.parent.mkdir(parents=True, exist_ok=True)

            existing = (
                pl.read_parquet(monthly_path)
                if monthly_path.exists()
                else self._empty_ohlcv_frame()
            )
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
            # Metadata scans are read-only; iterate the valid prefix without forcing a
            # full repair pass on every open.
            wal = BinaryWAL(wal_path, auto_repair=False)
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
        or any(root.glob("market_data_raw_aggtrades/*/*/date=*/part-*.parquet"))
        or any(root.glob("market_data_materialized/*/*/timeframe=*/date=*/manifest.json"))
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
    data_mode: str = "legacy",
    staleness_threshold_seconds: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, pl.DataFrame]:
    """Compatibility entrypoint retained for legacy import sites."""
    repo = ParquetMarketDataRepository(root_path)
    resolved_mode = normalize_data_mode(data_mode, default="legacy")
    out: dict[str, pl.DataFrame] = {}
    missing_symbols: list[str] = []

    symbols = list(symbol_list or [])
    symbol_count = len(symbols)
    for symbol_index, symbol in enumerate(symbols, start=1):
        progress_base = {
            "symbol": str(symbol),
            "symbol_index": symbol_index,
            "symbol_count": symbol_count,
            "timeframe": str(timeframe),
            "data_mode": str(resolved_mode),
        }
        if progress_callback is not None:
            progress_callback(
                "resource_bundle_symbol_fetch_started",
                progress_base,
            )
        symbol_started_at = time.perf_counter()
        if resolved_mode == "raw-first":
            try:
                frame = repo.load_committed_ohlcv_chunked(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    chunk_days=chunk_days,
                    warmup_bars=warmup_bars,
                    staleness_threshold_seconds=staleness_threshold_seconds,
                    progress_callback=progress_callback,
                    progress_context=progress_base,
                )
            except RawFirstDataMissingError:
                from lumina_quant.services.materialize_from_raw import (
                    materialize_raw_aggtrades_bundle,
                )

                materialize_raw_aggtrades_bundle(
                    root_path=root_path,
                    exchange=exchange,
                    symbol=str(symbol),
                    timeframes=[str(timeframe)],
                    start_date=start_date,
                    end_date=end_date,
                    producer="load_data_dict_from_parquet",
                    require_complete=True,
                )
                frame = repo.load_committed_ohlcv_chunked(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    chunk_days=chunk_days,
                    warmup_bars=warmup_bars,
                    staleness_threshold_seconds=staleness_threshold_seconds,
                    progress_callback=progress_callback,
                    progress_context=progress_base,
                )
        else:
            frame = repo.load_ohlcv_chunked(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                chunk_days=chunk_days,
                warmup_bars=warmup_bars,
                progress_callback=progress_callback,
                progress_context=progress_base,
            )

        row_count = int(frame.height) if frame is not None else 0
        if progress_callback is not None:
            progress_callback(
                "resource_bundle_symbol_fetch_completed",
                {
                    **progress_base,
                    "row_count": row_count,
                    "was_missing": bool(frame is None or frame.is_empty()),
                    "elapsed_seconds": round(
                        max(0.0, time.perf_counter() - symbol_started_at),
                        6,
                    ),
                },
            )
        if frame is None or frame.is_empty():
            missing_symbols.append(str(symbol))
            continue
        out[str(symbol)] = frame

    if resolved_mode == "raw-first" and missing_symbols:
        raise RawFirstDataMissingError(
            "Raw-first committed data missing for symbols: " + ", ".join(missing_symbols)
        )
    return out
