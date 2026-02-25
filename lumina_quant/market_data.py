"""Parquet-backed market-data helpers for local runtime workflows."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

MARKET_OHLCV_TABLE = "market_ohlcv"
MARKET_OHLCV_1S_TABLE = "market_ohlcv_1s"
FUTURES_FEATURE_POINTS_TABLE = "futures_feature_points"
DEFAULT_MARKET_DATA_DB_PATH = "data/market_parquet"
KNOWN_QUOTES = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
TIMEFRAME_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
    "M": 2_592_000_000,
}
EMPTY_OHLCV_SCHEMA = {
    "datetime": pl.Datetime(time_unit="ms"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
_FEATURE_COLUMNS = (
    "funding_rate",
    "funding_mark_price",
    "mark_price",
    "index_price",
    "open_interest",
    "liquidation_long_qty",
    "liquidation_short_qty",
    "liquidation_long_notional",
    "liquidation_short_notional",
)


class _QueryResult:
    """Minimal cursor-like wrapper used by compatibility connection objects."""

    def __init__(self, rows: list[Any]):
        self._rows = rows

    def fetchone(self) -> Any:
        if not self._rows:
            return None
        return self._rows[0]

    def fetchall(self) -> list[Any]:
        return list(self._rows)


@dataclass(slots=True)
class ParquetMarketDataConnection:
    """Compatibility connection object retained for legacy call sites."""

    db_path: str

    def execute(self, query: str, params: Any = None) -> _QueryResult:
        _ = params
        token = str(query or "").strip().lower()
        if token.startswith("select 1"):
            return _QueryResult([(1,)])
        if token.startswith("select") and "from futures_feature_points" in token:
            root = Path(self.db_path)
            pattern = str(
                root / "feature_points" / "exchange=*" / "symbol=*" / "date=*" / "*.parquet"
            )
            try:
                frame = pl.scan_parquet(pattern).sort("timestamp_ms").collect()
            except Exception:
                return _QueryResult([])
            if frame.is_empty():
                return _QueryResult([])
            try:
                from_idx = token.index("from")
                selected = str(query)[len("select") : from_idx]
            except ValueError:
                selected = "*"
            selected_cols = [item.strip() for item in selected.split(",") if item.strip()]
            if not selected_cols or selected_cols == ["*"]:
                selected_cols = list(frame.columns)
            rows: list[tuple[Any, ...]] = []
            for row in frame.iter_rows(named=True):
                rows.append(tuple(row.get(col) for col in selected_cols))
            return _QueryResult(rows)
        raise RuntimeError(
            "Direct SQL execution is not supported for parquet storage. "
            "Use market_data helper functions instead."
        )

    def close(self) -> None:
        return


def _resolve_market_root_path(db_path: str | os.PathLike[str] | None = None) -> Path:
    configured = str(
        db_path
        or os.getenv("LQ__STORAGE__MARKET_DATA_PARQUET_PATH")
        or os.getenv("LQ_MARKET_PARQUET_PATH")
        or DEFAULT_MARKET_DATA_DB_PATH
    ).strip()
    root = Path(configured).expanduser()
    if root.suffix and root.suffix.lower() != ".parquet":
        root = root.parent / "market_parquet"
    return root


def _parquet_repo(root_path: Path):
    from lumina_quant.parquet_market_data import ParquetMarketDataRepository

    return ParquetMarketDataRepository(root_path)


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
        schema=EMPTY_OHLCV_SCHEMA,
    )


def normalize_storage_backend(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "parquet", "local", "parquet-postgres"}:
        return "parquet-postgres"
    return "parquet-postgres"


def _resolve_storage_backend(backend: str | None = None) -> str:
    explicit = str(backend or "").strip()
    if explicit:
        return normalize_storage_backend(explicit)
    env_backend = os.getenv("LQ__STORAGE__BACKEND") or os.getenv("LQ_STORAGE_BACKEND")
    return normalize_storage_backend(env_backend)


def _build_market_data_repository(
    db_path: str,
    *,
    backend: str | None = None,
    **legacy: Any,
) -> Any:
    _ = _resolve_storage_backend(backend)
    _ = legacy
    return MarketDataRepository(db_path)


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format into BASE/QUOTE uppercase."""
    raw = str(symbol).strip().upper().replace("_", "/").replace("-", "/")
    if "/" in raw:
        base, quote = raw.split("/", 1)
        return f"{base}/{quote}"

    for quote in KNOWN_QUOTES:
        if raw.endswith(quote) and len(raw) > len(quote):
            base = raw[: -len(quote)]
            return f"{base}/{quote}"
    return raw


def symbol_csv_filename(symbol: str) -> str:
    """Return canonical CSV filename for a symbol."""
    return f"{normalize_symbol(symbol).replace('/', '')}.csv"


def symbol_csv_candidates(csv_dir: str, symbol: str) -> list[str]:
    """Return common CSV path candidates for a symbol."""
    normalized = normalize_symbol(symbol)
    compact = normalized.replace("/", "")
    return [
        os.path.join(csv_dir, f"{normalized}.csv"),
        os.path.join(csv_dir, f"{compact}.csv"),
        os.path.join(csv_dir, f"{normalized.replace('/', '_')}.csv"),
        os.path.join(csv_dir, f"{normalized.replace('/', '-')}.csv"),
    ]


def resolve_symbol_csv_path(csv_dir: str, symbol: str) -> str:
    """Resolve the first existing symbol CSV path, fallback to compact name."""
    candidates = symbol_csv_candidates(csv_dir, symbol)
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[1]


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert timeframe token like 1m/1h/1d into milliseconds."""
    token = normalize_timeframe_token(timeframe)
    if len(token) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    unit = token[-1]
    value = int(token[:-1])
    if value <= 0:
        raise ValueError(f"Invalid timeframe value: {timeframe}")
    unit_ms = TIMEFRAME_UNIT_MS.get(unit)
    if unit_ms is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return value * unit_ms


_TIMEFRAME_PATTERN = re.compile(r"^([1-9][0-9]*)([smhdwM])$")


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


def connect_market_data_db(db_path: str) -> ParquetMarketDataConnection:
    """Open a compatibility connection for parquet-backed market storage."""
    root = _resolve_market_root_path(db_path)
    root.mkdir(parents=True, exist_ok=True)
    return ParquetMarketDataConnection(str(root))


def resolve_1s_db_path(db_path: str) -> str:
    """Resolve market parquet root path for 1-second bars."""
    explicit = str(os.getenv("LQ_1S_DB_PATH", "")).strip()
    if explicit:
        return str(_resolve_market_root_path(explicit))
    return str(_resolve_market_root_path(db_path))


def connect_market_data_1s_db(db_path: str) -> ParquetMarketDataConnection:
    """Open a compatibility connection for 1-second parquet bars."""
    root = Path(resolve_1s_db_path(db_path))
    root.mkdir(parents=True, exist_ok=True)
    return ParquetMarketDataConnection(str(root))


def ensure_market_ohlcv_schema(conn: ParquetMarketDataConnection) -> None:
    _ = conn


def ensure_futures_feature_points_schema(conn: ParquetMarketDataConnection) -> None:
    _ = conn


def ensure_market_ohlcv_1s_schema(conn: ParquetMarketDataConnection) -> None:
    _ = conn


def _utc_iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).isoformat()


def _coerce_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = int(value)
        if abs(ts) < 10_000_000:
            return ts
        if abs(ts) < 100_000_000_000:
            return ts * 1000
        return ts
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    token = str(value).strip()
    if not token:
        return None
    if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
        return _coerce_timestamp_ms(int(token))
    try:
        dt = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _timestamp_ms_to_datetime(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).replace(tzinfo=None)


def _datetime_to_epoch_ms(value: datetime | None) -> int | None:
    if value is None:
        return None
    dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _normalize_exchange(exchange: str) -> str:
    return str(exchange).strip().lower()


def _normalize_symbol_token(symbol: str) -> str:
    return normalize_symbol(symbol)


def _series_path(root: Path, *, exchange: str, symbol: str, timeframe: str) -> Path:
    tf = normalize_timeframe_token(timeframe)
    compact_symbol = _normalize_symbol_token(symbol).replace("/", "")
    return (
        root
        / f"exchange={_normalize_exchange(exchange)}"
        / f"symbol={compact_symbol}"
        / f"timeframe={tf}"
    )


def _date_partition_path(
    root: Path,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    partition_date: datetime.date,
) -> Path:
    return _series_path(root, exchange=exchange, symbol=symbol, timeframe=timeframe) / (
        f"date={partition_date.isoformat()}"
    )


def _load_direct_ohlcv(
    root: Path,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
) -> pl.DataFrame:
    base = _series_path(root, exchange=exchange, symbol=symbol, timeframe=timeframe)
    pattern = str(base / "date=*" / "*.parquet")
    try:
        lazy = pl.scan_parquet(pattern)
    except Exception:
        return _empty_ohlcv_frame()

    start_ms = _coerce_timestamp_ms(start_date)
    end_ms = _coerce_timestamp_ms(end_date)
    if start_ms is not None:
        lazy = lazy.filter(pl.col("datetime") >= _timestamp_ms_to_datetime(start_ms))
    if end_ms is not None:
        lazy = lazy.filter(pl.col("datetime") <= _timestamp_ms_to_datetime(end_ms))

    try:
        data = lazy.select(["datetime", "open", "high", "low", "close", "volume"]).collect()
    except Exception:
        return _empty_ohlcv_frame()

    if data.is_empty():
        return _empty_ohlcv_frame()
    return data.sort("datetime").unique(subset=["datetime"], keep="last").sort("datetime")


def _ensure_ohlcv_frame(rows: Any) -> pl.DataFrame:
    if isinstance(rows, pl.DataFrame):
        frame = rows
        if "timestamp_ms" in frame.columns and "datetime" not in frame.columns:
            frame = frame.with_columns(
                pl.from_epoch(pl.col("timestamp_ms").cast(pl.Int64), time_unit="ms").alias(
                    "datetime"
                )
            )
    else:
        records: list[dict[str, Any]] = []
        for row in rows or []:
            if isinstance(row, dict):
                ts = _coerce_timestamp_ms(row.get("timestamp_ms", row.get("datetime")))
                if ts is None:
                    continue
                records.append(
                    {
                        "datetime": _timestamp_ms_to_datetime(ts),
                        "open": row.get("open"),
                        "high": row.get("high"),
                        "low": row.get("low"),
                        "close": row.get("close"),
                        "volume": row.get("volume"),
                    }
                )
                continue

            values = list(row)
            if len(values) < 6:
                continue
            ts = _coerce_timestamp_ms(values[0])
            if ts is None:
                continue
            records.append(
                {
                    "datetime": _timestamp_ms_to_datetime(ts),
                    "open": values[1],
                    "high": values[2],
                    "low": values[3],
                    "close": values[4],
                    "volume": values[5],
                }
            )
        frame = pl.DataFrame(records)

    if frame.is_empty():
        return _empty_ohlcv_frame()

    required = ["datetime", "open", "high", "low", "close", "volume"]
    available = [name for name in required if name in frame.columns]
    if len(available) != len(required):
        return _empty_ohlcv_frame()

    return (
        frame.select(required)
        .with_columns(
            [
                pl.col("datetime").cast(pl.Datetime(time_unit="ms"), strict=False).alias("datetime"),
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


def _upsert_ohlcv_frame(
    root: Path,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    frame: pl.DataFrame,
) -> int:
    if frame.is_empty():
        return 0

    with_dates = frame.with_columns(pl.col("datetime").dt.date().alias("partition_date"))
    partitions = with_dates.partition_by("partition_date", maintain_order=True)

    upserted = 0
    for partition in partitions:
        if partition.is_empty():
            continue
        partition_date = partition["partition_date"][0]
        if partition_date is None:
            continue
        date_path = _date_partition_path(
            root,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            partition_date=partition_date,
        )
        date_path.mkdir(parents=True, exist_ok=True)

        incoming = partition.drop("partition_date")
        existing_files = sorted(date_path.glob("*.parquet"))
        frames = [incoming]
        for file_path in existing_files:
            try:
                frames.append(pl.read_parquet(file_path))
            except Exception:
                continue

        merged = (
            pl.concat(frames, how="vertical")
            .sort("datetime")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
        )

        output_path = date_path / f"compact-{partition_date.isoformat()}.parquet"
        tmp_path = output_path.with_suffix(".tmp.parquet")
        merged.write_parquet(tmp_path, compression="zstd", statistics=True)
        tmp_path.replace(output_path)

        for file_path in existing_files:
            if file_path == output_path:
                continue
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass

        upserted += int(incoming.height)

    return upserted


def _load_feature_points(
    root: Path,
    *,
    exchange: str,
    symbol: str,
    start_date: Any = None,
    end_date: Any = None,
) -> pl.DataFrame:
    compact_symbol = normalize_symbol(symbol).replace("/", "")
    base = (
        root
        / "feature_points"
        / f"exchange={_normalize_exchange(exchange)}"
        / f"symbol={compact_symbol}"
    )
    pattern = str(base / "date=*" / "*.parquet")
    try:
        lazy = pl.scan_parquet(pattern)
    except Exception:
        return pl.DataFrame()

    start_ms = _coerce_timestamp_ms(start_date)
    end_ms = _coerce_timestamp_ms(end_date)
    if start_ms is not None:
        lazy = lazy.filter(pl.col("timestamp_ms") >= start_ms)
    if end_ms is not None:
        lazy = lazy.filter(pl.col("timestamp_ms") <= end_ms)
    try:
        frame = lazy.collect()
    except Exception:
        return pl.DataFrame()

    if frame.is_empty():
        return frame
    return frame.sort("timestamp_ms").unique(subset=["timestamp_ms"], keep="last").sort(
        "timestamp_ms"
    )


def _upsert_feature_points(
    root: Path,
    *,
    exchange: str,
    symbol: str,
    rows: list[dict[str, Any]],
) -> int:
    if not rows:
        return 0

    normalized_symbol = normalize_symbol(symbol)
    existing = _load_feature_points(root, exchange=exchange, symbol=normalized_symbol)
    incoming_records: list[dict[str, Any]] = []
    for row in rows:
        ts = _coerce_timestamp_ms(row.get("timestamp_ms"))
        if ts is None:
            continue
        record: dict[str, Any] = {
            "exchange": _normalize_exchange(exchange),
            "symbol": normalized_symbol,
            "timestamp_ms": int(ts),
            "datetime": _utc_iso_from_ms(int(ts)),
            "source": str(row.get("source") or "binance_futures_api"),
        }
        for col in _FEATURE_COLUMNS:
            value = row.get(col)
            record[col] = float(value) if value is not None else None
        incoming_records.append(record)

    if not incoming_records:
        return 0

    incoming = pl.DataFrame(incoming_records)
    canonical_columns = ["exchange", "symbol", "timestamp_ms", "datetime", "source", *_FEATURE_COLUMNS]

    def _align_columns(frame: pl.DataFrame) -> pl.DataFrame:
        out = frame
        for column in canonical_columns:
            if column not in out.columns:
                out = out.with_columns(pl.lit(None).alias(column))
        return out.select(canonical_columns)

    incoming = _align_columns(incoming)
    frames = [incoming]
    if not existing.is_empty():
        frames.append(_align_columns(existing))

    merged = pl.concat(frames, how="vertical_relaxed").sort("timestamp_ms")
    grouped_expr = [
        pl.col("exchange").drop_nulls().last().alias("exchange"),
        pl.col("symbol").drop_nulls().last().alias("symbol"),
        pl.col("datetime").drop_nulls().last().alias("datetime"),
        pl.col("source").drop_nulls().last().alias("source"),
    ]
    grouped_expr.extend(
        [pl.col(col).drop_nulls().last().alias(col) for col in _FEATURE_COLUMNS]
    )

    compacted = merged.group_by("timestamp_ms").agg(grouped_expr).sort("timestamp_ms")
    compacted = compacted.with_columns(
        pl.col("timestamp_ms")
        .cast(pl.Int64)
        .map_elements(lambda value: _timestamp_ms_to_datetime(value).date(), return_dtype=pl.Date)
        .alias("partition_date")
    )

    partitions = compacted.partition_by("partition_date", maintain_order=True)
    for partition in partitions:
        if partition.is_empty():
            continue
        partition_date = partition["partition_date"][0]
        if partition_date is None:
            continue
        date_path = (
            root
            / "feature_points"
            / f"exchange={_normalize_exchange(exchange)}"
            / f"symbol={normalized_symbol.replace('/', '')}"
            / f"date={partition_date.isoformat()}"
        )
        date_path.mkdir(parents=True, exist_ok=True)

        payload = partition.drop("partition_date")
        output_path = date_path / f"compact-{partition_date.isoformat()}.parquet"
        tmp_path = output_path.with_suffix(".tmp.parquet")
        payload.write_parquet(tmp_path, compression="zstd", statistics=True)
        tmp_path.replace(output_path)

        for file_path in list(date_path.glob("*.parquet")):
            if file_path == output_path:
                continue
            try:
                file_path.unlink()
            except FileNotFoundError:
                pass

    return len(incoming_records)


class MarketDataRepository:
    """Facade for local parquet market-data operations."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.root_path = _resolve_market_root_path(self.db_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._parquet_repo = _parquet_repo(self.root_path)
        self._prefer_1s_derived = str(
            os.getenv("LQ_PREFER_1S_DERIVED", "1")
        ).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def get_last_ohlcv_1s_timestamp_ms(self, *, exchange: str, symbol: str) -> int | None:
        frame = self._parquet_repo.load_ohlcv(
            exchange=_normalize_exchange(exchange),
            symbol=normalize_symbol(symbol),
            timeframe="1s",
        )
        if frame.is_empty():
            return None
        max_dt = frame["datetime"].max()
        if max_dt is None:
            return None
        return int(max_dt.timestamp() * 1000)

    def get_last_timestamp_ms(self, *, exchange: str, symbol: str, timeframe: str) -> int | None:
        frame = self.load_ohlcv(exchange=exchange, symbol=symbol, timeframe=timeframe)
        if frame.is_empty():
            return None
        max_dt = frame["datetime"].max()
        if max_dt is None:
            return None
        return int(max_dt.timestamp() * 1000)

    def market_data_exists(self, *, exchange: str, symbol: str, timeframe: str) -> bool:
        frame = self.load_ohlcv(exchange=exchange, symbol=symbol, timeframe=timeframe)
        return not frame.is_empty()

    def load_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        timeframe_token = normalize_timeframe_token(timeframe)
        normalized_exchange = _normalize_exchange(exchange)
        normalized_symbol = normalize_symbol(symbol)
        try:
            merged = self._parquet_repo.load_ohlcv(
                exchange=normalized_exchange,
                symbol=normalized_symbol,
                timeframe=timeframe_token,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            merged = _empty_ohlcv_frame()

        if not merged.is_empty():
            return merged

        direct = _load_direct_ohlcv(
            self.root_path,
            exchange=normalized_exchange,
            symbol=normalized_symbol,
            timeframe=timeframe_token,
            start_date=start_date,
            end_date=end_date,
        )
        if not direct.is_empty() or timeframe_token == "1s" or not self._prefer_1s_derived:
            return direct

        return direct

    def load_ohlcv_1s(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        return self.load_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe="1s",
            start_date=start_date,
            end_date=end_date,
        )

    def load_data_dict(
        self,
        *,
        exchange: str,
        symbol_list: list[str],
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> dict[str, pl.DataFrame]:
        out: dict[str, pl.DataFrame] = {}
        for symbol in symbol_list:
            df = self.load_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            if not df.is_empty():
                out[symbol] = df
        return out

    def export_ohlcv_to_csv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        csv_path: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> int:
        df = self.load_ohlcv(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.write_csv(csv_path)
        return int(df.height)

    def upsert_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        rows: Any,
    ) -> int:
        frame = _ensure_ohlcv_frame(rows)
        timeframe_token = normalize_timeframe_token(timeframe)
        if timeframe_token == "1s":
            return self._parquet_repo.upsert_1s(
                exchange=_normalize_exchange(exchange),
                symbol=normalize_symbol(symbol),
                rows=frame,
            )
        return _upsert_ohlcv_frame(
            self.root_path,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe_token,
            frame=frame,
        )

    def upsert_futures_feature_points(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: list[dict[str, Any]],
    ) -> int:
        return _upsert_feature_points(
            self.root_path,
            exchange=exchange,
            symbol=symbol,
            rows=rows,
        )

    def load_futures_feature_points(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        return _load_feature_points(
            self.root_path,
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )


def get_last_ohlcv_timestamp_ms(
    conn: ParquetMarketDataConnection,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> int | None:
    repo = MarketDataRepository(getattr(conn, "db_path", DEFAULT_MARKET_DATA_DB_PATH))
    return repo.get_last_timestamp_ms(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )


def get_last_ohlcv_1s_timestamp_ms(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    backend: str | None = None,
    **legacy: Any,
) -> int | None:
    _ = (backend, legacy)
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.get_last_ohlcv_1s_timestamp_ms(exchange=exchange, symbol=symbol)


def upsert_ohlcv_rows(
    conn: ParquetMarketDataConnection,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    rows: Any,
    source: str = "binance_api",
    db_path: str | None = None,
    backend: str | None = None,
    **legacy: Any,
) -> int:
    _ = (source, legacy)
    resolved = str(db_path or getattr(conn, "db_path", DEFAULT_MARKET_DATA_DB_PATH))
    repo = _build_market_data_repository(resolved, backend=backend)
    return repo.upsert_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        rows=rows,
    )


def upsert_ohlcv_rows_1s(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    rows: Any,
    backend: str | None = None,
    **legacy: Any,
) -> int:
    _ = legacy
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.upsert_ohlcv(exchange=exchange, symbol=symbol, timeframe="1s", rows=rows)


def upsert_futures_feature_points(
    conn: ParquetMarketDataConnection,
    *,
    exchange: str,
    symbol: str,
    rows: list[dict[str, Any]],
    source: str = "binance_futures_api",
) -> int:
    stamped_rows = []
    for row in rows:
        payload = dict(row)
        payload.setdefault("source", source)
        stamped_rows.append(payload)

    repo = MarketDataRepository(getattr(conn, "db_path", DEFAULT_MARKET_DATA_DB_PATH))
    return repo.upsert_futures_feature_points(exchange=exchange, symbol=symbol, rows=stamped_rows)


def upsert_futures_feature_points_rows(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    rows: list[dict[str, Any]],
    source: str = "binance_futures_api",
    backend: str | None = None,
    **legacy: Any,
) -> int:
    _ = (backend, legacy)
    conn = connect_market_data_db(db_path)
    try:
        return upsert_futures_feature_points(
            conn,
            exchange=exchange,
            symbol=symbol,
            rows=rows,
            source=source,
        )
    finally:
        conn.close()


def load_futures_feature_points_from_db(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    start_date: Any = None,
    end_date: Any = None,
    backend: str | None = None,
    **legacy: Any,
) -> pl.DataFrame:
    _ = (backend, legacy)
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.load_futures_feature_points(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )


def market_data_exists(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    backend: str | None = None,
    **legacy: Any,
) -> bool:
    _ = legacy
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.market_data_exists(exchange=exchange, symbol=symbol, timeframe=timeframe)


def load_ohlcv_coverage_from_db(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
    backend: str | None = None,
    **legacy: Any,
) -> tuple[int | None, int | None, int]:
    _ = legacy
    frame = load_ohlcv_from_db(
        db_path,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        backend=backend,
    )
    if frame.is_empty():
        return None, None, 0
    first_dt = frame["datetime"].min()
    last_dt = frame["datetime"].max()
    first_ts = _datetime_to_epoch_ms(first_dt)
    last_ts = _datetime_to_epoch_ms(last_dt)
    return first_ts, last_ts, int(frame.height)


def load_ohlcv_from_db(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
    backend: str | None = None,
    **legacy: Any,
) -> pl.DataFrame:
    _ = legacy
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.load_ohlcv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )


def load_ohlcv_1s_from_db(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    start_date: Any = None,
    end_date: Any = None,
    backend: str | None = None,
    **legacy: Any,
) -> pl.DataFrame:
    _ = legacy
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.load_ohlcv_1s(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )


def load_data_dict_from_db(
    db_path: str,
    *,
    exchange: str,
    symbol_list: list[str],
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
    backend: str | None = None,
    **legacy: Any,
) -> dict[str, pl.DataFrame]:
    _ = legacy
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.load_data_dict(
        exchange=exchange,
        symbol_list=symbol_list,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )


def export_ohlcv_to_csv(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    csv_path: str,
    start_date: Any = None,
    end_date: Any = None,
    backend: str | None = None,
    **legacy: Any,
) -> int:
    _ = legacy
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.export_ohlcv_to_csv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
    )
