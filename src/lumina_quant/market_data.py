"""Parquet-backed market-data helpers for local runtime workflows."""

from __future__ import annotations

import fcntl
import hashlib
import io
import math
import os
import stat
import uuid
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
from lumina_quant.backtesting.cli_contract import normalize_data_mode
from lumina_quant.symbols import canonical_symbol

MARKET_OHLCV_TABLE = "market_ohlcv"
MARKET_OHLCV_1S_TABLE = "market_ohlcv_1s"
FUTURES_FEATURE_POINTS_TABLE = "futures_feature_points"
DEFAULT_MARKET_DATA_DB_PATH = "data/market_parquet"
KNOWN_QUOTES = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
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
    "funding_fee_rate",
    "funding_fee_quote_per_unit",
    "mark_price",
    "index_price",
    "open_interest",
    "taker_buy_base_volume",
    "taker_sell_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
    "liquidation_long_qty",
    "liquidation_short_qty",
    "liquidation_long_notional",
    "liquidation_short_notional",
    "best_bid_price",
    "best_bid_quantity",
    "best_ask_price",
    "best_ask_quantity",
    "bbo_mid_price",
    "bbo_spread_bps",
    "book_depth_bid_notional_1pct",
    "book_depth_ask_notional_1pct",
    "book_depth_imbalance_1pct",
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
            repo = _parquet_repo(Path(self.db_path))
            with repo.generation_lock(exclusive=False) as root:
                paths = sorted(root.glob("feature_points/exchange=*/symbol=*/date=*/*.parquet"))
                if not paths:
                    return _QueryResult([])
                frame = pl.scan_parquet(paths).sort("timestamp_ms").collect()
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
    from lumina_quant.storage.parquet import ParquetMarketDataRepository

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


def _normalize_data_mode(value: str | None, *, default: str = "legacy") -> str:
    return normalize_data_mode(value, default=default)


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
) -> dict[str, pl.DataFrame]:
    """Owner entrypoint for parquet data loading contract."""
    from lumina_quant.storage.parquet.ohlcv_repo import (
        load_data_dict_from_parquet as _load,
    )

    return _load(
        root_path,
        exchange=exchange,
        symbol_list=symbol_list,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        chunk_days=chunk_days,
        warmup_bars=warmup_bars,
        data_mode=data_mode,
        staleness_threshold_seconds=staleness_threshold_seconds,
    )


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format into BASE/QUOTE uppercase."""
    return canonical_symbol(symbol)


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


def symbol_parquet_candidates(root_dir: str, symbol: str) -> list[str]:
    """Return common parquet path candidates for a symbol."""
    normalized = normalize_symbol(symbol)
    compact = normalized.replace("/", "")
    return [
        os.path.join(root_dir, f"{normalized}.parquet"),
        os.path.join(root_dir, f"{compact}.parquet"),
        os.path.join(root_dir, f"{normalized.replace('/', '_')}.parquet"),
        os.path.join(root_dir, f"{normalized.replace('/', '-')}.parquet"),
    ]


def external_symbol_candidate_paths(
    root_path: str | os.PathLike[str],
    symbol: str,
    *,
    symbol_map: dict[str, str] | None = None,
    include_csv: bool = True,
    include_parquet: bool = True,
) -> list[str]:
    """Return canonical external-data candidate paths for one symbol."""
    root = Path(root_path).expanduser()
    resolved_symbol_map = dict(symbol_map or {})
    candidates: list[str] = []
    if str(symbol) in resolved_symbol_map:
        mapped = str(resolved_symbol_map[str(symbol)] or "").strip()
        if mapped:
            candidates.append(str(root / mapped))
    if include_parquet:
        candidates.extend(symbol_parquet_candidates(str(root), str(symbol)))
    if include_csv:
        candidates.extend(symbol_csv_candidates(str(root), str(symbol)))
    return candidates


def resolve_symbol_csv_path(csv_dir: str, symbol: str) -> str:
    """Resolve the first existing symbol CSV path, fallback to compact name."""
    candidates = symbol_csv_candidates(csv_dir, symbol)
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[1]


def load_data_dict_from_external_root(
    root_path: str,
    *,
    symbol_list: list[str],
    symbol_map: dict[str, str] | None = None,
    start_date: Any = None,
    end_date: Any = None,
) -> dict[str, pl.DataFrame]:
    """Load canonical OHLCV frames from a user-managed external root."""
    from lumina_quant.compute.ohlcv_loader import OHLCVFrameLoader

    loader = OHLCVFrameLoader(start_date=start_date, end_date=end_date)
    root = Path(root_path).expanduser()
    normalized_symbols = [str(symbol) for symbol in list(symbol_list or []) if str(symbol)]
    if root.is_file() and len(normalized_symbols) > 1:
        raise RuntimeError(
            "Single-file external market data only supports one symbol. Use a directory root for multi-symbol external data."
        )
    data: dict[str, pl.DataFrame] = {}
    for symbol in normalized_symbols:
        frame = None
        if root.is_file():
            candidate_paths = [str(root)]
        else:
            candidate_paths = external_symbol_candidate_paths(
                root,
                str(symbol),
                symbol_map=symbol_map,
                include_csv=True,
                include_parquet=True,
            )
        for candidate in candidate_paths:
            path = Path(candidate)
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".parquet":
                    frame = loader.normalize(pl.read_parquet(path))
                else:
                    frame = loader.load_csv(str(path))
            except Exception:
                frame = None
            if frame is not None and not frame.is_empty():
                data[str(symbol)] = frame
                break
    return data


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert timeframe token like 1m/1h/1d into milliseconds.

    Thin delegate to the single canonical leaf util
    ``lumina_quant.data.timeframe``. The import is function-local so that
    importing ``market_data`` does not eagerly pull the ``data`` package (whose
    ``__init__`` re-imports back into ``market_data``) — that eager edge is the
    ``market_data <-> data.*`` import cycle this delegation breaks.
    """
    from lumina_quant.data.timeframe import timeframe_to_milliseconds as _impl

    return _impl(timeframe)


def normalize_timeframe_token(timeframe: str) -> str:
    """Normalize timeframe token — thin delegate to ``lumina_quant.data.timeframe``."""
    from lumina_quant.data.timeframe import normalize_timeframe_token as _impl

    return _impl(timeframe)


def connect_market_data_db(db_path: str) -> ParquetMarketDataConnection:
    """Open a read-only compatibility handle; locked writers create storage."""
    return ParquetMarketDataConnection(str(_resolve_market_root_path(db_path)))


def resolve_1s_db_path(db_path: str) -> str:
    """Resolve market parquet root path for 1-second bars."""
    explicit = str(os.getenv("LQ_1S_DB_PATH", "")).strip()
    if explicit:
        return str(_resolve_market_root_path(explicit))
    return str(_resolve_market_root_path(db_path))


def connect_market_data_1s_db(db_path: str) -> ParquetMarketDataConnection:
    """Open a read-only 1-second compatibility handle."""
    return ParquetMarketDataConnection(str(Path(resolve_1s_db_path(db_path))))


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


def _partition_parquet_paths(
    base: Path,
    *,
    start_date: Any = None,
    end_date: Any = None,
) -> list[str]:
    start_ms = _coerce_timestamp_ms(start_date)
    end_ms = _coerce_timestamp_ms(end_date)
    start_token = (
        _timestamp_ms_to_datetime(start_ms).date().isoformat() if start_ms is not None else None
    )
    end_token = _timestamp_ms_to_datetime(end_ms).date().isoformat() if end_ms is not None else None

    partition_dirs = sorted(path for path in base.glob("date=*") if path.is_dir())

    parquet_paths: list[str] = []
    for partition_dir in partition_dirs:
        token = partition_dir.name.partition("=")[2]
        if not token:
            continue
        if start_token is not None and token < start_token:
            continue
        if end_token is not None and token > end_token:
            continue
        parquet_paths.extend(str(path) for path in sorted(partition_dir.glob("*.parquet")))
    return parquet_paths


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
    parquet_paths = _partition_parquet_paths(
        base,
        start_date=start_date,
        end_date=end_date,
    )
    if not parquet_paths:
        return _empty_ohlcv_frame()
    lazy = pl.scan_parquet(parquet_paths)

    start_ms = _coerce_timestamp_ms(start_date)
    end_ms = _coerce_timestamp_ms(end_date)
    if start_ms is not None:
        lazy = lazy.filter(pl.col("datetime") >= _timestamp_ms_to_datetime(start_ms))
    if end_ms is not None:
        lazy = lazy.filter(pl.col("datetime") <= _timestamp_ms_to_datetime(end_ms))

    data = lazy.select(["datetime", "open", "high", "low", "close", "volume"]).collect()

    if data.is_empty():
        return _empty_ohlcv_frame()
    return data.sort("datetime").unique(subset=["datetime"], keep="last").sort("datetime")


def load_strict_ohlcv_route(
    db_path: str | os.PathLike[str],
    *,
    storage_route: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
) -> pl.DataFrame:
    """Read one explicitly declared local OHLCV layout without fallback."""
    if storage_route != "partitioned_ohlcv":
        raise ValueError(f"unsupported OHLCV storage route: {storage_route!r}")
    repo = _parquet_repo(Path(db_path))
    with repo.generation_lock(exclusive=False) as root:
        if not root.is_dir():
            raise ValueError("market-data root must already exist")
        base = _series_path(root, exchange=exchange, symbol=symbol, timeframe=timeframe)
        if not base.is_dir():
            raise FileNotFoundError(f"partitioned OHLCV series is missing: {base}")
        return _load_direct_ohlcv(
            root,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )


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
                pl.col("datetime")
                .cast(pl.Datetime(time_unit="ms"), strict=False)
                .alias("datetime"),
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
            frames.append(pl.read_parquet(file_path))

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
    progress_callback: Any = None,
) -> pl.DataFrame:
    compact_symbol = normalize_symbol(symbol).replace("/", "")
    base = (
        root
        / "feature_points"
        / f"exchange={_normalize_exchange(exchange)}"
        / f"symbol={compact_symbol}"
    )
    scan_started_at = perf_counter()
    parquet_paths = _partition_parquet_paths(
        base,
        start_date=start_date,
        end_date=end_date,
    )
    partition_count = len({Path(path).parent.name for path in parquet_paths})
    if progress_callback is not None:
        progress_callback(
            "resource_feature_partition_scan_completed",
            {
                "symbol": str(symbol),
                "partition_count": partition_count,
                "parquet_file_count": len(parquet_paths),
                "elapsed_seconds": round(max(0.0, perf_counter() - scan_started_at), 6),
            },
        )
    if not parquet_paths:
        return pl.DataFrame()
    lazy = pl.scan_parquet(parquet_paths)

    start_ms = _coerce_timestamp_ms(start_date)
    end_ms = _coerce_timestamp_ms(end_date)
    if start_ms is not None:
        lazy = lazy.filter(pl.col("timestamp_ms") >= start_ms)
    if end_ms is not None:
        lazy = lazy.filter(pl.col("timestamp_ms") <= end_ms)
    if progress_callback is not None:
        progress_callback(
            "resource_feature_collect_started",
            {
                "symbol": str(symbol),
                "partition_count": partition_count,
                "parquet_file_count": len(parquet_paths),
            },
        )
    collect_started_at = perf_counter()
    frame = lazy.collect()
    if progress_callback is not None:
        progress_callback(
            "resource_feature_collect_completed",
            {
                "symbol": str(symbol),
                "partition_count": partition_count,
                "parquet_file_count": len(parquet_paths),
                "row_count": int(frame.height),
                "elapsed_seconds": round(max(0.0, perf_counter() - collect_started_at), 6),
            },
        )

    if frame.is_empty():
        return frame
    return (
        frame.sort("timestamp_ms").unique(subset=["timestamp_ms"], keep="last").sort("timestamp_ms")
    )


def _load_feature_points_day(root: Path, *, exchange: str, symbol: str, day: str) -> pl.DataFrame:
    """Load exactly one feature-date partition while its generation is pinned."""
    try:
        parsed_day = datetime.strptime(day, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("feature day is invalid") from exc
    if parsed_day.isoformat() != day:
        raise ValueError("feature day is not canonical")
    date_path = (
        root
        / "feature_points"
        / f"exchange={_normalize_exchange(exchange)}"
        / f"symbol={normalize_symbol(symbol).replace('/', '')}"
        / f"date={day}"
    )
    paths = sorted(date_path.glob("*.parquet"))
    if not paths:
        return pl.DataFrame(schema=_FEATURE_SCHEMA)
    frame = pl.concat(
        [_align_feature_frame(pl.read_parquet(path)) for path in paths],
        how="vertical_relaxed",
    )
    return (
        _align_feature_frame(frame)
        .sort("timestamp_ms")
        .unique(subset=["timestamp_ms"], keep="last")
        .sort("timestamp_ms")
    )


@contextmanager
def _generation_lock(logical_root: Path, *, exclusive: bool):
    """Resolve and pin the logical root through the shared storage contract."""
    repo = _parquet_repo(logical_root)
    with repo.generation_lock(exclusive=exclusive) as physical_root:
        yield physical_root


@contextmanager
def _feature_partition_lock(root: Path, *, exchange: str, symbol: str, day: str):
    """Serialize all writers for one canonical feature partition."""
    lock = (
        root
        / "feature_points"
        / f"exchange={_normalize_exchange(exchange)}"
        / f"symbol={normalize_symbol(symbol).replace('/', '')}"
        / f"date={day}"
        / ".writer.lock"
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info, named = os.fstat(fd), os.lstat(lock)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError("unsafe feature partition lock")
        fcntl.flock(fd, fcntl.LOCK_EX)
        if (info.st_dev, info.st_ino) != (os.lstat(lock).st_dev, os.lstat(lock).st_ino):
            raise ValueError("feature partition lock changed")
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


_FEATURE_SCHEMA = {
    "exchange": pl.Utf8,
    "symbol": pl.Utf8,
    "timestamp_ms": pl.Int64,
    "datetime": pl.Utf8,
    "source": pl.Utf8,
    **dict.fromkeys(_FEATURE_COLUMNS, pl.Float64),
}
_FEATURE_CANONICAL_COLUMNS = list(_FEATURE_SCHEMA)


def _align_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for column in _FEATURE_CANONICAL_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None).alias(column))
    return out.select(_FEATURE_CANONICAL_COLUMNS).cast(_FEATURE_SCHEMA, strict=False)


def _atomic_feature_write(
    date_path: Path,
    output_path: Path,
    frame: pl.DataFrame,
    *,
    max_output_bytes: int | None = None,
) -> None:
    """Durably replace the compact file without touching control-plane files."""
    tmp = date_path / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    obsolete = [path for path in date_path.glob("*.parquet") if path != output_path]
    for path in obsolete:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("unsafe obsolete feature parquet")
    errors: list[BaseException] = []
    try:
        if max_output_bytes is None:
            frame.write_parquet(tmp, compression="zstd", statistics=True)
        else:
            buffer = io.BytesIO()
            frame.write_parquet(buffer, compression="zstd", statistics=True)
            payload = buffer.getvalue()
            if len(payload) > max_output_bytes:
                raise ValueError("feature parquet exceeds publication quota")
            output_fd = os.open(
                tmp,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(output_fd, view)
                    if written <= 0:
                        raise OSError("short feature parquet write")
                    view = view[written:]
                os.fsync(output_fd)
            finally:
                os.close(output_fd)
        fd = os.open(tmp, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("unsafe feature temporary parquet")
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, output_path)
        directory = os.open(date_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
            for path in obsolete:
                info = os.lstat(path)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("unsafe obsolete feature parquet")
                try:
                    path.unlink()
                except BaseException as exc:
                    errors.append(exc)
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException as exc:
        errors.insert(0, exc)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        except BaseException as exc:
            errors.append(exc)
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise ExceptionGroup("feature write and cleanup failures", errors)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _seal_write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("official funding seal readback failed")
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _feature_digest(frame: pl.DataFrame) -> str:
    data = json.dumps(
        _align_feature_frame(frame).sort("timestamp_ms").to_dicts(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(data.encode()).hexdigest()


def _publish_official_funding_day(
    root: Path,
    *,
    exchange: str,
    symbol: str,
    day: str,
    source: Path,
    expected_sha256: str,
    expected_byte_count: int,
    expected_row_count: int,
    provenance_receipt_sha256: str,
    max_output_bytes: int | None = None,
) -> Path:
    """Merge a receipt-authenticated funding day as one sealed transaction."""
    if (
        isinstance(expected_byte_count, bool)
        or isinstance(expected_row_count, bool)
        or not isinstance(expected_byte_count, int)
        or not isinstance(expected_row_count, int)
        or expected_byte_count < 0
        or expected_row_count <= 0
    ):
        raise ValueError("funding receipt count is invalid")
    if max_output_bytes is not None and (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
    ):
        raise ValueError("max_output_bytes must be a positive integer")
    if not all(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
        for value in (expected_sha256, provenance_receipt_sha256)
    ):
        raise ValueError("funding receipt hash is invalid")
    exchange, symbol = _normalize_exchange(exchange), normalize_symbol(symbol)
    try:
        parsed_day = datetime.strptime(day, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("funding day is invalid") from exc
    if parsed_day.isoformat() != day:
        raise ValueError("funding day is not canonical")
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("funding source is unsafe")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256 or before.st_size != expected_byte_count:
            raise ValueError("funding source bytes do not match receipt")
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as stream:
            official = pl.read_parquet(stream)
        after, named = os.fstat(fd), os.lstat(source)

        def identity(info: os.stat_result) -> tuple[int, ...]:
            return (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                info.st_nlink,
                info.st_uid,
                info.st_gid,
            )

        if identity(before) != identity(after) or (before.st_dev, before.st_ino) != (
            named.st_dev,
            named.st_ino,
        ):
            raise ValueError("funding source identity drift")
    finally:
        os.close(fd)
    required = ("timestamp_ms", "source_timestamp_ms", "exchange", "symbol", "funding_rate")
    expected_schema = {
        "timestamp_ms": pl.Int64,
        "source_timestamp_ms": pl.Int64,
        "exchange": pl.Utf8,
        "symbol": pl.Utf8,
        "funding_rate": pl.Float64,
    }
    if (
        tuple(official.columns) != required
        or official.schema != expected_schema
        or official.height != expected_row_count
    ):
        raise ValueError("funding source schema or count is invalid")
    prior = None
    for row in official.iter_rows(named=True):
        ts, source_ts, rate = row["timestamp_ms"], row["source_timestamp_ms"], row["funding_rate"]
        if (
            isinstance(ts, bool)
            or isinstance(source_ts, bool)
            or not isinstance(ts, int)
            or not isinstance(source_ts, int)
            or source_ts != ts
            or row["exchange"] != exchange
            or row["symbol"] != symbol
            or (prior is not None and ts <= prior)
            or _timestamp_ms_to_datetime(ts).date() != parsed_day
            or not isinstance(rate, float)
            or not math.isfinite(rate)
        ):
            raise ValueError("funding source semantic validation failed")
        prior = ts
    seal_fields = (
        "schema",
        "state",
        "source_sha256",
        "source_byte_count",
        "source_row_count",
        "provenance_receipt_sha256",
        "exchange",
        "symbol",
        "day",
        "output_sha256",
        "output_byte_count",
        "output_row_count",
        "semantic_digest",
    )
    with _generation_lock(root, exclusive=True) as physical_root:
        date_path = (
            physical_root
            / "feature_points"
            / f"exchange={exchange}"
            / f"symbol={symbol.replace('/', '')}"
            / f"date={day}"
        )
        output = date_path / f"compact-{day}.parquet"
        seal = date_path / "alpha_max_official_funding_seal.v1"
        if seal.exists():
            preliminary = json.loads(seal.read_text(encoding="utf-8"))
            expected_identity = {
                "schema": "alpha_max_official_funding_seal.v1",
                "source_sha256": expected_sha256,
                "source_byte_count": expected_byte_count,
                "source_row_count": expected_row_count,
                "provenance_receipt_sha256": provenance_receipt_sha256,
                "exchange": exchange,
                "symbol": symbol,
                "day": day,
            }
            if any(preliminary.get(key) != value for key, value in expected_identity.items()):
                raise ValueError("official funding seal conflicts with target")
        with _feature_partition_lock(physical_root, exchange=exchange, symbol=symbol, day=day):
            date_path.mkdir(parents=True, exist_ok=True)
            existing = _load_feature_points_day(
                physical_root, exchange=exchange, symbol=symbol, day=day
            )
            incoming = _align_feature_frame(
                official.with_columns(
                    pl.col("timestamp_ms")
                    .map_elements(_utc_iso_from_ms, return_dtype=pl.Utf8)
                    .alias("datetime"),
                    pl.lit("alpha_max_official").alias("source"),
                )
            )
            overlap = _align_feature_frame(existing).join(
                incoming.select("timestamp_ms", pl.col("funding_rate").alias("_official")),
                on="timestamp_ms",
                how="inner",
            )
            if not overlap.filter(
                pl.col("funding_rate").is_not_null()
                & pl.col("_official").is_not_null()
                & (pl.col("funding_rate") != pl.col("_official"))
            ).is_empty():
                raise ValueError("official funding conflicts with existing funding")
            expressions = [
                pl.col(column).drop_nulls().last().alias(column)
                for column in _FEATURE_CANONICAL_COLUMNS
                if column != "timestamp_ms"
            ]
            merged = _align_feature_frame(
                pl.concat([incoming, _align_feature_frame(existing)], how="vertical_relaxed")
                .group_by("timestamp_ms")
                .agg(expressions)
                .sort("timestamp_ms")
            )
            target = {
                "schema": "alpha_max_official_funding_seal.v1",
                "source_sha256": expected_sha256,
                "source_byte_count": expected_byte_count,
                "source_row_count": expected_row_count,
                "provenance_receipt_sha256": provenance_receipt_sha256,
                "exchange": exchange,
                "symbol": symbol,
                "day": day,
                "output_row_count": merged.height,
                "semantic_digest": _feature_digest(merged),
            }
            stored = None
            if seal.exists():
                stored = json.loads(seal.read_text(encoding="utf-8"))
                if set(stored) != set(seal_fields) or stored.get("schema") != target["schema"]:
                    raise ValueError("official funding seal is invalid")
                identity_fields = (
                    "source_sha256",
                    "source_byte_count",
                    "source_row_count",
                    "provenance_receipt_sha256",
                    "exchange",
                    "symbol",
                    "day",
                    "semantic_digest",
                )
                if not all(stored.get(key) == target[key] for key in identity_fields):
                    raise ValueError("official funding seal conflicts with target")
                if stored.get("state") not in {"pending", "final"}:
                    raise ValueError("official funding seal state is invalid")
            if stored is not None and stored["state"] == "final":
                if not output.exists():
                    raise ValueError("official funding seal conflicts with target")
                actual = {
                    "output_sha256": _sha256_path(output),
                    "output_byte_count": output.stat().st_size,
                    "output_row_count": _align_feature_frame(pl.read_parquet(output)).height,
                    "semantic_digest": _feature_digest(pl.read_parquet(output)),
                }
                if not all(stored.get(key) == actual[key] for key in actual):
                    raise ValueError("official funding seal conflicts with target")
                return output
            if stored is None:
                _seal_write(
                    seal,
                    {
                        **target,
                        "state": "pending",
                        "output_sha256": "",
                        "output_byte_count": 0,
                    },
                )
            if stored is not None and output.exists():
                current = _align_feature_frame(pl.read_parquet(output))
                if _feature_digest(current) != target["semantic_digest"]:
                    raise ValueError("pending official funding target conflicts")
            else:
                _atomic_feature_write(
                    date_path,
                    output,
                    merged,
                    max_output_bytes=max_output_bytes,
                )
            final = {
                **target,
                "state": "final",
                "output_sha256": _sha256_path(output),
                "output_byte_count": output.stat().st_size,
                "output_row_count": _align_feature_frame(pl.read_parquet(output)).height,
                "semantic_digest": _feature_digest(pl.read_parquet(output)),
            }
            if (
                final["output_row_count"] != target["output_row_count"]
                or final["semantic_digest"] != target["semantic_digest"]
            ):
                raise ValueError("official funding output readback conflicts")
            _seal_write(seal, final)
            return output


def _upsert_feature_points(
    root: Path, *, exchange: str, symbol: str, rows: list[dict[str, Any]]
) -> int:
    if not rows:
        return 0
    normalized_symbol = normalize_symbol(symbol)
    records: list[dict[str, Any]] = []
    for row in rows:
        ts = _coerce_timestamp_ms(row.get("timestamp_ms"))
        if ts is not None:
            records.append(
                {
                    "exchange": _normalize_exchange(exchange),
                    "symbol": normalized_symbol,
                    "timestamp_ms": int(ts),
                    "datetime": _utc_iso_from_ms(int(ts)),
                    "source": str(row.get("source") or "binance_futures_api"),
                    **{
                        col: (float(row[col]) if row.get(col) is not None else None)
                        for col in _FEATURE_COLUMNS
                    },
                }
            )
    if not records:
        return 0
    incoming = _align_feature_frame(pl.DataFrame(records, schema=_FEATURE_SCHEMA, strict=False))
    incoming = incoming.with_columns(
        pl.col("timestamp_ms")
        .map_elements(lambda value: _timestamp_ms_to_datetime(value).date(), return_dtype=pl.Date)
        .alias("_day")
    )
    with _generation_lock(root, exclusive=True) as physical_root:
        for partition in incoming.partition_by("_day", maintain_order=True):
            day = partition["_day"][0].isoformat()
            date_path = (
                physical_root
                / "feature_points"
                / f"exchange={_normalize_exchange(exchange)}"
                / f"symbol={normalized_symbol.replace('/', '')}"
                / f"date={day}"
            )
            output = date_path / f"compact-{day}.parquet"
            with _feature_partition_lock(
                physical_root, exchange=exchange, symbol=normalized_symbol, day=day
            ):
                date_path.mkdir(parents=True, exist_ok=True)
                if (date_path / "alpha_max_official_funding_seal.v1").exists():
                    raise ValueError("cannot modify sealed official funding day")
                current = _load_feature_points_day(
                    physical_root, exchange=exchange, symbol=normalized_symbol, day=day
                )
                merged = pl.concat(
                    [_align_feature_frame(current), partition.drop("_day")], how="vertical_relaxed"
                )
                expressions = [
                    pl.col(column).drop_nulls().last().alias(column)
                    for column in _FEATURE_CANONICAL_COLUMNS
                    if column != "timestamp_ms"
                ]
                compacted = _align_feature_frame(
                    merged.group_by("timestamp_ms").agg(expressions).sort("timestamp_ms")
                )
                _atomic_feature_write(date_path, output, compacted)
    return len(records)


class MarketDataRepository:
    """Facade for local parquet market-data operations."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self.logical_root_path = _resolve_market_root_path(self.db_path)
        self.root_path = self.logical_root_path
        self._parquet_repo = _parquet_repo(self.logical_root_path)
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
        with self._parquet_repo.generation_lock(exclusive=False) as physical_root:
            merged = self._parquet_repo.load_ohlcv(
                exchange=normalized_exchange,
                symbol=normalized_symbol,
                timeframe=timeframe_token,
                start_date=start_date,
                end_date=end_date,
            )
            if not merged.is_empty():
                return merged
            return _load_direct_ohlcv(
                physical_root,
                exchange=normalized_exchange,
                symbol=normalized_symbol,
                timeframe=timeframe_token,
                start_date=start_date,
                end_date=end_date,
            )

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
        with self._parquet_repo.generation_lock(exclusive=False):
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
        with _generation_lock(self.logical_root_path, exclusive=True) as physical_root:
            return _upsert_ohlcv_frame(
                physical_root,
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
            self.logical_root_path,
            exchange=exchange,
            symbol=symbol,
            rows=rows,
        )

    def publish_official_funding_day(
        self,
        *,
        exchange: str,
        symbol: str,
        day: str,
        source: Path,
        expected_sha256: str,
        expected_byte_count: int,
        expected_row_count: int,
        provenance_receipt_sha256: str,
        max_output_bytes: int | None = None,
    ) -> Path:
        return _publish_official_funding_day(
            self.logical_root_path,
            exchange=exchange,
            symbol=symbol,
            day=day,
            source=source,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
            expected_row_count=expected_row_count,
            provenance_receipt_sha256=provenance_receipt_sha256,
            max_output_bytes=max_output_bytes,
        )

    def load_futures_feature_points(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
        progress_callback: Any = None,
    ) -> pl.DataFrame:
        with _generation_lock(self.logical_root_path, exclusive=False) as physical_root:
            return _load_feature_points(
                physical_root,
                exchange=exchange,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                progress_callback=progress_callback,
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
    progress_callback: Any = None,
    **legacy: Any,
) -> pl.DataFrame:
    _ = (backend, legacy)
    repo = _build_market_data_repository(str(db_path), backend=backend)
    return repo.load_futures_feature_points(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress_callback,
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
