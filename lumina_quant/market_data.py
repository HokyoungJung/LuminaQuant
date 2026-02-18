"""Market data storage and loading helpers."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import polars as pl

MARKET_OHLCV_TABLE = "market_ohlcv"
MARKET_OHLCV_1S_TABLE = "market_ohlcv_1s"
KNOWN_QUOTES = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
TIMEFRAME_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}
EMPTY_OHLCV_SCHEMA = {
    "datetime": pl.Datetime(time_unit="ms"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


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
    token = str(timeframe).strip().lower()
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


def connect_market_data_db(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection for market OHLCV storage."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_1s_db_path(db_path: str) -> str:
    """Resolve dedicated compact SQLite path for 1-second bars."""
    explicit = str(os.getenv("LQ_1S_DB_PATH", "")).strip()
    if explicit:
        return explicit
    root, ext = os.path.splitext(str(db_path))
    suffix = ext if ext else ".db"
    return f"{root}_1s{suffix}"


def connect_market_data_1s_db(db_path: str) -> sqlite3.Connection:
    """Open compact SQLite connection for 1-second OHLCV storage."""
    resolved = resolve_1s_db_path(db_path)
    db_dir = os.path.dirname(resolved)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_market_ohlcv_schema(conn: sqlite3.Connection) -> None:
    """Ensure market OHLCV table and indexes are present."""
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {MARKET_OHLCV_TABLE} (
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            source TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (exchange, symbol, timeframe, timestamp_ms)
        );

        CREATE INDEX IF NOT EXISTS idx_market_ohlcv_symbol_time
        ON {MARKET_OHLCV_TABLE}(symbol, timeframe, timestamp_ms);
        """
    )
    conn.commit()


def ensure_market_ohlcv_1s_schema(conn: sqlite3.Connection) -> None:
    """Ensure compact 1-second OHLCV table and indexes are present."""
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {MARKET_OHLCV_1S_TABLE} (
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (exchange, symbol, timestamp_ms)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_market_ohlcv_1s_symbol_time
        ON {MARKET_OHLCV_1S_TABLE}(symbol, timestamp_ms);
        """
    )
    conn.commit()


def _utc_iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).isoformat()


def _coerce_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    if isinstance(value, (int, float)):
        numeric = int(value)
        # Heuristic: values below 1e11 are likely seconds.
        if abs(numeric) < 100_000_000_000:
            return numeric * 1000
        return numeric
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    return None


def get_last_ohlcv_timestamp_ms(
    conn: sqlite3.Connection,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> int | None:
    """Return latest stored bar timestamp for a stream key."""
    row = conn.execute(
        f"""
        SELECT MAX(timestamp_ms) AS max_ts
        FROM {MARKET_OHLCV_TABLE}
        WHERE exchange = ? AND symbol = ? AND timeframe = ?
        """,
        (
            str(exchange).strip().lower(),
            normalize_symbol(symbol),
            str(timeframe).strip().lower(),
        ),
    ).fetchone()
    if row is None:
        return None
    value = row["max_ts"]
    return int(value) if value is not None else None


def get_last_ohlcv_1s_timestamp_ms(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
) -> int | None:
    """Return latest stored compact 1-second bar timestamp for key."""
    repo = MarketDataRepository(db_path)
    return repo.get_last_ohlcv_1s_timestamp_ms(exchange=exchange, symbol=symbol)


class MarketDataRepository:
    """OOP facade for SQLite-backed market data operations."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)

    def get_last_ohlcv_1s_timestamp_ms(self, *, exchange: str, symbol: str) -> int | None:
        conn = connect_market_data_1s_db(self.db_path)
        try:
            ensure_market_ohlcv_1s_schema(conn)
            row = conn.execute(
                f"""
                SELECT MAX(timestamp_ms) AS max_ts
                FROM {MARKET_OHLCV_1S_TABLE}
                WHERE exchange = ? AND symbol = ?
                """,
                (
                    str(exchange).strip().lower(),
                    normalize_symbol(symbol),
                ),
            ).fetchone()
            if row is None:
                return None
            value = row["max_ts"]
            return int(value) if value is not None else None
        finally:
            conn.close()

    def market_data_exists(self, *, exchange: str, symbol: str, timeframe: str) -> bool:
        if str(timeframe).strip().lower() == "1s":
            conn = connect_market_data_1s_db(self.db_path)
            try:
                ensure_market_ohlcv_1s_schema(conn)
                row = conn.execute(
                    f"""
                    SELECT 1
                    FROM {MARKET_OHLCV_1S_TABLE}
                    WHERE exchange = ? AND symbol = ?
                    LIMIT 1
                    """,
                    (
                        str(exchange).strip().lower(),
                        normalize_symbol(symbol),
                    ),
                ).fetchone()
                return row is not None
            finally:
                conn.close()

        conn = connect_market_data_db(self.db_path)
        try:
            ensure_market_ohlcv_schema(conn)
            row = conn.execute(
                f"""
                SELECT 1
                FROM {MARKET_OHLCV_TABLE}
                WHERE exchange = ? AND symbol = ? AND timeframe = ?
                LIMIT 1
                """,
                (
                    str(exchange).strip().lower(),
                    normalize_symbol(symbol),
                    str(timeframe).strip().lower(),
                ),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def load_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        if str(timeframe).strip().lower() == "1s":
            return self.load_ohlcv_1s(
                exchange=exchange,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        conn = connect_market_data_db(self.db_path)
        try:
            ensure_market_ohlcv_schema(conn)
            start_ms = _coerce_timestamp_ms(start_date)
            end_ms = _coerce_timestamp_ms(end_date)

            query = (
                f"SELECT timestamp_ms, open, high, low, close, volume "
                f"FROM {MARKET_OHLCV_TABLE} "
                f"WHERE exchange = ? AND symbol = ? AND timeframe = ?"
            )
            params: list[Any] = [
                str(exchange).strip().lower(),
                normalize_symbol(symbol),
                str(timeframe).strip().lower(),
            ]
            if start_ms is not None:
                query += " AND timestamp_ms >= ?"
                params.append(start_ms)
            if end_ms is not None:
                query += " AND timestamp_ms <= ?"
                params.append(end_ms)
            query += " ORDER BY timestamp_ms"

            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        if not rows:
            return _empty_ohlcv_frame()

        frame = pl.DataFrame(
            rows,
            schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        return frame.with_columns(
            pl.from_epoch("timestamp_ms", time_unit="ms").alias("datetime")
        ).select(["datetime", "open", "high", "low", "close", "volume"])

    def load_ohlcv_1s(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        conn = connect_market_data_1s_db(self.db_path)
        try:
            ensure_market_ohlcv_1s_schema(conn)
            start_ms = _coerce_timestamp_ms(start_date)
            end_ms = _coerce_timestamp_ms(end_date)
            query = (
                f"SELECT timestamp_ms, open, high, low, close, volume "
                f"FROM {MARKET_OHLCV_1S_TABLE} "
                f"WHERE exchange = ? AND symbol = ?"
            )
            params: list[Any] = [
                str(exchange).strip().lower(),
                normalize_symbol(symbol),
            ]
            if start_ms is not None:
                query += " AND timestamp_ms >= ?"
                params.append(start_ms)
            if end_ms is not None:
                query += " AND timestamp_ms <= ?"
                params.append(end_ms)
            query += " ORDER BY timestamp_ms"
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        if not rows:
            return _empty_ohlcv_frame()

        frame = pl.DataFrame(
            rows,
            schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        return frame.with_columns(
            pl.from_epoch("timestamp_ms", time_unit="ms").alias("datetime")
        ).select(["datetime", "open", "high", "low", "close", "volume"])

    def load_data_dict(
        self,
        *,
        exchange: str,
        symbol_list: Sequence[str],
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


def upsert_ohlcv_rows(
    conn: sqlite3.Connection,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    rows: Sequence[Sequence[float]],
    source: str = "binance_api",
) -> int:
    """Insert or update OHLCV rows idempotently."""
    if not rows:
        return 0

    now = datetime.now(UTC).isoformat()
    payload: list[tuple[Any, ...]] = []
    stream_exchange = str(exchange).strip().lower()
    stream_symbol = normalize_symbol(symbol)
    stream_timeframe = str(timeframe).strip().lower()
    stream_source = str(source).strip().lower()

    for row in rows:
        timestamp_ms = int(row[0])
        payload.append(
            (
                stream_exchange,
                stream_symbol,
                stream_timeframe,
                timestamp_ms,
                _utc_iso_from_ms(timestamp_ms),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                stream_source,
                now,
            )
        )

    conn.executemany(
        f"""
        INSERT INTO {MARKET_OHLCV_TABLE}(
            exchange, symbol, timeframe, timestamp_ms, datetime,
            open, high, low, close, volume, source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(exchange, symbol, timeframe, timestamp_ms)
        DO UPDATE SET
            datetime = excluded.datetime,
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            source = excluded.source,
            updated_at = excluded.updated_at
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def upsert_ohlcv_rows_1s(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    rows: Sequence[Sequence[float]],
) -> int:
    """Insert/update compact 1-second OHLCV rows idempotently."""
    if not rows:
        return 0

    conn = connect_market_data_1s_db(db_path)
    try:
        ensure_market_ohlcv_1s_schema(conn)
        now = datetime.now(UTC).isoformat()
        payload: list[tuple[Any, ...]] = []
        stream_exchange = str(exchange).strip().lower()
        stream_symbol = normalize_symbol(symbol)
        for row in rows:
            payload.append(
                (
                    stream_exchange,
                    stream_symbol,
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    now,
                )
            )

        conn.executemany(
            f"""
            INSERT INTO {MARKET_OHLCV_1S_TABLE}(
                exchange, symbol, timestamp_ms, open, high, low, close, volume, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exchange, symbol, timestamp_ms)
            DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                updated_at = excluded.updated_at
            """,
            payload,
        )
        conn.commit()
        return len(payload)
    finally:
        conn.close()


def market_data_exists(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> bool:
    """Return True if at least one OHLCV row exists for key."""
    repo = MarketDataRepository(db_path)
    return repo.market_data_exists(exchange=exchange, symbol=symbol, timeframe=timeframe)


def load_ohlcv_from_db(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
) -> pl.DataFrame:
    """Load OHLCV data from SQLite into canonical DataFrame format."""
    repo = MarketDataRepository(db_path)
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
) -> pl.DataFrame:
    """Load compact 1-second OHLCV rows into canonical DataFrame format."""
    repo = MarketDataRepository(db_path)
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
    symbol_list: Sequence[str],
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
) -> dict[str, pl.DataFrame]:
    """Load a symbol->DataFrame dictionary from SQLite market data."""
    repo = MarketDataRepository(db_path)
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
) -> int:
    """Export OHLCV from SQLite into a CSV file. Returns exported row count."""
    repo = MarketDataRepository(db_path)
    return repo.export_ohlcv_to_csv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
    )
