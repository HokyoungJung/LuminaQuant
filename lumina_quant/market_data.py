"""Market data storage and loading helpers."""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import polars as pl
from lumina_quant.influx_market_data import InfluxMarketDataRepository

MARKET_OHLCV_TABLE = "market_ohlcv"
MARKET_OHLCV_1S_TABLE = "market_ohlcv_1s"
FUTURES_FEATURE_POINTS_TABLE = "futures_feature_points"
DEFAULT_MARKET_DATA_DB_PATH = "data/lq_market.sqlite3"
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


def normalize_storage_backend(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"sqlite", "sqlite3"}:
        return "sqlite"
    if token in {"influx", "influxdb"}:
        return "influxdb"
    return "influxdb"


def _resolve_storage_backend(backend: str | None = None) -> str:
    explicit = str(backend or "").strip()
    if explicit:
        return normalize_storage_backend(explicit)
    env_backend = os.getenv("LQ__STORAGE__BACKEND") or os.getenv("LQ_STORAGE_BACKEND") or "influxdb"
    return normalize_storage_backend(env_backend)


def _runtime_influx_defaults() -> tuple[str, str, str, str]:
    """Best-effort runtime fallback for influx connection settings."""
    try:
        from lumina_quant.configuration.loader import load_runtime_config

        runtime = load_runtime_config(config_path=os.getenv("LQ_CONFIG_PATH", "config.yaml"))
        storage = runtime.storage
        return (
            str(getattr(storage, "influx_url", "") or "").strip(),
            str(getattr(storage, "influx_org", "") or "").strip(),
            str(getattr(storage, "influx_bucket", "") or "").strip(),
            str(getattr(storage, "influx_token_env", "INFLUXDB_TOKEN") or "INFLUXDB_TOKEN").strip(),
        )
    except Exception:
        return "", "", "", "INFLUXDB_TOKEN"


def _build_market_data_repository(
    db_path: str,
    *,
    backend: str | None = None,
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> Any:
    explicit_backend = str(backend or "").strip()
    storage_backend = _resolve_storage_backend(backend)
    if storage_backend != "influxdb":
        return MarketDataRepository(db_path)

    runtime_url, runtime_org, runtime_bucket, runtime_token_env = _runtime_influx_defaults()

    token_env_name = str(
        influx_token_env
        or os.getenv("LQ__STORAGE__INFLUX_TOKEN_ENV")
        or os.getenv("INFLUX_TOKEN_ENV")
        or runtime_token_env
        or "INFLUXDB_TOKEN"
    ).strip()
    token_value = str(influx_token or os.getenv(token_env_name, "") or "").strip()
    resolved_url = str(
        influx_url
        or os.getenv("LQ__STORAGE__INFLUX_URL")
        or os.getenv("INFLUX_URL")
        or runtime_url
        or ""
    ).strip()
    resolved_org = str(
        influx_org
        or os.getenv("LQ__STORAGE__INFLUX_ORG")
        or os.getenv("INFLUX_ORG")
        or runtime_org
        or ""
    ).strip()
    resolved_bucket = str(
        influx_bucket
        or os.getenv("LQ__STORAGE__INFLUX_BUCKET")
        or os.getenv("INFLUX_BUCKET")
        or runtime_bucket
        or ""
    ).strip()

    has_required_influx_config = all((resolved_url, resolved_org, resolved_bucket, token_value))
    if not has_required_influx_config:
        raise ValueError(
            "InfluxDB configuration incomplete. Require url/org/bucket/token for backend=influxdb. "
            f"backend={storage_backend!r}, explicit_backend={explicit_backend!r}, "
            f"url_set={bool(resolved_url)}, org_set={bool(resolved_org)}, "
            f"bucket_set={bool(resolved_bucket)}, token_set={bool(token_value)}"
        )

    return InfluxMarketDataRepository(
        url=resolved_url,
        org=resolved_org,
        bucket=resolved_bucket,
        token=token_value,
    )


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
    """Normalize timeframe token while preserving month/minute semantics.

    Rules:
    - `1m` means 1 minute.
    - `1M` means 1 month.
    - Other units are normalized to lowercase (`s`, `h`, `d`, `w`).
    """
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


def ensure_futures_feature_points_schema(conn: sqlite3.Connection) -> None:
    """Ensure futures feature-points table and indexes are present."""
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {FUTURES_FEATURE_POINTS_TABLE} (
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            funding_rate REAL,
            funding_mark_price REAL,
            mark_price REAL,
            index_price REAL,
            open_interest REAL,
            liquidation_long_qty REAL,
            liquidation_short_qty REAL,
            liquidation_long_notional REAL,
            liquidation_short_notional REAL,
            source TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (exchange, symbol, timestamp_ms)
        );

        CREATE INDEX IF NOT EXISTS idx_futures_feature_points_symbol_time
        ON {FUTURES_FEATURE_POINTS_TABLE}(symbol, timestamp_ms);
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
            normalize_timeframe_token(timeframe),
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
    backend: str | None = None,
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> int | None:
    """Return latest stored compact 1-second bar timestamp for key."""
    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
    if isinstance(repo, InfluxMarketDataRepository):
        return repo.get_last_timestamp_ms(exchange=exchange, symbol=symbol)
    return repo.get_last_ohlcv_1s_timestamp_ms(exchange=exchange, symbol=symbol)


class MarketDataRepository:
    """OOP facade for SQLite-backed market data operations."""

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._prefer_1s_derived = str(
            os.getenv("LQ_PREFER_1S_DERIVED", "1")
        ).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    @staticmethod
    def _rows_to_ohlcv_frame(rows: Sequence[sqlite3.Row]) -> pl.DataFrame:
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

    def _load_derived_ohlcv_from_1s(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        timeframe_token = normalize_timeframe_token(timeframe)
        if timeframe_token == "1s":
            return self.load_ohlcv_1s(
                exchange=exchange,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        tf_ms = int(timeframe_to_milliseconds(timeframe_token))
        if tf_ms <= 1000:
            return _empty_ohlcv_frame()

        conn = connect_market_data_1s_db(self.db_path)
        try:
            ensure_market_ohlcv_1s_schema(conn)
            start_ms = _coerce_timestamp_ms(start_date)
            end_ms = _coerce_timestamp_ms(end_date)

            query = f"""
                WITH filtered AS (
                    SELECT timestamp_ms, open, high, low, close, volume
                    FROM {MARKET_OHLCV_1S_TABLE}
                    WHERE exchange = ? AND symbol = ?
            """
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

            query += """
                ), bucketed AS (
                    SELECT
                        ((timestamp_ms / ?) * ?) AS bucket_ms,
                        MIN(timestamp_ms) AS first_ts,
                        MAX(timestamp_ms) AS last_ts,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        SUM(volume) AS volume
                    FROM filtered
                    GROUP BY bucket_ms
                )
                SELECT
                    b.bucket_ms AS timestamp_ms,
                    o.open AS open,
                    b.high AS high,
                    b.low AS low,
                    c.close AS close,
                    b.volume AS volume
                FROM bucketed b
                JOIN filtered o ON o.timestamp_ms = b.first_ts
                JOIN filtered c ON c.timestamp_ms = b.last_ts
                ORDER BY b.bucket_ms
            """
            params.extend([tf_ms, tf_ms])
            rows = conn.execute(query, params).fetchall()
        finally:
            conn.close()

        return self._rows_to_ohlcv_frame(rows)

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
        if normalize_timeframe_token(timeframe) == "1s":
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
                    normalize_timeframe_token(timeframe),
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
        timeframe_token = normalize_timeframe_token(timeframe)
        if timeframe_token == "1s":
            return self.load_ohlcv_1s(
                exchange=exchange,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        if self._prefer_1s_derived:
            derived = self._load_derived_ohlcv_from_1s(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe_token,
                start_date=start_date,
                end_date=end_date,
            )
            if not derived.is_empty():
                return derived

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
                timeframe_token,
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

        return self._rows_to_ohlcv_frame(rows)

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

        return self._rows_to_ohlcv_frame(rows)

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
    db_path: str | None = None,
    backend: str | None = None,
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> int:
    """Insert or update OHLCV rows idempotently."""
    if not rows:
        if str(backend or "").strip():
            _build_market_data_repository(
                str(db_path or DEFAULT_MARKET_DATA_DB_PATH),
                backend=backend,
                influx_url=influx_url,
                influx_org=influx_org,
                influx_bucket=influx_bucket,
                influx_token=influx_token,
                influx_token_env=influx_token_env,
            )
        return 0

    stream_timeframe = normalize_timeframe_token(timeframe)
    repo = _build_market_data_repository(
        str(db_path or DEFAULT_MARKET_DATA_DB_PATH),
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
    if isinstance(repo, InfluxMarketDataRepository):
        payload = [
            (
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            )
            for row in rows
        ]
        return int(
            repo.write_ohlcv(
                exchange=str(exchange).strip().lower(),
                symbol=normalize_symbol(symbol),
                timeframe=stream_timeframe,
                rows=payload,
            )
        )

    now = datetime.now(UTC).isoformat()
    payload: list[tuple[Any, ...]] = []
    stream_exchange = str(exchange).strip().lower()
    stream_symbol = normalize_symbol(symbol)
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
    backend: str | None = None,
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> int:
    """Insert/update compact 1-second OHLCV rows idempotently."""
    if not rows:
        if str(backend or "").strip():
            _build_market_data_repository(
                db_path,
                backend=backend,
                influx_url=influx_url,
                influx_org=influx_org,
                influx_bucket=influx_bucket,
                influx_token=influx_token,
                influx_token_env=influx_token_env,
            )
        return 0

    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
    if isinstance(repo, InfluxMarketDataRepository):
        payload = [tuple(row) for row in rows]
        return int(repo.write_ohlcv_1s(exchange=exchange, symbol=symbol, rows=payload))

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


def upsert_futures_feature_points(
    conn: sqlite3.Connection,
    *,
    exchange: str,
    symbol: str,
    rows: Sequence[dict[str, Any]],
    source: str = "binance_futures_api",
) -> int:
    """Insert or update futures feature rows idempotently."""
    if not rows:
        return 0

    ensure_futures_feature_points_schema(conn)
    now = datetime.now(UTC).isoformat()
    stream_exchange = str(exchange).strip().lower()
    stream_symbol = normalize_symbol(symbol)
    stream_source = str(source).strip().lower()
    payload: list[tuple[Any, ...]] = []

    for row in rows:
        ts_raw = row.get("timestamp_ms")
        if ts_raw is None:
            continue
        ts = int(ts_raw)
        payload.append(
            (
                stream_exchange,
                stream_symbol,
                ts,
                _utc_iso_from_ms(ts),
                row.get("funding_rate"),
                row.get("funding_mark_price"),
                row.get("mark_price"),
                row.get("index_price"),
                row.get("open_interest"),
                row.get("liquidation_long_qty"),
                row.get("liquidation_short_qty"),
                row.get("liquidation_long_notional"),
                row.get("liquidation_short_notional"),
                stream_source,
                now,
            )
        )

    if not payload:
        return 0

    conn.executemany(
        f"""
        INSERT INTO {FUTURES_FEATURE_POINTS_TABLE}(
            exchange, symbol, timestamp_ms, datetime,
            funding_rate, funding_mark_price, mark_price, index_price, open_interest,
            liquidation_long_qty, liquidation_short_qty,
            liquidation_long_notional, liquidation_short_notional,
            source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(exchange, symbol, timestamp_ms)
        DO UPDATE SET
            funding_rate = COALESCE(excluded.funding_rate, funding_rate),
            funding_mark_price = COALESCE(excluded.funding_mark_price, funding_mark_price),
            mark_price = COALESCE(excluded.mark_price, mark_price),
            index_price = COALESCE(excluded.index_price, index_price),
            open_interest = COALESCE(excluded.open_interest, open_interest),
            liquidation_long_qty = COALESCE(excluded.liquidation_long_qty, liquidation_long_qty),
            liquidation_short_qty = COALESCE(excluded.liquidation_short_qty, liquidation_short_qty),
            liquidation_long_notional = COALESCE(excluded.liquidation_long_notional, liquidation_long_notional),
            liquidation_short_notional = COALESCE(excluded.liquidation_short_notional, liquidation_short_notional),
            source = excluded.source,
            updated_at = excluded.updated_at
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def upsert_futures_feature_points_rows(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    rows: Sequence[dict[str, Any]],
    source: str = "binance_futures_api",
    backend: str | None = None,
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> int:
    """Insert/update futures feature rows into configured storage backend."""
    if not rows:
        if str(backend or "").strip():
            _build_market_data_repository(
                db_path,
                backend=backend,
                influx_url=influx_url,
                influx_org=influx_org,
                influx_bucket=influx_bucket,
                influx_token=influx_token,
                influx_token_env=influx_token_env,
            )
        return 0

    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
    if isinstance(repo, InfluxMarketDataRepository):
        payload: list[dict[str, Any]] = []
        for row in rows:
            ts_raw = row.get("timestamp_ms")
            if ts_raw is None:
                continue
            payload.append(
                {
                    "timestamp_ms": int(ts_raw),
                    "funding_rate": row.get("funding_rate"),
                    "funding_mark_price": row.get("funding_mark_price"),
                    "mark_price": row.get("mark_price"),
                    "index_price": row.get("index_price"),
                    "open_interest": row.get("open_interest"),
                    "liquidation_long_qty": row.get("liquidation_long_qty"),
                    "liquidation_short_qty": row.get("liquidation_short_qty"),
                    "liquidation_long_notional": row.get("liquidation_long_notional"),
                    "liquidation_short_notional": row.get("liquidation_short_notional"),
                }
            )
        return int(
            repo.write_futures_feature_points(
                exchange=str(exchange).strip().lower(),
                symbol=normalize_symbol(symbol),
                rows=payload,
                source=str(source).strip().lower(),
            )
        )

    conn = connect_market_data_db(db_path)
    try:
        ensure_futures_feature_points_schema(conn)
        return int(
            upsert_futures_feature_points(
                conn,
                exchange=exchange,
                symbol=symbol,
                rows=rows,
                source=source,
            )
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
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> pl.DataFrame:
    """Load futures support-feature points from configured storage backend."""
    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
    stream_exchange = str(exchange).strip().lower()
    stream_symbol = normalize_symbol(symbol)
    if isinstance(repo, InfluxMarketDataRepository):
        return repo.load_futures_feature_points(
            exchange=stream_exchange,
            symbol=stream_symbol,
            start_date=start_date,
            end_date=end_date,
        )

    conn = connect_market_data_db(db_path)
    try:
        ensure_futures_feature_points_schema(conn)
        where = ["exchange = ?", "symbol = ?"]
        params: list[Any] = [stream_exchange, stream_symbol]
        if start_date is not None:
            where.append("timestamp_ms >= ?")
            start_ms = _coerce_timestamp_ms(start_date)
            if start_ms is not None:
                params.append(int(start_ms))
            else:
                params.append(0)
        if end_date is not None:
            where.append("timestamp_ms <= ?")
            end_ms = _coerce_timestamp_ms(end_date)
            if end_ms is not None:
                params.append(int(end_ms))
            else:
                params.append(9_223_372_036_854_775_807)
        query = (
            "SELECT datetime, funding_rate, funding_mark_price, mark_price, index_price, "
            "open_interest, liquidation_long_qty, liquidation_short_qty, "
            "liquidation_long_notional, liquidation_short_notional "
            f"FROM {FUTURES_FEATURE_POINTS_TABLE} "
            f"WHERE {' AND '.join(where)} ORDER BY timestamp_ms ASC"
        )
        rows = conn.execute(query, params).fetchall()
        if not rows:
            return pl.DataFrame(
                {
                    "datetime": [],
                    "funding_rate": [],
                    "funding_mark_price": [],
                    "mark_price": [],
                    "index_price": [],
                    "open_interest": [],
                    "liquidation_long_qty": [],
                    "liquidation_short_qty": [],
                    "liquidation_long_notional": [],
                    "liquidation_short_notional": [],
                },
                schema={
                    "datetime": pl.Datetime(time_unit="ms"),
                    "funding_rate": pl.Float64,
                    "funding_mark_price": pl.Float64,
                    "mark_price": pl.Float64,
                    "index_price": pl.Float64,
                    "open_interest": pl.Float64,
                    "liquidation_long_qty": pl.Float64,
                    "liquidation_short_qty": pl.Float64,
                    "liquidation_long_notional": pl.Float64,
                    "liquidation_short_notional": pl.Float64,
                },
            )
        payload = []
        for row in rows:
            payload.append(
                {
                    "datetime": datetime.fromisoformat(str(row["datetime"]).replace("Z", "+00:00")),
                    "funding_rate": row["funding_rate"],
                    "funding_mark_price": row["funding_mark_price"],
                    "mark_price": row["mark_price"],
                    "index_price": row["index_price"],
                    "open_interest": row["open_interest"],
                    "liquidation_long_qty": row["liquidation_long_qty"],
                    "liquidation_short_qty": row["liquidation_short_qty"],
                    "liquidation_long_notional": row["liquidation_long_notional"],
                    "liquidation_short_notional": row["liquidation_short_notional"],
                }
            )
        return pl.DataFrame(payload).sort("datetime")
    finally:
        conn.close()


def market_data_exists(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    backend: str | None = None,
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> bool:
    """Return True if at least one OHLCV row exists for key."""
    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
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
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> tuple[datetime | None, datetime | None, int]:
    """Return (first_datetime, last_datetime, row_count) for one OHLCV stream."""

    def _to_ms(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        text = str(value).strip()
        if not text:
            return None
        try:
            if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                numeric = int(text)
                if abs(numeric) < 100_000_000_000:
                    numeric *= 1000
                return numeric
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            dt = dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
    stream_exchange = str(exchange).strip().lower()
    stream_symbol = normalize_symbol(symbol)
    timeframe_token = normalize_timeframe_token(timeframe)

    if isinstance(repo, InfluxMarketDataRepository):
        first_ms, last_ms, count = repo.get_ohlcv_coverage_between(
            exchange=stream_exchange,
            symbol=stream_symbol,
            timeframe=timeframe_token,
            start_date=start_date,
            end_date=end_date,
        )
        first_dt = (
            datetime.fromtimestamp(int(first_ms) / 1000.0, tz=UTC).replace(tzinfo=None)
            if first_ms is not None
            else None
        )
        last_dt = (
            datetime.fromtimestamp(int(last_ms) / 1000.0, tz=UTC).replace(tzinfo=None)
            if last_ms is not None
            else None
        )
        return first_dt, last_dt, int(count or 0)

    start_ms = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    if timeframe_token == "1s":
        conn = connect_market_data_1s_db(db_path)
        try:
            ensure_market_ohlcv_1s_schema(conn)
            clauses = ["exchange = ?", "symbol = ?"]
            params: list[Any] = [stream_exchange, stream_symbol]
            if start_ms is not None:
                clauses.append("timestamp_ms >= ?")
                params.append(int(start_ms))
            if end_ms is not None:
                clauses.append("timestamp_ms <= ?")
                params.append(int(end_ms))
            row = conn.execute(
                f"""
                SELECT MIN(timestamp_ms), MAX(timestamp_ms), COUNT(*)
                FROM {MARKET_OHLCV_1S_TABLE}
                WHERE {" AND ".join(clauses)}
                """,
                tuple(params),
            ).fetchone()
        finally:
            conn.close()
    else:
        conn = connect_market_data_db(db_path)
        try:
            ensure_market_ohlcv_schema(conn)
            clauses = ["exchange = ?", "symbol = ?", "timeframe = ?"]
            params = [stream_exchange, stream_symbol, timeframe_token]
            if start_ms is not None:
                clauses.append("timestamp_ms >= ?")
                params.append(int(start_ms))
            if end_ms is not None:
                clauses.append("timestamp_ms <= ?")
                params.append(int(end_ms))
            row = conn.execute(
                f"""
                SELECT MIN(timestamp_ms), MAX(timestamp_ms), COUNT(*)
                FROM {MARKET_OHLCV_TABLE}
                WHERE {" AND ".join(clauses)}
                """,
                tuple(params),
            ).fetchone()
        finally:
            conn.close()

    if row is None:
        return None, None, 0
    first_ms_raw, last_ms_raw, count_raw = row
    if first_ms_raw is None or last_ms_raw is None:
        return None, None, 0
    first_dt = datetime.fromtimestamp(int(first_ms_raw) / 1000.0, tz=UTC).replace(tzinfo=None)
    last_dt = datetime.fromtimestamp(int(last_ms_raw) / 1000.0, tz=UTC).replace(tzinfo=None)
    return first_dt, last_dt, int(count_raw or 0)


def load_ohlcv_from_db(
    db_path: str,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
    backend: str | None = None,
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> pl.DataFrame:
    """Load OHLCV data into canonical DataFrame format from configured backend."""
    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
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
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> pl.DataFrame:
    """Load compact 1-second OHLCV rows into canonical DataFrame format."""
    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
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
    backend: str | None = None,
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> dict[str, pl.DataFrame]:
    """Load symbol->DataFrame dictionary from configured market-data backend."""
    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
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
    influx_url: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
    influx_token: str | None = None,
    influx_token_env: str = "INFLUXDB_TOKEN",
) -> int:
    """Export OHLCV from configured backend into CSV. Returns exported row count."""
    repo = _build_market_data_repository(
        db_path,
        backend=backend,
        influx_url=influx_url,
        influx_org=influx_org,
        influx_bucket=influx_bucket,
        influx_token=influx_token,
        influx_token_env=influx_token_env,
    )
    return repo.export_ohlcv_to_csv(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
    )
