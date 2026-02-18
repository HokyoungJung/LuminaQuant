"""Binance OHLCV synchronization helpers."""

from __future__ import annotations

import csv
import io
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import ccxt
from lumina_quant.market_data import (
    MARKET_OHLCV_1S_TABLE,
    MARKET_OHLCV_TABLE,
    connect_market_data_1s_db,
    connect_market_data_db,
    ensure_market_ohlcv_1s_schema,
    ensure_market_ohlcv_schema,
    export_ohlcv_to_csv,
    get_last_ohlcv_1s_timestamp_ms,
    get_last_ohlcv_timestamp_ms,
    normalize_symbol,
    symbol_csv_filename,
    timeframe_to_milliseconds,
    upsert_ohlcv_rows,
    upsert_ohlcv_rows_1s,
)


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def parse_timestamp_input(value: str | int | float | None) -> int | None:
    """Parse timestamp inputs in ISO8601/seconds/milliseconds into milliseconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            return numeric * 1000
        return numeric

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return parse_timestamp_input(int(text))
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def create_binance_exchange(
    *,
    api_key: str = "",
    secret_key: str = "",
    market_type: str = "spot",
    testnet: bool = False,
) -> ccxt.Exchange:
    """Create a configured Binance CCXT client."""
    exchange = ccxt.binance()
    exchange.enableRateLimit = True
    if api_key:
        exchange.apiKey = api_key
    if secret_key:
        exchange.secret = secret_key
    if str(market_type).lower() == "future":
        options = dict(getattr(exchange, "options", {}) or {})
        options["defaultType"] = "future"
        exchange.options = options
    if testnet:
        exchange.set_sandbox_mode(True)
    return exchange


def _fetch_ohlcv_with_retry(
    exchange: Any,
    symbol: str,
    timeframe: str,
    *,
    since_ms: int,
    limit: int,
    retries: int,
    base_wait_sec: float,
) -> list[list[float]]:
    wait = max(0.1, float(base_wait_sec))
    attempt = 0
    while True:
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except Exception:
            attempt += 1
            if attempt > max(0, int(retries)):
                raise
            time.sleep(wait)
            wait = min(wait * 2.0, 10.0)


def _fetch_trades_with_retry(
    exchange: Any,
    symbol: str,
    *,
    since_ms: int,
    limit: int,
    retries: int,
    base_wait_sec: float,
) -> list[dict[str, Any]]:
    wait = max(0.1, float(base_wait_sec))
    attempt = 0
    while True:
        try:
            return list(exchange.fetch_trades(symbol, since=since_ms, limit=limit))
        except Exception:
            attempt += 1
            if attempt > max(0, int(retries)):
                raise
            time.sleep(wait)
            wait = min(wait * 2.0, 10.0)


def _normalize_trades_to_1s_ohlcv(
    trades: Sequence[dict[str, Any]],
    *,
    cursor_ms: int,
    until_ms: int,
    previous_close: float | None,
) -> tuple[list[tuple[float, float, float, float, float, float]], int | None, float | None]:
    """Aggregate trades into 1-second OHLCV rows.

    Missing seconds inside the observed trade span are forward-filled with
    previous close and zero volume to keep a continuous 1s timeline.
    """
    buckets: dict[int, list[float]] = {}
    last_trade_ts: int | None = None

    for trade in trades:
        ts_raw = trade.get("timestamp")
        if ts_raw is None:
            continue
        ts = int(ts_raw)
        if ts < int(cursor_ms):
            continue
        if ts > int(until_ms):
            break

        price_raw = trade.get("price")
        if price_raw is None:
            continue
        price = float(price_raw)
        if price <= 0.0:
            continue

        amount_raw = trade.get("amount")
        volume = float(amount_raw) if amount_raw is not None else 0.0
        if volume < 0.0:
            volume = 0.0

        sec_ts = (ts // 1000) * 1000
        current = buckets.get(sec_ts)
        if current is None:
            buckets[sec_ts] = [price, price, price, price, volume]
        else:
            current[1] = max(current[1], price)
            current[2] = min(current[2], price)
            current[3] = price
            current[4] += volume

        if last_trade_ts is None or ts > last_trade_ts:
            last_trade_ts = ts

    if not buckets:
        return [], last_trade_ts, previous_close

    sorted_seconds = sorted(buckets.keys())
    first_second = max((int(cursor_ms) // 1000) * 1000, sorted_seconds[0])
    last_second = min((int(until_ms) // 1000) * 1000, sorted_seconds[-1])

    out: list[tuple[float, float, float, float, float, float]] = []
    prev_close = previous_close
    sec = first_second
    while sec <= last_second:
        row = buckets.get(sec)
        if row is None:
            if prev_close is None:
                sec += 1000
                continue
            o = prev_close
            h = prev_close
            low_price = prev_close
            c = prev_close
            v = 0.0
        else:
            o = float(row[0])
            h = float(row[1])
            low_price = float(row[2])
            c = float(row[3])
            v = float(row[4])
            prev_close = c
        out.append((float(sec), o, h, low_price, c, v))
        sec += 1000

    return out, last_trade_ts, prev_close


def _date_from_ms(timestamp_ms: int) -> date:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).date()


def _day_bounds_ms(day_value: date) -> tuple[int, int]:
    day_start = datetime(day_value.year, day_value.month, day_value.day, tzinfo=UTC)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = start_ms + 86_399_000
    return start_ms, end_ms


def _iter_days(start_ms: int, end_ms: int) -> list[date]:
    start_day = _date_from_ms(start_ms)
    end_day = _date_from_ms(end_ms)
    out: list[date] = []
    cur = start_day
    while cur <= end_day:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def _binance_archive_url(symbol: str, day_value: date, market_type: str) -> str:
    compact = normalize_symbol(symbol).replace("/", "")
    d = day_value.strftime("%Y-%m-%d")
    if str(market_type).strip().lower() == "future":
        return (
            "https://data.binance.vision/data/futures/um/daily/aggTrades/"
            f"{compact}/{compact}-aggTrades-{d}.zip"
        )
    return f"https://data.binance.vision/data/spot/daily/klines/{compact}/1s/{compact}-1s-{d}.zip"


def _download_zip_bytes(
    url: str,
    *,
    retries: int,
    base_wait_sec: float,
) -> bytes | None:
    wait = max(0.1, float(base_wait_sec))
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if int(getattr(exc, "code", 0)) == 404:
                return None
            attempt += 1
            if attempt > max(0, int(retries)):
                raise
            time.sleep(wait)
            wait = min(wait * 2.0, 10.0)
        except Exception:
            attempt += 1
            if attempt > max(0, int(retries)):
                raise
            time.sleep(wait)
            wait = min(wait * 2.0, 10.0)


def _archive_rows_to_1s_ohlcv(
    zip_blob: bytes,
    *,
    market_type: str,
    cursor_ms: int,
    until_ms: int,
    previous_close: float | None,
) -> tuple[list[tuple[float, float, float, float, float, float]], float | None]:
    """Convert Binance archive day file into 1-second OHLCV rows."""
    buckets: dict[int, list[float]] = {}

    with zipfile.ZipFile(io.BytesIO(zip_blob)) as zf:
        names = zf.namelist()
        if not names:
            return [], previous_close
        with zf.open(names[0], "r") as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8")
            reader = csv.reader(text_file)
            if str(market_type).strip().lower() == "future":
                # aggTrades: agg_id,price,qty,first_id,last_id,timestamp,is_buyer_maker[,best_match]
                for row in reader:
                    if len(row) < 6:
                        continue
                    try:
                        ts = int(row[5])
                        price = float(row[1])
                        qty = float(row[2])
                    except Exception:
                        continue
                    if ts < int(cursor_ms) or ts > int(until_ms):
                        continue
                    sec_ts = (ts // 1000) * 1000
                    current = buckets.get(sec_ts)
                    if current is None:
                        buckets[sec_ts] = [price, price, price, price, max(0.0, qty)]
                    else:
                        current[1] = max(current[1], price)
                        current[2] = min(current[2], price)
                        current[3] = price
                        current[4] += max(0.0, qty)
            else:
                # spot klines 1s: open_time,open,high,low,close,volume,...
                for row in reader:
                    if len(row) < 6:
                        continue
                    try:
                        ts = int(row[0])
                        o = float(row[1])
                        h = float(row[2])
                        low_price = float(row[3])
                        c = float(row[4])
                        v = float(row[5])
                    except Exception:
                        continue
                    if ts < int(cursor_ms) or ts > int(until_ms):
                        continue
                    buckets[ts] = [o, h, low_price, c, max(0.0, v)]

    if not buckets:
        return [], previous_close

    sorted_seconds = sorted(buckets.keys())
    first_second = max((int(cursor_ms) // 1000) * 1000, sorted_seconds[0])
    last_second = min((int(until_ms) // 1000) * 1000, sorted_seconds[-1])

    out: list[tuple[float, float, float, float, float, float]] = []
    prev_close = previous_close
    sec = first_second
    while sec <= last_second:
        row = buckets.get(sec)
        if row is None:
            if prev_close is None:
                sec += 1000
                continue
            o = prev_close
            h = prev_close
            low_price = prev_close
            c = prev_close
            v = 0.0
        else:
            o = float(row[0])
            h = float(row[1])
            low_price = float(row[2])
            c = float(row[3])
            v = float(row[4])
            prev_close = c
        out.append((float(sec), o, h, low_price, c, v))
        sec += 1000

    return out, prev_close


def _normalize_ohlcv_batch(
    batch: Sequence[Sequence[Any]],
    *,
    cursor_ms: int,
    until_ms: int,
) -> list[tuple[float, float, float, float, float, float]]:
    out: list[tuple[float, float, float, float, float, float]] = []
    last_ts = cursor_ms - 1
    for row in batch:
        if len(row) < 6:
            continue
        ts = int(row[0])
        if ts < cursor_ms or ts <= last_ts:
            continue
        if ts > until_ms:
            break
        o = float(row[1])
        h = float(row[2])
        low_price = float(row[3])
        c = float(row[4])
        v = float(row[5])
        out.append((float(ts), o, h, low_price, c, v))
        last_ts = ts
    return out


@dataclass(slots=True)
class SyncStats:
    """Synchronization result summary for a symbol."""

    symbol: str
    fetched_rows: int
    upserted_rows: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None


@dataclass(slots=True)
class SyncRequest:
    """Container for symbol sync boundaries and pagination parameters."""

    symbol: str
    timeframe: str
    start_ms: int
    end_ms: int
    limit: int = 1000
    max_batches: int = 100_000
    retries: int = 3
    base_wait_sec: float = 0.5


class MarketDataSyncService:
    """OOP facade for market data synchronization workflows."""

    def __init__(self, *, exchange: Any, db_path: str, exchange_id: str):
        self.exchange = exchange
        self.db_path = str(db_path)
        self.exchange_id = str(exchange_id)

    def get_symbol_coverage(
        self, *, symbol: str, timeframe: str
    ) -> tuple[int | None, int | None, int]:
        return get_symbol_ohlcv_coverage(
            db_path=self.db_path,
            exchange_id=self.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
        )

    def sync_symbol(self, request: SyncRequest) -> SyncStats:
        return sync_symbol_ohlcv(
            exchange=self.exchange,
            db_path=self.db_path,
            exchange_id=self.exchange_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_ms=request.start_ms,
            end_ms=request.end_ms,
            limit=request.limit,
            max_batches=request.max_batches,
            retries=request.retries,
            base_wait_sec=request.base_wait_sec,
        )

    def ensure_coverage(
        self,
        *,
        symbol_list: Sequence[str],
        timeframe: str,
        since_ms: int | None = None,
        until_ms: int | None = None,
        force_full: bool = False,
        limit: int = 1000,
        max_batches: int = 100_000,
        retries: int = 3,
        base_wait_sec: float = 0.5,
        export_csv_dir: str | None = None,
    ) -> list[SyncStats]:
        return ensure_market_data_coverage(
            exchange=self.exchange,
            db_path=self.db_path,
            exchange_id=self.exchange_id,
            symbol_list=symbol_list,
            timeframe=timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            force_full=force_full,
            limit=limit,
            max_batches=max_batches,
            retries=retries,
            base_wait_sec=base_wait_sec,
            export_csv_dir=export_csv_dir,
        )

    def sync_many(
        self,
        *,
        symbol_list: Sequence[str],
        timeframe: str,
        since_ms: int | None = None,
        until_ms: int | None = None,
        force_full: bool = False,
        limit: int = 1000,
        max_batches: int = 100_000,
        retries: int = 3,
        base_wait_sec: float = 0.5,
        export_csv_dir: str | None = None,
    ) -> list[SyncStats]:
        return sync_market_data(
            exchange=self.exchange,
            db_path=self.db_path,
            exchange_id=self.exchange_id,
            symbol_list=symbol_list,
            timeframe=timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            force_full=force_full,
            limit=limit,
            max_batches=max_batches,
            retries=retries,
            base_wait_sec=base_wait_sec,
            export_csv_dir=export_csv_dir,
        )


def get_symbol_ohlcv_coverage(
    *,
    db_path: str,
    exchange_id: str,
    symbol: str,
    timeframe: str,
) -> tuple[int | None, int | None, int]:
    """Return (first_ts, last_ts, row_count) for one OHLCV stream key."""
    if str(timeframe).strip().lower() == "1s":
        conn = connect_market_data_1s_db(db_path)
        try:
            ensure_market_ohlcv_1s_schema(conn)
            row = conn.execute(
                f"""
                SELECT
                    MIN(timestamp_ms) AS min_ts,
                    MAX(timestamp_ms) AS max_ts,
                    COUNT(*) AS row_count
                FROM {MARKET_OHLCV_1S_TABLE}
                WHERE exchange = ? AND symbol = ?
                """,
                (
                    str(exchange_id).strip().lower(),
                    normalize_symbol(symbol),
                ),
            ).fetchone()
            if row is None:
                return None, None, 0
            first_ts = int(row["min_ts"]) if row["min_ts"] is not None else None
            last_ts = int(row["max_ts"]) if row["max_ts"] is not None else None
            row_count = int(row["row_count"] or 0)
            return first_ts, last_ts, row_count
        finally:
            conn.close()

    conn = connect_market_data_db(db_path)
    try:
        ensure_market_ohlcv_schema(conn)
        row = conn.execute(
            f"""
            SELECT
                MIN(timestamp_ms) AS min_ts,
                MAX(timestamp_ms) AS max_ts,
                COUNT(*) AS row_count
            FROM {MARKET_OHLCV_TABLE}
            WHERE exchange = ? AND symbol = ? AND timeframe = ?
            """,
            (
                str(exchange_id).strip().lower(),
                normalize_symbol(symbol),
                str(timeframe).strip().lower(),
            ),
        ).fetchone()
        if row is None:
            return None, None, 0
        first_ts = int(row["min_ts"]) if row["min_ts"] is not None else None
        last_ts = int(row["max_ts"]) if row["max_ts"] is not None else None
        row_count = int(row["row_count"] or 0)
        return first_ts, last_ts, row_count
    finally:
        conn.close()


def ensure_market_data_coverage(
    *,
    exchange: Any,
    db_path: str,
    exchange_id: str,
    symbol_list: Sequence[str],
    timeframe: str,
    since_ms: int | None = None,
    until_ms: int | None = None,
    force_full: bool = False,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
    export_csv_dir: str | None = None,
) -> list[SyncStats]:
    """Ensure DB has OHLCV coverage for [since_ms, until_ms] across symbols.

    Unlike tail-only sync, this fills both head and tail gaps.
    """
    effective_until = int(until_ms) if until_ms is not None else _now_ms()
    effective_since = (
        int(since_ms)
        if since_ms is not None
        else int(datetime(2017, 1, 1, tzinfo=UTC).timestamp() * 1000)
    )
    effective_since = max(0, effective_since)
    if effective_until < effective_since:
        effective_until = effective_since

    tf_ms = timeframe_to_milliseconds(timeframe)
    summaries: list[SyncStats] = []

    for symbol in symbol_list:
        stream_symbol = normalize_symbol(symbol)
        first_ts, last_ts, row_count = get_symbol_ohlcv_coverage(
            db_path=db_path,
            exchange_id=exchange_id,
            symbol=stream_symbol,
            timeframe=timeframe,
        )

        windows: list[tuple[int, int]] = []
        if force_full or row_count <= 0 or first_ts is None or last_ts is None:
            windows.append((effective_since, effective_until))
        else:
            head_gap_end = min(effective_until, first_ts - tf_ms)
            if effective_since <= head_gap_end:
                windows.append((effective_since, head_gap_end))

            tail_gap_start = max(effective_since, last_ts + tf_ms)
            if tail_gap_start <= effective_until:
                windows.append((tail_gap_start, effective_until))

        fetched_rows = 0
        upserted_rows = 0
        for start_ms, end_ms in windows:
            if start_ms > end_ms:
                continue
            part = sync_symbol_ohlcv(
                exchange=exchange,
                db_path=db_path,
                exchange_id=exchange_id,
                symbol=stream_symbol,
                timeframe=timeframe,
                start_ms=start_ms,
                end_ms=end_ms,
                limit=limit,
                max_batches=max_batches,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            fetched_rows += int(part.fetched_rows)
            upserted_rows += int(part.upserted_rows)

        final_first, final_last, _ = get_symbol_ohlcv_coverage(
            db_path=db_path,
            exchange_id=exchange_id,
            symbol=stream_symbol,
            timeframe=timeframe,
        )
        summaries.append(
            SyncStats(
                symbol=stream_symbol,
                fetched_rows=fetched_rows,
                upserted_rows=upserted_rows,
                first_timestamp_ms=final_first,
                last_timestamp_ms=final_last,
            )
        )

        if export_csv_dir:
            csv_path = f"{export_csv_dir}/{symbol_csv_filename(stream_symbol)}"
            export_ohlcv_to_csv(
                db_path,
                exchange=str(exchange_id).lower(),
                symbol=stream_symbol,
                timeframe=timeframe,
                csv_path=csv_path,
            )

    return summaries


def sync_symbol_ohlcv(
    *,
    exchange: Any,
    db_path: str,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
) -> SyncStats:
    """Synchronize one symbol OHLCV range into SQLite with idempotent upserts."""
    conn = connect_market_data_db(db_path)
    try:
        ensure_market_ohlcv_schema(conn)
        tf_ms = timeframe_to_milliseconds(timeframe)
        stream_symbol = normalize_symbol(symbol)
        cursor = max(0, int(start_ms))
        until = max(cursor, int(end_ms))

        fetched_rows = 0
        upserted_rows = 0
        first_ts = None
        last_ts = None
        previous_close: float | None = None
        use_trade_fallback = False
        empty_trade_advance_ms = 86_400_000

        is_one_second = str(timeframe).strip().lower() == "1s"
        exchange_name = str(getattr(exchange, "id", "")).strip().lower()
        market_type = str(
            (getattr(exchange, "options", {}) or {}).get("defaultType", "spot")
        ).lower()

        if is_one_second and "binance" in exchange_name:
            # Binance archive is typically delayed; keep recent tail for API sync.
            archive_until = min(until, _now_ms() - (2 * 86_400_000))
            if cursor <= archive_until:
                for day_value in _iter_days(cursor, archive_until):
                    day_start_ms, day_end_ms = _day_bounds_ms(day_value)
                    range_start = max(cursor, day_start_ms)
                    range_end = min(archive_until, day_end_ms)
                    if range_start > range_end:
                        continue

                    url = _binance_archive_url(stream_symbol, day_value, market_type)
                    blob = _download_zip_bytes(
                        url,
                        retries=retries,
                        base_wait_sec=base_wait_sec,
                    )
                    if blob is None:
                        continue

                    archive_rows, previous_close = _archive_rows_to_1s_ohlcv(
                        blob,
                        market_type=market_type,
                        cursor_ms=range_start,
                        until_ms=range_end,
                        previous_close=previous_close,
                    )
                    if not archive_rows:
                        continue

                    fetched_rows += len(archive_rows)
                    if first_ts is None:
                        first_ts = int(archive_rows[0][0])
                    last_ts = int(archive_rows[-1][0])
                    previous_close = float(archive_rows[-1][4])
                    upserted_rows += upsert_ohlcv_rows_1s(
                        db_path,
                        exchange=str(exchange_id).lower(),
                        symbol=stream_symbol,
                        rows=archive_rows,
                    )

                    next_cursor = last_ts + 1000
                    if next_cursor > cursor:
                        cursor = next_cursor

                if cursor <= archive_until:
                    cursor = archive_until + 1000

        batch_count = 0
        while cursor <= until and batch_count < max(1, int(max_batches)):
            batch_count += 1
            normalized: list[tuple[float, float, float, float, float, float]] = []
            last_trade_ts: int | None = None

            if is_one_second:
                if not use_trade_fallback:
                    try:
                        raw_rows = _fetch_ohlcv_with_retry(
                            exchange,
                            stream_symbol,
                            timeframe,
                            since_ms=cursor,
                            limit=max(1, int(limit)),
                            retries=retries,
                            base_wait_sec=base_wait_sec,
                        )
                    except Exception:
                        use_trade_fallback = True
                        raw_rows = []
                else:
                    raw_rows = []

                if raw_rows:
                    normalized = _normalize_ohlcv_batch(raw_rows, cursor_ms=cursor, until_ms=until)
                else:
                    trades = _fetch_trades_with_retry(
                        exchange,
                        stream_symbol,
                        since_ms=cursor,
                        limit=max(1, int(limit)),
                        retries=retries,
                        base_wait_sec=base_wait_sec,
                    )
                    if not trades:
                        cursor += empty_trade_advance_ms
                        continue
                    normalized, last_trade_ts, previous_close = _normalize_trades_to_1s_ohlcv(
                        trades,
                        cursor_ms=cursor,
                        until_ms=until,
                        previous_close=previous_close,
                    )
            else:
                raw_rows = _fetch_ohlcv_with_retry(
                    exchange,
                    stream_symbol,
                    timeframe,
                    since_ms=cursor,
                    limit=max(1, int(limit)),
                    retries=retries,
                    base_wait_sec=base_wait_sec,
                )
                if not raw_rows:
                    break
                normalized = _normalize_ohlcv_batch(raw_rows, cursor_ms=cursor, until_ms=until)

            if not normalized:
                if is_one_second and last_trade_ts is not None:
                    cursor = max(cursor + 1000, int(last_trade_ts) + 1)
                else:
                    cursor += tf_ms
                continue

            fetched_rows += len(normalized)
            if first_ts is None:
                first_ts = int(normalized[0][0])
            last_ts = int(normalized[-1][0])
            previous_close = float(normalized[-1][4])

            if is_one_second:
                upserted_rows += upsert_ohlcv_rows_1s(
                    db_path,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                    rows=normalized,
                )
            else:
                upserted_rows += upsert_ohlcv_rows(
                    conn,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                    timeframe=timeframe,
                    rows=normalized,
                    source="binance_sync",
                )

            next_cursor = last_ts + tf_ms
            if next_cursor <= cursor:
                cursor += tf_ms
            else:
                cursor = next_cursor

            rate_limit_sec = float(getattr(exchange, "rateLimit", 0) or 0) / 1000.0
            if rate_limit_sec > 0:
                time.sleep(rate_limit_sec)

        return SyncStats(
            symbol=stream_symbol,
            fetched_rows=fetched_rows,
            upserted_rows=upserted_rows,
            first_timestamp_ms=first_ts,
            last_timestamp_ms=last_ts,
        )
    finally:
        conn.close()


def sync_market_data(
    *,
    exchange: Any,
    db_path: str,
    exchange_id: str,
    symbol_list: Sequence[str],
    timeframe: str,
    since_ms: int | None = None,
    until_ms: int | None = None,
    force_full: bool = False,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
    export_csv_dir: str | None = None,
) -> list[SyncStats]:
    """Synchronize OHLCV for multiple symbols and optionally export CSV copies."""
    effective_until = int(until_ms) if until_ms is not None else _now_ms()
    default_since = (
        int(since_ms)
        if since_ms is not None
        else int(datetime(2017, 1, 1, tzinfo=UTC).timestamp() * 1000)
    )

    conn = connect_market_data_db(db_path)
    try:
        ensure_market_ohlcv_schema(conn)
        stats: list[SyncStats] = []

        for symbol in symbol_list:
            stream_symbol = normalize_symbol(symbol)
            if str(timeframe).strip().lower() == "1s":
                last_ts = get_last_ohlcv_1s_timestamp_ms(
                    db_path,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                )
            else:
                last_ts = get_last_ohlcv_timestamp_ms(
                    conn,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                    timeframe=timeframe,
                )
            start_ms = default_since
            if last_ts is not None and not force_full:
                start_ms = last_ts + timeframe_to_milliseconds(timeframe)

            stat = sync_symbol_ohlcv(
                exchange=exchange,
                db_path=db_path,
                exchange_id=exchange_id,
                symbol=stream_symbol,
                timeframe=timeframe,
                start_ms=start_ms,
                end_ms=effective_until,
                limit=limit,
                max_batches=max_batches,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            stats.append(stat)

            if export_csv_dir:
                csv_path = f"{export_csv_dir}/{symbol_csv_filename(stream_symbol)}"
                export_ohlcv_to_csv(
                    db_path,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                    timeframe=timeframe,
                    csv_path=csv_path,
                )
        return stats
    finally:
        conn.close()
