"""Binance OHLCV synchronization helpers."""

from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.data.raw_first_lineage import raw_aggtrades_to_1s_frame, resample_1s_frame
from lumina_quant.exchanges.binance_futures_client import (
    BinanceFuturesClientConfig,
    BinanceFuturesRESTClient,
)
from lumina_quant.market_data import (
    connect_market_data_db,
    export_ohlcv_to_csv,
    get_last_ohlcv_1s_timestamp_ms,
    get_last_ohlcv_timestamp_ms,
    load_ohlcv_coverage_from_db,
    normalize_symbol,
    normalize_timeframe_token,
    symbol_csv_filename,
    timeframe_to_milliseconds,
    upsert_futures_feature_points_rows,
    upsert_ohlcv_rows,
    upsert_ohlcv_rows_1s,
)

_DEFAULT_RAW_ARCHIVE_CHUNK_ROWS = 250_000


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _is_local_storage(db_path: str, *, backend: str | None = None) -> bool:
    _ = (db_path, backend)
    return True


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


def create_binance_futures_client(
    *,
    api_key: str = "",
    secret_key: str = "",
    market_type: str = "future",
    testnet: bool = False,
) -> Any:
    """Create a native Binance USDⓈ-M Futures REST client."""
    if str(market_type or "future").strip().lower() != "future":
        raise ValueError("Binance historical sync supports USDⓈ-M futures only.")
    return BinanceFuturesRESTClient(
        BinanceFuturesClientConfig(
            api_key=str(api_key or ""),
            secret_key=str(secret_key or ""),
            testnet=bool(testnet),
        )
    )


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
            fetch_fn = getattr(exchange, "agg_trades", None)
            if callable(fetch_fn):
                rows = fetch_fn(
                    symbol=symbol,
                    start_time=int(since_ms),
                    end_time=min(int(since_ms) + 3_599_999, _now_ms()),
                    limit=max(1, min(int(limit), 1_000)),
                )
                return [
                    {
                        "id": int(row.get("a") or 0),
                        "timestamp": int(row.get("T") or 0),
                        "price": float(row.get("p") or 0.0),
                        "amount": float(row.get("q") or 0.0),
                        "side": "sell" if bool(row.get("m")) else "buy",
                        "isBuyerMaker": bool(row.get("m")),
                        "info": dict(row or {}),
                    }
                    for row in list(rows or [])
                ]
            return list(exchange.fetch_trades(symbol, since=since_ms, limit=limit) or [])
        except Exception:
            attempt += 1
            if attempt > max(0, int(retries)):
                raise
            time.sleep(wait)
            wait = min(wait * 2.0, 10.0)


def _date_from_ms(timestamp_ms: int) -> date:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).date()


def _day_bounds_ms(day_value: date) -> tuple[int, int]:
    day_start = datetime(day_value.year, day_value.month, day_value.day, tzinfo=UTC)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = start_ms + 86_400_000 - 1
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


def _last_1s_close(
    *,
    db_path: str,
    exchange_id: str,
    symbol: str,
    before_ms: int,
) -> float | None:
    if int(before_ms) < 0:
        return None
    repo = None
    try:
        from lumina_quant.storage.parquet import ParquetMarketDataRepository

        repo = ParquetMarketDataRepository(str(db_path))
        frame = repo.load_ohlcv(
            exchange=str(exchange_id).lower(),
            symbol=normalize_symbol(symbol),
            timeframe="1s",
            start_date=datetime.fromtimestamp(int(before_ms) / 1000.0, tz=UTC),
            end_date=datetime.fromtimestamp(int(before_ms) / 1000.0, tz=UTC),
        )
    except Exception:
        return None
    if frame is None or frame.is_empty():
        return None
    try:
        return float(frame["close"][-1])
    except Exception:
        return None


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


@dataclass(slots=True)
class FuturesFeatureSyncStats:
    """Synchronization summary for futures feature points."""

    symbol: str
    upserted_rows: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None


@dataclass(slots=True)
class RawAggTradesSyncStats:
    """Raw aggTrades synchronization summary for one symbol."""

    symbol: str
    fetched_rows: int
    upserted_rows: int
    first_timestamp_ms: int | None
    last_timestamp_ms: int | None
    checkpoint_timestamp_ms: int | None
    checkpoint_trade_id: int | None


def _normalize_raw_trade_row(row: dict[str, Any], *, fallback_id: int = 0) -> dict[str, Any] | None:
    timestamp_raw = row.get("timestamp")
    if timestamp_raw is None:
        return None
    try:
        timestamp_ms = int(timestamp_raw)
    except Exception:
        return None

    trade_id_raw = row.get("id", row.get("a", fallback_id))
    try:
        trade_id = int(trade_id_raw)
    except Exception:
        trade_id = int(fallback_id)

    price_raw = row.get("price")
    amount_raw = row.get("amount")
    if price_raw is None or amount_raw is None:
        return None
    try:
        price = float(price_raw)
        quantity = float(amount_raw)
    except Exception:
        return None
    if price <= 0.0 or quantity < 0.0:
        return None

    side_raw = row.get("side")
    is_buyer_maker = bool(
        row.get("isBuyerMaker")
        if "isBuyerMaker" in row
        else str(side_raw).strip().lower() == "sell"
    )

    return {
        "agg_trade_id": int(trade_id),
        "timestamp_ms": int(timestamp_ms),
        "price": float(price),
        "quantity": float(quantity),
        "is_buyer_maker": bool(is_buyer_maker),
    }


def _raw_archive_chunk_rows() -> int:
    raw = str(os.getenv("LQ_RAW_ARCHIVE_CHUNK_ROWS", "") or "").strip()
    if raw:
        try:
            return max(1_000, int(raw))
        except ValueError:
            pass
    return _DEFAULT_RAW_ARCHIVE_CHUNK_ROWS


def _iter_archive_rows_to_raw_aggtrades(
    zip_blob: bytes,
    *,
    cursor_ms: int,
    until_ms: int,
    chunk_rows: int | None = None,
):
    """Yield raw aggTrade archive rows in bounded chunks.

    Binance daily aggTrades archives can contain millions of rows.  Building a
    full-day ``list[dict]`` can push RSS above 8 GB on busy ETH/SOL days; the
    backfill path only needs append-ordered chunks, so parse and commit bounded
    pieces instead.
    """
    rows: list[dict[str, Any]] = []
    max_rows = max(1, int(chunk_rows or _raw_archive_chunk_rows()))
    with zipfile.ZipFile(io.BytesIO(zip_blob)) as zf:
        names = zf.namelist()
        if not names:
            return
        with zf.open(names[0], "r") as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding="utf-8")
            reader = csv.reader(text_file)
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    agg_trade_id = int(row[0])
                    price = float(row[1])
                    quantity = float(row[2])
                    timestamp_ms = int(row[5])
                except Exception:
                    continue
                if price <= 0.0 or quantity < 0.0:
                    continue
                if timestamp_ms < int(cursor_ms) or timestamp_ms > int(until_ms):
                    continue
                rows.append(
                    {
                        "agg_trade_id": agg_trade_id,
                        "timestamp_ms": timestamp_ms,
                        "price": price,
                        "quantity": max(0.0, quantity),
                        "is_buyer_maker": str(row[6]).strip().lower() in {"1", "true", "t", "yes"},
                    }
                )
                if len(rows) >= max_rows:
                    rows.sort(
                        key=lambda item: (int(item["timestamp_ms"]), int(item["agg_trade_id"]))
                    )
                    yield rows
                    rows = []
    rows.sort(key=lambda item: (int(item["timestamp_ms"]), int(item["agg_trade_id"])))
    if rows:
        yield rows


def _archive_rows_to_raw_aggtrades(
    zip_blob: bytes,
    *,
    cursor_ms: int,
    until_ms: int,
) -> list[dict[str, Any]]:
    """Return archive rows for tests/small callers.

    Production sync uses ``_iter_archive_rows_to_raw_aggtrades`` to avoid
    materializing high-volume daily archives in memory.
    """
    chunks = list(
        _iter_archive_rows_to_raw_aggtrades(
            zip_blob,
            cursor_ms=cursor_ms,
            until_ms=until_ms,
            chunk_rows=_raw_archive_chunk_rows(),
        )
    )
    if not chunks:
        return []
    return [row for chunk in chunks for row in chunk]


def sync_symbol_aggtrades_raw(
    *,
    exchange: Any,
    db_path: str,
    exchange_id: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    max_batches: int = 100_000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
    resume_from_checkpoint: bool = True,
) -> RawAggTradesSyncStats:
    """Collect Binance aggTrades into raw parquet partitions with checkpoint resume."""
    from lumina_quant.storage.parquet import ParquetMarketDataRepository

    repo = ParquetMarketDataRepository(str(db_path))
    stream_exchange = str(exchange_id).strip().lower() or "binance"
    stream_symbol = normalize_symbol(symbol)

    cursor = max(0, int(start_ms))
    until = max(cursor, int(end_ms))
    last_trade_id = -1
    if bool(resume_from_checkpoint):
        checkpoint = repo.read_raw_checkpoint(exchange=stream_exchange, symbol=stream_symbol)
        try:
            checkpoint_ts = int(checkpoint.get("last_timestamp_ms", 0) or 0)
        except Exception:
            checkpoint_ts = 0
        try:
            checkpoint_trade_id = int(checkpoint.get("last_trade_id", -1) or -1)
        except Exception:
            checkpoint_trade_id = -1
        if checkpoint_ts > cursor:
            cursor = checkpoint_ts
            last_trade_id = checkpoint_trade_id

    fetched_rows = 0
    upserted_rows = 0
    first_ts = None
    last_ts = None

    def _dedupe_new_rows(
        rows: list[dict[str, Any]],
        *,
        cursor_ms: int,
        last_trade_id_seen: int,
    ) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for item in rows:
            ts = int(item["timestamp_ms"])
            trade_id = int(item["agg_trade_id"])
            if ts < int(cursor_ms) or ts > int(until):
                continue
            if ts == int(cursor_ms) and trade_id <= int(last_trade_id_seen):
                continue
            key = (ts, trade_id)
            if key in seen:
                continue
            seen.add(key)
            normalized_rows.append(item)
        normalized_rows.sort(
            key=lambda item: (int(item["timestamp_ms"]), int(item["agg_trade_id"]))
        )
        return normalized_rows

    def _commit_batch(rows: list[dict[str, Any]], *, observed_until_ms: int) -> None:
        nonlocal fetched_rows, upserted_rows, first_ts, last_ts, last_trade_id, cursor
        if not rows:
            return
        fetched_rows += len(rows)
        upserted_rows += int(
            repo.append_raw_aggtrades(
                exchange=stream_exchange,
                symbol=stream_symbol,
                rows=rows,
            )
        )
        first_batch_ts = int(rows[0]["timestamp_ms"])
        first_ts = first_batch_ts if first_ts is None else min(first_ts, first_batch_ts)
        last_ts = int(rows[-1]["timestamp_ms"])
        last_trade_id = int(rows[-1]["agg_trade_id"])
        cursor = int(last_ts)

        checkpoint_payload = {
            "exchange": stream_exchange,
            "symbol": stream_symbol,
            "last_timestamp_ms": int(last_ts),
            "last_trade_id": int(last_trade_id),
            "observed_until_ms": int(observed_until_ms),
            "updated_at_utc": datetime.now(tz=UTC).isoformat(),
            "batch_rows": len(rows),
        }
        repo.write_raw_checkpoint(
            exchange=stream_exchange,
            symbol=stream_symbol,
            payload=checkpoint_payload,
        )
        repo.append_raw_wal_record(
            exchange=stream_exchange,
            symbol=stream_symbol,
            payload={
                "type": "aggtrades_raw_batch",
                "cursor": int(cursor),
                "rows": len(rows),
                "last_timestamp_ms": int(last_ts),
                "last_trade_id": int(last_trade_id),
                "observed_until_ms": int(observed_until_ms),
                "created_at_utc": datetime.now(tz=UTC).isoformat(),
            },
        )

    batch = 0
    archive_cutoff_ms = min(int(until), int(_now_ms()) - (2 * 86_400_000))
    if cursor <= archive_cutoff_ms:
        for day_value in _iter_days(cursor, archive_cutoff_ms):
            if batch >= max(1, int(max_batches)) or cursor > until:
                break
            batch += 1
            day_start_ms, day_end_ms = _day_bounds_ms(day_value)
            range_start = max(int(cursor), int(day_start_ms))
            range_end = min(int(archive_cutoff_ms), int(day_end_ms), int(until))
            if range_start > range_end:
                continue

            blob = _download_zip_bytes(
                _binance_archive_url(stream_symbol, day_value, "future"),
                retries=max(0, int(retries)),
                base_wait_sec=float(base_wait_sec),
            )
            if blob is None:
                continue
            committed_any = False
            for archive_rows in _iter_archive_rows_to_raw_aggtrades(
                blob,
                cursor_ms=range_start,
                until_ms=range_end,
            ):
                deduped = _dedupe_new_rows(
                    archive_rows,
                    cursor_ms=int(cursor),
                    last_trade_id_seen=int(last_trade_id),
                )
                if not deduped:
                    continue
                committed_any = True
                _commit_batch(deduped, observed_until_ms=int(range_end))
                if last_ts is not None and int(last_ts) >= int(until):
                    break
            if not committed_any:
                cursor = max(int(cursor), int(range_end) + 1)
                last_trade_id = -1
                continue
            cursor = max(int(cursor) + 1, int(range_end) + 1)
            last_trade_id = -1
            if last_ts is not None and int(last_ts) >= int(until):
                break

    while cursor <= until and batch < max(1, int(max_batches)):
        batch += 1
        raw_trades = _fetch_trades_with_retry(
            exchange,
            stream_symbol,
            since_ms=int(cursor),
            limit=max(1, int(limit)),
            retries=max(0, int(retries)),
            base_wait_sec=float(base_wait_sec),
        )
        if not raw_trades:
            break

        normalized_rows: list[dict[str, Any]] = []
        max_seen_ts = int(cursor)
        for index, row in enumerate(raw_trades, start=1):
            normalized = _normalize_raw_trade_row(dict(row or {}), fallback_id=index)
            if normalized is None:
                continue
            max_seen_ts = max(max_seen_ts, int(normalized["timestamp_ms"]))
            normalized_rows.append(normalized)

        deduped = _dedupe_new_rows(
            normalized_rows,
            cursor_ms=int(cursor),
            last_trade_id_seen=int(last_trade_id),
        )
        if not deduped:
            cursor = max(int(cursor) + 1, int(max_seen_ts) + 1)
            last_trade_id = -1
            continue

        _commit_batch(deduped, observed_until_ms=int(until))
        cursor = int(last_ts or cursor) + 1

        if int(last_ts or 0) >= int(until) or len(deduped) < max(1, int(limit)):
            break

    return RawAggTradesSyncStats(
        symbol=stream_symbol,
        fetched_rows=int(fetched_rows),
        upserted_rows=int(upserted_rows),
        first_timestamp_ms=first_ts,
        last_timestamp_ms=last_ts,
        checkpoint_timestamp_ms=last_ts,
        checkpoint_trade_id=(int(last_trade_id) if last_trade_id >= 0 else None),
    )


def _compact_symbol(symbol: str) -> str:
    return normalize_symbol(symbol).replace("/", "")


def _http_get_json(
    url: str,
    *,
    params: dict[str, Any],
    retries: int,
    base_wait_sec: float,
) -> Any:
    query = urllib.parse.urlencode(params)
    target = f"{url}?{query}" if query else url
    wait = max(0.1, float(base_wait_sec))
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(target, timeout=30) as resp:
                payload = resp.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            attempt += 1
            code = int(getattr(exc, "code", 0) or 0)
            if code in {400, 401, 403, 404}:
                raise RuntimeError(f"HTTP {code} for {target}") from exc
            if attempt > max(0, int(retries)):
                raise RuntimeError(f"HTTP {getattr(exc, 'code', '')} for {target}") from exc
            retry_after_raw = (
                exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
            )
            retry_after = 0.0
            if retry_after_raw is not None:
                try:
                    retry_after = max(0.0, float(str(retry_after_raw).strip()))
                except Exception:
                    retry_after = 0.0
            ceiling = 60.0 if code == 429 else 10.0
            sleep_for = max(wait, retry_after)
            time.sleep(sleep_for)
            wait = min(wait * 2.0, ceiling)
        except Exception:
            attempt += 1
            if attempt > max(0, int(retries)):
                raise
            time.sleep(wait)
            wait = min(wait * 2.0, 10.0)


def normalize_aggtrade_row(trade: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one native Binance aggTrade payload into raw aggTrades schema."""
    payload = dict(trade or {})
    ts_raw = payload.get("timestamp_ms", payload.get("timestamp", payload.get("T")))
    if ts_raw is None:
        return None
    try:
        timestamp_ms = int(ts_raw)
    except Exception:
        return None
    if timestamp_ms <= 0:
        return None

    price_raw = payload.get("price", payload.get("p"))
    amount_raw = payload.get("amount", payload.get("quantity", payload.get("q")))
    if price_raw is None or amount_raw is None:
        return None

    trade_id_raw = payload.get("agg_trade_id", payload.get("id"))
    if trade_id_raw is None:
        trade_id_raw = payload.get("tradeId")
    if trade_id_raw is None:
        trade_id_raw = payload.get("a")
    if trade_id_raw is None and isinstance(payload.get("info"), dict):
        trade_id_raw = payload["info"].get("a")
    if trade_id_raw is None:
        return None

    side = str(payload.get("side", "")).strip().lower()
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    maker_raw = payload.get("maker")
    if maker_raw is None:
        maker_raw = payload.get("is_buyer_maker", payload.get("m"))
    if maker_raw is None:
        maker_raw = info.get("m")
    is_buyer_maker = bool(maker_raw) if maker_raw is not None else side == "sell"

    quantity = float(amount_raw)
    if quantity < 0.0:
        return None

    return {
        "agg_trade_id": int(trade_id_raw),
        "timestamp_ms": int(timestamp_ms),
        "price": float(price_raw),
        "quantity": float(quantity),
        "is_buyer_maker": bool(is_buyer_maker),
    }


def fetch_aggtrades_batch(
    *,
    exchange: Any,
    symbol: str,
    since_ms: int,
    limit: int = 1000,
    retries: int = 3,
    base_wait_sec: float = 0.5,
) -> list[dict[str, Any]]:
    """Fetch and normalize one aggTrades batch from exchange trade endpoint."""
    rows = _fetch_trades_with_retry(
        exchange,
        symbol,
        since_ms=int(since_ms),
        limit=max(1, int(limit)),
        retries=max(0, int(retries)),
        base_wait_sec=float(base_wait_sec),
    )
    normalized: list[dict[str, Any]] = []
    for trade in rows:
        parsed = normalize_aggtrade_row(dict(trade or {}))
        if parsed is not None:
            normalized.append(parsed)
    normalized.sort(key=lambda item: (int(item["timestamp_ms"]), int(item["agg_trade_id"])))
    return normalized


def _merge_feature_point(
    store: dict[int, dict[str, Any]],
    timestamp_ms: int,
    fields: dict[str, Any],
) -> None:
    row = store.get(int(timestamp_ms))
    if row is None:
        row = {"timestamp_ms": int(timestamp_ms)}
        store[int(timestamp_ms)] = row
    row.update(fields)


def _optional_float_field(value: Any) -> float | None:
    """Parse an optional numeric API field without treating an empty string as data."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return float(value)


def _fetch_funding_history(
    *,
    symbol: str,
    since_ms: int,
    until_ms: int,
    retries: int,
    base_wait_sec: float,
) -> list[dict[str, Any]]:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    out: list[dict[str, Any]] = []
    cursor = max(0, int(since_ms))
    end_ms = int(until_ms)
    throttle_sec = max(0.0, float(base_wait_sec) * 0.25)
    while cursor <= end_ms:
        try:
            data = _http_get_json(
                url,
                params={
                    "symbol": _compact_symbol(symbol),
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
        except RuntimeError as exc:
            if "HTTP 400" in str(exc):
                break
            raise
        rows = list(data) if isinstance(data, list) else []
        if not rows:
            break
        out.extend(rows)
        last = int(rows[-1].get("fundingTime", cursor))
        if last < cursor:
            break
        cursor = last + 1
        if len(rows) < 1000:
            break
        if throttle_sec > 0.0:
            time.sleep(throttle_sec)
    return out


def _fetch_price_klines(
    *,
    symbol: str,
    price_type: str,
    interval: str,
    since_ms: int,
    until_ms: int,
    retries: int,
    base_wait_sec: float,
) -> list[list[Any]]:
    endpoint = "markPriceKlines" if price_type == "mark" else "indexPriceKlines"
    url = f"https://fapi.binance.com/fapi/v1/{endpoint}"
    out: list[list[Any]] = []
    cursor = max(0, int(since_ms))
    end_ms = int(until_ms)
    throttle_sec = max(0.0, float(base_wait_sec) * 0.25)
    while cursor <= end_ms:
        params = {
            "interval": str(interval),
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        }
        if price_type == "mark":
            params["symbol"] = _compact_symbol(symbol)
        else:
            params["pair"] = _compact_symbol(symbol)
        try:
            data = _http_get_json(
                url,
                params=params,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
        except RuntimeError as exc:
            if "HTTP 400" in str(exc):
                break
            raise
        rows = list(data) if isinstance(data, list) else []
        if not rows:
            break
        out.extend(rows)
        last_open_ms = int(rows[-1][0])
        if last_open_ms < cursor:
            break
        cursor = last_open_ms + 1
        if len(rows) < 1500:
            break
        if throttle_sec > 0.0:
            time.sleep(throttle_sec)
    return out


def _fetch_open_interest_history(
    *,
    symbol: str,
    period: str,
    since_ms: int,
    until_ms: int,
    retries: int,
    base_wait_sec: float,
) -> list[dict[str, Any]]:
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    out: list[dict[str, Any]] = []
    cursor = max(0, int(since_ms))
    end_ms = int(until_ms)
    throttle_sec = max(0.0, float(base_wait_sec) * 0.25)
    period_token = str(period).strip().lower()
    unit = period_token[-1:] if period_token else ""
    size_raw = period_token[:-1] if period_token else ""
    unit_ms = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }.get(unit, 300_000)
    try:
        size = max(1, int(size_raw or "1"))
    except Exception:
        size = 1
    request_span_ms = max(1, size * unit_ms * 500)
    while cursor <= end_ms:
        request_end_ms = min(end_ms, int(cursor) + int(request_span_ms) - 1)
        try:
            data = _http_get_json(
                url,
                params={
                    "symbol": _compact_symbol(symbol),
                    "period": str(period),
                    "startTime": cursor,
                    "endTime": request_end_ms,
                    "limit": 500,
                },
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
        except RuntimeError as exc:
            if "HTTP 400" in str(exc):
                if int(request_end_ms) >= int(end_ms):
                    break
                cursor = int(request_end_ms) + 1
                if throttle_sec > 0.0:
                    time.sleep(throttle_sec)
                continue
            raise
        rows = list(data) if isinstance(data, list) else []
        if not rows:
            if int(request_end_ms) >= int(end_ms):
                break
            cursor = int(request_end_ms) + 1
            if throttle_sec > 0.0:
                time.sleep(throttle_sec)
            continue
        out.extend(rows)
        last_ts = int(rows[-1].get("timestamp", cursor))
        if last_ts < cursor:
            break
        cursor = last_ts + 1
        if len(rows) < 500:
            break
        if throttle_sec > 0.0:
            time.sleep(throttle_sec)
    return out


def _fetch_liquidation_orders(
    *,
    symbol: str,
    since_ms: int,
    until_ms: int,
    retries: int,
    base_wait_sec: float,
) -> list[dict[str, Any]]:
    url = "https://fapi.binance.com/fapi/v1/allForceOrders"
    out: list[dict[str, Any]] = []
    cursor = max(0, int(since_ms))
    end_ms = int(until_ms)
    throttle_sec = max(0.0, float(base_wait_sec) * 0.25)
    while cursor <= end_ms:
        try:
            data = _http_get_json(
                url,
                params={
                    "symbol": _compact_symbol(symbol),
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
        except RuntimeError as exc:
            if any(
                code in str(exc)
                for code in ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404", "HTTP 429")
            ):
                break
            raise
        rows = list(data) if isinstance(data, list) else []
        if not rows:
            break
        out.extend(rows)
        last_ts = int(rows[-1].get("time", cursor))
        if last_ts < cursor:
            break
        cursor = last_ts + 1
        if len(rows) < 1000:
            break
        if throttle_sec > 0.0:
            time.sleep(throttle_sec)
    return out


def sync_futures_feature_points(
    *,
    db_path: str,
    exchange_id: str,
    symbol_list: Sequence[str],
    since_ms: int,
    until_ms: int,
    mark_index_interval: str = "1m",
    open_interest_period: str = "5m",
    include_funding: bool = True,
    include_mark_index: bool = True,
    include_open_interest: bool = True,
    include_liquidations: bool = True,
    retries: int = 3,
    base_wait_sec: float = 0.5,
    backend: str | None = None,
    **legacy: Any,
) -> list[FuturesFeatureSyncStats]:
    """Collect and store futures feature data points for strategy research."""
    _ = legacy
    summaries: list[FuturesFeatureSyncStats] = []
    stream_exchange = str(exchange_id).strip().lower()

    for symbol in symbol_list:
        stream_symbol = normalize_symbol(symbol)
        points: dict[int, dict[str, Any]] = {}

        if include_funding:
            funding_rows = _fetch_funding_history(
                symbol=stream_symbol,
                since_ms=since_ms,
                until_ms=until_ms,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            for row in funding_rows:
                ts = int(row.get("fundingTime", 0) or 0)
                if ts <= 0:
                    continue
                raw_funding_rate = row.get("fundingRate")
                funding_rate = float(raw_funding_rate) if raw_funding_rate is not None else None
                funding_mark_price = _optional_float_field(row.get("markPrice"))
                _merge_feature_point(
                    points,
                    ts,
                    {
                        "funding_rate": funding_rate,
                        "funding_mark_price": funding_mark_price,
                        "funding_fee_rate": funding_rate,
                        "funding_fee_quote_per_unit": (
                            float(funding_rate) * float(funding_mark_price)
                            if funding_rate is not None and funding_mark_price is not None
                            else None
                        ),
                    },
                )

        if include_mark_index:
            mark_rows = _fetch_price_klines(
                symbol=stream_symbol,
                price_type="mark",
                interval=mark_index_interval,
                since_ms=since_ms,
                until_ms=until_ms,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            for row in mark_rows:
                ts = int(row[0])
                _merge_feature_point(points, ts, {"mark_price": float(row[4])})

            index_rows = _fetch_price_klines(
                symbol=stream_symbol,
                price_type="index",
                interval=mark_index_interval,
                since_ms=since_ms,
                until_ms=until_ms,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            for row in index_rows:
                ts = int(row[0])
                _merge_feature_point(points, ts, {"index_price": float(row[4])})

        if include_open_interest:
            oi_rows = _fetch_open_interest_history(
                symbol=stream_symbol,
                period=open_interest_period,
                since_ms=since_ms,
                until_ms=until_ms,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            for row in oi_rows:
                ts = int(row.get("timestamp", 0) or 0)
                if ts <= 0:
                    continue
                oi_val = row.get("sumOpenInterestValue")
                if oi_val is None:
                    oi_val = row.get("sumOpenInterest")
                _merge_feature_point(
                    points,
                    ts,
                    {"open_interest": float(oi_val) if oi_val is not None else None},
                )

        if include_liquidations:
            liq_rows = _fetch_liquidation_orders(
                symbol=stream_symbol,
                since_ms=since_ms,
                until_ms=until_ms,
                retries=retries,
                base_wait_sec=base_wait_sec,
            )
            for row in liq_rows:
                ts = int(row.get("time", 0) or 0)
                if ts <= 0:
                    continue
                side = str(row.get("side", "")).upper()
                qty = float(row.get("origQty", 0.0) or 0.0)
                price = float(row.get("price", 0.0) or 0.0)
                notional = qty * price
                fields: dict[str, Any]
                if side == "SELL":
                    fields = {
                        "liquidation_long_qty": qty,
                        "liquidation_long_notional": notional,
                    }
                else:
                    fields = {
                        "liquidation_short_qty": qty,
                        "liquidation_short_notional": notional,
                    }
                _merge_feature_point(points, ts, fields)

        sorted_rows = [points[key] for key in sorted(points.keys())]
        upserted = upsert_futures_feature_points_rows(
            db_path,
            exchange=stream_exchange,
            symbol=stream_symbol,
            rows=sorted_rows,
            source="binance_futures_api",
            backend=backend,
        )
        first_ts = int(sorted_rows[0]["timestamp_ms"]) if sorted_rows else None
        last_ts = int(sorted_rows[-1]["timestamp_ms"]) if sorted_rows else None
        summaries.append(
            FuturesFeatureSyncStats(
                symbol=stream_symbol,
                upserted_rows=int(upserted),
                first_timestamp_ms=first_ts,
                last_timestamp_ms=last_ts,
            )
        )

    return summaries


class MarketDataSyncService:
    """OOP facade for market data synchronization workflows."""

    def __init__(
        self, *, exchange: Any, db_path: str, exchange_id: str, backend: str | None = None
    ):
        self.exchange = exchange
        self.db_path = str(db_path)
        self.exchange_id = str(exchange_id)
        self.backend = str(backend or "").strip() or None

    def get_symbol_coverage(
        self, *, symbol: str, timeframe: str
    ) -> tuple[int | None, int | None, int]:
        return get_symbol_ohlcv_coverage(
            db_path=self.db_path,
            exchange_id=self.exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            backend=self.backend,
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
            backend=self.backend,
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
            backend=self.backend,
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
            backend=self.backend,
            export_csv_dir=export_csv_dir,
        )


def get_symbol_ohlcv_coverage(
    *,
    db_path: str,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    backend: str | None = None,
) -> tuple[int | None, int | None, int]:
    """Return (first_ts, last_ts, row_count) for one OHLCV stream key."""
    _ = _is_local_storage(db_path, backend=backend)
    first_ts, last_ts, row_count = load_ohlcv_coverage_from_db(
        db_path,
        exchange=str(exchange_id).strip().lower(),
        symbol=normalize_symbol(symbol),
        timeframe=normalize_timeframe_token(timeframe),
        backend=backend,
    )
    return first_ts, last_ts, int(row_count)


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
    backend: str | None = None,
    export_csv_dir: str | None = None,
) -> list[SyncStats]:
    """Ensure OHLCV coverage exists via native futures aggTrades raw-first lineage."""
    from lumina_quant.services.materialize_from_raw import materialize_raw_aggtrades_bundle
    from lumina_quant.storage.parquet import ParquetMarketDataRepository

    effective_until = int(until_ms) if until_ms is not None else _now_ms()
    effective_since = (
        int(since_ms)
        if since_ms is not None
        else int(datetime(2017, 1, 1, tzinfo=UTC).timestamp() * 1000)
    )
    effective_since = max(0, effective_since)
    if effective_until < effective_since:
        effective_until = effective_since

    timeframe_token = normalize_timeframe_token(timeframe)
    required_timeframes = ["1s"] if timeframe_token == "1s" else ["1s", timeframe_token]
    repo = ParquetMarketDataRepository(str(db_path))
    summaries: list[SyncStats] = []

    for symbol in symbol_list:
        stream_symbol = normalize_symbol(symbol)
        raw_stats = sync_symbol_aggtrades_raw(
            exchange=exchange,
            db_path=db_path,
            exchange_id=exchange_id,
            symbol=stream_symbol,
            start_ms=int(effective_since),
            end_ms=int(effective_until),
            limit=limit,
            max_batches=max_batches,
            retries=retries,
            base_wait_sec=base_wait_sec,
            resume_from_checkpoint=not bool(force_full),
        )
        materialize_raw_aggtrades_bundle(
            root_path=str(db_path),
            exchange=str(exchange_id).strip().lower(),
            symbol=stream_symbol,
            timeframes=list(required_timeframes),
            start_date=datetime.fromtimestamp(int(effective_since) / 1000.0, tz=UTC).isoformat(),
            end_date=datetime.fromtimestamp(int(effective_until) / 1000.0, tz=UTC).isoformat(),
            producer="ensure_market_data_coverage",
            require_complete=True,
        )

        try:
            frame = repo.load_committed_ohlcv_chunked(
                exchange=str(exchange_id).strip().lower(),
                symbol=stream_symbol,
                timeframe=timeframe_token,
                start_date=datetime.fromtimestamp(
                    int(effective_since) / 1000.0, tz=UTC
                ).isoformat(),
                end_date=datetime.fromtimestamp(int(effective_until) / 1000.0, tz=UTC).isoformat(),
                chunk_days=7,
                warmup_bars=0,
                staleness_threshold_seconds=None,
            )
        except RawFirstDataMissingError:
            frame = pl.DataFrame()

        final_first = (
            int(frame["datetime"].min().timestamp() * 1000) if not frame.is_empty() else None
        )
        final_last = (
            int(frame["datetime"].max().timestamp() * 1000) if not frame.is_empty() else None
        )
        summaries.append(
            SyncStats(
                symbol=stream_symbol,
                fetched_rows=int(raw_stats.fetched_rows),
                upserted_rows=int(frame.height),
                first_timestamp_ms=final_first,
                last_timestamp_ms=final_last,
            )
        )

        if export_csv_dir:
            Path(export_csv_dir).mkdir(parents=True, exist_ok=True)
            csv_path = Path(export_csv_dir) / symbol_csv_filename(stream_symbol)
            if frame.is_empty():
                csv_path.write_text("", encoding="utf-8")
            else:
                frame.write_csv(csv_path)

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
    backend: str | None = None,
) -> SyncStats:
    """Synchronize one symbol OHLCV range using raw aggTrades as the source of truth."""
    _ = _is_local_storage(db_path, backend=backend)
    from lumina_quant.storage.parquet import ParquetMarketDataRepository

    timeframe_token = normalize_timeframe_token(timeframe)
    stream_symbol = normalize_symbol(symbol)
    cursor = max(0, int(start_ms))
    until = max(cursor, int(end_ms))

    raw_sync = sync_symbol_aggtrades_raw(
        exchange=exchange,
        db_path=db_path,
        exchange_id=exchange_id,
        symbol=stream_symbol,
        start_ms=cursor,
        end_ms=until,
        limit=limit,
        max_batches=max_batches,
        retries=retries,
        base_wait_sec=base_wait_sec,
        resume_from_checkpoint=False,
    )

    repo = ParquetMarketDataRepository(str(db_path))
    raw_frame = repo.load_raw_aggtrades(
        exchange=str(exchange_id).lower(),
        symbol=stream_symbol,
        start_date=datetime.fromtimestamp(cursor / 1000.0, tz=UTC).isoformat(),
        end_date=datetime.fromtimestamp(until / 1000.0, tz=UTC).isoformat(),
    )

    previous_close = _last_1s_close(
        db_path=db_path,
        exchange_id=str(exchange_id).lower(),
        symbol=stream_symbol,
        before_ms=int(cursor) - 1000,
    )
    frame_1s = raw_aggtrades_to_1s_frame(
        raw_frame,
        source=f"{exchange_id}:{stream_symbol}:sync_symbol_ohlcv",
        range_start_ms=int(cursor),
        range_end_ms=int(until),
        previous_close=previous_close,
        complete_through_ms=int(until),
    )
    if frame_1s.is_empty():
        return SyncStats(
            symbol=stream_symbol,
            fetched_rows=int(raw_sync.fetched_rows),
            upserted_rows=0,
            first_timestamp_ms=None,
            last_timestamp_ms=None,
        )

    upserted_rows = int(
        upsert_ohlcv_rows_1s(
            db_path,
            exchange=str(exchange_id).lower(),
            symbol=stream_symbol,
            rows=frame_1s,
            backend=backend,
        )
    )

    output_frame = frame_1s
    if timeframe_token != "1s":
        output_frame = resample_1s_frame(
            frame_1s,
            timeframe=timeframe_token,
            complete_through_ms=int(until),
        )
        if output_frame.is_empty():
            return SyncStats(
                symbol=stream_symbol,
                fetched_rows=int(raw_sync.fetched_rows),
                upserted_rows=int(upserted_rows),
                first_timestamp_ms=None,
                last_timestamp_ms=None,
            )
        conn = connect_market_data_db(db_path)
        try:
            upserted_rows += int(
                upsert_ohlcv_rows(
                    conn,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                    timeframe=timeframe_token,
                    rows=output_frame,
                    source="binance_futures_raw_first",
                    db_path=db_path,
                    backend=backend,
                )
            )
        finally:
            conn.close()

    first_dt = output_frame["datetime"][0]
    last_dt = output_frame["datetime"][-1]
    first_ts = int(
        (
            first_dt.replace(tzinfo=UTC) if first_dt.tzinfo is None else first_dt.astimezone(UTC)
        ).timestamp()
        * 1000
    )
    last_ts = int(
        (
            last_dt.replace(tzinfo=UTC) if last_dt.tzinfo is None else last_dt.astimezone(UTC)
        ).timestamp()
        * 1000
    )
    row_count = int(output_frame.height)
    return SyncStats(
        symbol=stream_symbol,
        fetched_rows=int(raw_sync.fetched_rows),
        upserted_rows=int(upserted_rows),
        first_timestamp_ms=first_ts if row_count > 0 else None,
        last_timestamp_ms=last_ts if row_count > 0 else None,
    )


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
    backend: str | None = None,
    export_csv_dir: str | None = None,
) -> list[SyncStats]:
    """Synchronize OHLCV for multiple symbols and optionally export CSV copies."""
    effective_until = int(until_ms) if until_ms is not None else _now_ms()
    default_since = (
        int(since_ms)
        if since_ms is not None
        else int(datetime(2017, 1, 1, tzinfo=UTC).timestamp() * 1000)
    )

    _ = _is_local_storage(db_path, backend=backend)
    conn = connect_market_data_db(db_path)
    try:
        stats: list[SyncStats] = []

        for symbol in symbol_list:
            stream_symbol = normalize_symbol(symbol)
            timeframe_token = normalize_timeframe_token(timeframe)
            if timeframe_token == "1s":
                last_ts = get_last_ohlcv_1s_timestamp_ms(
                    db_path,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                    backend=backend,
                )
            else:
                last_ts = get_last_ohlcv_timestamp_ms(
                    conn,
                    exchange=str(exchange_id).lower(),
                    symbol=stream_symbol,
                    timeframe=timeframe_token,
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
                backend=backend,
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
