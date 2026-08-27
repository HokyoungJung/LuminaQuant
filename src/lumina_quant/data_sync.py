"""Binance OHLCV synchronization helpers."""

from __future__ import annotations

import csv
import io
import json
from hashlib import sha256
import inspect
import math
import os
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
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
_FUNDING_TIMESTAMP_JITTER_MS = 1_000
_FUNDING_SETTLEMENT_GRANULARITY_MS = 60_000


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
    from_id: int | None = None,
    until_ms: int | None = None,
) -> list[dict[str, Any]]:
    wait = max(0.1, float(base_wait_sec))
    attempt = 0
    while True:
        try:
            fetch_fn = getattr(exchange, "agg_trades", None)
            if callable(fetch_fn):
                rows = fetch_fn(
                    symbol=symbol,
                    start_time=(int(since_ms) if from_id is None else None),
                    end_time=(
                        min(
                            int(since_ms) + 3_599_999,
                            int(until_ms) if until_ms is not None else _now_ms(),
                        )
                        if from_id is None
                        else None
                    ),
                    from_id=from_id,
                    limit=max(1, min(int(limit), 1_000)),
                )
                return list(rows or [])
            fetch_trades = exchange.fetch_trades
            if from_id is not None:
                try:
                    signature = inspect.signature(fetch_trades)
                    supports_params = "params" in signature.parameters or any(
                        parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in signature.parameters.values()
                    )
                except TypeError, ValueError:
                    supports_params = False
                if supports_params:
                    return list(
                        fetch_trades(
                            symbol,
                            since=since_ms,
                            limit=limit,
                            params={"fromId": int(from_id)},
                        )
                        or []
                    )
            return list(fetch_trades(symbol, since=since_ms, limit=limit) or [])
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
    end_ms = start_ms + 86_399_999
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


def _raw_archive_chunk_rows() -> int:
    raw = os.getenv("LQ_RAW_ARCHIVE_CHUNK_ROWS")
    if raw is None or not raw.strip():
        return _DEFAULT_RAW_ARCHIVE_CHUNK_ROWS
    text = raw.strip()
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("LQ_RAW_ARCHIVE_CHUNK_ROWS must be an integer row count")
    rows = int(text)
    if not 1_000 <= rows <= 1_000_000:
        raise ValueError("LQ_RAW_ARCHIVE_CHUNK_ROWS must be between 1000 and 1000000 rows")
    return rows


def _iter_archive_rows_to_raw_aggtrades(
    zip_blob: bytes,
    *,
    expected_member_name: str,
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
    expected_header = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
    ]
    expected_member_match = re.fullmatch(
        r"[A-Z0-9]+-aggTrades-(\d{4}-\d{2}-\d{2})\.csv",
        expected_member_name,
    )
    if expected_member_match is None:
        raise ValueError("aggTrades archive member identity is invalid")
    try:
        archive_day_bounds = _day_bounds_ms(
            datetime.strptime(expected_member_match.group(1), "%Y-%m-%d").date()
        )
    except ValueError:
        raise ValueError("aggTrades archive member has an invalid date") from None

    with zipfile.ZipFile(io.BytesIO(zip_blob)) as zf:
        members = zf.infolist()
        if len(members) != 1:
            raise ValueError("aggTrades archive must contain exactly one CSV member")
        member = members[0]
        mode = member.external_attr >> 16
        if (
            member.is_dir()
            or member.filename != expected_member_name
            or stat.S_ISLNK(mode)
            or (stat.S_IFMT(mode) not in {0, stat.S_IFREG})
        ):
            raise ValueError("aggTrades archive member must be the expected regular CSV file")

        def _validated_source_rows():
            previous_agg_trade_id: int | None = None
            previous_timestamp_ms: int | None = None
            source_row_count = 0
            data_row_count = 0
            with zf.open(member, "r") as raw_file:
                text_file = io.TextIOWrapper(raw_file, encoding="utf-8", newline="")
                reader = csv.reader(text_file, strict=True)
                for row in reader:
                    if source_row_count == 0 and row == expected_header:
                        source_row_count += 1
                        continue
                    source_row_count += 1
                    data_row_count += 1
                    if len(row) != 7 or any(value == "" or value != value.strip() for value in row):
                        raise ValueError("aggTrades archive contains a malformed CSV row")
                    try:
                        (
                            agg_trade_id_raw,
                            price_raw,
                            quantity_raw,
                            first_trade_id_raw,
                            last_trade_id_raw,
                            timestamp_raw,
                            is_buyer_maker_raw,
                        ) = row
                        if not all(
                            re.fullmatch(r"[0-9]+", value)
                            for value in (
                                agg_trade_id_raw,
                                first_trade_id_raw,
                                last_trade_id_raw,
                                timestamp_raw,
                            )
                        ) or not all(
                            re.fullmatch(
                                r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
                                value,
                            )
                            for value in (price_raw, quantity_raw)
                        ):
                            raise ValueError
                        agg_trade_id = int(agg_trade_id_raw)
                        first_trade_id = int(first_trade_id_raw)
                        last_trade_id = int(last_trade_id_raw)
                        timestamp_ms = int(timestamp_raw)
                        price = float(price_raw)
                        quantity = float(quantity_raw)
                    except TypeError, ValueError, OverflowError:
                        raise ValueError(
                            "aggTrades archive contains invalid numeric data"
                        ) from None
                    if (
                        not math.isfinite(price)
                        or not math.isfinite(quantity)
                        or price <= 0.0
                        or quantity <= 0.0
                        or last_trade_id < first_trade_id
                    ):
                        raise ValueError("aggTrades archive contains invalid trade data")
                    boolean_value = is_buyer_maker_raw.casefold()
                    if boolean_value not in {"true", "false"}:
                        raise ValueError("aggTrades archive contains an invalid buyer-maker flag")
                    if archive_day_bounds is not None and not (
                        archive_day_bounds[0] <= timestamp_ms <= archive_day_bounds[1]
                    ):
                        raise ValueError("aggTrades archive timestamp is outside its archive day")
                    if (
                        previous_agg_trade_id is not None and agg_trade_id <= previous_agg_trade_id
                    ) or (
                        previous_timestamp_ms is not None and timestamp_ms < previous_timestamp_ms
                    ):
                        raise ValueError("aggTrades archive is not globally ordered")
                    previous_agg_trade_id = agg_trade_id
                    previous_timestamp_ms = timestamp_ms
                    yield (
                        agg_trade_id,
                        timestamp_ms,
                        price,
                        quantity,
                        boolean_value == "true",
                    )
            if data_row_count == 0:
                raise ValueError("aggTrades archive CSV member is empty")

        try:
            for _ in _validated_source_rows():
                pass
            for (
                agg_trade_id,
                timestamp_ms,
                price,
                quantity,
                is_buyer_maker,
            ) in _validated_source_rows():
                if timestamp_ms < int(cursor_ms) or timestamp_ms > int(until_ms):
                    continue
                rows.append(
                    {
                        "agg_trade_id": agg_trade_id,
                        "timestamp_ms": timestamp_ms,
                        "price": price,
                        "quantity": quantity,
                        "is_buyer_maker": is_buyer_maker,
                    }
                )
                if len(rows) >= max_rows:
                    yield rows
                    rows = []
        except csv.Error:
            raise ValueError("aggTrades archive contains malformed CSV syntax") from None
    if rows:
        yield rows


def _archive_rows_to_raw_aggtrades(
    zip_blob: bytes,
    *,
    expected_member_name: str,
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
            expected_member_name=expected_member_name,
            cursor_ms=cursor_ms,
            until_ms=until_ms,
            chunk_rows=_raw_archive_chunk_rows(),
        )
    )
    if not chunks:
        return []
    return [row for chunk in chunks for row in chunk]


def _checkpoint_last_row_digest(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _validate_live_aggtrade_identity_order(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_last_row: Mapping[str, Any] | None = None,
) -> None:
    """Reject live aggTrades that cannot advance an authenticated stream."""
    previous_timestamp: int | None = None
    previous_trade_id: int | None = None
    for row in rows:
        timestamp_ms = int(row["timestamp_ms"])
        trade_id = int(row["agg_trade_id"])
        if previous_timestamp is not None and timestamp_ms < previous_timestamp:
            raise ValueError("Live aggTrades timestamps must be nondecreasing")
        if previous_trade_id is not None and trade_id <= previous_trade_id:
            raise ValueError("Live aggTrades aggregate IDs must be strictly increasing")
        previous_timestamp = timestamp_ms
        previous_trade_id = trade_id

    if not rows or checkpoint_last_row is None:
        return

    checkpoint_timestamp = int(checkpoint_last_row["timestamp_ms"])
    checkpoint_trade_id = int(checkpoint_last_row["agg_trade_id"])
    first_row = rows[0]
    first_timestamp = int(first_row["timestamp_ms"])
    first_trade_id = int(first_row["agg_trade_id"])
    if first_row == checkpoint_last_row:
        if len(rows) > 1 and int(rows[1]["agg_trade_id"]) <= checkpoint_trade_id:
            raise ValueError("Live aggTrade aggregate ID does not advance checkpoint cursor")
        return
    if first_timestamp < checkpoint_timestamp:
        raise ValueError("Live aggTrade timestamp precedes the checkpoint cursor")
    if first_trade_id <= checkpoint_trade_id:
        raise ValueError("Live aggTrade aggregate ID does not advance checkpoint cursor")


def _sync_symbol_aggtrades_raw_under_lease(
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
    lease: Any,
) -> RawAggTradesSyncStats:
    """Collect Binance aggTrades into raw parquet partitions with checkpoint resume."""
    from lumina_quant.storage.parquet import ParquetMarketDataRepository

    repo = ParquetMarketDataRepository(str(db_path))
    stream_exchange = str(exchange_id).strip().lower() or "binance"
    stream_symbol = normalize_symbol(symbol)
    repo.recover_raw_stream(exchange=stream_exchange, symbol=stream_symbol, lease=lease)

    cursor = max(0, int(start_ms))
    until = max(cursor, int(end_ms))
    last_trade_id = -1
    checkpoint_last_row: dict[str, Any] | None = None
    if bool(resume_from_checkpoint):
        checkpoint = repo.read_raw_checkpoint(
            exchange=stream_exchange, symbol=stream_symbol, lease=lease
        )
        if checkpoint != {}:
            checkpoint_last_row = dict(checkpoint["last_row"])
            persisted = repo.read_raw_recovery_bounds(
                exchange=stream_exchange,
                symbol=stream_symbol,
                checkpoint_last_row=checkpoint_last_row,
                lease=lease,
            )
            tail = dict(persisted.to_dicts()[-1])
            if tail != checkpoint_last_row:
                checkpoint_last_row = tail
                checkpoint = {
                    **checkpoint,
                    "last_timestamp_ms": tail["timestamp_ms"],
                    "last_trade_id": tail["agg_trade_id"],
                    "observed_until_ms": max(
                        int(checkpoint["observed_until_ms"]), int(tail["timestamp_ms"])
                    ),
                    "last_row": tail,
                    "last_row_sha256": _checkpoint_last_row_digest(tail),
                    "updated_at_utc": datetime.now(tz=UTC).isoformat(),
                }
                repo.write_raw_checkpoint(
                    exchange=stream_exchange, symbol=stream_symbol, payload=checkpoint, lease=lease
                )
                repo.append_raw_wal_record(
                    exchange=stream_exchange,
                    symbol=stream_symbol,
                    payload={
                        "type": "aggtrades_raw_checkpoint_recovery",
                        "last_trade_id": tail["agg_trade_id"],
                    },
                    lease=lease,
                )
            checkpoint_ts = int(checkpoint_last_row["timestamp_ms"])
            checkpoint_trade_id = int(checkpoint_last_row["agg_trade_id"])
            if checkpoint_ts >= cursor:
                cursor = checkpoint_ts
                last_trade_id = checkpoint_trade_id
        else:
            persisted = repo.read_raw_recovery_bounds(
                exchange=stream_exchange,
                symbol=stream_symbol,
                checkpoint_last_row=None,
                lease=lease,
            )
            if not persisted.is_empty():
                checkpoint_last_row = dict(persisted.to_dicts()[-1])
                checkpoint = {
                    "exchange": stream_exchange,
                    "symbol": stream_symbol,
                    "last_timestamp_ms": checkpoint_last_row["timestamp_ms"],
                    "last_trade_id": checkpoint_last_row["agg_trade_id"],
                    "observed_until_ms": checkpoint_last_row["timestamp_ms"],
                    "updated_at_utc": datetime.now(tz=UTC).isoformat(),
                    "batch_rows": 1,
                    "last_row": checkpoint_last_row,
                    "last_row_sha256": _checkpoint_last_row_digest(checkpoint_last_row),
                }
                repo.write_raw_checkpoint(
                    exchange=stream_exchange, symbol=stream_symbol, payload=checkpoint, lease=lease
                )
                repo.append_raw_wal_record(
                    exchange=stream_exchange,
                    symbol=stream_symbol,
                    payload={
                        "type": "aggtrades_raw_first_commit_recovery",
                        "last_trade_id": checkpoint_last_row["agg_trade_id"],
                    },
                    lease=lease,
                )
                cursor = max(cursor, int(checkpoint_last_row["timestamp_ms"]))
                last_trade_id = int(checkpoint_last_row["agg_trade_id"])

    fetched_rows = 0
    upserted_rows = 0
    first_ts = None
    last_ts = None

    def _filter_archive_rows(
        rows: list[dict[str, Any]],
        *,
        cursor_ms: int,
        last_trade_id_seen: int,
        checkpoint_boundary_row: Mapping[str, Any] | None,
        checkpoint_boundary_pending: bool,
    ) -> tuple[list[dict[str, Any]], bool]:
        filtered: list[dict[str, Any]] = []
        boundary_row = (
            dict(checkpoint_boundary_row) if checkpoint_boundary_row is not None else None
        )
        boundary_timestamp = int(boundary_row["timestamp_ms"]) if boundary_row is not None else None
        boundary_trade_id = int(boundary_row["agg_trade_id"]) if boundary_row is not None else None
        for item in rows:
            timestamp_ms = int(item["timestamp_ms"])
            trade_id = int(item["agg_trade_id"])
            if timestamp_ms < int(cursor_ms) or timestamp_ms > int(until):
                continue
            if checkpoint_boundary_pending:
                if timestamp_ms > int(boundary_timestamp):
                    raise ValueError("Archive aggTrade checkpoint identity is missing")
                if timestamp_ms == int(boundary_timestamp):
                    if trade_id < int(boundary_trade_id):
                        continue
                    if trade_id > int(boundary_trade_id):
                        raise ValueError(
                            "Archive aggTrade checkpoint is followed by a higher aggregate ID"
                        )
                    if item != boundary_row:
                        raise ValueError(
                            "Archive aggTrade checkpoint identity has a conflicting payload"
                        )
                    checkpoint_boundary_pending = False
                    continue
            if timestamp_ms == int(cursor_ms) and trade_id <= int(last_trade_id_seen):
                if (
                    checkpoint_last_row is None
                    or trade_id != int(last_trade_id_seen)
                    or item != checkpoint_last_row
                ):
                    raise ValueError("Archive aggTrade checkpoint overlap does not match last row")
                continue
            filtered.append(item)
        return filtered, checkpoint_boundary_pending

    def _validate_and_filter_live_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _validate_live_aggtrade_identity_order(
            rows,
            checkpoint_last_row=(checkpoint_last_row if last_trade_id >= 0 else None),
        )
        filtered: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            timestamp_ms = int(row["timestamp_ms"])
            trade_id = int(row["agg_trade_id"])
            # Binance may include the authenticated boundary row in a fromId
            # response.  Accept only its exact persisted identity before testing
            # the timestamp cursor; every other replay remains invalid.
            if (
                index == 0
                and checkpoint_last_row is not None
                and last_trade_id >= 0
                and row == checkpoint_last_row
            ):
                continue
            if timestamp_ms < int(cursor):
                raise ValueError("Live aggTrade timestamp precedes the checkpoint cursor")
            if last_trade_id >= 0 and trade_id <= int(last_trade_id):
                raise ValueError("Live aggTrade aggregate ID does not advance checkpoint cursor")
            if timestamp_ms > int(until):
                continue
            filtered.append(row)
        return filtered

    def _commit_batch(rows: list[dict[str, Any]], *, observed_until_ms: int) -> None:
        nonlocal fetched_rows, upserted_rows, first_ts, last_ts, last_trade_id, cursor
        nonlocal checkpoint_last_row
        if not rows:
            return
        fetched_rows += len(rows)
        repo.preflight_raw_aggtrades(
            exchange=stream_exchange,
            symbol=stream_symbol,
            rows=rows,
            lease=lease,
        )
        date_rows: list[dict[str, Any]] = []
        date_token: str | None = None
        for row in rows:
            row_date = (
                datetime.fromtimestamp(int(row["timestamp_ms"]) / 1000, tz=UTC).date().isoformat()
            )
            if date_token is not None and row_date != date_token:
                upserted_rows += int(
                    repo.append_raw_aggtrades(
                        exchange=stream_exchange,
                        symbol=stream_symbol,
                        rows=date_rows,
                        lease=lease,
                    )
                )
                date_rows = []
            date_token = row_date
            date_rows.append(row)
        if date_rows:
            upserted_rows += int(
                repo.append_raw_aggtrades(
                    exchange=stream_exchange,
                    symbol=stream_symbol,
                    rows=date_rows,
                    lease=lease,
                )
            )
        first_batch_ts = int(rows[0]["timestamp_ms"])
        first_ts = first_batch_ts if first_ts is None else min(first_ts, first_batch_ts)
        last_ts = int(rows[-1]["timestamp_ms"])
        last_trade_id = int(rows[-1]["agg_trade_id"])
        cursor = int(last_ts)

        last_row = dict(rows[-1])
        checkpoint_last_row = last_row
        checkpoint_payload = {
            "exchange": stream_exchange,
            "symbol": stream_symbol,
            "last_timestamp_ms": int(last_ts),
            "last_trade_id": int(last_trade_id),
            "observed_until_ms": int(observed_until_ms),
            "updated_at_utc": datetime.now(tz=UTC).isoformat(),
            "batch_rows": len(rows),
            "last_row": last_row,
            "last_row_sha256": _checkpoint_last_row_digest(last_row),
        }
        repo.write_raw_checkpoint(
            exchange=stream_exchange,
            symbol=stream_symbol,
            payload=checkpoint_payload,
            lease=lease,
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
            lease=lease,
        )

    batch = 0
    terminally_covered = False
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
                # An absent official daily archive is a continuity boundary, not
                # an empty day.  The live phase must begin at this exact cursor.
                cursor = int(range_start)
                last_trade_id = -1
                break
            archive_checkpoint_boundary = (
                dict(checkpoint_last_row)
                if (
                    checkpoint_last_row is not None
                    and last_trade_id >= 0
                    and int(checkpoint_last_row["timestamp_ms"]) == int(range_start)
                )
                else None
            )
            archive_checkpoint_pending = archive_checkpoint_boundary is not None
            for archive_rows in _iter_archive_rows_to_raw_aggtrades(
                blob,
                expected_member_name=(
                    f"{stream_symbol.replace('/', '')}-aggTrades-{day_value:%Y-%m-%d}.csv"
                ),
                cursor_ms=range_start,
                until_ms=range_end,
            ):
                deduped, archive_checkpoint_pending = _filter_archive_rows(
                    archive_rows,
                    cursor_ms=int(cursor),
                    last_trade_id_seen=int(last_trade_id),
                    checkpoint_boundary_row=archive_checkpoint_boundary,
                    checkpoint_boundary_pending=archive_checkpoint_pending,
                )
                if not deduped:
                    continue
                _commit_batch(
                    deduped,
                    observed_until_ms=int(deduped[-1]["timestamp_ms"]),
                )
            if archive_checkpoint_pending:
                raise ValueError("Archive aggTrade checkpoint identity is missing")
            # A validated official archive, including a day with no rows in the
            # requested interval, authoritatively covers the entire day range.
            cursor = int(range_end) + 1
            last_trade_id = -1
            if cursor > until:
                terminally_covered = True
                break

    while not terminally_covered and cursor <= until and batch < max(1, int(max_batches)):
        batch += 1
        request_cursor = int(cursor)
        used_from_id = last_trade_id >= 0
        raw_trades = _fetch_trades_with_retry(
            exchange,
            stream_symbol,
            since_ms=request_cursor,
            from_id=(int(last_trade_id) + 1 if used_from_id else None),
            until_ms=int(until),
            limit=max(1, int(limit)),
            retries=max(0, int(retries)),
            base_wait_sec=float(base_wait_sec),
        )
        if not raw_trades:
            if not used_from_id:
                page_end = min(request_cursor + 3_599_999, int(until))
                if page_end < int(until):
                    cursor = page_end + 1
                    continue
            if int(until) <= int(_now_ms()):
                terminally_covered = True
                break
            break

        normalized_rows = [normalize_aggtrade_row(row) for row in raw_trades]
        deduped = _validate_and_filter_live_rows(normalized_rows)
        if not deduped:
            if int(normalized_rows[0]["timestamp_ms"]) > int(until):
                terminally_covered = True
                break
            raise ValueError("Live aggTrade page does not make compound cursor progress")

        _commit_batch(
            deduped,
            observed_until_ms=int(deduped[-1]["timestamp_ms"]),
        )
        # Keep the timestamp boundary and advance by aggregate ID.  This retains
        # every same-millisecond trade even when a page ends mid-millisecond.
        cursor = int(last_ts or cursor)

        if len(normalized_rows) < max(1, int(limit)):
            if not used_from_id:
                page_end = min(request_cursor + 3_599_999, int(until))
                if page_end < int(until):
                    cursor = page_end + 1
                    last_trade_id = -1
                    continue
            if int(last_ts) >= int(until) or int(until) <= int(_now_ms()):
                terminally_covered = True
                break
            break

    if not terminally_covered:
        raise ValueError(
            "Incomplete aggTrade continuity: requested interval is not terminally covered"
        )

    terminal_checkpoint = repo.read_raw_checkpoint(
        exchange=stream_exchange,
        symbol=stream_symbol,
        lease=lease,
    )
    if terminal_checkpoint:
        terminal_last_row = dict(terminal_checkpoint["last_row"])
        repo.write_raw_checkpoint(
            exchange=stream_exchange,
            symbol=stream_symbol,
            payload={
                **terminal_checkpoint,
                "last_timestamp_ms": int(terminal_last_row["timestamp_ms"]),
                "last_trade_id": int(terminal_last_row["agg_trade_id"]),
                "observed_until_ms": int(until),
                "updated_at_utc": datetime.now(tz=UTC).isoformat(),
                "last_row": terminal_last_row,
                "last_row_sha256": _checkpoint_last_row_digest(terminal_last_row),
            },
            lease=lease,
        )
    persisted_checkpoint = repo.read_raw_checkpoint(
        exchange=stream_exchange,
        symbol=stream_symbol,
        lease=lease,
    )
    persisted_last_row = dict(persisted_checkpoint["last_row"]) if persisted_checkpoint else None
    return RawAggTradesSyncStats(
        symbol=stream_symbol,
        fetched_rows=int(fetched_rows),
        upserted_rows=int(upserted_rows),
        first_timestamp_ms=first_ts,
        last_timestamp_ms=(
            int(persisted_last_row["timestamp_ms"]) if persisted_last_row is not None else None
        ),
        checkpoint_timestamp_ms=(
            int(persisted_checkpoint["last_timestamp_ms"]) if persisted_checkpoint else None
        ),
        checkpoint_trade_id=(
            int(persisted_checkpoint["last_trade_id"]) if persisted_checkpoint else None
        ),
    )


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
    """Serialize one raw exchange/symbol stream across recovery and publication."""
    from lumina_quant.storage.parquet import ParquetMarketDataRepository

    repo = ParquetMarketDataRepository(str(db_path))
    stream_exchange = str(exchange_id).strip().lower() or "binance"
    stream_symbol = normalize_symbol(symbol)
    lease = repo.acquire_raw_symbol_stream_lease(exchange=stream_exchange, symbol=stream_symbol)
    try:
        return _sync_symbol_aggtrades_raw_under_lease(
            exchange=exchange,
            db_path=db_path,
            exchange_id=stream_exchange,
            symbol=stream_symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=limit,
            max_batches=max_batches,
            retries=retries,
            base_wait_sec=base_wait_sec,
            resume_from_checkpoint=resume_from_checkpoint,
            lease=lease,
        )
    finally:
        lease.release()


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


def normalize_aggtrade_row(trade: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a native Binance or CCXT aggTrade payload into raw schema."""
    if not isinstance(trade, Mapping):
        raise ValueError("aggTrade payload must be a mapping")
    payload = dict(trade)
    info = payload.get("info")
    if info is not None and not isinstance(info, Mapping):
        raise ValueError("aggTrade info must be a mapping")
    native_info = dict(info or {})

    native_keys = {"a", "T", "p", "q", "m"}
    native_mappings = [
        mapping for mapping in (payload, native_info) if native_keys.intersection(mapping)
    ]
    for mapping in native_mappings:
        if native_keys.intersection(mapping) != native_keys:
            raise ValueError("Native Binance aggTrade fields must be complete")

    def _integer(value: Any, *, field: str, minimum: int, native: bool) -> int:
        if native:
            if type(value) is int:
                parsed = value
            elif isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
                parsed = int(value)
            else:
                raise ValueError(f"Native Binance aggTrade {field} must be canonical digits")
        elif type(value) is int:
            parsed = value
        elif isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
            parsed = int(value)
        else:
            raise ValueError(f"aggTrade {field} must be an integer")
        if parsed < minimum:
            raise ValueError(f"aggTrade {field} is out of range")
        return parsed

    def _positive_finite(value: Any, *, field: str, native: bool) -> float:
        if native:
            if (
                not isinstance(value, str)
                or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None
            ):
                raise ValueError(f"Native Binance aggTrade {field} must be a canonical decimal")
            parsed = float(value)
        else:
            if type(value) in (int, float) or (
                isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value)
            ):
                parsed = float(value)
            else:
                raise ValueError(f"aggTrade {field} must be a decoded finite number")
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"aggTrade {field} must be a finite positive number")
        return parsed

    def _maker(value: Any, *, field: str, native: bool) -> bool:
        _ = native
        if type(value) is not bool:
            raise ValueError("aggTrade buyer-maker flag must be a boolean")
        return value

    def _agreed_value(
        keys: tuple[str, ...],
        *,
        field: str,
        parser: Any,
    ) -> Any:
        values = [
            parser(mapping[key], field=field, native=(key in native_keys))
            for mapping in (payload, native_info)
            for key in keys
            if key in mapping
        ]
        if not values:
            raise ValueError(f"aggTrade {field} is missing")
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"aggTrade {field} aliases disagree")
        return values[0]

    agg_trade_id = _agreed_value(
        ("agg_trade_id", "id", "tradeId", "a"),
        field="aggregate ID",
        parser=lambda value, *, field, native: _integer(
            value, field=field, minimum=0, native=native
        ),
    )
    timestamp_ms = _agreed_value(
        ("timestamp_ms", "timestamp", "T"),
        field="timestamp",
        parser=lambda value, *, field, native: _integer(
            value, field=field, minimum=1, native=native
        ),
    )
    price = _agreed_value(
        ("price", "p"),
        field="price",
        parser=lambda value, *, field, native: _positive_finite(value, field=field, native=native),
    )
    quantity = _agreed_value(
        ("amount", "quantity", "q"),
        field="quantity",
        parser=lambda value, *, field, native: _positive_finite(value, field=field, native=native),
    )
    maker = _agreed_value(
        ("is_buyer_maker", "isBuyerMaker", "maker", "m"),
        field="buyer-maker flag",
        parser=_maker,
    )
    return {
        "agg_trade_id": agg_trade_id,
        "timestamp_ms": timestamp_ms,
        "price": price,
        "quantity": quantity,
        "is_buyer_maker": maker,
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
    normalized = [normalize_aggtrade_row(trade) for trade in rows]
    _validate_live_aggtrade_identity_order(normalized)
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


def _normalize_funding_timestamp_ms(timestamp_ms: int) -> int:
    timestamp = int(timestamp_ms)
    remainder = timestamp % _FUNDING_SETTLEMENT_GRANULARITY_MS
    if remainder <= _FUNDING_TIMESTAMP_JITTER_MS:
        return timestamp - remainder
    if remainder >= _FUNDING_SETTLEMENT_GRANULARITY_MS - _FUNDING_TIMESTAMP_JITTER_MS:
        return timestamp + (_FUNDING_SETTLEMENT_GRANULARITY_MS - remainder)
    return timestamp
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
                ts = _normalize_funding_timestamp_ms(int(row.get("fundingTime", 0) or 0))
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
