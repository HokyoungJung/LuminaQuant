#!/usr/bin/env python3
"""Backfill official Binance USD-M bookTicker history into feature points.

This uses Binance's own public historical archive on ``data.binance.vision`` for
USD-M ``bookTicker`` files. It normalizes the archived CSVs into the existing
feature-point store so BBO-aware clean research can accumulate real historical
train/validation coverage instead of only forward sidecar data.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile

import polars as pl

from lumina_quant.data_sync import _download_zip_bytes
from lumina_quant.market_data import normalize_symbol, upsert_futures_feature_points_rows

try:
    from import_binance_book_ticker_history import normalize_bbo_frame
except ModuleNotFoundError:  # pragma: no cover - package import path under pytest
    from scripts.import_binance_book_ticker_history import normalize_bbo_frame


def public_book_ticker_archive_url(symbol: str, day_value: date) -> str:
    compact = normalize_symbol(symbol).replace("/", "")
    day_token = day_value.strftime("%Y-%m-%d")
    return (
        "https://data.binance.vision/data/futures/um/daily/bookTicker/"
        f"{compact}/{compact}-bookTicker-{day_token}.zip"
    )


def _iter_days(start_day: date, end_day: date) -> list[date]:
    if end_day < start_day:
        raise ValueError("end_date must be on or after start_date")
    out: list[date] = []
    cur = start_day
    while cur <= end_day:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def _read_first_csv_from_zip(blob: bytes) -> pl.DataFrame:
    with ZipFile(BytesIO(blob)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError("zip archive did not contain a CSV member")
        with archive.open(names[0]) as member:
            return pl.read_csv(member)


def _first_csv_name(blob: bytes) -> str:
    with ZipFile(BytesIO(blob)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError("zip archive did not contain a CSV member")
        return names[0]


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_none(value: object) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None or parsed != parsed:
        return None
    return int(parsed)


def _row_value(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _read_cadence_sampled_csv_from_zip(
    blob: bytes,
    *,
    symbol: str,
    cadence_seconds: int,
) -> tuple[pl.DataFrame, int]:
    cadence_ms = int(cadence_seconds) * 1000
    if cadence_ms <= 0:
        raise ValueError("cadence_seconds must be positive")
    normalized_symbol = normalize_symbol(symbol)
    rows: list[dict[str, object]] = []
    seen_buckets: set[int] = set()
    archive_rows = 0
    with ZipFile(BytesIO(blob)) as archive:
        name = _first_csv_name(blob)
        with archive.open(name) as member:
            reader = csv.DictReader(TextIOWrapper(member, encoding="utf-8"))
            for raw in reader:
                archive_rows += 1
                timestamp_ms = _int_or_none(
                    _row_value(
                        raw,
                        ("transaction_time", "timestamp_ms", "time", "timestamp", "event_time"),
                    )
                )
                if timestamp_ms is None:
                    continue
                bucket_ms = timestamp_ms // cadence_ms * cadence_ms
                if bucket_ms in seen_buckets:
                    continue
                bid = _float_or_none(
                    _row_value(raw, ("best_bid_price", "bid_price", "best_bid", "bid", "b"))
                )
                ask = _float_or_none(
                    _row_value(raw, ("best_ask_price", "ask_price", "best_ask", "ask", "a"))
                )
                if bid is None or ask is None:
                    continue
                bid_qty = _float_or_none(
                    _row_value(
                        raw,
                        ("best_bid_qty", "best_bid_quantity", "bid_quantity", "bid_qty", "B"),
                    )
                )
                ask_qty = _float_or_none(
                    _row_value(
                        raw,
                        ("best_ask_qty", "best_ask_quantity", "ask_quantity", "ask_qty", "A"),
                    )
                )
                mid = (bid + ask) / 2.0
                rows.append(
                    {
                        "timestamp_ms": timestamp_ms,
                        "symbol": normalized_symbol,
                        "best_bid_price": bid,
                        "best_bid_quantity": bid_qty,
                        "best_ask_price": ask,
                        "best_ask_quantity": ask_qty,
                        "bbo_mid_price": mid,
                        "bbo_spread_bps": ((ask - bid) / mid * 10000.0) if mid > 0.0 else None,
                    }
                )
                seen_buckets.add(bucket_ms)
    frame = pl.DataFrame(rows).sort(["symbol", "timestamp_ms"]) if rows else pl.DataFrame()
    return frame, archive_rows


def _sample_by_cadence(frame: pl.DataFrame, *, cadence_seconds: int | None) -> pl.DataFrame:
    if cadence_seconds is None:
        return frame
    cadence_ms = int(cadence_seconds) * 1000
    if cadence_ms <= 0:
        raise ValueError("cadence_seconds must be positive")
    return (
        frame.with_columns((pl.col("timestamp_ms") // cadence_ms * cadence_ms).alias("_bucket_ms"))
        .sort(["symbol", "timestamp_ms"])
        .group_by(["symbol", "_bucket_ms"], maintain_order=True)
        .first()
        .drop("_bucket_ms")
        .sort(["symbol", "timestamp_ms"])
    )


def backfill_public_book_ticker_history(
    *,
    db_path: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    exchange: str = "binance",
    cadence_seconds: int | None = None,
    max_rows_per_archive: int | None = None,
    retries: int = 2,
    base_wait_sec: float = 1.0,
) -> dict[str, object]:
    persisted_rows_by_symbol: dict[str, int] = {normalize_symbol(symbol): 0 for symbol in symbols}
    imported_archives: list[dict[str, object]] = []
    missing_archives: list[dict[str, object]] = []
    days = _iter_days(start_date, end_date)
    for symbol in symbols:
        normalized_symbol = normalize_symbol(symbol)
        for day_value in days:
            url = public_book_ticker_archive_url(normalized_symbol, day_value)
            blob = _download_zip_bytes(
                url, retries=int(retries), base_wait_sec=float(base_wait_sec)
            )
            if blob is None:
                missing_archives.append(
                    {"symbol": normalized_symbol, "date": day_value.isoformat(), "url": url}
                )
                continue
            if cadence_seconds is None:
                frame = _read_first_csv_from_zip(blob)
                normalized = normalize_bbo_frame(frame, symbol_override=normalized_symbol)
                archive_rows = int(normalized.height)
            else:
                normalized, archive_rows = _read_cadence_sampled_csv_from_zip(
                    blob,
                    symbol=normalized_symbol,
                    cadence_seconds=int(cadence_seconds),
                )
            if max_rows_per_archive is not None:
                normalized = normalized.head(int(max_rows_per_archive))
            rows = normalized.drop("symbol").to_dicts()
            persisted = upsert_futures_feature_points_rows(
                db_path,
                exchange=str(exchange),
                symbol=normalized_symbol,
                rows=rows,
                source="binance_public_book_ticker_archive",
            )
            persisted_rows_by_symbol[normalized_symbol] += int(persisted)
            imported_archives.append(
                {
                    "symbol": normalized_symbol,
                    "date": day_value.isoformat(),
                    "url": url,
                    "archive_rows": archive_rows,
                    "normalized_rows": int(normalized.height),
                    "persisted_rows": int(persisted),
                }
            )
    return {
        "artifact_kind": "binance_public_book_ticker_history_backfill",
        "exchange": str(exchange),
        "db_path": str(db_path),
        "symbols": [normalize_symbol(symbol) for symbol in symbols],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "imported_archive_count": len(imported_archives),
        "missing_archive_count": len(missing_archives),
        "cadence_seconds": cadence_seconds,
        "max_rows_per_archive": max_rows_per_archive,
        "persisted_rows_by_symbol": persisted_rows_by_symbol,
        "total_persisted_rows": int(sum(persisted_rows_by_symbol.values())),
        "imported_archives": imported_archives,
        "missing_archives": missing_archives,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/market_parquet")
    parser.add_argument("--symbols", required=True, help="Comma-separated Binance USD-M symbols")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--base-wait-sec", type=float, default=1.0)
    parser.add_argument(
        "--cadence-seconds",
        type=int,
        default=None,
        help="Optionally keep only the first BBO row in each cadence bucket.",
    )
    parser.add_argument(
        "--max-rows-per-archive",
        type=int,
        default=None,
        help="Optional safety cap after cadence sampling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = backfill_public_book_ticker_history(
        db_path=str(args.db_path),
        symbols=[token.strip() for token in str(args.symbols).split(",") if token.strip()],
        start_date=datetime.strptime(str(args.start_date), "%Y-%m-%d").date(),
        end_date=datetime.strptime(str(args.end_date), "%Y-%m-%d").date(),
        exchange=str(args.exchange),
        retries=int(args.retries),
        base_wait_sec=float(args.base_wait_sec),
        cadence_seconds=args.cadence_seconds,
        max_rows_per_archive=args.max_rows_per_archive,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
