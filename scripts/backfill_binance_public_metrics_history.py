#!/usr/bin/env python3
"""Backfill official Binance USD-M daily metrics into the feature-point store."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from lumina_quant.data_sync import _download_zip_bytes
from lumina_quant.market_data import normalize_symbol, upsert_futures_feature_points_rows


def public_metrics_archive_url(symbol: str, day_value: date) -> str:
    compact = normalize_symbol(symbol).replace("/", "")
    token = day_value.isoformat()
    return (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        f"{compact}/{compact}-metrics-{token}.zip"
    )


def _iter_days(start_day: date, end_day: date) -> list[date]:
    if end_day < start_day:
        raise ValueError("end_date must be on or after start_date")
    days: list[date] = []
    cursor = start_day
    while cursor <= end_day:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def metrics_rows_from_zip(
    blob: bytes,
    *,
    expected_symbol: str,
    expected_day: date,
) -> tuple[list[dict[str, Any]], str]:
    compact = normalize_symbol(expected_symbol).replace("/", "")
    expected_member = f"{compact}-metrics-{expected_day.isoformat()}.csv"
    window_start = datetime.combine(expected_day, datetime.min.time(), tzinfo=UTC)
    window_end = window_start + timedelta(days=1)
    with ZipFile(BytesIO(blob)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if csv_names != [expected_member]:
            raise ValueError(f"unexpected metrics archive members: {csv_names!r}")
        with archive.open(expected_member) as member:
            reader = csv.DictReader(TextIOWrapper(member, encoding="utf-8"))
            required = {"create_time", "symbol", "sum_open_interest", "sum_open_interest_value"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("metrics archive is missing required columns")
            rows: list[dict[str, Any]] = []
            previous_timestamp_ms: int | None = None
            for raw in reader:
                if str(raw.get("symbol") or "").strip().upper() != compact:
                    raise ValueError("metrics archive symbol does not match request")
                timestamp = datetime.fromisoformat(str(raw["create_time"]).strip()).replace(tzinfo=UTC)
                if not window_start <= timestamp <= window_end:
                    raise ValueError("metrics archive row is outside its [start, end] UTC window")
                timestamp_ms = int(timestamp.timestamp() * 1000)
                if previous_timestamp_ms is not None and timestamp_ms <= previous_timestamp_ms:
                    raise ValueError("metrics archive timestamps are not strictly increasing")
                value_token = raw.get("sum_open_interest_value") or raw.get("sum_open_interest")
                open_interest = float(value_token) if value_token not in (None, "") else math.nan
                if not math.isfinite(open_interest) or open_interest < 0.0:
                    raise ValueError("metrics archive open interest is invalid")
                rows.append({"timestamp_ms": timestamp_ms, "open_interest": open_interest})
                previous_timestamp_ms = timestamp_ms
    if not rows:
        raise ValueError("metrics archive contains no rows")
    return rows, expected_member


def backfill_public_metrics_history(
    *,
    db_path: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    exchange: str = "binance",
    retries: int = 2,
    base_wait_sec: float = 1.0,
) -> dict[str, Any]:
    normalized_symbols = [normalize_symbol(symbol) for symbol in symbols]
    persisted_by_symbol = dict.fromkeys(normalized_symbols, 0)
    imported: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for symbol in normalized_symbols:
        for day_value in _iter_days(start_date, end_date):
            url = public_metrics_archive_url(symbol, day_value)
            blob = _download_zip_bytes(
                url,
                retries=max(0, int(retries)),
                base_wait_sec=max(0.0, float(base_wait_sec)),
            )
            if blob is None:
                missing.append({"symbol": symbol, "date": day_value.isoformat(), "url": url})
                continue
            rows, member = metrics_rows_from_zip(
                blob,
                expected_symbol=symbol,
                expected_day=day_value,
            )
            persisted = upsert_futures_feature_points_rows(
                db_path,
                exchange=exchange,
                symbol=symbol,
                rows=rows,
                source="binance_public_metrics_archive",
            )
            persisted_by_symbol[symbol] += int(persisted)
            imported.append(
                {
                    "symbol": symbol,
                    "date": day_value.isoformat(),
                    "url": url,
                    "archive_member": member,
                    "archive_sha256": hashlib.sha256(blob).hexdigest(),
                    "archive_byte_count": len(blob),
                    "row_count": len(rows),
                    "persisted_rows": int(persisted),
                    "first_timestamp_ms": rows[0]["timestamp_ms"],
                    "last_timestamp_ms": rows[-1]["timestamp_ms"],
                }
            )
    return {
        "artifact_kind": "binance_public_metrics_history_backfill",
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "Binance public data.vision USD-M daily metrics",
        "exchange": exchange,
        "db_path": db_path,
        "symbols": normalized_symbols,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "imported_archive_count": len(imported),
        "missing_archive_count": len(missing),
        "persisted_rows_by_symbol": persisted_by_symbol,
        "total_persisted_rows": sum(persisted_by_symbol.values()),
        "imported_archives": imported,
        "missing_archives": missing,
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
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = backfill_public_metrics_history(
        db_path=str(args.db_path),
        symbols=[token.strip() for token in str(args.symbols).split(",") if token.strip()],
        start_date=datetime.strptime(str(args.start_date), "%Y-%m-%d").date(),
        end_date=datetime.strptime(str(args.end_date), "%Y-%m-%d").date(),
        exchange=str(args.exchange),
        retries=int(args.retries),
        base_wait_sec=float(args.base_wait_sec),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if str(args.output).strip():
        output = Path(str(args.output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if payload["missing_archive_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
