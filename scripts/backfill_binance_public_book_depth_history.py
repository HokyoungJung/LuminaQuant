#!/usr/bin/env python3
"""Backfill official Binance USD-M bookDepth history into feature points.

Binance public-data ``bookDepth`` archives are available through current fold
windows even where historical ``bookTicker`` archives are not. They do not carry
exact top-of-book prices, so this importer stores depth-derived microstructure
features separately from BBO columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, date, datetime, timedelta
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile

from lumina_quant.data_sync import _download_zip_bytes
from lumina_quant.market_data import normalize_symbol, upsert_futures_feature_points_rows


def public_book_depth_archive_url(symbol: str, day_value: date) -> str:
    compact = normalize_symbol(symbol).replace("/", "")
    day_token = day_value.strftime("%Y-%m-%d")
    return (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        f"{compact}/{compact}-bookDepth-{day_token}.zip"
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


def _first_csv_name(blob: bytes) -> str:
    with ZipFile(BytesIO(blob)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError("zip archive did not contain a CSV member")
        return names[0]


def _timestamp_ms(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


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


def _rows_from_book_depth_zip(
    blob: bytes, *, cadence_seconds: int | None = None
) -> tuple[list[dict[str, object]], int]:
    cadence_ms = int(cadence_seconds) * 1000 if cadence_seconds is not None else None
    if cadence_ms is not None and cadence_ms <= 0:
        raise ValueError("cadence_seconds must be positive")
    by_timestamp: dict[int, dict[str, float]] = {}
    scanned = 0
    with ZipFile(BytesIO(blob)) as archive:
        name = _first_csv_name(blob)
        with archive.open(name) as member:
            reader = csv.DictReader(TextIOWrapper(member, encoding="utf-8"))
            for raw in reader:
                scanned += 1
                ts = _timestamp_ms(raw.get("timestamp") or raw.get("time") or "")
                pct = _float_or_none(raw.get("percentage"))
                notional = _float_or_none(raw.get("notional"))
                if ts is None or pct is None or notional is None or pct == 0.0:
                    continue
                if cadence_ms is not None:
                    ts = ts // cadence_ms * cadence_ms
                slot = by_timestamp.setdefault(ts, {})
                if pct < 0.0:
                    current_pct = slot.get("bid_pct")
                    if current_pct is None or abs(pct) < abs(current_pct):
                        slot["bid_pct"] = pct
                        slot["bid_notional"] = notional
                else:
                    current_pct = slot.get("ask_pct")
                    if current_pct is None or abs(pct) < abs(current_pct):
                        slot["ask_pct"] = pct
                        slot["ask_notional"] = notional
    rows: list[dict[str, object]] = []
    for ts, slot in sorted(by_timestamp.items()):
        bid = slot.get("bid_notional")
        ask = slot.get("ask_notional")
        denom = abs(bid or 0.0) + abs(ask or 0.0)
        rows.append(
            {
                "timestamp_ms": int(ts),
                "book_depth_bid_notional_1pct": bid,
                "book_depth_ask_notional_1pct": ask,
                "book_depth_imbalance_1pct": ((bid or 0.0) - (ask or 0.0)) / denom
                if denom > 0.0
                else None,
            }
        )
    return rows, scanned


def backfill_public_book_depth_history(
    *,
    db_path: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    exchange: str = "binance",
    cadence_seconds: int | None = 3600,
    retries: int = 2,
    base_wait_sec: float = 1.0,
) -> dict[str, object]:
    persisted_rows_by_symbol: dict[str, int] = {normalize_symbol(symbol): 0 for symbol in symbols}
    imported_archives: list[dict[str, object]] = []
    missing_archives: list[dict[str, object]] = []
    for symbol in symbols:
        normalized_symbol = normalize_symbol(symbol)
        symbol_rows: list[dict[str, object]] = []
        symbol_archive_indexes: list[int] = []
        for day_value in _iter_days(start_date, end_date):
            url = public_book_depth_archive_url(normalized_symbol, day_value)
            blob = _download_zip_bytes(
                url, retries=int(retries), base_wait_sec=float(base_wait_sec)
            )
            if blob is None:
                missing_archives.append(
                    {"symbol": normalized_symbol, "date": day_value.isoformat(), "url": url}
                )
                continue
            rows, scanned = _rows_from_book_depth_zip(blob, cadence_seconds=cadence_seconds)
            symbol_rows.extend(rows)
            symbol_archive_indexes.append(len(imported_archives))
            imported_archives.append(
                {
                    "symbol": normalized_symbol,
                    "date": day_value.isoformat(),
                    "url": url,
                    "archive_rows": int(scanned),
                    "normalized_rows": len(rows),
                    "queued_rows": len(rows),
                }
            )
        persisted = 0
        if symbol_rows:
            persisted = upsert_futures_feature_points_rows(
                db_path,
                exchange=str(exchange),
                symbol=normalized_symbol,
                rows=symbol_rows,
                source="binance_public_book_depth_archive",
            )
        persisted_rows_by_symbol[normalized_symbol] += int(persisted)
        for archive_index in symbol_archive_indexes:
            imported_archives[archive_index]["symbol_persisted_rows"] = int(persisted)
    return {
        "artifact_kind": "binance_public_book_depth_history_backfill",
        "exchange": str(exchange),
        "db_path": str(db_path),
        "symbols": [normalize_symbol(symbol) for symbol in symbols],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "cadence_seconds": cadence_seconds,
        "imported_archive_count": len(imported_archives),
        "missing_archive_count": len(missing_archives),
        "persisted_rows_by_symbol": persisted_rows_by_symbol,
        "total_persisted_rows": int(sum(persisted_rows_by_symbol.values())),
        "imported_archives": imported_archives,
        "missing_archives": missing_archives,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/market_parquet")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--cadence-seconds", type=int, default=3600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--base-wait-sec", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = backfill_public_book_depth_history(
        db_path=str(args.db_path),
        symbols=[token.strip() for token in str(args.symbols).split(",") if token.strip()],
        start_date=datetime.strptime(str(args.start_date), "%Y-%m-%d").date(),
        end_date=datetime.strptime(str(args.end_date), "%Y-%m-%d").date(),
        exchange=str(args.exchange),
        cadence_seconds=args.cadence_seconds,
        retries=int(args.retries),
        base_wait_sec=float(args.base_wait_sec),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
