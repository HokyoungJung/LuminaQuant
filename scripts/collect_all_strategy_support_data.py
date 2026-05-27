"""Collect futures support-data aligned to the current OHLCV coverage."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.config import BaseConfig
from lumina_quant.data.support_inventory import (
    build_strategy_support_inventory,
    write_strategy_support_inventory,
)
from lumina_quant.data_sync import parse_timestamp_input, sync_futures_feature_points
from lumina_quant.market_data import load_futures_feature_points_from_db
from lumina_quant.symbols import canonical_symbol


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect strategy support-data (funding/OI/mark/index/liquidations)."
    )
    parser.add_argument("--symbols", nargs="*", default=list(getattr(BaseConfig, "SYMBOLS", [])))
    parser.add_argument(
        "--db-path", default=getattr(BaseConfig, "MARKET_DATA_PARQUET_PATH", "data/market_parquet")
    )
    parser.add_argument(
        "--exchange-id", default=getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance")
    )
    parser.add_argument(
        "--since",
        default="",
        help="Optional global lower bound. Default=per-symbol OHLCV start / resume.",
    )
    parser.add_argument(
        "--until", default="", help="Optional global upper bound. Default=per-symbol latest OHLCV."
    )
    parser.add_argument("--mark-index-interval", default="1m")
    parser.add_argument("--open-interest-period", default="5m")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Ignore feature-point resume and backfill full OHLCV-covered range.",
    )
    parser.add_argument(
        "--report-path",
        default="var/reports/strategy_support_data_collection_latest.json",
        help="JSON summary output path.",
    )
    parser.add_argument(
        "--inventory-json-path",
        default="var/reports/strategy_support_inventory_latest.json",
        help="Canonical JSON inventory output path.",
    )
    parser.add_argument(
        "--inventory-csv-path",
        default="var/reports/strategy_support_inventory_latest.csv",
        help="Canonical CSV inventory output path.",
    )
    parser.add_argument(
        "--skip-inventory",
        action="store_true",
        help="Skip post-collection inventory generation.",
    )
    return parser


def _symbol_ohlcv_bounds(
    *, root: Path, exchange_id: str, symbol: str
) -> tuple[int | None, int | None, int]:
    compact = canonical_symbol(symbol).replace("/", "")
    pattern = root / "market_ohlcv_1s" / str(exchange_id).lower() / compact / "*.parquet"
    files = sorted(pattern.parent.glob(pattern.name))
    if not files:
        return None, None, 0

    frame = (
        pl.scan_parquet([str(path) for path in files])
        .select(
            [
                pl.min("datetime").dt.epoch("ms").alias("min_ts"),
                pl.max("datetime").dt.epoch("ms").alias("max_ts"),
                pl.len().alias("rows"),
            ]
        )
        .collect()
    )
    if frame.is_empty():
        return None, None, 0
    row = frame.row(0, named=True)
    return (
        int(row["min_ts"]) if row["min_ts"] is not None else None,
        int(row["max_ts"]) if row["max_ts"] is not None else None,
        int(row["rows"] or 0),
    )


def _feature_last_timestamp(*, db_path: str, exchange_id: str, symbol: str) -> int | None:
    frame = load_futures_feature_points_from_db(
        db_path,
        exchange=str(exchange_id),
        symbol=str(symbol),
    )
    if frame.is_empty():
        return None
    maximum = frame.get_column("timestamp_ms").max()
    return int(maximum) if maximum is not None else None


def _iso_utc(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).isoformat()


def _collect_symbol(
    *,
    db_path: str,
    exchange_id: str,
    symbol: str,
    since_ms: int | None,
    until_ms: int | None,
    mark_index_interval: str,
    open_interest_period: str,
    retries: int,
    force_full: bool,
) -> dict[str, Any]:
    root = Path(db_path)
    ohlcv_min_ts, ohlcv_max_ts, ohlcv_rows = _symbol_ohlcv_bounds(
        root=root,
        exchange_id=exchange_id,
        symbol=symbol,
    )
    if ohlcv_min_ts is None or ohlcv_max_ts is None:
        return {
            "symbol": symbol,
            "status": "skipped_no_ohlcv",
            "ohlcv_rows": int(ohlcv_rows),
        }

    feature_last_ts = (
        None
        if force_full
        else _feature_last_timestamp(
            db_path=db_path,
            exchange_id=exchange_id,
            symbol=symbol,
        )
    )

    start_ms = int(ohlcv_min_ts)
    if since_ms is not None:
        start_ms = max(int(start_ms), int(since_ms))
    if feature_last_ts is not None:
        start_ms = max(int(start_ms), int(feature_last_ts + 1))

    end_ms = int(ohlcv_max_ts)
    if until_ms is not None:
        end_ms = min(int(end_ms), int(until_ms))

    if start_ms > end_ms:
        return {
            "symbol": symbol,
            "status": "up_to_date",
            "ohlcv_rows": int(ohlcv_rows),
            "ohlcv_start_ms": int(ohlcv_min_ts),
            "ohlcv_end_ms": int(ohlcv_max_ts),
            "feature_last_ts": feature_last_ts,
        }

    stats = sync_futures_feature_points(
        db_path=db_path,
        exchange_id=exchange_id,
        symbol_list=[symbol],
        since_ms=int(start_ms),
        until_ms=int(end_ms),
        mark_index_interval=str(mark_index_interval),
        open_interest_period=str(open_interest_period),
        retries=max(0, int(retries)),
    )
    summary = stats[0] if stats else None
    return {
        "symbol": symbol,
        "status": "ok",
        "ohlcv_rows": int(ohlcv_rows),
        "ohlcv_start_ms": int(ohlcv_min_ts),
        "ohlcv_end_ms": int(ohlcv_max_ts),
        "feature_last_ts": feature_last_ts,
        "requested_start_ms": int(start_ms),
        "requested_end_ms": int(end_ms),
        "upserted_rows": int(getattr(summary, "upserted_rows", 0) or 0),
        "first_timestamp_ms": getattr(summary, "first_timestamp_ms", None),
        "last_timestamp_ms": getattr(summary, "last_timestamp_ms", None),
    }


def main() -> None:
    args = _build_parser().parse_args()

    db_path = str(args.db_path)
    exchange_id = str(args.exchange_id)
    symbols = [
        canonical_symbol(symbol) for symbol in list(args.symbols or []) if str(symbol).strip()
    ]
    since_ms = parse_timestamp_input(args.since or None)
    until_ms = parse_timestamp_input(args.until or None)
    report_path = Path(str(args.report_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for symbol in symbols:
        print(f"[collect] {symbol}")
        try:
            result = _collect_symbol(
                db_path=db_path,
                exchange_id=exchange_id,
                symbol=symbol,
                since_ms=since_ms,
                until_ms=until_ms,
                mark_index_interval=str(args.mark_index_interval),
                open_interest_period=str(args.open_interest_period),
                retries=max(0, int(args.retries)),
                force_full=bool(args.force_full),
            )
        except Exception as exc:
            result = {
                "symbol": symbol,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "db_path": db_path,
        "exchange_id": exchange_id,
        "since_ms": since_ms,
        "until_ms": until_ms,
        "mark_index_interval": str(args.mark_index_interval),
        "open_interest_period": str(args.open_interest_period),
        "force_full": bool(args.force_full),
        "symbols": [
            {
                **row,
                "ohlcv_start_utc": _iso_utc(row.get("ohlcv_start_ms")),
                "ohlcv_end_utc": _iso_utc(row.get("ohlcv_end_ms")),
                "feature_last_utc": _iso_utc(row.get("feature_last_ts")),
                "requested_start_utc": _iso_utc(row.get("requested_start_ms")),
                "requested_end_utc": _iso_utc(row.get("requested_end_ms")),
                "first_timestamp_utc": _iso_utc(row.get("first_timestamp_ms")),
                "last_timestamp_utc": _iso_utc(row.get("last_timestamp_ms")),
            }
            for row in results
        ],
    }
    if not bool(args.skip_inventory):
        inventory_payload = build_strategy_support_inventory(
            db_path=db_path,
            exchange=exchange_id,
            symbols=symbols,
        )
        inventory_outputs = write_strategy_support_inventory(
            payload=inventory_payload,
            json_path=str(args.inventory_json_path),
            csv_path=str(args.inventory_csv_path),
        )
        payload["inventory"] = {
            "symbol_count": int(inventory_payload["symbol_count"]),
            **inventory_outputs,
        }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {report_path}")


if __name__ == "__main__":
    main()
