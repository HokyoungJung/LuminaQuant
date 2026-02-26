"""Synchronize Binance OHLCV data into parquet and optional CSV mirrors."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

from lumina_quant.config import BaseConfig, LiveConfig
from lumina_quant.data_sync import (
    create_binance_exchange,
    parse_timestamp_input,
    sync_market_data,
)


def _build_parser() -> argparse.ArgumentParser:
    default_base_tf = str(os.getenv("LQ_BASE_TIMEFRAME", "1s") or "1s").strip().lower()
    parser = argparse.ArgumentParser(
        description="Sync Binance OHLCV data into market-data parquet storage."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(BaseConfig.SYMBOLS),
        help="Symbols in BASE/QUOTE format (e.g., BTC/USDT ETH/USDT).",
    )
    parser.add_argument(
        "--timeframe",
        default=default_base_tf,
        help="OHLCV timeframe token (e.g., 1m, 5m, 1h, 1d).",
    )
    parser.add_argument(
        "--db-path",
        default=BaseConfig.MARKET_DATA_PARQUET_PATH,
        help="Parquet root path for market OHLCV storage.",
    )
    parser.add_argument(
        "--exchange-id",
        default=BaseConfig.MARKET_DATA_EXCHANGE,
        help="Exchange stream identifier for DB keys.",
    )
    parser.add_argument(
        "--market-type",
        default=LiveConfig.MARKET_TYPE,
        choices=["spot", "future"],
        help="Binance market type for CCXT client.",
    )
    parser.add_argument(
        "--since",
        default="2017-01-01T00:00:00+00:00",
        help="Backfill start (ISO8601 or unix seconds/ms).",
    )
    parser.add_argument(
        "--until",
        default="",
        help="Optional end time (ISO8601 or unix seconds/ms). Default is now UTC.",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Ignore existing DB tail and always fetch from --since.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Per-request OHLCV fetch limit.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100000,
        help="Safety cap for pagination loops per symbol.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for transient exchange API errors.",
    )
    parser.add_argument(
        "--export-csv-dir",
        default="data",
        help="If set, export synchronized DB bars to symbol CSV files under this dir.",
    )
    parser.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Disable CSV export mirrors after DB sync.",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Binance sandbox mode for CCXT client.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    since_ms = parse_timestamp_input(args.since)
    until_ms = parse_timestamp_input(args.until) if args.until else None
    export_csv_dir = None if args.no_export_csv else args.export_csv_dir

    exchange = create_binance_exchange(
        api_key=LiveConfig.BINANCE_API_KEY,
        secret_key=LiveConfig.BINANCE_SECRET_KEY,
        market_type=args.market_type,
        testnet=bool(args.testnet),
    )

    stats = sync_market_data(
        exchange=exchange,
        db_path=args.db_path,
        exchange_id=args.exchange_id,
        symbol_list=args.symbols,
        timeframe=args.timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        force_full=bool(args.force_full),
        limit=max(1, int(args.limit)),
        max_batches=max(1, int(args.max_batches)),
        retries=max(0, int(args.retries)),
        export_csv_dir=export_csv_dir,
    )

    print("\n=== Market Data Sync Summary ===")
    print(f"DB Path: {args.db_path}")
    print(f"Exchange: {args.exchange_id}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Completed at: {datetime.now(UTC).isoformat()}")
    for item in stats:
        print(
            f"- {item.symbol}: fetched={item.fetched_rows} upserted={item.upserted_rows} "
            f"first_ts={item.first_timestamp_ms} last_ts={item.last_timestamp_ms}"
        )


if __name__ == "__main__":
    main()
