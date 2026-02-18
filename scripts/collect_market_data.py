"""Ensure local DB market-data coverage for requested symbols/time windows."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

from lumina_quant.config import BaseConfig, LiveConfig
from lumina_quant.data_collector import auto_collect_market_data


def _parse_datetime_input(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        numeric = int(text)
        if abs(numeric) < 100_000_000_000:
            numeric *= 1000
        return datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _build_parser() -> argparse.ArgumentParser:
    default_base_tf = str(os.getenv("LQ_BASE_TIMEFRAME", "1s") or "1s").strip().lower()
    parser = argparse.ArgumentParser(
        description="Collect and fill missing OHLCV coverage into SQLite market-data DB."
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
        default=BaseConfig.MARKET_DATA_SQLITE_PATH,
        help="SQLite DB path for market OHLCV storage.",
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
        default="",
        help="Coverage start (ISO8601 or unix seconds/ms). Optional.",
    )
    parser.add_argument(
        "--until",
        default="",
        help="Coverage end (ISO8601 or unix seconds/ms). Optional; default now UTC.",
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
        "--testnet",
        action="store_true",
        help="Use Binance sandbox mode for CCXT client.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    since_dt = _parse_datetime_input(args.since)
    until_dt = _parse_datetime_input(args.until)

    stats = auto_collect_market_data(
        symbol_list=list(args.symbols),
        timeframe=str(args.timeframe),
        db_path=str(args.db_path),
        exchange_id=str(args.exchange_id),
        market_type=str(args.market_type),
        since_dt=since_dt,
        until_dt=until_dt,
        api_key=str(LiveConfig.BINANCE_API_KEY or ""),
        secret_key=str(LiveConfig.BINANCE_SECRET_KEY or ""),
        testnet=bool(args.testnet),
        limit=max(1, int(args.limit)),
        max_batches=max(1, int(args.max_batches)),
        retries=max(0, int(args.retries)),
    )

    print("\n=== Market Data Collector Summary ===")
    print(f"DB Path: {args.db_path}")
    print(f"Exchange: {args.exchange_id}")
    print(f"Market Type: {args.market_type}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Symbols: {', '.join(args.symbols)}")
    for item in stats:
        print(
            f"- {item['symbol']}: fetched={item['fetched_rows']} upserted={item['upserted_rows']} "
            f"first_ts={item['first_timestamp_ms']} last_ts={item['last_timestamp_ms']}"
        )


if __name__ == "__main__":
    main()
