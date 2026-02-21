"""Collect canonical 1s OHLCV plus futures feature points.

This workflow keeps 1s OHLCV as the canonical base stream and augments it
with derivatives-related feature points needed for research:
- funding history
- mark/index price klines
- open-interest history
- force-liquidation orders
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from lumina_quant.config import BaseConfig, LiveConfig
from lumina_quant.data_sync import (
    create_binance_exchange,
    parse_timestamp_input,
    sync_futures_feature_points,
    sync_market_data,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect 1s OHLCV and futures feature points into SQLite."
    )
    parser.add_argument("--symbols", nargs="+", default=list(BaseConfig.SYMBOLS))
    parser.add_argument("--db-path", default=BaseConfig.MARKET_DATA_SQLITE_PATH)
    parser.add_argument("--exchange-id", default=BaseConfig.MARKET_DATA_EXCHANGE)
    parser.add_argument("--since", default="2021-01-01T00:00:00+00:00")
    parser.add_argument("--until", default="")
    parser.add_argument("--market-type", choices=["spot", "future"], default="future")
    parser.add_argument("--ohlcv-limit", type=int, default=1000)
    parser.add_argument("--ohlcv-max-batches", type=int, default=100000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--mark-index-interval", default="1m")
    parser.add_argument("--open-interest-period", default="5m")
    parser.add_argument("--skip-ohlcv", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    since_ms = parse_timestamp_input(args.since)
    until_ms = parse_timestamp_input(args.until) if args.until else None
    if since_ms is None:
        raise SystemExit("--since must resolve to a valid timestamp")
    effective_until = (
        int(until_ms) if until_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
    )

    exchange = create_binance_exchange(
        api_key=str(LiveConfig.BINANCE_API_KEY or ""),
        secret_key=str(LiveConfig.BINANCE_SECRET_KEY or ""),
        market_type=str(args.market_type),
        testnet=bool(LiveConfig.IS_TESTNET),
    )
    try:
        if not args.skip_ohlcv:
            ohlcv_stats = sync_market_data(
                exchange=exchange,
                db_path=str(args.db_path),
                exchange_id=str(args.exchange_id),
                symbol_list=list(args.symbols),
                timeframe="1s",
                since_ms=int(since_ms),
                until_ms=effective_until,
                force_full=False,
                limit=max(1, int(args.ohlcv_limit)),
                max_batches=max(1, int(args.ohlcv_max_batches)),
                retries=max(0, int(args.retries)),
                export_csv_dir="data",
            )
            print("\n=== 1s OHLCV Sync ===")
            for row in ohlcv_stats:
                print(
                    f"- {row.symbol}: upserted={row.upserted_rows} first_ts={row.first_timestamp_ms} last_ts={row.last_timestamp_ms}"
                )

        if not args.skip_features:
            feature_stats = sync_futures_feature_points(
                db_path=str(args.db_path),
                exchange_id=str(args.exchange_id),
                symbol_list=list(args.symbols),
                since_ms=int(since_ms),
                until_ms=effective_until,
                mark_index_interval=str(args.mark_index_interval),
                open_interest_period=str(args.open_interest_period),
                retries=max(0, int(args.retries)),
            )
            print("\n=== Futures Feature Points Sync ===")
            for row in feature_stats:
                print(
                    f"- {row.symbol}: upserted={row.upserted_rows} first_ts={row.first_timestamp_ms} last_ts={row.last_timestamp_ms}"
                )
    finally:
        close_fn = getattr(exchange, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
