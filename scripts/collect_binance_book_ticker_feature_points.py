#!/usr/bin/env python3
"""Collect Binance USD-M bookTicker snapshots into futures feature points.

This is a forward-only BBO sidecar capture path for Binance-first microstructure
research. It writes cadence-bucketed best bid/ask snapshots into the existing
feature-point store without changing any live-execution flags.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import UTC, datetime
from typing import Any

from lumina_quant.market_data import upsert_futures_feature_points_rows
from lumina_quant.research_universe import BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS
from lumina_quant.live.binance_market_stream import (
    BinanceMarketStreamClient,
    BinanceMarketStreamConfig,
    NormalizedBookTicker,
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _bucket_start_ms(timestamp_ms: int, cadence_seconds: int) -> int:
    cadence_ms = max(1, int(cadence_seconds)) * 1000
    return int(timestamp_ms // cadence_ms) * cadence_ms


def _snapshot_row(book: NormalizedBookTicker, *, cadence_seconds: int) -> dict[str, Any]:
    bucket_ms = _bucket_start_ms(int(book.exchange_ts_ms), cadence_seconds)
    return {
        "timestamp_ms": int(bucket_ms),
        "best_bid_price": float(book.bid_price),
        "best_bid_quantity": float(book.bid_quantity),
        "best_ask_price": float(book.ask_price),
        "best_ask_quantity": float(book.ask_quantity),
        "bbo_mid_price": float(book.mid_price),
        "bbo_spread_bps": float(book.spread_bps),
    }


def collect_book_ticker_feature_points(
    *,
    db_path: str,
    symbols: list[str],
    duration_seconds: int,
    cadence_seconds: int,
    exchange: str = "binance",
) -> dict[str, Any]:
    stop_event = threading.Event()
    timer = threading.Timer(max(1, int(duration_seconds)), stop_event.set)
    buckets: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []

    def on_book(book: NormalizedBookTicker) -> None:
        key = (str(book.symbol), _bucket_start_ms(int(book.exchange_ts_ms), cadence_seconds))
        buckets[key] = _snapshot_row(book, cadence_seconds=cadence_seconds)

    def on_trade(_tick: Any) -> None:
        return None

    def on_error(exc: Exception) -> None:
        errors.append(str(exc))

    client = BinanceMarketStreamClient(
        BinanceMarketStreamConfig(
            symbols=list(symbols), include_book_ticker=True, use_agg_trade=False
        )
    )
    timer.start()
    started = time.time()
    try:
        client.run_ws_loop(
            stop_event=stop_event,
            on_trade=on_trade,
            on_book_ticker=on_book,
            on_error=on_error,
        )
    finally:
        stop_event.set()
        timer.cancel()

    per_symbol_rows: dict[str, list[dict[str, Any]]] = {}
    for (symbol, _bucket_ms), row in sorted(
        buckets.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        per_symbol_rows.setdefault(symbol, []).append(dict(row))

    persisted: dict[str, int] = {}
    for symbol, rows in per_symbol_rows.items():
        persisted[symbol] = upsert_futures_feature_points_rows(
            db_path,
            exchange=exchange,
            symbol=symbol,
            rows=rows,
            source=f"binance_book_ticker_{int(cadence_seconds)}s",
        )

    return {
        "artifact_kind": "binance_book_ticker_feature_points_capture",
        "generated_at_utc": _utc_now_iso(),
        "db_path": str(db_path),
        "exchange": str(exchange),
        "symbols": list(symbols),
        "duration_seconds": int(duration_seconds),
        "cadence_seconds": int(cadence_seconds),
        "captured_bucket_count": len(buckets),
        "persisted_rows_by_symbol": persisted,
        "total_persisted_rows": int(sum(persisted.values())),
        "errors": list(errors),
        "started_at_unix": float(started),
        "completed_at_unix": float(time.time()),
    }


def _summary_line(payload: dict[str, Any]) -> str:
    return (
        "bbo-capture"
        f" rows={int(payload.get('total_persisted_rows') or 0)}"
        f" buckets={int(payload.get('captured_bucket_count') or 0)}"
        f" errors={len(list(payload.get('errors') or []))}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/market_parquet")
    parser.add_argument("--symbols", default=",".join(BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS))
    parser.add_argument("--duration-seconds", type=int, default=30)
    parser.add_argument("--cadence-seconds", type=int, default=60)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = collect_book_ticker_feature_points(
        db_path=str(args.db_path),
        symbols=[item.strip() for item in str(args.symbols).split(",") if item.strip()],
        duration_seconds=max(1, int(args.duration_seconds)),
        cadence_seconds=max(1, int(args.cadence_seconds)),
    )
    try:
        if bool(args.summary):
            print(_summary_line(payload))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
