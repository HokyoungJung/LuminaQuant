"""Migrate SQLite market_ohlcv rows into InfluxDB market_ohlcv measurement."""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Sequence

from lumina_quant.influx_market_data import InfluxMarketDataRepository
from lumina_quant.market_data import normalize_symbol, normalize_timeframe_token


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate SQLite market_ohlcv rows to InfluxDB market_ohlcv."
    )
    parser.add_argument("--db-path", default="data/lq_market.sqlite3")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--timeframes", nargs="+", default=[])
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--influx-url", default="")
    parser.add_argument("--influx-org", default="")
    parser.add_argument("--influx-bucket", default="")
    parser.add_argument("--influx-token", default="")
    parser.add_argument("--influx-token-env", default="INFLUXDB_TOKEN")
    return parser


def _resolve_influx_config(args: argparse.Namespace) -> tuple[str, str, str, str]:
    token_env = str(args.influx_token_env or "INFLUXDB_TOKEN").strip() or "INFLUXDB_TOKEN"
    url = str(
        args.influx_url or os.getenv("LQ__STORAGE__INFLUX_URL") or os.getenv("INFLUX_URL") or ""
    ).strip()
    org = str(
        args.influx_org or os.getenv("LQ__STORAGE__INFLUX_ORG") or os.getenv("INFLUX_ORG") or ""
    ).strip()
    bucket = str(
        args.influx_bucket
        or os.getenv("LQ__STORAGE__INFLUX_BUCKET")
        or os.getenv("INFLUX_BUCKET")
        or ""
    ).strip()
    token = str(args.influx_token or os.getenv(token_env, "") or "").strip()
    if not (url and org and bucket and token):
        raise SystemExit(
            "Influx config missing. Provide --influx-url/--influx-org/--influx-bucket and token "
            "(--influx-token or env via --influx-token-env)."
        )
    return url, org, bucket, token


def _fetch_distinct(
    conn: sqlite3.Connection,
    *,
    exchange: str,
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[tuple[str, str]]:
    stream_exchange = str(exchange).strip().lower()
    clauses = ["exchange = ?"]
    params: list[object] = [stream_exchange]

    normalized_symbols = [normalize_symbol(sym) for sym in symbols if str(sym).strip()]
    if normalized_symbols:
        placeholders = ",".join("?" for _ in normalized_symbols)
        clauses.append(f"symbol IN ({placeholders})")
        params.extend(normalized_symbols)

    normalized_timeframes = [normalize_timeframe_token(tf) for tf in timeframes if str(tf).strip()]
    if normalized_timeframes:
        placeholders = ",".join("?" for _ in normalized_timeframes)
        clauses.append(f"timeframe IN ({placeholders})")
        params.extend(normalized_timeframes)

    where_clause = " AND ".join(clauses)
    query = (
        "SELECT DISTINCT symbol, timeframe FROM market_ohlcv "
        f"WHERE {where_clause} ORDER BY symbol, timeframe"
    )
    rows = conn.execute(query, tuple(params)).fetchall()
    return [(str(symbol), str(timeframe)) for symbol, timeframe in rows]


def main() -> None:
    args = _build_parser().parse_args()
    influx_url, influx_org, influx_bucket, influx_token = _resolve_influx_config(args)

    repo = InfluxMarketDataRepository(
        url=influx_url,
        org=influx_org,
        bucket=influx_bucket,
        token=influx_token,
    )

    conn = sqlite3.connect(str(args.db_path))
    conn.row_factory = sqlite3.Row
    try:
        streams = _fetch_distinct(
            conn,
            exchange=str(args.exchange),
            symbols=list(args.symbols),
            timeframes=list(args.timeframes),
        )
        if not streams:
            print("No matching rows in sqlite market_ohlcv.")
            return

        batch_size = max(1, int(args.batch_size))
        total_rows = 0
        stream_counts: dict[str, int] = {}
        stream_exchange = str(args.exchange).strip().lower()

        for symbol, timeframe in streams:
            cursor = conn.execute(
                """
                SELECT timestamp_ms, open, high, low, close, volume
                FROM market_ohlcv
                WHERE exchange = ? AND symbol = ? AND timeframe = ?
                ORDER BY timestamp_ms
                """,
                (stream_exchange, symbol, timeframe),
            )
            count = 0
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                payload = [
                    (
                        int(row["timestamp_ms"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    )
                    for row in rows
                ]
                count += int(
                    repo.write_ohlcv(
                        exchange=stream_exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        rows=payload,
                    )
                )

            total_rows += count
            stream_counts[f"{symbol}:{timeframe}"] = count
            print(f"[migrated] {symbol} {timeframe}: {count} rows")

        print(f"\nDone. migrated_streams={len(stream_counts)} total_rows={total_rows}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
