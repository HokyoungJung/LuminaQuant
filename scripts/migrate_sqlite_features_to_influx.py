"""Migrate SQLite futures_feature_points rows into InfluxDB measurement."""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Sequence

from lumina_quant.influx_market_data import InfluxMarketDataRepository
from lumina_quant.market_data import normalize_symbol


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate SQLite futures_feature_points to InfluxDB."
    )
    parser.add_argument("--db-path", default="data/lq_market.sqlite3")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", nargs="+", default=[])
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


def _fetch_symbols(
    conn: sqlite3.Connection,
    *,
    exchange: str,
    symbols: Sequence[str],
) -> list[str]:
    stream_exchange = str(exchange).strip().lower()
    clauses = ["exchange = ?"]
    params: list[object] = [stream_exchange]

    normalized_symbols = [normalize_symbol(sym) for sym in symbols if str(sym).strip()]
    if normalized_symbols:
        placeholders = ",".join("?" for _ in normalized_symbols)
        clauses.append(f"symbol IN ({placeholders})")
        params.extend(normalized_symbols)

    where_clause = " AND ".join(clauses)
    query = (
        "SELECT DISTINCT symbol FROM futures_feature_points "
        f"WHERE {where_clause} ORDER BY symbol"
    )
    try:
        rows = conn.execute(query, tuple(params)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [str(symbol) for (symbol,) in rows]


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
        symbols = _fetch_symbols(
            conn,
            exchange=str(args.exchange),
            symbols=list(args.symbols),
        )
        if not symbols:
            print("No matching rows in sqlite futures_feature_points.")
            return

        batch_size = max(1, int(args.batch_size))
        stream_exchange = str(args.exchange).strip().lower()
        total_rows = 0

        for symbol in symbols:
            cursor = conn.execute(
                """
                SELECT timestamp_ms, funding_rate, funding_mark_price, mark_price, index_price,
                       open_interest, liquidation_long_qty, liquidation_short_qty,
                       liquidation_long_notional, liquidation_short_notional, source
                FROM futures_feature_points
                WHERE exchange = ? AND symbol = ?
                ORDER BY timestamp_ms
                """,
                (stream_exchange, symbol),
            )
            count = 0
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                payload = []
                source_value = "binance_futures_api"
                for row in rows:
                    source_value = str(row["source"] or source_value)
                    payload.append(
                        {
                            "timestamp_ms": int(row["timestamp_ms"]),
                            "funding_rate": row["funding_rate"],
                            "funding_mark_price": row["funding_mark_price"],
                            "mark_price": row["mark_price"],
                            "index_price": row["index_price"],
                            "open_interest": row["open_interest"],
                            "liquidation_long_qty": row["liquidation_long_qty"],
                            "liquidation_short_qty": row["liquidation_short_qty"],
                            "liquidation_long_notional": row["liquidation_long_notional"],
                            "liquidation_short_notional": row["liquidation_short_notional"],
                        }
                    )
                count += int(
                    repo.write_futures_feature_points(
                        exchange=stream_exchange,
                        symbol=symbol,
                        rows=payload,
                        source=source_value,
                    )
                )
            total_rows += count
            print(f"[migrated] {symbol}: {count} rows")

        print(f"\nDone. migrated_symbols={len(symbols)} total_rows={total_rows}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
