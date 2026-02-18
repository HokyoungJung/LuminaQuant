"""Migrate legacy 1s rows from market_ohlcv into compact 1s DB."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumina_quant.market_data import (
    connect_market_data_db,
    ensure_market_ohlcv_schema,
    upsert_ohlcv_rows_1s,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate timeframe=1s rows to compact DB.")
    parser.add_argument("--db-path", default="data/lumina_quant.db")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Delete migrated 1s rows from legacy market_ohlcv table.",
    )
    args = parser.parse_args()

    db_path = str(args.db_path)
    if not Path(db_path).exists():
        raise FileNotFoundError(db_path)

    conn = connect_market_data_db(db_path)
    try:
        ensure_market_ohlcv_schema(conn)
        cur = conn.execute(
            """
            SELECT exchange, symbol, timestamp_ms, open, high, low, close, volume
            FROM market_ohlcv
            WHERE timeframe = '1s'
            ORDER BY symbol, timestamp_ms
            """
        )
        migrated = 0
        batch = []
        current_exchange = None
        current_symbol = None
        for row in cur:
            exchange = str(row[0]).strip().lower()
            symbol = str(row[1]).strip().upper()
            ohlcv = (
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
            )
            if current_exchange is None:
                current_exchange = exchange
                current_symbol = symbol
            if (
                exchange != current_exchange
                or symbol != current_symbol
                or len(batch) >= int(args.batch_size)
            ):
                if batch:
                    upsert_ohlcv_rows_1s(
                        db_path,
                        exchange=current_exchange,
                        symbol=current_symbol,
                        rows=batch,
                    )
                    migrated += len(batch)
                    batch = []
                current_exchange = exchange
                current_symbol = symbol
            batch.append(ohlcv)

        if batch and current_exchange and current_symbol:
            upsert_ohlcv_rows_1s(
                db_path,
                exchange=current_exchange,
                symbol=current_symbol,
                rows=batch,
            )
            migrated += len(batch)

        print(f"Migrated rows: {migrated}")

        if bool(args.delete_legacy):
            deleted = conn.execute("DELETE FROM market_ohlcv WHERE timeframe = '1s'").rowcount
            conn.commit()
            conn.execute("VACUUM")
            print(f"Deleted legacy rows: {deleted}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
