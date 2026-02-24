"""Compact partitioned parquet market-data files."""

from __future__ import annotations

import argparse
from datetime import date

from lumina_quant.parquet_market_data import ParquetMarketDataRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact parquet market-data partitions.")
    parser.add_argument("--root-path", default="data/market_parquet")
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="1s")
    parser.add_argument(
        "--date",
        default="",
        help="Optional partition date (YYYY-MM-DD). If omitted, compact all partitions.",
    )
    parser.add_argument(
        "--keep-sources",
        action="store_true",
        help="Keep source parquet files after writing compact output.",
    )
    return parser.parse_args()


def _print_result(result) -> None:
    print(
        f"Compacted {result.partition}: files {result.files_before}->{result.files_after}, "
        f"rows {result.rows_before}->{result.rows_after}."
    )


def main() -> int:
    args = parse_args()
    repo = ParquetMarketDataRepository(args.root_path)
    remove_sources = not bool(args.keep_sources)

    if str(args.date).strip():
        # Validate upfront to provide a clear CLI error.
        resolved_date = date.fromisoformat(str(args.date).strip())
        result = repo.compact_partition(
            exchange=str(args.exchange),
            symbol=str(args.symbol),
            timeframe=str(args.timeframe),
            partition_date=resolved_date,
            remove_sources=remove_sources,
        )
        _print_result(result)
        return 0

    results = repo.compact_all(
        exchange=str(args.exchange),
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        remove_sources=remove_sources,
    )
    if not results:
        print("No partitions found.")
        return 0

    files_before = 0
    files_after = 0
    rows_before = 0
    rows_after = 0
    for item in results:
        _print_result(item)
        files_before += item.files_before
        files_after += item.files_after
        rows_before += item.rows_before
        rows_after += item.rows_after

    print(
        f"Summary: partitions={len(results)} files={files_before}->{files_after} "
        f"rows={rows_before}->{rows_after}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
