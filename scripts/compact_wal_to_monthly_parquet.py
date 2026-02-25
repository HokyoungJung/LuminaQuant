"""Compact custom WAL files into monthly parquet (atomic replace)."""

from __future__ import annotations

import argparse
from pathlib import Path

from lumina_quant.config import BaseConfig
from lumina_quant.parquet_market_data import ParquetMarketDataRepository, normalize_symbol


def _discover_symbols(root: Path, exchange: str) -> list[str]:
    base = root / "market_ohlcv_1s" / str(exchange).strip().lower()
    if not base.exists():
        return []
    symbols: list[str] = []
    for item in sorted(base.iterdir()):
        if item.is_dir():
            symbols.append(normalize_symbol(item.name))
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact WAL records into bounded monthly parquet files per symbol."
    )
    parser.add_argument(
        "--root-path",
        default=str(BaseConfig.MARKET_DATA_PARQUET_PATH),
        help="Market data root path (default: config market parquet path).",
    )
    parser.add_argument(
        "--exchange",
        default=str(BaseConfig.MARKET_DATA_EXCHANGE),
        help="Exchange key (default: config market exchange).",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols (e.g. BTC/USDT,ETH/USDT). If empty, auto-discover.",
    )
    parser.add_argument(
        "--keep-wal",
        action="store_true",
        help="Do not truncate wal.bin after successful compaction (advances watermark only).",
    )
    args = parser.parse_args()

    root = Path(args.root_path).expanduser()
    repo = ParquetMarketDataRepository(root)
    exchange = str(args.exchange).strip().lower()

    if str(args.symbols).strip():
        symbols = [normalize_symbol(item.strip()) for item in str(args.symbols).split(",") if item.strip()]
    else:
        symbols = _discover_symbols(root, exchange)

    if not symbols:
        print("No symbols found for compaction.")
        return

    total_rows_before = 0
    total_rows_after = 0
    total_months = 0

    for symbol in symbols:
        results = repo.compact_all(
            exchange=exchange,
            symbol=symbol,
            timeframe="1s",
            remove_sources=(not bool(args.keep_wal)),
        )
        if not results:
            print(f"[{symbol}] no WAL records to compact")
            continue

        month_count = 0
        for result in results:
            month_count += 1
            total_months += 1
            total_rows_before += int(result.rows_before)
            total_rows_after += int(result.rows_after)
            print(
                f"[{symbol}] {Path(result.partition).name}: "
                f"rows_before={result.rows_before} rows_after={result.rows_after}"
            )
        print(f"[{symbol}] compacted months={month_count}")

    print(
        "Compaction complete: "
        f"symbols={len(symbols)} months={total_months} "
        f"rows_before={total_rows_before} rows_after={total_rows_after}"
    )


if __name__ == "__main__":
    main()
