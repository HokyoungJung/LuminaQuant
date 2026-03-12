"""Build canonical JSON/CSV inventory for futures support features."""

from __future__ import annotations

import argparse
import json

from lumina_quant.config import BaseConfig
from lumina_quant.data.support_inventory import (
    build_strategy_support_inventory,
    write_strategy_support_inventory,
)
from lumina_quant.symbols import canonical_symbol


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write canonical strategy support inventory reports.")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--db-path", default=getattr(BaseConfig, "MARKET_DATA_PARQUET_PATH", "data/market_parquet"))
    parser.add_argument("--exchange-id", default=getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance"))
    parser.add_argument(
        "--json-path",
        default="var/reports/strategy_support_inventory_latest.json",
        help="JSON inventory output path.",
    )
    parser.add_argument(
        "--csv-path",
        default="var/reports/strategy_support_inventory_latest.csv",
        help="CSV inventory output path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    symbols = [canonical_symbol(symbol) for symbol in list(args.symbols or []) if str(symbol).strip()]
    payload = build_strategy_support_inventory(
        db_path=str(args.db_path),
        exchange=str(args.exchange_id),
        symbols=symbols or None,
    )
    outputs = write_strategy_support_inventory(
        payload=payload,
        json_path=str(args.json_path),
        csv_path=str(args.csv_path),
    )
    print(json.dumps({"outputs": outputs, "symbol_count": payload["symbol_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
