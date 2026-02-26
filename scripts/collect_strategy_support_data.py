"""Plan or execute strategy-support data collection.

Defaults to plan-only mode so no data is fetched unless --execute is provided.
"""

from __future__ import annotations

import argparse
import json

from lumina_quant.config import BaseConfig
from lumina_quant.data_collector import collect_strategy_support_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan/execute futures support-data collection for strategies."
    )
    parser.add_argument("--symbols", nargs="+", default=list(BaseConfig.SYMBOLS))
    parser.add_argument("--db-path", default=BaseConfig.MARKET_DATA_PARQUET_PATH)
    parser.add_argument("--exchange-id", default=BaseConfig.MARKET_DATA_EXCHANGE)
    parser.add_argument("--since", default="2021-01-01T00:00:00+00:00")
    parser.add_argument("--until", default="")
    parser.add_argument("--mark-index-interval", default="1m")
    parser.add_argument("--open-interest-period", default="5m")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backend", default="parquet-postgres", help="Storage backend.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually fetch and upsert support data. Default is plan-only with no fetch.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = collect_strategy_support_data(
        db_path=str(args.db_path),
        exchange_id=str(args.exchange_id),
        symbol_list=list(args.symbols),
        since=args.since,
        until=(args.until or None),
        mark_index_interval=str(args.mark_index_interval),
        open_interest_period=str(args.open_interest_period),
        retries=max(0, int(args.retries)),
        execute=bool(args.execute),
        backend=str(args.backend),
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
