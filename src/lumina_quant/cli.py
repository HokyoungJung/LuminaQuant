"""CLI for the sanitized public sample pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from lumina_quant.live import paper_summary
from lumina_quant.pipeline import run_backtest_pipeline, run_paper_live_pipeline


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        default="sample_data/sample_ohlcv.csv",
        help="Local OHLCV CSV path",
    )
    parser.add_argument("--fast-window", type=int, default=3)
    parser.add_argument("--slow-window", type=int, default=8)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lq-public")
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest = subparsers.add_parser("backtest", help="Run local sample-data backtest")
    _add_common_arguments(backtest)
    paper_live = subparsers.add_parser("paper-live", help="Run local paper-live replay")
    _add_common_arguments(paper_live)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "fast_window": args.fast_window,
        "slow_window": args.slow_window,
        "initial_cash": args.initial_cash,
        "fee_bps": args.fee_bps,
    }
    if args.command == "backtest":
        result = run_backtest_pipeline(args.data, **common)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "paper-live":
        result = run_paper_live_pipeline(args.data, **common)
        print(json.dumps(paper_summary(result), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
