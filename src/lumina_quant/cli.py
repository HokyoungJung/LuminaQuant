"""CLI for the sanitized public sample pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from lumina_quant.live import paper_summary
from lumina_quant.metrics import compute_metrics
from lumina_quant.pipeline import (
    run_backtest_pipeline,
    run_optimization_pipeline,
    run_paper_live_pipeline,
)
from lumina_quant.rust_kernel import run_rust_backtest


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
    backtest.add_argument("--engine", choices=("python", "rust"), default="python")

    paper_live = subparsers.add_parser("paper-live", help="Run local paper-live replay")
    _add_common_arguments(paper_live)

    optimize = subparsers.add_parser("optimize", help="Run generic sample parameter search")
    optimize.add_argument("--data", default="sample_data/sample_ohlcv.csv")
    optimize.add_argument("--fast-grid", default="2,3,4")
    optimize.add_argument("--slow-grid", default="6,8,10")
    optimize.add_argument("--initial-cash", type=float, default=10_000.0)
    optimize.add_argument("--fee-bps", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "optimize":
        result = run_optimization_pipeline(
            args.data,
            fast_grid=args.fast_grid,
            slow_grid=args.slow_grid,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    common = {
        "fast_window": args.fast_window,
        "slow_window": args.slow_window,
        "initial_cash": args.initial_cash,
        "fee_bps": args.fee_bps,
    }
    if args.command == "backtest":
        if args.engine == "rust":
            print(json.dumps(run_rust_backtest(args.data, **common), indent=2, sort_keys=True))
            return 0
        result = run_backtest_pipeline(args.data, **common)
        payload = result.to_dict()
        payload["metrics"] = compute_metrics(result).to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "paper-live":
        result = run_paper_live_pipeline(args.data, **common)
        print(json.dumps(paper_summary(result), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
