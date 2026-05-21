"""CLI for the sanitized public sample pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from lumina_quant.config import (
    BacktestConfig,
    OptimizationConfig,
    PublicPipelineConfig,
    default_public_pipeline_config,
    load_public_pipeline_config,
)
from lumina_quant.live import paper_summary
from lumina_quant.metrics import compute_metrics
from lumina_quant.pipeline import (
    run_backtest_pipeline,
    run_optimization_pipeline,
    run_paper_live_pipeline,
)
from lumina_quant.rust_kernel import run_rust_backtest


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help="Optional public sample TOML config path",
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        default=None,
        help="Local OHLCV CSV path",
    )
    parser.add_argument("--fast-window", type=int, default=None)
    parser.add_argument("--slow-window", type=int, default=None)
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--fee-bps", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lq-public")
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest = subparsers.add_parser("backtest", help="Run local sample-data backtest")
    _add_config_argument(backtest)
    _add_common_arguments(backtest)
    backtest.add_argument("--engine", choices=("python", "rust"), default=None)

    paper_live = subparsers.add_parser("paper-live", help="Run local paper-live replay")
    _add_config_argument(paper_live)
    _add_common_arguments(paper_live)

    optimize = subparsers.add_parser("optimize", help="Run generic sample parameter search")
    _add_config_argument(optimize)
    optimize.add_argument("--data", default=None)
    optimize.add_argument("--method", choices=("grid", "optuna"), default=None)
    optimize.add_argument("--fast-grid", default=None)
    optimize.add_argument("--slow-grid", default=None)
    optimize.add_argument("--initial-cash", type=float, default=None)
    optimize.add_argument("--fee-bps", type=float, default=None)
    optimize.add_argument("--n-trials", type=int, default=None)
    optimize.add_argument("--sampler-seed", type=int, default=None)
    return parser


def _load_cli_config(path: str | None) -> PublicPipelineConfig:
    if path is None:
        return default_public_pipeline_config()
    return load_public_pipeline_config(path)


def _resolve_backtest_args(
    args: argparse.Namespace,
    base: BacktestConfig,
) -> tuple[str, dict[str, float | int]]:
    data_path = args.data if args.data is not None else base.data_path
    return data_path, {
        "fast_window": args.fast_window if args.fast_window is not None else base.fast_window,
        "slow_window": args.slow_window if args.slow_window is not None else base.slow_window,
        "initial_cash": args.initial_cash if args.initial_cash is not None else base.initial_cash,
        "fee_bps": args.fee_bps if args.fee_bps is not None else base.fee_bps,
    }


def _resolve_optimization_args(
    args: argparse.Namespace,
    base: OptimizationConfig,
) -> tuple[str, dict[str, float | int | str]]:
    data_path = args.data if args.data is not None else base.data_path
    return data_path, {
        "method": args.method if args.method is not None else base.method,
        "fast_grid": args.fast_grid if args.fast_grid is not None else base.fast_grid,
        "slow_grid": args.slow_grid if args.slow_grid is not None else base.slow_grid,
        "initial_cash": args.initial_cash if args.initial_cash is not None else base.initial_cash,
        "fee_bps": args.fee_bps if args.fee_bps is not None else base.fee_bps,
        "n_trials": args.n_trials if args.n_trials is not None else base.n_trials,
        "sampler_seed": args.sampler_seed if args.sampler_seed is not None else base.sampler_seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_cli_config(args.config)
    if args.command == "optimize":
        data_path, optimize_args = _resolve_optimization_args(args, config.optimization)
        result = run_optimization_pipeline(data_path, **optimize_args)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "backtest":
        data_path, common = _resolve_backtest_args(args, config.backtest)
        engine = args.engine if args.engine is not None else config.backtest.engine
        if engine == "rust":
            print(json.dumps(run_rust_backtest(data_path, **common), indent=2, sort_keys=True))
            return 0
        result = run_backtest_pipeline(data_path, **common)
        payload = result.to_dict()
        payload["metrics"] = compute_metrics(result).to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "paper-live":
        data_path, common = _resolve_backtest_args(args, config.paper_live)
        result = run_paper_live_pipeline(data_path, **common)
        print(json.dumps(paper_summary(result), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
