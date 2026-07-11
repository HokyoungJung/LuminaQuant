#!/usr/bin/env python3
"""One-touch append-only alpha-max exposed historical evaluation boundary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from lumina_quant.alpha_max_process_boundary import reject_ambient_lq_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the report-only alpha-max exposed historical evaluation.",
        allow_abbrev=False,
    )
    parser.add_argument("--sealed-prelock-directory", required=True)
    parser.add_argument("--embargo-feature-root", required=True)
    parser.add_argument("--historical-evaluation-raw-root", required=True)
    parser.add_argument("--historical-evaluation-feature-root", required=True)
    parser.add_argument("--exchange", required=True, choices=("binance",))
    parser.add_argument("--output-root", required=True)
    return parser


def _execute(args: argparse.Namespace) -> int:
    from lumina_quant.research.alpha_max_engine_runner import run_alpha_max_historical_process

    result = run_alpha_max_historical_process(
        sealed_prelock_directory=args.sealed_prelock_directory,
        embargo_feature_root=args.embargo_feature_root,
        historical_evaluation_raw_root=args.historical_evaluation_raw_root,
        historical_evaluation_feature_root=args.historical_evaluation_feature_root,
        exchange=args.exchange,
        output_root=args.output_root,
    )
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    reject_ambient_lq_environment()
    args = build_parser().parse_args(argv)
    return _execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
