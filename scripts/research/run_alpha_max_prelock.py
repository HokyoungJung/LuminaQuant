#!/usr/bin/env python3
"""Physically isolated alpha-max prelock command boundary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from lumina_quant.research.alpha_max_engine_runner import (
    reject_ambient_lq_environment,
    run_alpha_max_prelock_process,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed alpha-max prelock process.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract-manifest", required=True)
    parser.add_argument("--exchange", required=True, choices=("binance",))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--warmup-raw-root", required=True)
    parser.add_argument("--warmup-feature-root", required=True)
    parser.add_argument("--train-raw-root", required=True)
    parser.add_argument("--train-feature-root", required=True)
    parser.add_argument("--purge-raw-root", required=True)
    parser.add_argument("--purge-feature-root", required=True)
    parser.add_argument("--validation-raw-root", required=True)
    parser.add_argument("--validation-feature-root", required=True)
    parser.add_argument("--embargo-raw-root", required=True)
    parser.add_argument("--embargo-feature-root", required=True)
    return parser


def _execute(args: argparse.Namespace) -> int:
    result = run_alpha_max_prelock_process(
        config=args.config,
        contract_manifest=args.contract_manifest,
        exchange=args.exchange,
        output_root=args.output_root,
        warmup_raw_root=args.warmup_raw_root,
        warmup_feature_root=args.warmup_feature_root,
        train_raw_root=args.train_raw_root,
        train_feature_root=args.train_feature_root,
        purge_raw_root=args.purge_raw_root,
        purge_feature_root=args.purge_feature_root,
        validation_raw_root=args.validation_raw_root,
        validation_feature_root=args.validation_feature_root,
        embargo_raw_root=args.embargo_raw_root,
        embargo_feature_root=args.embargo_feature_root,
    )
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    # This gate precedes parser construction/argument-file access and every
    # user-controlled filesystem operation.
    reject_ambient_lq_environment()
    args = build_parser().parse_args(argv)
    return _execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
