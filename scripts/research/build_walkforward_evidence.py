#!/usr/bin/env python3
"""Freeze validation finalists or attach locked-OOS report-only evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lumina_quant.backtesting.walkforward_evidence import (
    build_report_only_evaluation,
    select_finalists,
)
from lumina_quant.research.run_card import atomic_write_text, stable_json_dumps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    selection = subparsers.add_parser("select", allow_abbrev=False)
    selection.add_argument("--walkforward", required=True, type=Path)
    selection.add_argument("--output", required=True, type=Path)
    selection.add_argument("--top-n", type=int, default=20)
    selection.add_argument("--minimum-pass-ratio", type=float, default=0.75)
    selection.add_argument("--minimum-mean-sharpe", type=float, default=0.35)
    report = subparsers.add_parser("report", allow_abbrev=False)
    report.add_argument("--walkforward", required=True, type=Path)
    report.add_argument("--selection", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "select":
        result = select_finalists(
            args.walkforward.resolve(),
            top_n=args.top_n,
            minimum_pass_ratio=args.minimum_pass_ratio,
            minimum_mean_sharpe=args.minimum_mean_sharpe,
        )
    else:
        result = build_report_only_evaluation(
            args.walkforward.resolve(),
            args.selection.resolve(),
        )
    atomic_write_text(args.output.resolve(), stable_json_dumps(result) + "\n")
    print(json.dumps({"status": result["status"], "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
