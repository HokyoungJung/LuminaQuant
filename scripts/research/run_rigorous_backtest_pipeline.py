#!/usr/bin/env python3
"""Run the canonical strategy-agnostic rigorous backtest pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lumina_quant.backtesting.rigorous_pipeline import STAGE_ORDER, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    receipt = run_pipeline(
        args.plan.resolve(),
        repository=repository,
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {"status": receipt["status"], "stages": STAGE_ORDER},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
