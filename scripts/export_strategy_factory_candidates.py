"""Export a large strategy candidate universe for research/tuning runs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.strategies.factory_candidate_set import (
    DEFAULT_TIMEFRAMES,
    DEFAULT_TOP10_PLUS_METALS,
    build_candidate_set,
    summarize_candidate_set,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export strategy-factory candidate set to JSON.")
    parser.add_argument("--output", default="reports/strategy_factory_candidates.json")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_TOP10_PLUS_METALS))
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES))
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    candidates = build_candidate_set(
        symbols=list(args.symbols),
        timeframes=list(args.timeframes),
        max_candidates=max(0, int(args.max_candidates)),
    )
    summary = summarize_candidate_set(candidates)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(candidates),
        "summary": summary,
        "candidates": candidates,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if bool(args.pretty) else None
    output.write_text(json.dumps(payload, indent=indent), encoding="utf-8")

    print(f"Saved candidate universe: {output}")
    print(f"Total candidates: {len(candidates)}")
    print(f"Families: {summary.get('families')}")


if __name__ == "__main__":
    main()
