from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lumina_quant.strategy_factory.candidate_library import DEFAULT_BINANCE_TOP10_PLUS_METALS  # noqa: E402
from lumina_quant.workflows.autonomous_portfolio_research_loop import (  # noqa: E402
    DEFAULT_BACKLOG_TIMEFRAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_ROOT,
    run_autonomous_portfolio_research_loop,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the autonomous portfolio research loop artifact index and audit outputs."
    )
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_BINANCE_TOP10_PLUS_METALS))
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_BACKLOG_TIMEFRAMES))
    parser.add_argument("--max-archive-crashes", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_autonomous_portfolio_research_loop(
        report_root=Path(args.report_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        symbols=[str(item) for item in list(args.symbols or [])],
        timeframes=[str(item) for item in list(args.timeframes or [])],
        max_archive_crashes=max(0, int(args.max_archive_crashes)),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
