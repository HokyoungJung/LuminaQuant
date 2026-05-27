"""Build an OOM-conscious article-inspired candidate manifest for sequential research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumina_quant.strategy_factory import (
    DEFAULT_BINANCE_TOP10_PLUS_METALS,
    build_article_pipeline_manifest,
)

DEFAULT_OUTPUT = Path(
    "var/reports/exact_window_backtests/followup_status/"
    "portfolio_incumbent_autoresearch_grouped/article_inspired_research_current"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5m", "15m", "30m", "1h", "4h"],
        help="Candidate timeframes to include. Defaults avoid 1s/1d to stay lightweight.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_BINANCE_TOP10_PLUS_METALS),
        help="Symbol universe for the manifest.",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=0,
        help="Optional cap per strategy family (0 = no cap).",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Optional total candidate cap (0 = no cap).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_article_pipeline_manifest(
        timeframes=args.timeframes,
        symbols=args.symbols,
        max_per_family=max(0, int(args.max_per_family)),
        max_total=max(0, int(args.max_total)),
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "article_pipeline_candidate_manifest_latest.json"
    md_path = output_dir / "article_pipeline_candidate_manifest_latest.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# article pipeline candidate manifest",
        "",
        f"- candidate_count: `{payload['candidate_count']}`",
        f"- timeframes: `{', '.join(payload['timeframes'])}`",
        f"- max_per_family: `{payload['max_per_family']}`",
        f"- max_total: `{payload['max_total']}`",
        "",
        "## families",
        *[f"- {family}: {count}" for family, count in sorted(payload["family_counts"].items())],
        "",
        "## article families",
        *[
            f"- {family}: {count}"
            for family, count in sorted(payload["article_family_counts"].items())
        ],
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
