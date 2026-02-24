"""Strategy-factory pipeline runner (local-only, parquet+postgres compatible)."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.config import BaseConfig
from lumina_quant.strategy_factory.pipeline import (
    build_shortlist_payload,
    render_shortlist_markdown,
    write_candidate_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strategy-factory shortlist pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Skip external execution.")
    parser.add_argument("--backend", default="parquet-postgres")
    parser.add_argument("--db-path", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--timeframes", nargs="+", default=["1m", "5m", "15m"])
    parser.add_argument("--seeds", nargs="+", default=["20260221"])
    parser.add_argument("--output-dir", default="reports")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path, manifest = write_candidate_manifest(
        output_dir=output_dir,
        timeframes=[str(item) for item in list(args.timeframes)],
        symbols=list(BaseConfig.SYMBOLS),
    )
    print(f"[PIPELINE] candidate manifest: {manifest_path}")

    selected_team = list(manifest.get("candidates") or [])
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "selected_team": selected_team,
        "mode": str(args.mode),
        "backend": str(args.backend),
        "db_path": str(args.db_path),
        "dry_run": bool(args.dry_run),
        "seeds": [str(item) for item in list(args.seeds)],
    }
    report_path = output_dir / f"strategy_factory_report_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    shortlist_payload = build_shortlist_payload(
        report=report,
        mode=str(args.mode),
        shortlist_max_total=10,
        shortlist_max_per_family=3,
        shortlist_max_per_timeframe=4,
        manifest_path=manifest_path,
        research_report_path=report_path,
    )
    shortlist_path = output_dir / f"strategy_factory_shortlist_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    shortlist_path.write_text(json.dumps(shortlist_payload, indent=2), encoding="utf-8")
    print(f"[PIPELINE] shortlist json: {shortlist_path}")

    shortlist_md = output_dir / f"strategy_factory_shortlist_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.md"
    shortlist_md.write_text(render_shortlist_markdown(shortlist_payload), encoding="utf-8")
    print(f"[PIPELINE] shortlist markdown: {shortlist_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
