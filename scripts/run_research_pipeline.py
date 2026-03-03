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
from lumina_quant.strategy_factory.research_runner import run_candidate_research


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strategy-factory shortlist pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Skip external execution.")
    parser.add_argument("--backend", default="parquet-postgres")
    parser.add_argument("--db-path", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--timeframes", nargs="+", default=list(BaseConfig.TIMEFRAMES))
    parser.add_argument("--seeds", nargs="+", default=["20260221"])
    parser.add_argument("--single-min-score", type=float, default=0.0)
    parser.add_argument("--single-min-return", type=float, default=0.0)
    parser.add_argument("--single-min-sharpe", type=float, default=0.7)
    parser.add_argument("--single-min-trades", type=int, default=20)
    parser.add_argument("--allow-multi-asset", action="store_true")
    parser.add_argument("--drop-single-without-metrics", action="store_true")
    parser.add_argument("--disable-weights", action="store_true")
    parser.add_argument("--weight-temperature", type=float, default=0.35)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--set-max-per-asset", type=int, default=2)
    parser.add_argument("--set-max-sets", type=int, default=16)
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
    research_report = run_candidate_research(
        candidates=selected_team,
        base_timeframe="1s",
        strategy_timeframes=[str(item) for item in list(args.timeframes)],
        symbol_universe=list(BaseConfig.SYMBOLS),
        stage1_keep_ratio=0.5,
        max_candidates=max(1, len(selected_team)),
    )
    report["selected_team"] = list(research_report.get("candidates") or [])
    report["split"] = research_report.get("split")
    report["base_timeframe"] = research_report.get("base_timeframe")
    report["strategy_timeframes"] = research_report.get("strategy_timeframes")
    report["stage1"] = research_report.get("stage1")
    report["data_sources"] = research_report.get("data_sources")

    report_path = output_dir / f"strategy_factory_report_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    shortlist_payload = build_shortlist_payload(
        report=report,
        mode=str(args.mode),
        shortlist_max_total=10,
        shortlist_max_per_family=3,
        shortlist_max_per_timeframe=4,
        single_min_score=float(args.single_min_score),
        single_min_return=float(args.single_min_return),
        single_min_sharpe=float(args.single_min_sharpe),
        single_min_trades=int(args.single_min_trades),
        allow_multi_asset=bool(args.allow_multi_asset),
        drop_single_without_metrics=bool(args.drop_single_without_metrics),
        include_weights=not bool(args.disable_weights),
        weight_temperature=float(args.weight_temperature),
        max_weight=float(args.max_weight),
        set_max_per_asset=max(1, int(args.set_max_per_asset)),
        set_max_sets=max(1, int(args.set_max_sets)),
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
