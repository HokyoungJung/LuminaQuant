"""Run advanced candidate research and emit shortlist-compatible reports."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.config import BaseConfig
from lumina_quant.strategy_factory.research_runner import (
    build_default_candidate_rows,
    run_candidate_research,
)
from lumina_quant.strategy_factory.selection import select_diversified_shortlist
from lumina_quant.symbols import CANONICAL_STRATEGY_TIMEFRAMES, canonicalize_symbol_list


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run advanced candidate research pipeline.")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--manifest", default="", help="Optional candidate manifest JSON path.")
    parser.add_argument("--symbols", nargs="+", default=list(BaseConfig.SYMBOLS))
    parser.add_argument("--timeframes", nargs="+", default=list(CANONICAL_STRATEGY_TIMEFRAMES))
    parser.add_argument("--base-timeframe", default="1s")
    parser.add_argument("--stage1-keep-ratio", type=float, default=0.35)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--score-config", default="", help="Optional scoring config JSON path.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_manifest_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("candidates") or [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _load_score_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"score config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid score config JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"score config must be a JSON object: {path}")
    return dict(payload)


def _shortlist_robust_score_params(score_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(score_config, dict):
        return None
    rank_weights = score_config.get("candidate_rank_score_weights")
    if not isinstance(rank_weights, dict):
        return None
    reject = score_config.get("reject_thresholds")
    reject_thresholds = reject if isinstance(reject, dict) else {}
    params: dict[str, Any] = {}
    mapping = {
        "sharpe_weight": "sharpe_weight",
        "deflated_sharpe_weight": "deflated_sharpe_weight",
        "pbo_penalty": "pbo_penalty",
        "return_weight": "return_weight",
        "drawdown_penalty": "drawdown_penalty",
        "turnover_penalty": "turnover_penalty",
        "turnover_threshold": "turnover_threshold",
    }
    for src, dst in mapping.items():
        if src in rank_weights:
            params[dst] = rank_weights[src]
    if "turnover_threshold" not in params and "max_turnover" in reject_thresholds:
        params["turnover_threshold"] = reject_thresholds["max_turnover"]
    return params or None


def _write_summary_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "name",
                "strategy_class",
                "family",
                "strategy_timeframe",
                "selection_score",
                "pass",
                "hard_reject",
                "oos_return",
                "oos_sharpe",
                "oos_deflated_sharpe",
                "oos_pbo",
                "oos_turnover",
                "oos_mdd",
                "trade_count",
                "cross_candidate_corr",
            ],
        )
        writer.writeheader()
        for row in candidates:
            oos = dict(row.get("oos") or {})
            writer.writerow(
                {
                    "candidate_id": row.get("candidate_id"),
                    "name": row.get("name"),
                    "strategy_class": row.get("strategy_class"),
                    "family": row.get("family"),
                    "strategy_timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                    "selection_score": float(row.get("selection_score", 0.0)),
                    "pass": bool(row.get("pass", False)),
                    "hard_reject": bool(row.get("hard_reject", False)),
                    "oos_return": float(oos.get("return", 0.0)),
                    "oos_sharpe": float(oos.get("sharpe", 0.0)),
                    "oos_deflated_sharpe": float(oos.get("deflated_sharpe", 0.0)),
                    "oos_pbo": float(oos.get("pbo", 1.0)),
                    "oos_turnover": float(oos.get("turnover", 0.0)),
                    "oos_mdd": float(oos.get("mdd", 0.0)),
                    "trade_count": float(oos.get("trade_count", 0.0)),
                    "cross_candidate_corr": float(oos.get("cross_candidate_corr", 0.0)),
                }
            )


def _render_shortlist_markdown(
    *,
    report_path: Path,
    shortlist: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Candidate Research Shortlist",
        "",
        f"- Source report: `{report_path}`",
        f"- Candidate count: {len(shortlist)}",
        "",
        "| # | Name | Strategy | TF | Family | OOS Sharpe | DSR | PBO | Score |",
        "|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(shortlist, start=1):
        oos = dict(row.get("oos") or {})
        lines.append(
            "| "
            f"{idx} | "
            f"{row.get('name', '')} | "
            f"{row.get('strategy_class', '')} | "
            f"{row.get('strategy_timeframe') or row.get('timeframe') or ''} | "
            f"{row.get('family', '')} | "
            f"{float(oos.get('sharpe', 0.0)):.3f} | "
            f"{float(oos.get('deflated_sharpe', 0.0)):.3f} | "
            f"{float(oos.get('pbo', 1.0)):.3f} | "
            f"{float(row.get('selection_score', 0.0)):.3f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_config: dict[str, Any] | None = None
    if str(args.score_config).strip():
        try:
            score_config = _load_score_config(Path(str(args.score_config)).resolve())
        except ValueError as exc:
            raise SystemExit(f"[RESEARCH] {exc}")

    symbols = canonicalize_symbol_list(list(args.symbols))
    timeframes = [str(token).strip().lower() for token in list(args.timeframes) if str(token).strip()]

    manifest_path = Path(str(args.manifest)).resolve() if str(args.manifest).strip() else None
    if manifest_path and manifest_path.exists():
        candidates = _load_manifest_candidates(manifest_path)
    else:
        candidates = build_default_candidate_rows(
            symbols=symbols,
            timeframes=timeframes,
            max_candidates=max(1, int(args.max_candidates)),
        )

    report = run_candidate_research(
        candidates=candidates,
        base_timeframe=str(args.base_timeframe),
        strategy_timeframes=timeframes,
        symbol_universe=symbols,
        stage1_keep_ratio=float(args.stage1_keep_ratio),
        max_candidates=max(1, int(args.max_candidates)),
        score_config=score_config,
    )

    shortlist_score_params = _shortlist_robust_score_params(score_config)
    shortlisted = select_diversified_shortlist(
        report.get("candidates") or [],
        mode="oos",
        max_total=max(1, int(args.top_k)),
        max_per_family=max(2, int(args.top_k // 2)),
        max_per_timeframe=max(2, int(args.top_k // 2)),
        single_min_score=0.0,
        single_min_return=0.0,
        single_min_sharpe=0.0,
        single_min_trades=5,
        allow_multi_asset=True,
        include_weights=True,
        robust_score_params=shortlist_score_params,
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    output_path = output_dir / f"candidate_research_{stamp}.json"
    latest_path = output_dir / "candidate_research_latest.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output_dir / f"candidate_research_{stamp}.csv"
    csv_latest = output_dir / "candidate_research_latest.csv"
    _write_summary_csv(csv_path, list(report.get("candidates") or []))
    _write_summary_csv(csv_latest, list(report.get("candidates") or []))

    # Team-report compatibility with existing shortlist pipeline.
    team_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": "v2",
        "base_timeframe": report.get("base_timeframe"),
        "strategy_timeframes": report.get("strategy_timeframes"),
        "split": report.get("split"),
        "source_report": str(output_path),
        "selected_team": shortlisted,
        "candidates": report.get("candidates") or [],
        "stage1": report.get("stage1") or {},
        "scoring_config": report.get("scoring_config") or {},
        "data_sources": report.get("data_sources") or {},
    }
    team_report_path = output_dir / f"strategy_factory_report_{stamp}.json"
    team_report_latest = output_dir / "strategy_factory_report_latest.json"
    team_report_path.write_text(json.dumps(team_report, indent=2), encoding="utf-8")
    team_report_latest.write_text(json.dumps(team_report, indent=2), encoding="utf-8")

    shortlist_md = output_dir / f"strategy_factory_shortlist_{stamp}.md"
    _render_shortlist_markdown(
        report_path=output_path,
        shortlist=shortlisted,
        output_path=shortlist_md,
    )

    print(f"[RESEARCH] candidates_in={len(candidates)}")
    print(f"[RESEARCH] candidates_stage2={len(list(report.get('candidates') or []))}")
    print(f"[RESEARCH] shortlisted={len(shortlisted)}")
    print(f"Saved: {output_path}")
    print(f"Saved latest: {latest_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved team report: {team_report_path}")
    print(f"Saved shortlist markdown: {shortlist_md}")
    if args.dry_run:
        print("[RESEARCH] dry-run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
