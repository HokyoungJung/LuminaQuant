"""Build dashboard-ready summary artifacts from live research reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.strategy_factory.selection import (
    select_diversified_shortlist,
    strategy_family,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_mode(report: dict[str, Any]) -> str:
    mode = str(report.get("mode") or "").strip().lower()
    if mode:
        return mode
    split = report.get("split")
    if isinstance(split, dict):
        token = str(split.get("mode") or "").strip().lower()
        if token:
            return token
    return "live"


def _latest_report_path(report_glob: str) -> Path:
    files = sorted(Path().glob(str(report_glob)))
    if not files:
        raise FileNotFoundError(f"No files matched report glob: {report_glob}")
    return files[-1]


def _candidate_metrics(candidate: dict[str, Any], mode: str) -> dict[str, float]:
    metric_key = "val" if str(mode).strip().lower() == "live" else "oos"
    metrics = candidate.get(metric_key) if isinstance(candidate.get(metric_key), dict) else {}
    if not metrics:
        metrics = candidate.get("oos") if isinstance(candidate.get("oos"), dict) else {}
    if not metrics:
        metrics = candidate.get("val") if isinstance(candidate.get("val"), dict) else {}
    return {
        "return": _safe_float(metrics.get("return"), 0.0),
        "sharpe": _safe_float(metrics.get("sharpe"), 0.0),
        "sortino": _safe_float(metrics.get("sortino"), 0.0),
        "mdd": _safe_float(metrics.get("mdd"), 0.0),
        "trades": int(_safe_float(metrics.get("trades"), 0.0)),
    }


def _candidate_hurdle(candidate: dict[str, Any], mode: str) -> dict[str, Any]:
    hurdle_key = "val" if str(mode).strip().lower() == "live" else "oos"
    hurdle = (candidate.get("hurdle_fields") or {}).get(hurdle_key)
    if not isinstance(hurdle, dict):
        hurdle = {}
    return {
        "pass": bool(hurdle.get("pass", False)),
        "score": _safe_float(hurdle.get("score"), 0.0),
        "excess_return": _safe_float(hurdle.get("excess_return"), 0.0),
        "target_return": _safe_float(hurdle.get("target_return"), 0.0),
        "realized_return": _safe_float(hurdle.get("realized_return"), 0.0),
    }


def _selected_rows(candidates: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        timeframe = str(
            candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""
        ).lower()
        symbols = [str(token) for token in list(candidate.get("symbols") or [])]
        metrics = _candidate_metrics(candidate, mode)
        hurdle = _candidate_hurdle(candidate, mode)
        family = str(candidate.get("family") or strategy_family(str(candidate.get("name", ""))))

        weight = candidate.get("weight")
        if weight is None:
            weight = candidate.get("recommended_weight")
        if weight is None:
            weight = candidate.get("portfolio_weight")

        row = {
            "rank": idx,
            "name": str(candidate.get("name", "")),
            "family": family,
            "strategy_timeframe": timeframe,
            "symbols": symbols,
            "symbol_count": len(symbols),
            "metrics": metrics,
            "hurdle": hurdle,
            "selection_score": _safe_float(
                candidate.get("shortlist_score", candidate.get("selection_score", hurdle["score"])),
                hurdle["score"],
            ),
            "source_report": str(candidate.get("report_path", candidate.get("source_report", ""))),
            "identity": str(candidate.get("identity", "")),
            "explicit_weight": None if weight is None else _safe_float(weight, 0.0),
        }
        rows.append(row)
    return rows


def _recommended_weights(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"method": "none", "weights": []}

    explicit = [row for row in rows if row.get("explicit_weight") is not None]
    if explicit and len(explicit) == len(rows):
        total = sum(max(0.0, _safe_float(row.get("explicit_weight"), 0.0)) for row in explicit)
        if total > 0:
            weights = [
                {
                    "name": row["name"],
                    "strategy_timeframe": row["strategy_timeframe"],
                    "weight": _safe_float(row["explicit_weight"], 0.0) / total,
                }
                for row in rows
            ]
            return {"method": "normalized_explicit", "weights": weights}

    weight = 1.0 / float(len(rows))
    weights = [
        {"name": row["name"], "strategy_timeframe": row["strategy_timeframe"], "weight": weight}
        for row in rows
    ]
    return {"method": "equal_weight_fallback", "weights": weights}


def _headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "selected_count": 0,
            "hurdle_pass_count": 0,
            "hurdle_pass_rate": 0.0,
            "avg_return": 0.0,
            "avg_sharpe": 0.0,
            "avg_sortino": 0.0,
            "avg_mdd": 0.0,
            "avg_trades": 0.0,
            "best_candidate": None,
        }

    pass_count = sum(1 for row in rows if bool((row.get("hurdle") or {}).get("pass", False)))
    returns = [row["metrics"]["return"] for row in rows]
    sharpes = [row["metrics"]["sharpe"] for row in rows]
    sortinos = [row["metrics"]["sortino"] for row in rows]
    mdds = [row["metrics"]["mdd"] for row in rows]
    trades = [row["metrics"]["trades"] for row in rows]
    best = max(
        rows,
        key=lambda row: (
            bool((row.get("hurdle") or {}).get("pass", False)),
            _safe_float((row.get("hurdle") or {}).get("score"), -1e9),
            _safe_float((row.get("metrics") or {}).get("return"), -1e9),
        ),
    )

    return {
        "selected_count": len(rows),
        "hurdle_pass_count": pass_count,
        "hurdle_pass_rate": float(pass_count) / float(len(rows)),
        "avg_return": sum(returns) / float(len(returns)),
        "avg_sharpe": sum(sharpes) / float(len(sharpes)),
        "avg_sortino": sum(sortinos) / float(len(sortinos)),
        "avg_mdd": sum(mdds) / float(len(mdds)),
        "avg_trades": sum(trades) / float(len(trades)),
        "best_candidate": {
            "name": best.get("name"),
            "strategy_timeframe": best.get("strategy_timeframe"),
            "return": best["metrics"]["return"],
            "sharpe": best["metrics"]["sharpe"],
            "sortino": best["metrics"]["sortino"],
            "mdd": best["metrics"]["mdd"],
            "trades": best["metrics"]["trades"],
            "hurdle_pass": bool((best.get("hurdle") or {}).get("pass", False)),
            "hurdle_score": _safe_float((best.get("hurdle") or {}).get("score"), 0.0),
        },
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    meta = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
    headline = payload.get("headline_metrics") if isinstance(payload.get("headline_metrics"), dict) else {}
    rows = list(payload.get("selected_team_table") or [])
    weights = payload.get("recommended_weights") if isinstance(payload.get("recommended_weights"), dict) else {}
    failures = list(payload.get("failed_runs") or [])

    lines = [
        "# Live Portfolio Dashboard Summary",
        "",
        f"- Generated: {payload.get('generated_at', '')}",
        f"- Source report: `{payload.get('source_report', '')}`",
        f"- Mode: `{meta.get('mode', '')}`",
        f"- Backend: `{meta.get('backend', '')}`",
        f"- Exchange / Market: `{meta.get('exchange', '')}` / `{meta.get('market_type', '')}`",
        f"- Base TF: `{meta.get('base_timeframe', '')}`",
        f"- Timeframes: `{', '.join(meta.get('timeframes') or [])}`",
        f"- Seeds: `{', '.join(str(x) for x in (meta.get('seeds') or []))}`",
        "",
        "## Headline Metrics",
        "",
        f"- Selected sleeves: **{headline.get('selected_count', 0)}**",
        f"- Hurdle pass: **{headline.get('hurdle_pass_count', 0)} / {headline.get('selected_count', 0)}**",
        f"- Avg return: **{_safe_float(headline.get('avg_return'), 0.0):.4f}**",
        f"- Avg sharpe: **{_safe_float(headline.get('avg_sharpe'), 0.0):.4f}**",
        f"- Avg sortino: **{_safe_float(headline.get('avg_sortino'), 0.0):.4f}**",
        f"- Avg mdd: **{_safe_float(headline.get('avg_mdd'), 0.0):.4f}**",
        f"- Avg trades: **{_safe_float(headline.get('avg_trades'), 0.0):.2f}**",
        "",
        "## Selected Team Table",
        "",
        "| # | Name | TF | Family | Return | Sharpe | Sortino | MDD | Trades | Hurdle Pass | Weight |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|:---:|---:|",
    ]

    if rows:
        weight_map = {
            (str(row.get("name", "")), str(row.get("strategy_timeframe", ""))): _safe_float(
                row.get("weight"), 0.0
            )
            for row in list(weights.get("weights") or [])
            if isinstance(row, dict)
        }
        for row in rows:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            hurdle = row.get("hurdle") if isinstance(row.get("hurdle"), dict) else {}
            key = (str(row.get("name", "")), str(row.get("strategy_timeframe", "")))
            lines.append(
                "| "
                f"{int(row.get('rank', 0))} | "
                f"{row.get('name', '')} | "
                f"{row.get('strategy_timeframe', '')} | "
                f"{row.get('family', '')} | "
                f"{_safe_float(metrics.get('return'), 0.0):.4f} | "
                f"{_safe_float(metrics.get('sharpe'), 0.0):.4f} | "
                f"{_safe_float(metrics.get('sortino'), 0.0):.4f} | "
                f"{_safe_float(metrics.get('mdd'), 0.0):.4f} | "
                f"{int(_safe_float(metrics.get('trades'), 0.0))} | "
                f"{'✅' if bool(hurdle.get('pass', False)) else '❌'} | "
                f"{_safe_float(weight_map.get(key), 0.0):.4f} |"
            )
    else:
        lines.append("| - | (no selected sleeves) | - | - | - | - | - | - | - | - | - |")

    lines.extend(["", "## Portfolio Weights", ""])
    lines.append(f"- Method: `{weights.get('method', 'none')}`")
    for row in list(weights.get("weights") or []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('name', '')} [{row.get('strategy_timeframe', '')}]: "
            f"{_safe_float(row.get('weight'), 0.0):.4f}"
        )

    if failures:
        lines.extend(
            [
                "",
                "## Failed Runs",
                "",
                "| Timeframe | Seed | Status | Reason |",
                "|---|---:|---|---|",
            ]
        )
        for row in failures:
            lines.append(
                "| "
                f"{row.get('timeframe', '')} | "
                f"{row.get('seed', '')} | "
                f"{row.get('status', '')} | "
                f"{row.get('reason', '')} |"
            )

    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build dashboard-ready live portfolio summary.")
    parser.add_argument("--report", default="", help="Optional explicit report JSON path.")
    parser.add_argument(
        "--report-glob",
        default="reports/strategy_team_research_live_*.json",
        help="Report glob used when --report is omitted.",
    )
    parser.add_argument("--out-dir", default="reports/dashboard")
    parser.add_argument("--prefix", default="live_portfolio_dashboard_summary")
    parser.add_argument("--max-selected", type=int, default=24)
    parser.add_argument("--max-per-family", type=int, default=8)
    parser.add_argument("--max-per-timeframe", type=int, default=6)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report_path = Path(args.report).expanduser() if str(args.report).strip() else _latest_report_path(
        str(args.report_glob)
    )
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    with report_path.open(encoding="utf-8") as fp:
        report = json.load(fp)

    mode = _resolve_mode(report)
    selected_team = list(report.get("selected_team") or [])
    if not selected_team:
        candidates = list(report.get("candidates") or [])
        if candidates:
            selected_team = select_diversified_shortlist(
                candidates,
                mode=mode,
                max_total=max(1, int(args.max_selected)),
                max_per_family=max(1, int(args.max_per_family)),
                max_per_timeframe=max(1, int(args.max_per_timeframe)),
            )

    selected_rows = _selected_rows(selected_team, mode)
    weights = _recommended_weights(selected_rows)
    for row, weight_row in zip(selected_rows, list(weights.get("weights") or []), strict=False):
        if isinstance(weight_row, dict):
            row["recommended_weight"] = _safe_float(weight_row.get("weight"), 0.0)

    families = Counter(str(row.get("family", "other")) for row in selected_rows)
    timeframes = Counter(str(row.get("strategy_timeframe", "")) for row in selected_rows)

    failed_runs = []
    for run in list(report.get("run_rows") or []):
        if not isinstance(run, dict):
            continue
        status = str(run.get("status", "")).strip().lower()
        if status in {"ok", "dry_run"}:
            continue
        failed_runs.append(
            {
                "timeframe": str(run.get("timeframe", "")),
                "seed": run.get("seed"),
                "status": str(run.get("status", "")),
                "reason": str(run.get("reason", "")),
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_report": str(report_path),
        "run_metadata": {
            "mode": mode,
            "generated_at": report.get("generated_at"),
            "exchange": report.get("exchange"),
            "market_type": report.get("market_type"),
            "backend": report.get("backend"),
            "base_timeframe": report.get("base_timeframe"),
            "base_timeframes": list(report.get("base_timeframes") or []),
            "timeframes": list(report.get("timeframes") or []),
            "seeds": list(report.get("seeds") or []),
            "strategy_set": report.get("strategy_set"),
            "topcap_symbols": list(report.get("topcap_symbols") or []),
            "candidate_manifest": report.get("candidate_manifest"),
            "all_candidates_count": int(_safe_float(report.get("all_candidates_count"), 0.0)),
            "selected_team_count": int(_safe_float(report.get("selected_team_count"), len(selected_rows))),
        },
        "selected_team_table": selected_rows,
        "selected_summary": {
            "family": dict(sorted(families.items(), key=lambda item: item[0])),
            "timeframe": dict(sorted(timeframes.items(), key=lambda item: item[0])),
        },
        "headline_metrics": _headline(selected_rows),
        "recommended_weights": weights,
        "failed_runs": failed_runs,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"{str(args.prefix).strip()}_{stamp}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")

    print(f"Source report: {report_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")
    print(
        "Headline: "
        f"selected={payload['headline_metrics']['selected_count']} "
        f"pass_rate={payload['headline_metrics']['hurdle_pass_rate']:.4f} "
        f"avg_return={payload['headline_metrics']['avg_return']:.6f}"
    )


if __name__ == "__main__":
    main()
