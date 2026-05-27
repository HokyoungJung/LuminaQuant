"""Compare canonical benchmark vs new sleeve(s) using a correlation-aware ensemble."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.portfolio_followup_rules import (
    build_correlation_aware_sparse_fold_ensemble,
    evaluate_weighted_portfolio,
    extract_portfolio_streams,
    extract_split_metrics,
)

FOLLOWUP_ROOT = Path("var/reports/exact_window_backtests/followup_status")
DEFAULT_BENCHMARK_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_incumbent_autoresearch_static_blend_latest.json"
)
DEFAULT_RESEARCH_REPORT_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "last_day_liquidity_regime_followup_current"
    / "research_run"
    / "candidate_research_latest.json"
)
DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "correlation_aware_ensemble_followup_current"
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--research-report", type=Path, default=DEFAULT_RESEARCH_REPORT_PATH)
    parser.add_argument("--extra-research-report", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-members", type=int, default=2)
    parser.add_argument("--correlation-penalty", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=0.80)
    parser.add_argument("--active-fold-ratio-floor", type=float, default=0.50)
    parser.add_argument("--max-pbo", type=float, default=0.75)
    parser.add_argument("--max-turnover", type=float, default=4.0)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _candidate_row_from_payload(
    payload: dict[str, Any], *, name: str, candidate_key: str, label: str
) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    if candidate_key == "canonical_benchmark_static_blend":
        metadata["bypass_preblend_gate"] = True
    return {
        "name": name,
        "candidate_key": candidate_key,
        "label": label,
        "train": dict(extract_split_metrics(payload).get("train") or {}),
        "val": dict(extract_split_metrics(payload).get("val") or {}),
        "oos": dict(extract_split_metrics(payload).get("oos") or {}),
        "return_streams": dict(extract_portfolio_streams(payload)),
        "metadata": metadata,
        "notes": list(payload.get("notes") or []),
    }


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_payload = _load_json(Path(args.benchmark_path).resolve())
    report_paths = [
        Path(args.research_report).resolve(),
        *[Path(path).resolve() for path in list(args.extra_research_report or [])],
    ]
    candidate_rows: list[dict[str, Any]] = []
    for report_path in report_paths:
        research_payload = _load_json(report_path)
        for row in list(research_payload.get("candidates") or []):
            if isinstance(row, dict):
                payload = dict(row)
                payload["source_report_path"] = str(report_path)
                candidate_rows.append(payload)
    if not candidate_rows:
        raise SystemExit("no candidates found in research report")

    benchmark_row = _candidate_row_from_payload(
        benchmark_payload,
        name="canonical_benchmark_static_blend",
        candidate_key="canonical_benchmark_static_blend",
        label="Canonical benchmark static blend",
    )
    benchmark_only = evaluate_weighted_portfolio([{**benchmark_row, "_saved_weight": 1.0}])

    sorted_candidates = sorted(
        candidate_rows,
        key=lambda row: float(
            row.get("selection_score", row.get("oos", {}).get("sharpe", float("-inf")))
        ),
        reverse=True,
    )
    best_new = dict(sorted_candidates[0])
    new_only = evaluate_weighted_portfolio([{**best_new, "_saved_weight": 1.0}])

    combined = build_correlation_aware_sparse_fold_ensemble(
        [benchmark_row, *sorted_candidates],
        max_members=max(1, int(args.max_members)),
        correlation_penalty=float(args.correlation_penalty),
        max_weight=float(args.max_weight),
        active_fold_ratio_floor=float(args.active_fold_ratio_floor),
        max_pbo=float(args.max_pbo),
        max_turnover=float(args.max_turnover),
    )

    payload = {
        "artifact_kind": "correlation_aware_ensemble_followup",
        "generated_at": _utc_now_iso(),
        "benchmark_path": str(Path(args.benchmark_path).resolve()),
        "research_report_paths": [str(path) for path in report_paths],
        "benchmark_only": benchmark_only,
        "best_new_candidate": {
            "name": best_new.get("name"),
            "selection_score": best_new.get("selection_score"),
            "source_report_path": best_new.get("source_report_path"),
            "train": best_new.get("train"),
            "val": best_new.get("val"),
            "oos": best_new.get("oos"),
        },
        "new_method_only": new_only,
        "combined_correlation_aware": combined,
    }

    json_path = output_dir / "correlation_aware_ensemble_followup_latest.json"
    md_path = output_dir / "correlation_aware_ensemble_followup_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _oos_metrics(section: dict[str, Any]) -> dict[str, Any]:
        return dict((section.get("portfolio_metrics") or {}).get("oos") or {})

    benchmark_oos = _oos_metrics(benchmark_only)
    new_oos = _oos_metrics(new_only)
    combined_oos = dict(
        ((combined.get("portfolio_payload") or {}).get("portfolio_metrics") or {}).get("oos") or {}
    )
    md_path.write_text(
        "\n".join(
            [
                "# correlation aware ensemble follow-up",
                "",
                f"- generated_at: `{payload['generated_at']}`",
                f"- benchmark_path: `{payload['benchmark_path']}`",
                *[f"- research_report_path: `{path}`" for path in payload["research_report_paths"]],
                "",
                "## OOS comparison",
                f"- benchmark_only: return `{float(benchmark_oos.get('total_return', benchmark_oos.get('return', 0.0))):.4%}`, sharpe `{float(benchmark_oos.get('sharpe', 0.0)):.4f}`, max_dd `{float(benchmark_oos.get('max_drawdown', 0.0)):.4%}`",
                f"- new_method_only: return `{float(new_oos.get('total_return', new_oos.get('return', 0.0))):.4%}`, sharpe `{float(new_oos.get('sharpe', 0.0)):.4f}`, max_dd `{float(new_oos.get('max_drawdown', 0.0)):.4%}`",
                f"- combined_correlation_aware: return `{float(combined_oos.get('total_return', combined_oos.get('return', 0.0))):.4%}`, sharpe `{float(combined_oos.get('sharpe', 0.0)):.4f}`, max_dd `{float(combined_oos.get('max_drawdown', 0.0)):.4%}`",
                "",
                "## Selected components",
                *[
                    f"- {item['name']}: weight `{float(item['weight']):.4f}`, avg_abs_oos_corr `{float(item['avg_abs_oos_corr']):.4f}`, base_score `{float(item['base_score']):.4f}`, adjusted_score `{float(item['adjusted_score']):.4f}`"
                    for item in list(combined.get("components") or [])
                ],
                "",
                "## Excluded candidates",
                *[
                    f"- {item['name']}: reason `{item['reason']}`"
                    for item in list(combined.get("excluded_candidates") or [])
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
