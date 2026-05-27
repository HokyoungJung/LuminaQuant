"""Build a sparse-fold-aware ensemble comparison from saved candidate artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lumina_quant.portfolio_followup_rules import build_sparse_fold_aware_ensemble

ROOT = Path(
    "var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped"
)
DEFAULT_OUTPUT_DIR = ROOT / "sparse_fold_ensemble_followup_current"
MIDBRIDGE_REPORT = (
    ROOT
    / "pair_spread_robustness_midbridge25_followup_current/research_run/candidate_research_latest.json"
)
ADAPTIVE_REPORT = (
    ROOT / "pair_spread_adaptive_rls_followup_current/research_run/candidate_research_latest.json"
)
BROADER_REPORT = (
    ROOT / "volatility_regime_residual_followup_current/research_run/candidate_research_latest.json"
)
CURRENT_BEST_NAME = "pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.5_0.65"


def _load_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)]


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_rows = _load_candidates(MIDBRIDGE_REPORT)
    adaptive_rows = _load_candidates(ADAPTIVE_REPORT)
    broader_rows = _load_candidates(BROADER_REPORT)

    current_best = next((row for row in current_rows if row.get("name") == CURRENT_BEST_NAME), None)
    if current_best is None:
        raise SystemExit(f"missing current best row: {CURRENT_BEST_NAME}")

    adaptive_best = max(
        adaptive_rows, key=lambda row: float((row.get("oos") or {}).get("sharpe", float("-inf")))
    )
    broader_best = max(
        broader_rows, key=lambda row: float((row.get("oos") or {}).get("sharpe", float("-inf")))
    )

    incumbent_only = build_sparse_fold_aware_ensemble([current_best], max_members=1)
    new_methods_only = build_sparse_fold_aware_ensemble(
        [adaptive_best, broader_best], max_members=2
    )
    combined = build_sparse_fold_aware_ensemble(
        [current_best, adaptive_best, broader_best], max_members=3
    )

    payload = {
        "artifact_kind": "sparse_fold_ensemble_followup",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "current_best_component": current_best.get("name"),
        "adaptive_component": adaptive_best.get("name"),
        "broader_component": broader_best.get("name"),
        "incumbent_only": incumbent_only,
        "new_methods_only": new_methods_only,
        "combined_sparse_fold_ensemble": combined,
    }

    json_path = output_dir / "sparse_fold_ensemble_followup_latest.json"
    md_path = output_dir / "sparse_fold_ensemble_followup_latest.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _metrics(section: dict) -> dict:
        return dict(
            ((section.get("portfolio_payload") or {}).get("portfolio_metrics") or {}).get("oos")
            or {}
        )

    inc_metrics = _metrics(incumbent_only)
    new_metrics = _metrics(new_methods_only)
    comb_metrics = _metrics(combined)
    md_path.write_text(
        "\n".join(
            [
                "# sparse fold aware ensemble follow-up",
                "",
                f"- current_best_component: `{current_best.get('name')}`",
                f"- adaptive_component: `{adaptive_best.get('name')}`",
                f"- broader_component: `{broader_best.get('name')}`",
                "",
                "## OOS comparison",
                f"- incumbent_only: return `{float(inc_metrics.get('total_return', 0.0)):.4%}`, sharpe `{float(inc_metrics.get('sharpe', 0.0)):.4f}`",
                f"- new_methods_only: return `{float(new_metrics.get('total_return', 0.0)):.4%}`, sharpe `{float(new_metrics.get('sharpe', 0.0)):.4f}`",
                f"- combined_sparse_fold_ensemble: return `{float(comb_metrics.get('total_return', 0.0)):.4%}`, sharpe `{float(comb_metrics.get('sharpe', 0.0)):.4f}`",
                "",
                "## Combined ensemble weights",
                *[
                    f"- {item['name']}: weight `{float(item['weight']):.4f}`, oos_pbo `{float(item['oos_pbo']):.3f}`, active_fold_ratio `{float(item['oos_active_fold_ratio']):.3f}`"
                    for item in list(combined.get("components") or [])
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
