#!/usr/bin/env python3
"""Full-tuning/upper-bound diagnostics over existing monthly walk-forward rows.

This is intentionally NOT a clean deployment selector. It answers the research
question: if we allow full hindsight tuning/oracle selection over already-built
candidate rows, what performance ceiling is visible under different risk and
candidate-scope constraints?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_monthly_refit_walkforward as wf  # noqa: E402

DEFAULT_INPUT_JSON = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_69_asset_mdd30_high_vol_20260602/"
    "mdd30_high_vol_walkforward_clean_recomputed_latest.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_69_asset_mdd30_high_vol_20260602/"
    "full_tuning_upper_bound_latest.json"
)
DEFAULT_OUTPUT_MD = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_69_asset_mdd30_high_vol_20260602/"
    "full_tuning_upper_bound_latest.md"
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    return wf._safe_float(value, default)


def _period_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    folds = list(payload.get("folds") or [])
    if not folds:
        return {}
    return {
        "first_oos_start": folds[0]["locked_oos"]["start"],
        "last_oos_end": folds[-1]["locked_oos"]["end"],
        "fold_ids": [fold["fold_id"] for fold in folds],
        "timeframes": list(payload.get("timeframes") or []),
        "slippage_bps": payload.get("cost_model", {}).get("slippage_bps"),
        "protocol": payload.get("protocol"),
    }


def _aggregate_synthetic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arr = np.asarray([_safe_float(row["locked_oos"]["total_return"]) for row in rows], dtype=float)
    oos_mdds = [_safe_float(row["locked_oos"]["mdd"]) for row in rows]
    val_returns = [_safe_float(row["validation"]["total_return"]) for row in rows]
    train_returns = [_safe_float(row["train"]["total_return"]) for row in rows]
    compounded = float(np.prod(1.0 + arr) - 1.0) if arr.size else 0.0
    fold_count = int(arr.size)
    annualized = (
        float((1.0 + compounded) ** (12.0 / fold_count) - 1.0)
        if fold_count and compounded > -1
        else 0.0
    )
    pos = arr[arr > 0.0]
    neg = arr[arr < 0.0]
    monthly_std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    downside_std = float(np.std(neg, ddof=1)) if neg.size > 1 else 0.0
    monthly_equity = np.cumprod(1.0 + arr) if arr.size else np.asarray([])
    monthly_equity_mdd = 0.0
    if monthly_equity.size:
        running_peak = np.maximum.accumulate(monthly_equity)
        monthly_equity_mdd = float(np.max(1.0 - monthly_equity / np.maximum(running_peak, 1e-12)))
    var_05 = float(np.quantile(arr, 0.05)) if arr.size else 0.0
    q95 = float(np.quantile(arr, 0.95)) if arr.size else 0.0
    cvar_25 = float(np.mean(arr[arr <= np.quantile(arr, 0.25)])) if arr.size else 0.0
    loss_streak = 0
    max_loss_streak = 0
    for value in arr:
        if value < 0.0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0
    return {
        "fold_count": fold_count,
        "compounded_oos_return": compounded,
        "annualized_oos_return_approx": annualized,
        "mean_oos_return": float(np.mean(arr)) if arr.size else 0.0,
        "median_oos_return": float(np.median(arr)) if arr.size else 0.0,
        "min_oos_return": float(np.min(arr)) if arr.size else 0.0,
        "latest_oos_return": float(arr[-1]) if arr.size else 0.0,
        "positive_oos_folds": int(np.sum(arr > 0.0)),
        "oos_hit_rate": float(np.mean(arr > 0.0)) if arr.size else 0.0,
        "max_oos_mdd": max(oos_mdds) if oos_mdds else 0.0,
        "monthly_equity_mdd": monthly_equity_mdd,
        "monthly_volatility": monthly_std,
        "monthly_downside_volatility": downside_std,
        "monthly_sharpe_approx": float(np.mean(arr) / monthly_std * math.sqrt(12.0))
        if monthly_std > 0
        else 0.0,
        "monthly_sortino_approx": float(np.mean(arr) / downside_std * math.sqrt(12.0))
        if downside_std > 0
        else 0.0,
        "profit_factor": float(np.sum(pos) / abs(np.sum(neg)))
        if pos.size and neg.size and abs(float(np.sum(neg))) > 0
        else 0.0,
        "omega_0": float(np.sum(np.maximum(arr, 0.0)) / np.sum(np.maximum(-arr, 0.0)))
        if arr.size and np.sum(np.maximum(-arr, 0.0)) > 0
        else 0.0,
        "monthly_var_05": var_05,
        "monthly_quantile_95": q95,
        "monthly_cvar_25": cvar_25,
        "tail_ratio_95_05": float(q95 / abs(var_05)) if abs(var_05) > 0.0 else 0.0,
        "avg_gain": float(np.mean(pos)) if pos.size else 0.0,
        "avg_loss": float(np.mean(neg)) if neg.size else 0.0,
        "max_loss_streak": max_loss_streak,
        "mean_validation_return": float(np.mean(val_returns)) if val_returns else 0.0,
        "mean_train_return": float(np.mean(train_returns)) if train_returns else 0.0,
    }


def _select_by_oracle(
    rows: Sequence[Mapping[str, Any]],
    *,
    filter_fn: Callable[[Mapping[str, Any]], bool],
    objective: Callable[[Mapping[str, Any]], tuple[float, ...]],
) -> list[dict[str, Any]]:
    by_fold: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if filter_fn(row):
            by_fold[str(row["fold_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for fold_id in sorted(by_fold):
        row = max(by_fold[fold_id], key=objective)
        selected.append(dict(row))
    return selected


def _scenario_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    description: str,
    filter_fn: Callable[[Mapping[str, Any]], bool],
    objective: Callable[[Mapping[str, Any]], tuple[float, ...]],
) -> dict[str, Any]:
    selected = _select_by_oracle(rows, filter_fn=filter_fn, objective=objective)
    agg = _aggregate_synthetic(selected)
    return {
        "scenario": name,
        "description": description,
        "deployable": False,
        "uses_locked_oos_oracle": True,
        "aggregate": agg,
        "selected_by_fold": [
            {
                "fold_id": row["fold_id"],
                "candidate_label": row["candidate_label"],
                "family": row["family"],
                "clean_promotion_eligible": row.get("clean_promotion_eligible"),
                "post_oos_research_variant": row.get("post_oos_research_variant"),
                "validation_return": row["validation"]["total_return"],
                "validation_mdd": row["validation"]["mdd"],
                "locked_oos_return": row["locked_oos"]["total_return"],
                "locked_oos_mdd": row["locked_oos"]["mdd"],
                "final_weights": row.get("final_weights") or {},
            }
            for row in selected
        ],
    }


def _build_scenarios(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = wf._sanitize_research_dependency_flags(list(payload.get("fold_candidate_rows") or []))

    def all_rows(row: Mapping[str, Any]) -> bool:
        return True

    def clean_rows(row: Mapping[str, Any]) -> bool:
        return bool(row.get("clean_promotion_eligible"))

    def mdd30_rows(row: Mapping[str, Any]) -> bool:
        return _safe_float(row.get("locked_oos", {}).get("mdd")) <= 0.30

    def mdd20_rows(row: Mapping[str, Any]) -> bool:
        return _safe_float(row.get("locked_oos", {}).get("mdd")) <= 0.20

    def clean_mdd20(row: Mapping[str, Any]) -> bool:
        return clean_rows(row) and mdd20_rows(row)

    def clean_mdd30(row: Mapping[str, Any]) -> bool:
        return clean_rows(row) and mdd30_rows(row)

    def objective_return(row: Mapping[str, Any]) -> tuple[float, ...]:
        return (
            _safe_float(row["locked_oos"]["total_return"]),
            -_safe_float(row["locked_oos"]["mdd"]),
        )

    def objective_calmar(row: Mapping[str, Any]) -> tuple[float, ...]:
        ret = _safe_float(row["locked_oos"]["total_return"])
        mdd = _safe_float(row["locked_oos"]["mdd"])
        return (ret / max(mdd, 0.02), ret, -mdd)

    return [
        _scenario_rows(
            rows,
            name="oracle_all_candidates_max_return",
            description="Best monthly locked-OOS return among all candidates, including post-OOS research rows.",
            filter_fn=all_rows,
            objective=objective_return,
        ),
        _scenario_rows(
            rows,
            name="oracle_all_candidates_mdd30_max_return",
            description="Best monthly locked-OOS return among all candidates with that fold OOS MDD <=30%.",
            filter_fn=mdd30_rows,
            objective=objective_return,
        ),
        _scenario_rows(
            rows,
            name="oracle_all_candidates_mdd20_max_return",
            description="Best monthly locked-OOS return among all candidates with that fold OOS MDD <=20%.",
            filter_fn=mdd20_rows,
            objective=objective_return,
        ),
        _scenario_rows(
            rows,
            name="oracle_clean_only_max_return",
            description="Best monthly locked-OOS return among clean candidates only. Still not deployable because OOS chooses the rule.",
            filter_fn=clean_rows,
            objective=objective_return,
        ),
        _scenario_rows(
            rows,
            name="oracle_clean_only_mdd30_max_return",
            description="Best monthly locked-OOS return among clean candidates with fold OOS MDD <=30%.",
            filter_fn=clean_mdd30,
            objective=objective_return,
        ),
        _scenario_rows(
            rows,
            name="oracle_clean_only_mdd20_max_return",
            description="Best monthly locked-OOS return among clean candidates with fold OOS MDD <=20%.",
            filter_fn=clean_mdd20,
            objective=objective_return,
        ),
        _scenario_rows(
            rows,
            name="oracle_clean_only_mdd20_calmar",
            description="Best monthly OOS Calmar among clean candidates with fold OOS MDD <=20%.",
            filter_fn=clean_mdd20,
            objective=objective_calmar,
        ),
    ]


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = _build_scenarios(payload)
    scenarios_by_comp = sorted(
        scenarios,
        key=lambda item: _safe_float(item["aggregate"]["compounded_oos_return"]),
        reverse=True,
    )
    baseline_clean = sorted(
        wf._aggregate_rows(
            [
                row
                for row in wf._sanitize_research_dependency_flags(
                    list(payload.get("fold_candidate_rows") or [])
                )
                if row.get("clean_promotion_eligible")
            ]
        ),
        key=lambda row: _safe_float(row["compounded_oos_return"]),
        reverse=True,
    )[:12]
    return {
        "artifact_kind": "alpha_zoo_69_asset_full_tuning_upper_bound",
        "generated_at_utc": _utc_now_iso(),
        "source_json": payload.get("output_paths", {}).get("json"),
        "period": _period_from_payload(payload),
        "method_note": (
            "full-tuning upper bound only: scenarios use locked-OOS oracle selection and are not deployable. "
            "Use this as a ceiling/diagnostic, not as clean live evidence."
        ),
        "baseline_clean_rankings_by_comp": baseline_clean,
        "scenarios_by_comp": scenarios_by_comp,
    }


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):.2%}"


def render_markdown(report: Mapping[str, Any]) -> str:
    scenarios = list(report.get("scenarios_by_comp") or [])
    baseline = list(report.get("baseline_clean_rankings_by_comp") or [])[:8]
    lines = [
        "# Full-tuning upper-bound diagnostics",
        "",
        f"- generated: `{report.get('generated_at_utc')}`",
        f"- OOS period: `{report.get('period', {}).get('first_oos_start')}` → `{report.get('period', {}).get('last_oos_end')}`",
        f"- note: {report.get('method_note')}",
        "",
        "## Scenario ranking",
        "",
        "| Rank | Scenario | Deployable | OOS oracle | OOS comp | Ann. approx | Hit | Max OOS MDD | Min OOS | Latest | Sharpe | Sortino | PF |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, scenario in enumerate(scenarios, start=1):
        agg = scenario["aggregate"]
        lines.append(
            f"| {idx} | `{scenario['scenario']}` | `{scenario['deployable']}` | `{scenario['uses_locked_oos_oracle']}` | "
            f"{_fmt_pct(agg['compounded_oos_return'])} | "
            f"{_fmt_pct(agg['annualized_oos_return_approx'])} | "
            f"{agg['positive_oos_folds']}/{agg['fold_count']} | "
            f"{_fmt_pct(agg['max_oos_mdd'])} | "
            f"{_fmt_pct(agg['min_oos_return'])} | "
            f"{_fmt_pct(agg['latest_oos_return'])} | "
            f"{_safe_float(agg['monthly_sharpe_approx']):.2f} | "
            f"{_safe_float(agg['monthly_sortino_approx']):.2f} | "
            f"{_safe_float(agg['profit_factor']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Clean baseline by comp",
            "",
            "| Rank | Candidate | Family | OOS comp | Hit | Max OOS MDD | Sharpe | Sortino | PF | Clean |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx, row in enumerate(baseline, start=1):
        lines.append(
            f"| {idx} | `{row['candidate_label']}` | `{row['family']}` | "
            f"{_fmt_pct(row['compounded_oos_return'])} | "
            f"{row['positive_oos_folds']}/{row['fold_count']} | "
            f"{_fmt_pct(row['max_oos_mdd'])} | "
            f"{_safe_float(row['monthly_sharpe_approx']):.2f} | "
            f"{_safe_float(row['monthly_sortino_approx']):.2f} | "
            f"{_safe_float(row['profit_factor']):.2f} | "
            f"`{bool(row.get('clean_promotion_eligible'))}` |"
        )
    for scenario in scenarios[:3]:
        lines.extend(
            [
                "",
                f"## Monthly choices: `{scenario['scenario']}`",
                "",
                "| Fold | Candidate | Family | Clean | Research | Val | Val MDD | OOS | OOS MDD |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in scenario["selected_by_fold"]:
            lines.append(
                f"| `{row['fold_id']}` | `{row['candidate_label']}` | `{row['family']}` | "
                f"`{bool(row.get('clean_promotion_eligible'))}` | "
                f"`{bool(row.get('post_oos_research_variant'))}` | "
                f"{_fmt_pct(row['validation_return'])} | "
                f"{_fmt_pct(row['validation_mdd'])} | "
                f"{_fmt_pct(row['locked_oos_return'])} | "
                f"{_fmt_pct(row['locked_oos_mdd'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The scenario rows are maximum-performance diagnostics because the selected candidate is chosen by locked-OOS performance.",
            "- `oracle_clean_only_*` shows that even if the source candidates are clean, choosing among them with OOS is still not clean for deployment.",
            "- The deployable reference remains the best pre-registered clean candidate unless a new selection rule is frozen before fresh-forward validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(Path(args.input_json).read_text("utf-8"))
    report = run(payload)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(wf._json_safe(report), indent=2, sort_keys=True) + "\n", "utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), "utf-8")
    print(
        json.dumps(
            {
                "output_json": str(out_json),
                "output_md": str(out_md),
                "scenarios_by_comp": report["scenarios_by_comp"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
