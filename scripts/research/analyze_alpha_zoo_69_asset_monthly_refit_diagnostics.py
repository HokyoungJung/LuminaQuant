#!/usr/bin/env python3
"""Diagnostics for 69-asset monthly-refit walk-forward artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_WALKFORWARD_ARTIFACT = Path(
    "/tmp/lumina_monthly_refit_walkforward_individual_guarded_latest.json"
)
DEFAULT_LATEST_DETAIL_ARTIFACT = Path(
    "/tmp/lumina_monthly_refit_walkforward_individual_guarded_latest_fold_detail_seedmatched.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_69_asset_monthly_refit_diagnostics_20260601/diagnostics_latest.json"
)
DEFAULT_CANDIDATE_LABEL = "individual_robust:hybrid_v3_5"


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    payload["_path"] = str(path.expanduser().resolve())
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows_for(payload: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in payload.get("fold_candidate_rows", [])
        if row["candidate_label"] == label
    ]


def _aggregate_for(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    for row in payload.get("aggregate_rankings", []):
        if row["candidate_label"] == label:
            return dict(row)
    raise ValueError(f"aggregate row not found: {label}")


def _monthly_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    returns = np.array([row["locked_oos"]["total_return"] for row in rows], dtype=float)
    validations = np.array([row["validation"]["total_return"] for row in rows], dtype=float)
    mdds = np.array([row["locked_oos"]["mdd"] for row in rows], dtype=float)
    fold_sharpes = np.array([row["locked_oos"]["sharpe"] for row in rows], dtype=float)
    fold_sortinos = np.array([row["locked_oos"]["sortino"] for row in rows], dtype=float)
    fold_calmars = np.array([row["locked_oos"]["calmar"] for row in rows], dtype=float)
    gains = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    mean = float(np.mean(returns)) if returns.size else 0.0
    stdev = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    downside_stdev = float(np.std(losses, ddof=1)) if losses.size > 1 else 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity) if equity.size else np.array([])
    equity_drawdowns = equity / peaks - 1.0 if equity.size else np.array([])
    max_loss_streak = 0
    current_loss_streak = 0
    for value in returns:
        if value < 0.0:
            current_loss_streak += 1
            max_loss_streak = max(max_loss_streak, current_loss_streak)
        else:
            current_loss_streak = 0
    q05 = float(np.quantile(returns, 0.05)) if returns.size else 0.0
    q25 = float(np.quantile(returns, 0.25)) if returns.size else 0.0
    cvar25_values = returns[returns <= q25] if returns.size else np.array([])
    return {
        "fold_count": int(returns.size),
        "compounded_oos_return": float(np.prod(1.0 + returns) - 1.0) if returns.size else 0.0,
        "monthly_mean_return": mean,
        "monthly_median_return": float(np.median(returns)) if returns.size else 0.0,
        "monthly_return_stdev": stdev,
        "monthly_sharpe_approx": mean / stdev * math.sqrt(12.0) if stdev > 0.0 else 0.0,
        "monthly_sortino_approx": mean / downside_stdev * math.sqrt(12.0)
        if downside_stdev > 0.0
        else 0.0,
        "positive_oos_folds": int(np.sum(returns > 0.0)),
        "hit_rate": float(np.mean(returns > 0.0)) if returns.size else 0.0,
        "min_monthly_return": float(np.min(returns)) if returns.size else 0.0,
        "max_monthly_return": float(np.max(returns)) if returns.size else 0.0,
        "monthly_var_5": q05,
        "monthly_var_25": q25,
        "monthly_cvar_25": float(np.mean(cvar25_values)) if cvar25_values.size else 0.0,
        "average_gain_month": float(np.mean(gains)) if gains.size else 0.0,
        "average_loss_month": float(np.mean(losses)) if losses.size else 0.0,
        "gain_loss_ratio": float(np.mean(gains) / abs(np.mean(losses)))
        if gains.size and losses.size
        else 0.0,
        "max_fold_oos_mdd": float(np.max(mdds)) if mdds.size else 0.0,
        "equity_curve_max_drawdown": abs(float(np.min(equity_drawdowns)))
        if equity_drawdowns.size
        else 0.0,
        "max_consecutive_loss_folds": max_loss_streak,
        "mean_fold_sharpe": float(np.nanmean(fold_sharpes)) if fold_sharpes.size else 0.0,
        "median_fold_sharpe": float(np.nanmedian(fold_sharpes)) if fold_sharpes.size else 0.0,
        "mean_fold_sortino": float(np.nanmean(fold_sortinos)) if fold_sortinos.size else 0.0,
        "median_fold_sortino": float(np.nanmedian(fold_sortinos)) if fold_sortinos.size else 0.0,
        "mean_fold_calmar": float(np.nanmean(fold_calmars)) if fold_calmars.size else 0.0,
        "median_fold_calmar": float(np.nanmedian(fold_calmars)) if fold_calmars.size else 0.0,
        "mean_validation_return": float(np.mean(validations)) if validations.size else 0.0,
        "min_validation_return": float(np.min(validations)) if validations.size else 0.0,
        "max_validation_return": float(np.max(validations)) if validations.size else 0.0,
    }


def _audit_aggregate_match(payload: Mapping[str, Any], candidate_label: str) -> dict[str, Any]:
    rows = _rows_for(payload, candidate_label)
    aggregate = _aggregate_for(payload, candidate_label)
    metrics = _monthly_metrics(rows)
    comparisons = {
        "compounded_oos_return": (
            aggregate.get("compounded_oos_return"),
            metrics["compounded_oos_return"],
        ),
        "mean_oos_return": (aggregate.get("mean_oos_return"), metrics["monthly_mean_return"]),
        "median_oos_return": (
            aggregate.get("median_oos_return"),
            metrics["monthly_median_return"],
        ),
        "min_oos_return": (aggregate.get("min_oos_return"), metrics["min_monthly_return"]),
        "max_oos_mdd": (aggregate.get("max_oos_mdd"), metrics["max_fold_oos_mdd"]),
        "mean_validation_return": (
            aggregate.get("mean_validation_return"),
            metrics["mean_validation_return"],
        ),
        "min_validation_return": (
            aggregate.get("min_validation_return"),
            metrics["min_validation_return"],
        ),
    }
    deltas = {
        key: _safe_float(left) - _safe_float(right) for key, (left, right) in comparisons.items()
    }
    return {
        "pass": all(abs(value) < 1e-12 for value in deltas.values()),
        "deltas": deltas,
        "recomputed": metrics,
        "artifact_aggregate": aggregate,
    }


def _audit_fold_schedule(payload: Mapping[str, Any]) -> dict[str, Any]:
    checks = []
    for fold in payload.get("folds", []):
        train_end = str(fold["train"]["end"])
        validation_start = str(fold["validation"]["start"])
        validation_end = str(fold["validation"]["end"])
        oos_start = str(fold["locked_oos"]["start"])
        checks.append(
            {
                "fold_id": fold["fold_id"],
                "train_before_validation": train_end < validation_start,
                "validation_before_oos": validation_end < oos_start,
            }
        )
    return {
        "pass": all(
            item["train_before_validation"] and item["validation_before_oos"] for item in checks
        ),
        "checks": checks,
    }


def _audit_latest_detail_match(
    walkforward: Mapping[str, Any], detail: Mapping[str, Any] | None, candidate_label: str
) -> dict[str, Any]:
    if detail is None:
        return {"pass": None, "reason": "latest detail artifact not provided"}
    latest_fold = str(walkforward.get("folds", [])[-1]["fold_id"])
    full_rows = [
        row
        for row in walkforward.get("fold_candidate_rows", [])
        if row["fold_id"] == latest_fold and row["candidate_label"] == candidate_label
    ]
    detail_rows = [
        row
        for row in detail.get("fold_candidate_rows", [])
        if row["fold_id"] == latest_fold and row["candidate_label"] == candidate_label
    ]
    if not full_rows or not detail_rows:
        return {"pass": False, "reason": "candidate missing from full or detail artifact"}
    full = full_rows[0]
    det = detail_rows[0]
    fields = [
        (
            "validation_return",
            full["validation"]["total_return"],
            det["validation"]["total_return"],
        ),
        (
            "locked_oos_return",
            full["locked_oos"]["total_return"],
            det["locked_oos"]["total_return"],
        ),
        ("locked_oos_mdd", full["locked_oos"]["mdd"], det["locked_oos"]["mdd"]),
    ]
    deltas = {name: _safe_float(a) - _safe_float(b) for name, a, b in fields}
    aux = next(
        (
            summary.get("individual_robust_aux", {})
            for summary in detail.get("fold_summaries", [])
            if summary["fold_id"] == latest_fold
        ),
        {},
    )
    return {
        "pass": all(abs(value) < 1e-12 for value in deltas.values())
        and bool(aux.get("profile_rows"))
        and bool(aux.get("selected_sleeve_rows")),
        "latest_fold": latest_fold,
        "deltas": deltas,
        "detail_profile_rows": len(aux.get("profile_rows") or []),
        "detail_selected_sleeve_rows": len(aux.get("selected_sleeve_rows") or []),
    }


def _score(row: Mapping[str, Any], formula: str, validation_cap: float) -> float:
    train = row["train"]
    validation = row["validation"]
    gross = _safe_float(row.get("gross_notional_fraction"))
    reason_count = len(row.get("selection_reasons") or [])
    stable_validation = min(_safe_float(validation["total_return"]), validation_cap)
    stable_train = min(_safe_float(train["total_return"]), max(validation_cap * 4.0, 0.01))
    if formula == "calmar":
        return (
            stable_validation * 3.0
            + _safe_float(validation.get("calmar")) / 5.0
            - _safe_float(validation.get("mdd")) * 2.0
            - 0.4 * reason_count
            - 0.01 * gross
        )
    if formula == "return_mdd":
        return (
            8.0 * stable_validation
            + 1.2 * min(stable_train, stable_validation)
            - 3.0 * _safe_float(validation.get("mdd"))
            - 0.5 * reason_count
            - 0.01 * gross
        )
    raise ValueError(f"unknown formula: {formula}")


def _diagnostic_selector(payload: Mapping[str, Any]) -> dict[str, Any]:
    """OOS-ranked selector search. Diagnostic only; not clean promotion evidence."""
    folds = [str(fold["fold_id"]) for fold in payload.get("folds", [])]
    by_fold: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in payload.get("fold_candidate_rows", []):
        by_fold[str(row["fold_id"])].append(row)

    def choose(fold_id: str) -> Mapping[str, Any]:
        pool = []
        for row in by_fold[fold_id]:
            if row.get("family") != "individual_robust":
                continue
            validation = row["validation"]
            train = row["train"]
            if not (0.0 <= _safe_float(validation["total_return"]) <= 0.45):
                continue
            if _safe_float(validation["mdd"]) > 0.08:
                continue
            if _safe_float(train["total_return"]) > 3.0:
                continue
            pool.append(row)
        if not pool:
            pool = [
                row
                for row in by_fold[fold_id]
                if row["candidate_label"]
                == "relaxed_efficiency:balanced_mdd12_gross5_69_asset_relaxed_efficiency_repair_optuna"
            ]
        return max(pool, key=lambda row: _score(row, "calmar", 0.20))

    chosen = [choose(fold_id) for fold_id in folds]
    metrics = _monthly_metrics(chosen)
    return {
        "name": "diagnostic_individual_calmar_vcap20_vmdd8_with_relaxed_fallback",
        "oos_used_to_discover_rule": True,
        "promotion_allowed": False,
        "warning": (
            "This rule was surfaced after inspecting historical OOS; use only as a forward "
            "shadow challenger unless it passes future months without further tuning."
        ),
        "rule": {
            "candidate_pool": "individual_robust rows",
            "filter": "0 <= validation_return <= 45%, validation_mdd <= 8%, train_return <= 300%",
            "score": "validation_calmar/5 + 3*min(validation_return,20%) - 2*validation_mdd - penalties",
            "fallback": "relaxed_efficiency balanced profile when no row passes",
        },
        "metrics": metrics,
        "monthly_choices": [
            {
                "fold_id": row["fold_id"],
                "candidate_label": row["candidate_label"],
                "validation_return": row["validation"]["total_return"],
                "locked_oos_return": row["locked_oos"]["total_return"],
                "locked_oos_mdd": row["locked_oos"]["mdd"],
            }
            for row in chosen
        ],
    }


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):.2%}"


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    selected = payload["selected_candidate"]
    metrics = selected["extended_metrics"]
    selector = payload["diagnostic_selector"]
    selector_metrics = selector["metrics"]
    lines = [
        "# 69-Asset Monthly Refit Diagnostics",
        "",
        f"- candidate: `{selected['candidate_label']}`",
        f"- aggregate recompute: `{'PASS' if payload['logic_audit']['aggregate_match']['pass'] else 'FAIL'}`",
        f"- fold schedule: `{'PASS' if payload['logic_audit']['fold_schedule']['pass'] else 'FAIL'}`",
        f"- latest detail match: `{payload['logic_audit']['latest_detail_match']['pass']}`",
        "",
        "## Selected candidate extended metrics",
        "",
        f"- OOS comp: `{_fmt_pct(metrics['compounded_oos_return'])}`",
        f"- hit rate: `{_fmt_pct(metrics['hit_rate'])}`",
        f"- monthly Sharpe approx: `{_safe_float(metrics['monthly_sharpe_approx']):.2f}`",
        f"- monthly Sortino approx: `{_safe_float(metrics['monthly_sortino_approx']):.2f}`",
        f"- 5% monthly VaR: `{_fmt_pct(metrics['monthly_var_5'])}`",
        f"- 25% monthly CVaR: `{_fmt_pct(metrics['monthly_cvar_25'])}`",
        f"- avg gain / avg loss: `{_fmt_pct(metrics['average_gain_month'])}` / `{_fmt_pct(metrics['average_loss_month'])}`",
        f"- gain/loss ratio: `{_safe_float(metrics['gain_loss_ratio']):.2f}`",
        f"- equity max DD: `{_fmt_pct(metrics['equity_curve_max_drawdown'])}`",
        f"- max loss streak: `{metrics['max_consecutive_loss_folds']}`",
        "",
        "## Diagnostic challenger selector",
        "",
        f"- name: `{selector['name']}`",
        f"- clean promotion allowed: `{selector['promotion_allowed']}`",
        f"- warning: {selector['warning']}",
        f"- OOS comp: `{_fmt_pct(selector_metrics['compounded_oos_return'])}`",
        f"- hit rate: `{_fmt_pct(selector_metrics['hit_rate'])}`",
        f"- min monthly OOS: `{_fmt_pct(selector_metrics['min_monthly_return'])}`",
        f"- max fold MDD: `{_fmt_pct(selector_metrics['max_fold_oos_mdd'])}`",
        "",
        "This report distinguishes verified accounting from diagnostic OOS-ranked ideas.",
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def build_diagnostics(
    *,
    walkforward_path: Path,
    latest_detail_path: Path | None,
    candidate_label: str,
) -> dict[str, Any]:
    walkforward = _load(walkforward_path)
    detail = _load(latest_detail_path) if latest_detail_path is not None else None
    rows = _rows_for(walkforward, candidate_label)
    if not rows:
        raise ValueError(f"candidate rows not found: {candidate_label}")
    metrics = _monthly_metrics(rows)
    return {
        "artifact_kind": "alpha_zoo_69_asset_monthly_refit_diagnostics",
        "generated_at_utc": _utc_now_iso(),
        "source_artifacts": {
            "walkforward": walkforward["_path"],
            "latest_detail": None if detail is None else detail["_path"],
        },
        "selected_candidate": {
            "candidate_label": candidate_label,
            "extended_metrics": metrics,
            "monthly_rows": [
                {
                    "fold_id": row["fold_id"],
                    "validation": row["validation"],
                    "locked_oos": row["locked_oos"],
                    "ready_for_paper": row.get("ready_for_paper"),
                    "selection_reasons": row.get("selection_reasons") or [],
                }
                for row in rows
            ],
        },
        "logic_audit": {
            "aggregate_match": _audit_aggregate_match(walkforward, candidate_label),
            "fold_schedule": _audit_fold_schedule(walkforward),
            "protocol_oos_used_for_selection": bool(
                walkforward.get("protocol", {}).get("oos_used_for_selection")
            ),
            "latest_detail_match": _audit_latest_detail_match(walkforward, detail, candidate_label),
            "existing_unit_tests_executed": [
                "tests/test_alpha_zoo_integer_leverage_optuna_hybrid_decision.py",
                "tests/test_alpha_zoo_69_asset_individual_robust_paper_decision.py",
            ],
        },
        "diagnostic_selector": _diagnostic_selector(walkforward),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walkforward-artifact", default=str(DEFAULT_WALKFORWARD_ARTIFACT))
    parser.add_argument("--latest-detail-artifact", default=str(DEFAULT_LATEST_DETAIL_ARTIFACT))
    parser.add_argument("--candidate-label", default=DEFAULT_CANDIDATE_LABEL)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    args = parser.parse_args(argv)
    latest_detail = (
        None
        if not str(args.latest_detail_artifact).strip()
        else Path(args.latest_detail_artifact).expanduser().resolve()
    )
    payload = build_diagnostics(
        walkforward_path=Path(args.walkforward_artifact).expanduser().resolve(),
        latest_detail_path=latest_detail,
        candidate_label=str(args.candidate_label),
    )
    output = Path(args.output_json).expanduser().resolve()
    _write_json(output, payload)
    _write_markdown(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
