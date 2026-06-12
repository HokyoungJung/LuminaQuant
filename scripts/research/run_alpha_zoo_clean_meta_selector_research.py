#!/usr/bin/env python3
"""Fast post-OOS clean-input meta-selector research from monthly WF rows.

This script does not rerun source Optuna. It consumes an existing monthly-refit
walk-forward JSON, tests deterministic train/validation-only selector formulas
over already-evaluated clean rows, and writes a shadow-only artifact.  The row
choices for each fold never read that fold's locked OOS fields, but the selector
family/grid itself is introduced after reviewing historical OOS, so every output
is explicitly marked fresh-forward-required and non-promotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.optimization.search_policy import build_bounded_grid_combinations  # noqa: E402

DEFAULT_SOURCE_JSON = REPO_ROOT / (
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606/"
    "alpha_zoo_85_asset_lagged_shadow_router_scaled_latest_20260606.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_clean_meta_selector_research_20260607"
)

EVIDENCE_CLASS = "shadow-freeze-only"
PROMOTION_LABELS = (
    "deployable-paper",
    "small-real-sleeve candidate",
    EVIDENCE_CLASS,
    "reject",
)

FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "dynamic_strict_relaxed": (
        "dynamic_conviction_switch",
        "strict_efficiency",
        "relaxed_efficiency",
    ),
    "strict_relaxed": ("strict_efficiency", "relaxed_efficiency"),
    "dynamic_only": ("dynamic_conviction_switch",),
    "profile_strict_relaxed": ("profile_optuna", "strict_efficiency", "relaxed_efficiency"),
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except TypeError, ValueError:
        return default
    return out if math.isfinite(out) else default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compound(returns: Sequence[float]) -> float:
    equity = 1.0
    for item in returns:
        equity *= 1.0 + float(item)
    return float(equity - 1.0)


def _equity_mdd(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for item in returns:
        equity *= 1.0 + float(item)
        peak = max(peak, equity)
        if peak > 0.0:
            mdd = max(mdd, 1.0 - equity / peak)
    return float(mdd)


def _sharpe(returns: Sequence[float]) -> float:
    values = [float(item) for item in returns]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / (len(values) - 1)
    std = math.sqrt(max(0.0, variance))
    return float(mean / std * math.sqrt(12.0)) if std > 0.0 else 0.0


def _profit_factor(returns: Sequence[float]) -> tuple[float, bool]:
    gains = sum(item for item in returns if item > 0.0)
    losses = -sum(item for item in returns if item < 0.0)
    if losses <= 0.0:
        return 0.0, True
    return float(gains / losses), False


def _candidate_allowed(row: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    if not bool(row.get("clean_promotion_eligible")):
        return False
    if bool(row.get("nested_hybrid_dependency")):
        return False
    if bool(row.get("post_oos_research_variant")):
        return False
    if row.get("selection_reasons"):
        return False
    if str(row.get("family")) not in FAMILY_GROUPS[str(params["family_group"])]:
        return False

    train = row.get("train") or {}
    validation = row.get("validation") or {}
    train_return = _safe_float(train.get("total_return"))
    validation_return = _safe_float(validation.get("total_return"))
    train_mdd = _safe_float(train.get("mdd"))
    validation_mdd = _safe_float(validation.get("mdd"))
    if train_return <= 0.0 or validation_return <= 0.0:
        return False
    if validation_mdd > _safe_float(params["validation_mdd_cap"]):
        return False
    if train_mdd > _safe_float(params["train_mdd_cap"]):
        return False
    return validation_return - max(train_return, 0.0) <= _safe_float(params["validation_spike_cap"])


def _selector_score(row: Mapping[str, Any], params: Mapping[str, Any]) -> float:
    """Score using train/validation fields only; locked OOS is intentionally absent."""
    train = row.get("train") or {}
    validation = row.get("validation") or {}
    train_return = _safe_float(train.get("total_return"))
    validation_return = _safe_float(validation.get("total_return"))
    train_mdd = _safe_float(train.get("mdd"))
    validation_mdd = _safe_float(validation.get("mdd"))
    validation_calmar = validation_return / max(validation_mdd, 0.01)
    train_calmar = train_return / max(train_mdd, 0.01)
    validation_spike = max(0.0, validation_return - max(train_return, 0.0))
    return float(
        _safe_float(params["return_weight"]) * min(train_return, validation_return)
        + _safe_float(params["calmar_weight"]) * (validation_calmar + train_calmar)
        - _safe_float(params["mdd_weight"]) * (validation_mdd + 0.5 * train_mdd)
        - _safe_float(params["spike_penalty_weight"]) * validation_spike
    )


def _build_grid() -> list[dict[str, Any]]:
    grid = build_bounded_grid_combinations(
        {
            "family_group": tuple(FAMILY_GROUPS),
            "validation_mdd_cap": (0.12, 0.18, 0.25, 0.35),
            "train_mdd_cap": (0.30, 0.40),
            "validation_spike_cap": (0.05, 0.15, 0.50),
            "return_weight": (2.0, 4.0, 8.0),
            "calmar_weight": (0.02, 0.05, 0.10),
            "mdd_weight": (0.5, 1.0, 2.0, 4.0),
            "spike_penalty_weight": (2.0,),
        },
        max_combinations=2048,
        justification=(
            "bounded deterministic meta-selector grid over existing clean fold rows; "
            "candidate choice uses train/validation only and all outputs remain fresh-forward shadow"
        ),
    )
    return grid.combinations


def _gate_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    positive = int(result.get("positive_oos_folds") or 0)
    folds = int(result.get("fold_count") or 0)
    max_mdd = _safe_float(result.get("max_oos_mdd"))
    blockers = [
        "post_oos_selector_grid_ranking_uses_historical_locked_oos",
        "fresh_forward_required_before_promotion",
    ]
    if folds >= 10 and positive < 5:
        blockers.append("positive_oos_folds_below_5_of_10")
    if max_mdd > 0.30:
        blockers.append("max_bar_oos_mdd_over_30pct_real_sleeve_block")
    return {
        "evidence_class": EVIDENCE_CLASS,
        "allowed_labels": list(PROMOTION_LABELS),
        "deployment_label": EVIDENCE_CLASS,
        "ready_for_real": False,
        "real_money_execution": False,
        "real_sleeve_allowed": False,
        "clean_mechanics": {
            "fold_choice_uses_locked_oos": False,
            "nested_hybrid_dependency": False,
            "candidate_filter_rejects_post_oos_rows": True,
        },
        "label_blockers": blockers,
        "numeric_gates": {
            "positive_oos_folds": positive,
            "fold_count": folds,
            "max_oos_mdd": max_mdd,
            "ten_bps_base_cost_required": True,
            "fifteen_bps_stress_required_for_real_sleeve": True,
        },
        "theory_plausibility": {
            "status": "plausible_as_freeze_candidate_only",
            "rationale": (
                "The selector ranks clean leaf/dynamic rows by train/validation return, "
                "Calmar-style risk efficiency, drawdown, and validation-spike penalty. "
                "Those are economically interpretable risk/return controls, but the "
                "formula family/grid was chosen after historical OOS review."
            ),
        },
    }


def _build_freeze_manifest(
    *,
    source_json: Path,
    source: Mapping[str, Any],
    output_dir: Path,
    grid_candidate_count: int,
    generated_at_utc: str,
) -> dict[str, Any]:
    protocol = source.get("protocol", {})
    data_coverage = source.get("data_coverage", {})
    return {
        "artifact_kind": "alpha_zoo_clean_meta_selector_freeze_manifest",
        "generated_at_utc": generated_at_utc,
        "selector_family": "clean_input_meta_selector",
        "selector_origin": "post_oos_research_freeze_candidate",
        "evidence_class_cap": EVIDENCE_CLASS,
        "allowed_family_groups": {key: list(value) for key, value in FAMILY_GROUPS.items()},
        "allowed_feature_inputs": [
            "family",
            "clean_promotion_eligible",
            "nested_hybrid_dependency",
            "post_oos_research_variant",
            "selection_reasons",
            "train.total_return",
            "train.mdd",
            "validation.total_return",
            "validation.mdd",
        ],
        "banned_selection_fields": [
            "locked_oos",
            "oos_return",
            "oos_mdd",
            "compounded_oos_return",
            "latest_oos_return",
            "positive_oos_folds",
            "monthly_sharpe_approx",
            "profit_factor",
        ],
        "objective_function": {
            "name": "train_validation_deployable_utility",
            "score_terms": [
                "return_weight * min(train_return, validation_return)",
                "calmar_weight * (validation_calmar + train_calmar)",
                "-mdd_weight * (validation_mdd + 0.5 * train_mdd)",
                "-spike_penalty_weight * max(0, validation_return - train_return)",
            ],
            "selection_metric_rule": (
                "Fold choice uses only train/validation score. Locked OOS is attached "
                "only after choice. Grid ranking is historical-OOS diagnostic and "
                "therefore caps the label at shadow-freeze-only."
            ),
        },
        "fold_schedule": protocol,
        "universe": {
            "requested_symbol_count": data_coverage.get("requested_symbol_count"),
            "loaded_symbol_count": data_coverage.get("loaded_symbol_count"),
            "global_earliest_utc": data_coverage.get("global_earliest_utc"),
            "global_latest_utc": data_coverage.get("global_latest_utc"),
            "timeframes": source.get("timeframes"),
        },
        "trial_budget": {
            "search_method": "bounded_deterministic_grid",
            "grid_candidate_count": grid_candidate_count,
            "optuna_trials": 0,
            "random_seeds": [],
        },
        "promotion_labels": list(PROMOTION_LABELS),
        "hard_gates": [
            "no_nested_oos_mining",
            "execution_cost_gate",
            "theory_plausibility_gate",
        ],
        "source_artifacts": {
            "source_json": str(source_json),
            "source_sha256": _sha256_file(source_json),
            "source_artifact_kind": source.get("artifact_kind"),
        },
        "report_destination": str(output_dir),
        "command_ledger": [
            {
                "cwd": str(REPO_ROOT),
                "command": (
                    "python scripts/research/run_alpha_zoo_clean_meta_selector_research.py "
                    f"--source-json {source_json} --output-dir {output_dir}"
                ),
            }
        ],
        "environment": {
            "python_version": sys.version.split()[0],
        },
    }


def evaluate_selector(
    *, rows_by_fold: Mapping[str, Sequence[Mapping[str, Any]]], params: Mapping[str, Any]
) -> dict[str, Any]:
    choices: list[dict[str, Any]] = []
    oos_returns: list[float] = []
    oos_mdds: list[float] = []
    for fold_id in sorted(rows_by_fold):
        scored = [
            (_selector_score(row, params), row)
            for row in rows_by_fold[fold_id]
            if _candidate_allowed(row, params)
        ]
        if not scored:
            choices.append(
                {
                    "fold_id": fold_id,
                    "candidate_label": "cash:no_eligible_candidate",
                    "family": "cash",
                    "selector_score": 0.0,
                    "oos_return": 0.0,
                    "oos_mdd": 0.0,
                }
            )
            oos_returns.append(0.0)
            oos_mdds.append(0.0)
            continue
        score, row = max(scored, key=lambda item: (item[0], str(item[1].get("candidate_label"))))
        locked_oos = row.get("locked_oos") or {}
        oos_return = _safe_float(locked_oos.get("total_return"))
        oos_mdd = _safe_float(locked_oos.get("mdd"))
        choices.append(
            {
                "fold_id": fold_id,
                "candidate_label": str(row.get("candidate_label")),
                "family": str(row.get("family")),
                "selector_score": float(score),
                "oos_return": oos_return,
                "oos_mdd": oos_mdd,
                "train_return": _safe_float((row.get("train") or {}).get("total_return")),
                "validation_return": _safe_float((row.get("validation") or {}).get("total_return")),
                "validation_mdd": _safe_float((row.get("validation") or {}).get("mdd")),
            }
        )
        oos_returns.append(oos_return)
        oos_mdds.append(oos_mdd)

    profit_factor, pf_unbounded = _profit_factor(oos_returns)
    result: dict[str, Any] = {
        "selector_id": "clean_input_meta_selector",
        "params": dict(params),
        "choices": choices,
        "fold_count": len(oos_returns),
        "compounded_oos_return": _compound(oos_returns),
        "annualized_oos_return_approx": (1.0 + _compound(oos_returns))
        ** (12.0 / max(1, len(oos_returns)))
        - 1.0,
        "monthly_equity_mdd": _equity_mdd(oos_returns),
        "max_oos_mdd": max(oos_mdds) if oos_mdds else 0.0,
        "min_oos_return": min(oos_returns) if oos_returns else 0.0,
        "latest_oos_return": oos_returns[-1] if oos_returns else 0.0,
        "positive_oos_folds": sum(1 for item in oos_returns if item > 0.0),
        "monthly_sharpe_approx": _sharpe(oos_returns),
        "profit_factor": profit_factor,
        "profit_factor_unbounded": pf_unbounded,
        "uses_locked_oos_for_fold_selection": False,
        "uses_locked_oos_for_selector_grid_ranking": True,
        "post_oos_research_variant": True,
        "requires_fresh_forward_shadow": True,
        "clean_promotion_eligible": False,
        "evidence_class": EVIDENCE_CLASS,
        "deployment_label": EVIDENCE_CLASS,
        "ready_for_real": False,
        "real_money_execution": False,
    }
    result["gate_summary"] = _gate_summary(result)
    return result


def run(source_json: Path, output_dir: Path) -> dict[str, Any]:
    source = json.loads(source_json.read_text(encoding="utf-8"))
    rows_by_fold: dict[str, list[Mapping[str, Any]]] = {}
    for row in source.get("fold_candidate_rows", []):
        rows_by_fold.setdefault(str(row.get("fold_id")), []).append(row)
    grid = _build_grid()
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_manifest_path = output_dir / "clean_meta_selector_freeze_manifest_latest.json"
    freeze_manifest = _build_freeze_manifest(
        source_json=source_json,
        source=source,
        output_dir=output_dir,
        grid_candidate_count=len(grid),
        generated_at_utc=_utc_now_iso(),
    )
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ranked = sorted(
        (evaluate_selector(rows_by_fold=rows_by_fold, params=params) for params in grid),
        key=lambda row: (
            _safe_float(row.get("compounded_oos_return")),
            -_safe_float(row.get("monthly_equity_mdd")),
            -_safe_float(row.get("max_oos_mdd")),
        ),
        reverse=True,
    )
    payload = {
        "artifact_kind": "alpha_zoo_clean_meta_selector_research",
        "generated_at_utc": _utc_now_iso(),
        "source_json": str(source_json),
        "source_artifact_kind": source.get("artifact_kind"),
        "protocol": source.get("protocol", {}),
        "selector_policy": {
            "fold_choice_inputs": ["train", "validation"],
            "locked_oos_used_for_fold_selection": False,
            "locked_oos_used_for_grid_ranking": True,
            "evidence_class_cap": EVIDENCE_CLASS,
            "interpretation": "post-OOS research only; freeze before fresh-forward use",
        },
        "freeze_manifest_path": str(freeze_manifest_path),
        "freeze_manifest_sha256": _sha256_file(freeze_manifest_path),
        "evidence_classes": list(PROMOTION_LABELS),
        "hard_gate_summary": {
            "no_nested_oos_mining": "mechanically_passed_for_fold_choice_but_oos_inspired_grid_caps_label",
            "execution_cost_gate": "base_10bps_in_source_artifact_only; live_fill_telemetry_required",
            "theory_plausibility_gate": "passed_as_freeze_candidate_only",
        },
        "degrees_of_freedom": {
            "primary_selector_family_count": 1,
            "control_family_count": 0,
            "grid_candidate_count": len(grid),
            "failed_or_demoted_candidates_disclosed": True,
        },
        "grid_candidate_count": len(grid),
        "best_selector": ranked[0] if ranked else {},
        "top_selectors": ranked[:20],
        "ready_for_real": False,
        "real_money_execution": False,
    }
    output_json = output_dir / "clean_meta_selector_research_latest.json"
    output_md = output_dir / "clean_meta_selector_research_latest.md"
    payload["output_paths"] = {"json": str(output_json), "markdown": str(output_md)}
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:.2f}%"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    best = payload.get("best_selector") or {}
    lines = [
        "# Alpha Zoo clean-input meta-selector research",
        "",
        f"- generated: `{payload.get('generated_at_utc')}`",
        f"- source: `{payload.get('source_json')}`",
        f"- freeze manifest: `{payload.get('freeze_manifest_path')}`",
        "- fold choice inputs: `train + validation only`",
        "- locked-OOS use: `grid ranking/report only`, not per-fold selection",
        f"- evidence class cap: `{payload.get('selector_policy', {}).get('evidence_class_cap')}`",
        "- status: `post-OOS research / fresh-forward shadow required / real-money false`",
        "",
        "## Best selector",
        "",
        f"- deployment label: `{best.get('deployment_label')}`",
        f"- OOS comp: `{_fmt_pct(best.get('compounded_oos_return'))}`",
        f"- annualized approx: `{_fmt_pct(best.get('annualized_oos_return_approx'))}`",
        f"- monthly equity MDD: `{_fmt_pct(best.get('monthly_equity_mdd'))}`",
        f"- max bar OOS MDD: `{_fmt_pct(best.get('max_oos_mdd'))}`",
        f"- positive folds: `{best.get('positive_oos_folds')}/{best.get('fold_count')}`",
        f"- Sharpe approx: `{_safe_float(best.get('monthly_sharpe_approx')):.2f}`",
        f"- params: `{json.dumps(best.get('params', {}), sort_keys=True)}`",
        "",
        "## Fold choices",
        "",
        "| Fold | Candidate | Family | Selector score | OOS | OOS MDD |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for choice in best.get("choices", []):
        lines.append(
            "| `{fold}` | `{label}` | `{family}` | {score:.4f} | {ret} | {mdd} |".format(
                fold=choice.get("fold_id"),
                label=choice.get("candidate_label"),
                family=choice.get("family"),
                score=_safe_float(choice.get("selector_score")),
                ret=_fmt_pct(choice.get("oos_return")),
                mdd=_fmt_pct(choice.get("oos_mdd")),
            )
        )
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "This artifact is not clean promotion evidence. It is a bounded way to identify a selector formula to freeze before future fresh-forward evaluation.",
            "The grid ranking uses the historical locked-OOS window diagnostically, so the label is capped at `shadow-freeze-only` even though each fold choice uses only train/validation fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(Path(args.source_json), Path(args.output_dir))
    print(json.dumps(payload["best_selector"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
