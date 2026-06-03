#!/usr/bin/env python3
"""Fast clean selector sweep over existing monthly walk-forward fold rows.

This script does not rerun strategy optimization. It loads an existing
walk-forward JSON, removes post-OOS research/shadow rows, and evaluates a
predefined set of train/validation-only selectors by copying the selected
candidate's locked OOS metrics after selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    "clean_selector_sweep_latest.json"
)
DEFAULT_OUTPUT_MD = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_69_asset_mdd30_high_vol_20260602/"
    "clean_selector_sweep_latest.md"
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    return wf._safe_float(value, default)


def _snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": str(row.get("candidate_label")),
        "family": str(row.get("family")),
        "train_return": _safe_float(row.get("train", {}).get("total_return")),
        "train_mdd": _safe_float(row.get("train", {}).get("mdd")),
        "validation_return": _safe_float(row.get("validation", {}).get("total_return")),
        "validation_mdd": _safe_float(row.get("validation", {}).get("mdd")),
        "row": row,
    }


def _eligible(
    snap: Mapping[str, Any],
    *,
    max_validation_mdd: float,
    max_train_mdd: float = 0.65,
    min_train_return: float = -0.02,
    min_validation_return: float = -0.02,
) -> bool:
    return bool(
        _safe_float(snap["train_return"]) >= min_train_return
        and _safe_float(snap["validation_return"]) >= min_validation_return
        and _safe_float(snap["validation_mdd"]) <= max_validation_mdd
        and _safe_float(snap["train_mdd"]) <= max_train_mdd
    )


def _utility(snap: Mapping[str, Any]) -> float:
    return float(
        min(_safe_float(snap["validation_return"]), 1.0)
        + 0.20 * min(_safe_float(snap["train_return"]), 3.0)
        - 1.0 * _safe_float(snap["validation_mdd"])
        - 0.25 * _safe_float(snap["train_mdd"])
    )


def _calmar(snap: Mapping[str, Any]) -> float:
    return float(
        _safe_float(snap["validation_return"]) / max(_safe_float(snap["validation_mdd"]), 0.02)
    )


def _sharpe_proxy(snap: Mapping[str, Any]) -> float:
    return float(
        _safe_float(snap["validation_return"]) / max(_safe_float(snap["validation_mdd"]), 0.02)
        + 0.15 * _safe_float(snap["train_return"]) / max(_safe_float(snap["train_mdd"]), 0.05)
    )


def _stable_score(snap: Mapping[str, Any]) -> float:
    return float(
        min(_safe_float(snap["validation_return"]), _safe_float(snap["train_return"]) * 0.75)
        - 1.2 * _safe_float(snap["validation_mdd"])
        - 0.2 * _safe_float(snap["train_mdd"])
    )


def _selector_specs() -> list[
    tuple[str, Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any] | None]]
]:
    specs: list[tuple[str, Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any] | None]]] = []

    def select(
        snaps: Sequence[Mapping[str, Any]],
        *,
        max_validation_mdd: float,
        score_fn: Callable[[Mapping[str, Any]], float],
        families: set[str] | None = None,
    ) -> Mapping[str, Any] | None:
        pool = [
            snap
            for snap in snaps
            if _eligible(snap, max_validation_mdd=max_validation_mdd)
            and (families is None or str(snap["family"]) in families)
        ]
        return max(pool, key=score_fn, default=None)

    score_fns: tuple[tuple[str, Callable[[Mapping[str, Any]], float]], ...] = (
        ("val_return", lambda snap: _safe_float(snap["validation_return"])),
        ("calmar", _calmar),
        ("utility", _utility),
        ("sharpeproxy", _sharpe_proxy),
        ("stable", _stable_score),
    )
    for mdd in (0.10, 0.12, 0.15, 0.18, 0.22, 0.30):
        for score_name, score_fn in score_fns:
            specs.append(
                (
                    f"clean_selector_sweep:{score_name}_mdd{int(mdd * 100)}",
                    lambda snaps, mdd=mdd, score_fn=score_fn: select(
                        snaps,
                        max_validation_mdd=mdd,
                        score_fn=score_fn,
                    ),
                )
            )

    family_sets = {
        "dynamic_switch_only": {"dynamic_conviction_switch"},
        "strict_or_dynamic": {"strict_efficiency", "dynamic_conviction_switch"},
        "dynamic_aware_only": {"dynamic_aware_hybrid"},
        "clean_hybrid_only": {"dynamic_aware_hybrid", "cross_candidate_hybrid"},
        "profile_or_dynamic": {"profile_optuna", "dynamic_conviction_switch"},
    }
    for family_name, families in family_sets.items():
        for mdd in (0.15, 0.18, 0.22, 0.30):
            specs.append(
                (
                    f"clean_selector_sweep:{family_name}_utility_mdd{int(mdd * 100)}",
                    lambda snaps, mdd=mdd, families=families: select(
                        snaps,
                        max_validation_mdd=mdd,
                        score_fn=_utility,
                        families=families,
                    ),
                )
            )
    return specs


def _synthetic_selector_row(
    *,
    selector_label: str,
    fold_id: str,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(selected["row"])
    return {
        "fold_id": fold_id,
        "family": "clean_selector_sweep",
        "candidate_label": selector_label,
        "source_profile_id": selector_label,
        "train": row["train"],
        "validation": row["validation"],
        "locked_oos": row["locked_oos"],
        "selected_candidate_label": row.get("candidate_label"),
        "selected_family": row.get("family"),
        "selection_inputs": ["train", "validation"],
        "uses_locked_oos_for_selection": False,
        "same_month_self_feeding": False,
        "current_fold_oos_used_for_weighting": False,
        "post_oos_research_variant": False,
        "requires_fresh_forward_shadow": False,
        "clean_promotion_eligible": True,
        "ready_for_paper": False,
        "selection_reasons": [],
        "final_weights": {str(row.get("candidate_label")): 1.0},
        "asset_gross_notional_fraction": row.get("asset_gross_notional_fraction") or {},
        "profile_kind": "diagnostic_train_validation_only_clean_selector",
        "candidate_tier": "diagnostic_not_auto_promoted",
    }


def run_sweep(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized_rows = wf._sanitize_research_dependency_flags(
        list(payload.get("fold_candidate_rows") or [])
    )
    clean_rows = [row for row in sanitized_rows if row.get("clean_promotion_eligible")]
    by_fold: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_fold[str(row["fold_id"])].append(row)

    selector_rows: list[dict[str, Any]] = []
    choices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for selector_label, selector in _selector_specs():
        for fold_id in sorted(by_fold):
            snaps = [_snapshot(row) for row in by_fold[fold_id]]
            selected = selector(snaps)
            if selected is None:
                continue
            synthetic = _synthetic_selector_row(
                selector_label=selector_label,
                fold_id=fold_id,
                selected=selected,
            )
            selector_rows.append(synthetic)
            choices[selector_label].append(
                {
                    "fold_id": fold_id,
                    "selected_candidate_label": selected["label"],
                    "selected_family": selected["family"],
                    "validation_return": selected["validation_return"],
                    "validation_mdd": selected["validation_mdd"],
                    "locked_oos_return": synthetic["locked_oos"]["total_return"],
                    "locked_oos_mdd": synthetic["locked_oos"]["mdd"],
                }
            )

    aggregates = wf._aggregate_rows(selector_rows)
    by_comp = sorted(
        aggregates, key=lambda row: _safe_float(row["compounded_oos_return"]), reverse=True
    )
    baseline_clean = sorted(
        wf._aggregate_rows(clean_rows),
        key=lambda row: _safe_float(row["compounded_oos_return"]),
        reverse=True,
    )[:12]
    return {
        "artifact_kind": "alpha_zoo_69_asset_clean_selector_sweep",
        "generated_at_utc": _utc_now_iso(),
        "source_json": payload.get("output_paths", {}).get("json"),
        "method_note": (
            "diagnostic only: selectors use train/validation rows only and exclude post-OOS research variants; "
            "choosing a new selector by historical OOS still requires fresh-forward confirmation"
        ),
        "selector_count": len(_selector_specs()),
        "fold_count": len(by_fold),
        "selector_rows": selector_rows,
        "selector_choices": choices,
        "selector_aggregate_rankings_by_default_sort": aggregates,
        "selector_aggregate_rankings_by_comp": by_comp,
        "baseline_clean_rankings_by_comp": baseline_clean,
    }


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):.2%}"


def render_markdown(report: Mapping[str, Any]) -> str:
    top_selectors = list(report.get("selector_aggregate_rankings_by_comp") or [])[:20]
    baseline = list(report.get("baseline_clean_rankings_by_comp") or [])[:10]
    lines = [
        "# Clean selector sweep over existing walk-forward rows",
        "",
        f"- generated: `{report.get('generated_at_utc')}`",
        f"- selector count: `{report.get('selector_count')}`",
        f"- folds: `{report.get('fold_count')}`",
        f"- note: {report.get('method_note')}",
        "",
        "## Existing clean baseline by comp",
        "",
        "| Rank | Candidate | Family | OOS comp | Hit | Max OOS MDD | Sharpe | Sortino | PF |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(baseline, start=1):
        lines.append(
            f"| {idx} | `{row['candidate_label']}` | `{row['family']}` | "
            f"{_fmt_pct(row['compounded_oos_return'])} | "
            f"{row['positive_oos_folds']}/{row['fold_count']} | "
            f"{_fmt_pct(row['max_oos_mdd'])} | "
            f"{_safe_float(row.get('monthly_sharpe_approx')):.2f} | "
            f"{_safe_float(row.get('monthly_sortino_approx')):.2f} | "
            f"{_safe_float(row.get('profit_factor')):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic selector sweep by OOS comp",
            "",
            "| Rank | Selector | OOS comp | Hit | Max OOS MDD | Min OOS | Latest | Sharpe | Sortino | PF |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(top_selectors, start=1):
        lines.append(
            f"| {idx} | `{row['candidate_label']}` | "
            f"{_fmt_pct(row['compounded_oos_return'])} | "
            f"{row['positive_oos_folds']}/{row['fold_count']} | "
            f"{_fmt_pct(row['max_oos_mdd'])} | "
            f"{_fmt_pct(row['min_oos_return'])} | "
            f"{_fmt_pct(row['latest_oos_return'])} | "
            f"{_safe_float(row.get('monthly_sharpe_approx')):.2f} | "
            f"{_safe_float(row.get('monthly_sortino_approx')):.2f} | "
            f"{_safe_float(row.get('profit_factor')):.2f} |"
        )
    if top_selectors:
        best = top_selectors[0]
        choices = list(report.get("selector_choices", {}).get(best["candidate_label"], []))
        lines.extend(
            [
                "",
                f"## Best diagnostic selector choices: `{best['candidate_label']}`",
                "",
                "| Fold | Selected clean candidate | Family | Val | Val MDD | OOS | OOS MDD |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for choice in choices:
            lines.append(
                f"| `{choice['fold_id']}` | `{choice['selected_candidate_label']}` | "
                f"`{choice['selected_family']}` | "
                f"{_fmt_pct(choice['validation_return'])} | "
                f"{_fmt_pct(choice['validation_mdd'])} | "
                f"{_fmt_pct(choice['locked_oos_return'])} | "
                f"{_fmt_pct(choice['locked_oos_mdd'])} |"
            )
    lines.extend(
        [
            "",
            "## Clean interpretation",
            "",
            "- These selector rules are fold-clean: they select only from train/validation metrics and exclude post-OOS research rows.",
            "- However, picking a new selector because it ranks well on this already-reviewed OOS window would still be OOS-mining.",
            "- Therefore the existing clean baseline remains the promotion candidate unless this selector family is frozen and validated on fresh-forward data.",
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
    report = run_sweep(payload)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(wf._json_safe(report), indent=2, sort_keys=True) + "\n", "utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), "utf-8")
    top = list(report.get("selector_aggregate_rankings_by_comp") or [])[:5]
    print(
        json.dumps(
            {"output_json": str(out_json), "output_md": str(out_md), "top_5": top},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
