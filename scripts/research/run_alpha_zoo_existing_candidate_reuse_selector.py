#!/usr/bin/env python3
"""Reuse-selector research over existing clean new-alpha candidate rows.

This script does not rerun source walk-forward generation. It consumes an existing
`alpha_zoo_clean_new_alpha_discovery` JSON, ranks already-evaluated candidates
inside each fold using train/validation-only scores, then attaches locked-OOS
report diagnostics after the fold choices are frozen. Because the selector
variants are introduced after seeing historical OOS, every output is
fresh-forward-required and non-promotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_clean_new_alpha_discovery as clean  # noqa: E402

DEFAULT_SOURCE_JSON = REPO_ROOT / (
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "indicator_kalman_ml_robust_selector_full_universe_20260609/"
    "clean_new_alpha_discovery_latest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "existing_candidate_reuse_selector_20260609"
)

VARIANTS = ("robust_top1", "robust_top2_equal", "robust_diverse3_equal")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_by_fold(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        fold_id = str(row.get("fold_id") or "")
        if fold_id:
            grouped[fold_id].append(row)
    return dict(sorted(grouped.items()))


def _eligible_sorted(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible = [
        row
        for row in rows
        if clean._eligible_for_policy(row, selection_policy=clean.ROBUST_SELECTION_POLICY)
    ]
    return sorted(
        eligible,
        key=lambda row: (
            clean._selection_score(row, selection_policy=clean.ROBUST_SELECTION_POLICY),
            str(row.get("model_id")),
        ),
        reverse=True,
    )


def _pick_variant(rows: Sequence[Mapping[str, Any]], variant: str) -> list[Mapping[str, Any]]:
    ranked = _eligible_sorted(rows)
    if variant == "robust_top1":
        return ranked[:1]
    if variant == "robust_top2_equal":
        return ranked[:2]
    if variant == "robust_diverse3_equal":
        picked: list[Mapping[str, Any]] = []
        families: set[str] = set()
        symbols: set[str] = set()
        for row in ranked:
            family = str(row.get("family"))
            symbol = str(row.get("symbol"))
            if family in families or symbol in symbols:
                continue
            picked.append(row)
            families.add(family)
            symbols.add(symbol)
            if len(picked) >= 3:
                return picked
        for row in ranked:
            if row not in picked:
                picked.append(row)
            if len(picked) >= 3:
                return picked
        return picked
    raise ValueError(f"unknown variant: {variant}")


def _fold_selection_row(
    fold_id: str, picked: Sequence[Mapping[str, Any]], variant: str
) -> dict[str, Any]:
    returns = [_safe_float(row.get("locked_oos_return_report_only")) for row in picked]
    mdds = [_safe_float(row.get("locked_oos_mdd_report_only")) for row in picked]
    return {
        "fold_id": fold_id,
        "variant": variant,
        "selected_count": len(picked),
        "locked_oos_return_report_only": sum(returns) / len(returns) if returns else 0.0,
        "locked_oos_mdd_report_only": max(mdds) if mdds else 0.0,
        "selected_candidates": [
            {
                "model_id": row.get("model_id"),
                "family": row.get("family"),
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "train_return": row.get("train_return"),
                "validation_return": row.get("validation_return"),
                "selection_score_robust_v1_train_validation_only": clean._selection_score(
                    row, selection_policy=clean.ROBUST_SELECTION_POLICY
                ),
                "locked_oos_return_report_only": row.get("locked_oos_return_report_only"),
                "locked_oos_mdd_report_only": row.get("locked_oos_mdd_report_only"),
            }
            for row in picked
        ],
    }


def evaluate_variants(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = _group_by_fold(rows)
    results: dict[str, Any] = {}
    for variant in VARIANTS:
        selected_rows = [
            _fold_selection_row(fold_id, _pick_variant(fold_rows, variant), variant)
            for fold_id, fold_rows in grouped.items()
        ]
        aggregate = clean._aggregate(selected_rows)
        aggregate["mean_selected_count"] = (
            sum(row["selected_count"] for row in selected_rows) / len(selected_rows)
            if selected_rows
            else 0.0
        )
        results[variant] = {
            "aggregate": aggregate,
            "selected_fold_rows": selected_rows,
        }
    return results


def _best_variant(results: Mapping[str, Any]) -> str | None:
    if not results:
        return None
    return max(
        results,
        key=lambda name: (
            _safe_float(results[name]["aggregate"].get("compounded_oos_return")),
            -_safe_float(results[name]["aggregate"].get("monthly_equity_mdd")),
            _safe_float(results[name]["aggregate"].get("positive_oos_folds")),
        ),
    )


def run(*, source_json: Path, output_dir: Path) -> dict[str, Any]:
    source = json.loads(source_json.read_text(encoding="utf-8"))
    rows = source.get("candidate_rows") or []
    if not isinstance(rows, list):
        raise ValueError("source JSON candidate_rows must be a list")
    results = evaluate_variants(rows)
    best = _best_variant(results)
    family_counts = Counter(str(row.get("family")) for row in rows)
    payload = {
        "artifact_kind": "alpha_zoo_existing_candidate_reuse_selector_research",
        "generated_at_utc": _utc_now_iso(),
        "source_json": str(source_json),
        "source_sha256": _sha256_file(source_json),
        "source_search_hash": source.get("pre_registered_search_space_sha256"),
        "candidate_rows": len(rows),
        "fold_count": len(_group_by_fold(rows)),
        "candidate_family_counts": dict(family_counts.most_common()),
        "selection_inputs": ["train", "validation"],
        "locked_oos_role": "report_only_after_reuse_selector_freeze",
        "selector_status": "post_failure_reuse_research_requires_fresh_forward",
        "variants": results,
        "best_variant_by_report_oos": best,
        "decision": {
            "promotion": "blocked",
            "ready_for_real": False,
            "real_money_execution": False,
            "shadow_execution": False,
            "reason": (
                "Existing candidates are reused with train/validation-only fold choices, "
                "but the reuse selector variants were designed after historical locked-OOS review. "
                "A fresh-forward slice and paper/fill telemetry are required before promotion."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "existing_candidate_reuse_selector_latest.json"
    output_md = output_dir / "existing_candidate_reuse_selector_latest.md"
    payload["output_paths"] = {"json": str(output_json), "markdown": str(output_md)}
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:.2f}%"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Existing candidate reuse selector research",
        "",
        f"- generated: `{payload.get('generated_at_utc')}`",
        f"- candidate rows: `{payload.get('candidate_rows')}`",
        f"- folds: `{payload.get('fold_count')}`",
        "- selection input: `train + validation only`",
        "- locked-OOS: `report only after fold freeze`",
        "- promotion: `blocked; fresh-forward required`",
        "",
        "## Variants",
        "",
        "| Variant | OOS comp | Annualized | MDD | Positive folds | PF |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    variants = payload.get("variants") or {}
    for name, result in variants.items():
        agg = result.get("aggregate") or {}
        lines.append(
            "| `{name}` | {comp} | {ann} | {mdd} | {pos}/{folds} | {pf:.2f} |".format(
                name=name,
                comp=_fmt_pct(agg.get("compounded_oos_return")),
                ann=_fmt_pct(agg.get("annualized_oos_return_approx")),
                mdd=_fmt_pct(agg.get("monthly_equity_mdd")),
                pos=agg.get("positive_oos_folds"),
                folds=agg.get("fold_count"),
                pf=_safe_float(agg.get("profit_factor")),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(source_json=Path(args.source_json), output_dir=Path(args.output_dir))
    best = payload.get("best_variant_by_report_oos")
    print(json.dumps(payload["variants"][best]["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
