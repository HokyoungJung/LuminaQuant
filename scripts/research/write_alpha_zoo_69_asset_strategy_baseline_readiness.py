#!/usr/bin/env python3
"""Write a freeze/readiness artifact for the user-designated 69-asset baselines.

The artifact is intentionally conservative: it does not promote live capital and
does not rerun optimization.  It gathers the current repository evidence for the
three designated reference strategies, records the exact source artifact hashes,
and emits the next gates required before any real-money review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ALPHA_V2_ROOT = (
    REPO_ROOT
    / "var"
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260508"
    / "alpha_v2"
)

DEFAULT_OUTPUT_DIR = ALPHA_V2_ROOT / "alpha_zoo_69_asset_strategy_baseline_readiness_20260613"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "strategy_baseline_readiness_latest.json"
DEFAULT_OUTPUT_MD = DEFAULT_OUTPUT_DIR / "strategy_baseline_readiness_latest.md"

BASELINE_LABELS = (
    "relaxed_efficiency:hybrid_v3_5",
    "fixed_relaxed_dynamic_blend:relaxed60_dynamic40",
    "dynamic_conviction_switch:t0.90_risk_capped_fallback",
)

DEFAULT_SOURCE_PATHS = {
    "exact_blend_20260603": ALPHA_V2_ROOT
    / "alpha_zoo_69_asset_exact_blend_full_tuning_20260603"
    / "exact_blend_full_tuning_walkforward_latest.json",
    "exact_blend_report_ko_20260603": ALPHA_V2_ROOT
    / "alpha_zoo_69_asset_exact_blend_full_tuning_20260603"
    / "exact_blend_selection_report_ko.md",
    "best_strategy_final_20260601": ALPHA_V2_ROOT
    / "alpha_zoo_69_asset_best_strategy_factory_20260601"
    / "best_strategy_final_recommendation_latest.json",
    "no_nested_recompute_20260604": ALPHA_V2_ROOT
    / "alpha_zoo_69_asset_no_nested_clean_recompute_20260604"
    / "no_nested_clean_recompute_latest.json",
    "no_nested_full_eval_20260604": ALPHA_V2_ROOT
    / "alpha_zoo_69_asset_clean_non_nested_full_eval_20260604_final"
    / "clean_non_nested_monthly_refit_full_20260604_final.json",
    "deep_research_conclusion_20260607": ALPHA_V2_ROOT
    / "deep_research_best_strategy_clean_oos_20260607"
    / "deep_research_best_strategy_clean_oos_20260607.json",
    "current_search_summary_20260609": ALPHA_V2_ROOT
    / "current_search_residual_dispersion_summary_20260609"
    / "current_search_residual_dispersion_summary_20260609.json",
    "existing_candidate_reuse_20260609": ALPHA_V2_ROOT
    / "existing_candidate_reuse_selector_20260609"
    / "existing_candidate_reuse_selector_latest.json",
    "leaf_rebuild_shadow_20260613": DEFAULT_OUTPUT_DIR / "leaf_rebuild_shadow_latest.json",
    "live_readiness_preflight_20260613": DEFAULT_OUTPUT_DIR
    / "live_readiness_preflight_latest.json",
}

ROW_SECTIONS = (
    "aggregate_rankings",
    "aggregate_rows",
    "comparison_rows",
    "clean_rankings",
    "clean_promotion_rankings",
    "clean_aggregate_rankings",
    "demoted_rankings",
    "demoted_nested_or_historical_rankings",
    "rankings",
)

METRIC_FIELDS = (
    "candidate_label",
    "family",
    "fold_count",
    "clean_promotion_eligible",
    "nested_hybrid_dependency",
    "post_oos_research_variant",
    "requires_fresh_forward_shadow",
    "uses_locked_oos_for_selection",
    "non_clean_reasons",
    "hard_stop_promotable",
    "hard_stop_reasons",
    "compounded_oos_return",
    "annualized_oos_return_approx",
    "max_oos_mdd",
    "monthly_equity_mdd",
    "monthly_sharpe_approx",
    "monthly_sortino_approx",
    "profit_factor",
    "positive_oos_folds",
    "oos_hit_rate",
    "min_oos_return",
    "latest_oos_return",
    "ready_for_paper_folds",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def _pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100.0:.2f}%"


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in METRIC_FIELDS if key in row}


def _compact_rows(rows: list[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return [_compact_row(row) for row in rows[:limit]]


def _rows_from_section(payload: Mapping[str, Any], section: str) -> list[Mapping[str, Any]]:
    rows = payload.get(section)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def find_candidate_rows(payload: Mapping[str, Any], label: str) -> dict[str, list[dict[str, Any]]]:
    """Return compact candidate rows by known aggregate/comparison sections."""
    by_section: dict[str, list[dict[str, Any]]] = {}
    for section in ROW_SECTIONS:
        rows: list[dict[str, Any]] = []
        for row in _rows_from_section(payload, section):
            row_label = row.get("candidate_label") or row.get("candidate") or row.get("label")
            if row_label == label:
                rows.append(_compact_row(row))
        if rows:
            by_section[section] = rows
    return by_section


def source_manifest(source_paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for name, path in sorted(source_paths.items()):
        resolved = path.resolve()
        manifest[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": _sha256(path),
            "size_bytes": resolved.stat().st_size if path.exists() and path.is_file() else None,
        }
    return manifest


def _primary_row(
    evidence: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    *,
    preferred_source: str,
    fallback_source: str | None = None,
) -> dict[str, Any] | None:
    for source in (preferred_source, fallback_source):
        if not source:
            continue
        sections = evidence.get(source) or {}
        for section in ("aggregate_rankings", "comparison_rows", "clean_rankings"):
            rows = sections.get(section)
            if rows:
                row = dict(rows[0])
                row["_source"] = source
                row["_section"] = section
                return row
    return None


def _status_for_label(label: str, primary: Mapping[str, Any] | None) -> tuple[str, list[str]]:
    if label == "fixed_relaxed_dynamic_blend:relaxed60_dynamic40":
        return (
            "diagnostic_shadow_only_rebuild_leaf_first",
            [
                "nested_hybrid_dependency",
                "post_oos_research_variant",
                "requires_fresh_forward_shadow",
                "old fixed blend family is disabled by current no-nested policy",
            ],
        )
    if label == "relaxed_efficiency:hybrid_v3_5":
        blockers = [
            "no fresh-forward fold after freeze",
            "no paper/testnet fill, BBO, slippage, partial-fill, cancel/reconcile telemetry",
            "historical OOS hit rate is only 5/10",
        ]
        if (
            primary
            and _safe_float(primary.get("max_oos_mdd"))
            and float(primary["max_oos_mdd"]) >= 0.18
        ):
            blockers.append("max OOS MDD near or above 18% review threshold")
        return ("historical_clean_reference_shadow_only", blockers)
    if label == "dynamic_conviction_switch:t0.90_risk_capped_fallback":
        return (
            "paper_control_shadow_only",
            [
                "risk-capped rule is forward-shadow only in final recommendation",
                "artifact-to-artifact OOS comp is unstable; do not use headline 53.38% as live expectation",
                "no paper/testnet execution telemetry",
            ],
        )
    return ("unknown_reference", ["label is not in the designated baseline set"])


def _assessment(
    label: str, evidence: Mapping[str, Mapping[str, list[dict[str, Any]]]]
) -> dict[str, Any]:
    if label == "dynamic_conviction_switch:t0.90_risk_capped_fallback":
        primary = _primary_row(
            evidence,
            preferred_source="best_strategy_final_20260601",
            fallback_source="no_nested_full_eval_20260604",
        )
    else:
        primary = _primary_row(
            evidence,
            preferred_source="exact_blend_20260603",
            fallback_source="no_nested_recompute_20260604",
        )
    status, blockers = _status_for_label(label, primary)
    return {
        "label": label,
        "status": status,
        "primary_metrics": primary or {},
        "real_money_ready": False,
        "real_money_execution_allowed": False,
        "blockers": blockers,
        "evidence": evidence,
    }


def _source_universe_symbols(loaded_sources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    exact = loaded_sources.get("exact_blend_20260603") or {}
    universe = exact.get("universe") if isinstance(exact, Mapping) else None
    if not isinstance(universe, Mapping):
        return []
    symbols = universe.get("symbols")
    if not isinstance(symbols, list):
        return []
    return [str(symbol) for symbol in symbols if str(symbol).strip()]


def _leaf_rebuild_command(symbols: list[str]) -> str:
    symbol_arg = f" --symbols {','.join(symbols)}" if symbols else ""
    return (
        "uv run python scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py"
        f"{symbol_arg}"
        " --families relaxed_efficiency,strict_efficiency,teacher_leaf_blend"
        " --source-symbol-workers 1"
        " --checkpoint-interval 1"
        " --checkpoint-markdown-interval 0"
        " --output-json var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
        "alpha_zoo_69_asset_strategy_baseline_readiness_20260613/leaf_rebuild_shadow_latest.json"
        " --output-md var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
        "alpha_zoo_69_asset_strategy_baseline_readiness_20260613/leaf_rebuild_shadow_latest.md"
    )


def _first_candidate_row(
    payload: Mapping[str, Any],
    label: str,
    *,
    sections: tuple[str, ...] = ROW_SECTIONS,
) -> dict[str, Any] | None:
    for section in sections:
        for row in _rows_from_section(payload, section):
            row_label = row.get("candidate_label") or row.get("candidate") or row.get("label")
            if row_label == label:
                compact = _compact_row(row)
                compact["_section"] = section
                return compact
    return None


def _leaf_rebuild_summary(loaded_sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = loaded_sources.get("leaf_rebuild_shadow_20260613")
    if not isinstance(payload, Mapping):
        return {"exists": False}

    data_coverage = (
        payload.get("data_coverage") if isinstance(payload.get("data_coverage"), Mapping) else {}
    )
    universe = payload.get("universe") if isinstance(payload.get("universe"), Mapping) else {}
    clean_rows = _rows_from_section(payload, "clean_promotion_rankings") or [
        row
        for row in _rows_from_section(payload, "aggregate_rankings")
        if row.get("clean_promotion_eligible") is True
    ]
    baseline_rows = {
        label: _first_candidate_row(
            payload,
            label,
            sections=(
                "aggregate_rankings",
                "clean_promotion_rankings",
                "demoted_nested_or_historical_rankings",
            ),
        )
        for label in BASELINE_LABELS
    }
    top_clean = _compact_rows(clean_rows, limit=8)
    top_promotable = _compact_rows(
        [row for row in clean_rows if row.get("hard_stop_promotable") is True],
        limit=5,
    )
    return {
        "exists": True,
        "generated_at_utc": payload.get("generated_at_utc") or payload.get("completed_at_utc"),
        "latest_available_data_utc": data_coverage.get("global_latest_utc"),
        "requested_symbol_count": data_coverage.get("requested_symbol_count")
        or universe.get("requested_symbol_count"),
        "loaded_symbol_count": data_coverage.get("loaded_symbol_count")
        or universe.get("loaded_symbol_count"),
        "fold_count": len(payload.get("folds") or []),
        "top_clean_rankings": top_clean,
        "top_hard_stop_promotable_rankings": top_promotable,
        "baseline_rows": baseline_rows,
        "interpretation": [
            "Fresh 69-symbol leaf-only shadow rebuild does not reproduce the old relaxed_efficiency 156.03% headline.",
            "Current top clean rows are dynamic-switch variants, but they fail hard-stop promotion because drawdown/risk improvement gates are not met.",
            "fixed_relaxed_dynamic_blend remains absent because current no-nested policy disables the portfolio-level fixed blend family.",
        ],
    }


def _live_preflight_summary(loaded_sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = loaded_sources.get("live_readiness_preflight_20260613")
    if not isinstance(payload, Mapping):
        return {"exists": False}
    checks = payload.get("checks") if isinstance(payload.get("checks"), Mapping) else {}
    status = payload.get("status") if isinstance(payload.get("status"), Mapping) else {}
    failed_keys = [
        key
        for key in (
            "ready_for_paper",
            "ready_for_shadow",
            "ready_for_testnet",
            "ready_for_canary",
            "ready_for_real",
            "ready_for_full",
        )
        if status.get(key) is False
    ]
    blocking_checks = [
        key
        for key in (
            "decision_allows_live_start",
            "decision_runtime_compatible",
            "refresh_is_stale",
            "shadow_parity_satisfied",
            "real_mode",
            "testnet",
        )
        if checks.get(key) in (False, True) and key != "testnet"
    ]
    return {
        "exists": True,
        "generated_at": payload.get("generated_at"),
        "recommended_action": payload.get("recommended_action"),
        "status": dict(status),
        "key_checks": {
            key: checks.get(key)
            for key in (
                "decision_allows_live_start",
                "decision_runtime_compatible",
                "refresh_is_stale",
                "shadow_parity_satisfied",
                "paper_mode",
                "testnet",
                "real_mode",
                "postgres_dsn_present",
                "artifact_real_money_veto",
                "artifact_post_oos_research_variant",
                "artifact_requires_fresh_forward_shadow",
                "artifact_clean_promotion_eligible",
            )
            if key in checks
        },
        "failed_readiness_flags": failed_keys,
        "blocking_check_flags": blocking_checks,
    }


def build_payload(
    *,
    source_paths: Mapping[str, Path] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    source_paths = dict(source_paths or DEFAULT_SOURCE_PATHS)
    loaded_sources: dict[str, Mapping[str, Any]] = {}
    missing_sources: list[str] = []
    for name, path in source_paths.items():
        if path.suffix.lower() != ".json":
            continue
        if not path.exists():
            missing_sources.append(name)
            continue
        loaded_sources[name] = _load_json(path)

    candidate_assessments: list[dict[str, Any]] = []
    for label in BASELINE_LABELS:
        evidence: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for source_name, payload in loaded_sources.items():
            rows = find_candidate_rows(payload, label)
            if rows:
                evidence[source_name] = rows
        candidate_assessments.append(_assessment(label, evidence))
    freeze_symbols = _source_universe_symbols(loaded_sources)

    return {
        "artifact_kind": "alpha_zoo_69_asset_strategy_baseline_readiness",
        "generated_at_utc": generated_at_utc or _utc_now_iso(),
        "baseline_labels": list(BASELINE_LABELS),
        "freeze_universe": {
            "source": "exact_blend_20260603",
            "symbol_count": len(freeze_symbols),
            "symbols": freeze_symbols,
        },
        "source_manifest": source_manifest(source_paths),
        "missing_json_sources": missing_sources,
        "global_verdict": {
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "paper_or_shadow_only": True,
            "decision": "block_real_money_until_fresh_forward_and_execution_telemetry",
        },
        "candidate_assessments": candidate_assessments,
        "latest_leaf_rebuild_shadow": _leaf_rebuild_summary(loaded_sources),
        "live_readiness_preflight": _live_preflight_summary(loaded_sources),
        "required_gates": [
            {
                "gate": "freeze",
                "requirement": "Freeze exact artifact hash, candidate family set, weights, thresholds, and selection rules before new evidence is observed.",
                "pass_condition": "No threshold/family/weight edits after freeze; generated manifest sha256 remains unchanged.",
            },
            {
                "gate": "leaf_only_rebuild_for_60_40",
                "requirement": "Rebuild the 60/40 idea directly from leaf sleeves; do not blend portfolio-level hybrid rows.",
                "pass_condition": "nested_hybrid_dependency=false and post_oos_research_variant=false before fresh-forward shadow.",
            },
            {
                "gate": "fresh_forward_shadow",
                "requirement": "Run at least 1-2 genuinely new monthly folds; prefer 4 folds before real-sleeve discussion.",
                "pass_condition": "Frozen strategy remains positive with acceptable MDD without changing rules.",
            },
            {
                "gate": "execution_cost",
                "requirement": "Replace 10bps proxy with paper/testnet BBO/fill telemetry.",
                "pass_condition": "Mean all-in round-trip cost <=10bps, p95 <=15bps, no unexplained reconciliation gaps.",
            },
            {
                "gate": "stress_cost",
                "requirement": "Run 10/15/20bps stress and liquidation-inclusive MDD checks.",
                "pass_condition": "10bps and 15bps remain positive; 20bps does not reveal tail/MDD collapse.",
            },
            {
                "gate": "canary_artifact",
                "requirement": "Generate a promoted decision artifact with canary/real flags only after all prior gates pass.",
                "pass_condition": "Artifact veto removed intentionally, with operator kill-switch and flatten runbook verified.",
            },
        ],
        "recommended_next_commands": [
            "uv run python scripts/research/write_alpha_zoo_69_asset_strategy_baseline_readiness.py",
            _leaf_rebuild_command(freeze_symbols),
            "uv run python scripts/ops/live_readiness_preflight.py",
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 69-asset strategy baseline readiness / freeze audit",
        "",
        f"- generated: `{payload['generated_at_utc']}`",
        "- Real-money: **blocked** (`ready_for_real=false`, `real_money_execution=false`).",
        "- Scope: user-designated three-strategy baseline set.",
        "",
        "## Baseline verdict",
        "",
        "| Strategy | Status | OOS comp | Max OOS MDD | Clean | Evidence source |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for assessment in payload["candidate_assessments"]:
        primary = dict(assessment.get("primary_metrics") or {})
        lines.append(
            "| `{label}` | `{status}` | {comp} | {mdd} | `{clean}` | `{source}` |".format(
                label=assessment["label"],
                status=assessment["status"],
                comp=_pct(primary.get("compounded_oos_return")),
                mdd=_pct(primary.get("max_oos_mdd")),
                clean=primary.get("clean_promotion_eligible", "n/a"),
                source=primary.get("_source", "n/a"),
            )
        )

    lines.extend(["", "## Blockers by strategy", ""])
    for assessment in payload["candidate_assessments"]:
        lines.append(f"### `{assessment['label']}`")
        lines.append("")
        lines.append(f"- status: `{assessment['status']}`")
        for blocker in assessment["blockers"]:
            lines.append(f"- blocker: {blocker}")
        lines.append("")

    leaf = payload.get("latest_leaf_rebuild_shadow")
    if isinstance(leaf, Mapping) and leaf.get("exists"):
        lines.extend(
            [
                "## Latest 69-symbol leaf-only shadow rebuild",
                "",
                f"- generated: `{leaf.get('generated_at_utc')}`",
                f"- latest data: `{leaf.get('latest_available_data_utc')}`",
                f"- symbols: requested `{leaf.get('requested_symbol_count')}`, loaded `{leaf.get('loaded_symbol_count')}`",
                f"- folds: `{leaf.get('fold_count')}`",
                "",
                "### Designated-baseline rows in latest leaf rebuild",
                "",
                "| Strategy | OOS comp | Max OOS MDD | Hit folds | Latest OOS | Clean | Hard-stop promotable |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        baseline_rows = (
            leaf.get("baseline_rows") if isinstance(leaf.get("baseline_rows"), Mapping) else {}
        )
        for label in BASELINE_LABELS:
            row = baseline_rows.get(label) if isinstance(baseline_rows, Mapping) else None
            if not isinstance(row, Mapping):
                lines.append(f"| `{label}` | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                "| `{label}` | {comp} | {mdd} | `{hits}` | {latest} | `{clean}` | `{promotable}` |".format(
                    label=label,
                    comp=_pct(row.get("compounded_oos_return")),
                    mdd=_pct(row.get("max_oos_mdd")),
                    hits=row.get("positive_oos_folds", "n/a"),
                    latest=_pct(row.get("latest_oos_return")),
                    clean=row.get("clean_promotion_eligible", "n/a"),
                    promotable=row.get("hard_stop_promotable", "n/a"),
                )
            )

        lines.extend(
            [
                "",
                "### Top clean rows in latest leaf rebuild",
                "",
                "| Rank | Strategy | OOS comp | Max OOS MDD | Hit folds | Latest OOS | Hard-stop promotable |",
                "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        top_rows = (
            leaf.get("top_clean_rankings")
            if isinstance(leaf.get("top_clean_rankings"), list)
            else []
        )
        for rank, row in enumerate(top_rows, start=1):
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {rank} | `{label}` | {comp} | {mdd} | `{hits}` | {latest} | `{promotable}` |".format(
                    rank=rank,
                    label=row.get("candidate_label", "n/a"),
                    comp=_pct(row.get("compounded_oos_return")),
                    mdd=_pct(row.get("max_oos_mdd")),
                    hits=row.get("positive_oos_folds", "n/a"),
                    latest=_pct(row.get("latest_oos_return")),
                    promotable=row.get("hard_stop_promotable", "n/a"),
                )
            )
        lines.append("")
        for item in leaf.get("interpretation", []):
            lines.append(f"- interpretation: {item}")
        lines.append("")

    preflight = payload.get("live_readiness_preflight")
    if isinstance(preflight, Mapping) and preflight.get("exists"):
        lines.extend(
            [
                "## Live-readiness preflight",
                "",
                f"- generated: `{preflight.get('generated_at')}`",
                f"- recommended action: `{preflight.get('recommended_action')}`",
                f"- failed readiness flags: `{', '.join(preflight.get('failed_readiness_flags') or [])}`",
                f"- key checks: `{json.dumps(preflight.get('key_checks') or {}, sort_keys=True)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Required gates before real-money review",
            "",
            "| Gate | Requirement | Pass condition |",
            "| --- | --- | --- |",
        ]
    )
    for gate in payload["required_gates"]:
        lines.append(f"| `{gate['gate']}` | {gate['requirement']} | {gate['pass_condition']} |")

    lines.extend(["", "## Source artifact hashes", ""])
    for name, info in payload["source_manifest"].items():
        exists = "Y" if info["exists"] else "N"
        lines.append(
            f"- `{name}` exists={exists} sha256=`{info['sha256'] or 'missing'}` path=`{info['path']}`"
        )

    lines.extend(["", "## Next safe local commands", ""])
    for command in payload["recommended_next_commands"]:
        lines.append(f"```bash\n{command}\n```")
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print a compact verdict instead of writing artifact files.",
    )
    args = parser.parse_args(argv)

    payload = build_payload()
    if args.check_only:
        print(
            json.dumps(
                {
                    "generated_at_utc": payload["generated_at_utc"],
                    "decision": payload["global_verdict"]["decision"],
                    "ready_for_real": payload["global_verdict"]["ready_for_real"],
                    "strategies": [
                        {
                            "label": item["label"],
                            "status": item["status"],
                            "primary_source": item["primary_metrics"].get("_source"),
                            "oos_comp": item["primary_metrics"].get("compounded_oos_return"),
                        }
                        for item in payload["candidate_assessments"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    write_outputs(payload, output_json=Path(args.output_json), output_md=Path(args.output_md))
    print(
        json.dumps({"output_json": args.output_json, "output_md": args.output_md}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
