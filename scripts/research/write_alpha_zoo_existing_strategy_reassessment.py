#!/usr/bin/env python3
"""Existing strategy reassessment smoke manifest for Alpha Zoo.

This script is intentionally lightweight: it enumerates registered/runnable
strategy classes and emits an auditable smoke manifest. It does not run
backtests, tune parameters, or use locked-OOS metrics for selection. Heavy
smoke/full-WF evaluation consumes this manifest in later stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.strategies import registry  # noqa: E402
from scripts.research import run_alpha_zoo_clean_new_alpha_discovery as gate_contracts  # noqa: E402

DEFAULT_CURRENT_TOP_JSON = Path("var/reports/current_top_models/current_top_models_20260618.json")
DEFAULT_OUTPUT_DIR = Path("var/reports/strategy_research")
DEFAULT_OUTPUT_STEM = "existing_strategy_reassessment_latest"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_tuple_attr(cls: type, name: str) -> tuple[str, ...]:
    value = getattr(cls, name, ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _schema_keys(strategy_name: str) -> list[str]:
    try:
        schema = registry.get_strategy_param_schema(strategy_name)
    except Exception:
        return []
    if not isinstance(schema, Mapping):
        return []
    return sorted(str(key) for key in schema)


def _known_evidence_for_strategy(
    strategy_name: str, current_top_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    needle = strategy_name.lower().replace("strategy", "")
    evidence: list[dict[str, Any]] = []
    for row in current_top_payload.get("core_selection_set") or []:
        if not isinstance(row, Mapping):
            continue
        model = str(row.get("model") or "")
        if needle and needle in model.lower():
            evidence.append(dict(row))
    return evidence


def _audit_strategy_row(
    strategy_name: str,
    strategy_cls: type,
    *,
    current_top_payload: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = registry.get_strategy_metadata(strategy_name)
    tier = str(metadata.get("tier", "live_default"))
    required_timeframes = _safe_tuple_attr(strategy_cls, "required_timeframes")
    required_features = _safe_tuple_attr(strategy_cls, "required_features")
    audit_flags: list[str] = []
    rejection_reasons: list[str] = [
        "requires_bounded_smoke_metrics",
        "requires_full_wf_metrics",
        "fresh_forward_required_before_promotion",
    ]
    if tier == "research_only":
        audit_flags.append("research_only_tier")
        rejection_reasons.append("research_only_until_cost_realistic_wf_passes")
    elif tier == "live_opt_in":
        audit_flags.append("live_opt_in_not_default")
    elif tier == "live_default":
        audit_flags.append("live_default_registry_tier_requires_recheck_before_new_promotion")
    else:
        audit_flags.append(f"unknown_tier:{tier}")
        rejection_reasons.append("unknown_registry_tier")
    if required_features:
        audit_flags.append("requires_feature_lookup")
    if required_timeframes:
        audit_flags.append("requires_timeframe_support")
    evidence = _known_evidence_for_strategy(strategy_name, current_top_payload)
    if evidence:
        audit_flags.append("has_current_top_model_name_match")
    return {
        "strategy_name": strategy_name,
        "class_name": getattr(strategy_cls, "__name__", strategy_name),
        "module": getattr(strategy_cls, "__module__", ""),
        "tier": tier,
        "runnable_registry_entry": True,
        "required_timeframes": list(required_timeframes),
        "required_features": list(required_features),
        "param_schema_keys": _schema_keys(strategy_name),
        "current_known_evidence": evidence,
        "smoke_status": "registry_enumerated_pending_bounded_smoke",
        "full_wf_promotion_eligible": False,
        "promotion_status": "not_promoted_requires_smoke_and_full_wf",
        "audit_flags": sorted(set(audit_flags)),
        "rejection_reasons": sorted(set(rejection_reasons)),
        "ready_for_real": False,
        "real_money_execution": False,
    }


def build_reassessment_payload(
    *,
    current_top_payload: Mapping[str, Any] | None = None,
    strategy_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    strategy_map = registry.get_strategy_map()
    selected_names = list(strategy_names) if strategy_names is not None else sorted(strategy_map)
    current_top_payload = current_top_payload or {}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for strategy_name in selected_names:
        strategy_cls = strategy_map.get(strategy_name)
        if strategy_cls is None:
            skipped.append(
                {
                    "strategy_name": strategy_name,
                    "skip_reason": "not_registered_or_not_importable",
                    "full_wf_promotion_eligible": False,
                    "ready_for_real": False,
                    "real_money_execution": False,
                }
            )
            continue
        rows.append(
            _audit_strategy_row(
                strategy_name,
                strategy_cls,
                current_top_payload=current_top_payload,
            )
        )
    tier_counts: dict[str, int] = {}
    for row in rows:
        tier = str(row.get("tier") or "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    payload = {
        "artifact_kind": "alpha_zoo_existing_strategy_reassessment_smoke_manifest",
        "generated_at_utc": _utc_now_iso(),
        "selection_inputs": ["registry_metadata", "train_validation_future_smoke_only"],
        "locked_oos_policy": "current_known_oos_is_report_context_only_not_selection",
        "gate_contract": gate_contracts._gate_contract_payload(
            selection_policy=gate_contracts.DEFAULT_SELECTION_POLICY
        ),
        "strategy_count": len(rows),
        "skipped_count": len(skipped),
        "tier_counts": dict(sorted(tier_counts.items())),
        "tried_universe": {
            "requested_strategy_names": list(selected_names),
            "registered_strategy_names": sorted(strategy_map),
            "enumerated_strategy_names": [row["strategy_name"] for row in rows],
            "skipped": skipped,
        },
        "strategy_rows": rows,
        "survivor_list": [],
        "full_wf_promotion_list": [],
        "correlation_matrix_status": "not_available_until_bounded_smoke_return_streams_exist",
        "current_top_control_evidence": list(current_top_payload.get("core_selection_set") or []),
        "real_money_execution": False,
        "ready_for_real": False,
    }
    return payload


def _fmt_list(values: Sequence[Any]) -> str:
    if not values:
        return "`none`"
    return ", ".join(f"`{value}`" for value in values)


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Existing strategy reassessment smoke manifest",
        "",
        f"- generated: `{payload.get('generated_at_utc')}`",
        f"- strategy rows: `{payload.get('strategy_count', 0)}`",
        f"- skipped: `{payload.get('skipped_count', 0)}`",
        f"- tier counts: `{payload.get('tier_counts', {})}`",
        "- selection input: `registry metadata + future train/validation bounded smoke only`",
        "- locked-OOS/current top evidence: `report context only`",
        "- full-WF promotion list: `empty until bounded smoke + strict gates pass`",
        "- real-money: `false`",
        "",
        "## Strategy audit rows",
        "",
        "| Strategy | Tier | Runnable | Required TF | Required features | Promotion | Audit flags | Rejection reasons |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("strategy_rows") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| `{strategy}` | `{tier}` | {runnable} | {timeframes} | {features} | `{promotion}` | {flags} | {reasons} |".format(
                strategy=row.get("strategy_name", ""),
                tier=row.get("tier", ""),
                runnable="yes" if row.get("runnable_registry_entry") else "no",
                timeframes=_fmt_list(row.get("required_timeframes") or []),
                features=_fmt_list(row.get("required_features") or []),
                promotion=row.get("promotion_status", ""),
                flags=_fmt_list(row.get("audit_flags") or []),
                reasons=_fmt_list(row.get("rejection_reasons") or []),
            )
        )
    lines.extend(
        [
            "",
            "## Current benchmark/control evidence",
            "",
            "| Role | Model | Clean | OOS comp | MDD | Status |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("current_top_control_evidence") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| `{role}` | `{model}` | {clean} | {comp:.2f}% | {mdd:.2f}% | `{status}` |".format(
                role=row.get("role", ""),
                model=row.get("model", ""),
                clean="yes" if row.get("clean") else "no",
                comp=float(row.get("oos_comp_pct") or 0.0),
                mdd=float(row.get("max_oos_mdd_pct") or 0.0),
                status=row.get("status", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Promotion outputs",
            "",
            "- survivor list: `[]`",
            "- full-WF promotion list: `[]`",
            "- reason: `bounded smoke metrics have not been run yet`",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any], *, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-top-json", default=str(DEFAULT_CURRENT_TOP_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma-separated strategy names. Default enumerates every registered strategy.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    strategies = tuple(item.strip() for item in str(args.strategies).split(",") if item.strip())
    payload = build_reassessment_payload(
        current_top_payload=_read_json(Path(args.current_top_json)),
        strategy_names=strategies or None,
    )
    payload["output_paths"] = write_outputs(
        payload,
        output_dir=Path(args.output_dir),
        stem=str(args.output_stem),
    )
    print(json.dumps(payload["output_paths"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
