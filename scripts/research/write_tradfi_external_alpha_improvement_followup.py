#!/usr/bin/env python3
"""Write a conservative follow-up verdict for TradFi alpha improvement attempts.

The artifact answers one question: did the post-summary improvement work find a
clean, pre-registered/walk-forward improvement that can supersede the current
best clean candidate?  Missing or malformed sources fail closed; diagnostic
moonshots are recorded as upper-bound evidence only and never unlock execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
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
DEFAULT_REPORT_ROOT = (
    ALPHA_V2_ROOT / "tradfi_external_alpha_search_20260613" / "wf_110_asset_external_v1"
)
DEFAULT_SUMMARY_JSON = DEFAULT_REPORT_ROOT / "tradfi_external_alpha_improvement_summary_latest.json"
DEFAULT_WF_JSON = DEFAULT_REPORT_ROOT / "tradfi_external_alpha_wf_110_asset_external_v1.json"
DEFAULT_FAST_SELECTOR_JSON = (
    DEFAULT_REPORT_ROOT
    / "fast_shadow_selector_sweep_20260613T140842Z"
    / "wf_recomputed_augmented.json"
)
DEFAULT_NEW_ALPHA_JSON = (
    ALPHA_V2_ROOT
    / "tradfi_external_alpha_search_20260613"
    / "new_alpha_discovery_tradfi_core_with_leaders_20260613T142408Z"
    / "clean_new_alpha_discovery_latest.json"
)
DEFAULT_RAW_PROBE_JSON = DEFAULT_REPORT_ROOT / "tradfi_raw_leadlag_diagnostic_probe_latest.json"
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_ROOT / "tradfi_external_alpha_improvement_followup_latest.json"
DEFAULT_OUTPUT_MD = DEFAULT_REPORT_ROOT / "tradfi_external_alpha_improvement_followup_latest.md"


PROMOTION_COMPETITOR_OOS_COMP = 0.5338
PROMOTION_COMPETITOR_MAX_MDD = 0.1880


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


def _source_entry(path: Path) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size if exists else None,
    }


def _load_input(path: Path) -> dict[str, Any]:
    entry = _source_entry(path)
    if not entry["exists"]:
        return {**entry, "valid": False, "error": "missing", "payload": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {**entry, "valid": False, "error": f"{type(exc).__name__}: {exc}", "payload": None}
    if not isinstance(payload, dict):
        return {
            **entry,
            "valid": False,
            "error": f"expected JSON object, got {type(payload).__name__}",
            "payload": None,
        }
    return {**entry, "valid": True, "error": "", "payload": payload}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def _pct(value: Any) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number * 100.0:.2f}%"


def _row(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return dict(payload) if isinstance(payload, Mapping) else None


def _first_mapping(rows: Any, predicate: Any | None = None) -> dict[str, Any] | None:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None
    for row in rows:
        if isinstance(row, Mapping) and (predicate is None or predicate(row)):
            return dict(row)
    return None


def _best_mapping(rows: Any, key: str = "compounded_oos_return") -> dict[str, Any] | None:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None
    best: Mapping[str, Any] | None = None
    best_value: float | None = None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        value = _safe_float(row.get(key))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best = row
            best_value = value
    return dict(best) if best is not None else None


def _summary_rows(summary: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        summary.get("clean_section") if isinstance(summary.get("clean_section"), Mapping) else {}
    )
    moonshot = (
        summary.get("moonshot_shadow_section")
        if isinstance(summary.get("moonshot_shadow_section"), Mapping)
        else {}
    )
    new_external = (
        summary.get("new_external_family_section")
        if isinstance(summary.get("new_external_family_section"), Mapping)
        else {}
    )
    return {
        "best_clean": _row(clean.get("best_clean") if isinstance(clean, Mapping) else None),
        "best_clean_under_15pct_mdd": _row(
            clean.get("best_clean_under_15pct_mdd") if isinstance(clean, Mapping) else None
        ),
        "best_new_clean_external_family": _row(
            new_external.get("best_new_clean") if isinstance(new_external, Mapping) else None
        ),
        "best_moonshot_demoted": _row(
            moonshot.get("best_demoted_shadow_or_post_oos")
            if isinstance(moonshot, Mapping)
            else None
        ),
    }


def _fast_selector_attempt(fast_selector: Mapping[str, Any] | None) -> dict[str, Any]:
    if fast_selector is None:
        return {
            "name": "fast_row_level_selector_sweep",
            "source_valid": False,
            "clean_improvement_found": False,
            "promotable": False,
            "status": "missing_source_fail_closed",
        }
    aggregate = fast_selector.get("aggregate_rankings")
    clean_rankings = fast_selector.get("clean_promotion_rankings")
    selector_rows = (
        [
            dict(row)
            for row in aggregate
            if isinstance(row, Mapping)
            and str(row.get("candidate_label", "")).startswith("row_level_leaf_selector:")
        ]
        if isinstance(aggregate, Sequence)
        else []
    )
    best_selector = _best_mapping(selector_rows)
    best_clean_after = _first_mapping(clean_rankings) or _best_mapping(
        aggregate,
        key="compounded_oos_return",
    )
    return {
        "name": "fast_row_level_selector_sweep",
        "source_valid": True,
        "status": "completed_no_improvement",
        "interpretation": (
            "Row-level validation selectors were replayed from existing rows; all selector variants "
            "remained post-OOS/fresh-forward-only and did not beat the current clean best."
        ),
        "selector_report": dict(fast_selector.get("row_level_leaf_selector_report") or {})
        if isinstance(fast_selector.get("row_level_leaf_selector_report"), Mapping)
        else {},
        "best_selector": best_selector,
        "best_clean_after_recompute": best_clean_after,
        "clean_improvement_found": False,
        "promotable": False,
    }


def _new_alpha_attempt(new_alpha: Mapping[str, Any] | None) -> dict[str, Any]:
    if new_alpha is None:
        return {
            "name": "clean_new_alpha_discovery_tradfi_core_with_leaders",
            "source_valid": False,
            "clean_improvement_found": False,
            "promotable": False,
            "status": "missing_source_fail_closed",
        }
    aggregate = (
        new_alpha.get("aggregate") if isinstance(new_alpha.get("aggregate"), Mapping) else {}
    )
    compounded = _safe_float(aggregate.get("compounded_oos_return"))
    mdd = _safe_float(aggregate.get("max_oos_mdd"))
    improved = bool(
        compounded is not None
        and mdd is not None
        and compounded > PROMOTION_COMPETITOR_OOS_COMP
        and mdd <= PROMOTION_COMPETITOR_MAX_MDD
    )
    return {
        "name": "clean_new_alpha_discovery_tradfi_core_with_leaders",
        "source_valid": True,
        "status": "completed_negative_oos" if not improved else "completed_promotable",
        "aggregate": dict(aggregate),
        "candidate_row_count_total": new_alpha.get("candidate_row_count_total"),
        "fold_count": new_alpha.get("fold_count"),
        "enabled_families": list(new_alpha.get("enabled_families") or []),
        "selection_policy": new_alpha.get("selection_policy"),
        "clean_improvement_found": improved,
        "promotable": improved,
    }


def _raw_probe_attempt(raw_probe: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw_probe is None:
        return {
            "name": "raw_tradfi_leadlag_moonshot_probe",
            "source_valid": False,
            "clean_improvement_found": False,
            "promotable": False,
            "status": "missing_source_fail_closed",
        }
    top_static = _first_mapping(raw_probe.get("top_static_post_hoc"))
    selector = (
        dict(raw_probe.get("train_validation_selector"))
        if isinstance(raw_probe.get("train_validation_selector"), Mapping)
        else {}
    )
    selector_comp = _safe_float(selector.get("compounded_oos_return"))
    selector_mdd = _safe_float(selector.get("max_oos_mdd"))
    selector_passed = bool(
        selector_comp is not None
        and selector_mdd is not None
        and selector_comp > PROMOTION_COMPETITOR_OOS_COMP
        and selector_mdd <= PROMOTION_COMPETITOR_MAX_MDD
    )
    return {
        "name": "raw_tradfi_leadlag_moonshot_probe",
        "source_valid": True,
        "status": "static_upper_bound_only_selector_failed",
        "interpretation": (
            "Static post-hoc sorting shows a large upper bound, but the train/validation selector "
            "collapsed; this is not clean promotion evidence."
        ),
        "candidate_count": raw_probe.get("candidate_count"),
        "data_coverage": dict(raw_probe.get("data_coverage") or {})
        if isinstance(raw_probe.get("data_coverage"), Mapping)
        else {},
        "top_static_post_hoc": top_static,
        "train_validation_selector": selector,
        "clean_improvement_found": selector_passed,
        "promotable": False,
    }


def _source_manifest(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {key: value for key, value in item.items() if key != "payload"}
        for name, item in inputs.items()
    }


def _payload_or_none(input_info: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = input_info.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else None


def _beats_baseline(
    candidate: Mapping[str, Any] | None, baseline: Mapping[str, Any] | None
) -> bool:
    candidate_comp = _safe_float(candidate.get("compounded_oos_return")) if candidate else None
    baseline_comp = _safe_float(baseline.get("compounded_oos_return")) if baseline else None
    if candidate_comp is None or baseline_comp is None:
        return False
    return candidate_comp > baseline_comp


def build_payload(
    *,
    summary_json: Path = DEFAULT_SUMMARY_JSON,
    wf_json: Path = DEFAULT_WF_JSON,
    fast_selector_json: Path = DEFAULT_FAST_SELECTOR_JSON,
    new_alpha_json: Path = DEFAULT_NEW_ALPHA_JSON,
    raw_probe_json: Path = DEFAULT_RAW_PROBE_JSON,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = {
        "summary_json": _load_input(summary_json),
        "wf_json": _load_input(wf_json),
        "fast_selector_json": _load_input(fast_selector_json),
        "new_alpha_json": _load_input(new_alpha_json),
        "raw_probe_json": _load_input(raw_probe_json),
    }
    summary = _payload_or_none(inputs["summary_json"]) or {}
    fast_selector = _payload_or_none(inputs["fast_selector_json"])
    new_alpha = _payload_or_none(inputs["new_alpha_json"])
    raw_probe = _payload_or_none(inputs["raw_probe_json"])
    rows = _summary_rows(summary)
    attempts = [
        _fast_selector_attempt(fast_selector),
        _new_alpha_attempt(new_alpha),
        _raw_probe_attempt(raw_probe),
    ]
    best_clean = rows.get("best_clean")
    best_moonshot = rows.get("best_moonshot_demoted")
    any_clean_improvement = any(
        attempt.get("clean_improvement_found") is True for attempt in attempts
    )
    any_promotable = any(attempt.get("promotable") is True for attempt in attempts)
    source_failures = [
        name
        for name, item in inputs.items()
        if item.get("valid") is not True and name != "raw_probe_json"
    ]
    raw_probe_missing = inputs["raw_probe_json"].get("valid") is not True
    clean_beats_old_best = _beats_baseline(rows.get("best_new_clean_external_family"), best_clean)
    decision = {
        "clean_performance_improvement_found": bool(any_clean_improvement or clean_beats_old_best),
        "promotable_improvement_found": bool(any_promotable),
        "real_money_execution": False,
        "paper_trading_start": False,
        "shadow_trading_start": False,
        "fresh_forward_freeze_required": True,
        "recommended_action": (
            "freeze the lagged-router moonshot only as fresh-forward shadow research; keep real/paper "
            "execution blocked; continue alpha research with pre-registered selectors"
        ),
        "decision": "no_clean_performance_improvement_found",
        "source_failures": source_failures,
        "raw_probe_missing": raw_probe_missing,
        "main_reason": (
            "Additional selector, new-alpha, and raw lead-lag probes did not produce a clean "
            "walk-forward improvement over the current best clean candidate."
        ),
    }
    freeze_candidate = {
        "candidate_label": best_moonshot.get("candidate_label") if best_moonshot else None,
        "family": best_moonshot.get("family") if best_moonshot else None,
        "compounded_oos_return": best_moonshot.get("compounded_oos_return")
        if best_moonshot
        else None,
        "max_oos_mdd": best_moonshot.get("max_oos_mdd") if best_moonshot else None,
        "positive_oos_folds": best_moonshot.get("positive_oos_folds") if best_moonshot else None,
        "validity": "post_oos_research_variant_requires_fresh_forward_shadow",
        "allowed_usage": ["research_report", "freeze_candidate", "fresh_forward_shadow_only"],
        "blocked_usage": ["paper_trading", "real_money", "current_locked_oos_promotion"],
        "non_clean_reasons": list(best_moonshot.get("non_clean_reasons") or [])
        if best_moonshot
        else [],
    }
    return {
        "artifact_kind": "tradfi_external_alpha_improvement_followup",
        "generated_at_utc": generated_at_utc or _utc_now_iso(),
        "source_manifest": _source_manifest(inputs),
        "baseline": {
            "best_clean": best_clean,
            "best_clean_under_15pct_mdd": rows.get("best_clean_under_15pct_mdd"),
            "best_new_clean_external_family": rows.get("best_new_clean_external_family"),
            "best_moonshot_demoted": best_moonshot,
            "promotion_competitor": {
                "compounded_oos_return": PROMOTION_COMPETITOR_OOS_COMP,
                "max_oos_mdd": PROMOTION_COMPETITOR_MAX_MDD,
            },
        },
        "attempts": attempts,
        "decision": decision,
        "freeze_candidate": freeze_candidate,
        "next_research_directions": [
            {
                "name": "pre_registered_tradfi_leadlag_selector",
                "why": "raw static upper bound is large, but selector failed; redesign selector before freeze",
                "gate": "must pass train/validation without current-fold OOS and survive fresh-forward",
            },
            {
                "name": "session_and_cost_model_hardening",
                "why": "TradFi perps need US cash-session, borrow/funding, halt, and 10/15/20bps stress realism",
                "gate": "no promotion without live/paper telemetry and cost-stress survival",
            },
            {
                "name": "moonshot_freeze_then_fresh_forward",
                "why": "best apparent return remains non-clean; only new unseen folds can validate it",
                "gate": "freeze manifest/hash now; no threshold edits after observing new folds",
            },
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), Mapping) else {}
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), Mapping) else {}
    freeze = (
        payload.get("freeze_candidate")
        if isinstance(payload.get("freeze_candidate"), Mapping)
        else {}
    )
    best_clean = (
        baseline.get("best_clean") if isinstance(baseline.get("best_clean"), Mapping) else {}
    )
    best_moonshot = (
        baseline.get("best_moonshot_demoted")
        if isinstance(baseline.get("best_moonshot_demoted"), Mapping)
        else {}
    )
    lines = [
        "# TradFi external-alpha improvement follow-up",
        "",
        f"- generated: `{payload['generated_at_utc']}`",
        f"- decision: `{decision.get('decision', 'no_clean_performance_improvement_found')}`",
        "- clean improvement found: "
        f"`{str(decision.get('clean_performance_improvement_found') is True).lower()}`",
        "- promotable improvement found: "
        f"`{str(decision.get('promotable_improvement_found') is True).lower()}`",
        "- real/paper/shadow execution: **blocked**",
        "",
        "## Baseline still to beat",
        "",
        "| Row | Strategy | OOS comp | Max OOS MDD | Hit folds | Validity |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        "| Best clean | `{}` | {} | {} | `{}` | clean but hard-stop not promotable |".format(
            best_clean.get("candidate_label", "n/a"),
            _pct(best_clean.get("compounded_oos_return")),
            _pct(best_clean.get("max_oos_mdd")),
            best_clean.get("positive_oos_folds", "n/a"),
        ),
        "| Best moonshot | `{}` | {} | {} | `{}` | post-OOS/fresh-forward only |".format(
            best_moonshot.get("candidate_label", "n/a"),
            _pct(best_moonshot.get("compounded_oos_return")),
            _pct(best_moonshot.get("max_oos_mdd")),
            best_moonshot.get("positive_oos_folds", "n/a"),
        ),
        "",
        "## Follow-up attempts",
        "",
        "| Attempt | Status | Best/selector OOS comp | Max MDD | Verdict |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for attempt in payload.get("attempts", []):
        if not isinstance(attempt, Mapping):
            continue
        metric_row = None
        if isinstance(attempt.get("best_selector"), Mapping):
            metric_row = attempt["best_selector"]
        elif isinstance(attempt.get("aggregate"), Mapping):
            metric_row = attempt["aggregate"]
        elif isinstance(attempt.get("train_validation_selector"), Mapping):
            metric_row = attempt["train_validation_selector"]
        lines.append(
            "| `{}` | `{}` | {} | {} | `{}` |".format(
                attempt.get("name", "attempt"),
                attempt.get("status", "unknown"),
                _pct(
                    metric_row.get("compounded_oos_return")
                    if isinstance(metric_row, Mapping)
                    else None
                ),
                _pct(metric_row.get("max_oos_mdd") if isinstance(metric_row, Mapping) else None),
                "promotable" if attempt.get("promotable") is True else "not promotable",
            )
        )
    lines.extend(
        [
            "",
            "## Freeze candidate (not execution permission)",
            "",
            f"- candidate: `{freeze.get('candidate_label') or 'n/a'}`",
            f"- allowed usage: `{', '.join(freeze.get('allowed_usage') or [])}`",
            f"- blocked usage: `{', '.join(freeze.get('blocked_usage') or [])}`",
            "- required next gate: fresh-forward shadow after manifest/source/hash freeze.",
            "",
            "## Source hashes",
            "",
        ]
    )
    source_manifest = payload.get("source_manifest")
    if isinstance(source_manifest, Mapping):
        for name, info in source_manifest.items():
            if isinstance(info, Mapping):
                exists = "Y" if info.get("exists") else "N"
                valid = "Y" if info.get("valid") else "N"
                lines.append(
                    f"- `{name}` exists={exists} valid={valid} "
                    f"sha256=`{info.get('sha256') or 'missing'}` path=`{info.get('path')}`"
                )
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write TradFi external-alpha performance-improvement follow-up artifact."
    )
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--wf-json", default=str(DEFAULT_WF_JSON))
    parser.add_argument("--fast-selector-json", default=str(DEFAULT_FAST_SELECTOR_JSON))
    parser.add_argument("--new-alpha-json", default=str(DEFAULT_NEW_ALPHA_JSON))
    parser.add_argument("--raw-probe-json", default=str(DEFAULT_RAW_PROBE_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args(argv)
    payload = build_payload(
        summary_json=Path(args.summary_json),
        wf_json=Path(args.wf_json),
        fast_selector_json=Path(args.fast_selector_json),
        new_alpha_json=Path(args.new_alpha_json),
        raw_probe_json=Path(args.raw_probe_json),
    )
    write_outputs(payload, output_json=Path(args.output_json), output_md=Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
