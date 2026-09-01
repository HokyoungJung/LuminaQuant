#!/usr/bin/env python3
"""Write the real-money path/preflight artifact for TradFi external-alpha research.

This report is deliberately conservative. It reads the latest walk-forward summary
and live-readiness preflight, then records whether any strategy may start shadow,
paper/testnet, canary, or real-money execution. A missing or adverse input fails
closed and leaves every trading mode disabled.
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
DEFAULT_REPORT_ROOT = (
    ALPHA_V2_ROOT / "tradfi_external_alpha_search_20260613" / "wf_110_asset_external_v1"
)
DEFAULT_SUMMARY_JSON = DEFAULT_REPORT_ROOT / "tradfi_external_alpha_improvement_summary_latest.json"
DEFAULT_WF_JSON = DEFAULT_REPORT_ROOT / "tradfi_external_alpha_wf_110_asset_external_v1.json"
DEFAULT_LIVE_PREFLIGHT_JSON = DEFAULT_REPORT_ROOT / "live_readiness_preflight_latest.json"
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_ROOT / "real_money_path_preflight_latest.json"
DEFAULT_OUTPUT_MD = DEFAULT_REPORT_ROOT / "real_money_path_preflight_latest.md"


_BLOCKED_STATUSES = {
    "blocked",
    "blocked_fail_closed",
    "needs_redesign",
    "needs_fresh_forward",
    "needs_execution_telemetry",
}


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_input(path: Path) -> dict[str, Any]:
    entry = _source_entry(path)
    if not entry["exists"]:
        return {
            **entry,
            "valid": False,
            "error": "missing",
            "payload": None,
        }
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            **entry,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "payload": None,
        }
    if not isinstance(payload, dict):
        return {
            **entry,
            "valid": False,
            "error": f"expected JSON object, got {type(payload).__name__}",
            "payload": None,
        }
    return {
        **entry,
        "valid": True,
        "error": "",
        "payload": payload,
    }


def _source_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    exists = path.exists() and path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": _sha256(path),
        "size_bytes": resolved.stat().st_size if exists else None,
    }


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if number == number and abs(number) != float("inf") else None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100.0:.2f}%"


def _row(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return dict(payload) if isinstance(payload, Mapping) else None


def _summary_rows(summary: Mapping[str, Any]) -> dict[str, Any]:
    clean_section = summary.get("clean_section")
    new_section = summary.get("new_external_family_section")
    moonshot_section = summary.get("moonshot_shadow_section")
    baseline_sections = summary.get("baseline_sections")
    return {
        "best_clean": _row(
            clean_section.get("best_clean") if isinstance(clean_section, Mapping) else None
        ),
        "best_clean_under_15pct_mdd": _row(
            clean_section.get("best_clean_under_15pct_mdd")
            if isinstance(clean_section, Mapping)
            else None
        ),
        "best_new_clean_external_family": _row(
            new_section.get("best_new_clean") if isinstance(new_section, Mapping) else None
        ),
        "best_new_diagnostic_external_family": _row(
            new_section.get("best_new_diagnostic") if isinstance(new_section, Mapping) else None
        ),
        "best_demoted_shadow_or_post_oos": _row(
            moonshot_section.get("best_demoted_shadow_or_post_oos")
            if isinstance(moonshot_section, Mapping)
            else None
        ),
        "baseline_sections": dict(baseline_sections)
        if isinstance(baseline_sections, Mapping)
        else {},
    }


def _gate(
    gate: str, status: str, detail: str, evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": status,
        "passed": status == "passed",
        "detail": detail,
        "evidence": dict(evidence or {}),
    }


def _live_status(live_preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    if live_preflight is None:
        return {
            "exists": False,
            "recommended_action": "block_until_preflight_gaps_closed",
            "status": {
                "ready_for_paper": False,
                "ready_for_shadow": False,
                "ready_for_canary": False,
                "ready_for_real": False,
                "ready_for_testnet": False,
                "ready_for_full": False,
            },
            "checks": {},
        }
    status = live_preflight.get("status")
    checks = live_preflight.get("checks")
    return {
        "exists": True,
        "generated_at": live_preflight.get("generated_at"),
        "recommended_action": live_preflight.get("recommended_action"),
        "status": dict(status) if isinstance(status, Mapping) else {},
        "checks": dict(checks) if isinstance(checks, Mapping) else {},
        "latest": dict(live_preflight.get("latest") or {})
        if isinstance(live_preflight.get("latest"), Mapping)
        else {},
    }


def _build_gates(
    *,
    source_status: Mapping[str, Any],
    summary: Mapping[str, Any],
    live: Mapping[str, Any],
    rows: Mapping[str, Any],
) -> list[dict[str, Any]]:
    decision = summary.get("decision") if isinstance(summary.get("decision"), Mapping) else {}
    run_coverage = (
        summary.get("run_coverage") if isinstance(summary.get("run_coverage"), Mapping) else {}
    )
    checks = (
        summary.get("schema_checks") if isinstance(summary.get("schema_checks"), Mapping) else {}
    )
    best_clean = rows.get("best_clean") if isinstance(rows.get("best_clean"), Mapping) else {}
    best_new = (
        rows.get("best_new_clean_external_family")
        if isinstance(rows.get("best_new_clean_external_family"), Mapping)
        else {}
    )
    live_status = live.get("status") if isinstance(live.get("status"), Mapping) else {}
    live_checks = live.get("checks") if isinstance(live.get("checks"), Mapping) else {}
    telemetry = (
        summary.get("execution_telemetry")
        if isinstance(summary.get("execution_telemetry"), Mapping)
        else {}
    )
    unexplained_gaps = _safe_int(telemetry.get("unexplained_reconciliation_gaps"))
    telemetry_passed = bool(
        telemetry.get("paper_testnet_telemetry_passed") is True
        and _safe_float(telemetry.get("mean_round_trip_cost_bps")) is not None
        and _safe_float(telemetry.get("mean_round_trip_cost_bps")) <= 10.0
        and _safe_float(telemetry.get("p95_round_trip_cost_bps")) is not None
        and _safe_float(telemetry.get("p95_round_trip_cost_bps")) <= 15.0
        and unexplained_gaps == 0
    )

    source_gates = [
        _gate(
            "summary_source_valid",
            "passed"
            if (source_status.get("summary_json") or {}).get("valid") is True
            else "blocked_fail_closed",
            "Summary artifact must exist and parse as a JSON object; otherwise write a fresh disabled artifact.",
            source_status.get("summary_json")
            if isinstance(source_status.get("summary_json"), Mapping)
            else {},
        ),
        _gate(
            "wf_source_valid",
            "passed"
            if (source_status.get("wf_json") or {}).get("valid") is True
            else "blocked_fail_closed",
            "Walk-forward artifact must exist and parse as a JSON object for provenance.",
            source_status.get("wf_json")
            if isinstance(source_status.get("wf_json"), Mapping)
            else {},
        ),
        _gate(
            "live_preflight_source_valid",
            "passed"
            if (source_status.get("live_preflight_json") or {}).get("valid") is True
            else "blocked_fail_closed",
            "Live-readiness preflight must exist and parse as a JSON object; otherwise fail closed.",
            source_status.get("live_preflight_json")
            if isinstance(source_status.get("live_preflight_json"), Mapping)
            else {},
        ),
    ]

    promotion_gates = [
        _gate(
            "latest_data_refresh",
            "passed" if run_coverage.get("data_latest_utc") else "blocked_fail_closed",
            "WF summary must identify the latest data timestamp.",
            {"data_latest_utc": run_coverage.get("data_latest_utc")},
        ),
        _gate(
            "walk_forward_report_schema",
            "passed"
            if checks.get("has_clean_section")
            and checks.get("has_demoted_section")
            and checks.get("cost_stress_bps") == [10, 15, 20]
            else "blocked_fail_closed",
            "Report must contain clean/demoted sections and 10/15/20bps cost stress schema.",
            checks,
        ),
        _gate(
            "clean_best_hard_stop_promotion",
            "passed" if best_clean.get("hard_stop_promotable") is True else "blocked",
            "Best clean candidate must pass hard-stop promotion before any paper/live start.",
            {
                "candidate_label": best_clean.get("candidate_label"),
                "hard_stop_promotable": best_clean.get("hard_stop_promotable"),
                "oos_comp": best_clean.get("compounded_oos_return"),
                "max_oos_mdd": best_clean.get("max_oos_mdd"),
            },
        ),
        _gate(
            "new_external_family_improvement",
            "passed"
            if decision.get("new_external_family_improved_clean_best") is True
            else "needs_redesign",
            "New TradFi/external-data families must improve the current clean aggregate.",
            {
                "new_external_family_improved_clean_best": decision.get(
                    "new_external_family_improved_clean_best"
                ),
                "best_new_candidate_label": best_new.get("candidate_label"),
                "best_new_oos_comp": best_new.get("compounded_oos_return"),
            },
        ),
        _gate(
            "summary_allows_shadow_or_paper",
            "passed" if decision.get("paper_or_shadow_start_allowed") is True else "blocked",
            "Research summary must explicitly allow shadow/paper before any execution wrapper starts.",
            {"paper_or_shadow_start_allowed": decision.get("paper_or_shadow_start_allowed")},
        ),
        _gate(
            "summary_allows_real_money",
            "passed" if decision.get("real_money_ready") is True else "blocked",
            "Research summary must explicitly allow real-money before canary/full review.",
            {"real_money_ready": decision.get("real_money_ready")},
        ),
        _gate(
            "live_preflight_paper_or_shadow",
            "passed"
            if live_status.get("ready_for_paper") is True
            or live_status.get("ready_for_shadow") is True
            else "blocked",
            "Live preflight must pass paper/testnet or shadow entry checks.",
            {
                "ready_for_paper": live_status.get("ready_for_paper"),
                "ready_for_shadow": live_status.get("ready_for_shadow"),
                "recommended_action": live.get("recommended_action"),
                "refresh_is_stale": live_checks.get("refresh_is_stale"),
                "decision_allows_live_start": live_checks.get("decision_allows_live_start"),
            },
        ),
        _gate(
            "live_preflight_real_money",
            "passed" if live_status.get("ready_for_real") is True else "blocked",
            "Live preflight must pass real/full entry checks before any capital is enabled.",
            {
                "ready_for_real": live_status.get("ready_for_real"),
                "ready_for_full": live_status.get("ready_for_full"),
                "artifact_real_money_veto": live_checks.get("artifact_real_money_veto"),
                "real_mode": live_checks.get("real_mode"),
                "testnet": live_checks.get("testnet"),
            },
        ),
        _gate(
            "execution_telemetry",
            "passed" if telemetry_passed else "needs_execution_telemetry",
            "Paper/testnet BBO, fill, cancel, partial-fill, slippage, and reconciliation telemetry is required before promotion.",
            dict(telemetry) if telemetry else {"telemetry_source": None},
        ),
    ]
    return source_gates + promotion_gates


def _path_stage(
    stage: int,
    name: str,
    status: str,
    entry_requirements: list[str],
    exit_requirements: list[str],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "name": name,
        "status": status,
        "entry_requirements": entry_requirements,
        "exit_requirements": exit_requirements,
    }


def _status_by_gate(gates: list[dict[str, Any]]) -> dict[str, str]:
    return {str(gate["gate"]): str(gate["status"]) for gate in gates}


def _all_passed(status_by_gate: Mapping[str, str], gate_names: set[str]) -> bool:
    return all(status_by_gate.get(gate_name) == "passed" for gate_name in gate_names)


def _global_verdict_and_stage_statuses(
    *,
    gates: list[dict[str, Any]],
    live: Mapping[str, Any],
    decision_reason: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    status_by_gate = _status_by_gate(gates)
    live_status = live.get("status") if isinstance(live.get("status"), Mapping) else {}
    research_gate_names = {
        "summary_source_valid",
        "wf_source_valid",
        "live_preflight_source_valid",
        "latest_data_refresh",
        "walk_forward_report_schema",
        "clean_best_hard_stop_promotion",
        "new_external_family_improvement",
    }
    paper_research_gate_names = research_gate_names | {"summary_allows_shadow_or_paper"}
    real_gate_names = research_gate_names | {
        "summary_allows_real_money",
        "live_preflight_real_money",
        "execution_telemetry",
    }
    paper_start = bool(
        _all_passed(status_by_gate, paper_research_gate_names)
        and live_status.get("ready_for_paper") is True
    )
    shadow_start = bool(
        _all_passed(status_by_gate, paper_research_gate_names)
        and live_status.get("ready_for_shadow") is True
    )
    real_start = _all_passed(status_by_gate, real_gate_names)
    allowed_start_modes = [
        mode
        for mode, allowed in (
            ("paper", paper_start),
            ("shadow", shadow_start),
            ("real", real_start),
        )
        if allowed
    ]
    if real_start:
        verdict_decision = "real_money_allowed_after_all_gates_passed"
    elif paper_start or shadow_start:
        verdict_decision = "paper_or_shadow_allowed_after_research_and_live_preflight_gates"
    else:
        verdict_decision = "block_all_execution_until_redesign_fresh_forward_and_telemetry"

    stage_statuses = {
        "research_block": "cleared" if allowed_start_modes else "current_state",
        "freeze_candidate": "complete"
        if _all_passed(status_by_gate, research_gate_names)
        else "not_started",
        "fresh_forward_shadow": "allowed" if shadow_start else "blocked",
        "paper_or_testnet_execution": "allowed" if paper_start else "blocked",
        "canary_real_money": "allowed" if real_start else "blocked",
    }
    return (
        {
            "ready_for_real": real_start,
            "real_money_execution": real_start,
            "real_execution_allowed": real_start,
            "paper_trading_start": paper_start,
            "shadow_trading_start": shadow_start,
            "allowed_start_modes": allowed_start_modes,
            "decision": verdict_decision,
            "reason": decision_reason,
        },
        stage_statuses,
    )


def build_payload(
    *,
    summary_json: Path = DEFAULT_SUMMARY_JSON,
    wf_json: Path = DEFAULT_WF_JSON,
    live_preflight_json: Path = DEFAULT_LIVE_PREFLIGHT_JSON,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    summary_input = _load_input(summary_json)
    wf_input = _load_input(wf_json)
    live_input = _load_input(live_preflight_json)
    source_status = {
        "summary_json": {key: value for key, value in summary_input.items() if key != "payload"},
        "wf_json": {key: value for key, value in wf_input.items() if key != "payload"},
        "live_preflight_json": {
            key: value for key, value in live_input.items() if key != "payload"
        },
    }
    summary = summary_input["payload"] if isinstance(summary_input.get("payload"), Mapping) else {}
    live_preflight = (
        live_input["payload"] if isinstance(live_input.get("payload"), Mapping) else None
    )
    rows = _summary_rows(summary)
    live = _live_status(live_preflight)
    gates = _build_gates(source_status=source_status, summary=summary, live=live, rows=rows)
    blocking_gates = [gate for gate in gates if gate["status"] in _BLOCKED_STATUSES]
    decision = summary.get("decision") if isinstance(summary.get("decision"), Mapping) else {}
    global_verdict, stage_statuses = _global_verdict_and_stage_statuses(
        gates=gates,
        live=live,
        decision_reason=decision.get("main_reason")
        or "WF/preflight gates did not support promotion.",
    )

    return {
        "artifact_kind": "tradfi_external_alpha_real_money_path_preflight",
        "generated_at_utc": generated_at_utc or _utc_now_iso(),
        "source_manifest": source_status,
        "research_decision": dict(decision),
        "run_coverage": dict(summary.get("run_coverage") or {})
        if isinstance(summary.get("run_coverage"), Mapping)
        else {},
        "key_results": rows,
        "live_readiness_preflight": live,
        "external_evidence_urls": dict(summary.get("external_evidence_urls") or {})
        if isinstance(summary.get("external_evidence_urls"), Mapping)
        else {},
        "promotion_gates": gates,
        "blocking_gates": [gate["gate"] for gate in blocking_gates],
        "global_verdict": global_verdict,
        "real_money_path": [
            _path_stage(
                0,
                "research_block",
                stage_statuses["research_block"],
                ["Keep all real/paper/shadow execution disabled."],
                ["Select a redesigned candidate that beats clean best without OOS leakage."],
            ),
            _path_stage(
                1,
                "freeze_candidate",
                stage_statuses["freeze_candidate"],
                [
                    "Freeze universe, candidate manifest, thresholds, source registry, and hashes before observing new OOS.",
                    "Reject paid/API-key/broker/live-only data unless explicitly approved and audited.",
                ],
                ["Manifest hash unchanged; no family/threshold edits after freeze."],
            ),
            _path_stage(
                2,
                "fresh_forward_shadow",
                stage_statuses["fresh_forward_shadow"],
                [
                    "Run genuinely new monthly folds after the freeze with no retuning.",
                    "Prefer at least four folds before any real-sleeve discussion.",
                ],
                [
                    "Positive net return under 10/15bps stress.",
                    "No tail/MDD collapse under 20bps stress.",
                    "No hidden post-OOS/non-clean flags.",
                ],
            ),
            _path_stage(
                3,
                "paper_or_testnet_execution",
                stage_statuses["paper_or_testnet_execution"],
                ["Live preflight ready_for_paper or ready_for_shadow must pass."],
                [
                    "Mean all-in round-trip cost <=10bps and p95 <=15bps.",
                    "BBO/fill/cancel/partial-fill/reconciliation telemetry has no unexplained gaps.",
                ],
            ),
            _path_stage(
                4,
                "canary_real_money",
                stage_statuses["canary_real_money"],
                [
                    "Research real_money_ready=true.",
                    "Live preflight ready_for_canary/ready_for_real=true.",
                    "Operator kill-switch, flatten runbook, and monitoring reviewed.",
                ],
                ["Canary passes before any full-size capital discussion."],
            ),
        ],
        "recommended_next_actions": [
            "Do not promote the current new external-alpha families; use them as diagnostics/risk-control inputs only.",
            "Redesign around stronger US-equity/session-aware alpha with a frozen manifest before new evidence.",
            "Rerun fresh-forward shadow after freeze; only then consider paper/testnet execution telemetry collection.",
            "Keep real-money blocked until every promotion gate and live-readiness preflight passes.",
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    verdict = (
        payload.get("global_verdict") if isinstance(payload.get("global_verdict"), Mapping) else {}
    )
    real_status = "allowed" if verdict.get("real_money_execution") is True else "blocked"
    paper_status = "allowed" if verdict.get("paper_trading_start") is True else "blocked"
    shadow_status = "allowed" if verdict.get("shadow_trading_start") is True else "blocked"
    key_results = (
        payload.get("key_results") if isinstance(payload.get("key_results"), Mapping) else {}
    )
    rows = [
        ("Best clean", key_results.get("best_clean")),
        ("Best clean <15% MDD", key_results.get("best_clean_under_15pct_mdd")),
        ("Best new clean external", key_results.get("best_new_clean_external_family")),
        ("Best moonshot/demoted", key_results.get("best_demoted_shadow_or_post_oos")),
    ]
    lines = [
        "# TradFi external-alpha real-money path / preflight",
        "",
        f"- generated: `{payload['generated_at_utc']}`",
        f"- verdict: `{verdict.get('decision', 'block')}`",
        f"- Real-money: **{real_status}** (`real_money_execution={str(verdict.get('real_money_execution') is True).lower()}`).",
        f"- Paper start: **{paper_status}** (`paper_trading_start={str(verdict.get('paper_trading_start') is True).lower()}`).",
        f"- Shadow start: **{shadow_status}** (`shadow_trading_start={str(verdict.get('shadow_trading_start') is True).lower()}`).",
        "",
        "## Key WF rows",
        "",
        "| Row | Strategy | OOS comp | Max OOS MDD | Hit folds | Latest OOS | Hard-stop promotable |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in rows:
        if not isinstance(row, Mapping):
            lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            "| {name} | `{label}` | {comp} | {mdd} | `{hits}` | {latest} | `{promotable}` |".format(
                name=name,
                label=row.get("candidate_label", "n/a"),
                comp=_pct(row.get("compounded_oos_return")),
                mdd=_pct(row.get("max_oos_mdd")),
                hits=row.get("positive_oos_folds", "n/a"),
                latest=_pct(row.get("latest_oos_return")),
                promotable=row.get("hard_stop_promotable", "n/a"),
            )
        )

    live = payload.get("live_readiness_preflight")
    live_status = (
        live.get("status")
        if isinstance(live, Mapping) and isinstance(live.get("status"), Mapping)
        else {}
    )
    live_checks = (
        live.get("checks")
        if isinstance(live, Mapping) and isinstance(live.get("checks"), Mapping)
        else {}
    )
    lines.extend(
        [
            "",
            "## Live-readiness preflight",
            "",
            f"- recommended action: `{live.get('recommended_action') if isinstance(live, Mapping) else 'block_until_preflight_gaps_closed'}`",
            f"- ready_for_paper: `{live_status.get('ready_for_paper')}`",
            f"- ready_for_shadow: `{live_status.get('ready_for_shadow')}`",
            f"- ready_for_real: `{live_status.get('ready_for_real')}`",
            f"- refresh_is_stale: `{live_checks.get('refresh_is_stale')}`",
            f"- decision_allows_live_start: `{live_checks.get('decision_allows_live_start')}`",
            "",
            "## Promotion gates",
            "",
            "| Gate | Status | Detail |",
            "| --- | --- | --- |",
        ]
    )
    for gate in payload.get("promotion_gates", []):
        if isinstance(gate, Mapping):
            lines.append(
                f"| `{gate.get('gate')}` | `{gate.get('status')}` | {gate.get('detail')} |"
            )

    lines.extend(["", "## Required path", ""])
    for stage in payload.get("real_money_path", []):
        if not isinstance(stage, Mapping):
            continue
        lines.append(f"### {stage.get('stage')}. `{stage.get('name')}` — `{stage.get('status')}`")
        for requirement in stage.get("entry_requirements", []):
            lines.append(f"- entry: {requirement}")
        for requirement in stage.get("exit_requirements", []):
            lines.append(f"- exit: {requirement}")
        lines.append("")

    lines.extend(["## Source hashes", ""])
    source_manifest = payload.get("source_manifest")
    if isinstance(source_manifest, Mapping):
        for name, info in source_manifest.items():
            if isinstance(info, Mapping):
                exists = "Y" if info.get("exists") else "N"
                lines.append(
                    f"- `{name}` exists={exists} sha256=`{info.get('sha256') or 'missing'}` path=`{info.get('path')}`"
                )
    lines.append("")
    return "\n".join(lines)


def write_outputs(payload: Mapping[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write TradFi external-alpha real-money path/preflight artifact."
    )
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--wf-json", default=str(DEFAULT_WF_JSON))
    parser.add_argument("--live-preflight-json", default=str(DEFAULT_LIVE_PREFLIGHT_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    args = parser.parse_args(argv)

    payload = build_payload(
        summary_json=Path(args.summary_json),
        wf_json=Path(args.wf_json),
        live_preflight_json=Path(args.live_preflight_json),
    )
    write_outputs(payload, output_json=Path(args.output_json), output_md=Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
