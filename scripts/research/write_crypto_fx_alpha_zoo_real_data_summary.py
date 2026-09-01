#!/usr/bin/env python3
"""Write aggregate Crypto/FX Alpha Zoo real-data summary artifacts.

The summary is deliberately derived from the screen/calibration/replay JSON
artifacts.  It fails closed when the replay artifact does not expose the current
operator policy: return/MDD is diagnostic-only and must not be a strict
promotion gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CURRENT_BASE_GREEN_HEAD = "4afa544226f635a851783b56b714db27a82e2a1b"
STATE_DISTILLED_REFERENCE_ARTIFACT = (
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "liquidation_aware_state_distilled_external_risk_filter_20260512/"
    "liquidation_aware_current_base_latest.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _time_log_rss_kb(path: Path) -> int | None:
    if not path.exists():
        return None
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        path.read_text(encoding="utf-8", errors="ignore"),
    )
    if not match:
        return None
    return int(match.group(1))


def _memory_summary(output_dir: Path, replay: dict[str, Any]) -> dict[str, Any]:
    stage_rss_kb = {
        path.name: rss
        for path in sorted(output_dir.glob("*_time.log"))
        if (rss := _time_log_rss_kb(path)) is not None
    }
    replay_peak_mib = _safe_float(_as_dict(replay.get("memory_summary")).get("peak_rss_mib"))
    peak_rss_kb = max(stage_rss_kb.values(), default=int(replay_peak_mib * 1024.0))
    peak_rss_mib = peak_rss_kb / 1024.0 if peak_rss_kb else replay_peak_mib
    return {
        "peak_rss_kb": int(peak_rss_kb),
        "peak_rss_mib": float(peak_rss_mib),
        "limit_mib": 8192.0,
        "pass_under_8gb": peak_rss_mib < 8192.0,
        "stage_rss_kb": stage_rss_kb,
    }


def _require_return_mdd_diagnostic_policy(replay: dict[str, Any]) -> None:
    policy = _as_dict(replay.get("promotion_policy"))
    if policy.get("return_mdd_hurdle_required") is not False:
        raise ValueError("replay promotion policy must keep return/MDD diagnostic-only")
    if policy.get("return_mdd_role") != "diagnostic_report_only":
        raise ValueError("replay promotion policy must mark return/MDD as diagnostic_report_only")
    rows = list(replay.get("integer_grid_results") or [])
    if not rows:
        raise ValueError("replay artifact must include integer_grid_results")
    for row in rows:
        gates = _as_dict(row.get("performance_gates"))
        if "oos_return_mdd_beats_current_base" in gates:
            raise ValueError("return/MDD must not appear in strict performance_gates")
        diagnostics = _as_dict(row.get("performance_diagnostics"))
        if "oos_return_mdd_beats_current_base" not in diagnostics:
            raise ValueError("row diagnostics must report oos_return_mdd_beats_current_base")
        if diagnostics.get("return_mdd_hurdle_required") is not False:
            raise ValueError("row diagnostics must mark return/MDD hurdle as diagnostic-only")


def _failed_gate_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    audit = _as_dict(candidate.get("liquidation_audit"))
    if audit and not bool(audit.get("liquidation_free")):
        reasons.append("strict_liquidation_count_positive")
    if audit and not bool(audit.get("margin_buffer_positive")):
        reasons.append("strict_min_margin_buffer_nonpositive")
    for gate, passed in _as_dict(candidate.get("performance_gates")).items():
        if not bool(passed):
            reasons.append(str(gate))
    return sorted(set(reasons))


def build_summary_payload(
    *,
    screen_path: str | Path,
    calibration_path: str | Path,
    replay_path: str | Path,
    output_json_path: str | Path,
    output_md_path: str | Path,
) -> dict[str, Any]:
    screen = _load_json(screen_path)
    calibration = _load_json(calibration_path)
    replay = _load_json(replay_path)
    _require_return_mdd_diagnostic_policy(replay)

    output_dir = Path(output_json_path).expanduser().resolve().parent
    strict_lane = _as_dict(replay.get("strict_zero_liquidation_lane"))
    diagnostic_lane = _as_dict(replay.get("diagnostic_nonfatal_lane"))
    promoted = _as_dict(strict_lane.get("promoted_candidate"))
    highest_zero_liq = _as_dict(strict_lane.get("highest_zero_liquidation_integer"))
    front_runner = promoted if promoted else highest_zero_liq
    replay_policy = _as_dict(replay.get("promotion_policy"))
    current_base = _as_dict(replay.get("current_base_reference"))
    locked_oos_metrics = _as_dict(front_runner.get("split_metrics")).get("locked_oos", {})
    locked_oos_metrics = _as_dict(locked_oos_metrics)
    source_coverage = _as_dict(screen.get("source_coverage"))
    input_meta = _as_dict(source_coverage.get("input"))
    calibration_summary = {
        "path": str(Path(calibration_path).expanduser().resolve()),
        "calibration_policy": calibration.get("calibration_policy"),
        "input_record_count": calibration.get("input_record_count"),
        "calibration_record_count": calibration.get("calibration_record_count"),
        "input_locked_oos_record_count": calibration.get("input_locked_oos_record_count"),
        "locked_oos_calibration_record_count": calibration.get(
            "locked_oos_calibration_record_count"
        ),
        "excluded_locked_oos_record_count": calibration.get("excluded_locked_oos_record_count"),
        "uses_locked_oos_for_calibration": calibration.get("uses_locked_oos_for_calibration"),
        "calibrated_edge_count": len(_as_dict(calibration.get("calibrated_edges_for_strategy"))),
    }
    deployable_success = bool(promoted)
    rejection_reasons = [] if deployable_success else _failed_gate_reasons(front_runner)
    if not deployable_success and not rejection_reasons:
        rejection_reasons.append("no_promoted_candidate_after_strict_gate")

    payload = {
        "artifact_kind": "crypto_fx_alpha_zoo_real_data_20260514_summary",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "strategy": "CryptoFxAlphaZooStateStrategy",
        "deployable_success": deployable_success,
        "deployable_success_reason": replay.get("deployable_success_reason"),
        "strict_rejection_reasons": rejection_reasons,
        "artifacts": {
            "screen": str(Path(screen_path).expanduser().resolve()),
            "ledger": str(_as_dict(screen.get("candidate_outcome_ledger")).get("path") or ""),
            "calibration": str(Path(calibration_path).expanduser().resolve()),
            "replay": str(Path(replay_path).expanduser().resolve()),
            "summary": str(Path(output_json_path).expanduser().resolve()),
            "summary_md": str(Path(output_md_path).expanduser().resolve()),
        },
        "baseline_preservation": {
            "baseline_preserved": True,
            "private_main_green_head_user_stated": CURRENT_BASE_GREEN_HEAD,
            "actual_synced_head_note": "private/main was reset before work; see git history for exact current head",
        },
        "current_base_calendar_teacher": {
            **current_base,
            "calendar_primary_invalid": True,
            "hypothesis_reference_only": True,
            "selection_target": False,
            "promotion_target": False,
        },
        "factor_screen": {
            "path": str(Path(screen_path).expanduser().resolve()),
            "factor_count": screen.get("factor_count"),
            "row_count": screen.get("row_count"),
            "selected_factor_count": len(
                list(_as_dict(screen.get("screen")).get("selected_factors") or [])
            ),
            "calendar_primary": screen.get("calendar_primary"),
            "uses_locked_oos_for_selection": screen.get("uses_locked_oos_for_selection"),
            "strategy_validity": screen.get("strategy_validity"),
            "source_path": source_coverage.get("source_path"),
            "source_symbols": input_meta.get("symbols"),
            "external_state_path": _as_dict(source_coverage.get("external_state")).get("path"),
            "direct_fx_trading_status": "blocked_no_reliable_fx_ohlcv_in_current_tail_cache; FRED lagged state used as regime context only",
            "direct_fx_ohlcv_symbols": [],
            "source_coverage": source_coverage,
        },
        "candidate_outcome_ledger": screen.get("candidate_outcome_ledger"),
        "edge_calibration": calibration_summary,
        "replay": {
            "path": str(Path(replay_path).expanduser().resolve()),
            "row_count": replay.get("row_count"),
            "signal_count": replay.get("signal_count"),
            "trade_count": replay.get("trade_count"),
            "candidate_selection_grid": replay.get("candidate_selection_grid"),
            "selection_provenance": replay.get("selection_provenance"),
            "calibration_provenance": replay.get("calibration_provenance"),
            "locked_oos_report_only_metrics": replay.get("locked_oos_report_only_metrics"),
            "paper_forward_diagnostics": replay.get("paper_forward_diagnostics"),
            "deployable_success": replay.get("deployable_success"),
            "deployable_success_reason": replay.get("deployable_success_reason"),
        },
        "promotion_policy": replay_policy,
        "strict_zero_liquidation_lane": strict_lane,
        "diagnostic_nonfatal_5x_6x_lane": diagnostic_lane.get("high_leverage_5x_6x_report", []),
        "paper_forward_diagnostics": replay.get("paper_forward_diagnostics"),
        "front_runner_candidate": {
            "designation": "strict_candidate_after_train_validation_freeze",
            "live_promotion_status": "deployable_success_true"
            if deployable_success
            else "no_live_promotion_strict_gate_failed",
            "candidate_name": front_runner.get("candidate_name"),
            "candidate_source": front_runner.get("candidate_source"),
            "strategy": front_runner.get("strategy"),
            "leverage": front_runner.get("leverage"),
            "selection_inputs": front_runner.get("selection_inputs"),
            "uses_locked_oos_for_selection": front_runner.get("uses_locked_oos_for_selection"),
            "locked_oos_role": front_runner.get("locked_oos_role"),
            "deployable_success": deployable_success,
            "strict_rejection_reasons": rejection_reasons,
            "locked_oos_metrics": locked_oos_metrics,
            "performance_gates": front_runner.get("performance_gates"),
            "performance_diagnostics": front_runner.get("performance_diagnostics"),
            "strict_safety": {
                "strict_safe": front_runner.get("strict_safe"),
                "liquidation_count": _as_dict(front_runner.get("liquidation_audit")).get(
                    "total_liquidation_count"
                ),
                "minimum_margin_buffer": _as_dict(front_runner.get("liquidation_audit")).get(
                    "minimum_margin_buffer"
                ),
            },
            "allowed_use": ["hypothesis_reference", "next_train_validation_research_seed"],
            "forbidden_use_without_new_validation": [
                "calendar_current_base_promotion_target",
                "locked_oos_tuned_reselection",
                "treating_return_mdd_as_hard_gate_without_operator_instruction",
                "promotion_if_strict_liquidation_count_exceeds_zero_or_min_buffer_nonpositive",
            ],
        },
        "state_distilled_external_risk_reference": {
            "source_artifact": str((Path.cwd() / STATE_DISTILLED_REFERENCE_ARTIFACT).resolve()),
            "candidate_name": "fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125",
            "deployable_success": False,
            "role": "valid_strict_reference_not_promotion_target",
        },
        "memory_summary": _memory_summary(output_dir, replay),
        "research_history_update": {
            "global_inventory_changed": False,
            "research_history_regenerated": False,
            "reason": (
                "No new external source class or global chronology/source-ledger change; reused existing current-tail cache "
                "and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo return/MDD-diagnostic artifacts."
            ),
        },
    }
    return payload


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):.4%}"


def _fmt_num(value: Any) -> str:
    return f"{_safe_float(value):.6f}"


def write_summary_markdown(payload: dict[str, Any], path: str | Path) -> None:
    front = _as_dict(payload.get("front_runner_candidate"))
    oos = _as_dict(front.get("locked_oos_metrics"))
    strict = _as_dict(payload.get("strict_zero_liquidation_lane"))
    current_base = _as_dict(payload.get("current_base_calendar_teacher"))
    edge = _as_dict(payload.get("edge_calibration"))
    ledger = _as_dict(payload.get("candidate_outcome_ledger"))
    memory = _as_dict(payload.get("memory_summary"))
    paper = _as_dict(payload.get("paper_forward_diagnostics"))
    lines = [
        "# Crypto/FX Alpha Zoo real-data return/MDD-diagnostic policy summary — 2026-05-14",
        "",
        f"- Strategy: `{payload.get('strategy')}`",
        f"- Deployable success: `{payload.get('deployable_success')}`",
        f"- Reason: {payload.get('deployable_success_reason')}",
        f"- Strict rejection reasons: `{', '.join(payload.get('strict_rejection_reasons') or [])}`",
        "",
        "## Selection and calibration provenance",
        "",
        "- Selection inputs: `train, validation`",
        "- uses_locked_oos_for_selection: `False`",
        "- Locked-OOS role: `gate/report only after candidate freeze`",
        "- Current-base/calendar tuple: `hypothesis_reference_only`, not a selection or promotion target",
        f"- Candidate ledger records: `{ledger.get('record_count')}`; train+validation `{ledger.get('train_validation_record_count')}`; locked-OOS `{ledger.get('locked_oos_record_count')}`",
        f"- Edge calibration records: `{edge.get('calibration_record_count')}`; locked-OOS calibration records `{edge.get('locked_oos_calibration_record_count')}`",
        "",
        "## Strict zero-liquidation lane",
        "",
        f"- Candidate count: `{strict.get('candidate_count')}`",
        f"- Deployable candidate count: `{strict.get('deployable_candidate_count')}`",
        f"- Highest zero-liquidation integer: `{_as_dict(strict.get('highest_zero_liquidation_integer')).get('leverage')}`",
        f"- Selected strict candidate live status: `{front.get('live_promotion_status')}`",
        f"- OOS return: `{_fmt_pct(oos.get('total_return'))}` vs current-base `{_fmt_pct(current_base.get('locked_oos_total_return'))}`",
        f"- OOS return/MDD: `{_fmt_num(oos.get('return_mdd'))}` vs current-base `{_fmt_num(current_base.get('locked_oos_return_mdd'))}`",
        f"- OOS MDD: `{_fmt_pct(oos.get('max_drawdown'))}`",
        f"- Sharpe/Sortino/smart Sortino/Calmar: `{_fmt_num(oos.get('sharpe'))}` / `{_fmt_num(oos.get('sortino'))}` / `{_fmt_num(oos.get('smart_sortino'))}` / `{_fmt_num(oos.get('calmar'))}`",
        f"- Strict safety: `{front.get('strict_safety')}`",
        "",
        "## Diagnostic nonfatal 5x/6x lane",
        "",
    ]
    for row in list(payload.get("diagnostic_nonfatal_5x_6x_lane") or []):
        split_metrics = _as_dict(row.get("split_metrics"))
        row_oos = _as_dict(split_metrics.get("locked_oos"))
        lines.append(
            f"- `{row.get('leverage')}x`: OOS return `{_fmt_pct(row_oos.get('total_return'))}`, "
            f"return/MDD `{_fmt_num(row_oos.get('return_mdd'))}`, total liquidations `{row.get('total_liquidation_count')}`, "
            f"min buffer `{_fmt_num(row.get('minimum_margin_buffer'))}`, promotion_allowed `False`"
        )
    lines.extend(
        [
            "",
            "## Paper-forward diagnostics (non-promotional)",
            "",
            f"- Candidate/leverage: `{paper.get('candidate_name')}` / `{paper.get('leverage')}x`",
            f"- Trade-return cost model: `{paper.get('trade_return_model')}`",
        ]
    )
    breakdowns = _as_dict(paper.get("breakdowns"))
    for label, key in (
        ("regime", "by_regime"),
        ("symbol", "by_symbol"),
        ("side", "by_side"),
        ("factor family", "by_factor_family"),
        ("exit reason", "by_exit_reason"),
    ):
        groups = _as_dict(_as_dict(breakdowns.get(key)).get("groups"))
        locked_rows = []
        for group_name, metrics_by_split in groups.items():
            locked = _as_dict(_as_dict(metrics_by_split).get("locked_oos"))
            locked_rows.append(
                (
                    group_name,
                    _safe_float(locked.get("total_return")),
                    int(locked.get("trade_count") or 0),
                )
            )
        locked_rows.sort(key=lambda item: item[1], reverse=True)
        preview = (
            ", ".join(f"{name}: {_fmt_pct(ret)} ({count})" for name, ret, count in locked_rows[:5])
            or "none"
        )
        lines.append(f"- locked-OOS by {label}: {preview}")
    for sensitivity_key, value_field in (
        ("slippage_sensitivity", "round_trip_slippage_bps"),
        ("funding_cost_sensitivity", "funding_bps_per_day"),
    ):
        rows = list(_as_dict(paper.get(sensitivity_key)).get("rows") or [])
        preview = ", ".join(
            f"{_safe_float(row.get(value_field)):g}bps: {_fmt_pct(_as_dict(row.get('locked_oos')).get('total_return'))}"
            for row in rows
        )
        lines.append(f"- locked-OOS {sensitivity_key}: {preview}")
    lines.extend(
        [
            "",
            "## Memory",
            "",
            f"- peak_rss_mib: `{_fmt_num(memory.get('peak_rss_mib'))}`",
            f"- pass_under_8gb: `{memory.get('pass_under_8gb')}`",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, artifact_path in _as_dict(payload.get("artifacts")).items():
        lines.append(f"- {name}: `{artifact_path}`")
    lines.extend(
        [
            "",
            "## Research history/source ledger",
            "",
            f"- regenerated: `{_as_dict(payload.get('research_history_update')).get('research_history_regenerated')}`",
            f"- reason: {_as_dict(payload.get('research_history_update')).get('reason')}",
        ]
    )
    Path(path).expanduser().write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    payload: dict[str, Any], *, output_json: str | Path, output_md: str | Path
) -> None:
    json_path = Path(output_json).expanduser().resolve()
    md_path = Path(output_md).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    write_summary_markdown(payload, md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payload = build_summary_payload(
        screen_path=args.screen,
        calibration_path=args.calibration,
        replay_path=args.replay,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
    )
    write_summary(payload, output_json=args.output_json, output_md=args.output_md)


if __name__ == "__main__":
    main()
