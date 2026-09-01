"""Refresh operator-facing portfolio playbook artifacts from current evidence."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GROUP_ROOT = Path(
    "var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped"
)
DEFAULT_BASE_PLAN = (
    GROUP_ROOT
    / "portfolio_candidate_overlay_review_current"
    / "portfolio_operating_plan_latest.json"
)
DEFAULT_SWITCH_VALIDATION = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_switch_vs_strategy1_validation_latest.json"
)
DEFAULT_SWITCH_RECOMMENDATION = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "refreshed_operating_switch_current"
    / "portfolio_operating_switch_latest.json"
)
DEFAULT_BEARISH_SCAN = (
    GROUP_ROOT
    / "current_switch_validation_current"
    / "current_regime_bearish_strategy_scan_latest.json"
)
DEFAULT_HYBRID_PORTFOLIO = (
    GROUP_ROOT / "portfolio_hybrid_online_current" / "hybrid_online_portfolio_latest.json"
)
DEFAULT_PRODUCTION_GUARDED = (
    GROUP_ROOT / "portfolio_production_guarded_current" / "production_guarded_portfolio_latest.json"
)
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "portfolio_candidate_overlay_review_current"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_mode_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    raw = dict(metrics or {})
    if not raw:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    if "total_return" not in raw and "oos" in raw and isinstance(raw.get("oos"), dict):
        raw = dict(raw.get("oos") or {})
    return {
        "total_return": _safe_float(
            raw.get("total_return", raw.get("oos_total_return", raw.get("return"))),
            0.0,
        ),
        "sharpe": _safe_float(raw.get("sharpe", raw.get("oos_sharpe")), 0.0),
        "max_drawdown": _safe_float(
            raw.get("max_drawdown", raw.get("oos_max_drawdown", raw.get("mdd"))),
            0.0,
        ),
    }


def _hybrid_mode_entry(hybrid_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(hybrid_payload or {})
    refreshed = dict(
        dict((payload.get("scenarios") or {}).get("refreshed_latest_tail") or {}).get(
            "split_metrics"
        )
        or {}
    )
    readiness = dict(payload.get("readiness") or {})
    metrics = _normalize_mode_metrics(dict(refreshed.get("oos") or {}))
    return {
        "allocation": {"hybrid_online_portfolio": 1.0},
        "why": "Guarded dynamic governor over cash + diversified sleeves + tactical pair. Use only when refreshed readiness stays positive versus cash.",
        "metrics": metrics,
        "readiness": readiness,
    }


def _production_guarded_mode_entry(production_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(production_payload or {})
    metrics = _normalize_mode_metrics(
        dict((payload.get("portfolio_metrics") or {}).get("oos") or {})
    )
    readiness = {
        "active_exposure": _safe_float(payload.get("active_exposure"), 0.0),
        "cash_weight": _safe_float(payload.get("cash_weight"), 0.0),
        "carry_candidate_included": bool(payload.get("carry_candidate_included")),
    }
    return {
        "allocation": {"production_guarded_portfolio": 1.0},
        "why": "Drawdown-aware saved-stream production candidate blended from hybrid/static/incumbent sleeves.",
        "metrics": metrics,
        "readiness": readiness,
    }


def _hybrid_source_sleeve_metrics(
    hybrid_payload: dict[str, Any] | None,
    *,
    scenario: str = "refreshed_latest_tail",
) -> dict[str, Any]:
    payload = dict(hybrid_payload or {})
    scenario_payload = dict(dict(payload.get("scenarios") or {}).get(scenario) or {})
    return {
        str(name): dict(metrics or {})
        for name, metrics in dict(scenario_payload.get("source_sleeve_metrics") or {}).items()
        if str(name).strip()
    }


def build_playbook(
    *,
    base_plan: dict[str, Any],
    switch_validation: dict[str, Any],
    switch_recommendation: dict[str, Any],
    bearish_scan: dict[str, Any],
    hybrid_payload: dict[str, Any] | None = None,
    production_guarded_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_modes = dict(base_plan.get("deployment_modes") or {})
    refreshed_metrics = dict(switch_validation.get("refreshed_metrics") or {})
    hybrid_source_metrics = _hybrid_source_sleeve_metrics(hybrid_payload)
    core_metrics = _normalize_mode_metrics(
        hybrid_source_metrics.get("soft_three_way_regime")
        or refreshed_metrics.get("soft_three_way_regime")
        or refreshed_metrics.get("switch_strategy_core_soft100")
    )
    balanced_metrics = _normalize_mode_metrics(
        hybrid_source_metrics.get("balanced_overlay_80_20")
        or refreshed_metrics.get("strategy1_balanced_overlay_80_20")
        or refreshed_metrics.get("balanced_overlay_80_20")
    )
    aggressive_metrics = _normalize_mode_metrics(
        hybrid_source_metrics.get("three_way_regime")
        or refreshed_metrics.get("three_way_regime")
        or refreshed_metrics.get("aggressive_three_way")
    )
    risk_off_mode = {
        "allocation": {"cash": 1.0},
        "why": "Use when all active sleeves are unhealthy in the refreshed latest-anchored window, especially in bearish/weak-liquidity conditions.",
        "metrics": _normalize_mode_metrics(refreshed_metrics.get("risk_off_cash")),
    }
    pair_rows = list(bearish_scan.get("ranked_by_oos_return_then_sharpe") or [])
    pair_row = next(
        (
            row
            for row in pair_rows
            if str(row.get("name") or "").startswith("pair_spread_1h_exec_tightstop_tp_fastexit")
        ),
        None,
    )
    pair_tactical_mode = {
        "allocation": {"pair_fast_exit": 1.0},
        "why": "Tactical sleeve-only mode for operators who must keep active exposure in a hostile regime. Use only when accepting sparse / high-PBO behavior.",
        "metrics": _normalize_mode_metrics(
            hybrid_source_metrics.get("pair_tactical_mode")
            or dict((pair_row or {}).get("oos") or {})
        ),
    }
    hybrid_guarded_mode = _hybrid_mode_entry(hybrid_payload)
    production_guarded_mode = _production_guarded_mode_entry(production_guarded_payload)
    refreshed_modes = {
        **base_modes,
        "risk_off_mode": risk_off_mode,
        "pair_tactical_mode": pair_tactical_mode,
        "hybrid_guarded_mode": hybrid_guarded_mode,
        "production_guarded_mode": production_guarded_mode,
    }
    for key, item in list(refreshed_modes.items()):
        mode_item = dict(item or {})
        mode_item["metrics"] = _normalize_mode_metrics(mode_item.get("metrics"))
        refreshed_modes[key] = mode_item
    if "core_mode" in refreshed_modes:
        refreshed_modes["core_mode"] = {
            **dict(refreshed_modes["core_mode"] or {}),
            "metrics": core_metrics,
        }
    if "balanced_overlay_mode" in refreshed_modes:
        refreshed_modes["balanced_overlay_mode"] = {
            **dict(refreshed_modes["balanced_overlay_mode"] or {}),
            "metrics": balanced_metrics,
        }
    if "aggressive_realized_mode" in refreshed_modes:
        refreshed_modes["aggressive_realized_mode"] = {
            **dict(refreshed_modes["aggressive_realized_mode"] or {}),
            "metrics": aggressive_metrics,
        }

    current_market_state = dict(switch_recommendation.get("current_market_state") or {})
    recommended_mode = dict(switch_recommendation.get("recommended_mode") or {})
    comparison = dict(switch_validation.get("comparison_switch_vs_strategy1") or {})

    return {
        "artifact_kind": "portfolio_operating_playbook",
        "generated_at": _utc_now_iso(),
        "base_plan_path": str(DEFAULT_BASE_PLAN.resolve()),
        "switch_validation_path": str(DEFAULT_SWITCH_VALIDATION.resolve()),
        "switch_recommendation_path": str(DEFAULT_SWITCH_RECOMMENDATION.resolve()),
        "bearish_scan_path": str(DEFAULT_BEARISH_SCAN.resolve()),
        "hybrid_portfolio_path": str(DEFAULT_HYBRID_PORTFOLIO.resolve()),
        "production_guarded_path": str(DEFAULT_PRODUCTION_GUARDED.resolve()),
        "current_market_state": current_market_state,
        "recommended_mode": recommended_mode,
        "deployment_modes": refreshed_modes,
        "pair_fast_exit_single_sleeve_rules": {
            "strategy_name": "pair_spread_1h_exec_tightstop_tp_fastexit_bnbusdt_trxusdt_2.5_0.75",
            "default_role": "overlay_or_tactical_only",
            "max_weight_inside_multi_sleeve_portfolio": 0.30,
            "preferred_overlay_weight": 0.20,
            "single_sleeve_allowed_only_when": [
                "operator explicitly wants active exposure instead of cash in a hostile regime",
                "pair_liquidity_state is not weak",
                "the latest refreshed pair sleeve remains positive on OOS return and Sharpe",
            ],
            "force_disable_when": [
                "pair_liquidity_state is weak or stale",
                "the refreshed pair sleeve turns negative on OOS return or Sharpe",
                "a safer active allocator becomes healthy and beats the sleeve on refreshed validation",
            ],
            "current_refreshed_oos": dict(
                hybrid_source_metrics.get("pair_tactical_mode")
                or dict((pair_row or {}).get("oos") or {})
            ),
        },
        "hybrid_guarded_rules": {
            "portfolio_name": "hybrid_online_portfolio",
            "default_role": "guarded_dynamic_candidate",
            "allowed_only_when": [
                "hybrid readiness beats cash on refreshed/latest-tail",
                "hybrid pair cap remains respected",
                "current pair_liquidity_state is not weak",
                "operator accepts guarded/pilot deployment rather than full promotion",
            ],
            "force_disable_when": [
                "hybrid refreshed OOS turns non-positive",
                "hybrid refreshed validation turns negative",
                "pair cap or memory guard is violated",
            ],
            "readiness": dict(hybrid_guarded_mode.get("readiness") or {}),
        },
        "risk_off_rules": {
            "trigger_summary": [
                "all active sleeves unhealthy in refreshed latest-anchored validation and no guarded hybrid candidate is approved",
                "bearish or defensive market state, or a regime/return mismatch where features improved but validated sleeves remain unhealthy",
                "weak pair liquidity or weak breadth / stressed volatility",
            ],
            "current_comparison_vs_strategy1": comparison,
        },
        "operator_checklist": [
            "Refresh final validation data before re-evaluating the switch.",
            "Read refreshed operating switch recommendation.",
            "If recommended mode is hybrid_guarded_mode, treat it as a guarded/pilot candidate rather than an unbounded full promotion.",
            "If recommended mode is production_guarded_mode, treat it as a low-leverage production candidate and confirm it still beats balanced/hybrid under the latest refreshed split.",
            "If recommended mode is risk_off_mode, do not override into an active portfolio unless the operator accepts tactical risk.",
            "If forcing active exposure in a bearish regime, prefer pair_tactical_mode over hard three-way when directional allocators are unhealthy.",
            "Keep pair_fast_exit at or below 30% inside diversified portfolios unless new evidence justifies more.",
        ],
    }


def write_playbook(payload: dict[str, Any], *, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "portfolio_operating_playbook_latest.json"
    md_path = output_dir / "portfolio_operating_playbook_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    current = dict(payload.get("current_market_state") or {})
    rec = dict(payload.get("recommended_mode") or {})
    modes = dict(payload.get("deployment_modes") or {})
    pair_rules = dict(payload.get("pair_fast_exit_single_sleeve_rules") or {})
    hybrid_rules = dict(payload.get("hybrid_guarded_rules") or {})
    risk_off = dict(payload.get("risk_off_rules") or {})
    lines = [
        "# portfolio operating playbook",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        "",
        "## Current state",
        f"- favored_group: `{current.get('favored_group')}`",
        f"- confidence: `{_safe_float(current.get('confidence'), 0.0):.4f}`",
        f"- trend_state: `{current.get('trend_state')}`",
        f"- breadth_state: `{current.get('breadth_state')}`",
        f"- volatility_state: `{current.get('volatility_state')}`",
        f"- pair_liquidity_state: `{current.get('pair_liquidity_state')}`",
        "",
        "## Recommended mode now",
        f"- mode: `{rec.get('mode')}`",
        f"- allocation: `{json.dumps(rec.get('allocation') or {}, sort_keys=True)}`",
        "",
        "## Deployment modes",
    ]
    for key in [
        "risk_off_mode",
        "hybrid_guarded_mode",
        "production_guarded_mode",
        "core_mode",
        "balanced_overlay_mode",
        "defensive_overlay_mode",
        "aggressive_realized_mode",
        "pair_tactical_mode",
    ]:
        if key not in modes:
            continue
        item = dict(modes.get(key) or {})
        metrics = dict(item.get("metrics") or {})
        lines.append(
            f"- {key}: {json.dumps(item.get('allocation') or {}, sort_keys=True)} | "
            f"OOS return={_safe_float(metrics.get('total_return'), 0.0):+.4%} | "
            f"sharpe={_safe_float(metrics.get('sharpe'), 0.0):.4f} | "
            f"max_dd={_safe_float(metrics.get('max_drawdown'), 0.0):.4%}"
        )
    lines += [
        "",
        "## Pair fast-exit single-sleeve rules",
        f"- strategy: `{pair_rules.get('strategy_name')}`",
        f"- default_role: `{pair_rules.get('default_role')}`",
        f"- preferred_overlay_weight: `{_safe_float(pair_rules.get('preferred_overlay_weight'), 0.0):.0%}`",
        f"- max_weight_inside_multi_sleeve_portfolio: `{_safe_float(pair_rules.get('max_weight_inside_multi_sleeve_portfolio'), 0.0):.0%}`",
        "- single_sleeve_allowed_only_when:",
        *[f"  - {item}" for item in list(pair_rules.get("single_sleeve_allowed_only_when") or [])],
        "- force_disable_when:",
        *[f"  - {item}" for item in list(pair_rules.get("force_disable_when") or [])],
        "",
        "## Hybrid guarded rules",
        f"- portfolio: `{hybrid_rules.get('portfolio_name')}`",
        f"- default_role: `{hybrid_rules.get('default_role')}`",
        f"- readiness: `{json.dumps(hybrid_rules.get('readiness') or {}, sort_keys=True)}`",
        "- allowed_only_when:",
        *[f"  - {item}" for item in list(hybrid_rules.get("allowed_only_when") or [])],
        "- force_disable_when:",
        *[f"  - {item}" for item in list(hybrid_rules.get("force_disable_when") or [])],
        "",
        "## Risk-off rules",
        *[f"- {item}" for item in list(risk_off.get("trigger_summary") or [])],
        f"- current_switch_vs_strategy1 Δreturn: `{_safe_float((risk_off.get('current_comparison_vs_strategy1') or {}).get('oos_return_delta'), 0.0):+.4%}`",
        f"- current_switch_vs_strategy1 Δsharpe: `{_safe_float((risk_off.get('current_comparison_vs_strategy1') or {}).get('oos_sharpe_delta'), 0.0):+.4f}`",
        f"- current_switch_vs_strategy1 Δmax_dd: `{_safe_float((risk_off.get('current_comparison_vs_strategy1') or {}).get('oos_max_drawdown_delta'), 0.0):+.4%}`",
        "",
        "## Operator checklist",
        *[f"- {item}" for item in list(payload.get("operator_checklist") or [])],
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, default=DEFAULT_BASE_PLAN)
    parser.add_argument("--switch-validation", type=Path, default=DEFAULT_SWITCH_VALIDATION)
    parser.add_argument("--switch-recommendation", type=Path, default=DEFAULT_SWITCH_RECOMMENDATION)
    parser.add_argument("--bearish-scan", type=Path, default=DEFAULT_BEARISH_SCAN)
    parser.add_argument("--hybrid-portfolio", type=Path, default=DEFAULT_HYBRID_PORTFOLIO)
    parser.add_argument("--production-guarded", type=Path, default=DEFAULT_PRODUCTION_GUARDED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = build_playbook(
        base_plan=_read_json(Path(args.base_plan).resolve()),
        switch_validation=_read_json(Path(args.switch_validation).resolve()),
        switch_recommendation=_read_json(Path(args.switch_recommendation).resolve()),
        bearish_scan=_read_json(Path(args.bearish_scan).resolve()),
        hybrid_payload=_read_json(Path(args.hybrid_portfolio).resolve()),
        production_guarded_payload=_read_json(Path(args.production_guarded).resolve())
        if Path(args.production_guarded).resolve().exists()
        else None,
    )
    json_path, md_path = write_playbook(payload, output_dir=Path(args.output_dir).resolve())
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
