#!/usr/bin/env python3
"""Write paper/shadow decision artifact for 69-asset individual-robust hybrid."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_WALKFORWARD_ARTIFACT = Path(
    "/tmp/lumina_monthly_refit_walkforward_individual_guarded_latest.json"
)
DEFAULT_LATEST_FOLD_ARTIFACT = DEFAULT_WALKFORWARD_ARTIFACT
DEFAULT_SELECTED_CANDIDATE_LABEL = "individual_robust:hybrid_v3_5"
DEFAULT_OUTPUT_DIR = (
    Path("var")
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260508"
    / "alpha_v2"
    / "alpha_zoo_69_asset_individual_robust_paper_decision_20260601"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "paper_shadow_decision_latest.json"

DEFAULT_GATE_THRESHOLDS: dict[str, float | int] = {
    "min_fold_count": 8,
    "min_compounded_oos_return": 0.0,
    "min_monthly_oos_return": -0.08,
    "max_oos_mdd": 0.16,
    "min_positive_validation_folds_ratio": 1.0,
    "min_ready_for_paper_folds": 7,
    "min_latest_oos_return": -0.02,
    "max_latest_validation_return": 0.35,
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    payload["_resolved_path"] = str(resolved)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latest_fold_id(payload: Mapping[str, Any]) -> str:
    folds = list(payload.get("folds") or [])
    if not folds:
        raise ValueError("walk-forward artifact has no folds")
    return str(folds[-1]["fold_id"])


def _fold_payload(payload: Mapping[str, Any], fold_id: str) -> dict[str, Any]:
    for fold in list(payload.get("folds") or []):
        if str(fold.get("fold_id")) == fold_id:
            return dict(fold)
    return {"fold_id": fold_id}


def _find_aggregate(payload: Mapping[str, Any], candidate_label: str) -> dict[str, Any]:
    for row in list(payload.get("aggregate_rankings") or []):
        if str(row.get("candidate_label")) == candidate_label:
            return dict(row)
    raise ValueError(f"candidate aggregate not found: {candidate_label}")


def _find_fold_candidate(
    payload: Mapping[str, Any], *, fold_id: str, candidate_label: str
) -> dict[str, Any]:
    for row in list(payload.get("fold_candidate_rows") or []):
        if (
            str(row.get("fold_id")) == fold_id
            and str(row.get("candidate_label")) == candidate_label
        ):
            return dict(row)
    raise ValueError(f"candidate row not found for {fold_id}: {candidate_label}")


def _find_individual_aux(payload: Mapping[str, Any], fold_id: str) -> dict[str, Any]:
    for summary in list(payload.get("fold_summaries") or []):
        if str(summary.get("fold_id")) == fold_id:
            return dict(summary.get("individual_robust_aux") or {})
    return {}


def _gate_check(name: str, passed: bool, actual: Any, expected: str) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _evaluate_gate(
    *,
    aggregate: Mapping[str, Any],
    latest_row: Mapping[str, Any],
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    fold_count = int(aggregate.get("fold_count") or 0)
    positive_validation_folds = int(aggregate.get("positive_validation_folds") or 0)
    ready_for_paper_folds = int(aggregate.get("ready_for_paper_folds") or 0)
    checks = [
        _gate_check(
            "candidate_family_is_individual_robust",
            str(aggregate.get("family")) == "individual_robust",
            aggregate.get("family"),
            "individual_robust",
        ),
        _gate_check(
            "fold_count",
            fold_count >= int(thresholds["min_fold_count"]),
            fold_count,
            f">= {thresholds['min_fold_count']}",
        ),
        _gate_check(
            "compounded_oos_return",
            _safe_float(aggregate.get("compounded_oos_return"))
            >= float(thresholds["min_compounded_oos_return"]),
            aggregate.get("compounded_oos_return"),
            f">= {thresholds['min_compounded_oos_return']}",
        ),
        _gate_check(
            "min_monthly_oos_return",
            _safe_float(aggregate.get("min_oos_return"))
            >= float(thresholds["min_monthly_oos_return"]),
            aggregate.get("min_oos_return"),
            f">= {thresholds['min_monthly_oos_return']}",
        ),
        _gate_check(
            "max_oos_mdd",
            _safe_float(aggregate.get("max_oos_mdd")) <= float(thresholds["max_oos_mdd"]),
            aggregate.get("max_oos_mdd"),
            f"<= {thresholds['max_oos_mdd']}",
        ),
        _gate_check(
            "positive_validation_folds",
            fold_count > 0
            and positive_validation_folds / fold_count
            >= float(thresholds["min_positive_validation_folds_ratio"]),
            f"{positive_validation_folds}/{fold_count}",
            f">= {thresholds['min_positive_validation_folds_ratio']:.0%}",
        ),
        _gate_check(
            "ready_for_paper_folds",
            ready_for_paper_folds >= int(thresholds["min_ready_for_paper_folds"]),
            f"{ready_for_paper_folds}/{fold_count}",
            f">= {thresholds['min_ready_for_paper_folds']}",
        ),
        _gate_check(
            "latest_oos_return",
            _safe_float(latest_row.get("locked_oos", {}).get("total_return"))
            >= float(thresholds["min_latest_oos_return"]),
            latest_row.get("locked_oos", {}).get("total_return"),
            f">= {thresholds['min_latest_oos_return']}",
        ),
        _gate_check(
            "latest_validation_return",
            _safe_float(latest_row.get("validation", {}).get("total_return"))
            <= float(thresholds["max_latest_validation_return"]),
            latest_row.get("validation", {}).get("total_return"),
            f"<= {thresholds['max_latest_validation_return']}",
        ),
    ]
    passed = all(bool(check["pass"]) for check in checks)
    return {
        "pass": passed,
        "decision": "keep_shadow_paper_candidate" if passed else "quarantine_shadow_candidate",
        "checks": checks,
        "thresholds": dict(thresholds),
    }


def _profile_allocations(
    *,
    final_weights: Mapping[str, Any],
    individual_aux: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    profile_rows = list(individual_aux.get("profile_rows") or [])
    for row in profile_rows:
        profile_id = str(row.get("profile_id"))
        profile_weight = _safe_float(final_weights.get(profile_id))
        profile_gross = _safe_float(row.get("gross_notional_fraction"))
        rows.append(
            {
                "profile_id": profile_id,
                "profile_weight": profile_weight,
                "profile_gross_notional_fraction": profile_gross,
                "weighted_gross_notional_fraction": profile_weight * profile_gross,
                "selected_sleeve_count": row.get("selected_sleeve_count"),
                "selection_reasons": list(row.get("selection_reasons") or []),
                "ready_for_paper": bool(row.get("ready_for_paper")),
                "train_return": row.get("train_return"),
                "validation_return": row.get("validation_return"),
                "validation_mdd": row.get("validation_mdd"),
                "asset_gross_notional_fraction": dict(
                    row.get("asset_gross_notional_fraction") or {}
                ),
            }
        )
    return sorted(rows, key=lambda item: _safe_float(item["profile_weight"]), reverse=True)


def _asset_exposure(
    *,
    final_weights: Mapping[str, Any],
    individual_aux: Mapping[str, Any],
) -> dict[str, float]:
    exposure: dict[str, float] = defaultdict(float)
    for row in list(individual_aux.get("profile_rows") or []):
        profile_id = str(row.get("profile_id"))
        profile_weight = _safe_float(final_weights.get(profile_id))
        for symbol, gross in dict(row.get("asset_gross_notional_fraction") or {}).items():
            exposure[str(symbol)] += profile_weight * _safe_float(gross)
    if exposure:
        return dict(sorted(exposure.items(), key=lambda item: item[1], reverse=True))

    for row in list(individual_aux.get("selected_sleeve_rows") or []):
        profile_weight = _safe_float(final_weights.get(str(row.get("parent_profile_id"))))
        symbol = str(row.get("symbol") or "")
        if symbol:
            exposure[symbol] += profile_weight * _safe_float(row.get("weighted_notional_fraction"))
    return dict(sorted(exposure.items(), key=lambda item: item[1], reverse=True))


def _selected_sleeves(
    *,
    final_weights: Mapping[str, Any],
    individual_aux: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sleeves = []
    for row in list(individual_aux.get("selected_sleeve_rows") or []):
        profile_id = str(row.get("parent_profile_id"))
        profile_weight = _safe_float(final_weights.get(profile_id))
        weighted_notional = profile_weight * _safe_float(row.get("weighted_notional_fraction"))
        if weighted_notional <= 1e-12:
            continue
        compact = dict(row)
        compact["final_profile_weight"] = profile_weight
        compact["final_weighted_notional_fraction"] = weighted_notional
        sleeves.append(compact)
    return sorted(
        sleeves,
        key=lambda item: _safe_float(item.get("final_weighted_notional_fraction")),
        reverse=True,
    )


def build_individual_robust_paper_decision_payload(
    *,
    walkforward_artifact_path: str | Path = DEFAULT_WALKFORWARD_ARTIFACT,
    latest_fold_artifact_path: str | Path = DEFAULT_LATEST_FOLD_ARTIFACT,
    candidate_label: str = DEFAULT_SELECTED_CANDIDATE_LABEL,
    thresholds: Mapping[str, float | int] = DEFAULT_GATE_THRESHOLDS,
) -> dict[str, Any]:
    walkforward = _load_json(walkforward_artifact_path)
    latest_detail = _load_json(latest_fold_artifact_path)
    latest_fold = _latest_fold_id(latest_detail)
    aggregate = _find_aggregate(walkforward, candidate_label)
    latest_row = _find_fold_candidate(
        latest_detail, fold_id=latest_fold, candidate_label=candidate_label
    )
    final_weights = dict(latest_row.get("final_weights") or {})
    individual_aux = _find_individual_aux(latest_detail, latest_fold)
    asset_exposure = _asset_exposure(final_weights=final_weights, individual_aux=individual_aux)
    profile_allocations = _profile_allocations(
        final_weights=final_weights, individual_aux=individual_aux
    )
    selected_sleeves = _selected_sleeves(final_weights=final_weights, individual_aux=individual_aux)
    gate = _evaluate_gate(aggregate=aggregate, latest_row=latest_row, thresholds=thresholds)
    universe = dict(walkforward.get("universe") or {})
    symbols = list(universe.get("symbols") or [])
    selected_symbols = sorted(asset_exposure) if asset_exposure else []
    decision = "paper_shadow_selected" if gate["pass"] else "paper_shadow_quarantine"
    return {
        "artifact_kind": "alpha_zoo_69_asset_individual_robust_paper_shadow_decision",
        "generated_at_utc": _utc_now_iso(),
        "decision": decision,
        "paper_testnet_only": True,
        "ready_for_paper_shadow": bool(gate["pass"]),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "selected_candidate_label": candidate_label,
        "selected_mode": "alpha_zoo_69_asset_individual_robust_hybrid_v3_5_shadow",
        "source_artifacts": {
            "walkforward_artifact": walkforward["_resolved_path"],
            "latest_fold_artifact": latest_detail["_resolved_path"],
        },
        "monitoring_universe": {
            "symbol_count": len(symbols),
            "symbols": symbols,
            "policy": "monitor_all_69_assets; trade only sleeves passing train_validation guards",
        },
        "selected_symbol_exposure": {
            "symbol_count": len(selected_symbols),
            "symbols": selected_symbols,
            "asset_gross_notional_fraction": asset_exposure,
            "gross_notional_fraction": sum(asset_exposure.values()),
            "source": "latest_fold_profile_asset_gross_times_final_hybrid_profile_weights",
            "available": bool(asset_exposure),
            "unavailable_reason": ""
            if asset_exposure
            else "rerun walk-forward artifact with individual_robust_aux profile_rows enabled",
        },
        "walkforward_gate": {
            **gate,
            "aggregate": aggregate,
            "latest_fold_id": latest_fold,
            "latest_fold_candidate": {
                "validation": latest_row.get("validation"),
                "locked_oos": latest_row.get("locked_oos"),
                "ready_for_paper": latest_row.get("ready_for_paper"),
                "selection_reasons": latest_row.get("selection_reasons") or [],
            },
        },
        "latest_fold_allocation": {
            "fold": _fold_payload(latest_detail, latest_fold),
            "candidate_label": latest_row.get("candidate_label"),
            "source_profile_id": latest_row.get("source_profile_id"),
            "final_profile_weights": final_weights,
            "profile_allocations": profile_allocations,
            "selected_sleeves": selected_sleeves,
            "selected_sleeve_count": len(selected_sleeves),
            "validation": latest_row.get("validation"),
            "locked_oos": latest_row.get("locked_oos"),
        },
        "promotion_ladder": {
            "current_stage": "paper_shadow",
            "next_allowed_stage": "testnet_or_live_small_after_forward_shadow",
            "additional_forward_oos_months_required": 2,
            "live_small_notional_fraction_after_review": "0.10x_to_0.20x",
            "automatic_real_money_promotion_allowed": False,
        },
        "stop_rules": [
            "quarantine if monthly OOS return <= -8%",
            "quarantine if live/paper drawdown exceeds 15%",
            "quarantine if two consecutive forward months are negative",
            "quarantine if validation spike guard repeatedly flags the selected hybrid",
            "real money remains blocked until separate exchange-fill telemetry review",
        ],
        "paper_testnet_validation_requirements": [
            "monthly refit on calendar day 1 using train plus previous 2 calendar months validation only",
            "all 69 assets remain in the monitor pool for future inclusion",
            "record selected sleeves, symbol exposure, realized spread, fee, slippage, rejects, cancels",
            "compare realized all-in round-trip cost against the 10bps replay assumption",
            "review this decision artifact after every monthly refit before changing allocation",
        ],
        "real_money_blockers": [
            "research_shadow_only_candidate: not approved for real-money execution",
            "forward paper/testnet telemetry is not yet sufficient for promotion",
            "exchange fill, slippage, funding, partial-fill, and reject telemetry is missing",
            "monthly OOS consistency is only 5/10 in the current walk-forward evidence",
        ],
    }


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):.2%}"


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    gate = dict(payload.get("walkforward_gate") or {})
    aggregate = dict(gate.get("aggregate") or {})
    exposure = dict(payload.get("selected_symbol_exposure") or {})
    top_exposure = list(dict(exposure.get("asset_gross_notional_fraction") or {}).items())[:15]
    lines = [
        "# 69-Asset Individual-Robust Paper/Shadow Decision",
        "",
        f"- decision: `{payload['decision']}`",
        f"- candidate: `{payload['selected_candidate_label']}`",
        f"- ready_for_paper_shadow: `{str(payload['ready_for_paper_shadow']).lower()}`",
        "- ready_for_real: `false`",
        "- real_money_execution: `false`",
        f"- walkforward OOS comp: `{_fmt_pct(aggregate.get('compounded_oos_return'))}`",
        f"- walkforward OOS pos: `{aggregate.get('positive_oos_folds')}/{aggregate.get('fold_count')}`",
        f"- min monthly OOS: `{_fmt_pct(aggregate.get('min_oos_return'))}`",
        f"- max OOS MDD: `{_fmt_pct(aggregate.get('max_oos_mdd'))}`",
        f"- monitored universe: `{payload['monitoring_universe']['symbol_count']}` assets",
        f"- selected exposure symbols: `{exposure.get('symbol_count')}`",
        "",
        "## Gate checks",
        "",
    ]
    for check in list(gate.get("checks") or []):
        marker = "PASS" if check.get("pass") else "FAIL"
        lines.append(
            f"- **{marker}** `{check.get('name')}`: actual `{check.get('actual')}`, "
            f"expected `{check.get('expected')}`"
        )
    lines.extend(["", "## Top selected symbol exposure", ""])
    if top_exposure:
        for symbol, value in top_exposure:
            lines.append(f"- `{symbol}`: `{_fmt_pct(value)}` gross notional fraction")
    else:
        lines.append(f"- unavailable: `{exposure.get('unavailable_reason')}`")
    lines.extend(
        [
            "",
            "## Stop rules",
            "",
            *[f"- {item}" for item in payload.get("stop_rules", [])],
            "",
            "This artifact is paper/shadow only and is not a real-money approval.",
            "",
        ]
    )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walkforward-artifact", default=str(DEFAULT_WALKFORWARD_ARTIFACT))
    parser.add_argument("--latest-fold-artifact", default=str(DEFAULT_LATEST_FOLD_ARTIFACT))
    parser.add_argument("--candidate-label", default=DEFAULT_SELECTED_CANDIDATE_LABEL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    payload = build_individual_robust_paper_decision_payload(
        walkforward_artifact_path=args.walkforward_artifact,
        latest_fold_artifact_path=args.latest_fold_artifact,
        candidate_label=args.candidate_label,
    )
    if args.check_only:
        print(
            json.dumps(
                {
                    "decision": payload["decision"],
                    "candidate": payload["selected_candidate_label"],
                    "ready_for_paper_shadow": payload["ready_for_paper_shadow"],
                    "ready_for_real": payload["ready_for_real"],
                    "selected_symbol_count": payload["selected_symbol_exposure"]["symbol_count"],
                    "gate_pass": payload["walkforward_gate"]["pass"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output = Path(args.output).expanduser().resolve()
    _write_json(output, payload)
    _write_markdown(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
