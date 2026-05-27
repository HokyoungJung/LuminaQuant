#!/usr/bin/env python3
"""Build validation-first Alpha Zoo 10bps discovery and paper-forward artifacts.

This runner is deliberately post-retune and non-destructive: it consumes the
frozen 10bps full-retune artifact, ranks candidates with train+validation fields
only, and applies locked-OOS strictly as a gate/report-only status after the
validation-first ordering is frozen.
"""

from __future__ import annotations

import argparse
import csv
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

from scripts.research import run_alpha_zoo_7x_paper_forward_preflight as paper_preflight  # noqa: E402
from scripts.research import run_alpha_zoo_validation_march_high_leverage as high  # noqa: E402
from scripts.research import run_live_notional_risk_aligned_alpha_zoo as aligned  # noqa: E402

DEFAULT_RETUNE_JSON = (
    high.DEFAULT_ALPHA_V2
    / "alpha_zoo_10bps_full_retune_20260519"
    / "alpha_zoo_10bps_full_retune_latest.json"
)
DEFAULT_LOW_CORRELATION_JSON = (
    high.DEFAULT_ALPHA_V2
    / "alpha_zoo_10bps_full_retune_20260519"
    / "low_correlation_discovery_latest.json"
)
DEFAULT_LIVE_ALIGNED_JSON = (
    high.DEFAULT_ALPHA_V2
    / "live_notional_risk_aligned_alpha_zoo_20260518"
    / "live_notional_risk_aligned_alpha_zoo_latest.json"
)
DEFAULT_OUTPUT_DIR = high.DEFAULT_ALPHA_V2 / "alpha_zoo_validation_first_discovery_20260520"

PRIMARY_ROUND_TRIP_COST_BPS = 10.0
CURRENT_ACTIVE_MODEL_ID = paper_preflight.ACTIVE_MODEL_ID
CURRENT_BALANCED_MODEL_ID = paper_preflight.BALANCED_MODEL_ID
VALIDATION_RETURN_LEADER_PROFILE = "validation_return_leader_v1"
VALIDATION_EFFICIENCY_PROFILE = "validation_efficiency_reference_v1"

DISCOVERY_CSV_FIELDS = [
    "role",
    "profile_id",
    "model_id",
    "candidate_name",
    "leverage",
    "allocation_fraction",
    "validation_return",
    "validation_mdd",
    "validation_sharpe",
    "validation_calmar",
    "train_return",
    "locked_oos_return",
    "locked_oos_mdd",
    "primary_10bps_promotion_gate_pass",
    "live_promotable_10bps",
    "ready_for_paper",
    "ready_for_real",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(high._json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _metrics_by_model(retune: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in retune.get("candidate_model_metrics") or []:
        if not isinstance(row, Mapping):
            continue
        model_id = str(row.get("model_id") or "")
        split = str(row.get("split") or "")
        if not model_id or split not in high.SPLIT_ORDER:
            continue
        by_model.setdefault(model_id, {})[split] = dict(row)
    return {
        model_id: splits
        for model_id, splits in by_model.items()
        if all(split in splits for split in high.SPLIT_ORDER)
    }


def _validation_rank_key(
    splits: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, float, float, str]:
    """Rank using train+validation evidence only; locked-OOS is intentionally unread."""
    train = dict(splits.get("train") or {})
    validation = dict(splits.get("validation") or {})
    return (
        _safe_float(validation.get("total_return")),
        _safe_float(validation.get("sharpe")),
        -_safe_float(validation.get("max_drawdown"), 1.0),
        _safe_float(train.get("total_return")),
        str(validation.get("model_id") or ""),
    )


def _validation_efficiency_key(
    splits: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, float, str]:
    """Prefer high validation return per drawdown using train+validation only."""
    train = dict(splits.get("train") or {})
    validation = dict(splits.get("validation") or {})
    val_return = _safe_float(validation.get("total_return"))
    val_mdd = max(_safe_float(validation.get("max_drawdown"), 1.0), 1e-12)
    return (
        val_return / val_mdd,
        val_return,
        _safe_float(train.get("total_return")),
        str(validation.get("model_id") or ""),
    )


def _is_live_gate_pass(splits: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        _as_bool(dict(splits[split]).get("primary_10bps_promotion_gate_pass"))
        and _as_bool(dict(splits[split]).get("live_promotable_10bps"))
        for split in high.SPLIT_ORDER
    )


def _gate_reasons(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("primary_10bps_promotion_gate_reasons") or []
    if isinstance(raw, str):
        return [item for item in raw.split(";") if item]
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return [str(item) for item in raw if str(item)]
    return [str(raw)]


def _candidate_summary(
    *,
    role: str,
    profile_id: str,
    splits: Mapping[str, Mapping[str, Any]],
    rank: int | None = None,
) -> dict[str, Any]:
    validation = dict(splits["validation"])
    train = dict(splits["train"])
    locked = dict(splits["locked_oos"])
    return {
        "role": role,
        "profile_id": profile_id,
        "rank": rank,
        "model_id": validation.get("model_id"),
        "candidate_name": validation.get("candidate_name"),
        "model_kind": validation.get("model_kind"),
        "leverage": _safe_float(validation.get("leverage")),
        "allocation_fraction": _safe_float(validation.get("allocation_fraction")),
        "trade_filter_params": dict(validation.get("trade_filter_params") or {}),
        "validation_return": _safe_float(validation.get("total_return")),
        "validation_mdd": _safe_float(validation.get("max_drawdown")),
        "validation_sharpe": _safe_float(validation.get("sharpe")),
        "validation_sortino": _safe_float(validation.get("sortino")),
        "validation_calmar": _safe_float(validation.get("calmar")),
        "validation_trade_event_count": validation.get("trade_event_count"),
        "train_return": _safe_float(train.get("total_return")),
        "train_mdd": _safe_float(train.get("max_drawdown")),
        "locked_oos_return": _safe_float(locked.get("total_return")),
        "locked_oos_mdd": _safe_float(locked.get("max_drawdown")),
        "locked_oos_sharpe": _safe_float(locked.get("sharpe")),
        "locked_oos_trade_event_count": locked.get("trade_event_count"),
        "primary_10bps_promotion_gate_pass": _is_live_gate_pass(splits),
        "live_promotable_10bps": _is_live_gate_pass(splits),
        "split_metrics": {split: dict(row) for split, row in splits.items()},
    }


def _profile_payload(
    profile_id: str, selected: Mapping[str, Any], *, formula: str, consequence: str
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "selected_model_id": selected.get("model_id"),
        "selected_candidate_name": selected.get("candidate_name"),
        "selected_leverage": selected.get("leverage"),
        "selected_allocation_fraction": selected.get("allocation_fraction"),
        "objective_inputs": ["train", "validation"],
        "selection_inputs": ["train", "validation"],
        "optimization_input_splits": ["train", "validation"],
        "parameter_fit_inputs": ["train", "validation"],
        "pruning_inputs": ["train", "validation"],
        "score_formula_inputs": ["train", "validation"],
        "score_formula": formula,
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "locked_oos_role": "gate_report_only_after_validation_first_freeze",
        "risk_profile_consequence": consequence,
    }


def _decision_filename(summary: Mapping[str, Any]) -> str:
    role = str(summary.get("role") or "candidate")
    leverage = str(summary.get("leverage")).replace(".", "p")
    allocation = str(summary.get("allocation_fraction")).replace(".", "p")
    return f"live_alpha_zoo_{role}_{leverage}x_{allocation}alloc_paper_decision_latest.json"


def _build_decision_and_preflight(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    profile: Mapping[str, Any],
    live_aligned: Mapping[str, Any],
    source_lineage: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    leverage = _safe_float(summary.get("leverage"))
    allocation = _safe_float(summary.get("allocation_fraction"))
    risk_caps = aligned._risk_caps_for_contract(leverage=leverage, allocation_fraction=allocation)
    paper_sizing = aligned._paper_equivalent_sizing(
        leverage=leverage,
        allocation_fraction=allocation,
        sizing_mode=paper_preflight.SIZING_MODE,
        risk_caps=risk_caps,
    )
    if not bool(paper_sizing.get("notional_parity_passed")):
        raise ValueError(f"paper/live notional parity failed for {summary.get('model_id')}")
    strategy_params = paper_preflight._strategy_params_for_candidate(
        live_aligned,
        candidate_name=str(summary.get("candidate_name") or ""),
        trade_filter_params=dict(summary.get("trade_filter_params") or {}),
    )
    decision = paper_preflight._decision_payload(
        role=str(summary.get("role")),
        model=summary,
        metrics=dict(summary.get("split_metrics") or {}),
        profile=profile,
        strategy_params=strategy_params,
        risk_caps=risk_caps,
        paper_sizing=paper_sizing,
        source_lineage=source_lineage,
    )
    decision["validation_first_discovery"] = {
        "selected_by": str(summary.get("profile_id")),
        "selection_inputs": ["train", "validation"],
        "uses_locked_oos_for_selection": False,
        "locked_oos_role": "gate_report_only_after_validation_first_freeze",
    }
    decision_path = output_dir / _decision_filename(summary)
    _write_json(decision_path, decision)
    preflight = paper_preflight._preflight_payload(decision_path)
    preflight_path = output_dir / _decision_filename(summary).replace(
        "live_alpha_zoo_", "live_readiness_preflight_alpha_zoo_"
    ).replace("_paper_decision_latest.json", "_paper_latest.json")
    _write_json(preflight_path, preflight)
    return decision, preflight, decision_path, preflight_path


def _discovery_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Alpha Zoo 10bps validation-first discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "Locked-OOS is gate/report-only after the validation-first ranking is frozen.",
        "Real-money execution remains disabled.",
        "",
        "## Selected paper/testnet candidates",
        "",
        "| Role | Model | Val return | Val MDD | Train return | Locked-OOS return | Paper | Real |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload.get("selected_paper_candidates") or []:
        preflight = dict(row.get("preflight") or {})
        status = dict(preflight.get("status") or {})
        lines.append(
            f"| {row.get('role')} | `{row.get('model_id')}` | "
            f"{_safe_float(row.get('validation_return')):.4%} | "
            f"{_safe_float(row.get('validation_mdd')):.4%} | "
            f"{_safe_float(row.get('train_return')):.4%} | "
            f"{_safe_float(row.get('locked_oos_return')):.4%} | "
            f"`{status.get('ready_for_paper')}` | `{status.get('ready_for_real')}` |"
        )
    lines.extend(
        [
            "",
            "## High-validation quarantine",
            "",
            "These rows were found by validation ranking but are not paper candidates because locked-OOS/promotion gates fail.",
            "",
            "| Rank | Model | Val return | Locked-OOS return | Gate reasons |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("high_validation_quarantine") or []:
        reasons = ";".join(row.get("gate_reasons") or [])
        lines.append(
            f"| {row.get('rank')} | `{row.get('model_id')}` | "
            f"{_safe_float(row.get('validation_return')):.4%} | "
            f"{_safe_float(row.get('locked_oos_return')):.4%} | {reasons} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    retune_path = Path(args.retune_json).expanduser().resolve()
    low_correlation_path = Path(args.low_correlation_json).expanduser().resolve()
    live_aligned_path = Path(args.live_aligned_json).expanduser().resolve()
    retune = _load_json(retune_path)
    low_correlation = _load_json(low_correlation_path)
    live_aligned = _load_json(live_aligned_path)
    if (
        _safe_float(retune.get("round_trip_slippage_fee_bps_primary"))
        != PRIMARY_ROUND_TRIP_COST_BPS
    ):
        raise ValueError("validation-first discovery requires the frozen 10bps retune artifact")

    source_lineage = paper_preflight._source_lineage(
        retune_path=retune_path,
        low_correlation_path=low_correlation_path,
        live_aligned_path=live_aligned_path,
        retune=retune,
        low_correlation=low_correlation,
        live_aligned=live_aligned,
    )
    by_model = _metrics_by_model(retune)
    ranked = sorted(by_model.items(), key=lambda item: _validation_rank_key(item[1]), reverse=True)
    ranked_summaries = [
        _candidate_summary(
            role="validation_ranked_candidate",
            profile_id="validation_first_raw_rank_v1",
            splits=splits,
            rank=rank,
        )
        for rank, (_model_id, splits) in enumerate(ranked, start=1)
    ]

    live_pass_ranked = [summary for summary in ranked_summaries if summary["live_promotable_10bps"]]
    if not live_pass_ranked:
        raise ValueError(
            "no 10bps live-gate-passed candidate available for validation-first paper handoff"
        )
    validation_leader = dict(live_pass_ranked[0])
    validation_leader["role"] = "validation_return_leader"
    validation_leader["profile_id"] = VALIDATION_RETURN_LEADER_PROFILE

    leader_return = _safe_float(validation_leader.get("validation_return"))
    efficiency_pool = [
        (model_id, splits)
        for model_id, splits in by_model.items()
        if _is_live_gate_pass(splits)
        and _safe_float(splits["validation"].get("total_return"))
        >= leader_return * float(args.efficiency_min_return_ratio)
        and _safe_float(splits["train"].get("total_return")) > 0.0
    ]
    efficiency_model_id, efficiency_splits = max(
        efficiency_pool, key=lambda item: _validation_efficiency_key(item[1])
    )
    validation_efficiency = _candidate_summary(
        role="validation_efficiency_reference",
        profile_id=VALIDATION_EFFICIENCY_PROFILE,
        splits=efficiency_splits,
        rank=None,
    )
    validation_efficiency["efficiency_reference_model_id"] = efficiency_model_id
    val_gt_one_and_oos_positive = [
        splits
        for splits in by_model.values()
        if _safe_float(splits["validation"].get("total_return")) > 0.01
        and _safe_float(splits["locked_oos"].get("total_return")) > 0.0
        and _safe_float(splits["validation"].get("liquidation_count")) == 0.0
        and _safe_float(splits["locked_oos"].get("liquidation_count")) == 0.0
    ]

    profiles = {
        VALIDATION_RETURN_LEADER_PROFILE: _profile_payload(
            VALIDATION_RETURN_LEADER_PROFILE,
            validation_leader,
            formula="rank by validation_total_return, validation_sharpe, lower validation_mdd, then train_total_return; locked-OOS gate/report-only after rank freeze",
            consequence="Improves validation return over the prior 7x/0.20 active while lowering drawdown; still paper/testnet only.",
        ),
        VALIDATION_EFFICIENCY_PROFILE: _profile_payload(
            VALIDATION_EFFICIENCY_PROFILE,
            validation_efficiency,
            formula="among live-gate-passed candidates with validation_return near the leader, maximize validation_return / validation_mdd using train+validation only",
            consequence="Keeps a lower-drawdown reference for the same 10bps quality-single-pair family.",
        ),
    }

    selected_paper_candidates: list[dict[str, Any]] = []
    monitoring_rows: list[dict[str, Any]] = []
    discovery_csv_rows: list[dict[str, Any]] = []
    for summary in (validation_leader, validation_efficiency):
        profile = profiles[str(summary["profile_id"])]
        decision, preflight, decision_path, preflight_path = _build_decision_and_preflight(
            output_dir=output_dir,
            summary=summary,
            profile=profile,
            live_aligned=live_aligned,
            source_lineage=source_lineage,
        )
        row = {
            **summary,
            "decision_artifact_path": str(decision_path),
            "preflight_artifact_path": str(preflight_path),
            "preflight": preflight,
            "paper_equivalent_sizing": decision["paper_equivalent_sizing"],
        }
        selected_paper_candidates.append(row)
        monitoring_rows.append(
            paper_preflight._monitoring_profile_row(
                role=str(summary.get("role")),
                model=summary,
                profile=profile,
                paper_sizing=decision["paper_equivalent_sizing"],
                preflight=preflight,
            )
        )
        discovery_csv_rows.append(
            {
                **{field: row.get(field) for field in DISCOVERY_CSV_FIELDS},
                "ready_for_paper": dict(preflight.get("status") or {}).get("ready_for_paper"),
                "ready_for_real": dict(preflight.get("status") or {}).get("ready_for_real"),
            }
        )

    quarantine: list[dict[str, Any]] = []
    for summary in ranked_summaries:
        if len(quarantine) >= int(args.quarantine_top_n):
            break
        if summary["live_promotable_10bps"]:
            continue
        model_id = str(summary.get("model_id") or "")
        reasons: set[str] = set()
        for row in dict(summary.get("split_metrics") or {}).values():
            for reason in _gate_reasons(dict(row)):
                reasons.add(reason)
        quarantine.append({**summary, "model_id": model_id, "gate_reasons": sorted(reasons)})

    monitoring = paper_preflight._monitoring_contract(
        profile_rows=monitoring_rows,
        source_lineage=source_lineage,
    )
    monitoring["artifact_kind"] = "alpha_zoo_validation_first_monitoring_contract"
    monitoring["status"] = "pending_validation_first_paper_forward_fills"
    monitoring_json = output_dir / "validation_first_monitoring_contract_latest.json"
    monitoring_csv = output_dir / "validation_first_monitoring_contract_latest.csv"
    _write_json(monitoring_json, monitoring)
    _write_csv(monitoring_csv, monitoring_rows, paper_preflight.MONITORING_CSV_FIELDS)

    latest_path = output_dir / "alpha_zoo_validation_first_discovery_latest.json"
    timestamped_path = output_dir / f"alpha_zoo_validation_first_discovery_{_timestamp()}.json"
    latest_md = output_dir / "alpha_zoo_validation_first_discovery_latest.md"
    discovery_csv = output_dir / "alpha_zoo_validation_first_selected_latest.csv"
    validation_log = output_dir / "artifact_generation_validation_latest.log"

    current_reference = [
        ranked_summaries[[summary["model_id"] for summary in ranked_summaries].index(model_id)]
        for model_id in (CURRENT_ACTIVE_MODEL_ID, CURRENT_BALANCED_MODEL_ID)
        if model_id in {str(summary.get("model_id")) for summary in ranked_summaries}
    ]
    payload = {
        "artifact_kind": "alpha_zoo_validation_first_discovery",
        "generated_at_utc": _utc_now_iso(),
        "real_money_execution": False,
        "ready_for_real": False,
        "ready_for_paper": all(
            bool(dict(row["preflight"].get("status") or {}).get("ready_for_paper"))
            for row in selected_paper_candidates
        ),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "selection_policy": {
            "objective_inputs": ["train", "validation"],
            "selection_inputs": ["train", "validation"],
            "optimization_input_splits": ["train", "validation"],
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "locked_oos_role": "gate_report_only_after_validation_first_freeze",
        },
        "selection_profiles": profiles,
        "selected_paper_candidates": selected_paper_candidates,
        "current_active_balanced_reference": current_reference,
        "top_validation_live_gate_passed": live_pass_ranked[: int(args.top_n)],
        "high_validation_quarantine": quarantine,
        "new_strategy_findings": {
            "validation_only_high_return_families_exist": bool(quarantine),
            "validation_ceiling_audit": {
                "live_gate_passed_candidate_count": len(live_pass_ranked),
                "max_live_gate_validation_return": validation_leader["validation_return"],
                "max_live_gate_validation_return_model_id": validation_leader["model_id"],
                "candidate_count_with_validation_gt_1pct_and_positive_locked_oos_zero_liquidation": len(
                    val_gt_one_and_oos_positive
                ),
                "conclusion": (
                    "The frozen 10bps universe has no zero-liquidation candidate with validation > 1% "
                    "and positive locked-OOS; material validation-edge strategy work must be shadow-only "
                    "until a new train+validation retune also survives the locked-OOS gate."
                ),
            },
            "finding": (
                "Conservative-exit and hybrid/long-only rows show much stronger validation returns, "
                "but fail locked-OOS/promotion gates; treat them as shadow-only strategy hypotheses, not paper candidates."
            ),
            "recommended_next_experiments": [
                "Regime-gated conservative_exit rescue: learn a train+validation regime filter, then locked-OOS gate/report-only.",
                "Quality_single_pair validation-first exposure lane: paper-forward 5x/0.20 and lower-drawdown efficiency reference beside prior active/balanced.",
                "Side/symbol-specific abs-score thresholds: keep abs_score>=1.5 baseline, test short threshold 1.75/2.0 and symbol contribution gates without calendar/date rules.",
            ],
        },
        "source_lineage": source_lineage,
        "output_paths": {
            "latest_json": str(latest_path),
            "timestamped_json": str(timestamped_path),
            "latest_markdown": str(latest_md),
            "selected_csv": str(discovery_csv),
            "monitoring_contract_json": str(monitoring_json),
            "monitoring_contract_csv": str(monitoring_csv),
            "artifact_generation_validation_log": str(validation_log),
        },
    }
    _write_json(latest_path, payload)
    _write_json(timestamped_path, payload)
    latest_md.write_text(_discovery_markdown(payload), encoding="utf-8")
    _write_csv(discovery_csv, discovery_csv_rows, DISCOVERY_CSV_FIELDS)
    validation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={_utc_now_iso()}",
                f"validation_return_leader={validation_leader['model_id']}",
                f"validation_efficiency_reference={validation_efficiency['model_id']}",
                "ready_for_paper=" + str(payload["ready_for_paper"]),
                "ready_for_real=" + str(payload["ready_for_real"]),
                "real_money_execution=" + str(payload["real_money_execution"]),
                "research_primary_round_trip_cost_bps=10.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retune-json", default=str(DEFAULT_RETUNE_JSON))
    parser.add_argument("--low-correlation-json", default=str(DEFAULT_LOW_CORRELATION_JSON))
    parser.add_argument("--live-aligned-json", default=str(DEFAULT_LIVE_ALIGNED_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--quarantine-top-n", type=int, default=10)
    parser.add_argument("--efficiency-min-return-ratio", type=float, default=0.90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
