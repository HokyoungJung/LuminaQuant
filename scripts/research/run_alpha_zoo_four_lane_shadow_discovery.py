#!/usr/bin/env python3
"""Build a four-lane Alpha Zoo paper-forward contract and shadow strategy audit.

This runner is artifact-derived and intentionally light-weight.  It joins the
existing 10bps active/balanced paper handoff with the validation-first paper
handoff, then audits the frozen 10bps candidate universe for new strategy
hypotheses.  Locked-OOS remains gate/report-only and no real-money artifact is
produced.
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

DEFAULT_ALPHA_V2 = high.DEFAULT_ALPHA_V2
DEFAULT_RETUNE_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_10bps_full_retune_20260519"
    / "alpha_zoo_10bps_full_retune_latest.json"
)
DEFAULT_ACTIVE_BALANCED_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_7x_paper_forward_preflight_20260519"
    / "alpha_zoo_7x_paper_forward_preflight_latest.json"
)
DEFAULT_VALIDATION_FIRST_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_validation_first_discovery_20260520"
    / "alpha_zoo_validation_first_discovery_latest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ALPHA_V2 / "alpha_zoo_four_lane_shadow_discovery_20260520"
PRIMARY_ROUND_TRIP_COST_BPS = 10.0

FOUR_LANE_CSV_FIELDS = [
    "lane_group",
    "role",
    "profile_id",
    "model_id",
    "leverage",
    "allocation_fraction",
    "target_notional_fraction_of_equity",
    "validation_return",
    "validation_mdd",
    "train_return",
    "locked_oos_return",
    "locked_oos_mdd",
    "locked_oos_liquidation_count",
    "notional_parity_passed",
    "ready_for_paper",
    "ready_for_real",
]

SHADOW_CSV_FIELDS = [
    "shadow_group",
    "rank",
    "model_id",
    "candidate_name",
    "variant_name",
    "leverage",
    "allocation_fraction",
    "validation_return",
    "validation_mdd",
    "validation_trade_event_count",
    "train_return",
    "locked_oos_return",
    "locked_oos_mdd",
    "locked_oos_trade_event_count",
    "locked_oos_liquidation_count",
    "primary_10bps_promotion_gate_pass",
    "live_promotable_10bps",
    "gate_reasons",
    "trade_filter_params",
    "shadow_status",
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
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.dumps(high._json_safe(value), sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return high._json_safe(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError, OverflowError:
        return default
    return parsed if math.isfinite(parsed) else default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _gate_reasons(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("primary_10bps_promotion_gate_reasons") or []
    if isinstance(raw, str):
        return [item for item in raw.split(";") if item]
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return [str(item) for item in raw if str(item)]
    return [str(raw)] if raw else []


def _split(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    metrics = row.get("split_metrics") or {}
    return dict(metrics.get(split) or {}) if isinstance(metrics, Mapping) else {}


def _all_split_gate_pass(row: Mapping[str, Any]) -> bool:
    return all(
        _as_bool(_split(row, split).get("primary_10bps_promotion_gate_pass"))
        and _as_bool(_split(row, split).get("live_promotable_10bps"))
        for split in high.SPLIT_ORDER
    )


def _lane_summary(row: Mapping[str, Any], *, lane_group: str) -> dict[str, Any]:
    train = _split(row, "train")
    validation = _split(row, "validation")
    locked = _split(row, "locked_oos")
    preflight = dict(row.get("preflight") or {})
    status = dict(preflight.get("status") or {})
    sizing = dict(row.get("paper_equivalent_sizing") or {})
    leverage = _safe_float(row.get("leverage") or validation.get("leverage"))
    allocation = _safe_float(
        row.get("allocation_fraction") or validation.get("allocation_fraction")
    )
    return {
        "lane_group": lane_group,
        "role": row.get("role"),
        "profile_id": row.get("profile_id"),
        "model_id": row.get("model_id"),
        "candidate_name": validation.get("candidate_name") or row.get("candidate_name"),
        "leverage": leverage,
        "allocation_fraction": allocation,
        "target_notional_fraction_of_equity": leverage * allocation,
        "sizing_mode": dict(sizing.get("fixture") or {}).get("sizing_mode")
        or paper_preflight.SIZING_MODE,
        "expected_replay_notional_for_10000_equity": sizing.get("expected_replay_notional"),
        "live_notional_for_10000_equity": sizing.get("live_notional"),
        "notional_parity_passed": bool(sizing.get("notional_parity_passed")),
        "risk_check_passed": bool(sizing.get("risk_check_passed")),
        "validation_return": _safe_float(
            validation.get("total_return") or row.get("validation_return")
        ),
        "validation_mdd": _safe_float(validation.get("max_drawdown") or row.get("validation_mdd")),
        "validation_sharpe": _safe_float(validation.get("sharpe") or row.get("validation_sharpe")),
        "validation_trade_event_count": validation.get("trade_event_count")
        or row.get("validation_trade_event_count"),
        "train_return": _safe_float(train.get("total_return") or row.get("train_return")),
        "train_mdd": _safe_float(train.get("max_drawdown") or row.get("train_mdd")),
        "locked_oos_return": _safe_float(
            locked.get("total_return") or row.get("locked_oos_return")
        ),
        "locked_oos_mdd": _safe_float(locked.get("max_drawdown") or row.get("locked_oos_mdd")),
        "locked_oos_liquidation_count": _safe_float(locked.get("liquidation_count")),
        "locked_oos_account_wipeout_count": _safe_float(locked.get("account_wipeout_count")),
        "locked_oos_trade_event_count": locked.get("trade_event_count")
        or row.get("locked_oos_trade_event_count"),
        "primary_10bps_promotion_gate_pass": _all_split_gate_pass(row),
        "live_promotable_10bps": _all_split_gate_pass(row),
        "ready_for_paper": bool(status.get("ready_for_paper")),
        "ready_for_real": bool(status.get("ready_for_real")),
        "real_money_execution": False,
        "decision_artifact_path": row.get("decision_artifact_path"),
        "preflight_artifact_path": row.get("preflight_artifact_path"),
    }


def _metrics_by_model(retune: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in retune.get("candidate_model_metrics") or []:
        if not isinstance(row, Mapping):
            continue
        model_id = str(row.get("model_id") or "")
        split = str(row.get("split") or "")
        if model_id and split in high.SPLIT_ORDER:
            by_model.setdefault(model_id, {})[split] = dict(row)
    return {
        model_id: splits
        for model_id, splits in by_model.items()
        if all(split in splits for split in high.SPLIT_ORDER)
    }


def _live_gate_pass(splits: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        _as_bool(dict(splits[split]).get("primary_10bps_promotion_gate_pass"))
        and _as_bool(dict(splits[split]).get("live_promotable_10bps"))
        for split in high.SPLIT_ORDER
    )


def _candidate_summary(
    model_id: str, splits: Mapping[str, Mapping[str, Any]], *, rank: int, shadow_group: str
) -> dict[str, Any]:
    train = dict(splits["train"])
    validation = dict(splits["validation"])
    locked = dict(splits["locked_oos"])
    reasons: set[str] = set()
    for row in (train, validation, locked):
        reasons.update(_gate_reasons(row))
    live_gate = _live_gate_pass(splits)
    locked_return = _safe_float(locked.get("total_return"))
    status = "paper_candidate" if live_gate else "shadow_only"
    if not live_gate and locked_return <= 0.0:
        status = "shadow_only_locked_oos_negative"
    elif not live_gate and _safe_float(locked.get("liquidation_count")) > 0.0:
        status = "shadow_only_locked_oos_liquidation"
    return {
        "shadow_group": shadow_group,
        "rank": rank,
        "model_id": model_id,
        "candidate_name": validation.get("candidate_name"),
        "model_kind": validation.get("model_kind"),
        "variant_name": validation.get("variant_name"),
        "trade_filter_params": dict(validation.get("trade_filter_params") or {}),
        "leverage": _safe_float(validation.get("leverage")),
        "allocation_fraction": _safe_float(validation.get("allocation_fraction")),
        "validation_return": _safe_float(validation.get("total_return")),
        "validation_mdd": _safe_float(validation.get("max_drawdown")),
        "validation_sharpe": _safe_float(validation.get("sharpe")),
        "validation_trade_event_count": validation.get("trade_event_count"),
        "train_return": _safe_float(train.get("total_return")),
        "train_mdd": _safe_float(train.get("max_drawdown")),
        "locked_oos_return": locked_return,
        "locked_oos_mdd": _safe_float(locked.get("max_drawdown")),
        "locked_oos_sharpe": _safe_float(locked.get("sharpe")),
        "locked_oos_trade_event_count": locked.get("trade_event_count"),
        "locked_oos_liquidation_count": _safe_float(locked.get("liquidation_count")),
        "primary_10bps_promotion_gate_pass": live_gate,
        "live_promotable_10bps": live_gate,
        "gate_reasons": sorted(reasons),
        "shadow_status": status,
    }


def _ranked_candidates(
    retune: Mapping[str, Any], predicate: Any, *, shadow_group: str, top_n: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, splits in _metrics_by_model(retune).items():
        summary = _candidate_summary(model_id, splits, rank=0, shadow_group=shadow_group)
        if predicate(summary):
            rows.append(summary)
    rows.sort(
        key=lambda row: (
            _safe_float(row.get("validation_return")),
            _safe_float(row.get("train_return")),
            _safe_float(row.get("locked_oos_return")),
            str(row.get("model_id") or ""),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows[:top_n], start=1):
        row["rank"] = idx
    return rows[:top_n]


def _quality_surface(retune: Mapping[str, Any], *, top_n: int) -> dict[str, Any]:
    live_quality = _ranked_candidates(
        retune,
        lambda row: (
            row.get("candidate_name") == "alpha_zoo_quality_single_pair"
            and row.get("live_promotable_10bps")
        ),
        shadow_group="quality_single_pair_live_surface",
        top_n=top_n,
    )
    by_variant: dict[str, int] = {}
    for row in live_quality:
        key = json.dumps(row.get("trade_filter_params") or {}, sort_keys=True)
        by_variant[key] = by_variant.get(key, 0) + 1
    return {
        "top_live_quality_candidates": live_quality,
        "live_quality_top_n": len(live_quality),
        "observed_live_quality_filter_params_in_top_n": by_variant,
        "finding": "The frozen 10bps live-gate surface is still dominated by quality_single_pair abs_factor_score_min=1.5; lower exposure improves validation/MDD but does not create >1% validation return.",
    }


def _monitoring_rows(lanes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in lanes:
        rows.append(
            {
                "role": lane.get("role"),
                "profile_id": lane.get("profile_id"),
                "model_id": lane.get("model_id"),
                "leverage": lane.get("leverage"),
                "allocation_fraction": lane.get("allocation_fraction"),
                "sizing_mode": lane.get("sizing_mode"),
                "target_notional_fraction_of_equity": lane.get(
                    "target_notional_fraction_of_equity"
                ),
                "isolated_margin_fraction_of_equity": lane.get("allocation_fraction"),
                "expected_replay_notional_for_10000_equity": lane.get(
                    "expected_replay_notional_for_10000_equity"
                ),
                "live_notional_for_10000_equity": lane.get("live_notional_for_10000_equity"),
                "notional_parity_passed": lane.get("notional_parity_passed"),
                "research_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
                "observed_round_trips": 0,
                "mean_all_in_round_trip_bps": None,
                "p95_all_in_round_trip_bps": None,
                "cost_status": "pending_no_fills",
                "ready_for_paper": lane.get("ready_for_paper"),
                "ready_for_real": False,
            }
        )
    return rows


def _monitoring_contract(
    lanes: Sequence[Mapping[str, Any]], *, source_lineage: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "artifact_kind": "alpha_zoo_four_lane_paper_forward_monitoring_contract",
        "generated_at_utc": _utc_now_iso(),
        "status": "pending_paper_forward_fills",
        "research_assumption": {
            "primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
            "mean_all_in_round_trip_bps_limit": 10.0,
            "p95_all_in_round_trip_bps_limit": 15.0,
        },
        "required_grouping_keys": [
            "timestamp_utc",
            "paper_lane_role",
            "model_id",
            "symbol",
            "side",
        ],
        "fill_observation_schema": [
            "timestamp_utc",
            "paper_lane_role",
            "model_id",
            "symbol",
            "side",
            "entry_order_id",
            "exit_order_id",
            "notional_quote",
            "realized_fee_bps",
            "realized_slippage_bps",
            "all_in_round_trip_bps",
            "maker_fill_ratio",
            "taker_fill_ratio",
            "partial_fill_ratio",
            "missed_signal_ratio",
            "liquidation_event",
            "account_equity_after_liquidation",
            "lane_drawdown",
        ],
        "formulas": {
            "target_notional": "equity * allocation_fraction * leverage",
            "isolated_margin": "equity * allocation_fraction",
            "all_in_round_trip_bps": "realized_fee_bps + realized_slippage_bps",
            "liquidation_inclusive_equity": "account equity after subtracting isolated liquidation losses",
        },
        "pass_fail_rules": {
            "realized_mean_cost_pass": "mean(all_in_round_trip_bps) <= 10.0",
            "realized_p95_cost_pass": "p95(all_in_round_trip_bps) <= 15.0",
            "active_drawdown_pass": "active lane drawdown <= 30%",
            "balanced_or_validation_reference_drawdown_pass": "reference lane drawdown <= 25%",
            "real_money_gate": "always false in this artifact; user approval and separate real preflight required",
        },
        "profile_rows": _monitoring_rows(lanes),
        "source_lineage": dict(source_lineage),
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Alpha Zoo 10bps four-lane paper-forward and shadow discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "Real-money execution remains disabled. Locked-OOS is gate/report-only.",
        "",
        "## Four paper/testnet lanes",
        "",
        "| Lane | Model | Lev/Alloc | Val return | Val MDD | Train return | Locked-OOS | Notional/equity | Paper | Real |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for lane in payload.get("four_lane_paper_candidates") or []:
        lines.append(
            f"| {lane.get('role')} | `{lane.get('model_id')}` | "
            f"{_safe_float(lane.get('leverage')):.1f}x/{_safe_float(lane.get('allocation_fraction')):.3f} | "
            f"{_safe_float(lane.get('validation_return')):.4%} | "
            f"{_safe_float(lane.get('validation_mdd')):.4%} | "
            f"{_safe_float(lane.get('train_return')):.4%} | "
            f"{_safe_float(lane.get('locked_oos_return')):.4%} | "
            f"{_safe_float(lane.get('target_notional_fraction_of_equity')):.1%} | "
            f"`{lane.get('ready_for_paper')}` | `{lane.get('ready_for_real')}` |"
        )
    lines.extend(
        [
            "",
            "## Shadow strategy findings",
            "",
            str(dict(payload.get("strategy_findings") or {}).get("summary") or ""),
            "",
            "### Top conservative-exit rescue hypotheses",
            "",
            "| Rank | Model | Val return | Train return | Locked-OOS | Status |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in (
        dict(payload.get("shadow_discovery") or {}).get("conservative_exit_rescue_hypotheses") or []
    ):
        lines.append(
            f"| {row.get('rank')} | `{row.get('model_id')}` | "
            f"{_safe_float(row.get('validation_return')):.4%} | "
            f"{_safe_float(row.get('train_return')):.4%} | "
            f"{_safe_float(row.get('locked_oos_return')):.4%} | {row.get('shadow_status')} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    retune_path = Path(args.retune_json).expanduser().resolve()
    active_balanced_path = Path(args.active_balanced_json).expanduser().resolve()
    validation_first_path = Path(args.validation_first_json).expanduser().resolve()
    retune = _load_json(retune_path)
    active_balanced = _load_json(active_balanced_path)
    validation_first = _load_json(validation_first_path)
    if (
        _safe_float(retune.get("round_trip_slippage_fee_bps_primary"))
        != PRIMARY_ROUND_TRIP_COST_BPS
    ):
        raise ValueError("four-lane discovery requires the frozen 10bps retune artifact")

    current_lanes = [
        _lane_summary(row, lane_group="current_active_balanced")
        for row in active_balanced.get("side_by_side_profiles") or []
    ]
    validation_lanes = [
        _lane_summary(row, lane_group="validation_first")
        for row in validation_first.get("selected_paper_candidates") or []
    ]
    lanes = current_lanes + validation_lanes
    if len(lanes) != 4:
        raise ValueError(f"expected four lanes, got {len(lanes)}")
    if any(not bool(lane.get("notional_parity_passed")) for lane in lanes):
        raise ValueError("all four lanes must pass replay/live notional parity")

    conservative = _ranked_candidates(
        retune,
        lambda row: "conservative_exit" in str(row.get("candidate_name") or ""),
        shadow_group="conservative_exit_rescue_hypothesis",
        top_n=int(args.shadow_top_n),
    )
    side_family = _ranked_candidates(
        retune,
        lambda row: (
            bool(row.get("trade_filter_params"))
            and row.get("candidate_name") != "alpha_zoo_quality_single_pair"
            and any(
                key in dict(row.get("trade_filter_params") or {})
                for key in ("side", "symbol", "dominant_factor_family")
            )
        ),
        shadow_group="side_family_threshold_hypothesis",
        top_n=int(args.shadow_top_n),
    )
    side_family_positive_oos = [
        row
        for row in side_family
        if _safe_float(row.get("locked_oos_return")) > 0.0
        and _safe_float(row.get("locked_oos_liquidation_count")) == 0.0
    ]
    quality_surface = _quality_surface(retune, top_n=int(args.quality_top_n))

    source_lineage = {
        "retune_10bps": str(retune_path),
        "active_balanced_paper_forward": str(active_balanced_path),
        "validation_first_discovery": str(validation_first_path),
        "selection_inputs": ["train", "validation"],
        "locked_oos_role": "gate_report_only_after_train_validation_selection_freeze",
        "round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
    }
    monitoring = _monitoring_contract(lanes, source_lineage=source_lineage)

    timestamp = _timestamp()
    latest_json = output_dir / "alpha_zoo_four_lane_shadow_discovery_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_four_lane_shadow_discovery_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_four_lane_shadow_discovery_latest.md"
    four_lane_csv = output_dir / "four_lane_paper_forward_candidates_latest.csv"
    shadow_csv = output_dir / "shadow_strategy_hypotheses_latest.csv"
    monitoring_json = output_dir / "four_lane_monitoring_contract_latest.json"
    monitoring_csv = output_dir / "four_lane_monitoring_contract_latest.csv"
    generation_log = output_dir / "artifact_generation_validation_latest.log"

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_four_lane_shadow_discovery",
        "generated_at_utc": _utc_now_iso(),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "ready_for_paper": all(bool(lane.get("ready_for_paper")) for lane in lanes),
        "ready_for_real": False,
        "real_money_execution": False,
        "four_lane_paper_candidates": lanes,
        "monitoring_contract": monitoring,
        "shadow_discovery": {
            "conservative_exit_rescue_hypotheses": conservative,
            "side_family_threshold_hypotheses": side_family,
            "quality_single_pair_surface": quality_surface,
        },
        "strategy_findings": {
            "summary": "Paper-forward should compare four quality_single_pair lanes; frozen 10bps data does not justify promoting conservative_exit or side/family filters because their validation edge fails locked-OOS gates.",
            "current_paper_action": "Run active 7x/0.20, balanced 6x/0.175, validation leader 5x/0.20, and efficiency reference 4x/0.175 side-by-side in paper/testnet only.",
            "side_family_positive_oos_zero_liq_in_top_shadow_count": len(side_family_positive_oos),
            "conservative_exit_top_validation_locked_oos_positive_count": sum(
                1 for row in conservative if _safe_float(row.get("locked_oos_return")) > 0.0
            ),
            "recommended_next_experiments": [
                "Fresh conservative_exit rescue retune with train+validation-only regime filter; locked-OOS gate/report-only after freeze.",
                "Quality_single_pair side-specific abs-score thresholds: LONG >=1.5 baseline, SHORT >=1.75/2.0, no calendar/date inputs.",
                "Symbol contribution/cost audit from paper fills before any new real-money gate.",
                "Expanded trade-filter retune with larger selected variant inventory and min train/validation trade-count guards.",
            ],
        },
        "source_lineage": source_lineage,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "four_lane_csv": str(four_lane_csv),
            "shadow_csv": str(shadow_csv),
            "monitoring_contract_json": str(monitoring_json),
            "monitoring_contract_csv": str(monitoring_csv),
            "artifact_generation_validation_log": str(generation_log),
        },
    }

    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    _write_csv(four_lane_csv, lanes, FOUR_LANE_CSV_FIELDS)
    _write_csv(shadow_csv, conservative + side_family, SHADOW_CSV_FIELDS)
    _write_json(monitoring_json, monitoring)
    _write_csv(monitoring_csv, monitoring["profile_rows"], paper_preflight.MONITORING_CSV_FIELDS)
    generation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"research_primary_round_trip_cost_bps={PRIMARY_ROUND_TRIP_COST_BPS}",
                f"four_lane_count={len(lanes)}",
                f"conservative_shadow_count={len(conservative)}",
                f"side_family_shadow_count={len(side_family)}",
                "ready_for_real=false",
                "real_money_execution=false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retune-json", default=str(DEFAULT_RETUNE_JSON))
    parser.add_argument("--active-balanced-json", default=str(DEFAULT_ACTIVE_BALANCED_JSON))
    parser.add_argument("--validation-first-json", default=str(DEFAULT_VALIDATION_FIRST_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--shadow-top-n", type=int, default=12)
    parser.add_argument("--quality-top-n", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
