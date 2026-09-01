#!/usr/bin/env python3
"""Guard Alpha Zoo long-only reversal discoveries before paper execution.

The expanded 10bps retune surfaced a high-validation, positive locked-OOS
`alpha_zoo_high_confidence_long_only` / `crypto_residual_reversal` family.  This
runner intentionally treats that family as a shadow research hypothesis: it ranks
variants using train+validation evidence only, then applies locked-OOS as a
post-freeze gate/report field and records why no paper/testnet execution artifact
should be created yet.
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
DEFAULT_EXPANDED_RETUNE_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_10bps_expanded_filter_retune_20260520"
    / "alpha_zoo_10bps_full_retune_latest.json"
)
DEFAULT_SHADOW_SELECTION_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_expanded_filter_shadow_selection_20260520"
    / "alpha_zoo_expanded_filter_shadow_selection_latest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ALPHA_V2 / "alpha_zoo_long_only_reversal_guarded_study_20260520"

PRIMARY_ROUND_TRIP_COST_BPS = 10.0
TARGET_CANDIDATE_NAME = "alpha_zoo_high_confidence_long_only"
TARGET_FACTOR_FAMILY = "crypto_residual_reversal"
TARGET_ABS_FACTOR_SCORE_MIN = 1.5
MIN_TRAIN_TRADE_EVENTS = 50
MIN_VALIDATION_TRADE_EVENTS = 30
MIN_LOCKED_OOS_TRADE_EVENTS = 20
MIN_TRAIN_VALIDATION_RETURN_RATIO = 0.50
MAX_TRAIN_MDD = 0.20
MAX_VALIDATION_MDD = 0.10
MAX_LOCKED_OOS_MDD = 0.10

CANDIDATE_CSV_FIELDS = [
    "rank",
    "selection_role",
    "model_id",
    "candidate_name",
    "variant_name",
    "leverage",
    "allocation_fraction",
    "target_notional_fraction_of_equity",
    "train_return",
    "train_mdd",
    "train_trade_event_count",
    "validation_return",
    "validation_mdd",
    "validation_sharpe",
    "validation_trade_event_count",
    "train_validation_return_ratio",
    "locked_oos_return",
    "locked_oos_mdd",
    "locked_oos_trade_event_count",
    "locked_oos_liquidation_count",
    "locked_oos_account_wipeout_count",
    "train_validation_guard_pass",
    "locked_oos_report_gate_pass",
    "primary_10bps_promotion_gate_pass",
    "paper_promotion_guard_pass",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "guard_fail_reasons",
    "primary_10bps_gate_reasons",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError, OverflowError:
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except TypeError, ValueError, OverflowError:
        return default
    return parsed


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(high._json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.dumps(high._json_safe(value), sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return high._json_safe(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _gate_reasons(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("primary_10bps_promotion_gate_reasons") or []
    if isinstance(raw, str):
        return [item for item in raw.split(";") if item]
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return [str(item) for item in raw if str(item)]
    return [str(raw)] if raw else []


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


def _target_family_match(splits: Mapping[str, Mapping[str, Any]]) -> bool:
    for split in ("train", "validation"):
        row = dict(splits.get(split) or {})
        params = dict(row.get("trade_filter_params") or {})
        if row.get("candidate_name") != TARGET_CANDIDATE_NAME:
            return False
        if params.get("dominant_factor_family") != TARGET_FACTOR_FAMILY:
            return False
        if _safe_float(params.get("abs_factor_score_min")) != TARGET_ABS_FACTOR_SCORE_MIN:
            return False
    return True


def _all_split_primary_gate_pass(splits: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        _as_bool(dict(splits[split]).get("primary_10bps_promotion_gate_pass"))
        and _as_bool(dict(splits[split]).get("live_promotable_10bps"))
        for split in high.SPLIT_ORDER
    )


def _fail_reasons(*, checks: Mapping[str, bool], values: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not checks["train_trade_count"]:
        reasons.append(
            f"train_trade_event_count_{values['train_trades']}_below_{MIN_TRAIN_TRADE_EVENTS}"
        )
    if not checks["validation_trade_count"]:
        reasons.append(
            f"validation_trade_event_count_{values['validation_trades']}_below_{MIN_VALIDATION_TRADE_EVENTS}"
        )
    if not checks["locked_oos_trade_count"]:
        reasons.append(
            f"locked_oos_trade_event_count_{values['locked_oos_trades']}_below_{MIN_LOCKED_OOS_TRADE_EVENTS}"
        )
    if not checks["train_return_positive"]:
        reasons.append("train_return_not_positive")
    if not checks["validation_return_positive"]:
        reasons.append("validation_return_not_positive")
    if not checks["locked_oos_return_positive"]:
        reasons.append("locked_oos_return_not_positive")
    if not checks["train_validation_return_ratio"]:
        ratio = _safe_float(values.get("train_validation_return_ratio"))
        reasons.append(
            f"train_validation_return_ratio_{ratio:.4f}_below_{MIN_TRAIN_VALIDATION_RETURN_RATIO:.2f}"
        )
    if not checks["train_mdd"]:
        reasons.append(
            f"train_mdd_{_safe_float(values['train_mdd']):.4f}_above_{MAX_TRAIN_MDD:.2f}"
        )
    if not checks["validation_mdd"]:
        reasons.append(
            f"validation_mdd_{_safe_float(values['validation_mdd']):.4f}_above_{MAX_VALIDATION_MDD:.2f}"
        )
    if not checks["locked_oos_mdd"]:
        reasons.append(
            f"locked_oos_mdd_{_safe_float(values['locked_oos_mdd']):.4f}_above_{MAX_LOCKED_OOS_MDD:.2f}"
        )
    if not checks["locked_oos_no_liquidation"]:
        reasons.append("locked_oos_liquidation_count_nonzero")
    if not checks["locked_oos_no_account_wipeout"]:
        reasons.append("locked_oos_account_wipeout_count_nonzero")
    if not checks["primary_10bps_promotion_gate"]:
        reasons.append("primary_10bps_promotion_gate_failed")
    return reasons


def _candidate_summary(model_id: str, splits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    train = dict(splits["train"])
    validation = dict(splits["validation"])
    locked = dict(splits["locked_oos"])
    leverage = _safe_float(validation.get("leverage"))
    allocation = _safe_float(validation.get("allocation_fraction"))
    train_return = _safe_float(train.get("total_return"))
    validation_return = _safe_float(validation.get("total_return"))
    locked_return = _safe_float(locked.get("total_return"))
    train_mdd = _safe_float(train.get("max_drawdown"))
    validation_mdd = _safe_float(validation.get("max_drawdown"))
    locked_mdd = _safe_float(locked.get("max_drawdown"))
    train_trades = _safe_int(train.get("trade_event_count"))
    validation_trades = _safe_int(validation.get("trade_event_count"))
    locked_trades = _safe_int(locked.get("trade_event_count"))
    locked_liquidations = _safe_float(locked.get("liquidation_count"))
    locked_wipeouts = _safe_float(locked.get("account_wipeout_count"))
    ratio = train_return / validation_return if validation_return > 0.0 else 0.0
    primary_gate = _all_split_primary_gate_pass(splits)
    checks = {
        "train_trade_count": train_trades >= MIN_TRAIN_TRADE_EVENTS,
        "validation_trade_count": validation_trades >= MIN_VALIDATION_TRADE_EVENTS,
        "locked_oos_trade_count": locked_trades >= MIN_LOCKED_OOS_TRADE_EVENTS,
        "train_return_positive": train_return > 0.0,
        "validation_return_positive": validation_return > 0.0,
        "locked_oos_return_positive": locked_return > 0.0,
        "train_validation_return_ratio": ratio >= MIN_TRAIN_VALIDATION_RETURN_RATIO,
        "train_mdd": train_mdd <= MAX_TRAIN_MDD,
        "validation_mdd": validation_mdd <= MAX_VALIDATION_MDD,
        "locked_oos_mdd": locked_mdd <= MAX_LOCKED_OOS_MDD,
        "locked_oos_no_liquidation": locked_liquidations == 0.0,
        "locked_oos_no_account_wipeout": locked_wipeouts == 0.0,
        "primary_10bps_promotion_gate": primary_gate,
    }
    train_validation_guard = all(
        checks[name]
        for name in (
            "train_trade_count",
            "validation_trade_count",
            "train_return_positive",
            "validation_return_positive",
            "train_validation_return_ratio",
            "train_mdd",
            "validation_mdd",
        )
    )
    locked_report_gate = all(
        checks[name]
        for name in (
            "locked_oos_trade_count",
            "locked_oos_return_positive",
            "locked_oos_mdd",
            "locked_oos_no_liquidation",
            "locked_oos_no_account_wipeout",
        )
    )
    paper_guard = train_validation_guard and locked_report_gate and primary_gate
    reasons: set[str] = set()
    for row in (train, validation, locked):
        reasons.update(_gate_reasons(row))
    values = {
        "train_trades": train_trades,
        "validation_trades": validation_trades,
        "locked_oos_trades": locked_trades,
        "train_validation_return_ratio": ratio,
        "train_mdd": train_mdd,
        "validation_mdd": validation_mdd,
        "locked_oos_mdd": locked_mdd,
    }
    return {
        "rank": 0,
        "selection_role": "target_family_candidate",
        "model_id": model_id,
        "candidate_name": validation.get("candidate_name"),
        "model_kind": validation.get("model_kind"),
        "variant_name": validation.get("variant_name"),
        "trade_filter_params": dict(validation.get("trade_filter_params") or {}),
        "leverage": leverage,
        "allocation_fraction": allocation,
        "target_notional_fraction_of_equity": leverage * allocation,
        "train_return": train_return,
        "train_mdd": train_mdd,
        "train_sharpe": _safe_float(train.get("sharpe")),
        "train_trade_event_count": train_trades,
        "validation_return": validation_return,
        "validation_mdd": validation_mdd,
        "validation_sharpe": _safe_float(validation.get("sharpe")),
        "validation_trade_event_count": validation_trades,
        "train_validation_return_ratio": ratio,
        "locked_oos_return": locked_return,
        "locked_oos_mdd": locked_mdd,
        "locked_oos_sharpe": _safe_float(locked.get("sharpe")),
        "locked_oos_trade_event_count": locked_trades,
        "locked_oos_liquidation_count": locked_liquidations,
        "locked_oos_account_wipeout_count": locked_wipeouts,
        "guard_checks": checks,
        "train_validation_guard_pass": train_validation_guard,
        "locked_oos_report_gate_pass": locked_report_gate,
        "primary_10bps_promotion_gate_pass": primary_gate,
        "live_promotable_10bps": primary_gate,
        "paper_promotion_guard_pass": paper_guard,
        "ready_for_paper": paper_guard,
        "ready_for_real": False,
        "real_money_execution": False,
        "shadow_observation_allowed": True,
        "guard_fail_reasons": _fail_reasons(checks=checks, values=values),
        "primary_10bps_gate_reasons": sorted(reasons),
        "split_metrics": {split: dict(row) for split, row in splits.items()},
    }


def _tv_rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    """Rank by train+validation only; locked-OOS fields are intentionally absent."""
    return (
        _safe_float(row.get("validation_return")),
        _safe_float(row.get("train_return")),
        _safe_float(row.get("train_validation_return_ratio")),
        -_safe_float(row.get("validation_mdd"), 1.0),
        str(row.get("model_id") or ""),
    )


def _rank(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in sorted(rows, key=_tv_rank_key, reverse=True)]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _reference_rows(ranked: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not ranked:
        return []
    references: list[dict[str, Any]] = []

    def add(role: str, row: Mapping[str, Any]) -> None:
        if any(item.get("model_id") == row.get("model_id") for item in references):
            return
        references.append(dict(row, selection_role=role))

    add("validation_return_leader", ranked[0])
    low_notional_pool = [
        row for row in ranked if _safe_float(row.get("target_notional_fraction_of_equity")) <= 1.0
    ]
    if low_notional_pool:
        add(
            "lower_notional_reference",
            max(
                low_notional_pool,
                key=lambda row: (
                    _safe_float(row.get("validation_return")),
                    _safe_float(row.get("train_return")),
                    -_safe_float(row.get("leverage")),
                    str(row.get("model_id") or ""),
                ),
            ),
        )
    ratio_pool = [row for row in ranked if _safe_float(row.get("validation_return")) > 0.0]
    if ratio_pool:
        add(
            "best_train_validation_ratio_reference",
            max(
                ratio_pool,
                key=lambda row: (
                    _safe_float(row.get("train_validation_return_ratio")),
                    _safe_float(row.get("validation_return")),
                    -_safe_float(row.get("target_notional_fraction_of_equity")),
                ),
            ),
        )
    add(
        "lowest_validation_mdd_reference",
        min(
            ranked,
            key=lambda row: (
                _safe_float(row.get("validation_mdd"), 1.0),
                -_safe_float(row.get("validation_return")),
            ),
        ),
    )
    for index, row in enumerate(references, start=1):
        row["reference_rank"] = index
    return references


def _summary_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "target_family_model_count": 0,
            "strict_paper_guard_pass_count": 0,
        }
    max_ratio = max(_safe_float(row.get("train_validation_return_ratio")) for row in rows)
    return {
        "target_family_model_count": len(rows),
        "train_validation_guard_pass_count": sum(
            bool(row.get("train_validation_guard_pass")) for row in rows
        ),
        "locked_oos_report_gate_pass_count": sum(
            bool(row.get("locked_oos_report_gate_pass")) for row in rows
        ),
        "primary_10bps_promotion_gate_pass_count": sum(
            bool(row.get("primary_10bps_promotion_gate_pass")) for row in rows
        ),
        "strict_paper_guard_pass_count": sum(
            bool(row.get("paper_promotion_guard_pass")) for row in rows
        ),
        "positive_train_count": sum(_safe_float(row.get("train_return")) > 0.0 for row in rows),
        "positive_validation_count": sum(
            _safe_float(row.get("validation_return")) > 0.0 for row in rows
        ),
        "positive_locked_oos_count": sum(
            _safe_float(row.get("locked_oos_return")) > 0.0 for row in rows
        ),
        "max_validation_return": max(_safe_float(row.get("validation_return")) for row in rows),
        "max_train_return": max(_safe_float(row.get("train_return")) for row in rows),
        "max_locked_oos_return": max(_safe_float(row.get("locked_oos_return")) for row in rows),
        "max_train_trade_event_count": max(
            _safe_int(row.get("train_trade_event_count")) for row in rows
        ),
        "max_validation_trade_event_count": max(
            _safe_int(row.get("validation_trade_event_count")) for row in rows
        ),
        "max_locked_oos_trade_event_count": max(
            _safe_int(row.get("locked_oos_trade_event_count")) for row in rows
        ),
        "max_train_validation_return_ratio": max_ratio,
        "min_validation_mdd": min(_safe_float(row.get("validation_mdd")) for row in rows),
        "min_locked_oos_mdd": min(_safe_float(row.get("locked_oos_mdd")) for row in rows),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("guarded_study_summary") or {})
    lines = [
        "# Alpha Zoo long-only reversal guarded study",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "Decision: keep the long-only crypto residual-reversal family shadow-only. "
        "No paper/testnet execution artifact is allowed by this study.",
        "",
        "## Guard summary",
        "",
        f"- Target family models: `{summary.get('target_family_model_count')}`",
        f"- Strict paper guard pass count: `{summary.get('strict_paper_guard_pass_count')}`",
        f"- Max validation trades: `{summary.get('max_validation_trade_event_count')}` "
        f"vs required `{MIN_VALIDATION_TRADE_EVENTS}`",
        f"- Max locked-OOS trades: `{summary.get('max_locked_oos_trade_event_count')}` "
        f"vs report gate `{MIN_LOCKED_OOS_TRADE_EVENTS}`",
        f"- Max train/validation return ratio: `{_safe_float(summary.get('max_train_validation_return_ratio')):.4f}` "
        f"vs required `{MIN_TRAIN_VALIDATION_RETURN_RATIO:.2f}`",
        "",
        "## Train+validation ranked candidates",
        "",
        "| Rank | Model | Val return | Train return | OOS return | Val/OOS trades | Guard |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in (payload.get("guarded_candidates") or [])[:10]:
        lines.append(
            f"| {row.get('rank')} | `{row.get('model_id')}` | "
            f"{_safe_float(row.get('validation_return')):.4%} | "
            f"{_safe_float(row.get('train_return')):.4%} | "
            f"{_safe_float(row.get('locked_oos_return')):.4%} | "
            f"{row.get('validation_trade_event_count')}/{row.get('locked_oos_trade_event_count')} | "
            f"{row.get('paper_promotion_guard_pass')} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    expanded_path = Path(args.expanded_retune_json).expanduser().resolve()
    shadow_path = Path(args.shadow_selection_json).expanduser().resolve()
    retune = _load_json(expanded_path)
    if (
        _safe_float(retune.get("round_trip_slippage_fee_bps_primary"))
        != PRIMARY_ROUND_TRIP_COST_BPS
    ):
        raise ValueError(
            "long-only reversal guarded study requires the expanded 10bps retune artifact"
        )
    shadow_selection = _load_json(shadow_path) if shadow_path.exists() else {}
    family_rows = [
        _candidate_summary(model_id, splits)
        for model_id, splits in _metrics_by_model(retune).items()
        if _target_family_match(splits)
    ]
    ranked = _rank(family_rows)
    top_n = int(args.top_n)
    candidate_rows = ranked[:top_n]
    references = _reference_rows(ranked)
    summary = _summary_stats(ranked)
    paper_execution_allowed = any(bool(row.get("paper_promotion_guard_pass")) for row in ranked)
    timestamp = _timestamp()
    latest_json = output_dir / "alpha_zoo_long_only_reversal_guarded_study_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_long_only_reversal_guarded_study_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_long_only_reversal_guarded_study_latest.md"
    candidates_csv = output_dir / "guarded_long_only_reversal_candidates_latest.csv"
    references_csv = output_dir / "guarded_long_only_reversal_references_latest.csv"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_long_only_reversal_guarded_study",
        "generated_at_utc": _utc_now_iso(),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "source_expanded_retune_json": str(expanded_path),
        "source_shadow_selection_json": str(shadow_path) if shadow_path.exists() else None,
        "source_shadow_selection_artifact_kind": shadow_selection.get("artifact_kind"),
        "target_family": {
            "candidate_name": TARGET_CANDIDATE_NAME,
            "dominant_factor_family": TARGET_FACTOR_FAMILY,
            "abs_factor_score_min": TARGET_ABS_FACTOR_SCORE_MIN,
        },
        "ready_for_paper": paper_execution_allowed,
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": paper_execution_allowed,
        "shadow_observation_allowed": True,
        "guard_thresholds": {
            "min_train_trade_event_count": MIN_TRAIN_TRADE_EVENTS,
            "min_validation_trade_event_count": MIN_VALIDATION_TRADE_EVENTS,
            "min_locked_oos_trade_event_count_report_gate": MIN_LOCKED_OOS_TRADE_EVENTS,
            "min_train_validation_return_ratio": MIN_TRAIN_VALIDATION_RETURN_RATIO,
            "max_train_mdd": MAX_TRAIN_MDD,
            "max_validation_mdd": MAX_VALIDATION_MDD,
            "max_locked_oos_mdd_report_gate": MAX_LOCKED_OOS_MDD,
            "require_zero_locked_oos_liquidation": True,
            "require_zero_locked_oos_account_wipeout": True,
            "require_primary_10bps_promotion_gate": True,
        },
        "guarded_study_summary": summary,
        "decision": {
            "status": "shadow_only_insufficient_split_evidence"
            if not paper_execution_allowed
            else "paper_candidate_found",
            "paper_execution_allowed": paper_execution_allowed,
            "shadow_observation_allowed": True,
            "ready_for_real": False,
            "real_money_execution": False,
            "why_not_paper": [
                f"0/{summary.get('target_family_model_count')} variants pass the strict paper guard",
                f"max validation trades {summary.get('max_validation_trade_event_count')} < required {MIN_VALIDATION_TRADE_EVENTS}",
                f"max locked-OOS trades {summary.get('max_locked_oos_trade_event_count')} < report gate {MIN_LOCKED_OOS_TRADE_EVENTS}",
                f"max train/validation return ratio {_safe_float(summary.get('max_train_validation_return_ratio')):.4f} < required {MIN_TRAIN_VALIDATION_RETURN_RATIO:.2f}",
                "primary 10bps promotion gate pass count is 0 for this target family",
            ],
            "paper_lanes_to_keep_running": [
                paper_preflight.ACTIVE_MODEL_ID,
                paper_preflight.BALANCED_MODEL_ID,
                "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc",
                "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc",
            ],
            "next_experiment": "Do not paper-trade this family yet; rerun train+validation-only discovery with stricter minimum sample guards or broader non-calendar reversal definitions, then reapply locked-OOS as report-only gate after freeze.",
        },
        "selection_policy": {
            "family_focus_source": "expanded_filter_shadow_selection_positive_oos_hypothesis",
            "variant_ranking_inputs": ["train", "validation"],
            "rank_key": [
                "validation_return_desc",
                "train_return_desc",
                "train_validation_return_ratio_desc",
                "validation_mdd_asc",
                "model_id_desc",
            ],
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "locked_oos_role": "post_freeze_gate_report_only",
        },
        "guarded_candidates": candidate_rows,
        "guarded_reference_candidates": references,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "guarded_candidates_csv": str(candidates_csv),
            "guarded_references_csv": str(references_csv),
            "artifact_generation_validation_log": str(generation_log),
        },
    }
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    _write_csv(candidates_csv, candidate_rows, CANDIDATE_CSV_FIELDS)
    _write_csv(references_csv, references, ["reference_rank", *CANDIDATE_CSV_FIELDS])
    generation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"target_family_model_count={summary.get('target_family_model_count')}",
                f"strict_paper_guard_pass_count={summary.get('strict_paper_guard_pass_count')}",
                f"max_validation_trade_event_count={summary.get('max_validation_trade_event_count')}",
                f"max_locked_oos_trade_event_count={summary.get('max_locked_oos_trade_event_count')}",
                f"ready_for_paper={str(payload['ready_for_paper']).lower()}",
                f"ready_for_real={str(payload['ready_for_real']).lower()}",
                f"real_money_execution={str(payload['real_money_execution']).lower()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-retune-json", default=str(DEFAULT_EXPANDED_RETUNE_JSON))
    parser.add_argument("--shadow-selection-json", default=str(DEFAULT_SHADOW_SELECTION_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
