#!/usr/bin/env python3
"""Run sample-guarded Alpha Zoo discovery from frozen 10bps retune evidence.

This runner is intentionally post-retune and governance-first.  It consumes the
expanded 10bps Alpha Zoo retune artifact, freezes train+validation ranking
profiles, and only then attaches locked-OOS as a gate/report-only field.  It is
not a real-money or live-execution runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
from collections import Counter
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
DEFAULT_LONG_ONLY_GUARDED_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_long_only_reversal_guarded_study_20260520"
    / "alpha_zoo_long_only_reversal_guarded_study_latest.json"
)
DEFAULT_EXPANDED_SHADOW_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_expanded_filter_shadow_selection_20260520"
    / "alpha_zoo_expanded_filter_shadow_selection_latest.json"
)
DEFAULT_FOUR_LANE_JSON = (
    DEFAULT_ALPHA_V2
    / "alpha_zoo_four_lane_shadow_discovery_20260520"
    / "alpha_zoo_four_lane_shadow_discovery_latest.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ALPHA_V2 / "alpha_zoo_sample_guarded_alpha_discovery_20260520"

PRIMARY_ROUND_TRIP_COST_BPS = 10.0
DEFAULT_AVG_BBO_SPREAD_BPS_ASSUMPTION = PRIMARY_ROUND_TRIP_COST_BPS / 5.0
DEFAULT_BBO_SPREAD_MULTIPLIER = 5.0
MEMORY_LIMIT_MIB = 8192.0
MIN_TRAIN_TRADES = 80
MIN_VALIDATION_TRADES = 30
MIN_LOCKED_OOS_TRADES = 20
MIN_VALIDATION_RETURN = 0.02
MIN_TRAIN_VALIDATION_RETURN_RATIO = 0.50
MAX_VALIDATION_MDD = 0.12
NOTIONAL_PARITY_EQUITY = 10_000.0

BASELINE_LANES = [
    {
        "role": "active",
        "model_id": paper_preflight.ACTIVE_MODEL_ID,
        "leverage": 7.0,
        "allocation_fraction": 0.20,
    },
    {
        "role": "balanced",
        "model_id": paper_preflight.BALANCED_MODEL_ID,
        "leverage": 6.0,
        "allocation_fraction": 0.175,
    },
    {
        "role": "validation_return_leader",
        "model_id": "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc",
        "leverage": 5.0,
        "allocation_fraction": 0.20,
    },
    {
        "role": "validation_efficiency_reference",
        "model_id": "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc",
        "leverage": 4.0,
        "allocation_fraction": 0.175,
    },
]

ASYMMETRY_GATE_REASONS = {
    "train_total_return_not_above_validation",
    "train_sharpe_not_above_validation",
    "train_sortino_not_above_validation",
    "train_smart_sortino_not_above_validation",
    "train_calmar_not_above_validation",
    "validation_mdd_above_train",
    "locked_oos_total_return_not_above_validation",
    "locked_oos_sharpe_not_above_validation",
    "locked_oos_sortino_not_above_validation",
    "locked_oos_smart_sortino_not_above_validation",
    "locked_oos_calmar_not_above_validation",
}

CANDIDATE_FIELDS = [
    "selection_rank",
    "status",
    "decision",
    "model_id",
    "candidate_name",
    "model_kind",
    "role",
    "variant_name",
    "leverage",
    "allocation_fraction",
    "target_notional_fraction_of_equity",
    "expected_replay_notional_for_10000_equity",
    "live_notional_for_10000_equity",
    "notional_parity_passed",
    "train_return",
    "train_mdd",
    "train_sharpe",
    "train_sortino",
    "train_smart_sortino",
    "train_calmar",
    "train_trade_event_count",
    "train_turnover_proxy",
    "train_return_per_turnover_proxy_bps",
    "train_return_per_turnover_proxy_pass",
    "validation_return",
    "validation_mdd",
    "validation_sharpe",
    "validation_sortino",
    "validation_smart_sortino",
    "validation_calmar",
    "validation_trade_event_count",
    "validation_turnover_proxy",
    "validation_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_pass",
    "train_validation_return_ratio",
    "locked_oos_return",
    "locked_oos_mdd",
    "locked_oos_sharpe",
    "locked_oos_sortino",
    "locked_oos_smart_sortino",
    "locked_oos_calmar",
    "locked_oos_trade_event_count",
    "locked_oos_turnover_proxy",
    "locked_oos_return_per_turnover_proxy_bps",
    "locked_oos_return_per_turnover_proxy_pass",
    "locked_oos_liquidation_count",
    "locked_oos_account_wipeout_count",
    "avg_bbo_spread_bps_assumption",
    "bbo_spread_multiplier",
    "return_per_turnover_proxy_threshold_bps",
    "execution_efficiency_proxy_gate_pass",
    "selection_eligible",
    "calendar_quarantined",
    "historical_oos_bucket_quarantined",
    "primary_10bps_promotion_gate_pass",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "rejection_reasons",
    "primary_10bps_gate_reasons",
    "trade_filter_params",
]

DECISION_FIELDS = [
    "decision_rank",
    "decision",
    "status",
    "model_id",
    "candidate_name",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "validation_return",
    "train_return",
    "locked_oos_return",
    "train_trade_event_count",
    "validation_trade_event_count",
    "locked_oos_trade_event_count",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "locked_oos_return_per_turnover_proxy_bps",
    "return_per_turnover_proxy_threshold_bps",
    "execution_efficiency_proxy_gate_pass",
    "replay_live_notional_parity",
    "rejection_reasons",
]

SHADOW_FIELDS = [
    "shadow_rank",
    "shadow_family",
    *CANDIDATE_FIELDS,
]

COST_FIELDS = [
    "rank",
    "model_id",
    "candidate_name",
    "round_trip_cost_bps",
    "split",
    "total_return",
    "max_drawdown",
    "trade_event_count",
    "metric_source",
    "diagnostic_only",
    "may_reduce_promotion_cost",
    "note",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_json_if_exists(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return _load_json(resolved) if resolved.exists() else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(high._json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.dumps(high._json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return high._json_safe(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
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
        if _safe_float(row.get("round_trip_slippage_fee_bps"), PRIMARY_ROUND_TRIP_COST_BPS) != PRIMARY_ROUND_TRIP_COST_BPS:
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


def _all_split_primary_gate_pass(splits: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        _as_bool(dict(splits[split]).get("primary_10bps_promotion_gate_pass"))
        and _as_bool(dict(splits[split]).get("live_promotable_10bps"))
        for split in high.SPLIT_ORDER
    )


def _split_value(row: Mapping[str, Any], key: str) -> float:
    return _safe_float(row.get(key))


def _return_per_turnover_threshold_bps(avg_bbo_spread_bps: float, multiplier: float) -> float:
    return avg_bbo_spread_bps * multiplier


def _turnover_proxy(trade_event_count: int, target_notional_fraction_of_equity: float) -> float:
    """Approximate notional turnover when exact BBO/turnover telemetry is absent."""
    return max(float(trade_event_count) * abs(target_notional_fraction_of_equity), 0.0)


def _return_per_turnover_proxy_bps(
    *,
    total_return: float,
    trade_event_count: int,
    target_notional_fraction_of_equity: float,
) -> float:
    turnover = _turnover_proxy(trade_event_count, target_notional_fraction_of_equity)
    if turnover <= 0.0:
        return 0.0
    return total_return * 10_000.0 / turnover


def _variant_inventory_summary(retune: Mapping[str, Any]) -> dict[str, Any]:
    inventory = [row for row in retune.get("candidate_variant_inventory") or [] if isinstance(row, Mapping)]
    source_counts = Counter(str(row.get("source") or "") for row in inventory)
    side_values: set[str] = set()
    symbol_values: set[str] = set()
    family_values: set[str] = set()
    threshold_values: set[float] = set()
    calendar_quarantine_count = 0
    for row in inventory:
        if _as_bool(row.get("calendar_primary")):
            calendar_quarantine_count += 1
        raw_params = row.get("params_json")
        try:
            params = json.loads(raw_params) if isinstance(raw_params, str) else dict(raw_params or {})
        except (TypeError, ValueError, json.JSONDecodeError):
            params = {}
        if params.get("side"):
            side_values.add(str(params["side"]))
        if params.get("symbol"):
            symbol_values.add(str(params["symbol"]))
        if params.get("dominant_factor_family"):
            family_values.add(str(params["dominant_factor_family"]))
        if params.get("abs_factor_score_min") is not None:
            threshold_values.add(_safe_float(params.get("abs_factor_score_min")))
    return {
        "variant_inventory_rows": len(inventory),
        "source_counts": dict(sorted(source_counts.items())),
        "calendar_quarantine_count": calendar_quarantine_count,
        "side_values_in_selected_metric_surface": sorted(side_values),
        "symbol_values_in_selected_metric_surface": sorted(symbol_values),
        "factor_families_in_selected_metric_surface": sorted(family_values),
        "abs_factor_score_min_values_in_selected_metric_surface": sorted(threshold_values),
        "symbol_grid_note": (
            "The upstream retune grid can evaluate symbol filters, but no symbol-filtered variant survived "
            "the train+validation selected metric/gate-pass surface used by this post-freeze runner."
        ),
    }


def _rejection_reasons(checks: Mapping[str, bool], values: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not checks["selection_eligible"]:
        if values.get("calendar_quarantined"):
            reasons.append("calendar_primary_or_calendar_rule_quarantine")
        if values.get("historical_oos_bucket_quarantined"):
            reasons.append("historical_oos_bucket_source_shadow_only")
    if not checks["train_trade_count"]:
        reasons.append(f"train_trade_event_count_{values['train_trades']}_below_{MIN_TRAIN_TRADES}")
    if not checks["validation_trade_count"]:
        reasons.append(f"validation_trade_event_count_{values['validation_trades']}_below_{MIN_VALIDATION_TRADES}")
    if not checks["locked_oos_trade_count"]:
        reasons.append(f"locked_oos_trade_event_count_{values['locked_oos_trades']}_below_{MIN_LOCKED_OOS_TRADES}")
    if not checks["validation_return"]:
        reasons.append(f"validation_return_{_safe_float(values['validation_return']):.4f}_below_{MIN_VALIDATION_RETURN:.2f}")
    if not checks["train_return"]:
        reasons.append("train_return_not_positive")
    if not checks["train_validation_return_ratio"]:
        reasons.append(
            f"train_validation_return_ratio_{_safe_float(values['train_validation_return_ratio']):.4f}_below_"
            f"{MIN_TRAIN_VALIDATION_RETURN_RATIO:.2f}"
        )
    if not checks["train_return_per_turnover_proxy"]:
        reasons.append(
            f"train_return_per_turnover_proxy_bps_"
            f"{_safe_float(values['train_return_per_turnover_proxy_bps']):.3f}_not_above_"
            f"{_safe_float(values['return_per_turnover_proxy_threshold_bps']):.3f}"
        )
    if not checks["validation_return_per_turnover_proxy"]:
        reasons.append(
            f"validation_return_per_turnover_proxy_bps_"
            f"{_safe_float(values['validation_return_per_turnover_proxy_bps']):.3f}_not_above_"
            f"{_safe_float(values['return_per_turnover_proxy_threshold_bps']):.3f}"
        )
    if not checks["validation_mdd"]:
        reasons.append(f"validation_mdd_{_safe_float(values['validation_mdd']):.4f}_above_{MAX_VALIDATION_MDD:.2f}")
    if not checks["locked_oos_return"]:
        reasons.append("locked_oos_return_not_positive")
    if not checks["locked_oos_return_per_turnover_proxy"]:
        reasons.append(
            f"locked_oos_return_per_turnover_proxy_bps_"
            f"{_safe_float(values['locked_oos_return_per_turnover_proxy_bps']):.3f}_not_above_"
            f"{_safe_float(values['return_per_turnover_proxy_threshold_bps']):.3f}"
        )
    if not checks["locked_oos_no_liquidation"]:
        reasons.append("locked_oos_liquidation_count_nonzero")
    if not checks["locked_oos_no_account_wipeout"]:
        reasons.append("locked_oos_account_wipeout_count_nonzero")
    if not checks["primary_10bps_promotion_gate"]:
        reasons.append("primary_10bps_promotion_gate_failed")
    return reasons


def _status_from_checks(
    *,
    checks: Mapping[str, bool],
    primary_gate_reasons: Sequence[str],
    rejection_reasons: Sequence[str],
) -> str:
    if not checks["selection_eligible"]:
        return "reject_or_quarantine"
    if checks["paper_candidate"]:
        return "paper_candidate"
    if not checks["validation_trade_count"] or not checks["locked_oos_trade_count"]:
        return "shadow_only_thin_sample"
    non_gate_checks = [
        name
        for name in (
            "train_trade_count",
            "validation_trade_count",
            "locked_oos_trade_count",
            "validation_return",
            "train_return",
            "train_validation_return_ratio",
            "train_return_per_turnover_proxy",
            "validation_return_per_turnover_proxy",
            "validation_mdd",
            "locked_oos_return",
            "locked_oos_return_per_turnover_proxy",
            "locked_oos_no_liquidation",
            "locked_oos_no_account_wipeout",
        )
    ]
    only_primary_gate_failed = all(checks[name] for name in non_gate_checks) and not checks[
        "primary_10bps_promotion_gate"
    ]
    reason_set = set(primary_gate_reasons)
    if only_primary_gate_failed and reason_set and reason_set <= ASYMMETRY_GATE_REASONS:
        return "paper_shadow_candidate"
    if "primary_10bps_promotion_gate_failed" in rejection_reasons and not any(
        reason.startswith("validation_trade_event_count_") or reason.startswith("locked_oos_trade_event_count_")
        for reason in rejection_reasons
    ):
        return "reject_or_quarantine"
    return "reject_or_quarantine"


def _candidate_summary(
    model_id: str,
    splits: Mapping[str, Mapping[str, Any]],
    *,
    avg_bbo_spread_bps_assumption: float = DEFAULT_AVG_BBO_SPREAD_BPS_ASSUMPTION,
    bbo_spread_multiplier: float = DEFAULT_BBO_SPREAD_MULTIPLIER,
) -> dict[str, Any]:
    train = dict(splits["train"])
    validation = dict(splits["validation"])
    locked = dict(splits["locked_oos"])
    leverage = _safe_float(validation.get("leverage"))
    allocation = _safe_float(validation.get("allocation_fraction"))
    target_notional = leverage * allocation
    train_return = _split_value(train, "total_return")
    validation_return = _split_value(validation, "total_return")
    locked_return = _split_value(locked, "total_return")
    train_trades = _safe_int(train.get("trade_event_count"))
    validation_trades = _safe_int(validation.get("trade_event_count"))
    locked_trades = _safe_int(locked.get("trade_event_count"))
    ratio = train_return / validation_return if validation_return > 0.0 else 0.0
    threshold_bps = _return_per_turnover_threshold_bps(avg_bbo_spread_bps_assumption, bbo_spread_multiplier)
    train_turnover_proxy = _turnover_proxy(train_trades, target_notional)
    validation_turnover_proxy = _turnover_proxy(validation_trades, target_notional)
    locked_turnover_proxy = _turnover_proxy(locked_trades, target_notional)
    train_return_per_turnover_proxy_bps = _return_per_turnover_proxy_bps(
        total_return=train_return,
        trade_event_count=train_trades,
        target_notional_fraction_of_equity=target_notional,
    )
    validation_return_per_turnover_proxy_bps = _return_per_turnover_proxy_bps(
        total_return=validation_return,
        trade_event_count=validation_trades,
        target_notional_fraction_of_equity=target_notional,
    )
    locked_return_per_turnover_proxy_bps = _return_per_turnover_proxy_bps(
        total_return=locked_return,
        trade_event_count=locked_trades,
        target_notional_fraction_of_equity=target_notional,
    )
    calendar_quarantined = any(_as_bool(dict(splits[split]).get("calendar_primary")) for split in high.SPLIT_ORDER)
    historical_oos_bucket = any(
        _as_bool(dict(splits[split]).get("candidate_universe_uses_locked_oos_bucket")) for split in high.SPLIT_ORDER
    )
    primary_gate = _all_split_primary_gate_pass(splits)
    primary_reasons = sorted({reason for row in (train, validation, locked) for reason in _gate_reasons(row)})
    values = {
        "calendar_quarantined": calendar_quarantined,
        "historical_oos_bucket_quarantined": historical_oos_bucket,
        "train_trades": train_trades,
        "validation_trades": validation_trades,
        "locked_oos_trades": locked_trades,
        "validation_return": validation_return,
        "train_validation_return_ratio": ratio,
        "validation_mdd": _split_value(validation, "max_drawdown"),
        "train_return_per_turnover_proxy_bps": train_return_per_turnover_proxy_bps,
        "validation_return_per_turnover_proxy_bps": validation_return_per_turnover_proxy_bps,
        "locked_oos_return_per_turnover_proxy_bps": locked_return_per_turnover_proxy_bps,
        "return_per_turnover_proxy_threshold_bps": threshold_bps,
    }
    checks: dict[str, bool] = {
        "selection_eligible": not calendar_quarantined and not historical_oos_bucket,
        "train_trade_count": train_trades >= MIN_TRAIN_TRADES,
        "validation_trade_count": validation_trades >= MIN_VALIDATION_TRADES,
        "locked_oos_trade_count": locked_trades >= MIN_LOCKED_OOS_TRADES,
        "validation_return": validation_return >= MIN_VALIDATION_RETURN,
        "train_return": train_return > 0.0,
        "train_validation_return_ratio": ratio >= MIN_TRAIN_VALIDATION_RETURN_RATIO,
        "train_return_per_turnover_proxy": train_return_per_turnover_proxy_bps > threshold_bps,
        "validation_return_per_turnover_proxy": validation_return_per_turnover_proxy_bps > threshold_bps,
        "validation_mdd": _split_value(validation, "max_drawdown") <= MAX_VALIDATION_MDD,
        "locked_oos_return": locked_return > 0.0,
        "locked_oos_return_per_turnover_proxy": locked_return_per_turnover_proxy_bps > threshold_bps,
        "locked_oos_no_liquidation": _split_value(locked, "liquidation_count") == 0.0,
        "locked_oos_no_account_wipeout": _split_value(locked, "account_wipeout_count") == 0.0,
        "primary_10bps_promotion_gate": primary_gate,
    }
    checks["paper_candidate"] = all(checks.values())
    rejection_reasons = _rejection_reasons(checks, values)
    status = _status_from_checks(
        checks=checks,
        primary_gate_reasons=primary_reasons,
        rejection_reasons=rejection_reasons,
    )
    ready_for_paper = status == "paper_candidate"
    return {
        "selection_rank": 0,
        "status": status,
        "decision": "paper_testnet_candidate" if ready_for_paper else "not_promoted_shadow_or_reject",
        "model_id": model_id,
        "candidate_name": validation.get("candidate_name"),
        "model_kind": validation.get("model_kind"),
        "role": validation.get("role"),
        "variant_name": validation.get("variant_name"),
        "trade_filter_params": dict(validation.get("trade_filter_params") or {}),
        "leverage": leverage,
        "allocation_fraction": allocation,
        "target_notional_fraction_of_equity": target_notional,
        "expected_replay_notional_for_10000_equity": NOTIONAL_PARITY_EQUITY * target_notional,
        "live_notional_for_10000_equity": NOTIONAL_PARITY_EQUITY * target_notional,
        "notional_parity_passed": True,
        "train_return": train_return,
        "train_mdd": _split_value(train, "max_drawdown"),
        "train_sharpe": _split_value(train, "sharpe"),
        "train_sortino": _split_value(train, "sortino"),
        "train_smart_sortino": _split_value(train, "smart_sortino"),
        "train_calmar": _split_value(train, "calmar"),
        "train_trade_event_count": train_trades,
        "train_turnover_proxy": train_turnover_proxy,
        "train_return_per_turnover_proxy_bps": train_return_per_turnover_proxy_bps,
        "train_return_per_turnover_proxy_pass": checks["train_return_per_turnover_proxy"],
        "validation_return": validation_return,
        "validation_mdd": _split_value(validation, "max_drawdown"),
        "validation_sharpe": _split_value(validation, "sharpe"),
        "validation_sortino": _split_value(validation, "sortino"),
        "validation_smart_sortino": _split_value(validation, "smart_sortino"),
        "validation_calmar": _split_value(validation, "calmar"),
        "validation_trade_event_count": validation_trades,
        "validation_turnover_proxy": validation_turnover_proxy,
        "validation_return_per_turnover_proxy_bps": validation_return_per_turnover_proxy_bps,
        "validation_return_per_turnover_proxy_pass": checks["validation_return_per_turnover_proxy"],
        "train_validation_return_ratio": ratio,
        "locked_oos_return": locked_return,
        "locked_oos_mdd": _split_value(locked, "max_drawdown"),
        "locked_oos_sharpe": _split_value(locked, "sharpe"),
        "locked_oos_sortino": _split_value(locked, "sortino"),
        "locked_oos_smart_sortino": _split_value(locked, "smart_sortino"),
        "locked_oos_calmar": _split_value(locked, "calmar"),
        "locked_oos_trade_event_count": locked_trades,
        "locked_oos_turnover_proxy": locked_turnover_proxy,
        "locked_oos_return_per_turnover_proxy_bps": locked_return_per_turnover_proxy_bps,
        "locked_oos_return_per_turnover_proxy_pass": checks["locked_oos_return_per_turnover_proxy"],
        "locked_oos_liquidation_count": _split_value(locked, "liquidation_count"),
        "locked_oos_account_wipeout_count": _split_value(locked, "account_wipeout_count"),
        "avg_bbo_spread_bps_assumption": avg_bbo_spread_bps_assumption,
        "bbo_spread_multiplier": bbo_spread_multiplier,
        "return_per_turnover_proxy_threshold_bps": threshold_bps,
        "execution_efficiency_proxy_gate_pass": (
            checks["train_return_per_turnover_proxy"]
            and checks["validation_return_per_turnover_proxy"]
            and checks["locked_oos_return_per_turnover_proxy"]
        ),
        "guard_checks": checks,
        "selection_eligible": checks["selection_eligible"],
        "calendar_quarantined": calendar_quarantined,
        "historical_oos_bucket_quarantined": historical_oos_bucket,
        "primary_10bps_promotion_gate_pass": primary_gate,
        "live_promotable_10bps": primary_gate,
        "ready_for_paper": ready_for_paper,
        "ready_for_real": False,
        "real_money_execution": False,
        "rejection_reasons": rejection_reasons,
        "primary_10bps_gate_reasons": primary_reasons,
        "split_metrics": {split: dict(row) for split, row in splits.items()},
    }


def _validation_strength_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        10.0 * _safe_float(row.get("validation_return"))
        + 0.15 * _safe_float(row.get("validation_sharpe"))
        + 0.15 * _safe_float(row.get("validation_sortino"))
        + 0.10 * _safe_float(row.get("validation_calmar"))
        - 2.0 * _safe_float(row.get("validation_mdd")),
        _safe_float(row.get("validation_return")),
        _safe_float(row.get("train_return")),
        -_safe_float(row.get("validation_mdd")),
        str(row.get("model_id") or ""),
    )


def _robustness_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    trade_bonus = math.log1p(_safe_int(row.get("train_trade_event_count")) + _safe_int(row.get("validation_trade_event_count")))
    ratio = min(_safe_float(row.get("train_validation_return_ratio")), 2.0)
    return (
        8.0 * _safe_float(row.get("validation_return"))
        + 1.5 * _safe_float(row.get("train_return"))
        + 0.5 * ratio
        + 0.02 * trade_bonus
        - 2.0 * _safe_float(row.get("validation_mdd")),
        _safe_float(row.get("validation_return")),
        ratio,
        trade_bonus,
        str(row.get("model_id") or ""),
    )


def _cost_efficiency_key(row: Mapping[str, Any]) -> tuple[float, float, float, str]:
    validation_trades = max(_safe_int(row.get("validation_trade_event_count")), 1)
    notional = max(_safe_float(row.get("target_notional_fraction_of_equity")), 0.01)
    return (
        _safe_float(row.get("validation_return")) / (validation_trades * PRIMARY_ROUND_TRIP_COST_BPS * notional),
        _safe_float(row.get("validation_return")) / notional,
        -notional,
        str(row.get("model_id") or ""),
    )


def _execution_efficiency_proxy_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float, str]:
    train_proxy = _safe_float(row.get("train_return_per_turnover_proxy_bps"))
    validation_proxy = _safe_float(row.get("validation_return_per_turnover_proxy_bps"))
    return (
        min(train_proxy, validation_proxy),
        0.5 * (train_proxy + validation_proxy),
        _safe_float(row.get("validation_return")),
        _safe_float(row.get("train_return")),
        -_safe_float(row.get("validation_mdd")),
        str(row.get("model_id") or ""),
    )


def _rank_candidates(rows: Sequence[Mapping[str, Any]], key_name: str, limit: int) -> list[dict[str, Any]]:
    key_map = {
        "validation_strength_v1": _validation_strength_key,
        "train_validation_robustness_v1": _robustness_key,
        "cost_efficiency_v1": _cost_efficiency_key,
        "execution_efficiency_proxy_v1": _execution_efficiency_proxy_key,
    }
    ranked = [dict(row) for row in sorted(rows, key=key_map[key_name], reverse=True)[:limit]]
    for rank, row in enumerate(ranked, start=1):
        row["profile_id"] = key_name
        row["profile_rank"] = rank
    return ranked


def _all_ranked_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in sorted(rows, key=_robustness_key, reverse=True)]
    for index, row in enumerate(ranked, start=1):
        row["selection_rank"] = index
    return ranked


def _paper_decision_rows(candidates: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    paper = [row for row in candidates if row.get("status") == "paper_candidate"]
    source_rows = paper if paper else list(candidates)[:limit]
    decisions: list[dict[str, Any]] = []
    for rank, row in enumerate(source_rows, start=1):
        decisions.append(
            {
                "decision_rank": rank,
                "decision": "paper_testnet_only_handoff" if row.get("status") == "paper_candidate" else "no_promotion",
                "status": row.get("status"),
                "model_id": row.get("model_id"),
                "candidate_name": row.get("candidate_name"),
                "ready_for_paper": bool(row.get("ready_for_paper")),
                "ready_for_real": False,
                "real_money_execution": False,
                "validation_return": row.get("validation_return"),
                "train_return": row.get("train_return"),
                "locked_oos_return": row.get("locked_oos_return"),
                "train_trade_event_count": row.get("train_trade_event_count"),
                "validation_trade_event_count": row.get("validation_trade_event_count"),
                "locked_oos_trade_event_count": row.get("locked_oos_trade_event_count"),
                "train_return_per_turnover_proxy_bps": row.get("train_return_per_turnover_proxy_bps"),
                "validation_return_per_turnover_proxy_bps": row.get("validation_return_per_turnover_proxy_bps"),
                "locked_oos_return_per_turnover_proxy_bps": row.get("locked_oos_return_per_turnover_proxy_bps"),
                "return_per_turnover_proxy_threshold_bps": row.get("return_per_turnover_proxy_threshold_bps"),
                "execution_efficiency_proxy_gate_pass": bool(row.get("execution_efficiency_proxy_gate_pass")),
                "replay_live_notional_parity": bool(row.get("notional_parity_passed")),
                "rejection_reasons": row.get("rejection_reasons"),
            }
        )
    return decisions


def _shadow_family(row: Mapping[str, Any]) -> str:
    params = dict(row.get("trade_filter_params") or {})
    if row.get("candidate_name") == "alpha_zoo_high_confidence_long_only" and params.get(
        "dominant_factor_family"
    ) == "crypto_residual_reversal":
        return "long_only_crypto_residual_reversal_shadow"
    if row.get("status") == "paper_shadow_candidate":
        return "paper_shadow_candidate"
    if row.get("status") == "shadow_only_thin_sample":
        return "thin_sample_shadow"
    return "rejected_shadow_context"


def _shadow_rows(candidates: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    shadow_like = [
        row
        for row in candidates
        if row.get("status") in {"paper_shadow_candidate", "shadow_only_thin_sample"}
        or row.get("candidate_name") == "alpha_zoo_high_confidence_long_only"
    ]
    ranked = shadow_like[:limit]
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        rows.append({"shadow_rank": rank, "shadow_family": _shadow_family(row), **dict(row)})
    return rows


def _cost_sensitivity_rows(candidates: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(candidates[:limit], start=1):
        split_metrics = dict(row.get("split_metrics") or {})
        for cost_bps in (5.0, 10.0, 15.0, 20.0):
            for split in high.SPLIT_ORDER:
                metric = dict(split_metrics.get(split) or {})
                primary = cost_bps == PRIMARY_ROUND_TRIP_COST_BPS
                rows.append(
                    {
                        "rank": rank,
                        "model_id": row.get("model_id"),
                        "candidate_name": row.get("candidate_name"),
                        "round_trip_cost_bps": cost_bps,
                        "split": split,
                        "total_return": metric.get("total_return") if primary else None,
                        "max_drawdown": metric.get("max_drawdown") if primary else None,
                        "trade_event_count": metric.get("trade_event_count") if primary else None,
                        "metric_source": "expanded_retune_primary_10bps" if primary else "not_replayed_in_sample_guarded_runner",
                        "diagnostic_only": True,
                        "may_reduce_promotion_cost": False,
                        "note": (
                            "primary promotion cost; exact replay metric"
                            if primary
                            else "requires dedicated cost replay; non-primary costs cannot reduce 10bps promotion gate"
                        ),
                    }
                )
    return rows


def _memory_summary(source_memory: Mapping[str, Any]) -> dict[str, Any]:
    ru_maxrss = _safe_float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    own_peak_mib = ru_maxrss / 1024.0 if sys.platform != "darwin" else ru_maxrss / (1024.0 * 1024.0)
    source_peak = _safe_float(source_memory.get("peak_rss_mib"))
    peak = max(own_peak_mib, source_peak)
    return {
        "limit_mib": MEMORY_LIMIT_MIB,
        "runner_peak_rss_mib": own_peak_mib,
        "source_peak_rss_mib": source_peak,
        "peak_rss_mib": peak,
        "pass_under_8gb": peak < MEMORY_LIMIT_MIB,
        "guard_status": "pass" if peak < MEMORY_LIMIT_MIB else "fail",
        "pass_fail_reason": f"peak_rss_mib={peak:.3f} below limit_mib={MEMORY_LIMIT_MIB:.1f}"
        if peak < MEMORY_LIMIT_MIB
        else f"peak_rss_mib={peak:.3f} exceeded limit_mib={MEMORY_LIMIT_MIB:.1f}",
    }


def _status_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("status") or "") for row in rows)
    return {
        "candidate_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "paper_candidate_count": counts.get("paper_candidate", 0),
        "paper_shadow_candidate_count": counts.get("paper_shadow_candidate", 0),
        "shadow_only_thin_sample_count": counts.get("shadow_only_thin_sample", 0),
        "reject_or_quarantine_count": counts.get("reject_or_quarantine", 0),
        "selection_eligible_count": sum(bool(row.get("selection_eligible")) for row in rows),
        "calendar_quarantined_count": sum(bool(row.get("calendar_quarantined")) for row in rows),
        "historical_oos_bucket_quarantined_count": sum(bool(row.get("historical_oos_bucket_quarantined")) for row in rows),
        "execution_efficiency_proxy_gate_pass_count": sum(
            bool(row.get("execution_efficiency_proxy_gate_pass")) for row in rows
        ),
        "train_return_per_turnover_proxy_pass_count": sum(
            bool(row.get("train_return_per_turnover_proxy_pass")) for row in rows
        ),
        "validation_return_per_turnover_proxy_pass_count": sum(
            bool(row.get("validation_return_per_turnover_proxy_pass")) for row in rows
        ),
        "locked_oos_return_per_turnover_proxy_pass_count": sum(
            bool(row.get("locked_oos_return_per_turnover_proxy_pass")) for row in rows
        ),
        "max_validation_return": max((_safe_float(row.get("validation_return")) for row in rows), default=0.0),
        "max_validation_return_per_turnover_proxy_bps": max(
            (_safe_float(row.get("validation_return_per_turnover_proxy_bps")) for row in rows), default=0.0
        ),
        "max_validation_trade_event_count": max((_safe_int(row.get("validation_trade_event_count")) for row in rows), default=0),
        "max_locked_oos_trade_event_count": max((_safe_int(row.get("locked_oos_trade_event_count")) for row in rows), default=0),
        "max_train_validation_return_ratio": max(
            (_safe_float(row.get("train_validation_return_ratio")) for row in rows), default=0.0
        ),
    }


def _profile_metadata(profile_id: str) -> dict[str, Any]:
    formulas = {
        "validation_strength_v1": "validation return, validation Sharpe/Sortino/Calmar, validation MDD penalty",
        "train_validation_robustness_v1": "validation return plus train return, train/validation return ratio, train+validation sample reward, validation MDD penalty",
        "cost_efficiency_v1": "validation return per validation trade, 10bps cost and notional exposure, with lower notional preference",
        "execution_efficiency_proxy_v1": "minimum and average train+validation return-per-turnover proxy bps; locked-OOS excluded from ranking",
    }
    return {
        "profile_id": profile_id,
        "objective_inputs": ["train", "validation"],
        "selection_inputs": ["train", "validation"],
        "optimization_input_splits": ["train", "validation"],
        "parameter_fit_inputs": ["train", "validation"],
        "pruning_inputs": ["train", "validation"],
        "score_formula_inputs": ["train", "validation"],
        "score_formula": formulas[profile_id],
        "locked_oos_role": "gate_report_only_after_train_validation_profile_freeze",
        "uses_locked_oos_for_discovery": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "uses_locked_oos_for_correlation": False,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("sample_guarded_summary") or {})
    decision = dict(payload.get("decision") or {})
    execution_policy = dict(payload.get("execution_efficiency_policy") or {})
    lines = [
        "# Alpha Zoo sample-guarded alpha discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "This artifact is paper/testnet research only. `ready_for_real=false` and `real_money_execution=false`.",
        "Locked-OOS is attached only after train+validation profile ranking freezes.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.get('status')}`",
        f"- Paper candidate count: `{summary.get('paper_candidate_count')}`",
        f"- Shadow/thin sample count: `{summary.get('shadow_only_thin_sample_count')}`",
        f"- Reject/quarantine count: `{summary.get('reject_or_quarantine_count')}`",
        f"- Primary cost: `{payload.get('research_primary_round_trip_cost_bps')}` bps round-trip",
        f"- Return/turnover proxy threshold: `{execution_policy.get('return_per_turnover_proxy_threshold_bps')}` bps "
        f"(avg BBO spread assumption `{execution_policy.get('avg_bbo_spread_bps_assumption')}` x "
        f"`{execution_policy.get('bbo_spread_multiplier')}`)",
        f"- Execution-efficiency proxy pass count: `{summary.get('execution_efficiency_proxy_gate_pass_count')}`",
        "",
        "## Top train+validation-ranked candidates",
        "",
        "| Rank | Status | Model | Val return | Train return | OOS return | R/T proxy bps T/V/O | Trades T/V/O | Reasons |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in (payload.get("sample_guarded_candidates") or [])[:15]:
        reasons = ", ".join((row.get("rejection_reasons") or [])[:3])
        lines.append(
            f"| {row.get('selection_rank')} | {row.get('status')} | `{row.get('model_id')}` | "
            f"{_safe_float(row.get('validation_return')):.4%} | "
            f"{_safe_float(row.get('train_return')):.4%} | "
            f"{_safe_float(row.get('locked_oos_return')):.4%} | "
            f"{_safe_float(row.get('train_return_per_turnover_proxy_bps')):.3f}/"
            f"{_safe_float(row.get('validation_return_per_turnover_proxy_bps')):.3f}/"
            f"{_safe_float(row.get('locked_oos_return_per_turnover_proxy_bps')):.3f} | "
            f"{row.get('train_trade_event_count')}/{row.get('validation_trade_event_count')}/{row.get('locked_oos_trade_event_count')} | "
            f"{reasons} |"
        )
    lines.extend(
        [
            "",
            "## Baseline lanes preserved",
            "",
            "| Role | Model | Leverage | Allocation |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for lane in payload.get("baseline_paper_lanes") or []:
        lines.append(
            f"| {lane.get('role')} | `{lane.get('model_id')}` | "
            f"{_safe_float(lane.get('leverage')):.1f} | {_safe_float(lane.get('allocation_fraction')):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    expanded_path = Path(args.expanded_retune_json).expanduser().resolve()
    avg_bbo_spread_bps = _safe_float(args.avg_bbo_spread_bps_assumption)
    bbo_spread_multiplier = _safe_float(args.bbo_spread_multiplier)
    if avg_bbo_spread_bps <= 0.0:
        raise ValueError("--avg-bbo-spread-bps-assumption must be positive")
    if bbo_spread_multiplier <= 0.0:
        raise ValueError("--bbo-spread-multiplier must be positive")
    return_per_turnover_threshold_bps = _return_per_turnover_threshold_bps(
        avg_bbo_spread_bps,
        bbo_spread_multiplier,
    )
    retune = _load_json(expanded_path)
    if _safe_float(retune.get("round_trip_slippage_fee_bps_primary")) != PRIMARY_ROUND_TRIP_COST_BPS:
        raise ValueError("sample-guarded discovery requires a 10bps expanded retune artifact")
    if _as_bool(retune.get("real_money_execution")):
        raise ValueError("source retune artifact unexpectedly allows real-money execution")

    long_only = _load_json_if_exists(args.long_only_guarded_json)
    expanded_shadow = _load_json_if_exists(args.expanded_shadow_json)
    four_lane = _load_json_if_exists(args.four_lane_json)

    raw_rows = [
        _candidate_summary(
            model_id,
            splits,
            avg_bbo_spread_bps_assumption=avg_bbo_spread_bps,
            bbo_spread_multiplier=bbo_spread_multiplier,
        )
        for model_id, splits in _metrics_by_model(retune).items()
    ]
    selection_eligible_rows = [row for row in raw_rows if row.get("selection_eligible")]
    ranked_rows = _all_ranked_candidates(selection_eligible_rows)
    quarantined = [dict(row, selection_rank=0) for row in raw_rows if not row.get("selection_eligible")]
    candidates = [*ranked_rows, *quarantined]
    summary = _status_summary(candidates)
    paper_candidates = [row for row in candidates if row.get("status") == "paper_candidate"]
    status = "paper_candidates_found" if paper_candidates else "no_new_paper_promotion_shadow_shortlist"
    timestamp = _timestamp()

    latest_json = output_dir / "alpha_zoo_sample_guarded_alpha_discovery_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_sample_guarded_alpha_discovery_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_sample_guarded_alpha_discovery_latest.md"
    candidates_csv = output_dir / "sample_guarded_candidates_latest.csv"
    decisions_csv = output_dir / "paper_candidate_decisions_latest.csv"
    shadow_csv = output_dir / "shadow_hypotheses_latest.csv"
    cost_csv = output_dir / "cost_sensitivity_latest.csv"
    generation_log = output_dir / "artifact_generation_validation_latest.log"

    profile_ids = [
        "validation_strength_v1",
        "train_validation_robustness_v1",
        "cost_efficiency_v1",
        "execution_efficiency_proxy_v1",
    ]
    profile_rankings = {
        profile_id: _rank_candidates(selection_eligible_rows, profile_id, int(args.profile_top_n))
        for profile_id in profile_ids
    }
    baseline_from_artifact = list(four_lane.get("four_lane_paper_candidates") or [])
    baseline_lanes = baseline_from_artifact if baseline_from_artifact else [dict(row) for row in BASELINE_LANES]
    for lane in baseline_lanes:
        lane["ready_for_real"] = False
        lane["real_money_execution"] = False

    decision_rows = _paper_decision_rows(candidates, limit=int(args.decision_top_n))
    shadow_rows = _shadow_rows(candidates, limit=int(args.shadow_top_n))
    cost_rows = _cost_sensitivity_rows(candidates, limit=int(args.cost_top_n))
    memory = _memory_summary(dict(retune.get("memory_summary") or {}))

    prior_long_only_models = [
        row
        for row in candidates
        if row.get("candidate_name") == "alpha_zoo_high_confidence_long_only"
        and dict(row.get("trade_filter_params") or {}).get("dominant_factor_family") == "crypto_residual_reversal"
    ]

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_sample_guarded_alpha_discovery",
        "generated_at_utc": _utc_now_iso(),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "ready_for_paper": bool(paper_candidates),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": bool(paper_candidates),
        "paper_testnet_only": True,
        "source_expanded_retune_json": str(expanded_path),
        "source_long_only_guarded_json": str(Path(args.long_only_guarded_json).expanduser().resolve()),
        "source_expanded_shadow_json": str(Path(args.expanded_shadow_json).expanduser().resolve()),
        "source_four_lane_json": str(Path(args.four_lane_json).expanduser().resolve()),
        "source_artifact_kinds": {
            "expanded_retune": retune.get("artifact_kind"),
            "long_only_guarded": long_only.get("artifact_kind"),
            "expanded_shadow": expanded_shadow.get("artifact_kind"),
            "four_lane": four_lane.get("artifact_kind"),
        },
        "decision": {
            "status": status,
            "paper_candidate_count": len(paper_candidates),
            "ready_for_real": False,
            "real_money_execution": False,
            "if_no_paper_candidate": "preserve no-promotion shadow shortlist with exact rejection reasons",
            "paper_candidate_policy": "paper/testnet-only handoff; real-money remains prohibited",
        },
        "promotion_thresholds": {
            "min_train_trade_event_count": MIN_TRAIN_TRADES,
            "min_validation_trade_event_count": MIN_VALIDATION_TRADES,
            "min_locked_oos_trade_event_count_report_gate": MIN_LOCKED_OOS_TRADES,
            "min_validation_return": MIN_VALIDATION_RETURN,
            "require_train_return_positive": True,
            "min_train_validation_return_ratio": MIN_TRAIN_VALIDATION_RETURN_RATIO,
            "max_validation_mdd": MAX_VALIDATION_MDD,
            "require_locked_oos_return_positive_report_gate": True,
            "require_zero_locked_oos_liquidation": True,
            "require_zero_locked_oos_account_wipeout": True,
            "require_primary_10bps_promotion_gate": True,
            "avg_bbo_spread_bps_assumption": avg_bbo_spread_bps,
            "bbo_spread_multiplier": bbo_spread_multiplier,
            "min_return_per_turnover_proxy_bps": return_per_turnover_threshold_bps,
            "require_train_return_per_turnover_proxy_above_threshold": True,
            "require_validation_return_per_turnover_proxy_above_threshold": True,
            "require_locked_oos_return_per_turnover_proxy_above_threshold_report_gate": True,
        },
        "selection_policy": {
            "candidate_freeze_inputs": ["train", "validation"],
            "profile_ranking_inputs": ["train", "validation"],
            "optimization_input_splits": ["train", "validation"],
            "parameter_fit_inputs": ["train", "validation"],
            "pruning_inputs": ["train", "validation"],
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "uses_locked_oos_for_correlation": False,
            "locked_oos_role": "gate_report_only_after_train_validation_profile_freeze",
            "calendar_rule_policy": "calendar_primary rows are quarantined before ranking",
            "execution_efficiency_proxy_policy": (
                "return_per_turnover_proxy_bps is computed from train+validation only for ranking; "
                "locked-OOS proxy is attached after ranking as a report-only promotion gate"
            ),
        },
        "execution_efficiency_policy": {
            "actual_avg_bbo_spread_source": "not_available_in_10bps_expanded_retune_artifact",
            "actual_turnover_source": "not_available_in_10bps_expanded_retune_artifact",
            "avg_bbo_spread_bps_assumption": avg_bbo_spread_bps,
            "bbo_spread_multiplier": bbo_spread_multiplier,
            "return_per_turnover_proxy_threshold_bps": return_per_turnover_threshold_bps,
            "threshold_formula": "avg_bbo_spread_bps_assumption * bbo_spread_multiplier",
            "turnover_proxy_formula": "trade_event_count * abs(leverage * allocation_fraction)",
            "return_per_turnover_proxy_bps_formula": "total_return * 10000 / turnover_proxy",
            "profile_ranking_inputs": ["train", "validation"],
            "promotion_gate_inputs": ["train", "validation", "locked_oos_report_gate"],
            "locked_oos_role": "gate_report_only_after_train_validation_profile_freeze",
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "actual_bbo_required_before_real_money": True,
        },
        "selection_profiles": {profile_id: _profile_metadata(profile_id) for profile_id in profile_ids},
        "profile_rankings": profile_rankings,
        "grid_coverage": _variant_inventory_summary(retune),
        "sample_guarded_summary": summary,
        "sample_guarded_candidates": candidates,
        "paper_candidates": paper_candidates,
        "paper_candidate_decisions": decision_rows,
        "shadow_hypotheses": shadow_rows,
        "cost_sensitivity": cost_rows,
        "baseline_paper_lanes": baseline_lanes,
        "prior_shadow_findings": {
            "long_only_guarded_summary": long_only.get("guarded_study_summary"),
            "expanded_shadow_decision": expanded_shadow.get("decision"),
            "long_only_crypto_residual_reversal_model_count": len(prior_long_only_models),
            "long_only_crypto_residual_reversal_paper_candidate_count": sum(
                row.get("status") == "paper_candidate" for row in prior_long_only_models
            ),
            "keep_long_only_crypto_residual_reversal_shadow_only": not any(
                row.get("status") == "paper_candidate" for row in prior_long_only_models
            ),
        },
        "memory_summary": memory,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "sample_guarded_candidates_csv": str(candidates_csv),
            "paper_candidate_decisions_csv": str(decisions_csv),
            "shadow_hypotheses_csv": str(shadow_csv),
            "cost_sensitivity_csv": str(cost_csv),
            "artifact_generation_validation_log": str(generation_log),
        },
    }

    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    _write_csv(candidates_csv, candidates, CANDIDATE_FIELDS)
    _write_csv(decisions_csv, decision_rows, DECISION_FIELDS)
    _write_csv(shadow_csv, shadow_rows, SHADOW_FIELDS)
    _write_csv(cost_csv, cost_rows, COST_FIELDS)
    generation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"primary_round_trip_cost_bps={PRIMARY_ROUND_TRIP_COST_BPS}",
                f"candidate_count={summary['candidate_count']}",
                f"paper_candidate_count={summary['paper_candidate_count']}",
                f"avg_bbo_spread_bps_assumption={avg_bbo_spread_bps}",
                f"bbo_spread_multiplier={bbo_spread_multiplier}",
                f"return_per_turnover_proxy_threshold_bps={return_per_turnover_threshold_bps}",
                f"execution_efficiency_proxy_gate_pass_count={summary['execution_efficiency_proxy_gate_pass_count']}",
                f"ready_for_paper={str(payload['ready_for_paper']).lower()}",
                f"ready_for_real={str(payload['ready_for_real']).lower()}",
                f"real_money_execution={str(payload['real_money_execution']).lower()}",
                f"uses_locked_oos_for_selection={str(payload['selection_policy']['uses_locked_oos_for_selection']).lower()}",
                f"memory_guard_status={memory['guard_status']}",
                f"latest_json={latest_json}",
                f"timestamped_json={timestamped_json}",
                f"sample_guarded_candidates_csv={candidates_csv}",
                f"paper_candidate_decisions_csv={decisions_csv}",
                f"shadow_hypotheses_csv={shadow_csv}",
                f"cost_sensitivity_csv={cost_csv}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-retune-json", default=str(DEFAULT_EXPANDED_RETUNE_JSON))
    parser.add_argument("--long-only-guarded-json", default=str(DEFAULT_LONG_ONLY_GUARDED_JSON))
    parser.add_argument("--expanded-shadow-json", default=str(DEFAULT_EXPANDED_SHADOW_JSON))
    parser.add_argument("--four-lane-json", default=str(DEFAULT_FOUR_LANE_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--profile-top-n", type=int, default=25)
    parser.add_argument("--decision-top-n", type=int, default=25)
    parser.add_argument("--shadow-top-n", type=int, default=60)
    parser.add_argument("--cost-top-n", type=int, default=25)
    parser.add_argument("--avg-bbo-spread-bps-assumption", type=float, default=DEFAULT_AVG_BBO_SPREAD_BPS_ASSUMPTION)
    parser.add_argument("--bbo-spread-multiplier", type=float, default=DEFAULT_BBO_SPREAD_MULTIPLIER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
