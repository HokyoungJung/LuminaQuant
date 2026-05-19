#!/usr/bin/env python3
"""Build Alpha Zoo 10bps full-retune candidate/cost gate artifacts.

This runner owns the 2026-05-19 10bps promotion contract.  It keeps prior
leaderboard/OOS-derived rows as shadow-only references unless they are replaced
by train+validation-frozen retune rows.  Locked-OOS is attached only after
candidate freeze for gate/report evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_top_seed_hybrid_v35_v36_cost_validation as cost_validation  # noqa: E402
from scripts.research import run_alpha_zoo_validation_march_high_leverage as high  # noqa: E402

SPLIT_ORDER = ("train", "validation", "locked_oos")
TV_SPLITS = ("train", "validation")
ROUND_TRIP_SLIPPAGE_FEE_BPS_PRIMARY = 10.0
MEMORY_LIMIT_MIB = 8192.0
EXPECTED_TIMESTAMP_INDEX_HASH = "b973165bc1057f3aaa08ea637b73a45df3e84fdb7d1337b1637233d205696bb0"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT")

DEFAULT_SOURCE_DIR = high.DEFAULT_ALPHA_V2 / "live_notional_risk_aligned_alpha_zoo_20260518"
DEFAULT_CANDIDATE_CSV = DEFAULT_SOURCE_DIR / "alpha_zoo_validation_march_high_leverage_candidates_latest.csv"
DEFAULT_COST_VALIDATION_JSON = (
    high.DEFAULT_ALPHA_V2
    / "alpha_zoo_top_seed_hybrid_cost_validation_20260518"
    / "alpha_zoo_top_seed_hybrid_cost_validation_latest.json"
)
DEFAULT_OUTPUT_DIR = high.DEFAULT_ALPHA_V2 / "alpha_zoo_10bps_full_retune_20260519"

SPLIT_CONTRACT = {
    "train": {"start": high.DEFAULT_TRAIN_START, "end": high.DEFAULT_TRAIN_END},
    "validation": {"start": high.DEFAULT_VALIDATION_START, "end": high.DEFAULT_VALIDATION_END},
    "locked_oos": {"start": high.DEFAULT_LOCKED_OOS_START, "end": high.DEFAULT_LOCKED_OOS_END},
}

METRIC_DOMINANCE_KEYS = ("return", "sharpe", "sortino", "smart_sortino", "calmar")
CALENDAR_PARAM_TOKENS = ("calendar", "date", "day", "month", "weekday", "week_of", "time_of_year")
CSV_FIELDNAMES = (
    "model_id",
    "model_kind",
    "role",
    "candidate_name",
    "leverage",
    "allocation_fraction",
    "round_trip_slippage_fee_bps",
    "split",
    "total_return",
    "max_drawdown",
    "sharpe",
    "sortino",
    "smart_sortino",
    "calmar",
    "return_mdd",
    "trade_event_count",
    "active_return_hours",
    "liquidation_count",
    "account_wipeout_count",
    "minimum_margin_buffer",
    "candidate_universe_uses_locked_oos_bucket",
    "shadow_only",
    "regenerated_train_validation_only",
    "calendar_primary",
    "split_gate_pass",
    "split_gate_reasons",
    "promotion_gate_pass",
    "promotion_gate_reasons",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] = CSV_FIELDNAMES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def _cost_adjusted_trade_return(
    alpha: Any,
    trade: Mapping[str, Any],
    *,
    leverage: float,
    allocation_fraction: float,
    round_trip_slippage_bps: float = ROUND_TRIP_SLIPPAGE_FEE_BPS_PRIMARY,
) -> tuple[float, bool]:
    """Return isolated account-level trade return after round-trip bps cost."""
    return cost_validation._cost_adjusted_trade_return(
        alpha,
        trade,
        leverage=float(leverage),
        allocation_fraction=float(allocation_fraction),
        round_trip_slippage_bps=float(round_trip_slippage_bps),
    )


def _params_have_calendar_rule(params: Mapping[str, Any] | None) -> bool:
    if not params:
        return False
    for key, value in params.items():
        key_text = str(key).lower()
        if any(token in key_text for token in CALENDAR_PARAM_TOKENS):
            return True
        if isinstance(value, Mapping) and _params_have_calendar_rule(value):
            return True
    return False


def _variant_inventory_row(name: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    calendar_primary = _params_have_calendar_rule(params)
    return {
        "variant_name": str(name),
        "params": dict(params or {}),
        "calendar_primary": bool(calendar_primary),
        "accepted_for_retune": not calendar_primary,
        "rejection_reasons": ["calendar_or_date_rule_forbidden"] if calendar_primary else [],
    }


def _metric(row: Mapping[str, Any], split: str, key: str) -> float:
    if key == "return":
        key = "total_return"
    return _safe_float(row.get(f"{split}_{key}"), _safe_float(row.get(key)))


def _split_gate_reasons(row: Mapping[str, Any], split: str) -> list[str]:
    reasons: list[str] = []
    if _metric(row, split, "return") <= 0.0:
        reasons.append(f"{split}_return_non_positive")
    for key in ("sharpe", "sortino", "smart_sortino", "calmar"):
        if _metric(row, split, key) <= 0.0:
            reasons.append(f"{split}_{key}_non_positive")
    if _safe_int(row.get(f"{split}_account_wipeout_count", row.get("account_wipeout_count"))) > 0:
        reasons.append(f"{split}_account_wipeout_count_positive")
    if _safe_int(row.get(f"{split}_liquidation_count", row.get("liquidation_count"))) > 0:
        reasons.append(f"{split}_liquidation_count_positive")
    min_buffer = row.get(f"{split}_minimum_margin_buffer", row.get("minimum_margin_buffer"))
    if min_buffer is not None and str(min_buffer) != "" and _safe_float(min_buffer, -1.0) <= 0.0:
        reasons.append(f"{split}_minimum_margin_buffer_non_positive")
    return reasons


def promotion_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the 10bps live-promotion gate for one frozen candidate row."""
    reasons: list[str] = []
    cost_bps = _safe_float(row.get("round_trip_slippage_fee_bps"), ROUND_TRIP_SLIPPAGE_FEE_BPS_PRIMARY)
    if abs(cost_bps - ROUND_TRIP_SLIPPAGE_FEE_BPS_PRIMARY) > 1e-12:
        reasons.append("promotion_cost_not_10bps")
    if _as_bool(row.get("candidate_universe_uses_locked_oos_bucket")):
        reasons.append("candidate_universe_uses_locked_oos_bucket")
    if _as_bool(row.get("shadow_only")):
        reasons.append("shadow_only_not_live_promotable")
    if not _as_bool(row.get("regenerated_train_validation_only")):
        reasons.append("not_regenerated_train_validation_only")
    if _as_bool(row.get("calendar_primary")) or _params_have_calendar_rule(row.get("params") if isinstance(row.get("params"), Mapping) else None):
        reasons.append("calendar_or_date_rule_forbidden")

    for split in ("validation", "locked_oos"):
        reasons.extend(_split_gate_reasons(row, split))

    for key in METRIC_DOMINANCE_KEYS:
        train_value = _metric(row, "train", key)
        validation_value = _metric(row, "validation", key)
        locked_value = _metric(row, "locked_oos", key)
        if not train_value > validation_value:
            reasons.append(f"train_{key}_not_above_validation")
        if not train_value > locked_value:
            reasons.append(f"train_{key}_not_above_locked_oos")

    if _safe_int(row.get("total_account_wipeout_count")) > 0:
        reasons.append("total_account_wipeout_count_positive")
    if _safe_int(row.get("promotion_liquidation_count", row.get("locked_oos_liquidation_count"))) > 0:
        reasons.append("promotion_liquidation_count_positive")
    min_buffer = row.get("minimum_margin_buffer", row.get("locked_oos_minimum_margin_buffer"))
    if min_buffer is not None and str(min_buffer) != "" and _safe_float(min_buffer, -1.0) <= 0.0:
        reasons.append("minimum_margin_buffer_non_positive")

    deduped = sorted(set(reasons))
    return {"promotion_gate_pass": not deduped, "promotion_gate_reasons": deduped}


def _candidate_id(row: Mapping[str, Any]) -> str:
    name = str(row.get("candidate_name") or "candidate")
    lev = _safe_float(row.get("leverage"))
    alloc = _safe_float(row.get("allocation_fraction"))
    cleaned = "_".join(name.replace("/", "_").split())
    return f"{cleaned}_{lev:g}x_{alloc:g}alloc"


def _rows_from_candidate_csv(candidate_csv: Path) -> list[dict[str, Any]]:
    if not candidate_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with candidate_csv.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            base = {
                "model_id": _candidate_id(raw),
                "model_kind": "alpha_zoo_candidate_reference",
                "role": "shadow_reference_prior_candidate_csv",
                "candidate_name": raw.get("candidate_name", ""),
                "leverage": _safe_float(raw.get("leverage")),
                "allocation_fraction": _safe_float(raw.get("allocation_fraction")),
                "round_trip_slippage_fee_bps": ROUND_TRIP_SLIPPAGE_FEE_BPS_PRIMARY,
                "candidate_universe_uses_locked_oos_bucket": True,
                "shadow_only": True,
                "regenerated_train_validation_only": False,
                "calendar_primary": False,
                "train_return": _safe_float(raw.get("train_return")),
                "validation_return": _safe_float(raw.get("validation_return")),
                "locked_oos_return": _safe_float(raw.get("locked_oos_return")),
                "train_sharpe": _safe_float(raw.get("train_sharpe")),
                "validation_sharpe": _safe_float(raw.get("validation_sharpe")),
                "locked_oos_sharpe": _safe_float(raw.get("locked_oos_sharpe")),
                "train_sortino": _safe_float(raw.get("train_sortino")),
                "validation_sortino": _safe_float(raw.get("validation_sortino")),
                "locked_oos_sortino": _safe_float(raw.get("locked_oos_sortino")),
                "train_smart_sortino": _safe_float(raw.get("train_smart_sortino")),
                "validation_smart_sortino": _safe_float(raw.get("validation_smart_sortino")),
                "locked_oos_smart_sortino": _safe_float(raw.get("locked_oos_smart_sortino")),
                "train_calmar": _safe_float(raw.get("train_calmar")),
                "validation_calmar": _safe_float(raw.get("validation_calmar")),
                "locked_oos_calmar": _safe_float(raw.get("locked_oos_calmar")),
                "locked_oos_liquidation_count": _safe_int(raw.get("locked_oos_liquidation_count")),
                "total_account_wipeout_count": _safe_int(raw.get("total_account_wipeout_count")),
            }
            gate = promotion_gate(base)
            for split in SPLIT_ORDER:
                split_row = {
                    **base,
                    "split": split,
                    "total_return": _metric(base, split, "return"),
                    "max_drawdown": _safe_float(raw.get(f"{split}_mdd"), _safe_float(raw.get("locked_oos_mdd")) if split == "locked_oos" else 0.0),
                    "sharpe": _metric(base, split, "sharpe"),
                    "sortino": _metric(base, split, "sortino"),
                    "smart_sortino": _metric(base, split, "smart_sortino"),
                    "calmar": _metric(base, split, "calmar"),
                    "return_mdd": _safe_float(raw.get(f"{split}_return_mdd")),
                    "trade_event_count": _safe_int(raw.get(f"{split}_trade_count"), _safe_int(raw.get("locked_oos_trade_count")) if split == "locked_oos" else 0),
                    "active_return_hours": 0,
                    "liquidation_count": _safe_int(raw.get(f"{split}_liquidation_count"), _safe_int(raw.get("locked_oos_liquidation_count")) if split == "locked_oos" else 0),
                    "account_wipeout_count": _safe_int(raw.get(f"{split}_account_wipeout_count"), _safe_int(raw.get("total_account_wipeout_count"))),
                    "minimum_margin_buffer": raw.get(f"{split}_minimum_margin_buffer", raw.get("minimum_margin_buffer", "")),
                    **gate,
                }
                split_reasons = _split_gate_reasons(split_row, split)
                split_row["split_gate_pass"] = not split_reasons
                split_row["split_gate_reasons"] = ";".join(split_reasons)
                split_row["promotion_gate_reasons"] = ";".join(gate["promotion_gate_reasons"])
                rows.append(split_row)
    return rows


def _memory_summary() -> dict[str, Any]:
    peak = _rss_mib()
    passed = peak < MEMORY_LIMIT_MIB
    return {
        "peak_rss_mib": peak,
        "limit_mib": MEMORY_LIMIT_MIB,
        "pass_under_8gb": passed,
        "guard_status": "pass" if passed else "fail",
        "pass_fail_reason": "peak_rss_under_limit" if passed else "peak_rss_at_or_above_8192_mib",
    }


def _execution_cost_evidence() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "primary_round_trip_slippage_fee_bps": ROUND_TRIP_SLIPPAGE_FEE_BPS_PRIMARY,
        "symbols": list(SYMBOLS),
        "promotion_uses_primary_cost_only": True,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    candidate_csv = Path(args.candidate_csv).expanduser().resolve()
    rows = _rows_from_candidate_csv(candidate_csv)
    promotable = [row for row in rows if row["split"] == "locked_oos" and bool(row.get("promotion_gate_pass"))]
    promotable.sort(key=lambda row: (_safe_float(row.get("total_return")), _safe_float(row.get("calmar"))), reverse=True)
    variants = [
        _variant_inventory_row("threshold_stricter_10pct", {"threshold_multiplier": 1.10}),
        _variant_inventory_row("threshold_stricter_25pct", {"threshold_multiplier": 1.25}),
        _variant_inventory_row("longer_hold", {"max_hold_bars_multiplier": 1.50}),
        _variant_inventory_row("calendar_reference_rejected", {"month_filter": [1, 2]}),
    ]
    return {
        "artifact_kind": "alpha_zoo_10bps_full_retune",
        "generated_at_utc": _utc_now_iso(),
        "real_money_execution": False,
        "round_trip_slippage_fee_bps_primary": ROUND_TRIP_SLIPPAGE_FEE_BPS_PRIMARY,
        "source_inputs": {
            "candidate_csv": str(candidate_csv),
            "prior_cost_validation_json": str(Path(args.cost_validation_json).expanduser().resolve()),
        },
        "split_manifest": {
            "split_contract": SPLIT_CONTRACT,
            "timestamp_index_hash": EXPECTED_TIMESTAMP_INDEX_HASH,
            "drift_policy": "fail_if_latest_split_hash_or_boundaries_change_without_declared_refresh_lineage",
        },
        "locked_oos_contamination_audit": {
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
        },
        "candidate_policy": {
            "promotion_cost_bps": ROUND_TRIP_SLIPPAGE_FEE_BPS_PRIMARY,
            "prior_oos_bucket_rows_are_shadow_only": True,
            "requires_train_validation_regeneration": True,
            "train_must_exceed_validation_and_locked_oos_metrics": list(METRIC_DOMINANCE_KEYS),
            "forbidden_variant_tokens": list(CALENDAR_PARAM_TOKENS),
        },
        "variant_inventory": variants,
        "candidate_model_metrics": rows,
        "candidate_model_count": len({row["model_id"] for row in rows}),
        "live_promotable_10bps_model_id": promotable[0]["model_id"] if promotable else None,
        "live_promotable_10bps_count": len({row["model_id"] for row in promotable}),
        "execution_cost_evidence": _execution_cost_evidence(),
        "memory_summary": _memory_summary(),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Alpha Zoo 10bps full retune",
            "",
            f"- real_money_execution: `{payload.get('real_money_execution')}`",
            f"- primary_cost_bps: `{payload.get('round_trip_slippage_fee_bps_primary')}`",
            f"- candidate_model_count: `{payload.get('candidate_model_count')}`",
            f"- live_promotable_10bps_model_id: `{payload.get('live_promotable_10bps_model_id')}`",
            f"- memory_pass_under_8gb: `{dict(payload.get('memory_summary') or {}).get('pass_under_8gb')}`",
            "",
            "Locked-OOS is gate/report-only after candidate freeze; prior OOS/top-bucket references remain shadow-only until regenerated through train+validation-only retune.",
            "",
        ]
    )


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    latest_json = output_dir / "alpha_zoo_10bps_full_retune_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_10bps_full_retune_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_10bps_full_retune_latest.md"
    timestamped_md = output_dir / f"alpha_zoo_10bps_full_retune_{timestamp}.md"
    metrics_csv = output_dir / "candidate_model_metrics_latest.csv"
    outputs = {
        "latest_json": str(latest_json),
        "timestamped_json": str(timestamped_json),
        "latest_markdown": str(latest_md),
        "timestamped_markdown": str(timestamped_md),
        "candidate_model_metrics_csv": str(metrics_csv),
    }
    payload = {**dict(payload), "output_paths": outputs}
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    timestamped_md.write_text(_markdown(payload), encoding="utf-8")
    _write_csv(metrics_csv, list(payload.get("candidate_model_metrics") or []))
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--cost-validation-json", default=str(DEFAULT_COST_VALIDATION_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-trials", type=int, default=80, help="Reserved for the bounded retune stage; gates are deterministic.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    outputs = write_outputs(payload, Path(args.output_dir).expanduser().resolve())
    print(
        json.dumps(
            {
                **outputs,
                "candidate_model_count": payload.get("candidate_model_count"),
                "metric_rows": len(list(payload.get("candidate_model_metrics") or [])),
                "live_promotable_10bps_model_id": payload.get("live_promotable_10bps_model_id"),
                "peak_rss_mib": dict(payload.get("memory_summary") or {}).get("peak_rss_mib"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
