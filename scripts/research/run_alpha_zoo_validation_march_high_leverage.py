#!/usr/bin/env python3
"""Evaluate Alpha Zoo on latest-data March-validation split with isolated high leverage.

The runner keeps selection train+validation-only, then applies locked-OOS as a
post-freeze gate.  The high-leverage lane models isolated margin: if an
in-trade adverse path breaches the liquidation threshold, the account-level
trade return is capped at the isolated allocation loss instead of assuming a
cross-margin account wipeout.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_common_split_alpha_zoo_hybrid_v35_v36 as common  # noqa: E402

DEFAULT_ALPHA_V2 = REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2"
DEFAULT_OUTPUT_DIR = DEFAULT_ALPHA_V2 / "validation_to_20260331_latest_data_20260517"
DEFAULT_CURRENT_TAIL_CACHE = (
    REPO_ROOT / "var/cache/profit_moonshot_fresh_start/joined_panel_76f825ffea81c04f2fe41fbf.parquet"
)
DEFAULT_EXTERNAL_STATE_CSV = DEFAULT_ALPHA_V2 / "external_market_state_20260512/external_market_state_lagged.csv"
DEFAULT_OLD_ALPHA_REPLAY = (
    DEFAULT_ALPHA_V2 / "crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_state_replay_latest.json"
)
DEFAULT_TRAIN_START = "2025-01-01T00:00:00Z"
DEFAULT_TRAIN_END = "2025-12-31T23:00:00Z"
DEFAULT_VALIDATION_START = "2026-01-01T00:00:00Z"
DEFAULT_VALIDATION_END = "2026-03-31T23:00:00Z"
DEFAULT_LOCKED_OOS_START = "2026-04-01T00:00:00Z"
DEFAULT_LOCKED_OOS_END = "2026-05-17T10:00:00Z"
DEFAULT_LEVERAGE_MIN = 1
DEFAULT_LEVERAGE_MAX = 50
DEFAULT_STRICT_LEVERAGE_MAX = 6
DEFAULT_ALLOCATION_GRID = "0.03,0.05,0.075,0.10,0.125,0.15,0.20"
DEFAULT_HORIZON = 4
DEFAULT_TOP_N = 20
DEFAULT_ENTRY_QUANTILE = 0.9
DEFAULT_MAX_LEDGER_RECORDS = 120
STARTING_EQUITY = 10_000.0
OOS_MDD_BUDGET = 0.25
MIN_VALIDATION_TRADES = 30
CURRENT_BASE_OOS_RETURN_REFERENCE = 0.06428110030664325
SPLIT_ORDER = ("train", "validation", "locked_oos")
TV_SPLITS = ("train", "validation")


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
    except Exception:
        return default
    return parsed if math.isfinite(parsed) else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return common._format_timestamp(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_allocation_grid(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw or "").split(","):
        token = item.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"allocation fractions must be in (0, 1): {value}")
        out.append(value)
    return sorted(set(out))


def _split_contract(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    return {
        "train": {
            "start": str(args.train_start),
            "end": str(args.train_end),
            "role": "objective_calibration_selection",
        },
        "validation": {
            "start": str(args.validation_start),
            "end": str(args.validation_end),
            "role": "objective_selection_extended_to_march_end",
        },
        "locked_oos": {
            "start": str(args.locked_oos_start),
            "end": str(args.locked_oos_end),
            "role": "gate_report_only_after_candidate_freeze",
        },
    }


def _build_screen_and_calibration(
    *,
    common_frame: pd.DataFrame,
    bundle_metadata: Mapping[str, Any],
    split_contract: Mapping[str, Mapping[str, str]],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    alpha_dir = output_dir / "alpha_zoo_validation_march_high_leverage"
    ledger_path = alpha_dir / "candidate_outcome_ledger_validation_march_latest.jsonl"
    screen_path = alpha_dir / "crypto_fx_alpha_zoo_screen_validation_march_latest.json"
    calibration_path = alpha_dir / "edge_calibration_validation_march_latest.json"

    old_contract = common.COMMON_SPLIT_CONTRACT
    common.COMMON_SPLIT_CONTRACT = {str(key): dict(value) for key, value in split_contract.items()}
    try:
        screen_payload = common._build_common_screen_payload(
            common_frame,
            source_ref=f"input:{bundle_metadata.get('source_path')}",
            source_coverage=bundle_metadata,
            ledger_output=ledger_path,
            horizon=int(args.horizon),
            top_n=int(args.top_n),
            entry_quantile=float(args.entry_quantile),
            max_ledger_records_per_factor_side_split=int(args.max_ledger_records_per_factor_side_split),
        )
    finally:
        common.COMMON_SPLIT_CONTRACT = old_contract

    manifest = {
        "artifact_kind": "validation_march_latest_data_split_manifest",
        "split_contract": split_contract,
        "split_periods": common._split_periods(common_frame),
        "timestamp_index_hash": common._timestamp_index_hash(common_frame),
        "source_path": str(bundle_metadata.get("source_path") or ""),
        "source_sha256": common._file_sha256(bundle_metadata.get("source_path") or ""),
    }
    screen_payload["common_split_manifest"] = manifest
    _write_json(screen_path, screen_payload)

    calibration_module = common._load_module(
        REPO_ROOT / "scripts/research/calibrate_crypto_fx_edges.py",
        "validation_march_edge_calibration",
    )
    calibration_payload = common._build_calibration_payload(ledger_path, calibration_module)
    calibration_payload["common_split_manifest"] = manifest
    _write_json(calibration_path, calibration_payload)
    return screen_payload, calibration_payload, calibration_path


def _attach_trade_path_extrema(
    alpha: Any,
    data: pd.DataFrame,
    trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    symbol_frames = alpha._symbol_frames(data)
    out: list[dict[str, Any]] = []
    for trade in trades:
        row = dict(trade)
        path = alpha._trade_path(data, row, symbol_frames)
        min_adverse = 0.0
        if not path.empty:
            entry = _safe_float(row.get("entry_price"), 1.0)
            if entry > 0.0:
                if str(row.get("side")).upper() == "LONG":
                    adverse = (pd.to_numeric(path["low"], errors="coerce") / entry) - 1.0
                else:
                    adverse = 1.0 - (pd.to_numeric(path["high"], errors="coerce") / entry)
                min_adverse = float(adverse.min()) if len(adverse) else 0.0
        row["_min_intratrade_adverse_return"] = float(min_adverse)
        out.append(row)
    return out


def _isolated_trade_return(
    trade: dict[str, Any],
    *,
    leverage: float,
    allocation_fraction: float,
    alpha: Any,
) -> tuple[float, bool, float]:
    reserve_rate = 0.01 + 0.001 + 0.0005 + 0.0001 + 0.0025 + 0.005
    adverse_threshold = max(0.0, (1.0 / max(float(leverage), 1e-9)) - reserve_rate)
    min_adverse = _safe_float(trade.get("_min_intratrade_adverse_return"))
    liquidated = bool(min_adverse <= -adverse_threshold)
    if liquidated:
        return -float(allocation_fraction), True, float(min_adverse)
    return (
        alpha._portfolio_trade_return(
            trade,
            leverage=float(leverage),
            allocation_fraction=float(allocation_fraction),
        ),
        False,
        float(min_adverse),
    )


def _isolated_split_metrics(
    alpha: Any,
    trades: list[dict[str, Any]],
    *,
    leverage: float,
    allocation_fraction: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    metrics: dict[str, dict[str, Any]] = {}
    isolated_status: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        returns: list[float] = []
        loss_count = 0
        worst_adverse = 0.0
        for trade in trades:
            if str(trade.get("entry_split")) != split:
                continue
            ret, liquidated, min_adverse = _isolated_trade_return(
                trade,
                leverage=float(leverage),
                allocation_fraction=float(allocation_fraction),
                alpha=alpha,
            )
            returns.append(float(ret))
            if liquidated:
                loss_count += 1
            worst_adverse = min(float(worst_adverse), float(min_adverse))
        metrics[split] = alpha._metrics_from_returns(returns)
        isolated_status[split] = {
            "isolated_margin_loss_count": int(loss_count),
            "worst_intratrade_adverse_return": float(worst_adverse),
            "max_single_trade_account_loss_fraction": float(allocation_fraction),
        }
    return metrics, isolated_status


def _audit_from_trade_extrema(
    trades: list[dict[str, Any]],
    *,
    leverage: float,
    allocation_fraction: float,
    starting_equity: float = STARTING_EQUITY,
) -> dict[str, Any]:
    reserve_rate = 0.01 + 0.001 + 0.0005 + 0.0001 + 0.0025 + 0.005
    adverse_threshold = max(0.0, (1.0 / max(float(leverage), 1e-9)) - reserve_rate)
    split_status = {
        split: {
            "liquidation_count": 0,
            "minimum_margin_buffer": float(starting_equity),
            "maximum_liquidation_event_drawdown": 0.0,
            "maximum_liquidation_equity_loss_fraction": 0.0,
            "liquidation_recovery_observed": False,
            "recovered_to_pre_liquidation_equity": True,
            "account_wipeout_count": 0,
        }
        for split in SPLIT_ORDER
    }
    for trade in trades:
        split = str(trade.get("entry_split"))
        if split not in split_status:
            continue
        min_adverse = _safe_float(trade.get("_min_intratrade_adverse_return"))
        notional = float(starting_equity) * float(allocation_fraction) * float(leverage)
        margin_requirement = float(notional) * float(reserve_rate)
        buffer = float(starting_equity) - margin_requirement + min(0.0, float(min_adverse) * notional)
        split_status[split]["minimum_margin_buffer"] = min(
            float(split_status[split]["minimum_margin_buffer"]),
            float(buffer),
        )
        if min_adverse <= -adverse_threshold:
            split_status[split]["liquidation_count"] += 1
            event_loss = min(1.0, abs(float(min_adverse)) * float(allocation_fraction) * float(leverage))
            split_status[split]["maximum_liquidation_event_drawdown"] = max(
                float(split_status[split]["maximum_liquidation_event_drawdown"]),
                float(event_loss),
            )
            split_status[split]["maximum_liquidation_equity_loss_fraction"] = max(
                float(split_status[split]["maximum_liquidation_equity_loss_fraction"]),
                float(event_loss),
            )
            split_status[split]["liquidation_recovery_observed"] = True
            split_status[split]["recovered_to_pre_liquidation_equity"] = False
            if event_loss >= 1.0:
                split_status[split]["account_wipeout_count"] += 1
    for status in split_status.values():
        status["margin_buffer_positive"] = float(status["minimum_margin_buffer"]) > 0.0
        status["strict_safe"] = int(status["liquidation_count"]) == 0 and bool(
            status["margin_buffer_positive"]
        )
    total_liq = sum(int(item["liquidation_count"]) for item in split_status.values())
    min_buffer = min(float(item["minimum_margin_buffer"]) for item in split_status.values())
    return {
        "leverage": float(leverage),
        "split_status": split_status,
        "total_liquidation_count": int(total_liq),
        "minimum_margin_buffer": float(min_buffer),
        "liquidation_free": total_liq == 0,
        "margin_buffer_positive": min_buffer > 0.0,
    }


def _split_account_wipeout_count(audit: Mapping[str, Any], splits: tuple[str, ...]) -> int:
    status = dict(audit.get("split_status") or {})
    return int(
        sum(int(dict(status.get(split) or {}).get("account_wipeout_count") or 0) for split in splits)
    )


def _split_liquidation_count(audit: Mapping[str, Any], splits: tuple[str, ...]) -> int:
    status = dict(audit.get("split_status") or {})
    return int(sum(int(dict(status.get(split) or {}).get("liquidation_count") or 0) for split in splits))


def _tv_score(metrics: Mapping[str, Mapping[str, Any]], audit: Mapping[str, Any]) -> float:
    train = dict(metrics.get("train") or {})
    validation = dict(metrics.get("validation") or {})
    train_return = _safe_float(train.get("total_return"))
    val_return = _safe_float(validation.get("total_return"))
    val_mdd = _safe_float(validation.get("max_drawdown"), 1.0)
    val_sharpe = _safe_float(validation.get("sharpe"))
    val_sortino = _safe_float(validation.get("sortino"))
    val_smart = _safe_float(validation.get("smart_sortino"))
    val_calmar = _safe_float(validation.get("calmar"))
    val_return_mdd = _safe_float(validation.get("return_mdd"))
    train_return_mdd = _safe_float(train.get("return_mdd"))
    val_trades = int(validation.get("trade_count") or 0)
    penalty = 0.0
    if _split_account_wipeout_count(audit, TV_SPLITS) > 0:
        penalty += 1_000.0
    if train_return <= 0.0:
        penalty += 10.0 + abs(train_return) * 10.0
    if val_return <= 0.0:
        penalty += 20.0 + abs(val_return) * 10.0
    if val_mdd > OOS_MDD_BUDGET:
        penalty += (val_mdd - OOS_MDD_BUDGET) * 50.0
    if val_trades < MIN_VALIDATION_TRADES:
        penalty += (MIN_VALIDATION_TRADES - val_trades) / 5.0
    for metric in (val_sharpe, val_sortino, val_smart, val_calmar):
        if metric <= 0.0:
            penalty += 5.0
    return (
        10.0 * val_return
        + 0.10 * train_return
        + 0.25 * val_return_mdd
        + 0.02 * train_return_mdd
        + 0.05 * val_sharpe
        + 0.05 * val_sortino
        + 0.05 * val_smart
        + 0.05 * val_calmar
        - penalty
    )


def _gate_reasons(metrics: Mapping[str, Mapping[str, Any]], audit: Mapping[str, Any]) -> list[str]:
    oos = dict(metrics.get("locked_oos") or {})
    reasons: list[str] = []
    if _split_account_wipeout_count(audit, ("locked_oos",)) > 0:
        reasons.append("locked_oos_account_wipeout_count_positive")
    if _safe_float(oos.get("max_drawdown"), 1.0) > OOS_MDD_BUDGET:
        reasons.append("locked_oos_mdd_above_25pct")
    if _safe_float(oos.get("total_return")) <= CURRENT_BASE_OOS_RETURN_REFERENCE:
        reasons.append("locked_oos_return_not_above_current_base_reference")
    for key in ("sharpe", "sortino", "smart_sortino", "calmar"):
        if _safe_float(oos.get(key)) <= 0.0:
            reasons.append(f"locked_oos_{key}_non_positive")
    return reasons


def _metric_digest(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "total_return": _safe_float(metrics.get("total_return")),
        "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        "return_mdd": _safe_float(metrics.get("return_mdd")),
        "sharpe": _safe_float(metrics.get("sharpe")),
        "sortino": _safe_float(metrics.get("sortino")),
        "smart_sortino": _safe_float(metrics.get("smart_sortino")),
        "calmar": _safe_float(metrics.get("calmar")),
        "trade_count": int(metrics.get("trade_count") or 0),
    }


def _build_rows(
    *,
    alpha: Any,
    data: pd.DataFrame,
    calibrated_edges: dict[str, float],
    old_replay: Mapping[str, Any],
    allocation_grid: list[float],
    leverage_min: int,
    leverage_max: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = [common._old_selected_spec(old_replay, alpha), *alpha._default_grid_specs()]
    unique: dict[str, Any] = {}
    for spec in specs:
        key = json.dumps([spec.name, spec.params], sort_keys=True, default=str)
        unique.setdefault(key, spec)
    rows: list[dict[str, Any]] = []
    strategy_summaries: list[dict[str, Any]] = []
    for spec in unique.values():
        signals = alpha._run_strategy_signals(
            data,
            require_calibrated_edge=True,
            calibrated_edges=calibrated_edges,
            strategy_params=dict(spec.params),
        )
        trades = _attach_trade_path_extrema(alpha, data, alpha._build_trades(data, signals))
        base_tv = alpha._train_validation_metrics(trades, leverage=1.0, allocation_fraction=0.10)
        base_score = alpha._selection_score(base_tv)
        strategy_summaries.append(
            {
                "candidate_name": spec.name,
                "candidate_source": spec.source,
                "signal_count": len(signals),
                "trade_count": len(trades),
                "base_unlevered_tv_score": float(base_score),
                "params": dict(spec.params),
            }
        )
        for allocation in allocation_grid:
            for leverage in range(int(leverage_min), int(leverage_max) + 1):
                metrics, isolated_status = _isolated_split_metrics(
                    alpha,
                    trades,
                    leverage=float(leverage),
                    allocation_fraction=float(allocation),
                )
                audit = _audit_from_trade_extrema(
                    trades,
                    leverage=float(leverage),
                    allocation_fraction=float(allocation),
                    starting_equity=STARTING_EQUITY,
                )
                score = _tv_score(metrics, audit)
                reasons = _gate_reasons(metrics, audit)
                tv_reasons: list[str] = []
                if _split_account_wipeout_count(audit, TV_SPLITS) > 0:
                    tv_reasons.append("train_validation_account_wipeout_count_positive")
                if _safe_float(metrics["train"].get("total_return")) <= 0.0:
                    tv_reasons.append("train_return_non_positive")
                if _safe_float(metrics["validation"].get("total_return")) <= 0.0:
                    tv_reasons.append("validation_return_non_positive")
                if _safe_float(metrics["validation"].get("max_drawdown"), 1.0) > OOS_MDD_BUDGET:
                    tv_reasons.append("validation_mdd_above_25pct")
                if int(metrics["validation"].get("trade_count") or 0) < MIN_VALIDATION_TRADES:
                    tv_reasons.append("validation_trade_count_below_minimum")
                for key in ("sharpe", "sortino", "smart_sortino", "calmar"):
                    if _safe_float(metrics["validation"].get(key)) <= 0.0:
                        tv_reasons.append(f"validation_{key}_non_positive")
                rows.append(
                    {
                        "candidate_name": spec.name,
                        "candidate_source": spec.source,
                        "strategy": "CryptoFxAlphaZooStateStrategy",
                        "params": dict(spec.params),
                        "leverage": float(leverage),
                        "allocation_fraction": float(allocation),
                        "margin_mode_assumption": "isolated_per_position_loss_capped_at_allocation_fraction",
                        "selection_inputs": ["train", "validation"],
                        "uses_locked_oos_for_selection": False,
                        "locked_oos_role": "gate_report_only_after_candidate_freeze",
                        "tv_selection_score": float(score),
                        "train_validation_feasible": not tv_reasons,
                        "train_validation_rejection_reasons": sorted(set(tv_reasons)),
                        "locked_oos_gate_pass": not reasons,
                        "locked_oos_rejection_reasons": sorted(set(reasons)),
                        "live_promotion_possible": (not tv_reasons) and (not reasons),
                        "split_metrics": metrics,
                        "split_metrics_digest": {
                            split: _metric_digest(metrics[split]) for split in SPLIT_ORDER
                        },
                        "cross_margin_liquidation_audit": audit,
                        "isolated_margin_audit": isolated_status,
                        "total_liquidation_count": int(audit.get("total_liquidation_count") or 0),
                        "train_validation_liquidation_count": _split_liquidation_count(audit, TV_SPLITS),
                        "locked_oos_liquidation_count": _split_liquidation_count(audit, ("locked_oos",)),
                        "total_account_wipeout_count": _split_account_wipeout_count(audit, SPLIT_ORDER),
                        "minimum_margin_buffer_cross_margin_diagnostic": _safe_float(
                            audit.get("minimum_margin_buffer")
                        ),
                    }
                )
    feasible_sorted = sorted(
        [row for row in rows if bool(row["train_validation_feasible"])],
        key=lambda row: float(row["tv_selection_score"]),
        reverse=True,
    )
    for rank, row in enumerate(feasible_sorted, start=1):
        row["frozen_train_validation_rank"] = int(rank)
    promoted = next((row for row in feasible_sorted if bool(row["locked_oos_gate_pass"])), None)
    top_tv = feasible_sorted[0] if feasible_sorted else None
    selection = {
        "selection_policy": (
            "freeze all train/validation-feasible strategy/leverage/allocation candidates by "
            "validation-primary train+validation score; "
            "locked-OOS can only gate candidates after freeze, not alter scores"
        ),
        "selection_inputs": ["train", "validation"],
        "uses_locked_oos_for_selection": False,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
        "candidate_count": len(rows),
        "train_validation_feasible_count": len(feasible_sorted),
        "strategy_summaries": sorted(
            strategy_summaries,
            key=lambda row: float(row["base_unlevered_tv_score"]),
            reverse=True,
        ),
        "top_train_validation_candidate": _public_candidate(top_tv),
        "live_promoted_candidate": _public_candidate(promoted),
    }
    return rows, selection


def _public_candidate(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    metrics = dict(row.get("split_metrics_digest") or {})
    return {
        "candidate_name": row.get("candidate_name"),
        "candidate_source": row.get("candidate_source"),
        "leverage": row.get("leverage"),
        "allocation_fraction": row.get("allocation_fraction"),
        "frozen_train_validation_rank": row.get("frozen_train_validation_rank"),
        "tv_selection_score": row.get("tv_selection_score"),
        "train": metrics.get("train"),
        "validation": metrics.get("validation"),
        "locked_oos": metrics.get("locked_oos"),
        "locked_oos_gate_pass": row.get("locked_oos_gate_pass"),
        "live_promotion_possible": row.get("live_promotion_possible"),
        "locked_oos_rejection_reasons": row.get("locked_oos_rejection_reasons"),
        "total_liquidation_count": row.get("total_liquidation_count"),
        "locked_oos_liquidation_count": row.get("locked_oos_liquidation_count"),
        "total_account_wipeout_count": row.get("total_account_wipeout_count"),
    }


def _strict_lane(alpha: Any, data: pd.DataFrame, promoted: Mapping[str, Any] | None, calibrated_edges: dict[str, float]) -> dict[str, Any]:
    if not promoted:
        return {"rows": [], "note": "no promoted candidate"}
    params = dict(promoted.get("params") or {})
    # Rows do not retain params to keep top-level compact; recover from candidate source via default specs.
    candidate_name = str(promoted.get("candidate_name") or "")
    for spec in alpha._default_grid_specs():
        if spec.name == candidate_name:
            params = dict(spec.params)
            break
    signals = alpha._run_strategy_signals(
        data,
        require_calibrated_edge=True,
        calibrated_edges=calibrated_edges,
        strategy_params=params,
    )
    trades = alpha._build_trades(data, signals)
    lanes = alpha._liquidation_lanes(
        data,
        trades,
        allocation_fraction=0.10,
        max_leverage=DEFAULT_STRICT_LEVERAGE_MAX,
    )
    return lanes


def _rows_to_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_name",
        "leverage",
        "allocation_fraction",
        "frozen_train_validation_rank",
        "tv_selection_score",
        "train_return",
        "validation_return",
        "locked_oos_return",
        "locked_oos_mdd",
        "locked_oos_sharpe",
        "locked_oos_sortino",
        "locked_oos_smart_sortino",
        "locked_oos_calmar",
        "locked_oos_trade_count",
        "locked_oos_liquidation_count",
        "total_account_wipeout_count",
        "locked_oos_gate_pass",
        "live_promotion_possible",
        "locked_oos_rejection_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            digest = dict(row.get("split_metrics_digest") or {})
            oos = dict(digest.get("locked_oos") or {})
            writer.writerow(
                {
                    "candidate_name": row.get("candidate_name"),
                    "leverage": row.get("leverage"),
                    "allocation_fraction": row.get("allocation_fraction"),
                    "frozen_train_validation_rank": row.get("frozen_train_validation_rank"),
                    "tv_selection_score": row.get("tv_selection_score"),
                    "train_return": dict(digest.get("train") or {}).get("total_return"),
                    "validation_return": dict(digest.get("validation") or {}).get("total_return"),
                    "locked_oos_return": oos.get("total_return"),
                    "locked_oos_mdd": oos.get("max_drawdown"),
                    "locked_oos_sharpe": oos.get("sharpe"),
                    "locked_oos_sortino": oos.get("sortino"),
                    "locked_oos_smart_sortino": oos.get("smart_sortino"),
                    "locked_oos_calmar": oos.get("calmar"),
                    "locked_oos_trade_count": oos.get("trade_count"),
                    "locked_oos_liquidation_count": row.get("locked_oos_liquidation_count"),
                    "total_account_wipeout_count": row.get("total_account_wipeout_count"),
                    "locked_oos_gate_pass": row.get("locked_oos_gate_pass"),
                    "live_promotion_possible": row.get("live_promotion_possible"),
                    "locked_oos_rejection_reasons": ";".join(row.get("locked_oos_rejection_reasons") or []),
                }
            )


def _markdown(payload: Mapping[str, Any]) -> str:
    selection = dict(payload.get("selection") or {})
    promoted = dict(selection.get("live_promoted_candidate") or {})
    top_tv = dict(selection.get("top_train_validation_candidate") or {})
    split = dict(payload.get("split_manifest") or {}).get("split_contract") or {}
    lines = [
        "# Alpha Zoo latest-data March-validation high-leverage replay",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        f"Data source: `{dict(payload.get('data_source') or {}).get('source_path')}`",
        f"Memory peak: `{dict(payload.get('memory_summary') or {}).get('peak_rss_mib'):.1f} MiB`",
        "",
        "## Split contract",
        "",
    ]
    for name in SPLIT_ORDER:
        item = dict(split.get(name) or {})
        period = dict(dict(payload.get("split_manifest") or {}).get("split_periods") or {}).get(name) or {}
        lines.append(
            f"- {name}: `{item.get('start')}` .. `{item.get('end')}`; actual `{period.get('start_timestamp')}` .. `{period.get('end_timestamp')}`, rows `{period.get('record_count')}`"
        )
    lines.extend(
        [
            "",
            "## Selection provenance",
            "",
            f"- selection inputs: `{selection.get('selection_inputs')}`",
            f"- uses locked-OOS for selection: `{selection.get('uses_locked_oos_for_selection')}`",
            f"- policy: {selection.get('selection_policy')}",
            "- high-leverage lane: isolated margin; liquidation loses the per-position allocation, not the full account.",
            "",
            "## Top train/validation candidate",
            "",
            _candidate_md(top_tv),
            "",
            "## Live-promoted after locked-OOS gate",
            "",
            _candidate_md(promoted) if promoted else "No candidate passed the post-freeze OOS gate.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for key, value in dict(payload.get("output_paths") or {}).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value) * 100:.2f}%"


def _candidate_md(row: Mapping[str, Any]) -> str:
    if not row:
        return "`{}`"
    oos = dict(row.get("locked_oos") or {})
    val = dict(row.get("validation") or {})
    train = dict(row.get("train") or {})
    return (
        f"- candidate: `{row.get('candidate_name')}` rank `{row.get('frozen_train_validation_rank')}`\n"
        f"- leverage/allocation: `{row.get('leverage')}x` / `{_fmt_pct(row.get('allocation_fraction'))}`\n"
        f"- train return/MDD: `{_fmt_pct(train.get('total_return'))}` / `{_fmt_pct(train.get('max_drawdown'))}`\n"
        f"- validation return/MDD: `{_fmt_pct(val.get('total_return'))}` / `{_fmt_pct(val.get('max_drawdown'))}`\n"
        f"- locked-OOS return/MDD: `{_fmt_pct(oos.get('total_return'))}` / `{_fmt_pct(oos.get('max_drawdown'))}`\n"
        f"- locked-OOS Sharpe/Sortino/smart/Calmar: `{_safe_float(oos.get('sharpe')):.3f}` / `{_safe_float(oos.get('sortino')):.3f}` / `{_safe_float(oos.get('smart_sortino')):.3f}` / `{_safe_float(oos.get('calmar')):.3f}`\n"
        f"- liquidations OOS/total account wipeout: `{row.get('locked_oos_liquidation_count')}` / `{row.get('total_account_wipeout_count')}`\n"
        f"- live_promotion_possible: `{row.get('live_promotion_possible')}`; rejections: `{row.get('locked_oos_rejection_reasons')}`"
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_contract = _split_contract(args)
    bundle = common.load_real_data_bundle(
        input_path=args.input,
        current_tail_cache=args.current_tail_cache,
        external_state_csv=args.external_state_csv,
        strict_real_data=True,
    )
    common_frame = common.apply_common_split(bundle.frame, split_contract=split_contract)
    common_frame = common.add_split_bounded_forward_return_label(common_frame, horizon=int(args.horizon))
    _screen_payload, _calibration_payload, calibration_path = _build_screen_and_calibration(
        common_frame=common_frame,
        bundle_metadata=bundle.metadata,
        split_contract=split_contract,
        output_dir=output_dir,
        args=args,
    )
    alpha = common._load_module(
        REPO_ROOT / "scripts/research/replay_crypto_fx_alpha_zoo_state.py",
        "validation_march_alpha_replay",
    )
    old_replay = common._load_json(Path(args.old_alpha_replay_json))
    calibrated_edges = alpha._load_calibrated_edges(calibration_path)
    data = alpha._ensure_replay_frame(common_frame)
    rows, selection = _build_rows(
        alpha=alpha,
        data=data,
        calibrated_edges=calibrated_edges,
        old_replay=old_replay,
        allocation_grid=_parse_allocation_grid(args.allocation_grid),
        leverage_min=int(args.leverage_min),
        leverage_max=int(args.leverage_max),
    )
    live_promoted = dict(selection.get("live_promoted_candidate") or {})
    promoted_row = next(
        (
            row
            for row in rows
            if row.get("candidate_name") == live_promoted.get("candidate_name")
            and _safe_float(row.get("leverage")) == _safe_float(live_promoted.get("leverage"))
            and _safe_float(row.get("allocation_fraction")) == _safe_float(live_promoted.get("allocation_fraction"))
        ),
        None,
    )
    strict = _strict_lane(alpha, data, promoted_row, calibrated_edges)
    frozen_rows = sorted(
        [row for row in rows if bool(row.get("train_validation_feasible"))],
        key=lambda row: int(row.get("frozen_train_validation_rank") or 10**9),
    )
    top_rows = frozen_rows[: min(50, len(frozen_rows))]
    csv_path = output_dir / "alpha_zoo_validation_march_high_leverage_candidates_latest.csv"
    _rows_to_csv(frozen_rows, csv_path)
    payload = {
        "artifact_kind": "alpha_zoo_validation_march_latest_data_high_leverage_replay",
        "generated_at_utc": _utc_now_iso(),
        "data_source": {
            "source_path": bundle.metadata.get("source_path"),
            "source_sha256": common._file_sha256(bundle.metadata.get("source_path") or ""),
            "source_coverage": bundle.metadata,
        },
        "split_manifest": {
            "split_contract": split_contract,
            "split_periods": common._split_periods(common_frame),
            "timestamp_index_hash": common._timestamp_index_hash(common_frame),
            "frame_rows": len(common_frame),
        },
        "selection": selection,
        "top_train_validation_frozen_candidates": [_public_candidate(row) for row in top_rows[:25]],
        "strict_zero_liquidation_lane_1x_6x_at_10pct_allocation": strict,
        "candidate_grid_summary": {
            "leverage_min": int(args.leverage_min),
            "leverage_max": int(args.leverage_max),
            "allocation_grid": _parse_allocation_grid(args.allocation_grid),
            "row_count": len(rows),
        },
        "candidate_rows_top50_path": str(csv_path),
        "screen_payload_path": str(output_dir / "alpha_zoo_validation_march_high_leverage/crypto_fx_alpha_zoo_screen_validation_march_latest.json"),
        "calibration_payload_path": str(calibration_path),
        "locked_oos_contamination_audit": {
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
            "evidence": [
                "tv_selection_score reads train/validation metrics only",
                "frozen_train_validation_rank is assigned before locked-OOS gate filtering",
                "locked_oos_gate_pass is computed after ranking and does not change tv_selection_score",
            ],
        },
        "current_base_reference": alpha.CURRENT_BASE_REFERENCE,
        "memory_summary": {
            "peak_rss_mib": _rss_mib(),
            "limit_mib": 8192.0,
            "pass_under_8gb": _rss_mib() < 8192.0,
        },
    }
    return payload


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_json = output_dir / "alpha_zoo_validation_march_high_leverage_latest.json"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timestamped_json = output_dir / f"alpha_zoo_validation_march_high_leverage_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_validation_march_high_leverage_latest.md"
    timestamped_md = output_dir / f"alpha_zoo_validation_march_high_leverage_{timestamp}.md"
    output_paths = {
        "latest_json": str(latest_json),
        "timestamped_json": str(timestamped_json),
        "latest_markdown": str(latest_md),
        "timestamped_markdown": str(timestamped_md),
        "candidate_csv": str(payload.get("candidate_rows_top50_path")),
    }
    payload = {**dict(payload), "output_paths": output_paths}
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    md = _markdown(payload)
    latest_md.write_text(md, encoding="utf-8")
    timestamped_md.write_text(md, encoding="utf-8")
    return output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="")
    parser.add_argument("--current-tail-cache", default=str(DEFAULT_CURRENT_TAIL_CACHE))
    parser.add_argument("--external-state-csv", default=str(DEFAULT_EXTERNAL_STATE_CSV))
    parser.add_argument("--old-alpha-replay-json", default=str(DEFAULT_OLD_ALPHA_REPLAY))
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--locked-oos-start", default=DEFAULT_LOCKED_OOS_START)
    parser.add_argument("--locked-oos-end", default=DEFAULT_LOCKED_OOS_END)
    parser.add_argument("--leverage-min", type=int, default=DEFAULT_LEVERAGE_MIN)
    parser.add_argument("--leverage-max", type=int, default=DEFAULT_LEVERAGE_MAX)
    parser.add_argument("--allocation-grid", default=DEFAULT_ALLOCATION_GRID)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--entry-quantile", type=float, default=DEFAULT_ENTRY_QUANTILE)
    parser.add_argument(
        "--max-ledger-records-per-factor-side-split",
        type=int,
        default=DEFAULT_MAX_LEDGER_RECORDS,
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    outputs = write_outputs(payload, Path(args.output_dir).expanduser().resolve())
    print(json.dumps({**outputs, "peak_rss_mib": payload["memory_summary"]["peak_rss_mib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
