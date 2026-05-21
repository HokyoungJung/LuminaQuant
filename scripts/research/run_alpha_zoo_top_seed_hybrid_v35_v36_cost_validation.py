#!/usr/bin/env python3
"""Build Alpha Zoo top-seed Hybrid v3.5/v3.6 cost-validation artifacts.

This runner is intentionally separate from
``run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py``.  The fixed-input runner
owns the A0+P0+E0+S1+S2+S3+S4 contract; this file owns the post-hoc Alpha Zoo
leaderboard seed-union experiment requested for the 2026-05-18 cost validation.

The seed basket is a diagnostic research basket assembled from current
leaderboard buckets.  Hybrid parameter fitting, Optuna objective/pruning, and
live/deployable selection metadata remain train+validation only; locked-OOS is
reported after freeze as a gate/report split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_validation_march_high_leverage as high  # noqa: E402
from scripts.research import run_common_split_alpha_zoo_hybrid_v35_v36 as common  # noqa: E402
from scripts.research import run_profit_moonshot_hybrid_v35_v36_fixed_inputs as hybrid  # noqa: E402

DEFAULT_SOURCE_DIR = high.DEFAULT_ALPHA_V2 / "live_notional_risk_aligned_alpha_zoo_20260518"
DEFAULT_LIVE_JSON = DEFAULT_SOURCE_DIR / "live_notional_risk_aligned_alpha_zoo_latest.json"
DEFAULT_CANDIDATE_CSV = DEFAULT_SOURCE_DIR / "alpha_zoo_validation_march_high_leverage_candidates_latest.csv"
DEFAULT_OUTPUT_DIR = (
    high.DEFAULT_ALPHA_V2 / "alpha_zoo_top_seed_hybrid_cost_validation_20260518"
)
REPORT_COST_BPS = (5.0, 10.0)
STREAM_COST_BPS = (0.0, 5.0, 10.0)
SPLIT_ORDER = ("train", "validation", "locked_oos")
REFERENCE_SPECS = (
    ("reference_fast_residual_7x_0p15", "alpha_zoo_fast_residual", 7.0, 0.15),
    ("reference_strict_zero_fast_residual_6x_0p10", "alpha_zoo_fast_residual", 6.0, 0.10),
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _candidate_key(candidate_name: Any, leverage: Any, allocation_fraction: Any) -> tuple[str, float, float]:
    return (
        str(candidate_name),
        round(_safe_float(leverage), 10),
        round(_safe_float(allocation_fraction), 10),
    )


def _candidate_label(row: Mapping[str, Any]) -> str:
    name, lev, alloc = _candidate_key(
        row.get("candidate_name"),
        row.get("leverage"),
        row.get("allocation_fraction"),
    )
    return f"{name} {lev:g}x/{alloc:g}"


def _model_id(prefix: str, row: Mapping[str, Any]) -> str:
    name, lev, alloc = _candidate_key(
        row.get("candidate_name"),
        row.get("leverage"),
        row.get("allocation_fraction"),
    )
    alloc_token = f"{alloc:g}".replace(".", "p")
    return f"{prefix}_{name}_{lev:g}x_{alloc_token}"


def _load_candidate_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "candidate_name",
        "leverage",
        "allocation_fraction",
        "train_return",
        "validation_return",
        "locked_oos_return",
        "locked_oos_mdd",
        "locked_oos_sharpe",
        "locked_oos_sortino",
        "locked_oos_smart_sortino",
        "locked_oos_calmar",
        "locked_oos_gate_pass",
        "live_promotion_possible",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("candidate CSV missing columns: " + ", ".join(missing))
    numeric_cols = [
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
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("locked_oos_gate_pass", "live_promotion_possible"):
        frame[col] = frame[col].map(_as_bool)
    frame["full_compound_return"] = (
        (1.0 + frame["train_return"])
        * (1.0 + frame["validation_return"])
        * (1.0 + frame["locked_oos_return"])
        - 1.0
    )
    frame["filtered_balanced_score"] = (
        10.0 * frame["locked_oos_return"]
        + 2.0 * frame["validation_return"]
        + frame["locked_oos_calmar"]
        + 0.25 * frame["locked_oos_sharpe"]
        + 0.25 * frame["locked_oos_sortino"]
        + 0.25 * frame["locked_oos_smart_sortino"]
        - 4.0 * frame["locked_oos_mdd"]
    )
    return frame


def _filtered_gate_frame(frame: pd.DataFrame, current_base_oos_return: float) -> pd.DataFrame:
    return frame[
        frame["train_return"].gt(frame["validation_return"])
        & frame["train_return"].gt(frame["locked_oos_return"])
        & frame["validation_return"].ge(0.10)
        & frame["locked_oos_return"].ge(float(current_base_oos_return))
        & frame["locked_oos_mdd"].le(high.OOS_MDD_BUDGET)
        & frame["locked_oos_sharpe"].gt(0.0)
        & frame["locked_oos_sortino"].gt(0.0)
        & frame["locked_oos_smart_sortino"].gt(0.0)
        & frame["locked_oos_calmar"].gt(0.0)
    ].copy()


def _top_rows(
    frame: pd.DataFrame,
    *,
    metric: str,
    ascending: bool = False,
    top_n: int = 3,
    low_drawdown_tiebreak: bool = False,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    sort_cols = [metric]
    sort_ascending = [ascending]
    if low_drawdown_tiebreak and "locked_oos_mdd" in frame.columns:
        sort_cols.append("locked_oos_mdd")
        sort_ascending.append(True)
    for col, asc in (
        ("tv_selection_score", False),
        ("frozen_train_validation_rank", True),
        ("candidate_name", True),
        ("leverage", True),
        ("allocation_fraction", True),
    ):
        if col in frame.columns:
            sort_cols.append(col)
            sort_ascending.append(asc)
    sorted_frame = frame.sort_values(sort_cols, ascending=sort_ascending, kind="mergesort")
    return [dict(row) for row in sorted_frame.head(int(top_n)).to_dict(orient="records")]


def select_seed_universe(
    candidate_frame: pd.DataFrame,
    *,
    current_base_oos_return: float = high.CURRENT_BASE_OOS_RETURN_REFERENCE,
    top_n: int = 3,
) -> dict[str, Any]:
    """Recompute the requested top-bucket union from a candidate dataframe."""
    live = candidate_frame[candidate_frame["live_promotion_possible"].astype(bool)].copy()
    filtered = _filtered_gate_frame(candidate_frame, current_base_oos_return)
    bucket_defs = [
        ("live_oos_return", live, "locked_oos_return", False, False),
        ("live_oos_sharpe", live, "locked_oos_sharpe", False, True),
        ("live_oos_sortino", live, "locked_oos_sortino", False, True),
        ("live_oos_smart_sortino", live, "locked_oos_smart_sortino", False, False),
        ("live_oos_calmar", live, "locked_oos_calmar", False, False),
        ("live_full_compound", live, "full_compound_return", False, False),
        ("filtered_balanced_score", filtered, "filtered_balanced_score", False, False),
        ("filtered_validation_return", filtered, "validation_return", False, False),
        ("filtered_oos_return", filtered, "locked_oos_return", False, False),
        ("filtered_oos_calmar", filtered, "locked_oos_calmar", False, False),
    ]
    buckets: list[dict[str, Any]] = []
    by_key: dict[tuple[str, float, float], dict[str, Any]] = {}
    bucket_membership: dict[tuple[str, float, float], list[str]] = defaultdict(list)
    for bucket_name, source, metric, ascending, low_mdd in bucket_defs:
        rows = _top_rows(
            source,
            metric=metric,
            ascending=ascending,
            top_n=top_n,
            low_drawdown_tiebreak=low_mdd,
        )
        bucket_rows: list[dict[str, Any]] = []
        for row in rows:
            key = _candidate_key(row.get("candidate_name"), row.get("leverage"), row.get("allocation_fraction"))
            by_key.setdefault(key, row)
            bucket_membership[key].append(bucket_name)
            bucket_rows.append(_public_seed_row(row, source_bucket=bucket_name))
        buckets.append(
            {
                "bucket": bucket_name,
                "metric": metric,
                "source_row_count": len(source),
                "top_n": int(top_n),
                "rows": bucket_rows,
            }
        )
    seed_rows: list[dict[str, Any]] = []
    for idx, (key, row) in enumerate(by_key.items(), start=1):
        public = _public_seed_row(row)
        public["seed_index"] = idx
        public["source_buckets"] = sorted(bucket_membership[key])
        seed_rows.append(public)
    return {
        "bucket_source": "current_live_aligned_candidate_csv",
        "top_n_per_bucket": int(top_n),
        "live_promotion_row_count": len(live),
        "filtered_gate_row_count": len(filtered),
        "current_base_oos_return_reference": float(current_base_oos_return),
        "buckets": buckets,
        "deduped_seed_count": len(seed_rows),
        "deduped_seed_universe": seed_rows,
        "post_hoc_research_basket_note": (
            "The seed basket intentionally uses current leaderboard/OOS buckets as a diagnostic "
            "research universe by user request. It is not a deployable live-selection procedure; "
            "hybrid fitting/objective/pruning use train+validation only."
        ),
    }


def _public_seed_row(row: Mapping[str, Any], *, source_bucket: str | None = None) -> dict[str, Any]:
    out = {
        "candidate_name": str(row.get("candidate_name")),
        "leverage": _safe_float(row.get("leverage")),
        "allocation_fraction": _safe_float(row.get("allocation_fraction")),
        "label": _candidate_label(row),
        "frozen_train_validation_rank": int(_safe_float(row.get("frozen_train_validation_rank"), 0.0)),
        "tv_selection_score": _safe_float(row.get("tv_selection_score")),
        "train_return": _safe_float(row.get("train_return")),
        "validation_return": _safe_float(row.get("validation_return")),
        "locked_oos_return": _safe_float(row.get("locked_oos_return")),
        "locked_oos_mdd": _safe_float(row.get("locked_oos_mdd")),
        "locked_oos_sharpe": _safe_float(row.get("locked_oos_sharpe")),
        "locked_oos_sortino": _safe_float(row.get("locked_oos_sortino")),
        "locked_oos_smart_sortino": _safe_float(row.get("locked_oos_smart_sortino")),
        "locked_oos_calmar": _safe_float(row.get("locked_oos_calmar")),
        "locked_oos_gate_pass": _as_bool(row.get("locked_oos_gate_pass")),
        "live_promotion_possible": _as_bool(row.get("live_promotion_possible")),
    }
    if source_bucket:
        out["source_bucket"] = source_bucket
    return out


def _find_candidate_row(
    frame: pd.DataFrame,
    *,
    candidate_name: str,
    leverage: float,
    allocation_fraction: float,
) -> dict[str, Any]:
    target = frame[
        frame["candidate_name"].astype(str).eq(str(candidate_name))
        & frame["leverage"].sub(float(leverage)).abs().le(1e-12)
        & frame["allocation_fraction"].sub(float(allocation_fraction)).abs().le(1e-12)
    ]
    if target.empty:
        raise ValueError(f"missing candidate row: {candidate_name} {leverage:g}x/{allocation_fraction:g}")
    return dict(target.iloc[0].to_dict())


def _spec_by_name(alpha: Any, old_replay: Mapping[str, Any]) -> dict[str, Any]:
    specs = [common._old_selected_spec(old_replay, alpha), *alpha._default_grid_specs()]
    if hasattr(alpha, "_sample_guarded_new_alpha_grid_specs"):
        specs.extend(alpha._sample_guarded_new_alpha_grid_specs())
    out: dict[str, Any] = {}
    for spec in specs:
        out.setdefault(str(spec.name), spec)
    return out


def _timestamp_seconds(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return int(ts.timestamp())


def _timestamp_arrays(data: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, int]]:
    per_ts = data[["timestamp", "split"]].drop_duplicates("timestamp").sort_values("timestamp")
    timestamps = np.asarray([_timestamp_seconds(ts) for ts in per_ts["timestamp"].tolist()], dtype=np.int64)
    split_values = per_ts["split"].astype(str).to_numpy(dtype=object)
    split_masks = {split: split_values == split for split in SPLIT_ORDER}
    return timestamps, split_masks, {int(ts): idx for idx, ts in enumerate(timestamps.tolist())}


def _cost_adjusted_trade_return(
    alpha: Any,
    trade: Mapping[str, Any],
    *,
    leverage: float,
    allocation_fraction: float,
    round_trip_slippage_bps: float,
) -> tuple[float, bool]:
    """Return isolated account-level trade return after round-trip bps cost."""
    _base_ret, liquidated, _min_adverse = high._isolated_trade_return(
        dict(trade),
        leverage=float(leverage),
        allocation_fraction=float(allocation_fraction),
        alpha=alpha,
    )
    if liquidated:
        return -float(allocation_fraction), True
    return (
        alpha._portfolio_trade_return(
            dict(trade),
            leverage=float(leverage),
            allocation_fraction=float(allocation_fraction),
            round_trip_slippage_bps=float(round_trip_slippage_bps),
        ),
        False,
    )


def _trade_counts_by_split(trades: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(SPLIT_ORDER, 0)
    for trade in trades:
        split = str(trade.get("entry_split"))
        if split in counts:
            counts[split] += 1
    return counts


def _split_account_wipeout_count_from_returns(returns: np.ndarray, mask: np.ndarray) -> int:
    equity = 1.0
    wipeouts = 0
    for raw in np.asarray(returns, dtype=float)[np.asarray(mask, dtype=bool)]:
        ret = float(raw) if math.isfinite(float(raw)) else 0.0
        equity *= max(0.0, 1.0 + ret)
        if equity <= 0.0:
            wipeouts += 1
    return int(wipeouts)


def _stream_from_trades(
    *,
    alpha: Any,
    trades: Sequence[Mapping[str, Any]],
    timestamps: np.ndarray,
    ts_to_idx: Mapping[int, int],
    split_masks: Mapping[str, np.ndarray],
    candidate_row: Mapping[str, Any],
    round_trip_slippage_bps: float,
) -> dict[str, Any]:
    leverage = _safe_float(candidate_row.get("leverage"))
    allocation = _safe_float(candidate_row.get("allocation_fraction"))
    buckets: dict[int, list[float]] = defaultdict(list)
    liquidated_by_split = dict.fromkeys(SPLIT_ORDER, 0)
    for trade in trades:
        try:
            exit_ts = _timestamp_seconds(trade.get("exit_time"))
        except (TypeError, ValueError, OverflowError):
            continue
        idx = ts_to_idx.get(exit_ts)
        if idx is None:
            continue
        ret, liquidated = _cost_adjusted_trade_return(
            alpha,
            trade,
            leverage=leverage,
            allocation_fraction=allocation,
            round_trip_slippage_bps=float(round_trip_slippage_bps),
        )
        buckets[int(idx)].append(float(ret))
        split = str(trade.get("entry_split"))
        if liquidated and split in liquidated_by_split:
            liquidated_by_split[split] += 1
    full = np.zeros(timestamps.shape[0], dtype=float)
    for idx, values in buckets.items():
        full[idx] = math.prod(1.0 + float(value) for value in values) - 1.0

    audit = high._audit_from_trade_extrema(
        [dict(item) for item in trades],
        leverage=leverage,
        allocation_fraction=allocation,
        starting_equity=high.STARTING_EQUITY,
    )
    trade_counts = _trade_counts_by_split(trades)
    split_metrics: dict[str, dict[str, Any]] = {}
    audit_status = dict(audit.get("split_status") or {})
    for split, mask in split_masks.items():
        metrics = hybrid._metrics_from_returns(full[mask])
        status = dict(audit_status.get(split) or {})
        metrics["trade_count"] = int(trade_counts.get(split, 0))
        metrics["active_return_hours"] = int(np.count_nonzero(np.abs(full[mask]) > 1e-12))
        metrics["liquidation_count"] = int(status.get("liquidation_count") or liquidated_by_split.get(split, 0))
        metrics["account_wipeout_count"] = int(status.get("account_wipeout_count") or 0)
        metrics["minimum_margin_buffer"] = status.get("minimum_margin_buffer")
        metrics["margin_buffer_positive"] = bool(status.get("margin_buffer_positive", True))
        metrics["margin_replay_available"] = True
        split_metrics[split] = metrics
    return {
        "label": _candidate_label(candidate_row),
        "candidate_name": str(candidate_row.get("candidate_name")),
        "candidate_source": "CryptoFxAlphaZooStateStrategy:top_seed_union",
        "leverage": leverage,
        "allocation_fraction": allocation,
        "target_allocation": allocation,
        "sleeve_gross_weight_sum": 1.0,
        "returns": full,
        "round_trip_slippage_fee_bps": float(round_trip_slippage_bps),
        "split_metrics": split_metrics,
        "trade_count_reconstructed": len(trades),
        "signal_count_reconstructed": int(candidate_row.get("signal_count_reconstructed") or 0),
        "total_liquidation_count": int(audit.get("total_liquidation_count") or 0),
        "total_account_wipeout_count": sum(
            int(dict(audit_status.get(split) or {}).get("account_wipeout_count") or 0)
            for split in SPLIT_ORDER
        ),
        "minimum_margin_buffer": audit.get("minimum_margin_buffer"),
        "uses_locked_oos_for_selection": False,
        "structural_hybrid_input": False,
    }


def _build_trade_cache(
    *,
    alpha: Any,
    data: pd.DataFrame,
    calibrated_edges: dict[str, float],
    specs: Mapping[str, Any],
    candidate_names: Iterable[str],
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for candidate_name in sorted({str(name) for name in candidate_names}):
        spec = specs.get(candidate_name)
        if spec is None:
            raise ValueError(f"cannot resolve Alpha Zoo spec: {candidate_name}")
        signals = alpha._run_strategy_signals(
            data,
            require_calibrated_edge=True,
            calibrated_edges=calibrated_edges,
            strategy_params=dict(spec.params),
        )
        trades = high._attach_trade_path_extrema(alpha, data, alpha._build_trades(data, signals))
        cache[candidate_name] = {
            "spec": spec,
            "signals": signals,
            "trades": trades,
            "signal_count": len(signals),
            "trade_count": len(trades),
        }
    return cache


def _gate_reasons(metrics: Mapping[str, Any], *, strict_zero_liquidation: bool = False) -> list[str]:
    reasons: list[str] = []
    if _safe_float(metrics.get("max_drawdown"), 1.0) > high.OOS_MDD_BUDGET:
        reasons.append("mdd_above_25pct")
    if _safe_float(metrics.get("total_return")) <= 0.0:
        reasons.append("return_non_positive")
    for key in ("sharpe", "sortino", "smart_sortino", "calmar"):
        if _safe_float(metrics.get(key)) <= 0.0:
            reasons.append(f"{key}_non_positive")
    if int(metrics.get("account_wipeout_count") or 0) > 0:
        reasons.append("account_wipeout_count_positive")
    if metrics.get("minimum_margin_buffer") is not None and _safe_float(
        metrics.get("minimum_margin_buffer"), -1.0
    ) <= 0.0:
        reasons.append("minimum_margin_buffer_non_positive")
    if strict_zero_liquidation and int(metrics.get("liquidation_count") or 0) > 0:
        reasons.append("strict_zero_liquidation_count_positive")
    return reasons


def _metric_rows_for_model(
    *,
    model_id: str,
    model_kind: str,
    role: str,
    cost_bps: float,
    split_metrics: Mapping[str, Mapping[str, Any]],
    candidate_name: str = "",
    leverage: float | None = None,
    allocation_fraction: float | None = None,
    strict_zero_liquidation: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    locked_reasons = _gate_reasons(
        dict(split_metrics.get("locked_oos") or {}),
        strict_zero_liquidation=strict_zero_liquidation,
    )
    deployable = not locked_reasons
    for split in SPLIT_ORDER:
        metrics = dict(split_metrics.get(split) or {})
        split_reasons = _gate_reasons(metrics, strict_zero_liquidation=strict_zero_liquidation)
        rows.append(
            {
                "model_id": model_id,
                "model_kind": model_kind,
                "role": role,
                "candidate_name": candidate_name,
                "leverage": "" if leverage is None else float(leverage),
                "allocation_fraction": "" if allocation_fraction is None else float(allocation_fraction),
                "round_trip_slippage_fee_bps": float(cost_bps),
                "split": split,
                "total_return": _safe_float(metrics.get("total_return")),
                "max_drawdown": _safe_float(metrics.get("max_drawdown")),
                "sharpe": _safe_float(metrics.get("sharpe")),
                "sortino": _safe_float(metrics.get("sortino")),
                "smart_sortino": _safe_float(metrics.get("smart_sortino")),
                "calmar": _safe_float(metrics.get("calmar")),
                "return_mdd": _safe_float(metrics.get("return_mdd")),
                "trade_event_count": int(metrics.get("trade_count") or 0),
                "active_return_hours": int(metrics.get("active_return_hours") or 0),
                "liquidation_count": int(metrics.get("liquidation_count") or 0),
                "account_wipeout_count": int(metrics.get("account_wipeout_count") or 0),
                "minimum_margin_buffer": metrics.get("minimum_margin_buffer"),
                "split_gate_pass": not split_reasons,
                "split_gate_reasons": ";".join(split_reasons),
                "locked_oos_deployable_gate_pass": deployable,
                "locked_oos_deployable_gate_reasons": ";".join(locked_reasons),
            }
        )
    return rows


def _public_hybrid_result(
    result: Mapping[str, Any],
    *,
    labels: list[str],
    returns: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    timestamps: np.ndarray,
    streams: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    portfolio, allocations = hybrid._full_resolution_allocations_for_result(
        result,
        returns=returns,
        split_masks=split_masks,
    )
    margin = hybrid._integrated_margin_replay(
        timestamps=timestamps,
        split_masks=split_masks,
        candidate_returns=returns,
        portfolio_returns=portfolio,
        allocations=allocations,
        streams=streams,
    )
    public = hybrid._public_result(result, labels)
    hybrid._attach_integrated_margin_to_splits(public, margin)
    for split, metrics in dict(public.get("splits") or {}).items():
        mask = np.asarray(split_masks[split], dtype=bool)
        metrics["account_wipeout_count"] = _split_account_wipeout_count_from_returns(portfolio, mask)
        metrics["active_return_hours"] = int(np.count_nonzero(np.abs(portfolio[mask]) > 1e-12))
        public["splits"][split] = metrics
    public["integrated_margin_replay"] = margin
    public["portfolio_returns"] = portfolio
    public["selection_provenance"] = {
        "selection_inputs": ["train", "validation"],
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
    }
    return public


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def _seed_selection_csv_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "source_buckets": ";".join(str(item) for item in list(row.get("source_buckets") or [])),
        }
        for row in list(selection.get("deduped_seed_universe") or [])
    ]


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Alpha Zoo top-seed Hybrid v3.5/v3.6 cost validation",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        "- real_money_execution_attempted: `false`",
        "- hybrid objective/fitting inputs: `train`, `validation` only",
        "- locked-OOS role: gate/report-only after seed/hybrid freeze",
        "- cost scenarios: round-trip slippage/fee `5bps` and `10bps`",
        "",
        "## Seed universe",
        "",
        f"- deduped seed count: `{dict(payload.get('seed_selection') or {}).get('deduped_seed_count')}`",
        "",
    ]
    for row in list(dict(payload.get("seed_selection") or {}).get("deduped_seed_universe") or []):
        lines.append(
            f"- {row.get('seed_index')}. `{row.get('label')}` via "
            f"`{', '.join(list(row.get('source_buckets') or []))}`"
        )
    lines.extend(
        [
            "",
            "## Cost metrics",
            "",
            "| cost bps | model | role | split | return | MDD | Sharpe | Sortino | Smart Sortino | Calmar | events | liq | wipeout | min buffer | OOS gate |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in list(payload.get("model_cost_metrics") or []):
        if float(row.get("round_trip_slippage_fee_bps") or 0.0) not in REPORT_COST_BPS:
            continue
        lines.append(
            f"| {float(row.get('round_trip_slippage_fee_bps') or 0.0):g} | `{row.get('model_id')}` | "
            f"{row.get('role')} | {row.get('split')} | {_fmt_pct(row.get('total_return'))} | "
            f"{_fmt_pct(row.get('max_drawdown'))} | {_fmt_num(row.get('sharpe'))} | "
            f"{_fmt_num(row.get('sortino'))} | {_fmt_num(row.get('smart_sortino'))} | "
            f"{_fmt_num(row.get('calmar'))} | {row.get('trade_event_count')} | "
            f"{row.get('liquidation_count')} | {row.get('account_wipeout_count')} | "
            f"{_fmt_num(row.get('minimum_margin_buffer'))} | {row.get('locked_oos_deployable_gate_pass')} |"
        )
    lines.extend(["", "## Hybrid final weights", ""])
    for row in list(payload.get("hybrid_weights") or []):
        lines.append(
            f"- {row.get('model_id')} cost={float(row.get('round_trip_slippage_fee_bps') or 0.0):g}bps "
            f"{row.get('candidate_label')}={_fmt_pct(row.get('weight'))}"
        )
    return "\n".join(lines) + "\n"


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):+.2%}"


def _fmt_num(value: Any) -> str:
    if value in {None, ""}:
        return ""
    parsed = _safe_float(value, float("nan"))
    if math.isnan(parsed):
        return ""
    if math.isinf(parsed):
        return "inf"
    return f"{parsed:.4f}"


def _build_alpha_runtime(args: argparse.Namespace, live_payload: Mapping[str, Any]) -> dict[str, Any]:
    split_contract = high._split_contract(args)
    bundle = common.load_real_data_bundle(
        input_path=args.input,
        current_tail_cache=args.current_tail_cache,
        external_state_csv=args.external_state_csv,
        strict_real_data=True,
    )
    common_frame = common.apply_common_split(bundle.frame, split_contract=split_contract)
    common_frame = common.add_split_bounded_forward_return_label(common_frame, horizon=int(args.horizon))
    alpha = common._load_module(
        REPO_ROOT / "scripts/research/replay_crypto_fx_alpha_zoo_state.py",
        "alpha_zoo_top_seed_hybrid_cost_validation_alpha_replay",
    )
    old_replay = common._load_json(Path(args.old_alpha_replay_json))
    calibration_path = Path(str(live_payload.get("calibration_payload_path") or args.alpha_calibration_json))
    calibrated_edges = alpha._load_calibrated_edges(calibration_path)
    data = alpha._ensure_replay_frame(common_frame)
    timestamps, split_masks, ts_to_idx = _timestamp_arrays(data)
    return {
        "split_contract": split_contract,
        "bundle_metadata": bundle.metadata,
        "common_frame": common_frame,
        "alpha": alpha,
        "old_replay": old_replay,
        "calibrated_edges": calibrated_edges,
        "calibration_path": calibration_path,
        "data": data,
        "timestamps": timestamps,
        "split_masks": split_masks,
        "ts_to_idx": ts_to_idx,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    live_json = Path(args.live_json).expanduser().resolve()
    candidate_csv = Path(args.candidate_csv).expanduser().resolve()
    live_payload = common._load_json(live_json)
    candidate_frame = _load_candidate_frame(candidate_csv)
    current_base = _safe_float(
        dict(live_payload.get("current_base_reference") or {}).get("locked_oos_total_return"),
        high.CURRENT_BASE_OOS_RETURN_REFERENCE,
    )
    seed_selection = select_seed_universe(
        candidate_frame,
        current_base_oos_return=current_base,
        top_n=int(args.top_n_per_bucket),
    )
    seed_rows = [dict(row) for row in list(seed_selection["deduped_seed_universe"])]
    reference_rows = []
    for role, name, leverage, allocation in REFERENCE_SPECS:
        row = _public_seed_row(
            _find_candidate_row(
                candidate_frame,
                candidate_name=name,
                leverage=leverage,
                allocation_fraction=allocation,
            )
        )
        row["reference_role"] = role
        reference_rows.append(row)

    all_stream_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for row in [*seed_rows, *reference_rows]:
        key = _candidate_key(row["candidate_name"], row["leverage"], row["allocation_fraction"])
        if key not in seen:
            all_stream_rows.append(dict(row))
            seen.add(key)

    runtime = _build_alpha_runtime(args, live_payload)
    alpha = runtime["alpha"]
    data = runtime["data"]
    timestamps = runtime["timestamps"]
    split_masks = runtime["split_masks"]
    trade_cache = _build_trade_cache(
        alpha=alpha,
        data=data,
        calibrated_edges=runtime["calibrated_edges"],
        specs=_spec_by_name(alpha, runtime["old_replay"]),
        candidate_names=[row["candidate_name"] for row in all_stream_rows],
    )
    for row in all_stream_rows:
        row["signal_count_reconstructed"] = trade_cache[str(row["candidate_name"])]["signal_count"]

    streams_by_cost: dict[float, dict[tuple[str, float, float], dict[str, Any]]] = {}
    for cost_bps in STREAM_COST_BPS:
        streams_by_cost[float(cost_bps)] = {}
        for row in all_stream_rows:
            cache = trade_cache[str(row["candidate_name"])]
            stream = _stream_from_trades(
                alpha=alpha,
                trades=list(cache["trades"]),
                timestamps=timestamps,
                ts_to_idx=runtime["ts_to_idx"],
                split_masks=split_masks,
                candidate_row=row,
                round_trip_slippage_bps=float(cost_bps),
            )
            streams_by_cost[float(cost_bps)][
                _candidate_key(row["candidate_name"], row["leverage"], row["allocation_fraction"])
            ] = stream

    model_rows: list[dict[str, Any]] = []
    hybrid_payloads: dict[str, Any] = {}
    hybrid_weights: list[dict[str, Any]] = []
    seed_keys = [
        _candidate_key(row["candidate_name"], row["leverage"], row["allocation_fraction"])
        for row in seed_rows
    ]
    for cost_bps in REPORT_COST_BPS:
        cost_streams = streams_by_cost[float(cost_bps)]
        seed_streams = [cost_streams[key] for key in seed_keys]
        returns = np.column_stack([np.asarray(stream["returns"], dtype=float) for stream in seed_streams])
        labels = [str(stream["label"]) for stream in seed_streams]
        v35 = hybrid._run_optuna(
            returns,
            split_masks,
            version="v3_5",
            n_trials=int(args.n_trials),
            seed=int(args.seed),
        )
        v36 = hybrid._run_optuna(
            returns,
            split_masks,
            version="v3_6",
            n_trials=int(args.n_trials),
            seed=int(args.seed),
        )
        for model_id, result in (
            ("hybrid_v3_5_seed_union", v35),
            ("hybrid_v3_6_seed_union", v36),
        ):
            public = _public_hybrid_result(
                result,
                labels=labels,
                returns=returns,
                split_masks=split_masks,
                timestamps=timestamps,
                streams=seed_streams,
            )
            public.pop("portfolio_returns", None)
            hybrid_payloads[f"{model_id}_{int(cost_bps)}bps"] = public
            model_rows.extend(
                _metric_rows_for_model(
                    model_id=model_id,
                    model_kind="hybrid",
                    role="hybrid_seed_union",
                    cost_bps=float(cost_bps),
                    split_metrics=dict(public.get("splits") or {}),
                )
            )
            for label, weight in dict(public.get("final_weight_by_candidate") or {}).items():
                hybrid_weights.append(
                    {
                        "round_trip_slippage_fee_bps": float(cost_bps),
                        "model_id": model_id,
                        "candidate_label": label,
                        "weight": float(weight),
                    }
                )

        for row in seed_rows:
            key = _candidate_key(row["candidate_name"], row["leverage"], row["allocation_fraction"])
            stream = cost_streams[key]
            model_rows.extend(
                _metric_rows_for_model(
                    model_id=_model_id("seed", row),
                    model_kind="individual_seed",
                    role="seed_universe",
                    cost_bps=float(cost_bps),
                    split_metrics=stream["split_metrics"],
                    candidate_name=str(row["candidate_name"]),
                    leverage=float(row["leverage"]),
                    allocation_fraction=float(row["allocation_fraction"]),
                )
            )
        for row in reference_rows:
            key = _candidate_key(row["candidate_name"], row["leverage"], row["allocation_fraction"])
            stream = cost_streams[key]
            strict = str(row.get("reference_role")) == "reference_strict_zero_fast_residual_6x_0p10"
            model_rows.extend(
                _metric_rows_for_model(
                    model_id=str(row["reference_role"]),
                    model_kind="reference",
                    role=str(row["reference_role"]),
                    cost_bps=float(cost_bps),
                    split_metrics=stream["split_metrics"],
                    candidate_name=str(row["candidate_name"]),
                    leverage=float(row["leverage"]),
                    allocation_fraction=float(row["allocation_fraction"]),
                    strict_zero_liquidation=strict,
                )
            )

    payload = {
        "artifact_kind": "alpha_zoo_top_seed_hybrid_v35_v36_cost_validation",
        "generated_at_utc": _utc_now_iso(),
        "real_money_execution": {"attempted": False, "authorization_required": True},
        "source_inputs": {
            "live_json": str(live_json),
            "candidate_csv": str(candidate_csv),
            "calibration_payload_path": str(runtime["calibration_path"]),
        },
        "split_manifest": {
            "split_contract": runtime["split_contract"],
            "split_periods": common._split_periods(runtime["common_frame"]),
            "timestamp_index_hash": common._timestamp_index_hash(runtime["common_frame"]),
            "frame_rows": len(runtime["common_frame"]),
        },
        "seed_selection": seed_selection,
        "reference_rows": reference_rows,
        "method_contract": {
            "input_universe": "deduped Alpha Zoo top-bucket seed streams",
            "fixed_input_runner_contract_unchanged": True,
            "hybrid_versions": ["v3_5", "v3_6"],
            "round_trip_slippage_fee_bps_reported": list(REPORT_COST_BPS),
            "round_trip_slippage_fee_bps_reference_streams": list(STREAM_COST_BPS),
        },
        "selection_policy": {
            "hybrid_selection_inputs": ["train", "validation"],
            "locked_oos": "gate_report_only_after_candidate_freeze",
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "post_hoc_seed_basket_uses_leaderboard_oos_buckets_by_request": True,
        },
        "locked_oos_contamination_audit": {
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
            "evidence": [
                "Hybrid Optuna objective calls only train and validation split metrics.",
                "Hybrid learned parameters are fit on train|validation masks only.",
                "Locked-OOS split metrics are attached after model freeze for gate/report rows.",
                "Seed basket is a documented post-hoc research basket from leaderboard buckets, not a deployable live-selection rule.",
            ],
        },
        "hybrid_results": hybrid_payloads,
        "model_cost_metrics": model_rows,
        "hybrid_weights": hybrid_weights,
        "memory_summary": {
            "peak_rss_mib": _rss_mib(),
            "limit_mib": 8192.0,
            "pass_under_8gb": _rss_mib() < 8192.0,
        },
    }
    return payload


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    latest_json = output_dir / "alpha_zoo_top_seed_hybrid_cost_validation_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_top_seed_hybrid_cost_validation_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_top_seed_hybrid_cost_validation_latest.md"
    timestamped_md = output_dir / f"alpha_zoo_top_seed_hybrid_cost_validation_{timestamp}.md"
    seed_json = output_dir / "seed_selection_latest.json"
    seed_csv = output_dir / "seed_selection_latest.csv"
    metrics_csv = output_dir / "model_cost_metrics_latest.csv"
    weights_csv = output_dir / "hybrid_weights_latest.csv"
    output_paths = {
        "latest_json": str(latest_json),
        "timestamped_json": str(timestamped_json),
        "latest_markdown": str(latest_md),
        "timestamped_markdown": str(timestamped_md),
        "seed_selection_json": str(seed_json),
        "seed_selection_csv": str(seed_csv),
        "model_cost_metrics_csv": str(metrics_csv),
        "hybrid_weights_csv": str(weights_csv),
    }
    payload = {**dict(payload), "output_paths": output_paths}
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    _write_json(seed_json, dict(payload.get("seed_selection") or {}))
    md = _markdown(payload)
    latest_md.write_text(md, encoding="utf-8")
    timestamped_md.write_text(md, encoding="utf-8")
    _write_csv(
        seed_csv,
        _seed_selection_csv_rows(dict(payload.get("seed_selection") or {})),
        [
            "seed_index",
            "candidate_name",
            "leverage",
            "allocation_fraction",
            "label",
            "source_buckets",
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
            "locked_oos_gate_pass",
            "live_promotion_possible",
        ],
    )
    _write_csv(
        metrics_csv,
        list(payload.get("model_cost_metrics") or []),
        [
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
            "split_gate_pass",
            "split_gate_reasons",
            "locked_oos_deployable_gate_pass",
            "locked_oos_deployable_gate_reasons",
        ],
    )
    _write_csv(
        weights_csv,
        list(payload.get("hybrid_weights") or []),
        ["round_trip_slippage_fee_bps", "model_id", "candidate_label", "weight"],
    )
    return output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="")
    parser.add_argument("--current-tail-cache", default=str(high.DEFAULT_CURRENT_TAIL_CACHE))
    parser.add_argument("--external-state-csv", default=str(high.DEFAULT_EXTERNAL_STATE_CSV))
    parser.add_argument("--old-alpha-replay-json", default=str(high.DEFAULT_OLD_ALPHA_REPLAY))
    parser.add_argument("--alpha-calibration-json", default="")
    parser.add_argument("--live-json", default=str(DEFAULT_LIVE_JSON))
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--train-start", default=high.DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=high.DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=high.DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=high.DEFAULT_VALIDATION_END)
    parser.add_argument("--locked-oos-start", default=high.DEFAULT_LOCKED_OOS_START)
    parser.add_argument("--locked-oos-end", default=high.DEFAULT_LOCKED_OOS_END)
    parser.add_argument("--horizon", type=int, default=high.DEFAULT_HORIZON)
    parser.add_argument("--top-n-per-bucket", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = build_payload(args)
    outputs = write_outputs(payload, output_dir)
    print(
        json.dumps(
            {
                **outputs,
                "seed_count": dict(payload.get("seed_selection") or {}).get("deduped_seed_count"),
                "metric_rows": len(list(payload.get("model_cost_metrics") or [])),
                "peak_rss_mib": dict(payload.get("memory_summary") or {}).get("peak_rss_mib"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
