#!/usr/bin/env python3
"""Optuna-tune v3.5/v3.6-style hybrids over integer-leverage paper profiles.

This runner supersedes the earlier coarse 5% grid hybrid as the decision
surface.  It consumes the frozen ``alpha_zoo_corr_integer_leverage_portfolio``
artifact, reconstructs 10bps-costed source-profile PnL streams, and runs two
Optuna searches that mirror the existing hybrid v3.5/v3.6 pattern:

* v3.5: warmup-learned default candidate + rolling return/error weights,
  high-volatility boost, max-weight cap, and bias/exposure dampening.
* v3.6: the same mechanics, but the default candidate is refreshed online from
  rolling train/validation score evidence.

Only train+validation are allowed in learning/objective/selection. locked-OOS is
attached after the Optuna params are frozen and remains gate/report-only.  The
runner performs no live or real-money execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_integer_leverage_hybrid_decision as grid_hybrid  # noqa: E402

ilp = grid_hybrid.ilp

DEFAULT_INTEGER_PORTFOLIO_ARTIFACT = grid_hybrid.DEFAULT_INTEGER_PORTFOLIO_ARTIFACT
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524"
)

ARTIFACT_KIND = "alpha_zoo_integer_leverage_optuna_hybrid_decision"
V35_PROFILE_ID = "hybrid_v3_5_optuna_three_profile_blend"
V36_PROFILE_ID = "hybrid_v3_6_optuna_three_profile_blend"
MIN_VALIDATION_RETURN = grid_hybrid.MIN_VALIDATION_RETURN
MAX_VALIDATION_MDD = grid_hybrid.MAX_VALIDATION_MDD
MAX_TRAIN_MDD = grid_hybrid.MAX_TRAIN_MDD
MAX_GROSS_NOTIONAL = grid_hybrid.MAX_GROSS_NOTIONAL

COMPARISON_FIELDS = [
    *grid_hybrid.COMPARISON_FIELDS,
    "optimizer",
    "hybrid_version",
    "best_value",
    "best_params",
    "learned_params",
    "average_weights_train_validation",
    "final_weights",
]

TRIAL_FIELDS = [
    "hybrid_version",
    "trial_number",
    "value",
    "state",
    "params",
    "train_return",
    "validation_return",
    "train_mdd",
    "validation_mdd",
    "train_rpt_bps",
    "validation_rpt_bps",
    "selection_reasons",
]


@dataclass(frozen=True)
class HybridParams:
    """Optuna knobs aligned to the existing hybrid v3.5/v3.6 runner."""

    bias_correction_alpha: float = 1.0
    bias_combine_ratio: float = 0.25
    max_single_weight: float = 0.78
    mape_window: int = 44
    bias_window: int = 15
    short_vol_window: int = 7
    warmup_ratio: float = 0.6
    min_boost: float = 0.05
    max_boost: float = 0.30


@dataclass(frozen=True)
class LearnedParams:
    high_vol_threshold: float
    default_idx: int
    high_vol_best_idx: int
    default_weight_ratio: float
    high_vol_weight_boost: float
    cv_score: float


@dataclass(frozen=True)
class OptunaModelResult:
    row: dict[str, Any]
    returns: pd.Series
    weights: pd.DataFrame
    allocations: list[dict[str, Any]]
    learned_params: LearnedParams
    params: HybridParams
    optuna: Mapping[str, Any] | None = None
    top_trials: Sequence[Mapping[str, Any]] = ()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    return grid_hybrid._json_safe(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    return _json_safe(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _safe_float(value: Any, default: float = 0.0) -> float:
    return grid_hybrid._safe_float(value, default)


def _period_return(values: np.ndarray | pd.Series) -> float:
    return grid_hybrid._period_return(values)


def _split_mask(index: pd.DatetimeIndex, split: str) -> np.ndarray:
    return ilp._split_mask(index, split)


def _split_metrics(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    total = _period_return(arr)
    return {
        "total_return": total,
        "max_drawdown": ilp.max_drawdown(arr),
        "return_mdd": total / ilp.max_drawdown(arr) if ilp.max_drawdown(arr) > 1e-12 else (float("inf") if total > 0 else 0.0),
        "mean_return": float(np.nanmean(arr)) if arr.size else 0.0,
        "volatility": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
        "downside_volatility": float(np.sqrt(np.nanmean(np.square(np.where(arr < 0.0, arr, 0.0))))) if arr.size else 0.0,
        "active_return_bars": int(np.count_nonzero(np.abs(arr) > 1e-12)),
    }


def _source_profile_streams(
    *,
    integer_payload: Mapping[str, Any],
    data_root: Path,
    feature_root: Path,
) -> tuple[list[grid_hybrid.ProfileStream], list[Mapping[str, Any]]]:
    if integer_payload.get("ready_for_real") is not False or integer_payload.get("real_money_execution") is not False:
        raise ValueError("integer portfolio artifact violates real-money disabled guard")
    if _safe_float(integer_payload.get("research_primary_round_trip_cost_bps")) != 10.0:
        raise ValueError("integer portfolio artifact is not using the primary 10bps round-trip cost")
    source_profile_rows = list(integer_payload.get("paper_testnet_candidate_profiles") or [])
    if len(source_profile_rows) != 3:
        raise ValueError(f"expected exactly three paper/testnet source profiles, found {len(source_profile_rows)}")

    correlation_payload = ilp._load_json(ilp.DEFAULT_CORRELATION_ARTIFACT)
    monitoring_payload = ilp._load_json(ilp.DEFAULT_MONITORING_ARTIFACT)
    ilp._assert_governance(correlation_payload, monitoring_payload)
    selected_rows = ilp._selected_rows_from_corr_payload(correlation_payload)
    captures = ilp.corr.capture_pnl_series(
        selected_rows,
        data_root=data_root,
        feature_root=feature_root,
        monitoring_payload=monitoring_payload,
    )
    bars_by_key = ilp._load_bars_for_rows(selected_rows, data_root=data_root)
    replays = ilp.build_candidate_replays(selected_rows, captures, bars_by_key=bars_by_key)
    replays_by_model_id = {replay.model_id: replay for replay in replays}
    union_index = pd.DatetimeIndex(sorted(set().union(*(set(replay.datetimes) for replay in replays))))
    profile_streams = [
        grid_hybrid._profile_stream_from_row(row, replays_by_model_id=replays_by_model_id, union_index=union_index)
        for row in source_profile_rows
    ]
    return profile_streams, source_profile_rows


def _softmax(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    clean = np.where(np.isfinite(clean), clean, -1e9)
    clean = np.clip(clean, -20.0, 20.0)
    shifted = clean - np.nanmax(clean)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0 or not math.isfinite(total):
        return np.full(clean.shape, 1.0 / float(clean.size), dtype=float)
    return exp / total


def _rolling_feature(returns: np.ndarray, end: int, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = max(0, int(end) - max(1, int(window)))
    hist = returns[start:end]
    k = returns.shape[1]
    if hist.size == 0:
        return np.zeros(k), np.zeros(k), np.zeros(k)
    mean = np.nanmean(hist, axis=0)
    std = np.nanstd(hist, axis=0, ddof=1) if hist.shape[0] > 1 else np.zeros(k)
    downside = np.sqrt(np.nanmean(np.square(np.where(hist < 0.0, hist, 0.0)), axis=0))
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 0.0)
    downside = np.where(np.isfinite(downside), downside, 0.0)
    return mean, std, downside


def _candidate_scores(returns: np.ndarray, end: int, window: int, priors: np.ndarray, prior_ratio: float) -> np.ndarray:
    mean, std, downside = _rolling_feature(returns, end, window)
    score = mean / (std + downside + 1e-9)
    score = np.where(np.isfinite(score), score, 0.0)
    return (1.0 - prior_ratio) * score + prior_ratio * priors


def _metrics_for_cv(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    total = _period_return(arr)
    mdd = ilp.max_drawdown(arr)
    mean = float(np.nanmean(arr)) if arr.size else 0.0
    vol = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    downside = float(np.sqrt(np.nanmean(np.square(np.where(arr < 0.0, arr, 0.0))))) if arr.size else 0.0
    sharpe = mean / vol * math.sqrt(24.0 * 365.0) if vol > 1e-12 else 0.0
    sortino = mean / downside * math.sqrt(24.0 * 365.0) if downside > 1e-12 else 0.0
    calmar = total / mdd if mdd > 1e-12 else (float("inf") if total > 0 else 0.0)
    return {"total_return": total, "max_drawdown": mdd, "sharpe": sharpe, "sortino": sortino, "calmar": calmar}


def _portfolio_returns_for_params(
    returns: np.ndarray,
    *,
    params: HybridParams,
    learned: LearnedParams,
    version: str,
    start_idx: int = 0,
    initial_portfolio_history: list[float] | None = None,
    allocation_stride: int = 24,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    k = returns.shape[1]
    out = np.zeros(returns.shape[0], dtype=float)
    weights_out = np.zeros((returns.shape[0], k), dtype=float)
    allocations: list[dict[str, Any]] = []
    portfolio_history = list(initial_portfolio_history or [])
    prior_scores = _candidate_scores(returns, max(1, start_idx), max(2, params.mape_window), np.zeros(k), 0.0)
    if not np.any(np.isfinite(prior_scores)):
        prior_scores = np.zeros(k)
    default_idx = int(learned.default_idx)
    high_vol_idx = int(learned.high_vol_best_idx)
    for t in range(returns.shape[0]):
        if t < start_idx:
            continue
        rolling_scores = _candidate_scores(
            returns,
            t,
            max(2, params.mape_window),
            prior_scores,
            max(0.0, min(1.0, params.bias_combine_ratio)),
        )
        enough = t >= max(2, params.mape_window)
        adaptive_weight_ratio = learned.default_weight_ratio
        adaptive_boost = learned.high_vol_weight_boost
        adaptive_cap = float(params.max_single_weight)
        if version == "v3_6" and enough:
            # v3.6 mapping: retain v3.5 mechanics/knobs but refresh the
            # default candidate online from rolling score evidence.  The score
            # window uses only history up to t; locked-OOS is not used in Optuna.
            default_idx = int(np.nanargmax(rolling_scores))
        score_weights = _softmax(rolling_scores)
        base = np.zeros(k, dtype=float)
        base[default_idx] = 1.0
        weights = adaptive_weight_ratio * base + (1.0 - adaptive_weight_ratio) * score_weights
        recent = returns[max(0, t - max(2, params.short_vol_window)):t]
        vol_feature = 0.0
        if recent.size:
            vol_feature = float(np.nanstd(np.nanmean(recent, axis=1)))
        if vol_feature > learned.high_vol_threshold and 0 <= high_vol_idx < k:
            weights[high_vol_idx] += adaptive_boost
        weights = np.maximum(weights, 0.0)
        if float(np.sum(weights)) <= 0.0:
            weights[:] = 1.0 / float(k)
        weights /= float(np.sum(weights))
        cap = max(1.0 / float(k), min(0.99, float(adaptive_cap)))
        for _ in range(3):
            over = weights > cap
            if not bool(np.any(over)):
                break
            excess = float(np.sum(weights[over] - cap))
            weights[over] = cap
            under = ~over
            if bool(np.any(under)) and excess > 0.0:
                under_sum = float(np.sum(weights[under]))
                if under_sum > 0.0:
                    weights[under] += excess * weights[under] / under_sum
        weights /= max(1e-12, float(np.sum(weights)))
        exposure = 1.0
        if len(portfolio_history) >= max(2, params.bias_window):
            ens = np.asarray(portfolio_history[-int(params.bias_window):], dtype=float)
            model_hist = returns[max(0, t - int(params.bias_window)):t, default_idx]
            ens_bias = float(np.nanmean(ens)) if ens.size else 0.0
            model_bias = float(np.nanmean(model_hist)) if model_hist.size else 0.0
            combined_bias = params.bias_combine_ratio * model_bias + (1.0 - params.bias_combine_ratio) * ens_bias
            denom = float(np.nanmean(np.abs(ens))) + 1e-9
            if combined_bias < 0.0:
                exposure = max(0.0, 1.0 - params.bias_correction_alpha * min(0.80, abs(combined_bias) / denom))
        ret = float(exposure * np.dot(weights, returns[t]))
        if not math.isfinite(ret):
            ret = 0.0
        out[t] = ret
        weights_out[t] = weights * exposure
        portfolio_history.append(ret)
        stride = max(1, int(allocation_stride))
        if t == returns.shape[0] - 1 or t % stride == 0:
            allocations.append(
                {
                    "index": int(t),
                    "default_idx": int(default_idx),
                    "high_vol_feature": float(vol_feature),
                    "exposure": float(exposure),
                    "adaptive_weight_ratio": float(adaptive_weight_ratio),
                    "adaptive_high_vol_boost": float(adaptive_boost),
                    "adaptive_max_single_weight": float(adaptive_cap),
                    "weights": [float(x) for x in weights],
                }
            )
    return out, weights_out, allocations


def _learn_params(returns: np.ndarray, params: HybridParams, opt_indices: np.ndarray) -> LearnedParams:
    opt = returns[opt_indices]
    n = opt.shape[0]
    warmup_n = max(10, min(n, int(n * params.warmup_ratio))) if n else 0
    warmup = opt[:warmup_n]
    if warmup.size == 0:
        return LearnedParams(0.0, 0, 0, 0.5, params.min_boost, 0.0)
    mean = np.nanmean(warmup, axis=0)
    std = np.nanstd(warmup, axis=0, ddof=1) if warmup.shape[0] > 1 else np.zeros(warmup.shape[1])
    scores = np.where(np.isfinite(mean / (std + 1e-9)), mean / (std + 1e-9), 0.0)
    default_idx = int(np.nanargmax(scores))
    vol_series = []
    for t in range(max(2, params.short_vol_window), warmup_n):
        recent = warmup[t - int(params.short_vol_window):t]
        vol_series.append(float(np.nanstd(np.nanmean(recent, axis=1))))
    threshold = float(np.nanpercentile(vol_series, 75)) if vol_series else 0.0
    hv_mask: list[int] = []
    for t in range(max(2, params.short_vol_window), warmup_n):
        recent = warmup[t - int(params.short_vol_window):t]
        if float(np.nanstd(np.nanmean(recent, axis=1))) > threshold:
            hv_mask.append(t)
    if hv_mask:
        hv_mean = np.nanmean(warmup[hv_mask], axis=0)
        high_vol_best = int(np.nanargmax(np.where(np.isfinite(hv_mean), hv_mean, -1e9)))
    else:
        high_vol_best = default_idx
    hv_gap = 0.0
    if hv_mask:
        hv_mean = np.nanmean(warmup[hv_mask], axis=0)
        best = float(hv_mean[high_vol_best])
        others = [float(v) for i, v in enumerate(hv_mean) if i != high_vol_best and math.isfinite(float(v))]
        if others:
            avg_other = float(np.nanmean(others))
            hv_gap = max(0.0, (best - avg_other) / (abs(avg_other) + abs(best) + 1e-9))
    boost = float(np.clip(0.10 + hv_gap * 0.5, params.min_boost, params.max_boost))
    ratios = (0.3, 0.4, 0.5, 0.6, 0.7)
    cv_start = max(2, warmup_n // 2)
    best_ratio = 0.5
    best_score = -1e18
    for ratio in ratios:
        learned = LearnedParams(
            high_vol_threshold=threshold,
            default_idx=default_idx,
            high_vol_best_idx=high_vol_best,
            default_weight_ratio=float(ratio),
            high_vol_weight_boost=boost,
            cv_score=0.0,
        )
        cv_returns, _, _ = _portfolio_returns_for_params(
            warmup,
            params=params,
            learned=learned,
            version="v3_5",
            start_idx=cv_start,
        )
        metrics = _metrics_for_cv(cv_returns[cv_start:])
        score = _safe_float(metrics.get("calmar")) + _safe_float(metrics.get("sharpe")) + _safe_float(metrics.get("sortino"))
        if score > best_score:
            best_score = score
            best_ratio = float(ratio)
    return LearnedParams(
        high_vol_threshold=threshold,
        default_idx=default_idx,
        high_vol_best_idx=high_vol_best,
        default_weight_ratio=best_ratio,
        high_vol_weight_boost=boost,
        cv_score=float(best_score),
    )


def _params_from_trial(trial: Any) -> HybridParams:
    return HybridParams(
        bias_correction_alpha=float(trial.suggest_float("bias_alpha", 0.85, 1.0)),
        bias_combine_ratio=float(trial.suggest_float("bias_combine_ratio", 0.2, 0.45)),
        max_single_weight=float(trial.suggest_float("max_weight", 0.7, 0.9)),
        mape_window=int(trial.suggest_int("mape_window", 25, 50)),
        bias_window=int(trial.suggest_int("bias_window", 10, 20)),
        short_vol_window=int(trial.suggest_int("short_vol_window", 5, 15)),
    )


def _weights_summary(
    *,
    profile_streams: Sequence[grid_hybrid.ProfileStream],
    weights_frame: pd.DataFrame,
    split: str,
) -> dict[str, float]:
    mask = _split_mask(weights_frame.index, split)
    if not bool(np.any(mask)):
        return {stream.profile_id: 0.0 for stream in profile_streams}
    avg = weights_frame.loc[mask].mean(axis=0).fillna(0.0)
    return {stream.profile_id: float(avg.get(stream.profile_id, 0.0)) for stream in profile_streams}


def _turnover_and_events_from_dynamic_weights(
    *,
    profile_streams: Sequence[grid_hybrid.ProfileStream],
    weights_frame: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, int], dict[str, int], dict[str, float]]:
    turnover_by_split: dict[str, float] = {}
    trade_events_by_split: dict[str, int] = {}
    liquidation_by_split: dict[str, int] = {}
    avg_weight_tv: dict[str, float] = {}
    for split in ilp.SPLIT_ORDER:
        mask = _split_mask(weights_frame.index, split)
        if bool(np.any(mask)):
            avg_weights = weights_frame.loc[mask].mean(axis=0).fillna(0.0)
        else:
            avg_weights = pd.Series(0.0, index=weights_frame.columns)
        if split in {"train", "validation"}:
            for stream in profile_streams:
                avg_weight_tv[stream.profile_id] = avg_weight_tv.get(stream.profile_id, 0.0) + float(avg_weights.get(stream.profile_id, 0.0)) / 2.0
        turnover_by_split[split] = float(
            sum(stream.turnover_by_split[split] * max(0.0, float(avg_weights.get(stream.profile_id, 0.0))) for stream in profile_streams)
        )
        active_streams = [stream for stream in profile_streams if float(avg_weights.get(stream.profile_id, 0.0)) > 1e-6]
        trade_events_by_split[split] = int(sum(stream.trade_events_by_split[split] for stream in active_streams))
        liquidation_by_split[split] = int(sum(stream.liquidation_count_by_split[split] for stream in active_streams))
    return turnover_by_split, trade_events_by_split, liquidation_by_split, avg_weight_tv


def _dynamic_asset_gross(
    *,
    profile_streams: Sequence[grid_hybrid.ProfileStream],
    avg_weights: Mapping[str, float],
) -> dict[str, float]:
    by_id = {stream.profile_id: stream for stream in profile_streams}
    asset_gross: dict[str, float] = defaultdict(float)
    for profile_id, weight in avg_weights.items():
        stream = by_id[profile_id]
        for asset, gross in stream.asset_gross_notional_fraction.items():
            asset_gross[asset] += _safe_float(gross) * float(weight)
    return dict(sorted(asset_gross.items()))


def _run_model(
    profile_streams: Sequence[grid_hybrid.ProfileStream],
    params: HybridParams,
    *,
    version: str,
    profile_id: str,
    optuna: Mapping[str, Any] | None = None,
    top_trials: Sequence[Mapping[str, Any]] = (),
) -> OptunaModelResult:
    labels = [stream.profile_id for stream in profile_streams]
    index = profile_streams[0].returns.index
    returns_matrix = np.column_stack([stream.returns.reindex(index, fill_value=0.0).to_numpy(dtype=float) for stream in profile_streams])
    opt_mask = _split_mask(index, "train") | _split_mask(index, "validation")
    opt_indices = np.flatnonzero(opt_mask)
    learned = _learn_params(returns_matrix, params, opt_indices)
    start_idx = int(opt_indices[0]) if opt_indices.size else 0
    portfolio, weights, allocations = _portfolio_returns_for_params(
        returns_matrix,
        params=params,
        learned=learned,
        version=version,
        start_idx=start_idx,
    )
    returns = pd.Series(portfolio, index=index, dtype=float)
    weights_frame = pd.DataFrame(weights, index=index, columns=labels, dtype=float)
    turnover, trade_events, liquidation, avg_weight_tv = _turnover_and_events_from_dynamic_weights(
        profile_streams=profile_streams,
        weights_frame=weights_frame,
    )
    train_val_weight_sum = sum(avg_weight_tv.values())
    if train_val_weight_sum > 0.0:
        avg_weight_tv = {k: float(v / train_val_weight_sum) for k, v in avg_weight_tv.items()}
    gross_notional = float(sum(stream.gross_notional_fraction * avg_weight_tv.get(stream.profile_id, 0.0) for stream in profile_streams))
    row = grid_hybrid._metric_row_from_stream(
        profile_id=profile_id,
        profile_kind="optuna_v3_5_style_train_validation_selected" if version == "v3_5" else "optuna_v3_6_style_train_validation_selected",
        candidate_tier="optuna_hybrid_relaxed_paper_testnet_candidate",
        leverage_map={},
        weights=avg_weight_tv,
        gross_notional_fraction=gross_notional,
        returns=returns,
        turnover_by_split=turnover,
        trade_events_by_split=trade_events,
        liquidation_count_by_split=liquidation,
        strict_promotion_profile=False,
        promotion_gate_pass=False,
        paper_testnet_candidate=False,
    )
    selection_reasons = grid_hybrid._selection_reasons(row)
    report_only_reasons = grid_hybrid._report_only_gate_reasons(row)
    row.update(
        {
            "optimizer": "optuna_tpe",
            "hybrid_version": version,
            "best_value": None if optuna is None else optuna.get("best_value"),
            "best_params": asdict(params),
            "learned_params": asdict(learned),
            "weights": avg_weight_tv,
            "average_weights_train_validation": avg_weight_tv,
            "average_weights_train": _weights_summary(profile_streams=profile_streams, weights_frame=weights_frame, split="train"),
            "average_weights_validation": _weights_summary(profile_streams=profile_streams, weights_frame=weights_frame, split="validation"),
            "average_weights_locked_oos_report_only": _weights_summary(
                profile_streams=profile_streams,
                weights_frame=weights_frame,
                split="locked_oos",
            ),
            "final_weights": {label: float(weights_frame.iloc[-1][label]) for label in labels},
            "asset_gross_notional_fraction": _dynamic_asset_gross(profile_streams=profile_streams, avg_weights=avg_weight_tv),
            "selection_reasons": selection_reasons,
            "report_only_gate_reasons": report_only_reasons,
        }
    )
    row["paper_testnet_candidate"] = not selection_reasons and not report_only_reasons
    row["ready_for_paper"] = row["paper_testnet_candidate"]
    row["ready_for_real"] = False
    row["real_money_execution"] = False
    return OptunaModelResult(
        row=row,
        returns=returns,
        weights=weights_frame,
        allocations=allocations,
        learned_params=learned,
        params=params,
        optuna=optuna,
        top_trials=top_trials,
    )


def _objective_score(row: Mapping[str, Any]) -> float:
    # Deliberately train+validation only. locked-OOS report fields are ignored
    # here and by _selection_reasons; tests lock this contract.
    base = grid_hybrid._train_validation_score(row)
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -1e9)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -1e9)
    penalty = 0.0
    if train <= 0.0:
        penalty += 10.0 + abs(train) * 20.0
    if validation < MIN_VALIDATION_RETURN:
        penalty += 8.0 + (MIN_VALIDATION_RETURN - validation) * 20.0
    if train < validation:
        penalty += 4.0 + (validation - train) * 8.0
    if _safe_float(row.get("validation_mdd")) > MAX_VALIDATION_MDD:
        penalty += 5.0 + (_safe_float(row.get("validation_mdd")) - MAX_VALIDATION_MDD) * 10.0
    if _safe_float(row.get("train_mdd")) > MAX_TRAIN_MDD:
        penalty += 2.0 + (_safe_float(row.get("train_mdd")) - MAX_TRAIN_MDD) * 5.0
    if train_rpt <= ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        penalty += 4.0 + max(0.0, ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS - train_rpt) / 5.0
    if val_rpt <= ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS:
        penalty += 4.0 + max(0.0, ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS - val_rpt) / 5.0
    return float(base - penalty)


def _run_optuna(
    profile_streams: Sequence[grid_hybrid.ProfileStream],
    *,
    version: str,
    n_trials: int,
    seed: int,
) -> OptunaModelResult:
    import optuna
    from optuna.samplers import TPESampler

    profile_id = V35_PROFILE_ID if version == "v3_5" else V36_PROFILE_ID

    def objective(trial: Any) -> float:
        result = _run_model(
            profile_streams,
            _params_from_trial(trial),
            version=version,
            profile_id=profile_id,
        )
        trial.set_user_attr("train_return", _safe_float(result.row.get("train_return")))
        trial.set_user_attr("validation_return", _safe_float(result.row.get("validation_return")))
        trial.set_user_attr("train_mdd", _safe_float(result.row.get("train_mdd")))
        trial.set_user_attr("validation_mdd", _safe_float(result.row.get("validation_mdd")))
        trial.set_user_attr("train_rpt_bps", _safe_float(result.row.get("train_return_per_turnover_proxy_bps")))
        trial.set_user_attr("validation_rpt_bps", _safe_float(result.row.get("validation_return_per_turnover_proxy_bps")))
        trial.set_user_attr("selection_reasons", list(grid_hybrid._selection_reasons(result.row)))
        return _objective_score(result.row)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=int(seed), n_startup_trials=min(32, max(1, int(n_trials) // 4))),
    )
    default = HybridParams()
    study.enqueue_trial(
        {
            "bias_alpha": default.bias_correction_alpha,
            "bias_combine_ratio": default.bias_combine_ratio,
            "max_weight": default.max_single_weight,
            "mape_window": default.mape_window,
            "bias_window": default.bias_window,
            "short_vol_window": default.short_vol_window,
        }
    )
    study.optimize(objective, n_trials=max(1, int(n_trials)), show_progress_bar=False, gc_after_trial=True)
    best_params = HybridParams(
        bias_correction_alpha=float(study.best_params["bias_alpha"]),
        bias_combine_ratio=float(study.best_params["bias_combine_ratio"]),
        max_single_weight=float(study.best_params["max_weight"]),
        mape_window=int(study.best_params["mape_window"]),
        bias_window=int(study.best_params["bias_window"]),
        short_vol_window=int(study.best_params["short_vol_window"]),
    )
    top_trials: list[dict[str, Any]] = []
    for trial in sorted(study.trials, key=lambda t: float(t.value) if t.value is not None else -1e18, reverse=True)[:30]:
        top_trials.append(
            {
                "hybrid_version": version,
                "trial_number": int(trial.number),
                "value": None if trial.value is None else float(trial.value),
                "state": str(trial.state.name),
                "params": dict(trial.params),
                "train_return": trial.user_attrs.get("train_return"),
                "validation_return": trial.user_attrs.get("validation_return"),
                "train_mdd": trial.user_attrs.get("train_mdd"),
                "validation_mdd": trial.user_attrs.get("validation_mdd"),
                "train_rpt_bps": trial.user_attrs.get("train_rpt_bps"),
                "validation_rpt_bps": trial.user_attrs.get("validation_rpt_bps"),
                "selection_reasons": trial.user_attrs.get("selection_reasons", []),
            }
        )
    optuna_payload = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "direction": "maximize_train_validation_score",
        "sampler": "TPESampler",
        "seed": int(seed),
        "selection_inputs": ["train", "validation"],
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "locked_oos_objective_columns_used": [],
        "external_method_source": str(
            Path("/home/hoky/DeepLearning/ensemble_strategies/models/hybrid")
            / ("v3_5.py" if version == "v3_5" else "v3_6.py")
        ),
    }
    final = _run_model(
        profile_streams,
        best_params,
        version=version,
        profile_id=profile_id,
        optuna=optuna_payload,
        top_trials=top_trials,
    )
    final.row["optuna"] = optuna_payload
    return final


def _grid_baseline_row(profile_streams: Sequence[grid_hybrid.ProfileStream]) -> dict[str, Any]:
    row, _ = grid_hybrid.select_hybrid_row(profile_streams)
    row = dict(row)
    row["optimizer"] = "coarse_5pct_grid_baseline_not_selected_by_this_runner"
    row["hybrid_version"] = "grid_baseline"
    row["best_value"] = None
    row["best_params"] = {"weight_step": grid_hybrid.WEIGHT_STEP, "min_profile_weight": grid_hybrid.MIN_PROFILE_WEIGHT}
    row["learned_params"] = {}
    row["average_weights_train_validation"] = row.get("weights", {})
    row["final_weights"] = row.get("weights", {})
    return row


def _base_rows(
    profile_streams: Sequence[grid_hybrid.ProfileStream],
    source_profile_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for stream, source_row in zip(profile_streams, source_profile_rows, strict=True):
        row = grid_hybrid._base_comparison_row(stream, source_row)
        row["optimizer"] = "source_profile"
        row["hybrid_version"] = "source_profile"
        row["best_value"] = None
        row["best_params"] = {}
        row["learned_params"] = {}
        row["average_weights_train_validation"] = row.get("weights", {})
        row["final_weights"] = row.get("weights", {})
        rows.append(row)
    return rows


def _choose_selected_optuna(results: Sequence[OptunaModelResult]) -> dict[str, Any]:
    train_val_pass = [result.row for result in results if not grid_hybrid._selection_reasons(result.row)]
    pool = train_val_pass or [result.row for result in results]
    # locked-OOS is not in the sort key. Report-only gate may still reject the
    # candidate after the frozen train+validation selection.
    return max(pool, key=lambda row: _objective_score(row))


def _render_pct(value: Any) -> str:
    return f"{_safe_float(value):.4%}"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Integer-Leverage Optuna Hybrid Decision",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Method correction",
        "",
        "- This artifact replaces the prior coarse 5% grid as the optimization decision surface.",
        "- Optuna/TPESampler tunes v3.5- and v3.6-style hybrid parameters, matching the existing `run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py` pattern.",
        "- v3.5 mapping: warmup-learned default profile + rolling return/error weights + high-vol boost + bias/exposure dampening.",
        "- v3.6 mapping: v3.5 mechanics plus online default-profile refresh from rolling scores.",
        "- Objective/learning/selection inputs: train + validation only. locked-OOS is report-only after frozen Optuna params.",
        "",
        "## Comparison",
        "",
        "| Profile | Version | Optimizer | Weights/avg TV weights | Gross | Train | Val | OOS report-only | Val MDD | OOS MDD | RPT T/V/OOS bps | Paper candidate |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["comparison_rows"]:
        weights = row.get("average_weights_train_validation") or row.get("weights") or {}
        lines.append(
            f"| `{row['profile_id']}` | `{row.get('hybrid_version', '')}` | `{row.get('optimizer', '')}` | "
            f"`{json.dumps(_json_safe(weights), sort_keys=True)}` | "
            f"{_safe_float(row['gross_notional_fraction']):.2f}x | "
            f"{_render_pct(row['train_return'])} | "
            f"{_render_pct(row['validation_return'])} | "
            f"{_render_pct(row['locked_oos_return_report_only'])} | "
            f"{_render_pct(row['validation_mdd'])} | "
            f"{_render_pct(row['locked_oos_mdd_report_only'])} | "
            f"{_safe_float(row['train_return_per_turnover_proxy_bps']):.2f}/"
            f"{_safe_float(row['validation_return_per_turnover_proxy_bps']):.2f}/"
            f"{_safe_float(row['locked_oos_return_per_turnover_proxy_bps_report_only']):.2f} | "
            f"{str(bool(row['paper_testnet_candidate'])).lower()} |"
        )
    selected = payload["selected_optuna_hybrid_profile"]
    lines.extend(
        [
            "",
            "## Selected Optuna hybrid",
            "",
            f"- profile: `{selected['profile_id']}`",
            f"- hybrid version: `{selected['hybrid_version']}`",
            f"- avg train+validation weights: `{json.dumps(_json_safe(selected['average_weights_train_validation']), sort_keys=True)}`",
            f"- final weights: `{json.dumps(_json_safe(selected['final_weights']), sort_keys=True)}`",
            f"- train/validation/OOS report-only: `{_render_pct(selected['train_return'])}` / "
            f"`{_render_pct(selected['validation_return'])}` / `{_render_pct(selected['locked_oos_return_report_only'])}`",
            f"- validation MDD / OOS MDD: `{_render_pct(selected['validation_mdd'])}` / "
            f"`{_render_pct(selected['locked_oos_mdd_report_only'])}`",
            f"- RPT bps train/validation/OOS: `{_safe_float(selected['train_return_per_turnover_proxy_bps']):.2f}` / "
            f"`{_safe_float(selected['validation_return_per_turnover_proxy_bps']):.2f}` / "
            f"`{_safe_float(selected['locked_oos_return_per_turnover_proxy_bps_report_only']):.2f}`",
            f"- selection reasons: `{selected['selection_reasons']}`",
            f"- report-only OOS gate reasons: `{selected['report_only_gate_reasons']}`",
            "",
            "## Governance",
            "",
            f"- primary round-trip cost bps: `{payload['research_primary_round_trip_cost_bps']}`",
            f"- return-per-turnover threshold bps: `{payload['return_per_turnover_threshold_bps']}`",
            "- ready_for_real: `false`",
            "- real_money_execution: `false`",
            "- real_execution_allowed: `false`",
            f"- locked-OOS used for selection: `{payload['selection_policy']['uses_locked_oos_for_selection']}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload_from_inputs(
    *,
    integer_payload: Mapping[str, Any],
    output_dir: Path,
    integer_artifact_path: Path,
    data_root: Path,
    feature_root: Path,
    n_trials: int,
    seed: int,
    write_outputs: bool = True,
) -> dict[str, Any]:
    profile_streams, source_profile_rows = _source_profile_streams(
        integer_payload=integer_payload,
        data_root=data_root,
        feature_root=feature_root,
    )
    base_rows = _base_rows(profile_streams, source_profile_rows)
    grid_row = _grid_baseline_row(profile_streams)
    v35 = _run_optuna(profile_streams, version="v3_5", n_trials=n_trials, seed=seed)
    v36 = _run_optuna(profile_streams, version="v3_6", n_trials=n_trials, seed=seed + 1)
    selected = _choose_selected_optuna([v35, v36])
    comparison_rows = [*base_rows, grid_row, v35.row, v36.row]
    timestamp = _timestamp()
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    latest_json = output_dir / "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_integer_leverage_optuna_hybrid_decision_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.md"
    comparison_csv = output_dir / "integer_leverage_optuna_hybrid_comparison_latest.csv"
    trial_csv = output_dir / "integer_leverage_optuna_hybrid_top_trials_latest.csv"
    methodology_md = output_dir / "integer_leverage_optuna_hybrid_methodology_latest.md"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    trial_rows = [*v35.top_trials, *v36.top_trials]
    payload: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at_utc": _utc_now_iso(),
        "source_integer_portfolio_artifact": str(integer_artifact_path),
        "source_grid_baseline_artifact": str(grid_hybrid.DEFAULT_OUTPUT_DIR / "alpha_zoo_integer_leverage_hybrid_decision_latest.json"),
        "research_primary_round_trip_cost_bps": ilp.PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": ilp.AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "bbo_spread_multiplier": ilp.BBO_SPREAD_MULTIPLIER,
        "return_per_turnover_threshold_bps": ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "ready_for_paper": bool(selected.get("paper_testnet_candidate")),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "paper_testnet_only": True,
        "optuna_hybrid_policy": {
            "source_profile_ids": [stream.profile_id for stream in profile_streams],
            "optimizer": "Optuna TPESampler",
            "n_trials_per_version": int(n_trials),
            "v3_5_mapping": "warmup-learned default profile + rolling return/error weights + high-volatility boost + bias/exposure dampening",
            "v3_6_mapping": "v3.5 mechanics plus online adaptive default-profile selection from rolling score evidence",
            "parameter_space_source": "scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py::_params_from_trial",
            "grid_baseline_role": "comparison_only_not_the_selected_optimizer",
            "selection_inputs": ["train", "validation"],
            "locked_oos_role": "gate/report-only after train+validation Optuna params freeze",
        },
        "selection_policy": {
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "no_calendar_date_hack": True,
            "model_id_source": "source integer profile rows derived from frozen corr decision artifact",
        },
        "selected_optuna_hybrid_profile": selected,
        "comparison_rows": comparison_rows,
        "hybrid_v3_5_optuna": {"row": v35.row, "optuna": v35.optuna, "top_trials": list(v35.top_trials), "allocation_samples": v35.allocations[:50]},
        "hybrid_v3_6_optuna": {"row": v36.row, "optuna": v36.optuna, "top_trials": list(v36.top_trials), "allocation_samples": v36.allocations[:50]},
        "profile_train_validation_corr_matrix": grid_hybrid._profile_corr_matrix(profile_streams, split="train_validation"),
        "profile_validation_corr_matrix": grid_hybrid._profile_corr_matrix(profile_streams, split="validation"),
        "profile_locked_oos_corr_matrix_report_only": grid_hybrid._profile_corr_matrix(profile_streams, split="locked_oos"),
        "runner_peak_rss_mib": local_peak_mib,
        "memory_summary": {"limit_mib": 8192.0, "pass_under_8gb": local_peak_mib < 8192.0},
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "comparison_csv": str(comparison_csv),
            "top_trials_csv": str(trial_csv),
            "methodology_markdown": str(methodology_md),
            "artifact_generation_validation_log": str(generation_log),
        },
    }
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(latest_json, payload)
        _write_json(timestamped_json, payload)
        latest_md.write_text(_render_markdown(payload), encoding="utf-8")
        _write_csv(comparison_csv, comparison_rows, COMPARISON_FIELDS)
        _write_csv(trial_csv, trial_rows, TRIAL_FIELDS)
        methodology_md.write_text(
            "# Integer-Leverage Optuna Hybrid Methodology\n\n"
            "- Corrects the previous coarse-grid hybrid: Optuna/TPESampler is the optimizer for the selected hybrid.\n"
            "- Source: three paper/testnet integer-leverage profiles from the frozen corr/integer artifact.\n"
            "- PnL streams: reconstructed from the same fixed position-state rules and integer asset leverage maps.\n"
            "- Cost: 10bps all-in round-trip backtest friction proxy is embedded before hybriding.\n"
            "- v3.5: warmup-learned default profile, rolling return/error weights, high-vol boost, max-weight cap, bias/exposure dampening.\n"
            "- v3.6: v3.5 mechanics plus online adaptive default-profile refresh from rolling score evidence.\n"
            "- Optuna space mirrors run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py: bias alpha/combine, max weight, mape window, bias window, short-vol window.\n"
            "- Objective/learning/selection: train+validation only. locked-OOS is not read for discovery, pruning, objective, fitting, or selection.\n"
            "- Real money: blocked. Paper/testnet only; monitoring must record BBO spread, all-in fee/slippage, liquidation-inclusive MDD, account wipeout, and replay/live notional parity.\n",
            encoding="utf-8",
        )
        generation_log.write_text(
            f"artifact_kind={ARTIFACT_KIND}\n"
            f"source_profile_count={len(source_profile_rows)}\n"
            f"n_trials_per_version={int(n_trials)}\n"
            f"selected_optuna_hybrid_profile={selected['profile_id']}\n"
            f"selected_hybrid_version={selected['hybrid_version']}\n"
            f"ready_for_paper={payload['ready_for_paper']}\n"
            f"ready_for_real={payload['ready_for_real']}\n"
            f"real_money_execution={payload['real_money_execution']}\n"
            f"locked_oos_used_for_selection={payload['selection_policy']['uses_locked_oos_for_selection']}\n"
            f"runner_peak_rss_mib={local_peak_mib:.2f}\n",
            encoding="utf-8",
        )
    return payload


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    integer_artifact = Path(args.integer_portfolio_artifact).expanduser().resolve()
    return build_payload_from_inputs(
        integer_payload=ilp._load_json(integer_artifact),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        integer_artifact_path=integer_artifact,
        data_root=Path(args.data_root).expanduser().resolve(),
        feature_root=Path(args.feature_root).expanduser().resolve(),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        write_outputs=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integer-portfolio-artifact", default=str(DEFAULT_INTEGER_PORTFOLIO_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(ilp.DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(ilp.DEFAULT_FEATURE_ROOT))
    parser.add_argument("--n-trials", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260524)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    selected = payload["selected_optuna_hybrid_profile"]
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "selected_optuna_hybrid_profile": selected,
                    "ready_for_paper": payload["ready_for_paper"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
