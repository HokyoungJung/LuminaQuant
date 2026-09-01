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
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.alpha_zoo.native_hybrid_optuna_backend import (  # noqa: E402
    evaluate_hybrid_portfolio_native,
)
from lumina_quant.optimization.search_policy import (  # noqa: E402
    optimization_search_policy_payload,
    run_optuna_study,
    suggest_params_from_optuna_config,
)
from scripts.research import run_alpha_zoo_integer_leverage_hybrid_decision as grid_hybrid  # noqa: E402

ilp = grid_hybrid.ilp

DEFAULT_INTEGER_PORTFOLIO_ARTIFACT = grid_hybrid.DEFAULT_INTEGER_PORTFOLIO_ARTIFACT
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_integer_leverage_optuna_hybrid_decision_20260524"
)
DEFAULT_STANDARD_LIVE_REFIT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_standard_live_refit_20260528"
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


HYBRID_OPTUNA_CONFIG: dict[str, dict[str, Any]] = {
    "bias_alpha": {"type": "float", "low": 0.85, "high": 1.0},
    "bias_combine_ratio": {"type": "float", "low": 0.15, "high": 0.50},
    "max_weight": {"type": "float", "low": 0.55, "high": 0.95},
    "mape_window": {"type": "int", "low": 18, "high": 60},
    "bias_window": {"type": "int", "low": 6, "high": 30},
    "short_vol_window": {"type": "int", "low": 4, "high": 24},
    "warmup_ratio": {"type": "float", "low": 0.40, "high": 0.80},
    "min_boost": {"type": "float", "low": 0.00, "high": 0.18},
    "max_boost": {"type": "float", "low": 0.16, "high": 0.50},
    "high_vol_threshold_quantile": {"type": "float", "low": 55.0, "high": 90.0},
    "high_vol_boost_base": {"type": "float", "low": 0.00, "high": 0.20},
    "high_vol_boost_scale": {"type": "float", "low": 0.15, "high": 0.85},
    "default_weight_ratio_floor": {"type": "float", "low": 0.15, "high": 0.50},
    "default_weight_ratio_ceiling": {"type": "float", "low": 0.50, "high": 0.90},
    "default_weight_ratio_steps": {"type": "int", "low": 3, "high": 7},
}

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
    high_vol_threshold_quantile: float = 75.0
    high_vol_boost_base: float = 0.10
    high_vol_boost_scale: float = 0.50
    default_weight_ratio_floor: float = 0.30
    default_weight_ratio_ceiling: float = 0.70
    default_weight_ratio_steps: int = 5


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
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _safe_float(value: Any, default: float = 0.0) -> float:
    return grid_hybrid._safe_float(value, default)


def _period_return(values: np.ndarray | pd.Series) -> float:
    return grid_hybrid._period_return(values)


_DEFAULT_ILP_SPLIT_MASK = ilp._split_mask
_DEFAULT_CORR_SPLIT_MASK = getattr(ilp.corr, "_split_mask", None)
_ACTIVE_SPLIT_WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None


def _coerce_split_windows(
    split_windows: Mapping[str, Any] | None,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None:
    if not split_windows:
        return None
    out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split in ilp.SPLIT_ORDER:
        raw = split_windows.get(split)
        if raw is None:
            continue
        if isinstance(raw, Mapping):
            start_raw = raw.get("start")
            end_raw = raw.get("end")
        else:
            start_raw, end_raw = raw
        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)
        if start.tzinfo is not None:
            start = start.tz_convert("UTC").tz_localize(None)
        if end.tzinfo is not None:
            end = end.tz_convert("UTC").tz_localize(None)
        out[split] = (start, end)
    missing = [split for split in ilp.SPLIT_ORDER if split not in out]
    if missing:
        raise ValueError(f"split window override missing splits: {missing}")
    return out


def _split_window_payload(
    split_windows: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> dict[str, dict[str, Any]]:
    if split_windows is None:
        return {
            split: {
                "start": str(start),
                "end": str(end),
                "enabled": bool(start <= end),
            }
            for split, (start, end) in getattr(ilp, "SPLITS", {}).items()
        }
    payload: dict[str, dict[str, Any]] = {}
    for split, (start, end) in split_windows.items():
        payload[split] = {
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
            "enabled": bool(start <= end),
            "role": "gate_report_only" if split == "locked_oos" else "selection_evidence",
        }
    return payload


def _split_mask(index: pd.Series | pd.DatetimeIndex, split: str) -> np.ndarray:
    if _ACTIVE_SPLIT_WINDOWS is None:
        return _DEFAULT_ILP_SPLIT_MASK(index, split)
    start, end = _ACTIVE_SPLIT_WINDOWS[split]
    values = pd.Series(index) if not isinstance(index, pd.Series) else index
    ts = pd.to_datetime(values)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ((ts >= start) & (ts <= end)).to_numpy()


@contextmanager
def _split_window_context(split_windows: Mapping[str, Any] | None):
    global _ACTIVE_SPLIT_WINDOWS
    coerced = _coerce_split_windows(split_windows)
    if coerced is None:
        yield
        return
    previous = _ACTIVE_SPLIT_WINDOWS
    previous_ilp = ilp._split_mask
    previous_corr = getattr(ilp.corr, "_split_mask", None)
    _ACTIVE_SPLIT_WINDOWS = coerced
    ilp._split_mask = _split_mask
    if previous_corr is not None:
        ilp.corr._split_mask = _split_mask
    try:
        yield
    finally:
        _ACTIVE_SPLIT_WINDOWS = previous
        ilp._split_mask = previous_ilp
        if previous_corr is not None:
            ilp.corr._split_mask = previous_corr


def _split_metrics(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    total = _period_return(arr)
    return {
        "total_return": total,
        "max_drawdown": ilp.max_drawdown(arr),
        "return_mdd": total / ilp.max_drawdown(arr)
        if ilp.max_drawdown(arr) > 1e-12
        else (float("inf") if total > 0 else 0.0),
        "mean_return": float(np.nanmean(arr)) if arr.size else 0.0,
        "volatility": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
        "downside_volatility": float(np.sqrt(np.nanmean(np.square(np.where(arr < 0.0, arr, 0.0)))))
        if arr.size
        else 0.0,
        "active_return_bars": int(np.count_nonzero(np.abs(arr) > 1e-12)),
    }


def _source_profile_streams(
    *,
    integer_payload: Mapping[str, Any],
    data_root: Path,
    feature_root: Path,
) -> tuple[list[grid_hybrid.ProfileStream], list[Mapping[str, Any]]]:
    if (
        integer_payload.get("ready_for_real") is not False
        or integer_payload.get("real_money_execution") is not False
    ):
        raise ValueError("integer portfolio artifact violates real-money disabled guard")
    if _safe_float(integer_payload.get("research_primary_round_trip_cost_bps")) != 10.0:
        raise ValueError(
            "integer portfolio artifact is not using the primary 10bps round-trip cost"
        )
    source_profile_rows = list(integer_payload.get("paper_testnet_candidate_profiles") or [])
    if len(source_profile_rows) != 3:
        raise ValueError(
            f"expected exactly three paper/testnet source profiles, found {len(source_profile_rows)}"
        )

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
    union_index = pd.DatetimeIndex(
        sorted(set().union(*(set(replay.datetimes) for replay in replays)))
    )
    profile_streams = [
        grid_hybrid._profile_stream_from_row(
            row, replays_by_model_id=replays_by_model_id, union_index=union_index
        )
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


def _rolling_feature(
    returns: np.ndarray, end: int, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _candidate_scores(
    returns: np.ndarray, end: int, window: int, priors: np.ndarray, prior_ratio: float
) -> np.ndarray:
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
    downside = (
        float(np.sqrt(np.nanmean(np.square(np.where(arr < 0.0, arr, 0.0))))) if arr.size else 0.0
    )
    sharpe = mean / vol * math.sqrt(24.0 * 365.0) if vol > 1e-12 else 0.0
    sortino = mean / downside * math.sqrt(24.0 * 365.0) if downside > 1e-12 else 0.0
    calmar = total / mdd if mdd > 1e-12 else (float("inf") if total > 0 else 0.0)
    return {
        "total_return": total,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }


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
    native = None
    if initial_portfolio_history is None:
        native = evaluate_hybrid_portfolio_native(
            returns,
            version=version,
            start_idx=start_idx,
            mape_window=max(2, params.mape_window),
            bias_window=max(1, params.bias_window),
            short_vol_window=max(2, params.short_vol_window),
            bias_correction_alpha=params.bias_correction_alpha,
            bias_combine_ratio=params.bias_combine_ratio,
            max_single_weight=params.max_single_weight,
            default_idx=learned.default_idx,
            high_vol_best_idx=learned.high_vol_best_idx,
            default_weight_ratio=learned.default_weight_ratio,
            high_vol_threshold=learned.high_vol_threshold,
            high_vol_weight_boost=learned.high_vol_weight_boost,
        )
    if native is not None:
        portfolio, weights_out, raw_weights, default_indices, high_vol_features, exposures = native
        stride = max(1, int(allocation_stride))
        allocations = []
        for t in range(max(0, int(start_idx)), returns.shape[0]):
            if t == returns.shape[0] - 1 or t % stride == 0:
                allocations.append(
                    {
                        "index": int(t),
                        "default_idx": int(default_indices[t]),
                        "high_vol_feature": float(high_vol_features[t]),
                        "exposure": float(exposures[t]),
                        "adaptive_weight_ratio": float(learned.default_weight_ratio),
                        "adaptive_high_vol_boost": float(learned.high_vol_weight_boost),
                        "adaptive_max_single_weight": float(params.max_single_weight),
                        "weights": [float(x) for x in raw_weights[t]],
                    }
                )
        return portfolio, weights_out, allocations

    k = returns.shape[1]
    out = np.zeros(returns.shape[0], dtype=float)
    weights_out = np.zeros((returns.shape[0], k), dtype=float)
    allocations: list[dict[str, Any]] = []
    portfolio_history = list(initial_portfolio_history or [])
    prior_scores = _candidate_scores(
        returns, max(1, start_idx), max(2, params.mape_window), np.zeros(k), 0.0
    )
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
        recent = returns[max(0, t - max(2, params.short_vol_window)) : t]
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
            ens = np.asarray(portfolio_history[-int(params.bias_window) :], dtype=float)
            model_hist = returns[max(0, t - int(params.bias_window)) : t, default_idx]
            ens_bias = float(np.nanmean(ens)) if ens.size else 0.0
            model_bias = float(np.nanmean(model_hist)) if model_hist.size else 0.0
            combined_bias = (
                params.bias_combine_ratio * model_bias
                + (1.0 - params.bias_combine_ratio) * ens_bias
            )
            denom = float(np.nanmean(np.abs(ens))) + 1e-9
            if combined_bias < 0.0:
                exposure = max(
                    0.0, 1.0 - params.bias_correction_alpha * min(0.80, abs(combined_bias) / denom)
                )
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


def _rolling_nanstd(values: np.ndarray, window: int) -> np.ndarray:
    """Return nan-aware population std for fixed-width trailing windows.

    Window ``i`` covers ``values[i : i + window]``.  This replaces repeated
    ``np.nanstd`` calls in Optuna warmup learning without changing the ddof=0
    semantics used by the original loop.
    """
    width = max(1, int(window))
    arr = np.asarray(values, dtype=float)
    if arr.size < width:
        return np.asarray([], dtype=float)
    finite = np.isfinite(arr)
    clean = np.where(finite, arr, 0.0)
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    sums = np.concatenate(([0.0], np.cumsum(clean)))
    sq_sums = np.concatenate(([0.0], np.cumsum(clean * clean)))

    window_counts = counts[width:] - counts[:-width]
    window_sums = sums[width:] - sums[:-width]
    window_sq_sums = sq_sums[width:] - sq_sums[:-width]
    mean = np.divide(
        window_sums,
        window_counts,
        out=np.zeros_like(window_sums, dtype=float),
        where=window_counts > 0,
    )
    variance = (
        np.divide(
            window_sq_sums,
            window_counts,
            out=np.zeros_like(window_sq_sums, dtype=float),
            where=window_counts > 0,
        )
        - mean * mean
    )
    return np.sqrt(np.maximum(variance, 0.0))


def _learn_params(
    returns: np.ndarray,
    params: HybridParams,
    opt_indices: np.ndarray,
    *,
    warmup_indices: np.ndarray | None = None,
) -> LearnedParams:
    # The objective fit window may include validation for scoring/final-refit
    # evidence, but warmup state learning is train-only by default.
    opt = returns[warmup_indices] if warmup_indices is not None else returns[opt_indices]
    n = opt.shape[0]
    warmup_n = max(10, min(n, int(n * params.warmup_ratio))) if n else 0
    warmup = opt[:warmup_n]
    if warmup.size == 0:
        return LearnedParams(0.0, 0, 0, 0.5, params.min_boost, 0.0)
    mean = np.nanmean(warmup, axis=0)
    std = np.nanstd(warmup, axis=0, ddof=1) if warmup.shape[0] > 1 else np.zeros(warmup.shape[1])
    scores = np.where(np.isfinite(mean / (std + 1e-9)), mean / (std + 1e-9), 0.0)
    default_idx = int(np.nanargmax(scores))
    start_t = max(2, params.short_vol_window)
    row_mean = np.nanmean(warmup, axis=1)
    rolling_vol = _rolling_nanstd(row_mean, int(params.short_vol_window))
    if start_t < int(params.short_vol_window):
        vol_series = rolling_vol[: max(0, warmup_n - start_t)]
    else:
        offset = start_t - int(params.short_vol_window)
        vol_series = rolling_vol[offset : offset + max(0, warmup_n - start_t)]
    threshold = (
        float(np.nanpercentile(vol_series, np.clip(params.high_vol_threshold_quantile, 1.0, 99.0)))
        if vol_series.size
        else 0.0
    )
    hv_positions = np.flatnonzero(vol_series > threshold) + start_t
    if hv_positions.size:
        hv_mean = np.nanmean(warmup[hv_positions], axis=0)
        high_vol_best = int(np.nanargmax(np.where(np.isfinite(hv_mean), hv_mean, -1e9)))
    else:
        hv_mean = np.asarray([], dtype=float)
        high_vol_best = default_idx
    hv_gap = 0.0
    if hv_positions.size:
        best = float(hv_mean[high_vol_best])
        finite_others = np.delete(hv_mean, high_vol_best)
        finite_others = finite_others[np.isfinite(finite_others)]
        if finite_others.size:
            avg_other = float(np.nanmean(finite_others))
            hv_gap = max(0.0, (best - avg_other) / (abs(avg_other) + abs(best) + 1e-9))
    boost = float(
        np.clip(
            params.high_vol_boost_base + hv_gap * params.high_vol_boost_scale,
            params.min_boost,
            params.max_boost,
        )
    )
    ratio_floor = float(min(params.default_weight_ratio_floor, params.default_weight_ratio_ceiling))
    ratio_ceiling = float(
        max(params.default_weight_ratio_floor, params.default_weight_ratio_ceiling)
    )
    ratio_steps = max(2, int(params.default_weight_ratio_steps))
    ratios = tuple(float(x) for x in np.linspace(ratio_floor, ratio_ceiling, ratio_steps))
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
        score = (
            _safe_float(metrics.get("calmar"))
            + _safe_float(metrics.get("sharpe"))
            + _safe_float(metrics.get("sortino"))
        )
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
    suggested = suggest_params_from_optuna_config(trial, HYBRID_OPTUNA_CONFIG)
    min_boost = float(suggested["min_boost"])
    max_boost = max(min_boost + 0.01, float(suggested["max_boost"]))
    ratio_floor = float(suggested["default_weight_ratio_floor"])
    ratio_ceiling = max(ratio_floor + 0.05, float(suggested["default_weight_ratio_ceiling"]))
    return HybridParams(
        bias_correction_alpha=float(suggested["bias_alpha"]),
        bias_combine_ratio=float(suggested["bias_combine_ratio"]),
        max_single_weight=float(suggested["max_weight"]),
        mape_window=int(suggested["mape_window"]),
        bias_window=int(suggested["bias_window"]),
        short_vol_window=int(suggested["short_vol_window"]),
        warmup_ratio=float(suggested["warmup_ratio"]),
        min_boost=min_boost,
        max_boost=max_boost,
        high_vol_threshold_quantile=float(suggested["high_vol_threshold_quantile"]),
        high_vol_boost_base=float(suggested["high_vol_boost_base"]),
        high_vol_boost_scale=float(suggested["high_vol_boost_scale"]),
        default_weight_ratio_floor=ratio_floor,
        default_weight_ratio_ceiling=ratio_ceiling,
        default_weight_ratio_steps=int(suggested["default_weight_ratio_steps"]),
    )


def _trial_params_from_hybrid(params: HybridParams) -> dict[str, Any]:
    return {
        "bias_alpha": params.bias_correction_alpha,
        "bias_combine_ratio": params.bias_combine_ratio,
        "max_weight": params.max_single_weight,
        "mape_window": params.mape_window,
        "bias_window": params.bias_window,
        "short_vol_window": params.short_vol_window,
        "warmup_ratio": params.warmup_ratio,
        "min_boost": params.min_boost,
        "max_boost": params.max_boost,
        "high_vol_threshold_quantile": params.high_vol_threshold_quantile,
        "high_vol_boost_base": params.high_vol_boost_base,
        "high_vol_boost_scale": params.high_vol_boost_scale,
        "default_weight_ratio_floor": params.default_weight_ratio_floor,
        "default_weight_ratio_ceiling": params.default_weight_ratio_ceiling,
        "default_weight_ratio_steps": params.default_weight_ratio_steps,
    }


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
                avg_weight_tv[stream.profile_id] = (
                    avg_weight_tv.get(stream.profile_id, 0.0)
                    + float(avg_weights.get(stream.profile_id, 0.0)) / 2.0
                )
        turnover_by_split[split] = float(
            sum(
                stream.turnover_by_split[split]
                * max(0.0, float(avg_weights.get(stream.profile_id, 0.0)))
                for stream in profile_streams
            )
        )
        active_streams = [
            stream
            for stream in profile_streams
            if float(avg_weights.get(stream.profile_id, 0.0)) > 1e-6
        ]
        trade_events_by_split[split] = int(
            sum(stream.trade_events_by_split[split] for stream in active_streams)
        )
        liquidation_by_split[split] = int(
            sum(stream.liquidation_count_by_split[split] for stream in active_streams)
        )
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
    fit_splits: Sequence[str] = ("train", "validation"),
    warmup_splits: Sequence[str] = ("train",),
    final_refit: bool = False,
    require_locked_oos_gate: bool = True,
) -> OptunaModelResult:
    labels = [stream.profile_id for stream in profile_streams]
    index = profile_streams[0].returns.index
    returns_matrix = np.column_stack(
        [
            stream.returns.reindex(index, fill_value=0.0).to_numpy(dtype=float)
            for stream in profile_streams
        ]
    )
    fit_mask = np.zeros(len(index), dtype=bool)
    for split in fit_splits:
        fit_mask |= _split_mask(index, str(split))
    opt_indices = np.flatnonzero(fit_mask)
    warmup_mask = np.zeros(len(index), dtype=bool)
    for split in warmup_splits:
        warmup_mask |= _split_mask(index, str(split))
    warmup_indices = np.flatnonzero(warmup_mask)
    learned = _learn_params(
        returns_matrix,
        params,
        opt_indices,
        warmup_indices=warmup_indices,
    )
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
    gross_notional = float(
        sum(
            stream.gross_notional_fraction * avg_weight_tv.get(stream.profile_id, 0.0)
            for stream in profile_streams
        )
    )
    row = grid_hybrid._metric_row_from_stream(
        profile_id=profile_id,
        profile_kind="optuna_v3_5_style_train_validation_selected"
        if version == "v3_5"
        else "optuna_v3_6_style_train_validation_selected",
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
    report_only_reasons = (
        grid_hybrid._report_only_gate_reasons(row) if require_locked_oos_gate else []
    )
    row.update(
        {
            "optimizer": "optuna_tpe",
            "hybrid_version": version,
            "best_value": None if optuna is None else optuna.get("best_value"),
            "best_params": asdict(params),
            "learned_params": asdict(learned),
            "fit_splits": [str(split) for split in fit_splits],
            "fit_bar_count": int(opt_indices.size),
            "warmup_splits": [str(split) for split in warmup_splits],
            "warmup_bar_count": int(warmup_indices.size),
            "warmup_policy": "warmup_ratio_applies_to_train_split_only",
            "final_refit": bool(final_refit),
            "locked_oos_gate_required": bool(require_locked_oos_gate),
            "test_set_policy": "locked_oos_gate_report_only"
            if require_locked_oos_gate
            else "disabled_for_live_final_refit_no_test_set_reserved",
            "weights": avg_weight_tv,
            "average_weights_train_validation": avg_weight_tv,
            "average_weights_train": _weights_summary(
                profile_streams=profile_streams, weights_frame=weights_frame, split="train"
            ),
            "average_weights_validation": _weights_summary(
                profile_streams=profile_streams, weights_frame=weights_frame, split="validation"
            ),
            "average_weights_locked_oos_report_only": _weights_summary(
                profile_streams=profile_streams,
                weights_frame=weights_frame,
                split="locked_oos",
            ),
            "final_weights": {label: float(weights_frame.iloc[-1][label]) for label in labels},
            "asset_gross_notional_fraction": _dynamic_asset_gross(
                profile_streams=profile_streams, avg_weights=avg_weight_tv
            ),
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
    fit_splits: Sequence[str] = ("train", "validation"),
    warmup_splits: Sequence[str] = ("train",),
    require_locked_oos_gate: bool = True,
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
            fit_splits=fit_splits,
            warmup_splits=warmup_splits,
            require_locked_oos_gate=require_locked_oos_gate,
        )
        trial.set_user_attr("train_return", _safe_float(result.row.get("train_return")))
        trial.set_user_attr("validation_return", _safe_float(result.row.get("validation_return")))
        trial.set_user_attr("train_mdd", _safe_float(result.row.get("train_mdd")))
        trial.set_user_attr("validation_mdd", _safe_float(result.row.get("validation_mdd")))
        trial.set_user_attr(
            "train_rpt_bps", _safe_float(result.row.get("train_return_per_turnover_proxy_bps"))
        )
        trial.set_user_attr(
            "validation_rpt_bps",
            _safe_float(result.row.get("validation_return_per_turnover_proxy_bps")),
        )
        trial.set_user_attr("selection_reasons", list(grid_hybrid._selection_reasons(result.row)))
        return _objective_score(result.row)

    default = HybridParams()
    sampler = TPESampler(seed=int(seed), n_startup_trials=min(32, max(1, int(n_trials) // 4)))
    study = run_optuna_study(
        optuna_module=optuna,
        objective=objective,
        n_trials=max(1, int(n_trials)),
        direction="maximize",
        seed=int(seed),
        sampler=sampler,
        enqueue_trials=[_trial_params_from_hybrid(default)],
        show_progress_bar=False,
    )
    best_params = _params_from_trial(study.best_trial)
    top_trials: list[dict[str, Any]] = []
    for trial in sorted(
        study.trials, key=lambda t: float(t.value) if t.value is not None else -1e18, reverse=True
    )[:30]:
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
        **optimization_search_policy_payload(
            search_method="optuna_tpe",
            objective_policy="maximize_train_validation_score_with_validation_holdout",
            selection_inputs=["train", "validation"],
            extra={
                "best_value": float(study.best_value),
                "best_params": dict(study.best_params),
                "n_trials": len(study.trials),
                "direction": "maximize_train_validation_score",
                "sampler": "TPESampler",
                "seed": int(seed),
                "fit_splits": [str(split) for split in fit_splits],
                "warmup_splits": [str(split) for split in warmup_splits],
                "warmup_policy": "warmup_ratio_applies_to_train_split_only",
                "search_space": HYBRID_OPTUNA_CONFIG,
            },
        ),
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
        fit_splits=fit_splits,
        warmup_splits=warmup_splits,
        require_locked_oos_gate=require_locked_oos_gate,
    )
    final.row["optuna"] = optuna_payload
    return final


def _grid_baseline_row(profile_streams: Sequence[grid_hybrid.ProfileStream]) -> dict[str, Any]:
    row, _ = grid_hybrid.select_hybrid_row(profile_streams)
    row = dict(row)
    row["optimizer"] = "coarse_5pct_grid_baseline_not_selected_by_this_runner"
    row["hybrid_version"] = "grid_baseline"
    row["best_value"] = None
    row["best_params"] = {
        "weight_step": grid_hybrid.WEIGHT_STEP,
        "min_profile_weight": grid_hybrid.MIN_PROFILE_WEIGHT,
    }
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


def _choose_selected_optuna_result(results: Sequence[OptunaModelResult]) -> OptunaModelResult:
    train_val_pass = [
        result for result in results if not grid_hybrid._selection_reasons(result.row)
    ]
    pool = train_val_pass or list(results)
    # locked-OOS is not in the sort key. Report-only gate may still reject the
    # candidate after the frozen train+validation selection.
    return max(pool, key=lambda result: _objective_score(result.row))


def _choose_selected_optuna(results: Sequence[OptunaModelResult]) -> dict[str, Any]:
    return _choose_selected_optuna_result(results).row


def _render_pct(value: Any) -> str:
    return f"{_safe_float(value):.4%}"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    standard_live_refit = bool(payload.get("standard_live_refit"))
    input_policy = (
        "- Standard live refit inputs: Optuna learns/optimizes on train only; "
        "validation is a recent holdout for scoring/selection; after selection, "
        "the frozen parameter set is final-refit on train+validation."
        if standard_live_refit
        else "- Objective/learning/selection inputs: train + validation only. "
        "locked-OOS is report-only after frozen Optuna params."
    )
    test_policy = (
        "- No locked test/OOS split is reserved for live final refit; live runtime "
        "uses frozen artifacts and still blocks real-money execution."
        if standard_live_refit
        else "- locked-OOS is attached after the Optuna params are frozen and remains gate/report-only."
    )
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
        input_policy,
        test_policy,
        "",
    ]
    plan = dict(dict(payload.get("data_coverage") or {}).get("standard_live_training_plan") or {})
    if plan:
        lines.extend(
            [
                "## Standard live-refit split",
                "",
                f"- train: `{plan.get('train', {}).get('start')}` → `{plan.get('train', {}).get('end')}`",
                f"- validation: `{plan.get('validation', {}).get('start')}` → `{plan.get('validation', {}).get('end')}`",
                f"- selection fit inputs: `{plan.get('selection_fit_inputs')}`",
                f"- final refit inputs: `{plan.get('final_refit_inputs')}`",
                f"- locked-OOS enabled: `{plan.get('locked_oos', {}).get('enabled')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison",
            "",
            "| Profile | Version | Optimizer | Weights/avg TV weights | Gross | Train | Val | OOS report-only | Val MDD | OOS MDD | RPT T/V/OOS bps | Paper candidate |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
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


def _selected_metric_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "profile_id",
        "hybrid_version",
        "optimizer",
        "train_return",
        "validation_return",
        "locked_oos_return_report_only",
        "train_mdd",
        "validation_mdd",
        "locked_oos_mdd_report_only",
        "train_return_per_turnover_proxy_bps",
        "validation_return_per_turnover_proxy_bps",
        "locked_oos_return_per_turnover_proxy_bps_report_only",
        "train_trade_event_count",
        "validation_trade_event_count",
        "locked_oos_trade_event_count_report_only",
        "selection_reasons",
        "report_only_gate_reasons",
        "fit_splits",
        "final_refit",
    ]
    return {key: row.get(key) for key in keys if key in row}


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
    split_windows: Mapping[str, Any] | None = None,
    standard_live_refit: bool = False,
    final_refit: bool = False,
    data_coverage: Mapping[str, Any] | None = None,
    prior_artifact_path: Path | None = None,
) -> dict[str, Any]:
    active_split_windows = _coerce_split_windows(split_windows)
    with _split_window_context(active_split_windows):
        profile_streams, source_profile_rows = _source_profile_streams(
            integer_payload=integer_payload,
            data_root=data_root,
            feature_root=feature_root,
        )
        base_rows = _base_rows(profile_streams, source_profile_rows)
        grid_row = _grid_baseline_row(profile_streams)
        fit_splits = ("train",) if standard_live_refit else ("train", "validation")
        require_locked_oos_gate = not standard_live_refit
        v35 = _run_optuna(
            profile_streams,
            version="v3_5",
            n_trials=n_trials,
            seed=seed,
            fit_splits=fit_splits,
            require_locked_oos_gate=require_locked_oos_gate,
        )
        v36 = _run_optuna(
            profile_streams,
            version="v3_6",
            n_trials=n_trials,
            seed=seed + 1,
            fit_splits=fit_splits,
            require_locked_oos_gate=require_locked_oos_gate,
        )
        selected_result = _choose_selected_optuna_result([v35, v36])
        selection_evidence = selected_result.row
        selected_final_result: OptunaModelResult | None = None
        selected = selection_evidence
        if final_refit:
            selected_final_result = _run_model(
                profile_streams,
                selected_result.params,
                version=str(selection_evidence["hybrid_version"]),
                profile_id=str(selection_evidence["profile_id"]),
                optuna=selected_result.optuna,
                top_trials=selected_result.top_trials,
                fit_splits=("train", "validation"),
                warmup_splits=("train",),
                final_refit=True,
                require_locked_oos_gate=require_locked_oos_gate,
            )
            selected_final_result.row["optuna"] = selected_result.optuna
            selected = selected_final_result.row
            selected["selection_evidence_metrics"] = _selected_metric_summary(selection_evidence)
        comparison_rows = [*base_rows, grid_row, v35.row, v36.row]
        corr_train_validation = grid_hybrid._profile_corr_matrix(
            profile_streams, split="train_validation"
        )
        corr_validation = grid_hybrid._profile_corr_matrix(profile_streams, split="validation")
        corr_oos = grid_hybrid._profile_corr_matrix(profile_streams, split="locked_oos")

    timestamp = _timestamp()
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    latest_json = output_dir / "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json"
    timestamped_json = (
        output_dir / f"alpha_zoo_integer_leverage_optuna_hybrid_decision_{timestamp}.json"
    )
    latest_md = output_dir / "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.md"
    comparison_csv = output_dir / "integer_leverage_optuna_hybrid_comparison_latest.csv"
    trial_csv = output_dir / "integer_leverage_optuna_hybrid_top_trials_latest.csv"
    methodology_md = output_dir / "integer_leverage_optuna_hybrid_methodology_latest.md"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    trial_rows = [*v35.top_trials, *v36.top_trials]
    previous_selected_summary: dict[str, Any] | None = None
    if prior_artifact_path is not None and Path(prior_artifact_path).exists():
        try:
            previous_payload = ilp._load_json(Path(prior_artifact_path))
            previous_selected_summary = _selected_metric_summary(
                dict(previous_payload.get("selected_optuna_hybrid_profile") or {})
            )
        except Exception as exc:  # pragma: no cover - report-only guard
            previous_selected_summary = {"error": f"failed_to_load_prior:{exc.__class__.__name__}"}
    payload: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at_utc": _utc_now_iso(),
        "source_integer_portfolio_artifact": str(integer_artifact_path),
        "source_grid_baseline_artifact": str(
            grid_hybrid.DEFAULT_OUTPUT_DIR
            / "alpha_zoo_integer_leverage_hybrid_decision_latest.json"
        ),
        "research_primary_round_trip_cost_bps": ilp.PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": ilp.AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "bbo_spread_multiplier": ilp.BBO_SPREAD_MULTIPLIER,
        "return_per_turnover_threshold_bps": ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "ready_for_paper": bool(selected.get("paper_testnet_candidate")),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "paper_testnet_only": True,
        "standard_live_refit": bool(standard_live_refit),
        "final_refit_after_selection": bool(final_refit),
        "split_manifest": _split_window_payload(active_split_windows),
        "data_coverage": dict(data_coverage or {}),
        "previous_selected_profile_comparison": previous_selected_summary,
        "optuna_hybrid_policy": {
            "source_profile_ids": [stream.profile_id for stream in profile_streams],
            "optimizer": "Optuna TPESampler",
            "n_trials_per_version": int(n_trials),
            "v3_5_mapping": "warmup-learned default profile + rolling return/error weights + high-volatility boost + bias/exposure dampening",
            "v3_6_mapping": "v3.5 mechanics plus online adaptive default-profile selection from rolling score evidence",
            "parameter_space_source": "scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py::HYBRID_OPTUNA_CONFIG",
            "all_internal_params_tuned": True,
            "grid_baseline_role": "comparison_only_not_the_selected_optimizer",
            "selection_fit_inputs": ["train"] if standard_live_refit else ["train", "validation"],
            "selection_score_inputs": ["train", "validation"],
            "final_refit_inputs": ["train", "validation"] if final_refit else [],
            "warmup_fit_inputs": ["train"],
            "warmup_ratio_scope": "train_split_only",
            "locked_oos_role": "disabled_for_live_final_refit_no_test_set_reserved"
            if standard_live_refit
            else "gate/report-only after train+validation Optuna params freeze",
        },
        "selection_policy": {
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "uses_validation_for_warmup_or_initial_state_learning": False,
            "uses_validation_for_parameter_fitting_before_selection": not bool(standard_live_refit),
            "final_refit_after_selection": bool(final_refit),
            "no_calendar_date_hack": True,
            "model_id_source": "source integer profile rows derived from frozen corr decision artifact",
        },
        "selected_optuna_hybrid_profile": selected,
        "selection_evidence_profile": selection_evidence,
        "comparison_rows": comparison_rows,
        "hybrid_v3_5_optuna": {
            "row": v35.row,
            "optuna": v35.optuna,
            "top_trials": list(v35.top_trials),
            "allocation_samples": v35.allocations[:50],
        },
        "hybrid_v3_6_optuna": {
            "row": v36.row,
            "optuna": v36.optuna,
            "top_trials": list(v36.top_trials),
            "allocation_samples": v36.allocations[:50],
        },
        "selected_final_refit": None
        if selected_final_result is None
        else {
            "row": selected_final_result.row,
            "allocation_samples": selected_final_result.allocations[:50],
        },
        "profile_train_validation_corr_matrix": corr_train_validation,
        "profile_validation_corr_matrix": corr_validation,
        "profile_locked_oos_corr_matrix_report_only": corr_oos,
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
            "- Optuna/TPESampler is the optimizer for the selected hybrid.\n"
            "- Source: three paper/testnet integer-leverage profiles from the frozen corr/integer artifact.\n"
            "- PnL streams: reconstructed from the same fixed position-state rules and integer asset leverage maps.\n"
            "- Cost: 10bps all-in round-trip backtest friction proxy is embedded before hybriding.\n"
            "- v3.5: warmup-learned default profile, rolling return/error weights, high-vol boost, max-weight cap, bias/exposure dampening.\n"
            "- v3.6: v3.5 mechanics plus online adaptive default-profile refresh from rolling score evidence.\n"
            "- Optuna space now covers every exposed HybridParams field, including warmup ratio, boost bounds/shape, high-vol quantile, and default-weight ratio candidate range.\n"
            "- Standard live refit: tune/learn on train only, score on train+recent validation, then final-refit learned state on train+validation after selection.\n"
            "- Real money: blocked. Paper/testnet only; monitoring must record BBO spread, all-in fee/slippage, liquidation-inclusive MDD, account wipeout, and replay/live notional parity.\n",
            encoding="utf-8",
        )
        generation_log.write_text(
            f"artifact_kind={ARTIFACT_KIND}\n"
            f"source_profile_count={len(source_profile_rows)}\n"
            f"n_trials_per_version={int(n_trials)}\n"
            f"standard_live_refit={bool(standard_live_refit)}\n"
            f"final_refit_after_selection={bool(final_refit)}\n"
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
        standard_live_refit=bool(args.standard_live_refit),
        final_refit=bool(args.final_refit),
        prior_artifact_path=Path(args.prior_artifact).expanduser().resolve()
        if str(args.prior_artifact).strip()
        else None,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integer-portfolio-artifact", default=str(DEFAULT_INTEGER_PORTFOLIO_ARTIFACT)
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(ilp.DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(ilp.DEFAULT_FEATURE_ROOT))
    parser.add_argument("--n-trials", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--standard-live-refit", action="store_true")
    parser.add_argument("--final-refit", action="store_true")
    parser.add_argument(
        "--prior-artifact",
        default=str(
            DEFAULT_OUTPUT_DIR / "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json"
        ),
    )
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
