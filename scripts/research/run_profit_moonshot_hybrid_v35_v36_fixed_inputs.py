#!/usr/bin/env python3
"""Run fixed-input Hybrid v3.5/v3.6 Optuna comparisons for profit moonshot candidates.

This runner ports the ensemble-strategies v3.5/v3.6 mechanics to the
Profit Moonshot candidate return-stream problem:

* v3.5: warmup-fixed default candidate + rolling error/return weights + high-vol boost.
* v3.6: same v3.5 core/Optuna knobs, but the default-candidate parameter is
  refreshed online from rolling scores, matching ensemble_strategies v3_6.py.
* Optuna: searches the same public v3.5/v3.6 knobs (bias alpha, bias combine
  ratio, max single weight, MAPE/rolling window, bias window, short-vol window).

The input universe is intentionally fixed by policy: A0 + P0 + E0 + S1 + S2 +
S3 + S4.  Locked-OOS is never used in the Optuna objective, pruning, parameter
learning, or model selection; it is evaluated only after candidate freeze.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import resource
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FRESH_PATH = REPO_ROOT / "scripts/research/replay_profit_moonshot_fresh_start.py"
TUNER_PATH = REPO_ROOT / "scripts/research/tune_profit_moonshot_fresh_portfolio.py"
ALPHA_REPLAY_PATH = REPO_ROOT / "scripts/research/replay_crypto_fx_alpha_zoo_state.py"

DEFAULT_ALPHA_V2 = REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2"
DEFAULT_COMPARISON_DIR = DEFAULT_ALPHA_V2 / "hybrid_optuna_alpha_zoo_comparison_20260514"
DEFAULT_OUTPUT_DIR = DEFAULT_ALPHA_V2 / "hybrid_v35_v36_fixed_inputs_20260517"
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market_parquet"
DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT"
STARTING_EQUITY = 10_000.0
HOURLY_PERIODS_PER_YEAR = 365 * 24

A0_NAME = "crypto_fx_alpha_zoo_state_calibrated"
P0_NAME = (
    "fresh_portfolio_train_val_monthly_return_budget_"
    "fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600__"
    "fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600__"
    "fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450__"
    "fresh_state_distilled_both_lb168_fast72_z050_ret120_h120_ls590_ss100_tp240"
)
E0_NAME = "fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125"
S1_NAME = "fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600"
S2_NAME = "fresh_state_distilled_both_lb168_fast72_z075_ret60_h168_ls590_ss100_tp600"
S3_NAME = "fresh_state_distilled_both_lb168_fast72_z075_ret120_h168_ls590_ss100_tp600"
S4_NAME = "fresh_state_distilled_both_lb168_fast72_z100_ret60_h168_ls590_ss100_tp450"
FIXED_INPUT_ORDER = ("A0", "P0", "E0", "S1", "S2", "S3", "S4")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _safe_optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    return float(parsed) if math.isfinite(parsed) else None


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp_for_idx(timestamps: np.ndarray, idx: int) -> str:
    return datetime.fromtimestamp(int(timestamps[idx]), tz=UTC).isoformat().replace("+00:00", "Z")


def _split_mask(timestamps: np.ndarray, start: datetime, end: datetime) -> np.ndarray:
    start_ts = int(start.replace(tzinfo=UTC).timestamp())
    end_ts = int(end.replace(tzinfo=UTC).timestamp())
    return (timestamps >= start_ts) & (timestamps <= end_ts)


def _fresh_split_mask(timestamps: np.ndarray, split: Any) -> np.ndarray:
    start_ts = int(datetime.combine(split.start, datetime.min.time(), tzinfo=UTC).timestamp())
    end_ts = int(datetime.combine(split.end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()) - 1
    return (timestamps >= start_ts) & (timestamps <= end_ts)


def _equity_to_returns(equity: Iterable[float]) -> np.ndarray:
    previous = STARTING_EQUITY
    out: list[float] = []
    for value in equity:
        current = max(1e-9, _safe_float(value, previous))
        out.append(current / max(1e-9, previous) - 1.0)
        previous = current
    return np.asarray(out, dtype=float)


def _curve_total_returns_to_incremental(equity: Iterable[float]) -> np.ndarray:
    return _equity_to_returns(equity)


def _returns_to_equity(returns: np.ndarray) -> list[float]:
    equity = STARTING_EQUITY
    out: list[float] = []
    for ret in np.asarray(returns, dtype=float):
        value = float(ret) if math.isfinite(float(ret)) else 0.0
        equity *= max(0.0, 1.0 + value)
        out.append(float(equity))
    return out


def _metrics_from_returns(returns: np.ndarray) -> dict[str, Any]:
    clean = np.asarray([float(x) if math.isfinite(float(x)) else 0.0 for x in returns], dtype=float)
    if clean.size == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "return_mdd": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "smart_sortino": 0.0,
            "calmar": 0.0,
            "trade_count": 0,
            "active_return_hours": 0,
            "liquidation_count": None,
            "minimum_margin_buffer": None,
            "margin_replay_available": False,
        }
    equity = np.cumprod(1.0 + clean)
    total = float(equity[-1] - 1.0)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    dd = np.where(peak > 0.0, (peak - equity) / peak, 0.0)
    mdd = float(np.nanmax(dd)) if dd.size else 0.0
    mean = float(np.nanmean(clean))
    sigma = float(np.nanstd(clean, ddof=1)) if clean.size > 1 else 0.0
    downside = clean[clean < 0.0]
    down_sigma = float(math.sqrt(float(np.mean(np.square(downside))))) if downside.size else 0.0
    sharpe = mean / sigma * math.sqrt(HOURLY_PERIODS_PER_YEAR) if sigma > 1e-12 else 0.0
    sortino = mean / down_sigma * math.sqrt(HOURLY_PERIODS_PER_YEAR) if down_sigma > 1e-12 else 0.0
    years = max(1e-9, clean.size / float(HOURLY_PERIODS_PER_YEAR))
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1.0 else -1.0
    calmar = cagr / mdd if mdd > 1e-12 else (float("inf") if cagr > 0 else 0.0)
    ret_mdd = total / mdd if mdd > 1e-12 else (float("inf") if total > 0 else 0.0)
    return {
        "total_return": total,
        "max_drawdown": mdd,
        "return_mdd": ret_mdd,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "smart_sortino": float(sortino / (1.0 + mdd)),
        "calmar": float(calmar),
        "cagr": cagr,
        "trade_count": int(np.count_nonzero(np.abs(clean) > 1e-12)),
        "active_return_hours": int(np.count_nonzero(np.abs(clean) > 1e-12)),
        "liquidation_count": None,
        "minimum_margin_buffer": None,
        "margin_replay_available": False,
    }


def _monthly_return(metrics: Mapping[str, Any]) -> float:
    cagr = _safe_float(metrics.get("cagr"), -1.0)
    if cagr <= -1.0:
        return -1.0
    return float((1.0 + cagr) ** (1.0 / 12.0) - 1.0)


def _train_val_score(train: Mapping[str, Any], val: Mapping[str, Any]) -> float:
    train_m = _monthly_return(train)
    val_m = _monthly_return(val)
    train_mdd = _safe_float(train.get("max_drawdown"), 1.0)
    val_mdd = _safe_float(val.get("max_drawdown"), 1.0)
    return float(
        80.0 * train_m
        + 140.0 * val_m
        + 1.5 * max(0.0, _safe_float(train.get("sharpe")))
        + 3.0 * max(0.0, _safe_float(val.get("sharpe")))
        + 1.0 * max(0.0, _safe_float(train.get("sortino")))
        + 2.0 * max(0.0, _safe_float(val.get("sortino")))
        + 0.20 * min(60.0, max(0.0, _safe_float(train.get("calmar"))))
        + 0.40 * min(80.0, max(0.0, _safe_float(val.get("calmar"))))
        - 8.0 * train_mdd
        - 16.0 * val_mdd
    )


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


@dataclass(frozen=True)
class HybridParams:
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


def _portfolio_returns_for_params(
    returns: np.ndarray,
    *,
    params: HybridParams,
    learned: LearnedParams,
    version: str,
    start_idx: int = 0,
    initial_portfolio_history: list[float] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    k = returns.shape[1]
    out = np.zeros(returns.shape[0], dtype=float)
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
            # Faithful v3.6 contract from ensemble_strategies:
            # keep the v3.5 mechanics and optimized knobs, but refresh Step A's
            # default model/candidate online from rolling performance.  Do not
            # introduce a new universe or make unrelated knobs OOS-adaptive.
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
        # Iterative cap/redistribute to preserve diversification.
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
        # Bias correction analogue: damp exposure when recent blended model/ensemble bias is negative.
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
        portfolio_history.append(ret)
        if t == returns.shape[0] - 1 or t % 24 == 0:
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
    return out, allocations


def _learn_params(returns: np.ndarray, params: HybridParams, opt_indices: np.ndarray) -> LearnedParams:
    opt = returns[opt_indices]
    n = opt.shape[0]
    warmup_n = max(10, min(n, int(n * params.warmup_ratio)))
    warmup = opt[:warmup_n]
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
    # CV over the external v3.5 weight-ratio candidate set.
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
        cv_returns, _ = _portfolio_returns_for_params(
            warmup,
            params=params,
            learned=learned,
            version="v3_5",
            start_idx=cv_start,
        )
        score = _safe_float(_metrics_from_returns(cv_returns[cv_start:]).get("calmar")) + _safe_float(
            _metrics_from_returns(cv_returns[cv_start:]).get("sharpe")
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


def _run_model(
    returns: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    params: HybridParams,
    *,
    version: str,
) -> dict[str, Any]:
    opt_mask = np.asarray(split_masks["train"] | split_masks["validation"], dtype=bool)
    opt_indices = np.flatnonzero(opt_mask)
    learned = _learn_params(returns, params, opt_indices)
    portfolio, allocations = _portfolio_returns_for_params(
        returns,
        params=params,
        learned=learned,
        version=version,
        start_idx=int(opt_indices[0]) if opt_indices.size else 0,
    )
    splits = {name: _metrics_from_returns(portfolio[mask]) for name, mask in split_masks.items()}
    score = _train_val_score(splits["train"], splits["validation"])
    gate = bool(
        _safe_float(splits["train"].get("total_return")) > 0.0
        and _safe_float(splits["validation"].get("total_return")) > 0.0
        and _safe_float(splits["train"].get("max_drawdown"), 1.0) <= 0.25
        and _safe_float(splits["validation"].get("max_drawdown"), 1.0) <= 0.25
    )
    return {
        "version": version,
        "params": asdict(params),
        "learned_params": asdict(learned),
        "splits": splits,
        "train_val_score": float(score),
        "train_val_gate": gate,
        "allocations": allocations,
        "final_weights": allocations[-1]["weights"] if allocations else [],
        "portfolio_returns": portfolio,
    }


def _params_from_trial(trial: Any) -> HybridParams:
    return HybridParams(
        bias_correction_alpha=float(trial.suggest_float("bias_alpha", 0.85, 1.0)),
        bias_combine_ratio=float(trial.suggest_float("bias_combine_ratio", 0.2, 0.45)),
        max_single_weight=float(trial.suggest_float("max_weight", 0.7, 0.9)),
        mape_window=int(trial.suggest_int("mape_window", 25, 50)),
        bias_window=int(trial.suggest_int("bias_window", 10, 20)),
        short_vol_window=int(trial.suggest_int("short_vol_window", 5, 15)),
    )


def _run_optuna(
    returns: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    *,
    version: str,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    import optuna
    from optuna.samplers import TPESampler

    def objective(trial: Any) -> float:
        result = _run_model(returns, split_masks, _params_from_trial(trial), version=version)
        # Hard reject train/validation failures without looking at OOS.
        if not bool(result["train_val_gate"]):
            return -1e6 + float(result["train_val_score"])
        return float(result["train_val_score"])

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
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
    study.optimize(objective, n_trials=max(1, int(n_trials)), show_progress_bar=False)
    best_params = HybridParams(
        bias_correction_alpha=float(study.best_params["bias_alpha"]),
        bias_combine_ratio=float(study.best_params["bias_combine_ratio"]),
        max_single_weight=float(study.best_params["max_weight"]),
        mape_window=int(study.best_params["mape_window"]),
        bias_window=int(study.best_params["bias_window"]),
        short_vol_window=int(study.best_params["short_vol_window"]),
    )
    final = _run_model(returns, split_masks, best_params, version=version)
    final["optuna"] = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "direction": "maximize_train_validation_score",
        "selection_inputs": ["train", "validation"],
        "uses_locked_oos_for_selection": False,
        "locked_oos_objective_columns_used": [],
        "external_method_source": str(Path("/home/hoky/DeepLearning/ensemble_strategies/models/hybrid") / ("v3_5.py" if version == "v3_5" else "v3_6.py")),
    }
    trial_rows = []
    for t in sorted(study.trials, key=lambda tr: float(tr.value) if tr.value is not None else -1e18, reverse=True)[:20]:
        trial_rows.append({"number": int(t.number), "value": None if t.value is None else float(t.value), "params": dict(t.params)})
    final["top_trials"] = trial_rows
    return final


def _add_returns(full: np.ndarray, mask: np.ndarray, split_returns: np.ndarray) -> None:
    idx = np.flatnonzero(mask)
    if idx.size != split_returns.size:
        raise RuntimeError(f"split length mismatch: mask={idx.size}, returns={split_returns.size}")
    full[idx] = split_returns


def _build_fresh_candidate_streams(
    *,
    fresh: Any,
    tuner: Any,
    arrays: Mapping[str, Any],
    splits: list[Any],
    specs_by_name: Mapping[str, Any],
    portfolio_payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    timestamps = np.asarray(arrays["timestamp"], dtype=np.int64)
    required_spec_names = {E0_NAME, S1_NAME, S2_NAME, S3_NAME, S4_NAME}
    p0 = dict(portfolio_payload.get("selected_by_validation") or portfolio_payload.get("selected_by_train_val_stability") or {})
    p0_sleeves = [str(item) for item in list(p0.get("sleeves") or [])]
    required_spec_names.update(p0_sleeves)
    missing = sorted(name for name in required_spec_names if name not in specs_by_name)
    if missing:
        raise RuntimeError("missing fixed input specs: " + ", ".join(missing))
    split_curves: dict[str, dict[str, list[float]]] = {name: {} for name in required_spec_names}
    split_payloads: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in required_spec_names}
    for name in sorted(required_spec_names):
        for split in splits:
            result = fresh._run_split(spec=specs_by_name[name], arrays=arrays, split=split, include_equity=True)
            split_curves[name][str(split.name)] = [float(x) for x in list(result.get("equity_history") or [])]
            split_payloads[name][str(split.name)] = dict(result)
    streams: dict[str, dict[str, Any]] = {}
    for label, name, leverage, source in (
        ("E0", E0_NAME, 4.0, "liquidation_aware_state_distilled_external_risk_filter_20260512"),
        ("S1", S1_NAME, 1.0, "state_distilled_leadership_unwind_20260511"),
        ("S2", S2_NAME, 1.0, "state_distilled_leadership_unwind_20260511"),
        ("S3", S3_NAME, 1.0, "state_distilled_leadership_unwind_20260511"),
        ("S4", S4_NAME, 1.0, "state_distilled_leadership_unwind_20260511"),
    ):
        full = np.zeros(timestamps.shape[0], dtype=float)
        source_metrics: dict[str, Any] = {}
        for split in splits:
            raw_name = str(split.name)
            mask = _fresh_split_mask(timestamps, split)
            curve = tuner._combine_equity([split_curves[name][raw_name]], mode="train_val_monthly_return_budget", leverage=leverage)
            _add_returns(full, mask, _curve_total_returns_to_incremental(curve))
            target = "validation" if raw_name == "val" else raw_name
            source_metrics[target] = dict(split_payloads[name][raw_name].get("metrics") or {})
        streams[label] = {
            "label": label,
            "candidate_name": name,
            "candidate_source": source,
            "leverage": leverage,
            "returns": full,
            "source_split_metrics": source_metrics,
            "uses_locked_oos_for_selection": False,
            "structural_hybrid_input": False,
        }
    # P0 aggregate from its source sleeves and leverage/mode.
    full = np.zeros(timestamps.shape[0], dtype=float)
    p0_mode = str(p0.get("mode") or "train_val_monthly_return_budget")
    p0_leverage = _safe_float(p0.get("leverage"), 3.0)
    p0_weights = [_safe_float(item) for item in list(p0.get("weights") or [])] or None
    p0_metrics: dict[str, Any] = {}
    for split in splits:
        raw_name = str(split.name)
        mask = _fresh_split_mask(timestamps, split)
        curve = tuner._combine_equity(
            [split_curves[name][raw_name] for name in p0_sleeves],
            mode=p0_mode,
            weights=p0_weights,
            leverage=p0_leverage,
        )
        _add_returns(full, mask, _curve_total_returns_to_incremental(curve))
        target = "validation" if raw_name == "val" else raw_name
        p0_metrics[target] = _metrics_from_returns(full[mask])
    streams["P0"] = {
        "label": "P0",
        "candidate_name": str(p0.get("name") or P0_NAME),
        "candidate_source": "state_distilled_market_state_next_tuning:selected_by_validation",
        "leverage": p0_leverage,
        "mode": p0_mode,
        "sleeves": p0_sleeves,
        "returns": full,
        "source_split_metrics": p0_metrics,
        "uses_locked_oos_for_selection": bool(dict(p0.get("locked_oos_policy") or {}).get("uses_locked_oos_for_selection", False)),
        "structural_hybrid_input": False,
    }
    return streams


def _build_alpha_stream(
    *,
    alpha: Any,
    timestamps: np.ndarray,
    alpha_replay_payload: Mapping[str, Any],
    calibration_path: Path,
    external_state_csv: Path,
) -> dict[str, Any]:
    source_path = Path(str(dict(alpha_replay_payload.get("source_coverage") or {}).get("source_path") or ""))
    if not source_path.is_absolute():
        source_path = REPO_ROOT / source_path
    from lumina_quant.research.crypto_fx_alpha_zoo_real_data import load_real_data_bundle

    bundle = load_real_data_bundle(
        current_tail_cache=source_path,
        external_state_csv=external_state_csv,
        strict_real_data=True,
    )
    data = alpha._ensure_replay_frame(
        _apply_common_split_contract(bundle.frame, _common_split_contract_from_payload(alpha_replay_payload))
    )
    edges = alpha._load_calibrated_edges(calibration_path)
    grid = dict(alpha_replay_payload.get("candidate_selection_grid") or {})
    params = dict(grid.get("selected_candidate_params") or {})
    signals = alpha._run_strategy_signals(
        data,
        require_calibrated_edge=True,
        calibrated_edges=edges,
        strategy_params=params,
    )
    trades = alpha._build_trades(data, signals)
    bucket: dict[int, list[float]] = {}
    ts_to_idx = {int(ts): idx for idx, ts in enumerate(timestamps.tolist())}
    for trade in trades:
        exit_ts = int(datetime.fromisoformat(str(trade["exit_time"])).replace(tzinfo=UTC).timestamp())
        idx = ts_to_idx.get(exit_ts)
        if idx is None:
            continue
        ret = alpha._portfolio_trade_return(trade, leverage=6.0, allocation_fraction=0.10)
        bucket.setdefault(idx, []).append(float(ret))
    full = np.zeros(timestamps.shape[0], dtype=float)
    for idx, values in bucket.items():
        value = math.prod(1.0 + float(v) for v in values) - 1.0
        full[idx] = float(value)
    return {
        "label": "A0",
        "candidate_name": A0_NAME,
        "candidate_source": "CryptoFxAlphaZooStateStrategy:alpha_zoo_conservative_exit:strict_6x",
        "leverage": 6.0,
        "returns": full,
        "trade_count_reconstructed": len(trades),
        "signal_count_reconstructed": len(signals),
        "selected_params": params,
        "uses_locked_oos_for_selection": False,
        "structural_hybrid_input": False,
    }


def _source_metrics_from_stream(stream: Mapping[str, Any], split_masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    arr = np.asarray(stream["returns"], dtype=float)
    return {name: _metrics_from_returns(arr[mask]) for name, mask in split_masks.items()}


def _period_payload(timestamps: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"start_timestamp": None, "end_timestamp": None, "record_count": 0}
    return {
        "start_timestamp": _timestamp_for_idx(timestamps, int(idx[0])),
        "end_timestamp": _timestamp_for_idx(timestamps, int(idx[-1])),
        "record_count": int(idx.size),
    }


def _parse_utc_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    if parsed.tzinfo is None:
        return parsed.tz_localize(UTC)
    return parsed.tz_convert(UTC)


def _common_split_contract_from_payload(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = dict(payload.get("common_split_manifest") or {})
    contract = dict(manifest.get("split_contract") or payload.get("common_split_contract") or {})
    return {str(key): dict(value) for key, value in contract.items() if isinstance(value, Mapping)}


def _apply_common_split_contract(frame: Any, contract: Mapping[str, Mapping[str, Any]]) -> Any:
    """Apply a replay payload's explicit split contract to the Alpha stream.

    The fixed-input hybrid runner reconstructs A0 from the Alpha replay source
    path. Historical Alpha replay artifacts did not carry split labels in that
    source file, so this helper is intentionally no-op unless a common-split
    replay payload provides a manifest. When present, it prevents the Alpha
    stream from falling back to the older fractional split assignment.
    """
    if not contract:
        return frame
    required = {"train", "validation", "locked_oos"}
    if not required.issubset(set(contract)):
        return frame
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    overall_start = min(_parse_utc_timestamp(contract[name]["start"]) for name in required)
    overall_end = max(_parse_utc_timestamp(contract[name]["end"]) for name in required)
    keep = ts.ge(overall_start) & ts.le(overall_end)
    out = out.loc[keep].copy()
    ts = ts.loc[keep]
    labels = pd.Series("outside_common_split", index=out.index, dtype="object")
    for name in ("train", "validation", "locked_oos"):
        start = _parse_utc_timestamp(contract[name]["start"])
        end = _parse_utc_timestamp(contract[name]["end"])
        labels.loc[ts.ge(start) & ts.le(end)] = name
    out["timestamp"] = ts.dt.tz_convert(UTC).dt.tz_localize(None)
    out["split"] = labels.to_numpy(dtype=object)
    return out[out["split"].isin(("train", "validation", "locked_oos"))].copy()


def _public_result(result: Mapping[str, Any], candidate_labels: list[str]) -> dict[str, Any]:
    out = {k: v for k, v in result.items() if k not in {"portfolio_returns"}}
    final_weights = list(out.get("final_weights") or [])
    out["final_weight_by_candidate"] = {
        label: float(final_weights[idx]) for idx, label in enumerate(candidate_labels) if idx < len(final_weights)
    }
    out["allocations_tail"] = list(out.get("allocations") or [])[-10:]
    out.pop("allocations", None)
    return out


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):+.2%}"


def _fmt_num(value: Any) -> str:
    parsed = _safe_optional_float(value)
    if parsed is None:
        return ""
    if math.isinf(parsed):
        return "inf"
    return f"{parsed:.3f}"


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Hybrid v3.5/v3.6 Optuna fixed-input comparison",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        "- fixed inputs: `A0 + P0 + E0 + S1 + S2 + S3 + S4`",
        "- external method source: `/home/hoky/DeepLearning/ensemble_strategies/models/hybrid/v3_5.py`, `v3_6.py`",
        "- selection/objective inputs: `train`, `validation` only",
        "- locked-OOS: report/gate-only after candidate freeze; not used by Optuna objective/pruning/selection",
        "",
        "## Split periods",
        "",
    ]
    for name, period in dict(payload.get("split_periods") or {}).items():
        lines.append(f"- {name}: `{period.get('start_timestamp')}` ~ `{period.get('end_timestamp')}`")
    lines.extend(["", "## Candidate inputs", "", "| label | candidate | source | train | validation | locked-OOS |", "|---|---|---|---:|---:|---:|"])
    for item in list(payload.get("candidate_inputs") or []):
        metrics = dict(item.get("hybrid_split_metrics") or {})
        def cell(split: str, metrics: Mapping[str, Any] = metrics) -> str:
            m = dict(metrics.get(split) or {})
            liq = "not_replayed" if m.get("liquidation_count") is None else str(m.get("liquidation_count"))
            buf = "not_replayed" if m.get("minimum_margin_buffer") is None else _fmt_num(m.get("minimum_margin_buffer"))
            return f"{_fmt_pct(m.get('total_return'))} / MDD {_fmt_pct(m.get('max_drawdown'))} / Sh {_fmt_num(m.get('sharpe'))} / Liq {liq} / Buf {buf}"
        lines.append(
            f"| {item.get('label')} | `{item.get('candidate_name')}` | `{item.get('candidate_source')}` | "
            f"{cell('train')} | {cell('validation')} | {cell('locked_oos')} |"
        )
    lines.extend(["", "## Hybrid Optuna results", "", "| model | TV score | train | validation | locked-OOS | OOS MDD | OOS Sharpe | OOS Sortino | OOS Calmar | OOS liquidation | OOS min buffer | deployable_success | rejection reasons |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"])
    for key in ("hybrid_v3_5_optuna", "hybrid_v3_6_optuna"):
        item = dict(payload.get(key) or {})
        splits = dict(item.get("splits") or {})
        train = dict(splits.get("train") or {})
        val = dict(splits.get("validation") or {})
        oos = dict(splits.get("locked_oos") or {})
        liq = "not_replayed" if oos.get("liquidation_count") is None else str(oos.get("liquidation_count"))
        buf = "not_replayed" if oos.get("minimum_margin_buffer") is None else _fmt_num(oos.get("minimum_margin_buffer"))
        reasons = ", ".join(str(x) for x in list(item.get("rejection_reasons") or []))
        lines.append(
            f"| {key} | {_fmt_num(item.get('train_val_score'))} | {_fmt_pct(train.get('total_return'))} | "
            f"{_fmt_pct(val.get('total_return'))} | {_fmt_pct(oos.get('total_return'))} | {_fmt_pct(oos.get('max_drawdown'))} | "
            f"{_fmt_num(oos.get('sharpe'))} | {_fmt_num(oos.get('sortino'))} | {_fmt_num(oos.get('calmar'))} | "
            f"{liq} | {buf} | {item.get('deployable_success')} | {reasons} |"
        )
    lines.extend(["", "## Final weights", ""])
    for key in ("hybrid_v3_5_optuna", "hybrid_v3_6_optuna"):
        weights = dict(dict(payload.get(key) or {}).get("final_weight_by_candidate") or {})
        preview = ", ".join(f"{name}={_fmt_pct(weight)}" for name, weight in weights.items())
        lines.append(f"- {key}: {preview}")
    return "\n".join(lines) + "\n"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    fresh = _load_module(FRESH_PATH, "fresh_replay_for_v35_v36_fixed_inputs")
    tuner = _load_module(TUNER_PATH, "fresh_tuner_for_v35_v36_fixed_inputs")
    alpha = _load_module(ALPHA_REPLAY_PATH, "alpha_replay_for_v35_v36_fixed_inputs")

    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    oos_end = datetime.fromisoformat(str(args.oos_end_date)).date()
    splits = fresh._split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    panel, data_metadata = fresh._joined_panel(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        symbols=symbols,
        start=start,
        end=end,
    )
    arrays = fresh._build_arrays(panel, symbols)
    timestamps = np.asarray(arrays["timestamp"], dtype=np.int64)
    specs_by_name = {spec.name: spec for spec in fresh._candidate_specs(arrays, symbols)}

    # Use the fresh split contract for this fixed-input hybrid experiment.
    split_masks = {
        "train": _fresh_split_mask(timestamps, splits[0]),
        "validation": _fresh_split_mask(timestamps, splits[1]),
        "locked_oos": _fresh_split_mask(timestamps, splits[2]),
    }
    split_periods = {name: _period_payload(timestamps, mask) for name, mask in split_masks.items()}

    portfolio_payload = _load_json(Path(args.portfolio_json))
    alpha_replay_payload = _load_json(Path(args.alpha_replay_json))
    streams = _build_fresh_candidate_streams(
        fresh=fresh,
        tuner=tuner,
        arrays=arrays,
        splits=splits,
        specs_by_name=specs_by_name,
        portfolio_payload=portfolio_payload,
    )
    streams["A0"] = _build_alpha_stream(
        alpha=alpha,
        timestamps=timestamps,
        alpha_replay_payload=alpha_replay_payload,
        calibration_path=Path(args.alpha_calibration_json),
        external_state_csv=Path(args.external_state_csv),
    )
    ordered = [streams[label] for label in FIXED_INPUT_ORDER]
    returns = np.column_stack([np.asarray(item["returns"], dtype=float) for item in ordered])
    labels = [str(item["label"]) for item in ordered]

    v35 = _run_optuna(returns, split_masks, version="v3_5", n_trials=int(args.n_trials), seed=int(args.seed))
    v36 = _run_optuna(returns, split_masks, version="v3_6", n_trials=int(args.n_trials), seed=int(args.seed))

    def annotate_live_policy(item: dict[str, Any]) -> dict[str, Any]:
        oos = dict(dict(item.get("splits") or {}).get("locked_oos") or {})
        reasons: list[str] = []
        if _safe_float(oos.get("max_drawdown"), 1.0) > 0.25:
            reasons.append("locked_oos_mdd_above_25pct")
        if _safe_float(oos.get("total_return")) <= 0.06428110030664325:
            reasons.append("locked_oos_return_not_above_invalid_current_base_reference")
        # This mixed A0/P0/source-sleeve allocator needs a dedicated integrated margin replay before live promotion.
        reasons.append("dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid")
        out = _public_result(item, labels)
        out["live_promotion_possible"] = False
        out["deployable_success"] = False
        out["margin_replay_available"] = False
        out["rejection_reasons"] = reasons
        out["selection_provenance"] = {
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
            "candidate_freeze_before_locked_oos_gate": True,
            "current_base_calendar_tuple_role": "hypothesis_reference_only",
        }
        return out

    candidate_inputs = []
    for item in ordered:
        candidate_inputs.append(
            {
                k: v
                for k, v in item.items()
                if k not in {"returns"}
            }
            | {
                "hybrid_split_metrics": _source_metrics_from_stream(item, split_masks),
                "hybrid_input_policy": {
                    "calendar_entry_rule": False,
                    "literal_hybrid_source": False,
                    "uses_locked_oos_for_selection": bool(item.get("uses_locked_oos_for_selection", False)),
                },
            }
        )

    payload = {
        "artifact_kind": "profit_moonshot_hybrid_v35_v36_fixed_inputs_optuna",
        "generated_at_utc": _utc_now_iso(),
        "method_contract": {
            "requested_fixed_inputs": list(FIXED_INPUT_ORDER),
            "input_universe": "A0 + P0 + E0 + S1 + S2 + S3 + S4",
            "external_reference_root": "/home/hoky/DeepLearning/ensemble_strategies",
            "v3_5_mapping": "warmup-fixed default candidate + rolling return/error weights + high-volatility boost + Optuna search space from ensemble_strategies v3_5.py",
            "v3_6_mapping": "v3.5 mapping plus online adaptive default-candidate selection from rolling scores; Optuna knobs/weight-ratio/high-vol boost/max-weight remain train-validation learned and frozen before locked-OOS",
            "structural_hybrid_inputs_allowed": False,
            "literal_hybrid_inputs_included": [],
        },
        "selection_policy": {
            "selection_inputs": ["train", "validation"],
            "locked_oos": "report_only_gate_only_after_candidate_freeze",
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_objective": False,
            "objective": "maximize train/validation stability score; locked-OOS is not read in objective",
        },
        "split_periods": split_periods,
        "data_metadata": data_metadata,
        "candidate_inputs": candidate_inputs,
        "hybrid_v3_5_optuna": annotate_live_policy(v35),
        "hybrid_v3_6_optuna": annotate_live_policy(v36),
        "memory_summary": {"peak_rss_mib": _rss_mib(), "limit_mib": 8192.0, "pass_under_8gb": _rss_mib() < 8192.0},
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default=str(DEFAULT_MARKET_ROOT))
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--oos-end-date", default="2026-05-06")
    parser.add_argument("--portfolio-json", default=str(DEFAULT_ALPHA_V2 / "state_distilled_market_state_next_20260512/portfolio_tuning_leadership_unwind_top18/fresh_portfolio_tuning_latest.json"))
    parser.add_argument("--alpha-replay-json", default=str(DEFAULT_ALPHA_V2 / "crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_state_replay_latest.json"))
    parser.add_argument("--alpha-calibration-json", default=str(DEFAULT_ALPHA_V2 / "crypto_fx_alpha_zoo_real_data_20260514/edge_calibration_latest.json"))
    parser.add_argument("--external-state-csv", default=str(DEFAULT_ALPHA_V2 / "external_market_state_20260512/external_market_state_lagged.csv"))
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    latest = output_dir / "hybrid_v35_v36_fixed_inputs_latest.json"
    latest.write_text(text, encoding="utf-8")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timestamped = output_dir / f"hybrid_v35_v36_fixed_inputs_{timestamp}.json"
    timestamped.write_text(text, encoding="utf-8")
    md = _markdown(payload)
    latest_md = output_dir / "hybrid_v35_v36_fixed_inputs_latest.md"
    latest_md.write_text(md, encoding="utf-8")
    timestamped.with_suffix(".md").write_text(md, encoding="utf-8")
    print(json.dumps({"json": str(latest), "markdown": str(latest_md), "peak_rss_mib": payload["memory_summary"]["peak_rss_mib"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
