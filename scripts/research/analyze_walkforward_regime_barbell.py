#!/usr/bin/env python3
"""Fast-path regime-aware barbell analysis for monthly walk-forward reports.

The script consumes an existing full monthly WF JSON and builds shadow-only
selector candidates from information that is available before each OOS fold:
train metrics, validation metrics, and lagged market-regime features ending at
the refit date.  Current-fold OOS is used only after the selector decision has
been frozen, for evaluation/reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.research_universe import BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS  # noqa: E402
from lumina_quant.strategy_factory import build_binance_futures_candidates  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402

DEFAULT_SOURCE_JSON = (
    REPO_ROOT
    / "var/reports/latest_full_walkforward_20260628/alpha_zoo_110_latest_full_wf_20260628.json"
)
DEFAULT_REPORT_DIR = REPO_ROOT / "var/reports/latest_full_walkforward_20260628"
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_DIR / "regime_barbell_selector_analysis_20260629.json"
DEFAULT_OUTPUT_MD = DEFAULT_REPORT_DIR / "regime_barbell_selector_analysis_20260629.md"
DEFAULT_DATA_ROOT = REPO_ROOT / "data/market_parquet/exchange=binance"
DEFAULT_CRISIS_RAW_LABEL = (
    "codex_lagged_leaf_router_grid:"
    "h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled"
)
DEFAULT_CRISIS_FALLBACK_LABEL = (
    "codex_lagged_leaf_router_grid:"
    "h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2"
)
DEFAULT_CLEAN_TOP_LABEL = (
    "dynamic_conviction_switch:t0.85_risk_capped_fallback_val_ret02_calmar80_gate_val_mdd30_scaled"
)
DEFAULT_POSITIVE_BASELINE_LABEL = "dynamic_conviction_switch:t0.85_risk_capped_fallback_val_mdd30_scaled"
BULL_POOL_FAMILIES = {
    "asset_timeframe_leverage",
    "cross_candidate_hybrid",
    "dynamic_conviction_switch",
    "individual_robust",
    "relaxed_efficiency",
    "strict_calm_leaf_selector",
    "strict_efficiency",
    "tradfi_momentum_regime_v1",
    "tradfi_vol_managed_v1",
    "bull_bear_regime_rotation_latest_data",
}
BULL_POOL_LABEL_HINTS = (
    "bull_bear_regime_rotation",
    "momentum",
    "trend",
    "topcap",
    "individual_robust",
    "asset_timeframe_leverage",
)


@dataclass(frozen=True)
class RegimeContext:
    fold_id: str
    validation_btc_return: float
    last_validation_month_btc_return: float
    validation_crypto_breadth: float
    last_validation_month_crypto_breadth: float
    validation_btc_ma_gap: float
    oos_btc_return: float | None
    decision: str
    reason: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "validation_btc_return": self.validation_btc_return,
            "last_validation_month_btc_return": self.last_validation_month_btc_return,
            "validation_crypto_breadth": self.validation_crypto_breadth,
            "last_validation_month_crypto_breadth": self.last_validation_month_crypto_breadth,
            "validation_btc_ma_gap": self.validation_btc_ma_gap,
            "oos_btc_return_analysis_only": self.oos_btc_return,
            "decision": self.decision,
            "reason": self.reason,
            "feature_cutoff": "validation_end/refit_minus_one_bar",
        }


@dataclass(frozen=True)
class SelectorSpec:
    label: str
    crisis_label: str
    bull_clean_only: bool
    bull_weight: float
    mixed_bull_weight: float
    bear_bull_weight: float
    min_bull_validation_return: float
    max_bull_validation_mdd: float
    recovery_bull_weight: float | None = None
    recovery_crisis_weight: float | None = None
    post_oos_research_variant: bool = False
    crisis_candidate_labels: tuple[str, ...] = ()
    recency_halflife_folds: float | None = None
    recency_min_history_folds: int = 0


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _safe_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _metric_block(total_return: float, mdd: float = 0.0, *, start: str | None = None, end: str | None = None) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "total_return": float(total_return),
        "mdd": float(max(0.0, mdd)),
        "calmar": float(total_return / max(mdd, 1e-12)) if mdd > 0 else 0.0,
    }


def _row_return(row: Mapping[str, Any], split: str) -> float:
    return _safe_float(dict(row.get(split) or {}).get("total_return"))


def _row_mdd(row: Mapping[str, Any], split: str) -> float:
    return _safe_float(dict(row.get(split) or {}).get("mdd"))


def _compound(returns: Sequence[float]) -> float:
    out = 1.0
    for value in returns:
        out *= 1.0 + float(value)
    return float(out - 1.0)


def _equity_mdd_from_returns(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(list(returns), dtype=float))
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / np.maximum(peak, 1e-12)))


def _max_drawdown(equity: np.ndarray) -> float:
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    return float(np.max(1.0 - arr / np.maximum(peak, 1e-12)))


def _aggregate_rows(label: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    oos_returns = [_row_return(row, "locked_oos") for row in rows]
    train_returns = [_row_return(row, "train") for row in rows]
    val_returns = [_row_return(row, "validation") for row in rows]
    oos_mdds = [_row_mdd(row, "locked_oos") for row in rows]
    val_mdds = [_row_mdd(row, "validation") for row in rows]
    arr = np.asarray(oos_returns, dtype=float)
    negative = arr[arr < 0.0]
    positive = arr[arr > 0.0]
    monthly_std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    downside_std = float(np.std(negative, ddof=1)) if negative.size > 1 else 0.0
    compounded = _compound(oos_returns)
    var_05 = float(np.quantile(arr, 0.05)) if arr.size else 0.0
    q25 = float(np.quantile(arr, 0.25)) if arr.size else 0.0
    q95 = float(np.quantile(arr, 0.95)) if arr.size else 0.0
    cvar_25 = float(np.mean(arr[arr <= q25])) if arr.size else 0.0
    loss_streak = 0
    max_loss_streak = 0
    for value in oos_returns:
        if value < 0.0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0
    fold_count = len(oos_returns)
    return {
        "candidate_label": label,
        "family": rows[0].get("family") if rows else None,
        "fold_count": fold_count,
        "compounded_oos_return": compounded,
        "positive_oos_folds": int(sum(value > 0.0 for value in oos_returns)),
        "oos_hit_rate": float(sum(value > 0.0 for value in oos_returns) / fold_count) if fold_count else 0.0,
        "min_oos_return": float(min(oos_returns)) if oos_returns else 0.0,
        "latest_oos_return": float(oos_returns[-1]) if oos_returns else 0.0,
        "mean_oos_return": float(np.mean(arr)) if arr.size else 0.0,
        "median_oos_return": float(np.median(arr)) if arr.size else 0.0,
        "max_oos_mdd": float(max(oos_mdds)) if oos_mdds else 0.0,
        "monthly_equity_mdd": _equity_mdd_from_returns(oos_returns),
        "monthly_volatility": monthly_std,
        "monthly_downside_volatility": downside_std,
        "monthly_sharpe_approx": float(np.mean(arr) / monthly_std * math.sqrt(12.0)) if monthly_std > 0 else 0.0,
        "monthly_sortino_approx": float(np.mean(arr) / downside_std * math.sqrt(12.0)) if downside_std > 0 else 0.0,
        "monthly_var_05": var_05,
        "monthly_quantile_95": q95,
        "monthly_cvar_25": cvar_25,
        "avg_gain": float(np.mean(positive)) if positive.size else 0.0,
        "avg_loss": float(np.mean(negative)) if negative.size else 0.0,
        "profit_factor": float(np.sum(positive) / abs(np.sum(negative))) if positive.size and negative.size and abs(float(np.sum(negative))) > 0 else 0.0,
        "max_loss_streak": int(max_loss_streak),
        "mean_train_return": float(np.mean(train_returns)) if train_returns else 0.0,
        "mean_validation_return": float(np.mean(val_returns)) if val_returns else 0.0,
        "positive_validation_folds": int(sum(value > 0.0 for value in val_returns)),
        "max_validation_mdd": float(max(val_mdds)) if val_mdds else 0.0,
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "readiness_label": "shadow_only_research_candidate",
    }


def _group_rows_by_fold(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("fold_id"))].append(row)
    return dict(out)


def _group_rows_by_label(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get("candidate_label"))].append(row)
    return dict(out)


def _find_row(rows_by_fold: Mapping[str, Sequence[Mapping[str, Any]]], fold_id: str, label: str) -> Mapping[str, Any] | None:
    for row in rows_by_fold.get(fold_id, []):
        if str(row.get("candidate_label")) == label:
            return row
    return None


def _row_uses_oos_for_selection(row: Mapping[str, Any]) -> bool:
    return any(
        _safe_bool(row.get(key))
        for key in (
            "uses_locked_oos_for_selection",
            "current_fold_oos_used_for_weighting",
            "same_month_self_feeding",
        )
    )


def _is_bull_pool_candidate(row: Mapping[str, Any], *, clean_only: bool) -> bool:
    if _row_uses_oos_for_selection(row):
        return False
    family = str(row.get("family") or "").lower()
    label = str(row.get("candidate_label") or "").lower()
    if "lagged_shadow_leaf_router" in family or "lagged_leaf_router" in label:
        return False
    if "diagnostic" in str(row.get("readiness_label") or "").lower():
        return False
    if clean_only and not _safe_bool(row.get("clean_promotion_eligible", True)):
        return False
    hinted = family in BULL_POOL_FAMILIES or any(hint in label for hint in BULL_POOL_LABEL_HINTS)
    if not hinted:
        return False
    val_ret = _row_return(row, "validation")
    train_ret = _row_return(row, "train")
    val_mdd = _row_mdd(row, "validation")
    train_mdd = _row_mdd(row, "train")
    return train_ret > 0.0 and val_ret >= 0.0 and val_mdd <= 0.35 and train_mdd <= 0.80


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def train_validation_bull_score(row: Mapping[str, Any]) -> float:
    """Train/validation-only sleeve score; locked OOS is deliberately absent."""
    val_ret = _clip(_row_return(row, "validation"), -0.25, 0.80)
    train_ret = _clip(_row_return(row, "train"), -0.25, 1.20)
    val_mdd = _clip(_row_mdd(row, "validation"), 0.0, 0.80)
    train_mdd = _clip(_row_mdd(row, "train"), 0.0, 1.20)
    val_calmar = val_ret / max(val_mdd, 0.03)
    return float(1.45 * val_ret + 0.35 * train_ret + 0.06 * val_calmar - 0.75 * val_mdd - 0.20 * train_mdd)


def select_bull_sleeve(
    fold_rows: Sequence[Mapping[str, Any]],
    *,
    clean_only: bool,
    min_validation_return: float,
    max_validation_mdd: float,
) -> Mapping[str, Any] | None:
    eligible: list[Mapping[str, Any]] = []
    for row in fold_rows:
        if not _is_bull_pool_candidate(row, clean_only=clean_only):
            continue
        if _row_return(row, "validation") < min_validation_return:
            continue
        if _row_mdd(row, "validation") > max_validation_mdd:
            continue
        eligible.append(row)
    if not eligible:
        return None
    return max(eligible, key=train_validation_bull_score)


def _lagged_recency_score(values: Sequence[float], *, halflife: float) -> float:
    if not values:
        return 0.0
    decay = math.log(2.0) / max(1e-9, float(halflife))
    weights = [math.exp(-decay * (len(values) - 1 - idx)) for idx in range(len(values))]
    denom = sum(weights)
    return float(sum(weight * value for weight, value in zip(weights, values, strict=False)) / denom)


def select_crisis_sleeve(
    *,
    fold_id: str,
    fold_order: Sequence[str],
    rows_by_fold: Mapping[str, Sequence[Mapping[str, Any]]],
    spec: SelectorSpec,
) -> Mapping[str, Any]:
    labels = tuple(spec.crisis_candidate_labels or (spec.crisis_label,))
    available = {
        label: row
        for label in labels
        if (row := _find_row(rows_by_fold, fold_id, label)) is not None
    }
    if not available:
        raise KeyError(f"missing crisis sleeve candidates {labels!r} for fold {fold_id}")
    if len(available) == 1 or spec.recency_halflife_folds is None:
        preferred = available.get(spec.crisis_label)
        return preferred if preferred is not None else next(iter(available.values()))
    current_idx = list(fold_order).index(fold_id)
    scores: dict[str, float] = {}
    for label, row in available.items():
        history: list[float] = []
        for prior_fold in fold_order[:current_idx]:
            prior = _find_row(rows_by_fold, prior_fold, label)
            if prior is not None:
                history.append(_row_return(prior, "locked_oos"))
        if len(history) < int(spec.recency_min_history_folds):
            scores[label] = 1.0 if label == spec.crisis_label else 0.0
            continue
        recency = _lagged_recency_score(history, halflife=float(spec.recency_halflife_folds))
        validation_bonus = 0.10 * _clip(_row_return(row, "validation"), -0.20, 0.50)
        mdd_penalty = 0.05 * _clip(_row_mdd(row, "validation"), 0.0, 0.50)
        scores[label] = recency + validation_bonus - mdd_penalty
    return available[max(scores, key=lambda label: (scores[label], label == spec.crisis_label))]


def selector_weights(
    decision: str,
    spec: SelectorSpec,
    *,
    has_bull: bool,
    reason: str = "",
) -> dict[str, float]:
    if not has_bull:
        return {"crisis": 1.0, "bull": 0.0, "cash": 0.0}
    token = str(decision).upper()
    if token == "BULL" and str(reason).startswith("washout_recovery"):
        bull = spec.recovery_bull_weight
        crisis = spec.recovery_crisis_weight
        if bull is None:
            bull = spec.bull_weight
        if crisis is None:
            crisis = 1.0 - float(bull)
        bull = _clip(float(bull), 0.0, 1.0)
        crisis = _clip(float(crisis), 0.0, 1.0 - bull)
        return {"bull": bull, "crisis": crisis, "cash": max(0.0, 1.0 - bull - crisis)}
    if token == "BULL":
        bull = spec.bull_weight
    elif token == "BEAR":
        bull = spec.bear_bull_weight
    else:
        bull = spec.mixed_bull_weight
    bull = _clip(bull, 0.0, 1.0)
    return {"bull": bull, "crisis": 1.0 - bull, "cash": 0.0}


def _combine_split(
    crisis: Mapping[str, Any],
    bull: Mapping[str, Any] | None,
    weights: Mapping[str, float],
    split: str,
) -> dict[str, Any]:
    c = dict(crisis.get(split) or {})
    b = dict(bull.get(split) or {}) if bull is not None else {}
    c_w = _safe_float(weights.get("crisis"))
    b_w = _safe_float(weights.get("bull")) if bull is not None else 0.0
    total_return = c_w * _safe_float(c.get("total_return")) + b_w * _safe_float(b.get("total_return"))
    mdd = min(1.0, c_w * _safe_float(c.get("mdd")) + b_w * _safe_float(b.get("mdd")))
    return _metric_block(total_return, mdd, start=c.get("start") or b.get("start"), end=c.get("end") or b.get("end"))


def build_selector_rows(
    *,
    fold_order: Sequence[str],
    rows_by_fold: Mapping[str, Sequence[Mapping[str, Any]]],
    regimes: Mapping[str, RegimeContext],
    spec: SelectorSpec,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fold_id in fold_order:
        crisis = select_crisis_sleeve(
            fold_id=fold_id,
            fold_order=fold_order,
            rows_by_fold=rows_by_fold,
            spec=spec,
        )
        bull = select_bull_sleeve(
            rows_by_fold.get(fold_id, []),
            clean_only=spec.bull_clean_only,
            min_validation_return=spec.min_bull_validation_return,
            max_validation_mdd=spec.max_bull_validation_mdd,
        )
        ctx = regimes[fold_id]
        weights = selector_weights(ctx.decision, spec, has_bull=bull is not None, reason=ctx.reason)
        row = {
            "candidate_label": spec.label,
            "fold_id": fold_id,
            "family": "regime_barbell_selector",
            "candidate_tier": "shadow_only_fast_path_selector",
            "readiness_label": "shadow_only_research_candidate",
            "allowed_usage_label": "shadow_only_not_real_money",
            "clean_promotion_eligible": False,
            "ready_for_paper": False,
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "uses_locked_oos_for_selection": False,
            "current_fold_oos_used_for_weighting": False,
            "same_month_self_feeding": False,
            "post_oos_research_variant": bool(spec.post_oos_research_variant),
            "selection_inputs": ["train", "validation", "lagged_market_regime_features"],
            "selection_policy": "validation_regime_gate_plus_train_validation_bull_sleeve_score",
            "selection_reasons": [ctx.reason, "locked_oos_evaluated_after_decision_only"],
            "selector_design_warning": (
                "post_oos_research_variant: guard design was introduced after reviewing existing OOS diagnostics"
                if spec.post_oos_research_variant
                else "pre_oos_input_only_fast_path_formula"
            ),
            "regime_decision": ctx.as_payload(),
            "barbell_weights": dict(weights),
            "crisis_sleeve_label": str(crisis.get("candidate_label")),
            "bull_sleeve_label": str(bull.get("candidate_label")) if bull is not None else None,
            "bull_sleeve_train_validation_score": train_validation_bull_score(bull) if bull is not None else None,
            "crisis_candidate_labels": list(spec.crisis_candidate_labels or (spec.crisis_label,)),
            "recency_halflife_folds": spec.recency_halflife_folds,
            "recency_min_history_folds": spec.recency_min_history_folds,
            "train": _combine_split(crisis, bull, weights, "train"),
            "validation": _combine_split(crisis, bull, weights, "validation"),
            "locked_oos": _combine_split(crisis, bull, weights, "locked_oos"),
            "mdd_proxy_note": "selector fold MDD is convex combination of source fold MDDs because intramonth equity curves are not in the existing WF JSON",
        }
        out.append(row)
    return out


def _return_between(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    if frame.empty:
        return None
    mask = (frame["datetime"] >= start) & (frame["datetime"] <= end)
    sub = frame.loc[mask]
    if len(sub) < 2:
        return None
    first = float(sub["close"].iloc[0])
    last = float(sub["close"].iloc[-1])
    if first <= 0.0:
        return None
    return float(last / first - 1.0)


def _ma_gap_at(frame: pd.DataFrame, end: pd.Timestamp, *, bars: int) -> float:
    if frame.empty:
        return 0.0
    sub = frame.loc[frame["datetime"] <= end].tail(max(2, int(bars)))
    if len(sub) < 2:
        return 0.0
    close = float(sub["close"].iloc[-1])
    ma = float(sub["close"].mean())
    return float(close / ma - 1.0) if ma > 0.0 else 0.0


def _last_validation_month_window(validation_start: pd.Timestamp, validation_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_period = validation_end.to_period("M")
    start = max(validation_start, end_period.start_time)
    return pd.Timestamp(start), validation_end


def classify_regime(
    *,
    validation_btc_return: float,
    last_validation_month_btc_return: float,
    validation_crypto_breadth: float,
    last_validation_month_crypto_breadth: float,
    validation_btc_ma_gap: float,
) -> tuple[str, str]:
    """Lagged regime gate. Uses only information ending before the OOS fold."""
    if last_validation_month_btc_return <= -0.02 or last_validation_month_crypto_breadth <= 0.35:
        return "BEAR", "last_validation_month_negative_or_breadth_breakdown"
    if (
        last_validation_month_btc_return >= 0.015
        and last_validation_month_crypto_breadth >= 0.45
        and validation_btc_return >= 0.0
        and validation_btc_ma_gap >= -0.015
    ):
        return (
            "BULL",
            "last_validation_month_positive_with_breadth_and_non_negative_refit_ma_gap",
        )
    if (
        last_validation_month_btc_return >= 0.020
        and last_validation_month_crypto_breadth >= 0.45
        and validation_btc_return <= -0.08
    ):
        return "BULL", "washout_recovery_bull_after_positive_last_validation_month"
    if validation_btc_return >= 0.04 and validation_crypto_breadth >= 0.55:
        return "BULL", "two_month_validation_uptrend_with_broad_crypto_participation"
    if validation_btc_return <= -0.06 and validation_crypto_breadth <= 0.50:
        return "BEAR", "two_month_validation_drawdown_with_weak_breadth"
    return "MIXED", "lagged_validation_regime_mixed_or_choppy"


def build_market_regimes(
    *,
    folds: Sequence[Mapping[str, Any]],
    data_root: Path,
    symbols: Sequence[str],
) -> dict[str, RegimeContext]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frames[str(symbol)] = broad69.load_symbol_bars(
                str(symbol).replace("/", ""),
                data_root=data_root,
                timeframe="30m",
            )
        except FileNotFoundError:
            continue
    if "BTC/USDT" not in frames and "BTCUSDT" in frames:
        frames["BTC/USDT"] = frames["BTCUSDT"]
    btc = frames.get("BTC/USDT")
    if btc is None:
        btc = frames.get("BTCUSDT")
    if btc is None or btc.empty:
        raise FileNotFoundError(f"BTCUSDT data not found under {data_root}")
    out: dict[str, RegimeContext] = {}
    for fold in folds:
        fold_id = str(fold["fold_id"])
        validation = dict(fold["validation"])
        locked_oos = dict(fold["locked_oos"])
        val_start = pd.Timestamp(validation["start"])
        val_end = pd.Timestamp(validation["end"])
        last_start, last_end = _last_validation_month_window(val_start, val_end)
        oos_start = pd.Timestamp(locked_oos["start"])
        oos_end = pd.Timestamp(locked_oos["end"])
        validation_btc = _return_between(btc, val_start, val_end) or 0.0
        last_btc = _return_between(btc, last_start, last_end) or 0.0
        val_symbol_returns = []
        last_symbol_returns = []
        for frame in frames.values():
            value = _return_between(frame, val_start, val_end)
            if value is not None:
                val_symbol_returns.append(value)
            last_value = _return_between(frame, last_start, last_end)
            if last_value is not None:
                last_symbol_returns.append(last_value)
        validation_breadth = (
            float(sum(value > 0.0 for value in val_symbol_returns) / len(val_symbol_returns))
            if val_symbol_returns
            else 0.0
        )
        last_breadth = (
            float(sum(value > 0.0 for value in last_symbol_returns) / len(last_symbol_returns))
            if last_symbol_returns
            else 0.0
        )
        ma_gap = _ma_gap_at(btc, val_end, bars=30 * 48)
        decision, reason = classify_regime(
            validation_btc_return=validation_btc,
            last_validation_month_btc_return=last_btc,
            validation_crypto_breadth=validation_breadth,
            last_validation_month_crypto_breadth=last_breadth,
            validation_btc_ma_gap=ma_gap,
        )
        out[fold_id] = RegimeContext(
            fold_id=fold_id,
            validation_btc_return=validation_btc,
            last_validation_month_btc_return=last_btc,
            validation_crypto_breadth=validation_breadth,
            last_validation_month_crypto_breadth=last_breadth,
            validation_btc_ma_gap=ma_gap,
            oos_btc_return=_return_between(btc, oos_start, oos_end),
            decision=decision,
            reason=reason,
        )
    return out


def _align_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for symbol, frame in frames.items():
        cols = frame[["datetime", "close"]].rename(columns={"close": symbol})
        merged = cols if merged is None else merged.merge(cols, on="datetime", how="inner")
    if merged is None:
        return pd.DataFrame()
    return merged.sort_values("datetime").reset_index(drop=True)


def _simulate_bull_bear_candidate(aligned: pd.DataFrame, *, params: Mapping[str, Any]) -> pd.DataFrame:
    symbols = [col for col in aligned.columns if col != "datetime"]
    if not symbols:
        return pd.DataFrame({"datetime": [], "return": [], "turnover": [], "equity": []})
    close = aligned[symbols].astype(float)
    simple_ret = close.pct_change().fillna(0.0)
    momentum_lookback = max(1, int(params.get("momentum_lookback", 48)))
    trend_ma_window = max(2, int(params.get("trend_ma_window", 48)))
    benchmark_symbol = str(params.get("benchmark_symbol") or "BTC/USDT")
    if benchmark_symbol not in symbols:
        compact = benchmark_symbol.replace("/", "")
        benchmark_symbol = compact if compact in symbols else symbols[0]
    benchmark_lookback = max(1, int(params.get("benchmark_lookback", momentum_lookback)))
    signal_threshold = max(0.0, float(params.get("signal_threshold", 0.015)))
    bull_breadth = max(0.0, min(1.0, float(params.get("bull_breadth", 0.58))))
    bear_breadth = max(0.0, min(1.0, float(params.get("bear_breadth", 0.55))))
    exit_breadth = max(0.0, min(1.0, float(params.get("exit_breadth", 0.40))))
    bench_bull = max(0.0, float(params.get("benchmark_bull_threshold", 0.005)))
    bench_bear = max(0.0, float(params.get("benchmark_bear_threshold", 0.005)))
    max_longs = max(0, int(params.get("max_longs", 8)))
    max_shorts = max(0, int(params.get("max_shorts", 6)))
    max_gross = max(0.0, float(params.get("max_gross", 1.0)))
    target_alloc = max(0.0, float(params.get("target_allocation", 0.90)))
    rebalance_bars = max(1, int(params.get("rebalance_bars", 3)))
    stop_loss_pct = max(0.0, float(params.get("stop_loss_pct", 0.10)))
    max_hold_bars = max(1, int(params.get("max_hold_bars", 180)))
    allow_short = bool(params.get("allow_short", True))
    cost_rate = max(0.0, float(params.get("cost_bps", 10.0))) / 10_000.0

    raw_mom = close / close.shift(momentum_lookback) - 1.0
    ma = close.rolling(trend_ma_window, min_periods=trend_ma_window).mean()
    score = raw_mom + 0.25 * (close / ma - 1.0)
    bench_ret = close[benchmark_symbol] / close[benchmark_symbol].shift(benchmark_lookback) - 1.0

    weights = np.zeros((len(aligned), len(symbols)), dtype=float)
    current = np.zeros(len(symbols), dtype=float)
    entry = np.full(len(symbols), np.nan, dtype=float)
    held = np.zeros(len(symbols), dtype=int)
    regime = "NEUTRAL"
    tick = 0
    for i in range(len(aligned)):
        row_scores = score.iloc[i]
        valid = row_scores.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            weights[i] = current
            continue
        up = valid[valid >= signal_threshold].sort_values(ascending=False)
        down = valid[valid <= -signal_threshold].sort_values(ascending=True)
        up_breadth = float(len(up) / len(valid))
        down_breadth = float(len(down) / len(valid))
        bench = float(bench_ret.iloc[i]) if math.isfinite(float(bench_ret.iloc[i])) else 0.0
        if up_breadth >= bull_breadth and bench >= bench_bull:
            next_regime = "BULL"
        elif allow_short and down_breadth >= bear_breadth and bench <= -bench_bear:
            next_regime = "BEAR"
        elif regime == "BULL" and up_breadth > exit_breadth and bench >= 0.0:
            next_regime = "BULL"
        elif regime == "BEAR" and down_breadth > exit_breadth and bench <= 0.0:
            next_regime = "BEAR"
        else:
            next_regime = "NEUTRAL"
        regime = next_regime
        tick += 1

        prices = close.iloc[i].to_numpy(dtype=float)
        active = current != 0.0
        held[active] += 1
        if stop_loss_pct > 0.0:
            long_stop = (current > 0.0) & np.isfinite(entry) & (prices <= entry * (1.0 - stop_loss_pct))
            short_stop = (current < 0.0) & np.isfinite(entry) & (prices >= entry * (1.0 + stop_loss_pct))
            expired = active & (held >= max_hold_bars)
            exit_mask = long_stop | short_stop | expired
            current[exit_mask] = 0.0
            entry[exit_mask] = np.nan
            held[exit_mask] = 0

        if regime == "NEUTRAL":
            current[:] = 0.0
            entry[:] = np.nan
            held[:] = 0
        elif tick % rebalance_bars == 0:
            target_names = list(up.index[:max_longs]) if regime == "BULL" else list(down.index[:max_shorts])
            direction = 1.0 if regime == "BULL" else -1.0
            scale = up_breadth if regime == "BULL" else down_breadth
            gross = target_alloc * min(max_gross, 1.0) * max(exit_breadth, scale)
            next_weights = np.zeros(len(symbols), dtype=float)
            if target_names:
                per_name = gross / len(target_names)
                for name in target_names:
                    next_weights[symbols.index(name)] = direction * per_name
            changed = np.abs(next_weights - current) > 1e-12
            entry[changed & (next_weights != 0.0)] = prices[changed & (next_weights != 0.0)]
            held[changed] = 0
            entry[next_weights == 0.0] = np.nan
            current = next_weights
        weights[i] = current

    gross_turnover = np.abs(weights - np.roll(weights, 1, axis=0)).sum(axis=1)
    gross_turnover[0] = np.abs(weights[0]).sum()
    portfolio_ret_raw = (np.roll(weights, 1, axis=0) * simple_ret.to_numpy(dtype=float)).sum(axis=1)
    portfolio_ret_raw[0] = 0.0
    portfolio_ret = portfolio_ret_raw - gross_turnover * cost_rate
    equity = np.cumprod(1.0 + portfolio_ret)
    return pd.DataFrame(
        {
            "datetime": aligned["datetime"].to_numpy(),
            "return": portfolio_ret,
            "turnover": gross_turnover,
            "equity": equity,
        }
    )


def _metrics_from_stream(stream: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    if stream.empty:
        return _metric_block(0.0, 0.0, start=start, end=end)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sub = stream.loc[(stream["datetime"] >= start_ts) & (stream["datetime"] <= end_ts)]
    if sub.empty:
        return _metric_block(0.0, 0.0, start=start, end=end)
    returns = sub["return"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + returns)
    total = float(equity[-1] - 1.0) if equity.size else 0.0
    mdd = _max_drawdown(equity)
    return _metric_block(total, mdd, start=start, end=end)


def evaluate_bull_bear_candidates(
    *,
    folds: Sequence[Mapping[str, Any]],
    data_root: Path,
    symbols: Sequence[str],
) -> list[dict[str, Any]]:
    candidate_defs = [
        row.to_dict()
        for row in build_binance_futures_candidates(
            timeframes=["30m", "1h", "4h", "1d"], symbols=list(symbols)
        )
        if row.strategy_class == "BullBearRegimeRotationStrategy"
    ]
    if not candidate_defs:
        return []
    by_timeframe: dict[str, pd.DataFrame] = {}
    for timeframe in sorted({str(item.get("strategy_timeframe") or item.get("timeframe")) for item in candidate_defs}):
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            compact = str(symbol).replace("/", "")
            try:
                frames[str(symbol)] = broad69.load_symbol_bars(
                    compact,
                    data_root=data_root,
                    timeframe=timeframe,
                )
            except FileNotFoundError:
                continue
        by_timeframe[timeframe] = _align_frames(frames)
    rows: list[dict[str, Any]] = []
    for candidate in candidate_defs:
        timeframe = str(candidate.get("strategy_timeframe") or candidate.get("timeframe"))
        aligned = by_timeframe.get(timeframe, pd.DataFrame())
        stream = _simulate_bull_bear_candidate(aligned, params=dict(candidate.get("params") or {}))
        for fold in folds:
            fold_id = str(fold["fold_id"])
            train = dict(fold["train"])
            validation = dict(fold["validation"])
            locked_oos = dict(fold["locked_oos"])
            label = f"new_bull_bear_regime_rotation:{candidate.get('name')}"
            rows.append(
                {
                    "candidate_label": label,
                    "fold_id": fold_id,
                    "family": "bull_bear_regime_rotation_latest_data",
                    "candidate_tier": "new_strategy_shadow_eval_latest_data",
                    "readiness_label": "shadow_only_research_candidate",
                    "allowed_usage_label": "shadow_only_not_real_money",
                    "clean_promotion_eligible": False,
                    "ready_for_paper": False,
                    "ready_for_real": False,
                    "real_money_execution": False,
                    "real_execution_allowed": False,
                    "uses_locked_oos_for_selection": False,
                    "current_fold_oos_used_for_weighting": False,
                    "selection_inputs": ["train", "validation", "pre_oos_static_candidate_definition"],
                    "selection_policy": "candidate_definition_static_then_train_validation_metrics_for_selector_pool",
                    "strategy_class": "BullBearRegimeRotationStrategy",
                    "source_candidate_id": candidate.get("candidate_id"),
                    "selected_symbols": list(symbols),
                    "selected_timeframes": [timeframe],
                    "timeframe": timeframe,
                    "train": _metrics_from_stream(stream, train["start"], train["end"]),
                    "validation": _metrics_from_stream(stream, validation["start"], validation["end"]),
                    "locked_oos": _metrics_from_stream(stream, locked_oos["start"], locked_oos["end"]),
                    "cost_model": "10bps_per_abs_weight_turnover_proxy",
                    "data_note": "evaluated after repository/data refresh; shadow-only and not part of original WF JSON",
                }
            )
    return rows


def _selector_specs() -> list[SelectorSpec]:
    return [
        SelectorSpec(
            label="regime_barbell_selector:v2_recency_weighted_raw_crisis_recovery_cash_guard_bull55",
            crisis_label=DEFAULT_CRISIS_RAW_LABEL,
            bull_clean_only=True,
            bull_weight=0.55,
            mixed_bull_weight=0.30,
            bear_bull_weight=0.0,
            min_bull_validation_return=0.0,
            max_bull_validation_mdd=0.20,
            recovery_bull_weight=0.0,
            recovery_crisis_weight=0.0,
            post_oos_research_variant=True,
            crisis_candidate_labels=(DEFAULT_CRISIS_RAW_LABEL, DEFAULT_CRISIS_FALLBACK_LABEL),
            recency_halflife_folds=2.0,
            recency_min_history_folds=3,
        ),
        SelectorSpec(
            label="regime_barbell_selector:v2_recovery_cash_guard_fallback_mdd20_bull55_mixed30_bear0",
            crisis_label=DEFAULT_CRISIS_FALLBACK_LABEL,
            bull_clean_only=True,
            bull_weight=0.55,
            mixed_bull_weight=0.30,
            bear_bull_weight=0.0,
            min_bull_validation_return=0.0,
            max_bull_validation_mdd=0.20,
            recovery_bull_weight=0.0,
            recovery_crisis_weight=0.0,
            post_oos_research_variant=True,
        ),
        SelectorSpec(
            label="regime_barbell_selector:v1_fallback_mdd20_bull65_mixed40_bear15",
            crisis_label=DEFAULT_CRISIS_FALLBACK_LABEL,
            bull_clean_only=False,
            bull_weight=0.65,
            mixed_bull_weight=0.40,
            bear_bull_weight=0.15,
            min_bull_validation_return=0.0,
            max_bull_validation_mdd=0.25,
        ),
        SelectorSpec(
            label="regime_barbell_selector:v1_raw_crisis_bull60_mixed35_bear10",
            crisis_label=DEFAULT_CRISIS_RAW_LABEL,
            bull_clean_only=False,
            bull_weight=0.60,
            mixed_bull_weight=0.35,
            bear_bull_weight=0.10,
            min_bull_validation_return=0.0,
            max_bull_validation_mdd=0.25,
        ),
        SelectorSpec(
            label="regime_barbell_selector:v1_clean_bull_fallback_mdd20_bull55_mixed30_bear10",
            crisis_label=DEFAULT_CRISIS_FALLBACK_LABEL,
            bull_clean_only=True,
            bull_weight=0.55,
            mixed_bull_weight=0.30,
            bear_bull_weight=0.10,
            min_bull_validation_return=0.0,
            max_bull_validation_mdd=0.20,
        ),
        SelectorSpec(
            label="regime_barbell_selector:v1_return_floor_fallback_mdd20_bull70_mixed35_bear0",
            crisis_label=DEFAULT_CRISIS_FALLBACK_LABEL,
            bull_clean_only=False,
            bull_weight=0.70,
            mixed_bull_weight=0.35,
            bear_bull_weight=0.0,
            min_bull_validation_return=0.01,
            max_bull_validation_mdd=0.18,
        ),
    ]


def _methodology_validation_audit(specs: Sequence[SelectorSpec]) -> dict[str, Any]:
    """Report whether the selector *design process* has independent WF evidence.

    Per-fold selector decisions can be contamination-free while the selector
    family/weights themselves remain post-OOS research hypotheses. Keep those
    two claims separate in the generated report.
    """
    post_oos_specs = [spec.label for spec in specs if spec.post_oos_research_variant]
    recency_specs = [spec.label for spec in specs if spec.recency_halflife_folds is not None]
    return {
        "per_fold_selector_decisions_walk_forward_valid": True,
        "current_fold_oos_used_by_selector_decisions": False,
        "prior_completed_oos_used_by_recency_weighting": bool(recency_specs),
        "prior_completed_oos_is_lagged_live_available": True,
        "selector_family_and_hyperparameters_chosen_after_source_oos_review": bool(post_oos_specs),
        "nested_walk_forward_methodology_validated": False,
        "methodology_status": "candidate_walk_forward_evaluated_not_methodology_walk_forward_validated",
        "post_oos_research_selector_labels": post_oos_specs,
        "recency_weighted_selector_labels": recency_specs,
        "clean_next_validation": (
            "Run a rolling-origin/nested selector-method WF where the selector family, "
            "candidate sleeves, weights, guards, and recency half-life are chosen only "
            "from data and completed folds available before each evaluated fold; then "
            "lock the method for fresh forward shadow."
        ),
    }


def _baseline_labels(payload: Mapping[str, Any]) -> list[str]:
    labels = [
        DEFAULT_CRISIS_RAW_LABEL,
        DEFAULT_CRISIS_FALLBACK_LABEL,
        DEFAULT_CLEAN_TOP_LABEL,
        DEFAULT_POSITIVE_BASELINE_LABEL,
    ]
    for row in list(payload.get("aggregate_rankings") or [])[:1]:
        label = str(row.get("candidate_label") or "")
        if label and label not in labels:
            labels.insert(0, label)
    return labels


def _baseline_comparison(payload: Mapping[str, Any], selector_aggregates: Sequence[Mapping[str, Any]], new_aggregates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source_aggs = {str(row.get("candidate_label")): dict(row) for row in list(payload.get("aggregate_rankings") or [])}
    out: list[dict[str, Any]] = []
    for label in _baseline_labels(payload):
        row = source_aggs.get(label)
        if row:
            item = {key: row.get(key) for key in (
                "candidate_label",
                "family",
                "compounded_oos_return",
                "positive_oos_folds",
                "fold_count",
                "max_oos_mdd",
                "monthly_equity_mdd",
                "monthly_sharpe_approx",
                "monthly_cvar_25",
                "min_oos_return",
                "latest_oos_return",
                "clean_promotion_eligible",
                "non_clean_reasons",
            )}
            item["comparison_role"] = "source_baseline"
            out.append(item)
    for row in selector_aggregates:
        item = dict(row)
        item["comparison_role"] = "new_selector"
        out.append(item)
    for row in new_aggregates[:4]:
        item = dict(row)
        item["comparison_role"] = "new_strategy_sleeve_eval"
        out.append(item)
    return out


def _fold_return_table(
    *,
    fold_order: Sequence[str],
    rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    selector_rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    regimes: Mapping[str, RegimeContext],
) -> list[dict[str, Any]]:
    labels = [
        DEFAULT_CRISIS_RAW_LABEL,
        DEFAULT_CRISIS_FALLBACK_LABEL,
        DEFAULT_CLEAN_TOP_LABEL,
        DEFAULT_POSITIVE_BASELINE_LABEL,
    ]
    out: list[dict[str, Any]] = []
    selector_lookup = {
        label: {str(row.get("fold_id")): row for row in rows}
        for label, rows in selector_rows_by_label.items()
    }
    source_lookup = {
        label: {str(row.get("fold_id")): row for row in rows_by_label.get(label, [])}
        for label in labels
    }
    for fold_id in fold_order:
        item: dict[str, Any] = {"fold_id": fold_id, "regime_decision": regimes[fold_id].decision}
        for label, by_fold in source_lookup.items():
            short = label
            if label == DEFAULT_CRISIS_RAW_LABEL:
                short = "raw_crisis"
            elif label == DEFAULT_CRISIS_FALLBACK_LABEL:
                short = "fallback_crisis"
            elif label == DEFAULT_CLEAN_TOP_LABEL:
                short = "clean_top"
            elif label == DEFAULT_POSITIVE_BASELINE_LABEL:
                short = "positive_fold_baseline"
            row = by_fold.get(fold_id)
            item[f"{short}_oos_return"] = _row_return(row, "locked_oos") if row else None
        for label, by_fold in selector_lookup.items():
            row = by_fold.get(fold_id)
            item[f"{label}_oos_return"] = _row_return(row, "locked_oos") if row else None
            if row:
                item[f"{label}_bull_sleeve"] = row.get("bull_sleeve_label")
                item[f"{label}_weights"] = row.get("barbell_weights")
        out.append(item)
    return out


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value) * 100:.2f}%"


def _candidate_fold_return_table(
    *,
    fold_order: Sequence[str],
    rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_labels: Sequence[str],
) -> list[dict[str, Any]]:
    lookups = {
        label: {str(row.get("fold_id")): row for row in rows_by_label.get(label, [])}
        for label in candidate_labels
    }
    out: list[dict[str, Any]] = []
    for fold_id in fold_order:
        item: dict[str, Any] = {"fold_id": fold_id}
        for label, by_fold in lookups.items():
            row = by_fold.get(fold_id)
            item[f"{label}_oos_return"] = _row_return(row, "locked_oos") if row else None
        out.append(item)
    return out


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Regime-aware barbell selector analysis")
    lines.append("")
    lines.append("Shadow/research only. `ready_for_real=false`; current-fold OOS is not used for selector decisions.")
    lines.append("")
    lines.append("## Contamination audit")
    audit = dict(payload.get("contamination_audit") or {})
    for key, value in audit.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Methodology validation audit")
    method_audit = dict(payload.get("methodology_validation_audit") or {})
    for key, value in method_audit.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## Baseline / selector comparison")
    lines.append("| role | label | comp | positive folds | min OOS | latest OOS | max OOS MDD | monthly MDD | Sharpe | CVaR25 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload.get("baseline_comparison", []):
        lines.append(
            "| "
            f"{row.get('comparison_role')} | `{row.get('candidate_label')}` | "
            f"{_fmt_pct(row.get('compounded_oos_return'))} | "
            f"{int(_safe_float(row.get('positive_oos_folds')))}/{int(_safe_float(row.get('fold_count')))} | "
            f"{_fmt_pct(row.get('min_oos_return'))} | "
            f"{_fmt_pct(row.get('latest_oos_return'))} | "
            f"{_fmt_pct(row.get('max_oos_mdd'))} | "
            f"{_fmt_pct(row.get('monthly_equity_mdd'))} | "
            f"{_safe_float(row.get('monthly_sharpe_approx')):.2f} | "
            f"{_fmt_pct(row.get('monthly_cvar_25'))} |"
        )
    lines.append("")
    lines.append("## Fold regime and OOS return table")
    selector_labels = [row["candidate_label"] for row in payload.get("selector_rankings", [])]
    table = list(payload.get("fold_oos_return_table") or [])
    header = ["fold", "regime", "raw", "fallback", "clean", "pos_base", *[f"selector_{i+1}" for i in range(len(selector_labels))]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in table:
        values = [
            str(row.get("fold_id")),
            str(row.get("regime_decision")),
            _fmt_pct(row.get("raw_crisis_oos_return")),
            _fmt_pct(row.get("fallback_crisis_oos_return")),
            _fmt_pct(row.get("clean_top_oos_return")),
            _fmt_pct(row.get("positive_fold_baseline_oos_return")),
        ]
        for label in selector_labels:
            values.append(_fmt_pct(row.get(f"{label}_oos_return")))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Selector labels:")
    for idx, label in enumerate(selector_labels, start=1):
        lines.append(f"- selector_{idx}: `{label}`")
    lines.append("")
    new_labels = [row["candidate_label"] for row in (payload.get("new_strategy_evaluation", {}).get("aggregate_rankings") or [])[:4]]
    new_table = list(payload.get("new_strategy_fold_oos_return_table") or [])
    if new_labels and new_table:
        lines.append("## Newly added strategy fold OOS table")
        header = ["fold", *[f"new_{i+1}" for i in range(len(new_labels))]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in new_table:
            values = [str(row.get("fold_id"))]
            for label in new_labels:
                values.append(_fmt_pct(row.get(f"{label}_oos_return")))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
        lines.append("New strategy labels:")
        for idx, label in enumerate(new_labels, start=1):
            lines.append(f"- new_{idx}: `{label}`")
        lines.append("")
    lines.append("## Regime decision audit")
    lines.append("| fold | decision | val BTC | last-val-month BTC | val breadth | last breadth | OOS BTC (analysis only) | reason |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for item in payload.get("market_regime_by_fold", []):
        lines.append(
            "| "
            f"{item.get('fold_id')} | {item.get('decision')} | "
            f"{_fmt_pct(item.get('validation_btc_return'))} | "
            f"{_fmt_pct(item.get('last_validation_month_btc_return'))} | "
            f"{_safe_float(item.get('validation_crypto_breadth')):.2f} | "
            f"{_safe_float(item.get('last_validation_month_crypto_breadth')):.2f} | "
            f"{_fmt_pct(item.get('oos_btc_return_analysis_only'))} | "
            f"{item.get('reason')} |"
        )
    lines.append("")
    lines.append("## Notes")
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    source_payload: Mapping[str, Any],
    source_json: Path,
    data_root: Path,
    evaluate_new_strategy: bool,
) -> dict[str, Any]:
    folds = list(source_payload.get("folds") or [])
    fold_order = [str(fold["fold_id"]) for fold in folds]
    source_rows = [dict(row) for row in list(source_payload.get("fold_candidate_rows") or [])]
    symbols = list(BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS)
    regimes = build_market_regimes(folds=folds, data_root=data_root, symbols=symbols)
    new_rows: list[dict[str, Any]] = []
    new_eval_error: str | None = None
    if evaluate_new_strategy:
        try:
            new_rows = evaluate_bull_bear_candidates(folds=folds, data_root=data_root, symbols=symbols)
        except Exception as exc:  # report-only path: preserve main selector analysis.
            new_eval_error = f"{type(exc).__name__}: {exc}"
    all_rows = [*source_rows, *new_rows]
    rows_by_fold = _group_rows_by_fold(all_rows)
    selector_rows_by_label: dict[str, list[dict[str, Any]]] = {}
    selector_aggregates: list[dict[str, Any]] = []
    selector_specs = _selector_specs()
    for spec in selector_specs:
        rows = build_selector_rows(
            fold_order=fold_order,
            rows_by_fold=rows_by_fold,
            regimes=regimes,
            spec=spec,
        )
        selector_rows_by_label[spec.label] = rows
        aggregate = _aggregate_rows(spec.label, rows)
        aggregate.update(
            {
                "family": "regime_barbell_selector",
                "clean_promotion_eligible": False,
                "non_clean_reasons": [
                    "new_shadow_selector_requires_fresh_forward_shadow",
                    *(
                        ["post_oos_research_variant_requires_fresh_forward_shadow"]
                        if spec.post_oos_research_variant
                        else []
                    ),
                ],
                "post_oos_research_variant": bool(spec.post_oos_research_variant),
                "uses_locked_oos_for_selection": False,
                "current_fold_oos_used_for_weighting": False,
                "ready_for_real": False,
                "real_money_execution": False,
                "real_execution_allowed": False,
            }
        )
        selector_aggregates.append(aggregate)
    selector_aggregates.sort(
        key=lambda row: (
            int(row.get("positive_oos_folds", 0)),
            _safe_float(row.get("compounded_oos_return")),
            _safe_float(row.get("min_oos_return")),
            -_safe_float(row.get("max_oos_mdd")),
        ),
        reverse=True,
    )
    new_aggregates = list(_aggregate_rows_by_label(new_rows))
    new_aggregates.sort(
        key=lambda row: (
            _safe_float(row.get("compounded_oos_return")),
            int(row.get("positive_oos_folds", 0)),
            _safe_float(row.get("min_oos_return")),
        ),
        reverse=True,
    )
    rows_by_label = _group_rows_by_label(all_rows)
    payload = {
        "artifact_kind": "regime_barbell_selector_fast_path_analysis",
        "generated_at_utc": _utc_now_iso(),
        "source_json": str(source_json),
        "data_root": str(data_root),
        "source_data_coverage": source_payload.get("data_coverage"),
        "analysis_scope": "existing_full_wf_json_plus_latest_data_new_bull_bear_shadow_eval",
        "real_money_execution": False,
        "ready_for_real": False,
        "market_regime_by_fold": [regimes[fold_id].as_payload() for fold_id in fold_order],
        "selector_rankings": selector_aggregates,
        "new_strategy_evaluation": {
            "enabled": bool(evaluate_new_strategy),
            "error": new_eval_error,
            "row_count": len(new_rows),
            "aggregate_rankings": new_aggregates,
        },
        "baseline_comparison": _baseline_comparison(source_payload, selector_aggregates, new_aggregates),
        "fold_oos_return_table": _fold_return_table(
            fold_order=fold_order,
            rows_by_label=rows_by_label,
            selector_rows_by_label=selector_rows_by_label,
            regimes=regimes,
        ),
        "new_strategy_fold_oos_return_table": _candidate_fold_return_table(
            fold_order=fold_order,
            rows_by_label=rows_by_label,
            candidate_labels=[str(row.get("candidate_label")) for row in new_aggregates[:4]],
        ),
        "regime_decision_audit": {
            label: [
                {
                    "fold_id": row["fold_id"],
                    "regime_decision": row["regime_decision"],
                    "barbell_weights": row["barbell_weights"],
                    "crisis_sleeve_label": row["crisis_sleeve_label"],
                    "bull_sleeve_label": row["bull_sleeve_label"],
                    "bull_sleeve_train_validation_score": row["bull_sleeve_train_validation_score"],
                    "oos_return_after_frozen_decision": row["locked_oos"]["total_return"],
                }
                for row in rows
            ]
            for label, rows in selector_rows_by_label.items()
        },
        "methodology_validation_audit": _methodology_validation_audit(selector_specs),
        "contamination_audit": {
            "selector_uses_current_fold_oos_for_gate": False,
            "selector_uses_current_fold_oos_for_bull_sleeve_selection": False,
            "selector_weight_inputs": "train metrics, validation metrics, validation-end BTC/crypto regime features only",
            "current_fold_oos_role": "evaluation_after_decision_only",
            "new_strategy_real_money": False,
            "shadow_only": True,
            "post_oos_research_variant_present": any(bool(row.get("post_oos_research_variant")) for row in selector_aggregates),
            "post_oos_research_variant_interpretation": "hypothesis_only_not_clean_scientific_evidence",
        },
        "notes": [
            "This is a fast-path recombination/evaluation report; it does not replace a full fresh monthly WF rerun.",
            "Selector fold MDD uses a convex source-row MDD proxy because existing WF JSON does not carry intramonth equity curves.",
            "New BullBearRegimeRotationStrategy rows are evaluated from latest refreshed data as research/shadow sleeves only.",
            "The v2 selectors are explicitly flagged as post-OOS research variants; treat their results as hypotheses needing nested method WF or fresh forward shadow, not as clean evidence.",
            "No ready_for_real claim is made; fresh forward shadow/paper-testnet evidence remains required.",
        ],
    }
    return payload


def _aggregate_rows_by_label(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_rows_by_label(rows)
    return [_aggregate_rows(label, grouped[label]) for label in sorted(grouped)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--skip-new-strategy-eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_json = Path(args.source_json).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    source_payload = json.loads(source_json.read_text("utf-8"))
    payload = build_report(
        source_payload=source_payload,
        source_json=source_json,
        data_root=data_root,
        evaluate_new_strategy=not bool(args.skip_new_strategy_eval),
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), "utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "top_selector": payload["selector_rankings"][0] if payload.get("selector_rankings") else None,
                "new_strategy_error": payload["new_strategy_evaluation"].get("error"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
