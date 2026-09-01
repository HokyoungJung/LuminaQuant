"""Causal dynamic portfolio allocator over saved sleeve return streams."""

from __future__ import annotations

import argparse
import itertools
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from lumina_quant.eval.exact_window_runtime import RSSLimitExceeded
from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    OOS_START_DATE,
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    PORTFOLIO_CURRENT_OPTIMIZATION,
    PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE,
    PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE,
    TRAIN_END_EXCLUSIVE,
    TRAIN_START,
    VAL_END_EXCLUSIVE,
    VAL_START as VAL_START_ISO,
    VAL_START_DATE,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
    resolve_incumbent_bundle_path,
    split_for_day_key,
    split_windows,
)

DEFAULT_INPUT = PORTFOLIO_ONE_SHOT_INCUMBENT_BUNDLE
DEFAULT_OUTPUT_DIR = FOLLOWUP_ROOT / "portfolio_dynamic_online_current"
COMPARISON_INPUT = FOLLOWUP_ROOT / "portfolio_comparison_latest.json"
VAL_START = VAL_START_DATE
OOS_START = OOS_START_DATE


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except TypeError, ValueError:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _canonical_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    numeric: float | None = None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str) and value.strip():
        token = value.strip()
        try:
            numeric = float(token)
        except ValueError:
            normalized = token.replace("Z", "+00:00") if token.endswith("Z") else token
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
    if numeric is None or not math.isfinite(numeric):
        return None
    magnitude = abs(numeric)
    if magnitude >= 1e15:
        return datetime.fromtimestamp(numeric / 1_000_000.0, tz=UTC)
    if magnitude >= 1e12:
        return datetime.fromtimestamp(numeric / 1_000.0, tz=UTC)
    if magnitude >= 1e9:
        return datetime.fromtimestamp(numeric, tz=UTC)
    return None


def _daily_compound_stream(stream: list[dict[str, Any]]) -> dict[str, float]:
    bucket: dict[str, list[float]] = {}
    for point in list(stream or []):
        raw_ts = point.get("datetime", point.get("t", point.get("timestamp")))
        dt = _canonical_timestamp(raw_ts)
        if dt is None:
            continue
        key = dt.astimezone(UTC).date().isoformat()
        bucket.setdefault(key, []).append(_safe_float(point.get("v"), 0.0))
    out: dict[str, float] = {}
    for day_key, returns in bucket.items():
        compounded = 1.0
        for ret in returns:
            compounded *= 1.0 + ret
        out[day_key] = compounded - 1.0
    return out


def _current_one_shot_comparison_entry(
    *,
    bundle_path: Path | None = None,
    portfolio_path: Path | None = None,
) -> dict[str, Any]:
    bundle_path = bundle_path or PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE
    portfolio_path = portfolio_path or PORTFOLIO_CURRENT_OPTIMIZATION
    portfolio_payload = _load_json(portfolio_path)
    _load_json(bundle_path)
    portfolio_metrics = dict(portfolio_payload.get("portfolio_metrics") or {})
    return {
        "path": str(portfolio_path.resolve()),
        "bundle_path": str(bundle_path.resolve()),
        "weights": list(portfolio_payload.get("weights") or []),
        "val": dict(portfolio_metrics.get("val") or {}),
        "oos": dict(portfolio_metrics.get("oos") or {}),
    }


def _maybe_current_one_shot_comparison_entry(
    *,
    bundle_path: Path | None = None,
    portfolio_path: Path | None = None,
) -> dict[str, Any] | None:
    bundle_path = bundle_path or PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE
    portfolio_path = portfolio_path or PORTFOLIO_CURRENT_OPTIMIZATION
    if not bundle_path.exists() or not portfolio_path.exists():
        return None
    return _current_one_shot_comparison_entry(
        bundle_path=bundle_path,
        portfolio_path=portfolio_path,
    )


def _metrics(returns: np.ndarray, periods_per_year: int = 365) -> dict[str, float]:
    if returns.size == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
        }
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    years = max(returns.size / float(periods_per_year), 1.0 / float(periods_per_year))
    cagr = float((equity[-1] ** (1.0 / years)) - 1.0) if equity[-1] > 0 else -1.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = float((mean / std) * math.sqrt(periods_per_year)) if std > 1e-12 else 0.0
    downside = returns[returns < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = (
        float((mean / downside_std) * math.sqrt(periods_per_year)) if downside_std > 1e-12 else 0.0
    )
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    max_drawdown = float(abs(np.min(drawdown))) if drawdown.size else 0.0
    calmar = float(cagr / max_drawdown) if max_drawdown > 1e-12 else 0.0
    volatility = float(std * math.sqrt(periods_per_year))
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
    }


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return float(abs(np.min(drawdown)))


@dataclass(slots=True)
class AllocatorParams:
    lookback_days: int
    rebalance_days: int
    min_trailing_sharpe: float
    min_trailing_return: float
    max_trailing_drawdown: float
    max_weight: float
    max_family_weight: float = 1.0
    correlation_penalty: float = 0.0
    cash_when_no_active: bool = True
    use_regime_features: bool = False
    regime_strength: float = 1.0


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)]
    if not rows:
        raise RuntimeError(f"no candidates found in {path}")
    return rows


def _build_daily_panel(
    rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    all_days: set[str] = set()
    series: dict[str, dict[str, float]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row.get("candidate_id") or row.get("name"))
        merged: dict[str, float] = {}
        for split in ("train", "val", "oos"):
            merged.update(
                _daily_compound_stream(list(((row.get("return_streams") or {}).get(split)) or []))
            )
        series[cid] = merged
        all_days.update(merged.keys())
        meta[cid] = {
            "name": row.get("name"),
            "strategy_class": row.get("strategy_class"),
            "family": row.get("family"),
            "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
            "metadata": dict(row.get("metadata") or {}),
        }
    ordered_days = sorted(all_days)
    matrix: dict[str, np.ndarray] = {}
    for cid, mapping in series.items():
        matrix[cid] = np.asarray(
            [_safe_float(mapping.get(day_key), 0.0) for day_key in ordered_days], dtype=float
        )
    return ordered_days, matrix, meta


def _split_index(day_key: str) -> str:
    return split_for_day_key(day_key)


def _load_regime_features(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    symbols = sorted(
        {str(symbol) for row in rows for symbol in list(row.get("symbols") or []) if str(symbol)}
    )
    if not symbols:
        return {}
    max_dt: datetime | None = None
    for row in rows:
        for split in ("train", "val", "oos"):
            for point in list(((row.get("return_streams") or {}).get(split)) or []):
                dt = _canonical_timestamp(
                    point.get("datetime", point.get("t", point.get("timestamp")))
                )
                if dt is None:
                    continue
                max_dt = dt if max_dt is None or dt > max_dt else max_dt
    if max_dt is None:
        return {}
    windows = {
        "train_start": TRAIN_START,
        "train_end_exclusive": TRAIN_END_EXCLUSIVE,
        "val_start": VAL_START_ISO,
        "val_end_exclusive": VAL_END_EXCLUSIVE,
        "actual_oos_end_exclusive": (max_dt + timedelta(days=1))
        .astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    module_path = Path(__file__).resolve().parent / "rolling_breakout_30m_regime_gate.py"
    spec = importlib.util.spec_from_file_location("rolling_breakout_30m_regime_gate", module_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        feature_frame = module._daily_feature_frame(symbols=symbols, windows=windows)
    except Exception:
        return {}
    features: dict[str, dict[str, Any]] = {}
    for row in feature_frame.to_dict(orient="records"):
        raw_date = row.get("date")
        if hasattr(raw_date, "date"):
            day_key = raw_date.date().isoformat()
        else:
            parsed = _canonical_timestamp(raw_date)
            if parsed is None:
                continue
            day_key = parsed.date().isoformat()
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if key == "date":
                continue
            if isinstance(value, (bool, np.bool_)):
                normalized[key] = bool(value)
            else:
                normalized[key] = (
                    _safe_float(value, 0.0)
                    if isinstance(value, (int, float, np.floating))
                    else value
                )
        features[day_key] = normalized
    return features


def _regime_multiplier(
    meta_row: dict[str, Any],
    regime_row: dict[str, Any],
    *,
    previous_active: bool,
    strength: float,
) -> float:
    if not regime_row or strength <= 0.0:
        return 1.0
    strategy_class = str(meta_row.get("strategy_class") or "")
    metadata = dict(meta_row.get("metadata") or {})
    multiplier = 1.0

    btc_up = bool(regime_row.get("btc_above_ma192", False))
    breadth_up = bool(regime_row.get("breadth_ma96_ge_60", False))
    vol_moderate = bool(regime_row.get("basket_vol_ratio_moderate", False))

    if strategy_class == "RollingBreakoutStrategy":
        conditions = list(metadata.get("activation_rule_conditions") or [])
        if conditions and not all(
            bool(regime_row.get(condition, False)) for condition in conditions
        ):
            return 0.0
        multiplier *= 1.0 + (0.35 * strength)
    elif strategy_class == "CompositeTrendStrategy":
        multiplier *= (1.0 + (0.25 * strength)) if btc_up else max(0.10, 1.0 - (0.55 * strength))
        multiplier *= (
            (1.0 + (0.10 * strength)) if breadth_up else max(0.20, 1.0 - (0.25 * strength))
        )
    elif strategy_class == "RegimeBreakoutCandidateStrategy":
        regime_ok = btc_up and breadth_up
        multiplier *= (1.0 + (0.30 * strength)) if regime_ok else max(0.10, 1.0 - (0.60 * strength))
        if vol_moderate:
            multiplier *= 1.0 + (0.10 * strength)
    elif strategy_class == "TopCapTimeSeriesMomentumStrategy":
        multiplier *= (1.0 + (0.15 * strength)) if btc_up else max(0.25, 1.0 - (0.35 * strength))
        if breadth_up:
            multiplier *= 1.0 + (0.05 * strength)

    if previous_active:
        multiplier *= 1.0 + (0.05 * strength)
    return max(0.0, float(multiplier))


def _compute_raw_scores(
    history: dict[str, np.ndarray],
    *,
    meta: dict[str, dict[str, Any]],
    regime_row: dict[str, Any] | None,
    previous_weights: dict[str, float] | None,
    min_trailing_sharpe: float,
    min_trailing_return: float,
    max_trailing_drawdown: float,
    use_regime_features: bool,
    regime_strength: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    raw_scores: dict[str, float] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for cid, returns in history.items():
        if returns.size == 0:
            diagnostics[cid] = {
                "trailing_return": 0.0,
                "trailing_sharpe": 0.0,
                "trailing_vol": 0.0,
                "trailing_drawdown": 0.0,
            }
            continue
        metric = _metrics(returns)
        trailing_return = float(metric["total_return"])
        trailing_sharpe = float(metric["sharpe"])
        trailing_vol = float(metric["volatility"])
        trailing_drawdown = _max_drawdown(returns)
        diagnostics[cid] = {
            "trailing_return": trailing_return,
            "trailing_sharpe": trailing_sharpe,
            "trailing_vol": trailing_vol,
            "trailing_drawdown": trailing_drawdown,
        }
        if trailing_sharpe < min_trailing_sharpe:
            continue
        if trailing_return < min_trailing_return:
            continue
        if trailing_drawdown > max_trailing_drawdown:
            continue
        vol_inv = 1.0 / max(trailing_vol, 1e-6)
        dd_penalty = max(0.05, 1.0 - (trailing_drawdown / max(max_trailing_drawdown, 1e-6)))
        regime_multiplier = 1.0
        if use_regime_features:
            regime_multiplier = _regime_multiplier(
                meta.get(cid) or {},
                regime_row or {},
                previous_active=bool((previous_weights or {}).get(cid, 0.0) > 0.0),
                strength=regime_strength,
            )
        diagnostics[cid]["regime_multiplier"] = float(regime_multiplier)
        if regime_multiplier <= 0.0:
            continue
        raw_scores[cid] = float(
            (1.0 + max(0.0, trailing_sharpe))
            * (1.0 + (8.0 * max(0.0, trailing_return)))
            * vol_inv
            * dd_penalty
            * regime_multiplier
        )
    return raw_scores, diagnostics


def _cap_and_normalize(raw_scores: dict[str, float], *, max_weight: float) -> dict[str, float]:
    if not raw_scores:
        return {}
    weights = {
        cid: score / max(sum(raw_scores.values()), 1e-12) for cid, score in raw_scores.items()
    }
    capped = {cid: min(max_weight, weight) for cid, weight in weights.items()}
    total = sum(capped.values())
    if total <= 1e-12:
        return {}
    # Do not renormalize after capping. Residual allocation remains as cash.
    return capped


def _family_of(meta_item: dict[str, Any]) -> str:
    return str(
        meta_item.get("family") or (meta_item.get("metadata") or {}).get("family") or "other"
    )


def _active_weighting(
    history_window: dict[str, np.ndarray],
    raw_scores: dict[str, float],
    *,
    meta: dict[str, dict[str, Any]],
    max_weight: float,
    max_family_weight: float,
    correlation_penalty: float,
) -> dict[str, float]:
    if not raw_scores:
        return {}
    ids = list(raw_scores.keys())
    vols: dict[str, float] = {}
    corr_penalties: dict[str, float] = {}
    for cid in ids:
        arr = np.asarray(history_window.get(cid, []), dtype=float)
        vols[cid] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    for cid in ids:
        arr_i = np.asarray(history_window.get(cid, []), dtype=float)
        pair_corrs: list[float] = []
        for other in ids:
            if other == cid:
                continue
            arr_j = np.asarray(history_window.get(other, []), dtype=float)
            n = min(arr_i.size, arr_j.size)
            if n < 3:
                continue
            x = arr_i[-n:]
            y = arr_j[-n:]
            sx = float(np.std(x, ddof=1))
            sy = float(np.std(y, ddof=1))
            if sx <= 1e-12 or sy <= 1e-12:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if math.isfinite(corr):
                pair_corrs.append(max(0.0, corr))
        corr_penalties[cid] = float(np.mean(pair_corrs)) if pair_corrs else 0.0

    adjusted_scores = {
        cid: float(
            raw_scores[cid]
            * (1.0 / max(vols[cid], 1e-6))
            / (1.0 + (correlation_penalty * corr_penalties[cid]))
        )
        for cid in ids
    }
    score_total = sum(max(0.0, score) for score in adjusted_scores.values())
    if score_total <= 1e-12:
        return {}

    desired = {cid: max(0.0, adjusted_scores[cid]) / score_total for cid in ids}
    ranked = sorted(desired.items(), key=lambda item: item[1], reverse=True)
    final: dict[str, float] = {}
    family_used: dict[str, float] = {}
    remaining_total = 1.0
    for cid, target in ranked:
        family = _family_of(meta.get(cid) or {})
        fam_remaining = max(0.0, max_family_weight - family_used.get(family, 0.0))
        allowed = min(max_weight, fam_remaining, remaining_total, target)
        if allowed <= 1e-12:
            continue
        final[cid] = float(allowed)
        family_used[family] = family_used.get(family, 0.0) + float(allowed)
        remaining_total -= float(allowed)
        if remaining_total <= 1e-12:
            break
    return final


def run_causal_dynamic_allocator(
    rows: list[dict[str, Any]],
    params: AllocatorParams,
    *,
    regime_features: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ordered_days, matrix, meta = _build_daily_panel(rows)
    ids = list(matrix.keys())
    resolved_regime_features = (
        regime_features
        if regime_features is not None
        else (_load_regime_features(rows) if params.use_regime_features else {})
    )
    allocations: list[dict[str, Any]] = []
    weights: dict[str, float] = {}
    daily_returns: list[float] = []
    day_splits: list[str] = []
    split_returns: dict[str, list[float]] = {"train": [], "val": [], "oos": []}

    for idx, day_key in enumerate(ordered_days):
        split = _split_index(day_key)
        rebalance_now = idx > 0 and (not weights or (idx % max(1, params.rebalance_days) == 0))
        if rebalance_now:
            prior_regime_key = ordered_days[idx - 1] if idx > 0 else None
            regime_row = (
                dict(resolved_regime_features.get(prior_regime_key) or {})
                if prior_regime_key
                else {}
            )
            history_window = {
                cid: matrix[cid][max(0, idx - params.lookback_days) : idx] for cid in ids
            }
            raw_scores, diagnostics = _compute_raw_scores(
                history_window,
                meta=meta,
                regime_row=regime_row,
                previous_weights=weights,
                min_trailing_sharpe=params.min_trailing_sharpe,
                min_trailing_return=params.min_trailing_return,
                max_trailing_drawdown=params.max_trailing_drawdown,
                use_regime_features=params.use_regime_features,
                regime_strength=params.regime_strength,
            )
            weights = _active_weighting(
                history_window,
                raw_scores,
                meta=meta,
                max_weight=params.max_weight,
                max_family_weight=params.max_family_weight,
                correlation_penalty=params.correlation_penalty,
            )
            cash_weight = 1.0 - sum(weights.values())
            allocations.append(
                {
                    "date": day_key,
                    "weights": dict(weights),
                    "cash_weight": max(0.0, cash_weight) if params.cash_when_no_active else 0.0,
                    "diagnostics": diagnostics,
                    "raw_scores": raw_scores,
                    "regime_row": regime_row,
                }
            )
        elif idx == 0:
            allocations.append(
                {
                    "date": day_key,
                    "weights": {},
                    "cash_weight": 1.0,
                    "diagnostics": {},
                    "raw_scores": {},
                    "regime_row": {},
                }
            )

        portfolio_ret = 0.0
        for cid, weight in weights.items():
            portfolio_ret += weight * float(matrix[cid][idx])
        if not weights and not params.cash_when_no_active:
            equal_weight = 1.0 / max(1, len(ids))
            portfolio_ret = float(sum(equal_weight * float(matrix[cid][idx]) for cid in ids))
        daily_returns.append(portfolio_ret)
        day_splits.append(split)
        split_returns[split].append(portfolio_ret)

    all_metrics = _metrics(np.asarray(daily_returns, dtype=float))
    return {
        "dates": ordered_days,
        "ids": ids,
        "meta": meta,
        "allocations": allocations,
        "daily_returns": daily_returns,
        "split_metrics": {
            split: _metrics(np.asarray(values, dtype=float))
            for split, values in split_returns.items()
        },
        "all_metrics": all_metrics,
        "day_splits": day_splits,
    }


def _search_objective(metrics: dict[str, float], *, cash_fraction: float = 0.0) -> float:
    return float(
        (1.0 * _safe_float(metrics.get("sharpe"), 0.0))
        + (0.35 * _safe_float(metrics.get("sortino"), 0.0))
        + (0.10 * _safe_float(metrics.get("calmar"), 0.0))
        + (10.0 * _safe_float(metrics.get("total_return"), 0.0))
        - (4.0 * _safe_float(metrics.get("max_drawdown"), 0.0))
        - (0.75 * _safe_float(metrics.get("volatility"), 0.0))
        - (0.25 * cash_fraction)
    )


def _mean_cash_fraction(allocations: list[dict[str, Any]], *, split: str) -> float:
    if not allocations:
        return 0.0
    values = [
        _safe_float(item.get("cash_weight"), 0.0)
        for item in allocations
        if _split_index(str(item.get("date") or "")) == split
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def _portfolio_return_streams_from_daily(
    dates: list[str],
    daily_returns: list[float],
) -> dict[str, list[dict[str, Any]]]:
    streams: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
    for day_key, day_return in zip(dates, daily_returns, strict=True):
        split = _split_index(str(day_key))
        streams[split].append({"t": f"{day_key}T00:00:00Z", "v": _safe_float(day_return, 0.0)})
    return streams


def search_dynamic_allocator(
    rows: list[dict[str, Any]],
    *,
    param_grid: list[AllocatorParams] | None = None,
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    grid = param_grid or [
        AllocatorParams(
            lookback_days=lb,
            rebalance_days=rb,
            min_trailing_sharpe=ms,
            min_trailing_return=mr,
            max_trailing_drawdown=mdd,
            max_weight=mw,
            max_family_weight=mfw,
            correlation_penalty=cp,
            use_regime_features=True,
            regime_strength=rs,
        )
        for lb, rb, ms, mr, mdd, mw, mfw, cp, rs in itertools.product(
            [5, 10, 20],
            [1, 3],
            [0.0, 0.25, 0.5],
            [0.0],
            [0.10, 0.15],
            [0.40, 0.50],
            [0.55, 0.70, 1.0],
            [0.0, 0.5, 1.0],
            [0.5, 1.0, 1.5],
        )
    ]
    best: dict[str, Any] | None = None
    regime_features = _load_regime_features(rows)
    total_candidates = len(grid)
    for idx, params in enumerate(grid, start=1):
        result = run_causal_dynamic_allocator(rows, params, regime_features=regime_features)
        val_metrics = dict((result.get("split_metrics") or {}).get("val") or {})
        allocations = list(result.get("allocations") or [])
        cash_fraction = _mean_cash_fraction(allocations, split="val")
        objective = _search_objective(val_metrics, cash_fraction=cash_fraction)
        candidate = {
            "params": asdict(params),
            "objective": objective,
            "result": result,
        }
        if best is None or objective > float(best["objective"]):
            best = candidate
        if progress_callback is not None:
            progress_callback(
                "dynamic_candidate_evaluated",
                {
                    "candidate_index": idx,
                    "candidate_count": total_candidates,
                    "objective": objective,
                    "val_total_return": _safe_float(val_metrics.get("total_return"), 0.0),
                    "val_sharpe": _safe_float(val_metrics.get("sharpe"), 0.0),
                    "params": dict(candidate["params"]),
                },
            )
    if best is None:
        raise RuntimeError("dynamic allocator search produced no result")
    return best


def _top_allocation_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    allocations = list(result.get("allocations") or [])
    if not allocations:
        return []
    latest = allocations[-1]
    weights = dict(latest.get("weights") or {})
    meta = dict(result.get("meta") or {})
    ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    rows: list[dict[str, Any]] = []
    for cid, weight in ranked:
        item = dict(meta.get(cid) or {})
        rows.append(
            {
                "candidate_id": cid,
                "name": item.get("name"),
                "strategy_class": item.get("strategy_class"),
                "timeframe": item.get("timeframe"),
                "weight": float(weight),
            }
        )
    return rows


def write_dynamic_allocator_report(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    resolved_input = resolve_incumbent_bundle_path(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name="causal_dynamic_portfolio",
        output_dir=output_dir,
        input_path=resolved_input,
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    status = "completed"
    error: str | None = None
    try:
        memory_guard.sample(
            event="dynamic_start",
            context={
                "requested_input_path": str(Path(input_path).resolve()),
                "resolved_input_path": str(resolved_input),
            },
        )
        rows = _load_candidates(resolved_input)
        memory_guard.checkpoint(
            "dynamic_candidates_loaded",
            {"candidate_count": len(rows)},
        )
        best = search_dynamic_allocator(rows, progress_callback=memory_guard.checkpoint)
        result = dict(best["result"])
    except RSSLimitExceeded as exc:
        status = "aborted_rss_guard"
        error = str(exc)
        raise
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(event="dynamic_finish", context={"status": status, "error": error})
        memory_summary = memory_guard.finalize(
            status=status,
            error=error,
            context={"resolved_input_path": str(resolved_input)},
        )
        memory_guard.release()
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "causal_dynamic_portfolio",
        "schema_version": "1.0",
        "input_path": str(resolved_input),
        "requested_input_path": str(Path(input_path).resolve()),
        "selection_basis": "validation_only_dynamic_search_on_preselected_current_sleeves",
        "objective_profile": "balanced_multi_metric",
        "split_windows": split_windows(),
        "memory_policy": memory_policy_payload(
            budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        ),
        "memory_summary": memory_summary,
        "best_params": dict(best["params"]),
        "validation_objective": float(best["objective"]),
        "split_metrics": dict(result.get("split_metrics") or {}),
        "all_metrics": dict(result.get("all_metrics") or {}),
        "allocation_count": len(list(result.get("allocations") or [])),
        "final_allocation": _top_allocation_rows(result),
        "dates": list(result.get("dates") or []),
        "daily_returns": list(result.get("daily_returns") or []),
        "portfolio_return_streams": _portfolio_return_streams_from_daily(
            list(result.get("dates") or []),
            list(result.get("daily_returns") or []),
        ),
        "allocations": list(result.get("allocations") or []),
        "universe_scope": "preselected_current_incumbent_bundle",
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"causal_dynamic_portfolio_{stamp}.json"
    latest_path = output_dir / "causal_dynamic_portfolio_latest.json"
    md_path = output_dir / f"causal_dynamic_portfolio_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Causal Dynamic Portfolio",
        "",
        f"- input_path: `{payload['input_path']}`",
        f"- selection_basis: `{payload['selection_basis']}`",
        f"- objective_profile: `{payload['objective_profile']}`",
        f"- validation_objective: `{payload['validation_objective']:.6f}`",
        f"- oos_start: `{dict(payload.get('split_windows') or {}).get('oos_start')}`",
        f"- memory_log: `{dict(payload.get('memory_summary') or {}).get('rss_log_path')}`",
        "",
        "## Best params",
        "",
        "```json",
        json.dumps(payload["best_params"], indent=2, sort_keys=True),
        "```",
        "",
        "## Split metrics",
        "",
        f"- train: {json.dumps(payload['split_metrics'].get('train') or {}, sort_keys=True)}",
        f"- val: {json.dumps(payload['split_metrics'].get('val') or {}, sort_keys=True)}",
        f"- oos: {json.dumps(payload['split_metrics'].get('oos') or {}, sort_keys=True)}",
        "",
        "## Final allocation",
        "",
    ]
    for row in payload["final_allocation"]:
        lines.append(
            f"- `{row.get('name')}` | strategy={row.get('strategy_class')} | tf={row.get('timeframe')} | weight={float(row.get('weight', 0.0)):.2%}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "latest_path": str(latest_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def write_dynamic_comparison(
    *,
    dynamic_payload: dict[str, Any],
    comparison_input: Path | None = None,
) -> dict[str, Any]:
    comparison_input = comparison_input or COMPARISON_INPUT
    existing = json.loads(comparison_input.read_text(encoding="utf-8"))
    current_entry = _maybe_current_one_shot_comparison_entry(
        bundle_path=PORTFOLIO_ONE_SHOT_CURRENT_BUNDLE,
        portfolio_path=PORTFOLIO_CURRENT_OPTIMIZATION,
    )
    if current_entry is not None:
        existing["current_one_shot_optimized"] = current_entry
    scope = list(existing.get("comparison_scope") or [])
    if "current_one_shot_optimized" not in scope:
        scope.append("current_one_shot_optimized")
    existing["comparison_scope"] = scope
    dynamic_val = dict((dynamic_payload.get("split_metrics") or {}).get("val") or {})
    dynamic_oos = dict((dynamic_payload.get("split_metrics") or {}).get("oos") or {})
    existing["causal_dynamic_portfolio"] = {
        "path": str(
            (
                comparison_input.parent
                / "portfolio_dynamic_online_current"
                / "causal_dynamic_portfolio_latest.json"
            ).resolve()
        ),
        "val": dynamic_val,
        "oos": dynamic_oos,
        "weights": list(dynamic_payload.get("final_allocation") or []),
        "best_params": dict(dynamic_payload.get("best_params") or {}),
    }
    scope = list(existing.get("comparison_scope") or [])
    if "causal_dynamic_portfolio" not in scope:
        scope.append("causal_dynamic_portfolio")
    existing["comparison_scope"] = scope
    existing["deltas"]["dynamic_vs_current_one_shot_oos_return"] = _safe_float(
        dynamic_oos.get("total_return"), 0.0
    ) - _safe_float(existing["current_one_shot_optimized"]["oos"].get("total_return"), 0.0)
    existing["deltas"]["dynamic_vs_current_one_shot_oos_sharpe"] = _safe_float(
        dynamic_oos.get("sharpe"), 0.0
    ) - _safe_float(existing["current_one_shot_optimized"]["oos"].get("sharpe"), 0.0)
    out_json = comparison_input.parent / "portfolio_dynamic_comparison_latest.json"
    out_md = comparison_input.parent / "portfolio_dynamic_comparison_latest.md"
    out_json.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Dynamic Portfolio Comparison",
        "",
        f"- dynamic_vs_current_one_shot_oos_return: {existing['deltas']['dynamic_vs_current_one_shot_oos_return']:.4%}",
        f"- dynamic_vs_current_one_shot_oos_sharpe: {existing['deltas']['dynamic_vs_current_one_shot_oos_sharpe']:.3f}",
        "",
        "## Dynamic OOS metrics",
        "",
        json.dumps(dynamic_oos, sort_keys=True),
        "",
        "## Dynamic final allocation",
        "",
    ]
    for row in list(dynamic_payload.get("final_allocation") or []):
        lines.append(
            f"- `{row.get('name')}` | strategy={row.get('strategy_class')} | tf={row.get('timeframe')} | weight={float(row.get('weight', 0.0)):.2%}"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json_path": str(out_json.resolve()),
        "md_path": str(out_md.resolve()),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a causal dynamic portfolio allocator over saved sleeve streams."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = write_dynamic_allocator_report(
        input_path=Path(args.input).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
    comparison = write_dynamic_comparison(dynamic_payload=report["payload"])
    print(report["latest_path"])
    print(report["md_path"])
    print(comparison["json_path"])
    print(comparison["md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
