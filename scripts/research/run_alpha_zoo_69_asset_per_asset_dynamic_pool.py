#!/usr/bin/env python3
"""Evaluate a no-fixed-core per-asset tuned dynamic 69-asset monitor.

The source artifact already contains per-symbol/profile Optuna-tuned rows built
from train/validation only. This runner keeps the whole 69-symbol universe in
the candidate pool, rebuilds the tuned row streams, and tests a dynamic selector:

* one tuned row can be active per symbol at each rebalance;
* only trailing realized strategy returns before the rebalance are used;
* the selected active-ready set is capped by top-N and gross/asset caps;
* locked OOS is report-only after the selector parameters are supplied.

It is a research/paper artifact only: real-money flags are always false.
"""

from __future__ import annotations

import argparse
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

from scripts.research import run_alpha_zoo_69_asset_clean_oos_gate as clean_gate  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_profile_optuna_hybrid_refit as profile69  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_walkforward_monitor as walkforward  # noqa: E402

DEFAULT_SOURCE_ARTIFACT = (
    profile69.DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json"
)
DEFAULT_OUTPUT_DIR = broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_per_asset_dynamic_pool_20260531"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "alpha_zoo_69_asset_per_asset_dynamic_pool_latest.json"
DEFAULT_TRAIN_START = clean_gate.DEFAULT_TRAIN_START
DEFAULT_TRAIN_END = clean_gate.DEFAULT_TRAIN_END
DEFAULT_VALIDATION_START = clean_gate.DEFAULT_VALIDATION_START
DEFAULT_VALIDATION_END = clean_gate.DEFAULT_VALIDATION_END
DEFAULT_LOCKED_OOS_START = clean_gate.DEFAULT_LOCKED_OOS_START
DEFAULT_LOCKED_OOS_END = clean_gate.DEFAULT_LOCKED_OOS_END


@dataclass(frozen=True)
class SelectorParams:
    lookback_days: int = 30
    rebalance_days: int = 3
    top_n: int = 10
    target_gross: float = 1.0
    min_trailing_return: float = 0.02
    fit_weight: float = 0.15
    vol_penalty: float = 0.5
    max_symbol_gross: float = 0.35


@dataclass(frozen=True)
class DynamicContext:
    index: pd.DatetimeIndex
    returns: np.ndarray
    positions: np.ndarray
    notionals: np.ndarray
    rows: tuple[dict[str, Any], ...]
    row_symbols: tuple[str, ...]
    universe_symbols: tuple[str, ...]
    fit_scores: np.ndarray
    candidate_pool_policy: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(broad69._json_safe(payload), indent=2, sort_keys=True) + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return default
    return parsed if math.isfinite(parsed) else default


def _parse_timestamp(value: Any) -> pd.Timestamp:
    return clean_gate._parse_timestamp(value)


def _total_return(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.prod(1.0 + arr) - 1.0) if arr.size else 0.0


def _max_drawdown(values: np.ndarray | Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(broad69.max_drawdown(arr)) if arr.size else 0.0


def _periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 365.0 * 24.0
    diffs = pd.Series(index).diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 365.0 * 24.0
    median_seconds = float(diffs.median())
    return 365.0 * 24.0 * 3600.0 / median_seconds if median_seconds > 0.0 else 365.0 * 24.0


def _window_payload(window: tuple[pd.Timestamp, pd.Timestamp], role: str) -> dict[str, Any]:
    return {
        "start": window[0].isoformat() + "Z",
        "end": window[1].isoformat() + "Z",
        "role": role,
        "enabled": True,
    }


def _period_metric(
    *,
    index: pd.DatetimeIndex,
    returns: np.ndarray,
    gross: np.ndarray,
    ready_count: np.ndarray,
    active_signal_count: np.ndarray,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, Any]:
    mask = np.asarray((index >= window[0]) & (index <= window[1]), dtype=bool)
    values = np.asarray(returns, dtype=float)[mask]
    period_index = index[mask]
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    downside = values[values < 0.0]
    down_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    annual = _periods_per_year(period_index)
    total = _total_return(values)
    mdd = _max_drawdown(values)
    return {
        "start": window[0].isoformat() + "Z",
        "end": window[1].isoformat() + "Z",
        "bar_count": int(values.size),
        "total_return": total,
        "mdd": mdd,
        "sharpe": mean / std * math.sqrt(annual) if std > 0.0 else 0.0,
        "sortino": mean / down_std * math.sqrt(annual) if down_std > 0.0 else 0.0,
        "calmar": total / mdd if mdd > 0.0 else 0.0,
        "avg_gross_notional_fraction": float(np.mean(gross[mask])) if mask.any() else 0.0,
        "max_gross_notional_fraction": float(np.max(gross[mask])) if mask.any() else 0.0,
        "avg_ready_symbol_or_row_count": float(np.mean(ready_count[mask])) if mask.any() else 0.0,
        "max_ready_symbol_or_row_count": int(np.max(ready_count[mask])) if mask.any() else 0,
        "avg_active_signal_count": float(np.mean(active_signal_count[mask])) if mask.any() else 0.0,
        "max_active_signal_count": int(np.max(active_signal_count[mask])) if mask.any() else 0,
    }


def row_fit_score(row: Mapping[str, Any]) -> float:
    """Train/validation-only row quality score used as a weak selector prior."""
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_mdd = _safe_float(row.get("train_mdd"))
    validation_mdd = _safe_float(row.get("validation_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    validation_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    validation_spike = max(0.0, validation - train)
    return float(
        2.0 * min(train, validation)
        + 0.5 * validation
        + min(train_rpt, validation_rpt, 200.0) / 500.0
        - 2.0 * validation_mdd
        - 0.3 * train_mdd
        - 1.2 * validation_spike
    )


def _candidate_pool_policy(
    *, source: Mapping[str, Any], stream_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    universe_symbols = tuple(
        str(symbol) for symbol in dict(source.get("universe") or {}).get("symbols") or ()
    )
    stream_symbols = sorted({str(row.get("symbol")) for row in stream_rows})
    train_eligibility = dict(source.get("train_eligibility") or {})
    train_ineligible = tuple(
        str(symbol) for symbol in train_eligibility.get("train_ineligible_symbols") or ()
    )
    return {
        "candidate_pool_symbol_count": len(universe_symbols),
        "candidate_pool_symbols": list(universe_symbols),
        "all_universe_symbols_remain_candidates": True,
        "candidate_pool_is_not_equal_to_current_positions": True,
        "current_stream_capable_symbol_count": len(stream_symbols),
        "current_stream_capable_symbols": stream_symbols,
        "train_ineligible_future_candidate_symbol_count": len(train_ineligible),
        "train_ineligible_future_candidate_symbols": list(train_ineligible),
        "active_set_contract": (
            "No fixed core is held. The selector ranks already tuned per-asset rows with "
            "only trailing information before each rebalance; symbols outside the stream-capable "
            "set remain monitor-only future candidates until later data refresh/refit."
        ),
    }


def build_dynamic_context(source_artifact: Path, windows: clean_gate.GateWindows) -> DynamicContext:
    source = json.loads(source_artifact.read_text(encoding="utf-8"))
    raw_rows = [
        dict(row)
        for row in source.get("asset_tuning_rows") or source.get("selected_sleeve_rows") or []
    ]
    if not raw_rows:
        raise ValueError("source artifact has no per-asset tuned rows")
    universe_symbols = tuple(
        str(symbol) for symbol in dict(source.get("universe") or {}).get("symbols") or ()
    )
    timeframes = tuple(str(timeframe) for timeframe in source.get("timeframes") or ())
    data_root = Path(
        dict(source.get("data_coverage") or {}).get("data_root") or broad69.DEFAULT_DATA_ROOT
    )
    bars, _coverage = broad69.load_all_bars(
        universe_symbols, data_root=data_root, timeframes=timeframes
    )
    split_windows = broad69.SplitWindows(train=windows.train, validation=windows.validation)
    cache = profile69.FeatureCache(
        bars_by_symbol_tf=bars,
        symbols=universe_symbols,
        timeframes=timeframes,
        _xsmom={},
        _anchor_returns={},
    )

    rebuilt_rows: list[dict[str, Any]] = []
    returns: list[pd.Series] = []
    positions: list[pd.Series] = []
    for index, raw in enumerate(raw_rows):
        row = dict(raw)
        params = dict(row.get("optuna_params") or {})
        if not params:
            for key in (
                "family",
                "timeframe",
                "side",
                "integer_leverage",
                "min_hold_bars",
                "cooldown_bars",
                "lookback_bars",
                "threshold",
                "exit_threshold",
            ):
                params[key] = row[key]
        stream = profile69._candidate_from_params(
            symbol=str(row["symbol"]),
            profile_id=f"per_asset_dynamic_source_row_{index}",
            params=params,
            cache=cache,
            windows=split_windows,
            allocation_fraction=float(row.get("allocation_fraction") or 0.10),
        )
        row["source_row_index"] = index
        row["source_profile_id"] = str(raw.get("profile_id"))
        row["profile_id"] = stream.row["profile_id"]
        row["fit_score"] = row_fit_score(row)
        rebuilt_rows.append(row)
        returns.append(stream.returns.sort_index())
        positions.append(stream.position.sort_index())

    aligned_index = pd.DatetimeIndex(
        sorted(set().union(*(set(series.index) for series in returns)))
    )
    returns_matrix = np.vstack(
        [series.reindex(aligned_index, fill_value=0.0).to_numpy(dtype=float) for series in returns]
    )
    position_matrix = np.vstack(
        [
            series.reindex(aligned_index, fill_value=0.0).to_numpy(dtype=float)
            for series in positions
        ]
    )
    notionals = np.asarray([_safe_float(row.get("notional_fraction")) for row in rebuilt_rows])
    return DynamicContext(
        index=aligned_index,
        returns=returns_matrix,
        positions=position_matrix,
        notionals=notionals,
        rows=tuple(rebuilt_rows),
        row_symbols=tuple(str(row.get("symbol")) for row in rebuilt_rows),
        universe_symbols=universe_symbols,
        fit_scores=np.asarray([_safe_float(row.get("fit_score")) for row in rebuilt_rows]),
        candidate_pool_policy=_candidate_pool_policy(source=source, stream_rows=rebuilt_rows),
    )


def _window_return_from_cumprod(cumprod: np.ndarray, start: int, end: int) -> np.ndarray:
    if end <= start:
        return np.zeros(cumprod.shape[0], dtype=float)
    base = cumprod[:, start - 1] if start > 0 else 1.0
    return cumprod[:, end - 1] / base - 1.0


def _window_std_from_cumsums(
    cumsum: np.ndarray, cumsum_square: np.ndarray, start: int, end: int
) -> np.ndarray:
    count = max(int(end - start), 1)
    total = cumsum[:, end - 1] - (cumsum[:, start - 1] if start > 0 else 0.0)
    total_square = cumsum_square[:, end - 1] - (cumsum_square[:, start - 1] if start > 0 else 0.0)
    variance = np.maximum(total_square / count - (total / count) ** 2, 0.0)
    return np.sqrt(variance)


def dynamic_weights(
    context: DynamicContext, params: SelectorParams
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if params.lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if params.rebalance_days <= 0:
        raise ValueError("rebalance_days must be positive")
    if params.top_n <= 0:
        raise ValueError("top_n must be positive")
    if params.target_gross <= 0.0:
        raise ValueError("target_gross must be positive")
    if params.max_symbol_gross <= 0.0:
        raise ValueError("max_symbol_gross must be positive")

    index = context.index
    returns = context.returns
    cumprod = np.cumprod(1.0 + returns, axis=1)
    cumsum = np.cumsum(returns, axis=1)
    cumsum_square = np.cumsum(returns * returns, axis=1)
    by_symbol: dict[str, list[int]] = defaultdict(list)
    for row_index, symbol in enumerate(context.row_symbols):
        by_symbol[str(symbol)].append(row_index)

    weights = np.zeros_like(returns, dtype=np.float32)
    selection_log: list[dict[str, Any]] = []
    start = index[0].normalize() + pd.Timedelta(days=params.lookback_days)
    for rebalance_time in pd.date_range(
        start=start, end=index[-1], freq=f"{params.rebalance_days}D"
    ):
        loc = int(np.searchsorted(index.values, np.datetime64(rebalance_time), side="left"))
        if loc >= len(index):
            break
        loc_next = int(
            np.searchsorted(
                index.values,
                np.datetime64(rebalance_time + pd.Timedelta(days=params.rebalance_days)),
                side="left",
            )
        )
        lookback_start = int(
            np.searchsorted(
                index.values,
                np.datetime64(rebalance_time - pd.Timedelta(days=params.lookback_days)),
                side="left",
            )
        )
        trailing_return = _window_return_from_cumprod(cumprod, lookback_start, loc)
        trailing_std = _window_std_from_cumsums(cumsum, cumsum_square, lookback_start, loc)
        scaled_vol = trailing_std * math.sqrt(max(loc - lookback_start, 1))
        score = (
            trailing_return
            + params.fit_weight * context.fit_scores
            - params.vol_penalty * scaled_vol
        )

        candidates: list[tuple[float, int]] = []
        for symbol, row_indices in by_symbol.items():
            eligible = [
                row_index
                for row_index in row_indices
                if trailing_return[row_index] >= params.min_trailing_return
            ]
            if not eligible:
                continue
            selected = max(eligible, key=lambda row_index: score[row_index])
            if score[selected] <= 0.0 and trailing_return[selected] <= 0.0:
                continue
            candidates.append((float(score[selected]), selected))
        active = [
            row_index for _score, row_index in sorted(candidates, reverse=True)[: params.top_n]
        ]
        if not active:
            continue
        current = np.zeros(returns.shape[0], dtype=float)
        gross_each = params.target_gross / len(active)
        for row_index in active:
            notional = float(context.notionals[row_index])
            if notional <= 0.0:
                continue
            allocation = min(gross_each, params.max_symbol_gross)
            current[row_index] = allocation / notional
        weights[:, loc:loc_next] = current[:, None]
        selection_log.append(
            {
                "rebalance_time": rebalance_time.isoformat() + "Z",
                "active_row_count": len(active),
                "active_symbols": [context.row_symbols[row_index] for row_index in active],
                "active_source_row_indices": [
                    int(context.rows[row_index].get("source_row_index") or 0)
                    for row_index in active
                ],
            }
        )
    return weights, selection_log


def evaluate_dynamic_selector(
    context: DynamicContext,
    params: SelectorParams,
    windows: clean_gate.GateWindows,
    folds: Sequence[walkforward.WalkForwardFold] = walkforward.DEFAULT_FOLDS,
) -> dict[str, Any]:
    weights, selection_log = dynamic_weights(context, params)
    portfolio_returns = np.sum(context.returns * weights, axis=0)
    gross = np.sum(np.abs(weights) * context.notionals[:, None], axis=0)
    ready_count = np.sum(np.abs(weights) > 1e-12, axis=0)
    active_signal_count = np.sum(
        (np.abs(weights) > 1e-12) & (np.abs(context.positions) > 1e-12), axis=0
    )

    clean_metrics = {
        "train": _period_metric(
            index=context.index,
            returns=portfolio_returns,
            gross=gross,
            ready_count=ready_count,
            active_signal_count=active_signal_count,
            window=windows.train,
        ),
        "validation": _period_metric(
            index=context.index,
            returns=portfolio_returns,
            gross=gross,
            ready_count=ready_count,
            active_signal_count=active_signal_count,
            window=windows.validation,
        ),
        "locked_oos": _period_metric(
            index=context.index,
            returns=portfolio_returns,
            gross=gross,
            ready_count=ready_count,
            active_signal_count=active_signal_count,
            window=windows.locked_oos,
        ),
    }

    fold_payloads: list[dict[str, Any]] = []
    validation_returns: list[float] = []
    oos_returns: list[float] = []
    validation_mdds: list[float] = []
    oos_mdds: list[float] = []
    for fold in folds:
        validation = _period_metric(
            index=context.index,
            returns=portfolio_returns,
            gross=gross,
            ready_count=ready_count,
            active_signal_count=active_signal_count,
            window=fold.validation,
        )
        locked_oos = _period_metric(
            index=context.index,
            returns=portfolio_returns,
            gross=gross,
            ready_count=ready_count,
            active_signal_count=active_signal_count,
            window=fold.locked_oos,
        )
        validation_returns.append(float(validation["total_return"]))
        oos_returns.append(float(locked_oos["total_return"]))
        validation_mdds.append(float(validation["mdd"]))
        oos_mdds.append(float(locked_oos["mdd"]))
        fold_payloads.append(
            {
                "fold_id": fold.fold_id,
                "train": _period_metric(
                    index=context.index,
                    returns=portfolio_returns,
                    gross=gross,
                    ready_count=ready_count,
                    active_signal_count=active_signal_count,
                    window=fold.train,
                ),
                "validation": validation,
                "locked_oos": locked_oos,
            }
        )

    row_attribution: list[dict[str, Any]] = []
    for row_index, row in enumerate(context.rows):
        row_weights = weights[row_index]
        if not np.any(np.abs(row_weights) > 1e-12):
            continue
        contribution = context.returns[row_index] * row_weights
        row_attribution.append(
            {
                "source_row_index": int(row.get("source_row_index") or 0),
                "symbol": row.get("symbol"),
                "source_profile_id": row.get("source_profile_id"),
                "timeframe": row.get("timeframe"),
                "side": row.get("side"),
                "family": row.get("family"),
                "fit_score": row.get("fit_score"),
                "ready_bar_count": int(np.count_nonzero(np.abs(row_weights) > 1e-12)),
                "avg_gross_notional_fraction": float(
                    np.mean(np.abs(row_weights) * float(context.notionals[row_index]))
                ),
                "total_contribution_return": _total_return(contribution),
            }
        )
    row_attribution.sort(
        key=lambda item: float(item.get("avg_gross_notional_fraction") or 0.0), reverse=True
    )

    all_validation_positive = bool(validation_returns) and all(
        value > 0.0 for value in validation_returns
    )
    all_oos_positive = bool(oos_returns) and all(value > 0.0 for value in oos_returns)
    return {
        "selector_params": asdict(params),
        "clean_metrics": clean_metrics,
        "walkforward_folds": fold_payloads,
        "walkforward_summary": {
            "fold_count": len(fold_payloads),
            "min_validation_return": min(validation_returns) if validation_returns else 0.0,
            "min_oos_return": min(oos_returns) if oos_returns else 0.0,
            "max_validation_mdd": max(validation_mdds) if validation_mdds else 0.0,
            "max_oos_mdd": max(oos_mdds) if oos_mdds else 0.0,
            "all_validation_positive": all_validation_positive,
            "all_oos_positive": all_oos_positive,
            "all_validation_and_oos_positive": all_validation_positive and all_oos_positive,
        },
        "activity_summary": {
            "avg_gross_notional_fraction": float(np.mean(gross)) if gross.size else 0.0,
            "max_gross_notional_fraction": float(np.max(gross)) if gross.size else 0.0,
            "avg_ready_row_count": float(np.mean(ready_count)) if ready_count.size else 0.0,
            "max_ready_row_count": int(np.max(ready_count)) if ready_count.size else 0,
            "avg_active_signal_count": float(np.mean(active_signal_count))
            if active_signal_count.size
            else 0.0,
            "max_active_signal_count": int(np.max(active_signal_count))
            if active_signal_count.size
            else 0,
        },
        "row_attribution": row_attribution,
        "selection_log_tail": selection_log[-20:],
    }


def _coerce_windows(args: argparse.Namespace) -> clean_gate.GateWindows:
    return clean_gate.GateWindows(
        train=(_parse_timestamp(args.train_start), _parse_timestamp(args.train_end)),
        validation=(_parse_timestamp(args.validation_start), _parse_timestamp(args.validation_end)),
        locked_oos=(_parse_timestamp(args.locked_oos_start), _parse_timestamp(args.locked_oos_end)),
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_artifact = Path(args.source_artifact).expanduser().resolve()
    windows = _coerce_windows(args)
    context = build_dynamic_context(source_artifact, windows)
    params = SelectorParams(
        lookback_days=int(args.lookback_days),
        rebalance_days=int(args.rebalance_days),
        top_n=int(args.top_n),
        target_gross=float(args.target_gross),
        min_trailing_return=float(args.min_trailing_return),
        fit_weight=float(args.fit_weight),
        vol_penalty=float(args.vol_penalty),
        max_symbol_gross=float(args.max_symbol_gross),
    )
    evaluation = evaluate_dynamic_selector(context, params, windows)
    wf_summary = dict(evaluation.get("walkforward_summary") or {})
    clean = dict(evaluation.get("clean_metrics") or {})
    clean_oos_pass = float(
        dict(clean.get("locked_oos") or {}).get("total_return") or 0.0
    ) > 0.0 and float(dict(clean.get("locked_oos") or {}).get("mdd") or 0.0) <= float(
        args.max_oos_mdd
    )
    promotion_reasons: list[str] = []
    if not clean_oos_pass:
        promotion_reasons.append("clean_locked_oos_return_or_mdd_gate_failed")
    if not bool(wf_summary.get("all_validation_and_oos_positive")):
        promotion_reasons.append("walkforward_validation_or_oos_has_negative_fold")
    if float(wf_summary.get("max_oos_mdd") or 0.0) > float(args.max_oos_mdd):
        promotion_reasons.append("walkforward_oos_mdd_above_limit")
    return {
        "artifact_kind": "alpha_zoo_69_asset_per_asset_dynamic_pool",
        "generated_at_utc": _utc_now_iso(),
        "source_artifact": str(source_artifact),
        "evaluation_policy": {
            "candidate_pool": "all_69_symbols_monitor_candidate_pool",
            "per_asset_tuned_rows_source": "asset_tuning_rows_from_train_validation_refit_artifact",
            "fixed_core_required": False,
            "one_active_tuned_row_per_symbol_per_rebalance": True,
            "rebalance_uses_only_trailing_returns_before_rebalance": True,
            "locked_oos_used_for_parameter_fitting": False,
            "locked_oos_used_for_report": True,
            "selector_params_supplied_externally_or_by_researcher": True,
            "real_money_execution": False,
        },
        "split_manifest": {
            "train": _window_payload(windows.train, "parameter_fitting_and_objective_training"),
            "validation": _window_payload(windows.validation, "holdout_selection_and_report"),
            "locked_oos": _window_payload(
                windows.locked_oos, "gate_report_only_after_train_validation_freeze"
            ),
        },
        "candidate_pool_policy": context.candidate_pool_policy,
        **evaluation,
        "clean_locked_oos_positive_mdd_gate_pass": clean_oos_pass,
        "ready_for_paper": bool(clean_oos_pass and not promotion_reasons),
        "ready_for_real": False,
        "paper_testnet_only": True,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "promotion_recommendation": "paper_shadow_only" if promotion_reasons else "paper_candidate",
        "promotion_blockers": promotion_reasons,
        "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _pct(value: Any) -> str:
    return f"{float(value):+.4%}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    clean = dict(payload.get("clean_metrics") or {})
    wf_summary = dict(payload.get("walkforward_summary") or {})
    params = dict(payload.get("selector_params") or {})
    pool = dict(payload.get("candidate_pool_policy") or {})
    lines = [
        "# 69-asset per-asset tuned dynamic pool evaluation",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        f"Source: `{payload.get('source_artifact')}`",
        f"Candidate pool / stream-capable: `{pool.get('candidate_pool_symbol_count')}` / `{pool.get('current_stream_capable_symbol_count')}`",
        f"Promotion: `{payload.get('promotion_recommendation')}`",
        f"Blockers: `{payload.get('promotion_blockers')}`",
        "",
        "## Selector params",
        "",
    ]
    for key, value in params.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Clean split metrics",
            "",
            "| split | return | MDD | Sharpe | avg gross | avg ready | avg signal |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "locked_oos"):
        row = dict(clean.get(split) or {})
        lines.append(
            f"| `{split}` | {_pct(row.get('total_return') or 0.0)} | {_pct(row.get('mdd') or 0.0)} | "
            f"{float(row.get('sharpe') or 0.0):.3f} | {float(row.get('avg_gross_notional_fraction') or 0.0):.3f} | "
            f"{float(row.get('avg_ready_symbol_or_row_count') or 0.0):.2f} | {float(row.get('avg_active_signal_count') or 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Walk-forward summary",
            "",
            f"- min validation return: `{_pct(wf_summary.get('min_validation_return') or 0.0)}`",
            f"- min OOS return: `{_pct(wf_summary.get('min_oos_return') or 0.0)}`",
            f"- max validation MDD: `{_pct(wf_summary.get('max_validation_mdd') or 0.0)}`",
            f"- max OOS MDD: `{_pct(wf_summary.get('max_oos_mdd') or 0.0)}`",
            f"- all validation/OOS positive: `{wf_summary.get('all_validation_and_oos_positive')}`",
            "",
            "| fold | validation | OOS | val MDD | OOS MDD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold in payload.get("walkforward_folds") or []:
        val = dict(dict(fold).get("validation") or {})
        oos = dict(dict(fold).get("locked_oos") or {})
        lines.append(
            f"| `{dict(fold).get('fold_id')}` | {_pct(val.get('total_return') or 0.0)} | {_pct(oos.get('total_return') or 0.0)} | "
            f"{_pct(val.get('mdd') or 0.0)} | {_pct(oos.get('mdd') or 0.0)} |"
        )
    lines.extend(
        [
            "",
            "## Top ready-row attribution",
            "",
            "| row | symbol | source profile | tf | side | avg gross | contribution |",
            "|---:|---|---|---:|---|---:|---:|",
        ]
    )
    for row in list(payload.get("row_attribution") or [])[:12]:
        lines.append(
            f"| {int(dict(row).get('source_row_index') or 0)} | `{dict(row).get('symbol')}` | `{dict(row).get('source_profile_id')}` | "
            f"`{dict(row).get('timeframe')}` | `{dict(row).get('side')}` | "
            f"{float(dict(row).get('avg_gross_notional_fraction') or 0.0):.4f} | {_pct(dict(row).get('total_contribution_return') or 0.0)} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--locked-oos-start", default=DEFAULT_LOCKED_OOS_START)
    parser.add_argument("--locked-oos-end", default=DEFAULT_LOCKED_OOS_END)
    parser.add_argument("--max-oos-mdd", type=float, default=clean_gate.DEFAULT_MAX_OOS_MDD)
    parser.add_argument("--lookback-days", type=int, default=SelectorParams.lookback_days)
    parser.add_argument("--rebalance-days", type=int, default=SelectorParams.rebalance_days)
    parser.add_argument("--top-n", type=int, default=SelectorParams.top_n)
    parser.add_argument("--target-gross", type=float, default=SelectorParams.target_gross)
    parser.add_argument(
        "--min-trailing-return", type=float, default=SelectorParams.min_trailing_return
    )
    parser.add_argument("--fit-weight", type=float, default=SelectorParams.fit_weight)
    parser.add_argument("--vol-penalty", type=float, default=SelectorParams.vol_penalty)
    parser.add_argument("--max-symbol-gross", type=float, default=SelectorParams.max_symbol_gross)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    output = Path(args.output).expanduser().resolve()
    _write_json(output, payload)
    output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
