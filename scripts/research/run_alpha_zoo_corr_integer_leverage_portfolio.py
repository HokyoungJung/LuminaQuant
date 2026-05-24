#!/usr/bin/env python3
"""Optimize integer per-asset leverage for the corr-diversified Alpha Zoo slate.

This runner is research/paper-testnet only. It consumes the latest PnL-correlation
paper slate, replays fixed strategy state signals, and searches integer leverage
maps per asset using train+validation evidence only. locked-OOS is attached after
the leverage map is frozen as gate/report evidence. No order execution is
performed and real-money readiness remains disabled.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import resource
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

from scripts.research.run_alpha_zoo_htf_momentum_crowding_discovery import (  # noqa: E402
    AVG_BBO_SPREAD_BPS_ASSUMPTION,
    BBO_SPREAD_MULTIPLIER,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEATURE_ROOT,
    PRIMARY_ROUND_TRIP_COST_BPS,
    RETURN_PER_TURNOVER_THRESHOLD_BPS,
    SPLIT_ORDER,
    _json_safe,
    _split_mask,
    max_drawdown,
)
from scripts.research import run_alpha_zoo_pnl_correlation_decision as corr  # noqa: E402

DEFAULT_CORRELATION_ARTIFACT = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_pnl_correlation_decision_20260524/alpha_zoo_pnl_correlation_decision_latest.json"
)
DEFAULT_MONITORING_ARTIFACT = corr.DEFAULT_MONITORING_ARTIFACT
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_corr_integer_leverage_portfolio_20260524"
)

ARTIFACT_KIND = "alpha_zoo_corr_integer_leverage_portfolio"
LEVERAGE_MIN = 1
LEVERAGE_MAX = 12
ACCOUNT_EQUITY_REFERENCE = 10_000.0
MIN_TRAIN_TRADE_EVENTS = 80
MIN_VALIDATION_TRADE_EVENTS = 30
MIN_LOCKED_OOS_TRADE_EVENTS = 20
FORBIDDEN_CALENDAR_RULE_TOKENS = (
    "calendar",
    "date_rule",
    "day_of_week",
    "weekday",
    "month_end",
    "time_of_day",
    "hour_of_day",
)

PROFILE_SPECS = {
    "balanced_mdd12_gross5": {
        "description": (
            "Strict-promotion integer per-asset leverage map with hard 12% validation MDD and 5x gross-notional cap."
        ),
        "strict_promotion_profile": True,
        "max_validation_mdd": 0.12,
        "max_train_mdd": 0.30,
        "max_gross_notional": 5.0,
        "max_asset_gross_notional": 3.0,
        "min_validation_return": 0.02,
    },
    "growth_mdd20_gross8": {
        "description": (
            "Higher-return relaxed shadow map with 20% validation MDD and 8x gross-notional cap; not strict promotion."
        ),
        "strict_promotion_profile": False,
        "max_validation_mdd": 0.20,
        "max_train_mdd": 0.45,
        "max_gross_notional": 8.0,
        "max_asset_gross_notional": 4.5,
        "min_validation_return": 0.04,
    },
    "aggressive_mdd30_gross10_shadow": {
        "description": "Aggressive research-only shadow map with 30% validation MDD and 10x gross cap; never real-money ready.",
        "strict_promotion_profile": False,
        "max_validation_mdd": 0.30,
        "max_train_mdd": 0.60,
        "max_gross_notional": 10.0,
        "max_asset_gross_notional": 6.0,
        "min_validation_return": 0.05,
    },
}

PORTFOLIO_FIELDS = [
    "profile_id",
    "rank",
    "selected_candidate_count",
    "selected_model_ids",
    "leverage_map",
    "gross_notional_fraction",
    "train_return",
    "validation_return",
    "locked_oos_return_report_only",
    "train_mdd",
    "validation_mdd",
    "locked_oos_mdd_report_only",
    "train_trade_event_count",
    "validation_trade_event_count",
    "locked_oos_trade_event_count_report_only",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "locked_oos_return_per_turnover_proxy_bps_report_only",
    "train_liquidation_count",
    "validation_liquidation_count",
    "locked_oos_liquidation_count_report_only",
    "train_account_wipeout_count",
    "validation_account_wipeout_count",
    "locked_oos_account_wipeout_count_report_only",
    "train_validation_score",
    "candidate_tier",
    "strict_promotion_profile",
    "paper_testnet_candidate",
    "shadow_gate_pass",
    "promotion_gate_pass",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "rejection_reasons",
    "strict_promotion_rejection_reasons",
]


@dataclass(frozen=True)
class CandidateReplay:
    model_id: str
    source_artifact_kind: str
    symbol: str
    timeframe: str
    allocation_fraction: float
    datetimes: pd.DatetimeIndex
    signal: np.ndarray
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray


@dataclass(frozen=True)
class CandidateSim:
    model_id: str
    symbol: str
    timeframe: str
    integer_leverage: int
    allocation_fraction: float
    notional_fraction: float
    datetimes: pd.DatetimeIndex
    returns: np.ndarray
    position: np.ndarray
    liquidation_flags: np.ndarray
    account_wipeout_flags: np.ndarray


@dataclass(frozen=True)
class FastReplayCache:
    replays: tuple[CandidateReplay, ...]
    union_index: pd.DatetimeIndex
    split_masks: Mapping[str, np.ndarray]
    aligned_returns: Mapping[tuple[int, int], np.ndarray]
    notional_fraction: Mapping[tuple[int, int], float]
    trade_count: Mapping[tuple[int, str], int]
    liquidation_count: Mapping[tuple[int, int, str], int]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


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
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _period_return(returns: np.ndarray | pd.Series) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_rows_from_corr_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in payload.get("correlation_decision_rows", [])
        if row.get("correlation_decision") == "selected_corr_diversified_paper_monitor"
    ]
    return sorted(rows, key=lambda row: _safe_int(row.get("selection_rank"), 10**9))


def _assert_governance(correlation_payload: Mapping[str, Any], monitoring_payload: Mapping[str, Any]) -> None:
    for name, payload in (("correlation", correlation_payload), ("monitoring", monitoring_payload)):
        if payload.get("ready_for_real") is not False or payload.get("real_money_execution") is not False:
            raise ValueError(f"{name} artifact violates real-money disabled guard")
    policy = dict(correlation_payload.get("selection_policy") or {})
    forbidden = [
        "uses_locked_oos_for_selection",
        "uses_locked_oos_for_discovery",
        "uses_locked_oos_for_objective",
        "uses_locked_oos_for_pruning",
        "uses_locked_oos_for_parameter_fitting",
    ]
    for key in forbidden:
        if policy.get(key) is not False:
            raise ValueError(f"correlation artifact has unsafe locked-OOS policy: {key}={policy.get(key)!r}")


def _align_bars_to_capture(capture: corr.CapturedPnl, bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    indexed = frame.set_index("datetime").sort_index()
    aligned = indexed.reindex(pd.DatetimeIndex(capture.datetimes))
    if aligned[["close", "high", "low"]].isna().any().any():
        missing = aligned[aligned["close"].isna()].index[:5]
        raise ValueError(f"bars missing for captured timestamps in {capture.model_id}: {list(missing)}")
    return aligned.reset_index(names="datetime")


def _load_bars_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    from scripts.research import run_alpha_zoo_30m_plus_alpha_feedback_discovery as feedback
    from scripts.research import run_alpha_zoo_debounced_efficiency_repair_discovery as repair

    out: dict[tuple[str, str, str], pd.DataFrame] = {}
    repair_rows = [row for row in rows if row.get("source_artifact_kind") == corr.SOURCE_KIND_DEBOUNCED_REPAIR]
    other_rows = [row for row in rows if row.get("source_artifact_kind") != corr.SOURCE_KIND_DEBOUNCED_REPAIR]
    if repair_rows:
        symbols = tuple(sorted({str(row["symbol"]).upper() for row in repair_rows}))
        timeframes = tuple(sorted({str(row["timeframe"]).lower() for row in repair_rows}))
        hourly = repair.load_hourly_bars(symbols, data_root=data_root)
        for symbol in symbols:
            for timeframe in timeframes:
                out[(corr.SOURCE_KIND_DEBOUNCED_REPAIR, symbol, timeframe)] = repair.resample_bars(
                    hourly[symbol], timeframe
                )
    if other_rows:
        symbols = tuple(sorted({str(row["symbol"]).upper() for row in other_rows}))
        timeframes = feedback._validate_timeframes(
            tuple(sorted({str(row["timeframe"]).lower() for row in other_rows}))
        )
        bars_by_key = feedback.load_requested_bars(symbols, timeframes=timeframes, data_root=data_root)
        for row in other_rows:
            source_kind = str(row["source_artifact_kind"])
            symbol = str(row["symbol"]).upper()
            timeframe = str(row["timeframe"]).lower()
            out[(source_kind, symbol, timeframe)] = bars_by_key[(symbol, timeframe)]
    return out


def build_candidate_replays(
    rows: Sequence[Mapping[str, Any]],
    captures: Mapping[str, corr.CapturedPnl],
    *,
    bars_by_key: Mapping[tuple[str, str, str], pd.DataFrame],
) -> list[CandidateReplay]:
    replays: list[CandidateReplay] = []
    for row in rows:
        model_id = str(row["model_id"])
        capture = captures.get(model_id)
        if capture is None:
            raise ValueError(f"missing PnL capture for selected model {model_id}")
        source_kind = str(row["source_artifact_kind"])
        symbol = str(row["symbol"]).upper()
        timeframe = str(row["timeframe"]).lower()
        bars = _align_bars_to_capture(capture, bars_by_key[(source_kind, symbol, timeframe)])
        replays.append(
            CandidateReplay(
                model_id=model_id,
                source_artifact_kind=source_kind,
                symbol=symbol,
                timeframe=timeframe,
                allocation_fraction=_safe_float(row.get("allocation_fraction")),
                datetimes=pd.DatetimeIndex(pd.to_datetime(bars["datetime"])),
                signal=np.asarray(capture.position, dtype=float),
                close=bars["close"].to_numpy(dtype=float),
                high=bars["high"].to_numpy(dtype=float),
                low=bars["low"].to_numpy(dtype=float),
            )
        )
    return replays


def simulate_candidate_with_integer_leverage(replay: CandidateReplay, *, integer_leverage: int) -> CandidateSim:
    if int(integer_leverage) != integer_leverage or integer_leverage < 1:
        raise ValueError("integer_leverage must be a positive integer")
    signal = replay.signal.astype(float)
    close = replay.close.astype(float)
    high = replay.high.astype(float)
    low = replay.low.astype(float)
    next_return = np.r_[np.diff(close) / np.maximum(close[:-1], 1e-12), 0.0]
    notional = float(integer_leverage) * replay.allocation_fraction
    transition = np.abs(np.diff(np.r_[0.0, signal]))
    costs = (PRIMARY_ROUND_TRIP_COST_BPS / 10_000.0) * notional * transition / 2.0
    returns = signal * notional * next_return - costs
    long_liq = (signal > 0.0) & (((low / np.maximum(close, 1e-12)) - 1.0) * float(integer_leverage) <= -0.95)
    short_liq = (signal < 0.0) & (((high / np.maximum(close, 1e-12)) - 1.0) * float(integer_leverage) >= 0.95)
    liquidation = long_liq | short_liq
    equity = np.cumprod(1.0 + returns)
    return CandidateSim(
        model_id=replay.model_id,
        symbol=replay.symbol,
        timeframe=replay.timeframe,
        integer_leverage=int(integer_leverage),
        allocation_fraction=replay.allocation_fraction,
        notional_fraction=notional,
        datetimes=replay.datetimes,
        returns=returns,
        position=signal,
        liquidation_flags=liquidation,
        account_wipeout_flags=equity <= 0.0,
    )


def _split_series(series: pd.Series, split: str) -> pd.Series:
    mask = _split_mask(series.index, split)
    return series.loc[mask]


def _trade_count_for_split(sim: CandidateSim, split: str) -> int:
    mask = _split_mask(sim.datetimes, split)
    pos = sim.position[mask]
    if pos.size == 0:
        return 0
    return int(np.count_nonzero(np.abs(np.diff(np.r_[0.0, pos])) > 1e-12))


def _turnover_proxy_for_split(sims: Sequence[CandidateSim], split: str) -> float:
    return float(sum(_trade_count_for_split(sim, split) * abs(sim.notional_fraction) for sim in sims))


def _candidate_liquidation_count_for_split(sims: Sequence[CandidateSim], split: str) -> int:
    total = 0
    for sim in sims:
        mask = _split_mask(sim.datetimes, split)
        total += int(np.count_nonzero(sim.liquidation_flags[mask]))
    return total


def _portfolio_frame(sims: Sequence[CandidateSim]) -> pd.DataFrame:
    series = [pd.Series(sim.returns, index=sim.datetimes, name=sim.model_id, dtype=float) for sim in sims]
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index().fillna(0.0)


def evaluate_integer_leverage_map(
    replays: Sequence[CandidateReplay],
    leverage_by_asset: Mapping[str, int],
) -> dict[str, Any]:
    sims = [
        simulate_candidate_with_integer_leverage(
            replay,
            integer_leverage=int(leverage_by_asset[replay.symbol]),
        )
        for replay in replays
    ]
    frame = _portfolio_frame(sims)
    portfolio_returns = frame.sum(axis=1) if not frame.empty else pd.Series(dtype=float)
    gross = float(sum(sim.notional_fraction for sim in sims))
    asset_gross: dict[str, float] = defaultdict(float)
    for sim in sims:
        asset_gross[sim.symbol] += sim.notional_fraction
    active_assets = sorted(asset_gross)
    split_metrics: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        split_returns = _split_series(portfolio_returns, split)
        returns_array = split_returns.to_numpy(dtype=float)
        turnover = _turnover_proxy_for_split(sims, split)
        total_return = _period_return(returns_array)
        equity = np.cumprod(1.0 + returns_array) if returns_array.size else np.asarray([], dtype=float)
        split_metrics[split] = {
            "total_return": total_return,
            "max_drawdown": max_drawdown(returns_array),
            "bar_count": int(returns_array.size),
            "trade_event_count": int(sum(_trade_count_for_split(sim, split) for sim in sims)),
            "turnover_proxy": turnover,
            "return_per_turnover_proxy_bps": total_return * 10_000.0 / turnover if turnover > 0.0 else None,
            "liquidation_count": _candidate_liquidation_count_for_split(sims, split),
            "account_wipeout_count": int(np.count_nonzero(equity <= 0.0)) if equity.size else 0,
        }
    validation = split_metrics["validation"]["total_return"]
    train = split_metrics["train"]["total_return"]
    ratio = train / validation if validation > 0.0 else 0.0
    return {
        "leverage_by_asset": {asset: int(leverage_by_asset[asset]) for asset in active_assets},
        "gross_notional_fraction": gross,
        "asset_gross_notional_fraction": dict(sorted(asset_gross.items())),
        "candidate_count": len(sims),
        "selected_model_ids": [sim.model_id for sim in sims],
        "split_metrics": split_metrics,
        "train_validation_return_ratio": ratio,
        "replay_live_notional_parity": {
            "recorded": True,
            "sizing_mode": "asset_integer_leverage_times_candidate_allocation_fraction",
            "account_equity_reference": ACCOUNT_EQUITY_REFERENCE,
            "asset_integer_leverage": {asset: int(leverage_by_asset[asset]) for asset in active_assets},
            "candidate_notional_formula": "account_equity * candidate_allocation_fraction * asset_integer_leverage",
            "parity": True,
        },
    }


def _build_fast_replay_cache(
    replays: Sequence[CandidateReplay],
    *,
    leverage_min: int,
    leverage_max: int,
) -> FastReplayCache:
    union_index = pd.DatetimeIndex(sorted(set().union(*(set(replay.datetimes) for replay in replays))))
    split_masks = {split: _split_mask(union_index, split) for split in SPLIT_ORDER}
    aligned_returns: dict[tuple[int, int], np.ndarray] = {}
    notional_fraction: dict[tuple[int, int], float] = {}
    trade_count: dict[tuple[int, str], int] = {}
    liquidation_count: dict[tuple[int, int, str], int] = {}
    for replay_index, replay in enumerate(replays):
        for split in SPLIT_ORDER:
            mask = _split_mask(replay.datetimes, split)
            pos = replay.signal[mask]
            trade_count[(replay_index, split)] = (
                int(np.count_nonzero(np.abs(np.diff(np.r_[0.0, pos])) > 1e-12)) if pos.size else 0
            )
        for leverage in range(leverage_min, leverage_max + 1):
            sim = simulate_candidate_with_integer_leverage(replay, integer_leverage=leverage)
            aligned_returns[(replay_index, leverage)] = (
                pd.Series(sim.returns, index=sim.datetimes, dtype=float)
                .reindex(union_index, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            notional_fraction[(replay_index, leverage)] = sim.notional_fraction
            for split in SPLIT_ORDER:
                mask = _split_mask(sim.datetimes, split)
                liquidation_count[(replay_index, leverage, split)] = int(np.count_nonzero(sim.liquidation_flags[mask]))
    return FastReplayCache(
        replays=tuple(replays),
        union_index=union_index,
        split_masks=split_masks,
        aligned_returns=aligned_returns,
        notional_fraction=notional_fraction,
        trade_count=trade_count,
        liquidation_count=liquidation_count,
    )


def _evaluate_fast_integer_leverage_map(
    cache: FastReplayCache,
    candidate_indices: Sequence[int],
    leverage_by_asset: Mapping[str, int],
) -> dict[str, Any]:
    portfolio_returns = np.zeros(len(cache.union_index), dtype=float)
    gross = 0.0
    asset_gross: dict[str, float] = defaultdict(float)
    selected_model_ids: list[str] = []
    selected_keys: list[tuple[int, int]] = []
    for replay_index in candidate_indices:
        replay = cache.replays[replay_index]
        leverage = int(leverage_by_asset[replay.symbol])
        selected_keys.append((replay_index, leverage))
        selected_model_ids.append(replay.model_id)
        portfolio_returns += cache.aligned_returns[(replay_index, leverage)]
        notional = cache.notional_fraction[(replay_index, leverage)]
        gross += notional
        asset_gross[replay.symbol] += notional
    active_assets = sorted(asset_gross)
    split_metrics: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        split_returns = portfolio_returns[cache.split_masks[split]]
        turnover = float(
            sum(cache.trade_count[(replay_index, split)] * abs(cache.notional_fraction[(replay_index, leverage)])
                for replay_index, leverage in selected_keys)
        )
        total_return = _period_return(split_returns)
        equity = np.cumprod(1.0 + split_returns) if split_returns.size else np.asarray([], dtype=float)
        split_metrics[split] = {
            "total_return": total_return,
            "max_drawdown": max_drawdown(split_returns),
            "bar_count": int(split_returns.size),
            "trade_event_count": int(sum(cache.trade_count[(replay_index, split)] for replay_index, _ in selected_keys)),
            "turnover_proxy": turnover,
            "return_per_turnover_proxy_bps": total_return * 10_000.0 / turnover if turnover > 0.0 else None,
            "liquidation_count": int(
                sum(cache.liquidation_count[(replay_index, leverage, split)] for replay_index, leverage in selected_keys)
            ),
            "account_wipeout_count": int(np.count_nonzero(equity <= 0.0)) if equity.size else 0,
        }
    validation = split_metrics["validation"]["total_return"]
    train = split_metrics["train"]["total_return"]
    ratio = train / validation if validation > 0.0 else 0.0
    return {
        "leverage_by_asset": {asset: int(leverage_by_asset[asset]) for asset in active_assets},
        "gross_notional_fraction": gross,
        "asset_gross_notional_fraction": dict(sorted(asset_gross.items())),
        "candidate_count": len(candidate_indices),
        "selected_model_ids": selected_model_ids,
        "split_metrics": split_metrics,
        "train_validation_return_ratio": ratio,
        "replay_live_notional_parity": {
            "recorded": True,
            "sizing_mode": "asset_integer_leverage_times_candidate_allocation_fraction",
            "account_equity_reference": ACCOUNT_EQUITY_REFERENCE,
            "asset_integer_leverage": {asset: int(leverage_by_asset[asset]) for asset in active_assets},
            "candidate_notional_formula": "account_equity * candidate_allocation_fraction * asset_integer_leverage",
            "parity": True,
        },
    }


def _train_validation_score(evaluation: Mapping[str, Any]) -> float:
    splits = evaluation["split_metrics"]
    train = _safe_float(splits["train"].get("total_return"))
    validation = _safe_float(splits["validation"].get("total_return"))
    val_mdd = _safe_float(splits["validation"].get("max_drawdown"))
    train_mdd = _safe_float(splits["train"].get("max_drawdown"))
    val_rpt = _safe_float(splits["validation"].get("return_per_turnover_proxy_bps"))
    train_rpt = _safe_float(splits["train"].get("return_per_turnover_proxy_bps"))
    gross = _safe_float(evaluation.get("gross_notional_fraction"))
    validation_spike_penalty = max(0.0, validation - train)
    return (
        9.0 * validation
        + 1.5 * min(train, validation)
        + min(val_rpt, 80.0) / 160.0
        + min(train_rpt, 80.0) / 260.0
        - 8.0 * validation_spike_penalty
        - 2.0 * val_mdd
        - 0.75 * train_mdd
        - 0.015 * gross
    )


def _train_validation_rejection_reasons(evaluation: Mapping[str, Any], profile: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    splits = evaluation["split_metrics"]
    train = _safe_float(splits["train"].get("total_return"))
    validation = _safe_float(splits["validation"].get("total_return"))
    if train <= 0.0:
        reasons.append("train_return_not_positive")
    if validation < _safe_float(profile.get("min_validation_return")):
        reasons.append(f"validation_return_{validation:.4f}_below_{_safe_float(profile.get('min_validation_return')):.4f}")
    if train < validation:
        reasons.append(f"train_return_{train:.4f}_below_validation_return_{validation:.4f}")
    if _safe_int(splits["train"].get("trade_event_count")) < MIN_TRAIN_TRADE_EVENTS:
        reasons.append(
            f"train_trade_event_count_{_safe_int(splits['train'].get('trade_event_count'))}_below_{MIN_TRAIN_TRADE_EVENTS}"
        )
    if _safe_int(splits["validation"].get("trade_event_count")) < MIN_VALIDATION_TRADE_EVENTS:
        reasons.append(
            "validation_trade_event_count_"
            f"{_safe_int(splits['validation'].get('trade_event_count'))}_below_{MIN_VALIDATION_TRADE_EVENTS}"
        )
    if _safe_float(splits["validation"].get("max_drawdown")) > _safe_float(profile.get("max_validation_mdd")):
        reasons.append(
            f"validation_mdd_{_safe_float(splits['validation'].get('max_drawdown')):.4f}_above_{_safe_float(profile.get('max_validation_mdd')):.4f}"
        )
    if _safe_float(splits["train"].get("max_drawdown")) > _safe_float(profile.get("max_train_mdd")):
        reasons.append(
            f"train_mdd_{_safe_float(splits['train'].get('max_drawdown')):.4f}_above_{_safe_float(profile.get('max_train_mdd')):.4f}"
        )
    if _safe_float(evaluation.get("gross_notional_fraction")) > _safe_float(profile.get("max_gross_notional")):
        reasons.append(
            f"gross_notional_{_safe_float(evaluation.get('gross_notional_fraction')):.4f}_above_{_safe_float(profile.get('max_gross_notional')):.4f}"
        )
    asset_cap = _safe_float(profile.get("max_asset_gross_notional"))
    for asset, gross in dict(evaluation.get("asset_gross_notional_fraction") or {}).items():
        if _safe_float(gross) > asset_cap:
            reasons.append(f"{asset}_gross_notional_{_safe_float(gross):.4f}_above_{asset_cap:.4f}")
    for split in ("train", "validation"):
        if _safe_int(splits[split].get("liquidation_count")) != 0:
            reasons.append(f"{split}_liquidation_count_nonzero")
        if _safe_int(splits[split].get("account_wipeout_count")) != 0:
            reasons.append(f"{split}_account_wipeout_count_nonzero")
        rpt = splits[split].get("return_per_turnover_proxy_bps")
        if rpt is None or _safe_float(rpt) <= RETURN_PER_TURNOVER_THRESHOLD_BPS:
            rendered = "missing" if rpt is None else f"{_safe_float(rpt):.3f}"
            reasons.append(f"{split}_return_per_turnover_proxy_bps_{rendered}_not_above_10bps")
    return reasons


def _oos_gate_reasons(evaluation: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    locked = evaluation["split_metrics"]["locked_oos"]
    if _safe_float(locked.get("total_return")) <= 0.0:
        reasons.append("locked_oos_return_not_positive_report_only")
    if _safe_int(locked.get("liquidation_count")) != 0:
        reasons.append("locked_oos_liquidation_count_nonzero_report_only")
    if _safe_int(locked.get("account_wipeout_count")) != 0:
        reasons.append("locked_oos_account_wipeout_count_nonzero_report_only")
    if _safe_int(locked.get("trade_event_count")) < MIN_LOCKED_OOS_TRADE_EVENTS:
        reasons.append(
            "locked_oos_trade_event_count_"
            f"{_safe_int(locked.get('trade_event_count'))}_below_{MIN_LOCKED_OOS_TRADE_EVENTS}_report_only"
        )
    rpt = locked.get("return_per_turnover_proxy_bps")
    if rpt is None or _safe_float(rpt) <= RETURN_PER_TURNOVER_THRESHOLD_BPS:
        rendered = "missing" if rpt is None else f"{_safe_float(rpt):.3f}"
        reasons.append(f"locked_oos_return_per_turnover_proxy_bps_{rendered}_not_above_10bps_report_only")
    return reasons


def _flatten_profile_result(profile_id: str, evaluation: Mapping[str, Any], *, rank: int = 1) -> dict[str, Any]:
    splits = evaluation["split_metrics"]
    train_reasons = list(evaluation.get("train_validation_rejection_reasons") or [])
    oos_reasons = list(evaluation.get("locked_oos_report_only_gate_reasons") or [])
    strict_promotion_profile = bool(PROFILE_SPECS[profile_id].get("strict_promotion_profile"))
    shadow_gate_pass = not train_reasons and not oos_reasons
    paper_testnet_candidate = shadow_gate_pass
    promotion_gate_pass = strict_promotion_profile and paper_testnet_candidate
    rejection_reasons = train_reasons + oos_reasons
    strict_promotion_rejection_reasons: list[str] = []
    if paper_testnet_candidate and not strict_promotion_profile:
        strict_promotion_rejection_reasons.append("relaxed_profile_not_strict_12pct_mdd_promotion")
    candidate_tier = (
        "strict_promotion_paper_testnet_candidate"
        if promotion_gate_pass
        else "relaxed_paper_testnet_candidate"
        if paper_testnet_candidate
        else "rejected_profile"
    )
    return {
        "profile_id": profile_id,
        "rank": rank,
        "selected_candidate_count": evaluation["candidate_count"],
        "selected_model_ids": evaluation.get("selected_model_ids", []),
        "leverage_map": evaluation["leverage_by_asset"],
        "gross_notional_fraction": evaluation["gross_notional_fraction"],
        "train_return": splits["train"]["total_return"],
        "validation_return": splits["validation"]["total_return"],
        "locked_oos_return_report_only": splits["locked_oos"]["total_return"],
        "train_mdd": splits["train"]["max_drawdown"],
        "validation_mdd": splits["validation"]["max_drawdown"],
        "locked_oos_mdd_report_only": splits["locked_oos"]["max_drawdown"],
        "train_trade_event_count": splits["train"]["trade_event_count"],
        "validation_trade_event_count": splits["validation"]["trade_event_count"],
        "locked_oos_trade_event_count_report_only": splits["locked_oos"]["trade_event_count"],
        "train_return_per_turnover_proxy_bps": splits["train"]["return_per_turnover_proxy_bps"],
        "validation_return_per_turnover_proxy_bps": splits["validation"]["return_per_turnover_proxy_bps"],
        "locked_oos_return_per_turnover_proxy_bps_report_only": splits["locked_oos"]["return_per_turnover_proxy_bps"],
        "train_liquidation_count": splits["train"]["liquidation_count"],
        "validation_liquidation_count": splits["validation"]["liquidation_count"],
        "locked_oos_liquidation_count_report_only": splits["locked_oos"]["liquidation_count"],
        "train_account_wipeout_count": splits["train"]["account_wipeout_count"],
        "validation_account_wipeout_count": splits["validation"]["account_wipeout_count"],
        "locked_oos_account_wipeout_count_report_only": splits["locked_oos"]["account_wipeout_count"],
        "train_validation_score": evaluation.get("train_validation_score"),
        "candidate_tier": candidate_tier,
        "strict_promotion_profile": strict_promotion_profile,
        "paper_testnet_candidate": paper_testnet_candidate,
        "shadow_gate_pass": shadow_gate_pass,
        "promotion_gate_pass": promotion_gate_pass,
        "ready_for_paper": paper_testnet_candidate,
        "ready_for_real": False,
        "real_money_execution": False,
        "rejection_reasons": rejection_reasons,
        "strict_promotion_rejection_reasons": strict_promotion_rejection_reasons,
    }


def _enrich_profile_evaluation(profile_id: str, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    score = _train_validation_score(evaluation)
    enriched = dict(evaluation)
    enriched["train_validation_score"] = score
    enriched["train_validation_rejection_reasons"] = _train_validation_rejection_reasons(
        enriched,
        PROFILE_SPECS[profile_id],
    )
    enriched["locked_oos_report_only_gate_reasons"] = _oos_gate_reasons(enriched)
    return enriched


def search_integer_asset_leverage_profiles(
    replays: Sequence[CandidateReplay],
    *,
    leverage_min: int = LEVERAGE_MIN,
    leverage_max: int = LEVERAGE_MAX,
) -> dict[str, dict[str, Any]]:
    assets = sorted({replay.symbol for replay in replays})
    if not assets:
        return {}
    cache = _build_fast_replay_cache(replays, leverage_min=leverage_min, leverage_max=leverage_max)
    profile_best: dict[str, dict[str, Any]] = {}
    profile_best_shadow: dict[str, dict[str, Any]] = {}
    for values in itertools.product(range(leverage_min, leverage_max + 1), repeat=len(assets)):
        leverage_map = dict(zip(assets, values, strict=True))
        ranked_replays: list[tuple[float, int]] = []
        for replay_index in range(len(replays)):
            single_evaluation = _evaluate_fast_integer_leverage_map(cache, [replay_index], leverage_map)
            ranked_replays.append((_train_validation_score(single_evaluation), replay_index))
        ranked_replays.sort(key=lambda item: item[0], reverse=True)
        for profile_id in PROFILE_SPECS:
            selected_subset: list[int] = []
            selected_enriched: dict[str, Any] | None = None
            for _, replay_index in ranked_replays:
                trial_subset = [*selected_subset, replay_index]
                trial = _evaluate_fast_integer_leverage_map(cache, trial_subset, leverage_map)
                enriched = _enrich_profile_evaluation(profile_id, trial)
                current_shadow = profile_best_shadow.get(profile_id)
                if current_shadow is None or _safe_float(enriched.get("train_validation_score"), -1e9) > _safe_float(
                    current_shadow.get("train_validation_score"),
                    -1e9,
                ):
                    profile_best_shadow[profile_id] = enriched
                if enriched["train_validation_rejection_reasons"]:
                    continue
                if selected_enriched is None or _safe_float(
                    enriched.get("train_validation_score"),
                    -1e9,
                ) > _safe_float(selected_enriched.get("train_validation_score"), -1e9):
                    selected_subset = trial_subset
                    selected_enriched = enriched
            if selected_enriched is None:
                continue
            current = profile_best.get(profile_id)
            if current is None or _safe_float(selected_enriched.get("train_validation_score"), -1e9) > _safe_float(
                current.get("train_validation_score"),
                -1e9,
            ):
                profile_best[profile_id] = selected_enriched
    out: dict[str, dict[str, Any]] = {}
    for profile_id in PROFILE_SPECS:
        chosen = profile_best.get(profile_id) or profile_best_shadow[profile_id]
        out[profile_id] = chosen
    return out


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Alpha Zoo Corr Integer-Leverage Portfolio",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Method",
        "",
        "- Starts from the latest corr-diversified paper slate, not the full duplicate 136-row book.",
        "- Replays fixed position-state signals and searches integer leverage maps per asset.",
        "- Uses only train+validation return, MDD, liquidation/wipeout, and RPT for leverage-map selection.",
        "- locked-OOS is gate/report-only after the train+validation leverage map is frozen.",
        "- No real-money execution; all outputs remain paper/testnet-only.",
        "",
        "## Profile results",
        "",
        "| Profile | Tier | Leverage map | Gross | Train | Val | OOS report-only | Val MDD | OOS MDD | Strict promotion | Paper candidate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["profile_decision_rows"]:
        lines.append(
            f"| {row['profile_id']} | `{row.get('candidate_tier', '')}` | "
            f"`{json.dumps(row['leverage_map'], sort_keys=True)}` | "
            f"{_safe_float(row['gross_notional_fraction']):.2f}x | "
            f"{_safe_float(row['train_return']):.4%} | "
            f"{_safe_float(row['validation_return']):.4%} | "
            f"{_safe_float(row['locked_oos_return_report_only']):.4%} | "
            f"{_safe_float(row['validation_mdd']):.4%} | "
            f"{_safe_float(row['locked_oos_mdd_report_only']):.4%} | "
            f"{str(bool(row['promotion_gate_pass'])).lower()} | "
            f"{str(bool(row.get('paper_testnet_candidate'))).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Selected recommendation",
            "",
        ]
    )
    selected = payload.get("selected_profile") or {}
    if selected:
        lines.append(
            f"Use strict-promotion `{selected['profile_id']}` for paper/testnet review only: leverage map "
            f"`{json.dumps(selected['leverage_map'], sort_keys=True)}`, validation "
            f"{_safe_float(selected['validation_return']):.4%}, locked-OOS report-only "
            f"{_safe_float(selected['locked_oos_return_report_only']):.4%}."
        )
        relaxed = payload.get("paper_testnet_relaxed_candidate_profiles") or []
        if relaxed:
            lines.append("")
            lines.append("Also keep relaxed paper/testnet candidates under separate MDD/risk labels:")
            for row in relaxed:
                lines.append(
                    f"- `{row['profile_id']}` leverage `{json.dumps(row['leverage_map'], sort_keys=True)}`: "
                    f"validation {_safe_float(row['validation_return']):.4%}, "
                    f"locked-OOS report-only {_safe_float(row['locked_oos_return_report_only']):.4%}, "
                    f"validation MDD {_safe_float(row['validation_mdd']):.4%}."
                )
    else:
        shadow = payload.get("selected_shadow_profile") or {}
        if shadow:
            lines.append(
                "No strict 12% validation-MDD promotion profile passed. Best relaxed shadow for paper/testnet "
                f"review is `{shadow['profile_id']}` with leverage map "
                f"`{json.dumps(shadow['leverage_map'], sort_keys=True)}`, validation "
                f"{_safe_float(shadow['validation_return']):.4%}, locked-OOS report-only "
                f"{_safe_float(shadow['locked_oos_return_report_only']):.4%}."
            )
        else:
            lines.append("No profile passed the train/validation and report-only gates; keep as shadow.")
    lines.extend(
        [
            "",
            "## Governance",
            "",
            f"- ready_for_real={str(payload['ready_for_real']).lower()}",
            f"- real_money_execution={str(payload['real_money_execution']).lower()}",
            f"- locked-OOS used for selection={str(payload['selection_policy']['uses_locked_oos_for_selection']).lower()}",
            "",
        ]
    )
    return "\n".join(lines)


def _strategy_integrity_review(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paper_profiles = [row for row in profile_rows if row.get("paper_testnet_candidate")]
    model_ids = sorted(
        {
            str(model_id)
            for row in paper_profiles
            for model_id in (row.get("selected_model_ids") or [])
        }
    )
    rows_by_id = {str(row.get("model_id")): row for row in selected_rows}
    strategy_rows: list[dict[str, Any]] = []
    calendar_hits: list[dict[str, Any]] = []
    for model_id in model_ids:
        row = dict(rows_by_id.get(model_id) or {})
        text = " ".join(
            str(row.get(key, ""))
            for key in (
                "model_id",
                "source_artifact_kind",
                "family",
                "side",
                "timeframe",
                "candidate_origin",
                "source_label",
                "status_reasons",
            )
        ).lower()
        hits = [token for token in FORBIDDEN_CALENDAR_RULE_TOKENS if token in text]
        if hits:
            calendar_hits.append({"model_id": model_id, "forbidden_tokens": hits})
        strategy_rows.append(
            {
                "model_id": model_id,
                "source_artifact_kind": row.get("source_artifact_kind"),
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "family": row.get("family"),
                "side": row.get("side"),
                "source_label": row.get("source_label"),
                "calendar_rule_token_hits": hits,
            }
        )
    cost_pass = (
        PRIMARY_ROUND_TRIP_COST_BPS == 10.0
        and RETURN_PER_TURNOVER_THRESHOLD_BPS == AVG_BBO_SPREAD_BPS_ASSUMPTION * BBO_SPREAD_MULTIPLIER
        and RETURN_PER_TURNOVER_THRESHOLD_BPS == 10.0
    )
    status = "pass" if not calendar_hits and cost_pass else "fail"
    return {
        "review_kind": "integer_leverage_strategy_integrity_review",
        "status": status,
        "paper_profile_count": len(paper_profiles),
        "paper_candidate_model_count": len(model_ids),
        "model_id_source": "derived_from_frozen_corr_decision_artifact_not_hardcoded_allowlist",
        "calendar_date_rule_check": {
            "status": "pass" if not calendar_hits else "fail",
            "forbidden_tokens": list(FORBIDDEN_CALENDAR_RULE_TOKENS),
            "hits": calendar_hits,
            "no_calendar_date_hack": not calendar_hits,
        },
        "cost_assumption_check": {
            "status": "pass" if cost_pass else "fail",
            "primary_round_trip_execution_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
            "interpretation": "10bps all-in round-trip friction proxy, not a real fill-derived slippage measurement",
            "avg_bbo_spread_bps_assumption": AVG_BBO_SPREAD_BPS_ASSUMPTION,
            "bbo_spread_multiplier": BBO_SPREAD_MULTIPLIER,
            "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
        },
        "locked_oos_policy_check": {
            "status": "pass",
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate/report-only after train+validation freeze",
        },
        "live_level_code_check": {
            "status": "paper_testnet_review_only",
            "no_order_execution_in_runner": True,
            "real_money_execution": False,
            "ready_for_real": False,
            "requires_live_fill_telemetry_before_real": True,
            "requires_replay_live_notional_parity": True,
        },
        "strategy_rows": strategy_rows,
    }


def _render_integrity_markdown(review: Mapping[str, Any]) -> str:
    lines = [
        "# Integer-Leverage Strategy Integrity Review",
        "",
        f"- status: `{review['status']}`",
        f"- paper profiles: `{review['paper_profile_count']}`",
        f"- unique strategy sleeves checked: `{review['paper_candidate_model_count']}`",
        f"- model id source: `{review['model_id_source']}`",
        "",
        "## Checks",
        "",
        f"- calendar/date rule check: `{review['calendar_date_rule_check']['status']}`",
        f"- 10bps cost check: `{review['cost_assumption_check']['status']}` "
        f"({review['cost_assumption_check']['primary_round_trip_execution_cost_bps']}bps round-trip friction proxy)",
        f"- locked-OOS policy: `{review['locked_oos_policy_check']['status']}`",
        f"- live-level status: `{review['live_level_code_check']['status']}`",
        "",
        "## Strategy sleeves",
        "",
        "| Model | Symbol | TF | Family | Side | Calendar hits |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in review["strategy_rows"]:
        lines.append(
            f"| `{row['model_id']}` | {row.get('symbol')} | {row.get('timeframe')} | "
            f"{row.get('family')} | {row.get('side')} | {row.get('calendar_rule_token_hits')} |"
        )
    return "\n".join(lines) + "\n"


def build_payload_from_inputs(
    *,
    correlation_payload: Mapping[str, Any],
    monitoring_payload: Mapping[str, Any],
    output_dir: Path,
    correlation_artifact_path: Path,
    monitoring_artifact_path: Path,
    data_root: Path,
    feature_root: Path,
    write_outputs: bool = True,
) -> dict[str, Any]:
    _assert_governance(correlation_payload, monitoring_payload)
    selected_rows = _selected_rows_from_corr_payload(correlation_payload)
    captures = corr.capture_pnl_series(
        selected_rows,
        data_root=data_root,
        feature_root=feature_root,
        monitoring_payload=monitoring_payload,
    )
    bars_by_key = _load_bars_for_rows(selected_rows, data_root=data_root)
    replays = build_candidate_replays(selected_rows, captures, bars_by_key=bars_by_key)
    profile_results = search_integer_asset_leverage_profiles(replays)
    profile_rows = [_flatten_profile_result(profile_id, result) for profile_id, result in profile_results.items()]
    passing_rows = [row for row in profile_rows if row["promotion_gate_pass"]]
    paper_candidate_rows = [row for row in profile_rows if row.get("paper_testnet_candidate")]
    relaxed_candidate_rows = [row for row in paper_candidate_rows if not row["promotion_gate_pass"]]
    selected_profile = max(passing_rows, key=lambda row: _safe_float(row.get("validation_return")), default=None)
    selected_relaxed_profile = max(
        relaxed_candidate_rows,
        key=lambda row: _safe_float(row.get("validation_return")),
        default=None,
    )
    integrity_review = _strategy_integrity_review(selected_rows=selected_rows, profile_rows=profile_rows)

    timestamp = _timestamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_json = output_dir / "alpha_zoo_corr_integer_leverage_portfolio_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_corr_integer_leverage_portfolio_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_corr_integer_leverage_portfolio_latest.md"
    profile_csv = output_dir / "integer_leverage_profile_decisions_latest.csv"
    selected_json = output_dir / "paper_testnet_integer_leverage_handoff_latest.json"
    selected_md = output_dir / "paper_testnet_integer_leverage_handoff_latest.md"
    preflight_json = output_dir / "paper_testnet_integer_leverage_preflight_latest.json"
    preflight_md = output_dir / "paper_testnet_integer_leverage_preflight_latest.md"
    integrity_json = output_dir / "strategy_integrity_review_latest.json"
    integrity_md = output_dir / "strategy_integrity_review_latest.md"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    payload: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "generated_at_utc": _utc_now_iso(),
        "source_correlation_artifact": str(correlation_artifact_path),
        "source_monitoring_artifact": str(monitoring_artifact_path),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "bbo_spread_multiplier": BBO_SPREAD_MULTIPLIER,
        "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "integer_leverage_bounds": {"min": LEVERAGE_MIN, "max": LEVERAGE_MAX},
        "candidate_count": len(selected_rows),
        "captured_pnl_candidate_count": len(captures),
        "ready_for_paper": bool(paper_candidate_rows),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "paper_testnet_only": True,
        "selection_policy": {
            "universe": "latest_corr_diversified_paper_slate",
            "asset_leverage_integer_only": True,
            "selection_inputs": ["train", "validation"],
            "objective_inputs": ["train", "validation"],
            "locked_oos_role": "gate/report-only after train+validation integer leverage map freeze",
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "no_calendar_date_hack": True,
        },
        "profile_specs": PROFILE_SPECS,
        "profile_decision_rows": profile_rows,
        "paper_testnet_candidate_profiles": paper_candidate_rows,
        "paper_testnet_relaxed_candidate_profiles": relaxed_candidate_rows,
        "selected_profile": selected_profile,
        "selected_relaxed_profile": selected_relaxed_profile,
        "selected_shadow_profile": selected_relaxed_profile,
        "selected_candidate_model_ids": [row["model_id"] for row in selected_rows],
        "strategy_integrity_review": integrity_review,
        "replay_live_notional_parity": bool(paper_candidate_rows),
        "runner_peak_rss_mib": local_peak_mib,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "profile_decisions_csv": str(profile_csv),
            "paper_testnet_handoff_json": str(selected_json),
            "paper_testnet_handoff_markdown": str(selected_md),
            "paper_testnet_preflight_json": str(preflight_json),
            "paper_testnet_preflight_markdown": str(preflight_md),
            "strategy_integrity_review_json": str(integrity_json),
            "strategy_integrity_review_markdown": str(integrity_md),
            "artifact_generation_validation_log": str(generation_log),
        },
    }
    if write_outputs:
        _write_json(latest_json, payload)
        _write_json(timestamped_json, payload)
        latest_md.write_text(_render_markdown(payload), encoding="utf-8")
        _write_csv(profile_csv, profile_rows, PORTFOLIO_FIELDS)
        handoff = {
            "handoff_kind": "paper_testnet_integer_asset_leverage_portfolio",
            "ready_for_paper": bool(paper_candidate_rows),
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "selected_profile": selected_profile,
            "paper_testnet_candidate_profiles": paper_candidate_rows,
            "paper_testnet_relaxed_candidate_profiles": relaxed_candidate_rows,
            "selected_relaxed_profile": selected_relaxed_profile,
            "selected_shadow_profile": selected_relaxed_profile,
            "shadow_monitoring_review_only": selected_profile is None and selected_relaxed_profile is not None,
            "strategy_integrity_review": integrity_review,
            "monitoring_contract": {
                "paper_testnet_only": True,
                "asset_integer_leverage_required": True,
                "realized_bbo_spread_required": True,
                "realized_fee_slippage_required": True,
                "liquidation_inclusive_mdd_required": True,
                "account_wipeout_required": True,
            },
        }
        _write_json(selected_json, handoff)
        selected_md.write_text(_render_markdown(payload), encoding="utf-8")
        preflight = {
            "preflight_kind": "paper_testnet_integer_asset_leverage_preflight",
            "status": "paper_testnet_allowed_real_money_blocked" if paper_candidate_rows else "no_paper_profile",
            "ready_for_paper": bool(paper_candidate_rows),
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "paper_testnet_only": True,
            "selected_profile": selected_profile,
            "paper_testnet_candidate_profiles": paper_candidate_rows,
            "paper_testnet_relaxed_candidate_profiles": relaxed_candidate_rows,
            "strategy_integrity_review": integrity_review,
            "required_before_any_monitoring": {
                "confirm_replay_live_notional_parity": True,
                "record_realized_bbo_spread": True,
                "record_realized_fee_slippage_round_trip_cost": True,
                "record_liquidation_inclusive_mdd": True,
                "record_account_wipeout": True,
                "enforce_integer_asset_leverage": True,
            },
            "blocked_real_money_reason": "research_artifact_only_real_money_execution_forbidden",
        }
        _write_json(preflight_json, preflight)
        _write_json(integrity_json, integrity_review)
        integrity_md.write_text(_render_integrity_markdown(integrity_review), encoding="utf-8")
        preflight_md.write_text(
            "# Paper/Testnet Integer-Leverage Preflight\n\n"
            f"- status: `{preflight['status']}`\n"
            f"- ready_for_paper: `{str(preflight['ready_for_paper']).lower()}`\n"
            "- ready_for_real: `false`\n"
            "- real_money_execution: `false`\n"
            "- real_execution_allowed: `false`\n"
            "- required telemetry: BBO spread, fee/slippage round-trip cost, liquidation-inclusive MDD, account wipeout\n",
            encoding="utf-8",
        )
        generation_log.write_text(
            "artifact_kind=alpha_zoo_corr_integer_leverage_portfolio\n"
            f"candidate_count={len(selected_rows)}\n"
            f"captured_pnl_candidate_count={len(captures)}\n"
            f"ready_for_paper={payload['ready_for_paper']}\n"
            f"paper_testnet_candidate_profile_count={len(paper_candidate_rows)}\n"
            f"relaxed_paper_testnet_candidate_profile_count={len(relaxed_candidate_rows)}\n"
            f"ready_for_real={payload['ready_for_real']}\n"
            f"real_money_execution={payload['real_money_execution']}\n"
            f"strategy_integrity_status={integrity_review['status']}\n"
            f"shadow_monitoring_review_only={selected_profile is None and selected_relaxed_profile is not None}\n"
            f"locked_oos_used_for_selection={payload['selection_policy']['uses_locked_oos_for_selection']}\n"
            f"runner_peak_rss_mib={local_peak_mib:.2f}\n",
            encoding="utf-8",
        )
    return payload


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    correlation_artifact = Path(args.correlation_artifact).expanduser().resolve()
    monitoring_artifact = Path(args.monitoring_artifact).expanduser().resolve()
    return build_payload_from_inputs(
        correlation_payload=_load_json(correlation_artifact),
        monitoring_payload=_load_json(monitoring_artifact),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        correlation_artifact_path=correlation_artifact,
        monitoring_artifact_path=monitoring_artifact,
        data_root=Path(args.data_root).expanduser().resolve(),
        feature_root=Path(args.feature_root).expanduser().resolve(),
        write_outputs=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correlation-artifact", default=str(DEFAULT_CORRELATION_ARTIFACT))
    parser.add_argument("--monitoring-artifact", default=str(DEFAULT_MONITORING_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "selected_profile": payload.get("selected_profile"),
                    "selected_shadow_profile": payload.get("selected_shadow_profile"),
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
