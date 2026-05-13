#!/usr/bin/env python3
"""Replay CryptoFxAlphaZooStateStrategy with real-data calibrated edges."""

from __future__ import annotations

import argparse
import json
import math
import queue
import resource
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lumina_quant.alpha_zoo.crypto_fx_factors import assign_time_splits
from lumina_quant.core.events import MarketBatchEvent, MarketEvent
from lumina_quant.research.crypto_fx_alpha_zoo_real_data import load_real_data_bundle
from lumina_quant.strategies.crypto_fx_alpha_zoo_state import CryptoFxAlphaZooStateStrategy

DEFAULT_OUTPUT = (
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_state_replay_latest.json"
)
CURRENT_BASE_REFERENCE = {
    "role": "hypothesis_reference_only",
    "selection_target": False,
    "promotion_target": False,
    "locked_oos_total_return": 0.06428110030664325,
    "locked_oos_return_mdd": 6.9168776779021455,
    "locked_oos_sharpe": 5.202361970933632,
}
PROMOTION_POLICY = {
    "name": "strict_liquidation_sharpe_sortino_calmar_gate_20260513",
    "return_mdd_hurdle_required": False,
    "return_mdd_role": "diagnostic_report_only",
    "risk_adjusted_substitutes": ["sharpe", "sortino", "smart_sortino", "calmar", "max_drawdown"],
    "requires_oos_return_beats_current_base": True,
    "requires_oos_mdd_lte": 0.25,
}


@dataclass(slots=True)
class _Bars:
    symbol_list: list[str]


@dataclass(frozen=True, slots=True)
class _GridSpec:
    name: str
    source: str
    params: dict[str, Any]


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


def _load_calibrated_edges(path: str | Path | None) -> dict[str, float]:
    raw = str(path or "").strip()
    if not raw:
        return {}
    payload = json.loads(Path(raw).expanduser().read_text(encoding="utf-8"))
    explicit = payload.get("calibrated_edges_for_strategy")
    if isinstance(explicit, dict):
        return {str(key): float(value) for key, value in explicit.items() if _safe_float(value) > 0.0}
    edges: dict[str, float] = {}
    for key, item in dict(payload.get("calibrations") or {}).items():
        decision = dict(item.get("decision") or {})
        if str(decision.get("action")) not in {"allow", "downsize"}:
            continue
        lower = _safe_float(decision.get("lower_confidence_edge_bps"))
        if lower <= 0.0:
            continue
        parts = str(key).split("|")
        if len(parts) >= 3:
            edges[f"{parts[2]}:{parts[1]}"] = max(edges.get(f"{parts[2]}:{parts[1]}", 0.0), lower)
        if len(parts) >= 2:
            edges[f"default:{parts[1]}"] = max(edges.get(f"default:{parts[1]}", 0.0), lower)
    return edges


def _ensure_replay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if "split" not in data.columns:
        data = assign_time_splits(data)
    return data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _run_strategy_signals(
    data: pd.DataFrame,
    *,
    require_calibrated_edge: bool,
    calibrated_edges: dict[str, float],
    strategy_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    symbols = sorted(str(item) for item in data["symbol"].dropna().unique())
    events: queue.Queue = queue.Queue()
    strategy = CryptoFxAlphaZooStateStrategy(
        _Bars(symbols),
        events,
        require_calibrated_edge=require_calibrated_edge,
        calibrated_edges=calibrated_edges,
        **dict(strategy_params or {}),
    )
    for ts, group in data.groupby("timestamp", sort=True):
        bars = tuple(
            MarketEvent(
                time=ts,
                symbol=str(row.symbol),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in group.itertuples(index=False)
        )
        strategy.calculate_signals(MarketBatchEvent(time=ts, bars=bars))
    signal_rows: list[dict[str, Any]] = []
    while not events.empty():
        signal = events.get()
        signal_rows.append(
            {
                "datetime": pd.Timestamp(signal.datetime).isoformat(),
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "strength": float(signal.strength),
                "price": float(getattr(signal, "price", 0.0) or 0.0),
                "metadata": signal.metadata or {},
            }
        )
    return signal_rows


def _price_lookup(data: pd.DataFrame) -> pd.DataFrame:
    return data.set_index(["timestamp", "symbol"]).sort_index()


def _split_lookup_map(data: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], str]:
    return {
        (pd.Timestamp(row.timestamp), str(row.symbol)): str(row.split)
        for row in data[["timestamp", "symbol", "split"]].itertuples(index=False)
    }


def _build_trades(data: pd.DataFrame, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_lookup = _split_lookup_map(data)
    last_rows = {str(row.symbol): row for row in data.sort_values("timestamp").itertuples(index=False)}
    positions: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    for signal in sorted(signals, key=lambda row: row["datetime"]):
        symbol = str(signal["symbol"])
        side = str(signal["signal_type"]).upper()
        ts = pd.Timestamp(signal["datetime"])
        price = _safe_float(signal.get("price"))
        if side in {"LONG", "SHORT"} and symbol not in positions:
            positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "entry_time": ts,
                "entry_price": price,
                "entry_split": split_lookup.get((ts, symbol), "unknown"),
                "entry_metadata": dict(signal.get("metadata") or {}),
            }
        elif side == "EXIT" and symbol in positions:
            trade = positions.pop(symbol)
            trade.update({"exit_time": ts, "exit_price": price, "exit_reason": "strategy_exit"})
            trades.append(trade)
    if positions:
        for symbol, trade in list(positions.items()):
            last = last_rows.get(symbol)
            if last is None:
                continue
            trade.update(
                {
                    "exit_time": pd.Timestamp(last.timestamp),
                    "exit_price": _safe_float(last.close),
                    "exit_reason": "end_of_sample",
                }
            )
            trades.append(trade)
    for trade in trades:
        mult = 1.0 if trade["side"] == "LONG" else -1.0
        entry = _safe_float(trade.get("entry_price"), 1.0)
        exit_ = _safe_float(trade.get("exit_price"), entry)
        trade["gross_return"] = mult * ((exit_ / entry) - 1.0) if entry > 0.0 else 0.0
    return trades


def _train_validation_metrics(
    trades: list[dict[str, Any]], *, leverage: float = 1.0, allocation_fraction: float = 0.10
) -> dict[str, dict[str, Any]]:
    metrics = _split_metrics(trades, leverage=leverage, allocation_fraction=allocation_fraction)
    return {split: metrics[split] for split in ("train", "validation")}


def _selection_score(metrics: dict[str, dict[str, Any]]) -> float:
    train = metrics.get("train", {})
    validation = metrics.get("validation", {})
    train_return = _safe_float(train.get("total_return"))
    validation_return = _safe_float(validation.get("total_return"))
    validation_mdd = _safe_float(validation.get("max_drawdown"), 1.0)
    validation_return_mdd = _safe_float(validation.get("return_mdd"))
    train_return_mdd = _safe_float(train.get("return_mdd"))
    validation_sharpe = _safe_float(validation.get("sharpe"))
    validation_smart_sortino = _safe_float(validation.get("smart_sortino"))
    validation_trades = int(validation.get("trade_count") or 0)
    penalty = 0.0
    if train_return <= 0.0:
        penalty += 10.0 + abs(train_return) * 10.0
    if validation_return <= 0.0:
        penalty += 20.0 + abs(validation_return) * 10.0
    if validation_mdd > 0.25:
        penalty += (validation_mdd - 0.25) * 20.0
    if validation_trades < 30:
        penalty += (30 - validation_trades) / 10.0
    return (
        validation_return_mdd
        + 0.25 * train_return_mdd
        + 0.25 * validation_sharpe
        + 0.25 * validation_smart_sortino
        + 0.50 * validation_return
        - penalty
    )


def _default_grid_specs() -> list[_GridSpec]:
    """Narrow, interpretable train/validation-only replay grid.

    The specs avoid calendar/time rules.  The labels document their hypothesis
    source, but every candidate remains a formulaic state/factor variant and is
    scored only on train/validation metrics before locked-OOS is opened.
    """
    return [
        _GridSpec("alpha_zoo_default", "crypto_fx_alpha_zoo_formulaic_default", {}),
        _GridSpec(
            "alpha_zoo_conservative_exit",
            "state_distilled_external_risk_filter_seed",
            {
                "entry_threshold": 0.95,
                "exit_threshold": 0.30,
                "stop_loss_pct": 0.025,
                "take_profit_pct": 0.055,
                "max_hold_bars": 48,
                "risk_off_long_multiplier": 0.10,
                "risk_off_short_multiplier": 1.20,
                "risk_on_long_multiplier": 1.15,
                "risk_on_short_multiplier": 0.35,
            },
        ),
        _GridSpec(
            "alpha_zoo_quality_single_pair",
            "residual_pair_seed",
            {
                "entry_threshold": 1.10,
                "exit_threshold": 0.35,
                "stop_loss_pct": 0.020,
                "take_profit_pct": 0.050,
                "max_hold_bars": 36,
                "max_longs": 1,
                "max_shorts": 1,
                "residual_momentum_weight": 0.45,
                "residual_reversal_weight": 0.20,
                "vwap_pressure_weight": 0.10,
                "breakout_failure_weight": 0.10,
                "trend_efficiency_weight": 0.15,
            },
        ),
        _GridSpec(
            "alpha_zoo_long_risk_on",
            "state_distilled_external_risk_filter_seed",
            {
                "entry_threshold": 0.85,
                "exit_threshold": 0.25,
                "stop_loss_pct": 0.025,
                "take_profit_pct": 0.060,
                "max_hold_bars": 72,
                "allow_shorts": False,
                "max_longs": 2,
                "max_shorts": 0,
                "risk_off_long_multiplier": 0.05,
                "risk_on_long_multiplier": 1.20,
            },
        ),
        _GridSpec(
            "alpha_zoo_fast_residual",
            "residual_pair_seed",
            {
                "fast_lookback_bars": 2,
                "slow_lookback_bars": 18,
                "history_window": 72,
                "entry_threshold": 0.90,
                "exit_threshold": 0.25,
                "stop_loss_pct": 0.025,
                "take_profit_pct": 0.050,
                "max_hold_bars": 36,
                "max_longs": 1,
                "max_shorts": 1,
            },
        ),
        _GridSpec(
            "alpha_zoo_high_confidence_single_pair",
            "state_distilled_external_risk_filter_seed",
            {
                "entry_threshold": 1.35,
                "exit_threshold": 0.45,
                "stop_loss_pct": 0.015,
                "take_profit_pct": 0.035,
                "max_hold_bars": 24,
                "max_longs": 1,
                "max_shorts": 1,
                "risk_off_long_multiplier": 0.05,
                "risk_off_short_multiplier": 1.25,
                "risk_on_long_multiplier": 1.15,
                "risk_on_short_multiplier": 0.25,
            },
        ),
        _GridSpec(
            "alpha_zoo_high_confidence_long_only",
            "state_distilled_external_risk_filter_seed",
            {
                "entry_threshold": 1.25,
                "exit_threshold": 0.40,
                "stop_loss_pct": 0.018,
                "take_profit_pct": 0.040,
                "max_hold_bars": 24,
                "allow_shorts": False,
                "max_longs": 1,
                "max_shorts": 0,
                "risk_off_long_multiplier": 0.05,
                "risk_on_long_multiplier": 1.20,
            },
        ),
        _GridSpec(
            "alpha_zoo_slow_residual_pair",
            "residual_pair_seed",
            {
                "fast_lookback_bars": 6,
                "slow_lookback_bars": 48,
                "history_window": 144,
                "entry_threshold": 1.00,
                "exit_threshold": 0.30,
                "stop_loss_pct": 0.020,
                "take_profit_pct": 0.060,
                "max_hold_bars": 48,
                "max_longs": 1,
                "max_shorts": 1,
                "residual_momentum_weight": 0.40,
                "residual_reversal_weight": 0.25,
                "vwap_pressure_weight": 0.10,
                "breakout_failure_weight": 0.10,
                "trend_efficiency_weight": 0.15,
            },
        ),
        _GridSpec(
            "alpha_zoo_short_risk_off",
            "state_distilled_external_risk_filter_seed",
            {
                "entry_threshold": 1.05,
                "exit_threshold": 0.35,
                "stop_loss_pct": 0.020,
                "take_profit_pct": 0.050,
                "max_hold_bars": 36,
                "max_longs": 0,
                "max_shorts": 2,
                "risk_off_short_multiplier": 1.30,
                "risk_on_short_multiplier": 0.25,
            },
        ),
    ]


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns:
        equity *= 1.0 + float(ret)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def _split_metrics(trades: list[dict[str, Any]], *, leverage: float = 1.0, allocation_fraction: float = 0.10) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "locked_oos"):
        split_trades = [trade for trade in trades if str(trade.get("entry_split")) == split]
        returns = [allocation_fraction * leverage * _safe_float(trade.get("gross_return")) for trade in split_trades]
        total = math.prod([1.0 + item for item in returns]) - 1.0 if returns else 0.0
        mdd = _max_drawdown(returns)
        mean = sum(returns) / len(returns) if returns else 0.0
        variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1) if len(returns) > 1 else 0.0
        sigma = math.sqrt(variance)
        downside = [item for item in returns if item < 0.0]
        down_sigma = math.sqrt(sum(item * item for item in downside) / len(downside)) if downside else 0.0
        sharpe = (mean / sigma * math.sqrt(len(returns))) if sigma > 1e-12 else 0.0
        sortino = (mean / down_sigma * math.sqrt(len(returns))) if down_sigma > 1e-12 else 0.0
        out[split] = {
            "total_return": float(total),
            "max_drawdown": float(mdd),
            "return_mdd": float(total / mdd) if mdd > 1e-12 else (float("inf") if total > 0 else 0.0),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "smart_sortino": float(sortino / (1.0 + mdd)),
            "calmar": float(total / mdd) if mdd > 1e-12 else (float("inf") if total > 0 else 0.0),
            "trade_count": len(split_trades),
        }
    return out


def _symbol_frames(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(symbol): group.sort_values("timestamp") for symbol, group in data.groupby("symbol", sort=False)}


def _trade_path(data: pd.DataFrame, trade: dict[str, Any], symbol_frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    symbol = str(trade["symbol"])
    start = pd.Timestamp(trade["entry_time"])
    end = pd.Timestamp(trade["exit_time"])
    source = (symbol_frames or {}).get(symbol)
    if source is None:
        source = data[data["symbol"].astype(str).eq(symbol)]
    return source[(source["timestamp"].ge(start)) & (source["timestamp"].le(end))]


def _audit_liquidation(
    data: pd.DataFrame,
    trades: list[dict[str, Any]],
    *,
    leverage: float,
    allocation_fraction: float,
    starting_equity: float = 10_000.0,
) -> dict[str, Any]:
    reserve_rate = 0.01 + 0.001 + 0.0005 + 0.0001 + 0.0025 + 0.005
    adverse_threshold = max(0.0, (1.0 / max(leverage, 1e-9)) - reserve_rate)
    split_status = {
        split: {
            "liquidation_count": 0,
            "minimum_margin_buffer": starting_equity,
            "maximum_liquidation_event_drawdown": 0.0,
            "maximum_liquidation_equity_loss_fraction": 0.0,
            "liquidation_recovery_observed": False,
            "recovered_to_pre_liquidation_equity": True,
            "account_wipeout_count": 0,
        }
        for split in ("train", "validation", "locked_oos")
    }
    symbol_frames = _symbol_frames(data)
    for trade in trades:
        split = str(trade.get("entry_split"))
        if split not in split_status:
            continue
        path = _trade_path(data, trade, symbol_frames)
        if path.empty:
            continue
        entry = _safe_float(trade.get("entry_price"), 1.0)
        if entry <= 0.0:
            continue
        if trade.get("side") == "LONG":
            adverse = (pd.to_numeric(path["low"], errors="coerce") / entry) - 1.0
        else:
            adverse = 1.0 - (pd.to_numeric(path["high"], errors="coerce") / entry)
        min_adverse = float(adverse.min()) if len(adverse) else 0.0
        notional = starting_equity * allocation_fraction * leverage
        margin_requirement = notional * reserve_rate
        buffer = starting_equity - margin_requirement + min(0.0, min_adverse * notional)
        split_status[split]["minimum_margin_buffer"] = min(split_status[split]["minimum_margin_buffer"], float(buffer))
        if min_adverse <= -adverse_threshold:
            split_status[split]["liquidation_count"] += 1
            event_loss = min(1.0, abs(min_adverse) * allocation_fraction * leverage)
            split_status[split]["maximum_liquidation_event_drawdown"] = max(
                split_status[split]["maximum_liquidation_event_drawdown"], event_loss
            )
            split_status[split]["maximum_liquidation_equity_loss_fraction"] = max(
                split_status[split]["maximum_liquidation_equity_loss_fraction"], event_loss
            )
            split_status[split]["liquidation_recovery_observed"] = True
            split_status[split]["recovered_to_pre_liquidation_equity"] = False
            if event_loss >= 1.0:
                split_status[split]["account_wipeout_count"] += 1
    for status in split_status.values():
        status["margin_buffer_positive"] = status["minimum_margin_buffer"] > 0.0
        status["strict_safe"] = status["liquidation_count"] == 0 and status["margin_buffer_positive"]
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


def _liquidation_lanes(data: pd.DataFrame, trades: list[dict[str, Any]], *, allocation_fraction: float, max_leverage: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for leverage in range(1, int(max_leverage) + 1):
        metrics = _split_metrics(trades, leverage=float(leverage), allocation_fraction=allocation_fraction)
        audit = _audit_liquidation(data, trades, leverage=float(leverage), allocation_fraction=allocation_fraction)
        oos = metrics["locked_oos"]
        oos_return = _safe_float(oos.get("total_return"))
        oos_mdd = _safe_float(oos.get("max_drawdown"), 1.0)
        return_mdd = oos_return / oos_mdd if oos_mdd > 1e-12 else (float("inf") if oos_return > 0 else 0.0)
        return_mdd_beats_reference = return_mdd > CURRENT_BASE_REFERENCE["locked_oos_return_mdd"]
        performance_gates = {
            "oos_mdd_within_25pct_budget": oos_mdd <= 0.25,
            "oos_return_beats_current_base": oos_return > CURRENT_BASE_REFERENCE["locked_oos_total_return"],
            "oos_sharpe_positive": _safe_float(oos.get("sharpe")) > 0.0,
            "oos_sortino_positive": _safe_float(oos.get("sortino")) > 0.0,
            "oos_smart_sortino_positive": _safe_float(oos.get("smart_sortino")) > 0.0,
            "oos_calmar_positive": _safe_float(oos.get("calmar")) > 0.0,
        }
        performance_diagnostics = {
            "oos_return_mdd": return_mdd,
            "current_base_reference_return_mdd": CURRENT_BASE_REFERENCE["locked_oos_return_mdd"],
            "oos_return_mdd_beats_current_base": return_mdd_beats_reference,
            "return_mdd_hurdle_required": False,
            "return_mdd_role": "diagnostic_report_only",
            "risk_adjusted_metrics_used_instead": ["sharpe", "sortino", "smart_sortino", "calmar", "max_drawdown"],
        }
        strict_safe = bool(audit["liquidation_free"] and audit["margin_buffer_positive"])
        deployable = strict_safe and all(performance_gates.values())
        rows.append(
            {
                "candidate_name": "crypto_fx_alpha_zoo_state_calibrated",
                "candidate_source": "alpha_zoo_train_validation_calibrated_replay",
                "strategy": "CryptoFxAlphaZooStateStrategy",
                "leverage": float(leverage),
                "selection_inputs": ["train", "validation"],
                "uses_locked_oos_for_selection": False,
                "locked_oos_role": "gate_report_only_after_candidate_freeze",
                "split_metrics": metrics,
                "liquidation_audit": audit,
                "promotion_policy": PROMOTION_POLICY,
                "performance_gates": performance_gates,
                "performance_diagnostics": performance_diagnostics,
                "strict_safe": strict_safe,
                "deployable_success": bool(deployable),
            }
        )
    strict_candidates = [row for row in rows if bool(row["strict_safe"])]
    deployable = [row for row in rows if bool(row["deployable_success"])]
    diagnostic = [
        {
            "candidate_name": row["candidate_name"],
            "leverage": row["leverage"],
            "diagnostic_only": True,
            "promotion_allowed": False,
            "promotion_eligible": False,
            "separate_from_strict_deploy": True,
            "split_metrics": row["split_metrics"],
            "split_diagnostics": row["liquidation_audit"]["split_status"],
            "total_liquidation_count": row["liquidation_audit"]["total_liquidation_count"],
            "minimum_margin_buffer": row["liquidation_audit"]["minimum_margin_buffer"],
        }
        for row in rows
        if int(row["leverage"]) in {5, 6}
    ]
    return {
        "integer_grid_results": rows,
            "strict_zero_liquidation_lane": {
                "lane": "strict_deploy",
                "requires_liquidation_count_zero": True,
                "requires_positive_min_margin_buffer": True,
                "promotion_policy": PROMOTION_POLICY,
                "promotion_rule": "train/validation/locked-OOS liquidation_count must be zero, every split minimum margin buffer must be positive, OOS MDD must stay within 25%, OOS return must beat the current-base reference, and Sharpe/Sortino/smart Sortino/Calmar must be positive. Return/MDD is diagnostic report-only, not a promotion hurdle.",
                "candidate_count": len(strict_candidates),
                "deployable_candidate_count": len(deployable),
                "highest_zero_liquidation_integer": max(strict_candidates, key=lambda row: row["leverage"], default={}),
            "promoted_candidate": max(deployable, key=lambda row: row["leverage"], default={}),
        },
        "diagnostic_nonfatal_lane": {
            "lane": "diagnostic_nonfatal_5x_6x",
            "diagnostic_only": True,
            "live_promotion_lane": "strict_deploy_lane_only",
            "high_leverage_5x_6x_report": diagnostic,
        },
    }


def replay_frame(
    frame: pd.DataFrame,
    *,
    require_calibrated_edge: bool = True,
    calibrated_edges: dict[str, float] | None = None,
    strategy_params: dict[str, Any] | None = None,
    max_leverage: int = 6,
    allocation_fraction: float = 0.10,
    source_metadata: dict[str, Any] | None = None,
    grid_specs: list[_GridSpec] | None = None,
) -> dict[str, Any]:
    data = _ensure_replay_frame(frame)
    edge_map = dict(calibrated_edges or {})
    specs = list(grid_specs or [_GridSpec("alpha_zoo_default", "crypto_fx_alpha_zoo_formulaic_default", dict(strategy_params or {}))])
    grid_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_signals: list[dict[str, Any]] = []
    selected_trades: list[dict[str, Any]] = []
    for spec in specs:
        params = {**dict(strategy_params or {}), **dict(spec.params)}
        candidate_signals = _run_strategy_signals(
            data,
            require_calibrated_edge=require_calibrated_edge,
            calibrated_edges=edge_map,
            strategy_params=params,
        )
        candidate_trades = _build_trades(data, candidate_signals)
        tv_metrics = _train_validation_metrics(candidate_trades, leverage=1.0, allocation_fraction=allocation_fraction)
        score = _selection_score(tv_metrics)
        row = {
            "candidate_name": spec.name,
            "candidate_source": spec.source,
            "strategy": "CryptoFxAlphaZooStateStrategy",
            "selection_score": float(score),
            "selection_metrics": tv_metrics,
            "signal_count": len(candidate_signals),
            "trade_count": len(candidate_trades),
            "params": params,
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "locked_oos_metrics_visible_during_selection": False,
        }
        grid_rows.append(row)
        if selected is None or score > float(selected["selection_score"]):
            selected = row
            selected_signals = candidate_signals
            selected_trades = candidate_trades
    signals = selected_signals
    trades = selected_trades
    unlevered_metrics = _split_metrics(trades, leverage=1.0, allocation_fraction=allocation_fraction)
    lanes = _liquidation_lanes(data, trades, allocation_fraction=allocation_fraction, max_leverage=max_leverage)
    oos = unlevered_metrics["locked_oos"]
    return_mdd = _safe_float(oos.get("return_mdd"))
    deployable = bool(lanes["strict_zero_liquidation_lane"].get("promoted_candidate"))
    return {
        "artifact_kind": "crypto_fx_alpha_zoo_state_real_data_replay",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "row_count": len(data),
        "signal_count": len(signals),
        "trade_count": len(trades),
        "signals": signals[:500],
        "trades": [
            {
                **trade,
                "entry_time": pd.Timestamp(trade["entry_time"]).isoformat(),
                "exit_time": pd.Timestamp(trade["exit_time"]).isoformat(),
            }
            for trade in trades[:500]
        ],
        "split_counts": dict(Counter(str(item) for item in data["split"])),
        "unlevered_split_metrics": unlevered_metrics,
        "candidate_selection_grid": {
            "grid_profile": "narrow_train_validation_formulaic",
            "selection_policy": "rank by train/validation-only score; locked-OOS hidden until candidate freeze",
            "candidate_count": len(grid_rows),
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "locked_oos_calibration_record_count": 0,
            "selected_candidate_name": (selected or {}).get("candidate_name", ""),
            "selected_candidate_source": (selected or {}).get("candidate_source", ""),
            "selected_candidate_params": (selected or {}).get("params", {}),
            "rows": sorted(grid_rows, key=lambda row: float(row["selection_score"]), reverse=True),
        },
        **lanes,
        "strategy_validity": CryptoFxAlphaZooStateStrategy.strategy_validity,
        "selection_provenance": {
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
            "candidate_freeze_before_locked_oos_gate": True,
            "current_base_calendar_tuple_role": "hypothesis_reference_only",
            "selection_excludes_current_base_calendar_tuple": True,
        },
        "calibration_provenance": {
            "require_calibrated_edge": bool(require_calibrated_edge),
            "calibrated_edge_count": len(edge_map),
            "calibrated_edge_keys": sorted(edge_map),
        },
        "current_base_reference": CURRENT_BASE_REFERENCE,
        "promotion_policy": PROMOTION_POLICY,
        "deployable_success": deployable,
        "deployable_success_reason": (
            "strict zero-liquidation lane passed revised OOS return, MDD, Sharpe/Sortino/Calmar gates; return/MDD is diagnostic-only"
            if deployable
            else "no strict zero-liquidation Alpha Zoo replay row passed the revised OOS return, MDD, Sharpe/Sortino/Calmar gates"
        ),
        "locked_oos_report_only_metrics": {
            "candidate_oos_return": _safe_float(oos.get("total_return")),
            "candidate_oos_return_mdd": return_mdd,
            "current_base_oos_return": CURRENT_BASE_REFERENCE["locked_oos_total_return"],
            "current_base_oos_return_mdd": CURRENT_BASE_REFERENCE["locked_oos_return_mdd"],
            "return_mdd_hurdle_required": False,
            "return_mdd_role": "diagnostic_report_only",
        },
        "memory_summary": {"peak_rss_mib": _rss_mib(), "limit_mib": 8192.0, "pass_under_8gb": _rss_mib() < 8192.0},
        "source_coverage": source_metadata or {},
        "uses_locked_oos_for_selection": False,
        "calendar_primary": False,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    strict = dict(payload.get("strict_zero_liquidation_lane") or {})
    diag = dict(payload.get("diagnostic_nonfatal_lane") or {})
    oos = dict(payload.get("locked_oos_report_only_metrics") or {})
    lines = [
        "# Crypto/FX Alpha Zoo state real-data replay",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        f"Rows: `{payload.get('row_count')}`",
        f"Signals: `{payload.get('signal_count')}`",
        f"Trades: `{payload.get('trade_count')}`",
        f"Deployable success: `{payload.get('deployable_success')}`",
        f"Reason: {payload.get('deployable_success_reason')}",
        "",
        "## Provenance",
        "",
        "- selection inputs: `train, validation`",
        "- uses_locked_oos_for_selection: `False`",
        "- locked-OOS role: `gate/report only after candidate freeze`",
        "- current-base/calendar tuple: `hypothesis_reference_only`, not selection/promotion target",
        "- promotion policy: return/MDD is `diagnostic_report_only`; Sharpe/Sortino/smart Sortino/Calmar and MDD cap carry the risk-adjusted gate",
        "",
        "## Locked-OOS report-only comparison",
        "",
        f"- candidate OOS return: `{_safe_float(oos.get('candidate_oos_return')):.4%}`",
        f"- candidate OOS return/MDD: `{_safe_float(oos.get('candidate_oos_return_mdd')):.6f}`",
        f"- current-base reference OOS return: `{_safe_float(oos.get('current_base_oos_return')):.4%}`",
        f"- current-base reference return/MDD: `{_safe_float(oos.get('current_base_oos_return_mdd')):.6f}`",
        "",
        "## Strict zero-liquidation lane",
        "",
        f"- strict candidate count: `{strict.get('candidate_count', 0)}`",
        f"- deployable candidate count: `{strict.get('deployable_candidate_count', 0)}`",
        "",
        "## Diagnostic nonfatal 5x/6x lane",
        "",
    ]
    for item in list(diag.get("high_leverage_5x_6x_report") or []):
        lines.append(
            f"- `{item.get('leverage')}x`: total_liquidations `{item.get('total_liquidation_count')}`, "
            f"min_margin_buffer `{_safe_float(item.get('minimum_margin_buffer')):.4f}`, promotion_allowed `False`"
        )
    lines.extend(
        [
            "",
            "## Memory",
            "",
            f"- peak_rss_mib: `{_safe_float(dict(payload.get('memory_summary') or {}).get('peak_rss_mib')):.3f}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="")
    parser.add_argument("--current-tail-cache", default="")
    parser.add_argument("--external-state-csv", default="")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--require-calibrated-edge", action="store_true", default=True)
    parser.add_argument("--allow-uncalibrated-edge", action="store_true")
    parser.add_argument("--strict-real-data", action="store_true")
    parser.add_argument("--max-leverage", type=int, default=6)
    parser.add_argument("--allocation-fraction", type=float, default=0.10)
    parser.add_argument("--grid-profile", choices=("none", "narrow"), default="narrow")
    args = parser.parse_args()

    bundle = load_real_data_bundle(
        input_path=args.input,
        current_tail_cache=args.current_tail_cache,
        external_state_csv=args.external_state_csv,
        strict_real_data=bool(args.strict_real_data),
    )
    edges = _load_calibrated_edges(args.calibration)
    payload = replay_frame(
        bundle.frame,
        require_calibrated_edge=not bool(args.allow_uncalibrated_edge),
        calibrated_edges=edges,
        max_leverage=max(1, int(args.max_leverage)),
        allocation_fraction=max(0.0, float(args.allocation_fraction)),
        source_metadata=bundle.metadata,
        grid_specs=_default_grid_specs() if args.grid_profile == "narrow" else None,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_markdown(payload, output.with_suffix(".md"))


if __name__ == "__main__":
    main()
