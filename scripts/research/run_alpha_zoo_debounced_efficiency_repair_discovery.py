#!/usr/bin/env python3
"""Repair ETH/SOL debounced Alpha Zoo execution efficiency on local real data.

This runner is research/paper-testnet only. It continues the diverse train-dominant
Alpha Zoo pass by focusing on ETH/SOL debounced momentum hysteresis variants and
tries to improve return-per-turnover with fewer, stronger state transitions. It
ranks with train+validation evidence only and attaches locked-OOS strictly after
candidate freeze as gate/report evidence. It never executes orders and never
enables real money.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
from collections.abc import Iterable, Mapping, Sequence
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
    PRIMARY_ROUND_TRIP_COST_BPS,
    RETURN_PER_TURNOVER_THRESHOLD_BPS,
    SPLIT_ORDER,
    SPLITS,
    SimResult,
    _json_safe,
    _split_mask,
    load_hourly_bars,
    resample_bars,
    simulate_symbol,
    split_metrics,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_debounced_efficiency_repair_discovery_20260523"
)
DEFAULT_PRIOR_ARTIFACT = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_diverse_train_dominant_discovery_20260522/"
    "alpha_zoo_diverse_train_dominant_discovery_latest.json"
)
DEFAULT_SYMBOLS = ("ETHUSDT", "SOLUSDT")
DEFAULT_TIMEFRAMES = ("1h", "2h", "4h")
BTC_REGIME_SYMBOL = "BTCUSDT"

PROMOTION_THRESHOLDS = {
    "min_train_trade_event_count": 80,
    "min_validation_trade_event_count": 30,
    "min_locked_oos_trade_event_count_report_gate": 20,
    "min_validation_return": 0.02,
    "require_train_return_positive": True,
    "require_train_return_gte_validation_return": True,
    "min_train_validation_return_ratio": 1.0,
    "max_validation_mdd": 0.12,
    "require_locked_oos_return_positive_report_gate": True,
    "require_zero_locked_oos_liquidation": True,
    "require_zero_locked_oos_account_wipeout": True,
    "min_return_per_turnover_proxy_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
    "primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
    "require_replay_live_notional_parity_recorded": True,
}

BASELINE_LANES = [
    {
        "lane": "active",
        "model_id": "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc",
        "leverage": 7.0,
        "allocation_fraction": 0.20,
        "notional_fraction": 1.40,
        "status": "baseline_preserved",
        "ready_for_real": False,
        "real_money_execution": False,
    },
    {
        "lane": "balanced",
        "model_id": "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc",
        "leverage": 6.0,
        "allocation_fraction": 0.175,
        "notional_fraction": 1.05,
        "status": "baseline_preserved",
        "ready_for_real": False,
        "real_money_execution": False,
    },
    {
        "lane": "validation_leader",
        "model_id": "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc",
        "leverage": 5.0,
        "allocation_fraction": 0.20,
        "notional_fraction": 1.00,
        "status": "baseline_preserved",
        "ready_for_real": False,
        "real_money_execution": False,
    },
    {
        "lane": "efficiency_reference",
        "model_id": "fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc",
        "leverage": 4.0,
        "allocation_fraction": 0.175,
        "notional_fraction": 0.70,
        "status": "baseline_preserved",
        "ready_for_real": False,
        "real_money_execution": False,
    },
]

TARGET_PRIOR_CANDIDATES = [
    {
        "model_id": "divtd_debounced_hysteresis_trend_1h_ethusdt_long_short_lb6_e0p02_x0p005_hold12_3p0x_0p15_5ea41145",
        "repair_focus": "raise train and locked-OOS return-per-turnover via fewer transitions",
        "train_return": 0.250916,
        "validation_return": 0.235473,
        "locked_oos_return": 0.013122,
        "trades": "596/147/55",
        "train_return_per_turnover_proxy_bps": 9.356,
        "locked_oos_return_per_turnover_proxy_bps": 5.302,
    },
    {
        "model_id": "divtd_debounced_hysteresis_trend_1h_solusdt_short_only_lb6_e0p02_x0p005_hold12_3p0x_0p15_08f4d887",
        "repair_focus": "preserve train dominance while lifting locked-OOS RPT above 10bps",
        "train_return": 0.355640,
        "validation_return": 0.099201,
        "locked_oos_return": 0.005645,
        "trades": "428/100/34",
        "locked_oos_return_per_turnover_proxy_bps": 3.690,
    },
]

CANDIDATE_FIELDS = [
    "rank",
    "model_id",
    "family",
    "symbol",
    "timeframe",
    "side",
    "lookback_bars",
    "entry_threshold",
    "exit_threshold",
    "min_hold_bars",
    "cooldown_bars",
    "vol_filter_mode",
    "vol_quantile_max",
    "adx_threshold",
    "trend_strength_threshold",
    "leverage",
    "allocation_fraction",
    "notional_fraction",
    "train_return",
    "validation_return",
    "locked_oos_return",
    "train_mdd",
    "validation_mdd",
    "locked_oos_mdd",
    "train_sharpe",
    "validation_sharpe",
    "locked_oos_sharpe",
    "train_trade_event_count",
    "validation_trade_event_count",
    "locked_oos_trade_event_count",
    "train_exposure_bar_count",
    "validation_exposure_bar_count",
    "locked_oos_exposure_bar_count",
    "train_validation_return_ratio",
    "train_minus_validation_return",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "locked_oos_return_per_turnover_proxy_bps",
    "train_dominant_sample_gate_pass",
    "execution_efficiency_proxy_gate_pass",
    "primary_10bps_promotion_gate_pass",
    "paper_candidate_gate_pass",
    "decision",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "replay_live_notional_parity",
    "locked_oos_liquidation_count",
    "locked_oos_account_wipeout_count",
    "rejection_reasons",
]

DECISION_FIELDS = [
    "decision_rank",
    "model_id",
    "decision",
    "family",
    "symbol",
    "timeframe",
    "train_return",
    "validation_return",
    "locked_oos_return",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "rejection_reasons",
]

SPECIAL_OUTPUT_FLAGS = (
    "train_dominant_sample_gate_pass",
    "execution_efficiency_proxy_gate_pass",
    "paper_candidate_gate_pass",
)


@dataclass(frozen=True)
class FilterProfile:
    name: str
    vol_quantile_max: float | None
    adx_threshold: float
    trend_strength_threshold: float


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
        return ";".join(str(v) for v in value)
    return _json_safe(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def _model_id(parts: Iterable[Any]) -> str:
    text = "_".join(str(part).replace("/", "_").replace(".", "p") for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"debrepair_{text}_{digest}".lower()


def _return_per_turnover(total_return: float, trade_events: int, notional_fraction: float) -> float | None:
    turnover = float(trade_events) * abs(float(notional_fraction))
    if turnover <= 0:
        return None
    return float(total_return) * 10000.0 / turnover


def _candidate_score(row: Mapping[str, Any]) -> float:
    """Train+validation-only score; locked-OOS is intentionally ignored."""
    train = float(row.get("train_return") or 0.0)
    validation = float(row.get("validation_return") or 0.0)
    val_mdd = float(row.get("validation_mdd") or 0.0)
    train_rpt = float(row.get("train_return_per_turnover_proxy_bps") or 0.0)
    val_rpt = float(row.get("validation_return_per_turnover_proxy_bps") or 0.0)
    train_shortfall_penalty = max(0.0, validation - train)
    efficiency_bonus = min(train_rpt, val_rpt, RETURN_PER_TURNOVER_THRESHOLD_BPS * 2.0) / 100.0
    turnover_penalty = float(row.get("validation_trade_event_count") or 0.0) / 5000.0
    return (
        4.0 * validation
        + 1.5 * min(train, validation)
        + efficiency_bonus
        - 6.0 * train_shortfall_penalty
        - 2.0 * val_mdd
        - turnover_penalty
    )


def _gate_candidate(row: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    train = float(row["train_return"])
    validation = float(row["validation_return"])
    ratio = row.get("train_validation_return_ratio")
    if int(row["train_trade_event_count"]) < PROMOTION_THRESHOLDS["min_train_trade_event_count"]:
        reasons.append(f"train_trade_event_count_{row['train_trade_event_count']}_below_80")
    if int(row["validation_trade_event_count"]) < PROMOTION_THRESHOLDS["min_validation_trade_event_count"]:
        reasons.append(f"validation_trade_event_count_{row['validation_trade_event_count']}_below_30")
    if int(row["locked_oos_trade_event_count"]) < PROMOTION_THRESHOLDS["min_locked_oos_trade_event_count_report_gate"]:
        reasons.append(f"locked_oos_trade_event_count_{row['locked_oos_trade_event_count']}_below_20")
    if validation < PROMOTION_THRESHOLDS["min_validation_return"]:
        reasons.append(f"validation_return_{validation:.4f}_below_0.02")
    if train <= 0:
        reasons.append("train_return_not_positive")
    if train < validation:
        reasons.append(f"train_return_{train:.4f}_below_validation_return_{validation:.4f}")
    if ratio is None or float(ratio) < PROMOTION_THRESHOLDS["min_train_validation_return_ratio"]:
        reasons.append(f"train_validation_return_ratio_{0.0 if ratio is None else ratio:.4f}_below_1.00")
    if float(row["validation_mdd"]) > PROMOTION_THRESHOLDS["max_validation_mdd"]:
        reasons.append(f"validation_mdd_{row['validation_mdd']:.4f}_above_0.12")
    if float(row["locked_oos_return"]) <= 0:
        reasons.append("locked_oos_return_not_positive")
    if int(row["locked_oos_liquidation_count"]) != 0:
        reasons.append("locked_oos_liquidation_count_nonzero")
    if int(row["locked_oos_account_wipeout_count"]) != 0:
        reasons.append("locked_oos_account_wipeout_count_nonzero")
    train_dominant_sample_gate_pass = not reasons

    efficiency_reasons: list[str] = []
    for split in SPLIT_ORDER:
        value = row.get(f"{split}_return_per_turnover_proxy_bps")
        if value is None or float(value) <= RETURN_PER_TURNOVER_THRESHOLD_BPS:
            rendered = "missing" if value is None else f"{float(value):.3f}"
            efficiency_reasons.append(
                f"{split}_return_per_turnover_proxy_bps_{rendered}_not_above_"
                f"{RETURN_PER_TURNOVER_THRESHOLD_BPS:.3f}"
            )
    execution_efficiency_proxy_gate_pass = not efficiency_reasons
    paper_candidate_gate_pass = train_dominant_sample_gate_pass and execution_efficiency_proxy_gate_pass
    if paper_candidate_gate_pass:
        decision = "paper_testnet_candidate_after_fill_preflight"
    elif train_dominant_sample_gate_pass:
        decision = "train_dominant_shadow_until_execution_efficiency"
    else:
        decision = "no_promotion_shadow_or_reject"
    row.update(
        {
            "train_dominant_sample_gate_pass": train_dominant_sample_gate_pass,
            "execution_efficiency_proxy_gate_pass": execution_efficiency_proxy_gate_pass,
            "primary_10bps_promotion_gate_pass": paper_candidate_gate_pass,
            "paper_candidate_gate_pass": paper_candidate_gate_pass,
            "decision": decision,
            "ready_for_paper": paper_candidate_gate_pass,
            "ready_for_real": False,
            "real_money_execution": False,
            "rejection_reasons": reasons + efficiency_reasons,
        }
    )
    return row


def _finalize_candidate(base: dict[str, Any], sim: SimResult, datetimes: pd.Series, *, timeframe: str) -> dict[str, Any]:
    row = dict(base)
    split_payload: dict[str, dict[str, Any]] = {}
    for split in SPLIT_ORDER:
        mask = _split_mask(datetimes, split)
        metrics = split_metrics(
            sim.returns[mask],
            sim.position[mask],
            sim.liquidation_flags[mask],
            sim.account_wipeout_flags[mask],
            timeframe=timeframe,
        )
        split_payload[split] = metrics
        row[f"{split}_return"] = metrics["total_return"]
        row[f"{split}_mdd"] = metrics["max_drawdown"]
        row[f"{split}_sharpe"] = metrics["sharpe"]
        row[f"{split}_sortino"] = metrics["sortino"]
        row[f"{split}_calmar"] = metrics["calmar"]
        row[f"{split}_trade_event_count"] = metrics["trade_event_count"]
        row[f"{split}_exposure_bar_count"] = metrics["exposure_bar_count"]
    validation = float(row.get("validation_return") or 0.0)
    train = float(row.get("train_return") or 0.0)
    row["train_validation_return_ratio"] = train / validation if validation > 0 else 0.0
    row["train_minus_validation_return"] = train - validation
    row["locked_oos_liquidation_count"] = int(split_payload["locked_oos"]["liquidation_count"])
    row["locked_oos_account_wipeout_count"] = int(split_payload["locked_oos"]["account_wipeout_count"])
    notional = float(row["notional_fraction"])
    for split in SPLIT_ORDER:
        row[f"{split}_return_per_turnover_proxy_bps"] = _return_per_turnover(
            float(row[f"{split}_return"]), int(row[f"{split}_trade_event_count"]), notional
        )
    row["replay_live_notional_parity"] = {
        "recorded": True,
        "sizing_mode": "notional_fraction_equals_leverage_times_allocation_fraction",
        "replay_notional_fraction": notional,
        "live_notional_fraction": notional,
        "parity": True,
    }
    row["train_validation_score"] = _candidate_score(row)
    return _gate_candidate(row)


def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high.index,
    )
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    tr_sum = true_range.rolling(lookback).sum().replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.rolling(lookback).sum() / tr_sum
    minus_di = 100.0 * minus_dm.rolling(lookback).sum() / tr_sum
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.rolling(lookback).mean()


def _volatility_mask(close: pd.Series, lookback: int, quantile_max: float | None) -> pd.Series:
    if quantile_max is None:
        return pd.Series(True, index=close.index)
    realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
    rolling_threshold = realized.rolling(max(24, lookback * 4)).quantile(quantile_max)
    return (realized <= rolling_threshold).fillna(False)


def _debounced_state_signal(
    long_entry: pd.Series,
    long_exit: pd.Series,
    short_entry: pd.Series | None = None,
    short_exit: pd.Series | None = None,
    *,
    side: str,
    min_hold_bars: int,
    cooldown_bars: int,
) -> np.ndarray:
    long_entry = long_entry.fillna(False).astype(bool)
    long_exit = long_exit.fillna(False).astype(bool)
    short_entry = pd.Series(False, index=long_entry.index) if short_entry is None else short_entry.fillna(False).astype(bool)
    short_exit = pd.Series(False, index=long_entry.index) if short_exit is None else short_exit.fillna(False).astype(bool)
    out = np.zeros(len(long_entry), dtype=float)
    state = 0.0
    bars_held = 10**9
    cooldown_remaining = 0
    for i in range(len(long_entry)):
        can_exit = bars_held >= min_hold_bars
        exited = False
        long_exit_now = state > 0 and bool(long_exit.iloc[i])
        short_exit_now = state < 0 and bool(short_exit.iloc[i])
        if can_exit and (long_exit_now or short_exit_now):
            state = 0.0
            bars_held = 0
            cooldown_remaining = cooldown_bars
            exited = True

        if state == 0.0:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif not exited:
                if side in {"long_only", "long_short"} and bool(long_entry.iloc[i]):
                    state = 1.0
                    bars_held = 0
                elif side in {"short_only", "long_short"} and bool(short_entry.iloc[i]):
                    state = -1.0
                    bars_held = 0

        out[i] = state
        if state != 0.0:
            bars_held += 1
    return out


def _filter_profiles() -> tuple[FilterProfile, ...]:
    return (
        FilterProfile("none", None, 0.0, 0.0),
        FilterProfile("low_vol_q65", 0.65, 0.0, 0.0),
        FilterProfile("low_vol_q55_adx15", 0.55, 15.0, 0.0),
        FilterProfile("adx20", None, 20.0, 0.0),
        FilterProfile("trend_strength2", None, 0.0, 2.0),
    )


def discover_repair_candidates(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    leverage_allocs = [(2.0, 0.15), (3.0, 0.10), (3.0, 0.15), (4.0, 0.10)]
    entry_exit_profiles = [(0.02, 0.005), (0.025, 0.0), (0.03, 0.0), (0.03, -0.005), (0.04, -0.005)]
    min_holds = (12, 18, 24, 36, 48)
    cooldowns = (0, 6, 12)
    lookbacks = (6, 12)
    side_map = {
        "ETHUSDT": ("long_short", "short_only"),
        "SOLUSDT": ("short_only", "long_short"),
    }

    for timeframe in timeframes:
        btc = bars_by_symbol_tf[(BTC_REGIME_SYMBOL, timeframe)][["datetime", "close"]].rename(
            columns={"close": "btc_close"}
        )
        for symbol in symbols:
            frame = bars_by_symbol_tf[(symbol, timeframe)].merge(btc, on="datetime", how="left").ffill()
            close = frame["close"].astype(float)
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            btc_close = frame["btc_close"].astype(float)
            btc_regime_fast = btc_close / btc_close.shift(12) - 1.0
            for lookback in lookbacks:
                momentum = close / close.shift(lookback) - 1.0
                realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
                trend_strength = momentum.abs() / (realized * np.sqrt(float(lookback))).replace(0.0, np.nan)
                adx = _adx_proxy(high, low, close, max(6, lookback))
                for entry_threshold, exit_threshold in entry_exit_profiles:
                    for min_hold_bars in min_holds:
                        for cooldown_bars in cooldowns:
                            for profile in _filter_profiles():
                                vol_ok = _volatility_mask(close, lookback, profile.vol_quantile_max)
                                adx_ok = adx >= profile.adx_threshold if profile.adx_threshold > 0 else True
                                strength_ok = (
                                    trend_strength >= profile.trend_strength_threshold
                                    if profile.trend_strength_threshold > 0
                                    else True
                                )
                                common_filter = vol_ok & adx_ok & strength_ok
                                long_entry = (
                                    (momentum > entry_threshold)
                                    & (btc_regime_fast > -0.02)
                                    & common_filter
                                )
                                short_entry = (
                                    (momentum < -entry_threshold)
                                    & (btc_regime_fast < 0.02)
                                    & common_filter
                                )
                                long_exit = (momentum < exit_threshold) | (~common_filter)
                                short_exit = (momentum > -exit_threshold) | (~common_filter)
                                for side in side_map.get(symbol, ("long_short",)):
                                    signal = _debounced_state_signal(
                                        long_entry,
                                        long_exit,
                                        short_entry,
                                        short_exit,
                                        side=side,
                                        min_hold_bars=min_hold_bars,
                                        cooldown_bars=cooldown_bars,
                                    )
                                    for leverage, allocation in leverage_allocs:
                                        sim = simulate_symbol(
                                            frame,
                                            signal,
                                            leverage=leverage,
                                            allocation_fraction=allocation,
                                        )
                                        base = {
                                            "model_id": _model_id(
                                                [
                                                    "debounced_efficiency_repair",
                                                    timeframe,
                                                    symbol,
                                                    side,
                                                    f"lb{lookback}",
                                                    f"e{entry_threshold}",
                                                    f"x{exit_threshold}",
                                                    f"hold{min_hold_bars}",
                                                    f"cool{cooldown_bars}",
                                                    profile.name,
                                                    f"{leverage}x",
                                                    allocation,
                                                ]
                                            ),
                                            "family": "debounced_momentum_hysteresis_efficiency_repair",
                                            "symbol": symbol,
                                            "timeframe": timeframe,
                                            "side": side,
                                            "lookback_bars": lookback,
                                            "entry_threshold": entry_threshold,
                                            "exit_threshold": exit_threshold,
                                            "min_hold_bars": min_hold_bars,
                                            "cooldown_bars": cooldown_bars,
                                            "vol_filter_mode": profile.name,
                                            "vol_quantile_max": profile.vol_quantile_max,
                                            "adx_threshold": profile.adx_threshold,
                                            "trend_strength_threshold": profile.trend_strength_threshold,
                                            "leverage": leverage,
                                            "allocation_fraction": allocation,
                                            "notional_fraction": leverage * allocation,
                                        }
                                        rows.append(
                                            _finalize_candidate(
                                                base,
                                                sim,
                                                frame["datetime"],
                                                timeframe=timeframe,
                                            )
                                        )
    return rows


def _rank_rows(rows: Sequence[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row.get("train_validation_score") or -1e9), reverse=True)
    if limit is not None:
        ranked = ranked[:limit]
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = rank
        out.append(item)
    return out


def _selected_output_rows(ranked_rows: Sequence[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    selected_ids = {str(row["model_id"]) for row in ranked_rows[:top_n]}
    for row in ranked_rows:
        if any(bool(row.get(flag)) for flag in SPECIAL_OUTPUT_FLAGS):
            selected_ids.add(str(row["model_id"]))
    if ranked_rows:
        best_validation = max(ranked_rows, key=lambda row: float(row.get("validation_return") or -1e9))
        selected_ids.add(str(best_validation["model_id"]))
    return [dict(row) for row in ranked_rows if str(row["model_id"]) in selected_ids]


def _decision_rows(rows: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    decisions = []
    for row in _selected_output_rows(_rank_rows(rows), top_n=limit):
        decisions.append(
            {
                "decision_rank": row["rank"],
                "model_id": row["model_id"],
                "decision": row["decision"],
                "family": row["family"],
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "train_return": row["train_return"],
                "validation_return": row["validation_return"],
                "locked_oos_return": row["locked_oos_return"],
                "ready_for_paper": row["ready_for_paper"],
                "ready_for_real": False,
                "real_money_execution": False,
                "rejection_reasons": row["rejection_reasons"],
            }
        )
    return decisions


def _shadow_rows(ranked_rows: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    shadows = [row for row in ranked_rows if row.get("decision") != "paper_testnet_candidate_after_fill_preflight"]
    sample_shadows = [row for row in shadows if row.get("train_dominant_sample_gate_pass")]
    if sample_shadows:
        return _selected_output_rows(sample_shadows, top_n=limit)
    return _selected_output_rows(shadows, top_n=limit)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    by_symbol: dict[str, int] = {}
    for row in rows:
        decisions[str(row["decision"])] = decisions.get(str(row["decision"]), 0) + 1
        by_symbol[str(row["symbol"])] = by_symbol.get(str(row["symbol"]), 0) + 1
    best_validation = max(rows, key=lambda row: float(row.get("validation_return") or -1e9)) if rows else {}
    best_gate = next((row for row in _rank_rows(rows) if row.get("train_dominant_sample_gate_pass")), None)
    paper_candidates = [row for row in rows if row.get("paper_candidate_gate_pass")]
    return {
        "candidate_count": len(rows),
        "symbol_counts": dict(sorted(by_symbol.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "train_return_gte_validation_return_count": sum(
            float(row.get("train_return") or 0.0) >= float(row.get("validation_return") or 0.0)
            for row in rows
        ),
        "train_dominant_sample_gate_pass_count": sum(
            bool(row.get("train_dominant_sample_gate_pass")) for row in rows
        ),
        "execution_efficiency_proxy_gate_pass_count": sum(
            bool(row.get("execution_efficiency_proxy_gate_pass")) for row in rows
        ),
        "paper_candidate_gate_pass_count": len(paper_candidates),
        "max_validation_return": float(best_validation.get("validation_return") or 0.0) if best_validation else None,
        "best_validation_model_id": best_validation.get("model_id") if best_validation else None,
        "best_train_dominant_sample_gate_model_id": best_gate.get("model_id") if best_gate else None,
        "best_paper_candidate_model_id": paper_candidates[0].get("model_id") if paper_candidates else None,
        "ready_for_paper": bool(paper_candidates),
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _paper_testnet_handoff(paper_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [dict(row) for row in paper_rows]
    return {
        "handoff_kind": "paper_testnet_only_debounced_efficiency_repair",
        "status": "paper_testnet_candidates_available" if candidates else "no_paper_candidates",
        "ready_for_paper": bool(candidates),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": bool(candidates),
        "real_execution_allowed": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "preflight": {
            "required_mode": "paper_or_testnet_only",
            "ready_for_real": False,
            "real_money_execution": False,
            "check_replay_live_notional_parity": True,
            "check_liquidation_account_wipeout": True,
            "check_realized_all_in_cost_bps_mean_lte": PRIMARY_ROUND_TRIP_COST_BPS,
            "check_realized_all_in_cost_bps_p95_lte": 15.0,
        },
        "monitoring_contract": {
            "record_realized_fee_bps": True,
            "record_realized_slippage_bps": True,
            "record_all_in_round_trip_bps": True,
            "record_bbo_spread_bps_at_submit": True,
            "record_liquidation_inclusive_mdd": True,
            "record_account_wipeout": True,
            "primary_research_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
            "return_per_turnover_gate_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
        },
    }


def _handoff_markdown(handoff: Mapping[str, Any]) -> str:
    candidates = list(handoff.get("candidates") or [])
    lines = [
        "# Paper/testnet handoff — debounced efficiency repair",
        "",
        f"- Status: `{handoff.get('status')}`",
        f"- Candidate count: `{handoff.get('candidate_count')}`",
        f"- `ready_for_paper={str(handoff.get('ready_for_paper')).lower()}`",
        "- `ready_for_real=false`",
        "- `real_money_execution=false`",
        "- Real-money execution remains prohibited; this handoff is paper/testnet-only.",
        "",
        "## Preflight contract",
        "",
        "- Required mode: paper/testnet only.",
        "- Confirm replay/live notional parity before observation.",
        "- Confirm liquidation/account-wipeout telemetry fields are wired into monitoring.",
        "- Record realized fee, slippage, all-in round-trip cost, BBO spread at submit, and notional.",
        "",
        "## Top candidates",
        "",
        "| Rank | Model | Symbol | TF | Side | Train | Val | OOS | RPT train/val/OOS |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if not candidates:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    for row in candidates[:20]:
        rpt = "/".join(
            "NA"
            if row.get(f"{split}_return_per_turnover_proxy_bps") is None
            else f"{float(row[f'{split}_return_per_turnover_proxy_bps']):.2f}"
            for split in SPLIT_ORDER
        )
        lines.append(
            f"| {row.get('rank')} | `{row.get('model_id')}` | {row.get('symbol')} | "
            f"{row.get('timeframe')} | {row.get('side')} | "
            f"{float(row.get('train_return') or 0.0):.4%} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {rpt} |"
        )
    lines.append("")
    return "\n".join(lines)


def _no_promotion_shortlist(rows: Sequence[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    ranked = _rank_rows(rows)
    shadows = _shadow_rows(ranked, limit=limit)
    return {
        "handoff_kind": "no_promotion_shadow_shortlist",
        "status": "no_new_paper_promotion_shadow_shortlist",
        "ready_for_paper": False,
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": False,
        "shadow_count": len(shadows),
        "shadows": shadows,
        "baseline_lanes_preserved": BASELINE_LANES,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("discovery_summary") or {})
    top = list(payload.get("top_candidates") or [])[:10]
    paper_rows = [
        row for row in payload.get("top_candidates", []) if row.get("paper_candidate_gate_pass")
    ][:10]
    lines = [
        "# Alpha Zoo debounced efficiency repair discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "Focused ETH/SOL debounced momentum hysteresis repair pass. ",
        "Locked-OOS is gate/report-only after train+validation candidate freeze. ",
        "Real-money remains blocked.",
        "",
        "## Summary",
        "",
        f"- Candidates evaluated: `{summary.get('candidate_count')}`",
        f"- Rows with train >= validation: `{summary.get('train_return_gte_validation_return_count')}`",
        f"- Train-dominant sample gate pass: `{summary.get('train_dominant_sample_gate_pass_count')}`",
        f"- Execution-efficiency proxy gate pass: `{summary.get('execution_efficiency_proxy_gate_pass_count')}`",
        f"- Full paper candidate gate pass: `{summary.get('paper_candidate_gate_pass_count')}`",
        f"- Decision: `{payload.get('decision_status')}`",
        "- `ready_for_real=false`, `real_money_execution=false`",
        "",
        "## Top train+validation-ranked rows",
        "",
        "| Rank | Symbol | TF | Side | Rule | Train | Val | OOS | RPT train/val/OOS | Decision |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in top:
        rpt = "/".join(
            "NA" if row.get(f"{split}_return_per_turnover_proxy_bps") is None else f"{float(row[f'{split}_return_per_turnover_proxy_bps']):.2f}"
            for split in SPLIT_ORDER
        )
        rule = (
            f"lb{row.get('lookback_bars')} e{row.get('entry_threshold')} x{row.get('exit_threshold')} "
            f"hold{row.get('min_hold_bars')} cool{row.get('cooldown_bars')} {row.get('vol_filter_mode')}"
        )
        lines.append(
            f"| {row.get('rank')} | {row.get('symbol')} | {row.get('timeframe')} | {row.get('side')} | "
            f"{rule} | {float(row.get('train_return') or 0.0):.4%} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {rpt} | {row.get('decision')} |"
        )
    lines.extend(
        [
            "",
            "## Paper/testnet-only candidates",
            "",
            "These rows pass strict train-dominant, sample, locked-OOS report, liquidation/wipeout, "
            "and return-per-turnover proxy gates. They remain `ready_for_real=false` and require "
            "paper/testnet preflight plus monitoring before any forward observation.",
            "",
            "| Rank | Symbol | TF | Side | Train | Val | OOS | Trades | RPT train/val/OOS |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    if not paper_rows:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    for row in paper_rows:
        rpt = "/".join(
            "NA"
            if row.get(f"{split}_return_per_turnover_proxy_bps") is None
            else f"{float(row[f'{split}_return_per_turnover_proxy_bps']):.2f}"
            for split in SPLIT_ORDER
        )
        trades = (
            f"{row.get('train_trade_event_count')}/"
            f"{row.get('validation_trade_event_count')}/"
            f"{row.get('locked_oos_trade_event_count')}"
        )
        lines.append(
            f"| {row.get('rank')} | {row.get('symbol')} | {row.get('timeframe')} | "
            f"{row.get('side')} | {float(row.get('train_return') or 0.0):.4%} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {trades} | {rpt} |"
        )
    gate_rows = [
        row
        for row in payload.get("top_candidates", [])
        if row.get("train_dominant_sample_gate_pass") and not row.get("paper_candidate_gate_pass")
    ]
    lines.extend(
        [
            "",
            "## Train-dominant sample-gate shadows",
            "",
            "Rows here pass train/validation/OOS sample and risk gates but may still fail execution efficiency.",
            "",
            "| Rank | Symbol | TF | Train | Val | OOS | Rejection focus |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in gate_rows[:10]:
        reasons = "; ".join(str(reason) for reason in list(row.get("rejection_reasons") or [])[:4])
        lines.append(
            f"| {row.get('rank')} | {row.get('symbol')} | {row.get('timeframe')} | "
            f"{float(row.get('train_return') or 0.0):.4%} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {reasons} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    prior_artifact = Path(args.prior_artifact).expanduser().resolve()
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    timeframes = tuple(tf.strip().lower() for tf in args.timeframes.split(",") if tf.strip())
    data_symbols = tuple(dict.fromkeys((BTC_REGIME_SYMBOL, *symbols)))

    hourly = load_hourly_bars(data_symbols, data_root=data_root)
    bars_by_symbol_tf: dict[tuple[str, str], pd.DataFrame] = {}
    for timeframe in timeframes:
        for symbol in data_symbols:
            bars_by_symbol_tf[(symbol, timeframe)] = resample_bars(hourly[symbol], timeframe)

    rows = discover_repair_candidates(bars_by_symbol_tf, symbols=symbols, timeframes=timeframes)
    ranked_rows = _rank_rows(rows)
    top_rows = _selected_output_rows(ranked_rows, top_n=int(args.top_n))
    decision_rows = _decision_rows(ranked_rows, limit=int(args.decision_top_n))
    shadow_payload = _no_promotion_shortlist(ranked_rows, limit=int(args.shadow_top_n))
    paper_rows = [row for row in ranked_rows if row.get("paper_candidate_gate_pass")]
    handoff = _paper_testnet_handoff(paper_rows)
    decision_status = (
        "paper_testnet_candidate_after_fill_preflight"
        if paper_rows
        else "no_new_paper_promotion_shadow_shortlist"
    )

    timestamp = _timestamp()
    latest_json = output_dir / "alpha_zoo_debounced_efficiency_repair_discovery_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_debounced_efficiency_repair_discovery_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_debounced_efficiency_repair_discovery_latest.md"
    candidates_csv = output_dir / "debounced_efficiency_repair_candidates_latest.csv"
    decisions_csv = output_dir / "debounced_efficiency_repair_decisions_latest.csv"
    shadow_csv = output_dir / "debounced_efficiency_repair_shadow_hypotheses_latest.csv"
    no_promotion_json = output_dir / "no_promotion_shadow_shortlist_latest.json"
    handoff_json = output_dir / "paper_testnet_handoff_latest.json"
    handoff_md = output_dir / "paper_testnet_handoff_latest.md"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_debounced_efficiency_repair_discovery",
        "generated_at_utc": _utc_now_iso(),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "bbo_spread_multiplier": BBO_SPREAD_MULTIPLIER,
        "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "ready_for_paper": bool(paper_rows),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": bool(paper_rows),
        "paper_testnet_only": True,
        "decision_status": decision_status,
        "source_prior_artifact": str(prior_artifact),
        "target_prior_candidates": TARGET_PRIOR_CANDIDATES,
        "baseline_lanes": BASELINE_LANES,
        "source_data": {
            "ohlcv_root": str(data_root),
            "symbols": list(symbols),
            "timeframes": list(timeframes),
            "bar_source": "local Binance 1s OHLCV parquet resampled through existing 1h+ loader",
        },
        "selection_policy": {
            "objective_inputs": ["train", "validation"],
            "selection_inputs": ["train", "validation"],
            "ranking_freeze_before_locked_oos_gate": True,
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "locked_oos_role": "gate_report_only_after_train_validation_candidate_freeze",
            "score_formula": (
                "4*validation_return + 1.5*min(train,validation) + capped_train_val_RPT_bonus "
                "- 6*max(0,validation-train) - 2*validation_mdd - validation_turnover_penalty"
            ),
            "trust_gate": "promotion requires train_return >= validation_return and train/validation ratio >= 1.0",
            "no_calendar_date_hack": True,
        },
        "strategy_family": "debounced_momentum_hysteresis_efficiency_repair",
        "repair_dimensions": [
            "expanded_min_hold",
            "entry_exit_threshold_redesign",
            "stronger_hysteresis_debounce",
            "volatility_regime_filter",
            "adx_like_trend_strength_proxy",
            "cooldown_after_exit",
            "simple_position_state_rule",
        ],
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "split_manifest": {
            split: {
                "start": str(start),
                "end": str(end),
                "role": "objective_selection" if split != "locked_oos" else "gate_report_only",
            }
            for split, (start, end) in SPLITS.items()
        },
        "discovery_summary": _summary(ranked_rows),
        "top_candidates": top_rows,
        "decision_rows": decision_rows,
        "paper_testnet_handoff": handoff,
        "no_promotion_shadow_shortlist": shadow_payload,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "candidates_csv": str(candidates_csv),
            "decisions_csv": str(decisions_csv),
            "shadow_hypotheses_csv": str(shadow_csv),
            "no_promotion_shadow_shortlist_json": str(no_promotion_json),
            "paper_testnet_handoff_json": str(handoff_json),
            "paper_testnet_handoff_markdown": str(handoff_md),
            "artifact_generation_validation_log": str(generation_log),
        },
        "runner_peak_rss_mib": local_peak_mib,
    }

    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    _write_csv(candidates_csv, top_rows, CANDIDATE_FIELDS)
    _write_csv(decisions_csv, decision_rows, DECISION_FIELDS)
    _write_csv(shadow_csv, shadow_payload["shadows"], CANDIDATE_FIELDS)
    _write_json(no_promotion_json, shadow_payload)
    _write_json(handoff_json, handoff)
    handoff_md.write_text(_handoff_markdown(handoff), encoding="utf-8")
    generation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"candidate_count={payload['discovery_summary']['candidate_count']}",
                f"output_candidate_count={len(top_rows)}",
                "uses_locked_oos_for_discovery=false",
                "uses_locked_oos_for_selection=false",
                "uses_locked_oos_for_objective=false",
                "uses_locked_oos_for_pruning=false",
                "uses_locked_oos_for_parameter_fitting=false",
                "ready_for_real=false",
                "real_money_execution=false",
                f"decision_status={decision_status}",
                f"paper_candidate_gate_pass_count={payload['discovery_summary']['paper_candidate_gate_pass_count']}",
                f"runner_peak_rss_mib={local_peak_mib:.3f}",
                f"latest_json={latest_json}",
                f"timestamped_json={timestamped_json}",
                f"no_promotion_shadow_shortlist_json={no_promotion_json}",
                f"paper_testnet_handoff_json={handoff_json}",
                f"paper_testnet_handoff_markdown={handoff_md}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--prior-artifact", default=str(DEFAULT_PRIOR_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--top-n", type=int, default=180)
    parser.add_argument("--decision-top-n", type=int, default=80)
    parser.add_argument("--shadow-top-n", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
