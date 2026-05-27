#!/usr/bin/env python3
"""Run a stricter 30m+ Alpha Zoo booster discovery pass.

This pass is research/paper-testnet only. It deliberately keeps the prior 30m+
feedback runner's hard promotion gates, native 1s->30m bar construction, and
locked-OOS discipline, but explores stronger train/validation-only alpha shapes:
relative-strength breakouts, multi-horizon trend consensus, pullback reclaims,
and volatility-squeeze expansion. Locked-OOS remains gate/report-only after the
train+validation ranking score is frozen.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_30m_plus_alpha_feedback_discovery as feedback  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_30m_plus_alpha_booster_discovery_20260523"
)
DEFAULT_SYMBOLS = feedback.DEFAULT_SYMBOLS
DEFAULT_TIMEFRAMES = feedback.DEFAULT_TIMEFRAMES
BAR_CONSTRUCTION = feedback.BAR_CONSTRUCTION
STRATEGY_SCOPE = feedback.STRATEGY_SCOPE

BOOSTER_TARGETS = {
    "preferred_min_validation_return": 0.10,
    "preferred_min_locked_oos_return_report_gate": 0.03,
    "preferred_min_all_split_return_per_turnover_bps_report_gate": 20.0,
    "preferred_max_validation_mdd": 0.10,
    "preferred_min_validation_half_return": 0.0,
    "preferred_min_locked_oos_trade_event_count": 20,
}

EXTRA_CANDIDATE_FIELDS = [
    "booster_train_validation_score",
    "validation_first_half_return",
    "validation_second_half_return",
    "validation_min_half_return",
    "booster_target_gate_pass",
    "booster_target_reasons",
]
CANDIDATE_FIELDS = list(dict.fromkeys([*feedback.CANDIDATE_FIELDS, *EXTRA_CANDIDATE_FIELDS]))
DECISION_FIELDS = list(dict.fromkeys([*feedback.DECISION_FIELDS, *EXTRA_CANDIDATE_FIELDS]))


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _period_return(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


def _validation_halves(sim: feedback.SimResult, datetimes: pd.Series) -> tuple[float, float, float]:
    start, end = feedback.SPLITS["validation"]
    midpoint = start + (end - start) / 2
    values = pd.Series(datetimes)
    first = ((values >= start) & (values <= midpoint)).to_numpy()
    second = ((values > midpoint) & (values <= end)).to_numpy()
    first_ret = _period_return(sim.returns[first])
    second_ret = _period_return(sim.returns[second])
    return first_ret, second_ret, min(first_ret, second_ret)


def _booster_score(row: Mapping[str, Any]) -> float:
    """Train+validation-only ranking score; locked-OOS is intentionally absent."""
    train = float(row.get("train_return") or 0.0)
    validation = float(row.get("validation_return") or 0.0)
    val_mdd = float(row.get("validation_mdd") or 0.0)
    val_half_min = float(row.get("validation_min_half_return") or 0.0)
    train_rpt = float(row.get("train_return_per_turnover_proxy_bps") or 0.0)
    val_rpt = float(row.get("validation_return_per_turnover_proxy_bps") or 0.0)
    val_trades = float(row.get("validation_trade_event_count") or 0.0)
    validation_spike_penalty = max(0.0, validation - train)
    train_overfit_penalty = max(0.0, train - 5.0 * max(validation, 1e-9))
    rpt_bonus = min(train_rpt, 60.0) / 250.0 + min(val_rpt, 60.0) / 180.0
    half_penalty = max(0.0, -val_half_min) * 3.0
    return (
        7.0 * validation
        + 2.0 * min(train, validation)
        + rpt_bonus
        + 1.5 * max(0.0, val_half_min)
        - 8.0 * validation_spike_penalty
        - 2.5 * val_mdd
        - 0.00025 * val_trades
        - 1.5 * train_overfit_penalty
        - half_penalty
    )


def _apply_booster_targets(row: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        float(row.get("validation_return") or 0.0)
        < BOOSTER_TARGETS["preferred_min_validation_return"]
    ):
        reasons.append(
            f"validation_return_{float(row.get('validation_return') or 0.0):.4f}_below_preferred_0.10"
        )
    if (
        float(row.get("locked_oos_return") or 0.0)
        < BOOSTER_TARGETS["preferred_min_locked_oos_return_report_gate"]
    ):
        reasons.append(
            f"locked_oos_return_{float(row.get('locked_oos_return') or 0.0):.4f}_below_preferred_0.03"
        )
    if float(row.get("validation_mdd") or 0.0) > BOOSTER_TARGETS["preferred_max_validation_mdd"]:
        reasons.append(
            f"validation_mdd_{float(row.get('validation_mdd') or 0.0):.4f}_above_preferred_0.10"
        )
    if (
        int(row.get("locked_oos_trade_event_count") or 0)
        < BOOSTER_TARGETS["preferred_min_locked_oos_trade_event_count"]
    ):
        reasons.append(
            f"locked_oos_trade_event_count_{int(row.get('locked_oos_trade_event_count') or 0)}_below_20"
        )
    if (
        float(row.get("validation_min_half_return") or 0.0)
        < BOOSTER_TARGETS["preferred_min_validation_half_return"]
    ):
        reasons.append(
            f"validation_min_half_return_{float(row.get('validation_min_half_return') or 0.0):.4f}_below_0"
        )
    for split in feedback.SPLIT_ORDER:
        value = row.get(f"{split}_return_per_turnover_proxy_bps")
        if (
            value is None
            or float(value)
            < BOOSTER_TARGETS["preferred_min_all_split_return_per_turnover_bps_report_gate"]
        ):
            rendered = "missing" if value is None else f"{float(value):.3f}"
            reasons.append(f"{split}_rpt_{rendered}_below_preferred_20bps")
    row["booster_target_reasons"] = reasons
    row["booster_target_gate_pass"] = bool(row.get("paper_candidate_gate_pass")) and not reasons
    if row["booster_target_gate_pass"]:
        row["decision"] = "paper_testnet_booster_candidate_after_fill_preflight"
    return row


def _finalize_booster_candidate(
    base: dict[str, Any],
    sim: feedback.SimResult,
    datetimes: pd.Series,
    *,
    timeframe: str,
) -> dict[str, Any]:
    row = feedback._finalize_candidate(base, sim, datetimes, timeframe=timeframe)
    first, second, min_half = _validation_halves(sim, datetimes)
    row["validation_first_half_return"] = first
    row["validation_second_half_return"] = second
    row["validation_min_half_return"] = min_half
    score = _booster_score(row)
    row["booster_train_validation_score"] = score
    row["train_validation_score"] = score
    return _apply_booster_targets(row)


def _trailing_state_signal(
    close: pd.Series,
    long_entry: pd.Series,
    short_entry: pd.Series,
    long_exit: pd.Series,
    short_exit: pd.Series,
    atr: pd.Series,
    *,
    side: str,
    min_hold_bars: int,
    cooldown_bars: int,
    trail_atr_mult: float,
) -> np.ndarray:
    close_values = close.astype(float).to_numpy()
    atr_values = atr.astype(float).to_numpy()
    long_entry_values = long_entry.fillna(False).astype(bool).to_numpy()
    short_entry_values = short_entry.fillna(False).astype(bool).to_numpy()
    long_exit_values = long_exit.fillna(False).astype(bool).to_numpy()
    short_exit_values = short_exit.fillna(False).astype(bool).to_numpy()
    signal = np.zeros(len(close_values), dtype=float)
    state = 0.0
    stop = np.nan
    bars_held = 10**9
    cooldown = 0
    for idx, price in enumerate(close_values):
        atr_value = atr_values[idx]
        if not np.isfinite(price) or not np.isfinite(atr_value) or atr_value <= 0.0:
            signal[idx] = state
            if state != 0.0:
                bars_held += 1
            continue
        can_exit = bars_held >= min_hold_bars
        exited = False
        if state > 0.0:
            stop = (
                price - trail_atr_mult * atr_value
                if not np.isfinite(stop)
                else max(stop, price - trail_atr_mult * atr_value)
            )
            if can_exit and (long_exit_values[idx] or price < stop):
                state = 0.0
                stop = np.nan
                bars_held = 0
                cooldown = cooldown_bars
                exited = True
        elif state < 0.0:
            stop = (
                price + trail_atr_mult * atr_value
                if not np.isfinite(stop)
                else min(stop, price + trail_atr_mult * atr_value)
            )
            if can_exit and (short_exit_values[idx] or price > stop):
                state = 0.0
                stop = np.nan
                bars_held = 0
                cooldown = cooldown_bars
                exited = True
        if state == 0.0:
            if cooldown > 0:
                cooldown -= 1
            elif not exited:
                if side in {"long_only", "long_short"} and long_entry_values[idx]:
                    state = 1.0
                    stop = price - trail_atr_mult * atr_value
                    bars_held = 0
                elif side in {"short_only", "long_short"} and short_entry_values[idx]:
                    state = -1.0
                    stop = price + trail_atr_mult * atr_value
                    bars_held = 0
        signal[idx] = state
        if state != 0.0:
            bars_held += 1
    return signal


def _rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=1).replace(0.0, np.nan)
    return (series - mean) / std


def _relative_inputs(frame: pd.DataFrame, timeframe: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = frame["close"].astype(float)
    btc_close = frame["btc_close"].astype(float)
    hours = feedback._timeframe_hours(timeframe)
    btc_lookback = max(2, int(24 / hours))
    rel = close / btc_close.replace(0.0, np.nan)
    rel_momentum = rel / rel.shift(max(2, int(12 / hours))) - 1.0
    btc_momentum = btc_close / btc_close.shift(btc_lookback) - 1.0
    return rel, rel_momentum, btc_momentum


def _candidate_base(
    *,
    family: str,
    model_parts: Sequence[Any],
    symbol: str,
    timeframe: str,
    side: str,
    lookback: int,
    threshold: float,
    exit_threshold: float,
    min_hold: int,
    cooldown: int,
    filter_label: str,
    leverage: float,
    allocation: float,
) -> dict[str, Any]:
    return {
        "model_id": feedback._model_id(["booster", *model_parts]),
        "family": family,
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "lookback_bars": lookback,
        "threshold": threshold,
        "exit_threshold": exit_threshold,
        "min_hold_bars": min_hold,
        "cooldown_bars": cooldown,
        "filter_label": filter_label,
        "feature_backed": False,
        "feature_coverage": {},
        "max_asof_feature_age_hours": None,
        "leverage": leverage,
        "allocation_fraction": allocation,
        "notional_fraction": leverage * allocation,
    }


def discover_booster_candidates(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    leverage_allocs = (
        (2.0, 0.15),
        (3.0, 0.10),
        (3.0, 0.15),
        (4.0, 0.125),
        (5.0, 0.10),
        (4.0, 0.15),
    )
    for timeframe in timeframes:
        btc = bars_by_symbol_tf[(feedback.BTC_REGIME_SYMBOL, timeframe)][
            ["datetime", "close"]
        ].rename(columns={"close": "btc_close"})
        hours = feedback._timeframe_hours(timeframe)
        for symbol in symbols:
            frame = (
                bars_by_symbol_tf[(symbol, timeframe)].merge(btc, on="datetime", how="left").ffill()
            )
            close = frame["close"].astype(float)
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            volume = frame["volume"].astype(float)
            _, rel_momentum, btc_momentum = _relative_inputs(frame, timeframe)

            # Family 1: relative-strength breakout with chandelier exit. This is the main
            # high-upside lane but still ranked only by train+validation strength.
            for lookback in (12, 24, 36):
                atr = feedback._atr(high, low, close, max(6, lookback))
                roll_high = high.shift(1).rolling(lookback).max()
                roll_low = low.shift(1).rolling(lookback).min()
                mid = (roll_high + roll_low) / 2.0
                adx = feedback._adx_proxy(high, low, close, max(6, lookback // 2))
                for atr_mult in (0.0, 0.25, 0.50):
                    for rel_threshold in (0.0, 0.005, 0.010, 0.015):
                        common_long = (
                            (rel_momentum > rel_threshold) & (btc_momentum > -0.015) & (adx >= 15.0)
                        )
                        common_short = (
                            (rel_momentum < -rel_threshold) & (btc_momentum < 0.015) & (adx >= 15.0)
                        )
                        long_entry = (close > roll_high + atr_mult * atr) & common_long
                        short_entry = (close < roll_low - atr_mult * atr) & common_short
                        long_exit = (close < mid) | (rel_momentum < -0.005)
                        short_exit = (close > mid) | (rel_momentum > 0.005)
                        for trail in (2.0, 3.0):
                            for min_hold in (6, 12, 18):
                                signal = _trailing_state_signal(
                                    close,
                                    long_entry,
                                    short_entry,
                                    long_exit,
                                    short_exit,
                                    atr,
                                    side="long_short",
                                    min_hold_bars=min_hold,
                                    cooldown_bars=2,
                                    trail_atr_mult=trail,
                                )
                                for leverage, allocation in leverage_allocs:
                                    base = _candidate_base(
                                        family="relative_strength_chandelier_breakout",
                                        model_parts=(
                                            "rs_chandelier",
                                            timeframe,
                                            symbol,
                                            f"lb{lookback}",
                                            f"atr{atr_mult}",
                                            f"rel{rel_threshold}",
                                            f"trail{trail}",
                                            f"hold{min_hold}",
                                            f"{leverage}x",
                                            allocation,
                                        ),
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        side="long_short",
                                        lookback=lookback,
                                        threshold=rel_threshold,
                                        exit_threshold=-0.005,
                                        min_hold=min_hold,
                                        cooldown=2,
                                        filter_label=f"adx15_btc_regime_atr{atr_mult}_trail{trail}",
                                        leverage=leverage,
                                        allocation=allocation,
                                    )
                                    sim = feedback.simulate_symbol(
                                        frame,
                                        signal,
                                        leverage=leverage,
                                        allocation_fraction=allocation,
                                    )
                                    rows.append(
                                        _finalize_booster_candidate(
                                            base, sim, frame["datetime"], timeframe=timeframe
                                        )
                                    )

            # Family 2: multi-horizon consensus momentum. This is simpler than the
            # chandelier lane and should trade less frequently with higher RPT.
            for short_lb, long_lb in ((6, 24), (12, 36), (12, 48)):
                short_mom = close / close.shift(short_lb) - 1.0
                long_mom = close / close.shift(long_lb) - 1.0
                acceleration = short_mom - long_mom / max(long_lb / short_lb, 1.0)
                adx = feedback._adx_proxy(high, low, close, max(6, short_lb))
                realized = close.pct_change().rolling(max(4, short_lb)).std(ddof=1)
                vol_cap = realized.rolling(max(48, long_lb * 3)).quantile(0.80)
                for mom_threshold in (0.005, 0.010, 0.020):
                    for adx_threshold in (12.0, 18.0, 24.0):
                        common = (adx >= adx_threshold) & (realized <= vol_cap)
                        long_entry = (
                            (short_mom > mom_threshold)
                            & (long_mom > mom_threshold)
                            & (acceleration > -0.005)
                            & (rel_momentum > -0.005)
                            & (btc_momentum > -0.02)
                            & common
                        )
                        short_entry = (
                            (short_mom < -mom_threshold)
                            & (long_mom < -mom_threshold)
                            & (acceleration < 0.005)
                            & (rel_momentum < 0.005)
                            & (btc_momentum < 0.02)
                            & common
                        )
                        long_exit = (short_mom < 0.0) | (adx < adx_threshold * 0.75)
                        short_exit = (short_mom > 0.0) | (adx < adx_threshold * 0.75)
                        for min_hold in (8, 12, 18):
                            for cooldown in (2, 4):
                                signal = feedback._debounced_state_signal(
                                    long_entry,
                                    long_exit,
                                    short_entry,
                                    short_exit,
                                    side="long_short",
                                    min_hold_bars=min_hold,
                                    cooldown_bars=cooldown,
                                )
                                for leverage, allocation in leverage_allocs:
                                    base = _candidate_base(
                                        family="multi_horizon_consensus_momentum",
                                        model_parts=(
                                            "mh_consensus",
                                            timeframe,
                                            symbol,
                                            f"s{short_lb}",
                                            f"l{long_lb}",
                                            f"thr{mom_threshold}",
                                            f"adx{adx_threshold}",
                                            f"hold{min_hold}",
                                            f"cool{cooldown}",
                                            f"{leverage}x",
                                            allocation,
                                        ),
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        side="long_short",
                                        lookback=long_lb,
                                        threshold=mom_threshold,
                                        exit_threshold=0.0,
                                        min_hold=min_hold,
                                        cooldown=cooldown,
                                        filter_label=f"adx{adx_threshold}_vol_q80_rel_btc",
                                        leverage=leverage,
                                        allocation=allocation,
                                    )
                                    sim = feedback.simulate_symbol(
                                        frame,
                                        signal,
                                        leverage=leverage,
                                        allocation_fraction=allocation,
                                    )
                                    rows.append(
                                        _finalize_booster_candidate(
                                            base, sim, frame["datetime"], timeframe=timeframe
                                        )
                                    )

            # Family 3: trend pullback reclaim. It intentionally waits for a pullback
            # and reclaim instead of buying every momentum extension.
            for fast, slow in ((6, 30), (8, 40), (12, 48)):
                ema_fast = close.ewm(span=fast, adjust=False).mean()
                ema_slow = close.ewm(span=slow, adjust=False).mean()
                slope = ema_slow / ema_slow.shift(max(2, fast)) - 1.0
                distance = (close - ema_fast) / ema_fast.replace(0.0, np.nan)
                dist_z = _rolling_zscore(distance, max(24, slow))
                adx = feedback._adx_proxy(high, low, close, max(6, fast))
                for pullback_z in (-0.25, -0.50, -0.75):
                    long_entry = (
                        (ema_fast > ema_slow)
                        & (slope > 0.0)
                        & (dist_z.shift(1) <= pullback_z)
                        & (close > ema_fast)
                        & (adx >= 12.0)
                        & (btc_momentum > -0.02)
                    )
                    short_entry = (
                        (ema_fast < ema_slow)
                        & (slope < 0.0)
                        & (dist_z.shift(1) >= -pullback_z)
                        & (close < ema_fast)
                        & (adx >= 12.0)
                        & (btc_momentum < 0.02)
                    )
                    long_exit = (close < ema_slow) | (slope < 0.0)
                    short_exit = (close > ema_slow) | (slope > 0.0)
                    for min_hold in (4, 8, 12):
                        signal = feedback._debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage, allocation in leverage_allocs:
                            base = _candidate_base(
                                family="trend_pullback_reclaim",
                                model_parts=(
                                    "pullback_reclaim",
                                    timeframe,
                                    symbol,
                                    f"fast{fast}",
                                    f"slow{slow}",
                                    f"z{pullback_z}",
                                    f"hold{min_hold}",
                                    f"{leverage}x",
                                    allocation,
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=slow,
                                threshold=pullback_z,
                                exit_threshold=0.0,
                                min_hold=min_hold,
                                cooldown=2,
                                filter_label="ema_reclaim_adx12_btc_regime",
                                leverage=leverage,
                                allocation=allocation,
                            )
                            sim = feedback.simulate_symbol(
                                frame, signal, leverage=leverage, allocation_fraction=allocation
                            )
                            rows.append(
                                _finalize_booster_candidate(
                                    base, sim, frame["datetime"], timeframe=timeframe
                                )
                            )

            # Family 4: volatility squeeze then volume-confirmed range expansion.
            for lookback in (24, 36, 48):
                atr = feedback._atr(high, low, close, max(6, lookback // 2))
                natr = atr / close.replace(0.0, np.nan)
                squeeze_threshold = natr.rolling(max(72, lookback * 3)).quantile(0.35)
                in_squeeze = natr <= squeeze_threshold
                squeeze_recent = (
                    in_squeeze.rolling(max(3, int(12 / max(hours, 0.5)))).max().fillna(0.0) > 0.0
                )
                volume_z = _rolling_zscore(volume, max(24, lookback))
                roll_high = high.shift(1).rolling(lookback).max()
                roll_low = low.shift(1).rolling(lookback).min()
                mid = (roll_high + roll_low) / 2.0
                for vol_z in (0.0, 0.5, 1.0):
                    long_entry = (
                        (close > roll_high)
                        & squeeze_recent
                        & (volume_z >= vol_z)
                        & (rel_momentum > -0.005)
                        & (btc_momentum > -0.02)
                    )
                    short_entry = (
                        (close < roll_low)
                        & squeeze_recent
                        & (volume_z >= vol_z)
                        & (rel_momentum < 0.005)
                        & (btc_momentum < 0.02)
                    )
                    long_exit = close < mid
                    short_exit = close > mid
                    for min_hold in (6, 12):
                        signal = _trailing_state_signal(
                            close,
                            long_entry,
                            short_entry,
                            long_exit,
                            short_exit,
                            atr,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=4,
                            trail_atr_mult=2.5,
                        )
                        for leverage, allocation in leverage_allocs:
                            base = _candidate_base(
                                family="volatility_squeeze_range_expansion",
                                model_parts=(
                                    "squeeze_expansion",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"volz{vol_z}",
                                    f"hold{min_hold}",
                                    f"{leverage}x",
                                    allocation,
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=vol_z,
                                exit_threshold=0.0,
                                min_hold=min_hold,
                                cooldown=4,
                                filter_label="natr_q35_volume_btc_rel",
                                leverage=leverage,
                                allocation=allocation,
                            )
                            sim = feedback.simulate_symbol(
                                frame, signal, leverage=leverage, allocation_fraction=allocation
                            )
                            rows.append(
                                _finalize_booster_candidate(
                                    base, sim, frame["datetime"], timeframe=timeframe
                                )
                            )
    return rows


def _rank_rows(rows: Sequence[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    ranked = sorted(
        rows, key=lambda row: float(row.get("booster_train_validation_score") or -1e9), reverse=True
    )
    if limit is not None:
        ranked = ranked[:limit]
    return [dict(row, rank=rank) for rank, row in enumerate(ranked, start=1)]


def _selected_output_rows(
    ranked_rows: Sequence[dict[str, Any]], *, top_n: int
) -> list[dict[str, Any]]:
    selected_ids = {str(row["model_id"]) for row in ranked_rows[:top_n]}
    for row in ranked_rows:
        if (
            row.get("paper_candidate_gate_pass")
            or row.get("booster_target_gate_pass")
            or row.get("execution_efficiency_proxy_gate_pass")
        ):
            selected_ids.add(str(row["model_id"]))
    if ranked_rows:
        best_validation = max(
            ranked_rows, key=lambda row: float(row.get("validation_return") or -1e9)
        )
        selected_ids.add(str(best_validation["model_id"]))
    return [dict(row) for row in ranked_rows if str(row["model_id"]) in selected_ids]


def _decision_rows(rows: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _selected_output_rows(_rank_rows(rows), top_n=limit):
        out.append(
            {
                field: row.get(field)
                for field in DECISION_FIELDS
                if field in row or field in feedback.DECISION_FIELDS
            }
        )
        out[-1]["decision_rank"] = row["rank"]
    return out


def _shadow_shortlist(ranked_rows: Sequence[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    shadows = [row for row in ranked_rows if not row.get("paper_candidate_gate_pass")]
    chosen = _selected_output_rows(shadows, top_n=limit) if shadows else []
    return {
        "handoff_kind": "no_promotion_shadow_shortlist",
        "status": "no_new_booster_paper_promotion_shadow_shortlist",
        "ready_for_paper": False,
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": False,
        "shadow_count": len(chosen),
        "shadows": chosen,
        "baseline_lanes_preserved": feedback.BASELINE_LANES,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    families: dict[str, int] = {}
    symbols: dict[str, int] = {}
    paper = [row for row in rows if row.get("paper_candidate_gate_pass")]
    booster = [row for row in rows if row.get("booster_target_gate_pass")]
    for row in rows:
        decisions[str(row.get("decision"))] = decisions.get(str(row.get("decision")), 0) + 1
        families[str(row.get("family"))] = families.get(str(row.get("family")), 0) + 1
        symbols[str(row.get("symbol"))] = symbols.get(str(row.get("symbol")), 0) + 1
    best_validation = (
        max(rows, key=lambda row: float(row.get("validation_return") or -1e9)) if rows else {}
    )
    return {
        "candidate_count": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "family_counts": dict(sorted(families.items())),
        "symbol_counts": dict(sorted(symbols.items())),
        "execution_efficiency_proxy_gate_pass_count": sum(
            bool(row.get("execution_efficiency_proxy_gate_pass")) for row in rows
        ),
        "paper_candidate_gate_pass_count": len(paper),
        "booster_target_gate_pass_count": len(booster),
        "max_validation_return": float(best_validation.get("validation_return") or 0.0)
        if best_validation
        else None,
        "best_validation_model_id": best_validation.get("model_id") if best_validation else None,
        "best_paper_candidate_model_id": paper[0].get("model_id") if paper else None,
        "best_booster_target_model_id": booster[0].get("model_id") if booster else None,
        "ready_for_paper": bool(paper),
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _paper_handoff(paper_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    handoff = feedback._paper_testnet_handoff(paper_rows)
    handoff["handoff_kind"] = "paper_testnet_only_30m_plus_alpha_booster_discovery"
    handoff["booster_targets"] = BOOSTER_TARGETS
    handoff["booster_target_candidate_count"] = sum(
        bool(row.get("booster_target_gate_pass")) for row in paper_rows
    )
    return handoff


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("discovery_summary") or {})
    top = list(payload.get("top_candidates") or [])[:12]
    paper = [
        row for row in payload.get("top_candidates", []) if row.get("paper_candidate_gate_pass")
    ][:12]
    booster = [
        row for row in payload.get("top_candidates", []) if row.get("booster_target_gate_pass")
    ][:12]
    lines = [
        "# Alpha Zoo 30m+ booster discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "Research/paper-testnet only. Locked-OOS is gate/report-only after train+validation ranking freeze.",
        "",
        f"- Candidates evaluated: `{summary.get('candidate_count')}`",
        f"- Paper candidate gate pass: `{summary.get('paper_candidate_gate_pass_count')}`",
        f"- Preferred booster target pass: `{summary.get('booster_target_gate_pass_count')}`",
        "- `ready_for_real=false`",
        "- `real_money_execution=false`",
        f"- Runner peak RSS MiB: `{payload.get('runner_peak_rss_mib')}`",
        "",
        "## Top train+validation-ranked rows",
        "",
        "| Rank | Symbol | TF | Family | Train | Val | OOS | RPT train/val/OOS | Decision | Reasons |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in top:
        rpt = "/".join(
            "NA"
            if row.get(f"{split}_return_per_turnover_proxy_bps") is None
            else f"{float(row[f'{split}_return_per_turnover_proxy_bps']):.2f}"
            for split in feedback.SPLIT_ORDER
        )
        reasons = "; ".join(
            (row.get("rejection_reasons") or row.get("booster_target_reasons") or [])[:3]
        )
        lines.append(
            f"| {row.get('rank')} | {row.get('symbol')} | {row.get('timeframe')} | {row.get('family')} | "
            f"{float(row.get('train_return') or 0.0):.4%} | {float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {rpt} | {row.get('decision')} | {reasons} |"
        )
    lines.extend(["", "## Paper/testnet candidates", ""])
    if not paper:
        lines.append("No paper/testnet candidates passed the strict primary gates.")
    else:
        lines.extend(
            [
                "| Rank | Model | Symbol | TF | Family | Train | Val | OOS | Booster target |",
                "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in paper:
            lines.append(
                f"| {row.get('rank')} | `{row.get('model_id')}` | {row.get('symbol')} | {row.get('timeframe')} | "
                f"{row.get('family')} | {float(row.get('train_return') or 0.0):.4%} | "
                f"{float(row.get('validation_return') or 0.0):.4%} | "
                f"{float(row.get('locked_oos_return') or 0.0):.4%} | "
                f"{str(bool(row.get('booster_target_gate_pass'))).lower()} |"
            )
    if booster:
        lines.extend(["", "## Preferred booster target candidates", ""])
        for row in booster:
            lines.append(
                f"- `{row.get('model_id')}`: {row.get('symbol')} {row.get('timeframe')} "
                f"train={float(row.get('train_return') or 0.0):.4%}, "
                f"val={float(row.get('validation_return') or 0.0):.4%}, "
                f"OOS={float(row.get('locked_oos_return') or 0.0):.4%}"
            )
    return "\n".join(lines) + "\n"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    timeframes = feedback._validate_timeframes(
        tuple(tf.strip().lower() for tf in args.timeframes.split(",") if tf.strip())
    )
    data_root = Path(args.data_root)
    feature_root = Path(args.feature_root)
    data_symbols = tuple(dict.fromkeys([feedback.BTC_REGIME_SYMBOL, *symbols]))
    bars_by_symbol_tf = feedback.load_requested_bars(
        data_symbols, timeframes=timeframes, data_root=data_root
    )
    for symbol in data_symbols:
        features = feedback.load_feature_points(symbol, feature_root=feature_root)
        for timeframe in timeframes:
            bars_by_symbol_tf[(symbol, timeframe)] = feedback._attach_features_with_age(
                bars_by_symbol_tf[(symbol, timeframe)],
                features,
                timeframe=timeframe,
            )
    rows = discover_booster_candidates(bars_by_symbol_tf, symbols=symbols, timeframes=timeframes)
    ranked = _rank_rows(rows)
    top_rows = _selected_output_rows(ranked, top_n=int(args.top_n))
    paper_rows = [row for row in ranked if row.get("paper_candidate_gate_pass")]
    handoff = _paper_handoff(paper_rows)
    shadow = _shadow_shortlist(ranked, limit=int(args.shadow_limit))

    latest_json = output_dir / "alpha_zoo_30m_plus_alpha_booster_discovery_latest.json"
    timestamped_json = (
        output_dir / f"alpha_zoo_30m_plus_alpha_booster_discovery_{_timestamp()}.json"
    )
    latest_md = output_dir / "alpha_zoo_30m_plus_alpha_booster_discovery_latest.md"
    candidates_csv = output_dir / "alpha_zoo_30m_plus_alpha_booster_candidates_latest.csv"
    decisions_csv = output_dir / "alpha_zoo_30m_plus_alpha_booster_decisions_latest.csv"
    shadows_csv = output_dir / "alpha_zoo_30m_plus_alpha_booster_shadow_hypotheses_latest.csv"
    handoff_json = output_dir / "paper_testnet_handoff_latest.json"
    handoff_md = output_dir / "paper_testnet_handoff_latest.md"
    no_promotion_json = output_dir / "no_promotion_shadow_shortlist_latest.json"

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_30m_plus_alpha_booster_discovery",
        "generated_at_utc": _utc_now_iso(),
        "ready_for_paper": bool(paper_rows),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_testnet_only": True,
        "research_primary_round_trip_cost_bps": feedback.PRIMARY_ROUND_TRIP_COST_BPS,
        "return_per_turnover_threshold_bps": feedback.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "bar_construction": BAR_CONSTRUCTION,
        "strategy_scope": STRATEGY_SCOPE,
        "source_data": {
            "ohlcv_root": str(data_root),
            "feature_root": str(feature_root),
            "symbols": list(symbols),
            "timeframes": list(timeframes),
            "default_timeframes": list(DEFAULT_TIMEFRAMES),
        },
        "selection_policy": {
            "score_formula": "booster_train_validation_score(validation, train, train/validation RPT, validation halves, validation MDD)",
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "ranking_freeze_before_locked_oos_gate": True,
            "locked_oos_role": "gate_report_only_after_train_validation_candidate_freeze",
            "train_return_below_validation_return_is_promotion_reject": True,
        },
        "promotion_thresholds": feedback.PROMOTION_THRESHOLDS,
        "booster_targets": BOOSTER_TARGETS,
        "split_manifest": {
            split: {"start": start.isoformat(), "end": end.isoformat()}
            for split, (start, end) in feedback.SPLITS.items()
        },
        "strategy_families": [
            "relative_strength_chandelier_breakout",
            "multi_horizon_consensus_momentum",
            "trend_pullback_reclaim",
            "volatility_squeeze_range_expansion",
        ],
        "external_research_references": feedback.EXTERNAL_RESEARCH_REFERENCES,
        "baseline_lanes_preserved": feedback.BASELINE_LANES,
        "discovery_summary": _summary(ranked),
        "top_candidates": top_rows,
        "decision_rows": _decision_rows(ranked, limit=int(args.top_n)),
        "paper_testnet_handoff": handoff,
        "no_promotion_shadow_shortlist": shadow,
        "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_md": str(latest_md),
            "candidates_csv": str(candidates_csv),
            "decisions_csv": str(decisions_csv),
            "shadow_hypotheses_csv": str(shadows_csv),
            "paper_testnet_handoff_json": str(handoff_json),
            "paper_testnet_handoff_md": str(handoff_md),
            "no_promotion_shadow_shortlist_json": str(no_promotion_json),
        },
        "booster_feedback_loop": {
            "motivation": "prior paper candidates were cost-efficient but absolute returns were modest",
            "target_result": "find stricter 30m+ paper/testnet candidates with materially higher validation/OOS returns without OOS fitting",
            "locked_oos_used_for_design": False,
            "memory_budget_mib": 8192,
        },
    }

    markdown = _markdown(payload)
    feedback._write_json(latest_json, payload)
    feedback._write_json(timestamped_json, payload)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.write_text(markdown, encoding="utf-8")
    feedback._write_csv(candidates_csv, top_rows, CANDIDATE_FIELDS)
    feedback._write_csv(decisions_csv, payload["decision_rows"], DECISION_FIELDS)
    feedback._write_csv(shadows_csv, shadow.get("shadows", []), CANDIDATE_FIELDS)
    feedback._write_json(handoff_json, handoff)
    handoff_md.write_text(feedback._handoff_markdown(handoff), encoding="utf-8")
    feedback._write_json(no_promotion_json, shadow)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(feedback.DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(feedback.DEFAULT_FEATURE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--top-n", type=int, default=250)
    parser.add_argument("--shadow-limit", type=int, default=75)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    summary = payload["discovery_summary"]
    print(
        json.dumps(
            {
                "artifact": payload["output_paths"]["latest_json"],
                "candidate_count": summary["candidate_count"],
                "paper_candidate_gate_pass_count": summary["paper_candidate_gate_pass_count"],
                "booster_target_gate_pass_count": summary["booster_target_gate_pass_count"],
                "best_paper_candidate_model_id": summary["best_paper_candidate_model_id"],
                "best_booster_target_model_id": summary["best_booster_target_model_id"],
                "ready_for_real": False,
                "real_money_execution": False,
                "runner_peak_rss_mib": payload["runner_peak_rss_mib"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
