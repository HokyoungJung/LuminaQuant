#!/usr/bin/env python3
"""Run an asset-diverse 30m+ Alpha Zoo discovery pass on local real data.

This runner is research/paper-testnet only. It broadens the search from one
asset lane into cross-asset-conditioned single-symbol strategies. Candidate
ranking uses train+validation evidence only; locked-OOS is attached after the
ranking freeze as a gate/report split. It never executes orders and never enables
real money.
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
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_asset_diverse_strategy_discovery_20260523"
)
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT")
DEFAULT_SHADOW_SYMBOLS = (
    "ADAUSDT",
    "AVAXUSDT",
    "DOGEUSDT",
    "TONUSDT",
    "XRPUSDT",
    "XAUUSDT",
    "XAGUSDT",
    "XPTUSDT",
    "XPDUSDT",
)
DEFAULT_TIMEFRAMES = ("1h", "2h", "4h", "6h", "12h")
BAR_CONSTRUCTION = feedback.BAR_CONSTRUCTION
STRATEGY_SCOPE = "single_symbol_cross_asset_conditioned_only"

ASSET_GROUPS: dict[str, tuple[str, ...]] = {
    "crypto_major": ("BTCUSDT", "ETHUSDT"),
    "crypto_high_beta_alt": ("SOLUSDT", "AVAXUSDT", "DOGEUSDT", "TONUSDT", "ADAUSDT"),
    "crypto_payment_alt": ("TRXUSDT", "XRPUSDT"),
    "crypto_exchange_beta": ("BNBUSDT",),
    "precious_metal_proxy": ("XAUUSDT", "XAGUSDT", "XPTUSDT", "XPDUSDT"),
}

EXTRA_CANDIDATE_FIELDS = [
    "asset_group",
    "universe_role",
    "cross_asset_inputs",
    "asset_diverse_train_validation_score",
    "validation_first_half_return",
    "validation_second_half_return",
    "validation_min_half_return",
]
CANDIDATE_FIELDS = list(dict.fromkeys([*feedback.CANDIDATE_FIELDS, *EXTRA_CANDIDATE_FIELDS]))
DECISION_FIELDS = list(dict.fromkeys([*feedback.DECISION_FIELDS, *EXTRA_CANDIDATE_FIELDS]))

STRATEGY_FAMILIES = [
    "cross_asset_rank_chandelier_breakout",
    "relative_residual_reclaim",
    "breadth_regime_pullback_reclaim",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _asset_group(symbol: str) -> str:
    normalized = symbol.strip().upper()
    for group, members in ASSET_GROUPS.items():
        if normalized in members:
            return group
    return "other"


def _parse_csv_symbols(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part.strip().upper() for part in value.split(",") if part.strip()))


def _period_return(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.prod(1.0 + returns) - 1.0)


def _validation_halves(
    sim: feedback.SimResult,
    datetimes: pd.Series,
) -> tuple[float, float, float]:
    start, end = feedback.SPLITS["validation"]
    midpoint = start + (end - start) / 2
    values = pd.Series(pd.to_datetime(datetimes))
    first = ((values >= start) & (values <= midpoint)).to_numpy()
    second = ((values > midpoint) & (values <= end)).to_numpy()
    first_ret = _period_return(sim.returns[first])
    second_ret = _period_return(sim.returns[second])
    return first_ret, second_ret, min(first_ret, second_ret)


def _asset_diverse_score(row: Mapping[str, Any]) -> float:
    """Train+validation-only ranking score; locked-OOS is intentionally absent."""
    train = float(row.get("train_return") or 0.0)
    validation = float(row.get("validation_return") or 0.0)
    train_rpt = float(row.get("train_return_per_turnover_proxy_bps") or 0.0)
    val_rpt = float(row.get("validation_return_per_turnover_proxy_bps") or 0.0)
    val_mdd = float(row.get("validation_mdd") or 0.0)
    val_trades = float(row.get("validation_trade_event_count") or 0.0)
    val_half = float(row.get("validation_min_half_return") or 0.0)
    spike_penalty = max(0.0, validation - train)
    overfit_penalty = max(0.0, train - 5.0 * max(validation, 1e-9))
    rpt_bonus = min(train_rpt, 70.0) / 260.0 + min(val_rpt, 70.0) / 180.0
    return (
        7.0 * validation
        + 2.0 * min(train, validation)
        + 1.25 * max(0.0, val_half)
        + rpt_bonus
        - 8.0 * spike_penalty
        - 2.25 * val_mdd
        - 0.00018 * val_trades
        - 1.25 * overfit_penalty
        - 2.5 * max(0.0, -val_half)
    )


def _force_shadow_only(row: dict[str, Any], reason: str) -> dict[str, Any]:
    reasons = list(row.get("rejection_reasons") or [])
    if reason not in reasons:
        reasons.append(reason)
    row.update(
        {
            "paper_candidate_gate_pass": False,
            "primary_10bps_promotion_gate_pass": False,
            "ready_for_paper": False,
            "ready_for_real": False,
            "real_money_execution": False,
            "decision": "no_promotion_shadow_or_reject",
            "rejection_reasons": reasons,
        }
    )
    return row


def _finalize_asset_candidate(
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
    score = _asset_diverse_score(row)
    row["asset_diverse_train_validation_score"] = score
    row["train_validation_score"] = score
    if row.get("universe_role") != "promotion_eligible":
        row = _force_shadow_only(row, "shadow_universe_not_promotion_eligible")
    return row


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
            next_stop = price - trail_atr_mult * atr_value
            stop = next_stop if not np.isfinite(stop) else max(stop, next_stop)
            if can_exit and (long_exit_values[idx] or price < stop):
                state = 0.0
                stop = np.nan
                bars_held = 0
                cooldown = cooldown_bars
                exited = True
        elif state < 0.0:
            next_stop = price + trail_atr_mult * atr_value
            stop = next_stop if not np.isfinite(stop) else min(stop, next_stop)
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


def _align_to_frame(series: pd.Series, datetimes: pd.Series) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(datetimes))
    if pd.api.types.is_bool_dtype(series):
        aligned_bool = series.reindex(index, fill_value=False).astype(bool)
        return pd.Series(aligned_bool.to_numpy(), index=datetimes.index)
    aligned_numeric = pd.to_numeric(series.reindex(index).ffill(), errors="coerce")
    return pd.Series(aligned_numeric.to_numpy(dtype=float), index=datetimes.index)


def _build_close_panel(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframe: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = bars_by_symbol_tf[(symbol, timeframe)][["datetime", "close"]].copy()
        frame["symbol"] = symbol
        frames.append(frame)
    panel = pd.concat(frames).pivot(index="datetime", columns="symbol", values="close")
    return panel.sort_index().dropna(how="any")


def _panel_state(panel: pd.DataFrame, lookback: int) -> dict[str, pd.Series | pd.DataFrame]:
    returns = panel.pct_change().mean(axis=1).fillna(0.0)
    market_index = (1.0 + returns).cumprod()
    momentum = panel / panel.shift(lookback) - 1.0
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    reverse_ranks = momentum.rank(axis=1, ascending=True, method="first")
    return {
        "market_momentum": market_index / market_index.shift(lookback) - 1.0,
        "breadth": (momentum > 0.0).sum(axis=1) / float(len(panel.columns)),
        "dispersion": momentum.std(axis=1, ddof=1),
        "momentum": momentum,
        "ranks": ranks,
        "reverse_ranks": reverse_ranks,
    }


def _rank_flags(
    symbol: str,
    state: Mapping[str, pd.Series | pd.DataFrame],
    datetimes: pd.Series,
    *,
    top_n: int,
    target_momentum: pd.Series,
    market_momentum: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    ranks = state["ranks"]
    reverse_ranks = state["reverse_ranks"]
    if isinstance(ranks, pd.DataFrame) and symbol in ranks.columns:
        strong = _align_to_frame(ranks[symbol] <= top_n, datetimes).fillna(False).astype(bool)
        weak = _align_to_frame(reverse_ranks[symbol] <= top_n, datetimes).fillna(False).astype(bool)
        return strong, weak
    rel = target_momentum - market_momentum
    return rel > 0.0, rel < 0.0


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
    promotion_symbols: set[str],
    cross_asset_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_id": feedback._model_id(["asset_diverse", *model_parts]),
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
        "asset_group": _asset_group(symbol),
        "universe_role": "promotion_eligible" if symbol in promotion_symbols else "shadow_probe_only",
        "cross_asset_inputs": dict(cross_asset_inputs),
    }


def discover_asset_diverse_candidates(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    shadow_symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    promotion_symbols = set(symbols)
    target_symbols = tuple(dict.fromkeys([*symbols, *shadow_symbols]))
    leverage_allocs = ((2.0, 0.15), (3.0, 0.10), (4.0, 0.125), (5.0, 0.10))

    for timeframe in timeframes:
        panel = _build_close_panel(bars_by_symbol_tf, symbols=symbols, timeframe=timeframe)
        panel_by_lookback = {lookback: _panel_state(panel, lookback) for lookback in (12, 24, 36, 48, 72)}
        base_closes = {
            base: bars_by_symbol_tf[(base, timeframe)][["datetime", "close"]].rename(
                columns={"close": f"{base.lower()}_close"}
            )
            for base in ("BTCUSDT", "ETHUSDT")
            if (base, timeframe) in bars_by_symbol_tf
        }
        for symbol in target_symbols:
            frame = bars_by_symbol_tf[(symbol, timeframe)].copy().sort_values("datetime")
            close = frame["close"].astype(float)
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            hours = feedback._timeframe_hours(timeframe)

            for lookback in (12, 24, 36):
                state = panel_by_lookback[lookback]
                market_mom = _align_to_frame(state["market_momentum"], frame["datetime"])
                breadth = _align_to_frame(state["breadth"], frame["datetime"])
                target_mom = close / close.shift(lookback) - 1.0
                roll_high = high.shift(1).rolling(lookback).max()
                roll_low = low.shift(1).rolling(lookback).min()
                mid = (roll_high + roll_low) / 2.0
                atr = feedback._atr(high, low, close, max(6, lookback // 2))
                adx = feedback._adx_proxy(high, low, close, max(6, lookback // 2))
                for top_n in (1, 2):
                    strong, weak = _rank_flags(
                        symbol,
                        state,
                        frame["datetime"],
                        top_n=top_n,
                        target_momentum=target_mom,
                        market_momentum=market_mom,
                    )
                    for mom_threshold in (0.005, 0.015):
                        for breadth_threshold in (0.40, 0.60):
                            for adx_threshold in (12.0, 18.0):
                                long_entry = (
                                    (close > roll_high)
                                    & strong
                                    & (target_mom > mom_threshold)
                                    & (breadth >= breadth_threshold)
                                    & (market_mom > -0.02)
                                    & (adx >= adx_threshold)
                                )
                                short_entry = (
                                    (close < roll_low)
                                    & weak
                                    & (target_mom < -mom_threshold)
                                    & (breadth <= 1.0 - breadth_threshold)
                                    & (market_mom < 0.02)
                                    & (adx >= adx_threshold)
                                )
                                long_exit = (close < mid) | (target_mom < 0.0) | (breadth < 0.35)
                                short_exit = (close > mid) | (target_mom > 0.0) | (breadth > 0.65)
                                for trail in (2.0, 3.0):
                                    for min_hold in (8, 12, 18):
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
                                                family="cross_asset_rank_chandelier_breakout",
                                                model_parts=(
                                                    "rank_chandelier",
                                                    timeframe,
                                                    symbol,
                                                    f"lb{lookback}",
                                                    f"top{top_n}",
                                                    f"mom{mom_threshold}",
                                                    f"breadth{breadth_threshold}",
                                                    f"adx{adx_threshold}",
                                                    f"trail{trail}",
                                                    f"hold{min_hold}",
                                                    f"{leverage}x",
                                                    allocation,
                                                ),
                                                symbol=symbol,
                                                timeframe=timeframe,
                                                side="long_short",
                                                lookback=lookback,
                                                threshold=mom_threshold,
                                                exit_threshold=0.0,
                                                min_hold=min_hold,
                                                cooldown=2,
                                                filter_label=f"top{top_n}_breadth_adx_trail{trail}",
                                                leverage=leverage,
                                                allocation=allocation,
                                                promotion_symbols=promotion_symbols,
                                                cross_asset_inputs={
                                                    "panel_symbols": list(symbols),
                                                    "rank_top_n": top_n,
                                                    "breadth_threshold": breadth_threshold,
                                                    "market_momentum_floor": -0.02,
                                                },
                                            )
                                            sim = feedback.simulate_symbol(
                                                frame,
                                                signal,
                                                leverage=leverage,
                                                allocation_fraction=allocation,
                                            )
                                            rows.append(
                                                _finalize_asset_candidate(
                                                    base,
                                                    sim,
                                                    frame["datetime"],
                                                    timeframe=timeframe,
                                                )
                                            )

            for base_symbol, base_close_frame in base_closes.items():
                if base_symbol == symbol:
                    continue
                merged = frame.merge(base_close_frame, on="datetime", how="left").ffill()
                base_close = merged[f"{base_symbol.lower()}_close"].astype(float)
                ratio = np.log(close / base_close.replace(0.0, np.nan))
                state = panel_by_lookback[24]
                market_mom = _align_to_frame(state["market_momentum"], frame["datetime"])
                for lookback in (24, 48, 72):
                    z = _rolling_zscore(ratio, lookback)
                    target_mom = close / close.shift(max(4, lookback // 3)) - 1.0
                    for z_entry in (1.0, 1.5, 2.0):
                        reclaim_long = (z.shift(1) < -z_entry) & (z >= -z_entry)
                        reclaim_short = (z.shift(1) > z_entry) & (z <= z_entry)
                        long_entry = reclaim_long & (target_mom > -0.005) & (market_mom > -0.02)
                        short_entry = reclaim_short & (target_mom < 0.005) & (market_mom < 0.02)
                        long_exit = (z > -0.05) | (target_mom < -0.02)
                        short_exit = (z < 0.05) | (target_mom > 0.02)
                        for min_hold in (6, 12):
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
                                    family="relative_residual_reclaim",
                                    model_parts=(
                                        "residual_reclaim",
                                        timeframe,
                                        symbol,
                                        base_symbol,
                                        f"lb{lookback}",
                                        f"z{z_entry}",
                                        f"hold{min_hold}",
                                        f"{leverage}x",
                                        allocation,
                                    ),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    side="long_short",
                                    lookback=lookback,
                                    threshold=z_entry,
                                    exit_threshold=0.05,
                                    min_hold=min_hold,
                                    cooldown=2,
                                    filter_label=f"base_{base_symbol.lower()}_market_momentum_guard",
                                    leverage=leverage,
                                    allocation=allocation,
                                    promotion_symbols=promotion_symbols,
                                    cross_asset_inputs={
                                        "base_symbol": base_symbol,
                                        "panel_symbols": list(symbols),
                                        "market_momentum_floor": -0.02,
                                    },
                                )
                                sim = feedback.simulate_symbol(
                                    frame,
                                    signal,
                                    leverage=leverage,
                                    allocation_fraction=allocation,
                                )
                                rows.append(
                                    _finalize_asset_candidate(base, sim, frame["datetime"], timeframe=timeframe)
                                )

            for fast, slow in ((6, 30), (8, 40), (12, 48)):
                state = panel_by_lookback[24]
                market_mom = _align_to_frame(state["market_momentum"], frame["datetime"])
                breadth = _align_to_frame(state["breadth"], frame["datetime"])
                ema_fast = close.ewm(span=fast, adjust=False).mean()
                ema_slow = close.ewm(span=slow, adjust=False).mean()
                slope = ema_slow / ema_slow.shift(max(2, int(12 / max(hours, 0.5)))) - 1.0
                distance = (close - ema_fast) / ema_fast.replace(0.0, np.nan)
                dist_z = _rolling_zscore(distance, max(24, slow))
                for pullback_z in (-0.25, -0.50, -0.75):
                    long_entry = (
                        (market_mom > 0.01)
                        & (breadth >= 0.60)
                        & (ema_fast > ema_slow)
                        & (slope > 0.0)
                        & (dist_z.shift(1) <= pullback_z)
                        & (close > ema_fast)
                    )
                    short_entry = (
                        (market_mom < -0.01)
                        & (breadth <= 0.40)
                        & (ema_fast < ema_slow)
                        & (slope < 0.0)
                        & (dist_z.shift(1) >= -pullback_z)
                        & (close < ema_fast)
                    )
                    long_exit = (close < ema_slow) | (breadth < 0.50) | (market_mom < -0.005)
                    short_exit = (close > ema_slow) | (breadth > 0.50) | (market_mom > 0.005)
                    for min_hold in (6, 12, 18):
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
                                family="breadth_regime_pullback_reclaim",
                                model_parts=(
                                    "breadth_pullback",
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
                                filter_label="cross_asset_breadth_regime_ema_reclaim",
                                leverage=leverage,
                                allocation=allocation,
                                promotion_symbols=promotion_symbols,
                                cross_asset_inputs={
                                    "panel_symbols": list(symbols),
                                    "long_breadth_floor": 0.60,
                                    "short_breadth_ceiling": 0.40,
                                },
                            )
                            sim = feedback.simulate_symbol(
                                frame,
                                signal,
                                leverage=leverage,
                                allocation_fraction=allocation,
                            )
                            rows.append(
                                _finalize_asset_candidate(base, sim, frame["datetime"], timeframe=timeframe)
                            )
    return rows


def _rank_rows(rows: Sequence[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: float(row.get("asset_diverse_train_validation_score") or -1e9),
        reverse=True,
    )
    if limit is not None:
        ranked = ranked[:limit]
    return [dict(row, rank=rank) for rank, row in enumerate(ranked, start=1)]


def _selected_output_rows(ranked_rows: Sequence[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    selected_ids = {str(row["model_id"]) for row in ranked_rows[:top_n]}
    for row in ranked_rows:
        if row.get("paper_candidate_gate_pass") or row.get("train_dominant_sample_gate_pass"):
            selected_ids.add(str(row["model_id"]))
    for group in sorted({str(row.get("asset_group")) for row in ranked_rows}):
        group_rows = [row for row in ranked_rows if row.get("asset_group") == group]
        if group_rows:
            selected_ids.add(str(group_rows[0]["model_id"]))
    if ranked_rows:
        best_validation = max(ranked_rows, key=lambda row: float(row.get("validation_return") or -1e9))
        selected_ids.add(str(best_validation["model_id"]))
    return [dict(row) for row in ranked_rows if str(row["model_id"]) in selected_ids]


def _decision_rows(rows: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _selected_output_rows(_rank_rows(rows), top_n=limit):
        item = {field: row.get(field) for field in DECISION_FIELDS}
        item["decision_rank"] = row["rank"]
        out.append(item)
    return out


def _shadow_shortlist(ranked_rows: Sequence[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    shadows = [row for row in ranked_rows if not row.get("paper_candidate_gate_pass")]
    chosen = _selected_output_rows(shadows, top_n=limit) if shadows else []
    return {
        "handoff_kind": "asset_diverse_no_promotion_shadow_shortlist",
        "status": "no_new_asset_diverse_paper_promotion_shadow_shortlist",
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
    asset_groups: dict[str, int] = {}
    universe_roles: dict[str, int] = {}
    paper = [row for row in rows if row.get("paper_candidate_gate_pass")]
    for row in rows:
        decisions[str(row.get("decision"))] = decisions.get(str(row.get("decision")), 0) + 1
        families[str(row.get("family"))] = families.get(str(row.get("family")), 0) + 1
        symbols[str(row.get("symbol"))] = symbols.get(str(row.get("symbol")), 0) + 1
        asset_groups[str(row.get("asset_group"))] = asset_groups.get(str(row.get("asset_group")), 0) + 1
        universe_roles[str(row.get("universe_role"))] = universe_roles.get(str(row.get("universe_role")), 0) + 1
    best_validation = max(rows, key=lambda row: float(row.get("validation_return") or -1e9)) if rows else {}
    return {
        "candidate_count": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "family_counts": dict(sorted(families.items())),
        "symbol_counts": dict(sorted(symbols.items())),
        "asset_group_counts": dict(sorted(asset_groups.items())),
        "universe_role_counts": dict(sorted(universe_roles.items())),
        "train_dominant_sample_gate_pass_count": sum(
            bool(row.get("train_dominant_sample_gate_pass")) for row in rows
        ),
        "execution_efficiency_proxy_gate_pass_count": sum(
            bool(row.get("execution_efficiency_proxy_gate_pass")) for row in rows
        ),
        "paper_candidate_gate_pass_count": len(paper),
        "max_validation_return": float(best_validation.get("validation_return") or 0.0)
        if best_validation
        else None,
        "best_validation_model_id": best_validation.get("model_id") if best_validation else None,
        "best_paper_candidate_model_id": paper[0].get("model_id") if paper else None,
        "ready_for_paper": bool(paper),
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _paper_handoff(paper_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    handoff = feedback._paper_testnet_handoff(paper_rows)
    handoff["handoff_kind"] = "paper_testnet_only_asset_diverse_strategy_discovery"
    handoff["strategy_scope"] = STRATEGY_SCOPE
    handoff["cross_asset_conditioned"] = True
    return handoff


def _coverage_manifest(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    promotion_symbols: set[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for symbol in symbols:
        symbol_payload: dict[str, Any] = {
            "asset_group": _asset_group(symbol),
            "universe_role": "promotion_eligible" if symbol in promotion_symbols else "shadow_probe_only",
            "timeframes": {},
        }
        for timeframe in timeframes:
            frame = bars_by_symbol_tf[(symbol, timeframe)]
            split_counts = {
                split: int(feedback._split_mask(frame["datetime"], split).sum())
                for split in feedback.SPLIT_ORDER
            }
            symbol_payload["timeframes"][timeframe] = {
                "first_datetime": str(frame["datetime"].min()) if not frame.empty else None,
                "last_datetime": str(frame["datetime"].max()) if not frame.empty else None,
                "bar_count": len(frame),
                "split_bar_counts": split_counts,
            }
        out[symbol] = symbol_payload
    return out


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("discovery_summary") or {})
    top = list(payload.get("top_candidates") or [])[:15]
    paper = [row for row in payload.get("top_candidates", []) if row.get("paper_candidate_gate_pass")][:15]
    lines = [
        "# Alpha Zoo asset-diverse strategy discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "Research/paper-testnet only. Single-symbol state rules use cross-asset filters;",
        "locked-OOS remains gate/report-only after train+validation ranking freeze.",
        "",
        f"- Candidates evaluated: `{summary.get('candidate_count')}`",
        f"- Asset groups: `{summary.get('asset_group_counts')}`",
        f"- Paper candidate gate pass: `{summary.get('paper_candidate_gate_pass_count')}`",
        "- `ready_for_real=false`",
        "- `real_money_execution=false`",
        f"- Runner peak RSS MiB: `{float(payload.get('runner_peak_rss_mib') or 0.0):.3f}`",
        "",
        "## Top train+validation-ranked rows",
        "",
        "| Rank | Symbol | Group | TF | Family | Train | Val | OOS | RPT train/val/OOS | Decision |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in top:
        rpt = "/".join(
            "NA"
            if row.get(f"{split}_return_per_turnover_proxy_bps") is None
            else f"{float(row[f'{split}_return_per_turnover_proxy_bps']):.2f}"
            for split in feedback.SPLIT_ORDER
        )
        lines.append(
            f"| {row.get('rank')} | {row.get('symbol')} | {row.get('asset_group')} | "
            f"{row.get('timeframe')} | {row.get('family')} | "
            f"{float(row.get('train_return') or 0.0):.4%} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {rpt} | "
            f"{row.get('decision')} |"
        )
    lines.extend(["", "## Paper/testnet-only candidates", ""])
    if not paper:
        lines.append("No asset-diverse paper/testnet candidates passed the strict primary gates.")
    else:
        lines.extend(
            [
                "| Rank | Model | Symbol | Group | TF | Family | Train | Val | OOS |",
                "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in paper:
            lines.append(
                f"| {row.get('rank')} | `{row.get('model_id')}` | {row.get('symbol')} | "
                f"{row.get('asset_group')} | {row.get('timeframe')} | {row.get('family')} | "
                f"{float(row.get('train_return') or 0.0):.4%} | "
                f"{float(row.get('validation_return') or 0.0):.4%} | "
                f"{float(row.get('locked_oos_return') or 0.0):.4%} |"
            )
    return "\n".join(lines) + "\n"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    symbols = _parse_csv_symbols(args.symbols)
    shadow_symbols = _parse_csv_symbols(args.shadow_symbols)
    timeframes = feedback._validate_timeframes(_parse_csv_symbols(args.timeframes.lower()))
    all_symbols = tuple(dict.fromkeys([*symbols, *shadow_symbols]))
    promotion_symbols = set(symbols)

    bars_by_symbol_tf = feedback.load_requested_bars(all_symbols, timeframes=timeframes, data_root=data_root)
    rows = discover_asset_diverse_candidates(
        bars_by_symbol_tf,
        symbols=symbols,
        shadow_symbols=shadow_symbols,
        timeframes=timeframes,
    )
    ranked = _rank_rows(rows)
    top_rows = _selected_output_rows(ranked, top_n=int(args.top_n))
    paper_rows = [row for row in ranked if row.get("paper_candidate_gate_pass")]
    handoff = _paper_handoff(paper_rows)
    shadow = _shadow_shortlist(ranked, limit=int(args.shadow_limit))
    decision_status = (
        "paper_testnet_candidate_after_fill_preflight"
        if paper_rows
        else "no_new_asset_diverse_paper_promotion_shadow_shortlist"
    )

    latest_json = output_dir / "alpha_zoo_asset_diverse_strategy_discovery_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_asset_diverse_strategy_discovery_{_timestamp()}.json"
    latest_md = output_dir / "alpha_zoo_asset_diverse_strategy_discovery_latest.md"
    candidates_csv = output_dir / "asset_diverse_strategy_candidates_latest.csv"
    decisions_csv = output_dir / "asset_diverse_strategy_decisions_latest.csv"
    shadows_csv = output_dir / "asset_diverse_strategy_shadow_hypotheses_latest.csv"
    handoff_json = output_dir / "paper_testnet_handoff_latest.json"
    handoff_md = output_dir / "paper_testnet_handoff_latest.md"
    no_promotion_json = output_dir / "no_promotion_shadow_shortlist_latest.json"
    generation_log = output_dir / "artifact_generation_validation_latest.log"

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_asset_diverse_strategy_discovery",
        "generated_at_utc": _utc_now_iso(),
        "ready_for_paper": bool(paper_rows),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": bool(paper_rows),
        "paper_testnet_only": True,
        "decision_status": decision_status,
        "research_primary_round_trip_cost_bps": feedback.PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": feedback.AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "bbo_spread_multiplier": feedback.BBO_SPREAD_MULTIPLIER,
        "return_per_turnover_threshold_bps": feedback.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "bar_construction": BAR_CONSTRUCTION,
        "strategy_scope": STRATEGY_SCOPE,
        "source_data": {
            "ohlcv_root": str(data_root),
            "promotion_symbols": list(symbols),
            "shadow_symbols": list(shadow_symbols),
            "timeframes": list(timeframes),
            "bar_source": "local Binance 1s OHLCV parquet aggregated to native 30m base",
            "coverage_manifest": _coverage_manifest(
                bars_by_symbol_tf,
                symbols=all_symbols,
                timeframes=timeframes,
                promotion_symbols=promotion_symbols,
            ),
        },
        "selection_policy": {
            "score_formula": (
                "asset_diverse_train_validation_score(validation, train, train/validation RPT, "
                "validation halves, validation MDD)"
            ),
            "objective_inputs": ["train", "validation"],
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "ranking_freeze_before_locked_oos_gate": True,
            "locked_oos_role": "gate_report_only_after_train_validation_candidate_freeze",
            "train_return_below_validation_return_is_promotion_reject": True,
            "shadow_symbols_are_not_promotable": True,
            "no_calendar_date_hack": True,
        },
        "strategy_families": STRATEGY_FAMILIES,
        "asset_groups": ASSET_GROUPS,
        "promotion_thresholds": feedback.PROMOTION_THRESHOLDS,
        "split_manifest": {
            split: {
                "start": str(start),
                "end": str(end),
                "role": "objective_selection" if split != "locked_oos" else "gate_report_only",
            }
            for split, (start, end) in feedback.SPLITS.items()
        },
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
            "latest_markdown": str(latest_md),
            "candidates_csv": str(candidates_csv),
            "decisions_csv": str(decisions_csv),
            "shadow_hypotheses_csv": str(shadows_csv),
            "paper_testnet_handoff_json": str(handoff_json),
            "paper_testnet_handoff_markdown": str(handoff_md),
            "no_promotion_shadow_shortlist_json": str(no_promotion_json),
            "artifact_generation_validation_log": str(generation_log),
        },
    }

    feedback._write_json(latest_json, payload)
    feedback._write_json(timestamped_json, payload)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    feedback._write_csv(candidates_csv, top_rows, CANDIDATE_FIELDS)
    feedback._write_csv(decisions_csv, payload["decision_rows"], DECISION_FIELDS)
    feedback._write_csv(shadows_csv, shadow.get("shadows", []), CANDIDATE_FIELDS)
    feedback._write_json(handoff_json, handoff)
    handoff_md.write_text(feedback._handoff_markdown(handoff), encoding="utf-8")
    feedback._write_json(no_promotion_json, shadow)
    generation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"candidate_count={payload['discovery_summary']['candidate_count']}",
                f"paper_candidate_gate_pass_count={payload['discovery_summary']['paper_candidate_gate_pass_count']}",
                "uses_locked_oos_for_discovery=false",
                "uses_locked_oos_for_selection=false",
                "uses_locked_oos_for_objective=false",
                "uses_locked_oos_for_pruning=false",
                "uses_locked_oos_for_parameter_fitting=false",
                "ready_for_real=false",
                "real_money_execution=false",
                f"runner_peak_rss_mib={payload['runner_peak_rss_mib']:.3f}",
                f"latest_json={latest_json}",
                f"timestamped_json={timestamped_json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(feedback.DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--shadow-symbols", default=",".join(DEFAULT_SHADOW_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--top-n", type=int, default=260)
    parser.add_argument("--shadow-limit", type=int, default=120)
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
                "best_paper_candidate_model_id": summary["best_paper_candidate_model_id"],
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
