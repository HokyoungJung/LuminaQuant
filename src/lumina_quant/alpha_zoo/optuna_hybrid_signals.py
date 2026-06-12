"""Signal math and bar-window helpers for the Alpha Zoo Optuna hybrid live adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.market_data import normalize_timeframe_token

from .optuna_hybrid_config import (
    INTRABAR_ATR_LOOKBACK,
    INTRABAR_MAX_STOP_COST_MULT,
    INTRABAR_MIN_STOP_COST_MULT,
    INTRABAR_RISK_TIMEFRAMES,
    INTRABAR_STOP_ATR_MULT,
    RETURN_PER_TURNOVER_THRESHOLD_BPS,
    WATCH_SYMBOLS,
    SourceSleeve,
    SleeveDecision,
    _symbol_aliases,
)
from .native_live_signal_backend import (
    evaluate_debounced_state_native,
    evaluate_trailing_state_native,
)


def _bars_to_frame(bars: list[Any]) -> pd.DataFrame:
    rows: list[tuple[Any, float, float, float, float, float]] = []
    for bar in bars:
        if isinstance(bar, dict):
            rows.append(
                (
                    bar.get("time") or bar.get("datetime"),
                    float(bar.get("open", 0.0)),
                    float(bar.get("high", 0.0)),
                    float(bar.get("low", 0.0)),
                    float(bar.get("close", 0.0)),
                    float(bar.get("volume", 0.0)),
                )
            )
        elif isinstance(bar, (tuple, list)) and len(bar) >= 6:
            rows.append(
                (bar[0], float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]), float(bar[5]))
            )
    return pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])


def completed_bars_only(
    aggregator: Any,
    symbol: str,
    timeframe: str,
    lookback_bars: int,
) -> list[Any]:
    """Return completed bars and drop the active working bar exposed by the aggregator."""
    if aggregator is None:
        return []
    for alias in _symbol_aliases(symbol):
        try:
            bars = list(aggregator.get_bars(alias, timeframe, n=max(2, int(lookback_bars) + 1)))
        except TypeError:
            try:
                bars = list(aggregator.get_bars(alias, timeframe, max(2, int(lookback_bars) + 1)))
            except (AttributeError, KeyError, TypeError, ValueError):
                bars = []
        except (AttributeError, KeyError, ValueError):
            bars = []
        if len(bars) >= 2:
            return bars[:-1]
    return []


def _time_key(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(lookback).mean()


def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
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


def debounced_state_signal(
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
    short_entry = (
        pd.Series(False, index=long_entry.index)
        if short_entry is None
        else short_entry.fillna(False).astype(bool)
    )
    short_exit = (
        pd.Series(False, index=long_entry.index)
        if short_exit is None
        else short_exit.fillna(False).astype(bool)
    )
    native = evaluate_debounced_state_native(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side=side,
        min_hold_bars=min_hold_bars,
        cooldown_bars=cooldown_bars,
    )
    if native is not None:
        return native

    out = np.zeros(len(long_entry), dtype=float)
    state = 0.0
    bars_held = 10**9
    cooldown_remaining = 0
    for idx in range(len(long_entry)):
        can_exit = bars_held >= min_hold_bars
        exited = False
        long_exit_now = state > 0 and bool(long_exit.iloc[idx])
        short_exit_now = state < 0 and bool(short_exit.iloc[idx])
        if can_exit and (long_exit_now or short_exit_now):
            state = 0.0
            bars_held = 0
            cooldown_remaining = cooldown_bars
            exited = True
        if state == 0.0:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif not exited:
                if side in {"long_only", "long_short"} and bool(long_entry.iloc[idx]):
                    state = 1.0
                    bars_held = 0
                elif side in {"short_only", "long_short"} and bool(short_entry.iloc[idx]):
                    state = -1.0
                    bars_held = 0
        out[idx] = state
        if state != 0.0:
            bars_held += 1
    return out


def trailing_state_signal(
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
    long_entry = long_entry.fillna(False).astype(bool)
    short_entry = short_entry.fillna(False).astype(bool)
    long_exit = long_exit.fillna(False).astype(bool)
    short_exit = short_exit.fillna(False).astype(bool)
    native = evaluate_trailing_state_native(
        close,
        long_entry,
        short_entry,
        long_exit,
        short_exit,
        atr,
        side=side,
        min_hold_bars=min_hold_bars,
        cooldown_bars=cooldown_bars,
        trail_atr_mult=trail_atr_mult,
    )
    if native is not None:
        return native

    close_values = close.astype(float).to_numpy()
    atr_values = atr.astype(float).to_numpy()
    long_entry_values = long_entry.to_numpy()
    short_entry_values = short_entry.to_numpy()
    long_exit_values = long_exit.to_numpy()
    short_exit_values = short_exit.to_numpy()

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
    source = series.copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.index))
    if pd.api.types.is_bool_dtype(source):
        aligned_bool = source.reindex(index, fill_value=False).astype(bool)
        return pd.Series(aligned_bool.to_numpy(), index=datetimes.index)
    aligned_numeric = pd.to_numeric(source.reindex(index).ffill(), errors="coerce")
    return pd.Series(aligned_numeric.to_numpy(dtype=float), index=datetimes.index)


def _panel_state(panel: pd.DataFrame, lookback: int) -> dict[str, pd.Series | pd.DataFrame]:
    returns = panel.pct_change().mean(axis=1).fillna(0.0)
    market_index = (1.0 + returns).cumprod()
    momentum = panel / panel.shift(lookback) - 1.0
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    reverse_ranks = momentum.rank(axis=1, ascending=True, method="first")
    valid_count = momentum.notna().sum(axis=1).replace(0, np.nan)
    return {
        "market_momentum": market_index / market_index.shift(lookback) - 1.0,
        "breadth": (momentum > 0.0).sum(axis=1) / valid_count,
        "dispersion": momentum.std(axis=1, ddof=1),
        "momentum": momentum,
        "ranks": ranks,
        "rank_pct": momentum.rank(axis=1, ascending=False, pct=True),
        "reverse_ranks": reverse_ranks,
    }


def _merge_close(frame: pd.DataFrame, other: pd.DataFrame, column: str) -> pd.DataFrame:
    close_frame = other[["datetime", "close"]].rename(columns={"close": column})
    return frame.merge(close_frame, on="datetime", how="left").ffill()


def _frame_for(aggregator: Any, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
    return _bars_to_frame(completed_bars_only(aggregator, symbol, timeframe, lookback))


def _intrabar_risk_frame_for(
    aggregator: Any,
    symbol: str,
    sleeve_timeframe: str,
    lookback: int,
) -> tuple[pd.DataFrame, str]:
    for timeframe in (*INTRABAR_RISK_TIMEFRAMES, sleeve_timeframe):
        frame = _frame_for(aggregator, symbol, timeframe, lookback)
        if len(frame) >= max(INTRABAR_ATR_LOOKBACK + 2, 4):
            return frame, timeframe
    return pd.DataFrame(), sleeve_timeframe


def _latest_atr_pct(frame: pd.DataFrame) -> float | None:
    if frame.empty or len(frame) < INTRABAR_ATR_LOOKBACK + 2:
        return None
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    atr = _atr(high, low, close, INTRABAR_ATR_LOOKBACK)
    latest_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    latest_close = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else 0.0
    if latest_atr <= 0.0 or latest_close <= 0.0:
        return None
    return float(latest_atr / latest_close)


def _clamp_stop_distance_pct(atr_pct: float | None) -> float:
    floor = (RETURN_PER_TURNOVER_THRESHOLD_BPS / 10_000.0) * INTRABAR_MIN_STOP_COST_MULT
    ceiling = (RETURN_PER_TURNOVER_THRESHOLD_BPS / 10_000.0) * INTRABAR_MAX_STOP_COST_MULT
    candidate = floor if atr_pct is None else max(floor, float(atr_pct) * INTRABAR_STOP_ATR_MULT)
    return float(min(max(candidate, floor), ceiling))


def _evaluate_debounced(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
) -> SleeveDecision | None:
    merged = _merge_close(frame, btc_frame, "btc_close")
    close = merged["close"].astype(float)
    high = merged["high"].astype(float)
    low = merged["low"].astype(float)
    btc_close = merged["btc_close"].astype(float)
    btc_regime_fast = btc_close / btc_close.shift(12) - 1.0
    lookback = sleeve.lookback
    momentum = close / close.shift(lookback) - 1.0
    realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
    trend_strength = momentum.abs() / (realized * np.sqrt(float(lookback))).replace(0.0, np.nan)
    adx = _adx_proxy(high, low, close, max(6, lookback))
    if sleeve.filter_label == "none":
        common_filter = pd.Series(True, index=frame.index)
    elif sleeve.filter_label.startswith("low_vol"):
        quantile = 0.65 if "q65" in sleeve.filter_label else 0.55
        common_filter = _volatility_mask(close, lookback, quantile)
    elif sleeve.filter_label.startswith("adx20"):
        common_filter = adx >= 20.0
    elif sleeve.filter_label.startswith("trend_strength2"):
        common_filter = trend_strength >= 2.0
    else:
        common_filter = adx >= 15.0
    long_entry = (momentum > sleeve.entry_threshold) & (btc_regime_fast > -0.02) & common_filter
    short_entry = (momentum < -sleeve.entry_threshold) & (btc_regime_fast < 0.02) & common_filter
    long_exit = (momentum < sleeve.exit_threshold) | (~common_filter)
    short_exit = (momentum > -sleeve.exit_threshold) | (~common_filter)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side=sleeve.side,
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = merged.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "momentum": float(momentum.iloc[-1]) if pd.notna(momentum.iloc[-1]) else None,
            "btc_regime_fast": float(btc_regime_fast.iloc[-1])
            if pd.notna(btc_regime_fast.iloc[-1])
            else None,
        },
    )


def _evaluate_booster(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
) -> SleeveDecision | None:
    merged = _merge_close(frame, btc_frame, "btc_close")
    close = merged["close"].astype(float)
    high = merged["high"].astype(float)
    low = merged["low"].astype(float)
    btc_close = merged["btc_close"].astype(float)
    hours = _timeframe_hours(sleeve.timeframe)
    btc_lookback = max(2, int(24 / hours))
    rel = close / btc_close.replace(0.0, np.nan)
    rel_momentum = rel / rel.shift(max(2, int(12 / hours))) - 1.0
    btc_momentum = btc_close / btc_close.shift(btc_lookback) - 1.0
    lookback = sleeve.lookback
    atr = _atr(high, low, close, max(6, lookback))
    roll_high = high.shift(1).rolling(lookback).max()
    roll_low = low.shift(1).rolling(lookback).min()
    mid = (roll_high + roll_low) / 2.0
    adx = _adx_proxy(high, low, close, max(6, lookback // 2))
    common_long = (rel_momentum > sleeve.rel_threshold) & (btc_momentum > -0.015) & (adx >= 15.0)
    common_short = (rel_momentum < -sleeve.rel_threshold) & (btc_momentum < 0.015) & (adx >= 15.0)
    long_entry = (close > roll_high + sleeve.atr_mult * atr) & common_long
    short_entry = (close < roll_low - sleeve.atr_mult * atr) & common_short
    long_exit = (close < mid) | (rel_momentum < -0.005)
    short_exit = (close > mid) | (rel_momentum > 0.005)
    signal = trailing_state_signal(
        close,
        long_entry,
        short_entry,
        long_exit,
        short_exit,
        atr,
        side="long_short",
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
        trail_atr_mult=sleeve.trail_atr_mult,
    )
    if len(signal) == 0:
        return None
    latest = merged.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "rel_momentum": float(rel_momentum.iloc[-1])
            if pd.notna(rel_momentum.iloc[-1])
            else None,
            "btc_momentum": float(btc_momentum.iloc[-1])
            if pd.notna(btc_momentum.iloc[-1])
            else None,
            "adx_proxy": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None,
        },
    )


def _build_panel(
    aggregator: Any,
    timeframe: str,
    lookback: int,
    symbols: tuple[str, ...] = WATCH_SYMBOLS,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in tuple(dict.fromkeys(symbols)):
        frame = _frame_for(aggregator, symbol, timeframe, lookback)
        if frame.empty:
            return pd.DataFrame()
        frames.append(frame[["datetime", "close"]].assign(symbol=str(symbol).replace("/", "")))
    panel = pd.concat(frames).pivot(index="datetime", columns="symbol", values="close")
    return panel.sort_index().dropna(how="any")


def _evaluate_residual(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    base_frame: pd.DataFrame,
    panel: pd.DataFrame,
) -> SleeveDecision | None:
    if panel.empty:
        return None
    merged = _merge_close(frame, base_frame, "base_close")
    close = merged["close"].astype(float)
    base_close = merged["base_close"].astype(float)
    ratio = np.log(close / base_close.replace(0.0, np.nan))
    market_mom = _align_to_frame(_panel_state(panel, 24)["market_momentum"], merged["datetime"])
    lookback = sleeve.lookback
    z = _rolling_zscore(ratio, lookback)
    target_mom = close / close.shift(max(4, lookback // 3)) - 1.0
    reclaim_long = (z.shift(1) < -sleeve.z_entry) & (z >= -sleeve.z_entry)
    reclaim_short = (z.shift(1) > sleeve.z_entry) & (z <= sleeve.z_entry)
    long_entry = reclaim_long & (target_mom > -0.005) & (market_mom > -0.02)
    short_entry = reclaim_short & (target_mom < 0.005) & (market_mom < 0.02)
    long_exit = (z > -0.05) | (target_mom < -0.02)
    short_exit = (z < 0.05) | (target_mom > 0.02)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side="long_short",
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = merged.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "residual_z": float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else None,
            "target_momentum": float(target_mom.iloc[-1])
            if pd.notna(target_mom.iloc[-1])
            else None,
            "market_momentum": float(market_mom.iloc[-1])
            if pd.notna(market_mom.iloc[-1])
            else None,
        },
    )


def _latest_float(series: pd.Series) -> float | None:
    value = series.iloc[-1] if len(series) else np.nan
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else None


def _evaluate_cross_sectional_momentum_rank(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    panel: pd.DataFrame,
) -> SleeveDecision | None:
    symbol = sleeve.symbol.replace("/", "")
    if panel.empty or symbol not in panel.columns:
        return None
    state = _panel_state(panel, sleeve.lookback)
    momentum_panel = state["momentum"]
    rank_panel = state["rank_pct"]
    if not isinstance(momentum_panel, pd.DataFrame) or not isinstance(rank_panel, pd.DataFrame):
        return None
    symbol_momentum = _align_to_frame(momentum_panel[symbol], frame["datetime"])
    symbol_rank = _align_to_frame(rank_panel[symbol], frame["datetime"])
    market_mom = _align_to_frame(state["market_momentum"], frame["datetime"])
    breadth = _align_to_frame(state["breadth"], frame["datetime"])
    top_pct = float(sleeve.entry_threshold)
    exit_rank = float(sleeve.exit_threshold)
    market_guard = float(sleeve.market_guard)
    breadth_guard = float(sleeve.breadth_guard)
    long_entry = (
        (symbol_rank <= top_pct)
        & (symbol_momentum > 0.0)
        & (market_mom > -market_guard)
        & (breadth >= breadth_guard)
    )
    short_entry = (
        (symbol_rank >= 1.0 - top_pct)
        & (symbol_momentum < 0.0)
        & (market_mom < market_guard)
        & (breadth <= 1.0 - breadth_guard)
    )
    long_exit = (symbol_rank > exit_rank) | (symbol_momentum < 0.0)
    short_exit = (symbol_rank < 1.0 - exit_rank) | (symbol_momentum > 0.0)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side=sleeve.side,
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = frame.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "symbol_momentum": _latest_float(symbol_momentum),
            "rank_pct": _latest_float(symbol_rank),
            "market_momentum": _latest_float(market_mom),
            "breadth": _latest_float(breadth),
        },
    )


def _evaluate_voladj_efficiency_repair(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    panel: pd.DataFrame,
) -> SleeveDecision | None:
    if panel.empty:
        return None
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    lookback = sleeve.lookback
    momentum = close / close.shift(lookback) - 1.0
    realized = close.pct_change().rolling(max(6, lookback // 2)).std(ddof=1)
    vol_adjusted = momentum / (realized * np.sqrt(float(lookback))).replace(0.0, np.nan)
    adx = _adx_proxy(high, low, close, max(6, lookback // 2))
    market_mom = _align_to_frame(
        _panel_state(panel, lookback)["market_momentum"], frame["datetime"]
    )
    market_abs_max = float(sleeve.market_abs_max) if sleeve.market_abs_max > 0.0 else 1.0
    common = (adx >= float(sleeve.adx_min)) & (market_mom.abs() < market_abs_max)
    entry = float(sleeve.entry_threshold)
    exit_z = float(sleeve.exit_threshold)
    long_entry = (vol_adjusted > entry) & common
    short_entry = (vol_adjusted < -entry) & common
    long_exit = (vol_adjusted < exit_z) | (~common)
    short_exit = (vol_adjusted > -exit_z) | (~common)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side=sleeve.side,
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = frame.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "vol_adjusted_momentum": _latest_float(vol_adjusted),
            "market_momentum": _latest_float(market_mom),
            "adx_proxy": _latest_float(adx),
        },
    )


def _evaluate_trend_pullback_reclaim(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    panel: pd.DataFrame,
) -> SleeveDecision | None:
    if panel.empty:
        return None
    close = frame["close"].astype(float)
    slow = sleeve.lookback
    fast = max(4, round(slow / max(1, int(sleeve.fast_divisor or 4))))
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    trend_slope = ema_slow / ema_slow.shift(max(2, slow // 6)) - 1.0
    distance = (close - ema_fast) / ema_fast.replace(0.0, np.nan)
    dist_z = _rolling_zscore(distance, max(24, slow))
    market_lookback = min(72, max(6, slow // 2))
    market_mom = _align_to_frame(
        _panel_state(panel, market_lookback)["market_momentum"], frame["datetime"]
    )
    pullback_z = float(sleeve.entry_threshold)
    trend_min = float(sleeve.trend_slope_min)
    market_guard = float(sleeve.market_guard)
    long_entry = (
        (ema_fast > ema_slow)
        & (trend_slope > trend_min)
        & (dist_z.shift(1) <= pullback_z)
        & (close > ema_fast)
        & (market_mom > -market_guard)
    )
    short_entry = (
        (ema_fast < ema_slow)
        & (trend_slope < -trend_min)
        & (dist_z.shift(1) >= -pullback_z)
        & (close < ema_fast)
        & (market_mom < market_guard)
    )
    long_exit = (close < ema_slow) | (trend_slope < 0.0)
    short_exit = (close > ema_slow) | (trend_slope > 0.0)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side=sleeve.side,
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = frame.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "pullback_z": _latest_float(dist_z),
            "trend_slope": _latest_float(trend_slope),
            "market_momentum": _latest_float(market_mom),
        },
    )


def _evaluate_voladj(
    sleeve: SourceSleeve,
    frame: pd.DataFrame,
    btc_frame: pd.DataFrame,
) -> SleeveDecision | None:
    merged = _merge_close(frame, btc_frame, "btc_close")
    close = merged["close"].astype(float)
    high = merged["high"].astype(float)
    low = merged["low"].astype(float)
    btc_close = merged["btc_close"].astype(float)
    hours = _timeframe_hours(sleeve.timeframe)
    btc_momentum = btc_close / btc_close.shift(max(2, int(12 / hours))) - 1.0
    lookback = sleeve.lookback
    momentum = close / close.shift(lookback) - 1.0
    realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
    vol_adjusted = momentum / (realized * np.sqrt(float(lookback))).replace(0.0, np.nan)
    adx = _adx_proxy(high, low, close, max(6, lookback))
    common = adx >= 15.0 if sleeve.filter_label == "adx15" else pd.Series(True, index=frame.index)
    long_entry = (vol_adjusted > sleeve.entry_threshold) & (btc_momentum > -0.02) & common
    short_entry = (vol_adjusted < -sleeve.entry_threshold) & (btc_momentum < 0.02) & common
    long_exit = (vol_adjusted < sleeve.exit_threshold) | (~common)
    short_exit = (vol_adjusted > -sleeve.exit_threshold) | (~common)
    signal = debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side="long_short",
        min_hold_bars=sleeve.min_hold_bars,
        cooldown_bars=sleeve.cooldown_bars,
    )
    if len(signal) == 0:
        return None
    latest = merged.iloc[-1]
    return SleeveDecision(
        signal=int(signal[-1]),
        completed_key=_time_key(latest["datetime"]),
        event_time=latest["datetime"],
        price=float(latest["close"]),
        diagnostics={
            "vol_adjusted_momentum": float(vol_adjusted.iloc[-1])
            if pd.notna(vol_adjusted.iloc[-1])
            else None,
            "btc_momentum": float(btc_momentum.iloc[-1])
            if pd.notna(btc_momentum.iloc[-1])
            else None,
            "adx_proxy": float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else None,
        },
    )


def _timeframe_hours(timeframe: str) -> float:
    token = normalize_timeframe_token(timeframe)
    if token.endswith("m"):
        return float(token[:-1]) / 60.0
    if token.endswith("h"):
        return float(token[:-1])
    if token.endswith("d"):
        return 24.0 * float(token[:-1])
    return 1.0


__all__ = [
    "_evaluate_cross_sectional_momentum_rank",
    "_evaluate_trend_pullback_reclaim",
    "_evaluate_voladj_efficiency_repair",
    "completed_bars_only",
    "debounced_state_signal",
    "trailing_state_signal",
]
