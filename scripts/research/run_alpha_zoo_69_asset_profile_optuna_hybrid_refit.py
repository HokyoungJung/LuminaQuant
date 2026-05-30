#!/usr/bin/env python3
"""Per-asset/profile Optuna rebuild for the 69-asset Alpha Zoo hybrid.

This runner fixes the broad 69-asset pass that only optimized a diversified
stream blend.  It rebuilds the three source profiles themselves across the
expanded universe:

1. Tune every symbol/profile pair with Optuna over timeframe, family, entry/exit,
   hold/cooldown, side, and integer leverage.
2. Tune each source profile's sleeve allocations with Optuna under its gross/MDD
   budget.
3. Feed the three rebuilt profile streams into the same v3.5/v3.6 Optuna hybrid
   machinery used by the live handoff runner.

No locked test/OOS set is reserved for this live-refit style artifact.  The most
recent 8 complete weeks remain validation/report evidence; all real-money flags
are intentionally false.
"""

from __future__ import annotations

import argparse
import csv
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

from lumina_quant.optimization.search_policy import (  # noqa: E402
    optimization_search_policy_payload,
    run_optuna_study,
)
from lumina_quant.research_universe import BINANCE_EXTENDED_RESEARCH_SYMBOLS  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research import run_alpha_zoo_integer_leverage_hybrid_decision as grid_hybrid  # noqa: E402
from scripts.research import run_alpha_zoo_integer_leverage_optuna_hybrid_decision as optuna_hybrid  # noqa: E402

try:  # pragma: no cover - dependency availability is covered by runtime/tests.
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore[assignment]

DEFAULT_OUTPUT_DIR = (
    broad69.ALPHA_V2_ROOT / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_20260530"
)
DEFAULT_DATA_ROOT = broad69.DEFAULT_DATA_ROOT
DEFAULT_TIMEFRAMES = broad69.DEFAULT_TIMEFRAMES
DEFAULT_ALLOCATION_FRACTION = 0.10
DEFAULT_ASSET_TRIALS = 36
DEFAULT_PROFILE_TRIALS = 96
DEFAULT_HYBRID_TRIALS = 160

DOMAIN_ANCHORS: dict[str, str] = {
    "crypto_beta_btc": "BTCUSDT",
    "crypto_liquidity_eth": "ETHUSDT",
    "high_beta_alt_sol": "SOLUSDT",
    "us_equity_beta_spy": "SPYUSDT",
    "tech_growth_qqq": "QQQUSDT",
    "precious_metal_gold": "XAUUSDT",
    "silver_metal_beta": "XAGUSDT",
    "energy_crude_beta": "CLUSDT",
}

PROFILE_SPECS = (
    {
        "profile_id": "balanced_mdd12_gross5_69_asset_profile_optuna",
        "max_validation_mdd": 0.12,
        "max_train_mdd": 0.35,
        "max_gross_notional": 5.0,
        "max_sleeves": 24,
        "max_integer_leverage": 6,
        "min_validation_return": 0.02,
        "top_symbol_share_cap": 0.25,
    },
    {
        "profile_id": "growth_mdd20_gross8_69_asset_profile_optuna",
        "max_validation_mdd": 0.20,
        "max_train_mdd": 0.45,
        "max_gross_notional": 8.0,
        "max_sleeves": 36,
        "max_integer_leverage": 10,
        "min_validation_return": 0.02,
        "top_symbol_share_cap": 0.30,
    },
    {
        "profile_id": "aggressive_mdd30_gross10_69_asset_profile_optuna",
        "max_validation_mdd": 0.30,
        "max_train_mdd": 0.60,
        "max_gross_notional": 10.0,
        "max_sleeves": 48,
        "max_integer_leverage": 12,
        "min_validation_return": 0.02,
        "top_symbol_share_cap": 0.35,
    },
)

CANDIDATE_FIELDS = [
    "profile_id",
    "symbol",
    "asset_group",
    "family",
    "timeframe",
    "side",
    "lookback_bars",
    "threshold",
    "exit_threshold",
    "min_hold_bars",
    "cooldown_bars",
    "integer_leverage",
    "notional_fraction",
    "train_return",
    "validation_return",
    "train_mdd",
    "validation_mdd",
    "train_trade_event_count",
    "validation_trade_event_count",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "train_validation_score",
    "profile_objective_score",
    "dominant_anchor",
    "dominant_anchor_abs_corr",
    "ready_for_paper",
    "rejection_reasons",
]

SLEEVE_FIELDS = [
    *CANDIDATE_FIELDS,
    "sleeve_multiplier",
    "weighted_notional_fraction",
    "profile_weight_rank",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    return broad69._json_safe(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", "utf-8")


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
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


@dataclass(frozen=True)
class FeatureCache:
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame]
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    _xsmom: dict[tuple[str, int], dict[str, Any]]
    _anchor_returns: dict[str, pd.DataFrame]

    def xsmom(self, timeframe: str, lookback: int) -> dict[str, Any]:
        key = (timeframe, int(lookback))
        if key in self._xsmom:
            return self._xsmom[key]
        bars_by_symbol = {
            symbol: self.bars_by_symbol_tf[(symbol, timeframe)] for symbol in self.symbols
        }
        panel = broad69._close_panel(bars_by_symbol, self.symbols)
        panel_returns = panel.pct_change().mean(axis=1).fillna(0.0)
        market_index = (1.0 + panel_returns).cumprod()
        momentum = panel / panel.shift(int(lookback)) - 1.0
        rank_pct = momentum.rank(axis=1, ascending=False, pct=True)
        breadth = (momentum > 0.0).sum(axis=1) / momentum.notna().sum(axis=1).replace(0, np.nan)
        out = {
            "momentum": momentum,
            "rank_pct": rank_pct,
            "market_momentum": market_index / market_index.shift(int(lookback)) - 1.0,
            "breadth": breadth,
        }
        self._xsmom[key] = out
        return out

    def anchor_returns(self, timeframe: str) -> pd.DataFrame:
        if timeframe in self._anchor_returns:
            return self._anchor_returns[timeframe]
        frames = []
        for anchor_name, symbol in DOMAIN_ANCHORS.items():
            if symbol not in self.symbols:
                continue
            frame = self.bars_by_symbol_tf.get((symbol, timeframe))
            if frame is None or frame.empty:
                continue
            series = (
                frame[["datetime", "close"]]
                .assign(datetime=lambda pdf: pd.to_datetime(pdf["datetime"]))
                .set_index("datetime")["close"]
                .astype(float)
                .pct_change()
                .rename(anchor_name)
            )
            frames.append(series)
        if not frames:
            out = pd.DataFrame()
        else:
            out = pd.concat(frames, axis=1).sort_index().fillna(0.0)
        self._anchor_returns[timeframe] = out
        return out


def _anchor_correlation_payload(
    returns: pd.Series, *, timeframe: str, cache: FeatureCache, windows: broad69.SplitWindows
) -> dict[str, Any]:
    anchors = cache.anchor_returns(timeframe)
    if anchors.empty or returns.empty:
        return {"anchor_correlations": {}, "dominant_anchor": None, "dominant_anchor_abs_corr": 0.0}
    aligned = anchors.join(returns.rename("strategy_return"), how="inner").fillna(0.0)
    if aligned.empty:
        return {"anchor_correlations": {}, "dominant_anchor": None, "dominant_anchor_abs_corr": 0.0}
    train_mask = broad69._split_mask(pd.Series(aligned.index), "train", windows)
    val_mask = broad69._split_mask(pd.Series(aligned.index), "validation", windows)
    selected = aligned.loc[train_mask | val_mask]
    if len(selected) < 3 or float(selected["strategy_return"].std(ddof=1)) <= 0.0:
        return {"anchor_correlations": {}, "dominant_anchor": None, "dominant_anchor_abs_corr": 0.0}
    correlations: dict[str, float] = {}
    for column in anchors.columns:
        if float(selected[column].std(ddof=1)) <= 0.0:
            continue
        value = selected["strategy_return"].corr(selected[column])
        if pd.notna(value) and math.isfinite(float(value)):
            correlations[str(column)] = float(value)
    if not correlations:
        return {"anchor_correlations": {}, "dominant_anchor": None, "dominant_anchor_abs_corr": 0.0}
    dominant, corr = max(correlations.items(), key=lambda item: abs(item[1]))
    return {
        "anchor_correlations": dict(sorted(correlations.items())),
        "dominant_anchor": dominant,
        "dominant_anchor_abs_corr": abs(float(corr)),
    }


def _params_from_trial(trial: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    family = trial.suggest_categorical(
        "family",
        [
            "cross_sectional_momentum_rank",
            "volatility_adjusted_trend_persistence",
            "trend_pullback_reclaim",
        ],
    )
    max_lev = int(spec["max_integer_leverage"])
    params = {
        "family": family,
        "timeframe": trial.suggest_categorical(
            "timeframe", list(spec.get("_timeframes", DEFAULT_TIMEFRAMES))
        ),
        "side": trial.suggest_categorical("side", ["long_short", "long_only", "short_only"]),
        "integer_leverage": trial.suggest_int("integer_leverage", 1, max_lev),
        "min_hold_bars": trial.suggest_int("min_hold_bars", 6, 72, step=6),
        "cooldown_bars": trial.suggest_int("cooldown_bars", 0, 18, step=3),
    }
    if family == "cross_sectional_momentum_rank":
        params.update(
            {
                "lookback_bars": trial.suggest_categorical(
                    "xsmom_lookback_bars", [6, 12, 24, 48, 72]
                ),
                "threshold": trial.suggest_float("xsmom_top_pct", 0.05, 0.30, step=0.05),
                "exit_threshold": trial.suggest_float("xsmom_exit_rank", 0.35, 0.65, step=0.05),
                "market_guard": trial.suggest_float("xsmom_market_guard", 0.00, 0.08, step=0.01),
                "breadth_guard": trial.suggest_float("xsmom_breadth_guard", 0.20, 0.50, step=0.05),
            }
        )
    elif family == "volatility_adjusted_trend_persistence":
        params.update(
            {
                "lookback_bars": trial.suggest_categorical(
                    "voladj_lookback_bars", [6, 12, 24, 48, 72]
                ),
                "threshold": trial.suggest_float("voladj_entry_z", 0.40, 2.00, step=0.10),
                "exit_threshold": trial.suggest_float("voladj_exit_z", 0.05, 0.60, step=0.05),
                "adx_min": trial.suggest_float("voladj_adx_min", 5.0, 30.0, step=2.5),
                "market_abs_max": trial.suggest_float(
                    "voladj_market_abs_max", 0.08, 0.40, step=0.04
                ),
            }
        )
    else:
        params.update(
            {
                "lookback_bars": trial.suggest_categorical(
                    "pullback_slow_bars", [24, 36, 48, 72, 96, 144]
                ),
                "fast_divisor": trial.suggest_categorical("pullback_fast_divisor", [3, 4, 6]),
                "threshold": trial.suggest_float("pullback_z", -2.00, -0.25, step=0.25),
                "exit_threshold": 0.0,
                "trend_slope_min": trial.suggest_float(
                    "pullback_trend_slope_min", 0.0, 0.03, step=0.005
                ),
                "market_guard": trial.suggest_float("pullback_market_guard", 0.00, 0.08, step=0.01),
            }
        )
    return params


def _series_from_frame(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype(float).reset_index(drop=True)


def _candidate_from_params(
    *,
    symbol: str,
    profile_id: str,
    params: Mapping[str, Any],
    cache: FeatureCache,
    windows: broad69.SplitWindows,
    allocation_fraction: float,
) -> broad69.CandidateStream:
    timeframe = str(params["timeframe"])
    frame = cache.bars_by_symbol_tf[(symbol, timeframe)]
    if frame.empty:
        raise ValueError(f"empty bars for {symbol} {timeframe}")
    datetimes = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
    close = _series_from_frame(frame, "close")
    high = _series_from_frame(frame, "high")
    low = _series_from_frame(frame, "low")
    family = str(params["family"])
    lookback = int(params["lookback_bars"])
    min_hold = int(params["min_hold_bars"])
    cooldown = int(params["cooldown_bars"])
    side = str(params["side"])

    if family == "cross_sectional_momentum_rank":
        xf = cache.xsmom(timeframe, lookback)
        symbol_momentum = pd.Series(
            xf["momentum"]
            .get(symbol, pd.Series(index=xf["momentum"].index, dtype=float))
            .reindex(datetimes)
            .to_numpy(),
            index=frame.index,
        )
        symbol_rank = pd.Series(
            xf["rank_pct"]
            .get(symbol, pd.Series(index=xf["rank_pct"].index, dtype=float))
            .reindex(datetimes)
            .to_numpy(),
            index=frame.index,
        )
        market_mom = pd.Series(
            xf["market_momentum"].reindex(datetimes).ffill().to_numpy(), index=frame.index
        )
        breadth = pd.Series(xf["breadth"].reindex(datetimes).ffill().to_numpy(), index=frame.index)
        top_pct = float(params["threshold"])
        exit_rank = float(params["exit_threshold"])
        market_guard = float(params.get("market_guard", 0.03))
        breadth_guard = float(params.get("breadth_guard", 0.35))
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
    elif family == "volatility_adjusted_trend_persistence":
        momentum = close / close.shift(lookback) - 1.0
        realized = close.pct_change().rolling(max(6, lookback // 2)).std(ddof=1)
        vol_adjusted = momentum / (realized * math.sqrt(float(lookback))).replace(0.0, np.nan)
        adx = broad69._adx_proxy(high, low, close, max(6, lookback // 2))
        market_mom = cache.xsmom(timeframe, lookback)["market_momentum"].reindex(datetimes).ffill()
        common = (adx >= float(params.get("adx_min", 12.0))) & (
            market_mom.abs().reset_index(drop=True) < float(params.get("market_abs_max", 0.25))
        )
        entry = float(params["threshold"])
        exit_z = float(params["exit_threshold"])
        long_entry = (vol_adjusted > entry) & common
        short_entry = (vol_adjusted < -entry) & common
        long_exit = (vol_adjusted < exit_z) | (~common)
        short_exit = (vol_adjusted > -exit_z) | (~common)
    else:
        slow = lookback
        fast = max(4, round(slow / max(1, int(params.get("fast_divisor", 4)))))
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        trend_slope = ema_slow / ema_slow.shift(max(2, slow // 6)) - 1.0
        distance = (close - ema_fast) / ema_fast.replace(0.0, np.nan)
        dist_z = broad69._rolling_zscore(distance, max(24, slow))
        pullback_z = float(params["threshold"])
        trend_min = float(params.get("trend_slope_min", 0.0))
        market_guard = float(params.get("market_guard", 0.03))
        market_mom = (
            cache.xsmom(timeframe, min(72, max(6, slow // 2)))["market_momentum"]
            .reindex(datetimes)
            .ffill()
            .reset_index(drop=True)
        )
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

    signal = broad69._debounced_state_signal(
        long_entry,
        long_exit,
        short_entry,
        short_exit,
        side=side,
        min_hold_bars=min_hold,
        cooldown_bars=cooldown,
    )
    leverage = int(params["integer_leverage"])
    sim = broad69.simulate_symbol(
        frame,
        signal,
        integer_leverage=leverage,
        allocation_fraction=allocation_fraction,
    )
    base = broad69._candidate_base(
        family=family,
        model_parts=(
            profile_id,
            family,
            timeframe,
            symbol,
            f"lb{lookback}",
            f"th{float(params['threshold']):.4g}",
            f"exit{float(params['exit_threshold']):.4g}",
            f"hold{min_hold}",
            f"cool{cooldown}",
            f"lev{leverage}",
            side,
        ),
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        lookback=lookback,
        threshold=float(params["threshold"]),
        exit_threshold=float(params["exit_threshold"]),
        min_hold=min_hold,
        cooldown=cooldown,
        integer_leverage=leverage,
        allocation_fraction=allocation_fraction,
    )
    base["profile_id"] = profile_id
    base["optuna_params"] = dict(params)
    row = broad69.finalize_candidate(
        base, sim, frame["datetime"], timeframe=timeframe, windows=windows
    )
    returns = pd.Series(sim.returns.astype(float), index=datetimes)
    position = pd.Series(sim.position.astype(float), index=datetimes)
    row.update(
        _anchor_correlation_payload(returns, timeframe=timeframe, cache=cache, windows=windows)
    )
    return broad69.CandidateStream(row=row, returns=returns, position=position)


def _candidate_objective(row: Mapping[str, Any], spec: Mapping[str, Any]) -> float:
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_mdd = _safe_float(row.get("train_mdd"))
    val_mdd = _safe_float(row.get("validation_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    anchor_abs = _safe_float(row.get("dominant_anchor_abs_corr"))
    train_events = int(row.get("train_trade_event_count") or 0)
    val_events = int(row.get("validation_trade_event_count") or 0)
    spike = max(0.0, validation - train)
    penalty = 0.0
    penalty += max(0.0, float(spec["min_validation_return"]) - validation) * 30.0
    penalty += max(0.0, val_mdd - float(spec["max_validation_mdd"])) * 20.0
    penalty += max(0.0, train_mdd - float(spec["max_train_mdd"])) * 6.0
    penalty += max(0.0, 10.0 - train_rpt) / 8.0
    penalty += max(0.0, 10.0 - val_rpt) / 6.0
    if train <= 0.0:
        penalty += 5.0 + abs(train) * 10.0
    if train_events < 20:
        penalty += (20 - train_events) / 10.0
    if val_events < 5:
        penalty += (5 - val_events) / 3.0
    # Domain filter: allow benchmark-aware strategies, but penalize single-anchor
    # clones.  Good sleeves can be related to BTC/ETH/SPY/QQQ/XAU/energy, but a
    # broad hybrid should not be just one benchmark in disguise.
    penalty += max(0.0, anchor_abs - 0.80) * 2.0
    return float(
        8.0 * validation
        + 1.2 * min(train, validation)
        + min(max(train_rpt, -20.0), 120.0) / 220.0
        + min(max(val_rpt, -20.0), 120.0) / 140.0
        - 4.0 * val_mdd
        - 1.0 * train_mdd
        - 4.0 * spike
        - penalty
    )


def tune_symbol_profile(
    *,
    symbol: str,
    spec: Mapping[str, Any],
    cache: FeatureCache,
    windows: broad69.SplitWindows,
    n_trials: int,
    seed: int,
    allocation_fraction: float,
) -> broad69.CandidateStream | None:
    if optuna is None:
        raise RuntimeError("Optuna is required for per-asset profile tuning")
    profile_id = str(spec["profile_id"])

    def objective(trial: Any) -> float:
        params = _params_from_trial(trial, spec)
        try:
            stream = _candidate_from_params(
                symbol=symbol,
                profile_id=profile_id,
                params=params,
                cache=cache,
                windows=windows,
                allocation_fraction=allocation_fraction,
            )
        except Exception as exc:  # pragma: no cover - defensive for bad data/trials.
            trial.set_user_attr("error", f"{exc.__class__.__name__}:{exc}")
            return -1e9
        score = _candidate_objective(stream.row, spec)
        trial.set_user_attr("score", score)
        for key in (
            "family",
            "timeframe",
            "side",
            "train_return",
            "validation_return",
            "train_mdd",
            "validation_mdd",
            "train_return_per_turnover_proxy_bps",
            "validation_return_per_turnover_proxy_bps",
        ):
            trial.set_user_attr(key, stream.row.get(key))
        return score

    study = run_optuna_study(
        optuna_module=optuna,
        objective=objective,
        n_trials=n_trials,
        direction="maximize",
        seed=seed,
        n_jobs=1,
        show_progress_bar=False,
    )
    best_params = dict(study.best_params)
    if not best_params:
        return None
    family = str(best_params.get("family", ""))
    # Optuna stores conditional family-prefixed keys; normalize them for deterministic rebuild.
    normalized = {
        "family": family,
        "timeframe": best_params["timeframe"],
        "side": best_params["side"],
        "integer_leverage": int(best_params["integer_leverage"]),
        "min_hold_bars": int(best_params["min_hold_bars"]),
        "cooldown_bars": int(best_params["cooldown_bars"]),
    }
    if family == "cross_sectional_momentum_rank":
        normalized.update(
            {
                "lookback_bars": int(best_params["xsmom_lookback_bars"]),
                "threshold": float(best_params["xsmom_top_pct"]),
                "exit_threshold": float(best_params["xsmom_exit_rank"]),
                "market_guard": float(best_params["xsmom_market_guard"]),
                "breadth_guard": float(best_params["xsmom_breadth_guard"]),
            }
        )
    elif family == "volatility_adjusted_trend_persistence":
        normalized.update(
            {
                "lookback_bars": int(best_params["voladj_lookback_bars"]),
                "threshold": float(best_params["voladj_entry_z"]),
                "exit_threshold": float(best_params["voladj_exit_z"]),
                "adx_min": float(best_params["voladj_adx_min"]),
                "market_abs_max": float(best_params["voladj_market_abs_max"]),
            }
        )
    else:
        normalized.update(
            {
                "lookback_bars": int(best_params["pullback_slow_bars"]),
                "fast_divisor": int(best_params["pullback_fast_divisor"]),
                "threshold": float(best_params["pullback_z"]),
                "exit_threshold": 0.0,
                "trend_slope_min": float(best_params["pullback_trend_slope_min"]),
                "market_guard": float(best_params["pullback_market_guard"]),
            }
        )
    stream = _candidate_from_params(
        symbol=symbol,
        profile_id=profile_id,
        params=normalized,
        cache=cache,
        windows=windows,
        allocation_fraction=allocation_fraction,
    )
    stream.row["profile_objective_score"] = _candidate_objective(stream.row, spec)
    stream.row["optuna_best_value"] = float(study.best_value)
    stream.row["optuna_n_trials"] = len(study.trials)
    return stream


def _aligned_matrix(streams: Sequence[broad69.CandidateStream]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        sorted(set().union(*(set(stream.returns.index) for stream in streams)))
    )
    return pd.DataFrame(
        {
            str(stream.row["model_id"]): stream.returns.reindex(index, fill_value=0.0)
            for stream in streams
        },
        index=index,
        dtype=float,
    ).sort_index()


def _split_return(values: np.ndarray) -> float:
    return float(np.prod(1.0 + values) - 1.0) if values.size else 0.0


def _profile_metrics_from_returns(
    returns: pd.Series,
    *,
    windows: broad69.SplitWindows,
    turnover_by_split: Mapping[str, float],
    events_by_split: Mapping[str, int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in broad69.SPLIT_ORDER:
        mask = broad69._split_mask(pd.Series(returns.index), split, windows)
        values = returns.to_numpy(dtype=float)[mask]
        total = _split_return(values)
        turnover = _safe_float(turnover_by_split.get(split))
        out[f"{split}_return"] = total
        out[f"{split}_mdd"] = broad69.max_drawdown(values)
        out[f"{split}_trade_event_count"] = int(events_by_split.get(split, 0))
        out[f"{split}_return_per_turnover_proxy_bps"] = (
            total * 10000.0 / turnover if turnover > 0.0 else None
        )
    return out


def _profile_concentration(
    streams: Sequence[broad69.CandidateStream], multipliers: np.ndarray
) -> dict[str, Any]:
    by_symbol: dict[str, float] = defaultdict(float)
    by_group: dict[str, float] = defaultdict(float)
    by_family: dict[str, float] = defaultdict(float)
    by_anchor: dict[str, float] = defaultdict(float)
    for mult, stream in zip(multipliers, streams, strict=True):
        if float(mult) <= 1e-9:
            continue
        notional = float(mult) * _safe_float(stream.row.get("notional_fraction"))
        by_symbol[str(stream.row["symbol"])] += notional
        by_group[str(stream.row["asset_group"])] += notional
        by_family[str(stream.row["family"])] += notional
        anchor = stream.row.get("dominant_anchor")
        if anchor:
            by_anchor[str(anchor)] += notional
    total = float(sum(by_symbol.values()))
    if total <= 0.0:
        return {
            "gross_notional_fraction": 0.0,
            "top_symbol": None,
            "top_symbol_share": 0.0,
            "top_asset_group": None,
            "top_asset_group_share": 0.0,
            "effective_symbol_count": 0.0,
            "symbol_shares": {},
            "asset_group_shares": {},
            "family_shares": {},
            "anchor_shares": {},
            "top_anchor": None,
            "top_anchor_share": 0.0,
        }
    symbol_shares = {k: float(v / total) for k, v in sorted(by_symbol.items())}
    group_shares = {k: float(v / total) for k, v in sorted(by_group.items())}
    family_shares = {k: float(v / total) for k, v in sorted(by_family.items())}
    anchor_shares = {k: float(v / total) for k, v in sorted(by_anchor.items())}
    top_symbol, top_share = max(symbol_shares.items(), key=lambda kv: kv[1])
    top_group, top_group_share = max(
        group_shares.items(), key=lambda kv: kv[1], default=(None, 0.0)
    )
    top_anchor, top_anchor_share = max(
        anchor_shares.items(), key=lambda kv: kv[1], default=(None, 0.0)
    )
    hhi = float(sum(v * v for v in symbol_shares.values()))
    return {
        "gross_notional_fraction": total,
        "top_symbol": top_symbol,
        "top_symbol_share": top_share,
        "top_asset_group": top_group,
        "top_asset_group_share": top_group_share,
        "effective_symbol_count": 1.0 / hhi if hhi > 0.0 else 0.0,
        "symbol_shares": symbol_shares,
        "asset_group_shares": group_shares,
        "family_shares": family_shares,
        "anchor_shares": anchor_shares,
        "top_anchor": top_anchor,
        "top_anchor_share": top_anchor_share,
    }


def tune_profile_allocations(
    *,
    spec: Mapping[str, Any],
    candidate_streams: Sequence[broad69.CandidateStream],
    windows: broad69.SplitWindows,
    n_trials: int,
    seed: int,
) -> tuple[grid_hybrid.ProfileStream, dict[str, Any], list[dict[str, Any]]]:
    if optuna is None:
        raise RuntimeError("Optuna is required for profile allocation tuning")
    profile_id = str(spec["profile_id"])
    ranked = sorted(
        candidate_streams,
        key=lambda stream: _safe_float(stream.row.get("profile_objective_score"), -1e18),
        reverse=True,
    )[: int(spec["max_sleeves"])]
    if not ranked:
        raise ValueError(f"no candidate streams for {profile_id}")
    matrix = _aligned_matrix(ranked)
    values = matrix.to_numpy(dtype=float)
    notionals = np.array([_safe_float(stream.row.get("notional_fraction")) for stream in ranked])

    def multipliers_from_trial(trial: Any) -> np.ndarray:
        raw = np.array(
            [trial.suggest_float(f"m_{idx}", 0.0, 1.0) for idx in range(len(ranked))], dtype=float
        )
        raw = np.where(raw < 0.05, 0.0, raw)
        if float(raw.sum()) <= 0.0:
            raw[
                int(np.argmax([_safe_float(s.row.get("profile_objective_score")) for s in ranked]))
            ] = 1.0
        gross = float(np.dot(raw, notionals))
        max_gross = float(spec["max_gross_notional"])
        if gross > max_gross and gross > 0.0:
            raw *= max_gross / gross
        return raw

    def score_multipliers(
        mult: np.ndarray,
    ) -> tuple[float, dict[str, Any], pd.Series, dict[str, float], dict[str, int]]:
        returns = pd.Series(values @ mult, index=matrix.index)
        turnover: dict[str, float] = {}
        events: dict[str, int] = {}
        for split in broad69.SPLIT_ORDER:
            turnover[split] = float(
                sum(
                    float(mult[idx])
                    * _safe_float(stream.row.get("notional_fraction"))
                    * int(stream.row.get(f"{split}_trade_event_count") or 0)
                    for idx, stream in enumerate(ranked)
                )
            )
            events[split] = int(
                sum(
                    int(stream.row.get(f"{split}_trade_event_count") or 0)
                    for idx, stream in enumerate(ranked)
                    if float(mult[idx]) > 1e-6
                )
            )
        metrics = _profile_metrics_from_returns(
            returns, windows=windows, turnover_by_split=turnover, events_by_split=events
        )
        conc = _profile_concentration(ranked, mult)
        train = _safe_float(metrics.get("train_return"))
        validation = _safe_float(metrics.get("validation_return"))
        train_mdd = _safe_float(metrics.get("train_mdd"))
        val_mdd = _safe_float(metrics.get("validation_mdd"))
        train_rpt = _safe_float(metrics.get("train_return_per_turnover_proxy_bps"), -100.0)
        val_rpt = _safe_float(metrics.get("validation_return_per_turnover_proxy_bps"), -100.0)
        spike = max(0.0, validation - train)
        penalty = 0.0
        penalty += max(0.0, float(spec["min_validation_return"]) - validation) * 30.0
        penalty += max(0.0, val_mdd - float(spec["max_validation_mdd"])) * 24.0
        penalty += max(0.0, train_mdd - float(spec["max_train_mdd"])) * 6.0
        penalty += max(0.0, 10.0 - train_rpt) / 5.0
        penalty += max(0.0, 10.0 - val_rpt) / 4.0
        penalty += (
            max(
                0.0, _safe_float(conc.get("top_symbol_share")) - float(spec["top_symbol_share_cap"])
            )
            * 4.0
        )
        penalty += max(0.0, _safe_float(conc.get("top_anchor_share")) - 0.45) * 2.5
        penalty += max(0.0, _safe_float(conc.get("top_asset_group_share"), 0.0) - 0.70) * 2.0
        if train <= 0.0:
            penalty += 5.0 + abs(train) * 10.0
        score = (
            9.0 * validation
            + 1.5 * min(train, validation)
            + min(max(train_rpt, -20.0), 120.0) / 200.0
            + min(max(val_rpt, -20.0), 120.0) / 120.0
            - 3.0 * val_mdd
            - 0.6 * train_mdd
            - 4.0 * spike
            - penalty
        )
        metrics["concentration"] = conc
        return float(score), metrics, returns, turnover, events

    def objective(trial: Any) -> float:
        score, metrics, _, _, _ = score_multipliers(multipliers_from_trial(trial))
        for key in (
            "train_return",
            "validation_return",
            "train_mdd",
            "validation_mdd",
            "train_return_per_turnover_proxy_bps",
            "validation_return_per_turnover_proxy_bps",
        ):
            trial.set_user_attr(key, metrics.get(key))
        trial.set_user_attr("top_symbol_share", metrics["concentration"].get("top_symbol_share"))
        return score

    equal = {
        f"m_{idx}": min(1.0, float(spec["max_gross_notional"]) / max(1.0, len(ranked)))
        for idx in range(len(ranked))
    }
    study = run_optuna_study(
        optuna_module=optuna,
        objective=objective,
        n_trials=n_trials,
        direction="maximize",
        seed=seed,
        enqueue_trials=[equal],
        n_jobs=1,
        show_progress_bar=False,
    )
    best_raw = np.array(
        [float(study.best_params.get(f"m_{idx}", 0.0)) for idx in range(len(ranked))]
    )
    best_raw = np.where(best_raw < 0.05, 0.0, best_raw)
    gross = float(np.dot(best_raw, notionals))
    if gross > float(spec["max_gross_notional"]) and gross > 0.0:
        best_raw *= float(spec["max_gross_notional"]) / gross
    if float(best_raw.sum()) <= 0.0:
        best_raw[0] = 1.0
    score, metrics, returns, turnover, events = score_multipliers(best_raw)
    for split in grid_hybrid.ilp.SPLIT_ORDER:
        turnover.setdefault(split, 0.0)
        events.setdefault(split, 0)
    selected_rows: list[dict[str, Any]] = []
    asset_gross: dict[str, float] = defaultdict(float)
    leverage_map: dict[str, int] = {}
    model_ids: list[str] = []
    liquidation_by_split = dict.fromkeys(grid_hybrid.ilp.SPLIT_ORDER, 0)
    for rank, (mult, stream) in enumerate(
        sorted(
            [(float(m), s) for m, s in zip(best_raw, ranked, strict=True) if float(m) > 1e-6],
            key=lambda item: item[0] * _safe_float(item[1].row.get("notional_fraction")),
            reverse=True,
        ),
        start=1,
    ):
        row = dict(stream.row)
        notional = mult * _safe_float(row.get("notional_fraction"))
        row.update(
            {
                "sleeve_multiplier": mult,
                "weighted_notional_fraction": notional,
                "profile_weight_rank": rank,
            }
        )
        selected_rows.append(row)
        asset_gross[str(row["symbol"])] += notional
        leverage_map[str(row["symbol"])] = max(
            int(leverage_map.get(str(row["symbol"]), 0)), int(row.get("integer_leverage") or 0)
        )
        model_ids.append(str(row["model_id"]))
    stream = grid_hybrid.ProfileStream(
        profile_id=profile_id,
        candidate_tier="per_asset_profile_optuna_rebuilt_source_profile",
        leverage_map=leverage_map,
        gross_notional_fraction=float(metrics["concentration"]["gross_notional_fraction"]),
        asset_gross_notional_fraction=dict(sorted(asset_gross.items())),
        selected_model_ids=tuple(model_ids),
        returns=returns.sort_index(),
        turnover_by_split=turnover,
        trade_events_by_split=events,
        liquidation_count_by_split=liquidation_by_split,
    )
    row = {
        "profile_id": profile_id,
        "profile_kind": "per_asset_profile_optuna_rebuilt_source_profile",
        "candidate_tier": "paper_testnet_profile_candidate_pending_forward_fill_telemetry",
        "leverage_map": leverage_map,
        "weights": {profile_id: 1.0},
        "gross_notional_fraction": stream.gross_notional_fraction,
        "strict_promotion_profile": False,
        "promotion_gate_pass": False,
        "optimizer": "optuna_tpe_per_profile_sleeve_allocation",
        "best_value": float(study.best_value),
        "profile_spec": dict(spec),
        "concentration": metrics["concentration"],
        "selected_sleeve_count": len(selected_rows),
        "report_only_gate_reasons": [],
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "locked_oos_return_report_only": 0.0,
        "locked_oos_mdd_report_only": 0.0,
        "locked_oos_trade_event_count_report_only": 0,
        "locked_oos_return_per_turnover_proxy_bps_report_only": None,
        "locked_oos_liquidation_count_report_only": 0,
        "locked_oos_account_wipeout_count_report_only": 0,
    }
    for split in broad69.SPLIT_ORDER:
        row[f"{split}_return"] = metrics[f"{split}_return"]
        row[f"{split}_mdd"] = metrics[f"{split}_mdd"]
        row[f"{split}_trade_event_count"] = metrics[f"{split}_trade_event_count"]
        row[f"{split}_return_per_turnover_proxy_bps"] = metrics[
            f"{split}_return_per_turnover_proxy_bps"
        ]
        row[f"{split}_liquidation_count"] = 0
        row[f"{split}_account_wipeout_count"] = 0
    row["train_validation_score"] = grid_hybrid._train_validation_score(row)
    row["selection_reasons"] = grid_hybrid._selection_reasons(row)
    row["paper_testnet_candidate"] = not row["selection_reasons"]
    row["ready_for_paper"] = row["paper_testnet_candidate"]
    row["profile_objective_score"] = score
    return stream, row, selected_rows


def optimize_static_profile_blend(
    profile_streams: Sequence[grid_hybrid.ProfileStream], *, n_trials: int, seed: int
) -> dict[str, Any]:
    """Optuna static blend with a hard train-dominance bias.

    v3.5/v3.6 are still reported, but this guarded blend is useful when the
    adaptive hybrid chases a validation spike.  It keeps the same three rebuilt
    source profiles and still uses Optuna rather than a grid.
    """
    if optuna is None:
        raise RuntimeError("Optuna is required for static guarded blend")
    labels = [stream.profile_id for stream in profile_streams]
    index = profile_streams[0].returns.index

    def row_for_weights(weights: Mapping[str, float]) -> dict[str, Any]:
        returns = sum(
            stream.returns.reindex(index, fill_value=0.0) * float(weights[stream.profile_id])
            for stream in profile_streams
        )
        turnover_by_split = {
            split: float(
                sum(
                    stream.turnover_by_split[split] * float(weights[stream.profile_id])
                    for stream in profile_streams
                )
            )
            for split in grid_hybrid.ilp.SPLIT_ORDER
        }
        events_by_split = {
            split: int(
                sum(
                    stream.trade_events_by_split[split]
                    for stream in profile_streams
                    if float(weights[stream.profile_id]) > 1e-6
                )
            )
            for split in grid_hybrid.ilp.SPLIT_ORDER
        }
        liquidation_by_split = {
            split: int(
                sum(
                    stream.liquidation_count_by_split[split]
                    for stream in profile_streams
                    if float(weights[stream.profile_id]) > 1e-6
                )
            )
            for split in grid_hybrid.ilp.SPLIT_ORDER
        }
        gross = float(
            sum(
                stream.gross_notional_fraction * float(weights[stream.profile_id])
                for stream in profile_streams
            )
        )
        row = grid_hybrid._metric_row_from_stream(
            profile_id="hybrid_static_train_dominance_guarded_three_profile_blend",
            profile_kind="optuna_static_guarded_profile_blend",
            candidate_tier="paper_testnet_candidate_if_train_dominance_gate_passes",
            leverage_map={},
            weights=dict(weights),
            gross_notional_fraction=gross,
            returns=returns,
            turnover_by_split=turnover_by_split,
            trade_events_by_split=events_by_split,
            liquidation_count_by_split=liquidation_by_split,
            strict_promotion_profile=False,
            promotion_gate_pass=False,
            paper_testnet_candidate=False,
        )
        reasons = grid_hybrid._selection_reasons(row)
        row.update(
            {
                "optimizer": "optuna_tpe_static_train_dominance_guarded",
                "hybrid_version": "static_guarded",
                "selection_reasons": reasons,
                "report_only_gate_reasons": [],
                "paper_testnet_candidate": not reasons,
                "ready_for_paper": not reasons,
                "ready_for_real": False,
                "real_money_execution": False,
                "real_execution_allowed": False,
                "final_weights": dict(weights),
                "average_weights_train_validation": dict(weights),
            }
        )
        return row

    def normalized_from_trial(trial: Any) -> dict[str, float]:
        raw = np.array([trial.suggest_float(f"w_{idx}", 0.0, 1.0) for idx in range(len(labels))])
        raw = np.where(raw < 0.03, 0.0, raw)
        if float(raw.sum()) <= 0.0:
            raw[:] = 1.0
        raw = raw / float(raw.sum())
        return {label: float(raw[idx]) for idx, label in enumerate(labels)}

    def objective(trial: Any) -> float:
        row = row_for_weights(normalized_from_trial(trial))
        train = _safe_float(row.get("train_return"))
        validation = _safe_float(row.get("validation_return"))
        train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
        val_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
        val_mdd = _safe_float(row.get("validation_mdd"))
        train_mdd = _safe_float(row.get("train_mdd"))
        spike = max(0.0, validation - train)
        penalty = 0.0
        penalty += 35.0 * spike
        penalty += max(0.0, 0.02 - validation) * 30.0
        penalty += max(0.0, val_mdd - 0.20) * 25.0
        penalty += max(0.0, train_mdd - 0.45) * 10.0
        penalty += max(0.0, 10.0 - train_rpt) / 4.0
        penalty += max(0.0, 10.0 - val_rpt) / 4.0
        return float(
            8.0 * validation
            + 2.0 * min(train, validation)
            + min(train_rpt, 120.0) / 160.0
            + min(val_rpt, 120.0) / 120.0
            - 3.0 * val_mdd
            - 0.75 * train_mdd
            - penalty
        )

    enqueue = []
    for idx in range(len(labels)):
        enqueue.append({f"w_{j}": 1.0 if j == idx else 0.0 for j in range(len(labels))})
    enqueue.append({f"w_{j}": 1.0 for j in range(len(labels))})
    study = run_optuna_study(
        optuna_module=optuna,
        objective=objective,
        n_trials=n_trials,
        direction="maximize",
        seed=seed,
        enqueue_trials=enqueue,
        n_jobs=1,
        show_progress_bar=False,
    )
    best_raw = np.array(
        [float(study.best_params.get(f"w_{idx}", 0.0)) for idx in range(len(labels))]
    )
    best_raw = np.where(best_raw < 0.03, 0.0, best_raw)
    if float(best_raw.sum()) <= 0.0:
        best_raw[:] = 1.0
    best_raw = best_raw / float(best_raw.sum())
    best_weights = {label: float(best_raw[idx]) for idx, label in enumerate(labels)}
    row = row_for_weights(best_weights)
    row["best_value"] = float(study.best_value)
    row["best_params"] = dict(study.best_params)
    row["top_trials"] = [
        {
            "trial_number": int(trial.number),
            "value": None if trial.value is None else float(trial.value),
            "params": dict(trial.params),
        }
        for trial in sorted(
            study.trials,
            key=lambda t: float(t.value) if t.value is not None else -1e18,
            reverse=True,
        )[:20]
    ]
    return row


def _split_windows_for_hybrid(
    plan_payload: Mapping[str, Any],
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split in ("train", "validation"):
        raw = plan_payload[split]
        out[split] = (
            pd.Timestamp(raw["start"]).tz_localize(None),
            pd.Timestamp(raw["end"]).tz_localize(None),
        )
    raw_oos = plan_payload.get("locked_oos", {})
    if raw_oos.get("start") is None or raw_oos.get("end") is None:
        validation_end = out["validation"][1]
        out["locked_oos"] = (validation_end + pd.Timedelta(hours=1), validation_end)
    else:
        out["locked_oos"] = (
            pd.Timestamp(raw_oos["start"]).tz_localize(None),
            pd.Timestamp(raw_oos["end"]).tz_localize(None),
        )
    return out


def _render_pct(value: Any) -> str:
    return f"{_safe_float(value):.4%}"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    selected = dict(payload.get("selected_optuna_hybrid_profile") or {})
    lines = [
        "# 69-asset per-profile Optuna hybrid refit",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "## Correction",
        "",
        "- This run rebuilds the three hybrid source profiles across the 69-symbol universe instead of applying one shared parameter set or only optimizing final stream weights.",
        "- Every symbol/profile pair is Optuna-tuned over family, timeframe, side, entry/exit, hold/cooldown, and integer leverage.",
        "- Domain anchors are tracked beyond BTC: BTC, ETH, SOL, SPY, QQQ, XAU, XAG, and crude proxy anchors are used to penalize single-benchmark clones and profile-level anchor concentration.",
        "- Each rebuilt source profile then gets its own Optuna sleeve-allocation pass, and the final blend reuses the existing v3.5/v3.6 Optuna hybrid engine.",
        "- No live or real-money execution is enabled; `ready_for_real=false` and `real_money_execution=false` remain invariant.",
        "",
        "## Selected hybrid",
        "",
        f"- profile: `{selected.get('profile_id')}`",
        f"- version: `{selected.get('hybrid_version')}`",
        f"- train / validation: `{_render_pct(selected.get('train_return'))}` / `{_render_pct(selected.get('validation_return'))}`",
        f"- train / validation MDD: `{_render_pct(selected.get('train_mdd'))}` / `{_render_pct(selected.get('validation_mdd'))}`",
        f"- RPT bps train / validation: `{_safe_float(selected.get('train_return_per_turnover_proxy_bps')):.2f}` / `{_safe_float(selected.get('validation_return_per_turnover_proxy_bps')):.2f}`",
        f"- gross notional: `{_safe_float(selected.get('gross_notional_fraction')):.4f}x`",
        f"- final weights: `{json.dumps(_json_safe(selected.get('final_weights') or {}), sort_keys=True)}`",
        f"- selection reasons: `{selected.get('selection_reasons')}`",
        "",
        "## Selected train/validation-legal portfolio",
        "",
        f"- selected legal profile: `{dict(payload.get('selected_train_validation_legal_portfolio') or {}).get('profile_id')}`",
        f"- train / validation: `{_render_pct(dict(payload.get('selected_train_validation_legal_portfolio') or {}).get('train_return'))}` / `{_render_pct(dict(payload.get('selected_train_validation_legal_portfolio') or {}).get('validation_return'))}`",
        f"- selection reasons: `{dict(payload.get('selected_train_validation_legal_portfolio') or {}).get('selection_reasons')}`",
        "",
        "## Rebuilt source profiles",
        "",
        "| Profile | Sleeves | Gross | Train | Validation | Val MDD | RPT T/V bps | Paper | Top symbol |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload.get("profile_rows", []):
        conc = dict(row.get("concentration") or {})
        lines.append(
            f"| `{row.get('profile_id')}` | {int(row.get('selected_sleeve_count') or 0)} | "
            f"{_safe_float(row.get('gross_notional_fraction')):.2f}x | "
            f"{_render_pct(row.get('train_return'))} | {_render_pct(row.get('validation_return'))} | "
            f"{_render_pct(row.get('validation_mdd'))} | "
            f"{_safe_float(row.get('train_return_per_turnover_proxy_bps')):.2f}/"
            f"{_safe_float(row.get('validation_return_per_turnover_proxy_bps')):.2f} | "
            f"{str(bool(row.get('ready_for_paper'))).lower()} | "
            f"`{conc.get('top_symbol')} {float(conc.get('top_symbol_share') or 0.0):.2%}` |"
        )
    lines.extend(
        [
            "",
            "## Governance",
            "",
            f"- search method: `{payload.get('optimization_policy', {}).get('search_method')}`",
            f"- asset trials/profile: `{payload.get('optimization_policy', {}).get('asset_trials_per_symbol_profile')}`",
            f"- profile allocation trials: `{payload.get('optimization_policy', {}).get('profile_allocation_trials')}`",
            f"- hybrid trials/version: `{payload.get('optimization_policy', {}).get('hybrid_trials_per_version')}`",
            f"- runner peak RSS MiB: `{_safe_float(payload.get('runner_peak_rss_mib')):.2f}`",
            f"- ready_for_real: `{str(bool(payload.get('ready_for_real'))).lower()}`",
            f"- real_money_execution: `{str(bool(payload.get('real_money_execution'))).lower()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    symbols = broad69._parse_csv(args.symbols)
    timeframes = broad69._parse_timeframes(args.timeframes)
    bars, coverage = broad69.load_all_bars(
        symbols, data_root=Path(args.data_root).expanduser().resolve(), timeframes=timeframes
    )
    windows = broad69.build_standard_split_windows(
        data_end_utc=str(coverage["global_latest_utc"]),
        validation_weeks=int(args.validation_weeks),
        bar_minutes=60,
    )
    cache = FeatureCache(
        bars_by_symbol_tf=bars,
        symbols=symbols,
        timeframes=timeframes,
        _xsmom={},
        _anchor_returns={},
    )

    asset_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    profile_streams: list[grid_hybrid.ProfileStream] = []
    for profile_idx, base_spec in enumerate(PROFILE_SPECS):
        spec = {**base_spec, "_timeframes": timeframes}
        streams: list[broad69.CandidateStream] = []
        for symbol_idx, symbol in enumerate(symbols):
            stream = tune_symbol_profile(
                symbol=symbol,
                spec=spec,
                cache=cache,
                windows=windows,
                n_trials=int(args.asset_trials),
                seed=int(args.seed) + profile_idx * 100_000 + symbol_idx,
                allocation_fraction=float(args.allocation_fraction),
            )
            if stream is None:
                continue
            asset_rows.append(dict(stream.row))
            if _safe_float(stream.row.get("profile_objective_score"), -1e18) > -1e8:
                streams.append(stream)
        profile_stream, profile_row, selected_sleeves = tune_profile_allocations(
            spec=spec,
            candidate_streams=streams,
            windows=windows,
            n_trials=int(args.profile_trials),
            seed=int(args.seed) + profile_idx * 10_000 + 500,
        )
        profile_streams.append(profile_stream)
        profile_rows.append(profile_row)
        sleeve_rows.extend(selected_sleeves)

    plan_payload = windows.as_payload()
    split_windows = _split_windows_for_hybrid(plan_payload)
    with optuna_hybrid._split_window_context(split_windows):
        v35 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_5",
            n_trials=int(args.hybrid_trials),
            seed=int(args.seed) + 700_000,
            fit_splits=("train", "validation"),
            require_locked_oos_gate=False,
        )
        v36 = optuna_hybrid._run_optuna(
            profile_streams,
            version="v3_6",
            n_trials=int(args.hybrid_trials),
            seed=int(args.seed) + 700_001,
            fit_splits=("train", "validation"),
            require_locked_oos_gate=False,
        )
        static_guarded = optimize_static_profile_blend(
            profile_streams, n_trials=int(args.hybrid_trials), seed=int(args.seed) + 710_000
        )
        selected_result = optuna_hybrid._choose_selected_optuna_result([v35, v36])
        corr_tv = grid_hybrid._profile_corr_matrix(profile_streams, split="train_validation")
        corr_val = grid_hybrid._profile_corr_matrix(profile_streams, split="validation")

    output_dir = Path(args.output_dir).expanduser().resolve()
    latest_json = output_dir / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.json"
    timestamped_json = (
        output_dir / f"alpha_zoo_69_asset_profile_optuna_hybrid_refit_{_timestamp()}.json"
    )
    latest_md = output_dir / "alpha_zoo_69_asset_profile_optuna_hybrid_refit_latest.md"
    assets_csv = output_dir / "alpha_zoo_69_asset_profile_tuned_assets_latest.csv"
    sleeves_csv = output_dir / "alpha_zoo_69_asset_profile_selected_sleeves_latest.csv"
    profiles_csv = output_dir / "alpha_zoo_69_asset_profile_rows_latest.csv"

    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    selected = dict(selected_result.row)
    legal_pool = [dict(static_guarded), *profile_rows, dict(v35.row), dict(v36.row)]
    legal_pass = [row for row in legal_pool if not row.get("selection_reasons")]
    selected_legal = max(
        legal_pass or legal_pool,
        key=lambda row: grid_hybrid._train_validation_score(row),
    )
    selected["ready_for_real"] = False
    selected["real_money_execution"] = False
    selected["real_execution_allowed"] = False
    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_69_asset_profile_optuna_hybrid_refit",
        "generated_at_utc": _utc_now_iso(),
        "universe": {"symbol_count": len(symbols), "symbols": list(symbols)},
        "timeframes": list(timeframes),
        "split_policy": plan_payload,
        "data_coverage": coverage,
        "research_primary_round_trip_cost_bps": broad69.PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": broad69.AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "return_per_turnover_threshold_bps": broad69.RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "paper_testnet_only": True,
        "ready_for_paper": bool(selected_legal.get("ready_for_paper")),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "optimization_policy": optimization_search_policy_payload(
            search_method="optuna_tpe_per_asset_profile_then_v35_v36_hybrid",
            objective_policy="train_validation_profile_rebuild_no_locked_oos_live_refit",
            selection_inputs=("train", "validation"),
            extra={
                "asset_trials_per_symbol_profile": int(args.asset_trials),
                "profile_allocation_trials": int(args.profile_trials),
                "hybrid_trials_per_version": int(args.hybrid_trials),
                "symbol_profile_tune_count": len(asset_rows),
                "profile_specs": list(PROFILE_SPECS),
                "domain_anchors": dict(DOMAIN_ANCHORS),
                "domain_filtering": "candidate single-anchor clone penalty plus profile top-anchor/top-symbol/top-group concentration penalty",
                "all_internal_params_tuned_per_asset_profile": True,
                "final_hybrid_engine": "run_alpha_zoo_integer_leverage_optuna_hybrid_decision::_run_optuna",
                "uses_test_set": False,
            },
        ),
        "profile_rows": profile_rows,
        "asset_tuning_rows": asset_rows,
        "selected_sleeve_rows": sleeve_rows,
        "selected_optuna_hybrid_profile": selected,
        "static_train_dominance_guarded_hybrid": static_guarded,
        "selected_train_validation_legal_portfolio": selected_legal,
        "hybrid_v3_5_optuna": {
            "row": v35.row,
            "optuna": v35.optuna,
            "top_trials": list(v35.top_trials),
        },
        "hybrid_v3_6_optuna": {
            "row": v36.row,
            "optuna": v36.optuna,
            "top_trials": list(v36.top_trials),
        },
        "profile_train_validation_corr_matrix": corr_tv,
        "profile_validation_corr_matrix": corr_val,
        "runner_peak_rss_mib": peak_mib,
        "memory_summary": {"limit_mib": 8192.0, "pass_under_8gb": peak_mib < 8192.0},
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_md": str(latest_md),
            "asset_tuning_csv": str(assets_csv),
            "selected_sleeves_csv": str(sleeves_csv),
            "profile_rows_csv": str(profiles_csv),
        },
    }

    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.parent.mkdir(parents=True, exist_ok=True)
    latest_md.write_text(_render_markdown(payload), encoding="utf-8")
    _write_csv(assets_csv, asset_rows, CANDIDATE_FIELDS)
    _write_csv(sleeves_csv, sleeve_rows, SLEEVE_FIELDS)
    _write_csv(
        profiles_csv,
        profile_rows,
        list(dict.fromkeys([*SLEEVE_FIELDS, "gross_notional_fraction", "selected_sleeve_count"])),
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(BINANCE_EXTENDED_RESEARCH_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--allocation-fraction", type=float, default=DEFAULT_ALLOCATION_FRACTION)
    parser.add_argument("--validation-weeks", type=int, default=broad69.STANDARD_VALIDATION_WEEKS)
    parser.add_argument("--asset-trials", type=int, default=DEFAULT_ASSET_TRIALS)
    parser.add_argument("--profile-trials", type=int, default=DEFAULT_PROFILE_TRIALS)
    parser.add_argument("--hybrid-trials", type=int, default=DEFAULT_HYBRID_TRIALS)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    selected = dict(payload["selected_optuna_hybrid_profile"])
    selected_legal = dict(payload.get("selected_train_validation_legal_portfolio") or {})
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "selected_optuna_hybrid_profile": {
                        key: selected.get(key)
                        for key in (
                            "profile_id",
                            "hybrid_version",
                            "train_return",
                            "validation_return",
                            "train_mdd",
                            "validation_mdd",
                            "train_return_per_turnover_proxy_bps",
                            "validation_return_per_turnover_proxy_bps",
                            "gross_notional_fraction",
                            "final_weights",
                            "selection_reasons",
                        )
                    },
                    "selected_train_validation_legal_portfolio": {
                        key: selected_legal.get(key)
                        for key in (
                            "profile_id",
                            "hybrid_version",
                            "train_return",
                            "validation_return",
                            "train_mdd",
                            "validation_mdd",
                            "train_return_per_turnover_proxy_bps",
                            "validation_return_per_turnover_proxy_bps",
                            "gross_notional_fraction",
                            "final_weights",
                            "selection_reasons",
                            "ready_for_paper",
                        )
                    },
                    "ready_for_paper": payload["ready_for_paper"],
                    "ready_for_real": payload["ready_for_real"],
                    "real_money_execution": payload["real_money_execution"],
                    "runner_peak_rss_mib": payload["runner_peak_rss_mib"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
