#!/usr/bin/env python3
"""Pre-registered clean new-alpha discovery for Alpha Zoo.

This runner searches only newly declared OHLCV alpha families and freezes a
single candidate from train+validation before attaching locked-OOS report/gate
metrics. It is intentionally separate from post-OOS selector research: no
post-OOS meta-selector or lagged-shadow output is used as material, objective,
threshold, tie-break, or enqueue input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.optimization.search_policy import optimization_search_policy_payload  # noqa: E402
from lumina_quant.research_universe import BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_monthly_refit_walkforward as monthly  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research.run_alpha_zoo_htf_momentum_crowding_discovery import DEFAULT_FEATURE_ROOT  # noqa: E402

DEFAULT_OUTPUT_DIR = broad69.ALPHA_V2_ROOT / "alpha_zoo_clean_new_alpha_discovery_20260607"
DEFAULT_TIMEFRAMES = ("1h", "4h")
DEFAULT_LEVERAGES = (2, 3, 4)
DEFAULT_ALLOCATION_FRACTION = 0.10

FAMILY_DESCRIPTIONS: dict[str, str] = {
    "volatility_squeeze_breakout": (
        "Enter with Donchian breakout only after Bollinger-width compression; "
        "exit on mid-band loss or opposite breakout. This is a pre-registered "
        "compression/expansion alpha, not a post-OOS router."
    ),
    "volume_absorption_reversal": (
        "Enter reversal after abnormal volume and wick/close absorption; exit "
        "on z-score normalization. Uses OHLCV only and no calendar features."
    ),
    "range_reclaim_continuation": (
        "Enter when price reclaims a prior range after false breakdown/breakout; "
        "exit on range midline failure. Uses only rolling range state."
    ),
    "cross_asset_lead_lag_momentum": (
        "Enter target symbols from lagged leader-symbol momentum when the target "
        "has not yet fully moved with the leader. This is a pre-registered "
        "cross-asset information-flow alpha using only prior bars."
    ),
    "feature_flow_crowding_reversal": (
        "Use local funding/open-interest/taker-flow feature points to fade crowded "
        "directional flow after funding/flow extremes. Feature coverage must exist "
        "in train and validation; locked-OOS features are report-only."
    ),
    "feature_liquidation_imbalance_reversal": (
        "Use local liquidation imbalance with funding/open-interest context to fade "
        "one-sided squeeze cascades after crowding extremes. Feature coverage must "
        "exist in train and validation; locked-OOS features are report-only."
    ),
    "feature_flow_oi_trend_continuation": (
        "Use Binance taker-flow plus open-interest expansion with neutral funding "
        "to continue internal Binance orderflow leadership without cross-venue or "
        "post-OOS selector inputs."
    ),
    "funding_oi_taker_crowding_continuation": (
        "Use Binance taker-flow and open-interest expansion with neutral funding "
        "to continue internal Binance perp crowding when feature-backed state is "
        "present. No post-OOS selector inputs."
    ),
    "perp_crowding_score_reversion": (
        "Compute a Binance-only perp crowding score from funding, open-interest, "
        "and liquidation imbalance z-scores, then fade one-sided crowding after "
        "extreme buildup. No post-OOS selector inputs."
    ),
    "feature_taker_flow_exhaustion_reversal": (
        "Fade short-horizon Binance taker-flow and price-extension exhaustion when "
        "funding remains contained. Uses only Binance feature points and prior bars."
    ),
    "feature_bbo_flow_exhaustion_reversal": (
        "Use Binance BBO spread expansion together with taker-flow exhaustion to fade "
        "microstructure dislocations. Requires historical BBO feature coverage."
    ),
    "feature_book_depth_imbalance_reversal": (
        "Use official Binance public-data bookDepth nearest-bucket notional imbalance "
        "as a post-2024 orderbook pressure source. This is a separate depth feature, "
        "not a synthetic BBO quote."
    ),
    "deep_research_funding_dislocation_trend_carry": (
        "Report-inspired leaf alpha: align medium-horizon momentum with perp funding "
        "carry and open-interest crowding caps. Requires train/validation feature "
        "coverage and remains blocked from promotion until fresh-forward/cost gates."
    ),
    "deep_research_vol_managed_momentum_crash_gate": (
        "Report-inspired leaf alpha: momentum entries only when realized volatility and "
        "BTC benchmark stress gates are acceptable. This is a leaf search input, not "
        "a post-OOS router or live promotion rule."
    ),
    "deep_research_flow_imbalance_liquidation_sweep": (
        "Report-inspired leaf alpha: contrarian sweep entries after return shocks, "
        "liquidation imbalance, taker-flow/depth confirmation, and spread-quality "
        "filters. Requires train/validation feature coverage."
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _timestamp(value: Any) -> pd.Timestamp:
    return (
        pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)
    )


def _search_space() -> dict[str, Any]:
    return {
        "families": FAMILY_DESCRIPTIONS,
        "timeframes": list(DEFAULT_TIMEFRAMES),
        "integer_leverages": list(DEFAULT_LEVERAGES),
        "allocation_fraction": DEFAULT_ALLOCATION_FRACTION,
        "leader_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "volatility_squeeze_breakout": {
            "lookback": [24, 48, 72],
            "compression_quantile": [0.15, 0.25],
            "breakout_window": [12, 24],
            "min_hold": [4, 8],
        },
        "volume_absorption_reversal": {
            "lookback": [24, 48],
            "volume_z": [1.5, 2.0],
            "wick_ratio": [0.45, 0.60],
            "min_hold": [4, 8],
        },
        "range_reclaim_continuation": {
            "lookback": [24, 48, 96],
            "false_break_z": [0.25, 0.50],
            "min_hold": [4, 8],
        },
        "cross_asset_lead_lag_momentum": {
            "lookback": [6, 12, 24],
            "leader_threshold": [0.01, 0.02],
            "target_lag_fraction": [0.25, 0.50],
            "min_hold": [4, 8],
        },
        "feature_flow_crowding_reversal": {
            "lookback": [12, 24],
            "funding_abs_min": [0.00005, 0.00010],
            "flow_abs_min": [0.15, 0.30],
            "oi_z_min": [0.5, 1.0],
            "min_hold": [4, 8],
        },
        "feature_liquidation_imbalance_reversal": {
            "lookback": [12, 24],
            "liq_z_min": [1.0, 1.5],
            "funding_abs_min": [0.00005, 0.00010],
            "oi_z_min": [0.5, 1.0],
            "min_hold": [4, 8],
        },
        "feature_flow_oi_trend_continuation": {
            "lookback": [12, 24],
            "flow_abs_min": [0.15, 0.30],
            "oi_z_min": [0.5, 1.0],
            "funding_abs_max": [0.00010, 0.00020],
            "min_hold": [4, 8],
        },
        "funding_oi_taker_crowding_continuation": {
            "lookback": [6, 12],
            "funding_abs_max": [0.00010, 0.00025],
            "imbalance_threshold": [0.0, 0.05],
            "min_hold": [8],
        },
        "perp_crowding_score_reversion": {
            "lookback": [24, 48],
            "crowding_threshold": [0.55, 0.75],
            "min_hold": [4, 8],
        },
        "feature_taker_flow_exhaustion_reversal": {
            "lookback": [6, 12],
            "flow_imbalance_min": [0.10, 0.15],
            "price_extension_min": [0.004, 0.008],
            "funding_abs_cap": [0.00015, 0.00030],
            "max_realized_vol": [0.008, 0.012],
            "min_hold": [4, 8],
        },
        "feature_bbo_flow_exhaustion_reversal": {
            "lookback": [6, 12],
            "spread_z_min": [1.0, 1.5],
            "flow_imbalance_min": [0.10, 0.15],
            "price_extension_min": [0.004, 0.008],
            "min_hold": [4, 8],
        },
        "feature_book_depth_imbalance_reversal": {
            "lookback": [6, 12],
            "depth_imbalance_min": [0.10, 0.20],
            "price_extension_min": [0.003, 0.006],
            "min_hold": [4, 8],
        },
        "deep_research_funding_dislocation_trend_carry": {
            "lookback": [12, 24, 48],
            "momentum_min": [0.004, 0.008],
            "funding_carry_min": [0.0, 0.00005],
            "oi_z_cap": [1.5, 2.5],
            "min_hold": [4, 8],
        },
        "deep_research_vol_managed_momentum_crash_gate": {
            "lookback": [12, 24, 48],
            "momentum_min": [0.006, 0.012],
            "realized_vol_max": [0.018, 0.028],
            "benchmark_crash_window": [12, 24],
            "benchmark_crash_return": [0.035, 0.055],
            "min_hold": [4, 8],
        },
        "deep_research_flow_imbalance_liquidation_sweep": {
            "lookback": [12, 24],
            "return_shock_min": [0.004, 0.008],
            "flow_or_depth_min": [0.10, 0.20],
            "liquidation_z_min": [1.0, 1.5],
            "max_spread_bps": [8.0, 12.0],
            "min_hold": [4, 8],
        },
    }


def _search_space_hash(space: Mapping[str, Any]) -> str:
    encoded = json.dumps(space, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compound(returns: Sequence[float]) -> float:
    equity = 1.0
    for value in returns:
        equity *= 1.0 + float(value)
    return float(equity - 1.0)


def _equity_mdd(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for value in returns:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak > 0.0:
            mdd = max(mdd, 1.0 - equity / peak)
    return float(mdd)


def _sharpe(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    std = float(np.std(arr, ddof=1))
    return float(np.mean(arr) / std * math.sqrt(12.0)) if std > 0.0 else 0.0


def _profit_factor(returns: Sequence[float]) -> tuple[float, bool]:
    gains = float(sum(value for value in returns if value > 0.0))
    losses = float(-sum(value for value in returns if value < 0.0))
    if losses <= 0.0:
        return 0.0, True
    return gains / losses, False


def _score_row(row: Mapping[str, Any]) -> float:
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_mdd = _safe_float(row.get("train_mdd"))
    validation_mdd = _safe_float(row.get("validation_mdd"))
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    validation_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    spike = max(0.0, validation - max(train, 0.0))
    active_penalty = 0.0
    if int(row.get("train_trade_event_count") or 0) < 40:
        active_penalty += 2.0
    if int(row.get("validation_trade_event_count") or 0) < 12:
        active_penalty += 2.0
    return float(
        7.0 * validation
        + 2.0 * min(train, validation)
        + min(max(train_rpt, -20.0), 160.0) / 160.0
        + min(max(validation_rpt, -20.0), 160.0) / 120.0
        - 3.5 * validation_mdd
        - 1.0 * train_mdd
        - 5.0 * spike
        - active_penalty
    )


def _eligible_for_freeze(row: Mapping[str, Any]) -> bool:
    if bool(row.get("feature_backed")):
        coverage = row.get("feature_coverage") or {}
        if (
            _safe_float(coverage.get("train")) < 0.60
            or _safe_float(coverage.get("validation")) < 0.60
        ):
            return False
    return (
        _safe_float(row.get("train_return")) > 0.0
        and _safe_float(row.get("validation_return")) > 0.0
        and _safe_float(row.get("validation_mdd")) <= 0.25
        and int(row.get("train_trade_event_count") or 0) >= 20
        and int(row.get("validation_trade_event_count") or 0) >= 6
        and _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0) > 10.0
    )


def _window_mask(
    datetimes: pd.Series | pd.DatetimeIndex, window: tuple[pd.Timestamp, pd.Timestamp]
) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(datetimes))
    return np.asarray((idx >= window[0]) & (idx <= window[1]), dtype=bool)


def _split_feature_coverage(
    frame: pd.DataFrame,
    *,
    train: tuple[pd.Timestamp, pd.Timestamp],
    validation: tuple[pd.Timestamp, pd.Timestamp],
    locked_oos: tuple[pd.Timestamp, pd.Timestamp],
    column: str = "feature_valid",
) -> dict[str, float]:
    if column not in frame.columns:
        return {"train": 0.0, "validation": 0.0, "locked_oos": 0.0}
    valid = frame[column].fillna(False).astype(bool).to_numpy()
    return {
        "train": float(np.mean(valid[_window_mask(frame["datetime"], train)])),
        "validation": float(np.mean(valid[_window_mask(frame["datetime"], validation)])),
        "locked_oos": float(np.mean(valid[_window_mask(frame["datetime"], locked_oos)])),
    }


def _attach_feature_points(
    bars: pd.DataFrame, features: pd.DataFrame, *, timeframe: str
) -> pd.DataFrame:
    out = bars.copy().sort_values("datetime")
    if features.empty:
        out["funding_rate"] = np.nan
        out["open_interest"] = np.nan
        out["taker_buy_sell_imbalance"] = np.nan
        out["liquidation_imbalance"] = np.nan
        out["bbo_spread_bps"] = np.nan
        out["book_depth_imbalance_1pct"] = np.nan
        out["funding_rate_age_hours"] = np.inf
        out["open_interest_age_hours"] = np.inf
        out["taker_buy_sell_imbalance_age_hours"] = np.inf
        out["liquidation_imbalance_age_hours"] = np.inf
        out["bbo_spread_bps_age_hours"] = np.inf
        out["book_depth_imbalance_1pct_age_hours"] = np.inf
        out["feature_depth_valid"] = False
        out["feature_age_hours"] = np.inf
        out["feature_valid"] = False
        out["feature_oi_flow_valid"] = False
        out["feature_liquidation_valid"] = False
        out["feature_bbo_valid"] = False
        return out
    feats = features.copy().sort_values("datetime")

    def numeric_feature(column: str) -> pd.Series:
        if column not in feats.columns:
            return pd.Series(np.nan, index=feats.index)
        return pd.to_numeric(feats[column], errors="coerce")

    buy = numeric_feature("taker_buy_quote_volume")
    sell = numeric_feature("taker_sell_quote_volume")
    denom = buy.fillna(0.0) + sell.fillna(0.0)
    feats["taker_buy_sell_imbalance"] = np.where(
        denom > 0.0, (buy.fillna(0.0) - sell.fillna(0.0)) / denom, np.nan
    )
    feats["funding_rate"] = numeric_feature("funding_rate")
    feats["open_interest"] = numeric_feature("open_interest")
    long_liq = numeric_feature("liquidation_long_notional")
    short_liq = numeric_feature("liquidation_short_notional")
    liq_denom = long_liq.abs().fillna(0.0) + short_liq.abs().fillna(0.0)
    feats["liquidation_imbalance"] = np.where(
        liq_denom > 0.0, (long_liq.fillna(0.0) - short_liq.fillna(0.0)) / liq_denom, np.nan
    )
    feats["bbo_spread_bps"] = numeric_feature("bbo_spread_bps")
    feats["book_depth_imbalance_1pct"] = numeric_feature("book_depth_imbalance_1pct")
    feature_columns = [
        "funding_rate",
        "open_interest",
        "taker_buy_sell_imbalance",
        "liquidation_imbalance",
        "bbo_spread_bps",
        "book_depth_imbalance_1pct",
    ]
    for column in feature_columns:
        observed_column = f"{column}_observed_datetime"
        observed = pd.Series(pd.NaT, index=feats.index, dtype="datetime64[ns]")
        observed.loc[feats[column].notna()] = feats.loc[feats[column].notna(), "datetime"]
        feats[observed_column] = observed.ffill()
        feats[column] = feats[column].ffill()
    feats = feats.rename(columns={"datetime": "feature_datetime"})
    merged = pd.merge_asof(
        out,
        feats[
            [
                "feature_datetime",
                "funding_rate",
                "funding_rate_observed_datetime",
                "open_interest",
                "open_interest_observed_datetime",
                "taker_buy_sell_imbalance",
                "taker_buy_sell_imbalance_observed_datetime",
                "liquidation_imbalance",
                "liquidation_imbalance_observed_datetime",
                "bbo_spread_bps",
                "bbo_spread_bps_observed_datetime",
                "book_depth_imbalance_1pct",
                "book_depth_imbalance_1pct_observed_datetime",
            ]
        ],
        left_on="datetime",
        right_on="feature_datetime",
        direction="backward",
    )
    age = (merged["datetime"] - merged["feature_datetime"]).dt.total_seconds() / 3600.0
    max_age = 24.0 if timeframe == "1h" else 48.0
    merged["feature_age_hours"] = age.fillna(np.inf)
    for column in feature_columns:
        observed_column = f"{column}_observed_datetime"
        age_column = f"{column}_age_hours"
        observed_age = (
            merged["datetime"] - pd.to_datetime(merged[observed_column])
        ).dt.total_seconds() / 3600.0
        merged[age_column] = observed_age.fillna(np.inf)
    flow_valid = (
        (merged["funding_rate_age_hours"] <= max_age)
        & (merged["taker_buy_sell_imbalance_age_hours"] <= max_age)
        & merged["funding_rate"].notna()
        & merged["taker_buy_sell_imbalance"].notna()
    )
    oi_flow_valid = (
        flow_valid
        & (merged["open_interest_age_hours"] <= max_age)
        & merged["open_interest"].notna()
    )
    merged["feature_valid"] = flow_valid
    merged["feature_oi_flow_valid"] = oi_flow_valid
    merged["feature_liquidation_valid"] = (
        oi_flow_valid
        & (merged["liquidation_imbalance_age_hours"] <= max_age)
        & merged["liquidation_imbalance"].notna()
    )
    merged["feature_bbo_valid"] = (
        flow_valid
        & (merged["bbo_spread_bps_age_hours"] <= max_age)
        & merged["bbo_spread_bps"].notna()
    )
    merged["feature_depth_valid"] = (
        merged["book_depth_imbalance_1pct_age_hours"] <= max_age
    ) & merged["book_depth_imbalance_1pct"].notna()
    return merged.drop(
        columns=[
            "feature_datetime",
            "funding_rate_observed_datetime",
            "open_interest_observed_datetime",
            "taker_buy_sell_imbalance_observed_datetime",
            "liquidation_imbalance_observed_datetime",
            "bbo_spread_bps_observed_datetime",
            "book_depth_imbalance_1pct_observed_datetime",
        ]
    )


def _load_feature_points_safe(symbol: str, *, feature_root: Path) -> pd.DataFrame:
    symbol_root = feature_root / f"symbol={symbol}"
    files = sorted(
        path for path in symbol_root.rglob("*.parquet") if not path.name.endswith(".tmp.parquet")
    )
    if not files:
        return pd.DataFrame(
            columns=[
                "datetime",
                "funding_rate",
                "open_interest",
                "taker_buy_quote_volume",
                "taker_sell_quote_volume",
                "liquidation_long_notional",
                "liquidation_short_notional",
                "bbo_spread_bps",
                "book_depth_imbalance_1pct",
            ]
        )
    lf = pl.scan_parquet([str(path) for path in files])
    schema = lf.collect_schema()
    wanted = [
        "timestamp_ms",
        "funding_rate",
        "open_interest",
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
        "liquidation_long_notional",
        "liquidation_short_notional",
        "bbo_spread_bps",
        "book_depth_imbalance_1pct",
    ]
    existing = [column for column in wanted if column in schema]
    frame = lf.select(existing).collect()
    pdf = pd.DataFrame(frame.to_dicts())
    if pdf.empty:
        return pd.DataFrame(columns=["datetime", *wanted[1:]])
    for column in wanted:
        if column not in pdf.columns:
            pdf[column] = np.nan
    pdf["datetime"] = pd.to_datetime(pdf["timestamp_ms"], unit="ms")
    return pdf.sort_values("datetime")


def _finalize_row(
    *,
    base: Mapping[str, Any],
    sim: broad69.SimResult,
    datetimes: pd.Series,
    timeframe: str,
    train: tuple[pd.Timestamp, pd.Timestamp],
    validation: tuple[pd.Timestamp, pd.Timestamp],
    locked_oos: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, Any]:
    row = dict(base)
    windows = broad69.SplitWindows(train=train, validation=validation)
    row = broad69.finalize_candidate(row, sim, datetimes, timeframe=timeframe, windows=windows)
    mask = _window_mask(pd.DatetimeIndex(pd.to_datetime(datetimes)), locked_oos)
    locked = broad69.split_metrics(
        sim.returns[mask],
        sim.position[mask],
        sim.liquidation_flags[mask],
        sim.account_wipeout_flags[mask],
        timeframe=timeframe,
    )
    row.update(
        {
            "locked_oos_return_report_only": locked["total_return"],
            "locked_oos_mdd_report_only": locked["max_drawdown"],
            "locked_oos_sharpe_report_only": locked["sharpe"],
            "locked_oos_sortino_report_only": locked["sortino"],
            "locked_oos_trade_event_count_report_only": locked["trade_event_count"],
            "locked_oos_liquidation_count_report_only": locked["liquidation_count"],
            "locked_oos_account_wipeout_count_report_only": locked["account_wipeout_count"],
            "selection_score_train_validation_only": _score_row(row),
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "post_oos_selector_trusted": False,
            "nested_hybrid_dependency": False,
            "split_simulation_policy": "continuous_full_period_signal_slice_report_only",
            "uses_continuous_position_state_across_split_boundaries": True,
            "clean_promotion_eligible": False,
            "label_blockers": [
                "continuous_position_state_across_split_boundaries",
                "fresh_forward_required_before_promotion",
            ],
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
        }
    )
    return row


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
    leverage: int,
    allocation_fraction: float,
) -> dict[str, Any]:
    return broad69._candidate_base(
        family=family,
        model_parts=model_parts,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        lookback=lookback,
        threshold=threshold,
        exit_threshold=exit_threshold,
        min_hold=min_hold,
        cooldown=2,
        integer_leverage=leverage,
        allocation_fraction=allocation_fraction,
    )


def _squeeze_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    close = frame["close"].astype(float).reset_index(drop=True)
    high = frame["high"].astype(float).reset_index(drop=True)
    low = frame["low"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    for lookback in (24, 48, 72):
        mid = close.rolling(lookback).mean()
        vol = close.rolling(lookback).std(ddof=1)
        width = (4.0 * vol / mid.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        for compression_quantile in (0.15, 0.25):
            compression = width <= width.rolling(lookback * 4).quantile(compression_quantile)
            for breakout_window in (12, 24):
                upper = high.rolling(breakout_window).max().shift(1)
                lower = low.rolling(breakout_window).min().shift(1)
                long_entry = compression & (close > upper)
                short_entry = compression & (close < lower)
                long_exit = close < mid
                short_exit = close > mid
                for min_hold in (4, 8):
                    signal = broad69._debounced_state_signal(
                        long_entry,
                        long_exit,
                        short_entry,
                        short_exit,
                        side="long_short",
                        min_hold_bars=min_hold,
                        cooldown_bars=2,
                    )
                    for leverage in leverages:
                        sim = broad69.simulate_symbol(
                            frame,
                            signal,
                            integer_leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                        )
                        base = _candidate_base(
                            family="volatility_squeeze_breakout",
                            model_parts=(
                                "squeeze",
                                timeframe,
                                symbol,
                                f"lb{lookback}",
                                f"q{compression_quantile}",
                                f"br{breakout_window}",
                                f"hold{min_hold}",
                                f"lev{leverage}",
                            ),
                            symbol=symbol,
                            timeframe=timeframe,
                            side="long_short",
                            lookback=lookback,
                            threshold=compression_quantile,
                            exit_threshold=0.0,
                            min_hold=min_hold,
                            leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                        )
                        out.append(
                            _finalize_row(
                                base=base,
                                sim=sim,
                                datetimes=datetimes,
                                timeframe=timeframe,
                                train=fold.train,
                                validation=fold.validation,
                                locked_oos=fold.locked_oos,
                            )
                        )
    return out


def _absorption_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    close = frame["close"].astype(float).reset_index(drop=True)
    open_ = frame["open"].astype(float).reset_index(drop=True)
    high = frame["high"].astype(float).reset_index(drop=True)
    low = frame["low"].astype(float).reset_index(drop=True)
    volume = frame["volume"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    body = (close - open_).abs()
    full_range = (high - low).replace(0.0, np.nan)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    ret_z = broad69._rolling_zscore(close.pct_change(fill_method=None), 24)
    for lookback in (24, 48):
        vol_z = broad69._rolling_zscore(volume, lookback)
        for volume_z in (1.5, 2.0):
            high_volume = vol_z >= volume_z
            for wick_ratio in (0.45, 0.60):
                lower_absorb = (lower_wick / full_range >= wick_ratio) & (body / full_range <= 0.55)
                upper_absorb = (upper_wick / full_range >= wick_ratio) & (body / full_range <= 0.55)
                long_entry = high_volume & lower_absorb & (ret_z.shift(1) < -0.5) & (close > open_)
                short_entry = high_volume & upper_absorb & (ret_z.shift(1) > 0.5) & (close < open_)
                long_exit = ret_z > 0.25
                short_exit = ret_z < -0.25
                for min_hold in (4, 8):
                    signal = broad69._debounced_state_signal(
                        long_entry,
                        long_exit,
                        short_entry,
                        short_exit,
                        side="long_short",
                        min_hold_bars=min_hold,
                        cooldown_bars=2,
                    )
                    for leverage in leverages:
                        sim = broad69.simulate_symbol(
                            frame,
                            signal,
                            integer_leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                        )
                        base = _candidate_base(
                            family="volume_absorption_reversal",
                            model_parts=(
                                "absorb",
                                timeframe,
                                symbol,
                                f"lb{lookback}",
                                f"vz{volume_z}",
                                f"wick{wick_ratio}",
                                f"hold{min_hold}",
                                f"lev{leverage}",
                            ),
                            symbol=symbol,
                            timeframe=timeframe,
                            side="long_short",
                            lookback=lookback,
                            threshold=volume_z,
                            exit_threshold=wick_ratio,
                            min_hold=min_hold,
                            leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                        )
                        out.append(
                            _finalize_row(
                                base=base,
                                sim=sim,
                                datetimes=datetimes,
                                timeframe=timeframe,
                                train=fold.train,
                                validation=fold.validation,
                                locked_oos=fold.locked_oos,
                            )
                        )
    return out


def _reclaim_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    close = frame["close"].astype(float).reset_index(drop=True)
    high = frame["high"].astype(float).reset_index(drop=True)
    low = frame["low"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    for lookback in (24, 48, 96):
        range_high = high.rolling(lookback).max().shift(1)
        range_low = low.rolling(lookback).min().shift(1)
        range_mid = (range_high + range_low) / 2.0
        range_width = (range_high - range_low).replace(0.0, np.nan)
        prior_close = close.shift(1)
        for false_break_z in (0.25, 0.50):
            long_entry = (prior_close < range_low - range_width * false_break_z) & (
                close > range_low
            )
            short_entry = (prior_close > range_high + range_width * false_break_z) & (
                close < range_high
            )
            long_exit = close < range_mid
            short_exit = close > range_mid
            for min_hold in (4, 8):
                signal = broad69._debounced_state_signal(
                    long_entry,
                    long_exit,
                    short_entry,
                    short_exit,
                    side="long_short",
                    min_hold_bars=min_hold,
                    cooldown_bars=2,
                )
                for leverage in leverages:
                    sim = broad69.simulate_symbol(
                        frame,
                        signal,
                        integer_leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                    )
                    base = _candidate_base(
                        family="range_reclaim_continuation",
                        model_parts=(
                            "reclaim",
                            timeframe,
                            symbol,
                            f"lb{lookback}",
                            f"fb{false_break_z}",
                            f"hold{min_hold}",
                            f"lev{leverage}",
                        ),
                        symbol=symbol,
                        timeframe=timeframe,
                        side="long_short",
                        lookback=lookback,
                        threshold=false_break_z,
                        exit_threshold=0.0,
                        min_hold=min_hold,
                        leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                    )
                    out.append(
                        _finalize_row(
                            base=base,
                            sim=sim,
                            datetimes=datetimes,
                            timeframe=timeframe,
                            train=fold.train,
                            validation=fold.validation,
                            locked_oos=fold.locked_oos,
                        )
                    )
    return out


def _lead_lag_rows(
    *,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    panel: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    frame = bars_by_symbol.get(symbol, pd.DataFrame())
    if frame.empty or symbol not in panel:
        return out
    datetimes = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
    target_close = panel[symbol].reindex(datetimes).ffill()
    leader_symbols = [
        item for item in ("BTCUSDT", "ETHUSDT", "SOLUSDT") if item in panel and item != symbol
    ]
    if not leader_symbols:
        return out
    target_returns = target_close.pct_change(fill_method=None)
    for leader in leader_symbols:
        leader_close = panel[leader].reindex(datetimes).ffill()
        for lookback in (6, 12, 24):
            leader_momentum = leader_close / leader_close.shift(lookback) - 1.0
            target_momentum = target_close / target_close.shift(lookback) - 1.0
            target_realized = (
                target_returns.rolling(max(6, lookback)).std(ddof=1).replace(0.0, np.nan)
            )
            residual = (target_momentum - leader_momentum) / (
                target_realized * math.sqrt(float(lookback))
            ).replace(0.0, np.nan)
            for leader_threshold in (0.01, 0.02):
                for target_lag_fraction in (0.25, 0.50):
                    long_entry = (
                        (leader_momentum.shift(1) > leader_threshold)
                        & (
                            target_momentum.shift(1)
                            < leader_momentum.shift(1) * target_lag_fraction
                        )
                        & (residual.shift(1) < 0.5)
                    )
                    short_entry = (
                        (leader_momentum.shift(1) < -leader_threshold)
                        & (
                            target_momentum.shift(1)
                            > leader_momentum.shift(1) * target_lag_fraction
                        )
                        & (residual.shift(1) > -0.5)
                    )
                    long_exit = (leader_momentum < 0.0) | (residual > 1.0)
                    short_exit = (leader_momentum > 0.0) | (residual < -1.0)
                    for min_hold in (4, 8):
                        signal = broad69._debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage in leverages:
                            sim = broad69.simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base = _candidate_base(
                                family="cross_asset_lead_lag_momentum",
                                model_parts=(
                                    "leadlag",
                                    timeframe,
                                    leader,
                                    symbol,
                                    f"lb{lookback}",
                                    f"thr{leader_threshold}",
                                    f"lag{target_lag_fraction}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=leader_threshold,
                                exit_threshold=target_lag_fraction,
                                min_hold=min_hold,
                                leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base["leader_symbol"] = leader
                            out.append(
                                _finalize_row(
                                    base=base,
                                    sim=sim,
                                    datetimes=frame["datetime"],
                                    timeframe=timeframe,
                                    train=fold.train,
                                    validation=fold.validation,
                                    locked_oos=fold.locked_oos,
                                )
                            )
    return out


def _feature_flow_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty or "feature_oi_flow_valid" not in frame.columns:
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_oi_flow_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    funding = pd.to_numeric(frame["funding_rate"], errors="coerce").reset_index(drop=True)
    open_interest = pd.to_numeric(frame["open_interest"], errors="coerce").reset_index(drop=True)
    flow = pd.to_numeric(frame["taker_buy_sell_imbalance"], errors="coerce").reset_index(drop=True)
    feature_valid = frame["feature_oi_flow_valid"].fillna(False).astype(bool).reset_index(drop=True)
    datetimes = frame["datetime"]
    returns = close.pct_change(fill_method=None)
    for lookback in (12, 24):
        oi_z = broad69._rolling_zscore(open_interest.pct_change(fill_method=None), lookback)
        ret_z = broad69._rolling_zscore(returns, lookback)
        for funding_abs_min in (0.00005, 0.00010):
            for flow_abs_min in (0.15, 0.30):
                for oi_z_min in (0.5, 1.0):
                    crowded_long = (
                        feature_valid
                        & (funding > funding_abs_min)
                        & (flow > flow_abs_min)
                        & (oi_z > oi_z_min)
                    )
                    crowded_short = (
                        feature_valid
                        & (funding < -funding_abs_min)
                        & (flow < -flow_abs_min)
                        & (oi_z > oi_z_min)
                    )
                    long_entry = crowded_short & (ret_z.shift(1) < 0.75)
                    short_entry = crowded_long & (ret_z.shift(1) > -0.75)
                    long_exit = (~feature_valid) | (funding > 0.0) | (flow > 0.0) | (ret_z > 0.75)
                    short_exit = (~feature_valid) | (funding < 0.0) | (flow < 0.0) | (ret_z < -0.75)
                    for min_hold in (4, 8):
                        signal = broad69._debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage in leverages:
                            sim = broad69.simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base = _candidate_base(
                                family="feature_flow_crowding_reversal",
                                model_parts=(
                                    "featureflow",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"fund{funding_abs_min}",
                                    f"flow{flow_abs_min}",
                                    f"oi{oi_z_min}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=flow_abs_min,
                                exit_threshold=funding_abs_min,
                                min_hold=min_hold,
                                leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base["feature_backed"] = True
                            base["feature_coverage"] = dict(coverage)
                            out.append(
                                _finalize_row(
                                    base=base,
                                    sim=sim,
                                    datetimes=datetimes,
                                    timeframe=timeframe,
                                    train=fold.train,
                                    validation=fold.validation,
                                    locked_oos=fold.locked_oos,
                                )
                            )
    return out


def _feature_liquidation_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty or "feature_valid" not in frame.columns:
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_liquidation_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    funding = pd.to_numeric(frame["funding_rate"], errors="coerce").reset_index(drop=True)
    open_interest = pd.to_numeric(frame["open_interest"], errors="coerce").reset_index(drop=True)
    liq_imbalance = pd.to_numeric(frame["liquidation_imbalance"], errors="coerce").reset_index(
        drop=True
    )
    feature_valid = (
        frame["feature_liquidation_valid"].fillna(False).astype(bool).reset_index(drop=True)
    )
    datetimes = frame["datetime"]
    returns = close.pct_change(fill_method=None)
    for lookback in (12, 24):
        oi_z = broad69._rolling_zscore(open_interest.pct_change(fill_method=None), lookback)
        liq_z = broad69._rolling_zscore(liq_imbalance.abs(), lookback)
        ret_z = broad69._rolling_zscore(returns, lookback)
        for liq_z_min in (1.0, 1.5):
            for funding_abs_min in (0.00005, 0.00010):
                for oi_z_min in (0.5, 1.0):
                    long_entry = (
                        feature_valid
                        & (liq_imbalance.shift(1) < -0.20)
                        & (liq_z.shift(1) > liq_z_min)
                        & (funding.shift(1) < -funding_abs_min)
                        & (oi_z.shift(1) > oi_z_min)
                        & (ret_z.shift(1) > -1.0)
                    )
                    short_entry = (
                        feature_valid
                        & (liq_imbalance.shift(1) > 0.20)
                        & (liq_z.shift(1) > liq_z_min)
                        & (funding.shift(1) > funding_abs_min)
                        & (oi_z.shift(1) > oi_z_min)
                        & (ret_z.shift(1) < 1.0)
                    )
                    long_exit = (~feature_valid) | (liq_imbalance > 0.0) | (ret_z > 0.75)
                    short_exit = (~feature_valid) | (liq_imbalance < 0.0) | (ret_z < -0.75)
                    for min_hold in (4, 8):
                        signal = broad69._debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage in leverages:
                            sim = broad69.simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base = _candidate_base(
                                family="feature_liquidation_imbalance_reversal",
                                model_parts=(
                                    "featureliq",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"liq{liq_z_min}",
                                    f"fund{funding_abs_min}",
                                    f"oi{oi_z_min}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=liq_z_min,
                                exit_threshold=funding_abs_min,
                                min_hold=min_hold,
                                leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base["feature_backed"] = True
                            base["feature_coverage"] = dict(coverage)
                            out.append(
                                _finalize_row(
                                    base=base,
                                    sim=sim,
                                    datetimes=datetimes,
                                    timeframe=timeframe,
                                    train=fold.train,
                                    validation=fold.validation,
                                    locked_oos=fold.locked_oos,
                                )
                            )
    return out


def _feature_flow_trend_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty or "feature_oi_flow_valid" not in frame.columns:
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_oi_flow_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    funding = pd.to_numeric(frame["funding_rate"], errors="coerce").reset_index(drop=True)
    open_interest = pd.to_numeric(frame["open_interest"], errors="coerce").reset_index(drop=True)
    flow = pd.to_numeric(frame["taker_buy_sell_imbalance"], errors="coerce").reset_index(drop=True)
    feature_valid = frame["feature_oi_flow_valid"].fillna(False).astype(bool).reset_index(drop=True)
    datetimes = frame["datetime"]
    for lookback in (12, 24):
        oi_z = broad69._rolling_zscore(open_interest.pct_change(fill_method=None), lookback)
        flow_smooth = flow.rolling(lookback).mean()
        for flow_abs_min in (0.15, 0.30):
            for oi_z_min in (0.5, 1.0):
                for funding_abs_max in (0.00010, 0.00020):
                    long_entry = (
                        feature_valid
                        & (flow_smooth.shift(1) > flow_abs_min)
                        & (oi_z.shift(1) > oi_z_min)
                        & (funding.shift(1).abs() <= funding_abs_max)
                    )
                    short_entry = (
                        feature_valid
                        & (flow_smooth.shift(1) < -flow_abs_min)
                        & (oi_z.shift(1) > oi_z_min)
                        & (funding.shift(1).abs() <= funding_abs_max)
                    )
                    long_exit = (
                        (~feature_valid)
                        | (flow_smooth < 0.0)
                        | (funding.abs() > funding_abs_max * 1.5)
                    )
                    short_exit = (
                        (~feature_valid)
                        | (flow_smooth > 0.0)
                        | (funding.abs() > funding_abs_max * 1.5)
                    )
                    for min_hold in (4, 8):
                        signal = broad69._debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage in leverages:
                            sim = broad69.simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base = _candidate_base(
                                family="feature_flow_oi_trend_continuation",
                                model_parts=(
                                    "featuretrend",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"flow{flow_abs_min}",
                                    f"oi{oi_z_min}",
                                    f"fundcap{funding_abs_max}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=flow_abs_min,
                                exit_threshold=funding_abs_max,
                                min_hold=min_hold,
                                leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base["feature_backed"] = True
                            base["feature_coverage"] = dict(coverage)
                            out.append(
                                _finalize_row(
                                    base=base,
                                    sim=sim,
                                    datetimes=datetimes,
                                    timeframe=timeframe,
                                    train=fold.train,
                                    validation=fold.validation,
                                    locked_oos=fold.locked_oos,
                                )
                            )
    return out


def _feature_crowding_continuation_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty or "feature_oi_flow_valid" not in frame.columns:
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_oi_flow_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    funding = pd.to_numeric(frame["funding_rate"], errors="coerce").reset_index(drop=True)
    open_interest = pd.to_numeric(frame["open_interest"], errors="coerce").reset_index(drop=True)
    flow = pd.to_numeric(frame["taker_buy_sell_imbalance"], errors="coerce").reset_index(drop=True)
    feature_valid = frame["feature_oi_flow_valid"].fillna(False).astype(bool).reset_index(drop=True)
    datetimes = frame["datetime"]
    for lookback in (6, 12):
        momentum = close / close.shift(lookback) - 1.0
        oi_change = open_interest / open_interest.shift(lookback) - 1.0
        for funding_abs_max in (0.00010, 0.00025):
            for imbalance_threshold in (0.0, 0.05):
                long_entry = (
                    (momentum > 0.01)
                    & (oi_change > 0.0)
                    & (flow > imbalance_threshold)
                    & (funding.abs() <= funding_abs_max)
                    & feature_valid
                )
                short_entry = (
                    (momentum < -0.01)
                    & (oi_change > 0.0)
                    & (flow < -imbalance_threshold)
                    & (funding.abs() <= funding_abs_max)
                    & feature_valid
                )
                long_exit = (momentum < 0.0) | (~feature_valid)
                short_exit = (momentum > 0.0) | (~feature_valid)
                signal = broad69._debounced_state_signal(
                    long_entry,
                    long_exit,
                    short_entry,
                    short_exit,
                    side="long_short",
                    min_hold_bars=8,
                    cooldown_bars=2,
                )
                for leverage in leverages:
                    sim = broad69.simulate_symbol(
                        frame,
                        signal,
                        integer_leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                    )
                    base = _candidate_base(
                        family="funding_oi_taker_crowding_continuation",
                        model_parts=(
                            "featurecrowding",
                            timeframe,
                            symbol,
                            f"lb{lookback}",
                            f"fund{funding_abs_max}",
                            f"imb{imbalance_threshold}",
                            "hold8",
                            f"lev{leverage}",
                        ),
                        symbol=symbol,
                        timeframe=timeframe,
                        side="long_short",
                        lookback=lookback,
                        threshold=imbalance_threshold,
                        exit_threshold=funding_abs_max,
                        min_hold=8,
                        leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                    )
                    base["feature_backed"] = True
                    base["feature_coverage"] = dict(coverage)
                    out.append(
                        _finalize_row(
                            base=base,
                            sim=sim,
                            datetimes=datetimes,
                            timeframe=timeframe,
                            train=fold.train,
                            validation=fold.validation,
                            locked_oos=fold.locked_oos,
                        )
                    )
    return out


def _perp_crowding_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty or "feature_valid" not in frame.columns:
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_liquidation_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    funding = pd.to_numeric(frame["funding_rate"], errors="coerce").reset_index(drop=True)
    open_interest = pd.to_numeric(frame["open_interest"], errors="coerce").reset_index(drop=True)
    liq_imbalance = pd.to_numeric(frame["liquidation_imbalance"], errors="coerce").reset_index(
        drop=True
    )
    feature_valid = (
        frame["feature_liquidation_valid"].fillna(False).astype(bool).reset_index(drop=True)
    )
    datetimes = frame["datetime"]
    returns = close.pct_change(fill_method=None)
    for lookback in (24, 48):
        funding_z = broad69._rolling_zscore(funding, max(16, lookback))
        oi_delta = open_interest.pct_change(fill_method=None)
        oi_delta_z = broad69._rolling_zscore(oi_delta, max(12, lookback // 2))
        liq_z = broad69._rolling_zscore(liq_imbalance, max(12, lookback // 2))
        crowding = np.tanh(
            0.45 * funding_z.fillna(0.0) + 0.35 * oi_delta_z.fillna(0.0) + 0.05 * liq_z.fillna(0.0)
        )
        ret_z = broad69._rolling_zscore(returns, max(12, lookback // 2))
        for crowding_threshold in (0.55, 0.75):
            long_entry = (
                feature_valid & (crowding.shift(1) <= -crowding_threshold) & (ret_z.shift(1) > -1.0)
            )
            short_entry = (
                feature_valid & (crowding.shift(1) >= crowding_threshold) & (ret_z.shift(1) < 1.0)
            )
            long_exit = (~feature_valid) | (crowding > -0.10) | (ret_z > 0.75)
            short_exit = (~feature_valid) | (crowding < 0.10) | (ret_z < -0.75)
            for min_hold in (4, 8):
                signal = broad69._debounced_state_signal(
                    long_entry,
                    long_exit,
                    short_entry,
                    short_exit,
                    side="long_short",
                    min_hold_bars=min_hold,
                    cooldown_bars=2,
                )
                for leverage in leverages:
                    sim = broad69.simulate_symbol(
                        frame,
                        signal,
                        integer_leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                    )
                    base = _candidate_base(
                        family="perp_crowding_score_reversion",
                        model_parts=(
                            "crowdscore",
                            timeframe,
                            symbol,
                            f"lb{lookback}",
                            f"thr{crowding_threshold}",
                            f"hold{min_hold}",
                            f"lev{leverage}",
                        ),
                        symbol=symbol,
                        timeframe=timeframe,
                        side="long_short",
                        lookback=lookback,
                        threshold=crowding_threshold,
                        exit_threshold=0.10,
                        min_hold=min_hold,
                        leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                    )
                    base["feature_backed"] = True
                    base["feature_coverage"] = dict(coverage)
                    out.append(
                        _finalize_row(
                            base=base,
                            sim=sim,
                            datetimes=datetimes,
                            timeframe=timeframe,
                            train=fold.train,
                            validation=fold.validation,
                            locked_oos=fold.locked_oos,
                        )
                    )
    return out


def _feature_taker_flow_exhaustion_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty or "feature_valid" not in frame.columns:
        return out
    coverage = _split_feature_coverage(
        frame, train=fold.train, validation=fold.validation, locked_oos=fold.locked_oos
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    funding = pd.to_numeric(frame["funding_rate"], errors="coerce").reset_index(drop=True)
    flow = pd.to_numeric(frame["taker_buy_sell_imbalance"], errors="coerce").reset_index(drop=True)
    feature_valid = frame["feature_valid"].fillna(False).astype(bool).reset_index(drop=True)
    datetimes = frame["datetime"]
    returns = close.pct_change(fill_method=None)
    for lookback in (6, 12):
        extension = close / close.shift(lookback) - 1.0
        realized_vol = returns.rolling(max(6, lookback)).std(ddof=1)
        for flow_imbalance_min in (0.10, 0.15):
            for price_extension_min in (0.004, 0.008):
                for funding_abs_cap in (0.00015, 0.00030):
                    for max_realized_vol in (0.008, 0.012):
                        long_entry = (
                            feature_valid
                            & (extension.shift(1) <= -price_extension_min)
                            & (flow.shift(1) <= -flow_imbalance_min)
                            & (funding.shift(1).abs() <= funding_abs_cap)
                            & (realized_vol.shift(1) <= max_realized_vol)
                        )
                        short_entry = (
                            feature_valid
                            & (extension.shift(1) >= price_extension_min)
                            & (flow.shift(1) >= flow_imbalance_min)
                            & (funding.shift(1).abs() <= funding_abs_cap)
                            & (realized_vol.shift(1) <= max_realized_vol)
                        )
                        long_exit = (~feature_valid) | (extension > 0.0) | (flow > 0.0)
                        short_exit = (~feature_valid) | (extension < 0.0) | (flow < 0.0)
                        for min_hold in (4, 8):
                            signal = broad69._debounced_state_signal(
                                long_entry,
                                long_exit,
                                short_entry,
                                short_exit,
                                side="long_short",
                                min_hold_bars=min_hold,
                                cooldown_bars=2,
                            )
                            for leverage in leverages:
                                sim = broad69.simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base = _candidate_base(
                                    family="feature_taker_flow_exhaustion_reversal",
                                    model_parts=(
                                        "flowexhaust",
                                        timeframe,
                                        symbol,
                                        f"lb{lookback}",
                                        f"flow{flow_imbalance_min}",
                                        f"ret{price_extension_min}",
                                        f"fundcap{funding_abs_cap}",
                                        f"vol{max_realized_vol}",
                                        f"hold{min_hold}",
                                        f"lev{leverage}",
                                    ),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    side="long_short",
                                    lookback=lookback,
                                    threshold=flow_imbalance_min,
                                    exit_threshold=price_extension_min,
                                    min_hold=min_hold,
                                    leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base["feature_backed"] = True
                                base["feature_coverage"] = dict(coverage)
                                out.append(
                                    _finalize_row(
                                        base=base,
                                        sim=sim,
                                        datetimes=datetimes,
                                        timeframe=timeframe,
                                        train=fold.train,
                                        validation=fold.validation,
                                        locked_oos=fold.locked_oos,
                                    )
                                )
    return out


def _feature_bbo_flow_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if (
        frame.empty
        or "feature_bbo_valid" not in frame.columns
        or "bbo_spread_bps" not in frame.columns
    ):
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_bbo_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    flow = pd.to_numeric(frame["taker_buy_sell_imbalance"], errors="coerce").reset_index(drop=True)
    spread = pd.to_numeric(frame["bbo_spread_bps"], errors="coerce").reset_index(drop=True)
    feature_valid = frame["feature_bbo_valid"].fillna(False).astype(bool).reset_index(drop=True)
    datetimes = frame["datetime"]
    for lookback in (6, 12):
        spread_z = broad69._rolling_zscore(spread, max(6, lookback))
        extension = close / close.shift(lookback) - 1.0
        for spread_z_min in (1.0, 1.5):
            for flow_imbalance_min in (0.10, 0.15):
                for price_extension_min in (0.004, 0.008):
                    long_entry = (
                        feature_valid
                        & (spread_z.shift(1) >= spread_z_min)
                        & (extension.shift(1) <= -price_extension_min)
                        & (flow.shift(1) <= -flow_imbalance_min)
                    )
                    short_entry = (
                        feature_valid
                        & (spread_z.shift(1) >= spread_z_min)
                        & (extension.shift(1) >= price_extension_min)
                        & (flow.shift(1) >= flow_imbalance_min)
                    )
                    long_exit = (~feature_valid) | (spread_z < 0.25) | (flow > 0.0)
                    short_exit = (~feature_valid) | (spread_z < 0.25) | (flow < 0.0)
                    for min_hold in (4, 8):
                        signal = broad69._debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage in leverages:
                            sim = broad69.simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base = _candidate_base(
                                family="feature_bbo_flow_exhaustion_reversal",
                                model_parts=(
                                    "bboflow",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"spr{spread_z_min}",
                                    f"flow{flow_imbalance_min}",
                                    f"ret{price_extension_min}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=spread_z_min,
                                exit_threshold=price_extension_min,
                                min_hold=min_hold,
                                leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base["feature_backed"] = True
                            base["feature_coverage"] = dict(coverage)
                            out.append(
                                _finalize_row(
                                    base=base,
                                    sim=sim,
                                    datetimes=datetimes,
                                    timeframe=timeframe,
                                    train=fold.train,
                                    validation=fold.validation,
                                    locked_oos=fold.locked_oos,
                                )
                            )
    return out


def _feature_book_depth_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if (
        frame.empty
        or "feature_depth_valid" not in frame.columns
        or "book_depth_imbalance_1pct" not in frame.columns
    ):
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_depth_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    depth = pd.to_numeric(frame["book_depth_imbalance_1pct"], errors="coerce").reset_index(
        drop=True
    )
    feature_valid = frame["feature_depth_valid"].fillna(False).astype(bool).reset_index(drop=True)
    datetimes = frame["datetime"]
    for lookback in (6, 12):
        extension = close / close.shift(lookback) - 1.0
        depth_smooth = depth.rolling(max(2, lookback // 2), min_periods=2).mean()
        for imbalance_min in (0.10, 0.20):
            for price_extension_min in (0.003, 0.006):
                long_entry = (
                    feature_valid
                    & (depth_smooth.shift(1) >= imbalance_min)
                    & (extension.shift(1) <= -price_extension_min)
                )
                short_entry = (
                    feature_valid
                    & (depth_smooth.shift(1) <= -imbalance_min)
                    & (extension.shift(1) >= price_extension_min)
                )
                long_exit = (~feature_valid) | (depth_smooth < 0.0) | (extension > 0.0)
                short_exit = (~feature_valid) | (depth_smooth > 0.0) | (extension < 0.0)
                for min_hold in (4, 8):
                    signal = broad69._debounced_state_signal(
                        long_entry,
                        long_exit,
                        short_entry,
                        short_exit,
                        side="long_short",
                        min_hold_bars=min_hold,
                        cooldown_bars=2,
                    )
                    for leverage in leverages:
                        sim = broad69.simulate_symbol(
                            frame,
                            signal,
                            integer_leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                        )
                        base = _candidate_base(
                            family="feature_book_depth_imbalance_reversal",
                            model_parts=(
                                "bookdepth",
                                timeframe,
                                symbol,
                                f"lb{lookback}",
                                f"imb{imbalance_min}",
                                f"ret{price_extension_min}",
                                f"hold{min_hold}",
                                f"lev{leverage}",
                            ),
                            symbol=symbol,
                            timeframe=timeframe,
                            side="long_short",
                            lookback=lookback,
                            threshold=imbalance_min,
                            exit_threshold=price_extension_min,
                            min_hold=min_hold,
                            leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                        )
                        base["feature_backed"] = True
                        base["feature_coverage"] = dict(coverage)
                        out.append(
                            _finalize_row(
                                base=base,
                                sim=sim,
                                datetimes=datetimes,
                                timeframe=timeframe,
                                train=fold.train,
                                validation=fold.validation,
                                locked_oos=fold.locked_oos,
                            )
                        )
    return out


def _deep_research_funding_dislocation_trend_carry_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty or "feature_oi_flow_valid" not in frame.columns:
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_oi_flow_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    funding = pd.to_numeric(frame["funding_rate"], errors="coerce").reset_index(drop=True)
    open_interest = pd.to_numeric(frame["open_interest"], errors="coerce").reset_index(drop=True)
    feature_valid = frame["feature_oi_flow_valid"].fillna(False).astype(bool).reset_index(
        drop=True
    )
    datetimes = frame["datetime"]
    for lookback in (12, 24, 48):
        momentum = close / close.shift(lookback) - 1.0
        funding_carry = funding.rolling(max(4, lookback // 2)).mean()
        oi_change = open_interest.pct_change(fill_method=None)
        oi_z = broad69._rolling_zscore(oi_change, max(8, lookback))
        for momentum_min in (0.004, 0.008):
            for funding_carry_min in (0.0, 0.00005):
                for oi_z_cap in (1.5, 2.5):
                    long_entry = (
                        feature_valid
                        & (momentum.shift(1) >= momentum_min)
                        & (funding_carry.shift(1) <= -funding_carry_min)
                        & (oi_z.shift(1).abs() <= oi_z_cap)
                    )
                    short_entry = (
                        feature_valid
                        & (momentum.shift(1) <= -momentum_min)
                        & (funding_carry.shift(1) >= funding_carry_min)
                        & (oi_z.shift(1).abs() <= oi_z_cap)
                    )
                    long_exit = (
                        (~feature_valid)
                        | (momentum <= 0.0)
                        | (funding_carry > max(0.00010, funding_carry_min * 2.0))
                    )
                    short_exit = (
                        (~feature_valid)
                        | (momentum >= 0.0)
                        | (funding_carry < -max(0.00010, funding_carry_min * 2.0))
                    )
                    for min_hold in (4, 8):
                        signal = broad69._debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage in leverages:
                            sim = broad69.simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base = _candidate_base(
                                family="deep_research_funding_dislocation_trend_carry",
                                model_parts=(
                                    "drfundingcarry",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"mom{momentum_min}",
                                    f"fund{funding_carry_min}",
                                    f"oicap{oi_z_cap}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=momentum_min,
                                exit_threshold=funding_carry_min,
                                min_hold=min_hold,
                                leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base["feature_backed"] = True
                            base["feature_coverage"] = dict(coverage)
                            base["source_report"] = "desktop-deep-research-report-20260608"
                            base["no_nested_oos_mining"] = True
                            out.append(
                                _finalize_row(
                                    base=base,
                                    sim=sim,
                                    datetimes=datetimes,
                                    timeframe=timeframe,
                                    train=fold.train,
                                    validation=fold.validation,
                                    locked_oos=fold.locked_oos,
                                )
                            )
    return out


def _deep_research_vol_managed_momentum_crash_gate_rows(
    *,
    frame: pd.DataFrame,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    benchmark_frame = bars_by_symbol.get("BTCUSDT", frame)
    benchmark_close = benchmark_frame["close"].astype(float).reset_index(drop=True)
    if len(benchmark_close) != len(close):
        benchmark_close = close
    returns = close.pct_change(fill_method=None)
    benchmark_returns = benchmark_close.pct_change(fill_method=None)
    for lookback in (12, 24, 48):
        momentum = close / close.shift(lookback) - 1.0
        realized_vol = returns.rolling(max(6, lookback)).std(ddof=1)
        for momentum_min in (0.006, 0.012):
            for realized_vol_max in (0.018, 0.028):
                for crash_window in (12, 24):
                    benchmark_crash = benchmark_close / benchmark_close.shift(crash_window) - 1.0
                    benchmark_vol = benchmark_returns.rolling(max(6, crash_window)).std(ddof=1)
                    for benchmark_crash_return in (0.035, 0.055):
                        stress = (benchmark_crash <= -benchmark_crash_return) | (
                            benchmark_vol > realized_vol_max
                        )
                        stress_prev = stress.shift(1)
                        stress_prev = stress_prev.where(stress_prev.notna(), False).astype(bool)
                        stress_now = stress.where(stress.notna(), False).astype(bool)
                        long_entry = (
                            (momentum.shift(1) >= momentum_min)
                            & (realized_vol.shift(1) <= realized_vol_max)
                            & (~stress_prev)
                        )
                        short_entry = (
                            (momentum.shift(1) <= -momentum_min)
                            & (realized_vol.shift(1) <= realized_vol_max)
                            & (~stress_prev)
                        )
                        long_exit = (momentum <= 0.0) | stress_now
                        short_exit = (momentum >= 0.0) | stress_now
                        for min_hold in (4, 8):
                            signal = broad69._debounced_state_signal(
                                long_entry,
                                long_exit,
                                short_entry,
                                short_exit,
                                side="long_short",
                                min_hold_bars=min_hold,
                                cooldown_bars=2,
                            )
                            for leverage in leverages:
                                sim = broad69.simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base = _candidate_base(
                                    family="deep_research_vol_managed_momentum_crash_gate",
                                    model_parts=(
                                        "drvolmom",
                                        timeframe,
                                        symbol,
                                        f"lb{lookback}",
                                        f"mom{momentum_min}",
                                        f"vol{realized_vol_max}",
                                        f"cr{crash_window}",
                                        f"crret{benchmark_crash_return}",
                                        f"hold{min_hold}",
                                        f"lev{leverage}",
                                    ),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    side="long_short",
                                    lookback=lookback,
                                    threshold=momentum_min,
                                    exit_threshold=realized_vol_max,
                                    min_hold=min_hold,
                                    leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base["source_report"] = "desktop-deep-research-report-20260608"
                                base["no_nested_oos_mining"] = True
                                out.append(
                                    _finalize_row(
                                        base=base,
                                        sim=sim,
                                        datetimes=datetimes,
                                        timeframe=timeframe,
                                        train=fold.train,
                                        validation=fold.validation,
                                        locked_oos=fold.locked_oos,
                                    )
                                )
    return out


def _deep_research_flow_imbalance_liquidation_sweep_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty or "feature_liquidation_valid" not in frame.columns:
        return out
    coverage = _split_feature_coverage(
        frame,
        train=fold.train,
        validation=fold.validation,
        locked_oos=fold.locked_oos,
        column="feature_liquidation_valid",
    )
    if coverage["train"] < 0.60 or coverage["validation"] < 0.60:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    flow = pd.to_numeric(frame["taker_buy_sell_imbalance"], errors="coerce").reset_index(drop=True)
    liquidation = pd.to_numeric(frame["liquidation_imbalance"], errors="coerce").reset_index(
        drop=True
    )
    spread = pd.to_numeric(frame.get("bbo_spread_bps", pd.Series(np.nan, index=frame.index)))
    spread = spread.reset_index(drop=True)
    depth = pd.to_numeric(
        frame.get("book_depth_imbalance_1pct", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).reset_index(drop=True)
    feature_valid = (
        frame["feature_liquidation_valid"].fillna(False).astype(bool).reset_index(drop=True)
    )
    datetimes = frame["datetime"]
    returns = close.pct_change(fill_method=None)
    for lookback in (12, 24):
        liquidation_z = broad69._rolling_zscore(liquidation, max(8, lookback))
        flow_smooth = flow.rolling(max(4, lookback // 2)).mean()
        depth_smooth = depth.rolling(max(4, lookback // 2)).mean()
        micro_pressure = pd.concat([flow_smooth, depth_smooth], axis=1).mean(
            axis=1, skipna=True
        )
        for return_shock_min in (0.004, 0.008):
            for flow_or_depth_min in (0.10, 0.20):
                for liquidation_z_min in (1.0, 1.5):
                    for max_spread_bps in (8.0, 12.0):
                        spread_ok = spread.isna() | (spread <= max_spread_bps)
                        long_entry = (
                            feature_valid
                            & spread_ok
                            & (returns.shift(1) <= -return_shock_min)
                            & (liquidation_z.shift(1) >= liquidation_z_min)
                            & (micro_pressure.shift(1) >= flow_or_depth_min)
                        )
                        short_entry = (
                            feature_valid
                            & spread_ok
                            & (returns.shift(1) >= return_shock_min)
                            & (liquidation_z.shift(1) <= -liquidation_z_min)
                            & (micro_pressure.shift(1) <= -flow_or_depth_min)
                        )
                        long_exit = (
                            (~feature_valid)
                            | (micro_pressure < 0.0)
                            | (returns > return_shock_min)
                        )
                        short_exit = (
                            (~feature_valid)
                            | (micro_pressure > 0.0)
                            | (returns < -return_shock_min)
                        )
                        for min_hold in (4, 8):
                            signal = broad69._debounced_state_signal(
                                long_entry,
                                long_exit,
                                short_entry,
                                short_exit,
                                side="long_short",
                                min_hold_bars=min_hold,
                                cooldown_bars=2,
                            )
                            for leverage in leverages:
                                sim = broad69.simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base = _candidate_base(
                                    family="deep_research_flow_imbalance_liquidation_sweep",
                                    model_parts=(
                                        "drflowsweep",
                                        timeframe,
                                        symbol,
                                        f"lb{lookback}",
                                        f"shock{return_shock_min}",
                                        f"flow{flow_or_depth_min}",
                                        f"liq{liquidation_z_min}",
                                        f"spr{max_spread_bps}",
                                        f"hold{min_hold}",
                                        f"lev{leverage}",
                                    ),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    side="long_short",
                                    lookback=lookback,
                                    threshold=flow_or_depth_min,
                                    exit_threshold=return_shock_min,
                                    min_hold=min_hold,
                                    leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base["feature_backed"] = True
                                base["feature_coverage"] = dict(coverage)
                                base["source_report"] = "desktop-deep-research-report-20260608"
                                base["no_nested_oos_mining"] = True
                                out.append(
                                    _finalize_row(
                                        base=base,
                                        sim=sim,
                                        datetimes=datetimes,
                                        timeframe=timeframe,
                                        train=fold.train,
                                        validation=fold.validation,
                                        locked_oos=fold.locked_oos,
                                    )
                                )
    return out


def _rows_for_fold(
    *,
    bars: Mapping[tuple[str, str], pd.DataFrame],
    symbols: Sequence[str],
    timeframes: Sequence[str],
    features_by_symbol: Mapping[str, pd.DataFrame] | None = None,
    fold: monthly.MonthlyFold,
    max_candidates_per_fold: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timeframe in timeframes:
        bars_by_symbol = {
            symbol: bars.get((symbol, timeframe), pd.DataFrame()) for symbol in symbols
        }
        panel = broad69._close_panel(bars_by_symbol, symbols)
        for symbol in symbols:
            frame = bars.get((symbol, timeframe), pd.DataFrame())
            if frame.empty:
                continue
            if pd.Timestamp(frame["datetime"].min()) > fold.train[0]:
                continue
            kwargs = {
                "frame": frame,
                "symbol": symbol,
                "timeframe": timeframe,
                "fold": fold,
                "leverages": DEFAULT_LEVERAGES,
                "allocation_fraction": DEFAULT_ALLOCATION_FRACTION,
            }
            rows.extend(_squeeze_rows(**kwargs))
            rows.extend(_absorption_rows(**kwargs))
            rows.extend(_reclaim_rows(**kwargs))
            rows.extend(
                _deep_research_vol_managed_momentum_crash_gate_rows(
                    **kwargs,
                    bars_by_symbol=bars_by_symbol,
                )
            )
            features = (features_by_symbol or {}).get(symbol, pd.DataFrame())
            if not features.empty:
                feature_frame = _attach_feature_points(frame, features, timeframe=timeframe)
                rows.extend(_feature_flow_rows(**{**kwargs, "frame": feature_frame}))
                rows.extend(_feature_liquidation_rows(**{**kwargs, "frame": feature_frame}))
                rows.extend(_feature_flow_trend_rows(**{**kwargs, "frame": feature_frame}))
                rows.extend(
                    _feature_crowding_continuation_rows(**{**kwargs, "frame": feature_frame})
                )
                rows.extend(_perp_crowding_rows(**{**kwargs, "frame": feature_frame}))
                rows.extend(
                    _feature_taker_flow_exhaustion_rows(**{**kwargs, "frame": feature_frame})
                )
                rows.extend(_feature_bbo_flow_rows(**{**kwargs, "frame": feature_frame}))
                rows.extend(_feature_book_depth_rows(**{**kwargs, "frame": feature_frame}))
                rows.extend(
                    _deep_research_funding_dislocation_trend_carry_rows(
                        **{**kwargs, "frame": feature_frame}
                    )
                )
                rows.extend(
                    _deep_research_flow_imbalance_liquidation_sweep_rows(
                        **{**kwargs, "frame": feature_frame}
                    )
                )
            if not panel.empty:
                rows.extend(
                    _lead_lag_rows(
                        bars_by_symbol=bars_by_symbol,
                        panel=panel,
                        symbol=symbol,
                        timeframe=timeframe,
                        fold=fold,
                        leverages=DEFAULT_LEVERAGES,
                        allocation_fraction=DEFAULT_ALLOCATION_FRACTION,
                    )
                )
    return sorted(rows, key=lambda row: _score_row(row), reverse=True)[:max_candidates_per_fold]


def _select_fold_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if _eligible_for_freeze(row)]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (_score_row(row), str(row.get("model_id"))))


def _aggregate(selected_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    returns = [_safe_float(row.get("locked_oos_return_report_only")) for row in selected_rows]
    mdds = [_safe_float(row.get("locked_oos_mdd_report_only")) for row in selected_rows]
    profit_factor, profit_factor_unbounded = _profit_factor(returns)
    return {
        "fold_count": len(selected_rows),
        "compounded_oos_return": _compound(returns),
        "annualized_oos_return_approx": (1.0 + _compound(returns)) ** (12.0 / max(1, len(returns)))
        - 1.0,
        "monthly_equity_mdd": _equity_mdd(returns),
        "max_oos_mdd": max(mdds) if mdds else 0.0,
        "positive_oos_folds": sum(1 for value in returns if value > 0.0),
        "min_oos_return": min(returns) if returns else 0.0,
        "latest_oos_return": returns[-1] if returns else 0.0,
        "monthly_sharpe_approx": _sharpe(returns),
        "profit_factor": profit_factor,
        "profit_factor_unbounded": profit_factor_unbounded,
    }


def run(
    *,
    data_root: Path,
    output_dir: Path,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    feature_root: Path | None = None,
    max_folds: int | None,
    max_candidates_per_fold: int,
) -> dict[str, Any]:
    search_space = _search_space()
    search_hash = _search_space_hash(search_space)
    bars, coverage = broad69.load_all_bars(symbols, data_root=data_root, timeframes=timeframes)
    features_by_symbol = (
        {symbol: _load_feature_points_safe(symbol, feature_root=feature_root) for symbol in symbols}
        if feature_root is not None
        else {}
    )
    latest = pd.Timestamp(coverage.get("latest_available_data") or coverage["global_latest_utc"])
    folds = monthly.build_monthly_folds(
        train_start=pd.Timestamp(monthly.DEFAULT_TRAIN_START),
        first_oos_start=pd.Timestamp(monthly.DEFAULT_FIRST_OOS_START),
        latest_data=latest,
        bar_minutes=30,
    )
    if max_folds is not None:
        folds = folds[-int(max_folds) :]

    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for fold in folds:
        rows = _rows_for_fold(
            bars=bars,
            symbols=symbols,
            timeframes=timeframes,
            features_by_symbol=features_by_symbol,
            fold=fold,
            max_candidates_per_fold=max_candidates_per_fold,
        )
        selected = _select_fold_candidate(rows)
        for row in rows:
            row["fold_id"] = fold.fold_id
            row["selected_by_train_validation_freeze"] = bool(
                selected is not None and row.get("model_id") == selected.get("model_id")
            )
            row["pre_registered_search_space_sha256"] = search_hash
            row["candidate_freeze_sha256"] = hashlib.sha256(
                json.dumps(
                    {
                        "fold_id": fold.fold_id,
                        "model_id": row.get("model_id"),
                        "score": row.get("selection_score_train_validation_only"),
                        "train": row.get("train_return"),
                        "validation": row.get("validation_return"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        fold_rows.extend(rows)
        if selected is not None:
            selected_payload = dict(selected)
            selected_payload["fold_id"] = fold.fold_id
            selected_payload["selected_by_train_validation_freeze"] = True
            selected_payload["pre_registered_search_space_sha256"] = search_hash
            selected_rows.append(selected_payload)

    aggregate = _aggregate(selected_rows)
    payload = {
        "artifact_kind": "alpha_zoo_clean_new_alpha_discovery",
        "generated_at_utc": _utc_now_iso(),
        "search_space": search_space,
        "pre_registered_search_space_sha256": search_hash,
        "optimization_policy": optimization_search_policy_payload(
            search_method="bounded_grid_pre_registered_new_alpha",
            objective_policy="rank_train_validation_only_then_attach_locked_oos_report_gate",
            selection_inputs=["train", "validation"],
            bounded_grid_justification=(
                "small deterministic OHLCV new-alpha grid; no post-OOS selector, "
                "lagged-shadow, nested, or locked-OOS objective inputs"
            ),
            extra={
                "post_oos_selector_trusted": False,
                "fresh_forward_required": True,
                "real_money_execution": False,
                "split_simulation_policy": "continuous_full_period_signal_slice_report_only",
                "clean_promotion_eligible": False,
                "label_blockers": [
                    "continuous_position_state_across_split_boundaries",
                    "fresh_forward_required_before_promotion",
                ],
            },
        ),
        "universe": {"symbols": list(symbols), "timeframes": list(timeframes)},
        "data_coverage": coverage,
        "fold_count": len(folds),
        "selected_fold_rows": selected_rows,
        "candidate_rows": fold_rows,
        "aggregate": aggregate,
        "ready_for_real": False,
        "real_money_execution": False,
        "fresh_forward_required": True,
        "split_simulation_policy": "continuous_full_period_signal_slice_report_only",
        "clean_promotion_eligible": False,
        "label_blockers": [
            "continuous_position_state_across_split_boundaries",
            "fresh_forward_required_before_promotion",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "clean_new_alpha_discovery_latest.json"
    output_md = output_dir / "clean_new_alpha_discovery_latest.md"
    payload["output_paths"] = {"json": str(output_json), "markdown": str(output_md)}
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value) * 100.0:.2f}%"


def _render_markdown(payload: Mapping[str, Any]) -> str:
    agg = payload.get("aggregate") or {}
    lines = [
        "# Alpha Zoo clean new-alpha discovery",
        "",
        f"- generated: `{payload.get('generated_at_utc')}`",
        f"- pre-registered search hash: `{payload.get('pre_registered_search_space_sha256')}`",
        "- selection input: `train + validation only`",
        "- locked-OOS: `report/gate only after freeze`",
        "- split simulation policy: `continuous_full_period_signal_slice_report_only`",
        "- clean promotion eligible: `false`",
        "- post-OOS selector trusted: `false`",
        "- real-money: `false`",
        "",
        "## Aggregate selected fold result",
        "",
        f"- OOS comp: `{_fmt_pct(agg.get('compounded_oos_return'))}`",
        f"- annualized approx: `{_fmt_pct(agg.get('annualized_oos_return_approx'))}`",
        f"- monthly equity MDD: `{_fmt_pct(agg.get('monthly_equity_mdd'))}`",
        f"- max OOS MDD: `{_fmt_pct(agg.get('max_oos_mdd'))}`",
        f"- positive folds: `{agg.get('positive_oos_folds')}/{agg.get('fold_count')}`",
        f"- Sharpe approx: `{_safe_float(agg.get('monthly_sharpe_approx')):.2f}`",
        "",
        "## Fold selections",
        "",
        "| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("selected_fold_rows", []):
        lines.append(
            "| `{fold}` | `{model}` | `{family}` | {train} | {validation} | {oos} | {mdd} |".format(
                fold=row.get("fold_id"),
                model=row.get("model_id"),
                family=row.get("family"),
                train=_fmt_pct(row.get("train_return")),
                validation=_fmt_pct(row.get("validation_return")),
                oos=_fmt_pct(row.get("locked_oos_return_report_only")),
                mdd=_fmt_pct(row.get("locked_oos_mdd_report_only")),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(broad69.DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates-per-fold", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        symbols=tuple(item.strip() for item in str(args.symbols).split(",") if item.strip()),
        timeframes=tuple(item.strip() for item in str(args.timeframes).split(",") if item.strip()),
        feature_root=Path(args.feature_root) if str(args.feature_root).strip() else None,
        max_folds=args.max_folds,
        max_candidates_per_fold=int(args.max_candidates_per_fold),
    )
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
