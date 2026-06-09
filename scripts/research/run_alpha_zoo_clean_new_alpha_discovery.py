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
import heapq
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from lumina_quant.alpha_zoo import native_alpha_fold_backend  # noqa: E402
from lumina_quant.optimization.search_policy import optimization_search_policy_payload  # noqa: E402
from lumina_quant.research_universe import BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_monthly_refit_walkforward as monthly  # noqa: E402
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69  # noqa: E402
from scripts.research.run_alpha_zoo_htf_momentum_crowding_discovery import DEFAULT_FEATURE_ROOT  # noqa: E402

DEFAULT_OUTPUT_DIR = broad69.ALPHA_V2_ROOT / "alpha_zoo_clean_new_alpha_discovery_20260607"
DEFAULT_TIMEFRAMES = ("1h", "4h")
DEFAULT_LEVERAGES = (2, 3, 4)
DEFAULT_ALLOCATION_FRACTION = 0.10
DEFAULT_FOLD_WORKERS = 1
DEFAULT_SIMULATION_BACKEND = native_alpha_fold_backend.ALPHA_FOLD_BACKEND_AUTO
DEFAULT_SELECTION_POLICY = "default_train_validation"
ROBUST_SELECTION_POLICY = "robust_train_validation_v1"
SELECTION_POLICIES = (DEFAULT_SELECTION_POLICY, ROBUST_SELECTION_POLICY)

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
    "btc_beta_residual_momentum": (
        "Enter residual target momentum after rolling BTC/ETH beta adjustment, "
        "with benchmark crash and realized-volatility gates. Uses only prior "
        "bars and is designed to avoid paying for plain market beta."
    ),
    "cross_sectional_vol_adjusted_momentum": (
        "Enter residual relative-strength momentum after subtracting the panel "
        "median move and scaling by realized volatility. Uses prior bars only, "
        "with market-stress and volatility gates to avoid raw beta chasing."
    ),
    "cross_sectional_dispersion_gated_momentum": (
        "Enter residual relative-strength momentum only when lagged cross-sectional "
        "return dispersion is not elevated versus its own rolling history. This "
        "keeps the leaf aligned with crypto momentum state-dependence research "
        "without using locked-OOS feedback."
    ),
    "cross_sectional_residual_reversal": (
        "Fade short-horizon panel-median residual shocks when the target is not "
        "in a strong own-trend regime. This is a cross-sectional stat-arb style "
        "mean-reversion leaf using prior bars only, with market-stress and "
        "volatility gates."
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
    "indicator_vwap_atr_bollinger_reversion": (
        "Indicator leaf alpha: fade standardized VWAP/Bollinger dislocations only when "
        "ATR and realized-volatility gates keep the setup inside a tradable 30m+ "
        "mean-reversion regime."
    ),
    "indicator_kalman_volatility_trend": (
        "Indicator leaf alpha: use a scalar Kalman price filter as an adaptive moving "
        "average, then trade standardized filter slope with ATR/volatility gates."
    ),
    "indicator_kalman_residual_reversion": (
        "Indicator leaf alpha: use scalar Kalman filter residual z-score as an "
        "adaptive mean-reversion setup, gated by weak filter slope and realized "
        "volatility so it does not fight strong trends."
    ),
    "indicator_vwap_kalman_pullback_continuation": (
        "Indicator leaf alpha: trade pullback continuation when an adaptive Kalman "
        "trend, rolling VWAP, Bollinger location, ATR distance, and train/validation "
        "volatility gates agree. Uses prior bars only and is a theory-plausible "
        "30m+ trend-reentry setup, not a hard-coded post-OOS rescue rule."
    ),
    "standardized_indicator_ridge_directional": (
        "Small ML-style leaf alpha: train-only standardized OHLCV indicator features "
        "feed a bounded ridge directional score; train/validation select thresholds "
        "and locked OOS remains report-only."
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
        "btc_beta_residual_momentum": {
            "lookback": [12, 24, 48],
            "beta_window": [48, 96],
            "residual_threshold": [0.003, 0.006],
            "benchmark_crash_gate": [0.025, 0.050],
            "max_realized_vol": [0.012, 0.020],
            "min_hold": [4, 8],
        },
        "cross_sectional_vol_adjusted_momentum": {
            "lookback": [12, 24, 48],
            "vol_window": [24, 48],
            "score_threshold": [0.50, 1.00],
            "market_stress_gate": [0.025, 0.050],
            "max_realized_vol": [0.012, 0.020],
            "min_hold": [4, 8],
        },
        "cross_sectional_dispersion_gated_momentum": {
            "lookback": [12, 24, 48],
            "vol_window": [24, 48],
            "score_threshold": [0.50, 1.00],
            "dispersion_window": [48, 96],
            "max_dispersion_quantile": [0.60, 0.80],
            "market_stress_gate": [0.025, 0.050],
            "max_realized_vol": [0.012, 0.020],
            "min_hold": [4, 8],
        },
        "cross_sectional_residual_reversal": {
            "lookback": [12, 24, 48],
            "vol_window": [24, 48],
            "residual_z_min": [1.25, 1.75],
            "max_trend_z": [0.75, 1.25],
            "market_stress_gate": [0.025, 0.050],
            "max_realized_vol": [0.012, 0.020],
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
        "indicator_vwap_atr_bollinger_reversion": {
            "lookback": [24, 48],
            "vwap_deviation_z": [1.25, 1.75],
            "bollinger_z": [1.5, 2.0],
            "max_atr_pct": [0.018, 0.028],
            "min_hold": [4, 8],
        },
        "indicator_kalman_volatility_trend": {
            "lookback": [24, 48],
            "process_noise": [0.0005, 0.002],
            "measurement_noise": [0.02, 0.08],
            "slope_z_min": [0.25, 0.50],
            "max_realized_vol": [0.018, 0.028],
            "min_hold": [4, 8],
        },
        "indicator_kalman_residual_reversion": {
            "lookback": [24, 48],
            "process_noise": [0.0005, 0.002],
            "measurement_noise": [0.02, 0.08],
            "residual_z_min": [1.25, 1.75],
            "max_slope_z": [0.25, 0.50],
            "max_realized_vol": [0.018, 0.028],
            "min_hold": [4, 8],
        },
        "indicator_vwap_kalman_pullback_continuation": {
            "lookback": [48, 96],
            "process_noise": [0.0005, 0.002],
            "measurement_noise": [0.02, 0.08],
            "trend_slope_z": [0.25, 0.50],
            "pullback_z": [0.25, 0.75],
            "max_atr_distance": [1.5, 2.5],
            "max_realized_vol": [0.018, 0.028],
            "min_hold": [4, 8],
        },
        "standardized_indicator_ridge_directional": {
            "feature_set": ["returns", "vwap", "bollinger", "atr", "kalman", "volume"],
            "ridge_alpha": [1.0, 10.0],
            "score_threshold": [0.0005, 0.0010],
            "standardization_scope": "train_only",
            "min_train_observations": 240,
            "min_hold": [4, 8],
        },
    }


def _search_space_hash(space: Mapping[str, Any]) -> str:
    encoded = json.dumps(space, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_enabled_families(enabled_families: Sequence[str] | None) -> tuple[str, ...]:
    if enabled_families is None:
        return tuple(FAMILY_DESCRIPTIONS)
    requested = tuple(str(item).strip() for item in enabled_families if str(item).strip())
    if not requested:
        return tuple(FAMILY_DESCRIPTIONS)
    unknown = sorted(set(requested) - set(FAMILY_DESCRIPTIONS))
    if unknown:
        raise ValueError(f"unknown alpha families: {', '.join(unknown)}")
    return requested


def _normalize_leverages(leverages: Sequence[int] | None) -> tuple[int, ...]:
    if leverages is None:
        return tuple(DEFAULT_LEVERAGES)
    normalized = tuple(int(item) for item in leverages)
    if not normalized:
        return tuple(DEFAULT_LEVERAGES)
    if any(value <= 0 for value in normalized):
        raise ValueError(f"leverages must be positive integers: {normalized}")
    return normalized


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


def _train_validation_return_ratio(row: Mapping[str, Any]) -> float:
    explicit = row.get("train_validation_return_ratio")
    if explicit is not None:
        return _safe_float(explicit, 999.0)
    validation = _safe_float(row.get("validation_return"))
    return _safe_float(row.get("train_return")) / max(abs(validation), 0.02)


def _robust_v1_eligible(row: Mapping[str, Any]) -> bool:
    if row.get("uses_locked_oos_for_selection"):
        return False
    if row.get("uses_locked_oos_for_objective"):
        return False
    if row.get("uses_locked_oos_for_pruning"):
        return False
    if row.get("uses_locked_oos_for_parameter_fitting"):
        return False
    train = _safe_float(row.get("train_return"))
    validation = _safe_float(row.get("validation_return"))
    train_mdd = _safe_float(row.get("train_mdd"))
    validation_mdd = _safe_float(row.get("validation_mdd"))
    ratio = _train_validation_return_ratio(row)
    if train <= 0.0 or validation <= 0.0:
        return False
    if train_mdd > 0.35 or validation_mdd > 0.12:
        return False
    if ratio < 0.25 or ratio > 3.0:
        return False
    if train - validation > 0.45:
        return False
    if int(row.get("validation_trade_event_count") or 0) < 2:
        return False
    train_rpt = _safe_float(row.get("train_return_per_turnover_proxy_bps"), -100.0)
    validation_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    return train_rpt > 0.0 and validation_rpt > 0.0


def _robust_v1_score_row(row: Mapping[str, Any]) -> float:
    validation = _safe_float(row.get("validation_return"))
    train = _safe_float(row.get("train_return"))
    validation_mdd = _safe_float(row.get("validation_mdd"))
    train_mdd = _safe_float(row.get("train_mdd"))
    validation_calmar = validation / max(validation_mdd, 0.02)
    train_calmar = train / max(train_mdd, 0.02)
    gap_penalty = abs(train - validation) / max(abs(validation), 0.05)
    validation_rpt = _safe_float(row.get("validation_return_per_turnover_proxy_bps"), -100.0)
    rpt_bonus = min(max(validation_rpt, -50.0), 200.0) / 100.0
    trade_count_bonus = min(float(row.get("validation_trade_event_count") or 0.0), 20.0) / 100.0
    return float(
        2.0 * validation_calmar
        + 0.5 * min(train_calmar, validation_calmar * 1.5)
        + 0.25 * rpt_bonus
        + trade_count_bonus
        - 0.3 * gap_penalty
        - 0.5 * validation_mdd
        - 0.25 * train_mdd
    )


def _selection_score(row: Mapping[str, Any], *, selection_policy: str) -> float:
    if selection_policy == DEFAULT_SELECTION_POLICY:
        return _score_row(row)
    if selection_policy == ROBUST_SELECTION_POLICY:
        return _robust_v1_score_row(row)
    raise ValueError(f"unknown selection policy: {selection_policy}")


def _eligible_for_policy(row: Mapping[str, Any], *, selection_policy: str) -> bool:
    if selection_policy == DEFAULT_SELECTION_POLICY:
        return _eligible_for_freeze(row)
    if selection_policy == ROBUST_SELECTION_POLICY:
        return _eligible_for_freeze(row) and _robust_v1_eligible(row)
    raise ValueError(f"unknown selection policy: {selection_policy}")


def _candidate_cap_key(row: Mapping[str, Any], *, selection_policy: str) -> tuple[bool, float, str]:
    return (
        _eligible_for_policy(row, selection_policy=selection_policy),
        _selection_score(row, selection_policy=selection_policy),
        str(row.get("model_id")),
    )


def _cap_rows_for_selection(
    rows: Sequence[dict[str, Any]],
    *,
    max_candidates_per_fold: int,
    selection_policy: str,
) -> list[dict[str, Any]]:
    if len(rows) <= max_candidates_per_fold or max_candidates_per_fold <= 0:
        return sorted(
            rows,
            key=lambda row: _candidate_cap_key(row, selection_policy=selection_policy),
            reverse=True,
        )
    return heapq.nlargest(
        max_candidates_per_fold,
        rows,
        key=lambda row: _candidate_cap_key(row, selection_policy=selection_policy),
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


def _true_range_pct(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"].astype(float).reset_index(drop=True)
    low = frame["low"].astype(float).reset_index(drop=True)
    close = frame["close"].astype(float).reset_index(drop=True)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    return true_range / close.replace(0.0, np.nan)


def _rolling_vwap(close: pd.Series, volume: pd.Series, lookback: int) -> pd.Series:
    volume_clean = volume.astype(float).clip(lower=0.0)
    notional = close.astype(float) * volume_clean
    denom = volume_clean.rolling(lookback).sum().replace(0.0, np.nan)
    return notional.rolling(lookback).sum() / denom


def _kalman_log_price_filter(
    close: pd.Series,
    *,
    process_noise: float,
    measurement_noise: float,
) -> pd.Series:
    observations = np.log(close.astype(float).replace(0.0, np.nan)).to_numpy(dtype=float)
    filtered = np.full(observations.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(observations))
    if finite.size == 0:
        return pd.Series(filtered, index=close.index)
    state = float(observations[finite[0]])
    covariance = 1.0
    process_var = max(1e-12, float(process_noise) ** 2)
    measurement_var = max(1e-12, float(measurement_noise) ** 2)
    for idx, observation in enumerate(observations):
        covariance += process_var
        if math.isfinite(float(observation)):
            gain = covariance / (covariance + measurement_var)
            state = state + gain * (float(observation) - state)
            covariance = (1.0 - gain) * covariance
        filtered[idx] = state
    return pd.Series(filtered, index=close.index)


def _train_standardized_frame(
    raw_features: pd.DataFrame,
    train_mask: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(index=raw_features.index)
    train_mask = np.asarray(train_mask, dtype=bool)
    for column in raw_features.columns:
        values = pd.to_numeric(raw_features[column], errors="coerce")
        train_values = values[train_mask & np.isfinite(values.to_numpy(dtype=float))]
        mean = float(train_values.mean()) if len(train_values) else 0.0
        std = float(train_values.std(ddof=1)) if len(train_values) > 1 else 0.0
        if not math.isfinite(std) or std <= 1e-12:
            out[column] = np.nan
        else:
            out[column] = (values - mean) / std
    return out


def _ridge_directional_score(
    features: pd.DataFrame,
    target: pd.Series,
    train_mask: np.ndarray,
    *,
    ridge_alpha: float,
    min_train_observations: int = 240,
) -> pd.Series | None:
    train_mask = np.asarray(train_mask, dtype=bool)
    x_all = features.to_numpy(dtype=float)
    y_all = target.to_numpy(dtype=float)
    finite_rows = np.isfinite(x_all).all(axis=1) & np.isfinite(y_all)
    train_rows = finite_rows & train_mask
    if int(np.count_nonzero(train_rows)) < int(min_train_observations):
        return None
    x_train = x_all[train_rows]
    y_train = y_all[train_rows]
    x_train_design = np.column_stack([np.ones(x_train.shape[0]), x_train])
    penalty = np.eye(x_train_design.shape[1], dtype=float) * max(0.0, float(ridge_alpha))
    penalty[0, 0] = 0.0
    lhs = x_train_design.T @ x_train_design + penalty
    rhs = x_train_design.T @ y_train
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    x_full = np.where(np.isfinite(x_all), x_all, 0.0)
    score = np.column_stack([np.ones(x_full.shape[0]), x_full]) @ beta
    score[~finite_rows] = np.nan
    return pd.Series(score, index=features.index)


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


_FRAME_ARRAY_CACHE: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray]] = {}
_SPLIT_MASK_CACHE: dict[
    tuple[int, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}


def _frame_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_key = id(frame)
    cached = _FRAME_ARRAY_CACHE.get(cache_key)
    if cached is not None and cached[0] == len(frame):
        return cached[1], cached[2], cached[3]
    close = frame["close"].to_numpy(dtype=float, copy=False)
    high = frame["high"].to_numpy(dtype=float, copy=False)
    low = frame["low"].to_numpy(dtype=float, copy=False)
    arrays = (
        np.ascontiguousarray(close, dtype=np.float64),
        np.ascontiguousarray(high, dtype=np.float64),
        np.ascontiguousarray(low, dtype=np.float64),
    )
    _FRAME_ARRAY_CACHE[cache_key] = (len(frame), arrays[0], arrays[1], arrays[2])
    return arrays


def _simulate_symbol(
    frame: pd.DataFrame,
    signal: np.ndarray,
    *,
    integer_leverage: int,
    allocation_fraction: float,
    round_trip_cost_bps: float = broad69.PRIMARY_ROUND_TRIP_COST_BPS,
    simulation_backend: str | None = None,
) -> broad69.SimResult:
    close, high, low = _frame_arrays(frame)
    signal_arr = np.ascontiguousarray(np.asarray(signal, dtype=np.float64), dtype=np.float64)
    if signal_arr.size != close.size:
        raise ValueError("signal length must equal bars length")
    returns, liquidation, account_wipeout = native_alpha_fold_backend.simulate_symbol_arrays(
        close,
        high,
        low,
        signal_arr,
        integer_leverage=int(integer_leverage),
        allocation_fraction=float(allocation_fraction),
        round_trip_cost_bps=float(round_trip_cost_bps),
        backend=simulation_backend,
    )
    return broad69.SimResult(
        returns=returns,
        position=signal_arr,
        liquidation_flags=liquidation,
        account_wipeout_flags=account_wipeout,
    )


def _split_masks(
    datetimes: pd.Series | pd.DatetimeIndex,
    *,
    train: tuple[pd.Timestamp, pd.Timestamp],
    validation: tuple[pd.Timestamp, pd.Timestamp],
    locked_oos: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (
        id(datetimes),
        len(datetimes),
        pd.Timestamp(train[0]),
        pd.Timestamp(train[1]),
        pd.Timestamp(validation[0]),
        pd.Timestamp(validation[1]),
        pd.Timestamp(locked_oos[0]),
        pd.Timestamp(locked_oos[1]),
    )
    cached = _SPLIT_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    idx = pd.DatetimeIndex(pd.to_datetime(datetimes))
    masks = (
        np.asarray((idx >= train[0]) & (idx <= train[1]), dtype=bool),
        np.asarray((idx >= validation[0]) & (idx <= validation[1]), dtype=bool),
        np.asarray((idx >= locked_oos[0]) & (idx <= locked_oos[1]), dtype=bool),
    )
    _SPLIT_MASK_CACHE[key] = masks
    return masks


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
    train_mask, validation_mask, locked_mask = _split_masks(
        datetimes,
        train=train,
        validation=validation,
        locked_oos=locked_oos,
    )
    for split, mask in (("train", train_mask), ("validation", validation_mask)):
        metrics = broad69.split_metrics(
            sim.returns[mask],
            sim.position[mask],
            sim.liquidation_flags[mask],
            sim.account_wipeout_flags[mask],
            timeframe=timeframe,
        )
        row[f"{split}_return"] = metrics["total_return"]
        row[f"{split}_mdd"] = metrics["max_drawdown"]
        row[f"{split}_sharpe"] = metrics["sharpe"]
        row[f"{split}_sortino"] = metrics["sortino"]
        row[f"{split}_calmar"] = metrics["calmar"]
        row[f"{split}_trade_event_count"] = metrics["trade_event_count"]
        row[f"{split}_exposure_bar_count"] = metrics["exposure_bar_count"]
    validation_return = float(row.get("validation_return") or 0.0)
    train_return = float(row.get("train_return") or 0.0)
    row["train_validation_return_ratio"] = (
        train_return / validation_return if validation_return > 0.0 else 0.0
    )
    row["train_minus_validation_return"] = train_return - validation_return
    notional = float(row["notional_fraction"])
    for split in broad69.SPLIT_ORDER:
        row[f"{split}_return_per_turnover_proxy_bps"] = broad69._return_per_turnover(
            float(row[f"{split}_return"]),
            int(row[f"{split}_trade_event_count"]),
            notional,
        )
    row["train_validation_score"] = broad69._candidate_score(row)
    row = broad69.gate_candidate(row)
    locked = broad69.split_metrics(
        sim.returns[locked_mask],
        sim.position[locked_mask],
        sim.liquidation_flags[locked_mask],
        sim.account_wipeout_flags[locked_mask],
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
    simulation_backend: str | None = None,
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
                        sim = _simulate_symbol(
                            frame,
                            signal,
                            integer_leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                            simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                        sim = _simulate_symbol(
                            frame,
                            signal,
                            integer_leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                            simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                    sim = _simulate_symbol(
                        frame,
                        signal,
                        integer_leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                        simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                            sim = _simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                                simulation_backend=simulation_backend,
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


def _btc_beta_residual_momentum_rows(
    *,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    panel: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    frame = bars_by_symbol.get(symbol, pd.DataFrame())
    if frame.empty or symbol not in panel:
        return out
    benchmark_symbol = "BTCUSDT" if symbol != "BTCUSDT" else "ETHUSDT"
    if benchmark_symbol not in panel:
        return out
    datetimes = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
    target_close = panel[symbol].reindex(datetimes).ffill()
    benchmark_close = panel[benchmark_symbol].reindex(datetimes).ffill()
    target_returns = target_close.pct_change(fill_method=None)
    benchmark_returns = benchmark_close.pct_change(fill_method=None)
    for lookback in (12, 24, 48):
        benchmark_momentum = benchmark_close / benchmark_close.shift(lookback) - 1.0
        realized_vol = target_returns.rolling(max(lookback, 24)).std(ddof=1)
        for beta_window in (48, 96):
            beta_local = target_returns.rolling(beta_window).cov(benchmark_returns) / (
                benchmark_returns.rolling(beta_window).var(ddof=1).replace(0.0, np.nan)
            )
            residual_local = target_returns - beta_local.clip(-3.0, 3.0) * benchmark_returns
            residual_signal = residual_local.rolling(lookback).sum()
            for residual_threshold in (0.003, 0.006):
                for benchmark_crash_gate in (0.025, 0.050):
                    for max_realized_vol in (0.012, 0.020):
                        long_entry = (
                            (residual_signal.shift(1) > residual_threshold)
                            & (benchmark_momentum.shift(1) > -benchmark_crash_gate)
                            & (realized_vol.shift(1) <= max_realized_vol)
                        )
                        short_entry = (
                            (residual_signal.shift(1) < -residual_threshold)
                            & (benchmark_momentum.shift(1) < benchmark_crash_gate)
                            & (realized_vol.shift(1) <= max_realized_vol)
                        )
                        long_exit = (residual_signal < 0.0) | (
                            realized_vol > max_realized_vol * 1.5
                        )
                        short_exit = (residual_signal > 0.0) | (
                            realized_vol > max_realized_vol * 1.5
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
                                sim = _simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                    simulation_backend=simulation_backend,
                                )
                                base = _candidate_base(
                                    family="btc_beta_residual_momentum",
                                    model_parts=(
                                        "betaresmom",
                                        timeframe,
                                        benchmark_symbol,
                                        symbol,
                                        f"lb{lookback}",
                                        f"bw{beta_window}",
                                        f"thr{residual_threshold}",
                                        f"crash{benchmark_crash_gate}",
                                        f"vol{max_realized_vol}",
                                        f"hold{min_hold}",
                                        f"lev{leverage}",
                                    ),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    side="long_short",
                                    lookback=lookback,
                                    threshold=residual_threshold,
                                    exit_threshold=benchmark_crash_gate,
                                    min_hold=min_hold,
                                    leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base["benchmark_symbol"] = benchmark_symbol
                                base["beta_window"] = beta_window
                                base["max_realized_vol"] = max_realized_vol
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


def _cross_sectional_vol_adjusted_momentum_rows(
    *,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    panel: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    frame = bars_by_symbol.get(symbol, pd.DataFrame())
    if frame.empty or symbol not in panel or len(panel.columns) < 3:
        return out
    datetimes = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
    close_panel = panel.reindex(datetimes).ffill()
    target_close = close_panel[symbol]
    target_returns = target_close.pct_change(fill_method=None)
    panel_returns = close_panel.pct_change(fill_method=None)
    for lookback in (12, 24, 48):
        target_momentum = target_close / target_close.shift(lookback) - 1.0
        panel_momentum = close_panel / close_panel.shift(lookback) - 1.0
        market_median_momentum = panel_momentum.median(axis=1, skipna=True)
        residual_momentum = target_momentum - market_median_momentum
        market_stress = panel_returns.median(axis=1, skipna=True).rolling(lookback).sum()
        for vol_window in (24, 48):
            realized_vol = target_returns.rolling(vol_window).std(ddof=1).replace(0.0, np.nan)
            score = residual_momentum / realized_vol
            for score_threshold in (0.50, 1.00):
                for market_stress_gate in (0.025, 0.050):
                    for max_realized_vol in (0.012, 0.020):
                        volatility_ok = realized_vol.shift(1) <= max_realized_vol
                        long_entry = (
                            (score.shift(1) >= score_threshold)
                            & (market_stress.shift(1) > -market_stress_gate)
                            & volatility_ok
                        )
                        short_entry = (
                            (score.shift(1) <= -score_threshold)
                            & (market_stress.shift(1) < market_stress_gate)
                            & volatility_ok
                        )
                        long_exit = (score < 0.0) | (realized_vol > max_realized_vol * 1.5)
                        short_exit = (score > 0.0) | (realized_vol > max_realized_vol * 1.5)
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
                                sim = _simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                    simulation_backend=simulation_backend,
                                )
                                base = _candidate_base(
                                    family="cross_sectional_vol_adjusted_momentum",
                                    model_parts=(
                                        "xsvamom",
                                        timeframe,
                                        symbol,
                                        f"lb{lookback}",
                                        f"vw{vol_window}",
                                        f"thr{score_threshold}",
                                        f"stress{market_stress_gate}",
                                        f"vol{max_realized_vol}",
                                        f"hold{min_hold}",
                                        f"lev{leverage}",
                                    ),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    side="long_short",
                                    lookback=lookback,
                                    threshold=score_threshold,
                                    exit_threshold=market_stress_gate,
                                    min_hold=min_hold,
                                    leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base["vol_window"] = vol_window
                                base["max_realized_vol"] = max_realized_vol
                                base["market_stress_gate"] = market_stress_gate
                                base["theory_plausibility_gate"] = (
                                    "cross_sectional_residual_momentum_vol_adjusted"
                                )
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


def _cross_sectional_dispersion_gated_momentum_rows(
    *,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    panel: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    frame = bars_by_symbol.get(symbol, pd.DataFrame())
    if frame.empty or symbol not in panel or len(panel.columns) < 3:
        return out
    datetimes = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
    close_panel = panel.reindex(datetimes).ffill()
    target_close = close_panel[symbol]
    target_returns = target_close.pct_change(fill_method=None)
    panel_returns = close_panel.pct_change(fill_method=None)
    for lookback in (12, 24, 48):
        target_momentum = target_close / target_close.shift(lookback) - 1.0
        panel_momentum = close_panel / close_panel.shift(lookback) - 1.0
        market_median_momentum = panel_momentum.median(axis=1, skipna=True)
        residual_momentum = target_momentum - market_median_momentum
        market_stress = panel_returns.median(axis=1, skipna=True).rolling(lookback).sum()
        cross_sectional_dispersion = (
            panel_returns.std(axis=1, skipna=True, ddof=1).rolling(lookback).mean()
        )
        for vol_window in (24, 48):
            realized_vol = target_returns.rolling(vol_window).std(ddof=1).replace(0.0, np.nan)
            score = residual_momentum / realized_vol
            for score_threshold in (0.50, 1.00):
                for dispersion_window in (48, 96):
                    rolling_dispersion_cap = cross_sectional_dispersion.rolling(
                        dispersion_window,
                        min_periods=max(12, dispersion_window // 2),
                    )
                    for max_dispersion_quantile in (0.60, 0.80):
                        dispersion_cap = rolling_dispersion_cap.quantile(
                            max_dispersion_quantile
                        )
                        dispersion_ok = (
                            cross_sectional_dispersion.shift(1) <= dispersion_cap.shift(1)
                        )
                        for market_stress_gate in (0.025, 0.050):
                            stress_ok = market_stress.shift(1) > -market_stress_gate
                            inverse_stress_ok = market_stress.shift(1) < market_stress_gate
                            for max_realized_vol in (0.012, 0.020):
                                volatility_ok = realized_vol.shift(1) <= max_realized_vol
                                long_entry = (
                                    (score.shift(1) >= score_threshold)
                                    & stress_ok
                                    & volatility_ok
                                    & dispersion_ok
                                )
                                short_entry = (
                                    (score.shift(1) <= -score_threshold)
                                    & inverse_stress_ok
                                    & volatility_ok
                                    & dispersion_ok
                                )
                                long_exit = (score < 0.0) | (
                                    realized_vol > max_realized_vol * 1.5
                                )
                                short_exit = (score > 0.0) | (
                                    realized_vol > max_realized_vol * 1.5
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
                                        sim = _simulate_symbol(
                                            frame,
                                            signal,
                                            integer_leverage=int(leverage),
                                            allocation_fraction=allocation_fraction,
                                            simulation_backend=simulation_backend,
                                        )
                                        base = _candidate_base(
                                            family="cross_sectional_dispersion_gated_momentum",
                                            model_parts=(
                                                "xsdispmom",
                                                timeframe,
                                                symbol,
                                                f"lb{lookback}",
                                                f"vw{vol_window}",
                                                f"thr{score_threshold}",
                                                f"dw{dispersion_window}",
                                                f"dq{max_dispersion_quantile}",
                                                f"stress{market_stress_gate}",
                                                f"vol{max_realized_vol}",
                                                f"hold{min_hold}",
                                                f"lev{leverage}",
                                            ),
                                            symbol=symbol,
                                            timeframe=timeframe,
                                            side="long_short",
                                            lookback=lookback,
                                            threshold=score_threshold,
                                            exit_threshold=max_dispersion_quantile,
                                            min_hold=min_hold,
                                            leverage=int(leverage),
                                            allocation_fraction=allocation_fraction,
                                        )
                                        base["vol_window"] = vol_window
                                        base["dispersion_window"] = dispersion_window
                                        base["max_dispersion_quantile"] = (
                                            max_dispersion_quantile
                                        )
                                        base["max_realized_vol"] = max_realized_vol
                                        base["market_stress_gate"] = market_stress_gate
                                        base["indicator_set"] = [
                                            "panel_median_residual_momentum",
                                            "realized_volatility",
                                            "rolling_cross_sectional_return_dispersion",
                                            "market_stress_gate",
                                        ]
                                        base["no_nested_oos_mining"] = True
                                        base["theory_plausibility_gate"] = (
                                            "cross_sectional_dispersion_state_dependent_momentum"
                                        )
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


def _cross_sectional_residual_reversal_rows(
    *,
    bars_by_symbol: Mapping[str, pd.DataFrame],
    panel: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    frame = bars_by_symbol.get(symbol, pd.DataFrame())
    if frame.empty or symbol not in panel or len(panel.columns) < 3:
        return out
    datetimes = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
    close_panel = panel.reindex(datetimes).ffill()
    target_close = close_panel[symbol]
    target_returns = target_close.pct_change(fill_method=None)
    panel_returns = close_panel.pct_change(fill_method=None)
    for lookback in (12, 24, 48):
        target_momentum = target_close / target_close.shift(lookback) - 1.0
        panel_momentum = close_panel / close_panel.shift(lookback) - 1.0
        market_median_momentum = panel_momentum.median(axis=1, skipna=True)
        residual_momentum = target_momentum - market_median_momentum
        residual_z = broad69._rolling_zscore(residual_momentum, lookback * 2)
        trend_z = broad69._rolling_zscore(target_momentum, lookback * 2).abs()
        market_stress = panel_returns.abs().median(axis=1, skipna=True).rolling(lookback).sum()
        for vol_window in (24, 48):
            realized_vol = target_returns.rolling(vol_window).std(ddof=1).replace(0.0, np.nan)
            for residual_z_min in (1.25, 1.75):
                for max_trend_z in (0.75, 1.25):
                    trend_ok = trend_z.shift(1) <= max_trend_z
                    for market_stress_gate in (0.025, 0.050):
                        stress_ok = market_stress.shift(1) <= market_stress_gate
                        for max_realized_vol in (0.012, 0.020):
                            volatility_ok = realized_vol.shift(1) <= max_realized_vol
                            setup_ok = trend_ok & stress_ok & volatility_ok
                            long_entry = (residual_z.shift(1) <= -residual_z_min) & setup_ok
                            short_entry = (residual_z.shift(1) >= residual_z_min) & setup_ok
                            vol_stop = realized_vol > max_realized_vol * 1.5
                            long_exit = (residual_z > -0.20) | vol_stop
                            short_exit = (residual_z < 0.20) | vol_stop
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
                                    sim = _simulate_symbol(
                                        frame,
                                        signal,
                                        integer_leverage=int(leverage),
                                        allocation_fraction=allocation_fraction,
                                        simulation_backend=simulation_backend,
                                    )
                                    base = _candidate_base(
                                        family="cross_sectional_residual_reversal",
                                        model_parts=(
                                            "xsresrev",
                                            timeframe,
                                            symbol,
                                            f"lb{lookback}",
                                            f"vw{vol_window}",
                                            f"z{residual_z_min}",
                                            f"tz{max_trend_z}",
                                            f"stress{market_stress_gate}",
                                            f"vol{max_realized_vol}",
                                            f"hold{min_hold}",
                                            f"lev{leverage}",
                                        ),
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        side="long_short",
                                        lookback=lookback,
                                        threshold=residual_z_min,
                                        exit_threshold=0.20,
                                        min_hold=min_hold,
                                        leverage=int(leverage),
                                        allocation_fraction=allocation_fraction,
                                    )
                                    base["vol_window"] = vol_window
                                    base["max_trend_z"] = max_trend_z
                                    base["market_stress_gate"] = market_stress_gate
                                    base["max_realized_vol"] = max_realized_vol
                                    base["indicator_set"] = [
                                        "panel_median_residual_momentum",
                                        "residual_z",
                                        "own_trend_z_gate",
                                        "realized_volatility",
                                        "market_stress_gate",
                                    ]
                                    base["no_nested_oos_mining"] = True
                                    base["theory_plausibility_gate"] = (
                                        "cross_sectional_residual_stat_arb_reversal"
                                    )
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
    simulation_backend: str | None = None,
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
                            sim = _simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                                simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                            sim = _simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                                simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                            sim = _simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                                simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                    sim = _simulate_symbol(
                        frame,
                        signal,
                        integer_leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                        simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                    sim = _simulate_symbol(
                        frame,
                        signal,
                        integer_leverage=int(leverage),
                        allocation_fraction=allocation_fraction,
                        simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                                sim = _simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                    simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                            sim = _simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                                simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                        sim = _simulate_symbol(
                            frame,
                            signal,
                            integer_leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                            simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
    feature_valid = frame["feature_oi_flow_valid"].fillna(False).astype(bool).reset_index(drop=True)
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
                            sim = _simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                                simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
                                sim = _simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                    simulation_backend=simulation_backend,
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
    simulation_backend: str | None = None,
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
        micro_pressure = pd.concat([flow_smooth, depth_smooth], axis=1).mean(axis=1, skipna=True)
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
                            (~feature_valid) | (micro_pressure < 0.0) | (returns > return_shock_min)
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
                                sim = _simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                    simulation_backend=simulation_backend,
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


def _indicator_vwap_atr_bollinger_reversion_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    volume = frame["volume"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    returns = close.pct_change(fill_method=None)
    true_range_pct = _true_range_pct(frame)
    for lookback in (24, 48):
        vwap = _rolling_vwap(close, volume, lookback)
        atr_pct = true_range_pct.rolling(lookback).mean()
        realized_vol = returns.rolling(lookback).std(ddof=1)
        middle = close.rolling(lookback).mean()
        sigma = close.rolling(lookback).std(ddof=1).replace(0.0, np.nan)
        bollinger_z = (close - middle) / sigma
        vwap_deviation = close / vwap.replace(0.0, np.nan) - 1.0
        vwap_deviation_z = broad69._rolling_zscore(vwap_deviation, lookback * 2)
        for vwap_z_min in (1.25, 1.75):
            for bollinger_z_min in (1.5, 2.0):
                for max_atr_pct in (0.018, 0.028):
                    volatility_ok = (atr_pct.shift(1) <= max_atr_pct) & (
                        realized_vol.shift(1) <= max_atr_pct
                    )
                    long_entry = (
                        (vwap_deviation_z.shift(1) <= -vwap_z_min)
                        & (bollinger_z.shift(1) <= -bollinger_z_min)
                        & volatility_ok
                    )
                    short_entry = (
                        (vwap_deviation_z.shift(1) >= vwap_z_min)
                        & (bollinger_z.shift(1) >= bollinger_z_min)
                        & volatility_ok
                    )
                    long_exit = (
                        (close >= vwap) | (bollinger_z > -0.25) | (atr_pct > max_atr_pct * 1.5)
                    )
                    short_exit = (
                        (close <= vwap) | (bollinger_z < 0.25) | (atr_pct > max_atr_pct * 1.5)
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
                            sim = _simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                                simulation_backend=simulation_backend,
                            )
                            base = _candidate_base(
                                family="indicator_vwap_atr_bollinger_reversion",
                                model_parts=(
                                    "vwapatrbbrev",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"vz{vwap_z_min}",
                                    f"bb{bollinger_z_min}",
                                    f"atr{max_atr_pct}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=vwap_z_min,
                                exit_threshold=bollinger_z_min,
                                min_hold=min_hold,
                                leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base["indicator_set"] = [
                                "rolling_vwap",
                                "atr_pct",
                                "bollinger_z",
                                "realized_volatility",
                                "rolling_standardization",
                            ]
                            base["theory_plausibility_gate"] = "vwap_atr_bollinger_reversion"
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


def _indicator_kalman_volatility_trend_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    log_close = np.log(close.replace(0.0, np.nan))
    returns = close.pct_change(fill_method=None)
    atr_pct = _true_range_pct(frame)
    for lookback in (24, 48):
        realized_vol = returns.rolling(lookback).std(ddof=1)
        for process_noise in (0.0005, 0.002):
            for measurement_noise in (0.02, 0.08):
                filtered = _kalman_log_price_filter(
                    close,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                )
                slope_window = max(3, lookback // 6)
                kalman_slope = filtered - filtered.shift(slope_window)
                slope_z = broad69._rolling_zscore(kalman_slope, lookback * 2)
                deviation_z = broad69._rolling_zscore(log_close - filtered, lookback * 2)
                for slope_z_min in (0.25, 0.50):
                    for max_realized_vol in (0.018, 0.028):
                        volatility_ok = (realized_vol.shift(1) <= max_realized_vol) & (
                            atr_pct.rolling(lookback).mean().shift(1) <= max_realized_vol * 1.5
                        )
                        long_entry = (
                            (slope_z.shift(1) >= slope_z_min)
                            & (deviation_z.shift(1) >= -0.75)
                            & (log_close.shift(1) >= filtered.shift(1))
                            & volatility_ok
                        )
                        short_entry = (
                            (slope_z.shift(1) <= -slope_z_min)
                            & (deviation_z.shift(1) <= 0.75)
                            & (log_close.shift(1) <= filtered.shift(1))
                            & volatility_ok
                        )
                        long_exit = (slope_z < 0.0) | (realized_vol > max_realized_vol * 1.5)
                        short_exit = (slope_z > 0.0) | (realized_vol > max_realized_vol * 1.5)
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
                                sim = _simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                    simulation_backend=simulation_backend,
                                )
                                base = _candidate_base(
                                    family="indicator_kalman_volatility_trend",
                                    model_parts=(
                                        "kalmanvoltrend",
                                        timeframe,
                                        symbol,
                                        f"lb{lookback}",
                                        f"q{process_noise}",
                                        f"r{measurement_noise}",
                                        f"sz{slope_z_min}",
                                        f"vol{max_realized_vol}",
                                        f"hold{min_hold}",
                                        f"lev{leverage}",
                                    ),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    side="long_short",
                                    lookback=lookback,
                                    threshold=slope_z_min,
                                    exit_threshold=max_realized_vol,
                                    min_hold=min_hold,
                                    leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base["indicator_set"] = [
                                    "kalman_log_price_filter",
                                    "kalman_slope_z",
                                    "atr_pct",
                                    "realized_volatility",
                                    "rolling_standardization",
                                ]
                                base["theory_plausibility_gate"] = "kalman_adaptive_trend_filter"
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


def _indicator_kalman_residual_reversion_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    log_close = np.log(close.replace(0.0, np.nan))
    returns = close.pct_change(fill_method=None)
    atr_pct = _true_range_pct(frame)
    for lookback in (24, 48):
        realized_vol = returns.rolling(lookback).std(ddof=1)
        for process_noise in (0.0005, 0.002):
            for measurement_noise in (0.02, 0.08):
                filtered = _kalman_log_price_filter(
                    close,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                )
                slope_window = max(3, lookback // 6)
                kalman_slope = filtered - filtered.shift(slope_window)
                slope_z = broad69._rolling_zscore(kalman_slope, lookback * 2)
                residual_z = broad69._rolling_zscore(log_close - filtered, lookback * 2)
                for residual_z_min in (1.25, 1.75):
                    for max_slope_z in (0.25, 0.50):
                        for max_realized_vol in (0.018, 0.028):
                            volatility_ok = (realized_vol.shift(1) <= max_realized_vol) & (
                                atr_pct.rolling(lookback).mean().shift(1) <= max_realized_vol * 1.5
                            )
                            long_entry = (
                                (residual_z.shift(1) <= -residual_z_min)
                                & (slope_z.shift(1).abs() <= max_slope_z)
                                & volatility_ok
                            )
                            short_entry = (
                                (residual_z.shift(1) >= residual_z_min)
                                & (slope_z.shift(1).abs() <= max_slope_z)
                                & volatility_ok
                            )
                            long_exit = (
                                (residual_z >= -0.15)
                                | (slope_z < -max_slope_z)
                                | (realized_vol > max_realized_vol * 1.5)
                            )
                            short_exit = (
                                (residual_z <= 0.15)
                                | (slope_z > max_slope_z)
                                | (realized_vol > max_realized_vol * 1.5)
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
                                    sim = _simulate_symbol(
                                        frame,
                                        signal,
                                        integer_leverage=int(leverage),
                                        allocation_fraction=allocation_fraction,
                                        simulation_backend=simulation_backend,
                                    )
                                    base = _candidate_base(
                                        family="indicator_kalman_residual_reversion",
                                        model_parts=(
                                            "kalmanresrev",
                                            timeframe,
                                            symbol,
                                            f"lb{lookback}",
                                            f"q{process_noise}",
                                            f"r{measurement_noise}",
                                            f"rz{residual_z_min}",
                                            f"szcap{max_slope_z}",
                                            f"vol{max_realized_vol}",
                                            f"hold{min_hold}",
                                            f"lev{leverage}",
                                        ),
                                        symbol=symbol,
                                        timeframe=timeframe,
                                        side="long_short",
                                        lookback=lookback,
                                        threshold=residual_z_min,
                                        exit_threshold=max_slope_z,
                                        min_hold=min_hold,
                                        leverage=int(leverage),
                                        allocation_fraction=allocation_fraction,
                                    )
                                    base["indicator_set"] = [
                                        "kalman_log_price_filter",
                                        "kalman_residual_z",
                                        "kalman_slope_z",
                                        "atr_pct",
                                        "realized_volatility",
                                        "rolling_standardization",
                                    ]
                                    base["theory_plausibility_gate"] = (
                                        "kalman_adaptive_residual_reversion"
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


def _indicator_vwap_kalman_pullback_continuation_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    volume = frame["volume"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    log_close = np.log(close.replace(0.0, np.nan))
    returns = close.pct_change(fill_method=None)
    true_range_pct = _true_range_pct(frame)
    for lookback in (48, 96):
        vwap = _rolling_vwap(close, volume, lookback)
        atr_pct = true_range_pct.rolling(lookback).mean()
        realized_vol = returns.rolling(lookback).std(ddof=1)
        middle = close.rolling(lookback).mean()
        sigma = close.rolling(lookback).std(ddof=1).replace(0.0, np.nan)
        bollinger_z = (close - middle) / sigma
        vwap_deviation = close / vwap.replace(0.0, np.nan) - 1.0
        vwap_deviation_z = broad69._rolling_zscore(vwap_deviation, lookback * 2)
        atr_distance = (close - vwap).abs() / (
            close.replace(0.0, np.nan) * atr_pct.replace(0.0, np.nan)
        )
        for process_noise in (0.0005, 0.002):
            for measurement_noise in (0.02, 0.08):
                filtered = _kalman_log_price_filter(
                    close,
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                )
                slope_window = max(6, lookback // 8)
                kalman_slope = filtered - filtered.shift(slope_window)
                slope_z = broad69._rolling_zscore(kalman_slope, lookback * 2)
                for trend_slope_z in (0.25, 0.50):
                    for pullback_z in (0.25, 0.75):
                        for max_atr_distance in (1.5, 2.5):
                            for max_realized_vol in (0.018, 0.028):
                                volatility_ok = (
                                    (realized_vol.shift(1) <= max_realized_vol)
                                    & (atr_pct.shift(1) <= max_realized_vol * 1.5)
                                    & (atr_distance.shift(1) <= max_atr_distance)
                                )
                                long_entry = (
                                    (slope_z.shift(1) >= trend_slope_z)
                                    & (log_close.shift(1) >= filtered.shift(1))
                                    & (
                                        (vwap_deviation_z.shift(1) <= -pullback_z)
                                        | (bollinger_z.shift(1) <= -pullback_z)
                                    )
                                    & volatility_ok
                                )
                                short_entry = (
                                    (slope_z.shift(1) <= -trend_slope_z)
                                    & (log_close.shift(1) <= filtered.shift(1))
                                    & (
                                        (vwap_deviation_z.shift(1) >= pullback_z)
                                        | (bollinger_z.shift(1) >= pullback_z)
                                    )
                                    & volatility_ok
                                )
                                long_exit = (
                                    (slope_z < 0.0)
                                    | ((close >= vwap) & (bollinger_z >= 0.25))
                                    | (realized_vol > max_realized_vol * 1.5)
                                )
                                short_exit = (
                                    (slope_z > 0.0)
                                    | ((close <= vwap) & (bollinger_z <= -0.25))
                                    | (realized_vol > max_realized_vol * 1.5)
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
                                        sim = _simulate_symbol(
                                            frame,
                                            signal,
                                            integer_leverage=int(leverage),
                                            allocation_fraction=allocation_fraction,
                                            simulation_backend=simulation_backend,
                                        )
                                        base = _candidate_base(
                                            family="indicator_vwap_kalman_pullback_continuation",
                                            model_parts=(
                                                "vwapkalmanpullback",
                                                timeframe,
                                                symbol,
                                                f"lb{lookback}",
                                                f"q{process_noise}",
                                                f"r{measurement_noise}",
                                                f"trend{trend_slope_z}",
                                                f"pb{pullback_z}",
                                                f"atrx{max_atr_distance}",
                                                f"vol{max_realized_vol}",
                                                f"hold{min_hold}",
                                                f"lev{leverage}",
                                            ),
                                            symbol=symbol,
                                            timeframe=timeframe,
                                            side="long_short",
                                            lookback=lookback,
                                            threshold=trend_slope_z,
                                            exit_threshold=pullback_z,
                                            min_hold=min_hold,
                                            leverage=int(leverage),
                                            allocation_fraction=allocation_fraction,
                                        )
                                        base["indicator_set"] = [
                                            "rolling_vwap",
                                            "kalman_log_price_filter",
                                            "kalman_slope_z",
                                            "bollinger_z",
                                            "atr_pct",
                                            "atr_distance_to_vwap",
                                            "realized_volatility",
                                            "rolling_standardization",
                                        ]
                                        base["no_nested_oos_mining"] = True
                                        base["theory_plausibility_gate"] = (
                                            "vwap_kalman_pullback_continuation"
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


def _standardized_indicator_ridge_directional_rows(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold: monthly.MonthlyFold,
    leverages: Sequence[int],
    allocation_fraction: float,
    simulation_backend: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if frame.empty:
        return out
    close = frame["close"].astype(float).reset_index(drop=True)
    volume = frame["volume"].astype(float).reset_index(drop=True)
    datetimes = frame["datetime"]
    train_mask = _window_mask(datetimes, fold.train)
    returns = close.pct_change(fill_method=None)
    next_return = close.shift(-1) / close.replace(0.0, np.nan) - 1.0
    true_range_pct = _true_range_pct(frame)
    for lookback in (24, 48):
        vwap = _rolling_vwap(close, volume, lookback)
        middle = close.rolling(lookback).mean()
        sigma = close.rolling(lookback).std(ddof=1).replace(0.0, np.nan)
        filtered = _kalman_log_price_filter(
            close,
            process_noise=0.001,
            measurement_noise=0.04,
        )
        raw_features = pd.DataFrame(
            {
                "return_1": returns,
                "momentum": close / close.shift(lookback) - 1.0,
                "realized_vol": returns.rolling(lookback).std(ddof=1),
                "atr_pct": true_range_pct.rolling(lookback).mean(),
                "bollinger_z": (close - middle) / sigma,
                "vwap_deviation": close / vwap.replace(0.0, np.nan) - 1.0,
                "kalman_slope": filtered - filtered.shift(max(3, lookback // 6)),
                "volume_z": broad69._rolling_zscore(volume, lookback * 2),
            }
        ).shift(1)
        standardized = _train_standardized_frame(raw_features, train_mask)
        for ridge_alpha in (1.0, 10.0):
            score = _ridge_directional_score(
                standardized,
                next_return,
                train_mask,
                ridge_alpha=ridge_alpha,
                min_train_observations=240,
            )
            if score is None:
                continue
            for score_threshold in (0.0005, 0.0010):
                long_entry = score >= score_threshold
                short_entry = score <= -score_threshold
                long_exit = score <= 0.0
                short_exit = score >= 0.0
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
                        sim = _simulate_symbol(
                            frame,
                            signal,
                            integer_leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                            simulation_backend=simulation_backend,
                        )
                        base = _candidate_base(
                            family="standardized_indicator_ridge_directional",
                            model_parts=(
                                "stdridge",
                                timeframe,
                                symbol,
                                f"lb{lookback}",
                                f"alpha{ridge_alpha}",
                                f"thr{score_threshold}",
                                f"hold{min_hold}",
                                f"lev{leverage}",
                            ),
                            symbol=symbol,
                            timeframe=timeframe,
                            side="long_short",
                            lookback=lookback,
                            threshold=score_threshold,
                            exit_threshold=0.0,
                            min_hold=min_hold,
                            leverage=int(leverage),
                            allocation_fraction=allocation_fraction,
                        )
                        base["uses_ml"] = True
                        base["ml_model"] = "bounded_ridge_directional_score"
                        base["ml_fit_scope"] = "train_only"
                        base["standardization_scope"] = "train_only"
                        base["indicator_set"] = [
                            "returns",
                            "vwap_deviation",
                            "bollinger_z",
                            "atr_pct",
                            "realized_volatility",
                            "kalman_slope",
                            "volume_z",
                        ]
                        base["no_nested_oos_mining"] = True
                        base["theory_plausibility_gate"] = "standardized_indicator_ml_leaf"
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
    selection_policy: str = DEFAULT_SELECTION_POLICY,
    enabled_families: Sequence[str] | None = None,
    leverages: Sequence[int] | None = DEFAULT_LEVERAGES,
    simulation_backend: str | None = DEFAULT_SIMULATION_BACKEND,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    enabled = set(_normalize_enabled_families(enabled_families))
    actual_leverages = _normalize_leverages(leverages)
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
                "leverages": actual_leverages,
                "allocation_fraction": DEFAULT_ALLOCATION_FRACTION,
                "simulation_backend": simulation_backend,
            }
            if "volatility_squeeze_breakout" in enabled:
                rows.extend(_squeeze_rows(**kwargs))
            if "volume_absorption_reversal" in enabled:
                rows.extend(_absorption_rows(**kwargs))
            if "range_reclaim_continuation" in enabled:
                rows.extend(_reclaim_rows(**kwargs))
            if "indicator_vwap_atr_bollinger_reversion" in enabled:
                rows.extend(_indicator_vwap_atr_bollinger_reversion_rows(**kwargs))
            if "indicator_kalman_volatility_trend" in enabled:
                rows.extend(_indicator_kalman_volatility_trend_rows(**kwargs))
            if "indicator_kalman_residual_reversion" in enabled:
                rows.extend(_indicator_kalman_residual_reversion_rows(**kwargs))
            if "indicator_vwap_kalman_pullback_continuation" in enabled:
                rows.extend(_indicator_vwap_kalman_pullback_continuation_rows(**kwargs))
            if "standardized_indicator_ridge_directional" in enabled:
                rows.extend(_standardized_indicator_ridge_directional_rows(**kwargs))
            if "deep_research_vol_managed_momentum_crash_gate" in enabled:
                rows.extend(
                    _deep_research_vol_managed_momentum_crash_gate_rows(
                        **kwargs,
                        bars_by_symbol=bars_by_symbol,
                    )
                )
            features = (features_by_symbol or {}).get(symbol, pd.DataFrame())
            if not features.empty:
                feature_frame = _attach_feature_points(frame, features, timeframe=timeframe)
                feature_kwargs = {**kwargs, "frame": feature_frame}
                if "feature_flow_crowding_reversal" in enabled:
                    rows.extend(_feature_flow_rows(**feature_kwargs))
                if "feature_liquidation_imbalance_reversal" in enabled:
                    rows.extend(_feature_liquidation_rows(**feature_kwargs))
                if "feature_flow_oi_trend_continuation" in enabled:
                    rows.extend(_feature_flow_trend_rows(**feature_kwargs))
                if "funding_oi_taker_crowding_continuation" in enabled:
                    rows.extend(_feature_crowding_continuation_rows(**feature_kwargs))
                if "perp_crowding_score_reversion" in enabled:
                    rows.extend(_perp_crowding_rows(**feature_kwargs))
                if "feature_taker_flow_exhaustion_reversal" in enabled:
                    rows.extend(_feature_taker_flow_exhaustion_rows(**feature_kwargs))
                if "feature_bbo_flow_exhaustion_reversal" in enabled:
                    rows.extend(_feature_bbo_flow_rows(**feature_kwargs))
                if "feature_book_depth_imbalance_reversal" in enabled:
                    rows.extend(_feature_book_depth_rows(**feature_kwargs))
                if "deep_research_funding_dislocation_trend_carry" in enabled:
                    rows.extend(
                        _deep_research_funding_dislocation_trend_carry_rows(**feature_kwargs)
                    )
                if "deep_research_flow_imbalance_liquidation_sweep" in enabled:
                    rows.extend(
                        _deep_research_flow_imbalance_liquidation_sweep_rows(**feature_kwargs)
                    )
            if not panel.empty:
                if "cross_asset_lead_lag_momentum" in enabled:
                    rows.extend(
                        _lead_lag_rows(
                            bars_by_symbol=bars_by_symbol,
                            panel=panel,
                            symbol=symbol,
                            timeframe=timeframe,
                            fold=fold,
                            leverages=actual_leverages,
                            allocation_fraction=DEFAULT_ALLOCATION_FRACTION,
                            simulation_backend=simulation_backend,
                        )
                    )
                if "btc_beta_residual_momentum" in enabled:
                    rows.extend(
                        _btc_beta_residual_momentum_rows(
                            bars_by_symbol=bars_by_symbol,
                            panel=panel,
                            symbol=symbol,
                            timeframe=timeframe,
                            fold=fold,
                            leverages=actual_leverages,
                            allocation_fraction=DEFAULT_ALLOCATION_FRACTION,
                            simulation_backend=simulation_backend,
                        )
                    )
                if "cross_sectional_vol_adjusted_momentum" in enabled:
                    rows.extend(
                        _cross_sectional_vol_adjusted_momentum_rows(
                            bars_by_symbol=bars_by_symbol,
                            panel=panel,
                            symbol=symbol,
                            timeframe=timeframe,
                            fold=fold,
                            leverages=actual_leverages,
                            allocation_fraction=DEFAULT_ALLOCATION_FRACTION,
                            simulation_backend=simulation_backend,
                        )
                    )
                if "cross_sectional_dispersion_gated_momentum" in enabled:
                    rows.extend(
                        _cross_sectional_dispersion_gated_momentum_rows(
                            bars_by_symbol=bars_by_symbol,
                            panel=panel,
                            symbol=symbol,
                            timeframe=timeframe,
                            fold=fold,
                            leverages=actual_leverages,
                            allocation_fraction=DEFAULT_ALLOCATION_FRACTION,
                            simulation_backend=simulation_backend,
                        )
                    )
                if "cross_sectional_residual_reversal" in enabled:
                    rows.extend(
                        _cross_sectional_residual_reversal_rows(
                            bars_by_symbol=bars_by_symbol,
                            panel=panel,
                            symbol=symbol,
                            timeframe=timeframe,
                            fold=fold,
                            leverages=actual_leverages,
                            allocation_fraction=DEFAULT_ALLOCATION_FRACTION,
                            simulation_backend=simulation_backend,
                        )
                    )
    return _cap_rows_for_selection(
        rows,
        max_candidates_per_fold=max_candidates_per_fold,
        selection_policy=selection_policy,
    )


def _select_fold_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_policy: str = DEFAULT_SELECTION_POLICY,
) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if _eligible_for_policy(row, selection_policy=selection_policy)]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            _selection_score(row, selection_policy=selection_policy),
            str(row.get("model_id")),
        ),
    )


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


def _realism_diagnostics(
    selected_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    *,
    selection_policy: str,
) -> dict[str, Any]:
    validation_returns = [_safe_float(row.get("validation_return")) for row in selected_rows]
    locked_oos_returns = [
        _safe_float(row.get("locked_oos_return_report_only")) for row in selected_rows
    ]
    validation_trade_counts = [
        int(row.get("validation_trade_event_count") or 0) for row in selected_rows
    ]
    validation_sharpes = [_safe_float(row.get("validation_sharpe")) for row in selected_rows]
    row_blockers = sorted(
        {
            str(blocker)
            for row in selected_rows
            for blocker in (row.get("label_blockers") or [])
            if str(blocker)
        }
    )
    reasons: list[str] = []
    if not selected_rows:
        reasons.append("no_selected_fold_rows")
    if selection_policy == ROBUST_SELECTION_POLICY:
        reasons.append("robust_selector_is_post_failure_diagnostic_requires_fresh_forward")
    if row_blockers:
        reasons.extend(row_blockers)
    if selected_rows and not all(bool(row.get("ready_for_real")) for row in selected_rows):
        reasons.append("selected_rows_not_ready_for_real_money")
    if selected_rows and any(
        bool(row.get("uses_continuous_position_state_across_split_boundaries"))
        for row in selected_rows
    ):
        reasons.append("continuous_position_state_split_simulation_not_live_equivalent")
    if validation_trade_counts and min(validation_trade_counts) < 30:
        reasons.append("some_validation_samples_below_30_trade_events")
    mean_validation_return = float(np.mean(validation_returns)) if validation_returns else 0.0
    mean_locked_oos_return = float(np.mean(locked_oos_returns)) if locked_oos_returns else 0.0
    if mean_validation_return > 0.10 and mean_locked_oos_return < mean_validation_return * 0.50:
        reasons.append("validation_to_locked_oos_decay_large")
    if validation_sharpes and max(validation_sharpes) > 5.0:
        reasons.append(
            "validation_sharpe_too_high_for_live_assumption_without_forward_fill_telemetry"
        )
    if _safe_float(aggregate.get("annualized_oos_return_approx")) > 1.0:
        reasons.append("hundred_pct_plus_oos_label_needs_independent_pre_registered_retest")
    live_plausibility = "not_supported" if reasons else "needs_shadow_confirmation"
    return {
        "live_performance_plausibility": live_plausibility,
        "real_money_execution": False,
        "ready_for_real": False,
        "selection_policy": selection_policy,
        "selected_fold_count": len(selected_rows),
        "mean_validation_return": mean_validation_return,
        "mean_locked_oos_return_report_only": mean_locked_oos_return,
        "locked_oos_compounded_return_report_only": _safe_float(
            aggregate.get("compounded_oos_return")
        ),
        "locked_oos_annualized_return_report_only": _safe_float(
            aggregate.get("annualized_oos_return_approx")
        ),
        "locked_oos_monthly_equity_mdd_report_only": _safe_float(
            aggregate.get("monthly_equity_mdd")
        ),
        "positive_locked_oos_fold_share": (
            sum(1 for value in locked_oos_returns if value > 0.0) / len(locked_oos_returns)
            if locked_oos_returns
            else 0.0
        ),
        "min_validation_trade_event_count": min(validation_trade_counts)
        if validation_trade_counts
        else 0,
        "max_validation_sharpe": max(validation_sharpes) if validation_sharpes else 0.0,
        "blockers": sorted(set(reasons)),
        "external_priors_reflected": [
            "time_series_and_cross_sectional_momentum",
            "volatility_managed_momentum_and_crash_gates",
            "lead_lag_information_flow",
            "btc_eth_beta_residual_momentum",
            "kalman_filter_residual_reversion",
            "cross_sectional_vol_adjusted_residual_momentum",
            "perp_funding_open_interest_taker_flow_crowding",
            "order_book_spread_depth_imbalance_microstructure",
            "train_only_standardized_indicator_ridge_leaf",
        ],
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
    selection_policy: str = DEFAULT_SELECTION_POLICY,
    enabled_families: Sequence[str] | None = None,
    fold_workers: int = DEFAULT_FOLD_WORKERS,
    leverages: Sequence[int] | None = DEFAULT_LEVERAGES,
    max_candidate_rows_output: int | None = None,
    simulation_backend: str | None = DEFAULT_SIMULATION_BACKEND,
) -> dict[str, Any]:
    if selection_policy not in SELECTION_POLICIES:
        raise ValueError(f"unknown selection policy: {selection_policy}")
    enabled_families_normalized = _normalize_enabled_families(enabled_families)
    actual_leverages = _normalize_leverages(leverages)
    fold_workers = max(1, int(fold_workers))
    normalized_simulation_backend = native_alpha_fold_backend.normalize_alpha_fold_backend(
        simulation_backend
    )
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
    fold_batches: list[tuple[monthly.MonthlyFold, list[dict[str, Any]]]] = []

    def rows_for(fold: monthly.MonthlyFold) -> list[dict[str, Any]]:
        return _rows_for_fold(
            bars=bars,
            symbols=symbols,
            timeframes=timeframes,
            features_by_symbol=features_by_symbol,
            fold=fold,
            max_candidates_per_fold=max_candidates_per_fold,
            selection_policy=selection_policy,
            enabled_families=enabled_families_normalized,
            leverages=actual_leverages,
            simulation_backend=normalized_simulation_backend,
        )

    if fold_workers == 1 or len(folds) <= 1:
        fold_batches = [(fold, rows_for(fold)) for fold in folds]
    else:
        fold_batches_buffer: list[tuple[monthly.MonthlyFold, list[dict[str, Any]]] | None] = [
            None
        ] * len(folds)
        with ThreadPoolExecutor(max_workers=min(fold_workers, len(folds))) as executor:
            future_to_index = {
                executor.submit(rows_for, fold): index for index, fold in enumerate(folds)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                fold_batches_buffer[index] = (folds[index], future.result())
        fold_batches = [batch for batch in fold_batches_buffer if batch is not None]

    for fold, rows in fold_batches:
        selected = _select_fold_candidate(rows, selection_policy=selection_policy)
        for row in rows:
            row["selection_policy"] = selection_policy
            row["selection_score_active_train_validation_only"] = _selection_score(
                row,
                selection_policy=selection_policy,
            )
            row["robust_train_validation_v1_eligible"] = _robust_v1_eligible(row)
            row["selection_score_robust_v1_train_validation_only"] = _robust_v1_score_row(row)
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
                        "score": row.get("selection_score_active_train_validation_only"),
                        "selection_policy": selection_policy,
                        "enabled_families": enabled_families_normalized,
                        "leverages": actual_leverages,
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
            selected_payload["selection_policy"] = selection_policy
            selected_payload["selection_score_active_train_validation_only"] = _selection_score(
                selected_payload,
                selection_policy=selection_policy,
            )
            selected_payload["robust_train_validation_v1_eligible"] = _robust_v1_eligible(
                selected_payload
            )
            selected_payload["selection_score_robust_v1_train_validation_only"] = (
                _robust_v1_score_row(selected_payload)
            )
            selected_rows.append(selected_payload)

    aggregate = _aggregate(selected_rows)
    candidate_row_count_total = len(fold_rows)
    if max_candidate_rows_output is None or int(max_candidate_rows_output) < 0:
        candidate_rows_for_output = fold_rows
    else:
        candidate_rows_for_output = fold_rows[: int(max_candidate_rows_output)]
    payload = {
        "artifact_kind": "alpha_zoo_clean_new_alpha_discovery",
        "generated_at_utc": _utc_now_iso(),
        "search_space": search_space,
        "pre_registered_search_space_sha256": search_hash,
        "selection_policy": selection_policy,
        "enabled_families": list(enabled_families_normalized),
        "integer_leverages": list(actual_leverages),
        "fold_workers": fold_workers,
        "simulation_backend": native_alpha_fold_backend.alpha_fold_backend_diagnostics(
            normalized_simulation_backend
        ),
        "candidate_cap_sort_policy": "eligible_first_active_train_validation_selection_score",
        "candidate_row_count_total": candidate_row_count_total,
        "candidate_rows_truncated": len(candidate_rows_for_output) < candidate_row_count_total,
        "max_candidate_rows_output": max_candidate_rows_output,
        "optimization_policy": optimization_search_policy_payload(
            search_method="bounded_grid_pre_registered_new_alpha",
            objective_policy="rank_train_validation_only_then_attach_locked_oos_report_gate",
            selection_inputs=["train", "validation"],
            bounded_grid_justification=(
                "small deterministic OHLCV new-alpha grid; no post-OOS selector, "
                "lagged-shadow, nested, or locked-OOS objective inputs"
            ),
            extra={
                "selection_policy": selection_policy,
                "enabled_families": list(enabled_families_normalized),
                "integer_leverages": list(actual_leverages),
                "fold_workers": fold_workers,
                "simulation_backend": native_alpha_fold_backend.alpha_fold_backend_diagnostics(
                    normalized_simulation_backend
                ),
                "candidate_row_count_total": candidate_row_count_total,
                "max_candidate_rows_output": max_candidate_rows_output,
                "candidate_cap_sort_policy": (
                    "eligible_first_active_train_validation_selection_score"
                ),
                "selection_policy_status": (
                    "post_failure_research_variant_requires_fresh_forward"
                    if selection_policy == ROBUST_SELECTION_POLICY
                    else "pre_existing_default_policy"
                ),
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
        "candidate_rows": candidate_rows_for_output,
        "aggregate": aggregate,
        "realism_diagnostics": _realism_diagnostics(
            selected_rows,
            aggregate,
            selection_policy=selection_policy,
        ),
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
        f"- selection policy: `{payload.get('selection_policy', DEFAULT_SELECTION_POLICY)}`",
        f"- enabled families: `{len(payload.get('enabled_families') or [])}`",
        f"- integer leverages: `{payload.get('integer_leverages', list(DEFAULT_LEVERAGES))}`",
        f"- fold workers: `{int(payload.get('fold_workers') or 1)}`",
        "- simulation backend: "
        f"`{(payload.get('simulation_backend') or {}).get('resolved_backend', 'unknown')}`",
        f"- candidate cap sort: `{payload.get('candidate_cap_sort_policy', 'legacy_score')}`",
        f"- candidate rows retained/written: `{payload.get('candidate_row_count_total', 0)}`/"
        f"`{len(payload.get('candidate_rows') or [])}`",
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
        "## Live realism diagnostics",
        "",
    ]
    realism = payload.get("realism_diagnostics") or {}
    blockers = realism.get("blockers") or []
    lines.extend(
        [
            f"- live plausibility: `{realism.get('live_performance_plausibility', 'unknown')}`",
            f"- mean validation return: `{_fmt_pct(realism.get('mean_validation_return'))}`",
            f"- mean locked-OOS return: `{_fmt_pct(realism.get('mean_locked_oos_return_report_only'))}`",
            f"- positive locked-OOS fold share: `{_safe_float(realism.get('positive_locked_oos_fold_share')):.2f}`",
            f"- min validation trade events: `{int(realism.get('min_validation_trade_event_count') or 0)}`",
            f"- max validation Sharpe: `{_safe_float(realism.get('max_validation_sharpe')):.2f}`",
            "- blockers: "
            + (", ".join(f"`{blocker}`" for blocker in blockers) if blockers else "`none`"),
            "",
        ]
    )
    lines.extend(
        [
            "## Fold selections",
            "",
            "| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
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
    parser.add_argument("--max-candidates-per-fold", type=int, default=10000)
    parser.add_argument(
        "--leverages",
        default=",".join(str(value) for value in DEFAULT_LEVERAGES),
        help="Comma-separated positive integer leverages. Use one value for faster probes.",
    )
    parser.add_argument(
        "--families",
        default="",
        help=(
            "Comma-separated alpha families to run for faster fold probes. "
            "Default empty value runs every registered family."
        ),
    )
    parser.add_argument(
        "--fold-workers",
        type=int,
        default=DEFAULT_FOLD_WORKERS,
        help="Parallel fold workers for candidate generation. 1 preserves serial execution.",
    )
    parser.add_argument(
        "--simulation-backend",
        choices=(
            native_alpha_fold_backend.ALPHA_FOLD_BACKEND_AUTO,
            native_alpha_fold_backend.ALPHA_FOLD_BACKEND_PYTHON,
            native_alpha_fold_backend.ALPHA_FOLD_BACKEND_RUST,
        ),
        default=DEFAULT_SIMULATION_BACKEND,
        help="Symbol simulation backend for fold candidate loops. auto uses Rust when built.",
    )
    parser.add_argument(
        "--max-candidate-rows-output",
        type=int,
        default=-1,
        help=(
            "Limit candidate_rows serialized into JSON. Use 0 for fast probe artifacts; "
            "-1 keeps all retained rows."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default=DEFAULT_SELECTION_POLICY,
        help=(
            "Train/validation-only fold selector. robust_train_validation_v1 is a "
            "post-failure research policy and must require fresh-forward before promotion."
        ),
    )
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
        selection_policy=str(args.selection_policy),
        enabled_families=tuple(
            item.strip() for item in str(args.families).split(",") if item.strip()
        )
        or None,
        fold_workers=int(args.fold_workers),
        leverages=tuple(
            int(item.strip()) for item in str(args.leverages).split(",") if item.strip()
        ),
        max_candidate_rows_output=int(args.max_candidate_rows_output),
        simulation_backend=str(args.simulation_backend),
    )
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
