#!/usr/bin/env python3
"""Research-only StateDistilledRegimeBoostPortfolio runner.

This script intentionally stays in the research/artifact lane.  It overlays the
existing state-distilled candidate hypotheses with explicitly tunable regime,
side-bias, volatility-targeted leverage, conditional booster, and neutral-pair
rules.  Selection is train/validation only; locked-OOS is opened after a frozen
config sidecar hash exists.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import resource
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

RUN_ID = "state_distilled_regime_boost_20260513"
DEFAULT_PANEL_PARQUET = Path(
    "var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet"
)
DEFAULT_EXTERNAL_STATE_CSV = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/external_market_state_20260512/"
    "external_market_state_lagged.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513"
)
DEFAULT_CANDIDATE_ARTIFACTS = (
    Path(
        "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
        "liquidation_aware_state_distilled_external_risk_filter_20260512/liquidation_aware_current_base_latest.json"
    ),
    Path(
        "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
        "liquidation_aware_state_distilled_20260511/liquidation_aware_current_base_latest.json"
    ),
)
CORE_A_CANDIDATE = "fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125"
CORE_B_CANDIDATE = "fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600"
CALENDAR_FIELDS = frozenset({"month", "day", "weekday", "hour", "minute", "session_hour"})
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT")
ALT_SYMBOLS = tuple(sym for sym in SYMBOLS if sym != "BTCUSDT")
STARTING_EQUITY = 10_000.0
ANNUALIZATION_HOURS = 365.0 * 24.0


@dataclass(frozen=True, slots=True)
class RegimeClassifierConfig:
    trend_bull_threshold: float = 0.75
    trend_bear_threshold: float = -0.75
    risk_bear_threshold: float = 1.00
    risk_stress_threshold: float = 1.50
    dispersion_threshold: float = 0.75
    vol_ratio_stress_threshold: float = 1.75
    btc_drawdown_stress_threshold: float = -0.12
    btc_trend_weight: float = 0.35
    eth_trend_weight: float = 0.25
    breadth_weight: float = 0.20
    vol_penalty_weight: float = 0.20
    vix_risk_weight: float = 0.35
    usd_risk_weight: float = 0.25
    curve_risk_weight: float = 0.20
    oil_risk_weight: float = 0.20


@dataclass(frozen=True, slots=True)
class SideBiasConfig:
    bull_long_multiplier: float = 1.25
    bull_short_multiplier: float = 0.60
    bear_long_multiplier: float = 0.45
    bear_short_multiplier: float = 1.30
    neutral_high_long_multiplier: float = 1.00
    neutral_high_short_multiplier: float = 1.00
    neutral_low_long_multiplier: float = 0.50
    neutral_low_short_multiplier: float = 0.50
    stress_long_multiplier: float = 0.00
    stress_short_multiplier: float = 0.00


@dataclass(frozen=True, slots=True)
class VolTargetLeverageConfig:
    base_leverage: float = 4.0
    min_effective_leverage: float = 1.0
    max_effective_leverage: float = 25.0
    target_annual_volatility: float = 2.00
    long_term_vol_floor: float = 0.02
    high_vol_ratio_threshold: float = 1.50
    medium_vol_ratio_threshold: float = 1.25
    low_vol_ratio_threshold: float = 0.85
    high_confidence_threshold: float = 0.80
    medium_confidence_threshold: float = 0.65
    high_vol_leverage: float = 2.0
    medium_vol_leverage: float = 3.0
    medium_confidence_leverage: float = 4.5
    stress_leverage: float = 1.0


@dataclass(frozen=True, slots=True)
class BoosterConfig:
    enabled: bool = True
    allocation_weight: float = 0.10
    max_leverage: float = 25.0
    min_confidence: float = 0.80
    max_vol_ratio: float = 0.85
    min_margin_safety_score: float = 0.60
    allowed_regimes: tuple[str, ...] = ("bull", "bear", "neutral_high_dispersion")


@dataclass(frozen=True, slots=True)
class NeutralPairOverlayConfig:
    enabled: bool = True
    allocation_weight: float = 0.10
    max_pair_leverage: float = 4.0
    min_rank_gap: float = 0.35
    hedge_ratio_floor: float = 0.25
    hedge_ratio_ceiling: float = 2.50
    fit_splits: tuple[str, ...] = ("train", "validation")


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    grid_limit: int = 64
    hard_grid_cap: int = 256
    train_return_weight: float = 0.35
    validation_return_weight: float = 0.65
    sharpe_weight: float = 0.08
    drawdown_penalty: float = 0.35
    liquidation_penalty: float = 100.0
    min_validation_return: float = -1.0
    strict_oos_mdd_max: float = 0.25


@dataclass(frozen=True, slots=True)
class MarginConfig:
    maintenance_margin_rate: float = 0.01
    liquidation_fee_rate: float = 0.005
    taker_fee_rate: float = 0.0004

    @property
    def liquidation_reserve_rate(self) -> float:
        return self.maintenance_margin_rate + self.liquidation_fee_rate + self.taker_fee_rate


@dataclass(frozen=True, slots=True)
class RegimeBoostConfig:
    regime: RegimeClassifierConfig = field(default_factory=RegimeClassifierConfig)
    side_bias: SideBiasConfig = field(default_factory=SideBiasConfig)
    leverage: VolTargetLeverageConfig = field(default_factory=VolTargetLeverageConfig)
    booster: BoosterConfig = field(default_factory=BoosterConfig)
    neutral_pair: NeutralPairOverlayConfig = field(default_factory=NeutralPairOverlayConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)
    long_entry_threshold: float = 0.75
    short_entry_threshold: float = -0.75
    core_a_weight: float = 0.45
    core_b_weight: float = 0.45
    rebalance_stride_hours: int = 24
    signal_fast_weight: float = 0.35
    signal_residual_weight: float = 0.65
    external_risk_downshift: float = 0.50
    external_risk_short_boost: float = 1.15


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(obj), sort_keys=True))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode()


def stable_json_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KiB, macOS returns bytes. This environment is Linux, but keep
    # the conversion defensive.
    return int(value if value > 10_000_000_000 else value * 1024)


def git_state() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    except Exception as exc:  # pragma: no cover - defensive for non-git smoke use
        return {"commit": "unknown", "dirty_tree": True, "dirty_files": [f"git_error:{exc}"]}
    dirty_files = [line[3:] if len(line) > 3 else line for line in status.splitlines()]
    return {"commit": sha, "dirty_tree": bool(dirty_files), "dirty_files": dirty_files}


def validate_strategy_card(card: dict[str, Any]) -> dict[str, Any]:
    fields = {str(item) for item in card.get("feature_fields", ())}
    calendar_hits = sorted(fields & CALENDAR_FIELDS)
    rejection_reasons: list[str] = []
    if bool(card.get("calendar_primary")):
        rejection_reasons.append("calendar_primary_true")
    if calendar_hits:
        rejection_reasons.append("calendar_fields_present")
    if bool(card.get("uses_locked_oos_for_selection")):
        rejection_reasons.append("locked_oos_used_for_selection")
    if not card.get("source_coverage"):
        rejection_reasons.append("missing_source_coverage")
    out = dict(card)
    out.update(
        {
            "calendar_fields_detected": calendar_hits,
            "strategy_valid": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
            "fail_closed": bool(rejection_reasons),
        }
    )
    return out


def base_strategy_validity_card() -> dict[str, Any]:
    return validate_strategy_card(
        {
            "strategy_name": "StateDistilledRegimeBoostPortfolio",
            "calendar_primary": False,
            "feature_fields": [
                "btc_ret_168h",
                "eth_ret_168h",
                "btc_ret_720h",
                "alt_breadth",
                "alt_dispersion",
                "realized_vol_72h",
                "realized_vol_720h",
                "btc_drawdown_168h",
                "external_risk_off_score_lag1",
                "external_usd_ret_z_lag1",
                "external_vix_z_lag1",
                "external_curve_z_lag1",
                "external_wti_ret_z_lag1",
            ],
            "source_coverage": {
                "crypto_current_tail_panel": str(DEFAULT_PANEL_PARQUET),
                "external_market_state_lagged": str(DEFAULT_EXTERNAL_STATE_CSV),
                "core_a_candidate": CORE_A_CANDIDATE,
                "core_b_candidate": CORE_B_CANDIDATE,
            },
            "selected_using_splits": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "locked_oos_role": "gate_report_only_after_candidate_freeze",
        }
    )


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not math.isfinite(std) or std <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def split_labels(
    times: list[Any], *, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> dict[Any, str]:
    unique = sorted(set(times))
    if not unique:
        return {}
    train_cut = max(1, math.floor(len(unique) * train_fraction))
    validation_cut = min(
        len(unique),
        max(train_cut + 1, math.floor(len(unique) * (train_fraction + validation_fraction))),
    )
    out: dict[Any, str] = {}
    for idx, ts in enumerate(unique):
        if idx < train_cut:
            out[ts] = "train"
        elif idx < validation_cut:
            out[ts] = "validation"
        else:
            out[ts] = "locked_oos"
    return out


def load_market_rows(panel_path: Path, external_state_csv: Path) -> list[dict[str, Any]]:
    if not panel_path.exists():
        raise FileNotFoundError(f"panel parquet not found: {panel_path}")
    if not external_state_csv.exists():
        raise FileNotFoundError(f"external state csv not found: {external_state_csv}")

    panel = pl.read_parquet(panel_path)
    panel = panel.sort("datetime")
    external = pl.read_csv(external_state_csv, try_parse_dates=True)
    panel = panel.with_columns(pl.col("datetime").dt.date().alias("effective_date"))
    frame = panel.join(external, on="effective_date", how="left").fill_null(0.0)
    rows = frame.to_dicts()
    times = [row["datetime"] for row in rows]
    labels = split_labels(times)
    for row in rows:
        row["split"] = labels.get(row["datetime"], "train")
    return rows


def _series(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([safe_float(row.get(name), np.nan) for row in rows], dtype=float)


def rolling_return(close: np.ndarray, window: int) -> np.ndarray:
    out = np.full(close.shape, np.nan, dtype=float)
    if window <= 0 or len(close) <= window:
        return out
    denom = close[:-window]
    valid = np.abs(denom) > 1e-12
    result = np.full(len(denom), np.nan, dtype=float)
    result[valid] = close[window:][valid] / denom[valid] - 1.0
    out[window:] = result
    return out


def rolling_vol(ret: np.ndarray, window: int) -> np.ndarray:
    out = np.full(ret.shape, np.nan, dtype=float)
    if window <= 1:
        return out
    for idx in range(window, len(ret)):
        sample = ret[idx - window + 1 : idx + 1]
        out[idx] = np.nanstd(sample)
    return out


def rolling_drawdown(close: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(close.shape, dtype=float)
    for idx in range(len(close)):
        start = max(0, idx - window + 1)
        peak = np.nanmax(close[start : idx + 1])
        out[idx] = (close[idx] / peak - 1.0) if peak and math.isfinite(peak) else 0.0
    return out


def build_feature_rows(
    rows: list[dict[str, Any]], config: RegimeBoostConfig
) -> list[dict[str, Any]]:
    n = len(rows)
    closes = {sym: _series(rows, f"{sym.lower()}_close") for sym in SYMBOLS}
    highs = {sym: _series(rows, f"{sym.lower()}_high") for sym in SYMBOLS}
    lows = {sym: _series(rows, f"{sym.lower()}_low") for sym in SYMBOLS}
    ret1 = {sym: np.r_[np.nan, closes[sym][1:] / closes[sym][:-1] - 1.0] for sym in SYMBOLS}
    ret72 = {sym: rolling_return(closes[sym], 72) for sym in SYMBOLS}
    ret168 = {sym: rolling_return(closes[sym], 168) for sym in SYMBOLS}
    rv72 = {sym: rolling_vol(ret1[sym], 72) for sym in SYMBOLS}
    rv720 = {sym: rolling_vol(ret1[sym], 720) for sym in SYMBOLS}
    long_vol = {
        sym: rolling_vol(ret1[sym], 720) * math.sqrt(ANNUALIZATION_HOURS) for sym in SYMBOLS
    }

    btc_ret_z = zscore(ret168["BTCUSDT"])
    eth_ret_z = zscore(ret168["ETHUSDT"])
    btc_rv_ratio = np.divide(
        rv72["BTCUSDT"], rv720["BTCUSDT"], out=np.ones(n), where=np.abs(rv720["BTCUSDT"]) > 1e-12
    )
    btc_rv_ratio_z = zscore(btc_rv_ratio)
    btc_drawdown = rolling_drawdown(closes["BTCUSDT"], 168)

    alt_ret72 = np.vstack([ret72[sym] for sym in ALT_SYMBOLS])
    alt_breadth = np.nanmean(alt_ret72 > 0.0, axis=0)
    alt_dispersion = np.nanstd(alt_ret72, axis=0)
    alt_dispersion_z = zscore(alt_dispersion)

    risk_score = np.asarray(
        [safe_float(row.get("external_risk_off_score_lag1"), 0.0) for row in rows], dtype=float
    )
    usd_z = np.asarray(
        [safe_float(row.get("external_usd_ret_z_lag1"), 0.0) for row in rows], dtype=float
    )
    vix_z = np.asarray(
        [safe_float(row.get("external_vix_z_lag1"), 0.0) for row in rows], dtype=float
    )
    curve_z = np.asarray(
        [safe_float(row.get("external_curve_z_lag1"), 0.0) for row in rows], dtype=float
    )
    oil_abs_z = np.abs(
        np.asarray(
            [safe_float(row.get("external_wti_ret_z_lag1"), 0.0) for row in rows], dtype=float
        )
    )
    cfg = config.regime
    composed_risk = (
        cfg.vix_risk_weight * vix_z
        + cfg.usd_risk_weight * usd_z
        + cfg.curve_risk_weight * curve_z
        + cfg.oil_risk_weight * oil_abs_z
    )
    risk = np.where(np.isfinite(risk_score), 0.5 * risk_score + 0.5 * composed_risk, composed_risk)
    trend_score = (
        cfg.btc_trend_weight * btc_ret_z
        + cfg.eth_trend_weight * eth_ret_z
        + cfg.breadth_weight * (2.0 * np.nan_to_num(alt_breadth, nan=0.5) - 1.0)
        - cfg.vol_penalty_weight * btc_rv_ratio_z
    )

    residual = {sym: ret168[sym] - ret168["BTCUSDT"] for sym in ALT_SYMBOLS}
    residual_z = {sym: zscore(residual[sym]) for sym in ALT_SYMBOLS}
    fast_z = {sym: zscore(ret72[sym]) for sym in ALT_SYMBOLS}

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        regime = classify_regime(
            trend_score=safe_float(trend_score[idx]),
            risk_score=safe_float(risk[idx]),
            dispersion_score=safe_float(alt_dispersion_z[idx]),
            vol_ratio=safe_float(btc_rv_ratio[idx], 1.0),
            btc_drawdown_168h=safe_float(btc_drawdown[idx]),
            config=cfg,
        )
        symbol_scores: dict[str, float] = {}
        for sym in ALT_SYMBOLS:
            symbol_scores[sym] = config.signal_residual_weight * safe_float(
                residual_z[sym][idx]
            ) + config.signal_fast_weight * safe_float(fast_z[sym][idx])
        ranked = sorted(symbol_scores.items(), key=lambda item: item[1], reverse=True)
        leader, leader_score = ranked[0]
        laggard, laggard_score = ranked[-1]
        rank_gap = safe_float(leader_score - laggard_score)
        confidence = compute_confidence(
            signal_score=max(abs(leader_score), abs(laggard_score)),
            rank_gap=rank_gap,
            dispersion_score=safe_float(alt_dispersion_z[idx]),
            trend_alignment=1.0 if regime in {"bull", "bear", "neutral_high_dispersion"} else 0.25,
            margin_safety_score=1.0,
        )
        enriched = dict(row)
        enriched.update(
            {
                "regime": regime,
                "trend_score": safe_float(trend_score[idx]),
                "risk_score": safe_float(risk[idx]),
                "dispersion_score": safe_float(alt_dispersion_z[idx]),
                "vol_ratio": safe_float(btc_rv_ratio[idx], 1.0),
                "btc_drawdown_168h": safe_float(btc_drawdown[idx]),
                "alt_breadth": safe_float(alt_breadth[idx], 0.5),
                "leader": leader,
                "leader_score": safe_float(leader_score),
                "laggard": laggard,
                "laggard_score": safe_float(laggard_score),
                "rank_gap": rank_gap,
                "confidence": confidence,
                "next_returns": {
                    sym: safe_float(ret1[sym][idx + 1]) if idx + 1 < n else 0.0 for sym in SYMBOLS
                },
                "prev_closes": {sym: safe_float(closes[sym][idx]) for sym in SYMBOLS},
                "next_highs": {
                    sym: safe_float(highs[sym][idx + 1])
                    if idx + 1 < n
                    else safe_float(highs[sym][idx])
                    for sym in SYMBOLS
                },
                "next_lows": {
                    sym: safe_float(lows[sym][idx + 1])
                    if idx + 1 < n
                    else safe_float(lows[sym][idx])
                    for sym in SYMBOLS
                },
                "long_term_vol": {sym: safe_float(long_vol[sym][idx], 1.0) for sym in SYMBOLS},
            }
        )
        out.append(enriched)
    return out


def classify_regime(
    *,
    trend_score: float,
    risk_score: float,
    dispersion_score: float,
    vol_ratio: float,
    btc_drawdown_168h: float,
    config: RegimeClassifierConfig,
) -> str:
    if (
        vol_ratio > config.vol_ratio_stress_threshold
        or btc_drawdown_168h < config.btc_drawdown_stress_threshold
        or risk_score > config.risk_stress_threshold
    ):
        return "stress"
    if trend_score > config.trend_bull_threshold and risk_score < config.risk_bear_threshold:
        return "bull"
    if trend_score < config.trend_bear_threshold or risk_score > config.risk_bear_threshold:
        return "bear"
    if (
        abs(trend_score) <= config.trend_bull_threshold
        and dispersion_score > config.dispersion_threshold
    ):
        return "neutral_high_dispersion"
    return "neutral_low_dispersion"


def side_multipliers(regime: str, config: SideBiasConfig) -> tuple[float, float]:
    if regime == "bull":
        return config.bull_long_multiplier, config.bull_short_multiplier
    if regime == "bear":
        return config.bear_long_multiplier, config.bear_short_multiplier
    if regime == "neutral_high_dispersion":
        return config.neutral_high_long_multiplier, config.neutral_high_short_multiplier
    if regime == "neutral_low_dispersion":
        return config.neutral_low_long_multiplier, config.neutral_low_short_multiplier
    return config.stress_long_multiplier, config.stress_short_multiplier


def compute_confidence(
    *,
    signal_score: float,
    rank_gap: float,
    dispersion_score: float,
    trend_alignment: float,
    margin_safety_score: float,
) -> float:
    raw = (
        0.25 * min(abs(signal_score) / 3.0, 1.0)
        + 0.25 * min(max(rank_gap, 0.0) / 4.0, 1.0)
        + 0.20 * min(max(dispersion_score, 0.0) / 3.0, 1.0)
        + 0.15 * min(max(trend_alignment, 0.0), 1.0)
        + 0.15 * min(max(margin_safety_score, 0.0), 1.0)
    )
    return float(max(0.0, min(1.0, raw)))


def asset_volatility_cap(long_term_vol: float, config: VolTargetLeverageConfig) -> float:
    vol = max(config.long_term_vol_floor, safe_float(long_term_vol, config.long_term_vol_floor))
    cap = config.target_annual_volatility / vol
    return float(max(config.min_effective_leverage, min(config.max_effective_leverage, cap)))


def effective_dynamic_leverage(
    *,
    confidence: float,
    vol_ratio: float,
    long_term_vol: float,
    regime: str,
    config: VolTargetLeverageConfig,
    requested_max_leverage: float | None = None,
) -> float:
    max_lev = min(
        config.max_effective_leverage, requested_max_leverage or config.max_effective_leverage
    )
    local_config = replace(config, max_effective_leverage=max_lev)
    if regime == "stress":
        requested = local_config.stress_leverage
    elif vol_ratio > local_config.high_vol_ratio_threshold:
        requested = local_config.high_vol_leverage
    elif vol_ratio > local_config.medium_vol_ratio_threshold:
        requested = local_config.medium_vol_leverage
    elif (
        confidence > local_config.high_confidence_threshold
        and vol_ratio < local_config.low_vol_ratio_threshold
    ):
        requested = max_lev
    elif confidence > local_config.medium_confidence_threshold:
        requested = local_config.medium_confidence_leverage
    else:
        requested = local_config.base_leverage
    return float(
        max(
            local_config.min_effective_leverage,
            min(requested, asset_volatility_cap(long_term_vol, local_config), max_lev),
        )
    )


def booster_allowed(row: dict[str, Any], config: RegimeBoostConfig, *, side: str) -> bool:
    del side
    booster = config.booster
    return bool(
        booster.enabled
        and str(row.get("regime")) in set(booster.allowed_regimes)
        and safe_float(row.get("confidence")) >= booster.min_confidence
        and safe_float(row.get("vol_ratio"), 1.0) <= booster.max_vol_ratio
    )


def grid_product(config: RegimeBoostConfig) -> list[RegimeBoostConfig]:
    configs: list[RegimeBoostConfig] = []
    core_scales = (0.10, 0.20, max(config.core_a_weight, config.core_b_weight))
    base_leverages = (1.0, 2.0, config.leverage.base_leverage)
    strides = (24, 72, 168)
    entry_thresholds = (0.75, 1.25)
    bull_longs = (1.10, config.side_bias.bull_long_multiplier)
    bear_shorts = (1.15, config.side_bias.bear_short_multiplier)
    booster_weights = (0.0, 0.05, config.booster.allocation_weight)
    booster_caps = (5.0, 10.0, config.booster.max_leverage)
    pair_weights = (0.0, 0.05, config.neutral_pair.allocation_weight)
    for (
        core_scale,
        base_leverage,
        entry_threshold,
        bull_long,
        bear_short,
        booster_weight,
        booster_cap,
        pair_weight,
        stride,
    ) in itertools.product(
        core_scales,
        base_leverages,
        entry_thresholds,
        bull_longs,
        bear_shorts,
        booster_weights,
        booster_caps,
        pair_weights,
        strides,
    ):
        configs.append(
            replace(
                config,
                core_a_weight=core_scale,
                core_b_weight=core_scale,
                rebalance_stride_hours=int(stride),
                long_entry_threshold=float(entry_threshold),
                short_entry_threshold=float(-entry_threshold),
                side_bias=replace(
                    config.side_bias,
                    bull_long_multiplier=bull_long,
                    bear_short_multiplier=bear_short,
                ),
                leverage=replace(config.leverage, base_leverage=base_leverage),
                booster=replace(
                    config.booster, allocation_weight=booster_weight, max_leverage=booster_cap
                ),
                neutral_pair=replace(config.neutral_pair, allocation_weight=pair_weight),
            )
        )
    return configs


def enforce_grid_cap(
    configs: list[RegimeBoostConfig], selection: SelectionConfig
) -> tuple[list[RegimeBoostConfig], dict[str, Any]]:
    product_space_size = len(configs)
    configured = max(1, int(selection.grid_limit))
    hard = max(1, int(selection.hard_grid_cap))
    evaluated = min(configured, hard, product_space_size)
    capped = configs[:evaluated]
    meta = {
        "configured_grid_limit": configured,
        "hard_grid_cap": hard,
        "product_space_size": product_space_size,
        "evaluated_count": evaluated,
        "skipped_pruned_count": max(0, product_space_size - evaluated),
        "search_space_hash": stable_json_hash([dataclass_to_dict(item) for item in configs]),
        "selection_score_fields": [
            "train.total_return",
            "validation.total_return",
            "train.sharpe",
            "validation.sharpe",
            "train.max_drawdown",
            "validation.max_drawdown",
            "train.liquidation_count",
            "validation.liquidation_count",
        ],
    }
    return capped, meta


def fit_neutral_pair_overlay(
    rows: list[dict[str, Any]], config: NeutralPairOverlayConfig
) -> dict[str, Any]:
    fit_rows = [row for row in rows if str(row.get("split")) in set(config.fit_splits)]
    if not fit_rows:
        return {
            "eligible_symbols": [],
            "hedge_ratios": {},
            "fit_splits": list(config.fit_splits),
            "as_of_policy": "no_fit_rows",
        }
    returns = {
        sym: np.asarray([safe_float(row["next_returns"].get(sym)) for row in fit_rows], dtype=float)
        for sym in ALT_SYMBOLS
    }
    vol = {sym: float(np.nanstd(vals)) for sym, vals in returns.items()}
    eligible = [sym for sym in ALT_SYMBOLS if math.isfinite(vol[sym]) and vol[sym] > 1e-12]
    hedge_ratios: dict[str, float] = {}
    for long_sym, short_sym in itertools.permutations(eligible, 2):
        ratio = vol[long_sym] / max(vol[short_sym], 1e-12)
        hedge_ratios[f"{long_sym}|{short_sym}"] = float(
            max(config.hedge_ratio_floor, min(config.hedge_ratio_ceiling, ratio))
        )
    return {
        "eligible_symbols": eligible,
        "hedge_ratios": hedge_ratios,
        "fit_splits": list(config.fit_splits),
        "fit_row_count": len(fit_rows),
        "as_of_policy": "train_validation_lagged_features_only_frozen_before_locked_oos",
        "uses_locked_oos_for_pair_fit": False,
    }


def _leg_liquidated(
    *,
    side: str,
    leverage: float,
    entry_price: float,
    high_price: float,
    low_price: float,
    margin: MarginConfig,
) -> bool:
    lev = max(leverage, 1e-12)
    adverse = max(0.0, (1.0 / lev) - margin.liquidation_reserve_rate)
    if adverse <= 0.0 or entry_price <= 0.0:
        return True
    if side == "long":
        return low_price <= entry_price * (1.0 - adverse)
    return high_price >= entry_price * (1.0 + adverse)


def evaluate_config(
    rows: list[dict[str, Any]],
    config: RegimeBoostConfig,
    pair_fit: dict[str, Any],
    *,
    include_locked_oos: bool,
    diagnostic_max_leverage: float | None = None,
) -> dict[str, Any]:
    split_returns: dict[str, list[float]] = defaultdict(list)
    split_liq: Counter[str] = Counter()
    split_buffers: dict[str, list[float]] = defaultdict(list)
    split_trade_count: Counter[str] = Counter()
    split_regimes: dict[str, Counter[str]] = defaultdict(Counter)
    leverage_values: list[float] = []
    booster_activations = 0
    pair_activations = 0
    long_count = 0
    short_count = 0

    allowed_pair_symbols = set(pair_fit.get("eligible_symbols") or [])
    hedge_ratios = dict(pair_fit.get("hedge_ratios") or {})

    stride = max(1, int(config.rebalance_stride_hours))
    for idx, row in enumerate(rows[:-1]):
        split = str(row.get("split", "train"))
        if idx % stride != 0:
            continue
        if split == "locked_oos" and not include_locked_oos:
            continue
        regime = str(row.get("regime"))
        split_regimes[split][regime] += 1
        long_mult, short_mult = side_multipliers(regime, config.side_bias)
        leader = str(row.get("leader"))
        laggard = str(row.get("laggard"))
        leader_score = safe_float(row.get("leader_score")) * long_mult
        laggard_score = safe_float(row.get("laggard_score")) * short_mult
        confidence = safe_float(row.get("confidence"))
        vol_ratio = safe_float(row.get("vol_ratio"), 1.0)
        returns = dict(row.get("next_returns") or {})
        prev_closes = dict(row.get("prev_closes") or {})
        highs = dict(row.get("next_highs") or {})
        lows = dict(row.get("next_lows") or {})
        long_vols = dict(row.get("long_term_vol") or {})

        portfolio_return = 0.0
        gross_notional = 0.0
        hourly_liq = 0

        def add_leg(
            symbol: str,
            side: str,
            weight: float,
            leverage: float,
            *,
            returns_by_symbol: dict[str, float] = returns,
            split_name: str = split,
            prev_close_by_symbol: dict[str, float] = prev_closes,
            high_by_symbol: dict[str, float] = highs,
            low_by_symbol: dict[str, float] = lows,
        ) -> None:
            nonlocal portfolio_return, gross_notional, hourly_liq, long_count, short_count
            if weight <= 0.0 or leverage <= 0.0:
                return
            sym_ret = safe_float(returns_by_symbol.get(symbol))
            signed_ret = sym_ret if side == "long" else -sym_ret
            portfolio_return += weight * leverage * signed_ret
            gross_notional += abs(weight * leverage)
            leverage_values.append(float(leverage))
            split_trade_count[split_name] += 1
            if side == "long":
                long_count += 1
            else:
                short_count += 1
            if _leg_liquidated(
                side=side,
                leverage=leverage,
                entry_price=safe_float(prev_close_by_symbol.get(symbol)),
                high_price=safe_float(high_by_symbol.get(symbol)),
                low_price=safe_float(low_by_symbol.get(symbol)),
                margin=config.margin,
            ):
                hourly_liq += 1

        def leg_leverage(
            symbol: str,
            requested_max: float | None = None,
            *,
            confidence_value: float = confidence,
            vol_ratio_value: float = vol_ratio,
            long_vol_by_symbol: dict[str, float] = long_vols,
            regime_name: str = regime,
        ) -> float:
            return effective_dynamic_leverage(
                confidence=confidence_value,
                vol_ratio=vol_ratio_value,
                long_term_vol=safe_float(long_vol_by_symbol.get(symbol), 1.0),
                regime=regime_name,
                config=config.leverage,
                requested_max_leverage=requested_max,
            )

        # Core A: external-risk state-distilled seed, macro risk downshifts longs and boosts shorts.
        risk_score = safe_float(row.get("risk_score"))
        core_a_long_weight = config.core_a_weight * (
            config.external_risk_downshift
            if risk_score > config.regime.risk_bear_threshold
            else 1.0
        )
        core_a_short_weight = config.core_a_weight * (
            config.external_risk_short_boost
            if risk_score > config.regime.risk_bear_threshold
            else 1.0
        )
        if leader_score >= config.long_entry_threshold:
            add_leg(leader, "long", core_a_long_weight, leg_leverage(leader))
        if laggard_score <= config.short_entry_threshold:
            add_leg(laggard, "short", core_a_short_weight, leg_leverage(laggard))

        # Core B: pure state-distilled leadership/unwind seed.
        if leader_score >= config.long_entry_threshold:
            add_leg(leader, "long", config.core_b_weight, leg_leverage(leader))
        if laggard_score <= config.short_entry_threshold:
            add_leg(laggard, "short", config.core_b_weight, leg_leverage(laggard))

        # Booster C: high-confidence sleeve, capped and volatility-targeted up to 25x.
        if config.booster.allocation_weight > 0.0:
            booster_side = (
                "short" if regime == "bear" and abs(laggard_score) >= abs(leader_score) else "long"
            )
            booster_symbol = laggard if booster_side == "short" else leader
            if booster_allowed(row, config, side=booster_side):
                lev = leg_leverage(
                    booster_symbol, diagnostic_max_leverage or config.booster.max_leverage
                )
                add_leg(booster_symbol, booster_side, config.booster.allocation_weight, lev)
                booster_activations += 1

        # Overlay D: frozen neutral-high-dispersion pair, fit on train/validation only.
        if (
            config.neutral_pair.enabled
            and config.neutral_pair.allocation_weight > 0.0
            and regime == "neutral_high_dispersion"
            and leader in allowed_pair_symbols
            and laggard in allowed_pair_symbols
            and safe_float(row.get("rank_gap")) >= config.neutral_pair.min_rank_gap
        ):
            hedge_ratio = safe_float(hedge_ratios.get(f"{leader}|{laggard}"), 1.0)
            pair_weight = config.neutral_pair.allocation_weight
            pair_leverage = min(config.neutral_pair.max_pair_leverage, leg_leverage(leader))
            add_leg(leader, "long", pair_weight, pair_leverage)
            add_leg(laggard, "short", pair_weight * hedge_ratio, pair_leverage)
            pair_activations += 1

        fee = abs(gross_notional) * config.margin.taker_fee_rate
        portfolio_return -= fee
        split_returns[split].append(float(portfolio_return))
        split_liq[split] += hourly_liq
        equity_estimate = STARTING_EQUITY * (1.0 + float(np.nansum(split_returns[split])))
        maintenance = max(
            0.0, equity_estimate * gross_notional * config.margin.maintenance_margin_rate
        )
        split_buffers[split].append(float(equity_estimate - maintenance))

    metrics: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "locked_oos"):
        metrics[split] = split_metrics(split_returns.get(split, []))
        metrics[split].update(
            {
                "liquidation_count": int(split_liq.get(split, 0)),
                "minimum_margin_buffer": float(min(split_buffers.get(split, [STARTING_EQUITY]))),
                "margin_buffer_positive": float(min(split_buffers.get(split, [STARTING_EQUITY])))
                > 0.0,
                "trade_count": int(split_trade_count.get(split, 0)),
                "regime_counts": dict(split_regimes.get(split, Counter())),
            }
        )
    return {
        "split_metrics": metrics,
        "max_effective_leverage": float(max(leverage_values, default=0.0)),
        "booster_activation_count": int(booster_activations),
        "pair_overlay_activation_count": int(pair_activations),
        "long_count": int(long_count),
        "short_count": int(short_count),
    }


def split_metrics(returns: list[float]) -> dict[str, float]:
    if not returns:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "smart_sortino": 0.0,
            "calmar": 0.0,
            "return_mdd": 0.0,
        }
    arr = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + np.nan_to_num(arr, nan=0.0))
    peak = np.maximum.accumulate(equity)
    drawdown = np.divide(equity, peak, out=np.ones_like(equity), where=peak > 0) - 1.0
    total_return = float(equity[-1] - 1.0)
    max_drawdown = float(abs(np.nanmin(drawdown)))
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    sharpe = mean / std * math.sqrt(ANNUALIZATION_HOURS) if std > 1e-12 else 0.0
    downside = arr[arr < 0.0]
    downside_std = float(np.nanstd(downside)) if len(downside) else 0.0
    sortino = mean / downside_std * math.sqrt(ANNUALIZATION_HOURS) if downside_std > 1e-12 else 0.0
    smart_sortino = sortino * max(0.0, min(1.0, 1.0 - max_drawdown))
    calmar = total_return / max_drawdown if max_drawdown > 1e-12 else 0.0
    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "smart_sortino": float(smart_sortino),
        "calmar": float(calmar),
        "return_mdd": float(calmar),
    }


def selection_score(metrics: dict[str, dict[str, Any]], config: SelectionConfig) -> float:
    train = metrics["train"]
    val = metrics["validation"]
    liq_penalty = (
        int(train.get("liquidation_count", 0)) + int(val.get("liquidation_count", 0))
    ) * config.liquidation_penalty
    score = (
        config.train_return_weight * safe_float(train.get("total_return"))
        + config.validation_return_weight * safe_float(val.get("total_return"))
        + config.sharpe_weight * (safe_float(train.get("sharpe")) + safe_float(val.get("sharpe")))
        - config.drawdown_penalty
        * (safe_float(train.get("max_drawdown")) + safe_float(val.get("max_drawdown")))
        - liq_penalty
    )
    if safe_float(val.get("total_return")) < config.min_validation_return:
        score -= 1.0
    return float(score)


def select_candidate_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if not row.get("uses_locked_oos_for_selection")]
    if not eligible:
        raise ValueError("no train/validation-only rows available for selection")
    return max(eligible, key=lambda row: safe_float(row.get("selection_score"), float("-inf")))


def strict_lane_promoted(
    metrics: dict[str, dict[str, Any]], selection: SelectionConfig
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for split in ("train", "validation", "locked_oos"):
        payload = dict(metrics.get(split) or {})
        if int(payload.get("liquidation_count") or 0) > 0:
            reasons.append(f"{split}_liquidation_count_positive")
        if safe_float(payload.get("minimum_margin_buffer"), 0.0) <= 0.0:
            reasons.append(f"{split}_margin_buffer_non_positive")
    validation = dict(metrics.get("validation") or {})
    locked_oos = dict(metrics.get("locked_oos") or {})
    if safe_float(validation.get("total_return"), 0.0) <= 0.0:
        reasons.append("validation_return_non_positive")
    if safe_float(locked_oos.get("total_return"), 0.0) <= 0.0:
        reasons.append("locked_oos_return_non_positive")
    for metric in ("sharpe", "sortino", "smart_sortino", "calmar"):
        if safe_float(locked_oos.get(metric), 0.0) <= 0.0:
            reasons.append(f"locked_oos_{metric}_non_positive")
    if safe_float(locked_oos.get("max_drawdown"), 1.0) > selection.strict_oos_mdd_max:
        reasons.append("locked_oos_mdd_gt_25pct")
    return not reasons, reasons


def run_selection(
    rows: list[dict[str, Any]], config: RegimeBoostConfig
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    pair_fit = fit_neutral_pair_overlay(rows, config.neutral_pair)
    configs, grid_meta = enforce_grid_cap(grid_product(config), config.selection)
    ledger: list[dict[str, Any]] = []
    for idx, candidate_config in enumerate(configs):
        eval_result = evaluate_config(rows, candidate_config, pair_fit, include_locked_oos=False)
        metrics = eval_result["split_metrics"]
        score = selection_score(metrics, candidate_config.selection)
        ledger.append(
            {
                "rank_input_index": idx,
                "selection_score": score,
                "selected_using_splits": ["train", "validation"],
                "uses_locked_oos_for_selection": False,
                "locked_oos_metrics_visible_during_selection": False,
                "config": dataclass_to_dict(candidate_config),
                "train": metrics["train"],
                "validation": metrics["validation"],
                "max_effective_leverage_train_validation": eval_result["max_effective_leverage"],
                "booster_activation_count_train_validation": eval_result[
                    "booster_activation_count"
                ],
                "pair_overlay_activation_count_train_validation": eval_result[
                    "pair_overlay_activation_count"
                ],
            }
        )
    selected = select_candidate_from_rows(ledger)
    ledger_sorted = sorted(
        ledger, key=lambda row: safe_float(row.get("selection_score")), reverse=True
    )
    return selected, ledger_sorted, grid_meta, pair_fit


def build_input_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in paths:
        manifest.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return manifest


def build_freeze_payload(
    *,
    selected: dict[str, Any],
    ledger_path: Path,
    input_manifest: list[dict[str, Any]],
    grid_meta: dict[str, Any],
    pair_fit: dict[str, Any],
) -> dict[str, Any]:
    selected_config = dict(selected["config"])
    return {
        "artifact_kind": "state_distilled_regime_boost_frozen_config",
        "run_id": RUN_ID,
        "frozen_at": utc_now(),
        "selected_candidate_ids": {
            "core_a": CORE_A_CANDIDATE,
            "core_b": CORE_B_CANDIDATE,
            "booster_seed": CORE_B_CANDIDATE,
            "neutral_pair_overlay": "train_validation_frozen_pair_fit",
        },
        "selected_config": selected_config,
        "selection_score": selected["selection_score"],
        "selection_metrics_train_validation_only": {
            "train": selected["train"],
            "validation": selected["validation"],
        },
        "selection_ledger_path": str(ledger_path),
        "selection_ledger_sha256": sha256_file(ledger_path),
        "input_artifact_manifest": input_manifest,
        "git_state": git_state(),
        "grid_metadata": grid_meta,
        "pair_fit": pair_fit,
        "uses_locked_oos_for_selection": False,
        "locked_oos_metrics_visible_during_selection": False,
        "candidate_freeze_before_locked_oos_gate": True,
    }


def write_freeze_artifacts(
    output_dir: Path, freeze_payload: dict[str, Any]
) -> tuple[Path, Path, str]:
    freeze_path = output_dir / "frozen_config.json"
    write_json(freeze_path, freeze_payload)
    freeze_hash = sha256_file(freeze_path)
    sidecar = {
        "artifact_kind": "state_distilled_regime_boost_freeze_manifest",
        "freeze_artifact_path": str(freeze_path),
        "freeze_artifact_hash": freeze_hash,
        "hash_policy": "sha256_of_frozen_config_payload_file_hash_not_embedded_inside_payload",
        "created_at": utc_now(),
    }
    sidecar_path = output_dir / "frozen_config.sha256.json"
    write_json(sidecar_path, sidecar)
    return freeze_path, sidecar_path, freeze_hash


def build_locked_oos_gate(
    *,
    freeze_path: Path,
    freeze_hash: str,
    frozen_config: RegimeBoostConfig,
    rows: list[dict[str, Any]],
    pair_fit: dict[str, Any],
) -> dict[str, Any]:
    opened_at = utc_now()
    result = evaluate_config(rows, frozen_config, pair_fit, include_locked_oos=True)
    promoted, reasons = strict_lane_promoted(result["split_metrics"], frozen_config.selection)
    return {
        "artifact_kind": "state_distilled_regime_boost_locked_oos_gate",
        "locked_oos_opened_at": opened_at,
        "freeze_artifact_path": str(freeze_path),
        "freeze_artifact_hash": freeze_hash,
        "candidate_freeze_before_locked_oos_gate": True,
        "uses_locked_oos_for_selection": False,
        "locked_oos_metrics_visible_during_selection": False,
        "selected_config": dataclass_to_dict(frozen_config),
        "split_metrics": result["split_metrics"],
        "max_effective_leverage": result["max_effective_leverage"],
        "booster_activation_count": result["booster_activation_count"],
        "pair_overlay_activation_count": result["pair_overlay_activation_count"],
        "long_count": result["long_count"],
        "short_count": result["short_count"],
        "strict_promoted_success": promoted,
        "strict_rejection_reasons": reasons,
    }


def config_from_dict(payload: dict[str, Any] | None = None) -> RegimeBoostConfig:
    if not payload:
        return RegimeBoostConfig()
    base = RegimeBoostConfig()

    def update_dataclass(obj: Any, values: dict[str, Any]) -> Any:
        allowed = {field_name for field_name in obj.__dataclass_fields__}  # type: ignore[attr-defined]
        return replace(obj, **{k: v for k, v in values.items() if k in allowed})

    return replace(
        base,
        regime=update_dataclass(base.regime, dict(payload.get("regime") or {})),
        side_bias=update_dataclass(base.side_bias, dict(payload.get("side_bias") or {})),
        leverage=update_dataclass(base.leverage, dict(payload.get("leverage") or {})),
        booster=update_dataclass(base.booster, dict(payload.get("booster") or {})),
        neutral_pair=update_dataclass(base.neutral_pair, dict(payload.get("neutral_pair") or {})),
        selection=update_dataclass(base.selection, dict(payload.get("selection") or {})),
        margin=update_dataclass(base.margin, dict(payload.get("margin") or {})),
        **{
            key: value
            for key, value in payload.items()
            if key
            in {
                "long_entry_threshold",
                "short_entry_threshold",
                "core_a_weight",
                "core_b_weight",
                "rebalance_stride_hours",
                "signal_fast_weight",
                "signal_residual_weight",
                "external_risk_downshift",
                "external_risk_short_boost",
            }
        },
    )


def config_from_selected(selected_config: dict[str, Any]) -> RegimeBoostConfig:
    return config_from_dict(selected_config)


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def diagnostic_leverage_sweep(
    rows: list[dict[str, Any]], config: RegimeBoostConfig, pair_fit: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cap in (5.0, 6.0, 10.0, 15.0, 25.0):
        diag_config = replace(
            config,
            booster=replace(
                config.booster,
                enabled=True,
                allocation_weight=max(config.booster.allocation_weight, 0.10),
                max_leverage=cap,
            ),
            leverage=replace(
                config.leverage,
                max_effective_leverage=min(25.0, max(config.leverage.max_effective_leverage, cap)),
            ),
        )
        result = evaluate_config(
            rows, diag_config, pair_fit, include_locked_oos=True, diagnostic_max_leverage=cap
        )
        split_status = result["split_metrics"]
        total_liq = sum(
            int(split_status[split].get("liquidation_count") or 0) for split in split_status
        )
        min_buffer = min(
            safe_float(split_status[split].get("minimum_margin_buffer"), STARTING_EQUITY)
            for split in split_status
        )
        out.append(
            {
                "lane": "diagnostic_nonfatal",
                "diagnostic_only": True,
                "live_promotion_lane": "strict_deploy_lane_only",
                "booster_cap": cap,
                "max_effective_leverage": result["max_effective_leverage"],
                "split_diagnostics": split_status,
                "total_liquidation_count": int(total_liq),
                "minimum_margin_buffer": float(min_buffer),
                "account_wipeout_count": int(min_buffer <= 0.0),
                "promotion_allowed": False,
                "separate_from_strict_deploy": True,
            }
        )
    return out


def format_pct(value: Any) -> str:
    return f"{100.0 * safe_float(value):+.4f}%"


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    strict = summary["strict_lane"]
    oos = strict["locked_oos"]
    train = strict["train"]
    val = strict["validation"]
    lines = [
        "# StateDistilledRegimeBoostPortfolio — Real current-tail report",
        "",
        f"Generated: `{summary['generated_at_utc']}`",
        "",
        "## Strategy / factor / calibration used",
        "",
        f"- Core A: `{CORE_A_CANDIDATE}` external-risk state-distilled seed.",
        f"- Core B: `{CORE_B_CANDIDATE}` pure leadership/unwind seed.",
        "- Overlay: tunable regime classifier, side-bias multipliers, volatility-targeted leverage, conditional booster up to 25x, and frozen neutral-pair overlay.",
        "- Calibration/selection: bounded grid, train+validation score only; locked-OOS opened after freeze as gate/report only.",
        "- Calendar/current-base teacher: hypothesis_reference_only, not selection or promotion target.",
        "",
        "## Train/validation selection provenance",
        "",
        f"- uses_locked_oos_for_selection: `{summary['selection']['uses_locked_oos_for_selection']}`",
        f"- locked_oos_metrics_visible_during_selection: `{summary['selection']['locked_oos_metrics_visible_during_selection']}`",
        f"- configured/evaluated/product grid: `{summary['selection']['configured_grid_limit']}` / `{summary['selection']['evaluated_count']}` / `{summary['selection']['product_space_size']}`",
        f"- freeze hash: `{summary['selection']['freeze_artifact_hash']}`",
        "",
        "## Strict zero-liquidation lane",
        "",
        "| Split | Return | MDD | Sharpe | Sortino | Calmar | Liq | Min buffer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in (("train", train), ("validation", val), ("locked_oos", oos)):
        lines.append(
            f"| {name} | {format_pct(payload['total_return'])} | {format_pct(payload['max_drawdown'])} | "
            f"{safe_float(payload['sharpe']):.4f} | {safe_float(payload['sortino']):.4f} | "
            f"{safe_float(payload['calmar']):.4f} | {int(payload['liquidation_count'])} | "
            f"{safe_float(payload['minimum_margin_buffer']):.4f} |"
        )
    lines.extend(
        [
            "",
            f"Strict promoted success: `{strict['promoted_success']}`",
            f"Strict rejection reasons: `{strict['rejection_reasons']}`",
            f"Max effective leverage: `{summary['booster']['max_effective_leverage']:.4f}`",
            "",
            "## Diagnostic high-leverage nonfatal lane",
            "",
            "| Booster cap | Max effective lev | OOS return | OOS MDD | Total liq | Min buffer | Promotion |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["diagnostic_nonfatal_lane"]["results"]:
        locked = row["split_diagnostics"]["locked_oos"]
        lines.append(
            f"| {safe_float(row['booster_cap']):.1f} | {safe_float(row['max_effective_leverage']):.4f} | "
            f"{format_pct(locked['total_return'])} | {format_pct(locked['max_drawdown'])} | "
            f"{int(row['total_liquidation_count'])} | {safe_float(row['minimum_margin_buffer']):.4f} | diagnostic-only |"
        )
    lines.extend(
        [
            "",
            "## Memory / artifacts",
            "",
            f"- Peak RSS: `{summary['memory']['peak_rss_bytes']}` bytes (`{summary['memory']['peak_rss_mib']:.2f}` MiB).",
            f"- Summary JSON: `{summary['artifact_paths']['summary_latest']}`",
            f"- Frozen config: `{summary['artifact_paths']['frozen_config']}`",
            f"- Locked-OOS gate: `{summary['artifact_paths']['locked_oos_gate']}`",
            f"- Selection ledger: `{summary['artifact_paths']['selection_ledger']}`",
            "",
            "Research history/source inventory: no global source ledger change; this uses the existing current-tail crypto panel and existing lagged FRED external-state source.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = json.loads(Path(args.config).read_text()) if args.config else None
    config = config_from_dict(config_payload)
    if args.grid_limit is not None:
        config = replace(
            config, selection=replace(config.selection, grid_limit=int(args.grid_limit))
        )
    if args.max_booster_leverage is not None:
        lev_cap = min(25.0, float(args.max_booster_leverage))
        config = replace(
            config,
            booster=replace(config.booster, max_leverage=lev_cap),
            leverage=replace(
                config.leverage,
                max_effective_leverage=min(
                    25.0, max(config.leverage.max_effective_leverage, lev_cap)
                ),
            ),
        )

    rows = build_feature_rows(
        load_market_rows(Path(args.panel_parquet), Path(args.external_state_csv)), config
    )
    selected, ledger, grid_meta, pair_fit = run_selection(rows, config)
    ledger_path = output_dir / "selection_ledger.jsonl"
    write_ledger(ledger_path, ledger)
    input_manifest = build_input_manifest(
        [
            Path(args.panel_parquet),
            Path(args.external_state_csv),
            *[Path(p) for p in args.candidate_artifacts],
        ]
    )
    freeze_payload = build_freeze_payload(
        selected=selected,
        ledger_path=ledger_path,
        input_manifest=input_manifest,
        grid_meta=grid_meta,
        pair_fit=pair_fit,
    )
    freeze_path, sidecar_path, freeze_hash = write_freeze_artifacts(output_dir, freeze_payload)
    frozen_config = config_from_selected(selected["config"])
    gate = build_locked_oos_gate(
        freeze_path=freeze_path,
        freeze_hash=freeze_hash,
        frozen_config=frozen_config,
        rows=rows,
        pair_fit=pair_fit,
    )
    gate_path = output_dir / "locked_oos_gate.json"
    write_json(gate_path, gate)

    diagnostics = diagnostic_leverage_sweep(rows, frozen_config, pair_fit)
    strict_metrics = gate["split_metrics"]
    strict_promoted = bool(gate["strict_promoted_success"])
    total_liq = sum(
        int(strict_metrics[split].get("liquidation_count") or 0) for split in strict_metrics
    )
    min_buffer = min(
        safe_float(strict_metrics[split].get("minimum_margin_buffer"), STARTING_EQUITY)
        for split in strict_metrics
    )
    strategy_card = base_strategy_validity_card()
    memory_bytes = peak_rss_bytes()
    summary = {
        "artifact_kind": "state_distilled_regime_boost_summary",
        "run_id": RUN_ID,
        "generated_at_utc": utc_now(),
        "strategy_name": "StateDistilledRegimeBoostPortfolio",
        "strategy_validity": strategy_card,
        "calendar_current_base_teacher": {
            "role": "hypothesis_reference_only",
            "used_as_selection_target": False,
            "used_as_promotion_target": False,
            "promotion_target_policy": "strict_liquidation_and_risk_metrics_only_return_mdd_diagnostic",
        },
        "selection": {
            **grid_meta,
            "selected_using_splits": ["train", "validation"],
            "uses_locked_oos_for_selection": False,
            "locked_oos_metrics_visible_during_selection": False,
            "candidate_freeze_before_locked_oos_gate": True,
            "selection_ledger_path": str(ledger_path),
            "selection_ledger_sha256": sha256_file(ledger_path),
            "freeze_artifact_path": str(freeze_path),
            "freeze_sidecar_path": str(sidecar_path),
            "freeze_artifact_hash": freeze_hash,
            "selected_score": selected["selection_score"],
            "train": selected["train"],
            "validation": selected["validation"],
        },
        "locked_oos_gate": {
            "path": str(gate_path),
            "locked_oos_opened_at": gate["locked_oos_opened_at"],
            "freeze_artifact_hash": gate["freeze_artifact_hash"],
            "uses_locked_oos_for_selection": False,
            "selected_params_byte_identical_to_freeze": stable_json_hash(gate["selected_config"])
            == stable_json_hash(freeze_payload["selected_config"]),
        },
        "strict_lane": {
            "lane": "strict_zero_liquidation",
            "promoted_success": strict_promoted and bool(strategy_card["strategy_valid"]),
            "rejection_reasons": list(gate["strict_rejection_reasons"]),
            "liquidation_count_total": int(total_liq),
            "min_margin_buffer": float(min_buffer),
            "oos_mdd_gate_max": frozen_config.selection.strict_oos_mdd_max,
            "train": strict_metrics["train"],
            "validation": strict_metrics["validation"],
            "locked_oos": strict_metrics["locked_oos"],
            "long_count": gate["long_count"],
            "short_count": gate["short_count"],
        },
        "booster": {
            "max_configured_leverage": frozen_config.booster.max_leverage,
            "max_effective_leverage": gate["max_effective_leverage"],
            "activation_count": gate["booster_activation_count"],
            "cap_policy": "asset_long_term_volatility_targeted_and_never_above_25x",
        },
        "neutral_pair_overlay": {
            "pair_fit": pair_fit,
            "activation_count": gate["pair_overlay_activation_count"],
            "uses_locked_oos_for_pair_fit": False,
        },
        "diagnostic_nonfatal_lane": {
            "lane": "diagnostic_nonfatal_high_leverage",
            "diagnostic_only": True,
            "live_promotion_distinct_from_strict_lane": True,
            "promotion_eligible": False,
            "reported_fields": [
                "liquidation_count",
                "event_drawdown",
                "equity_loss",
                "recovery",
                "account_wipeout_count",
                "minimum_margin_buffer",
            ],
            "results": diagnostics,
        },
        "memory": {
            "peak_rss_bytes": memory_bytes,
            "peak_rss_mib": memory_bytes / (1024.0 * 1024.0),
            "limit_bytes": int(args.max_rss_bytes),
            "under_8gb": memory_bytes < int(args.max_rss_bytes),
        },
        "selected_config": dataclass_to_dict(frozen_config),
        "artifact_paths": {},
    }
    summary_path = output_dir / "state_distilled_regime_boost_summary_latest.json"
    write_json(summary_path, summary)
    timestamped_summary_path = (
        output_dir
        / f"state_distilled_regime_boost_summary_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    write_json(timestamped_summary_path, summary)
    report_path = output_dir / "state_distilled_regime_boost_report_latest.md"
    summary["artifact_paths"] = {
        "summary_latest": str(summary_path),
        "summary_timestamped": str(timestamped_summary_path),
        "report_latest": str(report_path),
        "frozen_config": str(freeze_path),
        "freeze_sidecar": str(sidecar_path),
        "locked_oos_gate": str(gate_path),
        "selection_ledger": str(ledger_path),
    }
    write_json(summary_path, summary)
    write_json(timestamped_summary_path, summary)
    write_markdown_report(report_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-parquet", default=str(DEFAULT_PANEL_PARQUET))
    parser.add_argument("--external-state-csv", default=str(DEFAULT_EXTERNAL_STATE_CSV))
    parser.add_argument(
        "--candidate-artifacts", nargs="*", default=[str(p) for p in DEFAULT_CANDIDATE_ARTIFACTS]
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", default="")
    parser.add_argument("--grid-limit", type=int, default=None)
    parser.add_argument("--max-booster-leverage", type=float, default=None)
    parser.add_argument("--max-rss-bytes", type=int, default=8 * 1024**3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {
                "summary_path": summary["artifact_paths"]["summary_latest"],
                "strict_promoted_success": summary["strict_lane"]["promoted_success"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
