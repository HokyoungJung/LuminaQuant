#!/usr/bin/env python3
"""Discover new >=30m Alpha Zoo families with memory-safe feedback gates.

This runner is research/paper-testnet only. It constructs 30m bars directly from
local 1s OHLCV parquet, derives higher timeframes from that 30m base, ranks only
on train+validation evidence, and attaches locked-OOS strictly after candidate
freeze as gate/report evidence. It never executes orders and never enables real
money.
"""

from __future__ import annotations

import argparse
import csv
import gc
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
import polars as pl

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
    SPLITS,
    SimResult,
    _json_safe,
    _split_mask,
    load_feature_points,
    simulate_symbol,
    split_metrics,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_30m_plus_alpha_feedback_discovery_20260523"
)
DEFAULT_PRIOR_ARTIFACT = (
    REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/"
    "alpha_zoo_debounced_efficiency_repair_discovery_20260523/"
    "alpha_zoo_debounced_efficiency_repair_discovery_latest.json"
)
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT")
DEFAULT_TIMEFRAMES = ("30m", "1h", "2h", "4h", "6h")
BTC_REGIME_SYMBOL = "BTCUSDT"
BAR_CONSTRUCTION = "native_1s_to_30m_base_then_requested_timeframe"
STRATEGY_SCOPE = "single_symbol_only"

MIN_TRAIN_FEATURE_COVERAGE = 0.80
MIN_VALIDATION_FEATURE_COVERAGE = 0.80
MIN_LOCKED_OOS_FEATURE_COVERAGE = 0.80
FEATURE_COVERAGE_THRESHOLDS = {
    "train": MIN_TRAIN_FEATURE_COVERAGE,
    "validation": MIN_VALIDATION_FEATURE_COVERAGE,
    "locked_oos": MIN_LOCKED_OOS_FEATURE_COVERAGE,
}

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
    "min_train_feature_coverage": MIN_TRAIN_FEATURE_COVERAGE,
    "min_validation_feature_coverage": MIN_VALIDATION_FEATURE_COVERAGE,
    "min_locked_oos_feature_coverage_report_gate": MIN_LOCKED_OOS_FEATURE_COVERAGE,
}

EXTERNAL_RESEARCH_REFERENCES = [
    {
        "label": "AdaptiveTrend crypto trend-following research",
        "url": "https://arxiv.org/abs/2602.11708",
        "usage": "motivates >=30m/6h trend-following, volatility regimes, trailing exits, and cost robustness",
    },
    {
        "label": "A decade of evidence of trend following in cryptocurrencies",
        "url": "https://arxiv.org/abs/2009.12155",
        "usage": "supports trend-following as a viable crypto alpha family but not a real-money permission",
    },
    {
        "label": "Dynamic time-series momentum of cryptocurrencies",
        "url": "https://www.sciencedirect.com/science/article/pii/S1062940821000590",
        "usage": "motivates dynamic momentum cycles and volatility-acceleration filters across intraday frequencies",
    },
    {
        "label": "Binance USDⓈ-M funding rate history docs",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History",
        "usage": "documents funding-rate fields used only from local feature stores in this runner",
    },
    {
        "label": "Binance USDⓈ-M taker buy/sell volume docs",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume",
        "usage": "documents taker-flow fields; local stale/missing coverage fails closed",
    },
]

BASELINE_LANES = [
    {
        "lane": "active",
        "leverage": 7.0,
        "allocation_fraction": 0.20,
        "status": "baseline_preserved",
    },
    {
        "lane": "balanced",
        "leverage": 6.0,
        "allocation_fraction": 0.175,
        "status": "baseline_preserved",
    },
    {
        "lane": "validation_leader",
        "leverage": 5.0,
        "allocation_fraction": 0.20,
        "status": "baseline_preserved",
    },
    {
        "lane": "efficiency_reference",
        "leverage": 4.0,
        "allocation_fraction": 0.175,
        "status": "baseline_preserved",
    },
]
for _lane in BASELINE_LANES:
    _lane["notional_fraction"] = _lane["leverage"] * _lane["allocation_fraction"]
    _lane["ready_for_real"] = False
    _lane["real_money_execution"] = False

CANDIDATE_FIELDS = [
    "rank",
    "model_id",
    "family",
    "symbol",
    "timeframe",
    "side",
    "lookback_bars",
    "threshold",
    "exit_threshold",
    "min_hold_bars",
    "cooldown_bars",
    "filter_label",
    "feature_backed",
    "feature_train_coverage",
    "feature_validation_coverage",
    "feature_locked_oos_coverage",
    "max_asof_feature_age_hours",
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
    label: str
    min_adx: float
    max_vol_quantile: float | None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _timeframe_hours(timeframe: str) -> float:
    if timeframe.endswith("m"):
        return float(timeframe[:-1]) / 60.0
    if timeframe.endswith("h"):
        return float(timeframe[:-1])
    raise ValueError(f"unsupported timeframe {timeframe!r}")


def _pandas_rule(timeframe: str) -> str:
    if timeframe.endswith("m"):
        return f"{int(timeframe[:-1])}min"
    return f"{int(timeframe[:-1])}h"


def _validate_timeframes(timeframes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(tf.strip().lower() for tf in timeframes if tf.strip())
    if not normalized:
        raise ValueError("at least one timeframe is required")
    below_floor = [tf for tf in normalized if _timeframe_hours(tf) < 0.5]
    if below_floor:
        raise ValueError(f"timeframes below 30m are not allowed: {below_floor}")
    return normalized


def max_asof_feature_age_hours(timeframe: str) -> float:
    return max(24.0, 6.0 * _timeframe_hours(timeframe))


def _aggregate_file_to_30m(path: Path) -> pl.DataFrame:
    """Aggregate one 1s parquet shard to 30m buckets before combining shards.

    The previous all-files lazy scan could exceed the 8GB session cap on the
    five-symbol default run. This per-file reduction keeps native 1s->30m bar
    construction while retaining only tiny monthly 30m partials in memory.
    """
    target_ms = 30 * 60 * 1000
    lf = pl.scan_parquet(str(path)).filter(
        (pl.col("datetime") >= pl.datetime(2025, 1, 1, 0, 0, 0))
        & (pl.col("datetime") <= pl.datetime(2026, 5, 17, 10, 0, 0))
    )
    return (
        lf.with_columns(pl.col("datetime").dt.epoch("ms").alias("ts_ms"))
        .with_columns(((pl.col("ts_ms") // target_ms) * target_ms).alias("bucket_ms"))
        .group_by("bucket_ms")
        .agg(
            [
                pl.col("open").sort_by("ts_ms").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").sort_by("ts_ms").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("ts_ms").count().alias("source_count"),
                pl.col("ts_ms").min().alias("source_first_ts_ms"),
                pl.col("ts_ms").max().alias("source_last_ts_ms"),
            ]
        )
        .collect(engine="streaming")
    )


def _load_symbol_base_30m(symbol: str, *, data_root: Path) -> pd.DataFrame:
    symbol_root = data_root / symbol
    files = sorted(symbol_root.glob("2025-*.parquet")) + sorted(symbol_root.glob("2026-*.parquet"))
    if not files:
        raise FileNotFoundError(f"missing OHLCV parquet files for {symbol}: {symbol_root}")
    partials: list[pl.DataFrame] = []
    for path in files:
        partial = _aggregate_file_to_30m(path)
        if not partial.is_empty():
            partials.append(partial)
        del partial
        gc.collect()
    if not partials:
        raise ValueError(f"no 30m bars collected for {symbol}")
    frame = (
        pl.concat(partials, how="vertical")
        .group_by("bucket_ms")
        .agg(
            [
                pl.col("open").sort_by("source_first_ts_ms").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").sort_by("source_last_ts_ms").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop_nulls(["open", "high", "low", "close"])
        .sort("bucket_ms")
        .with_columns(pl.from_epoch("bucket_ms", time_unit="ms").alias("datetime"))
        .select(["datetime", "open", "high", "low", "close", "volume"])
    )
    pdf = pd.DataFrame(frame.to_dicts())
    if pdf.empty:
        raise ValueError(f"no 30m bars collected for {symbol}")
    pdf["datetime"] = pd.to_datetime(pdf["datetime"])
    pdf["symbol"] = symbol
    return pdf


def _resample_from_base_30m(base_30m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "30m":
        return base_30m.copy().reset_index(drop=True)
    frame = base_30m.sort_values("datetime").set_index("datetime")
    resampled = frame.resample(_pandas_rule(timeframe), label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "symbol": "first",
        }
    )
    return resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()


def load_requested_bars(
    symbols: Sequence[str],
    *,
    timeframes: Sequence[str],
    data_root: Path,
) -> dict[tuple[str, str], pd.DataFrame]:
    valid_timeframes = _validate_timeframes(timeframes)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in symbols:
        base = _load_symbol_base_30m(symbol, data_root=data_root)
        for timeframe in valid_timeframes:
            out[(symbol, timeframe)] = _resample_from_base_30m(base, timeframe)
    return out


def _attach_features_with_age(
    bars: pd.DataFrame, features: pd.DataFrame, *, timeframe: str
) -> pd.DataFrame:
    out = bars.copy().sort_values("datetime")
    max_age = max_asof_feature_age_hours(timeframe)
    if features.empty:
        out["funding_rate"] = np.nan
        out["open_interest"] = np.nan
        out["taker_buy_sell_imbalance"] = np.nan
        out["feature_age_hours"] = np.inf
        out["feature_valid"] = False
        return out
    feats = features.copy().sort_values("datetime")
    buy = pd.to_numeric(feats.get("taker_buy_quote_volume"), errors="coerce")
    sell = pd.to_numeric(feats.get("taker_sell_quote_volume"), errors="coerce")
    if buy is not None and sell is not None:
        denom = buy.fillna(0.0) + sell.fillna(0.0)
        feats["taker_buy_sell_imbalance"] = np.where(
            denom > 0.0,
            (buy.fillna(0.0) - sell.fillna(0.0)) / denom,
            np.nan,
        )
    elif "taker_buy_sell_imbalance" not in feats.columns:
        feats["taker_buy_sell_imbalance"] = np.nan
    feats["funding_rate"] = pd.to_numeric(feats.get("funding_rate"), errors="coerce")
    feats["open_interest"] = pd.to_numeric(feats.get("open_interest"), errors="coerce")
    feats = feats.rename(columns={"datetime": "feature_datetime"})
    merged = pd.merge_asof(
        out,
        feats[["feature_datetime", "funding_rate", "open_interest", "taker_buy_sell_imbalance"]],
        left_on="datetime",
        right_on="feature_datetime",
        direction="backward",
    )
    age = (merged["datetime"] - merged["feature_datetime"]).dt.total_seconds() / 3600.0
    merged["feature_age_hours"] = age.fillna(np.inf)
    merged["feature_valid"] = (
        (merged["feature_age_hours"] <= max_age)
        & merged["funding_rate"].notna()
        & merged["open_interest"].notna()
        & merged["taker_buy_sell_imbalance"].notna()
    )
    return merged.drop(columns=["feature_datetime"])


def _feature_coverage_by_split(frame: pd.DataFrame) -> dict[str, float]:
    if "feature_valid" not in frame.columns:
        return dict.fromkeys(SPLIT_ORDER, 0.0)
    coverage: dict[str, float] = {}
    valid = frame["feature_valid"].fillna(False).astype(bool).to_numpy()
    for split in SPLIT_ORDER:
        mask = _split_mask(frame["datetime"], split)
        coverage[split] = float(np.mean(valid[mask])) if np.any(mask) else 0.0
    return coverage


def _feature_coverage_reasons(row: Mapping[str, Any]) -> list[str]:
    if not bool(row.get("feature_backed")):
        return []
    coverage = row.get("feature_coverage") or {}
    reasons: list[str] = []
    for split in SPLIT_ORDER:
        threshold = FEATURE_COVERAGE_THRESHOLDS[split]
        value = float(coverage.get(split, 0.0))
        if value < threshold:
            reasons.append(f"{split}_feature_coverage_{value:.3f}_below_{threshold:.2f}")
    return reasons


def _return_per_turnover(
    total_return: float, trade_events: int, notional_fraction: float
) -> float | None:
    turnover = float(trade_events) * abs(float(notional_fraction))
    if turnover <= 0.0:
        return None
    return float(total_return) * 10000.0 / turnover


def _candidate_score(row: Mapping[str, Any]) -> float:
    train = float(row.get("train_return") or 0.0)
    validation = float(row.get("validation_return") or 0.0)
    train_rpt = float(row.get("train_return_per_turnover_proxy_bps") or 0.0)
    val_rpt = float(row.get("validation_return_per_turnover_proxy_bps") or 0.0)
    capped_rpt_bonus = min(train_rpt, 40.0) / 400.0 + min(val_rpt, 40.0) / 300.0
    validation_spike_penalty = max(0.0, validation - train)
    turnover_penalty = 0.0002 * float(row.get("validation_trade_event_count") or 0.0)
    return (
        4.0 * validation
        + 1.5 * min(train, validation)
        + capped_rpt_bonus
        - 6.0 * validation_spike_penalty
        - 2.0 * float(row.get("validation_mdd") or 0.0)
        - turnover_penalty
    )


def _gate_candidate(row: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if int(row["train_trade_event_count"]) < PROMOTION_THRESHOLDS["min_train_trade_event_count"]:
        reasons.append(f"train_trade_event_count_{row['train_trade_event_count']}_below_80")
    if (
        int(row["validation_trade_event_count"])
        < PROMOTION_THRESHOLDS["min_validation_trade_event_count"]
    ):
        reasons.append(
            f"validation_trade_event_count_{row['validation_trade_event_count']}_below_30"
        )
    if (
        int(row["locked_oos_trade_event_count"])
        < PROMOTION_THRESHOLDS["min_locked_oos_trade_event_count_report_gate"]
    ):
        reasons.append(
            f"locked_oos_trade_event_count_{row['locked_oos_trade_event_count']}_below_20"
        )
    if float(row["validation_return"]) < PROMOTION_THRESHOLDS["min_validation_return"]:
        reasons.append(f"validation_return_{row['validation_return']:.4f}_below_0.02")
    if float(row["train_return"]) <= 0.0:
        reasons.append("train_return_not_positive")
    if float(row["train_return"]) < float(row["validation_return"]):
        reasons.append("train_return_below_validation_return")
    ratio = row.get("train_validation_return_ratio")
    if ratio is None or float(ratio) < PROMOTION_THRESHOLDS["min_train_validation_return_ratio"]:
        reasons.append(
            f"train_validation_return_ratio_{0.0 if ratio is None else float(ratio):.4f}_below_1.00"
        )
    if float(row["validation_mdd"]) > PROMOTION_THRESHOLDS["max_validation_mdd"]:
        reasons.append(f"validation_mdd_{row['validation_mdd']:.4f}_above_0.12")
    if float(row["locked_oos_return"]) <= 0.0:
        reasons.append("locked_oos_return_not_positive")
    if int(row["locked_oos_liquidation_count"]) != 0:
        reasons.append("locked_oos_liquidation_count_nonzero")
    if int(row["locked_oos_account_wipeout_count"]) != 0:
        reasons.append("locked_oos_account_wipeout_count_nonzero")

    feature_reasons = _feature_coverage_reasons(row)
    feature_coverage_gate_pass = not feature_reasons
    sample_gate_pass = not reasons

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
    paper_candidate_gate_pass = (
        sample_gate_pass and feature_coverage_gate_pass and execution_efficiency_proxy_gate_pass
    )
    if paper_candidate_gate_pass:
        decision = "paper_testnet_candidate_after_fill_preflight"
    elif sample_gate_pass and feature_coverage_gate_pass:
        decision = "train_dominant_shadow_until_execution_efficiency"
    else:
        decision = "no_promotion_shadow_or_reject"
    row.update(
        {
            "train_dominant_sample_gate_pass": sample_gate_pass,
            "feature_coverage_gate_pass": feature_coverage_gate_pass,
            "execution_efficiency_proxy_gate_pass": execution_efficiency_proxy_gate_pass,
            "primary_10bps_promotion_gate_pass": paper_candidate_gate_pass,
            "paper_candidate_gate_pass": paper_candidate_gate_pass,
            "decision": decision,
            "ready_for_paper": paper_candidate_gate_pass,
            "ready_for_real": False,
            "real_money_execution": False,
            "rejection_reasons": reasons + feature_reasons + efficiency_reasons,
        }
    )
    return row


def _finalize_candidate(
    base: dict[str, Any], sim: SimResult, datetimes: pd.Series, *, timeframe: str
) -> dict[str, Any]:
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
    row["train_validation_return_ratio"] = train / validation if validation > 0.0 else 0.0
    row["train_minus_validation_return"] = train - validation
    row["locked_oos_liquidation_count"] = int(split_payload["locked_oos"]["liquidation_count"])
    row["locked_oos_account_wipeout_count"] = int(
        split_payload["locked_oos"]["account_wipeout_count"]
    )
    notional = float(row["notional_fraction"])
    for split in SPLIT_ORDER:
        row[f"{split}_return_per_turnover_proxy_bps"] = _return_per_turnover(
            float(row[f"{split}_return"]),
            int(row[f"{split}_trade_event_count"]),
            notional,
        )
    coverage = row.get("feature_coverage") or {}
    row["feature_train_coverage"] = coverage.get("train")
    row["feature_validation_coverage"] = coverage.get("validation")
    row["feature_locked_oos_coverage"] = coverage.get("locked_oos")
    row["replay_live_notional_parity"] = {
        "recorded": True,
        "sizing_mode": "notional_fraction_equals_leverage_times_allocation_fraction",
        "replay_notional_fraction": notional,
        "live_notional_fraction": notional,
        "parity": True,
    }
    row["train_validation_score"] = _candidate_score(row)
    return _gate_candidate(row)


def _model_id(parts: Iterable[Any]) -> str:
    text = "_".join(str(part).replace("/", "_").replace(".", "p") for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"a30fb_{text}_{digest}".lower()


def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=high.index
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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(lookback).mean()


def _volatility_quantile_mask(
    close: pd.Series, lookback: int, quantile_max: float | None
) -> pd.Series:
    if quantile_max is None:
        return pd.Series(True, index=close.index)
    realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
    threshold = realized.rolling(max(24, lookback * 4)).quantile(quantile_max)
    return (realized <= threshold).fillna(False)


def _debounced_state_signal(
    long_entry: pd.Series,
    long_exit: pd.Series,
    short_entry: pd.Series,
    short_exit: pd.Series,
    *,
    side: str,
    min_hold_bars: int,
    cooldown_bars: int,
) -> np.ndarray:
    long_entry = long_entry.fillna(False).astype(bool)
    long_exit = long_exit.fillna(False).astype(bool)
    short_entry = short_entry.fillna(False).astype(bool)
    short_exit = short_exit.fillna(False).astype(bool)
    out = np.zeros(len(long_entry), dtype=float)
    state = 0.0
    bars_held = 10**9
    cooldown_remaining = 0
    for i in range(len(long_entry)):
        can_exit = bars_held >= min_hold_bars
        exited = False
        if can_exit and (
            (state > 0.0 and bool(long_exit.iloc[i])) or (state < 0.0 and bool(short_exit.iloc[i]))
        ):
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
        FilterProfile("none", 0.0, None),
        FilterProfile("low_vol_q70", 0.0, 0.70),
        FilterProfile("adx15", 15.0, None),
        FilterProfile("adx20_low_vol_q75", 20.0, 0.75),
    )


def _base_feature_fields(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    valid = (
        frame.get("feature_valid", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    )
    funding = pd.to_numeric(
        frame.get("funding_rate", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    oi = pd.to_numeric(
        frame.get("open_interest", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    imbalance = pd.to_numeric(
        frame.get("taker_buy_sell_imbalance", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    return valid, funding, oi, imbalance


def discover_candidates(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    leverage_allocs = ((2.0, 0.15), (3.0, 0.10), (3.0, 0.15))
    min_holds = (4, 8, 12)
    cooldowns = (0, 2)

    for timeframe in timeframes:
        btc = bars_by_symbol_tf[(BTC_REGIME_SYMBOL, timeframe)][["datetime", "close"]].rename(
            columns={"close": "btc_close"}
        )
        for symbol in symbols:
            frame = (
                bars_by_symbol_tf[(symbol, timeframe)].merge(btc, on="datetime", how="left").ffill()
            )
            close = frame["close"].astype(float)
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            btc_close = frame["btc_close"].astype(float)
            btc_momentum = (
                btc_close / btc_close.shift(max(2, int(12 / _timeframe_hours(timeframe)))) - 1.0
            )

            for lookback in (6, 12, 24):
                momentum = close / close.shift(lookback) - 1.0
                realized = close.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
                vol_adjusted = momentum / (realized * np.sqrt(float(lookback))).replace(0.0, np.nan)
                adx = _adx_proxy(high, low, close, max(6, lookback))
                for threshold in (1.0, 1.5, 2.0):
                    for min_hold in min_holds:
                        for cooldown in cooldowns:
                            for profile in _filter_profiles():
                                vol_ok = _volatility_quantile_mask(
                                    close, lookback, profile.max_vol_quantile
                                )
                                adx_ok = adx >= profile.min_adx if profile.min_adx > 0.0 else True
                                common = vol_ok & adx_ok
                                long_entry = (
                                    (vol_adjusted > threshold) & (btc_momentum > -0.02) & common
                                )
                                short_entry = (
                                    (vol_adjusted < -threshold) & (btc_momentum < 0.02) & common
                                )
                                long_exit = (vol_adjusted < 0.25) | (~common)
                                short_exit = (vol_adjusted > -0.25) | (~common)
                                signal = _debounced_state_signal(
                                    long_entry,
                                    long_exit,
                                    short_entry,
                                    short_exit,
                                    side="long_short",
                                    min_hold_bars=min_hold,
                                    cooldown_bars=cooldown,
                                )
                                for leverage, allocation in leverage_allocs:
                                    base = {
                                        "model_id": _model_id(
                                            [
                                                "voladj_trend",
                                                timeframe,
                                                symbol,
                                                f"lb{lookback}",
                                                f"z{threshold}",
                                                f"hold{min_hold}",
                                                f"cool{cooldown}",
                                                profile.label,
                                                f"{leverage}x",
                                                allocation,
                                            ]
                                        ),
                                        "family": "volatility_adjusted_trend_persistence",
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "side": "long_short",
                                        "lookback_bars": lookback,
                                        "threshold": threshold,
                                        "exit_threshold": 0.25,
                                        "min_hold_bars": min_hold,
                                        "cooldown_bars": cooldown,
                                        "filter_label": profile.label,
                                        "feature_backed": False,
                                        "feature_coverage": {},
                                        "max_asof_feature_age_hours": None,
                                        "leverage": leverage,
                                        "allocation_fraction": allocation,
                                        "notional_fraction": leverage * allocation,
                                    }
                                    sim = simulate_symbol(
                                        frame,
                                        signal,
                                        leverage=leverage,
                                        allocation_fraction=allocation,
                                    )
                                    rows.append(
                                        _finalize_candidate(
                                            base, sim, frame["datetime"], timeframe=timeframe
                                        )
                                    )

            for lookback in (12, 24, 48):
                atr = _atr(high, low, close, max(6, lookback))
                rolling_high = high.shift(1).rolling(lookback).max()
                rolling_low = low.shift(1).rolling(lookback).min()
                rolling_mid = (rolling_high + rolling_low) / 2.0
                for atr_mult in (0.25, 0.50):
                    long_entry = (close > rolling_high + atr_mult * atr) & (btc_momentum > -0.01)
                    short_entry = (close < rolling_low - atr_mult * atr) & (btc_momentum < 0.01)
                    long_exit = close < rolling_mid
                    short_exit = close > rolling_mid
                    for min_hold in min_holds:
                        signal = _debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage, allocation in leverage_allocs:
                            base = {
                                "model_id": _model_id(
                                    [
                                        "donchian_atr",
                                        timeframe,
                                        symbol,
                                        f"lb{lookback}",
                                        f"atr{atr_mult}",
                                        f"hold{min_hold}",
                                        f"{leverage}x",
                                        allocation,
                                    ]
                                ),
                                "family": "donchian_atr_volatility_breakout",
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "side": "long_short",
                                "lookback_bars": lookback,
                                "threshold": atr_mult,
                                "exit_threshold": 0.0,
                                "min_hold_bars": min_hold,
                                "cooldown_bars": 2,
                                "filter_label": "btc_regime_mid_exit",
                                "feature_backed": False,
                                "feature_coverage": {},
                                "max_asof_feature_age_hours": None,
                                "leverage": leverage,
                                "allocation_fraction": allocation,
                                "notional_fraction": leverage * allocation,
                            }
                            sim = simulate_symbol(
                                frame, signal, leverage=leverage, allocation_fraction=allocation
                            )
                            rows.append(
                                _finalize_candidate(
                                    base, sim, frame["datetime"], timeframe=timeframe
                                )
                            )

            for fast, slow in ((6, 24), (12, 48)):
                ema_fast = close.ewm(span=fast, adjust=False).mean()
                ema_slow = close.ewm(span=slow, adjust=False).mean()
                slope = ema_slow / ema_slow.shift(max(2, fast)) - 1.0
                adx = _adx_proxy(high, low, close, max(6, fast))
                for adx_threshold in (15.0, 20.0):
                    common = adx >= adx_threshold
                    long_entry = (
                        (ema_fast > ema_slow) & (slope > 0.0) & (btc_momentum > -0.02) & common
                    )
                    short_entry = (
                        (ema_fast < ema_slow) & (slope < 0.0) & (btc_momentum < 0.02) & common
                    )
                    long_exit = (ema_fast < ema_slow) | (~common)
                    short_exit = (ema_fast > ema_slow) | (~common)
                    signal = _debounced_state_signal(
                        long_entry,
                        long_exit,
                        short_entry,
                        short_exit,
                        side="long_short",
                        min_hold_bars=8,
                        cooldown_bars=2,
                    )
                    for leverage, allocation in leverage_allocs:
                        base = {
                            "model_id": _model_id(
                                [
                                    "ma_slope_adx",
                                    timeframe,
                                    symbol,
                                    f"fast{fast}",
                                    f"slow{slow}",
                                    f"adx{adx_threshold}",
                                    f"{leverage}x",
                                    allocation,
                                ]
                            ),
                            "family": "ma_slope_adx_trend_filter",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "side": "long_short",
                            "lookback_bars": slow,
                            "threshold": adx_threshold,
                            "exit_threshold": 0.0,
                            "min_hold_bars": 8,
                            "cooldown_bars": 2,
                            "filter_label": "btc_regime_ema_cross",
                            "feature_backed": False,
                            "feature_coverage": {},
                            "max_asof_feature_age_hours": None,
                            "leverage": leverage,
                            "allocation_fraction": allocation,
                            "notional_fraction": leverage * allocation,
                        }
                        sim = simulate_symbol(
                            frame, signal, leverage=leverage, allocation_fraction=allocation
                        )
                        rows.append(
                            _finalize_candidate(base, sim, frame["datetime"], timeframe=timeframe)
                        )

            feature_valid, funding, oi, imbalance = _base_feature_fields(frame)
            feature_coverage = _feature_coverage_by_split(frame)
            for lookback in (6, 12):
                momentum = close / close.shift(lookback) - 1.0
                oi_change = oi / oi.shift(lookback) - 1.0
                for funding_abs_max in (0.00010, 0.00025):
                    for imbalance_threshold in (0.0, 0.05):
                        long_entry = (
                            (momentum > 0.01)
                            & (oi_change > 0.0)
                            & (imbalance > imbalance_threshold)
                            & (funding.abs() <= funding_abs_max)
                            & feature_valid
                            & (btc_momentum > -0.02)
                        )
                        short_entry = (
                            (momentum < -0.01)
                            & (oi_change > 0.0)
                            & (imbalance < -imbalance_threshold)
                            & (funding.abs() <= funding_abs_max)
                            & feature_valid
                            & (btc_momentum < 0.02)
                        )
                        long_exit = (momentum < 0.0) | (~feature_valid)
                        short_exit = (momentum > 0.0) | (~feature_valid)
                        signal = _debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=8,
                            cooldown_bars=2,
                        )
                        for leverage, allocation in leverage_allocs:
                            base = {
                                "model_id": _model_id(
                                    [
                                        "feature_crowding",
                                        timeframe,
                                        symbol,
                                        f"lb{lookback}",
                                        f"fund{funding_abs_max}",
                                        f"imb{imbalance_threshold}",
                                        f"{leverage}x",
                                        allocation,
                                    ]
                                ),
                                "family": "funding_oi_taker_crowding_continuation",
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "side": "long_short",
                                "lookback_bars": lookback,
                                "threshold": imbalance_threshold,
                                "exit_threshold": 0.0,
                                "min_hold_bars": 8,
                                "cooldown_bars": 2,
                                "filter_label": "feature_valid_btc_regime",
                                "feature_backed": True,
                                "feature_coverage": feature_coverage,
                                "max_asof_feature_age_hours": max_asof_feature_age_hours(timeframe),
                                "leverage": leverage,
                                "allocation_fraction": allocation,
                                "notional_fraction": leverage * allocation,
                            }
                            sim = simulate_symbol(
                                frame, signal, leverage=leverage, allocation_fraction=allocation
                            )
                            rows.append(
                                _finalize_candidate(
                                    base, sim, frame["datetime"], timeframe=timeframe
                                )
                            )
    return rows


def _rank_rows(rows: Sequence[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    ranked = sorted(
        rows, key=lambda row: float(row.get("train_validation_score") or -1e9), reverse=True
    )
    if limit is not None:
        ranked = ranked[:limit]
    return [dict(row, rank=rank) for rank, row in enumerate(ranked, start=1)]


def _selected_output_rows(
    ranked_rows: Sequence[dict[str, Any]], *, top_n: int
) -> list[dict[str, Any]]:
    selected_ids = {str(row["model_id"]) for row in ranked_rows[:top_n]}
    for row in ranked_rows:
        if any(bool(row.get(flag)) for flag in SPECIAL_OUTPUT_FLAGS):
            selected_ids.add(str(row["model_id"]))
    if ranked_rows:
        best_validation = max(
            ranked_rows, key=lambda row: float(row.get("validation_return") or -1e9)
        )
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
    shadows = [
        row
        for row in ranked_rows
        if row.get("decision") != "paper_testnet_candidate_after_fill_preflight"
    ]
    sample_shadows = [row for row in shadows if row.get("train_dominant_sample_gate_pass")]
    if sample_shadows:
        return _selected_output_rows(sample_shadows, top_n=limit)
    return _selected_output_rows(shadows, top_n=limit)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions: dict[str, int] = {}
    by_symbol: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for row in rows:
        decisions[str(row["decision"])] = decisions.get(str(row["decision"]), 0) + 1
        by_symbol[str(row["symbol"])] = by_symbol.get(str(row["symbol"]), 0) + 1
        by_family[str(row["family"])] = by_family.get(str(row["family"]), 0) + 1
    best_validation = (
        max(rows, key=lambda row: float(row.get("validation_return") or -1e9)) if rows else {}
    )
    sample_pass = [row for row in rows if row.get("train_dominant_sample_gate_pass")]
    paper_candidates = [row for row in rows if row.get("paper_candidate_gate_pass")]
    return {
        "candidate_count": len(rows),
        "symbol_counts": dict(sorted(by_symbol.items())),
        "family_counts": dict(sorted(by_family.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "train_return_gte_validation_return_count": sum(
            float(row.get("train_return") or 0.0) >= float(row.get("validation_return") or 0.0)
            for row in rows
        ),
        "train_dominant_sample_gate_pass_count": len(sample_pass),
        "feature_coverage_gate_pass_count": sum(
            bool(row.get("feature_coverage_gate_pass")) for row in rows
        ),
        "execution_efficiency_proxy_gate_pass_count": sum(
            bool(row.get("execution_efficiency_proxy_gate_pass")) for row in rows
        ),
        "paper_candidate_gate_pass_count": len(paper_candidates),
        "max_validation_return": float(best_validation.get("validation_return") or 0.0)
        if best_validation
        else None,
        "best_validation_model_id": best_validation.get("model_id") if best_validation else None,
        "best_train_dominant_sample_gate_model_id": sample_pass[0].get("model_id")
        if sample_pass
        else None,
        "best_paper_candidate_model_id": paper_candidates[0].get("model_id")
        if paper_candidates
        else None,
        "ready_for_paper": bool(paper_candidates),
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _paper_testnet_handoff(paper_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [dict(row) for row in paper_rows]
    return {
        "handoff_kind": "paper_testnet_only_30m_plus_alpha_feedback_discovery",
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


def _handoff_markdown(handoff: Mapping[str, Any]) -> str:
    candidates = list(handoff.get("candidates") or [])
    lines = [
        "# Paper/testnet handoff — 30m+ Alpha Zoo feedback discovery",
        "",
        f"- Status: `{handoff.get('status')}`",
        f"- Candidate count: `{handoff.get('candidate_count')}`",
        f"- `ready_for_paper={str(handoff.get('ready_for_paper')).lower()}`",
        "- `ready_for_real=false`",
        "- `real_money_execution=false`",
        "- Real-money execution remains prohibited; this handoff is paper/testnet-only.",
        "",
        "| Rank | Model | Symbol | TF | Family | Train | Val | OOS | RPT train/val/OOS |",
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
            f"| {row.get('rank')} | `{row.get('model_id')}` | {row.get('symbol')} | {row.get('timeframe')} | "
            f"{row.get('family')} | {float(row.get('train_return') or 0.0):.4%} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {rpt} |"
        )
    return "\n".join(lines) + "\n"


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("discovery_summary") or {})
    top = list(payload.get("top_candidates") or [])[:10]
    paper_rows = [
        row for row in payload.get("top_candidates", []) if row.get("paper_candidate_gate_pass")
    ][:10]
    lines = [
        "# Alpha Zoo 30m+ feedback discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "New >=30m Alpha Zoo discovery pass using native 30m bar construction and train+validation ranking.",
        "Locked-OOS is gate/report-only after candidate freeze. Real-money remains blocked.",
        "",
        "## Summary",
        "",
        f"- Candidates evaluated: `{summary.get('candidate_count')}`",
        f"- Train-dominant sample gate pass: `{summary.get('train_dominant_sample_gate_pass_count')}`",
        f"- Execution-efficiency proxy gate pass: `{summary.get('execution_efficiency_proxy_gate_pass_count')}`",
        f"- Full paper candidate gate pass: `{summary.get('paper_candidate_gate_pass_count')}`",
        f"- Decision: `{payload.get('decision_status')}`",
        f"- Runner peak RSS MiB: `{float(payload.get('runner_peak_rss_mib') or 0.0):.3f}`",
        "- `ready_for_real=false`, `real_money_execution=false`",
        "",
        "## Top train+validation-ranked rows",
        "",
        "| Rank | Symbol | TF | Family | Train | Val | OOS | RPT train/val/OOS | Decision |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in top:
        rpt = "/".join(
            "NA"
            if row.get(f"{split}_return_per_turnover_proxy_bps") is None
            else f"{float(row[f'{split}_return_per_turnover_proxy_bps']):.2f}"
            for split in SPLIT_ORDER
        )
        lines.append(
            f"| {row.get('rank')} | {row.get('symbol')} | {row.get('timeframe')} | {row.get('family')} | "
            f"{float(row.get('train_return') or 0.0):.4%} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {rpt} | {row.get('decision')} |"
        )
    lines.extend(
        [
            "",
            "## Paper/testnet-only candidates",
            "",
            "| Rank | Symbol | TF | Family | Train | Val | OOS | Trades | RPT train/val/OOS |",
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
            f"| {row.get('rank')} | {row.get('symbol')} | {row.get('timeframe')} | {row.get('family')} | "
            f"{float(row.get('train_return') or 0.0):.4%} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {trades} | {rpt} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    prior_artifact = Path(args.prior_artifact).expanduser().resolve()
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    timeframes = _validate_timeframes(
        tuple(tf.strip().lower() for tf in args.timeframes.split(",") if tf.strip())
    )
    data_symbols = tuple(dict.fromkeys((BTC_REGIME_SYMBOL, *symbols)))

    bars_by_symbol_tf = load_requested_bars(
        data_symbols, timeframes=timeframes, data_root=data_root
    )
    for symbol in data_symbols:
        features = load_feature_points(symbol, feature_root=feature_root)
        for timeframe in timeframes:
            bars_by_symbol_tf[(symbol, timeframe)] = _attach_features_with_age(
                bars_by_symbol_tf[(symbol, timeframe)],
                features,
                timeframe=timeframe,
            )

    rows = discover_candidates(bars_by_symbol_tf, symbols=symbols, timeframes=timeframes)
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
    latest_json = output_dir / "alpha_zoo_30m_plus_alpha_feedback_discovery_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_30m_plus_alpha_feedback_discovery_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_30m_plus_alpha_feedback_discovery_latest.md"
    candidates_csv = output_dir / "alpha_zoo_30m_plus_alpha_feedback_candidates_latest.csv"
    decisions_csv = output_dir / "alpha_zoo_30m_plus_alpha_feedback_decisions_latest.csv"
    shadow_csv = output_dir / "alpha_zoo_30m_plus_alpha_feedback_shadow_hypotheses_latest.csv"
    no_promotion_json = output_dir / "no_promotion_shadow_shortlist_latest.json"
    handoff_json = output_dir / "paper_testnet_handoff_latest.json"
    handoff_md = output_dir / "paper_testnet_handoff_latest.md"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_30m_plus_alpha_feedback_discovery",
        "generated_at_utc": _utc_now_iso(),
        "bar_construction": BAR_CONSTRUCTION,
        "strategy_scope": STRATEGY_SCOPE,
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
        "external_research_references": EXTERNAL_RESEARCH_REFERENCES,
        "baseline_lanes": BASELINE_LANES,
        "source_data": {
            "ohlcv_root": str(data_root),
            "feature_root": str(feature_root),
            "symbols": list(symbols),
            "timeframes": list(timeframes),
            "bar_source": "local Binance 1s OHLCV parquet aggregated to native 30m base",
        },
        "bounded_grid_policy": {
            "max_default_symbols": len(DEFAULT_SYMBOLS),
            "default_timeframes": list(DEFAULT_TIMEFRAMES),
            "max_output_top_n_default": 220,
            "multiprocessing_default": False,
            "raw_1s_symbols_loaded_concurrently": 1,
        },
        "feature_coverage_policy": {
            "min_train_feature_coverage": MIN_TRAIN_FEATURE_COVERAGE,
            "min_validation_feature_coverage": MIN_VALIDATION_FEATURE_COVERAGE,
            "min_locked_oos_feature_coverage": MIN_LOCKED_OOS_FEATURE_COVERAGE,
            "max_asof_feature_age_hours_formula": "max(24, 6*timeframe_hours)",
            "locked_oos_role": "post_freeze_report_gate_only_rejects_paper_promotion_not_selection",
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
        "strategy_families": [
            "volatility_adjusted_trend_persistence",
            "donchian_atr_volatility_breakout",
            "ma_slope_adx_trend_filter",
            "funding_oi_taker_crowding_continuation",
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
                f"bar_construction={BAR_CONSTRUCTION}",
                f"strategy_scope={STRATEGY_SCOPE}",
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
                f"paper_testnet_handoff_json={handoff_json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--prior-artifact", default=str(DEFAULT_PRIOR_ARTIFACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--top-n", type=int, default=220)
    parser.add_argument("--decision-top-n", type=int, default=100)
    parser.add_argument("--shadow-top-n", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
