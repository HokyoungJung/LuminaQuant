#!/usr/bin/env python3
"""Discover new 30m+ Alpha Zoo momentum/crowding families on local real data.

This runner is intentionally research/paper-testnet only. It ranks candidates with
train+validation evidence and attaches locked-OOS strictly after candidate freeze as
report/gate evidence. It does not execute orders and never enables real money.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data/market_parquet/market_ohlcv_1s/binance"
DEFAULT_FEATURE_ROOT = REPO_ROOT / "data/market_parquet/feature_points/exchange=binance"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_htf_momentum_crowding_discovery_20260522"
)
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TRXUSDT")
SPLITS = {
    "train": (pd.Timestamp("2025-01-01 00:00:00"), pd.Timestamp("2025-12-31 23:00:00")),
    "validation": (pd.Timestamp("2026-01-01 00:00:00"), pd.Timestamp("2026-03-31 23:00:00")),
    "locked_oos": (pd.Timestamp("2026-04-01 00:00:00"), pd.Timestamp("2026-05-17 10:00:00")),
}
SPLIT_ORDER = ("train", "validation", "locked_oos")
PRIMARY_ROUND_TRIP_COST_BPS = 10.0
AVG_BBO_SPREAD_BPS_ASSUMPTION = 2.0
BBO_SPREAD_MULTIPLIER = 5.0
RETURN_PER_TURNOVER_THRESHOLD_BPS = AVG_BBO_SPREAD_BPS_ASSUMPTION * BBO_SPREAD_MULTIPLIER

PROMOTION_THRESHOLDS = {
    "min_train_trade_event_count": 80,
    "min_validation_trade_event_count": 30,
    "min_locked_oos_trade_event_count_report_gate": 20,
    "min_validation_return": 0.02,
    "min_train_validation_return_ratio": 0.50,
    "max_validation_mdd": 0.12,
    "require_train_return_positive": True,
    "require_locked_oos_return_positive_report_gate": True,
    "require_zero_locked_oos_liquidation": True,
    "require_zero_locked_oos_account_wipeout": True,
    "min_return_per_turnover_proxy_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
}

CANDIDATE_FIELDS = [
    "rank",
    "model_id",
    "family",
    "symbol",
    "timeframe",
    "side",
    "lookback_bars",
    "threshold",
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
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "locked_oos_return_per_turnover_proxy_bps",
    "backtest_sample_gate_pass",
    "execution_efficiency_proxy_gate_pass",
    "paper_candidate_gate_pass",
    "decision",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
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
    "validation_return",
    "locked_oos_return",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "rejection_reasons",
]

SPECIAL_OUTPUT_FLAGS = (
    "backtest_sample_gate_pass",
    "execution_efficiency_proxy_gate_pass",
    "paper_candidate_gate_pass",
)

SOURCE_REFERENCES = [
    {
        "label": "AdaptiveTrend crypto trend-following research",
        "url": "https://arxiv.org/abs/2602.11708",
        "usage": "motivates longer 6h-style trend-following / adaptive momentum research lanes",
    },
    {
        "label": "Binance USDⓈ-M funding rate history docs",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History",
        "usage": "identifies available perp funding data for future/live crowding overlays",
    },
    {
        "label": "Binance USDⓈ-M open interest docs",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest",
        "usage": "identifies open-interest support data used by local feature-point stores when available",
    },
    {
        "label": "Binance USDⓈ-M taker buy/sell volume docs",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume",
        "usage": "documents future order-flow/crowding expansion input; local coverage is incomplete for all symbols",
    },
]


@dataclass(frozen=True)
class SimResult:
    returns: np.ndarray
    position: np.ndarray
    liquidation_flags: np.ndarray
    account_wipeout_flags: np.ndarray


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


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


def _split_mask(index: pd.Series | pd.DatetimeIndex, split: str) -> np.ndarray:
    start, end = SPLITS[split]
    values = pd.Series(index) if not isinstance(index, pd.Series) else index
    return ((values >= start) & (values <= end)).to_numpy()


def _hours_for_timeframe(timeframe: str) -> float:
    if timeframe.endswith("m"):
        return float(timeframe[:-1]) / 60.0
    if timeframe.endswith("h"):
        return float(timeframe[:-1])
    raise ValueError(f"unsupported timeframe {timeframe!r}")


def _pandas_rule(timeframe: str) -> str:
    if timeframe.endswith("m"):
        return f"{int(timeframe[:-1])}min"
    return f"{int(timeframe[:-1])}h"


def _load_symbol_hourly(symbol: str, *, data_root: Path) -> pd.DataFrame:
    symbol_root = data_root / symbol
    files = sorted(symbol_root.glob("2025-*.parquet")) + sorted(symbol_root.glob("2026-*.parquet"))
    if not files:
        raise FileNotFoundError(f"missing OHLCV 1s parquet files for {symbol}: {symbol_root}")
    lf = pl.scan_parquet([str(path) for path in files]).filter(
        (pl.col("datetime") >= pl.datetime(2025, 1, 1, 0, 0, 0))
        & (pl.col("datetime") <= pl.datetime(2026, 5, 17, 10, 0, 0))
    )
    frame = (
        lf.group_by_dynamic("datetime", every="1h", period="1h")
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .drop_nulls(["open", "high", "low", "close"])
        .sort("datetime")
        .collect()
    )
    pdf = pd.DataFrame(frame.to_dicts())
    if pdf.empty:
        raise ValueError(f"no hourly bars collected for {symbol}")
    pdf["datetime"] = pd.to_datetime(pdf["datetime"])
    pdf["symbol"] = symbol
    return pdf


def load_hourly_bars(symbols: Sequence[str], *, data_root: Path) -> dict[str, pd.DataFrame]:
    return {symbol: _load_symbol_hourly(symbol, data_root=data_root) for symbol in symbols}


def resample_bars(hourly: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = hourly.sort_values("datetime").set_index("datetime")
    if timeframe != "1h":
        frame = frame.resample(_pandas_rule(timeframe), label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
    frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return frame


def load_feature_points(symbol: str, *, feature_root: Path) -> pd.DataFrame:
    symbol_root = feature_root / f"symbol={symbol}"
    files = sorted(symbol_root.rglob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["datetime", "funding_rate", "open_interest"])
    lf = pl.scan_parquet([str(path) for path in files]).select(
        [
            "timestamp_ms",
            "funding_rate",
            "open_interest",
            "taker_buy_quote_volume",
            "taker_sell_quote_volume",
        ]
    )
    frame = lf.collect()
    pdf = pd.DataFrame(frame.to_dicts())
    if pdf.empty:
        return pd.DataFrame(columns=["datetime", "funding_rate", "open_interest"])
    pdf["datetime"] = pd.to_datetime(pdf["timestamp_ms"], unit="ms")
    return pdf.sort_values("datetime")


def attach_asof_features(bars: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        out = bars.copy()
        out["funding_rate"] = np.nan
        out["open_interest"] = np.nan
        out["taker_buy_sell_imbalance"] = np.nan
        return out
    feats = features.copy()
    buy = pd.to_numeric(feats.get("taker_buy_quote_volume"), errors="coerce")
    sell = pd.to_numeric(feats.get("taker_sell_quote_volume"), errors="coerce")
    if buy is not None and sell is not None:
        denom = buy.fillna(0.0) + sell.fillna(0.0)
        feats["taker_buy_sell_imbalance"] = np.where(
            denom > 0, (buy.fillna(0.0) - sell.fillna(0.0)) / denom, np.nan
        )
    else:
        feats["taker_buy_sell_imbalance"] = np.nan
    feats["funding_rate"] = pd.to_numeric(feats["funding_rate"], errors="coerce")
    feats["open_interest"] = pd.to_numeric(feats["open_interest"], errors="coerce")
    return pd.merge_asof(
        bars.sort_values("datetime"),
        feats[
            ["datetime", "funding_rate", "open_interest", "taker_buy_sell_imbalance"]
        ].sort_values("datetime"),
        on="datetime",
        direction="backward",
    )


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1.0 - (equity / np.maximum(peaks, 1e-12))
    return float(np.max(drawdowns)) if drawdowns.size else 0.0


def split_metrics(
    returns: np.ndarray,
    position: np.ndarray,
    liquidation_flags: np.ndarray,
    account_wipeout_flags: np.ndarray,
    *,
    timeframe: str,
) -> dict[str, float | int]:
    if returns.size == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "trade_event_count": 0,
            "exposure_bar_count": 0,
            "liquidation_count": 0,
            "account_wipeout_count": 0,
        }
    equity_return = float(np.prod(1.0 + returns) - 1.0)
    mdd = max_drawdown(returns)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    downside = returns[returns < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    periods_per_year = 365.0 * 24.0 / _hours_for_timeframe(timeframe)
    sharpe = mean / std * math.sqrt(periods_per_year) if std > 0 else 0.0
    sortino = mean / downside_std * math.sqrt(periods_per_year) if downside_std > 0 else 0.0
    calmar = equity_return / mdd if mdd > 0 else 0.0
    trade_events = int(np.count_nonzero(np.abs(np.diff(np.r_[0.0, position])) > 1e-12))
    return {
        "total_return": equity_return,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "trade_event_count": trade_events,
        "exposure_bar_count": int(np.count_nonzero(np.abs(position) > 1e-12)),
        "liquidation_count": int(np.count_nonzero(liquidation_flags)),
        "account_wipeout_count": int(np.count_nonzero(account_wipeout_flags)),
    }


def simulate_symbol(
    bars: pd.DataFrame,
    signal: np.ndarray,
    *,
    leverage: float,
    allocation_fraction: float,
    round_trip_cost_bps: float = PRIMARY_ROUND_TRIP_COST_BPS,
) -> SimResult:
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    signal = np.asarray(signal, dtype=float)
    if signal.size != close.size:
        raise ValueError("signal length must equal bars length")
    next_return = np.r_[np.diff(close) / np.maximum(close[:-1], 1e-12), 0.0]
    notional = leverage * allocation_fraction
    transition = np.abs(np.diff(np.r_[0.0, signal]))
    costs = (round_trip_cost_bps / 10000.0) * notional * transition / 2.0
    returns = signal * notional * next_return - costs
    long_liq = (signal > 0) & (((low / np.maximum(close, 1e-12)) - 1.0) * leverage <= -0.95)
    short_liq = (signal < 0) & (((high / np.maximum(close, 1e-12)) - 1.0) * leverage >= 0.95)
    liquidation = long_liq | short_liq
    equity = np.cumprod(1.0 + returns)
    wipeout = equity <= 0.0
    return SimResult(
        returns=returns,
        position=signal,
        liquidation_flags=liquidation,
        account_wipeout_flags=wipeout,
    )


def simulate_portfolio(
    close: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    leverage: float,
    allocation_fraction: float,
    round_trip_cost_bps: float = PRIMARY_ROUND_TRIP_COST_BPS,
) -> SimResult:
    aligned = close.sort_index()
    sig = signals.reindex(aligned.index).fillna(0.0).astype(float)
    pct = aligned.pct_change().shift(-1).fillna(0.0)
    notional = leverage * allocation_fraction
    gross = (sig * notional * pct).sum(axis=1).to_numpy(dtype=float)
    turnover = sig.diff().abs().sum(axis=1).fillna(sig.abs().sum(axis=1)).to_numpy(dtype=float)
    costs = (round_trip_cost_bps / 10000.0) * notional * turnover / 2.0
    returns = gross - costs
    position = (sig.abs().sum(axis=1) > 1e-12).astype(float).to_numpy()
    equity = np.cumprod(1.0 + returns)
    return SimResult(
        returns=returns,
        position=position,
        liquidation_flags=np.zeros_like(returns, dtype=bool),
        account_wipeout_flags=equity <= 0.0,
    )


def _return_per_turnover(
    total_return: float, trade_events: int, notional_fraction: float
) -> float | None:
    turnover = float(trade_events) * abs(float(notional_fraction))
    if turnover <= 0:
        return None
    return float(total_return) * 10000.0 / turnover


def _candidate_score(row: Mapping[str, Any]) -> float:
    return (
        6.0 * float(row.get("validation_return") or 0.0)
        + 0.5 * float(row.get("train_return") or 0.0)
        - 2.0 * float(row.get("validation_mdd") or 0.0)
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
    if float(row["train_return"]) <= 0:
        reasons.append("train_return_not_positive")
    ratio = row.get("train_validation_return_ratio")
    if ratio is None or float(ratio) < PROMOTION_THRESHOLDS["min_train_validation_return_ratio"]:
        reasons.append(
            f"train_validation_return_ratio_{0.0 if ratio is None else ratio:.4f}_below_0.50"
        )
    if float(row["validation_mdd"]) > PROMOTION_THRESHOLDS["max_validation_mdd"]:
        reasons.append(f"validation_mdd_{row['validation_mdd']:.4f}_above_0.12")
    if float(row["locked_oos_return"]) <= 0:
        reasons.append("locked_oos_return_not_positive")
    if int(row["locked_oos_liquidation_count"]) != 0:
        reasons.append("locked_oos_liquidation_count_nonzero")
    if int(row["locked_oos_account_wipeout_count"]) != 0:
        reasons.append("locked_oos_account_wipeout_count_nonzero")
    backtest_sample_gate_pass = not reasons

    efficiency_reasons: list[str] = []
    for split in SPLIT_ORDER:
        value = row.get(f"{split}_return_per_turnover_proxy_bps")
        if value is None or float(value) <= RETURN_PER_TURNOVER_THRESHOLD_BPS:
            rendered = "missing" if value is None else f"{float(value):.3f}"
            efficiency_reasons.append(
                f"{split}_return_per_turnover_proxy_bps_{rendered}_not_above_{RETURN_PER_TURNOVER_THRESHOLD_BPS:.3f}"
            )
    execution_efficiency_proxy_gate_pass = not efficiency_reasons
    paper_candidate_gate_pass = backtest_sample_gate_pass and execution_efficiency_proxy_gate_pass
    if paper_candidate_gate_pass:
        decision = "paper_testnet_candidate_after_fill_preflight"
    elif backtest_sample_gate_pass:
        decision = "validation_alpha_shadow_until_execution_efficiency"
    else:
        decision = "no_promotion_shadow_or_reject"
    row.update(
        {
            "backtest_sample_gate_pass": backtest_sample_gate_pass,
            "execution_efficiency_proxy_gate_pass": execution_efficiency_proxy_gate_pass,
            "paper_candidate_gate_pass": paper_candidate_gate_pass,
            "decision": decision,
            "ready_for_paper": paper_candidate_gate_pass,
            "ready_for_real": False,
            "real_money_execution": False,
            "rejection_reasons": reasons + efficiency_reasons,
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
        prefix = split
        row[f"{prefix}_return"] = metrics["total_return"]
        row[f"{prefix}_mdd"] = metrics["max_drawdown"]
        row[f"{prefix}_sharpe"] = metrics["sharpe"]
        row[f"{prefix}_sortino"] = metrics["sortino"]
        row[f"{prefix}_calmar"] = metrics["calmar"]
        row[f"{prefix}_trade_event_count"] = metrics["trade_event_count"]
        row[f"{prefix}_exposure_bar_count"] = metrics["exposure_bar_count"]
    validation = float(row.get("validation_return") or 0.0)
    train = float(row.get("train_return") or 0.0)
    row["train_validation_return_ratio"] = train / validation if validation > 0 else 0.0
    row["locked_oos_liquidation_count"] = int(split_payload["locked_oos"]["liquidation_count"])
    row["locked_oos_account_wipeout_count"] = int(
        split_payload["locked_oos"]["account_wipeout_count"]
    )
    notional = float(row["notional_fraction"])
    for split in SPLIT_ORDER:
        row[f"{split}_return_per_turnover_proxy_bps"] = _return_per_turnover(
            float(row[f"{split}_return"]), int(row[f"{split}_trade_event_count"]), notional
        )
    row["train_validation_score"] = _candidate_score(row)
    return _gate_candidate(row)


def _model_id(parts: Iterable[Any]) -> str:
    text = "_".join(str(part).replace("/", "_").replace(".", "p") for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"htf_{text}_{digest}".lower()


def discover_single_symbol_candidates(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    leverage_allocs = [(3.0, 0.10), (4.0, 0.10), (5.0, 0.10), (4.0, 0.15), (5.0, 0.15)]
    for timeframe in timeframes:
        btc = bars_by_symbol_tf[("BTCUSDT", timeframe)][["datetime", "close"]].rename(
            columns={"close": "btc_close"}
        )
        for symbol in symbols:
            frame = (
                bars_by_symbol_tf[(symbol, timeframe)].merge(btc, on="datetime", how="left").ffill()
            )
            close = frame["close"]
            btc_close = frame["btc_close"]
            # Long-only HTF momentum persistence: deliberately not a reversal/threshold tweak.
            for lookback in (6, 12, 24, 48):
                momentum = close / close.shift(lookback) - 1.0
                btc_momentum = btc_close / btc_close.shift(max(3, lookback // 2)) - 1.0
                for threshold in (0.005, 0.010, 0.020, 0.040):
                    signal = (
                        ((momentum > threshold) & (btc_momentum > 0.0))
                        .fillna(False)
                        .astype(float)
                        .to_numpy()
                    )
                    for leverage, allocation in leverage_allocs:
                        notional = leverage * allocation
                        sim = simulate_symbol(
                            frame, signal, leverage=leverage, allocation_fraction=allocation
                        )
                        model_id = _model_id(
                            [
                                "trend",
                                timeframe,
                                symbol,
                                f"lb{lookback}",
                                f"th{threshold}",
                                f"{leverage}x",
                                allocation,
                            ]
                        )
                        base = {
                            "model_id": model_id,
                            "family": "htf_trend_persistence",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "side": "long_only",
                            "lookback_bars": lookback,
                            "threshold": threshold,
                            "leverage": leverage,
                            "allocation_fraction": allocation,
                            "notional_fraction": notional,
                        }
                        rows.append(
                            _finalize_candidate(base, sim, frame["datetime"], timeframe=timeframe)
                        )

            # Donchian breakout continuation with BTC regime filter.
            for lookback in (12, 24, 48, 72):
                rolling_high = frame["high"].shift(1).rolling(lookback).max()
                btc_momentum = btc_close / btc_close.shift(max(3, lookback // 4)) - 1.0
                signal = (
                    ((close > rolling_high) & (btc_momentum > 0.0))
                    .fillna(False)
                    .astype(float)
                    .to_numpy()
                )
                for leverage, allocation in leverage_allocs:
                    notional = leverage * allocation
                    sim = simulate_symbol(
                        frame, signal, leverage=leverage, allocation_fraction=allocation
                    )
                    base = {
                        "model_id": _model_id(
                            [
                                "donchian",
                                timeframe,
                                symbol,
                                f"lb{lookback}",
                                f"{leverage}x",
                                allocation,
                            ]
                        ),
                        "family": "htf_donchian_breakout",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "side": "long_only",
                        "lookback_bars": lookback,
                        "threshold": 0.0,
                        "leverage": leverage,
                        "allocation_fraction": allocation,
                        "notional_fraction": notional,
                    }
                    rows.append(
                        _finalize_candidate(base, sim, frame["datetime"], timeframe=timeframe)
                    )

            # Funding squeeze continuation: long only when momentum is positive and funding is not crowded expensive.
            if "funding_rate" in frame.columns:
                for lookback in (6, 12, 24):
                    momentum = close / close.shift(lookback) - 1.0
                    btc_momentum = btc_close / btc_close.shift(max(3, lookback // 2)) - 1.0
                    oi_change = (
                        frame["open_interest"].astype(float)
                        / frame["open_interest"].astype(float).shift(lookback)
                        - 1.0
                    )
                    for funding_max in (-0.00002, 0.0, 0.00005):
                        signal = (
                            (
                                (momentum > 0.01)
                                & (btc_momentum > 0.0)
                                & (frame["funding_rate"].astype(float) <= funding_max)
                                & (oi_change.fillna(0.0) >= 0.0)
                            )
                            .fillna(False)
                            .astype(float)
                            .to_numpy()
                        )
                        for leverage, allocation in ((3.0, 0.10), (4.0, 0.10), (5.0, 0.10)):
                            notional = leverage * allocation
                            sim = simulate_symbol(
                                frame, signal, leverage=leverage, allocation_fraction=allocation
                            )
                            base = {
                                "model_id": _model_id(
                                    [
                                        "funding_squeeze",
                                        timeframe,
                                        symbol,
                                        f"lb{lookback}",
                                        funding_max,
                                        f"{leverage}x",
                                        allocation,
                                    ]
                                ),
                                "family": "funding_squeeze_continuation",
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "side": "long_only",
                                "lookback_bars": lookback,
                                "threshold": funding_max,
                                "leverage": leverage,
                                "allocation_fraction": allocation,
                                "notional_fraction": notional,
                            }
                            rows.append(
                                _finalize_candidate(
                                    base, sim, frame["datetime"], timeframe=timeframe
                                )
                            )
    return rows


def discover_cross_sectional_candidates(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timeframe in timeframes:
        closes = []
        for symbol in symbols:
            frame = bars_by_symbol_tf[(symbol, timeframe)][["datetime", "close"]].copy()
            frame["symbol"] = symbol
            closes.append(frame)
        panel = (
            pd.concat(closes)
            .pivot(index="datetime", columns="symbol", values="close")
            .sort_index()
            .dropna()
        )
        for lookback in (6, 12, 24, 48):
            momentum = panel / panel.shift(lookback) - 1.0
            btc_momentum = panel["BTCUSDT"] / panel["BTCUSDT"].shift(max(3, lookback // 2)) - 1.0
            ranks = momentum.rank(axis=1, ascending=False, method="first")
            for top_n in (1, 2, 3):
                signals = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
                eligible = (
                    (ranks <= top_n) & (momentum > 0.0) & (btc_momentum.to_numpy()[:, None] > 0.0)
                )
                signals[eligible] = 1.0 / float(top_n)
                for leverage, allocation in ((3.0, 0.10), (4.0, 0.10), (5.0, 0.10), (4.0, 0.15)):
                    sim = simulate_portfolio(
                        panel, signals, leverage=leverage, allocation_fraction=allocation
                    )
                    base = {
                        "model_id": _model_id(
                            [
                                "xsec_momentum",
                                timeframe,
                                f"top{top_n}",
                                f"lb{lookback}",
                                f"{leverage}x",
                                allocation,
                            ]
                        ),
                        "family": "liquid_cross_sectional_momentum",
                        "symbol": f"portfolio_top{top_n}",
                        "timeframe": timeframe,
                        "side": "long_only",
                        "lookback_bars": lookback,
                        "threshold": 0.0,
                        "leverage": leverage,
                        "allocation_fraction": allocation,
                        "notional_fraction": leverage * allocation,
                    }
                    rows.append(
                        _finalize_candidate(
                            base, sim, panel.index.to_series(index=panel.index), timeframe=timeframe
                        )
                    )
    return rows


def _rank_rows(rows: Sequence[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    ranked = sorted(
        rows, key=lambda row: float(row.get("train_validation_score") or -1e9), reverse=True
    )
    if limit is not None:
        ranked = ranked[:limit]
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = rank
        out.append(item)
    return out


def _selected_output_rows(
    ranked_rows: Sequence[dict[str, Any]], *, top_n: int
) -> list[dict[str, Any]]:
    """Keep top-ranked rows plus any gate-pass rows even if their score rank is lower.

    The no-promotion artifact must not hide sample-gate evidence behind a pure
    top-N score truncation. Ranking is still train+validation only; this helper
    only controls artifact inclusion after all candidates are frozen/ranked.
    """
    selected_model_ids = {str(row["model_id"]) for row in ranked_rows[:top_n]}
    for row in ranked_rows:
        if any(bool(row.get(flag)) for flag in SPECIAL_OUTPUT_FLAGS):
            selected_model_ids.add(str(row["model_id"]))
    if ranked_rows:
        best_validation = max(
            ranked_rows, key=lambda row: float(row.get("validation_return") or -1e9)
        )
        selected_model_ids.add(str(best_validation["model_id"]))
    return [dict(row) for row in ranked_rows if str(row["model_id"]) in selected_model_ids]


def _shadow_hypothesis_rows(
    ranked_rows: Sequence[dict[str, Any]], *, top_n: int = 80
) -> list[dict[str, Any]]:
    shadow_rows = [
        row
        for row in ranked_rows
        if row.get("decision")
        in {"validation_alpha_shadow_until_execution_efficiency", "no_promotion_shadow_or_reject"}
    ]
    selected = _selected_output_rows(shadow_rows, top_n=top_n)
    return selected


def _decision_rows(rows: Sequence[dict[str, Any]], *, limit: int = 40) -> list[dict[str, Any]]:
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
                "validation_return": row["validation_return"],
                "locked_oos_return": row["locked_oos_return"],
                "ready_for_paper": row["ready_for_paper"],
                "ready_for_real": False,
                "real_money_execution": False,
                "rejection_reasons": row["rejection_reasons"],
            }
        )
    return decisions


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    decisions: dict[str, int] = {}
    for row in rows:
        families[str(row["family"])] = families.get(str(row["family"]), 0) + 1
        decisions[str(row["decision"])] = decisions.get(str(row["decision"]), 0) + 1
    best_validation = (
        max(rows, key=lambda row: float(row.get("validation_return") or -1e9)) if rows else {}
    )
    best_gate = next(
        (row for row in _rank_rows(rows) if row.get("backtest_sample_gate_pass")), None
    )
    return {
        "candidate_count": len(rows),
        "family_counts": dict(sorted(families.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "backtest_sample_gate_pass_count": sum(
            bool(row.get("backtest_sample_gate_pass")) for row in rows
        ),
        "execution_efficiency_proxy_gate_pass_count": sum(
            bool(row.get("execution_efficiency_proxy_gate_pass")) for row in rows
        ),
        "paper_candidate_gate_pass_count": sum(
            bool(row.get("paper_candidate_gate_pass")) for row in rows
        ),
        "max_validation_return": float(best_validation.get("validation_return") or 0.0)
        if best_validation
        else None,
        "best_validation_model_id": best_validation.get("model_id") if best_validation else None,
        "best_backtest_sample_gate_model_id": best_gate.get("model_id") if best_gate else None,
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("discovery_summary") or {})
    top = list(payload.get("top_candidates") or [])[:8]
    lines = [
        "# Alpha Zoo HTF momentum/crowding discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "This is a new-alpha discovery run, not a retune of existing reversal/quality-single-pair lanes.",
        "Locked-OOS is gate/report-only after train+validation candidate freeze. Real-money remains blocked.",
        "",
        "## Summary",
        "",
        f"- Candidates evaluated: `{summary.get('candidate_count')}`",
        f"- Backtest sample gate pass: `{summary.get('backtest_sample_gate_pass_count')}`",
        f"- Execution-efficiency proxy gate pass: `{summary.get('execution_efficiency_proxy_gate_pass_count')}`",
        f"- Full paper candidate gate pass: `{summary.get('paper_candidate_gate_pass_count')}`",
        f"- Max validation return: `{summary.get('max_validation_return')}`",
        "- `ready_for_real=false`, `real_money_execution=false`",
        "",
        "## Top train+validation-ranked rows",
        "",
        "| Rank | Family | Symbol | TF | Val | OOS | Decision |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in top:
        lines.append(
            f"| {row.get('rank')} | {row.get('family')} | {row.get('symbol')} | {row.get('timeframe')} | "
            f"{float(row.get('validation_return') or 0.0):.4%} | {float(row.get('locked_oos_return') or 0.0):.4%} | "
            f"{row.get('decision')} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    symbols = tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    timeframes = tuple(tf.strip().lower() for tf in args.timeframes.split(",") if tf.strip())
    hourly = load_hourly_bars(symbols, data_root=data_root)
    feature_frames = {
        symbol: load_feature_points(symbol, feature_root=feature_root) for symbol in symbols
    }
    bars_by_symbol_tf: dict[tuple[str, str], pd.DataFrame] = {}
    for timeframe in timeframes:
        for symbol in symbols:
            frame = resample_bars(hourly[symbol], timeframe)
            frame = attach_asof_features(frame, feature_frames.get(symbol, pd.DataFrame()))
            bars_by_symbol_tf[(symbol, timeframe)] = frame
    rows = discover_single_symbol_candidates(
        bars_by_symbol_tf, symbols=symbols, timeframes=timeframes
    )
    rows.extend(
        discover_cross_sectional_candidates(
            bars_by_symbol_tf, symbols=symbols, timeframes=timeframes
        )
    )
    ranked_rows = _rank_rows(rows)
    top_rows = _selected_output_rows(ranked_rows, top_n=int(args.top_n))
    decision_rows = _decision_rows(ranked_rows, limit=int(args.decision_top_n))
    timestamp = _timestamp()
    latest_json = output_dir / "alpha_zoo_htf_momentum_crowding_discovery_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_htf_momentum_crowding_discovery_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_htf_momentum_crowding_discovery_latest.md"
    candidates_csv = output_dir / "htf_momentum_crowding_candidates_latest.csv"
    decisions_csv = output_dir / "htf_momentum_crowding_decisions_latest.csv"
    shadow_csv = output_dir / "htf_momentum_crowding_shadow_hypotheses_latest.csv"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    shadow_rows = _shadow_hypothesis_rows(ranked_rows)
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_htf_momentum_crowding_discovery",
        "generated_at_utc": _utc_now_iso(),
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "ready_for_paper": any(bool(row.get("paper_candidate_gate_pass")) for row in ranked_rows),
        "ready_for_real": False,
        "real_money_execution": False,
        "paper_execution_allowed": False,
        "paper_testnet_only": True,
        "source_references": SOURCE_REFERENCES,
        "source_data": {
            "ohlcv_root": str(data_root),
            "feature_root": str(feature_root),
            "symbols": list(symbols),
            "timeframes": list(timeframes),
            "bar_source": "local Binance 1s OHLCV parquet resampled to 30m+ HTF bars",
            "feature_source": "local feature_points funding/open-interest as-of join when present",
        },
        "selection_policy": {
            "objective_inputs": ["train", "validation"],
            "selection_inputs": ["train", "validation"],
            "uses_locked_oos_for_discovery": False,
            "uses_locked_oos_for_selection": False,
            "uses_locked_oos_for_objective": False,
            "uses_locked_oos_for_pruning": False,
            "uses_locked_oos_for_parameter_fitting": False,
            "locked_oos_role": "gate_report_only_after_train_validation_candidate_freeze",
            "score_formula": "6*validation_return + 0.5*train_return - 2*validation_mdd",
            "new_alpha_policy": "new 30m+ momentum/crowding families only; not a reversal/quality-single-pair retune",
        },
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
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_markdown": str(latest_md),
            "candidates_csv": str(candidates_csv),
            "decisions_csv": str(decisions_csv),
            "shadow_hypotheses_csv": str(shadow_csv),
            "artifact_generation_validation_log": str(generation_log),
        },
        "runner_peak_rss_mib": local_peak_mib,
    }
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.write_text(_markdown(payload), encoding="utf-8")
    _write_csv(candidates_csv, top_rows, CANDIDATE_FIELDS)
    _write_csv(decisions_csv, decision_rows, DECISION_FIELDS)
    _write_csv(shadow_csv, shadow_rows, CANDIDATE_FIELDS)
    generation_log.write_text(
        "\n".join(
            [
                f"generated_at_utc={payload['generated_at_utc']}",
                f"artifact_kind={payload['artifact_kind']}",
                f"candidate_count={payload['discovery_summary']['candidate_count']}",
                f"output_candidate_count={len(top_rows)}",
                f"backtest_sample_gate_pass_count={payload['discovery_summary']['backtest_sample_gate_pass_count']}",
                f"paper_candidate_gate_pass_count={payload['discovery_summary']['paper_candidate_gate_pass_count']}",
                "uses_locked_oos_for_selection=false",
                "ready_for_real=false",
                "real_money_execution=false",
                f"runner_peak_rss_mib={local_peak_mib:.3f}",
                f"latest_json={latest_json}",
                f"timestamped_json={timestamped_json}",
                f"candidates_csv={candidates_csv}",
                f"decisions_csv={decisions_csv}",
                f"shadow_hypotheses_csv={shadow_csv}",
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default="1h,2h,4h,6h")
    parser.add_argument("--top-n", type=int, default=160)
    parser.add_argument("--decision-top-n", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    print(json.dumps(payload["output_paths"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
