#!/usr/bin/env python3
"""Discover diverse train-dominant Alpha Zoo candidates on local real data.

This runner is research/paper-testnet only. It broadens the strategy families beyond
reversal and HTF momentum retunes, ranks on train+validation only, and treats
locked-OOS strictly as a post-freeze gate/report split. It never executes orders
and never enables real money.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
from collections.abc import Iterable, Mapping, Sequence
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
    DEFAULT_DATA_ROOT,
    DEFAULT_FEATURE_ROOT,
    PRIMARY_ROUND_TRIP_COST_BPS,
    RETURN_PER_TURNOVER_THRESHOLD_BPS,
    SPLIT_ORDER,
    SPLITS,
    SimResult,
    _json_safe,
    _split_mask,
    attach_asof_features,
    load_hourly_bars,
    resample_bars,
    simulate_portfolio,
    simulate_symbol,
    split_metrics,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_diverse_train_dominant_discovery_20260522"
)
DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "TRXUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
)
DEFAULT_TIMEFRAMES = ("1h", "2h", "4h", "6h", "12h")

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
    "exit_threshold",
    "min_hold_bars",
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

SOURCE_REFERENCES = [
    {
        "label": "AdaptiveTrend crypto trend-following research",
        "url": "https://arxiv.org/abs/2602.11708",
        "usage": "motivates longer-horizon trend-following, but this runner also tests non-trend families",
    },
    {
        "label": "Binance USDⓈ-M funding rate history docs",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History",
        "usage": "documents funding inputs for carry/crowding lanes when local feature points are available",
    },
    {
        "label": "Binance USDⓈ-M open interest docs",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest",
        "usage": "documents open-interest inputs for OI expansion and crowded reversal lanes",
    },
    {
        "label": "Binance USDⓈ-M taker buy/sell volume docs",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume",
        "usage": "documents taker-flow inputs used opportunistically through local feature-point joins",
    },
]


def load_feature_points(symbol: str, *, feature_root: Path) -> pd.DataFrame:
    """Load optional feature points even when some telemetry columns are absent."""
    symbol_root = feature_root / f"symbol={symbol}"
    files = sorted(symbol_root.rglob("*.parquet"))
    output_columns = [
        "datetime",
        "funding_rate",
        "open_interest",
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
    ]
    if not files:
        return pd.DataFrame(columns=output_columns)
    lf = pl.scan_parquet([str(path) for path in files])
    available = set(lf.collect_schema().names())
    selected = [
        column
        for column in [
            "timestamp_ms",
            "funding_rate",
            "open_interest",
            "taker_buy_quote_volume",
            "taker_sell_quote_volume",
        ]
        if column in available
    ]
    if "timestamp_ms" not in selected:
        return pd.DataFrame(columns=output_columns)
    frame = lf.select(selected).collect()
    pdf = pd.DataFrame(frame.to_dicts())
    if pdf.empty:
        return pd.DataFrame(columns=output_columns)
    pdf["datetime"] = pd.to_datetime(pdf["timestamp_ms"], unit="ms")
    for column in output_columns:
        if column != "datetime" and column not in pdf.columns:
            pdf[column] = np.nan
    return pdf[output_columns].sort_values("datetime")


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


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def _model_id(parts: Iterable[Any]) -> str:
    text = "_".join(str(part).replace("/", "_").replace(".", "p") for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"divtd_{text}_{digest}".lower()


def _return_per_turnover(
    total_return: float, trade_events: int, notional_fraction: float
) -> float | None:
    turnover = float(trade_events) * abs(float(notional_fraction))
    if turnover <= 0:
        return None
    return float(total_return) * 10000.0 / turnover


def _candidate_score(row: Mapping[str, Any]) -> float:
    """Train+validation-only score; locked-OOS is intentionally ignored."""
    train = float(row.get("train_return") or 0.0)
    validation = float(row.get("validation_return") or 0.0)
    val_mdd = float(row.get("validation_mdd") or 0.0)
    dominance_bonus = min(train, validation)
    train_shortfall_penalty = max(0.0, validation - train)
    return 4.0 * validation + 1.25 * dominance_bonus - 6.0 * train_shortfall_penalty - 2.0 * val_mdd


def _gate_candidate(row: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    train = float(row["train_return"])
    validation = float(row["validation_return"])
    ratio = row.get("train_validation_return_ratio")
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
    if validation < PROMOTION_THRESHOLDS["min_validation_return"]:
        reasons.append(f"validation_return_{validation:.4f}_below_0.02")
    if train <= 0:
        reasons.append("train_return_not_positive")
    if train < validation:
        reasons.append(f"train_return_{train:.4f}_below_validation_return_{validation:.4f}")
    if ratio is None or float(ratio) < PROMOTION_THRESHOLDS["min_train_validation_return_ratio"]:
        reasons.append(
            f"train_validation_return_ratio_{0.0 if ratio is None else ratio:.4f}_below_1.00"
        )
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
                f"{split}_return_per_turnover_proxy_bps_{rendered}_not_above_{RETURN_PER_TURNOVER_THRESHOLD_BPS:.3f}"
            )
    execution_efficiency_proxy_gate_pass = not efficiency_reasons
    paper_candidate_gate_pass = (
        train_dominant_sample_gate_pass and execution_efficiency_proxy_gate_pass
    )
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


def _stateful_signal(
    long_entry: pd.Series,
    long_exit: pd.Series,
    short_entry: pd.Series | None = None,
    short_exit: pd.Series | None = None,
    *,
    side: str,
    min_hold_bars: int = 0,
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
    out = np.zeros(len(long_entry), dtype=float)
    state = 0.0
    bars_held = 10**9
    for i in range(len(long_entry)):
        can_exit = bars_held >= min_hold_bars
        if can_exit and (
            (state > 0 and bool(long_exit.iloc[i])) or (state < 0 and bool(short_exit.iloc[i]))
        ):
            state = 0.0
            bars_held = 0
        if state == 0.0:
            if side in {"long_only", "long_short"} and bool(long_entry.iloc[i]):
                state = 1.0
                bars_held = 0
            elif side in {"short_only", "long_short"} and bool(short_entry.iloc[i]):
                state = -1.0
                bars_held = 0
        out[i] = state
        bars_held += 1
    return out


def _zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)


def _volatility_filter(
    close: pd.Series, lookback: int, quantile_window: int, max_quantile: float
) -> pd.Series:
    ret = close.pct_change()
    realized = ret.rolling(lookback).std(ddof=1)
    threshold = realized.rolling(quantile_window).quantile(max_quantile)
    return realized <= threshold


def discover_single_symbol_candidates(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    leverage_allocs = [(2.0, 0.10), (3.0, 0.10), (4.0, 0.10), (3.0, 0.15), (4.0, 0.15)]
    for timeframe in timeframes:
        btc = bars_by_symbol_tf[("BTCUSDT", timeframe)][["datetime", "close"]].rename(
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
            btc_regime_fast = btc_close / btc_close.shift(12) - 1.0

            # 1) Stateful momentum with entry/exit hysteresis: lower turnover than simple threshold retunes.
            for lookback in (6, 12, 24, 48):
                momentum = close / close.shift(lookback) - 1.0
                for entry_threshold, exit_threshold in ((0.01, 0.0), (0.02, 0.005), (0.04, 0.01)):
                    for side in ("long_only", "short_only", "long_short"):
                        signal = _stateful_signal(
                            (momentum > entry_threshold) & (btc_regime_fast > -0.02),
                            momentum < exit_threshold,
                            (momentum < -entry_threshold) & (btc_regime_fast < 0.02),
                            momentum > -exit_threshold,
                            side=side,
                        )
                        for leverage, allocation in leverage_allocs:
                            sim = simulate_symbol(
                                frame, signal, leverage=leverage, allocation_fraction=allocation
                            )
                            rows.append(
                                _finalize_candidate(
                                    {
                                        "model_id": _model_id(
                                            [
                                                "hysteresis_trend",
                                                timeframe,
                                                symbol,
                                                side,
                                                f"lb{lookback}",
                                                f"e{entry_threshold}",
                                                f"x{exit_threshold}",
                                                f"{leverage}x",
                                                allocation,
                                            ]
                                        ),
                                        "family": "stateful_momentum_hysteresis",
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "side": side,
                                        "lookback_bars": lookback,
                                        "threshold": entry_threshold,
                                        "exit_threshold": exit_threshold,
                                        "min_hold_bars": 0,
                                        "leverage": leverage,
                                        "allocation_fraction": allocation,
                                        "notional_fraction": leverage * allocation,
                                    },
                                    sim,
                                    frame["datetime"],
                                    timeframe=timeframe,
                                )
                            )

            # 2) Debounced momentum hysteresis: explicitly reduces churn via a minimum holding period.
            for lookback in (6, 12, 24):
                momentum = close / close.shift(lookback) - 1.0
                for entry_threshold, exit_threshold in ((0.02, 0.005), (0.04, 0.01)):
                    for side in ("long_only", "short_only", "long_short"):
                        for min_hold_bars in (6, 12, 24):
                            signal = _stateful_signal(
                                (momentum > entry_threshold) & (btc_regime_fast > -0.02),
                                momentum < exit_threshold,
                                (momentum < -entry_threshold) & (btc_regime_fast < 0.02),
                                momentum > -exit_threshold,
                                side=side,
                                min_hold_bars=min_hold_bars,
                            )
                            for leverage, allocation in leverage_allocs[:4]:
                                sim = simulate_symbol(
                                    frame, signal, leverage=leverage, allocation_fraction=allocation
                                )
                                rows.append(
                                    _finalize_candidate(
                                        {
                                            "model_id": _model_id(
                                                [
                                                    "debounced_hysteresis_trend",
                                                    timeframe,
                                                    symbol,
                                                    side,
                                                    f"lb{lookback}",
                                                    f"e{entry_threshold}",
                                                    f"x{exit_threshold}",
                                                    f"hold{min_hold_bars}",
                                                    f"{leverage}x",
                                                    allocation,
                                                ]
                                            ),
                                            "family": "debounced_momentum_hysteresis",
                                            "symbol": symbol,
                                            "timeframe": timeframe,
                                            "side": side,
                                            "lookback_bars": lookback,
                                            "threshold": entry_threshold,
                                            "exit_threshold": exit_threshold,
                                            "min_hold_bars": min_hold_bars,
                                            "leverage": leverage,
                                            "allocation_fraction": allocation,
                                            "notional_fraction": leverage * allocation,
                                        },
                                        sim,
                                        frame["datetime"],
                                        timeframe=timeframe,
                                    )
                                )

            # 2) Volatility contraction breakout continuation.
            for lookback in (12, 24, 48):
                low_vol = _volatility_filter(close, max(4, lookback // 2), lookback * 4, 0.35)
                prior_high = high.shift(1).rolling(lookback).max()
                prior_low = low.shift(1).rolling(lookback).min()
                mid = close.rolling(lookback).mean()
                for side in ("long_only", "short_only", "long_short"):
                    signal = _stateful_signal(
                        (close > prior_high) & low_vol,
                        close < mid,
                        (close < prior_low) & low_vol,
                        close > mid,
                        side=side,
                    )
                    for leverage, allocation in leverage_allocs[:4]:
                        sim = simulate_symbol(
                            frame, signal, leverage=leverage, allocation_fraction=allocation
                        )
                        rows.append(
                            _finalize_candidate(
                                {
                                    "model_id": _model_id(
                                        [
                                            "vol_contraction_breakout",
                                            timeframe,
                                            symbol,
                                            side,
                                            f"lb{lookback}",
                                            f"{leverage}x",
                                            allocation,
                                        ]
                                    ),
                                    "family": "volatility_contraction_breakout",
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "side": side,
                                    "lookback_bars": lookback,
                                    "threshold": 0.35,
                                    "exit_threshold": 0.0,
                                    "min_hold_bars": 0,
                                    "leverage": leverage,
                                    "allocation_fraction": allocation,
                                    "notional_fraction": leverage * allocation,
                                },
                                sim,
                                frame["datetime"],
                                timeframe=timeframe,
                            )
                        )

            # 3) Pullback in established trend: different entry logic from breakout/trend persistence.
            for long_lookback in (24, 48, 72):
                long_momentum = close / close.shift(long_lookback) - 1.0
                short_momentum = close / close.shift(max(3, long_lookback // 6)) - 1.0
                for pullback_threshold in (0.005, 0.01, 0.02):
                    signal = _stateful_signal(
                        (long_momentum > 0.03)
                        & (short_momentum < -pullback_threshold)
                        & (btc_regime_fast > -0.01),
                        (short_momentum > 0.0) | (long_momentum < 0.0),
                        (long_momentum < -0.03)
                        & (short_momentum > pullback_threshold)
                        & (btc_regime_fast < 0.01),
                        (short_momentum < 0.0) | (long_momentum > 0.0),
                        side="long_short",
                    )
                    for leverage, allocation in leverage_allocs[:4]:
                        sim = simulate_symbol(
                            frame, signal, leverage=leverage, allocation_fraction=allocation
                        )
                        rows.append(
                            _finalize_candidate(
                                {
                                    "model_id": _model_id(
                                        [
                                            "trend_pullback",
                                            timeframe,
                                            symbol,
                                            f"lb{long_lookback}",
                                            pullback_threshold,
                                            f"{leverage}x",
                                            allocation,
                                        ]
                                    ),
                                    "family": "trend_pullback_reentry",
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "side": "long_short",
                                    "lookback_bars": long_lookback,
                                    "threshold": pullback_threshold,
                                    "exit_threshold": 0.0,
                                    "min_hold_bars": 0,
                                    "leverage": leverage,
                                    "allocation_fraction": allocation,
                                    "notional_fraction": leverage * allocation,
                                },
                                sim,
                                frame["datetime"],
                                timeframe=timeframe,
                            )
                        )

            # 4) Range z-score mean reversion with explicit exits.
            for lookback in (24, 48, 72):
                z = _zscore(close, lookback)
                for z_entry in (1.5, 2.0, 2.5):
                    signal = _stateful_signal(
                        z < -z_entry, z > -0.1, z > z_entry, z < 0.1, side="long_short"
                    )
                    for leverage, allocation in leverage_allocs[:4]:
                        sim = simulate_symbol(
                            frame, signal, leverage=leverage, allocation_fraction=allocation
                        )
                        rows.append(
                            _finalize_candidate(
                                {
                                    "model_id": _model_id(
                                        [
                                            "range_zscore_reversion",
                                            timeframe,
                                            symbol,
                                            f"lb{lookback}",
                                            z_entry,
                                            f"{leverage}x",
                                            allocation,
                                        ]
                                    ),
                                    "family": "range_zscore_mean_reversion",
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "side": "long_short",
                                    "lookback_bars": lookback,
                                    "threshold": z_entry,
                                    "exit_threshold": 0.1,
                                    "min_hold_bars": 0,
                                    "leverage": leverage,
                                    "allocation_fraction": allocation,
                                    "notional_fraction": leverage * allocation,
                                },
                                sim,
                                frame["datetime"],
                                timeframe=timeframe,
                            )
                        )

            # 5) Funding/OI/taker-flow carry and crowding overlays, if feature coverage exists.
            if {"funding_rate", "open_interest", "taker_buy_sell_imbalance"}.issubset(
                frame.columns
            ):
                funding = frame["funding_rate"].astype(float)
                oi = frame["open_interest"].astype(float)
                taker_imb = frame["taker_buy_sell_imbalance"].astype(float)
                for lookback in (6, 12, 24):
                    momentum = close / close.shift(lookback) - 1.0
                    oi_change = oi / oi.shift(lookback) - 1.0
                    for funding_abs in (0.00002, 0.00005, 0.00010):
                        signal = _stateful_signal(
                            (funding <= -funding_abs) & (momentum > 0.005),
                            (funding > 0.0) | (momentum < 0.0),
                            (funding >= funding_abs) & (momentum < -0.005),
                            (funding < 0.0) | (momentum > 0.0),
                            side="long_short",
                        )
                        for leverage, allocation in leverage_allocs[:3]:
                            sim = simulate_symbol(
                                frame, signal, leverage=leverage, allocation_fraction=allocation
                            )
                            rows.append(
                                _finalize_candidate(
                                    {
                                        "model_id": _model_id(
                                            [
                                                "funding_carry_momentum",
                                                timeframe,
                                                symbol,
                                                f"lb{lookback}",
                                                funding_abs,
                                                f"{leverage}x",
                                                allocation,
                                            ]
                                        ),
                                        "family": "funding_carry_momentum",
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "side": "long_short",
                                        "lookback_bars": lookback,
                                        "threshold": funding_abs,
                                        "exit_threshold": 0.0,
                                        "min_hold_bars": 0,
                                        "leverage": leverage,
                                        "allocation_fraction": allocation,
                                        "notional_fraction": leverage * allocation,
                                    },
                                    sim,
                                    frame["datetime"],
                                    timeframe=timeframe,
                                )
                            )
                    for oi_threshold in (0.01, 0.03, 0.06):
                        signal = _stateful_signal(
                            (oi_change > oi_threshold) & (momentum > 0.01) & (taker_imb > -0.2),
                            momentum < 0.0,
                            (oi_change > oi_threshold) & (momentum < -0.01) & (taker_imb < 0.2),
                            momentum > 0.0,
                            side="long_short",
                        )
                        for leverage, allocation in leverage_allocs[:3]:
                            sim = simulate_symbol(
                                frame, signal, leverage=leverage, allocation_fraction=allocation
                            )
                            rows.append(
                                _finalize_candidate(
                                    {
                                        "model_id": _model_id(
                                            [
                                                "oi_expansion_flow_trend",
                                                timeframe,
                                                symbol,
                                                f"lb{lookback}",
                                                oi_threshold,
                                                f"{leverage}x",
                                                allocation,
                                            ]
                                        ),
                                        "family": "oi_expansion_flow_trend",
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "side": "long_short",
                                        "lookback_bars": lookback,
                                        "threshold": oi_threshold,
                                        "exit_threshold": 0.0,
                                        "min_hold_bars": 0,
                                        "leverage": leverage,
                                        "allocation_fraction": allocation,
                                        "notional_fraction": leverage * allocation,
                                    },
                                    sim,
                                    frame["datetime"],
                                    timeframe=timeframe,
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
        frames = []
        for symbol in symbols:
            frame = bars_by_symbol_tf[(symbol, timeframe)][["datetime", "close"]].copy()
            frame["symbol"] = symbol
            frames.append(frame)
        panel = (
            pd.concat(frames)
            .pivot(index="datetime", columns="symbol", values="close")
            .sort_index()
            .dropna()
        )
        for lookback in (6, 12, 24, 48):
            momentum = panel / panel.shift(lookback) - 1.0
            realized = panel.pct_change().rolling(max(4, lookback // 2)).std(ddof=1)
            ranks = momentum.rank(axis=1, ascending=False, method="first")
            reverse_ranks = momentum.rank(axis=1, ascending=True, method="first")
            low_vol_ranks = realized.rank(axis=1, ascending=True, method="first")
            for top_n in (1, 2, 3):
                # Market-neutral relative momentum: long winners / short losers, normalized gross exposure.
                signals = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
                long_mask = (ranks <= top_n) & (momentum > 0.0)
                short_mask = (reverse_ranks <= top_n) & (momentum < 0.0)
                signals[long_mask] = 0.5 / float(top_n)
                signals[short_mask] = -0.5 / float(top_n)
                for leverage, allocation in ((2.0, 0.10), (3.0, 0.10), (4.0, 0.10), (3.0, 0.15)):
                    sim = simulate_portfolio(
                        panel, signals, leverage=leverage, allocation_fraction=allocation
                    )
                    rows.append(
                        _finalize_candidate(
                            {
                                "model_id": _model_id(
                                    [
                                        "xsec_market_neutral_momentum",
                                        timeframe,
                                        f"top{top_n}",
                                        f"lb{lookback}",
                                        f"{leverage}x",
                                        allocation,
                                    ]
                                ),
                                "family": "cross_sectional_market_neutral_momentum",
                                "symbol": f"portfolio_top_bottom_{top_n}",
                                "timeframe": timeframe,
                                "side": "long_short",
                                "lookback_bars": lookback,
                                "threshold": 0.0,
                                "exit_threshold": 0.0,
                                "min_hold_bars": 0,
                                "leverage": leverage,
                                "allocation_fraction": allocation,
                                "notional_fraction": leverage * allocation,
                            },
                            sim,
                            panel.index.to_series(index=panel.index),
                            timeframe=timeframe,
                        )
                    )

                # Low-volatility momentum rotation: long strongest low-vol assets only.
                low_vol_signals = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
                eligible = (
                    (ranks <= top_n + 1) & (low_vol_ranks <= max(2, top_n + 1)) & (momentum > 0.0)
                )
                low_vol_signals[eligible] = 1.0 / float(top_n)
                row_sums = low_vol_signals.abs().sum(axis=1).replace(0.0, np.nan)
                low_vol_signals = low_vol_signals.div(row_sums, axis=0).fillna(0.0)
                for leverage, allocation in ((2.0, 0.10), (3.0, 0.10), (4.0, 0.10), (3.0, 0.15)):
                    sim = simulate_portfolio(
                        panel, low_vol_signals, leverage=leverage, allocation_fraction=allocation
                    )
                    rows.append(
                        _finalize_candidate(
                            {
                                "model_id": _model_id(
                                    [
                                        "xsec_lowvol_momentum",
                                        timeframe,
                                        f"top{top_n}",
                                        f"lb{lookback}",
                                        f"{leverage}x",
                                        allocation,
                                    ]
                                ),
                                "family": "cross_sectional_lowvol_momentum_rotation",
                                "symbol": f"portfolio_lowvol_top_{top_n}",
                                "timeframe": timeframe,
                                "side": "long_only",
                                "lookback_bars": lookback,
                                "threshold": 0.0,
                                "exit_threshold": 0.0,
                                "min_hold_bars": 0,
                                "leverage": leverage,
                                "allocation_fraction": allocation,
                                "notional_fraction": leverage * allocation,
                            },
                            sim,
                            panel.index.to_series(index=panel.index),
                            timeframe=timeframe,
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
    return _selected_output_rows(shadows, top_n=limit)


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
        (row for row in _rank_rows(rows) if row.get("train_dominant_sample_gate_pass")), None
    )
    train_gte_validation_count = sum(
        float(row.get("train_return") or 0.0) >= float(row.get("validation_return") or 0.0)
        for row in rows
    )
    return {
        "candidate_count": len(rows),
        "family_counts": dict(sorted(families.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "train_return_gte_validation_return_count": train_gte_validation_count,
        "train_dominant_sample_gate_pass_count": sum(
            bool(row.get("train_dominant_sample_gate_pass")) for row in rows
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
        "best_train_dominant_sample_gate_model_id": best_gate.get("model_id")
        if best_gate
        else None,
        "ready_for_real": False,
        "real_money_execution": False,
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    summary = dict(payload.get("discovery_summary") or {})
    top = list(payload.get("top_candidates") or [])[:10]
    lines = [
        "# Alpha Zoo diverse train-dominant discovery",
        "",
        f"Generated: `{payload.get('generated_at_utc')}`",
        "",
        "This run enforces `train_return >= validation_return` for promotion trust and explores diverse strategy families.",
        "Locked-OOS is gate/report-only after train+validation candidate freeze. Real-money remains blocked.",
        "",
        "## Summary",
        "",
        f"- Candidates evaluated: `{summary.get('candidate_count')}`",
        f"- Rows with train >= validation: `{summary.get('train_return_gte_validation_return_count')}`",
        f"- Train-dominant sample gate pass: `{summary.get('train_dominant_sample_gate_pass_count')}`",
        f"- Execution-efficiency proxy gate pass: `{summary.get('execution_efficiency_proxy_gate_pass_count')}`",
        f"- Full paper candidate gate pass: `{summary.get('paper_candidate_gate_pass_count')}`",
        f"- Max validation return: `{summary.get('max_validation_return')}`",
        "- `ready_for_real=false`, `real_money_execution=false`",
        "",
        "## Top train+validation-ranked rows",
        "",
        "| Rank | Family | Symbol | TF | Train | Val | OOS | Decision |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in top:
        lines.append(
            f"| {row.get('rank')} | {row.get('family')} | {row.get('symbol')} | {row.get('timeframe')} | "
            f"{float(row.get('train_return') or 0.0):.4%} | {float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {row.get('decision')} |"
        )
    gate_rows = [
        row
        for row in payload.get("top_candidates", [])
        if row.get("train_dominant_sample_gate_pass")
    ]
    lines.extend(
        [
            "",
            "## Train-dominant sample-gate shadows",
            "",
            "These rows satisfy train>=validation, split sample counts, positive locked-OOS, zero liquidation/account wipeout, and validation return/MDD gates; they are still shadow-only until execution efficiency passes.",
            "",
            "| Rank | Family | Symbol | TF | Train | Val | OOS | Rejection focus |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in gate_rows[:10]:
        reasons = "; ".join(str(reason) for reason in list(row.get("rejection_reasons") or [])[:3])
        lines.append(
            f"| {row.get('rank')} | {row.get('family')} | {row.get('symbol')} | {row.get('timeframe')} | "
            f"{float(row.get('train_return') or 0.0):.4%} | {float(row.get('validation_return') or 0.0):.4%} | "
            f"{float(row.get('locked_oos_return') or 0.0):.4%} | {reasons} |"
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
    shadow_rows = _shadow_rows(ranked_rows, limit=int(args.shadow_top_n))

    timestamp = _timestamp()
    latest_json = output_dir / "alpha_zoo_diverse_train_dominant_discovery_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_diverse_train_dominant_discovery_{timestamp}.json"
    latest_md = output_dir / "alpha_zoo_diverse_train_dominant_discovery_latest.md"
    candidates_csv = output_dir / "diverse_train_dominant_candidates_latest.csv"
    decisions_csv = output_dir / "diverse_train_dominant_decisions_latest.csv"
    shadow_csv = output_dir / "diverse_train_dominant_shadow_hypotheses_latest.csv"
    generation_log = output_dir / "artifact_generation_validation_latest.log"
    local_peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    payload: dict[str, Any] = {
        "artifact_kind": "alpha_zoo_diverse_train_dominant_discovery",
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
            "bar_source": "local Binance 1s OHLCV parquet resampled to 1h+ bars",
            "feature_source": "local feature_points funding/open-interest/taker-flow as-of join when present",
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
            "score_formula": "4*validation_return + 1.25*min(train_return,validation_return) - 6*max(0,validation_return-train_return) - 2*validation_mdd",
            "trust_gate": "promotion requires train_return >= validation_return and train/validation ratio >= 1.0",
            "new_alpha_policy": "diverse strategy families; not a reversal/quality-single-pair retune",
        },
        "strategy_families": [
            "stateful_momentum_hysteresis",
            "volatility_contraction_breakout",
            "trend_pullback_reentry",
            "range_zscore_mean_reversion",
            "debounced_momentum_hysteresis",
            "funding_carry_momentum",
            "oi_expansion_flow_trend",
            "cross_sectional_market_neutral_momentum",
            "cross_sectional_lowvol_momentum_rotation",
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
    latest_md.parent.mkdir(parents=True, exist_ok=True)
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
                f"train_return_gte_validation_return_count={payload['discovery_summary']['train_return_gte_validation_return_count']}",
                f"train_dominant_sample_gate_pass_count={payload['discovery_summary']['train_dominant_sample_gate_pass_count']}",
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
