#!/usr/bin/env python3
"""Run a 69-asset, 30m+ Alpha Zoo Optuna hybrid refit on direct 1m bars.

This runner is intentionally research / paper-testnet only. It extends the
current live-refit discipline from the ETH/SOL/TRX-heavy frozen hybrid to the
expanded Binance research universe:

* source bars are direct 1m OHLCV aggregated to 30m or higher;
* the latest 8 complete weeks are validation;
* candidate discovery, Optuna objective, and hybrid selection use train and
  validation only;
* no locked test/OOS is reserved in this live-refit mode;
* real-money execution remains hard-false until separate paper/testnet fill,
  BBO, slippage, protection, and reconciliation telemetry passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import sys
from collections import Counter, defaultdict
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

from lumina_quant.alpha_zoo.live_training_policy import (  # noqa: E402
    STANDARD_VALIDATION_WEEKS,
    compute_standard_live_training_plan,
)
from lumina_quant.optimization.search_policy import (  # noqa: E402
    optimization_search_policy_payload,
    run_optuna_study,
)
from lumina_quant.research_universe import (  # noqa: E402
    BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS,
    BINANCE_EXTENDED_RESEARCH_SYMBOLS,
    BINANCE_TRADFI_COMMODITY_SYMBOLS,
    BINANCE_TRADFI_EQUITY_SYMBOLS,
    BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    BINANCE_TRADFI_PREMARKET_SYMBOLS,
)

try:  # pragma: no cover - import availability is environment dependent.
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:  # pragma: no cover
    optuna = None

ALPHA_V2_ROOT = REPO_ROOT / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2"
DEFAULT_OUTPUT_DIR = ALPHA_V2_ROOT / "alpha_zoo_69_asset_optuna_hybrid_refit_20260530"
DEFAULT_DATA_ROOT = REPO_ROOT / "data/market_parquet/exchange=binance"
DEFAULT_PRIOR_STANDARD_REFIT = (
    ALPHA_V2_ROOT
    / "alpha_zoo_standard_live_refit_20260528/alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json"
)

PRIMARY_ROUND_TRIP_COST_BPS = 10.0
AVG_BBO_SPREAD_BPS_ASSUMPTION = 2.0
BBO_SPREAD_MULTIPLIER = 5.0
RETURN_PER_TURNOVER_THRESHOLD_BPS = AVG_BBO_SPREAD_BPS_ASSUMPTION * BBO_SPREAD_MULTIPLIER

DEFAULT_TIMEFRAMES = ("30m", "1h", "2h", "4h")
DEFAULT_LEVERAGES = (1, 2, 3, 4)
DEFAULT_ALLOCATION_FRACTION = 0.10
MAX_CACHED_STREAMS_PER_SYMBOL = 8

TRAIN_VAL_SELECTION_INPUTS = ("train", "validation")
SPLIT_ORDER = ("train", "validation")

PROMOTION_THRESHOLDS = {
    "min_train_trade_event_count": 80,
    "min_validation_trade_event_count": 30,
    "min_validation_return": 0.02,
    "require_train_return_positive": True,
    "require_train_return_gte_validation_return": True,
    "max_validation_mdd_strict": 0.12,
    "max_validation_mdd_relaxed": 0.20,
    "min_return_per_turnover_proxy_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
    "primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
}

ASSET_GROUPS: dict[str, tuple[str, ...]] = {
    "crypto_core": BINANCE_CORE_CRYPTO_RESEARCH_SYMBOLS,
    "tradfi_commodity": BINANCE_TRADFI_COMMODITY_SYMBOLS,
    "tradfi_etf_index": BINANCE_TRADFI_ETF_INDEX_SYMBOLS,
    "tradfi_equity": BINANCE_TRADFI_EQUITY_SYMBOLS,
    "tradfi_premarket": BINANCE_TRADFI_PREMARKET_SYMBOLS,
}

CANDIDATE_FIELDS = [
    "rank",
    "model_id",
    "family",
    "symbol",
    "asset_group",
    "timeframe",
    "side",
    "lookback_bars",
    "threshold",
    "exit_threshold",
    "min_hold_bars",
    "cooldown_bars",
    "integer_leverage",
    "allocation_fraction",
    "notional_fraction",
    "train_return",
    "validation_return",
    "train_mdd",
    "validation_mdd",
    "train_sharpe",
    "validation_sharpe",
    "train_trade_event_count",
    "validation_trade_event_count",
    "train_exposure_bar_count",
    "validation_exposure_bar_count",
    "train_validation_return_ratio",
    "train_minus_validation_return",
    "train_return_per_turnover_proxy_bps",
    "validation_return_per_turnover_proxy_bps",
    "train_validation_score",
    "sample_gate_pass",
    "execution_efficiency_proxy_gate_pass",
    "strict_backtest_gate_pass",
    "relaxed_backtest_gate_pass",
    "decision",
    "ready_for_paper",
    "ready_for_real",
    "real_money_execution",
    "real_execution_allowed",
    "rejection_reasons",
]

SELECTED_WEIGHT_FIELDS = [
    "hybrid_weight_rank",
    "model_id",
    "weight",
    "symbol",
    "asset_group",
    "family",
    "timeframe",
    "side",
    "integer_leverage",
    "notional_fraction",
    "weighted_notional_fraction",
    "train_return",
    "validation_return",
    "validation_mdd",
]

ATTRIBUTION_FIELDS = [
    "dimension",
    "key",
    "weight_sum",
    "weighted_notional_fraction",
    "train_simple_pnl_contribution",
    "validation_simple_pnl_contribution",
    "candidate_count",
]


@dataclass(frozen=True)
class SimResult:
    returns: np.ndarray
    position: np.ndarray
    liquidation_flags: np.ndarray
    account_wipeout_flags: np.ndarray


@dataclass(frozen=True)
class CandidateStream:
    row: dict[str, Any]
    returns: pd.Series
    position: pd.Series


@dataclass(frozen=True)
class SplitWindows:
    train: tuple[pd.Timestamp, pd.Timestamp]
    validation: tuple[pd.Timestamp, pd.Timestamp]

    def as_payload(self) -> dict[str, dict[str, Any]]:
        return {
            "train": {
                "start": self.train[0].isoformat(),
                "end": self.train[1].isoformat(),
                "role": "parameter_fitting_and_objective_training",
                "enabled": True,
            },
            "validation": {
                "start": self.validation[0].isoformat(),
                "end": self.validation[1].isoformat(),
                "role": "holdout_selection_and_report",
                "enabled": True,
            },
            "locked_oos": {
                "start": None,
                "end": None,
                "role": "disabled_for_live_final_refit_no_test_set_reserved",
                "enabled": False,
            },
        }


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip().upper() for item in value.split(",") if item.strip()))


def _parse_timeframes(value: str) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip()))
    for timeframe in out:
        _timeframe_minutes(timeframe)
    return out


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))


def _timeframe_minutes(timeframe: str) -> int:
    if timeframe.endswith("m"):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith("h"):
        minutes = int(timeframe[:-1]) * 60
    else:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    if minutes < 30:
        raise ValueError(f"minimum research timeframe is 30m, got {timeframe!r}")
    return minutes


def _polars_every(timeframe: str) -> str:
    minutes = _timeframe_minutes(timeframe)
    return f"{minutes}m" if minutes < 60 else f"{minutes // 60}h"


def _periods_per_year(timeframe: str) -> float:
    return 365.0 * 24.0 * 60.0 / float(_timeframe_minutes(timeframe))


def _asset_group(symbol: str) -> str:
    token = symbol.strip().upper()
    for group, members in ASSET_GROUPS.items():
        if token in members:
            return group
    return "other"


def _model_id(parts: Iterable[Any]) -> str:
    text = "_".join(str(part).replace("/", "_").replace(".", "p") for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"az69_{text}_{digest}".lower()


def _symbol_1m_files(data_root: Path, symbol: str) -> list[Path]:
    return sorted((data_root / f"symbol={symbol}" / "timeframe=1m").rglob("*.parquet"))


def load_symbol_bars(
    symbol: str,
    *,
    data_root: Path,
    timeframe: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load one symbol from direct 1m parquet and aggregate to ``timeframe``."""
    files = _symbol_1m_files(data_root, symbol)
    if not files:
        raise FileNotFoundError(f"missing direct 1m parquet files for {symbol}: {data_root}")
    lf = pl.scan_parquet([str(path) for path in files]).sort("datetime")
    if start is not None:
        lf = lf.filter(pl.col("datetime") >= pl.lit(start.to_pydatetime()))
    if end is not None:
        lf = lf.filter(pl.col("datetime") <= pl.lit(end.to_pydatetime()))
    every = _polars_every(timeframe)
    frame = (
        lf.group_by_dynamic("datetime", every=every, period=every, label="left", closed="left")
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.len().alias("source_1m_rows"),
            ]
        )
        .drop_nulls(["open", "high", "low", "close"])
        .sort("datetime")
        .collect()
    )
    pdf = pd.DataFrame(frame.to_dicts())
    if pdf.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "source_1m_rows",
                "symbol",
            ]
        )
    pdf["datetime"] = pd.to_datetime(pdf["datetime"])
    pdf["symbol"] = symbol
    return pdf


def _load_timeframe_panel(
    symbols: Sequence[str], *, data_root: Path, timeframe: str
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    bars: dict[str, pd.DataFrame] = {}
    coverage: dict[str, Any] = {}
    for symbol in symbols:
        frame = load_symbol_bars(symbol, data_root=data_root, timeframe=timeframe)
        bars[symbol] = frame
        if frame.empty:
            coverage[symbol] = {"rows": 0, "earliest": None, "latest": None}
        else:
            coverage[symbol] = {
                "rows": len(frame),
                "earliest": frame["datetime"].min().isoformat(),
                "latest": frame["datetime"].max().isoformat(),
            }
    return bars, coverage


def load_all_bars(
    symbols: Sequence[str], *, data_root: Path, timeframes: Sequence[str]
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, Any]]:
    bars: dict[tuple[str, str], pd.DataFrame] = {}
    coverage: dict[str, Any] = {"timeframes": {}, "missing_symbols": []}
    for timeframe in timeframes:
        loaded, tf_cov = _load_timeframe_panel(symbols, data_root=data_root, timeframe=timeframe)
        coverage["timeframes"][timeframe] = tf_cov
        for symbol, frame in loaded.items():
            bars[(symbol, timeframe)] = frame
            if frame.empty and symbol not in coverage["missing_symbols"]:
                coverage["missing_symbols"].append(symbol)
    latest_values = [
        pd.Timestamp(item["latest"])
        for tf_cov in coverage["timeframes"].values()
        for item in tf_cov.values()
        if item.get("latest")
    ]
    earliest_values = [
        pd.Timestamp(item["earliest"])
        for tf_cov in coverage["timeframes"].values()
        for item in tf_cov.values()
        if item.get("earliest")
    ]
    if not latest_values:
        raise ValueError("no direct 1m-derived bars loaded for any symbol/timeframe")
    coverage["symbol_count"] = len(symbols)
    coverage["global_latest_utc"] = max(latest_values).isoformat()
    coverage["global_earliest_utc"] = min(earliest_values).isoformat() if earliest_values else None
    coverage["data_root"] = str(data_root)
    coverage["source"] = "direct_1m_ohlcv_resampled_to_30m_plus"
    return bars, coverage


def build_standard_split_windows(
    *, data_end_utc: str | pd.Timestamp, validation_weeks: int, bar_minutes: int
) -> SplitWindows:
    plan = compute_standard_live_training_plan(
        data_end_utc=data_end_utc,
        validation_weeks=validation_weeks,
        bar_minutes=bar_minutes,
    )
    return SplitWindows(
        train=(
            pd.Timestamp(plan.train.start).tz_localize(None),
            pd.Timestamp(plan.train.end).tz_localize(None),
        ),
        validation=(
            pd.Timestamp(plan.validation.start).tz_localize(None),
            pd.Timestamp(plan.validation.end).tz_localize(None),
        ),
    )


def _split_mask(
    datetimes: pd.Series | pd.DatetimeIndex, split: str, windows: SplitWindows
) -> np.ndarray:
    values = pd.Series(datetimes) if not isinstance(datetimes, pd.Series) else datetimes
    start, end = getattr(windows, split)
    return ((values >= start) & (values <= end)).to_numpy()


def _frame_bounds(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"earliest": None, "latest": None}
    datetimes = pd.to_datetime(frame["datetime"])
    return {"earliest": datetimes.min().isoformat(), "latest": datetimes.max().isoformat()}


def build_train_eligibility_report(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    windows: SplitWindows,
) -> dict[str, Any]:
    """Report which symbol/timeframe pairs have physical train-window data.

    Validation-only listings are unsafe for live parameter fitting because they
    can look attractive only inside the holdout.  The standard policy is to
    exclude every symbol/timeframe with zero train bars from candidate fitting,
    portfolio allocation, and live promotion, while still recording it as
    data-only coverage for a future refit.
    """
    by_symbol: dict[str, Any] = {}
    train_ineligible_symbols: list[str] = []
    train_eligible_symbols: list[str] = []
    ineligible_symbol_timeframes: list[dict[str, Any]] = []
    for symbol in symbols:
        timeframe_payload: dict[str, Any] = {}
        eligible_timeframes: list[str] = []
        total_train_rows = 0
        total_validation_rows = 0
        for timeframe in timeframes:
            frame = bars_by_symbol_tf.get((symbol, timeframe), pd.DataFrame())
            if frame.empty:
                train_rows = 0
                validation_rows = 0
            else:
                datetimes = pd.Series(pd.to_datetime(frame["datetime"]))
                train_rows = int(_split_mask(datetimes, "train", windows).sum())
                validation_rows = int(_split_mask(datetimes, "validation", windows).sum())
            total_train_rows += train_rows
            total_validation_rows += validation_rows
            train_eligible = train_rows > 0
            if train_eligible:
                eligible_timeframes.append(str(timeframe))
            else:
                ineligible_symbol_timeframes.append(
                    {
                        "symbol": str(symbol),
                        "timeframe": str(timeframe),
                        "reason": "no_train_bars",
                    }
                )
            timeframe_payload[str(timeframe)] = {
                **_frame_bounds(frame),
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "train_eligible": train_eligible,
            }
        if eligible_timeframes:
            train_eligible_symbols.append(str(symbol))
        else:
            train_ineligible_symbols.append(str(symbol))
        by_symbol[str(symbol)] = {
            "train_eligible": bool(eligible_timeframes),
            "eligible_timeframes": eligible_timeframes,
            "train_rows_total": total_train_rows,
            "validation_rows_total": total_validation_rows,
            "timeframes": timeframe_payload,
        }
    return {
        "policy": (
            "exclude_symbol_timeframes_without_train_rows_from_parameter_fit_"
            "allocation_selection_and_live_promotion"
        ),
        "warmup_scope": "train_split_only",
        "symbol_count": len(symbols),
        "train_eligible_symbol_count": len(train_eligible_symbols),
        "train_ineligible_symbol_count": len(train_ineligible_symbols),
        "train_eligible_symbols": train_eligible_symbols,
        "train_ineligible_symbols": train_ineligible_symbols,
        "train_ineligible_symbol_timeframes": ineligible_symbol_timeframes,
        "symbols": by_symbol,
    }


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / np.maximum(peaks, 1e-12)
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
    total_return = float(np.prod(1.0 + returns) - 1.0)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    downside = returns[returns < 0.0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    annual = math.sqrt(_periods_per_year(timeframe))
    mdd = max_drawdown(returns)
    return {
        "total_return": total_return,
        "max_drawdown": mdd,
        "sharpe": mean / std * annual if std > 0.0 else 0.0,
        "sortino": mean / downside_std * annual if downside_std > 0.0 else 0.0,
        "calmar": total_return / mdd if mdd > 0.0 else 0.0,
        "trade_event_count": int(np.count_nonzero(np.abs(np.diff(np.r_[0.0, position])) > 1e-12)),
        "exposure_bar_count": int(np.count_nonzero(np.abs(position) > 1e-12)),
        "liquidation_count": int(np.count_nonzero(liquidation_flags)),
        "account_wipeout_count": int(np.count_nonzero(account_wipeout_flags)),
    }


def simulate_symbol(
    bars: pd.DataFrame,
    signal: np.ndarray,
    *,
    integer_leverage: int,
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
    notional = float(integer_leverage) * float(allocation_fraction)
    transition = np.abs(np.diff(np.r_[0.0, signal]))
    costs = (round_trip_cost_bps / 10000.0) * notional * transition / 2.0
    returns = signal * notional * next_return - costs
    long_liq = (signal > 0.0) & (
        ((low / np.maximum(close, 1e-12)) - 1.0) * integer_leverage <= -0.95
    )
    short_liq = (signal < 0.0) & (
        ((high / np.maximum(close, 1e-12)) - 1.0) * integer_leverage >= 0.95
    )
    liquidation = long_liq | short_liq
    equity = np.cumprod(1.0 + returns)
    return SimResult(
        returns=returns,
        position=signal,
        liquidation_flags=liquidation,
        account_wipeout_flags=equity <= 0.0,
    )


def _return_per_turnover(
    total_return: float, trade_events: int, notional_fraction: float
) -> float | None:
    turnover = float(trade_events) * abs(float(notional_fraction))
    if turnover <= 0.0:
        return None
    return float(total_return) * 10000.0 / turnover


def _rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=1).replace(0.0, np.nan)
    return (series - mean) / std


def _adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=high.index
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
    long_entry_values = long_entry.fillna(False).astype(bool).to_numpy()
    long_exit_values = long_exit.fillna(False).astype(bool).to_numpy()
    short_entry_values = short_entry.fillna(False).astype(bool).to_numpy()
    short_exit_values = short_exit.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(long_entry_values), dtype=float)
    state = 0.0
    bars_held = 10**9
    cooldown_remaining = 0
    for idx in range(len(out)):
        can_exit = bars_held >= min_hold_bars
        exited = False
        if can_exit and (
            (state > 0.0 and long_exit_values[idx]) or (state < 0.0 and short_exit_values[idx])
        ):
            state = 0.0
            bars_held = 0
            cooldown_remaining = cooldown_bars
            exited = True
        if state == 0.0:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
            elif not exited:
                if side in {"long_only", "long_short"} and long_entry_values[idx]:
                    state = 1.0
                    bars_held = 0
                elif side in {"short_only", "long_short"} and short_entry_values[idx]:
                    state = -1.0
                    bars_held = 0
        out[idx] = state
        if state != 0.0:
            bars_held += 1
    return out


def _candidate_score(row: Mapping[str, Any]) -> float:
    train = float(row.get("train_return") or 0.0)
    validation = float(row.get("validation_return") or 0.0)
    val_mdd = float(row.get("validation_mdd") or 0.0)
    train_rpt = float(row.get("train_return_per_turnover_proxy_bps") or 0.0)
    val_rpt = float(row.get("validation_return_per_turnover_proxy_bps") or 0.0)
    train_val_spike = max(0.0, validation - train)
    return (
        8.0 * validation
        + 1.5 * min(train, validation)
        + min(train_rpt, 100.0) / 120.0
        + min(val_rpt, 100.0) / 90.0
        - 4.0 * val_mdd
        - 5.0 * train_val_spike
    )


def gate_candidate(row: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if (
        int(row.get("train_trade_event_count") or 0)
        < PROMOTION_THRESHOLDS["min_train_trade_event_count"]
    ):
        reasons.append(f"train_trade_event_count_{row.get('train_trade_event_count')}_below_80")
    if (
        int(row.get("validation_trade_event_count") or 0)
        < PROMOTION_THRESHOLDS["min_validation_trade_event_count"]
    ):
        reasons.append(
            f"validation_trade_event_count_{row.get('validation_trade_event_count')}_below_30"
        )
    if float(row.get("validation_return") or 0.0) < PROMOTION_THRESHOLDS["min_validation_return"]:
        reasons.append(
            f"validation_return_{float(row.get('validation_return') or 0.0):.4f}_below_0.02"
        )
    if float(row.get("train_return") or 0.0) <= 0.0:
        reasons.append("train_return_not_positive")
    if float(row.get("train_return") or 0.0) < float(row.get("validation_return") or 0.0):
        reasons.append("train_return_below_validation_return_possible_validation_spike")
    if float(row.get("validation_mdd") or 0.0) > PROMOTION_THRESHOLDS["max_validation_mdd_relaxed"]:
        reasons.append(
            f"validation_mdd_{float(row.get('validation_mdd') or 0.0):.4f}_above_relaxed_0.20"
        )

    efficiency_reasons: list[str] = []
    for split in SPLIT_ORDER:
        value = row.get(f"{split}_return_per_turnover_proxy_bps")
        if value is None or float(value) <= RETURN_PER_TURNOVER_THRESHOLD_BPS:
            rendered = "missing" if value is None else f"{float(value):.3f}"
            efficiency_reasons.append(
                f"{split}_return_per_turnover_proxy_bps_{rendered}_not_above_{RETURN_PER_TURNOVER_THRESHOLD_BPS:.3f}"
            )
    sample_gate_pass = not reasons
    execution_efficiency_proxy_gate_pass = not efficiency_reasons
    strict_backtest_gate_pass = (
        sample_gate_pass
        and execution_efficiency_proxy_gate_pass
        and float(row.get("validation_mdd") or 0.0)
        <= PROMOTION_THRESHOLDS["max_validation_mdd_strict"]
    )
    relaxed_backtest_gate_pass = sample_gate_pass and execution_efficiency_proxy_gate_pass
    if strict_backtest_gate_pass:
        decision = "strict_paper_testnet_candidate_pending_forward_fill_telemetry"
    elif relaxed_backtest_gate_pass:
        decision = "relaxed_paper_testnet_candidate_pending_forward_fill_telemetry"
    elif sample_gate_pass:
        decision = "sample_pass_shadow_until_execution_efficiency"
    else:
        decision = "no_promotion_shadow_or_reject"
    row.update(
        {
            "sample_gate_pass": sample_gate_pass,
            "execution_efficiency_proxy_gate_pass": execution_efficiency_proxy_gate_pass,
            "strict_backtest_gate_pass": strict_backtest_gate_pass,
            "relaxed_backtest_gate_pass": relaxed_backtest_gate_pass,
            "decision": decision,
            "ready_for_paper": bool(strict_backtest_gate_pass or relaxed_backtest_gate_pass),
            "ready_for_real": False,
            "real_money_execution": False,
            "real_execution_allowed": False,
            "rejection_reasons": reasons + efficiency_reasons,
        }
    )
    return row


def finalize_candidate(
    base: Mapping[str, Any],
    sim: SimResult,
    datetimes: pd.Series,
    *,
    timeframe: str,
    windows: SplitWindows,
) -> dict[str, Any]:
    row = dict(base)
    for split in SPLIT_ORDER:
        mask = _split_mask(datetimes, split, windows)
        metrics = split_metrics(
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
    validation = float(row.get("validation_return") or 0.0)
    train = float(row.get("train_return") or 0.0)
    row["train_validation_return_ratio"] = train / validation if validation > 0.0 else 0.0
    row["train_minus_validation_return"] = train - validation
    notional = float(row["notional_fraction"])
    for split in SPLIT_ORDER:
        row[f"{split}_return_per_turnover_proxy_bps"] = _return_per_turnover(
            float(row[f"{split}_return"]), int(row[f"{split}_trade_event_count"]), notional
        )
    row["train_validation_score"] = _candidate_score(row)
    return gate_candidate(row)


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
    integer_leverage: int,
    allocation_fraction: float,
) -> dict[str, Any]:
    return {
        "model_id": _model_id(model_parts),
        "family": family,
        "symbol": symbol,
        "asset_group": _asset_group(symbol),
        "timeframe": timeframe,
        "side": side,
        "lookback_bars": lookback,
        "threshold": threshold,
        "exit_threshold": exit_threshold,
        "min_hold_bars": min_hold,
        "cooldown_bars": cooldown,
        "integer_leverage": int(integer_leverage),
        "allocation_fraction": float(allocation_fraction),
        "notional_fraction": float(integer_leverage) * float(allocation_fraction),
    }


def _store_stream_if_candidate(
    streams: list[CandidateStream], row: dict[str, Any], sim: SimResult, datetimes: pd.Series
) -> None:
    # Keep enough non-promoted shadow rows to analyze whether a diversified
    # hybrid can improve the broad 69-asset book. This is not a promotion gate:
    # every downstream artifact still records real-money false and paper/testnet
    # telemetry requirements.
    hybrid_stream_eligible = bool(row.get("ready_for_paper")) or (
        float(row.get("validation_return") or 0.0) > 0.0
        and int(row.get("train_trade_event_count") or 0) >= 40
        and int(row.get("validation_trade_event_count") or 0) >= 10
    )
    row["hybrid_stream_eligible"] = hybrid_stream_eligible
    if not hybrid_stream_eligible:
        return
    returns = pd.Series(
        sim.returns.astype(float), index=pd.DatetimeIndex(pd.to_datetime(datetimes))
    )
    position = pd.Series(sim.position.astype(float), index=returns.index)
    streams.append(CandidateStream(row=row, returns=returns, position=position))
    symbol = str(row.get("symbol"))
    symbol_streams = [stream for stream in streams if str(stream.row.get("symbol")) == symbol]
    if len(symbol_streams) > MAX_CACHED_STREAMS_PER_SYMBOL:
        worst = min(
            symbol_streams,
            key=lambda stream: float(stream.row.get("train_validation_score") or -1e18),
        )
        streams.remove(worst)


def _close_panel(
    bars_by_symbol: Mapping[str, pd.DataFrame], symbols: Sequence[str]
) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        frame = bars_by_symbol[symbol]
        if frame.empty:
            continue
        frames.append(frame[["datetime", "close"]].assign(symbol=symbol))
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .pivot(index="datetime", columns="symbol", values="close")
        .sort_index()
    )


def discover_candidates(
    bars_by_symbol_tf: Mapping[tuple[str, str], pd.DataFrame],
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    integer_leverages: Sequence[int],
    allocation_fraction: float,
    windows: SplitWindows,
) -> tuple[list[dict[str, Any]], list[CandidateStream]]:
    rows: list[dict[str, Any]] = []
    streams: list[CandidateStream] = []
    for timeframe in timeframes:
        bars_by_symbol = {symbol: bars_by_symbol_tf[(symbol, timeframe)] for symbol in symbols}
        panel = _close_panel(bars_by_symbol, symbols)
        if panel.empty:
            continue
        panel_returns = panel.pct_change().mean(axis=1).fillna(0.0)
        market_index = (1.0 + panel_returns).cumprod()
        for lookback in (12, 24, 48):
            momentum_panel = panel / panel.shift(lookback) - 1.0
            rank_pct = momentum_panel.rank(axis=1, ascending=False, pct=True)
            market_momentum = market_index / market_index.shift(lookback) - 1.0
            breadth = (momentum_panel > 0.0).sum(axis=1) / momentum_panel.notna().sum(
                axis=1
            ).replace(0, np.nan)
            for symbol in symbols:
                frame = bars_by_symbol[symbol]
                if frame.empty or len(frame) < lookback * 4:
                    continue
                datetimes = pd.DatetimeIndex(pd.to_datetime(frame["datetime"]))
                close = frame["close"].astype(float).reset_index(drop=True)
                high = frame["high"].astype(float).reset_index(drop=True)
                low = frame["low"].astype(float).reset_index(drop=True)
                symbol_momentum = pd.Series(
                    momentum_panel.get(symbol, pd.Series(index=panel.index, dtype=float))
                    .reindex(datetimes)
                    .to_numpy(),
                    index=frame.index,
                )
                symbol_rank = pd.Series(
                    rank_pct.get(symbol, pd.Series(index=panel.index, dtype=float))
                    .reindex(datetimes)
                    .to_numpy(),
                    index=frame.index,
                )
                market_mom = pd.Series(
                    market_momentum.reindex(datetimes).ffill().to_numpy(), index=frame.index
                )
                breadth_aligned = pd.Series(
                    breadth.reindex(datetimes).ffill().to_numpy(), index=frame.index
                )
                realized = close.pct_change().rolling(max(6, lookback // 2)).std(ddof=1)
                vol_adjusted = symbol_momentum / (realized * math.sqrt(float(lookback))).replace(
                    0.0, np.nan
                )
                adx = _adx_proxy(high, low, close, max(6, lookback // 2))

                for top_pct in (0.10, 0.20):
                    for min_hold in (6, 12):
                        long_entry = (
                            (symbol_rank <= top_pct)
                            & (symbol_momentum > 0.0)
                            & (market_mom > -0.03)
                            & (breadth_aligned >= 0.35)
                        )
                        short_entry = (
                            (symbol_rank >= 1.0 - top_pct)
                            & (symbol_momentum < 0.0)
                            & (market_mom < 0.03)
                            & (breadth_aligned <= 0.65)
                        )
                        long_exit = (symbol_rank > 0.45) | (symbol_momentum < 0.0)
                        short_exit = (symbol_rank < 0.55) | (symbol_momentum > 0.0)
                        signal = _debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage in integer_leverages:
                            sim = simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base = _candidate_base(
                                family="cross_sectional_momentum_rank",
                                model_parts=(
                                    "xsmom",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"top{top_pct}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=top_pct,
                                exit_threshold=0.45,
                                min_hold=min_hold,
                                cooldown=2,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            row = finalize_candidate(
                                base, sim, frame["datetime"], timeframe=timeframe, windows=windows
                            )
                            rows.append(row)
                            _store_stream_if_candidate(streams, row, sim, frame["datetime"])

                for threshold in (0.75, 1.25):
                    for min_hold in (6, 12):
                        common = (adx >= 12.0) & (market_mom.abs() < 0.25)
                        long_entry = (vol_adjusted > threshold) & common
                        short_entry = (vol_adjusted < -threshold) & common
                        long_exit = (vol_adjusted < 0.20) | (~common)
                        short_exit = (vol_adjusted > -0.20) | (~common)
                        signal = _debounced_state_signal(
                            long_entry,
                            long_exit,
                            short_entry,
                            short_exit,
                            side="long_short",
                            min_hold_bars=min_hold,
                            cooldown_bars=2,
                        )
                        for leverage in integer_leverages:
                            sim = simulate_symbol(
                                frame,
                                signal,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            base = _candidate_base(
                                family="volatility_adjusted_trend_persistence",
                                model_parts=(
                                    "voladj",
                                    timeframe,
                                    symbol,
                                    f"lb{lookback}",
                                    f"th{threshold}",
                                    f"hold{min_hold}",
                                    f"lev{leverage}",
                                ),
                                symbol=symbol,
                                timeframe=timeframe,
                                side="long_short",
                                lookback=lookback,
                                threshold=threshold,
                                exit_threshold=0.20,
                                min_hold=min_hold,
                                cooldown=2,
                                integer_leverage=int(leverage),
                                allocation_fraction=allocation_fraction,
                            )
                            row = finalize_candidate(
                                base, sim, frame["datetime"], timeframe=timeframe, windows=windows
                            )
                            rows.append(row)
                            _store_stream_if_candidate(streams, row, sim, frame["datetime"])

                for slow in (36, 72):
                    fast = max(6, slow // 4)
                    ema_fast = close.ewm(span=fast, adjust=False).mean()
                    ema_slow = close.ewm(span=slow, adjust=False).mean()
                    trend_slope = ema_slow / ema_slow.shift(max(2, slow // 6)) - 1.0
                    distance = (close - ema_fast) / ema_fast.replace(0.0, np.nan)
                    dist_z = _rolling_zscore(distance, max(24, slow))
                    for pullback_z in (-0.50, -1.00):
                        for min_hold in (6, 12):
                            long_entry = (
                                (ema_fast > ema_slow)
                                & (trend_slope > 0.0)
                                & (dist_z.shift(1) <= pullback_z)
                                & (close > ema_fast)
                                & (market_mom > -0.03)
                            )
                            short_entry = (
                                (ema_fast < ema_slow)
                                & (trend_slope < 0.0)
                                & (dist_z.shift(1) >= -pullback_z)
                                & (close < ema_fast)
                                & (market_mom < 0.03)
                            )
                            long_exit = (close < ema_slow) | (trend_slope < 0.0)
                            short_exit = (close > ema_slow) | (trend_slope > 0.0)
                            signal = _debounced_state_signal(
                                long_entry,
                                long_exit,
                                short_entry,
                                short_exit,
                                side="long_short",
                                min_hold_bars=min_hold,
                                cooldown_bars=2,
                            )
                            for leverage in integer_leverages:
                                sim = simulate_symbol(
                                    frame,
                                    signal,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                base = _candidate_base(
                                    family="trend_pullback_reclaim",
                                    model_parts=(
                                        "pullback",
                                        timeframe,
                                        symbol,
                                        f"slow{slow}",
                                        f"z{pullback_z}",
                                        f"hold{min_hold}",
                                        f"lev{leverage}",
                                    ),
                                    symbol=symbol,
                                    timeframe=timeframe,
                                    side="long_short",
                                    lookback=slow,
                                    threshold=pullback_z,
                                    exit_threshold=0.0,
                                    min_hold=min_hold,
                                    cooldown=2,
                                    integer_leverage=int(leverage),
                                    allocation_fraction=allocation_fraction,
                                )
                                row = finalize_candidate(
                                    base,
                                    sim,
                                    frame["datetime"],
                                    timeframe=timeframe,
                                    windows=windows,
                                )
                                rows.append(row)
                                _store_stream_if_candidate(streams, row, sim, frame["datetime"])
    return rows, streams


def _split_period_return(series: pd.Series, split: str, windows: SplitWindows) -> float:
    mask = _split_mask(pd.Series(series.index), split, windows)
    values = series.to_numpy(dtype=float)[mask]
    return float(np.prod(1.0 + values) - 1.0) if values.size else 0.0


def select_hybrid_streams(
    streams: Sequence[CandidateStream], *, max_streams: int, max_per_symbol: int
) -> list[CandidateStream]:
    sorted_streams = sorted(
        streams,
        key=lambda item: (
            bool(item.row.get("strict_backtest_gate_pass")),
            bool(item.row.get("relaxed_backtest_gate_pass")),
            float(item.row.get("train_validation_score") or -1e9),
        ),
        reverse=True,
    )
    selected: list[CandidateStream] = []
    per_symbol: Counter[str] = Counter()
    seen_models: set[str] = set()
    for stream in sorted_streams:
        symbol = str(stream.row["symbol"])
        if per_symbol[symbol] >= max_per_symbol:
            continue
        model_id = str(stream.row["model_id"])
        if model_id in seen_models:
            continue
        selected.append(stream)
        seen_models.add(model_id)
        per_symbol[symbol] += 1
        if len(selected) >= max_streams:
            break
    return selected


def _aligned_returns_matrix(streams: Sequence[CandidateStream]) -> pd.DataFrame:
    frames = [stream.returns.rename(str(stream.row["model_id"])) for stream in streams]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).fillna(0.0).sort_index()


def concentration_metrics(
    streams: Sequence[CandidateStream], weights: np.ndarray
) -> dict[str, Any]:
    gross_by_symbol: dict[str, float] = defaultdict(float)
    gross_by_group: dict[str, float] = defaultdict(float)
    gross_by_family: dict[str, float] = defaultdict(float)
    gross_by_timeframe: dict[str, float] = defaultdict(float)
    gross_by_rule_side: dict[str, float] = defaultdict(float)
    for weight, stream in zip(weights, streams, strict=True):
        notional = float(weight) * float(stream.row.get("notional_fraction") or 0.0)
        gross_by_symbol[str(stream.row["symbol"])] += notional
        gross_by_group[str(stream.row["asset_group"])] += notional
        gross_by_family[str(stream.row["family"])] += notional
        gross_by_timeframe[str(stream.row["timeframe"])] += notional
        gross_by_rule_side[str(stream.row.get("side") or "unknown")] += notional
    total = float(sum(gross_by_symbol.values()))

    def _shares(values: Mapping[str, float]) -> dict[str, float]:
        if total <= 0.0:
            return dict.fromkeys(values, 0.0)
        return {key: float(value) / total for key, value in sorted(values.items())}

    symbol_shares = _shares(gross_by_symbol)
    group_shares = _shares(gross_by_group)
    family_shares = _shares(gross_by_family)
    timeframe_shares = _shares(gross_by_timeframe)
    rule_side_shares = _shares(gross_by_rule_side)
    top_symbol, top_symbol_share = max(
        symbol_shares.items(), key=lambda kv: kv[1], default=(None, 0.0)
    )
    top_group, top_group_share = max(
        group_shares.items(), key=lambda kv: kv[1], default=(None, 0.0)
    )
    top_rule_side, top_rule_side_share = max(
        rule_side_shares.items(), key=lambda kv: kv[1], default=(None, 0.0)
    )
    hhi = float(sum(share * share for share in symbol_shares.values()))
    return {
        "total_weighted_notional_fraction": total,
        "top_symbol": top_symbol,
        "top_symbol_share": top_symbol_share,
        "top_asset_group": top_group,
        "top_asset_group_share": top_group_share,
        "top_rule_side": top_rule_side,
        "top_rule_side_share": top_rule_side_share,
        "symbol_hhi": hhi,
        "effective_symbol_count": 1.0 / hhi if hhi > 0.0 else 0.0,
        "symbol_shares": symbol_shares,
        "asset_group_shares": group_shares,
        "family_shares": family_shares,
        "timeframe_shares": timeframe_shares,
        "rule_side_shares": rule_side_shares,
        "concentration_flags": [
            flag
            for flag, active in {
                "top_symbol_share_above_35pct": top_symbol_share > 0.35,
                "top_asset_group_share_above_70pct": top_group_share > 0.70,
                "effective_symbol_count_below_4": (1.0 / hhi if hhi > 0.0 else 0.0) < 4.0,
            }.items()
            if active
        ],
    }


def _portfolio_metrics(
    portfolio_returns: pd.Series,
    exposure_proxy: pd.Series,
    *,
    timeframe: str,
    windows: SplitWindows,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    values = portfolio_returns.to_numpy(dtype=float)
    positions = (exposure_proxy.abs().to_numpy(dtype=float) > 1e-12).astype(float)
    zeros = np.zeros_like(values, dtype=bool)
    for split in SPLIT_ORDER:
        mask = _split_mask(pd.Series(portfolio_returns.index), split, windows)
        metrics = split_metrics(
            values[mask], positions[mask], zeros[mask], zeros[mask], timeframe=timeframe
        )
        for key, value in metrics.items():
            out[f"{split}_{key}"] = value
    return out


def optimize_hybrid(
    streams: Sequence[CandidateStream], *, windows: SplitWindows, n_trials: int, seed: int
) -> dict[str, Any]:
    if not streams:
        return {
            "status": "no_hybrid_streams",
            "selected": False,
            "weights": [],
            "metrics": {},
            "concentration": {},
        }
    if optuna is None:
        raise RuntimeError("Optuna is required for 69-asset hybrid refit")
    matrix = _aligned_returns_matrix(streams)
    train_mask = _split_mask(pd.Series(matrix.index), "train", windows)
    validation_mask = _split_mask(pd.Series(matrix.index), "validation", windows)
    train_values = matrix.to_numpy(dtype=float)[train_mask]
    validation_values = matrix.to_numpy(dtype=float)[validation_mask]
    notionals = np.array([float(stream.row.get("notional_fraction") or 0.0) for stream in streams])

    def _weights_from_trial(trial: Any) -> np.ndarray:
        raw = np.array(
            [trial.suggest_float(f"w_{idx}", 0.0, 1.0) for idx in range(len(streams))],
            dtype=float,
        )
        raw = np.where(raw < 0.05, 0.0, raw)
        if not np.isfinite(raw).all() or float(raw.sum()) <= 0.0:
            raw = np.ones(len(streams), dtype=float)
        return raw / float(raw.sum())

    def _objective(trial: Any) -> float:
        weights = _weights_from_trial(trial)
        train_returns = train_values @ weights
        validation_returns = validation_values @ weights
        train_return = float(np.prod(1.0 + train_returns) - 1.0) if train_returns.size else 0.0
        validation_return = (
            float(np.prod(1.0 + validation_returns) - 1.0) if validation_returns.size else 0.0
        )
        validation_mdd = max_drawdown(validation_returns)
        train_mdd = max_drawdown(train_returns)
        concentration = concentration_metrics(streams, weights)
        top_symbol_share = float(concentration.get("top_symbol_share") or 0.0)
        top_group_share = float(concentration.get("top_asset_group_share") or 0.0)
        gross = float(np.dot(weights, notionals))
        spike_penalty = max(0.0, validation_return - train_return)
        mdd_penalty = max(0.0, validation_mdd - 0.20) * 20.0 + max(0.0, train_mdd - 0.50) * 2.0
        concentration_penalty = (
            max(0.0, top_symbol_share - 0.35) * 3.0 + max(0.0, top_group_share - 0.70) * 1.5
        )
        gross_penalty = max(0.0, gross - 5.0) * 2.0
        return (
            9.0 * validation_return
            + 1.5 * min(train_return, validation_return)
            - 3.0 * validation_mdd
            - 4.0 * spike_penalty
            - mdd_penalty
            - concentration_penalty
            - gross_penalty
        )

    equal = {f"w_{idx}": 1.0 for idx in range(len(streams))}
    study = run_optuna_study(
        optuna_module=optuna,
        objective=_objective,
        n_trials=n_trials,
        direction="maximize",
        seed=seed,
        enqueue_trials=[equal],
        n_jobs=1,
        show_progress_bar=False,
    )
    best_raw = np.array(
        [float(study.best_params.get(f"w_{idx}", 0.0)) for idx in range(len(streams))]
    )
    best_raw = np.where(best_raw < 0.05, 0.0, best_raw)
    if float(best_raw.sum()) <= 0.0:
        best_raw = np.ones(len(streams), dtype=float)
    weights = best_raw / float(best_raw.sum())
    portfolio_returns = pd.Series(matrix.to_numpy(dtype=float) @ weights, index=matrix.index)
    exposure_proxy = pd.Series(np.zeros(len(matrix), dtype=float), index=matrix.index)
    for weight, stream in zip(weights, streams, strict=True):
        exposure_proxy = exposure_proxy.add(
            stream.position.reindex(matrix.index).fillna(0.0) * weight, fill_value=0.0
        )
    metrics = _portfolio_metrics(portfolio_returns, exposure_proxy, timeframe="1h", windows=windows)
    for split in SPLIT_ORDER:
        split_mask = _split_mask(pd.Series(matrix.index), split, windows)
        active_component_events = 0
        weighted_turnover = 0.0
        weighted_long_exposure_bar_notional = 0.0
        weighted_short_exposure_bar_notional = 0.0
        for weight, stream in zip(weights, streams, strict=True):
            if float(weight) <= 1e-6:
                continue
            row = stream.row
            events = int(row.get(f"{split}_trade_event_count") or 0)
            notional = abs(float(weight)) * abs(float(row.get("notional_fraction") or 0.0))
            position_values = (
                stream.position.reindex(matrix.index).fillna(0.0).to_numpy(dtype=float)
            )
            split_positions = position_values[split_mask]
            active_component_events += events
            weighted_turnover += notional * events
            weighted_long_exposure_bar_notional += notional * int(np.sum(split_positions > 0.0))
            weighted_short_exposure_bar_notional += notional * int(np.sum(split_positions < 0.0))
        split_return = float(metrics.get(f"{split}_total_return") or 0.0)
        directional_total = (
            weighted_long_exposure_bar_notional + weighted_short_exposure_bar_notional
        )
        metrics[f"{split}_component_trade_event_count"] = active_component_events
        metrics[f"{split}_weighted_turnover_proxy"] = weighted_turnover
        metrics[f"{split}_return_per_turnover_proxy_bps"] = (
            split_return * 10000.0 / weighted_turnover if weighted_turnover > 0.0 else None
        )
        metrics[f"{split}_weighted_long_exposure_bar_notional"] = (
            weighted_long_exposure_bar_notional
        )
        metrics[f"{split}_weighted_short_exposure_bar_notional"] = (
            weighted_short_exposure_bar_notional
        )
        metrics[f"{split}_long_exposure_share_when_active"] = (
            weighted_long_exposure_bar_notional / directional_total
            if directional_total > 0.0
            else None
        )
        metrics[f"{split}_short_exposure_share_when_active"] = (
            weighted_short_exposure_bar_notional / directional_total
            if directional_total > 0.0
            else None
        )
    concentration = concentration_metrics(streams, weights)
    return {
        "status": "optimized",
        "selected": True,
        "best_value": float(study.best_value),
        "n_trials": int(n_trials),
        "best_params": dict(study.best_params),
        "weights": weights.tolist(),
        "metrics": metrics,
        "concentration": concentration,
        "portfolio_returns": portfolio_returns,
        "returns_matrix": matrix,
        "top_trials": [
            {
                "number": int(trial.number),
                "value": float(trial.value) if trial.value is not None else None,
                "state": str(trial.state),
            }
            for trial in sorted(
                study.trials,
                key=lambda item: -1e18 if item.value is None else float(item.value),
                reverse=True,
            )[:10]
        ],
    }


def attribution_rows(
    streams: Sequence[CandidateStream], weights: np.ndarray, *, windows: SplitWindows
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for weight, stream in zip(weights, streams, strict=True):
        row = stream.row
        for dimension, key in (
            ("symbol", str(row["symbol"])),
            ("asset_group", str(row["asset_group"])),
            ("family", str(row["family"])),
            ("timeframe", str(row["timeframe"])),
            ("rule_side", str(row.get("side") or "unknown")),
        ):
            bucket = buckets.setdefault(
                (dimension, key),
                {
                    "dimension": dimension,
                    "key": key,
                    "weight_sum": 0.0,
                    "weighted_notional_fraction": 0.0,
                    "train_simple_pnl_contribution": 0.0,
                    "validation_simple_pnl_contribution": 0.0,
                    "candidate_count": 0,
                },
            )
            bucket["weight_sum"] += float(weight)
            bucket["weighted_notional_fraction"] += float(weight) * float(
                row.get("notional_fraction") or 0.0
            )
            bucket["train_simple_pnl_contribution"] += float(
                stream.returns[_split_mask(pd.Series(stream.returns.index), "train", windows)].sum()
            ) * float(weight)
            bucket["validation_simple_pnl_contribution"] += float(
                stream.returns[
                    _split_mask(pd.Series(stream.returns.index), "validation", windows)
                ].sum()
            ) * float(weight)
            bucket["candidate_count"] += 1
    return sorted(
        buckets.values(),
        key=lambda item: (
            item["dimension"],
            -float(item["weighted_notional_fraction"]),
            item["key"],
        ),
    )


def selected_weight_rows(
    streams: Sequence[CandidateStream], weights: Sequence[float]
) -> list[dict[str, Any]]:
    rows = []
    for idx, (stream, weight) in enumerate(
        sorted(zip(streams, weights, strict=True), key=lambda item: item[1], reverse=True), start=1
    ):
        row = dict(stream.row)
        row.update(
            {
                "hybrid_weight_rank": idx,
                "weight": float(weight),
                "weighted_notional_fraction": float(weight)
                * float(stream.row.get("notional_fraction") or 0.0),
            }
        )
        rows.append(row)
    return rows


def _correlation_rows(
    matrix: pd.DataFrame, selected_rows: Sequence[Mapping[str, Any]], *, windows: SplitWindows
) -> list[dict[str, Any]]:
    if matrix.empty:
        return []
    tv_mask = _split_mask(pd.Series(matrix.index), "train", windows) | _split_mask(
        pd.Series(matrix.index), "validation", windows
    )
    corr = matrix.loc[tv_mask].corr().fillna(0.0)
    meta = {str(row["model_id"]): row for row in selected_rows}
    rows: list[dict[str, Any]] = []
    ids = list(corr.columns)
    for left in ids:
        for right in ids:
            left_meta = meta.get(left, {})
            right_meta = meta.get(right, {})
            rows.append(
                {
                    "left_model_id": left,
                    "right_model_id": right,
                    "correlation_train_validation": float(corr.loc[left, right]),
                    "left_symbol": left_meta.get("symbol"),
                    "right_symbol": right_meta.get("symbol"),
                    "left_family": left_meta.get("family"),
                    "right_family": right_meta.get("family"),
                }
            )
    return rows


def _candidate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_decision = Counter(str(row.get("decision")) for row in rows)
    by_symbol_candidates = Counter(
        str(row.get("symbol"))
        for row in rows
        if row.get("ready_for_paper") or row.get("sample_gate_pass")
    )
    best_by_symbol: dict[str, dict[str, Any]] = {}
    for row in sorted(
        rows, key=lambda item: float(item.get("train_validation_score") or -1e9), reverse=True
    ):
        symbol = str(row.get("symbol"))
        if symbol not in best_by_symbol:
            best_by_symbol[symbol] = {
                "model_id": row.get("model_id"),
                "family": row.get("family"),
                "timeframe": row.get("timeframe"),
                "decision": row.get("decision"),
                "train_return": row.get("train_return"),
                "validation_return": row.get("validation_return"),
                "validation_mdd": row.get("validation_mdd"),
                "score": row.get("train_validation_score"),
            }
    return {
        "candidate_count": len(rows),
        "decision_counts": dict(by_decision),
        "symbols_with_sample_or_paper_rows": len(by_symbol_candidates),
        "sample_or_paper_rows_by_symbol_top20": dict(by_symbol_candidates.most_common(20)),
        "best_by_symbol": best_by_symbol,
    }


def gate_selected_hybrid(metrics: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    efficiency_reasons: list[str] = []
    train_return = float(metrics.get("train_total_return") or 0.0)
    validation_return = float(metrics.get("validation_total_return") or 0.0)
    validation_mdd = float(metrics.get("validation_max_drawdown") or 0.0)
    train_events = int(
        metrics.get("train_component_trade_event_count")
        or metrics.get("train_trade_event_count")
        or 0
    )
    validation_events = int(
        metrics.get("validation_component_trade_event_count")
        or metrics.get("validation_trade_event_count")
        or 0
    )
    if train_return <= 0.0:
        reasons.append("hybrid_train_return_not_positive")
    if validation_return < PROMOTION_THRESHOLDS["min_validation_return"]:
        reasons.append(f"hybrid_validation_return_{validation_return:.4f}_below_0.02")
    if train_return < validation_return:
        reasons.append("hybrid_train_return_below_validation_return_possible_validation_spike")
    if validation_mdd > PROMOTION_THRESHOLDS["max_validation_mdd_relaxed"]:
        reasons.append(f"hybrid_validation_mdd_{validation_mdd:.4f}_above_relaxed_0.20")
    if train_events < PROMOTION_THRESHOLDS["min_train_trade_event_count"]:
        reasons.append("hybrid_train_trade_event_count_below_80")
    if validation_events < PROMOTION_THRESHOLDS["min_validation_trade_event_count"]:
        reasons.append("hybrid_validation_trade_event_count_below_30")
    for split in SPLIT_ORDER:
        value = metrics.get(f"{split}_return_per_turnover_proxy_bps")
        if value is None or float(value) <= RETURN_PER_TURNOVER_THRESHOLD_BPS:
            rendered = "missing" if value is None else f"{float(value):.3f}"
            efficiency_reasons.append(
                f"hybrid_{split}_return_per_turnover_proxy_bps_{rendered}_not_above_{RETURN_PER_TURNOVER_THRESHOLD_BPS:.3f}"
            )
    relaxed = not reasons and not efficiency_reasons
    strict = relaxed and validation_mdd <= PROMOTION_THRESHOLDS["max_validation_mdd_strict"]
    return {
        "strict_backtest_gate_pass": strict,
        "relaxed_backtest_gate_pass": relaxed,
        "ready_for_paper": relaxed,
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "rejection_reasons": reasons + efficiency_reasons,
    }


def _load_prior_standard_refit(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = (
        payload.get("selection_evidence_profile")
        or payload.get("selected_optuna_hybrid_profile")
        or {}
    )
    if not isinstance(evidence, Mapping):
        return None
    return {
        "path": str(path),
        "profile_id": evidence.get("profile_id"),
        "train_return": evidence.get("train_return"),
        "validation_return": evidence.get("validation_return"),
        "validation_mdd": evidence.get("validation_mdd"),
        "gross_notional_fraction": evidence.get("gross_notional_fraction"),
        "train_rpt_bps": evidence.get("train_return_per_turnover_proxy_bps"),
        "validation_rpt_bps": evidence.get("validation_return_per_turnover_proxy_bps"),
        "active_weights": evidence.get("final_weights") or evidence.get("weights"),
    }


def _markdown_summary(payload: Mapping[str, Any]) -> str:
    hybrid = dict(payload.get("selected_hybrid") or {})
    metrics = dict(hybrid.get("metrics") or {})
    concentration = dict(hybrid.get("concentration") or {})
    lines = [
        "# Alpha Zoo 69-asset Optuna hybrid refit",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- universe: `{payload.get('universe', {}).get('symbol_count')}` symbols",
        f"- timeframes: `{', '.join(payload.get('timeframes', []))}`",
        "- execution: paper/testnet research only; real-money remains blocked.",
        "- data source: direct 1m OHLCV resampled to >=30m bars.",
        "",
        "## Split policy",
        "",
        "Latest 8 complete weeks are validation. Locked OOS/test set is disabled for this live final-refit mode; post-freeze paper/testnet forward telemetry is required before any real review.",
        "",
        "## Candidate summary",
        "",
    ]
    summary = dict(payload.get("candidate_summary") or {})
    lines.append(f"- candidate_count: `{summary.get('candidate_count')}`")
    lines.append(f"- decision_counts: `{summary.get('decision_counts')}`")
    lines.append("")
    lines.extend(
        [
            "## Selected hybrid",
            "",
            f"- status: `{hybrid.get('status')}`",
            f"- backtest gate: `{hybrid.get('backtest_gate')}`",
            f"- train return: `{float(metrics.get('train_total_return') or 0.0):.4%}`",
            f"- validation return: `{float(metrics.get('validation_total_return') or 0.0):.4%}`",
            f"- validation MDD: `{float(metrics.get('validation_max_drawdown') or 0.0):.4%}`",
            f"- train RPT proxy: `{metrics.get('train_return_per_turnover_proxy_bps')}` bps",
            f"- validation RPT proxy: `{metrics.get('validation_return_per_turnover_proxy_bps')}` bps",
            f"- top symbol share: `{float(concentration.get('top_symbol_share') or 0.0):.2%}` (`{concentration.get('top_symbol')}`)",
            f"- top rule-side share: `{float(concentration.get('top_rule_side_share') or 0.0):.2%}` (`{concentration.get('top_rule_side')}`)",
            f"- validation long/short exposure share: `{metrics.get('validation_long_exposure_share_when_active')}` / `{metrics.get('validation_short_exposure_share_when_active')}`",
            f"- effective symbol count: `{float(concentration.get('effective_symbol_count') or 0.0):.2f}`",
            f"- concentration flags: `{concentration.get('concentration_flags')}`",
            "",
            "## Real-money status",
            "",
            "`ready_for_real=false`, `real_money_execution=false`, and `real_execution_allowed=false` remain hard-false. The path to real requires paper/testnet fill/BBO/slippage/protective-order/reconciliation telemetry, not only this backtest/refit artifact.",
        ]
    )
    return "\n".join(lines) + "\n"


def _paper_testnet_handoff_summary(payload: Mapping[str, Any]) -> str:
    hybrid = dict(payload.get("selected_hybrid") or {})
    metrics = dict(hybrid.get("metrics") or {})
    concentration = dict(hybrid.get("concentration") or {})
    gate = dict(hybrid.get("backtest_gate") or {})
    return "\n".join(
        [
            "# 69-asset Alpha Zoo challenger paper/testnet handoff",
            "",
            "Status: backtest-gated paper/testnet challenger only. Real-money remains blocked.",
            "",
            "## Decision",
            "",
            f"- ready_for_paper: `{payload.get('ready_for_paper')}`",
            f"- ready_for_real: `{payload.get('ready_for_real')}`",
            f"- real_money_execution: `{payload.get('real_money_execution')}`",
            f"- real_execution_allowed: `{payload.get('real_execution_allowed')}`",
            f"- backtest_gate: `{gate}`",
            "",
            "## Backtest evidence",
            "",
            f"- train return: `{float(metrics.get('train_total_return') or 0.0):.4%}`",
            f"- validation return: `{float(metrics.get('validation_total_return') or 0.0):.4%}`",
            f"- validation MDD: `{float(metrics.get('validation_max_drawdown') or 0.0):.4%}`",
            f"- train/validation RPT proxy: `{metrics.get('train_return_per_turnover_proxy_bps')}` / `{metrics.get('validation_return_per_turnover_proxy_bps')}` bps",
            f"- component trade events train/validation: `{metrics.get('train_component_trade_event_count')}` / `{metrics.get('validation_component_trade_event_count')}`",
            f"- liquidation/account wipeout train/validation: `{metrics.get('train_liquidation_count')}`/`{metrics.get('train_account_wipeout_count')}` and `{metrics.get('validation_liquidation_count')}`/`{metrics.get('validation_account_wipeout_count')}`",
            "",
            "## Concentration evidence",
            "",
            f"- total weighted notional fraction: `{concentration.get('total_weighted_notional_fraction')}`",
            f"- top symbol: `{concentration.get('top_symbol')}` at `{float(concentration.get('top_symbol_share') or 0.0):.2%}`",
            f"- top asset group: `{concentration.get('top_asset_group')}` at `{float(concentration.get('top_asset_group_share') or 0.0):.2%}`",
            f"- effective symbol count: `{float(concentration.get('effective_symbol_count') or 0.0):.2f}`",
            f"- validation long/short exposure share: `{metrics.get('validation_long_exposure_share_when_active')}` / `{metrics.get('validation_short_exposure_share_when_active')}`",
            f"- concentration flags: `{concentration.get('concentration_flags')}`",
            "",
            "## Required before any live/real transition",
            "",
            "- Build or wire a dedicated live/paper adapter for this 69-asset rule set before exchange-connected paper execution.",
            "- Run only paper/testnet with limit-first order settings and hard real-money vetoes.",
            "- Collect 2-4 weeks of realized fill/BBO/spread/fee/slippage telemetry and compare all-in round-trip cost against the 10bps replay assumption.",
            "- Verify intended-vs-actual notional parity, partial/cancel/timeout rates, protective-order attach/reconciliation, liquidation-distance/margin buffers, and ongoing asset/direction concentration.",
            "- Do not set `ready_for_real=true` or `real_money_execution=true` from this artifact alone.",
            "",
        ]
    )


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    symbols = _parse_csv(args.symbols)
    timeframes = _parse_timeframes(args.timeframes)
    integer_leverages = _parse_ints(args.integer_leverages)
    bars, coverage = load_all_bars(symbols, data_root=data_root, timeframes=timeframes)
    split_windows = build_standard_split_windows(
        data_end_utc=str(coverage["global_latest_utc"]),
        validation_weeks=int(args.validation_weeks),
        bar_minutes=60,
    )
    rows, streams = discover_candidates(
        bars,
        symbols=symbols,
        timeframes=timeframes,
        integer_leverages=integer_leverages,
        allocation_fraction=float(args.allocation_fraction),
        windows=split_windows,
    )
    ranked_rows = sorted(
        rows, key=lambda row: float(row.get("train_validation_score") or -1e9), reverse=True
    )
    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx
    hybrid_streams = select_hybrid_streams(
        streams,
        max_streams=int(args.max_hybrid_streams),
        max_per_symbol=int(args.max_streams_per_symbol),
    )
    hybrid = optimize_hybrid(
        hybrid_streams,
        windows=split_windows,
        n_trials=int(args.n_trials),
        seed=int(args.seed),
    )
    weights = np.array(hybrid.get("weights") or [], dtype=float)
    selected_rows = selected_weight_rows(hybrid_streams, weights) if weights.size else []
    attribution = (
        attribution_rows(hybrid_streams, weights, windows=split_windows) if weights.size else []
    )
    matrix = hybrid.get("returns_matrix")
    corr_rows = (
        _correlation_rows(matrix, selected_rows, windows=split_windows)
        if isinstance(matrix, pd.DataFrame)
        else []
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_json = output_dir / "alpha_zoo_69_asset_optuna_hybrid_refit_latest.json"
    timestamped_json = output_dir / f"alpha_zoo_69_asset_optuna_hybrid_refit_{_timestamp()}.json"
    latest_md = output_dir / "alpha_zoo_69_asset_optuna_hybrid_refit_latest.md"
    candidates_csv = output_dir / "alpha_zoo_69_asset_candidates_latest.csv"
    weights_csv = output_dir / "alpha_zoo_69_asset_selected_weights_latest.csv"
    attribution_csv = output_dir / "alpha_zoo_69_asset_selected_attribution_latest.csv"
    corr_csv = output_dir / "alpha_zoo_69_asset_selected_corr_train_validation_latest.csv"
    handoff_md = output_dir / "alpha_zoo_69_asset_paper_testnet_handoff_latest.md"

    hybrid_payload = {
        key: value
        for key, value in hybrid.items()
        if key not in {"portfolio_returns", "returns_matrix"}
    }
    hybrid_gate = gate_selected_hybrid(dict(hybrid_payload.get("metrics") or {}))
    hybrid_payload["backtest_gate"] = hybrid_gate
    payload = {
        "artifact_kind": "alpha_zoo_69_asset_optuna_hybrid_refit",
        "generated_at": _utc_now_iso(),
        "ready_for_paper": bool(hybrid_gate["ready_for_paper"]),
        "ready_for_real": False,
        "real_money_execution": False,
        "real_execution_allowed": False,
        "paper_testnet_only": True,
        "research_primary_round_trip_cost_bps": PRIMARY_ROUND_TRIP_COST_BPS,
        "avg_bbo_spread_bps_assumption": AVG_BBO_SPREAD_BPS_ASSUMPTION,
        "return_per_turnover_threshold_bps": RETURN_PER_TURNOVER_THRESHOLD_BPS,
        "universe": {
            "symbol_count": len(symbols),
            "symbols": list(symbols),
            "source": "BINANCE_EXTENDED_RESEARCH_SYMBOLS",
        },
        "timeframes": list(timeframes),
        "integer_leverages": list(integer_leverages),
        "allocation_fraction_per_candidate": float(args.allocation_fraction),
        "data_coverage": coverage,
        "split_policy": split_windows.as_payload(),
        "optimization_policy": optimization_search_policy_payload(
            search_method="optuna_tpe",
            selection_inputs=TRAIN_VAL_SELECTION_INPUTS,
            objective_policy={
                "objective": "validation_return_plus_train_support_minus_mdd_and_concentration_penalties",
                "locked_oos": "disabled_for_live_final_refit",
                "candidate_stream_cap": int(args.max_hybrid_streams),
                "per_symbol_stream_cap": int(args.max_streams_per_symbol),
            },
            extra={
                "n_trials": int(args.n_trials),
                "seed": int(args.seed),
                "uses_test_set": False,
            },
        ),
        "promotion_thresholds": dict(PROMOTION_THRESHOLDS),
        "candidate_summary": _candidate_summary(ranked_rows),
        "selected_hybrid": hybrid_payload,
        "selected_weight_rows": selected_rows,
        "attribution_rows": attribution,
        "prior_standard_refit_comparison": _load_prior_standard_refit(
            Path(args.prior_standard_refit).expanduser().resolve()
        ),
        "real_transition_status": {
            "status": "blocked_pending_forward_paper_testnet_telemetry",
            "real_money_flags_intentionally_false": True,
            "required_before_real_review": [
                "two_to_four_weeks_paper_or_testnet_forward_fill_telemetry",
                "realized_bbo_spread_fee_slippage_all_in_cost_bps",
                "limit_order_fill_latency_partial_cancel_timeout_rates",
                "protective_stop_take_profit_attach_success_and_reconciliation",
                "intended_vs_actual_notional_parity",
                "liquidation_distance_margin_buffer_and_reconciliation_drift",
            ],
        },
        "output_paths": {
            "latest_json": str(latest_json),
            "timestamped_json": str(timestamped_json),
            "latest_md": str(latest_md),
            "candidates_csv": str(candidates_csv),
            "selected_weights_csv": str(weights_csv),
            "selected_attribution_csv": str(attribution_csv),
            "selected_corr_csv": str(corr_csv),
            "paper_testnet_handoff_md": str(handoff_md),
        },
        "runner_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    _write_csv(candidates_csv, ranked_rows, CANDIDATE_FIELDS)
    _write_csv(weights_csv, selected_rows, SELECTED_WEIGHT_FIELDS)
    _write_csv(attribution_csv, attribution, ATTRIBUTION_FIELDS)
    _write_csv(
        corr_csv,
        corr_rows,
        [
            "left_model_id",
            "right_model_id",
            "correlation_train_validation",
            "left_symbol",
            "right_symbol",
            "left_family",
            "right_family",
        ],
    )
    _write_json(latest_json, payload)
    _write_json(timestamped_json, payload)
    latest_md.write_text(_markdown_summary(payload), encoding="utf-8")
    handoff_md.write_text(_paper_testnet_handoff_summary(payload), encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbols", default=",".join(BINANCE_EXTENDED_RESEARCH_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument(
        "--integer-leverages", default=",".join(str(item) for item in DEFAULT_LEVERAGES)
    )
    parser.add_argument("--allocation-fraction", type=float, default=DEFAULT_ALLOCATION_FRACTION)
    parser.add_argument("--validation-weeks", type=int, default=STANDARD_VALIDATION_WEEKS)
    parser.add_argument("--max-hybrid-streams", type=int, default=48)
    parser.add_argument("--max-streams-per-symbol", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--prior-standard-refit", default=str(DEFAULT_PRIOR_STANDARD_REFIT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    selected = payload.get("selected_hybrid") or {}
    metrics = dict(selected.get("metrics") or {})
    printed_candidate_summary = {
        key: value
        for key, value in dict(payload["candidate_summary"]).items()
        if key != "best_by_symbol"
    }
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "candidate_summary": printed_candidate_summary,
                    "selected_hybrid_status": selected.get("status"),
                    "selected_hybrid_metrics": metrics,
                    "selected_hybrid_concentration": selected.get("concentration"),
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
