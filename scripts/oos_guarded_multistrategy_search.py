"""Leakage-safe multi-strategy search with final OOS holdout.

This script enforces:
- OOS is excluded from parameter search.
- Parameters are selected using train/validation only.
- Final OOS is evaluated once for selected candidates.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ccxt
import numpy as np
import polars as pl
from numpy.random import Generator

try:
    import optuna
    from optuna.trial import TrialState

    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    TrialState = None
    OPTUNA_AVAILABLE = False

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.market_data import load_data_dict_from_db
from lumina_quant.utils.performance import (
    create_drawdowns,
    create_sharpe_ratio,
    create_sortino_ratio,
)
from strategies.lag_convergence import LagConvergenceStrategy
from strategies.mean_reversion_std import MeanReversionStdStrategy
from strategies.pair_trading_zscore import PairTradingZScoreStrategy
from strategies.rolling_breakout import RollingBreakoutStrategy
from strategies.topcap_tsmom import TopCapTimeSeriesMomentumStrategy
from strategies.vwap_reversion import VwapReversionStrategy


@dataclass(slots=True)
class RunResult:
    metrics: dict
    curve: dict


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _annual_periods_for_timeframe(timeframe: str) -> int:
    mapping = {
        "1s": 31536000,
        "1m": 525600,
        "5m": 105120,
        "15m": 35040,
        "30m": 17520,
        "1h": 8760,
        "4h": 2190,
        "1d": 365,
    }
    return int(mapping.get(str(timeframe).strip().lower(), 252))


def _bars_per_day_for_timeframe(timeframe: str) -> float:
    mapping = {
        "1s": 86400.0,
        "1m": 1440.0,
        "5m": 288.0,
        "15m": 96.0,
        "30m": 48.0,
        "1h": 24.0,
        "4h": 6.0,
        "1d": 1.0,
    }
    return float(mapping.get(str(timeframe).strip().lower(), 1.0))


def _coverage_days_from_bounds(
    first: datetime | None,
    last: datetime | None,
    *,
    start_date: datetime,
    end_date: datetime,
) -> float:
    def _normalize(value: datetime | None) -> datetime | None:
        if not isinstance(value, datetime):
            return None
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)

    first_norm = _normalize(first)
    last_norm = _normalize(last)
    start_norm = _normalize(start_date)
    end_norm = _normalize(end_date)
    if first_norm is None or last_norm is None:
        return 0.0
    if start_norm is None or end_norm is None:
        return 0.0
    window_start = max(first_norm, start_norm)
    window_end = min(last_norm, end_norm)
    delta = window_end - window_start
    if not hasattr(delta, "total_seconds"):
        return 0.0
    return max(0.0, float(delta.total_seconds()) / 86400.0)


def _returns_from_curve(curve: dict) -> dict:
    datetimes = curve.get("datetime", [])
    totals = curve.get("total", [])
    out = {}
    if len(datetimes) < 2:
        return out

    def _ts_key(raw) -> int | None:
        if isinstance(raw, datetime):
            value = raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
            return int(value.timestamp())
        if isinstance(raw, np.datetime64):
            seconds = raw.astype("datetime64[s]").astype("int64")
            return int(seconds)
        text = str(raw)
        if not text:
            return None
        with contextlib.suppress(Exception):
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            parsed = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
            return int(parsed.timestamp())
        return None

    compressed_keys: list[int] = []
    compressed_totals: list[float] = []
    for idx in range(len(datetimes)):
        key = _ts_key(datetimes[idx])
        if key is None:
            continue
        total = _safe_float(totals[idx], 0.0)
        if compressed_keys and key == compressed_keys[-1]:
            compressed_totals[-1] = total
        else:
            compressed_keys.append(key)
            compressed_totals.append(total)

    if len(compressed_keys) < 2:
        return out

    prev = compressed_totals[0]
    for idx in range(1, len(compressed_keys)):
        cur = compressed_totals[idx]
        if prev > 0.0:
            out[compressed_keys[idx]] = (cur / prev) - 1.0
        prev = cur
    return out


def _metrics_from_returns(returns: list[float], annual_periods: int) -> dict:
    if not returns:
        return {
            "return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "mdd": 0.0,
            "bars": 0,
        }

    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "mdd": 0.0,
            "bars": 0,
        }

    equity = np.cumprod(1.0 + arr)
    drawdowns, _ = create_drawdowns(equity.tolist())
    max_dd = float(max(drawdowns) if drawdowns else 0.0)

    return {
        "return": float(equity[-1] - 1.0),
        "sharpe": float(create_sharpe_ratio(arr, periods=annual_periods)),
        "sortino": float(create_sortino_ratio(arr, periods=annual_periods)),
        "mdd": max_dd,
        "bars": int(arr.size),
    }


def _align_returns(
    return_maps: list[dict], *, mode: str = "intersection"
) -> tuple[list, np.ndarray]:
    if not return_maps:
        return [], np.zeros((0, 0), dtype=np.float64)

    match_mode = str(mode).strip().lower()
    if match_mode == "union":
        keys = set()
        for current in return_maps:
            keys |= set(current.keys())
    else:
        keys = set(return_maps[0].keys())
        for current in return_maps[1:]:
            keys &= set(current.keys())

    if not keys:
        return [], np.zeros((0, 0), dtype=np.float64)

    ordered = sorted(keys)
    matrix = np.asarray(
        [
            [_safe_float(return_maps[row].get(dt, 0.0), 0.0) for dt in ordered]
            for row in range(len(return_maps))
        ],
        dtype=np.float64,
    )
    return ordered, matrix


def _score_hurdle(metrics: dict, hurdle_return: float) -> float:
    excess_return = _safe_float(metrics.get("return", 0.0), 0.0) - _safe_float(hurdle_return, 0.0)
    return (
        excess_return * 220.0
        + _safe_float(metrics.get("sortino"), 0.0) * 1.8
        + _safe_float(metrics.get("sharpe"), 0.0) * 0.7
        - _safe_float(metrics.get("mdd"), 0.0) * 75.0
    )


def _period_days(start_date: datetime, end_date: datetime) -> float:
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        return 0.0
    seconds = (end_date - start_date).total_seconds()
    return max(0.0, float(seconds) / 86400.0)


def _annual_floor_return(days: float, annual_floor: float) -> float:
    if days <= 0.0:
        return 0.0
    floor = max(0.0, float(annual_floor))
    return float((1.0 + floor) ** (float(days) / 365.0) - 1.0)


def _benchmark_period_return(
    *,
    db_path: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    cache: dict,
) -> float:
    key = (
        str(db_path),
        str(exchange).strip().lower(),
        str(symbol).strip().upper(),
        str(timeframe).strip().lower(),
        start_date.isoformat() if isinstance(start_date, datetime) else str(start_date),
        end_date.isoformat() if isinstance(end_date, datetime) else str(end_date),
    )
    if key in cache:
        return _safe_float(cache[key], 0.0)

    frame = load_data_dict_from_db(
        db_path,
        exchange=exchange,
        symbol_list=[symbol],
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    ).get(symbol)

    out = 0.0
    if isinstance(frame, pl.DataFrame) and frame.height >= 2:
        first = _safe_float(frame["close"][0], 0.0)
        last = _safe_float(frame["close"][-1], 0.0)
        if first > 0.0:
            out = float((last / first) - 1.0)

    cache[key] = out
    return out


def _split_hurdle(
    *,
    db_path: str,
    exchange: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    annual_floor: float,
    benchmark_symbol: str,
    benchmark_cache: dict,
) -> dict:
    days = _period_days(start_date, end_date)
    floor_return = _annual_floor_return(days, annual_floor)
    benchmark_return = _benchmark_period_return(
        db_path=db_path,
        exchange=exchange,
        symbol=benchmark_symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        cache=benchmark_cache,
    )
    hurdle_return = max(floor_return, benchmark_return)
    return {
        "start": start_date.isoformat() if isinstance(start_date, datetime) else None,
        "end": end_date.isoformat() if isinstance(end_date, datetime) else None,
        "days": float(days),
        "annual_floor": float(annual_floor),
        "floor_return": float(floor_return),
        "benchmark_symbol": str(benchmark_symbol),
        "benchmark_return": float(benchmark_return),
        "hurdle_return": float(hurdle_return),
    }


def _hurdle_fields(metrics: dict, hurdle: dict) -> dict:
    realized = _safe_float(metrics.get("return", 0.0), 0.0)
    hurdle_return = _safe_float(hurdle.get("hurdle_return", 0.0), 0.0)
    excess = realized - hurdle_return
    return {
        "realized_return": float(realized),
        "hurdle_return": float(hurdle_return),
        "excess_return": float(excess),
        "pass": bool(excess >= 0.0),
        "score": float(_score_hurdle(metrics, hurdle_return)),
    }


def _build_topcap_universe(
    *,
    desired_count: int = 20,
    candidate_count: int = 120,
    market_type: str = "future",
) -> list[str]:
    stable = {
        "USDT",
        "USDC",
        "USDS",
        "USDE",
        "DAI",
        "FDUSD",
        "USD1",
        "TUSD",
        "BUSD",
        "PYUSD",
        "FRAX",
        "USD0",
        "USDD",
    }
    blacklist = {"WBT", "LEO", "WBTC", "STETH", "WEETH", "WETH", "USDT0", "FIGR_HELOC", "CC"}

    import urllib.request

    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        f"?vs_currency=usd&order=market_cap_desc&per_page={int(candidate_count)}&page=1&sparkline=false"
    )
    payload = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())

    exchange = ccxt.binance({"enableRateLimit": True})
    options = dict(getattr(exchange, "options", {}) or {})
    options["defaultType"] = str(market_type).strip().lower()
    exchange.options = options
    market_symbols = set(exchange.load_markets().keys())

    def _is_fetchable(symbol_name: str) -> bool:
        try:
            exchange.fetch_trades(symbol_name, limit=1)
            return True
        except Exception:
            return False

    out = []
    seen = set()
    for coin in payload:
        symbol = str(coin.get("symbol", "")).upper().strip()
        if not symbol or symbol in stable or symbol in blacklist or symbol in seen:
            continue
        pair = f"{symbol}/USDT"
        alt_pair = f"1000{symbol}/USDT"
        selected_pair = None
        if pair in market_symbols and _is_fetchable(pair):
            selected_pair = pair
        elif _is_fetchable(alt_pair):
            selected_pair = alt_pair

        if selected_pair:
            out.append(selected_pair)
            seen.add(symbol)
        if len(out) >= int(desired_count):
            break

    close_fn = getattr(exchange, "close", None)
    if callable(close_fn):
        close_fn()
    return out


def _read_symbol_coverage(
    *,
    db_path: str,
    exchange: str,
    symbol: str,
    timeframe: str,
) -> tuple[datetime | None, datetime | None, int]:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT MIN(datetime), MAX(datetime), COUNT(*)
            FROM market_ohlcv
            WHERE exchange = ? AND symbol = ? AND timeframe = ?
            """,
            (
                str(exchange).strip().lower(),
                str(symbol).strip().upper(),
                str(timeframe).strip().lower(),
            ),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None, None, 0
    first_raw, last_raw, count_raw = row
    first = datetime.fromisoformat(str(first_raw).replace("Z", "+00:00")) if first_raw else None
    last = datetime.fromisoformat(str(last_raw).replace("Z", "+00:00")) if last_raw else None
    return first, last, int(count_raw or 0)


def _read_symbol_window_coverage(
    *,
    db_path: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> tuple[datetime | None, datetime | None, int]:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT MIN(datetime), MAX(datetime), COUNT(*)
            FROM market_ohlcv
            WHERE exchange = ?
              AND symbol = ?
              AND timeframe = ?
              AND datetime >= ?
              AND datetime <= ?
            """,
            (
                str(exchange).strip().lower(),
                str(symbol).strip().upper(),
                str(timeframe).strip().lower(),
                start_date.isoformat(),
                end_date.isoformat(),
            ),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None, None, 0
    first_raw, last_raw, count_raw = row
    first = datetime.fromisoformat(str(first_raw).replace("Z", "+00:00")) if first_raw else None
    last = datetime.fromisoformat(str(last_raw).replace("Z", "+00:00")) if last_raw else None
    return first, last, int(count_raw or 0)


def _filter_symbols_by_coverage(
    symbols: list[str],
    *,
    db_path: str,
    exchange: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    min_coverage_days: float,
    min_row_ratio: float,
) -> tuple[list[str], dict[str, dict[str, float | int | str | bool | None]]]:
    bars_per_day = _bars_per_day_for_timeframe(timeframe)
    target_days = max(
        0.0,
        _coverage_days_from_bounds(start_date, end_date, start_date=start_date, end_date=end_date),
    )
    target_rows = max(1.0, target_days * bars_per_day)

    accepted: list[str] = []
    details: dict[str, dict[str, float | int | str | bool | None]] = {}
    for symbol in symbols:
        first, last, count = _read_symbol_window_coverage(
            db_path=db_path,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        coverage_days = _coverage_days_from_bounds(
            first,
            last,
            start_date=start_date,
            end_date=end_date,
        )
        row_ratio = float(count) / target_rows if target_rows > 0.0 else 0.0
        row_ratio = max(0.0, min(1.0, row_ratio))
        eligible = (coverage_days >= float(min_coverage_days)) and (
            row_ratio >= float(min_row_ratio)
        )

        details[symbol] = {
            "coverage_days": float(coverage_days),
            "row_ratio": float(row_ratio),
            "rows": int(count),
            "first": first.isoformat() if isinstance(first, datetime) else None,
            "last": last.isoformat() if isinstance(last, datetime) else None,
            "eligible": bool(eligible),
        }
        if eligible:
            accepted.append(symbol)

    return accepted, details


def _required_symbol_count(strategy_cls) -> int:
    if strategy_cls in {PairTradingZScoreStrategy, LagConvergenceStrategy}:
        return 2
    return 1


def _run_backtest(
    strategy_cls,
    params: dict,
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
    base_timeframe: str,
    db_path: str,
    exchange: str,
    data_cache: dict,
) -> RunResult | None:
    key = (
        tuple(symbols),
        timeframe,
        str(base_timeframe).strip().lower(),
        start_date.isoformat() if isinstance(start_date, datetime) else str(start_date),
        end_date.isoformat() if isinstance(end_date, datetime) else str(end_date),
    )
    if key not in data_cache:
        loaded = load_data_dict_from_db(
            db_path,
            exchange=exchange,
            symbol_list=symbols,
            timeframe=str(base_timeframe).strip().lower(),
            start_date=start_date,
            end_date=end_date,
        )
        data_cache[key] = loaded
    else:
        loaded = data_cache[key]

    present_symbols = [symbol for symbol in symbols if symbol in loaded]
    required_symbols = _required_symbol_count(strategy_cls)
    if len(present_symbols) < required_symbols:
        return None

    try:
        backtest = Backtest(
            csv_dir="data",
            symbol_list=present_symbols,
            start_date=start_date,
            end_date=end_date,
            data_handler_cls=HistoricCSVDataHandler,
            execution_handler_cls=SimulatedExecutionHandler,
            portfolio_cls=Portfolio,
            strategy_cls=strategy_cls,
            strategy_params=params,
            data_dict=loaded,
            record_history=True,
            track_metrics=True,
            record_trades=True,
            strategy_timeframe=str(timeframe).strip().lower(),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            backtest.simulate_trading(output=False)
    except Exception:
        return None

    portfolio = backtest.portfolio
    portfolio.create_equity_curve_dataframe()
    stats = dict(portfolio.output_summary_stats())
    curve = portfolio.equity_curve.to_dict(as_series=False)

    metrics = {
        "return": 0.0,
        "sharpe": _safe_float(stats.get("Sharpe Ratio"), 0.0),
        "sortino": _safe_float(stats.get("Sortino Ratio"), 0.0),
        "mdd": _safe_float(str(stats.get("Max Drawdown", "0")).replace("%", ""), 0.0) / 100.0,
        "trades": int(getattr(portfolio, "trade_count", 0)),
    }
    totals = curve.get("total", [])
    if len(totals) >= 2 and _safe_float(totals[0], 0.0) > 0.0:
        metrics["return"] = (_safe_float(totals[-1], 0.0) / _safe_float(totals[0], 1.0)) - 1.0
    return RunResult(metrics=metrics, curve=curve)


def _min_overlap_days(
    symbols: list[str],
    *,
    db_path: str,
    exchange: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> float:
    data = load_data_dict_from_db(
        db_path,
        exchange=exchange,
        symbol_list=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    starts = []
    ends = []
    for symbol in symbols:
        frame = data.get(symbol)
        if not isinstance(frame, pl.DataFrame) or frame.height == 0:
            return 0.0
        starts.append(frame["datetime"].min())
        ends.append(frame["datetime"].max())
    if not starts or not ends:
        return 0.0
    overlap_start = max(starts)
    overlap_end = min(ends)
    delta = overlap_end - overlap_start
    if not hasattr(delta, "total_seconds"):
        return 0.0
    return max(0.0, float(delta.total_seconds()) / 86400.0)


def _resolve_strategy_windows(
    *,
    strategy_name: str,
    symbols: list[str],
    db_path: str,
    exchange: str,
    timeframe: str,
    global_train_start: datetime,
    global_train_end: datetime,
    global_val_start: datetime,
    global_val_end: datetime,
    global_oos_start: datetime,
) -> tuple[datetime, datetime, datetime, datetime, str] | None:
    def _as_naive_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    strategy_token = str(strategy_name).strip().lower()
    if "xau_xag" not in strategy_token:
        return global_train_start, global_train_end, global_val_start, global_val_end, "global"

    train_start_ref = _as_naive_utc(global_train_start)
    train_end_ref = _as_naive_utc(global_train_end)
    val_start_ref = _as_naive_utc(global_val_start)
    val_end_ref = _as_naive_utc(global_val_end)
    oos_anchor = _as_naive_utc(global_oos_start)

    firsts: list[datetime] = []
    for symbol in symbols:
        first, _, _ = _read_symbol_coverage(
            db_path=db_path,
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
        )
        if isinstance(first, datetime):
            firsts.append(_as_naive_utc(first))

    if not firsts:
        return None

    overlap_start = max(firsts)

    if overlap_start <= train_end_ref:
        return train_start_ref, train_end_ref, val_start_ref, val_end_ref, "global"

    available_days = max(0, int((oos_anchor - overlap_start).days))
    adaptive_val_days = max(2, min(7, available_days // 3 if available_days >= 3 else 2))

    adaptive_val_start = oos_anchor - timedelta(days=adaptive_val_days)
    adaptive_train_start = overlap_start
    adaptive_train_end = adaptive_val_start
    adaptive_val_end = oos_anchor

    if adaptive_train_end <= adaptive_train_start + timedelta(days=2):
        return None
    return (
        adaptive_train_start,
        adaptive_train_end,
        adaptive_val_start,
        adaptive_val_end,
        "adaptive_short_history",
    )


def _sample_topcap_params(rng: random.Random) -> dict:
    return {
        "lookback_bars": rng.choice([8, 12, 16, 24, 32, 48, 72, 96]),
        "rebalance_bars": rng.choice([2, 4, 6, 8, 12, 16, 24]),
        "signal_threshold": rng.choice([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]),
        "stop_loss_pct": rng.choice([0.04, 0.05, 0.06, 0.08, 0.10, 0.12]),
        "max_longs": rng.choice([2, 3, 4, 5, 6, 7, 8]),
        "max_shorts": rng.choice([2, 3, 4, 5, 6, 7, 8]),
        "min_price": rng.choice([0.01, 0.05, 0.1, 0.2]),
        "btc_regime_ma": rng.choice([0, 48, 72, 96, 120, 168]),
        "btc_symbol": "BTC/USDT",
    }


def _sample_pair_params(rng: random.Random) -> dict:
    lookback = rng.choice([24, 36, 48, 60, 72, 96, 120])
    hedge = rng.choice(
        [max(lookback, 48), max(lookback, 72), max(lookback, 96), max(lookback, 144)]
    )
    return {
        "lookback_window": lookback,
        "hedge_window": hedge,
        "entry_z": rng.choice([1.4, 1.6, 1.8, 2.0, 2.2, 2.4]),
        "exit_z": rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        "stop_z": rng.choice([2.2, 2.6, 3.0, 3.4, 3.8]),
        "min_correlation": rng.choice([-0.3, -0.1, 0.0, 0.1, 0.2, 0.3]),
        "max_hold_bars": rng.choice([24, 48, 72, 96, 144, 240]),
        "cooldown_bars": rng.choice([0, 1, 2, 4, 8]),
        "reentry_z_buffer": rng.choice([0.0, 0.1, 0.2, 0.3]),
        "min_z_turn": rng.choice([0.0, 0.02, 0.05, 0.1]),
        "stop_loss_pct": rng.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
        "min_abs_beta": rng.choice([0.0, 0.01, 0.02, 0.05]),
        "max_abs_beta": rng.choice([3.0, 4.0, 6.0, 8.0]),
        "min_volume_window": rng.choice([8, 12, 24, 36]),
        "min_volume_ratio": rng.choice([0.0, 0.5, 0.8, 1.0]),
        "use_log_price": rng.choice([True, False]),
    }


def _suggest_topcap_params_optuna(trial) -> dict:
    return {
        "lookback_bars": trial.suggest_categorical(
            "lookback_bars", [8, 12, 16, 24, 32, 48, 72, 96]
        ),
        "rebalance_bars": trial.suggest_categorical("rebalance_bars", [2, 4, 6, 8, 12, 16, 24]),
        "signal_threshold": trial.suggest_categorical(
            "signal_threshold", [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06]
        ),
        "stop_loss_pct": trial.suggest_categorical(
            "stop_loss_pct", [0.04, 0.05, 0.06, 0.08, 0.1, 0.12]
        ),
        "max_longs": trial.suggest_categorical("max_longs", [2, 3, 4, 5, 6, 7, 8]),
        "max_shorts": trial.suggest_categorical("max_shorts", [2, 3, 4, 5, 6, 7, 8]),
        "min_price": trial.suggest_categorical("min_price", [0.01, 0.05, 0.1, 0.2]),
        "btc_regime_ma": trial.suggest_categorical("btc_regime_ma", [0, 48, 72, 96, 120, 168]),
        "btc_symbol": "BTC/USDT",
    }


def _suggest_pair_params_optuna(trial) -> dict:
    lookback = trial.suggest_categorical("lookback_window", [24, 36, 48, 60, 72, 96, 120])
    hedge_window_raw = trial.suggest_categorical("hedge_window_raw", [48, 72, 96, 144])
    return {
        "lookback_window": lookback,
        "hedge_window": max(int(lookback), int(hedge_window_raw)),
        "entry_z": trial.suggest_categorical("entry_z", [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]),
        "exit_z": trial.suggest_categorical("exit_z", [0.1, 0.2, 0.3, 0.4, 0.5]),
        "stop_z": trial.suggest_categorical("stop_z", [2.2, 2.6, 3.0, 3.4, 3.8]),
        "min_correlation": trial.suggest_categorical(
            "min_correlation", [-0.3, -0.1, 0.0, 0.1, 0.2, 0.3]
        ),
        "max_hold_bars": trial.suggest_categorical("max_hold_bars", [24, 48, 72, 96, 144, 240]),
        "cooldown_bars": trial.suggest_categorical("cooldown_bars", [0, 1, 2, 4, 8]),
        "reentry_z_buffer": trial.suggest_categorical("reentry_z_buffer", [0.0, 0.1, 0.2, 0.3]),
        "min_z_turn": trial.suggest_categorical("min_z_turn", [0.0, 0.02, 0.05, 0.1]),
        "stop_loss_pct": trial.suggest_categorical("stop_loss_pct", [0.01, 0.02, 0.03, 0.04, 0.05]),
        "min_abs_beta": trial.suggest_categorical("min_abs_beta", [0.0, 0.01, 0.02, 0.05]),
        "max_abs_beta": trial.suggest_categorical("max_abs_beta", [3.0, 4.0, 6.0, 8.0]),
        "min_volume_window": trial.suggest_categorical("min_volume_window", [8, 12, 24, 36]),
        "min_volume_ratio": trial.suggest_categorical("min_volume_ratio", [0.0, 0.5, 0.8, 1.0]),
        "use_log_price": trial.suggest_categorical("use_log_price", [True, False]),
    }


def _sample_pair_xau_lag_params(rng: random.Random) -> dict:
    base = _sample_pair_params(rng)
    base.update(
        {
            "vol_lag_bars": rng.choice([0, 1, 2, 3, 4]),
            "min_vol_convergence": rng.choice([0.0, 0.2, 0.4, 0.7, 1.0]),
            "vwap_window": rng.choice([0, 8, 12, 24, 48]),
            "atr_window": rng.choice([0, 8, 12, 24]),
            "atr_max_pct": rng.choice([1.0, 0.04, 0.03, 0.02]),
        }
    )
    return base


def _suggest_pair_xau_lag_params_optuna(trial) -> dict:
    base = _suggest_pair_params_optuna(trial)
    base.update(
        {
            "vol_lag_bars": trial.suggest_categorical("vol_lag_bars", [0, 1, 2, 3, 4]),
            "min_vol_convergence": trial.suggest_categorical(
                "min_vol_convergence", [0.0, 0.2, 0.4, 0.7, 1.0]
            ),
            "vwap_window": trial.suggest_categorical("vwap_window", [0, 8, 12, 24, 48]),
            "atr_window": trial.suggest_categorical("atr_window", [0, 8, 12, 24]),
            "atr_max_pct": trial.suggest_categorical("atr_max_pct", [1.0, 0.04, 0.03, 0.02]),
        }
    )
    return base


def _sample_breakout_params(rng: random.Random) -> dict:
    return {
        "lookback_bars": rng.choice([12, 16, 24, 32, 48, 64, 96]),
        "breakout_buffer": rng.choice([0.0, 0.001, 0.002, 0.005, 0.01]),
        "atr_window": rng.choice([8, 12, 14, 21, 34]),
        "atr_stop_multiplier": rng.choice([1.2, 1.8, 2.5, 3.0, 4.0]),
        "stop_loss_pct": rng.choice([0.01, 0.015, 0.02, 0.03, 0.05]),
        "allow_short": rng.choice([True, False]),
    }


def _suggest_breakout_params_optuna(trial) -> dict:
    return {
        "lookback_bars": trial.suggest_categorical("lookback_bars", [12, 16, 24, 32, 48, 64, 96]),
        "breakout_buffer": trial.suggest_categorical(
            "breakout_buffer", [0.0, 0.001, 0.002, 0.005, 0.01]
        ),
        "atr_window": trial.suggest_categorical("atr_window", [8, 12, 14, 21, 34]),
        "atr_stop_multiplier": trial.suggest_categorical(
            "atr_stop_multiplier", [1.2, 1.8, 2.5, 3.0, 4.0]
        ),
        "stop_loss_pct": trial.suggest_categorical(
            "stop_loss_pct", [0.01, 0.015, 0.02, 0.03, 0.05]
        ),
        "allow_short": trial.suggest_categorical("allow_short", [True, False]),
    }


def _sample_mean_reversion_params(rng: random.Random) -> dict:
    return {
        "window": rng.choice([16, 24, 32, 48, 64, 96, 128]),
        "entry_z": rng.choice([1.0, 1.2, 1.6, 2.0, 2.4, 2.8]),
        "exit_z": rng.choice([0.1, 0.2, 0.3, 0.5, 0.8]),
        "stop_loss_pct": rng.choice([0.01, 0.015, 0.02, 0.03, 0.05]),
        "allow_short": rng.choice([True, False]),
    }


def _suggest_mean_reversion_params_optuna(trial) -> dict:
    return {
        "window": trial.suggest_categorical("window", [16, 24, 32, 48, 64, 96, 128]),
        "entry_z": trial.suggest_categorical("entry_z", [1.0, 1.2, 1.6, 2.0, 2.4, 2.8]),
        "exit_z": trial.suggest_categorical("exit_z", [0.1, 0.2, 0.3, 0.5, 0.8]),
        "stop_loss_pct": trial.suggest_categorical(
            "stop_loss_pct", [0.01, 0.015, 0.02, 0.03, 0.05]
        ),
        "allow_short": trial.suggest_categorical("allow_short", [True, False]),
    }


def _sample_vwap_reversion_params(rng: random.Random) -> dict:
    return {
        "window": rng.choice([16, 24, 32, 48, 64, 96, 128]),
        "entry_dev": rng.choice([0.004, 0.008, 0.012, 0.016, 0.02, 0.03]),
        "exit_dev": rng.choice([0.0, 0.002, 0.004, 0.006, 0.01]),
        "stop_loss_pct": rng.choice([0.01, 0.015, 0.02, 0.03, 0.05]),
        "allow_short": rng.choice([True, False]),
    }


def _suggest_vwap_reversion_params_optuna(trial) -> dict:
    return {
        "window": trial.suggest_categorical("window", [16, 24, 32, 48, 64, 96, 128]),
        "entry_dev": trial.suggest_categorical(
            "entry_dev", [0.004, 0.008, 0.012, 0.016, 0.02, 0.03]
        ),
        "exit_dev": trial.suggest_categorical("exit_dev", [0.0, 0.002, 0.004, 0.006, 0.01]),
        "stop_loss_pct": trial.suggest_categorical(
            "stop_loss_pct", [0.01, 0.015, 0.02, 0.03, 0.05]
        ),
        "allow_short": trial.suggest_categorical("allow_short", [True, False]),
    }


def _sample_lag_convergence_params(rng: random.Random) -> dict:
    return {
        "lag_bars": rng.choice([1, 2, 3, 4, 6, 8]),
        "entry_threshold": rng.choice([0.004, 0.008, 0.012, 0.016, 0.02, 0.03]),
        "exit_threshold": rng.choice([0.001, 0.002, 0.004, 0.006, 0.01]),
        "stop_threshold": rng.choice([0.02, 0.03, 0.05, 0.08, 0.12]),
        "max_hold_bars": rng.choice([16, 24, 48, 72, 96, 144]),
        "stop_loss_pct": rng.choice([0.01, 0.02, 0.03, 0.05]),
    }


def _suggest_lag_convergence_params_optuna(trial) -> dict:
    return {
        "lag_bars": trial.suggest_categorical("lag_bars", [1, 2, 3, 4, 6, 8]),
        "entry_threshold": trial.suggest_categorical(
            "entry_threshold", [0.004, 0.008, 0.012, 0.016, 0.02, 0.03]
        ),
        "exit_threshold": trial.suggest_categorical(
            "exit_threshold", [0.001, 0.002, 0.004, 0.006, 0.01]
        ),
        "stop_threshold": trial.suggest_categorical(
            "stop_threshold", [0.02, 0.03, 0.05, 0.08, 0.12]
        ),
        "max_hold_bars": trial.suggest_categorical("max_hold_bars", [16, 24, 48, 72, 96, 144]),
        "stop_loss_pct": trial.suggest_categorical("stop_loss_pct", [0.01, 0.02, 0.03, 0.05]),
    }


def _search_strategy(
    *,
    name: str,
    strategy_cls,
    symbols: list[str],
    sampler,
    iterations: int,
    min_trades: int,
    train_start: datetime,
    train_end: datetime,
    val_start: datetime,
    val_end: datetime,
    timeframe: str,
    base_timeframe: str,
    db_path: str,
    exchange: str,
    data_cache: dict,
    rng: random.Random,
    selection_mode: str,
    train_hurdle_return: float,
    val_hurdle_return: float,
) -> dict | None:
    best = None
    for _ in range(max(1, int(iterations))):
        params = sampler(rng)
        train = _run_backtest(
            strategy_cls,
            params,
            symbols,
            train_start,
            train_end,
            timeframe,
            base_timeframe,
            db_path,
            exchange,
            data_cache,
        )
        if train is None:
            continue
        if int(train.metrics.get("trades", 0)) < min_trades:
            continue
        if _safe_float(train.metrics.get("return", 0.0)) <= -0.05:
            continue

        val = _run_backtest(
            strategy_cls,
            params,
            symbols,
            val_start,
            val_end,
            timeframe,
            base_timeframe,
            db_path,
            exchange,
            data_cache,
        )
        if val is None:
            continue
        if int(val.metrics.get("trades", 0)) < max(1, min_trades // 3):
            continue

        train_hurdle_return_f = _safe_float(train_hurdle_return, 0.0)
        val_hurdle_return_f = _safe_float(val_hurdle_return, 0.0)
        train_excess = _safe_float(train.metrics.get("return", 0.0), 0.0) - train_hurdle_return_f
        val_excess = _safe_float(val.metrics.get("return", 0.0), 0.0) - val_hurdle_return_f

        train_score = _score_hurdle(train.metrics, train_hurdle_return_f)
        val_score = _score_hurdle(val.metrics, val_hurdle_return_f)
        robustness_penalty = abs(train_score - val_score) * 0.35
        robust_score = (val_score * 0.7) + (train_score * 0.3) - robustness_penalty
        candidate = {
            "name": name,
            "symbols": list(symbols),
            "params": params,
            "train": train,
            "val": val,
            "train_hurdle_return": train_hurdle_return_f,
            "val_hurdle_return": val_hurdle_return_f,
            "train_excess_return": float(train_excess),
            "val_excess_return": float(val_excess),
            "train_hurdle_pass": bool(train_excess >= 0.0),
            "val_hurdle_pass": bool(val_excess >= 0.0),
            "val_score": val_score,
            "train_score": train_score,
            "robust_score": robust_score,
        }
        selection_score = (
            float(candidate["val_score"])
            if str(selection_mode).strip().lower() == "val"
            else float(candidate["robust_score"])
        )
        candidate["selection_score"] = selection_score

        if best is None or float(candidate["selection_score"]) > float(best["selection_score"]):
            best = candidate
    return best


def _search_strategy_optuna(
    *,
    name: str,
    strategy_cls,
    symbols: list[str],
    suggestor,
    iterations: int,
    min_trades: int,
    train_start: datetime,
    train_end: datetime,
    val_start: datetime,
    val_end: datetime,
    timeframe: str,
    base_timeframe: str,
    db_path: str,
    exchange: str,
    data_cache: dict,
    selection_mode: str,
    seed: int,
    n_jobs: int,
    top_k: int,
    train_hurdle_return: float,
    val_hurdle_return: float,
) -> dict | None:
    if not OPTUNA_AVAILABLE or optuna is None or TrialState is None:
        return None

    trial_cache: dict[str, RunResult | None] = {}

    def _cache_key(params: dict) -> str:
        return json.dumps(params, sort_keys=True)

    def _train_result(params: dict) -> RunResult | None:
        key = _cache_key(params)
        if key not in trial_cache:
            trial_cache[key] = _run_backtest(
                strategy_cls,
                params,
                symbols,
                train_start,
                train_end,
                timeframe,
                base_timeframe,
                db_path,
                exchange,
                data_cache,
            )
        return trial_cache.get(key)

    def objective(trial):
        params = suggestor(trial)
        train = _train_result(params)
        if train is None:
            return -1_000_000.0
        if int(train.metrics.get("trades", 0)) < min_trades:
            return -1_000_000.0
        if _safe_float(train.metrics.get("return", 0.0)) <= -0.05:
            return -1_000_000.0

        train_hurdle_return_f = _safe_float(train_hurdle_return, 0.0)
        train_excess = _safe_float(train.metrics.get("return", 0.0), 0.0) - train_hurdle_return_f
        if train_excess <= -0.20:
            return -1_000_000.0
        train_score = _score_hurdle(train.metrics, train_hurdle_return_f)
        trial.set_user_attr("params", params)
        trial.set_user_attr("train_score", float(train_score))
        trial.set_user_attr("train_excess_return", float(train_excess))
        trial.set_user_attr("train_hurdle_return", float(train_hurdle_return_f))
        return float(train_score)

    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max(1, int(iterations)), n_jobs=max(1, int(n_jobs)))

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and _safe_float(trial.value, -1_000_000.0) > -999999.0
    ]
    complete_trials.sort(key=lambda trial: _safe_float(trial.value, -1_000_000.0), reverse=True)

    best = None
    for trial in complete_trials[: max(1, int(top_k))]:
        params = trial.user_attrs.get("params")
        if not isinstance(params, dict):
            params = dict(trial.params)

        train = _train_result(params)
        if train is None:
            continue

        val = _run_backtest(
            strategy_cls,
            params,
            symbols,
            val_start,
            val_end,
            timeframe,
            base_timeframe,
            db_path,
            exchange,
            data_cache,
        )
        if val is None:
            continue
        if int(val.metrics.get("trades", 0)) < max(1, min_trades // 3):
            continue

        train_hurdle_return_f = _safe_float(
            trial.user_attrs.get("train_hurdle_return", train_hurdle_return), 0.0
        )
        val_hurdle_return_f = _safe_float(val_hurdle_return, 0.0)
        train_excess = _safe_float(
            trial.user_attrs.get("train_excess_return", 0.0),
            _safe_float(train.metrics.get("return", 0.0), 0.0) - train_hurdle_return_f,
        )
        val_excess = _safe_float(val.metrics.get("return", 0.0), 0.0) - val_hurdle_return_f

        train_score = _safe_float(
            trial.user_attrs.get(
                "train_score", _score_hurdle(train.metrics, train_hurdle_return_f)
            ),
            0.0,
        )
        val_score = _score_hurdle(val.metrics, val_hurdle_return_f)
        robustness_penalty = abs(train_score - val_score) * 0.35
        robust_score = (val_score * 0.7) + (train_score * 0.3) - robustness_penalty

        candidate = {
            "name": name,
            "symbols": list(symbols),
            "params": params,
            "train": train,
            "val": val,
            "train_hurdle_return": train_hurdle_return_f,
            "val_hurdle_return": val_hurdle_return_f,
            "train_excess_return": float(train_excess),
            "val_excess_return": float(val_excess),
            "train_hurdle_pass": bool(train_excess >= 0.0),
            "val_hurdle_pass": bool(val_excess >= 0.0),
            "val_score": val_score,
            "train_score": train_score,
            "robust_score": robust_score,
        }
        selection_score = (
            float(candidate["val_score"])
            if str(selection_mode).strip().lower() == "val"
            else float(candidate["robust_score"])
        )
        candidate["selection_score"] = selection_score

        if best is None or float(candidate["selection_score"]) > float(best["selection_score"]):
            best = candidate

    return best


def _optimize_ensemble_weights(
    candidate_names: list[str],
    val_curves: dict,
    annual_periods: int,
    rng: Generator,
    iterations: int,
    val_hurdle_return: float,
    max_weight: float = 0.70,
) -> dict | None:
    if len(candidate_names) < 2:
        return None

    maps = [_returns_from_curve(val_curves[name]) for name in candidate_names]
    _, matrix = _align_returns(maps, mode="union")
    if matrix.shape[1] <= 10:
        return None

    best = None
    for _ in range(max(100, int(iterations))):
        weights = rng.dirichlet([1.0] * len(candidate_names))
        if float(np.max(weights)) > float(max_weight):
            continue
        portfolio_returns = (weights.reshape(-1, 1) * matrix).sum(axis=0).tolist()
        metrics = _metrics_from_returns(portfolio_returns, annual_periods)
        excess_return = _safe_float(metrics.get("return", 0.0), 0.0) - _safe_float(
            val_hurdle_return, 0.0
        )
        score = _score_hurdle(metrics, _safe_float(val_hurdle_return, 0.0))
        current = {
            "weights": {
                candidate_names[idx]: float(weights[idx]) for idx in range(len(candidate_names))
            },
            "metrics": metrics,
            "excess_return": float(excess_return),
            "score": score,
        }
        if best is None or float(current["score"]) > float(best["score"]):
            best = current
    return best


def _evaluate_weighted_oos(
    weights: dict[str, float],
    oos_curves: dict,
    annual_periods: int,
) -> dict:
    names = [name for name in weights if name in oos_curves]
    if not names:
        return {"return": 0.0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0, "bars": 0}
    maps = [_returns_from_curve(oos_curves[name]) for name in names]
    _, matrix = _align_returns(maps, mode="union")
    if matrix.shape[1] == 0:
        return {"return": 0.0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0, "bars": 0}

    vector = np.asarray([_safe_float(weights[name], 0.0) for name in names], dtype=np.float64)
    if float(vector.sum()) <= 0.0:
        return {"return": 0.0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0, "bars": 0}
    vector = vector / float(vector.sum())
    returns = (vector.reshape(-1, 1) * matrix).sum(axis=0).tolist()
    return _metrics_from_returns(returns, annual_periods)


def _evaluate_ensemble_eligibility(
    *,
    strategy_name: str,
    coverage_days: float,
    oos_trades: int,
    oos_bars: int,
    min_ensemble_bars: int,
    min_ensemble_oos_trades: int,
    xau_xag_min_overlap_days: float,
    xau_xag_min_oos_trades: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(oos_bars) < int(min_ensemble_bars):
        reasons.append(f"oos_bars<{int(min_ensemble_bars)}")
    if int(oos_trades) < int(min_ensemble_oos_trades):
        reasons.append(f"oos_trades<{int(min_ensemble_oos_trades)}")

    if "xau_xag" in str(strategy_name).strip().lower():
        if float(coverage_days) < float(xau_xag_min_overlap_days):
            reasons.append(f"xau_xag_coverage_days<{float(xau_xag_min_overlap_days):.1f}")
        if int(oos_trades) < int(xau_xag_min_oos_trades):
            reasons.append(f"xau_xag_oos_trades<{int(xau_xag_min_oos_trades)}")

    return (len(reasons) == 0), reasons


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe multi-strategy OOS search")
    parser.add_argument("--db-path", default="data/lumina_quant.db")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--base-timeframe", default="1s")
    parser.add_argument("--market-type", choices=["spot", "future"], default="future")
    parser.add_argument("--mode", choices=["oos", "live"], default="oos")
    parser.add_argument("--start", default="")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--oos-days", type=int, default=30)
    parser.add_argument("--min-insample-days", type=int, default=365)
    parser.add_argument(
        "--strategy-set",
        choices=["all", "crypto-only", "xau-xag-only"],
        default="all",
        help="Choose strategy sleeves: all(crypto+xauxag), crypto-only, or xau-xag-only.",
    )
    parser.add_argument("--seed", type=int, default=20260215)
    parser.add_argument(
        "--annual-return-floor",
        type=float,
        default=0.10,
        help="Minimum annualized return hurdle used in split-level target conversion.",
    )
    parser.add_argument(
        "--benchmark-symbol",
        default="BTC/USDT",
        help="Benchmark symbol for buy-and-hold hurdle comparison.",
    )
    parser.add_argument("--topcap-iters", type=int, default=320)
    parser.add_argument("--pair-iters", type=int, default=260)
    parser.add_argument("--ensemble-iters", type=int, default=5000)
    parser.add_argument(
        "--search-engine",
        choices=["optuna", "random"],
        default="optuna",
        help="Optimization backend for in-sample parameter search.",
    )
    parser.add_argument("--optuna-jobs", type=int, default=1)
    parser.add_argument("--optuna-topk", type=int, default=24)
    parser.add_argument(
        "--selection-mode",
        choices=["val", "robust"],
        default="val",
        help="Candidate ranking rule: validation-only score or robustness score.",
    )
    parser.add_argument(
        "--topcap-count",
        type=int,
        default=10,
        help="Target number of top-cap non-stable symbols for the topcap sleeve.",
    )
    parser.add_argument(
        "--topcap-candidate-count",
        type=int,
        default=120,
        help="How many market-cap ranks to scan before selecting topcap symbols.",
    )
    parser.add_argument(
        "--topcap-symbols",
        nargs="+",
        default=[],
        help="Optional explicit symbol list (e.g. BTC/USDT ETH/USDT ...). Overrides auto topcap build.",
    )
    parser.add_argument(
        "--topcap-min-coverage-days",
        type=float,
        default=360.0,
        help="Minimum per-symbol coverage days for topcap universe inclusion.",
    )
    parser.add_argument(
        "--topcap-min-row-ratio",
        type=float,
        default=0.85,
        help="Minimum observed/expected row ratio for topcap symbol inclusion.",
    )
    parser.add_argument(
        "--topcap-min-symbols",
        type=int,
        default=8,
        help="Minimum topcap symbols after coverage filtering.",
    )
    parser.add_argument(
        "--ensemble-min-bars",
        type=int,
        default=20,
        help="Minimum OOS return bars required for ensemble eligibility.",
    )
    parser.add_argument(
        "--ensemble-min-oos-trades",
        type=int,
        default=1,
        help="Minimum OOS trade count required for ensemble eligibility.",
    )
    parser.add_argument(
        "--xau-xag-ensemble-min-overlap-days",
        type=float,
        default=120.0,
        help="Minimum XAU/XAG overlap days required before allowing ensemble inclusion.",
    )
    parser.add_argument(
        "--xau-xag-ensemble-min-oos-trades",
        type=int,
        default=2,
        help="Minimum XAU/XAG OOS trades required for ensemble inclusion.",
    )
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    annual_periods = _annual_periods_for_timeframe(args.timeframe)

    explicit_start = None
    if str(args.start).strip():
        explicit_start = datetime.strptime(str(args.start), "%Y-%m-%d")

    strategy_set = str(args.strategy_set).strip().lower()
    probe_symbols = []
    if strategy_set != "xau-xag-only":
        probe_symbols.extend(["BTC/USDT", "ETH/USDT", "BNB/USDT"])
    if strategy_set != "crypto-only":
        probe_symbols.extend(["XAU/USDT", "XAG/USDT"])
    if not probe_symbols:
        raise RuntimeError("No probe symbols resolved for split construction.")

    probe_start = explicit_start or datetime(2020, 1, 1)
    probe_data = load_data_dict_from_db(
        args.db_path,
        exchange=args.exchange,
        symbol_list=probe_symbols,
        timeframe=args.base_timeframe,
        start_date=probe_start,
        end_date=None,
    )
    if not probe_data:
        raise RuntimeError("No probe data found to build split windows.")

    full_end = None
    for frame in probe_data.values():
        raw_end = frame["datetime"].max()
        cur_end = (
            raw_end
            if isinstance(raw_end, datetime)
            else datetime.fromisoformat(str(raw_end).replace("Z", "+00:00"))
        )
        if full_end is None or cur_end > full_end:
            full_end = cur_end
    if full_end is None:
        raise RuntimeError("Failed to determine latest timestamp from probe data.")
    if isinstance(full_end, datetime) and full_end.tzinfo is not None:
        full_end = full_end.astimezone(UTC).replace(tzinfo=None)

    train_days = max(30, int(args.train_days))
    val_days = max(3, int(args.val_days))
    oos_days = max(0, int(args.oos_days))
    mode = str(args.mode).strip().lower()
    annual_return_floor = max(0.0, float(args.annual_return_floor))
    benchmark_symbol = str(args.benchmark_symbol).strip().upper()
    benchmark_cache = {}

    if mode == "live":
        val_end = full_end
        val_start = val_end - timedelta(days=val_days)
        train_end = val_start
        oos_start = full_end
    else:
        oos_start = full_end - timedelta(days=oos_days)
        val_end = oos_start
        val_start = val_end - timedelta(days=val_days)
        train_end = val_start

    start_date = train_end - timedelta(days=train_days)
    if explicit_start is not None and start_date < explicit_start:
        start_date = explicit_start

    in_sample_anchor = val_end if mode == "live" else oos_start
    in_sample_days = (in_sample_anchor - start_date).days
    if in_sample_days < int(max(1, int(args.min_insample_days))):
        raise RuntimeError(
            f"In-sample span is {in_sample_days} days, below required "
            f"{int(args.min_insample_days)} days."
        )

    if train_end <= start_date + timedelta(days=5):
        raise RuntimeError("Not enough history to create train/validation/OOS windows.")

    print(f"=== Leakage-Safe Split (mode={mode}) ===")
    print(f"Train: {start_date} ~ {train_end}")
    print(f"Val  : {val_start} ~ {val_end}")
    if mode == "live":
        print("OOS  : skipped (live mode)")
    else:
        print(f"OOS  : {oos_start} ~ {full_end}")

    global_train_hurdle = _split_hurdle(
        db_path=args.db_path,
        exchange=args.exchange,
        timeframe=args.base_timeframe,
        start_date=start_date,
        end_date=train_end,
        annual_floor=annual_return_floor,
        benchmark_symbol=benchmark_symbol,
        benchmark_cache=benchmark_cache,
    )
    global_val_hurdle = _split_hurdle(
        db_path=args.db_path,
        exchange=args.exchange,
        timeframe=args.base_timeframe,
        start_date=val_start,
        end_date=val_end,
        annual_floor=annual_return_floor,
        benchmark_symbol=benchmark_symbol,
        benchmark_cache=benchmark_cache,
    )
    global_eval_start = oos_start if mode != "live" else val_start
    global_eval_end = full_end if mode != "live" else val_end
    global_eval_hurdle = _split_hurdle(
        db_path=args.db_path,
        exchange=args.exchange,
        timeframe=args.base_timeframe,
        start_date=global_eval_start,
        end_date=global_eval_end,
        annual_floor=annual_return_floor,
        benchmark_symbol=benchmark_symbol,
        benchmark_cache=benchmark_cache,
    )

    raw_topcap_symbols: list[str] = []
    topcap_symbols: list[str] = []
    topcap_coverage: dict[str, dict[str, float | int | str | bool | None]] = {}
    dropped_topcap: list[str] = []

    strategies = []
    if strategy_set != "xau-xag-only":
        explicit_topcap = []
        seen_topcap = set()
        for raw in list(args.topcap_symbols):
            token = str(raw).strip().upper().replace("_", "/").replace("-", "/")
            if not token:
                continue
            if "/" not in token and token.endswith("USDT") and len(token) > 4:
                token = f"{token[:-4]}/USDT"
            if token in seen_topcap:
                continue
            seen_topcap.add(token)
            explicit_topcap.append(token)

        if explicit_topcap:
            raw_topcap_symbols = explicit_topcap
        else:
            raw_topcap_symbols = _build_topcap_universe(
                desired_count=max(2, int(args.topcap_count)),
                candidate_count=max(20, int(args.topcap_candidate_count)),
                market_type=str(args.market_type),
            )
        topcap_symbols, topcap_coverage = _filter_symbols_by_coverage(
            raw_topcap_symbols,
            db_path=args.db_path,
            exchange=args.exchange,
            timeframe=args.base_timeframe,
            start_date=start_date,
            end_date=full_end,
            min_coverage_days=float(args.topcap_min_coverage_days),
            min_row_ratio=float(args.topcap_min_row_ratio),
        )
        dropped_topcap = [symbol for symbol in raw_topcap_symbols if symbol not in topcap_symbols]
        if len(topcap_symbols) < int(max(2, int(args.topcap_min_symbols))):
            raise RuntimeError(
                "Topcap coverage gate removed too many symbols: "
                f"kept={len(topcap_symbols)} required={int(args.topcap_min_symbols)}"
            )

        print(
            f"Topcap universe raw={len(raw_topcap_symbols)} kept={len(topcap_symbols)} "
            f"(target={int(args.topcap_count)}, min_days={float(args.topcap_min_coverage_days):.1f}, "
            f"min_row_ratio={float(args.topcap_min_row_ratio):.2f})"
        )
        if dropped_topcap:
            print(f"Dropped topcap symbols ({len(dropped_topcap)}): {dropped_topcap}")
        print(f"Topcap symbols: {topcap_symbols}")

        strategies.extend(
            [
                {
                    "name": "topcap_tsmom",
                    "strategy_cls": TopCapTimeSeriesMomentumStrategy,
                    "symbols": topcap_symbols,
                    "sampler": _sample_topcap_params,
                    "suggestor": _suggest_topcap_params_optuna,
                    "iterations": int(args.topcap_iters),
                    "min_trades": 6,
                },
                {
                    "name": "pair_btc_bnb",
                    "strategy_cls": PairTradingZScoreStrategy,
                    "symbols": ["BTC/USDT", "BNB/USDT"],
                    "sampler": _sample_pair_params,
                    "suggestor": _suggest_pair_params_optuna,
                    "iterations": int(args.pair_iters),
                    "min_trades": 2,
                },
                {
                    "name": "rolling_breakout_topcap",
                    "strategy_cls": RollingBreakoutStrategy,
                    "symbols": topcap_symbols,
                    "sampler": _sample_breakout_params,
                    "suggestor": _suggest_breakout_params_optuna,
                    "iterations": int(args.topcap_iters),
                    "min_trades": 6,
                },
                {
                    "name": "mean_reversion_std_topcap",
                    "strategy_cls": MeanReversionStdStrategy,
                    "symbols": topcap_symbols,
                    "sampler": _sample_mean_reversion_params,
                    "suggestor": _suggest_mean_reversion_params_optuna,
                    "iterations": int(args.topcap_iters),
                    "min_trades": 6,
                },
                {
                    "name": "vwap_reversion_topcap",
                    "strategy_cls": VwapReversionStrategy,
                    "symbols": topcap_symbols,
                    "sampler": _sample_vwap_reversion_params,
                    "suggestor": _suggest_vwap_reversion_params_optuna,
                    "iterations": int(args.topcap_iters),
                    "min_trades": 6,
                },
                {
                    "name": "lag_convergence_btc_eth",
                    "strategy_cls": LagConvergenceStrategy,
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "sampler": _sample_lag_convergence_params,
                    "suggestor": _suggest_lag_convergence_params_optuna,
                    "iterations": int(args.pair_iters),
                    "min_trades": 2,
                },
            ]
        )

    if strategy_set != "crypto-only":
        strategies.extend(
            [
                {
                    "name": "pair_xau_xag",
                    "strategy_cls": PairTradingZScoreStrategy,
                    "symbols": ["XAU/USDT", "XAG/USDT"],
                    "sampler": _sample_pair_params,
                    "suggestor": _suggest_pair_params_optuna,
                    "iterations": int(args.pair_iters),
                    "min_trades": 2,
                },
                {
                    "name": "pair_xau_xag_lag",
                    "strategy_cls": PairTradingZScoreStrategy,
                    "symbols": ["XAU/USDT", "XAG/USDT"],
                    "sampler": _sample_pair_xau_lag_params,
                    "suggestor": _suggest_pair_xau_lag_params_optuna,
                    "iterations": int(args.pair_iters),
                    "min_trades": 2,
                },
                {
                    "name": "lag_convergence_xau_xag",
                    "strategy_cls": LagConvergenceStrategy,
                    "symbols": ["XAU/USDT", "XAG/USDT"],
                    "sampler": _sample_lag_convergence_params,
                    "suggestor": _suggest_lag_convergence_params_optuna,
                    "iterations": int(args.pair_iters),
                    "min_trades": 2,
                },
            ]
        )

    data_cache = {}
    winners = []
    for spec in strategies:
        print(f"\n[SEARCH] {spec['name']} iterations={spec['iterations']}")
        coverage_days = _min_overlap_days(
            spec["symbols"],
            db_path=args.db_path,
            exchange=args.exchange,
            timeframe=args.base_timeframe,
            start_date=start_date,
            end_date=full_end,
        )
        print(f"  coverage_days={coverage_days:.1f}")

        resolved_windows = _resolve_strategy_windows(
            strategy_name=str(spec["name"]),
            symbols=list(spec["symbols"]),
            db_path=args.db_path,
            exchange=args.exchange,
            timeframe=args.base_timeframe,
            global_train_start=start_date,
            global_train_end=train_end,
            global_val_start=val_start,
            global_val_end=val_end,
            global_oos_start=oos_start,
        )
        if resolved_windows is None:
            print("  -> no valid train/val window for this sleeve")
            continue
        (
            spec_train_start,
            spec_train_end,
            spec_val_start,
            spec_val_end,
            spec_window_mode,
        ) = resolved_windows
        if spec_window_mode != "global":
            print(
                f"  window_mode={spec_window_mode} "
                f"train={spec_train_start}~{spec_train_end} val={spec_val_start}~{spec_val_end}"
            )

        train_hurdle = _split_hurdle(
            db_path=args.db_path,
            exchange=args.exchange,
            timeframe=args.base_timeframe,
            start_date=spec_train_start,
            end_date=spec_train_end,
            annual_floor=annual_return_floor,
            benchmark_symbol=benchmark_symbol,
            benchmark_cache=benchmark_cache,
        )
        val_hurdle = _split_hurdle(
            db_path=args.db_path,
            exchange=args.exchange,
            timeframe=args.base_timeframe,
            start_date=spec_val_start,
            end_date=spec_val_end,
            annual_floor=annual_return_floor,
            benchmark_symbol=benchmark_symbol,
            benchmark_cache=benchmark_cache,
        )

        winner = None
        if str(args.search_engine).strip().lower() == "optuna" and OPTUNA_AVAILABLE:
            winner = _search_strategy_optuna(
                name=spec["name"],
                strategy_cls=spec["strategy_cls"],
                symbols=spec["symbols"],
                suggestor=spec["suggestor"],
                iterations=spec["iterations"],
                min_trades=spec["min_trades"],
                train_start=spec_train_start,
                train_end=spec_train_end,
                val_start=spec_val_start,
                val_end=spec_val_end,
                timeframe=args.timeframe,
                base_timeframe=args.base_timeframe,
                db_path=args.db_path,
                exchange=args.exchange,
                data_cache=data_cache,
                selection_mode=str(args.selection_mode),
                seed=int(args.seed),
                n_jobs=int(args.optuna_jobs),
                top_k=int(args.optuna_topk),
                train_hurdle_return=float(train_hurdle["hurdle_return"]),
                val_hurdle_return=float(val_hurdle["hurdle_return"]),
            )

        if winner is None:
            winner = _search_strategy(
                name=spec["name"],
                strategy_cls=spec["strategy_cls"],
                symbols=spec["symbols"],
                sampler=spec["sampler"],
                iterations=spec["iterations"],
                min_trades=spec["min_trades"],
                train_start=spec_train_start,
                train_end=spec_train_end,
                val_start=spec_val_start,
                val_end=spec_val_end,
                timeframe=args.timeframe,
                base_timeframe=args.base_timeframe,
                db_path=args.db_path,
                exchange=args.exchange,
                data_cache=data_cache,
                rng=rng,
                selection_mode=str(args.selection_mode),
                train_hurdle_return=float(train_hurdle["hurdle_return"]),
                val_hurdle_return=float(val_hurdle["hurdle_return"]),
            )
        if winner is None:
            print("  -> no valid candidate")
            continue

        eval_start = oos_start if mode != "live" else spec_val_start
        eval_end = full_end if mode != "live" else spec_val_end
        eval_label = "oos" if mode != "live" else "live_val"
        eval_hurdle = _split_hurdle(
            db_path=args.db_path,
            exchange=args.exchange,
            timeframe=args.base_timeframe,
            start_date=eval_start,
            end_date=eval_end,
            annual_floor=annual_return_floor,
            benchmark_symbol=benchmark_symbol,
            benchmark_cache=benchmark_cache,
        )

        oos = _run_backtest(
            spec["strategy_cls"],
            winner["params"],
            spec["symbols"],
            eval_start,
            eval_end,
            args.timeframe,
            args.base_timeframe,
            args.db_path,
            args.exchange,
            data_cache,
        )
        if oos is None:
            print("  -> final evaluation failed")
            continue

        winner["oos"] = oos
        winner["coverage_days"] = float(coverage_days)
        winner["window_mode"] = str(spec_window_mode)
        winner["train_start"] = spec_train_start.isoformat()
        winner["train_end"] = spec_train_end.isoformat()
        winner["val_start"] = spec_val_start.isoformat()
        winner["val_end"] = spec_val_end.isoformat()
        winner["train_hurdle"] = train_hurdle
        winner["val_hurdle"] = val_hurdle
        winner["oos_hurdle"] = eval_hurdle
        winner["train_hurdle_fields"] = _hurdle_fields(winner["train"].metrics, train_hurdle)
        winner["val_hurdle_fields"] = _hurdle_fields(winner["val"].metrics, val_hurdle)
        winner["oos_hurdle_fields"] = _hurdle_fields(winner["oos"].metrics, eval_hurdle)
        oos_bars = len(_returns_from_curve(oos.curve))
        winner["oos_bars"] = int(oos_bars)
        ensemble_eligible, ensemble_reasons = _evaluate_ensemble_eligibility(
            strategy_name=str(spec["name"]),
            coverage_days=float(coverage_days),
            oos_trades=int(oos.metrics.get("trades", 0)),
            oos_bars=int(oos_bars),
            min_ensemble_bars=int(args.ensemble_min_bars),
            min_ensemble_oos_trades=int(args.ensemble_min_oos_trades),
            xau_xag_min_overlap_days=float(args.xau_xag_ensemble_min_overlap_days),
            xau_xag_min_oos_trades=int(args.xau_xag_ensemble_min_oos_trades),
        )
        if not bool(winner["oos_hurdle_fields"].get("pass", False)):
            ensemble_reasons.append("oos_hurdle_not_met")
            ensemble_eligible = False
        winner["ensemble_eligible"] = bool(ensemble_eligible)
        winner["ensemble_skip_reason"] = "; ".join(ensemble_reasons) if ensemble_reasons else None
        winners.append(winner)

        print(f"  best params: {winner['params']}")
        print(f"  val metrics: {winner['val'].metrics}")
        print(f"  {eval_label} metrics: {winner['oos'].metrics}")
        print(
            f"  val_excess={winner['val_hurdle_fields']['excess_return']:.6f} "
            f"{eval_label}_excess={winner['oos_hurdle_fields']['excess_return']:.6f}"
        )
        if not bool(ensemble_eligible):
            print(f"  ensemble_eligible=False reason={winner['ensemble_skip_reason']}")

    if not winners:
        raise RuntimeError("No strategy produced a valid candidate.")

    ensemble_pool = [winner for winner in winners if bool(winner.get("ensemble_eligible", True))]
    ensemble_ineligible = [
        {
            "name": winner["name"],
            "reason": winner.get("ensemble_skip_reason"),
            "coverage_days": winner.get("coverage_days"),
            "oos_trades": winner["oos"].metrics.get("trades"),
            "oos_bars": winner.get("oos_bars", 0),
        }
        for winner in winners
        if not bool(winner.get("ensemble_eligible", True))
    ]
    val_curves = {winner["name"]: winner["val"].curve for winner in ensemble_pool}
    oos_curves = {winner["name"]: winner["oos"].curve for winner in ensemble_pool}
    names = [winner["name"] for winner in ensemble_pool]

    ensemble_best = None
    if len(names) >= 2:
        ensemble_best = _optimize_ensemble_weights(
            names,
            val_curves,
            annual_periods=annual_periods,
            rng=np.random.default_rng(int(args.seed)),
            iterations=int(args.ensemble_iters),
            val_hurdle_return=float(global_val_hurdle["hurdle_return"]),
        )
    else:
        print("\n[ENSEMBLE] skipped: fewer than 2 eligible sleeves")

    ensemble_oos_metrics = None
    ensemble_val_hurdle_fields = None
    ensemble_oos_hurdle_fields = None
    if ensemble_best is not None:
        ensemble_val_hurdle_fields = _hurdle_fields(
            ensemble_best.get("metrics", {}), global_val_hurdle
        )
        ensemble_oos_metrics = _evaluate_weighted_oos(
            ensemble_best["weights"],
            oos_curves,
            annual_periods=annual_periods,
        )
        ensemble_oos_hurdle_fields = _hurdle_fields(ensemble_oos_metrics, global_eval_hurdle)

    report = {
        "split": {
            "mode": mode,
            "train_start": start_date.isoformat(),
            "train_end": train_end.isoformat(),
            "val_start": val_start.isoformat(),
            "val_end": val_end.isoformat(),
            "oos_start": oos_start.isoformat() if mode != "live" else None,
            "oos_end": full_end.isoformat() if mode != "live" else None,
            "strategy_timeframe": args.timeframe,
            "base_timeframe": args.base_timeframe,
        },
        "candidates": [
            {
                "name": winner["name"],
                "symbols": winner["symbols"],
                "params": winner["params"],
                "train": winner["train"].metrics,
                "val": winner["val"].metrics,
                "oos": winner["oos"].metrics,
                "train_score": winner.get("train_score", 0.0),
                "val_score": winner["val_score"],
                "robust_score": winner.get("robust_score", winner["val_score"]),
                "selection_score": winner.get("selection_score", winner["val_score"]),
                "hurdle": {
                    "train": winner.get("train_hurdle"),
                    "val": winner.get("val_hurdle"),
                    "oos": winner.get("oos_hurdle"),
                },
                "hurdle_fields": {
                    "train": winner.get("train_hurdle_fields"),
                    "val": winner.get("val_hurdle_fields"),
                    "oos": winner.get("oos_hurdle_fields"),
                },
                "coverage_days": winner.get("coverage_days"),
                "window_mode": winner.get("window_mode", "global"),
                "window": {
                    "train_start": winner.get("train_start"),
                    "train_end": winner.get("train_end"),
                    "val_start": winner.get("val_start"),
                    "val_end": winner.get("val_end"),
                },
                "oos_bars": winner.get("oos_bars", 0),
                "ensemble_eligible": bool(winner.get("ensemble_eligible", True)),
                "ensemble_skip_reason": winner.get("ensemble_skip_reason"),
            }
            for winner in winners
        ],
        "ensemble": {
            "best_val": ensemble_best,
            "oos_metrics": ensemble_oos_metrics,
            "hurdle_fields": {
                "val": ensemble_val_hurdle_fields,
                "oos": ensemble_oos_hurdle_fields,
            },
            "eligible_names": names,
            "ineligible": ensemble_ineligible,
            "eligibility_policy": {
                "min_bars": int(args.ensemble_min_bars),
                "min_oos_trades": int(args.ensemble_min_oos_trades),
                "xau_xag_min_overlap_days": float(args.xau_xag_ensemble_min_overlap_days),
                "xau_xag_min_oos_trades": int(args.xau_xag_ensemble_min_oos_trades),
            },
        },
        "benchmark": {
            "symbol": benchmark_symbol,
            "annual_return_floor": float(annual_return_floor),
            "global": {
                "train": global_train_hurdle,
                "val": global_val_hurdle,
                "eval": global_eval_hurdle,
            },
        },
        "topcap_coverage_gate": {
            "requested_topcap_count": int(args.topcap_count),
            "requested_topcap_symbols": list(args.topcap_symbols),
            "raw_symbols": raw_topcap_symbols,
            "kept_symbols": topcap_symbols,
            "dropped_symbols": dropped_topcap,
            "min_coverage_days": float(args.topcap_min_coverage_days),
            "min_row_ratio": float(args.topcap_min_row_ratio),
            "details": topcap_coverage,
        },
    }

    metric_prefix = "oos" if mode != "live" else "live_val"
    print("\n=== Final Candidate Metrics ===")
    for row in report["candidates"]:
        print(
            {
                "name": row["name"],
                f"{metric_prefix}_return": row["oos"].get("return"),
                f"{metric_prefix}_sortino": row["oos"].get("sortino"),
                f"{metric_prefix}_sharpe": row["oos"].get("sharpe"),
                f"{metric_prefix}_mdd": row["oos"].get("mdd"),
                f"{metric_prefix}_trades": row["oos"].get("trades"),
            }
        )

    if ensemble_best is not None and ensemble_oos_metrics is not None:
        print("\n=== Ensemble (Weights tuned on Validation only) ===")
        print({"weights": ensemble_best["weights"], metric_prefix: ensemble_oos_metrics})
    elif ensemble_ineligible:
        print("\n=== Ensemble Eligibility Filtered Out Sleeves ===")
        for row in ensemble_ineligible:
            print(row)

    print(
        f"\nBenchmark {metric_prefix} return: "
        f"{float(global_eval_hurdle['benchmark_return']):.6f} "
        f"floor={float(global_eval_hurdle['floor_return']):.6f} "
        f"hurdle={float(global_eval_hurdle['hurdle_return']):.6f}"
    )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"oos_guarded_multistrategy_{mode}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
