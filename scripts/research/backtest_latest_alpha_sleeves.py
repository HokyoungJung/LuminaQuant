"""Backtest the latest alpha-sleeve strategy additions with fail-closed data audits.

The normal ``lq backtest`` CLI is optimized for the current 1s/raw-first execution
contract.  The newly-added alpha sleeves also need quick 1m research validation
over the existing local legacy parquet partitions.  This script keeps the actual
trading simulation on the repository's event-driven ``Backtest``/``Portfolio``/
``ExecutionModel`` stack, but makes the data contract explicit:

* load only local parquet OHLCV via ``MarketDataRepository``;
* never fill missing bars or synthesize prices;
* fail before simulation if timestamp gaps, duplicate bars, bad OHLC, null/NaN,
  or non-positive prices are detected;
* optionally cross-check symbols against official Binance exchangeInfo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import polars as pl

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.market_data import MarketDataRepository, normalize_symbol
from lumina_quant.strategies.registry import (
    get_strategy_names,
    get_strategy_tier,
    resolve_strategy_class,
)


NEW_ALPHA_SLEEVE_STRATEGIES: tuple[str, ...] = (
    # external_alpha_sleeves.py
    "LiquidityShockReversionStrategy",
    "VolManagedMomentumCrashGateStrategy",
    "FundingDislocationTrendCarryStrategy",
    "OpeningRangeContinuationStrategy",
    # robust_alpha_sleeves.py
    "DonchianAtrTrendStrategy",
    "VolatilitySqueezeBreakoutStrategy",
    "TakerFlowImbalanceContinuationStrategy",
    "PairsSpreadMeanReversionStrategy",
    "CrossSectionalShortTermReversalStrategy",
    # adaptive_crypto_alpha_sleeves.py
    "BenchmarkLeadLagContinuationStrategy",
    "ResidualMomentumRotationStrategy",
    "LowVolatilityMomentumStrategy",
    "NearHighMomentumStrategy",
    "FalseBreakoutReversalStrategy",
    # cross_asset_tradfi_alpha_sleeves.py
    "GoldSilverRatioMeanReversionStrategy",
    "GoldSilverRatioTrendStrategy",
    "EquityMetalRiskRegimeRotationStrategy",
    "EquityBenchmarkResidualReversalStrategy",
    "MetalEquityDivergenceReversalStrategy",
)

CORE_CRYPTO_METAL_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "TON/USDT",
    "AVAX/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
)

# Latest local feature-complete sleeve window has the required funding/flow
# columns for BTC/ETH/SOL.  BNB is intentionally excluded from feature-dependent
# sleeves because the local feature partitions have no taker quote volume for it.
FEATURE_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
PAIR_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "XAU/USDT", "XAG/USDT")
CRYPTO_PAIR_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
CRYPTO_FEATURE_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
METAL_PAIR_SYMBOLS: tuple[str, ...] = ("XAU/USDT", "XAG/USDT")
CROSS_ASSET_SYMBOLS: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
    "SPY/USDT",
    "QQQ/USDT",
    "AAPL/USDT",
    "MSFT/USDT",
    "NVDA/USDT",
    "TSLA/USDT",
)

STRATEGY_SYMBOLS: dict[str, tuple[str, ...]] = {
    "BitcoinBuyHoldStrategy": ("BTC/USDT",),
    "PairTradingZScoreStrategy": CRYPTO_PAIR_SYMBOLS,
    "PairSpreadZScoreStrategy": CRYPTO_PAIR_SYMBOLS,
    "SessionFilteredPairCarryStrategy": CRYPTO_PAIR_SYMBOLS,
    "TimeframePairZScoreReversionStrategy": CRYPTO_PAIR_SYMBOLS,
    "LeadLagSpilloverStrategy": CRYPTO_PAIR_SYMBOLS,
    "CrossCryptoSlowDiffusionStrategy": CRYPTO_FEATURE_SYMBOLS,
    "CryptoFxAlphaZooStateStrategy": CRYPTO_FEATURE_SYMBOLS,
    "DerivativesFlowSqueezeStrategy": CRYPTO_FEATURE_SYMBOLS,
    "HourlyShockReversionStrategy": CRYPTO_FEATURE_SYMBOLS,
    "PerpCrowdingCarryStrategy": CRYPTO_FEATURE_SYMBOLS,
    "TakerFlowExhaustionReversalStrategy": CRYPTO_FEATURE_SYMBOLS,
    "FundingDislocationTrendCarryStrategy": FEATURE_SYMBOLS,
    "TakerFlowImbalanceContinuationStrategy": FEATURE_SYMBOLS,
    "PairsSpreadMeanReversionStrategy": PAIR_SYMBOLS,
    "GoldSilverRatioMeanReversionStrategy": METAL_PAIR_SYMBOLS,
    "GoldSilverRatioTrendStrategy": METAL_PAIR_SYMBOLS,
    "EquityMetalRiskRegimeRotationStrategy": CROSS_ASSET_SYMBOLS,
    "EquityBenchmarkResidualReversalStrategy": CROSS_ASSET_SYMBOLS,
    "MetalEquityDivergenceReversalStrategy": CROSS_ASSET_SYMBOLS,
    "ResidualMomentumRotationStrategy": CROSS_ASSET_SYMBOLS,
    "LowVolatilityMomentumStrategy": CROSS_ASSET_SYMBOLS,
    "NearHighMomentumStrategy": CROSS_ASSET_SYMBOLS,
    "CrossSectionalShortTermReversalStrategy": CROSS_ASSET_SYMBOLS,
}

BINANCE_EXCHANGE_INFO_URLS: tuple[str, ...] = (
    "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "https://api.binance.com/api/v3/exchangeInfo",
)

TIMEFRAME_MS = 60_000
DEFAULT_START = "2026-05-01T00:00:00Z"
DEFAULT_END = "2026-05-03T00:00:00Z"


@dataclass(frozen=True, slots=True)
class StrategyRunSpec:
    strategy: str
    symbols: tuple[str, ...]


def _parse_datetime(value: str) -> datetime:
    token = str(value or "").strip()
    if not token:
        raise ValueError("empty datetime")
    parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return normalized.isoformat().replace("+00:00", "Z")


def _utc_epoch_ms(value: datetime) -> int:
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return int(normalized.timestamp() * 1000)


def _half_open_window_bar_count(start: datetime, end: datetime) -> int:
    start_ms = _utc_epoch_ms(start)
    end_ms = _utc_epoch_ms(end)
    if end_ms <= start_ms:
        raise ValueError("--end must be after --start")
    if start_ms % TIMEFRAME_MS or end_ms % TIMEFRAME_MS:
        raise ValueError("--start and --end must be aligned to exact UTC minutes")
    duration_ms = end_ms - start_ms
    if duration_ms % TIMEFRAME_MS:
        raise ValueError("backtest window must contain a whole number of 1m bars")
    return duration_ms // TIMEFRAME_MS


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _compact_symbol(symbol: str) -> str:
    return normalize_symbol(symbol).replace("/", "")


def _strategy_names_for_scope(scope: str) -> tuple[str, ...]:
    token = str(scope or "latest").strip().lower().replace("_", "-")
    if token == "latest":
        return NEW_ALPHA_SLEEVE_STRATEGIES
    if token in {"live", "all-live"}:
        return tuple(get_strategy_names(include_research_only=False))
    if token == "all":
        return tuple(get_strategy_names(include_research_only=True))
    raise ValueError(f"unsupported scope: {scope}")


def _strategy_specs(
    selected: list[str] | None = None,
    *,
    scope: str = "latest",
) -> list[StrategyRunSpec]:
    available = set(get_strategy_names(include_research_only=True))
    requested = tuple(selected or _strategy_names_for_scope(scope))
    specs: list[StrategyRunSpec] = []
    for strategy in requested:
        if strategy not in available:
            raise ValueError(f"Unknown strategy: {strategy}")
        symbols = STRATEGY_SYMBOLS.get(strategy, CORE_CRYPTO_METAL_SYMBOLS)
        specs.append(StrategyRunSpec(strategy=strategy, symbols=tuple(symbols)))
    return specs


def _zero_trade_reason(row: dict[str, Any]) -> str:
    if str(row.get("status")) != "pass":
        return ""
    if int(row.get("trade_count") or 0) > 0:
        return ""
    market_events = int(row.get("market_events") or 0)
    signals = int(row.get("signals") or 0)
    orders = int(row.get("orders") or 0)
    fills = int(row.get("fills") or 0)
    feature_status = str(row.get("feature_audit_status") or "")
    if market_events <= 0:
        return "no_market_events_loaded"
    if feature_status == "warn":
        return "no_trade_with_partial_feature_coverage"
    if signals <= 0:
        return "no_signal_generated_under_default_params_window"
    if orders <= 0:
        return "signals_generated_but_no_orders"
    if fills <= 0:
        return "orders_generated_but_no_fills"
    return "flat_after_rounding_or_zero_net_position"


def _required_features_for_strategy(strategy_cls: type) -> tuple[str, ...]:
    """Return declared feature dependencies without fabricating unknown needs.

    Most strategies expose ``required_features`` as a class-level tuple.  A few
    older/live strategies expose it as an instance property derived from default
    parameters.  For the scoreboard audit we instantiate only that property shape
    with ``bars=None, events=None``; if the property cannot be resolved safely, we
    return an empty tuple and let the actual backtest engine exercise the strategy.
    """
    raw_features = getattr(strategy_cls, "required_features", ())
    if isinstance(raw_features, property):
        try:
            raw_features = raw_features.__get__(strategy_cls(None, None), strategy_cls)
        except Exception:
            raw_features = ()
    if raw_features is None:
        return ()
    if isinstance(raw_features, str):
        return (raw_features,) if raw_features else ()
    if not isinstance(raw_features, Iterable):
        return ()
    try:
        return tuple(str(item) for item in raw_features if str(item))
    except TypeError:
        return ()


def _is_unavailable_runtime_error(message: str) -> bool:
    lowered = str(message or "").lower()
    return (
        "required_inputs are unavailable" in lowered
        or "required_features are unavailable" in lowered
        or "declared unsupported required_features" in lowered
    )


def _official_binance_symbols(timeout_seconds: int = 20) -> dict[str, Any]:
    symbols: set[str] = set()
    errors: dict[str, str] = {}
    counts: dict[str, int] = {}
    for url in BINANCE_EXCHANGE_INFO_URLS:
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                payload = json.load(response)
            source_symbols = {
                str(item.get("symbol") or "")
                for item in list(payload.get("symbols") or [])
                if str(item.get("symbol") or "")
            }
            symbols.update(source_symbols)
            counts[url] = len(source_symbols)
        except Exception as exc:
            errors[url] = repr(exc)
            counts[url] = 0
    return {
        "source_urls": list(BINANCE_EXCHANGE_INFO_URLS),
        "symbols": sorted(symbols),
        "errors": errors,
        "source_counts": counts,
    }


def _audit_exchange_symbols(
    symbols: list[str],
    *,
    enabled: bool,
    fail_on_error: bool,
) -> dict[str, Any]:
    compact = sorted({_compact_symbol(symbol) for symbol in symbols})
    if not enabled:
        return {
            "enabled": False,
            "status": "skipped",
            "checked_symbols": compact,
            "unmatched_symbols": [],
        }
    official = _official_binance_symbols()
    official_set = set(official["symbols"])
    unmatched = [symbol for symbol in compact if symbol not in official_set]
    status = "pass" if not unmatched and not official["errors"] else "warn"
    if unmatched:
        status = "fail"
    if fail_on_error and official["errors"]:
        status = "fail"
    return {
        "enabled": True,
        "status": status,
        "checked_symbols": compact,
        "matched_count": len([symbol for symbol in compact if symbol in official_set]),
        "unmatched_symbols": unmatched,
        "source_urls": official["source_urls"],
        "source_counts": official["source_counts"],
        "errors": official["errors"],
    }


def _frame_time_bounds(frame: pl.DataFrame) -> tuple[datetime | None, datetime | None]:
    if frame.is_empty() or "datetime" not in frame.columns:
        return None, None
    return cast(datetime | None, frame["datetime"].min()), cast(
        datetime | None, frame["datetime"].max()
    )


def audit_ohlcv_frame(
    symbol: str,
    frame: pl.DataFrame,
    *,
    start: datetime,
    end: datetime,
    max_gap_ratio: float,
) -> dict[str, Any]:
    """Audit one real OHLCV frame against an exact ``[start, end)`` minute grid."""
    expected = _half_open_window_bar_count(start, end)
    start_ms = _utc_epoch_ms(start)
    end_ms = _utc_epoch_ms(end)
    expected_last_ms = end_ms - TIMEFRAME_MS
    if frame.is_empty():
        return {
            "symbol": symbol,
            "status": "fail",
            "errors": ["empty_frame"],
            "warnings": [],
            "rows": 0,
            "window_contract": "[start,end)",
            "requested_start": _iso(start),
            "requested_end_exclusive": _iso(end),
            "expected_1m_bars": expected,
        }

    required = ("datetime", "open", "high", "low", "close", "volume")
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        return {
            "symbol": symbol,
            "status": "fail",
            "errors": ["missing_columns:" + ",".join(missing_columns)],
            "warnings": [],
            "rows": int(frame.height),
            "window_contract": "[start,end)",
            "requested_start": _iso(start),
            "requested_end_exclusive": _iso(end),
            "expected_1m_bars": expected,
        }

    selected = (
        frame.select(required)
        .sort("datetime")
        .with_columns(pl.col("datetime").dt.epoch("ms").alias("_timestamp_ms"))
    )
    first_time, last_time = _frame_time_bounds(selected)
    first_timestamp_ms = int(selected["_timestamp_ms"][0])
    last_timestamp_ms = int(selected["_timestamp_ms"][-1])
    unique_times = int(selected["_timestamp_ms"].n_unique())
    duplicates = int(selected.height - unique_times)
    in_window = (pl.col("_timestamp_ms") >= start_ms) & (pl.col("_timestamp_ms") < end_ms)
    on_grid = ((pl.col("_timestamp_ms") - start_ms) % TIMEFRAME_MS) == 0
    valid_unique_times = int(selected.filter(in_window & on_grid)["_timestamp_ms"].n_unique())
    missing_bars = max(0, expected - valid_unique_times)
    gap_ratio = float(missing_bars / expected) if expected else 0.0
    outside_window = int(selected.filter(~in_window).height)
    off_grid = int(selected.filter(~on_grid).height)

    errors: list[str] = []
    warnings: list[str] = []
    if first_timestamp_ms != start_ms:
        errors.append(f"first_timestamp_mismatch:{first_timestamp_ms}!={start_ms}")
    if last_timestamp_ms != expected_last_ms:
        errors.append(f"last_timestamp_mismatch:{last_timestamp_ms}!={expected_last_ms}")
    if outside_window:
        errors.append(f"outside_requested_window:{outside_window}")
    if off_grid:
        errors.append(f"off_requested_minute_grid:{off_grid}")
    if duplicates:
        errors.append(f"duplicate_timestamps:{duplicates}")
    if missing_bars:
        errors.append(f"missing_1m_bars:{missing_bars}/{expected}")

    numeric = selected.select(
        [
            *[
                pl.col(column).null_count().alias(f"{column}_nulls")
                for column in ("open", "high", "low", "close", "volume")
            ],
            *[
                pl.col(column).is_nan().sum().alias(f"{column}_nans")
                for column in ("open", "high", "low", "close", "volume")
            ],
            *[
                (pl.col(column) <= 0).sum().alias(f"{column}_nonpositive")
                for column in ("open", "high", "low", "close")
            ],
            (pl.col("volume") < 0).sum().alias("volume_negative"),
            (pl.col("volume") == 0).sum().alias("volume_zero"),
            (pl.col("high") < pl.max_horizontal("open", "close"))
            .sum()
            .alias("high_below_open_close"),
            (pl.col("low") > pl.min_horizontal("open", "close"))
            .sum()
            .alias("low_above_open_close"),
            (pl.col("high") < pl.col("low")).sum().alias("high_below_low"),
        ]
    ).row(0, named=True)

    for key, value in numeric.items():
        count = int(value or 0)
        if count <= 0:
            continue
        if key == "volume_zero":
            warnings.append(f"{key}:{count}")
        else:
            errors.append(f"{key}:{count}")

    first_close = _safe_float(selected["close"][0])
    last_close = _safe_float(selected["close"][-1])
    buy_hold_return = (last_close / first_close - 1.0) if first_close > 0 else 0.0
    return {
        "symbol": symbol,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "rows": int(selected.height),
        "unique_timestamps": unique_times,
        "expected_1m_bars": expected,
        "missing_1m_bars": missing_bars,
        "gap_ratio": gap_ratio,
        "configured_max_gap_ratio": float(max_gap_ratio),
        "window_contract": "[start,end)",
        "requested_start": _iso(start),
        "requested_end_exclusive": _iso(end),
        "expected_last_bar": _iso(end - timedelta(milliseconds=TIMEFRAME_MS)),
        "start": _iso(first_time),
        "end": _iso(last_time),
        "first_close": first_close,
        "last_close": last_close,
        "buy_hold_return": buy_hold_return,
    }


def _load_and_audit_data(
    repo: MarketDataRepository,
    *,
    exchange: str,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
    max_gap_ratio: float,
) -> tuple[dict[str, pl.DataFrame], list[dict[str, Any]]]:
    data: dict[str, pl.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    storage_end = end - timedelta(milliseconds=TIMEFRAME_MS)
    for symbol in symbols:
        frame, source_audit = repo.load_ohlcv_with_source_audit(
            exchange=exchange,
            symbol=symbol,
            timeframe="1m",
            start_date=start,
            end_date=storage_end,
        )
        audit = audit_ohlcv_frame(
            symbol,
            frame,
            start=start,
            end=end,
            max_gap_ratio=max_gap_ratio,
        )
        audit["source_lineage"] = source_audit
        audits.append(audit)
        if audit["status"] == "pass":
            data[symbol] = frame
    return data, audits


def _feature_bar_coverage(
    timestamps_ms: list[int],
    *,
    start_ms: int,
    end_ms: int,
    max_stale_ms: int,
) -> dict[str, Any]:
    expected_bars = max(0, (int(end_ms) - int(start_ms)) // TIMEFRAME_MS)
    timestamps = sorted({int(value) for value in timestamps_ms})
    available_bars = 0
    source_index = 0
    latest_source: int | None = None
    for bar_timestamp in range(int(start_ms), int(end_ms), TIMEFRAME_MS):
        while source_index < len(timestamps) and timestamps[source_index] <= bar_timestamp:
            latest_source = timestamps[source_index]
            source_index += 1
        if latest_source is not None and bar_timestamp - latest_source <= int(max_stale_ms):
            available_bars += 1
    source_gaps = [right - left for left, right in pairwise(timestamps)]
    return {
        "expected_bars": expected_bars,
        "available_bars": available_bars,
        "missing_bars": max(0, expected_bars - available_bars),
        "coverage_ratio": (float(available_bars / expected_bars) if expected_bars else 0.0),
        "first_source_timestamp_ms": timestamps[0] if timestamps else None,
        "last_source_timestamp_ms": timestamps[-1] if timestamps else None,
        "max_source_gap_ms": max(source_gaps, default=0),
    }


def _feature_audit(
    *,
    db_path: str,
    exchange: str,
    symbols: tuple[str, ...],
    start: datetime,
    end: datetime,
    required_features: tuple[str, ...],
) -> dict[str, Any]:
    if not required_features:
        return {"required_features": [], "status": "not_required", "symbols": {}}
    from lumina_quant.data.feature_points import FEATURE_POINT_MAX_STALE_MS
    from lumina_quant.market_data import load_futures_feature_points_from_db

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: dict[str, Any] = {}
    empty_symbols: list[str] = []
    complete_symbols: list[str] = []
    missing_features_by_symbol: dict[str, list[str]] = {}
    coverage_failures_by_symbol: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        frame = load_futures_feature_points_from_db(
            db_path,
            exchange=exchange,
            symbol=symbol,
            start_date=max(0, start_ms - FEATURE_POINT_MAX_STALE_MS),
            end_date=end_ms - 1,
        )
        item: dict[str, Any] = {
            "rows": int(frame.height),
            "columns": [],
            "non_null_counts": {},
            "coverage": {},
        }
        if frame.is_empty():
            empty_symbols.append(symbol)
            missing_features_by_symbol[symbol] = list(required_features)
        for feature in required_features:
            timestamps: list[int] = []
            if feature in frame.columns and "timestamp_ms" in frame.columns:
                timestamps = [
                    int(value)
                    for value in frame.filter(pl.col(feature).is_not_null())
                    .get_column("timestamp_ms")
                    .to_list()
                ]
            count = len(timestamps)
            item["non_null_counts"][feature] = count
            coverage = _feature_bar_coverage(
                timestamps,
                start_ms=start_ms,
                end_ms=end_ms,
                max_stale_ms=FEATURE_POINT_MAX_STALE_MS,
            )
            item["coverage"][feature] = coverage
            if count > 0:
                item["columns"].append(feature)
            if count == 0 or coverage["missing_bars"] > 0:
                missing_features_by_symbol.setdefault(symbol, []).append(feature)
                coverage_failures_by_symbol.setdefault(symbol, {})[feature] = coverage
        if not missing_features_by_symbol.get(symbol):
            complete_symbols.append(symbol)
        rows[symbol] = item
    status = "pass" if len(complete_symbols) == len(symbols) else "fail"
    return {
        "required_features": list(required_features),
        "status": status,
        "empty_feature_symbols": empty_symbols,
        "complete_feature_symbols": complete_symbols,
        "missing_features_by_symbol": missing_features_by_symbol,
        "coverage_failures_by_symbol": coverage_failures_by_symbol,
        "symbols": rows,
        "note": (
            "Every required feature must resolve on every 1m bar in the exact [start,end) "
            "window for every traded symbol; FeaturePointLookup uses only real points at "
            "or before the bar and an 8h stale limit."
        ),
    }


def _run_backtest(
    *,
    strategy: str,
    symbols: tuple[str, ...],
    data: dict[str, pl.DataFrame],
    data_root: str,
    exchange: str,
    start: datetime,
    end: datetime,
    annual_periods: int,
) -> dict[str, Any]:
    runtime_config = get_default_runtime_config()
    runtime_config.trading.timeframe = "1m"
    runtime_config.backtest.annual_periods = int(annual_periods)
    runtime_config.backtest.persist_output = False
    strategy_cls = resolve_strategy_class(strategy)
    params: dict[str, Any] = {}
    started = time.perf_counter()
    backtest = Backtest(
        "data",
        list(symbols),
        start,
        HistoricCSVDataHandler,
        SimulatedExecutionHandler,
        Portfolio,
        strategy_cls,
        strategy_params=params,
        end_date=end,
        data_dict=data,
        data_handler_kwargs={
            "feature_db_path": str(data_root),
            "feature_exchange": str(exchange),
        },
        record_history=False,
        track_metrics=True,
        record_trades=True,
        strategy_timeframe="1m",
        config=runtime_config,
    )
    backtest.simulate_trading(output=False)
    elapsed = time.perf_counter() - started
    totals = list(getattr(backtest.portfolio, "_metric_totals", []) or [])
    first_total = _safe_float(totals[0], 10_000.0) if totals else 10_000.0
    final_total = _safe_float(backtest.portfolio.current_holdings.get("total"), first_total)
    total_return = final_total / first_total - 1.0 if first_total > 0 else 0.0
    fast_stats = dict(backtest.portfolio.output_summary_stats_fast() or {})
    max_drawdown = _safe_float(fast_stats.get("max_drawdown"))
    min_total = min((_safe_float(value, final_total) for value in totals), default=final_total)
    equity_breach = bool(
        min_total <= 0.0 or final_total <= 0.0 or total_return <= -1.0 or max_drawdown > 1.0
    )
    return {
        "status": "pass",
        "strategy": strategy,
        "symbols": list(symbols),
        "start": _iso(start),
        "end": _iso(end),
        "elapsed_seconds": elapsed,
        "initial_equity": first_total,
        "final_equity": final_total,
        "total_return": total_return,
        "trade_count": int(getattr(backtest.portfolio, "trade_count", 0)),
        "market_events": int(getattr(backtest, "market_events", 0)),
        "signals": int(getattr(backtest, "signals", 0)),
        "orders": int(getattr(backtest, "orders", 0)),
        "fills": int(getattr(backtest, "fills", 0)),
        "fast_stats": fast_stats,
        "min_equity": min_total,
        "equity_breach": equity_breach,
        "risk_screen_status": "equity_breach" if equity_breach else "pass",
    }


def _benchmark_return(audits: list[dict[str, Any]]) -> float:
    for audit in audits:
        if audit.get("status") == "pass":
            return _safe_float(audit.get("buy_hold_return"))
    return 0.0


def _markdown_report(payload: dict[str, Any]) -> str:
    passed_results = [
        row for row in list(payload.get("strategy_results") or []) if row.get("status") == "pass"
    ]
    traded_results = [row for row in passed_results if int(row.get("trade_count") or 0) > 0]
    sorted_results = sorted(
        traded_results,
        key=lambda row: _safe_float(row.get("total_return")),
        reverse=True,
    )
    top_n = max(1, int(payload.get("top_n") or 20))
    lines = [
        "# Strategy backtest scoreboard",
        "",
        f"- scope: `{payload.get('scope', 'latest')}`",
        f"- generated_at: `{payload['generated_at']}`",
        f"- data_root: `{payload['data_root']}`",
        f"- exchange: `{payload['exchange']}`",
        "- timeframe: `1m`",
        f"- period `[start,end)`: `{payload['start']}` → `{payload['end']}`",
        f"- annual_periods: `{payload['annual_periods']}`",
        f"- strategy_count: `{len(payload['strategy_results'])}`",
        f"- pass_count: `{len(passed_results)}`",
        f"- excluded_count: `{len([row for row in payload['strategy_results'] if row.get('status') == 'excluded'])}`",
        "",
        "## Data integrity summary",
        "",
    ]
    exchange_audit = payload.get("exchange_symbol_audit") or {}
    feature_status_counts: dict[str, int] = {}
    for feature_audit in dict(payload.get("feature_audits") or {}).values():
        status = str(dict(feature_audit or {}).get("status") or "unknown")
        feature_status_counts[status] = feature_status_counts.get(status, 0) + 1
    data_warning_count = 0
    for audits in dict(payload.get("data_audits") or {}).values():
        for audit in list(audits or []):
            if dict(audit or {}).get("warnings"):
                data_warning_count += 1
    lines.extend(
        [
            f"- exchange symbol audit: `{exchange_audit.get('status')}` "
            f"(matched={exchange_audit.get('matched_count', 'n/a')}, "
            f"unmatched={len(exchange_audit.get('unmatched_symbols') or [])})",
            f"- feature audit statuses: `{feature_status_counts}`",
            f"- OHLCV warning rows: `{data_warning_count}`",
            "- OHLCV policy: exact requested [start,end) 1m grid, no gap fill, no interpolation, and no synthetic rows; any missing, duplicate, off-grid, or out-of-window bar fails before simulation.",
            "- Required external features must resolve on every 1m bar in the same half-open window under the bounded 8h stale policy; sparse columns fail before simulation.",
            "- Zero-volume bars are reported as warnings, not imputed.",
            "",
            f"## Top {min(top_n, len(sorted_results))} traded performers by total return",
            "",
            "- Pass-status strategies with zero completed trades are not treated as performance winners; they are listed in zero-trade diagnostics instead.",
            "| Rank | Strategy | Tier | Symbols | Return | Sharpe | CAGR | MDD | Trades | Signals | Zero-trade reason |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(sorted_results[:top_n], start=1):
        stats = dict(row.get("fast_stats") or {})
        lines.append(
            "| {rank} | `{strategy}` | `{tier}` | {symbol_count} | {ret:.2%} | {sharpe:.3f} | "
            "{cagr:.2%} | {mdd:.2%} | {trades} | {signals} | `{zero_reason}` |".format(
                rank=idx,
                strategy=row.get("strategy"),
                tier=row.get("tier", ""),
                symbol_count=len(row.get("symbols") or []),
                ret=_safe_float(row.get("total_return")),
                sharpe=_safe_float(stats.get("sharpe")),
                cagr=_safe_float(stats.get("cagr")),
                mdd=_safe_float(stats.get("max_drawdown")),
                trades=int(row.get("trade_count") or 0),
                signals=int(row.get("signals") or 0),
                zero_reason=row.get("zero_trade_reason") or "",
            )
        )

    zero_trade_rows = [row for row in passed_results if int(row.get("trade_count") or 0) == 0]
    lines.extend(["", "## Zero-trade diagnostics", ""])
    if not zero_trade_rows:
        lines.append("- No pass-status strategy finished with zero trades.")
    else:
        lines.extend(
            [
                "| Strategy | Market events | Signals | Orders | Fills | Reason |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in zero_trade_rows:
            lines.append(
                "| `{strategy}` | {market_events} | {signals} | {orders} | {fills} | `{reason}` |".format(
                    strategy=row.get("strategy"),
                    market_events=int(row.get("market_events") or 0),
                    signals=int(row.get("signals") or 0),
                    orders=int(row.get("orders") or 0),
                    fills=int(row.get("fills") or 0),
                    reason=row.get("zero_trade_reason") or "",
                )
            )

    lines.extend(
        [
            "",
            "## All strategy results",
            "",
            "| Strategy | Tier | Symbols | Return | Sharpe | CAGR | MDD | Trades | Signals | Feature audit | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["strategy_results"]:
        stats = dict(row.get("fast_stats") or {})
        lines.append(
            "| `{strategy}` | `{tier}` | {symbol_count} | {ret:.2%} | {sharpe:.3f} | "
            "{cagr:.2%} | {mdd:.2%} | {trades} | {signals} | `{feature_status}` | `{status}` |".format(
                strategy=row.get("strategy"),
                tier=row.get("tier", ""),
                symbol_count=len(row.get("symbols") or []),
                ret=_safe_float(row.get("total_return")),
                sharpe=_safe_float(stats.get("sharpe")),
                cagr=_safe_float(stats.get("cagr")),
                mdd=_safe_float(stats.get("max_drawdown")),
                trades=int(row.get("trade_count") or 0),
                signals=int(row.get("signals") or 0),
                feature_status=row.get("feature_audit_status", "not_required"),
                status=row.get("status"),
            )
        )
    lines.extend(["", "## Issues and audit notes", ""])
    issues = list(payload.get("issues") or [])
    if not issues:
        lines.append("- No fail-level data integrity or runtime issues remained after this run.")
    else:
        for issue in issues:
            lines.append(
                f"- `{issue.get('severity')}` `{issue.get('scope')}`: {issue.get('message')}"
            )
    lines.extend(["", "## Output files", ""])
    for key, value in sorted(dict(payload.get("output_paths") or {}).items()):
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def _write_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(payload["generated_at"]).replace(":", "").replace("-", "")
    timestamp = timestamp.replace(".", "_").replace("Z", "Z")
    json_path = output_dir / f"latest_alpha_sleeves_backtest_{timestamp}.json"
    md_path = output_dir / f"latest_alpha_sleeves_backtest_{timestamp}.md"
    latest_json = output_dir / "latest_alpha_sleeves_backtest_latest.json"
    latest_md = output_dir / "latest_alpha_sleeves_backtest_latest.md"
    payload["output_paths"] = {
        "timestamped_json": str(json_path),
        "timestamped_markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    json_path.write_text(text + "\n", encoding="utf-8")
    latest_json.write_text(text + "\n", encoding="utf-8")
    markdown = _markdown_report(payload)
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return dict(payload["output_paths"])


def run_latest_alpha_sleeve_backtests(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("LQ_BACKTEST_SUPPRESS_PARTIAL_FILL_LOGS", "1")
    os.environ.setdefault("LQ_BACKTEST_SUPPRESS_CIRCUIT_BREAKER_LOGS", "1")
    os.environ.setdefault("LQ_AUDIT_BACKTEST", "0")

    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    _half_open_window_bar_count(start, end)
    if float(args.max_gap_ratio) != 0.0:
        raise ValueError("--max-gap-ratio must be 0 for exact [start,end) auditing")

    scope = str(getattr(args, "scope", "latest") or "latest")
    top_n = int(getattr(args, "top_n", 20) or 20)
    specs = _strategy_specs(args.strategy, scope=scope)
    all_symbols = sorted({symbol for spec in specs for symbol in spec.symbols})
    exchange_audit = _audit_exchange_symbols(
        all_symbols,
        enabled=not bool(args.no_exchange_audit),
        fail_on_error=bool(args.fail_on_exchange_audit_error),
    )
    if exchange_audit.get("status") == "fail":
        raise RuntimeError("exchange symbol audit failed: " + json.dumps(exchange_audit))

    repo = MarketDataRepository(args.data_root)
    allow_unavailable = bool(getattr(args, "allow_unavailable", False))
    strategy_results: list[dict[str, Any]] = []
    data_audits: dict[str, list[dict[str, Any]]] = {}
    feature_audits: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    loaded_data, loaded_audits = _load_and_audit_data(
        repo,
        exchange=args.exchange,
        symbols=tuple(all_symbols),
        start=start,
        end=end,
        max_gap_ratio=float(args.max_gap_ratio),
    )
    audit_by_symbol = {str(audit["symbol"]): audit for audit in loaded_audits}

    for spec in specs:
        print(f"[RUN] {spec.strategy} symbols={','.join(spec.symbols)}", flush=True)
        data = {symbol: loaded_data[symbol] for symbol in spec.symbols if symbol in loaded_data}
        audits = [audit_by_symbol[symbol] for symbol in spec.symbols]
        data_audits[spec.strategy] = audits
        failed_audits = [audit for audit in audits if audit.get("status") != "pass"]
        if failed_audits:
            message = "data audit failed for " + ",".join(
                str(audit.get("symbol")) for audit in failed_audits
            )
            issues.append({"severity": "fail", "scope": spec.strategy, "message": message})
            strategy_results.append(
                {
                    "status": "fail",
                    "strategy": spec.strategy,
                    "tier": get_strategy_tier(spec.strategy),
                    "symbols": list(spec.symbols),
                    "error": message,
                    "benchmark_return_first_symbol": _benchmark_return(audits),
                    "fast_stats": {},
                    "trade_count": 0,
                    "market_events": 0,
                    "signals": 0,
                    "orders": 0,
                    "fills": 0,
                    "total_return": 0.0,
                    "feature_audit_status": "not_run",
                    "zero_trade_reason": "",
                }
            )
            if bool(args.fail_fast):
                raise RuntimeError(message)
            continue

        strategy_cls = resolve_strategy_class(spec.strategy)
        required_features = _required_features_for_strategy(strategy_cls)
        feature_audits[spec.strategy] = _feature_audit(
            db_path=args.data_root,
            exchange=args.exchange,
            symbols=spec.symbols,
            start=start,
            end=end,
            required_features=required_features,
        )
        feature_status = str(feature_audits[spec.strategy].get("status") or "")
        if feature_status == "fail":
            message = "required feature audit failed for all symbols"
            severity = "warn" if allow_unavailable else "fail"
            issues.append({"severity": severity, "scope": spec.strategy, "message": message})
            strategy_results.append(
                {
                    "status": "excluded" if allow_unavailable else "fail",
                    "strategy": spec.strategy,
                    "tier": get_strategy_tier(spec.strategy),
                    "symbols": list(spec.symbols),
                    "error": message,
                    "exclusion_reason": (
                        "required feature data unavailable; no missing feature was inferred, filled, or synthesized"
                        if allow_unavailable
                        else ""
                    ),
                    "benchmark_return_first_symbol": _benchmark_return(audits),
                    "fast_stats": {},
                    "trade_count": 0,
                    "market_events": 0,
                    "signals": 0,
                    "orders": 0,
                    "fills": 0,
                    "total_return": 0.0,
                    "feature_audit_status": feature_status,
                    "zero_trade_reason": "",
                }
            )
            if bool(args.fail_fast) and not allow_unavailable:
                raise RuntimeError(message)
            continue

        try:
            result = _run_backtest(
                strategy=spec.strategy,
                symbols=spec.symbols,
                data=data,
                data_root=str(args.data_root),
                exchange=str(args.exchange),
                start=start,
                end=end,
                annual_periods=int(args.annual_periods),
            )
            result["tier"] = get_strategy_tier(spec.strategy)
            result["benchmark_return_first_symbol"] = _benchmark_return(audits)
            result["feature_audit_status"] = feature_status
            if bool(result.get("equity_breach")):
                message = "equity_breach_invalidates_performance_ranking"
                result["status"] = "excluded"
                result["error"] = message
                result["exclusion_reason"] = (
                    "equity path reached <=0 or max drawdown exceeded 100%; "
                    "this is a bankruptcy/risk-model breach, not a valid performance winner"
                )
                issues.append({"severity": "warn", "scope": spec.strategy, "message": message})
            result["zero_trade_reason"] = _zero_trade_reason(result)
            strategy_results.append(result)
        except Exception as exc:
            message = repr(exc)
            unavailable = allow_unavailable and _is_unavailable_runtime_error(message)
            issues.append(
                {
                    "severity": "warn" if unavailable else "fail",
                    "scope": spec.strategy,
                    "message": message,
                }
            )
            strategy_results.append(
                {
                    "status": "excluded" if unavailable else "fail",
                    "strategy": spec.strategy,
                    "tier": get_strategy_tier(spec.strategy),
                    "symbols": list(spec.symbols),
                    "error": message,
                    "exclusion_reason": (
                        "runtime input/feature unavailable for this local execution path; no proxy data was fabricated"
                        if unavailable
                        else ""
                    ),
                    "benchmark_return_first_symbol": _benchmark_return(audits),
                    "fast_stats": {},
                    "trade_count": 0,
                    "market_events": 0,
                    "signals": 0,
                    "orders": 0,
                    "fills": 0,
                    "total_return": 0.0,
                    "feature_audit_status": feature_status,
                    "zero_trade_reason": "",
                }
            )
            if bool(args.fail_fast) and not unavailable:
                raise

    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "artifact_kind": "strategy_backtest_scoreboard",
        "scope": scope,
        "top_n": top_n,
        "generated_at": generated_at,
        "git_commit": _git_commit(),
        "data_root": str(Path(args.data_root).resolve()),
        "exchange": str(args.exchange),
        "timeframe": "1m",
        "start": _iso(start),
        "end": _iso(end),
        "window_contract": "[start,end)",
        "annual_periods": int(args.annual_periods),
        "max_gap_ratio": float(args.max_gap_ratio),
        "new_strategy_source_files": [
            "src/lumina_quant/strategies/external_alpha_sleeves.py",
            "src/lumina_quant/strategies/robust_alpha_sleeves.py",
            "src/lumina_quant/strategies/adaptive_crypto_alpha_sleeves.py",
            "src/lumina_quant/strategies/cross_asset_tradfi_alpha_sleeves.py",
            "src/lumina_quant/indicators/alpha_features.py",
        ],
        "exchange_symbol_audit": exchange_audit,
        "data_audits": data_audits,
        "feature_audits": feature_audits,
        "strategy_results": strategy_results,
        "issues": issues,
        "integrity_policy": {
            "no_gap_fill": True,
            "no_interpolation": True,
            "no_synthetic_rows": True,
            "missing_bar_policy": "exact_[start,end)_minute_grid_fail_closed_before_simulation",
            "duplicate_timestamp_policy": "fail_closed_before_simulation",
            "feature_staleness_policy": "FeaturePointLookup bounded <= 8h forward-fill only",
            "execution_model": "repository Backtest + Portfolio + SimulatedExecutionHandler",
        },
    }
    paths = _write_outputs(payload, Path(args.output_dir))
    print(json.dumps({"output_paths": paths, "issues": issues}, indent=2), flush=True)
    return payload


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run audited strategy backtests and write a fail-closed performance scoreboard."
    )
    parser.add_argument("--data-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--annual-periods", type=int, default=365 * 24 * 60)
    parser.add_argument("--max-gap-ratio", type=float, default=0.0)
    parser.add_argument(
        "--strategy",
        action="append",
        help="Run one named strategy; repeat to select multiple. Defaults to the selected --scope.",
    )
    parser.add_argument(
        "--scope",
        choices=("latest", "live", "all"),
        default="latest",
        help="Strategy universe when --strategy is omitted.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of traded performers to show in the markdown top table.",
    )
    parser.add_argument("--output-dir", default="var/reports/latest_new_strategy_backtests")
    parser.add_argument("--no-exchange-audit", action="store_true")
    parser.add_argument("--fail-on-exchange-audit-error", action="store_true")
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help=(
            "Record strategies with unavailable required inputs/features as excluded instead of "
            "fabricating data or failing the whole scoreboard."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_latest_alpha_sleeve_backtests(args)
    return 1 if any(issue.get("severity") == "fail" for issue in payload.get("issues") or []) else 0


if __name__ == "__main__":
    raise SystemExit(main())
