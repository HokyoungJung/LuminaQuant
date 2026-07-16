"""Candidate research runner with train/val/oos evaluation and robust metrics."""

from __future__ import annotations

import hashlib
import math
import os
import queue
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.configuration import get_default_runtime_config
from lumina_quant.market_data import load_futures_feature_points_from_db
from lumina_quant.storage.parquet import load_data_dict_from_parquet
from lumina_quant.data.timeframe import timeframe_to_milliseconds
from lumina_quant.symbols import (
    canonical_symbol,
    canonicalize_symbol_list,
)
from lumina_quant.strategy_factory import research_run_support as _research_run_support
from lumina_quant.strategy_factory import research_stage_support as _research_stage_support
from lumina_quant.strategy_factory import research_metrics as _research_metrics
from lumina_quant.strategy_factory.runtime_settings import (
    current_research_market_data_settings as _current_research_market_data_settings_impl,
)
from lumina_quant.strategy_factory.research_reporting import ResearchReportBuilder
from lumina_quant.strategy_factory.strategy_signal_dispatch import StrategySignalDispatcher

resolve_risk_free_config = _research_metrics.resolve_risk_free_config
_resolve_feature_points_path = _research_run_support._resolve_feature_points_path

_PERIODS_PER_YEAR = {
    "1s": 31_536_000,
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}

_MIN_BARS = 360
_METALS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}
_LEADERS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT")
_FEATURE_POINT_MAX_STALE_MS = 8 * 60 * 60 * 1000
_FEATURE_POINT_COLUMNS: tuple[str, ...] = (
    "funding_rate",
    "funding_mark_price",
    "funding_fee_rate",
    "funding_fee_quote_per_unit",
    "mark_price",
    "index_price",
    "open_interest",
    "taker_buy_base_volume",
    "taker_sell_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
    "liquidation_long_qty",
    "liquidation_short_qty",
    "liquidation_long_notional",
    "liquidation_short_notional",
    "best_bid_price",
    "best_bid_quantity",
    "best_ask_price",
    "best_ask_quantity",
    "bbo_mid_price",
    "bbo_spread_bps",
    "book_depth_bid_notional_1pct",
    "book_depth_ask_notional_1pct",
    "book_depth_imbalance_1pct",
)


def _current_research_market_data_settings(
    runtime_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _current_research_market_data_settings_impl(runtime_settings)


DEFAULT_RESEARCH_SCORING_CONFIG = _research_run_support.DEFAULT_RESEARCH_SCORING_CONFIG


def _resolve_score_config(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    return _research_run_support._resolve_score_config(overrides)


@dataclass(slots=True)
class SeriesBundle:
    symbol: str
    timeframe: str
    datetime: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


class _AlignedStrategyBarStore:
    def __init__(self, symbols: Sequence[str]):
        self.symbol_list = canonicalize_symbol_list(list(symbols))
        self._rows = {
            symbol: {
                "datetime": None,
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
            }
            for symbol in self.symbol_list
        }

    def set_bar(
        self,
        symbol: str,
        event_time: Any,
        *,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float,
    ) -> None:
        token = canonical_symbol(str(symbol))
        if token not in self._rows:
            return
        self._rows[token] = {
            "datetime": event_time,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(close_price),
            "volume": float(volume),
        }

    def get_latest_bar_value(self, symbol: str, value_type: str) -> float:
        token = canonical_symbol(str(symbol))
        row = self._rows.get(token) or {}
        value = row.get(str(value_type), 0.0)
        return float(value) if value is not None else 0.0

    def get_latest_bar_datetime(self, symbol: str) -> Any:
        token = canonical_symbol(str(symbol))
        row = self._rows.get(token) or {}
        return row.get("datetime")


def _simulate_event_driven_strategy_exposures(
    strategy_cls: type[Any],
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
) -> np.ndarray:
    from lumina_quant.core.events import MarketEvent, MarketWindowEvent

    canonical_symbols = canonicalize_symbol_list(list(symbols))
    raw_datetimes = np.asarray(aligned.get("datetime"))
    datetimes = np.asarray(raw_datetimes, dtype=object)
    n = len(datetimes)
    exposures = np.zeros((len(canonical_symbols), n), dtype=float)
    bars = _AlignedStrategyBarStore(canonical_symbols)
    events: queue.Queue[Any] = queue.Queue()
    strategy = strategy_cls(bars, events, **dict(params))
    position_state = dict.fromkeys(canonical_symbols, 0.0)
    symbol_index = {symbol: idx for idx, symbol in enumerate(canonical_symbols)}

    # Cross-sectional sleeves evaluate on MARKET_WINDOW snapshots (one coherent
    # event per bar carrying every symbol), matching the production window feed
    # and every lane test harness.  Classes that only inherit the BASE
    # ``Strategy.calculate_signals_window`` default (the legacy Alpha101/pair
    # simulator consumers) keep the per-symbol MARKET feed byte-identically --
    # window capability means the class OVERRIDES the window callback.
    from lumina_quant.strategy import Strategy as _BaseStrategy

    _window_method = getattr(type(strategy), "calculate_signals_window", None)
    window_capable = callable(_window_method) and _window_method is not (
        _BaseStrategy.calculate_signals_window
    )
    window_seconds = 60
    if n >= 2:
        try:
            if np.issubdtype(raw_datetimes.dtype, np.datetime64):
                spacing = float((raw_datetimes[1] - raw_datetimes[0]) / np.timedelta64(1, "s"))
            else:
                spacing = float(datetimes[1].timestamp() - datetimes[0].timestamp())
            if spacing > 0.0:
                window_seconds = max(1, int(spacing))
        except AttributeError, TypeError, ValueError, OverflowError:
            window_seconds = 60

    for idx in range(n):
        event_time = datetimes[idx]
        for symbol in canonical_symbols:
            bars.set_bar(
                symbol,
                event_time,
                open_price=float(aligned[f"{symbol}:open"][idx]),
                high_price=float(aligned[f"{symbol}:high"][idx]),
                low_price=float(aligned[f"{symbol}:low"][idx]),
                close_price=float(aligned[f"{symbol}:close"][idx]),
                volume=float(aligned[f"{symbol}:volume"][idx]),
            )

        if window_capable:
            bars_1s = {
                symbol: (
                    {
                        "time": event_time,
                        "open": float(aligned[f"{symbol}:open"][idx]),
                        "high": float(aligned[f"{symbol}:high"][idx]),
                        "low": float(aligned[f"{symbol}:low"][idx]),
                        "close": float(aligned[f"{symbol}:close"][idx]),
                        "volume": float(aligned[f"{symbol}:volume"][idx]),
                    },
                )
                for symbol in canonical_symbols
            }
            strategy.calculate_signals(
                MarketWindowEvent(
                    time=event_time,
                    window_seconds=window_seconds,
                    bars_1s=bars_1s,
                )
            )
        else:
            for symbol in canonical_symbols:
                strategy.calculate_signals(
                    MarketEvent(
                        time=event_time,
                        symbol=symbol,
                        open=float(aligned[f"{symbol}:open"][idx]),
                        high=float(aligned[f"{symbol}:high"][idx]),
                        low=float(aligned[f"{symbol}:low"][idx]),
                        close=float(aligned[f"{symbol}:close"][idx]),
                        volume=float(aligned[f"{symbol}:volume"][idx]),
                    )
                )

        while not events.empty():
            signal = events.get()
            token = canonical_symbol(str(getattr(signal, "symbol", "")))
            if token not in position_state:
                continue
            signal_type = str(getattr(signal, "signal_type", "")).upper()
            if signal_type == "LONG":
                position_state[token] = 1.0
            elif signal_type == "SHORT":
                position_state[token] = -1.0
            elif signal_type == "EXIT":
                position_state[token] = 0.0

        for symbol, value in position_state.items():
            exposures[symbol_index[symbol], idx] = float(value)

    return exposures


def _load_event_driven_strategy_impl(strategy_class: str) -> type[Any]:
    if strategy_class == "Alpha101FormulaStrategy":
        from lumina_quant.strategies.alpha101_formula import Alpha101FormulaStrategy

        return Alpha101FormulaStrategy
    if strategy_class == "PairTradingZScoreStrategy":
        from lumina_quant.strategies.pair_trading_zscore import PairTradingZScoreStrategy

        return PairTradingZScoreStrategy
    if strategy_class == "PairSpreadZScoreStrategy":
        from lumina_quant.strategies.pair_spread_zscore import PairSpreadZScoreStrategy

        return PairSpreadZScoreStrategy
    msg = f"Unsupported event-driven proxy strategy: {strategy_class}"
    raise ValueError(msg)


def _resolve_symbol_pair(
    symbols: Sequence[str],
    params: Mapping[str, Any],
) -> tuple[str, str, int, int]:
    symbol_x = canonical_symbol(str(params.get("symbol_x") or symbols[0]))
    symbol_y = canonical_symbol(str(params.get("symbol_y") or symbols[1]))
    if symbol_x not in symbols:
        symbol_x = symbols[0]
    if symbol_y not in symbols:
        symbol_y = symbols[1]
    return symbol_x, symbol_y, symbols.index(symbol_x), symbols.index(symbol_y)


def _pair_spread_fallback_exposures(
    *,
    aligned: Mapping[str, np.ndarray],
    symbol_x: str,
    symbol_y: str,
    length: int,
    entry_z: float,
    exit_z: float,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.nan_to_num(
        _pair_spread_z(
            aligned[f"{symbol_x}:close"],
            aligned[f"{symbol_y}:close"],
            window=lookback,
        ),
        nan=0.0,
    )

    long_spread = z <= -entry_z
    short_spread = z >= entry_z
    exit_cond = np.abs(z) <= exit_z

    x_pos = np.zeros(length, dtype=float)
    y_pos = np.zeros(length, dtype=float)
    x_pos = np.where(long_spread, -1.0, x_pos)
    y_pos = np.where(long_spread, 1.0, y_pos)
    x_pos = np.where(short_spread, 1.0, x_pos)
    y_pos = np.where(short_spread, -1.0, y_pos)
    x_pos = np.where(exit_cond, 0.0, x_pos)
    y_pos = np.where(exit_cond, 0.0, y_pos)
    return x_pos, y_pos


def _load_feature_cache(
    *,
    symbols: Sequence[str],
    start_date: Any = None,
    end_date: Any = None,
    market_data_settings: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, pl.DataFrame]:
    defaults = _current_research_market_data_settings(market_data_settings)
    db_path = str(defaults["parquet_root"])
    exchange = str(defaults["exchange"])
    cache: dict[str, pl.DataFrame] = {}

    normalized_symbols = canonicalize_symbol_list(symbols)
    symbol_count = len(normalized_symbols)
    for symbol_index, symbol in enumerate(normalized_symbols, start=1):
        if progress_callback is not None:
            progress_callback(
                "resource_feature_symbol_started",
                {
                    "symbol": symbol,
                    "symbol_index": symbol_index,
                    "symbol_count": symbol_count,
                    "loaded_count": max(0, symbol_index - 1),
                },
            )
        started_at = perf_counter()
        frame = _load_feature_frame(
            db_path=db_path,
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback,
        )
        normalized_frame = _normalize_feature_frame(frame)
        cache[symbol] = normalized_frame
        if progress_callback is not None:
            progress_callback(
                "resource_feature_symbol_loaded",
                {
                    "symbol": symbol,
                    "symbol_index": symbol_index,
                    "symbol_count": symbol_count,
                    "loaded_count": symbol_index,
                    "row_count": int(normalized_frame.height),
                    "elapsed_seconds": round(max(0.0, perf_counter() - started_at), 6),
                },
            )

    return cache


def _load_feature_frame(
    *,
    db_path: str,
    exchange: str,
    symbol: str,
    start_date: Any,
    end_date: Any,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> pl.DataFrame:
    try:
        try:
            return load_futures_feature_points_from_db(
                db_path,
                exchange=exchange,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                progress_callback=progress_callback,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return load_futures_feature_points_from_db(
                db_path,
                exchange=exchange,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return pl.DataFrame()
    except FileNotFoundError, OSError, RuntimeError, ValueError:
        return pl.DataFrame()


def _normalize_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "timestamp_ms" not in frame.columns:
        return pl.DataFrame()

    cleaned = frame.filter(pl.col("timestamp_ms").is_not_null()).with_columns(
        pl.col("timestamp_ms").cast(pl.Int64)
    )
    if cleaned.is_empty():
        return pl.DataFrame()

    for field in _FEATURE_POINT_COLUMNS:
        if field not in cleaned.columns:
            cleaned = cleaned.with_columns(pl.lit(None, dtype=pl.Float64).alias(field))

    cleaned = (
        cleaned.select(["timestamp_ms", *_FEATURE_POINT_COLUMNS])
        .sort("timestamp_ms")
        .unique(
            subset=["timestamp_ms"],
            keep="last",
        )
    )
    source_cols = [f"__{field}_source_timestamp_ms" for field in _FEATURE_POINT_COLUMNS]
    value_cols = [f"__{field}_ffill" for field in _FEATURE_POINT_COLUMNS]
    bounded = cleaned.with_columns(
        [
            pl.col("timestamp_ms").cast(pl.Int64),
            pl.from_epoch("timestamp_ms", time_unit="ms").alias("datetime"),
            *[
                pl.when(pl.col(field).is_not_null())
                .then(pl.col("timestamp_ms"))
                .otherwise(None)
                .cast(pl.Int64)
                .forward_fill()
                .alias(source_col)
                for field, source_col in zip(_FEATURE_POINT_COLUMNS, source_cols, strict=True)
            ],
            *[
                pl.col(field).cast(pl.Float64).forward_fill().alias(value_col)
                for field, value_col in zip(_FEATURE_POINT_COLUMNS, value_cols, strict=True)
            ],
        ]
    ).with_columns(
        [
            pl.when(
                pl.col(source_col).is_not_null()
                & ((pl.col("timestamp_ms") - pl.col(source_col)) <= _FEATURE_POINT_MAX_STALE_MS)
            )
            .then(pl.col(value_col))
            .otherwise(None)
            .alias(field)
            for field, source_col, value_col in zip(
                _FEATURE_POINT_COLUMNS,
                source_cols,
                value_cols,
                strict=True,
            )
        ]
    )
    return bounded.drop([*source_cols, *value_cols])


def _crowding_support_series(
    *,
    funding_rate: np.ndarray,
    open_interest: np.ndarray,
    mark_price: np.ndarray | None = None,
    index_price: np.ndarray | None = None,
    liquidation_long_notional: np.ndarray | None = None,
    liquidation_short_notional: np.ndarray | None = None,
    window: int = 96,
) -> dict[str, np.ndarray]:
    n = int(funding_rate.size)
    empty = np.full(n, np.nan, dtype=float)
    if n <= 0:
        return {
            "crowding_score": empty,
            "funding_z": empty,
            "oi_delta_z": empty,
            "basis_z": empty,
            "liquidation_imbalance_z": empty,
        }

    funding = np.asarray(funding_rate, dtype=float)
    oi = np.asarray(open_interest, dtype=float)
    mark = None if mark_price is None else np.asarray(mark_price, dtype=float)
    index = None if index_price is None else np.asarray(index_price, dtype=float)
    long_liq = (
        np.zeros(n, dtype=float)
        if liquidation_long_notional is None
        else np.nan_to_num(np.asarray(liquidation_long_notional, dtype=float), nan=0.0)
    )
    short_liq = (
        np.zeros(n, dtype=float)
        if liquidation_short_notional is None
        else np.nan_to_num(np.asarray(liquidation_short_notional, dtype=float), nan=0.0)
    )

    oi_prev = np.roll(oi, 1)
    oi_prev[0] = np.nan
    oi_delta = np.full(n, np.nan, dtype=float)
    oi_mask = np.isfinite(oi) & np.isfinite(oi_prev) & (oi > 0.0) & (oi_prev > 0.0)
    oi_delta[oi_mask] = np.log(oi[oi_mask] / oi_prev[oi_mask])

    basis = np.full(n, np.nan, dtype=float)
    if mark is not None and index is not None:
        basis_mask = np.isfinite(mark) & np.isfinite(index) & (np.abs(index) > 1e-12)
        basis[basis_mask] = (mark[basis_mask] - index[basis_mask]) / index[basis_mask]

    liq_den = np.abs(long_liq) + np.abs(short_liq) + 1e-12
    liq_imbalance = (long_liq - short_liq) / liq_den

    funding_z = _rolling_z(np.nan_to_num(funding, nan=0.0), max(16, int(window)))
    oi_delta_z = _rolling_z(np.nan_to_num(oi_delta, nan=0.0), max(16, int(window // 2)))
    basis_z = _rolling_z(np.nan_to_num(basis, nan=0.0), max(12, int(window // 2)))
    liq_z = _rolling_z(np.nan_to_num(liq_imbalance, nan=0.0), max(12, int(window // 2)))

    valid = np.isfinite(funding) & np.isfinite(oi)
    score = np.tanh(
        (0.45 * np.nan_to_num(funding_z, nan=0.0))
        + (0.35 * np.nan_to_num(oi_delta_z, nan=0.0))
        + (0.15 * np.nan_to_num(basis_z, nan=0.0))
        + (0.05 * np.nan_to_num(liq_z, nan=0.0))
    )
    score = np.where(valid, score, np.nan)
    return {
        "crowding_score": score.astype(float, copy=False),
        "funding_z": np.where(valid, funding_z, np.nan),
        "oi_delta_z": np.where(valid, oi_delta_z, np.nan),
        "basis_z": np.where(valid, basis_z, np.nan),
        "liquidation_imbalance_z": np.where(valid, liq_z, np.nan),
    }


def _hash_unit_interval(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)


def _hash_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _candidate_identity(candidate: dict[str, Any]) -> str:
    return _research_run_support._candidate_identity(candidate)


def adapt_legacy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Adapt legacy candidate fields to v2 contract without data loss."""
    return _research_run_support.adapt_legacy_candidate(candidate)


def _family_from_strategy(strategy_class: str) -> str:
    return _research_run_support._family_from_strategy(strategy_class)


def _coerce_utc_datetime(value: Any, *, end_of_day: bool = False) -> datetime | None:
    return _research_run_support._coerce_utc_datetime(value, end_of_day=end_of_day)


def _datetime_to_iso_z(value: datetime | None) -> str | None:
    return _research_run_support._datetime_to_iso_z(value)


def _to_numpy_datetime64_ms(value: datetime | None) -> np.datetime64 | None:
    if value is None:
        return None
    return np.datetime64(value.astimezone(UTC).replace(tzinfo=None), "ms")


def _split_window_bounds(
    split: Mapping[str, Any] | None,
) -> tuple[datetime | None, datetime | None]:
    return _research_run_support._split_window_bounds(split)


def _resolve_split_config(
    split: Mapping[str, Any] | None,
    *,
    strategy_timeframe: str,
) -> dict[str, Any]:
    return _research_run_support._resolve_split_config(
        split,
        strategy_timeframe=strategy_timeframe,
    )


# ── Research selection/split gate flags (opt-in; legacy default = OFF) ──────────
# These flags correct three validation-wiring defects from the 2026-07-06
# strategy-viability review. Every flag defaults to the legacy value so that,
# when a caller does not set it, the code path is byte-identical to today; the
# corrected behavior is opt-in via the strict research profile.
_RESEARCH_STRICT_SELECTION_GATE_DEFAULT = False
_RESEARCH_DSR_GATE_FLOOR_DEFAULT = 0.0
_RESEARCH_SPA_GATE_CEILING_DEFAULT = 1.0
_RESEARCH_USE_LOCKBOX_SPLIT_DEFAULT = False
_RESEARCH_LOCKBOX_OOS_FRAC_DEFAULT = 0.10
_RESEARCH_PURGE_EMBARGO_BARS_DEFAULT = 0
# HAC/Newey-West-correct the iid degrees of freedom behind the per-candidate DSR
# studentization (``research_metrics.deflated_sharpe_ratio``'s own ``hac_inference``
# kwarg). Default False -> byte-identical iid math; opt-in via
# ``score_config['research']['hac_inference']`` (see ``_research_flag``).
_RESEARCH_HAC_INFERENCE_DEFAULT = False
# Config-gated realistic cost floor for the VECTORIZED research scorer
# (performance_lever_measurement 2026-07-08). ``multiplier == 1.0`` AND
# ``override is None`` -> the exact legacy per-class map value (byte-identical);
# opt-in via ``score_config['research']['cost_rate_multiplier']`` /
# ``['cost_rate_bps_override']`` (see ``_candidate_cost_rate`` / ``_research_flag``).
_RESEARCH_COST_RATE_MULTIPLIER_DEFAULT = 1.0
_RESEARCH_COST_RATE_BPS_OVERRIDE_DEFAULT: float | None = None
_RESEARCH_SENTINEL = object()


def _research_flag(config: Any, name: str, default: Any) -> Any:
    """Read a research-gate flag defensively from a scoring-config carrier.

    The flag lives under a nested ``research`` section of the resolved scoring
    config (a Mapping). An object-style config (e.g. a SimpleNamespace with a
    ``research`` attribute or the flag as a direct attribute) is also supported.
    When the flag is absent the legacy ``default`` is returned, so callers that
    never set it keep today's behavior exactly.
    """
    research: Any = None
    if isinstance(config, Mapping):
        research = config.get("research")
    elif config is not None:
        research = getattr(config, "research", None)
    if isinstance(research, Mapping) and name in research:
        return research[name]
    if research is not None and not isinstance(research, Mapping):
        found = getattr(research, name, _RESEARCH_SENTINEL)
        if found is not _RESEARCH_SENTINEL:
            return found
    if config is not None and not isinstance(config, Mapping):
        return getattr(config, name, default)
    return default


def _split_flag(split: Mapping[str, Any] | None, name: str, default: Any) -> Any:
    """Read a split-structure flag (lockbox/purge) from the split mapping."""
    if isinstance(split, Mapping) and name in split:
        return split[name]
    return default


def _split_lengths(
    total: int, *, train_frac: float = 0.60, val_frac: float = 0.20
) -> tuple[slice, slice, slice]:
    if total <= 0:
        return slice(0, 0), slice(0, 0), slice(0, 0)
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(total - 1, val_end)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, total)


def _split_lengths_lockbox(
    total: int,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    oos_frac: float = _RESEARCH_LOCKBOX_OOS_FRAC_DEFAULT,
) -> tuple[slice, slice, slice, slice]:
    """4-way train/val/oos/lockbox split (opt-in lockbox discipline).

    The train/val boundaries match :func:`_split_lengths` so only the tail of the
    legacy OOS window is carved into a never-touched lockbox segment. Ranking and
    the binding hurdle run on validation; the lockbox is scored exactly once and
    reported as the sole OOS.
    """
    if total <= 0:
        return slice(0, 0), slice(0, 0), slice(0, 0), slice(0, 0)
    train_end = max(1, int(total * train_frac))
    val_end = max(train_end + 1, int(total * (train_frac + val_frac)))
    val_end = min(total - 1, val_end)
    oos_end = max(val_end + 1, int(total * (train_frac + val_frac + oos_frac)))
    oos_end = min(total - 1, oos_end) if val_end + 1 <= total - 1 else total
    oos_end = max(val_end, min(oos_end, total))
    return (
        slice(0, train_end),
        slice(train_end, val_end),
        slice(val_end, oos_end),
        slice(oos_end, total),
    )


def _safe_std(values: np.ndarray) -> float:
    return _research_metrics.safe_std(values)


def _expanding_std(values: np.ndarray, *, min_periods: int = 32) -> np.ndarray:
    """Per-bar expanding standard deviation using only data through each bar.

    Bars before ``min_periods`` observations return 0.0 so callers gating on
    ``sigma > eps`` stay flat through warmup.  This replaces full-sample
    ``_safe_std`` normalisation in signal paths, where a single whole-history
    scalar leaks future volatility into every bar's score.
    """
    arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    n = arr.size
    if n == 0:
        return np.asarray([], dtype=float)
    counts = np.arange(1, n + 1, dtype=float)
    csum = np.cumsum(arr)
    csum2 = np.cumsum(arr * arr)
    mean = csum / counts
    var = np.maximum(csum2 / counts - mean * mean, 0.0)
    std = np.sqrt(var)
    warmup = min(max(int(min_periods), 1), n)
    std[: warmup - 1] = 0.0
    return std


def _safe_mean(values: np.ndarray) -> float:
    return _research_metrics.safe_mean(values)


def _compute_metrics(
    returns: np.ndarray,
    *,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int,
    num_trials: int,
    metric_config: Any | None = None,
    timestamps: np.ndarray | None = None,
    hac_inference: bool = _RESEARCH_HAC_INFERENCE_DEFAULT,
) -> dict[str, float]:
    if returns.size == 0:
        return _compute_metric_summary(
            _empty_compute_metric_payload(),
            resolved_rf=None,
        )

    resolved_rf = resolve_risk_free_config(
        metric_config or get_default_runtime_config().backtest,
        periods_per_year=periods_per_year,
        timestamps=timestamps,
        size=int(returns.size),
    )
    metric_payload = _resolve_compute_metric_payload(
        returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        num_trials=num_trials,
        resolved_rf=resolved_rf,
        hac_inference=hac_inference,
    )
    return _compute_metric_summary(
        metric_payload,
        resolved_rf=resolved_rf,
    )


@dataclass(frozen=True, slots=True)
class _ComputedMetricPayload:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    turnover: float
    trade_count: float
    win_rate: float
    avg_trade: float
    exposure: float
    volatility: float
    stability: float
    rolling_sharpe_min: float
    worst_month: float
    benchmark_corr: float
    deflated_sharpe: float
    pbo: float
    active_fold_ratio: float
    inactive_fold_count: float
    failed_fold_ratio: float
    spa_pvalue: float


def _empty_compute_metric_payload() -> _ComputedMetricPayload:
    payload = _research_metrics.empty_compute_metric_payload()
    return _ComputedMetricPayload(**{field: getattr(payload, field) for field in payload.__slots__})


def _resolve_compute_metric_payload(
    returns: np.ndarray,
    *,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int,
    num_trials: int,
    resolved_rf: Any,
    hac_inference: bool = _RESEARCH_HAC_INFERENCE_DEFAULT,
) -> _ComputedMetricPayload:
    payload = _research_metrics.resolve_compute_metric_payload(
        returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark_returns,
        periods_per_year=periods_per_year,
        num_trials=num_trials,
        resolved_rf=resolved_rf,
        hac_inference=hac_inference,
    )
    return _ComputedMetricPayload(**{field: getattr(payload, field) for field in payload.__slots__})


def _compute_metric_summary(
    metric_payload: _ComputedMetricPayload,
    *,
    resolved_rf: Any | None,
) -> dict[str, float]:
    return _research_metrics.compute_metric_summary(
        _research_metrics.ComputedMetricPayload(
            **{field: getattr(metric_payload, field) for field in metric_payload.__slots__}
        ),
        resolved_rf=resolved_rf,
    )


def _hurdle_fields(
    train: dict[str, float],
    val: dict[str, float],
    oos: dict[str, float],
    *,
    scoring_config: Mapping[str, Any] | None = None,
    oos_sharpe_min: float | None = None,
    max_pbo: float | None = None,
    max_turnover: float | None = None,
    max_drawdown: float | None = None,
    bind_stage: str = "oos",
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    config = _resolve_hurdle_config(
        scoring_config=scoring_config,
        oos_sharpe_min=oos_sharpe_min,
        max_pbo=max_pbo,
        max_turnover=max_turnover,
        max_drawdown=max_drawdown,
    )
    fields = {
        "train": _pack_hurdle_stage(train, stage="train", config=config),
        "val": _pack_hurdle_stage(val, stage="val", config=config),
        "oos": _pack_hurdle_stage(oos, stage="oos", config=config),
    }
    # Under the lockbox discipline the binding pass/reject runs on validation so
    # the never-touched lockbox (reported in the ``oos`` slot) is not consumed by
    # selection. Default ``bind_stage="oos"`` keeps today's oos-binding behavior.
    binding = val if str(bind_stage).strip().lower() == "val" else oos
    hard_reject_reasons = _hurdle_hard_reject_reasons(
        binding, config=config, scoring_config=scoring_config
    )
    cost_ok = bool(
        float(binding.get("sharpe", 0.0)) >= config.oos_sharpe_min
        and float(binding.get("pbo", 1.0)) <= config.max_pbo
    )
    return fields, bool(cost_ok and not hard_reject_reasons), hard_reject_reasons


@dataclass(frozen=True, slots=True)
class _ResolvedHurdleConfig:
    weights: dict[str, float]
    in_sample_sharpe_min: float
    oos_sharpe_min: float
    max_pbo: float
    max_turnover: float
    max_drawdown: float
    min_trade_count: float


def _resolve_hurdle_config(
    *,
    scoring_config: Mapping[str, Any] | None,
    oos_sharpe_min: float | None,
    max_pbo: float | None,
    max_turnover: float | None,
    max_drawdown: float | None,
) -> _ResolvedHurdleConfig:
    cfg = _resolve_score_config(scoring_config)
    thresholds = dict(cfg["reject_thresholds"])
    return _ResolvedHurdleConfig(
        weights={key: float(value) for key, value in dict(cfg["hurdle_score_weights"]).items()},
        in_sample_sharpe_min=float(thresholds["in_sample_sharpe_min"]),
        oos_sharpe_min=float(
            thresholds["oos_sharpe_min"] if oos_sharpe_min is None else oos_sharpe_min
        ),
        max_pbo=float(thresholds["max_pbo"] if max_pbo is None else max_pbo),
        max_turnover=float(thresholds["max_turnover"] if max_turnover is None else max_turnover),
        max_drawdown=float(thresholds["max_drawdown"] if max_drawdown is None else max_drawdown),
        min_trade_count=float(thresholds["min_trade_count"]),
    )


def _pack_hurdle_stage(
    metrics: dict[str, float],
    *,
    stage: str,
    config: _ResolvedHurdleConfig,
) -> dict[str, Any]:
    score = (
        (config.weights["sharpe_weight"] * float(metrics.get("sharpe", 0.0)))
        + (config.weights["return_weight"] * float(metrics.get("return", 0.0)))
        + (config.weights["deflated_sharpe_weight"] * float(metrics.get("deflated_sharpe", 0.0)))
        - (config.weights["pbo_penalty"] * float(metrics.get("pbo", 1.0)))
        - (
            config.weights["turnover_penalty"]
            * max(0.0, float(metrics.get("turnover", 0.0)) - config.max_turnover)
        )
        - (
            config.weights["drawdown_penalty"]
            * max(0.0, float(metrics.get("mdd", 0.0)) - config.max_drawdown)
        )
        - (config.weights["spa_pvalue_penalty"] * float(metrics.get("spa_pvalue", 1.0)))
    )
    sharpe_min = config.in_sample_sharpe_min if stage != "oos" else config.oos_sharpe_min
    passed = bool(
        float(metrics.get("sharpe", 0.0)) >= sharpe_min
        and float(metrics.get("pbo", 1.0)) <= config.max_pbo
        and float(metrics.get("turnover", 0.0)) <= config.max_turnover
        and float(metrics.get("mdd", 0.0)) <= config.max_drawdown
        and float(metrics.get("trade_count", 0.0)) >= config.min_trade_count
    )
    return {
        "pass": passed,
        "score": float(score),
        "excess_return": float(metrics.get("return", 0.0)),
    }


def _hurdle_hard_reject_reasons(
    oos: dict[str, float],
    *,
    config: _ResolvedHurdleConfig,
    scoring_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hard_reject_reasons: dict[str, Any] = {}
    if float(oos.get("sharpe", 0.0)) < config.oos_sharpe_min:
        hard_reject_reasons["oos_sharpe"] = float(oos.get("sharpe", 0.0))
    if float(oos.get("pbo", 1.0)) > config.max_pbo:
        hard_reject_reasons["pbo"] = float(oos.get("pbo", 1.0))
    if float(oos.get("turnover", 0.0)) > config.max_turnover:
        hard_reject_reasons["turnover"] = float(oos.get("turnover", 0.0))
    if float(oos.get("mdd", 0.0)) > config.max_drawdown:
        hard_reject_reasons["max_drawdown"] = float(oos.get("mdd", 0.0))
    if float(oos.get("trade_count", 0.0)) < config.min_trade_count:
        hard_reject_reasons["trade_count"] = float(oos.get("trade_count", 0.0))
    # Strict-selection gate (opt-in): promote the deflated Sharpe / SPA reality
    # check from an advisory soft-score weight to a binding hard reject. The DSR
    # is already computed with the correct multiplicity count (num_trials =
    # candidate_count). Legacy default keeps DSR/SPA advisory-only, so this block
    # is skipped and the reason set is byte-identical when the flag is unset.
    if bool(
        _research_flag(
            scoring_config,
            "strict_selection_gate",
            _RESEARCH_STRICT_SELECTION_GATE_DEFAULT,
        )
    ):
        dsr_floor = float(
            _research_flag(scoring_config, "dsr_gate_floor", _RESEARCH_DSR_GATE_FLOOR_DEFAULT)
        )
        spa_ceiling = float(
            _research_flag(scoring_config, "spa_gate_ceiling", _RESEARCH_SPA_GATE_CEILING_DEFAULT)
        )
        deflated_sharpe = float(oos.get("deflated_sharpe", 0.0))
        if deflated_sharpe < dsr_floor:
            hard_reject_reasons["deflated_sharpe"] = deflated_sharpe
        spa_pvalue = float(oos.get("spa_pvalue", 1.0))
        if spa_pvalue > spa_ceiling:
            hard_reject_reasons["spa_pvalue"] = spa_pvalue
    return hard_reject_reasons


def _apply_lockbox_and_purge_masks(
    out: dict[str, np.ndarray],
    *,
    size: int,
    split: Mapping[str, Any] | None,
) -> dict[str, np.ndarray]:
    """Optionally add a never-touched lockbox mask and purge/embargo bars.

    Both behaviors are opt-in via the ``split`` mapping. With ``use_lockbox_split``
    the tail portion of the OOS window is carved into a ``lockbox`` mask (scored
    once, reported as OOS). With ``purge_embargo_bars`` the leading N bars of each
    downstream split are dropped so contiguous splits no longer abut. When neither
    flag is set the mask dict is returned unchanged (byte-identical to today).
    """
    use_lockbox = bool(_split_flag(split, "use_lockbox_split", _RESEARCH_USE_LOCKBOX_SPLIT_DEFAULT))
    purge = int(_split_flag(split, "purge_embargo_bars", _RESEARCH_PURGE_EMBARGO_BARS_DEFAULT) or 0)
    if not use_lockbox and purge <= 0:
        return out

    if use_lockbox:
        oos_mask = out.get("oos")
        oos_idx = (
            np.flatnonzero(oos_mask) if oos_mask is not None else np.asarray([], dtype=np.intp)
        )
        lockbox_mask = np.zeros(size, dtype=bool)
        if oos_idx.size == 1:
            lockbox_mask[oos_idx] = True
            out["oos"] = np.zeros(size, dtype=bool)
        elif oos_idx.size >= 2:
            half = oos_idx.size // 2  # front half stays OOS; tail carves the lockbox
            new_oos = np.zeros(size, dtype=bool)
            new_oos[oos_idx[:half]] = True
            lockbox_mask[oos_idx[half:]] = True
            out["oos"] = new_oos
        out["lockbox"] = lockbox_mask

    if purge > 0:
        for stage in ("val", "oos", "lockbox"):
            mask = out.get(stage)
            if mask is None:
                continue
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            trimmed = mask.copy()
            trimmed[idx[:purge]] = False
            out[stage] = trimmed
    return out


def _split_masks_from_datetimes(
    datetimes: np.ndarray,
    *,
    split: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    size = int(datetimes.size)
    empty = np.zeros(size, dtype=bool)
    if size <= 0:
        return {"train": empty, "val": empty, "oos": empty}

    if not isinstance(split, Mapping):
        split_train, split_val, split_oos = _split_lengths(size)
        train_mask = np.zeros(size, dtype=bool)
        val_mask = np.zeros(size, dtype=bool)
        oos_mask = np.zeros(size, dtype=bool)
        train_mask[split_train] = True
        val_mask[split_val] = True
        oos_mask[split_oos] = True
        return {"train": train_mask, "val": val_mask, "oos": oos_mask}

    resolved = _resolve_split_config(
        split, strategy_timeframe=str(split.get("strategy_timeframe") or "1m")
    )
    stage_bounds = {
        "train": (
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("train_start"))),
            _to_numpy_datetime64_ms(
                _coerce_utc_datetime(resolved.get("train_end"), end_of_day=True)
            ),
        ),
        "val": (
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("val_start"))),
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("val_end"), end_of_day=True)),
        ),
        "oos": (
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("oos_start"))),
            _to_numpy_datetime64_ms(_coerce_utc_datetime(resolved.get("oos_end"), end_of_day=True)),
        ),
    }

    ts = np.asarray(datetimes, dtype="datetime64[ms]")
    covered = np.zeros(size, dtype=bool)
    out: dict[str, np.ndarray] = {}
    for stage in ("train", "val", "oos"):
        start, end = stage_bounds[stage]
        mask = np.ones(size, dtype=bool)
        if start is not None:
            mask &= ts >= start
        if end is not None:
            mask &= ts <= end
        if start is not None and end is not None and end < start:
            mask &= False
        mask &= ~covered
        covered |= mask
        out[stage] = mask
    return _apply_lockbox_and_purge_masks(out, size=size, split=split)


def _align_series_to_timestamps(
    target_timestamps: np.ndarray,
    *,
    source_timestamps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray | None:
    target = np.asarray(target_timestamps, dtype="datetime64[ms]")
    source = np.asarray(source_timestamps, dtype="datetime64[ms]")
    arr = np.asarray(values, dtype=float)
    if target.size == 0 or source.size == 0 or arr.size == 0:
        return None
    if arr.size != source.size:
        return None
    idx = np.searchsorted(source, target)
    valid = (idx >= 0) & (idx < source.size)
    if not np.all(valid):
        return None
    if not np.all(source[idx] == target):
        return None
    return arr[idx]


def _series_to_stream(
    values: np.ndarray,
    *,
    timestamps: np.ndarray | None = None,
    offset: int = 0,
) -> list[dict[str, float | int]]:
    if timestamps is None:
        return [{"t": float(offset + idx), "v": float(value)} for idx, value in enumerate(values)]

    ts = np.asarray(timestamps, dtype="datetime64[ms]")
    out: list[dict[str, float | int]] = []
    for idx, value in enumerate(values):
        epoch_ms = int(ts[idx].astype("datetime64[ms]").astype(np.int64))
        out.append({"t": epoch_ms, "v": float(value)})
    return out


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _rolling_sample_std(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    win = max(2, int(window))
    if arr.size < win:
        return out

    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    csum = np.concatenate(([0.0], np.cumsum(clean, dtype=float)))
    csum2 = np.concatenate(([0.0], np.cumsum(clean * clean, dtype=float)))
    end = np.arange(win, clean.size + 1, dtype=int)
    start = end - win
    sums = csum[end] - csum[start]
    sums2 = csum2[end] - csum2[start]
    var = np.maximum(0.0, (sums2 - ((sums * sums) / float(win))) / float(win - 1))
    out[win - 1 :] = np.sqrt(var, dtype=float)
    return out


def _rolling_z(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if values.size < win:
        return out

    hist_len = win - 1
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    csum = np.concatenate(([0.0], np.cumsum(clean, dtype=float)))
    csum2 = np.concatenate(([0.0], np.cumsum(clean * clean, dtype=float)))
    latest_idx = np.arange(hist_len, clean.size, dtype=int)
    hist_start = latest_idx - hist_len
    hist_end = latest_idx
    sums = csum[hist_end] - csum[hist_start]
    sums2 = csum2[hist_end] - csum2[hist_start]
    mean = sums / float(hist_len)
    var = np.maximum(0.0, (sums2 - ((sums * sums) / float(hist_len))) / float(hist_len - 1))
    std = np.sqrt(var, dtype=float)
    latest = clean[hist_len:]
    result = np.zeros_like(latest, dtype=float)
    valid = std > 1e-12
    result[valid] = (latest[valid] - mean[valid]) / std[valid]
    out[hist_len:] = result
    return out


def _trend_efficiency_series(closes: np.ndarray, window: int = 55) -> np.ndarray:
    out = np.full(closes.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if closes.size < win + 1:
        return out
    for idx in range(win, closes.size):
        span = closes[idx - win : idx + 1]
        net = abs(float(span[-1] - span[0]))
        path = float(np.sum(np.abs(np.diff(span))))
        out[idx] = 0.0 if path <= 1e-12 else min(1.0, max(0.0, net / path))
    return out


def _vol_ratio_series(closes: np.ndarray, fast: int = 8, slow: int = 55) -> np.ndarray:
    out = np.full(closes.shape, np.nan, dtype=float)
    rets = np.diff(np.log(np.clip(closes, 1e-12, np.inf)), prepend=np.log(max(1e-12, closes[0])))
    f = max(4, int(fast))
    s = max(f + 2, int(slow))
    if rets.size < s:
        return out
    fast_std = _rolling_sample_std(rets, f)
    slow_std = _rolling_sample_std(rets, s)
    valid = np.isfinite(slow_std) & (slow_std > 1e-12)
    out[valid] = np.divide(fast_std[valid], slow_std[valid], dtype=float)
    out[np.isfinite(slow_std) & (slow_std <= 1e-12)] = 0.0
    return out


def _rolling_volatility_series(closes: np.ndarray, window: int) -> np.ndarray:
    win = max(8, int(window))
    if closes.size < win:
        return np.full(closes.shape, np.nan, dtype=float)
    log_close = np.log(np.clip(closes, 1e-12, np.inf))
    rets = np.diff(log_close, prepend=log_close[0])
    return _rolling_sample_std(rets, win)


def _composite_trend_signal_strength(
    *,
    sigma: float,
    crowding_score: float | None,
    crowding_reduce_threshold: float,
    risk_target_vol: float,
    max_signal_strength: float,
) -> float:
    sigma_floor = max(1e-6, float(sigma))
    strength = float(risk_target_vol) / sigma_floor
    strength = min(float(max_signal_strength), max(0.10, strength))
    if crowding_score is not None and abs(float(crowding_score)) >= float(
        crowding_reduce_threshold
    ):
        strength *= 0.5
    return float(max(0.05, strength))


def _composite_trend_position_series(
    *,
    close: np.ndarray,
    score: np.ndarray,
    gate: np.ndarray,
    long_gate: np.ndarray | None = None,
    short_gate: np.ndarray | None = None,
    crowding: np.ndarray | None,
    config: _CompositeTrendStrategyConfig,
) -> np.ndarray:
    close_arr = np.asarray(close, dtype=float)
    score_arr = np.asarray(score, dtype=float)
    gate_arr = np.asarray(gate, dtype=bool)
    long_gate_arr = gate_arr if long_gate is None else np.asarray(long_gate, dtype=bool)
    short_gate_arr = gate_arr if short_gate is None else np.asarray(short_gate, dtype=bool)
    crowd_arr = (
        np.full(close_arr.shape, np.nan, dtype=float)
        if crowding is None
        else np.asarray(crowding, dtype=float)
    )
    sigma_arr = _rolling_volatility_series(close_arr, config.vol_window)
    position = np.zeros(close_arr.shape, dtype=float)
    mode = 0
    bars_held = 0

    for idx in range(close_arr.size):
        step = _composite_trend_step_input(
            idx=idx,
            close_arr=close_arr,
            score_arr=score_arr,
            gate_arr=gate_arr,
            long_gate_arr=long_gate_arr,
            short_gate_arr=short_gate_arr,
            crowd_arr=crowd_arr,
            sigma_arr=sigma_arr,
            config=config,
        )
        mode, bars_held, position[idx] = _composite_trend_step(
            mode=mode,
            bars_held=bars_held,
            step=step,
            config=config,
        )

    return position


def _composite_trend_step_input(
    *,
    idx: int,
    close_arr: np.ndarray,
    score_arr: np.ndarray,
    gate_arr: np.ndarray,
    long_gate_arr: np.ndarray,
    short_gate_arr: np.ndarray,
    crowd_arr: np.ndarray,
    sigma_arr: np.ndarray,
    config: _CompositeTrendStrategyConfig,
) -> _CompositeTrendStepInput:
    close_i = float(close_arr[idx])
    score_i = float(score_arr[idx]) if np.isfinite(score_arr[idx]) else float("nan")
    gate_i = bool(gate_arr[idx]) if idx < gate_arr.size else False
    long_gate_i = bool(long_gate_arr[idx]) if idx < long_gate_arr.size else gate_i
    short_gate_i = bool(short_gate_arr[idx]) if idx < short_gate_arr.size else gate_i
    crowd_i = crowd_arr[idx] if idx < crowd_arr.size and np.isfinite(crowd_arr[idx]) else None
    strength = _composite_trend_signal_strength(
        sigma=float(sigma_arr[idx]) if np.isfinite(sigma_arr[idx]) else 0.0,
        crowding_score=crowd_i,
        crowding_reduce_threshold=config.crowding_reduce_threshold,
        risk_target_vol=config.risk_target_vol,
        max_signal_strength=config.max_signal_strength,
    )
    blocked = crowd_i is not None and abs(float(crowd_i)) >= config.crowding_block_threshold
    return _CompositeTrendStepInput(
        close_i=close_i,
        score_i=score_i,
        long_gate_i=long_gate_i,
        short_gate_i=short_gate_i,
        strength=strength,
        blocked=blocked,
    )


def _composite_trend_step(
    *,
    mode: int,
    bars_held: int,
    step: _CompositeTrendStepInput,
    config: _CompositeTrendStrategyConfig,
) -> tuple[int, int, float]:
    if not np.isfinite(step.close_i):
        return mode, bars_held, float(mode) * step.strength

    if mode != 0:
        next_bars_held = bars_held + 1
        if _composite_trend_should_exit(
            mode=mode,
            score_i=step.score_i,
            long_gate_i=step.long_gate_i,
            short_gate_i=step.short_gate_i,
            bars_held=next_bars_held,
            config=config,
        ):
            return 0, 0, 0.0
        return mode, next_bars_held, float(mode) * step.strength

    next_mode = _composite_trend_entry_mode(
        score_i=step.score_i,
        long_gate_i=step.long_gate_i,
        short_gate_i=step.short_gate_i,
        blocked=step.blocked,
        config=config,
    )
    if next_mode == 0:
        return 0, 0, 0.0
    return next_mode, 0, float(next_mode) * step.strength


def _composite_trend_should_exit(
    *,
    mode: int,
    score_i: float,
    long_gate_i: bool,
    short_gate_i: bool,
    bars_held: int,
    config: _CompositeTrendStrategyConfig,
) -> bool:
    if mode == 1:
        return (
            (not long_gate_i)
            or (not np.isfinite(score_i))
            or (score_i <= config.exit_score_cross)
            or (bars_held >= config.max_hold_bars)
        )
    return (
        (not short_gate_i)
        or (not np.isfinite(score_i))
        or (score_i >= -config.exit_score_cross)
        or (bars_held >= config.max_hold_bars)
    )


def _composite_trend_entry_mode(
    *,
    score_i: float,
    long_gate_i: bool,
    short_gate_i: bool,
    blocked: bool,
    config: _CompositeTrendStrategyConfig,
) -> int:
    if (not long_gate_i and not short_gate_i) or not np.isfinite(score_i) or blocked:
        return 0
    if long_gate_i and score_i >= config.long_threshold:
        return 1
    if config.allow_short and short_gate_i and score_i <= -config.short_threshold:
        return -1
    return 0


def _vwap_dev_z(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    window: int = 60,
    z_window: int = 120,
) -> np.ndarray:
    n = close.size
    out = np.full(close.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if n < win:
        return out
    dev = np.full(close.shape, np.nan, dtype=float)
    typical = (high + low + close) / 3.0
    vol = np.clip(volume, 0.0, np.inf)
    for idx in range(win, n + 1):
        p = typical[idx - win : idx]
        v = vol[idx - win : idx]
        den = float(np.sum(v))
        vw = float(np.sum(p * v) / den) if den > 1e-12 else float(np.mean(p))
        if vw <= 0.0:
            dev[idx - 1] = 0.0
        else:
            dev[idx - 1] = (close[idx - 1] / vw) - 1.0
    out = _rolling_z(np.nan_to_num(dev, nan=0.0), window=max(16, int(z_window)))
    return out


def _rolling_vwap_deviation(close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    out = np.full(close.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if close.size < win:
        return out
    vol = np.clip(np.asarray(volume, dtype=float), 0.0, np.inf)
    close_arr = np.asarray(close, dtype=float)
    for idx in range(win, close_arr.size + 1):
        px = close_arr[idx - win : idx]
        vv = vol[idx - win : idx]
        den = float(np.sum(vv))
        vw = float(np.sum(px * vv) / den) if den > 1e-12 else _safe_mean(px)
        if vw <= 1e-12:
            out[idx - 1] = 0.0
        else:
            out[idx - 1] = (float(close_arr[idx - 1]) / vw) - 1.0
    return out


def _rolling_channel(
    high: np.ndarray, low: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    high_out = np.full(high.shape, np.nan, dtype=float)
    low_out = np.full(low.shape, np.nan, dtype=float)
    win = max(8, int(window))
    if high.size < win + 1:
        return high_out, low_out
    hi = np.asarray(high, dtype=float)
    lo = np.asarray(low, dtype=float)
    for idx in range(win, hi.size):
        high_out[idx] = float(np.max(hi[idx - win : idx]))
        low_out[idx] = float(np.min(lo[idx - win : idx]))
    return high_out, low_out


def _rolling_slope_series(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    win = max(2, int(window))
    if values.size < win:
        return out

    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr).astype(np.int64)
    prefix_valid = np.concatenate(([0], np.cumsum(finite)))
    clean = np.nan_to_num(arr, nan=0.0)
    prefix_y = np.concatenate(([0.0], np.cumsum(clean)))
    indices = np.arange(arr.size, dtype=float)
    prefix_xy_abs = np.concatenate(([0.0], np.cumsum(clean * indices)))

    sum_x = float(win * (win - 1) / 2.0)
    sum_x2 = float(((win - 1) * win * ((2 * win) - 1)) / 6.0)
    denom = float((win * sum_x2) - (sum_x * sum_x))
    if abs(denom) <= 1e-12:
        return out

    for end in range(win - 1, arr.size):
        start = end - win + 1
        if int(prefix_valid[end + 1] - prefix_valid[start]) != win:
            continue
        sum_y = float(prefix_y[end + 1] - prefix_y[start])
        sum_xy = float(prefix_xy_abs[end + 1] - prefix_xy_abs[start]) - (float(start) * sum_y)
        numer = float((win * sum_xy) - (sum_x * sum_y))
        out[end] = numer / denom
    return out


def _composite_momentum_series(
    values: np.ndarray,
    *,
    windows: Sequence[int] = (8, 21, 55),
    weights: Sequence[float] = (0.5, 0.3, 0.2),
) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return out

    score = np.zeros(arr.shape, dtype=float)
    total_weight = np.zeros(arr.shape, dtype=float)
    for win, weight in zip(windows, weights, strict=True):
        window_i = max(1, int(win))
        if arr.size <= window_i:
            continue
        latest = arr[window_i:]
        base = arr[:-window_i]
        valid = np.isfinite(latest) & np.isfinite(base) & (latest > 0.0) & (base > 0.0)
        if not np.any(valid):
            continue
        contribution = np.zeros(latest.shape, dtype=float)
        contribution[valid] = np.log(latest[valid] / base[valid])
        score[window_i:] += float(weight) * contribution
        total_weight[window_i:] += abs(float(weight)) * valid.astype(float)

    valid_out = total_weight > 1e-12
    out[valid_out] = score[valid_out] / total_weight[valid_out]
    return out


@dataclass(frozen=True, slots=True)
class _VwapReversionConfig:
    window: int
    entry_dev: float
    exit_dev: float
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _VwapReversionStepInput:
    close_i: float
    deviation_i: float


def _resolve_vwap_reversion_config(params: Mapping[str, Any]) -> _VwapReversionConfig:
    return _VwapReversionConfig(
        window=max(8, int(params.get("window", 64))),
        entry_dev=float(params.get("entry_dev", 0.02)),
        exit_dev=max(0.0, float(params.get("exit_dev", 0.005))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.03)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _vwap_reversion_position_series(
    *,
    close: np.ndarray,
    volume: np.ndarray,
    config: _VwapReversionConfig,
) -> np.ndarray:
    deviation = np.nan_to_num(
        _rolling_vwap_deviation(close, volume, config.window),
        nan=0.0,
    )
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None

    for idx in range(close.size):
        step = _vwap_reversion_step_input(
            idx=idx,
            close=close,
            deviation=deviation,
        )
        mode, entry_price, position[idx] = _vwap_reversion_step(
            mode=mode,
            entry_price=entry_price,
            step=step,
            config=config,
        )

    return position


def _vwap_reversion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    deviation: np.ndarray,
) -> _VwapReversionStepInput:
    return _VwapReversionStepInput(
        close_i=float(close[idx]),
        deviation_i=float(deviation[idx]),
    )


def _vwap_reversion_step(
    *,
    mode: int,
    entry_price: float | None,
    step: _VwapReversionStepInput,
    config: _VwapReversionConfig,
) -> tuple[int, float | None, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, 0.0

    if mode == 1 and entry_price is not None:
        stop_hit = step.close_i <= entry_price * (1.0 - config.stop_loss_pct)
        revert_hit = step.deviation_i >= -config.exit_dev
        if stop_hit or revert_hit:
            return 0, None, 0.0
        return 1, entry_price, 1.0

    if mode == -1 and entry_price is not None:
        stop_hit = step.close_i >= entry_price * (1.0 + config.stop_loss_pct)
        revert_hit = step.deviation_i <= config.exit_dev
        if stop_hit or revert_hit:
            return 0, None, 0.0
        return -1, entry_price, -1.0

    if mode == 0:
        if step.deviation_i <= -config.entry_dev:
            return 1, step.close_i, 1.0
        if config.allow_short and step.deviation_i >= config.entry_dev:
            return -1, step.close_i, -1.0
    return mode, entry_price, 0.0


def _apply_vwap_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_vwap_reversion_config(params)
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        volume = np.asarray(aligned[f"{symbol}:volume"], dtype=float)
        exposures[s_idx] = _vwap_reversion_position_series(
            close=close,
            volume=volume,
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _RollingBreakoutConfig:
    lookback_bars: int
    breakout_buffer: float
    atr_window: int
    atr_stop_multiplier: float
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _RollingBreakoutStepInput:
    close_i: float
    stop_pct: float
    upper: float | None
    lower: float | None


def _resolve_rolling_breakout_config(params: Mapping[str, Any]) -> _RollingBreakoutConfig:
    return _RollingBreakoutConfig(
        lookback_bars=max(8, int(params.get("lookback_bars", 48))),
        breakout_buffer=float(params.get("breakout_buffer", 0.0)),
        atr_window=max(4, int(params.get("atr_window", 14))),
        atr_stop_multiplier=float(params.get("atr_stop_multiplier", 2.5)),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.03)),
        allow_short=bool(params.get("allow_short", False)),
    )


def _rolling_breakout_atr_pct(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> np.ndarray:
    atr_pct = np.full(close.shape, np.nan, dtype=float)
    if close.size == 0:
        return atr_pct
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    for idx in range(window, close.size + 1):
        atr_pct[idx - 1] = _safe_mean(
            tr[idx - window : idx] / np.clip(close[idx - window : idx], 1e-12, np.inf)
        )
    return atr_pct


def _rolling_breakout_position_series(
    *,
    close: np.ndarray,
    channel_high: np.ndarray,
    channel_low: np.ndarray,
    atr_pct: np.ndarray,
    config: _RollingBreakoutConfig,
) -> np.ndarray:
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None

    for idx in range(close.size):
        step = _rolling_breakout_step_input(
            idx=idx,
            close=close,
            channel_high=channel_high,
            channel_low=channel_low,
            atr_pct=atr_pct,
            config=config,
        )
        mode, entry_price, position[idx] = _rolling_breakout_step(
            mode=mode,
            entry_price=entry_price,
            step=step,
            config=config,
        )

    return position


def _rolling_breakout_step_input(
    *,
    idx: int,
    close: np.ndarray,
    channel_high: np.ndarray,
    channel_low: np.ndarray,
    atr_pct: np.ndarray,
    config: _RollingBreakoutConfig,
) -> _RollingBreakoutStepInput:
    stop_pct = max(
        config.stop_loss_pct,
        config.atr_stop_multiplier * float(atr_pct[idx])
        if np.isfinite(atr_pct[idx])
        else config.stop_loss_pct,
    )
    return _RollingBreakoutStepInput(
        close_i=float(close[idx]),
        stop_pct=stop_pct,
        upper=float(channel_high[idx]) if np.isfinite(channel_high[idx]) else None,
        lower=float(channel_low[idx]) if np.isfinite(channel_low[idx]) else None,
    )


def _rolling_breakout_step(
    *,
    mode: int,
    entry_price: float | None,
    step: _RollingBreakoutStepInput,
    config: _RollingBreakoutConfig,
) -> tuple[int, float | None, float]:
    if mode == 1 and entry_price is not None:
        stop_hit = step.close_i <= entry_price * (1.0 - step.stop_pct)
        exit_hit = step.lower is not None and step.close_i < step.lower
        if stop_hit or exit_hit:
            return 0, None, 0.0
        return 1, entry_price, 1.0

    if mode == -1 and entry_price is not None:
        stop_hit = step.close_i >= entry_price * (1.0 + step.stop_pct)
        exit_hit = step.upper is not None and step.close_i > step.upper
        if stop_hit or exit_hit:
            return 0, None, 0.0
        return -1, entry_price, -1.0

    if mode == 0 and step.upper is not None and step.lower is not None:
        if step.close_i >= step.upper * (1.0 + config.breakout_buffer):
            return 1, step.close_i, 1.0
        if config.allow_short and step.close_i <= step.lower * (1.0 - config.breakout_buffer):
            return -1, step.close_i, -1.0
    return mode, entry_price, 0.0


def _apply_rolling_breakout_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_rolling_breakout_config(params)
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        high = np.asarray(aligned[f"{symbol}:high"], dtype=float)
        low = np.asarray(aligned[f"{symbol}:low"], dtype=float)
        channel_high, channel_low = _rolling_channel(high, low, config.lookback_bars)
        atr_pct = _rolling_breakout_atr_pct(
            close=close,
            high=high,
            low=low,
            window=config.atr_window,
        )
        exposures[s_idx] = _rolling_breakout_position_series(
            close=close,
            channel_high=channel_high,
            channel_low=channel_low,
            atr_pct=atr_pct,
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _RegimeBreakoutCandidateConfig:
    lookback_window: int
    slope_window: int
    volatility_fast_window: int
    volatility_slow_window: int
    range_entry_threshold: float
    slope_entry_threshold: float
    momentum_floor: float
    max_volatility_ratio: float
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _RegimeBreakoutStepInput:
    close_i: float
    upper: float | None
    lower: float | None
    slope: float | None
    momentum: float | None
    vol_ok: bool
    range_pos: float | None


def _resolve_regime_breakout_candidate_config(
    params: Mapping[str, Any],
) -> _RegimeBreakoutCandidateConfig:
    volatility_fast_window = max(4, int(params.get("volatility_fast_window", 12)))
    return _RegimeBreakoutCandidateConfig(
        lookback_window=max(8, int(params.get("lookback_window", 48))),
        slope_window=max(4, int(params.get("slope_window", 21))),
        volatility_fast_window=volatility_fast_window,
        volatility_slow_window=max(
            volatility_fast_window + 2,
            int(params.get("volatility_slow_window", 48)),
        ),
        range_entry_threshold=float(params.get("range_entry_threshold", 0.70)),
        slope_entry_threshold=float(params.get("slope_entry_threshold", 0.0)),
        momentum_floor=float(params.get("momentum_floor", 0.0)),
        max_volatility_ratio=float(params.get("max_volatility_ratio", 1.8)),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.03)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _regime_breakout_candidate_position_series(
    *,
    close: np.ndarray,
    channel_high: np.ndarray,
    channel_low: np.ndarray,
    vol_ratio: np.ndarray,
    slope_series: np.ndarray,
    momentum_series: np.ndarray,
    config: _RegimeBreakoutCandidateConfig,
) -> np.ndarray:
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None

    for idx in range(close.size):
        step = _regime_breakout_step_input(
            idx=idx,
            close=close,
            channel_high=channel_high,
            channel_low=channel_low,
            vol_ratio=vol_ratio,
            slope_series=slope_series,
            momentum_series=momentum_series,
            config=config,
        )
        mode, entry_price, position[idx] = _regime_breakout_step(
            mode=mode,
            entry_price=entry_price,
            step=step,
            config=config,
        )

    return position


def _regime_breakout_step_input(
    *,
    idx: int,
    close: np.ndarray,
    channel_high: np.ndarray,
    channel_low: np.ndarray,
    vol_ratio: np.ndarray,
    slope_series: np.ndarray,
    momentum_series: np.ndarray,
    config: _RegimeBreakoutCandidateConfig,
) -> _RegimeBreakoutStepInput:
    close_i = float(close[idx])
    upper = float(channel_high[idx]) if np.isfinite(channel_high[idx]) else None
    lower = float(channel_low[idx]) if np.isfinite(channel_low[idx]) else None
    slope = float(slope_series[idx]) if np.isfinite(slope_series[idx]) else None
    momentum = float(momentum_series[idx]) if np.isfinite(momentum_series[idx]) else None
    range_pos = None
    if upper is not None and lower is not None and upper > lower:
        range_pos = (close_i - lower) / max(upper - lower, 1e-12)
    return _RegimeBreakoutStepInput(
        close_i=close_i,
        upper=upper,
        lower=lower,
        slope=slope,
        momentum=momentum,
        vol_ok=float(vol_ratio[idx]) <= config.max_volatility_ratio,
        range_pos=range_pos,
    )


def _regime_breakout_step(
    *,
    mode: int,
    entry_price: float | None,
    step: _RegimeBreakoutStepInput,
    config: _RegimeBreakoutCandidateConfig,
) -> tuple[int, float | None, float]:
    if (
        step.upper is None
        or step.lower is None
        or step.upper <= step.lower
        or step.slope is None
        or step.momentum is None
        or step.range_pos is None
    ):
        return mode, entry_price, 0.0

    if mode == 1 and entry_price is not None:
        stop_hit = step.close_i <= entry_price * (1.0 - config.stop_loss_pct)
        exit_hit = step.slope < 0.0 or step.range_pos < 0.50
        if stop_hit or exit_hit:
            return 0, None, 0.0
        return 1, entry_price, 1.0

    if mode == -1 and entry_price is not None:
        stop_hit = step.close_i >= entry_price * (1.0 + config.stop_loss_pct)
        exit_hit = step.slope > 0.0 or step.range_pos > 0.50
        if stop_hit or exit_hit:
            return 0, None, 0.0
        return -1, entry_price, -1.0

    if mode == 0 and step.vol_ok:
        if (
            step.range_pos >= config.range_entry_threshold
            and step.slope >= config.slope_entry_threshold
            and step.momentum >= config.momentum_floor
        ):
            return 1, step.close_i, 1.0
        if (
            config.allow_short
            and step.range_pos <= (1.0 - config.range_entry_threshold)
            and step.slope <= -config.slope_entry_threshold
            and step.momentum <= -config.momentum_floor
        ):
            return -1, step.close_i, -1.0
    return mode, entry_price, 0.0


def _apply_regime_breakout_candidate_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_regime_breakout_candidate_config(params)
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        high = np.asarray(aligned[f"{symbol}:high"], dtype=float)
        low = np.asarray(aligned[f"{symbol}:low"], dtype=float)
        channel_high, channel_low = _rolling_channel(high, low, config.lookback_window)
        vol_ratio = np.nan_to_num(
            _vol_ratio_series(
                close,
                config.volatility_fast_window,
                config.volatility_slow_window,
            ),
            nan=np.inf,
        )
        slope_series = _rolling_slope_series(close, config.slope_window)
        momentum_series = _composite_momentum_series(close)
        exposures[s_idx] = _regime_breakout_candidate_position_series(
            close=close,
            channel_high=channel_high,
            channel_low=channel_low,
            vol_ratio=vol_ratio,
            slope_series=slope_series,
            momentum_series=momentum_series,
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _BasisSnapbackReversionConfig:
    window: int
    entry_z: float
    exit_z: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _BasisSnapbackStepInput:
    close_i: float
    z_i: float


def _resolve_basis_snapback_reversion_config(
    params: Mapping[str, Any],
) -> _BasisSnapbackReversionConfig:
    return _BasisSnapbackReversionConfig(
        window=int(params.get("window", 96)),
        entry_z=float(params.get("entry_z", 1.8)),
        exit_z=max(0.0, float(params.get("exit_z", 0.4))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 12))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _basis_snapback_reversion_position_series(
    *,
    close: np.ndarray,
    basis_z: np.ndarray,
    config: _BasisSnapbackReversionConfig,
) -> np.ndarray:
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    bars_held = 0

    for idx in range(close.size):
        step = _basis_snapback_step_input(
            idx=idx,
            close=close,
            basis_z=basis_z,
        )
        mode, entry_price, bars_held, position[idx] = _basis_snapback_step(
            mode=mode,
            entry_price=entry_price,
            bars_held=bars_held,
            step=step,
            config=config,
        )

    return position


def _basis_snapback_step_input(
    *,
    idx: int,
    close: np.ndarray,
    basis_z: np.ndarray,
) -> _BasisSnapbackStepInput:
    return _BasisSnapbackStepInput(
        close_i=float(close[idx]),
        z_i=float(basis_z[idx]) if np.isfinite(basis_z[idx]) else float("nan"),
    )


def _basis_snapback_step(
    *,
    mode: int,
    entry_price: float | None,
    bars_held: int,
    step: _BasisSnapbackStepInput,
    config: _BasisSnapbackReversionConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, bars_held, float(mode)

    if mode == 1:
        next_bars_held = bars_held + 1
        should_exit = (
            (np.isfinite(step.z_i) and step.z_i >= -config.exit_z)
            or (next_bars_held >= config.max_hold_bars)
            or (
                entry_price is not None
                and step.close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return 1, entry_price, next_bars_held, 1.0

    if mode == -1:
        next_bars_held = bars_held + 1
        should_exit = (
            (np.isfinite(step.z_i) and step.z_i <= config.exit_z)
            or (next_bars_held >= config.max_hold_bars)
            or (
                entry_price is not None
                and step.close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return -1, entry_price, next_bars_held, -1.0

    if not np.isfinite(step.z_i):
        return 0, entry_price, bars_held, 0.0
    if step.z_i <= -config.entry_z:
        return 1, step.close_i, 0, 1.0
    if config.allow_short and step.z_i >= config.entry_z:
        return -1, step.close_i, 0, -1.0
    return 0, entry_price, bars_held, 0.0


def _apply_basis_snapback_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_basis_snapback_reversion_config(params)
    missing_symbols: list[str] = []
    for s_idx, symbol in enumerate(symbols):
        close = aligned[f"{symbol}:close"]
        mark = aligned.get(f"{symbol}:mark_price")
        index = aligned.get(f"{symbol}:index_price")
        if mark is None or index is None:
            missing_symbols.append(symbol)
            continue
        support = _crowding_support_series(
            funding_rate=np.zeros(len(close), dtype=float),
            open_interest=np.ones(len(close), dtype=float),
            mark_price=np.asarray(mark, dtype=float),
            index_price=np.asarray(index, dtype=float),
            liquidation_long_notional=np.zeros(len(close), dtype=float),
            liquidation_short_notional=np.zeros(len(close), dtype=float),
            window=config.window,
        )
        exposures[s_idx] = _basis_snapback_reversion_position_series(
            close=np.asarray(close, dtype=float),
            basis_z=np.asarray(support["basis_z"], dtype=float),
            config=config,
        )

    if missing_symbols:
        meta["missing_support_data"] = True
        meta["missing_support_symbols"] = missing_symbols


@dataclass(frozen=True, slots=True)
class _VolOfVolExhaustionFadeConfig:
    vol_window: int
    vol_z_window: int
    return_z_window: int
    vol_entry_z: float
    return_entry_z: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _VolOfVolExhaustionStepInput:
    close_i: float
    vol_z_i: float
    return_z_i: float


def _resolve_vol_of_vol_exhaustion_fade_config(
    params: Mapping[str, Any],
) -> _VolOfVolExhaustionFadeConfig:
    return _VolOfVolExhaustionFadeConfig(
        vol_window=max(8, int(params.get("vol_window", 24))),
        vol_z_window=max(8, int(params.get("vol_z_window", 48))),
        return_z_window=max(8, int(params.get("return_z_window", 24))),
        vol_entry_z=float(params.get("vol_entry_z", 1.8)),
        return_entry_z=float(params.get("return_entry_z", 1.2)),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 8))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _vol_of_vol_exhaustion_fade_position_series(
    *,
    close: np.ndarray,
    config: _VolOfVolExhaustionFadeConfig,
) -> np.ndarray:
    realized_vol = _rolling_realized_vol(close, config.vol_window)
    vol_z = np.nan_to_num(
        _rolling_z(np.nan_to_num(realized_vol, nan=0.0), config.vol_z_window),
        nan=0.0,
    )
    return_z = np.nan_to_num(_rolling_z(close, config.return_z_window), nan=0.0)
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    hold_bars = 0

    for idx in range(close.size):
        step = _vol_of_vol_exhaustion_step_input(
            idx=idx,
            close=close,
            vol_z=vol_z,
            return_z=return_z,
        )
        mode, entry_price, hold_bars, position[idx] = _vol_of_vol_exhaustion_step(
            mode=mode,
            entry_price=entry_price,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )

    return position


def _vol_of_vol_exhaustion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    vol_z: np.ndarray,
    return_z: np.ndarray,
) -> _VolOfVolExhaustionStepInput:
    return _VolOfVolExhaustionStepInput(
        close_i=float(close[idx]),
        vol_z_i=float(vol_z[idx]),
        return_z_i=float(return_z[idx]),
    )


def _vol_of_vol_exhaustion_step(
    *,
    mode: int,
    entry_price: float | None,
    hold_bars: int,
    step: _VolOfVolExhaustionStepInput,
    config: _VolOfVolExhaustionFadeConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, hold_bars, 0.0

    if mode == 1 and entry_price is not None:
        next_hold_bars = hold_bars + 1
        stop_hit = step.close_i <= entry_price * (1.0 - config.stop_loss_pct)
        revert_hit = step.return_z_i >= 0.0
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_hit or revert_hit or timeout_hit:
            return 0, None, 0, 0.0
        return 1, entry_price, next_hold_bars, 1.0

    if mode == -1 and entry_price is not None:
        next_hold_bars = hold_bars + 1
        stop_hit = step.close_i >= entry_price * (1.0 + config.stop_loss_pct)
        revert_hit = step.return_z_i <= 0.0
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_hit or revert_hit or timeout_hit:
            return 0, None, 0, 0.0
        return -1, entry_price, next_hold_bars, -1.0

    if mode == 0 and step.vol_z_i >= config.vol_entry_z:
        if step.return_z_i <= -config.return_entry_z:
            return 1, step.close_i, 0, 1.0
        if config.allow_short and step.return_z_i >= config.return_entry_z:
            return -1, step.close_i, 0, -1.0
    return mode, entry_price, hold_bars, 0.0


def _apply_vol_of_vol_exhaustion_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_vol_of_vol_exhaustion_fade_config(params)
    for s_idx, symbol in enumerate(symbols):
        exposures[s_idx] = _vol_of_vol_exhaustion_fade_position_series(
            close=np.asarray(aligned[f"{symbol}:close"], dtype=float),
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _MultiHorizonTrendExhaustionFadeConfig:
    short_window: int
    entry_z: float
    exit_z: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _MultiHorizonTrendExhaustionStepInput:
    close_i: float
    z_i: float
    long_i: float


def _resolve_multi_horizon_trend_exhaustion_fade_config(
    params: Mapping[str, Any],
) -> _MultiHorizonTrendExhaustionFadeConfig:
    return _MultiHorizonTrendExhaustionFadeConfig(
        short_window=max(4, int(params.get("short_window", 16))),
        entry_z=float(params.get("entry_z", 1.6)),
        exit_z=max(0.0, float(params.get("exit_z", 0.3))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 10))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _multi_horizon_trend_exhaustion_fade_position_series(
    *,
    close: np.ndarray,
    config: _MultiHorizonTrendExhaustionFadeConfig,
) -> np.ndarray:
    short_z = np.nan_to_num(_rolling_z(close, config.short_window), nan=0.0)
    long_mom = _composite_momentum_series(
        close,
        windows=(8, 21, 55),
        weights=(0.5, 0.3, 0.2),
    )
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    hold_bars = 0

    for idx in range(close.size):
        step = _multi_horizon_trend_exhaustion_step_input(
            idx=idx,
            close=close,
            short_z=short_z,
            long_mom=long_mom,
        )
        mode, entry_price, hold_bars, position[idx] = _multi_horizon_trend_exhaustion_step(
            mode=mode,
            entry_price=entry_price,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )

    return position


def _multi_horizon_trend_exhaustion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    short_z: np.ndarray,
    long_mom: np.ndarray,
) -> _MultiHorizonTrendExhaustionStepInput:
    return _MultiHorizonTrendExhaustionStepInput(
        close_i=float(close[idx]),
        z_i=float(short_z[idx]),
        long_i=float(long_mom[idx]) if np.isfinite(long_mom[idx]) else 0.0,
    )


def _multi_horizon_trend_exhaustion_step(
    *,
    mode: int,
    entry_price: float | None,
    hold_bars: int,
    step: _MultiHorizonTrendExhaustionStepInput,
    config: _MultiHorizonTrendExhaustionFadeConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, hold_bars, 0.0

    if mode == 1:
        next_hold_bars = hold_bars + 1
        should_exit = (
            step.z_i >= -config.exit_z
            or step.long_i < 0.0
            or next_hold_bars >= config.max_hold_bars
            or (
                entry_price is not None
                and step.close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return 1, entry_price, next_hold_bars, 1.0

    if mode == -1:
        next_hold_bars = hold_bars + 1
        should_exit = (
            step.z_i <= config.exit_z
            or step.long_i > 0.0
            or next_hold_bars >= config.max_hold_bars
            or (
                entry_price is not None
                and step.close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return -1, entry_price, next_hold_bars, -1.0

    if step.z_i >= config.entry_z and step.long_i <= 0.0 and config.allow_short:
        return -1, step.close_i, 0, -1.0
    if step.z_i <= -config.entry_z and step.long_i >= 0.0:
        return 1, step.close_i, 0, 1.0
    return 0, entry_price, hold_bars, 0.0


def _apply_multi_horizon_trend_exhaustion_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_multi_horizon_trend_exhaustion_fade_config(params)
    for s_idx, symbol in enumerate(symbols):
        exposures[s_idx] = _multi_horizon_trend_exhaustion_fade_position_series(
            close=np.asarray(aligned[f"{symbol}:close"], dtype=float),
            config=config,
        )


@dataclass(frozen=True, slots=True)
class _BreadthThrustFailureReversalConfig:
    momentum_lookback: int
    breadth_entry: float
    breadth_exit: float
    basket_return_floor: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _BreadthThrustStepInput:
    basket_close: float
    breadth: float
    mean_ret: float


def _resolve_breadth_thrust_failure_reversal_config(
    params: Mapping[str, Any],
) -> _BreadthThrustFailureReversalConfig:
    return _BreadthThrustFailureReversalConfig(
        momentum_lookback=max(2, int(params.get("momentum_lookback", 16))),
        breadth_entry=float(params.get("breadth_entry", 0.80)),
        breadth_exit=float(params.get("breadth_exit", 0.55)),
        basket_return_floor=float(params.get("basket_return_floor", 0.003)),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 8))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _apply_breadth_thrust_failure_reversal_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_breadth_thrust_failure_reversal_config(params)
    close_map_np = {
        symbol: np.asarray(aligned[f"{symbol}:close"], dtype=float) for symbol in symbols
    }

    basket_position = _breadth_thrust_failure_reversal_position_series(
        close_map_np=close_map_np,
        config=config,
    )
    exposures[:] = basket_position
    meta["cross_sectional"] = True


def _breadth_thrust_failure_reversal_position_series(
    *,
    close_map_np: Mapping[str, np.ndarray],
    config: _BreadthThrustFailureReversalConfig,
) -> np.ndarray:
    if not close_map_np:
        return np.zeros(0, dtype=float)

    n = len(next(iter(close_map_np.values())))
    basket_position = np.zeros(n, dtype=float)

    basket_state = 0.0
    basket_entry = np.nan
    hold_bars = 0
    for idx in range(n):
        step = _breadth_thrust_step_input(
            idx=idx,
            close_map_np=close_map_np,
            config=config,
        )
        basket_state, basket_entry, hold_bars, basket_position[idx] = _breadth_thrust_step(
            idx=idx,
            basket_state=basket_state,
            basket_entry=basket_entry,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )

    return basket_position


def _breadth_thrust_step_input(
    *,
    idx: int,
    close_map_np: Mapping[str, np.ndarray],
    config: _BreadthThrustFailureReversalConfig,
) -> _BreadthThrustStepInput:
    close_map = {symbol: float(close[idx]) for symbol, close in close_map_np.items()}
    basket_close = float(np.mean(list(close_map.values())))
    if idx < config.momentum_lookback:
        return _BreadthThrustStepInput(
            basket_close=basket_close,
            breadth=float("nan"),
            mean_ret=float("nan"),
        )

    breadth_values: list[float] = []
    basket_returns: list[float] = []
    for close in close_map_np.values():
        latest = float(close[idx])
        base = float(close[idx - config.momentum_lookback])
        if not np.isfinite(latest) or not np.isfinite(base) or base <= 0.0:
            continue
        ret = (latest / base) - 1.0
        basket_returns.append(float(ret))
        breadth_values.append(1.0 if ret > 0.0 else 0.0)
    if not breadth_values:
        return _BreadthThrustStepInput(
            basket_close=basket_close,
            breadth=float("nan"),
            mean_ret=float("nan"),
        )
    return _BreadthThrustStepInput(
        basket_close=basket_close,
        breadth=float(np.mean(np.asarray(breadth_values, dtype=float))),
        mean_ret=float(np.mean(np.asarray(basket_returns, dtype=float))),
    )


def _breadth_thrust_step(
    *,
    idx: int,
    basket_state: float,
    basket_entry: float,
    hold_bars: int,
    step: _BreadthThrustStepInput,
    config: _BreadthThrustFailureReversalConfig,
) -> tuple[float, float, int, float]:
    if basket_state != 0.0 and np.isfinite(basket_entry):
        next_hold_bars = hold_bars + 1
        stop_long = basket_state > 0.0 and step.basket_close <= float(basket_entry) * (
            1.0 - config.stop_loss_pct
        )
        stop_short = basket_state < 0.0 and step.basket_close >= float(basket_entry) * (
            1.0 + config.stop_loss_pct
        )
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_long or stop_short or timeout_hit:
            basket_state = 0.0
            basket_entry = np.nan
            hold_bars = 0
        else:
            hold_bars = next_hold_bars

    if idx < config.momentum_lookback or not (
        np.isfinite(step.breadth) and np.isfinite(step.mean_ret)
    ):
        return basket_state, basket_entry, hold_bars, basket_state

    if basket_state == 0.0:
        if (
            config.allow_short
            and step.breadth >= config.breadth_entry
            and step.mean_ret <= -config.basket_return_floor
        ):
            return -1.0, step.basket_close, 0, -1.0
        if (
            step.breadth <= (1.0 - config.breadth_entry)
            and step.mean_ret >= config.basket_return_floor
        ):
            return 1.0, step.basket_close, 0, 1.0
    elif abs(step.breadth - 0.5) <= abs(config.breadth_exit - 0.5):
        return 0.0, np.nan, 0, 0.0

    return basket_state, basket_entry, hold_bars, basket_state


@dataclass(frozen=True, slots=True)
class _LagConvergenceConfig:
    lag_bars: int
    entry_threshold: float
    exit_threshold: float
    stop_threshold: float
    max_hold_bars: int


def _resolve_lag_convergence_config(params: Mapping[str, Any]) -> _LagConvergenceConfig:
    entry_threshold = float(params.get("entry_threshold", 0.015))
    return _LagConvergenceConfig(
        lag_bars=max(1, int(params.get("lag_bars", 3))),
        entry_threshold=entry_threshold,
        exit_threshold=max(0.0, float(params.get("exit_threshold", 0.004))),
        stop_threshold=max(entry_threshold + 1e-9, float(params.get("stop_threshold", 0.05))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 96))),
    )


def _lag_convergence_pair_positions(
    *,
    spread: np.ndarray,
    n: int,
    config: _LagConvergenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x_pos = np.zeros(n, dtype=float)
    y_pos = np.zeros(n, dtype=float)
    mode = 0
    bars_held = 0
    for idx in range(n):
        if idx < config.lag_bars:
            continue
        value = float(spread[idx])
        if mode == 0:
            if value <= -config.entry_threshold:
                mode = 1
                bars_held = 0
            elif value >= config.entry_threshold:
                mode = -1
                bars_held = 0
        else:
            bars_held += 1
            if (
                abs(value) <= config.exit_threshold
                or abs(value) >= config.stop_threshold
                or bars_held >= config.max_hold_bars
            ):
                mode = 0
                bars_held = 0
        if mode == 1:
            x_pos[idx] = 1.0
            y_pos[idx] = -1.0
        elif mode == -1:
            x_pos[idx] = -1.0
            y_pos[idx] = 1.0
    return x_pos, y_pos


def _apply_lag_convergence_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
) -> None:
    symbol_x, symbol_y, x_idx, y_idx = _resolve_symbol_pair(symbols, params)
    config = _resolve_lag_convergence_config(params)

    x_close = np.asarray(aligned[f"{symbol_x}:close"], dtype=float)
    y_close = np.asarray(aligned[f"{symbol_y}:close"], dtype=float)
    x_base = np.roll(x_close, config.lag_bars)
    y_base = np.roll(y_close, config.lag_bars)
    x_base[: config.lag_bars] = x_close[: config.lag_bars]
    y_base[: config.lag_bars] = y_close[: config.lag_bars]
    mom_x = (
        np.divide(
            x_close,
            np.clip(x_base, 1e-12, np.inf),
            out=np.ones_like(x_close),
        )
        - 1.0
    )
    mom_y = (
        np.divide(
            y_close,
            np.clip(y_base, 1e-12, np.inf),
            out=np.ones_like(y_close),
        )
        - 1.0
    )
    spread = np.nan_to_num(mom_x - mom_y, nan=0.0)

    x_pos, y_pos = _lag_convergence_pair_positions(
        spread=spread,
        n=n,
        config=config,
    )
    exposures[x_idx] = x_pos
    exposures[y_idx] = y_pos


@dataclass(frozen=True, slots=True)
class _PerpCrowdingCarryConfig:
    window: int
    mild_funding: float
    extreme_funding: float
    entry_threshold: float
    exit_threshold: float
    stop_loss_pct: float
    max_hold_bars: int
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _CrowdingSupportInputs:
    close: np.ndarray
    funding_rate: np.ndarray
    open_interest: np.ndarray
    liquidation_long_notional: np.ndarray
    liquidation_short_notional: np.ndarray
    mark_price: np.ndarray | None
    index_price: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _PerpCarryStepInput:
    funding_i: float
    score_i: float
    oi_delta_z_i: float
    close_i: float


def _resolve_perp_crowding_carry_config(
    params: Mapping[str, Any],
) -> _PerpCrowdingCarryConfig:
    return _PerpCrowdingCarryConfig(
        window=int(params.get("window", 96)),
        mild_funding=float(params.get("mild_funding", 0.0002)),
        extreme_funding=float(params.get("extreme_funding", 0.0012)),
        entry_threshold=float(params.get("entry_threshold", 0.30)),
        exit_threshold=float(params.get("exit_threshold", 0.10)),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        max_hold_bars=int(params.get("max_hold_bars", 72)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _resolve_crowding_support_inputs(
    *,
    aligned: Mapping[str, np.ndarray],
    symbol: str,
) -> _CrowdingSupportInputs | None:
    funding = aligned.get(f"{symbol}:funding_rate")
    open_interest = aligned.get(f"{symbol}:open_interest")
    liquidation_long = aligned.get(f"{symbol}:liquidation_long_notional")
    liquidation_short = aligned.get(f"{symbol}:liquidation_short_notional")
    close = aligned.get(f"{symbol}:close")
    if (
        funding is None
        or open_interest is None
        or liquidation_long is None
        or liquidation_short is None
        or close is None
    ):
        return None

    mark_price = aligned.get(f"{symbol}:mark_price")
    index_price = aligned.get(f"{symbol}:index_price")
    return _CrowdingSupportInputs(
        close=np.asarray(close, dtype=float),
        funding_rate=np.asarray(funding, dtype=float),
        open_interest=np.asarray(open_interest, dtype=float),
        liquidation_long_notional=np.asarray(liquidation_long, dtype=float),
        liquidation_short_notional=np.asarray(liquidation_short, dtype=float),
        mark_price=None if mark_price is None else np.asarray(mark_price, dtype=float),
        index_price=None if index_price is None else np.asarray(index_price, dtype=float),
    )


def _note_support_data_symbol(
    meta: dict[str, Any],
    *,
    symbol: str,
    values: np.ndarray,
) -> None:
    if np.any(np.isfinite(values)):
        meta.setdefault("support_data_symbols", []).append(symbol)


def _finalize_missing_support_symbols(
    meta: dict[str, Any],
    *,
    missing_symbols: Sequence[str],
) -> None:
    if missing_symbols:
        meta["missing_support_data"] = True
        meta["missing_support_symbols"] = list(missing_symbols)


def _apply_perp_crowding_carry_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_perp_crowding_carry_config(params)
    missing_symbols: list[str] = []
    for s_idx, symbol in enumerate(symbols):
        support_inputs = _resolve_crowding_support_inputs(aligned=aligned, symbol=symbol)
        if support_inputs is None:
            missing_symbols.append(symbol)
            continue
        position, support = _perp_carry_position_series(
            support_inputs=support_inputs,
            config=config,
        )
        exposures[s_idx] = position
        _note_support_data_symbol(
            meta,
            symbol=symbol,
            values=np.asarray(support["crowding_score"], dtype=float),
        )

    _finalize_missing_support_symbols(meta, missing_symbols=missing_symbols)


def _pair_spread_z(px: np.ndarray, py: np.ndarray, window: int = 96) -> np.ndarray:
    n = min(px.size, py.size)
    out = np.full(n, np.nan, dtype=float)
    x = np.log(np.clip(px[-n:], 1e-12, np.inf))
    y = np.log(np.clip(py[-n:], 1e-12, np.inf))
    win = max(16, int(window))
    if n < win + 2:
        return out
    for idx in range(win, n + 1):
        x_tail = x[idx - win : idx]
        y_tail = y[idx - win : idx]
        vx = float(np.var(x_tail))
        if vx <= 1e-12:
            beta = 1.0
        else:
            beta = float(np.cov(x_tail, y_tail)[0, 1] / vx)
        spread_tail = y_tail - (beta * x_tail)
        mean = float(np.mean(spread_tail[:-1]))
        std = _safe_std(spread_tail[:-1])
        if std <= 1e-12:
            out[idx - 1] = 0.0
        else:
            out[idx - 1] = (float(spread_tail[-1]) - mean) / std
    return out


def _beta_neutral_residual_series(
    values: np.ndarray,
    benchmark: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    arr = np.asarray(values, dtype=float)
    bench = np.asarray(benchmark, dtype=float)
    n = min(arr.size, bench.size)
    if n < 3:
        return out

    residual_price = 100.0
    win = max(8, int(window))
    for idx in range(1, n):
        start = max(0, idx - win)
        asset_tail = arr[start : idx + 1]
        bench_tail = bench[start : idx + 1]
        if asset_tail.size < 3 or bench_tail.size != asset_tail.size:
            continue
        if not (
            np.all(np.isfinite(asset_tail))
            and np.all(np.isfinite(bench_tail))
            and np.all(asset_tail > 0.0)
            and np.all(bench_tail > 0.0)
        ):
            continue
        asset_rets = (asset_tail[1:] / asset_tail[:-1]) - 1.0
        bench_rets = (bench_tail[1:] / bench_tail[:-1]) - 1.0
        mean_asset = float(np.mean(asset_rets))
        mean_bench = float(np.mean(bench_rets))
        bench_var = float(np.mean((bench_rets - mean_bench) ** 2))
        if bench_var <= 1e-12:
            beta = 0.0
        else:
            cov = float(np.mean((asset_rets - mean_asset) * (bench_rets - mean_bench)))
            beta = cov / bench_var
        residual_ret = float(asset_rets[-1]) - (beta * float(bench_rets[-1]))
        residual_price = max(1e-9, residual_price * (1.0 + residual_ret))
        out[idx] = residual_price
    return out


@dataclass(frozen=True, slots=True)
class _ResidualBasketReversionConfig:
    residual_window: int
    entry_z: float
    exit_z: float
    rebalance_bars: int
    max_longs: int
    max_shorts: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _VolatilityRegimeResidualBasketReversionConfig:
    residual: _ResidualBasketReversionConfig
    btc_vol_fast: int
    btc_vol_slow: int
    btc_vol_ratio_cap: float
    dispersion_floor: float


def _resolve_residual_basket_reversion_config(
    params: Mapping[str, Any],
    *,
    residual_window_default: int = 48,
) -> _ResidualBasketReversionConfig:
    return _ResidualBasketReversionConfig(
        residual_window=max(8, int(params.get("residual_window", residual_window_default))),
        entry_z=float(params.get("entry_z", 1.8)),
        exit_z=max(0.0, float(params.get("exit_z", 0.4))),
        rebalance_bars=max(1, int(params.get("rebalance_bars", 2))),
        max_longs=max(0, int(params.get("max_longs", 1))),
        max_shorts=max(0, int(params.get("max_shorts", 1))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _resolve_volatility_regime_residual_basket_reversion_config(
    params: Mapping[str, Any],
) -> _VolatilityRegimeResidualBasketReversionConfig:
    return _VolatilityRegimeResidualBasketReversionConfig(
        residual=_resolve_residual_basket_reversion_config(params),
        btc_vol_fast=max(4, int(params.get("btc_vol_fast", 12))),
        btc_vol_slow=max(8, int(params.get("btc_vol_slow", 60))),
        btc_vol_ratio_cap=float(params.get("btc_vol_ratio_cap", 1.10)),
        dispersion_floor=max(0.0, float(params.get("dispersion_floor", 0.002))),
    )


def _cross_sectional_return_dispersion(close_map_np: Mapping[str, np.ndarray]) -> np.ndarray:
    symbols = list(close_map_np.keys())
    if not symbols:
        return np.asarray([], dtype=float)
    rows: list[np.ndarray] = []
    for symbol in symbols:
        close = np.clip(np.asarray(close_map_np[symbol], dtype=float), 1e-12, np.inf)
        log_close = np.log(close)
        rows.append(np.diff(log_close, prepend=log_close[0]))
    stacked = np.vstack(rows)
    if stacked.shape[0] < 2:
        return np.zeros(stacked.shape[1], dtype=float)
    return np.std(stacked, axis=0, ddof=1)


def _apply_residual_basket_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
    entry_gate: np.ndarray | None = None,
    session_gated: bool = False,
) -> None:
    config = _resolve_residual_basket_reversion_config(
        params,
        residual_window_default=64 if session_gated else 48,
    )
    btc_symbol = canonical_symbol(str(params.get("btc_symbol") or "BTC/USDT"))
    if btc_symbol not in symbols:
        btc_symbol = canonical_symbol(str(symbols[0]))

    btc_close = np.asarray(aligned[f"{btc_symbol}:close"], dtype=float)
    close_map_np = {
        symbol: np.asarray(aligned[f"{symbol}:close"], dtype=float) for symbol in symbols
    }
    residual_z_map = _residual_basket_reversion_z_map(
        symbols=symbols,
        close_map_np=close_map_np,
        btc_symbol=btc_symbol,
        btc_close=btc_close,
        config=config,
    )
    position_state = np.zeros(len(symbols), dtype=float)
    entry_price = np.full(len(symbols), np.nan, dtype=float)

    for idx in range(len(aligned["datetime"])):
        close_map = {symbol: float(close[idx]) for symbol, close in close_map_np.items()}
        _apply_residual_basket_reversion_exits(
            symbols=symbols,
            idx=idx,
            close_map=close_map,
            residual_z_map=residual_z_map,
            position_state=position_state,
            entry_price=entry_price,
            config=config,
        )

        if not _residual_basket_reversion_rebalance_due(
            idx=idx,
            config=config,
            entry_gate=entry_gate,
        ):
            exposures[:, idx] = position_state
            continue

        long_set, shorts = _residual_basket_reversion_targets(
            symbols=symbols,
            btc_symbol=btc_symbol,
            residual_z_map=residual_z_map,
            idx=idx,
            config=config,
        )

        _apply_residual_basket_reversion_targets_to_state(
            symbols=symbols,
            close_map=close_map,
            long_set=long_set,
            shorts=shorts,
            position_state=position_state,
            entry_price=entry_price,
        )
        exposures[:, idx] = position_state

    meta["cross_sectional"] = True
    meta["residualized_cross_sectional"] = True
    meta["btc_symbol"] = btc_symbol
    if session_gated:
        meta["session_gated"] = True


def _residual_basket_reversion_rebalance_due(
    *,
    idx: int,
    config: _ResidualBasketReversionConfig,
    entry_gate: np.ndarray | None,
) -> bool:
    return (
        idx >= config.residual_window
        and (idx + 1) % config.rebalance_bars == 0
        and (entry_gate is None or bool(entry_gate[idx]))
    )


def _apply_residual_basket_reversion_targets_to_state(
    *,
    symbols: Sequence[str],
    close_map: Mapping[str, float],
    long_set: set[str],
    shorts: Sequence[str],
    position_state: np.ndarray,
    entry_price: np.ndarray,
) -> None:
    short_set = set(shorts)
    for s_idx, symbol in enumerate(symbols):
        next_position = 1.0 if symbol in long_set else -1.0 if symbol in short_set else 0.0
        if next_position == 0.0:
            position_state[s_idx] = 0.0
            entry_price[s_idx] = np.nan
            continue
        if position_state[s_idx] != next_position or not np.isfinite(entry_price[s_idx]):
            entry_price[s_idx] = close_map[symbol]
        position_state[s_idx] = next_position


def _residual_basket_reversion_z_map(
    *,
    symbols: Sequence[str],
    close_map_np: Mapping[str, np.ndarray],
    btc_symbol: str,
    btc_close: np.ndarray,
    config: _ResidualBasketReversionConfig,
) -> dict[str, np.ndarray]:
    residual_z_map: dict[str, np.ndarray] = {}
    for symbol in symbols:
        close = close_map_np[symbol]
        if symbol == btc_symbol:
            residual_z_map[symbol] = np.zeros(close.shape, dtype=float)
            continue
        residual_series = _beta_neutral_residual_series(
            close, btc_close, window=config.residual_window
        )
        residual_z_map[symbol] = np.nan_to_num(
            _rolling_z(residual_series, config.residual_window),
            nan=0.0,
        )
    return residual_z_map


def _apply_residual_basket_reversion_exits(
    *,
    symbols: Sequence[str],
    idx: int,
    close_map: Mapping[str, float],
    residual_z_map: Mapping[str, np.ndarray],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    config: _ResidualBasketReversionConfig,
) -> None:
    for s_idx, symbol in enumerate(symbols):
        if position_state[s_idx] == 0.0 or not np.isfinite(entry_price[s_idx]):
            continue
        close_i = close_map[symbol]
        stop_long = position_state[s_idx] > 0.0 and close_i <= float(entry_price[s_idx]) * (
            1.0 - config.stop_loss_pct
        )
        stop_short = position_state[s_idx] < 0.0 and close_i >= float(entry_price[s_idx]) * (
            1.0 + config.stop_loss_pct
        )
        z_i = float(residual_z_map[symbol][idx]) if symbol in residual_z_map else 0.0
        exit_hit = abs(z_i) <= config.exit_z
        if stop_long or stop_short or exit_hit:
            position_state[s_idx] = 0.0
            entry_price[s_idx] = np.nan


def _residual_basket_reversion_targets(
    *,
    symbols: Sequence[str],
    btc_symbol: str,
    residual_z_map: Mapping[str, np.ndarray],
    idx: int,
    config: _ResidualBasketReversionConfig,
) -> tuple[set[str], list[str]]:
    ranked = [
        (float(residual_z_map[symbol][idx]), symbol) for symbol in symbols if symbol != btc_symbol
    ]
    ranked.sort(key=lambda item: item[0])
    longs = [symbol for z_i, symbol in ranked if z_i <= -config.entry_z][: config.max_longs]
    shorts = [symbol for z_i, symbol in reversed(ranked) if z_i >= config.entry_z][
        : config.max_shorts
    ]
    if not config.allow_short:
        shorts = []
    long_set = set(longs)
    shorts = [symbol for symbol in shorts if symbol not in long_set]
    return long_set, shorts


@dataclass(frozen=True, slots=True)
class _CrossAssetLiquidationContagionFadeConfig:
    window: int
    leader_liq_z_min: float
    return_shock_pct: float
    exit_z: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _CrossAssetLiquidationStepInput:
    leader_liq: float
    ret_z: float
    close_i: float


def _resolve_cross_asset_liquidation_contagion_fade_config(
    params: Mapping[str, Any],
) -> _CrossAssetLiquidationContagionFadeConfig:
    return _CrossAssetLiquidationContagionFadeConfig(
        window=max(8, int(params.get("window", 64))),
        leader_liq_z_min=float(params.get("leader_liq_z_min", 1.2)),
        return_shock_pct=float(params.get("return_shock_pct", 0.006)),
        exit_z=max(0.0, float(params.get("exit_z", 0.3))),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 12))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _apply_cross_asset_liquidation_contagion_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_cross_asset_liquidation_contagion_fade_config(params)

    liq_z_map: dict[str, np.ndarray] = {}
    return_z_map: dict[str, np.ndarray] = {}
    close_map_np: dict[str, np.ndarray] = {}
    valid_symbols: list[str] = []
    for symbol in symbols:
        liq_long = aligned.get(f"{symbol}:liquidation_long_notional")
        liq_short = aligned.get(f"{symbol}:liquidation_short_notional")
        close = aligned.get(f"{symbol}:close")
        if liq_long is None or liq_short is None or close is None:
            continue
        support = _crowding_support_series(
            funding_rate=np.zeros(len(close), dtype=float),
            open_interest=np.ones(len(close), dtype=float),
            liquidation_long_notional=np.asarray(liq_long, dtype=float),
            liquidation_short_notional=np.asarray(liq_short, dtype=float),
            window=config.window,
        )
        liq_z_map[symbol] = np.asarray(support["liquidation_imbalance_z"], dtype=float)
        close_arr = np.asarray(close, dtype=float)
        close_map_np[symbol] = close_arr
        prev_close = np.r_[close_arr[0], close_arr[:-1]]
        returns = (
            np.divide(
                close_arr,
                np.clip(prev_close, 1e-12, np.inf),
                out=np.ones_like(close_arr),
                where=np.isfinite(prev_close),
            )
            - 1.0
        )
        return_z_map[symbol] = np.nan_to_num(
            _rolling_z(np.nan_to_num(returns, nan=0.0), config.window), nan=0.0
        )
        valid_symbols.append(symbol)

    for s_idx, symbol in enumerate(symbols):
        if symbol not in valid_symbols:
            continue
        exposures[s_idx] = _cross_asset_liquidation_contagion_position_series(
            symbol=symbol,
            close_arr=close_map_np[symbol],
            valid_symbols=valid_symbols,
            liq_z_map=liq_z_map,
            return_z_map=return_z_map,
            config=config,
        )


def _cross_asset_liquidation_contagion_position_series(
    *,
    symbol: str,
    close_arr: np.ndarray,
    valid_symbols: Sequence[str],
    liq_z_map: Mapping[str, np.ndarray],
    return_z_map: Mapping[str, np.ndarray],
    config: _CrossAssetLiquidationContagionFadeConfig,
) -> np.ndarray:
    pos = np.zeros(close_arr.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    hold_bars = 0
    for idx in range(close_arr.size):
        step = _cross_asset_liquidation_step_input(
            idx=idx,
            symbol=symbol,
            valid_symbols=valid_symbols,
            liq_z_map=liq_z_map,
            return_z_map=return_z_map,
            close_arr=close_arr,
        )
        mode, entry_price, hold_bars, pos[idx] = _cross_asset_liquidation_step(
            mode=mode,
            entry_price=entry_price,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )
    return pos


def _cross_asset_liquidation_step_input(
    *,
    idx: int,
    symbol: str,
    valid_symbols: Sequence[str],
    liq_z_map: Mapping[str, np.ndarray],
    return_z_map: Mapping[str, np.ndarray],
    close_arr: np.ndarray,
) -> _CrossAssetLiquidationStepInput:
    leader_vals = [
        float(liq_z_map[leader][idx])
        for leader in valid_symbols
        if leader != symbol and np.isfinite(liq_z_map[leader][idx])
    ]
    leader_liq = float(np.mean(np.asarray(leader_vals, dtype=float))) if leader_vals else np.nan
    ret_z = float(return_z_map[symbol][idx]) if np.isfinite(return_z_map[symbol][idx]) else np.nan
    return _CrossAssetLiquidationStepInput(
        leader_liq=leader_liq,
        ret_z=ret_z,
        close_i=float(close_arr[idx]),
    )


def _cross_asset_liquidation_step(
    *,
    mode: int,
    entry_price: float | None,
    hold_bars: int,
    step: _CrossAssetLiquidationStepInput,
    config: _CrossAssetLiquidationContagionFadeConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, hold_bars, 0.0

    if mode == 1:
        next_hold_bars = hold_bars + 1
        should_exit = (
            (np.isfinite(step.ret_z) and step.ret_z >= -config.exit_z)
            or (next_hold_bars >= config.max_hold_bars)
            or (
                entry_price is not None
                and step.close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return 1, entry_price, next_hold_bars, 1.0

    if mode == -1:
        next_hold_bars = hold_bars + 1
        should_exit = (
            (np.isfinite(step.ret_z) and step.ret_z <= config.exit_z)
            or (next_hold_bars >= config.max_hold_bars)
            or (
                entry_price is not None
                and step.close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
            )
        )
        if should_exit:
            return 0, None, 0, 0.0
        return -1, entry_price, next_hold_bars, -1.0

    if not (np.isfinite(step.leader_liq) and np.isfinite(step.ret_z)):
        return 0, entry_price, hold_bars, 0.0
    if (
        step.leader_liq >= config.leader_liq_z_min
        and step.ret_z >= config.return_shock_pct
        and config.allow_short
    ):
        return -1, step.close_i, 0, -1.0
    if step.leader_liq <= -config.leader_liq_z_min and step.ret_z <= -config.return_shock_pct:
        return 1, step.close_i, 0, 1.0
    return 0, entry_price, hold_bars, 0.0


def _minute_of_day(values: np.ndarray) -> np.ndarray:
    datetimes = np.asarray(values, dtype="datetime64[m]")
    return np.mod(datetimes.astype("int64"), 1440)


def _rolling_realized_vol(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size < 3:
        return out
    returns = np.zeros(arr.shape, dtype=float)
    valid = np.isfinite(arr[1:]) & np.isfinite(arr[:-1]) & (arr[1:] > 0.0) & (arr[:-1] > 0.0)
    returns[1:][valid] = np.log(arr[1:][valid] / arr[:-1][valid])
    win = max(4, int(window))
    for idx in range(win, arr.size + 1):
        tail = returns[idx - win + 1 : idx]
        if tail.size < win:
            continue
        out[idx - 1] = _safe_std(tail)
    return out


def _align_bundles(
    bundles: Sequence[SeriesBundle],
    *,
    feature_cache: Mapping[str, pl.DataFrame] | None = None,
) -> dict[str, np.ndarray] | None:
    if not bundles:
        return None
    common_datetime = _common_bundle_datetime(bundles)
    if common_datetime is None:
        return None

    aligned: dict[str, np.ndarray] = {
        "datetime": common_datetime,
    }
    for bundle in bundles:
        prefix = bundle.symbol
        indices = _aligned_bundle_indices(bundle, common_datetime)
        if indices is None:
            return None
        aligned[f"{prefix}:open"] = bundle.open[indices]
        aligned[f"{prefix}:high"] = bundle.high[indices]
        aligned[f"{prefix}:low"] = bundle.low[indices]
        aligned[f"{prefix}:close"] = bundle.close[indices]
        aligned[f"{prefix}:volume"] = bundle.volume[indices]
        feature_frame = None if feature_cache is None else feature_cache.get(prefix)
        _augment_aligned_bundle_features(
            aligned,
            prefix=prefix,
            common_datetime=common_datetime,
            feature_frame=feature_frame,
        )
    return aligned


def _common_bundle_datetime(
    bundles: Sequence[SeriesBundle],
) -> np.ndarray | None:
    if not bundles:
        return None
    common_datetime = np.asarray(bundles[0].datetime, dtype="datetime64[ms]")
    for bundle in bundles[1:]:
        common_datetime = np.intersect1d(
            common_datetime,
            np.asarray(bundle.datetime, dtype="datetime64[ms]"),
            assume_unique=False,
        )
        if common_datetime.size < _MIN_BARS:
            return None
    if common_datetime.size < _MIN_BARS:
        return None
    return common_datetime


def _aligned_bundle_indices(
    bundle: SeriesBundle,
    common_datetime: np.ndarray,
) -> np.ndarray | None:
    bundle_datetime = np.asarray(bundle.datetime, dtype="datetime64[ms]")
    indices = np.searchsorted(bundle_datetime, common_datetime)
    if np.any(indices >= bundle_datetime.size) or np.any(
        bundle_datetime[indices] != common_datetime
    ):
        return None
    return indices


def _aligned_feature_points(
    *,
    common_datetime: np.ndarray,
    feature_frame: pl.DataFrame,
) -> pl.DataFrame | None:
    if feature_frame.is_empty():
        return None

    target = pl.DataFrame({"datetime": pl.Series("datetime", common_datetime)})
    for field in _FEATURE_POINT_COLUMNS:
        if field not in feature_frame.columns:
            feature_frame = feature_frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(field))
    feature_points = feature_frame.select(["datetime", *_FEATURE_POINT_COLUMNS])
    target_dtype = target.schema.get("datetime")
    feature_dtype = feature_points.schema.get("datetime")
    if target_dtype is not None and feature_dtype is not None and target_dtype != feature_dtype:
        feature_points = feature_points.with_columns(pl.col("datetime").cast(target_dtype))
    return target.join_asof(
        feature_points.sort("datetime"),
        on="datetime",
        strategy="backward",
        tolerance=timedelta(milliseconds=_FEATURE_POINT_MAX_STALE_MS),
    )


def _augment_aligned_bundle_features(
    aligned: dict[str, np.ndarray],
    *,
    prefix: str,
    common_datetime: np.ndarray,
    feature_frame: pl.DataFrame | None,
) -> None:
    if feature_frame is None:
        return

    joined = _aligned_feature_points(
        common_datetime=common_datetime,
        feature_frame=feature_frame,
    )
    if joined is None:
        return

    for field in _FEATURE_POINT_COLUMNS:
        aligned[f"{prefix}:{field}"] = joined.get_column(field).to_numpy()

    support = _crowding_support_series(
        funding_rate=np.asarray(joined.get_column("funding_rate").to_numpy(), dtype=float),
        open_interest=np.asarray(joined.get_column("open_interest").to_numpy(), dtype=float),
        mark_price=np.asarray(joined.get_column("mark_price").to_numpy(), dtype=float),
        index_price=np.asarray(joined.get_column("index_price").to_numpy(), dtype=float),
        liquidation_long_notional=np.asarray(
            joined.get_column("liquidation_long_notional").to_numpy(),
            dtype=float,
        ),
        liquidation_short_notional=np.asarray(
            joined.get_column("liquidation_short_notional").to_numpy(),
            dtype=float,
        ),
    )
    for key, values in support.items():
        aligned[f"{prefix}:{key}"] = values


def _perp_carry_position_series(
    *,
    support_inputs: _CrowdingSupportInputs,
    config: _PerpCrowdingCarryConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    support = _crowding_support_series(
        funding_rate=support_inputs.funding_rate,
        open_interest=support_inputs.open_interest,
        mark_price=support_inputs.mark_price,
        index_price=support_inputs.index_price,
        liquidation_long_notional=support_inputs.liquidation_long_notional,
        liquidation_short_notional=support_inputs.liquidation_short_notional,
        window=config.window,
    )
    score = np.asarray(support["crowding_score"], dtype=float)
    oi_delta_z = np.asarray(support["oi_delta_z"], dtype=float)
    funding = support_inputs.funding_rate
    close_arr = support_inputs.close

    position = np.zeros(close_arr.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    bars_held = 0

    for idx in range(close_arr.size):
        step = _perp_carry_step_input(
            idx=idx,
            funding=funding,
            score=score,
            oi_delta_z=oi_delta_z,
            close_arr=close_arr,
        )
        mode, entry_price, bars_held, position[idx] = _perp_carry_step(
            mode=mode,
            entry_price=entry_price,
            bars_held=bars_held,
            step=step,
            config=config,
        )

    return position, support


def _perp_carry_step_input(
    *,
    idx: int,
    funding: np.ndarray,
    score: np.ndarray,
    oi_delta_z: np.ndarray,
    close_arr: np.ndarray,
) -> _PerpCarryStepInput:
    return _PerpCarryStepInput(
        funding_i=float(funding[idx]),
        score_i=float(score[idx]),
        oi_delta_z_i=float(oi_delta_z[idx]),
        close_i=float(close_arr[idx]),
    )


def _perp_carry_step(
    *,
    mode: int,
    entry_price: float | None,
    bars_held: int,
    step: _PerpCarryStepInput,
    config: _PerpCrowdingCarryConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, bars_held, float(mode)

    if mode != 0:
        next_bars_held = bars_held + 1
        if _perp_carry_should_exit(
            mode=mode,
            score_i=step.score_i,
            funding_i=step.funding_i,
            close_i=step.close_i,
            entry_price=entry_price,
            bars_held=next_bars_held,
            config=config,
        ):
            return 0, None, 0, 0.0
        return mode, entry_price, next_bars_held, float(mode)

    next_mode = _perp_carry_entry_mode(
        funding_i=step.funding_i,
        score_i=step.score_i,
        oi_delta_z_i=step.oi_delta_z_i,
        config=config,
    )
    if next_mode == 0:
        return 0, entry_price, 0, 0.0
    return next_mode, step.close_i, 0, float(next_mode)


def _perp_carry_should_exit(
    *,
    mode: int,
    score_i: float,
    funding_i: float,
    close_i: float,
    entry_price: float | None,
    bars_held: int,
    config: _PerpCrowdingCarryConfig,
) -> bool:
    if mode == 1:
        return (
            (np.isfinite(score_i) and float(score_i) <= config.exit_threshold)
            or (np.isfinite(funding_i) and float(funding_i) >= config.extreme_funding)
            or (bars_held >= config.max_hold_bars)
            or (
                entry_price is not None
                and close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
    return (
        (np.isfinite(score_i) and float(score_i) >= -config.exit_threshold)
        or (np.isfinite(funding_i) and float(funding_i) <= -config.extreme_funding)
        or (bars_held >= config.max_hold_bars)
        or (
            entry_price is not None and close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
        )
    )


def _perp_carry_entry_mode(
    *,
    funding_i: float,
    score_i: float,
    oi_delta_z_i: float,
    config: _PerpCrowdingCarryConfig,
) -> int:
    if not (np.isfinite(funding_i) and np.isfinite(score_i) and np.isfinite(oi_delta_z_i)):
        return 0

    carry_long = (
        (funding_i > 0.0)
        and (funding_i <= config.mild_funding)
        and (score_i >= config.entry_threshold)
    )
    crowded_long = (
        (funding_i >= config.extreme_funding)
        and (oi_delta_z_i > 0.0)
        and (score_i >= config.entry_threshold)
    )
    carry_short = (
        (funding_i < 0.0)
        and (abs(funding_i) <= config.mild_funding)
        and (score_i <= -config.entry_threshold)
    )
    crowded_short = (
        (funding_i <= -config.extreme_funding)
        and (oi_delta_z_i < 0.0)
        and (score_i <= -config.entry_threshold)
    )

    if carry_long and not crowded_long:
        return 1
    if config.allow_short and (crowded_long or (carry_short and not crowded_short)):
        return -1
    if crowded_short:
        return 1
    return 0


def _returns_from_close(closes: np.ndarray) -> np.ndarray:
    if closes.size < 2:
        return np.zeros(closes.shape, dtype=float)
    return np.diff(closes, prepend=closes[0]) / np.clip(
        np.r_[closes[0], closes[:-1]], 1e-12, np.inf
    )


def _resolve_strategy_anchor_symbol(
    raw_symbol: Any,
    symbols: Sequence[str],
    *,
    default: str,
) -> str:
    resolved = canonical_symbol(str(raw_symbol or default))
    if resolved not in symbols and symbols:
        return canonical_symbol(str(symbols[0]))
    return resolved


@dataclass(frozen=True, slots=True)
class _CompositeTrendStrategyConfig:
    long_threshold: float
    short_threshold: float
    exit_score_cross: float
    te_min: float
    vr_min: float
    risk_target_vol: float
    max_signal_strength: float
    vol_window: int
    max_hold_bars: int
    allow_short: bool
    benchmark_regime_ma: int
    benchmark_symbol: str
    crowding_reduce_threshold: float
    crowding_block_threshold: float


@dataclass(frozen=True, slots=True)
class _CompositeTrendStepInput:
    close_i: float
    score_i: float
    long_gate_i: bool
    short_gate_i: bool
    strength: float
    blocked: bool


def _resolve_composite_trend_strategy_config(
    params: Mapping[str, Any],
    symbols: Sequence[str],
) -> _CompositeTrendStrategyConfig:
    return _CompositeTrendStrategyConfig(
        long_threshold=float(params.get("long_threshold", 0.55)),
        short_threshold=float(params.get("short_threshold", 0.55)),
        exit_score_cross=float(params.get("exit_score_cross", 0.05)),
        te_min=float(params.get("te_min", 0.25)),
        vr_min=float(params.get("vr_min", 0.85)),
        risk_target_vol=float(params.get("risk_target_vol", 0.004)),
        max_signal_strength=float(params.get("max_signal_strength", 2.0)),
        vol_window=int(params.get("vol_window", 120)),
        max_hold_bars=int(params.get("max_hold_bars", 640)),
        allow_short=bool(params.get("allow_short", True)),
        benchmark_regime_ma=max(0, int(params.get("benchmark_regime_ma", 0))),
        benchmark_symbol=_resolve_strategy_anchor_symbol(
            params.get("benchmark_symbol"),
            symbols,
            default="BTC/USDT",
        ),
        crowding_reduce_threshold=float(params.get("crowding_reduce_threshold", 0.55)),
        crowding_block_threshold=float(params.get("crowding_block_threshold", 0.85)),
    )


def _composite_trend_benchmark_long_gate(
    *,
    aligned: Mapping[str, np.ndarray],
    n: int,
    config: _CompositeTrendStrategyConfig,
) -> np.ndarray:
    gate = np.ones(n, dtype=bool)
    if config.benchmark_regime_ma <= 0:
        return gate

    benchmark_close = np.asarray(aligned[f"{config.benchmark_symbol}:close"], dtype=float)
    for idx in range(config.benchmark_regime_ma - 1, n):
        window = benchmark_close[idx - config.benchmark_regime_ma + 1 : idx + 1]
        if np.all(np.isfinite(window)):
            gate[idx] = bool(float(benchmark_close[idx]) >= _safe_mean(window))
    return gate


def _apply_composite_trend_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_composite_trend_strategy_config(params, symbols)
    benchmark_long_gate = _composite_trend_benchmark_long_gate(
        aligned=aligned,
        n=n,
        config=config,
    )

    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        volume = np.asarray(aligned[f"{symbol}:volume"], dtype=float)

        ret8 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 8), 1e-12, np.inf))
        ret21 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 21), 1e-12, np.inf))
        ret55 = np.log(np.clip(close, 1e-12, np.inf) / np.clip(np.roll(close, 55), 1e-12, np.inf))
        z8 = _rolling_z(ret8, 120)
        z21 = _rolling_z(ret21, 120)
        z55 = _rolling_z(ret55, 120)
        mom = (0.5 * np.nan_to_num(z8)) + (0.3 * np.nan_to_num(z21)) + (0.2 * np.nan_to_num(z55))
        te = np.nan_to_num(_trend_efficiency_series(close, 55), nan=0.0)
        vr = np.nan_to_num(_vol_ratio_series(close, 8, 55), nan=0.0)

        vol_shock = _rolling_z(np.nan_to_num(volume, nan=0.0), 64)
        score = mom * (1.0 + 0.25 * np.tanh(np.nan_to_num(vol_shock) / 2.0)) * np.clip(te, 0.0, 1.0)
        gate = (te >= config.te_min) & (vr >= config.vr_min)
        long_gate = gate & benchmark_long_gate

        crowding = aligned.get(f"{symbol}:crowding_score")
        exposures[s_idx] = _composite_trend_position_series(
            close=close,
            score=np.asarray(score, dtype=float),
            gate=np.asarray(gate, dtype=bool),
            long_gate=np.asarray(long_gate, dtype=bool),
            short_gate=np.asarray(gate, dtype=bool),
            crowding=None if crowding is None else np.asarray(crowding, dtype=float),
            config=config,
        )

    if config.benchmark_regime_ma > 0:
        meta["benchmark_regime_ma"] = config.benchmark_regime_ma
        meta["benchmark_symbol"] = config.benchmark_symbol
        meta["crash_aware_gate"] = True


@dataclass(frozen=True, slots=True)
class _TopCapTimeSeriesMomentumConfig:
    lookback_bars: int
    rebalance_bars: int
    signal_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    max_longs: int
    max_shorts: int
    min_price: float
    btc_regime_ma: int
    benchmark_drawdown_window: int
    benchmark_drawdown_limit: float
    residualize_btc: bool
    residualize_mean: bool
    btc_symbol: str


def _resolve_topcap_tsmom_config(
    params: Mapping[str, Any],
    symbols: Sequence[str],
) -> _TopCapTimeSeriesMomentumConfig:
    return _TopCapTimeSeriesMomentumConfig(
        lookback_bars=max(2, int(params.get("lookback_bars", 16))),
        rebalance_bars=max(1, int(params.get("rebalance_bars", 4))),
        signal_threshold=float(params.get("signal_threshold", 0.04)),
        stop_loss_pct=max(0.0, float(params.get("stop_loss_pct", 0.08))),
        take_profit_pct=max(0.0, float(params.get("take_profit_pct", 0.0))),
        max_longs=max(0, int(params.get("max_longs", 2))),
        max_shorts=max(0, int(params.get("max_shorts", 2))),
        min_price=max(0.0, float(params.get("min_price", 0.10))),
        btc_regime_ma=max(0, int(params.get("btc_regime_ma", 0))),
        benchmark_drawdown_window=max(0, int(params.get("benchmark_drawdown_window", 0))),
        benchmark_drawdown_limit=max(0.0, float(params.get("benchmark_drawdown_limit", 0.0))),
        residualize_btc=bool(params.get("residualize_btc", False)),
        residualize_mean=bool(params.get("residualize_mean", False)),
        btc_symbol=_resolve_strategy_anchor_symbol(
            params.get("btc_symbol"),
            symbols,
            default="BTC/USDT",
        ),
    )


def _apply_topcap_risk_exits(
    *,
    current_close_map: Mapping[str, float],
    symbols: Sequence[str],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    config: _TopCapTimeSeriesMomentumConfig,
) -> None:
    if config.stop_loss_pct <= 0.0 and config.take_profit_pct <= 0.0:
        return

    for s_idx, symbol in enumerate(symbols):
        if position_state[s_idx] == 0.0 or not np.isfinite(entry_price[s_idx]):
            continue

        close_i = current_close_map[symbol]
        if not np.isfinite(close_i) or close_i <= 0.0:
            continue

        stop_long = position_state[s_idx] > 0.0 and close_i <= float(entry_price[s_idx]) * (
            1.0 - config.stop_loss_pct
        )
        stop_short = position_state[s_idx] < 0.0 and close_i >= float(entry_price[s_idx]) * (
            1.0 + config.stop_loss_pct
        )
        tp_long = (
            config.take_profit_pct > 0.0
            and position_state[s_idx] > 0.0
            and close_i >= float(entry_price[s_idx]) * (1.0 + config.take_profit_pct)
        )
        tp_short = (
            config.take_profit_pct > 0.0
            and position_state[s_idx] < 0.0
            and close_i <= float(entry_price[s_idx]) * (1.0 - config.take_profit_pct)
        )
        if stop_long or stop_short or tp_long or tp_short:
            position_state[s_idx] = 0.0
            entry_price[s_idx] = np.nan


def _topcap_ranked_momentum_rows(
    *,
    close_by_symbol: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    idx: int,
    config: _TopCapTimeSeriesMomentumConfig,
) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for symbol in symbols:
        close_arr = close_by_symbol[symbol]
        latest = float(close_arr[idx])
        base = float(close_arr[idx - config.lookback_bars])
        if (
            latest < config.min_price
            or base <= 0.0
            or not np.isfinite(latest)
            or not np.isfinite(base)
        ):
            continue
        momentum = (latest / base) - 1.0
        if np.isfinite(momentum):
            rows.append((float(momentum), symbol))
    return rows


def _topcap_market_regime(
    btc_close: np.ndarray,
    *,
    idx: int,
    config: _TopCapTimeSeriesMomentumConfig,
) -> str:
    regime = "BOTH"
    if (
        config.benchmark_drawdown_window > 0
        and config.benchmark_drawdown_limit > 0.0
        and idx >= config.benchmark_drawdown_window - 1
    ):
        window = btc_close[idx - config.benchmark_drawdown_window + 1 : idx + 1]
        if np.all(np.isfinite(window)):
            peak = float(np.max(window))
            if peak > 0.0:
                drawdown = (float(btc_close[idx]) / peak) - 1.0
                if drawdown <= -config.benchmark_drawdown_limit:
                    regime = "RISK_OFF"

    if regime != "RISK_OFF" and config.btc_regime_ma > 0 and idx >= config.btc_regime_ma:
        window = btc_close[idx - config.btc_regime_ma + 1 : idx + 1]
        if np.all(np.isfinite(window)):
            regime = "RISK_ON" if float(btc_close[idx]) >= _safe_mean(window) else "RISK_OFF"
    return regime


def _topcap_residualized_rows(
    momentum_rows: list[tuple[float, str]],
    *,
    config: _TopCapTimeSeriesMomentumConfig,
) -> list[tuple[float, str]]:
    if not momentum_rows:
        return momentum_rows

    if not config.residualize_btc and not config.residualize_mean:
        momentum_rows.sort(key=lambda item: item[0])
        return momentum_rows

    momentum_map = {symbol: momentum for momentum, symbol in momentum_rows}
    if config.residualize_btc and config.btc_symbol in momentum_map:
        btc_momentum = float(momentum_map[config.btc_symbol])
        for symbol in list(momentum_map):
            momentum_map[symbol] = float(momentum_map[symbol]) - btc_momentum
    if config.residualize_mean and momentum_map:
        mean_momentum = _safe_mean(np.asarray(list(momentum_map.values()), dtype=float))
        for symbol in list(momentum_map):
            momentum_map[symbol] = float(momentum_map[symbol]) - mean_momentum

    residualized = [(float(momentum_map[symbol]), symbol) for _, symbol in momentum_rows]
    residualized.sort(key=lambda item: item[0])
    return residualized


def _cross_sectional_score_map(values: Mapping[str, float]) -> dict[str, float]:
    finite = {
        str(symbol): float(value) for symbol, value in values.items() if np.isfinite(float(value))
    }
    out = {str(symbol): 0.0 for symbol in values}
    if len(finite) < 2:
        return out
    arr = np.asarray(list(finite.values()), dtype=float)
    std = _safe_std(arr)
    if not np.isfinite(std) or std <= 1e-12:
        return out
    mean_value = _safe_mean(arr)
    for symbol, value in finite.items():
        out[symbol] = float((value - mean_value) / std)
    return out


def _cross_sectional_score_matrix(raw: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    scores = np.zeros_like(raw, dtype=float)
    if raw.size == 0 or valid_mask.size == 0:
        return scores

    counts = valid_mask.sum(axis=0, dtype=np.int64)
    usable = counts >= 2
    if not np.any(usable):
        return scores

    masked = np.where(valid_mask, raw, 0.0)
    means = np.zeros(raw.shape[1], dtype=float)
    means[usable] = masked[:, usable].sum(axis=0, dtype=float) / counts[usable]
    centered = np.where(valid_mask, raw - means[np.newaxis, :], 0.0)
    variances = np.zeros(raw.shape[1], dtype=float)
    variances[usable] = (centered[:, usable] ** 2).sum(axis=0, dtype=float) / counts[usable]
    std = np.sqrt(variances, dtype=float)
    usable = usable & np.isfinite(std) & (std > 1e-12)
    if not np.any(usable):
        return scores

    scores[:, usable] = np.where(valid_mask[:, usable], centered[:, usable] / std[usable], 0.0)
    scores[~np.isfinite(scores)] = 0.0
    return scores


def _carry_trend_factor_score_matrix(
    *,
    close_matrix: np.ndarray,
    realized_vol_matrix: np.ndarray,
    funding_z_matrix: np.ndarray,
    basis_z_matrix: np.ndarray,
    crowding_score_matrix: np.ndarray,
    config: _CarryTrendFactorRotationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    symbol_count, bar_count = close_matrix.shape
    valid_mask = np.zeros((symbol_count, bar_count), dtype=bool)
    if bar_count <= config.lookback_bars:
        return np.zeros_like(close_matrix, dtype=float), valid_mask

    momentum_raw = np.zeros_like(close_matrix, dtype=float)
    latest = close_matrix[:, config.lookback_bars :]
    base = close_matrix[:, : bar_count - config.lookback_bars]
    momentum_core = (
        np.divide(
            latest,
            base,
            out=np.full_like(latest, np.nan, dtype=float),
            where=np.isfinite(base) & (base > 0.0),
        )
        - 1.0
    )
    valid_core = (
        np.isfinite(latest)
        & np.isfinite(base)
        & np.isfinite(momentum_core)
        & (latest >= config.min_price)
        & (base > 0.0)
    )
    valid_mask[:, config.lookback_bars :] = valid_core
    momentum_raw[:, config.lookback_bars :] = np.where(valid_core, momentum_core, 0.0)

    defensive_raw = np.nan_to_num(-realized_vol_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    carry_raw = -(
        np.nan_to_num(funding_z_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        + (0.5 * np.nan_to_num(basis_z_matrix, nan=0.0, posinf=0.0, neginf=0.0))
    )
    crowding_raw = np.nan_to_num(-crowding_score_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        (config.trend_weight * _cross_sectional_score_matrix(momentum_raw, valid_mask))
        + (config.carry_weight * _cross_sectional_score_matrix(carry_raw, valid_mask))
        + (config.defensive_weight * _cross_sectional_score_matrix(defensive_raw, valid_mask))
        + (config.crowding_weight * _cross_sectional_score_matrix(crowding_raw, valid_mask)),
        valid_mask,
    )


def _carry_trend_factor_target_indices(
    *,
    score_column: np.ndarray,
    valid_indices: np.ndarray,
    regime: str,
    config: _CarryTrendFactorRotationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if valid_indices.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    ordered = valid_indices[np.argsort(score_column[valid_indices], kind="stable")]
    long_indices = ordered[score_column[ordered] >= config.signal_threshold]
    short_indices = ordered[score_column[ordered] <= -config.signal_threshold]

    if regime == "RISK_ON":
        short_indices = np.empty(0, dtype=np.int64)
    elif regime == "RISK_OFF":
        long_indices = np.empty(0, dtype=np.int64)

    if long_indices.size > 0 and config.max_longs > 0:
        long_indices = long_indices[::-1][: config.max_longs]
    else:
        long_indices = np.empty(0, dtype=np.int64)

    if not config.allow_short or config.max_shorts <= 0 or short_indices.size == 0:
        return long_indices, np.empty(0, dtype=np.int64)

    short_indices = short_indices[: config.max_shorts]
    if long_indices.size == 0:
        return long_indices, short_indices
    return long_indices, short_indices[~np.isin(short_indices, long_indices, assume_unique=False)]


@dataclass(frozen=True, slots=True)
class _CarryTrendFactorRotationConfig:
    lookback_bars: int
    rebalance_bars: int
    signal_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    max_longs: int
    max_shorts: int
    min_price: float
    btc_regime_ma: int
    benchmark_drawdown_window: int
    benchmark_drawdown_limit: float
    vol_window: int
    crowding_window: int
    trend_weight: float
    carry_weight: float
    defensive_weight: float
    crowding_weight: float
    allow_short: bool
    btc_symbol: str


def _resolve_carry_trend_factor_rotation_config(
    params: Mapping[str, Any],
    symbols: Sequence[str],
) -> _CarryTrendFactorRotationConfig:
    return _CarryTrendFactorRotationConfig(
        lookback_bars=max(4, int(params.get("lookback_bars", 24))),
        rebalance_bars=max(1, int(params.get("rebalance_bars", 8))),
        signal_threshold=max(0.0, float(params.get("signal_threshold", 0.25))),
        stop_loss_pct=max(0.0, float(params.get("stop_loss_pct", 0.05))),
        take_profit_pct=max(0.0, float(params.get("take_profit_pct", 0.0))),
        max_longs=max(0, int(params.get("max_longs", 3))),
        max_shorts=max(0, int(params.get("max_shorts", 2))),
        min_price=max(0.0, float(params.get("min_price", 0.10))),
        btc_regime_ma=max(0, int(params.get("btc_regime_ma", 72))),
        benchmark_drawdown_window=max(0, int(params.get("benchmark_drawdown_window", 0))),
        benchmark_drawdown_limit=max(0.0, float(params.get("benchmark_drawdown_limit", 0.0))),
        vol_window=max(4, int(params.get("vol_window", 48))),
        crowding_window=max(12, int(params.get("crowding_window", 72))),
        trend_weight=float(params.get("trend_weight", 0.55)),
        carry_weight=float(params.get("carry_weight", 0.20)),
        defensive_weight=float(params.get("defensive_weight", 0.15)),
        crowding_weight=float(params.get("crowding_weight", 0.10)),
        allow_short=bool(params.get("allow_short", True)),
        btc_symbol=_resolve_strategy_anchor_symbol(
            params.get("btc_symbol"),
            symbols,
            default="BTC/USDT",
        ),
    )


def _carry_trend_factor_rows(
    *,
    close_by_symbol: Mapping[str, np.ndarray],
    realized_vol_by_symbol: Mapping[str, np.ndarray],
    crowding_support_by_symbol: Mapping[str, dict[str, np.ndarray]],
    symbols: Sequence[str],
    idx: int,
    config: _CarryTrendFactorRotationConfig,
) -> list[tuple[float, str]]:
    raw_momentum: dict[str, float] = {}
    raw_carry: dict[str, float] = {}
    raw_defensive: dict[str, float] = {}
    raw_crowding: dict[str, float] = {}

    for symbol in symbols:
        close_arr = close_by_symbol[symbol]
        latest = float(close_arr[idx])
        base = float(close_arr[idx - config.lookback_bars])
        if (
            latest < config.min_price
            or base <= 0.0
            or not np.isfinite(latest)
            or not np.isfinite(base)
        ):
            continue

        raw_momentum[symbol] = float((latest / base) - 1.0)
        realized_vol = float(realized_vol_by_symbol[symbol][idx])
        raw_defensive[symbol] = float(-realized_vol) if np.isfinite(realized_vol) else 0.0

        support = crowding_support_by_symbol.get(symbol) or {}
        if support:
            funding_series = support.get("funding_z")
            basis_series = support.get("basis_z")
            crowding_series = support.get("crowding_score")
            funding_z = float(funding_series[idx]) if funding_series is not None else 0.0
            basis_z = float(basis_series[idx]) if basis_series is not None else 0.0
            crowding_score = float(crowding_series[idx]) if crowding_series is not None else 0.0
        else:
            funding_z = 0.0
            basis_z = 0.0
            crowding_score = 0.0
        raw_carry[symbol] = (
            float(-(funding_z + (0.5 * basis_z)))
            if np.isfinite(funding_z) or np.isfinite(basis_z)
            else 0.0
        )
        raw_crowding[symbol] = float(-crowding_score) if np.isfinite(crowding_score) else 0.0

    if not raw_momentum:
        return []

    momentum_score = _cross_sectional_score_map(raw_momentum)
    carry_score = _cross_sectional_score_map(raw_carry)
    defensive_score = _cross_sectional_score_map(raw_defensive)
    crowding_score = _cross_sectional_score_map(raw_crowding)

    rows = [
        (
            float(
                (config.trend_weight * momentum_score.get(symbol, 0.0))
                + (config.carry_weight * carry_score.get(symbol, 0.0))
                + (config.defensive_weight * defensive_score.get(symbol, 0.0))
                + (config.crowding_weight * crowding_score.get(symbol, 0.0))
            ),
            symbol,
        )
        for symbol in raw_momentum
    ]
    rows.sort(key=lambda item: item[0])
    return rows


def _carry_trend_factor_target_sets(
    score_rows: list[tuple[float, str]],
    *,
    regime: str,
    config: _CarryTrendFactorRotationConfig,
) -> tuple[set[str], set[str]]:
    longs = [symbol for score, symbol in reversed(score_rows) if score >= config.signal_threshold][
        : config.max_longs
    ]
    shorts: list[str] = []
    if config.allow_short:
        shorts = [symbol for score, symbol in score_rows if score <= -config.signal_threshold][
            : config.max_shorts
        ]
    if regime == "RISK_ON":
        shorts = []
    elif regime == "RISK_OFF":
        longs = []

    long_set = set(longs)
    short_set = {symbol for symbol in shorts if symbol not in long_set}
    return long_set, short_set


def _apply_carry_trend_factor_rotation_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_carry_trend_factor_rotation_config(params, symbols)
    symbol_count = len(symbols)
    close_matrix = np.vstack(
        [np.asarray(aligned[f"{symbol}:close"], dtype=float) for symbol in symbols]
    )
    btc_close = close_matrix[symbols.index(config.btc_symbol)]
    position_state = np.zeros(symbol_count, dtype=float)
    entry_price = np.full(symbol_count, np.nan, dtype=float)
    realized_vol_matrix = np.vstack(
        [
            _rolling_realized_vol(close_matrix[s_idx], config.vol_window)
            for s_idx in range(symbol_count)
        ]
    )

    crowding_support_by_symbol: dict[str, dict[str, np.ndarray]] = {}
    missing_symbols: list[str] = []
    funding_z_matrix = np.zeros_like(close_matrix, dtype=float)
    basis_z_matrix = np.zeros_like(close_matrix, dtype=float)
    crowding_score_matrix = np.zeros_like(close_matrix, dtype=float)
    for s_idx, symbol in enumerate(symbols):
        support_inputs = _resolve_crowding_support_inputs(aligned=aligned, symbol=symbol)
        if support_inputs is None:
            missing_symbols.append(symbol)
            continue
        support = _crowding_support_series(
            funding_rate=support_inputs.funding_rate,
            open_interest=support_inputs.open_interest,
            mark_price=support_inputs.mark_price,
            index_price=support_inputs.index_price,
            liquidation_long_notional=support_inputs.liquidation_long_notional,
            liquidation_short_notional=support_inputs.liquidation_short_notional,
            window=config.crowding_window,
        )
        crowding_support_by_symbol[symbol] = support
        _note_support_data_symbol(
            meta, symbol=symbol, values=np.asarray(support.get("crowding_score"), dtype=float)
        )
        funding_z_matrix[s_idx] = np.asarray(support.get("funding_z"), dtype=float)
        basis_z_matrix[s_idx] = np.asarray(support.get("basis_z"), dtype=float)
        crowding_score_matrix[s_idx] = np.asarray(support.get("crowding_score"), dtype=float)
    _finalize_missing_support_symbols(meta, missing_symbols=missing_symbols)

    score_matrix, valid_mask = _carry_trend_factor_score_matrix(
        close_matrix=close_matrix,
        realized_vol_matrix=realized_vol_matrix,
        funding_z_matrix=funding_z_matrix,
        basis_z_matrix=basis_z_matrix,
        crowding_score_matrix=crowding_score_matrix,
        config=config,
    )
    rebalance_due = (np.arange(n, dtype=np.int64) + 1) % max(1, config.rebalance_bars) == 0
    rebalance_due[: min(n, config.lookback_bars)] = False

    topcap_proxy_config = _TopCapTimeSeriesMomentumConfig(
        lookback_bars=config.lookback_bars,
        rebalance_bars=config.rebalance_bars,
        signal_threshold=config.signal_threshold,
        stop_loss_pct=config.stop_loss_pct,
        take_profit_pct=config.take_profit_pct,
        max_longs=config.max_longs,
        max_shorts=config.max_shorts if config.allow_short else 0,
        min_price=config.min_price,
        btc_regime_ma=config.btc_regime_ma,
        benchmark_drawdown_window=config.benchmark_drawdown_window,
        benchmark_drawdown_limit=config.benchmark_drawdown_limit,
        residualize_btc=False,
        residualize_mean=False,
        btc_symbol=config.btc_symbol,
    )

    for idx in range(n):
        close_column = close_matrix[:, idx]
        if topcap_proxy_config.stop_loss_pct > 0.0 or topcap_proxy_config.take_profit_pct > 0.0:
            active_mask = (
                (position_state != 0.0)
                & np.isfinite(entry_price)
                & np.isfinite(close_column)
                & (close_column > 0.0)
            )
            if np.any(active_mask):
                exit_mask = np.zeros(symbol_count, dtype=bool)
                if topcap_proxy_config.stop_loss_pct > 0.0:
                    exit_mask |= (
                        (position_state > 0.0)
                        & active_mask
                        & (
                            close_column
                            <= (entry_price * (1.0 - topcap_proxy_config.stop_loss_pct))
                        )
                    )
                    exit_mask |= (
                        (position_state < 0.0)
                        & active_mask
                        & (
                            close_column
                            >= (entry_price * (1.0 + topcap_proxy_config.stop_loss_pct))
                        )
                    )
                if topcap_proxy_config.take_profit_pct > 0.0:
                    exit_mask |= (
                        (position_state > 0.0)
                        & active_mask
                        & (
                            close_column
                            >= (entry_price * (1.0 + topcap_proxy_config.take_profit_pct))
                        )
                    )
                    exit_mask |= (
                        (position_state < 0.0)
                        & active_mask
                        & (
                            close_column
                            <= (entry_price * (1.0 - topcap_proxy_config.take_profit_pct))
                        )
                    )
                if np.any(exit_mask):
                    position_state[exit_mask] = 0.0
                    entry_price[exit_mask] = np.nan

        if rebalance_due[idx]:
            valid_indices = np.flatnonzero(valid_mask[:, idx])
            regime = _topcap_market_regime(
                btc_close,
                idx=idx,
                config=topcap_proxy_config,
            )
            long_indices, short_indices = _carry_trend_factor_target_indices(
                score_column=score_matrix[:, idx],
                valid_indices=valid_indices,
                regime=regime,
                config=config,
            )
            next_position = np.zeros(symbol_count, dtype=float)
            if long_indices.size > 0:
                next_position[long_indices] = 1.0
            if short_indices.size > 0:
                next_position[short_indices] = -1.0

            entry_updates = (next_position != 0.0) & (
                (position_state != next_position) | ~np.isfinite(entry_price)
            )
            entry_price[next_position == 0.0] = np.nan
            if np.any(entry_updates):
                entry_price[entry_updates] = close_column[entry_updates]
            position_state[:] = next_position
        exposures[:, idx] = position_state

    meta["cross_sectional"] = True
    meta["factor_rotation"] = True
    meta["factor_rotation_strategy"] = "CarryTrendFactorRotationStrategy"
    meta["trend_weight"] = config.trend_weight
    meta["carry_weight"] = config.carry_weight
    meta["defensive_weight"] = config.defensive_weight
    meta["crowding_weight"] = config.crowding_weight
    meta["allow_short"] = config.allow_short
    if config.benchmark_drawdown_window > 0 and config.benchmark_drawdown_limit > 0.0:
        meta["crash_aware_gate"] = True
        meta["benchmark_drawdown_window"] = config.benchmark_drawdown_window
        meta["benchmark_drawdown_limit"] = config.benchmark_drawdown_limit


def _topcap_target_sets(
    momentum_rows: list[tuple[float, str]],
    *,
    regime: str,
    config: _TopCapTimeSeriesMomentumConfig,
) -> tuple[set[str], set[str]]:
    longs = [
        symbol
        for momentum, symbol in reversed(momentum_rows)
        if momentum >= config.signal_threshold
    ][: config.max_longs]
    shorts = [symbol for momentum, symbol in momentum_rows if momentum <= -config.signal_threshold][
        : config.max_shorts
    ]
    if regime == "RISK_ON":
        shorts = []
    elif regime == "RISK_OFF":
        longs = []

    long_set = set(longs)
    short_set = {symbol for symbol in shorts if symbol not in long_set}
    return long_set, short_set


def _apply_topcap_target_positions(
    *,
    current_close_map: Mapping[str, float],
    symbols: Sequence[str],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    long_set: set[str],
    short_set: set[str],
) -> None:
    for s_idx, symbol in enumerate(symbols):
        next_position = 0.0
        if symbol in long_set:
            next_position = 1.0
        elif symbol in short_set:
            next_position = -1.0

        if next_position == 0.0:
            if position_state[s_idx] != 0.0:
                entry_price[s_idx] = np.nan
            position_state[s_idx] = 0.0
            continue

        if position_state[s_idx] != next_position or not np.isfinite(entry_price[s_idx]):
            entry_price[s_idx] = current_close_map[symbol]
        position_state[s_idx] = next_position


def _topcap_rebalance_due(
    *,
    idx: int,
    config: _TopCapTimeSeriesMomentumConfig,
) -> bool:
    return idx >= config.lookback_bars and (idx + 1) % config.rebalance_bars == 0


def _rebalance_topcap_positions(
    *,
    idx: int,
    current_close_map: Mapping[str, float],
    close_by_symbol: Mapping[str, np.ndarray],
    btc_close: np.ndarray,
    symbols: Sequence[str],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    config: _TopCapTimeSeriesMomentumConfig,
) -> None:
    momentum_rows = _topcap_ranked_momentum_rows(
        close_by_symbol=close_by_symbol,
        symbols=symbols,
        idx=idx,
        config=config,
    )
    regime = _topcap_market_regime(
        btc_close,
        idx=idx,
        config=config,
    )
    ranked_rows = _topcap_residualized_rows(momentum_rows, config=config)
    long_set, short_set = _topcap_target_sets(
        ranked_rows,
        regime=regime,
        config=config,
    )
    _apply_topcap_target_positions(
        current_close_map=current_close_map,
        symbols=symbols,
        position_state=position_state,
        entry_price=entry_price,
        long_set=long_set,
        short_set=short_set,
    )


def _apply_topcap_tsmom_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_topcap_tsmom_config(params, symbols)
    close_by_symbol = {
        symbol: np.asarray(aligned[f"{symbol}:close"], dtype=float) for symbol in symbols
    }
    btc_close = close_by_symbol[config.btc_symbol]
    position_state = np.zeros(len(symbols), dtype=float)
    entry_price = np.full(len(symbols), np.nan, dtype=float)

    for idx in range(n):
        current_close_map = {symbol: float(close_by_symbol[symbol][idx]) for symbol in symbols}
        _apply_topcap_risk_exits(
            current_close_map=current_close_map,
            symbols=symbols,
            position_state=position_state,
            entry_price=entry_price,
            config=config,
        )

        if not _topcap_rebalance_due(idx=idx, config=config):
            exposures[:, idx] = position_state
            continue

        _rebalance_topcap_positions(
            idx=idx,
            current_close_map=current_close_map,
            close_by_symbol=close_by_symbol,
            btc_close=btc_close,
            symbols=symbols,
            position_state=position_state,
            entry_price=entry_price,
            config=config,
        )
        exposures[:, idx] = position_state

    meta["cross_sectional"] = True
    if config.take_profit_pct > 0.0:
        meta["take_profit_pct"] = config.take_profit_pct
    if config.residualize_btc or config.residualize_mean:
        meta["residualized_cross_sectional"] = True
        meta["residualize_btc"] = config.residualize_btc
        meta["residualize_mean"] = config.residualize_mean
    if config.benchmark_drawdown_window > 0 and config.benchmark_drawdown_limit > 0.0:
        meta["crash_aware_gate"] = True
        meta["benchmark_drawdown_window"] = config.benchmark_drawdown_window
        meta["benchmark_drawdown_limit"] = config.benchmark_drawdown_limit


@dataclass(frozen=True, slots=True)
class _LastDayLiquidityRegimeConfig:
    momentum_lookback_bars: int
    signal_skip_bars: int
    liquidity_window: int
    volatility_window: int
    rebalance_bars: int
    signal_threshold: float
    liquidity_quantile: float
    max_longs: int
    max_shorts: int
    min_price: float
    max_realized_vol: float
    stop_loss_pct: float
    allow_short: bool
    illiquid_reversal: bool


def _resolve_last_day_liquidity_regime_config(
    params: Mapping[str, Any],
) -> _LastDayLiquidityRegimeConfig:
    return _LastDayLiquidityRegimeConfig(
        momentum_lookback_bars=max(4, int(params.get("momentum_lookback_bars", 24))),
        signal_skip_bars=max(0, int(params.get("signal_skip_bars", 1))),
        liquidity_window=max(4, int(params.get("liquidity_window", 24))),
        volatility_window=max(4, int(params.get("volatility_window", 24))),
        rebalance_bars=max(1, int(params.get("rebalance_bars", 6))),
        signal_threshold=max(0.0, float(params.get("signal_threshold", 0.012))),
        liquidity_quantile=min(1.0, max(0.0, float(params.get("liquidity_quantile", 0.60)))),
        max_longs=max(0, int(params.get("max_longs", 2))),
        max_shorts=max(0, int(params.get("max_shorts", 1))),
        min_price=max(0.0, float(params.get("min_price", 0.10))),
        max_realized_vol=max(0.0, float(params.get("max_realized_vol", 0.09))),
        stop_loss_pct=max(0.0, float(params.get("stop_loss_pct", 0.05))),
        allow_short=bool(params.get("allow_short", True)),
        illiquid_reversal=bool(params.get("illiquid_reversal", True)),
    )


def _lagged_return_series(values: np.ndarray, *, lookback: int, skip_bars: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    required = max(1, int(lookback)) + max(0, int(skip_bars))
    for idx in range(required, arr.size):
        latest_idx = idx - max(0, int(skip_bars))
        base_idx = latest_idx - max(1, int(lookback))
        if base_idx < 0 or latest_idx < 0:
            continue
        base = float(arr[base_idx])
        latest = float(arr[latest_idx])
        if not np.isfinite(base) or not np.isfinite(latest) or base <= 0.0:
            continue
        out[idx] = (latest / base) - 1.0
    return out


def _rolling_mean_dollar_volume_series(
    close: np.ndarray,
    volume: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    close_arr = np.asarray(close, dtype=float)
    volume_arr = np.asarray(volume, dtype=float)
    out = np.full(close_arr.shape, np.nan, dtype=float)
    win = max(1, int(window))
    for idx in range(win - 1, close_arr.size):
        close_tail = close_arr[idx - win + 1 : idx + 1]
        volume_tail = volume_arr[idx - win + 1 : idx + 1]
        valid = (
            np.isfinite(close_tail)
            & np.isfinite(volume_tail)
            & (close_tail > 0.0)
            & (volume_tail >= 0.0)
        )
        if not np.any(valid):
            continue
        dollar_volume = close_tail[valid] * volume_tail[valid]
        out[idx] = float(np.mean(dollar_volume))
    return out


def _liquidity_threshold_for_index(
    liquidity_by_symbol: Mapping[str, np.ndarray],
    *,
    idx: int,
    quantile: float,
) -> float | None:
    values = sorted(
        float(series[idx])
        for series in liquidity_by_symbol.values()
        if idx < len(series) and np.isfinite(series[idx]) and float(series[idx]) > 0.0
    )
    if not values:
        return None
    pos = min(len(values) - 1, max(0, math.floor((len(values) - 1) * quantile)))
    return float(values[pos])


def _rebalance_last_day_liquidity_positions(
    *,
    idx: int,
    symbols: Sequence[str],
    close_by_symbol: Mapping[str, np.ndarray],
    momentum_by_symbol: Mapping[str, np.ndarray],
    liquidity_by_symbol: Mapping[str, np.ndarray],
    realized_vol_by_symbol: Mapping[str, np.ndarray],
    position_state: np.ndarray,
    entry_price: np.ndarray,
    config: _LastDayLiquidityRegimeConfig,
) -> tuple[dict[str, float], dict[str, str], float | None]:
    liquidity_threshold = _liquidity_threshold_for_index(
        liquidity_by_symbol,
        idx=idx,
        quantile=config.liquidity_quantile,
    )
    adjusted_scores: dict[str, float] = {}
    liquidity_regimes: dict[str, str] = {}

    for symbol in symbols:
        close_value = float(close_by_symbol[symbol][idx])
        momentum = float(momentum_by_symbol[symbol][idx])
        liquidity = float(liquidity_by_symbol[symbol][idx])
        realized_vol = float(realized_vol_by_symbol[symbol][idx])
        if (
            not np.isfinite(close_value)
            or close_value < config.min_price
            or not np.isfinite(momentum)
            or not np.isfinite(liquidity)
            or liquidity <= 0.0
        ):
            continue
        if not np.isfinite(realized_vol):
            realized_vol = 0.0
        if realized_vol > config.max_realized_vol:
            continue

        is_liquid = liquidity_threshold is None or liquidity >= liquidity_threshold
        if is_liquid:
            adjusted_scores[symbol] = momentum
            liquidity_regimes[symbol] = "liquid_momentum"
        elif config.illiquid_reversal:
            adjusted_scores[symbol] = -momentum
            liquidity_regimes[symbol] = "illiquid_reversal"
        else:
            liquidity_regimes[symbol] = "filtered_illiquid"

    ordered = sorted(adjusted_scores.items(), key=lambda item: item[1])
    longs = [symbol for symbol, score in reversed(ordered) if score >= config.signal_threshold][
        : config.max_longs
    ]
    shorts: list[str] = []
    if config.allow_short and config.max_shorts > 0:
        shorts = [symbol for symbol, score in ordered if score <= -config.signal_threshold][
            : config.max_shorts
        ]
    long_set = set(longs)
    shorts = [symbol for symbol in shorts if symbol not in long_set]

    symbol_index = {symbol: s_idx for s_idx, symbol in enumerate(symbols)}
    targets = dict.fromkeys(longs, 1.0)
    targets.update(dict.fromkeys(shorts, -1.0))
    for symbol in symbols:
        target = float(targets.get(symbol, 0.0))
        s_idx = symbol_index[symbol]
        position_state[s_idx] = target
        entry_price[s_idx] = float(close_by_symbol[symbol][idx]) if abs(target) > 0.0 else np.nan

    return adjusted_scores, liquidity_regimes, liquidity_threshold


def _apply_last_day_liquidity_regime_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_last_day_liquidity_regime_config(params)
    close_by_symbol = {
        symbol: np.asarray(aligned[f"{symbol}:close"], dtype=float) for symbol in symbols
    }
    volume_by_symbol = {
        symbol: np.asarray(aligned[f"{symbol}:volume"], dtype=float) for symbol in symbols
    }
    momentum_by_symbol = {
        symbol: _lagged_return_series(
            close_arr,
            lookback=config.momentum_lookback_bars,
            skip_bars=config.signal_skip_bars,
        )
        for symbol, close_arr in close_by_symbol.items()
    }
    liquidity_by_symbol = {
        symbol: _rolling_mean_dollar_volume_series(
            close_by_symbol[symbol],
            volume_by_symbol[symbol],
            window=config.liquidity_window,
        )
        for symbol in symbols
    }
    realized_vol_by_symbol = {
        symbol: _rolling_realized_vol(close_arr, config.volatility_window)
        for symbol, close_arr in close_by_symbol.items()
    }

    position_state = np.zeros(len(symbols), dtype=float)
    entry_price = np.full(len(symbols), np.nan, dtype=float)
    last_adjusted_scores: dict[str, float] = {}
    last_regimes: dict[str, str] = {}
    last_liquidity_threshold: float | None = None

    for idx in range(n):
        current_close_map = {symbol: float(close_by_symbol[symbol][idx]) for symbol in symbols}
        for s_idx, symbol in enumerate(symbols):
            if not np.isfinite(entry_price[s_idx]) or abs(position_state[s_idx]) <= 0.0:
                continue
            current_close = current_close_map[symbol]
            if not np.isfinite(current_close) or current_close <= 0.0:
                continue
            if (
                position_state[s_idx] > 0.0
                and current_close <= entry_price[s_idx] * (1.0 - config.stop_loss_pct)
            ) or (
                position_state[s_idx] < 0.0
                and current_close >= entry_price[s_idx] * (1.0 + config.stop_loss_pct)
            ):
                position_state[s_idx] = 0.0
                entry_price[s_idx] = np.nan

        if idx % config.rebalance_bars == 0:
            (
                last_adjusted_scores,
                last_regimes,
                last_liquidity_threshold,
            ) = _rebalance_last_day_liquidity_positions(
                idx=idx,
                symbols=symbols,
                close_by_symbol=close_by_symbol,
                momentum_by_symbol=momentum_by_symbol,
                liquidity_by_symbol=liquidity_by_symbol,
                realized_vol_by_symbol=realized_vol_by_symbol,
                position_state=position_state,
                entry_price=entry_price,
                config=config,
            )

        exposures[:, idx] = position_state

    meta["cross_sectional"] = True
    meta["liquidity_conditioned"] = True
    meta["signal_family"] = "last_day_liquidity_regime"
    meta["liquidity_quantile"] = config.liquidity_quantile
    meta["illiquid_reversal"] = config.illiquid_reversal
    meta["last_liquidity_threshold"] = last_liquidity_threshold
    if last_adjusted_scores:
        meta["last_adjusted_scores"] = dict(last_adjusted_scores)
    if last_regimes:
        meta["last_liquidity_regimes"] = dict(last_regimes)


@dataclass(frozen=True, slots=True)
class _AbnormalReturnContinuationConfig:
    return_z_window: int
    entry_z: float
    exit_z: float
    hold_bars: int
    stop_loss_pct: float
    allow_short: bool


def _resolve_abnormal_return_continuation_config(
    params: Mapping[str, Any],
) -> _AbnormalReturnContinuationConfig:
    return _AbnormalReturnContinuationConfig(
        return_z_window=max(4, int(params.get("return_z_window", 20))),
        entry_z=max(0.1, float(params.get("entry_z", 1.5))),
        exit_z=max(0.0, float(params.get("exit_z", 0.25))),
        hold_bars=max(1, int(params.get("hold_bars", 2))),
        stop_loss_pct=max(0.0, float(params.get("stop_loss_pct", 0.06))),
        allow_short=bool(params.get("allow_short", True)),
    )


def _abnormal_return_continuation_position_series(
    *,
    close: np.ndarray,
    config: _AbnormalReturnContinuationConfig,
) -> np.ndarray:
    returns = _returns_from_close(close)
    zscore = np.nan_to_num(_rolling_z(returns, config.return_z_window), nan=0.0)
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    bars_held = 0

    for idx in range(close.size):
        price = float(close[idx])
        z_i = float(zscore[idx]) if np.isfinite(zscore[idx]) else 0.0

        if mode == 1:
            bars_held += 1
            should_exit = (
                bars_held >= config.hold_bars
                or z_i <= config.exit_z
                or (entry_price is not None and price <= entry_price * (1.0 - config.stop_loss_pct))
            )
            if should_exit:
                mode = 0
                entry_price = None
                bars_held = 0
                position[idx] = 0.0
                continue
            position[idx] = 1.0
            continue

        if mode == -1:
            bars_held += 1
            should_exit = (
                bars_held >= config.hold_bars
                or z_i >= -config.exit_z
                or (entry_price is not None and price >= entry_price * (1.0 + config.stop_loss_pct))
            )
            if should_exit:
                mode = 0
                entry_price = None
                bars_held = 0
                position[idx] = 0.0
                continue
            position[idx] = -1.0
            continue

        if z_i >= config.entry_z:
            mode = 1
            entry_price = price
            bars_held = 0
            position[idx] = 1.0
            continue
        if config.allow_short and z_i <= -config.entry_z:
            mode = -1
            entry_price = price
            bars_held = 0
            position[idx] = -1.0
            continue

    return position


def _apply_abnormal_return_continuation_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_abnormal_return_continuation_config(params)
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        exposures[s_idx] = _abnormal_return_continuation_position_series(close=close, config=config)
    meta["event_alpha"] = True
    meta["signal_family"] = "abnormal_return_continuation"
    meta["hold_bars"] = config.hold_bars
    meta["entry_z"] = config.entry_z
    meta["allow_short"] = config.allow_short


@dataclass(frozen=True, slots=True)
class _MeanReversionStdConfig:
    window: int
    entry_z: float
    exit_z: float
    stop_loss_pct: float
    allow_short: bool
    residualize_btc: bool
    btc_symbol: str


def _resolve_mean_reversion_std_config(
    params: Mapping[str, Any],
    symbols: Sequence[str],
) -> _MeanReversionStdConfig:
    return _MeanReversionStdConfig(
        window=max(8, int(params.get("window", 64))),
        entry_z=float(params.get("entry_z", 2.0)),
        exit_z=max(0.0, float(params.get("exit_z", 0.5))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.03)),
        allow_short=bool(params.get("allow_short", True)),
        residualize_btc=bool(params.get("residualize_btc", False)),
        btc_symbol=_resolve_strategy_anchor_symbol(
            params.get("btc_symbol"),
            symbols,
            default="BTC/USDT",
        ),
    )


def _mean_reversion_position_series(
    *,
    close: np.ndarray,
    signal_series: np.ndarray,
    window: int,
    entry_z: float,
    exit_z: float,
    stop_loss_pct: float,
    allow_short: bool,
) -> np.ndarray:
    zscore = np.nan_to_num(_rolling_z(signal_series, window), nan=0.0)
    pos = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None

    for idx in range(close.size):
        step = _mean_reversion_step_input(
            idx=idx,
            close=close,
            zscore=zscore,
        )
        mode, entry_price, pos[idx] = _mean_reversion_step(
            mode=mode,
            entry_price=entry_price,
            step=step,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_loss_pct=stop_loss_pct,
            allow_short=allow_short,
        )
    return pos


@dataclass(frozen=True, slots=True)
class _MeanReversionStepInput:
    close_i: float
    z_i: float


def _mean_reversion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    zscore: np.ndarray,
) -> _MeanReversionStepInput:
    return _MeanReversionStepInput(
        close_i=float(close[idx]),
        z_i=float(zscore[idx]),
    )


def _mean_reversion_step(
    *,
    mode: int,
    entry_price: float | None,
    step: _MeanReversionStepInput,
    entry_z: float,
    exit_z: float,
    stop_loss_pct: float,
    allow_short: bool,
) -> tuple[int, float | None, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, 0.0

    if mode == 1 and entry_price is not None:
        stop_hit = step.close_i <= entry_price * (1.0 - stop_loss_pct)
        revert_hit = step.z_i >= -exit_z
        if stop_hit or revert_hit:
            return 0, None, 0.0
        return 1, entry_price, 1.0

    if mode == -1 and entry_price is not None:
        stop_hit = step.close_i >= entry_price * (1.0 + stop_loss_pct)
        revert_hit = step.z_i <= exit_z
        if stop_hit or revert_hit:
            return 0, None, 0.0
        return -1, entry_price, -1.0

    if mode == 0:
        if step.z_i <= -entry_z:
            return 1, step.close_i, 1.0
        if allow_short and step.z_i >= entry_z:
            return -1, step.close_i, -1.0
    return mode, entry_price, 0.0


def _apply_mean_reversion_std_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_mean_reversion_std_config(params, symbols)
    btc_close = None
    if config.residualize_btc and config.btc_symbol in symbols:
        btc_close = np.asarray(aligned[f"{config.btc_symbol}:close"], dtype=float)

    for s_idx, symbol in enumerate(symbols):
        if config.residualize_btc and symbol == config.btc_symbol:
            continue

        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        signal_series = close
        if btc_close is not None:
            signal_series = _beta_neutral_residual_series(close, btc_close, window=config.window)
        exposures[s_idx] = _mean_reversion_position_series(
            close=close,
            signal_series=signal_series,
            window=config.window,
            entry_z=config.entry_z,
            exit_z=config.exit_z,
            stop_loss_pct=config.stop_loss_pct,
            allow_short=config.allow_short,
        )

    if config.residualize_btc:
        meta["residualized_single_asset"] = True
        meta["residualize_btc"] = True
        meta["btc_symbol"] = config.btc_symbol


@dataclass(frozen=True, slots=True)
class _ShockReversionFadeConfig:
    volume_window: int
    range_window: int
    volume_shock_z: float
    range_shock_z: float
    return_shock_pct: float
    revert_fraction: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _ShockReversionStepInput:
    close_i: float
    ret_i: float
    shock_ok: bool


def _resolve_shock_reversion_fade_config(
    params: Mapping[str, Any],
    *,
    volume_window_default: int,
    range_window_default: int,
    volume_shock_z_default: float,
    range_shock_z_default: float,
    return_shock_pct_default: float,
    revert_fraction_default: float,
) -> _ShockReversionFadeConfig:
    return _ShockReversionFadeConfig(
        volume_window=max(8, int(params.get("volume_window", volume_window_default))),
        range_window=max(8, int(params.get("range_window", range_window_default))),
        volume_shock_z=float(params.get("volume_shock_z", volume_shock_z_default)),
        range_shock_z=float(params.get("range_shock_z", range_shock_z_default)),
        return_shock_pct=max(1e-6, float(params.get("return_shock_pct", return_shock_pct_default))),
        revert_fraction=min(
            0.95, max(0.10, float(params.get("revert_fraction", revert_fraction_default)))
        ),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 12))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _range_pct_from_prev_close(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    prev_close = np.r_[close[0], close[:-1]]
    return np.divide(
        high - low,
        np.clip(prev_close, 1e-12, np.inf),
        out=np.zeros_like(close),
        where=np.isfinite(prev_close),
    )


def _shock_reversion_support_series(
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    config: _ShockReversionFadeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns = _returns_from_close(close)
    range_pct = _range_pct_from_prev_close(close=close, high=high, low=low)
    vol_z = np.nan_to_num(_rolling_z(np.nan_to_num(volume, nan=0.0), config.volume_window), nan=0.0)
    range_z = np.nan_to_num(
        _rolling_z(np.nan_to_num(range_pct, nan=0.0), config.range_window), nan=0.0
    )
    return returns, vol_z, range_z


def _shock_reversion_position_series(
    *,
    close: np.ndarray,
    returns: np.ndarray,
    vol_z: np.ndarray,
    range_z: np.ndarray,
    config: _ShockReversionFadeConfig,
    entry_gate: np.ndarray | None = None,
) -> np.ndarray:
    pos = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    target_price: float | None = None
    hold_bars = 0
    gate = (
        np.ones(close.shape, dtype=bool)
        if entry_gate is None
        else np.asarray(entry_gate, dtype=bool)
    )

    for idx in range(close.size):
        step = _shock_reversion_step_input(
            idx=idx,
            close=close,
            returns=returns,
            vol_z=vol_z,
            range_z=range_z,
            gate=gate,
            config=config,
        )
        mode, entry_price, target_price, hold_bars, pos[idx] = _shock_reversion_step(
            mode=mode,
            entry_price=entry_price,
            target_price=target_price,
            hold_bars=hold_bars,
            step=step,
            config=config,
        )
    return pos


def _shock_reversion_step_input(
    *,
    idx: int,
    close: np.ndarray,
    returns: np.ndarray,
    vol_z: np.ndarray,
    range_z: np.ndarray,
    gate: np.ndarray,
    config: _ShockReversionFadeConfig,
) -> _ShockReversionStepInput:
    close_i = float(close[idx])
    ret_i = float(returns[idx]) if np.isfinite(returns[idx]) else 0.0
    shock_ok = (
        bool(gate[idx])
        and float(vol_z[idx]) >= config.volume_shock_z
        and float(range_z[idx]) >= config.range_shock_z
    )
    return _ShockReversionStepInput(
        close_i=close_i,
        ret_i=ret_i,
        shock_ok=shock_ok,
    )


def _shock_reversion_step(
    *,
    mode: int,
    entry_price: float | None,
    target_price: float | None,
    hold_bars: int,
    step: _ShockReversionStepInput,
    config: _ShockReversionFadeConfig,
) -> tuple[int, float | None, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, target_price, hold_bars, 0.0

    if mode == 1 and entry_price is not None and target_price is not None:
        next_hold_bars = hold_bars + 1
        stop_hit = step.close_i <= entry_price * (1.0 - config.stop_loss_pct)
        revert_hit = step.close_i >= target_price
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_hit or revert_hit or timeout_hit:
            return 0, None, None, 0, 0.0
        return 1, entry_price, target_price, next_hold_bars, 1.0

    if mode == -1 and entry_price is not None and target_price is not None:
        next_hold_bars = hold_bars + 1
        stop_hit = step.close_i >= entry_price * (1.0 + config.stop_loss_pct)
        revert_hit = step.close_i <= target_price
        timeout_hit = next_hold_bars >= config.max_hold_bars
        if stop_hit or revert_hit or timeout_hit:
            return 0, None, None, 0, 0.0
        return -1, entry_price, target_price, next_hold_bars, -1.0

    if mode == 0 and step.shock_ok:
        if step.ret_i <= -config.return_shock_pct:
            return (
                1,
                step.close_i,
                step.close_i * (1.0 + (abs(step.ret_i) * config.revert_fraction)),
                0,
                1.0,
            )
        if config.allow_short and step.ret_i >= config.return_shock_pct:
            return (
                -1,
                step.close_i,
                step.close_i * (1.0 - (abs(step.ret_i) * config.revert_fraction)),
                0,
                -1.0,
            )
    return mode, entry_price, target_price, hold_bars, 0.0


def _apply_liquidity_shock_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_shock_reversion_fade_config(
        params,
        volume_window_default=64,
        range_window_default=48,
        volume_shock_z_default=1.5,
        range_shock_z_default=1.0,
        return_shock_pct_default=0.01,
        revert_fraction_default=0.50,
    )
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        high = np.asarray(aligned[f"{symbol}:high"], dtype=float)
        low = np.asarray(aligned[f"{symbol}:low"], dtype=float)
        volume = np.asarray(aligned[f"{symbol}:volume"], dtype=float)
        returns, vol_z, range_z = _shock_reversion_support_series(
            close=close,
            high=high,
            low=low,
            volume=volume,
            config=config,
        )
        exposures[s_idx] = _shock_reversion_position_series(
            close=close,
            returns=returns,
            vol_z=vol_z,
            range_z=range_z,
            config=config,
        )


def _apply_session_liquidity_vacuum_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    config = _resolve_shock_reversion_fade_config(
        params,
        volume_window_default=48,
        range_window_default=36,
        volume_shock_z_default=1.0,
        range_shock_z_default=0.8,
        return_shock_pct_default=0.006,
        revert_fraction_default=0.40,
    )
    session_window_minutes = max(5, int(params.get("session_window_minutes", 30)))
    transition_minutes = np.asarray((0, 480, 780), dtype=int)
    minute_of_day = _minute_of_day(np.asarray(aligned["datetime"]))
    session_gate = np.any(
        np.abs(minute_of_day[:, None] - transition_minutes[None, :]) <= session_window_minutes,
        axis=1,
    )

    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        high = np.asarray(aligned[f"{symbol}:high"], dtype=float)
        low = np.asarray(aligned[f"{symbol}:low"], dtype=float)
        volume = np.asarray(aligned[f"{symbol}:volume"], dtype=float)
        returns, vol_z, range_z = _shock_reversion_support_series(
            close=close,
            high=high,
            low=low,
            volume=volume,
            config=config,
        )
        exposures[s_idx] = _shock_reversion_position_series(
            close=close,
            returns=returns,
            vol_z=vol_z,
            range_z=range_z,
            config=config,
            entry_gate=session_gate,
        )


@dataclass(frozen=True, slots=True)
class _FundingLiquidationCrowdingFadeConfig:
    window: int
    crowding_entry: float
    crowding_exit: float
    liquidation_z_min: float
    return_shock_pct: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


@dataclass(frozen=True, slots=True)
class _FundingLiquidationCrowdingStepInput:
    close_i: float
    score_i: float
    liq_i: float
    ret_i: float


def _resolve_funding_liquidation_crowding_fade_config(
    params: Mapping[str, Any],
) -> _FundingLiquidationCrowdingFadeConfig:
    return _FundingLiquidationCrowdingFadeConfig(
        window=int(params.get("window", 96)),
        crowding_entry=float(params.get("crowding_entry", 0.85)),
        crowding_exit=float(params.get("crowding_exit", 0.25)),
        liquidation_z_min=float(params.get("liquidation_z_min", 1.0)),
        return_shock_pct=float(params.get("return_shock_pct", 0.01)),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 12))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.02)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _funding_liquidation_crowding_position_series(
    *,
    close: np.ndarray,
    score: np.ndarray,
    liquidation_z: np.ndarray,
    config: _FundingLiquidationCrowdingFadeConfig,
) -> np.ndarray:
    ret = _returns_from_close(close)
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    bars_held = 0

    for idx in range(close.size):
        step = _funding_liquidation_crowding_step_input(
            idx=idx,
            close=close,
            score=score,
            liquidation_z=liquidation_z,
            returns=ret,
        )
        mode, entry_price, bars_held, position[idx] = _funding_liquidation_crowding_step(
            mode=mode,
            entry_price=entry_price,
            bars_held=bars_held,
            step=step,
            config=config,
        )
    return position


def _funding_liquidation_crowding_step_input(
    *,
    idx: int,
    close: np.ndarray,
    score: np.ndarray,
    liquidation_z: np.ndarray,
    returns: np.ndarray,
) -> _FundingLiquidationCrowdingStepInput:
    return _FundingLiquidationCrowdingStepInput(
        close_i=float(close[idx]),
        score_i=float(score[idx]) if np.isfinite(score[idx]) else np.nan,
        liq_i=float(liquidation_z[idx]) if np.isfinite(liquidation_z[idx]) else np.nan,
        ret_i=float(returns[idx]) if np.isfinite(returns[idx]) else 0.0,
    )


def _funding_liquidation_crowding_step(
    *,
    mode: int,
    entry_price: float | None,
    bars_held: int,
    step: _FundingLiquidationCrowdingStepInput,
    config: _FundingLiquidationCrowdingFadeConfig,
) -> tuple[int, float | None, int, float]:
    if not np.isfinite(step.close_i):
        return mode, entry_price, bars_held, float(mode)

    if mode != 0:
        next_bars_held = bars_held + 1
        if _funding_liquidation_crowding_should_exit(
            mode=mode,
            score_i=step.score_i,
            close_i=step.close_i,
            entry_price=entry_price,
            bars_held=next_bars_held,
            config=config,
        ):
            return 0, None, 0, 0.0
        return mode, entry_price, next_bars_held, float(mode)

    next_mode = _funding_liquidation_crowding_entry_mode(
        score_i=step.score_i,
        liq_i=step.liq_i,
        ret_i=step.ret_i,
        config=config,
    )
    if next_mode != 0:
        return next_mode, step.close_i, 0, float(next_mode)
    return mode, entry_price, bars_held, 0.0


def _funding_liquidation_crowding_should_exit(
    *,
    mode: int,
    score_i: float,
    close_i: float,
    entry_price: float | None,
    bars_held: int,
    config: _FundingLiquidationCrowdingFadeConfig,
) -> bool:
    if mode == 1:
        return (
            (np.isfinite(score_i) and score_i >= -config.crowding_exit)
            or (bars_held >= config.max_hold_bars)
            or (
                entry_price is not None
                and close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
            )
        )
    return (
        (np.isfinite(score_i) and score_i <= config.crowding_exit)
        or (bars_held >= config.max_hold_bars)
        or (
            entry_price is not None and close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
        )
    )


def _funding_liquidation_crowding_entry_mode(
    *,
    score_i: float,
    liq_i: float,
    ret_i: float,
    config: _FundingLiquidationCrowdingFadeConfig,
) -> int:
    if not (np.isfinite(score_i) and np.isfinite(liq_i)):
        return 0

    long_signal = (
        score_i <= -config.crowding_entry
        and liq_i <= -config.liquidation_z_min
        and ret_i <= -config.return_shock_pct
    )
    short_signal = (
        config.allow_short
        and score_i >= config.crowding_entry
        and liq_i >= config.liquidation_z_min
        and ret_i >= config.return_shock_pct
    )
    if long_signal:
        return 1
    if short_signal:
        return -1
    return 0


def _apply_funding_liquidation_crowding_fade_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_funding_liquidation_crowding_fade_config(params)
    missing_symbols: list[str] = []
    for s_idx, symbol in enumerate(symbols):
        support_inputs = _resolve_crowding_support_inputs(aligned=aligned, symbol=symbol)
        if support_inputs is None:
            missing_symbols.append(symbol)
            continue

        support = _crowding_support_series(
            funding_rate=support_inputs.funding_rate,
            open_interest=support_inputs.open_interest,
            mark_price=support_inputs.mark_price,
            index_price=support_inputs.index_price,
            liquidation_long_notional=support_inputs.liquidation_long_notional,
            liquidation_short_notional=support_inputs.liquidation_short_notional,
            window=config.window,
        )
        score = np.asarray(support["crowding_score"], dtype=float)
        liquidation_z = np.asarray(support["liquidation_imbalance_z"], dtype=float)
        exposures[s_idx] = _funding_liquidation_crowding_position_series(
            close=support_inputs.close,
            score=score,
            liquidation_z=liquidation_z,
            config=config,
        )
        _note_support_data_symbol(meta, symbol=symbol, values=score)

    _finalize_missing_support_symbols(meta, missing_symbols=missing_symbols)


def _apply_vol_compression_vwap_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    entry_z = float(params.get("entry_z", 1.5))
    exit_z = float(params.get("exit_z", 0.35))
    compression_pct = float(params.get("compression_percentile", 0.30))
    comp_vol_ratio = float(
        params.get("compression_vol_ratio", params.get("compression_threshold", 0.85))
    )

    for s_idx, symbol in enumerate(symbols):
        close = aligned[f"{symbol}:close"]
        high = aligned[f"{symbol}:high"]
        low = aligned[f"{symbol}:low"]
        volume = aligned[f"{symbol}:volume"]

        dev_z = np.nan_to_num(
            _vwap_dev_z(high, low, close, volume, window=60, z_window=120),
            nan=0.0,
        )
        vr = np.nan_to_num(_vol_ratio_series(close, 12, 60), nan=0.0)
        bw = np.nan_to_num(_rolling_z(np.abs(_returns_from_close(close)), 64), nan=0.0)
        compression = (vr <= comp_vol_ratio) & (bw <= _normal_cdf(compression_pct) * 2.0)

        pos = np.zeros(close.shape, dtype=float)
        pos = np.where(compression & (dev_z <= -entry_z), 1.0, pos)
        pos = np.where(compression & (dev_z >= entry_z), -1.0, pos)
        pos = np.where(np.abs(dev_z) <= exit_z, 0.0, pos)
        exposures[s_idx] = pos


def _apply_leadlag_spillover_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    leader_symbols = [symbol for symbol in symbols if symbol in _LEADERS]
    laggards = [symbol for symbol in symbols if symbol not in _LEADERS and symbol not in _METALS]
    entry_score = float(params.get("entry_score", params.get("entry_spillover", 0.35)))
    exit_score = float(params.get("exit_score", params.get("exit_spillover", 0.08)))
    lag_order = max(1, int(params.get("max_lag", 3)))

    if not leader_symbols or not laggards:
        return

    n = exposures.shape[1]
    leader_ret = np.zeros((len(leader_symbols), n), dtype=float)
    for idx, sym in enumerate(leader_symbols):
        leader_ret[idx] = _returns_from_close(aligned[f"{sym}:close"])
    leader_mean = np.mean(leader_ret, axis=0)

    decay = np.exp(-np.arange(1, lag_order + 1, dtype=float))
    decay /= np.sum(decay)

    pred = np.zeros(n, dtype=float)
    for lag in range(1, lag_order + 1):
        # No wrap-around: ``np.roll`` would leak the LAST ``lag`` bars of
        # leader returns into the window head.  Bars without a full lag of
        # history contribute zero.
        shifted = np.zeros(n, dtype=float)
        if lag < n:
            shifted[lag:] = leader_mean[:-lag]
        pred += decay[lag - 1] * shifted

    for symbol in laggards:
        s_idx = symbols.index(symbol)
        follower_ret = _returns_from_close(aligned[f"{symbol}:close"])
        # Trailing (expanding) sigma per bar — the previous full-sample
        # ``_safe_std`` normalised every bar's score by FUTURE volatility.
        sigma = _expanding_std(follower_ret)
        score = np.divide(pred, sigma, out=np.zeros(n, dtype=float), where=sigma > 1e-12)
        pos = np.where(
            score >= entry_score,
            1.0,
            np.where(score <= -entry_score, -1.0, 0.0),
        )
        pos = np.where(np.abs(score) <= exit_score, 0.0, pos)
        exposures[s_idx] = pos


def _apply_session_gated_residual_basket_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    session_window_minutes = max(5, int(params.get("session_window_minutes", 30)))
    transition_minutes = np.asarray((0, 480, 780), dtype=int)
    minute_of_day = _minute_of_day(np.asarray(aligned["datetime"]))
    session_gate = np.any(
        np.abs(minute_of_day[:, None] - transition_minutes[None, :]) <= session_window_minutes,
        axis=1,
    )
    _apply_residual_basket_reversion_strategy(
        params=params,
        aligned=aligned,
        symbols=symbols,
        exposures=exposures,
        meta=meta,
        entry_gate=session_gate,
        session_gated=True,
    )


def _apply_volatility_regime_residual_basket_reversion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_volatility_regime_residual_basket_reversion_config(params)
    btc_symbol = canonical_symbol(str(params.get("btc_symbol") or "BTC/USDT"))
    if btc_symbol not in symbols:
        btc_symbol = canonical_symbol(str(symbols[0]))

    btc_close = np.asarray(aligned[f"{btc_symbol}:close"], dtype=float)
    close_map_np = {
        symbol: np.asarray(aligned[f"{symbol}:close"], dtype=float) for symbol in symbols
    }
    vol_ratio = np.nan_to_num(
        _vol_ratio_series(btc_close, config.btc_vol_fast, config.btc_vol_slow),
        nan=0.0,
    )
    dispersion = np.nan_to_num(_cross_sectional_return_dispersion(close_map_np), nan=0.0)
    entry_gate = (
        (vol_ratio > 0.0)
        & (vol_ratio <= config.btc_vol_ratio_cap)
        & (dispersion >= config.dispersion_floor)
    )

    _apply_residual_basket_reversion_strategy(
        params=params,
        aligned=aligned,
        symbols=symbols,
        exposures=exposures,
        meta=meta,
        entry_gate=entry_gate,
        session_gated=False,
    )
    meta["volatility_regime_gated"] = True
    meta["btc_vol_fast"] = int(config.btc_vol_fast)
    meta["btc_vol_slow"] = int(config.btc_vol_slow)
    meta["btc_vol_ratio_cap"] = float(config.btc_vol_ratio_cap)
    meta["dispersion_floor"] = float(config.dispersion_floor)


def _apply_pair_spread_strategy(
    *,
    strategy_class: str,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    n: int,
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    symbol_x, symbol_y, x_idx, y_idx = _resolve_symbol_pair(symbols, params)

    try:
        simulated = _simulate_event_driven_strategy_exposures(
            _load_event_driven_strategy_impl(strategy_class),
            params=params,
            aligned=aligned,
            symbols=(symbol_x, symbol_y),
        )
        exposures[x_idx] = simulated[0]
        exposures[y_idx] = simulated[1]
        meta["event_driven_proxy"] = True
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        if meta.get("_strict_actual_engine") is True:
            raise
        entry_z = float(params.get("entry_z", 2.0))
        exit_z = float(params.get("exit_z", 0.35))
        lookback = int(params.get("lookback_window", 96))
        x_pos, y_pos = _pair_spread_fallback_exposures(
            aligned=aligned,
            symbol_x=symbol_x,
            symbol_y=symbol_y,
            length=n,
            entry_z=entry_z,
            exit_z=exit_z,
            lookback=lookback,
        )
        exposures[x_idx] = x_pos
        exposures[y_idx] = y_pos
        meta["event_driven_proxy"] = False
        meta["event_driven_proxy_error"] = str(exc)


def _lagged_return_matrix(close_matrix: np.ndarray, lookback: int) -> np.ndarray:
    lookback_i = max(1, int(lookback))
    out = np.full(close_matrix.shape, np.nan, dtype=float)
    if close_matrix.shape[1] <= lookback_i:
        return out
    latest = close_matrix[:, lookback_i:]
    base = close_matrix[:, : close_matrix.shape[1] - lookback_i]
    valid = np.isfinite(latest) & np.isfinite(base) & (base > 0.0)
    out[:, lookback_i:] = np.where(
        valid,
        np.divide(
            latest,
            base,
            out=np.full_like(latest, np.nan, dtype=float),
            where=valid,
        )
        - 1.0,
        np.nan,
    )
    return out


def _target_indices_from_score(
    *,
    score_column: np.ndarray,
    valid_indices: np.ndarray,
    signal_threshold: float,
    max_longs: int,
    max_shorts: int,
    allow_short: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if valid_indices.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    ordered = valid_indices[np.argsort(score_column[valid_indices], kind="stable")]
    long_indices = ordered[score_column[ordered] >= signal_threshold]
    short_indices = ordered[score_column[ordered] <= -signal_threshold]
    if long_indices.size > 0 and max_longs > 0:
        long_indices = long_indices[::-1][:max_longs]
    else:
        long_indices = np.empty(0, dtype=np.int64)
    if not allow_short or short_indices.size == 0 or max_shorts <= 0:
        return long_indices, np.empty(0, dtype=np.int64)
    short_indices = short_indices[:max_shorts]
    if long_indices.size == 0:
        return long_indices, short_indices
    return long_indices, short_indices[~np.isin(short_indices, long_indices, assume_unique=False)]


def _apply_rebalanced_cross_sectional_targets(
    *,
    score_matrix: np.ndarray,
    valid_mask: np.ndarray,
    close_matrix: np.ndarray,
    exposures: np.ndarray,
    rebalance_bars: int,
    signal_threshold: float,
    max_longs: int,
    max_shorts: int,
    allow_short: bool,
    max_abs_exposure: float,
    stop_loss_pct: float = 0.0,
    long_scale: np.ndarray | None = None,
    short_scale: np.ndarray | None = None,
) -> None:
    symbol_count, bar_count = score_matrix.shape
    position_state = np.zeros(symbol_count, dtype=float)
    entry_price = np.full(symbol_count, np.nan, dtype=float)
    rebalance_due = (np.arange(bar_count, dtype=np.int64) + 1) % max(1, int(rebalance_bars)) == 0
    gross = max(0.0, float(max_abs_exposure))

    for idx in range(bar_count):
        close_column = close_matrix[:, idx]
        if stop_loss_pct > 0.0:
            active = (
                (position_state != 0.0)
                & np.isfinite(entry_price)
                & np.isfinite(close_column)
                & (close_column > 0.0)
            )
            exit_mask = np.zeros(symbol_count, dtype=bool)
            exit_mask |= (
                (position_state > 0.0)
                & active
                & (close_column <= (entry_price * (1.0 - stop_loss_pct)))
            )
            exit_mask |= (
                (position_state < 0.0)
                & active
                & (close_column >= (entry_price * (1.0 + stop_loss_pct)))
            )
            if np.any(exit_mask):
                position_state[exit_mask] = 0.0
                entry_price[exit_mask] = np.nan

        if rebalance_due[idx]:
            valid_indices = np.flatnonzero(valid_mask[:, idx])
            long_indices, short_indices = _target_indices_from_score(
                score_column=score_matrix[:, idx],
                valid_indices=valid_indices,
                signal_threshold=float(signal_threshold),
                max_longs=int(max_longs),
                max_shorts=int(max_shorts),
                allow_short=bool(allow_short),
            )
            next_state = np.zeros(symbol_count, dtype=float)
            if long_indices.size > 0:
                scale = (
                    np.ones(long_indices.size, dtype=float)
                    if long_scale is None
                    else np.asarray(long_scale[long_indices, idx], dtype=float)
                )
                next_state[long_indices] = np.clip(scale, 0.0, gross)
            if short_indices.size > 0:
                scale = (
                    np.ones(short_indices.size, dtype=float)
                    if short_scale is None
                    else np.asarray(short_scale[short_indices, idx], dtype=float)
                )
                next_state[short_indices] = -np.clip(scale, 0.0, gross)
            changed = next_state != position_state
            position_state = next_state
            fresh = changed & (position_state != 0.0) & np.isfinite(close_column)
            entry_price[fresh] = close_column[fresh]
            entry_price[position_state == 0.0] = np.nan

        exposures[:, idx] = position_state


def _apply_funding_dislocation_trend_carry_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    close_matrix = np.vstack(
        [np.asarray(aligned[f"{symbol}:close"], dtype=float) for symbol in symbols]
    )
    symbol_count, bar_count = close_matrix.shape
    fast = _lagged_return_matrix(close_matrix, int(params.get("fast_lookback_bars", 24)))
    mid = _lagged_return_matrix(close_matrix, int(params.get("mid_lookback_bars", 72)))
    slow = _lagged_return_matrix(close_matrix, int(params.get("slow_lookback_bars", 168)))
    trend_stack = np.stack([fast, mid, slow])
    trend_finite = np.isfinite(trend_stack)
    trend_counts = trend_finite.sum(axis=0)
    trend_raw = np.divide(
        np.where(trend_finite, trend_stack, 0.0).sum(axis=0),
        np.clip(trend_counts, 1, np.inf),
        out=np.full(close_matrix.shape, np.nan, dtype=float),
        where=trend_counts > 0,
    )
    valid_mask = np.isfinite(trend_raw)

    funding_z = np.zeros((symbol_count, bar_count), dtype=float)
    basis_z = np.zeros((symbol_count, bar_count), dtype=float)
    crowding_penalty = np.zeros((symbol_count, bar_count), dtype=float)
    missing_symbols: list[str] = []
    for s_idx, symbol in enumerate(symbols):
        support_inputs = _resolve_crowding_support_inputs(aligned=aligned, symbol=symbol)
        if support_inputs is None:
            missing_symbols.append(symbol)
            valid_mask[s_idx] = False
            continue
        support = _crowding_support_series(
            funding_rate=support_inputs.funding_rate,
            open_interest=support_inputs.open_interest,
            mark_price=support_inputs.mark_price,
            index_price=support_inputs.index_price,
            liquidation_long_notional=support_inputs.liquidation_long_notional,
            liquidation_short_notional=support_inputs.liquidation_short_notional,
            window=max(12, int(params.get("crowding_window", 72))),
        )
        funding_z[s_idx] = np.asarray(support["funding_z"], dtype=float)
        basis_z[s_idx] = np.asarray(support["basis_z"], dtype=float)
        crowding = np.asarray(support["crowding_score"], dtype=float)
        crowding_penalty[s_idx] = np.abs(np.nan_to_num(crowding, nan=0.0))
        valid_mask[s_idx] &= np.isfinite(funding_z[s_idx]) & np.isfinite(basis_z[s_idx])
        _note_support_data_symbol(meta, symbol=symbol, values=np.asarray(crowding, dtype=float))
    _finalize_missing_support_symbols(meta, missing_symbols=missing_symbols)

    trend_score = _cross_sectional_score_matrix(np.nan_to_num(trend_raw, nan=0.0), valid_mask)
    carry_score = _cross_sectional_score_matrix(-np.nan_to_num(funding_z, nan=0.0), valid_mask)
    basis_score = _cross_sectional_score_matrix(-np.nan_to_num(basis_z, nan=0.0), valid_mask)
    penalty_score = _cross_sectional_score_matrix(
        -np.nan_to_num(crowding_penalty, nan=0.0),
        valid_mask,
    )
    score_matrix = (
        float(params.get("trend_weight", 0.55)) * trend_score
        + float(params.get("carry_weight", 0.25)) * carry_score
        + float(params.get("basis_weight", 0.10)) * basis_score
        + float(params.get("crowding_penalty_weight", 0.10)) * penalty_score
    )
    _apply_rebalanced_cross_sectional_targets(
        score_matrix=score_matrix,
        valid_mask=valid_mask,
        close_matrix=close_matrix,
        exposures=exposures,
        rebalance_bars=int(params.get("rebalance_bars", 4)),
        signal_threshold=float(params.get("signal_threshold", 0.45)),
        max_longs=int(params.get("max_longs", 3)),
        max_shorts=int(params.get("max_shorts", 2)),
        allow_short=bool(params.get("allow_short", True)),
        max_abs_exposure=float(params.get("max_abs_exposure", 1.0)),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.045)),
    )
    meta["deep_research_leaf"] = "funding_dislocation_trend_carry"


def _rolling_realized_vol_for_deep_research(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size < 3:
        return out
    returns = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr[1:]) & np.isfinite(arr[:-1]) & (arr[1:] > 0.0) & (arr[:-1] > 0.0)
    returns[1:] = np.where(valid, np.log(arr[1:] / arr[:-1]), np.nan)
    win = max(4, int(window))
    for idx in range(win, arr.size + 1):
        tail = returns[idx - win + 1 : idx]
        finite = tail[np.isfinite(tail)]
        if finite.size < max(2, win // 2):
            continue
        out[idx - 1] = float(np.std(finite, ddof=1))
    return out


def _apply_vol_managed_momentum_crash_gate_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    close_matrix = np.vstack(
        [np.asarray(aligned[f"{symbol}:close"], dtype=float) for symbol in symbols]
    )
    momentum = _lagged_return_matrix(close_matrix, int(params.get("momentum_lookback_bars", 96)))
    valid_mask = np.isfinite(momentum)
    score_matrix = _cross_sectional_score_matrix(np.nan_to_num(momentum, nan=0.0), valid_mask)
    realized_vol = np.vstack(
        [
            _rolling_realized_vol_for_deep_research(
                close_matrix[s_idx],
                int(params.get("vol_window", 48)),
            )
            for s_idx in range(len(symbols))
        ]
    )
    target_vol = max(0.0, float(params.get("target_vol", 0.018)))
    max_leverage = max(0.0, float(params.get("max_leverage", 1.0)))
    vol_scale = np.divide(
        target_vol,
        np.clip(realized_vol, 1e-8, np.inf),
        out=np.zeros_like(realized_vol, dtype=float),
        where=np.isfinite(realized_vol),
    )
    vol_scale = np.clip(np.nan_to_num(vol_scale, nan=0.0, posinf=max_leverage), 0.0, max_leverage)

    benchmark_symbol = "BTC/USDT" if "BTC/USDT" in symbols else symbols[0]
    benchmark = close_matrix[symbols.index(benchmark_symbol)]
    crash_window = max(1, int(params.get("crash_window_bars", 24)))
    benchmark_ret = np.full(benchmark.shape, np.nan, dtype=float)
    if benchmark.size > crash_window:
        benchmark_ret[crash_window:] = (
            benchmark[crash_window:] / np.clip(benchmark[:-crash_window], 1e-12, np.inf)
        ) - 1.0
    fast_vol = _rolling_realized_vol_for_deep_research(
        benchmark,
        int(params.get("vol_window", 48)),
    )
    slow_vol = _rolling_realized_vol_for_deep_research(
        benchmark,
        int(params.get("vol_ratio_window", 192)),
    )
    vol_ratio = np.divide(
        fast_vol,
        np.clip(slow_vol, 1e-8, np.inf),
        out=np.zeros_like(fast_vol, dtype=float),
        where=np.isfinite(fast_vol) & np.isfinite(slow_vol),
    )
    stress = (benchmark_ret <= -float(params.get("crash_return_pct", 0.055))) | (
        vol_ratio >= float(params.get("vol_ratio_max", 2.4))
    )
    stress_reduce = np.clip(float(params.get("stress_reduce", 0.25)), 0.0, 1.0)
    long_scale = vol_scale * np.where(stress[np.newaxis, :], stress_reduce, 1.0)
    short_scale = vol_scale
    _apply_rebalanced_cross_sectional_targets(
        score_matrix=score_matrix,
        valid_mask=valid_mask,
        close_matrix=close_matrix,
        exposures=exposures,
        rebalance_bars=int(params.get("rebalance_bars", 4)),
        signal_threshold=float(params.get("signal_threshold", 0.35)),
        max_longs=int(params.get("max_longs", 3)),
        max_shorts=int(params.get("max_shorts", 2)),
        allow_short=bool(params.get("allow_short", True)),
        max_abs_exposure=max_leverage,
        long_scale=long_scale,
        short_scale=short_scale,
    )
    meta["deep_research_leaf"] = "vol_managed_momentum_crash_gate"
    meta["support_data_symbols"] = list(symbols)


@dataclass(frozen=True, slots=True)
class _FlowImbalanceLiquidationSweepConfig:
    window: int
    entry_score: float
    exit_score: float
    liquidation_z_min: float
    return_shock_pct: float
    max_spread_bps: float
    max_hold_bars: int
    stop_loss_pct: float
    allow_short: bool


def _resolve_flow_imbalance_liquidation_sweep_config(
    params: Mapping[str, Any],
) -> _FlowImbalanceLiquidationSweepConfig:
    return _FlowImbalanceLiquidationSweepConfig(
        window=max(8, int(params.get("window", 72))),
        entry_score=float(params.get("entry_score", 0.35)),
        exit_score=float(params.get("exit_score", 0.08)),
        liquidation_z_min=float(params.get("liquidation_z_min", 1.2)),
        return_shock_pct=float(params.get("return_shock_pct", 0.006)),
        max_spread_bps=float(params.get("max_spread_bps", 12.0)),
        max_hold_bars=max(1, int(params.get("max_hold_bars", 18))),
        stop_loss_pct=float(params.get("stop_loss_pct", 0.016)),
        allow_short=bool(params.get("allow_short", True)),
    )


def _flow_imbalance_series(
    *,
    aligned: Mapping[str, np.ndarray],
    symbol: str,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    buy_quote = aligned.get(f"{symbol}:taker_buy_quote_volume")
    sell_quote = aligned.get(f"{symbol}:taker_sell_quote_volume")
    if buy_quote is None or sell_quote is None:
        return None
    buy_arr = np.nan_to_num(np.asarray(buy_quote, dtype=float), nan=0.0)
    sell_arr = np.nan_to_num(np.asarray(sell_quote, dtype=float), nan=0.0)
    taker_imbalance = np.divide(
        buy_arr - sell_arr,
        np.abs(buy_arr) + np.abs(sell_arr) + 1e-12,
        out=np.zeros(close.shape, dtype=float),
    )

    depth = aligned.get(f"{symbol}:book_depth_imbalance_1pct")
    if depth is not None:
        book_imbalance = np.nan_to_num(np.asarray(depth, dtype=float), nan=0.0)
    else:
        bid_qty = aligned.get(f"{symbol}:best_bid_quantity")
        ask_qty = aligned.get(f"{symbol}:best_ask_quantity")
        if bid_qty is None or ask_qty is None:
            return None
        bid_arr = np.nan_to_num(np.asarray(bid_qty, dtype=float), nan=0.0)
        ask_arr = np.nan_to_num(np.asarray(ask_qty, dtype=float), nan=0.0)
        book_imbalance = np.divide(
            bid_arr - ask_arr,
            np.abs(bid_arr) + np.abs(ask_arr) + 1e-12,
            out=np.zeros(close.shape, dtype=float),
        )

    spread = aligned.get(f"{symbol}:bbo_spread_bps")
    if spread is not None:
        spread_bps = np.asarray(spread, dtype=float)
    else:
        bid = aligned.get(f"{symbol}:best_bid_price")
        ask = aligned.get(f"{symbol}:best_ask_price")
        if bid is None or ask is None:
            spread_bps = np.zeros(close.shape, dtype=float)
        else:
            bid_arr = np.asarray(bid, dtype=float)
            ask_arr = np.asarray(ask, dtype=float)
            mid = (bid_arr + ask_arr) * 0.5
            spread_bps = (
                np.divide(
                    ask_arr - bid_arr,
                    np.clip(mid, 1e-12, np.inf),
                    out=np.zeros(close.shape, dtype=float),
                    where=np.isfinite(mid),
                )
                * 10_000.0
            )

    flow_score = (0.60 * taker_imbalance) + (0.40 * book_imbalance)
    return np.nan_to_num(flow_score, nan=0.0), np.nan_to_num(spread_bps, nan=np.inf)


def _flow_imbalance_liquidation_position_series(
    *,
    close: np.ndarray,
    flow_score: np.ndarray,
    spread_bps: np.ndarray,
    liquidation_long_z: np.ndarray,
    liquidation_short_z: np.ndarray,
    config: _FlowImbalanceLiquidationSweepConfig,
) -> np.ndarray:
    returns = _returns_from_close(close)
    position = np.zeros(close.shape, dtype=float)
    mode = 0
    entry_price: float | None = None
    bars_held = 0
    for idx in range(close.size):
        close_i = float(close[idx])
        if not np.isfinite(close_i):
            position[idx] = float(mode)
            continue
        if mode != 0:
            bars_held += 1
            exit_long = mode == 1 and (
                float(flow_score[idx]) <= config.exit_score
                or bars_held >= config.max_hold_bars
                or (
                    entry_price is not None
                    and close_i <= float(entry_price) * (1.0 - config.stop_loss_pct)
                )
            )
            exit_short = mode == -1 and (
                float(flow_score[idx]) >= -config.exit_score
                or bars_held >= config.max_hold_bars
                or (
                    entry_price is not None
                    and close_i >= float(entry_price) * (1.0 + config.stop_loss_pct)
                )
            )
            if exit_long or exit_short:
                mode = 0
                entry_price = None
                bars_held = 0
            position[idx] = float(mode)
            continue

        spread_ok = np.isfinite(spread_bps[idx]) and float(spread_bps[idx]) <= config.max_spread_bps
        if not spread_ok:
            position[idx] = 0.0
            continue
        long_signal = (
            float(liquidation_long_z[idx]) >= config.liquidation_z_min
            and float(returns[idx]) <= -config.return_shock_pct
            and float(flow_score[idx]) >= config.entry_score
        )
        short_signal = (
            config.allow_short
            and float(liquidation_short_z[idx]) >= config.liquidation_z_min
            and float(returns[idx]) >= config.return_shock_pct
            and float(flow_score[idx]) <= -config.entry_score
        )
        if long_signal:
            mode = 1
            entry_price = close_i
        elif short_signal:
            mode = -1
            entry_price = close_i
        position[idx] = float(mode)
    return position


def _apply_flow_imbalance_liquidation_sweep_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    config = _resolve_flow_imbalance_liquidation_sweep_config(params)
    missing_symbols: list[str] = []
    for s_idx, symbol in enumerate(symbols):
        close = np.asarray(aligned[f"{symbol}:close"], dtype=float)
        long_liq = aligned.get(f"{symbol}:liquidation_long_notional")
        short_liq = aligned.get(f"{symbol}:liquidation_short_notional")
        flow_payload = _flow_imbalance_series(aligned=aligned, symbol=symbol, close=close)
        if long_liq is None or short_liq is None or flow_payload is None:
            missing_symbols.append(symbol)
            continue
        flow_score, spread_bps = flow_payload
        liquidation_long_z = _rolling_z(
            np.nan_to_num(np.asarray(long_liq, dtype=float), nan=0.0),
            config.window,
        )
        liquidation_short_z = _rolling_z(
            np.nan_to_num(np.asarray(short_liq, dtype=float), nan=0.0),
            config.window,
        )
        exposures[s_idx] = _flow_imbalance_liquidation_position_series(
            close=close,
            flow_score=flow_score,
            spread_bps=spread_bps,
            liquidation_long_z=np.nan_to_num(liquidation_long_z, nan=0.0),
            liquidation_short_z=np.nan_to_num(liquidation_short_z, nan=0.0),
            config=config,
        )
        _note_support_data_symbol(meta, symbol=symbol, values=flow_score)
    _finalize_missing_support_symbols(meta, missing_symbols=missing_symbols)
    meta["deep_research_leaf"] = "flow_imbalance_liquidation_sweep"


def _apply_micro_range_expansion_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
) -> None:
    lookback = max(8, int(params.get("lookback", 30)))
    range_z_threshold = float(params.get("range_z_threshold", 1.5))
    volume_z_threshold = float(params.get("volume_z_threshold", 1.0))

    for s_idx, symbol in enumerate(symbols):
        high = aligned[f"{symbol}:high"]
        low = aligned[f"{symbol}:low"]
        close = aligned[f"{symbol}:close"]
        volume = aligned[f"{symbol}:volume"]
        range_pct = np.divide(high - low, np.clip(close, 1e-12, np.inf))
        range_z = np.nan_to_num(_rolling_z(range_pct, lookback), nan=0.0)
        vol_z = np.nan_to_num(_rolling_z(volume, lookback), nan=0.0)
        ret = _returns_from_close(close)
        breakout = np.where(ret >= 0.0, 1.0, -1.0)
        active = (range_z >= range_z_threshold) & (vol_z >= volume_z_threshold)
        exposures[s_idx] = np.where(active, breakout, 0.0)


def _apply_alpha101_formula_strategy(
    *,
    params: Mapping[str, Any],
    aligned: Mapping[str, np.ndarray],
    symbols: Sequence[str],
    exposures: np.ndarray,
    meta: dict[str, Any],
) -> None:
    simulated = _simulate_event_driven_strategy_exposures(
        _load_event_driven_strategy_impl("Alpha101FormulaStrategy"),
        params=params,
        aligned=aligned,
        symbols=symbols,
    )
    exposures[:] = simulated
    meta["event_driven_proxy"] = True
    meta["formulaic_alpha101"] = True
    meta["alpha_id"] = int(params.get("alpha_id", 101))
    meta["alpha_param_override_count"] = len(
        params.get("alpha_param_overrides")
        if isinstance(params.get("alpha_param_overrides"), Mapping)
        else {}
    )


def _wrap_strategy_handler(
    func,
    *,
    include_n: bool = False,
    include_meta: bool = False,
    extra_kwargs: Mapping[str, Any] | None = None,
):
    extras = dict(extra_kwargs or {})

    def _handler(
        params: dict[str, Any],
        aligned: dict[str, np.ndarray],
        symbols: Sequence[str],
        n: int,
        exposures: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        kwargs = {
            "params": params,
            "aligned": aligned,
            "symbols": symbols,
            "exposures": exposures,
            **extras,
        }
        if include_n:
            kwargs["n"] = n
        if include_meta:
            kwargs["meta"] = meta
        func(**kwargs)

    return _handler


_STRATEGY_SIGNAL_DISPATCHER = StrategySignalDispatcher(
    handlers={
        "Alpha101FormulaStrategy": _wrap_strategy_handler(
            _apply_alpha101_formula_strategy,
            include_meta=True,
        ),
        "CompositeTrendStrategy": _wrap_strategy_handler(
            _apply_composite_trend_strategy,
            include_n=True,
            include_meta=True,
        ),
        "VolCompressionVWAPReversionStrategy": _wrap_strategy_handler(
            _apply_vol_compression_vwap_reversion_strategy,
        ),
        "VolCompressionVwapReversionStrategy": _wrap_strategy_handler(
            _apply_vol_compression_vwap_reversion_strategy,
        ),
        "VolatilityCompressionReversionStrategy": _wrap_strategy_handler(
            _apply_vol_compression_vwap_reversion_strategy,
        ),
        "LeadLagSpilloverStrategy": _wrap_strategy_handler(_apply_leadlag_spillover_strategy),
        "AbnormalReturnContinuationStrategy": _wrap_strategy_handler(
            _apply_abnormal_return_continuation_strategy,
            include_meta=True,
        ),
        "TopCapTimeSeriesMomentumStrategy": _wrap_strategy_handler(
            _apply_topcap_tsmom_strategy,
            include_n=True,
            include_meta=True,
        ),
        "LastDayLiquidityRegimeStrategy": _wrap_strategy_handler(
            _apply_last_day_liquidity_regime_strategy,
            include_n=True,
            include_meta=True,
        ),
        "CarryTrendFactorRotationStrategy": _wrap_strategy_handler(
            _apply_carry_trend_factor_rotation_strategy,
            include_n=True,
            include_meta=True,
        ),
        "MeanReversionStdStrategy": _wrap_strategy_handler(
            _apply_mean_reversion_std_strategy,
            include_meta=True,
        ),
        "LiquidityShockReversionStrategy": _wrap_strategy_handler(
            _apply_liquidity_shock_reversion_strategy,
        ),
        "SessionLiquidityVacuumFadeStrategy": _wrap_strategy_handler(
            _apply_session_liquidity_vacuum_fade_strategy,
        ),
        "FundingLiquidationCrowdingFadeStrategy": _wrap_strategy_handler(
            _apply_funding_liquidation_crowding_fade_strategy,
            include_meta=True,
        ),
        "FundingDislocationTrendCarryStrategy": _wrap_strategy_handler(
            _apply_funding_dislocation_trend_carry_strategy,
            include_meta=True,
        ),
        "VolManagedMomentumCrashGateStrategy": _wrap_strategy_handler(
            _apply_vol_managed_momentum_crash_gate_strategy,
            include_meta=True,
        ),
        "FlowImbalanceLiquidationSweepStrategy": _wrap_strategy_handler(
            _apply_flow_imbalance_liquidation_sweep_strategy,
            include_meta=True,
        ),
        "BasisSnapbackReversionStrategy": _wrap_strategy_handler(
            _apply_basis_snapback_reversion_strategy,
            include_meta=True,
        ),
        "VolOfVolExhaustionFadeStrategy": _wrap_strategy_handler(
            _apply_vol_of_vol_exhaustion_fade_strategy,
        ),
        "ResidualBasketReversionStrategy": _wrap_strategy_handler(
            _apply_residual_basket_reversion_strategy,
            include_meta=True,
        ),
        "SessionGatedResidualBasketReversionStrategy": _wrap_strategy_handler(
            _apply_session_gated_residual_basket_reversion_strategy,
            include_meta=True,
        ),
        "VolatilityRegimeResidualBasketReversionStrategy": _wrap_strategy_handler(
            _apply_volatility_regime_residual_basket_reversion_strategy,
            include_meta=True,
        ),
        "BreadthThrustFailureReversalStrategy": _wrap_strategy_handler(
            _apply_breadth_thrust_failure_reversal_strategy,
            include_n=True,
            include_meta=True,
        ),
        "CrossAssetLiquidationContagionFadeStrategy": _wrap_strategy_handler(
            _apply_cross_asset_liquidation_contagion_fade_strategy,
        ),
        "MultiHorizonTrendExhaustionFadeStrategy": _wrap_strategy_handler(
            _apply_multi_horizon_trend_exhaustion_fade_strategy,
        ),
        "VwapReversionStrategy": _wrap_strategy_handler(_apply_vwap_reversion_strategy),
        "RollingBreakoutStrategy": _wrap_strategy_handler(_apply_rolling_breakout_strategy),
        "RegimeBreakoutCandidateStrategy": _wrap_strategy_handler(
            _apply_regime_breakout_candidate_strategy,
        ),
        "PairSpreadZScoreStrategy": _wrap_strategy_handler(
            _apply_pair_spread_strategy,
            include_n=True,
            include_meta=True,
            extra_kwargs={"strategy_class": "PairSpreadZScoreStrategy"},
        ),
        "PairTradingZScoreStrategy": _wrap_strategy_handler(
            _apply_pair_spread_strategy,
            include_n=True,
            include_meta=True,
            extra_kwargs={"strategy_class": "PairTradingZScoreStrategy"},
        ),
        "LagConvergenceStrategy": _wrap_strategy_handler(
            _apply_lag_convergence_strategy,
            include_n=True,
        ),
        "PerpCrowdingCarryStrategy": _wrap_strategy_handler(
            _apply_perp_crowding_carry_strategy,
            include_meta=True,
        ),
        "MicroRangeExpansion1sStrategy": _wrap_strategy_handler(
            _apply_micro_range_expansion_strategy,
        ),
    },
    minimum_symbol_counts={
        "PairSpreadZScoreStrategy": 2,
        "PairTradingZScoreStrategy": 2,
        "LagConvergenceStrategy": 2,
        "LastDayLiquidityRegimeStrategy": 2,
    },
)


def _registry_simulator_router(
    strategy_class: str,
    params: dict[str, Any],
    aligned: dict[str, np.ndarray],
    symbols: Sequence[str],
) -> np.ndarray | None:
    """Route an unmapped candidate through its REAL registered strategy class.

    Resolves ``strategy_class`` via the plugin registry and simulates it over
    the aligned panel with the same event-driven harness the Alpha101 handler
    already uses.  Returns ``None`` (never raises) when the class cannot be
    resolved or simulated, in which case the dispatcher falls back to the
    generic proxy WITH an explicit ``generic_fallback_proxy`` meta label.
    """
    try:
        from lumina_quant.strategies.registry import resolve_strategy_class

        strategy_cls = resolve_strategy_class(strategy_class)
    except Exception:
        return None
    try:
        return _simulate_event_driven_strategy_exposures(
            strategy_cls,
            params=params,
            aligned=aligned,
            symbols=symbols,
        )
    except Exception:
        return None


def _strict_registry_simulator_router(
    strategy_class: str,
    params: dict[str, Any],
    aligned: dict[str, np.ndarray],
    symbols: Sequence[str],
) -> np.ndarray:
    """Run a registered strategy without converting failures into a proxy route."""
    from lumina_quant.strategies.registry import resolve_strategy_class

    strategy_cls = resolve_strategy_class(strategy_class)
    return _simulate_event_driven_strategy_exposures(
        strategy_cls,
        params=params,
        aligned=aligned,
        symbols=symbols,
    )


def _strategy_signal(
    candidate: dict[str, Any],
    *,
    aligned: dict[str, np.ndarray],
    symbols: Sequence[str],
    scoring_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    # Config-gated (default OFF -> legacy routing): when ON, candidates whose
    # strategy_class has no bespoke handler are executed as their REAL
    # registered class instead of being silently scored as the shared generic
    # momentum proxy. Strict actual-engine routing always enables the registry
    # route and forbids proxy substitution.
    require_actual_engine = bool(
        _research_flag(scoring_config, "require_actual_engine_routing", False)
    )
    route_unmapped = require_actual_engine or bool(
        _research_flag(scoring_config, "route_unmapped_registered_strategies", False)
    )
    return _STRATEGY_SIGNAL_DISPATCHER.dispatch(
        candidate,
        aligned=aligned,
        symbols=symbols,
        unmapped_router=(
            _strict_registry_simulator_router
            if require_actual_engine
            else _registry_simulator_router
        )
        if route_unmapped
        else None,
        require_actual_engine=require_actual_engine,
    )


def _candidate_cost_rate(
    candidate: dict[str, Any],
    *,
    scoring_config: Mapping[str, Any] | None = None,
) -> float:
    strategy = str(candidate.get("strategy_class") or candidate.get("strategy") or "").lower()
    # Resolve the per-class map value EXACTLY as before, first.
    if "micro" in strategy:
        map_value = 0.0012
    elif "pair" in strategy:
        map_value = 0.0008
    elif "lagconvergence" in strategy or "lag_convergence" in strategy or "leadlag" in strategy:
        map_value = 0.0007
    else:
        map_value = 0.0005
    # Config-gated realistic cost floor (MEASUREMENT integrity). With the shipped
    # defaults (``override is None`` AND ``multiplier == 1.0``) this returns the
    # exact ``map_value`` above -- byte-identical to the legacy scorer -- because
    # ``x * 1.0 == x`` under IEEE arithmetic and the override short-circuit is
    # skipped. An absolute bps override wins over the multiplier when both are set.
    override = _research_flag(
        scoring_config, "cost_rate_bps_override", _RESEARCH_COST_RATE_BPS_OVERRIDE_DEFAULT
    )
    if override is not None:
        return float(override) / 10_000.0
    multiplier = _research_flag(
        scoring_config, "cost_rate_multiplier", _RESEARCH_COST_RATE_MULTIPLIER_DEFAULT
    )
    if float(multiplier) != 1.0:
        return map_value * float(multiplier)
    return map_value


def _candidate_symbols_and_timeframe(
    candidate: Mapping[str, Any],
) -> tuple[list[str], str]:
    symbols = canonicalize_symbol_list(list(candidate.get("symbols") or []))
    timeframe = str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or "1m")
    return symbols, timeframe


def _candidate_effective_split(
    candidate: Mapping[str, Any],
    fallback_split: Mapping[str, Any] | None,
    *,
    timeframe: str,
) -> dict[str, Any] | None:
    """Resolve a candidate-specific split when coverage metadata provides one."""
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
    candidates: tuple[Any, ...] = (
        candidate.get("effective_split"),
        candidate.get("candidate_split"),
        metadata.get("effective_split") if isinstance(metadata, Mapping) else None,
        metadata.get("candidate_split") if isinstance(metadata, Mapping) else None,
        (
            metadata.get("data_window", {}).get("effective_split")
            if isinstance(metadata, Mapping) and isinstance(metadata.get("data_window"), Mapping)
            else None
        ),
    )
    for raw in candidates:
        if isinstance(raw, Mapping):
            resolved = dict(raw)
            resolved["strategy_timeframe"] = str(
                resolved.get("strategy_timeframe")
                or candidate.get("strategy_timeframe")
                or candidate.get("timeframe")
                or timeframe
            )
            # Backfill the research-profile lockbox/purge flags when the
            # candidate's own coverage-derived split metadata doesn't set them
            # itself, so a per-candidate data window still honors an active
            # strict-research profile. The candidate's own value (if any) wins.
            if isinstance(fallback_split, Mapping):
                for _flag_name in ("use_lockbox_split", "purge_embargo_bars"):
                    if _flag_name not in resolved and _flag_name in fallback_split:
                        resolved[_flag_name] = fallback_split[_flag_name]
            return _resolve_split_config(resolved, strategy_timeframe=str(timeframe))

    if isinstance(fallback_split, Mapping):
        resolved = dict(fallback_split)
        resolved["strategy_timeframe"] = str(
            resolved.get("strategy_timeframe")
            or candidate.get("strategy_timeframe")
            or candidate.get("timeframe")
            or timeframe
        )
        return _resolve_split_config(resolved, strategy_timeframe=str(timeframe))
    return None


def _candidate_bundle_list(
    *,
    symbols: Sequence[str],
    timeframe: str,
    cache: Mapping[tuple[str, str], SeriesBundle],
) -> list[SeriesBundle]:
    bundles: list[SeriesBundle] = []
    for symbol in symbols:
        bundle = cache.get((symbol, timeframe))
        if bundle is None:
            continue
        bundles.append(bundle)
    return bundles


def _insufficient_candidate_result(
    candidate: dict[str, Any],
    *,
    symbols: Sequence[str],
    timeframe: str,
    cache: Mapping[tuple[str, str], SeriesBundle],
    effective_split: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "missing_symbols": [symbol for symbol in symbols if (symbol, timeframe) not in cache]
    }
    if isinstance(effective_split, Mapping):
        metadata["effective_split"] = dict(effective_split)
    return {
        "error": "insufficient_data",
        "candidate": candidate,
        "returns": np.asarray([], dtype=float),
        "turnover": np.asarray([], dtype=float),
        "exposure": np.asarray([], dtype=float),
        "metadata": metadata,
    }


def _call_insufficient_candidate_result(
    candidate: dict[str, Any],
    *,
    symbols: Sequence[str],
    timeframe: str,
    cache: Mapping[tuple[str, str], SeriesBundle],
    effective_split: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        return _insufficient_candidate_result(
            candidate,
            symbols=symbols,
            timeframe=timeframe,
            cache=cache,
            effective_split=effective_split,
        )
    except TypeError as exc:
        if "effective_split" not in str(exc):
            raise
        return _insufficient_candidate_result(
            candidate,
            symbols=symbols,
            timeframe=timeframe,
            cache=cache,
        )


def _candidate_benchmark_series(
    *,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    timeframe: str,
    timestamps: np.ndarray,
    returns_size: int,
) -> np.ndarray | None:
    benchmark_entry = benchmark_cache.get(timeframe)
    if isinstance(benchmark_entry, Mapping):
        benchmark_datetime = benchmark_entry.get("datetime")
        benchmark_returns = benchmark_entry.get("returns")
        return _align_series_to_timestamps(
            timestamps,
            source_timestamps=np.asarray(
                benchmark_datetime if benchmark_datetime is not None else [],
                dtype="datetime64[ms]",
            ),
            values=np.asarray(
                benchmark_returns if benchmark_returns is not None else [], dtype=float
            ),
        )

    benchmark = np.asarray(benchmark_entry if benchmark_entry is not None else [], dtype=float)
    if benchmark.size < returns_size:
        return None
    return benchmark[-returns_size:]


def _metrics_for_mask(
    *,
    returns: np.ndarray,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark_returns: np.ndarray,
    timestamps: np.ndarray,
    mask: np.ndarray,
    periods_per_year: int,
    candidate_count: int,
    hac_inference: bool = _RESEARCH_HAC_INFERENCE_DEFAULT,
) -> dict[str, Any]:
    return _compute_metrics(
        returns[mask],
        turnover=turnover[mask],
        exposure=exposure[mask],
        benchmark_returns=benchmark_returns[mask],
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
        metric_config=get_default_runtime_config().backtest,
        timestamps=timestamps[mask],
        hac_inference=hac_inference,
    )


def _candidate_stage_metrics(
    *,
    returns: np.ndarray,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark: np.ndarray,
    timestamps: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    periods_per_year: int,
    candidate_count: int,
    hac_inference: bool = _RESEARCH_HAC_INFERENCE_DEFAULT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_metrics = _metrics_for_mask(
        returns=returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark,
        timestamps=timestamps,
        mask=split_masks["train"],
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
        hac_inference=hac_inference,
    )
    val_metrics = _metrics_for_mask(
        returns=returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark,
        timestamps=timestamps,
        mask=split_masks["val"],
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
        hac_inference=hac_inference,
    )
    oos_metrics = _metrics_for_mask(
        returns=returns,
        turnover=turnover,
        exposure=exposure,
        benchmark_returns=benchmark,
        timestamps=timestamps,
        mask=split_masks["oos"],
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
        hac_inference=hac_inference,
    )
    return train_metrics, val_metrics, oos_metrics


def _candidate_oos_cost_stress_metrics(
    *,
    returns_raw: np.ndarray,
    turnover: np.ndarray,
    exposure: np.ndarray,
    benchmark: np.ndarray,
    timestamps: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    cost_rate: float,
    periods_per_year: int,
    candidate_count: int,
    hac_inference: bool = _RESEARCH_HAC_INFERENCE_DEFAULT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    oos_mask = split_masks["oos"]
    oos_turnover = turnover[oos_mask]
    oos_x2 = returns_raw[oos_mask] - (oos_turnover * (cost_rate * 2.0))
    oos_x3 = returns_raw[oos_mask] - (oos_turnover * (cost_rate * 3.0))
    oos_stress_x2 = _compute_metrics(
        oos_x2,
        turnover=oos_turnover,
        exposure=exposure[oos_mask],
        benchmark_returns=benchmark[oos_mask],
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
        metric_config=get_default_runtime_config().backtest,
        timestamps=timestamps[oos_mask],
        hac_inference=hac_inference,
    )
    oos_stress_x3 = _compute_metrics(
        oos_x3,
        turnover=oos_turnover,
        exposure=exposure[oos_mask],
        benchmark_returns=benchmark[oos_mask],
        periods_per_year=periods_per_year,
        num_trials=candidate_count,
        metric_config=get_default_runtime_config().backtest,
        timestamps=timestamps[oos_mask],
        hac_inference=hac_inference,
    )
    return oos_stress_x2, oos_stress_x3


def _apply_cost_stress_hard_rejects(
    *,
    hard_reject: dict[str, Any],
    oos_stress_x2: Mapping[str, Any],
    oos_stress_x3: Mapping[str, Any],
    scoring_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(hard_reject)
    cfg = _resolve_score_config(scoring_config)
    stress_cfg = dict(cfg.get("cost_stress_thresholds") or {})
    if float(oos_stress_x2.get("sharpe", 0.0)) < float(stress_cfg.get("x2_sharpe_min", 0.0)):
        updated["stress_x2_sharpe"] = float(oos_stress_x2.get("sharpe", 0.0))
    if float(oos_stress_x3.get("sharpe", 0.0)) < float(stress_cfg.get("x3_sharpe_min", -0.25)):
        updated["stress_x3_sharpe"] = float(oos_stress_x3.get("sharpe", 0.0))
    return updated


def _stage1_prefilter_score(
    result: Mapping[str, Any],
    *,
    stage1_weights: Mapping[str, Any],
    stage1_error_score: float,
) -> float:
    if result.get("error"):
        return float(stage1_error_score)
    train = dict(result.get("train") or {})
    return float(
        (float(stage1_weights["sharpe_weight"]) * float(train.get("sharpe", 0.0)))
        + (float(stage1_weights["return_weight"]) * float(train.get("return", 0.0)))
        - (float(stage1_weights["pbo_penalty"]) * float(train.get("pbo", 1.0)))
    )


@dataclass(frozen=True, slots=True)
class _CandidateSignalPayload:
    symbols: list[str]
    timeframe: str
    timestamps: np.ndarray
    returns_raw: np.ndarray
    returns: np.ndarray
    turnover: np.ndarray
    exposure: np.ndarray
    meta: dict[str, Any]
    cost_rate: float


@dataclass(frozen=True, slots=True)
class _CandidateMetricPayload:
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    oos_metrics: dict[str, Any]
    oos_stress_x2: dict[str, Any]
    oos_stress_x3: dict[str, Any]
    # Populated only under the lockbox discipline (opt-in). When ``None`` the
    # legacy 3-way train/val/oos payload is used unchanged.
    lockbox_metrics: dict[str, Any] | None = None


def _load_candidate_signal_payload(
    candidate: dict[str, Any],
    *,
    cache: Mapping[tuple[str, str], SeriesBundle],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] | None = None,
    scoring_config: Mapping[str, Any] | None = None,
) -> _CandidateSignalPayload | None:
    symbols, timeframe = _candidate_symbols_and_timeframe(candidate)
    bundles = _candidate_bundle_list(symbols=symbols, timeframe=timeframe, cache=cache)
    cache_key = (
        timeframe,
        tuple(symbols),
        tuple(
            (
                bundle.symbol,
                int(bundle.datetime.size),
                int(np.asarray(bundle.datetime, dtype="datetime64[ms]")[0].astype(np.int64))
                if bundle.datetime.size
                else -1,
                int(np.asarray(bundle.datetime, dtype="datetime64[ms]")[-1].astype(np.int64))
                if bundle.datetime.size
                else -1,
            )
            for bundle in bundles
        ),
    )
    aligned = aligned_cache.get(cache_key) if aligned_cache is not None else None
    if aligned is None:
        aligned = _align_bundles(bundles, feature_cache=feature_cache)
        if aligned_cache is not None and aligned is not None:
            aligned_cache[cache_key] = aligned
    if aligned is None:
        return None

    returns_raw, turnover, exposure, meta = _strategy_signal(
        candidate, aligned=aligned, symbols=symbols, scoring_config=scoring_config
    )
    cost_rate = _candidate_cost_rate(candidate, scoring_config=scoring_config)
    timestamps = np.asarray(aligned.get("datetime"), dtype="datetime64[ms]")
    return _CandidateSignalPayload(
        symbols=symbols,
        timeframe=timeframe,
        timestamps=timestamps,
        returns_raw=returns_raw,
        returns=returns_raw - (turnover * cost_rate),
        turnover=turnover,
        exposure=exposure,
        meta=meta,
        cost_rate=float(cost_rate),
    )


def _evaluate_candidate_metric_payload(
    signal_payload: _CandidateSignalPayload,
    *,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    candidate_count: int,
    split: Mapping[str, Any] | None,
    scoring_config: Mapping[str, Any] | None = None,
) -> _CandidateMetricPayload | None:
    split_masks = _split_masks_from_datetimes(signal_payload.timestamps, split=split)
    periods_per_year = int(_PERIODS_PER_YEAR.get(signal_payload.timeframe, 365))
    benchmark = _candidate_benchmark_series(
        benchmark_cache=benchmark_cache,
        timeframe=signal_payload.timeframe,
        timestamps=signal_payload.timestamps,
        returns_size=signal_payload.returns.size,
    )
    if benchmark is None:
        return None
    # Opt-in HAC/Newey-West correction of the per-candidate DSR studentization
    # (forwarded to ``research_metrics.deflated_sharpe_ratio``). Resolved once
    # here from the same ``score_config['research']`` section that already
    # carries ``strict_selection_gate`` etc., so the flag is legacy-off unless
    # the strict research profile turns it on.
    hac_inference = bool(
        _research_flag(scoring_config, "hac_inference", _RESEARCH_HAC_INFERENCE_DEFAULT)
    )
    train_metrics, val_metrics, oos_metrics = _candidate_stage_metrics(
        returns=signal_payload.returns,
        turnover=signal_payload.turnover,
        exposure=signal_payload.exposure,
        benchmark=benchmark,
        timestamps=signal_payload.timestamps,
        split_masks=split_masks,
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
        hac_inference=hac_inference,
    )
    oos_stress_x2, oos_stress_x3 = _candidate_oos_cost_stress_metrics(
        returns_raw=signal_payload.returns_raw,
        turnover=signal_payload.turnover,
        exposure=signal_payload.exposure,
        benchmark=benchmark,
        timestamps=signal_payload.timestamps,
        split_masks=split_masks,
        cost_rate=signal_payload.cost_rate,
        periods_per_year=periods_per_year,
        candidate_count=candidate_count,
        hac_inference=hac_inference,
    )
    lockbox_metrics: dict[str, Any] | None = None
    lockbox_mask = split_masks.get("lockbox")
    if lockbox_mask is not None:
        lockbox_metrics = _metrics_for_mask(
            returns=signal_payload.returns,
            turnover=signal_payload.turnover,
            exposure=signal_payload.exposure,
            benchmark_returns=benchmark,
            timestamps=signal_payload.timestamps,
            mask=lockbox_mask,
            periods_per_year=periods_per_year,
            candidate_count=candidate_count,
            hac_inference=hac_inference,
        )
    return _CandidateMetricPayload(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        oos_metrics=oos_metrics,
        oos_stress_x2=oos_stress_x2,
        oos_stress_x3=oos_stress_x3,
        lockbox_metrics=lockbox_metrics,
    )


def _evaluate_candidate_hurdles(
    metric_payload: _CandidateMetricPayload,
    *,
    scoring_config: Mapping[str, Any] | None,
    split: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], bool, dict[str, Any]]:
    # Lockbox discipline (opt-in): the binding hurdle runs on validation and the
    # ``oos`` slot reports the never-touched lockbox scored once. Default keeps the
    # legacy oos-reported / oos-binding contract, byte-identical when the flag is
    # unset (``lockbox_metrics`` is ``None``).
    use_lockbox = metric_payload.lockbox_metrics is not None and bool(
        _split_flag(split, "use_lockbox_split", _RESEARCH_USE_LOCKBOX_SPLIT_DEFAULT)
    )
    reported_oos = metric_payload.lockbox_metrics if use_lockbox else metric_payload.oos_metrics
    bind_stage = "val" if use_lockbox else "oos"
    hurdle_fields, passed, hard_reject = _hurdle_fields(
        metric_payload.train_metrics,
        metric_payload.val_metrics,
        reported_oos,
        scoring_config=scoring_config,
        bind_stage=bind_stage,
    )
    hard_reject = _apply_cost_stress_hard_rejects(
        hard_reject=hard_reject,
        oos_stress_x2=metric_payload.oos_stress_x2,
        oos_stress_x3=metric_payload.oos_stress_x3,
        scoring_config=scoring_config,
    )
    passed = bool(passed and not hard_reject)
    for stage in ("train", "val", "oos"):
        hurdle_fields[stage]["pass"] = bool(hurdle_fields[stage]["pass"] and passed)
    return hurdle_fields, passed, hard_reject


def _candidate_result_payload(
    candidate: dict[str, Any],
    *,
    signal_payload: _CandidateSignalPayload,
    metric_payload: _CandidateMetricPayload,
    hurdle_fields: dict[str, dict[str, Any]],
    passed: bool,
    hard_reject: dict[str, Any],
    effective_split: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata = {
        "strategy_family": _family_from_strategy(str(candidate.get("strategy_class") or "")),
        "cost_rate": float(signal_payload.cost_rate),
        "aligned_bars": int(signal_payload.timestamps.size),
        **signal_payload.meta,
    }
    if isinstance(effective_split, Mapping):
        metadata["effective_split"] = dict(effective_split)
    # Under the lockbox discipline the reported ``oos`` is the never-touched
    # lockbox scored once; the intermediate oos segment is preserved separately
    # for transparency. Default keeps ``oos`` as the legacy oos metrics.
    use_lockbox = metric_payload.lockbox_metrics is not None and bool(
        _split_flag(effective_split, "use_lockbox_split", _RESEARCH_USE_LOCKBOX_SPLIT_DEFAULT)
    )
    reported_oos = metric_payload.lockbox_metrics if use_lockbox else metric_payload.oos_metrics
    payload = {
        "candidate": candidate,
        "timestamps": signal_payload.timestamps,
        "returns": signal_payload.returns,
        "turnover": signal_payload.turnover,
        "exposure": signal_payload.exposure,
        "train": metric_payload.train_metrics,
        "val": metric_payload.val_metrics,
        "oos": reported_oos,
        "oos_cost_stress": {
            "x2": {
                "sharpe": float(metric_payload.oos_stress_x2.get("sharpe", 0.0)),
                "return": float(metric_payload.oos_stress_x2.get("return", 0.0)),
            },
            "x3": {
                "sharpe": float(metric_payload.oos_stress_x3.get("sharpe", 0.0)),
                "return": float(metric_payload.oos_stress_x3.get("return", 0.0)),
            },
        },
        "hurdle_fields": hurdle_fields,
        "pass": bool(passed),
        "hard_reject_reasons": hard_reject,
        "metadata": metadata,
    }
    if use_lockbox:
        payload["oos_intermediate"] = metric_payload.oos_metrics
    return payload


def _evaluate_candidate(
    candidate: dict[str, Any],
    *,
    cache: Mapping[tuple[str, str], SeriesBundle],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] | None = None,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    candidate_count: int,
    scoring_config: Mapping[str, Any] | None = None,
    split: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    symbols, timeframe = _candidate_symbols_and_timeframe(candidate)
    effective_split = _candidate_effective_split(
        candidate,
        split,
        timeframe=timeframe,
    )
    signal_payload = _load_candidate_signal_payload(
        candidate,
        cache=cache,
        feature_cache=feature_cache,
        aligned_cache=aligned_cache,
        scoring_config=scoring_config,
    )
    if signal_payload is None:
        return _call_insufficient_candidate_result(
            candidate,
            symbols=symbols,
            timeframe=timeframe,
            cache=cache,
            effective_split=effective_split,
        )

    try:
        metric_payload = _evaluate_candidate_metric_payload(
            signal_payload,
            benchmark_cache=benchmark_cache,
            candidate_count=candidate_count,
            split=effective_split,
            scoring_config=scoring_config,
        )
    except TypeError as exc:
        # Back-compat with call sites (including existing test monkeypatches,
        # e.g. tests/test_candidate_effective_split.py) that replace
        # ``_evaluate_candidate_metric_payload`` with a stub predating the
        # ``scoring_config`` kwarg. Only the "unexpected keyword" shape is
        # swallowed; any other ``TypeError`` (a real bug) still propagates.
        if "scoring_config" not in str(exc):
            raise
        metric_payload = _evaluate_candidate_metric_payload(
            signal_payload,
            benchmark_cache=benchmark_cache,
            candidate_count=candidate_count,
            split=effective_split,
        )
    if metric_payload is None:
        return _call_insufficient_candidate_result(
            candidate,
            symbols=symbols,
            timeframe=timeframe,
            cache=cache,
            effective_split=effective_split,
        )
    hurdle_fields, passed, hard_reject = _evaluate_candidate_hurdles(
        metric_payload,
        scoring_config=scoring_config,
        split=effective_split,
    )
    return _candidate_result_payload(
        candidate,
        signal_payload=signal_payload,
        metric_payload=metric_payload,
        hurdle_fields=hurdle_fields,
        passed=passed,
        hard_reject=hard_reject,
        effective_split=effective_split,
    )


def _candidate_instability_penalty(
    row: dict[str, Any],
    *,
    scoring_config: Mapping[str, Any] | None = None,
) -> float:
    cfg = _resolve_score_config(scoring_config)
    weights = dict(cfg["candidate_rank_score_weights"])
    val = dict(row.get("val") or {})
    oos = dict(row.get("oos") or {})
    sharpe_gap = max(0.0, float(val.get("sharpe", 0.0)) - float(oos.get("sharpe", 0.0)))
    return_gap = max(0.0, float(val.get("return", 0.0)) - float(oos.get("return", 0.0)))
    turnover_gap = max(0.0, float(oos.get("turnover", 0.0)) - float(val.get("turnover", 0.0)))
    return float(
        (float(weights["instability_sharpe_penalty"]) * sharpe_gap)
        + (float(weights["instability_return_penalty"]) * return_gap)
        + (float(weights["instability_turnover_penalty"]) * turnover_gap)
    )


def _candidate_sparse_fold_penalty(
    row: dict[str, Any],
    *,
    scoring_config: Mapping[str, Any] | None = None,
) -> float:
    cfg = _resolve_score_config(scoring_config)
    weights = dict(cfg["candidate_rank_score_weights"])
    oos = dict(row.get("oos") or {})
    active_fold_ratio = max(0.0, min(1.0, float(oos.get("active_fold_ratio", 1.0))))
    inactive_fold_count = max(0.0, float(oos.get("inactive_fold_count", 0.0)))
    failed_fold_ratio = max(
        0.0, min(1.0, float(oos.get("failed_fold_ratio", float(oos.get("pbo", 1.0)))))
    )
    return float(
        (float(weights["inactive_fold_penalty"]) * inactive_fold_count)
        + (float(weights["failed_fold_penalty"]) * failed_fold_ratio)
        + (
            float(weights["low_active_fold_penalty"])
            * max(0.0, float(weights["active_fold_ratio_floor"]) - active_fold_ratio)
        )
    )


def _candidate_no_trade_train_penalty(
    row: dict[str, Any],
    *,
    scoring_config: Mapping[str, Any] | None = None,
) -> float:
    cfg = _resolve_score_config(scoring_config)
    weights = dict(cfg["candidate_rank_score_weights"])
    train = dict(row.get("train") or {})
    train_return = float(train.get("total_return", train.get("return", 0.0)) or 0.0)
    train_trade_count = float(train.get("trade_count", train.get("trades", 0.0)) or 0.0)
    if abs(train_return) <= 1e-12 and train_trade_count <= 0.0:
        return float(weights["no_trade_train_penalty"])
    return 0.0


def _candidate_row_effective_split(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    effective = row.get("effective_split")
    if isinstance(effective, Mapping):
        return effective
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        nested = metadata.get("effective_split")
        if isinstance(nested, Mapping):
            return nested
    return None


def _candidate_rank_score(
    row: dict[str, Any], *, scoring_config: Mapping[str, Any] | None = None
) -> float:
    cfg = _resolve_score_config(scoring_config)
    weights = dict(cfg["candidate_rank_score_weights"])
    # Lockbox discipline (opt-in): rank on validation so the reported OOS (the
    # never-touched lockbox) is not consumed by ranking. Legacy default ranks on
    # ``oos`` and is byte-identical when the flag is unset.
    rank_stage = (
        "val"
        if bool(
            _split_flag(
                _candidate_row_effective_split(row),
                "use_lockbox_split",
                _RESEARCH_USE_LOCKBOX_SPLIT_DEFAULT,
            )
        )
        else "oos"
    )
    oos = dict(row.get(rank_stage) or {})
    return float(
        (float(weights["sharpe_weight"]) * float(oos.get("sharpe", 0.0)))
        + (float(weights["deflated_sharpe_weight"]) * float(oos.get("deflated_sharpe", 0.0)))
        - (float(weights["pbo_penalty"]) * float(oos.get("pbo", 1.0)))
        + (float(weights["return_weight"]) * float(oos.get("return", 0.0)))
        - (
            float(weights["turnover_penalty"])
            * max(0.0, float(oos.get("turnover", 0.0)) - float(weights["turnover_threshold"]))
        )
        - (float(weights["drawdown_penalty"]) * float(oos.get("mdd", 0.0)))
        - _candidate_instability_penalty(row, scoring_config=scoring_config)
        - _candidate_sparse_fold_penalty(row, scoring_config=scoring_config)
        - _candidate_no_trade_train_penalty(row, scoring_config=scoring_config)
    )


def _read_csv_ohlcv(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path)
    cols = {str(col).lower(): col for col in frame.columns}
    required = {
        "open": cols.get("open"),
        "high": cols.get("high"),
        "low": cols.get("low"),
        "close": cols.get("close"),
        "volume": cols.get("volume"),
    }
    if any(value is None for value in required.values()):
        return pl.DataFrame()

    if "datetime" in cols:
        dt_col = cols["datetime"]
        dt_expr = pl.col(dt_col).cast(pl.Datetime(time_unit="ms"), strict=False)
    elif "timestamp" in cols:
        dt_col = cols["timestamp"]
        dt_expr = pl.from_epoch(pl.col(dt_col).cast(pl.Int64), time_unit="s").cast(
            pl.Datetime(time_unit="ms")
        )
    else:
        return pl.DataFrame()

    out = (
        frame.select(
            [
                dt_expr.alias("datetime"),
                pl.col(required["open"]).cast(pl.Float64).alias("open"),
                pl.col(required["high"]).cast(pl.Float64).alias("high"),
                pl.col(required["low"]).cast(pl.Float64).alias("low"),
                pl.col(required["close"]).cast(pl.Float64).alias("close"),
                pl.col(required["volume"]).cast(pl.Float64).alias("volume"),
            ]
        )
        .drop_nulls()
        .sort("datetime")
    )
    return out


def _synthetic_bundle(
    symbol: str,
    timeframe: str,
    *,
    bars: int = 2400,
    start_date: Any = None,
    end_date: Any = None,
) -> SeriesBundle:
    step_seconds, bars, start = _synthetic_bundle_window(
        timeframe=timeframe,
        bars=bars,
        start_date=start_date,
        end_date=end_date,
    )
    open_, high, low, close, volume, dt = _synthetic_bundle_arrays(
        symbol=symbol,
        timeframe=timeframe,
        bars=bars,
        start=start,
        step_seconds=step_seconds,
    )

    return SeriesBundle(
        symbol=symbol,
        timeframe=timeframe,
        datetime=dt,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _synthetic_bundle_window(
    *,
    timeframe: str,
    bars: int,
    start_date: Any,
    end_date: Any,
) -> tuple[int, int, datetime]:
    step_seconds = {
        "1s": 1,
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14_400,
        "1d": 86_400,
    }.get(timeframe, 60)
    start_bound = _coerce_utc_datetime(start_date)
    end_bound = _coerce_utc_datetime(end_date, end_of_day=True)
    if start_bound is not None and end_bound is not None and end_bound > start_bound:
        requested_bars = int(((end_bound - start_bound).total_seconds()) // step_seconds) + 1
        bars = max(_MIN_BARS, min(max(bars, requested_bars), 20_000))
        start = (
            start_bound
            if requested_bars <= bars
            else end_bound - timedelta(seconds=(bars - 1) * step_seconds)
        )
        return step_seconds, bars, start
    return step_seconds, bars, datetime.now(UTC) - timedelta(seconds=bars * step_seconds)


def _synthetic_bundle_arrays(
    *,
    symbol: str,
    timeframe: str,
    bars: int,
    start: datetime,
    step_seconds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(_hash_seed("synthetic", symbol, timeframe))
    open_ = np.zeros(bars, dtype=float)
    high = np.zeros(bars, dtype=float)
    low = np.zeros(bars, dtype=float)
    close = np.zeros(bars, dtype=float)
    volume = np.zeros(bars, dtype=float)
    dt = np.zeros(bars, dtype="datetime64[ms]")

    base = 100.0 + (20.0 * _hash_unit_interval(symbol, timeframe, "base"))
    drift = 0.00002 + (0.00008 * _hash_unit_interval(symbol, timeframe, "drift"))
    phase = 2.0 * math.pi * _hash_unit_interval(symbol, timeframe, "phase")
    price = base
    for idx in range(bars):
        regime = math.sin((idx / max(50.0, bars / 18.0)) + phase)
        step = drift + (0.001 * regime) + rng.gauss(0.0, 0.0025)

        o = max(0.1, price)
        c = max(0.1, o * (1.0 + step))
        wiggle = abs(rng.gauss(0.0, 0.0018)) + 0.0003
        open_[idx] = o
        high[idx] = max(o, c) * (1.0 + wiggle)
        low[idx] = min(o, c) * (1.0 - wiggle)
        close[idx] = c
        volume[idx] = max(1.0, 1200.0 * (1.0 + abs(regime)) + rng.uniform(-200.0, 200.0))
        dt[idx] = np.datetime64(
            (start + timedelta(seconds=idx * step_seconds)).replace(tzinfo=None), "ms"
        )
        price = c
    return open_, high, low, close, volume, dt


def _frame_to_bundle(symbol: str, timeframe: str, frame: pl.DataFrame) -> SeriesBundle:
    sorted_frame = frame.sort("datetime")
    return SeriesBundle(
        symbol=symbol,
        timeframe=timeframe,
        datetime=sorted_frame["datetime"].to_numpy(),
        open=sorted_frame["open"].to_numpy(),
        high=sorted_frame["high"].to_numpy(),
        low=sorted_frame["low"].to_numpy(),
        close=sorted_frame["close"].to_numpy(),
        volume=sorted_frame["volume"].to_numpy(),
    )


def _partitioned_1m_symbol_dir(
    *,
    parquet_root: str,
    exchange: str,
    symbol: str,
) -> Path:
    return (
        Path(parquet_root)
        / f"exchange={exchange}"
        / f"symbol={str(symbol).replace('/', '')}"
        / "timeframe=1m"
    )


def _partition_date_from_path(path: Path) -> datetime | None:
    if not path.name.startswith("date="):
        return None
    try:
        return datetime.fromisoformat(path.name.split("=", 1)[1]).replace(tzinfo=UTC)
    except ValueError:
        return None


def _load_partitioned_1m_frame(
    *,
    symbol: str,
    parquet_root: str,
    exchange: str,
    start_date: Any,
    end_date: Any,
    raw_cache: dict[tuple[str, str, str], pl.DataFrame],
) -> pl.DataFrame:
    start_dt = _coerce_utc_datetime(start_date)
    end_dt = _coerce_utc_datetime(end_date, end_of_day=True)
    cache_key = (
        str(symbol),
        _datetime_to_iso_z(start_dt) or "",
        _datetime_to_iso_z(end_dt) or "",
    )
    cached = raw_cache.get(cache_key)
    if cached is not None:
        return cached

    timeframe_dir = _partitioned_1m_symbol_dir(
        parquet_root=parquet_root,
        exchange=exchange,
        symbol=symbol,
    )
    if not timeframe_dir.exists():
        raw_cache[cache_key] = pl.DataFrame()
        return raw_cache[cache_key]

    paths: list[str] = []
    for date_dir in sorted(timeframe_dir.glob("date=*")):
        partition_dt = _partition_date_from_path(date_dir)
        if partition_dt is None:
            continue
        if start_dt is not None and partition_dt.date() < start_dt.date():
            continue
        if end_dt is not None and partition_dt.date() > end_dt.date():
            continue
        paths.extend(str(path) for path in sorted(date_dir.glob("*.parquet")))
    if not paths:
        raw_cache[cache_key] = pl.DataFrame()
        return raw_cache[cache_key]

    try:
        lazy = pl.scan_parquet(paths).select(["datetime", "open", "high", "low", "close", "volume"])
        if start_dt is not None:
            lazy = lazy.filter(pl.col("datetime") >= start_dt.replace(tzinfo=None))
        if end_dt is not None:
            lazy = lazy.filter(pl.col("datetime") <= end_dt.replace(tzinfo=None))
        frame = lazy.collect().sort("datetime")
    except FileNotFoundError, OSError, RuntimeError, ValueError:
        frame = pl.DataFrame()
    raw_cache[cache_key] = frame
    return frame


def _resample_partitioned_1m_frame(frame: pl.DataFrame, *, timeframe: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    timeframe_token = str(timeframe).strip().lower()
    if timeframe_token == "1m":
        return frame.select(["datetime", "open", "high", "low", "close", "volume"])
    try:
        tf_ms = int(timeframe_to_milliseconds(timeframe_token))
    except ValueError:
        return pl.DataFrame()
    return (
        frame.lazy()
        .with_columns(pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"))
        .with_columns(((pl.col("timestamp_ms") // tf_ms) * tf_ms).alias("bucket_ms"))
        .group_by("bucket_ms")
        .agg(
            [
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .sort("bucket_ms")
        .with_columns(pl.from_epoch("bucket_ms", time_unit="ms").alias("datetime"))
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .collect()
    )


def _load_partitioned_1m_resample_bundle(
    *,
    symbol: str,
    timeframe: str,
    parquet_root: str,
    exchange: str,
    start_date: Any,
    end_date: Any,
    min_bars: int,
    raw_cache: dict[tuple[str, str, str], pl.DataFrame],
) -> SeriesBundle | None:
    raw_1m = _load_partitioned_1m_frame(
        symbol=symbol,
        parquet_root=parquet_root,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        raw_cache=raw_cache,
    )
    resampled = _resample_partitioned_1m_frame(raw_1m, timeframe=timeframe)
    if resampled.is_empty() or resampled.height < max(1, int(min_bars)):
        return None
    return _frame_to_bundle(symbol, timeframe, resampled)


def _load_timeframe_parquet_frames(
    *,
    symbols: Sequence[str],
    timeframe: str,
    parquet_root: str,
    exchange: str,
    start_date: Any,
    end_date: Any,
    data_mode: str,
    chunk_days: int = 7,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, pl.DataFrame]:
    base_kwargs: dict[str, Any] = {
        "exchange": exchange,
        "symbol_list": list(symbols),
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "data_mode": str(data_mode or "legacy"),
    }
    attempts: list[dict[str, Any]] = [
        {
            **base_kwargs,
            "chunk_days": max(1, int(chunk_days)),
            "progress_callback": progress_callback,
        },
        {
            **base_kwargs,
            "chunk_days": max(1, int(chunk_days)),
        },
        dict(base_kwargs),
    ]
    try:
        last_type_error: TypeError | None = None
        for kwargs in attempts:
            if kwargs.get("progress_callback") is None:
                kwargs.pop("progress_callback", None)
            try:
                return load_data_dict_from_parquet(parquet_root, **kwargs)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                last_type_error = exc
                continue
        if last_type_error is not None:
            raise last_type_error
        return {}
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return {}
    except FileNotFoundError, OSError, RuntimeError, ValueError:
        return {}


def _csv_bundle_candidates(symbol: str) -> list[Path]:
    compact = symbol.replace("/", "")
    return [
        Path("data") / f"{compact}.csv",
        Path("data") / f"{symbol}.csv",
        Path("data") / f"{symbol.replace('/', '_')}.csv",
    ]


def _filter_csv_frame_date_bounds(
    frame_csv: pl.DataFrame,
    *,
    start_date: Any,
    end_date: Any,
) -> pl.DataFrame:
    if frame_csv.is_empty() or (start_date is None and end_date is None):
        return frame_csv

    start_bound = _coerce_utc_datetime(start_date)
    end_bound = _coerce_utc_datetime(end_date, end_of_day=True)
    if start_bound is not None:
        frame_csv = frame_csv.filter(pl.col("datetime") >= start_bound.replace(tzinfo=None))
    if end_bound is not None:
        frame_csv = frame_csv.filter(pl.col("datetime") <= end_bound.replace(tzinfo=None))
    return frame_csv


def _load_csv_bundle(
    *,
    symbol: str,
    timeframe: str,
    start_date: Any,
    end_date: Any,
    min_bars: int,
) -> SeriesBundle | None:
    for csv_path in _csv_bundle_candidates(symbol):
        if not csv_path.exists():
            continue
        try:
            frame_csv = _read_csv_ohlcv(csv_path)
        except FileNotFoundError, OSError, RuntimeError, ValueError:
            frame_csv = pl.DataFrame()
        frame_csv = _filter_csv_frame_date_bounds(
            frame_csv,
            start_date=start_date,
            end_date=end_date,
        )
        if frame_csv.is_empty() or frame_csv.height < max(1, int(min_bars)):
            continue
        return _frame_to_bundle(symbol, timeframe, frame_csv)
    return None


def _strict_missing_bundle_error(*, symbol: str, timeframe: str) -> RawFirstDataMissingError:
    return RawFirstDataMissingError(
        f"Real market data missing for {symbol}@{timeframe} in strict mode."
    )


def _synthetic_disabled_bundle_error(*, symbol: str, timeframe: str) -> RawFirstDataMissingError:
    return RawFirstDataMissingError(
        f"Synthetic fallback is disabled and real market data is missing for {symbol}@{timeframe}."
    )


def _bundle_from_frame(
    *,
    symbol: str,
    timeframe: str,
    frame: pl.DataFrame | None,
    min_bars: int,
) -> SeriesBundle | None:
    if frame is None or frame.is_empty() or frame.height < max(1, int(min_bars)):
        return None
    return _frame_to_bundle(symbol, timeframe, frame)


def _resolve_bundle_cache_entry(
    *,
    symbol: str,
    timeframe: str,
    frame: pl.DataFrame | None,
    parquet_root: str,
    exchange: str,
    start_date: Any,
    end_date: Any,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bars: int,
    partitioned_1m_cache: dict[tuple[str, str, str], pl.DataFrame],
) -> tuple[SeriesBundle, str]:
    parquet_bundle = _bundle_from_frame(
        symbol=symbol,
        timeframe=timeframe,
        frame=frame,
        min_bars=min_bars,
    )
    if parquet_bundle is not None:
        return parquet_bundle, "parquet"

    partitioned_bundle = _load_partitioned_1m_resample_bundle(
        symbol=symbol,
        timeframe=timeframe,
        parquet_root=parquet_root,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        min_bars=min_bars,
        raw_cache=partitioned_1m_cache,
    )
    if partitioned_bundle is not None:
        return partitioned_bundle, "partitioned_1m_resample"

    if not allow_csv_fallback and not allow_synthetic_fallback:
        raise _strict_missing_bundle_error(symbol=symbol, timeframe=timeframe)

    if allow_csv_fallback:
        csv_bundle = _load_csv_bundle(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            min_bars=min_bars,
        )
        if csv_bundle is not None:
            return csv_bundle, "csv"

    if not allow_synthetic_fallback:
        raise _synthetic_disabled_bundle_error(symbol=symbol, timeframe=timeframe)

    return _synthetic_bundle(
        symbol, timeframe, start_date=start_date, end_date=end_date
    ), "synthetic"


def _load_bundle_cache(
    *,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    required_pairs: Sequence[tuple[str, str]] | None = None,
    start_date: Any = None,
    end_date: Any = None,
    data_mode: str = "legacy",
    allow_csv_fallback: bool = True,
    allow_synthetic_fallback: bool = False,
    min_bars: int = _MIN_BARS,
    market_data_settings: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[tuple[str, str], SeriesBundle], dict[str, list[str]]]:
    cache: dict[tuple[str, str], SeriesBundle] = {}
    source_map: dict[str, list[str]] = {"parquet": [], "csv": [], "synthetic": []}
    partitioned_1m_cache: dict[tuple[str, str, str], pl.DataFrame] = {}

    defaults = _current_research_market_data_settings(market_data_settings)
    parquet_root = str(defaults["parquet_root"])
    exchange = str(defaults["exchange"])
    chunk_days = max(
        1,
        int(
            (market_data_settings or {}).get(
                "chunk_days",
                os.getenv("LQ_RESEARCH_CHUNK_DAYS", "7"),
            )
            or 7
        ),
    )

    required_by_timeframe: dict[str, list[str]] | None = None
    if required_pairs:
        required_by_timeframe = {}
        for raw_symbol, raw_timeframe in required_pairs:
            symbol = str(raw_symbol)
            timeframe = str(raw_timeframe)
            bucket = required_by_timeframe.setdefault(timeframe, [])
            if symbol not in bucket:
                bucket.append(symbol)
    load_plan: list[tuple[str, list[str]]] = []
    for timeframe in timeframes:
        if required_by_timeframe is None:
            timeframe_symbols = list(symbols)
        else:
            timeframe_symbols = list(required_by_timeframe.get(str(timeframe), []))
        if not timeframe_symbols:
            continue
        load_plan.append((str(timeframe), timeframe_symbols))

    total_count = sum(len(timeframe_symbols) for _, timeframe_symbols in load_plan)
    loaded_count = 0
    timeframe_count = len(load_plan)
    for timeframe_index, (timeframe, timeframe_symbols) in enumerate(load_plan, start=1):
        if progress_callback is not None:
            progress_callback(
                "resource_bundle_timeframe_started",
                {
                    "timeframe": timeframe,
                    "timeframe_index": timeframe_index,
                    "timeframe_count": timeframe_count,
                    "symbol_count": len(timeframe_symbols),
                    "loaded_count": loaded_count,
                    "total_count": total_count,
                },
            )
        timeframe_started_at = perf_counter()
        loaded = _load_timeframe_parquet_frames(
            symbols=timeframe_symbols,
            timeframe=timeframe,
            parquet_root=parquet_root,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
            data_mode=data_mode,
            chunk_days=chunk_days,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback(
                "resource_bundle_timeframe_completed",
                {
                    "timeframe": timeframe,
                    "timeframe_index": timeframe_index,
                    "timeframe_count": timeframe_count,
                    "symbol_count": len(timeframe_symbols),
                    "parquet_symbol_count": len(loaded),
                    "missing_symbol_count": max(0, len(timeframe_symbols) - len(loaded)),
                    "loaded_count": loaded_count,
                    "total_count": total_count,
                    "elapsed_seconds": round(max(0.0, perf_counter() - timeframe_started_at), 6),
                },
            )

        symbol_count = len(timeframe_symbols)
        for symbol_index, symbol in enumerate(timeframe_symbols, start=1):
            key = (symbol, timeframe)
            started_at = perf_counter()
            bundle, source = _resolve_bundle_cache_entry(
                symbol=symbol,
                timeframe=timeframe,
                frame=loaded.get(symbol),
                parquet_root=parquet_root,
                exchange=exchange,
                start_date=start_date,
                end_date=end_date,
                allow_csv_fallback=allow_csv_fallback,
                allow_synthetic_fallback=allow_synthetic_fallback,
                min_bars=min_bars,
                partitioned_1m_cache=partitioned_1m_cache,
            )
            cache[key] = bundle
            source_map.setdefault(source, [])
            source_map[source].append(f"{symbol}@{timeframe}")
            loaded_count += 1
            if progress_callback is not None:
                progress_callback(
                    "resource_bundle_item_loaded",
                    {
                        "symbol": symbol,
                        "symbol_index": symbol_index,
                        "symbol_count": symbol_count,
                        "timeframe": timeframe,
                        "timeframe_index": timeframe_index,
                        "timeframe_count": timeframe_count,
                        "loaded_count": loaded_count,
                        "total_count": total_count,
                        "source": source,
                        "bar_count": len(bundle.datetime),
                        "elapsed_seconds": round(max(0.0, perf_counter() - started_at), 6),
                    },
                )

    return cache, source_map


def _benchmark_cache(
    cache: Mapping[tuple[str, str], SeriesBundle],
    timeframes: Sequence[str],
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    timeframe_count = len(timeframes)
    for timeframe_index, tf in enumerate(timeframes, start=1):
        if progress_callback is not None:
            progress_callback(
                "resource_benchmark_timeframe_started",
                {
                    "timeframe": tf,
                    "timeframe_index": timeframe_index,
                    "timeframe_count": timeframe_count,
                    "built_count": max(0, timeframe_index - 1),
                },
            )
        started_at = perf_counter()
        bundle = cache.get(("BTC/USDT", tf))
        if bundle is None:
            out[tf] = {
                "datetime": np.asarray([], dtype="datetime64[ms]"),
                "returns": np.asarray([], dtype=float),
            }
        else:
            out[tf] = {
                "datetime": np.asarray(bundle.datetime, dtype="datetime64[ms]"),
                "returns": _returns_from_close(bundle.close),
            }
        if progress_callback is not None:
            progress_callback(
                "resource_benchmark_timeframe_built",
                {
                    "timeframe": tf,
                    "timeframe_index": timeframe_index,
                    "timeframe_count": timeframe_count,
                    "built_count": timeframe_index,
                    "return_count": len(np.asarray(out[tf]["returns"], dtype=float)),
                    "elapsed_seconds": round(max(0.0, perf_counter() - started_at), 6),
                },
            )
    return out


def _build_default_split(strategy_timeframe: str) -> dict[str, Any]:
    return _research_run_support._build_default_split(strategy_timeframe)


_ResearchRunScoringConfig = _research_run_support._ResearchRunScoringConfig


def _normalize_candidate_research_base_timeframe(base_timeframe: str) -> str:
    return _research_run_support._normalize_candidate_research_base_timeframe(base_timeframe)


def _resolve_research_run_scoring_config(
    *,
    score_config: Mapping[str, Any] | None,
    stage1_keep_ratio: float,
) -> _ResearchRunScoringConfig:
    return _research_run_support._resolve_research_run_scoring_config(
        score_config=score_config,
        stage1_keep_ratio=stage1_keep_ratio,
    )


def _adapt_candidate_inputs(
    candidates: Iterable[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    return _research_run_support._adapt_candidate_inputs(
        candidates,
        max_candidates=max_candidates,
    )


def _empty_candidate_research_report(
    *,
    base_timeframe: str,
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
    stage1_keep_ratio: float,
    scoring: _ResearchRunScoringConfig,
    split: Mapping[str, Any] | None,
    split_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _research_run_support._empty_candidate_research_report(
        base_timeframe=base_timeframe,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        split=split,
        split_overrides=split_overrides,
    )


def _resolve_research_run_timeframes_and_universe(
    *,
    adapted: Sequence[dict[str, Any]],
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    return _research_run_support._resolve_research_run_timeframes_and_universe(
        adapted=adapted,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
    )


def _research_resource_loader():
    return _research_stage_support.build_research_resource_loader(
        split_window_bounds=_split_window_bounds,
        datetime_to_iso_z=_datetime_to_iso_z,
        load_bundle_cache=_load_bundle_cache,
        load_feature_cache=_load_feature_cache,
        benchmark_cache=_benchmark_cache,
        canonicalize_symbol_list=canonicalize_symbol_list,
    )


def _load_research_run_resources(
    *,
    adapted: Sequence[dict[str, Any]],
    normalized_timeframes: Sequence[str],
    universe: Sequence[str],
    resolved_split: Mapping[str, Any],
    data_mode: str,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bundle_bars: int,
    market_data_settings: Mapping[str, Any] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[
    dict[tuple[str, str], SeriesBundle],
    dict[str, list[str]],
    dict[str, pl.DataFrame],
    dict[str, dict[str, np.ndarray]],
]:
    return _research_stage_support.load_research_run_resources(
        loader=_research_resource_loader(),
        adapted=adapted,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        data_mode=data_mode,
        allow_csv_fallback=allow_csv_fallback,
        allow_synthetic_fallback=allow_synthetic_fallback,
        min_bundle_bars=min_bundle_bars,
        market_data_settings=market_data_settings,
        progress_callback=progress_callback,
    )


def _evaluate_candidate_with_optional_split(
    candidate: dict[str, Any],
    *,
    cache: Mapping[tuple[str, str], SeriesBundle],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    benchmark_cache: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    candidate_count: int,
    scoring_config: Mapping[str, Any] | None,
    split: Mapping[str, Any],
) -> dict[str, Any]:
    return _research_stage_support.evaluate_candidate_with_optional_split(
        evaluate_candidate=_evaluate_candidate,
        stage1_prefilter_score=_stage1_prefilter_score,
        candidate=candidate,
        cache=cache,
        feature_cache=feature_cache,
        benchmark_cache=benchmark_cache,
        candidate_count=candidate_count,
        scoring_config=scoring_config,
        split=split,
    )


def _select_stage2_results(
    *,
    adapted: Sequence[dict[str, Any]],
    cache: Mapping[tuple[str, str], SeriesBundle],
    feature_cache: Mapping[str, pl.DataFrame] | None,
    aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] | None,
    benchmark: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    scoring: _ResearchRunScoringConfig,
    resolved_split: Mapping[str, Any],
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    return _research_stage_support.select_stage2_results(
        evaluate_candidate=_evaluate_candidate,
        stage1_prefilter_score=_stage1_prefilter_score,
        adapted=adapted,
        cache=cache,
        feature_cache=feature_cache,
        aligned_cache=aligned_cache,
        benchmark=benchmark,
        scoring=scoring,
        resolved_split=resolved_split,
        progress_callback=progress_callback,
    )


def _research_report_builder() -> ResearchReportBuilder:
    return ResearchReportBuilder(
        split_masks_from_datetimes=_split_masks_from_datetimes,
        split_lengths=_split_lengths,
        compute_metrics=_compute_metrics,
        hurdle_fields=_hurdle_fields,
        family_from_strategy=_family_from_strategy,
        canonicalize_symbol_list=canonicalize_symbol_list,
        series_to_stream=_series_to_stream,
        candidate_rank_score=_candidate_rank_score,
        correlation=_research_metrics.correlation,
        periods_per_year=_PERIODS_PER_YEAR,
        metric_config=get_default_runtime_config().backtest,
    )


def _result_timestamps_and_split_masks(
    result: Mapping[str, Any],
    *,
    resolved_split: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    return _research_report_builder().result_timestamps_and_split_masks(
        result,
        resolved_split=resolved_split,
    )


def _error_candidate_report_payload(
    *,
    result: Mapping[str, Any],
    resolved_scoring_config: Mapping[str, Any],
    failed_candidate_selection_score: float,
    candidate_count: int,
) -> dict[str, Any]:
    return _research_report_builder().error_candidate_report_payload(
        result=result,
        resolved_scoring_config=resolved_scoring_config,
        failed_candidate_selection_score=failed_candidate_selection_score,
        candidate_count=candidate_count,
    )


def _candidate_return_streams(
    *,
    returns: np.ndarray,
    timestamps: np.ndarray,
    split_masks: Mapping[str, np.ndarray],
    has_aligned_timestamps: bool,
) -> dict[str, list[dict[str, float | int]]]:
    return _research_report_builder().candidate_return_streams(
        returns=returns,
        timestamps=timestamps,
        split_masks=split_masks,
        has_aligned_timestamps=has_aligned_timestamps,
    )


def _successful_candidate_report_payload(
    *,
    result: Mapping[str, Any],
    resolved_split: Mapping[str, Any],
    resolved_scoring_config: Mapping[str, Any],
) -> dict[str, Any]:
    return _research_report_builder().successful_candidate_report_payload(
        result=result,
        resolved_split=resolved_split,
        resolved_scoring_config=resolved_scoring_config,
    )


def _report_candidates_from_stage2_results(
    *,
    stage2_results: Sequence[dict[str, Any]],
    candidate_count: int,
    resolved_split: Mapping[str, Any],
    scoring: _ResearchRunScoringConfig,
) -> list[dict[str, Any]]:
    from lumina_quant.strategy_factory.research_entrypoints import (
        _report_candidates_from_stage2_results as _report_candidates_from_stage2_results_impl,
    )

    return _report_candidates_from_stage2_results_impl(
        stage2_results=stage2_results,
        candidate_count=candidate_count,
        resolved_split=resolved_split,
        scoring=scoring,
    )


def _attach_cross_candidate_correlations(report_candidates: Sequence[dict[str, Any]]) -> None:
    _research_report_builder().attach_cross_candidate_correlations(report_candidates)


def _sorted_report_candidates(
    report_candidates: Sequence[dict[str, Any]],
    *,
    scoring: _ResearchRunScoringConfig,
) -> list[dict[str, Any]]:
    rows = list(report_candidates)
    _attach_cross_candidate_correlations(rows)
    rows.sort(
        key=lambda item: float(item.get("selection_score", scoring.sort_missing_selection_score)),
        reverse=True,
    )
    return rows


def _candidate_research_report_payload(
    *,
    base_tf: str,
    normalized_timeframes: Sequence[str],
    universe: Sequence[str],
    resolved_split: Mapping[str, Any],
    adapted: Sequence[dict[str, Any]],
    stage2_results: Sequence[dict[str, Any]],
    stage1_keep_ratio: float,
    scoring: _ResearchRunScoringConfig,
    data_sources: dict[str, list[str]],
    report_candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    from lumina_quant.strategy_factory.research_entrypoints import (
        _candidate_research_report_payload as _candidate_research_report_payload_impl,
    )

    return _candidate_research_report_payload_impl(
        base_tf=base_tf,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        adapted=adapted,
        stage2_results=stage2_results,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        data_sources=data_sources,
        report_candidates=report_candidates,
    )


def _run_candidate_research_with_adapted_candidates(
    *,
    base_tf: str,
    adapted: Sequence[dict[str, Any]],
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
    stage1_keep_ratio: float,
    scoring: _ResearchRunScoringConfig,
    split: Mapping[str, Any] | None,
    data_mode: str,
    allow_csv_fallback: bool,
    allow_synthetic_fallback: bool,
    min_bundle_bars: int,
    market_data_settings: Mapping[str, Any],
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    normalized_timeframes, universe = _resolve_research_run_timeframes_and_universe(
        adapted=adapted,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
    )
    split_timeframe = normalized_timeframes[0] if normalized_timeframes else "1m"
    resolved_split = _resolve_split_config(split, strategy_timeframe=split_timeframe)

    if progress_callback is not None:
        progress_callback(
            "resource_load_started",
            {
                "candidate_count": len(adapted),
                "normalized_timeframes": list(normalized_timeframes),
                "symbol_universe": list(universe),
            },
        )
    cache, data_sources, feature_cache, benchmark = _load_research_run_resources(
        adapted=adapted,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        data_mode=str(data_mode or "legacy"),
        allow_csv_fallback=bool(allow_csv_fallback),
        allow_synthetic_fallback=bool(allow_synthetic_fallback),
        min_bundle_bars=max(1, int(min_bundle_bars)),
        market_data_settings=market_data_settings,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            "resources_loaded",
            {
                "candidate_count": len(adapted),
                "normalized_timeframes": list(normalized_timeframes),
                "symbol_universe": list(universe),
                "bundle_count": len(cache),
                "feature_frame_count": len(feature_cache or {}),
                "benchmark_count": len(benchmark),
            },
        )
    aligned_cache: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    stage2_results = _select_stage2_results(
        adapted=adapted,
        cache=cache,
        feature_cache=feature_cache,
        aligned_cache=aligned_cache,
        benchmark=benchmark,
        scoring=scoring,
        resolved_split=resolved_split,
        progress_callback=progress_callback,
    )
    report_candidates = _report_candidates_from_stage2_results(
        stage2_results=stage2_results,
        candidate_count=len(adapted),
        resolved_split=resolved_split,
        scoring=scoring,
    )
    sorted_candidates = _sorted_report_candidates(report_candidates, scoring=scoring)
    if progress_callback is not None:
        progress_callback(
            "report_ready",
            {
                "reported_candidate_count": len(sorted_candidates),
                "top_report_candidates": [
                    {
                        "candidate_id": str(row.get("candidate_id") or row.get("name") or ""),
                        "name": str(row.get("name") or ""),
                        "selection_score": float(row.get("selection_score", 0.0) or 0.0),
                        "oos_total_return": float(
                            dict(row.get("oos") or {}).get(
                                "total_return",
                                dict(row.get("oos") or {}).get("return", 0.0),
                            )
                            or 0.0
                        ),
                        "oos_sharpe": float(dict(row.get("oos") or {}).get("sharpe", 0.0) or 0.0),
                    }
                    for row in sorted_candidates[: min(5, len(sorted_candidates))]
                ],
            },
        )
    return _candidate_research_report_payload(
        base_tf=base_tf,
        normalized_timeframes=normalized_timeframes,
        universe=universe,
        resolved_split=resolved_split,
        adapted=adapted,
        stage2_results=stage2_results,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        data_sources=data_sources,
        report_candidates=sorted_candidates,
    )


def run_candidate_research(
    *,
    candidates: Iterable[dict[str, Any]],
    base_timeframe: str = "1s",
    strategy_timeframes: Sequence[str] | None = None,
    symbol_universe: Sequence[str] | None = None,
    stage1_keep_ratio: float = 0.35,
    max_candidates: int = 512,
    score_config: Mapping[str, Any] | None = None,
    split: Mapping[str, Any] | None = None,
    data_mode: str = "legacy",
    allow_csv_fallback: bool = True,
    allow_synthetic_fallback: bool = False,
    min_bundle_bars: int = _MIN_BARS,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate candidate manifest into train/val/OOS report contract (v2)."""
    base_tf = _normalize_candidate_research_base_timeframe(base_timeframe)
    market_data_settings = _current_research_market_data_settings()
    scoring = _resolve_research_run_scoring_config(
        score_config=score_config,
        stage1_keep_ratio=stage1_keep_ratio,
    )
    adapted = _adapt_candidate_inputs(candidates, max_candidates=max_candidates)

    if not adapted:
        return _empty_candidate_research_report(
            base_timeframe=base_tf,
            strategy_timeframes=strategy_timeframes,
            symbol_universe=symbol_universe,
            stage1_keep_ratio=stage1_keep_ratio,
            scoring=scoring,
            split=split,
        )

    return _run_candidate_research_with_adapted_candidates(
        base_tf=base_tf,
        adapted=adapted,
        strategy_timeframes=strategy_timeframes,
        symbol_universe=symbol_universe,
        stage1_keep_ratio=stage1_keep_ratio,
        scoring=scoring,
        split=split,
        data_mode=data_mode,
        allow_csv_fallback=allow_csv_fallback,
        allow_synthetic_fallback=allow_synthetic_fallback,
        min_bundle_bars=min_bundle_bars,
        market_data_settings=market_data_settings,
        progress_callback=progress_callback,
    )


def build_default_candidate_rows(
    *,
    symbols: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    max_candidates: int = 512,
) -> list[dict[str, Any]]:
    """Build candidate rows from strategy-factory candidate library."""
    from lumina_quant.strategy_factory.research_entrypoints import (
        build_default_candidate_rows as _build_default_candidate_rows_entrypoint,
    )

    return _build_default_candidate_rows_entrypoint(
        symbols=symbols,
        timeframes=timeframes,
        max_candidates=max_candidates,
    )


__all__ = [
    "adapt_legacy_candidate",
    "build_default_candidate_rows",
    "run_candidate_research",
]
