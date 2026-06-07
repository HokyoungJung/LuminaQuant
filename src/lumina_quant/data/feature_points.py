"""Helpers for querying parquet-backed futures feature points."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from threading import Lock
from typing import Any, Final

import polars as pl

from lumina_quant.market_data import load_futures_feature_points_from_db, normalize_symbol

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
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
)


@dataclass(slots=True)
class _FeatureCache:
    timestamps_ms: list[int]
    columns: dict[str, list[float | None]]
    raw_columns: dict[str, list[float | None]]


class FeaturePointLookup:
    """Lazy, per-symbol lookup for latest feature values at or before a timestamp."""

    def __init__(
        self,
        *,
        db_path: str | None,
        exchange: str = "binance",
        start_date: Any = None,
        end_date: Any = None,
    ) -> None:
        self.db_path = str(db_path or "").strip()
        self.exchange = str(exchange or "binance").strip().lower() or "binance"
        self.start_date = start_date
        self.end_date = end_date
        self._cache: dict[str, _FeatureCache] = {}
        self._lock = Lock()

    def get_latest(
        self,
        symbol: str,
        field: str,
        *,
        timestamp_ms: int | None,
    ) -> float | None:
        """Return the latest non-null feature value at or before ``timestamp_ms``."""
        token = str(field or "").strip()
        if not self.db_path or not token or int(timestamp_ms or 0) <= 0:
            return None
        if token not in FEATURE_COLUMNS:
            return None

        cache = self._get_or_load(symbol)
        if not cache.timestamps_ms:
            return None

        idx = bisect_right(cache.timestamps_ms, int(timestamp_ms)) - 1
        if idx < 0:
            return None

        value = cache.columns.get(token, [None])[idx]
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if math.isfinite(parsed) else None

    def sum_between(
        self,
        symbol: str,
        field: str,
        *,
        start_timestamp_ms: int | None,
        end_timestamp_ms: int | None,
    ) -> float | None:
        """Return the sum of non-null feature values in an inclusive ms window."""
        token = str(field or "").strip()
        if (
            not self.db_path
            or not token
            or token not in FEATURE_COLUMNS
            or int(start_timestamp_ms or 0) <= 0
            or int(end_timestamp_ms or 0) <= 0
        ):
            return None
        start_ms = int(start_timestamp_ms or 0)
        end_ms = int(end_timestamp_ms or 0)
        if end_ms < start_ms:
            return None

        cache = self._get_or_load(symbol)
        if not cache.timestamps_ms:
            return None
        left = bisect_right(cache.timestamps_ms, start_ms - 1)
        right = bisect_right(cache.timestamps_ms, end_ms)
        if right <= left:
            return None

        values = [
            float(value)
            for value in cache.raw_columns.get(token, [])[left:right]
            if value is not None and math.isfinite(float(value))
        ]
        if not values:
            return None
        return float(sum(values))

    def _get_or_load(self, symbol: str) -> _FeatureCache:
        normalized = normalize_symbol(symbol)
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
            loaded = self._load_symbol(normalized)
            self._cache[normalized] = loaded
            return loaded

    def _load_symbol(self, symbol: str) -> _FeatureCache:
        frame = load_futures_feature_points_from_db(
            self.db_path,
            exchange=self.exchange,
            symbol=symbol,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        if frame.is_empty():
            empty = {field: [] for field in FEATURE_COLUMNS}
            return _FeatureCache(timestamps_ms=[], columns=empty, raw_columns=dict(empty))

        cleaned = frame.filter(pl.col("timestamp_ms").is_not_null()).with_columns(
            pl.col("timestamp_ms").cast(pl.Int64)
        )
        if cleaned.is_empty():
            empty = {field: [] for field in FEATURE_COLUMNS}
            return _FeatureCache(timestamps_ms=[], columns=empty, raw_columns=dict(empty))

        for field in FEATURE_COLUMNS:
            if field not in cleaned.columns:
                cleaned = cleaned.with_columns(pl.lit(None, dtype=pl.Float64).alias(field))

        cleaned = (
            cleaned.select(["timestamp_ms", *FEATURE_COLUMNS])
            .sort("timestamp_ms")
            .unique(
                subset=["timestamp_ms"],
                keep="last",
            )
        )
        raw_columns = {
            field: [
                float(value) if value is not None else None
                for value in cleaned.get_column(field).to_list()
            ]
            for field in FEATURE_COLUMNS
        }
        cleaned = cleaned.with_columns(
            [
                pl.col(field).cast(pl.Float64).fill_null(strategy="forward").alias(field)
                for field in FEATURE_COLUMNS
            ]
        )

        timestamps_ms = [int(value) for value in cleaned.get_column("timestamp_ms").to_list()]
        columns = {
            field: [
                float(value) if value is not None else None
                for value in cleaned.get_column(field).to_list()
            ]
            for field in FEATURE_COLUMNS
        }
        return _FeatureCache(timestamps_ms=timestamps_ms, columns=columns, raw_columns=raw_columns)


__all__ = ["FEATURE_COLUMNS", "FeaturePointLookup"]
