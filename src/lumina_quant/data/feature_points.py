"""Helpers for querying parquet-backed futures feature points."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from threading import Lock
from typing import Final

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
    "liquidation_long_qty",
    "liquidation_short_qty",
    "liquidation_long_notional",
    "liquidation_short_notional",
)


@dataclass(slots=True)
class _FeatureCache:
    timestamps_ms: list[int]
    columns: dict[str, list[float | None]]


class FeaturePointLookup:
    """Lazy, per-symbol lookup for latest feature values at or before a timestamp."""

    def __init__(self, *, db_path: str | None, exchange: str = "binance") -> None:
        self.db_path = str(db_path or "").strip()
        self.exchange = str(exchange or "binance").strip().lower() or "binance"
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
        )
        if frame.is_empty():
            return _FeatureCache(timestamps_ms=[], columns={field: [] for field in FEATURE_COLUMNS})

        cleaned = frame.filter(pl.col("timestamp_ms").is_not_null()).with_columns(
            pl.col("timestamp_ms").cast(pl.Int64)
        )
        if cleaned.is_empty():
            return _FeatureCache(timestamps_ms=[], columns={field: [] for field in FEATURE_COLUMNS})

        for field in FEATURE_COLUMNS:
            if field not in cleaned.columns:
                cleaned = cleaned.with_columns(pl.lit(None, dtype=pl.Float64).alias(field))

        cleaned = cleaned.select(["timestamp_ms", *FEATURE_COLUMNS]).sort("timestamp_ms").unique(
            subset=["timestamp_ms"],
            keep="last",
        )
        cleaned = cleaned.with_columns(
            [pl.col(field).cast(pl.Float64).fill_null(strategy="forward").alias(field) for field in FEATURE_COLUMNS]
        )

        timestamps_ms = [int(value) for value in cleaned.get_column("timestamp_ms").to_list()]
        columns = {
            field: [float(value) if value is not None else None for value in cleaned.get_column(field).to_list()]
            for field in FEATURE_COLUMNS
        }
        return _FeatureCache(timestamps_ms=timestamps_ms, columns=columns)


__all__ = ["FEATURE_COLUMNS", "FeaturePointLookup"]
