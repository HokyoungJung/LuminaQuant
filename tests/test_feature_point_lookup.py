from __future__ import annotations

from datetime import UTC, datetime

import pytest
import polars as pl

import lumina_quant.data.feature_points as feature_points
from lumina_quant.data.feature_points import (
    BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS,
    FEATURE_POINT_MAX_STALE_MS,
    FeaturePointLookup,
)
from lumina_quant.market_data import upsert_futures_feature_points_rows


def test_feature_point_lookup_forward_fills_latest_non_null_value(tmp_path):
    db_path = tmp_path / "market_parquet"
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "timestamp_ms": 1_700_000_000_000,
                "funding_rate": 0.0001,
                "funding_fee_rate": 0.0001,
                "funding_fee_quote_per_unit": 5.0,
            },
            {"timestamp_ms": 1_700_000_060_000, "mark_price": 50_000.0},
            {"timestamp_ms": 1_700_000_120_000, "open_interest": 1_250_000.0},
        ],
    )

    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=1_700_000_060_000) == 0.0001
    assert (
        lookup.get_latest("BTC/USDT", "funding_fee_rate", timestamp_ms=1_700_000_060_000) == 0.0001
    )
    assert (
        lookup.get_latest("BTC/USDT", "funding_fee_quote_per_unit", timestamp_ms=1_700_000_060_000)
        == 5.0
    )
    assert lookup.get_latest("BTC/USDT", "mark_price", timestamp_ms=1_700_000_060_000) == 50_000.0
    assert lookup.get_latest("BTC/USDT", "open_interest", timestamp_ms=1_700_000_060_000) is None
    assert (
        lookup.get_latest("BTC/USDT", "open_interest", timestamp_ms=1_700_000_120_000)
        == 1_250_000.0
    )
    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=1_699_999_000_000) is None


def test_feature_point_lookup_sums_raw_feature_window(tmp_path):
    db_path = tmp_path / "market_parquet"
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="ETH/USDT",
        rows=[
            {"timestamp_ms": 1_700_000_000_000, "taker_buy_quote_volume": 10.0},
            {"timestamp_ms": 1_700_000_060_000, "taker_buy_quote_volume": 15.0},
            {"timestamp_ms": 1_700_000_120_000, "taker_buy_quote_volume": None},
            {"timestamp_ms": 1_700_000_180_000, "taker_buy_quote_volume": 5.0},
        ],
    )

    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    assert (
        lookup.sum_between(
            "ETH/USDT",
            "taker_buy_quote_volume",
            start_timestamp_ms=1_700_000_000_000,
            end_timestamp_ms=1_700_000_120_000,
        )
        == 25.0
    )
    assert (
        lookup.sum_between(
            "ETH/USDT",
            "taker_buy_quote_volume",
            start_timestamp_ms=1_700_000_120_001,
            end_timestamp_ms=1_700_000_180_000,
        )
        == 5.0
    )
    assert (
        lookup.sum_between(
            "ETH/USDT",
            "taker_buy_quote_volume",
            start_timestamp_ms=1_700_000_240_000,
            end_timestamp_ms=1_700_000_300_000,
        )
        is None
    )


def test_feature_point_lookup_sums_exact_funding_settlements(tmp_path):
    db_path = tmp_path / "market_parquet"
    interval_ms = 8 * 60 * 60 * 1000
    start_ms = 1_767_225_600_000  # 2026-01-01 00:00 UTC
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": start_ms + interval_ms, "funding_fee_quote_per_unit": 0.1},
            {"timestamp_ms": start_ms + 2 * interval_ms, "funding_fee_quote_per_unit": 0.4},
            {"timestamp_ms": start_ms + 3 * interval_ms, "funding_fee_quote_per_unit": -0.3},
        ],
    )

    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    value, complete = lookup.funding_fee_sum_between(
        "BTC/USDT",
        start_timestamp_ms=start_ms,
        end_timestamp_ms=start_ms + 3 * interval_ms,
        interval_ms=interval_ms,
    )
    assert value == pytest.approx(0.2)
    assert complete is True


def test_feature_point_lookup_accepts_binance_settlement_source_jitter(tmp_path):
    db_path = tmp_path / "market_parquet"
    interval_ms = 8 * 60 * 60 * 1000
    start_ms = 1_767_225_600_000  # 2026-01-01 00:00 UTC
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": start_ms + interval_ms + 1, "funding_fee_quote_per_unit": 0.1},
            {
                "timestamp_ms": start_ms + 2 * interval_ms + 29,
                "funding_fee_quote_per_unit": 0.4,
            },
        ],
    )

    coverage = FeaturePointLookup(db_path=str(db_path), exchange="binance").funding_fee_sum_between(
        "BTC/USDT",
        start_timestamp_ms=start_ms,
        end_timestamp_ms=start_ms + 2 * interval_ms + BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS,
        interval_ms=interval_ms,
    )

    assert coverage.fee_sum == pytest.approx(0.5)
    assert coverage.complete is True
    assert coverage.last_consecutive_boundary_ms == start_ms + 2 * interval_ms


def test_feature_point_lookup_defers_current_boundary_and_rejects_ambiguous_evidence(tmp_path):
    db_path = tmp_path / "market_parquet"
    interval_ms = 8 * 60 * 60 * 1000
    start_ms = 1_767_225_600_000  # 2026-01-01 00:00 UTC
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": start_ms + interval_ms + 1, "funding_fee_quote_per_unit": 0.1},
            {"timestamp_ms": start_ms + interval_ms + 2, "funding_fee_quote_per_unit": 0.2},
        ],
    )
    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    deferred = lookup.funding_fee_sum_between(
        "BTC/USDT",
        start_timestamp_ms=start_ms,
        end_timestamp_ms=start_ms + interval_ms,
        interval_ms=interval_ms,
    )
    assert deferred.deferred_boundary_ms == start_ms + interval_ms
    assert deferred.missing_boundary_ms is None

    invalid = lookup.funding_fee_sum_between(
        "BTC/USDT",
        start_timestamp_ms=start_ms,
        end_timestamp_ms=start_ms + interval_ms + BINANCE_FUNDING_SOURCE_JITTER_TOLERANCE_MS,
        interval_ms=interval_ms,
    )
    assert invalid.invalid_boundary_ms == start_ms + interval_ms
    assert invalid.complete is False


def test_feature_point_lookup_rejects_duplicate_same_timestamp_funding_evidence(monkeypatch):
    rows = pl.DataFrame(
        {
            "timestamp_ms": [1_767_254_400_000, 1_767_254_400_000],
            "funding_fee_quote_per_unit": [0.1, 0.2],
        }
    )
    monkeypatch.setattr(
        feature_points,
        "load_futures_feature_points_from_db",
        lambda *args, **kwargs: rows,
    )

    with pytest.raises(ValueError, match="duplicate funding evidence timestamps"):
        FeaturePointLookup(db_path="unused").get_latest(
            "BTC/USDT",
            "funding_fee_quote_per_unit",
            timestamp_ms=1_767_254_400_000,
        )


def test_named_suite_preflights_funding_when_strategy_declares_no_features():
    from scripts.research.run_named_quant_suite import _preflight_required_features

    class EmptyFeatureRepository:
        def load_futures_feature_points(self, **_kwargs):
            return pl.DataFrame()

    with pytest.raises(RuntimeError, match="BTC/USDT:funding_fee_quote_per_unit"):
        _preflight_required_features(
            EmptyFeatureRepository(),
            exchange="binance",
            symbols=["BTC/USDT"],
            required_features=(),
            require_utc_funding=True,
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 1, 8, tzinfo=UTC),
        )


def test_feature_point_lookup_does_not_forward_fill_beyond_staleness_limit(tmp_path):
    db_path = tmp_path / "market_parquet"
    start_ms = 1_700_000_000_000
    stale_ms = start_ms + FEATURE_POINT_MAX_STALE_MS + 1
    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {"timestamp_ms": start_ms, "funding_rate": 0.0001},
            {"timestamp_ms": stale_ms, "mark_price": 50_000.0},
        ],
    )

    lookup = FeaturePointLookup(db_path=str(db_path), exchange="binance")

    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=start_ms) == 0.0001
    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=stale_ms) is None
