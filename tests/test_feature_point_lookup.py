from __future__ import annotations

from lumina_quant.data.feature_points import FEATURE_POINT_MAX_STALE_MS, FeaturePointLookup
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
