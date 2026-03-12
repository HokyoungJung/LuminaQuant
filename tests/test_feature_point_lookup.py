from __future__ import annotations

from lumina_quant.data.feature_points import FeaturePointLookup
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
    assert lookup.get_latest("BTC/USDT", "funding_fee_rate", timestamp_ms=1_700_000_060_000) == 0.0001
    assert lookup.get_latest("BTC/USDT", "funding_fee_quote_per_unit", timestamp_ms=1_700_000_060_000) == 5.0
    assert lookup.get_latest("BTC/USDT", "mark_price", timestamp_ms=1_700_000_060_000) == 50_000.0
    assert lookup.get_latest("BTC/USDT", "open_interest", timestamp_ms=1_700_000_060_000) is None
    assert lookup.get_latest("BTC/USDT", "open_interest", timestamp_ms=1_700_000_120_000) == 1_250_000.0
    assert lookup.get_latest("BTC/USDT", "funding_rate", timestamp_ms=1_699_999_000_000) is None
