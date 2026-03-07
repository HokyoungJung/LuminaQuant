from __future__ import annotations

from lumina_quant.market_data import (
    load_futures_feature_points_from_db,
    upsert_futures_feature_points_rows,
)


def test_feature_point_upsert_handles_sparse_mixed_numeric_rows(tmp_path):
    db_path = tmp_path / "market_parquet"

    upserted = upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="XAU/USDT",
        rows=[
            {"timestamp_ms": 1_700_000_000_000, "open_interest": 1},
            {
                "timestamp_ms": 1_700_000_060_000,
                "funding_rate": 0.00031,
                "funding_fee_rate": 0.00031,
                "funding_fee_quote_per_unit": 0.62,
            },
            {
                "timestamp_ms": 1_700_000_120_000,
                "liquidation_long_qty": 2,
                "liquidation_long_notional": 1.5,
            },
        ],
    )

    assert upserted == 3

    frame = load_futures_feature_points_from_db(
        str(db_path),
        exchange="binance",
        symbol="XAU/USDT",
    )
    assert frame.height == 3
    assert frame.get_column("funding_rate").drop_nulls().to_list() == [0.00031]
    assert frame.get_column("funding_fee_rate").drop_nulls().to_list() == [0.00031]
    assert frame.get_column("funding_fee_quote_per_unit").drop_nulls().to_list() == [0.62]
