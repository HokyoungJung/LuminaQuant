from __future__ import annotations

from lumina_quant.data.support_inventory import build_strategy_support_inventory
from lumina_quant.market_data import upsert_futures_feature_points_rows


def test_build_strategy_support_inventory_counts_feature_groups(tmp_path):
    db_path = tmp_path / "market_parquet"

    upsert_futures_feature_points_rows(
        str(db_path),
        exchange="binance",
        symbol="XAU/USDT",
        rows=[
            {
                "timestamp_ms": 1_700_000_000_000,
                "funding_rate": 0.0001,
                "funding_fee_rate": 0.0001,
                "funding_fee_quote_per_unit": 5.0,
            },
            {"timestamp_ms": 1_700_000_060_000, "mark_price": 2_050.0, "index_price": 2_049.5},
            {
                "timestamp_ms": 1_700_000_120_000,
                "open_interest": 12_345.0,
                "liquidation_long_qty": 2.0,
                "liquidation_long_notional": 4_100.0,
            },
        ],
    )

    payload = build_strategy_support_inventory(
        db_path=str(db_path),
        exchange="binance",
        symbols=["XAU/USDT", "BTC/USDT"],
    )

    assert payload["symbol_count"] == 2
    rows = {row["symbol"]: row for row in payload["symbols"]}
    xau_row = rows["XAUUSDT"]
    btc_row = rows["BTCUSDT"]
    assert xau_row["symbol"] == "XAUUSDT"
    assert xau_row["rows"] == 3
    assert xau_row["funding_rows"] == 1
    assert xau_row["funding_fee_rows"] == 1
    assert xau_row["mark_rows"] == 1
    assert xau_row["index_rows"] == 1
    assert xau_row["open_interest_rows"] == 1
    assert xau_row["liquidation_rows"] == 1
    assert xau_row["has_funding_fee"] is True
    assert xau_row["has_mark"] is True
    assert xau_row["has_index"] is True
    assert xau_row["has_open_interest"] is True
    assert xau_row["has_liquidation"] is True
    assert xau_row["oi_first_timestamp_ms"] == 1_700_000_120_000
    assert xau_row["oi_last_timestamp_ms"] == 1_700_000_120_000

    assert btc_row["symbol"] == "BTCUSDT"
    assert btc_row["rows"] == 0
    assert btc_row["has_funding_fee"] is False
    assert btc_row["has_mark"] is False
