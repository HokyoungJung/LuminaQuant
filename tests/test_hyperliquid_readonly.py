from __future__ import annotations

from lumina_quant.data.hyperliquid_readonly import (
    parse_candle_snapshot,
    parse_funding_history_page,
    parse_meta_asset_context_rows,
)


def test_parse_meta_asset_context_rows_maps_exact_feature_fields() -> None:
    payload = [
        {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
        [
            {"funding": "0.0001", "openInterest": "12", "oraclePx": "100.1", "markPx": "100.2"},
            {"funding": "-0.0002", "openInterest": "34", "oraclePx": "200.1", "markPx": "200.2"},
        ],
    ]

    rows = parse_meta_asset_context_rows(payload, symbols=["ETH/USDT"], timestamp_ms=123)

    assert rows == [
        {
            "symbol": "ETH/USDT",
            "coin": "ETH",
            "timestamp_ms": 123,
            "funding_rate": -0.0002,
            "funding_mark_price": 200.2,
            "mark_price": 200.2,
            "index_price": 200.1,
            "open_interest": 34.0,
            "raw_context": payload[1][1],
            "raw_asset": payload[0]["universe"][1],
        }
    ]


def test_parse_funding_history_page_sorts_and_skips_invalid_rows() -> None:
    page = parse_funding_history_page(
        "ETH",
        [
            {"coin": "ETH", "time": 3, "fundingRate": "bad"},
            {"coin": "ETH", "time": 2, "fundingRate": "0.02", "premium": "0.1"},
            {"coin": "ETH", "time": 1, "fundingRate": "0.01", "premium": None},
        ],
    )

    assert page.first_timestamp_ms == 1
    assert page.last_timestamp_ms == 2
    assert [row["timestamp_ms"] for row in page.rows] == [1, 2]
    assert [row["funding_rate"] for row in page.rows] == [0.01, 0.02]
    assert page.rows[1]["raw_premium"] == 0.1


def test_parse_candle_snapshot_keeps_candles_as_non_rawfirst_context() -> None:
    rows = parse_candle_snapshot(
        [
            {
                "t": 10,
                "T": 19,
                "s": "ETH",
                "i": "1h",
                "o": "1",
                "h": "2",
                "l": "0.5",
                "c": "1.5",
                "v": "7",
            },
        ]
    )

    assert rows == [
        {
            "coin": "ETH",
            "interval": "1h",
            "timestamp_ms": 10,
            "end_timestamp_ms": 19,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 7.0,
        }
    ]
