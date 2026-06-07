from __future__ import annotations

from lumina_quant.live.binance_market_stream import NormalizedBookTicker
from scripts import collect_binance_book_ticker_feature_points as module


def test_bucket_start_ms_rounds_down() -> None:
    assert module._bucket_start_ms(123_456, 60) == 120_000
    assert module._bucket_start_ms(123_456, 15) == 120_000


def test_snapshot_row_includes_bbo_fields() -> None:
    book = NormalizedBookTicker(
        symbol="BTC/USDT",
        exchange_ts_ms=123_456,
        bid_price=100.0,
        bid_quantity=2.0,
        ask_price=100.2,
        ask_quantity=3.0,
        receive_ts_ms=123_500,
    )

    row = module._snapshot_row(book, cadence_seconds=60)

    assert row["timestamp_ms"] == 120_000
    assert row["best_bid_price"] == 100.0
    assert row["best_ask_price"] == 100.2
    assert row["bbo_mid_price"] == 100.1
    assert row["bbo_spread_bps"] > 0.0


def test_summary_line_is_single_line() -> None:
    payload = {"total_persisted_rows": 5, "captured_bucket_count": 2, "errors": []}
    assert module._summary_line(payload) == "bbo-capture rows=5 buckets=2 errors=0"
