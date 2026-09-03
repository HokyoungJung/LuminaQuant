from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

import lumina_quant.configuration
from lumina_quant.market_data import MarketDataRepository


def test_funding_settlement_overlay_merges_without_replacing_market_features(
    tmp_path: Path,
) -> None:
    root = tmp_path / "market_parquet"
    repository = MarketDataRepository(str(root))
    timestamp = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
    repository.upsert_futures_feature_points(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[{"timestamp_ms": timestamp, "mark_price": 100.0}],
    )
    overlay = (
        root
        / "funding_settlements/exchange=binance/symbol=BTCUSDT/date=2026-01-01/official.parquet"
    )
    overlay.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "exchange": ["binance"],
            "symbol": ["BTC/USDT"],
            "timestamp_ms": [timestamp],
            "source_timestamp_ms": [timestamp + 2],
            "datetime": ["2026-01-01T00:00:00Z"],
            "source": ["binance_funding_rate_history"],
            "funding_rate": [0.0001],
            "funding_mark_price": [101.0],
            "funding_fee_rate": [0.0001],
            "funding_fee_quote_per_unit": [0.0101],
        }
    ).write_parquet(overlay)

    loaded = repository.load_futures_feature_points(
        exchange="binance",
        symbol="BTC/USDT",
    )

    assert loaded.height == 1
    row = loaded.to_dicts()[0]
    assert row["mark_price"] == 100.0
    assert row["funding_rate"] == 0.0001
    assert row["funding_mark_price"] == 101.0
    assert row["funding_fee_quote_per_unit"] == 0.0101
