from __future__ import annotations

from lumina_quant.storage.parquet import ParquetMarketDataRepository
from lumina_quant.services.materialize_from_raw import materialize_raw_aggtrades


def test_materialize_from_raw_produces_deterministic_committed_manifest(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_700_000_000_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_700_000_000_500,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
        ],
    )

    first = materialize_raw_aggtrades(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date=None,
        end_date=None,
    )
    second = materialize_raw_aggtrades(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start_date=None,
        end_date=None,
    )

    assert first
    assert second
    assert first[0].canonical_row_checksum == second[0].canonical_row_checksum

    loaded = repo.load_committed_ohlcv_chunked(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
    )
    assert loaded.height >= 1
