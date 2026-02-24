from __future__ import annotations

from pathlib import Path

import polars as pl
from lumina_quant.parquet_market_data import (
    ParquetMarketDataRepository,
    is_parquet_market_data_store,
    load_data_dict_from_parquet,
)


def _sample_1s_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:00:59Z",
                "2026-01-01T00:01:00Z",
            ],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_upsert_1s_writes_partitioned_parquet(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    written = repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())

    assert written == 4

    partition = (
        tmp_path
        / "exchange=binance"
        / "symbol=BTCUSDT"
        / "timeframe=1s"
        / "date=2026-01-01"
    )
    files = list(partition.glob("*.parquet"))
    assert files


def test_load_ohlcv_resamples_with_bucket_groupby(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())

    minute = repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1m")

    assert minute.height == 2
    assert minute["open"].to_list() == [100.0, 103.0]
    assert minute["high"].to_list() == [103.0, 104.0]
    assert minute["low"].to_list() == [99.0, 102.0]
    assert minute["close"].to_list() == [102.5, 103.5]
    assert minute["volume"].to_list() == [6.0, 4.0]


def test_compact_partition_merges_files(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    frame = _sample_1s_frame().head(2)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=frame)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=frame)

    result = repo.compact_partition(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2026-01-01",
        timeframe="1s",
        remove_sources=True,
    )

    assert result.files_before >= 2
    assert result.files_after == 1
    assert result.rows_after == 2


def test_load_data_dict_from_parquet_reads_symbols(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())

    loaded = load_data_dict_from_parquet(
        str(tmp_path),
        exchange="binance",
        symbol_list=["BTC/USDT", "ETH/USDT"],
        timeframe="1m",
        start_date="2026-01-01T00:00:00Z",
        end_date="2026-01-01T00:01:00Z",
        chunk_days=1,
    )

    assert "BTC/USDT" in loaded
    assert "ETH/USDT" not in loaded
    assert loaded["BTC/USDT"].height == 2


def test_is_parquet_market_data_store_detects_existing_partition_layout(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())

    assert is_parquet_market_data_store(str(tmp_path))
    assert is_parquet_market_data_store("anything", backend="parquet")
