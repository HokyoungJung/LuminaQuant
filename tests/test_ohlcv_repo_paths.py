from __future__ import annotations

from datetime import date

from lumina_quant.storage.parquet import ParquetMarketDataRepository


def test_raw_repository_paths_share_normalized_symbol_root(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))

    raw_root = tmp_path / "market_data_raw_aggtrades" / "binance" / "BTCUSDT"

    assert (
        repo.raw_partition_path(
            exchange="Binance",
            symbol="btc/usdt",
            partition_date=date(2025, 1, 2),
        )
        == raw_root / "date=2025-01-02" / "part-0000.parquet"
    )
    assert (
        repo.raw_checkpoint_path(exchange="Binance", symbol="btc/usdt")
        == raw_root / "checkpoint.json"
    )
    assert repo.raw_wal_path(exchange="Binance", symbol="btc/usdt") == raw_root / "wal.bin"


def test_materialized_repository_paths_accept_date_objects_and_normalize_timeframes(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))

    expected_root = (
        tmp_path
        / "market_data_materialized"
        / "binance"
        / "BTCUSDT"
        / "timeframe=1h"
        / "date=2025-01-02"
    )

    assert (
        repo.materialized_partition_root(
            exchange="Binance",
            symbol="BTC/USDT",
            timeframe="1H",
            partition_date=date(2025, 1, 2),
        )
        == expected_root
    )
    assert (
        repo.materialized_manifest_path(
            exchange="Binance",
            symbol="BTC/USDT",
            timeframe="1H",
            partition_date=date(2025, 1, 2),
        )
        == expected_root / "manifest.json"
    )
