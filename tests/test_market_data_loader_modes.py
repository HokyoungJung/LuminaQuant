from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.market_data import load_data_dict_from_parquet
from lumina_quant.storage.parquet import ParquetMarketDataRepository


def _seed_legacy_and_manifest(root) -> None:
    repo = ParquetMarketDataRepository(str(root))
    repo.upsert_1s(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "datetime": datetime(2026, 3, 1, 0, 0, tzinfo=UTC),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.2,
                "volume": 10.0,
            }
        ],
    )

    partition_date = "2026-03-01"
    partition_root = repo.materialized_partition_root(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date=partition_date,
    )
    commit_id = "seed-commit"
    commit_dir = partition_root / f"commit={commit_id}"
    commit_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 3, 1, 0, 0, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.2],
            "volume": [10.0],
        }
    ).with_columns(pl.col("datetime").dt.replace_time_zone(None).cast(pl.Datetime(time_unit="ms")))
    data_file = commit_dir / "part-0000.parquet"
    frame.write_parquet(data_file)
    dt = frame["datetime"][0]
    ts_ms = int(dt.timestamp() * 1000)
    repo.write_materialized_manifest(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date=partition_date,
        payload={
            "manifest_version": 1,
            "commit_id": commit_id,
            "symbol": "BTC/USDT",
            "timeframe": "1s",
            "partition": str(partition_root),
            "window_start_ms": ts_ms,
            "window_end_ms": ts_ms,
            "event_time_watermark_ms": ts_ms,
            "source_checkpoint_start": ts_ms,
            "source_checkpoint_end": ts_ms,
            "row_count": 1,
            "canonical_row_checksum": repo.canonical_row_checksum(frame),
            "data_files": [f"commit={commit_id}/part-0000.parquet"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "producer": "pytest",
            "status": "committed",
        },
    )


def test_loader_supports_legacy_and_raw_first_modes(tmp_path):
    _seed_legacy_and_manifest(tmp_path)

    legacy = load_data_dict_from_parquet(
        str(tmp_path),
        exchange="binance",
        symbol_list=["BTC/USDT"],
        timeframe="1s",
        data_mode="legacy",
    )
    raw_first = load_data_dict_from_parquet(
        str(tmp_path),
        exchange="binance",
        symbol_list=["BTC/USDT"],
        timeframe="1s",
        data_mode="raw-first",
    )

    assert "BTC/USDT" in legacy
    assert "BTC/USDT" in raw_first
    assert raw_first["BTC/USDT"].height == 1


def test_raw_first_missing_symbol_is_fail_fast(tmp_path):
    _seed_legacy_and_manifest(tmp_path)

    with pytest.raises(RawFirstDataMissingError):
        load_data_dict_from_parquet(
            str(tmp_path),
            exchange="binance",
            symbol_list=["BTC/USDT", "ETH/USDT"],
            timeframe="1s",
            data_mode="raw-first",
        )
