from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from lumina_quant.storage.parquet import ParquetMarketDataRepository
import pytest
from lumina_quant.storage.parquet.ohlcv_repo import SealedMonthlyPartitionConflictError


def _frame(rows):
    return pl.DataFrame(rows)


def _sealed_source(tmp_path: Path) -> tuple[Path, str, int, int]:
    source = tmp_path / "staging.parquet"
    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 4, 1, 0, 0), datetime(2026, 4, 1, 0, 0, 1)],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1.0, 2.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime("ms")))
    frame.write_parquet(source)
    raw = source.read_bytes()
    from hashlib import sha256

    return source, sha256(raw).hexdigest(), len(raw), frame.height


def test_wal_compaction_merges_into_monthly_parquet_and_dedupes(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    symbol = "BTC/USDT"

    repo.upsert_1s(
        exchange="binance",
        symbol=symbol,
        rows=_frame(
            [
                {
                    "datetime": "2026-01-31T23:59:59Z",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1.0,
                },
                {
                    "datetime": "2026-02-01T00:00:00Z",
                    "open": 200.0,
                    "high": 201.0,
                    "low": 199.0,
                    "close": 200.5,
                    "volume": 2.0,
                },
            ]
        ),
    )
    # Duplicate ts in WAL; latest should win.
    repo.upsert_1s(
        exchange="binance",
        symbol=symbol,
        rows=_frame(
            [
                {
                    "datetime": "2026-02-01T00:00:00Z",
                    "open": 210.0,
                    "high": 211.0,
                    "low": 209.0,
                    "close": 210.5,
                    "volume": 3.0,
                }
            ]
        ),
    )

    results = repo.compact_all(
        exchange="binance", symbol=symbol, timeframe="1s", remove_sources=True
    )
    assert results

    symbol_root = tmp_path / "market_ohlcv_1s" / "binance" / "BTCUSDT"
    assert (symbol_root / "2026-01.parquet").exists()
    assert (symbol_root / "2026-02.parquet").exists()
    assert (symbol_root / "wal.bin").exists()
    assert (symbol_root / "wal.bin").stat().st_size == 0

    frame = repo.load_ohlcv(
        exchange="binance",
        symbol=symbol,
        timeframe="1s",
        start_date=datetime(2026, 1, 31, 23, 59, 59),
        end_date=datetime(2026, 2, 1, 0, 0, 0),
    )
    assert frame.height == 2
    assert frame["close"].to_list() == [100.5, 210.5]


def test_load_path_merges_monthly_parquet_and_live_wal(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    symbol = "ETH/USDT"

    repo.upsert_1s(
        exchange="binance",
        symbol=symbol,
        rows=_frame(
            [
                {
                    "datetime": "2026-03-01T00:00:00Z",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1.0,
                }
            ]
        ),
    )
    repo.compact_all(exchange="binance", symbol=symbol, timeframe="1s", remove_sources=True)

    # Fresh row remains only in WAL until next compaction.
    repo.upsert_1s(
        exchange="binance",
        symbol=symbol,
        rows=_frame(
            [
                {
                    "datetime": "2026-03-01T00:00:00Z",
                    "open": 11.0,
                    "high": 11.0,
                    "low": 11.0,
                    "close": 11.0,
                    "volume": 2.0,
                },
                {
                    "datetime": "2026-03-01T00:00:01Z",
                    "open": 12.0,
                    "high": 12.0,
                    "low": 12.0,
                    "close": 12.0,
                    "volume": 2.0,
                },
            ]
        ),
    )

    merged = repo.load_ohlcv(
        exchange="binance",
        symbol=symbol,
        timeframe="1s",
        start_date="2026-03-01T00:00:00Z",
        end_date="2026-03-01T00:00:01Z",
    )
    assert merged.height == 2
    assert merged["close"].to_list() == [11.0, 12.0]


def test_publish_sealed_monthly_partition_is_idempotent_and_preserves_staging(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    source, digest, byte_count, rows = _sealed_source(tmp_path)
    before = source.read_bytes()

    target = repo.publish_sealed_monthly_partition(
        exchange="binance",
        symbol="BTC/USDT",
        month="2026-04",
        source=source,
        expected_sha256=digest,
        expected_byte_count=byte_count,
        expected_row_count=rows,
        provenance_receipt_sha256="a" * 64,
    )
    assert target.read_bytes() == before
    assert target.stat().st_nlink == source.stat().st_nlink == 1
    assert source.read_bytes() == before
    assert (target.with_suffix(".seal.json")).exists()
    assert not target.with_suffix(".pending.json").exists()

    assert (
        repo.publish_sealed_monthly_partition(
            exchange="binance",
            symbol="BTC/USDT",
            month="2026-04",
            source=source,
            expected_sha256=digest,
            expected_byte_count=byte_count,
            expected_row_count=rows,
            provenance_receipt_sha256="a" * 64,
        )
        == target
    )


def test_publish_sealed_monthly_partition_rejects_conflicts_and_compaction(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    source, digest, byte_count, rows = _sealed_source(tmp_path)
    target = repo.publish_sealed_monthly_partition(
        exchange="binance",
        symbol="BTC/USDT",
        month="2026-04",
        source=source,
        expected_sha256=digest,
        expected_byte_count=byte_count,
        expected_row_count=rows,
        provenance_receipt_sha256="b" * 64,
    )

    with pytest.raises(SealedMonthlyPartitionConflictError):
        repo.publish_sealed_monthly_partition(
            exchange="binance",
            symbol="BTC/USDT",
            month="2026-04",
            source=source,
            expected_sha256=digest,
            expected_byte_count=byte_count,
            expected_row_count=rows,
            provenance_receipt_sha256="c" * 64,
        )

    with pytest.raises(SealedMonthlyPartitionConflictError):
        repo.upsert_1s(
            exchange="binance",
            symbol="BTC/USDT",
            rows=_frame(
                [
                    {
                        "datetime": "2026-04-01T00:00:02Z",
                        "open": 12.0,
                        "high": 13.0,
                        "low": 11.0,
                        "close": 12.5,
                        "volume": 1.0,
                    }
                ]
            ),
        )
    repo.upsert_1s(
        exchange="binance",
        symbol="ETH/USDT",
        rows=_frame(
            [
                {
                    "datetime": "2026-04-01T00:00:02Z",
                    "open": 12.0,
                    "high": 13.0,
                    "low": 11.0,
                    "close": 12.5,
                    "volume": 1.0,
                }
            ]
        ),
    )
    repo.publish_sealed_monthly_partition(
        exchange="binance",
        symbol="ETH/USDT",
        month="2026-04",
        source=source,
        expected_sha256=digest,
        expected_byte_count=byte_count,
        expected_row_count=rows,
        provenance_receipt_sha256="d" * 64,
    )
    with pytest.raises(SealedMonthlyPartitionConflictError):
        repo.compact_wal_to_monthly_parquet(exchange="binance", symbol="ETH/USDT")
    assert target.exists()
