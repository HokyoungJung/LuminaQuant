from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest
from lumina_quant.backtesting.cli_contract import RawFirstManifestInvalidError
from lumina_quant.storage.parquet import ParquetMarketDataRepository


def _write_committed_partition(root, *, exchange="binance", symbol="BTC/USDT", timeframe="1s"):
    repo = ParquetMarketDataRepository(str(root))
    partition_date = "2026-03-01"
    partition_root = repo.materialized_partition_root(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        partition_date=partition_date,
    )
    commit_id = "20260301-checksum"
    commit_dir = partition_root / f"commit={commit_id}"
    commit_dir.mkdir(parents=True, exist_ok=True)

    frame = pl.DataFrame(
        {
            "datetime": [datetime(2026, 3, 1, 0, 0, tzinfo=UTC)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.5],
            "close": [100.5],
            "volume": [12.0],
        }
    ).with_columns(pl.col("datetime").dt.replace_time_zone(None).cast(pl.Datetime(time_unit="ms")))

    data_file = commit_dir / "part-0000.parquet"
    frame.write_parquet(data_file)
    checksum = repo.canonical_row_checksum(frame)
    dt = frame["datetime"][0]
    ts_ms = int(dt.timestamp() * 1000)

    manifest_payload = {
        "manifest_version": 1,
        "commit_id": commit_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "partition": str(partition_root),
        "window_start_ms": ts_ms,
        "window_end_ms": ts_ms,
        "event_time_watermark_ms": ts_ms,
        "source_checkpoint_start": ts_ms,
        "source_checkpoint_end": ts_ms,
        "row_count": 1,
        "canonical_row_checksum": checksum,
        "data_files": [f"commit={commit_id}/part-0000.parquet"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "producer": "pytest",
        "status": "committed",
    }
    repo.write_materialized_manifest(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        partition_date=partition_date,
        payload=manifest_payload,
    )
    return repo


def test_manifest_schema_roundtrip_loads_committed_rows(tmp_path):
    repo = _write_committed_partition(tmp_path)

    loaded = repo.load_committed_ohlcv_chunked(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
    )

    assert loaded.height == 1
    assert loaded["close"][0] == pytest.approx(100.5)


def test_manifest_missing_required_field_raises(tmp_path):
    repo = _write_committed_partition(tmp_path)
    manifest_path = repo.materialized_manifest_path(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        partition_date="2026-03-01",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("canonical_row_checksum", None)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RawFirstManifestInvalidError):
        repo.load_committed_ohlcv_chunked(
            exchange="binance",
            symbol="BTC/USDT",
            timeframe="1s",
        )
