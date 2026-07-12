from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from types import SimpleNamespace

import pytest

from lumina_quant import data_sync
from lumina_quant.data_collector import collect_binance_aggtrades_raw
from lumina_quant.eval.exact_window_runtime import HeavyRunLock
from lumina_quant.storage.parquet import ParquetMarketDataRepository
from lumina_quant.storage.parquet.ohlcv_repo import RawPartitionBusyError


class _ExchangeStub:
    def close(self):
        return None


def _zip_csv(payload: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload.csv", payload)
    return buffer.getvalue()


def test_archive_day_bounds_include_the_final_millisecond() -> None:
    start_ms, end_ms = data_sync._day_bounds_ms(date(2025, 1, 1))

    assert start_ms == 1_735_689_600_000
    assert end_ms == 1_735_775_999_999
    assert end_ms - start_ms + 1 == 86_400_000


def test_collect_binance_aggtrades_raw_checkpoint_resume(tmp_path, monkeypatch):
    calls: list[int] = []
    state = {"last": 0}

    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**kwargs):
        start_ms = int(kwargs["start_ms"])
        calls.append(start_ms)
        if start_ms == 0:
            state["last"] = 1_700_000_001_000
            return SimpleNamespace(
                fetched_rows=2,
                upserted_rows=2,
                first_timestamp_ms=1_700_000_000_000,
                last_timestamp_ms=1_700_000_001_000,
                checkpoint_timestamp_ms=1_700_000_001_000,
                checkpoint_trade_id=2,
            )
        state["last"] = 1_700_000_002_000
        return SimpleNamespace(
            fetched_rows=1,
            upserted_rows=1,
            first_timestamp_ms=1_700_000_002_000,
            last_timestamp_ms=1_700_000_002_000,
            checkpoint_timestamp_ms=1_700_000_002_000,
            checkpoint_trade_id=3,
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    first = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=0,
        until_ms=1_700_000_010_000,
        limit=1000,
        max_batches=10,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.write_raw_checkpoint(
        exchange="binance",
        symbol="BTC/USDT",
        payload={"last_timestamp_ms": state["last"], "last_trade_id": 2},
    )
    second = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=None,
        until_ms=1_700_000_010_000,
        limit=1000,
        max_batches=10,
    )

    assert calls == [0, 1_700_000_001_001]
    assert int(first["fetched_rows"]) == 2
    assert int(second["fetched_rows"]) == 1
    assert int(second["last_trade_id"]) == 3


def test_sync_symbol_aggtrades_raw_streams_archive_chunks(tmp_path, monkeypatch):
    start_ms = 1_735_689_600_000  # 2025-01-01T00:00:00Z
    row_count = 1_001
    end_ms = start_ms + ((row_count - 1) * 1_000)
    csv_payload = "\n".join(
        f"{1000 + idx},{100.0 + idx},0.5,{2000 + idx},{2000 + idx},{start_ms + idx * 1000},{str(idx % 2 == 0).lower()}"
        for idx in range(row_count)
    )
    archive = _zip_csv(csv_payload)

    monkeypatch.setattr(data_sync, "_download_zip_bytes", lambda *_args, **_kwargs: archive)
    monkeypatch.setattr(data_sync, "_now_ms", lambda: end_ms + (3 * 86_400_000))
    monkeypatch.setenv("LQ_RAW_ARCHIVE_CHUNK_ROWS", "1000")
    monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "128")
    monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "false")

    stats = data_sync.sync_symbol_aggtrades_raw(
        exchange=_ExchangeStub(),
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        start_ms=start_ms,
        end_ms=end_ms,
        max_batches=10,
        resume_from_checkpoint=False,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert stats.fetched_rows == row_count
    assert stats.upserted_rows == row_count
    assert raw.height == row_count
    assert raw["agg_trade_id"].head(3).to_list() == [1000, 1001, 1002]
    part_root = tmp_path / "market_data_raw_aggtrades/binance/BTCUSDT/date=2025-01-01"
    assert len(list(part_root.glob("part-*.parquet"))) == 2


def test_collect_binance_aggtrades_raw_bootstrap_lookback_used_without_checkpoint(
    tmp_path, monkeypatch
):
    observed_since: list[int] = []

    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**kwargs):
        observed_since.append(int(kwargs["start_ms"]))
        return SimpleNamespace(
            fetched_rows=0,
            upserted_rows=0,
            first_timestamp_ms=None,
            last_timestamp_ms=None,
            checkpoint_timestamp_ms=None,
            checkpoint_trade_id=None,
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    until_ms = 1_700_000_010_000
    result = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=None,
        until_ms=until_ms,
        bootstrap_lookback_hours=2,
        limit=1000,
        max_batches=10,
    )

    expected_since = until_ms - (2 * 60 * 60 * 1000)
    assert observed_since == [expected_since]
    assert int(result["start_cursor_ms"]) == expected_since


def test_collect_binance_aggtrades_raw_recovers_from_corrupt_partition(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "lumina_quant.data_collector.create_binance_futures_client", lambda **_: _ExchangeStub()
    )

    def _sync(**kwargs):
        repo = ParquetMarketDataRepository(str(tmp_path))
        repo.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    "agg_trade_id": 10,
                    "timestamp_ms": 1_700_000_100_000,
                    "price": 100.0,
                    "quantity": 0.1,
                    "is_buyer_maker": False,
                }
            ],
        )
        return SimpleNamespace(
            fetched_rows=1,
            upserted_rows=1,
            first_timestamp_ms=1_700_000_100_000,
            last_timestamp_ms=1_700_000_100_000,
            checkpoint_timestamp_ms=1_700_000_100_000,
            checkpoint_trade_id=10,
        )

    monkeypatch.setattr("lumina_quant.data_collector.sync_symbol_aggtrades_raw", _sync)

    corrupt_dir = tmp_path / "market_data_raw_aggtrades" / "binance" / "BTCUSDT" / "date=2023-11-14"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "part-0000.parquet").write_bytes(b"not-a-parquet-file")

    result = collect_binance_aggtrades_raw(
        db_path=str(tmp_path),
        exchange_id="binance",
        symbol="BTC/USDT",
        since_ms=1_700_000_100_000,
        until_ms=1_700_000_100_000,
        limit=1000,
        max_batches=10,
    )

    repo = ParquetMarketDataRepository(str(tmp_path))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert int(result["upserted_rows"]) == 1
    assert raw.height == 1
    assert list(corrupt_dir.glob("part-0000.corrupt-*.parquet"))


def test_append_raw_aggtrades_recovers_from_corrupt_partition(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    part_path = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    )
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.write_bytes(b"not-a-real-parquet-file")

    written = repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )

    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")
    corrupt_files = sorted(part_path.parent.glob("part-0000.corrupt-*.parquet"))

    assert written == 1
    assert raw.height == 1
    assert corrupt_files


def test_append_raw_aggtrades_uses_incremental_parts_for_append_only_batches(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))

    first_written = repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )
    second_written = repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    part_files = sorted(path.name for path in part_dir.glob("part-*.parquet"))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert first_written == 1
    assert second_written == 1
    assert part_files == ["part-0000.parquet", "part-0001.parquet"]
    assert raw.height == 2


def test_append_raw_aggtrades_compacts_back_to_single_part_when_batches_overlap(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))

    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    written = repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
            {
                "agg_trade_id": 3,
                "timestamp_ms": 1_735_689_602_000,
                "price": 102.0,
                "quantity": 0.3,
                "is_buyer_maker": False,
            },
        ],
    )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    part_files = sorted(path.name for path in part_dir.glob("part-*.parquet"))
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert written == 2
    assert part_files == ["part-0000.parquet"]
    assert raw.height == 3


def test_load_raw_aggtrades_skips_corrupt_incremental_parts(tmp_path):
    repo = ParquetMarketDataRepository(str(tmp_path))
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    (part_dir / "part-0002.parquet").write_bytes(b"not-a-real-parquet-file")

    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")

    assert raw.height == 2
    assert raw["agg_trade_id"].to_list() == [1, 2]


def test_append_raw_aggtrades_auto_compacts_when_part_threshold_exceeded(tmp_path, monkeypatch):
    repo = ParquetMarketDataRepository(str(tmp_path))
    monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "2")
    monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "true")

    for offset in range(3):
        repo.append_raw_aggtrades(
            exchange="binance",
            symbol="BTC/USDT",
            rows=[
                {
                    "agg_trade_id": offset + 1,
                    "timestamp_ms": 1_735_689_600_000 + offset,
                    "price": 100.0 + offset,
                    "quantity": 0.1,
                    "is_buyer_maker": bool(offset % 2),
                }
            ],
        )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    raw = repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")
    meta = json.loads(repo._raw_meta_path(exchange="binance", symbol="BTC/USDT").read_text())

    assert sorted(path.name for path in part_dir.glob("part-*.parquet")) == ["part-0000.parquet"]
    assert raw.height == 3
    assert meta["raw_compaction_required"] is False
    assert meta["last_raw_compaction_partition"] == "date=2025-01-01"


def test_append_raw_aggtrades_marks_meta_when_threshold_exceeded_but_auto_compact_disabled(
    tmp_path, monkeypatch
):
    repo = ParquetMarketDataRepository(str(tmp_path))
    monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "1")
    monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "false")

    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            }
        ],
    )
    repo.append_raw_aggtrades(
        exchange="binance",
        symbol="BTC/USDT",
        rows=[
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_601_000,
                "price": 101.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            }
        ],
    )

    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    meta = json.loads(repo._raw_meta_path(exchange="binance", symbol="BTC/USDT").read_text())

    assert sorted(path.name for path in part_dir.glob("part-*.parquet")) == [
        "part-0000.parquet",
        "part-0001.parquet",
    ]
    assert meta["raw_compaction_required"] is True
    assert meta["last_raw_part_count"] == 2


def test_append_raw_aggtrades_raises_when_partition_lock_is_held(tmp_path, monkeypatch):
    repo = ParquetMarketDataRepository(str(tmp_path))
    part_dir = repo.raw_partition_path(
        exchange="binance",
        symbol="BTC/USDT",
        partition_date="2025-01-01",
    ).parent
    part_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LQ_RAW_PARTITION_LOCK_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("LQ_RAW_PARTITION_LOCK_POLL_SECONDS", "0.01")
    lock = HeavyRunLock.acquire(
        lock_path=part_dir / ".raw-partition.lock",
        label="raw_partition",
        metadata={"run_id": "test-lock-holder"},
    )

    try:
        with pytest.raises(RawPartitionBusyError):
            repo.append_raw_aggtrades(
                exchange="binance",
                symbol="BTC/USDT",
                rows=[
                    {
                        "agg_trade_id": 1,
                        "timestamp_ms": 1_735_689_600_000,
                        "price": 100.0,
                        "quantity": 0.1,
                        "is_buyer_maker": False,
                    }
                ],
            )
    finally:
        lock.release()
