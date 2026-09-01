from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from lumina_quant.storage.parquet import (
    ParquetMarketDataRepository,
    is_parquet_market_data_store,
    load_data_dict_from_parquet,
)
from lumina_quant.storage.wal import BinaryWAL
from lumina_quant.storage.wal.native_backend import (
    append_ohlcv_frame_native,
    native_wal_append_available,
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


def test_ohlcv_datetime_normalization_accepts_string_temporal_and_epoch_ms() -> None:
    expected = [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 1, 0, 0, 1),
    ]
    base = {
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low": [99.0, 100.0],
        "close": [100.5, 101.5],
        "volume": [1.0, 2.0],
    }
    frames = (
        pl.DataFrame(
            {
                "datetime": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"],
                **base,
            }
        ),
        pl.DataFrame(
            {
                "datetime": [1_767_225_600_000, 1_767_225_601_000],
                **base,
            }
        ),
        pl.DataFrame(
            {
                "datetime": pl.datetime_range(
                    expected[0],
                    expected[1],
                    interval="1s",
                    time_unit="us",
                    eager=True,
                ),
                **base,
            }
        ),
    )

    normalized = [ParquetMarketDataRepository._ensure_ohlcv_frame(frame) for frame in frames]

    assert all(frame.schema["datetime"] == pl.Datetime("ms") for frame in normalized)
    assert all(frame["datetime"].to_list() == expected for frame in normalized)


def test_upsert_1s_writes_wal(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    written = repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())

    assert written == 4

    symbol_root = tmp_path / "market_ohlcv_1s" / "binance" / "BTCUSDT"
    assert (symbol_root / "wal.bin").exists()
    assert not list(symbol_root.glob("*.parquet"))


def test_upsert_1s_prefers_native_append_when_available(monkeypatch, tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    seen: dict[str, object] = {}

    def _fake_native(path, frame, *, fsync_after_write):
        seen["path"] = Path(path)
        seen["height"] = int(frame.height)
        seen["fsync_after_write"] = bool(fsync_after_write)
        return int(frame.height)

    def _unexpected_python_append(self, _rows):
        raise AssertionError("python WAL append fallback should not run")

    monkeypatch.setattr(
        "lumina_quant.storage.parquet.ohlcv_repo.append_ohlcv_frame_native",
        _fake_native,
    )
    monkeypatch.setattr(BinaryWAL, "append", _unexpected_python_append)

    written = repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())

    assert written == 4
    assert seen["height"] == 4
    assert seen["path"] == tmp_path / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "wal.bin"


def test_native_wal_append_matches_python_reader_when_available(tmp_path: Path):
    if not native_wal_append_available():
        pytest.skip("Rust raw-first backend is not built")

    repo = ParquetMarketDataRepository(tmp_path)
    frame = repo._ensure_ohlcv_frame(_sample_1s_frame())
    wal = BinaryWAL(tmp_path / "native-wal.bin", auto_repair=True)

    written = append_ohlcv_frame_native(
        wal.path,
        frame,
        fsync_after_write=True,
    )

    assert written == 4
    records = list(wal.iter_all())
    assert [item.ts_ms for item in records] == [
        1767225600000,
        1767225601000,
        1767225659000,
        1767225660000,
    ]
    assert [item.close for item in records] == [100.5, 101.5, 102.5, 103.5]


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


def test_load_ohlcv_reads_valid_prefix_without_forcing_repair(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())

    wal_path = tmp_path / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "wal.bin"
    original_size = wal_path.stat().st_size
    with wal_path.open("ab") as fh:
        fh.write(b"broken-tail")

    minute = repo.load_ohlcv(exchange="binance", symbol="BTC/USDT", timeframe="1m")

    assert wal_path.stat().st_size == original_size + len(b"broken-tail")
    assert minute.height == 2
    assert minute["open"].to_list() == [100.0, 103.0]
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

    assert result.files_after == 1
    assert result.rows_before >= 2
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


def test_load_data_dict_from_parquet_emits_fetch_progress(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())
    events: list[tuple[str, dict[str, object]]] = []

    loaded = load_data_dict_from_parquet(
        str(tmp_path),
        exchange="binance",
        symbol_list=["BTC/USDT", "ETH/USDT"],
        timeframe="1m",
        start_date="2026-01-01T00:00:00Z",
        end_date="2026-01-01T00:01:00Z",
        chunk_days=1,
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    assert "BTC/USDT" in loaded
    event_names = [name for name, _ in events]
    assert event_names == [
        "resource_bundle_symbol_fetch_started",
        "resource_bundle_symbol_window_loaded",
        "resource_bundle_symbol_fetch_completed",
        "resource_bundle_symbol_fetch_started",
        "resource_bundle_symbol_window_loaded",
        "resource_bundle_symbol_fetch_completed",
    ]
    assert events[0][1]["symbol"] == "BTC/USDT"
    assert events[1][1]["unit_kind"] == "chunk"
    assert events[1][1]["unit_index"] == 1
    assert events[2][1]["was_missing"] is False
    assert events[3][1]["symbol"] == "ETH/USDT"
    assert events[5][1]["was_missing"] is True


def test_is_parquet_market_data_store_detects_existing_partition_layout(tmp_path: Path):
    repo = ParquetMarketDataRepository(tmp_path)
    repo.upsert_1s(exchange="binance", symbol="BTC/USDT", rows=_sample_1s_frame())
    repo.compact_all(exchange="binance", symbol="BTC/USDT", timeframe="1s", remove_sources=True)

    assert is_parquet_market_data_store(str(tmp_path))
    assert is_parquet_market_data_store("anything", backend="parquet")
