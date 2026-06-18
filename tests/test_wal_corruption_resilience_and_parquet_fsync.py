"""Audit-hardening tests: WAL mid-file corruption resilience and parquet fsync."""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from lumina_quant.storage.wal.binary import (
    MAGIC,
    RECORD_LEN,
    VERSION,
    BinaryWAL,
    WALRecord,
    encode_record,
)
from lumina_quant.storage.parquet import ParquetMarketDataRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(ts_ms: int) -> WALRecord:
    return WALRecord(
        ts_ms=ts_ms,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
    )


def _corrupt_bytes(length: int = RECORD_LEN) -> bytes:
    """Return bytes that will fail CRC / magic checks."""
    return b"\xff" * length


def _write_wal_with_mid_corruption(
    path: Path, records_before: list[WALRecord], records_after: list[WALRecord]
) -> None:
    """Write valid records, then one corrupt slot, then more valid records."""
    with path.open("wb") as fh:
        for rec in records_before:
            fh.write(encode_record(rec))
        fh.write(_corrupt_bytes(RECORD_LEN))
        for rec in records_after:
            fh.write(encode_record(rec))


# ---------------------------------------------------------------------------
# WAL resilience tests
# ---------------------------------------------------------------------------


class TestWALMidFileCorruptionResilience:
    """scan_valid_length / iter_range / repair must survive a corrupt mid-file record."""

    def test_scan_valid_length_skips_corrupt_slot_and_finds_later_valid_records(
        self, tmp_path: Path
    ):
        """scan_valid_length returns byte offset past the last valid record (after the corrupt slot).

        Layout: [V, V, C, V, V] — 5 slots * 64 bytes = 320 bytes total.
        The last valid record (slot 4) ends at byte 320, so scan_valid_length == 320.
        The corrupt slot is skipped in-place; valid_end covers the full file extent.
        """
        wal_path = tmp_path / "wal.bin"
        before = [_make_record(1_000_000_000_000), _make_record(1_000_000_001_000)]
        after = [_make_record(1_000_000_002_000), _make_record(1_000_000_003_000)]
        _write_wal_with_mid_corruption(wal_path, before, after)

        wal = BinaryWAL(wal_path, auto_repair=False)
        valid_len = wal.scan_valid_length()

        # 5 slots total; last valid record ends at position 5*RECORD_LEN.
        # The corrupt slot is inside the range, but valid_end advances past all
        # valid slots. Repair will then truncate to valid_end (== file size here),
        # so nothing is removed — which is correct: the file has no unreachable tail.
        assert valid_len == 5 * RECORD_LEN

    def test_iter_range_recovers_records_before_and_after_corrupt_slot(self, tmp_path: Path):
        """iter_range must yield valid records on both sides of a corrupt slot."""
        wal_path = tmp_path / "wal.bin"
        before = [_make_record(1_000_000_000_000), _make_record(1_000_000_001_000)]
        after = [_make_record(1_000_000_002_000), _make_record(1_000_000_003_000)]
        _write_wal_with_mid_corruption(wal_path, before, after)

        wal = BinaryWAL(wal_path, auto_repair=False)
        records = list(wal.iter_all())

        ts_list = [r.ts_ms for r in records]
        assert ts_list == [
            1_000_000_000_000,
            1_000_000_001_000,
            1_000_000_002_000,
            1_000_000_003_000,
        ], f"Got: {ts_list}"

    def test_iter_range_does_not_raise_on_mid_file_corruption(self, tmp_path: Path):
        """iter_range must never raise, even with multiple corrupt slots."""
        wal_path = tmp_path / "wal.bin"
        with wal_path.open("wb") as fh:
            fh.write(encode_record(_make_record(1_000_000_000_000)))
            fh.write(_corrupt_bytes(RECORD_LEN))
            fh.write(encode_record(_make_record(1_000_000_001_000)))
            fh.write(_corrupt_bytes(RECORD_LEN))
            fh.write(encode_record(_make_record(1_000_000_002_000)))

        wal = BinaryWAL(wal_path, auto_repair=False)
        records = list(wal.iter_all())  # must not raise

        ts_list = [r.ts_ms for r in records]
        assert ts_list == [
            1_000_000_000_000,
            1_000_000_001_000,
            1_000_000_002_000,
        ], f"Got: {ts_list}"

    def test_repair_truncates_trailing_garbage_after_last_valid_record(self, tmp_path: Path):
        """repair() truncates bytes that follow the last valid record boundary.

        Layout: [V, V, C_trailing_garbage] — the corrupt bytes are a partial/invalid
        trailing region with no further valid records.  repair() must truncate to
        2 * RECORD_LEN.
        """
        wal_path = tmp_path / "wal.bin"
        with wal_path.open("wb") as fh:
            fh.write(encode_record(_make_record(1_000_000_000_000)))
            fh.write(encode_record(_make_record(1_000_000_001_000)))
            # Partial trailing corruption (not a full RECORD_LEN slot).
            fh.write(b"\xff" * 17)

        total = 2 * RECORD_LEN + 17
        assert wal_path.stat().st_size == total

        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()

        assert removed == 17
        assert wal_path.stat().st_size == 2 * RECORD_LEN

    def test_repair_mid_file_corrupt_slot_followed_by_valid_records_nothing_removed(
        self, tmp_path: Path
    ):
        """When a corrupt slot is followed by valid records, repair() removes nothing.

        Layout: [V, C, V] — valid_end == 3*RECORD_LEN == file_size, so 0 bytes removed
        and all three slots (2 valid + 1 corrupt) remain in the file.  iter_all()
        still yields only the 2 valid records because corrupt slots are skipped.
        """
        wal_path = tmp_path / "wal.bin"
        before = [_make_record(1_000_000_000_000)]
        after = [_make_record(1_000_000_001_000)]
        _write_wal_with_mid_corruption(wal_path, before, after)

        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()

        # Nothing is trimmed because the last valid record ends at file boundary.
        assert removed == 0
        assert wal_path.stat().st_size == 3 * RECORD_LEN
        # iter_all still recovers both valid records.
        records = list(wal.iter_all())
        assert [r.ts_ms for r in records] == [1_000_000_000_000, 1_000_000_001_000]

    def test_repair_via_auto_repair_on_init_does_not_raise(self, tmp_path: Path):
        """BinaryWAL(auto_repair=True) on a mid-file-corrupt file must not raise."""
        wal_path = tmp_path / "wal.bin"
        before = [_make_record(1_000_000_000_000)]
        after = [_make_record(1_000_000_001_000)]
        _write_wal_with_mid_corruption(wal_path, before, after)

        # Should not raise.
        wal = BinaryWAL(wal_path, auto_repair=True)
        records = list(wal.iter_all())
        assert len(records) == 2

    def test_iter_records_from_offset_skips_corrupt_slot(self, tmp_path: Path):
        """iter_records_from_offset must skip corrupt slots and continue."""
        wal_path = tmp_path / "wal.bin"
        with wal_path.open("wb") as fh:
            fh.write(encode_record(_make_record(1_000_000_000_000)))
            fh.write(_corrupt_bytes(RECORD_LEN))
            fh.write(encode_record(_make_record(1_000_000_001_000)))

        wal = BinaryWAL(wal_path, auto_repair=False)
        records = list(wal.iter_records_from_offset(0))
        ts_list = [r.ts_ms for r in records]
        assert ts_list == [1_000_000_000_000, 1_000_000_001_000], f"Got: {ts_list}"

    def test_pure_trailing_garbage_still_truncated(self, tmp_path: Path):
        """Trailing partial/garbage bytes are still removed by repair()."""
        wal_path = tmp_path / "wal.bin"
        rec = _make_record(1_000_000_000_000)
        with wal_path.open("wb") as fh:
            fh.write(encode_record(rec))
            fh.write(b"\xde\xad\xbe\xef")  # partial trailing garbage

        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()
        assert removed == 4
        assert wal_path.stat().st_size == RECORD_LEN


# ---------------------------------------------------------------------------
# Parquet fsync tests
# ---------------------------------------------------------------------------


class TestParquetRawWriteFsync:
    """Raw parquet writes must call os.fsync on the file descriptor."""

    def _raw_aggtrades_frame(self, n: int = 3) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "agg_trade_id": list(range(n)),
                "timestamp_ms": [1_700_000_000_000 + i * 1000 for i in range(n)],
                "price": [100.0 + i for i in range(n)],
                "quantity": [1.0] * n,
                "is_buyer_maker": [False] * n,
            }
        )

    def test_append_raw_aggtrades_fsyncs_new_part_file(self, tmp_path: Path):
        """append_raw_aggtrades (append-only path) must fsync the written parquet file."""
        repo = ParquetMarketDataRepository(tmp_path)
        frame = self._raw_aggtrades_frame()

        fsync_calls: list[int] = []
        real_fsync = os.fsync

        def _spy_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            real_fsync(fd)

        with patch("lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=_spy_fsync):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=frame)

        # At least one fsync call must have happened (file-level fsync).
        assert len(fsync_calls) >= 1, "Expected at least one os.fsync call on parquet write"

    def test_append_raw_aggtrades_fsyncs_on_merge_path(self, tmp_path: Path):
        """append_raw_aggtrades (merge/rewrite path) must fsync the tmp parquet file before rename."""
        repo = ParquetMarketDataRepository(tmp_path)

        # Write first batch so a part-0000 exists.
        frame1 = self._raw_aggtrades_frame(3)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=frame1)

        # Write an overlapping batch to trigger the merge path.
        frame2 = pl.DataFrame(
            {
                "agg_trade_id": [0, 1],
                "timestamp_ms": [1_700_000_000_000, 1_700_000_001_000],
                "price": [200.0, 201.0],
                "quantity": [2.0, 2.0],
                "is_buyer_maker": [True, True],
            }
        )

        fsync_calls: list[int] = []
        real_fsync = os.fsync

        def _spy_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            real_fsync(fd)

        with patch("lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=_spy_fsync):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=frame2)

        assert len(fsync_calls) >= 1, "Expected at least one os.fsync call on merged parquet write"
