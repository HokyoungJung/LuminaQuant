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
    decode_record,
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

    def test_repair_mid_file_corrupt_slot_followed_by_valid_records_slot_removed(
        self, tmp_path: Path
    ):
        """repair() physically removes an interior corrupt slot, leaving only valid records.

        Layout: [V, C, V] — repair() rewrites to [V, V], removing the corrupt slot.
        The file shrinks by RECORD_LEN and iter_all() yields both valid records without
        rescanning dead slots.
        """
        wal_path = tmp_path / "wal.bin"
        before = [_make_record(1_000_000_000_000)]
        after = [_make_record(1_000_000_001_000)]
        _write_wal_with_mid_corruption(wal_path, before, after)

        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()

        # The corrupt interior slot is physically removed.
        assert removed == RECORD_LEN
        assert wal_path.stat().st_size == 2 * RECORD_LEN
        # iter_all recovers both valid records from the compacted file.
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

    def test_compact_raw_partition_fsyncs_before_replace(self, tmp_path: Path):
        """_compact_raw_partition must fsync the tmp parquet before atomic-replace.

        We write enough parts to exceed the default compact threshold, then spy on
        os.fsync to confirm it is called (via _fsync_file) before the rename/replace.
        A crash between write_parquet and fsync would leave a corrupt compacted file;
        this test guards against regressing that durability gap.
        """
        repo = ParquetMarketDataRepository(tmp_path)

        # Write 6 separate batches with non-overlapping agg_trade_ids so each
        # lands as a distinct part file, ensuring the compact threshold is crossed.
        base_ts = 1_700_000_000_000
        for i in range(6):
            frame = pl.DataFrame(
                {
                    "agg_trade_id": [i * 100],
                    "timestamp_ms": [base_ts + i * 1_000],
                    "price": [100.0 + i],
                    "quantity": [1.0],
                    "is_buyer_maker": [False],
                }
            )
            # Bypass compaction enforcement on each intermediate write by calling
            # the low-level writer directly to build up multiple parts.
            part_root = (
                tmp_path
                / "market_raw_aggtrades"
                / "binance"
                / "BTCUSDT"
                / f"date={base_ts // 86_400_000}"
            )
            part_root.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(part_root / f"part-{i:04d}.parquet")

        fsync_calls: list[int] = []
        real_fsync = os.fsync

        def _spy_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            real_fsync(fd)

        part_root_path = (
            tmp_path
            / "market_raw_aggtrades"
            / "binance"
            / "BTCUSDT"
            / f"date={base_ts // 86_400_000}"
        )
        with patch("lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=_spy_fsync):
            repo._compact_raw_partition(
                exchange="binance",
                symbol="BTC/USDT",
                partition_root=part_root_path,
            )

        # os.fsync must have been called (covers the _fsync_file call on the tmp parquet).
        assert fsync_calls, "os.fsync was never called during _compact_raw_partition"


# ---------------------------------------------------------------------------
# WAL interior-corruption compaction tests
# ---------------------------------------------------------------------------


class TestWALRepairInteriorCompaction:
    """repair() must physically remove interior corrupt slots."""

    def test_repair_interior_corruption_produces_clean_file(self, tmp_path: Path):
        """Layout [V1][CORRUPT][V2]: repair() rewrites to [V1][V2], 0 dead slots left."""
        wal_path = tmp_path / "wal.bin"
        r1 = _make_record(1_000_000_000_000)
        r2 = _make_record(1_000_000_002_000)
        _write_wal_with_mid_corruption(wal_path, [r1], [r2])

        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()

        assert removed == RECORD_LEN, f"Expected {RECORD_LEN} bytes removed, got {removed}"
        assert wal_path.stat().st_size == 2 * RECORD_LEN

        # File must decode cleanly — no corrupt slots remain.
        with wal_path.open("rb") as fh:
            slots = [fh.read(RECORD_LEN) for _ in range(2)]
        assert all(decode_record(s) is not None for s in slots), (
            "Compact file still has corrupt slots"
        )

    def test_repair_interior_corruption_no_rescan_of_dead_slots(self, tmp_path: Path):
        """After repair(), a fresh iter_all() touches zero dead slots.

        We verify by reading the compacted file slot-by-slot: every decoded record
        must be valid (no None returns from decode_record), confirming dead slots
        have been physically removed and cannot be re-encountered on future reads.
        """
        wal_path = tmp_path / "wal.bin"
        before = [_make_record(1_000_000_000_000), _make_record(1_000_000_001_000)]
        after = [_make_record(1_000_000_003_000), _make_record(1_000_000_004_000)]
        # Layout: [V, V, C, V, V] — one interior corrupt slot
        _write_wal_with_mid_corruption(wal_path, before, after)

        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()

        assert removed == RECORD_LEN
        assert wal_path.stat().st_size == 4 * RECORD_LEN

        # Raw slot scan: every slot must be decodable — no dead slots in the file.
        null_slots = 0
        with wal_path.open("rb") as fh:
            while True:
                chunk = fh.read(RECORD_LEN)
                if not chunk or len(chunk) != RECORD_LEN:
                    break
                if decode_record(chunk) is None:
                    null_slots += 1
        assert null_slots == 0, f"Found {null_slots} dead slot(s) after repair()"

        # iter_all must yield all 4 valid records.
        records = list(wal.iter_all())
        assert [r.ts_ms for r in records] == [
            1_000_000_000_000,
            1_000_000_001_000,
            1_000_000_003_000,
            1_000_000_004_000,
        ]

    def test_repair_fsyncs_tmp_before_replace(self, tmp_path: Path):
        """repair() must fsync the tmp WAL file before atomic-replace (crash durability)."""
        wal_path = tmp_path / "wal.bin"
        r1 = _make_record(1_000_000_000_000)
        r2 = _make_record(1_000_000_002_000)
        _write_wal_with_mid_corruption(wal_path, [r1], [r2])

        fsync_fds: list[int] = []
        real_fsync = os.fsync

        def _spy(fd: int) -> None:
            fsync_fds.append(fd)
            real_fsync(fd)

        wal = BinaryWAL(wal_path, auto_repair=False)
        with patch("lumina_quant.storage.wal.binary.os.fsync", side_effect=_spy):
            wal.repair()

        assert fsync_fds, "os.fsync was never called during repair()"

    def test_repair_preserves_all_valid_records_no_data_loss(self, tmp_path: Path):
        """repair() must not drop any valid record — no-data-loss guarantee."""
        wal_path = tmp_path / "wal.bin"
        # Multiple corrupt slots interspersed: [V, C, V, C, V]
        with wal_path.open("wb") as fh:
            fh.write(encode_record(_make_record(1_000_000_000_000)))
            fh.write(_corrupt_bytes(RECORD_LEN))
            fh.write(encode_record(_make_record(1_000_000_001_000)))
            fh.write(_corrupt_bytes(RECORD_LEN))
            fh.write(encode_record(_make_record(1_000_000_002_000)))

        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()

        assert removed == 2 * RECORD_LEN
        assert wal_path.stat().st_size == 3 * RECORD_LEN

        records = list(wal.iter_all())
        assert [r.ts_ms for r in records] == [
            1_000_000_000_000,
            1_000_000_001_000,
            1_000_000_002_000,
        ]

    def test_repair_clean_file_returns_zero_no_rewrite(self, tmp_path: Path):
        """repair() on a file with no corruption returns 0 and does not rewrite."""
        wal_path = tmp_path / "wal.bin"
        with wal_path.open("wb") as fh:
            fh.write(encode_record(_make_record(1_000_000_000_000)))
            fh.write(encode_record(_make_record(1_000_000_001_000)))

        mtime_before = wal_path.stat().st_mtime_ns
        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()

        assert removed == 0
        # File must not have been rewritten (mtime unchanged).
        assert wal_path.stat().st_mtime_ns == mtime_before
