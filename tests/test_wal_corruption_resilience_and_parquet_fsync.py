"""Audit-hardening tests: WAL mid-file corruption resilience and parquet fsync."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
import os
import json
from hashlib import sha256
import stat
import struct
import zlib
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

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
from lumina_quant.storage.parquet.ohlcv_repo import (
    _RAW_CONTROL_MAX_BYTES,
    _RAW_WAL_MAX_RECORD_BYTES,
)


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
        """repair() removes an interior corrupt slot without losing later records."""
        wal_path = tmp_path / "wal.bin"
        before = [_make_record(1_000_000_000_000)]
        after = [_make_record(1_000_000_001_000)]
        _write_wal_with_mid_corruption(wal_path, before, after)

        wal = BinaryWAL(wal_path, auto_repair=False)
        removed = wal.repair()

        assert removed == RECORD_LEN
        assert wal_path.stat().st_size == 2 * RECORD_LEN
        assert [record.ts_ms for record in wal.iter_all()] == [
            1_000_000_000_000,
            1_000_000_001_000,
        ]

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


@dataclass(frozen=True)
class _RawProtocolEvent:
    """One attempted filesystem protocol operation, with a stable occurrence id."""

    occurrence: int
    kind: str
    path: Path
    other: Path | None = None
    offset: int | None = None
    requested: int | None = None
    completed: int | None = None
    error: type[BaseException] | None = None


class _RawProtocolLog:
    """Occurrence-aware evidence for the raw durability protocol."""

    def __init__(self) -> None:
        self.events: list[_RawProtocolEvent] = []
        self._fsync = os.fsync
        self._mkdir = os.mkdir
        self._replace = os.replace
        self._unlink = os.unlink
        self._write = os.write
        self._pwrite = os.pwrite
        self._open = os.open
        self._ftruncate = os.ftruncate

    @staticmethod
    def _fd_path(fd: int | None) -> Path:
        assert fd is not None
        return Path(os.readlink(f"/proc/self/fd/{fd}"))

    def _record(
        self,
        kind: str,
        path: Path,
        other: Path | None = None,
        *,
        offset: int | None = None,
        requested: int | None = None,
        completed: int | None = None,
        error: type[BaseException] | None = None,
    ) -> _RawProtocolEvent:
        event = _RawProtocolEvent(
            len(self.events), kind, path, other, offset, requested, completed, error
        )
        self.events.append(event)
        return event

    def fsync(self, fd: int) -> None:
        kind = "file-fsync" if stat.S_ISREG(os.fstat(fd).st_mode) else "parent-fsync"
        path = self._fd_path(fd)
        try:
            self._fsync(fd)
        except BaseException as exc:
            self._record(kind, path, error=type(exc))
            raise
        self._record(kind, path)

    def open(
        self, name: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        fd = self._open(name, flags, mode, dir_fd=dir_fd)
        if flags & (os.O_CREAT | os.O_EXCL | os.O_TRUNC):
            parent = self._fd_path(dir_fd) if dir_fd is not None else Path.cwd()
            self._record("open", parent / Path(name).name)
        return fd

    def write(self, fd: int, data: bytes) -> int:
        path = self._fd_path(fd)
        offset = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            completed = self._write(fd, data)
        except BaseException as exc:
            self._record("write", path, offset=offset, requested=len(data), error=type(exc))
            raise
        self._record("write", path, offset=offset, requested=len(data), completed=completed)
        return completed

    def pwrite(self, fd: int, data: bytes, offset: int) -> int:
        path = self._fd_path(fd)
        try:
            completed = self._pwrite(fd, data, offset)
        except BaseException as exc:
            self._record("pwrite", path, offset=offset, requested=len(data), error=type(exc))
            raise
        self._record("pwrite", path, offset=offset, requested=len(data), completed=completed)
        return completed

    def ftruncate(self, fd: int, length: int) -> None:
        path = self._fd_path(fd)
        try:
            self._ftruncate(fd, length)
        except BaseException as exc:
            self._record("truncate", path, requested=length, error=type(exc))
            raise
        self._record("truncate", path, requested=length)

    def mkdir(self, name: str | Path, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        self._mkdir(name, mode, dir_fd=dir_fd)
        path = Path(name)
        parent = (
            self._fd_path(dir_fd)
            if dir_fd is not None
            else (path.parent if path.is_absolute() else Path.cwd())
        )
        self._record("mkdir", parent / path.name)

    def replace(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        source_parent = self._fd_path(src_dir_fd) if src_dir_fd is not None else Path.cwd()
        destination_parent = self._fd_path(dst_dir_fd) if dst_dir_fd is not None else Path.cwd()
        self._replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        self._record("rename", source_parent / source, destination_parent / destination)

    def unlink(self, name: str | Path, *, dir_fd: int | None = None) -> None:
        self._unlink(name, dir_fd=dir_fd)
        parent = self._fd_path(dir_fd) if dir_fd is not None else Path.cwd()
        self._record("unlink", parent / name)

    def one(
        self, kind: str, path: Path, other: Path | None = None, *, occurrence: int = 0
    ) -> _RawProtocolEvent:
        matches = [
            event
            for event in self.events
            if event.kind == kind and event.path == path and event.other == other
        ]
        assert len(matches) > occurrence, self.events
        return matches[occurrence]

    def after(
        self, prior: _RawProtocolEvent, kind: str, path: Path, other: Path | None = None
    ) -> _RawProtocolEvent:
        matches = [
            event
            for event in self.events
            if event.occurrence > prior.occurrence
            and event.kind == kind
            and event.path == path
            and event.other == other
        ]
        assert matches, self.events
        return matches[0]

    def assert_chain(self, *events: _RawProtocolEvent) -> None:
        assert len({event.occurrence for event in events}) == len(events), events
        assert [event.occurrence for event in events] == sorted(
            event.occurrence for event in events
        ), events


def _raw_row(identity: int, timestamp_ms: int = 1_700_000_000_000) -> dict:
    return {
        "agg_trade_id": identity,
        "timestamp_ms": timestamp_ms,
        "price": 100.0,
        "quantity": 1.0,
        "is_buyer_maker": False,
    }


def _checkpoint_payload(row: dict) -> dict:
    import hashlib

    return {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "last_timestamp_ms": row["timestamp_ms"],
        "last_trade_id": row["agg_trade_id"],
        "observed_until_ms": row["timestamp_ms"],
        "updated_at_utc": "2025-01-01T00:00:00+00:00",
        "batch_rows": 1,
        "last_row": row,
        "last_row_sha256": hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _raw_snapshot(root: Path) -> dict[str, tuple]:
    """Non-following snapshot safe for links and special files."""
    if not root.exists():
        return {}
    snapshot: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        key = path.relative_to(root).as_posix()
        metadata = (
            info.st_dev,
            info.st_ino,
            stat.S_IFMT(info.st_mode),
            stat.S_IMODE(info.st_mode),
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if path.is_symlink():
            snapshot[key] = ("symlink", metadata, os.readlink(path))
        elif stat.S_ISREG(info.st_mode):
            import hashlib

            snapshot[key] = ("file", metadata, hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            snapshot[key] = ("other", metadata)
    return snapshot


def _forbid_mutations(monkeypatch):
    calls: list[str] = []
    for name in ("mkdir", "replace", "unlink"):
        original = getattr(os, name)

        def reject(*args, _name=name, **kwargs):
            calls.append(_name)
            raise AssertionError(f"invalid state attempted os.{_name}")

        monkeypatch.setattr(os, name, reject)
        assert original
    return calls


class TestRawOperationOrderedDurability:
    def test_append_orders_file_and_parent_durability_by_real_dir_fd_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        log = _RawProtocolLog()
        import polars as pl

        raw_write_parquet = repo._raw_write_parquet
        native_write_parquet = pl.DataFrame.write_parquet
        native_stage: Path | None = None

        def trace_native_write_parquet(frame, target, *args, **kwargs):
            result = native_write_parquet(frame, target, *args, **kwargs)
            assert native_stage is not None
            log._record("write-complete", native_stage, completed=native_stage.stat().st_size)
            return result

        def trace_raw_write_parquet(parent_fd: int, name: str, *, path: Path, frame) -> None:
            nonlocal native_stage
            native_stage = path
            raw_write_parquet(parent_fd, name, path=path, frame=frame)

        monkeypatch.setattr(pl.DataFrame, "write_parquet", trace_native_write_parquet)
        monkeypatch.setattr(repo, "_raw_write_parquet", trace_raw_write_parquet)
        with (
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=log.fsync),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.mkdir", side_effect=log.mkdir),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.replace", side_effect=log.replace),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.unlink", side_effect=log.unlink),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.write", side_effect=log.write),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.pwrite", side_effect=log.pwrite),
        ):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[_raw_row(1)])

        partition = repo._raw_partition_path(
            exchange="binance", symbol="BTC/USDT", partition_date="2023-11-14"
        )
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        descriptor = partition / ".raw-transaction.json"
        output = partition / "part-0000.parquet"
        descriptor_rename = next(
            event for event in log.events if event.kind == "rename" and event.other == descriptor
        )
        output_rename = next(
            event for event in log.events if event.kind == "rename" and event.other == output
        )
        descriptor_temp = descriptor_rename.path
        stage = output_rename.path
        descriptor_write = max(
            (
                event
                for event in log.events
                if event.kind in {"write", "pwrite"}
                and event.path == descriptor_temp
                and event.completed
                and event.occurrence < log.one("file-fsync", descriptor_temp).occurrence
            ),
            key=lambda event: event.occurrence,
        )
        descriptor_fsync = log.after(descriptor_write, "file-fsync", descriptor_temp)
        descriptor_parent = log.after(descriptor_rename, "parent-fsync", partition)
        output_writes = [
            event for event in log.events if event.kind == "write-complete" and event.path == stage
        ]
        assert len(output_writes) == 1
        output_write = output_writes[0]
        output_fsync = log.after(output_write, "file-fsync", stage)
        output_parent = log.after(output_rename, "parent-fsync", partition)
        inventory_renames = [
            event
            for event in log.events
            if event.kind == "rename"
            and event.other == root / ".raw-inventory.json"
            and event.occurrence > output_parent.occurrence
        ]
        assert len(inventory_renames) == 1
        inventory_rename = inventory_renames[-1]
        inventory_write = max(
            (
                event
                for event in log.events
                if event.kind in {"write", "pwrite"}
                and event.path == inventory_rename.path
                and event.completed
                and event.occurrence < inventory_rename.occurrence
            ),
            key=lambda event: event.occurrence,
        )
        inventory_fsync = max(
            (
                event
                for event in log.events
                if event.kind == "file-fsync"
                and event.path == inventory_rename.path
                and inventory_write.occurrence < event.occurrence < inventory_rename.occurrence
            ),
            key=lambda event: event.occurrence,
        )
        inventory_parent = log.after(inventory_rename, "parent-fsync", root)
        descriptor_unlink = log.one("unlink", descriptor)
        cleanup_parent = log.after(descriptor_unlink, "parent-fsync", partition)
        log.assert_chain(
            output_write,
            output_fsync,
            descriptor_write,
            descriptor_fsync,
            descriptor_rename,
            descriptor_parent,
            output_rename,
            output_parent,
            inventory_write,
            inventory_fsync,
            inventory_rename,
            inventory_parent,
            descriptor_unlink,
            cleanup_parent,
        )
        for event in (event for event in log.events if event.kind == "mkdir"):
            assert event.occurrence < log.after(event, "parent-fsync", event.path.parent).occurrence

    def test_native_parquet_writer_failure_has_no_completion_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import polars as pl

        repo = ParquetMarketDataRepository(tmp_path)
        log = _RawProtocolLog()

        def fail_native_write_parquet(*args, **kwargs):
            raise RuntimeError("native parquet failure")

        monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_native_write_parquet)
        with (
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=log.fsync),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.replace", side_effect=log.replace),
            pytest.raises(RuntimeError, match="native parquet failure"),
        ):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[_raw_row(1)])

        assert not [event for event in log.events if event.kind == "write-complete"]
        assert not [
            event
            for event in log.events
            if event.kind == "rename"
            and event.other is not None
            and event.other.name.startswith("part-")
        ]

    def test_append_creates_absent_nested_root_with_parent_durability(self, tmp_path: Path):
        repository_root = tmp_path / "absent" / "nested" / "repository"
        repo = ParquetMarketDataRepository(repository_root)
        log = _RawProtocolLog()

        with (
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=log.fsync),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.mkdir", side_effect=log.mkdir),
        ):
            assert (
                repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[_raw_row(1)])
                == 1
            )

        expected_directories = [
            tmp_path / "absent",
            tmp_path / "absent" / "nested",
            repository_root,
            repository_root / "market_data_raw_aggtrades",
            repository_root / "market_data_raw_aggtrades" / "binance",
            repository_root / "market_data_raw_aggtrades" / "binance" / "BTCUSDT",
            repository_root
            / "market_data_raw_aggtrades"
            / "binance"
            / "BTCUSDT"
            / "date=2023-11-14",
        ]
        mkdir_events = [event for event in log.events if event.kind == "mkdir"]
        assert [event.path for event in mkdir_events] == expected_directories
        for event in mkdir_events:
            immediate_barrier = log.after(event, "parent-fsync", event.path.parent)
            assert immediate_barrier.occurrence == event.occurrence + 1
        assert repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT").height == 1

    def test_checkpoint_and_wal_publish_file_before_namespace_durability(self, tmp_path: Path):
        repo = ParquetMarketDataRepository(tmp_path)
        row = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        log = _RawProtocolLog()
        with (
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=log.fsync),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.write", side_effect=log.write),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.pwrite", side_effect=log.pwrite),
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.replace", side_effect=log.replace),
        ):
            repo.write_raw_checkpoint(
                exchange="binance", symbol="BTC/USDT", payload=_checkpoint_payload(row)
            )
            repo.append_raw_wal_record(
                exchange="binance", symbol="BTC/USDT", payload={"event": "x"}
            )

        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        checkpoint_rename = next(
            event
            for event in log.events
            if event.kind == "rename" and event.other == root / "checkpoint.json"
        )
        checkpoint_write = max(
            (
                event
                for event in log.events
                if event.kind in {"write", "pwrite"}
                and event.path == checkpoint_rename.path
                and event.completed
                and event.occurrence < log.one("file-fsync", checkpoint_rename.path).occurrence
            ),
            key=lambda event: event.occurrence,
        )
        checkpoint_fsync = log.after(checkpoint_write, "file-fsync", checkpoint_rename.path)
        checkpoint_parent = log.after(checkpoint_rename, "parent-fsync", root)
        log.assert_chain(checkpoint_write, checkpoint_fsync, checkpoint_rename, checkpoint_parent)
        wal = root / "wal.bin"
        wal_write = max(
            (
                event
                for event in log.events
                if event.kind in {"write", "pwrite"} and event.path == wal and event.completed
            ),
            key=lambda event: event.occurrence,
        )
        wal_fsync = log.after(wal_write, "file-fsync", wal)
        wal_parent = log.after(wal_fsync, "parent-fsync", root)
        tail_rename = next(
            event
            for event in log.events
            if event.kind == "rename" and event.other == root / ".raw-wal-tail.json"
        )
        tail_write = max(
            (
                event
                for event in log.events
                if event.kind in {"write", "pwrite"}
                and event.path == tail_rename.path
                and event.completed
                and event.occurrence < log.one("file-fsync", tail_rename.path).occurrence
            ),
            key=lambda event: event.occurrence,
        )
        tail_fsync = log.after(tail_write, "file-fsync", tail_rename.path)
        tail_parent = log.after(tail_rename, "parent-fsync", root)
        log.assert_chain(
            wal_write, wal_fsync, wal_parent, tail_write, tail_fsync, tail_rename, tail_parent
        )
        assert (
            len({checkpoint_parent.occurrence, wal_parent.occurrence, tail_parent.occurrence}) == 3
        )

    @pytest.mark.parametrize(
        ("fault", "expected"),
        [
            ("short", b"control-bytes"),
            ("eintr", b"control-bytes"),
            ("zero", None),
        ],
    )
    def test_control_write_all_handles_short_and_interrupted_writes_without_truncation(
        self, tmp_path: Path, monkeypatch, fault: str, expected: bytes | None
    ):
        path = tmp_path / "control"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        real_write = os.write
        calls = 0

        def controlled_write(target_fd: int, data):
            nonlocal calls
            calls += 1
            if calls == 1 and fault == "eintr":
                raise InterruptedError
            if calls == 1 and fault == "short":
                return real_write(target_fd, bytes(data[:3]))
            if fault == "zero":
                return 0
            return real_write(target_fd, data)

        try:
            monkeypatch.setattr(
                "lumina_quant.storage.parquet.ohlcv_repo.os.write", controlled_write
            )
            if expected is None:
                with pytest.raises(OSError, match="no progress"):
                    ParquetMarketDataRepository._raw_write_all(fd, b"control-bytes")
            else:
                ParquetMarketDataRepository._raw_write_all(fd, b"control-bytes")
        finally:
            os.close(fd)
        assert path.read_bytes() == (expected or b"")

    @pytest.mark.parametrize(
        "position", ["descriptor", "inventory", "checkpoint", "wal", "wal_tail", "meta"]
    )
    def test_zero_control_write_never_publishes_truncated_control_file(
        self, tmp_path: Path, monkeypatch, position: str
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        row = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        partition = root / "date=2023-11-14"
        target = {
            "descriptor": partition / ".raw-transaction.json",
            "inventory": root / ".raw-inventory.json",
            "checkpoint": root / "checkpoint.json",
            "wal": root / "wal.bin",
            "wal_tail": root / ".raw-wal-tail.json",
            "meta": root / "compaction.meta.json",
        }[position]
        before = target.read_bytes() if target.exists() else None
        real_write = os.write

        def reject_control_write(fd: int, data):
            name = Path(os.readlink(f"/proc/self/fd/{fd}")).name
            prefixes = {
                "descriptor": ".raw-transaction-",
                "inventory": ".raw-inventory-",
                "checkpoint": ".raw-checkpoint-",
                "wal": "wal.bin",
                "wal_tail": ".raw-wal-tail-",
                "meta": ".raw-meta-",
            }
            if name == prefixes[position] or name.startswith(prefixes[position]):
                return 0
            return real_write(fd, data)

        monkeypatch.setattr(
            "lumina_quant.storage.parquet.ohlcv_repo.os.write", reject_control_write
        )
        if position in {"descriptor", "inventory"}:

            def operation() -> None:
                repo.append_raw_aggtrades(
                    exchange="binance",
                    symbol="BTC/USDT",
                    rows=[_raw_row(2, row["timestamp_ms"] + 1)],
                )
        elif position == "checkpoint":

            def operation() -> None:
                repo.write_raw_checkpoint(
                    exchange="binance",
                    symbol="BTC/USDT",
                    payload=_checkpoint_payload(row),
                )
        elif position in {"wal", "wal_tail"}:

            def operation() -> None:
                repo.append_raw_wal_record(
                    exchange="binance", symbol="BTC/USDT", payload={"event": "x"}
                )
        else:

            def operation() -> None:
                repo._write_raw_meta(exchange="binance", symbol="BTC/USDT", payload={})

        with pytest.raises(OSError, match="no progress"):
            operation()
        assert (target.read_bytes() if target.exists() else None) == before

    @pytest.mark.parametrize(
        "position", ["descriptor", "inventory", "checkpoint", "meta", "wal_tail", "wal"]
    )
    def test_oversized_raw_control_rejects_before_read_or_mutation(
        self, tmp_path: Path, monkeypatch, position: str
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        row = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload={"event": "one"})
        targets = {
            "descriptor": root / "date=2023-11-14" / ".raw-transaction.json",
            "inventory": root / ".raw-inventory.json",
            "checkpoint": root / "checkpoint.json",
            "meta": root / "compaction.meta.json",
            "wal_tail": root / ".raw-wal-tail.json",
            "wal": root / "wal.bin",
        }
        target = targets[position]
        bounded_window = 2 * _RAW_WAL_MAX_RECORD_BYTES + 2
        if position == "wal":
            (root / ".raw-wal-tail.json").unlink()
            prefix = b"unread-legacy-wal-prefix\n"
            target.write_bytes(prefix + b"x" * (bounded_window + 1))
        else:
            target.write_bytes(b"x" * (_RAW_CONTROL_MAX_BYTES + 1))
        before = _raw_snapshot(root)
        real_read = os.read
        real_lseek = os.lseek
        read_events: list[tuple[int, int, int, bytes]] = []
        seek_events: list[tuple[int, int, int]] = []
        suffix_start = max(0, target.stat().st_size - bounded_window)

        def record_lseek(fd: int, offset: int, whence: int) -> int:
            result = real_lseek(fd, offset, whence)
            path = Path(os.readlink(f"/proc/self/fd/{fd}"))
            if position == "wal" and path == target:
                seek_events.append((offset, whence, result))
            return result

        def reject_oversized_read(fd: int, count: int) -> bytes:
            path = Path(os.readlink(f"/proc/self/fd/{fd}"))
            offset = real_lseek(fd, 0, os.SEEK_CUR)
            data = real_read(fd, count)
            if path == target:
                if position != "wal":
                    pytest.fail(f"oversized {position} was read before its size was rejected")
                read_events.append((offset, count, len(data), data))
            return data

        calls = _forbid_mutations(monkeypatch)
        monkeypatch.setattr("lumina_quant.storage.parquet.ohlcv_repo.os.lseek", record_lseek)
        monkeypatch.setattr(
            "lumina_quant.storage.parquet.ohlcv_repo.os.read", reject_oversized_read
        )
        if position == "checkpoint":

            def operation() -> None:
                repo.read_raw_checkpoint(exchange="binance", symbol="BTC/USDT")

            message = "malformed"
        elif position == "meta":

            def operation() -> None:
                repo.append_raw_aggtrades(
                    exchange="binance",
                    symbol="BTC/USDT",
                    rows=[_raw_row(2, row["timestamp_ms"] + 1)],
                )

            message = "too large"
        else:

            def operation() -> None:
                repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")

            message = "too large"

        with pytest.raises(ValueError, match=message):
            operation()
        if position == "wal":
            assert seek_events
            assert read_events
            assert all(result == suffix_start for _, _, result in seek_events)
            assert all(offset == suffix_start for offset, _, _, _ in read_events)
            assert all(
                offset + actual <= target.stat().st_size for offset, _, actual, _ in read_events
            )
            assert sum(actual for _, _, actual, _ in read_events) <= bounded_window
            assert all(prefix not in data for _, _, _, data in read_events)
        assert calls == []
        assert _raw_snapshot(root) == before

    @pytest.mark.parametrize("fault", ["short", "eintr"])
    def test_wal_tail_publication_retries_short_and_interrupted_writes(
        self, tmp_path: Path, monkeypatch, fault: str
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload={"event": "one"})
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        before = _raw_snapshot(root)
        real_write = os.write
        injected = False

        def interrupt_or_shorten_tail(fd: int, data) -> int:
            nonlocal injected
            name = Path(os.readlink(f"/proc/self/fd/{fd}")).name
            if not injected and name.startswith(".raw-wal-tail-"):
                injected = True
                if fault == "eintr":
                    raise InterruptedError
                return real_write(fd, bytes(data[:7]))
            return real_write(fd, data)

        monkeypatch.setattr(
            "lumina_quant.storage.parquet.ohlcv_repo.os.write", interrupt_or_shorten_tail
        )
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload={"event": "two"})

        assert injected
        tail = root / ".raw-wal-tail.json"
        assert json.loads(tail.read_text())["record_count"] == 2
        assert not list(root.glob(".raw-wal-tail-*.tmp"))
        assert _raw_snapshot(root) != before
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")


class TestRawPartTransactionRecovery:
    @pytest.mark.parametrize(
        "boundary", ["descriptor", "output", "inventory", "descriptor_cleanup"]
    )
    def test_every_published_prefix_converges_to_exact_new_generation(
        self, tmp_path: Path, monkeypatch, boundary: str
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        second = _raw_row(2, first["timestamp_ms"] + 1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        before = json.loads((root / ".raw-inventory.json").read_text())
        real_replace, real_unlink = os.replace, os.unlink
        captured: dict | None = None

        def fail_after_replace(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
            real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            destination_path = Path(os.readlink(f"/proc/self/fd/{dst_dir_fd}")) / destination
            if (
                (boundary == "descriptor" and destination == ".raw-transaction.json")
                or (boundary == "output" and destination_path.name.startswith("part-"))
                or (boundary == "inventory" and destination == ".raw-inventory.json")
            ):
                raise OSError(boundary)

        def fail_after_unlink(name, *, dir_fd=None):
            nonlocal captured
            if boundary == "descriptor_cleanup" and name == ".raw-transaction.json":
                descriptor_path = Path(os.readlink(f"/proc/self/fd/{dir_fd}")) / name
                captured = json.loads(descriptor_path.read_text())
            real_unlink(name, dir_fd=dir_fd)
            if boundary == "descriptor_cleanup" and name == ".raw-transaction.json":
                raise OSError(boundary)

        monkeypatch.setattr(os, "replace", fail_after_replace)
        monkeypatch.setattr(os, "unlink", fail_after_unlink)
        with pytest.raises(OSError, match=boundary):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[second])

        monkeypatch.setattr(os, "replace", real_replace)
        monkeypatch.setattr(os, "unlink", real_unlink)
        descriptor = next(root.glob("date=*/.raw-transaction.json"), None)
        published = captured if captured is not None else json.loads(descriptor.read_text())
        assert published["base_inventory"] == before
        assert published["new_inventory"]["generation"] == before["generation"] + 1
        for entry in published["base_inventory"]["parts"]:
            assert (
                next(
                    candidate
                    for candidate in published["new_inventory"]["parts"]
                    if candidate["name"] == entry["name"]
                )
                == entry
            )
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        new = json.loads((root / ".raw-inventory.json").read_text())
        assert new == published["new_inventory"]
        assert sorted(entry["max_trade_id"] for entry in new["parts"]) == [1, 2]
        assert not list(root.glob("date=*/.raw-transaction.json"))
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert json.loads((root / ".raw-inventory.json").read_text()) == new

    def _crash_after_multi_obsolete_compaction_descriptor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[ParquetMarketDataRepository, Path, Path, dict, list[dict]]:
        monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "false")
        repo = ParquetMarketDataRepository(tmp_path)
        rows = [_raw_row(index, 1_700_000_000_000 + index) for index in range(1, 4)]
        for row in rows:
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])

        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        partition = root / "date=2023-11-14"
        raw_replace = repo._raw_replace

        class SimulatedAbruptTermination(BaseException):
            pass

        def stop_after_descriptor(parent_fd: int, source: str, destination: str) -> None:
            raw_replace(parent_fd, source, destination)
            if destination == ".raw-transaction.json":
                raise SimulatedAbruptTermination("descriptor published")

        monkeypatch.setattr(repo, "_raw_replace", stop_after_descriptor)
        with pytest.raises(SimulatedAbruptTermination, match="descriptor published"):
            repo._compact_raw_partition(
                exchange="binance", symbol="BTC/USDT", partition_root=partition
            )
        monkeypatch.setattr(repo, "_raw_replace", raw_replace)

        descriptor = json.loads((partition / ".raw-transaction.json").read_text())
        assert descriptor["base_inventory"] == json.loads(
            (root / ".raw-inventory.json").read_text()
        )
        expected_obsolete = sorted(
            entry["name"].rsplit("/", 1)[1]
            for entry in descriptor["base_inventory"]["parts"]
            if entry["name"].startswith(f"{partition.name}/")
        )
        assert len(expected_obsolete) >= 2
        assert descriptor["obsolete"] == expected_obsolete
        assert all((partition / name).is_file() for name in descriptor["obsolete"])
        assert (partition / descriptor["stage"]).is_file()
        return repo, root, partition, descriptor, rows

    def test_multi_obsolete_compaction_descriptor_recovery_converges_exactly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import polars as pl

        repo, root, partition, descriptor, rows = (
            self._crash_after_multi_obsolete_compaction_descriptor(tmp_path, monkeypatch)
        )
        descriptor_path = partition / ".raw-transaction.json"

        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert json.loads((root / ".raw-inventory.json").read_text()) == descriptor["new_inventory"]
        assert (
            pl.read_parquet(partition / descriptor["output"]).sort("agg_trade_id").to_dicts()
            == rows
        )
        assert not descriptor_path.exists()
        assert not list(partition.glob(".raw-stage-*.parquet"))
        assert not list(partition.glob(".raw-transaction-*.tmp"))

        stable = _raw_snapshot(root)
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert _raw_snapshot(root) == stable

    @pytest.mark.parametrize("negative", ["old-output", "base-identity", "stage", "temp"])
    def test_multi_obsolete_compaction_descriptor_rejects_changed_closure_without_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, negative: str
    ):
        repo, root, partition, descriptor, _ = (
            self._crash_after_multi_obsolete_compaction_descriptor(tmp_path, monkeypatch)
        )
        if negative == "old-output":
            (partition / descriptor["obsolete"][0]).write_bytes(b"changed")
        elif negative == "base-identity":
            inventory = root / ".raw-inventory.json"
            payload = json.loads(inventory.read_text())
            payload["generation"] += 1
            inventory.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            name = (
                ".raw-stage-11111111111111111111111111111111.parquet"
                if negative == "stage"
                else ".raw-transaction-22222222222222222222222222222222.tmp"
            )
            (partition / name).write_bytes(b"unrelated")
        before = _raw_snapshot(root)

        with pytest.raises(
            ValueError,
            match=(
                r"^unbound transaction artifact closure is invalid$"
                if negative in {"stage", "temp"}
                else (
                    r"^(?:transaction part bytes do not match descriptor|"
                    r"transaction part is not strict raw parquet|"
                    r"Raw aggTrades part is not strict parquet: .+|"
                    r"obsolete transaction part changed|"
                    r"Raw aggTrades inventory is malformed)$"
                )
            ),
        ):
            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert _raw_snapshot(root) == before

    def test_authenticated_stale_descriptor_replay_rejects_without_mutation(
        self, tmp_path: Path, monkeypatch
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        second = _raw_row(2, first["timestamp_ms"] + 1)
        third = _raw_row(3, second["timestamp_ms"] + 1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        real_replace = os.replace

        def stop_after_descriptor(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
            real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            if destination == ".raw-transaction.json":
                raise OSError("captured")

        with (
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.replace",
                side_effect=stop_after_descriptor,
            ),
            pytest.raises(OSError, match="captured"),
        ):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[second])

        descriptor = next(root.glob("date=*/.raw-transaction.json"))
        stale_descriptor = descriptor.read_bytes()
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[third])
        descriptor.write_bytes(stale_descriptor)

        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError, match="does not match current inventory"):
            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert calls == []
        assert _raw_snapshot(root) == before

    def test_generation_advanced_after_descriptor_validation_cannot_be_overwritten(
        self, tmp_path: Path
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        second = _raw_row(2, first["timestamp_ms"] + 1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        partition = root / "date=2023-11-14"
        real_replace = repo._raw_replace
        advanced = False
        competing_generation: int | None = None
        competing_inventory_sha256: str | None = None
        competing_namespace: dict[str, bytes] | None = None

        def advance_generation_after_descriptor(
            parent_fd: int, source: str, destination: str
        ) -> None:
            nonlocal advanced, competing_generation, competing_inventory_sha256, competing_namespace
            real_replace(parent_fd, source, destination)
            if destination != ".raw-transaction.json" or advanced:
                return
            advanced = True
            current = repo._authenticate_raw_inventory(
                exchange="binance", symbol="BTC/USDT", snapshot=True
            )
            repo._publish_raw_inventory(
                root=root,
                payload=repo._raw_inventory_payload(
                    exchange="binance",
                    symbol="BTCUSDT",
                    generation=current.generation + 1,
                    previous_inventory_sha256=current.inventory_sha256,
                    entries=list(current.entries),
                ),
            )
            competing = repo._authenticate_raw_inventory(
                exchange="binance", symbol="BTC/USDT", snapshot=True
            )
            competing_generation = competing.generation
            competing_inventory_sha256 = competing.inventory_sha256
            competing_namespace = _raw_snapshot(root)

        with (
            patch.object(repo, "_raw_replace", side_effect=advance_generation_after_descriptor),
            pytest.raises(ValueError, match="transaction does not match current inventory"),
        ):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[second])

        assert advanced
        assert competing_generation is not None
        assert competing_inventory_sha256 is not None
        assert competing_namespace is not None
        inventory = repo._authenticate_raw_inventory(
            exchange="binance", symbol="BTC/USDT", snapshot=True
        )
        assert inventory.generation == competing_generation
        assert inventory.inventory_sha256 == competing_inventory_sha256
        assert sorted(entry["max_trade_id"] for entry in inventory.entries) == [1]
        assert _raw_snapshot(root) == competing_namespace
        assert (partition / ".raw-transaction.json").is_file()
        assert len(list(partition.glob(".raw-stage-*.parquet"))) == 1


class TestRawUnpublishedArtifactRecovery:
    @pytest.mark.parametrize("boundary", ["stage", "descriptor_temp"])
    def test_unbound_transaction_artifacts_are_rejected_without_mutation(
        self, tmp_path: Path, monkeypatch, boundary: str
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        partition = root / "date=2023-11-14"
        original_open, original_replace = os.open, os.replace

        class SimulatedAbruptTermination(BaseException):
            pass

        def crash_after_stage_durability(name, flags, mode=0o777, *, dir_fd=None):
            if boundary == "stage" and str(name).startswith(".raw-transaction-"):
                raise SimulatedAbruptTermination("crash after stage durability")
            return original_open(name, flags, mode, dir_fd=dir_fd)

        def crash_after_descriptor_temp_durability(
            source, destination, *, src_dir_fd=None, dst_dir_fd=None
        ):
            if boundary == "descriptor_temp" and destination == ".raw-transaction.json":
                raise SimulatedAbruptTermination("crash after descriptor temp durability")
            return original_replace(
                source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd
            )

        monkeypatch.setattr(os, "open", crash_after_stage_durability)
        monkeypatch.setattr(os, "replace", crash_after_descriptor_temp_durability)
        with pytest.raises(SimulatedAbruptTermination, match="crash after"):
            repo.append_raw_aggtrades(
                exchange="binance",
                symbol="BTC/USDT",
                rows=[_raw_row(2, first["timestamp_ms"] + 1)],
            )
        monkeypatch.setattr(os, "open", original_open)
        monkeypatch.setattr(os, "replace", original_replace)

        unbound = [
            path
            for pattern in (
                ".raw-stage-*.parquet",
                ".raw-transaction-*.tmp",
            )
            for path in partition.glob(pattern)
        ]
        assert len(unbound) == (2 if boundary == "descriptor_temp" else 1)
        assert not (partition / ".raw-transaction.json").exists()
        before_recovery = _raw_snapshot(root)

        for _ in range(2):
            with pytest.raises(
                ValueError, match=r"^unbound transaction artifact closure is invalid$"
            ):
                repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
            assert _raw_snapshot(root) == before_recovery
            assert all(path.exists() for path in unbound)

    @pytest.mark.parametrize("artifact", ["stage", "descriptor-temp"])
    @pytest.mark.parametrize("operation", ["recover", "append", "compact"])
    def test_authenticated_descriptor_rejects_unbound_artifact_before_mutation(
        self, tmp_path: Path, monkeypatch, artifact: str, operation: str
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        partition = root / "date=2023-11-14"
        raw_replace = repo._raw_replace

        class SimulatedAbruptTermination(BaseException):
            pass

        def stop_after_descriptor(parent_fd: int, source: str, destination: str) -> None:
            raw_replace(parent_fd, source, destination)
            if destination == ".raw-transaction.json":
                raise SimulatedAbruptTermination("captured authenticated descriptor")

        monkeypatch.setattr(repo, "_raw_replace", stop_after_descriptor)
        with pytest.raises(SimulatedAbruptTermination, match="captured authenticated descriptor"):
            repo.append_raw_aggtrades(
                exchange="binance",
                symbol="BTC/USDT",
                rows=[_raw_row(2, first["timestamp_ms"] + 1)],
            )
        monkeypatch.setattr(repo, "_raw_replace", raw_replace)

        unrelated = (
            partition
            / {
                "stage": ".raw-stage-11111111111111111111111111111111.parquet",
                "descriptor-temp": ".raw-transaction-22222222222222222222222222222222.tmp",
            }[artifact]
        )
        unrelated.write_bytes(b"unbound")
        assert (partition / ".raw-transaction.json").is_file()
        before = _raw_snapshot(root)

        with pytest.raises(ValueError, match=r"^unbound transaction artifact closure is invalid$"):
            if operation == "recover":
                repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
            elif operation == "append":
                repo.append_raw_aggtrades(
                    exchange="binance",
                    symbol="BTC/USDT",
                    rows=[_raw_row(3, first["timestamp_ms"] + 2)],
                )
            else:
                repo._compact_raw_partition(
                    exchange="binance", symbol="BTC/USDT", partition_root=partition
                )
        assert _raw_snapshot(root) == before
        assert unrelated.is_file()


class TestRawObsoleteCleanupRecovery:
    @pytest.mark.parametrize("deleted_prefix", [0, 1, 2])
    def test_every_authorized_obsolete_deletion_prefix_recovers_idempotently(
        self, tmp_path: Path, monkeypatch, deleted_prefix: int
    ):
        monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "8")
        monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "false")
        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        rows = [_raw_row(index, first["timestamp_ms"] + index) for index in range(1, 4)]
        for row in rows:
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        partition = root / "date=2023-11-14"
        original_unlink = os.unlink
        deleted = 0

        def crash_at_obsolete_prefix(name, *, dir_fd=None):
            nonlocal deleted
            if name.startswith("part-"):
                if deleted == deleted_prefix:
                    raise OSError(f"crash after {deleted} obsolete deletions")
                original_unlink(name, dir_fd=dir_fd)
                deleted += 1
                return
            if name == ".raw-transaction.json" and deleted == deleted_prefix:
                raise OSError(f"crash after {deleted} obsolete deletions")
            original_unlink(name, dir_fd=dir_fd)

        monkeypatch.setattr(os, "unlink", crash_at_obsolete_prefix)
        with pytest.raises(OSError, match="obsolete deletions"):
            repo._compact_raw_partition(
                exchange="binance", symbol="BTC/USDT", partition_root=partition
            )
        monkeypatch.setattr(os, "unlink", original_unlink)

        descriptor = partition / ".raw-transaction.json"
        assert descriptor.exists()
        descriptor_document = json.loads(descriptor.read_text())
        actual_obsolete = [
            name
            for name in descriptor_document["obsolete"]
            if name != descriptor_document["output_entry"]["name"].rsplit("/", 1)[1]
        ]
        assert len(actual_obsolete) == 2
        assert deleted_prefix <= len(actual_obsolete)
        expected = descriptor_document["new_inventory"]
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert json.loads((root / ".raw-inventory.json").read_text()) == expected
        assert not descriptor.exists()
        assert not list(partition.glob(".raw-stage-*.parquet"))
        assert {path.name for path in partition.glob("part-*.parquet")} == {
            entry["name"].rsplit("/", 1)[1] for entry in expected["parts"]
        }


class TestRawFilesystemSafety:
    @pytest.mark.parametrize("exchange", ["../outside", "/absolute", "binance/unsafe"])
    def test_unsafe_exchange_cannot_escape_raw_root(self, tmp_path: Path, exchange: str):
        repo = ParquetMarketDataRepository(tmp_path)
        before = _raw_snapshot(tmp_path)
        with pytest.raises(ValueError):
            repo.append_raw_aggtrades(exchange=exchange, symbol="BTC/USDT", rows=[_raw_row(1)])
        assert _raw_snapshot(tmp_path) == before

    @pytest.mark.parametrize("kind", ["symlink", "fifo", "socket"])
    def test_hostile_intermediate_component_is_rejected_before_raw_file_mutation(
        self, monkeypatch, kind: str
    ):
        temporary = tempfile.TemporaryDirectory(prefix="lq-intermediate-")
        server: socket.socket | None = None
        try:
            repository_root = Path(temporary.name)
            repo = ParquetMarketDataRepository(repository_root)
            raw_root = repository_root / "market_data_raw_aggtrades"
            raw_root.mkdir()
            hostile = raw_root / "binance"
            external = repository_root / "external"
            if kind == "symlink":
                external.mkdir()
                (external / "sentinel").write_bytes(b"external")
                hostile.symlink_to(external, target_is_directory=True)
            elif kind == "fifo":
                os.mkfifo(hostile)
            else:
                server = socket.socket(socket.AF_UNIX)
                server.bind(str(hostile))

            before = _raw_snapshot(repository_root)
            external_before = _raw_snapshot(external) if kind == "symlink" else None
            calls = _forbid_mutations(monkeypatch)
            with pytest.raises((OSError, ValueError)):
                repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[_raw_row(1)])
            assert calls == []
            assert _raw_snapshot(repository_root) == before
            if external_before is not None:
                assert _raw_snapshot(external) == external_before
        finally:
            if server is not None:
                server.close()
            monkeypatch.undo()
            temporary.cleanup()

    @pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "socket", "directory"])
    @pytest.mark.parametrize(
        "position",
        ["part", "descriptor", "stage", "inventory", "checkpoint", "wal", "wal_tail", "meta"],
    )
    def test_hostile_managed_file_is_rejected_before_follow_or_mutation(
        self, tmp_path: Path, monkeypatch, kind: str, position: str
    ):
        short_root = (
            tempfile.TemporaryDirectory(dir="/tmp", prefix="lq") if kind == "socket" else None
        )
        server: socket.socket | None = None
        try:
            if short_root is not None:
                monkeypatch.chdir(short_root.name)
            repository_root = Path(".") if short_root is not None else tmp_path
            exchange, symbol = ("x", "Y/Z") if kind == "socket" else ("binance", "BTC/USDT")
            repo = ParquetMarketDataRepository(repository_root)
            row = _raw_row(1)
            repo.append_raw_aggtrades(exchange=exchange, symbol=symbol, rows=[row])
            root = repo._raw_symbol_root(exchange=exchange, symbol=symbol)
            partition = root / "date=2023-11-14"

            if position in {"descriptor", "stage"}:
                real_replace = os.replace

                def stop_after_descriptor(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
                    real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
                    if destination == ".raw-transaction.json":
                        raise OSError("captured")

                with (
                    patch(
                        "lumina_quant.storage.parquet.ohlcv_repo.os.replace",
                        side_effect=stop_after_descriptor,
                    ),
                    pytest.raises(OSError, match="captured"),
                ):
                    repo.append_raw_aggtrades(
                        exchange=exchange,
                        symbol=symbol,
                        rows=[_raw_row(2, row["timestamp_ms"] + 1)],
                    )
                descriptor = partition / ".raw-transaction.json"
                transaction = json.loads(descriptor.read_text())
                target = (
                    descriptor if position == "descriptor" else partition / transaction["stage"]
                )
            else:
                target = {
                    "part": partition / "part-0000.parquet",
                    "inventory": root / ".raw-inventory.json",
                    "checkpoint": root / "checkpoint.json",
                    "wal": root / "wal.bin",
                    "wal_tail": root / ".raw-wal-tail.json",
                    "meta": root / "compaction.meta.json",
                }[position]

            external = repository_root / f"external-{kind}-{position}"
            external.write_bytes(b"external")
            if target.exists():
                target.unlink()
            if kind == "symlink":
                target.symlink_to(external)
            elif kind == "hardlink":
                os.link(external, target)
            elif kind == "fifo":
                os.mkfifo(target)
                real_open = os.open

                def nonblocking_open(name, flags, mode=0o777, *, dir_fd=None):
                    if name == target.name and dir_fd is not None:
                        assert flags & os.O_NONBLOCK, (
                            "FIFO must be rejected without a blocking open"
                        )
                    return real_open(name, flags, mode, dir_fd=dir_fd)

                monkeypatch.setattr(os, "open", nonblocking_open)
            elif kind == "socket":
                server = socket.socket(socket.AF_UNIX)
                server.bind(str(target))
            else:
                target.mkdir()
            before, external_before = _raw_snapshot(root), external.lstat()
            calls = _forbid_mutations(monkeypatch)
            if position in {"part", "inventory"}:

                def operation() -> None:
                    repo.append_raw_aggtrades(
                        exchange=exchange,
                        symbol=symbol,
                        rows=[_raw_row(3, row["timestamp_ms"] + 2)],
                    )
            elif position in {"descriptor", "stage"}:

                def operation() -> None:
                    repo.recover_raw_stream(exchange=exchange, symbol=symbol)
            elif position == "checkpoint":

                def operation() -> None:
                    repo.write_raw_checkpoint(
                        exchange=exchange,
                        symbol=symbol,
                        payload=_checkpoint_payload(row),
                    )
            elif position in {"wal", "wal_tail"}:

                def operation() -> None:
                    repo.append_raw_wal_record(
                        exchange=exchange, symbol=symbol, payload={"event": "x"}
                    )
            else:
                monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "1")
                monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "false")

                def operation() -> None:
                    repo.append_raw_aggtrades(
                        exchange=exchange,
                        symbol=symbol,
                        rows=[_raw_row(3, row["timestamp_ms"] + 2)],
                    )

            with pytest.raises((ValueError, OSError, RuntimeError)):
                operation()
            assert calls == []
            assert _raw_snapshot(root) == before
            assert external.lstat() == external_before
        finally:
            if server is not None:
                server.close()
            if short_root is not None:
                monkeypatch.undo()
                short_root.cleanup()


class TestRawInventoryCompatibility:
    def test_fresh_stream_genesis_is_v2_and_missing_inventory_with_parts_fails_closed(
        self, tmp_path: Path, monkeypatch
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[_raw_row(1)])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory = root / ".raw-inventory.json"
        assert json.loads(inventory.read_text())["version"] == 2
        inventory.unlink()
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[_raw_row(2)])
        assert calls == []
        assert _raw_snapshot(root) == before


class TestRawInventoryAuthenticity:
    def test_genuine_stale_inventory_replay_fails_without_mutation(
        self, tmp_path: Path, monkeypatch
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory = root / ".raw-inventory.json"
        stale = inventory.read_bytes()
        repo.append_raw_aggtrades(
            exchange="binance", symbol="BTC/USDT", rows=[_raw_row(2, first["timestamp_ms"] + 1)]
        )
        inventory.write_bytes(stale)
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError):
            repo.append_raw_aggtrades(
                exchange="binance", symbol="BTC/USDT", rows=[_raw_row(3, first["timestamp_ms"] + 2)]
            )
        assert calls == []
        assert _raw_snapshot(root) == before

    def test_v1_inventory_requires_explicit_one_time_migration(self, tmp_path: Path):
        import hashlib

        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory = root / ".raw-inventory.json"
        current = json.loads(inventory.read_text())
        legacy_body = {"version": 1, "parts": current["parts"]}
        legacy = {
            **legacy_body,
            "inventory_sha256": hashlib.sha256(
                json.dumps(legacy_body, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        inventory.write_text(json.dumps(legacy, sort_keys=True, separators=(",", ":")))

        before = _raw_snapshot(root)
        with pytest.raises(ValueError, match="migration"):
            repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")
        assert _raw_snapshot(root) == before

        replacements: list[tuple[str, str]] = []
        real_replace = os.replace

        def record_replace(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
            replacements.append((str(source), str(destination)))
            real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        with patch(
            "lumina_quant.storage.parquet.ohlcv_repo.os.replace", side_effect=record_replace
        ):
            migrated = repo._authenticate_raw_inventory(
                exchange="binance", symbol="BTC/USDT", migrate=True
            )
            repo._authenticate_raw_inventory(exchange="binance", symbol="BTC/USDT", migrate=True)
        assert migrated[0]["name"] == current["parts"][0]["name"]
        assert json.loads(inventory.read_text())["version"] == 2
        assert [destination for _, destination in replacements].count(".raw-inventory.json") == 1

    def test_public_append_migrates_valid_v1_inventory_before_compaction(
        self, tmp_path: Path, monkeypatch
    ):
        import hashlib

        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory_path = root / ".raw-inventory.json"
        current = json.loads(inventory_path.read_text())
        original_parts = current["parts"]
        legacy_body = {"version": 1, "parts": original_parts}
        legacy_body["inventory_sha256"] = hashlib.sha256(
            json.dumps(legacy_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inventory_path.write_text(json.dumps(legacy_body, sort_keys=True, separators=(",", ":")))

        publications: list[dict] = []
        compaction_publication_counts: list[int] = []
        real_replace = os.replace
        real_compact = repo._compact_raw_partition

        def capture_inventory_publish(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
            if destination == ".raw-inventory.json":
                source_path = (
                    Path(os.readlink(f"/proc/self/fd/{src_dir_fd}")) / source
                    if src_dir_fd is not None
                    else Path(source)
                )
                publications.append(json.loads(source_path.read_text()))
            real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        def capture_compaction(*args, **kwargs):
            compaction_publication_counts.append(len(publications))
            return real_compact(*args, **kwargs)

        monkeypatch.setattr(repo, "_compact_raw_partition", capture_compaction)
        monkeypatch.setattr(
            "lumina_quant.storage.parquet.ohlcv_repo.os.replace",
            capture_inventory_publish,
        )
        monkeypatch.setenv("LQ_RAW_PARTITION_MAX_PARTS", "1")
        monkeypatch.setenv("LQ_RAW_COMPACT_ON_THRESHOLD", "true")
        assert (
            repo.append_raw_aggtrades(
                exchange="binance",
                symbol="BTC/USDT",
                rows=[_raw_row(2, first["timestamp_ms"] + 1)],
            )
            == 1
        )
        assert compaction_publication_counts
        migration_indexes = [
            index
            for index, payload in enumerate(publications)
            if payload["parts"] == original_parts
        ]
        assert migration_indexes == [0]
        assert migration_indexes[0] < compaction_publication_counts[0]
        assert publications[0]["version"] == 2
        assert publications[0]["parts"] == original_parts
        for previous, published in pairwise(publications):
            assert published["generation"] == previous["generation"] + 1
            assert published["previous_inventory_sha256"] == previous["inventory_sha256"]
        migrated = json.loads(inventory_path.read_text())
        assert migrated["version"] == 2
        assert [entry["max_trade_id"] for entry in migrated["parts"]] == [2]
        publication_count = len(publications)
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert len(publications) == publication_count

    def test_malformed_v1_inventory_rejects_public_append_without_mutation(
        self, tmp_path: Path, monkeypatch
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        row = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory_path = root / ".raw-inventory.json"
        legacy = {"version": 1, "parts": [], "inventory_sha256": "not-a-digest"}
        inventory_path.write_text(json.dumps(legacy, sort_keys=True, separators=(",", ":")))
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError):
            repo.append_raw_aggtrades(
                exchange="binance",
                symbol="BTC/USDT",
                rows=[_raw_row(2, row["timestamp_ms"] + 1)],
            )
        assert calls == []
        assert _raw_snapshot(root) == before

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda parts: list(reversed(parts)),
            lambda parts: [parts[0], parts[0]],
        ],
        ids=["noncanonical-order", "duplicate-part"],
    )
    def test_v2_inventory_rejects_noncanonical_or_duplicate_part_order(
        self, tmp_path: Path, monkeypatch, mutation
    ):
        import hashlib

        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        repo.append_raw_aggtrades(
            exchange="binance", symbol="BTC/USDT", rows=[_raw_row(2, first["timestamp_ms"] + 1)]
        )
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory_path = root / ".raw-inventory.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["parts"] = mutation(inventory["parts"])
        body = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
        inventory["inventory_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inventory_path.write_text(json.dumps(inventory, sort_keys=True, separators=(",", ":")))

        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError):
            repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")
        assert calls == []
        assert _raw_snapshot(root) == before

    def test_inode_replacement_with_same_bytes_and_restored_mtime_is_rejected(
        self, tmp_path: Path, monkeypatch
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        row = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory = json.loads((root / ".raw-inventory.json").read_text())
        part = root / inventory["parts"][0]["name"]
        original = part.stat()
        replacement = part.with_name(".replacement.parquet")
        replacement.write_bytes(part.read_bytes())
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        os.replace(replacement, part)
        assert part.stat().st_ino != original.st_ino
        assert part.stat().st_mtime_ns == original.st_mtime_ns
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError):
            repo.append_raw_aggtrades(
                exchange="binance", symbol="BTC/USDT", rows=[_raw_row(2, row["timestamp_ms"] + 1)]
            )
        assert calls == []
        assert _raw_snapshot(root) == before

    def test_same_inode_rewrite_with_changed_mtime_is_rejected(self, tmp_path: Path, monkeypatch):
        repo = ParquetMarketDataRepository(tmp_path)
        row = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory = json.loads((root / ".raw-inventory.json").read_text())
        part = root / inventory["parts"][0]["name"]
        original = part.stat()
        part.write_bytes(part.read_bytes())
        os.utime(
            part,
            ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000),
        )
        assert part.stat().st_ino == original.st_ino
        assert part.stat().st_mtime_ns == original.st_mtime_ns + 1_000_000
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError):
            repo.append_raw_aggtrades(
                exchange="binance", symbol="BTC/USDT", rows=[_raw_row(2, row["timestamp_ms"] + 1)]
            )
        assert calls == []
        assert _raw_snapshot(root) == before

    @pytest.mark.parametrize("operation", ["load", "merge"])
    def test_authenticated_content_digest_mismatch_is_rejected_when_part_is_read(
        self, tmp_path: Path, monkeypatch, operation: str
    ):
        import hashlib

        repo = ParquetMarketDataRepository(tmp_path)
        row = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory_path = root / ".raw-inventory.json"
        inventory = json.loads(inventory_path.read_text())
        inventory["parts"][0]["content_sha256"] = "0" * 64
        body = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
        inventory["inventory_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inventory_path.write_text(json.dumps(inventory, sort_keys=True, separators=(",", ":")))
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises((ValueError, RuntimeError)):
            if operation == "load":
                repo.load_raw_aggtrades(exchange="binance", symbol="BTC/USDT")
            else:
                repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        assert calls == []
        assert _raw_snapshot(root) == before

    def test_compaction_rejects_selected_digest_mismatch_without_hashing_history(
        self, tmp_path: Path, monkeypatch
    ):
        import hashlib

        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        second = _raw_row(2, first["timestamp_ms"] + 86_400_000)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[second])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory_path = root / ".raw-inventory.json"
        inventory = json.loads(inventory_path.read_text())
        historical, selected = inventory["parts"]
        historical_path = root / historical["name"]
        selected["content_sha256"] = "0" * 64
        body = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
        inventory["inventory_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inventory_path.write_text(json.dumps(inventory, sort_keys=True, separators=(",", ":")))
        digests: list[Path] = []
        raw_digest = ParquetMarketDataRepository._raw_file_digest

        def record_digest(self, path):
            if path == historical_path:
                digests.append(path)
            return raw_digest(self, path)

        monkeypatch.setattr(ParquetMarketDataRepository, "_raw_file_digest", record_digest)
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(RuntimeError, match="Cannot compact unreadable part"):
            repo._compact_raw_partition(
                exchange="binance",
                symbol="BTC/USDT",
                partition_root=root / selected["name"].split("/", 1)[0],
            )
        assert digests == []
        assert calls == []
        assert _raw_snapshot(root) == before

    def test_strict_tail_append_does_not_hash_untouched_authenticated_history(
        self, tmp_path: Path, monkeypatch
    ):
        import hashlib

        repo = ParquetMarketDataRepository(tmp_path)
        row = _raw_row(1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[row])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        inventory_path = root / ".raw-inventory.json"
        inventory = json.loads(inventory_path.read_text())
        historical = root / inventory["parts"][0]["name"]
        inventory["parts"][0]["content_sha256"] = "0" * 64
        body = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
        inventory["inventory_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        inventory_path.write_text(json.dumps(inventory, sort_keys=True, separators=(",", ":")))
        reads: list[Path] = []
        digests: list[Path] = []
        read_authenticated = ParquetMarketDataRepository._raw_read_authenticated_parquet
        file_digest = ParquetMarketDataRepository._raw_file_digest

        def record_read(self, path, entry):
            if path == historical:
                reads.append(path)
            return read_authenticated(self, path, entry)

        def record_digest(self, path):
            if path == historical:
                digests.append(path)
            return file_digest(self, path)

        monkeypatch.setattr(
            ParquetMarketDataRepository, "_raw_read_authenticated_parquet", record_read
        )
        monkeypatch.setattr(ParquetMarketDataRepository, "_raw_file_digest", record_digest)
        assert (
            repo.append_raw_aggtrades(
                exchange="binance",
                symbol="BTC/USDT",
                rows=[_raw_row(2, row["timestamp_ms"] + 86_400_000)],
            )
            == 1
        )
        assert reads == []
        assert digests == []

    def test_transaction_output_digest_mismatch_is_rejected_before_inventory_publish(
        self, tmp_path: Path, monkeypatch
    ):
        import hashlib

        repo = ParquetMarketDataRepository(tmp_path)
        first = _raw_row(1)
        second = _raw_row(2, first["timestamp_ms"] + 1)
        repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[first])
        root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
        real_replace = os.replace

        def stop_after_output(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
            real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            if destination.startswith("part-"):
                raise OSError("captured output")

        monkeypatch.setattr(os, "replace", stop_after_output)
        with pytest.raises(OSError, match="captured output"):
            repo.append_raw_aggtrades(exchange="binance", symbol="BTC/USDT", rows=[second])
        monkeypatch.setattr(os, "replace", real_replace)
        descriptor_path = next(root.glob("date=*/.raw-transaction.json"))
        descriptor = json.loads(descriptor_path.read_text())
        descriptor["output_entry"]["content_sha256"] = "0" * 64
        for entry in descriptor["new_inventory"]["parts"]:
            if entry["name"] == descriptor["output_entry"]["name"]:
                entry["content_sha256"] = "0" * 64
        inventory_body = {
            key: value
            for key, value in descriptor["new_inventory"].items()
            if key != "inventory_sha256"
        }
        descriptor["new_inventory"]["inventory_sha256"] = hashlib.sha256(
            json.dumps(inventory_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        descriptor_body = {
            key: value for key, value in descriptor.items() if key != "descriptor_sha256"
        }
        descriptor["descriptor_sha256"] = hashlib.sha256(
            json.dumps(descriptor_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        descriptor_path.write_text(json.dumps(descriptor, sort_keys=True, separators=(",", ":")))
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError, match="transaction part bytes"):
            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert calls == []
        assert _raw_snapshot(root) == before


class TestRawJsonlWalRecovery:
    def test_torn_suffix_is_truncated_before_a_new_canonical_record_is_appended(
        self, tmp_path: Path
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        first, second = {"event": "first"}, {"event": "second"}
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload=first)
        path = repo.raw_wal_path(exchange="binance", symbol="BTC/USDT")
        path.write_bytes(path.read_bytes() + b'{"torn"')
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload=second)

        records = self._v2_wal_records(path)
        assert [marker["sequence"] for marker in records] == [1, 2]
        assert [marker["payload"] for marker in records] == [first, second]
        assert records[0]["previous_record_sha256"] is None
        assert records[1]["previous_record_sha256"] == self._wal_record_digest(
            path.read_bytes().splitlines(keepends=True)[0]
        )

    @staticmethod
    def _tail_document(path: Path) -> dict:
        return json.loads(path.read_text())

    @staticmethod
    def _tail_with_hash(document: dict) -> bytes:
        import hashlib

        body = {key: value for key, value in document.items() if key != "tail_sha256"}
        document["tail_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    @staticmethod
    def _v2_wal_records(path: Path) -> list[dict]:
        lines = path.read_bytes().splitlines(keepends=True)
        envelopes = [json.loads(line) for line in lines]
        canonical = [
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for envelope in envelopes
        ]
        assert lines == canonical
        records = [envelope["_raw_wal_v2"] for envelope in envelopes]
        for index, marker in enumerate(records, start=1):
            assert marker["version"] == 2
            assert marker["sequence"] == index
            expected_previous = (
                None if index == 1 else TestRawJsonlWalRecovery._wal_record_digest(lines[index - 2])
            )
            assert marker["previous_record_sha256"] == expected_previous
        return records

    @staticmethod
    def _wal_record_digest(record: bytes) -> str:
        return sha256(record).hexdigest()

    def test_legacy_wal_without_sidecar_is_migrated_once(self, tmp_path: Path):
        repo = ParquetMarketDataRepository(tmp_path)
        wal = repo.raw_wal_path(exchange="binance", symbol="BTC/USDT")
        tail = wal.with_name(".raw-wal-tail.json")
        wal.parent.mkdir(parents=True)
        wal.write_bytes(b'{"event":"legacy"}\n')
        replacements: list[str] = []
        real_replace = os.replace

        def record_replace(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
            replacements.append(str(destination))
            real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        with patch(
            "lumina_quant.storage.parquet.ohlcv_repo.os.replace", side_effect=record_replace
        ):
            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        migrated = self._tail_document(tail)
        records = wal.read_bytes().splitlines(keepends=True)
        assert len(records) == 2
        assert json.loads(records[0]) == {"event": "legacy"}
        anchor = json.loads(records[1])["_raw_wal_v2"]
        assert anchor == {
            "version": 2,
            "sequence": 2,
            "previous_record_sha256": self._wal_record_digest(records[0]),
            "payload": {"_raw_wal_migration": "legacy"},
        }
        assert set(migrated) == {
            "version",
            "wal_identity",
            "last_offset",
            "last_length",
            "last_sha256",
            "record_count",
            "generation",
            "previous_tail_sha256",
            "tail_sha256",
        }
        assert migrated["version"] == 1
        assert set(migrated["wal_identity"]) == {
            "dev",
            "ino",
            "size",
            "mtime_ns",
            "ctime_ns",
        }
        assert migrated["last_offset"] + migrated["last_length"] == wal.stat().st_size
        assert migrated["record_count"] == 2
        assert migrated["last_length"] == len(records[1])
        assert migrated["last_sha256"] == self._wal_record_digest(records[1])
        assert migrated["tail_sha256"]
        assert (
            self._tail_document(tail)["tail_sha256"]
            == json.loads(self._tail_with_hash(dict(migrated)))["tail_sha256"]
        )
        assert replacements.count(".raw-wal-tail.json") == 1

        before = wal.read_bytes(), tail.read_bytes()
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert (wal.read_bytes(), tail.read_bytes()) == before

    @pytest.mark.parametrize(
        ("legacy_bytes", "record_count"),
        [
            (b'{"event":"legacy"}\n{"torn"', 2),
            (b'{"torn"', 1),
        ],
        ids=["legacy-prefix-with-torn-suffix", "torn-only"],
    )
    def test_every_legacy_wal_migration_mutation_prefix_recovers_idempotently(
        self, tmp_path: Path, legacy_bytes: bytes, record_count: int
    ):

        def seed(root: Path) -> ParquetMarketDataRepository:
            repo = ParquetMarketDataRepository(root)
            wal = repo.raw_wal_path(exchange="binance", symbol="BTC/USDT")
            wal.parent.mkdir(parents=True)
            wal.write_bytes(legacy_bytes)
            return repo

        baseline = seed(tmp_path / "baseline")
        baseline_log = _RawProtocolLog()
        with (
            patch("lumina_quant.storage.parquet.ohlcv_repo.os.open", side_effect=baseline_log.open),
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.mkdir", side_effect=baseline_log.mkdir
            ),
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.write", side_effect=baseline_log.write
            ),
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.pwrite", side_effect=baseline_log.pwrite
            ),
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.ftruncate",
                side_effect=baseline_log.ftruncate,
            ),
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=baseline_log.fsync
            ),
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.replace",
                side_effect=baseline_log.replace,
            ),
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.unlink", side_effect=baseline_log.unlink
            ),
        ):
            baseline.recover_raw_stream(exchange="binance", symbol="BTC/USDT")

        assert {event.kind for event in baseline_log.events} >= {
            "truncate",
            "write",
            "file-fsync",
            "parent-fsync",
            "rename",
        }
        assert any(
            "bootstrap" in event.path.name or "bootstrap" in (event.other or Path()).name
            for event in baseline_log.events
        ), baseline_log.events
        assert any(
            event.path.name == "wal.bin" and event.kind == "write" for event in baseline_log.events
        )
        assert any(
            event.other is not None and event.other.name == ".raw-wal-tail.json"
            for event in baseline_log.events
        )
        for publish in (
            event
            for event in baseline_log.events
            if event.kind == "rename"
            and event.other is not None
            and event.other.name
            in {".raw-wal-bootstrap.json", ".raw-wal-tail.json", ".raw-wal-migration.json"}
        ):
            write = max(
                (
                    event
                    for event in baseline_log.events
                    if event.kind in {"write", "pwrite"}
                    and event.path == publish.path
                    and event.completed
                    and event.occurrence < baseline_log.one("file-fsync", publish.path).occurrence
                ),
                key=lambda event: event.occurrence,
            )
            fsync = baseline_log.after(write, "file-fsync", publish.path)
            baseline_log.assert_chain(
                write,
                fsync,
                publish,
                baseline_log.after(publish, "parent-fsync", publish.other.parent),
            )

        for crash_after in range(len(baseline_log.events)):
            repo = seed(tmp_path / f"prefix-{crash_after}")
            log = _RawProtocolLog()

            class SimulatedAbruptTermination(BaseException):
                pass

            def crash_after_event(
                operation,
                *,
                event_log=log,
                crash_index=crash_after,
            ):
                def wrapped(*args, **kwargs):
                    result = operation(*args, **kwargs)
                    if len(event_log.events) == crash_index + 1:
                        raise SimulatedAbruptTermination(
                            f"crash after protocol event {crash_index}"
                        )
                    return result

                return wrapped

            with (
                patch(
                    "lumina_quant.storage.parquet.ohlcv_repo.os.open",
                    side_effect=crash_after_event(log.open),
                ),
                patch(
                    "lumina_quant.storage.parquet.ohlcv_repo.os.mkdir",
                    side_effect=crash_after_event(log.mkdir),
                ),
                patch(
                    "lumina_quant.storage.parquet.ohlcv_repo.os.write",
                    side_effect=crash_after_event(log.write),
                ),
                patch(
                    "lumina_quant.storage.parquet.ohlcv_repo.os.pwrite",
                    side_effect=crash_after_event(log.pwrite),
                ),
                patch(
                    "lumina_quant.storage.parquet.ohlcv_repo.os.ftruncate",
                    side_effect=crash_after_event(log.ftruncate),
                ),
                patch(
                    "lumina_quant.storage.parquet.ohlcv_repo.os.fsync",
                    side_effect=crash_after_event(log.fsync),
                ),
                patch(
                    "lumina_quant.storage.parquet.ohlcv_repo.os.replace",
                    side_effect=crash_after_event(log.replace),
                ),
                patch(
                    "lumina_quant.storage.parquet.ohlcv_repo.os.unlink",
                    side_effect=crash_after_event(log.unlink),
                ),
                pytest.raises(SimulatedAbruptTermination, match="crash after protocol event"),
            ):
                repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")

            root = repo._raw_symbol_root(exchange="binance", symbol="BTC/USDT")
            control_temp_publications = {
                ".raw-wal-bootstrap-": ".raw-wal-bootstrap.json",
                ".raw-wal-tail-": ".raw-wal-tail.json",
            }
            orphan_control_temps = [
                temp
                for prefix, publication in control_temp_publications.items()
                for temp in root.glob(f"{prefix}*.tmp")
                if (
                    len(token := temp.name.removeprefix(prefix).removesuffix(".tmp")) == 32
                    and all(character in "0123456789abcdef" for character in token)
                    and not (root / publication).exists()
                )
            ]
            if orphan_control_temps:
                for _ in range(2):
                    before_recovery = _raw_snapshot(root)
                    with pytest.raises(ValueError, match=r"^unbound raw control temp is present$"):
                        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
                    assert _raw_snapshot(root) == before_recovery
                continue

            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
            wal = repo.raw_wal_path(exchange="binance", symbol="BTC/USDT")
            records = wal.read_bytes().splitlines(keepends=True)
            anchor = json.loads(records[-1])["_raw_wal_v2"]
            if record_count == 2:
                assert json.loads(records[0]) == {"event": "legacy"}
                assert anchor == {
                    "version": 2,
                    "sequence": 2,
                    "previous_record_sha256": self._wal_record_digest(records[0]),
                    "payload": {"_raw_wal_migration": "legacy"},
                }
            else:
                assert len(records) == 1
                assert anchor == {
                    "version": 2,
                    "sequence": 1,
                    "previous_record_sha256": None,
                    "payload": {"_raw_wal_migration": "legacy"},
                }
            tail = self._tail_document(wal.with_name(".raw-wal-tail.json"))
            assert tail["record_count"] == record_count
            assert tail["last_sha256"] == self._wal_record_digest(records[-1])
            assert not list(root.glob("*.tmp"))
            assert not (root / ".raw-wal-bootstrap.json").exists()
            assert not (root / ".raw-wal-migration.json").exists()
            before = wal.read_bytes(), (root / ".raw-wal-tail.json").read_bytes()
            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
            assert (wal.read_bytes(), (root / ".raw-wal-tail.json").read_bytes()) == before

    @pytest.mark.parametrize("operation", ["append", "recover"])
    def test_established_wal_uses_only_sidecar_and_final_record(
        self, tmp_path: Path, monkeypatch, operation: str
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        for index in range(128):
            repo.append_raw_wal_record(
                exchange="binance", symbol="BTC/USDT", payload={"event": f"event-{index}"}
            )
        wal = repo.raw_wal_path(exchange="binance", symbol="BTC/USDT")
        tail = wal.with_name(".raw-wal-tail.json")
        tail_bytes = tail.read_bytes()
        tail_document = self._tail_document(tail)
        final_record = wal.read_bytes().splitlines(keepends=True)[-1]
        reads: list[tuple[Path, int, int, bytes]] = []
        preads: list[tuple[Path, int, int, bytes]] = []
        real_read = os.read
        real_pread = os.pread
        real_lseek = os.lseek

        def tracked_path(fd: int) -> Path:
            return Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()

        def record_reads(fd: int, count: int) -> bytes:
            path = tracked_path(fd)
            offset = real_lseek(fd, 0, os.SEEK_CUR)
            data = real_read(fd, count)
            if path in {wal.resolve(), tail.resolve()}:
                reads.append((path, offset, count, data))
                if path == wal.resolve() and offset < tail_document["last_offset"]:
                    pytest.fail("established WAL access reached historical bytes")
            return data

        def record_preads(fd: int, count: int, offset: int) -> bytes:
            path = tracked_path(fd)
            data = real_pread(fd, count, offset)
            if path in {wal.resolve(), tail.resolve()}:
                preads.append((path, offset, count, data))
                if path == wal.resolve() and offset < tail_document["last_offset"]:
                    pytest.fail("established WAL pread reached historical bytes")
            return data

        def forbid_raw_file_digest(*args, **kwargs) -> None:
            pytest.fail(f"established WAL path hashed a historical file: {args!r} {kwargs!r}")

        monkeypatch.setattr("lumina_quant.storage.parquet.ohlcv_repo.os.read", record_reads)
        monkeypatch.setattr("lumina_quant.storage.parquet.ohlcv_repo.os.pread", record_preads)
        monkeypatch.setattr(repo, "_raw_file_digest", forbid_raw_file_digest)
        if operation == "append":
            repo.append_raw_wal_record(
                exchange="binance", symbol="BTC/USDT", payload={"event": "four"}
            )
        else:
            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")

        observed = reads + preads
        allowed_reads = {
            (tail.resolve(), 0, len(tail_bytes), tail_bytes),
            (tail.resolve(), len(tail_bytes), 1, b""),
            (
                wal.resolve(),
                tail_document["last_offset"],
                len(final_record),
                final_record,
            ),
        }
        assert observed
        assert all(read in allowed_reads for read in observed)
        assert (
            wal.resolve(),
            tail_document["last_offset"],
            len(final_record),
            final_record,
        ) in observed
        wal_reads = [read for read in observed if read[0] == wal.resolve()]
        tail_reads = [read for read in observed if read[0] == tail.resolve()]
        assert len(wal_reads) == 1
        assert sum(len(data) for _, _, _, data in wal_reads) == len(final_record)
        assert len(tail_reads) <= 4
        assert sum(len(data) for _, _, _, data in tail_reads) <= 2 * len(tail_bytes)
        assert tail_document["last_length"] == len(final_record)
        assert len(final_record) <= _RAW_WAL_MAX_RECORD_BYTES

    def test_complete_wal_suffix_before_sidecar_publish_recovers_idempotently(self, tmp_path: Path):
        repo = ParquetMarketDataRepository(tmp_path)
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload={"event": "one"})
        wal = repo.raw_wal_path(exchange="binance", symbol="BTC/USDT")
        tail = wal.with_name(".raw-wal-tail.json")
        before_tail = self._tail_document(tail)
        real_replace = os.replace

        def stop_before_tail_replace(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
            if destination == ".raw-wal-tail.json":
                raise OSError("crash before tail publish")
            real_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        with (
            patch(
                "lumina_quant.storage.parquet.ohlcv_repo.os.replace",
                side_effect=stop_before_tail_replace,
            ),
            pytest.raises(OSError, match="crash before tail publish"),
        ):
            repo.append_raw_wal_record(
                exchange="binance", symbol="BTC/USDT", payload={"event": "two"}
            )

        records = self._v2_wal_records(wal)
        assert [marker["sequence"] for marker in records] == [1, 2]
        assert records[-1]["payload"] == {"event": "two"}
        assert records[-1]["previous_record_sha256"] == self._wal_record_digest(
            wal.read_bytes().splitlines(keepends=True)[-2]
        )
        assert self._tail_document(tail) == before_tail
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        recovered = self._tail_document(tail)
        assert recovered["record_count"] == before_tail["record_count"] + 1
        assert recovered["last_offset"] + recovered["last_length"] == wal.stat().st_size
        assert recovered["last_sha256"] == self._wal_record_digest(
            wal.read_bytes().splitlines(keepends=True)[-1]
        )
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert self._tail_document(tail) == recovered

    def test_torn_wal_suffix_truncates_to_authenticated_tail_and_fsyncs(self, tmp_path: Path):
        repo = ParquetMarketDataRepository(tmp_path)
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload={"event": "one"})
        wal = repo.raw_wal_path(exchange="binance", symbol="BTC/USDT")
        tail = wal.with_name(".raw-wal-tail.json")
        authenticated = self._tail_document(tail)
        expected = wal.read_bytes()
        wal.write_bytes(expected + b'{"torn"')
        log = _RawProtocolLog()

        with patch("lumina_quant.storage.parquet.ohlcv_repo.os.fsync", side_effect=log.fsync):
            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")

        rebuilt = self._tail_document(tail)
        record = wal.read_bytes()
        stat = wal.stat()
        assert record == expected
        assert rebuilt["generation"] == authenticated["generation"] + 1
        assert rebuilt["previous_tail_sha256"] == authenticated["tail_sha256"]
        assert rebuilt["record_count"] == authenticated["record_count"]
        assert rebuilt["last_offset"] == authenticated["last_offset"]
        assert rebuilt["last_length"] == authenticated["last_length"]
        assert rebuilt["last_sha256"] == authenticated["last_sha256"]
        assert rebuilt["last_sha256"] == sha256(record).hexdigest()
        assert rebuilt["last_offset"] + rebuilt["last_length"] == stat.st_size
        assert rebuilt["wal_identity"] == {
            "dev": stat.st_dev,
            "ino": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        log.assert_chain(
            log.one("file-fsync", wal),
            log.after(log.one("file-fsync", wal), "parent-fsync", wal.parent),
        )
        repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert self._tail_document(tail) == rebuilt

    @pytest.mark.parametrize("fault", ["missing", "divergent", "replayed"])
    def test_established_wal_tail_rejects_invalid_state_without_mutation(
        self, tmp_path: Path, monkeypatch, fault: str
    ):
        repo = ParquetMarketDataRepository(tmp_path)
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload={"event": "one"})
        wal = repo.raw_wal_path(exchange="binance", symbol="BTC/USDT")
        tail = wal.with_name(".raw-wal-tail.json")
        old_tail = tail.read_bytes()
        repo.append_raw_wal_record(exchange="binance", symbol="BTC/USDT", payload={"event": "two"})
        if fault == "missing":
            tail.unlink()
        elif fault == "divergent":
            document = self._tail_document(tail)
            document["last_length"] += 1
            tail.write_bytes(self._tail_with_hash(document))
        else:
            repo.append_raw_wal_record(
                exchange="binance", symbol="BTC/USDT", payload={"event": "three"}
            )
            tail.write_bytes(old_tail)

        root = wal.parent
        before = _raw_snapshot(root)
        calls = _forbid_mutations(monkeypatch)
        with pytest.raises(ValueError):
            repo.recover_raw_stream(exchange="binance", symbol="BTC/USDT")
        assert calls == []
        assert _raw_snapshot(root) == before


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
