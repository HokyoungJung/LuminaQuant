"""Custom binary WAL for 1-second OHLCV storage.

Record format (fixed 64 bytes, little-endian):
- MAGIC (4 bytes): b"LQWB"
- VERSION (1 byte): 1
- FLAGS (1 byte): reserved (0)
- RESERVED (2 bytes): padding
- RECORD_LEN (4 bytes): fixed record length (64)
- TS_MS (8 bytes, int64): UTC epoch milliseconds
- OPEN (8 bytes, float64)
- HIGH (8 bytes, float64)
- LOW (8 bytes, float64)
- CLOSE (8 bytes, float64)
- VOLUME (8 bytes, float64)
- CRC32 (4 bytes): CRC of VERSION..VOLUME bytes (excludes MAGIC and CRC field)
"""

from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAGIC = b"LQWB"
VERSION = 1
FLAGS_DEFAULT = 0
RECORD_LEN = 64

_BODY_NO_CRC_STRUCT = struct.Struct("<BBHIqddddd")
_BODY_WITH_CRC_STRUCT = struct.Struct("<BBHIqdddddI")
_RECORD_STRUCT = struct.Struct("<4sBBHIqdddddI")


@dataclass(slots=True, frozen=True)
class WALRecord:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.ts_ms / 1000.0, tz=UTC).replace(tzinfo=None)


def _coerce_timestamp_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        ts = int(value)
        if abs(ts) < 100_000_000_000:
            ts *= 1000
        return ts

    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(dt.astimezone(UTC).timestamp() * 1000)

    token = str(value or "").strip()
    if not token:
        raise ValueError("timestamp cannot be empty")
    if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
        return _coerce_timestamp_ms(int(token))

    dt = datetime.fromisoformat(token.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _coerce_record(row: WALRecord | Mapping[str, Any] | tuple[Any, ...]) -> WALRecord:
    if isinstance(row, WALRecord):
        record = row
    elif isinstance(row, Mapping):
        record = WALRecord(
            ts_ms=_coerce_timestamp_ms(row.get("ts_ms", row.get("datetime"))),
            open=float(row.get("open")),
            high=float(row.get("high")),
            low=float(row.get("low")),
            close=float(row.get("close")),
            volume=float(row.get("volume")),
        )
    else:
        values = tuple(row)
        if len(values) < 6:
            raise ValueError(f"expected at least 6 tuple values, got {len(values)}")
        record = WALRecord(
            ts_ms=_coerce_timestamp_ms(values[0]),
            open=float(values[1]),
            high=float(values[2]),
            low=float(values[3]),
            close=float(values[4]),
            volume=float(values[5]),
        )

    if record.ts_ms % 1000 != 0:
        raise ValueError(f"ts_ms must be second-aligned for 1s bars: {record.ts_ms}")
    return record


def encode_record(record: WALRecord, *, flags: int = FLAGS_DEFAULT) -> bytes:
    body = _BODY_NO_CRC_STRUCT.pack(
        VERSION,
        int(flags) & 0xFF,
        0,
        RECORD_LEN,
        int(record.ts_ms),
        float(record.open),
        float(record.high),
        float(record.low),
        float(record.close),
        float(record.volume),
    )
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return MAGIC + body + struct.pack("<I", crc)


def decode_record(payload: bytes) -> WALRecord | None:
    if len(payload) != RECORD_LEN:
        return None

    try:
        magic, version, _flags, _reserved, record_len, ts_ms, opn, high, low, close, volume, crc = (
            _RECORD_STRUCT.unpack(payload)
        )
    except struct.error:
        return None

    if magic != MAGIC:
        return None
    if int(version) != VERSION:
        return None
    if int(record_len) != RECORD_LEN:
        return None

    body = payload[4:-4]
    computed_crc = zlib.crc32(body) & 0xFFFFFFFF
    if int(crc) != computed_crc:
        return None

    if int(ts_ms) % 1000 != 0:
        return None

    return WALRecord(
        ts_ms=int(ts_ms),
        open=float(opn),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(volume),
    )


class BinaryWAL:
    """Append-only custom binary WAL with CRC validation and repair."""

    def __init__(
        self,
        path: str | Path,
        *,
        fsync_every_n_batches: int = 1,
        auto_repair: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_every_n_batches = max(1, int(fsync_every_n_batches))
        self._batches_since_fsync = 0

        if not self.path.exists():
            self.path.touch()
        if auto_repair:
            self.repair()

    def append(self, rows: Iterable[WALRecord | Mapping[str, Any] | tuple[Any, ...]]) -> int:
        encoded = bytearray()
        count = 0
        for row in rows:
            record = _coerce_record(row)
            encoded.extend(encode_record(record))
            count += 1

        if count == 0:
            return 0

        with self.path.open("ab") as fh:
            fh.write(encoded)
            fh.flush()
            self._batches_since_fsync += 1
            if self._batches_since_fsync >= self.fsync_every_n_batches:
                os.fsync(fh.fileno())
                self._batches_since_fsync = 0

        return count

    def force_fsync(self) -> None:
        with self.path.open("ab") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        self._batches_since_fsync = 0

    def iter_all(self) -> Iterator[WALRecord]:
        yield from self.iter_range(None, None)

    def iter_range(self, start_ts_ms: int | None, end_ts_ms: int | None) -> Iterator[WALRecord]:
        start = None if start_ts_ms is None else int(start_ts_ms)
        end = None if end_ts_ms is None else int(end_ts_ms)

        with self.path.open("rb") as fh:
            while True:
                chunk = fh.read(RECORD_LEN)
                if not chunk:
                    break
                if len(chunk) != RECORD_LEN:
                    # Partial trailing bytes — cannot form a complete record.
                    break

                record = decode_record(chunk)
                if record is None:
                    # Corrupt slot: skip and continue scanning for valid records
                    # further in the file (skip-and-resync).
                    continue

                if start is not None and record.ts_ms < start:
                    continue
                if end is not None and record.ts_ms > end:
                    # Keep scanning instead of early-breaking. Although WAL writes
                    # are usually time-ordered, late-arriving corrections may append
                    # older timestamps and still need to be visible in-range.
                    continue
                yield record

    def scan_valid_length(self) -> int:
        """Return byte length up to the last valid record boundary.

        Scans the entire file, skipping over corrupt records so that valid
        records appearing *after* a mid-file corruption are not silently lost.
        The returned length covers all records up to (and including) the last
        contiguous-or-scattered valid record; repair() will truncate everything
        past the last valid boundary.
        """
        import logging

        valid_end = 0
        corrupt_skipped = 0
        offset = 0
        with self.path.open("rb") as fh:
            while True:
                chunk = fh.read(RECORD_LEN)
                if not chunk:
                    break
                if len(chunk) != RECORD_LEN:
                    # Partial trailing bytes — note as corrupt, stop scanning.
                    corrupt_skipped += 1
                    break
                if decode_record(chunk) is None:
                    # Corrupt record: skip past it and keep scanning for more
                    # valid records further in the file.
                    corrupt_skipped += 1
                    offset += RECORD_LEN
                    continue
                offset += RECORD_LEN
                valid_end = offset

        if corrupt_skipped:
            logging.getLogger(__name__).warning(
                "WAL %s: skipped %d corrupt/partial record slot(s) during scan; "
                "last valid byte offset=%d",
                self.path,
                corrupt_skipped,
                valid_end,
            )
        return valid_end

    def repair(self) -> int:
        """Truncate past the last valid record boundary and return bytes removed.

        Unlike a simple leading-scan, this first calls scan_valid_length() which
        skips mid-file corrupt records, so repair discards only the unreachable
        tail bytes rather than everything after the first bad slot.
        """
        file_size = self.path.stat().st_size if self.path.exists() else 0
        valid_end = self.scan_valid_length()
        if valid_end >= file_size:
            return 0

        with self.path.open("r+b") as fh:
            fh.truncate(valid_end)
            fh.flush()
            os.fsync(fh.fileno())

        return int(file_size - valid_end)

    def truncate(self) -> None:
        with self.path.open("r+b") as fh:
            fh.truncate(0)
            fh.flush()
            os.fsync(fh.fileno())
        self._batches_since_fsync = 0

    def size_bytes(self) -> int:
        return int(self.path.stat().st_size if self.path.exists() else 0)

    def iter_records_from_offset(self, offset: int) -> Iterator[WALRecord]:
        cursor = max(0, int(offset))
        with self.path.open("rb") as fh:
            fh.seek(cursor)
            while True:
                chunk = fh.read(RECORD_LEN)
                if not chunk or len(chunk) != RECORD_LEN:
                    break
                record = decode_record(chunk)
                if record is None:
                    # Corrupt slot: skip and continue (skip-and-resync).
                    cursor += RECORD_LEN
                    continue
                yield record
                cursor += RECORD_LEN
