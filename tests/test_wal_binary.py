from __future__ import annotations

from pathlib import Path

from lumina_quant.storage.wal_binary import RECORD_LEN, BinaryWAL


def _rows(count: int):
    return [
        {
            "datetime": 1_700_000_000_000 + idx * 1000,
            "open": 100.0 + idx,
            "high": 101.0 + idx,
            "low": 99.0 + idx,
            "close": 100.5 + idx,
            "volume": 1.0 + idx,
        }
        for idx in range(count)
    ]


def test_wal_roundtrip_write_read(tmp_path: Path):
    wal = BinaryWAL(tmp_path / "wal.bin", auto_repair=True)
    written = wal.append(_rows(4))

    assert written == 4
    records = list(wal.iter_all())
    assert len(records) == 4
    assert records[0].ts_ms == 1_700_000_000_000
    assert records[-1].close == 103.5


def test_wal_repair_truncates_partial_tail(tmp_path: Path):
    wal_path = tmp_path / "wal.bin"
    wal = BinaryWAL(wal_path, auto_repair=True)
    wal.append(_rows(3))

    with wal_path.open("ab") as fh:
        fh.write(b"broken-tail")

    repaired = BinaryWAL(wal_path, auto_repair=True)
    assert repaired.size_bytes() == 3 * RECORD_LEN
    assert [item.ts_ms for item in repaired.iter_all()] == [
        1_700_000_000_000,
        1_700_000_001_000,
        1_700_000_002_000,
    ]


def test_wal_repair_truncates_invalid_crc_record(tmp_path: Path):
    wal_path = tmp_path / "wal.bin"
    wal = BinaryWAL(wal_path, auto_repair=True)
    wal.append(_rows(2))

    # Corrupt one byte in the second record payload.
    with wal_path.open("r+b") as fh:
        fh.seek(RECORD_LEN + 12)
        byte = fh.read(1)
        fh.seek(RECORD_LEN + 12)
        fh.write(bytes([byte[0] ^ 0xFF]))

    repaired = BinaryWAL(wal_path, auto_repair=True)
    assert repaired.size_bytes() == RECORD_LEN
    items = list(repaired.iter_all())
    assert len(items) == 1
    assert items[0].ts_ms == 1_700_000_000_000


def test_wal_iter_range_filters_bounds(tmp_path: Path):
    wal = BinaryWAL(tmp_path / "wal.bin", auto_repair=True)
    wal.append(_rows(6))

    subset = list(wal.iter_range(1_700_000_002_000, 1_700_000_004_000))
    assert [item.ts_ms for item in subset] == [
        1_700_000_002_000,
        1_700_000_003_000,
        1_700_000_004_000,
    ]
