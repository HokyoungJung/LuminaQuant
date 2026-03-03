"""Backward-compatible WAL exports.

Prefer importing from ``lumina_quant.storage.wal.binary``.
"""

from lumina_quant.storage.wal.binary import (
    FLAGS_DEFAULT,
    MAGIC,
    RECORD_LEN,
    VERSION,
    BinaryWAL,
    WALRecord,
    decode_record,
    encode_record,
)

__all__ = [
    "FLAGS_DEFAULT",
    "MAGIC",
    "RECORD_LEN",
    "VERSION",
    "BinaryWAL",
    "WALRecord",
    "decode_record",
    "encode_record",
]
