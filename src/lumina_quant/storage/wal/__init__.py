"""Write-ahead-log storage primitives."""

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
from lumina_quant.storage.wal.native_backend import (
    append_ohlcv_frame_native,
    native_wal_append_available,
)

__all__ = [
    "FLAGS_DEFAULT",
    "MAGIC",
    "RECORD_LEN",
    "VERSION",
    "BinaryWAL",
    "WALRecord",
    "append_ohlcv_frame_native",
    "decode_record",
    "encode_record",
    "native_wal_append_available",
]
