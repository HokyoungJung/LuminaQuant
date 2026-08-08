"""Monthly-parquet + custom-WAL market-data repository."""

from __future__ import annotations

import base64
import errno
import fcntl
import json
import io
import math
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import wraps
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import polars as pl
from lumina_quant.backtesting.cli_contract import (
    RawFirstDataMissingError,
    RawFirstManifestInvalidError,
    RawFirstStaleWindowError,
    normalize_data_mode,
)
from lumina_quant.data.raw_first_lineage import resample_1s_frame
from lumina_quant.data.timeframe import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.storage.wal.binary import BinaryWAL, WALRecord
from lumina_quant.storage.wal.native_backend import append_ohlcv_frame_native
from lumina_quant.symbols import canonical_symbol

_DEFAULT_SCHEMA: dict[str, pl.DataType] = {
    "datetime": pl.Datetime(time_unit="ms"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
_KNOWN_QUOTES = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
_OHLCV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
_CONFLICT_AUTHORIZATION_DOMAIN = b"luminaquant.alpha_max.canonical_conflict_authorization.v2\0"
_MANIFEST_REQUIRED_FIELDS = (
    "manifest_version",
    "commit_id",
    "symbol",
    "timeframe",
    "partition",
    "window_start_ms",
    "window_end_ms",
    "event_time_watermark_ms",
    "source_checkpoint_start",
    "source_checkpoint_end",
    "row_count",
    "canonical_row_checksum",
    "data_files",
    "created_at_utc",
    "producer",
    "status",
)
_RAW_AGGTRADES_SCHEMA: dict[str, pl.DataType] = {
    "agg_trade_id": pl.Int64,
    "timestamp_ms": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "is_buyer_maker": pl.Boolean,
}
_RAW_AGGTRADES_REQUIRED_COLUMNS = tuple(_RAW_AGGTRADES_SCHEMA.keys())
_MATERIALIZED_REQUIRED_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")
_MATERIALIZED_REQUIRED_MANIFEST_FIELDS = (
    "manifest_version",
    "commit_id",
    "symbol",
    "timeframe",
    "partition",
    "window_start_ms",
    "window_end_ms",
    "event_time_watermark_ms",
    "source_checkpoint_start",
    "source_checkpoint_end",
    "row_count",
    "canonical_row_checksum",
    "data_files",
    "created_at_utc",
    "producer",
    "status",
)
_RAW_PART_PATTERN = re.compile(r"^part-(\d+)\.parquet$")
_RAW_TRANSACTION_STAGE_PATTERN = re.compile(r"^\.raw-stage-[0-9a-f]{32}\.parquet$")
_RAW_TRANSACTION_TEMP_PATTERN = re.compile(r"^\.raw-transaction-[0-9a-f]{32}\.tmp$")
_RAW_CONTROL_TEMP_PATTERN = re.compile(
    r"^\.raw-(?:checkpoint|inventory|meta|wal-bootstrap|wal-tail)-[0-9a-f]{32}\.tmp$"
)
_RAW_CHECKPOINT_FIELDS = frozenset(
    {
        "exchange",
        "symbol",
        "last_timestamp_ms",
        "last_trade_id",
        "observed_until_ms",
        "updated_at_utc",
        "batch_rows",
        "last_row",
        "last_row_sha256",
    }
)
_RAW_WAL_MAX_RECORD_BYTES = 1_048_576
_RAW_CONTROL_MAX_BYTES = 8 * 1_048_576
_RAW_TRANSACTION_NAME = ".raw-transaction.json"

_RAW_INVENTORY_NAME = ".raw-inventory.json"
_RAW_WAL_NAME = "wal.bin"
_RAW_META_NAME = "compaction.meta.json"
_RAW_WAL_TAIL_NAME = ".raw-wal-tail.json"
_RAW_WAL_BOOTSTRAP_NAME = ".raw-wal-bootstrap.json"
_RAW_WAL_TAIL_VERSION = 1
_RAW_WAL_RECORD_VERSION = 2
_RAW_COMPONENT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_RAW_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_TOKEN_PATTERN = re.compile(r"^\d{4}-\d{2}$")
_CANONICAL_PARTITION_SEAL_SCHEMA = "alpha_max_canonical_partition_seal.v1"


@dataclass
class _GenerationPin:
    root: Path
    exclusive: bool
    active: bool = True


@dataclass
class _RawStreamLease:
    """Small, process-safe raw-stream lease backed by an owned flock file."""

    lock_path: Path
    _fd: int
    _generation_lock: Any | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self._fd)
            finally:
                try:
                    if self._generation_lock is not None:
                        self._generation_lock.__exit__(None, None, None)
                finally:
                    self._released = True


@dataclass(frozen=True)
class _RawInventorySnapshot:
    """One authenticated, immutable-on-disk inventory generation."""

    raw: bytes
    payload: dict[str, Any]
    entries: tuple[dict[str, Any], ...]
    generation: int
    inventory_sha256: str
    file_identity: tuple[int, int, int, int, int]


class RawPartitionBusyError(RuntimeError):
    """Raised when a raw partition cannot acquire its writer lock in time."""


class SealedMonthlyPartitionConflictError(RuntimeError):
    """Raised when a monthly partition is sealed or conflicts with a seal transaction."""


def _generation_guard(*, exclusive: bool):
    def decorate(method):
        @wraps(method)
        def guarded(self, *args, **kwargs):
            with self.generation_lock(exclusive=exclusive):
                return method(self, *args, **kwargs)

        return guarded

    return decorate


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format into BASE/QUOTE uppercase."""
    return canonical_symbol(symbol)


# ``normalize_timeframe_token`` / ``timeframe_to_milliseconds`` are imported from
# ``lumina_quant.data.timeframe`` (the single canonical timeframe util) and
# re-exported here for the existing in-module callers and ``storage.parquet``
# package re-exports.


@dataclass(slots=True)
class CompactionResult:
    """Compaction metadata for a single monthly parquet file."""

    partition: str
    files_before: int
    files_after: int
    rows_before: int
    rows_after: int


class ParquetMarketDataRepository:
    """Store and query OHLCV bars in monthly parquet + custom WAL layout.

    Layout:
    - monthly parquet: <root>/market_ohlcv_1s/<exchange>/<symbol>/<YYYY-MM>.parquet
    - wal:             <root>/market_ohlcv_1s/<exchange>/<symbol>/wal.bin
    """

    def __init__(self, root_path: str | Path):
        self.logical_root_path = Path(root_path)
        self._generation_lock_path = self.logical_root_path.parent / (
            f".{self.logical_root_path.name}.generation.lock"
        )
        self._generation_context: ContextVar[_GenerationPin | None] = ContextVar(
            f"lumina_quant_generation_{id(self)}", default=None
        )
        self._resolve_logical_root()

    def _resolve_logical_root(self) -> Path:
        try:
            root_info = os.lstat(self.logical_root_path)
        except FileNotFoundError:
            # Legacy writers may construct the repository before bootstrap.
            return self.logical_root_path
        if stat.S_ISDIR(root_info.st_mode):
            return self.logical_root_path
        if not stat.S_ISLNK(root_info.st_mode):
            raise ValueError("Canonical root must be a directory or trusted sibling symlink")
        target = os.readlink(self.logical_root_path)
        generation_base = f".{self.logical_root_path.name}.generations"
        target_parts = Path(target).parts
        if (
            os.path.isabs(target)
            or len(target_parts) != 2
            or target_parts[0] != generation_base
            or target_parts[1] in {"", ".", ".."}
        ):
            raise ValueError("Canonical root symlink must name an owned sibling generation")
        resolved = self.logical_root_path.parent / target
        resolved_info = os.lstat(resolved)
        if not stat.S_ISDIR(resolved_info.st_mode):
            raise ValueError("Canonical root symlink target must be a directory")
        root_after = os.lstat(self.logical_root_path)
        if (root_info.st_dev, root_info.st_ino) != (
            root_after.st_dev,
            root_after.st_ino,
        ) or os.readlink(self.logical_root_path) != target:
            raise ValueError("Canonical root symlink identity changed during resolution")
        return resolved

    @property
    def root_path(self) -> Path:
        pinned = self._generation_context.get()
        if pinned is not None and pinned.active:
            return pinned.root
        return self.logical_root_path

    @staticmethod
    def _ensure_generation_lock_parent(path: Path) -> None:
        missing: list[Path] = []
        cursor = path
        while True:
            try:
                info = os.lstat(cursor)
            except FileNotFoundError:
                missing.append(cursor)
                parent = cursor.parent
                if parent == cursor:
                    raise
                cursor = parent
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise SealedMonthlyPartitionConflictError(
                    "Generation lock parent is not a directory"
                )
            break
        for directory in reversed(missing):
            os.mkdir(directory)
            parent_fd = os.open(
                directory.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)

    @contextmanager
    def generation_lock(
        self,
        exclusive: bool,
        *,
        timeout_seconds: float | None = None,
        poll_seconds: float = 0.05,
        allow_incomplete_bootstrap: bool = False,
    ):
        """Pin one physical generation under the shared global lock."""
        pinned = self._generation_context.get()
        if pinned is not None and pinned.active:
            if exclusive and not pinned.exclusive:
                raise SealedMonthlyPartitionConflictError(
                    "Cannot upgrade a shared generation lock to exclusive"
                )
            yield pinned.root
            return

        if not self._generation_lock_path.parent.exists():
            self._ensure_generation_lock_parent(self._generation_lock_path.parent)
        parent_info = os.lstat(self._generation_lock_path.parent)
        if not stat.S_ISDIR(parent_info.st_mode):
            raise SealedMonthlyPartitionConflictError(
                "Generation lock parent is not a stable directory"
            )
        fd = os.open(self._generation_lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        token = None
        pin: _GenerationPin | None = None
        try:
            info = os.fstat(fd)
            path_info = os.lstat(self._generation_lock_path)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or info.st_gid != os.getgid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)
            ):
                raise SealedMonthlyPartitionConflictError(
                    "Generation lock is not a stable owned 0600 regular file"
                )
            lock_mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            if timeout_seconds is None:
                fcntl.flock(fd, lock_mode)
            else:
                deadline = time.monotonic() + max(0.0, float(timeout_seconds))
                while True:
                    try:
                        fcntl.flock(fd, lock_mode | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise RawPartitionBusyError(
                                f"Canonical generation is busy: {self.logical_root_path}"
                            ) from None
                        time.sleep(max(0.001, float(poll_seconds)))
            locked_path_info = os.lstat(self._generation_lock_path)
            if (info.st_dev, info.st_ino) != (
                locked_path_info.st_dev,
                locked_path_info.st_ino,
            ):
                raise SealedMonthlyPartitionConflictError("Generation lock identity changed")
            physical_root = self._resolve_logical_root()
            try:
                physical_info = os.stat(physical_root, follow_symlinks=False)
            except FileNotFoundError:
                physical_info = None
            if physical_info is not None:
                if (
                    not stat.S_ISDIR(physical_info.st_mode)
                    or self._resolve_logical_root() != physical_root
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Canonical generation changed during lock acquisition"
                    )
                repeated_info = os.stat(physical_root, follow_symlinks=False)
                if (physical_info.st_dev, physical_info.st_ino) != (
                    repeated_info.st_dev,
                    repeated_info.st_ino,
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Canonical generation identity changed during lock acquisition"
                    )
                if (
                    os.path.lexists(physical_root / ".bootstrap-incomplete")
                    and not allow_incomplete_bootstrap
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Canonical generation bootstrap is incomplete"
                    )
            pin = _GenerationPin(root=physical_root, exclusive=exclusive)
            token = self._generation_context.set(pin)
            yield physical_root
        finally:
            if pin is not None:
                pin.active = False
            try:
                if token is not None:
                    try:
                        self._generation_context.reset(token)
                    except ValueError:
                        # A deferred raw lease may be released by another context.
                        pass
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    @staticmethod
    def _normalize_exchange(exchange: str) -> str:
        value = str(exchange).strip().lower()
        if (
            not _RAW_COMPONENT_PATTERN.fullmatch(value)
            or value in {".", ".."}
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("Raw exchange must be a canonical safe path component")
        return value

    @staticmethod
    def _normalize_symbol_token(symbol: str) -> str:
        value = normalize_symbol(symbol).replace("/", "")
        if (
            not _RAW_COMPONENT_PATTERN.fullmatch(value.lower())
            or value in {".", ".."}
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("Raw symbol must normalize to a canonical safe path component")
        return value

    @staticmethod
    def _normalize_raw_exchange_token(exchange: str) -> str:
        if not isinstance(exchange, str) or exchange != exchange.strip() or "\x00" in exchange:
            raise ValueError("Raw exchange must be a canonical safe path component")
        if (
            not _RAW_COMPONENT_PATTERN.fullmatch(exchange.lower())
            or exchange in {".", ".."}
            or "/" in exchange
            or "\\" in exchange
            or any(ord(char) < 32 or ord(char) == 127 for char in exchange)
        ):
            raise ValueError("Raw exchange must be a canonical safe path component")
        return exchange.lower()

    @staticmethod
    def _normalize_raw_symbol_token(symbol: str) -> str:
        if not isinstance(symbol, str) or symbol != symbol.strip() or "\x00" in symbol:
            raise ValueError("Raw symbol must be a canonical safe path component")
        if symbol in {".", ".."} or "\\" in symbol or symbol.startswith("/"):
            raise ValueError("Raw symbol must be a canonical safe path component")
        separators = sum(symbol.count(separator) for separator in "/_-")
        if separators > 1 or (
            "/" in symbol
            and (
                not all(symbol.split("/"))
                or symbol.count("/") != 1
                or any(component in {".", ".."} for component in symbol.split("/"))
            )
        ):
            raise ValueError("Raw symbol has ambiguous pair separators")
        if any(ord(char) < 32 or ord(char) == 127 for char in symbol):
            raise ValueError("Raw symbol must be a canonical safe path component")
        value = normalize_symbol(symbol).replace("/", "")
        if (
            not _RAW_COMPONENT_PATTERN.fullmatch(value.lower())
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
        ):
            raise ValueError("Raw symbol must normalize to a canonical safe path component")
        return value

    @staticmethod
    def _empty_ohlcv_frame() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "datetime": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            schema=_DEFAULT_SCHEMA,
        )

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            numeric = int(value)
            if abs(numeric) < 100_000_000_000:
                numeric *= 1000
            dt = datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
        else:
            text = str(value).strip()
            if not text:
                return None
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            return dt.astimezone(UTC).replace(tzinfo=None)
        return dt

    @staticmethod
    def _datetime_to_ms(value: datetime | None) -> int | None:
        if value is None:
            return None
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(dt.astimezone(UTC).timestamp() * 1000)

    @staticmethod
    def _ms_to_datetime(ts_ms: int) -> datetime:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=UTC).replace(tzinfo=None)

    @staticmethod
    def _month_token(dt: datetime) -> str:
        return f"{dt.year:04d}-{dt.month:02d}"

    @staticmethod
    def _month_token_from_ms(ts_ms: int) -> str:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=UTC).strftime("%Y-%m")

    @staticmethod
    def _partition_date_token(partition_date: str | date) -> str:
        token = (
            partition_date.strftime("%Y-%m-%d")
            if isinstance(partition_date, date)
            else str(partition_date).strip()
        )
        if not _RAW_DATE_PATTERN.fullmatch(token):
            raise ValueError("Raw partition date must be YYYY-MM-DD")
        try:
            if date.fromisoformat(token).strftime("%Y-%m-%d") != token:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Raw partition date must be a valid UTC calendar date") from exc
        return token

    @staticmethod
    def _iter_month_tokens(start: datetime, end: datetime) -> list[str]:
        cursor = datetime(start.year, start.month, 1)
        stop = datetime(end.year, end.month, 1)
        out: list[str] = []
        while cursor <= stop:
            out.append(f"{cursor.year:04d}-{cursor.month:02d}")
            if cursor.month == 12:
                cursor = datetime(cursor.year + 1, 1, 1)
            else:
                cursor = datetime(cursor.year, cursor.month + 1, 1)
        return out

    def _symbol_root(self, *, exchange: str, symbol: str) -> Path:
        return (
            self.root_path
            / "market_ohlcv_1s"
            / self._normalize_exchange(exchange)
            / self._normalize_symbol_token(symbol)
        )

    def _monthly_path(self, *, exchange: str, symbol: str, month_token: str) -> Path:
        return self._symbol_root(exchange=exchange, symbol=symbol) / f"{month_token}.parquet"

    def _wal_path(self, *, exchange: str, symbol: str) -> Path:
        return self._symbol_root(exchange=exchange, symbol=symbol) / "wal.bin"

    @staticmethod
    def _strict_month_token(month: str) -> str:
        if not isinstance(month, str) or not _MONTH_TOKEN_PATTERN.fullmatch(month):
            raise ValueError("Month must be strict YYYY-MM")
        try:
            if datetime.strptime(month, "%Y-%m").strftime("%Y-%m") != month:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Month must be a valid calendar month") from exc
        return month

    def _monthly_lock_path(self, *, exchange: str, symbol: str, month_token: str) -> Path:
        return self._monthly_path(
            exchange=exchange, symbol=symbol, month_token=month_token
        ).with_suffix(".lock")

    def _monthly_pending_path(self, *, exchange: str, symbol: str, month_token: str) -> Path:
        return self._monthly_path(
            exchange=exchange, symbol=symbol, month_token=month_token
        ).with_suffix(".pending.json")

    def _monthly_seal_path(self, *, exchange: str, symbol: str, month_token: str) -> Path:
        return self._monthly_path(
            exchange=exchange, symbol=symbol, month_token=month_token
        ).with_suffix(".seal.json")

    @contextmanager
    def _monthly_lock(self, *, exchange: str, symbol: str, month_token: str):
        """Acquire the stable advisory lock for one canonical monthly partition."""
        lock_path = self._monthly_lock_path(
            exchange=exchange, symbol=symbol, month_token=month_token
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            path_info = os.lstat(lock_path)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)
            ):
                raise SealedMonthlyPartitionConflictError(
                    "Monthly lock is not a stable owned 0600 regular file"
                )
            fcntl.flock(fd, fcntl.LOCK_EX)
            path_info = os.lstat(lock_path)
            if (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino):
                raise SealedMonthlyPartitionConflictError("Monthly lock pathname changed")
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_nlink,
        )

    @classmethod
    def _assert_stable_regular_path(
        cls, path: Path, fd: int, *, owned: bool = False
    ) -> os.stat_result:
        info = os.fstat(fd)
        path_info = os.lstat(path)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (owned and info.st_uid != os.getuid())
            or (info.st_dev, info.st_ino) != (path_info.st_dev, path_info.st_ino)
        ):
            raise SealedMonthlyPartitionConflictError(
                f"{path.name} is not a stable owned regular file"
            )
        return info

    @staticmethod
    def _sha256_fd(fd: int) -> str:
        digest = sha256()
        os.lseek(fd, 0, os.SEEK_SET)
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        os.lseek(fd, 0, os.SEEK_SET)
        return digest.hexdigest()

    @staticmethod
    def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

    @staticmethod
    def _validate_sha256(value: str, *, name: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        return value

    @classmethod
    def _read_canonical_json(cls, path: Path) -> dict[str, Any]:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = cls._assert_stable_regular_path(path, fd, owned=True)
            if info.st_size > 64 * 1024:
                raise SealedMonthlyPartitionConflictError(f"{path.name} is too large")
            raw = bytearray()
            while len(raw) < info.st_size:
                chunk = os.read(fd, min(8192, info.st_size - len(raw)))
                if not chunk:
                    raise SealedMonthlyPartitionConflictError(f"{path.name} was truncated")
                raw.extend(chunk)
            if os.read(fd, 1) or cls._file_identity(info) != cls._file_identity(
                cls._assert_stable_regular_path(path, fd, owned=True)
            ):
                raise SealedMonthlyPartitionConflictError(f"{path.name} changed while read")
        finally:
            os.close(fd)
        try:
            payload = json.loads(
                bytes(raw).decode("utf-8"),
                object_pairs_hook=ParquetMarketDataRepository._json_object_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise SealedMonthlyPartitionConflictError(f"{path.name} is malformed") from exc
        if not isinstance(payload, dict) or bytes(raw) != cls._canonical_json_bytes(payload):
            raise SealedMonthlyPartitionConflictError(f"{path.name} is not canonical")
        return payload

    @staticmethod
    def _write_all(fd: int, payload: bytes | memoryview) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(errno.EIO, "short write")
            view = view[written:]

    @classmethod
    def _create_noreplace_file(cls, path: Path, payload: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            cls._write_all(fd, payload)
            os.fsync(fd)
            cls._assert_stable_regular_path(path, fd, owned=True)
        finally:
            os.close(fd)
        cls._fsync_dir(path.parent)

    @classmethod
    def _copy_fd_to_new_file(cls, source_fd: int, destination: Path) -> None:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.lseek(source_fd, 0, os.SEEK_SET)
            while chunk := os.read(source_fd, 1024 * 1024):
                cls._write_all(fd, chunk)
            os.fsync(fd)
            cls._assert_stable_regular_path(destination, fd, owned=True)
        finally:
            os.close(fd)
        cls._fsync_dir(destination.parent)

    @staticmethod
    def _rename_noreplace(source: Path, destination: Path) -> None:
        """Use Linux renameat2 so a concurrent writer can never be overwritten."""
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise OSError(errno.ENOSYS, "renameat2 unavailable") from exc
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)

    def _validate_canonical_monthly_frame(self, frame: pl.DataFrame, *, month: str) -> None:
        if tuple(frame.columns) != tuple(_OHLCV_COLUMNS) or any(
            frame.schema[name] != _DEFAULT_SCHEMA[name] for name in _OHLCV_COLUMNS
        ):
            raise ValueError("Canonical monthly parquet schema is invalid")
        if frame.is_empty():
            raise ValueError("Canonical monthly parquet may not be empty")
        rows = frame.select(_OHLCV_COLUMNS).iter_rows(named=True)
        previous: datetime | None = None
        for row in rows:
            timestamp = row["datetime"]
            values = (row["open"], row["high"], row["low"], row["close"], row["volume"])
            if timestamp is None or any(
                value is None or not math.isfinite(value) for value in values
            ):
                raise ValueError("Canonical monthly parquet contains null or non-finite values")
            if timestamp.tzinfo is not None or timestamp.microsecond % 1_000 != 0:
                raise ValueError("Canonical monthly timestamps must be UTC whole seconds")
            if self._month_token(timestamp) != month or timestamp.microsecond != 0:
                raise ValueError(
                    "Canonical monthly parquet rows must be UTC-second rows in the requested month"
                )
            if previous is not None and timestamp <= previous:
                raise ValueError("Canonical monthly parquet timestamps must be strictly increasing")
            if row["low"] > min(row["open"], row["close"]) or row["high"] < max(
                row["open"], row["close"]
            ):
                raise ValueError("Canonical monthly parquet violates OHLC invariants")
            if row["volume"] < 0:
                raise ValueError("Canonical monthly parquet volume must be non-negative")
            previous = timestamp

    @_generation_guard(exclusive=True)
    def publish_sealed_monthly_partition(
        self,
        *,
        exchange: str,
        symbol: str,
        month: str,
        source: Path,
        expected_sha256: str,
        expected_byte_count: int,
        expected_row_count: int,
        provenance_receipt_sha256: str,
    ) -> Path:
        """Publish one verified staging partition into the canonical immutable monthly store."""
        exchange_token = self._normalize_exchange(exchange)
        symbol_token = self._normalize_symbol_token(symbol)
        month = self._strict_month_token(month)
        expected_sha256 = self._validate_sha256(expected_sha256, name="expected_sha256")
        provenance_receipt_sha256 = self._validate_sha256(
            provenance_receipt_sha256, name="provenance_receipt_sha256"
        )
        if isinstance(expected_byte_count, bool) or expected_byte_count < 0:
            raise ValueError("expected_byte_count must be non-negative")
        if isinstance(expected_row_count, bool) or expected_row_count <= 0:
            raise ValueError("expected_row_count must be positive")
        source = Path(source)
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            source_stat = self._assert_stable_regular_path(source, source_fd)
            source_identity = self._file_identity(source_stat)
            if (
                source_stat.st_size != expected_byte_count
                or self._sha256_fd(source_fd) != expected_sha256
            ):
                raise ValueError("Staging source does not match expected bytes or SHA-256")
            with os.fdopen(os.dup(source_fd), "rb") as source_handle:
                source_frame = pl.read_parquet(source_handle)
            self._validate_canonical_monthly_frame(source_frame, month=month)
            if source_frame.height != expected_row_count:
                raise ValueError("Staging source row count does not match expectation")

            target = self._monthly_path(
                exchange=exchange_token, symbol=symbol_token, month_token=month
            )
            pending = self._monthly_pending_path(
                exchange=exchange_token, symbol=symbol_token, month_token=month
            )
            seal = self._monthly_seal_path(
                exchange=exchange_token, symbol=symbol_token, month_token=month
            )
            relative_path = str(target.relative_to(self.root_path))
            fields = {
                "schema": _CANONICAL_PARTITION_SEAL_SCHEMA,
                "relative_partition_path": relative_path,
                "sha256": expected_sha256,
                "byte_count": expected_byte_count,
                "row_count": expected_row_count,
                "month": month,
                "exchange": exchange_token,
                "symbol": symbol_token,
                "provenance_receipt_sha256": provenance_receipt_sha256,
            }
            pending_payload = {**fields, "status": "pending"}
            seal_payload = {**fields, "status": "sealed"}
            with self._monthly_lock(
                exchange=exchange_token, symbol=symbol_token, month_token=month
            ):
                if (
                    os.path.lexists(pending)
                    and self._read_canonical_json(pending) != pending_payload
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Monthly pending seal conflicts with publication"
                    )
                if os.path.lexists(seal) and self._read_canonical_json(seal) != seal_payload:
                    raise SealedMonthlyPartitionConflictError(
                        "Monthly final seal conflicts with publication"
                    )
                if not os.path.lexists(seal) and not os.path.lexists(pending):
                    self._create_noreplace_file(
                        pending, self._canonical_json_bytes(pending_payload)
                    )
                current_source_stat = self._assert_stable_regular_path(source, source_fd)
                if (
                    self._file_identity(current_source_stat) != source_identity
                    or self._sha256_fd(source_fd) != expected_sha256
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Staging source changed during publication"
                    )
                if os.path.lexists(target):
                    target_fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        target_stat = self._assert_stable_regular_path(target, target_fd)
                        if (
                            target_stat.st_size != expected_byte_count
                            or self._sha256_fd(target_fd) != expected_sha256
                        ):
                            raise SealedMonthlyPartitionConflictError(
                                "Existing monthly target conflicts with publication"
                            )
                        with os.fdopen(os.dup(target_fd), "rb") as target_handle:
                            target_frame = pl.read_parquet(target_handle)
                        self._validate_canonical_monthly_frame(target_frame, month=month)
                        if target_frame.height != expected_row_count:
                            raise SealedMonthlyPartitionConflictError(
                                "Existing monthly target row count conflicts"
                            )
                    finally:
                        os.close(target_fd)
                else:
                    if os.path.lexists(seal):
                        raise SealedMonthlyPartitionConflictError(
                            "Final seal exists without monthly target"
                        )
                    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.publish.tmp")
                    try:
                        self._copy_fd_to_new_file(source_fd, temporary)
                        self._rename_noreplace(temporary, target)
                        self._fsync_dir(target.parent)
                    except Exception as exc:
                        try:
                            if temporary.exists():
                                temp_info = os.lstat(temporary)
                                if not stat.S_ISREG(temp_info.st_mode) or temp_info.st_nlink != 1:
                                    raise SealedMonthlyPartitionConflictError(
                                        "Publication temporary became unsafe"
                                    )
                                temporary.unlink()
                                self._fsync_dir(temporary.parent)
                        except Exception as cleanup_exc:
                            exc.add_note(f"publication temporary cleanup failed: {cleanup_exc!r}")
                        raise
                    target_fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        target_stat = self._assert_stable_regular_path(target, target_fd)
                        if (
                            target_stat.st_size != expected_byte_count
                            or self._sha256_fd(target_fd) != expected_sha256
                        ):
                            raise SealedMonthlyPartitionConflictError(
                                "Published monthly target failed stable hash validation"
                            )
                    finally:
                        os.close(target_fd)
                current_source_stat = self._assert_stable_regular_path(source, source_fd)
                if (
                    self._file_identity(current_source_stat) != source_identity
                    or self._sha256_fd(source_fd) != expected_sha256
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Staging source changed during publication"
                    )
                if not os.path.lexists(seal):
                    self._create_noreplace_file(seal, self._canonical_json_bytes(seal_payload))
                if os.path.lexists(pending):
                    if self._read_canonical_json(pending) != pending_payload:
                        raise SealedMonthlyPartitionConflictError(
                            "Monthly pending seal changed during publication"
                        )
                    pending.unlink()
                    self._fsync_dir(pending.parent)
            return target
        finally:
            os.close(source_fd)

    def _signed_month_conflict_effects(
        self, existing: pl.DataFrame, incoming: pl.DataFrame
    ) -> dict[str, int | str]:
        """Describe the exact signed rows an authorized reconciliation may change."""
        columns = _OHLCV_COLUMNS[1:]
        incoming_renamed = incoming.rename({column: f"{column}_incoming" for column in columns})
        overlap = existing.join(incoming_renamed, on="datetime", how="inner")
        conflict = overlap.filter(
            pl.any_horizontal(
                [pl.col(column) != pl.col(f"{column}_incoming") for column in columns]
            )
        )
        canonical_only = existing.join(incoming.select("datetime"), on="datetime", how="anti")
        source_only = incoming.join(existing.select("datetime"), on="datetime", how="anti")

        def records(frame: pl.DataFrame, *, official: bool = False) -> list[dict[str, Any]]:
            suffix = "_incoming" if official else ""
            rows = (
                frame.select(
                    [
                        pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"),
                        *[pl.col(f"{column}{suffix}").alias(column) for column in columns],
                    ]
                )
                .sort("timestamp_ms")
                .to_dicts()
            )
            result = []
            for row in rows:
                values = [float(row.pop(column)) for column in columns]
                if not all(math.isfinite(value) for value in values):
                    raise SealedMonthlyPartitionConflictError(
                        "OHLCV effect contains non-finite value"
                    )
                result.append({"timestamp_ms": row["timestamp_ms"], "ohlcv": values})
            return result

        conflicts = []
        for canonical, official in zip(records(conflict), records(conflict, official=True)):
            conflicts.append(
                {
                    "timestamp_ms": canonical["timestamp_ms"],
                    "canonical": canonical["ohlcv"],
                    "official": official["ohlcv"],
                }
            )

        def digest(records_value: list[dict[str, Any]]) -> str:
            return sha256(self._canonical_json_bytes(records_value)).hexdigest()

        canonical_only_records = records(canonical_only)
        source_only_records = records(source_only)
        return {
            "conflict_rows": len(conflicts),
            "conflict_sha256": digest(conflicts),
            "canonical_only_rows": len(canonical_only_records),
            "canonical_only_sha256": digest(canonical_only_records),
            "source_only_rows": len(source_only_records),
            "source_only_sha256": digest(source_only_records),
        }

    def _verified_signed_month_authorization(
        self,
        *,
        receipt: Mapping[str, Any],
        public_key: bytes,
        entry: Mapping[str, Any],
        expected_run_id: str,
        expected_approval_sha256: str,
    ) -> dict[str, Any]:
        for value, label in (
            (expected_run_id, "trusted acquisition run ID"),
            (expected_approval_sha256, "trusted approval digest"),
        ):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise SealedMonthlyPartitionConflictError(f"{label} is invalid")
        if not isinstance(public_key, bytes) or len(public_key) != 32:
            raise SealedMonthlyPartitionConflictError("Conflict authority public key is invalid")
        try:
            receipt_value = json.loads(self._canonical_json_bytes(receipt))
            entry_bytes = self._canonical_json_bytes(entry)
        except (TypeError, ValueError) as exc:
            raise SealedMonthlyPartitionConflictError(
                "Conflict authorization is not canonical JSON"
            ) from exc
        if (
            not isinstance(receipt_value, dict)
            or set(receipt_value) != {"schema", "type", "authority_key_id", "message", "signature"}
            or receipt_value.get("schema")
            != "alpha_max_canonical_conflict_authorization_receipt.v2"
            or receipt_value.get("type") != "canonical_conflict_authorization"
            or receipt_value.get("authority_key_id") != sha256(public_key).hexdigest()
            or not isinstance(receipt_value.get("message"), dict)
            or not isinstance(receipt_value.get("signature"), str)
        ):
            raise SealedMonthlyPartitionConflictError("Conflict authorization envelope is invalid")
        message = receipt_value["message"]
        if (
            set(message)
            != {
                "schema",
                "scope",
                "decision",
                "canonical_root",
                "acquisition_request_id",
                "terminal_receipt_sha256",
                "source_manifest_sha256",
                "source_eligible_receipt_sha256",
                "predecessor_path",
                "predecessor_identity",
                "predecessor_inventory_sha256",
                "fresh_acquisition_audit_receipt_sha256",
                "acquisition_run_id",
                "composite_telemetry_sha256",
                "wal_transition_receipt_sha256",
                "wal_post_transition_inventory_sha256",
                "signed_terminal_request_sha256",
                "approval_sha256",
                "entries",
            }
            or message.get("schema") != "alpha_max_canonical_conflict_authorization_message.v2"
            or message.get("scope") != "canonical_conflict_reconciliation"
            or message.get("decision") != "approve_exact_effects"
            or not isinstance(message.get("entries"), list)
            or any(
                not isinstance(message.get(field), str)
                or re.fullmatch(r"[0-9a-f]{64}", message[field]) is None
                for field in (
                    "fresh_acquisition_audit_receipt_sha256",
                    "composite_telemetry_sha256",
                    "wal_transition_receipt_sha256",
                    "wal_post_transition_inventory_sha256",
                    "signed_terminal_request_sha256",
                )
            )
        ):
            raise SealedMonthlyPartitionConflictError("Conflict authorization message is invalid")
        unsigned = {key: value for key, value in receipt_value.items() if key != "signature"}
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                base64.b64decode(receipt_value["signature"], validate=True),
                _CONFLICT_AUTHORIZATION_DOMAIN + self._canonical_json_bytes(unsigned),
            )
        except (TypeError, ValueError, InvalidSignature) as exc:
            raise SealedMonthlyPartitionConflictError(
                "Conflict authorization signature is invalid"
            ) from exc
        if message.get("acquisition_run_id") != expected_run_id:
            raise SealedMonthlyPartitionConflictError(
                "Conflict authorization acquisition run ID does not match trusted context"
            )
        if message.get("approval_sha256") != expected_approval_sha256:
            raise SealedMonthlyPartitionConflictError(
                "Conflict authorization approval digest does not match trusted context"
            )
        matches = [
            candidate
            for candidate in message["entries"]
            if isinstance(candidate, dict) and self._canonical_json_bytes(candidate) == entry_bytes
        ]
        if len(matches) != 1:
            raise SealedMonthlyPartitionConflictError(
                "Conflict authorization entry is not signed exactly once"
            )
        return matches[0]

    def merge_signed_month_into_candidate(
        self,
        *,
        exchange: str,
        symbol: str,
        month: str,
        source: Path,
        expected_sha256: str,
        expected_byte_count: int,
        expected_row_count: int,
        provenance_receipt_sha256: str,
        max_output_bytes: int | None = None,
    ) -> Path:
        """Strictly merge a signed month without permitting differing overlaps."""
        return self._merge_signed_month_into_candidate(
            exchange=exchange,
            symbol=symbol,
            month=month,
            source=source,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
            expected_row_count=expected_row_count,
            provenance_receipt_sha256=provenance_receipt_sha256,
            max_output_bytes=max_output_bytes,
            authorization_entry=None,
        )

    def reconcile_authorized_signed_month_into_candidate(
        self,
        *,
        exchange: str,
        symbol: str,
        month: str,
        source: Path,
        expected_sha256: str,
        expected_byte_count: int,
        expected_row_count: int,
        provenance_receipt_sha256: str,
        conflict_authorization_receipt: Mapping[str, Any],
        conflict_authority_public_key: bytes,
        authorization_entry: Mapping[str, Any],
        expected_run_id: str,
        expected_approval_sha256: str,
        max_output_bytes: int | None = None,
    ) -> Path:
        """Reconcile exact effects from one independently signed authorization entry."""
        verified_entry = self._verified_signed_month_authorization(
            receipt=conflict_authorization_receipt,
            public_key=conflict_authority_public_key,
            entry=authorization_entry,
            expected_run_id=expected_run_id,
            expected_approval_sha256=expected_approval_sha256,
        )
        return self._merge_signed_month_into_candidate(
            exchange=exchange,
            symbol=symbol,
            month=month,
            source=source,
            expected_sha256=expected_sha256,
            expected_byte_count=expected_byte_count,
            expected_row_count=expected_row_count,
            provenance_receipt_sha256=provenance_receipt_sha256,
            max_output_bytes=max_output_bytes,
            authorization_entry=verified_entry,
        )

    @_generation_guard(exclusive=True)
    def _merge_signed_month_into_candidate(
        self,
        *,
        exchange: str,
        symbol: str,
        month: str,
        source: Path,
        expected_sha256: str,
        expected_byte_count: int,
        expected_row_count: int,
        provenance_receipt_sha256: str,
        max_output_bytes: int | None = None,
        authorization_entry: Mapping[str, Any] | None = None,
    ) -> Path:
        """Merge a signed month using only the two monthly dataframes."""
        exchange_token = self._normalize_exchange(exchange)
        symbol_token = self._normalize_symbol_token(symbol)
        month = self._strict_month_token(month)
        expected_sha256 = self._validate_sha256(expected_sha256, name="expected_sha256")
        provenance_receipt_sha256 = self._validate_sha256(
            provenance_receipt_sha256, name="provenance_receipt_sha256"
        )
        if max_output_bytes is not None and (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes <= 0
        ):
            raise ValueError("max_output_bytes must be a positive integer")
        target = self._monthly_path(exchange=exchange_token, symbol=symbol_token, month_token=month)
        start = datetime.strptime(month + "-01", "%Y-%m-%d")
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(milliseconds=1)
        if not self._load_wal_frame(
            exchange=exchange_token, symbol=symbol_token, start_date=start, end_date=end
        ).is_empty():
            raise SealedMonthlyPartitionConflictError(
                f"WAL overlaps managed monthly partition {month}"
            )
        if not target.exists():
            if authorization_entry is not None:
                raise SealedMonthlyPartitionConflictError(
                    "Conflict authorization requires an existing canonical monthly partition"
                )
            return self.publish_sealed_monthly_partition(
                exchange=exchange,
                symbol=symbol,
                month=month,
                source=source,
                expected_sha256=expected_sha256,
                expected_byte_count=expected_byte_count,
                expected_row_count=expected_row_count,
                provenance_receipt_sha256=provenance_receipt_sha256,
            )
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        target_fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
        temporary: Path | None = None
        try:
            source_stat = self._assert_stable_regular_path(source, source_fd)
            source_identity = self._file_identity(source_stat)
            target_stat, target_path_stat = os.fstat(target_fd), os.lstat(target)
            if (
                not stat.S_ISREG(target_stat.st_mode)
                or target_stat.st_nlink != 2
                or (target_stat.st_dev, target_stat.st_ino)
                != (target_path_stat.st_dev, target_path_stat.st_ino)
            ):
                raise SealedMonthlyPartitionConflictError(
                    "Candidate monthly target is not the bound clone hardlink"
                )
            target_identity = self._file_identity(target_stat)
            if (
                source_stat.st_size != expected_byte_count
                or self._sha256_fd(source_fd) != expected_sha256
            ):
                raise ValueError("Staging source does not match expected bytes or SHA-256")
            with os.fdopen(os.dup(source_fd), "rb") as handle:
                incoming = pl.read_parquet(handle)
            with os.fdopen(os.dup(target_fd), "rb") as handle:
                existing = pl.read_parquet(handle)
            self._validate_canonical_monthly_frame(incoming, month=month)
            self._validate_canonical_monthly_frame(existing, month=month)
            if incoming.height != expected_row_count:
                raise ValueError("Staging source row count does not match expectation")
            if authorization_entry is None:
                overlap = existing.join(incoming, on="datetime", how="inner", suffix="_incoming")
                if any(
                    overlap.filter(pl.col(column) != pl.col(f"{column}_incoming")).height
                    for column in _OHLCV_COLUMNS[1:]
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Signed monthly source conflicts with canonical OHLCV values"
                    )
                merged = pl.concat(
                    [
                        existing,
                        incoming.join(existing.select("datetime"), on="datetime", how="anti"),
                    ],
                    how="vertical",
                ).sort("datetime")
            else:
                required_entry_fields = {
                    "relative",
                    "source_sha256",
                    "source_byte_count",
                    "source_row_count",
                    "provenance_receipt_sha256",
                    "predecessor_identity",
                    "predecessor_sha256",
                    "predecessor_byte_count",
                    "predecessor_row_count",
                    "effects",
                }
                required_effect_fields = {
                    "conflict_rows",
                    "conflict_sha256",
                    "canonical_only_rows",
                    "canonical_only_sha256",
                    "source_only_rows",
                    "source_only_sha256",
                }
                signed_identity = authorization_entry.get("predecessor_identity")
                signed_effects = authorization_entry.get("effects")
                if (
                    set(authorization_entry) != required_entry_fields
                    or any(
                        type(authorization_entry[field]) is not int
                        or authorization_entry[field] < 0
                        for field in (
                            "source_byte_count",
                            "source_row_count",
                            "predecessor_byte_count",
                            "predecessor_row_count",
                        )
                    )
                    or authorization_entry["relative"] != str(target.relative_to(self.root_path))
                    or authorization_entry["source_sha256"] != expected_sha256
                    or authorization_entry["source_byte_count"] != expected_byte_count
                    or authorization_entry["source_row_count"] != expected_row_count
                    or authorization_entry["provenance_receipt_sha256"] != provenance_receipt_sha256
                    or not isinstance(signed_identity, list)
                    or len(signed_identity) != 9
                    or any(type(value) is not int for value in signed_identity)
                    or signed_identity[6] != 1
                    or (
                        target_stat.st_dev,
                        target_stat.st_ino,
                        target_stat.st_mode,
                        target_stat.st_size,
                        target_stat.st_mtime_ns,
                        target_stat.st_uid,
                        target_stat.st_gid,
                    )
                    != tuple(signed_identity[index] for index in (0, 1, 2, 3, 4, 7, 8))
                    or target_stat.st_nlink != signed_identity[6] + 1
                    or authorization_entry["predecessor_byte_count"] != target_stat.st_size
                    or authorization_entry["predecessor_row_count"] != existing.height
                    or authorization_entry["predecessor_sha256"] != self._sha256_fd(target_fd)
                    or not isinstance(signed_effects, dict)
                    or set(signed_effects) != required_effect_fields
                    or any(
                        type(signed_effects[field]) is not int or signed_effects[field] < 0
                        for field in (
                            "conflict_rows",
                            "canonical_only_rows",
                            "source_only_rows",
                        )
                    )
                    or signed_effects["conflict_rows"] == 0
                    or any(
                        not isinstance(signed_effects[field], str)
                        or re.fullmatch(r"[0-9a-f]{64}", signed_effects[field]) is None
                        for field in (
                            "conflict_sha256",
                            "canonical_only_sha256",
                            "source_only_sha256",
                        )
                    )
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Signed monthly conflict authorization entry is invalid"
                    )
                effects = self._signed_month_conflict_effects(existing, incoming)
                if signed_effects != effects:
                    raise SealedMonthlyPartitionConflictError(
                        "Signed monthly conflict authorization effects do not match"
                    )
                columns = _OHLCV_COLUMNS[1:]
                incoming_renamed = incoming.rename(
                    {column: f"{column}_incoming" for column in columns}
                )
                conflict_times = (
                    existing.join(incoming_renamed, on="datetime", how="inner")
                    .filter(
                        pl.any_horizontal(
                            [pl.col(column) != pl.col(f"{column}_incoming") for column in columns]
                        )
                    )
                    .select("datetime")
                )
                merged = pl.concat(
                    [
                        existing.join(conflict_times, on="datetime", how="anti"),
                        incoming.join(existing.select("datetime"), on="datetime", how="anti"),
                        incoming.join(conflict_times, on="datetime", how="inner"),
                    ],
                    how="vertical",
                ).sort("datetime")
            self._validate_canonical_monthly_frame(merged, month=month)
            pending = self._monthly_pending_path(
                exchange=exchange_token, symbol=symbol_token, month_token=month
            )
            seal = self._monthly_seal_path(
                exchange=exchange_token, symbol=symbol_token, month_token=month
            )
            fields = {
                "schema": _CANONICAL_PARTITION_SEAL_SCHEMA,
                "relative_partition_path": str(target.relative_to(self.root_path)),
                "month": month,
                "exchange": exchange_token,
                "symbol": symbol_token,
                "provenance_receipt_sha256": provenance_receipt_sha256,
            }
            with self._monthly_lock(
                exchange=exchange_token, symbol=symbol_token, month_token=month
            ):
                if os.path.lexists(pending):
                    raise SealedMonthlyPartitionConflictError(
                        "Monthly pending seal conflicts with candidate merge"
                    )
                if os.path.lexists(seal):
                    seal_info = os.lstat(seal)
                    if not stat.S_ISREG(seal_info.st_mode) or seal_info.st_nlink != 2:
                        raise SealedMonthlyPartitionConflictError(
                            "Candidate final seal is not the bound clone hardlink"
                        )
                    seal.unlink()
                    self._fsync_dir(seal.parent)
                self._create_noreplace_file(
                    pending, self._canonical_json_bytes({**fields, "status": "pending"})
                )
                if (
                    self._file_identity(self._assert_stable_regular_path(source, source_fd))
                    != source_identity
                    or self._file_identity(os.fstat(target_fd)) != target_identity
                    or (os.lstat(target).st_dev, os.lstat(target).st_ino) != target_identity[:2]
                ):
                    raise SealedMonthlyPartitionConflictError(
                        "Candidate merge source or target changed"
                    )
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.candidate.tmp")
                if max_output_bytes is None:
                    merged.write_parquet(temporary, compression="zstd", statistics=True)
                else:
                    buffer = io.BytesIO()
                    merged.write_parquet(buffer, compression="zstd", statistics=True)
                    payload = buffer.getvalue()
                    if len(payload) > max_output_bytes:
                        raise SealedMonthlyPartitionConflictError(
                            "Merged monthly parquet exceeds publication quota"
                        )
                    temporary_fd = os.open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                    )
                    try:
                        self._write_all(temporary_fd, payload)
                        os.fsync(temporary_fd)
                    finally:
                        os.close(temporary_fd)
                self._fsync_file(temporary)
                os.replace(temporary, target)
                temporary = None
                self._fsync_dir(target.parent)
                result_fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    result_stat = self._assert_stable_regular_path(target, result_fd)
                    result_sha = self._sha256_fd(result_fd)
                finally:
                    os.close(result_fd)
                self._create_noreplace_file(
                    seal,
                    self._canonical_json_bytes(
                        {
                            **fields,
                            "status": "sealed",
                            "sha256": result_sha,
                            "byte_count": result_stat.st_size,
                            "row_count": merged.height,
                        }
                    ),
                )
                pending.unlink()
                self._fsync_dir(pending.parent)
            return target
        finally:
            if temporary is not None and os.path.lexists(temporary):
                info = os.lstat(temporary)
                if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    temporary.unlink()
            os.close(target_fd)
            os.close(source_fd)

    def _meta_path(self, *, exchange: str, symbol: str) -> Path:
        return self._symbol_root(exchange=exchange, symbol=symbol) / "compaction.meta.json"

    def _raw_symbol_root(self, *, exchange: str, symbol: str) -> Path:
        root = self.root_path / "market_data_raw_aggtrades"
        path = (
            root
            / self._normalize_raw_exchange_token(exchange)
            / self._normalize_raw_symbol_token(symbol)
        )
        self._assert_raw_path_confined(path)
        return path

    def raw_partition_path(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_date: str | date,
    ) -> Path:
        return (
            self._raw_partition_path(
                exchange=exchange,
                symbol=symbol,
                partition_date=partition_date,
            )
            / "part-0000.parquet"
        )

    def raw_checkpoint_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_checkpoint_path(exchange=exchange, symbol=symbol)

    def raw_wal_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_wal_path(exchange=exchange, symbol=symbol)

    def _materialized_symbol_root(self, *, exchange: str, symbol: str) -> Path:
        return (
            self.root_path
            / "market_data_materialized"
            / self._normalize_exchange(exchange)
            / self._normalize_symbol_token(symbol)
        )

    def materialized_partition_root(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str | date,
    ) -> Path:
        return self._materialized_date_root(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            partition_date=partition_date,
        )

    def materialized_manifest_path(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str | date,
    ) -> Path:
        return self._materialized_manifest_path(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            partition_date=partition_date,
        )

    @staticmethod
    def _date_token_from_ms(ts_ms: int) -> str:
        return datetime.fromtimestamp(float(int(ts_ms)) / 1000.0, tz=UTC).strftime("%Y-%m-%d")

    @staticmethod
    def _format_checksum_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return format(float(value), ".12g")
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.astimezone(UTC).isoformat()
        return str(value)

    @classmethod
    def canonical_row_checksum(cls, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return sha256(b"").hexdigest()
        required = [column for column in _OHLCV_COLUMNS if column in frame.columns]
        if not required:
            return sha256(b"").hexdigest()
        ordered = frame.select(required).sort("datetime")
        digest = sha256()
        for row in ordered.iter_rows(named=True):
            line = "|".join(cls._format_checksum_value(row.get(col)) for col in required)
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def _validate_manifest_payload(
        self,
        *,
        manifest: dict[str, Any],
        exchange: str,
        symbol: str,
        timeframe: str,
        manifest_path: Path,
    ) -> dict[str, Any]:
        missing = [
            field for field in _MANIFEST_REQUIRED_FIELDS if manifest.get(field) in {None, ""}
        ]
        if missing:
            raise RawFirstManifestInvalidError(
                f"Manifest missing required fields {missing} at {manifest_path}."
            )

        if str(manifest.get("status", "")).strip().lower() != "committed":
            raise RawFirstDataMissingError(f"Manifest status is not committed at {manifest_path}.")

        expected_exchange = self._normalize_exchange(exchange)
        expected_symbol = self._normalize_symbol_token(symbol)
        if self._normalize_symbol_token(str(manifest.get("symbol", ""))) != expected_symbol:
            raise RawFirstManifestInvalidError(
                f"Manifest symbol mismatch at {manifest_path}: expected {symbol}."
            )
        if normalize_timeframe_token(
            str(manifest.get("timeframe", ""))
        ) != normalize_timeframe_token(timeframe):
            raise RawFirstManifestInvalidError(
                f"Manifest timeframe mismatch at {manifest_path}: expected {timeframe}."
            )
        partition = str(manifest.get("partition", "")).strip()
        if (
            partition
            and f"/{expected_exchange}/" not in partition
            and f"\\{expected_exchange}\\" not in partition
            and expected_exchange not in partition
        ):
            # Best-effort exchange sanity check while keeping backward compatibility.
            raise RawFirstManifestInvalidError(
                f"Manifest partition exchange mismatch at {manifest_path}: expected {expected_exchange}."
            )
        try:
            int(manifest.get("row_count", 0))
            int(manifest.get("window_start_ms", 0))
            int(manifest.get("window_end_ms", 0))
            int(manifest.get("event_time_watermark_ms", 0))
            int(manifest.get("source_checkpoint_start", 0))
            int(manifest.get("source_checkpoint_end", 0))
        except Exception as exc:
            raise RawFirstManifestInvalidError(
                f"Manifest numeric fields invalid at {manifest_path}: {exc}"
            ) from exc

        data_files = manifest.get("data_files")
        if not isinstance(data_files, list) or not data_files:
            raise RawFirstManifestInvalidError(
                f"Manifest data_files must be a non-empty list at {manifest_path}."
            )
        for item in data_files:
            token = str(item or "").strip()
            if not token:
                raise RawFirstManifestInvalidError(
                    f"Manifest data_files contains empty entry at {manifest_path}."
                )

        return manifest

    @staticmethod
    def _normalize_ohlcv_frame(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return ParquetMarketDataRepository._empty_ohlcv_frame()
        missing = [column for column in _OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise RawFirstManifestInvalidError(
                f"Materialized frame missing OHLCV columns: {missing}"
            )
        return (
            frame.select(_OHLCV_COLUMNS)
            .with_columns(
                [
                    ParquetMarketDataRepository._coerce_datetime_expr(pl.col("datetime")).alias(
                        "datetime"
                    ),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )
            .drop_nulls(subset=["datetime"])
            .sort("datetime")
        )

    def _raw_partition_path(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_date: str | date,
    ) -> Path:
        return (
            self._raw_symbol_root(
                exchange=exchange,
                symbol=symbol,
            )
            / f"date={self._partition_date_token(partition_date)}"
        )

    def _raw_wal_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_symbol_root(exchange=exchange, symbol=symbol) / _RAW_WAL_NAME

    def _raw_checkpoint_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_symbol_root(exchange=exchange, symbol=symbol) / "checkpoint.json"

    def _raw_meta_path(self, *, exchange: str, symbol: str) -> Path:
        return self._raw_symbol_root(exchange=exchange, symbol=symbol) / _RAW_META_NAME

    def _raw_components(self, path: Path) -> tuple[str, ...]:
        root = self.root_path.absolute()
        try:
            return path.absolute().relative_to(root).parts
        except ValueError as exc:
            raise ValueError("Raw aggTrades path escapes its repository root") from exc

    @staticmethod
    def _raw_checked_dir_fd(fd: int, *, path: Path) -> os.stat_result:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Raw aggTrades path is not a directory: {path}")
        return info

    def _raw_dir_fd(self, path: Path, *, create: bool = False) -> int:
        """Open a managed raw directory from filesystem-root anchored descriptors."""
        root = self.root_path.absolute()
        target_components = self._raw_components(path)
        root_components = root.parts[1:]
        if root.anchor != os.path.sep:
            raise ValueError("Raw aggTrades repository root must be absolute")

        fd = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        current = Path(root.anchor)
        try:
            self._raw_checked_dir_fd(fd, path=current)
            for component in (*root_components, *target_components):
                if component in {"", ".", ".."} or "/" in component:
                    raise ValueError("Raw aggTrades path has an unsafe component")
                current /= component
                try:
                    child = os.open(
                        component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    else:
                        os.fsync(fd)
                    child = os.open(
                        component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                    )
                try:
                    self._raw_checked_dir_fd(child, path=current)
                except Exception:
                    os.close(child)
                    raise
                os.close(fd)
                fd = child
            return fd
        except Exception:
            os.close(fd)
            raise

    def _assert_raw_path_confined(self, path: Path) -> None:
        """Validate syntax only; raw I/O must use `_raw_dir_fd` below."""
        for component in self._raw_components(path):
            if component in {".", ".."} or "/" in component:
                raise ValueError("Raw aggTrades path has an unsafe component")

    @staticmethod
    def _raw_checked_fd(fd: int, *, path: Path) -> os.stat_result:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Raw aggTrades file is not an owned regular file: {path}")
        return info

    def _raw_open_regular(
        self, parent_fd: int, name: str, flags: int, *, path: Path, mode: int = 0o600
    ) -> int:
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError("Raw aggTrades file name is unsafe")
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode, dir_fd=parent_fd)
        try:
            self._raw_checked_fd(fd, path=path)
            return fd
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _raw_write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            try:
                written = os.write(fd, view)
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError("Raw aggTrades control write made no progress")
            view = view[written:]

    @staticmethod
    def _raw_read_exact(fd: int, size: int, *, error: str) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            try:
                chunk = os.read(fd, remaining)
            except InterruptedError:
                continue
            if not chunk:
                raise ValueError(error)
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _raw_read_bytes(
        self, parent_fd: int, name: str, *, path: Path, max_bytes: int = _RAW_CONTROL_MAX_BYTES
    ) -> tuple[bytes, os.stat_result]:
        fd = self._raw_open_regular(parent_fd, name, os.O_RDONLY, path=path)
        try:
            before = self._raw_checked_fd(fd, path=path)
            if before.st_size > max_bytes:
                raise ValueError(f"Raw aggTrades control file is too large: {path}")
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                try:
                    chunk = os.read(fd, min(remaining, 1_048_576))
                except InterruptedError:
                    continue
                if not chunk:
                    raise ValueError(f"Raw aggTrades control file was truncated: {path}")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(fd, 1):
                raise ValueError(f"Raw aggTrades control file grew while in use: {path}")
            after = self._raw_checked_fd(fd, path=path)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError(f"Raw aggTrades file changed while in use: {path}")
            return b"".join(chunks), before
        finally:
            os.close(fd)

    def _raw_read_parquet(self, path: Path) -> pl.DataFrame:
        parent_fd = self._raw_dir_fd(path.parent, create=False)
        try:
            fd = self._raw_open_regular(parent_fd, path.name, os.O_RDONLY, path=path)
            try:
                before = self._raw_checked_fd(fd, path=path)
                frame = pl.read_parquet(f"/proc/self/fd/{fd}")
                after = self._raw_checked_fd(fd, path=path)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError(f"Raw aggTrades file changed while in use: {path}")
                return frame
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)

    def _raw_read_authenticated_parquet(self, path: Path, entry: Mapping[str, Any]) -> pl.DataFrame:
        parent_fd = self._raw_dir_fd(path.parent, create=False)
        try:
            fd = self._raw_open_regular(parent_fd, path.name, os.O_RDONLY, path=path)
            try:
                before = self._raw_checked_fd(fd, path=path)
                identity = entry["file_identity"]
                if (
                    before.st_dev != identity["dev"]
                    or before.st_ino != identity["ino"]
                    or before.st_size != identity["size"]
                    or before.st_mtime_ns != identity["mtime_ns"]
                    or before.st_size != entry["byte_count"]
                ):
                    raise ValueError(
                        f"Raw aggTrades part identity does not match inventory: {path}"
                    )
                digest = sha256()
                size = 0
                while chunk := os.read(fd, 1_048_576):
                    size += len(chunk)
                    digest.update(chunk)
                if size != entry["byte_count"] or digest.hexdigest() != entry["content_sha256"]:
                    raise ValueError(f"Raw aggTrades part bytes do not match inventory: {path}")
                os.lseek(fd, 0, os.SEEK_SET)
                frame = pl.read_parquet(f"/proc/self/fd/{fd}")
                frame = self._validate_raw_aggtrades_frame(frame)
                rows = frame.to_dicts()
                if (
                    frame.height != entry["row_count"]
                    or int(rows[0]["agg_trade_id"]) != entry["min_trade_id"]
                    or int(rows[-1]["agg_trade_id"]) != entry["max_trade_id"]
                    or int(rows[0]["timestamp_ms"]) != entry["min_timestamp_ms"]
                    or int(rows[-1]["timestamp_ms"]) != entry["max_timestamp_ms"]
                    or any(
                        self._partition_date_from_ms(row["timestamp_ms"]) != entry["date"]
                        for row in rows
                    )
                ):
                    raise ValueError(
                        f"Raw aggTrades part metadata does not match inventory: {path}"
                    )
                after = self._raw_checked_fd(fd, path=path)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError(f"Raw aggTrades file changed while in use: {path}")
                return frame
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)

    def _raw_write_parquet(
        self, parent_fd: int, name: str, *, path: Path, frame: pl.DataFrame
    ) -> None:
        fd = self._raw_open_regular(parent_fd, name, os.O_WRONLY, path=path)
        try:
            frame.write_parquet(f"/proc/self/fd/{fd}", compression="zstd", statistics=True)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _raw_replace(self, parent_fd: int, source: str, destination: str) -> None:
        os.replace(source, destination, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)

    def _raw_unlink(self, parent_fd: int, name: str) -> None:
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)

    def _ensure_raw_directory(self, path: Path) -> None:
        fd = self._raw_dir_fd(path, create=True)
        os.close(fd)

    def _raw_regular_stat(self, path: Path) -> os.stat_result:
        parent_fd = self._raw_dir_fd(path.parent, create=False)
        try:
            fd = self._raw_open_regular(parent_fd, path.name, os.O_RDONLY, path=path)
            try:
                return self._raw_checked_fd(fd, path=path)
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)

    def _assert_raw_regular_stable(self, path: Path, before: os.stat_result) -> None:
        after = self._raw_regular_stat(path)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"Raw aggTrades file changed while in use: {path}")

    @staticmethod
    def _raw_dir_names(fd: int) -> list[str]:
        return sorted(os.listdir(fd))

    def _raw_part_paths(self, partition_root: Path) -> list[Path]:
        fd = self._raw_dir_fd(partition_root, create=False)
        try:
            return [
                partition_root / name
                for name in self._raw_dir_names(fd)
                if _RAW_PART_PATTERN.fullmatch(name) is not None
            ]
        finally:
            os.close(fd)

    def _next_raw_part_path(self, partition_root: Path) -> Path:
        indices = [
            int(match.group(1))
            for path in self._raw_part_paths(partition_root)
            if (match := _RAW_PART_PATTERN.fullmatch(path.name)) is not None
        ]
        return partition_root / f"part-{max(indices, default=-1) + 1:04d}.parquet"

    def _materialized_date_root(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str | date,
    ) -> Path:
        token = normalize_timeframe_token(timeframe)
        return (
            self._materialized_symbol_root(exchange=exchange, symbol=symbol)
            / f"timeframe={token}"
            / f"date={self._partition_date_token(partition_date)}"
        )

    def _materialized_manifest_path(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str | date,
    ) -> Path:
        return (
            self._materialized_date_root(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                partition_date=partition_date,
            )
            / "manifest.json"
        )

    def _materialized_commit_dir(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str,
        commit_id: str,
    ) -> Path:
        return (
            self._materialized_date_root(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                partition_date=partition_date,
            )
            / f"commit={commit_id}"
        )

    @staticmethod
    def _coerce_datetime_expr(expr: pl.Expr) -> pl.Expr:
        parsed = expr.cast(pl.Utf8, strict=False).str.to_datetime(
            strict=False,
            time_zone="UTC",
        )
        as_datetime = expr.cast(pl.Datetime(time_zone="UTC"), strict=False)
        return (
            pl.coalesce([as_datetime, parsed])
            .dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None)
            .cast(pl.Datetime(time_unit="ms"))
        )

    @staticmethod
    def _ensure_ohlcv_frame(
        rows: pl.DataFrame | list[dict[str, Any]] | list[tuple[Any, ...]],
    ) -> pl.DataFrame:
        if isinstance(rows, pl.DataFrame):
            frame = rows
        else:
            frame = pl.DataFrame(rows)

        if frame.is_empty():
            return ParquetMarketDataRepository._empty_ohlcv_frame()

        required = ["datetime", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"OHLCV rows missing columns: {missing}")

        casted = frame.select(required).with_columns(
            [
                ParquetMarketDataRepository._coerce_datetime_expr(pl.col("datetime")).alias(
                    "datetime"
                ),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        return casted.drop_nulls(subset=["datetime"]).sort("datetime")

    @staticmethod
    def _collect_lazy(lazy_frame: pl.LazyFrame) -> pl.DataFrame:
        """Collect helper that prefers compute_engine adapter when available."""
        try:
            from lumina_quant.compute_engine import resolve_compute_engine

            engine = resolve_compute_engine()
            collect = getattr(engine, "collect", None)
            if callable(collect):
                return collect(lazy_frame)
        except Exception:
            pass
        return lazy_frame.collect(engine="streaming")

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as fh:
            os.fsync(fh.fileno())

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _partition_date_from_ms(timestamp_ms: int) -> str:
        dt = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _canonical_materialized_checksum(frame: pl.DataFrame) -> str:
        return ParquetMarketDataRepository.canonical_row_checksum(frame)

    @staticmethod
    def _empty_raw_aggtrades_frame() -> pl.DataFrame:
        return pl.DataFrame(
            {name: [] for name in _RAW_AGGTRADES_SCHEMA},
            schema=_RAW_AGGTRADES_SCHEMA,
        )

    @staticmethod
    def _validate_raw_aggtrades_frame(frame: pl.DataFrame) -> pl.DataFrame:
        """Reject malformed raw data instead of coercing, sorting, or overwriting it."""
        if set(frame.columns) != set(_RAW_AGGTRADES_REQUIRED_COLUMNS):
            missing = [
                name for name in _RAW_AGGTRADES_REQUIRED_COLUMNS if name not in frame.columns
            ]
            unexpected = [
                name for name in frame.columns if name not in _RAW_AGGTRADES_REQUIRED_COLUMNS
            ]
            raise ValueError(
                f"Raw aggTrades schema mismatch missing={missing} unexpected={unexpected}"
            )
        if any(frame.schema[name] != dtype for name, dtype in _RAW_AGGTRADES_SCHEMA.items()):
            raise ValueError("Raw aggTrades storage types are invalid")
        if frame.is_empty():
            return ParquetMarketDataRepository._empty_raw_aggtrades_frame()
        rows: list[dict[str, Any]] = []
        previous_timestamp: int | None = None
        previous_id: int | None = None
        previous_row: dict[str, Any] | None = None
        for row in frame.select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS)).iter_rows(named=True):
            if any(value is None for value in row.values()):
                raise ValueError("Raw aggTrades rows may not contain nulls")
            trade_id = row["agg_trade_id"]
            timestamp_ms = row["timestamp_ms"]
            price = row["price"]
            quantity = row["quantity"]
            buyer_maker = row["is_buyer_maker"]
            if type(trade_id) is not int or trade_id < 0:
                raise ValueError("Raw aggTrades aggregate IDs must be nonnegative integers")
            if type(timestamp_ms) is not int or timestamp_ms <= 0:
                raise ValueError("Raw aggTrades timestamps must be positive integers")
            if type(price) is not float or not math.isfinite(price) or price <= 0:
                raise ValueError("Raw aggTrades prices must be finite positive floats")
            if type(quantity) is not float or not math.isfinite(quantity) or quantity <= 0:
                raise ValueError("Raw aggTrades quantities must be finite positive floats")
            if type(buyer_maker) is not bool:
                raise ValueError("Raw aggTrades buyer-maker flags must be booleans")
            if previous_id is not None and trade_id == previous_id:
                if previous_row != row:
                    raise ValueError("Raw aggTrades duplicate aggregate ID conflicts")
                continue
            if previous_timestamp is not None and timestamp_ms < previous_timestamp:
                raise ValueError("Raw aggTrades timestamps must be nondecreasing")
            if previous_id is not None and trade_id <= previous_id:
                raise ValueError("Raw aggTrades aggregate IDs must be strictly increasing")
            rows.append(row)
            previous_timestamp = timestamp_ms
            previous_id = trade_id
            previous_row = row
        return pl.DataFrame(rows, schema=_RAW_AGGTRADES_SCHEMA)

    @staticmethod
    def _normalize_loaded_raw_aggtrades_frame(frame: pl.DataFrame) -> pl.DataFrame:
        return ParquetMarketDataRepository._validate_raw_aggtrades_frame(frame)

    @staticmethod
    def _ensure_raw_aggtrades_frame(
        rows: pl.DataFrame | list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> pl.DataFrame:
        if isinstance(rows, pl.DataFrame):
            frame = rows
        elif not rows:
            return ParquetMarketDataRepository._empty_raw_aggtrades_frame()
        else:
            frame = pl.DataFrame(rows)
        return ParquetMarketDataRepository._validate_raw_aggtrades_frame(frame)

    @staticmethod
    def _checkpoint_digest(row: Mapping[str, Any]) -> str:
        return sha256(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def validate_raw_checkpoint(
        self, payload: Mapping[str, Any], *, exchange: str, symbol: str
    ) -> dict[str, Any]:
        if set(payload) != _RAW_CHECKPOINT_FIELDS:
            raise ValueError("Raw aggTrades checkpoint schema is invalid")

        def integer(value: Any, field: str, minimum: int) -> int:
            if type(value) is not int or value < minimum:
                raise ValueError(f"Raw aggTrades checkpoint {field} is invalid")
            return value

        if type(payload["exchange"]) is not str or payload["exchange"] != exchange:
            raise ValueError("Raw aggTrades checkpoint exchange does not match stream")
        if type(payload["symbol"]) is not str or payload["symbol"] != symbol:
            raise ValueError("Raw aggTrades checkpoint symbol does not match stream")
        timestamp = integer(payload["last_timestamp_ms"], "last timestamp", 1)
        trade_id = integer(payload["last_trade_id"], "last aggregate ID", 0)
        observed = integer(payload["observed_until_ms"], "observed-until timestamp", timestamp)
        batch_rows = integer(payload["batch_rows"], "batch row count", 1)
        updated = payload["updated_at_utc"]
        try:
            parsed = (
                datetime.fromisoformat(updated.replace("Z", "+00:00"))
                if type(updated) is str and updated
                else None
            )
        except ValueError:
            parsed = None
        if parsed is None or parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError("Raw aggTrades checkpoint update time must be UTC")
        if not isinstance(payload["last_row"], Mapping):
            raise ValueError("Raw aggTrades checkpoint last row is invalid")
        frame = self._ensure_raw_aggtrades_frame([dict(payload["last_row"])])
        if frame.height != 1 or frame.to_dicts()[0] != dict(payload["last_row"]):
            raise ValueError("Raw aggTrades checkpoint last row types are invalid")
        row = frame.to_dicts()[0]
        digest = payload["last_row_sha256"]
        if row["timestamp_ms"] != timestamp or row["agg_trade_id"] != trade_id:
            raise ValueError("Raw aggTrades checkpoint last row does not match cursor")
        if (
            type(digest) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest != self._checkpoint_digest(row)
        ):
            raise ValueError("Raw aggTrades checkpoint last row binding is invalid")
        return {
            "exchange": exchange,
            "symbol": symbol,
            "last_timestamp_ms": timestamp,
            "last_trade_id": trade_id,
            "observed_until_ms": observed,
            "updated_at_utc": updated,
            "batch_rows": batch_rows,
            "last_row": row,
            "last_row_sha256": digest,
        }

    @staticmethod
    def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def _read_raw_checkpoint_unlocked(self, *, exchange: str, symbol: str) -> dict[str, Any]:
        root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
        try:
            root_fd = self._raw_dir_fd(root, create=False)
        except FileNotFoundError:
            return {}
        try:
            if "checkpoint.json" not in self._raw_dir_names(root_fd):
                return {}
            raw_bytes, _ = self._raw_read_bytes(
                root_fd, "checkpoint.json", path=root / "checkpoint.json"
            )
            raw = raw_bytes.decode("utf-8")
            payload = json.loads(
                raw,
                object_pairs_hook=self._json_object_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            if not isinstance(payload, dict):
                raise ValueError("checkpoint is not an object")
            checkpoint = self.validate_raw_checkpoint(payload, exchange=exchange, symbol=symbol)
            if raw != self._raw_canonical_json(checkpoint):
                raise ValueError("checkpoint is not canonical")
            return checkpoint
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Raw aggTrades checkpoint is malformed") from exc
        finally:
            os.close(root_fd)

    def read_raw_checkpoint(
        self, *, exchange: str, symbol: str, lease: _RawStreamLease | None = None
    ) -> dict[str, Any]:
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            self.recover_raw_stream(exchange=exchange, symbol=symbol, lease=active)
            self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
            checkpoint = self._read_raw_checkpoint_unlocked(exchange=exchange, symbol=symbol)
            if checkpoint:
                self.read_raw_recovery_bounds(
                    exchange=exchange,
                    symbol=symbol,
                    checkpoint_last_row=checkpoint["last_row"],
                    lease=active,
                )
            return checkpoint
        finally:
            if owns:
                active.release()

    def write_raw_checkpoint(
        self,
        *,
        exchange: str,
        symbol: str,
        payload: dict[str, Any],
        lease: _RawStreamLease | None = None,
    ) -> None:
        checkpoint = self.validate_raw_checkpoint(payload, exchange=exchange, symbol=symbol)
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            self.recover_raw_stream(exchange=exchange, symbol=symbol, lease=active)
            self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
            existing = self._read_raw_checkpoint_unlocked(exchange=exchange, symbol=symbol)
            if existing:
                old = existing["last_row"]
                new = checkpoint["last_row"]
                old_updated = datetime.fromisoformat(
                    existing["updated_at_utc"].replace("Z", "+00:00")
                )
                new_updated = datetime.fromisoformat(
                    checkpoint["updated_at_utc"].replace("Z", "+00:00")
                )
                if (
                    int(new["agg_trade_id"]) < int(old["agg_trade_id"])
                    or int(new["timestamp_ms"]) < int(old["timestamp_ms"])
                    or int(checkpoint["observed_until_ms"]) < int(existing["observed_until_ms"])
                    or new_updated < old_updated
                    or (int(new["agg_trade_id"]) == int(old["agg_trade_id"]) and new != old)
                ):
                    raise ValueError("Raw aggTrades checkpoint cursor regresses")
            if self.read_raw_recovery_bounds(
                exchange=exchange,
                symbol=symbol,
                checkpoint_last_row=checkpoint["last_row"],
                lease=active,
            ).is_empty():
                raise ValueError("Raw aggTrades checkpoint is not bound to persisted raw parquet")
            path = self._raw_checkpoint_path(exchange=exchange, symbol=symbol)
            encoded = self._raw_control_bytes(checkpoint, label="checkpoint")
            parent_fd = self._raw_dir_fd(path.parent, create=True)
            tmp = f".raw-checkpoint-{uuid.uuid4().hex}.tmp"
            try:
                fd = self._raw_open_regular(
                    parent_fd, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, path=path.parent / tmp
                )
                try:
                    self._raw_write_all(fd, encoded)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                self._raw_replace(parent_fd, tmp, path.name)
            except Exception:
                try:
                    if tmp in set(self._raw_dir_names(parent_fd)):
                        self._raw_unlink(parent_fd, tmp)
                except OSError, ValueError:
                    pass
                raise
            finally:
                os.close(parent_fd)
        finally:
            if owns:
                active.release()

    def _raw_wal_tail(self, *, root: Path, root_fd: int) -> dict[str, Any]:
        raw, _ = self._raw_read_bytes(root_fd, _RAW_WAL_TAIL_NAME, path=root / _RAW_WAL_TAIL_NAME)
        try:
            payload = json.loads(
                raw.decode(),
                object_pairs_hook=self._json_object_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            body = {key: value for key, value in payload.items() if key != "tail_sha256"}
            required = {
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
            if (
                not isinstance(payload, dict)
                or set(payload) != required
                or raw.decode() != self._raw_canonical_json(payload)
                or type(payload["version"]) is not int
                or payload["version"] != _RAW_WAL_TAIL_VERSION
                or payload["tail_sha256"]
                != sha256(self._raw_canonical_json(body).encode()).hexdigest()
                or not isinstance(payload["wal_identity"], dict)
                or set(payload["wal_identity"]) != {"dev", "ino", "size", "mtime_ns", "ctime_ns"}
                or any(type(value) is not int for value in payload["wal_identity"].values())
                or any(
                    type(payload[key]) is not int or payload[key] < 0
                    for key in ("last_offset", "last_length", "record_count", "generation")
                )
                or not 0 < payload["last_length"] <= _RAW_WAL_MAX_RECORD_BYTES
                or payload["last_offset"] + payload["last_length"]
                != payload["wal_identity"]["size"]
                or payload["record_count"] < 1
                or any(
                    type(payload[key]) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None
                    for key in ("last_sha256", "tail_sha256")
                )
                or (
                    payload["previous_tail_sha256"] is not None
                    and (
                        type(payload["previous_tail_sha256"]) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", payload["previous_tail_sha256"]) is None
                    )
                )
            ):
                raise ValueError("Raw aggTrades WAL tail schema is invalid")
            return payload
        except (UnicodeDecodeError, TypeError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Raw aggTrades WAL tail is malformed") from exc

    @staticmethod
    def _raw_wal_identity(info: os.stat_result) -> dict[str, int]:
        return {
            "dev": info.st_dev,
            "ino": info.st_ino,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }

    def _write_raw_wal_tail(self, *, root: Path, payload: Mapping[str, Any]) -> None:
        encoded = self._raw_control_bytes(payload, label="WAL tail")
        root_fd = self._raw_dir_fd(root, create=True)
        tmp = f".raw-wal-tail-{uuid.uuid4().hex}.tmp"
        try:
            fd = self._raw_open_regular(
                root_fd, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, path=root / tmp
            )
            try:
                self._raw_write_all(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._raw_replace(root_fd, tmp, _RAW_WAL_TAIL_NAME)
        except Exception:
            try:
                if tmp in set(self._raw_dir_names(root_fd)):
                    self._raw_unlink(root_fd, tmp)
            except OSError, ValueError:
                pass
            raise
        finally:
            os.close(root_fd)

    def _raw_wal_bootstrap(self, *, root: Path, root_fd: int) -> dict[str, Any]:
        raw, _ = self._raw_read_bytes(
            root_fd, _RAW_WAL_BOOTSTRAP_NAME, path=root / _RAW_WAL_BOOTSTRAP_NAME
        )
        try:
            payload = json.loads(
                raw.decode(),
                object_pairs_hook=self._json_object_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            if not isinstance(payload, dict) or raw != self._raw_control_bytes(
                payload, label="WAL bootstrap"
            ):
                raise ValueError("Raw aggTrades WAL bootstrap schema is invalid")
            body = {key: value for key, value in payload.items() if key != "bootstrap_sha256"}
            version = payload.get("version")
            common = (
                type(payload.get("record_hex")) is str
                and type(payload.get("record_sha256")) is str
                and type(payload.get("bootstrap_sha256")) is str
                and re.fullmatch(r"[0-9a-f]{64}", payload["record_sha256"]) is not None
                and re.fullmatch(r"[0-9a-f]{64}", payload["bootstrap_sha256"]) is not None
                and payload["bootstrap_sha256"]
                == sha256(self._raw_canonical_json(body).encode()).hexdigest()
            )
            if type(version) is not int or not common:
                raise ValueError("Raw aggTrades WAL bootstrap schema is invalid")
            record = bytes.fromhex(payload["record_hex"])
            if (
                len(record) > _RAW_WAL_MAX_RECORD_BYTES
                or sha256(record).hexdigest() != payload["record_sha256"]
            ):
                raise ValueError("Raw aggTrades WAL bootstrap record is invalid")
            marker = self._raw_wal_record(record, require_v2=True)
            if version == 1:
                if (
                    set(payload) != {"version", "record_hex", "record_sha256", "bootstrap_sha256"}
                    or marker["sequence"] != 1
                    or marker["previous_record_sha256"] is not None
                ):
                    raise ValueError("Raw aggTrades WAL bootstrap record is invalid")
                return payload
            if version != 2 or set(payload) != {
                "version",
                "kind",
                "valid_size",
                "record_count",
                "legacy_sha256",
                "record_hex",
                "record_sha256",
                "bootstrap_sha256",
            }:
                raise ValueError("Raw aggTrades WAL bootstrap schema is invalid")
            if (
                payload["kind"] != "legacy_migration"
                or type(payload["valid_size"]) is not int
                or payload["valid_size"] < 0
                or type(payload["record_count"]) is not int
                or payload["record_count"] < 0
                or type(payload["legacy_sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", payload["legacy_sha256"]) is None
                or marker["sequence"] != payload["record_count"] + 1
                or marker["previous_record_sha256"]
                != (payload["legacy_sha256"] if payload["record_count"] else None)
            ):
                raise ValueError("Raw aggTrades WAL bootstrap migration is invalid")
            return payload
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Raw aggTrades WAL bootstrap is malformed") from exc

    def _write_raw_wal_bootstrap(self, *, root: Path, record: bytes) -> None:
        body = {
            "version": 1,
            "record_hex": record.hex(),
            "record_sha256": sha256(record).hexdigest(),
        }
        payload = {
            **body,
            "bootstrap_sha256": sha256(self._raw_canonical_json(body).encode()).hexdigest(),
        }
        encoded = self._raw_control_bytes(payload, label="WAL bootstrap")
        root_fd = self._raw_dir_fd(root, create=True)
        tmp = f".raw-wal-bootstrap-{uuid.uuid4().hex}.tmp"
        try:
            fd = self._raw_open_regular(
                root_fd, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, path=root / tmp
            )
            try:
                self._raw_write_all(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._raw_replace(root_fd, tmp, _RAW_WAL_BOOTSTRAP_NAME)
        except Exception:
            try:
                if tmp in set(self._raw_dir_names(root_fd)):
                    self._raw_unlink(root_fd, tmp)
            except OSError, ValueError:
                pass
            raise
        finally:
            os.close(root_fd)

    def _write_raw_wal_migration_bootstrap(
        self,
        *,
        root: Path,
        valid_size: int,
        record_count: int,
        legacy_sha256: str,
        record: bytes,
    ) -> None:
        body = {
            "version": 2,
            "kind": "legacy_migration",
            "valid_size": valid_size,
            "record_count": record_count,
            "legacy_sha256": legacy_sha256,
            "record_hex": record.hex(),
            "record_sha256": sha256(record).hexdigest(),
        }
        payload = {
            **body,
            "bootstrap_sha256": sha256(self._raw_canonical_json(body).encode()).hexdigest(),
        }
        self._write_raw_wal_bootstrap_payload(root=root, payload=payload)

    def _write_raw_wal_bootstrap_payload(self, *, root: Path, payload: Mapping[str, Any]) -> None:
        encoded = self._raw_control_bytes(payload, label="WAL bootstrap")
        root_fd = self._raw_dir_fd(root, create=True)
        tmp = f".raw-wal-bootstrap-{uuid.uuid4().hex}.tmp"
        try:
            fd = self._raw_open_regular(
                root_fd, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, path=root / tmp
            )
            try:
                self._raw_write_all(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._raw_replace(root_fd, tmp, _RAW_WAL_BOOTSTRAP_NAME)
        except Exception:
            try:
                if tmp in set(self._raw_dir_names(root_fd)):
                    self._raw_unlink(root_fd, tmp)
            except OSError, ValueError:
                pass
            raise
        finally:
            os.close(root_fd)

    def _clear_raw_wal_bootstrap(self, *, root_fd: int) -> None:
        if _RAW_WAL_BOOTSTRAP_NAME in self._raw_dir_names(root_fd):
            self._raw_unlink(root_fd, _RAW_WAL_BOOTSTRAP_NAME)

    def _raw_wal_tail_payload(
        self,
        *,
        info: os.stat_result,
        offset: int,
        record: bytes,
        count: int,
        generation: int,
        previous: str | None,
    ) -> dict[str, Any]:
        body = {
            "version": _RAW_WAL_TAIL_VERSION,
            "wal_identity": self._raw_wal_identity(info),
            "last_offset": offset,
            "last_length": len(record),
            "last_sha256": sha256(record).hexdigest(),
            "record_count": count,
            "generation": generation,
            "previous_tail_sha256": previous,
        }
        return {**body, "tail_sha256": sha256(self._raw_canonical_json(body).encode()).hexdigest()}

    def _raw_wal_record(self, record: bytes, *, require_v2: bool = False) -> dict[str, Any] | None:
        if not record.endswith(b"\n") or len(record) > _RAW_WAL_MAX_RECORD_BYTES:
            raise ValueError("Raw aggTrades WAL record is invalid")
        try:
            value = json.loads(
                record[:-1].decode(),
                object_pairs_hook=self._json_object_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Raw aggTrades WAL has malformed complete record") from exc
        if (
            not isinstance(value, dict)
            or record != self._raw_canonical_json(value).encode() + b"\n"
        ):
            raise ValueError("Raw aggTrades WAL has noncanonical complete record")
        marker = value.get("_raw_wal_v2")
        if marker is None:
            if require_v2:
                raise ValueError("Raw aggTrades WAL record is missing v2 metadata")
            return None
        required = {"version", "sequence", "previous_record_sha256", "payload"}
        if (
            set(value) != {"_raw_wal_v2"}
            or not isinstance(marker, dict)
            or set(marker) != required
            or type(marker["version"]) is not int
            or marker["version"] != _RAW_WAL_RECORD_VERSION
            or type(marker["sequence"]) is not int
            or marker["sequence"] < 1
            or (
                marker["previous_record_sha256"] is not None
                and (
                    type(marker["previous_record_sha256"]) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", marker["previous_record_sha256"]) is None
                )
            )
            or not isinstance(marker["payload"], dict)
        ):
            raise ValueError("Raw aggTrades WAL v2 record schema is invalid")
        return marker

    def _raw_encode_wal_record(
        self, *, payload: Mapping[str, Any], sequence: int, previous: str | None
    ) -> bytes:
        return (
            self._raw_canonical_json(
                {
                    "_raw_wal_v2": {
                        "version": _RAW_WAL_RECORD_VERSION,
                        "sequence": sequence,
                        "previous_record_sha256": previous,
                        "payload": dict(payload),
                    }
                }
            ).encode()
            + b"\n"
        )

    def _raw_bounded_last_wal_record(self, fd: int) -> bytes | None:
        before = self._raw_checked_fd(fd, path=Path(_RAW_WAL_NAME))
        if before.st_size == 0:
            return None
        window = min(before.st_size, 2 * _RAW_WAL_MAX_RECORD_BYTES + 2)
        os.lseek(fd, before.st_size - window, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = window
        while remaining:
            try:
                chunk = os.read(fd, remaining)
            except InterruptedError:
                continue
            if not chunk:
                raise ValueError("Raw aggTrades WAL changed while reading its tail")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = self._raw_checked_fd(fd, path=Path(_RAW_WAL_NAME))
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("Raw aggTrades WAL changed while reading its tail")
        data = b"".join(chunks)
        if not data.endswith(b"\n"):
            suffix = data.rsplit(b"\n", 1)[-1]
            if len(suffix) > _RAW_WAL_MAX_RECORD_BYTES:
                raise ValueError("Raw aggTrades WAL torn suffix is too large")
        if data.endswith(b"\n"):
            complete = data
        elif b"\n" in data:
            complete = data.rsplit(b"\n", 1)[0] + b"\n"
        else:
            return None
        pieces = complete.rsplit(b"\n", 2)
        if len(pieces) < 2:
            return None
        record = pieces[-2] + b"\n"
        if len(record) > _RAW_WAL_MAX_RECORD_BYTES:
            raise ValueError("Raw aggTrades WAL record is too large")
        return record

    def _scan_legacy_raw_wal(self, fd: int, *, limit: int | None = None) -> tuple[int, int, str]:
        os.lseek(fd, 0, os.SEEK_SET)
        valid = 0
        count = 0
        pending = b""
        digest = sha256()
        remaining = limit
        while remaining is None or remaining:
            chunk_size = _RAW_WAL_MAX_RECORD_BYTES + 1
            if remaining is not None:
                chunk_size = min(chunk_size, remaining)
            chunk = os.read(fd, chunk_size)
            if not chunk:
                break
            if remaining is not None:
                remaining -= len(chunk)
            pending += chunk
            while b"\n" in pending:
                record, pending = pending.split(b"\n", 1)
                record += b"\n"
                if self._raw_wal_record(record) is not None:
                    raise ValueError("Raw aggTrades WAL v2 record is missing its tail sidecar")
                valid += len(record)
                count += 1
                digest.update(record)
            if len(pending) > _RAW_WAL_MAX_RECORD_BYTES:
                raise ValueError("Raw aggTrades WAL torn suffix is too large")
        if limit is not None and remaining:
            raise ValueError("Raw aggTrades WAL bootstrap WAL is truncated")
        if pending and limit is not None:
            raise ValueError("Raw aggTrades WAL bootstrap valid prefix is invalid")
        return valid, count, digest.hexdigest()

    def _migrate_raw_wal_tail(self, path: Path) -> dict[str, Any]:
        """One-time full validation and v2 anchoring for a legacy WAL."""
        root_fd = self._raw_dir_fd(path.parent, create=False)
        try:
            fd = self._raw_open_regular(root_fd, path.name, os.O_RDONLY, path=path)
            try:
                valid, count, legacy_sha256 = self._scan_legacy_raw_wal(fd)
            finally:
                os.close(fd)
        finally:
            os.close(root_fd)
        anchor = self._raw_encode_wal_record(
            payload={"_raw_wal_migration": "legacy"},
            sequence=count + 1,
            previous=legacy_sha256 if count else None,
        )
        self._write_raw_wal_migration_bootstrap(
            root=path.parent,
            valid_size=valid,
            record_count=count,
            legacy_sha256=legacy_sha256,
            record=anchor,
        )
        recovered = self._recover_raw_wal(path)
        if recovered is None:
            raise ValueError("Raw aggTrades WAL migration recovery did not publish a tail")
        return recovered

    def _recover_raw_wal_migration_bootstrap(
        self, *, path: Path, root: Path, root_fd: int, bootstrap: Mapping[str, Any]
    ) -> dict[str, Any]:
        record = bytes.fromhex(str(bootstrap["record_hex"]))
        if _RAW_WAL_TAIL_NAME in self._raw_dir_names(root_fd):
            tail = self._raw_wal_tail(root=root, root_fd=root_fd)
            if (
                tail["record_count"] != bootstrap["record_count"] + 1
                or tail["last_offset"] != bootstrap["valid_size"]
                or tail["last_length"] != len(record)
                or tail["last_sha256"] != bootstrap["record_sha256"]
            ):
                raise ValueError("Raw aggTrades WAL migration tail diverges")
            self._clear_raw_wal_bootstrap(root_fd=root_fd)
            return self._recover_raw_wal(path)
        if path.name not in self._raw_dir_names(root_fd):
            raise ValueError("Raw aggTrades WAL migration WAL is missing")
        fd = self._raw_open_regular(root_fd, path.name, os.O_RDWR, path=path)
        try:
            info = os.fstat(fd)
            if info.st_size < bootstrap["valid_size"]:
                raise ValueError("Raw aggTrades WAL migration WAL is truncated")
            valid, count, legacy_sha256 = self._scan_legacy_raw_wal(
                fd, limit=bootstrap["valid_size"]
            )
            if (
                valid != bootstrap["valid_size"]
                or count != bootstrap["record_count"]
                or legacy_sha256 != bootstrap["legacy_sha256"]
            ):
                raise ValueError("Raw aggTrades WAL migration prefix diverges")
            os.ftruncate(fd, bootstrap["valid_size"])
            os.fsync(fd)
            os.fsync(root_fd)
            os.lseek(fd, bootstrap["valid_size"], os.SEEK_SET)
            self._raw_write_all(fd, record)
            os.fsync(fd)
            info = os.fstat(fd)
        finally:
            os.close(fd)
        tail = self._raw_wal_tail_payload(
            info=info,
            offset=bootstrap["valid_size"],
            record=record,
            count=bootstrap["record_count"] + 1,
            generation=0,
            previous=None,
        )
        self._write_raw_wal_tail(root=root, payload=tail)
        self._clear_raw_wal_bootstrap(root_fd=root_fd)
        return tail

    def _recover_raw_wal(self, path: Path) -> dict[str, Any] | None:
        root = path.parent
        root_fd = self._raw_dir_fd(root, create=False)
        try:
            names = set(self._raw_dir_names(root_fd))
            if _RAW_WAL_BOOTSTRAP_NAME in names:
                bootstrap = self._raw_wal_bootstrap(root=root, root_fd=root_fd)
                if bootstrap["version"] == 2:
                    return self._recover_raw_wal_migration_bootstrap(
                        path=path, root=root, root_fd=root_fd, bootstrap=bootstrap
                    )
                if _RAW_WAL_TAIL_NAME in names:
                    raise ValueError("Raw aggTrades WAL bootstrap conflicts with tail sidecar")
                record = bytes.fromhex(bootstrap["record_hex"])
                if path.name not in names:
                    fd = self._raw_open_regular(
                        root_fd,
                        path.name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        path=path,
                    )
                    try:
                        self._raw_write_all(fd, record)
                        os.fsync(fd)
                        info = os.fstat(fd)
                    finally:
                        os.close(fd)
                    os.fsync(root_fd)
                else:
                    fd = self._raw_open_regular(root_fd, path.name, os.O_RDWR, path=path)
                    try:
                        info = os.fstat(fd)
                        if info.st_size == 0:
                            self._raw_write_all(fd, record)
                            os.fsync(fd)
                            info = os.fstat(fd)
                        elif info.st_size < len(record):
                            os.lseek(fd, 0, os.SEEK_SET)
                            prefix = self._raw_read_exact(
                                fd,
                                info.st_size,
                                error="Raw aggTrades WAL bootstrap WAL is truncated",
                            )
                            if prefix != record[: info.st_size]:
                                raise ValueError("Raw aggTrades WAL bootstrap WAL diverges")
                            os.ftruncate(fd, 0)
                            os.lseek(fd, 0, os.SEEK_SET)
                            self._raw_write_all(fd, record)
                            os.fsync(fd)
                            info = os.fstat(fd)
                        else:
                            if info.st_size != len(record):
                                raise ValueError("Raw aggTrades WAL bootstrap WAL diverges")
                            os.lseek(fd, 0, os.SEEK_SET)
                            if (
                                self._raw_read_exact(
                                    fd,
                                    len(record),
                                    error="Raw aggTrades WAL bootstrap WAL is truncated",
                                )
                                != record
                            ):
                                raise ValueError("Raw aggTrades WAL bootstrap WAL diverges")
                    finally:
                        os.close(fd)
                tail = self._raw_wal_tail_payload(
                    info=info,
                    offset=0,
                    record=record,
                    count=1,
                    generation=0,
                    previous=None,
                )
                self._write_raw_wal_tail(root=root, payload=tail)
                self._clear_raw_wal_bootstrap(root_fd=root_fd)
                return tail
            if path.name not in names:
                if _RAW_WAL_TAIL_NAME in names:
                    raise ValueError("Raw aggTrades WAL tail exists without WAL")
                return None
            if _RAW_WAL_TAIL_NAME not in names:
                fd = self._raw_open_regular(root_fd, path.name, os.O_RDONLY, path=path)
                try:
                    last = self._raw_bounded_last_wal_record(fd)
                    if last is not None and self._raw_wal_record(last) is not None:
                        raise ValueError("Raw aggTrades WAL v2 record is missing its tail sidecar")
                finally:
                    os.close(fd)
                return self._migrate_raw_wal_tail(path)
            tail = self._raw_wal_tail(root=root, root_fd=root_fd)
            fd = self._raw_open_regular(root_fd, path.name, os.O_RDWR, path=path)
            try:
                info = os.fstat(fd)
                expected = tail["wal_identity"]
                current = self._raw_wal_identity(info)
                if (
                    current["dev"] != expected["dev"]
                    or current["ino"] != expected["ino"]
                    or info.st_size < expected["size"]
                ):
                    raise ValueError("Raw aggTrades WAL identity diverges")
                os.lseek(fd, tail["last_offset"], os.SEEK_SET)
                record = self._raw_read_exact(
                    fd, tail["last_length"], error="Raw aggTrades WAL tail record is truncated"
                )
                if (
                    len(record) != tail["last_length"]
                    or sha256(record).hexdigest() != tail["last_sha256"]
                ):
                    raise ValueError("Raw aggTrades WAL tail record diverges")
                marker = self._raw_wal_record(record, require_v2=True)
                if marker["sequence"] != tail["record_count"]:
                    raise ValueError("Raw aggTrades WAL tail sequence diverges")
                if info.st_size == expected["size"]:
                    if current != expected:
                        raise ValueError("Raw aggTrades WAL identity diverges")
                    return tail
                if info.st_size - expected["size"] > _RAW_WAL_MAX_RECORD_BYTES:
                    raise ValueError("Raw aggTrades WAL suffix is too large")
                os.lseek(fd, expected["size"], os.SEEK_SET)
                suffix = self._raw_read_exact(
                    fd,
                    info.st_size - expected["size"],
                    error="Raw aggTrades WAL suffix is truncated",
                )
                if b"\n" not in suffix:
                    os.ftruncate(fd, expected["size"])
                    os.fsync(fd)
                    os.fsync(root_fd)
                    info = os.fstat(fd)
                    rebuilt = self._raw_wal_tail_payload(
                        info=info,
                        offset=tail["last_offset"],
                        record=record,
                        count=tail["record_count"],
                        generation=tail["generation"] + 1,
                        previous=tail["tail_sha256"],
                    )
                    self._write_raw_wal_tail(root=root, payload=rebuilt)
                    return rebuilt
                if suffix.count(b"\n") != 1 or not suffix.endswith(b"\n"):
                    raise ValueError("Raw aggTrades WAL suffix is invalid")
                suffix_marker = self._raw_wal_record(suffix, require_v2=True)
                if (
                    suffix_marker["sequence"] != tail["record_count"] + 1
                    or suffix_marker["previous_record_sha256"] != tail["last_sha256"]
                ):
                    raise ValueError("Raw aggTrades WAL suffix sequence diverges")
                info = os.fstat(fd)
            finally:
                os.close(fd)
        finally:
            os.close(root_fd)
        recovered = self._raw_wal_tail_payload(
            info=info,
            offset=tail["wal_identity"]["size"],
            record=suffix,
            count=tail["record_count"] + 1,
            generation=tail["generation"] + 1,
            previous=tail["tail_sha256"],
        )
        self._write_raw_wal_tail(root=root, payload=recovered)
        return recovered

    def _raw_file_digest(self, path: Path) -> tuple[int, str]:
        parent_fd = self._raw_dir_fd(path.parent, create=False)
        try:
            fd = self._raw_open_regular(parent_fd, path.name, os.O_RDONLY, path=path)
            try:
                self._raw_checked_fd(fd, path=path)
                digest = sha256()
                size = 0
                while chunk := os.read(fd, 1_048_576):
                    size += len(chunk)
                    digest.update(chunk)
                return size, digest.hexdigest()
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)

    def _raw_inventory_entry(self, path: Path) -> dict[str, Any]:
        before = self._raw_regular_stat(path)
        try:
            frame = self._validate_raw_aggtrades_frame(self._raw_read_parquet(path))
        except Exception as exc:
            raise ValueError(f"Raw aggTrades part is not strict parquet: {path}") from exc
        self._assert_raw_regular_stable(path, before)
        if frame.is_empty():
            raise ValueError(f"Raw aggTrades part is empty: {path}")
        rows = frame.to_dicts()
        token = path.parent.name.removeprefix("date=")
        if not _RAW_DATE_PATTERN.fullmatch(token) or any(
            self._partition_date_from_ms(row["timestamp_ms"]) != token for row in rows
        ):
            raise ValueError(f"Raw aggTrades part has invalid UTC date binding: {path}")
        byte_count, digest = self._raw_file_digest(path)
        self._assert_raw_regular_stable(path, before)
        return {
            "name": f"{path.parent.name}/{path.name}",
            "date": token,
            "byte_count": byte_count,
            "content_sha256": digest,
            "row_count": frame.height,
            "schema": {name: str(dtype) for name, dtype in _RAW_AGGTRADES_SCHEMA.items()},
            "min_trade_id": int(rows[0]["agg_trade_id"]),
            "max_trade_id": int(rows[-1]["agg_trade_id"]),
            "min_timestamp_ms": int(rows[0]["timestamp_ms"]),
            "max_timestamp_ms": int(rows[-1]["timestamp_ms"]),
            "file_identity": {
                "dev": before.st_dev,
                "ino": before.st_ino,
                "size": before.st_size,
                "mtime_ns": before.st_mtime_ns,
            },
        }

    @staticmethod
    def _raw_canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def _raw_control_bytes(cls, value: Mapping[str, Any], *, label: str) -> bytes:
        encoded = cls._raw_canonical_json(value).encode()
        if len(encoded) > _RAW_CONTROL_MAX_BYTES:
            raise ValueError(f"Raw aggTrades {label} is too large")
        return encoded

    @classmethod
    def _raw_inventory_payload(
        cls,
        *,
        exchange: str,
        symbol: str,
        generation: int,
        previous_inventory_sha256: str | None,
        entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        body = {
            "version": 2,
            "exchange": exchange,
            "symbol": symbol,
            "generation": generation,
            "previous_inventory_sha256": previous_inventory_sha256,
            "parts": sorted(entries, key=lambda item: item["name"]),
        }
        return {
            **body,
            "inventory_sha256": sha256(cls._raw_canonical_json(body).encode()).hexdigest(),
        }

    def _parse_raw_inventory(
        self, raw: str, *, exchange: str, symbol: str, allow_v1: bool = True
    ) -> dict[str, Any]:
        exchange = self._normalize_raw_exchange_token(exchange)
        symbol = self._normalize_raw_symbol_token(symbol)
        try:
            payload = json.loads(
                raw,
                object_pairs_hook=self._json_object_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            if not isinstance(payload, dict) or raw != self._raw_canonical_json(payload):
                raise ValueError("inventory is not canonical")
            version = payload.get("version")
            if type(version) is int and version == 1 and allow_v1:
                if set(payload) != {"version", "parts", "inventory_sha256"}:
                    raise ValueError("legacy inventory fields are invalid")
                body = {"version": 1, "parts": payload["parts"]}
                if (
                    payload["inventory_sha256"]
                    != sha256(self._raw_canonical_json(body).encode()).hexdigest()
                ):
                    raise ValueError("legacy inventory hash is invalid")
                return payload
            required = {
                "version",
                "exchange",
                "symbol",
                "generation",
                "previous_inventory_sha256",
                "parts",
                "inventory_sha256",
            }
            if set(payload) != required or type(version) is not int or version != 2:
                raise ValueError("inventory schema is invalid")
            if payload["exchange"] != exchange or payload["symbol"] != symbol:
                raise ValueError("inventory stream does not match")
            if type(payload["generation"]) is not int or payload["generation"] < 0:
                raise ValueError("inventory generation is invalid")
            predecessor = payload["previous_inventory_sha256"]
            if (payload["generation"] == 0) != (predecessor is None):
                raise ValueError("inventory predecessor is invalid")
            if predecessor is not None and (
                type(predecessor) is not str or not re.fullmatch(r"[0-9a-f]{64}", predecessor)
            ):
                raise ValueError("inventory predecessor is invalid")
            body = {key: value for key, value in payload.items() if key != "inventory_sha256"}
            if (
                type(payload["inventory_sha256"]) is not str
                or payload["inventory_sha256"]
                != sha256(self._raw_canonical_json(body).encode()).hexdigest()
            ):
                raise ValueError("inventory hash is invalid")
            if (
                not isinstance(payload["parts"], list)
                or any(
                    not isinstance(entry, dict) or type(entry.get("name")) is not str
                    for entry in payload["parts"]
                )
                or [entry["name"] for entry in payload["parts"]]
                != sorted(entry["name"] for entry in payload["parts"])
                or len({entry["name"] for entry in payload["parts"]}) != len(payload["parts"])
            ):
                raise ValueError("inventory part ordering is invalid")
            return payload
        except (TypeError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Raw aggTrades inventory is malformed") from exc

    def _publish_raw_inventory(self, *, root: Path, payload: Mapping[str, Any]) -> None:
        encoded = self._raw_control_bytes(payload, label="inventory")
        root_fd = self._raw_dir_fd(root, create=True)
        token = uuid.uuid4().hex
        tmp = f".raw-inventory-{token}.tmp"
        try:
            fd = self._raw_open_regular(
                root_fd, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, path=root / tmp
            )
            try:
                self._raw_write_all(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._raw_replace(root_fd, tmp, _RAW_INVENTORY_NAME)
        except Exception:
            try:
                if tmp in set(self._raw_dir_names(root_fd)):
                    self._raw_unlink(root_fd, tmp)
            except OSError, ValueError:
                pass
            raise
        finally:
            os.close(root_fd)

    def _authenticate_raw_inventory(
        self,
        *,
        exchange: str,
        symbol: str,
        migrate: bool = False,
        snapshot: bool = False,
        temporary_output: tuple[str, str, bool] | None = None,
        temporary_obsolete: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]] | _RawInventorySnapshot:
        exchange = self._normalize_raw_exchange_token(exchange)
        symbol = self._normalize_raw_symbol_token(symbol)
        root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
        try:
            root_fd = self._raw_dir_fd(root, create=False)
        except FileNotFoundError:
            if migrate:
                self._publish_raw_inventory(
                    root=root,
                    payload=self._raw_inventory_payload(
                        exchange=exchange,
                        symbol=symbol,
                        generation=0,
                        previous_inventory_sha256=None,
                        entries=[],
                    ),
                )
                if snapshot:
                    return self._authenticate_raw_inventory(
                        exchange=exchange, symbol=symbol, snapshot=True
                    )
            return []
        try:
            parts: list[Path] = []
            names = self._raw_dir_names(root_fd)
            for name in names:
                if not name.startswith("date="):
                    continue
                if not _RAW_DATE_PATTERN.fullmatch(name[5:]):
                    raise ValueError("Raw aggTrades date entry is unsafe")
                partition = root / name
                partition_fd = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
                )
                try:
                    parts.extend(
                        partition / part
                        for part in self._raw_dir_names(partition_fd)
                        if _RAW_PART_PATTERN.fullmatch(part) is not None
                    )
                finally:
                    os.close(partition_fd)
            actual = sorted(f"{part.parent.name}/{part.name}" for part in parts)
            temporary_name: str | None = None
            temporary_base_sha256: str | None = None
            temporary_replaces_base = False
            if temporary_output is not None:
                temporary_name, temporary_base_sha256, temporary_replaces_base = temporary_output
                if (
                    type(temporary_name) is not str
                    or type(temporary_base_sha256) is not str
                    or type(temporary_replaces_base) is not bool
                    or re.fullmatch(r"date=\d{4}-\d{2}-\d{2}/part-\d+\.parquet", temporary_name)
                    is None
                    or temporary_name not in actual
                ):
                    raise ValueError("Raw aggTrades temporary transaction output is invalid")
            if _RAW_INVENTORY_NAME not in names:
                if parts:
                    raise ValueError(
                        "Raw aggTrades inventory is missing after raw parts were established"
                    )
                if migrate:
                    self._publish_raw_inventory(
                        root=root,
                        payload=self._raw_inventory_payload(
                            exchange=exchange,
                            symbol=symbol,
                            generation=0,
                            previous_inventory_sha256=None,
                            entries=[],
                        ),
                    )
                    if snapshot:
                        return self._authenticate_raw_inventory(
                            exchange=exchange, symbol=symbol, snapshot=True
                        )
                return []
            raw, inventory_info = self._raw_read_bytes(
                root_fd, _RAW_INVENTORY_NAME, path=root / _RAW_INVENTORY_NAME
            )
            payload = self._parse_raw_inventory(
                raw.decode("utf-8"), exchange=exchange, symbol=symbol
            )
        finally:
            os.close(root_fd)
        entries = payload["parts"]
        expected = sorted(item.get("name") for item in entries if isinstance(item, dict))
        temporary_base_is_current = (
            temporary_name is not None and payload["inventory_sha256"] == temporary_base_sha256
        )
        extra_is_authorized = (
            temporary_base_is_current
            and temporary_name not in expected
            and actual == sorted([*expected, temporary_name])
        )
        replacement_is_authorized = (
            temporary_base_is_current
            and temporary_replaces_base
            and temporary_name in expected
            and actual == expected
        )
        authorized_obsolete = dict(temporary_obsolete or {})
        if any(
            re.fullmatch(r"date=\d{4}-\d{2}-\d{2}/part-\d+\.parquet", name) is None
            or not isinstance(entry, Mapping)
            for name, entry in authorized_obsolete.items()
        ):
            raise ValueError("Raw aggTrades temporary transaction obsolete part is invalid")
        present_obsolete = sorted(set(actual) & set(authorized_obsolete))
        obsolete_is_authorized = actual == sorted([*expected, *present_obsolete])
        if any(
            self._raw_inventory_entry(root / name) != authorized_obsolete[name]
            for name in present_obsolete
        ):
            raise ValueError("Raw aggTrades obsolete transaction part diverges")
        if (
            not isinstance(entries, list)
            or (actual != expected and not extra_is_authorized and not obsolete_is_authorized)
            or (
                actual == expected
                and temporary_base_is_current
                and temporary_name in expected
                and not replacement_is_authorized
            )
        ):
            raise ValueError("Raw aggTrades inventory diverges from raw parts")
        legacy = type(payload["version"]) is int and payload["version"] == 1
        required = {
            "name",
            "date",
            "byte_count",
            "content_sha256",
            "row_count",
            "schema",
            "min_trade_id",
            "max_trade_id",
            "min_timestamp_ms",
            "max_timestamp_ms",
            "file_identity",
        }
        previous: dict[str, Any] | None = None
        for entry in sorted(entries, key=lambda item: (item["min_trade_id"], item["name"])):
            if not isinstance(entry, dict) or set(entry) != required:
                raise ValueError("Raw aggTrades inventory entry is invalid")
            if entry["schema"] != {
                name: str(dtype) for name, dtype in _RAW_AGGTRADES_SCHEMA.items()
            }:
                raise ValueError("Raw aggTrades inventory schema is invalid")
            if (
                type(entry["name"]) is not str
                or not re.fullmatch(r"date=\d{4}-\d{2}-\d{2}/part-\d+\.parquet", entry["name"])
                or entry["date"] != entry["name"][5:15]
            ):
                raise ValueError("Raw aggTrades inventory part name is invalid")
            info = self._raw_regular_stat(root / entry["name"])
            is_authorized_replacement = (
                replacement_is_authorized and entry["name"] == temporary_name
            )
            identity = entry["file_identity"]
            expected_identity = {
                "dev": info.st_dev,
                "ino": info.st_ino,
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
            if not is_authorized_replacement and (
                not isinstance(identity, dict) or identity != expected_identity
            ):
                raise ValueError("Raw aggTrades inventory file identity diverges")
            if (
                type(entry["byte_count"]) is not int
                or entry["byte_count"] <= 0
                or (not is_authorized_replacement and entry["byte_count"] != info.st_size)
            ):
                raise ValueError("Raw aggTrades inventory byte count is invalid")
            if any(
                type(entry[key]) is not int
                for key in (
                    "row_count",
                    "min_trade_id",
                    "max_trade_id",
                    "min_timestamp_ms",
                    "max_timestamp_ms",
                )
            ):
                raise ValueError("Raw aggTrades inventory numeric field is invalid")
            if (
                entry["row_count"] <= 0
                or entry["min_trade_id"] < 0
                or entry["max_trade_id"] < entry["min_trade_id"]
                or entry["min_timestamp_ms"] <= 0
                or entry["max_timestamp_ms"] < entry["min_timestamp_ms"]
            ):
                raise ValueError("Raw aggTrades inventory ranges are invalid")
            if type(entry["content_sha256"]) is not str or not re.fullmatch(
                r"[0-9a-f]{64}", entry["content_sha256"]
            ):
                raise ValueError("Raw aggTrades inventory digest is invalid")
            if previous is not None and (
                entry["min_trade_id"] <= previous["max_trade_id"]
                or entry["min_timestamp_ms"] < previous["max_timestamp_ms"]
            ):
                raise ValueError("Raw aggTrades inventory global boundaries diverge")
            previous = entry
        if legacy:
            if not migrate:
                raise ValueError("Raw aggTrades inventory migration is required")
            migrated_entries = [
                self._raw_inventory_entry(root / entry["name"]) for entry in entries
            ]
            for old, migrated in zip(entries, migrated_entries, strict=True):
                if {key: value for key, value in migrated.items() if key != "file_identity"} != {
                    key: value for key, value in old.items() if key != "file_identity"
                } or migrated["file_identity"] != old["file_identity"]:
                    raise ValueError("Legacy raw aggTrades inventory content diverges")
            payload = self._raw_inventory_payload(
                exchange=exchange,
                symbol=symbol,
                generation=0,
                previous_inventory_sha256=None,
                entries=migrated_entries,
            )
            self._publish_raw_inventory(root=root, payload=payload)
            if snapshot:
                return self._authenticate_raw_inventory(
                    exchange=exchange, symbol=symbol, snapshot=True
                )
            entries = migrated_entries
        ordered_entries = sorted(entries, key=lambda item: (item["min_trade_id"], item["name"]))
        if snapshot:
            return _RawInventorySnapshot(
                raw=raw,
                payload=payload,
                entries=tuple(ordered_entries),
                generation=payload["generation"],
                inventory_sha256=payload["inventory_sha256"],
                file_identity=(
                    inventory_info.st_dev,
                    inventory_info.st_ino,
                    inventory_info.st_size,
                    inventory_info.st_mtime_ns,
                    inventory_info.st_ctime_ns,
                ),
            )
        return ordered_entries

    def _revalidate_raw_inventory_snapshot(
        self, *, exchange: str, symbol: str, snapshot: _RawInventorySnapshot
    ) -> None:
        root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
        root_fd = self._raw_dir_fd(root, create=False)
        try:
            raw, info = self._raw_read_bytes(
                root_fd, _RAW_INVENTORY_NAME, path=root / _RAW_INVENTORY_NAME
            )
        finally:
            os.close(root_fd)
        identity = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if (
            raw != snapshot.raw
            or identity != snapshot.file_identity
            or snapshot.payload["inventory_sha256"] != snapshot.inventory_sha256
            or snapshot.payload["generation"] != snapshot.generation
        ):
            raise ValueError("Raw aggTrades inventory changed before transaction publication")

    def _verify_raw_transaction_part(
        self, *, path: Path, partition: Path, byte_count: int, content_sha256: str
    ) -> None:
        self._raw_regular_stat(path)
        if self._raw_file_digest(path) != (byte_count, content_sha256):
            raise ValueError("transaction part bytes do not match descriptor")
        try:
            frame = self._validate_raw_aggtrades_frame(self._raw_read_parquet(path))
            token = partition.name.removeprefix("date=")
            if any(
                self._partition_date_from_ms(row["timestamp_ms"]) != token
                for row in frame.to_dicts()
            ):
                raise ValueError("transaction part contains wrong-date row")
        except Exception as exc:
            raise ValueError("transaction part is not strict raw parquet") from exc

    def _parse_raw_transaction(
        self, raw: str, *, exchange: str, symbol: str, partition: Path
    ) -> dict[str, Any]:
        exchange = self._normalize_raw_exchange_token(exchange)
        symbol = self._normalize_raw_symbol_token(symbol)
        try:
            descriptor = json.loads(
                raw,
                object_pairs_hook=self._json_object_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            required = {
                "version",
                "transaction_id",
                "exchange",
                "symbol",
                "partition",
                "base_inventory",
                "new_inventory",
                "output",
                "stage",
                "obsolete",
                "output_entry",
                "descriptor_sha256",
            }
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != required
                or raw != self._raw_canonical_json(descriptor)
            ):
                raise ValueError("transaction schema is invalid")
            body = {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
            if (
                type(descriptor["version"]) is not int
                or descriptor["version"] != 2
                or type(descriptor["transaction_id"]) is not str
                or not re.fullmatch(r"[0-9a-f]{32}", descriptor["transaction_id"])
                or descriptor["exchange"] != exchange
                or descriptor["symbol"] != symbol
                or descriptor["partition"] != partition.name
                or descriptor["descriptor_sha256"]
                != sha256(self._raw_canonical_json(body).encode()).hexdigest()
                or type(descriptor["output"]) is not str
                or _RAW_PART_PATTERN.fullmatch(descriptor["output"]) is None
                or type(descriptor["stage"]) is not str
                or descriptor["stage"] != f".raw-stage-{descriptor['transaction_id']}.parquet"
                or not isinstance(descriptor["obsolete"], list)
                or descriptor["obsolete"] != sorted(set(descriptor["obsolete"]))
                or any(_RAW_PART_PATTERN.fullmatch(item) is None for item in descriptor["obsolete"])
            ):
                raise ValueError("transaction descriptor is invalid")
            base = self._parse_raw_inventory(
                self._raw_canonical_json(descriptor["base_inventory"]),
                exchange=exchange,
                symbol=symbol,
                allow_v1=False,
            )
            new = self._parse_raw_inventory(
                self._raw_canonical_json(descriptor["new_inventory"]),
                exchange=exchange,
                symbol=symbol,
                allow_v1=False,
            )
            if (
                new["generation"] != base["generation"] + 1
                or new["previous_inventory_sha256"] != base["inventory_sha256"]
                or descriptor["output_entry"] not in new["parts"]
                or descriptor["output_entry"]["name"] != f"{partition.name}/{descriptor['output']}"
            ):
                raise ValueError("transaction inventory transition is invalid")
            base_by_name = {item["name"]: item for item in base["parts"]}
            new_by_name = {item["name"]: item for item in new["parts"]}
            obsolete_names = {f"{partition.name}/{item}" for item in descriptor["obsolete"]}
            if (
                not obsolete_names <= set(base_by_name)
                or (
                    descriptor["output_entry"]["name"] in base_by_name
                    and descriptor["output_entry"]["name"] not in obsolete_names
                )
                or set(new_by_name)
                != (set(base_by_name) - obsolete_names) | {descriptor["output_entry"]["name"]}
                or any(
                    item["name"].split("/", 1)[0] != partition.name
                    for item in (base_by_name[name] for name in obsolete_names)
                )
            ):
                raise ValueError("transaction delta is invalid")
            output_name = descriptor["output_entry"]["name"]
            unchanged = (set(base_by_name) - obsolete_names) - {output_name}
            if any(base_by_name[name] != new_by_name[name] for name in unchanged):
                raise ValueError("transaction unchanged entry diverges")
            return descriptor
        except (TypeError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Raw aggTrades transaction is malformed") from exc

    def _assert_raw_transaction_artifact_closure(
        self, *, names: set[str], descriptor: Mapping[str, Any] | None
    ) -> None:
        artifacts = {
            name
            for name in names
            if _RAW_TRANSACTION_STAGE_PATTERN.fullmatch(name) is not None
            or _RAW_TRANSACTION_TEMP_PATTERN.fullmatch(name) is not None
        }
        expected = (
            {descriptor["stage"]}
            if descriptor is not None and descriptor["stage"] in artifacts
            else set()
        )
        if artifacts != expected:
            raise ValueError("unbound transaction artifact closure is invalid")

    def _recover_raw_part_transaction(self, *, exchange: str, symbol: str, partition: Path) -> None:
        partition_fd = self._raw_dir_fd(partition, create=False)
        try:
            names = set(self._raw_dir_names(partition_fd))
            if _RAW_TRANSACTION_NAME not in names:
                self._assert_raw_transaction_artifact_closure(names=names, descriptor=None)
                return
            raw, _ = self._raw_read_bytes(
                partition_fd, _RAW_TRANSACTION_NAME, path=partition / _RAW_TRANSACTION_NAME
            )
            descriptor = self._parse_raw_transaction(
                raw.decode("utf-8"), exchange=exchange, symbol=symbol, partition=partition
            )
            self._assert_raw_transaction_artifact_closure(names=names, descriptor=descriptor)
            root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
            base = descriptor["base_inventory"]
            new = descriptor["new_inventory"]
            root_fd = self._raw_dir_fd(root, create=False)
            try:
                current_raw, _ = self._raw_read_bytes(
                    root_fd, _RAW_INVENTORY_NAME, path=root / _RAW_INVENTORY_NAME
                )
            finally:
                os.close(root_fd)
            current_control = self._parse_raw_inventory(
                current_raw.decode("utf-8"), exchange=exchange, symbol=symbol
            )
            current_identity = (
                current_control["generation"],
                current_control["inventory_sha256"],
            )
            if current_identity not in {
                (base["generation"], base["inventory_sha256"]),
                (new["generation"], new["inventory_sha256"]),
            }:
                raise ValueError("transaction does not match current inventory")
            current_is_base = current_identity == (
                base["generation"],
                base["inventory_sha256"],
            )
            current_is_new = current_identity == (
                new["generation"],
                new["inventory_sha256"],
            )
            stage_name = descriptor["stage"]
            output_name = descriptor["output"]
            has_stage = stage_name in names
            has_output = output_name in names
            base_by_name = {entry["name"]: entry for entry in base["parts"]}
            if has_stage and has_output:
                if not current_is_base or output_name not in descriptor["obsolete"]:
                    raise ValueError("transaction stage and output phase is invalid")
                base_output_entry = base_by_name[f"{partition.name}/{output_name}"]
                if self._raw_inventory_entry(partition / output_name) != base_output_entry:
                    raise ValueError("transaction base output identity changed")
                stage_entry = self._raw_inventory_entry(partition / stage_name)
                expected_stage_entry = {
                    **descriptor["output_entry"],
                    "name": f"{partition.name}/{stage_name}",
                }
                if stage_entry != expected_stage_entry:
                    raise ValueError("transaction stage identity changed")
            elif has_output:
                self._verify_raw_transaction_part(
                    path=partition / output_name,
                    partition=partition,
                    byte_count=descriptor["output_entry"]["byte_count"],
                    content_sha256=descriptor["output_entry"]["content_sha256"],
                )
                actual_output_entry = self._raw_inventory_entry(partition / output_name)
                if actual_output_entry != descriptor["output_entry"]:
                    raise ValueError("transaction output identity changed after rename")
            authorized_obsolete = {
                f"{partition.name}/{name}": base_by_name[f"{partition.name}/{name}"]
                for name in descriptor["obsolete"]
                if f"{partition.name}/{name}" not in {entry["name"] for entry in new["parts"]}
            }
            current = self._authenticate_raw_inventory(
                exchange=exchange,
                symbol=symbol,
                snapshot=True,
                temporary_output=(
                    f"{partition.name}/{output_name}",
                    base["inventory_sha256"],
                    output_name in descriptor["obsolete"],
                )
                if has_output and current_is_base and not has_stage
                else None,
                temporary_obsolete=authorized_obsolete if current_is_new else None,
            )
            if not isinstance(current, _RawInventorySnapshot):
                raise ValueError("Raw aggTrades inventory snapshot is unavailable")
            if (
                current.payload["generation"],
                current.payload["inventory_sha256"],
            ) == (new["generation"], new["inventory_sha256"]):
                names = set(self._raw_dir_names(partition_fd))
                for name in descriptor["obsolete"]:
                    if name == output_name or name not in names:
                        continue
                    entry = base_by_name[f"{partition.name}/{name}"]
                    info = self._raw_regular_stat(partition / name)
                    if (
                        info.st_dev != entry["file_identity"]["dev"]
                        or info.st_ino != entry["file_identity"]["ino"]
                        or info.st_size != entry["file_identity"]["size"]
                        or info.st_mtime_ns != entry["file_identity"]["mtime_ns"]
                    ):
                        raise ValueError("obsolete transaction part changed")
                    self._raw_unlink(partition_fd, name)
                self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
                self._raw_unlink(partition_fd, _RAW_TRANSACTION_NAME)
                return
            if (
                current.payload["generation"],
                current.payload["inventory_sha256"],
            ) != (base["generation"], base["inventory_sha256"]):
                raise ValueError("transaction does not match current inventory")
            if has_stage:
                self._verify_raw_transaction_part(
                    path=partition / stage_name,
                    partition=partition,
                    byte_count=descriptor["output_entry"]["byte_count"],
                    content_sha256=descriptor["output_entry"]["content_sha256"],
                )
                self._raw_replace(partition_fd, stage_name, output_name)
            elif not has_output:
                raise ValueError("transaction output is missing")
            if not has_output:
                self._verify_raw_transaction_part(
                    path=partition / output_name,
                    partition=partition,
                    byte_count=descriptor["output_entry"]["byte_count"],
                    content_sha256=descriptor["output_entry"]["content_sha256"],
                )
            actual_output_entry = self._raw_inventory_entry(partition / output_name)
            if actual_output_entry != descriptor["output_entry"]:
                raise ValueError("transaction output identity changed after rename")
            base_by_name = {item["name"]: item for item in base["parts"]}
            self._revalidate_raw_inventory_snapshot(
                exchange=exchange, symbol=symbol, snapshot=current
            )
            self._publish_raw_inventory(root=root, payload=new)
            names = set(self._raw_dir_names(partition_fd))
            for name in descriptor["obsolete"]:
                if name == output_name or name not in names:
                    continue
                path = partition / name
                info = self._raw_regular_stat(path)
                entry = base_by_name[f"{partition.name}/{name}"]
                if (
                    info.st_dev != entry["file_identity"]["dev"]
                    or info.st_ino != entry["file_identity"]["ino"]
                    or info.st_size != entry["file_identity"]["size"]
                    or info.st_mtime_ns != entry["file_identity"]["mtime_ns"]
                ):
                    raise ValueError("obsolete transaction part changed")
                self._raw_unlink(partition_fd, name)
            self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
            self._raw_unlink(partition_fd, _RAW_TRANSACTION_NAME)
        finally:
            os.close(partition_fd)

    def _publish_raw_part_set(
        self,
        *,
        exchange: str,
        symbol: str,
        partition: Path,
        output: Path,
        frame: pl.DataFrame,
        obsolete: list[Path],
    ) -> None:
        snapshot = self._authenticate_raw_inventory(
            exchange=exchange, symbol=symbol, migrate=True, snapshot=True
        )
        if not isinstance(snapshot, _RawInventorySnapshot):
            raise ValueError("Raw aggTrades inventory snapshot is unavailable")
        if len(snapshot.raw) > (_RAW_CONTROL_MAX_BYTES - 8192) // 2:
            raise ValueError("Raw aggTrades transaction descriptor is too large")
        base = snapshot.payload
        transaction_id = uuid.uuid4().hex
        stage_name = f".raw-stage-{transaction_id}.parquet"
        descriptor_temp = f".raw-transaction-{transaction_id}.tmp"
        staged = partition / stage_name
        partition_fd = self._raw_dir_fd(partition, create=True)
        try:
            fd = self._raw_open_regular(
                partition_fd, stage_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, path=staged
            )
            os.close(fd)
            try:
                self._raw_write_parquet(partition_fd, stage_name, path=staged, frame=frame)
                output_entry = self._raw_inventory_entry(staged)
                output_entry["name"] = f"{partition.name}/{output.name}"
                output_entry["date"] = partition.name[5:]
                base_by_name = {entry["name"]: entry for entry in base["parts"]}
                obsolete_names = sorted(path.name for path in obsolete)
                new_entries = [
                    entry
                    for name, entry in base_by_name.items()
                    if name not in {f"{partition.name}/{item}" for item in obsolete_names}
                ] + [output_entry]
                new = self._raw_inventory_payload(
                    exchange=self._normalize_raw_exchange_token(exchange),
                    symbol=self._normalize_raw_symbol_token(symbol),
                    generation=base["generation"] + 1,
                    previous_inventory_sha256=base["inventory_sha256"],
                    entries=new_entries,
                )
                descriptor_body = {
                    "version": 2,
                    "transaction_id": transaction_id,
                    "exchange": self._normalize_raw_exchange_token(exchange),
                    "symbol": self._normalize_raw_symbol_token(symbol),
                    "partition": partition.name,
                    "base_inventory": base,
                    "new_inventory": new,
                    "output": output.name,
                    "stage": stage_name,
                    "obsolete": obsolete_names,
                    "output_entry": output_entry,
                }
                descriptor = {
                    **descriptor_body,
                    "descriptor_sha256": sha256(
                        self._raw_canonical_json(descriptor_body).encode()
                    ).hexdigest(),
                }
                encoded_descriptor = self._raw_control_bytes(
                    descriptor, label="transaction descriptor"
                )
                temp = descriptor_temp
                fd = self._raw_open_regular(
                    partition_fd,
                    temp,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    path=partition / temp,
                )
                try:
                    self._raw_write_all(fd, encoded_descriptor)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                self._revalidate_raw_inventory_snapshot(
                    exchange=exchange, symbol=symbol, snapshot=snapshot
                )
                self._raw_replace(partition_fd, temp, _RAW_TRANSACTION_NAME)
                try:
                    self._revalidate_raw_inventory_snapshot(
                        exchange=exchange, symbol=symbol, snapshot=snapshot
                    )
                except (FileNotFoundError, OSError, ValueError) as exc:
                    raise ValueError("transaction does not match current inventory") from exc
                self._raw_replace(partition_fd, stage_name, output.name)
                self._recover_raw_part_transaction(
                    exchange=exchange, symbol=symbol, partition=partition
                )
            except Exception:
                names = set(self._raw_dir_names(partition_fd))
                if _RAW_TRANSACTION_NAME not in names:
                    if stage_name in names:
                        self._raw_unlink(partition_fd, stage_name)
                    if descriptor_temp in names:
                        self._raw_unlink(partition_fd, descriptor_temp)
                raise
        finally:
            os.close(partition_fd)

    def append_raw_wal_record(
        self,
        *,
        exchange: str,
        symbol: str,
        payload: dict[str, Any],
        lease: _RawStreamLease | None = None,
    ) -> None:
        try:
            canonical_payload = json.loads(self._raw_canonical_json(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Raw aggTrades WAL payload is not canonical JSON") from exc
        if not isinstance(canonical_payload, dict):
            raise ValueError("Raw aggTrades WAL payload must be an object")
        path = self._raw_wal_path(exchange=exchange, symbol=symbol)
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
            root_fd = self._raw_dir_fd(root, create=True)
            try:
                created = False
                committed = False
                owned_identity: tuple[int, int] | None = None
                try:
                    tail = self._recover_raw_wal(path)
                    sequence = 1 if tail is None else tail["record_count"] + 1
                    encoded = self._raw_encode_wal_record(
                        payload=canonical_payload,
                        sequence=sequence,
                        previous=None if tail is None else tail["last_sha256"],
                    )
                    if len(encoded) > _RAW_WAL_MAX_RECORD_BYTES:
                        raise ValueError("Raw aggTrades WAL record is too large")
                    created = _RAW_WAL_NAME not in self._raw_dir_names(root_fd)
                    if created:
                        self._write_raw_wal_bootstrap(root=root, record=encoded)
                    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
                    if created:
                        flags |= os.O_EXCL
                    fd = self._raw_open_regular(root_fd, _RAW_WAL_NAME, flags, path=path)
                    try:
                        opened = os.fstat(fd)
                        owned_identity = (opened.st_dev, opened.st_ino)
                        offset = opened.st_size
                        self._raw_write_all(fd, encoded)
                        os.fsync(fd)
                        info = os.fstat(fd)
                        committed = True
                    finally:
                        os.close(fd)
                    if created:
                        os.fsync(root_fd)
                    tail_payload = self._raw_wal_tail_payload(
                        info=info,
                        offset=offset,
                        record=encoded,
                        count=sequence,
                        generation=sequence - 1,
                        previous=None if tail is None else tail["tail_sha256"],
                    )
                    self._write_raw_wal_tail(root=root, payload=tail_payload)
                    if created:
                        self._clear_raw_wal_bootstrap(root_fd=root_fd)
                except Exception:
                    clear_bootstrap = created and not committed and owned_identity is None
                    if created and not committed and owned_identity is not None:
                        try:
                            fd = self._raw_open_regular(
                                root_fd, _RAW_WAL_NAME, os.O_RDONLY, path=path
                            )
                            try:
                                current = os.fstat(fd)
                            finally:
                                os.close(fd)
                            if (
                                current.st_dev,
                                current.st_ino,
                            ) == owned_identity and current.st_size == 0:
                                self._raw_unlink(root_fd, _RAW_WAL_NAME)
                                clear_bootstrap = True
                        except FileNotFoundError:
                            clear_bootstrap = True
                        except OSError, ValueError:
                            pass
                    if clear_bootstrap:
                        try:
                            self._clear_raw_wal_bootstrap(root_fd=root_fd)
                        except OSError, ValueError:
                            pass
                    raise
            finally:
                os.close(root_fd)
        finally:
            if owns:
                active.release()

    def recover_raw_stream(
        self, *, exchange: str, symbol: str, lease: _RawStreamLease | None = None
    ) -> None:
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
            try:
                root_fd = self._raw_dir_fd(root, create=False)
            except FileNotFoundError:
                return
            try:
                self._preflight_raw_controls(exchange=exchange, symbol=symbol)
                for name in sorted(self._raw_dir_names(root_fd)):
                    if not name.startswith("date="):
                        continue
                    if re.fullmatch(r"date=\d{4}-\d{2}-\d{2}", name) is None:
                        raise ValueError("Raw aggTrades partition name is invalid")
                    partition = root / name
                    partition_fd = os.open(
                        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
                    )
                    try:
                        for part_name in self._raw_dir_names(partition_fd):
                            if _RAW_PART_PATTERN.fullmatch(part_name) is not None:
                                self._raw_regular_stat(partition / part_name)
                    finally:
                        os.close(partition_fd)
                    self._recover_raw_part_transaction(
                        exchange=exchange, symbol=symbol, partition=partition
                    )
                root_names = set(self._raw_dir_names(root_fd))
                if {
                    _RAW_WAL_NAME,
                    _RAW_WAL_TAIL_NAME,
                    _RAW_WAL_BOOTSTRAP_NAME,
                } & root_names:
                    self._recover_raw_wal(self._raw_wal_path(exchange=exchange, symbol=symbol))
            finally:
                os.close(root_fd)
        finally:
            if owns:
                active.release()

    def append_raw_aggtrades(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: pl.DataFrame | list[dict[str, Any]] | tuple[dict[str, Any], ...],
        lease: _RawStreamLease | None = None,
    ) -> int:
        frame = self._ensure_raw_aggtrades_frame(rows)
        if frame.is_empty():
            return 0
        if (
            len({self._partition_date_from_ms(row["timestamp_ms"]) for row in frame.to_dicts()})
            != 1
        ):
            raise ValueError("Raw aggTrades append must contain exactly one UTC date")
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            self._preflight_raw_meta(exchange=exchange, symbol=symbol)
            self.recover_raw_stream(exchange=exchange, symbol=symbol, lease=active)
            self._authenticate_raw_inventory(exchange=exchange, symbol=symbol, migrate=True)
            return self._append_raw_aggtrades_unlocked(exchange=exchange, symbol=symbol, rows=frame)
        finally:
            if owns:
                active.release()

    def _preflight_raw_append(self, *, exchange: str, symbol: str, frame: pl.DataFrame) -> bool:
        """Use the authenticated inventory index; read bytes only for an overlap."""
        entries = self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
        rows = frame.to_dicts()
        minimum_id = int(rows[0]["agg_trade_id"])
        maximum_id = int(rows[-1]["agg_trade_id"])
        minimum_ts = int(rows[0]["timestamp_ms"])
        maximum_ts = int(rows[-1]["timestamp_ms"])
        if not entries:
            return True
        tail = entries[-1]
        token = self._partition_date_from_ms(minimum_ts)
        if (
            minimum_id > tail["max_trade_id"]
            and minimum_ts >= tail["max_timestamp_ms"]
            and token >= tail["date"]
        ):
            return True
        # Exceptional overlap/backfill: only candidates whose declared ranges intersect.
        root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
        incoming = {row["agg_trade_id"]: row for row in rows}
        candidates = [
            entry
            for entry in entries
            if not (entry["max_trade_id"] < minimum_id or entry["min_trade_id"] > maximum_id)
        ]
        for entry in candidates:
            path = root / entry["name"]
            existing = self._validate_raw_aggtrades_frame(
                self._raw_read_authenticated_parquet(path, entry)
            )
            for row in existing.to_dicts():
                if row["agg_trade_id"] in incoming and incoming[row["agg_trade_id"]] != row:
                    raise ValueError("Raw aggTrades duplicate aggregate ID conflicts")
        predecessor = [entry for entry in entries if entry["max_trade_id"] < minimum_id]
        successor = [entry for entry in entries if entry["min_trade_id"] > maximum_id]
        if predecessor and predecessor[-1]["max_timestamp_ms"] > minimum_ts:
            raise ValueError("Raw aggTrades timestamp regresses across partitions")
        if successor and maximum_ts > successor[0]["min_timestamp_ms"]:
            raise ValueError("Raw aggTrades timestamp regresses across partitions")
        return False

    def preflight_raw_aggtrades(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: pl.DataFrame | list[dict[str, Any]] | tuple[dict[str, Any], ...],
        lease: _RawStreamLease | None = None,
    ) -> None:
        """Fail closed on every date touched by a multi-date raw batch before publish."""
        frame = self._ensure_raw_aggtrades_frame(rows)
        if frame.is_empty():
            return
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            self.recover_raw_stream(exchange=exchange, symbol=symbol, lease=active)
            dated = frame.with_columns((pl.col("timestamp_ms") // 86_400_000).alias("_utc_day"))
            for slice_frame in dated.partition_by("_utc_day", as_dict=True).values():
                self._preflight_raw_append(
                    exchange=exchange,
                    symbol=symbol,
                    frame=slice_frame.select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS)),
                )
        finally:
            if owns:
                active.release()

    def _append_raw_aggtrades_unlocked(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: pl.DataFrame | list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> int:
        frame = self._ensure_raw_aggtrades_frame(rows)
        if frame.is_empty():
            return 0
        if (
            len({self._partition_date_from_ms(row["timestamp_ms"]) for row in frame.to_dicts()})
            != 1
        ):
            raise ValueError("Raw aggTrades append must contain exactly one UTC date")
        strict_tail = self._preflight_raw_append(exchange=exchange, symbol=symbol, frame=frame)

        stamped = frame.with_columns(
            (pl.col("timestamp_ms") // 1000).cast(pl.Int64).alias("_ts_sec")
        ).with_columns(
            pl.from_epoch(pl.col("_ts_sec"), time_unit="s")
            .dt.strftime("%Y-%m-%d")
            .alias("_partition_date")
        )

        for partition_date, partition in stamped.partition_by(
            "_partition_date", as_dict=True
        ).items():
            part_key = str(
                partition_date[0] if isinstance(partition_date, tuple) else partition_date
            )
            payload = self._validate_raw_aggtrades_frame(
                partition.select(list(_RAW_AGGTRADES_REQUIRED_COLUMNS))
            )
            if payload.is_empty():
                continue
            part_path = self._raw_partition_path(
                exchange=exchange,
                symbol=symbol,
                partition_date=part_key,
            )
            self._ensure_raw_directory(part_path)
            partition_lock = self._acquire_raw_partition_lock(
                exchange=exchange,
                symbol=symbol,
                partition_root=part_path,
            )
            try:
                if strict_tail:
                    entries = self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
                    part_entries = [entry for entry in entries if entry["date"] == part_key]
                    next_index = (
                        max(
                            (
                                int(
                                    _RAW_PART_PATTERN.fullmatch(
                                        entry["name"].rsplit("/", 1)[1]
                                    ).group(1)
                                )
                                for entry in part_entries
                            ),
                            default=-1,
                        )
                        + 1
                    )
                    self._publish_raw_part_set(
                        exchange=exchange,
                        symbol=symbol,
                        partition=part_path,
                        output=part_path / f"part-{next_index:04d}.parquet",
                        frame=payload,
                        obsolete=[],
                    )
                    self._enforce_raw_partition_controls(
                        exchange=exchange, symbol=symbol, partition_root=part_path
                    )
                    continue
                root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
                entries = self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
                existing_entries = [entry for entry in entries if entry["date"] == part_key]
                existing_paths = [root / entry["name"] for entry in existing_entries]
                existing_frames: list[pl.DataFrame] = []
                for entry, existing_path in zip(existing_entries, existing_paths, strict=True):
                    try:
                        existing_frames.append(
                            self._validate_raw_aggtrades_frame(
                                self._raw_read_authenticated_parquet(existing_path, entry)
                            )
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Cannot merge raw aggTrades with unreadable part or invalid data: "
                            f"{existing_path}"
                        ) from exc
                try:
                    existing_state = (
                        self._validate_raw_aggtrades_frame(
                            pl.concat(existing_frames, how="vertical_relaxed")
                        )
                        if existing_frames
                        else self._empty_raw_aggtrades_frame()
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot merge raw aggTrades with conflicting valid parts: {part_path}"
                    ) from exc
                existing_rows_by_id = {
                    row["agg_trade_id"]: row for row in existing_state.to_dicts()
                }
                new_rows: list[dict[str, Any]] = []
                has_overlap = False
                for row in payload.to_dicts():
                    existing_row = existing_rows_by_id.get(row["agg_trade_id"])
                    if existing_row is None:
                        new_rows.append(row)
                    elif existing_row != row:
                        raise ValueError("Raw aggTrades duplicate aggregate ID conflicts")
                    else:
                        has_overlap = True

                existing_tail_id = (
                    int(existing_state["agg_trade_id"][-1])
                    if not existing_state.is_empty()
                    else None
                )
                existing_tail_timestamp = (
                    int(existing_state["timestamp_ms"][-1])
                    if not existing_state.is_empty()
                    else None
                )
                append_only = not has_overlap and (
                    existing_tail_id is None
                    or (
                        int(payload["agg_trade_id"][0]) > existing_tail_id
                        and int(payload["timestamp_ms"][0]) >= existing_tail_timestamp
                    )
                )

                if append_only:
                    output_path = (
                        part_path / "part-0000.parquet"
                        if not existing_paths
                        else self._next_raw_part_path(part_path)
                    )
                    self._publish_raw_part_set(
                        exchange=exchange,
                        symbol=symbol,
                        partition=part_path,
                        output=output_path,
                        frame=payload,
                        obsolete=[],
                    )
                    self._enforce_raw_partition_controls(
                        exchange=exchange,
                        symbol=symbol,
                        partition_root=part_path,
                    )
                    continue

                merged = self._validate_raw_aggtrades_frame(
                    pl.DataFrame(
                        [*existing_state.to_dicts(), *new_rows],
                        schema=_RAW_AGGTRADES_SCHEMA,
                    ).sort("agg_trade_id")
                )

                output_path = part_path / "part-0000.parquet"
                self._publish_raw_part_set(
                    exchange=exchange,
                    symbol=symbol,
                    partition=part_path,
                    output=output_path,
                    frame=merged,
                    obsolete=existing_paths,
                )
                self._enforce_raw_partition_controls(
                    exchange=exchange,
                    symbol=symbol,
                    partition_root=part_path,
                )
            finally:
                partition_lock.release()

        return int(frame.height)

    def read_raw_recovery_bounds(
        self,
        *,
        exchange: str,
        symbol: str,
        checkpoint_last_row: Mapping[str, Any] | None,
        lease: _RawStreamLease | None = None,
    ) -> pl.DataFrame:
        """Return only the authenticated checkpoint binding and current tail."""
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            self.recover_raw_stream(exchange=exchange, symbol=symbol, lease=active)
            entries = self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
            if not entries:
                if checkpoint_last_row is not None:
                    raise ValueError(
                        "Raw aggTrades checkpoint is not bound to persisted raw parquet"
                    )
                return self._empty_raw_aggtrades_frame()
            root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
            selected = [entries[-1]]
            if checkpoint_last_row is not None:
                trade_id = checkpoint_last_row["agg_trade_id"]
                matches = [
                    entry
                    for entry in entries
                    if entry["min_trade_id"] <= trade_id <= entry["max_trade_id"]
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Raw aggTrades checkpoint is not bound to persisted raw parquet"
                    )
                selected.insert(0, matches[0])
            rows: list[dict[str, Any]] = []
            for entry in {entry["name"]: entry for entry in selected}.values():
                path = root / entry["name"]
                frame = self._validate_raw_aggtrades_frame(
                    self._raw_read_authenticated_parquet(path, entry)
                )
                if checkpoint_last_row is not None:
                    rows.extend(
                        row
                        for row in frame.to_dicts()
                        if row["agg_trade_id"] == checkpoint_last_row["agg_trade_id"]
                    )
                if entry is entries[-1]:
                    rows.append(frame.to_dicts()[-1])
            if checkpoint_last_row is not None and not any(
                row == dict(checkpoint_last_row) for row in rows
            ):
                raise ValueError("Raw aggTrades checkpoint is not bound to persisted raw parquet")
            return self._validate_raw_aggtrades_frame(
                pl.DataFrame(rows, schema=_RAW_AGGTRADES_SCHEMA)
            )
        finally:
            if owns:
                active.release()

    def read_raw_recovery_suffix(
        self,
        *,
        exchange: str,
        symbol: str,
        checkpoint_last_row: Mapping[str, Any] | None,
        lease: _RawStreamLease | None = None,
    ) -> pl.DataFrame:
        """Explicit bulk recovery API; steady sync uses bounded recovery bounds."""
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            self.recover_raw_stream(exchange=exchange, symbol=symbol, lease=active)
            entries = self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
            if not entries:
                return self._empty_raw_aggtrades_frame()
            root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
            loaded = self._validate_raw_aggtrades_frame(
                pl.concat(
                    [
                        self._validate_raw_aggtrades_frame(
                            self._raw_read_authenticated_parquet(root / entry["name"], entry)
                        )
                        for entry in entries
                    ],
                    how="vertical_relaxed",
                )
            )
            if checkpoint_last_row is None:
                return self._validate_raw_aggtrades_frame(loaded.tail(1))
            suffix = self._validate_raw_aggtrades_frame(
                loaded.filter(pl.col("agg_trade_id") >= checkpoint_last_row["agg_trade_id"])
            )
            if suffix.is_empty() or suffix.to_dicts()[0] != dict(checkpoint_last_row):
                raise ValueError("Raw aggTrades checkpoint is not bound to persisted raw parquet")
            return suffix
        except Exception as exc:
            raise RuntimeError("Cannot read raw aggTrades checkpoint recovery suffix") from exc
        finally:
            if owns:
                active.release()

    def load_raw_aggtrades(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
        lease: _RawStreamLease | None = None,
    ) -> pl.DataFrame:
        active, owns = self._raw_stream_lease(exchange=exchange, symbol=symbol, lease=lease)
        try:
            self.recover_raw_stream(exchange=exchange, symbol=symbol, lease=active)
            entries = self._authenticate_raw_inventory(exchange=exchange, symbol=symbol)
            start_dt = self._coerce_datetime(start_date)
            end_dt = self._coerce_datetime(end_date)
            if end_dt is not None and start_dt is not None and end_dt < start_dt:
                return self._empty_raw_aggtrades_frame()
            selected = [
                entry
                for entry in entries
                if (
                    start_dt is None
                    or datetime.fromisoformat(entry["date"]).date() >= start_dt.date()
                )
                and (
                    end_dt is None or datetime.fromisoformat(entry["date"]).date() <= end_dt.date()
                )
            ]
            if not selected:
                return self._empty_raw_aggtrades_frame()
            root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
            loaded = self._validate_raw_aggtrades_frame(
                pl.concat(
                    [
                        self._validate_raw_aggtrades_frame(
                            self._raw_read_authenticated_parquet(root / entry["name"], entry)
                        )
                        for entry in selected
                    ],
                    how="vertical_relaxed",
                )
            )
            start_ms = self._datetime_to_ms(start_dt)
            end_ms = self._datetime_to_ms(end_dt)
            if start_ms is not None:
                loaded = loaded.filter(pl.col("timestamp_ms") >= start_ms)
            if end_ms is not None:
                loaded = loaded.filter(pl.col("timestamp_ms") <= end_ms)
            return self._normalize_loaded_raw_aggtrades_frame(loaded)
        finally:
            if owns:
                active.release()

    def _iter_materialized_manifest_paths(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> list[Path]:
        root = (
            self._materialized_symbol_root(exchange=exchange, symbol=symbol)
            / f"timeframe={normalize_timeframe_token(timeframe)}"
        )
        if not root.exists():
            return []
        return sorted(root.glob("date=*/manifest.json"))

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RawFirstManifestInvalidError(f"Manifest JSON must contain an object at {path}.")
        return payload

    @staticmethod
    def _normalize_manifest_data_files(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        out: list[str] = []
        for item in value:
            token = str(item or "").strip()
            if token:
                out.append(token)
        return out

    def _validate_manifest_payload(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
        exchange: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        staleness_threshold_seconds: int | None = None,
    ) -> None:
        manifest_info = os.lstat(manifest_path)
        if not stat.S_ISREG(manifest_info.st_mode):
            raise RawFirstManifestInvalidError(
                f"Manifest path is not a regular file: {manifest_path}"
            )
        missing = [name for name in _MATERIALIZED_REQUIRED_MANIFEST_FIELDS if name not in manifest]
        if missing:
            raise RawFirstManifestInvalidError(
                f"Manifest missing required fields {missing}: {manifest_path}"
            )
        if str(manifest.get("status", "")).strip().lower() != "committed":
            raise RawFirstManifestInvalidError(
                f"Manifest status must be committed: {manifest_path}"
            )

        data_files = self._normalize_manifest_data_files(manifest.get("data_files"))
        if not data_files:
            raise RawFirstManifestInvalidError(f"Manifest data_files is empty: {manifest_path}")

        if exchange is not None:
            expected_exchange = self._normalize_exchange(exchange)
            partition = str(manifest.get("partition", "")).strip()
            if partition and expected_exchange not in partition:
                raise RawFirstManifestInvalidError(
                    f"Manifest exchange mismatch expected={expected_exchange}: {manifest_path}"
                )

        if symbol is not None:
            expected_symbol = self._normalize_symbol_token(symbol)
            actual_symbol = self._normalize_symbol_token(str(manifest.get("symbol", "")))
            if expected_symbol != actual_symbol:
                raise RawFirstManifestInvalidError(
                    f"Manifest symbol mismatch expected={symbol}: {manifest_path}"
                )

        if timeframe is not None:
            expected_tf = normalize_timeframe_token(timeframe)
            actual_tf = normalize_timeframe_token(str(manifest.get("timeframe", "")))
            if expected_tf != actual_tf:
                raise RawFirstManifestInvalidError(
                    f"Manifest timeframe mismatch expected={expected_tf}: {manifest_path}"
                )

        try:
            row_count = int(manifest["row_count"])
            watermark_ms = int(manifest["event_time_watermark_ms"])
            source_checkpoint_start = int(manifest["source_checkpoint_start"])
            source_checkpoint_end = int(manifest["source_checkpoint_end"])
            window_start_ms = int(manifest["window_start_ms"])
            window_end_ms = int(manifest["window_end_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RawFirstManifestInvalidError(
                f"Manifest generation metadata is invalid: {manifest_path}"
            ) from exc
        if (
            row_count < 0
            or watermark_ms <= 0
            or source_checkpoint_start < 0
            or source_checkpoint_end < source_checkpoint_start
            or window_start_ms < 0
            or window_end_ms < window_start_ms
        ):
            raise RawFirstManifestInvalidError(
                f"Manifest generation bounds are invalid: {manifest_path}"
            )
        commit_id = str(manifest.get("commit_id", ""))
        checksum = str(manifest.get("canonical_row_checksum", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", commit_id) or not re.fullmatch(
            r"[0-9a-f]{64}", checksum
        ):
            raise RawFirstManifestInvalidError(
                f"Manifest commit/checksum identity is invalid: {manifest_path}"
            )
        if staleness_threshold_seconds is None:
            return
        now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        lag_ms = max(0, now_ms - watermark_ms)
        if lag_ms > int(staleness_threshold_seconds) * 1000:
            raise RawFirstStaleWindowError(
                f"Manifest stale window detected lag_ms={lag_ms} threshold_s={staleness_threshold_seconds} "
                f"path={manifest_path}",
                symbol=str(manifest.get("symbol", "")),
                timeframe=str(manifest.get("timeframe", "")),
                lag_ms=int(lag_ms),
                commit_id=str(manifest.get("commit_id", "")),
            )

    def _load_committed_manifest_frame(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for data_path in self._resolve_manifest_data_paths(
            manifest=manifest,
            manifest_path=manifest_path,
        ):
            loaded = pl.read_parquet(data_path)
            missing = [
                name for name in _MATERIALIZED_REQUIRED_COLUMNS if name not in loaded.columns
            ]
            if missing:
                raise RawFirstManifestInvalidError(
                    f"Manifest data file missing columns {missing}: {data_path}"
                )
            frames.append(loaded.select(list(_MATERIALIZED_REQUIRED_COLUMNS)))

        if not frames:
            return self._empty_ohlcv_frame()

        merged = (
            pl.concat(frames, how="vertical_relaxed")
            .with_columns(
                [
                    self._coerce_datetime_expr(pl.col("datetime")).alias("datetime"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Float64),
                ]
            )
            .drop_nulls(subset=["datetime"])
            .sort("datetime")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
        )

        row_count = int(manifest.get("row_count", 0))
        if int(merged.height) != row_count:
            raise RawFirstManifestInvalidError(
                f"Manifest row_count mismatch expected={row_count} actual={merged.height} path={manifest_path}"
            )

        expected_checksum = str(manifest.get("canonical_row_checksum", "")).strip()
        actual_checksum = self._canonical_materialized_checksum(merged)
        if expected_checksum != actual_checksum:
            raise RawFirstManifestInvalidError(
                f"Manifest checksum mismatch expected={expected_checksum} actual={actual_checksum} "
                f"path={manifest_path}"
            )
        return merged

    @_generation_guard(exclusive=True)
    def write_materialized_manifest(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        partition_date: str,
        payload: dict[str, Any],
    ) -> Path:
        manifest_path = self._materialized_manifest_path(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            partition_date=partition_date,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(dict(payload or {}), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._fsync_file(tmp_path)
        tmp_path.replace(manifest_path)
        self._fsync_dir(manifest_path.parent)
        return manifest_path

    @_generation_guard(exclusive=False)
    def read_latest_materialized_manifest(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
    ) -> dict[str, Any] | None:
        token = normalize_timeframe_token(timeframe)
        manifests = self._iter_materialized_manifest_paths(
            exchange=exchange,
            symbol=symbol,
            timeframe=token,
        )
        latest_payload: dict[str, Any] | None = None
        latest_key: tuple[int, int, str, str] | None = None

        for manifest_path in manifests:
            payload = self._read_json_file(manifest_path)
            if not isinstance(payload, dict):
                raise RawFirstManifestInvalidError(
                    f"Manifest payload must be an object at {manifest_path}."
                )
            try:
                checkpoint_end = int(payload["source_checkpoint_end"])
                watermark_ms = int(payload["event_time_watermark_ms"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RawFirstManifestInvalidError(
                    f"Manifest generation metadata is invalid at {manifest_path}."
                ) from exc
            commit_id = str(payload.get("commit_id", "") or "")
            if not commit_id:
                raise RawFirstManifestInvalidError(
                    f"Manifest commit_id is missing at {manifest_path}."
                )
            key = (
                int(checkpoint_end),
                int(watermark_ms),
                commit_id,
                str(manifest_path),
            )
            if latest_key is None or key > latest_key:
                latest_key = key
                latest_payload = dict(payload)

        return latest_payload

    @staticmethod
    def _env_int(*, names: list[str], default: int) -> int:
        for name in names:
            raw = os.getenv(str(name), "").strip()
            if not raw:
                continue
            try:
                return int(raw)
            except Exception:
                continue
        return int(default)

    @staticmethod
    def _env_bool(*, names: list[str], default: bool) -> bool:
        for name in names:
            raw = os.getenv(str(name), "").strip().lower()
            if not raw:
                continue
            if raw in {"1", "true", "yes", "on"}:
                return True
            if raw in {"0", "false", "no", "off"}:
                return False
        return bool(default)

    def _resolve_wal_controls(self) -> tuple[int, bool, int]:
        wal_max_bytes = self._env_int(
            names=["LQ_WAL_MAX_BYTES", "LQ__STORAGE__WAL_MAX_BYTES"],
            default=268435456,
        )
        compact_on_threshold = self._env_bool(
            names=["LQ_WAL_COMPACT_ON_THRESHOLD", "LQ__STORAGE__WAL_COMPACT_ON_THRESHOLD"],
            default=True,
        )
        compaction_interval = self._env_int(
            names=[
                "LQ_WAL_COMPACTION_INTERVAL_SEC",
                "LQ_WAL_COMPACTION_INTERVAL_SECONDS",
                "LQ__STORAGE__WAL_COMPACTION_INTERVAL_SEC",
                "LQ__STORAGE__WAL_COMPACTION_INTERVAL_SECONDS",
            ],
            default=3600,
        )
        return (
            max(0, int(wal_max_bytes)),
            bool(compact_on_threshold),
            max(0, int(compaction_interval)),
        )

    @staticmethod
    def _parse_iso_utc(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _enforce_wal_growth_controls(self, *, exchange: str, symbol: str) -> None:
        wal_max_bytes, compact_on_threshold, compaction_interval_seconds = (
            self._resolve_wal_controls()
        )
        if wal_max_bytes <= 0:
            return

        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        if not wal_path.exists():
            return
        wal_size = int(wal_path.stat().st_size)
        if wal_size <= int(wal_max_bytes):
            return

        now = datetime.now(tz=UTC)
        meta = self._read_meta(exchange=exchange, symbol=symbol)
        meta["wal_compaction_required"] = True
        meta["last_wal_over_limit_detected_at"] = now.isoformat()
        last_attempt = self._parse_iso_utc(meta.get("last_compaction_attempt_at"))
        elapsed = None if last_attempt is None else max(0.0, (now - last_attempt).total_seconds())
        if not bool(compact_on_threshold):
            last_warning = self._parse_iso_utc(meta.get("last_wal_over_limit_warning_at"))
            warn_elapsed = None if last_warning is None else (now - last_warning).total_seconds()
            if warn_elapsed is None or warn_elapsed >= 60.0:
                print(
                    f"[WARN] WAL size {wal_size} bytes exceeds limit {wal_max_bytes} for {exchange}:{symbol}. "
                    "wal_compact_on_threshold=false, manual compaction required "
                    "(scripts/compact_wal_to_monthly_parquet.py)."
                )
                meta["last_wal_over_limit_warning_at"] = now.isoformat()
            self._write_meta(exchange=exchange, symbol=symbol, payload=meta)
            return

        can_compact = (
            compaction_interval_seconds <= 0
            or elapsed is None
            or float(elapsed) >= float(compaction_interval_seconds)
        )
        if can_compact:
            print(
                f"[WARN] WAL size {wal_size} bytes exceeds limit {wal_max_bytes} "
                f"for {exchange}:{symbol}. Triggering compaction."
            )
            try:
                self.compact_wal_to_monthly_parquet(
                    exchange=exchange,
                    symbol=symbol,
                    remove_sources=True,
                )
                meta = self._read_meta(exchange=exchange, symbol=symbol)
                meta["wal_compaction_required"] = False
                meta["last_wal_compaction_resolved_at"] = now.isoformat()
                self._write_meta(exchange=exchange, symbol=symbol, payload=meta)
            except Exception as exc:
                print(
                    f"[WARN] WAL compaction trigger failed for {exchange}:{symbol}: {exc}. "
                    "Continuing without blocking writes."
                )
                self._write_meta(exchange=exchange, symbol=symbol, payload=meta)
            return

        last_warning = self._parse_iso_utc(meta.get("last_wal_over_limit_warning_at"))
        warn_elapsed = None if last_warning is None else (now - last_warning).total_seconds()
        if warn_elapsed is None or warn_elapsed >= 60.0:
            wait_seconds = max(0, int(compaction_interval_seconds - int(elapsed or 0)))
            print(
                f"[WARN] WAL size {wal_size} bytes exceeds limit {wal_max_bytes} for {exchange}:{symbol} "
                f"but compaction interval not reached; retry in ~{wait_seconds}s."
            )
            meta["last_wal_over_limit_warning_at"] = now.isoformat()
        self._write_meta(exchange=exchange, symbol=symbol, payload=meta)

    def _read_meta(self, *, exchange: str, symbol: str) -> dict[str, Any]:
        path = self._meta_path(exchange=exchange, symbol=symbol)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Compaction metadata must be an object: {path}")
        return payload

    def _write_meta(self, *, exchange: str, symbol: str, payload: dict[str, Any]) -> None:
        path = self._meta_path(exchange=exchange, symbol=symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._fsync_file(tmp)
        tmp.replace(path)
        self._fsync_dir(path.parent)

    def _read_raw_meta(self, *, exchange: str, symbol: str) -> dict[str, Any]:
        path = self._raw_meta_path(exchange=exchange, symbol=symbol)
        root_fd = self._raw_dir_fd(path.parent, create=False)
        try:
            if _RAW_META_NAME not in self._raw_dir_names(root_fd):
                return {}
            raw_bytes, _ = self._raw_read_bytes(root_fd, _RAW_META_NAME, path=path)
        finally:
            os.close(root_fd)
        try:
            raw = raw_bytes.decode("utf-8")
            payload = json.loads(raw, object_pairs_hook=self._json_object_no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Raw compaction metadata is malformed") from exc
        allowed = {
            "raw_compaction_required",
            "last_raw_over_limit_detected_at",
            "last_raw_over_limit_partition",
            "last_raw_part_count",
            "last_raw_over_limit_warning_at",
            "last_raw_compaction_resolved_at",
            "last_raw_compaction_partition",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) - allowed
            or raw != self._raw_canonical_json(payload)
        ):
            raise ValueError("Raw compaction metadata is malformed")
        if (
            "raw_compaction_required" in payload
            and type(payload["raw_compaction_required"]) is not bool
        ):
            raise ValueError("Raw compaction metadata is malformed")
        for key in (
            "last_raw_over_limit_detected_at",
            "last_raw_over_limit_warning_at",
            "last_raw_compaction_resolved_at",
        ):
            if key in payload and (
                type(payload[key]) is not str or self._parse_iso_utc(payload[key]) is None
            ):
                raise ValueError("Raw compaction metadata is malformed")
        for key in ("last_raw_over_limit_partition", "last_raw_compaction_partition"):
            if key in payload and (
                type(payload[key]) is not str
                or not _RAW_DATE_PATTERN.fullmatch(payload[key].removeprefix("date="))
            ):
                raise ValueError("Raw compaction metadata is malformed")
        if "last_raw_part_count" in payload and (
            type(payload["last_raw_part_count"]) is not int or payload["last_raw_part_count"] < 0
        ):
            raise ValueError("Raw compaction metadata is malformed")
        return payload

    def _assert_no_unbound_raw_control_temps(
        self, *, root: Path, root_fd: int, names: set[str]
    ) -> None:
        for name in sorted(names):
            if _RAW_CONTROL_TEMP_PATTERN.fullmatch(name) is None:
                continue
            try:
                fd = self._raw_open_regular(root_fd, name, os.O_RDONLY, path=root / name)
            except (OSError, ValueError) as exc:
                raise ValueError("unbound raw control temp is present") from exc
            os.close(fd)
            raise ValueError("unbound raw control temp is present")

    def _preflight_raw_meta(self, *, exchange: str, symbol: str) -> None:
        """Reject unsafe persisted raw control state before raw publication begins."""
        try:
            self._read_raw_meta(exchange=exchange, symbol=symbol)
        except FileNotFoundError:
            pass

    def _preflight_raw_controls(self, *, exchange: str, symbol: str) -> None:
        """Bound and parse every existing control file before recovery mutates state."""
        raw_exchange = self._normalize_raw_exchange_token(exchange)
        raw_symbol = self._normalize_raw_symbol_token(symbol)
        checkpoint_symbol = normalize_symbol(symbol)
        root = self._raw_symbol_root(exchange=raw_exchange, symbol=raw_symbol)
        try:
            root_fd = self._raw_dir_fd(root, create=False)
        except FileNotFoundError:
            return
        try:
            names = set(self._raw_dir_names(root_fd))
            self._assert_no_unbound_raw_control_temps(root=root, root_fd=root_fd, names=names)
            if _RAW_INVENTORY_NAME in names:
                raw, _ = self._raw_read_bytes(
                    root_fd, _RAW_INVENTORY_NAME, path=root / _RAW_INVENTORY_NAME
                )
                self._parse_raw_inventory(
                    raw.decode(), exchange=raw_exchange, symbol=raw_symbol, allow_v1=True
                )
            if _RAW_WAL_TAIL_NAME in names:
                self._raw_wal_tail(root=root, root_fd=root_fd)
            if _RAW_WAL_BOOTSTRAP_NAME in names:
                self._raw_wal_bootstrap(root=root, root_fd=root_fd)
            for name in names:
                if not name.startswith("date="):
                    continue
                if _RAW_DATE_PATTERN.fullmatch(name[5:]) is None:
                    raise ValueError("Raw aggTrades partition name is invalid")
                partition = root / name
                partition_fd = self._raw_dir_fd(partition, create=False)
                try:
                    names = set(self._raw_dir_names(partition_fd))
                    descriptor = None
                    if _RAW_TRANSACTION_NAME in names:
                        raw, _ = self._raw_read_bytes(
                            partition_fd,
                            _RAW_TRANSACTION_NAME,
                            path=partition / _RAW_TRANSACTION_NAME,
                        )
                        descriptor = self._parse_raw_transaction(
                            raw.decode(),
                            exchange=raw_exchange,
                            symbol=raw_symbol,
                            partition=partition,
                        )
                    self._assert_raw_transaction_artifact_closure(
                        names=names, descriptor=descriptor
                    )
                finally:
                    os.close(partition_fd)
        finally:
            os.close(root_fd)
        self._read_raw_checkpoint_unlocked(exchange=raw_exchange, symbol=checkpoint_symbol)
        self._preflight_raw_meta(exchange=raw_exchange, symbol=raw_symbol)

    def _write_raw_meta(self, *, exchange: str, symbol: str, payload: dict[str, Any]) -> None:
        encoded = self._raw_control_bytes(payload, label="compaction metadata")
        try:
            self._read_raw_meta(exchange=exchange, symbol=symbol)
        except FileNotFoundError:
            pass
        path = self._raw_meta_path(exchange=exchange, symbol=symbol)
        root_fd = self._raw_dir_fd(path.parent, create=True)
        tmp = f".raw-meta-{uuid.uuid4().hex}.tmp"
        try:
            fd = self._raw_open_regular(
                root_fd, tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, path=path.parent / tmp
            )
            try:
                self._raw_write_all(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
            self._raw_replace(root_fd, tmp, _RAW_META_NAME)
        except Exception:
            try:
                if tmp in set(self._raw_dir_names(root_fd)):
                    self._raw_unlink(root_fd, tmp)
            except OSError, ValueError:
                pass
            raise
        finally:
            os.close(root_fd)

    def _resolve_raw_partition_lock_controls(self) -> tuple[float, float]:
        timeout_raw = str(
            os.getenv("LQ_RAW_PARTITION_LOCK_TIMEOUT_SECONDS", "")
            or os.getenv("LQ__STORAGE__RAW_PARTITION_LOCK_TIMEOUT_SECONDS", "")
        ).strip()
        poll_raw = str(
            os.getenv("LQ_RAW_PARTITION_LOCK_POLL_SECONDS", "")
            or os.getenv("LQ__STORAGE__RAW_PARTITION_LOCK_POLL_SECONDS", "")
        ).strip()
        try:
            timeout_seconds = max(0.1, float(timeout_raw)) if timeout_raw else 10.0
        except ValueError:
            timeout_seconds = 10.0
        try:
            poll_seconds = max(0.01, float(poll_raw)) if poll_raw else 0.05
        except ValueError:
            poll_seconds = 0.05
        return float(timeout_seconds), float(poll_seconds)

    def acquire_raw_symbol_stream_lease(self, *, exchange: str, symbol: str) -> _RawStreamLease:
        timeout_seconds, poll_seconds = self._resolve_raw_partition_lock_controls()
        generation = self.generation_lock(
            exclusive=True,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
        generation.__enter__()
        fd = -1
        try:
            root = self._raw_symbol_root(exchange=exchange, symbol=symbol)
            root_fd = self._raw_dir_fd(root, create=True)
            lock_path = root / ".raw-stream.lock"
            try:
                fd = self._raw_open_regular(
                    root_fd, lock_path.name, os.O_RDWR | os.O_CREAT, path=lock_path
                )
            finally:
                os.close(root_fd)
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return _RawStreamLease(
                        lock_path=lock_path,
                        _fd=fd,
                        _generation_lock=generation,
                    )
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RawPartitionBusyError(f"Raw symbol stream is busy: {root}") from None
                    time.sleep(poll_seconds)
        except BaseException as exc:
            if fd >= 0:
                os.close(fd)
            generation.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def _raw_stream_lease(
        self, *, exchange: str, symbol: str, lease: _RawStreamLease | None
    ) -> tuple[_RawStreamLease, bool]:
        expected = self._raw_symbol_root(exchange=exchange, symbol=symbol) / ".raw-stream.lock"
        if lease is not None:
            if lease.lock_path != expected or lease._released:
                raise ValueError("Raw aggTrades operation requires the matching live stream lease")
            return lease, False
        return self.acquire_raw_symbol_stream_lease(exchange=exchange, symbol=symbol), True

    def _acquire_raw_partition_lock(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_root: Path,
    ) -> _RawStreamLease:
        del exchange, symbol
        parent_fd = self._raw_dir_fd(partition_root, create=True)
        lock_path = partition_root / ".raw-partition.lock"
        try:
            fd = self._raw_open_regular(
                parent_fd,
                lock_path.name,
                os.O_RDWR | os.O_CREAT,
                path=lock_path,
            )
        finally:
            os.close(parent_fd)
        try:
            timeout_seconds, poll_seconds = self._resolve_raw_partition_lock_controls()
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return _RawStreamLease(lock_path=lock_path, _fd=fd)
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RawPartitionBusyError(
                            f"Raw partition is busy: {partition_root}"
                        ) from None
                    time.sleep(poll_seconds)
        except Exception:
            os.close(fd)
            raise

    def _resolve_raw_part_controls(self) -> tuple[int, bool]:
        raw_max_parts = self._env_int(
            names=["LQ_RAW_PARTITION_MAX_PARTS", "LQ__STORAGE__RAW_PARTITION_MAX_PARTS"],
            default=8,
        )
        compact_on_threshold = self._env_bool(
            names=["LQ_RAW_COMPACT_ON_THRESHOLD", "LQ__STORAGE__RAW_COMPACT_ON_THRESHOLD"],
            default=True,
        )
        return max(0, int(raw_max_parts)), bool(compact_on_threshold)

    def _compact_raw_partition(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_root: Path,
    ) -> int:
        self._assert_raw_path_confined(partition_root)
        self._preflight_raw_meta(exchange=exchange, symbol=symbol)
        self._preflight_raw_controls(exchange=exchange, symbol=symbol)
        entries = self._authenticate_raw_inventory(exchange=exchange, symbol=symbol, migrate=True)
        selected = [
            entry for entry in entries if entry["name"].split("/", 1)[0] == partition_root.name
        ]
        part_paths = [partition_root / entry["name"].split("/", 1)[1] for entry in selected]
        if not part_paths:
            return 0

        frames: list[pl.DataFrame] = []
        for path, entry in zip(part_paths, selected, strict=True):
            try:
                loaded = self._validate_raw_aggtrades_frame(
                    self._raw_read_authenticated_parquet(path, entry)
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot compact unreadable part or invalid raw aggTrades data: {path}"
                ) from exc
            if not loaded.is_empty():
                frames.append(loaded)

        if not frames:
            return 0

        try:
            merged = self._validate_raw_aggtrades_frame(pl.concat(frames, how="vertical_relaxed"))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot compact invalid raw aggTrades partition: {partition_root}"
            ) from exc
        output_path = partition_root / "part-0000.parquet"
        self._publish_raw_part_set(
            exchange=exchange,
            symbol=symbol,
            partition=partition_root,
            output=output_path,
            frame=merged,
            obsolete=part_paths,
        )
        return len(part_paths)

    def _enforce_raw_partition_controls(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_root: Path,
    ) -> None:
        max_parts, compact_on_threshold = self._resolve_raw_part_controls()
        if max_parts <= 0:
            return

        part_paths = self._raw_part_paths(partition_root)
        if len(part_paths) <= max_parts:
            return

        now = datetime.now(tz=UTC)
        meta = self._read_raw_meta(exchange=exchange, symbol=symbol)
        meta["raw_compaction_required"] = True
        meta["last_raw_over_limit_detected_at"] = now.isoformat()
        meta["last_raw_over_limit_partition"] = partition_root.name
        meta["last_raw_part_count"] = len(part_paths)

        if not compact_on_threshold:
            last_warning = self._parse_iso_utc(meta.get("last_raw_over_limit_warning_at"))
            warn_elapsed = None if last_warning is None else (now - last_warning).total_seconds()
            if warn_elapsed is None or warn_elapsed >= 60.0:
                print(
                    f"[WARN] Raw partition {partition_root} has {len(part_paths)} parts (limit {max_parts}); "
                    "raw_compact_on_threshold=false, manual compaction required."
                )
                meta["last_raw_over_limit_warning_at"] = now.isoformat()
            self._write_raw_meta(exchange=exchange, symbol=symbol, payload=meta)
            return

        self._compact_raw_partition(exchange=exchange, symbol=symbol, partition_root=partition_root)
        meta = self._read_raw_meta(exchange=exchange, symbol=symbol)
        meta["raw_compaction_required"] = False
        meta["last_raw_compaction_resolved_at"] = now.isoformat()
        meta["last_raw_compaction_partition"] = partition_root.name
        self._write_raw_meta(exchange=exchange, symbol=symbol, payload=meta)

    def _monthly_files_for_range(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[Path]:
        symbol_root = self._symbol_root(exchange=exchange, symbol=symbol)
        all_monthly = sorted(symbol_root.glob("????-??.parquet"))
        if not all_monthly:
            return []

        if start_date is None and end_date is None:
            return all_monthly

        if start_date is None:
            start_date = datetime(1970, 1, 1)
        if end_date is None:
            end_date = datetime(3000, 1, 1)
        months = set(self._iter_month_tokens(start_date, end_date))
        return [path for path in all_monthly if path.stem in months]

    def _load_monthly_frame(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.DataFrame:
        files = self._monthly_files_for_range(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if not files:
            return self._empty_ohlcv_frame()

        lazy = pl.scan_parquet([str(path) for path in files]).select(
            ["datetime", "open", "high", "low", "close", "volume"]
        )
        if start_date is not None:
            lazy = lazy.filter(pl.col("datetime") >= start_date)
        if end_date is not None:
            lazy = lazy.filter(pl.col("datetime") <= end_date)

        out = self._collect_lazy(lazy)
        if out.is_empty():
            return self._empty_ohlcv_frame()
        return out.sort("datetime")

    def _load_wal_frame(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> pl.DataFrame:
        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        if not wal_path.exists():
            return self._empty_ohlcv_frame()

        # Read paths only need the valid prefix. Avoid eagerly repairing/truncating the
        # WAL on every load because large healthy WALs can spend most of the batch budget
        # re-validating bytes that will be decoded again immediately afterwards.
        wal = BinaryWAL(wal_path, auto_repair=False)
        start_ms = self._datetime_to_ms(start_date)
        end_ms = self._datetime_to_ms(end_date)
        records = list(wal.iter_range(start_ms, end_ms))
        if not records:
            return self._empty_ohlcv_frame()

        return pl.DataFrame(
            {
                "datetime": [self._ms_to_datetime(item.ts_ms) for item in records],
                "open": [item.open for item in records],
                "high": [item.high for item in records],
                "low": [item.low for item in records],
                "close": [item.close for item in records],
                "volume": [item.volume for item in records],
                "_seq": list(range(len(records))),
            }
        ).with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))

    def _merge_monthly_and_wal(
        self,
        *,
        monthly: pl.DataFrame,
        wal: pl.DataFrame,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        merge_cols = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "_source_priority",
            "_seq",
        ]

        if not monthly.is_empty():
            frames.append(
                monthly.with_columns(
                    [
                        pl.lit(0).cast(pl.Int8).alias("_source_priority"),
                        pl.int_range(pl.len()).alias("_seq"),
                    ]
                ).select(merge_cols)
            )
        if not wal.is_empty():
            frames.append(
                wal.with_columns(pl.lit(1).cast(pl.Int8).alias("_source_priority")).select(
                    merge_cols
                )
            )

        if not frames:
            return self._empty_ohlcv_frame()

        merged = (
            pl.concat(frames, how="vertical_relaxed")
            .sort(["datetime", "_source_priority", "_seq"])
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
            .drop(["_source_priority", "_seq"], strict=False)
        )
        if merged.is_empty():
            return self._empty_ohlcv_frame()
        return merged.select(["datetime", "open", "high", "low", "close", "volume"])

    @_generation_guard(exclusive=True)
    def upsert_1s(
        self,
        *,
        exchange: str,
        symbol: str,
        rows: pl.DataFrame | list[dict[str, Any]] | list[tuple[Any, ...]],
    ) -> int:
        """Append OHLCV 1s rows into custom binary WAL."""
        frame = self._ensure_ohlcv_frame(rows)
        if frame.is_empty():
            return 0

        exchange_token = self._normalize_exchange(exchange)
        symbol_token = self._normalize_symbol_token(symbol)
        months = {self._month_token(value) for value in frame.get_column("datetime").to_list()}
        with self.generation_lock(exclusive=True):
            for month in months:
                if (
                    self._monthly_pending_path(
                        exchange=exchange_token, symbol=symbol_token, month_token=month
                    ).exists()
                    or self._monthly_seal_path(
                        exchange=exchange_token, symbol=symbol_token, month_token=month
                    ).exists()
                ):
                    raise SealedMonthlyPartitionConflictError(
                        f"Cannot append WAL rows for sealed monthly partition {month}"
                    )
            wal_path = self._wal_path(exchange=exchange_token, symbol=symbol_token)
            fsync_n = max(1, int(os.getenv("LQ_WAL_FSYNC_EVERY_N_BATCHES", "1") or "1"))
            wal = BinaryWAL(wal_path, fsync_every_n_batches=fsync_n, auto_repair=True)
            native_appended = append_ohlcv_frame_native(
                wal.path, frame, fsync_after_write=fsync_n <= 1
            )
            if native_appended is not None:
                appended = int(native_appended)
            else:
                records = [
                    WALRecord(
                        ts_ms=self._datetime_to_ms(item[0]) or 0,
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                    )
                    for item in frame.iter_rows(named=False)
                ]
                appended = int(wal.append(records))
            self._enforce_wal_growth_controls(exchange=exchange_token, symbol=symbol_token)
            return appended

    def _load_ohlcv_1s_merged(
        self,
        *,
        exchange: str,
        symbol: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        start_dt = self._coerce_datetime(start_date)
        end_dt = self._coerce_datetime(end_date)

        monthly = self._load_monthly_frame(
            exchange=exchange,
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
        )
        wal = self._load_wal_frame(
            exchange=exchange,
            symbol=symbol,
            start_date=start_dt,
            end_date=end_dt,
        )
        return self._merge_monthly_and_wal(monthly=monthly, wal=wal)

    @_generation_guard(exclusive=False)
    def load_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
    ) -> pl.DataFrame:
        """Load OHLCV using monthly parquet + WAL merge and bucket resampling."""
        timeframe_token = normalize_timeframe_token(timeframe)
        merged_1s = self._load_ohlcv_1s_merged(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if merged_1s.is_empty():
            return self._empty_ohlcv_frame()

        if timeframe_token == "1s":
            return merged_1s

        tf_ms = int(timeframe_to_milliseconds(timeframe_token))
        source = (
            merged_1s.lazy()
            .with_columns(pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"))
            .with_columns(((pl.col("timestamp_ms") // tf_ms) * tf_ms).alias("bucket_ms"))
        )

        # GPU-friendly aggregation: scalar expressions only, no UDF/group_by_dynamic.
        aggregated = (
            source.group_by("bucket_ms")
            .agg(
                [
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                ]
            )
            .sort("bucket_ms")
            .with_columns(pl.from_epoch("bucket_ms", time_unit="ms").alias("datetime"))
            .select(["datetime", "open", "high", "low", "close", "volume"])
        )
        return self._collect_lazy(aggregated)

    @staticmethod
    def _iter_chunks(
        *,
        start: datetime,
        end: datetime,
        chunk_days: int,
    ) -> list[tuple[datetime, datetime]]:
        if chunk_days <= 0:
            return [(start, end)]
        windows: list[tuple[datetime, datetime]] = []
        cursor = start
        delta = timedelta(days=chunk_days)
        while cursor <= end:
            chunk_end = min(end, cursor + delta - timedelta(microseconds=1))
            windows.append((cursor, chunk_end))
            cursor = chunk_end + timedelta(microseconds=1)
        return windows

    @_generation_guard(exclusive=False)
    def load_ohlcv_chunked(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
        chunk_days: int = 7,
        warmup_bars: int = 0,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        progress_context: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Load timeframe bars by chunk-days windows with optional warmup overlap."""
        start_dt = self._coerce_datetime(start_date)
        end_dt = self._coerce_datetime(end_date)
        if start_dt is None or end_dt is None or start_dt > end_dt:
            return self.load_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

        timeframe_token = normalize_timeframe_token(timeframe)
        tf_ms = int(timeframe_to_milliseconds(timeframe_token))
        warmup_ms = max(0, int(warmup_bars)) * max(1, tf_ms)
        frames: list[pl.DataFrame] = []
        chunk_ranges = list(
            self._iter_chunks(
                start=start_dt,
                end=end_dt,
                chunk_days=max(1, int(chunk_days)),
            )
        )
        progress_base = dict(progress_context or {})
        progress_base.setdefault("symbol", str(symbol))
        progress_base.setdefault("timeframe", str(timeframe_token))
        chunk_count = len(chunk_ranges)
        for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            chunk_started_at = time.perf_counter()
            query_start = (
                chunk_start - timedelta(milliseconds=warmup_ms) if warmup_ms > 0 else chunk_start
            )
            chunk = self.load_ohlcv(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe_token,
                start_date=query_start,
                end_date=chunk_end,
            )
            trimmed = (
                chunk.filter(pl.col("datetime") >= chunk_start) if not chunk.is_empty() else chunk
            )
            trimmed_row_count = int(trimmed.height) if trimmed is not None else 0
            if progress_callback is not None:
                progress_callback(
                    "resource_bundle_symbol_window_loaded",
                    {
                        **progress_base,
                        "unit_kind": "chunk",
                        "unit_index": chunk_index,
                        "unit_count": chunk_count,
                        "window_start": chunk_start.isoformat(),
                        "window_end": chunk_end.isoformat(),
                        "query_start": query_start.isoformat(),
                        "row_count": trimmed_row_count,
                        "elapsed_seconds": round(
                            max(0.0, time.perf_counter() - chunk_started_at),
                            6,
                        ),
                    },
                )
            if not trimmed.is_empty():
                frames.append(trimmed)

        if not frames:
            return self._empty_ohlcv_frame()

        return (
            pl.concat(frames, how="vertical")
            .sort("datetime")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
        )

    def _resolve_manifest_data_paths(
        self,
        *,
        manifest: dict[str, Any],
        manifest_path: Path,
    ) -> list[Path]:
        raw_files = manifest.get("data_files")
        commit_id = str(manifest.get("commit_id", ""))
        if (
            not isinstance(raw_files, list)
            or not raw_files
            or any(not isinstance(item, str) for item in raw_files)
            or len(set(raw_files)) != len(raw_files)
        ):
            raise RawFirstManifestInvalidError(
                f"Manifest data_files must be a unique non-empty list: {manifest_path}"
            )
        out: list[Path] = []
        for raw in raw_files:
            if not isinstance(raw, str):
                raise RawFirstManifestInvalidError(
                    f"Manifest data file token is not text: {manifest_path}"
                )
            relative = PurePosixPath(raw)
            if (
                relative.is_absolute()
                or len(relative.parts) != 2
                or relative.parts[0] != f"commit={commit_id}"
                or not re.fullmatch(r"part-[0-9]+\.parquet", relative.parts[1])
            ):
                raise RawFirstManifestInvalidError(f"Manifest data file escapes its commit: {raw}")
            commit_root = manifest_path.parent / relative.parts[0]
            try:
                commit_info = os.lstat(commit_root)
                data_path = commit_root / relative.parts[1]
                data_info = os.lstat(data_path)
            except FileNotFoundError as exc:
                raise RawFirstDataMissingError(
                    f"Manifest referenced data file missing: {raw}"
                ) from exc
            if not stat.S_ISDIR(commit_info.st_mode) or not stat.S_ISREG(data_info.st_mode):
                raise RawFirstManifestInvalidError(
                    f"Manifest data path is not a regular committed file: {raw}"
                )
            out.append(data_path)
        return out

    def _load_manifest_json(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RawFirstManifestInvalidError(
                f"Failed to parse manifest JSON at {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RawFirstManifestInvalidError(f"Manifest payload must be an object at {path}.")
        return payload

    @_generation_guard(exclusive=False)
    def load_committed_ohlcv_chunked(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Any = None,
        end_date: Any = None,
        chunk_days: int = 7,
        warmup_bars: int = 0,
        staleness_threshold_seconds: int | None = None,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        progress_context: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        token = normalize_timeframe_token(timeframe)
        start_dt = self._coerce_datetime(start_date)
        end_dt = self._coerce_datetime(end_date)
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            return self._empty_ohlcv_frame()

        query_start = start_dt
        if start_dt is not None and int(warmup_bars) > 0:
            tf_ms = int(timeframe_to_milliseconds(token))
            query_start = start_dt - timedelta(milliseconds=max(0, int(warmup_bars)) * tf_ms)

        manifests = self._iter_materialized_manifest_paths(
            exchange=exchange,
            symbol=symbol,
            timeframe=token,
        )
        if query_start is not None or end_dt is not None:
            bounded: list[Path] = []
            lower = query_start.date() if query_start is not None else date(1970, 1, 1)
            upper = end_dt.date() if end_dt is not None else date(3000, 1, 1)
            for path in manifests:
                parent = str(path.parent.name)
                if not parent.startswith("date="):
                    raise RawFirstManifestInvalidError(
                        f"Materialized partition name is invalid: {path.parent}"
                    )
                try:
                    part_date = date.fromisoformat(parent.split("=", 1)[1])
                except ValueError as exc:
                    raise RawFirstManifestInvalidError(
                        f"Materialized partition date is invalid: {path.parent}"
                    ) from exc
                if lower <= part_date <= upper:
                    bounded.append(path)
            manifests = bounded

        if not manifests:
            if token != "1s":
                source_1s = self.load_committed_ohlcv_chunked(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe="1s",
                    start_date=query_start,
                    end_date=end_dt,
                    chunk_days=chunk_days,
                    warmup_bars=warmup_bars,
                    staleness_threshold_seconds=staleness_threshold_seconds,
                )
                if source_1s.is_empty():
                    raise RawFirstDataMissingError(
                        f"No committed 1s rows available to rebuild {exchange}:{symbol}:{token}."
                    )
                latest_1s_ms = int(
                    source_1s.select(pl.col("datetime").dt.epoch("ms").max().alias("ts")).item()
                    or 0
                )
                rebuilt = resample_1s_frame(
                    source_1s,
                    timeframe=token,
                    complete_through_ms=int(latest_1s_ms) + 999,
                ).sort("datetime")
                if end_dt is not None:
                    rebuilt = rebuilt.filter(pl.col("datetime") <= end_dt)
                if start_dt is not None and int(warmup_bars) <= 0:
                    rebuilt = rebuilt.filter(pl.col("datetime") >= start_dt)
                if rebuilt.is_empty():
                    raise RawFirstDataMissingError(
                        f"Committed 1s rows exist but could not rebuild {exchange}:{symbol}:{token}."
                    )
                return rebuilt
            raise RawFirstDataMissingError(
                f"No committed manifests found for {exchange}:{symbol}:{token}."
            )

        frames: list[pl.DataFrame] = []
        newest_watermark_ms: int | None = None
        newest_commit_id: str | None = None
        progress_base = dict(progress_context or {})
        progress_base.setdefault("symbol", str(symbol))
        progress_base.setdefault("timeframe", str(token))
        manifest_count = len(manifests)
        for manifest_index, manifest_path in enumerate(manifests, start=1):
            manifest_started_at = time.perf_counter()
            manifest = self._read_json_file(manifest_path)
            self._validate_manifest_payload(
                manifest=manifest,
                manifest_path=manifest_path,
                exchange=exchange,
                symbol=symbol,
                timeframe=token,
                staleness_threshold_seconds=staleness_threshold_seconds,
            )
            frame = self._load_committed_manifest_frame(
                manifest=manifest,
                manifest_path=manifest_path,
            )
            if progress_callback is not None:
                progress_callback(
                    "resource_bundle_symbol_window_loaded",
                    {
                        **progress_base,
                        "unit_kind": "manifest",
                        "unit_index": manifest_index,
                        "unit_count": manifest_count,
                        "partition": str(manifest_path.parent.name),
                        "row_count": int(frame.height),
                        "elapsed_seconds": round(
                            max(0.0, time.perf_counter() - manifest_started_at),
                            6,
                        ),
                    },
                )
            if not frame.is_empty():
                frames.append(frame)
            watermark_ms = int(manifest["event_time_watermark_ms"])
            if watermark_ms > 0 and (
                newest_watermark_ms is None or watermark_ms >= newest_watermark_ms
            ):
                newest_watermark_ms = int(watermark_ms)
                newest_commit_id = str(manifest.get("commit_id", "") or "")

        if not frames:
            raise RawFirstDataMissingError(
                f"Committed manifests found but no rows for {exchange}:{symbol}:{token}."
            )

        merged = (
            pl.concat(frames, how="vertical_relaxed")
            .sort("datetime")
            .unique(subset=["datetime"], keep="last")
            .sort("datetime")
        )
        if query_start is not None:
            merged = merged.filter(pl.col("datetime") >= query_start)
        if end_dt is not None:
            merged = merged.filter(pl.col("datetime") <= end_dt)
        if start_dt is not None and int(warmup_bars) <= 0:
            merged = merged.filter(pl.col("datetime") >= start_dt)

        if merged.is_empty():
            raise RawFirstDataMissingError(
                f"No committed OHLCV rows in range for {exchange}:{symbol}:{token}."
            )

        if (
            staleness_threshold_seconds is not None
            and int(staleness_threshold_seconds) > 0
            and newest_watermark_ms is not None
        ):
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            lag_ms = max(0, int(now_ms - newest_watermark_ms))
            if lag_ms > int(staleness_threshold_seconds) * 1000:
                raise RawFirstStaleWindowError(
                    "Committed window stale for "
                    f"{exchange}:{symbol}:{token}: lag_ms={lag_ms} threshold_ms={int(staleness_threshold_seconds) * 1000}.",
                    symbol=str(symbol),
                    timeframe=str(token),
                    lag_ms=int(lag_ms),
                    commit_id=newest_commit_id,
                )

        return merged.select(_OHLCV_COLUMNS)

    @_generation_guard(exclusive=True)
    def compact_wal_to_monthly_parquet(
        self,
        *,
        exchange: str,
        symbol: str,
        remove_sources: bool = True,
    ) -> list[CompactionResult]:
        """Compact WAL records into monthly parquet files atomically."""
        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        if not wal_path.exists():
            return []

        wal = BinaryWAL(wal_path, auto_repair=True)
        wal_size = wal.size_bytes()

        meta = self._read_meta(exchange=exchange, symbol=symbol)
        offset = int(meta.get("wal_offset", 0) or 0)
        if offset < 0 or offset > wal_size:
            offset = 0

        records = list(wal.iter_records_from_offset(offset))
        if not records:
            return []

        by_month: dict[str, list[WALRecord]] = {}
        for record in records:
            by_month.setdefault(self._month_token_from_ms(record.ts_ms), []).append(record)

        results: list[CompactionResult] = []
        for month_token in sorted(by_month):
            monthly_path = self._monthly_path(
                exchange=exchange, symbol=symbol, month_token=month_token
            )
            pending = self._monthly_pending_path(
                exchange=exchange, symbol=symbol, month_token=month_token
            )
            seal = self._monthly_seal_path(
                exchange=exchange, symbol=symbol, month_token=month_token
            )
            with self._monthly_lock(exchange=exchange, symbol=symbol, month_token=month_token):
                if pending.exists() or seal.exists():
                    raise SealedMonthlyPartitionConflictError(
                        f"Cannot compact WAL into sealed monthly partition {month_token}"
                    )
                monthly_path.parent.mkdir(parents=True, exist_ok=True)
                existing = (
                    pl.read_parquet(monthly_path)
                    if monthly_path.exists()
                    else self._empty_ohlcv_frame()
                )
                incoming_rows = by_month[month_token]
                incoming = pl.DataFrame(
                    {
                        "datetime": [self._ms_to_datetime(item.ts_ms) for item in incoming_rows],
                        "open": [item.open for item in incoming_rows],
                        "high": [item.high for item in incoming_rows],
                        "low": [item.low for item in incoming_rows],
                        "close": [item.close for item in incoming_rows],
                        "volume": [item.volume for item in incoming_rows],
                        "_seq": list(range(len(incoming_rows))),
                    }
                ).with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))

                merged = self._merge_monthly_and_wal(monthly=existing, wal=incoming)
                tmp_path = monthly_path.with_suffix(".tmp.parquet")
                merged.write_parquet(tmp_path, compression="zstd", statistics=True)
                self._fsync_file(tmp_path)
                tmp_path.replace(monthly_path)
                self._fsync_dir(monthly_path.parent)

                results.append(
                    CompactionResult(
                        partition=str(monthly_path),
                        files_before=1 if existing.height > 0 else 0,
                        files_after=1,
                        rows_before=int(existing.height + incoming.height),
                        rows_after=int(merged.height),
                    )
                )

        if remove_sources:
            wal.truncate()
            next_offset = 0
        else:
            next_offset = wal.size_bytes()

        self._write_meta(
            exchange=exchange,
            symbol=symbol,
            payload={
                **meta,
                "wal_offset": int(next_offset),
                "updated_at": datetime.now(tz=UTC).isoformat(),
                "last_compaction_attempt_at": datetime.now(tz=UTC).isoformat(),
                "compacted_rows": len(records),
                "remove_sources": bool(remove_sources),
            },
        )
        return results

    @_generation_guard(exclusive=True)
    def compact_partition(
        self,
        *,
        exchange: str,
        symbol: str,
        partition_date: str | date,
        timeframe: str = "1s",
        remove_sources: bool = True,
    ) -> CompactionResult:
        """Compatibility wrapper: compact WAL and return the requested month summary."""
        _ = timeframe
        if isinstance(partition_date, str):
            resolved = date.fromisoformat(partition_date)
        else:
            resolved = partition_date
        month_token = f"{resolved.year:04d}-{resolved.month:02d}"
        results = self.compact_wal_to_monthly_parquet(
            exchange=exchange,
            symbol=symbol,
            remove_sources=remove_sources,
        )
        for result in results:
            if Path(result.partition).stem == month_token:
                return result
        monthly_path = self._monthly_path(exchange=exchange, symbol=symbol, month_token=month_token)
        return CompactionResult(str(monthly_path), 0, int(monthly_path.exists()), 0, 0)

    @_generation_guard(exclusive=True)
    def compact_all(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str = "1s",
        remove_sources: bool = True,
    ) -> list[CompactionResult]:
        """Compact every WAL-backed month for one symbol."""
        if normalize_timeframe_token(timeframe) != "1s":
            return []
        return self.compact_wal_to_monthly_parquet(
            exchange=exchange,
            symbol=symbol,
            remove_sources=remove_sources,
        )

    @_generation_guard(exclusive=False)
    def get_symbol_time_range(
        self,
        *,
        exchange: str,
        symbol: str,
    ) -> tuple[datetime | None, datetime | None]:
        """Return min/max datetime across monthly parquet + WAL for one symbol."""
        monthly_files = self._monthly_files_for_range(
            exchange=exchange,
            symbol=symbol,
            start_date=None,
            end_date=None,
        )

        min_dt: datetime | None = None
        max_dt: datetime | None = None

        if monthly_files:
            first = (
                pl.scan_parquet(str(monthly_files[0]))
                .select(pl.col("datetime").min().alias("min_dt"))
                .collect()
            )
            last = (
                pl.scan_parquet(str(monthly_files[-1]))
                .select(pl.col("datetime").max().alias("max_dt"))
                .collect()
            )
            left = first["min_dt"][0]
            right = last["max_dt"][0]
            if left is not None:
                min_dt = left
            if right is not None:
                max_dt = right

        wal_path = self._wal_path(exchange=exchange, symbol=symbol)
        if wal_path.exists():
            # Metadata scans are read-only; iterate the valid prefix without forcing a
            # full repair pass on every open.
            wal = BinaryWAL(wal_path, auto_repair=False)
            first_wal: WALRecord | None = None
            last_wal: WALRecord | None = None
            for record in wal.iter_all():
                if first_wal is None:
                    first_wal = record
                last_wal = record
            if first_wal is not None:
                first_dt = self._ms_to_datetime(first_wal.ts_ms)
                min_dt = first_dt if min_dt is None else min(min_dt, first_dt)
            if last_wal is not None:
                last_dt = self._ms_to_datetime(last_wal.ts_ms)
                max_dt = last_dt if max_dt is None else max(max_dt, last_dt)

        return min_dt, max_dt


def is_parquet_market_data_store(path: str, *, backend: str | None = None) -> bool:
    """Heuristic detector for parquet-backed market storage path."""
    backend_token = str(backend or "").strip().lower()
    if backend_token in {"parquet", "local", "postgres_parquet", "parquet_postgres"}:
        return True
    raw = str(path or "").strip()
    if not raw:
        return False
    if raw.lower().endswith(".parquet"):
        return True
    if "parquet" in Path(raw).name.lower():
        return True
    repo = ParquetMarketDataRepository(Path(raw))
    with repo.generation_lock(exclusive=False):
        root = repo.root_path
        if not root.exists():
            return False
        return bool(
            any(root.glob("market_ohlcv_1s/*/*/*.parquet"))
            or any(root.glob("market_ohlcv_1s/*/*/wal.bin"))
            or any(root.glob("market_data_raw_aggtrades/*/*/date=*/part-*.parquet"))
            or any(root.glob("market_data_materialized/*/*/timeframe=*/date=*/manifest.json"))
            or any(root.glob("exchange=*/symbol=*/timeframe=*/date=*/*.parquet"))
        )


def _load_data_dict_from_parquet_pinned(
    *,
    repo: ParquetMarketDataRepository,
    resolved_mode: str,
    exchange: str,
    symbol_list: list[str],
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
    chunk_days: int = 7,
    warmup_bars: int = 0,
    staleness_threshold_seconds: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, pl.DataFrame]:
    """Compatibility entrypoint retained for legacy import sites."""
    out: dict[str, pl.DataFrame] = {}
    missing_symbols: list[str] = []

    symbols = list(symbol_list or [])
    symbol_count = len(symbols)
    for symbol_index, symbol in enumerate(symbols, start=1):
        progress_base = {
            "symbol": str(symbol),
            "symbol_index": symbol_index,
            "symbol_count": symbol_count,
            "timeframe": str(timeframe),
            "data_mode": str(resolved_mode),
        }
        if progress_callback is not None:
            progress_callback(
                "resource_bundle_symbol_fetch_started",
                progress_base,
            )
        symbol_started_at = time.perf_counter()
        if resolved_mode == "raw-first":
            try:
                frame = repo.load_committed_ohlcv_chunked(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    chunk_days=chunk_days,
                    warmup_bars=warmup_bars,
                    staleness_threshold_seconds=staleness_threshold_seconds,
                    progress_callback=progress_callback,
                    progress_context=progress_base,
                )
            except RawFirstDataMissingError:
                from lumina_quant.services.materialize_from_raw import (
                    _materialize_raw_aggtrades_bundle_unlocked,
                )

                _materialize_raw_aggtrades_bundle_unlocked(
                    repo=repo,
                    exchange=exchange,
                    symbol=str(symbol),
                    timeframes=[str(timeframe)],
                    start_date=start_date,
                    end_date=end_date,
                    producer="load_data_dict_from_parquet",
                    require_complete=True,
                )
                frame = repo.load_committed_ohlcv_chunked(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    chunk_days=chunk_days,
                    warmup_bars=warmup_bars,
                    staleness_threshold_seconds=staleness_threshold_seconds,
                    progress_callback=progress_callback,
                    progress_context=progress_base,
                )
        else:
            frame = repo.load_ohlcv_chunked(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                chunk_days=chunk_days,
                warmup_bars=warmup_bars,
                progress_callback=progress_callback,
                progress_context=progress_base,
            )

        row_count = int(frame.height) if frame is not None else 0
        if progress_callback is not None:
            progress_callback(
                "resource_bundle_symbol_fetch_completed",
                {
                    **progress_base,
                    "row_count": row_count,
                    "was_missing": bool(frame is None or frame.is_empty()),
                    "elapsed_seconds": round(
                        max(0.0, time.perf_counter() - symbol_started_at),
                        6,
                    ),
                },
            )
        if frame is None or frame.is_empty():
            missing_symbols.append(str(symbol))
            continue
        out[str(symbol)] = frame

    if resolved_mode == "raw-first" and missing_symbols:
        raise RawFirstDataMissingError(
            "Raw-first committed data missing for symbols: " + ", ".join(missing_symbols)
        )
    return out


def load_data_dict_from_parquet(
    root_path: str,
    *,
    exchange: str,
    symbol_list: list[str],
    timeframe: str,
    start_date: Any = None,
    end_date: Any = None,
    chunk_days: int = 7,
    warmup_bars: int = 0,
    data_mode: str = "legacy",
    staleness_threshold_seconds: int | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, pl.DataFrame]:
    """Load one logical generation for the complete multi-symbol request."""
    repo = ParquetMarketDataRepository(root_path)
    resolved_mode = normalize_data_mode(data_mode, default="legacy")
    with repo.generation_lock(exclusive=resolved_mode == "raw-first"):
        return _load_data_dict_from_parquet_pinned(
            repo=repo,
            resolved_mode=resolved_mode,
            exchange=exchange,
            symbol_list=symbol_list,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            chunk_days=chunk_days,
            warmup_bars=warmup_bars,
            staleness_threshold_seconds=staleness_threshold_seconds,
            progress_callback=progress_callback,
        )
