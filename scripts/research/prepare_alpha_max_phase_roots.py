"""Materialize immutable, phase-owned Alpha-Max parquet roots.

The preparation step is intentionally separate from the Alpha-Max runners.  It
copies no files: every output parquet is decoded, clipped to its half-open
phase/availability interval, and written as a new regular file.  Publication is
an atomic rename into an output path that must not exist.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import secrets
import stat
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import polars as pl

from lumina_quant.symbols import canonical_symbol


EXCHANGE: Final = "binance"
CANDIDATE_SYMBOLS: Final = (
    "ADAUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "BTCUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "TONUSDT",
    "TRXUSDT",
    "XRPUSDT",
)
PHASE_INTERVALS: Final = {
    "warmup": (
        datetime(2022, 12, 31, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC),
    ),
    "train": (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2025, 6, 1, tzinfo=UTC),
    ),
    "purge": (
        datetime(2025, 6, 1, tzinfo=UTC),
        datetime(2025, 6, 8, tzinfo=UTC),
    ),
    "validation": (
        datetime(2025, 6, 8, tzinfo=UTC),
        datetime(2025, 8, 31, tzinfo=UTC),
    ),
    "embargo": (
        datetime(2025, 8, 31, tzinfo=UTC),
        datetime(2025, 9, 7, tzinfo=UTC),
    ),
    "historical_exposed_evaluation": (
        datetime(2025, 9, 7, tzinfo=UTC),
        datetime(2026, 7, 1, tzinfo=UTC),
    ),
}

_PHASE_IDS: Final = tuple(PHASE_INTERVALS)
_CONTRACT_SCHEMA: Final = "alpha_max_contract_manifest.v2"
_APPROVED_CONTRACT_MANIFEST_SHA256: Final = (
    "ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220"
)
_PREPARATION_SCHEMA: Final = "alpha_max_phase_root_preparation_manifest.v1"
_FUNDING_INTERVAL_MS: Final = 8 * 60 * 60 * 1000
_TON_FUNDING_INTERVAL_MS: Final = 4 * 60 * 60 * 1000
_FUNDING_SOURCE_MAX_JITTER_MS: Final = 1000
_RAW_INTERVAL_MS: Final = 1000
_RAW_OHLCV_COLUMNS: Final = ("open", "high", "low", "close", "volume")
_READ_BLOCK_BYTES: Final = 1024 * 1024
_RENAME_NOREPLACE: Final = 1


@dataclass(frozen=True, slots=True)
class SymbolAvailability:
    """Frozen raw and feature availability boundaries for one candidate."""

    symbol: str
    raw_start_utc: datetime
    raw_end_utc: datetime
    feature_start_utc: datetime
    feature_end_utc: datetime


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Content identity captured while a regular source file is held open."""

    relative_path: str
    sha256: str
    byte_count: int
    identity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FrozenTreeSnapshot:
    """Descriptor-observed identities and contents retained through publish."""

    directory_identities: dict[str, tuple[int, int, int]]
    file_fingerprints: dict[str, SourceFingerprint]


@dataclass(slots=True)
class DirectoryCapability:
    """One descriptor-pinned canonical directory capability."""

    path: Path
    descriptor: int
    identity: tuple[int, int, int]

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> DirectoryCapability:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("alpha_max_preparation_datetime_not_utc")
    return value.isoformat().replace("+00:00", "Z")


def _parse_canonical_utc(value: object, *, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"alpha_max_{field}_invalid")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"alpha_max_{field}_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or _iso_utc(parsed) != value:
        raise ValueError(f"alpha_max_{field}_invalid")
    return parsed


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _file_object_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
    )


def _fingerprint_object_identity(fingerprint: SourceFingerprint) -> tuple[int, int, int]:
    return (
        fingerprint.identity[0],
        fingerprint.identity[1],
        fingerprint.identity[2],
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
    )


def _canonical_absolute(path: Path, *, field: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or Path(os.path.normpath(candidate)) != candidate:
        raise ValueError(f"alpha_max_{field}_must_be_absolute_canonical")
    return candidate


def _pin_canonical_directory(path: Path, *, field: str) -> DirectoryCapability:
    candidate = _canonical_absolute(path, field=field)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(os.path.sep, flags)
    try:
        for part in candidate.parts[1:]:
            try:
                observed = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"alpha_max_{field}_missing") from exc
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"alpha_max_{field}_symlink_rejected")
            if not stat.S_ISDIR(observed.st_mode):
                raise ValueError(f"alpha_max_{field}_not_directory")
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ValueError(f"alpha_max_{field}_open_failed") from exc
            opened = os.fstat(child)
            if _directory_identity(observed) != _directory_identity(opened):
                os.close(child)
                raise ValueError(f"alpha_max_{field}_changed_during_open")
            os.close(descriptor)
            descriptor = child
        opened = os.fstat(descriptor)
        return DirectoryCapability(
            path=candidate,
            descriptor=descriptor,
            identity=_directory_identity(opened),
        )
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_pinned_directory(capability: DirectoryCapability, *, field: str) -> None:
    with _pin_canonical_directory(capability.path, field=field) as rebound:
        if rebound.identity != capability.identity:
            raise ValueError(f"alpha_max_{field}_path_changed")


def _safe_relative_parts(relative_path: str, *, field: str) -> tuple[str, ...]:
    path = Path(relative_path)
    if (
        not relative_path
        or "\0" in relative_path
        or "\\" in relative_path
        or path.is_absolute()
        or path.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"alpha_max_{field}_relative_path_invalid")
    return path.parts


def _open_relative_directory(
    root: DirectoryCapability,
    relative_parts: tuple[str, ...],
    *,
    field: str,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.dup(root.descriptor)
    try:
        for part in relative_parts:
            try:
                observed = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"alpha_max_{field}_missing") from exc
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"alpha_max_{field}_symlink_rejected")
            if not stat.S_ISDIR(observed.st_mode):
                raise ValueError(f"alpha_max_{field}_parent_not_directory")
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as exc:
                raise ValueError(f"alpha_max_{field}_open_failed") from exc
            opened = os.fstat(child)
            if _directory_identity(observed) != _directory_identity(opened):
                os.close(child)
                raise ValueError(f"alpha_max_{field}_changed_during_open")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _target_is_absent(parent: DirectoryCapability, target_name: str) -> bool:
    try:
        os.stat(target_name, dir_fd=parent.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise ValueError("alpha_max_output_root_stat_failed") from exc
    return False


def _require_absent_target(path: Path) -> tuple[Path, DirectoryCapability]:
    target = _canonical_absolute(path, field="output_root")
    parent = _pin_canonical_directory(target.parent, field="output_parent")
    if not _target_is_absent(parent, target.name):
        parent.close()
        raise ValueError("alpha_max_output_root_must_be_absent")
    return target, parent


def _read_regular_bytes(
    relative_path: str,
    *,
    root: DirectoryCapability,
    field: str,
) -> tuple[bytes, SourceFingerprint]:
    parts = _safe_relative_parts(relative_path, field=field)
    parent_fd = _open_relative_directory(root, parts[:-1], field=field)
    filename = parts[-1]
    try:
        try:
            observed = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"alpha_max_{field}_missing:{relative_path}") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"alpha_max_{field}_symlink_rejected:{relative_path}")
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"alpha_max_{field}_not_regular:{relative_path}")
        if int(observed.st_nlink) != 1:
            raise ValueError(f"alpha_max_{field}_hardlink_rejected:{relative_path}")

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(filename, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ValueError(f"alpha_max_{field}_read_failed:{relative_path}") from exc
    finally:
        os.close(parent_fd)

    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(observed):
            raise ValueError(f"alpha_max_{field}_changed_during_open:{relative_path}")
        digest = hashlib.sha256()
        payload = bytearray()
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            payload.extend(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if _file_identity(after) != _file_identity(opened):
            raise ValueError(f"alpha_max_{field}_changed_during_read:{relative_path}")
        if len(payload) != int(opened.st_size):
            raise ValueError(f"alpha_max_{field}_size_mismatch:{relative_path}")
    except OSError as exc:
        raise ValueError(f"alpha_max_{field}_read_failed:{relative_path}") from exc
    finally:
        os.close(descriptor)

    frozen = bytes(payload)
    return frozen, SourceFingerprint(
        relative_path=relative_path,
        sha256=digest.hexdigest(),
        byte_count=len(frozen),
        identity=_file_identity(opened),
    )


def _read_parquet(
    relative_path: str,
    *,
    root: DirectoryCapability,
    field: str,
) -> tuple[pl.DataFrame, SourceFingerprint]:
    payload, fingerprint = _read_regular_bytes(relative_path, root=root, field=field)
    try:
        frame = pl.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"alpha_max_{field}_parquet_invalid:{fingerprint.relative_path}") from exc
    if frame.is_empty():
        raise ValueError(f"alpha_max_{field}_parquet_empty:{fingerprint.relative_path}")
    return frame, fingerprint


def _read_contract_manifest(
    manifest_path: Path,
) -> tuple[tuple[SymbolAvailability, ...], str]:
    canonical = _canonical_absolute(manifest_path, field="contract_manifest")
    with _pin_canonical_directory(
        canonical.parent,
        field="contract_manifest_parent",
    ) as manifest_parent:
        payload, fingerprint = _read_regular_bytes(
            canonical.name,
            root=manifest_parent,
            field="contract_manifest",
        )
        _revalidate_pinned_directory(
            manifest_parent,
            field="contract_manifest_parent",
        )
    if fingerprint.sha256 != _APPROVED_CONTRACT_MANIFEST_SHA256:
        raise ValueError("alpha_max_contract_manifest_unapproved")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("alpha_max_contract_manifest_json_invalid") from exc
    if type(document) is not dict or _canonical_json_bytes(document) != payload:
        raise ValueError("alpha_max_contract_manifest_not_canonical")
    if document.get("schema_version") != _CONTRACT_SCHEMA or document.get("exchange") != EXCHANGE:
        raise ValueError("alpha_max_contract_manifest_identity_invalid")
    records = document.get("records")
    if type(records) is not list:
        raise ValueError("alpha_max_contract_manifest_records_invalid")
    observed_symbols = tuple(
        record.get("symbol") if type(record) is dict else None for record in records
    )
    if observed_symbols != CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_contract_manifest_symbols_invalid")

    availability: list[SymbolAvailability] = []
    for record in records:
        assert type(record) is dict
        symbol = record["symbol"]
        availability.append(
            SymbolAvailability(
                symbol=symbol,
                raw_start_utc=_parse_canonical_utc(
                    record.get("raw_availability_start_utc"),
                    field=f"{symbol.lower()}_raw_availability_start_utc",
                ),
                raw_end_utc=_parse_canonical_utc(
                    record.get("raw_availability_end_utc"),
                    field=f"{symbol.lower()}_raw_availability_end_utc",
                ),
                feature_start_utc=_parse_canonical_utc(
                    record.get("feature_availability_start_utc"),
                    field=f"{symbol.lower()}_feature_availability_start_utc",
                ),
                feature_end_utc=_parse_canonical_utc(
                    record.get("feature_availability_end_utc"),
                    field=f"{symbol.lower()}_feature_availability_end_utc",
                ),
            )
        )
        if (
            availability[-1].raw_start_utc >= availability[-1].raw_end_utc
            or availability[-1].feature_start_utc >= availability[-1].feature_end_utc
        ):
            raise ValueError(f"alpha_max_{symbol.lower()}_availability_interval_invalid")
    return tuple(availability), fingerprint.sha256


def _validate_phase_intervals() -> None:
    if tuple(PHASE_INTERVALS) != _PHASE_IDS:
        raise ValueError("alpha_max_phase_ids_changed")
    predecessor_end: datetime | None = None
    for phase_id in _PHASE_IDS:
        start, end = PHASE_INTERVALS[phase_id]
        if (
            start.tzinfo is None
            or start.utcoffset() != timedelta(0)
            or end.tzinfo is None
            or end.utcoffset() != timedelta(0)
            or start >= end
            or (predecessor_end is not None and start != predecessor_end)
        ):
            raise ValueError("alpha_max_phase_intervals_invalid")
        predecessor_end = end


def _month_floor(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1)
    return value.replace(month=value.month + 1)


def _day_floor(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _check_exact_symbol_exchange(frame: pl.DataFrame, *, symbol: str, field: str) -> None:
    if "symbol" in frame.columns:
        values = frame.get_column("symbol")
        expected = canonical_symbol(symbol)
        observed = {canonical_symbol(value) for value in values.cast(pl.String).to_list()}
        if values.null_count() or not expected or observed != {expected}:
            raise ValueError(f"alpha_max_{field}_symbol_mismatch")
    if "exchange" in frame.columns:
        values = frame.get_column("exchange")
        if values.null_count() or {
            str(value).lower() for value in values.cast(pl.String).to_list()
        } != {EXCHANGE}:
            raise ValueError(f"alpha_max_{field}_exchange_mismatch")


def _validate_strict_timestamps(timestamps: pl.Series, *, field: str) -> None:
    if timestamps.is_empty() or timestamps.null_count() or timestamps.n_unique() != len(timestamps):
        raise ValueError(f"alpha_max_{field}_timestamp_duplicate_or_null")
    diffs = timestamps.diff().drop_nulls()
    if not diffs.is_empty() and int(diffs.min()) <= 0:
        raise ValueError(f"alpha_max_{field}_timestamp_not_strictly_increasing")


def _validate_raw_ohlcv(frame: pl.DataFrame) -> None:
    schema = frame.schema
    if any(
        column not in schema or not schema[column].is_numeric() for column in _RAW_OHLCV_COLUMNS
    ):
        raise ValueError("alpha_max_raw_source_ohlcv_schema_invalid")
    normalized = frame.select(
        pl.col(column).cast(pl.Float64).alias(column) for column in _RAW_OHLCV_COLUMNS
    )
    if any(
        values.null_count() or not bool(values.is_finite().all())
        for values in (normalized.get_column(column) for column in _RAW_OHLCV_COLUMNS)
    ):
        raise ValueError("alpha_max_raw_source_ohlcv_value_invalid")
    if any(
        bool((normalized.get_column(column) <= 0.0).any())
        for column in ("open", "high", "low", "close")
    ):
        raise ValueError("alpha_max_raw_source_ohlc_nonpositive")
    if bool((normalized.get_column("volume") < 0.0).any()):
        raise ValueError("alpha_max_raw_source_volume_negative")
    if not normalized.filter(
        (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
        | (pl.col("high") < pl.col("low"))
    ).is_empty():
        raise ValueError("alpha_max_raw_source_ohlcv_relation_invalid")


def _funding_interval_ms(symbol: str) -> int:
    if symbol not in CANDIDATE_SYMBOLS:
        raise ValueError("alpha_max_feature_symbol_outside_candidates")
    return _TON_FUNDING_INTERVAL_MS if symbol == "TONUSDT" else _FUNDING_INTERVAL_MS


def _first_grid_boundary_ms(start_ms: int, interval_ms: int) -> int:
    return ((start_ms + interval_ms - 1) // interval_ms) * interval_ms


def _expected_grid_timestamps(start_ms: int, end_ms: int, interval_ms: int) -> list[int]:
    first = _first_grid_boundary_ms(start_ms, interval_ms)
    return list(range(first, end_ms, interval_ms))


def _prepare_raw_frame(
    frame: pl.DataFrame,
    *,
    symbol: str,
    partition_start: datetime,
    partition_end: datetime,
    owned_start: datetime,
    owned_end: datetime,
) -> pl.DataFrame:
    schema = frame.schema
    if "datetime" not in schema or not isinstance(schema["datetime"], pl.Datetime):
        raise ValueError("alpha_max_raw_source_timestamp_schema_invalid")
    _check_exact_symbol_exchange(frame, symbol=symbol, field="raw_source")
    _validate_raw_ohlcv(frame)
    source_datetimes = frame.get_column("datetime")
    if bool((source_datetimes.dt.nanosecond() != 0).fill_null(False).any()):
        raise ValueError("alpha_max_raw_source_timestamp_subsecond_invalid")
    with_timestamps = frame.with_columns(
        pl.col("datetime").dt.epoch("ms").alias("__alpha_max_timestamp_ms")
    )
    timestamps = with_timestamps.get_column("__alpha_max_timestamp_ms")
    _validate_strict_timestamps(timestamps, field="raw_source")
    if any(int(value) % 1000 for value in timestamps):
        raise ValueError("alpha_max_raw_source_timestamp_alignment_invalid")
    partition_start_ms = _epoch_ms(partition_start)
    partition_end_ms = _epoch_ms(partition_end)
    if int(timestamps.min()) < partition_start_ms or int(timestamps.max()) >= partition_end_ms:
        raise ValueError("alpha_max_raw_source_partition_bounds_invalid")

    clipped = with_timestamps.filter(
        (pl.col("__alpha_max_timestamp_ms") >= _epoch_ms(owned_start))
        & (pl.col("__alpha_max_timestamp_ms") < _epoch_ms(owned_end))
    )
    owned_start_ms = _epoch_ms(owned_start)
    owned_end_ms = _epoch_ms(owned_end)
    if (
        owned_start_ms % _RAW_INTERVAL_MS
        or owned_end_ms % _RAW_INTERVAL_MS
        or owned_end_ms <= owned_start_ms
    ):
        raise ValueError("alpha_max_raw_owned_interval_alignment_invalid")
    expected_row_count = (owned_end_ms - owned_start_ms) // _RAW_INTERVAL_MS
    clipped_timestamps = clipped.get_column("__alpha_max_timestamp_ms")
    clipped_diffs = clipped_timestamps.diff().drop_nulls()
    if (
        clipped.height != expected_row_count
        or clipped.is_empty()
        or int(clipped_timestamps[0]) != owned_start_ms
        or int(clipped_timestamps[-1]) != owned_end_ms - _RAW_INTERVAL_MS
        or (
            not clipped_diffs.is_empty()
            and (
                int(clipped_diffs.min()) != _RAW_INTERVAL_MS
                or int(clipped_diffs.max()) != _RAW_INTERVAL_MS
            )
        )
    ):
        raise ValueError("alpha_max_raw_owned_interval_not_exact_1s")
    return clipped.drop("__alpha_max_timestamp_ms")


def _prepare_feature_frame(
    frame: pl.DataFrame,
    *,
    symbol: str,
    partition_start: datetime,
    partition_end: datetime,
    owned_start: datetime,
    owned_end: datetime,
) -> pl.DataFrame:
    schema = frame.schema
    if "timestamp_ms" not in schema or not schema["timestamp_ms"].is_integer():
        raise ValueError("alpha_max_feature_source_timestamp_schema_invalid")
    if "funding_rate" not in schema or not schema["funding_rate"].is_numeric():
        raise ValueError("alpha_max_feature_source_funding_schema_invalid")
    _validate_strict_timestamps(
        frame.get_column("timestamp_ms").cast(pl.Int64),
        field="feature_source",
    )
    funding = frame.filter(
        pl.col("funding_rate").is_not_null() & pl.col("funding_rate").is_finite()
    )
    if funding.is_empty():
        raise ValueError("alpha_max_feature_source_funding_empty")
    _check_exact_symbol_exchange(funding, symbol=symbol, field="feature_source")
    source_timestamps = funding.get_column("timestamp_ms").cast(pl.Int64)
    _validate_strict_timestamps(source_timestamps, field="feature_source")
    if int(source_timestamps.min()) < _epoch_ms(partition_start) or int(
        source_timestamps.max()
    ) >= _epoch_ms(partition_end):
        raise ValueError("alpha_max_feature_source_partition_bounds_invalid")

    interval_ms = _funding_interval_ms(symbol)
    source_values = [int(value) for value in source_timestamps]
    canonical_values = [(value // interval_ms) * interval_ms for value in source_values]
    jitter_values = [
        source - canonical
        for source, canonical in zip(source_values, canonical_values, strict=True)
    ]
    if any(jitter < 0 or jitter > _FUNDING_SOURCE_MAX_JITTER_MS for jitter in jitter_values):
        raise ValueError("alpha_max_feature_source_timestamp_jitter_invalid")
    normalized = funding.with_columns(
        pl.Series("source_timestamp_ms", source_values, dtype=pl.Int64),
        pl.Series("__alpha_max_canonical_timestamp_ms", canonical_values, dtype=pl.Int64),
    )
    canonical = normalized.get_column("__alpha_max_canonical_timestamp_ms")
    _validate_strict_timestamps(canonical, field="feature_canonical")
    if int(canonical.min()) < _epoch_ms(partition_start) or int(canonical.max()) >= _epoch_ms(
        partition_end
    ):
        raise ValueError("alpha_max_feature_canonical_partition_bounds_invalid")

    owned_start_ms = _epoch_ms(owned_start)
    owned_end_ms = _epoch_ms(owned_end)
    clipped = normalized.filter(
        (pl.col("__alpha_max_canonical_timestamp_ms") >= owned_start_ms)
        & (pl.col("__alpha_max_canonical_timestamp_ms") < owned_end_ms)
    ).select(
        pl.col("__alpha_max_canonical_timestamp_ms").alias("timestamp_ms"),
        pl.col("source_timestamp_ms"),
        pl.lit(EXCHANGE, dtype=pl.String).alias("exchange"),
        pl.lit(symbol, dtype=pl.String).alias("symbol"),
        pl.col("funding_rate").cast(pl.Float64),
    )
    expected_timestamps = _expected_grid_timestamps(owned_start_ms, owned_end_ms, interval_ms)
    if (
        not expected_timestamps
        or clipped.get_column("timestamp_ms").to_list() != expected_timestamps
    ):
        raise ValueError("alpha_max_feature_funding_canonical_coverage_invalid")
    rates = clipped.get_column("funding_rate").to_list()
    if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in rates):
        raise ValueError("alpha_max_feature_funding_value_invalid")
    clipped_timestamps = clipped.get_column("timestamp_ms")
    _validate_strict_timestamps(clipped_timestamps, field="feature_output")
    source_output = clipped.get_column("source_timestamp_ms")
    _validate_strict_timestamps(source_output, field="feature_source_output")
    if any(
        not owned_start_ms <= source < owned_end_ms
        or source - settlement < 0
        or source - settlement > _FUNDING_SOURCE_MAX_JITTER_MS
        for source, settlement in zip(
            source_output.to_list(),
            clipped_timestamps.to_list(),
            strict=True,
        )
    ):
        raise ValueError("alpha_max_feature_output_timestamp_jitter_invalid")
    return clipped


def _fingerprint_owned_output(
    descriptor: int,
    *,
    relative_path: str,
) -> SourceFingerprint:
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
        raise ValueError("alpha_max_output_file_identity_invalid")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("alpha_max_output_file_read_failed") from exc
    if _file_identity(after) != _file_identity(opened):
        raise ValueError("alpha_max_output_file_changed_during_read")
    if byte_count != int(opened.st_size):
        raise ValueError("alpha_max_output_file_size_mismatch")
    return SourceFingerprint(
        relative_path=relative_path,
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        identity=_file_identity(opened),
    )


def _create_owned_directory_at(parent_fd: int, name: str) -> tuple[int, tuple[int, int, int]]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    created_name: str | None = None
    for _ in range(64):
        candidate = f".{name}.mkdir-{secrets.token_hex(8)}"
        try:
            os.mkdir(candidate, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            continue
        created_name = candidate
        break
    if created_name is None:
        raise ValueError("alpha_max_output_owned_directory_name_exhausted")

    child_fd: int | None = None
    try:
        created_observed = os.stat(
            created_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(created_observed.st_mode) or stat.S_ISLNK(created_observed.st_mode):
            raise ValueError("alpha_max_output_owned_directory_identity_invalid")
        child_fd = os.open(created_name, directory_flags, dir_fd=parent_fd)
        created_identity = _directory_identity(os.fstat(child_fd))
        if created_identity != _directory_identity(created_observed):
            raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
        try:
            _rename_noreplace_at(
                _load_renameat2(),
                parent_fd,
                created_name,
                parent_fd,
                name,
            )
        except OSError as exc:
            raise ValueError("alpha_max_output_owned_directory_publish_failed") from exc
        rebound = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _directory_identity(rebound) != created_identity:
            raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
        return child_fd, created_identity
    except Exception:
        if child_fd is not None:
            os.close(child_fd)
        raise


def _open_owned_output_parent(
    directory_fd: int,
    relative_parts: tuple[str, ...],
    *,
    snapshot: FrozenTreeSnapshot,
) -> int:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = os.dup(directory_fd)
    try:
        root_identity = _directory_identity(os.fstat(current_fd))
        if snapshot.directory_identities.get("") != root_identity:
            raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
        traversed: list[str] = []
        for part in relative_parts:
            traversed.append(part)
            relative = "/".join(traversed)
            expected_identity = snapshot.directory_identities.get(relative)
            try:
                observed = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            except FileNotFoundError:
                if expected_identity is not None:
                    raise ValueError("alpha_max_output_owned_directory_missing") from None
                child_fd, created_identity = _create_owned_directory_at(current_fd, part)
                snapshot.directory_identities[relative] = created_identity
            except OSError as exc:
                raise ValueError("alpha_max_output_owned_directory_stat_failed") from exc
            else:
                if (
                    expected_identity is None
                    or not stat.S_ISDIR(observed.st_mode)
                    or stat.S_ISLNK(observed.st_mode)
                    or _directory_identity(observed) != expected_identity
                ):
                    raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
                try:
                    child_fd = os.open(part, directory_flags, dir_fd=current_fd)
                except OSError as exc:
                    raise ValueError("alpha_max_output_owned_directory_open_failed") from exc
                if _directory_identity(os.fstat(child_fd)) != expected_identity:
                    os.close(child_fd)
                    raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
                rebound = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
                if _directory_identity(rebound) != expected_identity:
                    os.close(child_fd)
                    raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_owned_output(
    directory_fd: int,
    *,
    relative_path: str,
    snapshot: FrozenTreeSnapshot,
) -> int:
    parts = _safe_relative_parts(relative_path, field="output_owned_file")
    parent_fd = _open_owned_output_parent(
        directory_fd,
        parts[:-1],
        snapshot=snapshot,
    )
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(parts[-1], flags, 0o600, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError("alpha_max_output_file_create_failed") from exc
    finally:
        os.close(parent_fd)
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
        os.close(descriptor)
        raise ValueError("alpha_max_output_file_identity_invalid")
    return descriptor


def _write_output_parquet(
    frame: pl.DataFrame,
    *,
    temporary_fd: int,
    relative_path: str,
    ownership_snapshot: FrozenTreeSnapshot,
) -> SourceFingerprint:
    descriptor = _open_owned_output(
        temporary_fd,
        relative_path=relative_path,
        snapshot=ownership_snapshot,
    )
    try:
        with os.fdopen(os.dup(descriptor), "wb") as output:
            frame.write_parquet(output, compression="zstd", statistics=True)
            output.flush()
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        return _fingerprint_owned_output(
            descriptor,
            relative_path=relative_path,
        )
    finally:
        os.close(descriptor)


def _write_owned_bytes(
    payload: bytes,
    *,
    temporary_fd: int,
    relative_path: str,
    ownership_snapshot: FrozenTreeSnapshot,
) -> SourceFingerprint:
    descriptor = _open_owned_output(
        temporary_fd,
        relative_path=relative_path,
        snapshot=ownership_snapshot,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise ValueError("alpha_max_output_file_write_failed")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        return _fingerprint_owned_output(
            descriptor,
            relative_path=relative_path,
        )
    except OSError as exc:
        raise ValueError("alpha_max_output_file_write_failed") from exc
    finally:
        os.close(descriptor)


def _manifest_entry(
    *,
    phase_id: str,
    root_kind: str,
    symbol: str,
    owned_start: datetime,
    owned_end: datetime,
    output_relative_path: str,
    output_sha256: str,
    output_byte_count: int,
    output_row_count: int,
    source: SourceFingerprint,
) -> dict[str, Any]:
    return {
        "output_byte_count": output_byte_count,
        "output_relative_path": output_relative_path,
        "output_row_count": output_row_count,
        "output_sha256": output_sha256,
        "owned_end_utc": _iso_utc(owned_end),
        "owned_start_utc": _iso_utc(owned_start),
        "phase_id": phase_id,
        "root_kind": root_kind,
        "source_byte_count": source.byte_count,
        "source_relative_path": source.relative_path,
        "source_sha256": source.sha256,
        "symbol": symbol,
    }


def _feature_source_path(
    feature_root: DirectoryCapability,
    *,
    symbol: str,
    day: datetime,
) -> str:
    relative = f"exchange={EXCHANGE}/symbol={symbol}/date={day:%Y-%m-%d}"
    parts = _safe_relative_parts(relative, field="feature_partition")
    directory_fd = _open_relative_directory(feature_root, parts, field="feature_partition")
    try:
        with os.scandir(directory_fd) as iterator:
            names = tuple(sorted(entry.name for entry in iterator))
        if len(names) != 1 or not names[0].endswith(".parquet"):
            raise ValueError(f"alpha_max_feature_partition_inventory_invalid:{relative}")
        observed = os.stat(names[0], dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"alpha_max_feature_partition_symlink_rejected:{relative}")
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"alpha_max_feature_partition_inventory_invalid:{relative}")
        return f"{relative}/{names[0]}"
    finally:
        os.close(directory_fd)


def _validate_feature_sequence(
    bounds: list[tuple[int, int, int]],
    *,
    effective_start: datetime,
    effective_end: datetime,
    symbol: str,
    phase_id: str,
) -> None:
    if not bounds:
        raise ValueError(f"alpha_max_feature_partition_missing:{phase_id}:{symbol}")
    interval_ms = _funding_interval_ms(symbol)
    expected = _expected_grid_timestamps(
        _epoch_ms(effective_start),
        _epoch_ms(effective_end),
        interval_ms,
    )
    observed_count = sum(row_count for _, _, row_count in bounds)
    if (
        not expected
        or observed_count != len(expected)
        or bounds[0][0] != expected[0]
        or bounds[-1][1] != expected[-1]
        or any(right[0] - left[1] != interval_ms for left, right in pairwise(bounds))
    ):
        raise ValueError(f"alpha_max_feature_funding_coverage_incomplete:{phase_id}:{symbol}")


def _materialize_raw(
    *,
    temporary_root: Path,
    temporary_fd: int,
    ownership_snapshot: FrozenTreeSnapshot,
    raw_root: DirectoryCapability,
    availability: tuple[SymbolAvailability, ...],
    entries: list[dict[str, Any]],
    expected_outputs: set[str],
) -> None:
    for phase_id, (phase_start, phase_end) in PHASE_INTERVALS.items():
        for contract in availability:
            effective_start = max(phase_start, contract.raw_start_utc)
            effective_end = min(phase_end, contract.raw_end_utc)
            if effective_start >= effective_end:
                continue
            bounds: list[tuple[int, int, int]] = []
            month = _month_floor(effective_start)
            while month < effective_end:
                month_end = _next_month(month)
                owned_start = max(effective_start, month)
                owned_end = min(effective_end, month_end)
                source_path = f"{EXCHANGE}/{contract.symbol}/{month:%Y-%m}.parquet"
                frame, source = _read_parquet(
                    source_path,
                    root=raw_root,
                    field="raw_partition",
                )
                output = (
                    temporary_root
                    / phase_id
                    / "raw"
                    / "market_ohlcv_1s"
                    / EXCHANGE
                    / contract.symbol
                    / f"{month:%Y-%m}.parquet"
                )
                clipped = _prepare_raw_frame(
                    frame,
                    symbol=contract.symbol,
                    partition_start=month,
                    partition_end=month_end,
                    owned_start=owned_start,
                    owned_end=owned_end,
                )
                clipped_timestamps = clipped.get_column("datetime").dt.epoch("ms")
                bounds.append(
                    (
                        int(clipped_timestamps[0]),
                        int(clipped_timestamps[-1]),
                        clipped.height,
                    )
                )
                relative = output.relative_to(temporary_root).as_posix()
                output_fingerprint = _write_output_parquet(
                    clipped,
                    temporary_fd=temporary_fd,
                    relative_path=relative,
                    ownership_snapshot=ownership_snapshot,
                )
                _record_owned_file_descriptor(
                    temporary_fd,
                    output_fingerprint,
                    snapshot=ownership_snapshot,
                )
                expected_outputs.add(relative)
                entries.append(
                    _manifest_entry(
                        phase_id=phase_id,
                        root_kind="raw",
                        symbol=contract.symbol,
                        owned_start=owned_start,
                        owned_end=owned_end,
                        output_relative_path=relative,
                        output_sha256=output_fingerprint.sha256,
                        output_byte_count=output_fingerprint.byte_count,
                        output_row_count=clipped.height,
                        source=source,
                    )
                )
                month = month_end
            _validate_raw_sequence(
                bounds,
                effective_start=effective_start,
                effective_end=effective_end,
                symbol=contract.symbol,
                phase_id=phase_id,
            )


def _validate_raw_sequence(
    bounds: list[tuple[int, int, int]],
    *,
    effective_start: datetime,
    effective_end: datetime,
    symbol: str,
    phase_id: str,
) -> None:
    start_ms = _epoch_ms(effective_start)
    end_ms = _epoch_ms(effective_end)
    if start_ms % _RAW_INTERVAL_MS or end_ms % _RAW_INTERVAL_MS or start_ms >= end_ms:
        raise ValueError(f"alpha_max_raw_owned_interval_alignment_invalid:{phase_id}:{symbol}")
    expected_count = (end_ms - start_ms) // _RAW_INTERVAL_MS
    if (
        not bounds
        or sum(row_count for _, _, row_count in bounds) != expected_count
        or bounds[0][0] != start_ms
        or bounds[-1][1] != end_ms - _RAW_INTERVAL_MS
        or any(right[0] - left[1] != _RAW_INTERVAL_MS for left, right in pairwise(bounds))
    ):
        raise ValueError(f"alpha_max_raw_1s_coverage_incomplete:{phase_id}:{symbol}")


def _materialize_feature(
    *,
    temporary_root: Path,
    temporary_fd: int,
    ownership_snapshot: FrozenTreeSnapshot,
    feature_root: DirectoryCapability,
    availability: tuple[SymbolAvailability, ...],
    entries: list[dict[str, Any]],
    expected_outputs: set[str],
) -> None:
    for phase_id, (phase_start, phase_end) in PHASE_INTERVALS.items():
        for contract in availability:
            effective_start = max(phase_start, contract.feature_start_utc)
            effective_end = min(phase_end, contract.feature_end_utc)
            if effective_start >= effective_end:
                continue
            day = _day_floor(effective_start)
            bounds: list[tuple[int, int, int]] = []
            while day < effective_end:
                day_end = day + timedelta(days=1)
                owned_start = max(effective_start, day)
                owned_end = min(effective_end, day_end)
                source_path = _feature_source_path(
                    feature_root,
                    symbol=contract.symbol,
                    day=day,
                )
                frame, source = _read_parquet(
                    source_path,
                    root=feature_root,
                    field="feature_partition",
                )
                output = (
                    temporary_root
                    / phase_id
                    / "feature"
                    / "feature_points"
                    / f"exchange={EXCHANGE}"
                    / f"symbol={contract.symbol}"
                    / f"date={day:%Y-%m-%d}"
                    / "part-0.parquet"
                )
                clipped = _prepare_feature_frame(
                    frame,
                    symbol=contract.symbol,
                    partition_start=day,
                    partition_end=day_end,
                    owned_start=owned_start,
                    owned_end=owned_end,
                )
                timestamps = clipped.get_column("timestamp_ms")
                bounds.append((int(timestamps.min()), int(timestamps.max()), clipped.height))
                relative = output.relative_to(temporary_root).as_posix()
                output_fingerprint = _write_output_parquet(
                    clipped,
                    temporary_fd=temporary_fd,
                    relative_path=relative,
                    ownership_snapshot=ownership_snapshot,
                )
                _record_owned_file_descriptor(
                    temporary_fd,
                    output_fingerprint,
                    snapshot=ownership_snapshot,
                )
                expected_outputs.add(relative)
                entries.append(
                    _manifest_entry(
                        phase_id=phase_id,
                        root_kind="feature",
                        symbol=contract.symbol,
                        owned_start=owned_start,
                        owned_end=owned_end,
                        output_relative_path=relative,
                        output_sha256=output_fingerprint.sha256,
                        output_byte_count=output_fingerprint.byte_count,
                        output_row_count=clipped.height,
                        source=source,
                    )
                )
                day = day_end
            _validate_feature_sequence(
                bounds,
                effective_start=effective_start,
                effective_end=effective_end,
                symbol=contract.symbol,
                phase_id=phase_id,
            )


def _verify_output_inventory(temporary_root: Path, expected_outputs: set[str]) -> None:
    actual: set[str] = set()
    for phase_id in _PHASE_IDS:
        for root_kind in ("raw", "feature"):
            root = temporary_root / phase_id / root_kind
            if not root.is_dir() or root.is_symlink():
                raise ValueError("alpha_max_output_root_layout_invalid")
            for path in root.rglob("*"):
                observed = path.lstat()
                if stat.S_ISLNK(observed.st_mode):
                    raise ValueError("alpha_max_output_symlink_rejected")
                if stat.S_ISDIR(observed.st_mode):
                    continue
                if not stat.S_ISREG(observed.st_mode) or int(observed.st_nlink) != 1:
                    raise ValueError("alpha_max_output_file_identity_invalid")
                actual.add(path.relative_to(temporary_root).as_posix())
    if actual != expected_outputs:
        raise ValueError("alpha_max_output_partition_inventory_mismatch")


def _create_temporary_root(
    parent: DirectoryCapability,
    *,
    target_name: str,
) -> tuple[str, int, Path]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(64):
        temporary_name = f".{target_name}.prepare-{secrets.token_hex(8)}"
        try:
            os.mkdir(temporary_name, mode=0o700, dir_fd=parent.descriptor)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(temporary_name, flags, dir_fd=parent.descriptor)
        except Exception:
            # No safe compare-by-inode-and-rmdir primitive exists.  Retain the
            # private random entry rather than risk deleting a raced replacement.
            raise
        return temporary_name, descriptor, Path(f"/proc/self/fd/{descriptor}")
    raise ValueError("alpha_max_output_temporary_name_exhausted")


def _freeze_tree_descriptor(
    directory_fd: int,
    *,
    expected_snapshot: FrozenTreeSnapshot,
) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    observed_directories: set[str] = set()
    observed_files: set[str] = set()

    def walk(current_fd: int, relative_parts: tuple[str, ...]) -> None:
        relative_directory = "/".join(relative_parts)
        directory_before = os.fstat(current_fd)
        expected_directory_identity = expected_snapshot.directory_identities.get(relative_directory)
        if not stat.S_ISDIR(
            directory_before.st_mode
        ) or expected_directory_identity != _directory_identity(directory_before):
            raise ValueError("alpha_max_output_freeze_directory_identity_mismatch")
        observed_directories.add(relative_directory)
        with os.scandir(current_fd) as iterator:
            names = tuple(sorted(entry.name for entry in iterator))
        for name in names:
            relative_parts_child = (*relative_parts, name)
            relative = "/".join(relative_parts_child)
            observed = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError("alpha_max_output_freeze_symlink_rejected")
            if stat.S_ISDIR(observed.st_mode):
                expected_identity = expected_snapshot.directory_identities.get(relative)
                if expected_identity != _directory_identity(observed):
                    raise ValueError("alpha_max_output_freeze_directory_identity_mismatch")
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=current_fd)
                except OSError as exc:
                    raise ValueError("alpha_max_output_freeze_directory_open_failed") from exc
                try:
                    if _directory_identity(os.fstat(child_fd)) != expected_identity:
                        raise ValueError("alpha_max_output_freeze_directory_identity_mismatch")
                    walk(child_fd, relative_parts_child)
                    rebound = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                    if _directory_identity(rebound) != expected_identity:
                        raise ValueError("alpha_max_output_freeze_directory_identity_mismatch")
                finally:
                    os.close(child_fd)
                continue
            expected_fingerprint = expected_snapshot.file_fingerprints.get(relative)
            if (
                expected_fingerprint is None
                or not stat.S_ISREG(observed.st_mode)
                or int(observed.st_nlink) != 1
                or _file_identity(observed) != expected_fingerprint.identity
            ):
                raise ValueError("alpha_max_output_freeze_file_identity_mismatch")
            try:
                file_fd = os.open(name, file_flags, dir_fd=current_fd)
            except OSError as exc:
                raise ValueError("alpha_max_output_freeze_file_open_failed") from exc
            try:
                opened = os.fstat(file_fd)
                if _file_identity(opened) != expected_fingerprint.identity:
                    raise ValueError("alpha_max_output_freeze_file_identity_mismatch")
                os.fchmod(file_fd, 0o444)
                os.fsync(file_fd)
                frozen = os.fstat(file_fd)
                if (
                    _file_object_identity(frozen)
                    != _fingerprint_object_identity(expected_fingerprint)
                    or not stat.S_ISREG(frozen.st_mode)
                    or int(frozen.st_nlink) != 1
                    or stat.S_IMODE(frozen.st_mode) != 0o444
                ):
                    raise ValueError("alpha_max_output_freeze_file_mode_invalid")
                rebound = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                if _file_identity(rebound) != _file_identity(frozen):
                    raise ValueError("alpha_max_output_freeze_file_identity_mismatch")
            finally:
                os.close(file_fd)
            observed_files.add(relative)
        os.fchmod(current_fd, 0o555)
        os.fsync(current_fd)
        directory_after = os.fstat(current_fd)
        if (
            _directory_identity(directory_after) != expected_directory_identity
            or stat.S_IMODE(directory_after.st_mode) != 0o555
        ):
            raise ValueError("alpha_max_output_freeze_directory_mode_invalid")

    walk(directory_fd, ())
    if observed_directories != set(expected_snapshot.directory_identities) or observed_files != set(
        expected_snapshot.file_fingerprints
    ):
        raise ValueError("alpha_max_output_freeze_inventory_mismatch")


def _expected_frozen_directories(expected_files: set[str]) -> set[str]:
    expected = {""}
    expected.update(
        f"{phase_id}/{root_kind}" for phase_id in _PHASE_IDS for root_kind in ("raw", "feature")
    )
    expected.update(_PHASE_IDS)
    for relative in expected_files:
        parts = Path(relative).parts[:-1]
        expected.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    return expected


def _verify_frozen_tree_descriptor(
    directory_fd: int,
    *,
    expected_content: dict[str, tuple[int, str]],
    expected_preparation_bytes: bytes,
    expected_snapshot: FrozenTreeSnapshot | None = None,
) -> FrozenTreeSnapshot:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    observed_files: set[str] = set()
    observed_directories: dict[str, tuple[int, int, int]] = {}

    def walk(current_fd: int, relative_parts: tuple[str, ...]) -> None:
        current = os.fstat(current_fd)
        relative_directory = "/".join(relative_parts)
        if not stat.S_ISDIR(current.st_mode) or stat.S_IMODE(current.st_mode) != 0o555:
            raise ValueError("alpha_max_output_frozen_directory_mode_invalid")
        current_identity = _directory_identity(current)
        if expected_snapshot is not None:
            expected_identity = expected_snapshot.directory_identities.get(relative_directory)
            if expected_identity != current_identity:
                raise ValueError("alpha_max_output_frozen_directory_identity_mismatch")
        observed_directories[relative_directory] = current_identity
        with os.scandir(current_fd) as iterator:
            names = tuple(sorted(entry.name for entry in iterator))
        for name in names:
            observed = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            relative = "/".join((*relative_parts, name))
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError("alpha_max_output_frozen_symlink_rejected")
            if stat.S_ISDIR(observed.st_mode):
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=current_fd)
                except OSError as exc:
                    raise ValueError("alpha_max_output_frozen_directory_open_failed") from exc
                try:
                    opened_identity = _directory_identity(os.fstat(child_fd))
                    if opened_identity != _directory_identity(observed):
                        raise ValueError("alpha_max_output_frozen_directory_identity_mismatch")
                    walk(child_fd, (*relative_parts, name))
                    rebound = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                    if _directory_identity(rebound) != opened_identity:
                        raise ValueError("alpha_max_output_frozen_directory_identity_mismatch")
                finally:
                    os.close(child_fd)
                continue
            if (
                not stat.S_ISREG(observed.st_mode)
                or int(observed.st_nlink) != 1
                or stat.S_IMODE(observed.st_mode) != 0o444
            ):
                raise ValueError("alpha_max_output_frozen_file_mode_invalid")
            try:
                file_fd = os.open(name, file_flags, dir_fd=current_fd)
            except OSError as exc:
                raise ValueError("alpha_max_output_frozen_file_open_failed") from exc
            try:
                opened = os.fstat(file_fd)
                if _file_identity(opened) != _file_identity(observed):
                    raise ValueError("alpha_max_output_frozen_file_identity_mismatch")
            finally:
                os.close(file_fd)
            observed_files.add(relative)

    walk(directory_fd, ())
    expected_files = set(expected_content)
    if observed_files != expected_files or set(
        observed_directories
    ) != _expected_frozen_directories(expected_files):
        raise ValueError("alpha_max_output_frozen_inventory_mismatch")
    if expected_snapshot is not None and (
        set(expected_snapshot.file_fingerprints) != expected_files
        or set(expected_snapshot.directory_identities) != set(observed_directories)
    ):
        raise ValueError("alpha_max_output_frozen_inventory_mismatch")

    root = DirectoryCapability(
        path=Path(f"/proc/self/fd/{directory_fd}"),
        descriptor=directory_fd,
        identity=_directory_identity(os.fstat(directory_fd)),
    )
    observed_fingerprints: dict[str, SourceFingerprint] = {}
    for relative in sorted(expected_files):
        payload, fingerprint = _read_regular_bytes(
            relative,
            root=root,
            field="output_frozen_file",
        )
        expected_byte_count, expected_sha256 = expected_content[relative]
        if fingerprint.byte_count != expected_byte_count:
            raise ValueError(f"alpha_max_output_frozen_file_byte_count_mismatch:{relative}")
        if fingerprint.sha256 != expected_sha256:
            raise ValueError(f"alpha_max_output_frozen_file_sha256_mismatch:{relative}")
        if expected_snapshot is not None:
            expected_fingerprint = expected_snapshot.file_fingerprints[relative]
            if (
                _fingerprint_object_identity(fingerprint)
                != _fingerprint_object_identity(expected_fingerprint)
                or fingerprint.identity[3] != expected_fingerprint.identity[3]
            ):
                raise ValueError(f"alpha_max_output_frozen_file_identity_mismatch:{relative}")
        if relative == "preparation_manifest.json" and payload != expected_preparation_bytes:
            raise ValueError("alpha_max_preparation_manifest_final_mismatch")
        observed_fingerprints[relative] = fingerprint

    if "preparation_manifest.json" not in observed_fingerprints:
        raise ValueError("alpha_max_preparation_manifest_missing")
    return FrozenTreeSnapshot(
        directory_identities=observed_directories,
        file_fingerprints=observed_fingerprints,
    )


def _record_owned_directories_descriptor(
    directory_fd: int,
    *,
    snapshot: FrozenTreeSnapshot,
) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

    def walk(current_fd: int, relative_parts: tuple[str, ...]) -> None:
        relative = "/".join(relative_parts)
        identity = _directory_identity(os.fstat(current_fd))
        expected = snapshot.directory_identities.get(relative)
        if expected is not None and expected != identity:
            raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
        snapshot.directory_identities[relative] = identity
        with os.scandir(current_fd) as iterator:
            names = tuple(sorted(entry.name for entry in iterator))
        for name in names:
            observed = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                raise ValueError("alpha_max_output_owned_directory_inventory_invalid")
            child_fd = os.open(name, directory_flags, dir_fd=current_fd)
            try:
                if _directory_identity(os.fstat(child_fd)) != _directory_identity(observed):
                    raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
                walk(child_fd, (*relative_parts, name))
            finally:
                os.close(child_fd)

    walk(directory_fd, ())


def _record_owned_file_descriptor(
    directory_fd: int,
    fingerprint: SourceFingerprint,
    *,
    snapshot: FrozenTreeSnapshot,
) -> None:
    relative_path = fingerprint.relative_path
    parts = _safe_relative_parts(relative_path, field="output_owned_file")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current_fd = os.dup(directory_fd)
    try:
        root_identity = _directory_identity(os.fstat(current_fd))
        expected_root = snapshot.directory_identities.get("")
        if expected_root is not None and expected_root != root_identity:
            raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
        snapshot.directory_identities[""] = root_identity
        traversed: list[str] = []
        for part in parts[:-1]:
            observed = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                raise ValueError("alpha_max_output_owned_directory_identity_invalid")
            child_fd = os.open(part, directory_flags, dir_fd=current_fd)
            try:
                opened_identity = _directory_identity(os.fstat(child_fd))
                if opened_identity != _directory_identity(observed):
                    raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
            except Exception:
                os.close(child_fd)
                raise
            os.close(current_fd)
            current_fd = child_fd
            traversed.append(part)
            relative_directory = "/".join(traversed)
            expected_identity = snapshot.directory_identities.get(relative_directory)
            if expected_identity is not None and expected_identity != opened_identity:
                raise ValueError("alpha_max_output_owned_directory_identity_mismatch")
            snapshot.directory_identities[relative_directory] = opened_identity
    finally:
        os.close(current_fd)

    existing = snapshot.file_fingerprints.get(relative_path)
    if existing is not None and existing.identity != fingerprint.identity:
        raise ValueError("alpha_max_output_owned_file_identity_mismatch")
    snapshot.file_fingerprints[relative_path] = fingerprint


def _load_renameat2() -> Any:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise ValueError("alpha_max_output_root_atomic_noreplace_unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    return renameat2


def _rename_noreplace_at(
    renameat2: Any,
    source_parent_fd: int,
    source_name: str,
    target_parent_fd: int,
    target_name: str,
) -> None:
    ctypes.set_errno(0)
    result = renameat2(
        source_parent_fd,
        os.fsencode(source_name),
        target_parent_fd,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target_name)


def _verify_cleanup_tree_descriptor(
    directory_fd: int,
    *,
    snapshot: FrozenTreeSnapshot,
) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    observed_files: set[str] = set()
    observed_directories: set[str] = set()

    def walk(current_fd: int, relative_parts: tuple[str, ...]) -> None:
        relative_directory = "/".join(relative_parts)
        current_identity = _directory_identity(os.fstat(current_fd))
        if snapshot.directory_identities.get(relative_directory) != current_identity:
            raise ValueError("alpha_max_output_rollback_directory_identity_mismatch")
        observed_directories.add(relative_directory)
        with os.scandir(current_fd) as iterator:
            names = tuple(sorted(entry.name for entry in iterator))
        for name in names:
            relative_parts_child = (*relative_parts, name)
            relative = "/".join(relative_parts_child)
            observed = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
                expected_identity = snapshot.directory_identities.get(relative)
                if expected_identity != _directory_identity(observed):
                    raise ValueError("alpha_max_output_rollback_directory_identity_mismatch")
                child_fd = os.open(name, directory_flags, dir_fd=current_fd)
                try:
                    if _directory_identity(os.fstat(child_fd)) != expected_identity:
                        raise ValueError("alpha_max_output_rollback_directory_identity_mismatch")
                    walk(child_fd, relative_parts_child)
                finally:
                    os.close(child_fd)
                rebound = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                if _directory_identity(rebound) != expected_identity:
                    raise ValueError("alpha_max_output_rollback_directory_identity_mismatch")
                continue
            if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                raise ValueError("alpha_max_output_rollback_entry_type_invalid")
            expected_fingerprint = snapshot.file_fingerprints.get(relative)
            if expected_fingerprint is None or _file_object_identity(
                observed
            ) != _fingerprint_object_identity(expected_fingerprint):
                raise ValueError("alpha_max_output_rollback_file_identity_mismatch")
            observed_files.add(relative)

    walk(directory_fd, ())
    if observed_files != set(snapshot.file_fingerprints) or observed_directories != set(
        snapshot.directory_identities
    ):
        raise ValueError("alpha_max_output_rollback_inventory_mismatch")


def _rollback_directory_by_identity(
    parent_fd: int,
    *,
    expected_identity: tuple[int, int, int],
    snapshot: FrozenTreeSnapshot | None = None,
    missing_ok: bool = False,
) -> None:
    """Read-only retain-and-verify fallback for a failed owned tree.

    POSIX exposes no atomic compare-by-inode-and-unlink operation.  Destructive
    cleanup would therefore permit a same-UID name swap between validation and
    unlink/rmdir.  Failure paths deliberately retain the owned inode instead.
    """
    with os.scandir(parent_fd) as iterator:
        names = tuple(entry.name for entry in iterator)
    matches: list[str] = []
    for name in names:
        try:
            observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            continue
        if (
            stat.S_ISDIR(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode)
            and _directory_identity(observed) == expected_identity
        ):
            matches.append(name)
    if not matches and missing_ok:
        return
    if len(matches) != 1:
        raise ValueError("alpha_max_output_published_tree_identity_not_found")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(matches[0], flags, dir_fd=parent_fd)
    try:
        if _directory_identity(os.fstat(directory_fd)) != expected_identity:
            raise ValueError("alpha_max_output_rollback_identity_mismatch")
        if snapshot is not None:
            _verify_cleanup_tree_descriptor(directory_fd, snapshot=snapshot)
    finally:
        os.close(directory_fd)


def _publish(
    temporary_name: str,
    *,
    target_name: str,
    parent: DirectoryCapability,
    temporary_fd: int,
    expected_preparation_bytes: bytes,
    frozen_snapshot: FrozenTreeSnapshot,
) -> None:
    expected_identity = _directory_identity(os.fstat(temporary_fd))
    _revalidate_pinned_directory(parent, field="output_parent")
    if not _target_is_absent(parent, target_name):
        raise ValueError("alpha_max_output_root_must_remain_absent")
    renameat2 = _load_renameat2()
    expected_content = {
        relative: (fingerprint.byte_count, fingerprint.sha256)
        for relative, fingerprint in frozen_snapshot.file_fingerprints.items()
    }
    _verify_frozen_tree_descriptor(
        temporary_fd,
        expected_content=expected_content,
        expected_preparation_bytes=expected_preparation_bytes,
        expected_snapshot=frozen_snapshot,
    )
    try:
        _rename_noreplace_at(
            renameat2,
            parent.descriptor,
            temporary_name,
            parent.descriptor,
            target_name,
        )
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise ValueError("alpha_max_output_root_must_remain_absent")
        raise ValueError("alpha_max_output_root_atomic_publish_failed") from exc
    try:
        os.fsync(parent.descriptor)
        _revalidate_pinned_directory(parent, field="output_parent")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            published_fd = os.open(target_name, flags, dir_fd=parent.descriptor)
        except OSError as exc:
            raise ValueError("alpha_max_output_root_published_identity_mismatch") from exc
        try:
            if _directory_identity(os.fstat(published_fd)) != expected_identity:
                raise ValueError("alpha_max_output_root_published_identity_mismatch")
        finally:
            os.close(published_fd)
    except Exception:
        try:
            _rollback_directory_by_identity(
                parent.descriptor,
                expected_identity=expected_identity,
                snapshot=frozen_snapshot,
            )
            os.fsync(parent.descriptor)
        except Exception as rollback_error:
            raise ValueError("alpha_max_output_root_publish_rollback_failed") from rollback_error
        raise


def prepare_alpha_max_phase_roots(
    *,
    raw_root: Path,
    feature_root: Path,
    contract_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build and atomically publish all six phase-owned roots."""
    _validate_phase_intervals()
    canonical_raw_path = _canonical_absolute(raw_root, field="raw_source_root")
    canonical_feature_path = _canonical_absolute(feature_root, field="feature_source_root")
    target, target_parent = _require_absent_target(output_root)
    if target.is_relative_to(canonical_raw_path) or target.is_relative_to(canonical_feature_path):
        target_parent.close()
        raise ValueError("alpha_max_output_root_inside_source_rejected")

    raw_capability: DirectoryCapability | None = None
    feature_capability: DirectoryCapability | None = None
    temporary_fd: int | None = None
    temporary_name: str | None = None
    ownership_snapshot: FrozenTreeSnapshot | None = None
    frozen_snapshot: FrozenTreeSnapshot | None = None
    published = False
    try:
        raw_capability = _pin_canonical_directory(
            canonical_raw_path,
            field="raw_source_root",
        )
        feature_capability = _pin_canonical_directory(
            canonical_feature_path,
            field="feature_source_root",
        )
        availability, contract_sha = _read_contract_manifest(contract_manifest)
        temporary_name, temporary_fd, temporary_root = _create_temporary_root(
            target_parent,
            target_name=target.name,
        )
        ownership_snapshot = FrozenTreeSnapshot(
            directory_identities={"": _directory_identity(os.fstat(temporary_fd))},
            file_fingerprints={},
        )
        for phase_id in _PHASE_IDS:
            for root_kind in ("raw", "feature"):
                created_fd = _open_owned_output_parent(
                    temporary_fd,
                    (phase_id, root_kind),
                    snapshot=ownership_snapshot,
                )
                os.close(created_fd)

        entries: list[dict[str, Any]] = []
        expected_outputs: set[str] = set()
        _materialize_raw(
            temporary_root=temporary_root,
            temporary_fd=temporary_fd,
            ownership_snapshot=ownership_snapshot,
            raw_root=raw_capability,
            availability=availability,
            entries=entries,
            expected_outputs=expected_outputs,
        )
        _materialize_feature(
            temporary_root=temporary_root,
            temporary_fd=temporary_fd,
            ownership_snapshot=ownership_snapshot,
            feature_root=feature_capability,
            availability=availability,
            entries=entries,
            expected_outputs=expected_outputs,
        )
        entries.sort(key=lambda value: value["output_relative_path"])
        _verify_output_inventory(temporary_root, expected_outputs)

        availability_payload = {
            "feature": {
                "availability_end_by_symbol": {
                    value.symbol: _iso_utc(value.feature_end_utc) for value in availability
                },
                "availability_start_by_symbol": {
                    value.symbol: _iso_utc(value.feature_start_utc) for value in availability
                },
            },
            "raw": {
                "availability_end_by_symbol": {
                    value.symbol: _iso_utc(value.raw_end_utc) for value in availability
                },
                "availability_start_by_symbol": {
                    value.symbol: _iso_utc(value.raw_start_utc) for value in availability
                },
            },
        }
        preparation_manifest = {
            "availability": availability_payload,
            "availability_sha256_by_root_kind": {
                root_kind: _sha256_bytes(_canonical_json_bytes(availability_payload[root_kind]))
                for root_kind in ("raw", "feature")
            },
            "contract_manifest_schema_version": _CONTRACT_SCHEMA,
            "contract_manifest_sha256": contract_sha,
            "exchange": EXCHANGE,
            "file_count": len(entries),
            "files": entries,
            "phase_intervals": [
                {
                    "end_utc": _iso_utc(PHASE_INTERVALS[phase_id][1]),
                    "phase_id": phase_id,
                    "start_utc": _iso_utc(PHASE_INTERVALS[phase_id][0]),
                }
                for phase_id in _PHASE_IDS
            ],
            "schema_version": _PREPARATION_SCHEMA,
            "symbols": list(CANDIDATE_SYMBOLS),
        }
        preparation_bytes = _canonical_json_bytes(preparation_manifest)
        preparation_fingerprint = _write_owned_bytes(
            preparation_bytes,
            temporary_fd=temporary_fd,
            relative_path="preparation_manifest.json",
            ownership_snapshot=ownership_snapshot,
        )
        _record_owned_file_descriptor(
            temporary_fd,
            preparation_fingerprint,
            snapshot=ownership_snapshot,
        )

        expected_content = {
            entry["output_relative_path"]: (
                int(entry["output_byte_count"]),
                str(entry["output_sha256"]),
            )
            for entry in entries
        }
        expected_content["preparation_manifest.json"] = (
            len(preparation_bytes),
            _sha256_bytes(preparation_bytes),
        )
        if len(expected_content) != len(entries) + 1:
            raise ValueError("alpha_max_preparation_manifest_output_paths_duplicate")
        _verify_cleanup_tree_descriptor(
            temporary_fd,
            snapshot=ownership_snapshot,
        )
        _freeze_tree_descriptor(
            temporary_fd,
            expected_snapshot=ownership_snapshot,
        )
        frozen_snapshot = _verify_frozen_tree_descriptor(
            temporary_fd,
            expected_content=expected_content,
            expected_preparation_bytes=preparation_bytes,
            expected_snapshot=ownership_snapshot,
        )
        os.fsync(temporary_fd)
        _revalidate_pinned_directory(raw_capability, field="raw_source_root")
        _revalidate_pinned_directory(feature_capability, field="feature_source_root")
        _publish(
            temporary_name,
            target_name=target.name,
            parent=target_parent,
            temporary_fd=temporary_fd,
            expected_preparation_bytes=preparation_bytes,
            frozen_snapshot=frozen_snapshot,
        )
        published = True
        return {
            "file_count": len(entries),
            "output_root": str(target),
            "preparation_manifest_sha256": _sha256_bytes(preparation_bytes),
        }
    finally:
        active_error = sys.exception()
        if not published and temporary_name is not None:
            if temporary_fd is None:
                raise AssertionError("alpha_max_output_temporary_descriptor_missing")
            try:
                _rollback_directory_by_identity(
                    target_parent.descriptor,
                    expected_identity=_directory_identity(os.fstat(temporary_fd)),
                    snapshot=frozen_snapshot or ownership_snapshot,
                    missing_ok=True,
                )
            except Exception as cleanup_error:
                if active_error is None:
                    raise
                active_error.add_note(f"phase-root cleanup preserved: {cleanup_error!r}")
        if temporary_fd is not None:
            os.close(temporary_fd)
        if feature_capability is not None:
            feature_capability.close()
        if raw_capability is not None:
            raw_capability.close()
        target_parent.close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
        help="Absolute canonical market_ohlcv_1s root (contains binance/<symbol>).",
    )
    parser.add_argument(
        "--feature-root",
        required=True,
        type=Path,
        help="Absolute canonical feature_points root (contains exchange=binance).",
    )
    parser.add_argument("--contract-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the strict phase-root materializer."""
    args = _parse_args(argv)
    result = prepare_alpha_max_phase_roots(
        raw_root=args.raw_root,
        feature_root=args.feature_root,
        contract_manifest=args.contract_manifest,
        output_root=args.output_root,
    )
    print(_canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
