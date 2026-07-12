"""Helpers for querying parquet-backed futures feature points."""

from __future__ import annotations

import hashlib
import io
import math
import os
import stat
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from threading import Lock
from types import MappingProxyType
from typing import Any, Final

import polars as pl

from lumina_quant.market_data import load_futures_feature_points_from_db, normalize_symbol

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "funding_rate",
    "funding_mark_price",
    "funding_fee_rate",
    "funding_fee_quote_per_unit",
    "mark_price",
    "index_price",
    "open_interest",
    "taker_buy_base_volume",
    "taker_sell_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
    "liquidation_long_qty",
    "liquidation_short_qty",
    "liquidation_long_notional",
    "liquidation_short_notional",
    "best_bid_price",
    "best_bid_quantity",
    "best_ask_price",
    "best_ask_quantity",
    "bbo_mid_price",
    "bbo_spread_bps",
    "book_depth_bid_notional_1pct",
    "book_depth_ask_notional_1pct",
    "book_depth_imbalance_1pct",
)
FEATURE_POINT_MAX_STALE_MS: Final[int] = 8 * 60 * 60 * 1000


@dataclass(frozen=True, slots=True)
class FeaturePoint:
    """Latest feature value with canonical and official-source timestamps."""

    value: float
    source_timestamp_ms: int
    canonical_timestamp_ms: int | None = None

    def __post_init__(self) -> None:
        if self.canonical_timestamp_ms is None:
            object.__setattr__(self, "canonical_timestamp_ms", self.source_timestamp_ms)


@dataclass(frozen=True, slots=True)
class SealedFeatureFile:
    """One immutable feature partition accepted by a descriptor-bound lookup."""

    relative_path: str
    byte_count: int
    mode: int
    mtime_ns: int
    sha256: str

    def __post_init__(self) -> None:
        relative_path = str(self.relative_path or "")
        path = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\0" in relative_path
            or "\\" in relative_path
            or path.is_absolute()
            or relative_path != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
            or not relative_path.endswith(".parquet")
        ):
            raise ValueError("sealed_feature_relative_path_invalid")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("sealed_feature_byte_count_invalid")
        if type(self.mode) is not int or not 0 <= self.mode <= 0o7777:
            raise ValueError("sealed_feature_mode_invalid")
        if type(self.mtime_ns) is not int or self.mtime_ns < 0:
            raise ValueError("sealed_feature_mtime_invalid")
        sha256 = str(self.sha256 or "").lower()
        if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
            raise ValueError("sealed_feature_sha256_invalid")
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "sha256", sha256)


@dataclass(slots=True)
class _FeatureCache:
    timestamps_ms: list[int]
    columns: dict[str, list[float | None]]
    raw_columns: dict[str, list[float | None]]
    canonical_timestamps_ms: dict[str, list[int | None]]
    source_timestamps_ms: dict[str, list[int | None]]


class FeaturePointLookup:
    """Lazy, per-symbol lookup for latest feature values at or before a timestamp."""

    def __init__(
        self,
        *,
        db_path: str | None,
        exchange: str = "binance",
        start_date: Any = None,
        end_date: Any = None,
        sealed_files: tuple[SealedFeatureFile, ...] | None = None,
    ) -> None:
        self.db_path = str(db_path or "").strip()
        self.exchange = str(exchange or "binance").strip().lower() or "binance"
        self.start_date = start_date
        self.end_date = end_date
        self._cache: dict[str, _FeatureCache] = {}
        self._lock = Lock()
        self._sealed_root_fd: int | None = None
        self._sealed_root_identity: tuple[int, int, int] | None = None
        self._sealed_files: tuple[SealedFeatureFile, ...] | None = None
        self._sealed_files_by_symbol: Mapping[str, tuple[SealedFeatureFile, ...]] | None = None
        if sealed_files is not None:
            self._bind_sealed_files(sealed_files)

    def __del__(self) -> None:
        descriptor = getattr(self, "_sealed_root_fd", None)
        if descriptor is None:
            return
        try:
            os.close(descriptor)
        except OSError:
            pass
        self._sealed_root_fd = None

    @property
    def sealed_files(self) -> tuple[SealedFeatureFile, ...] | None:
        """Return the immutable sealed inventory, when descriptor binding is active."""
        return self._sealed_files

    def get_latest(
        self,
        symbol: str,
        field: str,
        *,
        timestamp_ms: int | None,
    ) -> float | None:
        """Return the latest non-null feature value at or before ``timestamp_ms``."""
        point = self.get_latest_point(symbol, field, timestamp_ms=timestamp_ms)
        return point.value if point is not None else None

    def get_latest_point(
        self,
        symbol: str,
        field: str,
        *,
        timestamp_ms: int | None,
    ) -> FeaturePoint | None:
        """Return the latest finite feature point at or before ``timestamp_ms``."""
        token = str(field or "").strip()
        if not self.db_path or not token or int(timestamp_ms or 0) <= 0:
            return None
        if token not in FEATURE_COLUMNS:
            return None

        cache = self._get_or_load(symbol)
        if not cache.timestamps_ms:
            return None

        idx = bisect_right(cache.timestamps_ms, int(timestamp_ms)) - 1
        if idx < 0:
            return None

        value = cache.columns.get(token, [None])[idx]
        if value is None:
            return None
        source_timestamp = cache.source_timestamps_ms.get(token, [None])[idx]
        canonical_timestamp = cache.canonical_timestamps_ms.get(token, [None])[idx]
        if source_timestamp is None or canonical_timestamp is None:
            return None
        if int(timestamp_ms) - int(canonical_timestamp) > FEATURE_POINT_MAX_STALE_MS:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        if not math.isfinite(parsed):
            return None
        return FeaturePoint(
            value=parsed,
            source_timestamp_ms=int(source_timestamp),
            canonical_timestamp_ms=int(canonical_timestamp),
        )

    def sum_between(
        self,
        symbol: str,
        field: str,
        *,
        start_timestamp_ms: int | None,
        end_timestamp_ms: int | None,
    ) -> float | None:
        """Return the sum of non-null feature values in an inclusive ms window."""
        token = str(field or "").strip()
        if (
            not self.db_path
            or not token
            or token not in FEATURE_COLUMNS
            or int(start_timestamp_ms or 0) <= 0
            or int(end_timestamp_ms or 0) <= 0
        ):
            return None
        start_ms = int(start_timestamp_ms or 0)
        end_ms = int(end_timestamp_ms or 0)
        if end_ms < start_ms:
            return None

        cache = self._get_or_load(symbol)
        if not cache.timestamps_ms:
            return None
        left = bisect_right(cache.timestamps_ms, start_ms - 1)
        right = bisect_right(cache.timestamps_ms, end_ms)
        if right <= left:
            return None

        values = [
            float(value)
            for value in cache.raw_columns.get(token, [])[left:right]
            if value is not None and math.isfinite(float(value))
        ]
        if not values:
            return None
        return float(sum(values))

    def _get_or_load(self, symbol: str) -> _FeatureCache:
        normalized = normalize_symbol(symbol)
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._cache.get(normalized)
            if cached is not None:
                return cached
            loaded = self._load_symbol(normalized)
            self._cache[normalized] = loaded
            return loaded

    def _load_symbol(self, symbol: str) -> _FeatureCache:
        if self._sealed_files is None:
            frame = load_futures_feature_points_from_db(
                self.db_path,
                exchange=self.exchange,
                symbol=symbol,
                start_date=self.start_date,
                end_date=self.end_date,
            )
        else:
            frame = self._load_sealed_symbol(symbol)
        if frame.is_empty():
            empty = {field: [] for field in FEATURE_COLUMNS}
            return _FeatureCache(
                timestamps_ms=[],
                columns=empty,
                raw_columns=dict(empty),
                canonical_timestamps_ms=dict(empty),
                source_timestamps_ms=dict(empty),
            )

        cleaned = frame.filter(pl.col("timestamp_ms").is_not_null()).with_columns(
            pl.col("timestamp_ms").cast(pl.Int64)
        )
        if cleaned.is_empty():
            empty = {field: [] for field in FEATURE_COLUMNS}
            return _FeatureCache(
                timestamps_ms=[],
                columns=empty,
                raw_columns=dict(empty),
                canonical_timestamps_ms=dict(empty),
                source_timestamps_ms=dict(empty),
            )

        for field in FEATURE_COLUMNS:
            if field not in cleaned.columns:
                cleaned = cleaned.with_columns(pl.lit(None, dtype=pl.Float64).alias(field))
        if "source_timestamp_ms" not in cleaned.columns:
            cleaned = cleaned.with_columns(
                pl.lit(None, dtype=pl.Int64).alias("source_timestamp_ms")
            )
        else:
            cleaned = cleaned.with_columns(
                pl.col("source_timestamp_ms").cast(pl.Int64, strict=False)
            )

        cleaned = (
            cleaned.select(["timestamp_ms", "source_timestamp_ms", *FEATURE_COLUMNS])
            .sort("timestamp_ms")
            .unique(
                subset=["timestamp_ms"],
                keep="last",
            )
        )
        raw_columns = {
            field: [
                float(value) if value is not None else None
                for value in cleaned.get_column(field).to_list()
            ]
            for field in FEATURE_COLUMNS
        }
        canonical_cols = [f"__{field}_canonical_timestamp_ms" for field in FEATURE_COLUMNS]
        source_cols = [f"__{field}_source_timestamp_ms" for field in FEATURE_COLUMNS]
        value_cols = [f"__{field}_ffill" for field in FEATURE_COLUMNS]
        bounded = cleaned.with_columns(
            [
                *[
                    pl.when(pl.col(field).is_not_null())
                    .then(pl.col("timestamp_ms"))
                    .otherwise(None)
                    .cast(pl.Int64)
                    .forward_fill()
                    .alias(canonical_col)
                    for field, canonical_col in zip(FEATURE_COLUMNS, canonical_cols, strict=True)
                ],
                *[
                    pl.when(pl.col(field).is_not_null())
                    .then(pl.coalesce("source_timestamp_ms", "timestamp_ms"))
                    .otherwise(None)
                    .cast(pl.Int64)
                    .forward_fill()
                    .alias(source_col)
                    for field, source_col in zip(FEATURE_COLUMNS, source_cols, strict=True)
                ],
                *[
                    pl.col(field).cast(pl.Float64).forward_fill().alias(value_col)
                    for field, value_col in zip(FEATURE_COLUMNS, value_cols, strict=True)
                ],
            ]
        ).with_columns(
            [
                pl.when(
                    pl.col(canonical_col).is_not_null()
                    & (
                        (pl.col("timestamp_ms") - pl.col(canonical_col))
                        <= FEATURE_POINT_MAX_STALE_MS
                    )
                )
                .then(pl.col(value_col))
                .otherwise(None)
                .alias(field)
                for field, canonical_col, value_col in zip(
                    FEATURE_COLUMNS,
                    canonical_cols,
                    value_cols,
                    strict=True,
                )
            ]
        )

        timestamps_ms = [int(value) for value in bounded.get_column("timestamp_ms").to_list()]
        columns = {
            field: [
                float(value) if value is not None else None
                for value in bounded.get_column(field).to_list()
            ]
            for field in FEATURE_COLUMNS
        }
        canonical_timestamps_ms = {
            field: [
                int(value) if value is not None else None
                for value in bounded.get_column(canonical_col).to_list()
            ]
            for field, canonical_col in zip(FEATURE_COLUMNS, canonical_cols, strict=True)
        }
        source_timestamps_ms = {
            field: [
                int(value) if value is not None else None
                for value in bounded.get_column(source_col).to_list()
            ]
            for field, source_col in zip(FEATURE_COLUMNS, source_cols, strict=True)
        }
        return _FeatureCache(
            timestamps_ms=timestamps_ms,
            columns=columns,
            raw_columns=raw_columns,
            canonical_timestamps_ms=canonical_timestamps_ms,
            source_timestamps_ms=source_timestamps_ms,
        )

    def _bind_sealed_files(self, sealed_files: tuple[SealedFeatureFile, ...]) -> None:
        if type(sealed_files) is not tuple or not sealed_files:
            raise TypeError("sealed_feature_files_must_be_nonempty_exact_tuple")
        if any(type(entry) is not SealedFeatureFile for entry in sealed_files):
            raise TypeError("sealed_feature_files_must_be_exact")
        if tuple(sorted(sealed_files, key=lambda entry: entry.relative_path)) != sealed_files:
            raise ValueError("sealed_feature_files_not_sorted")
        if len({entry.relative_path for entry in sealed_files}) != len(sealed_files):
            raise ValueError("sealed_feature_file_path_duplicate")
        if not self.db_path or not os.path.isabs(self.db_path):
            raise ValueError("sealed_feature_root_must_be_absolute")
        if os.path.normpath(self.db_path) != self.db_path:
            raise ValueError("sealed_feature_root_must_be_canonical")

        by_symbol: dict[str, list[SealedFeatureFile]] = {}
        for entry in sealed_files:
            symbol = _sealed_feature_symbol(entry.relative_path, exchange=self.exchange)
            by_symbol.setdefault(symbol, []).append(entry)

        descriptor = _open_sealed_feature_root(self.db_path)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                raise ValueError("sealed_feature_root_not_directory")
            self._sealed_root_fd = descriptor
            self._sealed_root_identity = _directory_identity(opened)
            self._sealed_files = sealed_files
            self._sealed_files_by_symbol = MappingProxyType(
                {symbol: tuple(entries) for symbol, entries in sorted(by_symbol.items())}
            )
        except Exception:
            self._sealed_root_fd = None
            os.close(descriptor)
            raise

    def _load_sealed_symbol(self, symbol: str) -> pl.DataFrame:
        descriptor = self._sealed_root_fd
        root_identity = self._sealed_root_identity
        by_symbol = self._sealed_files_by_symbol
        if descriptor is None or root_identity is None or by_symbol is None:
            raise ValueError("sealed_feature_capability_unavailable")
        try:
            if _directory_identity(os.fstat(descriptor)) != root_identity:
                raise ValueError("sealed_feature_root_identity_changed")
        except OSError as exc:
            raise ValueError("sealed_feature_root_capability_closed") from exc

        frames = [
            _read_sealed_feature_frame(descriptor, entry)
            for entry in by_symbol.get(normalize_symbol(symbol), ())
        ]
        if not frames:
            return pl.DataFrame()
        try:
            frame = pl.concat(frames, how="diagonal_relaxed")
        except Exception as exc:
            raise ValueError("sealed_feature_partition_concat_failed") from exc
        if "timestamp_ms" not in frame.columns:
            raise ValueError("sealed_feature_timestamp_column_missing")
        return frame.sort("timestamp_ms").unique(
            subset=["timestamp_ms"],
            keep="last",
        )


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(stat.S_IFMT(value.st_mode)))


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


def _open_sealed_feature_root(path: str) -> int:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(os.path.sep, directory_flags)
    try:
        for part in os.path.normpath(path).split(os.path.sep)[1:]:
            try:
                observed = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                raise ValueError("sealed_feature_root_stat_failed") from exc
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError("sealed_feature_root_symlink_rejected")
            if not stat.S_ISDIR(observed.st_mode):
                raise ValueError("sealed_feature_root_entry_not_directory")
            try:
                child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            except OSError as exc:
                raise ValueError("sealed_feature_root_open_failed") from exc
            try:
                opened = os.fstat(child_fd)
            except OSError:
                os.close(child_fd)
                raise
            if _directory_identity(observed) != _directory_identity(opened):
                os.close(child_fd)
                raise ValueError("sealed_feature_root_changed_during_open")
            os.close(parent_fd)
            parent_fd = child_fd
        return parent_fd
    except Exception:
        os.close(parent_fd)
        raise


def _sealed_feature_symbol(relative_path: str, *, exchange: str) -> str:
    parts = PurePosixPath(relative_path).parts
    if len(parts) >= 2 and parts[:2] == ("feature_points", f"exchange={exchange}"):
        scoped = parts[2:]
    elif parts and parts[0] == f"exchange={exchange}":
        scoped = parts[1:]
    else:
        scoped = parts
    if (
        len(scoped) != 3
        or not scoped[0].startswith("symbol=")
        or not scoped[1].startswith("date=")
        or not scoped[2].endswith(".parquet")
    ):
        raise ValueError("sealed_feature_partition_layout_invalid")
    symbol = normalize_symbol(scoped[0].removeprefix("symbol="))
    if not symbol:
        raise ValueError("sealed_feature_symbol_invalid")
    return symbol


def _open_sealed_feature_file(root_fd: int, relative_path: str) -> tuple[int, os.stat_result]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.dup(root_fd)
    try:
        parts = PurePosixPath(relative_path).parts
        for part in parts[:-1]:
            observed = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ValueError("sealed_feature_path_component_rejected")
            child_fd = os.open(part, directory_flags, dir_fd=current_fd)
            try:
                opened = os.fstat(child_fd)
            except OSError:
                os.close(child_fd)
                raise
            if _directory_identity(observed) != _directory_identity(opened):
                os.close(child_fd)
                raise ValueError("sealed_feature_path_changed_during_open")
            os.close(current_fd)
            current_fd = child_fd

        observed = os.stat(parts[-1], dir_fd=current_fd, follow_symlinks=False)
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError("sealed_feature_symlink_rejected")
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError("sealed_feature_not_regular")
        if int(observed.st_nlink) != 1:
            raise ValueError("sealed_feature_hardlink_rejected")
        file_fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
        try:
            opened = os.fstat(file_fd)
        except OSError:
            os.close(file_fd)
            raise
        if _file_identity(observed) != _file_identity(opened):
            os.close(file_fd)
            raise ValueError("sealed_feature_changed_during_open")
        return file_fd, opened
    except OSError as exc:
        raise ValueError("sealed_feature_open_failed") from exc
    finally:
        os.close(current_fd)


def _read_sealed_feature_frame(root_fd: int, entry: SealedFeatureFile) -> pl.DataFrame:
    file_fd, before = _open_sealed_feature_file(root_fd, entry.relative_path)
    try:
        if (
            int(before.st_size) != entry.byte_count
            or stat.S_IMODE(before.st_mode) != entry.mode
            or int(before.st_mtime_ns) != entry.mtime_ns
        ):
            raise ValueError("sealed_feature_metadata_mismatch")
        digest = hashlib.sha256()
        payload = bytearray()
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            payload.extend(block)
            digest.update(block)
            if len(payload) > entry.byte_count:
                raise ValueError("sealed_feature_byte_count_mismatch")
        after = os.fstat(file_fd)
        if _file_identity(before) != _file_identity(after):
            raise ValueError("sealed_feature_changed_during_read")
        if len(payload) != entry.byte_count or digest.hexdigest() != entry.sha256:
            raise ValueError("sealed_feature_content_mismatch")
    except OSError as exc:
        raise ValueError("sealed_feature_read_failed") from exc
    finally:
        os.close(file_fd)
    try:
        return pl.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError("sealed_feature_parquet_invalid") from exc


__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_POINT_MAX_STALE_MS",
    "FeaturePoint",
    "FeaturePointLookup",
    "SealedFeatureFile",
]
