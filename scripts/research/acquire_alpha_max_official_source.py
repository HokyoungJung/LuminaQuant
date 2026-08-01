#!/usr/bin/env python3
"""Fail-closed Binance Alpha-Max source acquirer (execution is opt-in)."""

from __future__ import annotations

import argparse
from array import array
from collections.abc import Iterable
from contextvars import ContextVar
from contextlib import ExitStack, contextmanager
import fcntl
import heapq
import csv
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import io
import itertools
import json
import math
import shutil
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import polars as pl

EXCHANGE = "binance"
SYMBOLS = (
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
CONTRACT_SCHEMA = "alpha_max_contract_manifest.v2"
CONTRACT_SHA256 = "ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220"
EVIDENCE_SHA256 = "214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719"
ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
API_BASE = "https://fapi.binance.com/fapi/v1"
ALLOWED_HOSTS = {"data.binance.vision", "fapi.binance.com"}
FUNDING_JITTER_MS = 1000
DERIVATION_VERSION = "alpha-max-binance-ohlcv-v4"
RAW_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
FUNDING_COLUMNS = ["timestamp_ms", "source_timestamp_ms", "exchange", "symbol", "funding_rate"]
DOWNLOAD_CHUNK = 1 << 20
DOWNLOAD_ATTEMPTS = 3
HOST_RESERVE_PATH = Path("/mnt/c")
HOST_RESERVE_BYTES = 21_474_836_480
MAX_LIVE_ARCHIVES = 1
ARCHIVE_RETENTION = "retired_after_double_derivation"

CANONICAL_ORDER_ARCHIVES = {
    ("BTCUSDT", "2023-05"): (
        "301acec76a7644aa73180fd7f8d913ce4eecfa7e7bca5057f1782f96d91b9ef0",
        468405603,
    ),
    ("BTCUSDT", "2023-10"): (
        "d3fe5fa477d68d6730248d634e1bd37ae4838839d78709ef355d9d9c6749fea4",
        492720741,
    ),
    ("SOLUSDT", "2023-11"): (
        "188c3145ecaab1cf546318c293fb4fef0e320a6dc05b14eea013a46209ebbd73",
        535864305,
    ),
    ("SOLUSDT", "2023-12"): (
        "c12bc6707c8fb6ab5f3fe712ad6c8b816053a27c6c6ead4f7dc98df7a098b70c",
        558456557,
    ),
}
AUTHENTICATED_DUPLICATE_AGGREGATE_ARCHIVES = {
    ("SOLUSDT", "2025-07"): (
        "07842c476aab159f008ffc4e95e421e181f75348610c23baeae1dc3799d4e89b",
        208568498,
        frozenset({926014272}),
    ),
}
ORDER_RECORD = struct.Struct(">qqdd")
ORDER_CHUNK_RECORDS = 250_000
ORDER_MERGE_FAN_IN = 64
POSITIVE_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


class AcquisitionError(ValueError):
    pass


@dataclass(frozen=True)
class Contract:
    symbol: str
    raw_start_ms: int
    raw_end_ms: int
    feature_start_ms: int
    feature_end_ms: int


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_file(path: Path) -> os.stat_result:
    try:
        item = path.lstat()
    except OSError as exc:
        raise AcquisitionError("required_file_missing") from exc
    if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        raise AcquisitionError("unsafe_file")
    return item


@contextmanager
def stable_file(path: Path) -> Iterable[io.BufferedReader]:
    """Open an absolute regular file through no-follow directory descriptors."""
    path = lexical(path)
    components = path.parts[1:]
    if not components:
        raise AcquisitionError("required_file_missing")
    current_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    namespace_root_uid = os.fstat(current_fd).st_uid
    fd = -1
    try:
        trusted_parent_directory(
            os.fstat(current_fd),
            False,
            namespace_root=True,
            namespace_root_uid=namespace_root_uid,
        )
        for index, component in enumerate(components[:-1]):
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
            except OSError as exc:
                raise AcquisitionError("unsafe_root_parent") from exc
            os.close(current_fd)
            current_fd = next_fd
            trusted_parent_directory(
                os.fstat(current_fd),
                index == len(components) - 2,
                namespace_root_uid=namespace_root_uid,
            )
        try:
            fd = os.open(
                components[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd
            )
        except OSError as exc:
            raise AcquisitionError("required_file_missing") from exc
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AcquisitionError("unsafe_file")
        with os.fdopen(os.dup(fd), "rb") as source:
            yield source
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ):
            raise AcquisitionError("unsafe_file")
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(current_fd)


def stable_file_bytes(path: Path, limit: int | None = None) -> bytes:
    with stable_file(path) as source:
        chunks: list[bytes] = []
        size = 0
        for block in iter(lambda: source.read(DOWNLOAD_CHUNK), b""):
            size += len(block)
            if limit is not None and size > limit:
                raise AcquisitionError("immutable_json_too_large")
            chunks.append(block)
        return b"".join(chunks)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with stable_file(path) as source:
        for block in iter(lambda: source.read(DOWNLOAD_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def opened_file_identity(source: io.BufferedReader) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    source.seek(0)
    for block in iter(lambda: source.read(DOWNLOAD_CHUNK), b""):
        digest.update(block)
        byte_count += len(block)
    source.seek(0)
    return digest.hexdigest(), byte_count


def authenticated_archive_identity(
    source: io.BufferedReader, receipt: dict[str, Any] | None
) -> tuple[str, int]:
    identity = opened_file_identity(source)
    if receipt is not None and (
        not isinstance(receipt, dict)
        or receipt.get("sha256") != identity[0]
        or receipt.get("byte_count") != identity[1]
    ):
        raise AcquisitionError("archive_receipt_invalid")
    return identity


def safe_file_bytes(path: Path) -> bytes:
    """Read one regular, singly-linked file without following any component."""
    return stable_file_bytes(path)


def parquet_frame(path: Path) -> pl.DataFrame:
    with stable_file(path) as source:
        return pl.read_parquet(source)


def stable_file_stat(path: Path) -> os.stat_result:
    with stable_file(path) as source:
        return os.fstat(source.fileno())


def scratch_path(root: Path) -> Path:
    root = lexical(root)
    return root.parent / f".alpha-max-scratch-{root.name}-{os.getuid()}"


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def assert_host_reserve(required_bytes: int = 0) -> None:
    """Fail before a write can consume the WSL host's reserved capacity."""
    if isinstance(required_bytes, bool) or required_bytes < 0:
        raise AcquisitionError("host_reserve_requirement_invalid")
    try:
        free = shutil.disk_usage(HOST_RESERVE_PATH).free
    except OSError as exc:
        raise AcquisitionError("host_reserve_unavailable") from exc
    if free - required_bytes <= HOST_RESERVE_BYTES:
        raise AcquisitionError("host_reserve_exhausted")


def mkdir_durable(path: Path) -> None:
    if path.exists():
        return
    mkdir_durable(path.parent)
    path.mkdir(mode=0o700)
    fsync_directory(path.parent)


def scratch_file_identity(path: Path) -> tuple[int, int]:
    item = path.lstat()
    if (
        not stat.S_ISREG(item.st_mode)
        or item.st_uid != os.getuid()
        or stat.S_IMODE(item.st_mode) != 0o600
        or item.st_nlink != 1
    ):
        raise AcquisitionError("unsafe_scratch_entry")
    return item.st_dev, item.st_ino


def remove_aggtrades_file(path: Path, identity: tuple[int, int] | None = None) -> None:
    expected = identity or scratch_file_identity(path)
    item = path.lstat()
    if (
        not stat.S_ISREG(item.st_mode)
        or item.st_uid != os.getuid()
        or stat.S_IMODE(item.st_mode) != 0o600
        or item.st_nlink != 1
        or (item.st_dev, item.st_ino) != expected
    ):
        raise AcquisitionError("unsafe_scratch_entry")
    path.unlink()
    fsync_directory(path.parent)


def scratch_directory(root: Path) -> Path:
    """Return a private, same-filesystem scratch directory beside an owned root."""
    root = lexical(root)
    safe_existing_directory(root.parent)
    scratch = scratch_path(root)
    try:
        os.mkdir(scratch, 0o700)
        fsync_directory(scratch.parent)
    except FileExistsError:
        pass
    try:
        item = scratch.lstat()
    except OSError as exc:
        raise AcquisitionError("unsafe_scratch_directory") from exc
    if (
        not stat.S_ISDIR(item.st_mode)
        or item.st_uid != os.getuid()
        or item.st_mode & 0o777 != 0o700
        or item.st_dev != root.parent.stat().st_dev
    ):
        raise AcquisitionError("unsafe_scratch_directory")
    candidates = list(scratch.iterdir())
    stale_aggtrades: list[tuple[Path, tuple[int, int]]] = []
    for candidate in candidates:
        item = candidate.lstat()
        if candidate.name.startswith(".aggtrades-"):
            stale_aggtrades.append((candidate, scratch_file_identity(candidate)))
            continue
        if not candidate.name.startswith((".acquire-", ".derive-", ".partial-", ".recovery-")):
            raise AcquisitionError("unsafe_scratch_entry")
        if not stat.S_ISREG(item.st_mode) or item.st_uid != os.getuid():
            raise AcquisitionError("unsafe_scratch_entry")
    for candidate, identity in stale_aggtrades:
        remove_aggtrades_file(candidate, identity)
    for candidate in candidates:
        if candidate.name.startswith(".aggtrades-"):
            continue
        candidate.unlink()
        fsync_directory(scratch)
    return scratch


def cleanup_scratch(*roots: Path) -> None:
    for root in roots:
        scratch_directory(root)


def scratch_file(root: Path, prefix: str, suffix: str = "") -> tuple[int, str]:
    return tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=scratch_directory(root))


def atomic_write(
    path: Path, data: bytes, replace: bool = False, scratch_root: Path | None = None
) -> None:
    assert_host_reserve(len(data))
    mkdir_durable(path.parent)
    fsync_directory(path.parent)
    if not replace and (path.exists() or path.is_symlink()):
        raise AcquisitionError(f"output_already_exists:{path}")
    fd, temporary = scratch_file(scratch_root or path.parent, ".acquire-")
    try:
        with os.fdopen(fd, "wb") as sink:
            sink.write(data)
            sink.flush()
            os.fsync(sink.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        fsync_directory(path.parent)
    except FileExistsError as exc:
        raise AcquisitionError(f"output_already_exists:{path}") from exc
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        fsync_directory(Path(temporary).parent)


def immutable_json(path: Path, value: Any, scratch_root: Path | None = None) -> str:
    data = canonical_bytes(value)
    atomic_write(path, data, scratch_root=scratch_root)
    return sha256(data)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AcquisitionError("immutable_json_invalid")
        value[key] = item
    return value


def read_json(path: Path) -> Any:
    try:
        raw = stable_file_bytes(path, limit=64 << 20)
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AcquisitionError("immutable_json_invalid") from exc
    if canonical_bytes(value) != raw:
        raise AcquisitionError("immutable_json_noncanonical")
    return value


def _reject_invalid_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def read_json_value(path: Path) -> Any:
    try:
        return json.loads(
            stable_file_bytes(path, limit=64 << 20),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_invalid_json_constant,
        )
    except (OSError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raise AcquisitionError("official_json_invalid") from exc


def utc_ms(value: object, label: str) -> int:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AcquisitionError(f"invalid_{label}")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AcquisitionError(f"invalid_{label}") from exc
    if parsed.tzinfo != UTC or parsed.isoformat().replace("+00:00", "Z") != value:
        raise AcquisitionError(f"invalid_{label}")
    return int(parsed.timestamp() * 1000)


def month_starts(start: int, end: int) -> Iterable[datetime]:
    current = datetime.fromtimestamp(start / 1000, UTC).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    while int(current.timestamp() * 1000) < end:
        yield current
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def day_starts(start: int, end: int) -> Iterable[datetime]:
    current = datetime.fromtimestamp(start / 1000, UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    while int(current.timestamp() * 1000) < end:
        yield current
        current += timedelta(days=1)


def month_bounds(month: datetime) -> tuple[int, int]:
    start = int(month.timestamp() * 1000)
    end = int(((month.replace(day=28) + timedelta(days=4)).replace(day=1)).timestamp() * 1000)
    return start, end


def load_contract(path: Path) -> list[Contract]:
    raw = safe_file_bytes(path)
    if sha256(raw) != CONTRACT_SHA256:
        raise AcquisitionError("contract_sha256_not_approved")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AcquisitionError("contract_manifest_unreadable") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != CONTRACT_SCHEMA
        or value.get("exchange") != EXCHANGE
    ):
        raise AcquisitionError("contract_manifest_invalid")
    records = value.get("records")
    if not isinstance(records, list) or [
        x.get("symbol") for x in records if isinstance(x, dict)
    ] != list(SYMBOLS):
        raise AcquisitionError("contract_symbol_inventory_invalid")
    result = []
    for item in records:
        required = {
            "market_type": "perpetual",
            "linear": True,
            "inverse": False,
            "quote_asset": "USDT",
            "margin_asset": "USDT",
            "settle_asset": "USDT",
            "volume_unit": "base_asset",
            "contract_multiplier": 1.0,
        }
        if not isinstance(item, dict) or any(item.get(k) != v for k, v in required.items()):
            raise AcquisitionError("contract_record_invalid")
        result.append(
            Contract(
                item["symbol"],
                utc_ms(item.get("raw_availability_start_utc"), "raw_start"),
                utc_ms(item.get("raw_availability_end_utc"), "raw_end"),
                utc_ms(item.get("feature_availability_start_utc"), "feature_start"),
                utc_ms(item.get("feature_availability_end_utc"), "feature_end"),
            )
        )
    if any(
        x.raw_end_ms <= x.raw_start_ms or x.feature_end_ms <= x.feature_start_ms for x in result
    ):
        raise AcquisitionError("contract_interval_invalid")
    return result


def load_evidence(path: Path) -> dict[str, Any]:
    raw = safe_file_bytes(path)
    if sha256(raw) != EVIDENCE_SHA256:
        raise AcquisitionError("availability_evidence_sha256_not_approved")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AcquisitionError("availability_evidence_unreadable") from exc
    text = raw.decode("utf-8", "strict")
    if (
        not isinstance(value, dict)
        or value.get("artifact_kind") != "alpha_max_official_availability_evidence.v1"
        or "TONUSDT" not in text
        or "08:00" not in text
        or "12:00" not in text
    ):
        raise AcquisitionError("availability_evidence_ton_claim_invalid")
    return value


def funding_interval(symbol: str) -> int:
    return 14_400_000 if symbol == "TONUSDT" else 28_800_000


def expected_settlements(contract: Contract) -> list[int]:
    interval = funding_interval(contract.symbol)
    first = ((contract.feature_start_ms + interval - 1) // interval) * interval
    return list(range(first, contract.feature_end_ms, interval))


def normalize_funding(contract: Contract, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interval = funding_interval(contract.symbol)
    values: dict[int, dict[str, Any]] = {}
    previous = -1
    evidence_settlement = (
        contract.feature_start_ms // interval * interval - 2 * interval
        if contract.symbol == "TONUSDT"
        else None
    )
    for item in pages:
        if (
            not isinstance(item, dict)
            or item.get("symbol") != contract.symbol
            or type(item.get("fundingTime")) is not int
        ):
            raise AcquisitionError("funding_api_schema_invalid")
        source = item["fundingTime"]
        try:
            rate = float(item.get("fundingRate"))
        except (ValueError, TypeError) as exc:
            raise AcquisitionError("funding_api_rate_invalid") from exc
        settlement = source // interval * interval
        if (
            source <= previous
            or source - settlement not in range(FUNDING_JITTER_MS + 1)
            or not math.isfinite(rate)
        ):
            raise AcquisitionError("funding_api_order_or_jitter_invalid")
        previous = source
        if settlement in values:
            raise AcquisitionError("funding_api_settlement_collision")
        values[settlement] = {
            "timestamp_ms": source,
            "source_timestamp_ms": source,
            "exchange": EXCHANGE,
            "symbol": contract.symbol,
            "funding_rate": rate,
        }
    forbidden_settlement = (
        evidence_settlement + interval if evidence_settlement is not None else None
    )
    if forbidden_settlement is not None and forbidden_settlement in values:
        raise AcquisitionError("funding_ton_unavailable_settlement_present")
    if evidence_settlement is not None:
        if evidence_settlement not in values:
            raise AcquisitionError("funding_ton_availability_proof_invalid")
        del values[evidence_settlement]
    if sorted(values) != expected_settlements(contract):
        raise AcquisitionError("funding_api_owned_grid_invalid")
    return [values[x] for x in expected_settlements(contract)]


def parse_checksum(data: bytes, filename: str) -> str:
    fields = data.decode("utf-8", "strict").strip().split()
    if (
        len(fields) not in (1, 2)
        or len(fields[0]) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in fields[0])
    ):
        raise AcquisitionError("archive_checksum_schema_invalid")
    if len(fields) == 2 and fields[1].lstrip("*") != filename:
        raise AcquisitionError("archive_checksum_filename_invalid")
    return fields[0].lower()


def archive_trades(
    archive_source: io.BufferedReader, expected_member: str, nominal_start: int, nominal_end: int
) -> Iterable[tuple[int, int, float, float]]:
    """Lazily validate and yield archive trades in archive order."""
    try:
        with zipfile.ZipFile(archive_source) as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise AcquisitionError("archive_zip_schema_invalid")
            member = members[0]
            mode = member.external_attr >> 16
            if member.filename != expected_member or member.is_dir() or not stat.S_ISREG(mode):
                raise AcquisitionError("archive_zip_schema_invalid")
            with archive.open(member) as source:
                reader = csv.reader(
                    io.TextIOWrapper(source, encoding="utf-8", newline=""), strict=True
                )
                try:
                    first_row = next(reader)
                except StopIteration:
                    first_row = []
                header = [
                    "agg_trade_id",
                    "price",
                    "quantity",
                    "first_trade_id",
                    "last_trade_id",
                    "transact_time",
                    "is_buyer_maker",
                ]
                if first_row == header:
                    rows = reader
                else:
                    if first_row and any(
                        value.lower() in {"id", "price", "time"}
                        or "id" in value.lower()
                        or "price" in value.lower()
                        or "time" in value.lower()
                        for value in first_row
                    ):
                        raise AcquisitionError("archive_csv_header_invalid")
                    rows = itertools.chain((first_row,), reader) if first_row else reader
                for fields in rows:
                    if len(fields) != 7 or any(value == "" for value in fields):
                        raise AcquisitionError("archive_csv_null_or_schema_invalid")
                    if (
                        not fields[0].isascii()
                        or not fields[0].isdigit()
                        or not fields[3].isascii()
                        or not fields[3].isdigit()
                        or not fields[4].isascii()
                        or not fields[4].isdigit()
                        or not fields[5].isascii()
                        or not fields[5].isdigit()
                        or POSITIVE_DECIMAL.fullmatch(fields[1]) is None
                        or POSITIVE_DECIMAL.fullmatch(fields[2]) is None
                        or fields[6].lower() not in {"true", "false"}
                    ):
                        raise AcquisitionError("archive_trade_value_or_month_bounds_invalid")
                    try:
                        aggregate_id = int(fields[0])
                        price = float(fields[1])
                        quantity = float(fields[2])
                        first_id = int(fields[3])
                        last_id = int(fields[4])
                        timestamp = int(fields[5])
                    except ValueError as exc:
                        raise AcquisitionError("archive_csv_null_or_schema_invalid") from exc
                    if (
                        aggregate_id < 0
                        or first_id < 0
                        or last_id < first_id
                        or timestamp < nominal_start
                        or timestamp >= nominal_end
                        or not math.isfinite(price)
                        or price <= 0
                        or not math.isfinite(quantity)
                        or quantity < 0
                    ):
                        raise AcquisitionError("archive_trade_value_or_month_bounds_invalid")
                    yield aggregate_id, timestamp, price, quantity
    except csv.Error as exc:
        raise AcquisitionError("archive_csv_schema_invalid") from exc
    except (zipfile.BadZipFile, UnicodeError) as exc:
        raise AcquisitionError("archive_zip_or_csv_invalid") from exc


def replay_archive_trades(
    trades: Iterable[tuple[int, int, float, float]],
    start_ms: int,
    end_ms: int,
    carry: float | None,
    duplicate_aggregate_ids: frozenset[int] = frozenset(),
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Replay ordered trades into the owned OHLCV seconds."""
    size = (end_ms - start_ms) // 1000
    op, hi, lo, cl = (array("d", [math.nan]) * size for _ in range(4))
    vol = array("d", [0.0]) * size
    first = last = None
    last_aggregate = last_timestamp = None
    observed_duplicate_aggregate_ids: set[int] = set()
    count = 0
    for aggregate_id, timestamp, price, quantity in trades:
        duplicate = last_aggregate is not None and aggregate_id == last_aggregate
        if (
            (last_aggregate is not None and aggregate_id < last_aggregate)
            or (
                duplicate
                and (
                    aggregate_id not in duplicate_aggregate_ids
                    or aggregate_id in observed_duplicate_aggregate_ids
                    or last_timestamp is None
                    or timestamp <= last_timestamp
                )
            )
            or (last_timestamp is not None and timestamp < last_timestamp)
        ):
            raise AcquisitionError("archive_trade_order_invalid")
        if duplicate:
            observed_duplicate_aggregate_ids.add(aggregate_id)
        last_aggregate, last_timestamp = aggregate_id, timestamp
        first = first or (timestamp, aggregate_id)
        last = (timestamp, aggregate_id)
        count += 1
        if timestamp < start_ms:
            carry = price
        elif timestamp < end_ms:
            second = (timestamp - start_ms) // 1000
            if math.isnan(op[second]):
                op[second] = hi[second] = lo[second] = price
            else:
                hi[second], lo[second] = max(hi[second], price), min(lo[second], price)
            cl[second], vol[second] = price, vol[second] + quantity
    if observed_duplicate_aggregate_ids != set(duplicate_aggregate_ids):
        raise AcquisitionError("archive_trade_order_invalid")
    if not count:
        raise AcquisitionError("archive_trade_empty")
    if math.isnan(op[0]) and carry is None:
        raise AcquisitionError("raw_first_owned_second_has_no_official_close")
    close = pl.col("close").fill_nan(None).forward_fill()
    if carry is not None:
        close = close.fill_null(carry)
    frame = (
        pl.DataFrame(
            {
                "datetime": pl.Series(
                    "datetime", range(start_ms, end_ms, 1000), dtype=pl.Int64
                ).cast(pl.Datetime("ms")),
                "open": pl.Series("open", op, dtype=pl.Float64),
                "high": pl.Series("high", hi, dtype=pl.Float64),
                "low": pl.Series("low", lo, dtype=pl.Float64),
                "close": pl.Series("close", cl, dtype=pl.Float64),
                "volume": pl.Series("volume", vol, dtype=pl.Float64),
            }
        )
        .with_columns(close.alias("close"))
        .with_columns(
            *[
                pl.col(name).fill_nan(None).fill_null(pl.col("close")).alias(name)
                for name in ("open", "high", "low")
            ]
        )
    )
    return frame, {
        "trade_count": count,
        "first_trade": first,
        "last_trade": last,
        "carry_close": frame["close"][-1],
    }


def canonical_frame_from_archive(
    archive_source: io.BufferedReader,
    start_ms: int,
    end_ms: int,
    carry: float | None,
    expected_member: str,
    nominal_start: int,
    nominal_end: int,
    scratch_root: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    scratch = scratch_directory(scratch_root)
    chunks: list[Path] = []
    identities: dict[Path, tuple[int, int]] = {}
    records: list[bytes] = []

    def flush() -> None:
        nonlocal records
        if not records:
            return
        records.sort(key=lambda value: ORDER_RECORD.unpack(value)[0])
        assert_host_reserve(len(records) * ORDER_RECORD.size)
        fd, name = tempfile.mkstemp(prefix=".aggtrades-chunk-", suffix=".bin", dir=scratch)
        target = Path(name)
        with os.fdopen(fd, "wb") as sink:
            identities[target] = scratch_file_identity(target)
            sink.write(b"".join(records))
            sink.flush()
            os.fsync(sink.fileno())
        fsync_directory(scratch)
        chunks.append(target)
        records = []

    def record(source: io.BufferedReader) -> tuple[int, int, float, float] | None:
        data = source.read(ORDER_RECORD.size)
        if not data:
            return None
        if len(data) != ORDER_RECORD.size:
            raise AcquisitionError("archive_order_canonicalization_failed")
        return ORDER_RECORD.unpack(data)

    try:
        for aggregate_id, timestamp, price, quantity in archive_trades(
            archive_source, expected_member, nominal_start, nominal_end
        ):
            records.append(ORDER_RECORD.pack(aggregate_id, timestamp, price, quantity))
            if len(records) == ORDER_CHUNK_RECORDS:
                flush()
        flush()
        if not chunks:
            return replay_archive_trades((), start_ms, end_ms, carry)
        while len(chunks) > 1:
            merged: list[Path] = []
            for offset in range(0, len(chunks), ORDER_MERGE_FAN_IN):
                group = chunks[offset : offset + ORDER_MERGE_FAN_IN]
                assert_host_reserve(sum(stable_file_stat(item).st_size for item in group))
                fd, name = tempfile.mkstemp(prefix=".aggtrades-merge-", suffix=".bin", dir=scratch)
                target = Path(name)
                with os.fdopen(fd, "wb") as sink:
                    identities[target] = scratch_file_identity(target)
                    with ExitStack() as stack:
                        sources = [stack.enter_context(stable_file(item)) for item in group]
                        heap = [
                            (value[0], index, value)
                            for index, source in enumerate(sources)
                            if (value := record(source)) is not None
                        ]
                        heapq.heapify(heap)
                        while heap:
                            _, index, value = heapq.heappop(heap)
                            sink.write(ORDER_RECORD.pack(*value))
                            if (following := record(sources[index])) is not None:
                                heapq.heappush(heap, (following[0], index, following))
                    sink.flush()
                    os.fsync(sink.fileno())
                for item in group:
                    identity = identities[item]
                    remove_aggtrades_file(item, identity)
                    del identities[item]
                fsync_directory(scratch)
                merged.append(target)
            chunks = merged
        with stable_file(chunks[0]) as source:
            return replay_archive_trades(
                iter(lambda: record(source), None), start_ms, end_ms, carry
            )
    except (OSError, struct.error) as exc:
        raise AcquisitionError("archive_order_canonicalization_failed") from exc
    finally:
        primary = sys.exception()
        cleanup_failures = 0
        for item, identity in list(identities.items()):
            try:
                remove_aggtrades_file(item, identity)
            except OSError, AcquisitionError:
                cleanup_failures += 1
        if cleanup_failures:
            if primary is None:
                raise AcquisitionError("archive_order_canonicalization_failed")
            primary.add_note(f"archive_order_cleanup_failed:{cleanup_failures}")


def frame_from_archive(
    path: Path,
    symbol: str,
    start_ms: int,
    end_ms: int,
    carry: float | None,
    month: str,
    scratch_root: Path | None = None,
    archive_receipt: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if start_ms % 1000 or end_ms % 1000 or end_ms <= start_ms:
        raise AcquisitionError("raw_owned_bounds_invalid")
    if not isinstance(symbol, str) or not symbol:
        raise AcquisitionError("archive_symbol_invalid")
    if not isinstance(month, str):
        raise AcquisitionError("archive_month_invalid")
    try:
        parsed_month = datetime.strptime(month, "%Y-%m")
    except ValueError as exc:
        raise AcquisitionError("archive_month_invalid") from exc
    if parsed_month.strftime("%Y-%m") != month:
        raise AcquisitionError("archive_month_invalid")
    nominal_start = int(parsed_month.replace(tzinfo=UTC).timestamp() * 1000)
    expected_member = f"{symbol}-aggTrades-{month}.csv"
    nominal_end = month_bounds(datetime.fromtimestamp(nominal_start / 1000, UTC))[1]
    allowlisted = CANONICAL_ORDER_ARCHIVES.get((symbol, month))
    duplicate_allowlisted = AUTHENTICATED_DUPLICATE_AGGREGATE_ARCHIVES.get((symbol, month))
    with stable_file(path) as archive_source:
        if (
            archive_receipt is not None
            or allowlisted is not None
            or duplicate_allowlisted is not None
        ):
            actual_digest, actual_byte_count = authenticated_archive_identity(
                archive_source, archive_receipt
            )
        else:
            actual_digest = actual_byte_count = None
        if allowlisted is not None:
            digest, byte_count = allowlisted
            if (
                actual_byte_count == byte_count
                and actual_digest == digest
                and isinstance(archive_receipt, dict)
                and archive_receipt.get("sha256") == digest
                and archive_receipt.get("byte_count") == byte_count
            ):
                if scratch_root is None:
                    raise AcquisitionError("archive_order_scratch_root_required")
                return canonical_frame_from_archive(
                    archive_source,
                    start_ms,
                    end_ms,
                    carry,
                    expected_member,
                    nominal_start,
                    nominal_end,
                    scratch_root,
                )
        if duplicate_allowlisted is not None:
            digest, byte_count, duplicate_aggregate_ids = duplicate_allowlisted
            if (
                actual_byte_count == byte_count
                and actual_digest == digest
                and isinstance(archive_receipt, dict)
                and archive_receipt.get("sha256") == digest
                and archive_receipt.get("byte_count") == byte_count
            ):
                return replay_archive_trades(
                    archive_trades(archive_source, expected_member, nominal_start, nominal_end),
                    start_ms,
                    end_ms,
                    carry,
                    duplicate_aggregate_ids,
                )
        return replay_archive_trades(
            archive_trades(archive_source, expected_member, nominal_start, nominal_end),
            start_ms,
            end_ms,
            carry,
        )


def root_identity(path: Path) -> list[int]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise AcquisitionError("unsafe_root") from exc
    try:
        item = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat.S_ISDIR(item.st_mode):
        raise AcquisitionError("unsafe_root")
    return [item.st_dev, item.st_ino]


def lexical(path: Path) -> Path:
    if not path.is_absolute():
        raise AcquisitionError("root_must_be_absolute")
    return Path(os.path.normpath(str(path)))


def trusted_parent_directory(
    item: os.stat_result,
    immediate: bool,
    *,
    namespace_root: bool = False,
    namespace_root_uid: int | None = None,
) -> None:
    if not stat.S_ISDIR(item.st_mode):
        raise AcquisitionError("unsafe_root_parent")
    writable = item.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if namespace_root:
        # A user-manager mount namespace maps host root ownership to an overflow ID.
        # The opened "/" anchor is safe when it is not group/other writable.
        if writable:
            raise AcquisitionError("unsafe_root_parent")
    elif immediate:
        if item.st_uid != os.getuid() or writable:
            raise AcquisitionError("unsafe_root_parent")
    else:
        trusted_uids = {0, os.getuid()}
        if namespace_root_uid is not None:
            trusted_uids.add(namespace_root_uid)
        if item.st_uid not in trusted_uids or (writable and not (item.st_mode & stat.S_ISVTX)):
            raise AcquisitionError("unsafe_root_parent")


def safe_existing_directory(path: Path) -> list[int]:
    """Open a trusted parent chain without following any path component."""
    current_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    namespace_root_uid = os.fstat(current_fd).st_uid
    try:
        trusted_parent_directory(
            os.fstat(current_fd),
            not path.parts[1:],
            namespace_root=True,
            namespace_root_uid=namespace_root_uid,
        )
        components = path.parts[1:]
        for index, component in enumerate(components):
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
            except OSError as exc:
                raise AcquisitionError("unsafe_root_parent") from exc
            os.close(current_fd)
            current_fd = next_fd
            trusted_parent_directory(
                os.fstat(current_fd),
                index == len(components) - 1,
                namespace_root_uid=namespace_root_uid,
            )
        item = os.fstat(current_fd)
        return [item.st_dev, item.st_ino]
    finally:
        os.close(current_fd)


def safe_root(path: Path, create: bool) -> tuple[list[int], list[int]]:
    """Validate all existing parents and create/open the final root via dir_fd."""
    parent = path.parent
    parent_identity = safe_existing_directory(parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            root_fd = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            if not create:
                raise AcquisitionError("unsafe_root")
            os.mkdir(path.name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            try:
                root_fd = os.open(
                    path.name,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise AcquisitionError("unsafe_root") from exc
        except OSError as exc:
            raise AcquisitionError("unsafe_root") from exc
        try:
            item = os.fstat(root_fd)
            if (
                not stat.S_ISDIR(item.st_mode)
                or item.st_uid != os.getuid()
                or item.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            ):
                raise AcquisitionError("unsafe_root")
            return parent_identity, [item.st_dev, item.st_ino]
        finally:
            os.close(root_fd)
    finally:
        os.close(parent_fd)


def assert_roots(output: Path, report: Path, forbidden: list[Path], execute: bool) -> None:
    output, report = lexical(output), lexical(report)
    scratch_output, scratch_report = scratch_path(output), scratch_path(report)
    if not forbidden:
        raise AcquisitionError("forbidden_root_required")
    forbidden_paths = [lexical(item) for item in forbidden]
    for candidate in forbidden_paths:
        if paths_overlap(output, candidate) or paths_overlap(report, candidate):
            raise AcquisitionError("root_is_forbidden")
    if paths_overlap(output, report):
        raise AcquisitionError("roots_overlap")
    reserved = (output, report, scratch_output, scratch_report)
    if any(paths_overlap(left, right) for left, right in itertools.combinations(reserved, 2)):
        raise AcquisitionError("scratch_roots_overlap")
    if any(
        paths_overlap(reserved_path, forbidden_path)
        for reserved_path in (scratch_output, scratch_report)
        for forbidden_path in forbidden_paths
    ):
        raise AcquisitionError("scratch_root_is_forbidden")
    safe_existing_directory(output.parent)
    safe_existing_directory(report.parent)
    output_exists = output.exists() or output.is_symlink()
    report_exists = report.exists() or report.is_symlink()
    if output_exists:
        safe_root(output, create=False)
    if report_exists:
        safe_root(report, create=False)
    # A report containing no artifacts or only an immutable plan is the intentional
    # pre-publication/plan-only boundary.
    report_prefix = {x.name for x in report.iterdir()} if report_exists else set()
    if (
        execute
        and output_exists != report_exists
        and not (not output_exists and report_exists and report_prefix in (set(), {"plan.json"}))
    ):
        raise AcquisitionError("roots_resume_pair_invalid")


def assert_input_paths(contract: Path, evidence: Path, forbidden: list[Path]) -> None:
    contract, evidence = lexical(contract), lexical(evidence)
    forbidden_paths = [lexical(item) for item in forbidden]
    if any(
        paths_overlap(source, forbidden_root)
        for source in (contract, evidence)
        for forbidden_root in forbidden_paths
    ):
        raise AcquisitionError("input_is_forbidden")


def _journal_lines(path: Path) -> list[bytes]:
    if not path.exists():
        return []
    raw = safe_file_bytes(path)
    if raw and not raw.endswith(b"\n"):
        raise AcquisitionError("journal_fragment_invalid")
    lines = raw.splitlines()
    for line in lines:
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AcquisitionError("journal_noncanonical") from exc
        if canonical_bytes(value) != line + b"\n":
            raise AcquisitionError("journal_noncanonical")
    return lines


def recover_pending_event(report: Path) -> None:
    pending = report / "acquisition.journal.pending.json"
    if not pending.exists():
        return
    event = read_json(pending)
    if not isinstance(event, dict):
        raise AcquisitionError("journal_pending_invalid")
    data = canonical_bytes(event)
    journal_path = report / "acquisition.journal.jsonl"
    raw = safe_file_bytes(journal_path) if journal_path.exists() else b""
    _complete, _separator, fragment = raw.rpartition(b"\n")
    if fragment:
        if not data.startswith(fragment):
            raise AcquisitionError("journal_fragment_invalid")
        fd = os.open(journal_path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
        try:
            missing = data[len(fragment) :]
            if os.write(fd, missing) != len(missing):
                raise AcquisitionError("journal_write_failed")
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(report)
    lines = _journal_lines(journal_path)
    if not lines or lines[-1] != data[:-1]:
        fd = os.open(
            journal_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            item = os.fstat(fd)
            if not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
                raise AcquisitionError("unsafe_journal")
            if os.write(fd, data) != len(data):
                raise AcquisitionError("journal_write_failed")
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(report)
    else:
        fd = os.open(journal_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(report)
    pending.unlink()
    fsync_directory(report)


def journal(report: Path, event: dict[str, Any]) -> None:
    """Durably publish an exact pending event before appending it to the journal."""
    recover_pending_event(report)
    immutable_json(report / "acquisition.journal.pending.json", event, report)
    recover_pending_event(report)


_VERIFIER_CODE_FD: ContextVar[int | None] = ContextVar("verifier_code_fd", default=None)


def code_hash() -> str:
    descriptor = _VERIFIER_CODE_FD.get()
    if descriptor is None:
        return file_sha256(Path(__file__))
    try:
        duplicate = os.dup(descriptor)
    except OSError as exc:
        raise AcquisitionError("verifier_code_fd_invalid") from exc
    try:
        before = os.fstat(duplicate)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise AcquisitionError("verifier_code_fd_invalid")
        digest = hashlib.sha256()
        os.lseek(duplicate, 0, os.SEEK_SET)
        while block := os.read(duplicate, 1 << 20):
            digest.update(block)
        after = os.fstat(duplicate)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise AcquisitionError("verifier_code_fd_changed")
        return digest.hexdigest()
    except OSError as exc:
        raise AcquisitionError("verifier_code_fd_invalid") from exc
    finally:
        os.close(duplicate)


def ownership_marker(output: Path, report: Path, run_id: str) -> dict[str, Any]:
    output_parent, output_identity = safe_root(output, create=False)
    report_parent, report_identity = safe_root(report, create=False)
    return {
        "schema": "alpha_max_owned_roots.v2",
        "run_id": run_id,
        "output_path": str(lexical(output)),
        "report_path": str(lexical(report)),
        "output_parent_identity": output_parent,
        "report_parent_identity": report_parent,
        "output_identity": output_identity,
        "report_identity": report_identity,
        "uid": os.getuid(),
        "contract_sha256": CONTRACT_SHA256,
        "availability_evidence_sha256": EVIDENCE_SHA256,
        "derivation_version": DERIVATION_VERSION,
        "code_sha256": code_hash(),
    }


def ownership(output: Path, report: Path, run_id: str) -> None:
    marker = ownership_marker(output, report, run_id)
    data = canonical_bytes(marker)
    for root in (output, report):
        path = root / ".alpha_max_owner.json"
        if path.exists() or path.is_symlink():
            if stable_file_bytes(path) != data:
                raise AcquisitionError("ownership_marker_invalid")
        else:
            atomic_write(path, data, scratch_root=root)


def recover_first_execute_prefix(output: Path, report: Path, plan_data: bytes, run_id: str) -> bool:
    """Accept only prefixes of plan-only → paired ownership initialization."""
    plan_path = report / "plan.json"
    if not report.exists() or not report.is_dir() or not plan_path.exists():
        return False
    checked_file(plan_path)
    if plan_path.read_bytes() != plan_data:
        return False
    if {item.name for item in report.iterdir()} - {"plan.json", ".alpha_max_owner.json"}:
        return False
    if output.exists() and (
        not output.is_dir() or {item.name for item in output.iterdir()} - {".alpha_max_owner.json"}
    ):
        return False
    if not output.exists():
        safe_root(output, create=True)
    ownership(output, report, run_id)
    return True


@contextmanager
def owned_run_lock(report: Path, exclusive: bool) -> Iterable[None]:
    """Serialize one authenticated run without creating a verifier-side lock file."""
    marker = report / ".alpha_max_owner.json"
    with stable_file(marker) as source:
        fcntl.flock(source.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(source.fileno(), fcntl.LOCK_UN)


class OwnedRunLock:
    def __init__(self, report: Path, exclusive: bool) -> None:
        marker = report / ".alpha_max_owner.json"
        checked_file(marker)
        self.fd = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        fcntl.flock(self.fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def close(self) -> None:
        if self.fd >= 0:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = -1

    def __del__(self) -> None:
        self.close()


_ACTIVE_EXECUTE_LOCK: ContextVar[OwnedRunLock | None] = ContextVar(
    "active_execute_lock", default=None
)


def release_execute_lock(function: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        lock_token = _ACTIVE_EXECUTE_LOCK.set(None)
        try:
            return function(*args, **kwargs)
        finally:
            lock = _ACTIVE_EXECUTE_LOCK.get()
            try:
                if lock is not None:
                    lock.close()
            finally:
                _ACTIVE_EXECUTE_LOCK.reset(lock_token)

    return wrapped


def recover_owned_hardlink_prefixes(output: Path, report: Path, plan_data: bytes) -> None:
    """Retire only scratch links paired with authenticated plan/owner destinations."""
    expected = [
        report / "plan.json",
        report / ".alpha_max_owner.json",
        output / ".alpha_max_owner.json",
    ]
    for root in (output, report):
        scratch = scratch_path(root)
        if not scratch.exists():
            continue
        item = scratch.lstat()
        if not stat.S_ISDIR(item.st_mode) or item.st_uid != os.getuid():
            raise AcquisitionError("unsafe_scratch_directory")
        for candidate in scratch.iterdir():
            candidate_item = candidate.lstat()
            if not stat.S_ISREG(candidate_item.st_mode) or candidate_item.st_nlink != 2:
                continue
            if not candidate.name.startswith(".acquire-"):
                raise AcquisitionError("unsafe_scratch_entry")
            for destination in expected:
                if destination.exists() and candidate.read_bytes() == destination.read_bytes():
                    if destination == report / "plan.json" and candidate.read_bytes() != plan_data:
                        continue
                    candidate.unlink()
                    fsync_directory(scratch)
                    break


def verify_ownership(output: Path, report: Path, run_id: str) -> dict[str, Any]:
    output_parent, output_identity = safe_root(output, create=False)
    report_parent, report_identity = safe_root(report, create=False)
    left = read_json(output / ".alpha_max_owner.json")
    right = read_json(report / ".alpha_max_owner.json")
    required = {
        "schema": "alpha_max_owned_roots.v2",
        "run_id": run_id,
        "output_path": str(lexical(output)),
        "report_path": str(lexical(report)),
        "output_parent_identity": output_parent,
        "report_parent_identity": report_parent,
        "output_identity": output_identity,
        "report_identity": report_identity,
        "uid": os.getuid(),
        "contract_sha256": CONTRACT_SHA256,
        "availability_evidence_sha256": EVIDENCE_SHA256,
        "derivation_version": DERIVATION_VERSION,
        "code_sha256": code_hash(),
    }
    if left != right or any(left.get(k) != v for k, v in required.items()):
        raise AcquisitionError("ownership_marker_invalid")
    return left


def checked_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_HOSTS
        or parsed.username
        or parsed.password
    ):
        raise AcquisitionError("official_url_invalid")


def _request_fields(url: str, query: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    checked_url(url)
    parsed = urllib.parse.urlsplit(url)
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if len({key for key, _value in pairs}) != len(pairs):
        raise AcquisitionError("official_query_mismatch")
    actual = dict(pairs)
    expected = {str(k): str(v) for k, v in (query or actual).items()}
    if actual != expected:
        raise AcquisitionError("official_query_mismatch")
    return url, expected


def canonical_utc(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AcquisitionError("cached_request_receipt_invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AcquisitionError("cached_request_receipt_invalid") from exc
    if parsed.tzinfo is None:
        raise AcquisitionError("cached_request_receipt_invalid")
    normalized = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if value != normalized:
        raise AcquisitionError("cached_request_receipt_invalid")
    return value


def request_receipt(
    path: Path, requested_url: str, query: dict[str, Any] | None = None
) -> dict[str, Any]:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt = read_json(receipt_path)
    requested, expected_query = _request_fields(requested_url, query)
    required = {
        "schema": "official_request_receipt.v1",
        "requested_url": requested,
        "final_url": requested,
        "final_host": urllib.parse.urlsplit(requested).hostname,
        "query": expected_query,
        "retrieved_at_utc": canonical_utc(receipt.get("retrieved_at_utc")),
        "byte_count": stable_file_stat(path).st_size,
        "sha256": file_sha256(path),
    }
    if (
        stable_file_bytes(receipt_path) != canonical_bytes(receipt)
        or set(receipt) != set(required)
        or receipt != required
    ):
        raise AcquisitionError("cached_request_receipt_invalid")
    checked_url(requested)
    return receipt


def fetch_receipt(
    url: str,
    destination: Path,
    query: dict[str, Any] | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    requested, expected_query = _request_fields(url, query)
    receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
    if destination.exists() or receipt_path.exists():
        if not destination.exists():
            raise AcquisitionError("cached_request_receipt_invalid")
        if receipt_path.exists():
            return request_receipt(destination, requested, expected_query)
        # An unreceipted payload is adopted only after an independent official
        # retrieval produces the exact same bytes.
        recovered = (
            scratch_directory(scratch_root or destination.parent) / f".recovery-{destination.name}"
        )
        try:
            fresh = fetch_receipt(requested, recovered, expected_query, scratch_root)
            if file_sha256(destination) != fresh["sha256"]:
                raise AcquisitionError("cached_request_orphan_mismatch")
            immutable_json(receipt_path, fresh, scratch_root)
            return request_receipt(destination, requested, expected_query)
        finally:
            for path in (recovered, recovered.with_suffix(recovered.suffix + ".receipt.json")):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                fsync_directory(path.parent)
    mkdir_durable(destination.parent)
    fsync_directory(destination.parent)
    for attempt in range(DOWNLOAD_ATTEMPTS):
        fd, temporary = scratch_file(scratch_root or destination.parent, ".partial-")
        digest = hashlib.sha256()
        count = 0
        try:
            request = urllib.request.Request(
                requested, headers={"User-Agent": "LuminaQuant-official-acquirer/1"}
            )
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                os.fdopen(fd, "wb") as sink,
            ):
                final = response.geturl()
                checked_url(final)
                if final != requested:
                    raise AcquisitionError("official_redirect_identity_invalid")
                while block := response.read(DOWNLOAD_CHUNK):
                    assert_host_reserve(len(block))
                    digest.update(block)
                    count += len(block)
                    sink.write(block)
                sink.flush()
                os.fsync(sink.fileno())
            receipt = {
                "schema": "official_request_receipt.v1",
                "requested_url": requested,
                "final_url": final,
                "final_host": urllib.parse.urlsplit(final).hostname,
                "query": expected_query,
                "retrieved_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "byte_count": count,
                "sha256": digest.hexdigest(),
            }
            os.link(temporary, destination)
            fsync_directory(destination.parent)
            immutable_json(receipt_path, receipt, scratch_root)
            return receipt
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            if attempt + 1 == DOWNLOAD_ATTEMPTS:
                raise AcquisitionError("official_download_failed") from exc
            time.sleep(0.25 * (2**attempt))
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            fsync_directory(Path(temporary).parent)
    raise AssertionError("unreachable")


def publish_frame(path: Path, frame: pl.DataFrame, scratch_root: Path | None = None) -> str:
    if path.exists() or path.is_symlink():
        raise AcquisitionError("preseeded_output_rejected")
    assert_host_reserve(max(int(frame.estimated_size()), 1 << 20))
    mkdir_durable(path.parent)
    fsync_directory(path.parent)
    fd, temp = scratch_file(scratch_root or path.parent, ".acquire-", ".parquet")
    os.close(fd)
    try:
        frame.write_parquet(temp)
        # The inode must be durable before its no-replace hard-link publication.
        with stable_file(Path(temp)) as source:
            source.flush()
            os.fsync(source.fileno())
        digest = file_sha256(Path(temp))
        assert_host_reserve()
        os.link(temp, path)
        fsync_directory(path.parent)
        os.unlink(temp)
        fsync_directory(Path(temp).parent)
        # Reopen only after the temporary hard link is gone, so the target is singly linked.
        if file_sha256(path) != digest:
            raise AcquisitionError("published_parquet_identity_invalid")
        return digest
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        fsync_directory(Path(temp).parent)


def frame_sha256(frame: pl.DataFrame, directory: Path, scratch_root: Path | None = None) -> str:
    assert_host_reserve(max(int(frame.estimated_size()), 1 << 20))
    fd, temporary = scratch_file(scratch_root or directory, ".derive-", ".parquet")
    os.close(fd)
    try:
        frame.write_parquet(temporary)
        return file_sha256(Path(temporary))
    finally:
        os.unlink(temporary)
        fsync_directory(Path(temporary).parent)


def expected_partition_receipt(
    relative: str,
    source_hash: str,
    output_path: Path,
    rows: int,
    start: int,
    end: int,
    input_carry: float | None,
    output_carry: float | None,
    page_hashes: list[str],
) -> dict[str, Any]:
    return {
        "schema": "alpha_max_partition_receipt.v2",
        "path": relative,
        "source_sha256": source_hash,
        "output_sha256": file_sha256(output_path),
        "rows": rows,
        "start_ms": start,
        "end_ms": end,
        "input_carry_close": input_carry,
        "output_carry_close": output_carry,
        "derivation_version": DERIVATION_VERSION,
        "code_sha256": code_hash(),
        "page_hashes": page_hashes,
    }


def partition_receipt(
    report: Path,
    relative: str,
    source_hash: str,
    output_path: Path,
    rows: int,
    start: int,
    end: int,
    input_carry: float | None,
    output_carry: float | None,
    page_hashes: list[str],
) -> dict[str, Any]:
    receipt = expected_partition_receipt(
        relative, source_hash, output_path, rows, start, end, input_carry, output_carry, page_hashes
    )
    path = partition_path(report, relative)
    if path.exists() or path.is_symlink():
        if read_json(path) != receipt:
            raise AcquisitionError("partition_resume_receipt_invalid")
    else:
        immutable_json(path, receipt, report)
    return receipt


def verified_partition(
    report: Path, relative: str, output: Path, receipt: dict[str, Any], frame: pl.DataFrame
) -> bool:
    path = partition_path(report, relative)
    if path.exists() and not output.exists():
        raise AcquisitionError("partition_resume_receipt_invalid")
    if not path.exists():
        return False
    if (
        read_json(path) != receipt
        or file_sha256(output) != receipt["output_sha256"]
        or file_sha256(output) != frame_sha256(frame, output.parent, output)
        or not parquet_frame(output).equals(frame)
    ):
        raise AcquisitionError("partition_resume_receipt_invalid")
    return True


def archive_evidence_paths(report: Path, symbol: str, label: str) -> dict[str, Path]:
    root = report / "provenance" / "archive-evidence" / symbol
    return {
        kind: root / f"{label}.{kind}.json"
        for kind in ("derivation", "retirement-intent", "deletion")
    }


def detached_archive_receipt(path: Path, url: str) -> dict[str, Any]:
    """Validate a retained canonical request receipt without requiring its retired body."""
    try:
        raw = stable_file_bytes(path)
        receipt = read_json(path)
        requested, query = _request_fields(url, None)
        required = {
            "schema": "official_request_receipt.v1",
            "requested_url": requested,
            "final_url": requested,
            "final_host": urllib.parse.urlsplit(requested).hostname,
            "query": query,
            "retrieved_at_utc": canonical_utc(receipt.get("retrieved_at_utc")),
            "byte_count": receipt.get("byte_count"),
            "sha256": receipt.get("sha256"),
        }
        if (
            raw != canonical_bytes(receipt)
            or set(receipt) != set(required)
            or receipt != required
            or not isinstance(receipt["byte_count"], int)
            or receipt["byte_count"] < 0
            or not isinstance(receipt["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])
        ):
            raise AcquisitionError("retired_archive_request_receipt_invalid")
    except (AcquisitionError, KeyError, TypeError, ValueError) as exc:
        if str(exc) == "retired_archive_request_receipt_invalid":
            raise
        raise AcquisitionError("retired_archive_request_receipt_invalid") from exc
    return receipt


def expected_archive_derivation(
    archive: Path,
    archive_receipt_path: Path,
    archive_url: str,
    target: Path,
    partition: dict[str, Any],
    prior: str | None,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "alpha_max_archive_derivation_receipt.v1",
        "output_path": partition["path"],
        "output_sha256": file_sha256(target),
        "output_byte_count": stable_file_stat(target).st_size,
        "rows": partition["rows"],
        "start_ms": partition["start_ms"],
        "end_ms": partition["end_ms"],
        "input_carry_close": partition["input_carry_close"],
        "output_carry_close": partition["output_carry_close"],
        "archive_url": archive_url,
        "archive_member": archive.name.removesuffix(".zip") + ".csv",
        "archive_sha256": receipt["sha256"],
        "archive_byte_count": receipt["byte_count"],
        "archive_request_receipt_sha256": sha256(stable_file_bytes(archive_receipt_path)),
        "checksum_payload_sha256": partition["source_sha256"],
        "checksum_request_receipt_sha256": sha256(
            stable_file_bytes(archive.with_name(archive.name + ".CHECKSUM.receipt.json"))
        ),
        "partition_receipt_sha256": sha256(canonical_bytes(partition)),
        "prior_derivation_receipt_sha256": prior,
        "derivation_version": DERIVATION_VERSION,
        "code_sha256": code_hash(),
    }


def expected_archive_intent(archive: Path, derivation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "alpha_max_archive_retirement_intent.v1",
        "derivation_receipt_sha256": sha256(canonical_bytes(derivation)),
        "partition_receipt_sha256": derivation["partition_receipt_sha256"],
        "archive_request_receipt_sha256": derivation["archive_request_receipt_sha256"],
        "archive_relative_path": str(archive.relative_to(archive.parents[3])),
        "archive_sha256": derivation["archive_sha256"],
        "archive_byte_count": derivation["archive_byte_count"],
        "output_path": derivation["output_path"],
        "output_sha256": derivation["output_sha256"],
    }


def archive_evidence(
    report: Path,
    symbol: str,
    label: str,
    archive: Path,
    archive_receipt_path: Path,
    archive_url: str,
    target: Path,
    partition: dict[str, Any],
    prior: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = archive_evidence_paths(report, symbol, label)
    derivation, intent, deletion = (read_json(paths[kind]) for kind in paths)
    receipt = detached_archive_receipt(archive_receipt_path, archive_url)
    required_derivation = expected_archive_derivation(
        archive, archive_receipt_path, archive_url, target, partition, prior, receipt
    )
    if derivation != required_derivation:
        raise AcquisitionError("archive_derivation_receipt_invalid")
    required_intent = expected_archive_intent(archive, derivation)
    if intent != required_intent:
        raise AcquisitionError("archive_retirement_intent_invalid")
    if archive.exists() or archive.is_symlink():
        raise AcquisitionError("retired_archive_body_present")
    required_deletion = {
        "schema": "alpha_max_archive_deletion_receipt.v1",
        "retirement_intent_sha256": sha256(canonical_bytes(intent)),
        "derivation_receipt_sha256": sha256(canonical_bytes(derivation)),
        "archive_relative_path": str(archive.relative_to(report)),
        "archive_sha256": receipt["sha256"],
        "archive_byte_count": receipt["byte_count"],
        "archive_absent": True,
    }
    if deletion != required_deletion:
        raise AcquisitionError("archive_deletion_receipt_invalid")
    return derivation, intent, deletion


def acquire_archive(
    contract: Contract, output: Path, report: Path, chosen_months: set[str] | None = None
) -> None:
    carry: float | None = None
    prior_derivation: str | None = None
    for month in month_starts(contract.raw_start_ms, contract.raw_end_ms):
        label = month.strftime("%Y-%m")
        if chosen_months is not None and label not in chosen_months:
            carry = None
            prior_derivation = None
            continue
        nominal_start, nominal_end = month_bounds(month)
        start, end = (
            max(contract.raw_start_ms, nominal_start),
            min(contract.raw_end_ms, nominal_end),
        )
        filename = f"{contract.symbol}-aggTrades-{label}.zip"
        url = f"{ARCHIVE_BASE}/{contract.symbol}/{filename}"
        archive = report / "provenance" / "archives" / contract.symbol / filename
        archive_receipt_path = archive.with_suffix(".zip.receipt.json")
        checksum = archive.with_name(filename + ".CHECKSUM")
        relative = str(Path("market_ohlcv_1s") / EXCHANGE / contract.symbol / f"{label}.parquet")
        target = output / relative
        paths = archive_evidence_paths(report, contract.symbol, label)

        # A completed prefix is entirely offline: never redownload or open a retired ZIP.
        if paths["deletion"].exists():
            partition = read_json(partition_path(report, relative))
            derivation, _intent, _deletion = archive_evidence(
                report,
                contract.symbol,
                label,
                archive,
                archive_receipt_path,
                url,
                target,
                partition,
                prior_derivation,
            )
            carry = partition["output_carry_close"]
            prior_derivation = sha256(canonical_bytes(derivation))
            continue
        if paths["derivation"].exists() or paths["retirement-intent"].exists():
            # A durable derivation is resumed without redownloading; an
            # intent-authorized missing body is recoverable before deletion evidence.
            if archive.is_symlink() or (
                not archive.exists() and not paths["retirement-intent"].exists()
            ):
                raise AcquisitionError("archive_retirement_absence_invalid")
            archive_receipt = (
                request_receipt(archive, url)
                if archive.exists()
                else detached_archive_receipt(archive_receipt_path, url)
            )
            checksum_receipt = request_receipt(checksum, url + ".CHECKSUM")
            expected = parse_checksum(stable_file_bytes(checksum), filename)
            if archive_receipt["sha256"] != expected:
                raise AcquisitionError("archive_checksum_mismatch")
            partition = read_json(partition_path(report, relative))
            actual = parquet_frame(target)
            assert_raw_frame(actual, start, end)
            output_carry = float(actual["close"][-1])
            required_partition = expected_partition_receipt(
                relative,
                expected,
                target,
                actual.height,
                start,
                end,
                carry,
                output_carry,
                [checksum_receipt["sha256"], archive_receipt["sha256"]],
            )
            if partition != required_partition:
                raise AcquisitionError("partition_resume_receipt_invalid")
            expected_derivation = expected_archive_derivation(
                archive,
                archive_receipt_path,
                url,
                target,
                partition,
                prior_derivation,
                archive_receipt,
            )
            if (
                not paths["derivation"].exists()
                or read_json(paths["derivation"]) != expected_derivation
            ):
                raise AcquisitionError("archive_derivation_receipt_invalid")
            derivation = expected_derivation
            intent = expected_archive_intent(archive, derivation)
            if paths["retirement-intent"].exists():
                if read_json(paths["retirement-intent"]) != intent:
                    raise AcquisitionError("archive_retirement_intent_invalid")
            else:
                immutable_json(paths["retirement-intent"], intent, report)
            if archive.exists():
                remove_aggtrades_file(archive)
            if archive.exists() or archive.is_symlink():
                raise AcquisitionError("archive_retirement_absence_invalid")
            deletion = {
                "schema": "alpha_max_archive_deletion_receipt.v1",
                "retirement_intent_sha256": sha256(canonical_bytes(intent)),
                "derivation_receipt_sha256": sha256(canonical_bytes(derivation)),
                "archive_relative_path": str(archive.relative_to(report)),
                "archive_sha256": archive_receipt["sha256"],
                "archive_byte_count": archive_receipt["byte_count"],
                "archive_absent": True,
            }
            immutable_json(paths["deletion"], deletion, report)
            carry = output_carry
            prior_derivation = sha256(canonical_bytes(derivation))
            continue
        if not target.exists() and partition_path(report, relative).exists():
            raise AcquisitionError("partition_resume_receipt_invalid")
        live = list((report / "provenance" / "archives").glob("**/*.zip"))
        if archive not in live and len(live) >= MAX_LIVE_ARCHIVES:
            raise AcquisitionError("max_live_archives_exceeded")

        checksum_receipt = fetch_receipt(url + ".CHECKSUM", checksum, scratch_root=report)
        expected = parse_checksum(stable_file_bytes(checksum), filename)
        archive_receipt = fetch_receipt(url, archive, scratch_root=report)
        if archive_receipt["sha256"] != expected:
            raise AcquisitionError("archive_checksum_mismatch")
        frame, _facts = frame_from_archive(
            archive, contract.symbol, start, end, carry, label, report, archive_receipt
        )
        output_carry = float(frame["close"][-1])
        if target.exists():
            receipt = expected_partition_receipt(
                relative,
                expected,
                target,
                frame.height,
                start,
                end,
                carry,
                output_carry,
                [checksum_receipt["sha256"], archive_receipt["sha256"]],
            )
            if not verified_partition(report, relative, target, receipt, frame):
                raise AcquisitionError("preseeded_output_rejected")
        else:
            publish_frame(target, frame, output)
            receipt = partition_receipt(
                report,
                relative,
                expected,
                target,
                frame.height,
                start,
                end,
                carry,
                output_carry,
                [checksum_receipt["sha256"], archive_receipt["sha256"]],
            )
        # Independently reopen the still-authenticated body before it can be retired.
        second, _facts = frame_from_archive(
            archive, contract.symbol, start, end, carry, label, report, archive_receipt
        )
        if not second.equals(frame) or not parquet_frame(target).equals(frame):
            raise AcquisitionError("archive_double_derivation_mismatch")
        derivation = expected_archive_derivation(
            archive, archive_receipt_path, url, target, receipt, prior_derivation, archive_receipt
        )
        immutable_json(paths["derivation"], derivation, report)
        intent = expected_archive_intent(archive, derivation)
        immutable_json(paths["retirement-intent"], intent, report)
        remove_aggtrades_file(archive)
        if archive.exists() or archive.is_symlink():
            raise AcquisitionError("archive_retirement_absence_invalid")
        deletion = {
            "schema": "alpha_max_archive_deletion_receipt.v1",
            "retirement_intent_sha256": sha256(canonical_bytes(intent)),
            "derivation_receipt_sha256": sha256(canonical_bytes(derivation)),
            "archive_relative_path": str(archive.relative_to(report)),
            "archive_sha256": archive_receipt["sha256"],
            "archive_byte_count": archive_receipt["byte_count"],
            "archive_absent": True,
        }
        immutable_json(paths["deletion"], deletion, report)
        carry = output_carry
        prior_derivation = sha256(canonical_bytes(derivation))
        journal(
            report,
            {"event": "raw_partition", "path": relative, "output_sha256": file_sha256(target)},
        )


def funding_pages(contract: Contract, report: Path) -> tuple[list[dict[str, Any]], list[str]]:
    interval = funding_interval(contract.symbol)
    cursor = (
        max(0, contract.feature_start_ms - 2 * interval)
        if contract.symbol == "TONUSDT"
        else contract.feature_start_ms
    )
    page_no = 0
    rows = []
    hashes = []
    while cursor < contract.feature_end_ms:
        query = {
            "symbol": contract.symbol,
            "startTime": cursor,
            "endTime": contract.feature_end_ms - 1,
            "limit": 1000,
        }
        url = f"{API_BASE}/fundingRate?{urllib.parse.urlencode(query)}"
        page_no += 1
        destination = (
            report / "provenance" / "funding_pages" / contract.symbol / f"{page_no:06d}.json"
        )
        receipt = fetch_receipt(url, destination, query, report)
        try:
            page = read_json_value(destination)
        except AcquisitionError as exc:
            raise AcquisitionError("funding_api_json_invalid") from exc
        if not isinstance(page, list) or any(
            not isinstance(x, dict)
            or x.get("symbol") != contract.symbol
            or type(x.get("fundingTime")) is not int
            for x in page
        ):
            raise AcquisitionError("funding_api_schema_invalid")
        hashes.append(receipt["sha256"])
        if not page:
            break
        times = [x["fundingTime"] for x in page]
        if any(b <= a for a, b in itertools.pairwise(times)) or times[0] < cursor:
            raise AcquisitionError("funding_api_page_order_invalid")
        rows.extend(page)
        cursor = times[-1] + 1
        if len(page) < 1000:
            break
    return rows, hashes


def acquire_funding(contract: Contract, output: Path, report: Path) -> None:
    rows, pages = funding_pages(contract, report)
    normalized = normalize_funding(contract, rows)
    for day in day_starts(contract.feature_start_ms, contract.feature_end_ms):
        start = int(day.timestamp() * 1000)
        end = start + 86_400_000
        owned = [
            x
            for x in normalized
            if start
            <= x["timestamp_ms"]
            // funding_interval(contract.symbol)
            * funding_interval(contract.symbol)
            < end
        ]
        if not owned:
            continue
        relative = str(
            Path("feature_points")
            / "exchange=binance"
            / f"symbol={contract.symbol}"
            / f"date={day:%Y-%m-%d}"
            / "funding.parquet"
        )
        target = output / relative
        source_hash = sha256(canonical_bytes(owned))
        frame = (
            pl.DataFrame(owned)
            .select(FUNDING_COLUMNS)
            .with_columns(
                pl.col("timestamp_ms").cast(pl.Int64),
                pl.col("source_timestamp_ms").cast(pl.Int64),
                pl.col("funding_rate").cast(pl.Float64),
            )
        )
        if not target.exists() and partition_path(report, relative).exists():
            raise AcquisitionError("partition_resume_receipt_invalid")
        if target.exists():
            receipt = expected_partition_receipt(
                relative, source_hash, target, frame.height, start, end, None, None, pages
            )
            verified = verified_partition(report, relative, target, receipt, frame)
            if verified:
                continue
            if file_sha256(target) != frame_sha256(
                frame, target.parent, output
            ) or not parquet_frame(target).equals(frame):
                raise AcquisitionError("preseeded_output_rejected")
        else:
            publish_frame(target, frame, output)
        partition_receipt(
            report, relative, source_hash, target, frame.height, start, end, None, None, pages
        )
        journal(
            report,
            {"event": "funding_partition", "path": relative, "output_sha256": file_sha256(target)},
        )


def bind_input_provenance(contract_path: Path, evidence_path: Path, report: Path) -> None:
    approved_inputs = (
        (contract_path, report / "provenance" / "contract_manifest.json", CONTRACT_SHA256),
        (evidence_path, report / "provenance" / "availability_evidence.json", EVIDENCE_SHA256),
    )
    bound: list[tuple[Path, bytes]] = []
    for source, destination, approved in approved_inputs:
        data = safe_file_bytes(source)
        if sha256(data) != approved:
            raise AcquisitionError("approved_input_changed")
        bound.append((destination, data))
    for destination, data in bound:
        if destination.exists():
            if safe_file_bytes(destination) != data:
                raise AcquisitionError("approved_input_provenance_changed")
        else:
            atomic_write(destination, data, scratch_root=report)


def partition_path(report: Path, relative: str) -> Path:
    return report / "partitions" / (sha256(relative.encode()) + ".json")


def assert_raw_frame(frame: pl.DataFrame, start: int, end: int) -> None:
    if frame.columns != RAW_COLUMNS or frame.dtypes != [pl.Datetime("ms"), *([pl.Float64] * 5)]:
        raise AcquisitionError("complete_raw_schema_invalid")
    if (
        frame.height != (end - start) // 1000
        or frame.null_count().select(pl.all().sum()).row(0) != (0,) * 6
    ):
        raise AcquisitionError("complete_raw_content_invalid")
    rows = frame.iter_rows()
    for index, (stamp, opening, high, low, close, volume) in enumerate(rows):
        if stamp != datetime.fromtimestamp((start + index * 1000) / 1000, UTC).replace(tzinfo=None):
            raise AcquisitionError("complete_raw_continuity_invalid")
        if (
            not all(math.isfinite(float(x)) for x in (opening, high, low, close, volume))
            or opening <= 0
            or low <= 0
            or high < max(opening, close)
            or low > min(opening, close)
            or volume < 0
        ):
            raise AcquisitionError("complete_raw_ohlcv_invalid")


def funding_pages_from_provenance(
    contract: Contract, report: Path
) -> tuple[list[dict[str, Any]], list[str], set[Path]]:
    cursor = (
        max(0, contract.feature_start_ms - 2 * funding_interval(contract.symbol))
        if contract.symbol == "TONUSDT"
        else contract.feature_start_ms
    )
    number = 0
    rows = []
    hashes = []
    paths = set()
    while cursor < contract.feature_end_ms:
        number += 1
        query = {
            "symbol": contract.symbol,
            "startTime": cursor,
            "endTime": contract.feature_end_ms - 1,
            "limit": 1000,
        }
        url = f"{API_BASE}/fundingRate?{urllib.parse.urlencode(query)}"
        path = report / "provenance" / "funding_pages" / contract.symbol / f"{number:06d}.json"
        receipt = request_receipt(path, url, query)
        paths.update((path, path.with_suffix(".json.receipt.json")))
        try:
            page = read_json_value(path)
        except AcquisitionError as exc:
            raise AcquisitionError("funding_api_json_invalid") from exc
        if not isinstance(page, list) or any(
            not isinstance(x, dict)
            or x.get("symbol") != contract.symbol
            or type(x.get("fundingTime")) is not int
            for x in page
        ):
            raise AcquisitionError("funding_api_schema_invalid")
        times = [x["fundingTime"] for x in page]
        if any(b <= a for a, b in itertools.pairwise(times)) or (times and times[0] < cursor):
            raise AcquisitionError("funding_api_page_order_invalid")
        rows.extend(page)
        hashes.append(receipt["sha256"])
        if not page or len(page) < 1000:
            break
        cursor = times[-1] + 1
    return rows, hashes, paths


def stable_tree(
    root: Path, file_digests: dict[str, str] | None = None
) -> tuple[set[str], set[str]]:
    """Walk a root through directory descriptors without following components."""
    root = lexical(root)
    files: set[str] = set()
    directories: set[str] = set()

    def walk(fd: int, prefix: Path) -> None:
        with os.scandir(fd) as entries:
            for entry in entries:
                relative = prefix / entry.name
                item = entry.stat(follow_symlinks=False)
                name = str(relative)
                if stat.S_ISLNK(item.st_mode):
                    raise AcquisitionError("complete_inventory_unsafe_object")
                if stat.S_ISDIR(item.st_mode):
                    child_fd = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=fd,
                    )
                    try:
                        child = os.fstat(child_fd)
                        if (child.st_dev, child.st_ino) != (item.st_dev, item.st_ino):
                            raise AcquisitionError("complete_inventory_unsafe_object")
                        directories.add(name)
                        walk(child_fd, relative)
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(item.st_mode) and item.st_nlink == 1:
                    if file_digests is not None:
                        try:
                            file_fd = os.open(
                                entry.name,
                                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                                dir_fd=fd,
                            )
                        except OSError as exc:
                            raise AcquisitionError("complete_inventory_unsafe_object") from exc
                        try:
                            opened = os.fstat(file_fd)
                            if (
                                not stat.S_ISREG(opened.st_mode)
                                or opened.st_nlink != 1
                                or (opened.st_dev, opened.st_ino) != (item.st_dev, item.st_ino)
                            ):
                                raise AcquisitionError("complete_inventory_unsafe_object")
                            with os.fdopen(os.dup(file_fd), "rb") as source:
                                before = os.fstat(file_fd)
                                file_digests[name] = hashlib.file_digest(
                                    source, "sha256"
                                ).hexdigest()
                                after = os.fstat(file_fd)
                                if (
                                    before.st_dev,
                                    before.st_ino,
                                    before.st_mode,
                                    before.st_size,
                                    before.st_mtime_ns,
                                    before.st_ctime_ns,
                                    before.st_nlink,
                                ) != (
                                    after.st_dev,
                                    after.st_ino,
                                    after.st_mode,
                                    after.st_size,
                                    after.st_mtime_ns,
                                    after.st_ctime_ns,
                                    after.st_nlink,
                                ):
                                    raise AcquisitionError("complete_inventory_unsafe_object")
                        finally:
                            os.close(file_fd)
                    files.add(name)
                else:
                    raise AcquisitionError("complete_inventory_unsafe_object")

    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise AcquisitionError("complete_inventory_unsafe_object") from exc
    try:
        walk(root_fd, Path())
    finally:
        os.close(root_fd)
    return files, directories


def manifest_value(output: Path, report: Path) -> dict[str, Any]:
    files = []
    for root, prefix, excluded in (
        (output, "output", {".alpha_max_owner.json"}),
        (
            report,
            "report",
            {
                "plan.json",
                ".alpha_max_owner.json",
                "source_manifest.json",
                "source_eligible_receipt.json",
            },
        ),
    ):
        tree_digests: dict[str, str] = {}
        tree_files, _directories = stable_tree(root, tree_digests)
        for relative in sorted(tree_files):
            if Path(relative).name not in excluded:
                files.append({"path": f"{prefix}/{relative}", "sha256": tree_digests[relative]})
    evidence_files = [
        artifact
        for artifact in files
        if artifact["path"].startswith("report/provenance/archive-evidence/")
    ]
    return {
        "schema": "alpha_max_official_source_manifest.v5",
        "contract_sha256": CONTRACT_SHA256,
        "availability_evidence_sha256": EVIDENCE_SHA256,
        "derivation_version": DERIVATION_VERSION,
        "storage_contract": {
            "host_reserve_path": str(HOST_RESERVE_PATH),
            "host_reserve_bytes": HOST_RESERVE_BYTES,
            "max_live_archives": MAX_LIVE_ARCHIVES,
            "archive_retention": ARCHIVE_RETENTION,
        },
        "archive_evidence_sha256": sha256(canonical_bytes(evidence_files)),
        "artifacts": files,
    }


def rebuild_manifest(output: Path, report: Path) -> str:
    data = canonical_bytes(manifest_value(output, report))
    atomic_write(report / "source_manifest.json", data, replace=True, scratch_root=report)
    return sha256(data)


def checked_tree(root: Path, expected_files: set[str]) -> None:
    """Reject every object not in the authenticated inventory."""
    actual_files, actual_dirs = stable_tree(root)
    expected_dirs: set[str] = set()
    for name in expected_files:
        parent = Path(name).parent
        while str(parent) != ".":
            expected_dirs.add(str(parent))
            parent = parent.parent
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise AcquisitionError("complete_inventory_invalid")


def full_plan() -> dict[str, Any]:
    return {
        "schema": "alpha_max_official_acquisition_plan.v4",
        "source_eligible": False,
        "symbols": list(SYMBOLS),
        "months": [],
        "contract_sha256": CONTRACT_SHA256,
        "availability_evidence_sha256": EVIDENCE_SHA256,
        # ponytail: /mnt/c is the deliberate WSL ceiling; upgrade only through a signed CLI/platform capacity probe.
        "storage_contract": {
            "host_reserve_path": str(HOST_RESERVE_PATH),
            "host_reserve_bytes": HOST_RESERVE_BYTES,
            "max_live_archives": MAX_LIVE_ARCHIVES,
            "archive_retention": ARCHIVE_RETENTION,
        },
    }


def verify_eligible(output: Path, report: Path, contracts: list[Contract]) -> None:
    plan = full_plan()
    plan_data = canonical_bytes(plan)
    run_id = sha256(plan_data)
    if (
        read_json(report / "plan.json") != plan
        or stable_file_bytes(report / "plan.json") != plan_data
    ):
        raise AcquisitionError("immutable_plan_invalid")
    verify_ownership(output, report, run_id)
    receipt = validate_complete(output, contracts, report / "provenance", report)
    receipt["source_manifest_sha256"] = file_sha256(report / "source_manifest.json")
    receipt["acquisition_journal_sha256"] = file_sha256(report / "acquisition.journal.jsonl")
    if read_json(report / "source_eligible_receipt.json") != receipt:
        raise AcquisitionError("source_eligible_receipt_invalid")


def validate_complete(
    output: Path, contracts: list[Contract], provenance: Path, report: Path | None = None
) -> dict[str, Any]:
    if report is None or provenance != report / "provenance":
        raise AcquisitionError("complete_report_required")
    contract_data = safe_file_bytes(provenance / "contract_manifest.json")
    evidence_data = safe_file_bytes(provenance / "availability_evidence.json")
    if sha256(contract_data) != CONTRACT_SHA256 or sha256(evidence_data) != EVIDENCE_SHA256:
        raise AcquisitionError("complete_input_provenance_invalid")
    exchange = provenance / "exchangeInfo.json"
    request_receipt(exchange, f"{API_BASE}/exchangeInfo")
    try:
        exchange_value = read_json_value(exchange)
    except AcquisitionError as exc:
        raise AcquisitionError("exchange_info_schema_invalid") from exc
    if not isinstance(exchange_value, dict) or not isinstance(exchange_value.get("symbols"), list):
        raise AcquisitionError("exchange_info_schema_invalid")
    required: set[str] = set()
    expected_report: set[Path] = {
        provenance / "contract_manifest.json",
        provenance / "availability_evidence.json",
        exchange,
        exchange.with_suffix(".json.receipt.json"),
        report / "acquisition.journal.jsonl",
    }
    expected_output: set[str] = set()
    for contract in contracts:
        for month in month_starts(contract.raw_start_ms, contract.raw_end_ms):
            expected_output.add(
                str(Path("market_ohlcv_1s") / EXCHANGE / contract.symbol / f"{month:%Y-%m}.parquet")
            )
        for day in day_starts(contract.feature_start_ms, contract.feature_end_ms):
            start = int(day.timestamp() * 1000)
            if any(
                start <= settlement < start + 86_400_000
                for settlement in expected_settlements(contract)
            ):
                expected_output.add(
                    str(
                        Path("feature_points")
                        / "exchange=binance"
                        / f"symbol={contract.symbol}"
                        / f"date={day:%Y-%m-%d}"
                        / "funding.parquet"
                    )
                )
    output_files, _output_dirs = stable_tree(output)
    if {path for path in output_files if path.endswith(".parquet")} != expected_output:
        raise AcquisitionError("complete_inventory_invalid")
    raw_total = funding_total = 0
    for contract in contracts:
        carry: float | None = None
        prior_derivation: str | None = None
        for month in month_starts(contract.raw_start_ms, contract.raw_end_ms):
            label = month.strftime("%Y-%m")
            nominal_start, nominal_end = month_bounds(month)
            start, end = (
                max(nominal_start, contract.raw_start_ms),
                min(nominal_end, contract.raw_end_ms),
            )
            relative = str(
                Path("market_ohlcv_1s") / EXCHANGE / contract.symbol / f"{label}.parquet"
            )
            required.add(relative)
            target = output / relative
            filename = f"{contract.symbol}-aggTrades-{label}.zip"
            archive = provenance / "archives" / contract.symbol / filename
            checksum = archive.with_name(filename + ".CHECKSUM")
            archive_url = f"{ARCHIVE_BASE}/{contract.symbol}/{filename}"
            checksum_receipt = request_receipt(checksum, archive_url + ".CHECKSUM")
            archive_receipt_path = archive.with_suffix(".zip.receipt.json")
            archive_receipt = detached_archive_receipt(archive_receipt_path, archive_url)
            expected_report.update(
                (
                    archive_receipt_path,
                    checksum,
                    checksum.with_suffix(".CHECKSUM.receipt.json"),
                    *archive_evidence_paths(report, contract.symbol, label).values(),
                )
            )
            source_hash = parse_checksum(stable_file_bytes(checksum), filename)
            if source_hash != archive_receipt["sha256"]:
                raise AcquisitionError("archive_checksum_mismatch")
            receipt = read_json(partition_path(report, relative))
            expected_report.add(partition_path(report, relative))
            if not target.exists():
                raise AcquisitionError("complete_inventory_missing")
            actual = parquet_frame(target)
            assert_raw_frame(actual, start, end)
            output_hash = file_sha256(target)
            output_carry = float(actual["close"][-1])
            expected_receipt = {
                "schema": "alpha_max_partition_receipt.v2",
                "path": relative,
                "source_sha256": source_hash,
                "output_sha256": output_hash,
                "rows": actual.height,
                "start_ms": start,
                "end_ms": end,
                "input_carry_close": carry,
                "output_carry_close": output_carry,
                "derivation_version": DERIVATION_VERSION,
                "code_sha256": code_hash(),
                "page_hashes": [checksum_receipt["sha256"], archive_receipt["sha256"]],
            }
            if receipt != expected_receipt:
                raise AcquisitionError("complete_raw_partition_receipt_invalid")
            derivation, _intent, _deletion = archive_evidence(
                report,
                contract.symbol,
                label,
                archive,
                archive_receipt_path,
                archive_url,
                target,
                receipt,
                prior_derivation,
            )
            carry = output_carry
            prior_derivation = sha256(canonical_bytes(derivation))
            raw_total += actual.height
        pages, page_hashes, page_paths = funding_pages_from_provenance(contract, report)
        expected_report.update(page_paths)
        normalized = normalize_funding(contract, pages)
        if contract.symbol == "TONUSDT":
            interval = funding_interval(contract.symbol)
            pre_owned = contract.feature_start_ms // interval * interval - 2 * interval
            unavailable = pre_owned + interval
            source_times = {x["fundingTime"] for x in pages}
            if (
                not any(t // interval * interval == pre_owned for t in source_times)
                or any(t // interval * interval == unavailable for t in source_times)
                or not normalized
                or normalized[0]["timestamp_ms"] // interval * interval
                != contract.feature_start_ms // interval * interval
            ):
                raise AcquisitionError("funding_ton_availability_proof_invalid")
        for day in day_starts(contract.feature_start_ms, contract.feature_end_ms):
            start = int(day.timestamp() * 1000)
            end = start + 86_400_000
            interval = funding_interval(contract.symbol)
            owned = [
                x for x in normalized if start <= x["timestamp_ms"] // interval * interval < end
            ]
            if not owned:
                continue
            relative = str(
                Path("feature_points")
                / "exchange=binance"
                / f"symbol={contract.symbol}"
                / f"date={day:%Y-%m-%d}"
                / "funding.parquet"
            )
            required.add(relative)
            target = output / relative
            if not target.exists():
                raise AcquisitionError("complete_inventory_missing")
            frame = parquet_frame(target)
            if (
                frame.columns != FUNDING_COLUMNS
                or frame.dtypes != [pl.Int64, pl.Int64, pl.String, pl.String, pl.Float64]
                or frame.null_count().select(pl.all().sum()).row(0) != (0,) * 5
            ):
                raise AcquisitionError("complete_funding_content_invalid")
            actual = frame.to_dicts()
            if actual != owned or any(not math.isfinite(x["funding_rate"]) for x in actual):
                raise AcquisitionError("complete_funding_rederivation_mismatch")
            source_hash = sha256(canonical_bytes(owned))
            receipt_path = partition_path(report, relative)
            expected_report.add(receipt_path)
            expected_receipt = {
                "schema": "alpha_max_partition_receipt.v2",
                "path": relative,
                "source_sha256": source_hash,
                "output_sha256": file_sha256(target),
                "rows": frame.height,
                "start_ms": start,
                "end_ms": end,
                "input_carry_close": None,
                "output_carry_close": None,
                "derivation_version": DERIVATION_VERSION,
                "code_sha256": code_hash(),
                "page_hashes": page_hashes,
            }
            if read_json(receipt_path) != expected_receipt:
                raise AcquisitionError("complete_funding_partition_receipt_invalid")
            funding_total += frame.height
    expected_output_files = required | {".alpha_max_owner.json"}
    expected_report_files = {str(path.relative_to(report)) for path in expected_report} | {
        "plan.json",
        ".alpha_max_owner.json",
        "source_manifest.json",
    }
    if (report / "source_eligible_receipt.json").exists():
        expected_report_files.add("source_eligible_receipt.json")
    checked_tree(output, expected_output_files)
    checked_tree(report, expected_report_files)
    manifest_path = report / "source_manifest.json"
    if manifest_path.exists() and stable_file_bytes(manifest_path) != canonical_bytes(
        manifest_value(output, report)
    ):
        raise AcquisitionError("source_manifest_stale")
    if raw_total != 1_066_681_730 or funding_total != 39_569:
        raise AcquisitionError("complete_row_totals_invalid")
    inventory_sha = sha256(canonical_bytes(sorted(required)))
    archive_evidence_digest = sha256(
        canonical_bytes(
            sorted(
                (
                    {
                        "path": f"report/{path.relative_to(report)}",
                        "sha256": file_sha256(path),
                    }
                    for contract in contracts
                    for label in (
                        month.strftime("%Y-%m")
                        for month in month_starts(contract.raw_start_ms, contract.raw_end_ms)
                    )
                    for path in archive_evidence_paths(report, contract.symbol, label).values()
                ),
                key=lambda artifact: artifact["path"],
            )
        )
    )
    return {
        "schema": "alpha_max_official_source_receipt.v4",
        "source_eligible": True,
        "raw_rows": raw_total,
        "funding_rows": funding_total,
        "contract_sha256": CONTRACT_SHA256,
        "availability_evidence_sha256": EVIDENCE_SHA256,
        "derivation_version": DERIVATION_VERSION,
        "code_sha256": code_hash(),
        "storage_contract": {
            "host_reserve_path": str(HOST_RESERVE_PATH),
            "host_reserve_bytes": HOST_RESERVE_BYTES,
            "max_live_archives": MAX_LIVE_ARCHIVES,
            "archive_retention": ARCHIVE_RETENTION,
        },
        "archive_evidence_sha256": archive_evidence_digest,
        "exchange_info_sha256": file_sha256(exchange),
        "inventory_sha256": inventory_sha,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-manifest", required=True, type=Path)
    parser.add_argument("--availability-evidence", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--forbidden-root", action="append", default=[], type=Path)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--months", nargs="+")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument(
        "--verify-eligible",
        action="store_true",
        help="read-only offline verification of the immutable full eligible receipt",
    )
    parser.add_argument("--validate-complete", action="store_true")
    parser.add_argument("--verifier-code-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify_eligible and (args.symbols or args.months or args.validate_complete):
        parser.error("--verify-eligible cannot be combined with selectors or --validate-complete")
    if args.validate_complete and not args.execute:
        parser.error("--validate-complete requires --execute")
    if args.verifier_code_fd is not None and not args.verify_eligible:
        parser.error("--verifier-code-fd requires --verify-eligible")
    return args


def invocation_verifier_code_fd(function: Any) -> Any:
    def wrapped(argv: list[str] | None = None) -> int:
        args = parse_args(argv)
        code_fd_token = _VERIFIER_CODE_FD.set(args.verifier_code_fd)
        try:
            return function(args)
        finally:
            _VERIFIER_CODE_FD.reset(code_fd_token)

    return wrapped


@release_execute_lock
@invocation_verifier_code_fd
def main(args: argparse.Namespace) -> int:
    assert_input_paths(args.contract_manifest, args.availability_evidence, args.forbidden_root)
    assert_roots(
        args.output_root,
        args.report_dir,
        args.forbidden_root,
        args.execute or args.verify_eligible,
    )
    contracts = load_contract(args.contract_manifest)
    load_evidence(args.availability_evidence)
    if args.verify_eligible:
        with owned_run_lock(args.report_dir, exclusive=True):
            verify_eligible(args.output_root, args.report_dir, contracts)
        return 0
    selected_symbols = set(args.symbols or SYMBOLS)
    selected_months = set(args.months or [])
    if not selected_symbols <= set(SYMBOLS):
        raise AcquisitionError("symbol_selector_invalid")
    valid_months = {
        month.strftime("%Y-%m")
        for c in contracts
        for month in month_starts(c.raw_start_ms, c.raw_end_ms)
    }
    if not selected_months <= valid_months or any(
        datetime.strptime(x, "%Y-%m").strftime("%Y-%m") != x for x in selected_months
    ):
        raise AcquisitionError("month_selector_invalid")
    plan = {
        **full_plan(),
        "symbols": sorted(selected_symbols),
        "months": sorted(selected_months),
    }
    plan_data = canonical_bytes(plan)
    run_id = sha256(plan_data)
    if not args.execute:
        report_exists = args.report_dir.exists() or args.report_dir.is_symlink()
        if report_exists:
            safe_root(args.report_dir, create=False)
        if report_exists and any(args.report_dir.iterdir()):
            raise AcquisitionError("plan_report_preseeded")
        safe_root(args.report_dir, create=True)
        immutable_json(args.report_dir / "plan.json", plan, args.report_dir)
        return 0
    output, report = args.output_root, args.report_dir
    output_exists = output.exists() or output.is_symlink()
    report_exists = report.exists() or report.is_symlink()
    if not output_exists and not report_exists:
        safe_root(report, create=True)
        immutable_json(report / "plan.json", plan, report)
    elif not output_exists and report_exists and not {x.name for x in report.iterdir()}:
        immutable_json(report / "plan.json", plan, report)
    recover_owned_hardlink_prefixes(output, report, plan_data)
    if recover_first_execute_prefix(output, report, plan_data, run_id):
        pass
    elif (
        output.exists()
        and report.exists()
        and read_json(report / "plan.json") == plan
        and (report / "plan.json").read_bytes() == plan_data
    ):
        verify_ownership(output, report, run_id)
        recover_owned_hardlink_prefixes(output, report, plan_data)
    else:
        raise AcquisitionError("roots_resume_pair_invalid")
    _ACTIVE_EXECUTE_LOCK.set(OwnedRunLock(report, exclusive=True))
    cleanup_scratch(output, report)
    if (report / "source_eligible_receipt.json").exists():
        if plan == full_plan():
            verify_eligible(output, report, contracts)
            return 0
        raise AcquisitionError("source_eligible_run_is_immutable")
    bind_input_provenance(args.contract_manifest, args.availability_evidence, report)
    journal(report, {"event": "started", "plan_sha256": run_id, "source_eligible": False})
    exchange = report / "provenance" / "exchangeInfo.json"
    fetch_receipt(f"{API_BASE}/exchangeInfo", exchange, scratch_root=report)
    try:
        exchange_value = read_json_value(exchange)
        symbols_now = {
            x.get("symbol") for x in exchange_value.get("symbols", []) if isinstance(x, dict)
        }
    except (AcquisitionError, AttributeError) as exc:
        raise AcquisitionError("exchange_info_schema_invalid") from exc
    selected = [x for x in contracts if x.symbol in selected_symbols]
    if any(x.symbol not in symbols_now and x.symbol != "TONUSDT" for x in selected):
        raise AcquisitionError("exchange_info_contract_missing")
    for contract in selected:
        if selected_months:
            chosen = {
                x
                for x in selected_months
                if x
                in {
                    m.strftime("%Y-%m")
                    for m in month_starts(contract.raw_start_ms, contract.raw_end_ms)
                }
            }
            for label in chosen:
                m = datetime.strptime(label, "%Y-%m").replace(tzinfo=UTC)
                start, end = month_bounds(m)
                if start < contract.raw_start_ms or end > contract.raw_end_ms:
                    raise AcquisitionError("pilot_month_must_be_whole_owned_month")
            acquire_archive(contract, output, report, chosen)
        else:
            acquire_archive(contract, output, report)
            acquire_funding(contract, output, report)
    journal(report, {"event": "acquisition_finished", "source_eligible": False})
    cleanup_scratch(output, report)
    rebuild_manifest(output, report)
    if args.validate_complete:
        if args.symbols or args.months:
            raise AcquisitionError("selectors_cannot_produce_source_eligible")
        journal(report, {"event": "validation_started", "source_eligible": False})
        rebuild_manifest(output, report)
        receipt = validate_complete(output, contracts, report / "provenance", report)
        receipt["source_manifest_sha256"] = file_sha256(report / "source_manifest.json")
        receipt["acquisition_journal_sha256"] = file_sha256(report / "acquisition.journal.jsonl")
        verify_ownership(output, report, run_id)
        immutable_json(report / "source_eligible_receipt.json", receipt, report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcquisitionError as exc:
        print(f"acquire_alpha_max_official_source: {exc}", file=sys.stderr)
        raise SystemExit(2)
