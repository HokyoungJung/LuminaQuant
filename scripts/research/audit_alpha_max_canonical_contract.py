#!/usr/bin/env python3
"""Audit canonical Alpha-Max raw and funding coverage against the approved contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

CONTRACT_SHA256 = "ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220"
RAW_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")
RAW_SCHEMA = {
    "datetime": pl.Datetime("ms"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
FUNDING_JITTER_MS = 1_000


def utc_ms(value: str) -> int:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return int(parsed.astimezone(UTC).timestamp() * 1000)


def month_starts(start_ms: int, end_ms: int):
    current = datetime.fromtimestamp(start_ms / 1000, UTC).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    while int(current.timestamp() * 1000) < end_ms:
        yield current
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def month_bounds(value: datetime) -> tuple[int, int]:
    start = int(value.timestamp() * 1000)
    end = int(((value.replace(day=28) + timedelta(days=4)).replace(day=1)).timestamp() * 1000)
    return start, end


def funding_interval_ms(symbol: str) -> int:
    return 14_400_000 if symbol == "TONUSDT" else 28_800_000


def expected_funding_times(symbol: str, start_ms: int, end_ms: int) -> list[int]:
    interval = funding_interval_ms(symbol)
    first = ((start_ms + interval - 1) // interval) * interval
    return list(range(first, end_ms, interval))


def load_contract(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTRACT_SHA256:
        raise ValueError("contract digest mismatch")
    value = json.loads(raw)
    if value.get("schema_version") != "alpha_max_contract_manifest.v2":
        raise ValueError("contract schema mismatch")
    return value


def audit_raw_partition(
    path: Path,
    *,
    symbol: str,
    month: str,
    start_ms: int,
    end_ms: int,
    deep: bool,
) -> dict[str, Any]:
    expected_rows = (end_ms - start_ms) // 1000
    partition_month = datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)
    partition_start_ms, partition_end_ms = month_bounds(partition_month)
    item: dict[str, Any] = {
        "symbol": symbol,
        "month": month,
        "path": str(path),
        "start_ms": start_ms,
        "end_ms_exclusive": end_ms,
        "expected_rows": expected_rows,
        "exists": path.is_file(),
        "total_rows": 0,
        "actual_rows": 0,
        "outside_contract_rows": 0,
        "mispartitioned_rows": 0,
        "first_ms": None,
        "last_ms": None,
        "schema_ok": False,
        "null_count": None,
        "deep_failures": None,
        "status": "missing",
    }
    if not path.is_file():
        return item

    schema = pl.read_parquet_schema(path)
    item["schema_ok"] = schema == RAW_SCHEMA
    try:
        timestamp_ms = pl.col("datetime").dt.epoch("ms")
        scan = pl.scan_parquet(path)
        partition_stats = (
            scan.select(
                pl.len().alias("total_rows"),
                ((timestamp_ms < partition_start_ms) | (timestamp_ms >= partition_end_ms))
                .sum()
                .alias("mispartitioned_rows"),
            )
            .collect()
            .row(0, named=True)
        )
        contract_scan = scan.filter((timestamp_ms >= start_ms) & (timestamp_ms < end_ms))
        expressions: list[pl.Expr] = [
            pl.len().alias("rows"),
            timestamp_ms.min().alias("first_ms"),
            timestamp_ms.max().alias("last_ms"),
            pl.sum_horizontal([pl.col(name).null_count() for name in RAW_COLUMNS]).alias(
                "null_count"
            ),
        ]
        if deep:
            expected_timestamp_ms = pl.int_range(0, pl.len(), dtype=pl.Int64) * 1000 + start_ms
            expressions.extend(
                [
                    (timestamp_ms != expected_timestamp_ms)
                    .sum()
                    .alias("timestamp_sequence_failures"),
                    pl.sum_horizontal(
                        [
                            (~pl.col(name).is_finite()).sum()
                            for name in ("open", "high", "low", "close", "volume")
                        ]
                    ).alias("non_finite_values"),
                    pl.sum_horizontal(
                        [(pl.col(name) <= 0).sum() for name in ("open", "high", "low", "close")]
                    ).alias("non_positive_prices"),
                    (pl.col("volume") < 0).sum().alias("negative_volume"),
                    (
                        (pl.col("high") < pl.max_horizontal("open", "close"))
                        | (pl.col("low") > pl.min_horizontal("open", "close"))
                        | (pl.col("high") < pl.col("low"))
                    )
                    .sum()
                    .alias("ohlc_failures"),
                ]
            )
        stats = contract_scan.select(expressions).collect().row(0, named=True)
    except Exception as exc:
        item["status"] = "unreadable"
        item["error"] = repr(exc)
        return item

    item["total_rows"] = int(partition_stats["total_rows"])
    item["mispartitioned_rows"] = int(partition_stats["mispartitioned_rows"])
    item["actual_rows"] = int(stats["rows"])
    item["outside_contract_rows"] = item["total_rows"] - item["actual_rows"]
    item["first_ms"] = int(stats["first_ms"]) if stats["first_ms"] is not None else None
    item["last_ms"] = int(stats["last_ms"]) if stats["last_ms"] is not None else None
    item["null_count"] = int(stats["null_count"])
    if deep:
        item["deep_failures"] = {
            name: int(stats[name])
            for name in (
                "timestamp_sequence_failures",
                "non_finite_values",
                "non_positive_prices",
                "negative_volume",
                "ohlc_failures",
            )
        }
    complete = (
        item["schema_ok"]
        and item["actual_rows"] == expected_rows
        and item["first_ms"] == start_ms
        and item["last_ms"] == end_ms - 1000
        and item["null_count"] == 0
        and item["mispartitioned_rows"] == 0
        and (not deep or not any(item["deep_failures"].values()))
    )
    item["status"] = ("complete" if deep else "inventory-complete") if complete else "incomplete"
    return item


def audit_funding(
    db_path: Path,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    expected = set(expected_funding_times(symbol, start_ms, end_ms))
    paths = sorted(
        (db_path / "feature_points" / "exchange=binance" / f"symbol={symbol}").glob(
            "date=*/*.parquet"
        )
    )
    occurrences: list[int] = []
    errors: list[str] = []
    funding_files = 0
    files_with_values = 0
    for path in paths:
        try:
            schema = pl.read_parquet_schema(path)
            if "funding_rate" not in schema:
                continue
            funding_files += 1
            if (
                "timestamp_ms" not in schema
                or schema["timestamp_ms"] != pl.Int64
                or schema["funding_rate"] != pl.Float64
            ):
                errors.append(f"{path}: funding schema mismatch")
                continue
            values = (
                pl.scan_parquet(path)
                .filter(
                    pl.col("funding_rate").is_not_null()
                    & (pl.col("timestamp_ms") >= start_ms)
                    & (pl.col("timestamp_ms") < end_ms)
                )
                .select(pl.col("timestamp_ms"))
                .collect()
                .get_column("timestamp_ms")
                .to_list()
            )
        except Exception as exc:
            errors.append(f"{path}: {exc!r}")
            continue
        if values:
            files_with_values += 1
            occurrences.extend(int(value) for value in values)

    interval_ms = funding_interval_ms(symbol)
    settlements = [value // interval_ms * interval_ms for value in occurrences]
    jitters = [
        value - settlement for value, settlement in zip(occurrences, settlements, strict=True)
    ]
    jitter_violations = sum(jitter < 0 or jitter > FUNDING_JITTER_MS for jitter in jitters)
    counts = Counter(settlements)
    actual = set(counts)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    duplicate_rows = sum(count - 1 for count in counts.values())
    complete = (
        not missing and not extra and duplicate_rows == 0 and jitter_violations == 0 and not errors
    )
    return {
        "symbol": symbol,
        "expected_rows": len(expected),
        "actual_rows_in_window": len(occurrences),
        "actual_expected_rows": len(expected & actual),
        "missing_rows": len(missing),
        "extra_rows_in_window": len(extra),
        "duplicate_rows_in_window": duplicate_rows,
        "jitter_violation_rows": jitter_violations,
        "max_observed_jitter_ms": max(jitters, default=None),
        "files_scanned": len(paths),
        "funding_files_scanned": funding_files,
        "files_with_values_in_window": files_with_values,
        "errors": errors,
        "first_missing_ms": missing[0] if missing else None,
        "last_missing_ms": missing[-1] if missing else None,
        "first_extra_ms": extra[0] if extra else None,
        "last_extra_ms": extra[-1] if extra else None,
        "status": "complete" if complete else "incomplete",
    }


def audit_contract(
    *,
    contract_path: Path,
    db_path: Path,
    deep: bool,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    raw_items: list[dict[str, Any]] = []
    funding_items: list[dict[str, Any]] = []
    for record in contract["records"]:
        symbol = str(record["symbol"])
        raw_start = utc_ms(record["raw_availability_start_utc"])
        raw_end = utc_ms(record["raw_availability_end_utc"])
        for month in month_starts(raw_start, raw_end):
            label = month.strftime("%Y-%m")
            nominal_start, nominal_end = month_bounds(month)
            start_ms = max(raw_start, nominal_start)
            end_ms = min(raw_end, nominal_end)
            raw_items.append(
                audit_raw_partition(
                    db_path / "market_ohlcv_1s" / "binance" / symbol / f"{label}.parquet",
                    symbol=symbol,
                    month=label,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    deep=deep,
                )
            )
        funding_items.append(
            audit_funding(
                db_path,
                symbol=symbol,
                start_ms=utc_ms(record["feature_availability_start_utc"]),
                end_ms=utc_ms(record["feature_availability_end_utc"]),
            )
        )

    accepted_raw_status = "complete" if deep else "inventory-complete"
    raw_complete = sum(item["status"] == accepted_raw_status for item in raw_items)
    funding_complete = all(item["status"] == "complete" for item in funding_items)
    return {
        "artifact_kind": "alpha_max_canonical_contract_inventory",
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": CONTRACT_SHA256,
        "db_path": str(db_path.resolve()),
        "audit_mode": "deep" if deep else "inventory",
        "raw": {
            "target_partitions": len(raw_items),
            "complete_partitions": raw_complete,
            "missing_or_incomplete_partitions": len(raw_items) - raw_complete,
            "target_rows": sum(item["expected_rows"] for item in raw_items),
            "complete_rows": sum(
                item["actual_rows"] for item in raw_items if item["status"] == accepted_raw_status
            ),
            "items": raw_items,
        },
        "funding": {
            "target_rows": sum(item["expected_rows"] for item in funding_items),
            "complete_rows": sum(item["actual_expected_rows"] for item in funding_items),
            "missing_rows": sum(item["missing_rows"] for item in funding_items),
            "extra_rows_in_window": sum(item["extra_rows_in_window"] for item in funding_items),
            "duplicate_rows_in_window": sum(
                item["duplicate_rows_in_window"] for item in funding_items
            ),
            "jitter_violation_rows": sum(item["jitter_violation_rows"] for item in funding_items),
            "error_count": sum(len(item["errors"]) for item in funding_items),
            "symbols": funding_items,
        },
        "status": (
            ("complete" if deep else "inventory-complete")
            if raw_complete == len(raw_items) and funding_complete
            else "incomplete"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deep", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = audit_contract(
        contract_path=args.contract,
        db_path=args.db,
        deep=bool(args.deep),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "raw_target_partitions": payload["raw"]["target_partitions"],
                "raw_complete_partitions": payload["raw"]["complete_partitions"],
                "raw_missing_or_incomplete": payload["raw"]["missing_or_incomplete_partitions"],
                "raw_target_rows": payload["raw"]["target_rows"],
                "raw_complete_rows": payload["raw"]["complete_rows"],
                "funding_target_rows": payload["funding"]["target_rows"],
                "funding_complete_rows": payload["funding"]["complete_rows"],
                "funding_missing_rows": payload["funding"]["missing_rows"],
                "funding_extra_rows": payload["funding"]["extra_rows_in_window"],
                "funding_duplicate_rows": payload["funding"]["duplicate_rows_in_window"],
                "funding_jitter_violations": payload["funding"]["jitter_violation_rows"],
                "funding_error_count": payload["funding"]["error_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] in {"complete", "inventory-complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
