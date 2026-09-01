#!/usr/bin/env python3
"""Verify the activated Alpha-Max generation through the public data pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.data.feature_points import FeaturePointLookup  # noqa: E402
from lumina_quant.data.raw_first_lineage import resample_1s_frame  # noqa: E402
from lumina_quant.market_data import MarketDataRepository  # noqa: E402
from lumina_quant.storage.parquet import ParquetMarketDataRepository  # noqa: E402
from scripts.research.audit_alpha_max_canonical_contract import (  # noqa: E402
    CONTRACT_SHA256,
    FUNDING_JITTER_MS,
    expected_funding_times,
    load_contract,
    utc_ms,
)

TIMEFRAMES = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}
SAMPLE_DURATION_MS = 2 * 3_600_000


def _aligned_sample_window(record: dict[str, Any]) -> tuple[int, int]:
    raw_start = utc_ms(record["raw_availability_start_utc"])
    raw_end = utc_ms(record["raw_availability_end_utc"])
    start_ms = ((raw_start + 3_600_000 - 1) // 3_600_000) * 3_600_000
    end_ms = min(start_ms + SAMPLE_DURATION_MS, raw_end)
    end_ms = end_ms // 3_600_000 * 3_600_000
    if end_ms - start_ms < SAMPLE_DURATION_MS:
        start_ms = end_ms - SAMPLE_DURATION_MS
    if start_ms < raw_start or end_ms > raw_end or end_ms - start_ms != SAMPLE_DURATION_MS:
        raise ValueError(
            f"contract has no aligned two-hour verification window: {record['symbol']}"
        )
    return start_ms, end_ms


def _datetime_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, UTC).replace(tzinfo=None)


def _frame_digest(frame: pl.DataFrame) -> str:
    return hashlib.sha256(frame.write_csv().encode()).hexdigest()


def _assert_exact_grid(frame: pl.DataFrame, *, start_ms: int, end_ms: int, step_ms: int) -> None:
    expected_rows = (end_ms - start_ms) // step_ms
    timestamps = frame.get_column("datetime").dt.epoch("ms")
    if (
        frame.height != expected_rows
        or timestamps.n_unique() != expected_rows
        or timestamps[0] != start_ms
        or timestamps[-1] != end_ms - step_ms
        or timestamps.to_list() != list(range(start_ms, end_ms, step_ms))
        or sum(frame.null_count().row(0)) != 0
    ):
        raise ValueError("loaded frame does not match the exact requested grid")


def verify_record(db_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    symbol = str(record["symbol"])
    start_ms, end_ms = _aligned_sample_window(record)
    parquet = ParquetMarketDataRepository(db_path)
    raw = parquet.load_ohlcv(
        exchange="binance",
        symbol=symbol,
        timeframe="1s",
        start_date=_datetime_ms(start_ms),
        end_date=_datetime_ms(end_ms - 1_000),
    )
    _assert_exact_grid(raw, start_ms=start_ms, end_ms=end_ms, step_ms=1_000)

    timeframe_receipts: dict[str, Any] = {}
    for timeframe, step_ms in TIMEFRAMES.items():
        loaded = parquet.load_ohlcv(
            exchange="binance",
            symbol=symbol,
            timeframe=timeframe,
            start_date=_datetime_ms(start_ms),
            end_date=_datetime_ms(end_ms - 1_000),
        )
        expected = resample_1s_frame(
            raw,
            timeframe=timeframe,
            complete_through_ms=end_ms - 1,
        )
        _assert_exact_grid(loaded, start_ms=start_ms, end_ms=end_ms, step_ms=step_ms)
        if not loaded.equals(expected):
            raise ValueError(f"{timeframe} loader differs from canonical 1s resampling")
        timeframe_receipts[timeframe] = {
            "rows": loaded.height,
            "sha256": _frame_digest(loaded),
        }

    facade = MarketDataRepository(str(db_path))
    if not facade._prefer_1s_derived:
        raise ValueError("public facade is not configured for raw-first precedence")
    public_1m, source_audit = facade.load_ohlcv_with_source_audit(
        exchange="binance",
        symbol=symbol,
        timeframe="1m",
        start_date=_datetime_ms(start_ms),
        end_date=_datetime_ms(end_ms - 60_000),
    )
    if (
        not public_1m.equals(
            parquet.load_ohlcv(
                exchange="binance",
                symbol=symbol,
                timeframe="1m",
                start_date=_datetime_ms(start_ms),
                end_date=_datetime_ms(end_ms - 60_000),
            )
        )
        or source_audit["precedence"] != "resampled_1s_derived_over_direct_1m"
        or source_audit["resampled_rows"] != SAMPLE_DURATION_MS // 60_000
        or source_audit["effective_resampled_rows"] != SAMPLE_DURATION_MS // 60_000
    ):
        raise ValueError("public facade did not use the complete raw-first minute grid")

    feature_start = utc_ms(record["feature_availability_start_utc"])
    feature_end = utc_ms(record["feature_availability_end_utc"])
    funding_times = expected_funding_times(symbol, feature_start, feature_end)
    if not funding_times:
        raise ValueError(f"contract contains no funding settlements: {symbol}")
    funding_settlement_ms = funding_times[len(funding_times) // 2]
    funding_query_ms = funding_settlement_ms + FUNDING_JITTER_MS
    lookup = FeaturePointLookup(
        db_path=str(db_path),
        exchange="binance",
        start_date=funding_query_ms,
        end_date=funding_query_ms,
    )
    funding_rate = lookup.get_latest(
        symbol,
        "funding_rate",
        timestamp_ms=funding_query_ms,
    )
    if funding_rate is None or not math.isfinite(funding_rate):
        raise ValueError(f"public feature lookup cannot read exact funding: {symbol}")

    return {
        "symbol": symbol,
        "sample_start_ms": start_ms,
        "sample_end_ms_exclusive": end_ms,
        "raw_1s_rows": raw.height,
        "raw_1s_sha256": _frame_digest(raw),
        "timeframes": timeframe_receipts,
        "public_1m_source_audit": source_audit,
        "funding_settlement_ms": funding_settlement_ms,
        "funding_query_ms": funding_query_ms,
        "funding_rate": funding_rate,
        "status": "complete",
    }


def verify_pipeline(*, contract_path: Path, db_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    logical_repository = ParquetMarketDataRepository(db_path)
    with logical_repository.generation_lock(exclusive=False) as pinned_root:
        pinned_info = pinned_root.stat()
        if not stat.S_ISDIR(pinned_info.st_mode):
            raise ValueError("pinned market-data generation is not a directory")
        pinned_identity = {
            "dev": int(pinned_info.st_dev),
            "ino": int(pinned_info.st_ino),
            "mode": int(stat.S_IMODE(pinned_info.st_mode)),
        }
        pinned_generation_path = str(pinned_root.resolve())
        records = [verify_record(pinned_root, record) for record in contract["records"]]
        rebound = pinned_root.stat()
        if (rebound.st_dev, rebound.st_ino) != (
            pinned_info.st_dev,
            pinned_info.st_ino,
        ):
            raise ValueError("pinned market-data generation changed")
    return {
        "artifact_kind": "alpha_max_canonical_pipeline_verification",
        "contract_sha256": CONTRACT_SHA256,
        "db_path": str(db_path.resolve()),
        "pinned_generation_path": pinned_generation_path,
        "pinned_generation_identity": pinned_identity,
        "raw_first_precedence": True,
        "order_routing": False,
        "symbol_count": len(records),
        "symbols": records,
        "status": "complete"
        if all(item["status"] == "complete" for item in records)
        else "incomplete",
        "verified_at_utc": datetime.now(UTC).isoformat(),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short JSON report write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.contract, args.db, args.output):
        if not path.is_absolute():
            raise ValueError("all paths must be absolute")
    payload = verify_pipeline(contract_path=args.contract, db_path=args.db)
    _write_json_atomic(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
