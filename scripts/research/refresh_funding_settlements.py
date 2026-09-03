#!/usr/bin/env python3
"""Refresh immutable official funding settlements without copying market data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from lumina_quant.data_sync import _fetch_funding_history, _normalize_funding_timestamp_ms
from lumina_quant.market_data import normalize_symbol
from lumina_quant.research.run_card import atomic_write_text, stable_json_dumps
from lumina_quant.research_universe import research_symbols_for_strategy


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _selection_symbols(path: Path, *, limit: int) -> tuple[str, ...]:
    payload = json.loads(path.read_bytes())
    selected = payload.get("selected")
    if type(selected) is not list:
        raise ValueError("selection artifact must contain selected")
    names = [
        row["strategy"]
        for row in selected[:limit]
        if type(row) is dict and type(row.get("strategy")) is str
    ]
    if not names:
        raise ValueError("selection contains no strategies")
    return tuple(
        dict.fromkeys(symbol for name in names for symbol in research_symbols_for_strategy(name))
    )


def _atomic_parquet(path: Path, frame: pl.DataFrame) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        payload = temporary.read_bytes()
        identity = {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
        if path.exists():
            existing = path.read_bytes()
            if _sha256_bytes(existing) != identity["sha256"]:
                raise ValueError(f"immutable funding settlement conflicts: {path}")
            return identity
        os.replace(temporary, path)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return identity
    finally:
        temporary.unlink(missing_ok=True)


def refresh(
    *,
    data_root: Path,
    selection: Path,
    start: datetime,
    end: datetime,
    limit: int,
    output: Path,
) -> dict[str, object]:
    if not start < end:
        raise ValueError("end must be after start")
    if data_root.is_symlink() or not data_root.is_dir():
        raise ValueError("data root must be a nonsymlink directory")
    symbols = _selection_symbols(selection, limit=limit)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000) - 1
    artifacts: list[dict[str, object]] = []
    source_receipts: list[dict[str, object]] = []
    for symbol in symbols:
        receipts: list[dict[str, object]] = []
        fetched = _fetch_funding_history(
            symbol=symbol,
            since_ms=start_ms,
            until_ms=end_ms,
            retries=5,
            base_wait_sec=0.5,
            source_receipts=receipts,
        )
        source_receipts.extend(receipts)
        by_timestamp: dict[int, dict[str, object]] = {}
        for row in fetched:
            if str(row.get("symbol") or "") != normalize_symbol(symbol).replace("/", ""):
                raise ValueError(f"funding response symbol mismatch: {symbol}")
            source_timestamp = int(row.get("fundingTime") or 0)
            timestamp = _normalize_funding_timestamp_ms(source_timestamp)
            rate = float(row.get("fundingRate"))
            mark_price = float(row.get("markPrice"))
            if (
                timestamp < start_ms
                or timestamp >= int(end.timestamp() * 1000)
                or not math.isfinite(rate)
                or not math.isfinite(mark_price)
                or mark_price <= 0.0
            ):
                continue
            candidate = {
                "exchange": "binance",
                "symbol": normalize_symbol(symbol),
                "timestamp_ms": timestamp,
                "source_timestamp_ms": source_timestamp,
                "datetime": datetime.fromtimestamp(timestamp / 1000, UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "source": "binance_funding_rate_history",
                "funding_rate": rate,
                "funding_mark_price": mark_price,
                "funding_fee_rate": rate,
                "funding_fee_quote_per_unit": rate * mark_price,
            }
            existing = by_timestamp.get(timestamp)
            if existing is not None and existing != candidate:
                raise ValueError(f"ambiguous funding settlement: {symbol}:{timestamp}")
            by_timestamp[timestamp] = candidate
        if not by_timestamp:
            continue
        frame = pl.DataFrame(list(by_timestamp.values())).sort("timestamp_ms")
        frame = frame.with_columns(
            pl.from_epoch("timestamp_ms", time_unit="ms").dt.date().cast(pl.Utf8).alias("_day")
        )
        for partition in frame.partition_by("_day", maintain_order=True):
            day = str(partition["_day"][0])
            stored = partition.drop("_day")
            path = (
                data_root
                / "funding_settlements"
                / "exchange=binance"
                / f"symbol={normalize_symbol(symbol).replace('/', '')}"
                / f"date={day}"
                / "official.parquet"
            )
            identity = _atomic_parquet(path, stored)
            artifacts.append(
                {
                    "path": str(path.relative_to(data_root)),
                    "symbol": normalize_symbol(symbol),
                    "day": day,
                    "rows": stored.height,
                    **identity,
                }
            )
    result = {
        "artifact_kind": "lumina_quant.official_funding_settlement_refresh.v1",
        "status": "complete",
        "data_root": str(data_root.resolve()),
        "selection": {
            "path": str(selection.resolve()),
            "sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        },
        "window": {"start_utc": start.isoformat(), "end_exclusive_utc": end.isoformat()},
        "symbols": list(symbols),
        "artifacts": artifacts,
        "source_receipts": source_receipts,
        "order_routing_enabled": False,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_text(output, stable_json_dumps(result) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = refresh(
        data_root=args.data_root.resolve(),
        selection=args.selection.resolve(),
        start=_datetime(args.start),
        end=_datetime(args.end),
        limit=max(1, args.limit),
        output=args.output.resolve(),
    )
    print(json.dumps({"status": result["status"], "artifacts": len(result["artifacts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
