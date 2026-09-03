#!/usr/bin/env python3
"""Validate the configured execution model against canonical raw aggTrades.

This is an execution-model gate, not a strategy-performance backtest. It runs
only after walk-forward finalists exist and checks MKT/LMT price and quantity
feasibility against the independent tick tape, one symbol at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from lumina_quant.backtesting.execution_model import ExecutionModelConfig
from lumina_quant.backtesting.tick_replay_validator import TickReplayValidator
from lumina_quant.configuration.loader import load_runtime_config
from lumina_quant.data.native_raw_first_backend import describe_raw_first_backend
from lumina_quant.market_data import normalize_symbol
from lumina_quant.research_universe import research_symbols_for_strategy

import polars as pl


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def load_raw_tick_window(
    *,
    data_root: Path,
    exchange: str,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Read only date partitions intersecting ``[start, end)`` without recovery writes."""
    compact = normalize_symbol(symbol).replace("/", "")
    stream_root = data_root / "market_data_raw_aggtrades" / exchange.strip().lower() / compact
    if stream_root.is_symlink() or not stream_root.is_dir():
        return pl.DataFrame()
    files = [
        path
        for day in _dates(start.date(), (end - timedelta(microseconds=1)).date())
        for path in sorted((stream_root / f"date={day.isoformat()}").glob("part-*.parquet"))
    ]
    if not files or any(path.is_symlink() or not path.is_file() for path in files):
        return pl.DataFrame()
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    frame = (
        pl.scan_parquet(files)
        .filter((pl.col("timestamp_ms") >= start_ms) & (pl.col("timestamp_ms") < end_ms))
        .select(
            "agg_trade_id",
            "timestamp_ms",
            "price",
            "quantity",
            "is_buyer_maker",
        )
        .sort("timestamp_ms", "agg_trade_id")
        .collect(engine="streaming")
    )
    if frame.is_empty():
        return frame
    if (
        frame.null_count().to_numpy().sum() > 0
        or frame.select(pl.struct(frame.columns).is_duplicated().any()).item()
        or (frame["price"] <= 0).any()
        or (frame["quantity"] <= 0).any()
        or (frame["timestamp_ms"].diff().drop_nulls() < 0).any()
    ):
        raise ValueError(f"raw aggTrades window is not canonical: {exchange}:{symbol}")
    return frame


def symbols_from_selection(path: Path, *, limit: int) -> tuple[str, ...]:
    """Resolve finalist strategy names to the centralized research universe."""
    payload = json.loads(path.read_bytes())
    selected = payload.get("selected")
    if type(selected) is not list:
        candidates = payload.get("strategy_results")
        if type(candidates) is not list:
            raise ValueError("selection artifact must contain selected or strategy_results")
        selected = sorted(
            (
                row
                for row in candidates
                if type(row) is dict
                and row.get("status") == "pass"
                and float(row.get("total_return") or 0.0) > 0.0
            ),
            key=lambda row: float(
                (row.get("fast_stats") or {}).get(
                    "sharpe",
                    row.get("sharpe_ratio") or 0.0,
                )
            ),
            reverse=True,
        )
    strategy_names = [
        row.get("strategy")
        for row in selected[:limit]
        if type(row) is dict and type(row.get("strategy")) is str
    ]
    if not strategy_names:
        return ()
    return tuple(
        dict.fromkeys(
            symbol
            for strategy_name in strategy_names
            for symbol in research_symbols_for_strategy(strategy_name)
        )
    )


def validate_ticks(
    *,
    data_root: Path,
    config_path: Path,
    exchange: str,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    output: Path,
) -> dict[str, Any]:
    if end <= start or end - start > timedelta(days=1):
        raise ValueError("tick validation window must be in (0, 24h]")
    if data_root.is_symlink() or not data_root.is_dir():
        raise ValueError("data_root must be a nonsymlink directory")
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("config must be a nonsymlink file")
    normalized_symbols = tuple(dict.fromkeys(str(item).strip().upper() for item in symbols))
    if not normalized_symbols or any(not item for item in normalized_symbols):
        raise ValueError("at least one non-empty symbol is required")
    backend = describe_raw_first_backend()
    if backend != "rust:pyo3":
        raise RuntimeError(f"Rust raw-first backend is required, observed {backend}")
    runtime = load_runtime_config(config_path)
    execution_config = ExecutionModelConfig.from_runtime(runtime)
    rows: list[dict[str, Any]] = []
    for symbol in normalized_symbols:
        raw = load_raw_tick_window(
            data_root=data_root,
            exchange=exchange,
            symbol=symbol,
            start=start,
            end=end,
        )
        if raw.is_empty():
            rows.append(
                {
                    "symbol": symbol,
                    "status": "fail",
                    "error": "canonical_raw_aggtrades_window_empty",
                    "raw_rows": 0,
                }
            )
            continue
        verdict = TickReplayValidator(
            raw,
            execution_cfg=execution_config,
            backend="rust",
        ).validate()
        passed = verdict.lmt_verdict == "PASS" and verdict.mkt_verdict == "PASS"
        rows.append(
            {
                "symbol": symbol,
                "status": "pass" if passed else "fail",
                "raw_rows": raw.height,
                "first_timestamp_ms": int(raw["timestamp_ms"].min()),
                "last_timestamp_ms": int(raw["timestamp_ms"].max()),
                "verdict": verdict.to_dict(),
            }
        )
        del raw
    payload = {
        "artifact_kind": "lumina_quant.execution_model_tick_validation.v1",
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "data_root": str(data_root.resolve()),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
        },
        "exchange": exchange,
        "symbols": list(normalized_symbols),
        "start_utc": start.isoformat(),
        "end_exclusive_utc": end.isoformat(),
        "backend": backend,
        "order_routing_enabled": False,
        "results": rows,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--selection-artifact", type=Path)
    parser.add_argument("--selection-limit", type=int, default=20)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = list(args.symbol or ())
    if args.selection_artifact is not None:
        symbols.extend(
            symbols_from_selection(
                args.selection_artifact.resolve(),
                limit=max(1, int(args.selection_limit)),
            )
        )
    if not symbols:
        payload = {
            "artifact_kind": "lumina_quant.execution_model_tick_validation.v1",
            "status": "skip_no_finalists",
            "data_root": str(args.data_root.resolve()),
            "config": {
                "path": str(args.config.resolve()),
                "sha256": _sha256(args.config.resolve()),
            },
            "exchange": str(args.exchange),
            "symbols": [],
            "window": {"start_utc": str(args.start), "end_exclusive_utc": str(args.end)},
            "order_routing_enabled": False,
            "results": [],
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_write(args.output.resolve(), payload)
        print(json.dumps({"status": payload["status"], "symbols": 0}))
        return 0
    payload = validate_ticks(
        data_root=args.data_root.resolve(),
        config_path=args.config.resolve(),
        exchange=str(args.exchange),
        symbols=symbols,
        start=_datetime(args.start),
        end=_datetime(args.end),
        output=args.output.resolve(),
    )
    print(json.dumps({"status": payload["status"], "symbols": len(payload["results"])}))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
