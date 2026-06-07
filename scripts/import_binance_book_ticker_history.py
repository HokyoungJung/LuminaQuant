#!/usr/bin/env python3
"""Import historical Binance book-ticker/BBO rows into feature points.

This is a generic ingestion path for externally captured Binance USD-M top-of-book
history. It does not bless any vendor; it only normalizes already-approved
historical rows into the local feature-point store so BBO-aware clean research
can eventually satisfy train/validation coverage requirements.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.market_data import normalize_symbol, upsert_futures_feature_points_rows

_REQUIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp_ms": ("timestamp_ms", "exchange_ts_ms", "ts_ms", "ts", "time", "timestamp"),
    "symbol": ("symbol", "s"),
    "best_bid_price": ("best_bid_price", "bid_price", "bid", "b"),
    "best_bid_quantity": ("best_bid_quantity", "bid_quantity", "bid_qty", "B"),
    "best_ask_price": ("best_ask_price", "ask_price", "ask", "a"),
    "best_ask_quantity": ("best_ask_quantity", "ask_quantity", "ask_qty", "A"),
}


def _read_input(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.read_ndjson(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"unsupported input format: {path.suffix}")


def _resolve_column(columns: list[str], aliases: tuple[str, ...], *, required: bool) -> str | None:
    exact = set(columns)
    for alias in aliases:
        if alias in exact:
            return alias
    lower = {column.lower(): column for column in columns}
    for alias in aliases:
        found = lower.get(alias.lower())
        if found is not None:
            return found
    if required:
        raise ValueError(f"missing required aliases {aliases}")
    return None


def _coerce_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return int(timestamp.timestamp() * 1000)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value) if value == value else None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return int(timestamp.timestamp() * 1000)


def _to_timestamp_ms_expr(column: str) -> pl.Expr:
    return pl.col(column).map_elements(_coerce_timestamp_ms, return_dtype=pl.Int64)


def normalize_bbo_frame(frame: pl.DataFrame, *, symbol_override: str | None = None) -> pl.DataFrame:
    columns = list(frame.columns)
    mapping = {
        key: _resolve_column(
            columns, aliases, required=(key != "symbol" or symbol_override is None)
        )
        for key, aliases in _REQUIRED_ALIASES.items()
    }
    payload = frame
    if mapping["symbol"] is None:
        payload = payload.with_columns(pl.lit(symbol_override).alias("symbol"))
        mapping["symbol"] = "symbol"
    payload = payload.select(
        [
            _to_timestamp_ms_expr(mapping["timestamp_ms"]).alias("timestamp_ms"),
            pl.col(mapping["symbol"]).cast(pl.Utf8).alias("symbol"),
            pl.col(mapping["best_bid_price"]).cast(pl.Float64).alias("best_bid_price"),
            pl.col(mapping["best_bid_quantity"]).cast(pl.Float64).alias("best_bid_quantity"),
            pl.col(mapping["best_ask_price"]).cast(pl.Float64).alias("best_ask_price"),
            pl.col(mapping["best_ask_quantity"]).cast(pl.Float64).alias("best_ask_quantity"),
        ]
    )
    payload = payload.filter(
        pl.col("timestamp_ms").is_not_null()
        & pl.col("symbol").is_not_null()
        & pl.col("best_bid_price").is_not_null()
        & pl.col("best_ask_price").is_not_null()
    )
    payload = payload.with_columns(
        [
            pl.col("symbol").map_elements(normalize_symbol, return_dtype=pl.Utf8),
            ((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2.0).alias("bbo_mid_price"),
            pl.when((pl.col("best_bid_price") + pl.col("best_ask_price")) > 0)
            .then(
                (
                    (pl.col("best_ask_price") - pl.col("best_bid_price"))
                    / ((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2.0)
                )
                * 10000.0
            )
            .otherwise(None)
            .alias("bbo_spread_bps"),
        ]
    )
    return payload.sort(["symbol", "timestamp_ms"])


def import_bbo_history(
    *,
    db_path: str,
    input_path: Path,
    exchange: str = "binance",
    symbol_override: str | None = None,
    source: str = "external_binance_bbo_history",
) -> dict[str, Any]:
    frame = _read_input(input_path)
    normalized = normalize_bbo_frame(frame, symbol_override=symbol_override)
    persisted: dict[str, int] = {}
    for symbol in normalized.get_column("symbol").unique().to_list():
        rows = normalized.filter(pl.col("symbol") == symbol).drop("symbol").to_dicts()
        persisted[str(symbol)] = upsert_futures_feature_points_rows(
            db_path,
            exchange=exchange,
            symbol=str(symbol),
            rows=rows,
            source=source,
        )
    return {
        "artifact_kind": "binance_book_ticker_history_import",
        "input_path": str(input_path),
        "exchange": str(exchange),
        "symbol_override": symbol_override,
        "row_count": int(normalized.height),
        "symbols": normalized.get_column("symbol").unique().to_list(),
        "persisted_rows_by_symbol": persisted,
        "total_persisted_rows": int(sum(persisted.values())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("--db-path", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol-override", default=None)
    parser.add_argument("--source", default="external_binance_bbo_history")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = import_bbo_history(
        db_path=str(args.db_path),
        input_path=Path(args.input_path).expanduser().resolve(),
        exchange=str(args.exchange),
        symbol_override=(str(args.symbol_override).strip() or None),
        source=str(args.source),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
