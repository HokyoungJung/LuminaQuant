"""Inventory helpers for parquet-backed futures support features."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.market_data import load_futures_feature_points_from_db

_LIQUIDATION_COLUMNS: tuple[str, ...] = (
    "liquidation_long_qty",
    "liquidation_short_qty",
    "liquidation_long_notional",
    "liquidation_short_notional",
)

INVENTORY_NOTES: dict[str, str | bool] = {
    "canonical_inventory": True,
    "liquidation": (
        "Rows may be zero where Binance futures liquidation endpoint is unavailable/returns HTTP 400."
    ),
    "index_price": (
        "Collector now uses pair param for indexPriceKlines; older sparse symbols reflect "
        "Binance response history or earlier collector state."
    ),
    "open_interest": (
        "Long-range chunking and empty-window skip are applied; rows reflect what Binance "
        "currently returns for each symbol."
    ),
}


def _iso_utc(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).isoformat()


def discover_feature_point_symbols(*, db_path: str, exchange: str = "binance") -> list[str]:
    root = Path(str(db_path)).expanduser() / "feature_points" / f"exchange={str(exchange).lower()}"
    symbols = sorted(
        path.name.split("=", 1)[1]
        for path in root.glob("symbol=*")
        if path.is_dir() and "=" in path.name
    )
    return symbols


def _column_count(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns or frame.is_empty():
        return 0
    return int(frame.select(pl.col(column).is_not_null().sum().alias("count")).item())


def _max_timestamp_for(frame: pl.DataFrame, predicate: pl.Expr) -> int | None:
    if frame.is_empty():
        return None
    filtered = frame.filter(predicate).select(pl.max("timestamp_ms").alias("max_ts"))
    value = filtered.item() if filtered.height == 1 else None
    return int(value) if value is not None else None


def _min_timestamp_for(frame: pl.DataFrame, predicate: pl.Expr) -> int | None:
    if frame.is_empty():
        return None
    filtered = frame.filter(predicate).select(pl.min("timestamp_ms").alias("min_ts"))
    value = filtered.item() if filtered.height == 1 else None
    return int(value) if value is not None else None


def _liquidation_present_expr() -> pl.Expr:
    expr = pl.lit(False)
    for column in _LIQUIDATION_COLUMNS:
        expr = expr | pl.col(column).is_not_null()
    return expr


def build_strategy_support_inventory(
    *,
    db_path: str,
    exchange: str = "binance",
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    requested_symbols = list(symbols or discover_feature_point_symbols(db_path=db_path, exchange=exchange))
    inventory_rows: list[dict[str, Any]] = []
    liquidation_expr = _liquidation_present_expr()

    for symbol in requested_symbols:
        frame = load_futures_feature_points_from_db(
            str(db_path),
            exchange=str(exchange),
            symbol=str(symbol),
        )
        if frame.is_empty():
            inventory_rows.append(
                {
                    "symbol": str(symbol).replace("/", ""),
                    "rows": 0,
                    "first_timestamp_ms": None,
                    "last_timestamp_ms": None,
                    "funding_rows": 0,
                    "funding_fee_rows": 0,
                    "mark_rows": 0,
                    "index_rows": 0,
                    "open_interest_rows": 0,
                    "liquidation_rows": 0,
                    "oi_first_timestamp_ms": None,
                    "oi_last_timestamp_ms": None,
                    "first_timestamp_utc": None,
                    "last_timestamp_utc": None,
                    "oi_first_timestamp_utc": None,
                    "oi_last_timestamp_utc": None,
                    "has_funding_fee": False,
                    "has_mark": False,
                    "has_index": False,
                    "has_open_interest": False,
                    "has_liquidation": False,
                }
            )
            continue

        cleaned = frame.select(
            [
                "timestamp_ms",
                "funding_rate",
                "funding_fee_rate",
                "funding_fee_quote_per_unit",
                "mark_price",
                "index_price",
                "open_interest",
                *_LIQUIDATION_COLUMNS,
            ]
        ).sort("timestamp_ms")

        first_timestamp_ms = cleaned.select(pl.min("timestamp_ms").alias("min_ts")).item()
        last_timestamp_ms = cleaned.select(pl.max("timestamp_ms").alias("max_ts")).item()
        funding_rows = _column_count(cleaned, "funding_rate")
        funding_fee_rows = int(
            cleaned.select(
                (
                    pl.col("funding_fee_rate").is_not_null()
                    | pl.col("funding_fee_quote_per_unit").is_not_null()
                )
                .sum()
                .alias("count")
            ).item()
        )
        mark_rows = _column_count(cleaned, "mark_price")
        index_rows = _column_count(cleaned, "index_price")
        open_interest_rows = _column_count(cleaned, "open_interest")
        liquidation_rows = int(cleaned.select(liquidation_expr.sum().alias("count")).item())
        oi_present = pl.col("open_interest").is_not_null()
        oi_first_timestamp_ms = _min_timestamp_for(cleaned, oi_present)
        oi_last_timestamp_ms = _max_timestamp_for(cleaned, oi_present)

        inventory_rows.append(
            {
                "symbol": str(symbol).replace("/", ""),
                "rows": int(cleaned.height),
                "first_timestamp_ms": int(first_timestamp_ms) if first_timestamp_ms is not None else None,
                "last_timestamp_ms": int(last_timestamp_ms) if last_timestamp_ms is not None else None,
                "funding_rows": funding_rows,
                "funding_fee_rows": funding_fee_rows,
                "mark_rows": mark_rows,
                "index_rows": index_rows,
                "open_interest_rows": open_interest_rows,
                "liquidation_rows": liquidation_rows,
                "oi_first_timestamp_ms": oi_first_timestamp_ms,
                "oi_last_timestamp_ms": oi_last_timestamp_ms,
                "first_timestamp_utc": _iso_utc(first_timestamp_ms),
                "last_timestamp_utc": _iso_utc(last_timestamp_ms),
                "oi_first_timestamp_utc": _iso_utc(oi_first_timestamp_ms),
                "oi_last_timestamp_utc": _iso_utc(oi_last_timestamp_ms),
                "has_funding_fee": funding_fee_rows > 0,
                "has_mark": mark_rows > 0,
                "has_index": index_rows > 0,
                "has_open_interest": open_interest_rows > 0,
                "has_liquidation": liquidation_rows > 0,
            }
        )

    inventory_rows.sort(key=lambda item: str(item["symbol"]))
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "exchange_id": str(exchange),
        "db_path": str(db_path),
        "symbol_count": len(inventory_rows),
        "symbols": inventory_rows,
        "notes": dict(INVENTORY_NOTES),
    }


def inventory_to_frame(payload: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(list(payload.get("symbols", [])))


def write_strategy_support_inventory(
    *,
    payload: dict[str, Any],
    json_path: str | None = None,
    csv_path: str | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if json_path:
        target = Path(str(json_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["json_path"] = str(target)
    if csv_path:
        target = Path(str(csv_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        inventory_to_frame(payload).write_csv(target)
        outputs["csv_path"] = str(target)
    return outputs


__all__ = [
    "INVENTORY_NOTES",
    "build_strategy_support_inventory",
    "discover_feature_point_symbols",
    "inventory_to_frame",
    "write_strategy_support_inventory",
]
