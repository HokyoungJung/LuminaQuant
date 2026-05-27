"""Canonical raw-first Binance futures market-data lineage helpers."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

import polars as pl
from lumina_quant.data.native_raw_first_backend import (
    RAW_FIRST_BACKEND_AUTO,
    RAW_FIRST_BACKEND_PYTHON,
    RAW_FIRST_BACKEND_RUST,
    aggregate_raw_aggtrades_to_1s_native,
    describe_raw_first_backend,
    normalize_raw_first_backend,
)

_TIMEFRAME_UNIT_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
    "M": 2_592_000_000,
}
_TIMEFRAME_PATTERN = re.compile(r"^([1-9][0-9]*)([smhdwM])$")

_RAW_COLUMNS = (
    "agg_trade_id",
    "timestamp_ms",
    "price",
    "quantity",
    "is_buyer_maker",
)
_OHLCV_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")


def _empty_ohlcv_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {column: [] for column in _OHLCV_COLUMNS},
        schema={
            "datetime": pl.Datetime(time_unit="ms"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )


def normalize_timeframe_token(timeframe: str) -> str:
    raw = str(timeframe or "").strip()
    if not raw:
        raise ValueError("Timeframe cannot be empty")
    value = raw[:-1].strip()
    unit_raw = raw[-1]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError(f"Invalid timeframe value: {timeframe}")
    unit = "M" if unit_raw == "M" else unit_raw.lower()
    token = f"{int(value)}{unit}"
    if _TIMEFRAME_PATTERN.fullmatch(token) is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return token


def timeframe_to_milliseconds(timeframe: str) -> int:
    token = normalize_timeframe_token(timeframe)
    unit = token[-1]
    value = int(token[:-1])
    unit_ms = _TIMEFRAME_UNIT_MS.get(unit)
    if unit_ms is None:
        raise ValueError(f"Unsupported timeframe unit in: {timeframe}")
    return int(value * unit_ms)


def normalize_exchange_timestamp_ms(value: Any, *, source: str) -> int:
    """Normalize official Binance timestamps and fail on non-ms units."""
    try:
        ts = int(value)
    except Exception as exc:  # pragma: no cover - defensive input guard
        raise ValueError(f"{source} timestamp is not an integer: {value!r}") from exc

    magnitude = abs(ts)
    if magnitude < 100_000_000_000:
        raise ValueError(f"{source} timestamp must be milliseconds, got seconds-like value {ts}.")
    if magnitude >= 100_000_000_000_000:
        raise ValueError(
            f"{source} timestamp must be milliseconds, got microseconds-like value {ts}."
        )
    return ts


def latest_complete_bucket_start_ms(
    *,
    timeframe: str,
    complete_through_ms: int | None,
) -> int | None:
    """Return the latest fully complete bucket start timestamp in ms."""
    if complete_through_ms is None:
        return None
    bucket_ms = int(timeframe_to_milliseconds(timeframe))
    boundary = int(complete_through_ms)
    if boundary < bucket_ms - 1:
        return None
    return (((boundary + 1) // bucket_ms) * bucket_ms) - bucket_ms


def _empty_raw_aggtrades_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {column: [] for column in _RAW_COLUMNS},
        schema={
            "agg_trade_id": pl.Int64,
            "timestamp_ms": pl.Int64,
            "price": pl.Float64,
            "quantity": pl.Float64,
            "is_buyer_maker": pl.Boolean,
        },
    )


def _coerce_raw_aggtrades_frame_polars(frame: pl.DataFrame, *, source: str) -> pl.DataFrame:
    missing = [column for column in _RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} rows missing raw aggTrades columns: {missing}")

    normalized = (
        frame.select(list(_RAW_COLUMNS))
        .with_columns(
            [
                pl.col("agg_trade_id").cast(pl.Int64),
                pl.col("timestamp_ms").cast(pl.Int64),
                pl.col("price").cast(pl.Float64),
                pl.col("quantity").cast(pl.Float64),
                pl.col("is_buyer_maker").cast(pl.Boolean, strict=False).fill_null(False),
            ]
        )
        .drop_nulls(subset=["timestamp_ms"])
    )
    if normalized.is_empty():
        return _empty_raw_aggtrades_frame()

    timestamp_abs = pl.col("timestamp_ms").abs()
    if normalized.select(timestamp_abs.lt(100_000_000_000).any()).item():
        example = int(normalized.filter(timestamp_abs.lt(100_000_000_000))["timestamp_ms"][0])
        raise ValueError(
            f"{source} timestamp must be milliseconds, got seconds-like value {example}."
        )
    if normalized.select(timestamp_abs.ge(100_000_000_000_000).any()).item():
        example = int(normalized.filter(timestamp_abs.ge(100_000_000_000_000))["timestamp_ms"][0])
        raise ValueError(
            f"{source} timestamp must be milliseconds, got microseconds-like value {example}."
        )

    normalized = normalized.filter((pl.col("price") > 0.0) & (pl.col("quantity") >= 0.0))
    if normalized.is_empty():
        return _empty_raw_aggtrades_frame()

    return (
        normalized.sort(["timestamp_ms", "agg_trade_id"])
        .unique(subset=["timestamp_ms", "agg_trade_id"], keep="last")
        .sort(["timestamp_ms", "agg_trade_id"])
    )


def coerce_raw_aggtrades_frame(
    rows: pl.DataFrame | Iterable[dict[str, Any]],
    *,
    source: str,
) -> pl.DataFrame:
    """Coerce raw aggTrades rows into canonical parquet schema."""
    if isinstance(rows, pl.DataFrame):
        if rows.is_empty():
            return _empty_raw_aggtrades_frame()
        return _coerce_raw_aggtrades_frame_polars(rows, source=source)

    frame = pl.DataFrame(list(rows or []))
    if frame.is_empty():
        return _empty_raw_aggtrades_frame()

    missing = [column for column in _RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} rows missing raw aggTrades columns: {missing}")

    normalized_rows: list[dict[str, Any]] = []
    for row in frame.select(list(_RAW_COLUMNS)).to_dicts():
        timestamp_ms = normalize_exchange_timestamp_ms(row.get("timestamp_ms"), source=source)
        price = float(row.get("price") or 0.0)
        quantity = float(row.get("quantity") or 0.0)
        if price <= 0.0 or quantity < 0.0:
            continue
        normalized_rows.append(
            {
                "agg_trade_id": int(row.get("agg_trade_id") or 0),
                "timestamp_ms": int(timestamp_ms),
                "price": float(price),
                "quantity": float(quantity),
                "is_buyer_maker": bool(row.get("is_buyer_maker")),
            }
        )

    if not normalized_rows:
        return _empty_raw_aggtrades_frame()
    return (
        pl.DataFrame(normalized_rows)
        .select(list(_RAW_COLUMNS))
        .sort(["timestamp_ms", "agg_trade_id"])
        .unique(subset=["timestamp_ms", "agg_trade_id"], keep="last")
        .sort(["timestamp_ms", "agg_trade_id"])
    )


def raw_aggtrades_to_1s_frame(
    rows: pl.DataFrame | Iterable[dict[str, Any]],
    *,
    source: str,
    range_start_ms: int | None = None,
    range_end_ms: int | None = None,
    previous_close: float | None = None,
    complete_through_ms: int | None = None,
    backend: str | None = None,
) -> pl.DataFrame:
    """Derive continuous canonical 1s OHLCV bars from raw aggTrades."""
    raw = coerce_raw_aggtrades_frame(rows, source=source)
    if raw.is_empty():
        return _empty_ohlcv_frame()

    effective_complete_through_ms = (
        int(complete_through_ms)
        if complete_through_ms is not None
        else int(range_end_ms)
        if range_end_ms is not None
        else int(raw.get_column("timestamp_ms")[-1])
    )
    last_complete_second = latest_complete_bucket_start_ms(
        timeframe="1s",
        complete_through_ms=effective_complete_through_ms,
    )
    if last_complete_second is None:
        return _empty_ohlcv_frame()

    first_trade_second = (int(raw.get_column("timestamp_ms")[0]) // 1000) * 1000
    start_second = (
        max(int(range_start_ms) // 1000 * 1000, first_trade_second)
        if previous_close is None and range_start_ms is not None
        else int(range_start_ms) // 1000 * 1000
        if range_start_ms is not None
        else first_trade_second
    )
    end_second = int(last_complete_second)
    if range_end_ms is not None:
        end_second = min(int(range_end_ms) // 1000 * 1000, end_second)
    if end_second < start_second:
        return _empty_ohlcv_frame()

    filtered = (
        raw.filter(
            (pl.col("timestamp_ms") >= int(range_start_ms))
            if range_start_ms is not None
            else pl.lit(True)
        )
        .filter(
            (pl.col("timestamp_ms") <= int(range_end_ms))
            if range_end_ms is not None
            else pl.lit(True)
        )
        .filter(pl.col("timestamp_ms") <= int(effective_complete_through_ms))
    )

    if filtered.is_empty():
        return _empty_ohlcv_frame()

    backend_token = normalize_raw_first_backend(backend)
    native_frame = aggregate_raw_aggtrades_to_1s_native(
        filtered,
        range_start_ms=int(range_start_ms) if range_start_ms is not None else None,
        range_end_ms=int(range_end_ms) if range_end_ms is not None else None,
        previous_close=float(previous_close) if previous_close is not None else None,
        complete_through_ms=int(effective_complete_through_ms),
        backend=backend_token,
    )
    if native_frame is not None:
        return native_frame

    bucketed = (
        filtered.with_columns(((pl.col("timestamp_ms") // 1000) * 1000).alias("bucket_ms"))
        .group_by("bucket_ms", maintain_order=True)
        .agg(
            [
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("quantity").sum().alias("volume"),
            ]
        )
        .sort("bucket_ms")
    )

    bucket_range = pl.DataFrame(
        {
            "bucket_ms": pl.int_range(
                int(start_second),
                int(end_second) + 1000,
                step=1000,
                eager=True,
            )
        }
    )

    carry_close = pl.col("close").forward_fill()
    if previous_close is not None:
        carry_close = carry_close.fill_null(float(previous_close))

    rebuilt = (
        bucket_range.join(bucketed, on="bucket_ms", how="left")
        .sort("bucket_ms")
        .with_columns(carry_close.alias("_carry_close"))
        .filter(pl.col("_carry_close").is_not_null())
        .with_columns(
            [
                pl.coalesce(["open", "_carry_close"]).cast(pl.Float64).alias("open"),
                pl.coalesce(["high", "_carry_close"]).cast(pl.Float64).alias("high"),
                pl.coalesce(["low", "_carry_close"]).cast(pl.Float64).alias("low"),
                pl.coalesce(["close", "_carry_close"]).cast(pl.Float64).alias("close"),
                pl.col("volume").fill_null(0.0).cast(pl.Float64).alias("volume"),
                pl.from_epoch("bucket_ms", time_unit="ms").alias("datetime"),
            ]
        )
        .select(list(_OHLCV_COLUMNS))
        .sort("datetime")
    )

    if rebuilt.is_empty():
        return _empty_ohlcv_frame()
    return rebuilt.with_columns(
        [
            pl.col("datetime").dt.replace_time_zone(None).cast(pl.Datetime(time_unit="ms")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        ]
    )


def resolve_raw_aggtrades_backend_name(backend: str | None = None) -> str:
    token = normalize_raw_first_backend(backend)
    if token == RAW_FIRST_BACKEND_PYTHON:
        return RAW_FIRST_BACKEND_PYTHON
    description = describe_raw_first_backend(token)
    if token == RAW_FIRST_BACKEND_AUTO and description == RAW_FIRST_BACKEND_PYTHON:
        return RAW_FIRST_BACKEND_PYTHON
    if token == RAW_FIRST_BACKEND_RUST and description.startswith(f"{RAW_FIRST_BACKEND_RUST}:"):
        return description
    return description if description else RAW_FIRST_BACKEND_AUTO


def resample_1s_frame(
    frame_1s: pl.DataFrame,
    *,
    timeframe: str,
    complete_through_ms: int | None = None,
) -> pl.DataFrame:
    """Resample canonical 1s bars into higher timeframes and drop incomplete tail."""
    token = normalize_timeframe_token(timeframe)
    if frame_1s.is_empty():
        return frame_1s
    if token == "1s":
        output = frame_1s.select(list(_OHLCV_COLUMNS))
    else:
        bucket_ms = int(timeframe_to_milliseconds(token))
        output = (
            frame_1s.with_columns(pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"))
            .with_columns(((pl.col("timestamp_ms") // bucket_ms) * bucket_ms).alias("bucket_ms"))
            .group_by("bucket_ms")
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
            .select(list(_OHLCV_COLUMNS))
        )

    latest_complete_bucket_ms = latest_complete_bucket_start_ms(
        timeframe=token,
        complete_through_ms=complete_through_ms,
    )
    if latest_complete_bucket_ms is None:
        return output.clear()
    return output.filter(pl.col("datetime").dt.epoch("ms") <= int(latest_complete_bucket_ms)).sort(
        "datetime"
    )


__all__ = [
    "coerce_raw_aggtrades_frame",
    "latest_complete_bucket_start_ms",
    "normalize_exchange_timestamp_ms",
    "raw_aggtrades_to_1s_frame",
    "resample_1s_frame",
    "resolve_raw_aggtrades_backend_name",
]
