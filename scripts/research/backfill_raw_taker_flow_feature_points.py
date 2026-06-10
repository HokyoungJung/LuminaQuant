"""Backfill taker-flow feature points from raw Binance aggTrades.

The live-equivalent derivatives alpha reads taker buy/sell volumes through the
same sidecar feature lookup used by live data handlers.  Raw aggTrades already
contain ``is_buyer_maker``; this script aggregates that raw evidence into
cadence-aligned feature-point rows without modifying OHLCV materialization.
"""

from __future__ import annotations

import argparse
import json
import resource
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.configuration import BacktestConfigView, get_default_runtime_config
from lumina_quant.data.feature_points import FEATURE_COLUMNS
from lumina_quant.market_data import normalize_symbol
from lumina_quant.storage.parquet import ParquetMarketDataRepository

BaseConfig = BacktestConfigView(get_default_runtime_config())

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/feature_replay/raw_taker_flow_backfill_latest.json"
)

META_COLUMNS = ("exchange", "symbol", "timestamp_ms", "datetime", "source")
CANONICAL_COLUMNS = (*META_COLUMNS, *FEATURE_COLUMNS)
TAKER_COLUMNS = (
    "taker_buy_base_volume",
    "taker_sell_base_volume",
    "taker_buy_quote_volume",
    "taker_sell_quote_volume",
)


def _coerce_date(value: str) -> date:
    return datetime.fromisoformat(str(value).strip()).date()


def _date_iter(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=idx) for idx in range((end - start).days + 1)]


def _iso_utc_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=UTC).isoformat()


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def aggregate_raw_taker_flow(
    raw: pl.DataFrame,
    *,
    exchange: str,
    symbol: str,
    cadence_seconds: int,
    source: str,
) -> pl.DataFrame:
    """Aggregate raw aggTrade sides into cadence-end feature-point rows."""
    cadence_ms = max(1, int(cadence_seconds)) * 1000
    if raw.is_empty():
        return _empty_feature_frame()

    required = {"timestamp_ms", "price", "quantity", "is_buyer_maker"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"raw aggTrades missing required columns: {missing}")

    normalized_symbol = normalize_symbol(symbol)
    exchange_token = str(exchange).lower()
    aggregated = (
        raw.select(["timestamp_ms", "price", "quantity", "is_buyer_maker"])
        .with_columns(
            [
                pl.col("timestamp_ms").cast(pl.Int64),
                pl.col("price").cast(pl.Float64),
                pl.col("quantity").cast(pl.Float64),
                pl.col("is_buyer_maker").cast(pl.Boolean),
            ]
        )
        .drop_nulls(subset=["timestamp_ms", "price", "quantity", "is_buyer_maker"])
        .with_columns(
            [
                (((pl.col("timestamp_ms") // cadence_ms) + 1) * cadence_ms)
                .cast(pl.Int64)
                .alias("timestamp_ms"),
                (pl.col("price") * pl.col("quantity")).alias("_quote_qty"),
            ]
        )
        .group_by("timestamp_ms")
        .agg(
            [
                pl.when(~pl.col("is_buyer_maker"))
                .then(pl.col("quantity"))
                .otherwise(0.0)
                .sum()
                .alias("taker_buy_base_volume"),
                pl.when(pl.col("is_buyer_maker"))
                .then(pl.col("quantity"))
                .otherwise(0.0)
                .sum()
                .alias("taker_sell_base_volume"),
                pl.when(~pl.col("is_buyer_maker"))
                .then(pl.col("_quote_qty"))
                .otherwise(0.0)
                .sum()
                .alias("taker_buy_quote_volume"),
                pl.when(pl.col("is_buyer_maker"))
                .then(pl.col("_quote_qty"))
                .otherwise(0.0)
                .sum()
                .alias("taker_sell_quote_volume"),
            ]
        )
        .sort("timestamp_ms")
    )
    if aggregated.is_empty():
        return _empty_feature_frame()

    frame = aggregated.with_columns(
        [
            pl.lit(exchange_token).alias("exchange"),
            pl.lit(normalized_symbol).alias("symbol"),
            pl.from_epoch("timestamp_ms", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .dt.to_string()
            .alias("datetime"),
            pl.lit(str(source)).alias("source"),
        ]
    )
    return _align_feature_frame(frame)


def _empty_feature_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={column: _feature_dtype(column) for column in CANONICAL_COLUMNS})


def _feature_dtype(column: str) -> pl.DataType:
    if column == "timestamp_ms":
        return pl.Int64
    if column in {"exchange", "symbol", "datetime", "source"}:
        return pl.Utf8
    return pl.Float64


def _align_feature_frame(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for column in CANONICAL_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=_feature_dtype(column)).alias(column))
    return out.select(CANONICAL_COLUMNS).with_columns(
        [
            pl.col("exchange").cast(pl.Utf8),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("timestamp_ms").cast(pl.Int64),
            pl.col("datetime").cast(pl.Utf8),
            pl.col("source").cast(pl.Utf8),
            *[pl.col(column).cast(pl.Float64) for column in FEATURE_COLUMNS],
        ]
    )


def _feature_symbol_root(*, db_path: Path, exchange: str, symbol: str) -> Path:
    compact = normalize_symbol(symbol).replace("/", "")
    return db_path / "feature_points" / f"exchange={str(exchange).lower()}" / f"symbol={compact}"


def _load_existing_feature_day(
    *,
    db_path: Path,
    exchange: str,
    symbol: str,
    partition_date: date,
) -> pl.DataFrame:
    day_root = _feature_symbol_root(db_path=db_path, exchange=exchange, symbol=symbol) / (
        f"date={partition_date.isoformat()}"
    )
    paths = sorted(day_root.glob("*.parquet"))
    frames: list[pl.DataFrame] = []
    for path in paths:
        try:
            loaded = pl.read_parquet(path)
        except Exception:
            continue
        if not loaded.is_empty():
            frames.append(_align_feature_frame(loaded))
    if not frames:
        return _empty_feature_frame()
    return _align_feature_frame(pl.concat(frames, how="vertical_relaxed"))


def merge_feature_points(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    """Coalesce incoming taker rows with existing funding/OI/liquidation rows."""
    frames = []
    if not existing.is_empty():
        frames.append(_align_feature_frame(existing))
    if not incoming.is_empty():
        frames.append(_align_feature_frame(incoming))
    if not frames:
        return _empty_feature_frame()

    merged = pl.concat(frames, how="vertical_relaxed").sort("timestamp_ms")
    grouped_expr = [
        pl.col("exchange").drop_nulls().last().alias("exchange"),
        pl.col("symbol").drop_nulls().last().alias("symbol"),
        pl.col("datetime").drop_nulls().last().alias("datetime"),
        pl.col("source").drop_nulls().last().alias("source"),
        *[pl.col(column).drop_nulls().last().alias(column) for column in FEATURE_COLUMNS],
    ]
    return _align_feature_frame(
        merged.group_by("timestamp_ms").agg(grouped_expr).sort("timestamp_ms")
    )


def _write_feature_partitions(
    *,
    db_path: Path,
    exchange: str,
    symbol: str,
    incoming: pl.DataFrame,
) -> int:
    if incoming.is_empty():
        return 0
    with_dates = incoming.with_columns(
        pl.from_epoch("timestamp_ms", time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.date()
        .alias("partition_date")
    )
    written = 0
    for partition in with_dates.partition_by("partition_date", maintain_order=True):
        if partition.is_empty():
            continue
        partition_date = partition["partition_date"][0]
        if partition_date is None:
            continue
        day = date.fromisoformat(str(partition_date))
        payload = partition.drop("partition_date")
        existing = _load_existing_feature_day(
            db_path=db_path,
            exchange=exchange,
            symbol=symbol,
            partition_date=day,
        )
        merged = merge_feature_points(existing, payload)
        day_root = _feature_symbol_root(db_path=db_path, exchange=exchange, symbol=symbol) / (
            f"date={day.isoformat()}"
        )
        day_root.mkdir(parents=True, exist_ok=True)
        output = day_root / f"compact-{day.isoformat()}.parquet"
        tmp = output.with_suffix(".tmp.parquet")
        merged.write_parquet(tmp, compression="zstd", statistics=True)
        tmp.replace(output)
        for old in list(day_root.glob("*.parquet")):
            if old == output:
                continue
            try:
                old.unlink()
            except FileNotFoundError:
                pass
        written += int(payload.height)
    return written


def backfill_symbol_day(
    *,
    repo: ParquetMarketDataRepository,
    db_path: Path,
    exchange: str,
    symbol: str,
    day: date,
    cadence_seconds: int,
    source: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    start = datetime.combine(day, time.min).replace(tzinfo=None)
    end = datetime.combine(day, time.max).replace(tzinfo=None)
    raw = repo.load_raw_aggtrades(
        exchange=exchange,
        symbol=symbol,
        start_date=start,
        end_date=end,
    )
    if raw.is_empty():
        return {
            "symbol": normalize_symbol(symbol),
            "date": day.isoformat(),
            "raw_rows": 0,
            "feature_rows": 0,
            "written_rows": 0,
            "status": "missing_raw",
        }

    features = aggregate_raw_taker_flow(
        raw,
        exchange=exchange,
        symbol=symbol,
        cadence_seconds=cadence_seconds,
        source=source,
    )
    written = (
        0
        if dry_run
        else _write_feature_partitions(
            db_path=db_path,
            exchange=exchange,
            symbol=symbol,
            incoming=features,
        )
    )
    return {
        "symbol": normalize_symbol(symbol),
        "date": day.isoformat(),
        "raw_rows": int(raw.height),
        "feature_rows": int(features.height),
        "written_rows": int(written),
        "status": "dry_run" if dry_run else "written",
        "rss_mb": round(_rss_mb(), 3),
    }


def run_backfill(
    *,
    db_path: Path,
    exchange: str,
    symbols: list[str],
    start: date,
    end: date,
    cadence_seconds: int,
    output_json: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo = ParquetMarketDataRepository(db_path)
    source = f"raw_aggtrades_taker_flow_{int(cadence_seconds)}s"
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for day in _date_iter(start, end):
            row = backfill_symbol_day(
                repo=repo,
                db_path=db_path,
                exchange=exchange,
                symbol=symbol,
                day=day,
                cadence_seconds=cadence_seconds,
                source=source,
                dry_run=dry_run,
            )
            rows.append(row)
            print(json.dumps({"event": "raw_taker_flow_day", **row}, sort_keys=True), flush=True)
    summary = {
        "artifact_kind": "raw_taker_flow_feature_backfill",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "db_path": str(db_path),
        "exchange": str(exchange).lower(),
        "symbols": [normalize_symbol(symbol) for symbol in symbols],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "cadence_seconds": int(cadence_seconds),
        "dry_run": bool(dry_run),
        "day_count": len(_date_iter(start, end)),
        "rows": rows,
        "totals": {
            "raw_rows": int(sum(int(row.get("raw_rows", 0)) for row in rows)),
            "feature_rows": int(sum(int(row.get("feature_rows", 0)) for row in rows)),
            "written_rows": int(sum(int(row.get("written_rows", 0)) for row in rows)),
            "missing_raw_days": int(sum(1 for row in rows if row.get("status") == "missing_raw")),
            "peak_rss_mb": round(_rss_mb(), 3),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=str(BaseConfig.MARKET_DATA_PARQUET_PATH))
    parser.add_argument("--exchange", default=str(BaseConfig.MARKET_DATA_EXCHANGE))
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--start", required=True, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument("--cadence-seconds", type=int, default=20)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_backfill(
        db_path=Path(args.db_path),
        exchange=str(args.exchange),
        symbols=[item.strip() for item in str(args.symbols).split(",") if item.strip()],
        start=_coerce_date(args.start),
        end=_coerce_date(args.end),
        cadence_seconds=max(1, int(args.cadence_seconds)),
        output_json=Path(args.output_json),
        dry_run=bool(args.dry_run),
    )
    print(
        json.dumps(
            {"output_json": str(Path(args.output_json).resolve()), "totals": summary["totals"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
