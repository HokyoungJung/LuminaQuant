"""Raw aggTrades -> committed materialized OHLCV service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.data.raw_first_lineage import (
    normalize_timeframe_token,
    raw_aggtrades_to_1s_frame,
    resample_1s_frame,
)
from lumina_quant.storage.parquet import ParquetMarketDataRepository

_OHLCV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


@dataclass(slots=True)
class MaterializedCommit:
    """One committed partition metadata record."""

    exchange: str
    symbol: str
    timeframe: str
    partition: str
    commit_id: str
    row_count: int
    canonical_row_checksum: str
    manifest_path: str


@dataclass(slots=True)
class MaterializedBundleResult:
    """One materializer cycle result across required timeframes."""

    boundary_id: str
    commits_by_timeframe: dict[str, list[MaterializedCommit]]
    missing_timeframes: tuple[str, ...]


def _empty_ohlcv_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "datetime": pl.Datetime(time_unit="ms"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )


def _resample_from_1s(
    frame_1s: pl.DataFrame,
    *,
    timeframe: str,
    complete_through_ms: int,
) -> pl.DataFrame:
    token = normalize_timeframe_token(timeframe)
    return resample_1s_frame(
        frame_1s,
        timeframe=token,
        complete_through_ms=int(complete_through_ms),
    ).select(_OHLCV_COLUMNS)


def _coerce_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            return numeric * 1000
        return numeric
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


def _load_raw_and_1s(
    *,
    repo: ParquetMarketDataRepository,
    exchange: str,
    symbol: str,
    start_date: Any,
    end_date: Any,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    raw = repo.load_raw_aggtrades(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    if raw.is_empty():
        raise RawFirstDataMissingError(
            f"No raw aggTrades available for {exchange}:{symbol} in requested range."
        )

    checkpoint = repo.read_raw_checkpoint(exchange=exchange, symbol=symbol)
    raw_start_ms = int(raw["timestamp_ms"].min() or 0)
    complete_through_ms = int(raw["timestamp_ms"].max() or 0)
    observed_until_ms = _coerce_timestamp_ms(checkpoint.get("observed_until_ms"))
    requested_end_ms = _coerce_timestamp_ms(end_date)
    if observed_until_ms is not None and int(observed_until_ms) >= int(raw_start_ms):
        complete_through_ms = min(int(complete_through_ms), int(observed_until_ms))
    if requested_end_ms is not None:
        complete_through_ms = min(int(complete_through_ms), int(requested_end_ms))
    bars_1s = raw_aggtrades_to_1s_frame(
        raw,
        source=f"{exchange}:{symbol}:materializer",
        range_start_ms=int(raw["timestamp_ms"].min() or 0),
        range_end_ms=int(complete_through_ms),
        complete_through_ms=int(complete_through_ms),
    )
    if bars_1s.is_empty():
        raise RawFirstDataMissingError(
            f"Raw aggTrades could not be materialized into OHLCV for {exchange}:{symbol}:1s."
        )
    return raw, bars_1s, int(complete_through_ms)


def _write_materialized_frame(
    *,
    repo: ParquetMarketDataRepository,
    raw: pl.DataFrame,
    materialized: pl.DataFrame,
    exchange: str,
    symbol: str,
    timeframe: str,
    producer: str,
    bundle_boundary_id: str,
    complete_through_ms: int,
) -> list[MaterializedCommit]:
    token = normalize_timeframe_token(timeframe)
    if materialized.is_empty():
        raise RawFirstDataMissingError(
            f"Raw aggTrades could not be materialized into OHLCV for {exchange}:{symbol}:{token}."
        )

    with_partition = materialized.with_columns(
        pl.col("datetime").dt.strftime("%Y-%m-%d").alias("partition_date")
    )
    commits: list[MaterializedCommit] = []

    for partition in with_partition.partition_by("partition_date", maintain_order=True):
        if partition.is_empty():
            continue
        partition_date = str(partition["partition_date"][0])
        payload = partition.drop("partition_date").sort("datetime")
        checksum = repo.canonical_row_checksum(payload)
        commit_id = f"{partition_date.replace('-', '')}-{checksum[:16]}"

        partition_root = repo.materialized_partition_root(
            exchange=exchange,
            symbol=symbol,
            timeframe=token,
            partition_date=partition_date,
        )
        commit_dir = partition_root / f"commit={commit_id}"
        commit_dir.mkdir(parents=True, exist_ok=True)
        data_file = commit_dir / "part-0000.parquet"
        if not data_file.exists():
            tmp_data_file = data_file.with_suffix(".tmp.parquet")
            payload.write_parquet(tmp_data_file, compression="zstd", statistics=True)
            tmp_data_file.replace(data_file)

        dt_min = payload["datetime"].min()
        dt_max = payload["datetime"].max()
        window_start_ms = int(dt_min.timestamp() * 1000) if dt_min is not None else 0
        window_end_ms = int(dt_max.timestamp() * 1000) if dt_max is not None else 0

        raw_partition = raw.filter(
            (pl.col("timestamp_ms") >= int(window_start_ms))
            & (pl.col("timestamp_ms") <= int(window_end_ms))
        )
        checkpoint_start = int(raw_partition["timestamp_ms"].min() or window_start_ms)
        checkpoint_end = int(raw_partition["timestamp_ms"].max() or window_end_ms)

        manifest_payload = {
            "manifest_version": 1,
            "commit_id": commit_id,
            "symbol": symbol,
            "timeframe": token,
            "partition": str(partition_root),
            "window_start_ms": int(window_start_ms),
            "window_end_ms": int(window_end_ms),
            "event_time_watermark_ms": int(complete_through_ms),
            "source_checkpoint_start": int(checkpoint_start),
            "source_checkpoint_end": int(max(checkpoint_end, int(complete_through_ms))),
            "row_count": int(payload.height),
            "canonical_row_checksum": str(checksum),
            "data_files": [str(Path(f"commit={commit_id}") / "part-0000.parquet")],
            "bundle_boundary_id": str(bundle_boundary_id),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "producer": str(producer),
            "status": "committed",
        }
        manifest_path = repo.write_materialized_manifest(
            exchange=exchange,
            symbol=symbol,
            timeframe=token,
            partition_date=partition_date,
            payload=manifest_payload,
        )
        commits.append(
            MaterializedCommit(
                exchange=str(exchange),
                symbol=str(symbol),
                timeframe=token,
                partition=partition_date,
                commit_id=commit_id,
                row_count=int(payload.height),
                canonical_row_checksum=str(checksum),
                manifest_path=str(manifest_path),
            )
        )

    if not commits:
        raise RawFirstDataMissingError(
            f"No partitions were produced for {exchange}:{symbol}:{token}."
        )
    return commits


def materialize_raw_aggtrades_bundle(
    *,
    root_path: str,
    exchange: str,
    symbol: str,
    timeframes: list[str],
    start_date: Any = None,
    end_date: Any = None,
    producer: str = "materialize_from_raw",
    require_complete: bool = True,
) -> MaterializedBundleResult:
    """Materialize required timeframe bundle from one raw load + one commit boundary."""
    repo = ParquetMarketDataRepository(root_path)
    raw, bars_1s, complete_through_ms = _load_raw_and_1s(
        repo=repo,
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    normalized_timeframes = list(dict.fromkeys(normalize_timeframe_token(tf) for tf in timeframes))
    if not normalized_timeframes:
        raise RawFirstDataMissingError("No materializer timeframes were provided.")

    prepared: dict[str, pl.DataFrame] = {}
    missing: list[str] = []
    for timeframe in normalized_timeframes:
        frame = _resample_from_1s(
            bars_1s,
            timeframe=timeframe,
            complete_through_ms=int(complete_through_ms),
        )
        if frame.is_empty():
            missing.append(str(timeframe))
            continue
        prepared[str(timeframe)] = frame

    if require_complete and missing:
        joined = ", ".join(missing)
        raise RawFirstDataMissingError(
            "Materializer required timeframe set is incomplete for "
            f"{exchange}:{symbol}. Missing: {joined}."
        )

    boundary_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    commits_by_timeframe: dict[str, list[MaterializedCommit]] = {}
    for timeframe in normalized_timeframes:
        frame = prepared.get(str(timeframe))
        if frame is None:
            continue
        commits_by_timeframe[str(timeframe)] = _write_materialized_frame(
            repo=repo,
            raw=raw,
            materialized=frame,
            exchange=exchange,
            symbol=symbol,
            timeframe=str(timeframe),
            producer=str(producer),
            bundle_boundary_id=boundary_id,
            complete_through_ms=int(complete_through_ms),
        )

    if require_complete and len(commits_by_timeframe) != len(normalized_timeframes):
        missing_remaining = [tf for tf in normalized_timeframes if tf not in commits_by_timeframe]
        joined = ", ".join(missing_remaining)
        raise RawFirstDataMissingError(
            "Materializer required timeframe set is incomplete for "
            f"{exchange}:{symbol}. Missing: {joined}."
        )

    return MaterializedBundleResult(
        boundary_id=boundary_id,
        commits_by_timeframe=commits_by_timeframe,
        missing_timeframes=tuple(missing),
    )


def materialize_raw_aggtrades(
    *,
    root_path: str,
    exchange: str,
    symbol: str,
    timeframe: str = "1s",
    start_date: Any = None,
    end_date: Any = None,
    producer: str = "materialize_from_raw",
) -> list[MaterializedCommit]:
    """Materialize raw aggTrades into immutable committed partitions + manifest."""
    result = materialize_raw_aggtrades_bundle(
        root_path=root_path,
        exchange=exchange,
        symbol=symbol,
        timeframes=[timeframe],
        start_date=start_date,
        end_date=end_date,
        producer=producer,
        require_complete=True,
    )
    token = normalize_timeframe_token(timeframe)
    commits = result.commits_by_timeframe.get(token)
    if not commits:
        raise RawFirstDataMissingError(
            f"No partitions were produced for {exchange}:{symbol}:{token}."
        )
    return commits


__all__ = [
    "MaterializedBundleResult",
    "MaterializedCommit",
    "materialize_raw_aggtrades",
    "materialize_raw_aggtrades_bundle",
]
