from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from lumina_quant.eval.final_validation import (
    build_latest_anchored_split,
    load_real_ohlcv_frame,
)
from lumina_quant.storage.parquet import ParquetMarketDataRepository


def _write_materialized_frame(
    root: Path,
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    partition_date: str,
    frame: pl.DataFrame,
) -> None:
    repo = ParquetMarketDataRepository(str(root))
    partition_root = repo.materialized_partition_root(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        partition_date=partition_date,
    )
    commit_id = f"seed-{timeframe}"
    commit_dir = partition_root / f"commit={commit_id}"
    commit_dir.mkdir(parents=True, exist_ok=True)
    data_file = commit_dir / "part-0000.parquet"
    frame.write_parquet(data_file)
    ts_start = int(frame["datetime"].min().timestamp() * 1000)
    ts_end = int(frame["datetime"].max().timestamp() * 1000)
    repo.write_materialized_manifest(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        partition_date=partition_date,
        payload={
            "manifest_version": 1,
            "commit_id": commit_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "partition": str(partition_root),
            "window_start_ms": ts_start,
            "window_end_ms": ts_end,
            "event_time_watermark_ms": ts_end,
            "source_checkpoint_start": ts_start,
            "source_checkpoint_end": ts_end,
            "row_count": int(frame.height),
            "canonical_row_checksum": repo.canonical_row_checksum(frame),
            "data_files": [f"commit={commit_id}/part-0000.parquet"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "producer": "pytest",
            "status": "committed",
        },
    )


def test_build_latest_anchored_split_trims_from_left() -> None:
    saved_oos_end = datetime(2026, 3, 17, 23, 59, 59, tzinfo=UTC)
    anchored_end = datetime(2026, 3, 20, 23, 59, 59, tzinfo=UTC)

    split = build_latest_anchored_split(
        saved_oos_end=saved_oos_end,
        anchored_oos_end=anchored_end,
    )

    assert split.oos_end == anchored_end
    assert split.train_start == datetime(2025, 1, 4, 0, 0, 0, tzinfo=UTC)
    assert split.val_start == datetime(2026, 1, 4, 0, 0, 0, tzinfo=UTC)
    assert split.oos_start == datetime(2026, 2, 4, 0, 0, 0, tzinfo=UTC)


def test_load_real_ohlcv_frame_rebuilds_from_lower_timeframe_and_drops_incomplete_tail(
    tmp_path: Path,
) -> None:
    frame_30m = pl.DataFrame(
        {
            "datetime": [
                datetime(2026, 1, 1, 0, 0),
                datetime(2026, 1, 1, 0, 30),
                datetime(2026, 1, 1, 1, 0),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1.0, 2.0, 3.0],
        }
    ).with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))
    _write_materialized_frame(
        tmp_path,
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="30m",
        partition_date="2026-01-01",
        frame=frame_30m,
    )

    rebuilt, info = load_real_ohlcv_frame(
        root_path=str(tmp_path),
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2026-01-01T00:00:00Z",
        end_date="2026-01-01T02:00:00Z",
    )

    assert info.rebuilt_from_lower_timeframe is True
    assert info.source_timeframe == "30m"
    assert rebuilt.height == 1
    assert rebuilt["datetime"][0] == datetime(2026, 1, 1, 0, 0)
    assert rebuilt["close"][0] == 102.0
