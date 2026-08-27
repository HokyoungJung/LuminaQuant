from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from lumina_quant.market_data import MarketDataRepository
from scripts.research import verify_alpha_max_canonical_pipeline as subject


def test_verify_record_exercises_raw_resampling_facade_and_funding(tmp_path: Path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=2)
    start_ms = int(start.timestamp() * 1000)
    timestamps = pl.datetime_range(
        start.replace(tzinfo=None),
        (end - timedelta(seconds=1)).replace(tzinfo=None),
        interval="1s",
        eager=True,
        time_unit="ms",
    )
    raw = pl.DataFrame(
        {
            "datetime": timestamps,
            "open": [100.0] * len(timestamps),
            "high": [101.0] * len(timestamps),
            "low": [99.0] * len(timestamps),
            "close": [100.5] * len(timestamps),
            "volume": [1.0] * len(timestamps),
        },
        schema={
            "datetime": pl.Datetime("ms"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )
    target = tmp_path / "market_ohlcv_1s" / "binance" / "BTCUSDT" / "2024-01.parquet"
    target.parent.mkdir(parents=True)
    raw.write_parquet(target)
    feature_repository = MarketDataRepository(str(tmp_path))
    feature_repository.upsert_futures_feature_points(
        exchange="binance",
        symbol="BTCUSDT",
        rows=[{"timestamp_ms": start_ms + 8 * 3_600_000, "funding_rate": 0.0001}],
    )
    record = {
        "symbol": "BTCUSDT",
        "raw_availability_start_utc": "2024-01-01T00:00:00Z",
        "raw_availability_end_utc": "2024-01-01T02:00:00Z",
        "feature_availability_start_utc": "2024-01-01T00:00:00Z",
        "feature_availability_end_utc": "2024-01-01T16:00:00Z",
    }

    receipt = subject.verify_record(tmp_path, record)

    assert receipt["status"] == "complete"
    assert receipt["raw_1s_rows"] == 7_200
    assert receipt["timeframes"]["1m"]["rows"] == 120
    assert receipt["timeframes"]["5m"]["rows"] == 24
    assert receipt["timeframes"]["1h"]["rows"] == 2
    assert receipt["public_1m_source_audit"]["effective_resampled_rows"] == 120
    assert receipt["funding_rate"] == 0.0001
