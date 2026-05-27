from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from lumina_quant.data.raw_first_lineage import (
    normalize_exchange_timestamp_ms,
    raw_aggtrades_to_1s_frame,
)
from lumina_quant.timeframe_aggregator import resample_ohlcv_frame_to_timeframe


def test_normalize_exchange_timestamp_ms_rejects_seconds_and_microseconds() -> None:
    with pytest.raises(ValueError):
        normalize_exchange_timestamp_ms(1_700_000_000, source="seconds")
    with pytest.raises(ValueError):
        normalize_exchange_timestamp_ms(1_700_000_000_000_000, source="microseconds")
    assert (
        normalize_exchange_timestamp_ms(1_700_000_000_000, source="milliseconds")
        == 1_700_000_000_000
    )


def test_raw_aggtrades_to_1s_frame_forward_fills_missing_seconds() -> None:
    frame = raw_aggtrades_to_1s_frame(
        [
            {
                "agg_trade_id": 1,
                "timestamp_ms": 1_735_689_600_000,
                "price": 100.0,
                "quantity": 0.1,
                "is_buyer_maker": False,
            },
            {
                "agg_trade_id": 2,
                "timestamp_ms": 1_735_689_602_000,
                "price": 102.0,
                "quantity": 0.2,
                "is_buyer_maker": True,
            },
        ],
        source="pytest",
        range_start_ms=1_735_689_600_000,
        range_end_ms=1_735_689_602_999,
        complete_through_ms=1_735_689_602_999,
    )

    assert frame.height == 3
    assert frame["close"].to_list() == [100.0, 100.0, 102.0]
    assert frame["volume"].to_list() == [0.1, 0.0, 0.2]


def test_resample_ohlcv_frame_to_timeframe_drops_incomplete_last_bucket() -> None:
    source = pl.DataFrame(
        {
            "datetime": [
                datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
                datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1.0, 2.0, 3.0],
        }
    ).with_columns(pl.col("datetime").dt.replace_time_zone(None).cast(pl.Datetime(time_unit="ms")))

    rebuilt = resample_ohlcv_frame_to_timeframe(
        source,
        source_timeframe="30m",
        timeframe="1h",
        drop_incomplete_last=True,
    )

    assert rebuilt.height == 1
    assert rebuilt["datetime"][0] == datetime(2026, 1, 1, 0, 0)
    assert rebuilt["open"][0] == 100.0
    assert rebuilt["close"][0] == 102.0
    assert rebuilt["volume"][0] == 3.0
