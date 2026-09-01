from __future__ import annotations

from datetime import UTC, datetime, timedelta
import math

import numpy as np
import polars as pl
import pytest
from lumina_quant._compute import grouped_fsum
from lumina_quant.data.raw_first_lineage import (
    normalize_exchange_timestamp_ms,
    raw_aggtrades_to_1s_frame,
    resample_1s_frame,
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


def test_resample_1s_frame_volume_is_bitwise_stable_across_calls_and_chunks() -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    volumes = [float(2**53), *([1.0] * 299), float(2**53), *([1.0] * 299)]
    frame = pl.DataFrame(
        {
            "datetime": [base + timedelta(seconds=offset) for offset in range(len(volumes))],
            "open": [100.0 + offset for offset in range(len(volumes))],
            "high": [101.0 + offset for offset in range(len(volumes))],
            "low": [99.0 + offset for offset in range(len(volumes))],
            "close": [100.5 + offset for offset in range(len(volumes))],
            "volume": volumes,
        }
    ).with_columns(pl.col("datetime").dt.replace_time_zone(None).cast(pl.Datetime(time_unit="ms")))
    chunked = pl.concat(
        [frame.slice(0, 137), frame.slice(137, 134), frame.slice(271)],
        rechunk=False,
    )
    complete_through_ms = int((base + timedelta(seconds=len(volumes))).timestamp() * 1000) - 1

    first = resample_1s_frame(
        frame,
        timeframe="5m",
        complete_through_ms=complete_through_ms,
    )
    repeated = resample_1s_frame(
        frame,
        timeframe="5m",
        complete_through_ms=complete_through_ms,
    )
    chunked_result = resample_1s_frame(
        chunked,
        timeframe="5m",
        complete_through_ms=complete_through_ms,
    )

    assert first.equals(repeated)
    assert first.equals(chunked_result)
    assert first["open"].to_list() == [100.0, 400.0]
    assert first["close"].to_list() == [399.5, 699.5]
    assert [value.hex() for value in first["volume"].to_list()] == [
        math.fsum(volumes[:300]).hex(),
        math.fsum(volumes[300:]).hex(),
    ]


def test_native_grouped_fsum_matches_python_exactly_and_rejects_invalid_input() -> None:
    buckets = np.array([1, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    values = np.array(
        [2**53, 1.0, 1.0, -(2**53), 1e100, 1.0, -1e100],
        dtype=np.float64,
    )

    actual_buckets, actual_sums = grouped_fsum(buckets, values)

    assert actual_buckets.tolist() == [1, 2]
    assert [value.hex() for value in actual_sums] == [
        math.fsum(values[:4]).hex(),
        math.fsum(values[4:]).hex(),
    ]
    with pytest.raises(ValueError, match="grouped_fsum_bucket_order_invalid"):
        grouped_fsum(np.array([2, 1], dtype=np.int64), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="grouped_fsum_nonfinite_value"):
        grouped_fsum(np.array([1], dtype=np.int64), np.array([math.inf]))
