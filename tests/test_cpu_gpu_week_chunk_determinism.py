from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from lumina_quant.compute_engine import GPUNotAvailableError, select_engine


def test_cpu_vs_gpu_week_chunk_within_tolerance():
    try:
        gpu_engine = select_engine(mode="gpu", verbose=False)
    except GPUNotAvailableError:
        pytest.skip("GPU engine is unavailable on this machine")

    cpu_engine = select_engine(mode="cpu", verbose=False)

    start = datetime(2026, 1, 1, 0, 0, 0)
    end = start + timedelta(days=7) - timedelta(seconds=1)
    datetimes = pl.datetime_range(start, end, interval="1s", eager=True)
    n = len(datetimes)

    idx = pl.int_range(0, n, eager=True).cast(pl.Float64)
    frame = pl.DataFrame(
        {
            "datetime": datetimes,
            "open": idx + 100.0,
            "high": idx + 100.3,
            "low": idx + 99.7,
            "close": idx + 100.1,
            "volume": (idx % 50.0) + 1.0,
        }
    )

    tf_ms = 60_000
    lazy = (
        frame.lazy()
        .with_columns(pl.col("datetime").dt.epoch("ms").alias("timestamp_ms"))
        .with_columns(((pl.col("timestamp_ms") // tf_ms) * tf_ms).alias("bucket_ms"))
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
    )

    cpu = cpu_engine.collect(lazy)
    gpu = gpu_engine.collect(lazy)

    assert cpu.height == gpu.height
    for col in ["open", "high", "low", "close", "volume"]:
        cpu_vals = cpu[col].to_list()
        gpu_vals = gpu[col].to_list()
        assert len(cpu_vals) == len(gpu_vals)
        for a, b in zip(cpu_vals, gpu_vals, strict=False):
            assert abs(float(a) - float(b)) <= 1e-9
