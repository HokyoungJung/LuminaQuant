"""Timeframe panel preparation for cost-aware experiments."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import polars as pl
from lumina_quant.market_data import normalize_timeframe_token, timeframe_to_milliseconds
from lumina_quant.parquet_market_data import ParquetMarketDataRepository


def _clean_ohlcv(frame: pl.DataFrame) -> pl.DataFrame:
    required = ["datetime", "open", "high", "low", "close", "volume"]
    for column in required:
        if column not in frame.columns:
            raise ValueError(f"Missing OHLCV column: {column}")
    return (
        frame.select(required)
        .with_columns(
            [
                pl.col("datetime").cast(pl.Datetime(time_unit="ms"), strict=False),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )
        .drop_nulls(subset=["datetime", "open", "high", "low", "close", "volume"])
        .sort("datetime")
        .unique(subset=["datetime"], keep="last")
    )


def resample_ohlcv_frame(frame: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    token = normalize_timeframe_token(timeframe)
    clean = _clean_ohlcv(frame)
    if clean.is_empty() or token == "1s":
        return clean

    tf_ms = int(timeframe_to_milliseconds(token))
    return (
        clean.lazy()
        .with_columns(pl.col("datetime").dt.epoch("ms").alias("ts_ms"))
        .with_columns(((pl.col("ts_ms") // tf_ms) * tf_ms).alias("bucket_ms"))
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
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .collect()
    )


def build_timeframe_panel_from_frames(
    base_frames: Mapping[str, pl.DataFrame],
    timeframes: list[str],
) -> dict[str, pl.DataFrame]:
    """Build aligned OHLCV panel from provided base-frequency frames."""
    panels: dict[str, pl.DataFrame] = {}
    for timeframe in [normalize_timeframe_token(item) for item in timeframes]:
        asset_frames: list[pl.DataFrame] = []
        for asset, frame in sorted(base_frames.items()):
            sampled = resample_ohlcv_frame(frame, timeframe)
            if sampled.is_empty():
                continue
            asset_frames.append(
                sampled.with_columns(
                    [
                        pl.lit(str(asset)).alias("asset"),
                        pl.lit(timeframe).alias("timeframe"),
                    ]
                )
            )
        panel = pl.concat(asset_frames, how="vertical") if asset_frames else pl.DataFrame()
        if not panel.is_empty():
            panel = panel.sort(["datetime", "asset"])
        panels[timeframe] = panel
    return panels


def build_timeframe_panel(
    *,
    market_data_root: str,
    exchange: str,
    assets: list[str],
    timeframes: list[str],
    start_date: str | datetime,
    end_date: str | datetime,
) -> dict[str, pl.DataFrame]:
    """Load OHLCV from parquet repository and build timeframe panels."""
    repo = ParquetMarketDataRepository(market_data_root)
    base_frames: dict[str, pl.DataFrame] = {}
    for asset in assets:
        frame = repo.load_ohlcv(
            exchange=exchange,
            symbol=asset,
            timeframe="1s",
            start_date=start_date,
            end_date=end_date,
        )
        base_frames[str(asset)] = frame
    return build_timeframe_panel_from_frames(base_frames=base_frames, timeframes=timeframes)


__all__ = [
    "build_timeframe_panel",
    "build_timeframe_panel_from_frames",
    "resample_ohlcv_frame",
]
