"""Liquidity metric computations for cost-aware execution."""

from __future__ import annotations

import polars as pl


def compute_liquidity_metrics(panel: pl.DataFrame, rolling_window: int = 20) -> pl.DataFrame:
    """Compute deterministic rolling ADV/ADTV/sigma per asset+timeframe."""
    if panel.is_empty():
        return panel

    window = max(2, int(rolling_window))
    by_keys = ["asset"] + (["timeframe"] if "timeframe" in panel.columns else [])

    enriched = (
        panel.sort([*by_keys, "datetime"])
        .with_columns(
            [
                (pl.col("close") * pl.col("volume")).alias("notional_volume"),
                pl.col("close").pct_change().over(by_keys).fill_null(0.0).alias("ret"),
            ]
        )
        .with_columns(
            [
                pl.col("volume")
                .rolling_mean(window_size=window, min_samples=1)
                .over(by_keys)
                .alias("adv"),
                pl.col("notional_volume")
                .rolling_mean(window_size=window, min_samples=1)
                .over(by_keys)
                .alias("adtv"),
                pl.col("ret")
                .rolling_std(window_size=window, min_samples=2)
                .over(by_keys)
                .fill_null(0.0)
                .alias("sigma"),
            ]
        )
        .drop("ret")
    )
    return enriched


__all__ = ["compute_liquidity_metrics"]
