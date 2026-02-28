"""Cross-sectional trend/momentum plugin."""

from __future__ import annotations

import polars as pl
from lumina_quant.strategies.plugin_interface import StrategyPlugin, register_plugin


@register_plugin("trend_momentum")
class TrendMomentumPlugin(StrategyPlugin):
    def compute_features(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        lookback = max(1, int(params.get("lookback_bars", 16)))
        by_keys = ["asset"]
        if "timeframe" in data.columns:
            by_keys.append("timeframe")
        return (
            data.sort([*by_keys, "datetime"])
            .with_columns(
                (
                    (pl.col("close") / pl.col("close").shift(lookback).over(by_keys)) - 1.0
                )
                .fill_null(0.0)
                .alias("momentum")
            )
        )

    def compute_signal(self, features: pl.DataFrame, params: dict) -> pl.DataFrame:
        _ = params
        return features.with_columns(pl.col("momentum").alias("signal"))

    def signal_to_targets(self, raw_signal: pl.DataFrame, params: dict) -> pl.DataFrame:
        top_n = max(1, int(params.get("top_n", 2)))
        bottom_n = max(0, int(params.get("bottom_n", 0)))
        allow_short = bool(params.get("allow_short", bottom_n > 0))

        ranked = raw_signal.with_columns(
            [
                pl.col("signal").rank(method="ordinal", descending=True).over("datetime").alias("rank_desc"),
                pl.col("signal").rank(method="ordinal", descending=False).over("datetime").alias("rank_asc"),
            ]
        )

        long_weight = 1.0 / float(top_n)
        short_weight = -1.0 / float(bottom_n) if allow_short and bottom_n > 0 else 0.0

        return ranked.with_columns(
            pl.when(pl.col("rank_desc") <= top_n)
            .then(long_weight)
            .when((pl.col("rank_asc") <= bottom_n) & allow_short)
            .then(short_weight)
            .otherwise(0.0)
            .alias("target_weight")
        ).select([column for column in raw_signal.columns if column != "signal"] + ["signal", "target_weight"])
