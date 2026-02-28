"""Cross-sectional mean-reversion plugin."""

from __future__ import annotations

import polars as pl
from lumina_quant.strategies.plugin_interface import StrategyPlugin, register_plugin


@register_plugin("xs_mean_reversion")
class CrossSectionalMeanReversionPlugin(StrategyPlugin):
    def compute_features(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        lookback = max(2, int(params.get("lookback_bars", 16)))
        by_keys = ["asset"]
        if "timeframe" in data.columns:
            by_keys.append("timeframe")
        sorted_data = data.sort([*by_keys, "datetime"])
        mean_col = pl.col("close").rolling_mean(window_size=lookback, min_samples=2).over(by_keys)
        std_col = (
            pl.col("close")
            .rolling_std(window_size=lookback, min_samples=2)
            .over(by_keys)
            .fill_null(1.0)
            .clip(1e-9, None)
        )
        return sorted_data.with_columns(
            [
                mean_col.alias("rolling_mean"),
                std_col.alias("rolling_std"),
            ]
        ).with_columns(((pl.col("close") - pl.col("rolling_mean")) / pl.col("rolling_std")).fill_null(0.0).alias("zscore"))

    def compute_signal(self, features: pl.DataFrame, params: dict) -> pl.DataFrame:
        _ = params
        return features.with_columns((-pl.col("zscore")).alias("signal"))

    def signal_to_targets(self, raw_signal: pl.DataFrame, params: dict) -> pl.DataFrame:
        top_n = max(1, int(params.get("top_n", 2)))
        bottom_n = max(1, int(params.get("bottom_n", 2)))
        allow_short = bool(params.get("allow_short", True))

        ranked = raw_signal.with_columns(
            [
                pl.col("signal").rank(method="ordinal", descending=True).over("datetime").alias("rank_desc"),
                pl.col("signal").rank(method="ordinal", descending=False).over("datetime").alias("rank_asc"),
            ]
        )

        long_weight = 1.0 / float(top_n)
        short_weight = -1.0 / float(bottom_n)

        return ranked.with_columns(
            pl.when(pl.col("rank_desc") <= top_n)
            .then(long_weight)
            .when((pl.col("rank_asc") <= bottom_n) & allow_short)
            .then(short_weight)
            .otherwise(0.0)
            .alias("target_weight")
        ).select([column for column in raw_signal.columns if column != "signal"] + ["signal", "target_weight"])
