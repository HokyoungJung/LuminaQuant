"""Exact Polars reproduction of dacapogo's daily research model."""

from __future__ import annotations

from typing import Any

import polars as pl

from lumina_quant.core.plugin_registry import register
from lumina_quant.strategies.plugin_interface import StrategyPlugin

FEE = 0.0005
SLIP = 0.0005
COST = 2 * FEE + SLIP
TP = 0.008
SL = 0.005
TOPK = 10

_REQUIRED = {"market", "date", "value", "open", "high", "close"}
_NUMERIC = {"value", "open", "high", "close"}


def _validate(data: pl.DataFrame) -> None:
    missing = sorted(_REQUIRED.difference(data.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    non_numeric = sorted(name for name in _NUMERIC if not data.schema[name].is_numeric())
    if non_numeric:
        raise TypeError(f"columns must be numeric: {', '.join(non_numeric)}")


def daily_candidates(
    data: pl.DataFrame,
    *,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pl.DataFrame:
    """Select each date's top ten markets by strictly previous-day value."""
    _validate(data)
    candidates = data.sort(["market", "date"]).with_columns(
        pl.col("value").shift(1).over("market").alias("prev_value")
    )
    valid = (
        (pl.col("prev_value") > 0)
        & pl.col("prev_value").is_finite()
        & (pl.col("open") > 0)
        & (pl.col("high") >= pl.col("open"))
        & pl.col("open").is_finite()
        & pl.col("high").is_finite()
        & pl.col("close").is_finite()
    )
    if start_date is not None:
        valid &= pl.col("date") >= pl.lit(start_date)
    if end_date is not None:
        valid &= pl.col("date") <= pl.lit(end_date)
    return (
        candidates.filter(valid)
        .with_columns(pl.col("prev_value").rank(descending=True).over("date").alias("rk"))
        .filter(pl.col("rk") <= TOPK)
    )


def _triggered_returns(candidates: pl.DataFrame) -> pl.DataFrame:
    trades = candidates.with_columns((pl.col("open") * 1.04).alias("entry")).filter(
        pl.col("high") >= pl.col("entry")
    )
    hit_tp = pl.col("high") >= pl.col("entry") * (1 + TP)
    close_below = pl.col("close") <= pl.col("entry") * (1 - SL)
    ret_close = pl.col("close") / pl.col("entry") - 1
    return trades.with_columns(
        (pl.when(hit_tp).then(TP).when(close_below).then(-SL).otherwise(ret_close) - COST).alias(
            "ret_opt"
        ),
        (pl.when(close_below).then(-SL).otherwise(ret_close) - COST).alias("ret_pess"),
    )


def _daily_returns(trades: pl.DataFrame) -> pl.DataFrame:
    return (
        trades.group_by("date", maintain_order=True)
        .agg(
            (pl.col("ret_pess").sum() / TOPK).alias("ret_pess"),
            (pl.col("ret_opt").sum() / TOPK).alias("ret_opt"),
        )
        .sort("date")
    )


def backtest_daily(
    data: pl.DataFrame,
    *,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return triggered trades and fixed-ten-slot daily returns."""
    trades = _triggered_returns(daily_candidates(data, start_date=start_date, end_date=end_date))
    return trades, _daily_returns(trades)


@register("strategy", "DacapogoDailySourceStrategy", interface="polars_batch")
class DacapogoDailySourceStrategy(StrategyPlugin):
    """Dedicated research wrapper around the exact daily source model."""

    required_timeframes = ("1d",)
    decision_cadence_seconds = 86_400
    required_features = ("market", "date", "value", "open", "high", "close")
    runner_kind = "dedicated_dacapogo_daily_research"

    def compute_features(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        return daily_candidates(
            data,
            start_date=params.get("start_date"),
            end_date=params.get("end_date"),
        )

    def compute_signal(self, features: pl.DataFrame, params: dict) -> pl.DataFrame:
        _ = params
        return _triggered_returns(features)

    def signal_to_targets(self, raw_signal: pl.DataFrame, params: dict) -> pl.DataFrame:
        _ = params
        return _daily_returns(raw_signal)

    def run(
        self, data: pl.DataFrame, params: dict | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        features = self.compute_features(data, params or {})
        trades = self.compute_signal(features, params or {})
        return trades, self.signal_to_targets(trades, params or {})


__all__ = [
    "COST",
    "FEE",
    "SL",
    "SLIP",
    "TOPK",
    "TP",
    "DacapogoDailySourceStrategy",
    "backtest_daily",
    "daily_candidates",
]
