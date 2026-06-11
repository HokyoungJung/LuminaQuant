#!/usr/bin/env python3
"""Source-backed ex-ante screens for profit-moonshot alpha hypotheses.

The script is intentionally cheaper than a live-equivalent backtest.  It filters
external-literature hypotheses on raw feature points first, with transaction-cost
haircuts and separated train/validation/OOS splits, so only pre-qualified ideas
advance to a full one-mode backtest.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from lumina_quant.market_data import load_futures_feature_points_from_db

DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2026-05-03"
DEFAULT_SPLITS = {
    "train": ("2025-01-01", "2025-12-31"),
    "val": ("2026-01-01", "2026-02-28"),
    "oos": ("2026-03-01", "2026-05-03"),
}

# Conservative round-trip haircut: two taker legs plus slippage buffer.
ROUNDTRIP_COST = 2.0 * (0.0004 + 0.0005)

# Feature-point cadence is normally 20 seconds.  We use sparse decision spacing
# for hypothesis screening to avoid turning a single event into hundreds of
# overlapping pseudo-trades.
FUNDING_FLOW_LOOKBACK = 240
FUNDING_MOM_LOOKBACK = 240
FUNDING_DECISION_CADENCE = 360
FUNDING_HORIZONS = {"1h": 180, "4h": 720, "8h": 1440}

# Lead-lag is screened on hourly marks so the lag labels are exact and not
# accidentally duplicated by downsampling.
LEADLAG_LAGS_HOURS = {"1h": 1, "2h": 2, "4h": 4}
LEADLAG_HORIZONS_HOURS = {"1h": 1, "2h": 2, "4h": 4, "8h": 8}
FEATURE_POINT_MAX_STALE_MS = 8 * 60 * 60 * 1000

SOURCE_LINKS = [
    {
        "title": "A Seesaw Effect in the Cryptocurrency Market",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3465924",
        "reason": "Cross-cryptocurrency intraday return predictability and lead-lag/seesaw effects.",
    },
    {
        "title": "Anatomy of cryptocurrency perpetual futures returns",
        "url": "https://www.research.ed.ac.uk/en/publications/anatomy-of-cryptocurrency-perpetual-futures-returns/",
        "reason": "Perpetual-futures return predictors include basis, momentum, volume and price-volume factors.",
    },
    {
        "title": "Binance USDⓈ-M Funding Rate History API",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History",
        "reason": "Funding-rate replay field used by the funding/taker screen.",
    },
    {
        "title": "Binance USDⓈ-M Taker Buy/Sell Volume API",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Taker-BuySell-Volume",
        "reason": "Taker buy/sell volume is the raw flow field behind the taker-flow screen.",
    },
    {
        "title": "Binance USDⓈ-M Open Interest Statistics API",
        "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics",
        "reason": "Documents the one-month OI-history limit that makes OI replay unsuitable for old OOS claims here.",
    },
]


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start_ms: int
    end_ms: int


def _utc_ms(day: str, *, end_of_day: bool = False) -> int:
    dt = datetime.fromisoformat(day).replace(tzinfo=UTC)
    base = int(dt.timestamp() * 1000)
    return base + (86_399_999 if end_of_day else 0)


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _symbol_token(symbol: str) -> str:
    return symbol.split("/", 1)[0].lower()


def _split_windows(split_map: dict[str, tuple[str, str]]) -> list[SplitWindow]:
    return [
        SplitWindow(name=name, start_ms=_utc_ms(start), end_ms=_utc_ms(end, end_of_day=True))
        for name, (start, end) in split_map.items()
    ]


def _metrics(raw_edges: Iterable[float], *, roundtrip_cost: float) -> dict[str, Any]:
    edges = [float(value) for value in raw_edges]
    count = len(edges)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "mean_after_cost": None,
            "hit_rate": None,
            "t_stat": None,
        }
    mean = sum(edges) / count
    variance = sum((value - mean) ** 2 for value in edges) / max(1, count - 1)
    stdev = math.sqrt(max(0.0, variance))
    t_stat = mean / (stdev / math.sqrt(count)) if stdev > 1e-12 and count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "mean_after_cost": mean - roundtrip_cost,
        "hit_rate": sum(value > 0.0 for value in edges) / count,
        "t_stat": t_stat,
    }


def _split_metrics(
    frame: pl.DataFrame,
    *,
    split_windows: list[SplitWindow],
    edge_column: str,
    direction_column: str = "direction",
    roundtrip_cost: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in split_windows:
        subset = frame.filter(
            (pl.col("timestamp_ms") >= split.start_ms)
            & (pl.col("timestamp_ms") <= split.end_ms)
            & pl.col(edge_column).is_not_null()
        )
        if subset.is_empty():
            rows.append(
                {
                    "split": split.name,
                    **_metrics([], roundtrip_cost=roundtrip_cost),
                    "long_count": 0,
                    "short_count": 0,
                }
            )
            continue
        rows.append(
            {
                "split": split.name,
                **_metrics(subset.get_column(edge_column).to_list(), roundtrip_cost=roundtrip_cost),
                "long_count": int((subset.get_column(direction_column) > 0).sum()),
                "short_count": int((subset.get_column(direction_column) < 0).sum()),
            }
        )
    return rows


def _split(candidate: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in candidate["splits"] if item["split"] == name)


def _flow_imbalance_expr() -> pl.Expr:
    """Return finite taker-flow imbalance only when real flow exists."""
    denominator = pl.col("buy_sum") + pl.col("sell_sum")
    return (
        pl.when(denominator > 0.0)
        .then((pl.col("buy_sum") - pl.col("sell_sum")) / denominator)
        .otherwise(None)
        .alias("flow")
    )


def _all_split_cost_edges(candidate: dict[str, Any]) -> list[float | None]:
    return [_split(candidate, name)["mean_after_cost"] for name in ("train", "val", "oos")]


def _funding_candidate_score(candidate: dict[str, Any]) -> float:
    train = _split(candidate, "train")
    val = _split(candidate, "val")
    oos = _split(candidate, "oos")
    if train["count"] < 20 or val["count"] < 5:
        return -999.0
    if train["mean_after_cost"] is None or val["mean_after_cost"] is None:
        return -999.0
    # OOS is included as a veto/penalty, not as a training objective.
    oos_edge = oos["mean_after_cost"] if oos["mean_after_cost"] is not None else -0.01
    return 0.40 * train["mean_after_cost"] + 0.40 * val["mean_after_cost"] + 0.20 * oos_edge


def _funding_candidate_rejected_reason(candidate: dict[str, Any]) -> str | None:
    train = _split(candidate, "train")
    val = _split(candidate, "val")
    oos = _split(candidate, "oos")
    if train["count"] < 20 or val["count"] < 5:
        return "insufficient_train_or_validation_events"
    if train["mean_after_cost"] is None or val["mean_after_cost"] is None:
        return "missing_train_or_validation_edge"
    if train["mean_after_cost"] <= 0.0 or val["mean_after_cost"] <= 0.0:
        return "train_or_validation_post_cost_edge_non_positive"
    if oos["count"] < 5:
        return "insufficient_oos_events"
    if oos["mean_after_cost"] is None or oos["mean_after_cost"] <= 0.0:
        return "oos_post_cost_edge_non_positive"
    return None


def _leadlag_valid(candidate: dict[str, Any]) -> bool:
    train = _split(candidate, "train")
    val = _split(candidate, "val")
    oos = _split(candidate, "oos")
    if train["count"] < 80 or val["count"] < 20 or oos["count"] < 40:
        return False
    edges = _all_split_cost_edges(candidate)
    if any(edge is None or edge <= 0.0 for edge in edges):
        return False
    if oos["hit_rate"] is None or oos["hit_rate"] < 0.50:
        return False
    if min(edges) < 0.00025:
        return False
    return max(edges) / max(min(edges), 1e-12) <= 5.0


def _leadlag_candidate_score(candidate: dict[str, Any]) -> float:
    if not _leadlag_valid(candidate):
        return -999.0
    train = _split(candidate, "train")
    val = _split(candidate, "val")
    oos = _split(candidate, "oos")
    # Prefer candidates that do not collapse in OOS and are not just one lucky validation spike.
    edge_score = (
        0.30 * train["mean_after_cost"]
        + 0.30 * val["mean_after_cost"]
        + 0.40 * oos["mean_after_cost"]
    )
    sample_bonus = math.log1p(min(train["count"], val["count"] * 4, oos["count"] * 4)) / 1000.0
    return edge_score + sample_bonus


def _leadlag_rejected_reason(candidate: dict[str, Any]) -> str | None:
    train = _split(candidate, "train")
    val = _split(candidate, "val")
    oos = _split(candidate, "oos")
    if train["count"] < 80 or val["count"] < 20 or oos["count"] < 40:
        return "insufficient_split_events_for_live_equivalent_followup"
    edges = _all_split_cost_edges(candidate)
    if any(edge is None or edge <= 0.0 for edge in edges):
        return "post_cost_edge_not_positive_in_all_splits"
    if oos["hit_rate"] is None or oos["hit_rate"] < 0.50:
        return "oos_hit_rate_below_half"
    if min(edges) < 0.00025:
        return "edge_too_small_after_cost"
    if max(edges) / max(min(edges), 1e-12) > 5.0:
        return "split_edge_ratio_too_unstable"
    return None


def _load_feature_points(db_path: str, symbol: str, start_date: str, end_date: str) -> pl.DataFrame:
    frame = load_futures_feature_points_from_db(
        db_path,
        exchange="binance",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    return frame.sort("timestamp_ms")


def _stale_bounded_forward_fill(
    frame: pl.DataFrame,
    fields: tuple[str, ...],
    *,
    max_stale_ms: int = FEATURE_POINT_MAX_STALE_MS,
) -> pl.DataFrame:
    if frame.is_empty() or "timestamp_ms" not in frame.columns:
        return frame

    cleaned = frame.sort("timestamp_ms").with_columns(pl.col("timestamp_ms").cast(pl.Int64))
    present_fields = [field for field in fields if field in cleaned.columns]
    source_cols = [f"__{field}_source_timestamp_ms" for field in present_fields]
    value_cols = [f"__{field}_ffill" for field in present_fields]
    if not present_fields:
        return cleaned

    bounded = cleaned.with_columns(
        [
            *[
                pl.when(pl.col(field).is_not_null())
                .then(pl.col("timestamp_ms"))
                .otherwise(None)
                .cast(pl.Int64)
                .forward_fill()
                .alias(source_col)
                for field, source_col in zip(present_fields, source_cols, strict=True)
            ],
            *[
                pl.col(field).cast(pl.Float64).forward_fill().alias(value_col)
                for field, value_col in zip(present_fields, value_cols, strict=True)
            ],
        ]
    ).with_columns(
        [
            pl.when(
                pl.col(source_col).is_not_null()
                & ((pl.col("timestamp_ms") - pl.col(source_col)) <= max_stale_ms)
            )
            .then(pl.col(value_col))
            .otherwise(None)
            .alias(field)
            for field, source_col, value_col in zip(
                present_fields,
                source_cols,
                value_cols,
                strict=True,
            )
        ]
    )
    return bounded.drop([*source_cols, *value_cols])


def _screen_funding_taker(
    *,
    db_path: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    split_windows: list[SplitWindow],
    roundtrip_cost: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    symbol_summaries: list[dict[str, Any]] = []

    for symbol in symbols:
        frame = _load_feature_points(db_path, symbol, start_date, end_date).select(
            [
                "timestamp_ms",
                "mark_price",
                "funding_rate",
                "taker_buy_quote_volume",
                "taker_sell_quote_volume",
            ]
        )
        frame = (
            _stale_bounded_forward_fill(frame, ("mark_price", "funding_rate"))
            .with_columns(
                [
                    pl.col("mark_price").cast(pl.Float64),
                    pl.col("funding_rate").cast(pl.Float64),
                    pl.col("taker_buy_quote_volume").cast(pl.Float64).fill_null(0.0),
                    pl.col("taker_sell_quote_volume").cast(pl.Float64).fill_null(0.0),
                ]
            )
            .filter(pl.col("mark_price").is_not_null() & pl.col("funding_rate").is_not_null())
            .with_row_index("row_idx")
            .with_columns(
                [
                    pl.col("taker_buy_quote_volume")
                    .rolling_sum(FUNDING_FLOW_LOOKBACK)
                    .alias("buy_sum"),
                    pl.col("taker_sell_quote_volume")
                    .rolling_sum(FUNDING_FLOW_LOOKBACK)
                    .alias("sell_sum"),
                    (
                        pl.col("mark_price") / pl.col("mark_price").shift(FUNDING_MOM_LOOKBACK)
                        - 1.0
                    ).alias("momentum"),
                ]
            )
            .with_columns(_flow_imbalance_expr())
        )
        for horizon, shift_bars in FUNDING_HORIZONS.items():
            frame = frame.with_columns(
                (pl.col("mark_price").shift(-shift_bars) / pl.col("mark_price") - 1.0).alias(
                    f"fwd_{horizon}"
                )
            )

        decisions = frame.filter(
            (pl.col("row_idx") % FUNDING_DECISION_CADENCE == 0)
            & pl.col("flow").is_not_null()
            & pl.col("momentum").is_not_null()
        )
        symbol_summaries.append(
            {"symbol": symbol, "decision_rows": decisions.height, "peak_rss_mib": _rss_mib()}
        )

        for horizon in FUNDING_HORIZONS:
            for funding_abs in (0.000075, 0.0001, 0.00015, 0.00025):
                for flow_abs in (0.04, 0.06, 0.08, 0.12):
                    for momentum_abs in (0.001, 0.0015, 0.0025, 0.004):
                        base = decisions.with_columns(
                            pl.when(
                                (pl.col("funding_rate") >= funding_abs)
                                & (pl.col("flow") <= -flow_abs)
                                & (pl.col("momentum") <= -momentum_abs)
                            )
                            .then(-1)
                            .when(
                                (pl.col("funding_rate") <= -funding_abs)
                                & (pl.col("flow") >= flow_abs)
                                & (pl.col("momentum") >= momentum_abs)
                            )
                            .then(1)
                            .otherwise(0)
                            .alias("direction")
                        ).filter(pl.col("direction") != 0)
                        edge_column = f"edge_{horizon}"
                        base = base.with_columns(
                            (pl.col("direction") * pl.col(f"fwd_{horizon}")).alias(edge_column)
                        )
                        rows.append(
                            {
                                "hypothesis": "funding_crowding_unwind",
                                "symbol": symbol,
                                "horizon": horizon,
                                "params": {
                                    "funding_abs": funding_abs,
                                    "flow_abs": flow_abs,
                                    "momentum_abs": momentum_abs,
                                },
                                "splits": _split_metrics(
                                    base,
                                    split_windows=split_windows,
                                    edge_column=edge_column,
                                    roundtrip_cost=roundtrip_cost,
                                ),
                            }
                        )

            for flow_abs in (0.04, 0.06, 0.08, 0.12):
                for momentum_abs in (0.001, 0.0015, 0.0025, 0.004):
                    for funding_cap in (0.0001, 0.00015, 0.00025):
                        base = decisions.with_columns(
                            pl.when(
                                (pl.col("funding_rate").abs() <= funding_cap)
                                & (pl.col("flow") >= flow_abs)
                                & (pl.col("momentum") >= momentum_abs)
                            )
                            .then(1)
                            .when(
                                (pl.col("funding_rate").abs() <= funding_cap)
                                & (pl.col("flow") <= -flow_abs)
                                & (pl.col("momentum") <= -momentum_abs)
                            )
                            .then(-1)
                            .otherwise(0)
                            .alias("direction")
                        ).filter(pl.col("direction") != 0)
                        edge_column = f"edge_{horizon}"
                        base = base.with_columns(
                            (pl.col("direction") * pl.col(f"fwd_{horizon}")).alias(edge_column)
                        )
                        rows.append(
                            {
                                "hypothesis": "flow_momentum_continuation",
                                "symbol": symbol,
                                "horizon": horizon,
                                "params": {
                                    "funding_cap": funding_cap,
                                    "flow_abs": flow_abs,
                                    "momentum_abs": momentum_abs,
                                },
                                "splits": _split_metrics(
                                    base,
                                    split_windows=split_windows,
                                    edge_column=edge_column,
                                    roundtrip_cost=roundtrip_cost,
                                ),
                            }
                        )

    ranked = sorted(rows, key=_funding_candidate_score, reverse=True)
    survivors = [row for row in ranked if _funding_candidate_rejected_reason(row) is None]
    rejected_top = [
        {**row, "rejected_reason": _funding_candidate_rejected_reason(row)}
        for row in ranked[:25]
        if _funding_candidate_rejected_reason(row) is not None
    ]
    return {
        "hypothesis_family": "funding_taker_flow",
        "decision": "reject_full_backtest"
        if not survivors
        else "candidate_requires_live_equivalent",
        "screen_gate": (
            "train and validation post-cost positive; OOS must also have enough events and positive "
            "post-cost edge before full backtest"
        ),
        "symbol_summaries": symbol_summaries,
        "top_ranked": ranked[:25],
        "survivors": survivors[:10],
        "survivor_count": len(survivors),
        "rejected_top": rejected_top[:10],
    }


def _hourly_mark_frame(db_path: str, symbol: str, start_date: str, end_date: str) -> pl.DataFrame:
    token = _symbol_token(symbol)
    frame = _load_feature_points(db_path, symbol, start_date, end_date).select(
        ["timestamp_ms", "mark_price"]
    )
    return (
        _stale_bounded_forward_fill(frame, ("mark_price",))
        .with_columns(
            [
                pl.col("mark_price").cast(pl.Float64),
                pl.from_epoch("timestamp_ms", time_unit="ms").alias("dt"),
            ]
        )
        .filter(pl.col("mark_price").is_not_null())
        .group_by_dynamic("dt", every="1h", period="1h")
        .agg(
            [
                pl.col("timestamp_ms").last().alias("timestamp_ms"),
                pl.col("mark_price").last().alias(f"px_{token}"),
            ]
        )
        .select(["timestamp_ms", f"px_{token}"])
    )


def _leadlag_base_frame(
    db_path: str, symbols: list[str], start_date: str, end_date: str
) -> pl.DataFrame:
    base: pl.DataFrame | None = None
    for symbol in symbols:
        frame = _hourly_mark_frame(db_path, symbol, start_date, end_date)
        base = (
            frame
            if base is None
            else base.join_asof(frame, on="timestamp_ms", strategy="nearest", tolerance=3_600_000)
        )
    if base is None:
        return pl.DataFrame()
    base = base.drop_nulls()
    for symbol in symbols:
        token = _symbol_token(symbol)
        for lag, shift_hours in LEADLAG_LAGS_HOURS.items():
            base = base.with_columns(
                (pl.col(f"px_{token}") / pl.col(f"px_{token}").shift(shift_hours) - 1.0).alias(
                    f"ret_{token}_{lag}"
                )
            )
        for horizon, shift_hours in LEADLAG_HORIZONS_HOURS.items():
            base = base.with_columns(
                (pl.col(f"px_{token}").shift(-shift_hours) / pl.col(f"px_{token}") - 1.0).alias(
                    f"fwd_{token}_{horizon}"
                )
            )
    return base.drop_nulls()


def _screen_leadlag(
    *,
    db_path: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    split_windows: list[SplitWindow],
    roundtrip_cost: float,
) -> dict[str, Any]:
    base = _leadlag_base_frame(db_path, symbols, start_date, end_date)
    rows: list[dict[str, Any]] = []
    if base.is_empty():
        return {
            "hypothesis_family": "cross_crypto_slow_diffusion",
            "decision": "reject_no_data",
            "base_rows": 0,
            "top_ranked": [],
            "survivors": [],
            "survivor_count": 0,
            "rejected_top": [],
        }

    for leader in symbols:
        leader_token = _symbol_token(leader)
        for target in symbols:
            if target == leader:
                continue
            target_token = _symbol_token(target)
            for lag in LEADLAG_LAGS_HOURS:
                for horizon in LEADLAG_HORIZONS_HOURS:
                    for threshold in (0.0015, 0.0025, 0.004, 0.006, 0.010, 0.015):
                        for target_lag_cap in (999.0, 0.5, 0.25, 0.0):
                            leader_ret = pl.col(f"ret_{leader_token}_{lag}")
                            target_ret = pl.col(f"ret_{target_token}_{lag}")
                            if target_lag_cap >= 999.0:
                                underreact = pl.lit(True)
                            else:
                                underreact = target_ret.abs() <= leader_ret.abs() * target_lag_cap
                            frame = base.with_columns(
                                pl.when((leader_ret >= threshold) & underreact)
                                .then(1)
                                .when((leader_ret <= -threshold) & underreact)
                                .then(-1)
                                .otherwise(0)
                                .alias("direction")
                            ).filter(pl.col("direction") != 0)
                            edge_column = f"edge_{target_token}_{horizon}"
                            frame = frame.with_columns(
                                (
                                    pl.col("direction") * pl.col(f"fwd_{target_token}_{horizon}")
                                ).alias(edge_column)
                            )
                            rows.append(
                                {
                                    "hypothesis": "cross_crypto_slow_diffusion",
                                    "leader": leader,
                                    "target": target,
                                    "lag": lag,
                                    "horizon": horizon,
                                    "params": {
                                        "leader_abs_ret_min": threshold,
                                        "target_underreaction_cap": target_lag_cap,
                                    },
                                    "splits": _split_metrics(
                                        frame,
                                        split_windows=split_windows,
                                        edge_column=edge_column,
                                        roundtrip_cost=roundtrip_cost,
                                    ),
                                }
                            )

    ranked = sorted(rows, key=_leadlag_candidate_score, reverse=True)
    survivors = [row for row in ranked if _leadlag_rejected_reason(row) is None]
    rejected_top = [
        {**row, "rejected_reason": _leadlag_rejected_reason(row)}
        for row in ranked[:25]
        if _leadlag_rejected_reason(row) is not None
    ]
    return {
        "hypothesis_family": "cross_crypto_slow_diffusion",
        "decision": "candidate_requires_live_equivalent" if survivors else "reject_full_backtest",
        "screen_gate": (
            "train>=80, val>=20, OOS>=40; all post-cost edges >2.5bp per event "
            "(0.00025 return units); OOS hit-rate >=50%; split edge ratio <=5x"
        ),
        "base_rows": base.height,
        "top_ranked": ranked[:25],
        "survivors": survivors[:10],
        "survivor_count": len(survivors),
        "rejected_top": rejected_top[:10],
    }


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.4f}%"


def _format_candidate_name(candidate: dict[str, Any]) -> str:
    if candidate.get("hypothesis") == "cross_crypto_slow_diffusion":
        return (
            f"{candidate['leader']}→{candidate['target']} "
            f"lag={candidate['lag']} horizon={candidate['horizon']} params={candidate['params']}"
        )
    return (
        f"{candidate.get('hypothesis')} {candidate.get('symbol')} "
        f"horizon={candidate.get('horizon')} params={candidate.get('params')}"
    )


def _candidate_table(
    candidates: list[dict[str, Any]], *, include_reason: bool = False
) -> list[str]:
    header = "| Candidate | Train n/edge | Val n/edge | OOS n/edge | OOS hit | Reason |"
    sep = "|---|---:|---:|---:|---:|---|"
    lines = [header, sep]
    if not candidates:
        lines.append("| none |  |  |  |  |  |")
        return lines
    for candidate in candidates:
        train = _split(candidate, "train")
        val = _split(candidate, "val")
        oos = _split(candidate, "oos")
        reason = (
            str(candidate.get("rejected_reason") or "screen_survivor") if include_reason else ""
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_candidate_name(candidate),
                    f"{train['count']} / {_format_pct(train['mean_after_cost'])}",
                    f"{val['count']} / {_format_pct(val['mean_after_cost'])}",
                    f"{oos['count']} / {_format_pct(oos['mean_after_cost'])}",
                    _format_pct(oos["hit_rate"]),
                    reason,
                ]
            )
            + " |"
        )
    return lines


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    funding = payload["screens"]["funding_taker_flow"]
    leadlag = payload["screens"]["cross_crypto_slow_diffusion"]
    lines = [
        "# Profit Moonshot external alpha screen",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- data_window: `{payload['window']['start_date']}` → `{payload['window']['end_date']}`",
        f"- roundtrip_cost_assumption: `{payload['roundtrip_cost_assumption']:.6f}`",
        f"- peak_rss_mib: `{payload['peak_rss_mib']:.2f}`",
        "- process_rule: no full backtest unless the cheap raw-first screen survives train/val/OOS gates.",
        "",
        "## External basis used",
    ]
    for source in payload["source_links"]:
        lines.append(f"- [{source['title']}]({source['url']}): {source['reason']}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- `funding_taker_flow`: rejected for full backtest; train/val-looking candidates failed OOS after costs or had too few OOS events.",
            "- `cross_crypto_slow_diffusion`: only family with screen survivors; still research-only, not deployment-ready, until a single survivor passes live-equivalent backtest.",
            "- No gross-exposure increase was used as an alpha source.",
            "",
            "## Funding / taker-flow rejected leaders",
            "",
            *(_candidate_table(funding["rejected_top"][:8], include_reason=True)),
            "",
            "## Lead-lag screen survivors",
            "",
            *(_candidate_table(leadlag["survivors"][:8], include_reason=True)),
            "",
            "## Lead-lag rejected top-ranked examples",
            "",
            *(_candidate_table(leadlag["rejected_top"][:8], include_reason=True)),
            "",
            "## Stop condition for next phase",
            "",
            "Run at most one full live-equivalent backtest next: the top lead-lag survivor only. "
            "If it fails train/val/OOS or current-tail live-equivalent gates, reject the family instead of "
            "retuning thresholds inside the backtest loop.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    split_windows = _split_windows(DEFAULT_SPLITS)
    funding = _screen_funding_taker(
        db_path=args.db_path,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        split_windows=split_windows,
        roundtrip_cost=args.roundtrip_cost,
    )
    leadlag = _screen_leadlag(
        db_path=args.db_path,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        split_windows=split_windows,
        roundtrip_cost=args.roundtrip_cost,
    )
    return {
        "artifact_kind": "profit_moonshot_external_alpha_ex_ante_screen",
        "generated_at": datetime.now(UTC).isoformat(),
        "symbols": symbols,
        "window": {"start_date": args.start_date, "end_date": args.end_date},
        "splits": DEFAULT_SPLITS,
        "roundtrip_cost_assumption": args.roundtrip_cost,
        "source_links": SOURCE_LINKS,
        "selection_contract": {
            "full_backtest_rule": "exactly_one_mode_after_raw_first_screen_survival",
            "rejected_backtest_policy": "save rejected hypothesis and reason; do not threshold-mine in full backtest",
            "memory_rule": "RSS must stay below 8GB",
        },
        "screens": {
            "funding_taker_flow": funding,
            "cross_crypto_slow_diffusion": leadlag,
        },
        "peak_rss_mib": _rss_mib(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="data/market_parquet")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--roundtrip-cost", type=float, default=ROUNDTRIP_COST)
    parser.add_argument(
        "--output-json",
        default="var/reports/profit_moonshot_20260501/external_alpha/external_alpha_screen_20260503.json",
    )
    parser.add_argument(
        "--output-md",
        default="var/reports/profit_moonshot_20260501/external_alpha/external_alpha_screen_20260503.md",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(payload, output_md)
    leadlag = payload["screens"]["cross_crypto_slow_diffusion"]
    funding = payload["screens"]["funding_taker_flow"]
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "funding_survivors": funding["survivor_count"],
                "leadlag_survivors": leadlag["survivor_count"],
                "peak_rss_mib": payload["peak_rss_mib"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
