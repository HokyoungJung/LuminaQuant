#!/usr/bin/env python3
"""Fetch lagged external risk-state features for profit moonshot research.

The output is intentionally daily and lagged by one observation so it can be
joined to hourly crypto replay without same-day macro/market close lookahead.
It is a state filter only; locked-OOS remains report/gate-only in downstream
replay.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/external_market_state_20260512"
)
FRED_SERIES = {
    "DTWEXBGS": "usd_broad",
    "VIXCLS": "vix",
    "DGS2": "ust2y",
    "DGS10": "ust10y",
    "DCOILWTICO": "wti",
}
MAX_EXTERNAL_STATE_STALE_DAYS = 10


def _fred_csv_url(series_id: str, start: date, end: date) -> str:
    return (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        f"id={series_id}&cosd={start.isoformat()}&coed={end.isoformat()}"
    )


def _fetch_fred_series(
    series_id: str, start: date, end: date, *, timeout: float = 30.0
) -> pl.DataFrame:
    url = _fred_csv_url(series_id, start, end)
    with urlopen(url, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    frame = pl.read_csv(StringIO(raw), null_values=[".", ""])
    if "observation_date" not in frame.columns or series_id not in frame.columns:
        raise ValueError(f"unexpected FRED CSV schema for {series_id}: {frame.columns}")
    return frame.select(
        pl.col("observation_date").str.to_date().alias("date"),
        pl.col(series_id).cast(pl.Float64, strict=False).alias(FRED_SERIES[series_id]),
    )


def _rolling_z(expr: pl.Expr, window: int = 60) -> pl.Expr:
    mean = expr.rolling_mean(window_size=window, min_samples=max(10, window // 3))
    std = expr.rolling_std(window_size=window, min_samples=max(10, window // 3))
    return (expr - mean) / std


def _stale_bounded_daily_forward_fill(
    panel: pl.DataFrame,
    columns: list[str],
    *,
    max_stale_days: int = MAX_EXTERNAL_STATE_STALE_DAYS,
) -> pl.DataFrame:
    if panel.is_empty() or "date" not in panel.columns:
        return panel
    present_columns = [column for column in columns if column in panel.columns]
    if not present_columns:
        return panel

    day_col = "__date_day"
    source_cols = [f"__{column}_source_day" for column in present_columns]
    value_cols = [f"__{column}_ffill" for column in present_columns]
    bounded = (
        panel.with_columns(pl.col("date").dt.epoch("d").cast(pl.Int64).alias(day_col))
        .with_columns(
            [
                *[
                    pl.when(pl.col(column).is_not_null())
                    .then(pl.col(day_col))
                    .otherwise(None)
                    .cast(pl.Int64)
                    .forward_fill()
                    .alias(source_col)
                    for column, source_col in zip(present_columns, source_cols, strict=True)
                ],
                *[
                    pl.col(column).cast(pl.Float64).forward_fill().alias(value_col)
                    for column, value_col in zip(present_columns, value_cols, strict=True)
                ],
            ]
        )
        .with_columns(
            [
                pl.when(
                    pl.col(source_col).is_not_null()
                    & ((pl.col(day_col) - pl.col(source_col)) <= int(max_stale_days))
                )
                .then(pl.col(value_col))
                .otherwise(None)
                .alias(column)
                for column, source_col, value_col in zip(
                    present_columns,
                    source_cols,
                    value_cols,
                    strict=True,
                )
            ]
        )
    )
    return bounded.drop([day_col, *source_cols, *value_cols])


def _build_external_state_panel(series_frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not series_frames:
        return pl.DataFrame(schema={"date": pl.Date})
    panel = series_frames[0]
    for frame in series_frames[1:]:
        panel = panel.join(frame, on="date", how="full", coalesce=True)
    panel = panel.sort("date")
    raw_columns = [column for column in panel.columns if column != "date"]
    panel = _stale_bounded_daily_forward_fill(panel, raw_columns)
    panel = panel.with_columns(
        [
            pl.col("usd_broad").pct_change(5).alias("usd_ret_5d"),
            pl.col("vix").pct_change(5).alias("vix_ret_5d"),
            (pl.col("ust10y") - pl.col("ust2y")).alias("curve_10y2y"),
            pl.col("wti").pct_change(5).alias("wti_ret_5d"),
        ]
    )
    panel = panel.with_columns(
        [
            _rolling_z(pl.col("usd_ret_5d")).alias("external_usd_ret_z"),
            _rolling_z(pl.col("vix")).alias("external_vix_z"),
            _rolling_z(pl.col("curve_10y2y")).alias("external_curve_z"),
            _rolling_z(pl.col("wti_ret_5d")).alias("external_wti_ret_z"),
        ]
    )
    panel = panel.with_columns(
        (
            pl.col("external_vix_z").fill_null(0.0)
            + pl.col("external_usd_ret_z").fill_null(0.0)
            - 0.50 * pl.col("external_wti_ret_z").fill_null(0.0)
            - 0.25 * pl.col("external_curve_z").fill_null(0.0)
        ).alias("external_risk_off_score")
    )
    lag_columns = [column for column in panel.columns if column.startswith("external_")]
    panel = panel.with_columns(
        [pl.col(column).shift(1).alias(f"{column}_lag1") for column in lag_columns]
    )
    keep_columns = ["date", *lag_columns, *(f"{column}_lag1" for column in lag_columns)]
    return panel.select(keep_columns).rename({"date": "effective_date"})


def build_payload(args: argparse.Namespace) -> tuple[dict[str, object], pl.DataFrame]:
    start = datetime.fromisoformat(str(args.start_date)).date()
    end = datetime.fromisoformat(str(args.end_date)).date()
    frames: list[pl.DataFrame] = []
    series_metadata: dict[str, object] = {}
    for series_id in [item.strip().upper() for item in str(args.series).split(",") if item.strip()]:
        if series_id not in FRED_SERIES:
            raise ValueError(
                f"unsupported FRED series {series_id}; supported={sorted(FRED_SERIES)}"
            )
        frame = _fetch_fred_series(series_id, start, end)
        frames.append(frame)
        series_metadata[series_id] = {
            "alias": FRED_SERIES[series_id],
            "url": _fred_csv_url(series_id, start, end),
            "rows": int(frame.height),
            "first_date": frame["date"][0].isoformat() if frame.height else None,
            "last_date": frame["date"][-1].isoformat() if frame.height else None,
        }
    panel = _build_external_state_panel(frames)
    payload: dict[str, object] = {
        "artifact_kind": "profit_moonshot_external_market_state",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": "FRED fredgraph.csv",
        "series": series_metadata,
        "lag_policy": "All external_*_lag1 fields are shifted by one daily observation before hourly replay join.",
        "rows": int(panel.height),
        "first_effective_date": panel["effective_date"][0].isoformat() if panel.height else None,
        "last_effective_date": panel["effective_date"][-1].isoformat() if panel.height else None,
        "feature_columns": [column for column in panel.columns if column.startswith("external_")],
    }
    return payload, panel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-05-08")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--series", default=",".join(FRED_SERIES))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, panel = build_payload(args)
    csv_path = output_dir / "external_market_state_lagged.csv"
    json_path = output_dir / "external_market_state_lagged.json"
    panel.write_csv(csv_path)
    payload["csv_path"] = str(csv_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"csv": str(csv_path), "json": str(json_path), "rows": panel.height}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
