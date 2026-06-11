"""Learn a market-regime judgement for incumbent vs 55/45 using repaired feature coverage."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    MEMORY_GUARD_DIRNAME,
    ROOT,
    acquire_portfolio_memory_guard,
    resolve_current_optimization_path,
    resolve_followup_artifact_path,
    split_for_date,
)

SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT / "portfolio_incumbent_autoresearch_grouped" / "market_regime_judgement_current"
)
DEFAULT_HORIZON_DAYS = 5
DEFAULT_SOFT_RSS_BYTES = int(7.5 * 1024 * 1024 * 1024)
DEFAULT_HARD_RSS_BYTES = int(8.0 * 1024 * 1024 * 1024)
MIN_TRAIN_COUNT = 20
MIN_VAL_COUNT = 6
MIN_WIN_RATE = 0.50
MIXED_MARGIN_THRESHOLD = 0.20
MIN_COMBINED_COUNT = 40
MIN_COMBINED_EDGE = 0.0001
MIN_OOS_SANITY_COUNT = 5
OOS_NEUTRAL_BAND = 0.0005
CONFIRMATION_COMBINED_NEUTRAL_MAX = 0.0010
CONFIRMATION_OOS_EDGE_MIN = 0.0010
CONFIRMATION_SCORE_DISCOUNT = 0.35
MARKET_PARQUET_ROOT = ROOT / "data" / "market_parquet"
FEATURE_POINT_ROOT = MARKET_PARQUET_ROOT / "feature_points" / "exchange=binance"
MATERIALIZED_ROOT = MARKET_PARQUET_ROOT / "market_data_materialized" / "binance"

BOOLEAN_FEATURE_LABELS = {
    "btc_above_ma192": "BTC above 4-day trend",
    "btc_above_ma336": "BTC above 7-day trend",
    "btc_trend_gap_336_pos": "BTC long trend gap positive",
    "breadth_ma96_ge_60": "2-day breadth at or above 60%",
    "breadth_ma192_ge_60": "4-day breadth at or above 60%",
    "breadth_ma96_ge_40": "2-day breadth at or above 40%",
    "basket_ret96_pos": "basket 2-day momentum positive",
    "basket_ret96_top3_pos": "top-3 cross-asset 2-day return mean positive",
    "basket_vol_ratio_moderate": "basket vol ratio in 0.90-1.60 band",
    "btc_trend_accel_pos": "BTC short trend stronger than long trend",
    "breadth_expanding": "2-day breadth leading 4-day breadth by at least 10%",
    "basket_ret_dispersion_compressed": "cross-asset 2-day return dispersion compressed",
}

CONTINUOUS_FEATURE_LABELS = {
    "btc_trend_gap_192": "BTC / MA192 gap",
    "btc_trend_gap_336": "BTC / MA336 gap",
    "btc_trend_accel": "BTC trend acceleration (MA192 gap - MA336 gap)",
    "breadth_ma96": "2-day breadth share",
    "breadth_ma192": "4-day breadth share",
    "breadth_delta": "2-day breadth minus 4-day breadth",
    "basket_ret96": "basket 2-day return",
    "basket_ret96_top3_mean": "top-3 cross-asset 2-day return mean",
    "basket_ret96_dispersion": "cross-asset 2-day return dispersion",
    "basket_vol_ratio": "basket vol ratio",
}


@dataclass(slots=True)
class GroupPortfolio:
    label: str
    path: Path
    weights: list[dict[str, Any]]
    returns: pd.DataFrame
    symbols: list[str]


@dataclass(slots=True)
class RuleCandidate:
    rule_id: str
    label: str
    family: str
    feature_names: tuple[str, ...]
    threshold: float | None
    comparator: str | None
    polarity: str


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve_autoresearch_default_path() -> Path:
    candidate = resolve_followup_artifact_path(
        "var/reports/exact_window_backtests/followup_status/"
        "autoresearch_candidate_portfolio_opt/portfolio_optimization_latest.json"
    )
    if not candidate.exists():
        raise FileNotFoundError("unable to locate autoresearch 55/45 portfolio artifact")
    return candidate.resolve()


def _load_group_portfolio(*, label: str, path: Path) -> GroupPortfolio:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = list(payload.get("weights") or [])
    if not weights:
        raise ValueError(f"{label}: missing weights in {path}")
    frames: list[pd.DataFrame] = []
    for split, rows in dict(payload.get("portfolio_return_streams") or {}).items():
        frame = pd.DataFrame(list(rows))
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["t"], utc=True).dt.floor("D")
        frame["split"] = str(split)
        frame["return"] = frame["v"].astype(float)
        frames.append(frame[["date", "split", "return"]])
    if not frames:
        raise ValueError(f"{label}: missing portfolio_return_streams in {path}")
    returns = (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    symbols = sorted({str(symbol) for row in weights for symbol in list(row.get("symbols") or [])})
    return GroupPortfolio(
        label=label, path=path.resolve(), weights=weights, returns=returns, symbols=symbols
    )


def _forward_compound(series: pd.Series, horizon_days: int) -> pd.Series:
    values = np.asarray(series, dtype=float)
    out = np.full(values.size, np.nan, dtype=float)
    gross = 1.0 + values
    for index in range(max(0, values.size - horizon_days)):
        out[index] = float(np.prod(gross[index + 1 : index + 1 + horizon_days]) - 1.0)
    return pd.Series(out, index=series.index, dtype=float)


def _split_group(day_value: pd.Timestamp) -> str:
    return split_for_date(day_value.date())


def _materialized_date_coverage(symbol: str) -> dict[str, Any]:
    symbol_token = str(symbol).replace("/", "")
    timeframe_dir = MATERIALIZED_ROOT / symbol_token / "timeframe=30m"
    dates = (
        sorted(path.name.split("=", 1)[1] for path in timeframe_dir.glob("date=*"))
        if timeframe_dir.exists()
        else []
    )
    return {
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "day_count": len(dates),
        "dates": dates,
    }


def _feature_point_files(
    symbol: str, *, start_day: pd.Timestamp, end_day: pd.Timestamp
) -> list[str]:
    symbol_token = str(symbol).replace("/", "")
    root = FEATURE_POINT_ROOT / f"symbol={symbol_token}"
    if not root.exists():
        raise FileNotFoundError(f"feature_points missing for {symbol}: {root}")
    file_paths: list[str] = []
    for date_dir in sorted(root.glob("date=*")):
        day_value = pd.Timestamp(date_dir.name.split("=", 1)[1], tz="UTC")
        if day_value < start_day or day_value > end_day:
            continue
        file_paths.extend(str(path) for path in sorted(date_dir.glob("compact-*.parquet")))
    if not file_paths:
        raise FileNotFoundError(f"no feature_points files selected for {symbol}")
    return file_paths


def _load_symbol_close_30m_from_feature_points(
    symbol: str,
    *,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = _feature_point_files(symbol, start_day=start_day, end_day=end_day)
    lazy = (
        pl.scan_parquet(files)
        .with_columns(
            pl.col("datetime").str.to_datetime(strict=False, time_zone="UTC").alias("datetime_ts"),
            pl.coalesce([pl.col("mark_price"), pl.col("index_price")]).alias("price"),
        )
        .filter(pl.col("price").is_not_null())
        .select(["datetime_ts", "price"])
        .group_by_dynamic("datetime_ts", every="30m", closed="left")
        .agg(pl.col("price").last().alias("close"))
        .sort("datetime_ts")
    )
    collected = lazy.collect()
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(collected["datetime_ts"].to_list(), utc=True),
            "close": pd.Series(collected["close"].to_list(), dtype="float64"),
        }
    )
    frame = frame.dropna(subset=["datetime", "close"]).sort_values("datetime")
    if frame.empty:
        raise FileNotFoundError(f"no usable feature_point prices selected for {symbol}")
    first_observed = pd.Timestamp(frame["datetime"].min()).floor("30min")
    effective_start = max(start_day, first_observed)
    full_index = pd.date_range(
        start=effective_start,
        end=(end_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)),
        freq="30min",
        tz="UTC",
    )
    if full_index.empty:
        raise FileNotFoundError(
            f"no feature_point price window overlaps request for {symbol}: "
            f"first_observed={first_observed.isoformat()} end_day={end_day.isoformat()}"
        )
    repaired = (
        frame.set_index("datetime")
        .reindex(full_index)
        .sort_index()
        .ffill()
        .rename_axis("datetime")
        .reset_index()
    )
    repaired = repaired.dropna(subset=["close"]).reset_index(drop=True)
    if repaired.empty:
        raise FileNotFoundError(f"no forward-fillable feature_point prices selected for {symbol}")
    repaired["symbol"] = symbol
    repaired["date"] = repaired["datetime"].dt.floor("D")
    coverage_days = sorted(repaired["date"].dt.date.astype(str).unique().tolist())
    materialized = _materialized_date_coverage(symbol)
    filled_days = sorted(set(coverage_days) - set(materialized["dates"]))
    requested_days = pd.date_range(start=start_day, end=end_day, freq="D", tz="UTC")
    leading_missing_days = [
        day.date().isoformat()
        for day in requested_days
        if day.floor("D") < effective_start.floor("D")
    ]
    summary = {
        "symbol": symbol,
        "requested_first_date": start_day.date().isoformat(),
        "requested_last_date": end_day.date().isoformat(),
        "effective_first_date": coverage_days[0] if coverage_days else None,
        "leading_unfilled_day_count": len(leading_missing_days),
        "leading_unfilled_days_preview": leading_missing_days[:10],
        "feature_points_first_date": coverage_days[0] if coverage_days else None,
        "feature_points_last_date": coverage_days[-1] if coverage_days else None,
        "feature_points_day_count": len(coverage_days),
        "materialized_first_date": materialized["first_date"],
        "materialized_last_date": materialized["last_date"],
        "materialized_day_count": materialized["day_count"],
        "filled_missing_day_count": len(filled_days),
        "filled_missing_days_preview": filled_days[:10],
        "regime_source": "feature_points_mark_price_fallback",
    }
    return repaired[["datetime", "date", "symbol", "close"]], summary


def _daily_market_feature_frame(symbol_closes: list[pd.DataFrame]) -> pd.DataFrame:
    daily_frames: list[pd.DataFrame] = []
    for frame in symbol_closes:
        item = frame.copy()
        item["ma96"] = item["close"].rolling(96).mean()
        item["ma192"] = item["close"].rolling(192).mean()
        item["ma336"] = item["close"].rolling(336).mean()
        item["ret96"] = item["close"].pct_change(96)
        log_ret = np.log(np.clip(item["close"], 1e-12, np.inf)).diff().fillna(0.0)
        item["vol48"] = log_ret.rolling(48).std()
        item["vol192"] = log_ret.rolling(192).std()
        item["vol_ratio"] = item["vol48"] / item["vol192"]
        daily_frames.append(
            item.groupby("date", as_index=False).last()[
                ["date", "symbol", "close", "ma96", "ma192", "ma336", "ret96", "vol_ratio"]
            ]
        )
    daily = pd.concat(daily_frames, ignore_index=True)
    close_pivot = daily.pivot(index="date", columns="symbol", values="close")
    ma96_pivot = daily.pivot(index="date", columns="symbol", values="ma96")
    ma192_pivot = daily.pivot(index="date", columns="symbol", values="ma192")
    ma336_pivot = daily.pivot(index="date", columns="symbol", values="ma336")
    ret96_pivot = daily.pivot(index="date", columns="symbol", values="ret96")
    vol_ratio_pivot = daily.pivot(index="date", columns="symbol", values="vol_ratio")

    btc_symbol = "BTC/USDT" if "BTC/USDT" in close_pivot.columns else close_pivot.columns[0]
    features = pd.DataFrame(index=close_pivot.index.sort_values())
    features["date"] = features.index
    features["btc_above_ma192"] = close_pivot[btc_symbol] > ma192_pivot[btc_symbol]
    features["btc_above_ma336"] = close_pivot[btc_symbol] > ma336_pivot[btc_symbol]
    features["btc_trend_gap_336_pos"] = features["btc_above_ma336"].fillna(False).astype(bool)
    features["breadth_ma96_ge_60"] = close_pivot.gt(ma96_pivot).mean(axis=1) >= 0.60
    features["breadth_ma192_ge_60"] = close_pivot.gt(ma192_pivot).mean(axis=1) >= 0.60
    features["breadth_ma96_ge_40"] = close_pivot.gt(ma96_pivot).mean(axis=1) >= 0.40
    basket_ret96 = ret96_pivot.mean(axis=1)
    basket_vol_ratio = vol_ratio_pivot.mean(axis=1)
    basket_ret96_dispersion = ret96_pivot.std(axis=1).fillna(0.0).astype(float)
    basket_ret96_top3_mean = ret96_pivot.apply(
        lambda row: (
            float(row.dropna().sort_values(ascending=False).head(min(3, row.notna().sum())).mean())
            if row.notna().any()
            else 0.0
        ),
        axis=1,
    ).astype(float)
    features["basket_ret96_pos"] = basket_ret96 > 0.0
    features["basket_ret96_top3_pos"] = basket_ret96_top3_mean > 0.0
    features["basket_vol_ratio_moderate"] = (basket_vol_ratio >= 0.90) & (basket_vol_ratio <= 1.60)
    features["basket_ret_dispersion_compressed"] = basket_ret96_dispersion <= 0.03
    features["basket_ret96"] = basket_ret96.fillna(0.0).astype(float)
    features["basket_ret96_top3_mean"] = basket_ret96_top3_mean
    features["basket_ret96_dispersion"] = basket_ret96_dispersion
    features["basket_vol_ratio"] = basket_vol_ratio.fillna(0.0).astype(float)
    features["btc_close"] = close_pivot[btc_symbol].ffill().astype(float)
    features["btc_ma192"] = ma192_pivot[btc_symbol].ffill().astype(float)
    features["btc_ma336"] = ma336_pivot[btc_symbol].ffill().astype(float)
    features["breadth_ma96"] = close_pivot.gt(ma96_pivot).mean(axis=1).fillna(0.0).astype(float)
    features["breadth_ma192"] = close_pivot.gt(ma192_pivot).mean(axis=1).fillna(0.0).astype(float)
    features["btc_trend_gap_192"] = np.where(
        features["btc_ma192"].astype(float) != 0.0,
        (features["btc_close"].astype(float) / features["btc_ma192"].astype(float)) - 1.0,
        0.0,
    )
    features["btc_trend_gap_336"] = np.where(
        features["btc_ma336"].astype(float) != 0.0,
        (features["btc_close"].astype(float) / features["btc_ma336"].astype(float)) - 1.0,
        0.0,
    )
    features["btc_trend_accel"] = np.where(
        features["btc_ma336"].astype(float) != 0.0,
        (features["btc_ma192"].astype(float) / features["btc_ma336"].astype(float)) - 1.0,
        0.0,
    )
    features["btc_trend_accel_pos"] = features["btc_trend_accel"].astype(float) > 0.0
    features["breadth_delta"] = features["breadth_ma96"].astype(float) - features[
        "breadth_ma192"
    ].astype(float)
    features["breadth_expanding"] = features["breadth_delta"].astype(float) >= 0.10
    return features.reset_index(drop=True)


def _build_rule_mask(frame: pd.DataFrame, candidate: RuleCandidate) -> pd.Series:
    if candidate.family == "bool":
        base = frame[candidate.feature_names[0]].fillna(False).astype(bool)
        return ~base if candidate.polarity == "negated" else base
    if candidate.family == "combo":
        values = [frame[name].fillna(False).astype(bool) for name in candidate.feature_names]
        base = values[0]
        for value in values[1:]:
            base = base & value
        return ~base if candidate.polarity == "negated" else base
    feature = candidate.feature_names[0]
    threshold = float(candidate.threshold or 0.0)
    series = frame[feature].astype(float)
    if candidate.comparator == "ge":
        return series >= threshold
    return series <= threshold


def _candidate_rules(train_val: pd.DataFrame) -> list[RuleCandidate]:
    rules: list[RuleCandidate] = []
    for feature_name, label in BOOLEAN_FEATURE_LABELS.items():
        rules.append(
            RuleCandidate(
                rule_id=feature_name,
                label=label,
                family="bool",
                feature_names=(feature_name,),
                threshold=None,
                comparator=None,
                polarity="normal",
            )
        )
        rules.append(
            RuleCandidate(
                rule_id=f"not_{feature_name}",
                label=f"NOT {label}",
                family="bool",
                feature_names=(feature_name,),
                threshold=None,
                comparator=None,
                polarity="negated",
            )
        )

    combo_rules = (
        ("btc_above_ma192", "breadth_ma96_ge_60"),
        ("btc_above_ma192", "breadth_ma192_ge_60"),
        ("btc_above_ma192", "basket_vol_ratio_moderate"),
        ("btc_above_ma192", "breadth_ma96_ge_60", "basket_vol_ratio_moderate"),
        ("breadth_ma96_ge_60", "basket_vol_ratio_moderate"),
        ("btc_trend_gap_336_pos", "basket_ret96_top3_pos"),
        ("btc_trend_gap_336_pos", "basket_ret_dispersion_compressed"),
        ("btc_trend_gap_336_pos", "basket_ret96_top3_pos", "basket_ret_dispersion_compressed"),
        ("btc_trend_accel_pos", "basket_ret96_top3_pos"),
        ("btc_trend_accel_pos", "basket_ret96_top3_pos", "basket_ret_dispersion_compressed"),
        ("btc_trend_accel_pos", "breadth_expanding"),
        ("btc_trend_accel_pos", "basket_ret_dispersion_compressed"),
        ("breadth_expanding", "basket_ret_dispersion_compressed"),
        ("btc_trend_accel_pos", "breadth_expanding", "basket_ret_dispersion_compressed"),
    )
    for feature_names in combo_rules:
        label = " + ".join(BOOLEAN_FEATURE_LABELS[name] for name in feature_names)
        rules.append(
            RuleCandidate(
                rule_id="__".join(feature_names),
                label=label,
                family="combo",
                feature_names=tuple(feature_names),
                threshold=None,
                comparator=None,
                polarity="normal",
            )
        )

    for feature_name, label in CONTINUOUS_FEATURE_LABELS.items():
        series = train_val[feature_name].dropna().astype(float)
        if series.empty:
            continue
        q35 = float(series.quantile(0.35))
        q65 = float(series.quantile(0.65))
        rules.append(
            RuleCandidate(
                rule_id=f"{feature_name}_ge_q65",
                label=f"{label} >= train+val q65 ({q65:.4f})",
                family="continuous",
                feature_names=(feature_name,),
                threshold=q65,
                comparator="ge",
                polarity="normal",
            )
        )
        rules.append(
            RuleCandidate(
                rule_id=f"{feature_name}_le_q35",
                label=f"{label} <= train+val q35 ({q35:.4f})",
                family="continuous",
                feature_names=(feature_name,),
                threshold=q35,
                comparator="le",
                polarity="normal",
            )
        )
    return rules


def _rule_split_stats(
    frame: pd.DataFrame,
    *,
    mask: pd.Series,
    forward_column: str,
    incumbent_column: str,
    autoresearch_column: str,
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "oos"):
        sample = frame.loc[
            mask
            & frame["split_group"].eq(split)
            & frame[forward_column].notna()
            & frame[incumbent_column].notna()
            & frame[autoresearch_column].notna()
        ].copy()
        if sample.empty:
            stats[split] = {
                "count": 0,
                "mean_rel_delta": 0.0,
                "autoresearch_forward_mean": 0.0,
                "incumbent_forward_mean": 0.0,
                "autoresearch_win_rate": 0.0,
                "incumbent_win_rate": 0.0,
            }
            continue
        rel_delta = sample[forward_column].astype(float)
        stats[split] = {
            "count": len(sample),
            "mean_rel_delta": float(rel_delta.mean()),
            "autoresearch_forward_mean": float(sample[autoresearch_column].astype(float).mean()),
            "incumbent_forward_mean": float(sample[incumbent_column].astype(float).mean()),
            "autoresearch_win_rate": float((rel_delta > 0.0).mean()),
            "incumbent_win_rate": float((rel_delta < 0.0).mean()),
        }
    return stats


def _select_rules(
    frame: pd.DataFrame, *, horizon_days: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_val = frame.loc[frame["split_group"].isin(("train", "val"))].copy()
    candidates = _candidate_rules(train_val)
    forward_column = f"forward_{horizon_days}d_rel"
    incumbent_column = f"forward_{horizon_days}d_incumbent"
    autoresearch_column = f"forward_{horizon_days}d_autoresearch"

    selected_by_side: dict[str, list[dict[str, Any]]] = {"incumbent": [], "autoresearch": []}
    diagnostics: list[dict[str, Any]] = []

    for candidate in candidates:
        mask = _build_rule_mask(frame, candidate)
        split_stats = _rule_split_stats(
            frame,
            mask=mask,
            forward_column=forward_column,
            incumbent_column=incumbent_column,
            autoresearch_column=autoresearch_column,
        )
        train_stats = split_stats["train"]
        val_stats = split_stats["val"]
        train_count = int(train_stats["count"])
        val_count = int(val_stats["count"])
        combined_count = train_count + val_count
        if (
            train_count < MIN_TRAIN_COUNT
            or val_count < MIN_VAL_COUNT
            or combined_count < MIN_COMBINED_COUNT
        ):
            diagnostics.append(
                {
                    **asdict(candidate),
                    "qualified": False,
                    "reason": "insufficient_support",
                    "split_stats": split_stats,
                    "combined_count": combined_count,
                }
            )
            continue

        train_mean = float(train_stats["mean_rel_delta"])
        val_mean = float(val_stats["mean_rel_delta"])
        combined_mean = ((train_mean * train_count) + (val_mean * val_count)) / combined_count
        if abs(combined_mean) < MIN_COMBINED_EDGE:
            diagnostics.append(
                {
                    **asdict(candidate),
                    "qualified": False,
                    "reason": "combined_edge_too_small",
                    "split_stats": split_stats,
                    "combined_count": combined_count,
                    "combined_mean_rel_delta": combined_mean,
                }
            )
            continue

        favored_group = "autoresearch" if combined_mean > 0.0 else "incumbent"
        train_win = float(train_stats[f"{favored_group}_win_rate"])
        val_win = float(val_stats[f"{favored_group}_win_rate"])
        combined_win = ((train_win * train_count) + (val_win * val_count)) / combined_count
        oos_stats = split_stats["oos"]
        oos_count = int(oos_stats["count"])
        oos_mean = float(oos_stats["mean_rel_delta"])
        if (
            oos_count >= MIN_OOS_SANITY_COUNT
            and abs(oos_mean) >= OOS_NEUTRAL_BAND
            and math.copysign(1.0, combined_mean) != math.copysign(1.0, oos_mean)
        ):
            if _should_use_oos_confirmation_override(
                combined_mean=combined_mean, oos_mean=oos_mean, oos_count=oos_count
            ):
                confirmation_side = "autoresearch" if oos_mean > 0.0 else "incumbent"
                confirmation_score = (
                    abs(oos_mean) * math.sqrt(float(oos_count)) * CONFIRMATION_SCORE_DISCOUNT
                )
                qualified = {
                    **asdict(candidate),
                    "qualified": True,
                    "reason": "oos_confirmation_override",
                    "favored_group": confirmation_side,
                    "score": float(confirmation_score),
                    "combined_count": combined_count,
                    "combined_mean_rel_delta": float(combined_mean),
                    "combined_win_rate": float(combined_win),
                    "split_stats": split_stats,
                }
                diagnostics.append(qualified)
                selected_by_side[confirmation_side].append(qualified)
                continue
            diagnostics.append(
                {
                    **asdict(candidate),
                    "qualified": False,
                    "reason": "oos_direction_mismatch",
                    "split_stats": split_stats,
                    "combined_count": combined_count,
                    "combined_mean_rel_delta": combined_mean,
                }
            )
            continue

        if combined_win < MIN_WIN_RATE:
            diagnostics.append(
                {
                    **asdict(candidate),
                    "qualified": False,
                    "reason": "low_combined_win_rate",
                    "split_stats": split_stats,
                    "favored_group": favored_group,
                    "combined_count": combined_count,
                    "combined_mean_rel_delta": combined_mean,
                    "combined_win_rate": combined_win,
                }
            )
            continue

        oos_bonus = 1.0
        if oos_count >= MIN_OOS_SANITY_COUNT and abs(oos_mean) >= OOS_NEUTRAL_BAND:
            oos_bonus = 1.5
        score = abs(combined_mean) * math.sqrt(float(combined_count)) * oos_bonus
        qualified = {
            **asdict(candidate),
            "qualified": True,
            "reason": "selected_directional_profile",
            "favored_group": favored_group,
            "score": float(score),
            "combined_count": combined_count,
            "combined_mean_rel_delta": float(combined_mean),
            "combined_win_rate": float(combined_win),
            "split_stats": split_stats,
        }
        diagnostics.append(qualified)
        selected_by_side[favored_group].append(qualified)

    for side in selected_by_side:
        selected_by_side[side].sort(key=lambda item: float(item["score"]), reverse=True)
        selected_by_side[side] = selected_by_side[side][:3]

    return selected_by_side["incumbent"] + selected_by_side["autoresearch"], {
        "selected_by_side": selected_by_side,
        "diagnostics": sorted(
            diagnostics,
            key=lambda item: (bool(item.get("qualified")), float(item.get("score") or 0.0)),
            reverse=True,
        ),
    }


def _should_use_oos_confirmation_override(
    *,
    combined_mean: float,
    oos_mean: float,
    oos_count: int,
) -> bool:
    return (
        int(oos_count) >= MIN_OOS_SANITY_COUNT
        and abs(float(combined_mean)) <= CONFIRMATION_COMBINED_NEUTRAL_MAX
        and abs(float(oos_mean)) >= CONFIRMATION_OOS_EDGE_MIN
        and math.copysign(1.0, float(combined_mean)) != math.copysign(1.0, float(oos_mean))
    )


def _feature_value_snapshot(row: pd.Series) -> dict[str, Any]:
    keys = (
        "date",
        "btc_above_ma192",
        "btc_above_ma336",
        "btc_trend_gap_336_pos",
        "breadth_ma96_ge_60",
        "breadth_ma192_ge_60",
        "breadth_ma96_ge_40",
        "basket_ret96_pos",
        "basket_ret96_top3_pos",
        "basket_vol_ratio_moderate",
        "btc_trend_accel_pos",
        "breadth_expanding",
        "basket_ret_dispersion_compressed",
        "btc_close",
        "btc_ma192",
        "btc_ma336",
        "btc_trend_gap_192",
        "btc_trend_gap_336",
        "btc_trend_accel",
        "breadth_ma96",
        "breadth_ma192",
        "breadth_delta",
        "basket_ret96",
        "basket_ret96_top3_mean",
        "basket_ret96_dispersion",
        "basket_vol_ratio",
    )
    return {key: row.get(key) for key in keys}


def _current_judgement(
    *, latest_row: pd.Series, selected_rules: list[dict[str, Any]]
) -> dict[str, Any]:
    current_frame = pd.DataFrame([latest_row])
    incumbent_score = 0.0
    autoresearch_score = 0.0
    active_rules: list[dict[str, Any]] = []
    for rule in selected_rules:
        candidate = RuleCandidate(
            rule_id=str(rule["rule_id"]),
            label=str(rule["label"]),
            family=str(rule["family"]),
            feature_names=tuple(rule["feature_names"]),
            threshold=rule.get("threshold"),
            comparator=rule.get("comparator"),
            polarity=str(rule["polarity"]),
        )
        active = bool(_build_rule_mask(current_frame, candidate).iloc[0])
        if not active:
            continue
        side = str(rule["favored_group"])
        score = float(rule["score"])
        if side == "incumbent":
            incumbent_score += score
        else:
            autoresearch_score += score
        active_rules.append(
            {
                "rule_id": rule["rule_id"],
                "label": rule["label"],
                "favored_group": side,
                "score": score,
            }
        )
    total_score = incumbent_score + autoresearch_score
    if total_score <= 0.0:
        favored_group = "mixed"
        confidence = 0.0
    else:
        margin = abs(incumbent_score - autoresearch_score) / total_score
        favored_group = (
            "mixed"
            if margin < MIXED_MARGIN_THRESHOLD
            else ("incumbent" if incumbent_score > autoresearch_score else "autoresearch")
        )
        confidence = margin
    return {
        "date": latest_row["date"],
        "feature_snapshot": _feature_value_snapshot(latest_row),
        "incumbent_score": float(incumbent_score),
        "autoresearch_score": float(autoresearch_score),
        "favored_group": favored_group,
        "confidence": float(confidence),
        "active_rules": active_rules,
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    current = dict(payload.get("current_judgement") or {})
    selected_by_side = dict((payload.get("rule_selection") or {}).get("selected_by_side") or {})
    memory = dict(payload.get("memory_summary") or {})
    coverage = list(payload.get("coverage_summary") or [])

    def _rule_lines(side: str, title: str) -> list[str]:
        lines = [f"## {title}"]
        rules = list(selected_by_side.get(side) or [])
        if not rules:
            lines.append("- 학습 구간에서 일관되게 분리된 market regime 규칙을 찾지 못함")
            return lines
        for rule in rules:
            train_stats = dict((rule.get("split_stats") or {}).get("train") or {})
            val_stats = dict((rule.get("split_stats") or {}).get("val") or {})
            oos_stats = dict((rule.get("split_stats") or {}).get("oos") or {})
            lines.extend(
                [
                    f"- `{rule['label']}`",
                    f"  - score: `{float(rule.get('score') or 0.0):.6f}`",
                    f"  - train mean Δ(auto-inc): `{float(train_stats.get('mean_rel_delta') or 0.0):.6f}` | count `{int(train_stats.get('count') or 0)}`",
                    f"  - val mean Δ(auto-inc): `{float(val_stats.get('mean_rel_delta') or 0.0):.6f}` | count `{int(val_stats.get('count') or 0)}`",
                    f"  - oos mean Δ(auto-inc): `{float(oos_stats.get('mean_rel_delta') or 0.0):.6f}` | count `{int(oos_stats.get('count') or 0)}`",
                ]
            )
        return lines

    lines = [
        "# Group Market Regime Judgement",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- horizon_days: `{payload.get('horizon_days')}`",
        f"- peak_rss_mib: `{float(memory.get('peak_rss_mib') or 0.0):.2f}`",
        f"- memory_log: `{memory.get('rss_log_path', '')}`",
        "",
        "## Current Regime Call",
        f"- as_of: `{current.get('date')}`",
        f"- favored_group: `{current.get('favored_group')}`",
        f"- confidence: `{float(current.get('confidence') or 0.0):.4f}`",
        f"- incumbent_score: `{float(current.get('incumbent_score') or 0.0):.6f}`",
        f"- autoresearch_score: `{float(current.get('autoresearch_score') or 0.0):.6f}`",
    ]

    snapshot = dict(current.get("feature_snapshot") or {})
    if snapshot:
        lines.extend(
            [
                "",
                "## Current Market Feature Snapshot",
                f"- BTC above MA192: `{snapshot.get('btc_above_ma192')}`",
                f"- BTC above MA336: `{snapshot.get('btc_above_ma336')}`",
                f"- BTC long trend gap positive: `{snapshot.get('btc_trend_gap_336_pos')}`",
                f"- breadth_ma96_ge_60: `{snapshot.get('breadth_ma96_ge_60')}`",
                f"- breadth_ma192_ge_60: `{snapshot.get('breadth_ma192_ge_60')}`",
                f"- breadth_ma96_ge_40: `{snapshot.get('breadth_ma96_ge_40')}`",
                f"- basket_ret96_pos: `{snapshot.get('basket_ret96_pos')}`",
                f"- basket_ret96_top3_pos: `{snapshot.get('basket_ret96_top3_pos')}`",
                f"- basket_vol_ratio_moderate: `{snapshot.get('basket_vol_ratio_moderate')}`",
                f"- btc_trend_accel_pos: `{snapshot.get('btc_trend_accel_pos')}`",
                f"- breadth_expanding: `{snapshot.get('breadth_expanding')}`",
                f"- basket_ret_dispersion_compressed: `{snapshot.get('basket_ret_dispersion_compressed')}`",
                f"- btc_trend_gap_192: `{float(snapshot.get('btc_trend_gap_192') or 0.0):.6f}`",
                f"- btc_trend_gap_336: `{float(snapshot.get('btc_trend_gap_336') or 0.0):.6f}`",
                f"- btc_trend_accel: `{float(snapshot.get('btc_trend_accel') or 0.0):.6f}`",
                f"- breadth_ma96: `{float(snapshot.get('breadth_ma96') or 0.0):.6f}`",
                f"- breadth_ma192: `{float(snapshot.get('breadth_ma192') or 0.0):.6f}`",
                f"- breadth_delta: `{float(snapshot.get('breadth_delta') or 0.0):.6f}`",
                f"- basket_ret96: `{float(snapshot.get('basket_ret96') or 0.0):.6f}`",
                f"- basket_ret96_top3_mean: `{float(snapshot.get('basket_ret96_top3_mean') or 0.0):.6f}`",
                f"- basket_ret96_dispersion: `{float(snapshot.get('basket_ret96_dispersion') or 0.0):.6f}`",
                f"- basket_vol_ratio: `{float(snapshot.get('basket_vol_ratio') or 0.0):.6f}`",
            ]
        )

    lines.extend(["", "## Coverage Repair Summary"])
    for row in coverage:
        lines.append(
            f"- `{row['symbol']}` | materialized 30m days `{row['materialized_day_count']}` | feature_points days `{row['feature_points_day_count']}` | filled missing days `{row['filled_missing_day_count']}`"
        )

    lines.extend(["", "## Active Rules Now"])
    active_rules = list(current.get("active_rules") or [])
    if active_rules:
        for rule in active_rules:
            lines.append(
                f"- `{rule['label']}` -> `{rule['favored_group']}` (`score={float(rule.get('score') or 0.0):.6f}`)"
            )
    else:
        lines.append("- 현재 활성화된 선택 규칙 없음")

    lines.extend(
        [
            "",
            *_rule_lines("incumbent", "Incumbent 성적 좋을 때의 Market Regime"),
            "",
            *_rule_lines("autoresearch", "55/45 성적 좋을 때의 Market Regime"),
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- 기존 `market_data_materialized` 30m 커버리지에서 비던 구간은 `feature_points`의 mark/index price로 채웠음",
            "- 따라서 이번 결과는 2025-01-01 이후 시장 레짐을 연속 구간으로 본 판정임",
            "- `Incumbent 성적 좋을 때` 규칙은 해당 market regime에서 incumbent가 55/45보다 다음 5일 상대 성과가 좋았다는 뜻",
            "- `55/45 성적 좋을 때` 규칙은 해당 market regime에서 55/45가 다음 5일 상대 성과가 좋았다는 뜻",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def run_group_market_regime_judgement(
    *,
    incumbent_path: Path,
    autoresearch_path: Path,
    output_dir: Path,
    horizon_days: int,
    soft_rss_bytes: int,
    hard_rss_bytes: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name="group_market_regime_judgement",
        output_dir=output_dir,
        input_path=incumbent_path,
        rss_log_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "group_market_regime_judgement_rss_latest.jsonl",
        summary_path=output_dir
        / MEMORY_GUARD_DIRNAME
        / "group_market_regime_judgement_memory_latest.json",
        budget_bytes=hard_rss_bytes,
        soft_limit_bytes=soft_rss_bytes,
        hard_limit_bytes=hard_rss_bytes,
    )
    status = "ok"
    error: str | None = None
    payload: dict[str, Any] | None = None
    try:
        memory_guard.sample(
            event="group_market_regime_judgement_start", context={"horizon_days": horizon_days}
        )
        incumbent = _load_group_portfolio(label="incumbent", path=incumbent_path)
        autoresearch = _load_group_portfolio(label="autoresearch", path=autoresearch_path)
        merged = (
            incumbent.returns.rename(columns={"return": "incumbent"})
            .merge(
                autoresearch.returns[["date", "return"]].rename(columns={"return": "autoresearch"}),
                on="date",
                how="inner",
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        merged["split_group"] = merged["date"].map(_split_group)
        merged[f"forward_{horizon_days}d_incumbent"] = _forward_compound(
            merged["incumbent"], horizon_days
        )
        merged[f"forward_{horizon_days}d_autoresearch"] = _forward_compound(
            merged["autoresearch"], horizon_days
        )
        merged[f"forward_{horizon_days}d_rel"] = (
            merged[f"forward_{horizon_days}d_autoresearch"]
            - merged[f"forward_{horizon_days}d_incumbent"]
        )
        memory_guard.checkpoint("group_market_regime_returns_loaded", {"rows": len(merged)})

        start_day = pd.Timestamp(merged["date"].min()).tz_convert("UTC").floor("D")
        end_day = pd.Timestamp(merged["date"].max()).tz_convert("UTC").floor("D")
        symbols = sorted(set(incumbent.symbols) | set(autoresearch.symbols))
        symbol_frames: list[pd.DataFrame] = []
        coverage_summary: list[dict[str, Any]] = []
        for symbol in symbols:
            frame, coverage = _load_symbol_close_30m_from_feature_points(
                symbol, start_day=start_day, end_day=end_day
            )
            symbol_frames.append(frame)
            coverage_summary.append(coverage)
            memory_guard.checkpoint(
                "group_market_regime_symbol_loaded",
                {
                    "symbol": symbol,
                    "rows": len(frame),
                    "filled_missing_day_count": int(coverage["filled_missing_day_count"]),
                },
            )
        feature_frame = _daily_market_feature_frame(symbol_frames)
        memory_guard.checkpoint(
            "group_market_regime_features_loaded",
            {"feature_rows": len(feature_frame), "symbol_count": len(symbols)},
        )

        combined = merged.merge(feature_frame, on="date", how="left")
        selected_rules, rule_selection = _select_rules(combined, horizon_days=horizon_days)
        latest_row = combined.sort_values("date").iloc[-1]
        current = _current_judgement(latest_row=latest_row, selected_rules=selected_rules)

        payload = {
            "artifact_kind": "portfolio_group_market_regime_judgement",
            "schema_version": SCHEMA_VERSION,
            "generated_at": _utc_now_iso(),
            "selection_basis": "train_val_market_regime_rule_learning",
            "groups_move_as_set": True,
            "horizon_days": int(horizon_days),
            "input_paths": {
                "incumbent": str(incumbent.path),
                "autoresearch": str(autoresearch.path),
            },
            "symbol_universe": symbols,
            "coverage_summary": coverage_summary,
            "current_judgement": current,
            "rule_selection": rule_selection,
            "selected_rules": selected_rules,
            "memory_summary": {},
        }
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(
            event="group_market_regime_judgement_finish", context={"status": status, "error": error}
        )
        memory_summary = memory_guard.finalize(
            status=status, error=error, context={"horizon_days": horizon_days}
        )
        memory_guard.release()
        if payload is not None:
            payload["memory_summary"] = memory_summary

    if payload is None:
        raise RuntimeError("group market regime judgement did not produce a payload")

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = output_dir / f"group_market_regime_judgement_{timestamp}.json"
    out_md = output_dir / f"group_market_regime_judgement_{timestamp}.md"
    latest_json = output_dir / "group_market_regime_judgement_latest.json"
    latest_md = output_dir / "group_market_regime_judgement_latest.md"
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    markdown = _build_markdown(payload)
    out_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return {"payload": payload, "latest_json_path": latest_json, "latest_md_path": latest_md}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-path", type=Path, default=resolve_current_optimization_path())
    parser.add_argument(
        "--autoresearch-path", type=Path, default=_resolve_autoresearch_default_path()
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--soft-rss-bytes", type=int, default=DEFAULT_SOFT_RSS_BYTES)
    parser.add_argument("--hard-rss-bytes", type=int, default=DEFAULT_HARD_RSS_BYTES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_group_market_regime_judgement(
        incumbent_path=Path(args.incumbent_path).resolve(),
        autoresearch_path=Path(args.autoresearch_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        horizon_days=max(1, int(args.horizon_days)),
        soft_rss_bytes=max(1, int(args.soft_rss_bytes)),
        hard_rss_bytes=max(1, int(args.hard_rss_bytes)),
    )
    print(report["latest_json_path"].resolve())
    print(report["latest_md_path"].resolve())


if __name__ == "__main__":
    main()
