"""Crypto/FX Alpha Zoo v0 factor panel and train/validation screen."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import operators as ops

CRYPTO_CANONICAL = ("BTC", "ETH", "SOL", "BNB", "TRX")
FX_CANONICAL = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "AUDJPY", "XAUUSD")
CALENDAR_FIELD_NAMES = frozenset({"month", "day", "weekday", "hour", "minute", "session_hour"})


@dataclass(frozen=True, slots=True)
class FactorSpec:
    """Describes a formulaic alpha factor without embedding a calendar rule."""

    name: str
    family: str
    market: str
    inputs: tuple[str, ...]
    description: str
    calendar_fields: tuple[str, ...] = ()
    parameters: Mapping[str, Any] | None = None

    @property
    def is_calendar_safe(self) -> bool:
        return not (set(self.calendar_fields) & CALENDAR_FIELD_NAMES)


def normalize_symbol(symbol: Any) -> str:
    token = str(symbol or "").upper().replace("-", "").replace("_", "").replace("/", "")
    if token.endswith("PERP"):
        token = token[:-4]
    if token.endswith("USDT") and len(token) > 4:
        return token[:-4]
    if token.endswith("USD") and len(token) > 3 and token not in FX_CANONICAL:
        return token[:-3]
    return token


def is_crypto_symbol(symbol: Any) -> bool:
    return normalize_symbol(symbol) in CRYPTO_CANONICAL or str(symbol).upper().endswith("USDT")


def is_fx_symbol(symbol: Any) -> bool:
    return normalize_symbol(symbol) in FX_CANONICAL


def infer_market(symbol: Any) -> str:
    if is_crypto_symbol(symbol):
        return "crypto"
    if is_fx_symbol(symbol):
        return "fx"
    return "unknown"


def _spec(
    name: str, family: str, market: str, inputs: Iterable[str], description: str, **params: Any
) -> FactorSpec:
    return FactorSpec(
        name=name,
        family=family,
        market=market,
        inputs=tuple(inputs),
        description=description,
        parameters=dict(params) if params else None,
    )


def build_crypto_fx_factor_specs() -> tuple[FactorSpec, ...]:
    """Return a bounded, deterministic 50+ factor set for crypto/FX research.

    The first pass deliberately uses Alpha191-style operators and excludes all
    fixed month/day/hour/session fields.  Volume/VWAP and funding factors are
    crypto-only; FX starts with price-only and basket-regime factors.
    """
    specs: list[FactorSpec] = []
    specs.extend(
        [
            _spec("kmid", "price_shape", "crypto_fx", ("open", "close"), "candle body / open"),
            _spec("klen", "price_shape", "crypto_fx", ("open", "high", "low"), "range / open"),
            _spec(
                "kup", "price_shape", "crypto_fx", ("open", "high", "close"), "upper wick / open"
            ),
            _spec(
                "klow", "price_shape", "crypto_fx", ("open", "low", "close"), "lower wick / open"
            ),
        ]
    )
    for window in (2, 4, 8, 12, 24, 48):
        specs.extend(
            [
                _spec(
                    f"ret_{window}",
                    "momentum_reversal",
                    "crypto_fx",
                    ("close",),
                    f"{window}-bar return",
                    window=window,
                ),
                _spec(
                    f"mom_z_{window}",
                    "momentum_reversal",
                    "crypto_fx",
                    ("close",),
                    f"rolling z-score of {window}-bar return",
                    window=window,
                ),
                _spec(
                    f"range_position_{window}",
                    "price_shape",
                    "crypto_fx",
                    ("high", "low", "close"),
                    f"close position inside {window}-bar range",
                    window=window,
                ),
                _spec(
                    f"trend_efficiency_{window}",
                    "price_shape",
                    "crypto_fx",
                    ("close",),
                    f"net move divided by path length over {window} bars",
                    window=window,
                ),
                _spec(
                    f"realized_vol_{window}",
                    "volatility_regime",
                    "crypto_fx",
                    ("close",),
                    f"rolling close-to-close volatility over {window} bars",
                    window=window,
                ),
            ]
        )
    for window in (12, 24, 48):
        specs.extend(
            [
                _spec(
                    f"breakout_failure_{window}",
                    "breakout_failure",
                    "crypto_fx",
                    ("high", "low", "close"),
                    f"failed high/low breakout over {window} bars",
                    window=window,
                ),
                _spec(
                    f"volume_shock_z_{window}",
                    "volume_vwap",
                    "crypto",
                    ("volume",),
                    f"crypto volume shock z-score over {window} bars",
                    window=window,
                ),
                _spec(
                    f"volume_vwap_pressure_{window}",
                    "volume_vwap",
                    "crypto",
                    ("volume", "vwap", "close"),
                    f"volume shock times close-vwap sign over {window} bars",
                    window=window,
                ),
            ]
        )
    for window in (4, 12, 24, 48):
        specs.extend(
            [
                _spec(
                    f"btc_residual_ret_{window}",
                    "benchmark_relative",
                    "crypto",
                    ("close", "BTC"),
                    f"asset return minus BTC return over {window} bars",
                    window=window,
                ),
                _spec(
                    f"btc_residual_z_{window}",
                    "benchmark_relative",
                    "crypto",
                    ("close", "BTC"),
                    f"rolling z-score of BTC residual return over {window} bars",
                    window=window,
                ),
                _spec(
                    f"crypto_leadership_rank_{window}",
                    "benchmark_relative",
                    "crypto",
                    ("close",),
                    f"cross-sectional crypto residual rank over {window} bars",
                    window=window,
                ),
            ]
        )
    for window in (12, 24):
        specs.extend(
            [
                _spec(
                    f"funding_z_{window}",
                    "perp_context",
                    "crypto",
                    ("funding_rate",),
                    f"funding z-score over {window} bars when available",
                    window=window,
                ),
                _spec(
                    f"funding_crowding_reversal_{window}",
                    "perp_context",
                    "crypto",
                    ("funding_rate", "close"),
                    f"negative funding z-score with price confirmation over {window} bars",
                    window=window,
                ),
                _spec(
                    f"fx_usd_strength_{window}",
                    "fx_regime",
                    "fx_regime",
                    ("EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF"),
                    f"USD basket strength over {window} bars",
                    window=window,
                ),
                _spec(
                    f"fx_jpy_risk_off_{window}",
                    "fx_regime",
                    "fx_regime",
                    ("USDJPY", "AUDJPY"),
                    f"JPY risk-off basket over {window} bars",
                    window=window,
                ),
            ]
        )
    return tuple(specs)


def factor_columns(specs: Sequence[FactorSpec] | None = None) -> tuple[str, ...]:
    return tuple(spec.name for spec in (specs or build_crypto_fx_factor_specs()))


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("timestamp", "symbol", "close"))
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("timestamp column contains unparsable values")
    out["symbol"] = out["symbol"].astype(str)
    for column in ("open", "high", "low", "close", "volume", "vwap", "funding_rate"):
        if column not in out.columns:
            if column == "open" or column in {"high", "low", "vwap"}:
                out[column] = out["close"]
            else:
                out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    if "market" not in out.columns:
        out["market"] = out["symbol"].map(infer_market)
    out["market"] = out["market"].astype(str)
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _map_btc_return(out: pd.DataFrame, ret_col: str) -> pd.Series:
    btc_rows = out[out["symbol"].map(lambda value: normalize_symbol(value) == "BTC")]
    if btc_rows.empty:
        return pd.Series(0.0, index=out.index, dtype=float)
    btc_by_time = btc_rows.drop_duplicates("timestamp", keep="last").set_index("timestamp")[ret_col]
    return out["timestamp"].map(btc_by_time).astype(float).fillna(0.0)


def _basket_return(out: pd.DataFrame, ret_col: str, weights: Mapping[str, float]) -> pd.Series:
    rows = out[out["symbol"].map(lambda value: normalize_symbol(value) in weights)]
    if rows.empty:
        return pd.Series(0.0, index=out.index, dtype=float)
    tmp = rows.assign(
        _name=rows["symbol"].map(normalize_symbol), _weight=lambda df: df["_name"].map(weights)
    )
    basket = (tmp[ret_col] * tmp["_weight"]).groupby(tmp["timestamp"], sort=False).sum()
    return out["timestamp"].map(basket).astype(float).fillna(0.0)


def compute_factor_frame(
    frame: pd.DataFrame, *, specs: Sequence[FactorSpec] | None = None
) -> pd.DataFrame:
    """Compute the v0 crypto/FX alpha-zoo factor panel.

    All rolling features are grouped by symbol and use trailing values only.  FX
    basket factors are copied to every row at the same timestamp so they can be
    used as crypto regime filters without directly trading FX in v0.
    """
    selected_specs = tuple(specs or build_crypto_fx_factor_specs())
    for spec in selected_specs:
        if not spec.is_calendar_safe:
            raise ValueError(f"calendar field is not allowed in factor spec: {spec.name}")

    out = _prepare_frame(frame)
    symbol = out["symbol"]
    market = out["market"]
    crypto_mask = market.eq("crypto") | symbol.map(is_crypto_symbol)
    price_denom = out["open"].where(out["open"].abs() > 1e-12)
    range_denom = (out["high"] - out["low"]).where((out["high"] - out["low"]).abs() > 1e-12)

    out["kmid"] = (out["close"] - out["open"]) / price_denom
    out["klen"] = (out["high"] - out["low"]) / price_denom
    out["kup"] = (out["high"] - out[["open", "close"]].max(axis=1)) / price_denom
    out["klow"] = (out[["open", "close"]].min(axis=1) - out["low"]) / price_denom

    ret_1 = ops.returns(out["close"], 1, by=symbol)
    for window in (2, 4, 8, 12, 24, 48):
        ret_col = f"ret_{window}"
        out[ret_col] = ops.returns(out["close"], window, by=symbol)
        out[f"mom_z_{window}"] = ops.zscore(out[ret_col], max(8, window * 2), by=symbol)
        low_roll = ops.rolling_min(out["low"], window, by=symbol)
        high_roll = ops.rolling_max(out["high"], window, by=symbol)
        out[f"range_position_{window}"] = (
            (out["close"] - low_roll)
            / (high_roll - low_roll).where((high_roll - low_roll).abs() > 1e-12)
        ) - 0.5
        path = ops.rolling_sum(ret_1.abs(), window, by=symbol)
        out[f"trend_efficiency_{window}"] = (
            out["close"] - ops.delay(out["close"], window, by=symbol)
        ).abs() / (
            ops.delay(out["close"], window, by=symbol).abs() * path.where(path.abs() > 1e-12)
        )
        out[f"realized_vol_{window}"] = ops.rolling_std(ret_1, window, by=symbol)

    for window in (12, 24, 48):
        prev_high = ops.delay(ops.rolling_max(out["high"], window, by=symbol), 1, by=symbol)
        prev_low = ops.delay(ops.rolling_min(out["low"], window, by=symbol), 1, by=symbol)
        high_fail = (out["high"] > prev_high) & (out["close"] < prev_high)
        low_fail = (out["low"] < prev_low) & (out["close"] > prev_low)
        out[f"breakout_failure_{window}"] = np.select(
            [high_fail, low_fail],
            [
                -((out["high"] - out["close"]) / range_denom),
                (out["close"] - out["low"]) / range_denom,
            ],
            default=0.0,
        )
        volume_z = ops.zscore(out["volume"], window, by=symbol).where(crypto_mask)
        close_vwap_dist = (
            (out["close"] / out["vwap"].where(out["vwap"].abs() > 1e-12)) - 1.0
        ).where(crypto_mask)
        out[f"volume_shock_z_{window}"] = volume_z
        out[f"volume_vwap_pressure_{window}"] = volume_z * np.sign(close_vwap_dist)

    for window in (4, 12, 24, 48):
        ret_col = f"ret_{window}"
        btc_ret = _map_btc_return(out, ret_col)
        residual = (out[ret_col] - btc_ret).where(crypto_mask)
        out[f"btc_residual_ret_{window}"] = residual
        out[f"btc_residual_z_{window}"] = ops.zscore(
            residual.fillna(0.0), max(8, window * 2), by=symbol
        ).where(crypto_mask)
        out[f"crypto_leadership_rank_{window}"] = ops.rank(
            residual.where(crypto_mask), by=out["timestamp"]
        ).where(crypto_mask)

    for window in (12, 24):
        funding_z = ops.zscore(out["funding_rate"], window, by=symbol).where(crypto_mask)
        out[f"funding_z_{window}"] = funding_z
        out[f"funding_crowding_reversal_{window}"] = (
            -funding_z * np.sign(out[f"ret_{min(window, 24)}"])
        ).where(crypto_mask)
        out[f"fx_usd_strength_{window}"] = _basket_return(
            out,
            f"ret_{window}",
            {"EURUSD": -1.0, "GBPUSD": -1.0, "AUDUSD": -1.0, "USDJPY": 1.0, "USDCHF": 1.0},
        )
        out[f"fx_jpy_risk_off_{window}"] = _basket_return(
            out, f"ret_{window}", {"USDJPY": -1.0, "AUDJPY": -1.0}
        )

    for col in factor_columns(selected_specs):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def assign_time_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    split_column: str = "split",
) -> pd.DataFrame:
    """Assign chronological train/validation/locked_oos labels by timestamp."""
    if "timestamp" not in frame.columns:
        raise ValueError("timestamp column is required for chronological splits")
    out = frame.copy()
    unique_times = pd.Index(pd.to_datetime(out["timestamp"].drop_duplicates()).sort_values())
    if unique_times.empty:
        out[split_column] = "train"
        return out
    train_cut = max(1, math.floor(len(unique_times) * float(train_fraction)))
    validation_cut = max(
        train_cut + 1, math.floor(len(unique_times) * float(train_fraction + validation_fraction))
    )
    validation_cut = min(validation_cut, len(unique_times))
    train_times = set(unique_times[:train_cut])
    validation_times = set(unique_times[train_cut:validation_cut])

    def _label(value: Any) -> str:
        ts = pd.Timestamp(value)
        if ts in train_times:
            return "train"
        if ts in validation_times:
            return "validation"
        return "locked_oos"

    out[split_column] = out["timestamp"].map(_label)
    return out


def add_forward_return_label(
    frame: pd.DataFrame, *, horizon: int = 4, label_column: str = "forward_return"
) -> pd.DataFrame:
    """Add a simple causal forward-return label for factor IC screening.

    Labels are allowed for report/gate evaluation, but selection code below uses
    only rows whose split is train/validation.
    """
    _require_columns(frame, ("symbol", "close"))
    out = frame.sort_values(["symbol", "timestamp"]).copy()
    future = out.groupby("symbol", sort=False)["close"].shift(-max(1, int(horizon)))
    out[label_column] = (future / out["close"].where(out["close"].abs() > 1e-12)) - 1.0
    return out


def _ic(values: pd.Series, labels: pd.Series) -> float | None:
    joined = pd.concat([values, labels], axis=1).dropna()
    if len(joined) < 3:
        return None
    left_rank = joined.iloc[:, 0].rank()
    right_rank = joined.iloc[:, 1].rank()
    if left_rank.nunique(dropna=True) < 2 or right_rank.nunique(dropna=True) < 2:
        return None
    corr = float(left_rank.corr(right_rank))
    return corr if math.isfinite(corr) else None


def _quantile_spread(values: pd.Series, labels: pd.Series) -> float | None:
    joined = pd.concat([values, labels], axis=1).dropna()
    if len(joined) < 10:
        return None
    ranks = joined.iloc[:, 0].rank(pct=True)
    top = joined.loc[ranks >= 0.8].iloc[:, 1]
    bottom = joined.loc[ranks <= 0.2].iloc[:, 1]
    if top.empty or bottom.empty:
        return None
    spread = float(top.mean() - bottom.mean())
    return spread if math.isfinite(spread) else None


def screen_factor_frame(
    frame: pd.DataFrame,
    *,
    factors: Sequence[str] | None = None,
    label_column: str = "forward_return",
    split_column: str = "split",
    top_n: int = 20,
) -> dict[str, Any]:
    """Screen factors with train/validation-only ranking and OOS report-only stats."""
    _require_columns(frame, (label_column, split_column))
    factor_list = tuple(factors or [col for col in factor_columns() if col in frame.columns])
    rows: list[dict[str, Any]] = []
    for factor in factor_list:
        split_stats: dict[str, dict[str, float | int | None]] = {}
        for split in ("train", "validation", "locked_oos"):
            part = frame[frame[split_column].eq(split)]
            split_stats[split] = {
                "n": int(part[[factor, label_column]].dropna().shape[0]) if factor in part else 0,
                "rank_ic": _ic(part[factor], part[label_column]) if factor in part else None,
                "quantile_spread": _quantile_spread(part[factor], part[label_column])
                if factor in part
                else None,
            }
        train_ic = split_stats["train"]["rank_ic"]
        val_ic = split_stats["validation"]["rank_ic"]
        if train_ic is None or val_ic is None:
            selection_score = float("-inf")
        elif train_ic * val_ic <= 0.0:
            selection_score = min(abs(float(train_ic)), abs(float(val_ic))) * -1.0
        else:
            selection_score = min(abs(float(train_ic)), abs(float(val_ic)))
        rows.append(
            {
                "factor": factor,
                "selection_score": selection_score,
                "selected_using_splits": ["train", "validation"],
                "uses_locked_oos_for_selection": False,
                "split_stats": split_stats,
            }
        )
    rows.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    selected = rows[: max(0, int(top_n))]
    return {
        "artifact_kind": "crypto_fx_alpha_zoo_screen",
        "selection_policy": "train_validation_only_locked_oos_report_only",
        "uses_locked_oos_for_selection": False,
        "selected_factors": selected,
        "all_factors": rows,
    }
