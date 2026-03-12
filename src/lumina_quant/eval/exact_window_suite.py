from __future__ import annotations

import gc
import json
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from lumina_quant.config import BaseConfig
from lumina_quant.market_data import load_data_dict_from_parquet
from lumina_quant.strategy_factory.candidate_library import build_binance_futures_candidates
from lumina_quant.strategy_factory import research_runner as rr
from lumina_quant.symbols import CANONICAL_STRATEGY_TIMEFRAMES, canonical_symbol, canonicalize_symbol_list, normalize_strategy_timeframes


DEFAULT_TRAIN_START = "2025-01-01"
DEFAULT_VAL_START = "2026-01-01"
DEFAULT_OOS_START = "2026-02-01"
DEFAULT_REQUESTED_OOS_END = "2026-03-09"  # end-exclusive for requested March 8 inclusive
DEFAULT_ADAPTIVE_PROBE_START = "2025-01-01"
LOW_RAM_EXCLUDED_STRATEGIES = {"MicroRangeExpansion1sStrategy"}
LOW_RAM_EXCLUDED_TIMEFRAMES = {"1s"}
FEATURE_REQUIRED_STRATEGIES = {
    "CompositeTrendStrategy",
    "PerpCrowdingCarryStrategy",
}
METAL_SYMBOLS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}


def _resolve_low_ram_timeframes(
    requested: list[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    raw = list(requested or list(CANONICAL_STRATEGY_TIMEFRAMES))
    excluded = sorted({str(tf) for tf in raw if str(tf) in LOW_RAM_EXCLUDED_TIMEFRAMES})
    normalized = normalize_strategy_timeframes(
        [tf for tf in raw if tf not in LOW_RAM_EXCLUDED_TIMEFRAMES],
        required=CANONICAL_STRATEGY_TIMEFRAMES,
        strict_subset=True,
    )
    if raw and not normalized:
        raise ValueError(
            "All requested exact-window timeframes were excluded by the low-RAM profile."
        )
    return raw, list(normalized), excluded


def _assert_low_ram_exclusions(rows: list[dict[str, Any]], *, row_kind: str) -> None:
    offenders: list[str] = []
    for row in rows:
        strategy_class = str(row.get("strategy_class") or "")
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "")
        if strategy_class in LOW_RAM_EXCLUDED_STRATEGIES or timeframe in LOW_RAM_EXCLUDED_TIMEFRAMES:
            offenders.append(
                f"{row_kind}: strategy={strategy_class or '<unknown>'} timeframe={timeframe or '<unknown>'}"
            )
    if offenders:
        joined = "; ".join(offenders[:5])
        raise RuntimeError(f"Low-RAM exclusions violated for exact-window suite: {joined}")


def _build_candidate_result_row(
    *,
    candidate: dict[str, Any],
    timeframe: str,
    symbols_for_candidate: list[str],
    metrics: dict[str, dict[str, Any]],
    hurdles: dict[str, Any],
    hard_reject: dict[str, Any],
    streams: dict[str, list[dict[str, Any]]],
    cost_rate: float,
    runtime_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_metadata = dict(candidate.get("metadata") or {})
    merged_metadata = {
        **candidate_metadata,
        **dict(runtime_metadata or {}),
        "cost_rate": float(cost_rate),
    }
    return {
        "candidate_id": str(candidate.get("candidate_id")),
        "name": str(candidate.get("name")),
        "strategy_class": str(candidate.get("strategy_class")),
        "family": str(candidate.get("family") or rr._family_from_strategy(str(candidate.get("strategy_class")))),
        "strategy_timeframe": str(timeframe),
        "symbols": list(symbols_for_candidate),
        "params": dict(candidate.get("params") or {}),
        "notes": str(candidate.get("notes") or ""),
        "tags": [str(tag) for tag in list(candidate.get("tags") or []) if str(tag)],
        "train": metrics["train"],
        "val": metrics["val"],
        "oos": metrics["oos"],
        "hurdle_fields": dict(hurdles or {}),
        "hard_reject_reasons": dict(hard_reject or {}),
        "return_streams": dict(streams or {}),
        "metadata": merged_metadata,
    }


def _min_bars_for_timeframe(
    timeframe: str,
    *,
    window_start: datetime | None = None,
    window_end_exclusive: datetime | None = None,
) -> int:
    token = str(timeframe or "").strip().lower()
    if token == "1d":
        if window_start is not None and window_end_exclusive is not None:
            span_days = max(1, int((window_end_exclusive - window_start).total_seconds() // 86400))
            if span_days <= 40:
                return 28
        return 45
    if token == "4h":
        return 180
    if token == "1h":
        return 240
    return rr._MIN_BARS


def parse_utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    token = str(value).strip()
    if not token:
        raise ValueError("empty datetime token")
    if len(token) == 10:
        return datetime.strptime(token, "%Y-%m-%d").replace(tzinfo=UTC)
    token = token.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(token)
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def resolve_exact_window_suite_windows(
    *,
    train_start: str | datetime | None = None,
    val_start: str | datetime | None = None,
    oos_start: str | datetime | None = None,
    requested_oos_end_exclusive: str | datetime | None = None,
) -> dict[str, datetime]:
    train = parse_utc_datetime(train_start or DEFAULT_TRAIN_START)
    val = parse_utc_datetime(val_start or DEFAULT_VAL_START)
    oos = parse_utc_datetime(oos_start or DEFAULT_OOS_START)
    requested_end = parse_utc_datetime(requested_oos_end_exclusive or DEFAULT_REQUESTED_OOS_END)
    if not (train < val < oos < requested_end):
        raise ValueError(
            "Exact-window windows must satisfy train_start < val_start < oos_start < requested_oos_end_exclusive."
        )
    return {
        "train_start": train,
        "val_start": val,
        "oos_start": oos,
        "requested_oos_end_exclusive": requested_end,
    }


def resolve_coverage_adaptive_windows(
    *,
    symbols: list[str],
    root_path: str,
    exchange: str,
    requested_oos_end_exclusive: str | datetime | None = None,
    probe_start: str | datetime | None = None,
    profile: str = "coverage_adaptive",
    probe_timeframe: str = "1m",
    chunk_days: int = 31,
) -> dict[str, Any]:
    requested_end = parse_utc_datetime(requested_oos_end_exclusive or DEFAULT_REQUESTED_OOS_END)
    probe_start_dt = parse_utc_datetime(probe_start or DEFAULT_ADAPTIVE_PROBE_START)
    coverage_rows, _full_start_symbols, common_end = discover_symbol_coverage(
        symbols=canonicalize_symbol_list(list(symbols or [])),
        root_path=root_path,
        exchange=exchange,
        suite_start=probe_start_dt,
        requested_oos_end_exclusive=requested_end,
        probe_timeframe=probe_timeframe,
        chunk_days=max(7, int(chunk_days)),
    )
    available_rows = [
        row
        for row in coverage_rows
        if row.get("coverage_start") and row.get("coverage_end")
    ]
    if not available_rows:
        raise RuntimeError("Adaptive exact-window profile found no symbol coverage rows with valid bounds.")
    common_start = max(parse_utc_datetime(str(row["coverage_start"])) for row in available_rows)
    if common_end is None:
        common_end = min(parse_utc_datetime(str(row["coverage_end"])) for row in available_rows)
    actual_end_exclusive = min(common_end + timedelta(milliseconds=1), requested_end)
    total_days = max(1, int((actual_end_exclusive - common_start).total_seconds() // 86400))
    profile_token = str(profile or "coverage_adaptive").strip().lower().replace("-", "_")
    if profile_token in {
        "metals",
        "mixed_assets",
        "metals_4h",
        "mixed_assets_4h",
    }:
        min_train_days, min_val_days, min_oos_days = 16, 8, 10
        phase_fraction = 0.22
        oos_fraction = 0.28
    elif profile_token in {
        "metals_1d",
        "mixed_assets_1d",
        "coverage_adaptive_1d",
    }:
        min_train_days, min_val_days, min_oos_days = 14, 7, 10
        phase_fraction = 0.20
        oos_fraction = 0.30
    elif profile_token in {"coverage_adaptive_4h"}:
        min_train_days, min_val_days, min_oos_days = 18, 8, 10
        phase_fraction = 0.20
        oos_fraction = 0.28
    else:
        min_train_days, min_val_days, min_oos_days = 28, 14, 14
        phase_fraction = 0.25
        oos_fraction = 0.25
    minimum_required_days = min_train_days + min_val_days + min_oos_days
    if total_days < minimum_required_days:
        raise RuntimeError(
            "Adaptive exact-window profile found insufficient overlapping history. "
            f"profile={profile_token!r} requires at least {minimum_required_days} days, found {total_days}."
        )
    oos_days = max(min_oos_days, min(28, round(total_days * oos_fraction)))
    val_days = max(min_val_days, min(28, round(total_days * phase_fraction)))
    train_days = total_days - val_days - oos_days
    if train_days < min_train_days:
        deficit = min_train_days - train_days
        reduce_each = math.ceil(deficit / 2.0)
        val_days = max(min_val_days, val_days - reduce_each)
        oos_days = max(min_oos_days, oos_days - (deficit - max(0, reduce_each)))
        train_days = total_days - val_days - oos_days
    if train_days < min_train_days:
        raise RuntimeError(
            "Adaptive exact-window profile could not allocate train/val/oos windows "
            f"for profile={profile_token!r} within {total_days} overlapping days."
        )
    train_start = common_start
    val_start = train_start + timedelta(days=train_days)
    oos_start = val_start + timedelta(days=val_days)
    if not (train_start < val_start < oos_start < actual_end_exclusive):
        raise RuntimeError("Adaptive exact-window profile produced invalid ordered windows.")
    return {
        "profile": profile_token,
        "train_start": train_start,
        "val_start": val_start,
        "oos_start": oos_start,
        "requested_oos_end_exclusive": actual_end_exclusive,
        "coverage_rows": coverage_rows,
        "common_start": common_start,
        "common_end": common_end,
        "total_days": total_days,
        "allocation_days": {
            "train": train_days,
            "val": val_days,
            "oos": oos_days,
        },
    }


def _to_naive_utc(dt: datetime) -> datetime:
    return parse_utc_datetime(dt).astimezone(UTC).replace(tzinfo=None)


def _np_datetime(dt: datetime) -> np.datetime64:
    return np.datetime64(_to_naive_utc(dt), "ms")


def half_open_slice_indices(timestamps: np.ndarray, start: datetime, end: datetime) -> slice:
    values = np.asarray(timestamps, dtype="datetime64[ms]")
    left = int(np.searchsorted(values, _np_datetime(start), side="left"))
    right = int(np.searchsorted(values, _np_datetime(end), side="left"))
    return slice(left, right)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _daily_stream_from_timestamps(timestamps: np.ndarray, returns: np.ndarray) -> list[dict[str, float]]:
    days = np.asarray(timestamps, dtype="datetime64[D]")
    values = np.asarray(returns, dtype=float)
    compounded: dict[int, float] = {}
    for day, value in zip(days, values, strict=True):
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        day_ms = int(day.astype("datetime64[ms]").astype(np.int64))
        compounded[day_ms] = compounded.get(day_ms, 1.0) * (1.0 + numeric)
    return [
        {"t": float(day_ms), "v": float(equity - 1.0)}
        for day_ms, equity in sorted(compounded.items())
    ]


def compound_returns(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def aggregate_stream_by_period(stream: list[dict[str, float]], *, period: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        epoch_ms = int(point.get("t", 0))
        dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
        if period == "day":
            key = dt.date().isoformat()
        elif period == "month":
            key = f"{dt.year:04d}-{dt.month:02d}"
        else:
            raise ValueError(f"unsupported period: {period}")
        buckets[key].append(float(point.get("v", 0.0)))
    rows = []
    for key in sorted(buckets):
        rows.append({"period": key, "return": compound_returns(buckets[key])})
    return rows


def _daily_return_map(stream: list[dict[str, float]]) -> dict[str, float]:
    rows = aggregate_stream_by_period(stream, period="day")
    return {str(row["period"]): float(row["return"]) for row in rows}


def _daily_returns_array(stream: list[dict[str, float]]) -> np.ndarray:
    rows = aggregate_stream_by_period(stream, period="day")
    return np.asarray([float(row["return"]) for row in rows], dtype=float)


def _metrics_daily(returns: np.ndarray) -> dict[str, float]:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
        }
    total_return = float(np.prod(1.0 + arr) - 1.0)
    years = max(arr.size / 365.0, 1.0 / 365.0)
    cagr = float(math.exp(math.log1p(max(-0.999999, total_return)) / years) - 1.0)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    sharpe = 0.0 if sigma <= 1e-12 else (mu / sigma) * math.sqrt(365.0)
    downside = arr[arr < 0.0]
    dsigma = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = 0.0 if dsigma <= 1e-12 else (mu / dsigma) * math.sqrt(365.0)
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(equity)
    dd = 1.0 - np.divide(equity, np.maximum(peaks, 1e-12))
    max_dd = float(np.max(dd)) if dd.size else 0.0
    calmar = 0.0 if max_dd <= 1e-12 else cagr / max_dd
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "volatility": float(sigma * math.sqrt(365.0)),
    }


def _validation_score(row: dict[str, Any], *, scoring_config: dict[str, Any] | None = None) -> float:
    cfg = rr._resolve_score_config(scoring_config)
    weights = dict(cfg["candidate_rank_score_weights"])
    metrics = dict(row.get("val") or {})
    return float(
        (float(weights["sharpe_weight"]) * float(metrics.get("sharpe", 0.0)))
        + (float(weights["deflated_sharpe_weight"]) * float(metrics.get("deflated_sharpe", 0.0)))
        - (float(weights["pbo_penalty"]) * float(metrics.get("pbo", 1.0)))
        + (float(weights["return_weight"]) * float(metrics.get("return", 0.0)))
        - (
            float(weights["turnover_penalty"])
            * max(0.0, float(metrics.get("turnover", 0.0)) - float(weights["turnover_threshold"]))
        )
        - (float(weights["drawdown_penalty"]) * float(metrics.get("mdd", 0.0)))
        - rr._candidate_instability_penalty(row, scoring_config=scoring_config)
    )


def _strict_load_frame(
    *,
    root_path: str,
    exchange: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> pl.DataFrame:
    frames = load_data_dict_from_parquet(
        root_path,
        exchange=exchange,
        symbol_list=[symbol],
        timeframe=timeframe,
        start_date=_to_naive_utc(start),
        end_date=_to_naive_utc(end),
        chunk_days=max(1, int(chunk_days)),
        warmup_bars=0,
        data_mode="legacy",
    )
    return frames.get(symbol, pl.DataFrame())


def _coerce_frame_datetime_bounds(frame: pl.DataFrame) -> tuple[datetime | None, datetime | None]:
    if frame.is_empty() or "datetime" not in frame.columns:
        return None, None
    dt = frame.get_column("datetime")
    start = dt.min()
    end = dt.max()
    if start is None or end is None:
        return None, None
    start_dt = parse_utc_datetime(start.to_pydatetime() if hasattr(start, "to_pydatetime") else start)
    end_dt = parse_utc_datetime(end.to_pydatetime() if hasattr(end, "to_pydatetime") else end)
    return start_dt, end_dt


def discover_symbol_coverage(
    *,
    symbols: list[str],
    root_path: str,
    exchange: str,
    suite_start: datetime,
    requested_oos_end_exclusive: datetime,
    probe_timeframe: str = "1m",
    chunk_days: int = 31,
) -> tuple[list[dict[str, Any]], list[str], datetime | None]:
    rows: list[dict[str, Any]] = []
    full_start_symbols: list[str] = []
    end_candidates: list[datetime] = []
    for symbol in symbols:
        frame = _strict_load_frame(
            root_path=root_path,
            exchange=exchange,
            symbol=symbol,
            timeframe=probe_timeframe,
            start=suite_start,
            end=requested_oos_end_exclusive,
            chunk_days=chunk_days,
        )
        start_dt, end_dt = _coerce_frame_datetime_bounds(frame)
        full_start = bool(start_dt is not None and start_dt <= suite_start)
        rows.append(
            {
                "symbol": symbol,
                "coverage_start": start_dt.isoformat() if start_dt else None,
                "coverage_end": end_dt.isoformat() if end_dt else None,
                "full_start_coverage": full_start,
                "requested_oos_end": requested_oos_end_exclusive.isoformat(),
            }
        )
        if full_start and end_dt is not None:
            full_start_symbols.append(symbol)
            end_candidates.append(end_dt)
    actual_common_end = min(end_candidates) if end_candidates else None
    return rows, full_start_symbols, actual_common_end


def _prefixed_ohlcv(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    return frame.select(
        [
            pl.col("datetime"),
            pl.col("open").alias(f"{symbol}:open"),
            pl.col("high").alias(f"{symbol}:high"),
            pl.col("low").alias(f"{symbol}:low"),
            pl.col("close").alias(f"{symbol}:close"),
            pl.col("volume").alias(f"{symbol}:volume"),
        ]
    )


def strict_align_bundles(
    *,
    symbol_frames: dict[str, pl.DataFrame],
    feature_cache: dict[str, pl.DataFrame],
    benchmark_frame: pl.DataFrame | None,
    window_start: datetime,
    window_end_exclusive: datetime,
    timeframe: str,
) -> dict[str, np.ndarray] | None:
    ordered = sorted(symbol_frames.items())
    if not ordered:
        return None
    min_bars = _min_bars_for_timeframe(
        timeframe,
        window_start=window_start,
        window_end_exclusive=window_end_exclusive,
    )
    merged: pl.DataFrame | None = None
    for symbol, frame in ordered:
        if frame.is_empty() or frame.height < min_bars:
            return None
        prefixed = _prefixed_ohlcv(frame, symbol)
        merged = prefixed if merged is None else merged.join(prefixed, on="datetime", how="inner")
        if merged is None or merged.is_empty():
            return None
    assert merged is not None
    merged = merged.filter(
        (pl.col("datetime") >= pl.lit(_to_naive_utc(window_start)).cast(pl.Datetime(time_unit="ms")))
        & (pl.col("datetime") < pl.lit(_to_naive_utc(window_end_exclusive)).cast(pl.Datetime(time_unit="ms")))
    ).sort("datetime")
    if merged.height < min_bars:
        return None
    if benchmark_frame is not None and not benchmark_frame.is_empty():
        bench = benchmark_frame.select([pl.col("datetime"), pl.col("close").alias("__bench_close")])
        merged = merged.join(bench, on="datetime", how="left")
    aligned: dict[str, np.ndarray] = {"datetime": merged.get_column("datetime").to_numpy()}
    for symbol, _ in ordered:
        for field in ("open", "high", "low", "close", "volume"):
            aligned[f"{symbol}:{field}"] = merged.get_column(f"{symbol}:{field}").to_numpy()
        feature_frame = feature_cache.get(symbol)
        if feature_frame is None or feature_frame.is_empty():
            continue
        filtered = feature_frame.filter(
            (pl.col("datetime") >= pl.lit(_to_naive_utc(window_start)).cast(pl.Datetime(time_unit="ms")))
            & (pl.col("datetime") < pl.lit(_to_naive_utc(window_end_exclusive)).cast(pl.Datetime(time_unit="ms")))
        ).sort("datetime")
        if filtered.is_empty():
            continue
        joined = merged.select([pl.col("datetime")]).join_asof(
            filtered.select(["datetime", *rr._FEATURE_POINT_COLUMNS]).sort("datetime"),
            on="datetime",
            strategy="backward",
        )
        for field in rr._FEATURE_POINT_COLUMNS:
            aligned[f"{symbol}:{field}"] = joined.get_column(field).to_numpy()
        support = rr._crowding_support_series(
            funding_rate=np.asarray(joined.get_column("funding_rate").to_numpy(), dtype=float),
            open_interest=np.asarray(joined.get_column("open_interest").to_numpy(), dtype=float),
            mark_price=np.asarray(joined.get_column("mark_price").to_numpy(), dtype=float),
            index_price=np.asarray(joined.get_column("index_price").to_numpy(), dtype=float),
            liquidation_long_notional=np.asarray(joined.get_column("liquidation_long_notional").to_numpy(), dtype=float),
            liquidation_short_notional=np.asarray(joined.get_column("liquidation_short_notional").to_numpy(), dtype=float),
        )
        for key, values in support.items():
            aligned[f"{symbol}:{key}"] = values
    if "__bench_close" in merged.columns:
        aligned["benchmark_close"] = merged.get_column("__bench_close").fill_null(strategy="forward").fill_null(strategy="backward").to_numpy()
    return aligned


def _split_periods(
    *,
    train_start: datetime,
    val_start: datetime,
    oos_start: datetime,
    actual_oos_end_exclusive: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    return {
        "train": (train_start, val_start),
        "val": (val_start, oos_start),
        "oos": (oos_start, actual_oos_end_exclusive),
    }


def _monthly_btc_thresholds(
    *,
    root_path: str,
    exchange: str,
    window_start: datetime,
    actual_oos_end_exclusive: datetime,
) -> dict[str, dict[str, float]]:
    frame = _strict_load_frame(
        root_path=root_path,
        exchange=exchange,
        symbol="BTC/USDT",
        timeframe="1d",
        start=window_start,
        end=actual_oos_end_exclusive,
        chunk_days=62,
    )
    if frame.is_empty():
        return {}
    closes = frame.get_column("close").to_list()
    dts = frame.get_column("datetime").to_list()
    by_month: dict[str, list[float]] = defaultdict(list)
    for dt_raw, close in zip(dts, closes, strict=True):
        dt = parse_utc_datetime(dt_raw.to_pydatetime() if hasattr(dt_raw, "to_pydatetime") else dt_raw)
        key = f"{dt.year:04d}-{dt.month:02d}"
        by_month[key].append(float(close))
    out: dict[str, dict[str, float]] = {}
    for key, values in sorted(by_month.items()):
        if not values:
            continue
        btc_ret = 0.0 if len(values) < 2 else float(values[-1] / max(values[0], 1e-12) - 1.0)
        out[key] = {
            "btc_buy_hold_return": btc_ret,
            "threshold": max(0.02, btc_ret),
        }
    return out


def _monthly_hurdle_rows(stream: list[dict[str, float]], thresholds: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    realized = {row["period"]: float(row["return"]) for row in aggregate_stream_by_period(stream, period="month")}
    rows: list[dict[str, Any]] = []
    for month in sorted(realized):
        benchmark = dict(thresholds.get(month) or {})
        actual = float(realized.get(month, 0.0))
        threshold = float(benchmark.get("threshold", 0.02))
        btc_ret = float(benchmark.get("btc_buy_hold_return", 0.0))
        strict_pass = bool(actual >= threshold)
        btc_pass = bool(actual >= btc_ret)
        rows.append(
            {
                "month": month,
                "strategy_return": actual,
                "btc_buy_hold_return": btc_ret,
                "threshold": threshold,
                "excess_vs_threshold": actual - threshold,
                "strict_pass": strict_pass,
                "btc_pass": btc_pass,
                "pass": strict_pass,
            }
        )
    return rows


def _latest_three_month_rows(*row_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in row_groups:
        for row in list(group or []):
            month = str(row.get("month") or "").strip()
            if month:
                merged[month] = dict(row)
    return [merged[month] for month in sorted(merged)[-3:]]


def _recent_three_month_two_pct_pass(*row_groups: list[dict[str, Any]]) -> bool:
    latest_rows = _latest_three_month_rows(*row_groups)
    if len(latest_rows) < 3:
        return False
    return all(float(row.get("strategy_return", 0.0)) >= 0.02 for row in latest_rows)


def _portfolio_weights(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not best_rows:
        return []
    pool = [row for row in best_rows if bool(row.get("promoted"))]
    basis = "promoted_only"
    if not pool:
        pool = [row for row in best_rows if bool(row.get("candidate_pool_eligible") or row.get("btc_beating_candidate"))]
        basis = "candidate_pool_candidates"
    if not pool:
        pool = list(best_rows)
        basis = "best_per_strategy_fallback"

    raw: dict[str, float] = {}
    for row in pool:
        cid = str(row.get("candidate_id"))
        val_daily = _daily_returns_array(list((row.get("return_streams") or {}).get("val") or []))
        vol = float(np.std(val_daily, ddof=1)) if val_daily.size > 1 else 0.0
        inv_vol = 1.0 / max(vol, 1e-6)
        quality = max(0.05, float((row.get("val") or {}).get("deflated_sharpe", 0.0)) + 0.5)
        raw[cid] = quality * inv_vol
    total = float(sum(raw.values()))
    weights = {cid: (value / total if total > 0 else 1.0 / max(1, len(raw))) for cid, value in raw.items()}

    cap = max(0.35, 1.0 / max(1, len(weights)))
    capped = True
    while capped:
        capped = False
        over = {cid: w for cid, w in weights.items() if w > cap}
        if not over:
            break
        excess = sum(w - cap for w in over.values())
        for cid in over:
            weights[cid] = cap
        under = [cid for cid, w in weights.items() if w < cap - 1e-12]
        if under and excess > 0:
            under_total = sum(weights[cid] for cid in under)
            if under_total <= 0:
                add = excess / len(under)
                for cid in under:
                    weights[cid] += add
            else:
                for cid in under:
                    weights[cid] += excess * (weights[cid] / under_total)
            capped = True
    norm = float(sum(weights.values()))
    if norm > 0:
        for cid in list(weights):
            weights[cid] /= norm
    weight_rows = []
    for row in pool:
        cid = str(row.get("candidate_id"))
        weight_rows.append(
            {
                "candidate_id": cid,
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "family": row.get("family"),
                "symbols": list(row.get("symbols") or []),
                "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                "weight": float(weights.get(cid, 0.0)),
                "basis": basis,
            }
        )
    return sorted(weight_rows, key=lambda item: item["weight"], reverse=True)


def _portfolio_stream(best_rows: list[dict[str, Any]], weights: list[dict[str, Any]], split: str) -> list[dict[str, float]]:
    if not weights:
        return []
    weight_map = {str(row["candidate_id"]): float(row["weight"]) for row in weights if float(row.get("weight", 0.0)) > 0.0}
    day_map: dict[str, float] = defaultdict(float)
    timestamp_lookup: dict[str, int] = {}
    for row in best_rows:
        cid = str(row.get("candidate_id"))
        weight = weight_map.get(cid, 0.0)
        if weight <= 0.0:
            continue
        daily_rows = aggregate_stream_by_period(list((row.get("return_streams") or {}).get(split) or []), period="day")
        for item in daily_rows:
            day = str(item["period"])
            day_map[day] += weight * float(item["return"])
            if day not in timestamp_lookup:
                day_dt = parse_utc_datetime(day)
                timestamp_lookup[day] = int(day_dt.timestamp() * 1000)
    return [
        {"t": timestamp_lookup[day], "v": float(day_map[day])}
        for day in sorted(day_map)
    ]


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Exact-Window Validation Suite",
        "",
        "## Windows",
        f"- Train: `{summary['windows']['train_start']}` → `{summary['windows']['train_end_exclusive']}`",
        f"- Validation: `{summary['windows']['val_start']}` → `{summary['windows']['val_end_exclusive']}`",
        f"- OOS requested end-exclusive: `{summary['windows']['requested_oos_end_exclusive']}`",
        f"- OOS actual end-exclusive: `{summary['windows']['actual_oos_end_exclusive']}`",
        f"- Actual max timestamp used: `{summary['windows']['actual_max_timestamp']}`",
        "",
        "## Universe",
        f"- Eligible symbols: {', '.join(summary.get('eligible_symbols') or [])}",
        f"- Excluded symbols: {', '.join(summary.get('excluded_symbols') or [])}",
        f"- Candidate count: {int(summary.get('candidate_count', 0))}",
        f"- Evaluated count: {int(summary.get('evaluated_count', 0))}",
        "",
        "## Best Per Strategy",
        "",
        "| Strategy | Name | TF | Val Score | Promoted | Val Hurdle | OOS Return | OOS Sharpe |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("best_per_strategy") or []:
        lines.append(
            f"| {row.get('strategy_class','')} | {row.get('name','')} | {row.get('strategy_timeframe','')} | "
            f"{float(row.get('validation_score', 0.0)):.3f} | {int(bool(row.get('promoted')))} | "
            f"{int(bool(row.get('validation_hurdle_pass')))} | {float((row.get('oos') or {}).get('return', 0.0)):.2%} | "
            f"{float((row.get('oos') or {}).get('sharpe', 0.0)):.3f} |"
        )
    lines.extend([
        "",
        "## Portfolio",
        "",
        f"- Construction basis: `{summary.get('portfolio', {}).get('construction_basis', '')}`",
        "",
        "| Name | Strategy | Weight | OOS Return | OOS Sharpe |",
        "|---|---|---:|---:|---:|",
    ])
    for row in summary.get("portfolio", {}).get("weights") or []:
        lines.append(
            f"| {row.get('name','')} | {row.get('strategy_class','')} | {float(row.get('weight',0.0)):.2%} | "
            f"{float(row.get('oos_return',0.0)):.2%} | {float(row.get('oos_sharpe',0.0)):.3f} |"
        )
    lines.extend(["", "## Portfolio Monthly Hurdle", ""])
    for row in summary.get("portfolio", {}).get("monthly_hurdle") or []:
        lines.append(
            f"- {row.get('month')}: return={float(row.get('strategy_return',0.0)):.2%}, "
            f"btc={float(row.get('btc_buy_hold_return',0.0)):.2%}, threshold={float(row.get('threshold',0.0)):.2%}, "
            f"pass={bool(row.get('pass'))}"
        )
    return "\n".join(lines) + "\n"


def run_exact_window_suite(
    *,
    output_dir: str = "var/reports/exact_window_backtests",
    score_config: dict[str, Any] | None = None,
    timeframes: list[str] | None = None,
    symbols: list[str] | None = None,
    chunk_days: int = 7,
    allow_metals: bool = False,
    train_start: str | datetime | None = None,
    val_start: str | datetime | None = None,
    oos_start: str | datetime | None = None,
    requested_oos_end_exclusive: str | datetime | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root_path = str(getattr(BaseConfig, "MARKET_DATA_PARQUET_PATH", "data/market_parquet"))
    exchange = str(getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance") or "binance")
    resolved_windows = resolve_exact_window_suite_windows(
        train_start=train_start,
        val_start=val_start,
        oos_start=oos_start,
        requested_oos_end_exclusive=requested_oos_end_exclusive,
    )
    train_start = resolved_windows["train_start"]
    val_start = resolved_windows["val_start"]
    oos_start = resolved_windows["oos_start"]
    requested_oos_end_exclusive = resolved_windows["requested_oos_end_exclusive"]

    requested_symbols = canonicalize_symbol_list(list(symbols or BaseConfig.SYMBOLS))
    requested_timeframes_raw, requested_timeframes, excluded_requested_timeframes = (
        _resolve_low_ram_timeframes(list(timeframes or list(CANONICAL_STRATEGY_TIMEFRAMES)))
    )
    coverage_rows, _full_start_symbols, common_end = discover_symbol_coverage(
        symbols=requested_symbols,
        root_path=root_path,
        exchange=exchange,
        suite_start=train_start,
        requested_oos_end_exclusive=requested_oos_end_exclusive,
        chunk_days=max(7, int(chunk_days)),
    )
    if common_end is None:
        raise RuntimeError("No symbol has full coverage from train start through the requested window.")
    actual_max_timestamp = min(common_end, requested_oos_end_exclusive - timedelta(milliseconds=1))
    actual_oos_end_exclusive = actual_max_timestamp + timedelta(milliseconds=1)
    eligible_symbols = [
        row["symbol"]
        for row in coverage_rows
        if row["full_start_coverage"] and row.get("coverage_end") and parse_utc_datetime(row["coverage_end"]) >= actual_max_timestamp
    ]
    if not allow_metals:
        eligible_symbols = [
            symbol for symbol in eligible_symbols if canonical_symbol(symbol) not in METAL_SYMBOLS
        ]
    excluded_symbols = [symbol for symbol in requested_symbols if symbol not in eligible_symbols]
    if not eligible_symbols:
        raise RuntimeError("No eligible symbols remain after full-coverage filtering.")

    normalized_timeframes = list(requested_timeframes)
    candidates = [item.to_dict() for item in build_binance_futures_candidates(timeframes=normalized_timeframes, symbols=eligible_symbols)]
    candidates = [row for row in candidates if str(row.get("strategy_class")) not in LOW_RAM_EXCLUDED_STRATEGIES]
    _assert_low_ram_exclusions(candidates, row_kind="candidate")
    candidate_count = len(candidates)
    if candidate_count <= 0:
        raise RuntimeError("No candidates generated for exact-window suite.")

    scoring_scope = None
    if isinstance(score_config, dict):
        scoring_scope = dict(score_config.get("candidate_research") or score_config)
    resolved_scoring = rr._resolve_score_config(scoring_scope)
    feature_start = train_start.isoformat().replace("+00:00", "Z")
    feature_end = actual_max_timestamp.isoformat().replace("+00:00", "Z")
    feature_symbol_cache: dict[str, pl.DataFrame] = {}
    results: list[dict[str, Any]] = []
    periods = _split_periods(
        train_start=train_start,
        val_start=val_start,
        oos_start=oos_start,
        actual_oos_end_exclusive=actual_oos_end_exclusive,
    )

    if progress_callback is not None:
        progress_callback(
            "suite_start",
            {
                "candidate_count": candidate_count,
                "requested_timeframes": normalized_timeframes,
                "eligible_symbols": eligible_symbols,
            },
        )

    for timeframe in sorted({str(row.get("strategy_timeframe") or row.get("timeframe") or "1m") for row in candidates}):
        tf_candidates = [row for row in candidates if str(row.get("strategy_timeframe") or row.get("timeframe") or "1m") == timeframe]
        benchmark_frame = _strict_load_frame(
            root_path=root_path,
            exchange=exchange,
            symbol="BTC/USDT",
            timeframe=timeframe,
            start=train_start,
            end=actual_oos_end_exclusive,
            chunk_days=max(1, int(chunk_days)),
        )
        if progress_callback is not None:
            progress_callback(
                "timeframe_start",
                {
                    "timeframe": timeframe,
                    "candidate_count": len(tf_candidates),
                },
            )
        aligned_cache: dict[tuple[str, ...], dict[str, np.ndarray] | None] = {}
        for index, candidate in enumerate(tf_candidates, start=1):
            symbols_for_candidate = canonicalize_symbol_list(list(candidate.get("symbols") or []))
            if any(symbol not in eligible_symbols for symbol in symbols_for_candidate):
                continue
            strategy_class = str(candidate.get("strategy_class") or "")
            needs_features = strategy_class in FEATURE_REQUIRED_STRATEGIES
            cache_key = tuple(symbols_for_candidate)
            aligned = aligned_cache.get(cache_key)
            if cache_key not in aligned_cache:
                symbol_frames: dict[str, pl.DataFrame] = {}
                for symbol in symbols_for_candidate:
                    frame = _strict_load_frame(
                        root_path=root_path,
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        start=train_start,
                        end=actual_oos_end_exclusive,
                        chunk_days=max(1, int(chunk_days)),
                    )
                    if frame.is_empty():
                        symbol_frames = {}
                        break
                    symbol_frames[symbol] = frame
                if not symbol_frames:
                    aligned_cache[cache_key] = None
                    continue
                feature_subset: dict[str, pl.DataFrame] = {}
                if needs_features:
                    for symbol in symbols_for_candidate:
                        if symbol not in feature_symbol_cache:
                            feature_symbol_cache[symbol] = rr._load_feature_cache(
                                symbols=[symbol],
                                start_date=feature_start,
                                end_date=feature_end,
                            ).get(symbol, pl.DataFrame())
                        feature_subset[symbol] = feature_symbol_cache[symbol]
                aligned = strict_align_bundles(
                    symbol_frames=symbol_frames,
                    feature_cache=feature_subset,
                    benchmark_frame=benchmark_frame,
                    window_start=train_start,
                    window_end_exclusive=actual_oos_end_exclusive,
                    timeframe=timeframe,
                )
                aligned_cache[cache_key] = aligned
            if aligned is None:
                continue
            returns_raw, turnover, exposure, meta = rr._strategy_signal(candidate, aligned=aligned, symbols=symbols_for_candidate)
            timestamps = np.asarray(aligned["datetime"], dtype="datetime64[ms]")
            cost_rate = rr._candidate_cost_rate(candidate)
            returns = np.asarray(returns_raw, dtype=float) - (np.asarray(turnover, dtype=float) * cost_rate)
            bench_close = (
                np.asarray(aligned["benchmark_close"], dtype=float)
                if "benchmark_close" in aligned
                else np.zeros_like(returns)
            )
            benchmark_returns = rr._returns_from_close(bench_close) if bench_close.size else np.zeros_like(returns)
            slices = {name: half_open_slice_indices(timestamps, start, end) for name, (start, end) in periods.items()}
            if any((sl.stop - sl.start) <= 0 for sl in slices.values()):
                continue
            metrics = {}
            streams = {}
            for name, sl in slices.items():
                metrics[name] = rr._compute_metrics(
                    returns[sl],
                    turnover=np.asarray(turnover, dtype=float)[sl],
                    exposure=np.asarray(exposure, dtype=float)[sl],
                    benchmark_returns=np.asarray(benchmark_returns, dtype=float)[sl],
                    periods_per_year=int(rr._PERIODS_PER_YEAR.get(timeframe, 365)),
                    num_trials=max(1, candidate_count),
                )
                streams[name] = _daily_stream_from_timestamps(timestamps[sl], returns[sl])
            hurdles, _, hard_reject = rr._hurdle_fields(
                metrics["train"],
                metrics["val"],
                metrics["oos"],
                scoring_config=resolved_scoring,
            )
            results.append(
                _build_candidate_result_row(
                    candidate=candidate,
                    timeframe=timeframe,
                    symbols_for_candidate=symbols_for_candidate,
                    metrics=metrics,
                    hurdles=hurdles,
                    hard_reject=hard_reject,
                    streams=streams,
                    cost_rate=float(cost_rate),
                    runtime_metadata=dict(meta or {}),
                )
            )
            if progress_callback is not None:
                progress_callback(
                    "candidate_evaluated",
                    {
                        "timeframe": timeframe,
                        "candidate_index": index,
                        "candidate_count": len(tf_candidates),
                        "strategy_class": str(candidate.get("strategy_class") or ""),
                        "evaluated_count": len(results),
                    },
                )
        if progress_callback is not None:
            progress_callback(
                "timeframe_complete",
                {
                    "timeframe": timeframe,
                    "candidate_count": len(tf_candidates),
                    "evaluated_count": len(results),
                },
            )
        feature_symbol_cache.clear()
        gc.collect()

    if not results:
        raise RuntimeError("Exact-window suite produced no evaluated candidates.")
    _assert_low_ram_exclusions(results, row_kind="result")

    thresholds = _monthly_btc_thresholds(
        root_path=root_path,
        exchange=exchange,
        window_start=val_start,
        actual_oos_end_exclusive=actual_oos_end_exclusive,
    )

    best_by_strategy: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row.get("strategy_class"))].append(row)

    for strategy_class, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda item: _validation_score(item, scoring_config=resolved_scoring), reverse=True)
        top = dict(ranked[0])
        top["validation_score"] = _validation_score(top, scoring_config=resolved_scoring)
        val_months = _monthly_hurdle_rows(list((top.get("return_streams") or {}).get("val") or []), thresholds)
        oos_months = _monthly_hurdle_rows(list((top.get("return_streams") or {}).get("oos") or []), thresholds)
        val_hurdle_pass = all(bool(row.get("strict_pass")) for row in val_months if str(row.get("month", "")).startswith("2026-01"))
        val_btc_hurdle_pass = all(bool(row.get("btc_pass")) for row in val_months if str(row.get("month", "")).startswith("2026-01"))
        oos_btc_hurdle_pass = all(bool(row.get("btc_pass")) for row in oos_months)
        recent_three_months = _latest_three_month_rows(val_months, oos_months)
        recent_three_month_two_pct_pass = _recent_three_month_two_pct_pass(val_months, oos_months)
        train_pass = bool((top.get("hurdle_fields") or {}).get("train", {}).get("pass", False))
        val_pass = bool((top.get("hurdle_fields") or {}).get("val", {}).get("pass", False))
        top["validation_monthly_hurdle"] = val_months
        top["oos_monthly_hurdle"] = oos_months
        top["recent_three_months"] = recent_three_months
        top["validation_hurdle_pass"] = bool(val_hurdle_pass)
        top["validation_btc_hurdle_pass"] = bool(val_btc_hurdle_pass)
        top["oos_btc_hurdle_pass"] = bool(oos_btc_hurdle_pass)
        top["recent_three_month_two_pct_pass"] = bool(recent_three_month_two_pct_pass)
        top["promoted"] = train_pass and val_pass and bool(val_hurdle_pass)
        top["btc_beating_candidate"] = train_pass and val_pass and bool(val_btc_hurdle_pass)
        top["three_month_two_pct_candidate"] = train_pass and val_pass and bool(recent_three_month_two_pct_pass)
        top["candidate_pool_eligible"] = bool(
            top.get("promoted")
            or top.get("btc_beating_candidate")
            or top.get("three_month_two_pct_candidate")
        )
        best_by_strategy.append(top)

    weights = _portfolio_weights(best_by_strategy)
    for row in weights:
        cid = str(row.get("candidate_id"))
        source = next((item for item in best_by_strategy if str(item.get("candidate_id")) == cid), None)
        if source is not None:
            row["oos_return"] = float((source.get("oos") or {}).get("return", 0.0))
            row["oos_sharpe"] = float((source.get("oos") or {}).get("sharpe", 0.0))
    portfolio_streams = {split: _portfolio_stream(best_by_strategy, weights, split) for split in ("train", "val", "oos")}
    portfolio = {
        "construction_basis": weights[0].get("basis") if weights else "none",
        "selection": {
            "fit_split": "val",
            "report_split": "oos",
            "selection_basis": "validation_only",
        },
        "weights": weights,
        "train": _metrics_daily(_daily_returns_array(portfolio_streams["train"])),
        "val": _metrics_daily(_daily_returns_array(portfolio_streams["val"])),
        "oos": _metrics_daily(_daily_returns_array(portfolio_streams["oos"])),
        "monthly_hurdle": _monthly_hurdle_rows(portfolio_streams["oos"], thresholds),
        "return_streams": portfolio_streams,
    }

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "windows": {
            "train_start": train_start.isoformat(),
            "train_end_exclusive": val_start.isoformat(),
            "val_start": val_start.isoformat(),
            "val_end_exclusive": oos_start.isoformat(),
            "requested_oos_end_exclusive": requested_oos_end_exclusive.isoformat(),
            "actual_oos_end_exclusive": actual_oos_end_exclusive.isoformat(),
            "actual_max_timestamp": actual_max_timestamp.isoformat(),
        },
        "execution_profile": {
            "requested_symbols": requested_symbols,
            "requested_timeframes_raw": requested_timeframes_raw,
            "requested_timeframes": normalized_timeframes,
            "low_ram_profile": True,
            "allow_metals": bool(allow_metals),
            "custom_windows": bool(
                train_start != parse_utc_datetime(DEFAULT_TRAIN_START)
                or val_start != parse_utc_datetime(DEFAULT_VAL_START)
                or oos_start != parse_utc_datetime(DEFAULT_OOS_START)
                or requested_oos_end_exclusive != parse_utc_datetime(DEFAULT_REQUESTED_OOS_END)
            ),
        },
        "eligible_symbols": eligible_symbols,
        "excluded_symbols": excluded_symbols,
        "coverage": coverage_rows,
        "candidate_count": candidate_count,
        "evaluated_count": len(results),
        "promoted_count": sum(1 for row in best_by_strategy if bool(row.get("promoted"))),
        "btc_beating_candidate_count": sum(1 for row in best_by_strategy if bool(row.get("btc_beating_candidate"))),
        "three_month_two_pct_candidate_count": sum(
            1 for row in best_by_strategy if bool(row.get("three_month_two_pct_candidate"))
        ),
        "provisional_candidate_count": sum(
            1 for row in best_by_strategy if bool(row.get("candidate_pool_eligible")) and not bool(row.get("promoted"))
        ),
        "candidate_pool_count": sum(1 for row in best_by_strategy if bool(row.get("candidate_pool_eligible"))),
        "best_per_strategy": best_by_strategy,
        "portfolio": portfolio,
        "monthly_thresholds": thresholds,
        "notes": {
            "local_data_gap": (
                None
                if actual_oos_end_exclusive >= requested_oos_end_exclusive
                else (
                    "Requested OOS through "
                    f"{(requested_oos_end_exclusive - timedelta(milliseconds=1)).isoformat()} inclusive, "
                    "but local on-disk market data ends on "
                    f"{actual_max_timestamp.isoformat()}, so the suite clamps to the actual last available timestamp."
                )
            ),
            "low_ram_exclusions": sorted(LOW_RAM_EXCLUDED_STRATEGIES),
            "timeframe_exclusions": sorted(LOW_RAM_EXCLUDED_TIMEFRAMES),
            "requested_timeframes_excluded": excluded_requested_timeframes,
            "strict_low_ram_exclusion": True,
            "metals_allowed": bool(allow_metals),
            "candidate_pool_policy": "Strict promoted strategies stay primary. If none are promoted, keep provisional shortlist candidates that either beat BTC on the validation month or deliver at least 2% in each of the latest three months.",
            "manual_profile": "Use a higher-timeframe/core-symbol profile when RAM is constrained; this run reports only the explicitly requested symbol/timeframe slice.",
        },
    }
    summary_path = out_dir / f"exact_window_suite_summary_{stamp}.json"
    latest_path = out_dir / "exact_window_suite_summary_latest.json"
    details_path = out_dir / f"exact_window_candidate_details_{stamp}.json"
    details_latest = out_dir / "exact_window_candidate_details_latest.json"
    md_path = out_dir / f"exact_window_suite_summary_{stamp}.md"
    md_latest = out_dir / "exact_window_suite_summary_latest.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    details_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    details_latest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = _render_markdown(summary)
    md_path.write_text(markdown, encoding="utf-8")
    md_latest.write_text(markdown, encoding="utf-8")
    if progress_callback is not None:
        progress_callback(
            "suite_complete",
            {
                "evaluated_count": len(results),
                "promoted_count": int(summary.get("promoted_count") or 0),
                "summary_path": str(summary_path),
                "details_path": str(details_path),
            },
        )
    return summary
