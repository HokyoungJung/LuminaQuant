from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.backtesting.cli_contract import RawFirstDataMissingError
from lumina_quant.configuration import BacktestConfigView, get_default_runtime_config

BacktestConfig = BacktestConfigView(get_default_runtime_config())
from lumina_quant.eval.final_validation import (
    build_latest_anchored_split,
    discover_latest_common_complete_timestamp,
    required_feature_symbols,
    required_symbol_timeframes,
)
from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    OOS_START,
    PORTFOLIO_CURRENT_OPTIMIZATION,
    TRAIN_START,
    VAL_END_EXCLUSIVE,
    VAL_START,
    acquire_portfolio_memory_guard,
    portfolio_followup_default_budget_bytes,
    resolve_current_optimization_path,
    resolve_incumbent_bundle_path,
)
from lumina_quant.storage.parquet import normalize_symbol, timeframe_to_milliseconds
from lumina_quant.symbol_universe import canonicalize_research_symbol
from lumina_quant.strategy_factory import run_candidate_research
from lumina_quant.utils.risk_free import resolve_risk_free_config, sharpe_ratio, sortino_ratio

FEATURE_REQUIRED_STRATEGIES = {"CompositeTrendStrategy", "PerpCrowdingCarryStrategy"}
DEFAULT_REFRESH_REPORT = FOLLOWUP_ROOT / "final_portfolio_validation_data_refresh_latest.json"
DEFAULT_SUPPORT_INVENTORY_JSON = (
    FOLLOWUP_ROOT / "final_portfolio_validation_support_inventory_latest.json"
)
DEFAULT_OUTPUT_JSON = FOLLOWUP_ROOT / "final_portfolio_validation_latest.json"
DEFAULT_OUTPUT_MD = FOLLOWUP_ROOT / "final_portfolio_validation_latest.md"
DEFAULT_RSS_LOG = FOLLOWUP_ROOT / "final_portfolio_validation_rss_latest.jsonl"
DEFAULT_SOFT_RSS_BYTES = int(7.2 * 1024 * 1024 * 1024)
STRICT_VALIDATION_DATA_MODE = "legacy"
STRICT_VALIDATION_CACHE_VERSION = "v2"
DEFAULT_STRICT_VALIDATION_CACHE_DIR = FOLLOWUP_ROOT / "strict_validation_cache"


def parse_utc(value: str | None) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    normalized = token.replace("Z", "+00:00") if token.endswith("Z") else token
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value.isoformat().replace("+00:00", "Z")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _normalize_cache_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_cache_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_cache_value(item) for item in value)
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _timestamp_token(point: dict[str, Any], idx: int) -> tuple[str, tuple[int, float, float]]:
    raw = point.get("datetime", point.get("t"))
    if isinstance(raw, str) and raw.strip():
        dt = parse_utc(raw)
        if dt is not None:
            return f"dt:{iso_utc(dt)}", (0, dt.timestamp(), float(idx))
    try:
        numeric = float(raw)
    except Exception:
        numeric = float(idx)
        return f"seq:{idx}", (2, numeric, float(idx))
    if abs(numeric) >= 1e12:
        return f"ms:{numeric:.0f}", (0, numeric / 1000.0, float(idx))
    return f"num:{numeric:.12g}", (1, numeric, float(idx))


def _aggregate_stream(stream: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    bucket: dict[str, dict[str, Any]] = {}
    for idx, point in enumerate(list(stream or [])):
        token, sort_key = _timestamp_token(point, idx)
        if token not in bucket:
            bucket[token] = {
                "sort_key": sort_key,
                "value": 0.0,
                "raw": point.get("datetime", point.get("t")),
            }
        bucket[token]["value"] += safe_float(point.get("v"), 0.0)
    return bucket


def _stream_from_aggregate(aggregated: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _token, row in sorted(aggregated.items(), key=lambda item: item[1]["sort_key"]):
        payload = {"t": row["raw"], "v": float(row["value"])}
        if isinstance(row["raw"], str):
            payload["datetime"] = row["raw"]
        out.append(payload)
    return out


def _concat_streams(
    lhs_stream: list[dict[str, Any]], rhs_stream: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    aggregated = _aggregate_stream(lhs_stream)
    for token, row in _aggregate_stream(rhs_stream).items():
        if token not in aggregated:
            aggregated[token] = row
        else:
            aggregated[token]["value"] += float(row["value"])
    return _stream_from_aggregate(aggregated)


def _weighted_stream(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    bucket: dict[str, dict[str, Any]] = {}
    for row in rows:
        weight = safe_float(row.get("_saved_weight"), 0.0)
        for idx, point in enumerate(list(((row.get("return_streams") or {}).get(split)) or [])):
            token, sort_key = _timestamp_token(point, idx)
            if token not in bucket:
                bucket[token] = {
                    "sort_key": sort_key,
                    "value": 0.0,
                    "raw": point.get("datetime", point.get("t")),
                }
            bucket[token]["value"] += weight * safe_float(point.get("v"), 0.0)
    return _stream_from_aggregate(bucket)


def _aligned_arrays(
    lhs_stream: list[dict[str, Any]],
    rhs_stream: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    lhs = _aggregate_stream(lhs_stream)
    rhs = _aggregate_stream(rhs_stream)
    merged = {**lhs}
    for token, row in rhs.items():
        merged.setdefault(token, row)
    ordered = [token for token, _ in sorted(merged.items(), key=lambda item: item[1]["sort_key"])]
    lhs_values = np.asarray(
        [safe_float((lhs.get(token) or {}).get("value"), 0.0) for token in ordered],
        dtype=float,
    )
    rhs_values = np.asarray(
        [safe_float((rhs.get(token) or {}).get("value"), 0.0) for token in ordered],
        dtype=float,
    )
    return lhs_values, rhs_values


def _stream_timestamps(stream: list[dict[str, Any]]) -> np.ndarray:
    timestamps: list[Any] = []
    for point in list(stream or []):
        raw = point.get("datetime", point.get("t"))
        dt = parse_utc(raw if isinstance(raw, str) else None)
        if dt is None and isinstance(raw, (int, float)):
            scale = 1000.0 if abs(float(raw)) >= 1e12 else 1.0
            dt = datetime.fromtimestamp(float(raw) / scale, tz=UTC)
        timestamps.append(dt if dt is not None else raw)
    return np.asarray(timestamps, dtype=object)


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = 1.0 - np.divide(equity, np.maximum(peaks, 1e-12))
    return float(np.max(drawdown)) if drawdown.size else 0.0


def _metrics(
    returns: np.ndarray,
    *,
    periods_per_year: int = 365,
    timestamps: np.ndarray | None = None,
    metric_config: Any | None = None,
) -> dict[str, float]:
    if returns.size == 0:
        return {
            "total_return": 0.0,
            "return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "risk_free_annual": 0.0,
            "risk_free_per_period": 0.0,
            "sortino_target_annual": 0.0,
            "sortino_target_per_period": 0.0,
        }
    resolved_rf = resolve_risk_free_config(
        metric_config or BacktestConfig,
        periods_per_year=int(periods_per_year),
        timestamps=timestamps,
    )
    total_return = float(np.prod(1.0 + returns) - 1.0)
    years = max(1.0 / periods_per_year, returns.size / periods_per_year)
    cagr = float(math.exp(math.log1p(max(-0.999999, total_return)) / years) - 1.0)
    sigma = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    mdd = _max_drawdown(returns)
    calmar = 0.0 if mdd <= 1e-12 else cagr / mdd
    return {
        "total_return": total_return,
        "return": total_return,
        "cagr": cagr,
        "sharpe": float(
            sharpe_ratio(
                returns,
                periods_per_year=int(periods_per_year),
                risk_free_per_period=float(resolved_rf.per_period_rate),
            )
        ),
        "sortino": float(
            sortino_ratio(
                returns,
                periods_per_year=int(periods_per_year),
                target_per_period=float(resolved_rf.sortino_target_per_period),
            )
        ),
        "calmar": float(calmar),
        "max_drawdown": float(mdd),
        "volatility": float(sigma * math.sqrt(periods_per_year)),
        "risk_free_annual": float(resolved_rf.annual_rate),
        "risk_free_per_period": float(resolved_rf.per_period_rate),
        "sortino_target_annual": float(resolved_rf.sortino_target_annual),
        "sortino_target_per_period": float(resolved_rf.sortino_target_per_period),
    }


def build_validation_split(oos_end: datetime) -> dict[str, str]:
    val_end_exclusive = parse_utc(VAL_END_EXCLUSIVE)
    train_start = parse_utc(TRAIN_START)
    val_start = parse_utc(VAL_START)
    oos_start = parse_utc(OOS_START)
    if train_start is None or val_start is None or oos_start is None or val_end_exclusive is None:
        raise ValueError("portfolio split contract constants could not be parsed")
    train_end = val_start - timedelta(seconds=1)
    val_end = val_end_exclusive - timedelta(seconds=1)
    return {
        "train_start": iso_utc(train_start) or "",
        "train_end": iso_utc(train_end) or "",
        "val_start": iso_utc(val_start) or "",
        "val_end": iso_utc(val_end) or "",
        "oos_start": iso_utc(oos_start) or "",
        "oos_end": iso_utc(oos_end) or "",
    }


def _research_symbol(symbol: str) -> str:
    canonical = canonicalize_research_symbol(str(symbol))
    if canonical:
        return canonical
    return normalize_symbol(str(symbol))


def _load_support_inventory_rows(refresh_payload: dict[str, Any]) -> list[dict[str, Any]]:
    support_paths: list[Path] = []
    support_meta = refresh_payload.get("support_inventory")
    if isinstance(support_meta, dict):
        json_path = str(support_meta.get("json_path") or "").strip()
        if json_path:
            support_paths.append(Path(json_path))
    support_paths.append(DEFAULT_SUPPORT_INVENTORY_JSON)
    for path in support_paths:
        try:
            resolved = Path(path).expanduser().resolve()
        except Exception:
            continue
        if not resolved.exists():
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = list(payload.get("symbols") or [])
        if rows:
            return rows
    return []


def _normalized_refresh_rows(
    refresh_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ohlcv_rows = list(refresh_payload.get("ohlcv_results") or refresh_payload.get("results") or [])
    feature_rows = list(refresh_payload.get("feature_results") or [])
    if feature_rows:
        return ohlcv_rows, feature_rows
    inventory_rows = _load_support_inventory_rows(refresh_payload)
    if not inventory_rows:
        return ohlcv_rows, []
    derived_feature_rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        symbol = str(row.get("symbol") or "").strip()
        last_timestamp_utc = str(row.get("last_timestamp_utc") or "").strip()
        if not symbol or not last_timestamp_utc:
            continue
        derived_feature_rows.append(
            {
                "symbol": symbol,
                "last_timestamp_utc": last_timestamp_utc,
                "feature_source": "support_inventory",
            }
        )
    return ohlcv_rows, derived_feature_rows


def _latest_complete_bucket_start(last_1s_dt: datetime, timeframe: str) -> datetime:
    tf_ms = int(timeframe_to_milliseconds(timeframe))
    last_ms = int(last_1s_dt.timestamp() * 1000)
    if tf_ms <= 1_000:
        return last_1s_dt.replace(microsecond=0)
    adjusted = last_ms - (tf_ms - 1_000)
    if adjusted < 0:
        raise ValueError(f"Insufficient 1s history to derive a complete {timeframe} bucket.")
    start_ms = (adjusted // tf_ms) * tf_ms
    return datetime.fromtimestamp(start_ms / 1000.0, tz=UTC)


def _latest_common_complete_time(
    *,
    refresh_payload: dict[str, Any],
    required_pairs: list[tuple[str, str]],
    feature_symbols: list[str],
) -> tuple[datetime, list[dict[str, str]]]:
    ohlcv_rows, feature_rows = _normalized_refresh_rows(refresh_payload)
    ohlcv_last: dict[str, datetime] = {}
    for row in ohlcv_rows:
        symbol = _research_symbol(row.get("symbol") or "")
        dt = parse_utc(row.get("after_ohlcv_max_utc"))
        if symbol and dt is not None:
            ohlcv_last[symbol] = dt

    feature_last: dict[str, datetime] = {}
    feature_source_by_symbol: dict[str, str] = {}
    for row in feature_rows:
        symbol = _research_symbol(row.get("symbol") or "")
        dt = parse_utc(row.get("last_timestamp_utc"))
        if symbol and dt is not None:
            feature_last[symbol] = dt
            feature_source_by_symbol[symbol] = str(row.get("feature_source") or "feature_points")

    candidates: list[tuple[datetime, dict[str, str]]] = []
    for symbol, timeframe in required_pairs:
        dt_1s = ohlcv_last.get(symbol)
        if dt_1s is None:
            raise ValueError(
                f"Missing refreshed 1s coverage for {symbol} required by {timeframe} validation."
            )
        complete_dt = _latest_complete_bucket_start(dt_1s, timeframe)
        candidates.append(
            (
                complete_dt,
                {
                    "source": "market_data",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "last_complete_utc": iso_utc(complete_dt) or "",
                },
            )
        )

    for symbol in list(feature_symbols or []):
        dt = feature_last.get(_research_symbol(symbol))
        if dt is None:
            raise ValueError(f"Missing refreshed feature coverage for {symbol}.")
        candidates.append(
            (
                dt,
                {
                    "source": feature_source_by_symbol.get(
                        _research_symbol(symbol), "feature_points"
                    ),
                    "symbol": _research_symbol(symbol),
                    "timeframe": "feature",
                    "last_complete_utc": iso_utc(dt) or "",
                },
            )
        )

    if not candidates:
        raise ValueError("No refreshed real-data coverage candidates available.")
    anchor_dt, _ = min(candidates, key=lambda item: item[0])
    evidence = [item[1] for item in sorted(candidates, key=lambda item: item[0])]
    return anchor_dt, evidence


def _group_candidates_for_strict_research(
    candidates: list[dict[str, Any]],
    strategy_timeframes: list[str],
    symbol_universe: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, ...]] = []
    for candidate in list(candidates or []):
        candidate_symbols = sorted(
            {
                _research_symbol(symbol)
                for symbol in list(candidate.get("symbols") or [])
                if str(symbol).strip()
            }
        ) or list(symbol_universe)
        candidate_timeframes = (
            (
                str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or "")
                .strip()
                .lower(),
            )
            if str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or "").strip()
            else tuple(strategy_timeframes)
        )
        bucket = grouped.get(candidate_timeframes)
        if bucket is None:
            bucket = {"timeframes": list(candidate_timeframes), "symbols": [], "candidates": []}
            grouped[candidate_timeframes] = bucket
            ordered_keys.append(candidate_timeframes)
        bucket["candidates"].append(candidate)
        for symbol in candidate_symbols:
            if symbol not in bucket["symbols"]:
                bucket["symbols"].append(symbol)
    return [grouped[key] for key in ordered_keys]


def _build_comparison(
    saved_metrics: dict[str, Any],
    refreshed_metrics: dict[str, Any],
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "oos"):
        saved_split = dict(saved_metrics.get(split) or {})
        refreshed_split = dict(refreshed_metrics.get(split) or {})
        comparison[split] = {
            "total_return_delta": safe_float(refreshed_split.get("total_return"), 0.0)
            - safe_float(saved_split.get("total_return"), 0.0),
            "sharpe_delta": safe_float(refreshed_split.get("sharpe"), 0.0)
            - safe_float(saved_split.get("sharpe"), 0.0),
            "max_drawdown_delta": safe_float(refreshed_split.get("max_drawdown"), 0.0)
            - safe_float(saved_split.get("max_drawdown"), 0.0),
        }
    return comparison


def _required_symbol_timeframes(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return sorted(required_symbol_timeframes(rows))


def _saved_weight_rows(
    refreshed_rows: list[dict[str, Any]],
    saved_weights: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(row.get("candidate_id") or ""): dict(row) for row in refreshed_rows}
    by_name = {str(row.get("name") or ""): dict(row) for row in refreshed_rows}
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for weight_row in saved_weights:
        key_id = str(weight_row.get("candidate_id") or "")
        key_name = str(weight_row.get("name") or "")
        source = by_id.get(key_id) or by_name.get(key_name)
        if source is None:
            missing.append(key_id or key_name or "unknown")
            continue
        row = dict(source)
        row["_saved_weight"] = safe_float(weight_row.get("weight"), 0.0)
        resolved.append(row)
    if missing:
        raise RuntimeError(
            "saved incumbent weights missing refreshed candidates: " + ", ".join(missing)
        )
    return resolved


def _daily_aggregate_stream(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        dt = parse_utc(point.get("datetime") if isinstance(point.get("datetime"), str) else None)
        if dt is None:
            raw_t = point.get("t")
            if isinstance(raw_t, str):
                dt = parse_utc(raw_t)
            if dt is None:
                try:
                    scale = 1000.0 if abs(float(raw_t)) >= 1e12 else 1.0
                    dt = datetime.fromtimestamp(float(raw_t) / scale, tz=UTC)
                except Exception:
                    continue
        day_key = dt.strftime("%Y-%m-%d")
        buckets[day_key].append(safe_float(point.get("v"), 0.0))

    out: list[dict[str, Any]] = []
    for day_key in sorted(buckets):
        returns = np.asarray(buckets[day_key], dtype=float)
        out.append(
            {
                "t": f"{day_key}T00:00:00Z",
                "datetime": f"{day_key}T00:00:00Z",
                "v": float(np.prod(1.0 + returns) - 1.0),
            }
        )
    return out


def _extend_saved_rows(
    saved_rows: list[dict[str, Any]],
    refreshed_rows: list[dict[str, Any]],
    saved_weights: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    saved_by_id = {str(row.get("candidate_id") or ""): dict(row) for row in saved_rows}
    saved_by_name = {str(row.get("name") or ""): dict(row) for row in saved_rows}
    refreshed_by_id = {str(row.get("candidate_id") or ""): dict(row) for row in refreshed_rows}
    refreshed_by_name = {str(row.get("name") or ""): dict(row) for row in refreshed_rows}
    combined_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for weight_row in saved_weights:
        key_id = str(weight_row.get("candidate_id") or "")
        key_name = str(weight_row.get("name") or "")
        saved = saved_by_id.get(key_id) or saved_by_name.get(key_name)
        refreshed = refreshed_by_id.get(key_id) or refreshed_by_name.get(key_name)
        if saved is None:
            missing.append(key_id or key_name or "unknown")
            continue
        combined = dict(saved)
        combined["_saved_weight"] = safe_float(weight_row.get("weight"), 0.0)
        return_streams = dict(combined.get("return_streams") or {})
        saved_oos_stream = list(return_streams.get("oos") or [])
        extension_oos_stream = _daily_aggregate_stream(
            list(((refreshed or {}).get("return_streams") or {}).get("oos") or [])
        )
        combined_oos_stream = _concat_streams(saved_oos_stream, extension_oos_stream)
        return_streams["oos"] = combined_oos_stream
        combined["return_streams"] = return_streams

        oos_returns = np.asarray(
            [safe_float(point.get("v"), 0.0) for point in combined_oos_stream],
            dtype=float,
        )
        refreshed_oos = dict((refreshed or {}).get("oos") or {})
        saved_oos = dict(saved.get("oos") or {})
        combined["oos"] = {
            **saved_oos,
            **_metrics(
                oos_returns,
                periods_per_year=365,
                timestamps=_stream_timestamps(combined_oos_stream),
            ),
            "return": _metrics(
                oos_returns,
                periods_per_year=365,
                timestamps=_stream_timestamps(combined_oos_stream),
            )["total_return"],
            "trade_count": safe_float(saved_oos.get("trade_count"), 0.0)
            + safe_float(refreshed_oos.get("trade_count"), 0.0),
            "turnover": safe_float(saved_oos.get("turnover"), 0.0)
            + safe_float(refreshed_oos.get("turnover"), 0.0),
        }
        combined_rows.append(combined)

    if missing:
        raise RuntimeError("saved incumbent rows missing in bundle: " + ", ".join(missing))
    return combined_rows


def _component_summaries(
    rows: list[dict[str, Any]], oos_stream: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        component_oos = dict(row.get("oos") or {})
        lhs, rhs = _aligned_arrays(
            list(((row.get("return_streams") or {}).get("oos")) or []),
            oos_stream,
        )
        corr = 0.0
        if lhs.size >= 8 and rhs.size >= 8:
            sigma_l = float(np.std(lhs, ddof=1))
            sigma_r = float(np.std(rhs, ddof=1))
            if sigma_l > 1e-12 and sigma_r > 1e-12:
                corr = float(np.corrcoef(lhs, rhs)[0, 1])
                if not math.isfinite(corr):
                    corr = 0.0
        out.append(
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                "saved_weight": safe_float(row.get("_saved_weight"), 0.0),
                "oos_total_return": safe_float(
                    component_oos.get("total_return", component_oos.get("return")),
                    0.0,
                ),
                "oos_sharpe": safe_float(component_oos.get("sharpe"), 0.0),
                "oos_max_drawdown": safe_float(
                    component_oos.get("max_drawdown", component_oos.get("mdd")),
                    0.0,
                ),
                "oos_turnover": safe_float(component_oos.get("turnover"), 0.0),
                "portfolio_oos_corr": corr,
            }
        )
    return out


def _monthly_oos_returns(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
    monthly: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        dt = parse_utc(point.get("datetime") if isinstance(point.get("datetime"), str) else None)
        if dt is None:
            try:
                dt = datetime.fromtimestamp(float(point.get("t")) / 1000.0, tz=UTC)
            except Exception:
                continue
        monthly[dt.strftime("%Y-%m")].append(safe_float(point.get("v"), 0.0))
    out = []
    for month in sorted(monthly):
        returns = np.asarray(monthly[month], dtype=float)
        out.append(
            {
                "month": month,
                "total_return": float(np.prod(1.0 + returns) - 1.0),
                "days": int(returns.size),
            }
        )
    return out


def _stream_end_utc(stream: list[dict[str, Any]]) -> datetime | None:
    latest: datetime | None = None
    for point in list(stream or []):
        raw_datetime = point.get("datetime")
        raw_t = point.get("t")
        dt = parse_utc(
            raw_datetime
            if isinstance(raw_datetime, str)
            else (raw_t if isinstance(raw_t, str) else None)
        )
        if dt is None:
            try:
                scale = 1000.0 if abs(float(raw_t)) >= 1e12 else 1.0
                dt = datetime.fromtimestamp(float(raw_t) / scale, tz=UTC)
            except Exception:
                continue
        latest = dt if latest is None else max(latest, dt)
    return latest


def _strict_validation_cache_key(
    *,
    candidates: list[dict[str, Any]],
    strategy_timeframes: list[str],
    symbol_universe: list[str],
    split: dict[str, str],
    min_bundle_bars: int,
) -> str:
    normalized_candidates = [
        {
            "candidate_id": str(candidate.get("candidate_id") or "").strip(),
            "name": str(candidate.get("name") or "").strip(),
            "strategy_name": str(candidate.get("strategy_name") or "").strip(),
            "strategy_class": str(candidate.get("strategy_class") or "").strip(),
            "strategy_timeframe": str(
                candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""
            )
            .strip()
            .lower(),
            "symbols": sorted(
                str(symbol).strip()
                for symbol in list(candidate.get("symbols") or [])
                if str(symbol).strip()
            ),
            "leverage": int(
                safe_float(
                    candidate.get("leverage"),
                    safe_float((candidate.get("params") or {}).get("leverage"), 0.0),
                )
            ),
            "params": _normalize_cache_value(dict(candidate.get("params") or {})),
        }
        for candidate in list(candidates or [])
    ]
    payload = {
        "version": STRICT_VALIDATION_CACHE_VERSION,
        "data_mode": STRICT_VALIDATION_DATA_MODE,
        "strategy_timeframes": list(strategy_timeframes),
        "symbol_universe": list(symbol_universe),
        "split": dict(split),
        "min_bundle_bars": int(min_bundle_bars),
        "candidates": normalized_candidates,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_strict_validation_cache(cache_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_strict_validation_cache(cache_path: Path, report: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _run_strict_research(
    *,
    candidates: list[dict[str, Any]],
    strategy_timeframes: list[str],
    symbol_universe: list[str],
    split: dict[str, str],
    min_bundle_bars: int,
) -> dict[str, Any]:
    combined_candidates: list[dict[str, Any]] = []
    combined_data_sources: dict[str, list[Any]] = {}
    report_generated_at: str | None = None

    for group in _group_candidates_for_strict_research(
        candidates, strategy_timeframes, symbol_universe
    ):
        cache_key = _strict_validation_cache_key(
            candidates=list(group["candidates"]),
            strategy_timeframes=list(group["timeframes"]),
            symbol_universe=list(group["symbols"]),
            split=split,
            min_bundle_bars=min_bundle_bars,
        )
        cache_path = DEFAULT_STRICT_VALIDATION_CACHE_DIR / f"{cache_key}.json"
        report = _load_strict_validation_cache(cache_path)
        if report is None:
            report = run_candidate_research(
                candidates=list(group["candidates"]),
                strategy_timeframes=list(group["timeframes"]),
                symbol_universe=list(group["symbols"]),
                split=split,
                stage1_keep_ratio=1.0,
                max_candidates=max(1, len(list(group["candidates"]))),
                data_mode=STRICT_VALIDATION_DATA_MODE,
                allow_csv_fallback=False,
                allow_synthetic_fallback=False,
                min_bundle_bars=max(1, int(min_bundle_bars)),
            )
            _write_strict_validation_cache(cache_path, report)
        data_sources = dict(report.get("data_sources") or {})
        if list(data_sources.get("synthetic") or []):
            raise RuntimeError("Strict validation unexpectedly used synthetic fallback.")
        if list(data_sources.get("csv") or []):
            raise RuntimeError("Strict validation unexpectedly used CSV fallback.")
        report_candidates = list(report.get("candidates") or [])
        for candidate in list(group["candidates"]):
            candidate_id = str(candidate.get("candidate_id") or "").strip()
            candidate_name = str(candidate.get("name") or "").strip()
            matched_candidates = [
                row
                for row in report_candidates
                if (candidate_id and str(row.get("candidate_id") or "").strip() == candidate_id)
                or (candidate_name and str(row.get("name") or "").strip() == candidate_name)
            ] or report_candidates
            if not matched_candidates:
                candidate_label = str(
                    candidate.get("name") or candidate.get("candidate_id") or "unknown"
                )
                raise RuntimeError(
                    f"Strict validation returned no candidate rows for {candidate_label}."
                )
            combined_candidates.extend(matched_candidates)
        for key, values in data_sources.items():
            target = combined_data_sources.setdefault(str(key), [])
            if isinstance(values, list):
                for value in values:
                    if value not in target:
                        target.append(value)
            elif values not in target:
                target.append(values)
        report_generated_at = str(report.get("generated_at") or report_generated_at or "")
        gc.collect()

    return {
        "generated_at": report_generated_at,
        "data_sources": combined_data_sources,
        "candidates": combined_candidates,
    }


def evaluate_saved_weight_portfolio(
    rows: list[dict[str, Any]], *, metric_config: Any | None = None
) -> dict[str, Any]:
    raw_streams = {split: _weighted_stream(rows, split) for split in ("train", "val", "oos")}
    daily_streams = {
        split: _daily_aggregate_stream(stream) for split, stream in raw_streams.items()
    }
    metrics: dict[str, dict[str, Any]] = {}
    weighted_component_summaries: dict[str, dict[str, float]] = {}
    for split, stream in daily_streams.items():
        returns = np.asarray([safe_float(item.get("v"), 0.0) for item in stream], dtype=float)
        metrics[split] = _metrics(
            returns,
            periods_per_year=365,
            timestamps=_stream_timestamps(stream),
            metric_config=metric_config,
        )
        weighted_component_summaries[split] = {
            "trade_count": float(
                sum(
                    safe_float((row.get(split) or {}).get("trade_count"), 0.0)
                    * safe_float(row.get("_saved_weight"), 0.0)
                    for row in rows
                )
            ),
            "turnover": float(
                sum(
                    safe_float((row.get(split) or {}).get("turnover"), 0.0)
                    * safe_float(row.get("_saved_weight"), 0.0)
                    for row in rows
                )
            ),
            "benchmark_corr": float(
                sum(
                    safe_float((row.get(split) or {}).get("benchmark_corr"), 0.0)
                    * safe_float(row.get("_saved_weight"), 0.0)
                    for row in rows
                )
            ),
        }
    oos_returns = np.asarray(
        [safe_float(item.get("v"), 0.0) for item in daily_streams["oos"]],
        dtype=float,
    )
    oos_timestamps = _stream_timestamps(daily_streams["oos"])
    weighted_turnover = weighted_component_summaries["oos"].get("turnover", 0.0)
    weighted_cost = float(
        sum(
            safe_float((row.get("metadata") or {}).get("cost_rate"), 0.0005)
            * safe_float(row.get("_saved_weight"), 0.0)
            for row in rows
        )
    )
    cost_x2 = oos_returns - (max(0.0, 2.0 - 1.0) * weighted_turnover * weighted_cost)
    cost_x3 = oos_returns - (max(0.0, 3.0 - 1.0) * weighted_turnover * weighted_cost)
    sensitivity = {
        "cost_stress": {
            "x2": _metrics(
                cost_x2,
                periods_per_year=365,
                timestamps=oos_timestamps,
                metric_config=metric_config,
            ),
            "x3": _metrics(
                cost_x3,
                periods_per_year=365,
                timestamps=oos_timestamps,
                metric_config=metric_config,
            ),
        },
        "param_drift": {
            "minus_10pct_signal": _metrics(
                oos_returns * 0.9,
                periods_per_year=365,
                timestamps=oos_timestamps,
                metric_config=metric_config,
            ),
            "plus_10pct_signal": _metrics(
                oos_returns * 1.1,
                periods_per_year=365,
                timestamps=oos_timestamps,
                metric_config=metric_config,
            ),
        },
    }
    return {
        "portfolio_metrics": metrics,
        "weighted_component_summaries": weighted_component_summaries,
        "portfolio_return_streams": raw_streams,
        "portfolio_daily_return_streams": daily_streams,
        "component_rows": _component_summaries(rows, raw_streams["oos"]),
        "oos_monthly_returns": _monthly_oos_returns(daily_streams["oos"]),
        "sensitivity": sensitivity,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    continuity = dict(payload.get("continuity_validation") or {})
    exact = dict(payload.get("exact_window_final_validation") or {})
    exact_metrics = dict((exact.get("portfolio_metrics") or {}).get("oos") or {})
    saved_metrics = dict((payload.get("saved_portfolio_metrics") or {}).get("oos") or {})
    comparison = dict(((exact.get("comparison_vs_saved_artifact") or {}).get("oos")) or {})
    lines = [
        "# Portfolio Validation Suite",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- status: `{payload.get('status')}`",
        f"- latest_common_complete_utc: `{payload.get('latest_common_complete_utc')}`",
        f"- recommended_action: `{payload.get('recommended_action')}`",
        "",
        "## Final exact-window validation",
        "",
        f"- validation_oos_end: `{(exact.get('validation_split') or {}).get('oos_end')}`",
        f"- risk_free_source: `{exact_metrics.get('risk_free_annual')}` annual | per-period `{exact_metrics.get('risk_free_per_period')}`",
        "",
        "| Metric | Saved artifact | Exact-window refreshed | Delta |",
        "|---|---:|---:|---:|",
        f"| total_return | {safe_float(saved_metrics.get('total_return'), 0.0):.4%} | {safe_float(exact_metrics.get('total_return'), 0.0):.4%} | {safe_float(comparison.get('total_return_delta'), 0.0):.4%} |",
        f"| sharpe | {safe_float(saved_metrics.get('sharpe'), 0.0):.3f} | {safe_float(exact_metrics.get('sharpe'), 0.0):.3f} | {safe_float(comparison.get('sharpe_delta'), 0.0):.3f} |",
        f"| sortino | {safe_float(saved_metrics.get('sortino'), 0.0):.3f} | {safe_float(exact_metrics.get('sortino'), 0.0):.3f} | n/a |",
        f"| max_drawdown | {safe_float(saved_metrics.get('max_drawdown'), 0.0):.4%} | {safe_float(exact_metrics.get('max_drawdown'), 0.0):.4%} | {safe_float(comparison.get('max_drawdown_delta'), 0.0):.4%} |",
        "",
        "## Metric semantics",
        "",
        "- `portfolio_metrics` are true portfolio return-stream metrics.",
        "- `weighted_component_summaries` are weighted component aggregates for turnover / trade_count / benchmark_corr and are not portfolio-return-derived metrics.",
        "",
        "## Continuity validation",
        "",
        f"- status: `{continuity.get('status')}`",
        f"- extension_start: `{continuity.get('extension_start_utc')}`",
        f"- extension_end: `{continuity.get('extension_end_utc')}`",
        "",
        "## Components",
        "",
        "| Name | Weight | OOS Return | OOS Sharpe | Portfolio Corr |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in list(exact.get("component_rows") or []):
        lines.append(
            f"| `{row['name']}` | {safe_float(row.get('saved_weight'), 0.0):.2%} | {safe_float(row.get('oos_total_return'), 0.0):.4%} | {safe_float(row.get('oos_sharpe'), 0.0):.3f} | {safe_float(row.get('portfolio_oos_corr'), 0.0):.3f} |"
        )
    lines.extend(["", "## Coverage evidence", ""])
    for item in list(payload.get("coverage_evidence") or []):
        lines.append(
            f"- `{item.get('source')}` `{item.get('symbol')}` `{item.get('timeframe')}` -> `{item.get('last_complete_utc')}`"
        )
    if payload.get("error"):
        lines.extend(["", "## Error", "", f"- `{payload['error']}`"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the saved incumbent portfolio on strict real data."
    )
    parser.add_argument("--bundle-path", default="")
    parser.add_argument("--portfolio-path", default=str(PORTFOLIO_CURRENT_OPTIMIZATION))
    parser.add_argument("--refresh-report", default=str(DEFAULT_REFRESH_REPORT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--rss-log", default=str(DEFAULT_RSS_LOG))
    parser.add_argument(
        "--memory-budget-bytes",
        type=int,
        default=portfolio_followup_default_budget_bytes(),
    )
    parser.add_argument("--soft-rss-bytes", type=int, default=DEFAULT_SOFT_RSS_BYTES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_path = resolve_incumbent_bundle_path(args.bundle_path or None)
    portfolio_path = resolve_current_optimization_path(args.portfolio_path)
    refresh_report_path = Path(args.refresh_report)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    portfolio_payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    refresh_payload = (
        json.loads(refresh_report_path.read_text(encoding="utf-8"))
        if refresh_report_path.exists()
        else {}
    )

    selected_team = list(
        bundle_payload.get("selected_team") or bundle_payload.get("candidates") or []
    )
    if not selected_team:
        raise RuntimeError(f"no incumbent candidate rows found in {bundle_path}")
    saved_weights = list(portfolio_payload.get("weights") or [])
    if not saved_weights:
        raise RuntimeError(f"no saved weights found in {portfolio_path}")

    symbols = sorted(
        {
            _research_symbol(symbol)
            for row in selected_team
            for symbol in list(row.get("symbols") or [])
            if str(symbol).strip()
        }
    )
    timeframes = sorted(
        {
            str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip().lower()
            for row in selected_team
            if str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip()
        }
    )
    feature_symbols = required_feature_symbols(
        selected_team,
        feature_required_strategies=FEATURE_REQUIRED_STRATEGIES,
    )
    required_pairs = _required_symbol_timeframes(selected_team)

    guard = acquire_portfolio_memory_guard(
        run_name="final_portfolio_validation",
        output_dir=output_json.parent,
        input_path=bundle_path,
        metadata={
            "portfolio_path": str(portfolio_path.resolve()),
            "refresh_report_path": (
                str(refresh_report_path.resolve()) if refresh_report_path.exists() else None
            ),
            "soft_rss_bytes": max(1, int(args.soft_rss_bytes)),
        },
        budget_bytes=max(1, int(args.memory_budget_bytes)),
        rss_log_path=Path(args.rss_log),
        soft_limit_bytes=max(1, int(args.soft_rss_bytes)),
        hard_limit_bytes=max(1, int(args.memory_budget_bytes)),
    )

    payload: dict[str, Any] = {
        "artifact_kind": "portfolio_validation_suite",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "error": None,
        "bundle_path": str(bundle_path),
        "portfolio_path": str(portfolio_path.resolve()),
        "refresh_report_path": str(refresh_report_path.resolve())
        if refresh_report_path.exists()
        else None,
        "symbols": symbols,
        "timeframes": timeframes,
        "coverage_evidence": [],
        "saved_portfolio_metrics": dict(portfolio_payload.get("portfolio_metrics") or {}),
    }

    try:
        guard.checkpoint("start", {"symbols": symbols, "timeframes": timeframes})
        saved_oos_end = _stream_end_utc(
            list((portfolio_payload.get("portfolio_return_streams") or {}).get("oos") or [])
        )
        if saved_oos_end is None:
            raise RuntimeError("saved incumbent artifact is missing OOS return streams")

        continuity_validation: dict[str, Any]
        continuity_evidence: list[dict[str, str]] = []
        if refresh_payload:
            continuity_end, continuity_evidence = _latest_common_complete_time(
                refresh_payload=refresh_payload,
                required_pairs=required_pairs,
                feature_symbols=feature_symbols,
            )
            extension_start = saved_oos_end + timedelta(seconds=1)
            if extension_start <= continuity_end:
                continuity_split = {
                    "train_start": iso_utc(extension_start),
                    "train_end": iso_utc(extension_start),
                    "val_start": iso_utc(extension_start),
                    "val_end": iso_utc(extension_start),
                    "oos_start": iso_utc(extension_start),
                    "oos_end": iso_utc(continuity_end),
                }
                guard.checkpoint(
                    "continuity_extension_start",
                    {
                        "extension_start_utc": iso_utc(extension_start),
                        "extension_end_utc": iso_utc(continuity_end),
                    },
                )
                continuity_report = _run_strict_research(
                    candidates=selected_team,
                    strategy_timeframes=timeframes,
                    symbol_universe=symbols,
                    split=continuity_split,
                    min_bundle_bars=1,
                )
                guard.checkpoint(
                    "continuity_report_loaded",
                    {"candidate_count": len(list(continuity_report.get("candidates") or []))},
                )
                continuity_rows = _extend_saved_rows(
                    selected_team,
                    list(continuity_report.get("candidates") or []),
                    saved_weights,
                )
                continuity_eval = evaluate_saved_weight_portfolio(continuity_rows)
                guard.checkpoint(
                    "continuity_evaluated",
                    {
                        "oos_total_return": safe_float(
                            ((continuity_eval.get("portfolio_metrics") or {}).get("oos") or {}).get(
                                "total_return"
                            ),
                            0.0,
                        )
                    },
                )
                continuity_validation = {
                    "status": "completed",
                    "extension_start_utc": iso_utc(extension_start),
                    "extension_end_utc": iso_utc(continuity_end),
                    "candidate_report": {
                        "generated_at": continuity_report.get("generated_at"),
                        "data_sources": continuity_report.get("data_sources"),
                        "candidate_count": len(list(continuity_report.get("candidates") or [])),
                    },
                    "portfolio_metrics": continuity_eval.get("portfolio_metrics"),
                    "weighted_component_summaries": continuity_eval.get(
                        "weighted_component_summaries"
                    ),
                }
            else:
                continuity_validation = {
                    "status": "no_extension_needed",
                    "extension_start_utc": iso_utc(extension_start),
                    "extension_end_utc": iso_utc(continuity_end),
                    "portfolio_metrics": None,
                    "weighted_component_summaries": None,
                }
        else:
            continuity_validation = {
                "status": "artifact_missing",
                "extension_start_utc": None,
                "extension_end_utc": None,
                "portfolio_metrics": None,
                "weighted_component_summaries": None,
            }

        latest_common_complete, exact_coverage = discover_latest_common_complete_timestamp(
            root_path=str(BacktestConfig.MARKET_DATA_PARQUET_PATH),
            exchange=str(BacktestConfig.MARKET_DATA_EXCHANGE),
            rows=selected_team,
            feature_symbols=feature_symbols,
            suite_start=parse_utc(TRAIN_START) or datetime(2025, 1, 1, tzinfo=UTC),
        )
        if latest_common_complete < saved_oos_end:
            raise RawFirstDataMissingError(
                "Latest common complete real-data timestamp predates the saved incumbent OOS end; final validation cannot proceed."
            )
        exact_split = build_latest_anchored_split(
            saved_oos_end=saved_oos_end,
            anchored_oos_end=latest_common_complete,
        ).as_dict()
        guard.checkpoint(
            "exact_validation_start",
            {
                "candidate_count": len(selected_team),
                "latest_common_complete_utc": iso_utc(latest_common_complete),
            },
        )
        exact_report = _run_strict_research(
            candidates=selected_team,
            strategy_timeframes=timeframes,
            symbol_universe=symbols,
            split=exact_split,
            min_bundle_bars=1,
        )
        guard.checkpoint(
            "exact_validation_report_loaded",
            {"candidate_count": len(list(exact_report.get("candidates") or []))},
        )
        exact_rows = _saved_weight_rows(list(exact_report.get("candidates") or []), saved_weights)
        exact_eval = evaluate_saved_weight_portfolio(exact_rows)
        guard.checkpoint(
            "exact_validation_evaluated",
            {
                "oos_total_return": safe_float(
                    ((exact_eval.get("portfolio_metrics") or {}).get("oos") or {}).get(
                        "total_return"
                    ),
                    0.0,
                )
            },
        )
        exact_comparison = _build_comparison(
            dict(portfolio_payload.get("portfolio_metrics") or {}),
            dict(exact_eval.get("portfolio_metrics") or {}),
        )
        oos_delta = dict(exact_comparison.get("oos") or {})
        recommended_action = (
            "Exact-window refreshed validation materially diverged from the saved incumbent; review data freshness and deployment assumptions before final sign-off."
            if abs(safe_float(oos_delta.get("total_return_delta"), 0.0)) > 0.01
            or abs(safe_float(oos_delta.get("sharpe_delta"), 0.0)) > 0.5
            else "Exact-window refreshed validation remains directionally consistent with the saved incumbent; proceed with manual final review and evidence collation."
        )

        payload.update(
            {
                "latest_common_complete_utc": iso_utc(latest_common_complete),
                "recommended_action": recommended_action,
                "coverage_evidence": [
                    *continuity_evidence,
                    *[
                        {
                            "source": "real_data",
                            "symbol": info.symbol,
                            "timeframe": info.timeframe,
                            "last_complete_utc": info.end_utc or "",
                        }
                        for info in exact_coverage
                    ],
                ],
                "continuity_validation": continuity_validation,
                "exact_window_final_validation": {
                    "status": "completed",
                    "validation_split": exact_split,
                    "latest_common_complete_utc": iso_utc(latest_common_complete),
                    "candidate_report": {
                        "generated_at": exact_report.get("generated_at"),
                        "data_sources": exact_report.get("data_sources"),
                        "candidate_count": len(list(exact_report.get("candidates") or [])),
                    },
                    "portfolio_metrics": exact_eval.get("portfolio_metrics"),
                    "weighted_component_summaries": exact_eval.get("weighted_component_summaries"),
                    "portfolio_daily_return_streams": exact_eval.get(
                        "portfolio_daily_return_streams"
                    ),
                    "comparison_vs_saved_artifact": exact_comparison,
                    "component_rows": exact_eval.get("component_rows"),
                    "oos_monthly_returns": exact_eval.get("oos_monthly_returns"),
                    "sensitivity": exact_eval.get("sensitivity"),
                    "saved_weight_total": float(
                        sum(safe_float(row.get("weight"), 0.0) for row in saved_weights)
                    ),
                    "risk_free_reference": {
                        "mode": getattr(BacktestConfig, "RISK_FREE_MODE", "us_treasury_constant"),
                        "tenor": getattr(BacktestConfig, "RISK_FREE_TENOR", "3m"),
                        "annual_rate": float(getattr(BacktestConfig, "RISK_FREE_ANNUAL", 0.0)),
                        "sortino_target_mode": getattr(
                            BacktestConfig, "SORTINO_TARGET_MODE", "same_as_rf"
                        ),
                        "sortino_target_annual": float(
                            getattr(BacktestConfig, "SORTINO_TARGET_ANNUAL", 0.0)
                        ),
                    },
                },
            }
        )
        guard.checkpoint("completed", {"recommended_action": recommended_action})
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload["memory"] = guard.finalize(
            status=str(payload.get("status") or "completed"),
            error=str(payload.get("error") or "") or None,
        )
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        output_md.write_text(build_markdown(payload), encoding="utf-8")
        guard.release()

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
