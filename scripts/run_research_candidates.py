"""Run advanced candidate research and emit shortlist-compatible reports."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from lumina_quant.configuration import (
    BacktestConfigView,
    RuntimeConfig,
    get_default_runtime_config,
    load_runtime_config,
)
from lumina_quant.configuration.schema import ResearchConfig
from lumina_quant.storage.parquet import load_data_dict_from_parquet
from lumina_quant.strategy_factory import (
    build_default_candidate_rows,
    run_candidate_research,
)
from lumina_quant.strategy_factory.selection import (
    apply_selection_reject_and_dedup,
    select_diversified_shortlist,
)
from lumina_quant.symbols import CANONICAL_STRATEGY_TIMEFRAMES, canonicalize_symbol_list
from lumina_quant.strategy_factory.research_run_support import research_config_to_overrides

_METALS = {"XAU/USDT", "XAG/USDT", "XPT/USDT", "XPD/USDT"}
_RESEARCH_PROMOTION_MAX_SPLIT_DRAWDOWN = 0.15
_RESEARCH_STRICT_LIQUIDATION_COUNT_MAX = 0
_SHORTLIST_SELECTION_MODE = "val"
_STRICT_DSR_FLOOR = 0.90
_STRICT_SPA_CEILING = 0.05
_STRICT_PBO_CEILING = 0.50
_TIMEFRAME_MIN_COVERAGE_BARS = {
    "30m": 360,
    "1h": 240,
    "4h": 90,
    "1d": 20,
}
_TIMEFRAME_SECONDS = {
    "30m": 30 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}


def _current_base_config() -> BacktestConfigView:
    return BacktestConfigView(get_default_runtime_config())


def _strict_research_config() -> ResearchConfig:
    """Return the fail-closed diagnostic contract used by this entrypoint."""
    return ResearchConfig(
        strict_selection_gate=True,
        use_lockbox_split=True,
        purge_embargo_bars=1,
        hac_inference=True,
        enforce_selection_reject_gate=True,
        dsr_gate_floor=_STRICT_DSR_FLOOR,
        spa_gate_ceiling=_STRICT_SPA_CEILING,
        pbo_gate_ceiling=_STRICT_PBO_CEILING,
        max_cross_trial_pbo=_STRICT_PBO_CEILING,
    )


DEFAULT_SHORTLIST_SELECTION_CONFIG: dict[str, Any] = {
    "drop_single_without_metrics": False,
    "single_min_score": 0.0,
    "single_min_return": 0.0,
    "single_min_sharpe": 0.0,
    "single_min_trades": 5,
    "allow_multi_asset": False,
    "max_per_lineage": 1,
    "include_weights": True,
    "weight_temperature": 0.35,
    "max_weight": 0.35,
}

ROBUST_SCORE_PARAM_KEYS: tuple[str, ...] = (
    "sharpe_weight",
    "deflated_sharpe_weight",
    "pbo_penalty",
    "return_weight",
    "drawdown_penalty",
    "turnover_penalty",
    "turnover_threshold",
    "cross_corr_penalty",
    "inactive_fold_penalty",
    "failed_fold_penalty",
    "low_active_fold_penalty",
    "active_fold_ratio_floor",
    "no_trade_train_penalty",
    "failed_candidate_scale",
    "sentinel_floor_score",
    "failed_candidate_base_penalty",
    "shortlist_missing_score_fallback",
    "weight_exp_clamp_floor",
    "pair_multi_mix_bonus",
    "mdd_risk_penalty_coeff",
    "dsr_spa_hard_gate",
    "strict_selection_gate",
    "enforce_selection_reject_gate",
    "dsr_gate_floor",
    "spa_gate_ceiling",
    "pbo_gate_ceiling",
)
_MIN_COVERAGE_BARS = 360


def _score_config_scope(score_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(score_config, dict):
        return {}
    nested = score_config.get("candidate_research")
    if isinstance(nested, dict):
        return nested
    return score_config


def _safe_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(minimum, parsed)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _optional_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return default
    return _safe_float(value, 0.0)


def _resolve_shortlist_selection_config(
    score_config: dict[str, Any] | None,
    *,
    top_k: int,
) -> dict[str, Any]:
    scope = _score_config_scope(score_config)
    raw = scope.get("shortlist_selection")
    shortlist_cfg = raw if isinstance(raw, dict) else {}

    resolved: dict[str, Any] = {
        "max_total": max(1, int(top_k)),
        "max_per_family": max(2, int(top_k // 2)),
        "max_per_timeframe": max(2, int(top_k // 2)),
        **DEFAULT_SHORTLIST_SELECTION_CONFIG,
    }

    if "max_per_family" in shortlist_cfg:
        resolved["max_per_family"] = _safe_int(
            shortlist_cfg.get("max_per_family"),
            resolved["max_per_family"],
            minimum=1,
        )
    if "max_per_timeframe" in shortlist_cfg:
        resolved["max_per_timeframe"] = _safe_int(
            shortlist_cfg.get("max_per_timeframe"),
            resolved["max_per_timeframe"],
            minimum=1,
        )
    if "drop_single_without_metrics" in shortlist_cfg:
        resolved["drop_single_without_metrics"] = _safe_bool(
            shortlist_cfg.get("drop_single_without_metrics"),
            bool(resolved["drop_single_without_metrics"]),
        )
    if "single_min_score" in shortlist_cfg:
        resolved["single_min_score"] = _optional_float(
            shortlist_cfg.get("single_min_score"),
            resolved.get("single_min_score"),
        )
    if "single_min_return" in shortlist_cfg:
        resolved["single_min_return"] = _optional_float(
            shortlist_cfg.get("single_min_return"),
            resolved.get("single_min_return"),
        )
    if "single_min_sharpe" in shortlist_cfg:
        resolved["single_min_sharpe"] = _optional_float(
            shortlist_cfg.get("single_min_sharpe"),
            resolved.get("single_min_sharpe"),
        )
    if "single_min_trades" in shortlist_cfg:
        resolved["single_min_trades"] = _safe_int(
            shortlist_cfg.get("single_min_trades"),
            int(resolved["single_min_trades"]),
            minimum=0,
        )
    if "allow_multi_asset" in shortlist_cfg:
        resolved["allow_multi_asset"] = _safe_bool(
            shortlist_cfg.get("allow_multi_asset"),
            bool(resolved["allow_multi_asset"]),
        )
    if "max_per_lineage" in shortlist_cfg:
        resolved["max_per_lineage"] = _safe_int(
            shortlist_cfg.get("max_per_lineage"),
            int(resolved["max_per_lineage"]),
            minimum=1,
        )
    if "include_weights" in shortlist_cfg:
        resolved["include_weights"] = _safe_bool(
            shortlist_cfg.get("include_weights"),
            bool(resolved["include_weights"]),
        )
    if "weight_temperature" in shortlist_cfg:
        resolved["weight_temperature"] = max(
            0.0,
            _safe_float(
                shortlist_cfg.get("weight_temperature"), float(resolved["weight_temperature"])
            ),
        )
    if "max_weight" in shortlist_cfg:
        resolved["max_weight"] = max(
            0.0,
            _safe_float(shortlist_cfg.get("max_weight"), float(resolved["max_weight"])),
        )

    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run advanced candidate research pipeline.")
    base_config = _current_base_config()
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--manifest", default="", help="Optional candidate manifest JSON path.")
    parser.add_argument("--symbols", nargs="+", default=list(base_config.SYMBOLS))
    parser.add_argument("--timeframes", nargs="+", default=list(CANONICAL_STRATEGY_TIMEFRAMES))
    parser.add_argument("--base-timeframe", default="1s")
    parser.add_argument("--stage1-keep-ratio", type=float, default=0.35)
    parser.add_argument("--max-candidates", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--score-config", default="", help="Optional scoring config JSON path.")
    parser.add_argument(
        "--config",
        default="",
        help="RuntimeConfig profile YAML path for this research process.",
    )
    parser.add_argument(
        "--cost-rate-bps-override",
        type=float,
        default=None,
        help="Finite non-negative research cost-rate override in basis points; requires config.",
    )
    parser.add_argument(
        "--train-start", default="", help="Exact split train window start (ISO/date)."
    )
    parser.add_argument("--train-end", default="", help="Exact split train window end (ISO/date).")
    parser.add_argument(
        "--validation-start", default="", help="Exact split validation window start (ISO/date)."
    )
    parser.add_argument(
        "--validation-end", default="", help="Exact split validation window end (ISO/date)."
    )
    parser.add_argument(
        "--oos-start", default="", help="Exact split OOS/test window start (ISO/date)."
    )
    parser.add_argument("--oos-end", default="", help="Exact split OOS/test window end (ISO/date).")
    parser.add_argument(
        "--skip-coverage-rebuild",
        action="store_true",
        help="Skip the exact-split preflight coverage scan and evaluate the requested windows directly.",
    )
    parser.add_argument(
        "--min-bundle-bars",
        type=int,
        default=360,
        help="Minimum bars required when loading each symbol/timeframe bundle.",
    )
    parser.add_argument(
        "--disable-csv-fallback",
        action="store_true",
        help="Do not use CSV fallback data when parquet data is missing.",
    )
    parser.add_argument(
        "--disable-synthetic-fallback",
        action="store_true",
        help=(
            "Do not synthesize missing market data; missing bundles remain missing. "
            "This is the default production/backtest policy."
        ),
    )
    parser.add_argument(
        "--enable-synthetic-fallback",
        action="store_true",
        help="Opt in to synthetic data fallback for isolated smoke tests only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_manifest_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("candidates") or [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _restrict_candidates_to_symbol_universe(
    candidates: list[dict[str, Any]],
    symbol_universe: list[str],
) -> list[dict[str, Any]]:
    allowed = set(canonicalize_symbol_list(symbol_universe))
    if not allowed:
        return list(candidates)

    restricted: list[dict[str, Any]] = []
    for row in candidates:
        candidate = dict(row)
        candidate_symbols = canonicalize_symbol_list(list(candidate.get("symbols") or []))
        if not candidate_symbols:
            restricted.append(candidate)
            continue

        filtered_symbols = [symbol for symbol in candidate_symbols if symbol in allowed]
        if not filtered_symbols:
            continue
        if filtered_symbols != candidate_symbols:
            candidate["symbols"] = filtered_symbols
            metadata = dict(candidate.get("metadata") or {})
            metadata["screened_symbol_subset"] = list(filtered_symbols)
            metadata["screened_symbol_count"] = len(filtered_symbols)
            candidate["metadata"] = metadata
        restricted.append(candidate)
    return restricted


def _load_score_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"score config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid score config JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"score config must be a JSON object: {path}")
    return dict(payload)


def _validate_split_token(value: Any, *, field: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise ValueError(f"missing exact split field: {field}")

    normalized = token.replace("Z", "+00:00") if token.endswith("Z") else token
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        try:
            datetime.fromisoformat(f"{normalized}T00:00:00+00:00")
        except ValueError as exc:
            raise ValueError(f"invalid ISO/date token for {field}: {token}") from exc
    return token


def _build_exact_split(args: argparse.Namespace) -> dict[str, str] | None:
    raw = {
        "train_start": getattr(args, "train_start", ""),
        "train_end": getattr(args, "train_end", ""),
        "val_start": getattr(args, "validation_start", ""),
        "val_end": getattr(args, "validation_end", ""),
        "oos_start": getattr(args, "oos_start", ""),
        "oos_end": getattr(args, "oos_end", ""),
    }
    if not any(str(value or "").strip() for value in raw.values()):
        return None

    missing = [field for field, value in raw.items() if not str(value or "").strip()]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"exact split requires all window boundaries; missing: {missing_text}")

    resolved = {field: _validate_split_token(value, field=field) for field, value in raw.items()}
    resolved["mode"] = "exact_dates"
    return resolved


def _coerce_utc_datetime(value: Any, *, end_of_day: bool = False) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            dt = datetime.fromisoformat(f"{normalized}T00:00:00+00:00")
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
        else:
            if end_of_day and len(text) == 10 and text[4] == "-" and text[7] == "-":
                dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _isoformat_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _split_bounds(split: dict[str, Any] | None) -> tuple[datetime | None, datetime | None]:
    if not isinstance(split, dict):
        return None, None
    starts = [
        _coerce_utc_datetime(split.get("train_start")),
        _coerce_utc_datetime(split.get("val_start")),
        _coerce_utc_datetime(split.get("oos_start")),
    ]
    ends = [
        _coerce_utc_datetime(split.get("train_end"), end_of_day=True),
        _coerce_utc_datetime(split.get("val_end"), end_of_day=True),
        _coerce_utc_datetime(split.get("oos_end"), end_of_day=True),
    ]
    valid_starts = [item for item in starts if item is not None]
    valid_ends = [item for item in ends if item is not None]
    return (
        min(valid_starts) if valid_starts else None,
        max(valid_ends) if valid_ends else None,
    )


def _min_coverage_bars_for_timeframe(timeframe: str) -> int:
    return int(_TIMEFRAME_MIN_COVERAGE_BARS.get(str(timeframe).lower(), _MIN_COVERAGE_BARS))


def _timeframe_delta(timeframe: str) -> timedelta:
    return timedelta(seconds=int(_TIMEFRAME_SECONDS.get(str(timeframe).lower(), 60 * 60)))


def _candidate_window_split(
    *,
    template_split: dict[str, Any],
    window_start: datetime,
    window_end: datetime,
    timeframe: str,
) -> dict[str, Any] | None:
    if window_end <= window_start:
        return None
    min_stage = _timeframe_delta(timeframe)
    if window_end - window_start < min_stage * 3:
        return None

    span_seconds = (window_end - window_start).total_seconds()
    train_end = window_start + timedelta(seconds=span_seconds * 0.60)
    val_end = window_start + timedelta(seconds=span_seconds * 0.80)
    if train_end <= window_start or val_end <= train_end or window_end <= val_end:
        return None

    return {
        **{
            key: value
            for key, value in dict(template_split).items()
            if key
            not in {
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "oos_start",
                "oos_end",
                "validation_start",
                "validation_end",
            }
        },
        "train_start": _isoformat_z(window_start),
        "train_end": _isoformat_z(train_end),
        "val_start": _isoformat_z(train_end),
        "val_end": _isoformat_z(val_end),
        "oos_start": _isoformat_z(val_end),
        "oos_end": _isoformat_z(window_end),
        "strategy_timeframe": str(timeframe),
        "mode": "candidate_data_window",
    }


def _rebuild_candidates_after_coverage(
    *,
    candidates: list[dict[str, Any]],
    symbols: list[str],
    timeframes: list[str],
    split: dict[str, Any] | None,
    progress_callback: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    if split is None:
        return candidates, split, {"enabled": False, "used_candidate_count": len(candidates)}

    start_bound, requested_end = _split_bounds(split)
    if start_bound is None or requested_end is None:
        return candidates, split, {"enabled": False, "used_candidate_count": len(candidates)}

    base_config = _current_base_config()
    db_path = str(getattr(base_config, "MARKET_DATA_PARQUET_PATH", "data/market_parquet"))
    exchange = str(getattr(base_config, "MARKET_DATA_EXCHANGE", "binance") or "binance")
    coverage: dict[tuple[str, str], dict[str, Any]] = {}
    last_values: list[datetime] = []
    if progress_callback is not None:
        progress_callback(
            "coverage_scan_started",
            {
                "symbol_count": len(symbols),
                "timeframe_count": len(timeframes),
                "requested_candidate_count": len(candidates),
            },
        )

    for timeframe_index, timeframe in enumerate(timeframes, start=1):
        loaded = load_data_dict_from_parquet(
            db_path,
            exchange=exchange,
            symbol_list=list(symbols),
            timeframe=timeframe,
            start_date=_isoformat_z(start_bound),
            end_date=_isoformat_z(requested_end),
        )
        for symbol in symbols:
            frame = loaded.get(symbol)
            if frame is None or frame.is_empty():
                coverage[(symbol, timeframe)] = {"rows": 0, "first": None, "last": None}
                continue
            first = _coerce_utc_datetime(frame["datetime"].min())
            last = _coerce_utc_datetime(frame["datetime"].max())
            rows = int(frame.height)
            coverage[(symbol, timeframe)] = {"rows": rows, "first": first, "last": last}
            if (
                rows >= _min_coverage_bars_for_timeframe(timeframe)
                and first is not None
                and last is not None
            ):
                last_values.append(last)
        if progress_callback is not None:
            available_symbol_count = sum(
                1
                for symbol in symbols
                if int((coverage.get((symbol, timeframe)) or {}).get("rows", 0))
                >= _min_coverage_bars_for_timeframe(timeframe)
            )
            progress_callback(
                "coverage_timeframe_loaded",
                {
                    "timeframe": timeframe,
                    "timeframe_index": timeframe_index,
                    "timeframe_count": len(timeframes),
                    "available_symbol_count": available_symbol_count,
                },
            )

    actual_max = min(last_values) if last_values else requested_end
    resolved_split = dict(split)
    resolved_split["requested_oos_end"] = split.get("oos_end")
    resolved_split["actual_max_timestamp"] = _isoformat_z(actual_max)
    if actual_max is not None:
        resolved_split["oos_end"] = _isoformat_z(min(requested_end, actual_max))
    effective_oos_end = _coerce_utc_datetime(resolved_split.get("oos_end"), end_of_day=True)

    available_pairs: set[tuple[str, str]] = set()
    filtered: list[dict[str, Any]] = []
    adjusted_count = 0
    for row in candidates:
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "1m")
        candidate_symbols = canonicalize_symbol_list(list(row.get("symbols") or []))
        if not candidate_symbols:
            continue
        symbol_coverages = [coverage.get((symbol, timeframe)) or {} for symbol in candidate_symbols]
        min_rows = _min_coverage_bars_for_timeframe(timeframe)
        if any(
            int(item.get("rows", 0) or 0) < min_rows
            or item.get("first") is None
            or item.get("last") is None
            for item in symbol_coverages
        ):
            continue

        data_start = max(item["first"] for item in symbol_coverages)
        data_end = min(item["last"] for item in symbol_coverages)
        window_start = max(start_bound, data_start)
        window_end = min(requested_end, data_end)
        if effective_oos_end is not None:
            window_end = min(window_end, effective_oos_end)
        full_requested_end = effective_oos_end or requested_end
        if data_start <= start_bound and data_end >= full_requested_end:
            candidate_split = {**dict(resolved_split), "strategy_timeframe": str(timeframe)}
        else:
            candidate_split = _candidate_window_split(
                template_split=resolved_split,
                window_start=window_start,
                window_end=window_end,
                timeframe=timeframe,
            )
        if candidate_split is None:
            continue

        candidate = dict(row)
        metadata = dict(candidate.get("metadata") or {})
        metadata["effective_split"] = dict(candidate_split)
        metadata["data_window"] = {
            "mode": "candidate_data_window",
            "first": _isoformat_z(data_start),
            "last": _isoformat_z(data_end),
            "window_start": _isoformat_z(window_start),
            "window_end": _isoformat_z(window_end),
            "min_rows": int(min_rows),
            "symbols": {
                symbol: {
                    "rows": int((coverage.get((symbol, timeframe)) or {}).get("rows", 0) or 0),
                    "first": _isoformat_z((coverage.get((symbol, timeframe)) or {}).get("first")),
                    "last": _isoformat_z((coverage.get((symbol, timeframe)) or {}).get("last")),
                }
                for symbol in candidate_symbols
            },
        }
        candidate["metadata"] = metadata
        candidate["effective_split"] = dict(candidate_split)
        if (
            _coerce_utc_datetime(candidate_split.get("train_start")) != start_bound
            or _coerce_utc_datetime(candidate_split.get("oos_end"), end_of_day=True)
            != effective_oos_end
        ):
            adjusted_count += 1
        filtered.append(candidate)
        available_pairs.update((symbol, timeframe) for symbol in candidate_symbols)
    summary = {
        "enabled": True,
        "requested_candidate_count": len(candidates),
        "filtered_candidate_count": len(filtered),
        "used_candidate_count": len(filtered),
        "candidate_window_adjusted_count": int(adjusted_count),
        "fallback_to_precoverage": False,
        "dropped_candidate_count": max(0, len(candidates) - len(filtered)),
        "actual_max_timestamp": resolved_split.get("actual_max_timestamp"),
        "available_pairs": sorted(f"{symbol}@{timeframe}" for symbol, timeframe in available_pairs),
    }
    if progress_callback is not None:
        progress_callback("coverage_rebuilt", summary)
    return filtered, resolved_split, summary


def _symbols_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    fallback: list[str],
) -> list[str]:
    symbols = canonicalize_symbol_list(
        symbol for candidate in candidates for symbol in list(candidate.get("symbols") or [])
    )
    return symbols or list(fallback)


def _timeframes_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    fallback: list[str],
) -> list[str]:
    out: list[str] = []
    for candidate in candidates:
        token = (
            str(candidate.get("strategy_timeframe") or candidate.get("timeframe") or "")
            .strip()
            .lower()
        )
        if token and token not in out:
            out.append(token)
    return out or list(fallback)


def _load_runtime_config_for_args(
    args: argparse.Namespace,
) -> tuple[RuntimeConfig | None, str | None]:
    config_token = str(getattr(args, "config", "") or "").strip()
    if not config_token:
        config_token = str(os.environ.get("LQ_CONFIG_PATH", "") or "").strip()
    cost_override = getattr(args, "cost_rate_bps_override", None)
    if cost_override is not None:
        if not config_token:
            raise ValueError("--cost-rate-bps-override requires --config or LQ_CONFIG_PATH")
        if not math.isfinite(float(cost_override)) or float(cost_override) < 0.0:
            raise ValueError("--cost-rate-bps-override must be finite and non-negative")
    if not config_token:
        return None, None

    config_path = Path(config_token).resolve()
    try:
        runtime_config = load_runtime_config(str(config_path))
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError(f"failed to load --config {config_path}: {exc}") from exc
    if cost_override is not None:
        runtime_config.research.cost_rate_bps_override = float(cost_override)
    return runtime_config, str(config_path)


def _profile_selection_caps(runtime_config: RuntimeConfig | None) -> dict[str, int | None]:
    if runtime_config is None:
        return {
            "max_per_symbol_basket": None,
            "max_per_lineage": None,
            "max_per_family_basket": None,
        }
    research = runtime_config.research

    def _cap(name: str) -> int | None:
        value = getattr(research, name, None)
        if value is None:
            return None
        return _safe_int(value, 1, minimum=1)

    return {
        "max_per_symbol_basket": _cap("max_per_symbol_basket"),
        "max_per_lineage": _cap("max_per_lineage"),
        "max_per_family_basket": _cap("max_per_family_basket"),
    }


def _score_config_research(score_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(score_config, Mapping):
        return {}
    research = score_config.get("research")
    return research if isinstance(research, Mapping) else {}


def _effective_score_config_research(
    runtime_config: RuntimeConfig | None,
    score_config: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    profile_research: dict[str, Any] = {}
    if runtime_config is not None:
        profile_research.update(
            research_config_to_overrides(runtime_config)["score_config_research"]
        )
    effective = dict(profile_research)
    effective.update(_score_config_research(score_config))
    for key in ("cost_rate_multiplier", "cost_rate_bps_override"):
        if key in profile_research:
            effective[key] = profile_research[key]
    return effective


def _profile_strict_selection_enabled(
    runtime_config: RuntimeConfig | None,
    score_config: Mapping[str, Any] | None = None,
) -> bool:
    effective = _effective_score_config_research(runtime_config, score_config)
    return bool(
        effective.get("strict_selection_gate", False)
        or effective.get("enforce_selection_reject_gate", False)
    )


def _strict_shortlist_mode(
    runtime_config: RuntimeConfig | None,
    score_config: Mapping[str, Any] | None = None,
) -> str:
    if not _profile_strict_selection_enabled(runtime_config, score_config):
        return "oos"
    if runtime_config is None:
        return "oos"
    split = research_config_to_overrides(runtime_config)["split"]
    return "val" if bool(split.get("use_lockbox_split", False)) else "oos"


def _registered_routing_requested(
    runtime_config: RuntimeConfig | None,
    score_config: Mapping[str, Any] | None = None,
) -> bool:
    effective = _effective_score_config_research(runtime_config, score_config)
    return bool(effective.get("route_unmapped_registered_strategies", False))


def _candidate_evaluation_mode(candidate: Mapping[str, Any]) -> str:
    for key in ("meta", "metadata"):
        metadata = candidate.get(key)
        if isinstance(metadata, Mapping):
            mode = metadata.get("evaluation_mode")
            if isinstance(mode, str):
                return mode
    return ""


def _has_actual_validation_metrics(candidate: Mapping[str, Any]) -> bool:
    for key in ("validation", "val"):
        metrics = candidate.get(key)
        if not isinstance(metrics, Mapping):
            continue
        if any(
            field in metrics and metrics.get(field) is not None
            for field in (
                "return",
                "total_return",
                "sharpe",
                "mdd",
                "trades",
                "deflated_sharpe",
                "spa_pvalue",
                "pbo",
                "approx_pbo",
            )
        ):
            return True
    return False


def _effective_cost_rate_bps(candidates: list[dict[str, Any]]) -> list[float]:
    rates: set[float] = set()
    for candidate in candidates:
        metadata = candidate.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        try:
            cost_rate = float(metadata.get("cost_rate"))
        except TypeError, ValueError:
            continue
        if math.isfinite(cost_rate):
            rates.add(cost_rate * 10_000.0)
    return sorted(rates)


def _run_candidate_research_with_optional_split(
    *,
    candidates: list[dict[str, Any]],
    base_timeframe: str,
    strategy_timeframes: list[str],
    symbol_universe: list[str],
    stage1_keep_ratio: float,
    max_candidates: int,
    score_config: dict[str, Any] | None,
    exact_split: dict[str, str] | None,
    min_bundle_bars: int = 360,
    allow_csv_fallback: bool = True,
    allow_synthetic_fallback: bool = False,
    progress_callback: Any = None,
    research_config: ResearchConfig | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "candidates": candidates,
        "base_timeframe": base_timeframe,
        "strategy_timeframes": strategy_timeframes,
        "symbol_universe": symbol_universe,
        "stage1_keep_ratio": float(stage1_keep_ratio),
        "max_candidates": int(max_candidates),
        "score_config": score_config,
    }
    signature = inspect.signature(run_candidate_research)
    param_names = set(signature.parameters)
    supports_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if research_config is not None:
        if "research_config" not in param_names and not supports_var_kwargs:
            raise TypeError("candidate research runner does not support required research_config")
        kwargs["research_config"] = research_config
    if progress_callback is not None and (
        "progress_callback" in param_names or supports_var_kwargs
    ):
        kwargs["progress_callback"] = progress_callback
    if "min_bundle_bars" in param_names or supports_var_kwargs:
        kwargs["min_bundle_bars"] = max(1, int(min_bundle_bars))
    if "allow_csv_fallback" in param_names or supports_var_kwargs:
        kwargs["allow_csv_fallback"] = bool(allow_csv_fallback)
    if "allow_synthetic_fallback" in param_names or supports_var_kwargs:
        kwargs["allow_synthetic_fallback"] = bool(allow_synthetic_fallback)
    if research_config is not None:
        kwargs["research_config"] = research_config
    if not exact_split:
        return run_candidate_research(**kwargs)

    split_param_candidates = ("split", "requested_split", "exact_split", "split_windows")
    for param_name in split_param_candidates:
        if param_name in param_names or supports_var_kwargs:
            kwargs[param_name] = dict(exact_split)
            return run_candidate_research(**kwargs)

    family_mappings = (
        {
            "train_start": "train_start",
            "train_end": "train_end",
            "val_start": "val_start",
            "val_end": "val_end",
            "oos_start": "oos_start",
            "oos_end": "oos_end",
        },
        {
            "train_start": "train_start",
            "train_end": "train_end",
            "validation_start": "val_start",
            "validation_end": "val_end",
            "oos_start": "oos_start",
            "oos_end": "oos_end",
        },
        {
            "train_start": "train_start",
            "train_end": "train_end",
            "val_start": "val_start",
            "val_end": "val_end",
            "test_start": "oos_start",
            "test_end": "oos_end",
        },
        {
            "train_start": "train_start",
            "train_end": "train_end",
            "validation_start": "val_start",
            "validation_end": "val_end",
            "test_start": "oos_start",
            "test_end": "oos_end",
        },
    )
    for mapping in family_mappings:
        if set(mapping).issubset(param_names):
            kwargs.update(
                {param_name: exact_split[split_key] for param_name, split_key in mapping.items()}
            )
            return run_candidate_research(**kwargs)

    raise ValueError(
        "exact split requested, but run_candidate_research does not expose a supported "
        "exact-window interface yet"
    )


def _shortlist_robust_score_params(score_config: dict[str, Any] | None) -> dict[str, Any] | None:
    scope = _score_config_scope(score_config)
    rank_weights = scope.get("candidate_rank_score_weights")
    if not isinstance(rank_weights, dict):
        rank_weights = {}
    reject = scope.get("reject_thresholds")
    reject_thresholds = reject if isinstance(reject, dict) else {}
    shortlist_selection = scope.get("shortlist_selection")
    shortlist_cfg = shortlist_selection if isinstance(shortlist_selection, dict) else {}
    robust_overrides_raw = shortlist_cfg.get("robust_score_params")
    robust_overrides = robust_overrides_raw if isinstance(robust_overrides_raw, dict) else {}
    research_raw = scope.get("research")
    research_cfg = research_raw if isinstance(research_raw, dict) else {}

    params: dict[str, Any] = {}
    for key in ROBUST_SCORE_PARAM_KEYS:
        if key in rank_weights:
            params[key] = rank_weights[key]
    if "turnover_threshold" not in params and "max_turnover" in reject_thresholds:
        params["turnover_threshold"] = reject_thresholds["max_turnover"]
    for key in ROBUST_SCORE_PARAM_KEYS:
        if key in robust_overrides:
            params[key] = robust_overrides[key]
    for key in (
        "strict_selection_gate",
        "enforce_selection_reject_gate",
        "dsr_gate_floor",
        "spa_gate_ceiling",
        "pbo_gate_ceiling",
    ):
        if key in research_cfg and key not in robust_overrides:
            params[key] = research_cfg[key]
    return params or None


def _effective_shortlist_robust_score_params(
    runtime_config: RuntimeConfig | None,
    score_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    profile_activation = (
        research_config_to_overrides(runtime_config) if runtime_config is not None else None
    )
    params = dict(
        profile_activation["robust_score_params"] if profile_activation is not None else {}
    )
    params.update(_shortlist_robust_score_params(score_config) or {})

    effective_research = _effective_score_config_research(runtime_config, score_config)
    for key in (
        "strict_selection_gate",
        "enforce_selection_reject_gate",
        "dsr_gate_floor",
        "spa_gate_ceiling",
        "pbo_gate_ceiling",
    ):
        if key in effective_research:
            params[key] = effective_research[key]
    return params or None


def _write_summary_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "name",
                "strategy_class",
                "family",
                "strategy_timeframe",
                "selection_score",
                "selection_split",
                "validation_hurdle_pass",
                "promotion_eligible",
                "hard_reject",
                "validation_return",
                "validation_sharpe",
                "validation_deflated_sharpe",
                "validation_pbo",
                "validation_turnover",
                "validation_mdd",
                "validation_trade_count",
                "oos_return",
                "oos_sharpe",
                "oos_deflated_sharpe",
                "oos_pbo",
                "oos_turnover",
                "oos_mdd",
                "trade_count",
                "cross_candidate_corr",
            ],
        )
        writer.writeheader()
        for row in candidates:
            validation = dict(row.get("validation") or row.get("val") or {})
            oos = dict(row.get("oos") or {})
            writer.writerow(
                {
                    "candidate_id": row.get("candidate_id"),
                    "name": row.get("name"),
                    "strategy_class": row.get("strategy_class"),
                    "family": row.get("family"),
                    "strategy_timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                    "selection_score": float(row.get("selection_score", 0.0)),
                    "selection_split": "validation",
                    "validation_hurdle_pass": bool(
                        row.get("validation_hurdle_pass", row.get("pass", False))
                    ),
                    "promotion_eligible": False,
                    "hard_reject": bool(row.get("hard_reject", False)),
                    "validation_return": float(validation.get("return", 0.0)),
                    "validation_sharpe": float(validation.get("sharpe", 0.0)),
                    "validation_deflated_sharpe": float(validation.get("deflated_sharpe", 0.0)),
                    "validation_pbo": float(validation.get("pbo", 1.0)),
                    "validation_turnover": float(validation.get("turnover", 0.0)),
                    "validation_mdd": float(validation.get("mdd", 0.0)),
                    "validation_trade_count": float(
                        validation.get("trade_count", validation.get("trades", 0.0))
                    ),
                    "oos_return": float(oos.get("return", 0.0)),
                    "oos_sharpe": float(oos.get("sharpe", 0.0)),
                    "oos_deflated_sharpe": float(oos.get("deflated_sharpe", 0.0)),
                    "oos_pbo": float(oos.get("pbo", 1.0)),
                    "oos_turnover": float(oos.get("turnover", 0.0)),
                    "oos_mdd": float(oos.get("mdd", 0.0)),
                    "trade_count": float(oos.get("trade_count", 0.0)),
                    "cross_candidate_corr": float(oos.get("cross_candidate_corr", 0.0)),
                }
            )


def _render_shortlist_markdown(
    *,
    report_path: Path,
    shortlist: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lines = [
        "# Candidate Research Shortlist",
        "",
        f"- Source report: `{report_path}`",
        f"- Candidate count: {len(shortlist)}",
        "- Selection split: `validation`; reported OOS is a locked, report-only lockbox.",
        "- Statistical limitation: row-level `approx_pbo` / SPA-like diagnostics do not satisfy family-wide CSCV/PBO or Hansen SPA; this shortlist is not promotion-eligible by itself.",
        "- Embargo limitation: one boundary bar is a minimum guard, not proof of purging for unknown multi-bar holding/label horizons.",
        "- First-class fallback: `risk_off_cash` / `no_position` remains a valid research output.",
        "",
        "| # | Name | Strategy | TF | Family | Validation Sharpe | Validation DSR | Validation PBO | Report-only OOS Sharpe | Score |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(shortlist, start=1):
        validation = dict(row.get("validation") or row.get("val") or {})
        oos = dict(row.get("oos") or {})
        lines.append(
            "| "
            f"{idx} | "
            f"{row.get('name', '')} | "
            f"{row.get('strategy_class', '')} | "
            f"{row.get('strategy_timeframe') or row.get('timeframe') or ''} | "
            f"{row.get('family', '')} | "
            f"{float(validation.get('sharpe', 0.0)):.3f} | "
            f"{float(validation.get('deflated_sharpe', 0.0)):.3f} | "
            f"{float(validation.get('pbo', 1.0)):.3f} | "
            f"{float(oos.get('sharpe', 0.0)):.3f} | "
            f"{float(row.get('shortlist_score', row.get('selection_score', 0.0))):.3f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _progress_metric_summary(split_block: Any) -> dict[str, float]:
    block = dict(split_block or {})
    return {
        "total_return": float(block.get("total_return", block.get("return", 0.0)) or 0.0),
        "sharpe": float(block.get("sharpe", 0.0) or 0.0),
        "max_drawdown": float(block.get("max_drawdown", block.get("mdd", 0.0)) or 0.0),
        "trade_count": float(block.get("trade_count", block.get("trades", 0.0)) or 0.0),
    }


def _progress_elapsed_seconds(payload: Mapping[str, Any]) -> float:
    return round(float(payload.get("elapsed_seconds", 0.0) or 0.0), 6)


def _merge_slowest_entries(
    existing: list[dict[str, Any]],
    candidate: Mapping[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in existing]
    rows.append(dict(candidate))
    rows.sort(
        key=lambda row: float(row.get("elapsed_seconds", 0.0) or 0.0),
        reverse=True,
    )
    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for row in rows:
        row_key = "|".join(
            [
                str(row.get("symbol") or ""),
                str(row.get("timeframe") or ""),
                str(row.get("source") or ""),
                str(row.get("kind") or ""),
            ]
        )
        if row_key in seen_keys:
            continue
        seen_keys.add(row_key)
        deduped.append(row)
        if len(deduped) >= limit:
            break
    return deduped


class _ResearchProgressWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        manifest_path: Path | None,
        score_config_path: str,
        base_timeframe: str,
        strategy_timeframes: list[str],
        symbol_universe: list[str],
        requested_split: dict[str, Any] | None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.started_at = datetime.now(UTC)
        self.stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        self.json_path = self.output_dir / f"candidate_research_progress_{self.stamp}.json"
        self.json_latest = self.output_dir / "candidate_research_progress_latest.json"
        self.md_path = self.output_dir / f"candidate_research_progress_{self.stamp}.md"
        self.md_latest = self.output_dir / "candidate_research_progress_latest.md"
        self.log_path = self.output_dir / f"candidate_research_progress_{self.stamp}.log"
        self.log_latest = self.output_dir / "candidate_research_progress_latest.log"
        self.state: dict[str, Any] = {
            "artifact_kind": "candidate_research_progress",
            "status": "running",
            "started_at": self.started_at.isoformat().replace("+00:00", "Z"),
            "updated_at": self.started_at.isoformat().replace("+00:00", "Z"),
            "current_stage": "initialized",
            "manifest_path": str(manifest_path) if manifest_path else "",
            "score_config_path": str(score_config_path or ""),
            "base_timeframe": str(base_timeframe),
            "strategy_timeframes": list(strategy_timeframes),
            "symbol_universe": list(symbol_universe),
            "requested_split": dict(requested_split or {}),
            "progress": {
                "candidate_count": 0,
                "evaluated_count": 0,
                "keep_count": 0,
                "selected_count": 0,
            },
            "resources": {},
            "resource_load": {
                "overall": {
                    "completed_units": 0,
                    "total_units": 0,
                    "completion_ratio": 0.0,
                    "active_detail": {},
                },
                "bundle": {
                    "status": "pending",
                    "loaded_count": 0,
                    "total_count": 0,
                    "elapsed_seconds": 0.0,
                    "source_counts": {},
                    "current_timeframe": {},
                    "current_symbol_fetch": {},
                    "recent_timeframes": [],
                    "latest_symbol_fetch": {},
                    "latest_window": {},
                    "recent_windows": [],
                    "latest_item": {},
                    "slowest_items": [],
                },
                "feature": {
                    "status": "pending",
                    "loaded_count": 0,
                    "symbol_count": 0,
                    "elapsed_seconds": 0.0,
                    "current_symbol": {},
                    "latest_partition_scan": {},
                    "latest_collect": {},
                    "latest_symbol": {},
                    "slowest_symbols": [],
                },
                "benchmark": {
                    "status": "pending",
                    "built_count": 0,
                    "timeframe_count": 0,
                    "elapsed_seconds": 0.0,
                    "current_timeframe": {},
                    "latest_timeframe": {},
                    "slowest_timeframes": [],
                },
            },
            "latest_candidate": {},
            "top_stage1_candidates": [],
            "selected_candidates": [],
            "report_preview": {},
            "recent_events": [],
            "final_artifacts": {},
        }
        self._write_latest()
        self._append_log(
            "initialized",
            {
                "manifest_path": str(manifest_path) if manifest_path else "",
                "candidate_count": 0,
            },
        )

    @property
    def artifact_paths(self) -> dict[str, str]:
        return {
            "json": str(self.json_path),
            "json_latest": str(self.json_latest),
            "markdown": str(self.md_path),
            "markdown_latest": str(self.md_latest),
            "log": str(self.log_path),
            "log_latest": str(self.log_latest),
        }

    def __call__(self, event: str, payload: Mapping[str, Any]) -> None:
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        progress = dict(self.state.get("progress") or {})
        self.state["updated_at"] = timestamp
        self.state["current_stage"] = str(event)
        self.state.setdefault("recent_events", [])
        recent_events = list(self.state["recent_events"])
        recent_events.append(
            {
                "timestamp": timestamp,
                "event": str(event),
                "context": dict(payload or {}),
            }
        )
        self.state["recent_events"] = recent_events[-20:]
        resource_load = dict(self.state.get("resource_load") or {})
        bundle_state = dict(resource_load.get("bundle") or {})
        feature_state = dict(resource_load.get("feature") or {})
        benchmark_state = dict(resource_load.get("benchmark") or {})

        if event == "resources_loaded":
            progress["candidate_count"] = int(
                payload.get("candidate_count", progress.get("candidate_count", 0)) or 0
            )
            self.state["resources"] = dict(payload or {})
        elif event == "resource_bundle_load_started":
            bundle_state.update(
                {
                    "status": "running",
                    "loaded_count": 0,
                    "total_count": int(payload.get("total_count", 0) or 0),
                    "symbol_count": int(payload.get("symbol_count", 0) or 0),
                    "timeframe_count": int(payload.get("timeframe_count", 0) or 0),
                    "normalized_timeframes": list(payload.get("normalized_timeframes") or []),
                    "symbol_universe": list(payload.get("symbol_universe") or []),
                    "elapsed_seconds": 0.0,
                    "source_counts": {},
                    "current_timeframe": {},
                    "current_symbol_fetch": {},
                    "recent_timeframes": [],
                    "latest_symbol_fetch": {},
                    "latest_window": {},
                    "recent_windows": [],
                    "latest_item": {},
                    "slowest_items": [],
                }
            )
        elif event == "resource_bundle_timeframe_started":
            bundle_state.update(
                {
                    "status": "running",
                    "current_timeframe": dict(payload or {}),
                }
            )
        elif event == "resource_bundle_timeframe_completed":
            recent_timeframes = [
                dict(row) for row in list(bundle_state.get("recent_timeframes") or [])
            ]
            recent_timeframes.append(dict(payload or {}))
            bundle_state.update(
                {
                    "status": "running",
                    "current_timeframe": {},
                    "recent_timeframes": recent_timeframes[-5:],
                }
            )
        elif event == "resource_bundle_symbol_fetch_started":
            bundle_state.update(
                {
                    "status": "running",
                    "current_symbol_fetch": dict(payload or {}),
                }
            )
        elif event == "resource_bundle_symbol_window_loaded":
            recent_windows = [dict(row) for row in list(bundle_state.get("recent_windows") or [])]
            recent_windows.append(dict(payload or {}))
            bundle_state.update(
                {
                    "status": "running",
                    "latest_window": dict(payload or {}),
                    "recent_windows": recent_windows[-8:],
                }
            )
        elif event == "resource_bundle_symbol_fetch_completed":
            bundle_state.update(
                {
                    "status": "running",
                    "current_symbol_fetch": {},
                    "latest_symbol_fetch": dict(payload or {}),
                }
            )
        elif event == "resource_bundle_item_loaded":
            source_counts = dict(bundle_state.get("source_counts") or {})
            source = str(payload.get("source") or "")
            if source:
                source_counts[source] = int(source_counts.get(source, 0) or 0) + 1
            bundle_state.update(
                {
                    "status": "running",
                    "loaded_count": int(
                        payload.get("loaded_count", bundle_state.get("loaded_count", 0)) or 0
                    ),
                    "total_count": int(
                        payload.get("total_count", bundle_state.get("total_count", 0)) or 0
                    ),
                    "source_counts": source_counts,
                    "latest_item": dict(payload or {}),
                    "slowest_items": _merge_slowest_entries(
                        list(bundle_state.get("slowest_items") or []),
                        dict(payload or {}),
                    ),
                }
            )
        elif event == "resource_bundle_load_completed":
            bundle_state.update(
                {
                    "status": "completed",
                    "loaded_count": int(
                        payload.get("bundle_count", bundle_state.get("loaded_count", 0)) or 0
                    ),
                    "bundle_count": int(payload.get("bundle_count", 0) or 0),
                    "total_count": int(
                        payload.get("total_count", bundle_state.get("total_count", 0)) or 0
                    ),
                    "elapsed_seconds": _progress_elapsed_seconds(payload),
                    "source_counts": dict(
                        payload.get("source_counts") or bundle_state.get("source_counts") or {}
                    ),
                    "current_timeframe": {},
                    "current_symbol_fetch": {},
                }
            )
        elif event == "resource_feature_load_started":
            feature_state.update(
                {
                    "status": "running",
                    "loaded_count": 0,
                    "symbol_count": int(payload.get("symbol_count", 0) or 0),
                    "feature_symbols": list(payload.get("feature_symbols") or []),
                    "elapsed_seconds": 0.0,
                    "current_symbol": {},
                    "latest_partition_scan": {},
                    "latest_collect": {},
                    "latest_symbol": {},
                    "slowest_symbols": [],
                }
            )
        elif event == "resource_feature_symbol_started":
            feature_state.update(
                {
                    "status": "running",
                    "current_symbol": dict(payload or {}),
                }
            )
        elif event == "resource_feature_partition_scan_completed":
            feature_state.update(
                {
                    "status": "running",
                    "latest_partition_scan": dict(payload or {}),
                }
            )
        elif event == "resource_feature_collect_started":
            feature_state.update(
                {
                    "status": "running",
                    "latest_collect": {
                        **dict(payload or {}),
                        "status": "running",
                    },
                }
            )
        elif event == "resource_feature_collect_completed":
            feature_state.update(
                {
                    "status": "running",
                    "latest_collect": {
                        **dict(payload or {}),
                        "status": "completed",
                    },
                }
            )
        elif event == "resource_feature_symbol_loaded":
            feature_state.update(
                {
                    "status": "running",
                    "loaded_count": int(
                        payload.get("loaded_count", feature_state.get("loaded_count", 0)) or 0
                    ),
                    "symbol_count": int(
                        payload.get("symbol_count", feature_state.get("symbol_count", 0)) or 0
                    ),
                    "current_symbol": {},
                    "latest_symbol": dict(payload or {}),
                    "slowest_symbols": _merge_slowest_entries(
                        list(feature_state.get("slowest_symbols") or []),
                        {**dict(payload or {}), "kind": "feature_symbol"},
                    ),
                }
            )
        elif event == "resource_feature_load_completed":
            feature_state.update(
                {
                    "status": "completed",
                    "loaded_count": int(
                        payload.get("feature_frame_count", feature_state.get("loaded_count", 0))
                        or 0
                    ),
                    "symbol_count": int(
                        payload.get("symbol_count", feature_state.get("symbol_count", 0)) or 0
                    ),
                    "feature_frame_count": int(payload.get("feature_frame_count", 0) or 0),
                    "nonempty_symbol_count": int(payload.get("nonempty_symbol_count", 0) or 0),
                    "total_rows": int(payload.get("total_rows", 0) or 0),
                    "elapsed_seconds": _progress_elapsed_seconds(payload),
                    "current_symbol": {},
                }
            )
        elif event == "resource_benchmark_build_started":
            benchmark_state.update(
                {
                    "status": "running",
                    "built_count": 0,
                    "timeframe_count": int(payload.get("timeframe_count", 0) or 0),
                    "normalized_timeframes": list(payload.get("normalized_timeframes") or []),
                    "elapsed_seconds": 0.0,
                    "current_timeframe": {},
                    "latest_timeframe": {},
                    "slowest_timeframes": [],
                }
            )
        elif event == "resource_benchmark_timeframe_started":
            benchmark_state.update(
                {
                    "status": "running",
                    "current_timeframe": dict(payload or {}),
                }
            )
        elif event == "resource_benchmark_timeframe_built":
            benchmark_state.update(
                {
                    "status": "running",
                    "built_count": int(
                        payload.get("built_count", benchmark_state.get("built_count", 0)) or 0
                    ),
                    "timeframe_count": int(
                        payload.get("timeframe_count", benchmark_state.get("timeframe_count", 0))
                        or 0
                    ),
                    "current_timeframe": {},
                    "latest_timeframe": dict(payload or {}),
                    "slowest_timeframes": _merge_slowest_entries(
                        list(benchmark_state.get("slowest_timeframes") or []),
                        {**dict(payload or {}), "kind": "benchmark_timeframe"},
                    ),
                }
            )
        elif event == "resource_benchmark_build_completed":
            benchmark_state.update(
                {
                    "status": "completed",
                    "built_count": int(
                        payload.get("benchmark_count", benchmark_state.get("built_count", 0)) or 0
                    ),
                    "benchmark_count": int(payload.get("benchmark_count", 0) or 0),
                    "timeframe_count": int(
                        payload.get("timeframe_count", benchmark_state.get("timeframe_count", 0))
                        or 0
                    ),
                    "nonempty_timeframe_count": int(
                        payload.get("nonempty_timeframe_count", 0) or 0
                    ),
                    "elapsed_seconds": _progress_elapsed_seconds(payload),
                    "current_timeframe": {},
                }
            )
        elif event == "candidate_evaluated":
            progress["candidate_count"] = int(
                payload.get("candidate_count", progress.get("candidate_count", 0)) or 0
            )
            progress["evaluated_count"] = int(
                payload.get("candidate_index", progress.get("evaluated_count", 0)) or 0
            )
            candidate_snapshot = dict(payload or {})
            self.state["latest_candidate"] = candidate_snapshot
            top_rows = [dict(row) for row in list(self.state.get("top_stage1_candidates") or [])]
            top_rows.append(candidate_snapshot)
            top_rows.sort(
                key=lambda row: float(
                    row.get("stage1_prefilter_score", float("-inf")) or float("-inf")
                ),
                reverse=True,
            )
            deduped: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for row in top_rows:
                candidate_id = str(row.get("candidate_id") or row.get("name") or "")
                if candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                deduped.append(row)
                if len(deduped) >= 5:
                    break
            self.state["top_stage1_candidates"] = deduped
        elif event == "stage1_ranked":
            progress["candidate_count"] = int(
                payload.get("candidate_count", progress.get("candidate_count", 0)) or 0
            )
            progress["keep_count"] = int(
                payload.get("keep_count", progress.get("keep_count", 0)) or 0
            )
            self.state["top_stage1_candidates"] = [
                dict(row) for row in list(payload.get("top_stage1_candidates") or [])
            ]
            self.state["keep_ratio_applied"] = float(payload.get("keep_ratio_applied", 0.0) or 0.0)
        elif event == "stage2_selected":
            progress["selected_count"] = int(
                payload.get("selected_count", progress.get("selected_count", 0)) or 0
            )
            self.state["selected_candidates"] = [
                dict(row) for row in list(payload.get("selected_candidates") or [])
            ]
        elif event == "report_ready":
            self.state["report_preview"] = dict(payload or {})

        overall_total_units = (
            int(bundle_state.get("total_count", 0) or 0)
            + int(feature_state.get("symbol_count", 0) or 0)
            + int(benchmark_state.get("timeframe_count", 0) or 0)
        )
        overall_completed_units = (
            int(bundle_state.get("loaded_count", 0) or 0)
            + int(feature_state.get("loaded_count", 0) or 0)
            + int(benchmark_state.get("built_count", 0) or 0)
        )
        active_detail: dict[str, Any] = {}
        if bundle_state.get("current_symbol_fetch"):
            active_detail = {
                "phase": "bundle_symbol_fetch",
                **dict(bundle_state.get("current_symbol_fetch") or {}),
            }
        elif bundle_state.get("current_timeframe"):
            active_detail = {
                "phase": "bundle_timeframe",
                **dict(bundle_state.get("current_timeframe") or {}),
            }
        elif feature_state.get("current_symbol"):
            active_detail = {
                "phase": "feature_symbol",
                **dict(feature_state.get("current_symbol") or {}),
            }
        elif benchmark_state.get("current_timeframe"):
            active_detail = {
                "phase": "benchmark_timeframe",
                **dict(benchmark_state.get("current_timeframe") or {}),
            }

        resource_load["overall"] = {
            "completed_units": overall_completed_units,
            "total_units": overall_total_units,
            "completion_ratio": round(
                (overall_completed_units / overall_total_units) if overall_total_units > 0 else 0.0,
                6,
            ),
            "active_detail": active_detail,
        }
        resource_load["bundle"] = bundle_state
        resource_load["feature"] = feature_state
        resource_load["benchmark"] = benchmark_state
        self.state["resource_load"] = resource_load
        self.state["progress"] = progress
        self._append_log(event, dict(payload or {}))
        self._write_latest()

    def complete(
        self,
        *,
        final_artifacts: Mapping[str, Any],
        report: Mapping[str, Any],
        shortlisted: list[dict[str, Any]],
    ) -> None:
        self.state["status"] = "completed"
        self.state["completed_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self.state["current_stage"] = "completed"
        self.state["final_artifacts"] = dict(final_artifacts or {})
        self.state["final_summary"] = {
            "reported_candidate_count": len(list(report.get("candidates") or [])),
            "shortlisted_count": len(shortlisted),
            "top_shortlist_names": [
                str(row.get("name") or row.get("candidate_id") or "")
                for row in shortlisted[: min(5, len(shortlisted))]
            ],
        }
        self._append_log(
            "completed",
            {
                "shortlisted_count": len(shortlisted),
                "reported_candidate_count": len(list(report.get("candidates") or [])),
            },
        )
        self._write_latest()

    def fail(self, error: str) -> None:
        self.state["status"] = "failed"
        self.state["current_stage"] = "failed"
        self.state["error"] = str(error)
        self.state["failed_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        self._append_log("failed", {"error": str(error)})
        self._write_latest()

    def _append_log(self, event: str, payload: Mapping[str, Any]) -> None:
        line = (
            f"[{datetime.now(UTC).isoformat().replace('+00:00', 'Z')}] "
            f"{event} "
            f"{json.dumps(dict(payload or {}), ensure_ascii=False, sort_keys=True)}\n"
        )
        for path in (self.log_path, self.log_latest):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def _write_latest(self) -> None:
        payload = json.dumps(self.state, indent=2, ensure_ascii=False)
        markdown = self._render_markdown()
        for path, content in (
            (self.json_path, payload),
            (self.json_latest, payload),
            (self.md_path, markdown),
            (self.md_latest, markdown),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _render_markdown(self) -> str:
        progress = dict(self.state.get("progress") or {})
        lines = [
            "# Candidate Research Progress",
            "",
            f"- Status: `{self.state.get('status', 'running')}`",
            f"- Current stage: `{self.state.get('current_stage', 'initialized')}`",
            f"- Started at: `{self.state.get('started_at', '')}`",
            f"- Updated at: `{self.state.get('updated_at', '')}`",
            f"- Candidate progress: `{int(progress.get('evaluated_count', 0))}/{int(progress.get('candidate_count', 0))}`",
            f"- Stage-2 keep count: `{int(progress.get('keep_count', 0))}`",
            f"- Selected count: `{int(progress.get('selected_count', 0))}`",
            "",
        ]

        resource_load = dict(self.state.get("resource_load") or {})
        overall_state = dict(resource_load.get("overall") or {})
        bundle_state = dict(resource_load.get("bundle") or {})
        feature_state = dict(resource_load.get("feature") or {})
        benchmark_state = dict(resource_load.get("benchmark") or {})
        if resource_load:
            lines.extend(
                [
                    "## Resource load progress",
                    "",
                    f"- Overall resource progress: `{int(overall_state.get('completed_units', 0))}/{int(overall_state.get('total_units', 0))}` "
                    f"(`{float(overall_state.get('completion_ratio', 0.0) or 0.0):.1%}`)",
                    f"- Bundle cache: `{bundle_state.get('status', 'pending')}` "
                    f"(`{int(bundle_state.get('loaded_count', 0))}/{int(bundle_state.get('total_count', 0))}`, "
                    f"`{float(bundle_state.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)",
                    f"- Feature cache: `{feature_state.get('status', 'pending')}` "
                    f"(`{int(feature_state.get('loaded_count', 0))}/{int(feature_state.get('symbol_count', 0))}`, "
                    f"`{float(feature_state.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)",
                    f"- Benchmark build: `{benchmark_state.get('status', 'pending')}` "
                    f"(`{int(benchmark_state.get('built_count', 0))}/{int(benchmark_state.get('timeframe_count', 0))}`, "
                    f"`{float(benchmark_state.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)",
                ]
            )
            active_detail = dict(overall_state.get("active_detail") or {})
            if active_detail:
                phase = str(active_detail.get("phase") or "")
                if phase == "bundle_timeframe":
                    lines.append(
                        "- Active bundle timeframe scan: "
                        f"`{active_detail.get('timeframe', '')}` "
                        f"(`{int(active_detail.get('timeframe_index', 0))}/{int(active_detail.get('timeframe_count', 0))}`, "
                        f"symbols `{int(active_detail.get('symbol_count', 0))}`)"
                    )
                elif phase == "bundle_symbol_fetch":
                    lines.append(
                        "- Active bundle symbol fetch: "
                        f"`{active_detail.get('symbol', '')}@{active_detail.get('timeframe', '')}` "
                        f"(`{int(active_detail.get('symbol_index', 0))}/{int(active_detail.get('symbol_count', 0))}`)"
                    )
                elif phase == "feature_symbol":
                    lines.append(
                        "- Active feature symbol: "
                        f"`{active_detail.get('symbol', '')}` "
                        f"(`{int(active_detail.get('symbol_index', 0))}/{int(active_detail.get('symbol_count', 0))}`)"
                    )
                elif phase == "benchmark_timeframe":
                    lines.append(
                        "- Active benchmark timeframe: "
                        f"`{active_detail.get('timeframe', '')}` "
                        f"(`{int(active_detail.get('timeframe_index', 0))}/{int(active_detail.get('timeframe_count', 0))}`)"
                    )
            recent_timeframes = [
                dict(row) for row in list(bundle_state.get("recent_timeframes") or [])
            ]
            if recent_timeframes:
                rendered = ", ".join(
                    f"{row.get('timeframe', '')}:{int(row.get('parquet_symbol_count', 0) or 0)}/{int(row.get('symbol_count', 0) or 0)} "
                    f"loaded in {float(row.get('elapsed_seconds', 0.0) or 0.0):.3f}s"
                    for row in recent_timeframes[-3:]
                )
                lines.append(f"- Recent bundle timeframe scans: `{rendered}`")
            latest_symbol_fetch = dict(bundle_state.get("latest_symbol_fetch") or {})
            if latest_symbol_fetch:
                lines.append(
                    "- Latest bundle symbol fetch: "
                    f"`{latest_symbol_fetch.get('symbol', '')}@{latest_symbol_fetch.get('timeframe', '')}` "
                    f"(rows `{int(latest_symbol_fetch.get('row_count', 0) or 0)}`, "
                    f"elapsed `{float(latest_symbol_fetch.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)"
                )
            latest_window = dict(bundle_state.get("latest_window") or {})
            if latest_window:
                lines.append(
                    "- Latest bundle window: "
                    f"`{latest_window.get('unit_kind', 'chunk')}` "
                    f"`{int(latest_window.get('unit_index', 0) or 0)}/{int(latest_window.get('unit_count', 0) or 0)}` "
                    f"for `{latest_window.get('symbol', '')}@{latest_window.get('timeframe', '')}` "
                    f"(rows `{int(latest_window.get('row_count', 0) or 0)}`, "
                    f"elapsed `{float(latest_window.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)"
                )
            recent_windows = [dict(row) for row in list(bundle_state.get("recent_windows") or [])]
            if recent_windows:
                rendered = ", ".join(
                    f"{row.get('symbol', '')}@{row.get('timeframe', '')}:{row.get('unit_kind', 'chunk')} "
                    f"{int(row.get('unit_index', 0) or 0)}/{int(row.get('unit_count', 0) or 0)} "
                    f"in {float(row.get('elapsed_seconds', 0.0) or 0.0):.3f}s"
                    for row in recent_windows[-3:]
                )
                lines.append(f"- Recent bundle windows: `{rendered}`")
            latest_bundle = dict(bundle_state.get("latest_item") or {})
            if latest_bundle:
                lines.append(
                    "- Latest bundle item: "
                    f"`{latest_bundle.get('symbol', '')}@{latest_bundle.get('timeframe', '')}` "
                    f"(source `{latest_bundle.get('source', '')}`, "
                    f"bars `{int(latest_bundle.get('bar_count', 0) or 0)}`, "
                    f"elapsed `{float(latest_bundle.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)"
                )
            source_counts = dict(bundle_state.get("source_counts") or {})
            if source_counts:
                rendered_counts = ", ".join(
                    f"{source}={count}" for source, count in sorted(source_counts.items())
                )
                lines.append(f"- Bundle sources: `{rendered_counts}`")
            slowest_bundle = list(bundle_state.get("slowest_items") or [])
            if slowest_bundle:
                rendered = ", ".join(
                    f"{row.get('symbol', '')}@{row.get('timeframe', '')}:{float(row.get('elapsed_seconds', 0.0) or 0.0):.3f}s"
                    for row in slowest_bundle[:3]
                )
                lines.append(f"- Slowest bundle items: `{rendered}`")
            latest_feature = dict(feature_state.get("latest_symbol") or {})
            if latest_feature:
                lines.append(
                    "- Latest feature symbol: "
                    f"`{latest_feature.get('symbol', '')}` "
                    f"(rows `{int(latest_feature.get('row_count', 0) or 0)}`, "
                    f"elapsed `{float(latest_feature.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)"
                )
            latest_partition_scan = dict(feature_state.get("latest_partition_scan") or {})
            if latest_partition_scan:
                lines.append(
                    "- Latest feature partition scan: "
                    f"`{latest_partition_scan.get('symbol', '')}` "
                    f"(partitions `{int(latest_partition_scan.get('partition_count', 0) or 0)}`, "
                    f"files `{int(latest_partition_scan.get('parquet_file_count', 0) or 0)}`, "
                    f"elapsed `{float(latest_partition_scan.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)"
                )
            latest_collect = dict(feature_state.get("latest_collect") or {})
            if latest_collect:
                lines.append(
                    "- Latest feature collect: "
                    f"`{latest_collect.get('symbol', '')}` "
                    f"(`{latest_collect.get('status', 'completed')!s}`; "
                    f"rows `{int(latest_collect.get('row_count', 0) or 0)}`, "
                    f"files `{int(latest_collect.get('parquet_file_count', 0) or 0)}`, "
                    f"elapsed `{float(latest_collect.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)"
                )
            if int(feature_state.get("total_rows", 0) or 0) > 0:
                lines.append(
                    "- Feature rows loaded: "
                    f"`{int(feature_state.get('total_rows', 0))}` "
                    f"across `{int(feature_state.get('nonempty_symbol_count', 0))}` non-empty symbols"
                )
            slowest_feature = list(feature_state.get("slowest_symbols") or [])
            if slowest_feature:
                rendered = ", ".join(
                    f"{row.get('symbol', '')}:{float(row.get('elapsed_seconds', 0.0) or 0.0):.3f}s"
                    for row in slowest_feature[:3]
                )
                lines.append(f"- Slowest feature symbols: `{rendered}`")
            latest_benchmark = dict(benchmark_state.get("latest_timeframe") or {})
            if latest_benchmark:
                lines.append(
                    "- Latest benchmark timeframe: "
                    f"`{latest_benchmark.get('timeframe', '')}` "
                    f"(returns `{int(latest_benchmark.get('return_count', 0) or 0)}`, "
                    f"elapsed `{float(latest_benchmark.get('elapsed_seconds', 0.0) or 0.0):.3f}s`)"
                )
            if benchmark_state.get("status") == "completed":
                lines.append(
                    "- Benchmark non-empty timeframes: "
                    f"`{int(benchmark_state.get('nonempty_timeframe_count', 0) or 0)}`"
                )
            slowest_benchmark = list(benchmark_state.get("slowest_timeframes") or [])
            if slowest_benchmark:
                rendered = ", ".join(
                    f"{row.get('timeframe', '')}:{float(row.get('elapsed_seconds', 0.0) or 0.0):.3f}s"
                    for row in slowest_benchmark[:3]
                )
                lines.append(f"- Slowest benchmark timeframes: `{rendered}`")
            lines.extend(["", ""])

        latest_candidate = dict(self.state.get("latest_candidate") or {})
        if latest_candidate:
            lines.extend(
                [
                    "## Latest candidate",
                    "",
                    f"- Name: `{latest_candidate.get('name', '')}`",
                    f"- Stage-1 prefilter score: `{float(latest_candidate.get('stage1_prefilter_score', 0.0) or 0.0):.6f}`",
                    f"- OOS return/sharpe/maxDD: `{_progress_metric_summary(latest_candidate.get('oos'))['total_return']:+.4%}` / "
                    f"`{_progress_metric_summary(latest_candidate.get('oos'))['sharpe']:.4f}` / "
                    f"`{_progress_metric_summary(latest_candidate.get('oos'))['max_drawdown']:.4%}`",
                    "",
                ]
            )

        top_rows = [dict(row) for row in list(self.state.get("top_stage1_candidates") or [])]
        if top_rows:
            lines.extend(
                [
                    "## Top stage-1 candidates",
                    "",
                    "| # | Name | TF | Stage1 | OOS Return | OOS Sharpe | OOS MaxDD |",
                    "|---:|---|---|---:|---:|---:|---:|",
                ]
            )
            for index, row in enumerate(top_rows, start=1):
                oos = _progress_metric_summary(row.get("oos"))
                lines.append(
                    "| "
                    f"{index} | "
                    f"{row.get('name', '')} | "
                    f"{row.get('strategy_timeframe', '')} | "
                    f"{float(row.get('stage1_prefilter_score', 0.0) or 0.0):.6f} | "
                    f"{oos['total_return']:+.4%} | "
                    f"{oos['sharpe']:.4f} | "
                    f"{oos['max_drawdown']:.4%} |"
                )
            lines.append("")

        selected_rows = [dict(row) for row in list(self.state.get("selected_candidates") or [])]
        if selected_rows:
            lines.extend(["## Current stage-2 selection", ""])
            for row in selected_rows:
                lines.append(
                    f"- `{row.get('name', '')}` "
                    f"(stage1 `{float(row.get('stage1_prefilter_score', 0.0) or 0.0):.6f}`, "
                    f"OOS `{_progress_metric_summary(row.get('oos'))['total_return']:+.4%}`)"
                )
            lines.append("")

        final_artifacts = dict(self.state.get("final_artifacts") or {})
        if final_artifacts:
            lines.extend(["## Final artifacts", ""])
            for key, value in final_artifacts.items():
                lines.append(f"- {key}: `{value}`")
            lines.append("")

        return "\n".join(lines)


def main() -> int:
    args = _build_parser().parse_args()
    try:
        runtime_config, runtime_config_path = _load_runtime_config_for_args(args)
    except ValueError as exc:
        raise SystemExit(f"[RESEARCH] {exc}")
    score_config: dict[str, Any] | None = None
    if str(args.score_config).strip():
        try:
            score_config = _load_score_config(Path(str(args.score_config)).resolve())
        except ValueError as exc:
            raise SystemExit(f"[RESEARCH] {exc}")
    score_config_scope = _score_config_scope(score_config)

    symbols = canonicalize_symbol_list(list(args.symbols))
    timeframes = [
        str(token).strip().lower() for token in list(args.timeframes) if str(token).strip()
    ]

    manifest_path = Path(str(args.manifest)).resolve() if str(args.manifest).strip() else None
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise SystemExit(f"[RESEARCH] manifest file not found: {manifest_path}")
        candidates = _load_manifest_candidates(manifest_path)
    else:
        candidates = build_default_candidate_rows(
            symbols=symbols,
            timeframes=timeframes,
            max_candidates=max(1, int(args.max_candidates)),
        )
    candidates = _restrict_candidates_to_symbol_universe(list(candidates), symbols)
    if args.dry_run:
        print(f"[RESEARCH] dry-run mode: candidate_count={len(candidates)}")
        return 0

    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_split = {
        "train_start": str(getattr(args, "train_start", "") or ""),
        "train_end": str(getattr(args, "train_end", "") or ""),
        "val_start": str(getattr(args, "validation_start", "") or ""),
        "val_end": str(getattr(args, "validation_end", "") or ""),
        "oos_start": str(getattr(args, "oos_start", "") or ""),
        "oos_end": str(getattr(args, "oos_end", "") or ""),
    }
    progress_writer = _ResearchProgressWriter(
        output_dir=output_dir,
        manifest_path=manifest_path,
        score_config_path=str(args.score_config),
        base_timeframe=str(args.base_timeframe),
        strategy_timeframes=timeframes,
        symbol_universe=symbols,
        requested_split=requested_split if any(requested_split.values()) else None,
    )

    coverage_summary: dict[str, Any] | None = None
    try:
        exact_split = _build_exact_split(args)
        if exact_split is not None and not bool(args.skip_coverage_rebuild):
            exact_split["strategy_timeframe"] = (
                timeframes[0] if timeframes else str(args.base_timeframe)
            )
            candidates, exact_split, coverage_summary = _rebuild_candidates_after_coverage(
                candidates=list(candidates),
                symbols=symbols,
                timeframes=timeframes,
                split=exact_split,
                progress_callback=progress_writer,
            )
        elif exact_split is not None:
            progress_writer(
                "coverage_rebuild_skipped",
                {
                    "requested_candidate_count": len(candidates),
                    "reason": "cli_skip_coverage_rebuild",
                },
            )
        research_symbols = _symbols_from_candidates(list(candidates), fallback=symbols)
        research_timeframes = _timeframes_from_candidates(list(candidates), fallback=timeframes)
        allow_synthetic_fallback = bool(
            getattr(args, "enable_synthetic_fallback", False)
        ) and not bool(getattr(args, "disable_synthetic_fallback", False))
        report = _run_candidate_research_with_optional_split(
            candidates=candidates,
            base_timeframe=str(args.base_timeframe),
            strategy_timeframes=research_timeframes,
            symbol_universe=research_symbols,
            stage1_keep_ratio=float(args.stage1_keep_ratio),
            max_candidates=max(1, int(args.max_candidates)),
            score_config=score_config_scope or None,
            exact_split=exact_split,
            min_bundle_bars=max(1, int(getattr(args, "min_bundle_bars", 360))),
            allow_csv_fallback=not bool(getattr(args, "disable_csv_fallback", False)),
            allow_synthetic_fallback=allow_synthetic_fallback,
            progress_callback=progress_writer,
            research_config=_strict_research_config(),
        )
    except ValueError as exc:
        progress_writer.fail(str(exc))
        raise SystemExit(f"[RESEARCH] {exc}")
    except Exception as exc:
        progress_writer.fail(str(exc))
        raise

    if exact_split is not None:
        report["requested_split"] = dict(exact_split)
        report["effective_split"] = dict(report.get("split") or exact_split)
        report["split_mode"] = "exact"
    else:
        report["effective_split"] = dict(report.get("split") or {})
        report["split_mode"] = "default"
    if coverage_summary is not None:
        report["coverage_rebuild"] = coverage_summary
    report["progress_artifacts"] = progress_writer.artifact_paths

    shortlist_config = _resolve_shortlist_selection_config(
        score_config_scope or None,
        top_k=max(1, int(args.top_k)),
    )
    profile_activation = (
        research_config_to_overrides(runtime_config) if runtime_config is not None else None
    )
    profile_score_overrides = (
        profile_activation["score_config_research"] if profile_activation is not None else {}
    )
    shortlist_score_params = dict(
        _effective_shortlist_robust_score_params(runtime_config, score_config_scope or None) or {}
    )
    shortlist_score_params.update(
        {
            "strict_selection_gate": True,
            "enforce_selection_reject_gate": True,
            "dsr_gate_floor": max(
                _STRICT_DSR_FLOOR,
                float(shortlist_score_params.get("dsr_gate_floor", _STRICT_DSR_FLOOR)),
            ),
            "spa_gate_ceiling": min(
                _STRICT_SPA_CEILING,
                float(shortlist_score_params.get("spa_gate_ceiling", _STRICT_SPA_CEILING)),
            ),
            "pbo_gate_ceiling": min(
                _STRICT_PBO_CEILING,
                float(shortlist_score_params.get("pbo_gate_ceiling", _STRICT_PBO_CEILING)),
            ),
        }
    )
    shortlist_input = list(report.get("candidates") or [])
    for row in shortlist_input:
        row["validation_hurdle_pass"] = row.get("pass") is True
        row["promotion_eligible"] = False
    profile_strict_selection = True
    shortlist_selection_mode = _SHORTLIST_SELECTION_MODE
    profile_caps = _profile_selection_caps(runtime_config)
    routing_requested = _registered_routing_requested(runtime_config, score_config_scope or None)
    proxy_rejected_count = 0
    missing_validation_rejected_count = 0
    strict_shortlist_input = shortlist_input
    if profile_strict_selection and shortlist_selection_mode == "val":
        strict_shortlist_input = [
            row for row in strict_shortlist_input if _has_actual_validation_metrics(row)
        ]
        missing_validation_rejected_count = len(shortlist_input) - len(strict_shortlist_input)
    if profile_strict_selection and routing_requested:
        before_proxy_filter = len(strict_shortlist_input)
        strict_shortlist_input = [
            row
            for row in strict_shortlist_input
            if _candidate_evaluation_mode(row) != "generic_fallback_proxy"
        ]
        proxy_rejected_count = before_proxy_filter - len(strict_shortlist_input)
    if profile_strict_selection:
        gated_shortlist_input = apply_selection_reject_and_dedup(
            strict_shortlist_input,
            mode=shortlist_selection_mode,
            robust_score_params=shortlist_score_params,
            max_per_symbol_basket=profile_caps["max_per_symbol_basket"],
            max_per_lineage=profile_caps["max_per_lineage"],
            max_per_family_basket=profile_caps["max_per_family_basket"],
            enabled=True,
        )
    else:
        gated_shortlist_input = strict_shortlist_input

    shortlist_weights_enabled = bool(shortlist_config["include_weights"]) and not (
        profile_strict_selection and shortlist_selection_mode == "val"
    )
    if runtime_config is not None or profile_strict_selection or routing_requested:
        runtime_research = runtime_config.research if runtime_config is not None else None
        score_research = _score_config_research(score_config_scope or None)
        report["research_profile_audit"] = {
            "config_path": runtime_config_path,
            "cost_rate_bps_override": (
                getattr(runtime_research, "cost_rate_bps_override", None)
                if runtime_research is not None
                else score_research.get("cost_rate_bps_override")
            ),
            "resolved_cost": {
                "cost_rate_multiplier": profile_score_overrides.get(
                    "cost_rate_multiplier",
                    score_research.get("cost_rate_multiplier"),
                ),
                "cost_rate_bps_override": profile_score_overrides.get(
                    "cost_rate_bps_override",
                    score_research.get("cost_rate_bps_override"),
                ),
                "effective_cost_rate_bps": _effective_cost_rate_bps(shortlist_input),
            },
            "strict_selection_enabled": profile_strict_selection,
            "shortlist_selection_mode": shortlist_selection_mode,
            "portfolio_weights_enabled": shortlist_weights_enabled,
            "registered_routing_requested": routing_requested,
            "generic_fallback_proxy_rejected_count": proxy_rejected_count,
            "missing_validation_rejected_count": missing_validation_rejected_count,
            "strict_thresholds": {
                key: shortlist_score_params.get(key)
                for key in ("dsr_gate_floor", "spa_gate_ceiling", "pbo_gate_ceiling")
            }
            if shortlist_score_params
            else {},
            "dedup_caps": dict(profile_caps),
            "shortlist_input_count": len(shortlist_input),
            "shortlist_gated_count": len(gated_shortlist_input),
            "shortlist_rejected_count": len(shortlist_input) - len(gated_shortlist_input),
        }
    shortlisted = select_diversified_shortlist(
        shortlist_input,
        mode=shortlist_selection_mode,
        max_total=int(shortlist_config["max_total"]),
        max_per_family=int(shortlist_config["max_per_family"]),
        max_per_timeframe=int(shortlist_config["max_per_timeframe"]),
        single_min_score=shortlist_config.get("single_min_score"),
        drop_single_without_metrics=bool(shortlist_config["drop_single_without_metrics"]),
        single_min_return=shortlist_config.get("single_min_return"),
        single_min_sharpe=shortlist_config.get("single_min_sharpe"),
        single_min_trades=int(shortlist_config["single_min_trades"]),
        allow_multi_asset=bool(shortlist_config["allow_multi_asset"]),
        max_per_lineage=int(shortlist_config["max_per_lineage"]),
        include_weights=shortlist_weights_enabled,
        weight_temperature=float(shortlist_config["weight_temperature"]),
        max_weight=float(shortlist_config["max_weight"]),
        robust_score_params=shortlist_score_params,
    )

    risk_off_mode = {
        "mode": "risk_off_cash",
        "label": "no_position",
        "allocation": {"cash": 1.0},
        "cash_weight": 1.0,
        "reason": (
            "Research outputs keep cash / no-position as a first-class hostile-regime fallback "
            "instead of forcing active risk."
        ),
    }
    report["risk_off_mode"] = risk_off_mode
    effective_split = dict(report.get("split") or report.get("effective_split") or {})
    if (
        effective_split.get("use_lockbox_split") is not True
        or int(effective_split.get("purge_embargo_bars", 0) or 0) < 1
    ):
        raise RuntimeError(
            "strict research run did not resolve the required lockbox/purge contract"
        )
    for row in report.get("candidates") or []:
        if row.get("error"):
            continue
        row_split = dict(
            row.get("effective_split")
            or dict(row.get("metadata") or {}).get("effective_split")
            or {}
        )
        if (
            row_split.get("use_lockbox_split") is not True
            or int(row_split.get("purge_embargo_bars", 0) or 0) < 1
        ):
            raise RuntimeError(
                "candidate did not resolve the required lockbox/purge contract: "
                f"{row.get('candidate_id') or row.get('name')}"
            )
    selection_contract = {
        "ranking_split": "validation",
        "reported_oos_role": "locked_report_only",
        "use_lockbox_split": True,
        "purge_embargo_bars": int(effective_split["purge_embargo_bars"]),
        "dsr_floor": float(shortlist_score_params["dsr_gate_floor"]),
        "spa_ceiling": float(shortlist_score_params["spa_gate_ceiling"]),
        "pbo_ceiling": float(shortlist_score_params["pbo_gate_ceiling"]),
        "promotion_eligible": False,
        "statistical_limitations": [
            "Per-row approx_pbo is not family-wide CSCV/PBO.",
            "Per-row SPA-like p-values are not whole-universe Hansen SPA.",
            "A complete synchronized candidate-by-time matrix and trial ledger remain required.",
            "The one-bar boundary purge is not sufficient evidence for candidates with unknown multi-bar label or holding horizons.",
        ],
    }
    report["selection_contract"] = selection_contract
    report["promotion_eligible"] = False

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    output_path = output_dir / f"candidate_research_{stamp}.json"
    latest_path = output_dir / "candidate_research_latest.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output_dir / f"candidate_research_{stamp}.csv"
    csv_latest = output_dir / "candidate_research_latest.csv"
    _write_summary_csv(csv_path, list(report.get("candidates") or []))
    _write_summary_csv(csv_latest, list(report.get("candidates") or []))

    # Team-report compatibility with existing shortlist pipeline.
    team_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_version": "v2",
        "base_timeframe": report.get("base_timeframe"),
        "strategy_timeframes": report.get("strategy_timeframes"),
        "split": report.get("split"),
        "requested_split": report.get("requested_split") or {},
        "effective_split": report.get("effective_split") or {},
        "split_mode": report.get("split_mode")
        or ("exact" if exact_split is not None else "default"),
        "source_report": str(output_path),
        # This row-level diagnostic cannot enter promotion/portfolio consumers.
        "selected_team": [],
        "diagnostic_shortlist": shortlisted,
        "risk_off_mode": risk_off_mode,
        "candidates": report.get("candidates") or [],
        "stage1": report.get("stage1") or {},
        "scoring_config": report.get("scoring_config") or {},
        "shortlist_config": {
            **shortlist_config,
            "robust_score_params": shortlist_score_params or {},
        },
        "selection_contract": selection_contract,
        "data_sources": report.get("data_sources") or {},
        "research_profile_audit": report.get("research_profile_audit") or {},
    }
    if coverage_summary is not None:
        team_report["coverage_rebuild"] = coverage_summary
    team_report["progress_artifacts"] = progress_writer.artifact_paths
    team_report_path = output_dir / f"strategy_factory_report_{stamp}.json"
    team_report_latest = output_dir / "strategy_factory_report_latest.json"
    team_report_path.write_text(json.dumps(team_report, indent=2), encoding="utf-8")
    team_report_latest.write_text(json.dumps(team_report, indent=2), encoding="utf-8")

    shortlist_md = output_dir / f"strategy_factory_shortlist_{stamp}.md"
    _render_shortlist_markdown(
        report_path=output_path,
        shortlist=shortlisted,
        output_path=shortlist_md,
    )
    progress_writer.complete(
        final_artifacts={
            "candidate_report": str(output_path),
            "candidate_report_latest": str(latest_path),
            "team_report": str(team_report_path),
            "team_report_latest": str(team_report_latest),
            "summary_csv": str(csv_path),
            "summary_csv_latest": str(csv_latest),
            "shortlist_markdown": str(shortlist_md),
        },
        report=report,
        shortlisted=shortlisted,
    )

    print(f"[RESEARCH] candidates_in={len(candidates)}")
    print(f"[RESEARCH] candidates_stage2={len(list(report.get('candidates') or []))}")
    print(f"[RESEARCH] shortlisted={len(shortlisted)}")
    if coverage_summary and coverage_summary.get("actual_max_timestamp"):
        print(f"[RESEARCH] actual_max_timestamp={coverage_summary['actual_max_timestamp']}")
    print(f"Saved progress JSON latest: {progress_writer.json_latest}")
    print(f"Saved progress markdown latest: {progress_writer.md_latest}")
    print(f"Saved progress log latest: {progress_writer.log_latest}")
    print(f"Saved: {output_path}")
    print(f"Saved latest: {latest_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved team report: {team_report_path}")
    print(f"Saved shortlist markdown: {shortlist_md}")
    if args.dry_run:
        print("[RESEARCH] dry-run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
