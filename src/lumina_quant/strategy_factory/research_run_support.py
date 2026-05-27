"""Shared low-risk support helpers for candidate-research entrypoints.

This module is intentionally limited to pure/lightweight contract helpers so
entrypoint modules can depend less on the monolithic ``research_runner``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.symbols import (
    CANONICAL_STRATEGY_TIMEFRAMES,
    canonicalize_symbol_list,
    normalize_strategy_timeframes,
)
from lumina_quant.strategy_factory.runtime_settings import (
    current_research_market_data_settings,
)

DEFAULT_RESEARCH_SCORING_CONFIG: dict[str, Any] = {
    "stage1_prefilter_weights": {
        "sharpe_weight": 2.0,
        "return_weight": 20.0,
        "pbo_penalty": 2.0,
    },
    "candidate_rank_score_weights": {
        "sharpe_weight": 2.8,
        "deflated_sharpe_weight": 1.4,
        "pbo_penalty": 2.0,
        "return_weight": 35.0,
        "turnover_penalty": 2.5,
        "drawdown_penalty": 3.0,
        "turnover_threshold": 2.5,
        "instability_sharpe_penalty": 0.75,
        "instability_return_penalty": 35.0,
        "instability_turnover_penalty": 1.0,
        "inactive_fold_penalty": 0.75,
        "failed_fold_penalty": 1.5,
        "low_active_fold_penalty": 2.0,
        "active_fold_ratio_floor": 0.75,
        "no_trade_train_penalty": 1_000_000.0,
    },
    "hurdle_score_weights": {
        "sharpe_weight": 2.4,
        "return_weight": 35.0,
        "deflated_sharpe_weight": 1.2,
        "pbo_penalty": 2.0,
        "turnover_penalty": 4.0,
        "drawdown_penalty": 5.0,
        "spa_pvalue_penalty": 1.0,
    },
    "reject_thresholds": {
        "in_sample_sharpe_min": -0.1,
        "oos_sharpe_min": 0.35,
        "max_pbo": 0.45,
        "max_turnover": 2.5,
        "max_drawdown": 0.45,
        "min_trade_count": 5.0,
    },
    "cost_stress_thresholds": {
        "x2_sharpe_min": 0.0,
        "x3_sharpe_min": -0.25,
    },
    "keep_ratio_bounds": {
        "min": 0.05,
        "max": 1.0,
    },
    "score_fallbacks": {
        "stage1_error_score": -1_000_000.0,
        "failed_candidate_selection_score": -1_000_000.0,
        "sort_missing_selection_score": -1_000_000.0,
    },
}


def _resolve_score_config(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    resolved = deepcopy(DEFAULT_RESEARCH_SCORING_CONFIG)
    if not isinstance(overrides, Mapping):
        return resolved
    for key, default_value in resolved.items():
        override_value = overrides.get(key)
        if isinstance(default_value, dict) and isinstance(override_value, Mapping):
            for sub_key in default_value:
                if sub_key in override_value:
                    default_value[sub_key] = override_value[sub_key]
        elif override_value is not None and not isinstance(default_value, dict):
            resolved[key] = override_value
    return resolved


def _resolve_feature_points_path() -> Path:
    candidates: list[Path] = []
    defaults = current_research_market_data_settings()

    for raw in (
        os.getenv("LQ_MARKET_PARQUET_PATH", ""),
        defaults["parquet_root"],
        "data/market_parquet",
    ):
        token = str(raw or "").strip()
        if not token:
            continue
        path = Path(token).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        candidates.append(path / "feature_points")

    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        candidates.append(parent / "data" / "market_parquet" / "feature_points")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return (
        candidates[0].resolve()
        if candidates
        else (Path.cwd() / "data" / "market_parquet" / "feature_points").resolve()
    )


def _candidate_identity(candidate: dict[str, Any]) -> str:
    payload = {
        "name": str(candidate.get("name", "")),
        "strategy_class": str(candidate.get("strategy_class", "")),
        "strategy_timeframe": str(
            candidate.get("strategy_timeframe") or candidate.get("timeframe") or ""
        ),
        "symbols": list(candidate.get("symbols") or []),
        "params": dict(candidate.get("params") or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _family_from_strategy(strategy_class: str) -> str:
    token = str(strategy_class).strip().lower()
    if "abnormalreturncontinuation" in token or "abnormal_return_continuation" in token:
        return "event_alpha"
    if "lastdayliquidityregime" in token or "last_day_liquidity_regime" in token:
        return "cross_sectional"
    if "composite" in token or "trend" in token:
        return "trend"
    if "vwap" in token or "reversion" in token:
        return "mean_reversion"
    if "leadlag" in token:
        return "intraday_alpha"
    if "pair" in token:
        return "market_neutral"
    if "perp" in token or "carry" in token:
        return "carry"
    if "micro" in token:
        return "micro"
    return "other"


def adapt_legacy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Adapt legacy candidate fields to v2 contract without data loss."""
    row = dict(candidate)

    if not row.get("strategy_timeframe") and row.get("timeframe"):
        row["strategy_timeframe"] = str(row.get("timeframe"))
    if not row.get("timeframe") and row.get("strategy_timeframe"):
        row["timeframe"] = str(row.get("strategy_timeframe"))

    if not row.get("strategy_class") and row.get("strategy"):
        row["strategy_class"] = str(row.get("strategy"))
    if not row.get("strategy") and row.get("strategy_class"):
        row["strategy"] = str(row.get("strategy_class"))

    row["strategy_timeframe"] = str(row.get("strategy_timeframe") or "1m").strip().lower()
    row["timeframe"] = str(row.get("timeframe") or row["strategy_timeframe"]).strip().lower()

    symbols = canonicalize_symbol_list(list(row.get("symbols") or []))
    row["symbols"] = symbols

    if not row.get("candidate_id"):
        row["candidate_id"] = _candidate_identity(row)

    row["params"] = dict(row.get("params") or {})
    row["name"] = str(row.get("name") or row.get("candidate_id"))
    row["strategy_class"] = str(row.get("strategy_class") or row.get("strategy") or "")
    row["strategy"] = str(row.get("strategy") or row.get("strategy_class") or "")
    row["family"] = str(row.get("family") or _family_from_strategy(row["strategy_class"]))
    return row


@dataclass(frozen=True, slots=True)
class _ResearchRunScoringConfig:
    resolved_scoring_config: dict[str, Any]
    keep_ratio_applied: float
    stage1_weights: dict[str, Any]
    stage1_error_score: float
    failed_candidate_selection_score: float
    sort_missing_selection_score: float


def _normalize_candidate_research_base_timeframe(base_timeframe: str) -> str:
    base_tf = str(base_timeframe).strip().lower() or "1s"
    return "1s" if base_tf != "1s" else base_tf


def _resolve_research_run_scoring_config(
    *,
    score_config: Mapping[str, Any] | None,
    stage1_keep_ratio: float,
) -> _ResearchRunScoringConfig:
    resolved_scoring_config = _resolve_score_config(score_config)
    keep_ratio_cfg = dict(resolved_scoring_config["keep_ratio_bounds"])
    score_fallbacks = dict(resolved_scoring_config["score_fallbacks"])
    return _ResearchRunScoringConfig(
        resolved_scoring_config=resolved_scoring_config,
        keep_ratio_applied=max(
            float(keep_ratio_cfg["min"]),
            min(float(keep_ratio_cfg["max"]), float(stage1_keep_ratio)),
        ),
        stage1_weights=dict(resolved_scoring_config["stage1_prefilter_weights"]),
        stage1_error_score=float(score_fallbacks["stage1_error_score"]),
        failed_candidate_selection_score=float(score_fallbacks["failed_candidate_selection_score"]),
        sort_missing_selection_score=float(score_fallbacks["sort_missing_selection_score"]),
    )


def _adapt_candidate_inputs(
    candidates: Iterable[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    adapted = [adapt_legacy_candidate(item) for item in candidates]
    limit = int(max_candidates)
    if limit > 0 and len(adapted) > limit:
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        family_order: list[str] = []
        for row in adapted:
            family = str(row.get("family") or "other")
            if family not in by_family:
                family_order.append(family)
            by_family[family].append(row)

        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        while len(selected) < limit and family_order:
            progressed = False
            next_order: list[str] = []
            for family in family_order:
                queue = by_family.get(family) or []
                while queue:
                    row = queue.pop(0)
                    candidate_id = str(row.get("candidate_id") or "")
                    if candidate_id in seen_ids:
                        continue
                    selected.append(row)
                    seen_ids.add(candidate_id)
                    progressed = True
                    break
                if queue:
                    next_order.append(family)
                if len(selected) >= limit:
                    break
            if not progressed:
                break
            family_order = next_order

        if len(selected) < limit:
            for row in adapted:
                candidate_id = str(row.get("candidate_id") or "")
                if candidate_id in seen_ids:
                    continue
                selected.append(row)
                if len(selected) >= limit:
                    break

        adapted = selected
    return adapted


def _resolve_research_run_timeframes_and_universe(
    *,
    adapted: Sequence[dict[str, Any]],
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    discovered_timeframes = sorted(
        {
            str(row.get("strategy_timeframe") or row.get("timeframe") or "1m").strip().lower()
            for row in adapted
        }
    )
    normalized_timeframes = normalize_strategy_timeframes(
        strategy_timeframes or discovered_timeframes or CANONICAL_STRATEGY_TIMEFRAMES,
        required=CANONICAL_STRATEGY_TIMEFRAMES,
        strict_subset=True,
    )
    candidate_symbols = canonicalize_symbol_list(
        itertools.chain.from_iterable(list(row.get("symbols") or []) for row in adapted)
    )
    if symbol_universe:
        universe = canonicalize_symbol_list(symbol_universe)
    elif candidate_symbols:
        universe = candidate_symbols
    else:
        universe = canonicalize_symbol_list(current_research_market_data_settings()["symbols"])
    return normalized_timeframes, universe


def _coerce_utc_datetime(value: Any, *, end_of_day: bool = False) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, np.datetime64):
        epoch_ms = int(value.astype("datetime64[ms]").astype(np.int64))
        dt = datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)
    elif isinstance(value, (int, float)):
        numeric = int(value)
        if abs(numeric) < 100_000_000_000:
            numeric *= 1000
        dt = datetime.fromtimestamp(numeric / 1000.0, tz=UTC)
    else:
        text = str(value).strip()
        if not text:
            return None
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            dt = datetime.fromisoformat(text)
            if end_of_day:
                dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _datetime_to_iso_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _resolve_split_stage_end(
    split: Mapping[str, Any] | None,
    *,
    stage: str,
) -> datetime | None:
    if not isinstance(split, Mapping):
        return None

    inclusive_keys: tuple[str, ...]
    exclusive_keys: tuple[str, ...]
    if stage == "train":
        inclusive_keys = ("train_end",)
        exclusive_keys = ("train_end_exclusive",)
    elif stage == "val":
        inclusive_keys = ("val_end", "validation_end")
        exclusive_keys = ("val_end_exclusive", "validation_end_exclusive")
    elif stage == "oos":
        inclusive_keys = ("oos_end", "test_end")
        exclusive_keys = (
            "actual_oos_end_exclusive",
            "oos_end_exclusive",
            "requested_oos_end_exclusive",
            "test_end_exclusive",
        )
    else:
        msg = f"unsupported split stage: {stage}"
        raise ValueError(msg)

    for key in inclusive_keys:
        value = _coerce_utc_datetime(split.get(key), end_of_day=True)
        if value is not None:
            return value

    for key in exclusive_keys:
        value = _coerce_utc_datetime(split.get(key))
        if value is not None:
            return value - timedelta(milliseconds=1)
    return None


def _split_window_bounds(
    split: Mapping[str, Any] | None,
) -> tuple[datetime | None, datetime | None]:
    if not isinstance(split, Mapping):
        return None, None
    starts = [
        _coerce_utc_datetime(split.get("train_start")),
        _coerce_utc_datetime(split.get("val_start")),
        _coerce_utc_datetime(split.get("oos_start") or split.get("test_start")),
    ]
    ends = [
        _resolve_split_stage_end(split, stage="train"),
        _resolve_split_stage_end(split, stage="val"),
        _resolve_split_stage_end(split, stage="oos"),
    ]
    valid_starts = [item for item in starts if item is not None]
    valid_ends = [item for item in ends if item is not None]
    return (
        min(valid_starts) if valid_starts else None,
        max(valid_ends) if valid_ends else None,
    )


def _build_default_split(strategy_timeframe: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    train_start = now - timedelta(days=360)
    train_end = now - timedelta(days=150)
    val_start = train_end + timedelta(days=1)
    val_end = now - timedelta(days=60)
    oos_start = val_end + timedelta(days=1)
    oos_end = now
    return {
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "val_start": val_start.isoformat(),
        "val_end": val_end.isoformat(),
        "oos_start": oos_start.isoformat(),
        "oos_end": oos_end.isoformat(),
        "strategy_timeframe": str(strategy_timeframe),
    }


def _resolve_split_config(
    split: Mapping[str, Any] | None,
    *,
    strategy_timeframe: str,
) -> dict[str, Any]:
    resolved = (
        dict(split) if isinstance(split, Mapping) else _build_default_split(strategy_timeframe)
    )
    train_start = _coerce_utc_datetime(resolved.get("train_start"))
    train_end = _resolve_split_stage_end(resolved, stage="train")
    val_start = _coerce_utc_datetime(resolved.get("val_start"))
    val_end = _resolve_split_stage_end(resolved, stage="val")
    oos_start = _coerce_utc_datetime(resolved.get("oos_start") or resolved.get("test_start"))
    oos_end = _resolve_split_stage_end(resolved, stage="oos")
    return {
        **resolved,
        "train_start": _datetime_to_iso_z(train_start),
        "train_end": _datetime_to_iso_z(train_end),
        "val_start": _datetime_to_iso_z(val_start),
        "val_end": _datetime_to_iso_z(val_end),
        "oos_start": _datetime_to_iso_z(oos_start),
        "oos_end": _datetime_to_iso_z(oos_end),
        "strategy_timeframe": str(
            resolved.get("strategy_timeframe") or resolved.get("timeframe") or strategy_timeframe
        ),
        "mode": str(
            resolved.get("mode")
            or ("exact_dates" if isinstance(split, Mapping) else "rolling_default")
        ),
    }


def _empty_candidate_research_report(
    *,
    base_timeframe: str,
    strategy_timeframes: Sequence[str] | None,
    symbol_universe: Sequence[str] | None,
    stage1_keep_ratio: float,
    scoring: _ResearchRunScoringConfig,
    split: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized_timeframes = normalize_strategy_timeframes(
        strategy_timeframes or CANONICAL_STRATEGY_TIMEFRAMES,
        required=CANONICAL_STRATEGY_TIMEFRAMES,
        strict_subset=True,
    )
    empty_split = _resolve_split_config(
        split,
        strategy_timeframe=normalized_timeframes[0] if normalized_timeframes else "1m",
    )
    return {
        "schema_version": "v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_timeframe": base_timeframe,
        "strategy_timeframes": normalized_timeframes,
        "symbol_universe": canonicalize_symbol_list(
            symbol_universe or current_research_market_data_settings()["symbols"]
        ),
        "split": empty_split,
        "candidates": [],
        "stage1": {
            "input_count": 0,
            "selected_count": 0,
            "keep_ratio": float(stage1_keep_ratio),
            "keep_ratio_applied": float(scoring.keep_ratio_applied),
        },
        "scoring_config": scoring.resolved_scoring_config,
    }


__all__ = [
    "DEFAULT_RESEARCH_SCORING_CONFIG",
    "_ResearchRunScoringConfig",
    "_adapt_candidate_inputs",
    "_build_default_split",
    "_coerce_utc_datetime",
    "_datetime_to_iso_z",
    "_empty_candidate_research_report",
    "_family_from_strategy",
    "_normalize_candidate_research_base_timeframe",
    "_resolve_research_run_scoring_config",
    "_resolve_research_run_timeframes_and_universe",
    "_resolve_score_config",
    "_resolve_split_config",
    "_split_window_bounds",
    "adapt_legacy_candidate",
]
