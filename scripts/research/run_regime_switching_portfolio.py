"""Regime-aware portfolio switcher over saved portfolio return streams.

This script treats whole saved portfolios as candidate sleeves, then allocates
between them using only prior information:

- trailing performance over a rolling lookback
- ex-ante regime features from the research universe
- hysteresis to reduce churn
- sleeve-level turnover costs to account for rebalancing

It is intentionally lightweight and single-lane so it can be run under the
shared follow-up memory contract without breaching the 8 GiB session cap.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from lumina_quant.eval.exact_window_runtime import RSSLimitExceeded
from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    PORTFOLIO_CURRENT_OPTIMIZATION,
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
    resolve_current_optimization_path,
    resolve_followup_artifact_path,
    split_for_day_key,
    split_windows,
)

DEFAULT_OUTPUT_DIR = FOLLOWUP_ROOT / "portfolio_regime_switch_current"
DEFAULT_COMPARISON_INPUT = FOLLOWUP_ROOT / "portfolio_max_performance_decision_latest.json"
DEFAULT_CONTINUITY_REPORT = FOLLOWUP_ROOT / "portfolio_continuity_validation_latest.json"
DEFAULT_OVERLAY_PORTFOLIO = (
    FOLLOWUP_ROOT / "portfolio_overlay_current" / "causal_overlay_portfolio_latest.json"
)
DEFAULT_DYNAMIC_PORTFOLIO = (
    FOLLOWUP_ROOT / "portfolio_dynamic_online_current" / "causal_dynamic_portfolio_latest.json"
)
_AUTORESEARCH_GLOB = (
    ".omx/worktrees/autoresearch-*/var/reports/exact_window_backtests/"
    "followup_status/autoresearch_candidate_portfolio_opt/portfolio_optimization_latest.json"
)

_helper_spec = importlib.util.spec_from_file_location(
    "run_causal_dynamic_portfolio",
    Path(__file__).resolve().parent / "run_causal_dynamic_portfolio.py",
)
if _helper_spec is None or _helper_spec.loader is None:
    raise RuntimeError("Failed to load run_causal_dynamic_portfolio helpers")
_helper = importlib.util.module_from_spec(_helper_spec)
sys.modules[_helper_spec.name] = _helper
_helper_spec.loader.exec_module(_helper)

_overlay_spec = importlib.util.spec_from_file_location(
    "run_causal_overlay_portfolio",
    Path(__file__).resolve().parent / "run_causal_overlay_portfolio.py",
)
if _overlay_spec is None or _overlay_spec.loader is None:
    raise RuntimeError("Failed to load run_causal_overlay_portfolio helpers")
_overlay = importlib.util.module_from_spec(_overlay_spec)
sys.modules[_overlay_spec.name] = _overlay
_overlay_spec.loader.exec_module(_overlay)

_REBUILT_STREAM_CACHE: dict[str, dict[str, list[dict[str, Any]]]] = {}
_CANDIDATE_ROWS_CACHE: dict[str, list[dict[str, Any]]] = {}
_REGIME_FEATURE_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_REBUILT_STREAM_CACHE_DIR = DEFAULT_OUTPUT_DIR / "rebuild_stream_cache"


def _safe_float(value: Any, default: float = 0.0) -> float:
    return _helper._safe_float(value, default)


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value)]
    return value


def _discover_latest_autoresearch_candidate() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    matches = sorted(
        root.glob(_AUTORESEARCH_GLOB),
        key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.as_posix()),
        reverse=True,
    )
    return matches[0].resolve() if matches else None


def _cached_candidate_rows(input_path: Path) -> list[dict[str, Any]]:
    key = str(input_path.resolve())
    cached = _CANDIDATE_ROWS_CACHE.get(key)
    if cached is not None:
        return cached
    rows = _helper._load_candidates(input_path.resolve())
    _CANDIDATE_ROWS_CACHE[key] = rows
    return rows


def _cached_regime_features(
    input_path: Path, rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    key = str(input_path.resolve())
    cached = _REGIME_FEATURE_CACHE.get(key)
    if cached is not None:
        return cached
    features = _helper._load_regime_features(rows)
    _REGIME_FEATURE_CACHE[key] = features
    return features


def _portfolio_name(path: Path, payload: dict[str, Any]) -> str:
    explicit = str(payload.get("portfolio_name") or payload.get("name") or "").strip()
    if explicit:
        return explicit
    selection_basis = str(payload.get("selection_basis") or "").strip()
    if selection_basis == "rolled_over_from_promoted_challenger":
        return "current_one_shot_incumbent"
    if selection_basis == "bounded_archived_exact_window_pair_rebalance":
        return "autoresearch_pair_55_45"
    return path.parent.name or path.stem


def _rebuild_stream_cache_key(source_path: Path, payload: dict[str, Any]) -> str:
    source = source_path.resolve()
    digest = hashlib.sha256()
    digest.update(str(source).encode("utf-8"))
    if source.exists():
        stat = source.stat()
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(
        json.dumps(
            {
                "artifact_kind": payload.get("artifact_kind"),
                "input_path": payload.get("input_path"),
                "backbone_path": payload.get("backbone_path"),
                "best_params": payload.get("best_params"),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _rebuild_stream_cache_path(source_path: Path, payload: dict[str, Any]) -> Path:
    return _REBUILT_STREAM_CACHE_DIR / f"{_rebuild_stream_cache_key(source_path, payload)}.json"


def _load_rebuilt_stream_cache(
    source_path: Path, payload: dict[str, Any]
) -> dict[str, list[dict[str, Any]]] | None:
    cache_path = _rebuild_stream_cache_path(source_path, payload)
    if not cache_path.exists():
        return None
    cached = _load_json(cache_path)
    streams = cached.get("portfolio_return_streams")
    if not isinstance(streams, dict):
        return None
    return {
        split: [dict(point) for point in list(streams.get(split) or [])]
        for split in ("train", "val", "oos")
    }


def _write_rebuilt_stream_cache(
    source_path: Path,
    payload: dict[str, Any],
    streams: dict[str, list[dict[str, Any]]],
) -> None:
    cache_path = _rebuild_stream_cache_path(source_path, payload)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "artifact_kind": "regime_switch_rebuilt_stream_cache",
                "source_path": str(source_path.resolve()),
                "generated_at": datetime.now(UTC).isoformat(),
                "portfolio_return_streams": _json_ready(streams),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _supports_allocation_reconstruction(payload: dict[str, Any]) -> bool:
    allocations = list(payload.get("allocations") or [])
    input_path = payload.get("input_path")
    if not allocations or not input_path:
        return False
    try:
        resolved_input = resolve_followup_artifact_path(Path(str(input_path))).resolve()
        rows = _cached_candidate_rows(resolved_input)
    except Exception:
        return False

    referenced_ids: set[str] = set()
    for allocation in allocations:
        weights = dict(allocation.get("weights") or {})
        referenced_ids.update(str(cid) for cid in weights if str(cid))
    if not referenced_ids:
        referenced_ids.update(
            str(row.get("candidate_id") or row.get("name") or "")
            for row in list(payload.get("final_allocation") or [])
            if str(row.get("candidate_id") or row.get("name") or "")
        )
    if not referenced_ids:
        return False

    available_stream_ids = {
        str(row.get("candidate_id") or row.get("name") or "")
        for row in rows
        if isinstance(row.get("return_streams"), dict) and row.get("return_streams")
    }
    return referenced_ids <= available_stream_ids


def _stream_mode(path: Path, payload: dict[str, Any]) -> str:
    if isinstance(payload.get("portfolio_return_streams"), dict) and payload.get(
        "portfolio_return_streams"
    ):
        return "saved_streams"
    if payload.get("daily_returns") and payload.get("dates"):
        return "saved_daily_returns"
    if _load_rebuilt_stream_cache(path, payload) is not None:
        return "cached_rebuild_streams"
    if _supports_allocation_reconstruction(payload):
        return "allocation_reconstructible"
    return "rebuild_required"


def _extract_portfolio_components(payload: dict[str, Any]) -> list[dict[str, Any]]:
    components = list(payload.get("weights") or [])
    if components:
        rows: list[dict[str, Any]] = []
        for row in components:
            item = dict(row)
            item["candidate_id"] = str(item.get("candidate_id") or item.get("name") or "")
            item["name"] = item.get("name") or item["candidate_id"]
            item["weight"] = _safe_float(item.get("weight"), 0.0)
            item["family"] = str(item.get("family") or "other")
            item["strategy_class"] = str(item.get("strategy_class") or "")
            item["symbols"] = [str(symbol) for symbol in list(item.get("symbols") or [])]
            rows.append(item)
        return rows

    source_components = list(payload.get("source_components") or [])
    rows = []
    for row in source_components:
        item = dict(row)
        item["candidate_id"] = str(item.get("candidate_id") or item.get("name") or "")
        item["name"] = item.get("name") or item["candidate_id"]
        item["weight"] = _safe_float(item.get("weight"), 0.0)
        item["family"] = str(item.get("family") or "other")
        item["strategy_class"] = str(item.get("strategy_class") or "")
        item["symbols"] = [str(symbol) for symbol in list(item.get("symbols") or [])]
        rows.append(item)
    if rows:
        return rows

    final_allocation = list(payload.get("final_allocation") or [])
    if not final_allocation:
        return []

    candidate_lookup: dict[str, dict[str, Any]] = {}
    input_path = payload.get("input_path")
    if input_path:
        try:
            resolved_input = resolve_followup_artifact_path(Path(str(input_path))).resolve()
            candidate_lookup = {
                str(row.get("candidate_id") or row.get("name") or ""): dict(row)
                for row in _cached_candidate_rows(resolved_input)
            }
        except Exception:
            candidate_lookup = {}

    rows = []
    for row in final_allocation:
        item = dict(row)
        cid = str(item.get("candidate_id") or item.get("name") or "")
        source = dict(candidate_lookup.get(cid) or {})
        item["candidate_id"] = cid
        item["name"] = item.get("name") or source.get("name") or cid
        item["weight"] = _safe_float(item.get("weight"), 0.0)
        item["family"] = str(
            item.get("family")
            or source.get("family")
            or _helper._strategy_family(source.get("strategy_class") or item.get("strategy_class"))
            or "other"
        )
        item["strategy_class"] = str(
            item.get("strategy_class") or source.get("strategy_class") or ""
        )
        item["symbols"] = [
            str(symbol) for symbol in list(item.get("symbols") or source.get("symbols") or [])
        ]
        item["timeframe"] = str(
            item.get("timeframe")
            or source.get("strategy_timeframe")
            or source.get("timeframe")
            or ""
        )
        rows.append(item)
    return rows


def _streams_from_daily_returns(
    dates: list[str],
    daily_returns: list[float],
) -> dict[str, list[dict[str, Any]]]:
    streams: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
    for raw_date, raw_ret in zip(dates, daily_returns, strict=True):
        day_key = str(raw_date)
        split = split_for_day_key(day_key)
        streams[split].append({"t": f"{day_key}T00:00:00Z", "v": _safe_float(raw_ret, 0.0)})
    return streams


def _reconstruct_streams_from_saved_allocations(
    payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]] | None:
    allocations = list(payload.get("allocations") or [])
    input_path = payload.get("input_path")
    if not allocations or not input_path:
        return None
    try:
        resolved_input = resolve_followup_artifact_path(Path(str(input_path))).resolve()
        rows = _cached_candidate_rows(resolved_input)
    except Exception:
        return None

    series = {
        str(row.get("candidate_id") or row.get("name") or ""): {
            **_helper._daily_compound_stream(
                list((row.get("return_streams") or {}).get("train") or [])
            ),
            **_helper._daily_compound_stream(
                list((row.get("return_streams") or {}).get("val") or [])
            ),
            **_helper._daily_compound_stream(
                list((row.get("return_streams") or {}).get("oos") or [])
            ),
        }
        for row in rows
    }
    dates: list[str] = []
    daily_returns: list[float] = []
    for allocation in allocations:
        day_key = str(allocation.get("date") or "").strip()
        if not day_key:
            continue
        dates.append(day_key)
        weights = dict(allocation.get("weights") or {})
        daily_returns.append(
            sum(
                _safe_float(weight, 0.0) * _safe_float(series.get(str(cid), {}).get(day_key), 0.0)
                for cid, weight in weights.items()
            )
        )
    if not dates:
        return None
    return _streams_from_daily_returns(dates, daily_returns)


def _rebuild_missing_portfolio_streams(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    resolved_source = source_path.resolve() if source_path is not None else None
    if resolved_source is not None:
        persisted = _load_rebuilt_stream_cache(resolved_source, payload)
        if persisted is not None:
            return persisted

    cache_key = json.dumps(
        {
            "source_path": str(resolved_source) if resolved_source is not None else None,
            "artifact_kind": payload.get("artifact_kind"),
            "input_path": payload.get("input_path"),
            "backbone_path": payload.get("backbone_path"),
            "best_params": payload.get("best_params"),
        },
        sort_keys=True,
    )
    cached_streams = _REBUILT_STREAM_CACHE.get(cache_key)
    if cached_streams is not None:
        return cached_streams

    artifact_kind = str(payload.get("artifact_kind") or "")
    best_params = dict(payload.get("best_params") or {})
    reconstructed = _reconstruct_streams_from_saved_allocations(payload)
    if reconstructed is not None:
        if resolved_source is not None:
            _write_rebuilt_stream_cache(resolved_source, payload, reconstructed)
        _REBUILT_STREAM_CACHE[cache_key] = reconstructed
        return reconstructed
    if not best_params:
        return None

    if artifact_kind == "causal_dynamic_portfolio":
        input_path = payload.get("input_path")
        if not input_path:
            return None
        resolved_input = resolve_followup_artifact_path(Path(str(input_path))).resolve()
        rows = _cached_candidate_rows(resolved_input)
        regime_features = _cached_regime_features(resolved_input, rows)
        result = _helper.run_causal_dynamic_allocator(
            rows,
            _helper.AllocatorParams(**best_params),
            regime_features=regime_features,
        )
        streams = _streams_from_daily_returns(
            list(result.get("dates") or []),
            list(result.get("daily_returns") or []),
        )
        if resolved_source is not None:
            _write_rebuilt_stream_cache(resolved_source, payload, streams)
        _REBUILT_STREAM_CACHE[cache_key] = streams
        return streams

    if artifact_kind == "causal_overlay_portfolio":
        input_path = payload.get("input_path")
        backbone_path = payload.get("backbone_path")
        if not input_path or not backbone_path:
            return None
        resolved_input = resolve_followup_artifact_path(Path(str(input_path))).resolve()
        rows = _cached_candidate_rows(resolved_input)
        regime_features = _cached_regime_features(resolved_input, rows)
        backbone_weights = _overlay._load_backbone_weights(
            resolve_followup_artifact_path(Path(str(backbone_path))).resolve()
        )
        result = _overlay.run_causal_overlay_allocator(
            rows,
            backbone_weights,
            _overlay.OverlayParams(**best_params),
            regime_features=regime_features,
        )
        streams = _streams_from_daily_returns(
            list(result.get("dates") or []),
            list(result.get("daily_returns") or []),
        )
        if resolved_source is not None:
            _write_rebuilt_stream_cache(resolved_source, payload, streams)
        _REBUILT_STREAM_CACHE[cache_key] = streams
        return streams

    return None


def _extract_portfolio_return_streams(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    portfolio_streams = payload.get("portfolio_return_streams")
    if isinstance(portfolio_streams, dict) and portfolio_streams:
        return {
            split: [dict(point) for point in list(portfolio_streams.get(split) or [])]
            for split in ("train", "val", "oos")
        }

    dates = [str(day) for day in list(payload.get("dates") or [])]
    daily_returns = [_safe_float(value, 0.0) for value in list(payload.get("daily_returns") or [])]
    if dates and daily_returns and len(dates) == len(daily_returns):
        return _streams_from_daily_returns(dates, daily_returns)

    rebuilt = _rebuild_missing_portfolio_streams(payload, source_path=source_path)
    if rebuilt is not None:
        return rebuilt

    raise RuntimeError("portfolio payload missing portfolio_return_streams and daily_returns")


def _build_candidate_metadata(
    *,
    path: Path,
    alias: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(path)
    components = _extract_portfolio_components(payload)
    name = str(alias or _portfolio_name(path, payload))
    metadata = {
        "candidate_id": name,
        "name": name,
        "path": str(path.resolve()),
        "selection_basis": str(payload.get("selection_basis") or ""),
        "artifact_kind": str(payload.get("artifact_kind") or ""),
        "stream_mode": _stream_mode(path, payload),
        "portfolio_metrics": dict(
            payload.get("portfolio_metrics") or payload.get("split_metrics") or {}
        ),
        "components": components,
        "symbols": sorted(
            {str(symbol) for row in components for symbol in list(row.get("symbols") or [])}
        ),
    }
    return metadata, payload


def _build_candidate(
    *,
    path: Path,
    alias: str | None = None,
) -> dict[str, Any]:
    metadata, payload = _build_candidate_metadata(path=path, alias=alias)
    metadata["return_streams"] = _extract_portfolio_return_streams(payload, source_path=path)
    return metadata


def _load_switch_candidates(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_path in paths:
        resolved = resolve_followup_artifact_path(Path(raw_path)).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"candidate portfolio path not found: {raw_path}")
        candidate = _build_candidate(path=resolved)
        cid = str(candidate["candidate_id"])
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        rows.append(candidate)
    if not rows:
        raise RuntimeError("no switch candidates could be loaded")
    return rows


def _load_switch_candidate_metadata(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_path in paths:
        resolved = resolve_followup_artifact_path(Path(raw_path)).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"candidate portfolio path not found: {raw_path}")
        candidate, _payload = _build_candidate_metadata(path=resolved)
        cid = str(candidate["candidate_id"])
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        rows.append(candidate)
    if not rows:
        raise RuntimeError("no switch candidates could be loaded")
    return rows


def _load_runtime_risk_state(
    path: Path | None = None,
) -> dict[str, Any]:
    report_path = resolve_followup_artifact_path(path or DEFAULT_CONTINUITY_REPORT)
    if not report_path.exists():
        return {
            "continuity_report_path": str(report_path),
            "continuity_status": "missing",
            "continuity_failed": False,
            "symbols": set(),
            "timeframes": set(),
            "error": None,
        }
    payload = _load_json(report_path)
    status = str(payload.get("status") or "unknown")
    return {
        "continuity_report_path": str(report_path),
        "continuity_status": status,
        "continuity_failed": status.lower() not in {"completed", "passed", "ok", "success"},
        "symbols": {str(symbol) for symbol in list(payload.get("symbols") or [])},
        "timeframes": {str(timeframe) for timeframe in list(payload.get("timeframes") or [])},
        "error": payload.get("error"),
    }


def _candidate_panel(
    rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    all_days: set[str] = set()
    series: dict[str, dict[str, float]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = str(row["candidate_id"])
        merged: dict[str, float] = {}
        for split in ("train", "val", "oos"):
            merged.update(
                _helper._daily_compound_stream(
                    list((row.get("return_streams") or {}).get(split) or [])
                )
            )
        all_days.update(merged.keys())
        series[cid] = merged
        components = list(row.get("components") or [])
        meta[cid] = {
            "name": row.get("name"),
            "path": row.get("path"),
            "selection_basis": row.get("selection_basis"),
            "stream_mode": row.get("stream_mode"),
            "symbols": list(row.get("symbols") or []),
            "components": components,
            "component_count": len(components),
            "families": sorted(
                {str(component.get("family") or "other") for component in components}
            ),
        }
    ordered_days = sorted(all_days)
    matrix = {
        cid: np.asarray(
            [_safe_float(series[cid].get(day_key), 0.0) for day_key in ordered_days], dtype=float
        )
        for cid in series
    }
    return ordered_days, matrix, meta


def _regime_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_rows: list[dict[str, Any]] = []
    for row in rows:
        feature_rows.append(
            {
                "symbols": list(row.get("symbols") or []),
                "return_streams": dict(row.get("return_streams") or {}),
            }
        )
    return feature_rows


def _component_regime_multiplier(
    component: dict[str, Any],
    regime_row: dict[str, Any],
    *,
    previous_active: bool,
    strength: float,
) -> float:
    strategy_class = str(component.get("strategy_class") or "")
    if strategy_class == "PairSpreadZScoreStrategy":
        btc_up = bool(regime_row.get("btc_above_ma192", False))
        breadth_up = bool(regime_row.get("breadth_ma96_ge_60", False))
        vol_moderate = bool(regime_row.get("basket_vol_ratio_moderate", False))
        trend_hot = btc_up and breadth_up
        multiplier = 1.0
        multiplier *= (
            (1.0 + (0.20 * strength)) if vol_moderate else max(0.55, 1.0 - (0.20 * strength))
        )
        if trend_hot:
            multiplier *= max(0.70, 1.0 - (0.15 * strength))
        elif (not btc_up) and (not breadth_up):
            multiplier *= 1.0 + (0.10 * strength)
        if previous_active:
            multiplier *= 1.0 + (0.05 * strength)
        return max(0.0, float(multiplier))

    fallback_meta = {
        "strategy_class": strategy_class,
        "metadata": dict(component.get("metadata") or {}),
    }
    return float(
        _helper._regime_multiplier(
            fallback_meta,
            regime_row,
            previous_active=previous_active,
            strength=strength,
        )
    )


def _portfolio_regime_score(
    candidate_meta: dict[str, Any],
    regime_row: dict[str, Any] | None,
    *,
    previous_active: bool,
    strength: float,
) -> tuple[float, list[dict[str, Any]]]:
    if not regime_row or strength <= 0.0:
        return 1.0, []
    components = list(candidate_meta.get("components") or [])
    if not components:
        return 1.0, []
    total_weight = sum(
        max(0.0, _safe_float(component.get("weight"), 0.0)) for component in components
    )
    if total_weight <= 1e-12:
        total_weight = float(len(components))
    weighted = 0.0
    diagnostics: list[dict[str, Any]] = []
    for component in components:
        weight = max(0.0, _safe_float(component.get("weight"), 0.0))
        multiplier = _component_regime_multiplier(
            component,
            regime_row,
            previous_active=previous_active,
            strength=strength,
        )
        weighted += (weight or 1.0) * multiplier
        diagnostics.append(
            {
                "candidate_id": component.get("candidate_id"),
                "name": component.get("name"),
                "weight": weight,
                "strategy_class": component.get("strategy_class"),
                "family": component.get("family"),
                "regime_multiplier": float(multiplier),
            }
        )
    return max(0.0, float(weighted / total_weight)), diagnostics


def _positive_corr_penalty(history: dict[str, np.ndarray], ids: list[str], cid: str) -> float:
    arr_i = np.asarray(history.get(cid, []), dtype=float)
    penalties: list[float] = []
    for other in ids:
        if other == cid:
            continue
        arr_j = np.asarray(history.get(other, []), dtype=float)
        n = min(arr_i.size, arr_j.size)
        if n < 3:
            continue
        x = arr_i[-n:]
        y = arr_j[-n:]
        sx = float(np.std(x, ddof=1))
        sy = float(np.std(y, ddof=1))
        if sx <= 1e-12 or sy <= 1e-12:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if math.isfinite(corr):
            penalties.append(max(0.0, corr))
    return float(np.mean(penalties)) if penalties else 0.0


def _blend_scores(
    history: dict[str, np.ndarray],
    raw_scores: dict[str, float],
    *,
    target_total_exposure: float,
    max_portfolio_weight: float,
    correlation_penalty: float,
) -> dict[str, float]:
    if not raw_scores:
        return {}
    target_total_exposure = max(0.0, min(1.0, float(target_total_exposure)))
    if target_total_exposure <= 1e-12:
        return {}
    ids = list(raw_scores.keys())
    adjusted = {}
    for cid in ids:
        corr_pen = _positive_corr_penalty(history, ids, cid)
        adjusted[cid] = float(raw_scores[cid] / (1.0 + (correlation_penalty * corr_pen)))
    total = sum(max(0.0, score) for score in adjusted.values())
    if total <= 1e-12:
        return {}
    positive_ids = [cid for cid, score in adjusted.items() if max(0.0, score) > 1e-12]
    if not positive_ids:
        return {}
    remaining = target_total_exposure
    unresolved = sorted(positive_ids, key=lambda cid: adjusted[cid], reverse=True)
    out = dict.fromkeys(positive_ids, 0.0)
    while unresolved and remaining > 1e-12:
        unresolved_total = sum(max(0.0, adjusted[cid]) for cid in unresolved)
        if unresolved_total <= 1e-12:
            break
        capped: list[str] = []
        for cid in list(unresolved):
            desired = remaining * (max(0.0, adjusted[cid]) / unresolved_total)
            capacity = max(0.0, max_portfolio_weight - out[cid])
            allowed = min(capacity, desired)
            out[cid] += float(allowed)
            if capacity - allowed <= 1e-12:
                capped.append(cid)
        remaining = max(0.0, target_total_exposure - sum(out.values()))
        unresolved = [
            cid
            for cid in unresolved
            if cid not in capped and (max_portfolio_weight - out[cid]) > 1e-12
        ]
        if not capped:
            break
    if remaining > 1e-12 and unresolved:
        unresolved_total = sum(max(0.0, adjusted[cid]) for cid in unresolved)
        if unresolved_total > 1e-12:
            for cid in unresolved:
                capacity = max(0.0, max_portfolio_weight - out[cid])
                desired = remaining * (max(0.0, adjusted[cid]) / unresolved_total)
                out[cid] += float(min(capacity, desired))
    return {cid: weight for cid, weight in out.items() if weight > 1e-12}


def _aggregate_component_weights(
    portfolio_weights: dict[str, float],
    *,
    meta: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for cid, portfolio_weight in portfolio_weights.items():
        candidate_meta = dict(meta.get(cid) or {})
        components = list(candidate_meta.get("components") or [])
        for component in components:
            key = str(component.get("candidate_id") or component.get("name") or "")
            if not key:
                continue
            contribution = float(portfolio_weight) * _safe_float(component.get("weight"), 0.0)
            entry = aggregated.setdefault(
                key,
                {
                    "candidate_id": key,
                    "name": component.get("name") or key,
                    "strategy_class": component.get("strategy_class"),
                    "family": component.get("family"),
                    "timeframe": component.get("timeframe"),
                    "symbols": list(component.get("symbols") or []),
                    "weight": 0.0,
                    "source_portfolios": [],
                },
            )
            entry["weight"] = _safe_float(entry.get("weight"), 0.0) + contribution
            sources = set(entry.get("source_portfolios") or [])
            sources.add(str(candidate_meta.get("name") or cid))
            entry["source_portfolios"] = sorted(sources)
    return aggregated


def _turnover(prev_weights: dict[str, float], next_weights: dict[str, float]) -> float:
    ids = set(prev_weights) | set(next_weights)
    return float(
        sum(
            abs(_safe_float(prev_weights.get(cid), 0.0) - _safe_float(next_weights.get(cid), 0.0))
            for cid in ids
        )
    )


def _infer_incumbent_id(meta: dict[str, dict[str, Any]]) -> str | None:
    for cid, item in meta.items():
        selection_basis = str(item.get("selection_basis") or "")
        name = str(item.get("name") or cid).lower()
        if selection_basis == "rolled_over_from_promoted_challenger":
            return cid
        if "incumbent" in name or "current_one_shot" in name:
            return cid
    return next(iter(meta), None)


def _apply_incumbent_floor(
    weights: dict[str, float],
    *,
    incumbent_id: str | None,
    floor_weight: float,
) -> dict[str, float]:
    if incumbent_id is None or floor_weight <= 0.0:
        return dict(weights)
    resolved = dict(weights)
    incumbent_weight = _safe_float(resolved.get(incumbent_id), 0.0)
    if incumbent_weight >= floor_weight:
        return resolved
    donor_ids = [
        cid for cid in resolved if cid != incumbent_id and _safe_float(resolved.get(cid), 0.0) > 0.0
    ]
    donor_total = sum(_safe_float(resolved.get(cid), 0.0) for cid in donor_ids)
    needed = min(max(0.0, floor_weight - incumbent_weight), donor_total)
    if needed <= 1e-12:
        return resolved
    if donor_total > 1e-12:
        for cid in donor_ids:
            share = _safe_float(resolved.get(cid), 0.0) / donor_total
            resolved[cid] = max(0.0, _safe_float(resolved.get(cid), 0.0) - (needed * share))
    resolved[incumbent_id] = incumbent_weight + needed
    return {cid: weight for cid, weight in resolved.items() if weight > 1e-12}


def _cap_turnover_transition(
    prev_weights: dict[str, float],
    target_weights: dict[str, float],
    *,
    max_turnover: float,
) -> dict[str, float]:
    if max_turnover <= 0.0:
        return dict(prev_weights)
    realized_turnover = _turnover(prev_weights, target_weights)
    if realized_turnover <= max_turnover + 1e-12:
        return dict(target_weights)
    blend = max_turnover / max(realized_turnover, 1e-12)
    ids = set(prev_weights) | set(target_weights)
    blended = {
        cid: _safe_float(prev_weights.get(cid), 0.0)
        + blend
        * (_safe_float(target_weights.get(cid), 0.0) - _safe_float(prev_weights.get(cid), 0.0))
        for cid in ids
    }
    return {cid: weight for cid, weight in blended.items() if weight > 1e-12}


@dataclass(slots=True)
class SwitchParams:
    lookback_days: int
    rebalance_days: int
    min_trailing_sharpe: float
    min_trailing_return: float
    max_trailing_drawdown: float
    max_portfolio_weight: float
    correlation_penalty: float = 0.0
    regime_strength: float = 1.0
    hysteresis_bonus: float = 0.15
    turnover_cost_bps: float = 6.0
    cash_buffer: float = 0.0
    kelly_shrinkage: float = 1.0
    switch_score_hurdle: float = 0.10
    min_regime_score: float = 0.70
    max_sleeve_turnover: float = 0.75
    incumbent_floor_weight: float = 0.20
    require_continuity_pass: bool = True
    allow_rebuilt_candidates: bool = False


def _kelly_target_exposure(
    history: dict[str, np.ndarray],
    target_weights: dict[str, float],
    *,
    cash_buffer: float,
    kelly_shrinkage: float,
) -> float:
    base_exposure = max(0.0, min(1.0 - max(0.0, cash_buffer), sum(target_weights.values())))
    if base_exposure <= 1e-12 or not target_weights:
        return 0.0
    weighted: np.ndarray | None = None
    for cid, weight in target_weights.items():
        raw_series = history.get(cid)
        if raw_series is None:
            series = np.asarray([], dtype=float)
        else:
            series = np.asarray(raw_series, dtype=float)
        if series.size == 0:
            continue
        contribution = float(weight) * series
        weighted = contribution if weighted is None else weighted + contribution
    if weighted is None or weighted.size < 2:
        return base_exposure
    mean_ret = float(np.mean(weighted))
    variance = float(np.var(weighted, ddof=1))
    if variance <= 1e-12 or not math.isfinite(variance):
        return base_exposure
    raw_fraction = max(0.0, mean_ret / variance)
    kelly_fraction = max(
        0.0, min(1.0 - max(0.0, cash_buffer), raw_fraction * max(0.0, kelly_shrinkage))
    )
    return max(base_exposure, kelly_fraction)


def _allocation_summary(
    allocations: list[dict[str, Any]],
    *,
    candidate_ids: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "days": len(allocations),
        "avg_cash_weight": 0.0,
        "avg_sleeve_turnover": 0.0,
        "max_cash_weight": 0.0,
        "max_sleeve_turnover": 0.0,
        "avg_candidate_weights": dict.fromkeys(candidate_ids, 0.0),
    }
    if not allocations:
        return summary
    cash_values = [_safe_float(row.get("cash_weight"), 0.0) for row in allocations]
    turnover_values = [_safe_float(row.get("sleeve_turnover"), 0.0) for row in allocations]
    summary["avg_cash_weight"] = float(np.mean(cash_values))
    summary["avg_sleeve_turnover"] = float(np.mean(turnover_values))
    summary["max_cash_weight"] = float(max(cash_values))
    summary["max_sleeve_turnover"] = float(max(turnover_values))
    summary["avg_candidate_weights"] = {
        cid: float(
            np.mean([_safe_float((row.get("weights") or {}).get(cid), 0.0) for row in allocations])
        )
        for cid in candidate_ids
    }
    return summary


def _candidate_runtime_risk(
    candidate_meta: dict[str, Any],
    *,
    risk_state: dict[str, Any],
    incumbent_id: str | None,
    candidate_id: str,
    params: SwitchParams,
) -> dict[str, Any]:
    symbols = {str(symbol) for symbol in list(candidate_meta.get("symbols") or [])}
    components = list(candidate_meta.get("components") or [])
    timeframes = {
        str(component.get("timeframe") or "")
        for component in components
        if component.get("timeframe")
    }
    stream_mode = str(candidate_meta.get("stream_mode") or "unknown")
    rebuild_required = stream_mode == "rebuild_required"
    overlaps_continuity = bool(symbols & set(risk_state.get("symbols") or set())) and bool(
        timeframes & set(risk_state.get("timeframes") or set())
    )
    continuity_blocked = bool(
        params.require_continuity_pass
        and risk_state.get("continuity_failed")
        and overlaps_continuity
        and candidate_id != incumbent_id
    )
    rebuild_blocked = bool(
        rebuild_required and not params.allow_rebuilt_candidates and candidate_id != incumbent_id
    )
    return {
        "stream_mode": stream_mode,
        "rebuild_required": rebuild_required,
        "rebuild_blocked": rebuild_blocked,
        "continuity_blocked": continuity_blocked,
        "continuity_overlaps_candidate": overlaps_continuity,
        "continuity_status": risk_state.get("continuity_status"),
    }


def run_regime_switch_allocator(
    rows: list[dict[str, Any]],
    params: SwitchParams,
    *,
    regime_features: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ordered_days, matrix, meta = _candidate_panel(rows)
    ids = list(matrix.keys())
    resolved_regime = (
        regime_features
        if regime_features is not None
        else _helper._load_regime_features(_regime_feature_rows(rows))
    )
    incumbent_id = _infer_incumbent_id(meta)
    runtime_risk = _load_runtime_risk_state()
    allocations: list[dict[str, Any]] = []
    daily_returns: list[float] = []
    split_returns: dict[str, list[float]] = {"train": [], "val": [], "oos": []}
    portfolio_weights: dict[str, float] = {}
    sleeve_weights: dict[str, float] = {}

    for idx, day_key in enumerate(ordered_days):
        split = split_for_day_key(day_key)
        rebalance_now = idx > 0 and (
            not portfolio_weights or (idx % max(1, params.rebalance_days) == 0)
        )
        turnover_cost = 0.0
        if rebalance_now:
            prior_day = ordered_days[idx - 1]
            regime_row = dict(resolved_regime.get(prior_day) or {})
            regime_available = bool(regime_row)
            history = {cid: matrix[cid][max(0, idx - params.lookback_days) : idx] for cid in ids}
            raw_scores: dict[str, float] = {}
            diagnostics: dict[str, dict[str, Any]] = {}
            regime_component_rows: dict[str, list[dict[str, Any]]] = {}
            for cid, returns in history.items():
                metric = _helper._metrics(np.asarray(returns, dtype=float))
                trailing_return = float(metric["total_return"])
                trailing_sharpe = float(metric["sharpe"])
                trailing_drawdown = float(metric["max_drawdown"])
                trailing_vol = float(metric["volatility"])
                regime_score, component_diags = _portfolio_regime_score(
                    meta.get(cid) or {},
                    regime_row,
                    previous_active=bool(_safe_float(portfolio_weights.get(cid), 0.0) > 0.0),
                    strength=params.regime_strength,
                )
                regime_component_rows[cid] = component_diags
                diagnostics[cid] = {
                    "trailing_return": trailing_return,
                    "trailing_sharpe": trailing_sharpe,
                    "trailing_drawdown": trailing_drawdown,
                    "trailing_volatility": trailing_vol,
                    "regime_score": regime_score,
                    "regime_available": regime_available,
                }
                risk_diag = _candidate_runtime_risk(
                    meta.get(cid) or {},
                    risk_state=runtime_risk,
                    incumbent_id=incumbent_id,
                    candidate_id=cid,
                    params=params,
                )
                diagnostics[cid].update(risk_diag)
                if trailing_sharpe < params.min_trailing_sharpe:
                    continue
                if trailing_return < params.min_trailing_return:
                    continue
                if trailing_drawdown > params.max_trailing_drawdown:
                    continue
                if risk_diag["rebuild_blocked"] or risk_diag["continuity_blocked"]:
                    continue
                if regime_available and regime_score < params.min_regime_score:
                    continue
                vol_inv = 1.0 / max(trailing_vol, 1e-6)
                dd_penalty = max(
                    0.05, 1.0 - (trailing_drawdown / max(params.max_trailing_drawdown, 1e-6))
                )
                persistence = 1.0 + (params.hysteresis_bonus if cid in portfolio_weights else 0.0)
                switch_drag = 1.0
                if cid not in portfolio_weights:
                    switch_drag = 1.0 / max(1.0, 1.0 + params.switch_score_hurdle)
                raw_scores[cid] = float(
                    (1.0 + max(0.0, trailing_sharpe))
                    * (1.0 + (8.0 * max(0.0, trailing_return)))
                    * vol_inv
                    * dd_penalty
                    * max(0.0, regime_score)
                    * persistence
                    * switch_drag
                )
            if not regime_available and incumbent_id is not None:
                incumbent_only = _safe_float(raw_scores.get(incumbent_id), 0.0)
                raw_scores = {incumbent_id: incumbent_only} if incumbent_only > 0.0 else {}
            target_total_exposure = max(0.0, 1.0 - params.cash_buffer)
            if raw_scores:
                unconstrained = _blend_scores(
                    history,
                    raw_scores,
                    target_total_exposure=target_total_exposure,
                    max_portfolio_weight=params.max_portfolio_weight,
                    correlation_penalty=params.correlation_penalty,
                )
                target_total_exposure = _kelly_target_exposure(
                    history,
                    unconstrained,
                    cash_buffer=params.cash_buffer,
                    kelly_shrinkage=params.kelly_shrinkage,
                )
            next_portfolio_weights = _blend_scores(
                history,
                raw_scores,
                target_total_exposure=target_total_exposure,
                max_portfolio_weight=params.max_portfolio_weight,
                correlation_penalty=params.correlation_penalty,
            )
            scaled_portfolio_weights = dict(next_portfolio_weights)
            scaled_portfolio_weights = _apply_incumbent_floor(
                scaled_portfolio_weights,
                incumbent_id=incumbent_id,
                floor_weight=params.incumbent_floor_weight,
            )
            next_sleeve_map = _aggregate_component_weights(scaled_portfolio_weights, meta=meta)
            next_sleeve_weights = {
                cid: _safe_float(item.get("weight"), 0.0) for cid, item in next_sleeve_map.items()
            }
            sleeve_turnover = _turnover(sleeve_weights, next_sleeve_weights)
            if sleeve_weights and sleeve_turnover > params.max_sleeve_turnover:
                scaled_portfolio_weights = _cap_turnover_transition(
                    portfolio_weights,
                    scaled_portfolio_weights,
                    max_turnover=params.max_sleeve_turnover,
                )
                scaled_portfolio_weights = _apply_incumbent_floor(
                    scaled_portfolio_weights,
                    incumbent_id=incumbent_id,
                    floor_weight=params.incumbent_floor_weight,
                )
                next_sleeve_map = _aggregate_component_weights(scaled_portfolio_weights, meta=meta)
                next_sleeve_weights = {
                    cid: _safe_float(item.get("weight"), 0.0)
                    for cid, item in next_sleeve_map.items()
                }
                sleeve_turnover = _turnover(sleeve_weights, next_sleeve_weights)
            turnover_cost = (max(0.0, params.turnover_cost_bps) / 10_000.0) * sleeve_turnover
            portfolio_weights = scaled_portfolio_weights
            sleeve_weights = next_sleeve_weights
            cash_weight = max(params.cash_buffer, 1.0 - sum(portfolio_weights.values()))
            allocations.append(
                {
                    "date": day_key,
                    "weights": dict(portfolio_weights),
                    "cash_weight": cash_weight,
                    "sleeve_weights": [
                        dict(item)
                        for item in sorted(
                            next_sleeve_map.values(),
                            key=lambda item: float(item.get("weight", 0.0)),
                            reverse=True,
                        )
                    ],
                    "diagnostics": diagnostics,
                    "component_regime_diagnostics": regime_component_rows,
                    "raw_scores": raw_scores,
                    "regime_row": regime_row,
                    "incumbent_id": incumbent_id,
                    "runtime_risk_state": runtime_risk,
                    "turnover_cost": float(turnover_cost),
                    "sleeve_turnover": float(sleeve_turnover),
                    "target_total_exposure": float(target_total_exposure),
                }
            )
        elif idx == 0:
            allocations.append(
                {
                    "date": day_key,
                    "weights": {},
                    "cash_weight": 1.0,
                    "sleeve_weights": [],
                    "diagnostics": {},
                    "component_regime_diagnostics": {},
                    "raw_scores": {},
                    "regime_row": {},
                    "incumbent_id": incumbent_id,
                    "runtime_risk_state": runtime_risk,
                    "turnover_cost": 0.0,
                    "sleeve_turnover": 0.0,
                    "target_total_exposure": 0.0,
                }
            )

        day_return = sum(
            float(portfolio_weights.get(cid, 0.0)) * float(matrix[cid][idx]) for cid in ids
        )
        day_return -= turnover_cost
        daily_returns.append(day_return)
        split_returns[split].append(day_return)

    return {
        "dates": ordered_days,
        "daily_returns": daily_returns,
        "allocations": allocations,
        "meta": meta,
        "runtime_risk_state": runtime_risk,
        "split_metrics": {
            split: _helper._metrics(np.asarray(values, dtype=float))
            for split, values in split_returns.items()
        },
        "all_metrics": _helper._metrics(np.asarray(daily_returns, dtype=float)),
    }


def _mean_split_fraction(
    allocations: list[dict[str, Any]],
    *,
    split: str,
    key: str,
) -> float:
    values = [
        _safe_float(item.get(key), 0.0)
        for item in allocations
        if split_for_day_key(str(item.get("date") or "")) == split
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def _objective(
    metrics: dict[str, float],
    *,
    cash_fraction: float,
    turnover_fraction: float,
) -> float:
    return float(
        (1.0 * _safe_float(metrics.get("sharpe"), 0.0))
        + (0.35 * _safe_float(metrics.get("sortino"), 0.0))
        + (0.15 * _safe_float(metrics.get("calmar"), 0.0))
        + (12.0 * _safe_float(metrics.get("total_return"), 0.0))
        - (4.0 * _safe_float(metrics.get("max_drawdown"), 0.0))
        - (0.75 * _safe_float(metrics.get("volatility"), 0.0))
        - (0.60 * cash_fraction)
        - (1.50 * turnover_fraction)
    )


def search_regime_switch_allocator(
    rows: list[dict[str, Any]],
    *,
    param_grid: list[SwitchParams] | None = None,
    progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    regime_features = _helper._load_regime_features(_regime_feature_rows(rows))
    grid = param_grid or [
        SwitchParams(*combo)
        for combo in itertools.product(
            [10, 14, 21],  # lookback
            [7],  # weekly rebalance
            [0.0, 0.25],  # min sharpe
            [0.0],  # min return
            [0.10, 0.15],  # max dd
            [0.55, 0.75],  # max portfolio weight
            [0.5],  # corr penalty
            [1.0, 1.5],  # regime strength
            [0.05, 0.20],  # hysteresis
            [8.0],  # turnover bps
            [0.0],  # cash buffer
            [0.75, 1.0, 1.25],  # kelly shrinkage
        )
    ]
    best: dict[str, Any] | None = None
    total_candidates = len(grid)
    for idx, params in enumerate(grid, start=1):
        result = run_regime_switch_allocator(rows, params, regime_features=regime_features)
        val_metrics = dict((result.get("split_metrics") or {}).get("val") or {})
        allocations = list(result.get("allocations") or [])
        cash_fraction = _mean_split_fraction(allocations, split="val", key="cash_weight")
        turnover_fraction = _mean_split_fraction(allocations, split="val", key="sleeve_turnover")
        objective = _objective(
            val_metrics,
            cash_fraction=cash_fraction,
            turnover_fraction=turnover_fraction,
        )
        candidate = {
            "params": asdict(params),
            "objective": objective,
            "result": result,
        }
        if best is None or objective > float(best["objective"]):
            best = candidate
        if progress_callback is not None:
            progress_callback(
                "regime_switch_candidate_evaluated",
                {
                    "candidate_index": idx,
                    "candidate_count": total_candidates,
                    "objective": objective,
                    "val_total_return": _safe_float(val_metrics.get("total_return"), 0.0),
                    "val_sharpe": _safe_float(val_metrics.get("sharpe"), 0.0),
                    "params": dict(candidate["params"]),
                },
            )
    if best is None:
        raise RuntimeError("regime switch search produced no result")
    return best


def _portfolio_return_streams_from_daily(
    dates: list[str],
    daily_returns: list[float],
) -> dict[str, list[dict[str, Any]]]:
    return _streams_from_daily_returns(dates, daily_returns)


def _final_allocation_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    allocations = list(result.get("allocations") or [])
    if not allocations:
        return []
    latest = allocations[-1]
    meta = dict(result.get("meta") or {})
    rows = []
    for cid, weight in sorted(
        dict(latest.get("weights") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        item = dict(meta.get(cid) or {})
        rows.append(
            {
                "candidate_id": cid,
                "name": item.get("name") or cid,
                "selection_basis": item.get("selection_basis"),
                "symbols": list(item.get("symbols") or []),
                "families": list(item.get("families") or []),
                "weight": float(weight),
            }
        )
    return rows


def write_regime_switch_report(
    *,
    input_paths: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_guard = acquire_portfolio_memory_guard(
        run_name="regime_switching_portfolio",
        output_dir=output_dir,
        input_path="::".join(input_paths),
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    status = "completed"
    error: str | None = None
    try:
        memory_guard.sample(event="regime_switch_start", context={"input_paths": list(input_paths)})
        rows = _load_switch_candidates(input_paths)
        memory_guard.checkpoint(
            "regime_switch_candidates_loaded",
            {"candidate_count": len(rows), "candidate_ids": [row["candidate_id"] for row in rows]},
        )
        best = search_regime_switch_allocator(rows, progress_callback=memory_guard.checkpoint)
        result = dict(best["result"])
    except RSSLimitExceeded as exc:
        status = "aborted_rss_guard"
        error = str(exc)
        raise
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        memory_guard.sample(
            event="regime_switch_finish", context={"status": status, "error": error}
        )
        memory_summary = memory_guard.finalize(
            status=status,
            error=error,
            context={"input_paths": list(input_paths)},
        )
        memory_guard.release()

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "regime_switching_portfolio",
        "schema_version": "1.0",
        "input_paths": list(input_paths),
        "selection_basis": "validation_only_regime_aware_portfolio_switching",
        "objective_profile": "balanced_multi_metric_with_turnover_penalty",
        "split_windows": split_windows(),
        "memory_policy": memory_policy_payload(
            budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        ),
        "memory_summary": memory_summary,
        "best_params": dict(best["params"]),
        "validation_objective": float(best["objective"]),
        "runtime_risk_state": dict(result.get("runtime_risk_state") or {}),
        "split_metrics": dict(result.get("split_metrics") or {}),
        "all_metrics": dict(result.get("all_metrics") or {}),
        "allocation_count": len(list(result.get("allocations") or [])),
        "final_allocation": _final_allocation_rows(result),
        "allocations": list(result.get("allocations") or []),
        "allocation_summary": {
            "all": _allocation_summary(
                list(result.get("allocations") or []),
                candidate_ids=[row["candidate_id"] for row in rows],
            ),
            "train": _allocation_summary(
                [
                    item
                    for item in list(result.get("allocations") or [])
                    if split_for_day_key(str(item.get("date") or "")) == "train"
                ],
                candidate_ids=[row["candidate_id"] for row in rows],
            ),
            "val": _allocation_summary(
                [
                    item
                    for item in list(result.get("allocations") or [])
                    if split_for_day_key(str(item.get("date") or "")) == "val"
                ],
                candidate_ids=[row["candidate_id"] for row in rows],
            ),
            "oos": _allocation_summary(
                [
                    item
                    for item in list(result.get("allocations") or [])
                    if split_for_day_key(str(item.get("date") or "")) == "oos"
                ],
                candidate_ids=[row["candidate_id"] for row in rows],
            ),
        },
        "portfolio_return_streams": _portfolio_return_streams_from_daily(
            list(result.get("dates") or []),
            list(result.get("daily_returns") or []),
        ),
        "candidate_universe": [
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "path": row.get("path"),
                "selection_basis": row.get("selection_basis"),
                "stream_mode": row.get("stream_mode"),
                "symbols": list(row.get("symbols") or []),
                "component_count": len(list(row.get("components") or [])),
            }
            for row in rows
        ],
        "universe_scope": "saved_portfolio_candidates",
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"regime_switching_portfolio_{stamp}.json"
    latest_path = output_dir / "regime_switching_portfolio_latest.json"
    md_path = output_dir / f"regime_switching_portfolio_{stamp}.md"
    serializable_payload = _json_ready(payload)
    json_path.write_text(json.dumps(serializable_payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(serializable_payload, indent=2), encoding="utf-8")
    lines = [
        "# Regime Switching Portfolio",
        "",
        f"- selection_basis: `{payload['selection_basis']}`",
        f"- validation_objective: `{payload['validation_objective']:.6f}`",
        f"- memory_log: `{dict(payload.get('memory_summary') or {}).get('rss_log_path')}`",
        "",
        "## Candidate universe",
        "",
    ]
    for row in payload["candidate_universe"]:
        lines.append(
            f"- `{row['name']}` | selection_basis={row['selection_basis']} | components={row['component_count']} | symbols={','.join(row['symbols'])}"
        )
    lines += [
        "",
        "## Best params",
        "",
        "```json",
        json.dumps(payload["best_params"], indent=2, sort_keys=True),
        "```",
        "",
        "## Split metrics",
        "",
        f"- train: {json.dumps(payload['split_metrics'].get('train') or {}, sort_keys=True)}",
        f"- val: {json.dumps(payload['split_metrics'].get('val') or {}, sort_keys=True)}",
        f"- oos: {json.dumps(payload['split_metrics'].get('oos') or {}, sort_keys=True)}",
        "",
        "## Cash / allocation summary",
        "",
        f"- all: {json.dumps(payload['allocation_summary'].get('all') or {}, sort_keys=True)}",
        f"- train: {json.dumps(payload['allocation_summary'].get('train') or {}, sort_keys=True)}",
        f"- val: {json.dumps(payload['allocation_summary'].get('val') or {}, sort_keys=True)}",
        f"- oos: {json.dumps(payload['allocation_summary'].get('oos') or {}, sort_keys=True)}",
        "",
        "## Final portfolio weights",
        "",
    ]
    for row in payload["final_allocation"]:
        lines.append(
            f"- `{row['name']}` | weight={row['weight']:.2%} | families={','.join(row['families']) or 'n/a'} | symbols={','.join(row['symbols']) or 'n/a'}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "latest_path": str(latest_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def write_regime_switch_comparison(
    *,
    switch_payload: dict[str, Any],
    comparison_input: Path | None = None,
) -> dict[str, Any]:
    comparison_path = comparison_input or DEFAULT_COMPARISON_INPUT
    base = _load_json(resolve_followup_artifact_path(comparison_path))
    switch_oos = dict((switch_payload.get("split_metrics") or {}).get("oos") or {})
    base["regime_switching_portfolio"] = {
        "path": str((DEFAULT_OUTPUT_DIR / "regime_switching_portfolio_latest.json").resolve()),
        "val": dict((switch_payload.get("split_metrics") or {}).get("val") or {}),
        "oos": switch_oos,
        "weights": list(switch_payload.get("final_allocation") or []),
        "best_params": dict(switch_payload.get("best_params") or {}),
    }
    scope = list(base.get("comparison_scope") or [])
    if "regime_switching_portfolio" not in scope:
        scope.append("regime_switching_portfolio")
    base["comparison_scope"] = scope
    incumbent_oos = dict((base.get("current_one_shot_optimized") or {}).get("oos") or {})
    if "deltas" not in base or not isinstance(base["deltas"], dict):
        base["deltas"] = {}
    base["deltas"]["regime_switch_vs_current_one_shot_oos_return"] = _safe_float(
        switch_oos.get("total_return"), 0.0
    ) - _safe_float(incumbent_oos.get("total_return"), 0.0)
    base["deltas"]["regime_switch_vs_current_one_shot_oos_sharpe"] = _safe_float(
        switch_oos.get("sharpe"), 0.0
    ) - _safe_float(incumbent_oos.get("sharpe"), 0.0)
    out_json = comparison_path.parent / "portfolio_regime_switch_comparison_latest.json"
    out_md = comparison_path.parent / "portfolio_regime_switch_comparison_latest.md"
    out_json.write_text(json.dumps(base, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Regime Switch Comparison",
        "",
        f"- regime_switch_vs_current_one_shot_oos_return: {base['deltas']['regime_switch_vs_current_one_shot_oos_return']:.4%}",
        f"- regime_switch_vs_current_one_shot_oos_sharpe: {base['deltas']['regime_switch_vs_current_one_shot_oos_sharpe']:.3f}",
        "",
        "## Regime switch OOS metrics",
        "",
        json.dumps(switch_oos, sort_keys=True),
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json_path": str(out_json.resolve()), "md_path": str(out_md.resolve())}


def write_regime_switch_preflight(
    *,
    input_paths: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_risk = _load_runtime_risk_state()
    rows = _load_switch_candidate_metadata(input_paths)
    ordered_meta = {row["candidate_id"]: row for row in rows}
    incumbent_id = _infer_incumbent_id(ordered_meta)
    params = SwitchParams(
        lookback_days=5,
        rebalance_days=1,
        min_trailing_sharpe=0.0,
        min_trailing_return=0.0,
        max_trailing_drawdown=0.15,
        max_portfolio_weight=0.75,
    )
    candidate_status = []
    for row in rows:
        cid = str(row["candidate_id"])
        risk_diag = _candidate_runtime_risk(
            row,
            risk_state=runtime_risk,
            incumbent_id=incumbent_id,
            candidate_id=cid,
            params=params,
        )
        blocking_reasons: list[str] = []
        if risk_diag["rebuild_blocked"]:
            blocking_reasons.append("rebuild_blocked")
        if risk_diag["continuity_blocked"]:
            blocking_reasons.append("continuity_blocked")
        candidate_status.append(
            {
                "candidate_id": cid,
                "name": row.get("name") or cid,
                "path": row.get("path"),
                "selection_basis": row.get("selection_basis"),
                "stream_mode": row.get("stream_mode"),
                "symbols": list(row.get("symbols") or []),
                "families": sorted(
                    {
                        str(component.get("family") or "other")
                        for component in list(row.get("components") or [])
                    }
                ),
                "component_count": len(list(row.get("components") or [])),
                "is_incumbent": cid == incumbent_id,
                "blocking_reasons": blocking_reasons,
                "risk": risk_diag,
            }
        )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "regime_switching_preflight",
        "schema_version": "1.0",
        "input_paths": list(input_paths),
        "incumbent_id": incumbent_id,
        "runtime_risk_state": runtime_risk,
        "candidate_status": candidate_status,
        "summary": {
            "candidate_count": len(candidate_status),
            "blocked_candidate_count": sum(
                1 for row in candidate_status if row["blocking_reasons"]
            ),
            "rebuild_blocked_count": sum(
                1 for row in candidate_status if "rebuild_blocked" in row["blocking_reasons"]
            ),
            "continuity_blocked_count": sum(
                1 for row in candidate_status if "continuity_blocked" in row["blocking_reasons"]
            ),
        },
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"regime_switching_preflight_{stamp}.json"
    latest_path = output_dir / "regime_switching_preflight_latest.json"
    md_path = output_dir / f"regime_switching_preflight_{stamp}.md"
    serializable_payload = _json_ready(payload)
    json_path.write_text(json.dumps(serializable_payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(serializable_payload, indent=2), encoding="utf-8")
    lines = [
        "# Regime Switching Preflight",
        "",
        f"- incumbent_id: `{incumbent_id}`",
        f"- continuity_status: `{runtime_risk.get('continuity_status')}`",
        f"- continuity_failed: `{runtime_risk.get('continuity_failed')}`",
        f"- blocked_candidates: `{payload['summary']['blocked_candidate_count']}` / `{payload['summary']['candidate_count']}`",
        "",
        "## Candidate status",
        "",
    ]
    for row in candidate_status:
        blockers = ",".join(row["blocking_reasons"]) or "none"
        lines.append(
            f"- `{row['name']}` | incumbent={row['is_incumbent']} | stream_mode={row['stream_mode']} | blockers={blockers} | symbols={','.join(row['symbols']) or 'n/a'}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "payload": payload,
        "json_path": str(json_path.resolve()),
        "latest_path": str(latest_path.resolve()),
        "md_path": str(md_path.resolve()),
    }


def _default_input_paths() -> list[str]:
    inputs = [str(resolve_current_optimization_path(PORTFOLIO_CURRENT_OPTIMIZATION))]
    for candidate in (DEFAULT_OVERLAY_PORTFOLIO, DEFAULT_DYNAMIC_PORTFOLIO):
        resolved = resolve_followup_artifact_path(candidate)
        if resolved.exists():
            inputs.append(str(resolved.resolve()))
    autoresearch = _discover_latest_autoresearch_candidate()
    if autoresearch is not None:
        inputs.append(str(autoresearch))
    return inputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a regime-aware portfolio switcher over saved portfolio streams."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=None,
        help="Portfolio optimization JSON path. May be provided multiple times.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write a low-memory risk preflight report without loading or rebuilding return streams.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_paths = list(args.inputs or _default_input_paths())
    if args.preflight_only:
        preflight = write_regime_switch_preflight(
            input_paths=input_paths,
            output_dir=Path(args.output_dir).resolve(),
        )
        print(preflight["latest_path"])
        print(preflight["md_path"])
        return 0
    report = write_regime_switch_report(
        input_paths=input_paths,
        output_dir=Path(args.output_dir).resolve(),
    )
    comparison = write_regime_switch_comparison(switch_payload=report["payload"])
    print(report["latest_path"])
    print(report["md_path"])
    print(comparison["json_path"])
    print(comparison["md_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
