"""Strictly re-evaluate grouped allocator leverage winners on candidate-level research paths."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    resolve_current_optimization_path,
    resolve_followup_artifact_path,
    resolve_incumbent_bundle_path,
)

ROOT = Path(__file__).resolve().parents[2]

_market_spec = importlib.util.spec_from_file_location(
    "run_group_market_regime_judgement",
    Path(__file__).resolve().parent / "run_group_market_regime_judgement.py",
)
if _market_spec is None or _market_spec.loader is None:
    raise RuntimeError("Failed to load run_group_market_regime_judgement helpers")
_market = importlib.util.module_from_spec(_market_spec)
sys.modules[_market_spec.name] = _market
_market_spec.loader.exec_module(_market)

_three_way_spec = importlib.util.spec_from_file_location(
    "run_three_way_market_regime_allocator",
    Path(__file__).resolve().parent / "run_three_way_market_regime_allocator.py",
)
if _three_way_spec is None or _three_way_spec.loader is None:
    raise RuntimeError("Failed to load run_three_way_market_regime_allocator helpers")
_three_way = importlib.util.module_from_spec(_three_way_spec)
sys.modules[_three_way_spec.name] = _three_way
_three_way_spec.loader.exec_module(_three_way)

_validation_spec = importlib.util.spec_from_file_location(
    "validate_saved_incumbent_portfolio",
    Path(__file__).resolve().parent / "validate_saved_incumbent_portfolio.py",
)
if _validation_spec is None or _validation_spec.loader is None:
    raise RuntimeError("Failed to load validate_saved_incumbent_portfolio helpers")
_validation = importlib.util.module_from_spec(_validation_spec)
sys.modules[_validation_spec.name] = _validation
_validation_spec.loader.exec_module(_validation)

_decision_spec = importlib.util.spec_from_file_location(
    "write_portfolio_max_performance_decision",
    Path(__file__).resolve().parent / "write_portfolio_max_performance_decision.py",
)
if _decision_spec is None or _decision_spec.loader is None:
    raise RuntimeError("Failed to load write_portfolio_max_performance_decision helpers")
_decision = importlib.util.module_from_spec(_decision_spec)
sys.modules[_decision_spec.name] = _decision
_decision_spec.loader.exec_module(_decision)

_leverage_tuning_spec = importlib.util.spec_from_file_location(
    "run_grouped_allocator_leverage_tuning",
    Path(__file__).resolve().parent / "run_grouped_allocator_leverage_tuning.py",
)
if _leverage_tuning_spec is None or _leverage_tuning_spec.loader is None:
    raise RuntimeError("Failed to load run_grouped_allocator_leverage_tuning helpers")
_leverage_tuning = importlib.util.module_from_spec(_leverage_tuning_spec)
sys.modules[_leverage_tuning_spec.name] = _leverage_tuning
_leverage_tuning_spec.loader.exec_module(_leverage_tuning)

DEFAULT_LEVERAGE_TUNING_PATH = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_allocator_leverage_tuning_current"
    / "grouped_allocator_leverage_tuning_latest.json"
)
DEFAULT_OUTPUT_DIR = (
    FOLLOWUP_ROOT
    / "portfolio_incumbent_autoresearch_grouped"
    / "grouped_allocator_strict_leverage_validation_current"
)
DEFAULT_DECISION_PATH = FOLLOWUP_ROOT / "portfolio_max_performance_decision_latest.json"
DEFAULT_INCUMBENT_BUNDLE = resolve_incumbent_bundle_path(_decision.DEFAULT_INCUMBENT_BUNDLE)
DEFAULT_BLEND_WEIGHT = 0.85
DEFAULT_HORIZON_DAYS = int(_market.DEFAULT_HORIZON_DAYS)
DEFAULT_SOFT_RSS_BYTES = int(_market.DEFAULT_SOFT_RSS_BYTES)
DEFAULT_HARD_RSS_BYTES = int(_market.DEFAULT_HARD_RSS_BYTES)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return _three_way._json_default(value)


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _path_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return dict(payload)


def _candidate_list_from_any_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = (
            payload.get("candidates") or payload.get("selected_team") or payload.get("rows") or []
        )
        return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _matches_candidate_row(
    row: dict[str, Any],
    *,
    candidate_id: str,
    candidate_name: str,
) -> bool:
    row_id, row_name = _candidate_key(row)
    return bool(candidate_id and row_id == candidate_id) or bool(
        candidate_name and row_name == candidate_name
    )


def _resolve_candidate_from_strict_cache(
    *,
    candidate_id: str,
    candidate_name: str,
) -> dict[str, Any] | None:
    cache_dir = Path(_validation.DEFAULT_STRICT_VALIDATION_CACHE_DIR).expanduser().resolve()
    if not cache_dir.exists():
        return None
    cache_paths = sorted(
        cache_dir.glob("*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.as_posix()),
        reverse=True,
    )
    for cache_path in cache_paths:
        payload = _validation._load_strict_validation_cache(cache_path)
        if not isinstance(payload, dict):
            continue
        for row in list(payload.get("candidates") or []):
            if not isinstance(row, dict):
                continue
            if _matches_candidate_row(
                row, candidate_id=candidate_id, candidate_name=candidate_name
            ):
                resolved = dict(row)
                resolved.setdefault("source_path", str(cache_path))
                resolved.setdefault("artifact_path", str(cache_path))
                return resolved
    return None


def _portfolio_oos_end(payload: dict[str, Any]) -> datetime:
    streams = dict(payload.get("portfolio_return_streams") or {})
    oos_stream = list(streams.get("oos") or [])
    end = _validation._stream_end_utc(oos_stream)
    if end is None:
        raise RuntimeError("portfolio artifact is missing OOS return streams")
    return end


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    return (
        str(candidate.get("candidate_id") or "").strip(),
        str(candidate.get("name") or "").strip(),
    )


def _resolve_candidate_from_source_component(component: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(component.get("candidate_id") or "").strip()
    candidate_name = str(component.get("name") or "").strip()
    raw_artifact_path = str(component.get("artifact_path") or "").strip()
    artifact_path = (
        resolve_followup_artifact_path(raw_artifact_path) if raw_artifact_path else Path()
    )
    if raw_artifact_path and artifact_path.exists():
        rows = _candidate_list_from_any_json(artifact_path)
        for row in rows:
            if _matches_candidate_row(
                row, candidate_id=candidate_id, candidate_name=candidate_name
            ):
                return dict(row)

    cached = _resolve_candidate_from_strict_cache(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
    )
    if cached is not None:
        return cached
    label = candidate_id or candidate_name or artifact_path.name
    source_label = raw_artifact_path or str(artifact_path) or "<missing-artifact-path>"
    raise RuntimeError(
        f"unable to resolve candidate {label} from {source_label} or strict validation cache"
    )


def _resolve_portfolio_candidates(
    *, bundle_payload: dict[str, Any] | None = None, portfolio_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    if bundle_payload is not None:
        rows = bundle_payload.get("selected_team") or bundle_payload.get("candidates") or []
        if rows:
            return [dict(row) for row in rows if isinstance(row, dict)]

    direct_rows = (
        portfolio_payload.get("selected_team") or portfolio_payload.get("candidates") or []
    )
    direct_candidates = [dict(row) for row in direct_rows if isinstance(row, dict)]
    if direct_candidates and any(isinstance(row.get("params"), dict) for row in direct_candidates):
        return direct_candidates

    source_components = list(portfolio_payload.get("source_components") or [])
    if source_components:
        return [
            _resolve_candidate_from_source_component(dict(component))
            for component in source_components
            if isinstance(component, dict)
        ]

    weights = [
        dict(row) for row in list(portfolio_payload.get("weights") or []) if isinstance(row, dict)
    ]
    if weights and all(isinstance(row.get("params"), dict) for row in weights):
        return weights

    raise RuntimeError(
        "portfolio artifact does not contain resolvable candidate definitions with params"
    )


def _apply_group_leverage(
    candidates: list[dict[str, Any]], *, leverage: int
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    lev = max(1, int(leverage))
    for candidate in candidates:
        row = copy.deepcopy(candidate)
        params = dict(row.get("params") or {})
        params["leverage"] = lev
        row["params"] = params
        row["leverage"] = lev
        resolved.append(row)
    return resolved


def _symbol_universe(candidates: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            _validation._research_symbol(symbol)
            for row in candidates
            for symbol in list(row.get("symbols") or [])
            if str(symbol).strip()
        }
    )


def _strategy_timeframes(candidates: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip().lower()
            for row in candidates
            if str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip()
        }
    )


def _strict_candidate_report(
    *, candidates: list[dict[str, Any]], split: dict[str, str]
) -> dict[str, Any]:
    symbols = _symbol_universe(candidates)
    timeframes = _strategy_timeframes(candidates)
    if not symbols or not timeframes:
        raise RuntimeError("strict candidate report requires non-empty symbols and timeframes")
    return _validation._run_strict_research(
        candidates=candidates,
        strategy_timeframes=timeframes,
        symbol_universe=symbols,
        split=split,
        min_bundle_bars=1,
    )


def _apply_candidate_level_leverage_to_stream(
    stream: list[dict[str, Any]],
    *,
    leverage: int,
    label: str,
) -> tuple[list[dict[str, Any]], int]:
    if int(leverage) <= 1 or not list(stream or []):
        return [dict(point) for point in list(stream or []) if isinstance(point, dict)], 0

    def _resolve_point_timestamp(point: dict[str, Any]) -> pd.Timestamp:
        raw_datetime = point.get("datetime")
        if isinstance(raw_datetime, str) and raw_datetime.strip():
            parsed = pd.to_datetime(raw_datetime, utc=True, errors="coerce")
            if not pd.isna(parsed):
                return pd.Timestamp(parsed)

        raw_t = point.get("t")
        if isinstance(raw_t, str) and raw_t.strip():
            parsed = pd.to_datetime(raw_t, utc=True, errors="coerce")
            if not pd.isna(parsed):
                return pd.Timestamp(parsed)

        if raw_t is not None:
            try:
                raw_numeric = float(raw_t)
            except (TypeError, ValueError):
                raw_numeric = None
            if raw_numeric is not None:
                unit = (
                    "ms" if abs(raw_numeric) >= 1e12 else "s" if abs(raw_numeric) >= 1e9 else None
                )
                parsed = (
                    pd.to_datetime(raw_numeric, unit=unit, utc=True, errors="coerce")
                    if unit is not None
                    else pd.to_datetime(raw_numeric, utc=True, errors="coerce")
                )
                if not pd.isna(parsed):
                    return pd.Timestamp(parsed)

        return pd.NaT

    frame = pd.DataFrame(
        [
            {
                "date": _resolve_point_timestamp(point),
                "split_group": "split",
                "state": str(label),
                "base_return": float(point.get("v") or 0.0),
                "raw_datetime": point.get("datetime"),
                "raw_t": point.get("t"),
            }
            for point in list(stream or [])
            if isinstance(point, dict)
        ]
    )
    if frame.empty:
        return [], 0
    frame = frame.dropna(subset=["date"]).reset_index(drop=True)
    tuned, liquidation_counts = _leverage_tuning._apply_state_leverage(
        frame[["date", "split_group", "state", "base_return"]],
        leverage_by_state={str(label): int(leverage)},
    )
    out: list[dict[str, Any]] = []
    for base_row, tuned_row in zip(
        frame.to_dict(orient="records"), tuned.to_dict(orient="records"), strict=False
    ):
        item: dict[str, Any] = {
            "v": float(tuned_row.get("leveraged_return", 0.0)),
            "t": base_row.get("raw_t") or pd.Timestamp(base_row["date"]).isoformat(),
        }
        raw_datetime = base_row.get("raw_datetime")
        if raw_datetime:
            item["datetime"] = raw_datetime
        else:
            item["datetime"] = pd.Timestamp(base_row["date"]).isoformat()
        out.append(item)
    return out, int(liquidation_counts.get(str(label), 0))


def _apply_candidate_level_leverage_to_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    leveraged_rows: list[dict[str, Any]] = []
    liquidation_counts: dict[str, int] = {}
    for row in list(rows or []):
        updated = copy.deepcopy(row)
        leverage = max(
            1, int((updated.get("params") or {}).get("leverage", updated.get("leverage", 1)) or 1)
        )
        total_liquidations = 0
        original_streams = dict(updated.get("return_streams") or {})
        leveraged_streams: dict[str, list[dict[str, Any]]] = {}
        for split_name in ("train", "val", "oos"):
            leveraged_stream, liq_count = _apply_candidate_level_leverage_to_stream(
                list(original_streams.get(split_name) or []),
                leverage=leverage,
                label=str(updated.get("candidate_id") or updated.get("name") or "candidate"),
            )
            leveraged_streams[split_name] = leveraged_stream
            total_liquidations += int(liq_count)
        updated["return_streams"] = leveraged_streams
        metadata = dict(updated.get("metadata") or {})
        metadata["candidate_level_leverage"] = leverage
        metadata["candidate_level_liquidation_count"] = total_liquidations
        updated["metadata"] = metadata
        liquidation_counts[
            str(updated.get("candidate_id") or updated.get("name") or "candidate")
        ] = total_liquidations
        leveraged_rows.append(updated)
    return leveraged_rows, liquidation_counts


def _build_group_portfolio_payload(
    *,
    label: str,
    source_payload: dict[str, Any],
    source_path: Path,
    leverage: int,
    strict_report: dict[str, Any],
    refreshed_rows: list[dict[str, Any]],
    eval_payload: dict[str, Any],
) -> dict[str, Any]:
    raw_streams = dict(eval_payload.get("portfolio_return_streams") or {})
    daily_streams = dict(eval_payload.get("portfolio_daily_return_streams") or {})
    return {
        "artifact_kind": f"strict_leverage_validated_{label}_portfolio",
        "generated_at": _utc_now_iso(),
        "selection_basis": "strict_candidate_level_leverage_research",
        "source_portfolio_path": str(source_path.resolve()),
        "source_artifact_kind": str(
            source_payload.get("artifact_kind") or "portfolio_optimization"
        ),
        "configured_group_leverage": int(leverage),
        "weights": [
            dict(row) for row in list(source_payload.get("weights") or []) if isinstance(row, dict)
        ],
        "source_components": [
            dict(row)
            for row in list(source_payload.get("source_components") or [])
            if isinstance(row, dict)
        ],
        "portfolio_metrics": dict(eval_payload.get("portfolio_metrics") or {}),
        "portfolio_return_streams": daily_streams,
        "portfolio_daily_return_streams": daily_streams,
        "portfolio_intraday_return_streams": raw_streams,
        "weighted_component_summaries": dict(
            eval_payload.get("weighted_component_summaries") or {}
        ),
        "component_rows": list(eval_payload.get("component_rows") or []),
        "oos_monthly_returns": list(eval_payload.get("oos_monthly_returns") or []),
        "sensitivity": dict(eval_payload.get("sensitivity") or {}),
        "strict_candidate_report": {
            "generated_at": strict_report.get("generated_at"),
            "data_sources": strict_report.get("data_sources"),
            "candidate_count": len(list(strict_report.get("candidates") or [])),
        },
        "validated_candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "name": row.get("name"),
                "strategy_class": row.get("strategy_class"),
                "timeframe": row.get("strategy_timeframe") or row.get("timeframe"),
                "symbols": list(row.get("symbols") or []),
                "leverage": int(
                    (row.get("params") or {}).get("leverage", row.get("leverage", leverage))
                ),
                "liquidation_count": int((row.get("metadata") or {}).get("liquidation_count") or 0),
            }
            for row in refreshed_rows
        ],
        "notes": [
            "Portfolio streams were rebuilt from strict candidate re-evaluation using candidate-level leverage.",
            "This artifact is intended for grouped allocator re-validation and optimism checks.",
        ],
    }


def _build_blend_payload(
    *,
    incumbent_payload: dict[str, Any],
    autoresearch_payload: dict[str, Any],
    blend_weight: float,
) -> dict[str, Any]:
    incumbent_weight = float(blend_weight)
    autoresearch_weight = float(1.0 - incumbent_weight)
    blend_rows = [
        {
            "candidate_id": "current_one_shot_incumbent",
            "name": "current_one_shot_incumbent",
            "_saved_weight": incumbent_weight,
            "return_streams": dict(incumbent_payload.get("portfolio_return_streams") or {}),
            "train": dict((incumbent_payload.get("portfolio_metrics") or {}).get("train") or {}),
            "val": dict((incumbent_payload.get("portfolio_metrics") or {}).get("val") or {}),
            "oos": dict((incumbent_payload.get("portfolio_metrics") or {}).get("oos") or {}),
            "metadata": {"cost_rate": 0.0005},
        },
        {
            "candidate_id": "autoresearch_pair_55_45",
            "name": "autoresearch_pair_55_45",
            "_saved_weight": autoresearch_weight,
            "return_streams": dict(autoresearch_payload.get("portfolio_return_streams") or {}),
            "train": dict((autoresearch_payload.get("portfolio_metrics") or {}).get("train") or {}),
            "val": dict((autoresearch_payload.get("portfolio_metrics") or {}).get("val") or {}),
            "oos": dict((autoresearch_payload.get("portfolio_metrics") or {}).get("oos") or {}),
            "metadata": {"cost_rate": 0.0005},
        },
    ]
    eval_payload = _validation.evaluate_saved_weight_portfolio(blend_rows)
    raw_streams = dict(eval_payload.get("portfolio_return_streams") or {})
    daily_streams = dict(eval_payload.get("portfolio_daily_return_streams") or {})
    return {
        "artifact_kind": "strict_grouped_incumbent_autoresearch_static_blend_portfolio",
        "generated_at": _utc_now_iso(),
        "selection_basis": "strict_candidate_level_leverage_research_static_group_blend",
        "groups_move_as_set": True,
        "best_weights": {
            "current_one_shot_incumbent": incumbent_weight,
            "autoresearch_pair_55_45": autoresearch_weight,
        },
        "final_group_allocation": [
            {"candidate_id": "current_one_shot_incumbent", "weight": incumbent_weight},
            {"candidate_id": "autoresearch_pair_55_45", "weight": autoresearch_weight},
        ],
        "input_paths": [
            str(Path(str(incumbent_payload.get("source_portfolio_path") or "")).resolve()),
            str(Path(str(autoresearch_payload.get("source_portfolio_path") or "")).resolve()),
        ],
        "weight_grid": [
            {
                "current_one_shot_incumbent": incumbent_weight,
                "autoresearch_pair_55_45": autoresearch_weight,
            }
        ],
        "validation_objective": float(
            _three_way._objective(
                dict((eval_payload.get("portfolio_metrics") or {}).get("val") or {}),
                turnover_fraction=float(
                    ((eval_payload.get("weighted_component_summaries") or {}).get("val") or {}).get(
                        "turnover", 0.0
                    )
                ),
            )
        ),
        "split_metrics": dict(eval_payload.get("portfolio_metrics") or {}),
        "portfolio_metrics": dict(eval_payload.get("portfolio_metrics") or {}),
        "portfolio_return_streams": daily_streams,
        "portfolio_daily_return_streams": daily_streams,
        "portfolio_intraday_return_streams": raw_streams,
        "weighted_component_summaries": dict(
            eval_payload.get("weighted_component_summaries") or {}
        ),
        "component_rows": list(eval_payload.get("component_rows") or []),
        "oos_monthly_returns": list(eval_payload.get("oos_monthly_returns") or []),
        "sensitivity": dict(eval_payload.get("sensitivity") or {}),
        "dates": [
            str(item.get("t"))
            for item in list(
                (eval_payload.get("portfolio_daily_return_streams") or {}).get("train") or []
            )
        ]
        + [
            str(item.get("t"))
            for item in list(
                (eval_payload.get("portfolio_daily_return_streams") or {}).get("val") or []
            )
        ]
        + [
            str(item.get("t"))
            for item in list(
                (eval_payload.get("portfolio_daily_return_streams") or {}).get("oos") or []
            )
        ],
        "daily_returns": [
            float(item.get("v", 0.0))
            for split in ("train", "val", "oos")
            for item in list(
                (eval_payload.get("portfolio_daily_return_streams") or {}).get(split) or []
            )
        ],
        "group_universe": ["current_one_shot_incumbent", "autoresearch_pair_55_45"],
        "notes": [
            "Static blend rebuilt from strict incumbent/autoresearch group portfolios.",
            f"Blend weights fixed at incumbent {incumbent_weight:.0%} / autoresearch {autoresearch_weight:.0%}.",
        ],
    }


def _write_payload(
    *, payload: dict[str, Any], output_dir: Path, stem: str, markdown: str
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_json = output_dir / f"{stem}_{timestamp}.json"
    out_md = output_dir / f"{stem}_{timestamp}.md"
    latest_json = output_dir / f"{stem}_latest.json"
    latest_md = output_dir / f"{stem}_latest.md"
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    out_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return {"json": latest_json, "md": latest_md}


def _comparison_block(
    strict_metrics: dict[str, Any], baseline_metrics: dict[str, Any]
) -> dict[str, dict[str, float]]:
    return _validation._build_comparison(baseline_metrics, strict_metrics)


def _apply_allocator_state_leverage_to_payload(
    *,
    allocator_payload: dict[str, Any],
    leverage_by_state: dict[str, Any],
) -> dict[str, Any]:
    resolved_leverage = {
        str(key): max(1, int(value)) for key, value in dict(leverage_by_state or {}).items()
    }
    if not resolved_leverage:
        return dict(allocator_payload)

    states = pd.DataFrame(list(allocator_payload.get("states") or []))
    required_columns = {"date", "split_group", "state", "return"}
    if states.empty or not required_columns.issubset(states.columns):
        return dict(allocator_payload)

    states = states.copy()
    states["date"] = pd.to_datetime(states["date"], utc=True)
    tuned, liquidation_counts = _leverage_tuning._apply_state_leverage(
        states[["date", "split_group", "state", "return"]].rename(
            columns={"return": "base_return"}
        ),
        leverage_by_state=resolved_leverage,
    )
    split_metrics = _three_way._metrics_by_split(
        tuned.rename(columns={"leveraged_return": "metric_return"}),
        "metric_return",
    )

    state_rows = list(allocator_payload.get("states") or [])
    merged_states: list[dict[str, Any]] = []
    for base_row, tuned_row in zip(state_rows, tuned.to_dict(orient="records"), strict=False):
        merged = dict(base_row)
        merged["base_return"] = float(tuned_row.get("base_return", 0.0))
        merged["return"] = float(tuned_row.get("leveraged_return", 0.0))
        merged["applied_leverage"] = int(tuned_row.get("leverage", 1))
        merged["segment_equity"] = float(tuned_row.get("segment_equity", 1.0))
        merged["segment_floor"] = float(tuned_row.get("segment_floor", 0.0))
        merged["liquidated"] = bool(tuned_row.get("liquidated", False))
        merged_states.append(merged)

    updated = dict(allocator_payload)
    updated["unleveraged_split_metrics"] = dict(updated.get("split_metrics") or {})
    updated["split_metrics"] = split_metrics
    updated["daily_returns"] = [float(value) for value in tuned["leveraged_return"]]
    updated["dates"] = [pd.Timestamp(value).isoformat() for value in tuned["date"]]
    updated["states"] = merged_states
    updated["state_leverage_validation"] = {
        "leverage_by_state": resolved_leverage,
        "liquidation_counts": {str(key): int(value) for key, value in liquidation_counts.items()},
        "selection_basis": "allocator_state_level_leverage_applied_after_strict_sleeve_rebuild",
    }
    if merged_states:
        updated["current_state"] = dict(merged_states[-1])
    return updated


def _resolve_current_promoted_candidate(decision_payload: dict[str, Any]) -> dict[str, Any]:
    winner = dict(decision_payload.get("winner") or {})
    winner_key = str(winner.get("candidate_key") or "").strip()
    candidates = [
        dict(row) for row in list(decision_payload.get("candidates") or []) if isinstance(row, dict)
    ]
    for row in candidates:
        if str(row.get("candidate_key") or "").strip() == winner_key:
            return row
    raise RuntimeError(
        f"unable to resolve promoted challenger {winner_key or 'unknown'} from decision artifact"
    )


def _build_markdown(payload: dict[str, Any]) -> str:
    strict_allocator = dict(payload.get("strict_allocator") or {})
    baseline = dict(payload.get("current_promoted_challenger") or {})
    comparison = dict(payload.get("comparison_vs_promoted_challenger") or {})
    lines = [
        "# Grouped Allocator Strict Leverage Validation",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- strict_oos_end: `{(payload.get('validation_split') or {}).get('oos_end')}`",
        f"- incumbent leverage: `{(payload.get('leverage_by_state') or {}).get('incumbent')}x`",
        f"- autoresearch leverage: `{(payload.get('leverage_by_state') or {}).get('autoresearch_55_45')}x`",
        "",
        "## Strict group portfolio metrics",
    ]
    for label in ("incumbent", "autoresearch_55_45", "blend_85_15", "allocator"):
        metrics = dict((payload.get("strict_group_metrics") or {}).get(label) or {})
        oos = dict(metrics.get("oos") or {})
        lines.append(
            f"- {label}: OOS return `{_validation.safe_float(oos.get('total_return'), 0.0):.4%}` | "
            f"Sharpe `{_validation.safe_float(oos.get('sharpe'), 0.0):.4f}` | "
            f"max DD `{_validation.safe_float(oos.get('max_drawdown'), 0.0):.4%}`"
        )
    lines.extend(["", "## Strict allocator vs current promoted challenger", ""])
    lines.append(
        f"- baseline: `{baseline.get('label')}` | strict allocator artifact: `{strict_allocator.get('artifact_path')}`"
    )
    for split in ("train", "val", "oos"):
        strict_split = dict((strict_allocator.get("split_metrics") or {}).get(split) or {})
        baseline_split = dict(baseline.get(split) or {})
        delta = dict(comparison.get(split) or {})
        lines.append(
            f"- {split}: strict return `{_validation.safe_float(strict_split.get('total_return'), 0.0):.4%}` "
            f"vs baseline `{_validation.safe_float(baseline_split.get('total_return'), 0.0):.4%}` "
            f"(Δ `{_validation.safe_float(delta.get('total_return_delta'), 0.0):.4%}`), "
            f"Sharpe Δ `{_validation.safe_float(delta.get('sharpe_delta'), 0.0):.4f}`, "
            f"max DD Δ `{_validation.safe_float(delta.get('max_drawdown_delta'), 0.0):.4%}`"
        )
    notes = list(payload.get("notes") or [])
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines).strip() + "\n"


def run_grouped_allocator_strict_leverage_validation(
    *,
    incumbent_bundle_path: Path,
    incumbent_portfolio_path: Path,
    autoresearch_portfolio_path: Path,
    leverage_tuning_path: Path,
    decision_path: Path,
    output_dir: Path,
    incumbent_leverage: int | None,
    autoresearch_leverage: int | None,
    blend_weight: float,
    horizon_days: int,
    soft_rss_bytes: int,
    hard_rss_bytes: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    incumbent_bundle = _path_payload(incumbent_bundle_path)
    incumbent_portfolio = _path_payload(incumbent_portfolio_path)
    autoresearch_portfolio = _path_payload(autoresearch_portfolio_path)
    leverage_payload = _path_payload(leverage_tuning_path)
    decision_payload = _path_payload(decision_path)

    leverage_by_state = dict(
        (leverage_payload.get("best_result") or {}).get("leverage_by_state") or {}
    )
    resolved_incumbent_leverage = max(
        1, int(incumbent_leverage or leverage_by_state.get("incumbent") or 1)
    )
    resolved_autoresearch_leverage = max(
        1, int(autoresearch_leverage or leverage_by_state.get("autoresearch_55_45") or 1)
    )

    common_oos_end = min(
        _portfolio_oos_end(incumbent_portfolio), _portfolio_oos_end(autoresearch_portfolio)
    )
    validation_split = _validation.build_validation_split(common_oos_end)

    incumbent_candidates = _apply_group_leverage(
        _resolve_portfolio_candidates(
            bundle_payload=incumbent_bundle, portfolio_payload=incumbent_portfolio
        ),
        leverage=resolved_incumbent_leverage,
    )
    autoresearch_candidates = _apply_group_leverage(
        _resolve_portfolio_candidates(portfolio_payload=autoresearch_portfolio),
        leverage=resolved_autoresearch_leverage,
    )

    incumbent_report = _strict_candidate_report(
        candidates=incumbent_candidates, split=validation_split
    )
    incumbent_rows = _validation._saved_weight_rows(
        list(incumbent_report.get("candidates") or []),
        [
            dict(row)
            for row in list(incumbent_portfolio.get("weights") or [])
            if isinstance(row, dict)
        ],
    )
    incumbent_rows = _apply_group_leverage(incumbent_rows, leverage=resolved_incumbent_leverage)
    incumbent_rows, incumbent_liquidations = _apply_candidate_level_leverage_to_rows(incumbent_rows)
    incumbent_eval = _validation.evaluate_saved_weight_portfolio(incumbent_rows)
    incumbent_payload = _build_group_portfolio_payload(
        label="incumbent",
        source_payload=incumbent_portfolio,
        source_path=incumbent_portfolio_path,
        leverage=resolved_incumbent_leverage,
        strict_report=incumbent_report,
        refreshed_rows=incumbent_rows,
        eval_payload=incumbent_eval,
    )
    incumbent_payload["candidate_level_liquidations"] = incumbent_liquidations
    incumbent_written = _write_payload(
        payload=incumbent_payload,
        output_dir=output_dir / "strict_incumbent_portfolio_current",
        stem="strict_incumbent_portfolio",
        markdown=f"# strict incumbent portfolio\n\n- leverage: `{resolved_incumbent_leverage}x`\n",
    )

    autoresearch_report = _strict_candidate_report(
        candidates=autoresearch_candidates, split=validation_split
    )
    autoresearch_rows = _validation._saved_weight_rows(
        list(autoresearch_report.get("candidates") or []),
        [
            dict(row)
            for row in list(autoresearch_portfolio.get("weights") or [])
            if isinstance(row, dict)
        ],
    )
    autoresearch_rows = _apply_group_leverage(
        autoresearch_rows, leverage=resolved_autoresearch_leverage
    )
    autoresearch_rows, autoresearch_liquidations = _apply_candidate_level_leverage_to_rows(
        autoresearch_rows
    )
    autoresearch_eval = _validation.evaluate_saved_weight_portfolio(autoresearch_rows)
    autoresearch_payload = _build_group_portfolio_payload(
        label="autoresearch_55_45",
        source_payload=autoresearch_portfolio,
        source_path=autoresearch_portfolio_path,
        leverage=resolved_autoresearch_leverage,
        strict_report=autoresearch_report,
        refreshed_rows=autoresearch_rows,
        eval_payload=autoresearch_eval,
    )
    autoresearch_payload["candidate_level_liquidations"] = autoresearch_liquidations
    autoresearch_written = _write_payload(
        payload=autoresearch_payload,
        output_dir=output_dir / "strict_autoresearch_portfolio_current",
        stem="strict_autoresearch_portfolio",
        markdown=f"# strict autoresearch 55/45 portfolio\n\n- leverage: `{resolved_autoresearch_leverage}x`\n",
    )

    blend_payload = _build_blend_payload(
        incumbent_payload=incumbent_payload,
        autoresearch_payload=autoresearch_payload,
        blend_weight=float(blend_weight),
    )
    blend_written = _write_payload(
        payload=blend_payload,
        output_dir=output_dir / "strict_blend_portfolio_current",
        stem="strict_grouped_blend_portfolio",
        markdown=(
            "# strict grouped blend portfolio\n\n"
            f"- incumbent weight: `{float(blend_weight):.2%}`\n"
            f"- autoresearch weight: `{1.0 - float(blend_weight):.2%}`\n"
        ),
    )

    strict_market = _market.run_group_market_regime_judgement(
        incumbent_path=incumbent_written["json"],
        autoresearch_path=autoresearch_written["json"],
        output_dir=output_dir / "strict_market_regime_judgement_current",
        horizon_days=max(1, int(horizon_days)),
        soft_rss_bytes=max(1, int(soft_rss_bytes)),
        hard_rss_bytes=max(1, int(hard_rss_bytes)),
    )
    strict_allocator = _three_way.run_three_way_market_regime_allocator(
        incumbent_path=incumbent_written["json"],
        blend_path=blend_written["json"],
        autoresearch_path=autoresearch_written["json"],
        market_judgement_path=strict_market["latest_json_path"],
        output_dir=output_dir / "strict_three_way_market_regime_allocator_current",
        soft_rss_bytes=max(1, int(soft_rss_bytes)),
        hard_rss_bytes=max(1, int(hard_rss_bytes)),
    )

    promoted_candidate = _resolve_current_promoted_candidate(decision_payload)
    strict_allocator_payload = _apply_allocator_state_leverage_to_payload(
        allocator_payload=dict(strict_allocator["payload"]),
        leverage_by_state={
            "incumbent": resolved_incumbent_leverage,
            "blend_85_15": int(leverage_by_state.get("blend_85_15") or 1),
            "autoresearch_55_45": resolved_autoresearch_leverage,
        },
    )
    strict_allocator_payload["artifact_path"] = str(strict_allocator["latest_json_path"].resolve())
    comparison = _comparison_block(
        dict(strict_allocator_payload.get("split_metrics") or {}),
        {
            "train": dict(promoted_candidate.get("train") or {}),
            "val": dict(promoted_candidate.get("val") or {}),
            "oos": dict(promoted_candidate.get("oos") or {}),
        },
    )

    payload = {
        "artifact_kind": "grouped_allocator_strict_leverage_validation",
        "generated_at": _utc_now_iso(),
        "selection_basis": "strict_candidate_level_leverage_validation_before_allocator_promotion",
        "validation_split": validation_split,
        "input_paths": {
            "incumbent_bundle": str(incumbent_bundle_path.resolve()),
            "incumbent_portfolio": str(incumbent_portfolio_path.resolve()),
            "autoresearch_portfolio": str(autoresearch_portfolio_path.resolve()),
            "leverage_tuning": str(leverage_tuning_path.resolve()),
            "decision": str(decision_path.resolve()),
        },
        "leverage_by_state": {
            "incumbent": resolved_incumbent_leverage,
            "blend_85_15": int(leverage_by_state.get("blend_85_15") or 1),
            "autoresearch_55_45": resolved_autoresearch_leverage,
        },
        "strict_artifact_paths": {
            "incumbent": str(incumbent_written["json"].resolve()),
            "autoresearch_55_45": str(autoresearch_written["json"].resolve()),
            "blend_85_15": str(blend_written["json"].resolve()),
            "market_judgement": str(strict_market["latest_json_path"].resolve()),
            "allocator": str(strict_allocator["latest_json_path"].resolve()),
        },
        "strict_group_metrics": {
            "incumbent": dict(incumbent_payload.get("portfolio_metrics") or {}),
            "autoresearch_55_45": dict(autoresearch_payload.get("portfolio_metrics") or {}),
            "blend_85_15": dict(blend_payload.get("portfolio_metrics") or {}),
            "allocator": dict(strict_allocator_payload.get("split_metrics") or {}),
        },
        "current_promoted_challenger": promoted_candidate,
        "strict_allocator": strict_allocator_payload,
        "comparison_vs_promoted_challenger": comparison,
        "notes": [
            "Strict sleeve validation reruns the selected incumbent and 55/45 sleeves through run_candidate_research with candidate-level leverage.",
            "Blend and allocator artifacts are rebuilt from those strict portfolio streams before comparing against the currently promoted challenger.",
            "Allocator daily state returns are then re-levered using leverage_by_state so proxy tuning and strict validation use the same state-level leverage semantics.",
            "If strict deltas materially degrade the challenger, treat the prior leverage-tuning result as optimistic proxy-only evidence.",
        ],
    }
    markdown = _build_markdown(payload)
    written = _write_payload(
        payload=payload,
        output_dir=output_dir,
        stem="grouped_allocator_strict_leverage_validation",
        markdown=markdown,
    )
    return {
        "payload": payload,
        "latest_json_path": written["json"],
        "latest_md_path": written["md"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-bundle-path", type=Path, default=DEFAULT_INCUMBENT_BUNDLE)
    parser.add_argument(
        "--incumbent-portfolio-path", type=Path, default=resolve_current_optimization_path()
    )
    parser.add_argument(
        "--autoresearch-portfolio-path",
        type=Path,
        default=_market._resolve_autoresearch_default_path(),
    )
    parser.add_argument("--leverage-tuning-path", type=Path, default=DEFAULT_LEVERAGE_TUNING_PATH)
    parser.add_argument("--decision-path", type=Path, default=DEFAULT_DECISION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--incumbent-leverage", type=int, default=None)
    parser.add_argument("--autoresearch-leverage", type=int, default=None)
    parser.add_argument("--blend-weight", type=float, default=DEFAULT_BLEND_WEIGHT)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--soft-rss-bytes", type=int, default=DEFAULT_SOFT_RSS_BYTES)
    parser.add_argument("--hard-rss-bytes", type=int, default=DEFAULT_HARD_RSS_BYTES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_grouped_allocator_strict_leverage_validation(
        incumbent_bundle_path=resolve_incumbent_bundle_path(args.incumbent_bundle_path),
        incumbent_portfolio_path=Path(args.incumbent_portfolio_path).resolve(),
        autoresearch_portfolio_path=Path(args.autoresearch_portfolio_path).resolve(),
        leverage_tuning_path=Path(args.leverage_tuning_path).resolve(),
        decision_path=Path(args.decision_path).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        incumbent_leverage=args.incumbent_leverage,
        autoresearch_leverage=args.autoresearch_leverage,
        blend_weight=float(args.blend_weight),
        horizon_days=max(1, int(args.horizon_days)),
        soft_rss_bytes=max(1, int(args.soft_rss_bytes)),
        hard_rss_bytes=max(1, int(args.hard_rss_bytes)),
    )
    print(report["latest_json_path"].resolve())
    print(report["latest_md_path"].resolve())


if __name__ == "__main__":
    main()
