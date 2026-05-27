"""Search a basis-deduped low-memory portfolio universe from saved return streams."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumina_quant.eval.exact_window_suite import _metrics_daily
from lumina_quant.portfolio_split_contract import (
    FOLLOWUP_ROOT,
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
)

DEFAULT_OUTPUT_DIR = FOLLOWUP_ROOT / "portfolio_superiority_meta_search"
DEFAULT_WEIGHT_STEP = 0.05
DEFAULT_TOP_K = 25
RAW_55_45_LINEAGE = "raw_55_45"
DERIVED_80_20_LINEAGE = "derived_80_20"
NEUTRAL_LINEAGE = "neutral"
RAW_UNIVERSE_NAME = "u1_raw_basis"
DERIVED_UNIVERSE_NAME = "u2_derived_basis"
VALIDATION_OBJECTIVE_FORMULA = (
    "(1.0 * val_sharpe) + (0.35 * val_sortino) + (0.10 * val_calmar) + "
    "(10.0 * val_total_return) - (4.0 * val_max_drawdown) - (0.75 * val_volatility)"
)
ROBUST_PROMOTION_GATES = {
    "train_total_return_gt": 0.0,
    "val_total_return_gt": 0.0,
    "train_sharpe_gte": -0.10,
    "oos_total_return_delta_gt": 0.0,
    "oos_monthly_mean_gte": 0.02,
    "oos_sharpe_relief_gte": 0.50,
}
SPARSE_COMPONENT_ACTIVE_DAY_RATIO_MAX = 0.05
SPARSE_COMPONENT_ACTIVE_DAYS_MAX = 2
SPARSE_COMPONENT_WEIGHT_CAP = 0.20


class BasisUniverseError(ValueError):
    """Raised when a search universe mixes forbidden basis lineages."""


class CandidatePayloadError(ValueError):
    """Raised when a candidate payload is missing required metrics/streams."""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_key(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("candidate_key")
        or candidate.get("name")
        or candidate.get("label")
        or candidate.get("portfolio_name")
        or "candidate"
    ).strip()


def infer_basis_lineage(candidate: dict[str, Any]) -> str:
    explicit = str(candidate.get("lineage") or candidate.get("basis_lineage") or "").strip().lower()
    if explicit in {RAW_55_45_LINEAGE, DERIVED_80_20_LINEAGE, NEUTRAL_LINEAGE}:
        return explicit

    hints: list[str] = []
    for field in (
        "candidate_key",
        "name",
        "label",
        "selection_basis",
        "source_artifact_kind",
        "portfolio_name",
    ):
        raw = candidate.get(field)
        if raw is not None:
            hints.append(str(raw))
    for note in list(candidate.get("notes") or []):
        hints.append(str(note))
    token = " ".join(hints).lower()
    if any(marker in token for marker in ("55_45", "55/45", "raw 55", "autoresearch_pair_55_45")):
        return RAW_55_45_LINEAGE
    if any(marker in token for marker in ("80_20", "80/20", "derived 80", "static_blend")):
        return DERIVED_80_20_LINEAGE
    return NEUTRAL_LINEAGE


def ensure_basis_dedupe(candidates: Sequence[dict[str, Any]]) -> None:
    lineages = {infer_basis_lineage(dict(candidate)) for candidate in candidates}
    if RAW_55_45_LINEAGE in lineages and DERIVED_80_20_LINEAGE in lineages:
        raise BasisUniverseError(
            "raw 55/45 and derived 80/20 candidates cannot coexist in the same search universe"
        )


def build_basis_search_universes(
    *,
    incumbent: dict[str, Any],
    raw_55_45: dict[str, Any],
    derived_80_20: dict[str, Any],
    soft_allocator: dict[str, Any],
    regime_switch: dict[str, Any],
    grouped_base: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    universes = {
        RAW_UNIVERSE_NAME: [
            {**copy.deepcopy(incumbent), "lineage": NEUTRAL_LINEAGE},
            {**copy.deepcopy(raw_55_45), "lineage": RAW_55_45_LINEAGE},
            {**copy.deepcopy(soft_allocator), "lineage": NEUTRAL_LINEAGE},
            {**copy.deepcopy(regime_switch), "lineage": NEUTRAL_LINEAGE},
        ],
        DERIVED_UNIVERSE_NAME: [
            {**copy.deepcopy(incumbent), "lineage": NEUTRAL_LINEAGE},
            {**copy.deepcopy(derived_80_20), "lineage": DERIVED_80_20_LINEAGE},
            {**copy.deepcopy(soft_allocator), "lineage": NEUTRAL_LINEAGE},
            {**copy.deepcopy(regime_switch), "lineage": NEUTRAL_LINEAGE},
        ],
    }
    for rows in universes.values():
        ensure_basis_dedupe(rows)
    return universes


def _normalize_stream(stream: Sequence[dict[str, Any]]) -> list[tuple[pd.Timestamp, float]]:
    normalized: list[tuple[pd.Timestamp, float]] = []
    for point in list(stream or []):
        if not isinstance(point, dict):
            continue
        raw_dt = point.get("datetime")
        raw_t = point.get("t")
        ts = pd.to_datetime(raw_dt, utc=True, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(raw_t, utc=True, errors="coerce", unit="ms")
        if pd.isna(ts) and isinstance(raw_t, (int, float)):
            ts = pd.to_datetime(float(raw_t), utc=True, errors="coerce", unit="ms")
        if pd.isna(ts):
            continue
        normalized.append((ts, _safe_float(point.get("v"), 0.0)))
    normalized.sort(key=lambda item: item[0])
    return normalized


def _aggregate_stream(stream: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[pd.Timestamp, float] = defaultdict(float)
    for ts, value in _normalize_stream(stream):
        bucket[ts] += value
    return [{"t": float(ts.timestamp() * 1000.0), "v": float(bucket[ts])} for ts in sorted(bucket)]


def _daily_aggregate_stream(stream: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket: dict[pd.Timestamp, float] = defaultdict(float)
    for ts, value in _normalize_stream(stream):
        day = ts.normalize()
        bucket[day] += value
    return [
        {"t": float(ts.timestamp() * 1000.0), "datetime": ts.isoformat(), "v": float(bucket[ts])}
        for ts in sorted(bucket)
    ]


def _extract_streams(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    for key in ("portfolio_return_streams", "combined_streams", "return_streams"):
        block = payload.get(key)
        if isinstance(block, dict) and any(
            list(block.get(split) or []) for split in ("train", "val", "oos")
        ):
            return {
                split: [
                    dict(point) for point in list(block.get(split) or []) if isinstance(point, dict)
                ]
                for split in ("train", "val", "oos")
            }

    states = list(payload.get("states") or [])
    if states:
        rebuilt: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "oos": []}
        for row in states:
            if not isinstance(row, dict):
                continue
            split = str(row.get("split_group") or "").strip().lower()
            if split not in rebuilt:
                continue
            token = str(row.get("date") or "")
            rebuilt[split].append(
                {
                    "datetime": token,
                    "t": token,
                    "v": _safe_float(row.get("return"), 0.0),
                }
            )
        if any(rebuilt.values()):
            return rebuilt

    dates = list(payload.get("dates") or [])
    daily_returns = list(payload.get("daily_returns") or [])
    if dates and len(dates) == len(daily_returns):
        rebuilt = {"train": [], "val": [], "oos": []}
        for raw_dt, raw_ret in zip(dates, daily_returns, strict=False):
            dt = pd.to_datetime(raw_dt, utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            split = "oos"
            if hasattr(dt, "date"):
                try:
                    from lumina_quant.portfolio_split_contract import split_for_date

                    split = split_for_date(dt.date())
                except Exception:
                    split = "oos"
            rebuilt[split].append(
                {
                    "datetime": dt.isoformat(),
                    "t": float(dt.timestamp() * 1000.0),
                    "v": _safe_float(raw_ret, 0.0),
                }
            )
        if any(rebuilt.values()):
            return rebuilt
    raise CandidatePayloadError(
        "candidate payload is missing portfolio_return_streams/combined_streams"
    )


def _extract_split_metrics(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    for key in ("portfolio_metrics", "metrics", "split_metrics"):
        block = payload.get(key)
        if not isinstance(block, dict):
            continue
        for split in ("train", "val", "oos"):
            split_payload = block.get(split)
            if isinstance(split_payload, dict):
                extracted[split] = dict(split_payload)
    for split in ("train", "val", "oos"):
        direct = payload.get(split)
        if split not in extracted and isinstance(direct, dict):
            extracted[split] = dict(direct)
    return extracted


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(candidate.get("payload") or {})
    artifact_path: str | None = None
    if not payload:
        raw_path = candidate.get("artifact_path")
        if raw_path is None:
            raise CandidatePayloadError(
                f"candidate {_candidate_key(candidate)!r} is missing payload/artifact_path"
            )
        artifact_path = str(Path(str(raw_path)).resolve())
        payload = dict(_load_json(Path(artifact_path)))
    else:
        raw_path = candidate.get("artifact_path")
        if raw_path is not None:
            artifact_path = str(Path(str(raw_path)).resolve())

    streams = _extract_streams(payload)
    daily_streams = {
        split: _daily_aggregate_stream(streams.get(split) or [])
        for split in ("train", "val", "oos")
    }
    metrics = _extract_split_metrics(payload)
    for split in ("train", "val", "oos"):
        if split not in metrics:
            returns = np.asarray(
                [_safe_float(point.get("v"), 0.0) for point in daily_streams[split]], dtype=float
            )
            generated = dict(_metrics_daily(returns))
            generated["return"] = float(generated.get("total_return", 0.0))
            metrics[split] = generated

    def _split_participation(stream: Sequence[dict[str, Any]]) -> dict[str, float | int]:
        total_days = len(list(stream or []))
        active_days = sum(
            1 for point in list(stream or []) if abs(_safe_float(point.get("v"), 0.0)) > 1e-12
        )
        active_day_ratio = 0.0 if total_days <= 0 else float(active_days) / float(total_days)
        return {
            "total_days": int(total_days),
            "active_days": int(active_days),
            "active_day_ratio": float(active_day_ratio),
        }

    normalized = {
        "candidate_key": _candidate_key(candidate),
        "label": str(candidate.get("label") or _candidate_key(candidate)),
        "artifact_path": artifact_path,
        "lineage": infer_basis_lineage(candidate),
        "selection_basis": str(
            candidate.get("selection_basis") or payload.get("selection_basis") or ""
        ),
        "source_artifact_kind": str(
            candidate.get("source_artifact_kind") or payload.get("artifact_kind") or ""
        ),
        "notes": [
            str(note)
            for note in list(candidate.get("notes") or payload.get("notes") or [])
            if str(note).strip()
        ],
        "portfolio_metrics": metrics,
        "portfolio_return_streams": {
            split: list(streams.get(split) or []) for split in ("train", "val", "oos")
        },
        "portfolio_daily_return_streams": daily_streams,
        "participation": {
            split: _split_participation(daily_streams[split]) for split in ("train", "val", "oos")
        },
    }
    normalized["lineage"] = infer_basis_lineage(normalized)
    return normalized


def iter_weight_grid(
    count: int, *, step: float = DEFAULT_WEIGHT_STEP
) -> Iterator[tuple[float, ...]]:
    if count <= 0:
        return iter(())
    units = round(1.0 / float(step))
    if not math.isclose(units * float(step), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"weight step must evenly divide 1.0, got {step!r}")
    total_units = int(units)

    def _walk(
        slots_left: int, remaining_units: int, prefix: list[int]
    ) -> Iterator[tuple[int, ...]]:
        if slots_left == 1:
            yield (*prefix, remaining_units)
            return
        for current in range(remaining_units + 1):
            yield from _walk(slots_left - 1, remaining_units - current, [*prefix, current])

    for combo in _walk(count, total_units, []):
        yield tuple(float(unit) * float(step) for unit in combo)


def _weighted_stream(
    candidates: Sequence[dict[str, Any]],
    weights: Sequence[float],
    split: str,
) -> list[dict[str, Any]]:
    bucket: dict[pd.Timestamp, float] = defaultdict(float)
    for candidate, weight in zip(candidates, weights, strict=True):
        if weight <= 0.0:
            continue
        for ts, value in _normalize_stream(
            list((candidate.get("portfolio_return_streams") or {}).get(split) or [])
        ):
            bucket[ts] += float(weight) * value
    return [
        {"t": float(ts.timestamp() * 1000.0), "datetime": ts.isoformat(), "v": float(bucket[ts])}
        for ts in sorted(bucket)
    ]


def _portfolio_metrics_from_stream(stream: Sequence[dict[str, Any]]) -> dict[str, float]:
    daily_stream = _daily_aggregate_stream(stream)
    returns = np.asarray([_safe_float(point.get("v"), 0.0) for point in daily_stream], dtype=float)
    metrics = dict(_metrics_daily(returns))
    metrics["return"] = float(metrics.get("total_return", 0.0))
    metrics["days"] = int(returns.size)
    return metrics


def monthly_oos_returns(stream: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    monthly: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        dt = pd.to_datetime(point.get("datetime") or point.get("t"), utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        monthly[str(dt.strftime("%Y-%m"))].append(_safe_float(point.get("v"), 0.0))
    rows: list[dict[str, Any]] = []
    for month in sorted(monthly):
        values = np.asarray(monthly[month], dtype=float)
        rows.append(
            {
                "month": month,
                "total_return": float(np.prod(1.0 + values) - 1.0),
                "days": int(values.size),
            }
        )
    return rows


def oos_monthly_mean(monthly_returns: Sequence[dict[str, Any]]) -> float:
    if not monthly_returns:
        return 0.0
    return float(
        np.mean(
            np.asarray(
                [_safe_float(row.get("total_return"), 0.0) for row in monthly_returns], dtype=float
            )
        )
    )


def validation_objective(metrics: dict[str, Any]) -> float:
    return float(
        (1.0 * _safe_float(metrics.get("sharpe"), 0.0))
        + (0.35 * _safe_float(metrics.get("sortino"), 0.0))
        + (0.10 * _safe_float(metrics.get("calmar"), 0.0))
        + (10.0 * _safe_float(metrics.get("total_return", metrics.get("return")), 0.0))
        - (4.0 * _safe_float(metrics.get("max_drawdown", metrics.get("mdd")), 0.0))
        - (0.75 * _safe_float(metrics.get("volatility"), 0.0))
    )


def robustness_gate_failures(
    *,
    candidate_metrics: dict[str, dict[str, Any]],
    incumbent_oos: dict[str, Any],
    monthly_returns: Sequence[dict[str, Any]],
) -> list[str]:
    train = dict(candidate_metrics.get("train") or {})
    val = dict(candidate_metrics.get("val") or {})
    oos = dict(candidate_metrics.get("oos") or {})
    reasons: list[str] = []
    if (
        _safe_float(train.get("total_return", train.get("return")), 0.0)
        <= ROBUST_PROMOTION_GATES["train_total_return_gt"]
    ):
        reasons.append("train_total_return<=0")
    if (
        _safe_float(val.get("total_return", val.get("return")), 0.0)
        <= ROBUST_PROMOTION_GATES["val_total_return_gt"]
    ):
        reasons.append("val_total_return<=0")
    if _safe_float(train.get("sharpe"), 0.0) < ROBUST_PROMOTION_GATES["train_sharpe_gte"]:
        reasons.append("train_sharpe<-0.10")
    incumbent_return = _safe_float(
        incumbent_oos.get("total_return", incumbent_oos.get("return")), 0.0
    )
    oos_return = _safe_float(oos.get("total_return", oos.get("return")), 0.0)
    if (oos_return - incumbent_return) <= ROBUST_PROMOTION_GATES["oos_total_return_delta_gt"]:
        reasons.append("oos_total_return_delta<=0")
    if oos_monthly_mean(monthly_returns) < ROBUST_PROMOTION_GATES["oos_monthly_mean_gte"]:
        reasons.append("oos_monthly_mean<0.02")
    incumbent_drawdown = _safe_float(
        incumbent_oos.get("max_drawdown", incumbent_oos.get("mdd")), 0.0
    )
    oos_drawdown = _safe_float(oos.get("max_drawdown", oos.get("mdd")), 0.0)
    incumbent_sharpe = _safe_float(incumbent_oos.get("sharpe"), 0.0)
    oos_sharpe = _safe_float(oos.get("sharpe"), 0.0)
    if oos_drawdown > incumbent_drawdown and oos_sharpe < (
        incumbent_sharpe + ROBUST_PROMOTION_GATES["oos_sharpe_relief_gte"]
    ):
        reasons.append("oos_drawdown_worse_without_sharpe_relief")
    return reasons


def sparse_component_gate_failures(
    *,
    candidates: Sequence[dict[str, Any]],
    weights: Sequence[float],
    active_day_ratio_max: float = SPARSE_COMPONENT_ACTIVE_DAY_RATIO_MAX,
    active_days_max: int = SPARSE_COMPONENT_ACTIVE_DAYS_MAX,
    sparse_weight_cap: float = SPARSE_COMPONENT_WEIGHT_CAP,
) -> tuple[list[str], list[dict[str, Any]]]:
    reasons: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for candidate, weight in zip(candidates, weights, strict=True):
        if float(weight) <= 0.0:
            continue
        participation = dict((candidate.get("participation") or {}).get("oos") or {})
        active_day_ratio = _safe_float(participation.get("active_day_ratio"), 0.0)
        active_days = int(participation.get("active_days") or 0)
        total_days = int(participation.get("total_days") or 0)
        is_sparse = bool(
            (total_days > 0 and active_day_ratio <= float(active_day_ratio_max))
            or (total_days > 0 and active_days <= int(active_days_max))
        )
        diagnostics.append(
            {
                "candidate_key": candidate.get("candidate_key"),
                "label": candidate.get("label"),
                "weight": float(weight),
                "oos_active_days": active_days,
                "oos_total_days": total_days,
                "oos_active_day_ratio": active_day_ratio,
                "sparse_reason": (
                    "active_days"
                    if (total_days > 0 and active_days <= int(active_days_max))
                    else ("active_day_ratio" if is_sparse else "")
                ),
                "is_sparse": bool(is_sparse),
            }
        )
        if is_sparse and float(weight) > float(sparse_weight_cap):
            reasons.append(
                f"sparse_component_weight_above_cap:{candidate.get('candidate_key')}:{float(weight):.3f}>{float(sparse_weight_cap):.3f}"
            )
    return reasons, diagnostics


def _combo_weights_record(
    candidates: Sequence[dict[str, Any]],
    weights: Sequence[float],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_key": candidate.get("candidate_key"),
            "label": candidate.get("label"),
            "lineage": candidate.get("lineage"),
            "weight": float(weight),
        }
        for candidate, weight in zip(candidates, weights, strict=True)
        if float(weight) > 0.0
    ]


def evaluate_weight_combo(
    *,
    candidates: Sequence[dict[str, Any]],
    weights: Sequence[float],
    incumbent_oos: dict[str, Any],
) -> dict[str, Any]:
    split_streams = {
        split: _weighted_stream(candidates, weights, split) for split in ("train", "val", "oos")
    }
    metrics = {
        split: _portfolio_metrics_from_stream(split_streams[split])
        for split in ("train", "val", "oos")
    }
    monthly = monthly_oos_returns(split_streams["oos"])
    reasons = robustness_gate_failures(
        candidate_metrics=metrics,
        incumbent_oos=incumbent_oos,
        monthly_returns=monthly,
    )
    sparse_reasons, participation = sparse_component_gate_failures(
        candidates=candidates,
        weights=weights,
    )
    reasons.extend(sparse_reasons)
    oos = dict(metrics.get("oos") or {})
    return {
        "weights": _combo_weights_record(candidates, weights),
        "component_participation": participation,
        "train": dict(metrics.get("train") or {}),
        "val": dict(metrics.get("val") or {}),
        "oos": oos,
        "validation_objective": validation_objective(metrics.get("val") or {}),
        "oos_monthly_returns": monthly,
        "oos_monthly_mean": oos_monthly_mean(monthly),
        "oos_total_return_delta": _safe_float(oos.get("total_return", oos.get("return")), 0.0)
        - _safe_float(incumbent_oos.get("total_return", incumbent_oos.get("return")), 0.0),
        "oos_sharpe_delta": _safe_float(oos.get("sharpe"), 0.0)
        - _safe_float(incumbent_oos.get("sharpe"), 0.0),
        "oos_max_drawdown_delta": _safe_float(oos.get("max_drawdown", oos.get("mdd")), 0.0)
        - _safe_float(incumbent_oos.get("max_drawdown", incumbent_oos.get("mdd")), 0.0),
        "promotable": not reasons,
        "rejection_reasons": reasons,
    }


def _top_rows(rows: Sequence[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    return [dict(row) for row in list(rows)[: max(1, int(top_k))]]


def memory_ledger_row(
    *,
    universe_name: str,
    combination_count: int,
    candidate_count: int,
    guard_summary: dict[str, Any],
    rss_log_path: Path,
) -> dict[str, Any]:
    return {
        "artifact_kind": "portfolio_superiority_memory_ledger",
        "generated_at": datetime.now(UTC).isoformat(),
        "universe_name": universe_name,
        "candidate_count": int(candidate_count),
        "combination_count": int(combination_count),
        "memory_policy": dict(guard_summary.get("memory_policy") or {}),
        "status": guard_summary.get("status"),
        "summary_path": str(guard_summary.get("summary_path") or ""),
        "rss_log_path": str(rss_log_path.resolve()),
        "context": dict(guard_summary.get("context") or {}),
    }


def run_meta_search(
    *,
    universe_name: str,
    candidates: Sequence[dict[str, Any]],
    incumbent_key: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    weight_step: float = DEFAULT_WEIGHT_STEP,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    normalized = [normalize_candidate(dict(candidate)) for candidate in candidates]
    ensure_basis_dedupe(normalized)
    incumbent = next(
        (
            candidate
            for candidate in normalized
            if str(candidate.get("candidate_key")) == str(incumbent_key)
        ),
        None,
    )
    if incumbent is None:
        raise CandidatePayloadError(
            f"incumbent_key={incumbent_key!r} not present in candidate universe"
        )

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = resolved_output_dir / f"{universe_name}_leaderboard_latest.json"
    rejection_path = resolved_output_dir / f"{universe_name}_rejections_latest.json"
    summary_json_path = resolved_output_dir / f"{universe_name}_summary_latest.json"
    summary_md_path = resolved_output_dir / f"{universe_name}_summary_latest.md"
    memory_ledger_path = resolved_output_dir / f"{universe_name}_memory_ledger_latest.json"

    guard = acquire_portfolio_memory_guard(
        run_name=f"portfolio_superiority_meta_{universe_name}",
        output_dir=resolved_output_dir,
        metadata={
            "universe_name": universe_name,
            "candidate_count": len(normalized),
            "weight_step": weight_step,
        },
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    results: list[dict[str, Any]] = []
    try:
        guard.checkpoint(
            "portfolio_superiority_meta_search_start", {"universe_name": universe_name}
        )
        for index, weights in enumerate(
            iter_weight_grid(len(normalized), step=weight_step), start=1
        ):
            result = evaluate_weight_combo(
                candidates=normalized,
                weights=weights,
                incumbent_oos=dict((incumbent.get("portfolio_metrics") or {}).get("oos") or {}),
            )
            result["combo_index"] = index
            results.append(result)
            if index == 1 or index % 250 == 0:
                guard.sample(
                    event="portfolio_superiority_meta_combo",
                    context={"combo_index": index, "promotable": result["promotable"]},
                )

        ranked = sorted(
            results,
            key=lambda row: (
                _safe_float(row.get("validation_objective"), 0.0),
                _safe_float(row.get("oos_total_return_delta"), 0.0),
                -_safe_float(row.get("oos_max_drawdown_delta"), 0.0),
            ),
            reverse=True,
        )
        rejections = [dict(row) for row in ranked if not bool(row.get("promotable"))]
        winner = next((dict(row) for row in ranked if bool(row.get("promotable"))), None)
        winner_status = "promoted_challenger" if winner is not None else "retained_incumbent"
        winner_payload = winner or {
            "candidate_key": incumbent.get("candidate_key"),
            "label": incumbent.get("label"),
            "rejection_reasons": ["no_promotable_candidate"],
            "promotable": False,
            "weights": [
                {
                    "candidate_key": incumbent.get("candidate_key"),
                    "label": incumbent.get("label"),
                    "weight": 1.0,
                }
            ],
            "oos": dict((incumbent.get("portfolio_metrics") or {}).get("oos") or {}),
        }
        guard_summary = guard.finalize(
            status="completed",
            context={
                "winner_status": winner_status,
                "combination_count": len(ranked),
                "candidate_count": len(normalized),
            },
        )
        ledger = memory_ledger_row(
            universe_name=universe_name,
            combination_count=len(ranked),
            candidate_count=len(normalized),
            guard_summary=guard_summary,
            rss_log_path=guard.rss_log_path,
        )
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "artifact_kind": "portfolio_superiority_meta_search",
            "schema_version": "1.0",
            "universe_name": universe_name,
            "selection_basis": "validation_objective_then_locked_oos",
            "weight_step": float(weight_step),
            "validation_objective_formula": VALIDATION_OBJECTIVE_FORMULA,
            "robust_promotion_gates": dict(ROBUST_PROMOTION_GATES),
            "sparse_component_gates": {
                "oos_active_day_ratio_max": float(SPARSE_COMPONENT_ACTIVE_DAY_RATIO_MAX),
                "oos_active_days_max": int(SPARSE_COMPONENT_ACTIVE_DAYS_MAX),
                "sparse_weight_cap": float(SPARSE_COMPONENT_WEIGHT_CAP),
            },
            "memory_policy": memory_policy_payload(
                budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
            ),
            "candidate_universe": [
                {
                    "candidate_key": candidate.get("candidate_key"),
                    "label": candidate.get("label"),
                    "lineage": candidate.get("lineage"),
                    "artifact_path": candidate.get("artifact_path"),
                    "selection_basis": candidate.get("selection_basis"),
                    "source_artifact_kind": candidate.get("source_artifact_kind"),
                }
                for candidate in normalized
            ],
            "incumbent_key": incumbent_key,
            "combination_count": len(ranked),
            "leaderboard": _top_rows(ranked, top_k=top_k),
            "rejections": _top_rows(rejections, top_k=top_k),
            "winner_status": winner_status,
            "winner": winner_payload,
            "memory_ledger_path": str(memory_ledger_path.resolve()),
        }
        leaderboard_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        rejection_path.write_text(
            json.dumps(
                {
                    "generated_at": payload["generated_at"],
                    "artifact_kind": "portfolio_superiority_rejections",
                    "schema_version": "1.0",
                    "universe_name": universe_name,
                    "rejections": rejections,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        memory_ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary_json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary_md_path.write_text(
            "\n".join(
                [
                    f"# portfolio superiority meta search ({universe_name})",
                    "",
                    "- selection_basis: `validation_objective_then_locked_oos`",
                    f"- weight_step: `{weight_step:.2f}`",
                    f"- combination_count: `{len(ranked)}`",
                    f"- winner_status: `{winner_status}`",
                    f"- winner_label: `{winner_payload.get('label') or winner_payload.get('candidate_key')}`",
                    f"- winner_promotable: `{winner_payload.get('promotable')}`",
                    f"- winner_rejection_reasons: `{json.dumps(list(winner_payload.get('rejection_reasons') or []))}`",
                    f"- memory_ledger: `{memory_ledger_path.resolve()}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "leaderboard_json_path": str(leaderboard_path.resolve()),
            "rejections_json_path": str(rejection_path.resolve()),
            "memory_ledger_path": str(memory_ledger_path.resolve()),
            "summary_json_path": str(summary_json_path.resolve()),
            "summary_md_path": str(summary_md_path.resolve()),
            "winner_status": winner_status,
        }
    except Exception as exc:
        guard.finalize(status="failed", error=str(exc))
        raise
    finally:
        guard.release()


def _load_universe_json(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("universe json must be a mapping")
    return dict(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the portfolio-superiority meta search.")
    parser.add_argument("--universe-json", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--weight-step", type=float, default=DEFAULT_WEIGHT_STEP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    universe_payload = _load_universe_json(Path(args.universe_json).resolve())
    result = run_meta_search(
        universe_name=str(universe_payload.get("universe_name") or Path(args.universe_json).stem),
        candidates=[
            dict(row)
            for row in list(universe_payload.get("candidates") or [])
            if isinstance(row, dict)
        ],
        incumbent_key=str(universe_payload.get("incumbent_key") or ""),
        output_dir=Path(args.output_dir).resolve(),
        weight_step=float(args.weight_step),
        top_k=max(1, int(args.top_k)),
    )
    print(result["summary_json_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
