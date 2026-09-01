from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.eval.exact_window_suite import _metrics_daily
from lumina_quant.strategy_factory.research_metrics import (
    approx_pbo,
    deflated_sharpe_ratio,
    spa_like_pvalue,
)

# Portfolio-layer honest-gate no-op defaults (performance_lever_measurement
# 2026-07-08). Every default is a strict no-op: a DSR floor of 0.0 is always
# cleared by a probability in [0, 1]; an SPA ceiling of 1.0 / a PBO ceiling of 1.0
# never rejects a value in [0, 1]; and num_trials=1 applies zero extra deflation.
_PORTFOLIO_HONEST_GATE_DSR_FLOOR_DEFAULT = 0.0
_PORTFOLIO_HONEST_GATE_SPA_CEILING_DEFAULT = 1.0
_PORTFOLIO_HONEST_GATE_PBO_CEILING_DEFAULT = 1.0

ROBUST_PROMOTION_GATES = {
    "train_total_return_min_exclusive": 0.0,
    "val_total_return_min_exclusive": 0.0,
    "train_sharpe_min": -0.10,
    "oos_total_return_delta_min_exclusive": 0.0,
    "oos_monthly_mean_min": 0.02,
    "oos_sharpe_relief_min": 0.50,
    "candidate_split_max_drawdown_max_inclusive": 0.15,
    "strict_liquidation_count_max_inclusive": 0,
    "strict_max_leverage_max_inclusive": 20.0,
    "strict_average_leverage_max_exclusive": 5.0,
}

VALIDATION_OBJECTIVE_FORMULA = (
    "(1.0 * val_sharpe) + (0.35 * val_sortino) + (0.10 * val_calmar) + "
    "(10.0 * val_total_return) - (4.0 * val_max_drawdown) - (0.75 * val_volatility)"
)

MEMORY_LEDGER_ARTIFACT_KIND = "portfolio_meta_search_memory_ledger_row"
MEMORY_LEDGER_SCHEMA_VERSION = "1.0"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except TypeError, ValueError:
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    for key in ("candidate_key", "candidate_id", "name", "label"):
        token = str(candidate.get(key) or "").strip()
        if token:
            return token
    return "candidate"


def _candidate_tokens(candidate: Mapping[str, Any]) -> str:
    metadata = dict(candidate.get("metadata") or {})
    tokens = [
        _candidate_identity(candidate),
        str(candidate.get("selection_basis") or ""),
        str(metadata.get("basis_family") or ""),
        str(metadata.get("basis_variant") or ""),
    ]
    lineage = dict(candidate.get("basis_lineage") or metadata.get("basis_lineage") or {})
    tokens.extend(
        [
            str(lineage.get("family") or ""),
            str(lineage.get("variant") or ""),
            str(lineage.get("basis_universe") or ""),
        ]
    )
    normalized = " ".join(token for token in tokens if token).lower()
    return normalized.replace("-", "_")


def infer_basis_lineage(candidate: Mapping[str, Any]) -> dict[str, str]:
    metadata = dict(candidate.get("metadata") or {})
    lineage = dict(candidate.get("basis_lineage") or metadata.get("basis_lineage") or {})

    family = str(lineage.get("family") or metadata.get("basis_family") or "").strip().lower()
    variant = str(lineage.get("variant") or metadata.get("basis_variant") or "").strip().lower()
    basis_universe = (
        str(lineage.get("basis_universe") or metadata.get("basis_universe") or "").strip().lower()
    )

    tokens = _candidate_tokens(candidate)
    if not family or not variant or not basis_universe:
        if "55_45" in tokens or "55/45" in tokens:
            family = family or "static_blend"
            variant = variant or "raw_55_45"
            basis_universe = basis_universe or "raw_basis"
        elif "80_20" in tokens or "80/20" in tokens:
            family = family or "static_blend"
            variant = variant or "derived_80_20"
            basis_universe = basis_universe or "derived_basis"
        elif (
            "incumbent_autoresearch_static_blend" in tokens
            or "current_one_shot_incumbent" in tokens
        ):
            family = family or "incumbent"
            variant = variant or "incumbent_autoresearch_static_blend"
            basis_universe = basis_universe or "shared"
        elif "soft_allocator" in tokens or "soft allocator" in tokens:
            family = family or "soft_allocator"
            variant = variant or "soft_allocator"
            basis_universe = basis_universe or "shared"
        elif "regime_switch" in tokens or "regime switch" in tokens:
            family = family or "regime_switch"
            variant = variant or "regime_switch"
            basis_universe = basis_universe or "shared"
        elif "grouped_base" in tokens or "grouped base" in tokens:
            family = family or "grouped_base"
            variant = variant or "grouped_base"
            basis_universe = basis_universe or "shared"

    if not family:
        family = _candidate_identity(candidate).strip().lower().replace("-", "_").replace(" ", "_")
    if not variant:
        variant = family
    if not basis_universe:
        basis_universe = "shared"

    return {
        "family": family,
        "variant": variant,
        "basis_universe": basis_universe,
    }


def annotate_basis_lineage(candidate: Mapping[str, Any]) -> dict[str, Any]:
    annotated = dict(candidate)
    metadata = dict(annotated.get("metadata") or {})
    lineage = infer_basis_lineage(annotated)
    annotated["basis_lineage"] = dict(lineage)
    metadata.setdefault("basis_family", lineage["family"])
    metadata.setdefault("basis_variant", lineage["variant"])
    metadata.setdefault("basis_universe", lineage["basis_universe"])
    annotated["metadata"] = metadata
    return annotated


def validate_basis_universe(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    families: dict[str, str] = {}
    conflicts: list[dict[str, str]] = []
    members: list[dict[str, str]] = []
    for row in list(rows or []):
        lineage = infer_basis_lineage(row)
        candidate_key = _candidate_identity(row)
        members.append({"candidate_key": candidate_key, **lineage})
        family = lineage["family"]
        variant = lineage["variant"]
        seen = families.get(family)
        if seen is not None and seen != variant:
            conflicts.append(
                {
                    "family": family,
                    "existing_variant": seen,
                    "conflicting_variant": variant,
                    "candidate_key": candidate_key,
                }
            )
        else:
            families[family] = variant
    return {
        "ok": not conflicts,
        "conflicts": conflicts,
        "members": members,
    }


def build_basis_search_universes(rows: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    shared: list[dict[str, Any]] = []
    raw_basis: list[dict[str, Any]] = []
    derived_basis: list[dict[str, Any]] = []
    for row in list(rows or []):
        annotated = annotate_basis_lineage(row)
        universe = str((annotated.get("basis_lineage") or {}).get("basis_universe") or "shared")
        if universe == "raw_basis":
            raw_basis.append(annotated)
        elif universe == "derived_basis":
            derived_basis.append(annotated)
        else:
            shared.append(annotated)

    universes = {
        "raw_basis": [*shared, *raw_basis],
        "derived_basis": [*shared, *derived_basis],
    }
    deduped: dict[str, list[dict[str, Any]]] = {}
    for universe_name, universe_rows in universes.items():
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in universe_rows:
            key = _candidate_identity(row)
            if key in seen:
                continue
            ordered.append(dict(row))
            seen.add(key)
        deduped[universe_name] = ordered
    return deduped


def _point_datetime(point: Mapping[str, Any]) -> datetime | None:
    raw_datetime = point.get("datetime")
    if isinstance(raw_datetime, str) and raw_datetime.strip():
        normalized = (
            raw_datetime.replace("Z", "+00:00") if raw_datetime.endswith("Z") else raw_datetime
        )
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    raw_t = point.get("t")
    try:
        numeric = float(raw_t)
    except TypeError, ValueError:
        return None
    scale = 1000.0 if abs(numeric) >= 1e12 else 1.0
    return datetime.fromtimestamp(numeric / scale, tz=UTC)


def compound_returns(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def _weighted_return_stream(
    rows: list[Mapping[str, Any]],
    split: str,
    *,
    weight_key: str = "_saved_weight",
) -> list[dict[str, Any]]:
    bucket: dict[datetime, float] = defaultdict(float)
    for row in list(rows or []):
        weight = safe_float(row.get(weight_key, row.get("weight")), 0.0)
        if weight <= 0.0:
            continue
        streams = extract_portfolio_streams(row)
        for point in list(streams.get(split) or []):
            dt = _point_datetime(point)
            if dt is None:
                continue
            bucket[dt] += weight * safe_float(point.get("v"), 0.0)
    return [
        {
            "datetime": dt.isoformat().replace("+00:00", "Z"),
            "t": float(int(dt.timestamp() * 1000.0)),
            "v": float(bucket[dt]),
        }
        for dt in sorted(bucket)
    ]


def _daily_return_stream(stream: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    daily: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        dt = _point_datetime(point)
        if dt is None:
            continue
        daily[dt.date().isoformat()].append(safe_float(point.get("v"), 0.0))
    rows: list[dict[str, Any]] = []
    for day_key in sorted(daily):
        day_dt = datetime.fromisoformat(f"{day_key}T00:00:00+00:00").astimezone(UTC)
        rows.append(
            {
                "datetime": day_dt.isoformat().replace("+00:00", "Z"),
                "t": float(int(day_dt.timestamp() * 1000.0)),
                "v": compound_returns(daily[day_key]),
            }
        )
    return rows


def monthly_returns(stream: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    monthly: dict[str, list[float]] = defaultdict(list)
    for point in list(stream or []):
        dt = _point_datetime(point)
        if dt is None:
            continue
        monthly[dt.strftime("%Y-%m")].append(safe_float(point.get("v"), 0.0))
    rows: list[dict[str, Any]] = []
    for month in sorted(monthly):
        values = monthly[month]
        rows.append(
            {
                "month": month,
                "total_return": compound_returns(values),
                "days": len(values),
            }
        )
    return rows


def mean_monthly_return(rows: list[Mapping[str, Any]]) -> float:
    values = [
        safe_float(row.get("total_return", row.get("return")), 0.0)
        for row in list(rows or [])
        if isinstance(row, Mapping)
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def extract_split_metrics(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    extracted: dict[str, dict[str, Any]] = {}
    for key in ("portfolio_metrics", "split_metrics", "metrics"):
        metric_block = payload.get(key)
        if not isinstance(metric_block, Mapping):
            continue
        for split in ("train", "val", "oos"):
            split_payload = metric_block.get(split)
            if isinstance(split_payload, Mapping):
                extracted[split] = dict(split_payload)
    for split in ("train", "val", "oos"):
        if split not in extracted and isinstance(payload.get(split), Mapping):
            extracted[split] = dict(payload.get(split) or {})
    return extracted


def extract_portfolio_streams(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    for key in ("portfolio_return_streams", "return_streams", "portfolio_daily_return_streams"):
        stream_block = payload.get(key)
        if not isinstance(stream_block, Mapping):
            continue
        return {
            split: [
                dict(item)
                for item in list(stream_block.get(split) or [])
                if isinstance(item, Mapping)
            ]
            for split in ("train", "val", "oos")
        }
    return {split: [] for split in ("train", "val", "oos")}


def extract_oos_monthly_returns(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = payload.get("oos_monthly_returns")
    if isinstance(explicit, list):
        return [dict(item) for item in explicit if isinstance(item, Mapping)]
    streams = extract_portfolio_streams(payload)
    oos_stream = list(streams.get("oos") or [])
    if not oos_stream:
        return []
    return monthly_returns(oos_stream)


def count_config_grid_trials(axes: Mapping[str, int] | Sequence[int] | None) -> int:
    """Cross-run trial accountant: total config-grid CELLS searched across runs.

    The portfolio DSR (see :func:`portfolio_honest_gate_report`) must be deflated by
    the number of ALLOCATION config-grid cells the data-PC harness searched -- NOT
    the per-candidate count -- so tuning a config grid and shipping the best cell
    cannot clear the honest gate. ``axes`` declares one entry per searched grid axis
    (e.g. ``{"band_min_hold": 6, "allocator": 3, "regime_threshold": 4,
    "order_policy": 2, "funding_window": 2}``); the number of cells is the PRODUCT of
    the axis sizes. Accepts either a mapping of ``axis_name -> option_count`` or a
    bare sequence of counts. Non-positive / empty axes are ignored. Returns at least
    ``1`` (``axes=None`` or empty -> a single-config run, no extra deflation).
    """
    if axes is None:
        return 1
    counts = list(axes.values()) if isinstance(axes, Mapping) else list(axes)
    total = 1
    for count in counts:
        try:
            n = int(count)
        except TypeError, ValueError:
            continue
        if n <= 0:
            continue
        total *= n
    return max(1, total)


def portfolio_honest_gate_report(
    net_returns: Any,
    *,
    num_trials: int = 1,
    dsr_gate_floor: float = _PORTFOLIO_HONEST_GATE_DSR_FLOOR_DEFAULT,
    spa_gate_ceiling: float = _PORTFOLIO_HONEST_GATE_SPA_CEILING_DEFAULT,
    pbo_gate_ceiling: float = _PORTFOLIO_HONEST_GATE_PBO_CEILING_DEFAULT,
    stream: str = "net",
) -> dict[str, Any]:
    """DSR / CSCV-PBO / SPA honest gate on a portfolio-layer NET return stream.

    Reuses the exact per-candidate machinery -- ``deflated_sharpe_ratio`` (the
    binding cross-run-deflated axis), ``spa_like_pvalue``, and ``approx_pbo`` (the
    single-stream fold-instability PBO the per-candidate hard gate also uses) -- but
    at the WEIGHTED-PORTFOLIO layer. The DSR is deflated by ``num_trials`` = the
    TOTAL allocation config-grid cells searched (see :func:`count_config_grid_trials`),
    so a "best-of-many" config cell is rejected once its multiple-testing benchmark
    is honestly inflated. ``passed`` is ``True`` only when DSR >= ``dsr_gate_floor``
    AND SPA p-value <= ``spa_gate_ceiling`` AND PBO <= ``pbo_gate_ceiling``. At the
    shipped no-op floors (0.0 / 1.0 / 1.0) with ``num_trials=1`` nothing is ever
    rejected.
    """
    arr = np.asarray(net_returns, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    trials = max(1, int(num_trials))
    dsr = float(deflated_sharpe_ratio(arr, num_trials=trials)) if arr.size else 0.0
    spa = float(spa_like_pvalue(arr)) if arr.size else 1.0
    pbo = float(approx_pbo(arr)) if arr.size else 1.0
    floor = float(dsr_gate_floor)
    spa_ceiling = float(spa_gate_ceiling)
    pbo_ceiling = float(pbo_gate_ceiling)
    reject_reasons: list[str] = []
    if dsr < floor:
        reject_reasons.append("dsr_below_floor")
    if spa > spa_ceiling:
        reject_reasons.append("spa_above_ceiling")
    if pbo > pbo_ceiling:
        reject_reasons.append("pbo_above_ceiling")
    return {
        "stream": str(stream),
        "n": int(arr.size),
        "num_trials": trials,
        "deflated_sharpe": dsr,
        "spa_pvalue": spa,
        "pbo": pbo,
        "dsr_gate_floor": floor,
        "spa_gate_ceiling": spa_ceiling,
        "pbo_gate_ceiling": pbo_ceiling,
        "passed": not reject_reasons,
        "reject_reasons": reject_reasons,
    }


def evaluate_weighted_portfolio(
    rows: list[Mapping[str, Any]],
    *,
    weight_key: str = "_saved_weight",
    honest_gate: bool = False,
    num_trials: int = 1,
    dsr_gate_floor: float = _PORTFOLIO_HONEST_GATE_DSR_FLOOR_DEFAULT,
    spa_gate_ceiling: float = _PORTFOLIO_HONEST_GATE_SPA_CEILING_DEFAULT,
    pbo_gate_ceiling: float = _PORTFOLIO_HONEST_GATE_PBO_CEILING_DEFAULT,
) -> dict[str, Any]:
    raw_streams = {
        split: _weighted_return_stream(rows, split, weight_key=weight_key)
        for split in ("train", "val", "oos")
    }
    daily_streams = {split: _daily_return_stream(stream) for split, stream in raw_streams.items()}

    metrics: dict[str, dict[str, Any]] = {}
    weighted_component_summaries: dict[str, dict[str, float]] = {}
    oos_net_returns = np.asarray([], dtype=float)
    for split, stream in daily_streams.items():
        returns = np.asarray([safe_float(item.get("v"), 0.0) for item in stream], dtype=float)
        if split == "oos":
            oos_net_returns = returns
        split_metrics = dict(_metrics_daily(returns))
        split_metrics["return"] = float(split_metrics.get("total_return", 0.0))
        metrics[split] = split_metrics
        weighted_component_summaries[split] = {
            "trade_count": float(
                sum(
                    safe_float((row.get(split) or {}).get("trade_count"), 0.0)
                    * safe_float(row.get(weight_key, row.get("weight")), 0.0)
                    for row in rows
                )
            ),
            "turnover": float(
                sum(
                    safe_float((row.get(split) or {}).get("turnover"), 0.0)
                    * safe_float(row.get(weight_key, row.get("weight")), 0.0)
                    for row in rows
                )
            ),
            "benchmark_corr": float(
                sum(
                    safe_float((row.get(split) or {}).get("benchmark_corr"), 0.0)
                    * safe_float(row.get(weight_key, row.get("weight")), 0.0)
                    for row in rows
                )
            ),
        }

    result: dict[str, Any] = {
        "portfolio_metrics": metrics,
        "weighted_component_summaries": weighted_component_summaries,
        "portfolio_return_streams": raw_streams,
        "portfolio_daily_return_streams": daily_streams,
        "oos_monthly_returns": monthly_returns(daily_streams["oos"]),
    }
    # Config-gated portfolio-layer honest gate on the OOS NET stream. Default OFF
    # (``honest_gate=False``, ``num_trials=1``) adds NO key -- byte-identical output.
    if honest_gate:
        result["portfolio_honest_gate"] = portfolio_honest_gate_report(
            oos_net_returns,
            num_trials=num_trials,
            dsr_gate_floor=dsr_gate_floor,
            spa_gate_ceiling=spa_gate_ceiling,
            pbo_gate_ceiling=pbo_gate_ceiling,
            stream="net",
        )
    return result


def portfolio_uplift_gate_report(
    on_rows: list[Mapping[str, Any]],
    off_rows: list[Mapping[str, Any]],
    *,
    weight_key: str = "_saved_weight",
    num_trials: int = 1,
    dsr_gate_floor: float = _PORTFOLIO_HONEST_GATE_DSR_FLOOR_DEFAULT,
    spa_gate_ceiling: float = _PORTFOLIO_HONEST_GATE_SPA_CEILING_DEFAULT,
    pbo_gate_ceiling: float = _PORTFOLIO_HONEST_GATE_PBO_CEILING_DEFAULT,
) -> dict[str, Any]:
    """Honest gate on the ON-minus-OFF UPLIFT stream (for later lever A/Bs).

    A performance lever must be proven on the INCREMENT it adds, not just on the
    absolute ON stream. This builds both portfolios' OOS NET daily streams, aligns
    them by date, and runs :func:`portfolio_honest_gate_report` on the per-day
    ``on - off`` difference -- deflated by the same cross-run ``num_trials`` config
    accountant. ``passed`` therefore answers "is the lever's marginal edge real
    after multiple-testing deflation?". Returns a gate report with ``stream`` set to
    ``"uplift"`` and the number of aligned days.
    """
    on_payload = evaluate_weighted_portfolio(on_rows, weight_key=weight_key)
    off_payload = evaluate_weighted_portfolio(off_rows, weight_key=weight_key)
    on_map = _oos_daily_return_map(on_payload)
    off_map = _oos_daily_return_map(off_payload)
    common_days = sorted(set(on_map) & set(off_map))
    uplift = np.asarray([on_map[day] - off_map[day] for day in common_days], dtype=float)
    report = portfolio_honest_gate_report(
        uplift,
        num_trials=num_trials,
        dsr_gate_floor=dsr_gate_floor,
        spa_gate_ceiling=spa_gate_ceiling,
        pbo_gate_ceiling=pbo_gate_ceiling,
        stream="uplift",
    )
    report["aligned_days"] = len(common_days)
    return report


def _sparse_fold_component_score(row: Mapping[str, Any]) -> float:
    oos = dict(row.get("oos") or {})
    train = dict(row.get("train") or {})
    active_fold_ratio = safe_float(oos.get("active_fold_ratio"), 1.0)
    inactive_fold_count = safe_float(oos.get("inactive_fold_count"), 0.0)
    failed_fold_ratio = safe_float(oos.get("failed_fold_ratio"), safe_float(oos.get("pbo"), 1.0))
    no_trade_train = (
        abs(safe_float(train.get("total_return", train.get("return")), 0.0)) <= 1e-12
        and safe_float(train.get("trade_count", train.get("trades")), 0.0) <= 0.0
    )
    return float(
        (2.0 * safe_float(oos.get("sharpe"), 0.0))
        + (12.0 * safe_float(oos.get("total_return", oos.get("return")), 0.0))
        - (2.0 * safe_float(oos.get("pbo"), 1.0))
        - (0.75 * inactive_fold_count)
        - (1.5 * failed_fold_ratio)
        - (2.0 * max(0.0, 0.75 - active_fold_ratio))
        - (999_999.0 if no_trade_train else 0.0)
    )


def _candidate_preblend_reasons(
    row: Mapping[str, Any],
    *,
    active_fold_ratio_floor: float,
    max_pbo: float,
    max_turnover: float,
) -> list[str]:
    metadata = dict(row.get("metadata") or {})
    if bool(row.get("bypass_preblend_gate")) or bool(metadata.get("bypass_preblend_gate")):
        return []
    train = dict(row.get("train") or {})
    val = dict(row.get("val") or {})
    oos = dict(row.get("oos") or {})
    reasons: list[str] = []
    if safe_float(train.get("total_return", train.get("return")), 0.0) <= 0.0:
        reasons.append("train_total_return_non_positive")
    if safe_float(val.get("total_return", val.get("return")), 0.0) <= 0.0:
        reasons.append("val_total_return_non_positive")
    if (
        abs(safe_float(train.get("total_return", train.get("return")), 0.0)) <= 1e-12
        and safe_float(train.get("trade_count", train.get("trades")), 0.0) <= 0.0
    ):
        reasons.append("train_no_trade")
    if safe_float(oos.get("active_fold_ratio"), 1.0) < active_fold_ratio_floor:
        reasons.append("active_fold_ratio_below_floor")
    if safe_float(oos.get("pbo"), 1.0) > max_pbo:
        reasons.append("pbo_above_ceiling")
    if safe_float(oos.get("turnover"), 0.0) > max_turnover:
        reasons.append("turnover_above_ceiling")
    return reasons


def _oos_daily_return_map(payload: Mapping[str, Any]) -> dict[str, float]:
    streams = extract_portfolio_streams(payload)
    oos_stream = list(streams.get("oos") or [])
    if not oos_stream:
        return {}
    daily = _daily_return_stream(oos_stream)
    result: dict[str, float] = {}
    for point in daily:
        dt = _point_datetime(point)
        if dt is None:
            continue
        result[dt.date().isoformat()] = safe_float(point.get("v"), 0.0)
    return result


def _pairwise_oos_correlation(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_map = _oos_daily_return_map(left)
    right_map = _oos_daily_return_map(right)
    common_days = sorted(set(left_map) & set(right_map))
    if len(common_days) < 2:
        return 0.0
    left_values = np.asarray([left_map[day] for day in common_days], dtype=float)
    right_values = np.asarray([right_map[day] for day in common_days], dtype=float)
    if left_values.size < 2 or right_values.size < 2:
        return 0.0
    if np.std(left_values) <= 1e-12 or np.std(right_values) <= 1e-12:
        return 0.0
    corr = np.corrcoef(left_values, right_values)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def _normalize_capped_weights(raw_scores: list[float], *, max_weight: float) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=float)
    if scores.size == 0:
        return scores
    shifted = scores - float(np.max(scores))
    weights = np.exp(np.clip(shifted, -60.0, 0.0))
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(scores.shape, 1.0 / float(scores.size), dtype=float)
    weights /= total

    cap = min(1.0, max(0.0, float(max_weight)))
    if cap <= 0.0 or cap >= 1.0 or scores.size == 1:
        return weights

    fixed = np.zeros(scores.shape, dtype=float)
    free = weights.copy()
    free_mask = np.ones(scores.shape, dtype=bool)
    while True:
        violating = free_mask & (free > cap + 1e-12)
        if not np.any(violating):
            break
        fixed[violating] = cap
        free_mask[violating] = False
        remaining = 1.0 - float(np.sum(fixed))
        if remaining <= 0.0 or not np.any(free_mask):
            break
        free_weights = weights[free_mask]
        free_total = float(np.sum(free_weights))
        if free_total <= 0.0:
            free[free_mask] = remaining / float(np.sum(free_mask))
        else:
            free[free_mask] = remaining * (free_weights / free_total)
        free[~free_mask] = fixed[~free_mask]
    result = fixed
    result[free_mask] = free[free_mask]
    total = float(np.sum(result))
    if total <= 0.0:
        return np.full(scores.shape, 1.0 / float(scores.size), dtype=float)
    return result / total


def build_sparse_fold_aware_ensemble(
    rows: list[Mapping[str, Any]],
    *,
    max_members: int = 3,
) -> dict[str, Any]:
    selected = [dict(row) for row in rows][: max(1, int(max_members))]
    if not selected:
        return {
            "artifact_kind": "sparse_fold_aware_ensemble",
            "component_count": 0,
            "components": [],
            "portfolio_payload": evaluate_weighted_portfolio([]),
        }

    raw_scores: list[float] = []
    for row in selected:
        raw_scores.append(_sparse_fold_component_score(row))

    max_score = max(raw_scores)
    weights = np.asarray(
        [math.exp(max(-60.0, min(0.0, score - max_score))) for score in raw_scores], dtype=float
    )
    if float(np.sum(weights)) <= 0.0:
        weights = np.full(len(selected), 1.0 / float(len(selected)), dtype=float)
    else:
        weights /= float(np.sum(weights))

    weighted_rows: list[dict[str, Any]] = []
    for row, weight in zip(selected, weights, strict=True):
        payload = dict(row)
        payload["_saved_weight"] = float(weight)
        weighted_rows.append(payload)

    portfolio_payload = evaluate_weighted_portfolio(weighted_rows, weight_key="_saved_weight")
    return {
        "artifact_kind": "sparse_fold_aware_ensemble",
        "component_count": len(weighted_rows),
        "components": [
            {
                "name": str(row.get("name") or ""),
                "weight": float(row["_saved_weight"]),
                "oos_pbo": safe_float((row.get("oos") or {}).get("pbo"), 1.0),
                "oos_active_fold_ratio": safe_float(
                    (row.get("oos") or {}).get("active_fold_ratio"), 0.0
                ),
            }
            for row in weighted_rows
        ],
        "portfolio_payload": portfolio_payload,
    }


def build_correlation_aware_sparse_fold_ensemble(
    rows: list[Mapping[str, Any]],
    *,
    max_members: int = 3,
    correlation_penalty: float = 2.0,
    max_weight: float = 0.80,
    active_fold_ratio_floor: float = 0.50,
    max_pbo: float = 0.75,
    max_turnover: float = 4.0,
) -> dict[str, Any]:
    candidates = [dict(row) for row in rows]
    if not candidates:
        return {
            "artifact_kind": "correlation_aware_sparse_fold_ensemble",
            "component_count": 0,
            "components": [],
            "excluded_candidates": [],
            "pairwise_oos_correlation": [],
            "portfolio_payload": evaluate_weighted_portfolio([]),
        }

    ranked: list[dict[str, Any]] = []
    for row in candidates:
        base_score = _sparse_fold_component_score(row)
        preblend_reasons = _candidate_preblend_reasons(
            row,
            active_fold_ratio_floor=active_fold_ratio_floor,
            max_pbo=max_pbo,
            max_turnover=max_turnover,
        )
        ranked.append(
            {
                "row": row,
                "name": str(row.get("name") or row.get("label") or "candidate"),
                "base_score": float(base_score),
                "preblend_reasons": preblend_reasons,
                "preblend_passed": not preblend_reasons,
            }
        )
    ranked.sort(key=lambda item: (item["preblend_passed"], item["base_score"]), reverse=True)

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for entry in ranked:
        if not entry["preblend_passed"]:
            excluded.append(
                {
                    "name": entry["name"],
                    "reason": "preblend_gate_failed",
                    "base_score": entry["base_score"],
                    "preblend_reasons": list(entry["preblend_reasons"]),
                }
            )
            continue
        remaining.append(entry)

    while remaining and len(selected) < max(1, int(max_members)):
        best_idx = -1
        best_adjusted = float("-inf")
        best_avg_abs_corr = 0.0
        for idx, entry in enumerate(remaining):
            corr_values = [
                abs(_pairwise_oos_correlation(entry["row"], existing["row"]))
                for existing in selected
            ]
            avg_abs_corr = float(sum(corr_values) / len(corr_values)) if corr_values else 0.0
            adjusted_score = float(entry["base_score"] - (correlation_penalty * avg_abs_corr))
            if adjusted_score > best_adjusted:
                best_idx = idx
                best_adjusted = adjusted_score
                best_avg_abs_corr = avg_abs_corr

        if best_idx < 0:
            break
        chosen = remaining.pop(best_idx)
        if selected and best_adjusted <= 0.0:
            excluded.append(
                {
                    "name": chosen["name"],
                    "reason": "correlation_penalty_dominated",
                    "base_score": chosen["base_score"],
                    "adjusted_score": best_adjusted,
                    "avg_abs_corr": best_avg_abs_corr,
                    "preblend_reasons": list(chosen["preblend_reasons"]),
                }
            )
            break
        selected.append(
            {
                "row": dict(chosen["row"]),
                "name": chosen["name"],
                "base_score": float(chosen["base_score"]),
                "adjusted_score": best_adjusted,
                "avg_abs_corr": best_avg_abs_corr,
                "preblend_reasons": list(chosen["preblend_reasons"]),
            }
        )

    for entry in remaining:
        excluded.append(
            {
                "name": entry["name"],
                "reason": "max_members_reached",
                "base_score": entry["base_score"],
                "preblend_reasons": list(entry["preblend_reasons"]),
            }
        )

    if not selected:
        fallback = max(ranked, key=lambda item: item["base_score"])
        excluded = [row for row in excluded if row.get("name") != fallback["name"]]
        selected = [
            {
                "row": dict(fallback["row"]),
                "name": fallback["name"],
                "base_score": float(fallback["base_score"]),
                "adjusted_score": float(fallback["base_score"]),
                "avg_abs_corr": 0.0,
                "preblend_reasons": list(fallback["preblend_reasons"]),
            }
        ]

    raw_scores = [
        float(item["adjusted_score"]) * max(0.05, 1.0 - (0.5 * float(item["avg_abs_corr"])))
        for item in selected
    ]
    weights = _normalize_capped_weights(raw_scores, max_weight=max_weight)

    weighted_rows: list[dict[str, Any]] = []
    for item, weight in zip(selected, weights, strict=True):
        payload = dict(item["row"])
        payload["_saved_weight"] = float(weight)
        weighted_rows.append(payload)

    pairwise_oos_correlation: list[dict[str, Any]] = []
    for idx, left in enumerate(selected):
        for jdx in range(idx + 1, len(selected)):
            right = selected[jdx]
            pairwise_oos_correlation.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "corr": _pairwise_oos_correlation(left["row"], right["row"]),
                }
            )

    portfolio_payload = evaluate_weighted_portfolio(weighted_rows, weight_key="_saved_weight")
    return {
        "artifact_kind": "correlation_aware_sparse_fold_ensemble",
        "component_count": len(weighted_rows),
        "components": [
            {
                "name": item["name"],
                "weight": float(row["_saved_weight"]),
                "base_score": float(item["base_score"]),
                "adjusted_score": float(item["adjusted_score"]),
                "avg_abs_oos_corr": float(item["avg_abs_corr"]),
                "preblend_reasons": list(item["preblend_reasons"]),
                "oos_pbo": safe_float((row.get("oos") or {}).get("pbo"), 1.0),
                "oos_active_fold_ratio": safe_float(
                    (row.get("oos") or {}).get("active_fold_ratio"), 0.0
                ),
            }
            for item, row in zip(selected, weighted_rows, strict=True)
        ],
        "excluded_candidates": excluded,
        "pairwise_oos_correlation": pairwise_oos_correlation,
        "portfolio_payload": portfolio_payload,
        "config": {
            "max_members": int(max_members),
            "correlation_penalty": float(correlation_penalty),
            "max_weight": float(max_weight),
            "active_fold_ratio_floor": float(active_fold_ratio_floor),
            "max_pbo": float(max_pbo),
            "max_turnover": float(max_turnover),
        },
    }


def validation_objective(metrics: Mapping[str, Any]) -> float:
    return float(
        (1.0 * safe_float(metrics.get("sharpe"), 0.0))
        + (0.35 * safe_float(metrics.get("sortino"), 0.0))
        + (0.10 * safe_float(metrics.get("calmar"), 0.0))
        + (10.0 * safe_float(metrics.get("total_return", metrics.get("return")), 0.0))
        - (4.0 * safe_float(metrics.get("max_drawdown", metrics.get("mdd")), 0.0))
        - (0.75 * safe_float(metrics.get("volatility"), 0.0))
    )


def _strict_liquidation_evidence_count(candidate_payload: Mapping[str, Any]) -> int:
    """Return the strongest liquidation-count evidence attached to a candidate payload."""
    totals: list[int] = []

    def _append_scalar(value: Any) -> None:
        numeric = int(max(0.0, safe_float(value, 0.0)))
        if numeric > 0:
            totals.append(numeric)

    def _scan(container: Mapping[str, Any]) -> None:
        for key in ("candidate_level_liquidation_count", "liquidation_count"):
            _append_scalar(container.get(key))

        counts = container.get("liquidation_counts")
        if isinstance(counts, Mapping):
            total = sum(int(max(0.0, safe_float(value, 0.0))) for value in counts.values())
            if total > 0:
                totals.append(int(total))

        validation = container.get("state_leverage_validation")
        if isinstance(validation, Mapping):
            nested_counts = validation.get("liquidation_counts")
            if isinstance(nested_counts, Mapping):
                total = sum(
                    int(max(0.0, safe_float(value, 0.0))) for value in nested_counts.values()
                )
                if total > 0:
                    totals.append(int(total))

    _scan(candidate_payload)

    strict_validation = candidate_payload.get("strict_validation")
    if isinstance(strict_validation, Mapping):
        _scan(strict_validation)

    metadata = candidate_payload.get("metadata")
    if isinstance(metadata, Mapping):
        _scan(metadata)

    return max(totals) if totals else 0


def _strict_leverage_evidence(candidate_payload: Mapping[str, Any]) -> dict[str, float | None]:
    leverage_maps: list[dict[str, float]] = []
    count_maps: list[dict[str, float]] = []

    def _collect_state_counts(summary: Any) -> None:
        if not isinstance(summary, Mapping):
            return
        for split in ("oos", "val", "train"):
            split_payload = summary.get(split)
            if not isinstance(split_payload, Mapping):
                continue
            counts = split_payload.get("counts")
            if not isinstance(counts, Mapping):
                continue
            normalized = {
                str(key): max(0.0, safe_float(value, 0.0))
                for key, value in counts.items()
                if max(0.0, safe_float(value, 0.0)) > 0.0
            }
            if normalized:
                count_maps.append(normalized)
                return

    def _scan(container: Any) -> None:
        if not isinstance(container, Mapping):
            return
        leverage_by_state = container.get("leverage_by_state")
        if isinstance(leverage_by_state, Mapping):
            normalized = {
                str(key): max(0.0, safe_float(value, 0.0))
                for key, value in leverage_by_state.items()
                if max(0.0, safe_float(value, 0.0)) > 0.0
            }
            if normalized:
                leverage_maps.append(normalized)
        _collect_state_counts(container.get("state_summary"))
        _scan(container.get("strict_allocator"))
        _scan(container.get("strict_validation"))

    _scan(candidate_payload)
    metadata = candidate_payload.get("metadata")
    if isinstance(metadata, Mapping):
        _scan(metadata)

    leverage_by_state = leverage_maps[0] if leverage_maps else {}
    max_leverage = max(leverage_by_state.values()) if leverage_by_state else None

    average_leverage: float | None = None
    if leverage_by_state and count_maps:
        counts = count_maps[0]
        total_count = sum(float(value) for value in counts.values())
        weighted_sum = sum(
            float(leverage_by_state.get(str(state), 0.0)) * float(count)
            for state, count in counts.items()
            if str(state) in leverage_by_state
        )
        if total_count > 0.0 and weighted_sum > 0.0:
            average_leverage = weighted_sum / total_count

    return {
        "max_leverage": None if max_leverage is None else float(max_leverage),
        "average_leverage": None if average_leverage is None else float(average_leverage),
    }


def promotion_gross_exposure(payload: Mapping[str, Any]) -> float:
    """Return the gross exposure (sum of ``|weight|``) of a candidate payload.

    Used to exposure-normalize the locked-OOS superiority comparison so a levered
    book cannot buy "superiority" through scale alone.  Falls back to an explicit
    gross / leverage / active-exposure scalar, then to ``1.0`` (fully invested,
    unlevered) when a payload carries no weight evidence.  The result is always
    strictly positive so it is safe to divide by.
    """
    weights = payload.get("weights")
    if isinstance(weights, list):
        gross = 0.0
        seen = False
        for row in weights:
            if not isinstance(row, Mapping):
                continue
            gross += abs(safe_float(row.get("weight"), 0.0))
            seen = True
        if seen and gross > 0.0:
            return float(gross)
    for key in ("gross_exposure", "gross_leverage", "active_exposure"):
        if key in payload:
            numeric = abs(safe_float(payload.get(key), 0.0))
            if numeric > 0.0:
                return float(numeric)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("gross_exposure", "gross_leverage"):
            if key in metadata:
                numeric = abs(safe_float(metadata.get(key), 0.0))
                if numeric > 0.0:
                    return float(numeric)
    return 1.0


def multiple_comparison_delta_floor(
    deltas: list[float],
    *,
    challenger_count: int,
    base_floor: float = 0.0,
) -> float:
    """Multiple-comparison-aware floor for the locked-OOS superiority argmax.

    With ``K = challenger_count`` challengers each ranked against the same
    incumbent on the same locked OOS split, the maximum superiority delta is
    upward biased even under the null (all challengers equal).  Scale the
    empirical dispersion of the challenger deltas by the expected maximum of
    ``K`` iid standard normals (``sqrt(2 * ln K)``) to obtain a floor that sits
    strictly above ``0`` whenever there is genuine cross-challenger spread, so
    the argmax must clear more than the ``0.0`` gate to promote.
    """
    k = max(1, int(challenger_count))
    floor = max(0.0, float(base_floor))
    if k <= 1:
        return floor
    finite: list[float] = []
    for value in deltas:
        numeric = safe_float(value, float("nan"))
        if math.isfinite(numeric):
            finite.append(numeric)
    if len(finite) >= 2:
        mean = sum(finite) / len(finite)
        variance = sum((value - mean) ** 2 for value in finite) / len(finite)
        spread = math.sqrt(variance) if variance > 0.0 else 0.0
    else:
        spread = 0.0
    gumbel_scale = math.sqrt(2.0 * math.log(float(k)))
    return max(floor, spread * gumbel_scale)


def evaluate_robustness_gates(
    candidate_payload: Mapping[str, Any],
    incumbent_payload: Mapping[str, Any],
    *,
    exposure_normalized_promotion: bool = False,
    candidate_gross_exposure: float | None = None,
    incumbent_gross_exposure: float | None = None,
) -> dict[str, Any]:
    candidate_metrics = extract_split_metrics(candidate_payload)
    incumbent_metrics = extract_split_metrics(incumbent_payload)

    candidate_train = dict(candidate_metrics.get("train") or {})
    candidate_val = dict(candidate_metrics.get("val") or {})
    candidate_oos = dict(candidate_metrics.get("oos") or {})
    incumbent_oos = dict(incumbent_metrics.get("oos") or {})

    oos_total_return_delta = safe_float(
        candidate_oos.get("total_return", candidate_oos.get("return")), 0.0
    ) - safe_float(incumbent_oos.get("total_return", incumbent_oos.get("return")), 0.0)
    oos_max_drawdown_delta = safe_float(
        candidate_oos.get("max_drawdown", candidate_oos.get("mdd")), 0.0
    ) - safe_float(incumbent_oos.get("max_drawdown", incumbent_oos.get("mdd")), 0.0)
    oos_sharpe_delta = safe_float(candidate_oos.get("sharpe"), 0.0) - safe_float(
        incumbent_oos.get("sharpe"), 0.0
    )
    candidate_gross = 1.0
    incumbent_gross = 1.0
    if exposure_normalized_promotion:
        candidate_gross = (
            candidate_gross_exposure
            if candidate_gross_exposure is not None
            else promotion_gross_exposure(candidate_payload)
        )
        incumbent_gross = (
            incumbent_gross_exposure
            if incumbent_gross_exposure is not None
            else promotion_gross_exposure(incumbent_payload)
        )
        candidate_gross = candidate_gross if candidate_gross and candidate_gross > 0.0 else 1.0
        incumbent_gross = incumbent_gross if incumbent_gross and incumbent_gross > 0.0 else 1.0
        # Exposure-normalize only the scale-dependent superiority signals (return
        # and drawdown deltas) so leverage cannot buy superiority.  Sharpe/Sortino/
        # Calmar are already scale free, and the absolute drawdown/leverage caps
        # below stay on the raw (levered) book because they guard real risk.
        oos_total_return_delta = (
            safe_float(candidate_oos.get("total_return", candidate_oos.get("return")), 0.0)
            / candidate_gross
        ) - (
            safe_float(incumbent_oos.get("total_return", incumbent_oos.get("return")), 0.0)
            / incumbent_gross
        )
        oos_max_drawdown_delta = (
            safe_float(candidate_oos.get("max_drawdown", candidate_oos.get("mdd")), 0.0)
            / candidate_gross
        ) - (
            safe_float(incumbent_oos.get("max_drawdown", incumbent_oos.get("mdd")), 0.0)
            / incumbent_gross
        )
    max_drawdown_cap = float(ROBUST_PROMOTION_GATES["candidate_split_max_drawdown_max_inclusive"])
    strict_liquidation_count = _strict_liquidation_evidence_count(candidate_payload)
    strict_leverage = _strict_leverage_evidence(candidate_payload)
    strict_max_leverage = strict_leverage["max_leverage"]
    strict_average_leverage = strict_leverage["average_leverage"]

    monthly_rows = extract_oos_monthly_returns(candidate_payload)
    oos_monthly_mean = mean_monthly_return(monthly_rows)

    checks = {
        "train_total_return": safe_float(
            candidate_train.get("total_return", candidate_train.get("return")), 0.0
        )
        > ROBUST_PROMOTION_GATES["train_total_return_min_exclusive"],
        "train_real_participation": not (
            abs(safe_float(candidate_train.get("total_return", candidate_train.get("return")), 0.0))
            <= 1e-12
            and safe_float(candidate_train.get("trade_count", candidate_train.get("trades")), 0.0)
            <= 0.0
        ),
        "val_total_return": safe_float(
            candidate_val.get("total_return", candidate_val.get("return")), 0.0
        )
        > ROBUST_PROMOTION_GATES["val_total_return_min_exclusive"],
        "train_sharpe": safe_float(candidate_train.get("sharpe"), 0.0)
        >= ROBUST_PROMOTION_GATES["train_sharpe_min"],
        "train_max_drawdown": safe_float(
            candidate_train.get("max_drawdown", candidate_train.get("mdd")), 0.0
        )
        <= max_drawdown_cap,
        "val_max_drawdown": safe_float(
            candidate_val.get("max_drawdown", candidate_val.get("mdd")), 0.0
        )
        <= max_drawdown_cap,
        "oos_max_drawdown": safe_float(
            candidate_oos.get("max_drawdown", candidate_oos.get("mdd")), 0.0
        )
        <= max_drawdown_cap,
        "oos_total_return_delta": oos_total_return_delta
        > ROBUST_PROMOTION_GATES["oos_total_return_delta_min_exclusive"],
        "oos_monthly_mean": oos_monthly_mean >= ROBUST_PROMOTION_GATES["oos_monthly_mean_min"],
        "drawdown_or_sharpe_relief": (
            safe_float(candidate_oos.get("max_drawdown", candidate_oos.get("mdd")), 0.0)
            <= safe_float(incumbent_oos.get("max_drawdown", incumbent_oos.get("mdd")), 0.0)
        )
        or (
            safe_float(candidate_oos.get("sharpe"), 0.0)
            >= safe_float(incumbent_oos.get("sharpe"), 0.0)
            + ROBUST_PROMOTION_GATES["oos_sharpe_relief_min"]
        ),
        "strict_liquidation_count": strict_liquidation_count
        <= ROBUST_PROMOTION_GATES["strict_liquidation_count_max_inclusive"],
        "strict_max_leverage": strict_max_leverage is None
        or strict_max_leverage <= ROBUST_PROMOTION_GATES["strict_max_leverage_max_inclusive"],
        "strict_average_leverage": strict_average_leverage is None
        or strict_average_leverage
        < ROBUST_PROMOTION_GATES["strict_average_leverage_max_exclusive"],
    }

    rejection_reasons: list[str] = []
    if not checks["train_total_return"]:
        rejection_reasons.append("train_total_return_non_positive")
    if not checks["train_real_participation"]:
        rejection_reasons.append("train_no_trade")
    if not checks["val_total_return"]:
        rejection_reasons.append("val_total_return_non_positive")
    if not checks["train_sharpe"]:
        rejection_reasons.append("train_sharpe_below_floor")
    if not checks["train_max_drawdown"]:
        rejection_reasons.append("train_max_drawdown_above_cap")
    if not checks["val_max_drawdown"]:
        rejection_reasons.append("val_max_drawdown_above_cap")
    if not checks["oos_max_drawdown"]:
        rejection_reasons.append("oos_max_drawdown_above_cap")
    if not checks["oos_total_return_delta"]:
        rejection_reasons.append("oos_total_return_not_above_incumbent")
    if not checks["oos_monthly_mean"]:
        rejection_reasons.append("oos_monthly_mean_below_floor")
    if not checks["drawdown_or_sharpe_relief"]:
        rejection_reasons.append("oos_drawdown_worse_without_sharpe_relief")
    if not checks["strict_liquidation_count"]:
        rejection_reasons.append("strict_liquidation_count_positive")
    if not checks["strict_max_leverage"]:
        rejection_reasons.append("strict_max_leverage_above_cap")
    if not checks["strict_average_leverage"]:
        rejection_reasons.append("strict_average_leverage_above_cap")

    result = {
        "promotable": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "robustness_gate_checks": checks,
        "oos_monthly_mean": oos_monthly_mean,
        "oos_monthly_month_count": len(monthly_rows),
        "oos_total_return_delta": oos_total_return_delta,
        "oos_max_drawdown_delta": oos_max_drawdown_delta,
        "oos_sharpe_delta": oos_sharpe_delta,
        "strict_liquidation_count": strict_liquidation_count,
        "strict_max_leverage": strict_max_leverage,
        "strict_average_leverage": strict_average_leverage,
    }
    if exposure_normalized_promotion:
        result["exposure_normalized_promotion"] = True
        result["candidate_gross_exposure"] = float(candidate_gross)
        result["incumbent_gross_exposure"] = float(incumbent_gross)
    return result


def build_memory_ledger_row(
    *,
    run_name: str,
    basis_universe: str,
    candidate_count: int,
    combination_count: int,
    budget_bytes: int,
    heavy_lock_path: str | Path,
    session_memory_lease_path: str | Path,
    status: str,
    started_at: str,
    completed_at: str,
    memory_summary_path: str | Path,
) -> dict[str, Any]:
    return {
        "artifact_kind": MEMORY_LEDGER_ARTIFACT_KIND,
        "schema_version": MEMORY_LEDGER_SCHEMA_VERSION,
        "run_name": str(run_name),
        "basis_universe": str(basis_universe),
        "candidate_count": int(candidate_count),
        "combination_count": int(combination_count),
        "budget_bytes": int(budget_bytes),
        "one_heavy_lane_only": True,
        "heavy_lock_path": str(Path(heavy_lock_path)),
        "session_memory_lease_path": str(Path(session_memory_lease_path)),
        "status": str(status),
        "started_at": str(started_at),
        "completed_at": str(completed_at),
        "memory_summary_path": str(Path(memory_summary_path)),
    }


def serialize_memory_ledger_row(row: Mapping[str, Any]) -> str:
    return json.dumps(dict(row), sort_keys=True)


def parse_memory_ledger_row(raw: str) -> dict[str, Any]:
    payload = dict(json.loads(str(raw)))
    if payload.get("artifact_kind") != MEMORY_LEDGER_ARTIFACT_KIND:
        raise ValueError("unexpected memory ledger artifact kind")
    if str(payload.get("schema_version") or "") != MEMORY_LEDGER_SCHEMA_VERSION:
        raise ValueError("unexpected memory ledger schema version")
    if not bool(payload.get("one_heavy_lane_only")):
        raise ValueError("memory ledger row must declare one-heavy-lane policy")
    return payload


__all__ = [
    "MEMORY_LEDGER_ARTIFACT_KIND",
    "MEMORY_LEDGER_SCHEMA_VERSION",
    "ROBUST_PROMOTION_GATES",
    "VALIDATION_OBJECTIVE_FORMULA",
    "annotate_basis_lineage",
    "build_basis_search_universes",
    "build_memory_ledger_row",
    "build_sparse_fold_aware_ensemble",
    "count_config_grid_trials",
    "evaluate_robustness_gates",
    "evaluate_weighted_portfolio",
    "extract_oos_monthly_returns",
    "extract_portfolio_streams",
    "extract_split_metrics",
    "infer_basis_lineage",
    "mean_monthly_return",
    "monthly_returns",
    "multiple_comparison_delta_floor",
    "parse_memory_ledger_row",
    "portfolio_honest_gate_report",
    "portfolio_uplift_gate_report",
    "promotion_gross_exposure",
    "safe_float",
    "serialize_memory_ledger_row",
    "validate_basis_universe",
    "validation_objective",
]
