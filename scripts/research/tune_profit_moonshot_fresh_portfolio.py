#!/usr/bin/env python3
"""Tune fresh-start multi-sleeve portfolios from replay survivors/candidates.

This is intentionally downstream of replay_profit_moonshot_fresh_start.py.  It
selects sleeve candidates using train/validation evidence only, combines their
stateful replay equity curves, and reports OOS as report-only evidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.portfolio.optimizer_core import safe_float as _safe_float  # noqa: E402

from lumina_quant.portfolio_split_contract import (  # noqa: E402
    PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    acquire_portfolio_memory_guard,
    memory_policy_payload,
)

FRESH_PATH = REPO_ROOT / "scripts/research/replay_profit_moonshot_fresh_start.py"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/h1_h2_calendar_allocator"
)
DEFAULT_CANDIDATE_CSV = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion"
    / "fresh_start_overhaul_replay_candidates.csv"
)
DEFAULT_CURRENT_BASE_ARTIFACT = (
    REPO_ROOT
    / "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json"
)
CURRENT_CHAMPION_TRAIN_RETURN = 0.035993
CURRENT_CHAMPION_VAL_RETURN = 0.026755
CURRENT_CHAMPION_OOS_RETURN = 0.012181
CURRENT_CHAMPION_OOS_MDD = 0.001662
CURRENT_CHAMPION_OOS_RETURN_RISK = CURRENT_CHAMPION_OOS_RETURN / CURRENT_CHAMPION_OOS_MDD
CURRENT_BASE_TRAIN_MONTHLY_RETURN = 0.020000000000000018
CURRENT_BASE_VAL_MONTHLY_RETURN = 0.09848998131232078
CURRENT_BASE_TRAIN_RETURN = 0.26820739157769213
CURRENT_BASE_VAL_RETURN = 0.19971302124703927
CURRENT_BASE_TRAIN_MDD = 0.06905953159200336
CURRENT_BASE_VAL_MDD = 0.06493479922326455
CURRENT_BASE_TRAIN_SHARPE = 1.7212556285918195
CURRENT_BASE_VAL_SHARPE = 4.09640969448007
CURRENT_BASE_TRAIN_SORTINO = 1.51511166991183
CURRENT_BASE_VAL_SORTINO = 4.885875261022447
CURRENT_BASE_TRAIN_CALMAR = 3.8842110332761934
CURRENT_BASE_VAL_CALMAR = 32.1417458698937
CURRENT_BASE_LEVERAGE = 2.3427334297703024
CURRENT_BASE_SLEEVE_COUNT = 4
CURRENT_BASE_OOS_RETURN = 0.06858164444753312
CURRENT_BASE_OOS_MDD = 0.008197728267604966
CURRENT_BASE_OOS_RETURN_RISK = CURRENT_BASE_OOS_RETURN / CURRENT_BASE_OOS_MDD
BASELINE_OOS_RETURN = CURRENT_CHAMPION_OOS_RETURN
SHADOW_OOS_MDD = 0.001778
MIN_STABLE_MONTHLY_RETURN = 0.02
MIN_BUFFERED_TRAIN_MONTHLY_RETURN = 0.0225
MIN_RAW_TRAIN_MONTHLY_RETURN = 0.01
MIN_RAW_VAL_MONTHLY_RETURN = 0.02
MAX_ACCEPTABLE_OOS_MDD = 0.25
SUCCESS_SHARPE = 2.0
SUCCESS_SORTINO = 3.0
SUCCESS_SMART_SORTINO = 3.0
SUCCESS_CALMAR = 1.0
SUCCESS_TRAIN_SHARPE = 1.5
SUCCESS_TRAIN_SORTINO = 1.5
SUCCESS_TRAIN_CALMAR = 1.0
SUCCESS_VAL_SHARPE = 3.0
SUCCESS_VAL_SORTINO = 3.0
SUCCESS_VAL_CALMAR = 3.0
TARGET_BUDGET_TRAIN_RETURN = 0.05
TARGET_BUDGET_VAL_RETURN = 0.04
MIN_TARGET_BUDGET_LEVERAGE = 0.20
MAX_TRAIN_VAL_MONTHLY_BUDGET_LEVERAGE = 6.0
MIN_INTEGER_LEVERAGE = 1
RUN_NAME = "profit_moonshot_fresh_portfolio_tuning"
LOCKBOX_POLICY = {
    "selection_label": "train_val_validation_only",
    "locked_oos_label": "locked_oos_report_only",
    "locked_oos_gate_label": "locked_oos_gate_only",
    "diagnostic_best_oos_label": "diagnostic_locked_oos_not_selection_authority",
    "diagnostic_not_promoted_label": "diagnostic_not_promoted",
    "improved_label": "train_val_ranked_locked_oos_gate_passed",
    "oos_is_report_only": True,
    "oos_is_gate_only": True,
    "current_champion_oos_return": CURRENT_CHAMPION_OOS_RETURN,
    "current_base_oos_return": CURRENT_BASE_OOS_RETURN,
    "current_base_oos_return_risk": CURRENT_BASE_OOS_RETURN_RISK,
    "minimum_stable_monthly_return": MIN_STABLE_MONTHLY_RETURN,
    "maximum_acceptable_oos_mdd": MAX_ACCEPTABLE_OOS_MDD,
}

CURRENT_BASE_GATE_KEYS = (
    "train_val_stability_beats_current_base",
    "oos_return_beats_current_base",
    "oos_return_risk_beats_current_base",
)
TRAIN_VAL_STABILITY_COMPONENTS = (
    "train_monthlyized_return",
    "validation_monthlyized_return",
    "train_sharpe",
    "validation_sharpe",
    "train_sortino",
    "validation_sortino",
    "train_calmar",
    "validation_calmar",
    "train_max_drawdown",
    "validation_max_drawdown",
    "leverage",
    "sleeve_count",
)


def _load_fresh_module() -> Any:
    spec = importlib.util.spec_from_file_location("replay_profit_moonshot_fresh_start", FRESH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {FRESH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_repo_path(value: Any) -> Path | None:
    token = str(value or "").strip()
    if not token or token.lower() in {"none", "null", "false"}:
        return None
    path = Path(token)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists() or path.suffix.lower() != ".json":
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _candidate_by_name(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for key in (
        "best_success_candidate",
        "selected_by_validation",
        "diagnostic_best_oos",
        "candidate",
    ):
        candidate = payload.get(key)
        if not isinstance(candidate, dict):
            continue
        if not name or str(candidate.get("name") or "") == name:
            return dict(candidate)
    return dict(payload) if all(key in payload for key in ("splits", "gates")) else {}


def _load_current_base_candidate(path_value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the current retained base without making it a train/val selector input."""
    path = _resolve_repo_path(path_value)
    policy: dict[str, Any] = {
        "selection_inputs": ["train", "validation"],
        "locked_oos_use": ["report", "gate"],
        "uses_locked_oos_for_selection": False,
        "train_val_stability_formula": "frozen_weighted_train_validation_score_v1",
        "current_base_train_val_stability_score": CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE,
        "train_val_stability_components": list(TRAIN_VAL_STABILITY_COMPONENTS),
        "current_base_gate_keys": list(CURRENT_BASE_GATE_KEYS),
        "current_base_artifact": str(path) if path is not None else "",
        "current_base_available": False,
    }
    if path is None:
        policy["status"] = "disabled"
        return {}, policy

    payload = _load_json_mapping(path)
    if not payload:
        policy["status"] = "missing_or_invalid"
        return {}, policy

    candidate = dict(payload)
    source_path = _resolve_repo_path(payload.get("source_artifact"))
    if source_path is not None:
        source_payload = _load_json_mapping(source_path)
        source_candidate = _candidate_by_name(source_payload, str(payload.get("name") or ""))
        if source_candidate:
            candidate = source_candidate
            candidate.setdefault("source_artifact", str(source_path))
            candidate.setdefault("candidate_artifact", str(path))
    elif any(
        isinstance(payload.get(key), dict)
        for key in ("best_success_candidate", "selected_by_validation")
    ):
        candidate = _candidate_by_name(payload, "")

    if candidate:
        policy["status"] = "loaded"
        policy["current_base_available"] = True
        policy["current_base_name"] = str(candidate.get("name") or payload.get("name") or "")
    else:
        policy["status"] = "candidate_not_found"
    return candidate, policy


def _metrics_sources(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for key in ("return_quality", "metrics"):
        value = candidate.get(key)
        if isinstance(value, dict):
            sources.append(value)
    sources.append(candidate)
    return sources


def _split_payload(candidate: dict[str, Any], split_name: str) -> dict[str, Any]:
    splits = candidate.get("splits")
    if not isinstance(splits, dict):
        return {}
    split = splits.get(split_name)
    return dict(split) if isinstance(split, dict) else {}


def _split_metric(candidate: dict[str, Any], split_name: str, metric_name: str) -> float:
    split = _split_payload(candidate, split_name)
    metrics = split.get("metrics")
    if isinstance(metrics, dict) and metric_name in metrics:
        return _safe_float(metrics.get(metric_name))
    if metric_name == "round_trips" and "round_trips" in split:
        return _safe_float(split.get("round_trips"))

    aliases = {
        ("train", "total_return"): ("train_total_return",),
        ("train", "monthlyized_return"): ("train_monthlyized_return", "train_monthly_return"),
        ("train", "max_drawdown"): ("train_max_drawdown",),
        ("train", "sharpe"): ("train_sharpe",),
        ("train", "sortino"): ("train_sortino",),
        ("train", "calmar"): ("train_calmar",),
        ("train", "round_trips"): ("train_round_trips",),
        ("val", "total_return"): ("validation_total_return", "val_total_return"),
        ("val", "monthlyized_return"): (
            "validation_monthlyized_return",
            "val_monthlyized_return",
            "validation_monthly_return",
            "val_monthly_return",
        ),
        ("val", "max_drawdown"): ("validation_max_drawdown", "val_max_drawdown"),
        ("val", "sharpe"): ("validation_sharpe", "val_sharpe"),
        ("val", "sortino"): ("validation_sortino", "val_sortino"),
        ("val", "calmar"): ("validation_calmar", "val_calmar"),
        ("val", "round_trips"): ("validation_round_trips", "val_round_trips"),
        ("oos", "total_return"): ("locked_oos_total_return", "oos_total_return"),
        ("oos", "monthlyized_return"): (
            "locked_oos_monthlyized_return",
            "oos_monthlyized_return",
            "locked_oos_monthly_return",
            "oos_monthly_return",
        ),
        ("oos", "max_drawdown"): ("locked_oos_max_drawdown", "oos_max_drawdown"),
        ("oos", "sharpe"): ("locked_oos_sharpe", "oos_sharpe"),
        ("oos", "sortino"): ("locked_oos_sortino", "oos_sortino"),
        ("oos", "calmar"): ("locked_oos_calmar", "oos_calmar"),
        ("oos", "round_trips"): ("locked_oos_round_trips", "oos_round_trips"),
    }
    for source in _metrics_sources(candidate):
        for alias in aliases.get((split_name, metric_name), ()):
            if alias in source:
                return _safe_float(source.get(alias))

    if metric_name == "monthlyized_return":
        cagr = _split_metric(candidate, split_name, "cagr")
        if cagr > -1.0:
            return float((1.0 + cagr) ** (1.0 / 12.0) - 1.0)
    return 0.0


def _candidate_leverage(candidate: dict[str, Any]) -> float:
    for source in _metrics_sources(candidate):
        if "leverage" in source:
            return _safe_float(source.get("leverage"), 1.0)
    return _safe_float(candidate.get("leverage"), 1.0)


def _candidate_sleeve_count(candidate: dict[str, Any]) -> int:
    for source in _metrics_sources(candidate):
        if "sleeve_count" in source:
            return round(_safe_float(source.get("sleeve_count"), 0.0))
    sleeves = candidate.get("sleeves")
    if isinstance(sleeves, list | tuple):
        return len(sleeves)
    return round(_safe_float(candidate.get("sleeve_count"), 0.0))


def _train_val_stability_component_payload(candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "train_monthlyized_return": _split_metric(candidate, "train", "monthlyized_return"),
        "validation_monthlyized_return": _split_metric(candidate, "val", "monthlyized_return"),
        "train_sharpe": _split_metric(candidate, "train", "sharpe"),
        "validation_sharpe": _split_metric(candidate, "val", "sharpe"),
        "train_sortino": _split_metric(candidate, "train", "sortino"),
        "validation_sortino": _split_metric(candidate, "val", "sortino"),
        "train_calmar": _split_metric(candidate, "train", "calmar"),
        "validation_calmar": _split_metric(candidate, "val", "calmar"),
        "train_max_drawdown": _split_metric(candidate, "train", "max_drawdown"),
        "validation_max_drawdown": _split_metric(candidate, "val", "max_drawdown"),
        "leverage": _candidate_leverage(candidate),
        "sleeve_count": float(_candidate_sleeve_count(candidate)),
    }


def _train_val_stability_score_from_components(
    components: dict[str, float],
    *,
    base_leverage: float = CURRENT_BASE_LEVERAGE,
    base_sleeve_count: int = CURRENT_BASE_SLEEVE_COUNT,
) -> float:
    """Frozen H2 train/validation-only score; locked-OOS is intentionally absent."""
    return (
        35.0 * min(float(components["train_monthlyized_return"]), 0.06)
        + 45.0 * min(float(components["validation_monthlyized_return"]), 0.12)
        + 0.40 * float(components["train_sharpe"])
        + 0.60 * float(components["validation_sharpe"])
        + 0.35 * float(components["train_sortino"])
        + 0.55 * float(components["validation_sortino"])
        + 0.20 * min(float(components["train_calmar"]), 20.0)
        + 0.30 * min(float(components["validation_calmar"]), 60.0)
        - 35.0 * float(components["train_max_drawdown"])
        - 45.0 * float(components["validation_max_drawdown"])
        - 0.15 * max(0.0, float(components["leverage"]) - float(base_leverage))
        - 0.25 * max(0.0, float(components["sleeve_count"]) - float(base_sleeve_count))
    )


CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE = _train_val_stability_score_from_components(
    {
        "train_monthlyized_return": CURRENT_BASE_TRAIN_MONTHLY_RETURN,
        "validation_monthlyized_return": CURRENT_BASE_VAL_MONTHLY_RETURN,
        "train_sharpe": CURRENT_BASE_TRAIN_SHARPE,
        "validation_sharpe": CURRENT_BASE_VAL_SHARPE,
        "train_sortino": CURRENT_BASE_TRAIN_SORTINO,
        "validation_sortino": CURRENT_BASE_VAL_SORTINO,
        "train_calmar": CURRENT_BASE_TRAIN_CALMAR,
        "validation_calmar": CURRENT_BASE_VAL_CALMAR,
        "train_max_drawdown": CURRENT_BASE_TRAIN_MDD,
        "validation_max_drawdown": CURRENT_BASE_VAL_MDD,
        "leverage": CURRENT_BASE_LEVERAGE,
        "sleeve_count": float(CURRENT_BASE_SLEEVE_COUNT),
    }
)


def _train_val_stability_score(
    candidate: dict[str, Any],
    *,
    base_candidate: dict[str, Any] | None = None,
) -> float:
    base_leverage = _candidate_leverage(base_candidate) if base_candidate else CURRENT_BASE_LEVERAGE
    base_sleeve_count = (
        _candidate_sleeve_count(base_candidate) if base_candidate else CURRENT_BASE_SLEEVE_COUNT
    )
    return _train_val_stability_score_from_components(
        _train_val_stability_component_payload(candidate),
        base_leverage=base_leverage,
        base_sleeve_count=base_sleeve_count,
    )


def _current_base_comparison(
    item: dict[str, Any],
    current_base_candidate: dict[str, Any],
) -> dict[str, Any]:
    item_components = _train_val_stability_component_payload(item)
    base_components = _train_val_stability_component_payload(current_base_candidate)
    item_score = _train_val_stability_score(item, base_candidate=current_base_candidate)
    base_score = _train_val_stability_score(
        current_base_candidate, base_candidate=current_base_candidate
    )
    oos_return = _split_metric(item, "oos", "total_return")
    oos_mdd = _split_metric(item, "oos", "max_drawdown")
    base_oos_return = _split_metric(current_base_candidate, "oos", "total_return")
    base_oos_mdd = _split_metric(current_base_candidate, "oos", "max_drawdown")
    oos_return_risk = _return_risk_score(oos_return, oos_mdd)
    base_oos_return_risk = _return_risk_score(base_oos_return, base_oos_mdd)
    gates = {
        "train_val_stability_beats_current_base": item_score > base_score,
        "oos_return_beats_current_base": oos_return > base_oos_return,
        "oos_return_risk_beats_current_base": oos_return_risk > base_oos_return_risk,
    }
    return {
        "current_base_name": str(current_base_candidate.get("name") or ""),
        "train_val_stability_formula": "frozen_weighted_train_validation_score_v1",
        "candidate_train_val_stability": item_components,
        "current_base_train_val_stability": base_components,
        "candidate_train_val_stability_score": item_score,
        "current_base_train_val_stability_score": base_score,
        "candidate_oos_return": oos_return,
        "current_base_oos_return": base_oos_return,
        "candidate_oos_return_risk": oos_return_risk,
        "current_base_oos_return_risk": base_oos_return_risk,
        "gates": gates,
    }


def _compact_current_base_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    return {
        "name": candidate.get("name"),
        "mode": candidate.get("mode"),
        "sleeves": list(candidate.get("sleeves") or []),
        "leverage": _safe_float(candidate.get("leverage"), 1.0),
        "source_artifact": candidate.get("source_artifact"),
        "candidate_artifact": candidate.get("candidate_artifact"),
        "train_val_stability": _train_val_stability_component_payload(candidate),
        "train_val_stability_score": _train_val_stability_score(
            candidate, base_candidate=candidate
        ),
        "locked_oos": {
            "total_return": _split_metric(candidate, "oos", "total_return"),
            "max_drawdown": _split_metric(candidate, "oos", "max_drawdown"),
            "return_risk": _return_risk_score(
                _split_metric(candidate, "oos", "total_return"),
                _split_metric(candidate, "oos", "max_drawdown"),
            ),
        },
    }


def _no_improvement_lifecycle(
    *,
    current_base_candidate: dict[str, Any],
    selected_by_validation: dict[str, Any],
    best_success_candidate: dict[str, Any],
    success_candidate_count: int,
) -> dict[str, Any]:
    if not current_base_candidate:
        status = "current_base_unavailable"
    elif success_candidate_count > 0:
        status = "improvement_found"
    else:
        status = "no_improvement_current_base_retained"
    return {
        "status": status,
        "no_improvement": bool(current_base_candidate) and success_candidate_count == 0,
        "current_base_retained": bool(current_base_candidate) and success_candidate_count == 0,
        "current_base_name": current_base_candidate.get("name") if current_base_candidate else "",
        "selected_by_train_val_name": selected_by_validation.get("name")
        if selected_by_validation
        else "",
        "best_improvement_candidate_name": (
            best_success_candidate.get("name") if best_success_candidate else ""
        ),
        "success_candidate_count": int(success_candidate_count),
        "selection_basis": "train_val_stability_then_locked_oos_current_base_gates",
    }


def _validation_score(row: dict[str, str]) -> float:
    val_ret = _safe_float(row.get("val_total_return"))
    train_ret = _safe_float(row.get("train_total_return"))
    val_sharpe = _safe_float(row.get("val_sharpe"))
    val_mdd = _safe_float(row.get("val_max_drawdown"), 1.0)
    trips = max(1.0, _safe_float(row.get("val_round_trips"), 0.0))
    return (
        val_ret * 100.0
        + train_ret * 25.0
        + val_sharpe * 0.15
        - val_mdd * 50.0
        + math.log1p(trips) * 0.01
    )


def _row_filters(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("filters") or {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _calendar_neighborhood_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    if str(row.get("family") or "").strip() != "calendar_rotation":
        return None
    filters = _row_filters(row)
    long_symbol = str(filters.get("calendar_long_symbol") or "dynamic_long").upper()
    short_symbol = str(filters.get("calendar_short_symbol") or "dynamic_short").upper()
    hold_bars = round(_safe_float(filters.get("hold_bars"), _safe_float(row.get("hold_bars"), 0.0)))
    hold_bucket = int(round(hold_bars / 24.0) * 24) if hold_bars > 0 else 0
    threshold = _safe_float(filters.get("threshold"), _safe_float(row.get("threshold"), 0.0))
    take_profit = _safe_float(
        filters.get("take_profit_pct"), _safe_float(row.get("take_profit_pct"), 0.0)
    )
    threshold_bucket = round(threshold / 0.003) if threshold > 0.0 else 0
    take_profit_bucket = round(take_profit / 0.012) if take_profit > 0.0 else 0
    return (
        "calendar_rotation",
        long_symbol,
        short_symbol,
        hold_bucket,
        threshold_bucket,
        take_profit_bucket,
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(item) for item in values)
    if len(arr) == 1:
        return arr[0]
    pos = min(max(float(q), 0.0), 1.0) * float(len(arr) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return arr[lo]
    frac = pos - float(lo)
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def _calendar_neighborhood_score(group_rows: list[dict[str, str]]) -> float:
    train_returns = [_safe_float(row.get("train_total_return")) for row in group_rows]
    val_returns = [_safe_float(row.get("val_total_return")) for row in group_rows]
    val_mdds = [_safe_float(row.get("val_max_drawdown"), 1.0) for row in group_rows]
    trips = sum(max(0.0, _safe_float(row.get("val_round_trips"))) for row in group_rows)
    train_median = _quantile(train_returns, 0.50)
    val_median = _quantile(val_returns, 0.50)
    val_lq = _quantile(val_returns, 0.25)
    val_mdd_median = _quantile(val_mdds, 0.50)
    dispersion = (
        float(np.std(np.asarray(val_returns, dtype=float))) if len(val_returns) > 1 else 0.0
    )
    return (
        val_median * 100.0
        + train_median * 25.0
        + val_lq * 20.0
        - val_mdd_median * 50.0
        - dispersion * 75.0
        + math.log1p(max(1.0, trips)) * 0.01
    )


def _append_unique(
    selected: list[dict[str, str]],
    selected_names: set[str],
    rows: list[dict[str, str]],
    *,
    limit: int,
) -> None:
    for row in rows:
        if len(selected) >= limit:
            return
        name = str(row.get("name") or "")
        if name and name not in selected_names:
            selected.append(row)
            selected_names.add(name)


def _candidate_pool_with_metadata(
    rows: list[dict[str, str]],
    *,
    top_n: int,
    family_quota: int = 0,
    calendar_neighborhood_reps: int = 0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if _safe_float(row.get("train_total_return")) > 0.0
        and _safe_float(row.get("val_total_return")) > 0.0
        and _safe_float(row.get("val_round_trips")) >= 1.0
    ]
    eligible.sort(key=_validation_score, reverse=True)
    limit = max(0, int(top_n))
    metadata: dict[str, Any] = {
        "selection_basis": "train_val_only",
        "pool_order": "validation_score_descending_after_calendar_neighborhood_reps",
        "calendar_neighborhood_reps": max(0, int(calendar_neighborhood_reps)),
        "family_quota": int(family_quota),
        "uses_locked_oos_for_selection": False,
        "calendar_neighborhoods": [],
    }
    if limit == 0:
        return [], metadata

    selected: list[dict[str, str]] = []
    selected_names: set[str] = set()
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in eligible:
        key = _calendar_neighborhood_key(row)
        if key is not None:
            groups.setdefault(key, []).append(row)

    scored_groups: list[dict[str, Any]] = []
    representatives: list[dict[str, str]] = []
    for key, group_rows in groups.items():
        representative = max(group_rows, key=_validation_score)
        representatives.append(representative)
        scored_groups.append(
            {
                "key": list(key),
                "score": float(_calendar_neighborhood_score(group_rows)),
                "size": len(group_rows),
                "representative": representative.get("name"),
                "train_median_return": _quantile(
                    [_safe_float(row.get("train_total_return")) for row in group_rows], 0.50
                ),
                "val_median_return": _quantile(
                    [_safe_float(row.get("val_total_return")) for row in group_rows], 0.50
                ),
                "val_lower_quartile_return": _quantile(
                    [_safe_float(row.get("val_total_return")) for row in group_rows], 0.25
                ),
            }
        )
    scored_groups.sort(key=lambda item: (item["score"], item["size"]), reverse=True)
    metadata["calendar_neighborhoods"] = scored_groups[:25]

    if calendar_neighborhood_reps > 0:
        representatives_by_name = {str(row.get("name") or ""): row for row in representatives}
        ordered_reps = [
            representatives_by_name[str(item.get("representative") or "")]
            for item in scored_groups[: max(0, int(calendar_neighborhood_reps))]
            if str(item.get("representative") or "") in representatives_by_name
        ]
        _append_unique(selected, selected_names, ordered_reps, limit=limit)

    if family_quota > 0:
        families = sorted(
            {row.get("family") or "unknown" for row in eligible},
            key=lambda family: max(
                (
                    _validation_score(row)
                    for row in eligible
                    if (row.get("family") or "unknown") == family
                ),
                default=float("-inf"),
            ),
            reverse=True,
        )
        for family in families:
            family_rows = [row for row in eligible if (row.get("family") or "unknown") == family]
            _append_unique(
                selected, selected_names, family_rows[: max(0, int(family_quota))], limit=limit
            )
            if len(selected) >= limit:
                metadata["selected_names"] = [row.get("name") for row in selected]
                return selected, metadata

    _append_unique(selected, selected_names, eligible, limit=limit)
    metadata["selected_names"] = [row.get("name") for row in selected]
    return selected, metadata


def _candidate_pool(
    rows: list[dict[str, str]],
    *,
    top_n: int,
    family_quota: int = 0,
    calendar_neighborhood_reps: int = 0,
) -> list[dict[str, str]]:
    pool, _metadata = _candidate_pool_with_metadata(
        rows,
        top_n=top_n,
        family_quota=family_quota,
        calendar_neighborhood_reps=calendar_neighborhood_reps,
    )
    return pool


def _rss_mib() -> float:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss or 0)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def _combine_equity(
    curves: list[list[float]],
    *,
    mode: str,
    weights: list[float] | None = None,
    leverage: float = 1.0,
) -> list[float]:
    if not curves:
        return []
    min_len = min(len(curve) for curve in curves)
    if min_len <= 0:
        return []
    returns = []
    for curve in curves:
        arr = np.asarray(curve[:min_len], dtype=float)
        returns.append(arr / 10_000.0 - 1.0)
    stacked = np.vstack(returns)
    if weights is None:
        if mode in {
            "additive_sleeves",
            "train_val_target_return_budget",
            "train_val_monthly_return_budget",
        }:
            weight_arr = np.ones(stacked.shape[0], dtype=float)
        elif mode == "equal_weight":
            weight_arr = np.full(stacked.shape[0], 1.0 / float(stacked.shape[0]), dtype=float)
        else:
            raise ValueError(f"weights required for combine mode: {mode}")
    else:
        weight_arr = np.asarray(weights, dtype=float)
    combined = float(leverage) * np.dot(weight_arr, stacked)
    return [float(10_000.0 * (1.0 + item)) for item in combined]


def _normalize_weights(raw_weights: list[float]) -> list[float]:
    clean = [max(0.0, _safe_float(item)) for item in raw_weights]
    total = sum(clean)
    if total <= 0.0:
        return [1.0 / float(len(clean)) for _ in clean] if clean else []
    return [float(item / total) for item in clean]


def _validation_return_risk_weights(
    *,
    combo_names: tuple[str, ...],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
) -> list[float]:
    raw: list[float] = []
    for name in combo_names:
        train_metrics = split_payloads[name]["train"].get("metrics") or {}
        val_metrics = split_payloads[name]["val"].get("metrics") or {}
        train_ret = max(0.0, _safe_float(train_metrics.get("total_return")))
        val_ret = max(0.0, _safe_float(val_metrics.get("total_return")))
        val_sharpe = max(0.0, _safe_float(val_metrics.get("sharpe")))
        val_mdd = max(1e-6, _safe_float(val_metrics.get("max_drawdown"), 1.0))
        raw.append((val_ret * 4.0 + train_ret + val_sharpe * 0.01) / val_mdd)
    return _normalize_weights(raw)


def _train_val_mdd_budget_weights(
    *,
    combo_names: tuple[str, ...],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
) -> list[float]:
    raw: list[float] = []
    for name in combo_names:
        train_metrics = split_payloads[name]["train"].get("metrics") or {}
        val_metrics = split_payloads[name]["val"].get("metrics") or {}
        train_ret = max(0.0, _safe_float(train_metrics.get("total_return")))
        val_ret = max(0.0, _safe_float(val_metrics.get("total_return")))
        val_sharpe = max(0.0, _safe_float(val_metrics.get("sharpe")))
        train_mdd = _safe_float(train_metrics.get("max_drawdown"), 1.0)
        val_mdd = _safe_float(val_metrics.get("max_drawdown"), 1.0)
        mdd_budget = max(1e-6, train_mdd, val_mdd)
        raw.append((val_ret * 4.0 + train_ret + val_sharpe * 0.01) / mdd_budget)
    return _normalize_weights(raw)


def _calendar_cluster_key(row: dict[str, Any] | None, name: str) -> tuple[Any, ...]:
    if not row or str(row.get("family") or "").strip() != "calendar_rotation":
        return ("non_calendar", name)
    filters = _row_filters(row)
    long_symbol = str(filters.get("calendar_long_symbol") or "dynamic_long").upper()
    short_symbol = str(filters.get("calendar_short_symbol") or "dynamic_short").upper()
    hold_bars = round(_safe_float(filters.get("hold_bars"), 0.0))
    take_profit = _safe_float(filters.get("take_profit_pct"), 0.0)
    take_profit_bucket = round(take_profit / 0.012) if take_profit > 0.0 else 0
    return ("calendar_rotation", long_symbol, short_symbol, hold_bars, take_profit_bucket)


def _curve_return_vector(curves: dict[str, list[float]]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for split_name in ("train", "val"):
        arr = np.asarray(curves.get(split_name) or [], dtype=float)
        if arr.size >= 2:
            vectors.append(np.diff(arr / 10_000.0))
    if not vectors:
        return np.asarray([], dtype=float)
    return np.concatenate(vectors)


def _curve_corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    n = min(int(lhs.size), int(rhs.size))
    if n < 2:
        return 0.0
    left = lhs[-n:]
    right = rhs[-n:]
    if float(np.std(left)) <= 0.0 or float(np.std(right)) <= 0.0:
        return 0.0
    corr = float(np.corrcoef(left, right)[0, 1])
    return corr if math.isfinite(corr) else 0.0


def _cluster_assignments(
    *,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    candidate_rows_by_name: dict[str, dict[str, Any]],
    correlation_threshold: float,
) -> dict[str, str]:
    clusters: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}
    for name in combo_names:
        key = _calendar_cluster_key(candidate_rows_by_name.get(name), name)
        vector = _curve_return_vector(split_curves.get(name, {}))
        assigned_id = ""
        for cluster in clusters:
            if cluster["key"] != key:
                continue
            if _curve_corr(vector, cluster["prototype"]) >= correlation_threshold:
                assigned_id = str(cluster["id"])
                break
        if not assigned_id:
            assigned_id = f"cluster_{len(clusters) + 1}"
            clusters.append({"id": assigned_id, "key": key, "prototype": vector})
        assignments[name] = assigned_id
    return assignments


def _cap_cluster_weights(
    raw_weights: list[float],
    *,
    cluster_ids: list[str],
    cluster_cap: float,
    sleeve_cap: float,
) -> list[float]:
    weights = _normalize_weights(raw_weights)
    if not weights:
        return []
    cluster_cap = max(0.01, min(1.0, float(cluster_cap)))
    sleeve_cap = max(0.01, min(1.0, float(sleeve_cap)))
    weights = [min(weight, sleeve_cap) for weight in weights]
    for _ in range(12):
        excess = 0.0
        for idx, weight in enumerate(list(weights)):
            if weight > sleeve_cap:
                excess += weight - sleeve_cap
                weights[idx] = sleeve_cap
        cluster_sums = {
            cluster_id: sum(
                weight
                for weight, cid in zip(weights, cluster_ids, strict=True)
                if cid == cluster_id
            )
            for cluster_id in set(cluster_ids)
        }
        for cluster_id, total in cluster_sums.items():
            if total <= cluster_cap:
                continue
            scale = cluster_cap / total
            for idx, cid in enumerate(cluster_ids):
                if cid == cluster_id:
                    old = weights[idx]
                    weights[idx] = old * scale
                    excess += old - weights[idx]
        if excess <= 1e-12:
            break
        cluster_sums = {
            cluster_id: sum(
                weight
                for weight, cid in zip(weights, cluster_ids, strict=True)
                if cid == cluster_id
            )
            for cluster_id in set(cluster_ids)
        }
        capacities = [
            max(0.0, min(sleeve_cap - weight, cluster_cap - cluster_sums[cid]))
            for weight, cid in zip(weights, cluster_ids, strict=True)
        ]
        total_capacity = sum(capacities)
        if total_capacity <= 1e-12:
            break
        for idx, capacity in enumerate(capacities):
            weights[idx] += excess * capacity / total_capacity
    total = sum(weights)
    if total > 0.0 and abs(total - 1.0) > 1e-9:
        missing = 1.0 - total
        if missing > 0.0:
            cluster_sums = {
                cluster_id: sum(
                    weight
                    for weight, cid in zip(weights, cluster_ids, strict=True)
                    if cid == cluster_id
                )
                for cluster_id in set(cluster_ids)
            }
            capacities = [
                max(0.0, min(sleeve_cap - weight, cluster_cap - cluster_sums[cid]))
                for weight, cid in zip(weights, cluster_ids, strict=True)
            ]
            total_capacity = sum(capacities)
            if total_capacity > 1e-12:
                for idx, capacity in enumerate(capacities):
                    weights[idx] += missing * capacity / total_capacity
        else:
            weights = [weight / total for weight in weights]
    return weights


def _cluster_capped_validation_weights(
    *,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
    candidate_rows_by_name: dict[str, dict[str, Any]],
    cluster_cap: float,
    sleeve_cap: float,
    correlation_threshold: float,
) -> tuple[list[float], dict[str, Any]]:
    raw_weights = _train_val_mdd_budget_weights(
        combo_names=combo_names, split_payloads=split_payloads
    )
    assignments = _cluster_assignments(
        combo_names=combo_names,
        split_curves=split_curves,
        candidate_rows_by_name=candidate_rows_by_name,
        correlation_threshold=correlation_threshold,
    )
    cluster_ids = [assignments[name] for name in combo_names]
    weights = _cap_cluster_weights(
        raw_weights, cluster_ids=cluster_ids, cluster_cap=cluster_cap, sleeve_cap=sleeve_cap
    )
    cluster_weights = {
        cluster_id: sum(
            weight for weight, cid in zip(weights, cluster_ids, strict=True) if cid == cluster_id
        )
        for cluster_id in sorted(set(cluster_ids))
    }
    diagnostics = {
        "selection_basis": "train_val_only_cluster_capped",
        "uses_locked_oos_for_selection": False,
        "cluster_cap": float(cluster_cap),
        "sleeve_cap": float(sleeve_cap),
        "correlation_threshold": float(correlation_threshold),
        "clusters": dict(assignments),
        "cluster_weights": cluster_weights,
        "raw_train_val_mdd_budget_weights": [float(item) for item in raw_weights],
    }
    return weights, diagnostics


def _train_val_target_return_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
) -> tuple[float, dict[str, Any]]:
    split_returns: dict[str, float] = {}
    for split_name in ("train", "val"):
        equity = _combine_equity(
            [split_curves[name][split_name] for name in combo_names],
            mode="additive_sleeves",
            weights=None,
            leverage=1.0,
        )
        metrics = fresh._metrics_from_equity_totals(
            equity,
            periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
        )
        split_returns[split_name] = _safe_float(metrics.get("total_return"))

    required_leverages: list[float] = []
    if split_returns["train"] > 0.0:
        required_leverages.append(TARGET_BUDGET_TRAIN_RETURN / split_returns["train"])
    if split_returns["val"] > 0.0:
        required_leverages.append(TARGET_BUDGET_VAL_RETURN / split_returns["val"])
    raw_leverage = max(required_leverages, default=1.0)
    if not math.isfinite(raw_leverage):
        raw_leverage = 1.0
    leverage = max(MIN_TARGET_BUDGET_LEVERAGE, min(1.0, float(raw_leverage)))
    return leverage, {
        "selection_basis": "train_val_target_return_budget",
        "uses_locked_oos_for_selection": False,
        "target_train_return": TARGET_BUDGET_TRAIN_RETURN,
        "target_val_return": TARGET_BUDGET_VAL_RETURN,
        "raw_train_return": split_returns["train"],
        "raw_val_return": split_returns["val"],
        "raw_required_leverage": float(raw_leverage),
        "min_leverage": MIN_TARGET_BUDGET_LEVERAGE,
    }


def _split_metrics_for_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_name: str,
    leverage: float,
) -> dict[str, Any]:
    equity = _combine_equity(
        [split_curves[name][split_name] for name in combo_names],
        mode="train_val_monthly_return_budget",
        leverage=leverage,
    )
    if not equity:
        return {}
    return fresh._metrics_from_equity_totals(
        equity,
        periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
    )


def _leverage_for_monthly_return(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_name: str,
    max_leverage: float,
) -> float:
    base_metrics = _split_metrics_for_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name=split_name,
        leverage=1.0,
    )
    if _monthlyized_return(base_metrics) >= MIN_STABLE_MONTHLY_RETURN:
        return 1.0
    raw_total_return = _safe_float(base_metrics.get("total_return"))
    raw_cagr = _safe_float(base_metrics.get("cagr"))
    if raw_total_return <= 0.0 or raw_cagr <= -1.0:
        return max_leverage
    target_cagr = (1.0 + MIN_STABLE_MONTHLY_RETURN) ** 12.0 - 1.0
    annualization = 1.0
    if raw_total_return > -1.0 and raw_cagr > -1.0:
        try:
            annualization = math.log1p(raw_cagr) / math.log1p(raw_total_return)
        except (ValueError, ZeroDivisionError):
            annualization = 1.0
    if not math.isfinite(annualization) or annualization <= 0.0:
        annualization = 1.0
    target_total_return = (1.0 + target_cagr) ** (1.0 / annualization) - 1.0
    required = target_total_return / raw_total_return
    if not math.isfinite(required):
        return max_leverage
    return max(1.0, min(max_leverage, float(required)))


def _max_train_val_mdd_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_name: str,
    max_leverage: float,
) -> float:
    base_metrics = _split_metrics_for_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name=split_name,
        leverage=1.0,
    )
    raw_mdd = _safe_float(base_metrics.get("max_drawdown"), 0.0)
    if raw_mdd <= 0.0:
        return max_leverage
    cap = MAX_ACCEPTABLE_OOS_MDD / raw_mdd
    if not math.isfinite(cap):
        return max_leverage
    return max(0.0, min(max_leverage, float(cap)))


def _split_quality_for_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    leverage: float,
) -> dict[str, dict[str, float]]:
    quality: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val"):
        metrics = _split_metrics_for_leverage(
            fresh=fresh,
            combo_names=combo_names,
            split_curves=split_curves,
            split_name=split_name,
            leverage=leverage,
        )
        quality[split_name] = {
            "monthlyized_return": _monthlyized_return(metrics),
            "total_return": _safe_float(metrics.get("total_return")),
            "max_drawdown": _safe_float(metrics.get("max_drawdown"), 1.0),
            "sharpe": _safe_float(metrics.get("sharpe")),
            "sortino": _safe_float(metrics.get("sortino")),
            "calmar": _safe_float(metrics.get("calmar")),
        }
    return quality


def _integer_leverage_grid(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    max_integer_leverage: int,
    raw_train_monthly: float,
    raw_val_monthly: float,
) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for leverage in range(MIN_INTEGER_LEVERAGE, max_integer_leverage + 1):
        quality = _split_quality_for_leverage(
            fresh=fresh,
            combo_names=combo_names,
            split_curves=split_curves,
            leverage=float(leverage),
        )
        components = {
            "train_monthlyized_return": quality["train"]["monthlyized_return"],
            "validation_monthlyized_return": quality["val"]["monthlyized_return"],
            "train_sharpe": quality["train"]["sharpe"],
            "validation_sharpe": quality["val"]["sharpe"],
            "train_sortino": quality["train"]["sortino"],
            "validation_sortino": quality["val"]["sortino"],
            "train_calmar": quality["train"]["calmar"],
            "validation_calmar": quality["val"]["calmar"],
            "train_max_drawdown": quality["train"]["max_drawdown"],
            "validation_max_drawdown": quality["val"]["max_drawdown"],
            "leverage": float(leverage),
            "sleeve_count": float(len(combo_names)),
        }
        raw_quality_pass = (
            raw_train_monthly >= MIN_RAW_TRAIN_MONTHLY_RETURN
            and raw_val_monthly >= MIN_RAW_VAL_MONTHLY_RETURN
        )
        buffered_train_pass = (
            components["train_monthlyized_return"] >= MIN_BUFFERED_TRAIN_MONTHLY_RETURN
        )
        val_floor_pass = components["validation_monthlyized_return"] >= MIN_STABLE_MONTHLY_RETURN
        grid.append(
            {
                "leverage": leverage,
                "score": _train_val_stability_score_from_components(components),
                "components": components,
                "raw_quality_pass": raw_quality_pass,
                "buffered_train_pass": buffered_train_pass,
                "val_floor_pass": val_floor_pass,
                "train_val_gate_pass": raw_quality_pass and buffered_train_pass and val_floor_pass,
            }
        )
    return grid


def _train_val_monthly_return_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
) -> tuple[float, dict[str, Any]]:
    max_leverage = MAX_TRAIN_VAL_MONTHLY_BUDGET_LEVERAGE
    raw_quality = _split_quality_for_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        leverage=1.0,
    )
    raw_train_monthly = raw_quality["train"]["monthlyized_return"]
    raw_val_monthly = raw_quality["val"]["monthlyized_return"]
    train_required = _leverage_for_monthly_return(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name="train",
        max_leverage=max_leverage,
    )
    val_required = _leverage_for_monthly_return(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name="val",
        max_leverage=max_leverage,
    )
    train_mdd_cap = _max_train_val_mdd_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name="train",
        max_leverage=max_leverage,
    )
    val_mdd_cap = _max_train_val_mdd_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_name="val",
        max_leverage=max_leverage,
    )
    raw_required = max(train_required, val_required, 1.0)
    safe_cap = min(max_leverage, train_mdd_cap, val_mdd_cap)
    max_integer_leverage = max(
        MIN_INTEGER_LEVERAGE,
        math.floor(max(MIN_INTEGER_LEVERAGE, safe_cap)),
    )
    grid = _integer_leverage_grid(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        max_integer_leverage=max_integer_leverage,
        raw_train_monthly=raw_train_monthly,
        raw_val_monthly=raw_val_monthly,
    )
    train_val_pass_grid = [item for item in grid if bool(item["train_val_gate_pass"])]
    floor_grid = [
        item
        for item in grid
        if item["leverage"] >= math.ceil(raw_required - 1e-12)
        and bool(item["buffered_train_pass"])
        and bool(item["val_floor_pass"])
    ]
    selection_pool = train_val_pass_grid or floor_grid or grid
    selected = max(
        selection_pool,
        key=lambda item: (
            _safe_float(item.get("score")),
            -int(item.get("leverage") or 0),
        ),
    )
    leverage = float(selected["leverage"])
    diagnostics = {
        "selection_basis": "train_val_monthly_return_budget",
        "uses_locked_oos_for_selection": False,
        "target_monthly_return": MIN_STABLE_MONTHLY_RETURN,
        "buffered_train_monthly_return": MIN_BUFFERED_TRAIN_MONTHLY_RETURN,
        "minimum_raw_train_monthly_return": MIN_RAW_TRAIN_MONTHLY_RETURN,
        "minimum_raw_val_monthly_return": MIN_RAW_VAL_MONTHLY_RETURN,
        "max_train_val_mdd_budget": MAX_ACCEPTABLE_OOS_MDD,
        "max_leverage": max_leverage,
        "integer_leverage_only": True,
        "integer_leverage_grid": [int(item["leverage"]) for item in grid],
        "train_required_leverage": float(train_required),
        "val_required_leverage": float(val_required),
        "train_mdd_cap_leverage": float(train_mdd_cap),
        "val_mdd_cap_leverage": float(val_mdd_cap),
        "continuous_required_leverage": float(raw_required),
        "raw_required_leverage": float(raw_required),
        "max_safe_integer_leverage": int(max_integer_leverage),
        "selected_leverage": float(leverage),
        "selected_integer_leverage": int(selected["leverage"]),
        "selected_grid_score": float(selected["score"]),
        "selected_grid_train_val_gate_pass": bool(selected["train_val_gate_pass"]),
        "raw_train_monthlyized_return": raw_train_monthly,
        "raw_val_monthlyized_return": raw_val_monthly,
        "raw_train_total_return": raw_quality["train"]["total_return"],
        "raw_val_total_return": raw_quality["val"]["total_return"],
        "leverage_grid": [
            {
                "leverage": int(item["leverage"]),
                "score": float(item["score"]),
                "train_monthlyized_return": float(item["components"]["train_monthlyized_return"]),
                "val_monthlyized_return": float(
                    item["components"]["validation_monthlyized_return"]
                ),
                "train_max_drawdown": float(item["components"]["train_max_drawdown"]),
                "val_max_drawdown": float(item["components"]["validation_max_drawdown"]),
                "train_val_gate_pass": bool(item["train_val_gate_pass"]),
            }
            for item in grid
        ],
    }
    return leverage, diagnostics


def _mode_weights_and_leverage(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
    mode: str,
    candidate_rows_by_name: dict[str, dict[str, Any]] | None = None,
    cluster_cap: float = 0.50,
    sleeve_cap: float = 0.35,
    correlation_threshold: float = 0.85,
) -> tuple[list[float] | None, float, dict[str, Any]]:
    if mode in {"additive_sleeves", "equal_weight"}:
        return None, 1.0, {}
    if mode == "train_val_target_return_budget":
        leverage, diagnostics = _train_val_target_return_leverage(
            fresh=fresh,
            combo_names=combo_names,
            split_curves=split_curves,
        )
        return None, leverage, diagnostics
    if mode == "train_val_monthly_return_budget":
        leverage, diagnostics = _train_val_monthly_return_leverage(
            fresh=fresh,
            combo_names=combo_names,
            split_curves=split_curves,
        )
        return None, leverage, diagnostics
    weights = _validation_return_risk_weights(
        combo_names=combo_names, split_payloads=split_payloads
    )
    if mode == "validation_return_risk_weight":
        return weights, 1.0, {"selection_basis": "train_val_validation_return_risk"}
    if mode == "validation_drawdown_budget":
        validation_equity = _combine_equity(
            [split_curves[name]["val"] for name in combo_names],
            mode=mode,
            weights=weights,
            leverage=1.0,
        )
        validation_metrics = fresh._metrics_from_equity_totals(
            validation_equity,
            periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
        )
        val_mdd = _safe_float(validation_metrics.get("max_drawdown"), 1.0)
        target_val_mdd = SHADOW_OOS_MDD * 0.75
        leverage = 1.0 if val_mdd <= 0.0 else target_val_mdd / val_mdd
        return (
            weights,
            max(0.20, min(6.0, float(leverage))),
            {
                "selection_basis": "train_val_validation_drawdown_budget",
                "target_val_mdd": target_val_mdd,
            },
        )
    if mode == "cluster_capped_validation_weight":
        capped_weights, diagnostics = _cluster_capped_validation_weights(
            combo_names=combo_names,
            split_curves=split_curves,
            split_payloads=split_payloads,
            candidate_rows_by_name=candidate_rows_by_name or {},
            cluster_cap=cluster_cap,
            sleeve_cap=sleeve_cap,
            correlation_threshold=correlation_threshold,
        )
        return capped_weights, 1.0, diagnostics
    raise ValueError(f"unknown combine mode: {mode}")


def _return_risk_score(total_return: Any, max_drawdown: Any) -> float:
    return _safe_float(total_return) / max(1e-9, _safe_float(max_drawdown, 1.0))


def _monthlyized_return(metrics: dict[str, Any]) -> float:
    """Return geometric monthly return from the annualized split CAGR."""
    cagr = _safe_float(metrics.get("cagr"), -1.0)
    if cagr <= -1.0:
        return -1.0
    return float((1.0 + cagr) ** (1.0 / 12.0) - 1.0)


def _smart_sortino(metrics: dict[str, Any]) -> float:
    """Sortino penalized for missing the monthly return floor and for using MDD budget.

    This local "smart sortino" is deliberately stricter than raw Sortino: a low
    return / tiny-drawdown row cannot pass only because the denominator is small.
    """
    monthly_return = _monthlyized_return(metrics)
    return_floor_factor = max(0.0, min(1.0, monthly_return / MIN_STABLE_MONTHLY_RETURN))
    mdd = max(0.0, _safe_float(metrics.get("max_drawdown"), 1.0))
    drawdown_budget_factor = max(
        0.0, 1.0 - min(mdd, MAX_ACCEPTABLE_OOS_MDD) / MAX_ACCEPTABLE_OOS_MDD
    )
    return _safe_float(metrics.get("sortino")) * return_floor_factor * drawdown_budget_factor


def _improved_candidate_from_gates(gates: dict[str, bool]) -> bool:
    required = [
        "train_positive",
        "val_positive",
        "train_return_beats_current_champion",
        "val_return_beats_current_champion",
        "train_monthly_return_gte_2pct",
        "train_monthly_return_buffer_gte_2_25pct",
        "val_monthly_return_gte_2pct",
        "raw_train_monthly_return_gte_1pct",
        "raw_val_monthly_return_gte_2pct",
        "integer_leverage",
        "train_sharpe_high",
        "train_sortino_high",
        "train_calmar_high",
        "val_sharpe_high",
        "val_sortino_high",
        "val_calmar_high",
        "oos_monthly_return_gte_2pct",
        "oos_return_beats_current_champion",
        "oos_return_risk_beats_current_champion",
        "oos_mdd_within_25pct_budget",
        "oos_sharpe_high",
        "oos_sortino_high",
        "oos_smart_sortino_high",
        "oos_calmar_high",
        "oos_trades_not_starved",
    ]
    if any(key in gates for key in CURRENT_BASE_GATE_KEYS):
        required.extend(CURRENT_BASE_GATE_KEYS)
    if any(
        key in gates
        for key in (
            "liquidation_within_tolerance",
            "liquidation_free",
            "margin_buffer_positive",
            "all_splits_liquidation_safe",
            "train_validation_liquidation_safe",
        )
    ):
        if "liquidation_within_tolerance" in gates:
            required.extend(
                [
                    "liquidation_within_tolerance",
                    "margin_buffer_positive",
                    "all_splits_liquidation_within_tolerance",
                    "train_validation_liquidation_within_tolerance",
                ]
            )
        else:
            required.extend(
                [
                    "liquidation_free",
                    "margin_buffer_positive",
                    "all_splits_liquidation_safe",
                    "train_validation_liquidation_safe",
                ]
            )
    return all(bool(gates.get(key)) for key in required)


def _with_promotion_labels(item: dict[str, Any]) -> dict[str, Any]:
    gates = {str(key): bool(value) for key, value in dict(item.get("gates") or {}).items()}
    failed_gates = [key for key, ok in gates.items() if not ok]
    improved = _improved_candidate_from_gates(gates)
    item["gates"] = gates
    item["failed_gates"] = failed_gates
    item["improved_candidate"] = improved
    item["success_candidate"] = improved
    item["diagnostic_not_promoted"] = not improved
    item["promotion_status"] = (
        "improved_success_candidate"
        if improved
        else LOCKBOX_POLICY["diagnostic_not_promoted_label"]
    )
    item["locked_oos_policy"] = {
        "selection_basis": "train_val_only",
        "locked_oos_label": LOCKBOX_POLICY["locked_oos_label"],
        "oos_is_report_only": True,
        "oos_is_gate_only": True,
        "uses_locked_oos_for_selection": False,
    }
    return item


def _combo_metrics(
    *,
    fresh: Any,
    combo_names: tuple[str, ...],
    split_curves: dict[str, dict[str, list[float]]],
    split_payloads: dict[str, dict[str, dict[str, Any]]],
    mode: str,
    candidate_rows_by_name: dict[str, dict[str, Any]] | None = None,
    current_base_candidate: dict[str, Any] | None = None,
    cluster_cap: float = 0.50,
    sleeve_cap: float = 0.35,
    correlation_threshold: float = 0.85,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": f"fresh_portfolio_{mode}_" + "__".join(combo_names),
        "mode": mode,
        "sleeves": list(combo_names),
        "sleeve_count": len(combo_names),
        "splits": {},
    }
    weights, leverage, allocator_diagnostics = _mode_weights_and_leverage(
        fresh=fresh,
        combo_names=combo_names,
        split_curves=split_curves,
        split_payloads=split_payloads,
        mode=mode,
        candidate_rows_by_name=candidate_rows_by_name,
        cluster_cap=cluster_cap,
        sleeve_cap=sleeve_cap,
        correlation_threshold=correlation_threshold,
    )
    out["weights"] = [float(item) for item in weights] if weights is not None else []
    out["leverage"] = float(leverage)
    if allocator_diagnostics:
        out["allocator_diagnostics"] = allocator_diagnostics
    for split_name in ("train", "val", "oos"):
        curves = [split_curves[name][split_name] for name in combo_names]
        equity = _combine_equity(curves, mode=mode, weights=weights, leverage=leverage)
        metrics = fresh._metrics_from_equity_totals(
            equity,
            periods=int(getattr(fresh, "HOURLY_PERIODS_PER_YEAR", 365 * 24)),
        )
        round_trips = sum(
            int(split_payloads[name][split_name].get("round_trips") or 0) for name in combo_names
        )
        fills = sum(int(split_payloads[name][split_name].get("fills") or 0) for name in combo_names)
        out["splits"][split_name] = {
            "metrics": metrics,
            "round_trips": round_trips,
            "fills": fills,
            "final_equity": float(equity[-1]) if equity else 10_000.0,
        }
    train = out["splits"]["train"]["metrics"]
    val = out["splits"]["val"]["metrics"]
    oos = out["splits"]["oos"]["metrics"]
    train_return = _safe_float(train.get("total_return"))
    val_return = _safe_float(val.get("total_return"))
    oos_return = _safe_float(oos.get("total_return"))
    oos_mdd = _safe_float(oos.get("max_drawdown"), 1.0)
    train_monthly_return = _monthlyized_return(train)
    val_monthly_return = _monthlyized_return(val)
    oos_monthly_return = _monthlyized_return(oos)
    oos_smart_sortino = _smart_sortino(oos)
    raw_train_monthly_return = _safe_float(
        allocator_diagnostics.get("raw_train_monthlyized_return"),
        train_monthly_return if leverage == 1.0 else -1.0,
    )
    raw_val_monthly_return = _safe_float(
        allocator_diagnostics.get("raw_val_monthlyized_return"),
        val_monthly_return if leverage == 1.0 else -1.0,
    )
    raw_train_total_return = _safe_float(
        allocator_diagnostics.get("raw_train_total_return"),
        train_return if leverage == 1.0 else -1.0,
    )
    raw_val_total_return = _safe_float(
        allocator_diagnostics.get("raw_val_total_return"),
        val_return if leverage == 1.0 else -1.0,
    )
    integer_leverage = math.isclose(
        float(leverage), float(round(leverage)), rel_tol=0.0, abs_tol=1e-9
    )
    gates = {
        "train_positive": train_return > 0.0,
        "val_positive": val_return > 0.0,
        "train_return_beats_current_champion": train_return > CURRENT_CHAMPION_TRAIN_RETURN,
        "val_return_beats_current_champion": val_return > CURRENT_CHAMPION_VAL_RETURN,
        "train_monthly_return_gte_2pct": train_monthly_return >= MIN_STABLE_MONTHLY_RETURN,
        "train_monthly_return_buffer_gte_2_25pct": (
            train_monthly_return >= MIN_BUFFERED_TRAIN_MONTHLY_RETURN
        ),
        "val_monthly_return_gte_2pct": val_monthly_return >= MIN_STABLE_MONTHLY_RETURN,
        "raw_train_monthly_return_gte_1pct": raw_train_monthly_return
        >= MIN_RAW_TRAIN_MONTHLY_RETURN,
        "raw_val_monthly_return_gte_2pct": raw_val_monthly_return >= MIN_RAW_VAL_MONTHLY_RETURN,
        "integer_leverage": integer_leverage,
        "train_sharpe_high": _safe_float(train.get("sharpe")) >= SUCCESS_TRAIN_SHARPE,
        "train_sortino_high": _safe_float(train.get("sortino")) >= SUCCESS_TRAIN_SORTINO,
        "train_calmar_high": _safe_float(train.get("calmar")) >= SUCCESS_TRAIN_CALMAR,
        "val_sharpe_high": _safe_float(val.get("sharpe")) >= SUCCESS_VAL_SHARPE,
        "val_sortino_high": _safe_float(val.get("sortino")) >= SUCCESS_VAL_SORTINO,
        "val_calmar_high": _safe_float(val.get("calmar")) >= SUCCESS_VAL_CALMAR,
        "oos_monthly_return_gte_2pct": oos_monthly_return >= MIN_STABLE_MONTHLY_RETURN,
        "oos_return_beats_current_champion": oos_return > CURRENT_CHAMPION_OOS_RETURN,
        "oos_return_risk_beats_current_champion": (
            _return_risk_score(oos_return, oos_mdd) > CURRENT_CHAMPION_OOS_RETURN_RISK
        ),
        "oos_mdd_within_25pct_budget": oos_mdd <= MAX_ACCEPTABLE_OOS_MDD,
        "oos_sharpe_high": _safe_float(oos.get("sharpe")) >= SUCCESS_SHARPE,
        "oos_sortino_high": _safe_float(oos.get("sortino")) >= SUCCESS_SORTINO,
        "oos_smart_sortino_high": oos_smart_sortino >= SUCCESS_SMART_SORTINO,
        "oos_calmar_high": _safe_float(oos.get("calmar")) >= SUCCESS_CALMAR,
        "oos_trades_not_starved": int(out["splits"]["oos"].get("round_trips") or 0) >= 5,
    }
    train_val_stability = _train_val_stability_component_payload(out)
    train_val_stability_score = _train_val_stability_score(
        out,
        base_candidate=current_base_candidate,
    )
    out["return_quality"] = {
        "train_monthlyized_return": train_monthly_return,
        "val_monthlyized_return": val_monthly_return,
        "oos_monthlyized_return": oos_monthly_return,
        "raw_train_monthlyized_return": raw_train_monthly_return,
        "raw_val_monthlyized_return": raw_val_monthly_return,
        "raw_train_total_return": raw_train_total_return,
        "raw_val_total_return": raw_val_total_return,
        "selected_integer_leverage": round(float(leverage)) if integer_leverage else None,
        "continuous_required_leverage": allocator_diagnostics.get("continuous_required_leverage"),
        "oos_smart_sortino": oos_smart_sortino,
        "minimum_stable_monthly_return": MIN_STABLE_MONTHLY_RETURN,
        "minimum_buffered_train_monthly_return": MIN_BUFFERED_TRAIN_MONTHLY_RETURN,
        "minimum_raw_train_monthly_return": MIN_RAW_TRAIN_MONTHLY_RETURN,
        "minimum_raw_val_monthly_return": MIN_RAW_VAL_MONTHLY_RETURN,
        "maximum_acceptable_oos_mdd": MAX_ACCEPTABLE_OOS_MDD,
        "minimum_oos_sharpe": SUCCESS_SHARPE,
        "minimum_oos_sortino": SUCCESS_SORTINO,
        "minimum_oos_smart_sortino": SUCCESS_SMART_SORTINO,
        "minimum_oos_calmar": SUCCESS_CALMAR,
        "minimum_train_sharpe": SUCCESS_TRAIN_SHARPE,
        "minimum_train_sortino": SUCCESS_TRAIN_SORTINO,
        "minimum_train_calmar": SUCCESS_TRAIN_CALMAR,
        "minimum_val_sharpe": SUCCESS_VAL_SHARPE,
        "minimum_val_sortino": SUCCESS_VAL_SORTINO,
        "minimum_val_calmar": SUCCESS_VAL_CALMAR,
        "train_val_stability_score": train_val_stability_score,
        "current_base_train_val_stability_score": CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE,
        "current_base_oos_return": CURRENT_BASE_OOS_RETURN,
        "current_base_oos_return_risk": CURRENT_BASE_OOS_RETURN_RISK,
    }
    out["train_val_stability"] = train_val_stability
    out["train_val_stability_score"] = train_val_stability_score
    out["validation_score"] = (
        _safe_float(val.get("total_return")) * 100.0
        + _safe_float(train.get("total_return")) * 25.0
        + _safe_float(val.get("sharpe")) * 0.15
        - _safe_float(val.get("max_drawdown"), 1.0) * 50.0
        + math.log1p(max(1, int(out["splits"]["val"].get("round_trips") or 0))) * 0.01
    )
    if current_base_candidate:
        current_base_comparison = _current_base_comparison(out, current_base_candidate)
        out["current_base_comparison"] = current_base_comparison
        gates.update(current_base_comparison["gates"])
    out["gates"] = gates
    return _with_promotion_labels(out)


def _flatten_row(item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": item["name"],
        "mode": item["mode"],
        "sleeve_count": item["sleeve_count"],
        "sleeves": ",".join(item["sleeves"]),
        "weights": ",".join(f"{float(weight):.6f}" for weight in item.get("weights") or []),
        "leverage": _safe_float(item.get("leverage"), 1.0),
        "validation_score": item["validation_score"],
        "train_val_stability_score": _train_val_stability_sort_key(item),
        "success_candidate": item["success_candidate"],
        "improved_candidate": item.get("improved_candidate", False),
        "diagnostic_not_promoted": item.get("diagnostic_not_promoted", False),
        "promotion_status": item.get("promotion_status", ""),
        "failed_gates": ",".join(item.get("failed_gates") or []),
    }
    if item.get("return_quality"):
        quality = dict(item["return_quality"])
        for key in (
            "train_monthlyized_return",
            "val_monthlyized_return",
            "oos_monthlyized_return",
            "raw_train_monthlyized_return",
            "raw_val_monthlyized_return",
            "raw_train_total_return",
            "raw_val_total_return",
            "selected_integer_leverage",
            "continuous_required_leverage",
            "oos_smart_sortino",
        ):
            row[key] = _safe_float(quality.get(key), 0.0)
    if item.get("train_val_stability"):
        for key, value in dict(item["train_val_stability"]).items():
            row[f"train_val_stability_{key}"] = _safe_float(value, 0.0)
    if item.get("current_base_comparison"):
        current_base = dict(item["current_base_comparison"])
        row["current_base_name"] = current_base.get("current_base_name", "")
        row["current_base_train_val_stability_score"] = _safe_float(
            current_base.get("current_base_train_val_stability_score"), 0.0
        )
        row["current_base_oos_return"] = _safe_float(
            current_base.get("current_base_oos_return"), 0.0
        )
        row["current_base_oos_return_risk"] = _safe_float(
            current_base.get("current_base_oos_return_risk"), 0.0
        )
        row["candidate_oos_return_risk"] = _safe_float(
            current_base.get("candidate_oos_return_risk"), 0.0
        )
        for key in CURRENT_BASE_GATE_KEYS:
            row[key] = bool(dict(current_base.get("gates") or {}).get(key))
    if item.get("allocator_diagnostics"):
        diagnostics = dict(item["allocator_diagnostics"])
        row["allocator_selection_basis"] = diagnostics.get("selection_basis", "")
        row["allocator_integer_leverage_only"] = bool(diagnostics.get("integer_leverage_only"))
        row["allocator_selected_grid_score"] = _safe_float(
            diagnostics.get("selected_grid_score"), 0.0
        )
    for split_name, split in item["splits"].items():
        metrics = split["metrics"]
        for key in (
            "total_return",
            "cagr",
            "max_drawdown",
            "sharpe",
            "sortino",
            "calmar",
            "volatility",
        ):
            row[f"{split_name}_{key}"] = _safe_float(metrics.get(key), 0.0)
        row[f"{split_name}_round_trips"] = int(split.get("round_trips") or 0)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):+.4%}"


def _fmt_float(value: Any) -> str:
    return f"{_safe_float(value):.6f}"


def _markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    selected = payload.get("selected_by_validation") or {}
    best_success = payload.get("best_success_candidate") or {}
    best_oos = payload.get("diagnostic_best_oos") or {}
    quarantine = payload.get("diagnostic_quarantine") or []
    memory_policy = payload.get("memory_policy") or {}
    memory_summary = payload.get("memory_summary") or {}
    lockbox_policy = payload.get("lockbox_policy") or LOCKBOX_POLICY
    combo_limit = payload.get("combo_limit_policy") or {}
    base_policy = payload.get("base_policy") or {}
    lifecycle = payload.get("no_improvement_lifecycle") or {}
    lines = [
        "# Profit moonshot fresh portfolio tuning",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Policy",
        "",
        "- Sleeve universe is restricted to train-positive and validation-positive fresh-start candidates.",
        "- Portfolio selection is train/validation-stability primary; locked-OOS is report-only / gate-only.",
        "- `diagnostic_best_oos` is not a deployable selection if it differs from validation selection.",
        f"- Selection label: `{lockbox_policy.get('selection_label')}`.",
        f"- Locked-OOS label: `{lockbox_policy.get('locked_oos_label')}`.",
        f"- Locked-OOS gate label: `{lockbox_policy.get('locked_oos_gate_label')}`.",
        f"- Diagnostic quarantine label: `{lockbox_policy.get('diagnostic_not_promoted_label')}`.",
        f"- Current-base artifact: `{base_policy.get('current_base_artifact', '')}`.",
        f"- Current-base status: `{base_policy.get('status', 'not_recorded')}`.",
        "- Train/validation stability objective: "
        f"`frozen_weighted_train_validation_score_v1` (current base "
        f"`{CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE:.6f}`).",
        f"- No-improvement lifecycle: `{lifecycle.get('status', 'not_recorded')}`.",
        f"- Stable-return floor: train, validation, and locked-OOS monthlyized return `>={MIN_STABLE_MONTHLY_RETURN:.2%}`.",
        f"- Train buffer: post-leverage train monthlyized return `>={MIN_BUFFERED_TRAIN_MONTHLY_RETURN:.2%}` and raw/unlevered train monthlyized return `>={MIN_RAW_TRAIN_MONTHLY_RETURN:.2%}`.",
        "- Leverage policy: `train_val_monthly_return_budget` uses an integer train/validation-only grid; continuous floor-fitting leverage is diagnostic only.",
        f"- MDD budget: locked-OOS max drawdown `≤{MAX_ACCEPTABLE_OOS_MDD:.2%}`.",
        f"- Quality floors: OOS Sharpe `≥{SUCCESS_SHARPE:.1f}`, Sortino `≥{SUCCESS_SORTINO:.1f}`, smart Sortino `≥{SUCCESS_SMART_SORTINO:.1f}`, Calmar `≥{SUCCESS_CALMAR:.1f}`.",
        f"- Incumbent improvement still requires current-champion return/risk improvement from OOS return `>{CURRENT_CHAMPION_OOS_RETURN:.4%}`.",
        "",
        "## Runtime guard",
        "",
        f"- Heavy-run lock: `{memory_policy.get('heavy_lock_path')}`",
        f"- Explicit memory budget: `{memory_policy.get('explicit_budget_bytes')}` bytes",
        f"- RSS summary: `{memory_summary.get('summary_path') or memory_summary.get('rss_log_path')}`",
        "",
        "## Summary",
        "",
        f"- Candidate sleeves considered: `{payload['candidate_sleeve_count']}`",
        f"- Portfolio specs evaluated: `{payload['portfolio_spec_count']}`",
        f"- Combo cap per size: `{combo_limit.get('max_combos_per_size', 'n/a')}`; skipped by size: `{combo_limit.get('limit_hits_by_size', {})}`",
        f"- Success candidates: `{payload['success_candidate_count']}`",
        f"- Peak RSS: `{payload['peak_rss_mib']:.3f} MiB`",
        "",
    ]
    if selected:
        split = selected.get("splits", {})
        lines.extend(
            [
                "## Selected by validation",
                "",
                f"- `{selected.get('name')}`",
                f"- sleeves: `{', '.join(selected.get('sleeves') or [])}`",
                f"- train: `{_fmt_pct(split.get('train', {}).get('metrics', {}).get('total_return'))}`",
                f"- val: `{_fmt_pct(split.get('val', {}).get('metrics', {}).get('total_return'))}`",
                f"- locked OOS: `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('total_return'))}`, Sharpe `{_fmt_float(split.get('oos', {}).get('metrics', {}).get('sharpe'))}`, MDD `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('max_drawdown'))}`",
                f"- monthlyized train/val/OOS: `{_fmt_pct((selected.get('return_quality') or {}).get('train_monthlyized_return'))}` / `{_fmt_pct((selected.get('return_quality') or {}).get('val_monthlyized_return'))}` / `{_fmt_pct((selected.get('return_quality') or {}).get('oos_monthlyized_return'))}`; smart Sortino `{_fmt_float((selected.get('return_quality') or {}).get('oos_smart_sortino'))}`",
                f"- raw monthlyized train/val: `{_fmt_pct((selected.get('return_quality') or {}).get('raw_train_monthlyized_return'))}` / `{_fmt_pct((selected.get('return_quality') or {}).get('raw_val_monthlyized_return'))}`; leverage `{_fmt_float(selected.get('leverage'))}`",
                f"- promotion status: `{selected.get('promotion_status')}` / failed gates: `{','.join(selected.get('failed_gates') or [])}`",
                "",
            ]
        )
    if best_success:
        split = best_success.get("splits", {})
        lines.extend(
            [
                "## Best success candidate",
                "",
                "- Ranked by the frozen train/validation stability score after all locked-OOS gates pass.",
                f"- `{best_success.get('name')}`",
                f"- sleeves: `{', '.join(best_success.get('sleeves') or [])}`",
                f"- train: `{_fmt_pct(split.get('train', {}).get('metrics', {}).get('total_return'))}`",
                f"- val: `{_fmt_pct(split.get('val', {}).get('metrics', {}).get('total_return'))}`",
                f"- locked OOS: `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('total_return'))}`, Sharpe `{_fmt_float(split.get('oos', {}).get('metrics', {}).get('sharpe'))}`, MDD `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('max_drawdown'))}`",
                f"- monthlyized train/val/OOS: `{_fmt_pct((best_success.get('return_quality') or {}).get('train_monthlyized_return'))}` / `{_fmt_pct((best_success.get('return_quality') or {}).get('val_monthlyized_return'))}` / `{_fmt_pct((best_success.get('return_quality') or {}).get('oos_monthlyized_return'))}`; smart Sortino `{_fmt_float((best_success.get('return_quality') or {}).get('oos_smart_sortino'))}`",
                f"- raw monthlyized train/val: `{_fmt_pct((best_success.get('return_quality') or {}).get('raw_train_monthlyized_return'))}` / `{_fmt_pct((best_success.get('return_quality') or {}).get('raw_val_monthlyized_return'))}`; leverage `{_fmt_float(best_success.get('leverage'))}`",
                f"- promotion status: `{best_success.get('promotion_status')}` / failed gates: `{','.join(best_success.get('failed_gates') or [])}`",
                "",
            ]
        )
    if best_oos:
        split = best_oos.get("splits", {})
        lines.extend(
            [
                "## Diagnostic best OOS (not selection authority)",
                "",
                f"- `{best_oos.get('name')}`",
                f"- train: `{_fmt_pct(split.get('train', {}).get('metrics', {}).get('total_return'))}`",
                f"- val: `{_fmt_pct(split.get('val', {}).get('metrics', {}).get('total_return'))}`",
                f"- locked OOS: `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('total_return'))}`, Sharpe `{_fmt_float(split.get('oos', {}).get('metrics', {}).get('sharpe'))}`, MDD `{_fmt_pct(split.get('oos', {}).get('metrics', {}).get('max_drawdown'))}`",
                f"- monthlyized locked OOS: `{_fmt_pct((best_oos.get('return_quality') or {}).get('oos_monthlyized_return'))}`; smart Sortino `{_fmt_float((best_oos.get('return_quality') or {}).get('oos_smart_sortino'))}`",
                f"- promotion status: `{best_oos.get('promotion_status')}`",
                "",
            ]
        )
    if quarantine:
        lines.extend(
            [
                "## H6 diagnostic quarantine",
                "",
                "- High-return locked-OOS diagnostics that fail promotion gates are retained as research evidence only.",
                "- Quarantined rows use the explicit `diagnostic_not_promoted` label and are not promoted success.",
                "",
                "| rank | name | mode | locked OOS | locked OOS MDD | failed gates |",
                "|---:|---|---|---:|---:|---|",
            ]
        )
        for idx, item in enumerate(quarantine[:10], start=1):
            split = item.get("splits", {})
            oos = split.get("oos", {}).get("metrics", {})
            lines.append(
                f"| {idx} | `{item.get('name')}` | `{item.get('mode')}` | "
                f"{_fmt_pct(oos.get('total_return'))} | {_fmt_pct(oos.get('max_drawdown'))} | "
                f"`{','.join(item.get('failed_gates') or [])}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Top rows",
            "",
            "| rank | name | mode | leverage | success | train | val | locked OOS | OOS monthly | OOS MDD | OOS Sharpe | smart Sortino | failed gates |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, row in enumerate(rows[:15], start=1):
        lines.append(
            f"| {idx} | `{row['name']}` | `{row['mode']}` | {_fmt_float(row.get('leverage'))} | {row['success_candidate']} | "
            f"{_fmt_pct(row['train_total_return'])} | {_fmt_pct(row['val_total_return'])} | "
            f"{_fmt_pct(row['oos_total_return'])} | {_fmt_pct(row.get('oos_monthlyized_return'))} | "
            f"{_fmt_pct(row['oos_max_drawdown'])} | {_fmt_float(row['oos_sharpe'])} | "
            f"{_fmt_float(row.get('oos_smart_sortino'))} | `{row['failed_gates']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _train_val_stability_sort_key(item: dict[str, Any]) -> float:
    if "train_val_stability_score" in item:
        return _safe_float(item.get("train_val_stability_score"))
    quality = item.get("return_quality")
    if isinstance(quality, dict) and "train_val_stability_score" in quality:
        return _safe_float(quality.get("train_val_stability_score"))
    if item.get("train_val_stability"):
        return _train_val_stability_score(item)
    return _safe_float(item.get("validation_score"))


def _portfolio_report_sort_key(item: dict[str, Any]) -> tuple[bool, float, bool, str]:
    return (
        not bool(item["success_candidate"]),
        -_train_val_stability_sort_key(item),
        item.get("promotion_status") == LOCKBOX_POLICY["diagnostic_not_promoted_label"],
        str(item.get("name") or ""),
    )


def build_payload(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fresh = _load_fresh_module()
    rows = _read_rows(Path(args.candidate_csv))
    current_base_path = (
        ""
        if bool(getattr(args, "no_current_base", False))
        else getattr(args, "current_base_artifact", str(DEFAULT_CURRENT_BASE_ARTIFACT))
    )
    current_base_candidate, base_policy = _load_current_base_candidate(current_base_path)
    pool, pool_policy = _candidate_pool_with_metadata(
        rows,
        top_n=int(args.top_n),
        family_quota=int(args.family_quota),
        calendar_neighborhood_reps=int(args.calendar_neighborhood_reps),
    )
    current_base_sleeves = [
        str(name) for name in current_base_candidate.get("sleeves") or [] if str(name)
    ]
    if current_base_sleeves:
        rows_by_name = {str(row.get("name") or ""): row for row in rows}
        pool_names = {str(row.get("name") or "") for row in pool}
        anchored_rows: list[dict[str, str]] = []
        for name in current_base_sleeves:
            row = rows_by_name.get(name)
            if not row or name in pool_names:
                continue
            if (
                _safe_float(row.get("train_total_return")) <= 0.0
                or _safe_float(row.get("val_total_return")) <= 0.0
            ):
                continue
            anchored_rows.append(row)
            pool_names.add(name)
        if anchored_rows:
            pool.extend(anchored_rows)
        pool_policy["current_base_anchor_names"] = current_base_sleeves
        pool_policy["current_base_anchor_added"] = [
            str(row.get("name") or "") for row in anchored_rows
        ]
        pool_policy["current_base_anchor_selection_basis"] = "current_base_train_val_recheck_only"
    oos_end = datetime.fromisoformat(str(args.oos_end_date)).date()
    splits = fresh._split_windows(oos_end=oos_end)
    start = min(split.start for split in splits)
    end = max(split.end for split in splits)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    panel, data_metadata = fresh._joined_panel(
        market_root=Path(args.market_root),
        exchange=str(args.exchange),
        symbols=symbols,
        start=start,
        end=end,
    )
    arrays = fresh._build_arrays(panel, symbols)
    specs_by_name = {spec.name: spec for spec in fresh._candidate_specs(arrays, symbols)}
    pool = [row for row in pool if row["name"] in specs_by_name]
    candidate_rows_by_name = {row["name"]: dict(row) for row in pool}

    split_curves: dict[str, dict[str, list[float]]] = {}
    split_payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for row in pool:
        name = row["name"]
        split_curves[name] = {}
        split_payloads[name] = {}
        for split in splits:
            result = fresh._run_split(
                spec=specs_by_name[name], arrays=arrays, split=split, include_equity=True
            )
            split_curves[name][split.name] = list(result.get("equity_history") or [])
            split_payloads[name][split.name] = result

    portfolio_items: list[dict[str, Any]] = []
    names = [row["name"] for row in pool]
    max_k = max(2, min(int(args.max_sleeves), len(names))) if names else 0
    max_combos_per_size = max(1, int(args.max_combos_per_size))
    combo_limit_hits: dict[str, int] = {}
    evaluated_combos: set[frozenset[str]] = set()
    forced_combo_names: list[tuple[str, ...]] = []
    if 2 <= len(current_base_sleeves) <= max_k and all(
        name in names for name in current_base_sleeves
    ):
        forced_combo_names.append(tuple(current_base_sleeves))

    def _append_combo(combo: tuple[str, ...]) -> None:
        combo_key = frozenset(combo)
        if combo_key in evaluated_combos:
            return
        evaluated_combos.add(combo_key)
        for mode in (
            "equal_weight",
            "additive_sleeves",
            "train_val_target_return_budget",
            "train_val_monthly_return_budget",
            "validation_return_risk_weight",
            "validation_drawdown_budget",
            "cluster_capped_validation_weight",
        ):
            portfolio_items.append(
                _combo_metrics(
                    fresh=fresh,
                    combo_names=combo,
                    split_curves=split_curves,
                    split_payloads=split_payloads,
                    mode=mode,
                    candidate_rows_by_name=candidate_rows_by_name,
                    current_base_candidate=current_base_candidate,
                    cluster_cap=float(args.cluster_cap),
                    sleeve_cap=float(args.sleeve_cap),
                    correlation_threshold=float(args.correlation_threshold),
                )
            )

    for combo in forced_combo_names:
        _append_combo(combo)
    for size in range(2, max_k + 1):
        total_for_size = math.comb(len(names), size)
        if total_for_size > max_combos_per_size:
            combo_limit_hits[str(size)] = int(total_for_size - max_combos_per_size)
        for combo in itertools.islice(itertools.combinations(names, size), max_combos_per_size):
            _append_combo(combo)
    portfolio_items.sort(key=_portfolio_report_sort_key)
    csv_rows = [_flatten_row(item) for item in portfolio_items]
    selected_by_validation = max(portfolio_items, key=_train_val_stability_sort_key, default={})
    success_candidates = [item for item in portfolio_items if bool(item["success_candidate"])]
    best_success_candidate = max(
        success_candidates,
        key=_train_val_stability_sort_key,
        default={},
    )
    retained_base_candidate = _compact_current_base_candidate(current_base_candidate)
    selected_best_candidate = best_success_candidate or retained_base_candidate
    selection_outcome = (
        "improved_success_candidate"
        if best_success_candidate
        else "no_improvement_current_base_retained"
        if retained_base_candidate
        else "no_improvement_no_base"
    )
    diagnostic_best_oos = max(
        portfolio_items,
        key=lambda item: (
            _safe_float(item["splits"]["oos"]["metrics"].get("total_return")),
            _safe_float(item["splits"]["oos"]["metrics"].get("sharpe")),
        ),
        default={},
    )
    diagnostic_quarantine = sorted(
        [
            item
            for item in portfolio_items
            if bool(item.get("diagnostic_not_promoted"))
            and _safe_float(item["splits"]["oos"]["metrics"].get("total_return"))
            > CURRENT_CHAMPION_OOS_RETURN
        ],
        key=lambda item: (
            _safe_float(item["splits"]["oos"]["metrics"].get("total_return")),
            -_safe_float(item["splits"]["oos"]["metrics"].get("max_drawdown"), 1.0),
        ),
        reverse=True,
    )
    payload = {
        "artifact_kind": "profit_moonshot_fresh_portfolio_tuning",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "oos_end_date": oos_end.isoformat(),
        "candidate_csv": str(args.candidate_csv),
        "candidate_sleeve_count": len(pool),
        "portfolio_spec_count": len(portfolio_items),
        "combo_limit_policy": {
            "max_combos_per_size": max_combos_per_size,
            "limit_hits_by_size": combo_limit_hits,
            "pool_order": pool_policy.get("pool_order"),
            "family_quota": int(args.family_quota),
            "forced_current_base_combos": [list(combo) for combo in forced_combo_names],
        },
        "candidate_pool_policy": pool_policy,
        "allocator_policy": {
            "modes": [
                "equal_weight",
                "additive_sleeves",
                "train_val_target_return_budget",
                "train_val_monthly_return_budget",
                "validation_return_risk_weight",
                "validation_drawdown_budget",
                "cluster_capped_validation_weight",
            ],
            "cluster_cap": float(args.cluster_cap),
            "sleeve_cap": float(args.sleeve_cap),
            "correlation_threshold": float(args.correlation_threshold),
            "selection_basis": "train_val_stability_only",
            "uses_locked_oos_for_selection": False,
            "train_val_stability_formula": "frozen_weighted_train_validation_score_v1",
            "train_val_stability_components": list(TRAIN_VAL_STABILITY_COMPONENTS),
            "integer_leverage_only": True,
            "integer_leverage_grid": list(
                range(MIN_INTEGER_LEVERAGE, int(MAX_TRAIN_VAL_MONTHLY_BUDGET_LEVERAGE) + 1)
            ),
            "minimum_buffered_train_monthly_return": MIN_BUFFERED_TRAIN_MONTHLY_RETURN,
            "minimum_raw_train_monthly_return": MIN_RAW_TRAIN_MONTHLY_RETURN,
            "minimum_raw_val_monthly_return": MIN_RAW_VAL_MONTHLY_RETURN,
        },
        "success_candidate_count": len(success_candidates),
        "best_success_candidate": best_success_candidate,
        "selected_best_candidate": selected_best_candidate,
        "selection_outcome": selection_outcome,
        "selected_by_validation": selected_by_validation,
        "selected_by_train_val_stability": selected_by_validation,
        "diagnostic_best_oos": diagnostic_best_oos,
        "diagnostic_quarantine": diagnostic_quarantine[:50],
        "data_metadata": data_metadata,
        "peak_rss_mib": _rss_mib(),
        "lockbox_policy": dict(LOCKBOX_POLICY),
        "base_policy": base_policy,
        "current_base_candidate": retained_base_candidate,
        "no_improvement_lifecycle": _no_improvement_lifecycle(
            current_base_candidate=current_base_candidate,
            selected_by_validation=selected_by_validation,
            best_success_candidate=best_success_candidate,
            success_candidate_count=len(success_candidates),
        ),
        "memory_policy": memory_policy_payload(
            budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        ),
    }
    return payload, csv_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-root", default="data/market_parquet")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,TRX/USDT")
    parser.add_argument("--oos-end-date", default="2026-05-06")
    parser.add_argument("--candidate-csv", default=str(DEFAULT_CANDIDATE_CSV))
    parser.add_argument("--current-base-artifact", default=str(DEFAULT_CURRENT_BASE_ARTIFACT))
    parser.add_argument("--no-current-base", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-n", type=int, default=18)
    parser.add_argument("--family-quota", type=int, default=0)
    parser.add_argument("--calendar-neighborhood-reps", type=int, default=8)
    parser.add_argument("--cluster-cap", type=float, default=0.50)
    parser.add_argument("--sleeve-cap", type=float, default=0.35)
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--max-sleeves", type=int, default=5)
    parser.add_argument("--max-combos-per-size", type=int, default=12_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fresh_portfolio_tuning_latest.json"
    csv_path = output_dir / "fresh_portfolio_tuning_candidates.csv"
    md_path = output_dir / "fresh_portfolio_tuning_latest.md"
    memory_guard = acquire_portfolio_memory_guard(
        run_name=RUN_NAME,
        output_dir=output_dir,
        input_path=args.candidate_csv,
        metadata={
            "script": Path(__file__).name,
            "top_n": int(args.top_n),
            "family_quota": int(args.family_quota),
            "calendar_neighborhood_reps": int(args.calendar_neighborhood_reps),
            "cluster_cap": float(args.cluster_cap),
            "sleeve_cap": float(args.sleeve_cap),
            "correlation_threshold": float(args.correlation_threshold),
            "max_sleeves": int(args.max_sleeves),
            "current_base_artifact": str(args.current_base_artifact),
            "no_current_base": bool(args.no_current_base),
            "locked_oos_label": LOCKBOX_POLICY["locked_oos_label"],
            "locked_oos_gate_label": LOCKBOX_POLICY["locked_oos_gate_label"],
            "selection_label": LOCKBOX_POLICY["selection_label"],
        },
        budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES,
    )
    finalized = False
    try:
        memory_guard.checkpoint(
            "start",
            {
                "candidate_csv": str(args.candidate_csv),
                "top_n": int(args.top_n),
                "family_quota": int(args.family_quota),
                "calendar_neighborhood_reps": int(args.calendar_neighborhood_reps),
                "max_sleeves": int(args.max_sleeves),
                "current_base_artifact": str(args.current_base_artifact),
                "no_current_base": bool(args.no_current_base),
            },
        )
        payload, rows = build_payload(args)
        payload["lockbox_policy"] = dict(LOCKBOX_POLICY)
        payload["memory_policy"] = memory_policy_payload(
            budget_bytes=PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
        )
        payload["rss_log_path"] = str(memory_guard.rss_log_path)
        payload["memory_summary_path"] = str(memory_guard.summary_path)
        _write_csv(csv_path, rows)
        memory_guard.checkpoint(
            "artifacts_prepared",
            {
                "portfolio_spec_count": int(payload["portfolio_spec_count"]),
                "success_candidate_count": int(payload["success_candidate_count"]),
            },
        )
        memory_summary = memory_guard.finalize(
            status="completed",
            context={
                "json_path": str(json_path),
                "markdown_path": str(md_path),
                "csv_path": str(csv_path),
                "portfolio_spec_count": int(payload["portfolio_spec_count"]),
                "success_candidate_count": int(payload["success_candidate_count"]),
            },
        )
        finalized = True
        memory_summary["summary_path"] = str(memory_guard.summary_path)
        payload["memory_summary"] = memory_summary
        payload["peak_rss_mib"] = max(
            _safe_float(payload.get("peak_rss_mib")),
            _safe_float(memory_summary.get("peak_rss_bytes")) / (1024.0 * 1024.0),
        )
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(_markdown(payload, rows) + "\n", encoding="utf-8")
    except Exception as exc:
        if not finalized:
            memory_guard.finalize(
                status="failed", error=str(exc), context={"script": Path(__file__).name}
            )
        raise
    finally:
        memory_guard.release()
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "csv": str(csv_path),
                "success_candidate_count": payload["success_candidate_count"],
                "peak_rss_mib": payload["peak_rss_mib"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
