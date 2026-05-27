"""Validate the reboot-safe profit-moonshot under-8GB mission result.

This mission validator is intentionally artifact-driven. It does not run heavy
research; it reads `.omx/specs/autoresearch-profit-moonshot-pass-under-8gb/result.json`
and fails until the final result contains: a passing candidate artifact, RSS
evidence under the explicit 8 GiB cap, local test evidence, CI evidence, and a
push/commit record when source changes were delivered.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSION_ROOT = REPO_ROOT / ".omx" / "specs" / "autoresearch-profit-moonshot-pass-under-8gb"
DEFAULT_RESULT_PATH = MISSION_ROOT / "result.json"
DEFAULT_VALIDATION_PATH = MISSION_ROOT / "validation_latest.json"
DEFAULT_REPORT_DIR = (
    REPO_ROOT
    / "var"
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260507"
    / "external_overhaul"
)
DEFAULT_MARKDOWN_PATH = DEFAULT_REPORT_DIR / "mission_validation_latest.md"
EIGHT_GIB_BYTES = 8 * 1024 * 1024 * 1024
CURRENT_CHAMPION_OOS_RETURN = 0.012181
CURRENT_BASE_TRAIN_MONTHLY_RETURN = 0.020000000000000018
CURRENT_BASE_VAL_MONTHLY_RETURN = 0.09848998131232078
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
MIN_STABLE_MONTHLY_RETURN = 0.02
MIN_BUFFERED_TRAIN_MONTHLY_RETURN = 0.0225
MIN_RAW_TRAIN_MONTHLY_RETURN = 0.01
MIN_RAW_VAL_MONTHLY_RETURN = 0.02
MAX_ACCEPTABLE_OOS_MDD = 0.25
MIN_OOS_SHARPE = 2.0
MIN_OOS_SORTINO = 3.0
MIN_OOS_SMART_SORTINO = 3.0
MIN_OOS_CALMAR = 1.0
MIN_TRAIN_SHARPE = 1.5
MIN_TRAIN_SORTINO = 1.5
MIN_TRAIN_CALMAR = 1.0
MIN_VAL_SHARPE = 3.0
MIN_VAL_SORTINO = 3.0
MIN_VAL_CALMAR = 3.0
_TIME_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """A single validator check result."""

    name: str
    passed: bool
    detail: str

    def as_payload(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _resolve_path(value: Any, *, repo_root: Path) -> Path | None:
    token = str(value or "").strip()
    if not token:
        return None
    path = Path(token)
    if path.is_absolute():
        return path
    return repo_root / path


def _compact_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(path)


def _nested_get(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_present(payload: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _candidate_artifact_path(result: Mapping[str, Any], *, repo_root: Path) -> Path | None:
    evidence = _as_mapping(result.get("evidence"))
    artifacts = _as_mapping(result.get("artifacts"))
    value = _first_present(
        result,
        (
            "passing_candidate_artifact",
            "passing_candidate_path",
            "candidate_artifact",
            "candidate_path",
        ),
    )
    if value is None:
        value = _first_present(evidence, ("passing_candidate_artifact", "candidate_artifact"))
    if value is None:
        value = _first_present(artifacts, ("passing_candidate_artifact", "candidate_artifact"))
    return _resolve_path(value, repo_root=repo_root)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "pass",
        "passed",
        "success",
    }


def _candidate_label_ok(result: Mapping[str, Any], candidate_payload: Mapping[str, Any]) -> bool:
    return any(
        _truthy(value)
        for value in (
            result.get("research_success_candidate"),
            result.get("live_equivalent_selection_eligible"),
            result.get("current_base_retained"),
            result.get("no_improvement_current_base_retained"),
            candidate_payload.get("research_success_candidate"),
            candidate_payload.get("live_equivalent_selection_eligible"),
            candidate_payload.get("current_base_retained"),
            candidate_payload.get("no_improvement_current_base_retained"),
            _nested_get(candidate_payload, "labels", "research_success_candidate"),
            _nested_get(candidate_payload, "labels", "live_equivalent_selection_eligible"),
            _nested_get(candidate_payload, "labels", "current_base_retained"),
            _nested_get(candidate_payload, "labels", "no_improvement_current_base_retained"),
        )
    )


def _no_improvement_base_retained(
    result: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> bool:
    labels = _as_mapping(candidate_payload.get("labels"))
    lifecycle = _as_mapping(result.get("no_improvement_lifecycle"))
    outcome_values = {
        str(result.get("selection_outcome") or "").strip().lower(),
        str(result.get("promotion_status") or "").strip().lower(),
        str(candidate_payload.get("selection_outcome") or "").strip().lower(),
        str(candidate_payload.get("promotion_status") or "").strip().lower(),
        str(lifecycle.get("status") or "").strip().lower(),
    }
    return any(
        value
        in {
            "no_improvement_current_base_retained",
            "no_improvement_base_preserved",
            "current_base_retained",
        }
        for value in outcome_values
    ) or any(
        _truthy(value)
        for value in (
            result.get("current_base_retained"),
            result.get("no_improvement_current_base_retained"),
            lifecycle.get("current_base_retained"),
            lifecycle.get("no_improvement"),
            candidate_payload.get("current_base_retained"),
            candidate_payload.get("no_improvement_current_base_retained"),
            labels.get("current_base_retained"),
            labels.get("no_improvement_current_base_retained"),
        )
    )


def _source_candidate_from_artifact(
    candidate_payload: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    source_path = _resolve_path(candidate_payload.get("source_artifact"), repo_root=repo_root)
    if source_path is None or not source_path.exists() or source_path.suffix.lower() != ".json":
        return {}
    source_payload = _load_json(source_path)
    candidate_name = str(candidate_payload.get("name") or "")
    for key in (
        "best_success_candidate",
        "selected_best_candidate",
        "selected_by_train_val_stability",
        "selected_by_validation",
        "diagnostic_best_oos",
        "current_base_candidate",
        "candidate",
    ):
        source_candidate = source_payload.get(key)
        if not isinstance(source_candidate, Mapping):
            continue
        if not candidate_name or str(source_candidate.get("name") or "") == candidate_name:
            return dict(source_candidate)
    if all(key in source_payload for key in ("splits", "gates")):
        return dict(source_payload)
    return {}


def _monthlyized_from_cagr(cagr: Any) -> float | None:
    parsed = _safe_float(cagr)
    if parsed is None or parsed <= -1.0:
        return None
    return float((1.0 + parsed) ** (1.0 / 12.0) - 1.0)


def _metric_sources(
    result: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    source_candidate: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    for source in (
        candidate_payload.get("return_quality"),
        candidate_payload.get("metrics"),
        candidate_payload.get("train_val_stability"),
        result.get("candidate_metrics"),
        source_candidate.get("return_quality"),
        source_candidate.get("train_val_stability"),
    ):
        if isinstance(source, Mapping):
            sources.append(source)
    return sources


def _first_float_from_sources(
    sources: Iterable[Mapping[str, Any]], keys: Iterable[str]
) -> float | None:
    for source in sources:
        for key in keys:
            parsed = _safe_float(source.get(key))
            if parsed is not None:
                return parsed
    return None


def _split_metrics(source_candidate: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    splits = source_candidate.get("splits")
    if not isinstance(splits, Mapping):
        return {}
    split = splits.get(split_name)
    if not isinstance(split, Mapping):
        return {}
    metrics = split.get("metrics")
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def _split_monthlyized_return(
    sources: Iterable[Mapping[str, Any]],
    source_candidate: Mapping[str, Any],
    split_name: str,
) -> float | None:
    key_aliases = {
        "train": ("train_monthlyized_return", "train_monthly_return"),
        "val": (
            "val_monthlyized_return",
            "validation_monthlyized_return",
            "val_monthly_return",
            "validation_monthly_return",
        ),
        "oos": (
            "oos_monthlyized_return",
            "locked_oos_monthlyized_return",
            "oos_monthly_return",
            "locked_oos_monthly_return",
        ),
    }
    direct = _first_float_from_sources(sources, key_aliases[split_name])
    if direct is not None:
        return direct
    return _monthlyized_from_cagr(_split_metrics(source_candidate, split_name).get("cagr"))


def _locked_oos_metric(
    sources: Iterable[Mapping[str, Any]],
    source_candidate: Mapping[str, Any],
    metric_name: str,
) -> float | None:
    aliases = {
        "total_return": ("locked_oos_total_return", "oos_total_return"),
        "max_drawdown": ("locked_oos_max_drawdown", "oos_max_drawdown"),
        "sharpe": ("locked_oos_sharpe", "oos_sharpe"),
        "sortino": ("locked_oos_sortino", "oos_sortino"),
        "smart_sortino": ("locked_oos_smart_sortino", "oos_smart_sortino"),
        "calmar": ("locked_oos_calmar", "oos_calmar"),
    }
    direct = _first_float_from_sources(sources, aliases[metric_name])
    if direct is not None:
        return direct
    return _safe_float(_split_metrics(source_candidate, "oos").get(metric_name))


def _train_val_metric(
    sources: Iterable[Mapping[str, Any]],
    source_candidate: Mapping[str, Any],
    split_name: str,
    metric_name: str,
) -> float | None:
    aliases = {
        ("train", "total_return"): ("train_total_return",),
        ("train", "max_drawdown"): ("train_max_drawdown",),
        ("train", "sharpe"): ("train_sharpe",),
        ("train", "sortino"): ("train_sortino",),
        ("train", "calmar"): ("train_calmar",),
        ("val", "total_return"): ("validation_total_return", "val_total_return"),
        ("val", "max_drawdown"): ("validation_max_drawdown", "val_max_drawdown"),
        ("val", "sharpe"): ("validation_sharpe", "val_sharpe"),
        ("val", "sortino"): ("validation_sortino", "val_sortino"),
        ("val", "calmar"): ("validation_calmar", "val_calmar"),
    }
    direct = _first_float_from_sources(sources, aliases[(split_name, metric_name)])
    if direct is not None:
        return direct
    return _safe_float(_split_metrics(source_candidate, split_name).get(metric_name))


def _raw_monthlyized_return(sources: Iterable[Mapping[str, Any]], split_name: str) -> float | None:
    aliases = {
        "train": ("raw_train_monthlyized_return", "unlevered_train_monthlyized_return"),
        "val": (
            "raw_val_monthlyized_return",
            "raw_validation_monthlyized_return",
            "unlevered_val_monthlyized_return",
            "unlevered_validation_monthlyized_return",
        ),
    }
    return _first_float_from_sources(sources, aliases[split_name])


def _candidate_leverage(
    sources: Iterable[Mapping[str, Any]],
    candidate_payload: Mapping[str, Any],
    source_candidate: Mapping[str, Any],
) -> float:
    direct = _first_float_from_sources(sources, ("leverage",))
    if direct is not None:
        return direct
    parsed = _safe_float(candidate_payload.get("leverage"))
    if parsed is not None:
        return parsed
    parsed = _safe_float(source_candidate.get("leverage"))
    return 1.0 if parsed is None else parsed


def _candidate_sleeve_count(
    sources: Iterable[Mapping[str, Any]],
    candidate_payload: Mapping[str, Any],
    source_candidate: Mapping[str, Any],
) -> int:
    direct = _first_float_from_sources(sources, ("sleeve_count",))
    if direct is not None:
        return round(direct)
    for payload in (candidate_payload, source_candidate):
        parsed = _safe_float(payload.get("sleeve_count"))
        if parsed is not None:
            return round(parsed)
        sleeves = payload.get("sleeves")
        if isinstance(sleeves, list | tuple):
            return len(sleeves)
    return 0


def _train_val_stability_score(
    *,
    train_monthly: float | None,
    val_monthly: float | None,
    train_sharpe: float | None,
    val_sharpe: float | None,
    train_sortino: float | None,
    val_sortino: float | None,
    train_calmar: float | None,
    val_calmar: float | None,
    train_mdd: float | None,
    val_mdd: float | None,
    leverage: float,
    sleeve_count: int,
) -> float | None:
    values = (
        train_monthly,
        val_monthly,
        train_sharpe,
        val_sharpe,
        train_sortino,
        val_sortino,
        train_calmar,
        val_calmar,
        train_mdd,
        val_mdd,
    )
    if any(value is None for value in values):
        return None
    return (
        35.0 * min(float(train_monthly), 0.06)
        + 45.0 * min(float(val_monthly), 0.12)
        + 0.40 * float(train_sharpe)
        + 0.60 * float(val_sharpe)
        + 0.35 * float(train_sortino)
        + 0.55 * float(val_sortino)
        + 0.20 * min(float(train_calmar), 20.0)
        + 0.30 * min(float(val_calmar), 60.0)
        - 35.0 * float(train_mdd)
        - 45.0 * float(val_mdd)
        - 0.15 * max(0.0, float(leverage) - CURRENT_BASE_LEVERAGE)
        - 0.25 * max(0.0, float(sleeve_count) - CURRENT_BASE_SLEEVE_COUNT)
    )


CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE = _train_val_stability_score(
    train_monthly=CURRENT_BASE_TRAIN_MONTHLY_RETURN,
    val_monthly=CURRENT_BASE_VAL_MONTHLY_RETURN,
    train_sharpe=CURRENT_BASE_TRAIN_SHARPE,
    val_sharpe=CURRENT_BASE_VAL_SHARPE,
    train_sortino=CURRENT_BASE_TRAIN_SORTINO,
    val_sortino=CURRENT_BASE_VAL_SORTINO,
    train_calmar=CURRENT_BASE_TRAIN_CALMAR,
    val_calmar=CURRENT_BASE_VAL_CALMAR,
    train_mdd=CURRENT_BASE_TRAIN_MDD,
    val_mdd=CURRENT_BASE_VAL_MDD,
    leverage=CURRENT_BASE_LEVERAGE,
    sleeve_count=CURRENT_BASE_SLEEVE_COUNT,
)


def _smart_sortino(
    monthly_return: float | None, max_drawdown: float | None, sortino: float | None
) -> float | None:
    if monthly_return is None or max_drawdown is None or sortino is None:
        return None
    return_floor_factor = max(0.0, min(1.0, monthly_return / MIN_STABLE_MONTHLY_RETURN))
    drawdown_budget_factor = max(
        0.0,
        1.0 - min(max(0.0, max_drawdown), MAX_ACCEPTABLE_OOS_MDD) / MAX_ACCEPTABLE_OOS_MDD,
    )
    return sortino * return_floor_factor * drawdown_budget_factor


def _liquidation_margin_evidence(
    result: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    source_candidate: Mapping[str, Any],
) -> dict[str, float | int | None]:
    """Return strongest liquidation/margin evidence when a promoted artifact provides it."""
    counts: list[int] = []
    buffers: list[float] = []
    ratios: list[float] = []
    event_drawdowns: list[float] = []
    event_loss_fractions: list[float] = []
    allowed_total_liquidations: list[int] = []
    max_allowed_event_drawdowns: list[float] = []
    max_allowed_loss_fractions: list[float] = []
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, Mapping):
            for key in ("liquidation_count", "liquidations", "liquidation_event_count_total"):
                if key in value:
                    parsed = _safe_float(value.get(key))
                    if parsed is not None:
                        counts.append(max(0, int(parsed)))
            if "minimum_margin_buffer" in value:
                parsed_buffer = _safe_float(value.get("minimum_margin_buffer"))
                if parsed_buffer is not None:
                    buffers.append(float(parsed_buffer))
            if "minimum_margin_ratio" in value:
                parsed_ratio = _safe_float(value.get("minimum_margin_ratio"))
                if parsed_ratio is not None:
                    ratios.append(float(parsed_ratio))
            for key in ("maximum_liquidation_event_drawdown", "event_drawdown"):
                if key in value:
                    parsed_drawdown = _safe_float(value.get(key))
                    if parsed_drawdown is not None:
                        event_drawdowns.append(float(parsed_drawdown))
            for key in ("maximum_liquidation_equity_loss_fraction", "equity_loss_fraction"):
                if key in value:
                    parsed_loss = _safe_float(value.get(key))
                    if parsed_loss is not None:
                        event_loss_fractions.append(float(parsed_loss))
            tolerance = value.get("liquidation_tolerance")
            if isinstance(tolerance, Mapping):
                parsed_allowed = _safe_float(tolerance.get("allowed_total_liquidations"))
                if parsed_allowed is not None:
                    allowed_total_liquidations.append(max(0, int(parsed_allowed)))
                parsed_event_drawdown = _safe_float(tolerance.get("max_liquidation_event_drawdown"))
                if parsed_event_drawdown is not None:
                    max_allowed_event_drawdowns.append(float(parsed_event_drawdown))
                parsed_loss = _safe_float(tolerance.get("max_liquidation_equity_loss_fraction"))
                if parsed_loss is not None:
                    max_allowed_loss_fractions.append(float(parsed_loss))
            for key in (
                "liquidation_aware_validation",
                "liquidation_validation",
                "margin_validation",
                "margin_safety",
                "policy",
                "metrics",
                "splits",
                "forced_5x",
                "selected_by_train_validation",
                "selected_by_train_validation_retune",
                "best_deployable_train_validation_retune",
                "current_base_reference_result",
                "train",
                "validation",
                "val",
                "oos",
            ):
                if key in value:
                    visit(value.get(key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for source in (result, candidate_payload, source_candidate):
        visit(source)

    return {
        "liquidation_count": max(counts) if counts else None,
        "minimum_margin_buffer": min(buffers) if buffers else None,
        "minimum_margin_ratio": min(ratios) if ratios else None,
        "maximum_liquidation_event_drawdown": max(event_drawdowns) if event_drawdowns else None,
        "maximum_liquidation_equity_loss_fraction": max(event_loss_fractions)
        if event_loss_fractions
        else None,
        "allowed_total_liquidations": max(allowed_total_liquidations)
        if allowed_total_liquidations
        else None,
        "max_liquidation_event_drawdown": min(max_allowed_event_drawdowns)
        if max_allowed_event_drawdowns
        else None,
        "max_liquidation_equity_loss_fraction": min(max_allowed_loss_fractions)
        if max_allowed_loss_fractions
        else None,
    }


def _candidate_return_quality_check(
    result: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> EvidenceCheck:
    source_candidate = _source_candidate_from_artifact(candidate_payload, repo_root=repo_root)
    sources = _metric_sources(result, candidate_payload, source_candidate)
    train_monthly = _split_monthlyized_return(sources, source_candidate, "train")
    val_monthly = _split_monthlyized_return(sources, source_candidate, "val")
    oos_monthly = _split_monthlyized_return(sources, source_candidate, "oos")
    raw_train_monthly = _raw_monthlyized_return(sources, "train")
    raw_val_monthly = _raw_monthlyized_return(sources, "val")
    oos_return = _locked_oos_metric(sources, source_candidate, "total_return")
    oos_mdd = _locked_oos_metric(sources, source_candidate, "max_drawdown")
    oos_sharpe = _locked_oos_metric(sources, source_candidate, "sharpe")
    oos_sortino = _locked_oos_metric(sources, source_candidate, "sortino")
    oos_smart_sortino = _locked_oos_metric(sources, source_candidate, "smart_sortino")
    oos_calmar = _locked_oos_metric(sources, source_candidate, "calmar")
    train_sharpe = _train_val_metric(sources, source_candidate, "train", "sharpe")
    train_sortino = _train_val_metric(sources, source_candidate, "train", "sortino")
    train_calmar = _train_val_metric(sources, source_candidate, "train", "calmar")
    train_mdd = _train_val_metric(sources, source_candidate, "train", "max_drawdown")
    val_sharpe = _train_val_metric(sources, source_candidate, "val", "sharpe")
    val_sortino = _train_val_metric(sources, source_candidate, "val", "sortino")
    val_calmar = _train_val_metric(sources, source_candidate, "val", "calmar")
    val_mdd = _train_val_metric(sources, source_candidate, "val", "max_drawdown")
    leverage = _candidate_leverage(sources, candidate_payload, source_candidate)
    sleeve_count = _candidate_sleeve_count(sources, candidate_payload, source_candidate)
    oos_return_risk = (
        None
        if oos_return is None or oos_mdd is None or oos_mdd <= 0.0
        else float(oos_return) / float(oos_mdd)
    )
    train_val_stability_score = _train_val_stability_score(
        train_monthly=train_monthly,
        val_monthly=val_monthly,
        train_sharpe=train_sharpe,
        val_sharpe=val_sharpe,
        train_sortino=train_sortino,
        val_sortino=val_sortino,
        train_calmar=train_calmar,
        val_calmar=val_calmar,
        train_mdd=train_mdd,
        val_mdd=val_mdd,
        leverage=leverage,
        sleeve_count=sleeve_count,
    )
    no_improvement_base_retained = _no_improvement_base_retained(result, candidate_payload)
    if oos_smart_sortino is None:
        oos_smart_sortino = _smart_sortino(oos_monthly, oos_mdd, oos_sortino)
    liquidation_evidence = _liquidation_margin_evidence(result, candidate_payload, source_candidate)
    liquidation_count = liquidation_evidence["liquidation_count"]
    minimum_margin_buffer = liquidation_evidence["minimum_margin_buffer"]
    minimum_margin_ratio = liquidation_evidence["minimum_margin_ratio"]
    maximum_liquidation_event_drawdown = liquidation_evidence["maximum_liquidation_event_drawdown"]
    maximum_liquidation_equity_loss_fraction = liquidation_evidence[
        "maximum_liquidation_equity_loss_fraction"
    ]
    allowed_total_liquidations = liquidation_evidence["allowed_total_liquidations"]
    max_liquidation_event_drawdown = liquidation_evidence["max_liquidation_event_drawdown"]
    max_liquidation_equity_loss_fraction = liquidation_evidence[
        "max_liquidation_equity_loss_fraction"
    ]
    details = {
        "minimum_stable_monthly_return": MIN_STABLE_MONTHLY_RETURN,
        "minimum_buffered_train_monthly_return": MIN_BUFFERED_TRAIN_MONTHLY_RETURN,
        "minimum_raw_train_monthly_return": MIN_RAW_TRAIN_MONTHLY_RETURN,
        "minimum_raw_val_monthly_return": MIN_RAW_VAL_MONTHLY_RETURN,
        "maximum_acceptable_oos_mdd": MAX_ACCEPTABLE_OOS_MDD,
        "minimum_oos_sharpe": MIN_OOS_SHARPE,
        "minimum_oos_sortino": MIN_OOS_SORTINO,
        "minimum_oos_smart_sortino": MIN_OOS_SMART_SORTINO,
        "minimum_oos_calmar": MIN_OOS_CALMAR,
        "minimum_train_sharpe": MIN_TRAIN_SHARPE,
        "minimum_train_sortino": MIN_TRAIN_SORTINO,
        "minimum_train_calmar": MIN_TRAIN_CALMAR,
        "minimum_val_sharpe": MIN_VAL_SHARPE,
        "minimum_val_sortino": MIN_VAL_SORTINO,
        "minimum_val_calmar": MIN_VAL_CALMAR,
        "current_champion_oos_return": CURRENT_CHAMPION_OOS_RETURN,
        "current_base_oos_return": CURRENT_BASE_OOS_RETURN,
        "current_base_oos_return_risk": CURRENT_BASE_OOS_RETURN_RISK,
        "current_base_train_val_stability_score": CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE,
        "no_improvement_base_retained": no_improvement_base_retained,
        "train_monthlyized_return": train_monthly,
        "val_monthlyized_return": val_monthly,
        "raw_train_monthlyized_return": raw_train_monthly,
        "raw_val_monthlyized_return": raw_val_monthly,
        "train_sharpe": train_sharpe,
        "train_sortino": train_sortino,
        "train_calmar": train_calmar,
        "train_max_drawdown": train_mdd,
        "val_sharpe": val_sharpe,
        "val_sortino": val_sortino,
        "val_calmar": val_calmar,
        "val_max_drawdown": val_mdd,
        "leverage": leverage,
        "sleeve_count": sleeve_count,
        "train_val_stability_score": train_val_stability_score,
        "locked_oos_monthlyized_return": oos_monthly,
        "locked_oos_total_return": oos_return,
        "locked_oos_max_drawdown": oos_mdd,
        "locked_oos_return_risk": oos_return_risk,
        "locked_oos_sharpe": oos_sharpe,
        "locked_oos_sortino": oos_sortino,
        "locked_oos_smart_sortino": oos_smart_sortino,
        "locked_oos_calmar": oos_calmar,
        "liquidation_count": liquidation_count,
        "minimum_margin_buffer": minimum_margin_buffer,
        "minimum_margin_ratio": minimum_margin_ratio,
        "maximum_liquidation_event_drawdown": maximum_liquidation_event_drawdown,
        "maximum_liquidation_equity_loss_fraction": maximum_liquidation_equity_loss_fraction,
        "allowed_total_liquidations": allowed_total_liquidations,
        "max_liquidation_event_drawdown": max_liquidation_event_drawdown,
        "max_liquidation_equity_loss_fraction": max_liquidation_equity_loss_fraction,
    }
    current_base_return_check = oos_return is not None and (
        oos_return >= CURRENT_BASE_OOS_RETURN
        if no_improvement_base_retained
        else oos_return > CURRENT_BASE_OOS_RETURN
    )
    current_base_return_risk_check = oos_return_risk is not None and (
        oos_return_risk >= CURRENT_BASE_OOS_RETURN_RISK
        if no_improvement_base_retained
        else oos_return_risk > CURRENT_BASE_OOS_RETURN_RISK
    )
    current_base_train_val_check = train_val_stability_score is not None and (
        train_val_stability_score >= CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE
        if no_improvement_base_retained
        else train_val_stability_score > CURRENT_BASE_TRAIN_VAL_STABILITY_SCORE
    )
    raw_train_check = (
        no_improvement_base_retained
        if raw_train_monthly is None
        else raw_train_monthly >= MIN_RAW_TRAIN_MONTHLY_RETURN
    )
    raw_val_check = (
        no_improvement_base_retained
        if raw_val_monthly is None
        else raw_val_monthly >= MIN_RAW_VAL_MONTHLY_RETURN
    )
    integer_leverage_check = no_improvement_base_retained or abs(leverage - round(leverage)) <= 1e-9
    if no_improvement_base_retained:
        liquidation_check = True if liquidation_count is None else int(liquidation_count) == 0
        margin_buffer_check = (
            True if minimum_margin_buffer is None else float(minimum_margin_buffer) > 0.0
        )
    else:
        liquidation_check = liquidation_count is not None and int(liquidation_count) == 0
        margin_buffer_check = (
            minimum_margin_buffer is not None and float(minimum_margin_buffer) > 0.0
        )
    if liquidation_count is None or int(liquidation_count) == 0:
        liquidation_event_drawdown_check = True
        liquidation_loss_fraction_check = True
    else:
        liquidation_event_drawdown_check = (
            maximum_liquidation_event_drawdown is not None
            and max_liquidation_event_drawdown is not None
            and float(maximum_liquidation_event_drawdown) <= float(max_liquidation_event_drawdown)
        )
        liquidation_loss_fraction_check = (
            maximum_liquidation_equity_loss_fraction is not None
            and max_liquidation_equity_loss_fraction is not None
            and float(maximum_liquidation_equity_loss_fraction)
            <= float(max_liquidation_equity_loss_fraction)
        )
    checks = (
        train_monthly is not None and train_monthly >= MIN_STABLE_MONTHLY_RETURN,
        train_monthly is not None and train_monthly >= MIN_BUFFERED_TRAIN_MONTHLY_RETURN
        if not no_improvement_base_retained
        else train_monthly is not None and train_monthly >= MIN_STABLE_MONTHLY_RETURN,
        val_monthly is not None and val_monthly >= MIN_STABLE_MONTHLY_RETURN,
        raw_train_check,
        raw_val_check,
        integer_leverage_check,
        liquidation_check,
        margin_buffer_check,
        liquidation_event_drawdown_check,
        liquidation_loss_fraction_check,
        train_sharpe is not None and train_sharpe >= MIN_TRAIN_SHARPE,
        train_sortino is not None and train_sortino >= MIN_TRAIN_SORTINO,
        train_calmar is not None and train_calmar >= MIN_TRAIN_CALMAR,
        val_sharpe is not None and val_sharpe >= MIN_VAL_SHARPE,
        val_sortino is not None and val_sortino >= MIN_VAL_SORTINO,
        val_calmar is not None and val_calmar >= MIN_VAL_CALMAR,
        oos_monthly is not None and oos_monthly >= MIN_STABLE_MONTHLY_RETURN,
        oos_return is not None and oos_return > CURRENT_CHAMPION_OOS_RETURN,
        current_base_return_check,
        current_base_return_risk_check,
        current_base_train_val_check,
        oos_mdd is not None and oos_mdd <= MAX_ACCEPTABLE_OOS_MDD,
        oos_sharpe is not None and oos_sharpe >= MIN_OOS_SHARPE,
        oos_sortino is not None and oos_sortino >= MIN_OOS_SORTINO,
        oos_smart_sortino is not None and oos_smart_sortino >= MIN_OOS_SMART_SORTINO,
        oos_calmar is not None and oos_calmar >= MIN_OOS_CALMAR,
    )
    return EvidenceCheck(
        "candidate_return_quality_contract",
        all(checks),
        json.dumps(details, ensure_ascii=False, sort_keys=True),
    )


def _flatten_evidence_paths(value: Any) -> list[Any]:
    items: list[Any] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            for key in ("path", "log_path", "rss_log_path", "summary_path", "artifact_path"):
                if item.get(key):
                    items.append(item[key])
        else:
            items.append(item)
    return items


def _rss_evidence_paths(result: Mapping[str, Any], *, repo_root: Path) -> list[Path]:
    evidence = _as_mapping(result.get("evidence"))
    memory = _as_mapping(result.get("memory_evidence"))
    candidates: list[Any] = []
    for source in (result, evidence, memory):
        candidates.extend(
            _flatten_evidence_paths(
                _first_present(
                    source,
                    (
                        "rss_under_8gb_logs",
                        "rss_evidence",
                        "rss_logs",
                        "memory_summaries",
                        "memory_summary_paths",
                    ),
                )
            )
        )
    paths: list[Path] = []
    seen: set[str] = set()
    for value in candidates:
        path = _resolve_path(value, repo_root=repo_root)
        if path is None:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed or abs(parsed) == float("inf"):
        return None
    return parsed


def _candidate_peak_rss_bytes(payload: Mapping[str, Any]) -> int | None:
    keys = (
        "peak_rss_bytes",
        "max_rss_bytes",
        "maximum_resident_set_size_bytes",
        "rss_bytes",
    )
    for key in keys:
        parsed = _safe_float(payload.get(key))
        if parsed is not None:
            return int(parsed)
    mib_keys = ("peak_rss_mib", "max_rss_mib", "rss_mib")
    for key in mib_keys:
        parsed = _safe_float(payload.get(key))
        if parsed is not None:
            return int(parsed * 1024 * 1024)
    kb_keys = ("peak_rss_kb", "max_rss_kb", "maximum_resident_set_size_kb")
    for key in kb_keys:
        parsed = _safe_float(payload.get(key))
        if parsed is not None:
            return int(parsed * 1024)
    for nested_key in ("memory_summary", "summary", "rss", "resource_usage"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            parsed = _candidate_peak_rss_bytes(nested)
            if parsed is not None:
                return parsed
    return None


def _parse_rss_peak_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return None
    if path.suffix.lower() == ".jsonl":
        peaks: list[int] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                peak = _candidate_peak_rss_bytes(payload)
                if peak is not None:
                    peaks.append(peak)
        return max(peaks) if peaks else None
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            return _candidate_peak_rss_bytes(payload)
    matches = [int(match.group(1)) * 1024 for match in _TIME_RSS_RE.finditer(text)]
    return max(matches) if matches else None


def _evidence_items(result: Mapping[str, Any], *keys: str) -> list[Any]:
    items: list[Any] = []
    evidence = _as_mapping(result.get("evidence"))
    for source in (result, evidence):
        for key in keys:
            items.extend(_as_list(source.get(key)))
    return items


def _has_success_evidence(items: list[Any], *, accepted_keys: tuple[str, ...]) -> bool:
    for item in items:
        if isinstance(item, Mapping):
            if any(_truthy(item.get(key)) for key in accepted_keys):
                return True
            conclusion = str(item.get("conclusion") or item.get("status") or "").strip().lower()
            if conclusion in {"success", "passed", "pass", "green", "completed"}:
                return True
        elif str(item or "").strip():
            return True
    return False


def _mutex_evidence_items(result: Mapping[str, Any]) -> list[Any]:
    items: list[Any] = []
    evidence = _as_mapping(result.get("evidence"))
    memory = _as_mapping(result.get("memory_evidence"))
    for source in (result, evidence, memory):
        for key in ("heavy_mutex_evidence", "mutex_evidence", "exclusive_run_evidence"):
            items.extend(_as_list(source.get(key)))
    return [item for item in items if item not in (None, "", [], {})]


def _mutex_evidence_check(result: Mapping[str, Any]) -> EvidenceCheck:
    items = _mutex_evidence_items(result)
    required = bool(result.get("requires_heavy_mutex_evidence") is True or items)
    if not required:
        return EvidenceCheck(
            "heavy_run_mutex_evidence",
            True,
            "not required by result; no heavy replay evidence declared",
        )

    ok_items = 0
    details: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            status = str(item.get("status") or item.get("conclusion") or "").strip().lower()
            lock_path = str(item.get("lock_path") or item.get("path") or "").strip()
            exclusive = _truthy(item.get("exclusive")) or _truthy(item.get("exclusive_lock"))
            overlap_ok = str(
                item.get("overlap_check") or item.get("overlap_status") or ""
            ).lower() in {
                "passed",
                "pass",
                "ok",
                "no_overlap",
                "none",
            }
            passed = bool(
                lock_path
                and exclusive
                and (status in {"passed", "completed", "success"} or overlap_ok)
            )
            ok_items += int(passed)
            details.append(
                {
                    "lock_path": lock_path,
                    "status": status,
                    "exclusive": exclusive,
                    "overlap_ok": overlap_ok,
                    "passed": passed,
                }
            )
        else:
            text = str(item or "").strip()
            passed = bool(text)
            ok_items += int(passed)
            details.append({"text": text, "passed": passed})
    return EvidenceCheck(
        "heavy_run_mutex_evidence",
        ok_items > 0,
        json.dumps(details, ensure_ascii=False, sort_keys=True)
        if details
        else "requires_heavy_mutex_evidence=true but no mutex evidence present",
    )


def validate(
    result_path: Path = DEFAULT_RESULT_PATH, *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Return a mission validation payload without mutating mission result state."""
    result = _load_json(result_path)
    checks: list[EvidenceCheck] = []

    declared_pass = bool(
        result.get("passed") is True and str(result.get("status") or "").lower() == "passed"
    )
    checks.append(
        EvidenceCheck(
            "declared_pass_status",
            declared_pass,
            "result has passed=true and status=passed"
            if declared_pass
            else "result is not yet status=passed with passed=true",
        )
    )

    candidate_path = _candidate_artifact_path(result, repo_root=repo_root)
    candidate_payload: dict[str, Any] = {}
    candidate_exists = bool(candidate_path and candidate_path.exists())
    if candidate_exists and candidate_path is not None and candidate_path.suffix.lower() == ".json":
        candidate_payload = _load_json(candidate_path)
    checks.append(
        EvidenceCheck(
            "passing_candidate_artifact_exists",
            candidate_exists,
            _compact_path(candidate_path, repo_root=repo_root)
            if candidate_path
            else "missing passing_candidate_artifact",
        )
    )
    label_ok = bool(candidate_exists and _candidate_label_ok(result, candidate_payload))
    checks.append(
        EvidenceCheck(
            "candidate_lifecycle_label",
            label_ok,
            "research/live-equivalent success label present"
            if label_ok
            else "missing research_success_candidate/live_equivalent label",
        )
    )
    checks.append(
        _candidate_return_quality_check(result, candidate_payload, repo_root=repo_root)
        if candidate_exists
        else EvidenceCheck(
            "candidate_return_quality_contract",
            False,
            "missing passing candidate artifact for return-quality validation",
        )
    )

    rss_paths = _rss_evidence_paths(result, repo_root=repo_root)
    rss_details: list[dict[str, Any]] = []
    rss_ok = bool(rss_paths)
    for path in rss_paths:
        peak = _parse_rss_peak_bytes(path)
        path_ok = bool(path.exists() and peak is not None and peak < EIGHT_GIB_BYTES)
        rss_ok = rss_ok and path_ok
        rss_details.append(
            {
                "path": _compact_path(path, repo_root=repo_root),
                "exists": path.exists(),
                "peak_rss_bytes": peak,
                "under_8gib": path_ok,
            }
        )
    checks.append(
        EvidenceCheck(
            "rss_under_8gib_evidence",
            rss_ok,
            json.dumps(rss_details, ensure_ascii=False, sort_keys=True)
            if rss_details
            else "missing RSS evidence paths",
        )
    )
    checks.append(_mutex_evidence_check(result))

    tests_items = _evidence_items(result, "test_evidence", "tests_and_ci_evidence", "tests")
    tests_ok = bool(
        result.get("tests_passed") is True
        or _has_success_evidence(tests_items, accepted_keys=("passed", "success"))
    )
    checks.append(
        EvidenceCheck(
            "local_tests_evidence",
            tests_ok,
            f"{len(tests_items)} test evidence item(s)"
            if tests_ok
            else "missing successful local test evidence",
        )
    )

    ci_items = _evidence_items(result, "ci_evidence", "tests_and_ci_evidence", "ci")
    ci_ok = bool(
        result.get("ci_passed") is True
        or str(result.get("ci_status") or "").lower() == "success"
        or _has_success_evidence(ci_items, accepted_keys=("passed", "success"))
    )
    checks.append(
        EvidenceCheck(
            "ci_success_evidence",
            ci_ok,
            f"{len(ci_items)} CI evidence item(s)" if ci_ok else "missing successful CI evidence",
        )
    )

    source_changed = result.get("source_changed")
    if source_changed is None:
        source_changed = "git_push_if_code_changed" in set(
            result.get("required_final_evidence") or []
        )
    git_items = _evidence_items(result, "git_evidence", "git_push", "push_evidence")
    git_ok = (not bool(source_changed)) or bool(
        result.get("pushed_commit")
        or result.get("pushed_ref")
        or _has_success_evidence(git_items, accepted_keys=("pushed", "success"))
    )
    checks.append(
        EvidenceCheck(
            "git_push_evidence_if_source_changed",
            git_ok,
            "source unchanged or push evidence present"
            if git_ok
            else "source_changed requires git push evidence",
        )
    )

    passed = all(check.passed for check in checks)
    payload = {
        "artifact_kind": "profit_moonshot_pass_under_8gb_mission_validation",
        "schema_version": "1.0",
        "generated_at": _utc_now_iso(),
        "status": "passed" if passed else "running",
        "passed": passed,
        "result_path": _compact_path(result_path, repo_root=repo_root),
        "budget_bytes": EIGHT_GIB_BYTES,
        "checks": [check.as_payload() for check in checks],
        "rss_evidence": rss_details,
        "summary": "mission evidence complete" if passed else "mission evidence incomplete",
    }
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Profit Moonshot Under-8GB Mission Validation",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Passed: `{payload.get('passed')}`",
        f"- Result path: `{payload.get('result_path')}`",
        f"- Budget bytes: `{payload.get('budget_bytes')}`",
        "",
        "## Checks",
    ]
    for check in payload.get("checks") or []:
        if not isinstance(check, Mapping):
            continue
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- {mark} `{check.get('name')}` — {check.get('detail')}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_VALIDATION_PATH))
    parser.add_argument("--markdown-path", default=str(DEFAULT_MARKDOWN_PATH))
    args = parser.parse_args(argv)

    payload = validate(Path(args.result_path))
    output_path = Path(args.output_path)
    markdown_path = Path(args.markdown_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
