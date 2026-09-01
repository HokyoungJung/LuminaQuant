#!/usr/bin/env python3
"""Build an auditable >=1-minute strategy catalog and scorecards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.strategies.plugin_interface import StrategyPlugin
from lumina_quant.strategies.registry import (
    get_strategy_map,
    get_strategy_param_schema,
    get_strategy_tier,
)
from lumina_quant.strategy_factory.candidate_library import (
    DEFAULT_BINANCE_TOP10_PLUS_METALS,
    build_binance_futures_candidates,
)

_SCHEMA_VERSION = "lumina_quant.strategy_catalog.v1"
_EVIDENCE_SCHEMA_VERSION = "lumina_quant.strategy_evidence.v1"
_COMPARISON_ARTIFACT_KIND = "common_period_1m_cold_start_strategy_screen"
_MIN_SCOPE_SECONDS = 60
_RESEARCH_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
_CANDIDATE_SYMBOLS = tuple(DEFAULT_BINANCE_TOP10_PLUS_METALS)
_TIMEFRAME_RE = re.compile(r"^(\d+)([smhdw])$")
_TIMEFRAME_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_ARTIFACT_NAMES = (
    "strategy_catalog.json",
    "strategy_catalog.csv",
    "strategy_scorecards.md",
    "family_scorecards.csv",
    "family_scorecards.md",
)
_POST_SNAPSHOT_FAMILY_OVERRIDES = {
    "EquityCurveKillSwitchOverlayStrategy": "volatility_risk_overlay",
    "KalmanPairsStatArbStrategy": "mean_reversion_relative_value",
    "MaScoreVolTargetRotationStrategy": "trend_momentum",
    "NoiseFilteredVolatilityBreakoutStrategy": "breakout",
    "PcaResidualStatArbStrategy": "mean_reversion_relative_value",
    "PrevDayBoxQuartileReversionStrategy": "mean_reversion_relative_value",
    "RsiDivergenceScaleOutStrategy": "mean_reversion_relative_value",
    "TrendGatedIbsReversionStrategy": "mean_reversion_relative_value",
    "TurtleUnitPyramidingStrategy": "breakout",
}


def get_catalog_strategy_map() -> dict[str, type[Any]]:
    """Return event strategies plus registered research batch plugins."""
    strategies: dict[str, type[Any]] = dict(get_strategy_map())
    strategies.update(
        {
            name: strategy_cls
            for name, strategy_cls in GLOBAL_REGISTRY.get_all("strategy").items()
            if GLOBAL_REGISTRY.get_interface("strategy", name) == "polars_batch"
            and issubclass(strategy_cls, StrategyPlugin)
        }
    )
    return strategies


def _sha256(path: Path, *, logical_path: str | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": logical_path if logical_path is not None else path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _module_source_identity(
    module_name: str,
    *,
    logical_path: str,
) -> dict[str, Any]:
    module = __import__(module_name, fromlist=["__name__"])
    source = Path(str(getattr(module, "__file__", ""))).resolve()
    if not source.is_file():
        raise ValueError(f"module source file not found for {module_name!r}")
    return {
        "module": module_name,
        **_sha256(source, logical_path=logical_path),
    }


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _canonical_utc_timestamp(value: Any, *, label: str) -> str:
    token = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp: {token!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def timeframe_seconds(value: Any) -> int | None:
    token = str(value or "").strip().lower()
    match = _TIMEFRAME_RE.fullmatch(token)
    if match is None:
        return None
    return int(match.group(1)) * _TIMEFRAME_UNIT_SECONDS[match.group(2)]


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable):
        values = value
    else:
        return ()
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def strategy_scope_metadata(strategy_cls: type[Any]) -> dict[str, Any]:
    required_timeframes = _string_tuple(getattr(strategy_cls, "required_timeframes", ()))
    required_seconds = [
        seconds
        for seconds in (timeframe_seconds(token) for token in required_timeframes)
        if seconds is not None
    ]
    raw_cadence = getattr(strategy_cls, "decision_cadence_seconds", None)
    cadence_seconds = (
        int(raw_cadence)
        if isinstance(raw_cadence, (int, float)) and math.isfinite(raw_cadence) and raw_cadence > 0
        else None
    )

    exclusion_reason: str | None = None
    if any(seconds < _MIN_SCOPE_SECONDS for seconds in required_seconds):
        exclusion_reason = "requires_sub_minute_timeframe_dependency"
    elif cadence_seconds is not None and cadence_seconds < _MIN_SCOPE_SECONDS:
        exclusion_reason = "decision_cadence_below_one_minute"

    explicit_seconds = list(required_seconds)
    if cadence_seconds is not None:
        explicit_seconds.append(cadence_seconds)
    if exclusion_reason is not None:
        cadence_status = "excluded_sub_minute"
    elif explicit_seconds:
        cadence_status = "explicit_in_scope"
    else:
        cadence_status = "unknown"

    scope_status = (
        "excluded_sub_minute"
        if exclusion_reason is not None
        else ("verified_in_scope" if explicit_seconds else "scope_unverified")
    )
    return {
        "in_scope": exclusion_reason is None,
        "scope_status": scope_status,
        "exclusion_reason": exclusion_reason,
        "required_timeframes": list(required_timeframes),
        "required_timeframe_seconds": required_seconds,
        "decision_cadence_seconds": cadence_seconds,
        "cadence_status": cadence_status,
    }


def _candidate_index() -> dict[str, dict[str, Any]]:
    rows = build_binance_futures_candidates(
        timeframes=_RESEARCH_TIMEFRAMES,
        symbols=_CANDIDATE_SYMBOLS,
    )
    index: dict[str, dict[str, Any]] = {}
    for candidate in rows:
        entry = index.setdefault(
            str(candidate.strategy_class),
            {"families": set(), "timeframes": set(), "candidate_count": 0},
        )
        entry["families"].add(str(candidate.family))
        entry["timeframes"].add(str(candidate.timeframe))
        entry["candidate_count"] += 1
    return {
        name: {
            "families": sorted(value["families"]),
            "timeframes": sorted(
                value["timeframes"], key=lambda token: timeframe_seconds(token) or 0
            ),
            "candidate_count": int(value["candidate_count"]),
        }
        for name, value in sorted(index.items())
    }


def _canonical_candidate_family(token: str) -> str:
    value = str(token).strip().lower().replace("-", "_")
    if value == "deep_research_leaf":
        return ""
    if "cross_section" in value:
        return "cross_sectional"
    if "breakout" in value:
        return "breakout"
    if (
        value in {"mean_reversion", "market_neutral", "relative_value"}
        or "reversion" in value
        or "pair" in value
    ):
        return "mean_reversion_relative_value"
    if value in {"trend", "momentum", "time_series_momentum"}:
        return "trend_momentum"
    if "carry" in value or "derivative" in value:
        return "derivatives_carry"
    if "season" in value or "calendar" in value:
        return "seasonality"
    if "overlay" in value or "risk" in value or "volatility" in value:
        return "volatility_risk_overlay"
    if "micro" in value or "intraday" in value or "flow" in value:
        return "microstructure_intraday"
    if value == "event_alpha":
        return "event_alpha"
    if value == "intermarket":
        return "intermarket"
    if value == "formulaic_alpha":
        return "formulaic_alpha"
    if "ensemble" in value or "router" in value or "regime" in value:
        return "ensemble_regime_router"
    return ""


def resolve_family(
    strategy_name: str,
    candidate_families: Iterable[str],
    strategy_overrides: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    override = dict((strategy_overrides or {}).get(strategy_name) or {})
    explicit = str(override.get("family") or "").strip()
    if explicit:
        return explicit, "evidence_override"
    post_snapshot = _POST_SNAPSHOT_FAMILY_OVERRIDES.get(strategy_name)
    if post_snapshot is not None:
        return post_snapshot, "post_snapshot_semantic_override"

    mapped = sorted(
        {
            family
            for token in candidate_families
            if token
            for family in (_canonical_candidate_family(token),)
            if family
        }
    )
    if len(mapped) == 1:
        return mapped[0], "candidate_library"
    return "unresolved", "unresolved_ambiguous_or_unmapped"


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _finite_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except TypeError, ValueError:
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    parsed = _finite_float(value)
    return None if parsed is None or not parsed.is_integer() else int(parsed)


def _bool_value(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_comparison_contract(summary_path: Path, comparison_path: Path) -> dict[str, Any]:
    if not comparison_path.is_file():
        raise FileNotFoundError(f"comparison CSV file not found: {comparison_path}")
    payload = _load_json_object(summary_path, label="comparison summary")
    if payload.get("artifact_kind") != _COMPARISON_ARTIFACT_KIND:
        raise ValueError(f"comparison summary artifact_kind must be {_COMPARISON_ARTIFACT_KIND!r}")
    generated_at = _canonical_utc_timestamp(
        payload.get("generated_at_utc"),
        label="comparison summary generated_at_utc",
    )
    strategy_count = _int_or_none(payload.get("strategy_count"))
    if strategy_count is None or strategy_count <= 0:
        raise ValueError("comparison summary strategy_count must be a positive integer")

    windows_raw = payload.get("windows")
    if not isinstance(windows_raw, Mapping) or set(windows_raw) != {"full", "recent"}:
        raise ValueError("comparison summary must define exactly full and recent windows")
    windows: dict[str, dict[str, Any]] = {}
    for label in ("full", "recent"):
        row = windows_raw.get(label)
        if not isinstance(row, Mapping):
            raise ValueError(f"comparison summary window {label!r} must be an object")
        start = _canonical_utc_timestamp(row.get("start"), label=f"{label} start")
        end = _canonical_utc_timestamp(
            row.get("end_exclusive"),
            label=f"{label} end_exclusive",
        )
        days = _int_or_none(row.get("days"))
        if days is None or days <= 0 or start >= end:
            raise ValueError(f"comparison summary window {label!r} is invalid")
        windows[label] = {"start": start, "end_exclusive": end, "days": days}
    if not (
        windows["full"]["start"]
        < windows["recent"]["start"]
        < windows["recent"]["end_exclusive"]
        == windows["full"]["end_exclusive"]
    ):
        raise ValueError("recent comparison window must be nested at the end of full")

    methodology = payload.get("methodology")
    if not isinstance(methodology, Mapping) or not str(
        methodology.get("timeframe") or ""
    ).startswith("exact 1m "):
        raise ValueError("comparison summary does not attest exact 1m methodology")
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or not any(
        "Not independent OOS" in str(item) for item in limitations
    ):
        raise ValueError("comparison summary must disclose non-independent OOS provenance")

    status_counts = payload.get("status_counts")
    if not isinstance(status_counts, Mapping):
        raise ValueError("comparison summary status_counts must be an object")
    for label in ("full", "recent"):
        counts = status_counts.get(label)
        parsed_counts = (
            [_int_or_none(raw) for raw in counts.values()] if isinstance(counts, Mapping) else []
        )
        if (
            not isinstance(counts, Mapping)
            or any(value is None or value < 0 for value in parsed_counts)
            or sum(value for value in parsed_counts if value is not None) != strategy_count
        ):
            raise ValueError(
                f"comparison summary {label} status counts do not equal strategy_count"
            )

    provenance = payload.get("window_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("comparison summary window_provenance must be an object")
    for label in ("full", "recent"):
        window_provenance = provenance.get(label)
        if (
            not isinstance(window_provenance, Mapping)
            or not str(window_provenance.get("git_commit") or "").strip()
        ):
            raise ValueError(f"comparison summary {label} provenance is incomplete")
        integrity = window_provenance.get("integrity_policy")
        if not isinstance(integrity, Mapping) or not all(
            integrity.get(key) is True
            for key in ("no_gap_fill", "no_interpolation", "no_synthetic_rows")
        ):
            raise ValueError(f"comparison summary {label} integrity policy is not fail-closed")

    artifacts = payload.get("artifacts")
    expected = artifacts.get(comparison_path.name) if isinstance(artifacts, Mapping) else None
    actual = _sha256(comparison_path, logical_path=comparison_path.name)
    if not isinstance(expected, Mapping) or any(
        expected.get(key) != actual[key] for key in ("bytes", "sha256")
    ):
        raise ValueError("comparison CSV does not match its summary artifact identity")

    return {
        "artifact_kind": _COMPARISON_ARTIFACT_KIND,
        "generated_at_utc": generated_at,
        "strategy_count": strategy_count,
        "timeframe": "1m",
        "windows": windows,
        "selection_provenance": "unsealed",
        "full_window_role": "diagnostic_screen",
        "recent_window_role": "nested_cold_start_sensitivity_not_independent_oos",
        "promotion_use": "forbidden",
        "summary": _sha256(summary_path, logical_path=summary_path.name),
        "comparison": actual,
    }


def load_comparison_rows(
    path: Path,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"comparison CSV file not found: {path}")
    provenance = {
        "artifact_kind": str((contract or {}).get("artifact_kind") or "unverified_comparison_csv"),
        "selection_provenance": str((contract or {}).get("selection_provenance") or "unknown"),
        "full_window_role": str((contract or {}).get("full_window_role") or "unverified"),
        "recent_window_role": str((contract or {}).get("recent_window_role") or "unverified"),
    }
    if contract:
        provenance["source_summary_sha256"] = str(
            dict(contract.get("summary") or {}).get("sha256") or ""
        )
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            name = str(raw.get("strategy") or "").strip()
            if not name:
                raise ValueError(f"comparison CSV contains a row without strategy: {path}")
            if name in rows:
                raise ValueError(f"comparison CSV contains duplicate strategy {name!r}: {path}")
            rows[name] = {
                "provenance": dict(provenance),
                "tier": str(raw.get("tier") or ""),
                "symbols": _int_or_none(raw.get("symbols")),
                "full": {
                    "status": str(raw.get("full_status") or "missing"),
                    "return": _finite_float(raw.get("full_return")),
                    "max_drawdown": _finite_float(raw.get("full_max_drawdown")),
                    "sharpe": _finite_float(raw.get("full_sharpe")),
                    "trades": _int_or_none(raw.get("full_trades")),
                    "log_daily": _finite_float(raw.get("full_log_daily")),
                },
                "recent": {
                    "status": str(raw.get("recent_status") or "missing"),
                    "return": _finite_float(raw.get("recent_return")),
                    "max_drawdown": _finite_float(raw.get("recent_max_drawdown")),
                    "sharpe": _finite_float(raw.get("recent_sharpe")),
                    "trades": _int_or_none(raw.get("recent_trades")),
                    "log_daily": _finite_float(raw.get("recent_log_daily")),
                },
                "comparable": _bool_value(raw.get("comparable")),
                "delta_log_daily": _finite_float(raw.get("delta_log_daily")),
                "daily_gap": _finite_float(raw.get("daily_gap")),
                "robust_log_daily": _finite_float(raw.get("robust_log_daily")),
            }
    expected_count = _int_or_none((contract or {}).get("strategy_count"))
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"comparison CSV has {len(rows)} rows; summary requires {expected_count}")
    return rows


def _normalize_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != _EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"evidence schema_version must be {_EVIDENCE_SCHEMA_VERSION!r}")
    as_of_date = str(payload.get("as_of_date") or "").strip()
    try:
        date.fromisoformat(as_of_date)
    except ValueError as exc:
        raise ValueError("evidence as_of_date must be an ISO-8601 date") from exc
    scope = str(payload.get("scope") or "").strip()
    if not scope:
        raise ValueError("evidence scope must be non-empty")

    families_raw = payload.get("families") or {}
    if not isinstance(families_raw, Mapping) or not families_raw:
        raise ValueError("evidence families must be a non-empty object")
    families: dict[str, dict[str, Any]] = {}
    for raw_id, raw_family in families_raw.items():
        family_id = str(raw_id).strip()
        if not family_id or not isinstance(raw_family, Mapping):
            raise ValueError("evidence family ids must be non-empty objects")
        family = dict(raw_family)
        for key in ("title", "thesis", "research_decision"):
            if not str(family.get(key) or "").strip():
                raise ValueError(f"evidence family {family_id!r} requires non-empty {key}")
        evidence_ids = family.get("evidence_ids")
        if not isinstance(evidence_ids, list) or any(
            not isinstance(value, str) or not value.strip() for value in evidence_ids
        ):
            raise ValueError(f"evidence family {family_id!r} evidence_ids must be a string list")
        families[family_id] = {
            **family,
            "evidence_ids": sorted(set(evidence_ids)),
        }

    overrides_raw = payload.get("strategy_overrides") or {}
    if not isinstance(overrides_raw, Mapping):
        raise ValueError("evidence strategy_overrides must be an object")
    sources_raw = payload.get("sources") or []
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("evidence sources must be a non-empty list")
    required_source_fields = {
        "id",
        "title",
        "authors",
        "publication",
        "date",
        "type",
        "grade",
        "url",
        "supports",
        "limitations",
    }
    sources: list[dict[str, Any]] = []
    for index, raw_source in enumerate(sources_raw):
        if not isinstance(raw_source, Mapping):
            raise ValueError(f"evidence source at index {index} must be an object")
        source = dict(raw_source)
        missing = sorted(
            key
            for key in required_source_fields
            if not isinstance(source.get(key), str) or not str(source.get(key)).strip()
        )
        if missing:
            raise ValueError(f"evidence source at index {index} has missing fields: {missing}")
        if not str(source["url"]).startswith("https://"):
            raise ValueError(f"evidence source {source['id']!r} URL must use HTTPS")
        sources.append(source)

    source_ids = [str(row["id"]) for row in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("evidence source ids must be non-empty and unique")
    known_source_ids = set(source_ids)
    for family_id, family in families.items():
        unknown = set(map(str, family.get("evidence_ids") or [])) - known_source_ids
        if unknown:
            raise ValueError(
                f"evidence family {family_id!r} references unknown sources: {sorted(unknown)}"
            )
    for strategy_name, override in overrides_raw.items():
        if not str(strategy_name).strip() or not isinstance(override, Mapping):
            raise ValueError(f"evidence strategy override {strategy_name!r} must be an object")
        family_id = str(override.get("family") or "").strip()
        if not family_id:
            raise ValueError(f"evidence strategy override {strategy_name!r} requires a family")
        if family_id not in families:
            raise ValueError(
                f"evidence strategy override {strategy_name!r} references unknown family {family_id!r}"
            )
        evidence_ids = override.get("evidence_ids")
        if not isinstance(evidence_ids, list) or any(
            not isinstance(value, str) or not value.strip() for value in evidence_ids
        ):
            raise ValueError(
                f"evidence strategy override {strategy_name!r} evidence_ids must be a string list"
            )
        unknown = set(evidence_ids) - known_source_ids
        if unknown:
            raise ValueError(
                f"evidence strategy override {strategy_name!r} references unknown sources: {sorted(unknown)}"
            )
        dedicated = override.get("dedicated_diagnostic")
        if dedicated is not None:
            if not isinstance(dedicated, Mapping):
                raise ValueError(
                    f"evidence strategy override {strategy_name!r} dedicated diagnostic "
                    "must be an object"
                )
            artifact = dedicated.get("artifact")
            windows = dedicated.get("windows")
            decision = dedicated.get("decision")
            if (
                not isinstance(artifact, Mapping)
                or not isinstance(windows, Mapping)
                or set(windows) != {"full", "recent"}
                or not isinstance(decision, Mapping)
                or dedicated.get("role") != "dedicated_research_diagnostic_not_common_scorecard"
                or dedicated.get("promotion_use") != "forbidden"
                or decision.get("gate_pass") is not False
                or decision.get("locked_action") != "cash"
            ):
                raise ValueError(
                    f"evidence strategy override {strategy_name!r} has an invalid "
                    "dedicated diagnostic contract"
                )
            relative = PurePosixPath(str(artifact.get("path") or ""))
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise ValueError(
                    f"evidence strategy override {strategy_name!r} has an unsafe "
                    "dedicated diagnostic path"
                )
            actual = _sha256(
                Path(__file__).resolve().parents[2].joinpath(*relative.parts),
                logical_path=relative.as_posix(),
            )
            if any(actual[key] != artifact.get(key) for key in ("path", "bytes", "sha256")):
                raise ValueError(
                    f"evidence strategy override {strategy_name!r} dedicated diagnostic "
                    "artifact identity mismatch"
                )
            for window, values in windows.items():
                if (
                    not isinstance(values, Mapping)
                    or not values
                    or any(
                        not isinstance(key, str)
                        or not key.endswith("_return")
                        or not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        for key, value in values.items()
                    )
                ):
                    raise ValueError(
                        f"evidence strategy override {strategy_name!r} has invalid "
                        f"dedicated diagnostic metrics for {window}"
                    )

    principles = payload.get("principles")
    hard_rejections = payload.get("hard_rejections")
    for label, values in (
        ("principles", principles),
        ("hard_rejections", hard_rejections),
    ):
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            raise ValueError(f"evidence {label} must be a non-empty string list")

    experiments_raw = payload.get("recommended_experiments")
    if not isinstance(experiments_raw, list):
        raise ValueError("evidence recommended_experiments must be a list")
    experiments: list[dict[str, Any]] = []
    experiment_ids: set[str] = set()
    for index, raw_experiment in enumerate(experiments_raw):
        if not isinstance(raw_experiment, Mapping):
            raise ValueError(f"evidence experiment at index {index} must be an object")
        experiment = dict(raw_experiment)
        for key in ("id", "change", "promotion_gate"):
            if not isinstance(experiment.get(key), str) or not str(experiment.get(key)).strip():
                raise ValueError(f"evidence experiment at index {index} requires non-empty {key}")
        existing_classes = experiment.get("existing_classes")
        if not isinstance(existing_classes, list) or any(
            not isinstance(value, str) or not value.strip() for value in existing_classes
        ):
            raise ValueError(
                f"evidence experiment {experiment['id']!r} existing_classes must be a string list"
            )
        experiment_id = str(experiment["id"])
        if experiment_id in experiment_ids:
            raise ValueError(f"duplicate evidence experiment id: {experiment_id!r}")
        experiment_ids.add(experiment_id)
        experiments.append(experiment)

    return {
        "schema_version": _EVIDENCE_SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "scope": scope,
        "principles": list(principles or ()),
        "sources": sources,
        "families": families,
        "strategy_overrides": {
            str(key): {
                **dict(value),
                "evidence_ids": sorted(set(value.get("evidence_ids") or [])),
            }
            for key, value in overrides_raw.items()
            if isinstance(value, Mapping)
        },
        "recommended_experiments": experiments,
        "hard_rejections": list(hard_rejections or ()),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "provenance": {
            "artifact_kind": "not_evaluated_in_supplied_comparison",
            "selection_provenance": "not_available",
            "full_window_role": "not_available",
            "recent_window_role": "not_available",
        },
        "full": {
            "status": "not_available",
            "return": None,
            "max_drawdown": None,
            "sharpe": None,
            "trades": None,
            "log_daily": None,
        },
        "recent": {
            "status": "not_available",
            "return": None,
            "max_drawdown": None,
            "sharpe": None,
            "trades": None,
            "log_daily": None,
        },
        "comparable": False,
        "delta_log_daily": None,
        "daily_gap": None,
        "robust_log_daily": None,
        "symbols": None,
        "tier": "",
    }


def _apply_metric_semantics(family: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        **dict(metrics),
        "full": dict(metrics.get("full") or {}),
        "recent": dict(metrics.get("recent") or {}),
        "provenance": dict(metrics.get("provenance") or {}),
    }
    if family != "rebalancing_diversification":
        return normalized
    raw_diagnostic = {
        "full": dict(normalized["full"]),
        "recent": dict(normalized["recent"]),
        "provenance": dict(normalized["provenance"]),
    }
    for window in ("full", "recent"):
        normalized[window] = {
            **normalized[window],
            "status": "matched_control_missing",
            "return": None,
            "max_drawdown": None,
            "sharpe": None,
            "log_daily": None,
        }
    normalized.update(
        {
            "comparable": False,
            "delta_log_daily": None,
            "daily_gap": None,
            "robust_log_daily": None,
            "raw_total_return_diagnostic": raw_diagnostic,
            "scorecard_exclusion_reason": (
                "Identical-basket buy-and-hold matched control is missing; raw total return "
                "cannot identify a rebalancing/diversification premium."
            ),
        }
    )
    return normalized


def build_catalog(
    *,
    strategy_map: Mapping[str, type[Any]],
    candidate_index: Mapping[str, Mapping[str, Any]],
    comparison_rows: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    generated_at: str,
    comparison_source: Mapping[str, Any],
    evidence_source: Mapping[str, Any],
) -> dict[str, Any]:
    overrides = dict(evidence.get("strategy_overrides") or {})
    evidence_families = dict(evidence.get("families") or {})
    comparison_contract = dict(comparison_source.get("contract") or {})
    if comparison_rows:
        if comparison_contract.get("artifact_kind") != _COMPARISON_ARTIFACT_KIND:
            raise ValueError("comparison rows require a validated common-period summary contract")
        if _int_or_none(comparison_contract.get("strategy_count")) != len(comparison_rows):
            raise ValueError("comparison contract count does not match comparison rows")
        if any(
            dict(row.get("provenance") or {}).get("artifact_kind") != _COMPARISON_ARTIFACT_KIND
            for row in comparison_rows.values()
        ):
            raise ValueError("comparison rows do not carry validated screen provenance")
    unknown_overrides = sorted(set(overrides) - set(strategy_map))
    if unknown_overrides:
        raise ValueError(
            f"evidence overrides reference unregistered strategies: {unknown_overrides}"
        )
    unknown_experiment_classes = sorted(
        {
            strategy_name
            for experiment in evidence.get("recommended_experiments") or []
            for strategy_name in experiment.get("existing_classes") or []
            if strategy_name not in strategy_map
        }
    )
    if unknown_experiment_classes:
        raise ValueError(
            f"evidence experiments reference unregistered strategies: {unknown_experiment_classes}"
        )
    strategies: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    for name, strategy_cls in sorted(strategy_map.items()):
        scope = strategy_scope_metadata(strategy_cls)
        if not scope["in_scope"]:
            exclusions.append({"strategy": name, **scope})
            continue
        candidates = dict(candidate_index.get(name) or {})
        candidate_families = list(candidates.get("families") or [])
        module_name = str(getattr(strategy_cls, "__module__", ""))
        family, family_source = resolve_family(name, candidate_families, overrides)
        override = dict(overrides.get(name) or {})
        family_evidence = dict(evidence_families.get(family) or {})
        evidence_ids = sorted(
            {str(value) for value in list(override.get("evidence_ids") or []) if str(value).strip()}
        )
        family_context_evidence_ids = sorted(
            {
                str(value)
                for value in list(family_evidence.get("evidence_ids") or [])
                if str(value).strip()
            }
        )
        tier = get_strategy_tier(name)
        execution_interface = GLOBAL_REGISTRY.get_interface("strategy", name) or "event_driven"
        param_count = len(get_strategy_param_schema(name))
        metrics = _apply_metric_semantics(
            family,
            comparison_rows.get(name) or _empty_metrics(),
        )
        strategies.append(
            {
                "strategy": name,
                "module": module_name,
                "tier": tier,
                "execution_interface": execution_interface,
                "runner_kind": str(
                    getattr(strategy_cls, "runner_kind", None) or "event_backtest_engine"
                ),
                "live_execution_supported": execution_interface == "event_driven"
                and tier in {"live_default", "live_opt_in"},
                "family": family,
                "family_source": family_source,
                "family_title": str(
                    family_evidence.get("title") or family.replace("_", " ").title()
                ),
                "evidence_ids": evidence_ids,
                "family_context_evidence_ids": family_context_evidence_ids,
                "cadence": scope,
                "required_features": list(
                    _string_tuple(getattr(strategy_cls, "required_features", ()))
                ),
                "uses_timeframe_aggregator": bool(
                    getattr(strategy_cls, "uses_timeframe_aggregator", False)
                ),
                "candidate_library": {
                    "families": candidate_families,
                    "timeframes": list(candidates.get("timeframes") or []),
                    "candidate_count": int(candidates.get("candidate_count") or 0),
                    "mapped": bool(candidates),
                },
                "param_count": param_count,
                "metrics": metrics,
                "dedicated_diagnostic": (
                    dict(override["dedicated_diagnostic"])
                    if isinstance(override.get("dedicated_diagnostic"), Mapping)
                    else None
                ),
                "research_note": str(override.get("note") or ""),
            }
        )

    strategy_names = {row["strategy"] for row in strategies}
    registered_names = set(strategy_map)
    excluded_names = {row["strategy"] for row in exclusions}
    unmatched_comparison = sorted(set(comparison_rows) - strategy_names)
    comparison_excluded_by_scope = sorted(set(comparison_rows) & excluded_names)
    comparison_unregistered = sorted(set(comparison_rows) - registered_names)
    if comparison_unregistered:
        raise ValueError(
            "comparison contains strategies absent from the current registry: "
            f"{comparison_unregistered}"
        )
    candidate_only = [
        {
            "strategy_class": name,
            "families": list(row.get("families") or []),
            "timeframes": list(row.get("timeframes") or []),
            "candidate_count": int(row.get("candidate_count") or 0),
            "status": "candidate_library_definition_not_registered",
        }
        for name, row in sorted(candidate_index.items())
        if name not in registered_names
    ]
    tier_counts = Counter(row["tier"] for row in strategies)
    family_counts = Counter(row["family"] for row in strategies)
    cadence_counts = Counter(row["cadence"]["cadence_status"] for row in strategies)
    scope_status_counts = Counter(row["cadence"]["scope_status"] for row in strategies)
    scope_by_strategy = {
        **{str(row["strategy"]): str(row["cadence"]["scope_status"]) for row in strategies},
        **{str(row["strategy"]): str(row["scope_status"]) for row in exclusions},
    }
    raw_registry_scorecard_rows = [
        {
            "strategy": name,
            "scope_status": scope_by_strategy[name],
            "metrics": {
                "provenance": dict(row.get("provenance") or {}),
                "full": {
                    key: dict(row.get("full") or {}).get(key)
                    for key in ("return", "sharpe", "max_drawdown")
                },
                "recent": {
                    key: dict(row.get("recent") or {}).get(key)
                    for key in ("return", "sharpe", "max_drawdown")
                },
                "comparable": bool(row.get("comparable")),
            },
        }
        for name, row in sorted(comparison_rows.items())
    ]
    scorecard_summary = _build_scorecard_summary(
        comparison_rows.values(), strategies, registry_count=len(strategy_map)
    )
    family_rows = _build_family_rows(strategies, evidence_families)
    metric_artifact_kind = str(comparison_contract.get("artifact_kind") or "not_supplied")

    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "scope": {
            "minimum_timeframe_seconds": _MIN_SCOPE_SECONDS,
            "minimum_timeframe": "1m",
            "rule": (
                "Exclude any explicit sub-minute dependency/cadence; retain strategies "
                "without explicit cadence metadata as scope_unverified for complete "
                "registry reconciliation, never as verified >=1-minute eligibility."
            ),
            "candidate_library_timeframes": list(_RESEARCH_TIMEFRAMES),
            "candidate_library_symbols": list(_CANDIDATE_SYMBOLS),
        },
        "sources": {
            "comparison": dict(comparison_source),
            "evidence": dict(evidence_source),
            "generator": _sha256(
                Path(__file__).resolve(),
                logical_path="scripts/research/build_strategy_catalog.py",
            ),
            "registry": _module_source_identity(
                "lumina_quant.strategies.registry",
                logical_path="src/lumina_quant/strategies/registry.py",
            ),
            "candidate_library": _module_source_identity(
                "lumina_quant.strategy_factory.candidate_library",
                logical_path="src/lumina_quant/strategy_factory/candidate_library.py",
            ),
        },
        "counts": {
            "registry": len(strategy_map),
            "in_scope": len(strategies),
            "verified_in_scope": scope_status_counts.get("verified_in_scope", 0),
            "scope_unverified": scope_status_counts.get("scope_unverified", 0),
            "excluded": len(exclusions),
            "comparison_rows": len(comparison_rows),
            "comparison_unmatched": len(unmatched_comparison),
            "comparison_excluded_by_scope": len(comparison_excluded_by_scope),
            "comparison_unregistered": len(comparison_unregistered),
            "candidate_only_strategy_classes": len(candidate_only),
            "unresolved_family": family_counts.get("unresolved", 0),
            "tier": dict(sorted(tier_counts.items())),
            "family": dict(sorted(family_counts.items())),
            "cadence_status": dict(sorted(cadence_counts.items())),
            "scope_status": dict(sorted(scope_status_counts.items())),
        },
        "metric_contract": {
            "artifact_kind": metric_artifact_kind,
            "selection_provenance": str(
                comparison_contract.get("selection_provenance") or "not_available"
            ),
            "timeframe": comparison_contract.get("timeframe"),
            "windows": dict(comparison_contract.get("windows") or {}),
            "recent_window": str(comparison_contract.get("recent_window_role") or "not_available"),
            "missing_values": "NA_not_zero",
            "promotion_use": "forbidden",
        },
        "scorecard_summary": scorecard_summary,
        "evidence_contract": {
            "schema_version": evidence.get("schema_version"),
            "as_of_date": evidence.get("as_of_date"),
            "scope": evidence.get("scope"),
            "principles": list(evidence.get("principles") or []),
            "recommended_experiments": list(evidence.get("recommended_experiments") or []),
            "hard_rejections": list(evidence.get("hard_rejections") or []),
        },
        "evidence_sources": list(evidence.get("sources") or []),
        "exclusions": exclusions,
        "unmatched_comparison_strategies": unmatched_comparison,
        "comparison_excluded_by_scope": comparison_excluded_by_scope,
        "comparison_unregistered_strategies": comparison_unregistered,
        "candidate_only_strategy_classes": candidate_only,
        "raw_registry_scorecard_rows": raw_registry_scorecard_rows,
        "strategies": strategies,
        "families": family_rows,
    }


def _median(values: Iterable[Any]) -> float | None:
    finite = [value for value in (_finite_float(item) for item in values) if value is not None]
    return statistics.median(finite) if finite else None


def _comparable_scorecard_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    materialized = list(rows)
    comparable: list[Mapping[str, Any]] = []
    for row in materialized:
        metrics = row.get("metrics")
        metric_row = metrics if isinstance(metrics, Mapping) else row
        if bool(metric_row.get("comparable")) and all(
            _finite_float(dict(metric_row.get(window) or {}).get(metric)) is not None
            for window in ("full", "recent")
            for metric in ("return", "sharpe", "max_drawdown")
        ):
            comparable.append(metric_row)
    return materialized, comparable


def _scorecard_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    materialized, comparable = _comparable_scorecard_rows(rows)
    return {
        "strategy_count": len(materialized),
        "comparable_count": len(comparable),
        "positive_both_count": sum(
            float(dict(row["full"])["return"]) > 0.0 and float(dict(row["recent"])["return"]) > 0.0
            for row in comparable
        ),
    }


def _build_scorecard_summary(
    comparison_rows: Iterable[Mapping[str, Any]],
    strategies: Iterable[Mapping[str, Any]],
    *,
    registry_count: int | None = None,
) -> dict[str, dict[str, Any]]:
    catalog_rows = list(strategies)
    verified_rows = [
        row
        for row in catalog_rows
        if dict(row.get("cadence") or {}).get("scope_status") == "verified_in_scope"
    ]
    raw_stats = _scorecard_stats(comparison_rows)
    evaluated_strategy_count = raw_stats["strategy_count"]
    if registry_count is not None:
        raw_stats["strategy_count"] = int(registry_count)
    raw_stats["evaluated_strategy_count"] = evaluated_strategy_count
    raw_stats["not_evaluated_strategy_count"] = max(
        0,
        raw_stats["strategy_count"] - evaluated_strategy_count,
    )
    return {
        "raw_registry_diagnostic": {
            **raw_stats,
            "scope": "registry_accounting_with_supplied_common_screen_rows",
            "matched_control_policy": "raw_total_return_not_identification",
        },
        "catalog_controlled_diagnostic": {
            **_scorecard_stats(catalog_rows),
            "scope": "explicit_sub_minute_excluded_scope_unverified_retained",
            "matched_control_policy": "unmatched_rebalancing_metrics_suppressed",
        },
        "verified_ge_1m_controlled": {
            **_scorecard_stats(verified_rows),
            "scope": "verified_in_scope_only",
            "matched_control_policy": "unmatched_rebalancing_metrics_suppressed",
        },
    }


def _build_family_rows(
    strategies: Iterable[Mapping[str, Any]], evidence_families: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in strategies:
        grouped[str(row["family"])].append(row)
    out: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        verified = [
            row
            for row in rows
            if dict(row.get("cadence") or {}).get("scope_status") == "verified_in_scope"
        ]
        _, catalog_comparable_metrics = _comparable_scorecard_rows(rows)
        verified_rows, verified_comparable_metrics = _comparable_scorecard_rows(verified)
        evidence = dict(evidence_families.get(family) or {})
        out.append(
            {
                "family": family,
                "title": str(evidence.get("title") or family.replace("_", " ").title()),
                "thesis": str(evidence.get("thesis") or ""),
                "research_decision": str(evidence.get("research_decision") or "unclassified"),
                "evidence_ids": sorted(
                    {str(value) for value in list(evidence.get("evidence_ids") or [])}
                ),
                "strategy_count": len(rows),
                "verified_strategy_count": len(verified_rows),
                "scope_unverified_strategy_count": len(rows) - len(verified_rows),
                "scorecard_scope": "verified_ge_1m_controlled",
                "comparable_count": len(verified_comparable_metrics),
                "coverage_ratio": (
                    len(verified_comparable_metrics) / len(verified_rows) if verified_rows else None
                ),
                "positive_both_count": sum(
                    float(dict(row["full"])["return"]) > 0.0
                    and float(dict(row["recent"])["return"]) > 0.0
                    for row in verified_comparable_metrics
                ),
                "catalog_diagnostic_comparable_count": len(catalog_comparable_metrics),
                "catalog_diagnostic_positive_both_count": sum(
                    float(dict(row["full"])["return"]) > 0.0
                    and float(dict(row["recent"])["return"]) > 0.0
                    for row in catalog_comparable_metrics
                ),
                "full_median_return": _median(
                    dict(row["full"]).get("return") for row in verified_comparable_metrics
                ),
                "recent_median_return": _median(
                    dict(row["recent"]).get("return") for row in verified_comparable_metrics
                ),
                "full_median_sharpe": _median(
                    dict(row["full"]).get("sharpe") for row in verified_comparable_metrics
                ),
                "recent_median_sharpe": _median(
                    dict(row["recent"]).get("sharpe") for row in verified_comparable_metrics
                ),
                "full_median_max_drawdown": _median(
                    dict(row["full"]).get("max_drawdown") for row in verified_comparable_metrics
                ),
                "recent_median_max_drawdown": _median(
                    dict(row["recent"]).get("max_drawdown") for row in verified_comparable_metrics
                ),
                "strategies": sorted(str(row["strategy"]) for row in rows),
            }
        )
    return out


def _fmt_number(value: Any, digits: int = 3) -> str:
    parsed = _finite_float(value)
    return "NA" if parsed is None else f"{parsed:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    parsed = _finite_float(value)
    return "NA" if parsed is None else f"{100.0 * parsed:.3f}%"


def _strategy_csv_rows(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in catalog["strategies"]:
        metrics = strategy["metrics"]
        raw_diagnostic = dict(metrics.get("raw_total_return_diagnostic") or {})
        raw_full = dict(raw_diagnostic.get("full") or {})
        raw_recent = dict(raw_diagnostic.get("recent") or {})
        cadence = strategy["cadence"]
        candidates = strategy["candidate_library"]
        rows.append(
            {
                "strategy": strategy["strategy"],
                "family": strategy["family"],
                "family_source": strategy["family_source"],
                "tier": strategy["tier"],
                "module": strategy["module"],
                "execution_interface": strategy["execution_interface"],
                "runner_kind": strategy["runner_kind"],
                "live_execution_supported": strategy["live_execution_supported"],
                "scope_status": cadence["scope_status"],
                "cadence_status": cadence["cadence_status"],
                "decision_cadence_seconds": cadence["decision_cadence_seconds"],
                "required_timeframes": ";".join(cadence["required_timeframes"]),
                "candidate_families": ";".join(candidates["families"]),
                "candidate_timeframes": ";".join(candidates["timeframes"]),
                "candidate_count": candidates["candidate_count"],
                "evidence_ids": ";".join(strategy["evidence_ids"]),
                "metric_artifact_kind": metrics["provenance"]["artifact_kind"],
                "metric_selection_provenance": metrics["provenance"]["selection_provenance"],
                "scorecard_exclusion_reason": metrics.get("scorecard_exclusion_reason"),
                "full_status": metrics["full"]["status"],
                "full_return": metrics["full"]["return"],
                "full_sharpe": metrics["full"]["sharpe"],
                "full_max_drawdown": metrics["full"]["max_drawdown"],
                "full_trades": metrics["full"]["trades"],
                "recent_status": metrics["recent"]["status"],
                "recent_return": metrics["recent"]["return"],
                "recent_sharpe": metrics["recent"]["sharpe"],
                "recent_max_drawdown": metrics["recent"]["max_drawdown"],
                "recent_trades": metrics["recent"]["trades"],
                "raw_diagnostic_full_return": raw_full.get("return"),
                "raw_diagnostic_recent_return": raw_recent.get("return"),
                "comparable": metrics["comparable"],
                "daily_gap": metrics["daily_gap"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _strategy_markdown(catalog: Mapping[str, Any]) -> str:
    scorecards = catalog["scorecard_summary"]
    raw = scorecards["raw_registry_diagnostic"]
    catalog_controlled = scorecards["catalog_controlled_diagnostic"]
    verified = scorecards["verified_ge_1m_controlled"]
    lines = [
        "# LuminaQuant ≥1분 전략 분류·성과 카드",
        "",
        f"- 생성: `{catalog['generated_at_utc']}`",
        f"- 레지스트리 {catalog['counts']['registry']}개 / 명시적으로 ≥1분 검증 {catalog['counts']['verified_in_scope']}개 / cadence 미명시·범위 미확인 {catalog['counts']['scope_unverified']}개 / 1분 미만 제외 {catalog['counts']['excluded']}개",
        f"- candidate-library 전용(미등록) 클래스 {catalog['counts']['candidate_only_strategy_classes']}개 / family 미해결 {catalog['counts']['unresolved_family']}개",
        "- cadence 미명시 전략은 완전한 레지스트리 회계를 위해 수록했지만 ≥1분 적격으로 검증된 것이 아닙니다.",
        "- 성과 출처: 공통 exact-1m 중첩 화면. 선택 provenance가 봉인되지 않았으므로 **독립 OOS·배포 증거가 아닙니다**.",
        "- recent는 full 내부의 cold-start 민감도이며 독립 OOS가 아닙니다. NA는 0이 아니라 미측정/비교불가입니다.",
        "- rebalancing/diversification은 동일 바스켓 buy-and-hold 대조군이 없어 raw total return을 scorecard에서 억제했습니다.",
        f"- 성과 층위: raw 공통화면 `{raw['evaluated_strategy_count']}/{raw['strategy_count']}`개 평가, 양쪽 양수/비교 가능 `{raw['positive_both_count']}/{raw['comparable_count']}`; cadence 미확인 포함 catalog-controlled 진단 `{catalog_controlled['positive_both_count']}/{catalog_controlled['comparable_count']}`; **검증된 ≥1분 controlled scorecard `{verified['positive_both_count']}/{verified['comparable_count']}`**.",
        "- Evidence 열은 해당 클래스에 직접 매핑한 근거만 표시합니다. Family 문헌은 설계 맥락이지 각 구현의 검증이 아닙니다.",
        "",
        "| Strategy | Family | Tier | Scope | Cadence | Full status | Full return | Full Sharpe | Full MDD | Full trades | Recent return | Recent Sharpe | Direct evidence |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in catalog["strategies"]:
        metrics = row["metrics"]
        cadence = row["cadence"]
        cadence_label = (
            f"{cadence['decision_cadence_seconds']}s"
            if cadence["decision_cadence_seconds"] is not None
            else cadence["cadence_status"]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["strategy"],
                    row["family"],
                    row["tier"],
                    cadence["scope_status"],
                    cadence_label,
                    metrics["full"]["status"],
                    _fmt_pct(metrics["full"].get("return")),
                    _fmt_number(metrics["full"].get("sharpe")),
                    _fmt_pct(metrics["full"].get("max_drawdown")),
                    str(
                        metrics["full"].get("trades")
                        if metrics["full"].get("trades") is not None
                        else "NA"
                    ),
                    _fmt_pct(metrics["recent"].get("return")),
                    _fmt_number(metrics["recent"].get("sharpe")),
                    ", ".join(row["evidence_ids"]) or "NA",
                ]
            )
            + " |"
        )
    if catalog["exclusions"]:
        lines.extend(
            ["", "## 범위 제외", "", "| Strategy | Reason | Required timeframes |", "|---|---|---|"]
        )
        for row in catalog["exclusions"]:
            lines.append(
                f"| {row['strategy']} | {row['exclusion_reason']} | {', '.join(row['required_timeframes']) or 'NA'} |"
            )
    if catalog["candidate_only_strategy_classes"]:
        lines.extend(
            [
                "",
                "## Candidate library 전용 클래스",
                "",
                "아래 이름은 후보 생성기에는 있으나 전략 레지스트리에는 없습니다. "
                "등록 전략 성과로 계산하지 않았습니다.",
                "",
                "| Class | Candidate families | Timeframes | Candidates |",
                "|---|---|---|---:|",
            ]
        )
        for row in catalog["candidate_only_strategy_classes"]:
            lines.append(
                f"| {row['strategy_class']} | {', '.join(row['families']) or 'NA'} | "
                f"{', '.join(row['timeframes']) or 'NA'} | {row['candidate_count']} |"
            )
    return "\n".join(lines) + "\n"


def _family_csv_rows(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "strategies"}
        for row in catalog["families"]
    ]


def _family_markdown(catalog: Mapping[str, Any]) -> str:
    lines = [
        "# LuminaQuant 전략군 성과 카드",
        "",
        "- 주 scorecard는 `scope_status=verified_in_scope`이면서 full/recent 모두 비교 가능한 유한값만 사용합니다. 실패·미측정값은 0으로 대체하지 않습니다.",
        "- cadence 미확인 행은 catalog diagnostic 열에만 남기며 검증된 ≥1분 성과에 포함하지 않습니다.",
        "- 공통기간 화면은 독립 OOS가 아니며 연구 우선순위용 진단입니다.",
        "",
        "| Family | Decision | Catalog strategies | Verified ≥1m strategies | Verified comparable | Verified positive both | Catalog diagnostic comparable | Catalog diagnostic positive both | Full median return | Recent median return | Full median Sharpe | Recent median Sharpe |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in catalog["families"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["family"],
                    row["research_decision"],
                    str(row["strategy_count"]),
                    str(row["verified_strategy_count"]),
                    str(row["comparable_count"]),
                    str(row["positive_both_count"]),
                    str(row["catalog_diagnostic_comparable_count"]),
                    str(row["catalog_diagnostic_positive_both_count"]),
                    _fmt_pct(row["full_median_return"]),
                    _fmt_pct(row["recent_median_return"]),
                    _fmt_number(row["full_median_sharpe"]),
                    _fmt_number(row["recent_median_sharpe"]),
                ]
            )
            + " |"
        )
        if row["thesis"]:
            lines.append(f"\n**{row['title']}** — {row['thesis']}\n")
    return "\n".join(lines) + "\n"


def write_artifacts(catalog: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strategy_catalog.json").write_text(_json_dump(catalog), encoding="utf-8")
    _write_csv(output_dir / "strategy_catalog.csv", _strategy_csv_rows(catalog))
    (output_dir / "strategy_scorecards.md").write_text(
        _strategy_markdown(catalog), encoding="utf-8"
    )
    _write_csv(output_dir / "family_scorecards.csv", _family_csv_rows(catalog))
    (output_dir / "family_scorecards.md").write_text(_family_markdown(catalog), encoding="utf-8")
    manifest = {
        "schema_version": "lumina_quant.strategy_catalog_manifest.v1",
        "generated_at_utc": catalog["generated_at_utc"],
        "artifacts": [_sha256(output_dir / name, logical_path=name) for name in _ARTIFACT_NAMES],
    }
    manifest_path = output_dir / "strategy_catalog_manifest.json"
    manifest_path.write_text(_json_dump(manifest), encoding="utf-8")
    return manifest


def _source_metadata(path: Path | None, *, supplied: bool, kind: str) -> dict[str, Any]:
    if path is None:
        return {"kind": kind, "status": "not_supplied", "path": None}
    return {
        "kind": kind,
        "status": "loaded" if supplied else "discovered",
        **_sha256(path, logical_path=path.name),
    }


def _derived_generated_at(
    *,
    comparison_contract: Mapping[str, Any],
    evidence: Mapping[str, Any],
    explicit: str,
) -> str:
    if explicit:
        return _canonical_utc_timestamp(explicit, label="--generated-at")
    if comparison_contract:
        return _canonical_utc_timestamp(
            comparison_contract.get("generated_at_utc"),
            label="comparison contract generated_at_utc",
        )
    as_of_date = str(evidence.get("as_of_date") or "")
    try:
        parsed = date.fromisoformat(as_of_date)
    except ValueError as exc:
        raise ValueError(
            "cannot derive deterministic generation time from evidence as_of_date"
        ) from exc
    return (
        datetime(
            parsed.year,
            parsed.month,
            parsed.day,
            tzinfo=UTC,
        )
        .isoformat()
        .replace("+00:00", "Z")
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--comparison-summary", default="")
    parser.add_argument("--evidence-json", required=True)
    parser.add_argument("--generated-at", default="")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    comparison_path = Path(args.comparison_csv).expanduser().resolve()
    summary_path = (
        Path(args.comparison_summary).expanduser().resolve()
        if args.comparison_summary
        else comparison_path.with_name("common_period_summary.json")
    )
    evidence_path = Path(args.evidence_json).expanduser().resolve()
    comparison_contract = load_comparison_contract(summary_path, comparison_path)
    comparison_rows = load_comparison_rows(comparison_path, comparison_contract)
    evidence_payload = _load_json_object(evidence_path, label="evidence")
    evidence = _normalize_evidence(evidence_payload)
    comparison_source = {
        **_source_metadata(
            comparison_path,
            supplied=True,
            kind="common_period_comparison",
        ),
        "summary": _source_metadata(
            summary_path,
            supplied=True,
            kind="common_period_summary",
        ),
        "contract": comparison_contract,
    }
    catalog = build_catalog(
        strategy_map=get_catalog_strategy_map(),
        candidate_index=_candidate_index(),
        comparison_rows=comparison_rows,
        evidence=evidence,
        generated_at=_derived_generated_at(
            comparison_contract=comparison_contract,
            evidence=evidence,
            explicit=args.generated_at,
        ),
        comparison_source=comparison_source,
        evidence_source=_source_metadata(
            evidence_path,
            supplied=True,
            kind="literature_evidence",
        ),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_artifacts(catalog, output_dir)
    print(f"[STRATEGY-CATALOG] registry={catalog['counts']['registry']}")
    print(f"[STRATEGY-CATALOG] in_scope={catalog['counts']['in_scope']}")
    print(f"[STRATEGY-CATALOG] excluded={catalog['counts']['excluded']}")
    print(f"[STRATEGY-CATALOG] saved={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
