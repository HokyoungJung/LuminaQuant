#!/usr/bin/env python3
"""Stage and safely publish the generated strategy catalog into an Obsidian vault."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import time
import unicodedata
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, cast

_SCHEMA_VERSION = "lumina_quant.obsidian_strategy_graph.v1"
_CATALOG_METRIC_ARTIFACT_KIND = "common_period_1m_cold_start_strategy_screen"
_DEFAULT_NAMESPACE = "LuminaQuant/Strategy Research Generated"
_MAX_FUTURE_SKEW_SECONDS = 300
_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _content_identity(path: Path) -> dict[str, Any]:
    identity = _sha256(path)
    return {"bytes": identity["bytes"], "sha256": identity["sha256"]}


def _bytes_identity(data: bytes) -> dict[str, Any]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _durable_replace(source: Path, destination: Path) -> None:
    source_parent = source.parent
    destination_parent = destination.parent
    os.replace(source, destination)
    _fsync_directory(source_parent)
    if destination_parent != source_parent:
        _fsync_directory(destination_parent)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(_json_dump(payload))
        handle.flush()
        os.fsync(handle.fileno())
    _durable_replace(temporary, path)


def _acquire_exclusive_flock(handle: Any) -> None:
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(0.01)


def _safe_namespace(value: str) -> PurePosixPath:
    raw = str(value).replace("\\", "/").strip()
    if raw.startswith("/"):
        raise ValueError(f"unsafe Obsidian namespace: {value!r}")
    token = raw.strip("/")
    path = PurePosixPath(token)
    if not token or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe Obsidian namespace: {value!r}")
    return path


def _safe_filename(value: Any) -> str:
    token = unicodedata.normalize("NFC", str(value).strip())
    token = re.sub(r"[\\/:*?\"<>|#\[\]]+", "-", token)
    token = re.sub(r"\s+", " ", token).strip(" .")
    if not token or token in {".", ".."}:
        raise ValueError(f"unsafe note filename: {value!r}")
    if token.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"platform-reserved note filename: {value!r}")
    return token


def _note_filename(value: Any) -> str:
    raw = unicodedata.normalize("NFC", str(value).strip())
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{_safe_filename(raw)}--{digest}"


def _note_id(kind: str, value: Any) -> str:
    token = str(kind).strip().lower().replace("_", "-")
    if not token or any(character not in "abcdefghijklmnopqrstuvwxyz-" for character in token):
        raise ValueError("note id kind is invalid")
    raw = unicodedata.normalize("NFC", str(value).strip())
    if not raw:
        raise ValueError("note id value is empty")
    return f"lq-generated-{token}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _validate_catalog(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        "lumina_quant.strategy_catalog.v1"
    ):
        raise ValueError("invalid strategy catalog schema")
    strategies = payload.get("strategies")
    families = payload.get("families")
    evidence = payload.get("evidence_sources")
    exclusions = payload.get("exclusions")
    raw_registry_rows = payload.get("raw_registry_scorecard_rows")
    counts = payload.get("counts")
    if (
        not isinstance(strategies, list)
        or not isinstance(families, list)
        or not isinstance(evidence, list)
        or not isinstance(exclusions, list)
        or not isinstance(raw_registry_rows, list)
        or not isinstance(counts, dict)
    ):
        raise ValueError("invalid strategy catalog collections")
    generated_at = str(payload.get("generated_at_utc") or "")
    try:
        parsed_generated_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("catalog generated_at_utc must be ISO-8601") from exc
    if parsed_generated_at.tzinfo is None or parsed_generated_at.utcoffset() is None:
        raise ValueError("catalog generated_at_utc must include a UTC offset")
    if parsed_generated_at.astimezone(UTC) > datetime.now(UTC) + timedelta(
        seconds=_MAX_FUTURE_SKEW_SECONDS
    ):
        raise ValueError("catalog generated_at_utc is implausibly in the future")

    names = [str(row.get("strategy") or "") for row in strategies if isinstance(row, dict)]
    family_ids = [str(row.get("family") or "") for row in families if isinstance(row, dict)]
    evidence_ids = [str(row.get("id") or "") for row in evidence if isinstance(row, dict)]
    for label, values, rows in (
        ("strategy", names, strategies),
        ("family", family_ids, families),
        ("evidence", evidence_ids, evidence),
    ):
        if (
            len(values) != len(rows)
            or any(not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"catalog {label} ids must be non-empty and unique")
    excluded_names = [str(row.get("strategy") or "") for row in exclusions if isinstance(row, dict)]
    if (
        len(excluded_names) != len(exclusions)
        or any(not value for value in excluded_names)
        or len(excluded_names) != len(set(excluded_names))
        or set(excluded_names) & set(names)
    ):
        raise ValueError("catalog exclusions must be unique and disjoint from strategies")
    if int(counts.get("in_scope", -1)) != len(strategies) or int(counts.get("excluded", -1)) != len(
        exclusions
    ):
        raise ValueError("catalog counts do not match strategy/exclusion rows")
    if int(counts.get("registry", -1)) != len(strategies) + len(exclusions):
        raise ValueError("catalog registry count does not match scope partition")
    verified_in_scope = int(counts.get("verified_in_scope", -1))
    scope_unverified = int(counts.get("scope_unverified", -1))
    if (
        verified_in_scope < 0
        or scope_unverified < 0
        or verified_in_scope + scope_unverified != len(strategies)
    ):
        raise ValueError("catalog verified/unverified scope counts do not partition strategies")
    scorecards = payload.get("scorecard_summary")
    required_scorecards = {
        "raw_registry_diagnostic": int(counts["registry"]),
        "catalog_controlled_diagnostic": len(strategies),
        "verified_ge_1m_controlled": verified_in_scope,
    }
    if not isinstance(scorecards, dict):
        raise ValueError("catalog scorecard_summary must be an object")
    if set(scorecards) != set(required_scorecards):
        raise ValueError("catalog scorecard_summary must define all scorecard layers")
    for label, expected_strategy_count in required_scorecards.items():
        layer = scorecards.get(label)
        if not isinstance(layer, dict):
            raise ValueError(f"catalog scorecard layer {label!r} must be an object")
        values = [
            layer.get(field)
            for field in (
                "strategy_count",
                "comparable_count",
                "positive_both_count",
            )
        ]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError(f"catalog scorecard layer {label!r} has invalid counts or scope")
        strategy_count, comparable_count, positive_both_count = cast(list[int], values)
        if (
            strategy_count != expected_strategy_count
            or not 0 <= positive_both_count <= comparable_count <= strategy_count
            or not str(layer.get("scope") or "")
            or not str(layer.get("matched_control_policy") or "")
        ):
            raise ValueError(f"catalog scorecard layer {label!r} has invalid counts or scope")
        if label == "raw_registry_diagnostic":
            evaluated = layer.get("evaluated_strategy_count")
            not_evaluated = layer.get("not_evaluated_strategy_count")
            if (
                not isinstance(evaluated, int)
                or isinstance(evaluated, bool)
                or not isinstance(not_evaluated, int)
                or isinstance(not_evaluated, bool)
                or evaluated != int(counts.get("comparison_rows", -1))
                or evaluated + not_evaluated != strategy_count
            ):
                raise ValueError("catalog raw scorecard does not reconcile evaluated rows")

    family_set = set(family_ids)
    evidence_set = set(evidence_ids)
    required_evidence_fields = {
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
    for row in evidence:
        missing = sorted(
            key
            for key in required_evidence_fields
            if not isinstance(row.get(key), str) or not str(row.get(key)).strip()
        )
        if missing or not str(row.get("url") or "").startswith("https://"):
            raise ValueError(f"catalog evidence {row.get('id')!r} has invalid fields: {missing}")
    metric_contract = payload.get("metric_contract")
    if not isinstance(metric_contract, dict) or (
        metric_contract.get("selection_provenance") != "unsealed"
        or metric_contract.get("promotion_use") != "forbidden"
        or metric_contract.get("artifact_kind") != _CATALOG_METRIC_ARTIFACT_KIND
        or metric_contract.get("timeframe") != "1m"
    ):
        raise ValueError(
            "catalog metric contract must be validated exact-1m, unsealed, and promotion-forbidden"
        )
    for row in strategies:
        family = str(row.get("family") or "")
        if family not in family_set:
            raise ValueError(f"strategy references unknown family: {row.get('strategy')}")
        execution_interface = row.get("execution_interface")
        runner_kind = row.get("runner_kind")
        live_execution_supported = row.get("live_execution_supported")
        if execution_interface not in {"event_driven", "polars_batch"}:
            raise ValueError(f"strategy {row.get('strategy')!r} has invalid execution interface")
        if not isinstance(runner_kind, str) or not runner_kind.strip():
            raise ValueError(f"strategy {row.get('strategy')!r} has invalid runner kind")
        if not isinstance(live_execution_supported, bool):
            raise ValueError(f"strategy {row.get('strategy')!r} has invalid live support flag")
        if live_execution_supported and (
            execution_interface != "event_driven"
            or row.get("tier") not in {"live_default", "live_opt_in"}
        ):
            raise ValueError(f"strategy {row.get('strategy')!r} has unsafe live support claim")
        dedicated = row.get("dedicated_diagnostic")
        if dedicated is not None:
            if (
                not isinstance(dedicated, dict)
                or dedicated.get("role") != "dedicated_research_diagnostic_not_common_scorecard"
                or dedicated.get("promotion_use") != "forbidden"
                or not isinstance(dedicated.get("artifact"), dict)
                or not isinstance(dedicated.get("windows"), dict)
                or set(dedicated["windows"]) != {"full", "recent"}
                or not isinstance(dedicated.get("decision"), dict)
                or dedicated["decision"].get("gate_pass") is not False
                or dedicated["decision"].get("locked_action") != "cash"
            ):
                raise ValueError(
                    f"strategy {row.get('strategy')!r} has invalid dedicated diagnostic"
                )
            artifact = dedicated["artifact"]
            relative = PurePosixPath(str(artifact.get("path") or ""))
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise ValueError(
                    f"strategy {row.get('strategy')!r} has unsafe diagnostic artifact path"
                )
            actual = _sha256(Path(__file__).resolve().parents[2].joinpath(*relative.parts))
            if any(actual[key] != artifact.get(key) for key in ("bytes", "sha256")):
                raise ValueError(
                    f"strategy {row.get('strategy')!r} diagnostic artifact identity mismatch"
                )
            for values in dedicated["windows"].values():
                if (
                    not isinstance(values, dict)
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
                        f"strategy {row.get('strategy')!r} has invalid diagnostic metrics"
                    )
        cadence = row.get("cadence")
        if not isinstance(cadence, dict) or cadence.get("scope_status") not in {
            "verified_in_scope",
            "scope_unverified",
        }:
            raise ValueError(f"strategy {row.get('strategy')!r} has invalid scope status")
        direct_evidence = row.get("evidence_ids")
        context_evidence = row.get("family_context_evidence_ids")
        for label, values in (
            ("direct", direct_evidence),
            ("family-context", context_evidence),
        ):
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(
                    f"strategy {row.get('strategy')!r} has invalid {label} evidence ids"
                )
        unknown_evidence = set(direct_evidence) - evidence_set
        if unknown_evidence:
            raise ValueError(
                f"strategy {row.get('strategy')!r} references unknown evidence: "
                f"{sorted(unknown_evidence)}"
            )
        unknown_context = set(context_evidence) - evidence_set
        if unknown_context:
            raise ValueError(
                f"strategy {row.get('strategy')!r} references unknown family-context "
                f"evidence: {sorted(unknown_context)}"
            )
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("provenance"), dict):
            raise ValueError(f"strategy {row.get('strategy')!r} has invalid metric provenance")
        artifact_kind = str(metrics["provenance"].get("artifact_kind") or "")
        if artifact_kind == "not_evaluated_in_supplied_comparison":
            if bool(metrics.get("comparable")) or any(
                dict(metrics.get(window) or {}).get("status") != "not_available"
                or any(
                    dict(metrics.get(window) or {}).get(field) is not None
                    for field in ("return", "sharpe", "max_drawdown", "trades")
                )
                for window in ("full", "recent")
            ):
                raise ValueError(f"strategy {row.get('strategy')!r} has unsealed non-NA metrics")
        elif artifact_kind != _CATALOG_METRIC_ARTIFACT_KIND:
            raise ValueError(f"strategy {row.get('strategy')!r} lacks validated screen provenance")
        for window in ("full", "recent"):
            block = metrics.get(window)
            if not isinstance(block, dict) or not str(block.get("status") or ""):
                raise ValueError(f"strategy {row.get('strategy')!r} has invalid {window} metrics")
            for field in ("return", "sharpe", "max_drawdown"):
                value = block.get(field)
                if value is not None and (
                    not isinstance(value, (int, float)) or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"strategy {row.get('strategy')!r} has non-finite {window}.{field}"
                    )
            trades = block.get("trades")
            if trades is not None and (
                not isinstance(trades, int) or isinstance(trades, bool) or trades < 0
            ):
                raise ValueError(f"strategy {row.get('strategy')!r} has invalid {window}.trades")
        if not isinstance(metrics.get("comparable"), bool):
            raise ValueError(f"strategy {row.get('strategy')!r} has invalid comparable flag")

    actual_verified = sum(
        row["cadence"]["scope_status"] == "verified_in_scope" for row in strategies
    )
    actual_unverified = len(strategies) - actual_verified
    if (actual_verified, actual_unverified) != (verified_in_scope, scope_unverified):
        raise ValueError("catalog scope counts do not match strategy rows")

    def comparable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row["metrics"]["comparable"]
            and all(
                row["metrics"][window].get(metric) is not None
                for window in ("full", "recent")
                for metric in ("return", "sharpe", "max_drawdown")
            )
        ]

    def current_scorecard(rows: list[dict[str, Any]]) -> tuple[int, int]:
        comparable = comparable_rows(rows)
        return len(comparable), sum(
            float(row["metrics"]["full"]["return"]) > 0.0
            and float(row["metrics"]["recent"]["return"]) > 0.0
            for row in comparable
        )

    scope_by_strategy = {
        **{str(row["strategy"]): str(row["cadence"]["scope_status"]) for row in strategies},
        **{str(row["strategy"]): str(row.get("scope_status") or "") for row in exclusions},
    }
    raw_names: list[str] = []
    for raw_row in raw_registry_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("catalog raw scorecard rows must be objects")
        strategy = str(raw_row.get("strategy") or "")
        raw_names.append(strategy)
        if not strategy or scope_by_strategy.get(strategy) != raw_row.get("scope_status"):
            raise ValueError("catalog raw scorecard row has invalid strategy scope")
        metrics = raw_row.get("metrics")
        provenance = dict(metrics.get("provenance") or {}) if isinstance(metrics, dict) else {}
        if (
            not isinstance(metrics, dict)
            or provenance.get("artifact_kind") != _CATALOG_METRIC_ARTIFACT_KIND
            or not isinstance(metrics.get("comparable"), bool)
        ):
            raise ValueError("catalog raw scorecard row has invalid metric provenance")
        for window in ("full", "recent"):
            block = metrics.get(window)
            if not isinstance(block, dict):
                raise ValueError("catalog raw scorecard row has invalid metric window")
            for field in ("return", "sharpe", "max_drawdown"):
                value = block.get(field)
                if value is not None and (
                    not isinstance(value, (int, float)) or not math.isfinite(float(value))
                ):
                    raise ValueError("catalog raw scorecard row has non-finite metrics")
    if (
        len(raw_names) != int(scorecards["raw_registry_diagnostic"]["evaluated_strategy_count"])
        or len(raw_names) != int(counts.get("comparison_rows", -1))
        or len(raw_names) != len(set(raw_names))
        or not set(raw_names) <= set(scope_by_strategy)
    ):
        raise ValueError("catalog raw scorecard rows do not reconcile comparison coverage")
    raw_counts = current_scorecard(raw_registry_rows)
    raw_layer = scorecards["raw_registry_diagnostic"]
    if raw_counts != (raw_layer["comparable_count"], raw_layer["positive_both_count"]):
        raise ValueError("catalog raw scorecard summary does not match raw rows")

    catalog_counts = current_scorecard(strategies)
    verified_counts = current_scorecard(
        [row for row in strategies if row["cadence"]["scope_status"] == "verified_in_scope"]
    )
    for label, actual in (
        ("catalog_controlled_diagnostic", catalog_counts),
        ("verified_ge_1m_controlled", verified_counts),
    ):
        layer = scorecards[label]
        if actual != (layer["comparable_count"], layer["positive_both_count"]):
            raise ValueError(f"catalog scorecard layer {label!r} does not match strategy rows")
    listed_strategy_families: dict[str, str] = {}
    for row in families:
        family = str(row.get("family") or "")
        family_evidence = row.get("evidence_ids")
        listed_strategies = row.get("strategies")
        if (
            not isinstance(family_evidence, list)
            or any(not isinstance(value, str) or not value for value in family_evidence)
            or len(family_evidence) != len(set(family_evidence))
        ):
            raise ValueError(f"family {family!r} has invalid evidence ids")
        if (
            not isinstance(listed_strategies, list)
            or any(not isinstance(value, str) or not value for value in listed_strategies)
            or len(listed_strategies) != len(set(listed_strategies))
            or int(row.get("strategy_count", -1)) != len(listed_strategies)
        ):
            raise ValueError(f"family {family!r} has invalid strategy membership")
        unknown_evidence = set(family_evidence) - evidence_set
        if unknown_evidence:
            raise ValueError(
                f"family {family!r} references unknown evidence: {sorted(unknown_evidence)}"
            )
        unknown_strategies = set(listed_strategies) - set(names)
        if unknown_strategies:
            raise ValueError(
                f"family {family!r} references unknown strategies: {sorted(unknown_strategies)}"
            )
        for strategy in listed_strategies:
            if strategy in listed_strategy_families:
                raise ValueError(f"strategy {strategy!r} is listed in multiple families")
            listed_strategy_families[strategy] = family
    expected_strategy_families = {str(row["strategy"]): str(row["family"]) for row in strategies}
    if listed_strategy_families != expected_strategy_families:
        raise ValueError("catalog family membership does not match strategy rows")

    def same_optional_number(actual: Any, expected: float | None) -> bool:
        if expected is None:
            return actual is None
        return (
            isinstance(actual, (int, float))
            and math.isfinite(float(actual))
            and math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )

    strategy_by_name = {str(row["strategy"]): row for row in strategies}
    for family_row in families:
        member_rows = [strategy_by_name[name] for name in family_row["strategies"]]
        if family_row.get("scorecard_scope") != "verified_ge_1m_controlled":
            raise ValueError(f"family {family_row.get('family')!r} has invalid scorecard scope")
        verified_rows = [
            row for row in member_rows if row["cadence"]["scope_status"] == "verified_in_scope"
        ]
        verified_comparable = comparable_rows(verified_rows)
        catalog_comparable = comparable_rows(member_rows)
        expected_counts = {
            "verified_strategy_count": len(verified_rows),
            "scope_unverified_strategy_count": len(member_rows) - len(verified_rows),
            "comparable_count": len(verified_comparable),
            "positive_both_count": current_scorecard(verified_rows)[1],
            "catalog_diagnostic_comparable_count": len(catalog_comparable),
            "catalog_diagnostic_positive_both_count": current_scorecard(member_rows)[1],
        }
        if any(int(family_row.get(key, -1)) != value for key, value in expected_counts.items()):
            raise ValueError(
                f"family {family_row.get('family')!r} scorecard counts do not match strategy rows"
            )
        expected_coverage = len(verified_comparable) / len(verified_rows) if verified_rows else None
        if not same_optional_number(family_row.get("coverage_ratio"), expected_coverage):
            raise ValueError(
                f"family {family_row.get('family')!r} coverage does not match strategy rows"
            )
        for field, window, metric in (
            ("full_median_return", "full", "return"),
            ("recent_median_return", "recent", "return"),
            ("full_median_sharpe", "full", "sharpe"),
            ("recent_median_sharpe", "recent", "sharpe"),
            ("full_median_max_drawdown", "full", "max_drawdown"),
            ("recent_median_max_drawdown", "recent", "max_drawdown"),
        ):
            values = [float(row["metrics"][window][metric]) for row in verified_comparable]
            expected_median = statistics.median(values) if values else None
            if not same_optional_number(family_row.get(field), expected_median):
                raise ValueError(
                    f"family {family_row.get('family')!r} {field} does not match strategy rows"
                )
    return dict(payload)


def _load_catalog(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"strategy catalog file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid strategy catalog JSON ({path}): {exc}") from exc
    return _validate_catalog(payload)


def _load_catalog_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"strategy catalog file not found: {path}")
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strategy catalog JSON ({path}): {exc}") from exc
    return _validate_catalog(payload), _bytes_identity(data)


def _frontmatter(values: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        elif isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {json.dumps(str(item), ensure_ascii=False)}" for item in value)
            continue
        else:
            rendered = json.dumps(str(value), ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    lines.extend(["---", ""])
    return "\n".join(lines)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{100.0 * float(value):.3f}%"
    except TypeError, ValueError:
        return "NA"


def _fmt_number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.3f}"
    except TypeError, ValueError:
        return "NA"


def _note_target(namespace: PurePosixPath, category: str, name: Any) -> str:
    return str(namespace / category / _note_filename(name))


def _assert_unique_note_filenames(rows: list[dict[str, Any]], *, key: str, category: str) -> None:
    names = [_note_filename(row.get(key)) for row in rows]
    normalized = [unicodedata.normalize("NFC", name).casefold() for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{category} note filenames collide after sanitization")


def _index_target(namespace: PurePosixPath) -> str:
    return str(namespace / "Strategy Research Index")


def _strategy_note(row: dict[str, Any], *, namespace: PurePosixPath, generated_at: str) -> str:
    strategy = str(row["strategy"])
    family = str(row["family"])
    metrics = dict(row.get("metrics") or {})
    full = dict(metrics.get("full") or {})
    recent = dict(metrics.get("recent") or {})
    cadence = dict(row.get("cadence") or {})
    candidate = dict(row.get("candidate_library") or {})
    dedicated = (
        dict(row.get("dedicated_diagnostic") or {})
        if isinstance(row.get("dedicated_diagnostic"), dict)
        else {}
    )
    evidence_ids = list(row.get("evidence_ids") or [])
    family_context_evidence_ids = list(row.get("family_context_evidence_ids") or [])
    lines = [
        _frontmatter(
            {
                "id": _note_id("strategy", strategy),
                "lq_generated": True,
                "lq_type": "strategy",
                "strategy": strategy,
                "family": family,
                "tier": row.get("tier"),
                "execution_interface": row.get("execution_interface"),
                "runner_kind": row.get("runner_kind"),
                "live_execution_supported": row.get("live_execution_supported"),
                "scope_status": cadence.get("scope_status"),
                "cadence_status": cadence.get("cadence_status"),
                "full_return": full.get("return"),
                "full_sharpe": full.get("sharpe"),
                "full_max_drawdown": full.get("max_drawdown"),
                "recent_return": recent.get("return"),
                "recent_sharpe": recent.get("sharpe"),
                "metric_provenance": (metrics.get("provenance") or {}).get("artifact_kind"),
                "dedicated_diagnostic_sha256": (
                    dict(dedicated.get("artifact") or {}).get("sha256") if dedicated else None
                ),
                "generated_at_utc": generated_at,
            }
        ),
        f"# {strategy}",
        "",
        f"- 전략군: [[{_note_target(namespace, 'Families', family)}|{family}]]",
        f"- Tier: `{row.get('tier', '')}`",
        f"- 모듈: `{row.get('module', '')}`",
        f"- execution interface: `{row.get('execution_interface', '')}`",
        f"- runner kind: `{row.get('runner_kind', '')}`",
        f"- live execution supported: `{bool(row.get('live_execution_supported'))}`",
        f"- scope: `{cadence.get('scope_status', 'scope_unverified')}`",
        f"- cadence: `{cadence.get('decision_cadence_seconds') or cadence.get('cadence_status', 'unknown')}`",
        f"- required timeframes: `{', '.join(cadence.get('required_timeframes') or []) or 'unknown'}`",
        f"- required features: `{', '.join(row.get('required_features') or []) or 'OHLCV/unspecified'}`",
        "",
        "## 공통기간 성적",
        "",
        "이 수치는 exact-1m 공통기간 진단 화면이며, 선택 provenance가 봉인되지 않아 독립 OOS 또는 배포 증거가 아닙니다. Recent는 full 내부 cold-start 민감도입니다.",
        "",
        "| Window | Status | Return | Sharpe | MDD | Trades |",
        "|---|---|---:|---:|---:|---:|",
        f"| Full | {full.get('status', 'NA')} | {_fmt_pct(full.get('return'))} | {_fmt_number(full.get('sharpe'))} | {_fmt_pct(full.get('max_drawdown'))} | {full.get('trades') if full.get('trades') is not None else 'NA'} |",
        f"| Recent nested sensitivity | {recent.get('status', 'NA')} | {_fmt_pct(recent.get('return'))} | {_fmt_number(recent.get('sharpe'))} | {_fmt_pct(recent.get('max_drawdown'))} | {recent.get('trades') if recent.get('trades') is not None else 'NA'} |",
        "",
        "## 후보 라이브러리 연결",
        "",
        f"- mapped: `{bool(candidate.get('mapped'))}` / candidates: `{candidate.get('candidate_count', 0)}`",
        f"- families: `{', '.join(candidate.get('families') or []) or 'none'}`",
        f"- timeframes: `{', '.join(candidate.get('timeframes') or []) or 'none'}`",
        "",
        "## 전략 직접 근거",
        "",
    ]
    if family in {"derivatives_carry", "derivatives_directional_crowding"}:
        index = lines.index("## 공통기간 성적")
        lines[index:index] = [
            "> [!warning] Carry 명칭 주의",
            "> 이 클래스는 단일-leg 방향성 funding/crowding 신호이며, "
            "long-spot/short-perp delta-neutral carry 또는 arbitrage가 아닙니다.",
            "",
        ]
    if row.get("execution_interface") == "polars_batch" and row.get("tier") == "research_only":
        index = lines.index("## 공통기간 성적")
        lines[index:index] = [
            "> [!warning] 전용 연구 러너",
            "> 이 전략은 polars_batch research-only이며 이벤트/라이브 실행을 지원하지 않습니다. "
            f"전용 러너 `{row.get('runner_kind')}`만 사용합니다.",
            "",
        ]
    if cadence.get("scope_status") == "scope_unverified":
        index = lines.index("## 공통기간 성적")
        lines[index:index] = [
            "> [!warning] ≥1분 범위 미검증",
            "> 이 클래스는 cadence/timeframe metadata가 없어 아래 수치를 검증된 ≥1분 "
            "scorecard와 family 집계에 포함하지 않았습니다.",
            "",
        ]
    if family == "rebalancing_diversification":
        index = lines.index("## 공통기간 성적")
        lines[index:index] = [
            "> [!warning] Matched-control 누락",
            "> 동일 초기 바스켓 buy-and-hold 대조군이 없어 raw total return을 성과에서 억제했습니다. "
            "현재 값으로 rebalancing premium을 주장할 수 없습니다.",
            "",
        ]
    if dedicated:
        artifact = dict(dedicated["artifact"])
        windows = dict(dedicated["windows"])
        full_diagnostic = dict(windows["full"])
        recent_diagnostic = dict(windows["recent"])
        metric_names = sorted(set(full_diagnostic) | set(recent_diagnostic))
        diagnostic_lines = [
            "## 전용 연구 진단",
            "",
            "이 표는 공통 exact-1m scorecard가 아니라 별도 전용 러너의 해시 고정 진단입니다. "
            "선택·승격·실행에는 사용할 수 없습니다.",
            "",
            f"- artifact: `{artifact['path']}`",
            f"- SHA-256: `{artifact['sha256']}` / bytes: `{artifact['bytes']}`",
            f"- gate/action: `{dedicated['decision']['gate_pass']}` / "
            f"`{dedicated['decision']['locked_action']}`",
            "",
            "| Metric | Full | Recent |",
            "|---|---:|---:|",
            *[
                f"| {metric.replace('_', ' ')} | "
                f"{_fmt_pct(full_diagnostic.get(metric))} | "
                f"{_fmt_pct(recent_diagnostic.get(metric))} |"
                for metric in metric_names
            ],
            "",
        ]
        index = lines.index("## 후보 라이브러리 연결")
        lines[index:index] = diagnostic_lines
    if evidence_ids:
        lines.extend(
            f"- [[{_note_target(namespace, 'Evidence', evidence_id)}|{evidence_id}]]"
            for evidence_id in evidence_ids
        )
    else:
        lines.append("- 직접 매핑된 근거 없음. 전략군 문헌은 이 구현 자체의 검증이 아닙니다.")
    lines.extend(["", "## 전략군 설계 맥락", ""])
    if family_context_evidence_ids:
        lines.extend(
            f"- [[{_note_target(namespace, 'Evidence', evidence_id)}|{evidence_id}]]"
            for evidence_id in family_context_evidence_ids
        )
    else:
        lines.append("- 매핑된 전략군 맥락 근거 없음")
    if row.get("research_note"):
        lines.extend(["", "## 분류 메모", "", str(row["research_note"])])
    lines.extend(["", f"- 상위 인덱스: [[{_index_target(namespace)}]]", ""])
    return "\n".join(lines)


def _family_note(row: dict[str, Any], *, namespace: PurePosixPath, generated_at: str) -> str:
    family = str(row["family"])
    lines = [
        _frontmatter(
            {
                "id": _note_id("strategy_family", family),
                "lq_generated": True,
                "lq_type": "strategy_family",
                "family": family,
                "strategy_count": row.get("strategy_count"),
                "verified_strategy_count": row.get("verified_strategy_count"),
                "comparable_count": row.get("comparable_count"),
                "scorecard_scope": row.get("scorecard_scope"),
                "research_decision": row.get("research_decision"),
                "generated_at_utc": generated_at,
            }
        ),
        f"# {row.get('title') or family}",
        "",
        f"- family id: `{family}`",
        f"- 연구 판단: `{row.get('research_decision', 'unclassified')}`",
        f"- 카탈로그 전략 {row.get('strategy_count', 0)}개 / 명시적으로 ≥1분 검증 {row.get('verified_strategy_count', 0)}개 / 검증 범위 full+recent 비교 가능 {row.get('comparable_count', 0)}개 / 양쪽 양수 {row.get('positive_both_count', 0)}개",
        f"- cadence 미확인 포함 catalog diagnostic: 비교 가능 {row.get('catalog_diagnostic_comparable_count', 0)}개 / 양쪽 양수 {row.get('catalog_diagnostic_positive_both_count', 0)}개",
        "- 성과 집계는 비교 가능한 유한값만 사용하며 NA/실패를 0으로 바꾸지 않습니다.",
        "- Family 중앙값과 주 scorecard는 `scope_status=verified_in_scope`만 사용합니다.",
        "- 공통기간 화면은 독립 OOS가 아닙니다.",
        "",
        "## 논문·온라인 근거가 지지하는 범위",
        "",
        str(row.get("thesis") or "검증된 family thesis가 아직 없습니다."),
        "",
        "## Family 성적",
        "",
        "| Metric | Full | Recent nested sensitivity |",
        "|---|---:|---:|",
        f"| Median return | {_fmt_pct(row.get('full_median_return'))} | {_fmt_pct(row.get('recent_median_return'))} |",
        f"| Median Sharpe | {_fmt_number(row.get('full_median_sharpe'))} | {_fmt_number(row.get('recent_median_sharpe'))} |",
        f"| Median MDD | {_fmt_pct(row.get('full_median_max_drawdown'))} | {_fmt_pct(row.get('recent_median_max_drawdown'))} |",
        "",
        "## 전략",
        "",
    ]
    lines.extend(
        f"- [[{_note_target(namespace, 'Strategies', strategy)}|{strategy}]]"
        for strategy in list(row.get("strategies") or [])
    )
    lines.extend(["", "## 근거", ""])
    evidence_ids = list(row.get("evidence_ids") or [])
    if evidence_ids:
        lines.extend(
            f"- [[{_note_target(namespace, 'Evidence', evidence_id)}|{evidence_id}]]"
            for evidence_id in evidence_ids
        )
    else:
        lines.append("- 매핑된 근거 없음")
    lines.extend(["", f"- 상위 인덱스: [[{_index_target(namespace)}]]", ""])
    return "\n".join(lines)


def _evidence_note(
    source: dict[str, Any],
    *,
    namespace: PurePosixPath,
    generated_at: str,
    families: list[str],
    strategies: list[str],
    contextual_strategies: list[str],
) -> str:
    evidence_id = str(source["id"])
    lines = [
        _frontmatter(
            {
                "id": _note_id("evidence", evidence_id),
                "lq_generated": True,
                "lq_type": "evidence",
                "evidence_id": evidence_id,
                "grade": source.get("grade"),
                "evidence_type": source.get("type"),
                "generated_at_utc": generated_at,
            }
        ),
        f"# {source.get('title') or evidence_id}",
        "",
        f"- id: `{evidence_id}`",
        f"- grade/type: `{source.get('grade', 'NA')}` / `{source.get('type', 'NA')}`",
        f"- publication: `{source.get('publication', 'NA')}`",
        f"- date: `{source.get('date', 'NA')}`",
        f"- URL: {source.get('url', 'NA')}",
        "",
        "## 지지 범위",
        "",
        str(source.get("supports") or "명시되지 않음"),
        "",
        "## 제한",
        "",
        str(source.get("limitations") or "외부 성과는 LuminaQuant 성과가 아닙니다."),
        "",
        "## 연결 전략군",
        "",
    ]
    lines.extend(
        f"- [[{_note_target(namespace, 'Families', family)}|{family}]]" for family in families
    )
    lines.extend(["", "## 연결 전략", ""])
    if strategies:
        lines.extend(
            f"- [[{_note_target(namespace, 'Strategies', strategy)}|{strategy}]]"
            for strategy in strategies
        )
    else:
        lines.append("- 직접 매핑된 전략 없음")
    lines.extend(["", "## 전략군 맥락으로 연결된 전략", ""])
    if contextual_strategies:
        lines.extend(
            f"- [[{_note_target(namespace, 'Strategies', strategy)}|{strategy}]]"
            for strategy in contextual_strategies
        )
    else:
        lines.append("- 전략군 맥락 연결 없음")
    lines.extend(["", f"- 상위 인덱스: [[{_index_target(namespace)}]]", ""])
    return "\n".join(lines)


def _index_note(catalog: dict[str, Any], *, namespace: PurePosixPath, generated_at: str) -> str:
    scorecards = catalog["scorecard_summary"]
    raw = scorecards["raw_registry_diagnostic"]
    catalog_controlled = scorecards["catalog_controlled_diagnostic"]
    verified = scorecards["verified_ge_1m_controlled"]
    lines = [
        _frontmatter(
            {
                "id": "lq-generated-strategy-research-index",
                "lq_generated": True,
                "lq_type": "strategy_research_index",
                "strategy_count": catalog["counts"]["in_scope"],
                "family_count": len(catalog["families"]),
                "evidence_count": len(catalog.get("evidence_sources") or []),
                "generated_at_utc": generated_at,
            }
        ),
        "# LuminaQuant ≥1분 전략 연구 그래프",
        "",
        f"- 레지스트리 `{catalog['counts']['registry']}` / 명시적으로 ≥1분 검증 `{catalog['counts'].get('verified_in_scope', 'NA')}` / cadence 미명시·범위 미확인 `{catalog['counts'].get('scope_unverified', 'NA')}` / 1분 미만 제외 `{catalog['counts']['excluded']}`",
        "- cadence 미명시 전략은 완전한 레지스트리 회계를 위해 수록했지만 ≥1분 적격으로 검증된 것이 아닙니다.",
        "- 문헌·온라인 결과는 설계 prior입니다. 아래 성적만 현재 공급된 LuminaQuant 진단 artifact에서 왔습니다.",
        "- 성적 화면은 selection provenance가 봉인되지 않은 중첩 공통기간 화면이며 독립 OOS/배포 증거가 아닙니다.",
        "- **현재 결정은 `do_not_promote / research_only_no_execution`입니다. `live_execution_supported`는 인터페이스 능력 표시일 뿐 주문·배분 승인이나 실행 권한이 아닙니다.**",
        f"- 성과 층위: raw 공통화면 `{raw['evaluated_strategy_count']}/{raw['strategy_count']}`개 평가, 양쪽 양수/비교 가능 `{raw['positive_both_count']}/{raw['comparable_count']}`; cadence 미확인 포함 catalog-controlled `{catalog_controlled['positive_both_count']}/{catalog_controlled['comparable_count']}`; **검증된 ≥1분 controlled `{verified['positive_both_count']}/{verified['comparable_count']}`**.",
        "",
        "## 전략군",
        "",
    ]
    lines.extend(
        f"- [[{_note_target(namespace, 'Families', row['family'])}|{row.get('title') or row['family']}]] — catalog {row['strategy_count']}개 / verified ≥1m {row['verified_strategy_count']}개 / verified 비교 가능 {row['comparable_count']}개"
        for row in catalog["families"]
    )
    lines.extend(["", "## 근거", ""])
    lines.extend(
        f"- [[{_note_target(namespace, 'Evidence', row['id'])}|{row.get('title') or row['id']}]]"
        for row in catalog.get("evidence_sources") or []
    )
    lines.extend(["", "## 범위 제외", ""])
    for row in catalog.get("exclusions") or []:
        lines.append(f"- `{row['strategy']}` — `{row['exclusion_reason']}`")
    lines.append("")
    return "\n".join(lines)


def _assert_contained_without_symlinks(root: Path, target: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"root must be a real directory, not a symlink: {root}")
    root_absolute = root.absolute()
    target_absolute = target.absolute()
    try:
        relative = target_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {target}") from exc
    resolved_root = root.resolve(strict=True)
    current = root_absolute
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"refusing symlinked path component: {current}")
        if current.exists() and not current.resolve(strict=True).is_relative_to(resolved_root):
            raise ValueError(f"resolved path escapes root: {current}")


def _assert_no_symlink_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"generated tree root must be a real directory: {root}")
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*dirnames, *filenames]:
            path = base / name
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError(f"generated tree contains a symlink: {path}")


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve(strict=False)
    second_resolved = second.resolve(strict=False)
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def validate_graph(namespace_root: Path, namespace: PurePosixPath) -> dict[str, Any]:
    namespace = _safe_namespace(str(namespace))
    _assert_no_symlink_tree(namespace_root)
    note_paths = sorted(namespace_root.rglob("*.md"))
    targets = {
        str(namespace / path.relative_to(namespace_root).with_suffix("")) for path in note_paths
    }
    broken: list[dict[str, str]] = []
    link_count = 0
    for path in note_paths:
        text = path.read_text(encoding="utf-8")
        for target in _LINK_RE.findall(text):
            link_count += 1
            normalized = str(PurePosixPath(target.strip()))
            if normalized not in targets:
                broken.append(
                    {"source": str(path.relative_to(namespace_root)), "target": normalized}
                )
    if broken:
        raise ValueError(f"generated Obsidian graph contains broken links: {broken[:5]}")
    return {"note_count": len(note_paths), "link_count": link_count, "broken_link_count": 0}


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    identity = _content_identity(path)
    return {"path": path.relative_to(root).as_posix(), **identity}


def _verify_generated_tree(
    namespace_root: Path,
    namespace: PurePosixPath,
) -> dict[str, Any]:
    namespace = _safe_namespace(str(namespace))
    _assert_no_symlink_tree(namespace_root)
    marker = namespace_root / "_generated_manifest.json"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"generated namespace marker missing or invalid: {namespace_root}")
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid generated namespace manifest: {marker}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"invalid generated namespace manifest schema: {marker}")
    if manifest.get("namespace") != str(namespace):
        raise ValueError(f"generated namespace manifest targets another namespace: {marker}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"generated namespace manifest lacks artifacts: {marker}")

    expected_paths: set[str] = set()
    for row in artifacts:
        if not isinstance(row, dict):
            raise ValueError(f"invalid artifact row in generated manifest: {marker}")
        relative = PurePosixPath(str(row.get("path") or ""))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"unsafe artifact path in generated manifest: {relative}")
        token = relative.as_posix()
        if token in expected_paths:
            raise ValueError(f"duplicate artifact path in generated manifest: {token}")
        expected_paths.add(token)
        path = namespace_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"generated artifact missing or invalid: {path}")
        identity = _content_identity(path)
        if identity != {
            "bytes": int(row.get("bytes", -1)),
            "sha256": str(row.get("sha256") or ""),
        }:
            raise ValueError(f"generated artifact hash mismatch: {path}")

    actual_paths = {
        path.relative_to(namespace_root).as_posix()
        for path in namespace_root.rglob("*")
        if path.is_file() and path != marker
    }
    if actual_paths != expected_paths:
        raise ValueError(
            "generated namespace contains modified, missing, or extra files: "
            f"expected={len(expected_paths)} actual={len(actual_paths)}"
        )
    graph = validate_graph(namespace_root, namespace)
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or any(
        int(counts.get(key, -1)) != int(value) for key, value in graph.items()
    ):
        raise ValueError(f"generated namespace graph counts do not match manifest: {marker}")
    return manifest


def stage_catalog(
    catalog: dict[str, Any],
    *,
    catalog_path: Path,
    staging_root: Path,
    namespace: PurePosixPath,
    generated_at: str,
    catalog_identity: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    namespace = _safe_namespace(str(namespace))
    catalog = _validate_catalog(catalog)
    staging_root.mkdir(parents=True, exist_ok=True)
    stage_namespace = staging_root.joinpath(*namespace.parts)
    _assert_contained_without_symlinks(staging_root, stage_namespace.parent)
    previous = stage_namespace.parent / f".{stage_namespace.name}.previous"
    if not stage_namespace.exists() and previous.exists():
        _verify_generated_tree(previous, namespace)
        _durable_replace(previous, stage_namespace)
    if stage_namespace.exists():
        _verify_generated_tree(stage_namespace, namespace)
    if previous.exists():
        _verify_generated_tree(previous, namespace)
        shutil.rmtree(previous)
    for leftover in stage_namespace.parent.glob(f".{stage_namespace.name}.build-*"):
        if leftover.is_symlink():
            raise ValueError(f"refusing unowned staging build directory: {leftover}")
        if not (leftover / "_build_in_progress.json").is_file():
            _verify_generated_tree(leftover, namespace)
        shutil.rmtree(leftover)

    build_namespace = stage_namespace.parent / f".{stage_namespace.name}.build-{uuid.uuid4().hex}"
    build_namespace.mkdir(parents=True)
    _atomic_write_json(
        build_namespace / "_build_in_progress.json",
        {"schema_version": _SCHEMA_VERSION, "namespace": str(namespace)},
    )
    for category in ("Strategies", "Families", "Evidence"):
        (build_namespace / category).mkdir()

    strategies = sorted(catalog["strategies"], key=lambda row: str(row["strategy"]))
    families = sorted(catalog["families"], key=lambda row: str(row["family"]))
    sources = sorted(catalog.get("evidence_sources") or [], key=lambda row: str(row.get("id")))
    source_ids = [str(row.get("id") or "") for row in sources]
    if any(not token for token in source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("catalog evidence source ids must be non-empty and unique")
    _assert_unique_note_filenames(strategies, key="strategy", category="strategy")
    _assert_unique_note_filenames(families, key="family", category="family")
    _assert_unique_note_filenames(sources, key="id", category="evidence")

    family_ids = {str(row["family"]) for row in families}
    if any(str(row["family"]) not in family_ids for row in strategies):
        raise ValueError("strategy references a family missing from catalog families")

    (build_namespace / "Strategy Research Index.md").write_text(
        _index_note(catalog, namespace=namespace, generated_at=generated_at), encoding="utf-8"
    )
    for row in strategies:
        path = build_namespace / "Strategies" / f"{_note_filename(row['strategy'])}.md"
        path.write_text(
            _strategy_note(dict(row), namespace=namespace, generated_at=generated_at),
            encoding="utf-8",
        )
    for row in families:
        path = build_namespace / "Families" / f"{_note_filename(row['family'])}.md"
        path.write_text(
            _family_note(dict(row), namespace=namespace, generated_at=generated_at),
            encoding="utf-8",
        )

    family_by_evidence: dict[str, list[str]] = {source_id: [] for source_id in source_ids}
    strategy_by_evidence: dict[str, list[str]] = {source_id: [] for source_id in source_ids}
    strategy_context_by_evidence: dict[str, list[str]] = {source_id: [] for source_id in source_ids}
    for row in families:
        for source_id in row.get("evidence_ids") or []:
            if source_id in family_by_evidence:
                family_by_evidence[source_id].append(str(row["family"]))
    for row in strategies:
        for source_id in row.get("evidence_ids") or []:
            if source_id in strategy_by_evidence:
                strategy_by_evidence[source_id].append(str(row["strategy"]))
        for source_id in row.get("family_context_evidence_ids") or []:
            if source_id in strategy_context_by_evidence:
                strategy_context_by_evidence[source_id].append(str(row["strategy"]))
    for source in sources:
        source_id = str(source["id"])
        path = build_namespace / "Evidence" / f"{_note_filename(source_id)}.md"
        path.write_text(
            _evidence_note(
                dict(source),
                namespace=namespace,
                generated_at=generated_at,
                families=sorted(family_by_evidence[source_id]),
                strategies=sorted(strategy_by_evidence[source_id]),
                contextual_strategies=sorted(strategy_context_by_evidence[source_id]),
            ),
            encoding="utf-8",
        )

    graph = validate_graph(build_namespace, namespace)
    artifacts = [
        _relative_artifact(path, build_namespace) for path in sorted(build_namespace.rglob("*.md"))
    ]
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "catalog_generated_at_utc": str(catalog.get("generated_at_utc") or generated_at),
        "namespace": str(namespace),
        "source_catalog": dict(catalog_identity or _content_identity(catalog_path)),
        "exporter": _content_identity(Path(__file__).resolve()),
        "counts": {
            **graph,
            "strategies": len(strategies),
            "families": len(families),
            "evidence": len(sources),
        },
        "artifacts": artifacts,
    }
    _atomic_write_json(build_namespace / "_generated_manifest.json", manifest)
    (build_namespace / "_build_in_progress.json").unlink()
    _verify_generated_tree(build_namespace, namespace)
    try:
        if stage_namespace.exists():
            _durable_replace(stage_namespace, previous)
        _durable_replace(build_namespace, stage_namespace)
        _verify_generated_tree(stage_namespace, namespace)
    except Exception:
        if build_namespace.exists():
            shutil.rmtree(build_namespace)
        if previous.exists():
            _verify_generated_tree(previous, namespace)
            if stage_namespace.exists():
                quarantine = stage_namespace.parent / (
                    f".{stage_namespace.name}.failed-{uuid.uuid4().hex}"
                )
                _durable_replace(stage_namespace, quarantine)
                try:
                    _durable_replace(previous, stage_namespace)
                finally:
                    shutil.rmtree(quarantine, ignore_errors=True)
            else:
                _durable_replace(previous, stage_namespace)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    return stage_namespace, manifest


def _manifest_time(manifest: dict[str, Any]) -> datetime:
    token = str(manifest.get("catalog_generated_at_utc") or "")
    try:
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid catalog generation time in manifest: {token!r}") from exc


def _assert_not_stale(existing: dict[str, Any], incoming: dict[str, Any]) -> None:
    existing_time = _manifest_time(existing)
    incoming_time = _manifest_time(incoming)
    existing_source = dict(existing.get("source_catalog") or {})
    incoming_source = dict(incoming.get("source_catalog") or {})
    future_limit = datetime.now(UTC) + timedelta(seconds=_MAX_FUTURE_SKEW_SECONDS)
    if existing_time > future_limit and incoming_time <= future_limit:
        return
    if incoming_time < existing_time:
        raise ValueError("refusing to replace generated namespace with an older catalog")
    if incoming_time == existing_time and incoming_source != existing_source:
        raise ValueError("refusing unrelated catalog content with the same generation time")


def _clear_journal(path: Path) -> None:
    if path.exists():
        path.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _validate_apply_receipt(
    payload: Any,
    *,
    manifest: dict[str, Any],
    manifest_identity: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Obsidian apply receipt payload must be an object")
    expected = {
        "schema_version": _SCHEMA_VERSION,
        "counts": manifest["counts"],
        "source_catalog": manifest["source_catalog"],
        "stage_manifest": manifest_identity,
        "exporter": manifest["exporter"],
        "applied": True,
        "destination": str(destination),
        "installed_manifest": manifest_identity,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Obsidian apply receipt does not match installed manifest")
    return dict(payload)


def _complete_apply_receipt(
    base: dict[str, Any] | None,
    *,
    manifest: dict[str, Any],
    manifest_identity: dict[str, Any],
    destination: Path,
    backup: Path | None,
) -> dict[str, Any]:
    payload = dict(base or {})
    payload.setdefault("schema_version", _SCHEMA_VERSION)
    payload.setdefault("counts", manifest["counts"])
    payload.setdefault("source_catalog", manifest["source_catalog"])
    payload.setdefault("stage_manifest", manifest_identity)
    payload.setdefault("exporter", manifest["exporter"])
    payload.update(
        {
            "applied": True,
            "destination": str(destination),
            "backup": str(backup) if backup is not None else None,
            "installed_manifest": manifest_identity,
        }
    )
    return _validate_apply_receipt(
        payload,
        manifest=manifest,
        manifest_identity=manifest_identity,
        destination=destination,
    )


def _recover_apply_journal(
    *,
    journal_path: Path,
    destination: Path,
    namespace: PurePosixPath,
    vault_root: Path,
    receipt_path: Path | None = None,
) -> None:
    if not journal_path.exists():
        return
    _assert_contained_without_symlinks(vault_root, journal_path)
    if journal_path.is_symlink():
        raise ValueError(f"Obsidian swap journal must not be a symlink: {journal_path}")
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Obsidian swap journal: {journal_path}") from exc
    if not isinstance(journal, dict) or journal.get("destination") != str(destination):
        raise ValueError(f"Obsidian swap journal does not match destination: {journal_path}")
    incoming = Path(str(journal.get("incoming") or ""))
    backup_token = str(journal.get("backup") or "")
    backup = Path(backup_token) if backup_token else None
    target_manifest_sha256 = str(journal.get("target_manifest_sha256") or "")
    journal_receipt = journal.get("receipt")
    if len(target_manifest_sha256) != 64:
        raise ValueError(f"Obsidian swap journal target digest is invalid: {journal_path}")
    if journal_receipt is not None and (
        not isinstance(journal_receipt, dict)
        or receipt_path is None
        or journal_receipt.get("path") != str(receipt_path)
        or not isinstance(journal_receipt.get("payload"), dict)
        or receipt_path.is_symlink()
    ):
        raise ValueError(f"Obsidian swap journal receipt is invalid: {journal_path}")
    if incoming.parent != destination.parent or not incoming.name.startswith(
        f".{destination.name}.incoming-"
    ):
        raise ValueError(f"unsafe incoming path in Obsidian swap journal: {journal_path}")
    if backup is not None and backup.parent != vault_root / ".luminaquant-generated-backups":
        raise ValueError(f"unsafe backup path in Obsidian swap journal: {journal_path}")
    _assert_contained_without_symlinks(vault_root, incoming)
    if backup is not None:
        _assert_contained_without_symlinks(vault_root, backup)

    def verify_target(path: Path) -> dict[str, Any]:
        manifest = _verify_generated_tree(path, namespace)
        actual_sha = _content_identity(path / "_generated_manifest.json")["sha256"]
        if actual_sha != target_manifest_sha256:
            raise ValueError(f"Obsidian swap journal incoming digest mismatch: {journal_path}")
        return manifest

    def finalize_target(path: Path, manifest: dict[str, Any]) -> None:
        identity = _content_identity(path / "_generated_manifest.json")
        if journal_receipt is not None:
            assert receipt_path is not None
            payload = _validate_apply_receipt(
                journal_receipt["payload"],
                manifest=manifest,
                manifest_identity=identity,
                destination=destination,
            )
            _atomic_write_json(receipt_path, payload)
        _clear_journal(journal_path)

    if destination.exists():
        installed_manifest = _verify_generated_tree(destination, namespace)
        installed_sha = _content_identity(destination / "_generated_manifest.json")["sha256"]
        if installed_sha == target_manifest_sha256:
            if incoming.exists():
                verify_target(incoming)
                shutil.rmtree(incoming)
            finalize_target(destination, installed_manifest)
            return
        if not incoming.exists() or backup is None or backup.exists():
            raise ValueError(f"cannot safely resume Obsidian swap journal: {journal_path}")
        incoming_manifest = verify_target(incoming)
        _assert_not_stale(installed_manifest, incoming_manifest)
        _durable_replace(destination, backup)
        _durable_replace(incoming, destination)
        recovered_manifest = verify_target(destination)
        finalize_target(destination, recovered_manifest)
        return

    if incoming.exists():
        incoming_manifest = verify_target(incoming)
        if backup is not None and backup.exists():
            backup_manifest = _verify_generated_tree(backup, namespace)
            _assert_not_stale(backup_manifest, incoming_manifest)
        _durable_replace(incoming, destination)
        recovered_manifest = verify_target(destination)
        finalize_target(destination, recovered_manifest)
        return
    if backup is not None and backup.exists():
        _verify_generated_tree(backup, namespace)
        _durable_replace(backup, destination)
        _clear_journal(journal_path)
        return
    raise ValueError(f"cannot recover incomplete Obsidian swap: {journal_path}")


def apply_staged_namespace(
    stage_namespace: Path,
    *,
    vault_root: Path,
    namespace: PurePosixPath,
    generated_at: str,
    expected_stage_manifest: dict[str, Any] | None = None,
    receipt_path: Path | None = None,
    receipt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    namespace = _safe_namespace(str(namespace))
    try:
        vault_mode = vault_root.stat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"Obsidian vault root not found: {vault_root}")
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"Obsidian vault root unavailable: {vault_root}",
        ) from exc
    if not stat.S_ISDIR(vault_mode):
        raise NotADirectoryError(f"Obsidian vault root is not a directory: {vault_root}")
    destination = vault_root.joinpath(*namespace.parts)
    if _paths_overlap(stage_namespace, vault_root):
        raise ValueError("staging and Obsidian vault paths must not overlap")
    _assert_contained_without_symlinks(vault_root, destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_contained_without_symlinks(vault_root, destination)
    journal_path = destination.parent / f".{destination.name}.swap-journal.json"
    lock_path = destination.parent / f".{destination.name}.apply.lock"
    _assert_contained_without_symlinks(vault_root, lock_path)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        stage_manifest = _verify_generated_tree(stage_namespace, namespace)
        actual_stage_identity = _content_identity(stage_namespace / "_generated_manifest.json")
        if expected_stage_manifest is not None and actual_stage_identity != dict(
            expected_stage_manifest
        ):
            raise ValueError("staged Obsidian manifest changed before apply")
        _recover_apply_journal(
            journal_path=journal_path,
            destination=destination,
            namespace=namespace,
            vault_root=vault_root,
            receipt_path=receipt_path,
        )
        incoming = destination.parent / f".{destination.name}.incoming-{uuid.uuid4().hex}"
        backup: Path | None = None
        destination_moved = False
        incoming_installed = False
        installation_verified = False
        result: dict[str, Any] = {}
        final_receipt: dict[str, Any] | None = None
        shutil.copytree(stage_namespace, incoming)
        try:
            incoming_manifest = _verify_generated_tree(incoming, namespace)
            if incoming_manifest != stage_manifest:
                raise ValueError("incoming Obsidian copy does not match staged manifest")
            if destination.exists():
                existing_manifest = _verify_generated_tree(destination, namespace)
                _assert_not_stale(existing_manifest, incoming_manifest)
                backup_root = vault_root / ".luminaquant-generated-backups"
                _assert_contained_without_symlinks(vault_root, backup_root)
                backup_root.mkdir(parents=True, exist_ok=True)
                _assert_contained_without_symlinks(vault_root, backup_root)
                stamp = generated_at.replace(":", "").replace("-", "")
                backup = backup_root / f"{stamp}-{destination.name}-{uuid.uuid4().hex[:8]}"
            incoming_identity = _content_identity(incoming / "_generated_manifest.json")
            result = {
                "applied": True,
                "destination": str(destination),
                "backup": str(backup) if backup is not None else None,
                "installed_manifest": incoming_identity,
            }
            if receipt_path is not None:
                final_receipt = _complete_apply_receipt(
                    receipt_payload,
                    manifest=incoming_manifest,
                    manifest_identity=incoming_identity,
                    destination=destination,
                    backup=backup,
                )
            _atomic_write_json(
                journal_path,
                {
                    "schema_version": _SCHEMA_VERSION,
                    "destination": str(destination),
                    "incoming": str(incoming),
                    "backup": str(backup) if backup is not None else None,
                    "target_manifest_sha256": incoming_identity["sha256"],
                    "receipt": (
                        {"path": str(receipt_path), "payload": final_receipt}
                        if final_receipt is not None
                        else None
                    ),
                },
            )
            if backup is not None:
                _durable_replace(destination, backup)
                destination_moved = True
            _durable_replace(incoming, destination)
            incoming_installed = True
            installed_manifest = _verify_generated_tree(destination, namespace)
            if installed_manifest != incoming_manifest:
                raise ValueError("installed Obsidian namespace does not match incoming manifest")
            if _content_identity(destination / "_generated_manifest.json") != incoming_identity:
                raise ValueError("installed Obsidian manifest identity changed during apply")
            installation_verified = True
            if receipt_path is not None:
                assert final_receipt is not None
                _atomic_write_json(receipt_path, final_receipt)
            _clear_journal(journal_path)
        except Exception:
            if installation_verified:
                raise
            if incoming.exists():
                _verify_generated_tree(incoming, namespace)
                shutil.rmtree(incoming)
            if incoming_installed and destination.exists():
                _assert_contained_without_symlinks(vault_root, destination)
                shutil.rmtree(destination)
            if (
                destination_moved
                and backup is not None
                and backup.exists()
                and not destination.exists()
            ):
                _durable_replace(backup, destination)
            _clear_journal(journal_path)
            raise
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--namespace", default=_DEFAULT_NAMESPACE)
    parser.add_argument("--vault-root", default="")
    parser.add_argument("--apply", action="store_true")
    return parser


def export_catalog_snapshot(
    *,
    catalog_path: Path,
    staging_root: Path,
    namespace: PurePosixPath,
    vault_root: Path | None,
    apply: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    namespace = _safe_namespace(str(namespace))
    catalog_path = catalog_path.expanduser().resolve()
    staging_root = staging_root.expanduser().resolve()
    vault_root = vault_root.expanduser().resolve() if vault_root is not None else None
    if apply and vault_root is None:
        raise ValueError("vault_root is required when apply is true")
    if vault_root is not None and _paths_overlap(staging_root, vault_root):
        raise ValueError("staging and Obsidian vault paths must not overlap")

    staging_root.mkdir(parents=True, exist_ok=True)
    stage_parent = staging_root.joinpath(*namespace.parts[:-1])
    _assert_contained_without_symlinks(staging_root, stage_parent)
    stage_parent.mkdir(parents=True, exist_ok=True)
    export_lock_path = stage_parent / f".{namespace.name}.export.lock"
    _assert_contained_without_symlinks(staging_root, export_lock_path)
    with export_lock_path.open("a+b") as export_lock:
        _acquire_exclusive_flock(export_lock)
        catalog, catalog_identity = _load_catalog_snapshot(catalog_path)
        exported_at = _utc_now()
        content_generated_at = str(catalog.get("generated_at_utc") or exported_at)
        stage_namespace, manifest = stage_catalog(
            catalog,
            catalog_path=catalog_path,
            staging_root=staging_root,
            namespace=namespace,
            generated_at=content_generated_at,
            catalog_identity=catalog_identity,
        )
        if _content_identity(catalog_path) != catalog_identity:
            raise ValueError("strategy catalog changed while the Obsidian snapshot was staged")
        verified_manifest = _verify_generated_tree(stage_namespace, namespace)
        expected_manifest_fields = {
            "namespace": str(namespace),
            "catalog_generated_at_utc": content_generated_at,
            "source_catalog": catalog_identity,
            "exporter": _content_identity(Path(__file__).resolve()),
            "counts": manifest["counts"],
        }
        if verified_manifest != manifest or any(
            verified_manifest.get(key) != value for key, value in expected_manifest_fields.items()
        ):
            raise ValueError("staged Obsidian manifest does not match the catalog snapshot")
        stage_manifest_identity = _content_identity(stage_namespace / "_generated_manifest.json")
        receipt: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "exported_at_utc": exported_at,
            "stage": str(stage_namespace),
            "counts": manifest["counts"],
            "source_catalog": catalog_identity,
            "stage_manifest": stage_manifest_identity,
            "exporter": _content_identity(Path(__file__).resolve()),
            "applied": False,
        }
        receipt_path = staging_root / "obsidian_export_receipt.json"
        attempt_path = staging_root / "obsidian_export_attempt_latest.json"
        _atomic_write_json(attempt_path, {**receipt, "status": "staged"})
        if apply:
            assert vault_root is not None
            try:
                receipt.update(
                    apply_staged_namespace(
                        stage_namespace,
                        vault_root=vault_root,
                        namespace=namespace,
                        generated_at=exported_at,
                        expected_stage_manifest=stage_manifest_identity,
                        receipt_path=receipt_path,
                        receipt_payload=receipt,
                    )
                )
            except Exception as exc:
                sanitized_message = " ".join(str(exc).split())[:500]
                _atomic_write_json(
                    attempt_path,
                    {
                        **receipt,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_errno": getattr(exc, "errno", None),
                        "error_message": sanitized_message,
                        "requested_vault_root": str(vault_root),
                    },
                )
                raise
            _atomic_write_json(attempt_path, {**receipt, "status": "applied"})
        else:
            _atomic_write_json(receipt_path, receipt)
        return receipt, manifest


def main() -> int:
    args = _build_parser().parse_args()
    namespace = _safe_namespace(args.namespace)
    vault_root = Path(args.vault_root) if str(args.vault_root).strip() else None
    if args.apply and vault_root is None:
        raise SystemExit("--vault-root is required with --apply")
    receipt, manifest = export_catalog_snapshot(
        catalog_path=Path(args.catalog),
        staging_root=Path(args.staging_root),
        namespace=namespace,
        vault_root=vault_root,
        apply=bool(args.apply),
    )
    print(f"[OBSIDIAN] notes={manifest['counts']['note_count']}")
    print(f"[OBSIDIAN] links={manifest['counts']['link_count']}")
    print(f"[OBSIDIAN] stage={receipt['stage']}")
    print(f"[OBSIDIAN] applied={receipt['applied']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
