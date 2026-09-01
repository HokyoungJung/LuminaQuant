#!/usr/bin/env python3
"""Write immutable baseline provenance for TradFi external-alpha research.

This artifact is the G001 foundation for the TradFi external-alpha search.  It
does not run optimization or start paper/shadow/live trading.  It freezes the
current 110-asset TradFi-aware report inputs, PRD/test-spec hashes, data/universe
coverage, current baseline strategy labels, and an external-evidence source
registry with credential/usage boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ALPHA_V2_ROOT = (
    REPO_ROOT
    / "var"
    / "reports"
    / "profit_moonshot_20260501"
    / "current_tail_20260508"
    / "alpha_v2"
)
SOURCE_REPORT_DIR = ALPHA_V2_ROOT / "alpha_zoo_110_asset_tradfi_aware_wf_20260613"
DEFAULT_BASELINE_WF_JSON = SOURCE_REPORT_DIR / "tradfi_aware_wf_latest.json"
DEFAULT_BASELINE_SUMMARY_JSON = SOURCE_REPORT_DIR / "tradfi_aware_improvement_summary_latest.json"
DEFAULT_PRD_PATH = REPO_ROOT / ".omx" / "plans" / "prd-tradfi-external-alpha-search-20260613.md"
DEFAULT_TEST_SPEC_PATH = (
    REPO_ROOT / ".omx" / "plans" / "test-spec-tradfi-external-alpha-search-20260613.md"
)
DEFAULT_OUTPUT_DIR = ALPHA_V2_ROOT / "tradfi_external_alpha_search_20260613"

BASELINE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("current_best_clean", "current best clean"),
    ("best_clean_mdd15", "best clean under 15pct MDD"),
    ("current_best_demoted_shadow", "current best demoted shadow"),
)

BASELINE_FIELDS: tuple[str, ...] = (
    "candidate_label",
    "family",
    "clean_promotion_eligible",
    "compounded_oos_return_pct",
    "max_oos_mdd_pct",
    "latest_oos_return_pct",
    "positive_oos_folds",
    "fold_count",
    "hard_stop_promotable",
    "non_clean_reasons",
)

REQUIRED_SOURCE_FIELDS: tuple[str, ...] = (
    "source_id",
    "strategy_class",
    "source_url",
    "license_usage_note",
    "credential_requirement",
    "credential_required",
    "paid_required",
    "broker_or_live_required",
    "update_cadence",
    "release_lag_policy",
    "cache_path",
    "allowed_usage_label",
    "cycle_allowed",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_artifact(path: Path, *, role: str, required: bool = True) -> dict[str, Any]:
    exists = path.exists() and path.is_file()
    return {
        "role": role,
        "path": str(path),
        "exists": exists,
        "required": required,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size if exists else None,
    }


def _compact_baseline_row(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {}
    return {key: row[key] for key in BASELINE_FIELDS if key in row}


def _extract_current_baseline_labels(summary_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for key, title in BASELINE_SECTIONS:
        row = summary_payload.get(key)
        compact = _compact_baseline_row(row if isinstance(row, Mapping) else None)
        labels.append(
            {
                "section": key,
                "title": title,
                "candidate_label": compact.get("candidate_label"),
                "metrics": compact,
                "readiness_label": _readiness_label_for_section(key, compact),
            }
        )
    return labels


def _readiness_label_for_section(section: str, row: Mapping[str, Any]) -> str:
    if section == "current_best_demoted_shadow":
        return "shadow_freeze_only_requires_fresh_forward"
    if row.get("hard_stop_promotable") is True:
        return "clean_hard_stop_promotable"
    if row.get("clean_promotion_eligible") is True:
        return "clean_candidate_not_promotable"
    return "diagnostic_only"


def _extract_historical_baseline_labels(summary_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = summary_payload.get("baseline_candidates_recorded")
    if not isinstance(rows, list):
        return []
    return [
        {
            "label": str(row.get("label", "")),
            "verdict": row.get("verdict"),
            "notes": row.get("notes"),
            "compounded_oos_return_pct": row.get("compounded_oos_return_pct"),
        }
        for row in rows
        if isinstance(row, Mapping) and str(row.get("label", "")).strip()
    ]


def _extract_data_coverage(
    wf_payload: Mapping[str, Any],
    summary_payload: Mapping[str, Any],
) -> dict[str, Any]:
    wf_coverage = wf_payload.get("data_coverage")
    summary_coverage = summary_payload.get("data_coverage")
    universe = wf_payload.get("universe")
    wf_coverage = wf_coverage if isinstance(wf_coverage, Mapping) else {}
    summary_coverage = summary_coverage if isinstance(summary_coverage, Mapping) else {}
    universe = universe if isinstance(universe, Mapping) else {}

    latest = summary_coverage.get("latest_available_data") or wf_coverage.get("global_latest_utc")
    requested = summary_coverage.get("requested_symbol_count") or universe.get(
        "requested_symbol_count"
    )
    loaded = summary_coverage.get("loaded_symbol_count") or universe.get("loaded_symbol_count")
    missing = summary_coverage.get("missing_symbol_count") or universe.get("missing_symbol_count")
    fold_count = summary_coverage.get("fold_count")
    if fold_count is None:
        folds = wf_payload.get("folds")
        fold_count = len(folds) if isinstance(folds, list) else None

    compact = {
        "latest_available_data_utc": latest,
        "requested_symbol_count": requested,
        "loaded_symbol_count": loaded,
        "missing_symbol_count": missing,
        "fold_count": fold_count,
        "source": wf_coverage.get("source"),
        "global_earliest_utc": wf_coverage.get("global_earliest_utc"),
        "global_latest_utc": wf_coverage.get("global_latest_utc"),
    }
    compact["data_coverage_sha256"] = _payload_sha256(compact)
    return compact


def _extract_universe(wf_payload: Mapping[str, Any]) -> dict[str, Any]:
    universe = wf_payload.get("universe")
    data_coverage = wf_payload.get("data_coverage")
    universe = universe if isinstance(universe, Mapping) else {}
    data_coverage = data_coverage if isinstance(data_coverage, Mapping) else {}
    symbols = (
        universe.get("symbols")
        or universe.get("loaded_symbols")
        or data_coverage.get("symbols_with_any_rows")
    )
    normalized = (
        sorted(str(symbol) for symbol in symbols if str(symbol).strip())
        if isinstance(symbols, list)
        else []
    )
    missing = universe.get("missing_symbols")
    if not isinstance(missing, list):
        missing = (
            data_coverage.get("missing_symbols")
            if isinstance(data_coverage.get("missing_symbols"), list)
            else []
        )
    payload = {
        "symbol_count": len(normalized),
        "symbols": normalized,
        "missing_symbols": sorted(str(symbol) for symbol in missing),
    }
    return {
        "symbol_count": payload["symbol_count"],
        "symbols": payload["symbols"],
        "missing_symbols": payload["missing_symbols"],
        "universe_sha256": _payload_sha256(payload),
    }


def build_external_source_registry(*, generated_at_utc: str | None = None) -> dict[str, Any]:
    entries = [
        {
            "source_id": "moreira_muir_volatility_managed_portfolios",
            "strategy_class": "tradfi_vol_managed_v1",
            "readiness_rank": 1,
            "source_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2659431",
            "license_usage_note": "Public abstract/paper reference; use as strategy-class evidence only unless separately cached under license-compatible terms.",
            "credential_requirement": "none",
            "credential_required": False,
            "paid_required": False,
            "broker_or_live_required": False,
            "update_cadence": "static research paper",
            "release_lag_policy": "No market-data feature release lag; fixes family thesis only.",
            "cache_path": "var/cache/external_alpha_sources/moreira_muir_volatility_managed_portfolios.html",
            "allowed_usage_label": "strategy_class_documentation_only",
            "cycle_allowed": True,
        },
        {
            "source_id": "moskowitz_ooi_pedersen_time_series_momentum",
            "strategy_class": "tradfi_momentum_regime_v1",
            "readiness_rank": 2,
            "source_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463",
            "license_usage_note": "Public paper reference; no proprietary data copied into this repo.",
            "credential_requirement": "none",
            "credential_required": False,
            "paid_required": False,
            "broker_or_live_required": False,
            "update_cadence": "static research paper",
            "release_lag_policy": "No market-data feature release lag; horizons must be frozen before replay.",
            "cache_path": "var/cache/external_alpha_sources/moskowitz_ooi_pedersen_time_series_momentum.html",
            "allowed_usage_label": "strategy_class_documentation_only",
            "cycle_allowed": True,
        },
        {
            "source_id": "gao_han_li_zhou_intraday_momentum",
            "strategy_class": "tradfi_intraday_session_v1",
            "readiness_rank": 3,
            "source_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2552752",
            "license_usage_note": "Public paper reference; intraday implementation requires repo-owned bar data and session checks.",
            "credential_requirement": "none",
            "credential_required": False,
            "paid_required": False,
            "broker_or_live_required": False,
            "update_cadence": "static research paper",
            "release_lag_policy": "Only use bars available before decision time; enforce bar-close lag.",
            "cache_path": "var/cache/external_alpha_sources/gao_han_li_zhou_intraday_momentum.html",
            "allowed_usage_label": "strategy_class_documentation_only",
            "cycle_allowed": True,
        },
        {
            "source_id": "nyse_hours_calendars",
            "strategy_class": "tradfi_session_calendar_controls",
            "readiness_rank": 4,
            "source_url": "https://www.nyse.com/trade/hours-calendars",
            "license_usage_note": "Official public calendar reference; cache only derived session assumptions, not live trading data.",
            "credential_requirement": "none",
            "credential_required": False,
            "paid_required": False,
            "broker_or_live_required": False,
            "update_cadence": "exchange calendar updates",
            "release_lag_policy": "Calendar must be known before session mask generation; unknown early closes block clean promotion.",
            "cache_path": "var/cache/external_alpha_sources/nyse_hours_calendars.json",
            "allowed_usage_label": "session_control_documentation",
            "cycle_allowed": True,
        },
        {
            "source_id": "ny_fed_staff_report_917_overnight_drift",
            "strategy_class": "tradfi_overnight_split_v1",
            "readiness_rank": 5,
            "source_url": "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr917.pdf",
            "license_usage_note": "Public central-bank research reference; not direct evidence for Binance TradFi instruments.",
            "credential_requirement": "none",
            "credential_required": False,
            "paid_required": False,
            "broker_or_live_required": False,
            "update_cadence": "static staff report",
            "release_lag_policy": "Diagnostic only until instrument timestamp validity and overnight window mapping are proven.",
            "cache_path": "var/cache/external_alpha_sources/ny_fed_staff_report_917_overnight_drift.pdf",
            "allowed_usage_label": "diagnostic_only_until_timestamp_validated",
            "cycle_allowed": True,
        },
        {
            "source_id": "fama_french_data_library",
            "strategy_class": "factor_regime_router_v1",
            "readiness_rank": 6,
            "source_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
            "license_usage_note": "Free public factor library reference; downstream files must retain source URL and cache hash.",
            "credential_requirement": "none",
            "credential_required": False,
            "paid_required": False,
            "broker_or_live_required": False,
            "update_cadence": "daily/monthly factor file updates depending on dataset",
            "release_lag_policy": "Clean use requires publication/update-date lag; missing lag keeps feature diagnostic_only.",
            "cache_path": "var/cache/external_alpha_sources/fama_french_factor_file.csv",
            "allowed_usage_label": "diagnostic_only_until_release_lag_encoded",
            "cycle_allowed": True,
        },
        {
            "source_id": "fed_fred_ddp_csv_download",
            "strategy_class": "macro_risk_filters",
            "readiness_rank": 7,
            "source_url": "https://www.federalreserve.gov/data/data-download-fred-information.htm",
            "license_usage_note": "Federal Reserve/FRED public download documentation; no API-key use in this cycle.",
            "credential_requirement": "none for documented CSV/download surfaces",
            "credential_required": False,
            "paid_required": False,
            "broker_or_live_required": False,
            "update_cadence": "series-dependent",
            "release_lag_policy": "Release calendar must be encoded; otherwise macro features remain diagnostic_only.",
            "cache_path": "var/cache/external_alpha_sources/fred_no_key_csv_series.csv",
            "allowed_usage_label": "diagnostic_only_until_release_calendar_encoded",
            "cycle_allowed": True,
        },
        {
            "source_id": "fred_api_key_docs_excluded",
            "strategy_class": "macro_risk_filters",
            "readiness_rank": 8,
            "source_url": "https://fred.stlouisfed.org/docs/api/api_key.html",
            "license_usage_note": "Documents an API-key surface; explicitly excluded from this no-credential cycle.",
            "credential_requirement": "api_key_required",
            "credential_required": True,
            "paid_required": False,
            "broker_or_live_required": False,
            "update_cadence": "series-dependent",
            "release_lag_policy": "Excluded; do not load via API key in this cycle.",
            "cache_path": None,
            "allowed_usage_label": "excluded_requires_api_key_reference_only",
            "cycle_allowed": False,
        },
    ]
    registry = {
        "artifact_kind": "tradfi_external_alpha_source_registry",
        "generated_at_utc": generated_at_utc or _utc_now_iso(),
        "policy": {
            "cycle": "G001-baseline-provenance-foundation",
            "free_unauthenticated_only": True,
            "paid_api_key_broker_live_sources_allowed": False,
            "missing_release_lag_demotes_to": "diagnostic_only",
        },
        "sources": entries,
    }
    registry["validation"] = validate_source_registry(registry)
    registry["source_registry_sha256"] = _payload_sha256(
        {key: value for key, value in registry.items() if key != "source_registry_sha256"}
    )
    return registry


def validate_source_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    sources = registry.get("sources")
    violations: list[dict[str, Any]] = []
    excluded_source_ids: list[str] = []
    allowed_source_ids: list[str] = []
    if not isinstance(sources, list):
        return {
            "valid": False,
            "violations": [{"source_id": None, "reason": "sources must be a list"}],
            "allowed_source_ids": [],
            "excluded_source_ids": [],
        }

    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            violations.append({"source_id": f"index:{index}", "reason": "source must be object"})
            continue
        source_id = str(source.get("source_id", f"index:{index}"))
        missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in source]
        if missing:
            violations.append(
                {"source_id": source_id, "reason": "missing_required_fields", "fields": missing}
            )
        cycle_allowed = source.get("cycle_allowed") is True
        if cycle_allowed:
            allowed_source_ids.append(source_id)
        else:
            excluded_source_ids.append(source_id)
        blocked_flags = {
            "credential_required": source.get("credential_required") is True,
            "paid_required": source.get("paid_required") is True,
            "broker_or_live_required": source.get("broker_or_live_required") is True,
        }
        if cycle_allowed and any(blocked_flags.values()):
            violations.append(
                {
                    "source_id": source_id,
                    "reason": "cycle_allowed_source_requires_disallowed_access",
                    "flags": blocked_flags,
                }
            )
        if cycle_allowed and not str(source.get("release_lag_policy", "")).strip():
            violations.append({"source_id": source_id, "reason": "missing_release_lag_policy"})
        if cycle_allowed and not str(source.get("allowed_usage_label", "")).strip():
            violations.append({"source_id": source_id, "reason": "missing_allowed_usage_label"})
    return {
        "valid": not violations,
        "violations": violations,
        "allowed_source_ids": allowed_source_ids,
        "excluded_source_ids": excluded_source_ids,
    }


def build_payload(
    *,
    baseline_wf_json: Path = DEFAULT_BASELINE_WF_JSON,
    baseline_summary_json: Path = DEFAULT_BASELINE_SUMMARY_JSON,
    prd_path: Path = DEFAULT_PRD_PATH,
    test_spec_path: Path = DEFAULT_TEST_SPEC_PATH,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at_utc or _utc_now_iso()
    wf_payload = _load_json(baseline_wf_json)
    summary_payload = _load_json(baseline_summary_json)
    if not isinstance(wf_payload, Mapping):
        raise ValueError("baseline_wf_json must contain a JSON object")
    if not isinstance(summary_payload, Mapping):
        raise ValueError("baseline_summary_json must contain a JSON object")

    source_registry = build_external_source_registry(generated_at_utc=generated_at)
    current_baselines = _extract_current_baseline_labels(summary_payload)
    historical_baselines = _extract_historical_baseline_labels(summary_payload)
    data_coverage = _extract_data_coverage(wf_payload, summary_payload)
    universe = _extract_universe(wf_payload)
    source_artifacts = {
        "baseline_wf_json": _source_artifact(
            baseline_wf_json, role="current 110-asset TradFi-aware WF report"
        ),
        "baseline_summary_json": _source_artifact(
            baseline_summary_json, role="current 110-asset TradFi-aware summary"
        ),
        "prd": _source_artifact(prd_path, role="approved RALPLAN PRD"),
        "test_spec": _source_artifact(test_spec_path, role="approved RALPLAN test spec"),
    }
    missing_required = [
        name for name, info in source_artifacts.items() if info["required"] and not info["exists"]
    ]

    payload = {
        "artifact_kind": "tradfi_external_alpha_baseline_evidence_snapshot",
        "generated_at_utc": generated_at,
        "ultragoal_story_id": "G001-baseline-provenance-foundation",
        "policy": {
            "real_money_execution": False,
            "paper_trading_start": False,
            "shadow_trading_start": False,
            "allowed_data_sources": "free_unauthenticated_or_reference_only",
            "readiness_cap_if_missing_provenance": "diagnostic_only",
        },
        "source_artifacts": source_artifacts,
        "missing_required_artifacts": missing_required,
        "data_coverage": data_coverage,
        "universe": universe,
        "current_baseline_labels": current_baselines,
        "historical_baseline_candidates_recorded": historical_baselines,
        "baseline_labels_sha256": _payload_sha256(
            {
                "current_baseline_labels": current_baselines,
                "historical_baseline_candidates_recorded": historical_baselines,
            }
        ),
        "external_source_registry": {
            "artifact_kind": source_registry["artifact_kind"],
            "source_count": len(source_registry["sources"]),
            "source_registry_sha256": source_registry["source_registry_sha256"],
            "validation": source_registry["validation"],
        },
        "readiness_summary": {
            "clean_candidate_count": sum(
                1
                for item in current_baselines
                if item.get("readiness_label") == "clean_candidate_not_promotable"
            ),
            "shadow_freeze_only_count": sum(
                1
                for item in current_baselines
                if item.get("readiness_label") == "shadow_freeze_only_requires_fresh_forward"
            ),
            "real_money_status": summary_payload.get("real_money_status", {}),
            "conclusion": (
                "Baseline provenance is frozen for research only; no paper, shadow, canary, "
                "or real-money trading is started by this artifact."
            ),
        },
        "source_registry_payload": source_registry,
    }
    payload["baseline_evidence_snapshot_sha256"] = _payload_sha256(
        {key: value for key, value in payload.items() if key != "baseline_evidence_snapshot_sha256"}
    )
    return payload


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# TradFi external-alpha baseline provenance",
        "",
        f"- generated: `{payload['generated_at_utc']}`",
        f"- Ultragoal story: `{payload['ultragoal_story_id']}`",
        "- Trading starts: **none** (`real_money_execution=false`, `paper/shadow=false`).",
        "",
        "## Immutable hashes",
        "",
    ]
    for name, info in dict(payload["source_artifacts"]).items():
        exists = "Y" if info.get("exists") else "N"
        lines.append(
            f"- `{name}` exists={exists} sha256=`{info.get('sha256') or 'missing'}` path=`{info.get('path')}`"
        )
    lines.extend(
        [
            f"- universe_sha256: `{payload['universe']['universe_sha256']}`",
            f"- data_coverage_sha256: `{payload['data_coverage']['data_coverage_sha256']}`",
            f"- baseline_labels_sha256: `{payload['baseline_labels_sha256']}`",
            f"- source_registry_sha256: `{payload['external_source_registry']['source_registry_sha256']}`",
            "",
            "## Data coverage",
            "",
            f"- latest available data: `{payload['data_coverage'].get('latest_available_data_utc')}`",
            f"- symbols: requested `{payload['data_coverage'].get('requested_symbol_count')}`, loaded `{payload['data_coverage'].get('loaded_symbol_count')}`, missing `{payload['data_coverage'].get('missing_symbol_count')}`",
            f"- folds: `{payload['data_coverage'].get('fold_count')}`",
            "",
            "## Current baseline labels",
            "",
            "| Section | Candidate | OOS comp | Max OOS MDD | Hit folds | Latest OOS | Readiness |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in payload["current_baseline_labels"]:
        metrics = dict(item.get("metrics") or {})
        lines.append(
            "| `{section}` | `{label}` | {comp} | {mdd} | `{hits}` | {latest} | `{readiness}` |".format(
                section=item.get("section"),
                label=item.get("candidate_label"),
                comp=_pct(metrics.get("compounded_oos_return_pct")),
                mdd=_pct(metrics.get("max_oos_mdd_pct")),
                hits=metrics.get("positive_oos_folds", "n/a"),
                latest=_pct(metrics.get("latest_oos_return_pct")),
                readiness=item.get("readiness_label"),
            )
        )
    lines.extend(
        [
            "",
            "## External source boundary",
            "",
            f"- registry sources: `{payload['external_source_registry']['source_count']}`",
            f"- registry valid: `{payload['external_source_registry']['validation']['valid']}`",
            f"- excluded sources: `{', '.join(payload['external_source_registry']['validation']['excluded_source_ids'])}`",
            "",
            "## Readiness conclusion",
            "",
            f"- {payload['readiness_summary']['conclusion']}",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except TypeError, ValueError:
        return "n/a"
    if number != number or abs(number) == float("inf"):
        return "n/a"
    return f"{number:.4f}%"


def write_outputs(payload: Mapping[str, Any], *, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(payload)
    source_registry = snapshot.pop("source_registry_payload")

    snapshot_path = output_dir / "baseline_evidence_snapshot.json"
    snapshot_md_path = output_dir / "baseline_evidence_snapshot.md"
    registry_path = output_dir / "external_source_registry.json"
    sha_path = output_dir / "baseline_evidence_snapshot.sha256"

    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    snapshot_md_path.write_text(render_markdown(snapshot), encoding="utf-8")
    registry_path.write_text(
        json.dumps(source_registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sha_path.write_text(f"{_sha256_file(snapshot_path)}  {snapshot_path.name}\n", encoding="utf-8")
    return {
        "baseline_evidence_snapshot_json": str(snapshot_path),
        "baseline_evidence_snapshot_md": str(snapshot_md_path),
        "external_source_registry_json": str(registry_path),
        "baseline_evidence_snapshot_sha256": str(sha_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-wf-json", type=Path, default=DEFAULT_BASELINE_WF_JSON)
    parser.add_argument("--baseline-summary-json", type=Path, default=DEFAULT_BASELINE_SUMMARY_JSON)
    parser.add_argument("--prd-path", type=Path, default=DEFAULT_PRD_PATH)
    parser.add_argument("--test-spec-path", type=Path, default=DEFAULT_TEST_SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-at-utc")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build and validate payload, print a compact summary, and skip file writes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = build_payload(
        baseline_wf_json=args.baseline_wf_json,
        baseline_summary_json=args.baseline_summary_json,
        prd_path=args.prd_path,
        test_spec_path=args.test_spec_path,
        generated_at_utc=args.generated_at_utc,
    )
    if args.check_only:
        print(
            json.dumps(
                {
                    "baseline_snapshot_sha256": payload["baseline_evidence_snapshot_sha256"],
                    "latest_available_data_utc": payload["data_coverage"][
                        "latest_available_data_utc"
                    ],
                    "missing_required_artifacts": payload["missing_required_artifacts"],
                    "source_registry_valid": payload["external_source_registry"]["validation"][
                        "valid"
                    ],
                    "universe_sha256": payload["universe"]["universe_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(write_outputs(payload, output_dir=args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
