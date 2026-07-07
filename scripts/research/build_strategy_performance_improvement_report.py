#!/usr/bin/env python3
"""Build the strategy performance-improvement WF evidence envelope.

The monthly-refit walk-forward runner owns the expensive research computation.
This thin wrapper normalizes its JSON (or records a source-missing blocker) into
one review artifact with the fields needed by the 2026-07-07 team handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.optimization.native_backend import backend_selection_details  # noqa: E402

DEFAULT_REPORT_ROOT = Path("var/reports/strategy_performance_improvement_20260707")
DEFAULT_SOURCE_JSON = (
    DEFAULT_REPORT_ROOT
    / "full_universe_walkforward"
    / "full_universe_walkforward_summary_latest.json"
)
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_ROOT / "wf_report_normalized_latest.json"
DEFAULT_OUTPUT_MD = DEFAULT_REPORT_ROOT / "wf_report_normalized_latest.md"
DEFAULT_COMMAND_LOG = DEFAULT_REPORT_ROOT / "command_log.md"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", "utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except TypeError, ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _display_path(path: Path) -> str:
    """Return a stable repo-relative path for committed report artifacts."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _is_no_data_blocker(payload: Mapping[str, Any] | None) -> bool:
    if payload is None:
        return False
    decision = str(payload.get("pass_fail_decision") or "").lower()
    claim_status = str(payload.get("full_universe_claim_status") or "").lower()
    if decision == "fail_blocked_no_data":
        return True
    if claim_status == "blocked_not_claimed_missing_direct_1m_bars":
        return True
    blocker = str(payload.get("blocker") or "").lower()
    return (
        "no direct 1m" in blocker
        or "no direct 1m-derived" in blocker
        or "missing_direct_1m" in blocker
        or "missing direct 1m" in blocker
    )


def _is_blocked_payload(payload: Mapping[str, Any] | None) -> bool:
    if payload is None:
        return False
    status = str(payload.get("status") or "").lower()
    decision = str(payload.get("pass_fail_decision") or "").lower()
    return status == "blocked" or decision.startswith("fail_") or bool(payload.get("blocker"))


def _load_source(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    meta: dict[str, Any] = {
        "source_json_path": _display_path(resolved),
        "source_json_exists": resolved.exists(),
    }
    if not resolved.exists():
        meta.update(
            {
                "source_json_sha256": None,
                "source_status": "missing",
                "source_error": "source walk-forward JSON not found; report is a blocker envelope",
            }
        )
        return None, meta
    payload = json.loads(resolved.read_text("utf-8"))
    meta.update(
        {
            "source_json_sha256": _file_sha256(resolved),
            "source_status": "loaded",
            "source_error": None,
        }
    )
    return payload, meta


def data_coverage_counts(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "source": "source_json_missing",
            "requested_symbol_count": 0,
            "loaded_symbol_count": 0,
            "missing_symbol_count": None,
            "global_latest_utc": None,
            "timeframes": {},
        }
    if _is_no_data_blocker(payload):
        blocked_counts = dict(payload.get("data_coverage_counts") or {})
        loaded = _safe_int(
            blocked_counts.get("loaded"), _safe_int(blocked_counts.get("loaded_symbol_count"))
        )
        missing = _safe_int(
            blocked_counts.get("missing"), _safe_int(blocked_counts.get("missing_symbol_count"))
        )
        train_eligible = _safe_int(
            blocked_counts.get("train_eligible"),
            _safe_int(blocked_counts.get("train_eligible_symbol_count")),
        )
        monitor_only = _safe_int(
            blocked_counts.get("monitor_only"),
            _safe_int(blocked_counts.get("monitor_only_symbol_count")),
        )
        requested = _safe_int(
            blocked_counts.get("requested"),
            _safe_int(blocked_counts.get("requested_symbol_count"), loaded + missing),
        )
        return {
            "source": "blocked_walkforward_payload",
            "basis": blocked_counts.get("basis")
            or "walk-forward source reports a no-data blocker before selection",
            "requested_symbol_count": requested,
            "loaded_symbol_count": loaded,
            "missing_symbol_count": missing,
            "train_eligible_symbol_count": train_eligible,
            "monitor_only_symbol_count": monitor_only,
            "global_latest_utc": None,
            "timeframes": {},
        }
    coverage = dict(payload.get("data_coverage") or {})
    universe = dict(payload.get("universe") or {})
    missing_symbols = list(coverage.get("missing_symbols") or universe.get("missing_symbols") or [])
    timeframe_coverage = dict(payload.get("timeframe_coverage") or {})
    timeframe_counts: dict[str, Any] = {}
    for timeframe, row_raw in timeframe_coverage.items():
        row = dict(row_raw or {})
        timeframe_counts[str(timeframe)] = {
            "symbols_with_rows": _safe_int(row.get("symbols_with_rows")),
            "symbols_without_rows": _safe_int(row.get("symbols_without_rows")),
            "total_rows": _safe_int(row.get("total_rows")),
            "latest": row.get("latest"),
        }
    return {
        "source": "walkforward_payload",
        "requested_symbol_count": _safe_int(
            universe.get("requested_symbol_count"),
            _safe_int(
                coverage.get("requested_symbol_count"), _safe_int(universe.get("symbol_count"))
            ),
        ),
        "loaded_symbol_count": _safe_int(
            universe.get("loaded_symbol_count"), _safe_int(coverage.get("loaded_symbol_count"))
        ),
        "missing_symbol_count": _safe_int(
            universe.get("missing_symbol_count"),
            len(missing_symbols)
            if missing_symbols
            else _safe_int(coverage.get("missing_symbol_count")),
        ),
        "missing_symbols_sample": [str(item) for item in missing_symbols[:30]],
        "global_latest_utc": coverage.get("global_latest_utc"),
        "timeframes": timeframe_counts,
    }


def leak_checks(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "status": "not_evaluable_source_json_missing",
            "pass": False,
            "uses_locked_oos_for_selection_true": None,
            "real_money_execution_true": None,
            "current_fold_oos_used_for_weighting_true": None,
            "same_month_self_feeding_true": None,
            "dynamic_self_feed_audit_pass": None,
            "bridge_protocol_audit_pass": None,
        }
    if _is_no_data_blocker(payload):
        source_leaks = dict(payload.get("leak_checks") or {})
        return {
            "status": "not_evaluable_blocked_missing_direct_1m_bars",
            "pass": False,
            "basis": source_leaks.get("basis")
            or "Full WF blocked before selection because no direct 1m bars were loaded.",
            "uses_locked_oos_for_selection_true": bool(
                source_leaks.get("uses_locked_oos_for_selection", False)
            ),
            "real_money_execution_true": None,
            "current_fold_oos_used_for_weighting_true": bool(
                source_leaks.get("uses_locked_oos_for_weighting", False)
                or source_leaks.get("current_fold_oos_used_for_weighting", False)
            ),
            "same_month_self_feeding_true": None,
            "dynamic_self_feed_audit_pass": None,
            "bridge_protocol_audit_pass": None,
        }
    if _is_blocked_payload(payload):
        return {
            "status": "not_evaluable_blocked",
            "pass": False,
            "basis": str(payload.get("blocker") or "source payload is blocked before evaluation"),
            "uses_locked_oos_for_selection_true": None,
            "real_money_execution_true": None,
            "current_fold_oos_used_for_weighting_true": None,
            "same_month_self_feeding_true": None,
            "dynamic_self_feed_audit_pass": None,
            "bridge_protocol_audit_pass": None,
        }
    rows = [dict(row) for row in payload.get("fold_candidate_rows") or []]
    bool_counts: Counter[str] = Counter()
    for row in rows:
        for key in (
            "uses_locked_oos_for_selection",
            "real_money_execution",
            "current_fold_oos_used_for_weighting",
            "same_month_self_feeding",
        ):
            if bool(row.get(key)):
                bool_counts[key] += 1
    dynamic_audit = dict(payload.get("dynamic_self_feed_audit") or {})
    bridge_audit = dict(payload.get("bridge_protocol_audit") or {})
    dynamic_ok = bool(dynamic_audit.get("no_same_month_dynamic_self_feeding", True))
    bridge_ok = not any(
        bool(bridge_audit.get(key))
        for key in (
            "post_oos_expansion_for_same_protocol",
            "current_fold_oos_used_for_bridge_weighting",
            "same_month_dynamic_self_feeding",
        )
    )
    hard_fail_count = sum(bool_counts.values())
    return {
        "status": "evaluated",
        "pass": hard_fail_count == 0 and dynamic_ok and bridge_ok,
        "row_count": len(rows),
        "uses_locked_oos_for_selection_true": int(bool_counts["uses_locked_oos_for_selection"]),
        "real_money_execution_true": int(bool_counts["real_money_execution"]),
        "current_fold_oos_used_for_weighting_true": int(
            bool_counts["current_fold_oos_used_for_weighting"]
        ),
        "same_month_self_feeding_true": int(bool_counts["same_month_self_feeding"]),
        "dynamic_self_feed_audit_pass": dynamic_ok,
        "bridge_protocol_audit_pass": bridge_ok,
        "dynamic_self_feed_audit": dynamic_audit,
        "bridge_protocol_audit": bridge_audit,
    }


def chunk_sizes(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "source": "source_json_missing",
            "fold_count": 0,
            "fold_candidate_row_count": 0,
            "rows_by_fold": {},
        }
    if _is_no_data_blocker(payload):
        source_chunks = dict(payload.get("chunk_sizes") or {})
        return {
            "source": "blocked_walkforward_payload",
            "fold_count": 0,
            "fold_summary_count": 0,
            "fold_candidate_row_count": 0,
            "rows_by_fold": {},
            **source_chunks,
        }
    rows = [dict(row) for row in payload.get("fold_candidate_rows") or []]
    rows_by_fold = Counter(str(row.get("fold_id") or "<missing>") for row in rows)
    return {
        "source": "fold_candidate_rows",
        "fold_count": len(payload.get("folds") or []),
        "fold_summary_count": len(payload.get("fold_summaries") or []),
        "fold_candidate_row_count": len(rows),
        "rows_by_fold": dict(sorted(rows_by_fold.items())),
    }


def _derive_full_universe_claim_status(
    payload: Mapping[str, Any] | None,
    counts: Mapping[str, Any],
    override: str,
) -> str:
    if override != "auto":
        return override
    if payload is None:
        return "not_claimed_source_json_missing"
    if _is_no_data_blocker(payload):
        return "blocked_not_claimed_missing_direct_1m_bars"
    if _is_blocked_payload(payload):
        return "blocked_not_claimed_other"
    requested = _safe_int(counts.get("requested_symbol_count"))
    loaded = _safe_int(counts.get("loaded_symbol_count"))
    missing = _safe_int(counts.get("missing_symbol_count"))
    completed = bool(payload.get("completed_at_utc"))
    if requested > 0 and loaded >= requested and missing == 0 and completed:
        return "claimed_loaded_all_requested_symbols_completed_walkforward"
    if requested > 0 and missing == 0 and loaded >= requested:
        return "not_claimed_checkpoint_or_incomplete_walkforward"
    return "not_claimed_missing_or_unloaded_symbols"


def build_report(
    *,
    source_json: Path = DEFAULT_SOURCE_JSON,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    command_log_path: Path = DEFAULT_COMMAND_LOG,
    worker_count: int | None = None,
    full_universe_claim_status: str = "auto",
) -> dict[str, Any]:
    source_payload, source_meta = _load_source(source_json)
    coverage_counts = data_coverage_counts(source_payload)
    leaks = leak_checks(source_payload)
    chunks = chunk_sizes(source_payload)
    trial_policy = dict((source_payload or {}).get("trial_policy") or {})
    derived_worker_count = _safe_int(trial_policy.get("source_symbol_workers"), 1)
    if worker_count is not None:
        derived_worker_count = max(1, int(worker_count))
    peak_rss_mib = _safe_float((source_payload or {}).get("runner_peak_rss_mib"))
    if peak_rss_mib is None:
        peak_rss_mib = _safe_float((source_payload or {}).get("peak_rss_mb"))
    command_log = command_log_path.expanduser().resolve()
    output_json_resolved = output_json.expanduser().resolve()
    output_md_resolved = output_md.expanduser().resolve()
    report: dict[str, Any] = {
        "artifact_kind": "strategy_performance_improvement_wf_report",
        "generated_at_utc": _utc_now_iso(),
        **source_meta,
        "command_log_path": _display_path(command_log),
        "command_log_exists": command_log.exists(),
        "worker_count": derived_worker_count,
        "chunk_sizes": chunks,
        "peak_rss_mb": peak_rss_mib,
        "peak_rss_mib": peak_rss_mib,
        "peak_rss_source": (
            "runner_peak_rss_mib_or_blocker_peak_rss_mb"
            if peak_rss_mib is not None
            else "not_available"
        ),
        "native_backend_status": backend_selection_details(),
        "data_coverage_counts": coverage_counts,
        "leak_checks": leaks,
        "full_universe_claim_status": _derive_full_universe_claim_status(
            source_payload, coverage_counts, full_universe_claim_status
        ),
        "walkforward_summary": {
            "fold_count": len((source_payload or {}).get("folds") or []),
            "aggregate_ranking_count": len((source_payload or {}).get("aggregate_rankings") or []),
            "clean_promotion_count": len(
                (source_payload or {}).get("clean_promotion_rankings") or []
            ),
            "completed_at_utc": (source_payload or {}).get("completed_at_utc"),
            "output_paths": dict((source_payload or {}).get("output_paths") or {}),
            "status": (source_payload or {}).get("status"),
            "pass_fail_decision": (source_payload or {}).get("pass_fail_decision"),
            "selected_variant_count": len((source_payload or {}).get("selected_variants") or []),
        },
        "output_paths": {
            "json": _display_path(output_json_resolved),
            "markdown": _display_path(output_md_resolved),
        },
    }
    if source_payload and source_payload.get("blocker"):
        report["source_blocker"] = str(source_payload["blocker"])
    _write_json(output_json_resolved, report)
    output_md_resolved.parent.mkdir(parents=True, exist_ok=True)
    output_md_resolved.write_text(render_markdown(report), "utf-8")
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    source_status = report.get("source_status")
    counts = dict(report.get("data_coverage_counts") or {})
    leaks = dict(report.get("leak_checks") or {})
    chunks = dict(report.get("chunk_sizes") or {})
    native_status = dict(report.get("native_backend_status") or {})
    return "\n".join(
        [
            "# Strategy performance-improvement WF evidence report",
            "",
            f"- generated_at_utc: `{report.get('generated_at_utc')}`",
            f"- source_status: `{source_status}`",
            f"- source_json_path: `{report.get('source_json_path')}`",
            f"- command_log_path: `{report.get('command_log_path')}`",
            f"- worker_count: `{report.get('worker_count')}`",
            f"- fold_count: `{chunks.get('fold_count')}`",
            f"- fold_candidate_row_count: `{chunks.get('fold_candidate_row_count')}`",
            f"- peak_rss_mb: `{report.get('peak_rss_mb')}`",
            f"- peak_rss_source: `{report.get('peak_rss_source')}`",
            f"- native_backend: `{native_status.get('backend')}` (pyo3_available={native_status.get('pyo3_available')})",
            f"- requested/loaded/missing symbols: `{counts.get('requested_symbol_count')}` / `{counts.get('loaded_symbol_count')}` / `{counts.get('missing_symbol_count')}`",
            f"- latest_data_utc: `{counts.get('global_latest_utc')}`",
            f"- leak_checks_pass: `{leaks.get('pass')}`",
            f"- full_universe_claim_status: `{report.get('full_universe_claim_status')}`",
            "",
            "## Interpretation",
            "",
            "This wrapper is an evidence normalizer. It does not run or promote the expensive walk-forward by itself.",
            "If `source_status` is `missing`, treat this artifact as a concrete blocker report rather than full-universe success evidence.",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", default=str(DEFAULT_SOURCE_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--command-log-path", default=str(DEFAULT_COMMAND_LOG))
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument(
        "--full-universe-claim-status",
        default="auto",
        help="Override auto-derived claim status; use a not_claimed_* value for blocker artifacts.",
    )
    parser.add_argument(
        "--require-source",
        action="store_true",
        help="Exit non-zero if --source-json is missing after writing the blocker envelope.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        source_json=Path(args.source_json),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        command_log_path=Path(args.command_log_path),
        worker_count=args.worker_count,
        full_universe_claim_status=str(args.full_universe_claim_status),
    )
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    if args.require_source and report.get("source_status") != "loaded":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
