"""Reporting helpers for exact-window suite artifacts and diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SUMMARY_LATEST = "exact_window_suite_summary_latest.json"
DETAILS_LATEST = "exact_window_candidate_details_latest.json"
FAIL_ANALYSIS_LATEST = "exact_window_fail_analysis_latest.json"
FAIL_ANALYSIS_MD_LATEST = "exact_window_fail_analysis_latest.md"
RSS_LOG_LATEST = "exact_window_rss_latest.jsonl"
MEMORY_EVIDENCE_LATEST = "exact_window_memory_evidence_latest.json"
MEMORY_EVIDENCE_MD_LATEST = "exact_window_memory_evidence_latest.md"
REGISTRY_LATEST = "exact_window_backtest_registry_latest.json"

_MAX_REGISTRY_ENTRIES = 200

_LATEST_ARTIFACT_NAMES = {
    "summary": SUMMARY_LATEST,
    "details": DETAILS_LATEST,
    "fail_analysis": FAIL_ANALYSIS_LATEST,
    "fail_analysis_md": FAIL_ANALYSIS_MD_LATEST,
    "rss_log": RSS_LOG_LATEST,
    "memory_evidence": MEMORY_EVIDENCE_LATEST,
    "memory_evidence_md": MEMORY_EVIDENCE_MD_LATEST,
    "registry": REGISTRY_LATEST,
}


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _text_dump(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(payload), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _registry_signature(payload: dict[str, Any]) -> str:
    signature_payload = {
        "requested_symbols": sorted([str(value or "") for value in list(payload.get("requested_symbols") or [])]),
        "requested_timeframes": sorted([str(value or "") for value in list(payload.get("requested_timeframes") or [])]),
        "allow_metals": bool(payload.get("allow_metals")),
        "window_profile": str(payload.get("window_profile") or ""),
        "train_start": str(payload.get("train_start") or ""),
        "val_start": str(payload.get("val_start") or ""),
        "oos_start": str(payload.get("oos_start") or ""),
        "requested_oos_end_exclusive": str(payload.get("requested_oos_end_exclusive") or ""),
        "chunk_days": int(payload.get("chunk_days") or 0),
    }
    token = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _registry_entry(
    run_id: str,
    batch_id: str,
    *,
    status: str,
    run_signature: str,
    manifest_path: str,
    summary_path: str | None,
    details_path: str | None,
    fail_analysis_path: str | None,
    memory_evidence_path: str | None,
    requested_timeframes: list[str],
    requested_symbols: list[str],
    allow_metals: bool,
    batch_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "batch_id": batch_id,
        "run_signature": run_signature,
        "status": status,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "details_path": details_path,
        "fail_analysis_path": fail_analysis_path,
        "memory_evidence_path": memory_evidence_path,
        "requested_timeframes": requested_timeframes,
        "requested_symbols": requested_symbols,
        "allow_metals": bool(allow_metals),
        "requested_oos_end_exclusive": batch_payload.get("requested_oos_end_exclusive"),
        "train_start": batch_payload.get("train_start"),
        "val_start": batch_payload.get("val_start"),
        "oos_start": batch_payload.get("oos_start"),
        "window_profile": batch_payload.get("window_profile"),
        "chunk_days": int(batch_payload.get("chunk_days") or 0),
        "promoted_count": int(batch_payload.get("promoted_count") or 0),
        "evaluated_count": int(batch_payload.get("evaluated_count") or 0),
        "peak_rss_mib": float(batch_payload.get("peak_rss_mib") or 0.0),
        "error": batch_payload.get("error"),
        "updated_at_utc": _now_iso(),
    }


def resolve_exact_window_artifact_paths(
    output_dir: str | Path = "var/reports/exact_window_backtests",
) -> dict[str, Path | None]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest_pointer = root / "latest.json"
    run_root = root
    if latest_pointer.exists():
        try:
            pointer = _json_load(latest_pointer)
        except Exception:
            pointer = None
        if isinstance(pointer, dict):
            run_dir = str(pointer.get("run_dir") or "").strip()
            run_id = str(pointer.get("run_id") or "").strip()
            candidates = [Path(run_dir)] if run_dir else []
            if run_id:
                candidates.append(root / run_id)
            for candidate in candidates:
                resolved = candidate if candidate.is_absolute() else (root / candidate)
                if resolved.exists():
                    run_root = resolved.resolve()
                    break

    def _resolve_latest(name: str) -> Path:
        direct = root / name
        if direct.exists():
            return direct.resolve()
        preferred = run_root / name
        if preferred.exists():
            return preferred.resolve()
        return preferred.resolve() if run_root != root else (root / name).resolve()

    return {
        "root": root,
        "run_root": run_root,
        "latest_pointer": latest_pointer if latest_pointer.exists() else None,
        "summary": _resolve_latest(SUMMARY_LATEST),
        "details": _resolve_latest(DETAILS_LATEST),
        "fail_analysis": _resolve_latest(FAIL_ANALYSIS_LATEST),
        "fail_analysis_md": _resolve_latest(FAIL_ANALYSIS_MD_LATEST),
        "rss_log": _resolve_latest(RSS_LOG_LATEST),
        "memory_evidence": _resolve_latest(MEMORY_EVIDENCE_LATEST),
        "memory_evidence_md": _resolve_latest(MEMORY_EVIDENCE_MD_LATEST),
        "registry": _resolve_latest(REGISTRY_LATEST),
    }


def sync_exact_window_latest_aliases(
    output_dir: str | Path = "var/reports/exact_window_backtests",
    *,
    artifact_paths: dict[str, Path | None] | None = None,
) -> dict[str, Path | None]:
    resolved_paths = dict(artifact_paths or resolve_exact_window_artifact_paths(output_dir))
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for key, filename in _LATEST_ARTIFACT_NAMES.items():
        source = resolved_paths.get(key)
        if not isinstance(source, Path) or not source.exists():
            continue
        target = root / filename
        if source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        resolved_paths[key] = target.resolve()

    resolved_paths["root"] = root
    run_root = resolved_paths.get("run_root")
    resolved_paths["run_root"] = Path(run_root).resolve() if isinstance(run_root, Path) else root
    latest_pointer = resolved_paths.get("latest_pointer")
    resolved_paths["latest_pointer"] = (
        Path(latest_pointer).resolve()
        if isinstance(latest_pointer, Path) and latest_pointer.exists()
        else (root / "latest.json").resolve() if (root / "latest.json").exists() else None
    )

    registry_source = resolved_paths.get("registry")
    if isinstance(registry_source, Path) and registry_source.exists():
        source_payload = _json_load(registry_source)
        if isinstance(source_payload, dict):
            target = root / REGISTRY_LATEST
            target.write_text(json.dumps(source_payload, sort_keys=True, indent=2), encoding="utf-8")
            resolved_paths["registry"] = target.resolve()
    return resolved_paths


def resolve_backtest_registry(output_dir: str | Path = "var/reports/exact_window_backtests") -> list[dict[str, Any]]:
    paths = resolve_exact_window_artifact_paths(output_dir)
    registry_path = paths.get("registry")
    if not isinstance(registry_path, Path):
        return []
    if not registry_path.exists():
        return []
    payload = _json_load(registry_path)
    if not isinstance(payload, dict):
        return []
    return [dict(row) for row in list(payload.get("entries") or []) if isinstance(row, dict)]


def upsert_backtest_registry(
    output_dir: str | Path,
    *,
    run_id: str,
    batch_id: str,
    status: str,
    run_signature: str,
    manifest_path: str,
    summary_path: str | None,
    details_path: str | None,
    fail_analysis_path: str | None,
    memory_evidence_path: str | None,
    requested_timeframes: list[str],
    requested_symbols: list[str],
    allow_metals: bool,
    batch_payload: dict[str, Any],
    max_entries: int = _MAX_REGISTRY_ENTRIES,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / REGISTRY_LATEST
    existing = _json_load(registry_path) if registry_path.exists() else {}
    if not isinstance(existing, dict):
        existing = {}
    entries = [dict(row) for row in list(existing.get("entries") or []) if isinstance(row, dict)]

    new_entry = _registry_entry(
        run_id=run_id,
        batch_id=batch_id,
        status=status,
        run_signature=run_signature,
        manifest_path=manifest_path,
        summary_path=summary_path,
        details_path=details_path,
        fail_analysis_path=fail_analysis_path,
        memory_evidence_path=memory_evidence_path,
        requested_timeframes=requested_timeframes,
        requested_symbols=requested_symbols,
        allow_metals=allow_metals,
        batch_payload=batch_payload,
    )

    updated_entries: list[dict[str, Any]] = []
    replaced = False
    for entry in entries:
        if (
            str(entry.get("run_id") or "")
            == str(new_entry.get("run_id") or "")
            or str(entry.get("run_signature") or "")
            == str(new_entry.get("run_signature") or "")
        ):
            updated_entries.append(new_entry)
            replaced = True
            continue
        updated_entries.append(entry)
    if not replaced:
        updated_entries.append(new_entry)

    unique: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for entry in sorted(
        updated_entries,
        key=lambda item: str(item.get("updated_at_utc") or ""),
        reverse=True,
    ):
        key = (str(entry.get("run_id") or ""), str(entry.get("run_signature") or ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(entry)

    bounded = unique[: int(max_entries)]
    payload = {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "entry_count": len(bounded),
        "entries": bounded,
    }
    _json_dump(registry_path, payload)
    return payload


def _reason_stage(reason: str) -> str:
    if reason in {"train_hurdle", "validation_hurdle", "oos_hurdle"}:
        return reason.replace("_hurdle", "")
    if reason in {"oos_sharpe", "pbo", "turnover", "max_drawdown", "trade_count"}:
        return "oos_hard_reject"
    if reason == "rss_guard":
        return "memory_guard"
    if reason == "promoted":
        return "promoted"
    if reason == "skipped":
        return "skipped"
    return "selection"


def _reason_note(reason: str, row: dict[str, Any]) -> str:
    hard_reject = dict(row.get("hard_reject_reasons") or {})
    if reason in hard_reject:
        return f"{reason}={hard_reject[reason]}"
    if reason == "rss_guard":
        return "rss guard triggered in candidate metadata"
    if reason == "skipped":
        return "candidate marked skipped in metadata"
    if reason == "train_hurdle":
        return "train hurdle failed before promotion"
    if reason == "validation_hurdle":
        return "validation hurdle failed before promotion"
    if reason == "oos_hurdle":
        return "oos hurdle failed without explicit hard reject key"
    if reason == "promoted":
        return "candidate promoted by summary best_per_strategy selection"
    return "candidate did not promote"


def _rejection_reasons(row: dict[str, Any], *, promoted_ids: set[str]) -> list[str]:
    reasons: list[str] = []
    metadata = dict(row.get("metadata") or {})
    if bool(metadata.get("rss_guard_triggered")):
        reasons.append("rss_guard")
    if bool(metadata.get("skipped")):
        reasons.append("skipped")
    for key in sorted(dict(row.get("hard_reject_reasons") or {})):
        if key not in reasons:
            reasons.append(key)
    if reasons:
        return reasons

    hurdle_fields = dict(row.get("hurdle_fields") or {})
    train_pass = bool((hurdle_fields.get("train") or {}).get("pass"))
    val_pass = bool((hurdle_fields.get("val") or {}).get("pass"))
    oos_pass = bool((hurdle_fields.get("oos") or {}).get("pass"))
    candidate_id = str(row.get("candidate_id") or "")
    if candidate_id and candidate_id in promoted_ids:
        return ["promoted"]
    if not train_pass:
        return ["train_hurdle"]
    if not val_pass:
        return ["validation_hurdle"]
    if not oos_pass:
        return ["oos_hurdle"]
    return ["not_promoted"]


def _proposal_text(reason: str, *, timeframes: list[str]) -> str:
    if reason == "oos_sharpe":
        return (
            "Prioritize evidence-backed parameter expansion or new candidate families on "
            f"{', '.join(timeframes) or 'the weak timeframes'} where OOS Sharpe dominates failures."
        )
    if reason == "trade_count":
        return (
            "Increase trade opportunity on the flagged timeframes via broader parameter grids, "
            "symbol additions, or lower-frequency entry filters."
        )
    if reason == "pbo":
        return (
            "Reduce overfit pressure by simplifying parameter grids and keeping only the most "
            "stable families on the affected timeframes."
        )
    if reason == "max_drawdown":
        return "Tighten risk controls or downweight the highest-drawdown families before reruns."
    if reason in {"train_hurdle", "validation_hurdle"}:
        return (
            "Investigate feature quality and candidate construction before expanding the search "
            "space on the failing split."
        )
    if reason == "rss_guard":
        return "Reduce chunk_days or candidate batch size before the next monitored rerun."
    return "Use the grouped fail counts to choose one bounded follow-up change set before rerunning."


def build_fail_analysis(summary: dict[str, Any], details: list[dict[str, Any]]) -> dict[str, Any]:
    promoted_ids = {
        str(row.get("candidate_id") or "")
        for row in list(summary.get("best_per_strategy") or [])
        if bool(row.get("promoted"))
    }
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    timeframe_reason_counts: Counter[tuple[str, str]] = Counter()
    strategy_reason_counts: Counter[tuple[str, str]] = Counter()

    for detail in list(details or []):
        reasons = _rejection_reasons(detail, promoted_ids=promoted_ids)
        timeframe = str(detail.get("strategy_timeframe") or detail.get("timeframe") or "unknown")
        strategy_class = str(detail.get("strategy_class") or "unknown")
        oos = dict(detail.get("oos") or {})
        metadata = dict(detail.get("metadata") or {})
        for reason in reasons:
            row = {
                "candidate_id": str(detail.get("candidate_id") or ""),
                "name": str(detail.get("name") or ""),
                "family": str(detail.get("family") or ""),
                "timeframe": timeframe,
                "strategy_class": strategy_class,
                "rejection_reason": reason,
                "hurdle_type": _reason_stage(reason),
                "oos_trade_count": float(oos.get("trade_count", 0.0)),
                "oos_mdd": float(oos.get("mdd", 0.0)),
                "oos_sharpe": float(oos.get("sharpe", 0.0)),
                "rss_guard_triggered": bool(metadata.get("rss_guard_triggered") or reason == "rss_guard"),
                "skipped": bool(metadata.get("skipped") or reason == "skipped"),
                "notes": _reason_note(reason, detail),
            }
            rows.append(row)
            reason_counts[reason] += 1
            timeframe_reason_counts[(timeframe, reason)] += 1
            strategy_reason_counts[(strategy_class, reason)] += 1

    proposals = []
    for reason, count in reason_counts.most_common(3):
        if reason == "promoted":
            continue
        impacted_rows = [row for row in rows if row["rejection_reason"] == reason]
        timeframes = sorted({str(row["timeframe"]) for row in impacted_rows})
        strategy_classes = sorted({str(row["strategy_class"]) for row in impacted_rows})
        proposals.append(
            {
                "rejection_reason": reason,
                "count": int(count),
                "timeframes": timeframes,
                "strategy_classes": strategy_classes,
                "proposal": _proposal_text(reason, timeframes=timeframes),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_generated_at": str(summary.get("generated_at") or ""),
        "windows": dict(summary.get("windows") or {}),
        "execution_profile": dict(summary.get("execution_profile") or {}),
        "promoted_count": int(summary.get("promoted_count") or 0),
        "evaluated_count": int(summary.get("evaluated_count") or len(details or [])),
        "rows": rows,
        "counts_by_rejection_reason": [
            {"rejection_reason": reason, "count": int(count)}
            for reason, count in reason_counts.most_common()
        ],
        "counts_by_timeframe_reason": [
            {"timeframe": timeframe, "rejection_reason": reason, "count": int(count)}
            for (timeframe, reason), count in sorted(
                timeframe_reason_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
        "counts_by_strategy_reason": [
            {"strategy_class": strategy_class, "rejection_reason": reason, "count": int(count)}
            for (strategy_class, reason), count in sorted(
                strategy_reason_counts.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
        "strategy_next_steps": proposals,
    }


def _render_fail_analysis_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exact-Window Fail Analysis",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Source summary generated at: `{payload.get('source_generated_at', '')}`",
        f"- Evaluated candidates: {int(payload.get('evaluated_count', 0))}",
        f"- Promoted candidates: {int(payload.get('promoted_count', 0))}",
        f"- Clamp max timestamp: `{payload.get('windows', {}).get('actual_max_timestamp', '')}`",
        "",
        "## Rejection Counts",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for row in list(payload.get("counts_by_rejection_reason") or []):
        lines.append(f"| {row.get('rejection_reason', '')} | {int(row.get('count', 0))} |")

    lines.extend(["", "## Suggested Next Steps", ""])
    proposals = list(payload.get("strategy_next_steps") or [])
    if not proposals:
        lines.append("- No follow-up proposals generated.")
    else:
        for row in proposals:
            lines.append(
                "- "
                f"{row.get('rejection_reason', '')}: {row.get('proposal', '')} "
                f"(count={int(row.get('count', 0))}, timeframes={', '.join(row.get('timeframes') or [])})"
            )
    return "\n".join(lines) + "\n"


def write_fail_analysis_bundle(
    *,
    output_dir: str | Path = "var/reports/exact_window_backtests",
    summary: dict[str, Any] | None = None,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    paths = resolve_exact_window_artifact_paths(output_dir)
    summary_path = Path(paths["summary"] or Path(output_dir) / SUMMARY_LATEST)
    details_path = Path(paths["details"] or Path(output_dir) / DETAILS_LATEST)
    resolved_summary = summary if summary is not None else _json_load(summary_path)
    resolved_details = details if details is not None else _json_load(details_path)
    if not isinstance(resolved_summary, dict):
        raise ValueError(f"summary payload must be an object: {summary_path}")
    if not isinstance(resolved_details, list):
        raise ValueError(f"details payload must be a list: {details_path}")

    payload = build_fail_analysis(resolved_summary, resolved_details)
    run_root = Path(paths["run_root"] or Path(output_dir)).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = run_root / f"exact_window_fail_analysis_{stamp}.json"
    md_path = run_root / f"exact_window_fail_analysis_{stamp}.md"
    json_latest = run_root / FAIL_ANALYSIS_LATEST
    md_latest = run_root / FAIL_ANALYSIS_MD_LATEST
    _json_dump(json_path, payload)
    _json_dump(json_latest, payload)
    markdown = _render_fail_analysis_markdown(payload)
    _text_dump(md_path, markdown)
    _text_dump(md_latest, markdown)
    return {
        "payload": payload,
        "json_path": json_path,
        "json_latest": json_latest,
        "md_path": md_path,
        "md_latest": md_latest,
    }


def build_memory_evidence(
    memory_summary: dict[str, Any],
    *,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(memory_summary)
    payload["generated_at"] = datetime.now(UTC).isoformat()
    if isinstance(summary, dict):
        payload["windows"] = dict(summary.get("windows") or {})
        payload["execution_profile"] = dict(summary.get("execution_profile") or {})
    return payload


def _render_memory_evidence_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exact-Window Memory Evidence",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Status: `{payload.get('status', '')}`",
        f"- Peak RSS MiB: {float(payload.get('peak_rss_mib') or 0.0):.2f}",
        f"- Budget MiB: {float(payload.get('budget_mib') or 0.0):.2f}",
        f"- Soft limit MiB: {float(payload.get('soft_limit_mib') or 0.0):.2f}",
        f"- Hard limit MiB: {float(payload.get('hard_limit_mib') or 0.0):.2f}",
        f"- RSS log: `{payload.get('rss_log_path', '')}`",
    ]
    if payload.get("error"):
        lines.append(f"- Error: `{payload.get('error')}`")
    return "\n".join(lines) + "\n"


def write_memory_evidence_bundle(
    *,
    output_dir: str | Path = "var/reports/exact_window_backtests",
    memory_summary: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = resolve_exact_window_artifact_paths(output_dir)
    run_root = Path(paths["run_root"] or Path(output_dir)).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = build_memory_evidence(memory_summary, summary=summary)
    json_path = run_root / f"exact_window_memory_evidence_{stamp}.json"
    md_path = run_root / f"exact_window_memory_evidence_{stamp}.md"
    json_latest = run_root / MEMORY_EVIDENCE_LATEST
    md_latest = run_root / MEMORY_EVIDENCE_MD_LATEST
    _json_dump(json_path, payload)
    _json_dump(json_latest, payload)
    markdown = _render_memory_evidence_markdown(payload)
    _text_dump(md_path, markdown)
    _text_dump(md_latest, markdown)
    return {
        "payload": payload,
        "json_path": json_path,
        "json_latest": json_latest,
        "md_path": md_path,
        "md_latest": md_latest,
    }


__all__ = [
    "DETAILS_LATEST",
    "FAIL_ANALYSIS_LATEST",
    "MEMORY_EVIDENCE_LATEST",
    "REGISTRY_LATEST",
    "RSS_LOG_LATEST",
    "SUMMARY_LATEST",
    "_registry_signature",
    "build_fail_analysis",
    "build_memory_evidence",
    "resolve_backtest_registry",
    "resolve_exact_window_artifact_paths",
    "sync_exact_window_latest_aliases",
    "upsert_backtest_registry",
    "write_fail_analysis_bundle",
    "write_memory_evidence_bundle",
]
