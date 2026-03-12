from __future__ import annotations

import json
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CANONICAL_REGISTRY_LATEST = "exact_window_backtest_registry_latest.json"
RECOVERED_REGISTRY_LATEST = "exact_window_backtest_registry_recovered_latest.json"
SIGNATURE_REGISTRY_LATEST = "exact_window_run_registry.jsonl"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _string_or_none(value: Any) -> str | None:
    token = str(value or "").strip()
    return token or None


def _listify(values: Any) -> list[str]:
    if not isinstance(values, list | tuple | set):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _prefer_string(primary: Any, secondary: Any) -> str | None:
    return _string_or_none(primary) or _string_or_none(secondary)


def _prefer_list(primary: Any, secondary: Any) -> list[str]:
    left = _listify(primary)
    return left or _listify(secondary)


def _prefer_bool(primary: Any, secondary: Any) -> bool:
    if isinstance(primary, bool):
        return primary
    if isinstance(secondary, bool):
        return secondary
    return bool(primary or secondary)


def _prefer_int(primary: Any, secondary: Any) -> int:
    try:
        left = int(primary)
        if left != 0:
            return left
    except (TypeError, ValueError):
        pass
    try:
        return int(secondary)
    except (TypeError, ValueError):
        return 0


def _prefer_float(primary: Any, secondary: Any) -> float:
    try:
        left = float(primary)
        if left > 0.0:
            return left
    except (TypeError, ValueError):
        pass
    try:
        right = float(secondary)
    except (TypeError, ValueError):
        return 0.0
    return right if right > 0.0 else 0.0


def _extract_json_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    start = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : index + 1]
                start = None
                try:
                    payload = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    blocks.append(payload)
    return blocks


def _parse_cli_metadata(text: str) -> dict[str, Any]:
    command_match = re.search(r'Command being timed:\s+"([^"]+)"', text)
    if not command_match:
        return {"requested_timeframes": [], "requested_symbols": [], "chunk_days": 0}
    tokens = shlex.split(command_match.group(1))
    requested_timeframes: list[str] = []
    requested_symbols: list[str] = []
    chunk_days = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "--timeframes":
            i += 1
            while i < len(tokens) and not tokens[i].startswith("--"):
                requested_timeframes.append(tokens[i])
                i += 1
            continue
        if token == "--symbols":
            i += 1
            while i < len(tokens) and not tokens[i].startswith("--"):
                requested_symbols.append(tokens[i])
                i += 1
            continue
        if token == "--chunk-days" and i + 1 < len(tokens):
            try:
                chunk_days = int(tokens[i + 1])
            except ValueError:
                chunk_days = 0
            i += 2
            continue
        i += 1
    return {
        "requested_timeframes": requested_timeframes,
        "requested_symbols": requested_symbols,
        "chunk_days": chunk_days,
    }


def _peak_rss_mib(text: str) -> float:
    match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)
    if not match:
        return 0.0
    return float(int(match.group(1)) / 1024.0)


def _manifest_enrichment(path: str | None) -> dict[str, Any]:
    token = _string_or_none(path)
    if not token:
        return {}
    payload = _json_load(Path(token))
    if payload is None:
        return {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    return {
        "batch_id": _string_or_none(payload.get("batch_id")),
        "run_signature": _string_or_none(payload.get("run_signature")),
        "status": _string_or_none(payload.get("status")),
        "summary_path": _prefer_string(payload.get("summary_path"), artifacts.get("summary")),
        "details_path": _prefer_string(payload.get("details_path"), artifacts.get("details")),
        "fail_analysis_path": _prefer_string(
            payload.get("fail_analysis_path"),
            artifacts.get("fail_analysis"),
        ),
        "memory_evidence_path": _prefer_string(
            payload.get("memory_evidence_path"),
            artifacts.get("memory_evidence"),
        ),
        "requested_timeframes": _listify(payload.get("requested_timeframes")),
        "requested_symbols": _listify(payload.get("requested_symbols")),
        "allow_metals": payload.get("allow_metals"),
        "requested_oos_end_exclusive": _string_or_none(payload.get("requested_oos_end_exclusive")),
        "train_start": _string_or_none(payload.get("train_start")),
        "val_start": _string_or_none(payload.get("val_start")),
        "oos_start": _string_or_none(payload.get("oos_start")),
        "window_profile": _string_or_none(payload.get("window_profile")),
        "chunk_days": payload.get("chunk_days"),
        "promoted_count": payload.get("promoted_count"),
        "evaluated_count": payload.get("evaluated_count"),
        "error": payload.get("error"),
    }


def _memory_peak_rss_mib(path: str | None) -> float:
    token = _string_or_none(path)
    if not token:
        return 0.0
    payload = _json_load(Path(token))
    if payload is None:
        return 0.0
    try:
        value = float(payload.get("peak_rss_mib") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0.0 else 0.0


def _summary_requested_timeframes(path: str | None) -> list[str]:
    token = _string_or_none(path)
    if not token:
        return []
    payload = _json_load(Path(token))
    if payload is None:
        return []
    execution_profile = payload.get("execution_profile")
    if not isinstance(execution_profile, dict):
        return []
    return _listify(execution_profile.get("requested_timeframes"))


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            token = raw.strip()
            if not token:
                continue
            try:
                payload = json.loads(token)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _merge_registry_entry(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(secondary)
    merged.update(primary)
    merged["run_id"] = _prefer_string(primary.get("run_id"), secondary.get("run_id")) or ""
    merged["batch_id"] = _prefer_string(primary.get("batch_id"), secondary.get("batch_id")) or ""
    primary_sig = _string_or_none(primary.get("run_signature"))
    secondary_sig = _string_or_none(secondary.get("run_signature"))
    if primary_sig and not primary_sig.startswith("log:"):
        merged["run_signature"] = primary_sig
    elif secondary_sig and not secondary_sig.startswith("log:"):
        merged["run_signature"] = secondary_sig
    else:
        merged["run_signature"] = primary_sig or secondary_sig or ""
    for key in (
        "status",
        "manifest_path",
        "summary_path",
        "details_path",
        "fail_analysis_path",
        "memory_evidence_path",
        "requested_oos_end_exclusive",
        "train_start",
        "val_start",
        "oos_start",
        "error",
        "window_profile",
        "log_path",
    ):
        merged[key] = _prefer_string(primary.get(key), secondary.get(key))
    for key in ("requested_timeframes", "requested_symbols"):
        merged[key] = _prefer_list(primary.get(key), secondary.get(key))
    merged["allow_metals"] = _prefer_bool(primary.get("allow_metals"), secondary.get("allow_metals"))
    for key in ("chunk_days", "promoted_count", "evaluated_count"):
        merged[key] = _prefer_int(primary.get(key), secondary.get(key))
    merged["peak_rss_mib"] = max(
        _prefer_float(primary.get("peak_rss_mib"), 0.0),
        _prefer_float(secondary.get("peak_rss_mib"), 0.0),
    )
    merged["updated_at_utc"] = _prefer_string(primary.get("updated_at_utc"), secondary.get("updated_at_utc")) or _now_iso()
    if not merged["batch_id"] and merged["requested_timeframes"]:
        merged["batch_id"] = "-".join(merged["requested_timeframes"])
    if not merged["window_profile"]:
        merged["window_profile"] = "archived_log"
    return merged


def _enrich_registry_entry(entry: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(entry)
    manifest = _manifest_enrichment(_string_or_none(entry.get("manifest_path")))
    enriched = _merge_registry_entry(manifest, enriched)
    if not enriched.get("requested_timeframes"):
        enriched["requested_timeframes"] = _summary_requested_timeframes(
            _string_or_none(enriched.get("summary_path"))
        )
    if not enriched.get("batch_id") and enriched.get("requested_timeframes"):
        enriched["batch_id"] = "-".join(enriched["requested_timeframes"])
    peak_from_memory = _memory_peak_rss_mib(_string_or_none(enriched.get("memory_evidence_path")))
    enriched["peak_rss_mib"] = max(
        _prefer_float(enriched.get("peak_rss_mib"), 0.0),
        peak_from_memory,
    )
    if not _string_or_none(enriched.get("window_profile")):
        enriched["window_profile"] = "archived_log"
    enriched["updated_at_utc"] = _now_iso()
    return enriched


def _canonical_signature_entries(
    *,
    report_root: Path,
    existing_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    entries_by_run_id: dict[str, dict[str, Any]] = {}
    for entry in list(existing_entries or []):
        run_id = str(entry.get("run_id") or "").strip()
        run_signature = str(entry.get("run_signature") or "").strip()
        if not run_id or not run_signature or run_signature.startswith("log:"):
            continue
        entries_by_run_id[run_id] = _enrich_registry_entry(entry)

    signature_registry_path = report_root / SIGNATURE_REGISTRY_LATEST
    for record in _iter_jsonl(signature_registry_path):
        run_id = str(record.get("run_id") or "").strip()
        if not run_id:
            continue
        batch_id = str(record.get("batch_id") or "").strip()
        manifest_path = _string_or_none(record.get("manifest_path"))
        manifest = _manifest_enrichment(manifest_path)
        entry = {
            "run_id": run_id,
            "batch_id": batch_id or _string_or_none(manifest.get("batch_id")) or "",
            "run_signature": _string_or_none(record.get("signature")) or _string_or_none(manifest.get("run_signature")) or "",
            "status": _string_or_none(record.get("status")) or _string_or_none(manifest.get("status")) or "",
            "manifest_path": manifest_path,
            "summary_path": _prefer_string(record.get("summary_path"), manifest.get("summary_path")),
            "details_path": _prefer_string(record.get("details_path"), manifest.get("details_path")),
            "fail_analysis_path": _prefer_string(record.get("fail_analysis_path"), manifest.get("fail_analysis_path")),
            "memory_evidence_path": _prefer_string(record.get("memory_evidence_path"), manifest.get("memory_evidence_path")),
            "requested_timeframes": _prefer_list(
                manifest.get("requested_timeframes"),
                batch_id.split("-") if batch_id else [],
            ),
            "requested_symbols": _listify(manifest.get("requested_symbols")),
            "allow_metals": manifest.get("allow_metals"),
            "requested_oos_end_exclusive": _prefer_string(
                manifest.get("requested_oos_end_exclusive"),
                record.get("requested_oos_end_exclusive"),
            ),
            "train_start": _prefer_string(manifest.get("train_start"), record.get("train_start")),
            "val_start": _prefer_string(manifest.get("val_start"), record.get("val_start")),
            "oos_start": _prefer_string(manifest.get("oos_start"), record.get("oos_start")),
            "window_profile": _prefer_string(manifest.get("window_profile"), record.get("window_profile")),
            "chunk_days": _prefer_int(manifest.get("chunk_days"), record.get("chunk_days")),
            "promoted_count": _prefer_int(manifest.get("promoted_count"), record.get("promoted_count")),
            "evaluated_count": _prefer_int(manifest.get("evaluated_count"), record.get("evaluated_count")),
            "peak_rss_mib": _memory_peak_rss_mib(
                _prefer_string(record.get("memory_evidence_path"), manifest.get("memory_evidence_path"))
            ),
            "error": record.get("error") or manifest.get("error"),
            "updated_at_utc": _prefer_string(record.get("completed_at_utc"), record.get("started_at_utc")) or _now_iso(),
        }
        entries_by_run_id[run_id] = _merge_registry_entry(
            _enrich_registry_entry(entry),
            entries_by_run_id.get(run_id, {}),
        )
    return sorted(entries_by_run_id.values(), key=lambda item: str(item.get("updated_at_utc") or ""), reverse=True)


def scan_exact_window_logs(log_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    patterns = ("exact_window_*.log", "lq-exact-*.log")
    for pattern in patterns:
        for path in sorted(log_dir.glob(pattern)):
            text = path.read_text(encoding="utf-8", errors="ignore")
            blocks = _extract_json_blocks(text)
            if not blocks:
                continue
            cli_meta = _parse_cli_metadata(text)
            peak_rss_mib = _peak_rss_mib(text)
            for payload in blocks:
                if "run_id" not in payload or "status" not in payload:
                    continue
                if "manifest_path" not in payload and "snapshot" not in payload and "active_run" in payload:
                    continue
                requested_timeframes = list(cli_meta.get("requested_timeframes") or [])
                if not requested_timeframes and payload.get("timeframe"):
                    requested_timeframes = [str(payload.get("timeframe"))]
                batch_id = "-".join(requested_timeframes) if requested_timeframes else str(payload.get("batch_id") or "")
                entry = {
                    "run_id": str(payload.get("run_id") or ""),
                    "batch_id": batch_id,
                    "run_signature": f"log:{path.stem}:{payload.get('run_id')}",
                    "status": str(payload.get("status") or ""),
                    "manifest_path": payload.get("manifest_path"),
                    "summary_path": payload.get("summary_latest") or payload.get("summary_path"),
                    "details_path": payload.get("details_latest") or payload.get("details_path"),
                    "fail_analysis_path": payload.get("fail_analysis_latest") or payload.get("fail_analysis_path"),
                    "memory_evidence_path": payload.get("memory_evidence_latest") or payload.get("memory_evidence_path"),
                    "requested_timeframes": requested_timeframes,
                    "requested_symbols": list(cli_meta.get("requested_symbols") or payload.get("eligible_symbols") or []),
                    "allow_metals": bool(payload.get("allow_metals")),
                    "requested_oos_end_exclusive": None,
                    "train_start": None,
                    "val_start": None,
                    "oos_start": None,
                    "window_profile": "archived_log",
                    "chunk_days": int(cli_meta.get("chunk_days") or 0),
                    "promoted_count": int(payload.get("promoted_count") or 0),
                    "evaluated_count": int(payload.get("evaluated_count") or 0),
                    "peak_rss_mib": peak_rss_mib,
                    "error": payload.get("error"),
                    "log_path": str(path.resolve()),
                    "updated_at_utc": _now_iso(),
                }
                entries.append(_enrich_registry_entry(entry))
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        run_id = str(entry.get("run_id") or "").strip()
        if not run_id:
            continue
        existing = deduped.get(run_id)
        deduped[run_id] = _merge_registry_entry(entry, existing or {})
    return sorted(deduped.values(), key=lambda item: (str(item.get("run_id")), str(item.get("log_path"))))


def write_exact_window_canonical_registry(*, report_root: Path) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    registry_path = report_root / CANONICAL_REGISTRY_LATEST
    existing_payload = _json_load(registry_path) if registry_path.exists() else None
    existing_entries = [
        dict(row)
        for row in list((existing_payload or {}).get("entries") or [])
        if isinstance(row, dict)
    ]
    entries = _canonical_signature_entries(report_root=report_root, existing_entries=existing_entries)
    payload = {
        "schema_version": "1.0",
        "artifact_kind": "exact_window_canonical_registry",
        "provenance": "derived_from_exact_window_run_registry_jsonl",
        "generated_at": _now_iso(),
        "entry_count": len(entries),
        "entries": entries,
    }
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "registry_path": str(registry_path.resolve()),
        "entry_count": len(entries),
        "source_registry": str((report_root / SIGNATURE_REGISTRY_LATEST).resolve()),
    }


def write_exact_window_log_archive(*, log_dir: Path, report_root: Path) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    followup_root = report_root / "followup_status"
    followup_root.mkdir(parents=True, exist_ok=True)
    log_entries = scan_exact_window_logs(log_dir)
    recovered_registry_path = report_root / RECOVERED_REGISTRY_LATEST
    existing_payload = _json_load(recovered_registry_path) if recovered_registry_path.exists() else None
    existing_entries = [
        dict(row)
        for row in list((existing_payload or {}).get("entries") or [])
        if isinstance(row, dict)
    ]
    merged_by_run_id: dict[str, dict[str, Any]] = {}
    for entry in existing_entries:
        run_id = str(entry.get("run_id") or "").strip()
        if run_id:
            merged_by_run_id[run_id] = _enrich_registry_entry(entry)
    for entry in log_entries:
        run_id = str(entry.get("run_id") or "").strip()
        if not run_id:
            continue
        merged_by_run_id[run_id] = _merge_registry_entry(entry, merged_by_run_id.get(run_id, {}))
    entries = sorted(
        merged_by_run_id.values(),
        key=lambda item: (str(item.get("run_id")), str(item.get("log_path"))),
    )
    registry_payload = {
        "schema_version": "1.0",
        "artifact_kind": "exact_window_recovered_registry",
        "provenance": "recovered_from_logs_plus_prior_archive",
        "generated_at": _now_iso(),
        "entry_count": len(entries),
        "entries": entries,
    }
    recovered_registry_path.write_text(json.dumps(registry_payload, indent=2, sort_keys=True), encoding="utf-8")

    archive_payload = {
        "schema_version": "1.0",
        "artifact_kind": "exact_window_log_archive",
        "provenance": "recovered_from_logs_plus_prior_archive",
        "generated_at": _now_iso(),
        "entry_count": len(entries),
        "entries": entries,
        "note": (
            "Recovered advisory archive from logs and prior archive snapshots. "
            "Do not treat this as the canonical duplicate-signature registry."
        ),
    }
    archive_json = followup_root / "backtest_log_archive_latest.json"
    archive_md = followup_root / "backtest_log_archive_latest.md"
    archive_json.write_text(json.dumps(archive_payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# exact-window backtest log archive",
        "",
        f"- generated_at: `{archive_payload['generated_at']}`",
        f"- entry_count: `{len(entries)}`",
        "- note: recovered advisory archive only; canonical duplicate checks must use `exact_window_run_registry.jsonl`.",
        "",
        "## runs",
    ]
    for entry in entries:
        lines.append(
            f"- `{entry['run_id']}` | status={entry['status']} | tf={','.join(entry['requested_timeframes']) or entry['batch_id'] or 'n/a'} | "
            f"peak_rss={float(entry['peak_rss_mib'] or 0.0):.1f} MiB | log={entry['log_path']}"
        )
    archive_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "recovered_registry_path": str(recovered_registry_path.resolve()),
        "archive_json": str(archive_json.resolve()),
        "archive_md": str(archive_md.resolve()),
        "entry_count": len(entries),
    }


__all__ = [
    "CANONICAL_REGISTRY_LATEST",
    "RECOVERED_REGISTRY_LATEST",
    "SIGNATURE_REGISTRY_LATEST",
    "scan_exact_window_logs",
    "write_exact_window_canonical_registry",
    "write_exact_window_log_archive",
]
