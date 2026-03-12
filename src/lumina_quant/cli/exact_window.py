from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lumina_quant.config import BaseConfig
from lumina_quant.eval.exact_window_reporting import (
    DETAILS_LATEST,
    MEMORY_EVIDENCE_LATEST,
    RSS_LOG_LATEST,
    SUMMARY_LATEST,
    upsert_backtest_registry,
    sync_exact_window_latest_aliases,
    write_fail_analysis_bundle,
    write_memory_evidence_bundle,
)
from lumina_quant.eval.exact_window_runtime import (
    HeavyRunActiveError,
    HeavyRunLock,
    RSSGuard,
    RSSLimitExceeded,
)
from lumina_quant.eval.exact_window_suite import (
    LOW_RAM_EXCLUDED_TIMEFRAMES,
    resolve_coverage_adaptive_windows,
    run_exact_window_suite,
)
from lumina_quant.symbols import CANONICAL_STRATEGY_TIMEFRAMES, normalize_strategy_timeframes


def _load_score_config(path: str) -> dict[str, Any] | None:
    token = str(path or "").strip()
    if not token:
        return None
    payload = json.loads(Path(token).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"score config must be a JSON object: {path}")
    return payload


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_stamp() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_progress_row(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "timestamp_utc",
        "batch_id",
        "status",
        "timeframes",
        "evaluated_count",
        "promoted_count",
        "summary_path",
        "details_path",
        "rss_log_path",
        "fail_analysis_path",
        "memory_evidence_path",
        "notes",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _git_commit_marker() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        dirty = False
    return commit, dirty


def _candidate_library_hash() -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "strategy_factory"
        / "candidate_library.py"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_signature_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        try:
            return [str(item).strip() for item in value]
        except Exception:
            return []
    if isinstance(value, dict):
        return {
            str(key).strip(): _normalize_signature_value(item) for key, item in value.items()
        }
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sorted_signature_list(values: Any) -> list[str]:
    return sorted({str(item).strip() for item in list(values) if str(item).strip()})


def _score_config_signature(path: str) -> str:
    token = str(path or "").strip()
    if not token:
        return "missing:empty"
    resolved = Path(token).resolve()
    try:
        return hashlib.sha256(resolved.read_bytes()).hexdigest()
    except OSError:
        return f"missing:{resolved}"


def _candidate_run_signature(
    *,
    candidate_library_hash: str,
    batch_timeframes: list[str],
    symbols: list[str],
    requested_timeframes: list[str],
    resolved_windows: dict[str, Any],
    score_config_path: str,
    chunk_days: int,
    window_profile: str,
    allow_metals: bool,
) -> str:
    payload = {
        "candidate_library_hash": str(candidate_library_hash),
        "batch_timeframes": _sorted_signature_list(batch_timeframes),
        "requested_timeframes": _sorted_signature_list(requested_timeframes),
        "symbols": _sorted_signature_list(symbols),
        "resolved_windows": {
            key: _normalize_signature_value(resolved_windows.get(key))
            for key in sorted({"train_start", "val_start", "oos_start", "requested_oos_end_exclusive"})
            if key in resolved_windows
        },
        "window_profile": str(window_profile or "default"),
        "score_config_signature": _score_config_signature(score_config_path),
        "chunk_days": int(chunk_days),
        "allow_metals": bool(allow_metals),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _registry_path(output_root: Path) -> Path:
    return output_root / "exact_window_run_registry.jsonl"


def _iter_signature_registry(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield entry


def _find_completed_signature_entry(path: Path, *, signature: str) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for entry in _iter_signature_registry(path):
        if str(entry.get("signature", "")) != signature:
            continue
        if str(entry.get("status", "")) != "completed":
            continue
        if latest is None:
            latest = dict(entry)
            continue
        timestamp = str(entry.get("completed_at_utc", ""))
        if timestamp > str(latest.get("completed_at_utc", "")):
            latest = dict(entry)
    return latest


def _append_signature_entry(
    path: Path,
    *,
    signature: str,
    run_id: str,
    status: str,
    batch_id: str,
    run_root: str,
    batch_dir: str,
    manifest_path: str,
    summary_path: str,
    details_path: str,
    fail_analysis_path: str | None = None,
    memory_evidence_path: str | None = None,
    error: str | None = None,
) -> None:
    record: dict[str, Any] = {
        "signature": signature,
        "run_id": run_id,
        "status": status,
        "batch_id": batch_id,
        "run_root": str(run_root),
        "batch_dir": str(batch_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "details_path": str(details_path),
        "fail_analysis_path": str(fail_analysis_path) if fail_analysis_path else None,
        "memory_evidence_path": str(memory_evidence_path) if memory_evidence_path else None,
        "error": error,
        "started_at_utc": _utc_now().isoformat(),
        "completed_at_utc": _utc_now().isoformat(),
        "request_path": str(path),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _build_resolved_windows(
    args: argparse.Namespace,
    symbols: list[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    resolved_train_start = str(args.train_start or "").strip() or None
    resolved_val_start = str(args.val_start or "").strip() or None
    resolved_oos_start = str(args.oos_start or "").strip() or None
    resolved_requested_oos_end = str(args.requested_oos_end or "").strip() or None
    profile_token = str(args.window_profile or "default").strip() or "default"
    adaptive = None
    if (
        profile_token != "default"
        and not resolved_train_start
        and not resolved_val_start
        and not resolved_oos_start
    ):
        adaptive = resolve_coverage_adaptive_windows(
            symbols=list(symbols or []),
            root_path=str(getattr(BaseConfig, "MARKET_DATA_PARQUET_PATH", "data/market_parquet")),
            exchange=str(getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance") or "binance"),
            requested_oos_end_exclusive=resolved_requested_oos_end,
            profile=profile_token,
            chunk_days=max(7, int(args.chunk_days)),
        )
        resolved_train_start = (
            adaptive["train_start"].isoformat()
            if hasattr(adaptive["train_start"], "isoformat")
            else str(adaptive["train_start"])
        )
        resolved_val_start = (
            adaptive["val_start"].isoformat()
            if hasattr(adaptive["val_start"], "isoformat")
            else str(adaptive["val_start"])
        )
        resolved_oos_start = (
            adaptive["oos_start"].isoformat()
            if hasattr(adaptive["oos_start"], "isoformat")
            else str(adaptive["oos_start"])
        )
        resolved_requested_oos_end = (
            adaptive["requested_oos_end_exclusive"].isoformat()
            if hasattr(adaptive["requested_oos_end_exclusive"], "isoformat")
            else str(adaptive["requested_oos_end_exclusive"])
        )
    resolved = {
        "train_start": resolved_train_start,
        "val_start": resolved_val_start,
        "oos_start": resolved_oos_start,
        "requested_oos_end_exclusive": resolved_requested_oos_end,
    }
    return resolved, adaptive


def _resolve_batch_timeframes(raw_timeframes: list[str]) -> list[str]:
    requested = list(raw_timeframes or list(CANONICAL_STRATEGY_TIMEFRAMES))
    normalized = list(
        normalize_strategy_timeframes(
            [tf for tf in requested if tf not in LOW_RAM_EXCLUDED_TIMEFRAMES],
            required=CANONICAL_STRATEGY_TIMEFRAMES,
            strict_subset=True,
        )
    )
    return normalized or ["all"]


def _path_or_default(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved if resolved.exists() else path)


def _registry_batch_payload(
    *,
    status: str,
    error: str | None,
    summary: dict[str, Any] | None,
    memory_bundle: dict[str, Any] | None,
    batch_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(batch_payload or {})
    payload.setdefault("train_start", "")
    payload.setdefault("val_start", "")
    payload.setdefault("oos_start", "")
    payload.setdefault("requested_oos_end_exclusive", "")
    payload.setdefault("window_profile", "default")
    payload.setdefault("chunk_days", 0)
    payload.setdefault("status", status)
    payload.setdefault("error", error)
    payload["evaluated_count"] = int(summary.get("evaluated_count") or 0) if isinstance(summary, dict) else 0
    payload["promoted_count"] = int(summary.get("promoted_count") or 0) if isinstance(summary, dict) else 0
    if isinstance(memory_bundle, dict):
        mem_payload = memory_bundle.get("payload")
        if isinstance(mem_payload, dict):
            payload["peak_rss_mib"] = float(mem_payload.get("peak_rss_mib") or 0.0)
        else:
            payload.setdefault("peak_rss_mib", 0.0)
    else:
        payload.setdefault("peak_rss_mib", 0.0)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact-window low-RAM validation suite."
    )
    parser.add_argument("--output-dir", default="var/reports/exact_window_backtests")
    parser.add_argument("--score-config", default="configs/score_config.example.json")
    parser.add_argument("--timeframes", nargs="+", default=[])
    parser.add_argument("--symbols", nargs="+", default=[])
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--train-start", default="")
    parser.add_argument("--val-start", default="")
    parser.add_argument("--oos-start", default="")
    parser.add_argument("--requested-oos-end", default="")
    parser.add_argument(
        "--window-profile",
        default="default",
        choices=[
            "default",
            "coverage_adaptive",
            "coverage_adaptive_4h",
            "coverage_adaptive_1d",
            "metals",
            "metals_4h",
            "metals_1d",
            "mixed_assets",
            "mixed_assets_4h",
            "mixed_assets_1d",
        ],
        help=(
            "Optional adaptive window profile. "
            "When selected and explicit train/val/oos windows are omitted, "
            "derive windows from overlapping symbol coverage."
        ),
    )
    parser.add_argument(
        "--allow-metals",
        action="store_true",
        help="Allow metals (XAU/XAG/XPT/XPD) in the exact-window symbol universe.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--adopt-run-dir",
        default="",
        help="Adopt/report an existing exact-window batch directory without starting a new heavy run.",
    )
    parser.add_argument(
        "--existing-pid",
        type=int,
        default=0,
        help="Optional PID for an already-running exact-window process when using --adopt-run-dir.",
    )
    parser.add_argument(
        "--existing-log-path",
        default="",
        help="Optional existing log/monitor path to record in adopted run artifacts.",
    )
    parser.add_argument(
        "--rss-log-path",
        default="",
        help="Optional JSONL log path for RSS samples (default: run-batch/exact_window_rss_latest.jsonl).",
    )
    parser.add_argument(
        "--soft-rss-bytes",
        type=int,
        default=None,
        help="Optional soft RSS limit in bytes. Defaults to 60%% of cgroup/memavailable budget.",
    )
    parser.add_argument(
        "--hard-rss-bytes",
        type=int,
        default=None,
        help="Optional hard RSS limit in bytes. Defaults to 80%% of cgroup/memavailable budget.",
    )
    parser.add_argument(
        "--skip-fail-analysis",
        action="store_true",
        help="Skip fail-analysis artifact generation after the suite run.",
    )
    parser.add_argument(
        "--emit-memory-baseline",
        action="store_true",
        help="Do not run the suite; capture a lightweight memory-baseline artifact only.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore signature cache and rerun even if windows and config are unchanged.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = str(args.run_id or "").strip() or f"exact_window_{_utc_stamp()}"
    requested_timeframes = list(args.timeframes or [])
    requested_symbols = list(args.symbols or [])
    batch_timeframes = _resolve_batch_timeframes(requested_timeframes)
    batch_id = "-".join(batch_timeframes)
    adopt_run_dir = (
        Path(str(args.adopt_run_dir or "").strip()).resolve()
        if str(args.adopt_run_dir or "").strip()
        else None
    )

    resolved_windows, adaptive_windows = _build_resolved_windows(
        args=args,
        symbols=requested_symbols,
    )

    candidate_hash = _candidate_library_hash()
    run_signature = _candidate_run_signature(
        candidate_library_hash=candidate_hash,
        batch_timeframes=batch_timeframes,
        symbols=requested_symbols,
        requested_timeframes=requested_timeframes,
        resolved_windows=resolved_windows,
        score_config_path=str(args.score_config or ""),
        chunk_days=max(1, int(args.chunk_days)),
        window_profile=str(args.window_profile or "default"),
        allow_metals=bool(args.allow_metals),
    )

    if (
        not bool(args.emit_memory_baseline)
        and adopt_run_dir is None
        and not bool(args.force_rerun)
    ):
        registry_entry = _find_completed_signature_entry(
            _registry_path(output_root),
            signature=run_signature,
        )
        if registry_entry is not None:
            reused_summary = Path(str(registry_entry.get("summary_path") or "")).resolve()
            reused_details = Path(str(registry_entry.get("details_path") or "")).resolve()
            reused_fail_analysis = Path(
                str(registry_entry.get("fail_analysis_path") or "")
            ).resolve()
            reused_memory = Path(
                str(registry_entry.get("memory_evidence_path") or "")
            ).resolve()
            run_root = output_root / run_id
            run_root.mkdir(parents=True, exist_ok=True)
            batch_dir = run_root / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)
            root_latest_path = output_root / "latest.json"
            run_latest_path = run_root / "latest.json"
            manifest_path = run_root / "manifest.json"
            progress_path = run_root / "progress.csv"

            latest_pointer = {
                "schema_version": "1.0",
                "run_id": run_id,
                "run_root": str(run_root),
                "run_dir": str(batch_dir),
                "batch_id": batch_id,
                "updated_at_utc": _utc_now().isoformat(),
                "status": "skipped_duplicate",
                "run_signature": run_signature,
                "heavy_lock_path": str(output_root / "exact_window_heavy_run.lock"),
                "summary_path": _path_or_default(reused_summary),
                "details_path": _path_or_default(reused_details),
                "fail_analysis_path": _path_or_default(reused_fail_analysis),
                "memory_evidence_path": _path_or_default(reused_memory),
            }
            manifest = {
                "schema_version": "1.0",
                "status": "skipped_duplicate",
                "run_id": run_id,
                "batch_id": batch_id,
                "reused_run_id": str(registry_entry.get("run_id") or ""),
                "started_at_utc": _utc_now().isoformat(),
                "completed_at_utc": _utc_now().isoformat(),
                "code_commit_sha": _git_commit_marker()[0],
                "git_dirty": _git_commit_marker()[1],
                "candidate_library_hash": candidate_hash,
                "run_signature": run_signature,
                "requested_timeframes": requested_timeframes,
                "requested_symbols": requested_symbols,
                "allow_metals": bool(args.allow_metals),
                "chunk_days": max(1, int(args.chunk_days)),
                "window_profile": str(args.window_profile or "default"),
                "windows": resolved_windows,
                "train_start": resolved_windows.get("train_start"),
                "val_start": resolved_windows.get("val_start"),
                "oos_start": resolved_windows.get("oos_start"),
                "requested_oos_end_exclusive": resolved_windows.get("requested_oos_end_exclusive"),
                "run_root": str(run_root),
                "batch_dir": str(batch_dir),
                "artifacts": {
                    "manifest_path": str(manifest_path),
                    "progress_path": str(progress_path),
                    "run_latest_path": str(run_latest_path),
                    "root_latest_path": str(root_latest_path),
                    "summary_path": _path_or_default(reused_summary),
                    "details_path": _path_or_default(reused_details),
                    "fail_analysis_path": _path_or_default(reused_fail_analysis),
                    "memory_evidence_path": _path_or_default(reused_memory),
                },
            }
            if adaptive_windows is not None:
                manifest["adaptive_windows"] = {
                    key: value if not hasattr(value, "isoformat") else value.isoformat()
                    for key, value in adaptive_windows.items()
                    if key != "coverage_rows"
                }
            _write_json(manifest_path, manifest)
            _write_json(run_latest_path, latest_pointer)
            _write_json(root_latest_path, latest_pointer)
            _append_progress_row(
                progress_path,
                {
                    "timestamp_utc": _utc_now().isoformat(),
                    "batch_id": batch_id,
                    "status": "skipped_duplicate",
                    "timeframes": ",".join(batch_timeframes),
                    "evaluated_count": 0,
                    "promoted_count": 0,
                    "summary_path": _path_or_default(reused_summary),
                    "details_path": _path_or_default(reused_details),
                    "rss_log_path": "",
                    "fail_analysis_path": _path_or_default(reused_fail_analysis),
                    "memory_evidence_path": _path_or_default(reused_memory),
                    "notes": "skipped duplicate run signature",
                },
            )
            summary_path = reused_summary
            details_path = reused_details
            upsert_backtest_registry(
                output_root,
                run_id=run_id,
                batch_id=batch_id,
                status="skipped_duplicate",
                run_signature=run_signature,
                manifest_path=str(manifest_path),
                summary_path=str(summary_path),
                details_path=str(details_path),
                fail_analysis_path=str(reused_fail_analysis),
                memory_evidence_path=str(reused_memory),
                requested_timeframes=requested_timeframes,
                requested_symbols=requested_symbols,
                allow_metals=bool(args.allow_metals),
                batch_payload=_registry_batch_payload(
                    status="skipped_duplicate",
                    error=None,
                    summary=None,
                    memory_bundle=None,
                    batch_payload={
                        "train_start": resolved_windows.get("train_start"),
                        "val_start": resolved_windows.get("val_start"),
                        "oos_start": resolved_windows.get("oos_start"),
                        "requested_oos_end_exclusive": resolved_windows.get("requested_oos_end_exclusive"),
                        "window_profile": str(args.window_profile or "default"),
                        "chunk_days": max(1, int(args.chunk_days)),
                    },
                ),
            )
            print(
                json.dumps(
                    {
                        "status": "skipped_duplicate",
                        "run_id": run_id,
                        "manifest_path": str(manifest_path),
                        "progress_path": str(progress_path),
                        "latest_path": str(root_latest_path),
                        "run_latest_path": str(run_latest_path),
                        "summary_latest": _path_or_default(reused_summary),
                        "details_latest": _path_or_default(reused_details),
                        "fail_analysis_latest": _path_or_default(reused_fail_analysis),
                        "rss_log_latest": "",
                        "memory_evidence_latest": _path_or_default(reused_memory),
                        "run_signature": run_signature,
                        "reused_run_id": str(registry_entry.get("run_id") or ""),
                    },
                    indent=2,
                )
            )
            return 0

    heavy_lock_path = output_root / "exact_window_heavy_run.lock"
    run_lock: HeavyRunLock | None = None
    if not bool(args.emit_memory_baseline) and adopt_run_dir is None:
        try:
            run_lock = HeavyRunLock.acquire(
                lock_path=heavy_lock_path,
                label="exact_window",
                metadata={
                    "run_id": run_id,
                    "batch_id": batch_id,
                    "requested_timeframes": requested_timeframes,
                    "requested_symbols": requested_symbols,
                    "allow_metals": bool(args.allow_metals),
                    "run_signature": run_signature,
                },
            )
        except HeavyRunActiveError as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked_active_run",
                        "run_id": run_id,
                        "batch_id": batch_id,
                        "lock_path": str(exc.lock_path),
                        "active_run": dict(exc.active_payload),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 3

    run_root = output_root / run_id
    batch_dir = adopt_run_dir if adopt_run_dir is not None else (run_root / batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)

    root_latest_path = output_root / "latest.json"
    run_latest_path = run_root / "latest.json"
    manifest_path = run_root / "manifest.json"
    progress_path = run_root / "progress.csv"
    rss_log_path = (
        Path(args.rss_log_path).resolve()
        if args.rss_log_path
        else Path(args.existing_log_path).resolve()
        if args.existing_log_path
        else batch_dir / RSS_LOG_LATEST
    )

    commit_sha, git_dirty = _git_commit_marker()
    try:
        manifest = {
            "schema_version": "1.0",
            "status": "running",
            "run_id": run_id,
            "batch_id": batch_id,
            "started_at_utc": _utc_now().isoformat(),
            "code_commit_sha": commit_sha,
            "git_dirty": git_dirty,
            "candidate_library_hash": candidate_hash,
            "run_signature": run_signature,
            "requested_timeframes": requested_timeframes,
            "requested_symbols": requested_symbols,
            "allow_metals": bool(args.allow_metals),
            "chunk_days": max(1, int(args.chunk_days)),
            "max_parquet_workers": 1,
            "clamp_ts": "2026-03-07T10:00:00Z",
                "window_profile": str(args.window_profile or "default"),
                "windows": resolved_windows,
                "train_start": resolved_windows.get("train_start"),
                "val_start": resolved_windows.get("val_start"),
                "oos_start": resolved_windows.get("oos_start"),
                "requested_oos_end_exclusive": resolved_windows.get("requested_oos_end_exclusive"),
                "run_root": str(run_root),
                "batch_dir": str(batch_dir),
                "artifacts": {
                    "manifest_path": str(manifest_path),
                    "progress_path": str(progress_path),
                "run_latest_path": str(run_latest_path),
                "root_latest_path": str(root_latest_path),
                "rss_log_path": str(rss_log_path),
                "heavy_lock_path": str(heavy_lock_path),
            },
        }
        latest_pointer = {
            "schema_version": "1.0",
            "run_id": run_id,
            "run_root": str(run_root),
            "run_dir": str(batch_dir),
            "batch_id": batch_id,
            "updated_at_utc": _utc_now().isoformat(),
            "status": "running",
            "heavy_lock_path": str(heavy_lock_path),
        }
        _write_json(manifest_path, manifest)
        _write_json(run_latest_path, latest_pointer)
        _write_json(root_latest_path, latest_pointer)

        if adopt_run_dir is not None:
            summary_path = batch_dir / SUMMARY_LATEST
            details_path = batch_dir / DETAILS_LATEST
            fail_analysis_path = batch_dir / "exact_window_fail_analysis_latest.json"
            memory_evidence_path = batch_dir / MEMORY_EVIDENCE_LATEST
            existing_pid = int(args.existing_pid or 0)
            pid_running = existing_pid > 0 and Path(f"/proc/{existing_pid}").exists()
            status = (
                "running"
                if pid_running
                else "completed"
                if summary_path.exists()
                else "pending"
            )
            fail_bundle = (
                write_fail_analysis_bundle(output_dir=output_root)
                if summary_path.exists() and details_path.exists() and not args.skip_fail_analysis
                else None
            )
            memory_bundle = write_memory_evidence_bundle(
                output_dir=output_root,
                memory_summary={
                    "generated_at": _utc_now().isoformat(),
                    "status": status,
                    "rss_log_path": _path_or_default(rss_log_path),
                    "existing_pid": existing_pid or None,
                },
                summary=json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None,
            )
            if fail_bundle is not None:
                fail_analysis_path = Path(fail_bundle["json_latest"])
            if memory_bundle is not None:
                memory_evidence_path = Path(memory_bundle["json_latest"])
            stable_paths = sync_exact_window_latest_aliases(output_root)
            summary_path = Path(stable_paths.get("summary") or summary_path)
            details_path = Path(stable_paths.get("details") or details_path)
            rss_log_path = Path(stable_paths.get("rss_log") or rss_log_path)
            fail_analysis_path = Path(stable_paths.get("fail_analysis") or fail_analysis_path)
            memory_evidence_path = Path(stable_paths.get("memory_evidence") or memory_evidence_path)
            adopted_summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else None
            )
            latest_pointer.update(
                {
                    "updated_at_utc": _utc_now().isoformat(),
                    "status": status,
                    "run_dir": str(batch_dir),
                    "summary_path": _path_or_default(summary_path),
                    "details_path": _path_or_default(details_path),
                    "rss_log_path": _path_or_default(rss_log_path),
                    "fail_analysis_path": _path_or_default(fail_analysis_path),
                    "memory_evidence_path": _path_or_default(memory_evidence_path),
                    "existing_pid": existing_pid or None,
                }
            )
            manifest.update(
                {
                    "status": status,
                    "completed_at_utc": _utc_now().isoformat(),
                    "adopted_existing_run": True,
                    "existing_pid": existing_pid or None,
                    "existing_log_path": _path_or_default(rss_log_path),
                    "artifacts": {
                        **dict(manifest.get("artifacts") or {}),
                        "summary_path": _path_or_default(summary_path),
                        "details_path": _path_or_default(details_path),
                        "fail_analysis_path": _path_or_default(fail_analysis_path),
                        "memory_evidence_path": _path_or_default(memory_evidence_path),
                    },
                }
            )
            _write_json(run_latest_path, latest_pointer)
            _write_json(root_latest_path, latest_pointer)
            _write_json(manifest_path, manifest)
            _append_progress_row(
                progress_path,
                {
                    "timestamp_utc": _utc_now().isoformat(),
                    "batch_id": batch_id,
                    "status": status,
                    "timeframes": ",".join(batch_timeframes),
                    "evaluated_count": 0,
                    "promoted_count": 0,
                    "summary_path": _path_or_default(summary_path),
                    "details_path": _path_or_default(details_path),
                    "rss_log_path": _path_or_default(rss_log_path),
                    "fail_analysis_path": _path_or_default(fail_analysis_path),
                    "memory_evidence_path": _path_or_default(memory_evidence_path),
                    "notes": f"adopted_existing_run pid={existing_pid or 'n/a'}",
                },
            )
            upsert_backtest_registry(
                output_root,
                run_id=run_id,
                batch_id=batch_id,
                status=status,
                run_signature=run_signature,
                manifest_path=str(manifest_path),
                summary_path=str(summary_path),
                details_path=str(details_path),
                fail_analysis_path=str(fail_analysis_path),
                memory_evidence_path=str(memory_evidence_path),
                requested_timeframes=requested_timeframes,
                requested_symbols=requested_symbols,
                allow_metals=bool(args.allow_metals),
                batch_payload=_registry_batch_payload(
                    status=status,
                    error=None,
                    summary=adopted_summary,
                    memory_bundle=memory_bundle,
                    batch_payload={
                        "train_start": resolved_windows.get("train_start"),
                        "val_start": resolved_windows.get("val_start"),
                        "oos_start": resolved_windows.get("oos_start"),
                        "requested_oos_end_exclusive": resolved_windows.get("requested_oos_end_exclusive"),
                        "window_profile": str(args.window_profile or "default"),
                        "chunk_days": max(1, int(args.chunk_days)),
                    },
                ),
            )
            print(
                json.dumps(
                    {
                        "status": status,
                        "run_id": run_id,
                        "manifest_path": str(manifest_path),
                        "progress_path": str(progress_path),
                        "latest_path": str(root_latest_path),
                        "run_latest_path": str(run_latest_path),
                        "summary_latest": _path_or_default(summary_path),
                        "details_latest": _path_or_default(details_path),
                        "fail_analysis_latest": _path_or_default(fail_analysis_path),
                        "rss_log_latest": _path_or_default(rss_log_path),
                        "memory_evidence_latest": _path_or_default(memory_evidence_path),
                        "existing_pid": existing_pid or None,
                    },
                    indent=2,
                )
            )
            return 0

        guard = RSSGuard(
            log_path=rss_log_path,
            soft_limit_bytes=args.soft_rss_bytes,
            hard_limit_bytes=args.hard_rss_bytes,
        )
        guard.sample(
            event="cli_start",
            context={
                "mode": "baseline_probe" if bool(args.emit_memory_baseline) else "suite_run",
                "timeframes": requested_timeframes,
                "symbols": requested_symbols,
                "allow_metals": bool(args.allow_metals),
                "run_id": run_id,
                "batch_id": batch_id,
            },
        )

        summary: dict[str, Any] | None = None
        fail_bundle: dict[str, Any] | None = None
        memory_bundle: dict[str, Any] | None = None
        status = "baseline_probe" if bool(args.emit_memory_baseline) else "completed"
        error: str | None = None
        rc = 0

        try:
            if not args.emit_memory_baseline:
                score_config = _load_score_config(args.score_config)
                summary = run_exact_window_suite(
                    output_dir=str(batch_dir),
                    score_config=score_config,
                    timeframes=requested_timeframes,
                    symbols=requested_symbols,
                    chunk_days=max(1, int(args.chunk_days)),
                    allow_metals=bool(args.allow_metals),
                    train_start=resolved_windows.get("train_start"),
                    val_start=resolved_windows.get("val_start"),
                    oos_start=resolved_windows.get("oos_start"),
                    requested_oos_end_exclusive=resolved_windows.get("requested_oos_end_exclusive"),
                    progress_callback=guard.checkpoint,
                )
                if adaptive_windows is not None:
                    manifest["adaptive_windows"] = {
                        key: value if not hasattr(value, "isoformat") else value.isoformat()
                        for key, value in adaptive_windows.items()
                        if key != "coverage_rows"
                    }
                if not args.skip_fail_analysis:
                    fail_bundle = write_fail_analysis_bundle(output_dir=output_root, summary=summary)
        except RSSLimitExceeded as exc:
            status = "aborted_rss_guard"
            error = str(exc)
            rc = 2
        except Exception as exc:  # pragma: no cover
            status = "failed"
            error = str(exc)
            rc = 1
        finally:
            if not args.emit_memory_baseline:
                guard.sample(event="cli_finish", context={"status": status, "run_id": run_id})
                memory_bundle = write_memory_evidence_bundle(
                    output_dir=output_root,
                    memory_summary=guard.finalize(status=status, error=error),
                    summary=summary,
                )

        stable_paths = sync_exact_window_latest_aliases(output_root)
        summary_path = Path(stable_paths.get("summary") or (batch_dir / SUMMARY_LATEST))
        details_path = Path(stable_paths.get("details") or (batch_dir / DETAILS_LATEST))
        rss_path = Path(stable_paths.get("rss_log") or rss_log_path)
        fail_analysis_path = (
            Path(fail_bundle["json_latest"])
            if fail_bundle
            else Path(
                stable_paths.get("fail_analysis")
                or (batch_dir / "exact_window_fail_analysis_latest.json")
            )
        )
        memory_evidence_path = (
            Path(memory_bundle["json_latest"])
            if memory_bundle
            else Path(stable_paths.get("memory_evidence") or (batch_dir / MEMORY_EVIDENCE_LATEST))
        )
        if fail_bundle:
            fail_analysis_path = Path(stable_paths.get("fail_analysis") or fail_analysis_path)
        if memory_bundle:
            memory_evidence_path = Path(stable_paths.get("memory_evidence") or memory_evidence_path)

        latest_pointer.update(
            {
                "updated_at_utc": _utc_now().isoformat(),
                "status": status,
                "summary_path": str(summary_path),
                "details_path": str(details_path),
                "rss_log_path": str(rss_path),
                "fail_analysis_path": str(fail_analysis_path),
                "memory_evidence_path": str(memory_evidence_path),
                "promoted_count": int(summary.get("promoted_count") or 0)
                if isinstance(summary, dict)
                else 0,
            }
        )
        manifest.update(
            {
                "status": status,
                "completed_at_utc": _utc_now().isoformat(),
                "train_start": resolved_windows.get("train_start"),
                "val_start": resolved_windows.get("val_start"),
                "oos_start": resolved_windows.get("oos_start"),
                "requested_oos_end_exclusive": resolved_windows.get("requested_oos_end_exclusive"),
                "error": error,
                "windows": dict(summary.get("windows") or {}) if isinstance(summary, dict) else resolved_windows,
                "execution_profile": {
                    **(
                        dict(summary.get("execution_profile") or {})
                        if isinstance(summary, dict)
                        else {}
                    ),
                    "train_start": resolved_windows.get("train_start"),
                    "val_start": resolved_windows.get("val_start"),
                    "oos_start": resolved_windows.get("oos_start"),
                    "requested_oos_end_exclusive": resolved_windows.get("requested_oos_end_exclusive"),
                    "chunk_days": max(1, int(args.chunk_days)),
                    "requested_timeframes": requested_timeframes,
                    "requested_symbols": requested_symbols,
                    "allow_metals": bool(args.allow_metals),
                    "window_profile": str(args.window_profile or "default"),
                },
                "evaluated_count": int(summary.get("evaluated_count") or 0)
                if isinstance(summary, dict)
                else 0,
                "promoted_count": int(summary.get("promoted_count") or 0)
                if isinstance(summary, dict)
                else 0,
                "artifacts": {
                    **dict(manifest.get("artifacts") or {}),
                    "summary_path": str(summary_path),
                    "details_path": str(details_path),
                    "fail_analysis_path": str(fail_analysis_path),
                    "memory_evidence_path": str(memory_evidence_path),
                },
            }
        )
        _write_json(run_latest_path, latest_pointer)
        _write_json(root_latest_path, latest_pointer)
        _write_json(manifest_path, manifest)
        _append_progress_row(
            progress_path,
            {
                "timestamp_utc": _utc_now().isoformat(),
                "batch_id": batch_id,
                "status": status,
                "timeframes": ",".join(batch_timeframes),
                "evaluated_count": int(summary.get("evaluated_count") or 0)
                if isinstance(summary, dict)
                else 0,
                "promoted_count": int(summary.get("promoted_count") or 0)
                if isinstance(summary, dict)
                else 0,
                "summary_path": str(summary_path),
                "details_path": str(details_path),
                "rss_log_path": str(rss_path),
                "fail_analysis_path": str(fail_analysis_path),
                "memory_evidence_path": str(memory_evidence_path),
                "notes": error or "",
            },
        )
        upsert_backtest_registry(
            output_root,
            run_id=run_id,
            batch_id=batch_id,
            status=status,
            run_signature=run_signature,
            manifest_path=str(manifest_path),
            summary_path=str(summary_path),
            details_path=str(details_path),
            fail_analysis_path=str(fail_analysis_path),
            memory_evidence_path=str(memory_evidence_path),
            requested_timeframes=requested_timeframes,
            requested_symbols=requested_symbols,
            allow_metals=bool(args.allow_metals),
            batch_payload=_registry_batch_payload(
                status=status,
                error=error,
                summary=summary,
                memory_bundle=memory_bundle,
                batch_payload={
                    "train_start": resolved_windows.get("train_start"),
                    "val_start": resolved_windows.get("val_start"),
                    "oos_start": resolved_windows.get("oos_start"),
                    "requested_oos_end_exclusive": resolved_windows.get("requested_oos_end_exclusive"),
                    "window_profile": str(args.window_profile or "default"),
                    "chunk_days": max(1, int(args.chunk_days)),
                },
            ),
        )

        _append_signature_entry(
            _registry_path(output_root),
            signature=run_signature,
            run_id=run_id,
            status=status,
            batch_id=batch_id,
            run_root=str(run_root),
            batch_dir=str(batch_dir),
            manifest_path=str(manifest_path),
            summary_path=str(summary_path),
            details_path=str(details_path),
            fail_analysis_path=str(fail_analysis_path),
            memory_evidence_path=str(memory_evidence_path),
            error=error,
        )

        payload = {
            "status": status,
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "progress_path": str(progress_path),
            "latest_path": str(root_latest_path),
            "run_latest_path": str(run_latest_path),
            "summary_latest": str(summary_path),
            "details_latest": str(details_path),
            "fail_analysis_latest": str(fail_analysis_path),
            "rss_log_latest": str(rss_path),
            "memory_evidence_latest": str(memory_evidence_path),
            "heavy_lock_path": str(heavy_lock_path),
            "allow_metals": bool(args.allow_metals),
            "run_signature": run_signature,
            "eligible_symbols": summary.get("eligible_symbols") if isinstance(summary, dict) else [],
            "best_strategy_count": len(summary.get("best_per_strategy") or [])
            if isinstance(summary, dict)
            else 0,
            "promoted_count": int(summary.get("promoted_count") or 0)
            if isinstance(summary, dict)
            else 0,
            "portfolio_weight_count": (
                len((summary.get("portfolio") or {}).get("weights") or [])
                if isinstance(summary, dict)
                else 0
            ),
        }
        if error:
            payload["error"] = error
        stream = sys.stdout if rc == 0 else sys.stderr
        print(json.dumps(payload, indent=2), file=stream)
        return int(rc)
    finally:
        if run_lock is not None:
            run_lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
