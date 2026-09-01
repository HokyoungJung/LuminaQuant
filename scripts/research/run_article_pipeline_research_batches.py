"""Run article-pipeline candidate research sequentially in OOM-conscious batches."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "var/reports/exact_window_backtests/followup_status/"
    "portfolio_incumbent_autoresearch_grouped/article_inspired_research_current"
)
DEFAULT_BATCHES_PATH = DEFAULT_ROOT / "article_pipeline_research_batches_latest.json"
DEFAULT_RUNS_DIR = DEFAULT_ROOT / "batch_runs"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-path", type=Path, default=DEFAULT_BATCHES_PATH)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--batch-ids",
        default="",
        help="Optional comma-separated batch ids to run (e.g. batch_01,batch_02).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional cap on number of batches to run in this invocation (0 = all).",
    )
    parser.add_argument(
        "--stop-after-errors",
        type=int,
        default=1,
        help="Stop after this many batch failures.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _selected_batches(
    payload: dict[str, Any], raw_batch_ids: str, max_batches: int
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in list(payload.get("batches") or []) if isinstance(row, dict)]
    wanted = {token.strip() for token in str(raw_batch_ids or "").split(",") if token.strip()}
    if wanted:
        rows = [row for row in rows if str(row.get("batch_id")) in wanted]
    if max_batches > 0:
        rows = rows[: max(1, int(max_batches))]
    return rows


def _write_batch_manifest(*, run_dir: Path, source_manifest: Path, batch: dict[str, Any]) -> Path:
    source_payload = _load_payload(source_manifest)
    wanted_ids = {str(token) for token in list(batch.get("candidate_ids") or [])}
    candidates = [
        dict(row)
        for row in list(source_payload.get("candidates") or [])
        if isinstance(row, dict) and str(row.get("candidate_id")) in wanted_ids
    ]
    manifest = {
        "artifact_kind": "article_pipeline_batch_manifest",
        "generated_at": _utc_now_iso(),
        "source_manifest": str(source_manifest.resolve()),
        "batch": dict(batch),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    manifest_path = run_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _extract_max_rss_kb(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Maximum resident set size (kbytes):" in line:
            try:
                return int(line.rsplit(":", 1)[-1].strip())
            except ValueError:
                return None
    return None


def _run_batch(*, repo_root: Path, manifest_path: Path, run_dir: Path) -> int:
    batch_manifest = _load_payload(manifest_path)
    candidates = [
        dict(row) for row in list(batch_manifest.get("candidates") or []) if isinstance(row, dict)
    ]
    timeframes = sorted(
        {
            str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip().lower()
            for row in candidates
            if str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip()
        }
    )
    symbols = sorted(
        {
            str(symbol).strip()
            for row in candidates
            for symbol in list(row.get("symbols") or [])
            if str(symbol).strip()
        }
    )
    log_path = run_dir / "batch.log"
    timeframes_arg = " ".join(timeframes)
    symbols_arg = " ".join(symbols)
    cmd = (
        "set -euo pipefail; "
        f"/usr/bin/time -v uv run python scripts/run_research_candidates.py "
        f"--manifest {manifest_path.as_posix()} "
        f"--timeframes {timeframes_arg} "
        f"--symbols {symbols_arg} "
        f"--max-candidates {max(1, len(candidates))} "
        f"--top-k {max(1, min(len(candidates), 8))} "
        f"--output-dir {run_dir.as_posix()} "
        f"> {log_path.as_posix()} 2>&1"
    )
    completed = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        cwd=repo_root,
        check=False,
    )
    return int(completed.returncode)


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    batches_path = Path(args.batches_path).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    batches_payload = _load_payload(batches_path)
    source_manifest = Path(str(batches_payload.get("source_manifest") or "")).resolve()
    batches = _selected_batches(
        batches_payload,
        raw_batch_ids=str(args.batch_ids or ""),
        max_batches=max(0, int(args.max_batches)),
    )

    summary_rows: list[dict[str, Any]] = []
    error_count = 0
    for batch in batches:
        batch_id = str(batch.get("batch_id") or "batch_unknown")
        run_dir = runs_dir / batch_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "strategy_factory_report_latest.json"
        if result_path.exists():
            max_rss_kb = _extract_max_rss_kb(run_dir / "batch.log")
            print(f"[{_utc_now_iso()}] skip existing {batch_id}", flush=True)
            summary_rows.append(
                {
                    "batch_id": batch_id,
                    "status": "skipped_existing",
                    "candidate_count": int(batch.get("candidate_count") or 0),
                    "run_dir": str(run_dir),
                    "max_rss_kb": max_rss_kb,
                }
            )
            continue

        manifest_path = _write_batch_manifest(
            run_dir=run_dir,
            source_manifest=source_manifest,
            batch=batch,
        )
        print(
            f"[{_utc_now_iso()}] start {batch_id} "
            f"strategy={batch.get('strategy_class')} tf={batch.get('timeframe')} "
            f"candidates={batch.get('candidate_count')}",
            flush=True,
        )
        if args.dry_run:
            summary_rows.append(
                {
                    "batch_id": batch_id,
                    "status": "dry_run",
                    "candidate_count": int(batch.get("candidate_count") or 0),
                    "run_dir": str(run_dir),
                    "manifest_path": str(manifest_path),
                }
            )
            continue

        return_code = _run_batch(repo_root=repo_root, manifest_path=manifest_path, run_dir=run_dir)
        max_rss_kb = _extract_max_rss_kb(run_dir / "batch.log")
        status = "ok" if return_code == 0 else "error"
        print(
            f"[{_utc_now_iso()}] done {batch_id} status={status} max_rss_kb={max_rss_kb}",
            flush=True,
        )
        summary_rows.append(
            {
                "batch_id": batch_id,
                "status": status,
                "candidate_count": int(batch.get("candidate_count") or 0),
                "run_dir": str(run_dir),
                "manifest_path": str(manifest_path),
                "max_rss_kb": max_rss_kb,
                "return_code": int(return_code),
            }
        )
        if return_code != 0:
            error_count += 1
            if error_count >= max(1, int(args.stop_after_errors)):
                break

    summary = {
        "artifact_kind": "article_pipeline_batch_run_summary",
        "generated_at": _utc_now_iso(),
        "batches_path": str(batches_path),
        "runs_dir": str(runs_dir),
        "requested_batch_count": len(batches),
        "completed_batch_count": len(summary_rows),
        "error_count": error_count,
        "rows": summary_rows,
    }
    summary_path = runs_dir / "run_summary_latest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
