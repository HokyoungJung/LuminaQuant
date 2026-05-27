from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _autoresearch_root(repo_root: Path) -> Path:
    return repo_root / ".omx" / "logs" / "autoresearch"


def _resolve_run_id(*, repo_root: Path, run_id: str, latest: bool) -> str:
    normalized = str(run_id or "").strip()
    if normalized:
        return normalized
    if not latest:
        raise ValueError("provide --run-id or pass --latest")

    runs_root = _autoresearch_root(repo_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"autoresearch log root not found: {runs_root}")

    def _candidate_key(path: Path) -> tuple[str, int, str]:
        manifest_path = path / "manifest.json"
        updated_at = ""
        if manifest_path.exists():
            try:
                payload = _load_json(manifest_path)
            except Exception:
                payload = {}
            updated_at = str(
                payload.get("updated_at")
                or payload.get("completed_at")
                or payload.get("created_at")
                or ""
            )
        return (updated_at, path.stat().st_mtime_ns, path.name)

    candidates = sorted((path for path in runs_root.iterdir() if path.is_dir()), key=_candidate_key)
    if candidates:
        return candidates[-1].name
    raise FileNotFoundError(f"no autoresearch runs found under {runs_root}")


def build_summary(*, repo_root: Path, run_id: str, tail_entries: int) -> dict[str, Any]:
    run_dir = _autoresearch_root(repo_root) / run_id
    manifest_path = run_dir / "manifest.json"
    ledger_path = run_dir / "iteration-ledger.json"
    evaluator_path = run_dir / "latest-evaluator-result.json"
    candidate_path = run_dir / "candidate.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found for run {run_id}: {manifest_path}")

    manifest = _load_json(manifest_path)
    ledger = _load_json(ledger_path) if ledger_path.exists() else {"entries": []}
    evaluator = _load_json(evaluator_path) if evaluator_path.exists() else None
    candidate = _load_json(candidate_path) if candidate_path.exists() else None
    entries = list(ledger.get("entries") or [])
    recent_entries = entries[-max(0, int(tail_entries)) :] if tail_entries else []

    return {
        "run": {
            "run_id": manifest.get("run_id", run_id),
            "status": manifest.get("status"),
            "iteration": manifest.get("iteration"),
            "keep_policy": manifest.get("keep_policy"),
            "baseline_commit": manifest.get("baseline_commit"),
            "last_kept_commit": manifest.get("last_kept_commit"),
            "last_kept_score": manifest.get("last_kept_score"),
            "worktree_path": manifest.get("worktree_path"),
            "results_file": manifest.get("results_file"),
            "mission_dir": manifest.get("mission_dir"),
            "created_at": manifest.get("created_at"),
            "updated_at": manifest.get("updated_at"),
            "completed_at": manifest.get("completed_at"),
        },
        "paths": {
            "run_dir": str(run_dir),
            "manifest": str(manifest_path),
            "ledger": str(ledger_path),
            "latest_evaluator": str(evaluator_path),
            "candidate": str(candidate_path),
        },
        "latest_evaluator": evaluator,
        "candidate": candidate,
        "recent_ledger_entries": recent_entries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect_autoresearch_run.py",
        description="Summarize the latest or selected OMX autoresearch run.",
    )
    parser.add_argument("--run-id", default="", help="Specific autoresearch run id to inspect.")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Inspect the most recently updated autoresearch run under .omx/logs/autoresearch/.",
    )
    parser.add_argument(
        "--tail-entries",
        type=int,
        default=5,
        help="How many recent ledger entries to include in the summary.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(_default_repo_root()),
        help="Repository root containing .omx/logs/autoresearch/ (defaults to this repository).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    run_id = _resolve_run_id(repo_root=repo_root, run_id=args.run_id, latest=bool(args.latest))
    summary = build_summary(
        repo_root=repo_root, run_id=run_id, tail_entries=max(0, int(args.tail_entries))
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
