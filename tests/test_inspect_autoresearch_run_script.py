from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ops" / "inspect_autoresearch_run.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_inspect_autoresearch_run_latest_summary(tmp_path: Path):
    runs_root = tmp_path / ".omx" / "logs" / "autoresearch"
    older = runs_root / "run-older"
    newer = runs_root / "run-newer"

    _write_json(
        older / "manifest.json",
        {
            "run_id": "run-older",
            "status": "running",
            "iteration": 1,
            "keep_policy": "score_improvement",
            "updated_at": "2026-03-18T11:00:00Z",
        },
    )
    _write_json(
        newer / "manifest.json",
        {
            "run_id": "run-newer",
            "status": "running",
            "iteration": 3,
            "keep_policy": "score_improvement",
            "last_kept_score": 1.23,
            "worktree_path": "/tmp/worktree",
            "results_file": "/tmp/worktree/results.tsv",
            "updated_at": "2026-03-18T12:00:00Z",
        },
    )
    _write_json(
        newer / "latest-evaluator-result.json", {"pass": True, "score": 1.4, "status": "pass"}
    )
    _write_json(
        newer / "candidate.json",
        {
            "status": "candidate",
            "candidate_commit": "abc123",
            "base_commit": "def456",
            "description": "candidate summary",
            "notes": ["note one"],
            "created_at": "2026-03-18T00:00:00Z",
        },
    )
    _write_json(
        newer / "iteration-ledger.json",
        {
            "entries": [
                {"iteration": 1, "decision": "baseline"},
                {"iteration": 2, "decision": "discard"},
                {"iteration": 3, "decision": "keep"},
            ]
        },
    )

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(tmp_path),
        "--latest",
        "--tail-entries",
        "2",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["run"]["run_id"] == "run-newer"
    assert payload["run"]["last_kept_score"] == 1.23
    assert payload["latest_evaluator"]["score"] == 1.4
    assert payload["candidate"]["candidate_commit"] == "abc123"
    assert [entry["decision"] for entry in payload["recent_ledger_entries"]] == ["discard", "keep"]


def test_inspect_autoresearch_run_requires_latest_or_run_id(tmp_path: Path):
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert "provide --run-id or pass --latest" in result.stderr
