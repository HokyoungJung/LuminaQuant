#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TEAM_NAME = "luminaquant-autonomous-researc"
MAILBOX = REPO_ROOT / ".omx/state/team" / TEAM_NAME / "mailbox/leader-fixed.json"
CURRENT_OPT = (
    REPO_ROOT
    / "var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json"
)
LATEST_RUN = REPO_ROOT / "var/reports/exact_window_backtests/latest.json"

RELAUNCH_CMD = """cd /home/hoky/Quants-agent/LuminaQuant
./scripts/dev/relaunch_autonomous_team.sh"""


def run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return proc.stderr.strip() or proc.stdout.strip()
    return proc.stdout.strip()


def load_json(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def print_json(label: str, obj: dict | None, limit: int = 4000) -> None:
    print(f"\n## {label}")
    if obj is None:
        print("missing")
        return
    print(json.dumps(obj, indent=2)[:limit])


def print_mailbox_tail() -> None:
    print("\n## Mailbox tail")
    data = load_json(MAILBOX)
    if data is None:
        print(f"missing: {MAILBOX}")
        return
    for msg in data.get("messages", [])[-3:]:
        print("---")
        print(msg.get("created_at"))
        body = (msg.get("body") or "").strip()
        print(body[:1500])


def print_rss_tail(run_obj: dict | None) -> None:
    print("\n## Latest run RSS tail")
    if not run_obj:
        print("missing latest run metadata")
        return
    run_root = run_obj.get("run_root")
    candidates = []
    if run_root:
        root = pathlib.Path(run_root)
        candidates.extend(root.glob("**/exact_window_rss_latest.jsonl"))
    rss_path = run_obj.get("rss_log_path")
    if rss_path:
        candidates.append(pathlib.Path(rss_path))
    seen = set()
    for rss_file in candidates:
        if rss_file in seen:
            continue
        seen.add(rss_file)
        if not rss_file.exists():
            continue
        lines = rss_file.read_text().strip().splitlines()
        print(f"source: {rss_file}")
        for raw in lines[-5:]:
            print(raw)
        return
    print("no rss log found for latest run")


def main() -> int:
    print(f"Repo: {REPO_ROOT}")
    print(f"Branch: {run(['git', 'branch', '--show-current'])}")
    print("\n## Team status")
    team_status = run(["omx", "team", "status", TEAM_NAME, "--json"])
    print(team_status)
    if "No such team" in team_status or "No resumable team" in team_status:
        print("\n## Relaunch")
        print(RELAUNCH_CMD)

    current_opt = load_json(CURRENT_OPT)
    print_json(
        "Current incumbent metrics",
        None
        if current_opt is None
        else {
            "selection_basis": current_opt.get("selection_basis"),
            "portfolio_metrics": current_opt.get("portfolio_metrics"),
            "weights": current_opt.get("weights"),
        },
    )

    latest_run = load_json(LATEST_RUN)
    print_json("Latest run status", latest_run)
    print_mailbox_tail()
    print_rss_tail(latest_run)
    print("\n## Worker pane")
    print("omx sparkshell --tmux-pane %2 --tail-lines 400")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
