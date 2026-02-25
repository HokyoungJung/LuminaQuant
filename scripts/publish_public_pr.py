"""Create a sanitized public sync branch + PR from private code.

Workflow:
1) start from local/private source ref (default: private/main)
2) build a fresh branch from origin/main
3) merge source ref (auto-resolve conflicts by preferring source content)
4) remove protected/sensitive paths from stage
5) commit + push branch
6) open PR to origin/main (optional)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

PROTECTED_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    ".env",
    ".omx",
    ".sisyphus",
    "data",
    "logs",
    "reports",
    "best_optimized_parameters",
    "equity.csv",
    "trades.csv",
    "live_equity.csv",
    "live_trades.csv",
    "strategies",
    "lumina_quant/indicators",
    "lumina_quant/data_sync.py",
    "lumina_quant/data_collector.py",
    "scripts/sync_binance_ohlcv.py",
    "scripts/collect_market_data.py",
    "scripts/collect_universe_1s.py",
    "tests/test_data_sync.py",
)

SENSITIVE_PATH_RE = re.compile(
    r"^strategies/"
    r"|^lumina_quant/indicators/"
    r"|^data/"
    r"|^logs/"
    r"|^reports/"
    r"|^best_optimized_parameters/"
    r"|^\.omx/"
    r"|^\.sisyphus/"
    r"|^AGENTS\.md$"
    r"|^\.env$"
    r"|^lumina_quant/data_sync\.py$"
    r"|^lumina_quant/data_collector\.py$"
    r"|^scripts/sync_binance_ohlcv\.py$"
    r"|^scripts/collect_market_data\.py$"
    r"|^scripts/collect_universe_1s\.py$"
    r"|^tests/test_data_sync\.py$"
    r"|(^|/)live_?equity\.csv$"
    r"|(^|/)live_?trades\.csv$"
    r"|(^|/)equity\.csv$"
    r"|(^|/)trades\.csv$"
)

DEFAULT_PR_BODY = """## Summary
- conflict-free sanitized sync branch from private source
- protected paths remain excluded from public main
- preserves safe runtime/docs improvements only

## Safety gates
- protected path unstage/removal
- sensitive-path regex validation before commit
- PR-based publish (no direct push to public main)
"""


@dataclass(slots=True)
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


def _run(cmd: list[str], *, check: bool = True, capture: bool = True) -> CmdResult:
    proc = subprocess.run(cmd, check=False, capture_output=capture, text=True)
    result = CmdResult(
        returncode=proc.returncode,
        stdout=(proc.stdout or "").strip(),
        stderr=(proc.stderr or "").strip(),
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}")
    return result


def _git(*args: str, check: bool = True) -> CmdResult:
    return _run(["git", *args], check=check, capture=True)


def _gh(*args: str, check: bool = True) -> CmdResult:
    return _run(["gh", *args], check=check, capture=True)


def _current_branch() -> str:
    return _git("branch", "--show-current").stdout.strip()


def _ensure_clean_worktree() -> None:
    status = _git("status", "--porcelain").stdout
    if status:
        raise RuntimeError("Working tree is not clean. Commit/stash changes first.")


def _staged_names() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMRT").stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_sensitive_path(path: str) -> bool:
    return bool(SENSITIVE_PATH_RE.search(str(path).strip()))


def _find_sensitive_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if is_sensitive_path(path)]


def _remove_protected_paths_from_stage() -> None:
    for protected_path in PROTECTED_PATHS:
        _git("rm", "-r", "--cached", "--ignore-unmatch", "--", protected_path, check=False)


def _default_branch_name(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def _detect_repo() -> str | None:
    if shutil.which("gh") is None:
        return None
    result = _gh("repo", "view", "--json", "nameWithOwner", check=False)
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        payload = json.loads(result.stdout)
    except Exception:
        return None
    value = str(payload.get("nameWithOwner") or "").strip()
    return value or None


def _build_pr_create_cmd(*, repo: str | None, base_branch: str, head_branch: str, title: str) -> list[str]:
    cmd = ["gh", "pr", "create", "--base", base_branch, "--head", head_branch, "--title", title]
    if repo:
        cmd.extend(["--repo", repo])
    cmd.extend(["--body", DEFAULT_PR_BODY])
    return cmd


def _find_existing_pr(*, repo: str | None, head_branch: str) -> int | None:
    cmd = ["gh", "pr", "list", "--state", "open", "--head", head_branch, "--json", "number"]
    if repo:
        cmd.extend(["--repo", repo])
    result = _run(cmd, check=False, capture=True)
    if result.returncode != 0 or not result.stdout:
        return None
    try:
        payload = json.loads(result.stdout)
    except Exception:
        return None
    if not payload:
        return None
    try:
        return int(payload[0]["number"])
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sanitized public sync branch + PR.")
    parser.add_argument("--source-ref", default="private/main")
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--head-branch", default="")
    parser.add_argument("--head-prefix", default="public-sync")
    parser.add_argument(
        "--commit-message",
        default="chore(public): sanitized sync from private source",
    )
    parser.add_argument(
        "--pr-title",
        default="chore(public): conflict-free sanitized sync",
    )
    parser.add_argument("--repo", default="")
    parser.add_argument("--no-pr", action="store_true", help="Skip gh pr create.")
    parser.add_argument("--auto-merge", action="store_true", help="Enable PR auto-merge (merge method).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_branch = _current_branch()
    repo = str(args.repo).strip() or _detect_repo()
    head_branch = str(args.head_branch).strip() or _default_branch_name(str(args.head_prefix))

    try:
        _ensure_clean_worktree()
        _git("fetch", "--all", "--prune")
        _git("checkout", "-B", head_branch, args.base_ref)

        merge_result = _git(
            "merge",
            args.source_ref,
            "--no-commit",
            "--no-ff",
            check=False,
        )
        if merge_result.returncode != 0:
            print("[WARN] Merge had conflicts; preferring source side before filtering.", file=sys.stderr)
            _git("checkout", "--theirs", "--", ".", check=False)
            _git("add", "-A", check=False)

        _git("checkout", args.base_ref, "--", ".gitignore")
        _git("reset")
        _git("add", ".")
        _remove_protected_paths_from_stage()

        staged = _staged_names()
        sensitive = _find_sensitive_paths(staged)
        if sensitive:
            joined = "\n - ".join(["", *sensitive])
            raise RuntimeError(f"Sensitive paths are still staged:{joined}")

        if not staged:
            print("[INFO] No public-safe changes to publish.")
            return 0

        _git("commit", "-m", args.commit_message)
        _git("push", "-u", args.remote, head_branch)
        print(f"[OK] Pushed sanitized branch: {head_branch}")

        if args.no_pr:
            return 0

        if shutil.which("gh") is None:
            print("[WARN] gh not found. Skipping PR create.")
            return 0

        pr_number = _find_existing_pr(repo=repo, head_branch=head_branch)
        if pr_number is None:
            create_cmd = _build_pr_create_cmd(
                repo=repo,
                base_branch=args.base_branch,
                head_branch=head_branch,
                title=args.pr_title,
            )
            created = _run(create_cmd, check=True, capture=True)
            if created.stdout:
                print(created.stdout)
            pr_number = _find_existing_pr(repo=repo, head_branch=head_branch)
        else:
            print(f"[INFO] Reusing existing open PR #{pr_number}.")

        if args.auto_merge and pr_number is not None:
            merge_cmd = ["gh", "pr", "merge", str(pr_number), "--auto", "--merge"]
            if repo:
                merge_cmd.extend(["--repo", repo])
            _run(merge_cmd, check=True, capture=True)
            print(f"[OK] Enabled auto-merge for PR #{pr_number}.")

        return 0
    finally:
        if _current_branch() != original_branch:
            _git("checkout", original_branch, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
