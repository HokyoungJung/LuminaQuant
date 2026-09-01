"""Create a sanitized public sync branch + PR from private code.

Workflow:
1) start from local/private source ref (default: private/main)
2) build a fresh branch from origin/main
3) merge source ref (auto-resolve conflicts by preferring source content)
4) restore protected/sensitive paths to the public-base version
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

INTERNAL_PUBLISH_PATHS: tuple[str, ...] = (
    "publish_api.ps1",
    "publish_api.sh",
    "scripts/publish_public_pr.py",
    "tests/test_publish_public_pr.py",
    ".github/workflows/private-ci.yml",
    "docs/WORKFLOW.md",
    "docs/WSL_CLONE_PRIVATE_PUBLIC.md",
    "docs/kr/WORKFLOW.md",
)

PROTECTED_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    ".agents",
    ".codex",
    ".env",
    ".omx",
    ".sisyphus",
    "var",
    "data",
    "logs",
    "reports",
    "best_optimized_parameters",
    "equity.csv",
    "trades.csv",
    "live_equity.csv",
    "live_trades.csv",
    ".github/hardcoded_params_baseline.json",
    "src/lumina_quant/strategies",
    "lumina_quant/strategies",
    "strategies",
    "src/lumina_quant/indicators",
    "lumina_quant/indicators",
    "src/lumina_quant/data_sync.py",
    "src/lumina_quant/data_collector.py",
    "lumina_quant/data_sync.py",
    "lumina_quant/data_collector.py",
    "src/lumina_quant/strategy_factory/candidate_library.py",
    "src/lumina_quant/strategy_factory/research_runner.py",
    "src/lumina_quant/workflows/alpha_research_pipeline.py",
    "src/lumina_quant/dashboard/exact_window_bundle.py",
    "src/lumina_quant/cli/exact_window.py",
    "src/lumina_quant/eval/exact_window_decision.py",
    "src/lumina_quant/eval/exact_window_log_archive.py",
    "src/lumina_quant/eval/exact_window_reporting.py",
    "src/lumina_quant/eval/exact_window_runtime.py",
    "src/lumina_quant/eval/exact_window_suite.py",
    "scripts/sync_binance_ohlcv.py",
    "scripts/research/rebuild_exact_window_registry_from_logs.py",
    "scripts/research/refresh_exact_window_bundle.py",
    "scripts/run_research_pipeline.py",
    "scripts/run_research_hurdle.py",
    "scripts/run_bulk_research.py",
    "scripts/research/run_llm_alpha_pipeline.py",
    "scripts/research/write_exact_window_deployment_combo.py",
    "scripts/collect_strategy_support_data.py",
    "scripts/collect_all_strategy_support_data.py",
    "scripts/backfill_funding_fee_features.py",
    "scripts/collect_market_data.py",
    "scripts/collect_universe_1s.py",
    "scripts/run_exact_window_suite.py",
    "scripts/run_portfolio_optimization.py",
    "scripts/run_research_candidates.py",
    "scripts/ci/architecture_gate_binance_native.sh",
    "tests/test_advanced_alpha_indicators.py",
    "tests/test_alpha_research_pipeline.py",
    "tests/test_alpha101_formula_strategy.py",
    "tests/test_data_sync.py",
    "tests/test_exact_window_cli.py",
    "tests/test_exact_window_dashboard_loader.py",
    "tests/test_exact_window_decision.py",
    "tests/test_exact_window_deployment_combo.py",
    "tests/test_exact_window_log_archive.py",
    "tests/test_exact_window_pair_focus_profiles.py",
    "tests/test_exact_window_reporting.py",
    "tests/test_exact_window_runtime.py",
    "tests/test_exact_window_suite.py",
    "tests/test_research_runner_feature_support.py",
    "tests/test_research_runner_exact_split.py",
    "tests/test_research_runner_scoring_config.py",
    "tests/test_run_exact_window_suite_script.py",
    "tests/test_run_llm_alpha_pipeline_script.py",
    "tests/test_run_portfolio_optimization_script.py",
    "tests/test_run_research_candidates_script.py",
    "tests/test_run_research_pipeline_script.py",
    "tests/test_strategy_alias_compat.py",
    "tests/test_strategy_factory_library.py",
    "tests/test_strategy_support_collection_profiles.py",
    "tests/test_topcap_tsmom_strategy.py",
    *INTERNAL_PUBLISH_PATHS,
)

DROP_FROM_PUBLIC_PATHS: tuple[str, ...] = INTERNAL_PUBLISH_PATHS

LEGACY_PUBLIC_CLEANUP_PATHS: tuple[str, ...] = (
    ".github/hardcoded_params_baseline.json",
    "AGENTS.md",
    "configs/profiles/research.yaml",
    "configs/quant_framework/strategies",
    "docs/FUTURES_STRATEGY_FACTORY.md",
    "docs/QUANT_RESEARCH_FACTORY.md",
    "docs/STRATEGY_FACTORY_WORKER3_REPORT.md",
    "docs/STRATEGY_FACTORY_WORKER4_REPORT.md",
    "docs/kr/FUTURES_STRATEGY_FACTORY.md",
    "scripts/benchmark_indicators.py",
    "scripts/build_live_portfolio_dashboard_summary.py",
    "scripts/build_strategy_support_inventory.py",
    "scripts/collect_strategy_support_data.py",
    "scripts/export_research_candidates.py",
    "scripts/generate_alpha_card_template.py",
    "scripts/run_bulk_research.py",
    "scripts/run_portfolio_optimization.py",
    "scripts/run_research_candidates.py",
    "scripts/run_research_hurdle.py",
    "scripts/run_research_pipeline.py",
    "scripts/select_research_shortlist.py",
    "src/lumina_quant/data_collector.py",
    "src/lumina_quant/data_sync.py",
    "src/lumina_quant/indicators/__init__.py",
    "src/lumina_quant/strategies/__init__.py",
    "src/lumina_quant/strategies/registry.py",
    "src/lumina_quant/strategy_factory/candidate_library.py",
    "src/lumina_quant/strategy_factory/research_runner.py",
    "tests/test_accelerated_indicators.py",
    "tests/test_advanced_alpha_indicators.py",
    "tests/test_advanced_strategy_state.py",
    "tests/test_alpha101_registry_compiler.py",
    "tests/test_alpha_card_template.py",
    "tests/test_bulk_research_script.py",
    "tests/test_candidate_strategy_factory.py",
    "tests/test_factory_fast_indicators.py",
    "tests/test_formulaic_alpha_runtime.py",
    "tests/test_formulaic_alpha_subset.py",
    "tests/test_indicator_expansion.py",
    "tests/test_indicators_core.py",
    "tests/test_indicators_extended.py",
    "tests/test_rare_event_indicator.py",
    "tests/test_rare_event_strategy.py",
    "tests/test_research_pipeline.py",
    "tests/test_research_runner_feature_support.py",
    "tests/test_research_runner_scoring_config.py",
    "tests/test_run_portfolio_optimization_script.py",
    "tests/test_run_research_candidates_script.py",
    "tests/test_run_research_hurdle_script.py",
    "tests/test_run_research_pipeline_script.py",
    "tests/test_select_research_shortlist_script.py",
    "tests/test_strategy_catalog_new_strategies.py",
    "tests/test_strategy_factory_library.py",
    "tests/test_strategy_long_short_support.py",
    "tests/test_strategy_plugins_timeframe.py",
    "tests/test_strategy_registry_defaults.py",
    "tests/test_strategy_support_inventory.py",
    "tests/test_topcap_tsmom_strategy.py",
    "var",
)

DROP_FROM_PUBLIC_PATHS = DROP_FROM_PUBLIC_PATHS + LEGACY_PUBLIC_CLEANUP_PATHS

SENSITIVE_PATH_RE = re.compile(
    r"^src/lumina_quant/strategies/"
    r"|^lumina_quant/strategies/"
    r"|^strategies/"
    r"|^src/lumina_quant/indicators/"
    r"|^lumina_quant/indicators/"
    r"|^data/"
    r"|^var/"
    r"|^logs/"
    r"|^reports/"
    r"|^best_optimized_parameters/"
    r"|^\.omx/"
    r"|^\.sisyphus/"
    r"|^\.agents/"
    r"|^\.codex/"
    r"|^AGENTS\.md$"
    r"|^\.env$"
    r"|^\.github/hardcoded_params_baseline\.json$"
    r"|^src/lumina_quant/data_sync\.py$"
    r"|^src/lumina_quant/data_collector\.py$"
    r"|^lumina_quant/data_sync\.py$"
    r"|^lumina_quant/data_collector\.py$"
    r"|^src/lumina_quant/strategy_factory/candidate_library\.py$"
    r"|^src/lumina_quant/strategy_factory/research_runner\.py$"
    r"|^src/lumina_quant/workflows/alpha_research_pipeline\.py$"
    r"|^src/lumina_quant/dashboard/exact_window_bundle\.py$"
    r"|^src/lumina_quant/cli/exact_window\.py$"
    r"|^src/lumina_quant/eval/exact_window_decision\.py$"
    r"|^src/lumina_quant/eval/exact_window_log_archive\.py$"
    r"|^src/lumina_quant/eval/exact_window_reporting\.py$"
    r"|^src/lumina_quant/eval/exact_window_runtime\.py$"
    r"|^src/lumina_quant/eval/exact_window_suite\.py$"
    r"|^scripts/sync_binance_ohlcv\.py$"
    r"|^scripts/research/rebuild_exact_window_registry_from_logs\.py$"
    r"|^scripts/research/refresh_exact_window_bundle\.py$"
    r"|^scripts/run_research_pipeline\.py$"
    r"|^scripts/run_research_hurdle\.py$"
    r"|^scripts/run_bulk_research\.py$"
    r"|^scripts/research/run_llm_alpha_pipeline\.py$"
    r"|^scripts/research/write_exact_window_deployment_combo\.py$"
    r"|^scripts/collect_strategy_support_data\.py$"
    r"|^scripts/collect_all_strategy_support_data\.py$"
    r"|^scripts/backfill_funding_fee_features\.py$"
    r"|^scripts/collect_market_data\.py$"
    r"|^scripts/collect_universe_1s\.py$"
    r"|^scripts/run_exact_window_suite\.py$"
    r"|^scripts/run_portfolio_optimization\.py$"
    r"|^scripts/run_research_candidates\.py$"
    r"|^scripts/ci/architecture_gate_binance_native\.sh$"
    r"|^tests/test_advanced_alpha_indicators\.py$"
    r"|^tests/test_alpha_research_pipeline\.py$"
    r"|^tests/test_alpha101_formula_strategy\.py$"
    r"|^tests/test_data_sync\.py$"
    r"|^tests/test_exact_window_cli\.py$"
    r"|^tests/test_exact_window_dashboard_loader\.py$"
    r"|^tests/test_exact_window_decision\.py$"
    r"|^tests/test_exact_window_deployment_combo\.py$"
    r"|^tests/test_exact_window_log_archive\.py$"
    r"|^tests/test_exact_window_pair_focus_profiles\.py$"
    r"|^tests/test_exact_window_reporting\.py$"
    r"|^tests/test_exact_window_runtime\.py$"
    r"|^tests/test_exact_window_suite\.py$"
    r"|^tests/test_research_runner_feature_support\.py$"
    r"|^tests/test_research_runner_exact_split\.py$"
    r"|^tests/test_research_runner_scoring_config\.py$"
    r"|^tests/test_run_exact_window_suite_script\.py$"
    r"|^tests/test_run_llm_alpha_pipeline_script\.py$"
    r"|^tests/test_run_portfolio_optimization_script\.py$"
    r"|^tests/test_run_research_candidates_script\.py$"
    r"|^tests/test_run_research_pipeline_script\.py$"
    r"|^tests/test_strategy_alias_compat\.py$"
    r"|^tests/test_strategy_factory_library\.py$"
    r"|^tests/test_strategy_support_collection_profiles\.py$"
    r"|^tests/test_topcap_tsmom_strategy\.py$"
    r"|^publish_api\.ps1$"
    r"|^publish_api\.sh$"
    r"|^scripts/publish_public_pr\.py$"
    r"|^tests/test_publish_public_pr\.py$"
    r"|^\.github/workflows/private-ci\.yml$"
    r"|^docs/WORKFLOW\.md$"
    r"|^docs/WSL_CLONE_PRIVATE_PUBLIC\.md$"
    r"|^docs/kr/WORKFLOW\.md$"
    r"|(^|/)live_?equity\.csv$"
    r"|(^|/)live_?trades\.csv$"
    r"|(^|/)equity\.csv$"
    r"|(^|/)trades\.csv$"
)

CONTENT_SCAN_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".sh",
    ".ps1",
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".md",
    ".txt",
)
CONTENT_SCAN_EXEMPT_PATHS: tuple[str, ...] = ()

ALLOWED_PUBLIC_PATHS: tuple[str, ...] = (
    "src/lumina_quant/strategies/sample_public_strategy.py",
    "lumina_quant/strategies/sample_public_strategy.py",
    "src/lumina_quant/indicators/sample_public_indicator.py",
    "lumina_quant/indicators/sample_public_indicator.py",
    "src/lumina_quant/backtesting/portfolio_backtest.py",
    "src/lumina_quant/compute/indicators.py",
    "src/lumina_quant/services/portfolio.py",
    "src/lumina_quant/strategy.py",
    "tests/test_funding_liquidation.py",
    "tests/test_portfolio_fast_stats.py",
    "tests/test_portfolio_sizing.py",
    "tests/test_portfolio_trade_recording.py",
)

ALLOWED_PUBLIC_SAMPLE_CONTENT_TOKENS: tuple[str, ...] = (
    "sample_public_strategy",
    "sample_public_indicator",
    "PublicSampleStrategy",
    "public sample strategy",
    "public sample indicator",
    "sample strategy",
    "sample indicator",
)

GENERIC_SENSITIVE_PATH_RE = re.compile(
    r"(^|/)[^/]*(strategy|strategies|indicator|indicators|research|exact_window|alpha|funding|momentum|reversion|breakout|portfolio)[^/]*$",
    re.IGNORECASE,
)

GENERIC_SENSITIVE_CONTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bstrategy_factory\b", re.IGNORECASE),
    re.compile(r"\bresearch_runner\b", re.IGNORECASE),
    re.compile(r"\bexact_window\b", re.IGNORECASE),
    re.compile(r"\bstrateg(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"\bindicators?\b", re.IGNORECASE),
    re.compile(r"\bresearch\b", re.IGNORECASE),
    re.compile(r"\balpha\b", re.IGNORECASE),
    re.compile(r"\bfunding\b", re.IGNORECASE),
    re.compile(r"\bmomentum\b", re.IGNORECASE),
    re.compile(r"\breversion\b", re.IGNORECASE),
    re.compile(r"\bbreakout\b", re.IGNORECASE),
    re.compile(r"\bportfolio\b", re.IGNORECASE),
)

DEFAULT_PR_BODY = """## Summary
- conflict-free sanitized sync branch from private source
- protected paths are restored from the current public base
- preserves safe runtime/docs improvements only

## Safety gates
- protected path restore-from-base
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
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )
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


def _staged_any_names() -> list[str]:
    out = _git("diff", "--cached", "--name-only").stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def _tracked_names() -> list[str]:
    out = _git("ls-files").stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def _is_allowed_public_path(path: str) -> bool:
    normalized = str(path or "").strip()
    return normalized in ALLOWED_PUBLIC_PATHS


def is_sensitive_path(path: str) -> bool:
    normalized = str(path or "").strip()
    if not normalized:
        return False
    if _is_allowed_public_path(normalized):
        return False
    return bool(
        SENSITIVE_PATH_RE.search(normalized) or GENERIC_SENSITIVE_PATH_RE.search(normalized)
    )


def _find_sensitive_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if is_sensitive_path(path)]


def _module_token_from_path(path: str) -> str | None:
    normalized = str(path or "").strip()
    if normalized.startswith("src/") and normalized.endswith(".py"):
        return normalized.removeprefix("src/").removesuffix(".py").replace("/", ".")
    if normalized.startswith("lumina_quant/") and normalized.endswith(".py"):
        return normalized.removesuffix(".py").replace("/", ".")
    return None


def _content_sensitive_patterns() -> tuple[re.Pattern[str], ...]:
    patterns: list[re.Pattern[str]] = []
    for protected_path in PROTECTED_PATHS:
        normalized = str(protected_path or "").strip().strip("/")
        if not normalized:
            continue
        is_exact_file = "." in normalized.rsplit("/", 1)[-1]
        if is_exact_file:
            patterns.append(
                re.compile(rf"(?<![A-Za-z0-9_./-]){re.escape(normalized)}(?![A-Za-z0-9_./-])")
            )
        else:
            patterns.append(re.compile(rf"(?<![A-Za-z0-9_./-]){re.escape(normalized)}/[^\s'\"]+"))
        if (
            normalized.startswith("src/lumina_quant/strategies")
            or normalized.startswith("lumina_quant/strategies")
            or normalized == "strategies"
        ):
            patterns.append(
                re.compile(
                    r"\blumina_quant\.strategies\.(?!sample_public_strategy\b)[A-Za-z_][\w.]*"
                )
            )
            continue
        if normalized.startswith("src/lumina_quant/indicators") or normalized.startswith(
            "lumina_quant/indicators"
        ):
            patterns.append(
                re.compile(
                    r"\blumina_quant\.indicators\.(?!sample_public_indicator\b)[A-Za-z_][\w.]*"
                )
            )
            continue
        module_token = _module_token_from_path(normalized)
        if module_token:
            patterns.append(re.compile(rf"\b{re.escape(module_token)}\b"))
    return tuple(patterns)


def _should_content_scan(path: str) -> bool:
    normalized = str(path or "").strip()
    if not normalized or is_sensitive_path(normalized):
        return False
    if _is_allowed_public_path(normalized):
        return False
    if normalized in CONTENT_SCAN_EXEMPT_PATHS:
        return False
    return normalized.endswith(CONTENT_SCAN_EXTENSIONS)


def _find_sensitive_content_paths(paths: list[str]) -> list[str]:
    patterns = _content_sensitive_patterns()
    flagged: list[str] = []
    for path in paths:
        if not _should_content_scan(path):
            continue
        candidate = str(path).strip()
        try:
            with open(candidate, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            continue
        normalized_text = text
        for token in ALLOWED_PUBLIC_SAMPLE_CONTENT_TOKENS:
            normalized_text = re.sub(re.escape(token), "", normalized_text, flags=re.IGNORECASE)
        if any(pattern.search(normalized_text) for pattern in patterns):
            flagged.append(candidate)
    return flagged


def _public_tree_exposures() -> tuple[list[str], list[str]]:
    tracked = _tracked_names()
    return _find_sensitive_paths(tracked), _find_sensitive_content_paths(tracked)


def _restore_protected_paths_from_base() -> None:
    for protected_path in PROTECTED_PATHS:
        _git(
            "restore",
            "--source=HEAD",
            "--staged",
            "--worktree",
            "--",
            protected_path,
            check=False,
        )


def _drop_paths_from_public() -> None:
    for path in DROP_FROM_PUBLIC_PATHS:
        _git("rm", "-r", "-f", "--ignore-unmatch", "--", path, check=False)


def _default_branch_name(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def _detect_repo(remote: str) -> str | None:
    remote_name = str(remote or "").strip()
    if remote_name:
        result = _git("remote", "get-url", remote_name, check=False)
        if result.returncode == 0 and result.stdout:
            url = result.stdout.strip()
            if url.endswith(".git"):
                url = url[:-4]
            if url.startswith("git@github.com:"):
                return url.removeprefix("git@github.com:")
            marker = "github.com/"
            if marker in url:
                return url.split(marker, 1)[1].strip("/") or None
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


def _build_pr_create_cmd(
    *, repo: str | None, base_branch: str, head_branch: str, title: str
) -> list[str]:
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
    parser.add_argument(
        "--auto-merge", action="store_true", help="Enable PR auto-merge (merge method)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_branch = _current_branch()
    repo = str(args.repo).strip() or _detect_repo(str(args.remote))
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
            print(
                "[WARN] Merge had conflicts; preferring source side before filtering.",
                file=sys.stderr,
            )
            _git("checkout", "--theirs", "--", ".", check=False)
            _git("add", "-A", check=False)

        _git("reset")
        _git("add", ".")
        _restore_protected_paths_from_base()
        _drop_paths_from_public()

        staged = _staged_names()
        sensitive = _find_sensitive_paths(staged)
        if sensitive:
            joined = "\n - ".join(["", *sensitive])
            raise RuntimeError(f"Sensitive paths are still staged:{joined}")
        content_sensitive = _find_sensitive_content_paths(staged)
        if content_sensitive:
            joined = "\n - ".join(["", *content_sensitive])
            raise RuntimeError(f"Sensitive content references are still staged:{joined}")

        tree_sensitive_paths, tree_sensitive_content = _public_tree_exposures()
        if tree_sensitive_paths:
            joined = "\n - ".join(["", *tree_sensitive_paths[:100]])
            raise RuntimeError(f"Public tree still contains sensitive paths:{joined}")
        if tree_sensitive_content:
            joined = "\n - ".join(["", *tree_sensitive_content[:100]])
            print(
                f"[WARN] Public tree still contains legacy sensitive-content references:{joined}",
                file=sys.stderr,
            )

        if not _staged_any_names():
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
            _git("checkout", "-f", original_branch, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
