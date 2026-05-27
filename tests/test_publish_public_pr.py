from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "publish_public_pr.py"
_SPEC = importlib.util.spec_from_file_location("publish_public_pr_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {_SCRIPT_PATH}")
publish_public_pr = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = publish_public_pr
_SPEC.loader.exec_module(publish_public_pr)


def test_sensitive_path_detection_blocks_private_trees():
    assert publish_public_pr.is_sensitive_path("scripts/publish_public_pr.py")
    assert publish_public_pr.is_sensitive_path("publish_api.ps1")
    assert publish_public_pr.is_sensitive_path("publish_api.sh")
    assert publish_public_pr.is_sensitive_path("tests/test_publish_public_pr.py")
    assert publish_public_pr.is_sensitive_path(".github/workflows/private-ci.yml")
    assert publish_public_pr.is_sensitive_path("docs/WORKFLOW.md")
    assert publish_public_pr.is_sensitive_path("docs/WSL_CLONE_PRIVATE_PUBLIC.md")
    assert publish_public_pr.is_sensitive_path("docs/kr/WORKFLOW.md")
    assert publish_public_pr.is_sensitive_path("lumina_quant/strategies/rsi_strategy.py")
    assert publish_public_pr.is_sensitive_path("strategies/rsi_strategy.py")
    assert publish_public_pr.is_sensitive_path("lumina_quant/indicators/formulaic_alpha.py")
    assert publish_public_pr.is_sensitive_path(
        "src/lumina_quant/strategy_factory/candidate_library.py"
    )
    assert publish_public_pr.is_sensitive_path(
        "src/lumina_quant/strategy_factory/research_runner.py"
    )
    assert publish_public_pr.is_sensitive_path(
        "src/lumina_quant/workflows/alpha_research_pipeline.py"
    )
    assert publish_public_pr.is_sensitive_path("src/lumina_quant/eval/exact_window_suite.py")
    assert publish_public_pr.is_sensitive_path("src/lumina_quant/dashboard/exact_window_bundle.py")
    assert publish_public_pr.is_sensitive_path("scripts/run_research_candidates.py")
    assert publish_public_pr.is_sensitive_path("tests/test_exact_window_suite.py")
    assert publish_public_pr.is_sensitive_path("scripts/research/run_llm_alpha_pipeline.py")
    assert publish_public_pr.is_sensitive_path(
        "scripts/research/write_exact_window_deployment_combo.py"
    )
    assert publish_public_pr.is_sensitive_path("tests/test_strategy_factory_library.py")
    assert publish_public_pr.is_sensitive_path("tests/test_research_runner_feature_support.py")
    assert publish_public_pr.is_sensitive_path("scripts/run_bulk_research.py")
    assert publish_public_pr.is_sensitive_path("scripts/run_research_pipeline.py")
    assert publish_public_pr.is_sensitive_path("scripts/run_research_hurdle.py")
    assert publish_public_pr.is_sensitive_path("scripts/ci/architecture_gate_binance_native.sh")
    assert publish_public_pr.is_sensitive_path("tests/test_run_research_pipeline_script.py")
    assert publish_public_pr.is_sensitive_path("reports/benchmarks/latest.json")
    assert publish_public_pr.is_sensitive_path(".agents/skills/alpha-research-pipeline/SKILL.md")
    assert publish_public_pr.is_sensitive_path(".codex/prompts/architect.md")
    assert publish_public_pr.is_sensitive_path(".github/hardcoded_params_baseline.json")
    assert publish_public_pr.is_sensitive_path(".env")


def test_sensitive_path_detection_allows_public_runtime_paths():
    assert not publish_public_pr.is_sensitive_path("run_backtest.py")
    assert not publish_public_pr.is_sensitive_path("lumina_quant/backtesting/chunked_runner.py")
    assert not publish_public_pr.is_sensitive_path("docs/RUNBOOK_1Y_1S_LOCAL.md")
    assert not publish_public_pr.is_sensitive_path(".github/workflows/ci.yml")
    assert not publish_public_pr.is_sensitive_path("src/lumina_quant/strategy.py")
    assert not publish_public_pr.is_sensitive_path("src/lumina_quant/services/portfolio.py")
    assert not publish_public_pr.is_sensitive_path(
        "src/lumina_quant/backtesting/portfolio_backtest.py"
    )
    assert not publish_public_pr.is_sensitive_path("tests/test_portfolio_sizing.py")
    assert not publish_public_pr.is_sensitive_path("src/lumina_quant/dashboard/retired_stub.py")
    assert not publish_public_pr.is_sensitive_path(
        "src/lumina_quant/strategies/sample_public_strategy.py"
    )
    assert not publish_public_pr.is_sensitive_path(
        "src/lumina_quant/indicators/sample_public_indicator.py"
    )


def test_sensitive_path_detection_blocks_generic_strategy_indicator_like_names():
    assert publish_public_pr.is_sensitive_path("docs/FUTURES_STRATEGY_FACTORY.md")
    assert publish_public_pr.is_sensitive_path("configs/profiles/research.yaml")
    assert publish_public_pr.is_sensitive_path("scripts/build_live_portfolio_dashboard_summary.py")


def test_default_branch_name_uses_prefix():
    branch = publish_public_pr._default_branch_name("public-sync")
    assert branch.startswith("public-sync-")


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path) -> None:
    _run_git(repo, "init", "-b", "main")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.com")


def test_restore_protected_paths_from_base_preserves_public_samples(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    protected = repo / "src" / "lumina_quant" / "strategies"
    protected.mkdir(parents=True)
    sample_file = protected / "sample_public_strategy.py"
    sample_file.write_text("PUBLIC_SAMPLE = True\n", encoding="utf-8")
    safe_file = repo / "README.md"
    safe_file.write_text("public base\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "base")

    sample_file.write_text("PRIVATE_SECRET = True\n", encoding="utf-8")
    safe_file.write_text("public safe change\n", encoding="utf-8")
    _run_git(repo, "add", ".")

    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        publish_public_pr,
        "PROTECTED_PATHS",
        ("src/lumina_quant/strategies",),
    )

    publish_public_pr._restore_protected_paths_from_base()

    assert sample_file.read_text(encoding="utf-8") == "PUBLIC_SAMPLE = True\n"
    staged = _run_git(repo, "diff", "--cached", "--name-only").stdout.splitlines()
    assert "README.md" in staged
    assert "src/lumina_quant/strategies/sample_public_strategy.py" not in staged


def test_restore_protected_paths_from_base_drops_private_only_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    (repo / "README.md").write_text("public base\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "base")

    private_only = repo / "src" / "lumina_quant" / "indicators" / "advanced_alpha.py"
    private_only.parent.mkdir(parents=True)
    private_only.write_text("SECRET_FACTOR = 1\n", encoding="utf-8")
    _run_git(repo, "add", ".")

    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        publish_public_pr,
        "PROTECTED_PATHS",
        ("src/lumina_quant/indicators",),
    )

    publish_public_pr._restore_protected_paths_from_base()

    assert not private_only.exists()
    staged = _run_git(repo, "diff", "--cached", "--name-only").stdout.splitlines()
    assert "src/lumina_quant/indicators/advanced_alpha.py" not in staged


def test_drop_paths_from_public_removes_existing_public_publish_tooling(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    publish_script = repo / "scripts" / "publish_public_pr.py"
    publish_script.parent.mkdir(parents=True)
    publish_script.write_text("print('public copy')\n", encoding="utf-8")
    workflow_doc = repo / "docs" / "WORKFLOW.md"
    workflow_doc.parent.mkdir(parents=True, exist_ok=True)
    workflow_doc.write_text("# internal workflow\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "base")

    monkeypatch.chdir(repo)
    monkeypatch.setattr(
        publish_public_pr,
        "DROP_FROM_PUBLIC_PATHS",
        ("scripts/publish_public_pr.py", "docs/WORKFLOW.md"),
    )

    publish_public_pr._drop_paths_from_public()

    assert not publish_script.exists()
    assert not workflow_doc.exists()
    assert publish_public_pr._staged_names() == []
    assert set(publish_public_pr._staged_any_names()) == {
        "docs/WORKFLOW.md",
        "scripts/publish_public_pr.py",
    }
    staged_deleted = _run_git(repo, "diff", "--cached", "--name-status").stdout.splitlines()
    assert staged_deleted == ["D\tdocs/WORKFLOW.md", "D\tscripts/publish_public_pr.py"]


def test_legacy_public_cleanup_inventory_covers_known_private_paths():
    assert "AGENTS.md" in publish_public_pr.DROP_FROM_PUBLIC_PATHS
    assert "configs/quant_framework/strategies" in publish_public_pr.DROP_FROM_PUBLIC_PATHS
    assert "src/lumina_quant/strategies/registry.py" in publish_public_pr.DROP_FROM_PUBLIC_PATHS
    assert "tests/test_strategy_registry_defaults.py" in publish_public_pr.DROP_FROM_PUBLIC_PATHS
    assert "var" in publish_public_pr.DROP_FROM_PUBLIC_PATHS


def test_publish_flow_keeps_source_gitignore_for_safe_generated_artifacts(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "README.md").write_text("public base\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "base")

    _run_git(repo, "checkout", "-b", "source")
    (repo / ".gitignore").write_text("generated/\n", encoding="utf-8")
    (repo / "README.md").write_text("public safe change\n", encoding="utf-8")
    generated = repo / "generated" / "artifact.txt"
    generated.parent.mkdir(parents=True)
    generated.write_text("local build output\n", encoding="utf-8")
    _run_git(repo, "add", ".gitignore", "README.md")
    _run_git(repo, "commit", "-m", "source")

    monkeypatch.chdir(repo)
    _run_git(repo, "checkout", "main")
    merge = _run_git(repo, "merge", "source", "--no-commit", "--no-ff")
    assert merge.returncode == 0

    _run_git(repo, "reset")
    _run_git(repo, "add", ".")

    staged = _run_git(repo, "diff", "--cached", "--name-only").stdout.splitlines()
    assert ".gitignore" in staged
    assert "README.md" in staged
    assert "generated/artifact.txt" not in staged


def test_content_sensitive_detection_blocks_public_path_with_private_reference(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: ci\njobs:\n  test:\n    steps:\n      - run: python scripts/run_research_pipeline.py\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/ci.yml"])
    assert flagged == [".github/workflows/ci.yml"]


def test_content_sensitive_detection_allows_public_safe_ci_workflow(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "\n".join(
            [
                "name: ci",
                "jobs:",
                "  quality:",
                "    steps:",
                "      - run: uv sync --extra optimize --extra dev --extra live --extra dashboard",
                "      - run: uv run python scripts/check_architecture.py",
                "      - run: uv run pytest -q",
                "      - run: cd apps/dashboard_web && npm run build",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/ci.yml"])
    assert flagged == []


def test_content_sensitive_detection_allows_generic_public_domain_terms(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    readme = repo / "README.md"
    readme.write_text(
        "This document discusses strategy research and indicator rollout details.\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths(["README.md"])
    assert flagged == []


def test_content_sensitive_detection_allows_explicit_public_sample_terms(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    readme = repo / "README.md"
    readme.write_text(
        "Public sample strategy docs may reference PublicSampleStrategy and sample_public_indicator.\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths(["README.md"])
    assert flagged == []


def test_content_sensitive_detection_derives_module_patterns_from_protected_inventory(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    module = repo / "public_probe.py"
    module.write_text(
        "from lumina_quant.eval.exact_window_suite import run_exact_window_suite\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths(["public_probe.py"])
    assert flagged == ["public_probe.py"]


def test_content_sensitive_detection_blocks_root_level_exact_protected_file_references(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "root-files.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: root-files\njobs:\n  check:\n    steps:\n      - run: cat AGENTS.md && cat .env && cat equity.csv\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/root-files.yml"])
    assert flagged == [".github/workflows/root-files.yml"]


def test_content_sensitive_detection_blocks_protected_directory_descendants(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "release.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: release\njobs:\n  publish:\n    steps:\n      - run: cp reports/private.json artifacts/out.json\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/release.yml"])
    assert flagged == [".github/workflows/release.yml"]


def test_content_sensitive_detection_blocks_nested_protected_directory_descendants(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "nested.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: nested\njobs:\n  publish:\n    steps:\n      - run: cat src/lumina_quant/strategies/private_alpha.py\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths([".github/workflows/nested.yml"])
    assert flagged == [".github/workflows/nested.yml"]


def test_content_sensitive_detection_blocks_dot_prefixed_exact_protected_paths(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    workflow = repo / ".github" / "workflows" / "dot-protected.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        "name: dot-protected\njobs:\n  check:\n    steps:\n      - run: cat .github/hardcoded_params_baseline.json\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths(
        [".github/workflows/dot-protected.yml"]
    )
    assert flagged == [".github/workflows/dot-protected.yml"]


def test_public_tree_exposures_catch_preexisting_sensitive_public_tree_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    readme = repo / "README.md"
    readme.write_text("strategy research content\n", encoding="utf-8")
    strategy_doc = repo / "docs" / "FUTURES_STRATEGY_FACTORY.md"
    strategy_doc.parent.mkdir(parents=True, exist_ok=True)
    strategy_doc.write_text("safe?\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "base")

    monkeypatch.chdir(repo)
    path_hits, content_hits = publish_public_pr._public_tree_exposures()
    assert "docs/FUTURES_STRATEGY_FACTORY.md" in path_hits
    assert content_hits == []


def test_content_sensitive_detection_ignores_publish_sanitizer_self_test_file(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    test_file = repo / "tests" / "test_publish_public_pr.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "assert 'scripts/run_research_pipeline.py'\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    flagged = publish_public_pr._find_sensitive_content_paths(["tests/test_publish_public_pr.py"])
    assert flagged == []
