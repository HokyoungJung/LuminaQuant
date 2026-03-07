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
    assert publish_public_pr.is_sensitive_path("lumina_quant/strategies/rsi_strategy.py")
    assert publish_public_pr.is_sensitive_path("strategies/rsi_strategy.py")
    assert publish_public_pr.is_sensitive_path("lumina_quant/indicators/formulaic_alpha.py")
    assert publish_public_pr.is_sensitive_path("reports/benchmarks/latest.json")
    assert publish_public_pr.is_sensitive_path(".github/hardcoded_params_baseline.json")
    assert publish_public_pr.is_sensitive_path(".env")


def test_sensitive_path_detection_allows_public_runtime_paths():
    assert not publish_public_pr.is_sensitive_path("run_backtest.py")
    assert not publish_public_pr.is_sensitive_path("lumina_quant/backtesting/chunked_runner.py")
    assert not publish_public_pr.is_sensitive_path("docs/RUNBOOK_1Y_1S_LOCAL.md")


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
