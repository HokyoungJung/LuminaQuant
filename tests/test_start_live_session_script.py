from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ops" / "start_live_session.sh"
STOP_SCRIPT = ROOT / "scripts" / "ops" / "stop_live_session.sh"
ALIAS_SCRIPT = ROOT / "scripts" / "ops" / "install_shell_aliases.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_stop(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(STOP_SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_alias(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ALIAS_SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_start_live_session_help_mentions_run_bot_difference() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "run_bot.sh" in result.stdout
    assert "one safe launch" in result.stdout


def test_start_live_session_dry_run_prints_preparation_and_launch_steps() -> None:
    result = _run("--dry-run", "--no-env-file", "--dsn", "postgresql:///luminaquant")

    assert result.returncode == 0
    assert "uv run python scripts/init_postgres_schema.py" in result.stdout
    assert (
        "uv run python scripts/research/refresh_final_portfolio_validation_data.py" in result.stdout
    )
    assert "uv run python scripts/ops/live_readiness_preflight.py" in result.stdout
    assert (
        "uv run python scripts/research/write_portfolio_live_readiness_decision.py" in result.stdout
    )
    assert "uv run lq live --transport poll" in result.stdout
    assert "--stop-file /tmp/lq-paper.stop" in result.stdout


def test_start_live_session_real_mode_requires_allow_real() -> None:
    result = _run("--real", "--dry-run", "--no-env-file")

    assert result.returncode == 2
    assert "--real requires --allow-real" in result.stderr


def test_start_live_session_real_mode_rejects_skip_preflight() -> None:
    result = _run("--real", "--allow-real", "--skip-preflight", "--dry-run", "--no-env-file")

    assert result.returncode == 2
    assert "cannot use --skip-preflight" in result.stderr


def test_start_live_session_real_mode_dry_run_adds_real_flag_and_keeps_preflight() -> None:
    result = _run(
        "--real", "--allow-real", "--dry-run", "--no-env-file", "--dsn", "postgresql:///luminaquant"
    )

    assert result.returncode == 0
    assert "Preflight: 1" in result.stdout
    assert "uv run python scripts/ops/live_readiness_preflight.py" in result.stdout
    assert "--enable-live-real" in result.stdout


def test_stop_live_session_wrapper_defaults_to_paper_stop_file() -> None:
    result = _run_stop()

    assert result.returncode == 0
    assert "/tmp/lq-paper.stop" in result.stdout


def test_install_shell_aliases_print_contains_easy_helpers() -> None:
    result = _run_alias("--print")

    assert result.returncode == 0
    assert "lq-paper-on()" in result.stdout
    assert "lq-paper-off()" in result.stdout
    assert "lq-real-on()" in result.stdout
    assert "lq-real-off()" in result.stdout
