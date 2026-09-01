from __future__ import annotations

from pathlib import Path

from lumina_quant import portfolio_split_contract as contract


def test_resolve_followup_artifact_path_prefers_repo_artifact_over_worktree_copy(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    target_rel = Path(
        "var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json"
    )
    repo_target = repo_root / target_rel
    worktree_target = repo_root / ".omx" / "team" / "demo" / "worktrees" / "worker-9" / target_rel
    repo_target.parent.mkdir(parents=True, exist_ok=True)
    worktree_target.parent.mkdir(parents=True, exist_ok=True)
    repo_target.write_text("repo", encoding="utf-8")
    worktree_target.write_text("worktree", encoding="utf-8")
    worktree_target.touch()

    monkeypatch.setattr(contract, "ROOT", repo_root)

    resolved = contract.resolve_followup_artifact_path(target_rel)

    assert resolved == repo_target


def test_resolve_followup_artifact_path_falls_back_to_latest_worktree(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    target_rel = Path(
        "var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json"
    )
    newer = repo_root / ".omx" / "team" / "demo" / "worktrees" / "worker-2" / target_rel
    older = repo_root / ".omx" / "team" / "demo" / "worktrees" / "worker-1" / target_rel
    older.parent.mkdir(parents=True, exist_ok=True)
    newer.parent.mkdir(parents=True, exist_ok=True)
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    older.touch()
    newer.touch()

    monkeypatch.setattr(contract, "ROOT", repo_root)

    resolved = contract.resolve_followup_artifact_path(target_rel)

    assert resolved == newer


def test_resolve_followup_artifact_path_uses_generic_worktree_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    target_rel = Path(
        "var/reports/exact_window_backtests/followup_status/autoresearch_candidate_portfolio_opt/portfolio_optimization_latest.json"
    )
    generic = repo_root / ".omx" / "worktrees" / "autoresearch-run" / target_rel
    generic.parent.mkdir(parents=True, exist_ok=True)
    generic.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(contract, "ROOT", repo_root)

    resolved = contract.resolve_followup_artifact_path(target_rel)

    assert resolved == generic.resolve()


def test_resolve_current_optimization_path_uses_worktree_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    current_opt = (
        repo_root
        / ".omx"
        / "team"
        / "demo"
        / "worktrees"
        / "worker-1"
        / "var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json"
    )
    current_opt.parent.mkdir(parents=True, exist_ok=True)
    current_opt.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(contract, "ROOT", repo_root)
    monkeypatch.setattr(
        contract,
        "PORTFOLIO_CURRENT_OPTIMIZATION",
        Path(
            "var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json"
        ),
    )

    resolved = contract.resolve_current_optimization_path()

    assert resolved == current_opt


def test_resolve_current_optimization_path_uses_worktree_fallback_for_absolute_repo_path(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    absolute_repo_target = (
        repo_root
        / "var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json"
    )
    current_opt = (
        repo_root
        / ".omx"
        / "team"
        / "demo"
        / "worktrees"
        / "worker-1"
        / "var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json"
    )
    current_opt.parent.mkdir(parents=True, exist_ok=True)
    current_opt.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(contract, "ROOT", repo_root)

    resolved = contract.resolve_current_optimization_path(absolute_repo_target)

    assert resolved == current_opt
