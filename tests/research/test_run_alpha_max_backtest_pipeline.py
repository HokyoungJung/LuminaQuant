from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.research import run_alpha_max_backtest_pipeline as subject


def _plan(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    files: dict[str, str] = {}
    for name in ("contract_manifest", "config", "prior_trial_blob"):
        path = inputs / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        files[name] = str(path)
    canonical_db = inputs / "market_parquet"
    canonical_db.mkdir()
    phase_roots: dict[str, str] = {}
    for name in (
        "warmup_raw",
        "warmup_feature",
        "train_raw",
        "train_feature",
        "purge_raw",
        "purge_feature",
        "validation_raw",
        "validation_feature",
        "embargo_raw",
        "embargo_feature",
        "historical_evaluation_raw",
        "historical_evaluation_feature",
    ):
        path = inputs / name
        path.mkdir()
        phase_roots[name] = str(path)
    plan: dict[str, object] = {
        "schema_version": subject.SCHEMA,
        "exchange": "binance",
        "order_routing_enabled": False,
        "contract_manifest": files["contract_manifest"],
        "canonical_db": str(canonical_db),
        "config": files["config"],
        "prior_trial_blob": files["prior_trial_blob"],
        "phase_roots": phase_roots,
        "run_root": str(tmp_path / "run"),
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path, plan


def test_plan_builds_one_integration_first_backtest_pipeline(tmp_path: Path) -> None:
    plan_path, expected = _plan(tmp_path)

    plan = subject.load_plan(plan_path)
    commands = subject.build_commands(
        plan,
        python=Path("/usr/bin/python3"),
        repository=Path("/repo"),
    )

    assert plan == expected
    assert [stage for stage, _ in commands] == list(subject.STAGES)
    assert commands[0][1][1].endswith("verify_alpha_max_canonical_pipeline.py")
    assert commands[1][1][1].endswith("run_alpha_max_prelock.py")
    assert "--validation-raw-root" in commands[1][1]
    assert commands[2][1][1].endswith("run_alpha_max_historical_evaluation.py")
    assert commands[3][1][-2:] == [
        "--output",
        f"{expected['run_root']}/observability/validation.json",
    ]
    assert commands[3][1][commands[3][1].index("--manifest-root") + 1] == (
        f"{expected['run_root']}/prelock"
    )
    assert commands[4][1][commands[4][1].index("--domain") + 1] == ("historical_exposed_evaluation")


def test_plan_rejects_order_routing_and_incomplete_phase_inventory(tmp_path: Path) -> None:
    plan_path, plan = _plan(tmp_path)
    plan["order_routing_enabled"] = True
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="safety_invalid"):
        subject.load_plan(plan_path)

    plan["order_routing_enabled"] = False
    phase_roots = dict(plan["phase_roots"])
    phase_roots.pop("validation_raw")
    plan["phase_roots"] = phase_roots
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="phase_roots_invalid"):
        subject.load_plan(plan_path)


def test_pipeline_creates_checkpoint_parent_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, plan = _plan(tmp_path)
    observed: list[Path] = []

    def complete(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        run_root = Path(str(plan["run_root"]))
        observed.append(run_root / "checkpoints")
        assert observed[-1].is_dir()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subject.subprocess, "run", complete)

    receipt = subject.run_pipeline(plan_path)

    assert receipt["status"] == "complete"
    assert len(observed) == len(subject.STAGES)
