from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumina_quant.backtesting import rigorous_pipeline as pipeline


def _plan(repository: Path) -> tuple[Path, dict]:
    data_root = repository / "data" / "market_parquet"
    data_root.mkdir(parents=True)
    data_receipt = repository / "data_receipt.json"
    data_receipt.write_text('{"status":"complete"}\n')
    script = repository / "scripts" / "stage.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(0)\n")
    run_root = repository / "var" / "backtests" / "run"
    stages = []
    for name in pipeline.STAGE_ORDER:
        stages.append(
            {
                "name": name,
                "script": str(script),
                "arguments": [f"${{RUN_ROOT}}/{name}.json"],
                "inputs": [str(data_receipt)],
                "outputs": [f"${{RUN_ROOT}}/{name}.json"],
                "environment": {"LQ_RAW_FIRST_BACKEND": "rust"},
                "accepted_return_codes": [0],
            }
        )
    value = {
        "schema_version": pipeline.SCHEMA,
        "data_root": str(data_root),
        "data_receipt": str(data_receipt),
        "run_root": str(run_root),
        "order_routing_enabled": False,
        "memory_max_bytes": 1024**3,
        "stages": stages,
    }
    path = repository / "plan.json"
    path.write_text(json.dumps(value))
    return path, value


def test_plan_binds_one_data_root_and_safe_run_tree(tmp_path: Path) -> None:
    path, _value = _plan(tmp_path)

    result = pipeline.load_plan(path, repository=tmp_path)

    assert result["data_root"] == tmp_path / "data" / "market_parquet"
    assert result["run_root"] == tmp_path / "var" / "backtests" / "run"
    assert tuple(stage["name"] for stage in result["stages"]) == pipeline.STAGE_ORDER


def test_plan_resolves_portable_repository_relative_paths(tmp_path: Path) -> None:
    path, value = _plan(tmp_path)
    value["data_root"] = "data/market_parquet"
    value["data_receipt"] = "data_receipt.json"
    value["run_root"] = "var/backtests/run"
    for stage in value["stages"]:
        stage["script"] = "scripts/stage.py"
        stage["inputs"] = ["data_receipt.json"]
    path.write_text(json.dumps(value))

    result = pipeline.load_plan(path, repository=tmp_path)

    assert result["data_root"] == tmp_path / "data" / "market_parquet"
    assert result["run_root"] == tmp_path / "var" / "backtests" / "run"


def test_plan_rejects_output_inside_canonical_data(tmp_path: Path) -> None:
    path, value = _plan(tmp_path)
    value["stages"][0]["outputs"] = ["${DATA_ROOT}/forbidden.json"]
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="outputs must be nonsymlink"):
        pipeline.load_plan(path, repository=tmp_path)


def test_resume_reuses_content_addressed_stage_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _value = _plan(tmp_path)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        if argv[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="a" * 40 + "\n", returncode=0)
        calls.append(list(argv))
        Path(argv[-1]).write_text(json.dumps({"stage": Path(argv[-1]).stem}) + "\n")
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    first = pipeline.run_pipeline(path, repository=tmp_path)
    second = pipeline.run_pipeline(path, repository=tmp_path, resume=True)

    assert first["status"] == "complete"
    assert len(calls) == len(pipeline.STAGE_ORDER)
    assert [row["status"] for row in second["stages"]] == ["reused"] * len(pipeline.STAGE_ORDER)
