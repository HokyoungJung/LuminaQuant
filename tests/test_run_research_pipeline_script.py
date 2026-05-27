from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_run_research_pipeline_dry_run_skips_outputs(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_research_pipeline.py"
    env = dict(os.environ)
    env["LQ_GPU_MODE"] = "cpu"
    env["LQ_CONFIG_PATH"] = str(root / "config.yaml")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path),
            "--timeframes",
            "1m",
            "--seeds",
            "20260221",
            "--dry-run",
        ],
        cwd=str(tmp_path),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "dry-run mode: no output files written" in result.stdout
    assert not list(tmp_path.glob("strategy_factory_*.json")), (
        "dry-run should not write pipeline output json"
    )
    assert not list(tmp_path.glob("strategy_factory_*.md")), (
        "dry-run should not write pipeline output markdown"
    )
