from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_candidate_research_script_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_candidate_research.py"
    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        str(tmp_path),
        "--max-candidates",
        "48",
        "--timeframes",
        "1m",
        "5m",
        "1h",
        "--symbols",
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XAU/USDT",
        "XAG/USDT",
    ]

    result = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Saved:" in result.stdout
    assert (tmp_path / "candidate_research_latest.json").exists()
    assert (tmp_path / "strategy_factory_report_latest.json").exists()


def test_run_candidate_research_script_smoke_with_score_config(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_candidate_research.py"
    score_cfg_path = tmp_path / "score_config.json"
    score_cfg_path.write_text(
        json.dumps(
            {
                "candidate_rank_score_weights": {
                    "sharpe_weight": 0.2,
                    "return_weight": 45.0,
                },
                "keep_ratio_bounds": {"min": 0.1, "max": 0.8},
            }
        ),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        str(tmp_path),
        "--max-candidates",
        "24",
        "--timeframes",
        "1m",
        "5m",
        "--symbols",
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "--score-config",
        str(score_cfg_path),
    ]

    result = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    latest = tmp_path / "candidate_research_latest.json"
    assert latest.exists()
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload.get("scoring_config", {}).get("candidate_rank_score_weights", {}).get("return_weight") == 45.0
