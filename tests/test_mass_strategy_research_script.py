from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_mass_strategy_research_dry_run_smoke():
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "run_mass_strategy_research.py"
    cmd = [sys.executable, str(script), "--dry-run", "--timeframes", "1m", "--seeds", "20260221"]
    result = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert result.returncode == 0
    assert "[PIPELINE] candidate manifest:" in result.stdout
    assert "[PIPELINE] shortlist json:" in result.stdout
