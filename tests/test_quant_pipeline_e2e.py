from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from lumina_quant.strategy_factory.research_runner import (
    build_default_candidate_rows,
    run_candidate_research,
)
from lumina_quant.strategy_factory.selection import select_diversified_shortlist


def test_quant_pipeline_end_to_end(tmp_path: Path):
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XAU/USDT", "XAG/USDT"]
    timeframes = ["1m", "5m", "1h"]

    candidates = build_default_candidate_rows(
        symbols=symbols,
        timeframes=timeframes,
        max_candidates=64,
    )
    assert candidates

    report = run_candidate_research(
        candidates=candidates,
        base_timeframe="1s",
        strategy_timeframes=timeframes,
        symbol_universe=symbols,
        stage1_keep_ratio=0.5,
        max_candidates=64,
    )

    rows = list(report.get("candidates") or [])
    assert rows
    assert all("return_streams" in row for row in rows)

    shortlist = select_diversified_shortlist(
        rows,
        mode="oos",
        max_total=12,
        max_per_family=6,
        max_per_timeframe=6,
        include_weights=True,
    )
    assert shortlist

    research_path = tmp_path / "candidate_research.json"
    team_path = tmp_path / "strategy_factory_report.json"
    output_dir = tmp_path / "reports"

    research_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    team_path.write_text(
        json.dumps(
            {
                "generated_at": report.get("generated_at"),
                "schema_version": "v2",
                "selected_team": shortlist,
                "candidates": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_portfolio_optimization.py"
    cmd = [
        sys.executable,
        str(script),
        "--research-report",
        str(research_path),
        "--team-report",
        str(team_path),
        "--output-dir",
        str(output_dir),
        "--max-strategies",
        "10",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    latest = output_dir / "portfolio_optimization_latest.json"
    assert latest.exists()

    payload = json.loads(latest.read_text(encoding="utf-8"))
    weights = list(payload.get("weights") or [])
    assert weights

    total_weight = sum(float(row.get("weight", 0.0)) for row in weights)
    assert abs(total_weight - 1.0) < 1e-6
