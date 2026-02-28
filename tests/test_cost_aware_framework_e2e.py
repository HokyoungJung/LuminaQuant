from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cost_aware_framework import run_cost_aware_framework


def test_cost_aware_framework_e2e(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]

    experiment = {
        "base_config": str(root / "configs/quant_framework/base.yaml"),
        "experiment_id": "e2e_cost_aware",
        "timeframes": ["5m", "1h", "1d"],
        "run_controls": {
            "start_date": "2025-01-01T00:00:00Z",
            "end_date": "2025-01-02T00:00:00Z",
            "seed": 7,
            "use_synthetic_data": True,
            "synthetic_bars": 600,
            "initial_capital": 50000,
        },
        "strategies": [
            {"config": str(root / "configs/quant_framework/strategies/trend_momentum.yaml")},
            {"config": str(root / "configs/quant_framework/strategies/xs_mean_reversion.yaml")},
        ],
    }

    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    report_dir = run_cost_aware_framework(
        experiment_path=str(exp_path),
        output_root=str(tmp_path / "reports"),
        run_id="e2e-run",
    )

    assert (report_dir / "summary.json").exists()
    assert (report_dir / "tables.csv").exists()
    assert (report_dir / "config_resolved.yaml").exists()

    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert "results" in summary
    assert "post_cost_timeframe_ranking" in summary
    assert "best_timeframe" in summary
    assert "ranking_objective" in summary
    assert summary["ranking_objective"]["name"] in {"composite", "sharpe", "total_return", "drawdown"}
    for strategy_name, rows in summary["post_cost_timeframe_ranking"].items():
        _ = strategy_name
        assert rows
        assert "post_cost_score" in rows[0]
    assert "calibration" in summary
    for payload in summary["calibration"].values():
        assert "mae_before_bps" in payload
        assert "mae_after_bps" in payload
        assert "error_reduction_pct" in payload
    assert len(summary["results"]) >= 6  # 2 strategies x 3 timeframes
