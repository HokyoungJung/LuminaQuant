from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_portfolio_optimization_script_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]

    research = {
        "schema_version": "v2",
        "candidates": [
            {
                "candidate_id": "c1",
                "name": "trend_a",
                "strategy_class": "CompositeTrendStrategy",
                "family": "trend",
                "strategy_timeframe": "1h",
                "symbols": ["BTC/USDT"],
                "oos": {"sharpe": 1.2, "return": 0.1, "deflated_sharpe": 0.6, "pbo": 0.2, "turnover": 0.4},
                "pass": True,
                "return_streams": {
                    "train": [{"t": float(i), "v": 0.0003} for i in range(80)],
                    "val": [{"t": float(i), "v": 0.0002} for i in range(40)],
                    "oos": [{"t": float(i), "v": 0.0004} for i in range(40)],
                },
                "metadata": {"cost_rate": 0.0005},
            },
            {
                "candidate_id": "c2",
                "name": "pair_b",
                "strategy_class": "PairSpreadZScoreStrategy",
                "family": "market_neutral",
                "strategy_timeframe": "15m",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "oos": {"sharpe": 1.0, "return": 0.08, "deflated_sharpe": 0.5, "pbo": 0.25, "turnover": 0.8},
                "pass": True,
                "return_streams": {
                    "train": [{"t": float(i), "v": 0.0002} for i in range(80)],
                    "val": [{"t": float(i), "v": 0.0001} for i in range(40)],
                    "oos": [{"t": float(i), "v": 0.0003} for i in range(40)],
                },
                "metadata": {"cost_rate": 0.0006},
            },
        ],
    }

    research_path = tmp_path / "candidate_research.json"
    team_path = tmp_path / "team_report.json"
    out_dir = tmp_path / "reports"

    research_path.write_text(json.dumps(research), encoding="utf-8")
    team_path.write_text(
        json.dumps({"selected_team": research["candidates"]}),
        encoding="utf-8",
    )

    script = root / "scripts" / "run_portfolio_optimization.py"
    cmd = [
        sys.executable,
        str(script),
        "--research-report",
        str(research_path),
        "--team-report",
        str(team_path),
        "--output-dir",
        str(out_dir),
        "--max-strategies",
        "2",
    ]

    result = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (out_dir / "portfolio_optimization_latest.json").exists()
    assert (out_dir / "portfolio_optimization_latest.json").read_text(encoding="utf-8")


def test_run_portfolio_optimization_uses_score_config_weights_and_caps(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]

    candidates = [
        {
            "candidate_id": "c1",
            "name": "trend_a",
            "strategy_class": "CompositeTrendStrategy",
            "family": "trend",
            "strategy_timeframe": "1h",
            "symbols": ["BTC/USDT"],
            "oos": {"sharpe": 0.6, "return": 0.03, "deflated_sharpe": 0.2, "pbo": 0.4, "turnover": 0.3},
            "pass": True,
            "return_streams": {
                "train": [{"t": float(i), "v": 0.0002} for i in range(60)],
                "val": [{"t": float(i), "v": 0.0001} for i in range(30)],
                "oos": [{"t": float(i), "v": 0.0002} for i in range(30)],
            },
            "metadata": {"cost_rate": 0.0005},
        },
        {
            "candidate_id": "c2",
            "name": "trend_b",
            "strategy_class": "CompositeTrendStrategy",
            "family": "trend",
            "strategy_timeframe": "1h",
            "symbols": ["ETH/USDT"],
            "oos": {"sharpe": 0.3, "return": 0.02, "deflated_sharpe": 0.1, "pbo": 0.3, "turnover": 0.2},
            "pass": True,
            "return_streams": {
                "train": [{"t": float(i), "v": 0.0001} for i in range(60)],
                "val": [{"t": float(i), "v": 0.00005} for i in range(30)],
                "oos": [{"t": float(i), "v": 0.0001} for i in range(30)],
            },
            "metadata": {"cost_rate": 0.0005},
        },
    ]

    research_path = tmp_path / "candidate_research.json"
    team_path = tmp_path / "team_report.json"
    score_config_path = tmp_path / "score_config.json"
    out_dir = tmp_path / "reports"
    research_path.write_text(json.dumps({"schema_version": "v2", "candidates": candidates}), encoding="utf-8")
    team_path.write_text(json.dumps({"selected_team": candidates}), encoding="utf-8")
    score_config_path.write_text(
        json.dumps(
            {
                "portfolio_optimization": {
                    "candidate_rank_score_weights": {
                        "sharpe_weight": 1.0,
                        "deflated_sharpe_weight": 0.5,
                        "pbo_penalty": 3.0,
                        "return_weight": 60.0,
                    },
                    "vol_targeting": {
                        "target_vol_floor": 0.20,
                        "vol_scale_cap": 1.25,
                        "vol_scale_epsilon": 1e-9,
                    },
                    "sensitivity": {
                        "cost_stress_x2_multiplier": 1.0,
                        "cost_stress_x3_multiplier": 1.0,
                        "signal_drift_down_multiplier": 1.0,
                        "signal_drift_up_multiplier": 1.0,
                    },
                    "constraints": {
                        "max_strategy": 0.55,
                        "max_family": 0.90,
                        "max_asset": 0.95,
                        "max_metals": 0.20,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    script = root / "scripts" / "run_portfolio_optimization.py"
    cmd = [
        sys.executable,
        str(script),
        "--research-report",
        str(research_path),
        "--team-report",
        str(team_path),
        "--score-config",
        str(score_config_path),
        "--output-dir",
        str(out_dir),
        "--max-strategies",
        "2",
    ]
    result = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    payload = json.loads((out_dir / "portfolio_optimization_latest.json").read_text(encoding="utf-8"))
    scoring = dict(payload.get("scoring") or {})
    weights = dict(scoring.get("candidate_rank_score_weights") or {})
    assert float(weights.get("return_weight", 0.0)) == 60.0
    assert float(weights.get("pbo_penalty", 0.0)) == 3.0

    vol_targeting = dict(scoring.get("vol_targeting") or {})
    assert float(vol_targeting.get("target_vol_floor", 0.0)) == 0.2
    assert float(vol_targeting.get("vol_scale_cap", 0.0)) == 1.25
    assert float(vol_targeting.get("vol_scale_epsilon", 0.0)) == 1e-9

    sensitivity_scoring = dict(scoring.get("sensitivity") or {})
    assert float(sensitivity_scoring.get("cost_stress_x2_multiplier", 0.0)) == 1.0
    assert float(sensitivity_scoring.get("cost_stress_x3_multiplier", 0.0)) == 1.0
    assert float(sensitivity_scoring.get("signal_drift_down_multiplier", 0.0)) == 1.0
    assert float(sensitivity_scoring.get("signal_drift_up_multiplier", 0.0)) == 1.0

    portfolio_oos = dict((payload.get("portfolio_metrics") or {}).get("oos") or {})
    sensitivity = dict(payload.get("sensitivity") or {})
    cost_stress = dict(sensitivity.get("cost_stress") or {})
    param_drift = dict(sensitivity.get("param_drift") or {})
    sections = (
        dict(cost_stress.get("x2") or {}),
        dict(cost_stress.get("x3") or {}),
        dict(param_drift.get("minus_10pct_signal") or {}),
        dict(param_drift.get("plus_10pct_signal") or {}),
    )
    for section in sections:
        assert abs(float(section.get("total_return", 0.0)) - float(portfolio_oos.get("total_return", 0.0))) < 1e-12

    constraints = dict(payload.get("constraints") or {})
    configured = dict(constraints.get("configured") or {})
    assert float(configured.get("max_strategy", 0.0)) == 0.55
    assert float(constraints.get("max_strategy", 0.0)) <= 0.550001


def test_run_portfolio_optimization_enforces_strategy_cap_when_feasible(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]

    def _candidate(idx: int) -> dict:
        cid = f"c{idx}"
        family = "trend" if idx % 2 == 0 else "market_neutral"
        symbol = "BTC/USDT" if idx % 3 == 0 else ("ETH/USDT" if idx % 3 == 1 else "SOL/USDT")
        return {
            "candidate_id": cid,
            "name": f"candidate_{cid}",
            "strategy_class": "CompositeTrendStrategy" if family == "trend" else "PairSpreadZScoreStrategy",
            "family": family,
            "strategy_timeframe": "1h",
            "symbols": [symbol],
            "oos": {
                "sharpe": 1.5 - (idx * 0.05),
                "return": 0.08 - (idx * 0.001),
                "deflated_sharpe": 0.7 - (idx * 0.01),
                "pbo": 0.1 + (idx * 0.01),
                "turnover": 0.2 + (idx * 0.02),
            },
            "pass": True,
            "return_streams": {
                "train": [{"t": float(i), "v": 0.0002 + (idx * 1e-6)} for i in range(120)],
                "val": [{"t": float(i), "v": 0.00015 + (idx * 1e-6)} for i in range(60)],
                "oos": [{"t": float(i), "v": 0.00025 + (idx * 1e-6)} for i in range(60)],
            },
            "metadata": {"cost_rate": 0.0005},
        }

    candidates = [_candidate(i) for i in range(10)]
    research_path = tmp_path / "candidate_research.json"
    team_path = tmp_path / "team_report.json"
    out_dir = tmp_path / "reports"
    research_path.write_text(json.dumps({"schema_version": "v2", "candidates": candidates}), encoding="utf-8")
    team_path.write_text(json.dumps({"selected_team": candidates}), encoding="utf-8")

    script = root / "scripts" / "run_portfolio_optimization.py"
    cmd = [
        sys.executable,
        str(script),
        "--research-report",
        str(research_path),
        "--team-report",
        str(team_path),
        "--output-dir",
        str(out_dir),
        "--max-strategies",
        "10",
    ]
    result = subprocess.run(cmd, cwd=str(root), check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    payload = json.loads((out_dir / "portfolio_optimization_latest.json").read_text(encoding="utf-8"))
    weights = list(payload.get("weights") or [])
    assert len(weights) == 10

    total_weight = sum(float(row.get("weight", 0.0)) for row in weights)
    assert abs(total_weight - 1.0) < 1e-6

    constraints = dict(payload.get("constraints") or {})
    max_strategy = float(constraints.get("max_strategy", 0.15))
    assert max_strategy <= 0.150001
    assert all(float(row.get("weight", 0.0)) <= max_strategy + 1e-6 for row in weights)
