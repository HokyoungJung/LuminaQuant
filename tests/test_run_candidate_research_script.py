from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_candidate_research.py"
    spec = importlib.util.spec_from_file_location("run_candidate_research_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_candidate_research module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


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
                "shortlist_selection": {
                    "max_per_family": 3,
                    "max_per_timeframe": 3,
                    "single_min_score": -0.1,
                    "single_min_return": -0.1,
                    "single_min_sharpe": -0.1,
                    "single_min_trades": 1,
                    "allow_multi_asset": True,
                    "include_weights": True,
                    "weight_temperature": 0.2,
                    "max_weight": 0.5,
                    "robust_score_params": {
                        "failed_candidate_scale": 0.2,
                        "mdd_risk_penalty_coeff": 3.0,
                    },
                },
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

    team_report = json.loads((tmp_path / "strategy_factory_report_latest.json").read_text(encoding="utf-8"))
    shortlist_config = dict(team_report.get("shortlist_config") or {})
    assert int(shortlist_config.get("max_per_family", 0)) == 3
    assert int(shortlist_config.get("max_per_timeframe", 0)) == 3
    assert float(shortlist_config.get("weight_temperature", 0.0)) == 0.2
    assert float(shortlist_config.get("max_weight", 0.0)) == 0.5
    robust_params = dict(shortlist_config.get("robust_score_params") or {})
    assert float(robust_params.get("failed_candidate_scale", 0.0)) == 0.2
    assert float(robust_params.get("mdd_risk_penalty_coeff", 0.0)) == 3.0


def test_shortlist_selection_config_nested_scope_and_cli_precedence():
    payload = {
        "candidate_research": {
            "shortlist_selection": {
                "max_per_family": 7,
                "max_per_timeframe": 5,
                "weight_temperature": 0.2,
                "max_weight": 0.5,
            }
        }
    }
    scope = MODULE._score_config_scope(payload)
    resolved = MODULE._resolve_shortlist_selection_config(scope, top_k=11)
    assert resolved["max_total"] == 11
    assert resolved["max_per_family"] == 7
    assert resolved["max_per_timeframe"] == 5
    assert float(resolved["weight_temperature"]) == 0.2
    assert float(resolved["max_weight"]) == 0.5


def test_shortlist_robust_score_params_cross_corr_penalty_and_override_precedence():
    payload = {
        "candidate_research": {
            "candidate_rank_score_weights": {
                "turnover_penalty": 2.2,
                "cross_corr_penalty": 0.9,
            },
            "reject_thresholds": {"max_turnover": 1.7},
            "shortlist_selection": {
                "robust_score_params": {
                    "cross_corr_penalty": 0.4,
                }
            },
        }
    }
    scope = MODULE._score_config_scope(payload)
    params = MODULE._shortlist_robust_score_params(scope)
    assert params is not None
    assert float(params.get("turnover_penalty", 0.0)) == 2.2
    assert float(params.get("turnover_threshold", 0.0)) == 1.7
    assert float(params.get("cross_corr_penalty", 0.0)) == 0.4
