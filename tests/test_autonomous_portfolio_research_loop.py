from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "lumina_quant" / "workflows" / "autonomous_portfolio_research_loop.py"
SPEC = importlib.util.spec_from_file_location("autonomous_portfolio_research_loop", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load autonomous_portfolio_research_loop module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_stack_audit_reflects_promoted_challenger(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    followup_root = report_root / "followup_status"
    followup_root.mkdir(parents=True, exist_ok=True)
    current_opt_dir = followup_root / "portfolio_one_shot_current_opt"
    current_opt_dir.mkdir(parents=True, exist_ok=True)

    incumbent_bundle = {
        "artifact_kind": "portfolio_one_shot_incumbent_bundle",
        "candidates": [
            {
                "name": "legacy-topcap",
                "strategy_class": "TopCapTimeSeriesMomentumStrategy",
                "strategy_timeframe": "1h",
                "portfolio_weight": 0.5,
                "train": {
                    "total_return": -0.10,
                    "sharpe": -0.4,
                    "stability": -1.5,
                    "rolling_sharpe_min": -20.0,
                },
                "val": {"total_return": 0.02},
                "oos": {"total_return": 0.03},
            },
            {
                "name": "legacy-regime",
                "strategy_class": "RegimeBreakoutCandidateStrategy",
                "strategy_timeframe": "1h",
                "portfolio_weight": 0.5,
                "train": {
                    "total_return": -0.12,
                    "sharpe": -0.6,
                    "stability": -1.6,
                    "rolling_sharpe_min": -24.0,
                },
                "val": {"total_return": 0.03},
                "oos": {"total_return": 0.04},
            },
        ],
    }
    incumbent_portfolio = {
        "artifact_kind": "portfolio_optimization",
        "portfolio_metrics": {
            "train": {"total_return": -0.11},
            "val": {"total_return": 0.025},
            "oos": {"total_return": 0.035},
        },
    }
    portfolio_decision = {
        "winner": {
            "candidate_key": "autonomous_cross_sectional_1h_tradecount_pair_topcap_opt",
            "label": "Autonomous 1h tradecount pair+topcap challenger",
            "status": "promoted_challenger",
        }
    }
    exact_window_decision = {"promoted_total": 0, "next_action": "ralplan_team_ralph_required"}

    (followup_root / "portfolio_one_shot_incumbent_bundle_latest.json").write_text(
        json.dumps(incumbent_bundle),
        encoding="utf-8",
    )
    (current_opt_dir / "portfolio_optimization_latest.json").write_text(
        json.dumps(incumbent_portfolio),
        encoding="utf-8",
    )
    (followup_root / "portfolio_max_performance_decision_latest.json").write_text(
        json.dumps(portfolio_decision),
        encoding="utf-8",
    )
    (report_root / "exact_window_decision_latest.json").write_text(
        json.dumps(exact_window_decision),
        encoding="utf-8",
    )

    output_path = followup_root / "autonomous_research_loop" / "stack_audit_latest.md"
    result = MODULE.build_stack_audit(report_root=report_root, output_path=output_path)
    rendered = output_path.read_text(encoding="utf-8")

    assert result["path"] == str(output_path.resolve())
    assert (
        "Current promotion winner: `Autonomous 1h tradecount pair+topcap challenger` (promoted_challenger)"
        in rendered
    )
    assert (
        "A challenger now wins locked OOS while the current incumbent artifacts still describe the older baseline."
        in rendered
    )
    assert (
        "The locked-OOS promotion flow now favors `Autonomous 1h tradecount pair+topcap challenger`"
        in rendered
    )
    assert "The incumbent is still the locked-OOS winner" not in rendered
