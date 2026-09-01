from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "write_portfolio_operating_playbook.py"
)
SPEC = importlib.util.spec_from_file_location("write_portfolio_operating_playbook", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_playbook_includes_hybrid_guarded_mode() -> None:
    payload = MODULE.build_playbook(
        base_plan={
            "deployment_modes": {
                "core_mode": {"allocation": {"soft_three_way_regime": 1.0}},
                "balanced_overlay_mode": {
                    "allocation": {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}
                },
                "aggressive_realized_mode": {"allocation": {"three_way_regime": 1.0}},
            }
        },
        switch_validation={
            "refreshed_metrics": {
                "risk_off_cash": {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
                "strategy1_balanced_overlay_80_20": {
                    "total_return": -0.01,
                    "sharpe": -1.0,
                    "max_drawdown": 0.02,
                },
                "switch_strategy_core_soft100": {
                    "total_return": -0.02,
                    "sharpe": -2.0,
                    "max_drawdown": 0.03,
                },
                "aggressive_three_way": {
                    "total_return": -0.03,
                    "sharpe": -3.0,
                    "max_drawdown": 0.04,
                },
            },
            "comparison_switch_vs_strategy1": {
                "oos_return_delta": 0.01,
                "oos_sharpe_delta": 1.0,
                "oos_max_drawdown_delta": -0.01,
            },
        },
        switch_recommendation={
            "current_market_state": {
                "favored_group": "incumbent",
                "pair_liquidity_state": "normal",
            },
            "recommended_mode": {
                "mode": "hybrid_guarded_mode",
                "allocation": {"hybrid_online_portfolio": 1.0},
            },
        },
        bearish_scan={"ranked_by_oos_return_then_sharpe": []},
        hybrid_payload={
            "scenarios": {
                "refreshed_latest_tail": {
                    "split_metrics": {
                        "oos": {"total_return": 0.02, "sharpe": 2.0, "max_drawdown": 0.01}
                    },
                    "source_sleeve_metrics": {
                        "soft_three_way_regime": {
                            "oos": {"total_return": 0.03, "sharpe": 1.5, "max_drawdown": 0.02}
                        },
                        "balanced_overlay_80_20": {
                            "oos": {"total_return": 0.04, "sharpe": 1.7, "max_drawdown": 0.015}
                        },
                        "three_way_regime": {
                            "oos": {"total_return": 0.05, "sharpe": 1.9, "max_drawdown": 0.03}
                        },
                    },
                }
            },
            "readiness": {"recommended_stage": "guarded_candidate", "beats_cash_refreshed": True},
        },
        production_guarded_payload={
            "active_exposure": 0.95,
            "cash_weight": 0.05,
            "carry_candidate_included": False,
            "portfolio_metrics": {
                "oos": {"total_return": 0.03, "sharpe": 1.8, "max_drawdown": 0.008},
            },
        },
    )
    hybrid = payload["deployment_modes"]["hybrid_guarded_mode"]
    production = payload["deployment_modes"]["production_guarded_mode"]
    assert hybrid["allocation"] == {"hybrid_online_portfolio": 1.0}
    assert production["allocation"] == {"production_guarded_portfolio": 1.0}
    assert production["metrics"]["total_return"] == 0.03
    assert hybrid["metrics"]["total_return"] == 0.02
    assert payload["deployment_modes"]["core_mode"]["metrics"]["total_return"] == 0.03
    assert payload["deployment_modes"]["balanced_overlay_mode"]["metrics"]["total_return"] == 0.04
    assert (
        payload["deployment_modes"]["aggressive_realized_mode"]["metrics"]["total_return"] == 0.05
    )
    assert payload["recommended_mode"]["mode"] == "hybrid_guarded_mode"
