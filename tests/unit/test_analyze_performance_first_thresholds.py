import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "analyze_performance_first_thresholds.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_performance_first_thresholds", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_report_captures_frontier_maxima_for_current_hybrid_case() -> None:
    switch_payload = {
        "recommended_mode": {"mode": "hybrid_guarded_mode"},
        "current_market_state": {
            "favored_group": "mixed",
            "confidence": 0.0,
            "trend_state": "bullish",
            "breadth_state": "broad",
            "volatility_state": "calm",
            "pair_liquidity_state": "normal",
        },
    }
    hybrid_payload = {
        "readiness": {
            "beats_balanced_refreshed": True,
            "beats_pair_tactical_refreshed": True,
            "recommended_stage": "pilot_candidate",
        },
        "scenarios": {
            "refreshed_latest_tail": {
                "split_metrics": {
                    "val": {"total_return": 0.06537, "sharpe": 3.2857},
                    "oos": {"total_return": 0.006868, "sharpe": 3.2370, "max_drawdown": 0.002573},
                },
                "source_sleeve_metrics": {
                    "balanced_overlay_80_20": {
                        "val": {"total_return": 0.08308, "sharpe": 4.1120},
                        "oos": {
                            "total_return": 0.001091,
                            "sharpe": 0.4828,
                            "max_drawdown": 0.005162,
                        },
                    }
                },
            }
        },
    }

    report = MODULE.build_report(
        switch_payload=switch_payload,
        hybrid_payload=hybrid_payload,
        return_grid=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        sharpe_grid=[0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
        val_return_grid=[0.03, 0.04, 0.05, 0.06, 0.07],
        val_sharpe_grid=[2.0, 2.5, 3.0, 3.5, 4.0],
    )

    assert report["switch_mode"] == "hybrid_guarded_mode"
    assert report["current_thresholds"]["min_oos_return_edge"] == 0.005
    assert report["current_thresholds"]["min_oos_sharpe_edge"] == 2.5
    assert report["current_thresholds"]["min_val_return"] == 0.06
    assert report["current_thresholds"]["min_val_sharpe"] == 3.0
    assert report["frontier"]["passing_count"] == 300
    assert (
        report["frontier"]["frontier_maxima"]["max_return_edge_threshold_that_still_promotes"]
        == 0.005
    )
    assert (
        report["frontier"]["frontier_maxima"]["max_sharpe_edge_threshold_that_still_promotes"]
        == 2.5
    )
    assert report["frontier"]["frontier_maxima"]["max_min_val_return_that_still_promotes"] == 0.06
    assert report["frontier"]["frontier_maxima"]["max_min_val_sharpe_that_still_promotes"] == 3.0
