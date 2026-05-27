import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "analyze_hybrid_live_investability.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_hybrid_live_investability", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_report_flags_operational_gaps_even_when_structure_is_strong(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("live: {}\n", encoding="utf-8")

    refresh_path = tmp_path / "refresh.json"
    refresh_path.write_text(
        json.dumps({"status": "completed", "collection_cutoff_utc": "2026-04-17T00:00:00Z"}),
        encoding="utf-8",
    )
    decision_path = tmp_path / "decision.json"  # intentionally absent

    monkeypatch.setattr(
        MODULE,
        "load_runtime_config",
        lambda config_path: type(
            "RuntimeConfigStub",
            (),
            {
                "live": type(
                    "LiveStub",
                    (),
                    {
                        "mode": "paper",
                        "market_data_source": "committed",
                        "order_state_source": "polling",
                        "testnet": True,
                        "require_real_enable_flag": True,
                        "startup_reconciliation_hard_fail": False,
                        "exchange": type(
                            "ExchangeStub",
                            (),
                            {
                                "driver": "binance_futures",
                                "name": "binance",
                                "market_type": "future",
                            },
                        )(),
                    },
                )()
            },
        )(),
    )
    monkeypatch.setattr(
        MODULE,
        "datetime",
        type(
            "FrozenDateTime",
            (),
            {
                "now": staticmethod(lambda tz=None: MODULE._parse_utc("2026-04-17T00:15:00Z")),
                "fromisoformat": staticmethod(__import__("datetime").datetime.fromisoformat),
            },
        ),
    )

    switch_payload = {
        "recommended_mode": {
            "mode": "hybrid_guarded_mode",
            "allocation": {"hybrid_online_portfolio": 1.0},
        },
        "current_market_state": {
            "favored_group": "mixed",
            "confidence": 0.0,
            "trend_state": "bullish",
            "breadth_state": "broad",
            "volatility_state": "calm",
            "pair_liquidity_state": "normal",
        },
    }
    replay_payload = {
        "coverage_summary": {"coverage_gap_day_count": 18},
        "current_profile": {
            "min_oos_return_edge": 0.005,
            "min_oos_sharpe_edge": 2.5,
            "min_val_return": 0.06,
            "min_val_sharpe": 3.0,
        },
        "current_profile_result": {
            "oos_metrics": {"total_return": 0.0068, "sharpe": 3.4, "max_drawdown": 0.0024},
            "mode_counts": {"hybrid_guarded_mode": 5},
            "last_mode": "core_mode",
        },
        "strict_current_profile_result": {
            "oos_metrics": {"total_return": 0.0034, "sharpe": 1.66, "max_drawdown": 0.0027},
            "mode_counts": {"hybrid_guarded_mode": 2},
            "last_mode": "core_mode",
        },
    }
    hybrid_payload = {
        "config": {"pair_weight_cap": 0.25},
        "scenarios": {
            "refreshed_latest_tail": {
                "allocations": [
                    {
                        "date": "2026-02-28",
                        "split": "val",
                        "default_sleeve": "soft_three_way_regime",
                        "cash_weight": 0.10,
                        "weights": {"soft_three_way_regime": 0.90},
                    },
                    {
                        "date": "2026-03-01",
                        "split": "oos",
                        "default_sleeve": "soft_three_way_regime",
                        "cash_weight": 0.12,
                        "weights": {"soft_three_way_regime": 0.88},
                    },
                    {
                        "date": "2026-03-02",
                        "split": "oos",
                        "default_sleeve": "balanced_overlay_80_20",
                        "cash_weight": 0.12,
                        "weights": {
                            "soft_three_way_regime": 0.70,
                            "balanced_overlay_80_20": 0.05,
                            "pair_tactical_mode": 0.13,
                        },
                    },
                    {
                        "date": "2026-03-03",
                        "split": "oos",
                        "default_sleeve": "balanced_overlay_80_20",
                        "cash_weight": 0.15,
                        "weights": {
                            "soft_three_way_regime": 0.65,
                            "balanced_overlay_80_20": 0.05,
                            "pair_tactical_mode": 0.15,
                        },
                    },
                ],
                "daily_returns": [0.001, 0.002, 0.003, -0.001],
                "split_metrics": {
                    "oos": {"total_return": 0.0068, "sharpe": 3.23, "max_drawdown": 0.0026},
                },
            }
        },
    }

    report = MODULE.build_report(
        switch_payload=switch_payload,
        replay_payload=replay_payload,
        hybrid_payload=hybrid_payload,
        config_path=config_path,
        refresh_path=refresh_path,
        decision_path=decision_path,
        one_way_cost_bps_grid=[5.0, 10.0],
    )

    assert report["investability_verdict"]["strategy_structure_ready"] is True
    assert report["investability_verdict"]["real_ready_now"] is False
    assert report["investability_verdict"]["verdict"] == "research_ready_but_not_real_ready"
    assert "decision_artifact_missing" in report["live_readiness"]["gaps"]
    assert "startup_reconciliation_hard_fail_disabled" in report["live_readiness"]["gaps"]
    assert report["hybrid_oos"]["allocation_summary"]["max_pair_weight"] == 0.15
    assert len(report["hybrid_oos"]["cost_stress"]) == 2
    assert (
        report["hybrid_oos"]["cost_stress"][1]["adjusted_total_return"]
        < report["hybrid_oos"]["cost_stress"][0]["adjusted_total_return"]
    )
