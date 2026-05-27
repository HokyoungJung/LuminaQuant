import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "write_portfolio_master_scoreboard.py"
)
SPEC = importlib.util.spec_from_file_location("write_portfolio_master_scoreboard", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_master_scoreboard_promotes_hybrid_when_switch_does() -> None:
    switch_payload = {
        "as_of_date": "2026-04-14 00:00:00+00:00",
        "current_market_state": {
            "favored_group": "mixed",
            "confidence": 0.0,
            "trend_state": "bullish",
            "breadth_state": "broad",
            "volatility_state": "calm",
            "pair_liquidity_state": "normal",
        },
        "recommended_mode": {
            "mode": "hybrid_guarded_mode",
            "allocation": {"hybrid_online_portfolio": 1.0},
            "rationale": [
                "favored_group=mixed confidence=0.0000",
                "Mixed/calm regime and the guarded hybrid materially outperforms balanced -> promote hybrid guarded mode.",
            ],
        },
    }
    switch_validation = {
        "latest_common_complete_utc": "2026-04-14T10:59:16Z",
        "artifact_paths": {
            "refreshed_incumbent": "/tmp/incumbent.json",
            "refreshed_autoresearch": "/tmp/auto.json",
            "refreshed_blend": "/tmp/blend.json",
            "refreshed_soft_allocator": "/tmp/soft.json",
            "refreshed_three_way_allocator": "/tmp/hard.json",
            "refreshed_pair_candidate": "/tmp/pair.json",
            "refreshed_balanced_overlay": "/tmp/balanced.json",
        },
    }
    hybrid_payload = {
        "split_windows": {
            "train_start": "2025-01-01",
            "train_end_inclusive": "2025-12-31",
            "val_start": "2026-01-01",
            "val_end_inclusive": "2026-02-28",
            "oos_start": "2026-03-01",
            "oos_end_inclusive": "latest",
        },
        "online_policy": {
            "warmup_ratio": 0.6,
            "warmup_days": 255,
            "lookback_days": 18,
            "online_start": "2025-09-13",
        },
        "readiness": {
            "beats_balanced_refreshed": True,
            "beats_cash_refreshed": True,
            "beats_pair_tactical_refreshed": False,
            "max_rss_under_8gib": True,
            "pair_cap_respected": True,
            "recommended_stage": "guarded_candidate",
        },
        "scenarios": {
            "refreshed_latest_tail": {
                "split_metrics": {
                    "oos": {
                        "total_return": 0.004819,
                        "sharpe": 3.2304,
                        "max_drawdown": 0.001772,
                    }
                },
                "source_sleeve_metrics": {
                    "pair_tactical_mode": {
                        "oos": {
                            "total_return": 0.013392,
                            "sharpe": 2.1124,
                            "max_drawdown": 0.01137,
                        }
                    },
                    "balanced_overlay_80_20": {
                        "oos": {
                            "total_return": 0.002152,
                            "sharpe": 0.6995,
                            "max_drawdown": 0.006295,
                        }
                    },
                    "three_way_regime": {
                        "oos": {
                            "total_return": 0.001973,
                            "sharpe": 1.3087,
                            "max_drawdown": 0.007291,
                        }
                    },
                    "soft_three_way_regime": {
                        "oos": {
                            "total_return": 0.000887,
                            "sharpe": 0.7423,
                            "max_drawdown": 0.005734,
                        }
                    },
                },
                "comparison_rows": [
                    {
                        "name": "production_guarded_portfolio",
                        "total_return": 0.003654,
                        "sharpe": 1.6080,
                        "max_drawdown": 0.003184,
                    },
                    {
                        "name": "static_blend_76_24",
                        "total_return": 0.004307,
                        "sharpe": 3.0424,
                        "max_drawdown": 0.005106,
                    },
                    {
                        "name": "incumbent_only",
                        "total_return": -0.003154,
                        "sharpe": -2.5718,
                        "max_drawdown": 0.007291,
                    },
                ],
            }
        },
    }
    pair_candidate_payload = {
        "name": "pair_spread_1h_bridge_atr_bnbusdt_trxusdt_2.35_0.58",
    }

    scoreboard = MODULE.build_master_scoreboard(
        switch_payload=switch_payload,
        switch_validation=switch_validation,
        hybrid_payload=hybrid_payload,
        pair_candidate_payload=pair_candidate_payload,
    )
    onepager = MODULE.build_onepager_payload(scoreboard)

    assert scoreboard["current_default"]["mode"] == "hybrid_guarded_mode"
    assert scoreboard["hybrid_challenger"]["why_not_default"].startswith(
        "Promoted to the live default"
    )
    assert scoreboard["refreshed_live_scoreboard"][1]["status"] == "switch_default"
    assert {row["name"] for row in scoreboard["refreshed_live_scoreboard"]} >= {
        "production_guarded_mode",
        "static_blend_76_24",
        "incumbent_only",
    }
    assert [row["name"] for row in scoreboard["benchmark_reference_rows"]] == [
        "static_blend_76_24",
        "incumbent_only",
    ]
    assert [row["name"] for row in onepager["benchmark_reference_rows"]] == [
        "static_blend_76_24",
        "incumbent_only",
    ]
    assert onepager["default_live_mode"]["mode"] == "hybrid_guarded_mode"


def test_build_master_scoreboard_keeps_benchmark_rows_from_source_sleeve_metrics() -> None:
    switch_payload = {
        "as_of_date": "2026-04-14 00:00:00+00:00",
        "current_market_state": {
            "favored_group": "mixed",
            "confidence": 0.0,
            "trend_state": "bullish",
            "breadth_state": "broad",
            "volatility_state": "calm",
            "pair_liquidity_state": "normal",
        },
        "recommended_mode": {
            "mode": "balanced_overlay_mode",
            "allocation": {"soft_three_way_regime": 0.8, "pair_tactical_mode": 0.2},
            "rationale": ["Keep balanced as the switch default."],
        },
    }
    switch_validation = {
        "latest_common_complete_utc": "2026-04-14T10:59:16Z",
        "artifact_paths": {
            "refreshed_incumbent": "/tmp/incumbent.json",
            "refreshed_autoresearch": "/tmp/auto.json",
            "refreshed_blend": "/tmp/blend.json",
            "refreshed_soft_allocator": "/tmp/soft.json",
            "refreshed_three_way_allocator": "/tmp/hard.json",
            "refreshed_pair_candidate": "/tmp/pair.json",
            "refreshed_balanced_overlay": "/tmp/balanced.json",
        },
    }
    hybrid_payload = {
        "split_windows": {
            "train_start": "2025-01-01",
            "train_end_inclusive": "2025-12-31",
            "val_start": "2026-01-01",
            "val_end_inclusive": "2026-02-28",
            "oos_start": "2026-03-01",
            "oos_end_inclusive": "latest",
        },
        "online_policy": {
            "warmup_ratio": 0.6,
            "warmup_days": 255,
            "lookback_days": 18,
            "online_start": "2025-09-13",
        },
        "readiness": {
            "recommended_stage": "guarded_candidate",
        },
        "scenarios": {
            "refreshed_latest_tail": {
                "split_metrics": {
                    "oos": {
                        "total_return": 0.004819,
                        "sharpe": 3.2304,
                        "max_drawdown": 0.001772,
                    }
                },
                "source_sleeve_metrics": {
                    "pair_tactical_mode": {
                        "oos": {
                            "total_return": 0.013392,
                            "sharpe": 2.1124,
                            "max_drawdown": 0.01137,
                        }
                    },
                    "balanced_overlay_80_20": {
                        "oos": {
                            "total_return": 0.002152,
                            "sharpe": 0.6995,
                            "max_drawdown": 0.006295,
                        }
                    },
                    "three_way_regime": {
                        "oos": {
                            "total_return": 0.001973,
                            "sharpe": 1.3087,
                            "max_drawdown": 0.007291,
                        }
                    },
                    "soft_three_way_regime": {
                        "oos": {
                            "total_return": 0.000887,
                            "sharpe": 0.7423,
                            "max_drawdown": 0.005734,
                        }
                    },
                    "static_blend_76_24": {
                        "oos": {
                            "total_return": 0.004307,
                            "sharpe": 3.0424,
                            "max_drawdown": 0.005106,
                        }
                    },
                    "incumbent_only": {
                        "oos": {
                            "total_return": -0.003154,
                            "sharpe": -2.5718,
                            "max_drawdown": 0.007291,
                        }
                    },
                },
                "comparison_rows": [
                    {
                        "name": "production_guarded_portfolio",
                        "total_return": 0.003654,
                        "sharpe": 1.6080,
                        "max_drawdown": 0.003184,
                    },
                    {
                        "name": "pair_tactical_mode",
                        "total_return": 0.013392,
                        "sharpe": 2.1124,
                        "max_drawdown": 0.01137,
                    },
                    {
                        "name": "balanced_overlay_80_20",
                        "total_return": 0.002152,
                        "sharpe": 0.6995,
                        "max_drawdown": 0.006295,
                    },
                    {
                        "name": "three_way_regime",
                        "total_return": 0.001973,
                        "sharpe": 1.3087,
                        "max_drawdown": 0.007291,
                    },
                    {
                        "name": "soft_three_way_regime",
                        "total_return": 0.000887,
                        "sharpe": 0.7423,
                        "max_drawdown": 0.005734,
                    },
                ],
            }
        },
    }
    pair_candidate_payload = {"name": "pair_spread_candidate"}

    scoreboard = MODULE.build_master_scoreboard(
        switch_payload=switch_payload,
        switch_validation=switch_validation,
        hybrid_payload=hybrid_payload,
        pair_candidate_payload=pair_candidate_payload,
    )
    master_markdown = MODULE._build_master_markdown(scoreboard)
    onepager_markdown = MODULE._build_onepager_markdown(MODULE.build_onepager_payload(scoreboard))

    assert {row["name"] for row in scoreboard["refreshed_live_scoreboard"]} >= {
        "production_guarded_mode",
        "static_blend_76_24",
        "incumbent_only",
    }
    static_row = next(
        row for row in scoreboard["benchmark_reference_rows"] if row["name"] == "static_blend_76_24"
    )
    incumbent_row = next(
        row for row in scoreboard["benchmark_reference_rows"] if row["name"] == "incumbent_only"
    )
    assert static_row["oos_total_return"] == 0.004307
    assert incumbent_row["oos_total_return"] == -0.003154
    assert "Benchmark anchors" in master_markdown
    assert "benchmark anchor `static_blend_76_24`" in onepager_markdown
