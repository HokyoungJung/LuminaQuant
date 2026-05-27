from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.research.write_exact_window_deployment_combo import _build_scenarios


def test_build_scenarios_falls_back_to_experimental_watchlist_without_strict_anchor(tmp_path: Path):
    details_path = tmp_path / "exact_window_candidate_details_latest.json"
    details_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "cand-topcap",
                    "name": "topcap_tsmom_1h_balanced_16_4_0.015",
                    "strategy_class": "TopCapTimeSeriesMomentumStrategy",
                    "family": "cross_sectional",
                    "strategy_timeframe": "1h",
                    "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
                    "train": {"return": -0.10, "sharpe": -0.4, "pbo": 0.125, "trade_count": 1200},
                    "val": {"return": 0.018, "sharpe": 1.64, "pbo": 0.25, "trade_count": 90},
                    "oos": {"return": 0.032, "sharpe": 1.46, "pbo": 0.375, "trade_count": 119},
                    "return_streams": {
                        "train": [{"t": 1, "v": 0.0}],
                        "val": [{"t": 2, "v": 0.018}],
                        "oos": [{"t": 3, "v": 0.032}],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    decision = {
        "timeframe_rows": [],
        "source_batches": [
            {
                "details_path": str(details_path),
            }
        ],
    }

    scenarios = _build_scenarios({}, decision, "2026-03-10T12:00:00+00:00")

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["scenario_id"] == "experimental_research_watchlist"
    assert scenario["components"][0]["name"] == "topcap_tsmom_1h_balanced_16_4_0.015"
    assert "research_only" in scenario["components"][0]["risk_flags"]


def test_experimental_watchlist_filters_sparse_1h_candidates(tmp_path: Path):
    details_path = tmp_path / "exact_window_candidate_details_latest.json"
    details_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "cand-sparse-1h",
                    "name": "pair_spread_1h_sparse_candidate",
                    "strategy_class": "PairSpreadZScoreStrategy",
                    "family": "market_neutral",
                    "strategy_timeframe": "1h",
                    "symbols": ["BTC/USDT", "XAG/USDT"],
                    "train": {"return": -0.05, "sharpe": -0.1, "pbo": 0.25, "trade_count": 40},
                    "val": {"return": 0.02, "sharpe": 1.8, "pbo": 0.25, "trade_count": 12},
                    "oos": {"return": 0.03, "sharpe": 1.6, "pbo": 0.25, "trade_count": 8},
                    "return_streams": {
                        "train": [{"t": 1, "v": 0.0}],
                        "val": [{"t": 2, "v": 0.02}],
                        "oos": [{"t": 3, "v": 0.03}],
                    },
                },
                {
                    "candidate_id": "cand-valid-1h",
                    "name": "topcap_tsmom_1h_valid_candidate",
                    "strategy_class": "TopCapTimeSeriesMomentumStrategy",
                    "family": "cross_sectional",
                    "strategy_timeframe": "1h",
                    "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
                    "train": {"return": -0.05, "sharpe": -0.1, "pbo": 0.25, "trade_count": 1200},
                    "val": {"return": 0.02, "sharpe": 1.8, "pbo": 0.25, "trade_count": 90},
                    "oos": {"return": 0.03, "sharpe": 1.6, "pbo": 0.25, "trade_count": 24},
                    "return_streams": {
                        "train": [{"t": 1, "v": 0.0}],
                        "val": [{"t": 2, "v": 0.02}],
                        "oos": [{"t": 3, "v": 0.03}],
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    decision = {
        "timeframe_rows": [],
        "source_batches": [
            {
                "details_path": str(details_path),
            }
        ],
    }

    scenarios = _build_scenarios({}, decision, "2026-03-11T12:30:00+00:00")

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["scenario_id"] == "experimental_research_watchlist"
    component_names = [component["name"] for component in scenario["components"]]
    assert "pair_spread_1h_sparse_candidate" not in component_names
    assert "topcap_tsmom_1h_valid_candidate" in component_names


def test_build_scenarios_carries_committee_and_final_decision_to_components(tmp_path: Path):
    details_path = tmp_path / "exact_window_candidate_details_latest.json"
    details_path.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "cand-anchored",
                    "name": "topcap_tsmom_1h_balanced_16_4_0.015",
                    "strategy_class": "TopCapTimeSeriesMomentumStrategy",
                    "family": "cross_sectional",
                    "strategy_timeframe": "1m",
                    "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"],
                    "train": {"return": 0.01, "sharpe": 1.1, "pbo": 0.25, "trade_count": 20},
                    "val": {"return": 0.02, "sharpe": 1.9, "pbo": 0.21, "trade_count": 45},
                    "oos": {"return": 0.03, "sharpe": 1.6, "pbo": 0.18, "trade_count": 12},
                    "promoted": True,
                    "committee": {
                        "technical_score": 0.71,
                        "support_score": 0.82,
                        "regime_score": 0.66,
                        "robustness_score": 0.73,
                        "risk_veto": False,
                        "risk_flags": ["oos_positive_skew_warning"],
                        "final_decision": "promote",
                    },
                    "return_streams": {
                        "train": [{"t": 1, "v": 0.0}],
                        "val": [{"t": 2, "v": 0.02}],
                        "oos": [{"t": 3, "v": 0.03}],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    decision = {
        "timeframe_rows": [],
        "source_batches": [
            {
                "details_path": str(details_path),
            }
        ],
    }

    scenarios = _build_scenarios({}, decision, "2026-03-12T12:00:00+00:00")
    assert len(scenarios) == 1
    scenario = scenarios[0]
    component = scenario["components"][0]
    assert component["final_decision"] == "promote"
    assert component["committee"]["final_decision"] == "promote"
    assert component["committee"]["risk_flags"] == ["oos_positive_skew_warning"]
    assert set(component["risk_flags"]) >= {
        "research_only",
        "no_strict_anchor",
        "oos_positive_skew_warning",
    }
