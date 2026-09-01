import importlib.util
import sys
from pathlib import Path

import pandas as pd

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "replay_performance_first_switch.py"
)
SPEC = importlib.util.spec_from_file_location("replay_performance_first_switch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_replay_report_prefers_hybrid_profile_when_override_is_decisive(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "_market_judgements_by_day",
        lambda **kwargs: {
            "2026-03-01": {
                "date": "2026-03-01T00:00:00+00:00",
                "favored_group": "mixed",
                "confidence": 0.0,
                "feature_snapshot": {
                    "btc_above_ma192": True,
                    "btc_above_ma336": True,
                    "btc_trend_gap_192": 0.02,
                    "btc_trend_gap_336": 0.03,
                    "btc_trend_accel": 0.01,
                    "breadth_ma96": 1.0,
                    "breadth_ma192": 1.0,
                    "breadth_delta": 0.05,
                    "basket_vol_ratio": 0.45,
                },
            },
            "2026-03-02": {
                "date": "2026-03-02T00:00:00+00:00",
                "favored_group": "mixed",
                "confidence": 0.0,
                "feature_snapshot": {
                    "btc_above_ma192": True,
                    "btc_above_ma336": True,
                    "btc_trend_gap_192": 0.02,
                    "btc_trend_gap_336": 0.03,
                    "btc_trend_accel": 0.01,
                    "breadth_ma96": 1.0,
                    "breadth_ma192": 1.0,
                    "breadth_delta": 0.05,
                    "basket_vol_ratio": 0.45,
                },
            },
            "2026-03-03": {
                "date": "2026-03-03T00:00:00+00:00",
                "favored_group": "mixed",
                "confidence": 0.0,
                "feature_snapshot": {
                    "btc_above_ma192": True,
                    "btc_above_ma336": True,
                    "btc_trend_gap_192": 0.02,
                    "btc_trend_gap_336": 0.03,
                    "btc_trend_accel": 0.01,
                    "breadth_ma96": 1.0,
                    "breadth_ma192": 1.0,
                    "breadth_delta": 0.05,
                    "basket_vol_ratio": 0.45,
                },
            },
            "2026-03-04": {
                "date": "2026-03-04T00:00:00+00:00",
                "favored_group": "mixed",
                "confidence": 0.0,
                "feature_snapshot": {
                    "btc_above_ma192": True,
                    "btc_above_ma336": True,
                    "btc_trend_gap_192": 0.02,
                    "btc_trend_gap_336": 0.03,
                    "btc_trend_accel": 0.01,
                    "breadth_ma96": 1.0,
                    "breadth_ma192": 1.0,
                    "breadth_delta": 0.05,
                    "basket_vol_ratio": 0.45,
                },
            },
        },
    )
    monkeypatch.setattr(
        MODULE._SWITCH,
        "_load_symbol_volume_signal",
        lambda **kwargs: MODULE._SWITCH.SymbolVolumeSignal(
            symbol=str(kwargs["symbol"]),
            as_of_date=str(kwargs["as_of_date"]),
            latest_available_date=str(kwargs["as_of_date"]),
            stale_days=0,
            latest_dollar_volume=1000.0,
            lookback_mean_dollar_volume=800.0,
            volume_ratio=1.25,
            comparison_mode="same_time_of_day",
            state="strong" if "BNB" in str(kwargs["symbol"]) else "normal",
        ),
    )

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
    market_payload = {"selected_rules": [{"rule_id": "x"}], "symbol_universe": ["BTC/USDT"]}
    soft_payload = {
        "states": [
            {
                "date": "2026-03-01T00:00:00+00:00",
                "effective_incumbent_exposure": 0.95,
                "effective_autoresearch_exposure": 0.05,
                "state": "blend_85_15",
                "raw_target_state": "incumbent",
            }
        ],
        "dates": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"],
        "daily_returns": [0.0005, 0.0004, 0.0006, 0.0005],
    }
    hard_payload = {
        "states": [
            {
                "date": "2026-03-01T00:00:00+00:00",
                "state": "blend_85_15",
                "raw_target_state": "incumbent",
            }
        ],
        "dates": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"],
        "daily_returns": [0.0007, 0.0006, 0.0008, 0.0007],
    }
    balanced_payload = {
        "portfolio_metrics": {"val": {"total_return": 0.08, "sharpe": 4.1}},
        "portfolio_return_streams": {
            "oos": [
                {"datetime": "2026-03-01T00:00:00+00:00", "v": 0.0020},
                {"datetime": "2026-03-02T00:00:00+00:00", "v": -0.0010},
                {"datetime": "2026-03-03T00:00:00+00:00", "v": 0.0020},
                {"datetime": "2026-03-04T00:00:00+00:00", "v": -0.0010},
            ]
        },
    }
    pair_payload = {
        "return_streams": {
            "oos": [
                {"datetime": "2026-03-01T00:00:00+00:00", "v": 0.0010},
                {"datetime": "2026-03-02T00:00:00+00:00", "v": -0.0008},
                {"datetime": "2026-03-03T00:00:00+00:00", "v": 0.0009},
                {"datetime": "2026-03-04T00:00:00+00:00", "v": -0.0007},
            ]
        }
    }
    hybrid_payload = {
        "readiness": {
            "beats_balanced_refreshed": True,
            "beats_pair_tactical_refreshed": True,
            "pair_cap_respected": True,
            "recommended_stage": "pilot_candidate",
        },
        "scenarios": {
            "refreshed_latest_tail": {
                "dates": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"],
                "daily_returns": [0.004, 0.003, 0.004, 0.003],
                "allocations": [
                    {"date": "2026-03-01", "split": "oos"},
                    {"date": "2026-03-02", "split": "oos"},
                    {"date": "2026-03-03", "split": "oos"},
                    {"date": "2026-03-04", "split": "oos"},
                ],
                "split_metrics": {
                    "val": {"total_return": 0.065, "sharpe": 3.3},
                    "oos": {"total_return": 0.006, "sharpe": 3.2, "max_drawdown": 0.0025},
                },
                "source_sleeve_metrics": {
                    "balanced_overlay_80_20": {
                        "val": {"total_return": 0.08, "sharpe": 4.1},
                        "oos": {"total_return": 0.002, "sharpe": 0.5, "max_drawdown": 0.005},
                    }
                },
            }
        },
    }

    report = MODULE.build_replay_report(
        switch_payload=switch_payload,
        market_payload=market_payload,
        soft_payload=soft_payload,
        hard_payload=hard_payload,
        balanced_payload=balanced_payload,
        pair_payload=pair_payload,
        hybrid_payload=hybrid_payload,
        return_grid=[0.004],
        sharpe_grid=[2.0],
        val_return_grid=[0.05],
        val_sharpe_grid=[3.0],
    )

    current = report["current_profile_result"]
    assert report["current_switch_mode"] == "hybrid_guarded_mode"
    assert report["coverage_summary"]["market_judgement_missing_days"] == 0
    assert "strict_current_profile_result" in report
    assert current["last_mode"] == "hybrid_guarded_mode"
    assert current["mode_counts"]["hybrid_guarded_mode"] >= 1
    assert current["oos_metrics"]["total_return"] > 0.0


def test_market_judgements_by_day_normalizes_timestamp_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE._MARKET,
        "_load_symbol_close_30m_from_feature_points",
        lambda *args, **kwargs: (pd.DataFrame({"x": [1]}), {}),
    )
    monkeypatch.setattr(
        MODULE._MARKET,
        "_daily_market_feature_frame",
        lambda frames: pd.DataFrame({"date": [pd.Timestamp("2026-03-01", tz="UTC")]}),
    )
    monkeypatch.setattr(
        MODULE._MARKET,
        "_current_judgement",
        lambda **kwargs: {
            "date": "2026-03-01 00:00:00+00:00",
            "favored_group": "mixed",
            "confidence": 0.0,
            "feature_snapshot": {},
        },
    )

    judgements = MODULE._market_judgements_by_day(
        market_payload={"selected_rules": [{"rule_id": "x"}], "symbol_universe": ["BTC/USDT"]},
        replay_days=["2026-03-01"],
    )

    assert "2026-03-01" in judgements


def test_adjust_pair_liquidity_state_for_coverage_neutralizes_future_coverage_gap() -> None:
    signals = [
        MODULE._SWITCH.SymbolVolumeSignal(
            symbol="BNB/USDT",
            as_of_date="2026-03-01",
            latest_available_date="2025-01-01",
            stale_days=424,
            latest_dollar_volume=1.0,
            lookback_mean_dollar_volume=1.0,
            volume_ratio=1.0,
            comparison_mode="full_day",
            state="stale",
        ),
        MODULE._SWITCH.SymbolVolumeSignal(
            symbol="TRX/USDT",
            as_of_date="2026-03-01",
            latest_available_date="2025-01-01",
            stale_days=424,
            latest_dollar_volume=1.0,
            lookback_mean_dollar_volume=1.0,
            volume_ratio=1.0,
            comparison_mode="full_day",
            state="stale",
        ),
    ]

    state, symbols = MODULE._adjust_pair_liquidity_state_for_coverage(
        signals=signals,
        as_of_date=MODULE.date(2026, 3, 1),
        coverage_index={
            "BNB/USDT": [MODULE.date(2025, 1, 1), MODULE.date(2026, 3, 19)],
            "TRX/USDT": [MODULE.date(2025, 1, 1), MODULE.date(2026, 3, 19)],
        },
    )

    assert state == "normal"
    assert symbols == ["BNB/USDT", "TRX/USDT"]
