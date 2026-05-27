import importlib.util
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "write_portfolio_operating_switch.py"
)
SPEC = importlib.util.spec_from_file_location("write_portfolio_operating_switch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _base_operating_plan() -> dict:
    return {
        "deployment_modes": {
            "core_mode": {"allocation": {"soft_three_way_regime": 1.0}},
            "balanced_overlay_mode": {
                "allocation": {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2},
                "metrics": {"oos_total_return": 0.02, "oos_sharpe": 1.0, "oos_max_drawdown": 0.01},
            },
            "defensive_overlay_mode": {
                "allocation": {"soft_three_way_regime": 0.7, "pair_fast_exit": 0.3}
            },
            "aggressive_realized_mode": {"allocation": {"three_way_regime": 1.0}},
            "hybrid_guarded_mode": {"allocation": {"hybrid_online_portfolio": 1.0}},
            "production_guarded_mode": {"allocation": {"production_guarded_portfolio": 1.0}},
        }
    }


def _ts(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp() * 1000)


def test_load_symbol_volume_signal_classifies_strong_volume(tmp_path: Path) -> None:
    date_dir = tmp_path / "BNBUSDT" / "date=2026-03-07"
    date_dir.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "timestamp_ms": [
                1772841600000,
                1772845200000,
                1772755200000,
                1772668800000,
            ],
            "price": [100.0, 100.0, 100.0, 100.0],
            "quantity": [30.0, 30.0, 10.0, 10.0],
            "agg_trade_id": [1, 2, 3, 4],
            "is_buyer_maker": [True, True, False, False],
        }
    )
    frame.write_parquet(date_dir / "part-0000.parquet")

    signal = MODULE._load_symbol_volume_signal(
        raw_aggtrades_root=tmp_path,
        symbol="BNB/USDT",
        as_of_date=MODULE._parse_as_of_date("2026-03-07T00:00:00+00:00"),
        lookback_days=7,
    )

    assert signal.symbol == "BNB/USDT"
    assert signal.latest_available_date == "2026-03-07"
    assert signal.volume_ratio is not None and signal.volume_ratio > 1.5
    assert signal.comparison_mode == "same_time_of_day"
    assert signal.state == "strong"


def test_load_symbol_volume_signal_uses_same_time_baseline_for_partial_current_day(
    tmp_path: Path,
) -> None:
    symbol_root = tmp_path / "BNBUSDT"
    daily_payloads = {
        "2026-03-05": {
            "timestamp_ms": [_ts(2026, 3, 5, 8), _ts(2026, 3, 5, 10), _ts(2026, 3, 5, 22)],
            "price": [100.0, 100.0, 100.0],
            "quantity": [0.3, 0.3, 0.4],
        },
        "2026-03-06": {
            "timestamp_ms": [_ts(2026, 3, 6, 8), _ts(2026, 3, 6, 10), _ts(2026, 3, 6, 22)],
            "price": [100.0, 100.0, 100.0],
            "quantity": [0.3, 0.3, 0.4],
        },
        "2026-03-07": {
            "timestamp_ms": [_ts(2026, 3, 7, 8), _ts(2026, 3, 7, 10)],
            "price": [100.0, 100.0],
            "quantity": [0.35, 0.35],
        },
    }
    for day, payload in daily_payloads.items():
        date_dir = symbol_root / f"date={day}"
        date_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                **payload,
                "agg_trade_id": list(range(1, len(payload["timestamp_ms"]) + 1)),
                "is_buyer_maker": [True] * len(payload["timestamp_ms"]),
            }
        ).write_parquet(date_dir / "part-0000.parquet")

    signal = MODULE._load_symbol_volume_signal(
        raw_aggtrades_root=tmp_path,
        symbol="BNB/USDT",
        as_of_date=MODULE._parse_as_of_date("2026-03-07T10:00:00+00:00"),
        lookback_days=7,
    )

    assert signal.comparison_mode == "same_time_of_day"
    assert signal.volume_ratio is not None and signal.volume_ratio > 1.1
    assert signal.state == "strong"


def test_recommend_operating_mode_prefers_balanced_overlay_for_current_like_state() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.73,
            "feature_snapshot": {
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "btc_trend_gap_192": -0.03,
                "btc_trend_gap_336": -0.01,
                "btc_trend_accel": 0.02,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "breadth_delta": 0.0,
                "basket_vol_ratio": 0.45,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.80,
            "effective_autoresearch_exposure": 0.20,
        },
        hard_current_state={
            "state": "autoresearch_55_45",
            "raw_target_state": "incumbent",
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="strong",
    )

    assert decision.mode == "balanced_overlay_mode"
    assert decision.allocation == {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}


def test_recommend_operating_mode_falls_back_to_core_when_pair_liquidity_is_weak() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.80,
            "feature_snapshot": {
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "btc_trend_gap_192": -0.05,
                "btc_trend_gap_336": -0.03,
                "btc_trend_accel": -0.01,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "breadth_delta": -0.1,
                "basket_vol_ratio": 0.90,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.85,
            "effective_autoresearch_exposure": 0.15,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.01, "oos_sharpe": 0.5},
        },
        hard_current_state={"state": "incumbent", "raw_target_state": "incumbent"},
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="weak",
    )

    assert decision.mode == "core_mode"
    assert decision.allocation == {"soft_three_way_regime": 1.0}


def test_recommend_operating_mode_can_select_aggressive_realized_mode() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "autoresearch",
            "confidence": 0.80,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.04,
                "btc_trend_gap_336": 0.03,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.7,
                "breadth_ma192": 0.6,
                "breadth_delta": 0.1,
                "basket_vol_ratio": 0.95,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.40,
            "effective_autoresearch_exposure": 0.60,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.05, "oos_sharpe": 2.0},
        },
        hard_current_state={
            "state": "autoresearch_55_45",
            "raw_target_state": "autoresearch_55_45",
            "_allocator_health": {"healthy": True, "oos_total_return": 0.04, "oos_sharpe": 1.8},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="strong",
    )

    assert decision.mode == "aggressive_realized_mode"
    assert decision.allocation == {"three_way_regime": 1.0}


def test_recommend_operating_mode_blocks_aggressive_when_hard_allocator_is_unhealthy() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "autoresearch",
            "confidence": 0.90,
            "feature_snapshot": {
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "btc_trend_gap_192": -0.02,
                "btc_trend_gap_336": -0.01,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.4,
                "breadth_ma192": 0.2,
                "breadth_delta": 0.2,
                "basket_vol_ratio": 0.8,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.60,
            "effective_autoresearch_exposure": 0.40,
            "_allocator_health": {"healthy": False, "oos_total_return": -0.01, "oos_sharpe": -1.0},
        },
        hard_current_state={
            "state": "autoresearch_55_45",
            "raw_target_state": "autoresearch_55_45",
            "_allocator_health": {"healthy": False, "oos_total_return": -0.05, "oos_sharpe": -3.0},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="normal",
    )

    assert decision.mode == "balanced_overlay_mode"
    assert decision.allocation == {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}


def test_recommend_operating_mode_uses_risk_off_when_all_active_sleeves_are_unhealthy() -> None:
    plan = _base_operating_plan()
    plan["deployment_modes"]["balanced_overlay_mode"]["metrics"] = {
        "oos_total_return": -0.01,
        "oos_sharpe": -1.0,
        "oos_max_drawdown": 0.02,
    }
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "autoresearch",
            "confidence": 0.90,
            "feature_snapshot": {
                "btc_above_ma192": False,
                "btc_above_ma336": False,
                "btc_trend_gap_192": -0.02,
                "btc_trend_gap_336": -0.01,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.2,
                "breadth_ma192": 0.2,
                "breadth_delta": 0.0,
                "basket_vol_ratio": 0.8,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.60,
            "effective_autoresearch_exposure": 0.40,
            "_allocator_health": {"healthy": False, "oos_total_return": -0.01, "oos_sharpe": -1.0},
        },
        hard_current_state={
            "state": "autoresearch_55_45",
            "raw_target_state": "autoresearch_55_45",
            "_allocator_health": {"healthy": False, "oos_total_return": -0.05, "oos_sharpe": -3.0},
        },
        operating_plan_payload=plan,
        pair_liquidity_state="normal",
    )

    assert decision.mode == "risk_off_mode"
    assert decision.allocation == {"cash": 1.0}


def test_balanced_strategy_health_prefers_hybrid_source_metrics() -> None:
    health = MODULE._balanced_strategy_health(
        operating_plan_payload=_base_operating_plan(),
        balanced_strategy_payload={
            "portfolio_metrics": {
                "val": {
                    "total_return": -0.02,
                    "sharpe": -1.0,
                    "max_drawdown": 0.03,
                    "volatility": 0.01,
                },
                "oos": {"total_return": -0.01, "sharpe": -1.0, "max_drawdown": 0.02},
            }
        },
        hybrid_source_metrics={
            "balanced_overlay_80_20": {
                "val": {"total_return": 0.01, "sharpe": 1.0},
                "oos": {"total_return": 0.02, "sharpe": 1.5, "max_drawdown": 0.01},
            }
        },
    )

    assert health["healthy"] is True
    assert health["oos_total_return"] == 0.02
    assert health["oos_sharpe"] == 1.5


def test_recommend_operating_mode_uses_risk_off_even_in_bullish_state_when_all_active_sleeves_are_unhealthy() -> (
    None
):
    plan = _base_operating_plan()
    plan["deployment_modes"]["balanced_overlay_mode"]["metrics"] = {
        "oos_total_return": -0.01,
        "oos_sharpe": -1.0,
        "oos_max_drawdown": 0.02,
    }
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.95,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.04,
                "btc_trend_gap_336": 0.03,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 1.0,
                "breadth_ma192": 1.0,
                "breadth_delta": 0.2,
                "basket_vol_ratio": 0.5,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.85,
            "effective_autoresearch_exposure": 0.15,
            "_allocator_health": {"healthy": False, "oos_total_return": -0.02, "oos_sharpe": -2.0},
        },
        hard_current_state={
            "state": "incumbent",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": False, "oos_total_return": -0.03, "oos_sharpe": -3.0},
        },
        operating_plan_payload=plan,
        pair_liquidity_state="normal",
    )

    assert decision.mode == "risk_off_mode"
    assert decision.allocation == {"cash": 1.0}


def test_recommend_operating_mode_prefers_hybrid_guarded_when_legacy_sleeves_are_unhealthy_but_hybrid_is_healthy() -> (
    None
):
    plan = _base_operating_plan()
    plan["deployment_modes"]["balanced_overlay_mode"]["metrics"] = {
        "oos_total_return": -0.01,
        "oos_sharpe": -1.0,
        "oos_max_drawdown": 0.02,
    }
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.95,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.03,
                "btc_trend_gap_336": 0.02,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.9,
                "breadth_ma192": 0.8,
                "breadth_delta": 0.2,
                "basket_vol_ratio": 0.6,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.85,
            "effective_autoresearch_exposure": 0.15,
            "_allocator_health": {"healthy": False, "oos_total_return": -0.02, "oos_sharpe": -2.0},
        },
        hard_current_state={
            "state": "incumbent",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": False, "oos_total_return": -0.03, "oos_sharpe": -3.0},
        },
        operating_plan_payload=plan,
        pair_liquidity_state="normal",
        hybrid_health={
            "healthy": True,
            "recommended_stage": "guarded_candidate",
            "oos_total_return": 0.02,
            "oos_sharpe": 2.0,
            "beats_cash_refreshed": True,
            "pair_cap_respected": True,
            "max_rss_under_8gib": True,
        },
    )

    assert decision.mode == "hybrid_guarded_mode"
    assert decision.allocation == {"hybrid_online_portfolio": 1.0}


def test_recommend_operating_mode_promotes_hybrid_guarded_in_mixed_calm_when_it_materially_beats_balanced() -> (
    None
):
    decision = MODULE.recommend_operating_mode(
        current_judgement={
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
        soft_current_state={
            "effective_incumbent_exposure": 0.95,
            "effective_autoresearch_exposure": 0.05,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.0009, "oos_sharpe": 0.74},
        },
        hard_current_state={
            "state": "blend_85_15",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": True, "oos_total_return": 0.0020, "oos_sharpe": 1.31},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="normal",
        balanced_health={
            "healthy": True,
            "val_total_return": 0.0841,
            "val_sharpe": 4.18,
            "oos_total_return": 0.00215,
            "oos_sharpe": 0.70,
            "oos_max_drawdown": 0.00629,
        },
        hybrid_health={
            "healthy": True,
            "recommended_stage": "guarded_candidate",
            "beats_balanced_refreshed": True,
            "oos_total_return": 0.00482,
            "oos_sharpe": 3.23,
            "oos_max_drawdown": 0.00177,
            "val_total_return": 0.07826,
            "val_sharpe": 3.99,
        },
    )

    assert decision.mode == "hybrid_guarded_mode"
    assert decision.allocation == {"hybrid_online_portfolio": 1.0}
    assert any("materially outperforms balanced" in item for item in decision.rationale)


def test_recommend_operating_mode_can_promote_production_guarded_in_mixed_calm() -> None:
    decision = MODULE.recommend_operating_mode(
        current_judgement={
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
        soft_current_state={
            "effective_incumbent_exposure": 0.95,
            "effective_autoresearch_exposure": 0.05,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.0009, "oos_sharpe": 0.74},
        },
        hard_current_state={
            "state": "blend_85_15",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": True, "oos_total_return": 0.0020, "oos_sharpe": 1.31},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="normal",
        balanced_health={
            "healthy": True,
            "val_total_return": 0.03,
            "val_sharpe": 1.0,
            "oos_total_return": 0.00109,
            "oos_sharpe": 0.4828,
            "oos_max_drawdown": 0.005162,
        },
        hybrid_health={
            "healthy": False,
            "recommended_stage": "do_not_integrate",
            "beats_balanced_refreshed": False,
            "oos_total_return": -0.001,
            "oos_sharpe": -0.2,
            "oos_max_drawdown": 0.006,
            "val_total_return": -0.002,
            "val_sharpe": -0.5,
        },
        production_health={
            "healthy": True,
            "val_total_return": 0.04,
            "val_sharpe": 1.5,
            "oos_total_return": 0.00365,
            "oos_sharpe": 1.61,
            "oos_max_drawdown": 0.00318,
        },
    )

    assert decision.mode == "production_guarded_mode"
    assert decision.allocation == {"production_guarded_portfolio": 1.0}
    assert any("production-guarded candidate beats balanced" in item for item in decision.rationale)


def test_recommend_operating_mode_uses_performance_first_override_when_hybrid_oos_edge_is_decisive() -> (
    None
):
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "mixed",
            "confidence": 0.0,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.0236,
                "btc_trend_gap_336": 0.0288,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 1.0,
                "breadth_ma192": 1.0,
                "breadth_delta": 0.05,
                "basket_vol_ratio": 0.46,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.95,
            "effective_autoresearch_exposure": 0.05,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.0009, "oos_sharpe": 0.74},
        },
        hard_current_state={
            "state": "blend_85_15",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": True, "oos_total_return": 0.0020, "oos_sharpe": 1.31},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="normal",
        balanced_health={
            "healthy": True,
            "val_total_return": 0.08308,
            "val_sharpe": 4.1120,
            "oos_total_return": 0.00109,
            "oos_sharpe": 0.4828,
            "oos_max_drawdown": 0.005162,
        },
        hybrid_health={
            "healthy": True,
            "recommended_stage": "pilot_candidate",
            "beats_balanced_refreshed": True,
            "beats_pair_tactical_refreshed": True,
            "val_total_return": 0.06537,
            "val_sharpe": 3.2857,
            "oos_total_return": 0.006868,
            "oos_sharpe": 3.2370,
            "oos_max_drawdown": 0.002573,
        },
    )

    assert decision.mode == "hybrid_guarded_mode"
    assert decision.allocation == {"hybrid_online_portfolio": 1.0}
    assert any("performance-first override" in item for item in decision.rationale)


def test_recommend_operating_mode_requires_decisive_oos_edge_for_performance_first_override() -> (
    None
):
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "mixed",
            "confidence": 0.0,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.0236,
                "btc_trend_gap_336": 0.0288,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 1.0,
                "breadth_ma192": 1.0,
                "breadth_delta": 0.05,
                "basket_vol_ratio": 0.46,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.95,
            "effective_autoresearch_exposure": 0.05,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.0009, "oos_sharpe": 0.74},
        },
        hard_current_state={
            "state": "blend_85_15",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": True, "oos_total_return": 0.0020, "oos_sharpe": 1.31},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="normal",
        balanced_health={
            "healthy": True,
            "val_total_return": 0.08308,
            "val_sharpe": 4.1120,
            "oos_total_return": 0.0040,
            "oos_sharpe": 1.5,
            "oos_max_drawdown": 0.005162,
        },
        hybrid_health={
            "healthy": True,
            "recommended_stage": "pilot_candidate",
            "beats_balanced_refreshed": True,
            "beats_pair_tactical_refreshed": True,
            "val_total_return": 0.06537,
            "val_sharpe": 3.2857,
            "oos_total_return": 0.006868,
            "oos_sharpe": 3.2370,
            "oos_max_drawdown": 0.002573,
        },
    )

    assert decision.mode == "balanced_overlay_mode"
    assert decision.allocation == {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}


def test_recommend_operating_mode_keeps_balanced_overlay_when_incumbent_is_healthy_and_hybrid_is_only_a_challenger() -> (
    None
):
    decision = MODULE.recommend_operating_mode(
        current_judgement={
            "favored_group": "incumbent",
            "confidence": 0.82,
            "feature_snapshot": {
                "btc_above_ma192": True,
                "btc_above_ma336": True,
                "btc_trend_gap_192": 0.03,
                "btc_trend_gap_336": 0.02,
                "btc_trend_accel": 0.01,
                "breadth_ma96": 0.90,
                "breadth_ma192": 0.85,
                "breadth_delta": 0.10,
                "basket_vol_ratio": 0.60,
            },
        },
        soft_current_state={
            "effective_incumbent_exposure": 0.85,
            "effective_autoresearch_exposure": 0.15,
            "_allocator_health": {"healthy": True, "oos_total_return": 0.01, "oos_sharpe": 0.8},
        },
        hard_current_state={
            "state": "incumbent",
            "raw_target_state": "incumbent",
            "_allocator_health": {"healthy": True, "oos_total_return": 0.015, "oos_sharpe": 1.1},
        },
        operating_plan_payload=_base_operating_plan(),
        pair_liquidity_state="normal",
        balanced_health={
            "healthy": True,
            "val_total_return": 0.05,
            "val_sharpe": 2.0,
            "oos_total_return": 0.02,
            "oos_sharpe": 1.0,
            "oos_max_drawdown": 0.01,
        },
        hybrid_health={
            "healthy": True,
            "recommended_stage": "guarded_candidate",
            "beats_balanced_refreshed": True,
            "val_total_return": 0.03,
            "val_sharpe": 1.7,
            "oos_total_return": 0.0205,
            "oos_sharpe": 1.3,
            "oos_max_drawdown": 0.0095,
        },
    )

    assert decision.mode == "balanced_overlay_mode"
    assert decision.allocation == {"soft_three_way_regime": 0.8, "pair_fast_exit": 0.2}


def test_build_operating_switch_payload_includes_market_states() -> None:
    payload = MODULE.build_operating_switch_payload(
        market_judgement_payload={
            "_path": "/tmp/market.json",
            "current_judgement": {
                "date": "2026-03-07T00:00:00+00:00",
                "favored_group": "incumbent",
                "confidence": 0.73,
                "feature_snapshot": {
                    "btc_above_ma192": False,
                    "btc_above_ma336": False,
                    "btc_trend_gap_192": -0.03,
                    "btc_trend_gap_336": -0.01,
                    "btc_trend_accel": 0.02,
                    "breadth_ma96": 0.2,
                    "breadth_ma192": 0.2,
                    "breadth_delta": 0.0,
                    "basket_vol_ratio": 0.45,
                },
            },
        },
        soft_allocator_payload={
            "_path": "/tmp/soft.json",
            "current_state": {
                "effective_incumbent_exposure": 0.8,
                "effective_autoresearch_exposure": 0.2,
            },
        },
        three_way_allocator_payload={
            "_path": "/tmp/hard.json",
            "current_state": {"state": "autoresearch_55_45", "raw_target_state": "incumbent"},
        },
        operating_plan_payload={"_path": "/tmp/plan.json", **_base_operating_plan()},
        pair_volume_signals=[
            MODULE.SymbolVolumeSignal(
                symbol="BNB/USDT",
                as_of_date="2026-03-07",
                latest_available_date="2026-03-07",
                stale_days=0,
                latest_dollar_volume=1000.0,
                lookback_mean_dollar_volume=800.0,
                volume_ratio=1.25,
                comparison_mode="full_day",
                state="strong",
            )
        ],
        feature_lookback_days=21,
    )

    assert payload["current_market_state"]["trend_state"] == "bearish"
    assert payload["current_market_state"]["volatility_state"] == "calm"
    assert payload["recommended_mode"]["mode"] == "balanced_overlay_mode"
    assert payload["recommended_mode"]["allocation"] == {
        "soft_three_way_regime": 0.8,
        "pair_fast_exit": 0.2,
    }


def test_build_operating_switch_payload_can_freeze_saved_market_judgement() -> None:
    original_loader = MODULE._load_latest_market_judgement
    MODULE._load_latest_market_judgement = lambda **_: {
        "date": "2026-04-14T00:00:00+00:00",
        "favored_group": "autoresearch",
        "confidence": 1.0,
        "feature_snapshot": {
            "btc_above_ma192": True,
            "btc_above_ma336": True,
            "btc_trend_gap_192": 0.04,
            "btc_trend_gap_336": 0.03,
            "btc_trend_accel": 0.01,
            "breadth_ma96": 0.8,
            "breadth_ma192": 0.8,
            "breadth_delta": 0.1,
            "basket_vol_ratio": 0.6,
        },
    }
    try:
        payload = MODULE.build_operating_switch_payload(
            market_judgement_payload={
                "_path": "/tmp/market.json",
                "current_judgement": {
                    "date": "2026-03-07T00:00:00+00:00",
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
                        "breadth_delta": 0.1,
                        "basket_vol_ratio": 0.45,
                    },
                },
            },
            soft_allocator_payload={
                "_path": "/tmp/soft.json",
                "current_state": {
                    "effective_incumbent_exposure": 0.95,
                    "effective_autoresearch_exposure": 0.05,
                    "_allocator_health": {
                        "healthy": True,
                        "oos_total_return": 0.0009,
                        "oos_sharpe": 0.74,
                    },
                },
            },
            three_way_allocator_payload={
                "_path": "/tmp/hard.json",
                "current_state": {
                    "state": "blend_85_15",
                    "raw_target_state": "incumbent",
                    "_allocator_health": {
                        "healthy": True,
                        "oos_total_return": 0.0020,
                        "oos_sharpe": 1.31,
                    },
                },
            },
            operating_plan_payload={"_path": "/tmp/plan.json", **_base_operating_plan()},
            pair_volume_signals=[
                MODULE.SymbolVolumeSignal(
                    symbol="BNB/USDT",
                    as_of_date="2026-03-07",
                    latest_available_date="2026-03-07",
                    stale_days=0,
                    latest_dollar_volume=1000.0,
                    lookback_mean_dollar_volume=800.0,
                    volume_ratio=1.25,
                    comparison_mode="full_day",
                    state="strong",
                )
            ],
            feature_lookback_days=21,
            market_judgement_mode="saved",
            balanced_strategy_payload={
                "portfolio_metrics": {
                    "val": {"total_return": 0.0841, "sharpe": 4.18},
                    "oos": {"total_return": 0.00215, "sharpe": 0.70, "max_drawdown": 0.00629},
                }
            },
            hybrid_portfolio_payload={
                "readiness": {
                    "beats_cash_refreshed": True,
                    "beats_balanced_refreshed": True,
                    "beats_pair_tactical_refreshed": False,
                    "pair_cap_respected": True,
                    "max_rss_under_8gib": True,
                    "recommended_stage": "guarded_candidate",
                },
                "scenarios": {
                    "refreshed_latest_tail": {
                        "split_metrics": {
                            "val": {"total_return": 0.07826, "sharpe": 3.99},
                            "oos": {
                                "total_return": 0.00482,
                                "sharpe": 3.23,
                                "max_drawdown": 0.00177,
                            },
                        }
                    }
                },
            },
        )
    finally:
        MODULE._load_latest_market_judgement = original_loader

    assert payload["input_paths"]["market_judgement_mode"] == "saved"
    assert payload["current_market_state"]["favored_group"] == "mixed"
    assert payload["recommended_mode"]["mode"] == "hybrid_guarded_mode"


def test_reboot_validation_profile_uses_reboot_paths_with_latest_market_mode() -> None:
    defaults = MODULE._profile_defaults("reboot_validation")

    assert defaults["market_judgement_mode"] == "latest"
    assert "current_switch_validation_current" in str(defaults["market_judgement_path"])
    assert "refreshed_operating_switch_current" in str(defaults["output_dir"])


def test_build_operating_switch_payload_passes_as_of_override_to_latest_loader() -> None:
    original_loader = MODULE._load_latest_market_judgement
    seen: dict[str, object] = {}

    def _fake_latest_loader(**kwargs):
        seen["as_of_date"] = kwargs.get("as_of_date")
        return {
            "date": "2026-04-14T00:00:00+00:00",
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
                "breadth_delta": 0.1,
                "basket_vol_ratio": 0.45,
            },
        }

    MODULE._load_latest_market_judgement = _fake_latest_loader
    try:
        payload = MODULE.build_operating_switch_payload(
            market_judgement_payload={
                "_path": "/tmp/market.json",
                "current_judgement": {
                    "date": "2026-03-07T00:00:00+00:00",
                    "favored_group": "incumbent",
                    "confidence": 1.0,
                },
            },
            soft_allocator_payload={
                "_path": "/tmp/soft.json",
                "current_state": {
                    "effective_incumbent_exposure": 0.95,
                    "effective_autoresearch_exposure": 0.05,
                },
            },
            three_way_allocator_payload={
                "_path": "/tmp/hard.json",
                "current_state": {"state": "blend_85_15", "raw_target_state": "incumbent"},
            },
            operating_plan_payload={"_path": "/tmp/plan.json", **_base_operating_plan()},
            pair_volume_signals=[
                MODULE.SymbolVolumeSignal(
                    symbol="BNB/USDT",
                    as_of_date="2026-04-14",
                    latest_available_date="2026-04-14",
                    stale_days=0,
                    latest_dollar_volume=1000.0,
                    lookback_mean_dollar_volume=800.0,
                    volume_ratio=1.25,
                    comparison_mode="full_day",
                    state="strong",
                )
            ],
            feature_lookback_days=21,
            as_of_date_override=date(2026, 4, 14),
        )
    finally:
        MODULE._load_latest_market_judgement = original_loader

    assert seen["as_of_date"] == date(2026, 4, 14)
    assert payload["input_paths"]["as_of_date_override"] == "2026-04-14"
