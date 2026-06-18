from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import polars as pl

from scripts.research import run_alpha_zoo_clean_new_alpha_discovery as module


def _row(**overrides):
    row = {
        "train_return": 0.20,
        "validation_return": 0.10,
        "train_mdd": 0.05,
        "validation_mdd": 0.04,
        "train_trade_event_count": 50,
        "validation_trade_event_count": 15,
        "train_return_per_turnover_proxy_bps": 30.0,
        "validation_return_per_turnover_proxy_bps": 20.0,
        "locked_oos_return_report_only": -0.90,
        "locked_oos_mdd_report_only": 0.90,
    }
    row.update(overrides)
    return row


def test_fast_finalize_row_matches_reference_finalize_metrics() -> None:
    datetimes = pd.date_range("2025-01-01", periods=12, freq="h")
    close = np.linspace(100.0, 105.0, len(datetimes))
    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )
    signal = np.array([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    sim = module._simulate_symbol(
        frame,
        signal,
        integer_leverage=2,
        allocation_fraction=0.1,
        round_trip_cost_bps=10.0,
        simulation_backend="python",
    )
    base = module._candidate_base(
        family="unit",
        model_parts=("unit", "1h", "BTCUSDT", "lev2"),
        symbol="BTCUSDT",
        timeframe="1h",
        side="long_short",
        lookback=4,
        threshold=0.1,
        exit_threshold=0.0,
        min_hold=1,
        leverage=2,
        allocation_fraction=0.1,
    )
    train = (datetimes[0], datetimes[3])
    validation = (datetimes[4], datetimes[7])
    locked_oos = (datetimes[8], datetimes[-1])

    fast = module._finalize_row(
        base=base,
        sim=sim,
        datetimes=frame["datetime"],
        timeframe="1h",
        train=train,
        validation=validation,
        locked_oos=locked_oos,
    )
    reference = module.broad69.finalize_candidate(
        base,
        sim,
        frame["datetime"],
        timeframe="1h",
        windows=module.broad69.SplitWindows(train=train, validation=validation),
    )

    for key in (
        "train_return",
        "train_mdd",
        "train_sharpe",
        "train_trade_event_count",
        "validation_return",
        "validation_mdd",
        "validation_sharpe",
        "validation_trade_event_count",
        "train_return_per_turnover_proxy_bps",
        "validation_return_per_turnover_proxy_bps",
        "train_validation_score",
        "gate_pass",
    ):
        fast_value = fast.get(key)
        reference_value = reference.get(key)
        if isinstance(fast_value, float):
            assert fast_value == pytest.approx(reference_value)
        else:
            assert fast_value == reference_value
    assert fast["locked_oos_return_report_only"] == pytest.approx(
        module.broad69.split_metrics(
            sim.returns[module._window_mask(frame["datetime"], locked_oos)],
            sim.position[module._window_mask(frame["datetime"], locked_oos)],
            sim.liquidation_flags[module._window_mask(frame["datetime"], locked_oos)],
            sim.account_wipeout_flags[module._window_mask(frame["datetime"], locked_oos)],
            timeframe="1h",
        )["total_return"]
    )


def test_score_row_ignores_locked_oos_report_fields() -> None:
    base = _row(locked_oos_return_report_only=-0.90, locked_oos_mdd_report_only=0.90)
    changed = _row(locked_oos_return_report_only=2.50, locked_oos_mdd_report_only=0.01)

    assert module._score_row(base) == pytest.approx(module._score_row(changed))


def test_robust_selector_score_ignores_locked_oos_report_fields() -> None:
    base = _row(locked_oos_return_report_only=-0.90, locked_oos_mdd_report_only=0.90)
    changed = _row(locked_oos_return_report_only=2.50, locked_oos_mdd_report_only=0.01)

    assert module._robust_v1_score_row(base) == pytest.approx(module._robust_v1_score_row(changed))
    assert module._robust_v1_eligible(base) == module._robust_v1_eligible(changed)


def test_select_fold_candidate_uses_train_validation_score_not_oos() -> None:
    selected = module._select_fold_candidate(
        [
            _row(model_id="oos_winner", validation_return=0.03, locked_oos_return_report_only=1.0),
            _row(
                model_id="validation_winner",
                validation_return=0.15,
                locked_oos_return_report_only=-0.2,
            ),
        ]
    )

    assert selected is not None
    assert selected["model_id"] == "validation_winner"


def test_robust_select_fold_candidate_uses_train_validation_robustness_not_oos() -> None:
    selected = module._select_fold_candidate(
        [
            _row(
                model_id="oos_winner_overfit",
                train_return=0.80,
                validation_return=0.10,
                locked_oos_return_report_only=2.0,
            ),
            _row(
                model_id="stable_train_validation",
                train_return=0.12,
                validation_return=0.10,
                locked_oos_return_report_only=-0.5,
            ),
        ],
        selection_policy=module.ROBUST_SELECTION_POLICY,
    )

    assert selected is not None
    assert selected["model_id"] == "stable_train_validation"


def test_candidate_cap_keeps_robust_eligible_rows_first() -> None:
    capped = module._cap_rows_for_selection(
        [
            _row(
                model_id="ineligible_oos_looking_winner",
                train_return=0.95,
                validation_return=0.10,
                locked_oos_return_report_only=3.0,
            ),
            _row(
                model_id="eligible_stable",
                train_return=0.12,
                validation_return=0.10,
                locked_oos_return_report_only=-0.5,
            ),
        ],
        max_candidates_per_fold=1,
        selection_policy=module.ROBUST_SELECTION_POLICY,
    )

    assert [row["model_id"] for row in capped] == ["eligible_stable"]


def test_rows_for_fold_family_subset_skips_unrequested_generators(monkeypatch) -> None:
    datetimes = pd.date_range("2025-01-01", periods=8, freq="h")
    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": np.linspace(100.0, 101.0, len(datetimes)),
            "high": np.linspace(101.0, 102.0, len(datetimes)),
            "low": np.linspace(99.0, 100.0, len(datetimes)),
            "close": np.linspace(100.0, 101.0, len(datetimes)),
            "volume": np.full(len(datetimes), 1000.0),
        }
    )

    class Fold:
        train = (datetimes[0], datetimes[2])
        validation = (datetimes[3], datetimes[5])
        locked_oos = (datetimes[6], datetimes[-1])

    def fake_squeeze_rows(**_kwargs):
        return [_row(model_id="squeeze", family="volatility_squeeze_breakout")]

    def fail_unrequested(**_kwargs):
        raise AssertionError("unrequested family generator should not run")

    monkeypatch.setattr(module, "_squeeze_rows", fake_squeeze_rows)
    monkeypatch.setattr(module, "_absorption_rows", fail_unrequested)
    rows = module._rows_for_fold(
        bars={("BTCUSDT", "1h"): frame},
        symbols=("BTCUSDT",),
        timeframes=("1h",),
        fold=Fold(),
        max_candidates_per_fold=10,
        enabled_families=("volatility_squeeze_breakout",),
    )

    assert [row["family"] for row in rows] == ["volatility_squeeze_breakout"]


def test_realism_diagnostics_blocks_live_assumption_for_suspicious_validation() -> None:
    diagnostics = module._realism_diagnostics(
        [
            _row(
                ready_for_real=False,
                uses_continuous_position_state_across_split_boundaries=True,
                label_blockers=["fresh_forward_required_before_promotion"],
                validation_return=0.30,
                locked_oos_return_report_only=0.01,
                validation_trade_event_count=10,
                validation_sharpe=6.5,
            )
        ],
        {
            "compounded_oos_return": 0.01,
            "annualized_oos_return_approx": 0.012,
            "monthly_equity_mdd": 0.02,
        },
        selection_policy=module.DEFAULT_SELECTION_POLICY,
    )

    assert diagnostics["live_performance_plausibility"] == "not_supported"
    assert diagnostics["ready_for_real"] is False
    assert diagnostics["real_money_execution"] is False
    assert "validation_to_locked_oos_decay_large" in diagnostics["blockers"]
    assert "selected_rows_not_ready_for_real_money" in diagnostics["blockers"]
    assert (
        "order_book_spread_depth_imbalance_microstructure"
        in diagnostics["external_priors_reflected"]
    )


def test_search_space_hash_is_stable_and_excludes_oos_results() -> None:
    search_space = module._search_space()
    first = module._search_space_hash(search_space)
    second = module._search_space_hash(module._search_space())

    assert list(search_space["families"]) == [
        "volatility_squeeze_breakout",
        "volume_absorption_reversal",
        "range_reclaim_continuation",
        "cross_asset_lead_lag_momentum",
        "btc_beta_residual_momentum",
        "cross_sectional_vol_adjusted_momentum",
        "cross_sectional_dispersion_gated_momentum",
        "cross_sectional_residual_reversal",
        "feature_flow_crowding_reversal",
        "feature_liquidation_imbalance_reversal",
        "feature_flow_oi_trend_continuation",
        "funding_oi_taker_crowding_continuation",
        "perp_crowding_score_reversion",
        "feature_taker_flow_exhaustion_reversal",
        "feature_bbo_flow_exhaustion_reversal",
        "feature_book_depth_imbalance_reversal",
        "feature_microstructure_squeeze_reversal",
        "deep_research_funding_dislocation_trend_carry",
        "deep_research_vol_managed_momentum_crash_gate",
        "deep_research_flow_imbalance_liquidation_sweep",
        "indicator_vwap_atr_bollinger_reversion",
        "indicator_kalman_volatility_trend",
        "indicator_kalman_residual_reversion",
        "indicator_vwap_kalman_pullback_continuation",
        "standardized_indicator_ridge_directional",
    ]
    assert first == second
    assert first == "b4bdb079ba4d9ad5e7202b053858c6d219359126dddaa2a546e38c863ccccca4"


def test_lead_lag_family_is_covered_and_flat_split_labeled() -> None:
    datetimes = pd.date_range("2025-01-01", periods=180, freq="h")
    leader_close = np.r_[np.linspace(100.0, 101.0, 60), np.linspace(101.0, 140.0, 120)]
    target_close = np.r_[np.linspace(100.0, 100.5, 90), np.linspace(100.5, 125.0, 90)]

    def frame(close: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "datetime": datetimes,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(close), 1000.0),
            }
        )

    class Fold:
        train = (datetimes[0], datetimes[59])
        validation = (datetimes[60], datetimes[119])
        locked_oos = (datetimes[120], datetimes[-1])

    bars_by_symbol = {"BTCUSDT": frame(leader_close), "ETHUSDT": frame(target_close)}
    panel = module.broad69._close_panel(bars_by_symbol, ("BTCUSDT", "ETHUSDT"))

    rows = module._lead_lag_rows(
        bars_by_symbol=bars_by_symbol,
        panel=panel,
        symbol="ETHUSDT",
        timeframe="1h",
        fold=Fold(),
        leverages=(2,),
        allocation_fraction=0.1,
    )

    assert rows
    assert {row["family"] for row in rows} == {"cross_asset_lead_lag_momentum"}
    assert {row["split_simulation_policy"] for row in rows} == {
        "continuous_full_period_signal_slice_report_only"
    }
    assert {row["clean_promotion_eligible"] for row in rows} == {False}
    assert all(
        "continuous_position_state_across_split_boundaries" in row["label_blockers"] for row in rows
    )


def test_btc_beta_residual_momentum_family_is_pre_registered() -> None:
    datetimes = pd.date_range("2025-01-01", periods=220, freq="h")
    btc_close = np.linspace(100.0, 130.0, len(datetimes))
    residual_lift = np.r_[np.zeros(110), np.linspace(0.0, 8.0, 110)]
    eth_close = btc_close * 1.02 + residual_lift

    def frame(close: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "datetime": datetimes,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(close), 1000.0),
            }
        )

    class Fold:
        train = (datetimes[0], datetimes[89])
        validation = (datetimes[90], datetimes[159])
        locked_oos = (datetimes[160], datetimes[-1])

    bars_by_symbol = {"BTCUSDT": frame(btc_close), "ETHUSDT": frame(eth_close)}
    panel = module.broad69._close_panel(bars_by_symbol, ("BTCUSDT", "ETHUSDT"))

    rows = module._btc_beta_residual_momentum_rows(
        bars_by_symbol=bars_by_symbol,
        panel=panel,
        symbol="ETHUSDT",
        timeframe="1h",
        fold=Fold(),
        leverages=(2,),
        allocation_fraction=0.1,
    )

    assert rows
    assert {row["family"] for row in rows} == {"btc_beta_residual_momentum"}
    assert {row["benchmark_symbol"] for row in rows} == {"BTCUSDT"}
    assert all(row["uses_locked_oos_for_selection"] is False for row in rows)


def test_kalman_residual_reversion_family_is_pre_registered() -> None:
    datetimes = pd.date_range("2025-01-01", periods=260, freq="h")
    trend = np.linspace(100.0, 104.0, len(datetimes))
    wave = np.sin(np.linspace(0.0, 18.0 * np.pi, len(datetimes))) * 2.0
    close = trend + wave

    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )

    class Fold:
        train = (datetimes[0], datetimes[99])
        validation = (datetimes[100], datetimes[189])
        locked_oos = (datetimes[190], datetimes[-1])

    rows = module._indicator_kalman_residual_reversion_rows(
        frame=frame,
        symbol="ETHUSDT",
        timeframe="1h",
        fold=Fold(),
        leverages=(2,),
        allocation_fraction=0.1,
    )

    assert rows
    assert {row["family"] for row in rows} == {"indicator_kalman_residual_reversion"}
    assert all("kalman_residual_z" in row["indicator_set"] for row in rows)
    assert all(row["uses_locked_oos_for_selection"] is False for row in rows)


def test_vwap_kalman_pullback_continuation_family_is_pre_registered() -> None:
    datetimes = pd.date_range("2025-01-01", periods=520, freq="h")
    trend = np.linspace(100.0, 135.0, len(datetimes))
    pullback_wave = 2.0 * np.sin(np.linspace(0.0, 16.0 * np.pi, len(datetimes)))
    close = trend + pullback_wave
    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        }
    )

    class Fold:
        train = (datetimes[0], datetimes[299])
        validation = (datetimes[300], datetimes[419])
        locked_oos = (datetimes[420], datetimes[-1])

    rows = module._indicator_vwap_kalman_pullback_continuation_rows(
        frame=frame,
        symbol="ETHUSDT",
        timeframe="1h",
        fold=Fold(),
        leverages=(2,),
        allocation_fraction=0.1,
        simulation_backend="python",
    )

    assert rows
    assert {row["family"] for row in rows} == {"indicator_vwap_kalman_pullback_continuation"}
    assert all("kalman_slope_z" in row["indicator_set"] for row in rows)
    assert {row["theory_plausibility_gate"] for row in rows} == {
        "vwap_kalman_pullback_continuation"
    }
    assert all(row["uses_locked_oos_for_selection"] is False for row in rows)


def test_cross_sectional_vol_adjusted_momentum_family_is_pre_registered() -> None:
    datetimes = pd.date_range("2025-01-01", periods=220, freq="h")
    base = np.linspace(100.0, 115.0, len(datetimes))
    eth_close = base + np.r_[np.zeros(110), np.linspace(0.0, 12.0, 110)]
    sol_close = base * 0.95
    xrp_close = base * 1.03 - np.r_[np.zeros(110), np.linspace(0.0, 4.0, 110)]

    def frame(close: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "datetime": datetimes,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(close), 1000.0),
            }
        )

    class Fold:
        train = (datetimes[0], datetimes[89])
        validation = (datetimes[90], datetimes[159])
        locked_oos = (datetimes[160], datetimes[-1])

    bars_by_symbol = {
        "ETHUSDT": frame(eth_close),
        "SOLUSDT": frame(sol_close),
        "XRPUSDT": frame(xrp_close),
    }
    panel = module.broad69._close_panel(
        bars_by_symbol,
        ("ETHUSDT", "SOLUSDT", "XRPUSDT"),
    )

    rows = module._cross_sectional_vol_adjusted_momentum_rows(
        bars_by_symbol=bars_by_symbol,
        panel=panel,
        symbol="ETHUSDT",
        timeframe="1h",
        fold=Fold(),
        leverages=(2,),
        allocation_fraction=0.1,
    )

    assert rows
    assert {row["family"] for row in rows} == {"cross_sectional_vol_adjusted_momentum"}
    assert all(row["uses_locked_oos_for_selection"] is False for row in rows)


def test_cross_sectional_dispersion_gated_momentum_family_is_pre_registered() -> None:
    datetimes = pd.date_range("2025-01-01", periods=260, freq="h")
    base = np.linspace(100.0, 115.0, len(datetimes))
    eth_close = base + np.r_[np.zeros(120), np.linspace(0.0, 10.0, 140)]
    sol_close = base * 0.97
    xrp_close = base * 1.03 - np.r_[np.zeros(120), np.linspace(0.0, 2.0, 140)]

    def frame(close: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "datetime": datetimes,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(close), 1000.0),
            }
        )

    class Fold:
        train = (datetimes[0], datetimes[99])
        validation = (datetimes[100], datetimes[189])
        locked_oos = (datetimes[190], datetimes[-1])

    bars_by_symbol = {
        "ETHUSDT": frame(eth_close),
        "SOLUSDT": frame(sol_close),
        "XRPUSDT": frame(xrp_close),
    }
    panel = module.broad69._close_panel(
        bars_by_symbol,
        ("ETHUSDT", "SOLUSDT", "XRPUSDT"),
    )

    rows = module._cross_sectional_dispersion_gated_momentum_rows(
        bars_by_symbol=bars_by_symbol,
        panel=panel,
        symbol="ETHUSDT",
        timeframe="1h",
        fold=Fold(),
        leverages=(2,),
        allocation_fraction=0.1,
        simulation_backend="python",
    )

    assert rows
    assert {row["family"] for row in rows} == {"cross_sectional_dispersion_gated_momentum"}
    assert all(row["no_nested_oos_mining"] is True for row in rows)
    assert all("rolling_cross_sectional_return_dispersion" in row["indicator_set"] for row in rows)
    assert all(row["uses_locked_oos_for_selection"] is False for row in rows)


def test_cross_sectional_residual_reversal_family_is_pre_registered() -> None:
    datetimes = pd.date_range("2025-01-01", periods=260, freq="h")
    market = np.linspace(100.0, 112.0, len(datetimes))
    shock = np.zeros(len(datetimes))
    shock[90:130] = np.linspace(0.0, 10.0, 40)
    shock[130:180] = np.linspace(10.0, -4.0, 50)
    shock[180:] = np.linspace(-4.0, 0.0, len(datetimes) - 180)
    eth_close = market + shock
    sol_close = market * 0.98
    xrp_close = market * 1.03

    def frame(close: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "datetime": datetimes,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(close), 1000.0),
            }
        )

    class Fold:
        train = (datetimes[0], datetimes[99])
        validation = (datetimes[100], datetimes[189])
        locked_oos = (datetimes[190], datetimes[-1])

    bars_by_symbol = {
        "ETHUSDT": frame(eth_close),
        "SOLUSDT": frame(sol_close),
        "XRPUSDT": frame(xrp_close),
    }
    panel = module.broad69._close_panel(
        bars_by_symbol,
        ("ETHUSDT", "SOLUSDT", "XRPUSDT"),
    )

    rows = module._cross_sectional_residual_reversal_rows(
        bars_by_symbol=bars_by_symbol,
        panel=panel,
        symbol="ETHUSDT",
        timeframe="1h",
        fold=Fold(),
        leverages=(2,),
        allocation_fraction=0.1,
        simulation_backend="python",
    )

    assert rows
    assert {row["family"] for row in rows} == {"cross_sectional_residual_reversal"}
    assert all(row["no_nested_oos_mining"] is True for row in rows)
    assert {row["theory_plausibility_gate"] for row in rows} == {
        "cross_sectional_residual_stat_arb_reversal"
    }
    assert all(row["uses_locked_oos_for_selection"] is False for row in rows)


def test_load_feature_points_safe_tolerates_missing_taker_columns(tmp_path: Path) -> None:
    day_dir = tmp_path / "symbol=BTCUSDT" / "date=2025-01-01"
    day_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp_ms": [1735689600000],
            "funding_rate": [0.0001],
            "open_interest": [1234.0],
        }
    ).write_parquet(day_dir / "compact.parquet")

    loaded = module._load_feature_points_safe("BTCUSDT", feature_root=tmp_path)

    assert list(loaded.columns) == [
        "timestamp_ms",
        "funding_rate",
        "open_interest",
        "taker_buy_quote_volume",
        "taker_sell_quote_volume",
        "liquidation_long_notional",
        "liquidation_short_notional",
        "bbo_spread_bps",
        "book_depth_imbalance_1pct",
        "datetime",
    ]
    assert loaded["taker_buy_quote_volume"].isna().all()
    assert loaded["taker_sell_quote_volume"].isna().all()
    assert loaded["liquidation_long_notional"].isna().all()
    assert loaded["liquidation_short_notional"].isna().all()
    assert loaded["bbo_spread_bps"].isna().all()


def test_load_feature_points_safe_ignores_tmp_parquet(tmp_path: Path) -> None:
    day_dir = tmp_path / "symbol=BTCUSDT" / "date=2025-01-01"
    day_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp_ms": [1735689600000],
            "funding_rate": [0.0001],
            "open_interest": [1234.0],
            "taker_buy_quote_volume": [200.0],
            "taker_sell_quote_volume": [100.0],
        }
    ).write_parquet(day_dir / "compact.tmp.parquet")

    loaded = module._load_feature_points_safe("BTCUSDT", feature_root=tmp_path)

    assert loaded.empty


def test_attach_feature_points_builds_liquidation_imbalance() -> None:
    bars = pd.DataFrame(
        {"datetime": pd.date_range("2025-01-01", periods=2, freq="h"), "close": [1.0, 1.0]}
    )
    features = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=2, freq="h"),
            "funding_rate": [0.0001, 0.0001],
            "open_interest": [1000.0, 1001.0],
            "taker_buy_quote_volume": [200.0, 100.0],
            "taker_sell_quote_volume": [100.0, 200.0],
            "liquidation_long_notional": [300.0, 50.0],
            "liquidation_short_notional": [100.0, 150.0],
        }
    )

    attached = module._attach_feature_points(bars, features, timeframe="1h")

    assert attached["liquidation_imbalance"].tolist() == pytest.approx([0.5, -0.5])
    assert attached["feature_valid"].tolist() == [True, True]
    assert attached["feature_liquidation_valid"].tolist() == [True, True]


def test_attach_feature_points_keeps_taker_flow_valid_without_liquidations() -> None:
    bars = pd.DataFrame(
        {"datetime": pd.date_range("2025-01-01", periods=2, freq="h"), "close": [1.0, 1.0]}
    )
    features = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=2, freq="h"),
            "funding_rate": [0.0001, -0.0001],
            "open_interest": [1000.0, 1001.0],
            "taker_buy_quote_volume": [200.0, 100.0],
            "taker_sell_quote_volume": [100.0, 200.0],
        }
    )

    attached = module._attach_feature_points(bars, features, timeframe="1h")

    assert attached["feature_valid"].tolist() == [True, True]
    assert attached["feature_liquidation_valid"].tolist() == [False, False]
    assert attached["feature_bbo_valid"].tolist() == [False, False]


def test_attach_feature_points_aligns_sparse_feature_sources_independently() -> None:
    bars = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01 00:30", "2025-01-01 01:00"]),
            "close": [1.0, 1.0],
        }
    )
    features = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 00:10",
                    "2025-01-01 00:20",
                ]
            ),
            "funding_rate": [0.0001, np.nan, np.nan],
            "open_interest": [np.nan, 1000.0, np.nan],
            "taker_buy_quote_volume": [np.nan, np.nan, 200.0],
            "taker_sell_quote_volume": [np.nan, np.nan, 100.0],
        }
    )

    attached = module._attach_feature_points(bars, features, timeframe="1h")

    assert attached["funding_rate"].tolist() == pytest.approx([0.0001, 0.0001])
    assert attached["open_interest"].tolist() == pytest.approx([1000.0, 1000.0])
    assert attached["taker_buy_sell_imbalance"].tolist() == pytest.approx([1.0 / 3.0, 1.0 / 3.0])
    assert attached["feature_valid"].tolist() == [True, True]
    assert attached["feature_oi_flow_valid"].tolist() == [True, True]
    assert attached["feature_liquidation_valid"].tolist() == [False, False]


def test_attach_feature_points_keeps_flow_valid_without_open_interest() -> None:
    bars = pd.DataFrame(
        {"datetime": pd.date_range("2025-01-01", periods=2, freq="h"), "close": [1.0, 1.0]}
    )
    features = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=2, freq="h"),
            "funding_rate": [0.0001, -0.0001],
            "taker_buy_quote_volume": [200.0, 100.0],
            "taker_sell_quote_volume": [100.0, 200.0],
        }
    )

    attached = module._attach_feature_points(bars, features, timeframe="1h")

    assert attached["feature_valid"].tolist() == [True, True]
    assert attached["feature_oi_flow_valid"].tolist() == [False, False]
    assert attached["feature_liquidation_valid"].tolist() == [False, False]
    assert attached["feature_bbo_valid"].tolist() == [False, False]


def test_attach_feature_points_empty_features_sets_all_validity_flags() -> None:
    bars = pd.DataFrame(
        {"datetime": pd.date_range("2025-01-01", periods=2, freq="h"), "close": [1.0, 1.0]}
    )

    attached = module._attach_feature_points(bars, pd.DataFrame(), timeframe="1h")

    assert attached["feature_valid"].tolist() == [False, False]
    assert attached["feature_oi_flow_valid"].tolist() == [False, False]
    assert attached["feature_liquidation_valid"].tolist() == [False, False]
    assert attached["feature_bbo_valid"].tolist() == [False, False]
    assert attached["feature_depth_valid"].tolist() == [False, False]


def test_deep_research_report_rows_are_gated_and_report_only() -> None:
    datetimes = pd.date_range("2025-01-01", periods=180, freq="h")
    close = np.linspace(100.0, 130.0, len(datetimes))
    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(datetimes), 1000.0),
            "funding_rate": np.full(len(datetimes), -0.00008),
            "open_interest": np.linspace(1000.0, 1120.0, len(datetimes)),
            "taker_buy_sell_imbalance": np.full(len(datetimes), 0.30),
            "liquidation_imbalance": np.full(len(datetimes), 0.0),
            "bbo_spread_bps": np.full(len(datetimes), 4.0),
            "book_depth_imbalance_1pct": np.full(len(datetimes), 0.20),
            "feature_valid": np.full(len(datetimes), True),
            "feature_oi_flow_valid": np.full(len(datetimes), True),
            "feature_liquidation_valid": np.full(len(datetimes), True),
            "feature_bbo_valid": np.full(len(datetimes), True),
            "feature_depth_valid": np.full(len(datetimes), True),
        }
    )
    frame.loc[90, "close"] = 115.0
    frame.loc[91, "close"] = 108.0
    frame.loc[91, "liquidation_imbalance"] = 1.0

    class Fold:
        train = (datetimes[0], datetimes[59])
        validation = (datetimes[60], datetimes[119])
        locked_oos = (datetimes[120], datetimes[-1])

    kwargs = {
        "frame": frame,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "fold": Fold(),
        "leverages": (2,),
        "allocation_fraction": 0.1,
    }
    rows = (
        module._deep_research_funding_dislocation_trend_carry_rows(**kwargs)
        + module._deep_research_vol_managed_momentum_crash_gate_rows(
            **kwargs,
            bars_by_symbol={"BTCUSDT": frame},
        )
        + module._deep_research_flow_imbalance_liquidation_sweep_rows(**kwargs)
    )

    families = {row["family"] for row in rows}
    assert "deep_research_funding_dislocation_trend_carry" in families
    assert "deep_research_vol_managed_momentum_crash_gate" in families
    assert "deep_research_flow_imbalance_liquidation_sweep" in families
    assert {row["source_report"] for row in rows} == {"desktop-deep-research-report-20260608"}
    assert {row["no_nested_oos_mining"] for row in rows} == {True}
    assert {row["uses_locked_oos_for_selection"] for row in rows} == {False}
    assert {row["real_money_execution"] for row in rows} == {False}
    assert {row["clean_promotion_eligible"] for row in rows} == {False}


def test_microstructure_squeeze_reversal_rows_are_feature_backed() -> None:
    datetimes = pd.date_range("2025-01-01", periods=240, freq="h")
    phase = np.linspace(0.0, 16.0 * np.pi, len(datetimes))
    close = 100.0 + np.sin(phase) * 3.0
    extension = pd.Series(close).pct_change(6).fillna(0.0).to_numpy()
    pressure_sign = np.where(extension < 0.0, -1.0, 1.0)
    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(datetimes), 1000.0),
            "funding_rate": np.zeros(len(datetimes)),
            "open_interest": np.linspace(1000.0, 1100.0, len(datetimes)),
            "taker_buy_sell_imbalance": 0.25 * pressure_sign,
            "liquidation_imbalance": np.zeros(len(datetimes)),
            "bbo_spread_bps": 3.0 + np.abs(np.sin(phase)) * 20.0,
            "book_depth_imbalance_1pct": -0.25 * pressure_sign,
            "feature_valid": np.full(len(datetimes), True),
            "feature_oi_flow_valid": np.full(len(datetimes), True),
            "feature_liquidation_valid": np.full(len(datetimes), True),
            "feature_bbo_valid": np.full(len(datetimes), True),
            "feature_depth_valid": np.full(len(datetimes), True),
        }
    )

    class Fold:
        train = (datetimes[0], datetimes[79])
        validation = (datetimes[80], datetimes[159])
        locked_oos = (datetimes[160], datetimes[-1])

    rows = module._feature_microstructure_squeeze_reversal_rows(
        frame=frame,
        symbol="ETHUSDT",
        timeframe="1h",
        fold=Fold(),
        leverages=(2,),
        allocation_fraction=0.1,
    )

    assert rows
    assert {row["family"] for row in rows} == {"feature_microstructure_squeeze_reversal"}
    assert {row["feature_backed"] for row in rows} == {True}
    assert all("bbo_spread_bps" in row["microstructure_inputs"] for row in rows)
    assert all(row["uses_locked_oos_for_selection"] is False for row in rows)


def test_indicator_and_train_only_ml_rows_are_report_only() -> None:
    datetimes = pd.date_range("2025-01-01", periods=520, freq="h")
    base = np.linspace(100.0, 140.0, len(datetimes))
    wave = 3.0 * np.sin(np.linspace(0.0, 18.0, len(datetimes)))
    close = base + wave
    high_spread = 1.005 + 0.002 * np.sin(np.linspace(0.0, 13.0, len(datetimes)))
    low_spread = 0.995 - 0.002 * np.cos(np.linspace(0.0, 11.0, len(datetimes)))
    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": close * high_spread,
            "low": close * low_spread,
            "close": close,
            "volume": 1000.0 + 50.0 * np.cos(np.linspace(0.0, 21.0, len(datetimes))),
        }
    )

    class Fold:
        train = (datetimes[0], datetimes[359])
        validation = (datetimes[360], datetimes[459])
        locked_oos = (datetimes[460], datetimes[-1])

    kwargs = {
        "frame": frame,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "fold": Fold(),
        "leverages": (2,),
        "allocation_fraction": 0.1,
    }
    rows = (
        module._indicator_vwap_atr_bollinger_reversion_rows(**kwargs)
        + module._indicator_kalman_volatility_trend_rows(**kwargs)
        + module._indicator_kalman_residual_reversion_rows(**kwargs)
        + module._standardized_indicator_ridge_directional_rows(**kwargs)
    )

    families = {row["family"] for row in rows}
    assert "indicator_vwap_atr_bollinger_reversion" in families
    assert "indicator_kalman_volatility_trend" in families
    assert "indicator_kalman_residual_reversion" in families
    assert "standardized_indicator_ridge_directional" in families
    assert {row["uses_locked_oos_for_selection"] for row in rows} == {False}
    assert {row["real_money_execution"] for row in rows} == {False}
    assert {row["clean_promotion_eligible"] for row in rows} == {False}
    ml_rows = [row for row in rows if row["family"] == "standardized_indicator_ridge_directional"]
    assert ml_rows
    assert {row["uses_ml"] for row in ml_rows} == {True}
    assert {row["ml_fit_scope"] for row in ml_rows} == {"train_only"}
    assert {row["standardization_scope"] for row in ml_rows} == {"train_only"}
    assert {row["no_nested_oos_mining"] for row in ml_rows} == {True}


def test_run_writes_policy_flags_with_synthetic_loader(monkeypatch, tmp_path: Path) -> None:
    datetimes = pd.date_range("2025-01-01", periods=240, freq="h")
    close = np.linspace(100.0, 130.0, len(datetimes))
    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1000.0, 1500.0, len(datetimes)),
        }
    )

    def fake_load_all_bars(symbols, *, data_root, timeframes):
        return {
            (symbol, timeframe): frame.copy() for symbol in symbols for timeframe in timeframes
        }, {"latest_available_data": datetimes[-1].isoformat()}

    class Fold:
        fold_id = "2025-03"
        train = (datetimes[0], datetimes[80])
        validation = (datetimes[81], datetimes[160])
        locked_oos = (datetimes[161], datetimes[-1])

    monkeypatch.setattr(module.broad69, "load_all_bars", fake_load_all_bars)
    monkeypatch.setattr(module.monthly, "build_monthly_folds", lambda **_: [Fold()])

    payload = module.run(
        data_root=tmp_path,
        output_dir=tmp_path / "out",
        symbols=("BTCUSDT",),
        timeframes=("1h",),
        max_folds=None,
        max_candidates_per_fold=5,
    )

    assert payload["optimization_policy"]["uses_locked_oos_for_selection"] is False
    assert payload["optimization_policy"]["post_oos_selector_trusted"] is False
    assert payload["fresh_forward_required"] is True
    assert payload["real_money_execution"] is False
    assert payload["fold_workers"] == 1
    assert len(payload["enabled_families"]) == len(module.FAMILY_DESCRIPTIONS)
    assert Path(payload["output_paths"]["json"]).exists()
    assert Path(payload["output_paths"]["markdown"]).exists()
    assert payload["realism_diagnostics"]["live_performance_plausibility"] == "not_supported"
    assert payload["realism_diagnostics"]["real_money_execution"] is False
    assert "Live realism diagnostics" in Path(payload["output_paths"]["markdown"]).read_text()
    loaded = json.loads(Path(payload["output_paths"]["json"]).read_text())
    assert (
        loaded["pre_registered_search_space_sha256"]
        == payload["pre_registered_search_space_sha256"]
    )

    robust_payload = module.run(
        data_root=tmp_path,
        output_dir=tmp_path / "out_robust",
        symbols=("BTCUSDT",),
        timeframes=("1h",),
        max_folds=None,
        max_candidates_per_fold=5,
        selection_policy=module.ROBUST_SELECTION_POLICY,
        enabled_families=("indicator_kalman_residual_reversion",),
        fold_workers=2,
    )

    assert robust_payload["selection_policy"] == module.ROBUST_SELECTION_POLICY
    assert robust_payload["enabled_families"] == ["indicator_kalman_residual_reversion"]
    assert robust_payload["fold_workers"] == 2
    assert robust_payload["simulation_backend"]["requested_backend"] == "auto"
    assert (
        robust_payload["optimization_policy"]["selection_policy"] == module.ROBUST_SELECTION_POLICY
    )
    robust_loaded = json.loads(Path(robust_payload["output_paths"]["json"]).read_text())
    assert robust_loaded["selection_policy"] == module.ROBUST_SELECTION_POLICY
    assert all(
        "selection_score_robust_v1_train_validation_only" in row
        for row in robust_loaded["candidate_rows"]
    )
    assert (
        robust_loaded["realism_diagnostics"]["selection_policy"] == module.ROBUST_SELECTION_POLICY
    )
    assert (
        "robust_selector_is_post_failure_diagnostic_requires_fresh_forward"
        in robust_loaded["realism_diagnostics"]["blockers"]
    )



def test_promotion_gate_contract_exposes_benchmarks_and_forbidden_oos_uses() -> None:
    contract = module._gate_contract_payload(selection_policy=module.DEFAULT_SELECTION_POLICY)

    assert contract["selection_inputs"] == ["train", "validation"]
    assert contract["locked_oos_policy"] == "report_gate_only_after_train_validation_freeze"
    assert contract["benchmarks"]["shadow_compounded_oos_return"] == pytest.approx(0.6442)
    assert contract["benchmarks"]["shadow_return_mdd"] == pytest.approx(3.49)
    assert contract["benchmarks"]["clean_paper_compounded_oos_return"] == pytest.approx(0.3439)
    assert contract["benchmarks"]["max_mdd"] == pytest.approx(0.30)
    assert "uses_locked_oos_for_tie_break" in contract["locked_oos_forbidden_uses"]
    assert "uses_locked_oos_for_correlation" in contract["locked_oos_forbidden_uses"]
    assert "uses_locked_oos_for_sizing" in contract["locked_oos_forbidden_uses"]
    assert contract["real_money_execution"] is False


def test_promotion_gate_report_flags_oos_contamination_without_using_oos_for_selection() -> None:
    clean = _row(
        model_id="clean_candidate",
        locked_oos_return_report_only=0.50,
        locked_oos_mdd_report_only=0.10,
        locked_oos_liquidation_count_report_only=0,
        locked_oos_account_wipeout_count_report_only=0,
        uses_locked_oos_for_tie_break=False,
        uses_locked_oos_for_correlation=False,
        uses_locked_oos_for_sizing=False,
        selected_by_train_validation_freeze=True,
    )
    contaminated = dict(clean)
    contaminated["uses_locked_oos_for_tie_break"] = True
    contaminated["locked_oos_return_report_only"] = 10.0

    clean_gate = module._promotion_gate_report(
        clean, selection_policy=module.DEFAULT_SELECTION_POLICY
    )
    contaminated_gate = module._promotion_gate_report(
        contaminated, selection_policy=module.DEFAULT_SELECTION_POLICY
    )

    assert clean_gate["can_advance_to_full_wf"] is True
    assert contaminated_gate["can_advance_to_full_wf"] is False
    assert "locked_oos_used_by_selection_path" in contaminated_gate["rejection_reasons"]
    assert module._score_row(clean) == pytest.approx(module._score_row(contaminated))


def test_promotion_summary_and_tried_universe_report_retain_rejection_reasons() -> None:
    accepted = _row(model_id="accepted", family="volatility_squeeze_breakout")
    accepted.update(
        module._promotion_gate_report(
            accepted, selection_policy=module.DEFAULT_SELECTION_POLICY
        )
    )
    weak = _row(
        model_id="weak",
        family="feature_bbo_flow_exhaustion_reversal",
        feature_backed=True,
        feature_coverage={"train": 0.1, "validation": 0.1, "locked_oos": 1.0},
    )
    weak.update(module._promotion_gate_report(weak, selection_policy=module.DEFAULT_SELECTION_POLICY))

    summary = module._promotion_summary([accepted, weak], [accepted])

    assert summary["candidate_count_total"] == 2
    assert summary["selected_fold_count"] == 1
    assert summary["family_counts"]["volatility_squeeze_breakout"] == 1
    assert summary["promotion_status_counts"]["shadow_research_only_weak_data"] == 1
    assert summary["rejection_reason_counts"]["insufficient_train_validation_feature_coverage"] == 1



def test_all_locked_oos_usage_flags_fail_selection_eligibility() -> None:
    for key in module._LOCKED_OOS_USAGE_KEYS:
        row = _row(**{key: True})

        assert not module._eligible_for_policy(
            row, selection_policy=module.DEFAULT_SELECTION_POLICY
        ), key
        assert not module._eligible_for_policy(
            row, selection_policy=module.ROBUST_SELECTION_POLICY
        ), key


def test_promotion_gate_fails_closed_for_missing_or_bad_oos_risk_metrics() -> None:
    missing = _row()
    for key in (
        "locked_oos_return_report_only",
        "locked_oos_mdd_report_only",
        "locked_oos_liquidation_count_report_only",
        "locked_oos_account_wipeout_count_report_only",
    ):
        missing.pop(key, None)
    high_mdd = _row(
        locked_oos_return_report_only=1.0,
        locked_oos_mdd_report_only=0.31,
        locked_oos_liquidation_count_report_only=0,
        locked_oos_account_wipeout_count_report_only=0,
    )
    liquidation = _row(
        locked_oos_return_report_only=1.0,
        locked_oos_mdd_report_only=0.10,
        locked_oos_liquidation_count_report_only=1,
        locked_oos_account_wipeout_count_report_only=0,
    )

    for row in (missing, high_mdd, liquidation):
        gate = module._promotion_gate_report(row, selection_policy=module.DEFAULT_SELECTION_POLICY)

        assert gate["can_advance_to_full_wf"] is False
        assert gate["report_only_locked_oos"]["risk_gate_pass"] is False

    missing_gate = module._promotion_gate_report(
        missing, selection_policy=module.DEFAULT_SELECTION_POLICY
    )
    assert missing_gate["report_only_locked_oos"]["risk_metrics_present"] is False
    assert "missing_locked_oos_risk_metrics" in missing_gate["rejection_reasons"]
    assert "report_gate_mdd_above_30pct" in module._promotion_gate_report(
        high_mdd, selection_policy=module.DEFAULT_SELECTION_POLICY
    )["rejection_reasons"]
    assert "report_gate_locked_oos_liquidation" in module._promotion_gate_report(
        liquidation, selection_policy=module.DEFAULT_SELECTION_POLICY
    )["rejection_reasons"]



def test_markdown_exposes_candidate_level_gate_audit_rows() -> None:
    row = _row(
        model_id="candidate-a",
        family="volatility_squeeze_breakout",
        fold_id="2026-01",
        candidate_freeze_sha256="abcdef1234567890",
        selected_by_train_validation_freeze=True,
        locked_oos_return_report_only=0.50,
        locked_oos_mdd_report_only=0.10,
        locked_oos_liquidation_count_report_only=0,
        locked_oos_account_wipeout_count_report_only=0,
    )
    gate = module._promotion_gate_report(row, selection_policy=module.DEFAULT_SELECTION_POLICY)
    row["promotion_gate"] = gate
    row["promotion_status"] = gate["promotion_status"]
    row["rejection_reasons"] = gate["rejection_reasons"]
    payload = {
        "generated_at_utc": "2026-06-18T00:00:00Z",
        "pre_registered_search_space_sha256": "hash",
        "selection_policy": module.DEFAULT_SELECTION_POLICY,
        "enabled_families": ["volatility_squeeze_breakout"],
        "integer_leverages": [2],
        "fold_workers": 1,
        "simulation_backend": {"resolved_backend": "python"},
        "candidate_cap_sort_policy": "eligible_first_active_train_validation_selection_score",
        "candidate_row_count_total": 1,
        "candidate_rows": [row],
        "selected_fold_rows": [row],
        "gate_contract": module._gate_contract_payload(selection_policy=module.DEFAULT_SELECTION_POLICY),
        "promotion_summary": module._promotion_summary([row], [row]),
        "tried_universe": {
            "enabled_families": ["volatility_squeeze_breakout"],
            "skipped_families": [],
        },
        "aggregate": {"fold_count": 1, "compounded_oos_return": 0.5},
        "realism_diagnostics": {"blockers": []},
    }

    markdown = module._render_markdown(payload)

    assert "## Candidate gate audit rows" in markdown
    assert "`candidate-a`" in markdown
    assert "abcdef123456" in markdown
    assert "## Full-WF promotion candidates" in markdown
    assert "selection rule: `survivor_manifest_train_validation_freeze_only`" in markdown


def _manifest_keys(value):
    if isinstance(value, dict):
        keys = []
        for key, inner in value.items():
            keys.append(str(key))
            keys.extend(_manifest_keys(inner))
        return keys
    if isinstance(value, list):
        keys = []
        for inner in value:
            keys.extend(_manifest_keys(inner))
        return keys
    return []


def _survivor_row(**overrides):
    row = _row(
        model_id="survivor-a",
        family="volatility_squeeze_breakout",
        fold_id="2026-01",
        symbol="BTCUSDT",
        timeframe="1h",
        side="long_short",
        lookback=24,
        threshold=0.10,
        exit_threshold=0.0,
        min_hold=4,
        integer_leverage=2,
        allocation_fraction=0.10,
        selected_by_train_validation_freeze=True,
        candidate_freeze_sha256="freeze-a",
        selection_score_active_train_validation_only=0.25,
        selection_score_robust_v1_train_validation_only=0.18,
        locked_oos_return_report_only=0.50,
        locked_oos_mdd_report_only=0.10,
        locked_oos_liquidation_count_report_only=0,
        locked_oos_account_wipeout_count_report_only=0,
    )
    row.update(overrides)
    row["promotion_gate"] = module._promotion_gate_report(
        row, selection_policy=module.DEFAULT_SELECTION_POLICY
    )
    row["promotion_status"] = row["promotion_gate"]["promotion_status"]
    row["rejection_reasons"] = row["promotion_gate"]["rejection_reasons"]
    return row


def test_survivor_manifest_excludes_locked_oos_metrics_from_freeze_contract() -> None:
    row = _survivor_row()

    manifest = module._survivor_manifest_payload(
        [row],
        search_hash="search-hash",
        selection_policy=module.DEFAULT_SELECTION_POLICY,
        enabled_families=("volatility_squeeze_breakout",),
        leverages=(2,),
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )

    assert manifest["frozen_survivor_count"] == 1
    assert manifest["full_wf_retest_candidate_count"] == 1
    assert manifest["selection_inputs"] == ["train", "validation"]
    assert manifest["optimizer_holdout_use_allowed"] is False
    assert manifest["fresh_forward_required"] is True
    assert manifest["real_money_execution"] is False
    assert not [key for key in _manifest_keys(manifest) if "locked_oos" in key]
    survivor = manifest["frozen_survivors"][0]
    assert survivor["eligible_for_full_wf_retest"] is True
    assert survivor["full_wf_retest_blockers"] == []
    assert survivor["selected_by_train_validation_freeze"] is True


def test_survivor_manifest_hash_ignores_report_only_oos_metric_changes() -> None:
    base = _survivor_row(
        locked_oos_return_report_only=0.50,
        locked_oos_mdd_report_only=0.10,
    )
    changed = _survivor_row(
        locked_oos_return_report_only=-0.90,
        locked_oos_mdd_report_only=0.90,
    )

    kwargs = dict(
        search_hash="search-hash",
        selection_policy=module.DEFAULT_SELECTION_POLICY,
        enabled_families=("volatility_squeeze_breakout",),
        leverages=(2,),
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )

    assert module._survivor_manifest_payload([base], **kwargs)[
        "survivor_manifest_sha256"
    ] == module._survivor_manifest_payload([changed], **kwargs)[
        "survivor_manifest_sha256"
    ]


def test_survivor_manifest_blocks_holdout_contaminated_retest_candidate() -> None:
    row = _survivor_row(uses_locked_oos_for_tie_break=True)

    manifest = module._survivor_manifest_payload(
        [row],
        search_hash="search-hash",
        selection_policy=module.DEFAULT_SELECTION_POLICY,
        enabled_families=("volatility_squeeze_breakout",),
        leverages=(2,),
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )

    assert manifest["frozen_survivor_count"] == 1
    assert manifest["full_wf_retest_candidate_count"] == 0
    survivor = manifest["frozen_survivors"][0]
    assert survivor["eligible_for_full_wf_retest"] is False
    assert survivor["holdout_usage_contamination_detected"] is True
    assert "holdout_used_by_train_validation_path" in survivor["full_wf_retest_blockers"]


def test_unselected_eligible_row_cannot_advance_to_full_wf() -> None:
    unselected = _survivor_row(
        model_id="unselected",
        selected_by_train_validation_freeze=False,
    )
    selected = _survivor_row(
        model_id="selected",
        selected_by_train_validation_freeze=True,
    )

    unselected_gate = module._promotion_gate_report(
        unselected, selection_policy=module.DEFAULT_SELECTION_POLICY
    )
    selected_gate = module._promotion_gate_report(
        selected, selection_policy=module.DEFAULT_SELECTION_POLICY
    )
    manifest = module._survivor_manifest_payload(
        [selected],
        search_hash="search-hash",
        selection_policy=module.DEFAULT_SELECTION_POLICY,
        enabled_families=("volatility_squeeze_breakout",),
        leverages=(2,),
        symbols=("BTCUSDT",),
        timeframes=("1h",),
    )
    payload = {
        "generated_at_utc": "2026-06-18T00:00:00Z",
        "pre_registered_search_space_sha256": "hash",
        "selection_policy": module.DEFAULT_SELECTION_POLICY,
        "enabled_families": ["volatility_squeeze_breakout"],
        "integer_leverages": [2],
        "fold_workers": 1,
        "simulation_backend": {"resolved_backend": "python"},
        "candidate_cap_sort_policy": "eligible_first_active_train_validation_selection_score",
        "candidate_row_count_total": 2,
        "candidate_rows": [
            {**unselected, "promotion_gate": unselected_gate},
            {**selected, "promotion_gate": selected_gate},
        ],
        "selected_fold_rows": [selected],
        "survivor_manifest": manifest,
        "gate_contract": module._gate_contract_payload(selection_policy=module.DEFAULT_SELECTION_POLICY),
        "promotion_summary": module._promotion_summary([unselected, selected], [selected]),
        "tried_universe": {"enabled_families": ["volatility_squeeze_breakout"], "skipped_families": []},
        "aggregate": {"fold_count": 1, "compounded_oos_return": 0.5},
        "realism_diagnostics": {"blockers": []},
    }

    assert unselected_gate["train_validation_smoke_survivor"] is True
    assert unselected_gate["can_advance_to_full_wf"] is False
    assert selected_gate["can_advance_to_full_wf"] is True
    markdown = module._render_markdown(payload)
    assert "full-WF retest candidates: `1`" in markdown
    assert "count: `1`" in markdown


def test_run_writes_survivor_manifest_artifact(monkeypatch, tmp_path: Path) -> None:
    datetimes = pd.date_range("2025-01-01", periods=240, freq="h")
    close = np.linspace(100.0, 130.0, len(datetimes))
    frame = pd.DataFrame(
        {
            "datetime": datetimes,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1000.0, 1500.0, len(datetimes)),
        }
    )

    def fake_load_all_bars(symbols, *, data_root, timeframes):
        return {
            (symbol, timeframe): frame.copy() for symbol in symbols for timeframe in timeframes
        }, {"latest_available_data": datetimes[-1].isoformat()}

    class Fold:
        fold_id = "2025-03"
        train = (datetimes[0], datetimes[80])
        validation = (datetimes[81], datetimes[160])
        locked_oos = (datetimes[161], datetimes[-1])

    monkeypatch.setattr(module.broad69, "load_all_bars", fake_load_all_bars)
    monkeypatch.setattr(module.monthly, "build_monthly_folds", lambda **_: [Fold()])

    payload = module.run(
        data_root=tmp_path,
        output_dir=tmp_path / "out",
        symbols=("BTCUSDT",),
        timeframes=("1h",),
        max_folds=None,
        max_candidates_per_fold=5,
        enabled_families=("volatility_squeeze_breakout",),
        max_candidate_rows_output=5,
    )

    manifest_path = Path(payload["output_paths"]["survivor_manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifact_kind"] == "alpha_zoo_clean_new_alpha_survivor_manifest"
    assert manifest["selection_inputs"] == ["train", "validation"]
    assert manifest["optimizer_holdout_use_allowed"] is False
    assert manifest["real_money_execution"] is False
    assert manifest["survivor_manifest_sha256"] == payload["survivor_manifest"]["survivor_manifest_sha256"]
    markdown = Path(payload["output_paths"]["markdown"]).read_text()
    assert "## Survivor manifest" in markdown
    assert "full-WF retest candidates" in markdown

