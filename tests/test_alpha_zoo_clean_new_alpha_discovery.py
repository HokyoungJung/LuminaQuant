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
        "feature_flow_crowding_reversal",
        "feature_liquidation_imbalance_reversal",
        "feature_flow_oi_trend_continuation",
        "funding_oi_taker_crowding_continuation",
        "perp_crowding_score_reversion",
        "feature_taker_flow_exhaustion_reversal",
        "feature_bbo_flow_exhaustion_reversal",
        "feature_book_depth_imbalance_reversal",
        "deep_research_funding_dislocation_trend_carry",
        "deep_research_vol_managed_momentum_crash_gate",
        "deep_research_flow_imbalance_liquidation_sweep",
        "indicator_vwap_atr_bollinger_reversion",
        "indicator_kalman_volatility_trend",
        "standardized_indicator_ridge_directional",
    ]
    assert first == second
    assert first == "57121f6a8ade6faeaf1a83b06276728a8f3590d320d5af501ce3115e9b260a82"


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
        + module._standardized_indicator_ridge_directional_rows(**kwargs)
    )

    families = {row["family"] for row in rows}
    assert "indicator_vwap_atr_bollinger_reversion" in families
    assert "indicator_kalman_volatility_trend" in families
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
    )

    assert robust_payload["selection_policy"] == module.ROBUST_SELECTION_POLICY
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
