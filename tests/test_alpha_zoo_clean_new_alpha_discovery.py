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


def test_search_space_hash_is_stable_and_excludes_oos_results() -> None:
    search_space = module._search_space()
    first = module._search_space_hash(search_space)
    second = module._search_space_hash(module._search_space())

    assert list(search_space["families"]) == [
        "volatility_squeeze_breakout",
        "volume_absorption_reversal",
        "range_reclaim_continuation",
        "cross_asset_lead_lag_momentum",
        "feature_flow_crowding_reversal",
        "feature_liquidation_imbalance_reversal",
        "feature_flow_oi_trend_continuation",
        "funding_oi_taker_crowding_continuation",
        "perp_crowding_score_reversion",
        "feature_taker_flow_exhaustion_reversal",
        "feature_bbo_flow_exhaustion_reversal",
    ]
    assert first == second
    assert first == "1d421663b0f9f785a18d69e5068c81f1816005598f5a378bfeb697239f2488f6"


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
        "datetime",
    ]
    assert loaded["taker_buy_quote_volume"].isna().all()
    assert loaded["taker_sell_quote_volume"].isna().all()
    assert loaded["liquidation_long_notional"].isna().all()
    assert loaded["liquidation_short_notional"].isna().all()
    assert loaded["bbo_spread_bps"].isna().all()


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
    loaded = json.loads(Path(payload["output_paths"]["json"]).read_text())
    assert (
        loaded["pre_registered_search_space_sha256"]
        == payload["pre_registered_search_space_sha256"]
    )
