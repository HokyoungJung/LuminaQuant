from __future__ import annotations

from datetime import UTC, date, datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "scripts" / "research" / "replay_profit_moonshot_fresh_start.py"
SPEC = importlib.util.spec_from_file_location("replay_profit_moonshot_fresh_start", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load replay_profit_moonshot_fresh_start module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from scripts.research.replay_eth_shock_filters import _materialized_paths
from scripts.research.replay_profit_moonshot_fresh_start import FreshSpec


def test_fresh_start_hourly_loader_matches_raw_first_1s_aggregation_when_data_exists() -> None:
    market_root = Path("data/market_parquet")
    day = date(2026, 5, 5)
    one_second_paths = _materialized_paths(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1s",
        start=day,
        end=day,
    )
    one_hour_paths = _materialized_paths(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="1h",
        start=day,
        end=day,
    )
    if not one_second_paths or not one_hour_paths:
        pytest.skip("raw-first materialized fixture data is not present")

    loaded = MODULE._load_symbol_hourly(
        market_root=market_root,
        exchange="binance",
        symbol="BTC/USDT",
        start=day,
        end=day,
    )
    aggregated = (
        pl.scan_parquet(one_second_paths)
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .with_columns(pl.col("datetime").cast(pl.Datetime(time_unit="ms")))
        .sort("datetime")
        .group_by_dynamic("datetime", every="1h", period="1h", closed="left", label="left")
        .agg(
            [
                pl.col("open").first().alias("btcusdt_open"),
                pl.col("high").max().alias("btcusdt_high"),
                pl.col("low").min().alias("btcusdt_low"),
                pl.col("close").last().alias("btcusdt_close"),
                pl.col("volume").sum().alias("btcusdt_volume"),
            ]
        )
        .drop_nulls(["btcusdt_open", "btcusdt_close"])
        .collect()
        .sort("datetime")
    )

    assert loaded.height == aggregated.height
    assert (
        loaded.select("datetime", "btcusdt_close").to_dicts()
        == aggregated.select("datetime", "btcusdt_close").to_dicts()
    )


def test_candidate_signal_adaptive_trend_and_persistence_rules() -> None:
    arrays = {
        "datetime": [datetime(2026, 5, 1, h, tzinfo=UTC) for h in range(6)],
        "symbols": ("BTC/USDT",),
        "btcusdt_open": [100.0] * 6,
        "btcusdt_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "btcusdt_ret_12h": np.array([np.nan] * 5 + [0.05]),
        "btcusdt_ret_3h": np.array([np.nan, np.nan, np.nan, 0.01, 0.01, 0.01]),
        "btcusdt_resid_z_12h": [np.nan] * 6,
        "btcusdt_resid_z_3h": [np.nan] * 6,
        "btcusdt_funding_ffill": [0.0] * 6,
        "btcusdt_open_interest_ffill": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
        "market_ret_12h": np.array([np.nan] * 5 + [0.05]),
        "btcusdt_flow_imbalance_3h": [0.25] * 6,
        "btcusdt_flow_imbalance_2h": [0.25] * 6,
    }
    arrays["btcusdt_oi_delta_12h"] = np.full(6, 0.02)
    arrays["btcusdt_ret_6h"] = np.array([np.nan] * 5 + [0.03])
    arrays["market_ret_6h"] = np.array([np.nan] * 5 + [0.02])
    spec = FreshSpec(
        name="adaptive_trend_test",
        family="adaptive_trend",
        lookback_bars=12,
        threshold=0.01,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        adaptive_lookback_bars=12,
    )
    symbol, side, reason = MODULE._candidate_signal(spec, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"

    persistence = FreshSpec(
        name="flow_persistence_test",
        family="flow_imbalance_persistence",
        lookback_bars=3,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        flow_lookback_bars=2,
        flow_threshold=0.0,
        flow_persistence_bars=3,
        flow_persistence_threshold=0.2,
        min_abs_return=0.001,
    )
    arrays["btcusdt_ret_3h"] = np.array([np.nan, np.nan, 0.002, 0.003, 0.003, 0.003])
    arrays["btcusdt_resid_z_3h"] = np.array([np.nan] * 6)
    symbol, side, reason = MODULE._candidate_signal(persistence, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"


def test_candidate_signal_cross_sharpe_rank_and_funding_oi() -> None:
    dt = [datetime(2026, 5, 1, h, tzinfo=UTC) for h in range(6)]
    arrays = {
        "datetime": dt,
        "symbols": ("BTC/USDT", "ETH/USDT"),
        "btcusdt_open": [100.0] * 6,
        "btcusdt_close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "ethusdt_open": [50.0] * 6,
        "ethusdt_close": [50.0, 49.0, 49.0, 49.0, 49.0, 49.0],
        "btcusdt_ret_6h": np.array([np.nan] * 4 + [0.02, 0.03]),
        "ethusdt_ret_6h": np.array([np.nan] * 4 + [-0.01, -0.03]),
        "btcusdt_resid_z_6h": [np.nan] * 6,
        "ethusdt_resid_z_6h": [np.nan] * 6,
        "btcusdt_funding_ffill": [0.0] * 6,
        "ethusdt_funding_ffill": [0.0] * 6,
        "btcusdt_oi_delta_12h": np.array([np.nan] * 5 + [0.5]),
        "ethusdt_oi_delta_12h": np.array([np.nan] * 5 + [-0.5]),
        "btcusdt_open_interest_ffill": [20.0] * 6,
        "ethusdt_open_interest_ffill": [20.0] * 6,
        "btcusdt_flow_imbalance_3h": [0.0] * 6,
        "ethusdt_flow_imbalance_3h": [0.0] * 6,
        "market_ret_24h": np.array([np.nan] * 6),
        "market_ret_6h": np.array([np.nan] * 5 + [0.005]),
    }
    arrays["btcusdt_ret_24h"] = np.array([np.nan] * 5 + [0.03])
    arrays["ethusdt_ret_24h"] = np.array([np.nan] * 5 + [-0.03])
    arrays["btcusdt_ret_3h"] = np.array([np.nan] * 3 + [0.01, 0.01, 0.01])
    arrays["ethusdt_ret_3h"] = np.array([np.nan] * 3 + [0.01, 0.01, 0.01])
    arrays["btcusdt_resid_z_3h"] = np.array([np.nan] * 3 + [0.4, 0.6, 0.8])
    arrays["ethusdt_resid_z_3h"] = np.array([np.nan] * 3 + [0.1, 0.2, 0.3])

    cross = FreshSpec(
        name="cross_sharpe_rank",
        family="cross_sectional_sharpe_rank",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        sharpe_lookback_bars=6,
        sharpe_rank_min=0.0,
        min_abs_return=0.001,
    )
    symbol, side, reason = MODULE._candidate_signal(cross, arrays, 5)
    assert (symbol, side) == ("BTC/USDT", "LONG")
    assert reason == ""

    oi = FreshSpec(
        name="funding_oi",
        family="funding_oi_carry_fade",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=1,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        funding_rank_min=0.01,
        oi_rank_min=0.1,
        sharpe_lookback_bars=12,
        min_abs_return=0.001,
    )
    arrays["btcusdt_funding_ffill"] = [0.02] * 6
    arrays["ethusdt_funding_ffill"] = [-0.02] * 6
    symbol, side, reason = MODULE._candidate_signal(oi, arrays, 5)
    assert (symbol, side) in {("BTC/USDT", "SHORT"), ("ETH/USDT", "LONG")}
    assert reason == ""


def test_candidate_signal_new_inverse_families() -> None:
    dt = [datetime(2026, 5, 1, h, tzinfo=UTC) for h in range(8)]
    arrays = {
        "datetime": dt,
        "symbols": ("BTC/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "ethusdt"),
        "btcusdt_close": np.array([100.0] * 8),
        "ethusdt_close": np.array([50.0] * 8),
        "btcusdt_ret_6h": np.array([np.nan] * 7 + [0.03]),
        "ethusdt_ret_6h": np.array([np.nan] * 7 + [-0.03]),
        "btcusdt_ret_12h": np.array([np.nan] * 7 + [0.04]),
        "ethusdt_ret_12h": np.array([np.nan] * 7 + [-0.04]),
        "btcusdt_resid_z_6h": np.array([np.nan] * 7 + [2.0]),
        "ethusdt_resid_z_6h": np.array([np.nan] * 7 + [-2.0]),
        "btcusdt_funding_ffill": np.array([0.0002] * 8),
        "ethusdt_funding_ffill": np.array([-0.0002] * 8),
        "btcusdt_flow_imbalance_3h": np.zeros(8),
        "ethusdt_flow_imbalance_3h": np.zeros(8),
        "btcusdt_rv_24h": np.array([0.005] * 8),
        "ethusdt_rv_24h": np.array([0.005] * 8),
        "btcusdt_rv_24h_mean_72h": np.array([0.010] * 8),
        "ethusdt_rv_24h_mean_72h": np.array([0.010] * 8),
        "market_ret_6h": np.array([np.nan] * 7 + [0.02]),
        "market_ret_12h": np.array([np.nan] * 7 + [0.03]),
    }

    residual = FreshSpec(
        name="residual_momentum",
        family="residual_momentum",
        lookback_bars=6,
        threshold=1.5,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        min_abs_return=0.01,
    )
    assert MODULE._candidate_signal(residual, arrays, 7) == ("BTC/USDT", "LONG", "")

    trend_fade = FreshSpec(
        name="adaptive_trend_fade",
        family="adaptive_trend_fade",
        lookback_bars=12,
        adaptive_lookback_bars=12,
        threshold=0.01,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        min_abs_return=0.01,
    )
    assert MODULE._candidate_signal(trend_fade, arrays, 7) == ("BTC/USDT", "SHORT", "")

    compression_fade = FreshSpec(
        name="compression_fade",
        family="compression_breakout_fade",
        lookback_bars=6,
        threshold=0.01,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        rv_lookback_bars=24,
        compression_quantile=0.70,
    )
    assert MODULE._candidate_signal(compression_fade, arrays, 7) == ("BTC/USDT", "SHORT", "")

    funding_momentum = FreshSpec(
        name="funding_momentum",
        family="funding_carry_momentum",
        lookback_bars=6,
        threshold=1.5,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        funding_rank_min=0.0001,
        min_abs_return=0.01,
    )
    assert MODULE._candidate_signal(funding_momentum, arrays, 7) == ("BTC/USDT", "LONG", "")

    sharpe_reversal = FreshSpec(
        name="sharpe_reversal",
        family="cross_sectional_sharpe_reversal",
        lookback_bars=6,
        sharpe_lookback_bars=6,
        sharpe_rank_min=0.01,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        min_abs_return=0.01,
    )
    assert MODULE._candidate_signal(sharpe_reversal, arrays, 7) == ("BTC/USDT", "SHORT", "")


def test_candidate_signal_flow_imbalance_persistence_and_exhaustion() -> None:
    dt = [datetime(2026, 5, 1, h, tzinfo=UTC) for h in range(6)]
    arrays = {
        "datetime": dt,
        "symbols": ("BTC/USDT",),
        "btcusdt_open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "btcusdt_high": [101.0] * 6,
        "btcusdt_low": [99.0] * 6,
        "btcusdt_close": [100.0, 101.0, 100.0, 101.0, 100.0, 101.0],
        "btcusdt_volume": [1000.0] * 6,
        "btcusdt_funding_ffill": [0.0] * 6,
        "btcusdt_open_interest_ffill": [10.0] * 6,
        "btcusdt_oi_delta_12h": np.full(6, 0.1),
        "market_ret_6h": np.array([np.nan, np.nan, 0.003, 0.003, 0.004, 0.004]),
        "btcusdt_ret_6h": np.array([np.nan, np.nan, 0.0, 0.0, -0.003, -0.002]),
        "btcusdt_flow_imbalance_3h": np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06]),
        "btcusdt_flow_imbalance_6h": np.array([np.nan, np.nan, 0.04, 0.05, 0.04, 0.03]),
        "btcusdt_flow_imbalance_12h": np.array([np.nan] * 6),
        "btcusdt_resid_z_6h": np.array([np.nan] * 6),
        "btcusdt_resid_z_12h": np.array([np.nan] * 6),
        "btcusdt_ret_3h": np.array([np.nan, np.nan, 0.002, 0.002, 0.002, 0.002]),
        "btcusdt_ret_12h": np.array([np.nan, np.nan, 0.0, 0.0, -0.0, 0.0]),
    }

    persistence = FreshSpec(
        name="flow_imbalance_persistence_smoke",
        family="flow_imbalance_persistence",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=6,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        flow_lookback_bars=6,
        flow_threshold=0.03,
        flow_persistence_bars=3,
        flow_persistence_threshold=0.03,
    )
    symbol, side, reason = MODULE._candidate_signal(persistence, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"

    exhaustion = FreshSpec(
        name="flow_imbalance_exhaustion_smoke",
        family="flow_imbalance_exhaustion",
        lookback_bars=6,
        threshold=0.0,
        hold_bars=6,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.01,
        flow_lookback_bars=3,
        flow_threshold=0.0,
        flow_persistence_threshold=0.02,
        min_abs_return=0.001,
    )
    arrays["btcusdt_flow_imbalance_3h"] = np.array([np.nan, np.nan, -0.05, -0.05, -0.06, -0.07])
    arrays["btcusdt_ret_6h"] = np.array([np.nan, np.nan, -0.003, -0.0025, -0.004, -0.0035])
    symbol, side, reason = MODULE._candidate_signal(exhaustion, arrays, 5)
    assert reason == ""
    assert symbol == "BTC/USDT"
    assert side == "LONG"


def test_candidate_signal_calendar_rotation_selects_month_side() -> None:
    arrays = {
        "datetime": [
            datetime(2026, 1, 15, tzinfo=UTC),
            datetime(2026, 3, 15, tzinfo=UTC),
        ],
        "symbols": ("BTC/USDT", "TRX/USDT"),
        "symbol_prefixes": ("btcusdt", "trxusdt"),
        "btcusdt_close": np.array([100.0, 100.0]),
        "trxusdt_close": np.array([20.0, 20.0]),
        "btcusdt_ret_72h": np.array([-0.05, 0.01]),
        "trxusdt_ret_72h": np.array([-0.01, 0.04]),
        "market_ret_72h": np.array([-0.03, 0.02]),
    }
    spec = FreshSpec(
        name="calendar_rotation",
        family="calendar_rotation",
        lookback_bars=72,
        threshold=0.002,
        hold_bars=48,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        min_abs_return=0.002,
        calendar_long_months=(3, 4, 5),
        calendar_short_months=(1, 2),
    )

    assert MODULE._candidate_signal(spec, arrays, 0) == ("BTC/USDT", "SHORT", "")
    assert MODULE._candidate_signal(spec, arrays, 1) == ("TRX/USDT", "LONG", "")

    fixed_long = FreshSpec(
        name="calendar_rotation_fixed",
        family="calendar_rotation",
        lookback_bars=72,
        threshold=0.002,
        hold_bars=48,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        min_abs_return=0.002,
        calendar_long_months=(3, 4, 5),
        calendar_short_months=(1, 2),
        calendar_long_symbol="BTCUSDT",
    )
    assert MODULE._candidate_signal(fixed_long, arrays, 1) == ("BTC/USDT", "LONG", "")


def test_state_proxy_signals_do_not_depend_on_calendar_month() -> None:
    arrays = {
        "datetime": [
            datetime(2026, 1, 15, tzinfo=UTC),
            datetime(2026, 6, 15, tzinfo=UTC),
        ],
        "symbols": ("TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("trxusdt", "ethusdt"),
        "trxusdt_close": np.array([20.0, 20.0]),
        "ethusdt_close": np.array([100.0, 100.0]),
        "trxusdt_ret_168h": np.array([0.025, 0.025]),
        "ethusdt_ret_168h": np.array([-0.020, -0.020]),
        "trxusdt_resid_z_168h": np.array([1.25, 1.25]),
        "ethusdt_resid_z_168h": np.array([-1.10, -1.10]),
        "trxusdt_flow_imbalance_6h": np.array([0.04, 0.04]),
        "ethusdt_flow_imbalance_6h": np.array([-0.03, -0.03]),
        "trxusdt_funding_ffill": np.array([0.0, 0.0]),
        "ethusdt_funding_ffill": np.array([0.0, 0.0]),
        "market_ret_168h": np.array([0.01, 0.01]),
    }
    spec = FreshSpec(
        name="state_trx_momentum_proxy",
        family="state_momentum_proxy",
        lookback_bars=168,
        threshold=1.0,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        take_profit_pct=0.024,
        min_abs_return=0.015,
        primary_symbol="TRXUSDT",
        secondary_symbol="ETHUSDT",
        flow_lookback_bars=6,
        flow_threshold=0.03,
    )

    assert MODULE._candidate_signal(spec, arrays, 0) == ("TRX/USDT", "LONG", "")
    assert MODULE._candidate_signal(spec, arrays, 1) == ("TRX/USDT", "LONG", "")

    spread = FreshSpec(
        name="state_trx_eth_spread_proxy",
        family="state_relative_strength_spread",
        lookback_bars=168,
        threshold=0.015,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.024,
        min_abs_return=0.015,
        primary_symbol="TRXUSDT",
        secondary_symbol="ETHUSDT",
        flow_lookback_bars=6,
        flow_threshold=0.03,
        sharpe_rank_min=1.0,
        spread_hedge_ratio=1.0,
    )
    assert MODULE._state_relative_strength_spread_signal(spread, arrays, 0) == (
        "TRX/USDT",
        "ETH/USDT",
        "LONG_SPREAD",
        "",
    )


def test_state_distilled_leadership_unwind_is_dynamic_and_calendar_free() -> None:
    arrays = {
        "datetime": [
            datetime(2026, 1, 15, tzinfo=UTC),
            datetime(2026, 6, 15, tzinfo=UTC),
            datetime(2026, 6, 16, tzinfo=UTC),
        ],
        "symbols": ("BTC/USDT", "SOL/USDT", "TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "solusdt", "trxusdt", "ethusdt"),
        "btcusdt_close": np.array([100.0, 100.0, 100.0]),
        "solusdt_close": np.array([40.0, 40.0, 40.0]),
        "trxusdt_close": np.array([20.0, 20.0, 20.0]),
        "ethusdt_close": np.array([50.0, 50.0, 50.0]),
        "btcusdt_ret_72h": np.array([0.006, 0.008, 0.032]),
        "solusdt_ret_72h": np.array([0.006, 0.008, 0.035]),
        "trxusdt_ret_72h": np.array([0.030, 0.030, 0.010]),
        "ethusdt_ret_72h": np.array([-0.018, -0.018, -0.012]),
        "btcusdt_ret_168h": np.array([0.006, 0.008, 0.034]),
        "solusdt_ret_168h": np.array([0.006, 0.008, 0.038]),
        "trxusdt_ret_168h": np.array([0.033, 0.033, 0.012]),
        "ethusdt_ret_168h": np.array([-0.020, -0.020, -0.015]),
        "btcusdt_resid_z_168h": np.array([0.20, 0.30, 1.60]),
        "solusdt_resid_z_168h": np.array([0.20, 0.30, 1.80]),
        "trxusdt_resid_z_168h": np.array([1.45, 1.45, 0.35]),
        "ethusdt_resid_z_168h": np.array([-1.25, -1.25, -0.80]),
        "btcusdt_flow_imbalance_6h": np.array([0.01, 0.01, 0.05]),
        "solusdt_flow_imbalance_6h": np.array([0.01, 0.01, 0.06]),
        "trxusdt_flow_imbalance_6h": np.array([0.06, 0.06, 0.00]),
        "ethusdt_flow_imbalance_6h": np.array([-0.04, -0.04, -0.02]),
        "btcusdt_funding_ffill": np.array([0.0000, 0.0000, 0.0000]),
        "solusdt_funding_ffill": np.array([0.0001, 0.0001, 0.0001]),
        "trxusdt_funding_ffill": np.array([0.0001, 0.0001, 0.0001]),
        "ethusdt_funding_ffill": np.array([0.0004, 0.0004, 0.0004]),
        "btcusdt_oi_delta_6h": np.array([0.00, 0.00, 0.01]),
        "solusdt_oi_delta_6h": np.array([0.00, 0.00, 0.01]),
        "trxusdt_oi_delta_6h": np.array([0.01, 0.01, 0.00]),
        "ethusdt_oi_delta_6h": np.array([0.03, 0.03, 0.03]),
        "market_ret_72h": np.array([0.010, 0.010, 0.012]),
        "market_ret_168h": np.array([0.006, 0.006, 0.008]),
    }
    spec = FreshSpec(
        name="state_distilled_dynamic",
        family="state_distilled_leadership_unwind",
        lookback_bars=168,
        adaptive_lookback_bars=72,
        threshold=1.0,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.060,
        min_abs_return=0.012,
        flow_lookback_bars=6,
        flow_threshold=0.03,
        sharpe_rank_min=1.0,
        oi_rank_min=0.02,
    )

    assert MODULE._candidate_signal(spec, arrays, 0) == ("TRX/USDT", "LONG", "")
    assert MODULE._candidate_signal(spec, arrays, 1) == ("TRX/USDT", "LONG", "")
    assert MODULE._candidate_signal(spec, arrays, 2) == ("SOL/USDT", "LONG", "")

    assert spec.calendar_long_months == ()
    assert spec.calendar_short_months == ()
    assert not spec.primary_symbol
    assert not spec.secondary_symbol

    arrays["external_risk_off_score_lag1"] = np.array([2.0, 2.0, -0.2])
    ext_gated = FreshSpec(
        name="state_distilled_external_risk_filter",
        family="state_distilled_external_risk_filter",
        lookback_bars=168,
        adaptive_lookback_bars=72,
        threshold=1.0,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.060,
        min_abs_return=0.012,
        flow_lookback_bars=6,
        flow_threshold=0.03,
        sharpe_rank_min=1.0,
        oi_rank_min=0.02,
        external_risk_max=1.25,
    )
    assert MODULE._candidate_signal(ext_gated, arrays, 0) == (
        "",
        "",
        "external_risk_long_block",
    )
    assert MODULE._candidate_signal(ext_gated, arrays, 2) == ("SOL/USDT", "LONG", "")


def test_state_distilled_leadership_unwind_can_short_crowded_laggard() -> None:
    arrays = {
        "datetime": [datetime(2026, 2, 15, tzinfo=UTC)],
        "symbols": ("BTC/USDT", "TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "trxusdt", "ethusdt"),
        "btcusdt_close": np.array([100.0]),
        "trxusdt_close": np.array([20.0]),
        "ethusdt_close": np.array([50.0]),
        "btcusdt_ret_72h": np.array([-0.008]),
        "trxusdt_ret_72h": np.array([-0.004]),
        "ethusdt_ret_72h": np.array([-0.035]),
        "btcusdt_ret_168h": np.array([-0.010]),
        "trxusdt_ret_168h": np.array([-0.006]),
        "ethusdt_ret_168h": np.array([-0.040]),
        "btcusdt_resid_z_168h": np.array([-0.20]),
        "trxusdt_resid_z_168h": np.array([0.10]),
        "ethusdt_resid_z_168h": np.array([-1.60]),
        "btcusdt_flow_imbalance_6h": np.array([-0.01]),
        "trxusdt_flow_imbalance_6h": np.array([0.01]),
        "ethusdt_flow_imbalance_6h": np.array([-0.07]),
        "btcusdt_funding_ffill": np.array([0.0000]),
        "trxusdt_funding_ffill": np.array([0.0001]),
        "ethusdt_funding_ffill": np.array([0.0005]),
        "btcusdt_oi_delta_6h": np.array([0.00]),
        "trxusdt_oi_delta_6h": np.array([0.00]),
        "ethusdt_oi_delta_6h": np.array([0.04]),
        "market_ret_72h": np.array([-0.020]),
        "market_ret_168h": np.array([-0.025]),
    }
    spec = FreshSpec(
        name="state_distilled_unwind",
        family="state_distilled_leadership_unwind",
        lookback_bars=168,
        adaptive_lookback_bars=72,
        threshold=1.0,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.060,
        min_abs_return=0.012,
        flow_lookback_bars=6,
        flow_threshold=0.03,
        sharpe_rank_min=1.0,
        oi_rank_min=0.02,
    )

    assert MODULE._candidate_signal(spec, arrays, 0) == ("ETH/USDT", "SHORT", "")


def test_state_distilled_crowded_unwind_v2_shorts_weakening_leader_without_calendar() -> None:
    arrays = {
        "datetime": [
            datetime(2026, 1, 15, tzinfo=UTC),
            datetime(2026, 6, 15, tzinfo=UTC),
            datetime(2026, 6, 16, tzinfo=UTC),
        ],
        "symbols": ("BTC/USDT", "SOL/USDT", "TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "solusdt", "trxusdt", "ethusdt"),
        "btcusdt_close": np.array([100.0, 100.0, 100.0]),
        "solusdt_close": np.array([40.0, 40.0, 40.0]),
        "trxusdt_close": np.array([20.0, 20.0, 20.0]),
        "ethusdt_close": np.array([50.0, 50.0, 50.0]),
        "btcusdt_ret_24h": np.array([0.004, 0.004, 0.003]),
        "solusdt_ret_24h": np.array([0.002, 0.002, 0.026]),
        "trxusdt_ret_24h": np.array([0.004, 0.004, 0.010]),
        "ethusdt_ret_24h": np.array([-0.001, -0.001, 0.005]),
        "btcusdt_ret_168h": np.array([0.006, 0.006, 0.004]),
        "solusdt_ret_168h": np.array([0.060, 0.060, 0.070]),
        "trxusdt_ret_168h": np.array([0.022, 0.022, 0.052]),
        "ethusdt_ret_168h": np.array([0.006, 0.006, 0.040]),
        "btcusdt_resid_z_168h": np.array([0.10, 0.10, 0.05]),
        "solusdt_resid_z_168h": np.array([2.20, 2.20, 2.05]),
        "trxusdt_resid_z_168h": np.array([0.60, 0.60, 1.90]),
        "ethusdt_resid_z_168h": np.array([-0.20, -0.20, 1.75]),
        "btcusdt_flow_imbalance_6h": np.array([0.00, 0.00, 0.00]),
        "solusdt_flow_imbalance_6h": np.array([-0.07, -0.07, 0.04]),
        "trxusdt_flow_imbalance_6h": np.array([0.01, 0.01, 0.02]),
        "ethusdt_flow_imbalance_6h": np.array([0.00, 0.00, 0.01]),
        "btcusdt_funding_ffill": np.array([0.0, 0.0, 0.0]),
        "solusdt_funding_ffill": np.array([0.00016, 0.00016, 0.00015]),
        "trxusdt_funding_ffill": np.array([0.00002, 0.00002, 0.00010]),
        "ethusdt_funding_ffill": np.array([0.0, 0.0, 0.00009]),
        "btcusdt_oi_delta_12h": np.array([0.00, 0.00, 0.00]),
        "solusdt_oi_delta_12h": np.array([0.040, 0.040, 0.030]),
        "trxusdt_oi_delta_12h": np.array([0.010, 0.010, 0.025]),
        "ethusdt_oi_delta_12h": np.array([0.000, 0.000, 0.020]),
        "market_ret_24h": np.array([0.001, 0.001, 0.000]),
        "market_ret_168h": np.array([0.010, 0.010, 0.008]),
    }
    spec = FreshSpec(
        name="state_crowded_unwind_v2",
        family="state_distilled_crowded_unwind_v2",
        lookback_bars=168,
        adaptive_lookback_bars=24,
        threshold=1.75,
        hold_bars=96,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.040,
        min_abs_return=0.020,
        allow_long=False,
        allow_short=True,
        flow_lookback_bars=6,
        flow_threshold=0.04,
        funding_rank_min=0.00010,
        oi_rank_min=0.020,
        sharpe_lookback_bars=12,
        sharpe_rank_min=0.20,
    )

    assert MODULE._candidate_signal(spec, arrays, 0) == ("SOL/USDT", "SHORT", "")
    assert MODULE._candidate_signal(spec, arrays, 1) == ("SOL/USDT", "SHORT", "")
    assert MODULE._candidate_signal(spec, arrays, 2) == ("SOL/USDT", "SHORT", "")
    assert spec.entry_hours == ()
    assert spec.entry_days_of_month == ()
    assert spec.calendar_long_months == ()
    assert spec.calendar_short_months == ()


def test_calendar_teacher_state_similarity_is_state_based_and_calendar_free() -> None:
    arrays = {
        "datetime": [
            datetime(2026, 1, 15, tzinfo=UTC),
            datetime(2026, 6, 15, tzinfo=UTC),
            datetime(2026, 6, 16, tzinfo=UTC),
        ],
        "symbols": ("BTC/USDT", "SOL/USDT", "TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "solusdt", "trxusdt", "ethusdt"),
        "btcusdt_close": np.array([100.0, 100.0, 100.0]),
        "solusdt_close": np.array([40.0, 40.0, 40.0]),
        "trxusdt_close": np.array([20.0, 20.0, 20.0]),
        "ethusdt_close": np.array([50.0, 50.0, 50.0]),
        "btcusdt_ret_24h": np.array([0.002, 0.002, -0.010]),
        "solusdt_ret_24h": np.array([0.010, 0.010, -0.006]),
        "trxusdt_ret_24h": np.array([0.018, 0.018, -0.008]),
        "ethusdt_ret_24h": np.array([-0.006, -0.006, -0.020]),
        "btcusdt_ret_168h": np.array([0.006, 0.006, -0.020]),
        "solusdt_ret_168h": np.array([0.018, 0.018, 0.008]),
        "trxusdt_ret_168h": np.array([0.032, 0.032, -0.010]),
        "ethusdt_ret_168h": np.array([-0.018, -0.018, -0.035]),
        "btcusdt_resid_z_168h": np.array([0.10, 0.10, -0.20]),
        "solusdt_resid_z_168h": np.array([0.70, 0.70, 0.50]),
        "trxusdt_resid_z_168h": np.array([1.20, 1.20, -0.30]),
        "ethusdt_resid_z_168h": np.array([-1.00, -1.00, -1.40]),
        "btcusdt_flow_imbalance_6h": np.array([0.00, 0.00, -0.02]),
        "solusdt_flow_imbalance_6h": np.array([0.02, 0.02, 0.00]),
        "trxusdt_flow_imbalance_6h": np.array([0.05, 0.05, -0.02]),
        "ethusdt_flow_imbalance_6h": np.array([-0.02, -0.02, -0.05]),
        "btcusdt_funding_ffill": np.array([0.0, 0.0, 0.0]),
        "solusdt_funding_ffill": np.array([0.00002, 0.00002, 0.00001]),
        "trxusdt_funding_ffill": np.array([0.00002, 0.00002, 0.00001]),
        "ethusdt_funding_ffill": np.array([0.00001, 0.00001, 0.00012]),
        "btcusdt_oi_delta_12h": np.array([0.00, 0.00, 0.00]),
        "solusdt_oi_delta_12h": np.array([0.01, 0.01, 0.01]),
        "trxusdt_oi_delta_12h": np.array([0.02, 0.02, 0.01]),
        "ethusdt_oi_delta_12h": np.array([0.01, 0.01, 0.04]),
        "market_ret_24h": np.array([0.004, 0.004, -0.015]),
        "market_ret_168h": np.array([0.006, 0.006, -0.020]),
        "external_risk_off_score_lag1": np.array([-0.5, -0.5, 1.4]),
    }
    spec = FreshSpec(
        name="calendar_teacher_state_similarity",
        family="calendar_teacher_state_similarity",
        lookback_bars=168,
        adaptive_lookback_bars=24,
        threshold=0.50,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.045,
        min_abs_return=0.006,
        flow_lookback_bars=6,
        flow_threshold=0.02,
        sharpe_lookback_bars=12,
        sharpe_rank_min=0.50,
        funding_rank_min=0.00010,
        oi_rank_min=0.012,
        broad_min_abs=0.006,
        external_risk_max=1.25,
    )

    assert MODULE._candidate_signal(spec, arrays, 0) == ("TRX/USDT", "LONG", "")
    assert MODULE._candidate_signal(spec, arrays, 1) == ("TRX/USDT", "LONG", "")
    assert MODULE._candidate_signal(spec, arrays, 2) == ("ETH/USDT", "SHORT", "")
    assert spec.entry_hours == ()
    assert spec.entry_days_of_month == ()
    assert spec.calendar_long_months == ()
    assert spec.calendar_short_months == ()
    assert not spec.primary_symbol
    assert not spec.secondary_symbol

    fade = FreshSpec(
        name="calendar_teacher_state_fade",
        family="calendar_teacher_state_fade",
        lookback_bars=168,
        adaptive_lookback_bars=24,
        threshold=0.50,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.045,
        min_abs_return=0.006,
        flow_lookback_bars=6,
        flow_threshold=0.02,
        sharpe_lookback_bars=12,
        sharpe_rank_min=0.50,
        funding_rank_min=0.00010,
        oi_rank_min=0.012,
        broad_min_abs=0.006,
        external_risk_max=1.25,
    )
    assert MODULE._candidate_signal(fade, arrays, 0) == ("TRX/USDT", "SHORT", "")
    assert MODULE._candidate_signal(fade, arrays, 2) == ("ETH/USDT", "LONG", "")


def test_state_distilled_non_calendar_families_reject_calendar_entry_fields() -> None:
    args = MODULE.parse_args(["--spec-family", "calendar_teacher_state_similarity"])
    invalid = FreshSpec(
        name="invalid_calendar_state_unwind",
        family="calendar_teacher_state_similarity",
        lookback_bars=168,
        threshold=1.75,
        hold_bars=96,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.040,
        entry_hours=(2, 10),
        entry_days_of_month=(15,),
        calendar_long_months=(3, 4, 5),
        calendar_short_months=(1, 2),
    )
    valid = FreshSpec(
        name="valid_state_unwind",
        family="calendar_teacher_state_similarity",
        lookback_bars=168,
        threshold=1.75,
        hold_bars=96,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.040,
    )

    filtered, metadata = MODULE._filter_specs([invalid, valid], args)

    assert [spec.name for spec in filtered] == ["valid_state_unwind"]
    assert metadata["rejected_non_calendar_spec_count"] == 1
    assert metadata["rejected_non_calendar_specs"] == {
        "invalid_calendar_state_unwind": [
            "entry_hours",
            "entry_days_of_month",
            "calendar_long_months",
            "calendar_short_months",
        ]
    }


def test_calendar_veto_and_day_window_do_not_promote_blocked_entries() -> None:
    arrays = {
        "datetime": [
            datetime(2026, 3, 15, 2, tzinfo=UTC),
            datetime(2026, 3, 16, 2, tzinfo=UTC),
        ],
        "symbols": ("BTC/USDT", "TRX/USDT"),
        "symbol_prefixes": ("btcusdt", "trxusdt"),
        "btcusdt_close": np.array([100.0, 100.0]),
        "trxusdt_close": np.array([20.0, 20.0]),
        "btcusdt_ret_168h": np.array([0.03, 0.03]),
        "trxusdt_ret_168h": np.array([0.04, 0.04]),
        "market_ret_168h": np.array([0.02, 0.02]),
        "market_ret_24h": np.array([0.01, 0.01]),
        "btcusdt_resid_z_168h": np.array([-2.0, -2.0]),
        "trxusdt_resid_z_168h": np.array([0.0, 0.0]),
        "btcusdt_funding_ffill": np.array([0.0, 0.0]),
        "trxusdt_funding_ffill": np.array([0.0, 0.0]),
        "btcusdt_flow_imbalance_6h": np.array([0.0, 0.0]),
        "trxusdt_flow_imbalance_6h": np.array([0.0, 0.0]),
    }
    fixed_veto = FreshSpec(
        name="calendar_veto_fixed",
        family="calendar_rotation",
        lookback_bars=168,
        threshold=0.018,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        take_profit_pct=0.060,
        min_abs_return=0.018,
        calendar_long_months=(3,),
        calendar_short_months=(1,),
        calendar_long_symbol="BTCUSDT",
        calendar_veto_resid_z=1.0,
    )
    assert MODULE._candidate_signal(fixed_veto, arrays, 0) == ("", "", "calendar_residual_veto")

    dynamic_veto = FreshSpec(
        name="calendar_veto_dynamic",
        family="calendar_rotation",
        lookback_bars=168,
        threshold=0.018,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        take_profit_pct=0.060,
        min_abs_return=0.018,
        calendar_long_months=(3,),
        calendar_short_months=(1,),
        calendar_veto_resid_z=1.0,
    )
    assert MODULE._candidate_signal(dynamic_veto, arrays, 0) == ("TRX/USDT", "LONG", "")

    day_window = FreshSpec(
        name="calendar_day_window",
        family="calendar_rotation",
        lookback_bars=168,
        threshold=0.018,
        hold_bars=120,
        cooldown_bars=0,
        stop_loss_pct=0.0,
        take_profit_pct=0.060,
        min_abs_return=0.018,
        entry_days_of_month=(16,),
        entry_hours=(2,),
        calendar_long_months=(3,),
        calendar_short_months=(1,),
        calendar_long_symbol="TRXUSDT",
    )
    assert MODULE._candidate_signal(day_window, arrays, 0) == ("", "", "entry_day_block")
    assert MODULE._candidate_signal(day_window, arrays, 1) == ("TRX/USDT", "LONG", "")


def test_calendar_spread_split_runs_two_legs_and_reports_equity() -> None:
    datetimes = [datetime(2026, 3, 1, hour, tzinfo=UTC) for hour in range(12)]
    timestamps = np.array([int(item.timestamp()) for item in datetimes])
    trx_close = np.linspace(100.0, 110.0, len(datetimes))
    eth_close = np.linspace(100.0, 94.0, len(datetimes))
    arrays = {
        "datetime": datetimes,
        "timestamp": timestamps,
        "symbols": ("TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("trxusdt", "ethusdt"),
        "trxusdt_open": trx_close,
        "trxusdt_high": trx_close * 1.001,
        "trxusdt_low": trx_close * 0.999,
        "trxusdt_close": trx_close,
        "trxusdt_volume": np.full(len(datetimes), 1_000.0),
        "ethusdt_open": eth_close,
        "ethusdt_high": eth_close * 1.001,
        "ethusdt_low": eth_close * 0.999,
        "ethusdt_close": eth_close,
        "ethusdt_volume": np.full(len(datetimes), 1_000.0),
        "trxusdt_ret_168h": np.full(len(datetimes), 0.03),
        "ethusdt_ret_168h": np.full(len(datetimes), -0.01),
    }
    spec = FreshSpec(
        name="calendar_spread_smoke",
        family="calendar_spread",
        lookback_bars=168,
        threshold=0.0,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.001,
        min_abs_return=0.0,
        calendar_long_months=(3,),
        calendar_short_months=(1,),
        calendar_long_symbol="TRXUSDT",
        calendar_short_symbol="ETHUSDT",
        long_allocation_scale=2.0,
        short_allocation_scale=2.0,
        spread_hedge_ratio=1.0,
    )
    result = MODULE._run_split(
        spec=spec,
        arrays=arrays,
        split=MODULE.SplitWindow(
            name="train", start=date(2026, 3, 1), end=date(2026, 3, 1), role="train"
        ),
        include_equity=True,
    )

    assert result["round_trips"] >= 1
    assert result["fills"] >= 4
    assert len(result["equity_history"]) == len(datetimes)
    assert result["metrics"]["total_return"] > 0.0


def test_residual_pair_reversion_spread_signal_selects_extreme_pair() -> None:
    datetimes = [datetime(2026, 5, 1, hour, tzinfo=UTC) for hour in range(8)]
    arrays = {
        "datetime": datetimes,
        "timestamp": np.array([int(item.timestamp()) for item in datetimes]),
        "symbols": ("BTC/USDT", "TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "trxusdt", "ethusdt"),
        "btcusdt_close": np.full(8, 100.0),
        "trxusdt_close": np.full(8, 50.0),
        "ethusdt_close": np.full(8, 75.0),
        "btcusdt_resid_z_24h": np.full(8, 2.2),
        "trxusdt_resid_z_24h": np.full(8, -2.4),
        "ethusdt_resid_z_24h": np.full(8, -0.2),
    }
    spec = FreshSpec(
        name="residual_pair_smoke",
        family="residual_pair_reversion_spread",
        lookback_bars=24,
        threshold=1.0,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )

    assert MODULE._residual_pair_spread_signal(spec, arrays, 7) == (
        "TRX/USDT",
        "BTC/USDT",
        "LONG_SPREAD",
        "",
    )


def test_residual_pair_momentum_spread_signal_selects_extreme_pair() -> None:
    datetimes = [datetime(2026, 5, 1, hour, tzinfo=UTC) for hour in range(8)]
    arrays = {
        "datetime": datetimes,
        "timestamp": np.array([int(item.timestamp()) for item in datetimes]),
        "symbols": ("BTC/USDT", "TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "trxusdt", "ethusdt"),
        "btcusdt_close": np.full(8, 100.0),
        "trxusdt_close": np.full(8, 50.0),
        "ethusdt_close": np.full(8, 75.0),
        "btcusdt_resid_z_24h": np.full(8, 2.2),
        "trxusdt_resid_z_24h": np.full(8, -2.4),
        "ethusdt_resid_z_24h": np.full(8, 0.2),
    }
    spec = FreshSpec(
        name="residual_pair_momentum_smoke",
        family="residual_pair_momentum_spread",
        lookback_bars=24,
        threshold=1.0,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )

    assert MODULE._residual_pair_momentum_spread_signal(spec, arrays, 7) == (
        "BTC/USDT",
        "TRX/USDT",
        "LONG_SPREAD",
        "",
    )


def test_residual_pair_reversion_spread_split_runs_two_legs_and_reports_equity() -> None:
    datetimes = [datetime(2026, 5, 1, hour, tzinfo=UTC) for hour in range(12)]
    timestamps = np.array([int(item.timestamp()) for item in datetimes])
    long_close = np.linspace(100.0, 106.0, len(datetimes))
    short_close = np.linspace(100.0, 95.0, len(datetimes))
    flat_close = np.full(len(datetimes), 100.0)
    arrays = {
        "datetime": datetimes,
        "timestamp": timestamps,
        "symbols": ("BTC/USDT", "TRX/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "trxusdt", "ethusdt"),
        "btcusdt_open": short_close,
        "btcusdt_high": short_close * 1.001,
        "btcusdt_low": short_close * 0.999,
        "btcusdt_close": short_close,
        "btcusdt_volume": np.full(len(datetimes), 1_000.0),
        "trxusdt_open": long_close,
        "trxusdt_high": long_close * 1.001,
        "trxusdt_low": long_close * 0.999,
        "trxusdt_close": long_close,
        "trxusdt_volume": np.full(len(datetimes), 1_000.0),
        "ethusdt_open": flat_close,
        "ethusdt_high": flat_close * 1.001,
        "ethusdt_low": flat_close * 0.999,
        "ethusdt_close": flat_close,
        "ethusdt_volume": np.full(len(datetimes), 1_000.0),
        "btcusdt_resid_z_24h": np.full(len(datetimes), 2.1),
        "trxusdt_resid_z_24h": np.full(len(datetimes), -2.3),
        "ethusdt_resid_z_24h": np.zeros(len(datetimes)),
    }
    spec = FreshSpec(
        name="residual_pair_spread_smoke",
        family="residual_pair_reversion_spread",
        lookback_bars=24,
        threshold=1.0,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.01,
        take_profit_pct=0.001,
        long_allocation_scale=2.0,
        short_allocation_scale=2.0,
        spread_hedge_ratio=1.0,
    )

    result = MODULE._run_split(
        spec=spec,
        arrays=arrays,
        split=MODULE.SplitWindow(
            name="train", start=date(2026, 5, 1), end=date(2026, 5, 1), role="train"
        ),
        include_equity=True,
    )

    assert result["round_trips"] >= 1
    assert result["fills"] >= 4
    assert len(result["equity_history"]) == len(datetimes)
    assert result["metrics"]["total_return"] > 0.0


def test_compression_expansion_downside_short_is_short_only() -> None:
    dt = [datetime(2026, 5, 1, hour, tzinfo=UTC) for hour in range(8)]
    arrays = {
        "datetime": dt,
        "symbols": ("BTC/USDT", "ETH/USDT"),
        "symbol_prefixes": ("btcusdt", "ethusdt"),
        "btcusdt_close": np.full(8, 100.0),
        "ethusdt_close": np.full(8, 50.0),
        "btcusdt_ret_12h": np.full(8, -0.02),
        "ethusdt_ret_12h": np.full(8, 0.03),
        "btcusdt_rv_24h": np.full(8, 0.004),
        "ethusdt_rv_24h": np.full(8, 0.004),
        "btcusdt_rv_24h_mean_72h": np.full(8, 0.010),
        "ethusdt_rv_24h_mean_72h": np.full(8, 0.010),
    }
    spec = FreshSpec(
        name="compression_downside_short_smoke",
        family="compression_expansion_downside_short",
        lookback_bars=12,
        threshold=0.01,
        hold_bars=4,
        cooldown_bars=0,
        stop_loss_pct=0.006,
        take_profit_pct=0.012,
        compression_quantile=0.55,
        allow_long=False,
        allow_short=True,
    )

    symbol, side, reason = MODULE._candidate_signal(spec, arrays, 7)
    assert reason == ""
    assert (symbol, side) == ("BTC/USDT", "SHORT")

    arrays["btcusdt_ret_12h"] = np.full(8, 0.02)
    symbol, side, reason = MODULE._candidate_signal(spec, arrays, 7)
    assert (symbol, side, reason) == ("", "", "signal_missing")


def test_spec_filters_are_train_validation_universe_controls() -> None:
    args = MODULE.parse_args(
        [
            "--spec-family",
            "calendar_spread",
            "--spec-name-contains",
            "spread",
            "--max-specs",
            "1",
        ]
    )
    specs = [
        FreshSpec(
            name="fresh_calendar_trx_daywin_mid",
            family="calendar_rotation",
            lookback_bars=168,
            threshold=0.018,
            hold_bars=120,
            cooldown_bars=0,
            stop_loss_pct=0.0,
            take_profit_pct=0.060,
        ),
        FreshSpec(
            name="fresh_calendar_spread_trx_eth",
            family="calendar_spread",
            lookback_bars=168,
            threshold=0.018,
            hold_bars=120,
            cooldown_bars=0,
            stop_loss_pct=0.006,
            take_profit_pct=0.024,
        ),
    ]

    filtered, metadata = MODULE._filter_specs(specs, args)

    assert [spec.name for spec in filtered] == ["fresh_calendar_spread_trx_eth"]
    assert metadata["unfiltered_spec_count"] == 2
    assert metadata["filtered_spec_count"] == 1
    assert metadata["spec_family"] == ["calendar_spread"]


def test_external_state_panel_is_lagged_and_joined_to_hourly_panel(tmp_path: Path) -> None:
    module_path = ROOT / "scripts" / "research" / "fetch_profit_moonshot_external_state.py"
    spec = importlib.util.spec_from_file_location(
        "fetch_profit_moonshot_external_state", module_path
    )
    assert spec is not None and spec.loader is not None
    external_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external_module)

    dates = pl.date_range(date(2025, 1, 1), date(2025, 4, 30), "1d", eager=True)
    frames = [
        pl.DataFrame({"date": dates, "usd_broad": np.linspace(100.0, 112.0, len(dates))}),
        pl.DataFrame({"date": dates, "vix": np.linspace(12.0, 24.0, len(dates))}),
        pl.DataFrame({"date": dates, "ust2y": np.linspace(4.0, 3.8, len(dates))}),
        pl.DataFrame({"date": dates, "ust10y": np.linspace(4.5, 4.2, len(dates))}),
        pl.DataFrame({"date": dates, "wti": np.linspace(80.0, 70.0, len(dates))}),
    ]
    external = external_module._build_external_state_panel(frames)

    assert "external_risk_off_score_lag1" in external.columns
    assert external["external_risk_off_score_lag1"][0] is None
    assert external["external_risk_off_score_lag1"][80] == external["external_risk_off_score"][79]

    csv_path = tmp_path / "external.csv"
    external.write_csv(csv_path)
    hourly = pl.DataFrame(
        {
            "datetime": [
                datetime(2025, 3, 25, 0, tzinfo=UTC),
                datetime(2025, 3, 25, 1, tzinfo=UTC),
            ],
            "btcusdt_close": [100.0, 101.0],
        }
    )
    joined, metadata = MODULE._join_external_state(hourly, csv_path)

    assert metadata["joined"] is True
    assert "external_risk_off_score_lag1" in joined.columns
    assert joined["external_risk_off_score_lag1"][0] == joined["external_risk_off_score_lag1"][1]


def test_external_state_forward_fill_is_staleness_bounded() -> None:
    module_path = ROOT / "scripts" / "research" / "fetch_profit_moonshot_external_state.py"
    spec = importlib.util.spec_from_file_location(
        "fetch_profit_moonshot_external_state", module_path
    )
    assert spec is not None and spec.loader is not None
    external_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(external_module)

    panel = external_module._stale_bounded_daily_forward_fill(
        pl.DataFrame(
            {
                "date": [date(2025, 1, 1), date(2025, 1, 15)],
                "vix": [12.0, None],
            }
        ),
        ["vix"],
    )

    assert panel["vix"].to_list() == [12.0, None]


def test_candidate_specs_include_external_inspired_families() -> None:
    arrays = {
        "btcusdt_rv_24h": np.linspace(0.002, 0.003, 6),
        "btcusdt_ret_6h": np.zeros(6),
        "btcusdt_ret_12h": np.zeros(6),
        "btcusdt_ret_24h": np.zeros(6),
        "btcusdt_ret_48h": np.zeros(6),
        "btcusdt_ret_72h": np.zeros(6),
    }
    specs = MODULE._candidate_specs(arrays=arrays, symbols=["BTC/USDT"])
    families = {spec.family for spec in specs}
    assert "adaptive_trend" in families
    assert "cross_sectional_sharpe_rank" in families
    assert "flow_imbalance_persistence" in families
    assert "flow_imbalance_exhaustion" in families
    assert "funding_oi_carry_fade" in families
    assert "state_distilled_leadership_unwind" in families
    assert "state_distilled_crowded_unwind_v2" in families
    assert "state_distilled_external_risk_filter" in families
    assert "calendar_teacher_state_similarity" in families
    assert "calendar_teacher_state_fade" in families
    assert "state_momentum_proxy" in families
    assert "state_relative_strength_spread" in families
    assert "calendar_rotation" in families
    assert "residual_momentum" in families
    assert "adaptive_trend_fade" in families
    assert "cross_sectional_sharpe_reversal" in families
    assert "compression_breakout_fade" in families
    assert "funding_carry_momentum" in families
    assert "funding_oi_exhaustion_reversal" in families
    assert "beta_residual_reversion" in families
    assert "dispersion_compression_breakout_unwind" in families
    assert "vol_regime_margin_scaled_momentum" in families
    assert "calendar_spread" in families
    assert "residual_pair_reversion_spread" in families
    assert "residual_pair_momentum_spread" in families
    assert "compression_expansion_downside_short" in families
    names = {spec.name for spec in specs}
    assert any(name.startswith("fresh_calendar_trx_veto_") for name in names)
    assert any(name.startswith("fresh_calendar_trx_daywin_") for name in names)
    assert any(name.startswith("fresh_state_trx_mom_") for name in names)
    assert any(name.startswith("fresh_state_distilled_both_") for name in names)
    assert any(name.startswith("fresh_state_distilled_longonly_") for name in names)
    assert any(name.startswith("fresh_state_crowded_unwind_v2_") for name in names)
    assert any(name.startswith("fresh_state_distilled_ext_") for name in names)
    assert any(name.startswith("fresh_calendar_teacher_state_") for name in names)
    assert any(name.startswith("fresh_calendar_teacher_fade_") for name in names)
    assert any(name.startswith("fresh_state_trx_longonly_") for name in names)
    assert any(name.startswith("fresh_state_trx_dual_mom_") for name in names)
    assert any(name.startswith("fresh_state_trx_eth_spread_") for name in names)
    assert any(name.startswith("fresh_pair_resid_revert_spread_") for name in names)
    assert any(name.startswith("fresh_pair_resid_mom_spread_") for name in names)
    assert any(name.startswith("fresh_compression_downside_short_") for name in names)
    assert any(name.startswith("fresh_funding_oi_exhaustion_rev_") for name in names)
    assert any(name.startswith("fresh_beta_resid_reversion_") for name in names)
    assert any(name.startswith("fresh_dispersion_compression_state_") for name in names)
    assert any(name.startswith("fresh_vol_regime_margin_scaled_") for name in names)
    for spec in specs:
        if spec.family in MODULE.NON_CALENDAR_SPEC_FAMILIES:
            assert spec.entry_hours == ()
            assert spec.entry_days_of_month == ()
            assert spec.calendar_long_months == ()
            assert spec.calendar_short_months == ()


def test_integrated_market_state_specs_have_clean_non_calendar_variants() -> None:
    arrays = {
        "btcusdt_rv_24h": np.linspace(0.002, 0.003, 6),
        "btcusdt_ret_6h": np.zeros(6),
        "btcusdt_ret_12h": np.zeros(6),
        "btcusdt_ret_24h": np.zeros(6),
        "btcusdt_ret_48h": np.zeros(6),
        "btcusdt_ret_72h": np.zeros(6),
    }
    specs = MODULE._candidate_specs(arrays=arrays, symbols=["BTC/USDT"])
    lane_predicates = {
        "crowded_leadership_unwind_v2": lambda spec: (
            spec.family == "state_distilled_crowded_unwind_v2"
        ),
        "funding_oi_exhaustion_reversal": lambda spec: (
            spec.family == "funding_oi_exhaustion_reversal"
        ),
        "beta_residual_reversion": lambda spec: spec.family == "beta_residual_reversion",
        "dispersion_compression_state": lambda spec: (
            spec.family == "dispersion_compression_breakout_unwind"
        ),
        "vol_regime_margin_scaled_momentum": lambda spec: (
            spec.family == "vol_regime_margin_scaled_momentum"
        ),
        "state_distilled_external_risk_filter": lambda spec: (
            spec.family == "state_distilled_external_risk_filter"
        ),
        "calendar_teacher_state_similarity": lambda spec: (
            spec.family == "calendar_teacher_state_similarity"
        ),
        "calendar_teacher_state_fade": lambda spec: spec.family == "calendar_teacher_state_fade",
    }

    for lane, predicate in lane_predicates.items():
        exposed = [spec for spec in specs if predicate(spec)]
        clean = [
            spec
            for spec in exposed
            if not (
                spec.calendar_long_months
                or spec.calendar_short_months
                or spec.entry_days_of_month
                or spec.entry_hours
            )
        ]
        assert exposed, lane
        assert clean, lane


def test_regime_exposure_scales_single_leg_replay_when_not_volume_capped() -> None:
    datetimes = [datetime(2026, 5, 1, hour, tzinfo=UTC) for hour in range(8)]
    timestamps = np.array([int(item.timestamp()) for item in datetimes])
    close = np.linspace(100.0, 108.0, len(datetimes))
    arrays = {
        "datetime": datetimes,
        "timestamp": timestamps,
        "symbols": ("BTC/USDT",),
        "symbol_prefixes": ("btcusdt",),
        "btcusdt_open": close,
        "btcusdt_high": close * 1.02,
        "btcusdt_low": close * 0.99,
        "btcusdt_close": close,
        "btcusdt_volume": np.full(len(datetimes), 100_000.0),
        "btcusdt_ret_6h": np.full(len(datetimes), 0.03),
        "btcusdt_resid_z_6h": np.full(len(datetimes), 2.0),
        "btcusdt_funding_ffill": np.zeros(len(datetimes)),
        "market_ret_6h": np.full(len(datetimes), 0.02),
    }

    def run_scaled(scale: float) -> dict[str, object]:
        spec = FreshSpec(
            name=f"scaled_residual_momentum_{scale}",
            family="residual_momentum",
            lookback_bars=6,
            threshold=1.0,
            hold_bars=3,
            cooldown_bars=0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            min_abs_return=0.01,
            allow_long=True,
            allow_short=False,
            long_allocation_scale=scale,
            short_allocation_scale=scale,
        )
        return MODULE._run_split(
            spec=spec,
            arrays=arrays,
            split=MODULE.SplitWindow(
                name="train",
                start=date(2026, 5, 1),
                end=date(2026, 5, 1),
                role="train",
            ),
            include_equity=True,
        )

    low = run_scaled(0.5)
    high = run_scaled(2.0)

    assert low["fills"] == high["fills"]
    assert high["final_equity"] > low["final_equity"]
    assert high["metrics"]["total_return"] > low["metrics"]["total_return"]


def test_joined_panel_reuses_cache_without_reloading_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"load": 0}
    datetimes = [datetime(2026, 1, 1, hour, tzinfo=UTC) for hour in range(2)]

    def _fake_load_symbol_hourly(**_kwargs: object) -> pl.DataFrame:
        calls["load"] += 1
        return pl.DataFrame(
            {
                "datetime": datetimes,
                "btcusdt_open": [100.0, 101.0],
                "btcusdt_high": [101.0, 102.0],
                "btcusdt_low": [99.0, 100.0],
                "btcusdt_close": [100.5, 101.5],
                "btcusdt_volume": [10.0, 11.0],
            }
        )

    def _fake_load_feature_hourly(**_kwargs: object) -> tuple[pl.DataFrame, dict[str, object]]:
        return pl.DataFrame({"datetime": []}, schema={"datetime": pl.Datetime(time_unit="ms")}), {
            "symbol": "BTC/USDT",
            "rows": 0,
        }

    monkeypatch.setattr(MODULE, "_load_symbol_hourly", _fake_load_symbol_hourly)
    monkeypatch.setattr(MODULE, "_load_feature_hourly", _fake_load_feature_hourly)

    panel, meta = MODULE._joined_panel(
        market_root=tmp_path / "market",
        exchange="binance",
        symbols=["BTC/USDT"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        cache_dir=tmp_path / "cache",
    )
    assert panel.height == 2
    assert meta["panel_cache"]["cache_hit"] is False
    assert calls["load"] == 1

    monkeypatch.setattr(MODULE, "_load_symbol_hourly", lambda **_kwargs: pytest.fail("cache miss"))
    cached, cached_meta = MODULE._joined_panel(
        market_root=tmp_path / "market",
        exchange="binance",
        symbols=["BTC/USDT"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
        cache_dir=tmp_path / "cache",
    )
    assert cached.height == 2
    assert cached_meta["panel_cache"]["cache_hit"] is True


class _FakeMemoryGuard:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.rss_log_path = output_dir / "_memory_guard" / "fresh_replay_rss_latest.jsonl"
        self.summary_path = output_dir / "_memory_guard" / "fresh_replay_memory_latest.json"
        self.checkpoints: list[tuple[str, dict[str, object] | None]] = []
        self.finalize_calls: list[dict[str, object]] = []
        self.released = False

    def checkpoint(self, event: str, context: dict[str, object] | None = None) -> None:
        self.checkpoints.append((event, context))

    def finalize(
        self,
        *,
        status: str,
        error: str | None = None,
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = {
            "status": status,
            "error": error,
            "context": context or {},
            "peak_rss_bytes": 256 * 1024 * 1024,
            "rss_log_path": str(self.rss_log_path),
        }
        self.finalize_calls.append(payload)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text("{}", encoding="utf-8")
        return payload

    def release(self) -> None:
        self.released = True


def _tiny_replay_payload() -> dict[str, object]:
    return {
        "artifact_kind": "profit_moonshot_fresh_start_overhaul_replay",
        "generated_at_utc": "2026-05-07T00:00:00Z",
        "market_root": "data/market_parquet",
        "exchange": "binance",
        "symbols": ["BTC/USDT"],
        "oos_end_date": "2026-05-06",
        "split_windows": [],
        "data_metadata": {},
        "gate_policy": {},
        "spec_count": 1,
        "replay_survivor_count": 0,
        "success_candidate_count": 0,
        "top_results": [],
        "peak_rss_mib": 1.0,
    }


def test_fresh_replay_main_wraps_execution_with_memory_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guards: list[_FakeMemoryGuard] = []
    captured: dict[str, object] = {}

    def _fake_acquire(**kwargs: object) -> _FakeMemoryGuard:
        captured.update(kwargs)
        guard = _FakeMemoryGuard(Path(kwargs["output_dir"]))
        guards.append(guard)
        return guard

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(MODULE, "build_payload", lambda args: (_tiny_replay_payload(), []))

    assert MODULE.main(["--output-dir", str(tmp_path / "external_overhaul")]) == 0

    payload = json.loads(
        (tmp_path / "external_overhaul" / "fresh_start_overhaul_replay_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert captured["run_name"] == MODULE.RUN_NAME
    assert captured["budget_bytes"] == MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    assert (
        payload["memory_policy"]["explicit_budget_bytes"]
        == MODULE.PORTFOLIO_FOLLOWUP_EXPLICIT_BUDGET_BYTES
    )
    assert payload["rss_log_path"].endswith("fresh_replay_rss_latest.jsonl")
    assert payload["memory_summary_path"].endswith("fresh_replay_memory_latest.json")
    assert payload["memory_summary"]["status"] == "completed"
    assert guards[0].checkpoints[0][0] == "start"
    assert guards[0].finalize_calls[0]["status"] == "completed"
    assert guards[0].released is True


def test_fresh_replay_main_finalizes_failed_guard_when_payload_build_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guards: list[_FakeMemoryGuard] = []

    def _fake_acquire(**kwargs: object) -> _FakeMemoryGuard:
        guard = _FakeMemoryGuard(Path(kwargs["output_dir"]))
        guards.append(guard)
        return guard

    def _boom(args: object) -> tuple[dict[str, object], list[dict[str, object]]]:
        raise RuntimeError("replay build failed")

    monkeypatch.setattr(MODULE, "acquire_portfolio_memory_guard", _fake_acquire)
    monkeypatch.setattr(MODULE, "build_payload", _boom)

    with pytest.raises(RuntimeError, match="replay build failed"):
        MODULE.main(["--output-dir", str(tmp_path / "external_overhaul")])

    assert guards[0].finalize_calls[0]["status"] == "failed"
    assert guards[0].finalize_calls[0]["error"] == "replay build failed"
    assert guards[0].released is True
