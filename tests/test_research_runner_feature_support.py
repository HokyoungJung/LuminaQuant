from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from lumina_quant.strategy_factory import research_runner
from lumina_quant.strategy_factory import research_run_support


def _minute_datetimes(length: int) -> np.ndarray:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    values = [start + timedelta(minutes=idx) for idx in range(length)]
    return pl.Series("datetime", values).to_numpy()


def _bundle(symbol: str, *, length: int = 420) -> research_runner.SeriesBundle:
    close = np.linspace(100.0, 120.0, length, dtype=float)
    return research_runner.SeriesBundle(
        symbol=symbol,
        timeframe="1m",
        datetime=_minute_datetimes(length),
        open=close - 0.2,
        high=close + 0.4,
        low=close - 0.5,
        close=close,
        volume=np.linspace(1_000.0, 1_400.0, length, dtype=float),
    )


def test_align_bundles_augments_feature_series_from_feature_cache():
    bundle = _bundle("BTC/USDT")
    datetimes = pl.Series("datetime", bundle.datetime)
    feature_frame = pl.DataFrame(
        {
            "datetime": datetimes,
            "funding_rate": np.linspace(0.00005, 0.00025, datetimes.len(), dtype=float),
            "funding_mark_price": np.linspace(100.1, 120.1, datetimes.len(), dtype=float),
            "funding_fee_rate": np.linspace(0.00005, 0.00025, datetimes.len(), dtype=float),
            "funding_fee_quote_per_unit": np.linspace(0.005, 0.03, datetimes.len(), dtype=float),
            "mark_price": np.linspace(100.1, 120.1, datetimes.len(), dtype=float),
            "index_price": np.linspace(100.0, 120.0, datetimes.len(), dtype=float),
            "open_interest": np.linspace(1_000_000.0, 1_600_000.0, datetimes.len(), dtype=float),
            "liquidation_long_qty": np.linspace(10.0, 20.0, datetimes.len(), dtype=float),
            "liquidation_short_qty": np.linspace(5.0, 8.0, datetimes.len(), dtype=float),
            "liquidation_long_notional": np.linspace(
                100_000.0, 140_000.0, datetimes.len(), dtype=float
            ),
            "liquidation_short_notional": np.linspace(
                30_000.0, 40_000.0, datetimes.len(), dtype=float
            ),
        }
    )

    aligned = research_runner._align_bundles(
        [bundle],
        feature_cache={"BTC/USDT": feature_frame},
    )

    assert aligned is not None
    assert "BTC/USDT:funding_rate" in aligned
    assert "BTC/USDT:open_interest" in aligned
    assert "BTC/USDT:crowding_score" in aligned
    assert np.isclose(float(aligned["BTC/USDT:funding_fee_rate"][-1]), 0.00025)
    assert np.isfinite(float(aligned["BTC/USDT:crowding_score"][-1]))


def test_research_run_uses_candidate_symbols_as_default_universe() -> None:
    adapted = [
        {
            "strategy_timeframe": "1h",
            "timeframe": "1h",
            "symbols": ["BTC/USDT", "ETH/USDT"],
        }
    ]

    _, universe = research_run_support._resolve_research_run_timeframes_and_universe(
        adapted=adapted,
        strategy_timeframes=None,
        symbol_universe=None,
    )

    assert universe == ["BTC/USDT", "ETH/USDT"]


def test_align_bundles_ignores_empty_feature_frames():
    bundle = _bundle("BTC/USDT")
    feature_frame = pl.DataFrame(
        schema={
            "datetime": pl.Datetime("ms"),
            **dict.fromkeys(research_runner._FEATURE_POINT_COLUMNS, pl.Float64),
        }
    )

    aligned = research_runner._align_bundles(
        [bundle],
        feature_cache={"BTC/USDT": feature_frame},
    )

    assert aligned is not None
    assert "BTC/USDT:open" in aligned
    assert "BTC/USDT:funding_rate" not in aligned
    assert "BTC/USDT:crowding_score" not in aligned


def test_load_candidate_signal_payload_reuses_alignment_cache(monkeypatch):
    bundle = _bundle("BTC/USDT")
    cache = {("BTC/USDT", "1m"): bundle}
    aligned_payload = {
        "datetime": bundle.datetime,
        "BTC/USDT:open": bundle.open,
        "BTC/USDT:high": bundle.high,
        "BTC/USDT:low": bundle.low,
        "BTC/USDT:close": bundle.close,
        "BTC/USDT:volume": bundle.volume,
    }
    calls = {"count": 0}

    def _fake_align(bundles, *, feature_cache=None):
        _ = (bundles, feature_cache)
        calls["count"] += 1
        return aligned_payload

    def _fake_strategy_signal(candidate, *, aligned, symbols):
        _ = (candidate, aligned, symbols)
        length = len(bundle.datetime)
        return (
            np.zeros(length, dtype=float),
            np.zeros(length, dtype=float),
            np.zeros(length, dtype=float),
            {},
        )

    monkeypatch.setattr(research_runner, "_align_bundles", _fake_align)
    monkeypatch.setattr(research_runner, "_strategy_signal", _fake_strategy_signal)

    candidate = {
        "strategy_class": "MovingAverageCrossStrategy",
        "strategy_timeframe": "1m",
        "symbols": ["BTC/USDT"],
    }
    aligned_cache: dict[tuple[object, ...], object] = {}

    first = research_runner._load_candidate_signal_payload(
        candidate,
        cache=cache,
        feature_cache=None,
        aligned_cache=aligned_cache,
    )
    second = research_runner._load_candidate_signal_payload(
        candidate,
        cache=cache,
        feature_cache=None,
        aligned_cache=aligned_cache,
    )

    assert first is not None and second is not None
    assert calls["count"] == 1


def test_rolling_z_matches_reference_loop():
    values = np.asarray([0.0, 0.2, -0.1, 0.3, 0.1, -0.2, 0.25, 0.4, -0.15, 0.05], dtype=float)
    window = 8
    expected = np.full(values.shape, np.nan, dtype=float)
    for idx in range(window, values.size + 1):
        tail = values[idx - window : idx]
        hist = tail[:-1]
        latest = tail[-1]
        std = research_runner._safe_std(hist)
        expected[idx - 1] = (
            0.0 if std <= 1e-12 else (float(latest) - research_runner._safe_mean(hist)) / std
        )

    actual = research_runner._rolling_z(values, window)
    np.testing.assert_allclose(actual[window - 1 :], expected[window - 1 :], rtol=1e-9, atol=1e-9)


def test_common_bundle_datetime_returns_none_for_empty_input():
    assert research_runner._common_bundle_datetime([]) is None


def test_perp_carry_strategy_signal_uses_actual_support_feature_arrays():
    length = 420
    close = np.linspace(100.0, 118.0, length, dtype=float)
    funding = 0.00005 + np.linspace(0.0, 0.00020, length, dtype=float)
    open_interest = 1_000_000.0 * np.exp(np.linspace(0.0, 0.25, length, dtype=float))
    liq_long = 120_000.0 + np.linspace(0.0, 40_000.0, length, dtype=float)
    liq_short = 20_000.0 + np.linspace(0.0, 5_000.0, length, dtype=float)

    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(1_000.0, 1_500.0, length, dtype=float),
        "BTC/USDT:funding_rate": funding,
        "BTC/USDT:open_interest": open_interest,
        "BTC/USDT:liquidation_long_notional": liq_long,
        "BTC/USDT:liquidation_short_notional": liq_short,
        "BTC/USDT:mark_price": close * 1.001,
        "BTC/USDT:index_price": close,
    }
    candidate = {
        "strategy_class": "PerpCrowdingCarryStrategy",
        "params": {
            "window": 96,
            "mild_funding": 0.0004,
            "extreme_funding": 0.002,
            "entry_threshold": 0.05,
            "exit_threshold": 0.02,
            "stop_loss_pct": 0.05,
            "max_hold_bars": 512,
            "allow_short": True,
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert returns_raw.shape == (length,)
    assert turnover.shape == (length,)
    assert exposure.shape == (length,)
    assert meta.get("missing_support_data") is not True
    assert np.any(np.abs(exposure) > 0.0)


def test_composite_trend_blocks_entries_when_crowding_score_is_extreme():
    length = 420
    close = np.linspace(100.0, 160.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.2,
        "BTC/USDT:high": close + 0.6,
        "BTC/USDT:low": close - 0.6,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(2_000.0, 4_000.0, length, dtype=float),
        "BTC/USDT:crowding_score": np.full(length, 1.0, dtype=float),
    }
    candidate = {
        "strategy_class": "CompositeTrendStrategy",
        "params": {
            "long_threshold": 0.10,
            "short_threshold": 0.10,
            "te_min": 0.0,
            "vr_min": 0.0,
            "crowding_reduce_threshold": 0.55,
            "crowding_block_threshold": 0.85,
        },
    }

    _, _, exposure, _ = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert np.allclose(exposure, 0.0)


def test_last_day_liquidity_regime_strategy_routes_liquid_momentum_and_illiquid_reversal():
    length = 120
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}

    close_map = {
        "BTC/USDT": np.linspace(100.0, 145.0, length, dtype=float),
        "ETH/USDT": np.linspace(140.0, 120.0, length, dtype=float),
        "BNB/USDT": np.linspace(80.0, 120.0, length, dtype=float),
        "TRX/USDT": np.linspace(90.0, 60.0, length, dtype=float),
    }
    volume_map = {
        "BTC/USDT": np.linspace(5_000.0, 6_000.0, length, dtype=float),
        "ETH/USDT": np.linspace(4_500.0, 5_500.0, length, dtype=float),
        "BNB/USDT": np.linspace(200.0, 250.0, length, dtype=float),
        "TRX/USDT": np.linspace(180.0, 220.0, length, dtype=float),
    }
    for symbol in symbols:
        close = close_map[symbol]
        aligned[f"{symbol}:open"] = close - 0.2
        aligned[f"{symbol}:high"] = close + 0.4
        aligned[f"{symbol}:low"] = close - 0.4
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = volume_map[symbol]

    candidate = {
        "strategy_class": "LastDayLiquidityRegimeStrategy",
        "params": {
            "momentum_lookback_bars": 24,
            "signal_skip_bars": 1,
            "liquidity_window": 24,
            "volatility_window": 24,
            "rebalance_bars": 6,
            "signal_threshold": 0.01,
            "liquidity_quantile": 0.50,
            "max_longs": 2,
            "max_shorts": 1,
            "min_price": 0.10,
            "max_realized_vol": 1.0,
            "stop_loss_pct": 0.05,
            "allow_short": True,
            "illiquid_reversal": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert turnover.shape == (length,)
    assert exposure.shape == (length,)
    assert meta.get("liquidity_conditioned") is True
    assert meta.get("signal_family") == "last_day_liquidity_regime"
    assert np.any(turnover > 0.0)


def test_abnormal_return_continuation_strategy_generates_event_exposure() -> None:
    length = 80
    symbols = ["BNB/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    close = np.concatenate(
        [
            np.linspace(100.0, 102.0, 20, dtype=float),
            np.linspace(102.0, 125.0, 5, dtype=float),
            np.linspace(125.0, 130.0, 20, dtype=float),
            np.linspace(130.0, 120.0, 35, dtype=float),
        ]
    )
    aligned["BNB/USDT:open"] = close - 0.2
    aligned["BNB/USDT:high"] = close + 0.4
    aligned["BNB/USDT:low"] = close - 0.4
    aligned["BNB/USDT:close"] = close
    aligned["BNB/USDT:volume"] = np.linspace(1_000.0, 1_500.0, length, dtype=float)

    candidate = {
        "strategy_class": "AbnormalReturnContinuationStrategy",
        "params": {
            "return_z_window": 12,
            "entry_z": 1.0,
            "exit_z": 0.15,
            "hold_bars": 2,
            "stop_loss_pct": 0.08,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert turnover.shape == (length,)
    assert exposure.shape == (length,)
    assert meta.get("signal_family") == "abnormal_return_continuation"
    assert np.any(turnover > 0.0)


def test_composite_trend_respects_long_only_retune_and_exposure_damping():
    length = 420
    close = np.linspace(160.0, 100.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close + 0.2,
        "BTC/USDT:high": close + 0.6,
        "BTC/USDT:low": close - 0.6,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(2_000.0, 4_000.0, length, dtype=float),
    }

    shortable = {
        "strategy_class": "CompositeTrendStrategy",
        "params": {
            "long_threshold": 0.10,
            "short_threshold": 0.10,
            "exit_score_cross": 0.03,
            "te_min": 0.0,
            "vr_min": 0.0,
            "risk_target_vol": 0.0025,
            "max_signal_strength": 0.60,
            "allow_short": True,
        },
    }
    long_only = {
        "strategy_class": "CompositeTrendStrategy",
        "params": {
            **dict(shortable["params"]),
            "allow_short": False,
        },
    }

    _, _, exposure_shortable, _ = research_runner._strategy_signal(
        shortable,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )
    _, _, exposure_long_only, _ = research_runner._strategy_signal(
        long_only,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert np.min(exposure_shortable) < -0.05
    assert np.max(np.abs(exposure_shortable)) <= 0.60 + 1e-9
    assert np.all(exposure_long_only >= -1e-9)
    assert np.max(np.abs(exposure_long_only)) < 0.05


def test_composite_trend_benchmark_gate_blocks_longs_in_benchmark_crash_state():
    length = 180
    symbols = ["BTC/USDT", "ETH/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}

    btc_close = np.linspace(160.0, 100.0, length, dtype=float)
    eth_close = np.concatenate(
        [
            np.linspace(100.0, 104.0, 60, dtype=float),
            np.linspace(104.0, 145.0, length - 60, dtype=float),
        ]
    )
    for symbol, close in {"BTC/USDT": btc_close, "ETH/USDT": eth_close}.items():
        aligned[f"{symbol}:open"] = close - 0.2
        aligned[f"{symbol}:high"] = close + 0.6
        aligned[f"{symbol}:low"] = close - 0.6
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.linspace(2_000.0, 4_000.0, length, dtype=float)

    base_params = {
        "long_threshold": 0.10,
        "short_threshold": 0.10,
        "exit_score_cross": 0.03,
        "te_min": 0.0,
        "vr_min": 0.0,
        "risk_target_vol": 0.0025,
        "max_signal_strength": 0.60,
        "allow_short": False,
    }
    ungated = {"strategy_class": "CompositeTrendStrategy", "params": dict(base_params)}
    gated = {
        "strategy_class": "CompositeTrendStrategy",
        "params": {
            **dict(base_params),
            "benchmark_regime_ma": 48,
            "benchmark_symbol": "BTC/USDT",
        },
    }

    _, _, exposure, _ = research_runner._strategy_signal(
        ungated,
        aligned=aligned,
        symbols=symbols,
    )
    _, _, gated_exposure, gated_meta = research_runner._strategy_signal(
        gated,
        aligned=aligned,
        symbols=symbols,
    )

    assert np.max(exposure) > 0.05
    assert np.allclose(gated_exposure, 0.0)
    assert gated_meta.get("crash_aware_gate") is True
    assert gated_meta.get("benchmark_symbol") == "BTC/USDT"
    assert gated_meta.get("benchmark_regime_ma") == 48


def test_composite_trend_helpers_cover_entry_and_exit_branches():
    config = research_runner._CompositeTrendStrategyConfig(
        long_threshold=0.10,
        short_threshold=0.10,
        exit_score_cross=0.03,
        te_min=0.0,
        vr_min=0.0,
        risk_target_vol=0.0025,
        max_signal_strength=0.60,
        vol_window=32,
        max_hold_bars=8,
        allow_short=True,
        benchmark_regime_ma=0,
        benchmark_symbol="BTC/USDT",
        crowding_reduce_threshold=0.55,
        crowding_block_threshold=0.85,
    )

    assert (
        research_runner._composite_trend_entry_mode(
            score_i=0.2,
            long_gate_i=True,
            short_gate_i=True,
            blocked=False,
            config=config,
        )
        == 1
    )
    assert (
        research_runner._composite_trend_entry_mode(
            score_i=-0.2,
            long_gate_i=True,
            short_gate_i=True,
            blocked=False,
            config=config,
        )
        == -1
    )
    assert (
        research_runner._composite_trend_should_exit(
            mode=1,
            score_i=0.01,
            long_gate_i=True,
            short_gate_i=True,
            bars_held=1,
            config=config,
        )
        is True
    )


def test_composite_trend_position_series_exits_active_long_and_short_positions():
    config = research_runner._CompositeTrendStrategyConfig(
        long_threshold=0.10,
        short_threshold=0.10,
        exit_score_cross=0.03,
        te_min=0.0,
        vr_min=0.0,
        risk_target_vol=0.0025,
        max_signal_strength=0.60,
        vol_window=2,
        max_hold_bars=8,
        allow_short=True,
        benchmark_regime_ma=0,
        benchmark_symbol="BTC/USDT",
        crowding_reduce_threshold=0.55,
        crowding_block_threshold=0.85,
    )
    gate = np.asarray([True, True, True], dtype=bool)
    close = np.asarray([100.0, 101.0, 102.0], dtype=float)

    long_position = research_runner._composite_trend_position_series(
        close=close,
        score=np.asarray([0.2, 0.01, 0.0], dtype=float),
        gate=gate,
        long_gate=gate,
        short_gate=gate,
        crowding=None,
        config=config,
    )
    short_position = research_runner._composite_trend_position_series(
        close=close,
        score=np.asarray([-0.2, -0.01, 0.0], dtype=float),
        gate=gate,
        long_gate=gate,
        short_gate=gate,
        crowding=None,
        config=config,
    )

    assert long_position[0] > 0.0
    assert np.array_equal(long_position[1:], np.zeros(2, dtype=float))
    assert short_position[0] < 0.0
    assert np.array_equal(short_position[1:], np.zeros(2, dtype=float))
    assert (
        research_runner._composite_trend_should_exit(
            mode=-1,
            score_i=-0.01,
            long_gate_i=True,
            short_gate_i=True,
            bars_held=1,
            config=config,
        )
        is True
    )


def test_composite_trend_position_series_falls_back_to_primary_gate_and_damps_crowding(
    monkeypatch,
):
    config = research_runner._CompositeTrendStrategyConfig(
        long_threshold=0.10,
        short_threshold=0.10,
        exit_score_cross=0.03,
        te_min=0.0,
        vr_min=0.0,
        risk_target_vol=0.006,
        max_signal_strength=0.60,
        vol_window=8,
        max_hold_bars=8,
        allow_short=True,
        benchmark_regime_ma=0,
        benchmark_symbol="BTC/USDT",
        crowding_reduce_threshold=0.55,
        crowding_block_threshold=0.85,
    )
    monkeypatch.setattr(
        research_runner,
        "_rolling_volatility_series",
        lambda closes, window: np.full(np.asarray(closes, dtype=float).shape, 0.01, dtype=float),
    )

    position = research_runner._composite_trend_position_series(
        close=np.asarray([100.0, 101.0, 102.0, 103.0], dtype=float),
        score=np.asarray([0.2, 0.2, 0.2, 0.2], dtype=float),
        gate=np.asarray([True, True, True, True], dtype=bool),
        long_gate=np.asarray([True], dtype=bool),
        short_gate=np.asarray([False], dtype=bool),
        crowding=np.asarray([0.0, 0.6, 0.6, 0.6], dtype=float),
        config=config,
    )

    assert np.array_equal(position, np.asarray([0.6, 0.3, 0.3, 0.3], dtype=float))


def test_lag_convergence_strategy_signal_trades_xpt_xpd_pair():
    length = 120
    x_close = np.full(length, 100.0, dtype=float)
    y_close = np.full(length, 100.0, dtype=float)
    x_close[10:] = 102.5
    y_close[10:] = 100.0
    x_close[40:] = 100.2
    y_close[40:] = 100.1

    aligned = {
        "datetime": _minute_datetimes(length),
        "XPT/USDT:open": x_close,
        "XPT/USDT:high": x_close + 0.2,
        "XPT/USDT:low": x_close - 0.2,
        "XPT/USDT:close": x_close,
        "XPT/USDT:volume": np.full(length, 100.0, dtype=float),
        "XPD/USDT:open": y_close,
        "XPD/USDT:high": y_close + 0.2,
        "XPD/USDT:low": y_close - 0.2,
        "XPD/USDT:close": y_close,
        "XPD/USDT:volume": np.full(length, 100.0, dtype=float),
    }
    candidate = {
        "strategy_class": "LagConvergenceStrategy",
        "params": {
            "symbol_x": "XPT/USDT",
            "symbol_y": "XPD/USDT",
            "lag_bars": 1,
            "entry_threshold": 0.01,
            "exit_threshold": 0.002,
            "stop_threshold": 0.05,
            "max_hold_bars": 24,
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["XPT/USDT", "XPD/USDT"],
    )

    assert returns_raw.shape == (length,)
    assert turnover.shape == (length,)
    assert exposure.shape == (length,)
    assert meta == {}
    assert np.allclose(exposure, 0.0)
    assert np.any(turnover > 0.0)


def test_breadth_thrust_failure_reversal_strategy_signal_produces_exposure():
    length = 60
    symbols = [f"S{idx}/USDT" for idx in range(5)]
    aligned = {"datetime": _minute_datetimes(length)}
    for idx, symbol in enumerate(symbols):
        close = np.full(length, 100.0, dtype=float)
        if idx < 4:
            close[20:25] = np.linspace(100.0, 101.0, 5, dtype=float)
            close[25:35] = 101.0
        else:
            close[20:25] = np.linspace(100.0, 88.0, 5, dtype=float)
            close[25:35] = 88.0

        if idx < 3:
            close[35:40] = np.linspace(close[34], 102.0, 5, dtype=float)
            close[40:] = 102.0
        else:
            close[35:40] = np.linspace(close[34], 98.0, 5, dtype=float)
            close[40:] = 98.0

        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 100.0, dtype=float)

    candidate = {
        "strategy_class": "BreadthThrustFailureReversalStrategy",
        "params": {
            "momentum_lookback": 4,
            "breadth_entry": 0.8,
            "breadth_exit": 0.6,
            "basket_return_floor": 0.003,
            "max_hold_bars": 20,
            "stop_loss_pct": 0.05,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (length,)
    assert meta.get("cross_sectional") is True
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)


def test_breadth_thrust_failure_reversal_position_series_holds_state_when_breadth_is_missing():
    close_map_np = {
        "S0/USDT": np.asarray([100.0, 100.0, 101.0, np.nan], dtype=float),
        "S1/USDT": np.asarray([100.0, 100.0, 101.0, np.nan], dtype=float),
        "S2/USDT": np.asarray([100.0, 100.0, 101.0, np.nan], dtype=float),
        "S3/USDT": np.asarray([100.0, 100.0, 101.0, np.nan], dtype=float),
        "S4/USDT": np.asarray([100.0, 100.0, 70.0, np.nan], dtype=float),
    }

    position = research_runner._breadth_thrust_failure_reversal_position_series(
        close_map_np=close_map_np,
        config=research_runner._BreadthThrustFailureReversalConfig(
            momentum_lookback=2,
            breadth_entry=0.8,
            breadth_exit=0.6,
            basket_return_floor=0.02,
            max_hold_bars=8,
            stop_loss_pct=0.05,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([0.0, 0.0, -1.0, -1.0], dtype=float))


def test_perp_carry_preserves_default_config(monkeypatch):
    captured: dict[str, float | int | bool] = {}

    def _stub_perp_carry_position_series(
        *,
        support_inputs,
        config,
    ):
        captured.update(
            {
                "window": int(config.window),
                "mild_funding": float(config.mild_funding),
                "extreme_funding": float(config.extreme_funding),
                "entry_threshold": float(config.entry_threshold),
                "exit_threshold": float(config.exit_threshold),
                "stop_loss_pct": float(config.stop_loss_pct),
                "max_hold_bars": int(config.max_hold_bars),
                "allow_short": bool(config.allow_short),
            }
        )
        shape = np.asarray(support_inputs.close, dtype=float).shape
        return np.zeros(shape, dtype=float), {"crowding_score": np.zeros(shape, dtype=float)}

    monkeypatch.setattr(research_runner, "_resolve_feature_points_path", lambda: Path(__file__))
    monkeypatch.setattr(
        research_runner,
        "_perp_carry_position_series",
        _stub_perp_carry_position_series,
    )

    length = 120
    close = np.linspace(100.0, 104.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
        "BTC/USDT:funding_rate": np.full(length, 0.0001, dtype=float),
        "BTC/USDT:open_interest": np.full(length, 1_000_000.0, dtype=float),
        "BTC/USDT:liquidation_long_notional": np.full(length, 120_000.0, dtype=float),
        "BTC/USDT:liquidation_short_notional": np.full(length, 20_000.0, dtype=float),
        "BTC/USDT:mark_price": close * 1.001,
        "BTC/USDT:index_price": close,
    }

    research_runner._strategy_signal(
        {"strategy_class": "PerpCrowdingCarryStrategy", "params": {}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert captured == {
        "window": 96,
        "mild_funding": 0.0002,
        "extreme_funding": 0.0012,
        "entry_threshold": 0.3,
        "exit_threshold": 0.1,
        "stop_loss_pct": 0.02,
        "max_hold_bars": 72,
        "allow_short": True,
    }


def test_perp_carry_position_series_shorts_on_crowded_positive_funding(monkeypatch):
    support = {
        "crowding_score": np.asarray([0.45, 0.45, 0.0], dtype=float),
        "oi_delta_z": np.asarray([1.2, 1.2, 0.0], dtype=float),
    }
    monkeypatch.setattr(research_runner, "_crowding_support_series", lambda **kwargs: support)

    position, returned_support = research_runner._perp_carry_position_series(
        support_inputs=research_runner._CrowdingSupportInputs(
            close=np.asarray([100.0, 101.0, 101.0], dtype=float),
            funding_rate=np.asarray([0.0025, 0.0025, 0.0], dtype=float),
            open_interest=np.asarray([1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
            liquidation_long_notional=np.asarray([100_000.0, 100_000.0, 100_000.0], dtype=float),
            liquidation_short_notional=np.asarray([20_000.0, 20_000.0, 20_000.0], dtype=float),
            mark_price=None,
            index_price=None,
        ),
        config=research_runner._PerpCrowdingCarryConfig(
            window=24,
            mild_funding=0.0002,
            extreme_funding=0.0012,
            entry_threshold=0.3,
            exit_threshold=0.1,
            stop_loss_pct=0.02,
            max_hold_bars=8,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([-1.0, 0.0, 0.0], dtype=float))
    assert returned_support is support


def test_perp_carry_position_series_respects_allow_short_flag(monkeypatch):
    monkeypatch.setattr(
        research_runner,
        "_crowding_support_series",
        lambda **kwargs: {
            "crowding_score": np.asarray([0.45, 0.45, 0.0], dtype=float),
            "oi_delta_z": np.asarray([1.2, 1.2, 0.0], dtype=float),
        },
    )

    position, _ = research_runner._perp_carry_position_series(
        support_inputs=research_runner._CrowdingSupportInputs(
            close=np.asarray([100.0, 101.0, 101.0], dtype=float),
            funding_rate=np.asarray([0.0025, 0.0025, 0.0], dtype=float),
            open_interest=np.asarray([1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
            liquidation_long_notional=np.asarray([100_000.0, 100_000.0, 100_000.0], dtype=float),
            liquidation_short_notional=np.asarray([20_000.0, 20_000.0, 20_000.0], dtype=float),
            mark_price=None,
            index_price=None,
        ),
        config=research_runner._PerpCrowdingCarryConfig(
            window=24,
            mild_funding=0.0002,
            extreme_funding=0.0012,
            entry_threshold=0.3,
            exit_threshold=0.1,
            stop_loss_pct=0.02,
            max_hold_bars=8,
            allow_short=False,
        ),
    )

    assert np.array_equal(position, np.zeros(3, dtype=float))


def test_perp_carry_position_series_holds_mode_through_nonfinite_close(monkeypatch):
    monkeypatch.setattr(
        research_runner,
        "_crowding_support_series",
        lambda **kwargs: {
            "crowding_score": np.asarray([-0.45, -0.45, 0.0], dtype=float),
            "oi_delta_z": np.asarray([-1.2, -1.2, 0.0], dtype=float),
        },
    )

    position, _ = research_runner._perp_carry_position_series(
        support_inputs=research_runner._CrowdingSupportInputs(
            close=np.asarray([100.0, np.nan, 101.0], dtype=float),
            funding_rate=np.asarray([-0.0025, -0.0025, 0.0], dtype=float),
            open_interest=np.asarray([1_000_000.0, 1_000_000.0, 1_000_000.0], dtype=float),
            liquidation_long_notional=np.asarray([100_000.0, 100_000.0, 100_000.0], dtype=float),
            liquidation_short_notional=np.asarray([20_000.0, 20_000.0, 20_000.0], dtype=float),
            mark_price=None,
            index_price=None,
        ),
        config=research_runner._PerpCrowdingCarryConfig(
            window=24,
            mild_funding=0.0002,
            extreme_funding=0.0012,
            entry_threshold=0.3,
            exit_threshold=0.1,
            stop_loss_pct=0.02,
            max_hold_bars=8,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([1.0, 1.0, 0.0], dtype=float))


def test_perp_carry_should_exit_covers_long_and_short_branches():
    config = research_runner._PerpCrowdingCarryConfig(
        window=24,
        mild_funding=0.0002,
        extreme_funding=0.0012,
        entry_threshold=0.3,
        exit_threshold=0.1,
        stop_loss_pct=0.02,
        max_hold_bars=8,
        allow_short=True,
    )

    assert (
        research_runner._perp_carry_should_exit(
            mode=1,
            score_i=0.05,
            funding_i=0.0,
            close_i=100.0,
            entry_price=100.0,
            bars_held=1,
            config=config,
        )
        is True
    )
    assert (
        research_runner._perp_carry_should_exit(
            mode=-1,
            score_i=-0.05,
            funding_i=0.0,
            close_i=100.0,
            entry_price=100.0,
            bars_held=1,
            config=config,
        )
        is True
    )


def test_mean_reversion_std_strategy_signal_produces_exposure():
    length = 180
    close = np.full(length, 100.0, dtype=float)
    close[80] = 92.0
    close[81:] = np.linspace(93.0, 101.0, length - 81, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 100.0, dtype=float),
    }
    candidate = {
        "strategy_class": "MeanReversionStdStrategy",
        "params": {
            "window": 32,
            "entry_z": 1.8,
            "exit_z": 0.4,
            "stop_loss_pct": 0.03,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)


def test_mean_reversion_position_series_resumes_after_nonfinite_close(monkeypatch):
    monkeypatch.setattr(
        research_runner,
        "_rolling_z",
        lambda values, window: np.asarray([-2.0, -2.0, -2.0], dtype=float),
    )

    position = research_runner._mean_reversion_position_series(
        close=np.asarray([100.0, np.nan, 101.0], dtype=float),
        signal_series=np.asarray([100.0, 99.0, 98.0], dtype=float),
        window=16,
        entry_z=1.2,
        exit_z=0.3,
        stop_loss_pct=0.02,
        allow_short=True,
    )

    assert np.array_equal(position, np.asarray([1.0, 0.0, 1.0], dtype=float))


def test_mean_reversion_std_strategy_signal_can_residualize_btc():
    length = 180
    btc_close = np.full(length, 100.0, dtype=float)
    btc_close[80] = 92.0
    btc_close[81:] = np.linspace(93.0, 101.0, length - 81, dtype=float)
    eth_close = btc_close.copy()
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": btc_close,
        "BTC/USDT:high": btc_close + 0.2,
        "BTC/USDT:low": btc_close - 0.2,
        "BTC/USDT:close": btc_close,
        "BTC/USDT:volume": np.full(length, 100.0, dtype=float),
        "ETH/USDT:open": eth_close,
        "ETH/USDT:high": eth_close + 0.2,
        "ETH/USDT:low": eth_close - 0.2,
        "ETH/USDT:close": eth_close,
        "ETH/USDT:volume": np.full(length, 100.0, dtype=float),
    }
    raw_candidate = {
        "strategy_class": "MeanReversionStdStrategy",
        "params": {
            "window": 32,
            "entry_z": 1.8,
            "exit_z": 0.4,
            "stop_loss_pct": 0.03,
            "allow_short": True,
        },
    }
    residual_candidate = {
        "strategy_class": "MeanReversionStdStrategy",
        "params": {
            "window": 32,
            "entry_z": 1.8,
            "exit_z": 0.4,
            "stop_loss_pct": 0.03,
            "allow_short": True,
            "residualize_btc": True,
            "btc_symbol": "BTC/USDT",
        },
    }

    _, raw_turnover, raw_exposure, raw_meta = research_runner._strategy_signal(
        raw_candidate,
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT"],
    )
    _, residual_turnover, residual_exposure, residual_meta = research_runner._strategy_signal(
        residual_candidate,
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT"],
    )

    assert raw_meta == {}
    assert np.any(np.abs(raw_exposure) > 0.0)
    assert np.any(raw_turnover > 0.0)
    assert residual_meta.get("residualized_single_asset") is True
    assert residual_meta.get("residualize_btc") is True
    assert residual_meta.get("btc_symbol") == "BTC/USDT"
    assert np.allclose(residual_exposure, 0.0)
    assert np.allclose(residual_turnover, 0.0)


def test_funding_liquidation_crowding_fade_strategy_signal_produces_exposure():
    length = 420
    close = np.linspace(100.0, 118.0, length, dtype=float)
    close[220:228] *= np.linspace(1.0, 1.03, 8, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(1_000.0, 1_500.0, length, dtype=float),
        "BTC/USDT:funding_rate": np.linspace(0.0001, 0.0015, length, dtype=float),
        "BTC/USDT:open_interest": 1_000_000.0 * np.exp(np.linspace(0.0, 0.3, length, dtype=float)),
        "BTC/USDT:liquidation_long_notional": np.linspace(10_000.0, 120_000.0, length, dtype=float),
        "BTC/USDT:liquidation_short_notional": np.linspace(5_000.0, 20_000.0, length, dtype=float),
        "BTC/USDT:mark_price": close * 1.001,
        "BTC/USDT:index_price": close,
    }
    candidate = {
        "strategy_class": "FundingLiquidationCrowdingFadeStrategy",
        "params": {
            "window": 96,
            "crowding_entry": 0.6,
            "crowding_exit": 0.2,
            "liquidation_z_min": 0.2,
            "return_shock_pct": 0.002,
            "stop_loss_pct": 0.02,
            "max_hold_bars": 24,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert "BTC/USDT" in list(meta.get("support_data_symbols") or [])


def test_funding_liquidation_crowding_preserves_default_config(monkeypatch):
    captured: dict[str, float | int | bool] = {}

    def _stub_crowding_support_series(**kwargs):
        funding_rate = np.asarray(kwargs["funding_rate"], dtype=float)
        captured["support_window"] = int(kwargs["window"])
        shape = funding_rate.shape
        return {
            "crowding_score": np.zeros(shape, dtype=float),
            "liquidation_imbalance_z": np.zeros(shape, dtype=float),
        }

    def _stub_funding_liquidation_position_series(*, close, score, liquidation_z, config):
        captured.update(
            {
                "window": int(config.window),
                "crowding_entry": float(config.crowding_entry),
                "crowding_exit": float(config.crowding_exit),
                "liquidation_z_min": float(config.liquidation_z_min),
                "return_shock_pct": float(config.return_shock_pct),
                "max_hold_bars": int(config.max_hold_bars),
                "stop_loss_pct": float(config.stop_loss_pct),
                "allow_short": bool(config.allow_short),
            }
        )
        return np.zeros(np.asarray(close, dtype=float).shape, dtype=float)

    monkeypatch.setattr(research_runner, "_resolve_feature_points_path", lambda: Path(__file__))
    monkeypatch.setattr(research_runner, "_crowding_support_series", _stub_crowding_support_series)
    monkeypatch.setattr(
        research_runner,
        "_funding_liquidation_crowding_position_series",
        _stub_funding_liquidation_position_series,
    )

    length = 120
    close = np.linspace(100.0, 104.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
        "BTC/USDT:funding_rate": np.full(length, 0.0001, dtype=float),
        "BTC/USDT:open_interest": np.full(length, 1_000_000.0, dtype=float),
        "BTC/USDT:liquidation_long_notional": np.full(length, 120_000.0, dtype=float),
        "BTC/USDT:liquidation_short_notional": np.full(length, 20_000.0, dtype=float),
        "BTC/USDT:mark_price": close * 1.001,
        "BTC/USDT:index_price": close,
    }

    research_runner._strategy_signal(
        {"strategy_class": "FundingLiquidationCrowdingFadeStrategy", "params": {}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert captured == {
        "support_window": 96,
        "window": 96,
        "crowding_entry": 0.85,
        "crowding_exit": 0.25,
        "liquidation_z_min": 1.0,
        "return_shock_pct": 0.01,
        "max_hold_bars": 12,
        "stop_loss_pct": 0.02,
        "allow_short": True,
    }


def _funding_liquidation_crowding_config(**overrides):
    config = {
        "window": 96,
        "crowding_entry": 0.85,
        "crowding_exit": 0.25,
        "liquidation_z_min": 1.0,
        "return_shock_pct": 0.01,
        "max_hold_bars": 12,
        "stop_loss_pct": 0.02,
        "allow_short": True,
    }
    config.update(overrides)
    return research_runner._FundingLiquidationCrowdingFadeConfig(**config)


def test_funding_liquidation_crowding_helpers_cover_entry_and_exit_branches():
    config = _funding_liquidation_crowding_config()

    assert (
        research_runner._funding_liquidation_crowding_entry_mode(
            score_i=-0.9,
            liq_i=-1.2,
            ret_i=-0.02,
            config=config,
        )
        == 1
    )
    assert (
        research_runner._funding_liquidation_crowding_entry_mode(
            score_i=0.9,
            liq_i=1.2,
            ret_i=0.02,
            config=config,
        )
        == -1
    )
    assert (
        research_runner._funding_liquidation_crowding_should_exit(
            mode=1,
            score_i=-0.1,
            close_i=100.0,
            entry_price=100.0,
            bars_held=1,
            config=config,
        )
        is True
    )
    assert (
        research_runner._funding_liquidation_crowding_should_exit(
            mode=-1,
            score_i=0.1,
            close_i=100.0,
            entry_price=100.0,
            bars_held=1,
            config=config,
        )
        is True
    )


def test_funding_liquidation_crowding_position_series_exits_long_and_short_positions():
    config = _funding_liquidation_crowding_config()
    long_close = np.asarray([100.0, 98.0, 99.0, 100.0], dtype=float)
    short_close = np.asarray([100.0, 102.0, 101.0, 100.0], dtype=float)

    long_position = research_runner._funding_liquidation_crowding_position_series(
        close=long_close,
        score=np.asarray([-0.9, -0.9, -0.1, 0.0], dtype=float),
        liquidation_z=np.asarray([-1.2, -1.2, -1.2, 0.0], dtype=float),
        config=config,
    )
    short_position = research_runner._funding_liquidation_crowding_position_series(
        close=short_close,
        score=np.asarray([0.9, 0.9, 0.1, 0.0], dtype=float),
        liquidation_z=np.asarray([1.2, 1.2, 1.2, 0.0], dtype=float),
        config=config,
    )

    assert long_position[1] > 0.0
    assert np.array_equal(long_position[[0, 2, 3]], np.zeros(3, dtype=float))
    assert short_position[1] < 0.0
    assert np.array_equal(short_position[[0, 2, 3]], np.zeros(3, dtype=float))


def test_funding_liquidation_crowding_position_series_resumes_after_nonfinite_close():
    position = research_runner._funding_liquidation_crowding_position_series(
        close=np.asarray([100.0, 98.0, np.nan, 99.0], dtype=float),
        score=np.asarray([-0.9, -0.9, -0.9, -0.9], dtype=float),
        liquidation_z=np.asarray([-1.2, -1.2, -1.2, -1.2], dtype=float),
        config=_funding_liquidation_crowding_config(),
    )

    assert np.array_equal(position, np.asarray([0.0, 1.0, 1.0, 1.0], dtype=float))


def _deep_research_base_aligned(
    symbols: list[str],
    *,
    length: int = 240,
) -> dict[str, np.ndarray]:
    aligned: dict[str, np.ndarray] = {"datetime": _minute_datetimes(length)}
    for idx, symbol in enumerate(symbols):
        drift = 0.02 * (idx + 1)
        close = 100.0 + np.linspace(0.0, drift * length, length, dtype=float)
        if idx % 2:
            close = 120.0 - np.linspace(0.0, drift * length * 0.6, length, dtype=float)
        close = close + (0.35 * (idx + 1) * np.sin(np.linspace(0.0, 10.0, length)))
        aligned[f"{symbol}:open"] = close - 0.1
        aligned[f"{symbol}:high"] = close + 0.3
        aligned[f"{symbol}:low"] = close - 0.3
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.linspace(1_000.0, 1_800.0, length, dtype=float)
        aligned[f"{symbol}:funding_rate"] = np.linspace(
            -0.0002 + (idx * 0.0001),
            0.0002 + (idx * 0.0001),
            length,
            dtype=float,
        )
        aligned[f"{symbol}:open_interest"] = 1_000_000.0 * np.exp(
            np.linspace(0.0, 0.10 + (idx * 0.03), length, dtype=float)
        )
        aligned[f"{symbol}:liquidation_long_notional"] = np.linspace(
            10_000.0,
            20_000.0 + (idx * 1_000.0),
            length,
            dtype=float,
        )
        aligned[f"{symbol}:liquidation_short_notional"] = np.linspace(
            8_000.0,
            14_000.0 + (idx * 1_000.0),
            length,
            dtype=float,
        )
        aligned[f"{symbol}:mark_price"] = close * (1.0 + (idx * 0.0002))
        aligned[f"{symbol}:index_price"] = close
    return aligned


def test_deep_research_funding_dislocation_trend_carry_signal_produces_exposure():
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    aligned = _deep_research_base_aligned(symbols, length=260)
    candidate = {
        "strategy_class": "FundingDislocationTrendCarryStrategy",
        "params": {
            "fast_lookback_bars": 8,
            "mid_lookback_bars": 16,
            "slow_lookback_bars": 32,
            "rebalance_bars": 4,
            "signal_threshold": 0.0,
            "max_longs": 1,
            "max_shorts": 0,
            "crowding_window": 24,
            "trend_weight": 0.55,
            "carry_weight": 0.25,
            "basis_weight": 0.10,
            "crowding_penalty_weight": 0.10,
            "stop_loss_pct": 0.05,
            "max_abs_exposure": 1.0,
            "allow_short": False,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (260,)
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert meta.get("deep_research_leaf") == "funding_dislocation_trend_carry"
    assert set(meta.get("support_data_symbols") or []) == set(symbols)


def test_deep_research_vol_managed_momentum_crash_gate_signal_produces_exposure():
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    aligned = _deep_research_base_aligned(symbols, length=220)
    candidate = {
        "strategy_class": "VolManagedMomentumCrashGateStrategy",
        "params": {
            "momentum_lookback_bars": 24,
            "rebalance_bars": 4,
            "vol_window": 12,
            "target_vol": 0.02,
            "max_leverage": 1.0,
            "signal_threshold": 0.0,
            "max_longs": 1,
            "max_shorts": 0,
            "crash_window_bars": 12,
            "crash_return_pct": 0.20,
            "vol_ratio_window": 48,
            "vol_ratio_max": 10.0,
            "stress_reduce": 0.25,
            "allow_short": False,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (220,)
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert float(np.nanmax(np.abs(exposure))) <= 1.0
    assert meta.get("deep_research_leaf") == "vol_managed_momentum_crash_gate"
    assert set(meta.get("support_data_symbols") or []) == set(symbols)


def test_deep_research_flow_imbalance_liquidation_sweep_signal_produces_exposure():
    length = 180
    close = np.linspace(100.0, 104.0, length, dtype=float)
    close[90] = 98.0
    close[91:100] = np.linspace(98.5, 101.0, 9, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 1_000.0, dtype=float),
        "BTC/USDT:taker_buy_quote_volume": np.full(length, 100_000.0, dtype=float),
        "BTC/USDT:taker_sell_quote_volume": np.full(length, 100_000.0, dtype=float),
        "BTC/USDT:book_depth_imbalance_1pct": np.zeros(length, dtype=float),
        "BTC/USDT:bbo_spread_bps": np.full(length, 4.0, dtype=float),
        "BTC/USDT:liquidation_long_notional": np.full(length, 10_000.0, dtype=float),
        "BTC/USDT:liquidation_short_notional": np.full(length, 10_000.0, dtype=float),
    }
    aligned["BTC/USDT:taker_buy_quote_volume"][90] = 500_000.0
    aligned["BTC/USDT:book_depth_imbalance_1pct"][90] = 0.75
    aligned["BTC/USDT:liquidation_long_notional"][90] = 250_000.0
    candidate = {
        "strategy_class": "FlowImbalanceLiquidationSweepStrategy",
        "params": {
            "window": 16,
            "entry_score": 0.25,
            "exit_score": 0.05,
            "liquidation_z_min": 0.0,
            "return_shock_pct": 0.006,
            "max_spread_bps": 10.0,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.02,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert np.any(exposure > 0.0)
    assert np.any(turnover > 0.0)
    assert meta.get("deep_research_leaf") == "flow_imbalance_liquidation_sweep"
    assert "BTC/USDT" in list(meta.get("support_data_symbols") or [])


def test_liquidity_shock_reversion_strategy_signal_produces_exposure():
    length = 240
    close = np.linspace(100.0, 108.0, length, dtype=float)
    close[120:128] = np.linspace(103.0, 95.0, 8, dtype=float)
    close[128:144] = np.linspace(95.0, 101.0, 16, dtype=float)
    volume = np.full(length, 1_000.0, dtype=float)
    volume[120:128] = np.linspace(4_000.0, 5_500.0, 8, dtype=float)
    high = close + 0.3
    low = close - 0.3
    high[120:128] = close[120:128] * 1.03
    low[120:128] = close[120:128] * 0.97
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": high,
        "BTC/USDT:low": low,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": volume,
    }
    candidate = {
        "strategy_class": "LiquidityShockReversionStrategy",
        "params": {
            "volume_window": 24,
            "range_window": 16,
            "volume_shock_z": 0.8,
            "range_shock_z": 0.8,
            "return_shock_pct": 0.01,
            "revert_fraction": 0.50,
            "max_hold_bars": 18,
            "stop_loss_pct": 0.03,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)


def test_shock_reversion_position_series_resumes_after_nonfinite_close():
    position = research_runner._shock_reversion_position_series(
        close=np.asarray([100.0, np.nan, 99.0], dtype=float),
        returns=np.asarray([-0.05, -0.01, -0.01], dtype=float),
        vol_z=np.asarray([2.0, 2.0, 0.0], dtype=float),
        range_z=np.asarray([2.0, 2.0, 0.0], dtype=float),
        config=research_runner._ShockReversionFadeConfig(
            volume_window=24,
            range_window=16,
            volume_shock_z=0.8,
            range_shock_z=0.8,
            return_shock_pct=0.01,
            revert_fraction=0.50,
            max_hold_bars=18,
            stop_loss_pct=0.03,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([1.0, 0.0, 1.0], dtype=float))


def test_session_liquidity_vacuum_fade_strategy_signal_respects_session_gate():
    length = 120
    close = np.linspace(100.0, 102.0, length, dtype=float)
    close[16:22] = np.linspace(100.0, 94.0, 6, dtype=float)
    close[22:34] = np.linspace(94.0, 99.5, 12, dtype=float)
    volume = np.full(length, 900.0, dtype=float)
    volume[16:22] = np.linspace(3_500.0, 4_200.0, 6, dtype=float)
    high = close + 0.2
    low = close - 0.2
    high[16:22] = close[16:22] * 1.025
    low[16:22] = close[16:22] * 0.975
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": high,
        "BTC/USDT:low": low,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": volume,
    }
    candidate = {
        "strategy_class": "SessionLiquidityVacuumFadeStrategy",
        "params": {
            "volume_window": 8,
            "range_window": 8,
            "volume_shock_z": 0.5,
            "range_shock_z": 0.5,
            "return_shock_pct": 0.01,
            "revert_fraction": 0.45,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.03,
            "allow_short": False,
            "session_window_minutes": 30,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.max(exposure) > 0.0
    assert np.all(exposure >= 0.0)
    assert np.any(turnover > 0.0)


def test_basis_snapback_reversion_strategy_signal_produces_exposure():
    length = 420
    close = np.linspace(100.0, 118.0, length, dtype=float)
    mark = close.copy()
    index = close.copy()
    mark[200:212] *= np.linspace(1.0, 1.03, 12, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(1_000.0, 1_500.0, length, dtype=float),
        "BTC/USDT:mark_price": mark,
        "BTC/USDT:index_price": index,
    }
    candidate = {
        "strategy_class": "BasisSnapbackReversionStrategy",
        "params": {
            "window": 48,
            "entry_z": 0.8,
            "exit_z": 0.2,
            "max_hold_bars": 24,
            "stop_loss_pct": 0.02,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert meta == {}


def test_basis_snapback_reversion_preserves_default_window(monkeypatch):
    windows: list[int] = []

    def _stub_crowding_support_series(
        *,
        funding_rate,
        open_interest,
        mark_price,
        index_price,
        liquidation_long_notional,
        liquidation_short_notional,
        window=64,
    ):
        windows.append(int(window))
        shape = np.asarray(mark_price, dtype=float).shape
        return {"basis_z": np.zeros(shape, dtype=float)}

    monkeypatch.setattr(research_runner, "_resolve_feature_points_path", lambda: Path(__file__))
    monkeypatch.setattr(
        research_runner,
        "_crowding_support_series",
        _stub_crowding_support_series,
    )

    length = 120
    close = np.linspace(100.0, 104.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
        "BTC/USDT:mark_price": close * 1.001,
        "BTC/USDT:index_price": close,
    }

    research_runner._strategy_signal(
        {"strategy_class": "BasisSnapbackReversionStrategy", "params": {}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert windows == [96]


def test_basis_snapback_reversion_position_series_holds_through_missing_basis_signal():
    position = research_runner._basis_snapback_reversion_position_series(
        close=np.asarray([100.0, 101.0, 102.0], dtype=float),
        basis_z=np.asarray([-2.0, np.nan, -2.0], dtype=float),
        config=research_runner._BasisSnapbackReversionConfig(
            window=48,
            entry_z=1.8,
            exit_z=0.4,
            max_hold_bars=12,
            stop_loss_pct=0.02,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([1.0, 1.0, 1.0], dtype=float))


def test_vol_of_vol_exhaustion_fade_strategy_signal_produces_exposure(monkeypatch):
    def _stub_rolling_realized_vol(values, window):
        realized = np.zeros(np.asarray(values, dtype=float).shape, dtype=float)
        realized[60:67] = np.linspace(0.0, 3.0, 7, dtype=float)
        return realized

    def _stub_rolling_z(values, window):
        out = np.zeros(np.asarray(values, dtype=float).shape, dtype=float)
        if int(window) == 24:
            out[60:65] = 2.0
        elif int(window) == 12:
            out[60:64] = -1.5
            out[64:] = 0.5
        return out

    monkeypatch.setattr(research_runner, "_rolling_realized_vol", _stub_rolling_realized_vol)
    monkeypatch.setattr(research_runner, "_rolling_z", _stub_rolling_z)

    length = 120
    close = np.full(length, 100.0, dtype=float)
    close[64:] = 100.5
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }
    candidate = {
        "strategy_class": "VolOfVolExhaustionFadeStrategy",
        "params": {
            "vol_window": 16,
            "vol_z_window": 24,
            "return_z_window": 12,
            "vol_entry_z": 1.0,
            "return_entry_z": 0.8,
            "max_hold_bars": 8,
            "stop_loss_pct": 0.03,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(exposure > 0.0)
    assert np.any(turnover > 0.0)


def test_vol_of_vol_exhaustion_fade_preserves_default_windows(monkeypatch):
    vol_windows: list[int] = []
    z_windows: list[int] = []

    def _stub_rolling_realized_vol(values, window):
        vol_windows.append(int(window))
        return np.zeros(np.asarray(values, dtype=float).shape, dtype=float)

    def _stub_rolling_z(values, window):
        z_windows.append(int(window))
        return np.zeros(np.asarray(values, dtype=float).shape, dtype=float)

    monkeypatch.setattr(research_runner, "_rolling_realized_vol", _stub_rolling_realized_vol)
    monkeypatch.setattr(research_runner, "_rolling_z", _stub_rolling_z)

    length = 120
    close = np.linspace(100.0, 104.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }

    research_runner._strategy_signal(
        {"strategy_class": "VolOfVolExhaustionFadeStrategy", "params": {}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert vol_windows == [24]
    assert z_windows == [48, 24]


def test_vol_of_vol_exhaustion_position_series_resumes_after_nonfinite_close(monkeypatch):
    def _stub_rolling_realized_vol(values, window):
        return np.zeros(np.asarray(values, dtype=float).shape, dtype=float)

    def _stub_rolling_z(values, window):
        out = np.zeros(np.asarray(values, dtype=float).shape, dtype=float)
        if int(window) == 48:
            out[1:3] = 2.0
        elif int(window) == 24:
            out[1:] = -2.0
        return out

    monkeypatch.setattr(research_runner, "_rolling_realized_vol", _stub_rolling_realized_vol)
    monkeypatch.setattr(research_runner, "_rolling_z", _stub_rolling_z)

    position = research_runner._vol_of_vol_exhaustion_fade_position_series(
        close=np.asarray([100.0, 98.0, np.nan, 99.0], dtype=float),
        config=research_runner._VolOfVolExhaustionFadeConfig(
            vol_window=24,
            vol_z_window=48,
            return_z_window=24,
            vol_entry_z=1.8,
            return_entry_z=1.2,
            max_hold_bars=8,
            stop_loss_pct=0.02,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([0.0, 1.0, 0.0, 1.0], dtype=float))


def test_residual_basket_reversion_strategy_signal_produces_exposure():
    length = 420
    btc_close = np.linspace(100.0, 120.0, length, dtype=float)
    eth_close = np.linspace(100.0, 112.0, length, dtype=float)
    bnb_close = np.linspace(100.0, 108.0, length, dtype=float)
    eth_close[200:210] *= np.linspace(0.96, 0.92, 10, dtype=float)
    bnb_close[200:210] *= np.linspace(1.04, 1.08, 10, dtype=float)
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol, close in {
        "BTC/USDT": btc_close,
        "ETH/USDT": eth_close,
        "BNB/USDT": bnb_close,
    }.items():
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 120.0, dtype=float)

    candidate = {
        "strategy_class": "ResidualBasketReversionStrategy",
        "params": {
            "residual_window": 32,
            "entry_z": 0.8,
            "exit_z": 0.2,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 1,
            "stop_loss_pct": 0.02,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    )

    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert meta.get("cross_sectional") is True
    assert meta.get("residualized_cross_sectional") is True
    assert meta.get("btc_symbol") == "BTC/USDT"


def test_session_gated_residual_basket_reversion_strategy_signal_produces_exposure():
    length = 420
    datetimes = _minute_datetimes(length)
    btc_close = np.linspace(100.0, 120.0, length, dtype=float)
    eth_close = np.linspace(100.0, 112.0, length, dtype=float)
    bnb_close = np.linspace(100.0, 108.0, length, dtype=float)
    eth_close[200:210] *= np.linspace(0.96, 0.92, 10, dtype=float)
    bnb_close[200:210] *= np.linspace(1.04, 1.08, 10, dtype=float)
    aligned = {"datetime": datetimes}
    for symbol, close in {
        "BTC/USDT": btc_close,
        "ETH/USDT": eth_close,
        "BNB/USDT": bnb_close,
    }.items():
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 120.0, dtype=float)

    candidate = {
        "strategy_class": "SessionGatedResidualBasketReversionStrategy",
        "params": {
            "residual_window": 32,
            "entry_z": 0.8,
            "exit_z": 0.2,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 1,
            "stop_loss_pct": 0.02,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
            "session_window_minutes": 180,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    )

    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert meta.get("session_gated") is True
    assert meta.get("residualized_cross_sectional") is True


def test_volatility_regime_residual_basket_reversion_strategy_signal_produces_exposure():
    length = 420
    btc_close = np.linspace(100.0, 118.0, length, dtype=float)
    eth_close = np.linspace(100.0, 112.0, length, dtype=float)
    bnb_close = np.linspace(100.0, 108.0, length, dtype=float)
    sol_close = np.linspace(100.0, 110.0, length, dtype=float)
    eth_close[200:210] *= np.linspace(0.96, 0.92, 10, dtype=float)
    bnb_close[200:210] *= np.linspace(1.04, 1.08, 10, dtype=float)
    sol_close[200:210] *= np.linspace(0.95, 0.91, 10, dtype=float)
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol, close in {
        "BTC/USDT": btc_close,
        "ETH/USDT": eth_close,
        "BNB/USDT": bnb_close,
        "SOL/USDT": sol_close,
    }.items():
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 120.0, dtype=float)

    candidate = {
        "strategy_class": "VolatilityRegimeResidualBasketReversionStrategy",
        "params": {
            "residual_window": 32,
            "entry_z": 0.8,
            "exit_z": 0.2,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 1,
            "stop_loss_pct": 0.02,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
            "btc_vol_fast": 12,
            "btc_vol_slow": 60,
            "btc_vol_ratio_cap": 1.2,
            "dispersion_floor": 0.0005,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
    )

    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert meta.get("volatility_regime_gated") is True
    assert meta.get("residualized_cross_sectional") is True
    assert meta.get("btc_vol_fast") == 12


def test_session_gated_residual_basket_reversion_preserves_default_window(monkeypatch):
    windows: list[int] = []

    def _stub_beta_neutral_residual_series(values, benchmark, *, window):
        windows.append(int(window))
        return np.asarray(values, dtype=float)

    monkeypatch.setattr(
        research_runner,
        "_beta_neutral_residual_series",
        _stub_beta_neutral_residual_series,
    )

    length = 120
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol, base in {"BTC/USDT": 100.0, "ETH/USDT": 104.0, "BNB/USDT": 98.0}.items():
        close = np.linspace(base, base * 1.03, length, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 100.0, dtype=float)

    candidate = {
        "strategy_class": "SessionGatedResidualBasketReversionStrategy",
        "params": {
            "entry_z": 0.8,
            "exit_z": 0.2,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 1,
            "stop_loss_pct": 0.02,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
            "session_window_minutes": 180,
        },
    }

    research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    )

    assert windows
    assert all(window == 64 for window in windows)


def test_residual_basket_reversion_preserves_default_window(monkeypatch):
    windows: list[int] = []

    def _stub_beta_neutral_residual_series(values, benchmark, *, window):
        windows.append(int(window))
        return np.asarray(values, dtype=float)

    monkeypatch.setattr(
        research_runner,
        "_beta_neutral_residual_series",
        _stub_beta_neutral_residual_series,
    )

    length = 120
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol, base in {"BTC/USDT": 100.0, "ETH/USDT": 104.0, "BNB/USDT": 98.0}.items():
        close = np.linspace(base, base * 1.03, length, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 100.0, dtype=float)

    candidate = {
        "strategy_class": "ResidualBasketReversionStrategy",
        "params": {
            "entry_z": 0.8,
            "exit_z": 0.2,
            "rebalance_bars": 2,
            "max_longs": 1,
            "max_shorts": 1,
            "stop_loss_pct": 0.02,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
        },
    }

    research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    )

    assert windows
    assert all(window == 48 for window in windows)


def test_residual_basket_reversion_targets_respect_allow_short_and_overlap():
    config = research_runner._ResidualBasketReversionConfig(
        residual_window=48,
        entry_z=0.8,
        exit_z=0.2,
        rebalance_bars=2,
        max_longs=1,
        max_shorts=2,
        stop_loss_pct=0.02,
        allow_short=False,
    )
    residual_z_map = {
        "BTC/USDT": np.asarray([0.0], dtype=float),
        "ETH/USDT": np.asarray([-1.2], dtype=float),
        "BNB/USDT": np.asarray([1.4], dtype=float),
    }

    long_set, shorts = research_runner._residual_basket_reversion_targets(
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        btc_symbol="BTC/USDT",
        residual_z_map=residual_z_map,
        idx=0,
        config=config,
    )

    assert long_set == {"ETH/USDT"}
    assert shorts == []

    overlap_config = research_runner._ResidualBasketReversionConfig(
        residual_window=48,
        entry_z=0.0,
        exit_z=0.2,
        rebalance_bars=2,
        max_longs=2,
        max_shorts=2,
        stop_loss_pct=0.02,
        allow_short=True,
    )
    overlap_z_map = {
        "BTC/USDT": np.asarray([0.0], dtype=float),
        "ETH/USDT": np.asarray([0.0], dtype=float),
        "BNB/USDT": np.asarray([0.0], dtype=float),
    }

    overlap_longs, overlap_shorts = research_runner._residual_basket_reversion_targets(
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        btc_symbol="BTC/USDT",
        residual_z_map=overlap_z_map,
        idx=0,
        config=overlap_config,
    )

    assert overlap_longs == {"ETH/USDT", "BNB/USDT"}
    assert overlap_shorts == []


def test_apply_residual_basket_reversion_strategy_respects_rebalance_gate(monkeypatch):
    target_indices: list[int] = []
    length = 12

    monkeypatch.setattr(
        research_runner,
        "_residual_basket_reversion_z_map",
        lambda **_: {
            "BTC/USDT": np.zeros(length, dtype=float),
            "ETH/USDT": np.zeros(length, dtype=float),
            "BNB/USDT": np.zeros(length, dtype=float),
        },
    )
    monkeypatch.setattr(
        research_runner,
        "_apply_residual_basket_reversion_exits",
        lambda **_: None,
    )

    def _stub_targets(**kwargs):
        target_indices.append(int(kwargs["idx"]))
        return {"ETH/USDT"}, ["BNB/USDT"]

    monkeypatch.setattr(
        research_runner,
        "_residual_basket_reversion_targets",
        _stub_targets,
    )

    aligned = {"datetime": _minute_datetimes(length)}
    for symbol, base in {"BTC/USDT": 100.0, "ETH/USDT": 104.0, "BNB/USDT": 98.0}.items():
        close = np.linspace(base, base + 5.0, length, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 100.0, dtype=float)

    exposures = np.zeros((3, length), dtype=float)
    meta: dict[str, object] = {}

    research_runner._apply_residual_basket_reversion_strategy(
        params={
            "residual_window": 8,
            "rebalance_bars": 2,
            "btc_symbol": "BTC/USDT",
        },
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        exposures=exposures,
        meta=meta,
        entry_gate=np.asarray(
            [False, False, False, False, False, False, False, False, False, True, False, True],
            dtype=bool,
        ),
    )

    assert target_indices == [9, 11]
    assert np.all(exposures[:, :9] == 0.0)
    assert np.all(exposures[0] == 0.0)
    assert np.array_equal(exposures[1, 9:], np.asarray([1.0, 1.0, 1.0], dtype=float))
    assert np.array_equal(exposures[2, 9:], np.asarray([-1.0, -1.0, -1.0], dtype=float))
    assert meta["cross_sectional"] is True
    assert meta["residualized_cross_sectional"] is True
    assert meta["btc_symbol"] == "BTC/USDT"


def test_cross_asset_liquidation_contagion_fade_strategy_signal_produces_exposure():
    length = 420
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol in ["BTC/USDT", "ETH/USDT", "BNB/USDT"]:
        close = np.linspace(100.0, 110.0, length, dtype=float)
        if symbol == "ETH/USDT":
            close[220:228] *= np.linspace(1.0, 1.03, 8, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 100.0, dtype=float)
        long_liq = np.linspace(10_000.0, 20_000.0, length, dtype=float)
        short_liq = np.linspace(5_000.0, 10_000.0, length, dtype=float)
        if symbol != "ETH/USDT":
            long_liq[220:228] = np.linspace(50_000.0, 140_000.0, 8, dtype=float)
        aligned[f"{symbol}:liquidation_long_notional"] = long_liq
        aligned[f"{symbol}:liquidation_short_notional"] = short_liq

    candidate = {
        "strategy_class": "CrossAssetLiquidationContagionFadeStrategy",
        "params": {
            "window": 48,
            "leader_liq_z_min": 0.4,
            "return_shock_pct": 0.2,
            "exit_z": 0.2,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.02,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
    )

    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert meta == {}


def test_cross_asset_liquidation_contagion_position_series_resumes_after_nonfinite_close():
    position = research_runner._cross_asset_liquidation_contagion_position_series(
        symbol="ETH/USDT",
        close_arr=np.asarray([100.0, np.nan, 99.0], dtype=float),
        valid_symbols=["BTC/USDT", "ETH/USDT"],
        liq_z_map={
            "BTC/USDT": np.asarray([1.0, 1.0, 1.0], dtype=float),
            "ETH/USDT": np.asarray([0.0, 0.0, 0.0], dtype=float),
        },
        return_z_map={
            "ETH/USDT": np.asarray([0.5, 0.5, 0.5], dtype=float),
        },
        config=research_runner._CrossAssetLiquidationContagionFadeConfig(
            window=48,
            leader_liq_z_min=0.4,
            return_shock_pct=0.2,
            exit_z=0.2,
            max_hold_bars=12,
            stop_loss_pct=0.02,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([-1.0, 0.0, -1.0], dtype=float))


def test_multi_horizon_trend_exhaustion_fade_strategy_signal_produces_exposure():
    length = 180
    close = np.linspace(100.0, 120.0, length, dtype=float)
    close[120:135] *= np.linspace(1.0, 1.12, 15, dtype=float)
    close[135:] *= np.linspace(1.12, 1.05, length - 135, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 100.0, dtype=float),
    }
    candidate = {
        "strategy_class": "MultiHorizonTrendExhaustionFadeStrategy",
        "params": {
            "short_window": 8,
            "entry_z": 1.2,
            "exit_z": 0.2,
            "max_hold_bars": 12,
            "stop_loss_pct": 0.02,
            "allow_short": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
    assert meta == {}


def test_multi_horizon_trend_exhaustion_fade_preserves_default_short_window(monkeypatch):
    windows: list[int] = []

    def _stub_rolling_z(values, window):
        windows.append(int(window))
        return np.zeros(np.asarray(values, dtype=float).shape, dtype=float)

    monkeypatch.setattr(research_runner, "_rolling_z", _stub_rolling_z)
    monkeypatch.setattr(
        research_runner,
        "_composite_momentum_series",
        lambda values, **kwargs: np.zeros(np.asarray(values, dtype=float).shape, dtype=float),
    )

    length = 120
    close = np.linspace(100.0, 104.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }

    research_runner._strategy_signal(
        {"strategy_class": "MultiHorizonTrendExhaustionFadeStrategy", "params": {}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert windows == [16]


def test_multi_horizon_trend_exhaustion_fade_resumes_after_nonfinite_close(monkeypatch):
    monkeypatch.setattr(
        research_runner,
        "_rolling_z",
        lambda values, window: np.asarray([-2.0, -2.0, -2.0], dtype=float),
    )
    monkeypatch.setattr(
        research_runner,
        "_composite_momentum_series",
        lambda values, **kwargs: np.asarray([1.0, 1.0, 1.0], dtype=float),
    )

    position = research_runner._multi_horizon_trend_exhaustion_fade_position_series(
        close=np.asarray([100.0, np.nan, 101.0], dtype=float),
        config=research_runner._MultiHorizonTrendExhaustionFadeConfig(
            short_window=8,
            entry_z=1.2,
            exit_z=0.2,
            max_hold_bars=12,
            stop_loss_pct=0.02,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([1.0, 0.0, 1.0], dtype=float))


def test_vwap_reversion_strategy_signal_produces_exposure():
    length = 180
    close = np.full(length, 100.0, dtype=float)
    close[70:85] = np.linspace(100.0, 95.0, 15, dtype=float)
    close[85:110] = np.linspace(95.0, 101.0, 25, dtype=float)
    close[110:] = 101.0
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }
    candidate = {
        "strategy_class": "VwapReversionStrategy",
        "params": {
            "window": 24,
            "entry_dev": 0.015,
            "exit_dev": 0.003,
            "stop_loss_pct": 0.03,
            "allow_short": False,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(exposure > 0.0)
    assert np.any(turnover > 0.0)


def test_vwap_reversion_resumes_after_nonfinite_close(monkeypatch):
    monkeypatch.setattr(
        research_runner,
        "_rolling_vwap_deviation",
        lambda close, volume, window: np.asarray([-0.03, -0.03, -0.03], dtype=float),
    )

    position = research_runner._vwap_reversion_position_series(
        close=np.asarray([100.0, np.nan, 101.0], dtype=float),
        volume=np.asarray([120.0, 120.0, 120.0], dtype=float),
        config=research_runner._VwapReversionConfig(
            window=16,
            entry_dev=0.02,
            exit_dev=0.005,
            stop_loss_pct=0.03,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([1.0, 0.0, 1.0], dtype=float))


def test_rolling_breakout_strategy_signal_produces_exposure():
    length = 160
    close = np.linspace(100.0, 112.0, length, dtype=float)
    high = close + 0.5
    low = close - 0.5
    close[90:] += 3.0
    high[90:] = close[90:] + 0.5
    low[90:] = close[90:] - 0.5
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": high,
        "BTC/USDT:low": low,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }
    candidate = {
        "strategy_class": "RollingBreakoutStrategy",
        "params": {
            "lookback_bars": 24,
            "breakout_buffer": 0.0,
            "atr_window": 8,
            "atr_stop_multiplier": 1.5,
            "stop_loss_pct": 0.02,
            "allow_short": False,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(exposure > 0.0)
    assert np.any(turnover > 0.0)


def test_rolling_breakout_resumes_across_nonfinite_close():
    position = research_runner._rolling_breakout_position_series(
        close=np.asarray([100.0, 101.0, np.nan, 102.0], dtype=float),
        channel_high=np.asarray([np.nan, 100.0, 100.0, 100.0], dtype=float),
        channel_low=np.asarray([np.nan, 99.0, 99.0, 99.0], dtype=float),
        atr_pct=np.asarray([np.nan, 0.01, 0.01, 0.01], dtype=float),
        config=research_runner._RollingBreakoutConfig(
            lookback_bars=24,
            breakout_buffer=0.0,
            atr_window=8,
            atr_stop_multiplier=1.5,
            stop_loss_pct=0.02,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([0.0, 1.0, 1.0, 1.0], dtype=float))


def test_vwap_reversion_preserves_default_window(monkeypatch):
    windows: list[int] = []

    def _stub_rolling_vwap_deviation(close, volume, window):
        windows.append(int(window))
        return np.zeros_like(np.asarray(close, dtype=float))

    monkeypatch.setattr(
        research_runner,
        "_rolling_vwap_deviation",
        _stub_rolling_vwap_deviation,
    )

    length = 120
    close = np.linspace(100.0, 104.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.2,
        "BTC/USDT:low": close - 0.2,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }

    research_runner._strategy_signal(
        {"strategy_class": "VwapReversionStrategy", "params": {}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert windows == [64]


def test_rolling_breakout_preserves_default_lookback(monkeypatch):
    channel_windows: list[int] = []
    atr_windows: list[int] = []

    def _stub_rolling_channel(high, low, window):
        channel_windows.append(int(window))
        shape = np.asarray(high, dtype=float).shape
        return np.full(shape, np.nan, dtype=float), np.full(shape, np.nan, dtype=float)

    def _stub_rolling_breakout_atr_pct(*, close, high, low, window):
        atr_windows.append(int(window))
        return np.full(np.asarray(close, dtype=float).shape, np.nan, dtype=float)

    monkeypatch.setattr(
        research_runner,
        "_rolling_channel",
        _stub_rolling_channel,
    )
    monkeypatch.setattr(
        research_runner,
        "_rolling_breakout_atr_pct",
        _stub_rolling_breakout_atr_pct,
    )

    length = 120
    close = np.linspace(100.0, 104.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.5,
        "BTC/USDT:low": close - 0.5,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }

    research_runner._strategy_signal(
        {"strategy_class": "RollingBreakoutStrategy", "params": {}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert channel_windows == [48]
    assert atr_windows == [14]


def test_topcap_tsmom_strategy_signal_produces_cross_sectional_exposure():
    length = 120
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for idx, symbol in enumerate(symbols):
        base = 100.0 + (idx * 10.0)
        close = np.full(length, base, dtype=float)
        if symbol in {"ETH/USDT", "BNB/USDT"}:
            close[20:] = np.linspace(base, base * 1.25, length - 20, dtype=float)
        elif symbol in {"SOL/USDT", "TRX/USDT"}:
            close[20:] = np.linspace(base, base * 0.85, length - 20, dtype=float)
        else:
            close[20:] = np.linspace(base, base * 1.05, length - 20, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (length,)
    assert meta.get("cross_sectional") is True
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)


def test_topcap_tsmom_strategy_signal_supports_zero_short_budget():
    length = 120
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for idx, symbol in enumerate(symbols):
        base = 100.0 + (idx * 10.0)
        close = np.full(length, base, dtype=float)
        if symbol in {"ETH/USDT", "BNB/USDT"}:
            close[20:] = np.linspace(base, base * 1.25, length - 20, dtype=float)
        elif symbol in {"SOL/USDT", "TRX/USDT"}:
            close[20:] = np.linspace(base, base * 0.85, length - 20, dtype=float)
        else:
            close[20:] = np.linspace(base, base * 1.05, length - 20, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 0,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (length,)
    assert meta.get("cross_sectional") is True
    assert np.any(exposure > 0.0)
    assert np.all(exposure >= 0.0)
    assert np.any(turnover > 0.0)


def test_topcap_tsmom_strategy_signal_respects_stop_loss_between_rebalances():
    length = 80
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol in symbols:
        base = 100.0
        close = np.full(length, base, dtype=float)
        if symbol == "BTC/USDT":
            close[12:] = np.linspace(base, base * 1.04, length - 12, dtype=float)
        elif symbol == "ETH/USDT":
            close[:24] = np.linspace(base, base * 1.18, 24, dtype=float)
            close[24:32] = np.linspace(base * 1.18, base * 0.92, 8, dtype=float)
            close[32:] = np.linspace(base * 0.92, base * 0.95, length - 32, dtype=float)
        elif symbol == "BNB/USDT":
            close[12:] = np.linspace(base, base * 1.22, length - 12, dtype=float)
        else:
            close[12:] = np.linspace(base, base * 0.92, length - 12, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    base_candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 8,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.20,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
        },
    }
    tightstop_candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            **dict(base_candidate["params"]),
            "stop_loss_pct": 0.03,
        },
    }

    _, _, base_exposure, _ = research_runner._strategy_signal(
        base_candidate,
        aligned=aligned,
        symbols=symbols,
    )
    _, _, tight_exposure, _ = research_runner._strategy_signal(
        tightstop_candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert np.sum(np.abs(tight_exposure[24:32])) < np.sum(np.abs(base_exposure[24:32]))
    assert np.max(np.abs(base_exposure[24:32])) > 0.0
    assert np.min(np.abs(tight_exposure[24:32])) == 0.0


def test_topcap_tsmom_strategy_signal_respects_take_profit_between_rebalances():
    length = 80
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol in symbols:
        base = 100.0
        close = np.full(length, base, dtype=float)
        if symbol == "BTC/USDT":
            close[12:] = np.linspace(base, base * 1.04, length - 12, dtype=float)
        elif symbol == "ETH/USDT":
            close[:28] = np.linspace(base, base * 1.24, 28, dtype=float)
            close[28:] = np.linspace(base * 1.24, base * 1.25, length - 28, dtype=float)
        elif symbol == "BNB/USDT":
            close[12:] = np.linspace(base, base * 1.20, length - 12, dtype=float)
        else:
            close[12:] = np.linspace(base, base * 0.92, length - 12, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    base_candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 8,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
        },
    }
    tp_candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            **dict(base_candidate["params"]),
            "take_profit_pct": 0.05,
        },
    }

    _, _, base_exposure, _ = research_runner._strategy_signal(
        base_candidate,
        aligned=aligned,
        symbols=symbols,
    )
    _, _, tp_exposure, meta = research_runner._strategy_signal(
        tp_candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert meta.get("take_profit_pct") == 0.05
    assert np.sum(np.abs(tp_exposure[20:24])) < np.sum(np.abs(base_exposure[20:24]))
    assert np.sum(np.abs(tp_exposure[41:47])) < np.sum(np.abs(base_exposure[41:47]))


def test_topcap_tsmom_strategy_signal_can_residualize_common_btc_factor():
    length = 120
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for idx, symbol in enumerate(symbols):
        base = 100.0 + (idx * 10.0)
        close = np.full(length, base, dtype=float)
        close[20:] = np.linspace(base, base * 1.25, length - 20, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
            "residualize_btc": True,
            "residualize_mean": True,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (length,)
    assert meta.get("cross_sectional") is True
    assert meta.get("residualized_cross_sectional") is True
    assert meta.get("residualize_btc") is True
    assert meta.get("residualize_mean") is True
    assert np.allclose(exposure, 0.0)
    assert np.allclose(turnover, 0.0)


def test_topcap_tsmom_strategy_signal_can_apply_benchmark_drawdown_gate():
    length = 120
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "TRX/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol in symbols:
        base = 100.0
        if symbol == "BTC/USDT":
            close = np.concatenate(
                [
                    np.linspace(base, base * 1.08, 36, dtype=float),
                    np.linspace(base * 1.08, base * 0.68, length - 36, dtype=float),
                ]
            )
        elif symbol in {"ETH/USDT", "BNB/USDT", "SOL/USDT"}:
            close = np.linspace(base, base * 1.25, length, dtype=float)
        else:
            close = np.linspace(base, base * 0.95, length, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
            "benchmark_drawdown_window": 24,
            "benchmark_drawdown_limit": 0.08,
        },
    }
    base_candidate = {
        "strategy_class": "TopCapTimeSeriesMomentumStrategy",
        "params": {
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.08,
            "max_longs": 2,
            "max_shorts": 2,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
        },
    }

    _, _, base_exposure, _ = research_runner._strategy_signal(
        base_candidate,
        aligned=aligned,
        symbols=symbols,
    )
    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=symbols,
    )

    assert exposure.shape == (length,)
    assert meta.get("crash_aware_gate") is True
    assert meta.get("benchmark_drawdown_window") == 24
    assert meta.get("benchmark_drawdown_limit") == 0.08
    assert np.max(base_exposure[-10:]) > 0.0
    assert np.max(exposure[-10:]) <= 1e-9
    assert np.any(turnover > 0.0)


def test_apply_topcap_tsmom_strategy_rebalances_only_on_cadence(monkeypatch):
    rebalance_indices: list[int] = []

    monkeypatch.setattr(
        research_runner,
        "_apply_topcap_risk_exits",
        lambda **kwargs: None,
    )

    def _stub_ranked_momentum_rows(**kwargs):
        rebalance_indices.append(int(kwargs["idx"]))
        return [(0.5, "ETH/USDT"), (-0.5, "BNB/USDT")]

    monkeypatch.setattr(
        research_runner,
        "_topcap_ranked_momentum_rows",
        _stub_ranked_momentum_rows,
    )
    monkeypatch.setattr(research_runner, "_topcap_market_regime", lambda *args, **kwargs: "BOTH")
    monkeypatch.setattr(research_runner, "_topcap_residualized_rows", lambda rows, **kwargs: rows)
    monkeypatch.setattr(
        research_runner,
        "_topcap_target_sets",
        lambda rows, **kwargs: ({"ETH/USDT"}, {"BNB/USDT"}),
    )

    length = 12
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    aligned = {"datetime": _minute_datetimes(length)}
    for symbol, base in {"BTC/USDT": 100.0, "ETH/USDT": 104.0, "BNB/USDT": 98.0}.items():
        close = np.linspace(base, base + 5.0, length, dtype=float)
        aligned[f"{symbol}:open"] = close
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 150.0, dtype=float)

    exposures = np.zeros((len(symbols), length), dtype=float)
    meta: dict[str, object] = {}

    research_runner._apply_topcap_tsmom_strategy(
        params={
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.01,
            "stop_loss_pct": 0.0,
            "max_longs": 1,
            "max_shorts": 1,
            "min_price": 0.1,
            "btc_regime_ma": 0,
            "btc_symbol": "BTC/USDT",
        },
        aligned=aligned,
        symbols=symbols,
        n=length,
        exposures=exposures,
        meta=meta,
    )

    assert rebalance_indices == [9, 11]
    assert np.all(exposures[:, :9] == 0.0)
    assert np.all(exposures[0] == 0.0)
    assert np.array_equal(exposures[1, 9:], np.asarray([1.0, 1.0, 1.0], dtype=float))
    assert np.array_equal(exposures[2, 9:], np.asarray([-1.0, -1.0, -1.0], dtype=float))
    assert meta["cross_sectional"] is True


def test_carry_trend_factor_rotation_strategy_prefers_uncrowded_trend_leaders(monkeypatch):
    def _stub_crowding_support_series(**kwargs):
        funding = float(np.asarray(kwargs["funding_rate"], dtype=float)[0])
        length = len(np.asarray(kwargs["funding_rate"], dtype=float))
        if funding <= 0.00012:  # ETH: favorable carry, uncrowded
            return {
                "crowding_score": np.full(length, -0.8, dtype=float),
                "funding_z": np.full(length, -1.0, dtype=float),
                "oi_delta_z": np.zeros(length, dtype=float),
                "basis_z": np.full(length, -0.4, dtype=float),
                "liquidation_imbalance_z": np.zeros(length, dtype=float),
            }
        if funding <= 0.00045:  # BTC: neutral
            return {
                "crowding_score": np.zeros(length, dtype=float),
                "funding_z": np.zeros(length, dtype=float),
                "oi_delta_z": np.zeros(length, dtype=float),
                "basis_z": np.zeros(length, dtype=float),
                "liquidation_imbalance_z": np.zeros(length, dtype=float),
            }
        if funding <= 0.00075:  # BNB: crowded/expensive long
            return {
                "crowding_score": np.full(length, 1.1, dtype=float),
                "funding_z": np.full(length, 1.3, dtype=float),
                "oi_delta_z": np.zeros(length, dtype=float),
                "basis_z": np.full(length, 0.8, dtype=float),
                "liquidation_imbalance_z": np.zeros(length, dtype=float),
            }
        # SOL: crowded and negative trend -> short
        return {
            "crowding_score": np.full(length, 0.7, dtype=float),
            "funding_z": np.full(length, 1.1, dtype=float),
            "oi_delta_z": np.zeros(length, dtype=float),
            "basis_z": np.full(length, 0.5, dtype=float),
            "liquidation_imbalance_z": np.zeros(length, dtype=float),
        }

    monkeypatch.setattr(research_runner, "_crowding_support_series", _stub_crowding_support_series)

    length = 16
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    close_map = {
        "BTC/USDT": np.linspace(100.0, 108.0, length, dtype=float),
        "ETH/USDT": np.linspace(100.0, 126.0, length, dtype=float),
        "BNB/USDT": np.linspace(100.0, 116.0, length, dtype=float),
        "SOL/USDT": np.linspace(100.0, 84.0, length, dtype=float),
    }
    funding_map = {
        "BTC/USDT": 0.00030,
        "ETH/USDT": 0.00010,
        "BNB/USDT": 0.00060,
        "SOL/USDT": 0.00100,
    }

    aligned = {"datetime": _minute_datetimes(length)}
    for symbol in symbols:
        close = close_map[symbol]
        aligned[f"{symbol}:open"] = close - 0.1
        aligned[f"{symbol}:high"] = close + 0.2
        aligned[f"{symbol}:low"] = close - 0.2
        aligned[f"{symbol}:close"] = close
        aligned[f"{symbol}:volume"] = np.full(length, 120.0, dtype=float)
        aligned[f"{symbol}:funding_rate"] = np.full(length, funding_map[symbol], dtype=float)
        aligned[f"{symbol}:open_interest"] = np.linspace(
            1_000_000.0, 1_100_000.0, length, dtype=float
        )
        aligned[f"{symbol}:liquidation_long_notional"] = np.full(length, 100_000.0, dtype=float)
        aligned[f"{symbol}:liquidation_short_notional"] = np.full(length, 40_000.0, dtype=float)
        aligned[f"{symbol}:mark_price"] = close * 1.001
        aligned[f"{symbol}:index_price"] = close

    exposures = np.zeros((len(symbols), length), dtype=float)
    meta: dict[str, object] = {}

    research_runner._apply_carry_trend_factor_rotation_strategy(
        params={
            "lookback_bars": 8,
            "rebalance_bars": 2,
            "signal_threshold": 0.10,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "max_longs": 1,
            "max_shorts": 1,
            "min_price": 0.10,
            "btc_regime_ma": 0,
            "benchmark_drawdown_window": 0,
            "benchmark_drawdown_limit": 0.0,
            "vol_window": 6,
            "crowding_window": 24,
            "trend_weight": 0.55,
            "carry_weight": 0.20,
            "defensive_weight": 0.15,
            "crowding_weight": 0.10,
            "allow_short": True,
            "btc_symbol": "BTC/USDT",
        },
        aligned=aligned,
        symbols=symbols,
        n=length,
        exposures=exposures,
        meta=meta,
    )

    turnover = np.sum(np.abs(np.diff(exposures, axis=1)), axis=0, dtype=float)

    assert exposures.shape == (len(symbols), length)
    assert meta["cross_sectional"] is True
    assert meta["factor_rotation"] is True
    assert meta["factor_rotation_strategy"] == "CarryTrendFactorRotationStrategy"
    assert np.any(turnover > 0.0)
    assert exposures[symbols.index("ETH/USDT"), -1] == 1.0
    assert exposures[symbols.index("SOL/USDT"), -1] == -1.0
    assert exposures[symbols.index("BNB/USDT"), -1] == 0.0


def test_regime_breakout_strategy_signal_still_trades_persistent_breakout():
    length = 160
    close = np.linspace(100.0, 170.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.2,
        "BTC/USDT:high": close + 0.6,
        "BTC/USDT:low": close - 0.6,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 250.0, dtype=float),
    }
    candidate = {
        "strategy_class": "RegimeBreakoutCandidateStrategy",
        "params": {
            "lookback_window": 36,
            "slope_window": 18,
            "volatility_fast_window": 10,
            "volatility_slow_window": 40,
            "range_entry_threshold": 0.65,
            "slope_entry_threshold": 0.0008,
            "momentum_floor": 0.002,
            "max_volatility_ratio": 2.5,
            "stop_loss_pct": 0.02,
            "allow_short": False,
        },
    }

    _, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert meta == {}
    assert np.any(exposure > 0.0)
    assert np.any(turnover > 0.0)


def test_regime_breakout_strategy_signal_respects_composite_momentum_gate():
    length = 160
    down = np.linspace(220.0, 100.0, 120, dtype=float)
    rebound = np.linspace(100.0, 108.0, length - 120, dtype=float)
    close = np.concatenate([down, rebound])
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.2,
        "BTC/USDT:high": close + 0.6,
        "BTC/USDT:low": close - 0.6,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 250.0, dtype=float),
    }
    candidate = {
        "strategy_class": "RegimeBreakoutCandidateStrategy",
        "params": {
            "lookback_window": 36,
            "slope_window": 18,
            "volatility_fast_window": 10,
            "volatility_slow_window": 40,
            "range_entry_threshold": 0.65,
            "slope_entry_threshold": 0.0008,
            "momentum_floor": 0.01,
            "max_volatility_ratio": 2.5,
            "stop_loss_pct": 0.02,
            "allow_short": False,
        },
    }

    _, turnover, exposure, _ = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert exposure.shape == (length,)
    assert np.all(exposure >= 0.0)
    assert not np.any(exposure > 0.0)
    assert not np.any(turnover > 0.0)


def test_regime_breakout_position_series_resumes_after_invalid_indicator_bar():
    position = research_runner._regime_breakout_candidate_position_series(
        close=np.asarray([100.0, 101.0, 102.0], dtype=float),
        channel_high=np.asarray([101.0, np.nan, 103.0], dtype=float),
        channel_low=np.asarray([99.0, np.nan, 99.0], dtype=float),
        vol_ratio=np.asarray([1.0, 1.0, 1.0], dtype=float),
        slope_series=np.asarray([0.1, 0.1, 0.1], dtype=float),
        momentum_series=np.asarray([0.1, 0.1, 0.1], dtype=float),
        config=research_runner._RegimeBreakoutCandidateConfig(
            lookback_window=8,
            slope_window=4,
            volatility_fast_window=4,
            volatility_slow_window=8,
            range_entry_threshold=0.5,
            slope_entry_threshold=0.0,
            momentum_floor=0.0,
            max_volatility_ratio=2.0,
            stop_loss_pct=0.05,
            allow_short=True,
        ),
    )

    assert np.array_equal(position, np.asarray([1.0, 0.0, 1.0], dtype=float))


def test_regime_breakout_preserves_default_windows(monkeypatch):
    observed: dict[str, list[int]] = {
        "channel": [],
        "vol_fast": [],
        "vol_slow": [],
        "slope": [],
    }

    def _stub_rolling_channel(high, low, window):
        observed["channel"].append(int(window))
        shape = np.asarray(high, dtype=float).shape
        return np.full(shape, np.nan, dtype=float), np.full(shape, np.nan, dtype=float)

    def _stub_vol_ratio_series(values, fast_window, slow_window):
        observed["vol_fast"].append(int(fast_window))
        observed["vol_slow"].append(int(slow_window))
        return np.full(np.asarray(values, dtype=float).shape, np.nan, dtype=float)

    def _stub_rolling_slope_series(values, window):
        observed["slope"].append(int(window))
        return np.full(np.asarray(values, dtype=float).shape, np.nan, dtype=float)

    monkeypatch.setattr(research_runner, "_rolling_channel", _stub_rolling_channel)
    monkeypatch.setattr(research_runner, "_vol_ratio_series", _stub_vol_ratio_series)
    monkeypatch.setattr(research_runner, "_rolling_slope_series", _stub_rolling_slope_series)

    length = 120
    close = np.linspace(100.0, 104.0, length, dtype=float)
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close - 0.1,
        "BTC/USDT:high": close + 0.5,
        "BTC/USDT:low": close - 0.5,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 120.0, dtype=float),
    }

    research_runner._strategy_signal(
        {"strategy_class": "RegimeBreakoutCandidateStrategy", "params": {}},
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert observed["channel"] == [48]
    assert observed["vol_fast"] == [12]
    assert observed["vol_slow"] == [48]
    assert observed["slope"] == [21]


def _pair_aligned_prices(length: int = 320) -> dict[str, np.ndarray]:
    x_close = []
    y_close = []
    for idx in range(length):
        y_price = 100.0 + (0.08 * idx)
        spread_noise = 0.08 * np.sin(idx / 5.0)
        if 220 <= idx <= 236:
            spread_noise += 1.6
        if 280 <= idx <= 295:
            spread_noise -= 1.7
        x_price = y_price + spread_noise
        x_close.append(x_price)
        y_close.append(y_price)

    x_close_arr = np.asarray(x_close, dtype=float)
    y_close_arr = np.asarray(y_close, dtype=float)
    return {
        "datetime": _minute_datetimes(length),
        "XAU/USDT:open": x_close_arr,
        "XAU/USDT:high": x_close_arr + 0.2,
        "XAU/USDT:low": x_close_arr - 0.2,
        "XAU/USDT:close": x_close_arr,
        "XAU/USDT:volume": np.full(length, 100.0, dtype=float),
        "XAG/USDT:open": y_close_arr,
        "XAG/USDT:high": y_close_arr + 0.2,
        "XAG/USDT:low": y_close_arr - 0.2,
        "XAG/USDT:close": y_close_arr,
        "XAG/USDT:volume": np.full(length, 100.0, dtype=float),
    }


def test_pair_trading_strategy_signal_uses_event_driven_proxy():
    aligned = _pair_aligned_prices()
    candidate = {
        "strategy_class": "PairTradingZScoreStrategy",
        "params": {
            "lookback_window": 30,
            "hedge_window": 60,
            "entry_z": 1.5,
            "exit_z": 0.25,
            "stop_z": 4.5,
            "min_correlation": -1.0,
            "max_hold_bars": 120,
            "cooldown_bars": 0,
            "reentry_z_buffer": 0.0,
            "symbol_x": "XAU/USDT",
            "symbol_y": "XAG/USDT",
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["XAU/USDT", "XAG/USDT"],
    )

    assert returns_raw.shape == (320,)
    assert turnover.shape == (320,)
    assert exposure.shape == (320,)
    assert meta.get("event_driven_proxy") is True
    assert np.any(turnover > 0.0)
    assert np.any(np.abs(returns_raw) > 0.0)


def test_pair_trading_strategy_signal_respects_reentry_buffer():
    aligned = _pair_aligned_prices()
    candidate = {
        "strategy_class": "PairTradingZScoreStrategy",
        "params": {
            "lookback_window": 30,
            "hedge_window": 60,
            "entry_z": 1.5,
            "exit_z": 0.25,
            "stop_z": 4.5,
            "min_correlation": -1.0,
            "max_hold_bars": 120,
            "cooldown_bars": 0,
            "reentry_z_buffer": 10.0,
            "symbol_x": "XAU/USDT",
            "symbol_y": "XAG/USDT",
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["XAU/USDT", "XAG/USDT"],
    )

    assert returns_raw.shape == (320,)
    assert turnover.shape == (320,)
    assert exposure.shape == (320,)
    assert meta.get("event_driven_proxy") is True
    assert not np.any(turnover > 0.0)
    assert not np.any(np.abs(returns_raw) > 0.0)


def test_pair_trading_single_symbol_falls_back_to_generic_momentum():
    length = 128
    close = np.concatenate(
        [
            np.linspace(100.0, 101.0, 64, dtype=float),
            np.linspace(101.0, 109.0, 64, dtype=float),
        ]
    )
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": close,
        "BTC/USDT:high": close + 0.1,
        "BTC/USDT:low": close - 0.1,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.full(length, 1000.0, dtype=float),
    }
    candidate = {
        "strategy_class": "PairTradingZScoreStrategy",
        "params": {
            "symbol_x": "BTC/USDT",
            "symbol_y": "ETH/USDT",
        },
    }

    _, _, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    ret = research_runner._returns_from_close(close)
    mom = np.nan_to_num(research_runner._rolling_z(ret, 64), nan=0.0)
    expected = np.where(mom >= 0.4, 1.0, np.where(mom <= -0.4, -1.0, 0.0))

    np.testing.assert_array_equal(exposure, expected)
    assert meta == {}


def test_alpha101_formula_strategy_signal_uses_event_driven_proxy():
    length = 96
    open_price = np.linspace(100.0, 112.0, length, dtype=float)
    body = np.concatenate(
        [
            np.full(24, 0.05, dtype=float),
            np.full(24, 0.35, dtype=float),
            np.full(24, -0.35, dtype=float),
            np.full(24, 0.0, dtype=float),
        ]
    )
    close = open_price + body
    high = np.maximum(open_price, close) + 0.08
    low = np.minimum(open_price, close) - 0.08
    aligned = {
        "datetime": _minute_datetimes(length),
        "BTC/USDT:open": open_price,
        "BTC/USDT:high": high,
        "BTC/USDT:low": low,
        "BTC/USDT:close": close,
        "BTC/USDT:volume": np.linspace(1000.0, 1200.0, length, dtype=float),
    }
    candidate = {
        "strategy_class": "Alpha101FormulaStrategy",
        "params": {
            "alpha_id": 101,
            "rank_window": 20,
            "history_window": 24,
            "score_window": 8,
            "entry_z": 0.9,
            "exit_z": 0.15,
            "signal_sign": 1.0,
            "stop_loss_pct": 0.03,
            "allow_short": True,
            "alpha_param_overrides": {"alpha101.101.const.001": 0.01},
        },
    }

    returns_raw, turnover, exposure, meta = research_runner._strategy_signal(
        candidate,
        aligned=aligned,
        symbols=["BTC/USDT"],
    )

    assert returns_raw.shape == (length,)
    assert turnover.shape == (length,)
    assert exposure.shape == (length,)
    assert meta.get("event_driven_proxy") is True
    assert meta.get("formulaic_alpha101") is True
    assert meta.get("alpha_param_override_count") == 1
    assert np.any(np.abs(exposure) > 0.0)
    assert np.any(turnover > 0.0)
