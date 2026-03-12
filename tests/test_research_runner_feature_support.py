from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from lumina_quant.strategy_factory import research_runner


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
            "liquidation_long_notional": np.linspace(100_000.0, 140_000.0, datetimes.len(), dtype=float),
            "liquidation_short_notional": np.linspace(30_000.0, 40_000.0, datetimes.len(), dtype=float),
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
