from __future__ import annotations

import numpy as np

from lumina_quant.indicators.advanced_alpha import (
    cross_leadlag_spillover,
    perp_crowding_score,
    pv_trend_score,
    volcomp_vwap_pressure,
)


def _build_ohlcv(length: int = 320):
    idx = np.arange(length, dtype=float)
    close = 100.0 + (0.03 * idx) + np.sin(idx / 12.0)
    high = close * (1.0 + 0.004)
    low = close * (1.0 - 0.004)
    volume = 1000.0 + (40.0 * np.sin(idx / 8.0)) + (idx * 0.3)
    return high, low, close, volume


def test_pv_trend_score_exposes_gate_and_components():
    high, low, close, volume = _build_ohlcv(420)

    result = pv_trend_score(
        close,
        volume,
        high=high,
        low=low,
    )

    assert result["available"] is True
    assert isinstance(result["gate"], bool)
    assert -10.0 <= float(result["score"]) <= 10.0
    assert 0.0 <= float(result["trend_efficiency"]) <= 1.0


def test_volcomp_vwap_pressure_outputs_compression_flags():
    high, low, close, volume = _build_ohlcv(420)

    # Flatten latest segment to induce compression.
    close[-80:] = close[-80] + np.linspace(0.0, 0.1, 80)
    high[-80:] = close[-80:] * 1.001
    low[-80:] = close[-80:] * 0.999

    result = volcomp_vwap_pressure(high, low, close, volume)

    assert result["available"] is True
    assert isinstance(result["active"], bool)
    assert np.isfinite(float(result["score"]))
    assert -20.0 <= float(result["deviation_z"]) <= 20.0


def test_cross_leadlag_spillover_excludes_metals_and_predicts_lagger():
    n = 300
    idx = np.arange(n, dtype=float)

    btc = 100 + np.cumsum(0.05 + 0.02 * np.sin(idx / 9.0))
    eth = 90 + np.cumsum(0.04 + 0.02 * np.sin(idx / 10.0))
    bnb = 70 + np.cumsum(0.03 + 0.02 * np.sin(idx / 11.0))
    sol = 60 + np.cumsum(0.03 + 0.01 * np.sin(idx / 7.0))

    # Laggard follows leader basket with one-bar lag.
    lagger = 40 + np.cumsum(np.r_[0.0, np.diff((btc + eth + bnb + sol) / 4.0)[:-1]])

    result = cross_leadlag_spillover(
        {
            "BTC/USDT": btc,
            "ETH/USDT": eth,
            "BNB/USDT": bnb,
            "SOL/USDT": sol,
            "ADA/USDT": lagger,
            "XAU/USDT": np.linspace(1900, 1910, n),
            "XAG/USDT": np.linspace(23, 24, n),
        },
        max_lag=3,
        ridge_alpha=1.0,
        window=240,
    )

    assert result["available"] is True
    assert "XAU/USDT" in set(result["excluded_symbols"])
    assert "XAG/USDT" in set(result["excluded_symbols"])
    assert "ADA/USDT" in set(result["predictions"])
    ada = result["predictions"]["ADA/USDT"]
    assert np.isfinite(float(ada["score"]))


def test_perp_crowding_score_outputs_component_zscores():
    n = 220
    idx = np.arange(n, dtype=float)

    funding = 0.0002 + (0.00005 * np.sin(idx / 7.0))
    open_interest = 1_000_000 + np.cumsum(5_000 + 900 * np.sin(idx / 15.0))
    liq_long = 100_000 + 5000 * np.sin(idx / 6.0)
    liq_short = 90_000 + 4000 * np.cos(idx / 5.5)

    result = perp_crowding_score(
        funding_rate=funding,
        open_interest=open_interest,
        liquidation_long_notional=liq_long,
        liquidation_short_notional=liq_short,
        window=96,
    )

    assert result["available"] is True
    assert -1.0 <= float(result["score"]) <= 1.0
    assert "components" in result
    assert np.isfinite(float(result["components"]["funding_z"]))
    assert np.isfinite(float(result["components"]["oi_delta_z"]))
