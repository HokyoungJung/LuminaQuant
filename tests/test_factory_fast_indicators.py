from __future__ import annotations

import math

import numpy as np
from lumina_quant.indicators.factory_fast import (
    composite_momentum_latest,
    rolling_range_position_latest,
    rolling_slope_latest,
    volatility_ratio_latest,
)
from lumina_quant.indicators.futures_fast import (
    normalized_true_range_latest,
    rolling_log_return_volatility_latest,
    trend_efficiency_latest,
    volume_shock_zscore_latest,
)


def test_factory_fast_core_indicators_return_finite_values():
    closes = np.linspace(100.0, 120.0, 240)
    highs = closes + 1.0
    lows = closes - 1.0

    slope = rolling_slope_latest(closes, window=32)
    range_pos = rolling_range_position_latest(highs, lows, closes, window=32)
    vol_ratio = volatility_ratio_latest(closes, fast_window=16, slow_window=64)
    momentum = composite_momentum_latest(closes, windows=(8, 21, 55), weights=(0.5, 0.3, 0.2))

    assert slope is not None and math.isfinite(slope)
    assert range_pos is not None and 0.0 <= range_pos <= 1.0
    assert vol_ratio is not None and vol_ratio > 0.0
    assert momentum is not None and math.isfinite(momentum)


def test_futures_fast_indicators_return_finite_values():
    closes = np.linspace(100.0, 140.0, 320)
    highs = closes + 0.7
    lows = closes - 0.9
    volumes = np.linspace(1000.0, 4000.0, 320)

    vol = rolling_log_return_volatility_latest(closes, window=64, annualization=1.0)
    ntr = normalized_true_range_latest(highs, lows, closes, window=64)
    shock = volume_shock_zscore_latest(volumes, window=64)
    eff = trend_efficiency_latest(closes, window=64)

    assert vol is not None and math.isfinite(vol) and vol >= 0.0
    assert ntr is not None and math.isfinite(ntr) and ntr >= 0.0
    assert shock is not None and math.isfinite(shock)
    assert eff is not None and 0.0 <= eff <= 1.0
