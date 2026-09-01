"""Deterministic checks for breakout_noise and stat_arb indicator primitives."""

from __future__ import annotations

import math

import numpy as np

from lumina_quant.indicators import (
    KalmanHedgeState,
    average_noise_ratio,
    bar_noise_ratio,
    kalman_hedge_ratio,
    kalman_hedge_ratio_step,
    kalman_spread,
    moving_average_score,
    pca_residual_sscores,
    range_volatility_target_weight,
    volatility_breakout_levels,
)


def test_bar_noise_ratio_bounds_and_degenerate() -> None:
    assert bar_noise_ratio(100, 110, 90, 110) == 0.5
    assert math.isclose(bar_noise_ratio(100, 110, 90, 100), 1.0)
    assert bar_noise_ratio(100, 110, 90, 90) == 0.5
    assert bar_noise_ratio(100, 100, 100, 100) is None
    assert bar_noise_ratio(None, 1, 0, 1) is None
    assert bar_noise_ratio(float("nan"), 1, 0, 1) is None


def test_average_noise_ratio_requires_full_window_and_skips_flat_bars() -> None:
    opens, highs, lows, closes = [100] * 20, [110] * 20, [90] * 20, [110] * 20
    assert math.isclose(average_noise_ratio(opens, highs, lows, closes, period=20), 0.5)
    assert average_noise_ratio(opens[:19], highs, lows, closes, period=20) is None
    # More than half of the window flat -> None (not enough valid bars).
    flat_highs = [100] * 15 + [110] * 5
    flat_lows = [100] * 15 + [90] * 5
    assert average_noise_ratio(opens, flat_highs, flat_lows, closes, period=20) is None


def test_volatility_breakout_levels() -> None:
    assert volatility_breakout_levels(100, 110, 90, k=0.5) == (110.0, 90.0)
    assert volatility_breakout_levels(100, 110, 90, k=0.0) == (None, None)
    assert volatility_breakout_levels(100, 90, 110, k=0.5) == (None, None)
    assert volatility_breakout_levels(None, 110, 90, k=0.5) == (None, None)


def test_moving_average_score() -> None:
    rising = list(range(1, 31))
    assert moving_average_score(rising) == 1.0
    assert moving_average_score(rising[::-1]) == 0.0
    assert moving_average_score(rising[:19]) is None
    assert moving_average_score(rising, windows=()) is None
    # Price above the short MA only: SMA3=8.17 < 8.5 < SMA5=8.9 < SMA10 < SMA20.
    path = [12.0] * 17 + [8.0, 8.0, 8.0, 8.5]
    assert moving_average_score(path, windows=(3, 5, 10, 20)) == 0.25


def test_range_volatility_target_weight() -> None:
    assert math.isclose(range_volatility_target_weight(110, 90, 100, target_vol=0.02), 0.1)
    assert range_volatility_target_weight(101, 99, 100, target_vol=0.05, cap=1.0) == 1.0
    assert range_volatility_target_weight(101, 99, 100, target_vol=0.05, cap=2.0) == 2.0
    assert range_volatility_target_weight(100, 100, 100, target_vol=0.02) is None
    assert range_volatility_target_weight(110, 90, 100, target_vol=0.0) is None


def test_kalman_hedge_ratio_recovers_static_and_drifting_beta() -> None:
    rng = np.random.default_rng(0)
    xs = np.cumsum(rng.normal(size=400)) + 50.0
    ys = 2.0 * xs + 1.0 + rng.normal(scale=0.01, size=400)
    state = kalman_hedge_ratio(ys, xs, delta=1e-5, obs_noise=1e-2)
    assert state is not None and state.updates == 400
    assert abs(state.beta - 2.0) < 0.05
    assert abs(kalman_spread(state, ys[-1], xs[-1])) < 0.5
    assert state.innovation_z is not None
    # Round-trip serialisation.
    restored = KalmanHedgeState.from_dict(state.to_dict())
    assert restored == state
    assert KalmanHedgeState.from_dict({"beta": 1}) is None
    # Drifting beta is tracked with a larger process noise.
    xs2 = np.cumsum(rng.normal(size=600)) + 100.0
    beta_path = np.where(np.arange(600) < 300, 1.0, 1.5)
    ys2 = beta_path * xs2 + rng.normal(scale=0.05, size=600)
    tracked = kalman_hedge_ratio(ys2, xs2, delta=1e-3, obs_noise=1e-2)
    assert tracked is not None and abs(tracked.beta - 1.5) < 0.1
    # Bad observations are ignored, not fatal.
    assert kalman_hedge_ratio_step(None, float("nan"), 1.0) is None
    assert kalman_hedge_ratio([1.0], [1.0]) is None


def test_pca_residual_sscores_shape_and_gating() -> None:
    rng = np.random.default_rng(3)
    rows, cols = 120, 5
    factor = rng.normal(size=rows)
    panel = np.column_stack(
        [factor * 0.02 + rng.normal(scale=0.005, size=rows) for _ in range(cols)]
    )
    eps = np.zeros(rows)
    for t in range(1, rows):
        eps[t] = 0.7 * eps[t - 1] + rng.normal(scale=0.01)
    panel[:, 0] = factor * 0.02 + np.diff(np.concatenate([[0.0], eps]))
    scores = pca_residual_sscores(panel.tolist(), n_factors=1)
    assert len(scores) == cols
    assert all(s is None or math.isfinite(s) for s in scores)
    assert scores[0] is not None  # the injected OU residual is estimable
    # Too short / too narrow / non-finite -> all None (never raises).
    assert pca_residual_sscores(panel[:10].tolist(), n_factors=1) == [None] * cols
    assert pca_residual_sscores(panel[:, :2].tolist(), n_factors=1) == [None, None]
    bad = panel.copy()
    bad[5, 2] = float("nan")
    assert pca_residual_sscores(bad.tolist(), n_factors=1) == [None] * cols
    assert pca_residual_sscores([], n_factors=1) == []
    # A zero-variance column is skipped, others still scored.
    flat = panel.copy()
    flat[:, 1] = 0.0
    flat_scores = pca_residual_sscores(flat.tolist(), n_factors=1)
    assert flat_scores[1] is None and any(s is not None for s in flat_scores)
