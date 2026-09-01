"""Deterministic unit tests for the Corwin-Schultz high-low spread indicator.

Covers the T1 INDICATOR-ISOLATION fixtures from the spread-stress lane spec (a
staircase path and an oscillation path with an IDENTICAL per-bar ``ln(H/L)``,
hence identical Parkinson volatility, that nonetheless yield a zero vs a
strictly-positive Corwin-Schultz spread), plus the never-raise degenerate
guards.  Pure closed-form fixtures; no randomness.
"""

from __future__ import annotations

import math

from lumina_quant.indicators.hl_spread import corwin_schultz_spread

# Per-bar log-range shared by BOTH isolation paths (identical Parkinson vol).
_K = math.log(1.05)
# Exact Corwin-Schultz spread for a fully-overlapping [100, 105] pair:
# alpha = k, S = 2*(e^k - 1)/(1 + e^k) = 2*(1.05 - 1)/(1 + 1.05) = 0.1 / 2.05.
_OSCILLATION_SPREAD = 2.0 * (math.exp(_K) - 1.0) / (1.0 + math.exp(_K))


def _staircase(n: int) -> tuple[list[float], list[float]]:
    """Non-overlapping staircase: bar i = [100*1.05^i, 105*1.05^i]."""
    lows = [100.0 * (1.05**i) for i in range(n)]
    highs = [105.0 * (1.05**i) for i in range(n)]
    return highs, lows


def _oscillation(n: int) -> tuple[list[float], list[float]]:
    """Fully-overlapping oscillation: every bar = [100, 105]."""
    return [105.0] * n, [100.0] * n


# --------------------------------------------------------------------------- #
# T1 indicator isolation: same Parkinson vol, divergent spread
# --------------------------------------------------------------------------- #


def test_staircase_path_has_zero_spread() -> None:
    highs, lows = _staircase(6)
    # Every bar shares ln(H/L) == k (identical Parkinson vol to the oscillation).
    for high, low in zip(highs, lows, strict=True):
        assert math.isclose(math.log(high / low), _K, rel_tol=1e-12)
    spread = corwin_schultz_spread(highs, lows, smooth_window=5)
    assert spread == 0.0


def test_oscillation_path_has_positive_spread() -> None:
    highs, lows = _oscillation(6)
    for high, low in zip(highs, lows, strict=True):
        assert math.isclose(math.log(high / low), _K, rel_tol=1e-12)
    spread = corwin_schultz_spread(highs, lows, smooth_window=5)
    assert spread is not None
    assert math.isclose(spread, _OSCILLATION_SPREAD, rel_tol=1e-8)
    assert spread > 0.0


def test_spread_is_not_parkinson_volatility() -> None:
    # The two paths have byte-identical per-bar ranges yet divergent spreads:
    # the estimator keys on adjacent-bar OVERLAP, not the per-bar range level.
    hi_a, lo_a = _staircase(6)
    hi_b, lo_b = _oscillation(6)
    spread_a = corwin_schultz_spread(hi_a, lo_a, smooth_window=5)
    spread_b = corwin_schultz_spread(hi_b, lo_b, smooth_window=5)
    assert spread_a == 0.0
    assert spread_b is not None and spread_b > spread_a


def test_single_pair_smooth_window_one() -> None:
    highs, lows = _oscillation(2)
    spread = corwin_schultz_spread(highs, lows, smooth_window=1)
    assert spread is not None
    assert math.isclose(spread, _OSCILLATION_SPREAD, rel_tol=1e-8)


def test_smoothing_averages_pairs() -> None:
    # Half clean (spread 0), half overlap (spread S): the smoothed value sits
    # strictly between 0 and S because it averages per-pair spreads.
    overlap_h, overlap_l = _oscillation(3)
    stair_h, stair_l = _staircase(3)
    highs = stair_h + overlap_h
    lows = stair_l + overlap_l
    spread = corwin_schultz_spread(highs, lows, smooth_window=5)
    assert spread is not None
    assert 0.0 < spread < _OSCILLATION_SPREAD


# --------------------------------------------------------------------------- #
# never-raise degenerate guards
# --------------------------------------------------------------------------- #


def test_short_input_returns_none() -> None:
    assert corwin_schultz_spread([105.0, 105.0], [100.0, 100.0], smooth_window=5) is None
    assert corwin_schultz_spread([], [], smooth_window=5) is None


def test_high_below_low_returns_none() -> None:
    highs = [105.0, 90.0, 105.0, 105.0, 105.0, 105.0]  # bar 1 has H < L
    lows = [100.0, 95.0, 100.0, 100.0, 100.0, 100.0]
    assert corwin_schultz_spread(highs, lows, smooth_window=5) is None


def test_non_finite_and_non_positive_return_none() -> None:
    base_low = [100.0] * 6
    assert corwin_schultz_spread([105.0, float("nan")] + [105.0] * 4, base_low) is None
    assert corwin_schultz_spread([105.0, float("inf")] + [105.0] * 4, base_low) is None
    assert corwin_schultz_spread([105.0, -5.0] + [105.0] * 4, base_low) is None
    assert corwin_schultz_spread([105.0] * 6, [100.0, 0.0] + [100.0] * 4) is None


def test_bad_types_never_raise() -> None:
    assert corwin_schultz_spread(["x", "y", "z"], [1.0, 1.0, 1.0], smooth_window=2) is None
    assert corwin_schultz_spread([None, None, None], [None, None, None], smooth_window=2) is None


def test_smooth_window_floor() -> None:
    # smooth_window <= 0 is floored to 1 (needs 2 bars).
    highs, lows = _oscillation(2)
    spread = corwin_schultz_spread(highs, lows, smooth_window=0)
    assert spread is not None and spread > 0.0
