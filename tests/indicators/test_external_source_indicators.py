"""Deterministic tests for the external-source indicator primitives.

Covers the five modules absorbed from the DeepLearning / precious-metal /
Reference-Price research repos: spectral_cycle (trailing periodogram),
stationarity (pure-Python ADF + AR(1) half-life), garch (deterministic
variance-targeted GARCH(1,1)), flow_share (turnover-share / CDF extremeness),
and ensemble_weights (inverse-error / softmax / disagreement / hit-rate).
All fixtures are synthetic and seeded via a hand-rolled LCG so every value is
reproducible bit-for-bit.  No backtest is run.
"""

from __future__ import annotations

import math

from lumina_quant.indicators.ensemble_weights import (
    blend_weights,
    direction_hit_rate,
    disagreement_coefficient,
    inverse_error_weights,
    softmax_weights,
)
from lumina_quant.indicators.flow_share import (
    cdf_extremeness,
    cross_sectional_share,
    dense_rank_desc,
    normal_cdf,
    share_weighted_return,
)
from lumina_quant.indicators.garch import garch11_fit, garch11_next_variance
from lumina_quant.indicators.spectral_cycle import cycle_phase_fraction, dominant_cycle
from lumina_quant.indicators.stationarity import (
    adf_critical_value,
    adf_t_statistic,
    ar1_half_life,
)

_TWO_PI = 2.0 * math.pi


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield (state / 2**31 - 0.5) * 2.0


def _noise(seed: int, count: int) -> list[float]:
    gen = _lcg(seed)
    return [next(gen) for _ in range(count)]


# --------------------------------------------------------------------------- #
# spectral_cycle
# --------------------------------------------------------------------------- #
def test_dominant_cycle_recovers_pure_tone() -> None:
    period = 16.0
    xs = [0.05 * math.cos(_TWO_PI * t / period) for t in range(96)]
    result = dominant_cycle(xs, min_period=4, max_period=40)
    assert result is not None
    found_period, amplitude, phase, purity = result
    assert found_period == period
    assert abs(amplitude - 0.05) < 0.005
    # Latest bar t=95: phase fraction = (95 mod 16) / 16 = 0.9375.
    fraction = cycle_phase_fraction(phase)
    assert fraction is not None and abs(fraction - 0.9375) < 0.02
    assert purity > 0.2


def test_dominant_cycle_purity_separates_tone_from_noise() -> None:
    period = 16.0
    tone = [0.05 * math.cos(_TWO_PI * t / period) for t in range(96)]
    noise = _noise(3, 96)
    tone_result = dominant_cycle(tone, min_period=4, max_period=40)
    noise_result = dominant_cycle(noise, min_period=4, max_period=40)
    assert tone_result is not None and noise_result is not None
    assert tone_result[3] > 2.5 * noise_result[3]


def test_dominant_cycle_guards() -> None:
    assert dominant_cycle([1.0, 2.0, 3.0], min_period=4, max_period=20) is None
    assert dominant_cycle([], min_period=4, max_period=20) is None
    assert dominant_cycle([float("nan")] * 64, min_period=4, max_period=20) is None
    # Pure line detrends to ~zero everywhere -> no meaningful cycle allowed to win.
    line = [float(t) for t in range(64)]
    result = dominant_cycle(line, min_period=4, max_period=20)
    assert result is None or result[1] < 1e-9
    # Scan range empty when the window cannot hold two full min_period cycles.
    assert dominant_cycle([1.0, 0.0] * 4, min_period=8, max_period=32) is None


def test_cycle_phase_fraction_wraps_and_guards() -> None:
    assert cycle_phase_fraction(0.0) == 0.0
    fraction = cycle_phase_fraction(math.pi)
    assert fraction is not None and abs(fraction - 0.5) < 1e-12
    wrapped = cycle_phase_fraction(2.5 * _TWO_PI)
    assert wrapped is not None and abs(wrapped - 0.5) < 1e-12
    assert cycle_phase_fraction(float("inf")) is None


# --------------------------------------------------------------------------- #
# stationarity
# --------------------------------------------------------------------------- #
def _ar_series(seed: int, phi: float, count: int) -> list[float]:
    gen = _lcg(seed)
    series = [0.0]
    for _ in range(count):
        series.append(phi * series[-1] + next(gen))
    return series


def _random_walk(seed: int, count: int) -> list[float]:
    gen = _lcg(seed)
    series = [0.0]
    for _ in range(count):
        series.append(series[-1] + next(gen))
    return series


def test_adf_discriminates_stationary_from_unit_root() -> None:
    critical = adf_critical_value("5%")
    assert critical == -2.86
    stationary_stat = adf_t_statistic(_ar_series(42, 0.5, 300), lags=1)
    walk_stat = adf_t_statistic(_random_walk(7, 300), lags=1)
    assert stationary_stat is not None and stationary_stat < critical
    assert walk_stat is not None and walk_stat > critical


def test_adf_guards_and_critical_values() -> None:
    assert adf_critical_value("1%") == -3.43
    assert adf_critical_value("10%") == -2.57
    assert adf_critical_value("7%") is None
    assert adf_t_statistic([1.0, 2.0, 3.0], lags=1) is None
    assert adf_t_statistic([float("nan")] * 50, lags=0) is None
    # A constant series has zero-variance regressors -> singular -> None.
    assert adf_t_statistic([5.0] * 50, lags=0) is None


def test_ar1_half_life_matches_theory() -> None:
    # phi = 0.5 -> half-life = -ln(2)/ln(0.5) = 1 bar.
    half_life = ar1_half_life(_ar_series(42, 0.5, 300))
    assert half_life is not None and 0.5 < half_life < 2.0
    # A near-unit-root walk yields either no fit or a very long half-life.
    walk_half_life = ar1_half_life(_random_walk(7, 300))
    assert walk_half_life is None or walk_half_life > 20.0
    assert ar1_half_life([1.0, 2.0, 3.0]) is None
    assert ar1_half_life([5.0] * 50) is None


# --------------------------------------------------------------------------- #
# garch
# --------------------------------------------------------------------------- #
def _clustered_returns(seed: int, count: int) -> list[float]:
    gen = _lcg(seed)
    returns = []
    for i in range(count):
        sigma = 0.03 if (i // 100) % 2 == 0 else 0.01
        returns.append(sigma * next(gen))
    return returns


def test_garch11_fit_is_deterministic_and_bounded() -> None:
    returns = _clustered_returns(99, 400)
    fit = garch11_fit(returns)
    assert fit is not None
    omega, alpha, beta = fit
    assert omega > 0.0
    assert 0.02 <= alpha <= 0.30
    assert 0.60 <= beta <= 0.98
    assert alpha + beta <= 0.998
    assert garch11_fit(returns) == fit


def test_garch11_next_variance_reacts_to_shock() -> None:
    returns = _clustered_returns(99, 400)
    fit = garch11_fit(returns)
    assert fit is not None
    omega, alpha, beta = fit
    calm = garch11_next_variance(returns, omega=omega, alpha=alpha, beta=beta)
    shocked = garch11_next_variance([*returns, 0.10], omega=omega, alpha=alpha, beta=beta)
    assert calm is not None and calm > 0.0
    assert shocked is not None and shocked > calm


def test_garch11_guards() -> None:
    assert garch11_fit([0.001] * 10) is None  # too short
    assert garch11_fit([0.0] * 100) is None  # flat -> no variance
    returns = _clustered_returns(5, 100)
    assert garch11_next_variance(returns, omega=-1.0, alpha=0.1, beta=0.8) is None
    assert garch11_next_variance(returns, omega=1e-6, alpha=0.6, beta=0.6) is None
    assert garch11_next_variance(returns, omega=float("nan"), alpha=0.1, beta=0.8) is None


def test_garch11_huge_finite_returns_return_none_not_raise() -> None:
    # 1e200 passes the isfinite gate but its squared deviation overflows the
    # float range; the contract is None, never OverflowError.
    huge = [1e200, -1e200] * 24
    assert garch11_fit(huge) is None
    assert garch11_next_variance(huge, omega=1e-6, alpha=0.05, beta=0.9) is None


# --------------------------------------------------------------------------- #
# flow_share
# --------------------------------------------------------------------------- #
def test_cross_sectional_share_and_rank() -> None:
    assert cross_sectional_share(25.0, 100.0) == 0.25
    assert cross_sectional_share(-1.0, 100.0) is None
    assert cross_sectional_share(1.0, 0.0) is None
    assert dense_rank_desc([5.0, 3.0, 3.0, 1.0], 3.0) == 2
    assert dense_rank_desc([5.0, 3.0, 1.0], 5.0) == 1
    assert dense_rank_desc([], 1.0) is None
    assert dense_rank_desc([1.0, float("nan")], 1.0) is None


def test_cdf_extremeness_bounds() -> None:
    assert normal_cdf(0.0) == 0.5
    assert cdf_extremeness(0.0) == 0.0
    high = cdf_extremeness(3.0)
    low = cdf_extremeness(-3.0)
    assert high is not None and low is not None
    assert high > 0.99 and abs(high - low) < 1e-12
    assert cdf_extremeness(float("inf")) is None


def test_share_weighted_return() -> None:
    result = share_weighted_return([0.3, 0.2], [0.01, -0.02])
    expected = (0.3 * 0.01 + 0.2 * -0.02) / 0.5
    assert result is not None and abs(result - expected) < 1e-12
    assert share_weighted_return([0.0, 0.0], [0.01, 0.02]) is None
    assert share_weighted_return([0.5], [0.01, 0.02]) is None
    assert share_weighted_return([-0.1, 0.6], [0.01, 0.02]) is None


# --------------------------------------------------------------------------- #
# ensemble_weights
# --------------------------------------------------------------------------- #
def test_inverse_error_weights_orders_by_error() -> None:
    weights = inverse_error_weights([0.01, 0.02, 0.04])
    assert weights is not None
    assert abs(sum(weights) - 1.0) < 1e-12
    assert weights[0] > weights[1] > weights[2]
    assert inverse_error_weights([]) is None
    assert inverse_error_weights([-0.01, 0.02]) is None


def test_softmax_weights_temperature() -> None:
    sharp = softmax_weights([1.0, 2.0], temperature=0.1)
    soft = softmax_weights([1.0, 2.0], temperature=10.0)
    assert sharp is not None and soft is not None
    assert sharp[1] > soft[1] > 0.5
    assert abs(sum(sharp) - 1.0) < 1e-12
    assert softmax_weights([1.0, 2.0], temperature=0.0) is None


def test_blend_weights_ratio() -> None:
    blended = blend_weights([1.0, 0.0], [0.5, 0.5], ratio=0.5)
    assert blended is not None
    assert abs(blended[0] - 0.75) < 1e-12 and abs(sum(blended) - 1.0) < 1e-12
    assert blend_weights([1.0], [0.5, 0.5], ratio=0.5) is None
    assert blend_weights([1.0, 0.0], [0.5, 0.5], ratio=1.5) is None
    assert blend_weights([0.0, 0.0], [0.0, 0.0], ratio=0.5) is None


def test_disagreement_coefficient_gate() -> None:
    consensus = disagreement_coefficient([1.0, 1.02, 0.98])
    divergent = disagreement_coefficient([1.0, -1.0, 3.0])
    assert consensus is not None and divergent is not None
    assert consensus < 0.05 < divergent
    assert disagreement_coefficient([1.0]) is None
    assert disagreement_coefficient([1.0, -1.0]) is None  # near-zero mean
    # Huge-but-finite inputs must return None, never OverflowError.
    assert disagreement_coefficient([1e200, -1e200, 5e199]) is None


def test_direction_hit_rate_deadband() -> None:
    rate = direction_hit_rate([0.1, -0.2, 0.3], [0.05, -0.1, -0.2])
    assert rate is not None and abs(rate - 2.0 / 3.0) < 1e-12
    banded = direction_hit_rate([0.1, -0.2], [0.001, -0.1], deadband=0.01)
    assert banded == 1.0  # first pair excluded by the deadband
    assert direction_hit_rate([0.1], [0.0], deadband=0.01) is None
    assert direction_hit_rate([0.1, 0.2], [0.1]) is None
