"""Deterministic tests for the shared trailing log-price regression primitives.

Covers the orthonormal polynomial basis (orthonormality + caching), signed-R^2
trend quality, the closed-form slope t-statistic, and the orthonormal path
convexity score -- including the load-bearing ORTHOGONALITY-TO-LEVEL property
(a linear trend added to the log-price path leaves the convexity score
unchanged) and never-raise on degenerate input.
"""

from __future__ import annotations

import math

from lumina_quant.indicators.log_price_regression import (
    log_price_trend_fit,
    orthonormal_path_convexity,
    orthonormal_polynomial_basis,
    signed_trend_quality,
    trailing_log_closes,
    trend_slope_t_stat,
)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------- #
# orthonormal polynomial basis
# --------------------------------------------------------------------------- #


def test_basis_is_orthonormal() -> None:
    for n in (3, 8, 20, 56):
        basis = orthonormal_polynomial_basis(n)
        assert basis is not None
        p0, p1, p2 = basis
        assert len(p0) == len(p1) == len(p2) == n
        # unit norm
        for vec in (p0, p1, p2):
            assert abs(_dot(vec, vec) - 1.0) < 1e-9
        # mutually orthogonal
        assert abs(_dot(p0, p1)) < 1e-9
        assert abs(_dot(p0, p2)) < 1e-9
        assert abs(_dot(p1, p2)) < 1e-9
        # p0 is the (normalized) constant vector
        assert all(abs(value - p0[0]) < 1e-12 for value in p0)
        # p1 is monotone in the index (a pure linear ramp, zero mean)
        assert abs(sum(p1)) < 1e-9
        assert all(p1[i] < p1[i + 1] for i in range(n - 1))


def test_basis_none_below_degree() -> None:
    assert orthonormal_polynomial_basis(2) is None
    assert orthonormal_polynomial_basis(1) is None


def test_basis_is_cached_identity() -> None:
    # lru_cache returns the identical object for the same window length.
    assert orthonormal_polynomial_basis(30) is orthonormal_polynomial_basis(30)


# --------------------------------------------------------------------------- #
# trailing_log_closes
# --------------------------------------------------------------------------- #


def test_trailing_log_closes_guards() -> None:
    assert trailing_log_closes([100.0, 101.0], 5) is None  # too short
    assert trailing_log_closes([100.0, -1.0, 102.0, 103.0], 4) is None  # non-positive
    assert trailing_log_closes([100.0, float("nan"), 102.0, 103.0], 4) is None
    out = trailing_log_closes([math.exp(1.0), math.exp(2.0), math.exp(3.0)], 3)
    assert out is not None
    assert all(abs(out[i] - (i + 1)) < 1e-9 for i in range(3))


# --------------------------------------------------------------------------- #
# signed trend quality
# --------------------------------------------------------------------------- #


def test_signed_trend_quality_smooth_up_and_down() -> None:
    up = [math.exp(0.01 * t + 0.001 * ((-1) ** t)) for t in range(60)]
    down = [math.exp(-0.01 * t + 0.001 * ((-1) ** t)) for t in range(60)]
    su = signed_trend_quality(up, window=56)
    sd = signed_trend_quality(down, window=56)
    assert su is not None and sd is not None
    assert su[0] > 0.9 and su[2] > 0.0  # signed_r2 > 0, slope > 0
    assert sd[0] < -0.9 and sd[2] < 0.0
    assert 0.9 < su[1] <= 1.0  # r2


def test_signed_trend_quality_zigzag_low_quality() -> None:
    # A net-directional zigzag has a much lower R^2 than a smooth trend.
    zig = [math.exp(0.01 * t + 0.05 * ((-1) ** t)) for t in range(60)]
    sq = signed_trend_quality(zig, window=56)
    assert sq is not None
    assert 0.0 < sq[1] < 0.95  # positive but degraded fit quality
    assert sq[2] > 0.0  # net upward slope


def test_signed_trend_quality_none_on_short() -> None:
    assert signed_trend_quality([100.0] * 10, window=56) is None


def test_log_price_trend_fit_matches_signed() -> None:
    series = [math.exp(0.02 * t) for t in range(40)]
    fit = log_price_trend_fit(series, window=30)
    sq = signed_trend_quality(series, window=30)
    assert fit is not None and sq is not None
    assert abs(fit[0] - sq[2]) < 1e-12  # slope
    assert abs(fit[1] - sq[1]) < 1e-12  # r2


# --------------------------------------------------------------------------- #
# slope t-statistic
# --------------------------------------------------------------------------- #


def test_trend_slope_t_stat_monotone_and_guards() -> None:
    assert trend_slope_t_stat(0.5, 2) is None  # n <= 2
    low = trend_slope_t_stat(0.2, 56)
    high = trend_slope_t_stat(0.9, 56)
    assert low is not None and high is not None
    assert high > low > 0.0
    # perfect fit saturates rather than diverging
    sat = trend_slope_t_stat(1.0, 56)
    assert sat is not None and math.isfinite(sat)


# --------------------------------------------------------------------------- #
# orthonormal path convexity
# --------------------------------------------------------------------------- #


def test_convexity_sign_convex_concave_linear() -> None:
    n = 56
    convex = [math.exp(1e-4 * (t - n) ** 2) for t in range(n)]  # U opening up
    concave = [math.exp(-1e-4 * (t - n) ** 2) for t in range(n)]
    linear = [math.exp(0.01 * t) for t in range(n)]
    cvx = orthonormal_path_convexity(convex, window=n)
    ccv = orthonormal_path_convexity(concave, window=n)
    lin = orthonormal_path_convexity(linear, window=n)
    assert cvx is not None and ccv is not None and lin is not None
    assert cvx > 0.0
    assert ccv < 0.0
    assert abs(lin) < 1e-6  # pure linear log-trend -> zero curvature


def test_convexity_orthogonal_to_linear_trend() -> None:
    """The property test: adding ANY linear trend leaves the score unchanged."""
    n = 56
    base = [math.exp(0.02 * math.sin(t / 7.0)) for t in range(n)]
    # Same path plus a linear log-ramp (slope b) and a level shift (a).
    for a, b in ((0.0, 0.01), (1.5, -0.03), (-2.0, 0.05)):
        shifted = [base[t] * math.exp(a + b * t) for t in range(n)]
        s0 = orthonormal_path_convexity(base, window=n)
        s1 = orthonormal_path_convexity(shifted, window=n)
        assert s0 is not None and s1 is not None
        assert abs(s0 - s1) < 1e-9, (a, b, s0, s1)


def test_convexity_none_on_degenerate() -> None:
    assert orthonormal_path_convexity([100.0] * 5, window=56) is None
    assert orthonormal_path_convexity([100.0, -1.0] * 30, window=56) is None
    # constant path -> zero-variance normalization is floored, curvature ~0
    const = orthonormal_path_convexity([100.0] * 56, window=56)
    assert const is None or abs(const) < 1e-6
