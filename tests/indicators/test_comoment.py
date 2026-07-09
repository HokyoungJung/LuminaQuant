"""Deterministic tests for the shared co-moment primitives.

Covers ``conditional_semibeta`` (benchmark-sign-conditioned OLS betas) and
``standardized_coskewness`` (Harvey-Siddique beta-residualized third co-moment)
with closed-form fixtures, degenerate-input ``None`` returns, and never-raise
safety.  Pure direct-import; no strategy machinery involved.
"""

from __future__ import annotations

import math

from lumina_quant.indicators.comoment import conditional_semibeta, standardized_coskewness

# --------------------------------------------------------------------------- #
# conditional_semibeta
# --------------------------------------------------------------------------- #


def test_conditional_semibeta_closed_form_two_sided() -> None:
    market = [-1.0, -3.0, 1.0, 3.0] * 12  # 24 down obs, 24 up obs
    asset = [2.0 * m if m < 0 else 0.5 * m for m in market]
    beta_minus, beta_plus = conditional_semibeta(asset, market, threshold=0.0, min_side_obs=8)
    assert beta_minus is not None and beta_plus is not None
    assert abs(beta_minus - 2.0) < 1e-9
    assert abs(beta_plus - 0.5) < 1e-9


def test_conditional_semibeta_missing_down_side_returns_none() -> None:
    market = [0.5, 1.0, 1.5, 2.0] * 10  # strictly positive => no down obs
    asset = [1.2 * m for m in market]
    beta_minus, beta_plus = conditional_semibeta(asset, market, threshold=0.0, min_side_obs=8)
    assert beta_minus is None
    assert beta_plus is not None


def test_conditional_semibeta_degenerate_side_variance_returns_none() -> None:
    # Every down bar carries the SAME market return => zero within-side variance.
    market = [-1.0, -1.0, 1.0, 2.0] * 12
    asset = [3.0 * m for m in market]
    beta_minus, beta_plus = conditional_semibeta(asset, market, threshold=0.0, min_side_obs=8)
    assert beta_minus is None
    assert beta_plus is not None


def test_conditional_semibeta_respects_min_side_obs() -> None:
    market = [-1.0, -3.0, 1.0, 3.0] * 3  # only 6 obs per side
    asset = [2.0 * m if m < 0 else 0.5 * m for m in market]
    beta_minus, beta_plus = conditional_semibeta(asset, market, threshold=0.0, min_side_obs=8)
    assert beta_minus is None
    assert beta_plus is None


def test_conditional_semibeta_never_raises_on_degenerate_input() -> None:
    assert conditional_semibeta([], []) == (None, None)
    assert conditional_semibeta([1.0], [1.0]) == (None, None)
    assert conditional_semibeta([float("nan"), 1.0, 2.0, 3.0], [-1.0, -2.0, 1.0, 2.0]) == (
        None,
        None,
    )
    assert conditional_semibeta([1.0, 2.0, 3.0, 4.0], [float("inf"), -2.0, 1.0, 2.0]) == (
        None,
        None,
    )
    # NaN threshold is rejected without raising.
    assert conditional_semibeta([1.0, 2.0], [1.0, 2.0], threshold=float("nan")) == (None, None)


def test_conditional_semibeta_alignment_trims_to_common_tail() -> None:
    market = [-1.0, -3.0, 1.0, 3.0] * 12
    asset = [2.0 * m if m < 0 else 0.5 * m for m in market]
    # Prepend junk to the asset; alignment trims to the common trailing length.
    padded_asset = [999.0, -999.0, *asset]
    beta_minus, beta_plus = conditional_semibeta(padded_asset, market, min_side_obs=8)
    assert beta_minus is not None and beta_plus is not None
    assert abs(beta_minus - 2.0) < 1e-9
    assert abs(beta_plus - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# standardized_coskewness
# --------------------------------------------------------------------------- #


def _symmetric_market(n: int) -> list[float]:
    return [(-1.0) ** t * (0.01 if (t // 2) % 2 == 0 else 0.03) for t in range(n)]


def test_standardized_coskewness_negative_for_negative_coskew_asset() -> None:
    market = _symmetric_market(120)
    c = sum(m * m for m in market) / len(market)
    asset = [-8.0 * (m * m - c) for m in market]
    coskew = standardized_coskewness(asset, market, beta_residualize=True)
    assert coskew is not None
    assert coskew < -0.3


def test_standardized_coskewness_positive_for_positive_coskew_asset() -> None:
    market = _symmetric_market(120)
    c = sum(m * m for m in market) / len(market)
    asset = [8.0 * (m * m - c) for m in market]
    coskew = standardized_coskewness(asset, market, beta_residualize=True)
    assert coskew is not None
    assert coskew > 0.3


def test_standardized_coskewness_near_zero_for_linear_asset() -> None:
    market = _symmetric_market(120)
    # A (near) linear loading plus a small symmetric idiosyncratic wiggle that is
    # decoupled from the benchmark: the beta-residualized third co-moment is ~0.
    asset = [1.3 * m + (0.001 if (t // 3) % 2 == 0 else -0.001) for t, m in enumerate(market)]
    coskew = standardized_coskewness(asset, market, beta_residualize=True)
    assert coskew is not None
    assert abs(coskew) < 0.1


def test_standardized_coskewness_perfectly_linear_asset_returns_none() -> None:
    # A perfectly linear loading residualizes to zero variance => no coskew to report.
    market = _symmetric_market(120)
    asset = [1.3 * m for m in market]
    assert standardized_coskewness(asset, market, beta_residualize=True) is None


def test_standardized_coskewness_beta_residualize_false_path() -> None:
    market = _symmetric_market(120)
    c = sum(m * m for m in market) / len(market)
    asset = [-8.0 * (m * m - c) for m in market]
    raw = standardized_coskewness(asset, market, beta_residualize=False)
    assert raw is not None and math.isfinite(raw)


def test_standardized_coskewness_constant_benchmark_returns_none() -> None:
    market = [0.01] * 120  # zero variance benchmark
    asset = list(range(120))
    assert standardized_coskewness(asset, market) is None


def test_standardized_coskewness_short_and_nonfinite_return_none() -> None:
    assert standardized_coskewness([1.0, 2.0], [1.0, 2.0]) is None  # < 3 obs
    assert standardized_coskewness([1.0, float("nan"), 3.0], [1.0, 2.0, 3.0]) is None
    assert standardized_coskewness([], []) is None


def test_standardized_coskewness_run_twice_bit_identical() -> None:
    market = _symmetric_market(120)
    c = sum(m * m for m in market) / len(market)
    asset = [-8.0 * (m * m - c) for m in market]
    first = standardized_coskewness(asset, market)
    second = standardized_coskewness(asset, market)
    assert first == second
