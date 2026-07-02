"""Tests for the additive pure-numpy portfolio optimizers.

Covers ERC known-answer (uncorrelated equal-variance -> 50/50), degenerate /
near-singular covariance robustness, HRP deterministic tie-break + equal-weight
reproduction, projected-gradient validity, run-twice determinism, and the
opt-in config/param wiring (default preserves legacy behavior).
"""

from __future__ import annotations

import numpy as np
import pytest

from lumina_quant.configuration.schema import PortfolioConfig, RuntimeConfig
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.portfolio import optimizers_extra as ox
from lumina_quant.tuning.param_registry import (
    ParamRegistry,
    portfolio_optimizer_param_schema,
    register_portfolio_optimizer_params,
)


def _orthogonal_equal_var_returns() -> tuple[list[str], np.ndarray]:
    """Two uncorrelated, equal-variance return series (exact zero sample cov)."""
    a = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
    b = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0], dtype=float)
    matrix = np.column_stack([a, b]) * 0.01
    return ["A", "B"], matrix


def _valid_weights(weights: dict[str, float] | np.ndarray) -> bool:
    values = (
        np.asarray(list(weights.values()), dtype=float)
        if isinstance(weights, dict)
        else np.asarray(weights, dtype=float)
    )
    if values.size == 0:
        return True
    if not np.all(np.isfinite(values)):
        return False
    if np.any(values < -1e-12):
        return False
    return abs(float(values.sum()) - 1.0) <= 1e-9


# ---------------------------------------------------------------------------
# ERC
# ---------------------------------------------------------------------------


def test_erc_kat_uncorrelated_equal_variance_is_fifty_fifty() -> None:
    cov = np.array([[0.04, 0.0], [0.0, 0.04]], dtype=float)
    weights = ox.erc_weights(cov)
    np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-12)


def test_erc_kat_via_returns_and_registered_class() -> None:
    ids, matrix = _orthogonal_equal_var_returns()
    cls = GLOBAL_REGISTRY.get("portfolio", "ERC")
    assert cls is not None
    weights = cls().allocate(ids, matrix)
    assert _valid_weights(weights)
    np.testing.assert_allclose([weights["A"], weights["B"]], [0.5, 0.5], atol=1e-9)


def test_erc_unequal_variance_underweights_riskier_asset() -> None:
    cov = np.array([[0.01, 0.0], [0.0, 0.09]], dtype=float)
    weights = ox.erc_weights(cov)
    assert _valid_weights(weights)
    # The higher-variance asset must carry less weight for equal risk contribution.
    assert weights[0] > weights[1]
    # Risk contributions should be (approximately) equal at the solution.
    contrib = weights * (cov @ weights)
    assert abs(float(contrib[0] - contrib[1])) <= 1e-8


def test_erc_degenerate_singular_covariance_stays_finite() -> None:
    # Perfectly collinear -> rank-1, singular covariance.
    cov = np.array([[0.04, 0.04], [0.04, 0.04]], dtype=float)
    weights = ox.erc_weights(cov, max_iter=2000)
    assert _valid_weights(weights)


def test_erc_near_singular_covariance_stays_finite() -> None:
    cov = np.array([[0.04, 0.039999], [0.039999, 0.04]], dtype=float)
    weights = ox.erc_weights(cov, max_iter=5000)
    assert _valid_weights(weights)


def test_erc_single_asset_is_full_weight() -> None:
    assert ox.erc_weights(np.array([[0.02]], dtype=float)).tolist() == [1.0]


# ---------------------------------------------------------------------------
# Max-diversification / mean-variance (projected gradient, no scipy)
# ---------------------------------------------------------------------------


def test_max_diversification_valid_and_equal_for_symmetric_case() -> None:
    cov = np.array([[0.04, 0.0], [0.0, 0.04]], dtype=float)
    weights = ox.max_diversification_weights(cov)
    assert _valid_weights(weights)
    np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-6)


def test_mean_variance_equal_means_equal_var_is_equal_weight() -> None:
    mean = np.array([0.001, 0.001], dtype=float)
    cov = np.array([[0.04, 0.0], [0.0, 0.04]], dtype=float)
    weights = ox.mean_variance_weights(mean, cov, risk_aversion=5.0)
    assert _valid_weights(weights)
    np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-6)


def test_mean_variance_tilts_toward_higher_mean() -> None:
    mean = np.array([0.02, 0.001], dtype=float)
    cov = np.array([[0.04, 0.0], [0.0, 0.04]], dtype=float)
    weights = ox.mean_variance_weights(mean, cov, risk_aversion=1.0)
    assert _valid_weights(weights)
    assert weights[0] > weights[1]


def test_project_to_simplex_is_valid_projection() -> None:
    projected = ox.project_to_simplex(np.array([0.7, 0.2, -0.5, 1.3]))
    assert _valid_weights(projected)
    assert np.all(projected >= -1e-12)


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------


def test_hrp_equal_variance_uncorrelated_reproduces_equal_weights() -> None:
    ids = ["A", "B", "C", "D"]
    rng = np.random.default_rng(7)
    # Independent equal-variance columns -> singleton clusters -> equal weights.
    matrix = rng.standard_normal((256, 4)) * 0.01
    # Force exactly equal per-asset variance so inverse-variance ties resolve equal.
    matrix = matrix / matrix.std(axis=0, keepdims=True)
    weights = ox.hrp_weights_from_returns(ids, matrix, corr_threshold=0.6)
    assert _valid_weights(weights)
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.25], atol=5e-2)


def test_hrp_deterministic_tie_break_is_order_stable() -> None:
    ids = ["A", "B", "C", "D"]
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((200, 4)) * 0.02
    first = ox.hrp_weights_from_returns(ids, matrix, corr_threshold=0.6)
    second = ox.hrp_weights_from_returns(ids, matrix, corr_threshold=0.6)
    np.testing.assert_array_equal(first, second)
    assert _valid_weights(first)


def test_hrp_registered_class_matches_core() -> None:
    ids = ["A", "B", "C"]
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((128, 3)) * 0.015
    cls = GLOBAL_REGISTRY.get("portfolio", "HRP")
    assert cls is not None
    via_class = cls(corr_threshold=0.6).allocate(ids, matrix)
    via_core = ox.hrp_weights_from_returns(ids, matrix, corr_threshold=0.6)
    assert _valid_weights(via_class)
    np.testing.assert_allclose([via_class[c] for c in ids], via_core, atol=1e-12)


# ---------------------------------------------------------------------------
# Determinism (run twice -> identical) across all registered optimizers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ERC", "MaxDiversification", "MeanVariance", "HRP"])
def test_registered_optimizer_run_twice_is_identical(name: str) -> None:
    ids = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((150, 4)) * 0.01
    cls = GLOBAL_REGISTRY.get("portfolio", name)
    assert cls is not None
    first = cls().allocate(ids, matrix)
    second = cls().allocate(ids, matrix)
    assert _valid_weights(first)
    assert first == second


def test_upper_bound_caps_are_respected() -> None:
    ids = ["A", "B", "C"]
    rng = np.random.default_rng(5)
    matrix = rng.standard_normal((120, 3)) * 0.02
    weights = ox.ERCPortfolio().allocate(ids, matrix, upper={"A": 0.4, "B": 0.4, "C": 0.4})
    assert _valid_weights(weights)
    for value in weights.values():
        assert value <= 0.4 + 1e-9


def test_empty_and_mismatched_inputs_return_empty() -> None:
    assert ox.ERCPortfolio().allocate([], np.zeros((0, 0))) == {}
    # Column count mismatched with ids -> empty (defensive, no crash).
    assert ox.HRPPortfolio().allocate(["A", "B"], np.zeros((10, 3))) == {}


# ---------------------------------------------------------------------------
# Opt-in config / param wiring — default preserves legacy behavior
# ---------------------------------------------------------------------------


def test_portfolio_config_default_is_legacy() -> None:
    assert PortfolioConfig().allocation_method == "legacy"
    assert RuntimeConfig().portfolio.allocation_method == "legacy"


def test_param_schema_names_match_config_fields() -> None:
    schema = portfolio_optimizer_param_schema()
    config_fields = set(PortfolioConfig.__dataclass_fields__.keys())
    assert set(schema.keys()) == config_fields
    # allocation_method is non-tunable so search never leaves legacy silently.
    assert schema["allocation_method"].tunable is False
    assert schema["allocation_method"].default == "legacy"


def test_register_portfolio_optimizer_params_roundtrip() -> None:
    registry = ParamRegistry()
    register_portfolio_optimizer_params(registry)
    defaults = registry.default_params("portfolio_allocation")
    assert defaults["allocation_method"] == "legacy"
    assert defaults["mv_risk_aversion"] == pytest.approx(5.0)
    assert defaults["cov_window"] == 0


def _risk_contributions(cov: np.ndarray, weights: np.ndarray) -> np.ndarray:
    marginal = cov @ weights
    total = float(weights @ marginal)
    return (weights * marginal) / total


@pytest.mark.parametrize("rho", [0.5, 0.9, 0.999])
def test_erc_equalizes_risk_contributions_on_correlated_cov(rho: float) -> None:
    """Regression guard: ERC must equalize risk contributions on CORRELATED cov.

    The prior solver renormalized its working vector mid-sweep and converged to a
    wrong fixed point (e.g. [0.59, 0.41] on a symmetric correlated cov). The
    diagonal-only KATs above did not catch it because renormalization is harmless
    when off-diagonals are zero.
    """
    cov = np.array([[0.04, 0.04 * rho], [0.04 * rho, 0.04]], dtype=float)
    weights = ox.erc_weights(cov)
    # Symmetric inputs must yield symmetric weights and equal risk contributions.
    np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-6)
    rc = _risk_contributions(cov, weights)
    assert float(rc.max() - rc.min()) < 1e-8


def test_erc_equal_risk_contribution_random_correlated() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 3))
    cov = a @ a.T  # SPD, correlated
    weights = ox.erc_weights(cov)
    rc = _risk_contributions(cov, weights)
    assert float(rc.max() - rc.min()) < 1e-7
    assert rc == pytest.approx(np.full(3, 1.0 / 3.0), abs=1e-6)
