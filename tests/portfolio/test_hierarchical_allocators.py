"""Analytic / counterexample checks for the hierarchical and robust allocators."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY
from lumina_quant.portfolio import hierarchical as hier
from lumina_quant.portfolio.quality_gated_allocation import (
    _build_allocator,
    allocate_quality_gated,
    build_allocation_manifest,
)


def _two_pair_cov() -> np.ndarray:
    # (A,B) and (C,D) are 0.9-correlated pairs; A,B unit variance, C,D variance 4.
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = 0.9
    corr[2, 3] = corr[3, 2] = 0.9
    std = np.array([1.0, 1.0, 2.0, 2.0])
    return corr * np.outer(std, std)


def _brute_force_simplex(objective, n: int, grid: int = 60) -> np.ndarray:
    """Exhaustive simplex grid search (n <= 3) used as an independent oracle."""
    best_val, best_w = np.inf, None
    for combo in itertools.product(range(grid + 1), repeat=n - 1):
        if sum(combo) > grid:
            continue
        w = np.array([*combo, grid - sum(combo)], dtype=float) / grid
        val = objective(w)
        if val < best_val:
            best_val, best_w = val, w
    return best_w


# Hand-derived target: IVP cluster variances 0.95 vs 3.8 -> alpha_left = 0.8,
# equal split inside each pair -> (0.4, 0.4, 0.1, 0.1).
_TARGET = np.array([0.4, 0.4, 0.1, 0.1])


def test_linkage_and_seriation_group_correlated_pairs() -> None:
    cov = _two_pair_cov()
    dist = hier.correlation_distance(hier._cov_to_corr(cov))
    link = hier.hierarchical_linkage(dist, method="single")
    assert link.shape == (3, 4)
    assert np.isclose(link[0, 2], np.sqrt(0.05)) and np.isclose(link[1, 2], np.sqrt(0.05))
    assert link[2, 3] == 4.0
    order = hier.quasi_diagonal_order(link, 4)
    assert sorted(order) == [0, 1, 2, 3]
    assert {order[0], order[1]} in ({0, 1}, {2, 3})
    assert hier.cut_tree(link, 4, n_clusters=2) == [0, 0, 1, 1]
    assert hier.cut_tree(link, 4, n_clusters=4) == [0, 1, 2, 3]


def test_hrp_dendrogram_matches_hand_derivation() -> None:
    cov = _two_pair_cov()
    for method in ("single", "complete", "average", "ward"):
        w = hier.hrp_dendrogram_weights(cov, linkage_method=method)
        assert np.allclose(w, _TARGET, atol=1e-9), (method, w)
        assert np.isclose(w.sum(), 1.0)
    assert np.allclose(hier.hrp_dendrogram_weights(np.diag([1.0, 4.0])), [0.8, 0.2])
    assert hier.hrp_dendrogram_weights(np.zeros((0, 0))).size == 0
    assert np.allclose(hier.hrp_dendrogram_weights(np.array([[2.0]])), [1.0])


def test_constrained_hrp_box_projection_and_fail_closed() -> None:
    cov = _two_pair_cov()
    w = hier.hrp_dendrogram_weights(cov, bounds={"lower": 0.05, "upper": 0.30})
    assert np.isclose(w.sum(), 1.0)
    assert w.max() <= 0.30 + 1e-9 and w.min() >= 0.05 - 1e-9
    assert np.isclose(w[0], 0.30) and np.isclose(w[1], 0.30)
    # Auditor counterexample: unconstrained tilt would push asset 2 above 0.4 and
    # asset 1 below its 0.2 floor; both bounds must hold after projection.
    cov3 = np.diag([0.1, 0.1, 0.01])
    lo, hi = np.array([0.05, 0.2, 0.1]), np.array([0.6, 0.4, 0.8])
    w3 = hier.hrp_dendrogram_weights(cov3, bounds={"lower": lo, "upper": hi})
    assert np.all(w3 >= lo - 1e-9) and np.all(w3 <= hi + 1e-9)
    assert np.isclose(w3.sum(), 1.0)
    # Infeasible boxes fail closed instead of silently ignoring the constraint.
    with pytest.raises(ValueError):
        hier.hrp_dendrogram_weights(cov, bounds={"lower": 0.5, "upper": 0.6})
    with pytest.raises(ValueError):
        hier.hrp_dendrogram_weights(cov, bounds={"lower": 0.0, "upper": 0.2})
    with pytest.raises(ValueError):
        hier.project_box_simplex([0.5, 0.5], [0.6, 0.6], [1.0, 1.0])
    # Box-simplex projection is exact on a known case.
    proj = hier.project_box_simplex([0.9, 0.05, 0.05], [0.1, 0.1, 0.1], [0.5, 1.0, 1.0])
    assert np.isclose(proj.sum(), 1.0) and np.isclose(proj[0], 0.5)
    assert np.allclose(proj[1:], 0.25)


def test_herc_and_nco_agree_with_hrp_on_symmetric_two_cluster_case() -> None:
    cov = _two_pair_cov()
    assert np.allclose(hier.herc_weights(cov, n_clusters=2), _TARGET, atol=1e-9)
    assert np.allclose(hier.nco_weights(cov, n_clusters=2), _TARGET, atol=1e-6)
    for fn in (hier.herc_weights, hier.nco_weights):
        w = fn(cov)
        assert np.isclose(w.sum(), 1.0) and np.all(w >= -1e-12)
        assert np.allclose(w, fn(cov))


def test_herc_uses_inverse_vol_inside_clusters() -> None:
    # Two independent clusters; inside cluster A the variances are 1 and 4:
    # HERC (naive risk parity) splits 2:1 (1/sigma), HRP's IVP would split 4:1.
    corr = np.eye(4)
    corr[0, 1] = corr[1, 0] = 0.9
    corr[2, 3] = corr[3, 2] = 0.9
    std = np.array([1.0, 2.0, 1.0, 1.0])
    cov = corr * np.outer(std, std)
    w = hier.herc_weights(cov, n_clusters=2)
    assert np.isclose(w[0] / w[1], 2.0, atol=1e-9)
    hrp = hier.hrp_dendrogram_weights(cov)
    assert np.isclose(hrp[0] / hrp[1], 4.0, atol=1e-9)


def test_silhouette_singletons_score_zero_and_two_clusters_win() -> None:
    cov = _two_pair_cov()
    dist = hier.correlation_distance(hier._cov_to_corr(cov))
    link = hier.hierarchical_linkage(dist)
    # k = 3 forces a singleton; it must not be rewarded with silhouette 1.
    assert hier._mean_silhouette(dist, np.array([0, 0, 1, 2])) < hier._mean_silhouette(
        dist, np.array([0, 0, 1, 1])
    )
    assert hier.silhouette_optimal_clusters(dist, link) == 2


def test_long_only_min_variance_matches_brute_force_and_kkt() -> None:
    cov = np.array([[0.04, 0.02, 0.0], [0.02, 0.09, -0.01], [0.0, -0.01, 0.16]])
    w = hier.long_only_min_variance(cov)
    oracle = _brute_force_simplex(lambda v: float(v @ cov @ v), 3, grid=200)
    assert np.isclose(w.sum(), 1.0) and np.all(w >= -1e-12)
    assert float(w @ cov @ w) <= float(oracle @ cov @ oracle) + 1e-9
    assert hier.min_variance_kkt_residual(cov, w) < 1e-8
    # A case where the unconstrained solution shorts an asset: the long-only
    # optimum must sit on the boundary (asset 2 at zero) -- not a clipped
    # analytic answer.
    cov_short = np.array([[0.01, 0.009, 0.02], [0.009, 0.01, 0.0], [0.02, 0.0, 0.5]])
    assert np.linalg.eigvalsh(cov_short).min() > 0.0  # a real (PD) covariance
    unconstrained = np.linalg.solve(cov_short, np.ones(3))
    assert (unconstrained / unconstrained.sum())[2] < 0.0  # analytic answer shorts asset 2
    w2 = hier.long_only_min_variance(cov_short)
    assert np.allclose(w2, [0.5, 0.5, 0.0], atol=1e-9)
    oracle2 = _brute_force_simplex(lambda v: float(v @ cov_short @ v), 3, grid=200)
    assert float(w2 @ cov_short @ w2) <= float(oracle2 @ cov_short @ oracle2) + 1e-9
    assert hier.min_variance_kkt_residual(cov_short, w2) < 1e-8


def test_long_only_max_sharpe_matches_brute_force() -> None:
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.16]])
    mu = np.array([0.03, 0.05, -0.02])
    w = hier.long_only_max_sharpe(cov, mu)

    def neg_sharpe(v: np.ndarray) -> float:
        var = float(v @ cov @ v)
        return np.inf if var <= 0 else -float(v @ mu) / np.sqrt(var)

    oracle = _brute_force_simplex(neg_sharpe, 3, grid=200)
    assert np.isclose(w.sum(), 1.0) and np.all(w >= -1e-12)
    assert neg_sharpe(w) <= neg_sharpe(oracle) + 1e-9
    assert w[2] < 1e-9  # negative-mean asset is excluded, not shorted
    # The positive-return QP substitution does not cover all-negative means;
    # silently returning min-variance solves the wrong objective.
    with pytest.raises(ValueError, match="positive"):
        hier.long_only_max_sharpe(cov, -np.abs(mu))


def test_long_only_max_sharpe_invalid_solver_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hier, "_active_set_qp", lambda *args, **kwargs: np.zeros(2))
    with pytest.raises(RuntimeError, match="max-Sharpe solver"):
        hier.long_only_max_sharpe(np.eye(2), np.array([0.1, 0.2]))


def test_nco_max_sharpe_tilts_toward_higher_mean() -> None:
    cov = np.diag([1.0, 1.0, 1.0, 1.0])
    mu = np.array([0.02, 0.01, 0.01, 0.01])
    w = hier.nco_weights(cov, mu=mu, n_clusters=2)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]
    with pytest.raises(ValueError, match="positive"):
        hier.nco_weights(cov, mu=-mu, n_clusters=2)


def test_wasserstein_dro_bcz_exact_formulation() -> None:
    cov = np.array([[0.04, 0.02, 0.0], [0.02, 0.09, -0.01], [0.0, -0.01, 0.16]])
    # radius = 0 (no target) == long-only min variance on a NON-diagonal covariance.
    w0 = hier.wasserstein_dro_weights(cov, radius=0.0)
    assert np.allclose(w0, hier.long_only_min_variance(cov), atol=1e-6)
    assert hier.min_variance_kkt_residual(cov, w0) < 1e-6
    # radius > 0: the returned point minimises the robust objective against a
    # brute-force simplex oracle and against min-var / equal-weight alternatives.
    radius = 0.02
    w = hier.wasserstein_dro_weights(cov, radius=radius)

    def obj(v: np.ndarray) -> float:
        return hier.wasserstein_dro_objective(cov, v, radius=radius)

    oracle = _brute_force_simplex(obj, 3, grid=200)
    assert obj(w) <= obj(oracle) + 1e-9
    assert obj(w) <= obj(w0) + 1e-12 and obj(w) <= obj(np.full(3, 1 / 3)) + 1e-12
    # Larger radius pulls toward 1/N (diagonal case, strict monotone distance).
    diag = np.diag([1.0, 4.0, 9.0])
    prev = np.linalg.norm(hier.wasserstein_dro_weights(diag, radius=0.0) - 1 / 3)
    for r in (0.01, 0.1, 1.0, 10.0):
        dist = np.linalg.norm(hier.wasserstein_dro_weights(diag, radius=r) - 1 / 3)
        assert dist <= prev + 1e-9
        prev = dist
    # Target-return constraint: robust mean mu'w - eps||w|| >= target holds when
    # feasible, and an unreachable target raises.
    mu = np.array([0.001, 0.003, 0.002])
    eps = np.sqrt(1e-5)

    def robust_mean(v: np.ndarray) -> float:
        return float(mu @ v) - eps * float(np.linalg.norm(v))

    w_free = hier.wasserstein_dro_weights(cov, radius=1e-5)
    # Independent oracle for the feasibility ceiling (max robust mean on the simplex).
    ceiling = robust_mean(_brute_force_simplex(lambda v: -robust_mean(v), 3, grid=200))
    assert robust_mean(w_free) < ceiling  # so a target in between is feasible AND binding
    target = 0.5 * (robust_mean(w_free) + ceiling)
    w_t = hier.wasserstein_dro_weights(cov, mu=mu, radius=1e-5, target_return=target)
    assert robust_mean(w_t) >= target - 1e-9
    assert np.isclose(w_t.sum(), 1.0) and np.all(w_t >= -1e-12)
    # binding target costs risk versus the unconstrained robust min-var
    assert hier.wasserstein_dro_objective(cov, w_t, radius=1e-5) > (
        hier.wasserstein_dro_objective(cov, w_free, radius=1e-5)
    )
    # ... and is the cheapest feasible point against the brute-force oracle.
    feasible_oracle = _brute_force_simplex(
        lambda v: (
            hier.wasserstein_dro_objective(cov, v, radius=1e-5)
            if robust_mean(v) >= target - 1e-12
            else np.inf
        ),
        3,
        grid=200,
    )
    assert hier.wasserstein_dro_objective(cov, w_t, radius=1e-5) <= (
        hier.wasserstein_dro_objective(cov, feasible_oracle, radius=1e-5) + 1e-9
    )
    with pytest.raises(ValueError):
        hier.wasserstein_dro_weights(cov, mu=mu, radius=1e-5, target_return=ceiling + 1e-3)
    # A low target is non-binding -> identical to the unconstrained solution.
    assert np.allclose(
        hier.wasserstein_dro_weights(cov, mu=mu, radius=1e-5, target_return=-1.0), w_free
    )
    radius_zero_cov = np.diag([1.0, 100.0, 10_000.0])
    radius_zero = hier.wasserstein_dro_weights(
        radius_zero_cov,
        mu=np.ones(3),
        radius=0.0,
        target_return=-1.0,
        max_iter=1,
    )
    np.testing.assert_allclose(
        radius_zero,
        hier.long_only_min_variance(radius_zero_cov, max_iter=1),
    )
    with pytest.raises(ValueError, match="radius"):
        hier.wasserstein_dro_weights(cov, radius=-1e-6)
    with pytest.raises(ValueError, match="expected returns"):
        hier.wasserstein_dro_weights(cov, mu=[np.nan, 0.0, 0.0], target_return=0.0)
    with pytest.raises(RuntimeError, match="certificate"):
        hier.wasserstein_dro_weights(np.diag([1.0, 100.0, 10_000.0]), radius=0.1, max_iter=1)
    with pytest.raises(ValueError, match="singleton"):
        hier.wasserstein_dro_weights([[1.0]], mu=[0.0], radius=0.01, target_return=0.0)


def test_indefinite_covariance_fails_closed_but_noise_is_clipped() -> None:
    indefinite = np.array([[0.01, 0.009, 0.05], [0.009, 0.01, 0.0], [0.05, 0.0, 0.5]])
    assert np.linalg.eigvalsh(indefinite).min() < -1e-4
    for fn in (
        hier.hrp_dendrogram_weights,
        hier.herc_weights,
        hier.nco_weights,
        hier.long_only_min_variance,
        hier.wasserstein_dro_weights,
        hier.graph_inverse_centrality_weights,
    ):
        with pytest.raises(ValueError):
            fn(indefinite)
    noisy = np.diag([1.0, 2.0, 3.0]).astype(float)
    noisy[0, 0] = -1e-12  # numerical noise, not a real defect
    w = hier.hrp_dendrogram_weights(noisy)
    assert np.isclose(w.sum(), 1.0)


def test_graph_inverse_centrality_is_permutation_equivariant() -> None:
    corr = np.eye(5)
    corr[0, 1:] = corr[1:, 0] = 0.45  # star: asset 0 is the hub (PSD: 1 - 4*0.45^2 > 0)
    assert np.linalg.eigvalsh(corr).min() > 0.0
    w = hier.graph_inverse_centrality_weights(corr)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] == w.min() and np.allclose(w[1:], w[1])
    rng = np.random.default_rng(5)
    a = rng.normal(size=(200, 6))
    cov = np.cov(a.T)
    base = hier.graph_inverse_centrality_weights(cov)
    perm = np.array([3, 0, 5, 1, 4, 2])
    permuted = hier.graph_inverse_centrality_weights(cov[np.ix_(perm, perm)])
    assert np.allclose(permuted, base[perm])
    assert np.allclose(hier.graph_inverse_centrality_weights(np.eye(2)), [0.5, 0.5])
    for floor in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="floor"):
            hier.graph_inverse_centrality_weights(cov, floor=floor)


def test_plugin_wrappers_registered_and_apply_upper_caps() -> None:
    cov = _two_pair_cov()
    rng = np.random.default_rng(0)
    returns = rng.normal(size=(300, 4)) @ np.linalg.cholesky(cov).T
    ids = ["a", "b", "c", "d"]
    for name in ("HRPDendrogram", "HERC", "NCO", "WassersteinDRO", "GraphInverseCentrality"):
        cls = GLOBAL_REGISTRY.get("portfolio", name)
        assert cls is not None, name
        weights = cls().allocate(ids, returns, upper=dict.fromkeys(ids, 0.3))
        assert set(weights) == set(ids)
        assert np.isclose(sum(weights.values()), 1.0)
        assert max(weights.values()) <= 0.3 + 1e-9
        assert cls().allocate([], returns) == {}
    # DRO wrapper: explicit radius/target are honoured; sample (n divisor) covariance.
    dro = hier.WassersteinDROPortfolio(radius=1e-4, target_return=-1.0)
    assert np.isclose(sum(dro.allocate(ids, returns).values()), 1.0)


def test_quality_gated_dispatch_accepts_new_methods_and_records_params() -> None:
    rng = np.random.default_rng(1)
    sleeves = {f"s{i}": rng.normal(0.001, 0.01, 300).tolist() for i in range(6)}
    legacy = allocate_quality_gated(sleeves, method="hrp", min_families=1)
    for method in (
        "hrp_dendrogram",
        "hrp_full",
        "herc",
        "nco",
        "wasserstein_dro",
        "graph_inverse_centrality",
        "graph",
    ):
        weights = allocate_quality_gated(sleeves, method=method, upper=0.4, min_families=1)
        assert weights and np.isclose(sum(weights.values()), 1.0)
        assert max(weights.values()) <= 0.4 + 1e-9
        assert set(weights) == set(legacy)
    constrained = allocate_quality_gated(
        sleeves,
        method="constrained_hrp",
        upper=0.4,
        allocator_params={"lower": 0.05, "upper_bound": 0.3},
        min_families=1,
    )
    assert min(constrained.values()) >= 0.05 - 1e-9
    assert max(constrained.values()) <= 0.3 + 1e-9
    with pytest.raises(ValueError, match="requires explicit"):
        allocate_quality_gated(sleeves, method="constrained_hrp", min_families=1)
    assert _build_allocator("herc", {"n_clusters": 2}).n_clusters == 2
    with pytest.raises(ValueError):
        _build_allocator("bogus")
    source = [
        {
            "id": "src",
            "path": "x",
            "sha256": "0" * 64,
            "max_age_hours": 1,
            "ready": True,
            "portfolio_ready": True,
        }
    ]
    spec = {
        sid: {"returns": series, "turnover": 0.1, "family": sid} for sid, series in sleeves.items()
    }
    manifest = build_allocation_manifest(
        spec,
        source_artifacts=source,
        method="nco",
        min_families=1,
        allocator_params={"n_clusters": 2},
    )
    assert manifest["allocation_method"] == "nco"
    assert manifest["allocator_params"] == {"n_clusters": 2}
    # Legacy call without allocator_params still records an (empty) provenance block.
    legacy_manifest = build_allocation_manifest(
        spec, source_artifacts=source, method="erc", min_families=1
    )
    assert "allocator_params" not in legacy_manifest  # default path byte-identical
