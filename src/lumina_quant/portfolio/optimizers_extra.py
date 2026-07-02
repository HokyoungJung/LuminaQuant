"""Pure-numpy portfolio allocation optimizers (additive, opt-in).

This module adds four long-only, fully-deterministic portfolio construction
methods on top of the shared primitives in :mod:`optimizer_core`:

* ``ERC``               — full-covariance Equal-Risk-Contribution (risk parity).
* ``MaxDiversification`` — maximize the diversification ratio via projected
  gradient ascent (NO scipy).
* ``MeanVariance``       — mean-variance utility maximization via projected
  gradient ascent (NO scipy).
* ``HRP``                — correlation-cluster Hierarchical-Risk-Parity that
  reuses :func:`optimizer_core.cluster_by_correlation` with a deterministic
  tie-break.

Every routine is deterministic: iterations start from equal weights, use a
fixed iteration budget + tolerance, and rely only on sorted / vectorized numpy
operations (no ``dict``/``set`` iteration-order dependence, no RNG needed).  The
covariance estimate reuses the Ledoit-Wolf shrinkage from ``optimizer_core`` so
it stays positive-definite even when names exceed observations.

These optimizers are OPT-IN.  They are registered on :data:`GLOBAL_REGISTRY`
under kind ``"portfolio"`` but are NOT imported by ``portfolio/__init__`` and
are NOT wired into any default execution path, so importing this module has no
effect on the golden backtest / walk-forward numerics unless a caller
explicitly selects ``portfolio.allocation_method``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lumina_quant.core.plugin_registry import register
from lumina_quant.portfolio.optimizer_core import (
    cluster_by_correlation,
    ledoit_wolf_shrunk_covariance,
    project_simplex_with_upper_bounds,
)

__all__ = [
    "ERCPortfolio",
    "HRPPortfolio",
    "MaxDiversificationPortfolio",
    "MeanVariancePortfolio",
    "erc_weights",
    "hrp_weights_from_returns",
    "max_diversification_weights",
    "mean_variance_weights",
]


# ---------------------------------------------------------------------------
# Deterministic numeric helpers (pure numpy)
# ---------------------------------------------------------------------------


def _clean_matrix(returns_matrix: Any) -> np.ndarray:
    """Return a finite ``(T, N)`` float matrix (NaN/inf -> 0.0)."""
    matrix = np.asarray(returns_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    if not np.all(np.isfinite(matrix)):
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def _clean_cov(cov: Any) -> np.ndarray:
    """Coerce ``cov`` to a finite symmetric ``(N, N)`` float matrix."""
    matrix = np.asarray(cov, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return np.zeros((0, 0), dtype=float)
    if not np.all(np.isfinite(matrix)):
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # Symmetrize defensively; downstream algorithms assume Sigma == Sigma.T.
    return 0.5 * (matrix + matrix.T)


def project_to_simplex(vector: Any, *, total: float = 1.0) -> np.ndarray:
    """Euclidean projection of ``vector`` onto ``{w >= 0, sum(w) == total}``.

    Deterministic sort-based projection (Duchi et al., 2008).  Falls back to
    equal weights when the projection collapses to zero mass.
    """
    values = np.asarray(vector, dtype=float).reshape(-1)
    n = int(values.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    target = float(total)
    if target <= 0.0:
        return np.zeros(n, dtype=float)
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    sorted_desc = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_desc) - target
    index = np.arange(1, n + 1, dtype=float)
    condition = sorted_desc - cumulative / index > 0.0
    positive = np.nonzero(condition)[0]
    if positive.size == 0:
        return np.full(n, target / float(n), dtype=float)
    rho = int(positive[-1])
    theta = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - theta, 0.0)
    mass = float(projected.sum())
    if mass <= 0.0:
        return np.full(n, target / float(n), dtype=float)
    return projected * (target / mass)


def _equal_weights(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)
    return np.full(n, 1.0 / float(n), dtype=float)


def _diag_variance(cov: np.ndarray) -> np.ndarray:
    var = np.clip(np.diag(cov).astype(float), 0.0, None)
    return var


def _inverse_variance_weights(var: np.ndarray) -> np.ndarray:
    """Deterministic inverse-variance weights with an equal-weight fallback."""
    n = int(var.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    positive = var > 0.0
    if not np.any(positive):
        return _equal_weights(n)
    inv = np.zeros(n, dtype=float)
    inv[positive] = 1.0 / var[positive]
    total = float(inv.sum())
    if total <= 0.0:
        return _equal_weights(n)
    return inv / total


# ---------------------------------------------------------------------------
# Core optimizers (operate on covariance / returns, return numpy weight vectors)
# ---------------------------------------------------------------------------


def erc_weights(
    cov: Any,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Equal-Risk-Contribution (risk-parity) long-only weights.

    Uses the cyclical coordinate-descent solver of Griveau-Billion et al. (2013)
    for the convex risk-parity program ``min 0.5 w'Sigma w - sum b_i ln(w_i)``
    with equal risk budgets ``b_i = 1/N``.  Each coordinate update solves the 1D
    quadratic ``a_i w_i^2 + c_i w_i - b_i = 0`` in closed form (always yielding a
    positive root), which is contractive and does not oscillate.  The diagonal
    term ``a_i = Sigma_ii`` is floored by a tiny fraction of ``trace(Sigma)`` so a
    degenerate / near-singular covariance stays finite (the well-conditioned case
    is untouched — its diagonal already exceeds the floor).

    For two uncorrelated equal-variance assets this returns exactly ``[0.5, 0.5]``.
    """
    matrix = _clean_cov(cov)
    n = int(matrix.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    budget = 1.0 / float(n)
    weights = _equal_weights(n)
    trace = float(np.trace(matrix))
    floor = max(1e-18, 1e-12 * (trace / float(n) if trace > 0.0 else 1.0))
    diag = np.clip(np.diag(matrix).astype(float), floor, None)
    iterations = max(1, int(max_iter))
    tolerance = max(0.0, float(tol))

    for _ in range(iterations):
        previous = weights / float(weights.sum())
        for i in range(n):
            a = float(diag[i])
            # c_i = off-diagonal marginal risk contributed by the other names.
            c = float(matrix[i] @ weights) - a * float(weights[i])
            root = (-c + np.sqrt(c * c + 4.0 * a * budget)) / (2.0 * a)
            weights[i] = root if root > 0.0 else floor
        # Convergence is measured on normalized COPIES only; the raw coordinate-
        # descent iterate (weights) MUST stay un-normalized across sweeps so it
        # remains on the scale-fixed Griveau-Billion fixed point (Sigma y)_i =
        # b_i / y_i. Renormalizing the working vector mid-solve moves it off that
        # scale and converges to a wrong point with UNEQUAL risk contributions on
        # any correlated covariance. Final normalization happens once, below.
        current = weights / float(weights.sum())
        if float(np.abs(current - previous).max()) <= tolerance:
            break

    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return _equal_weights(n)
    return weights / total


def max_diversification_weights(
    cov: Any,
    *,
    max_iter: int = 500,
    step: float = 0.05,
    tol: float = 1e-12,
) -> np.ndarray:
    """Maximize the diversification ratio ``(w . sigma) / sqrt(w' Sigma w)``.

    Projected gradient ascent on the long-only unit simplex.  No scipy; the
    gradient is normalized so a single fixed ``step`` is scale-robust, and every
    iterate is Euclidean-projected back onto the simplex.  Deterministic: starts
    from equal weights and stops on a fixed iteration budget / step-size floor.
    """
    matrix = _clean_cov(cov)
    n = int(matrix.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    sigma = np.sqrt(_diag_variance(matrix))
    weights = _equal_weights(n)
    iterations = max(1, int(max_iter))
    base_step = max(0.0, float(step))
    tolerance = max(0.0, float(tol))

    for _ in range(iterations):
        quad = float(weights @ matrix @ weights)
        den = np.sqrt(quad) if quad > 0.0 else 0.0
        if den <= 0.0:
            break
        numerator = float(weights @ sigma)
        # d/dw [ (w.sigma) / sqrt(w'Sw) ] = sigma/den - (w.sigma) * (Sw) / den^3
        gradient = sigma / den - numerator * (matrix @ weights) / (den**3)
        norm = float(np.linalg.norm(gradient))
        if norm <= tolerance:
            break
        candidate = project_to_simplex(weights + base_step * gradient / norm, total=1.0)
        move = float(np.linalg.norm(candidate - weights))
        weights = candidate
        if move <= tolerance:
            break

    total = float(weights.sum())
    if total <= 0.0:
        return _equal_weights(n)
    return weights / total


def mean_variance_weights(
    mean: Any,
    cov: Any,
    *,
    risk_aversion: float = 5.0,
    max_iter: int = 500,
    step: float = 0.05,
    tol: float = 1e-12,
) -> np.ndarray:
    """Maximize the mean-variance utility ``w'mu - 0.5 * ra * w' Sigma w``.

    Projected gradient ascent on the long-only unit simplex (no scipy).  With
    equal means and an equal-variance diagonal covariance this returns equal
    weights.  Deterministic: equal-weight start, fixed iteration budget.
    """
    matrix = _clean_cov(cov)
    n = int(matrix.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    mu = np.asarray(mean, dtype=float).reshape(-1)
    if mu.size != n:
        mu = np.zeros(n, dtype=float)
    if not np.all(np.isfinite(mu)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    ra = max(0.0, float(risk_aversion))
    weights = _equal_weights(n)
    iterations = max(1, int(max_iter))
    base_step = max(0.0, float(step))
    tolerance = max(0.0, float(tol))

    for _ in range(iterations):
        gradient = mu - ra * (matrix @ weights)
        norm = float(np.linalg.norm(gradient))
        if norm <= tolerance:
            break
        candidate = project_to_simplex(weights + base_step * gradient / norm, total=1.0)
        move = float(np.linalg.norm(candidate - weights))
        weights = candidate
        if move <= tolerance:
            break

    total = float(weights.sum())
    if total <= 0.0:
        return _equal_weights(n)
    return weights / total


def _returns_to_stream_map(ids: list[str], matrix: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    """Build a deterministic ``id -> stream`` map from returns columns."""
    stream_map: dict[str, list[dict[str, Any]]] = {}
    rows = int(matrix.shape[0])
    for col, cid in enumerate(ids):
        stream_map[cid] = [{"t": float(row), "v": float(matrix[row, col])} for row in range(rows)]
    return stream_map


def hrp_weights_from_returns(
    ids: list[str],
    returns_matrix: Any,
    *,
    corr_threshold: float = 0.60,
    cov_window: int | None = None,
) -> np.ndarray:
    """Correlation-cluster Hierarchical-Risk-Parity weights (deterministic).

    Clusters are formed by reusing :func:`optimizer_core.cluster_by_correlation`
    (which preserves the supplied ``ids`` order, giving a stable "linkage" order).
    Within each cluster, capital is split by inverse variance; across clusters it
    is split by inverse cluster (sub-portfolio) variance.  Both splits fall back
    to equal weights on ties / non-positive variance, so equal-variance
    uncorrelated assets reproduce equal weights exactly.
    """
    matrix = _clean_matrix(returns_matrix)
    n = len(ids)
    if n == 0 or matrix.shape[1] != n:
        return _equal_weights(n)
    if n == 1:
        return np.ones(1, dtype=float)

    cov, _intensity = ledoit_wolf_shrunk_covariance(matrix, window=cov_window)
    if cov.shape != (n, n):
        return _equal_weights(n)
    var = _diag_variance(cov)

    stream_map = _returns_to_stream_map(list(ids), matrix)
    clusters = cluster_by_correlation(list(ids), stream_map, threshold=float(corr_threshold))
    index_of = {cid: pos for pos, cid in enumerate(ids)}

    weights = np.zeros(n, dtype=float)
    cluster_variances: list[float] = []
    cluster_local: list[tuple[list[int], np.ndarray]] = []
    for cluster in clusters:
        members = [index_of[cid] for cid in cluster]
        sub_var = var[members]
        local = _inverse_variance_weights(sub_var)
        sub_cov = cov[np.ix_(members, members)]
        cluster_var = float(local @ sub_cov @ local)
        cluster_variances.append(max(0.0, cluster_var))
        cluster_local.append((members, local))

    across = _inverse_variance_weights(np.asarray(cluster_variances, dtype=float))
    for (members, local), alloc in zip(cluster_local, across):
        for member, share in zip(members, local):
            weights[member] = float(alloc) * float(share)

    total = float(weights.sum())
    if total <= 0.0:
        return _equal_weights(n)
    return weights / total


# ---------------------------------------------------------------------------
# Registered plugin wrappers
# ---------------------------------------------------------------------------


def _apply_upper_bounds(
    ids: list[str], weights: np.ndarray, upper: dict[str, float] | None
) -> dict[str, float]:
    """Return ``id -> weight`` dict, optionally re-projected under per-name caps.

    When ``upper`` is provided the raw weights are routed through
    :func:`optimizer_core.project_simplex_with_upper_bounds` (reused) so the same
    cap semantics as the production allocator apply.
    """
    raw = {cid: float(weights[pos]) for pos, cid in enumerate(ids)}
    if not upper:
        return raw
    bounds = {cid: float(upper.get(cid, 1.0)) for cid in ids}
    return project_simplex_with_upper_bounds(raw, upper=bounds, target_sum=1.0)


@register("portfolio", "ERC")
class ERCPortfolio:
    """Equal-Risk-Contribution allocator (see :func:`erc_weights`)."""

    def __init__(
        self,
        *,
        max_iter: int = 10_000,
        tol: float = 1e-10,
        cov_window: int | None = None,
    ) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.cov_window = cov_window

    def allocate(
        self,
        ids: list[str],
        returns_matrix: Any,
        *,
        upper: dict[str, float] | None = None,
    ) -> dict[str, float]:
        ids = list(ids)
        matrix = _clean_matrix(returns_matrix)
        if not ids or matrix.shape[1] != len(ids):
            return {}
        cov, _ = ledoit_wolf_shrunk_covariance(matrix, window=self.cov_window)
        weights = erc_weights(cov, max_iter=self.max_iter, tol=self.tol)
        return _apply_upper_bounds(ids, weights, upper)


@register("portfolio", "MaxDiversification")
class MaxDiversificationPortfolio:
    """Maximum-diversification allocator (see :func:`max_diversification_weights`)."""

    def __init__(
        self,
        *,
        max_iter: int = 500,
        step: float = 0.05,
        tol: float = 1e-12,
        cov_window: int | None = None,
    ) -> None:
        self.max_iter = int(max_iter)
        self.step = float(step)
        self.tol = float(tol)
        self.cov_window = cov_window

    def allocate(
        self,
        ids: list[str],
        returns_matrix: Any,
        *,
        upper: dict[str, float] | None = None,
    ) -> dict[str, float]:
        ids = list(ids)
        matrix = _clean_matrix(returns_matrix)
        if not ids or matrix.shape[1] != len(ids):
            return {}
        cov, _ = ledoit_wolf_shrunk_covariance(matrix, window=self.cov_window)
        weights = max_diversification_weights(
            cov, max_iter=self.max_iter, step=self.step, tol=self.tol
        )
        return _apply_upper_bounds(ids, weights, upper)


@register("portfolio", "MeanVariance")
class MeanVariancePortfolio:
    """Mean-variance allocator (see :func:`mean_variance_weights`)."""

    def __init__(
        self,
        *,
        risk_aversion: float = 5.0,
        max_iter: int = 500,
        step: float = 0.05,
        tol: float = 1e-12,
        cov_window: int | None = None,
    ) -> None:
        self.risk_aversion = float(risk_aversion)
        self.max_iter = int(max_iter)
        self.step = float(step)
        self.tol = float(tol)
        self.cov_window = cov_window

    def allocate(
        self,
        ids: list[str],
        returns_matrix: Any,
        *,
        upper: dict[str, float] | None = None,
    ) -> dict[str, float]:
        ids = list(ids)
        matrix = _clean_matrix(returns_matrix)
        if not ids or matrix.shape[1] != len(ids):
            return {}
        cov, _ = ledoit_wolf_shrunk_covariance(matrix, window=self.cov_window)
        mean = matrix.mean(axis=0) if matrix.shape[0] > 0 else np.zeros(len(ids))
        weights = mean_variance_weights(
            mean,
            cov,
            risk_aversion=self.risk_aversion,
            max_iter=self.max_iter,
            step=self.step,
            tol=self.tol,
        )
        return _apply_upper_bounds(ids, weights, upper)


@register("portfolio", "HRP")
class HRPPortfolio:
    """Correlation-cluster HRP allocator (see :func:`hrp_weights_from_returns`)."""

    def __init__(
        self,
        *,
        corr_threshold: float = 0.60,
        cov_window: int | None = None,
    ) -> None:
        self.corr_threshold = float(corr_threshold)
        self.cov_window = cov_window

    def allocate(
        self,
        ids: list[str],
        returns_matrix: Any,
        *,
        upper: dict[str, float] | None = None,
    ) -> dict[str, float]:
        ids = list(ids)
        matrix = _clean_matrix(returns_matrix)
        if not ids or matrix.shape[1] != len(ids):
            return {}
        weights = hrp_weights_from_returns(
            ids,
            matrix,
            corr_threshold=self.corr_threshold,
            cov_window=self.cov_window,
        )
        return _apply_upper_bounds(ids, weights, upper)
