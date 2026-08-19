"""Hierarchical / robust portfolio allocators (pure numpy, deterministic, opt-in).

The existing ``method="hrp"`` in :mod:`quality_gated_allocation` is a greedy
correlation-THRESHOLD clustering (see ``optimizers_extra.hrp_weights_from_returns``).
This module adds the literature-faithful hierarchical family plus two robust
alternatives, all long-only and scipy-free:

* :func:`hrp_dendrogram_weights` -- Lopez de Prado (2016) Hierarchical Risk
  Parity: correlation distance ``sqrt(0.5*(1-rho))`` -> agglomerative linkage
  (single / complete / average / ward, Lance-Williams) -> quasi-diagonal
  seriation -> recursive bisection with inverse-variance cluster risk.  Optional
  per-asset ``bounds`` give *constrained HRP*: infeasible boxes FAIL CLOSED
  (``ValueError``), each bisection split is clipped to the interval that keeps
  both children's boxes satisfiable, and the result is finally projected onto
  ``{lo <= w <= hi, sum w = 1}`` exactly (box-simplex Euclidean projection).
* :func:`herc_weights` -- Raffinot (2018) Hierarchical Equal Risk Contribution:
  same dendrogram, cluster count by explicit ``n_clusters`` or by the mean
  silhouette (singletons scored 0 -- NOTE: Raffinot uses the Gap statistic,
  which needs reference-distribution sampling; the silhouette is a deterministic
  stand-in and is documented as such), top-down equal-risk-contribution split
  between sub-trees, naive risk parity (inverse-vol) inside the final clusters.
* :func:`nco_weights` -- Lopez de Prado (2019) Nested Clustered Optimization,
  long-only variant: cluster, solve the long-only min-variance QP (or long-only
  max-Sharpe via the ``mu'y = 1`` QP substitution) INSIDE each cluster with a
  convergent projected-gradient QP solver (:func:`long_only_min_variance`,
  :func:`long_only_max_sharpe`; KKT-checkable, see :func:`min_variance_kkt_residual`),
  then across the reduced cluster covariance, and compose.
* :func:`wasserstein_dro_weights` -- Blanchet, Chen & Zhou (2022) exact
  type-2 Wasserstein DRO mean-variance (``p = q = 2``): with ``eps = sqrt(radius)``
  minimise ``sqrt(w'Sw) + eps*||w||_2`` over the long-only simplex subject to
  ``mu'w - eps*||w||_2 >= target_return`` (constraint only when ``target_return``
  is given).  Solved by projected gradient with Armijo backtracking on the convex
  objective and a dual bisection for the target constraint; an infeasible target
  raises ``ValueError``.  ``radius = 0`` without a binding target recovers the
  long-only min-variance solution.  ``radius`` is in SQUARED-return units (a
  daily-return panel needs ~1e-5..1e-4, not 0.01).
* :func:`graph_inverse_centrality_weights` -- *full-graph inverse-centrality
  graph-risk heuristic*: adjacency ``A_ij = max(|rho_ij|, floor)`` (``A_ii = 0``),
  Perron eigenvector centrality ``z`` (normalised), exact solution of
  ``min sum_i z_i w_i^2`` on the simplex -> ``w ~ 1/z``.  Permutation-equivariant
  by construction.  This is NOT a reproduction of the MST/PMFG peripherality
  portfolios of Pozzi-Di Matteo-Aste (2013); it only borrows the "de-weight the
  central names" idea and honours the graph-portfolio slot of the request.

Every routine takes a covariance matrix (callers use the repo's Ledoit-Wolf
estimate unless stated) and returns a numpy weight vector summing to one;
degenerate input falls back to equal weights instead of raising, EXCEPT where a
caller declaration is wrong or infeasible -- an unknown ``linkage_method`` /
``allocator_params`` key, an infeasible constrained-HRP box, an unreachable DRO
target or a materially non-PSD covariance -- which fails closed with
``ValueError``/``TypeError`` so a typo in a research JSON cannot silently
degrade to a different allocator.  Registered plugin wrappers
(``HRPDendrogram``, ``HERC``, ``NCO``, ``WassersteinDRO``, ``GraphInverseCentrality``)
share the ``allocate(ids, returns_matrix, *, upper=None)`` contract of
:mod:`optimizers_extra` and are wired into ``quality_gated_allocation`` behind
new ``method`` tokens; nothing here changes any default path.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from lumina_quant.core.plugin_registry import register
from lumina_quant.portfolio.optimizer_core import ledoit_wolf_shrunk_covariance
from lumina_quant.portfolio.optimizers_extra import (
    _apply_upper_bounds,
    _clean_cov,
    _clean_matrix,
    _equal_weights,
    _inverse_variance_weights,
    project_to_simplex,
)

__all__ = [
    "GraphInverseCentralityPortfolio",
    "HERCPortfolio",
    "HRPDendrogramPortfolio",
    "NCOPortfolio",
    "WassersteinDROPortfolio",
    "correlation_distance",
    "cut_tree",
    "graph_inverse_centrality_weights",
    "herc_weights",
    "hierarchical_linkage",
    "hrp_dendrogram_weights",
    "long_only_max_sharpe",
    "long_only_min_variance",
    "min_variance_kkt_residual",
    "nco_weights",
    "project_box_simplex",
    "quasi_diagonal_order",
    "silhouette_optimal_clusters",
    "wasserstein_dro_objective",
    "wasserstein_dro_weights",
]

_LINKAGES = ("single", "complete", "average", "ward")
_TOL = 1e-12
_PSD_REL_TOL = 1e-8


def _ensure_psd(cov: Any) -> np.ndarray:
    """Symmetrise ``cov`` and fail closed when it is materially non-PSD.

    Numerical noise (``min eigenvalue > -1e-8 * scale``) is clipped to zero via
    the eigen-decomposition; a genuinely indefinite matrix (which would make the
    convex programmes below meaningless) raises ``ValueError``.
    """
    matrix = _clean_cov(cov)
    n = int(matrix.shape[0])
    if n <= 1:
        return matrix
    try:
        evals, evecs = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        return matrix
    scale = max(1.0, float(np.abs(evals).max()))
    lowest = float(evals.min())
    if lowest < -_PSD_REL_TOL * scale:
        raise ValueError(
            f"covariance is not positive semi-definite (min eigenvalue {lowest:.3e}); "
            "refusing to allocate on an indefinite matrix"
        )
    if lowest < 0.0:
        matrix = (evecs * np.clip(evals, 0.0, None)) @ evecs.T
        matrix = 0.5 * (matrix + matrix.T)
    return matrix


# ---------------------------------------------------------------------------
# Distance / linkage / seriation primitives
# ---------------------------------------------------------------------------


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0.0, cov / denom, 0.0)
    corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def correlation_distance(corr: Any) -> np.ndarray:
    """Lopez de Prado distance ``sqrt(0.5 * (1 - rho))`` (a proper metric)."""
    matrix = np.clip(np.asarray(corr, dtype=float), -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - matrix), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return dist


def hierarchical_linkage(dist: Any, *, method: str = "single") -> np.ndarray:
    """Agglomerative linkage matrix ``(n-1, 4)`` = ``[left, right, distance, size]``.

    Same row semantics as ``scipy.cluster.hierarchy.linkage`` (new cluster ``i``
    gets id ``n + i``) but computed with the Lance-Williams update in plain
    numpy.  Ties resolve to the lowest ``(i, j)`` index pair, so the output is
    deterministic.  ``method`` in ``single|complete|average|ward``.
    """
    token = str(method or "single").strip().lower()
    if token not in _LINKAGES:
        raise ValueError(f"unsupported linkage {method!r} (expected one of {_LINKAGES})")
    d = np.array(dist, dtype=float, copy=True)
    n = int(d.shape[0])
    if n < 2:
        return np.zeros((0, 4), dtype=float)
    if token == "ward":
        # Ward operates on squared Euclidean distances; Lance-Williams below is
        # written for squared distances, so square here and take sqrt on output.
        d = d * d
    active = list(range(n))
    ids = {i: i for i in range(n)}  # position -> cluster id
    sizes = dict.fromkeys(range(n), 1.0)
    out = np.zeros((n - 1, 4), dtype=float)
    # ponytail: O(n^3) naive agglomeration; fine for a few hundred sleeves/assets,
    # swap for a nearest-neighbour chain if universes get large.
    for step in range(n - 1):
        m = len(active)
        best = (math.inf, -1, -1)
        for a in range(m):
            row = d[active[a], active[a + 1 :]]
            if row.size == 0:
                continue
            b_rel = int(np.argmin(row))
            value = float(row[b_rel])
            if value < best[0]:
                best = (value, a, a + 1 + b_rel)
        value, a, b = best
        pa, pb = active[a], active[b]
        ida, idb = ids[pa], ids[pb]
        na, nb = sizes[pa], sizes[pb]
        new_id = n + step
        out[step] = (
            float(min(ida, idb)),
            float(max(ida, idb)),
            math.sqrt(max(0.0, value)) if token == "ward" else value,
            na + nb,
        )
        # Lance-Williams update of distances from the merged cluster to others.
        for c in active:
            if c in (pa, pb):
                continue
            dac, dbc, dab = d[pa, c], d[pb, c], d[pa, pb]
            nc = sizes[c]
            if token == "single":
                new = min(dac, dbc)
            elif token == "complete":
                new = max(dac, dbc)
            elif token == "average":
                new = (na * dac + nb * dbc) / (na + nb)
            else:  # ward (squared distances)
                tot = na + nb + nc
                new = ((na + nc) * dac + (nb + nc) * dbc - nc * dab) / tot
            d[pa, c] = d[c, pa] = new
        sizes[pa] = na + nb
        ids[pa] = new_id
        del active[b]
    return out


def _children(linkage: np.ndarray, n: int) -> dict[int, tuple[int, int]]:
    return {
        n + row: (int(linkage[row, 0]), int(linkage[row, 1])) for row in range(linkage.shape[0])
    }


def _leaves(node: int, children: Mapping[int, tuple[int, int]]) -> list[int]:
    stack, out = [node], []
    while stack:
        cur = stack.pop()
        if cur in children:
            left, right = children[cur]
            stack.append(right)
            stack.append(left)
        else:
            out.append(cur)
    return out


def quasi_diagonal_order(linkage: Any, n: int) -> list[int]:
    """Leaf order that places correlated assets adjacent (getQuasiDiag)."""
    link = np.asarray(linkage, dtype=float)
    if n <= 1:
        return list(range(max(0, n)))
    if link.shape[0] != n - 1:
        return list(range(n))
    root = n + int(link.shape[0]) - 1
    return _leaves(root, _children(link, n))


def cut_tree(linkage: Any, n: int, *, n_clusters: int) -> list[int]:
    """Flat cluster labels (0..k-1, ordered by first appearance) for ``n_clusters``."""
    link = np.asarray(linkage, dtype=float)
    k = max(1, min(int(n_clusters), n))
    if n <= 0:
        return []
    if link.shape[0] != n - 1 or k >= n:
        return list(range(n)) if k >= n else [0] * n
    children = _children(link, n)
    # Undo the last (k-1) merges: the roots of the remaining forest are the clusters.
    roots = [n + int(link.shape[0]) - 1]
    for row in range(int(link.shape[0]) - 1, int(link.shape[0]) - k, -1):
        node = n + row
        if node in roots:
            roots.remove(node)
            roots.extend(children[node])
    labels = [0] * n
    ordered_roots = sorted(roots, key=lambda r: min(_leaves(r, children)))
    for label, root in enumerate(ordered_roots):
        for leaf in _leaves(root, children):
            labels[leaf] = label
    return labels


def _mean_silhouette(d: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette; a singleton cluster member scores 0 (Rousseeuw convention)."""
    n = int(d.shape[0])
    uniq = np.unique(labels)
    if uniq.size < 2:
        return -1.0
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        own = labels == labels[i]
        own[i] = False
        if not np.any(own):
            scores[i] = 0.0
            continue
        a = float(d[i, own].mean())
        b = math.inf
        for lab in uniq:
            if lab == labels[i]:
                continue
            other = labels == lab
            if np.any(other):
                b = min(b, float(d[i, other].mean()))
        if not np.isfinite(b):
            scores[i] = 0.0
            continue
        denom = max(a, b)
        scores[i] = 0.0 if denom <= 0.0 else (b - a) / denom
    return float(scores.mean())


def silhouette_optimal_clusters(
    dist: Any, linkage: Any, *, max_clusters: int = 10, min_clusters: int = 2
) -> int:
    """Number of clusters (cutting ``linkage``) maximising the mean silhouette.

    Deterministic stand-in for the Gap statistic used by Raffinot (2018); ties
    resolve to the smaller ``k``.
    """
    d = np.asarray(dist, dtype=float)
    n = int(d.shape[0])
    if n < 3:
        return max(1, min(n, int(min_clusters)))
    lo = max(2, int(min_clusters))
    hi = max(lo, min(int(max_clusters), n - 1))
    best_k, best_score = lo, -math.inf
    for k in range(lo, hi + 1):
        labels = np.asarray(cut_tree(linkage, n, n_clusters=k))
        score = _mean_silhouette(d, labels)
        if score > best_score + 1e-12:
            best_k, best_score = k, score
    return best_k


# ---------------------------------------------------------------------------
# Convex solvers (projected gradient with Armijo backtracking, no scipy)
# ---------------------------------------------------------------------------


def project_box_simplex(vector: Any, lower: Any, upper: Any) -> np.ndarray:
    """Euclidean projection onto ``{lo <= w <= hi, sum(w) == 1}``.

    ``w_i = clip(v_i - tau, lo_i, hi_i)`` with ``tau`` found by bisection (the
    sum is monotone in ``tau``).  Raises ``ValueError`` when the box cannot hold
    a probability vector (``sum(lo) > 1`` or ``sum(hi) < 1`` or ``lo > hi``).
    """
    v = np.asarray(vector, dtype=float).reshape(-1)
    n = int(v.size)
    lo = np.broadcast_to(np.asarray(lower, dtype=float), (n,)).astype(float)
    hi = np.broadcast_to(np.asarray(upper, dtype=float), (n,)).astype(float)
    if n == 0:
        return np.zeros(0, dtype=float)
    if np.any(hi < lo - _TOL) or float(lo.sum()) > 1.0 + 1e-9 or float(hi.sum()) < 1.0 - 1e-9:
        raise ValueError(
            "infeasible box for a probability vector: "
            f"sum(lower)={float(lo.sum()):.6f}, sum(upper)={float(hi.sum()):.6f}"
        )
    tau_lo = float((v - hi).min()) - 1.0
    tau_hi = float((v - lo).max()) + 1.0
    for _ in range(200):
        tau = 0.5 * (tau_lo + tau_hi)
        total = float(np.clip(v - tau, lo, hi).sum())
        if total > 1.0:
            tau_lo = tau
        else:
            tau_hi = tau
        if tau_hi - tau_lo < 1e-15:
            break
    w = np.clip(v - 0.5 * (tau_lo + tau_hi), lo, hi)
    # Final exact renormalisation inside the box (numerical residue only).
    residual = 1.0 - float(w.sum())
    if abs(residual) > 1e-12:
        slack = (hi - w) if residual > 0 else (w - lo)
        total_slack = float(slack.sum())
        if total_slack > 0.0:
            w = w + residual * slack / total_slack
    return w


def _projected_gradient(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    project: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    *,
    step0: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Projected gradient with Armijo backtracking (monotone descent, convex f)."""
    w = project(np.asarray(start, dtype=float))
    fw = objective(w)
    step = max(1e-12, float(step0))
    for _ in range(max(1, int(max_iter))):
        g = gradient(w)
        # Backtrack until sufficient decrease along the projected direction.
        while True:
            cand = project(w - step * g)
            direction = cand - w
            fc = objective(cand)
            if fc <= fw + 1e-4 * float(g @ direction) + 1e-18 or step < 1e-16:
                break
            step *= 0.5
        move = float(np.linalg.norm(cand - w))
        if fc <= fw:
            w, fw = cand, fc
        if move <= tol:
            break
        step = min(step * 2.0, float(step0))
    return w


def _lipschitz_step(cov: np.ndarray) -> float:
    try:
        top = float(np.linalg.eigvalsh(cov).max())
    except np.linalg.LinAlgError:
        top = float(np.trace(cov))
    return 1.0 / max(top, 1e-12)


def _active_set_qp(matrix: np.ndarray, c: np.ndarray, *, max_iter: int) -> np.ndarray | None:
    """Solve ``min 1/2 w'Sw  s.t. c'w = 1, w >= 0`` by a primal active-set method.

    Stationarity on the support ``S`` gives ``w_S = nu * S_SS^{-1} c_S`` with
    ``nu = 1 / (c_S' S_SS^{-1} c_S)``; negative entries are dropped, and an excluded
    asset ``j`` is added while its reduced gradient ``(S w)_j - nu*c_j`` is
    negative.  Finite for a strictly convex objective (Lawson-Hanson style);
    returns ``None`` if it fails to settle within ``max_iter`` passes.
    """
    n = int(matrix.shape[0])
    positive = c > 0.0
    if not np.any(positive):
        return None
    support = np.flatnonzero(positive).tolist()
    ridge = np.eye(n) * (1e-14 * max(1.0, float(np.trace(matrix))))
    for _ in range(max(1, int(max_iter))):
        idx = np.asarray(support, dtype=int)
        sub = matrix[np.ix_(idx, idx)] + ridge[np.ix_(idx, idx)]
        try:
            direction = np.linalg.solve(sub, c[idx])
        except np.linalg.LinAlgError:
            direction = np.linalg.pinv(sub) @ c[idx]
        denom = float(c[idx] @ direction)
        if not np.isfinite(denom) or denom <= 0.0:
            return None
        nu = 1.0 / denom
        w_support = nu * direction
        if np.any(w_support < -1e-15):
            worst = int(np.argmin(w_support))
            if len(support) == 1:
                return None
            support.pop(worst)
            continue
        w = np.zeros(n, dtype=float)
        w[idx] = np.maximum(w_support, 0.0)
        reduced = matrix @ w - nu * c
        excluded = [j for j in range(n) if j not in support]
        if not excluded:
            return w
        worst_j = min(excluded, key=lambda j: float(reduced[j]))
        if float(reduced[worst_j]) >= -1e-14 * max(1.0, abs(nu)):
            return w
        support.append(worst_j)
        support.sort()
    return None


def long_only_min_variance(cov: Any, *, max_iter: int = 20_000, tol: float = 1e-13) -> np.ndarray:
    """Long-only minimum-variance weights ``min w'Sw, w in simplex`` (QP, exact).

    Solved by a finite primal active-set method; a projected-gradient run with
    Armijo backtracking is the guarded fallback (the better objective wins).
    Verify with :func:`min_variance_kkt_residual`.
    """
    matrix = _ensure_psd(cov)
    n = int(matrix.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    if float(np.trace(matrix)) <= 0.0:
        return _equal_weights(n)
    exact = _active_set_qp(matrix, np.ones(n), max_iter=4 * n + 8)
    pgd = _projected_gradient(
        lambda w: 0.5 * float(w @ matrix @ w),
        lambda w: matrix @ w,
        lambda v: project_to_simplex(v, total=1.0),
        _equal_weights(n),
        step0=_lipschitz_step(matrix),
        max_iter=max_iter,
        tol=tol,
    )
    if exact is None or not np.all(np.isfinite(exact)):
        return pgd
    return exact if float(exact @ matrix @ exact) <= float(pgd @ matrix @ pgd) + 1e-15 else pgd


def min_variance_kkt_residual(cov: Any, weights: Any) -> float:
    """KKT residual of a long-only min-variance candidate (0 at the optimum).

    With ``g = S w``: every held asset must have a common multiplier and every
    zero-weight asset must have no smaller gradient.  Returns the largest
    stationarity, dual-feasibility, or simplex residual on a stable absolute scale.
    """
    matrix = _clean_cov(cov)
    w = np.asarray(weights, dtype=float)
    g = matrix @ w
    support = w > 1e-10
    if not np.any(support):
        return math.inf
    lam = float(np.mean(g[support]))
    held = float(np.max(np.abs(g[support] - lam)))
    excluded = float(np.max(np.maximum(lam - g[~support], 0.0))) if np.any(~support) else 0.0
    feasibility = max(abs(float(w.sum()) - 1.0), max(0.0, -float(w.min())))
    return max(held, excluded, feasibility) / max(1.0, float(np.max(np.abs(g))))


def _project_positive_hyperplane(vector: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Projection onto ``{y >= 0, mu'y == 1}`` via ``y = max(0, v + tau*mu)`` (bisection)."""

    def _total(tau: float) -> float:
        return float(mu @ np.maximum(0.0, vector + tau * mu))

    lo, hi = -1.0, 1.0
    for _ in range(200):
        if _total(lo) <= 1.0:
            break
        lo *= 2.0
    for _ in range(200):
        if _total(hi) >= 1.0:
            break
        hi *= 2.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if _total(mid) < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return np.maximum(0.0, vector + 0.5 * (lo + hi) * mu)


def long_only_max_sharpe(
    cov: Any, mu: Any, *, max_iter: int = 20_000, tol: float = 1e-13
) -> np.ndarray:
    """Long-only maximum-Sharpe weights via ``min y'Sy s.t. mu'y = 1, y >= 0``, ``w = y/sum(y)``.

    Requires at least one positive expected return.  The all-non-positive case
    is not represented by the positive-return QP substitution and therefore
    fails closed instead of silently solving a different objective.
    """
    matrix = _ensure_psd(cov)
    n = int(matrix.shape[0])
    mean = np.asarray(mu, dtype=float).reshape(-1)
    if n == 0:
        return np.zeros(0, dtype=float)
    if mean.size != n or not np.all(np.isfinite(mean)):
        raise ValueError("expected returns must be finite and match covariance dimensions")
    if n == 1:
        return np.ones(1, dtype=float)
    if float(mean.max()) <= 0.0:
        raise ValueError("long-only max-Sharpe requires at least one positive expected return")
    start = np.where(mean > 0.0, 1.0, 0.0)
    start = start / max(float(mean @ start), 1e-18)
    y = _active_set_qp(matrix, mean, max_iter=4 * n + 8)
    if y is None or not np.all(np.isfinite(y)):
        y = _projected_gradient(
            lambda v: 0.5 * float(v @ matrix @ v),
            lambda v: matrix @ v,
            lambda v: _project_positive_hyperplane(v, mean),
            start,
            step0=_lipschitz_step(matrix),
            max_iter=max_iter,
            tol=tol,
        )
    total = float(y.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("max-Sharpe solver returned invalid transformed weights")
    return y / total


# ---------------------------------------------------------------------------
# HRP (dendrogram, optional box constraints)
# ---------------------------------------------------------------------------


def _cluster_variance(cov: np.ndarray, members: Sequence[int]) -> float:
    """HRP cluster risk: variance of the inverse-VARIANCE (IVP) sub-portfolio."""
    idx = np.asarray(list(members), dtype=int)
    sub = cov[np.ix_(idx, idx)]
    w = _inverse_variance_weights(np.clip(np.diag(sub), 0.0, None))
    return max(0.0, float(w @ sub @ w))


def _inverse_vol_weights(var: np.ndarray) -> np.ndarray:
    """Naive risk parity: weights proportional to ``1 / sigma`` (equal weight if degenerate)."""
    n = int(var.size)
    if n == 0:
        return np.zeros(0, dtype=float)
    positive = var > 0.0
    if not np.any(positive):
        return _equal_weights(n)
    inv = np.zeros(n, dtype=float)
    inv[positive] = 1.0 / np.sqrt(var[positive])
    total = float(inv.sum())
    return inv / total if total > 0.0 else _equal_weights(n)


def _naive_rp_cluster_variance(cov: np.ndarray, members: Sequence[int]) -> float:
    """HERC cluster risk: variance of the inverse-VOL (naive risk parity) sub-portfolio."""
    idx = np.asarray(list(members), dtype=int)
    sub = cov[np.ix_(idx, idx)]
    w = _inverse_vol_weights(np.clip(np.diag(sub), 0.0, None))
    return max(0.0, float(w @ sub @ w))


def _resolve_bounds(
    bounds: Mapping[str, Any] | tuple[Any, Any] | None, n: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if bounds is None:
        return None
    if isinstance(bounds, Mapping):
        lower, upper = bounds.get("lower", 0.0), bounds.get("upper", 1.0)
    else:
        lower, upper = bounds
    lo = np.broadcast_to(np.asarray(lower, dtype=float), (n,)).astype(float)
    hi = np.broadcast_to(np.asarray(upper, dtype=float), (n,)).astype(float)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("constrained HRP bounds must be finite")
    if np.any(lo < 0.0) or np.any(hi > 1.0):
        raise ValueError("constrained HRP bounds must lie in [0, 1]")
    if np.any(hi < lo - _TOL) or float(lo.sum()) > 1.0 + 1e-9 or float(hi.sum()) < 1.0 - 1e-9:
        raise ValueError(
            "infeasible constrained-HRP box: need lower <= upper elementwise and "
            f"sum(lower) <= 1 <= sum(upper) (got {float(lo.sum()):.6f} / {float(hi.sum()):.6f})"
        )
    return lo, hi


def hrp_dendrogram_weights(
    cov: Any,
    *,
    linkage_method: str = "single",
    bounds: Mapping[str, Any] | tuple[Any, Any] | None = None,
) -> np.ndarray:
    """Lopez de Prado HRP weights from a covariance matrix.

    ``bounds`` (optional) = ``{"lower": scalar|array, "upper": scalar|array}``
    or ``(lower, upper)``.  When given: (1) infeasible boxes raise
    ``ValueError`` (fail closed), (2) each recursive split is clipped to the
    interval that keeps both children's boxes satisfiable, (3) the result is
    projected onto ``{lo <= w <= hi, sum w = 1}`` exactly.
    """
    matrix = _ensure_psd(cov)
    n = int(matrix.shape[0])
    box = _resolve_bounds(bounds, n)  # raises early on an infeasible box
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    dist = correlation_distance(_cov_to_corr(matrix))
    link = hierarchical_linkage(dist, method=linkage_method)
    order = quasi_diagonal_order(link, n)
    weights = np.ones(n, dtype=float)
    clusters: list[list[int]] = [list(order)]
    while clusters:
        nxt: list[list[int]] = []
        for items in clusters:
            if len(items) < 2:
                continue
            half = len(items) // 2
            left, right = items[:half], items[half:]
            v_left, v_right = _cluster_variance(matrix, left), _cluster_variance(matrix, right)
            total = v_left + v_right
            alpha = 0.5 if total <= 0.0 else 1.0 - v_left / total
            parent = float(weights[left[0]])  # every member carries the parent's weight
            if box is not None and parent > 0.0:
                lo, hi = box
                lo_l, hi_l = float(lo[left].sum()), float(hi[left].sum())
                lo_r, hi_r = float(lo[right].sum()), float(hi[right].sum())
                a_min = max(lo_l, parent - hi_r) / parent
                a_max = min(hi_l, parent - lo_r) / parent
                if a_min <= a_max:
                    alpha = min(max(alpha, a_min), a_max)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            nxt.extend([left, right])
        clusters = nxt
    if box is not None:
        lo, hi = box
        weights = project_box_simplex(weights, lo, hi)
    total = float(weights.sum())
    return weights / total if total > 0.0 else _equal_weights(n)


# ---------------------------------------------------------------------------
# HERC
# ---------------------------------------------------------------------------


def herc_weights(
    cov: Any,
    *,
    linkage_method: str = "ward",
    n_clusters: int | None = None,
    max_clusters: int = 10,
) -> np.ndarray:
    """Raffinot (2018) Hierarchical Equal Risk Contribution weights.

    Cluster risk = variance of the inverse-VOL (naive risk parity, ``1/sigma``)
    portfolio of that cluster; a sub-tree's risk is the sum over the final
    clusters it contains.  Each split above the cut allocates
    ``alpha_left = 1 - risk_left / (risk_left + risk_right)``; inside a final
    cluster the assets get inverse-vol (``1/sigma``) weights -- unlike HRP, which
    uses inverse-variance inside its bisection.  Cluster count: explicit
    ``n_clusters`` or the mean-silhouette optimum (deterministic stand-in for
    the paper's Gap statistic).
    """
    matrix = _ensure_psd(cov)
    n = int(matrix.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    dist = correlation_distance(_cov_to_corr(matrix))
    link = hierarchical_linkage(dist, method=linkage_method)
    k = (
        int(n_clusters)
        if n_clusters
        else silhouette_optimal_clusters(dist, link, max_clusters=max_clusters)
    )
    labels = np.asarray(cut_tree(link, n, n_clusters=k), dtype=int)
    children = _children(link, n)
    cluster_risk: dict[int, float] = {}
    for lab in np.unique(labels):
        members = np.flatnonzero(labels == lab).tolist()
        cluster_risk[int(lab)] = _naive_rp_cluster_variance(matrix, members)

    def _subtree_risk(node: int) -> float:
        leaves = _leaves(node, children)
        labs = {int(labels[leaf]) for leaf in leaves}
        return sum(cluster_risk[lab] for lab in labs)

    def _is_within_cluster(node: int) -> bool:
        leaves = _leaves(node, children)
        return len({int(labels[leaf]) for leaf in leaves}) == 1

    cluster_weight = dict.fromkeys(cluster_risk, 0.0)
    root = n + int(link.shape[0]) - 1
    stack: list[tuple[int, float]] = [(root, 1.0)]
    while stack:
        node, w = stack.pop()
        if _is_within_cluster(node):
            lab = int(labels[_leaves(node, children)[0]])
            cluster_weight[lab] += w
            continue
        left, right = children[node]
        r_left, r_right = _subtree_risk(left), _subtree_risk(right)
        total = r_left + r_right
        alpha = 0.5 if total <= 0.0 else 1.0 - r_left / total
        stack.append((left, w * alpha))
        stack.append((right, w * (1.0 - alpha)))
    weights = np.zeros(n, dtype=float)
    for lab, w in cluster_weight.items():
        members = np.flatnonzero(labels == lab)
        local = _inverse_vol_weights(np.clip(np.diag(matrix)[members], 0.0, None))
        weights[members] = w * local
    total = float(weights.sum())
    return weights / total if total > 0.0 else _equal_weights(n)


# ---------------------------------------------------------------------------
# NCO
# ---------------------------------------------------------------------------


def nco_weights(
    cov: Any,
    *,
    mu: Any | None = None,
    linkage_method: str = "single",
    n_clusters: int | None = None,
    max_clusters: int = 10,
) -> np.ndarray:
    """Lopez de Prado (2019) Nested Clustered Optimization (long-only variant).

    Clusters come from the correlation-distance dendrogram (silhouette-optimal
    count unless ``n_clusters`` is given).  Intra-cluster and inter-cluster
    problems are the long-only min-variance QP (``mu=None``) or long-only
    max-Sharpe (``mu`` given), each solved to tolerance by projected gradient
    (KKT-checkable), not an unconstrained analytic solution clipped afterwards.
    """
    matrix = _ensure_psd(cov)
    n = int(matrix.shape[0])
    mean = None if mu is None else np.asarray(mu, dtype=float).reshape(-1)
    if mean is not None and (mean.size != n or not np.all(np.isfinite(mean))):
        raise ValueError("expected returns must be finite and match covariance dimensions")
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)
    dist = correlation_distance(_cov_to_corr(matrix))
    link = hierarchical_linkage(dist, method=linkage_method)
    k = (
        int(n_clusters)
        if n_clusters
        else silhouette_optimal_clusters(dist, link, max_clusters=max_clusters)
    )
    labels = np.asarray(cut_tree(link, n, n_clusters=k), dtype=int)
    uniq = [int(v) for v in np.unique(labels)]

    def _solve(sub_cov: np.ndarray, sub_mu: np.ndarray | None) -> np.ndarray:
        if sub_mu is None:
            return long_only_min_variance(sub_cov)
        return long_only_max_sharpe(sub_cov, sub_mu)

    intra = np.zeros((n, len(uniq)), dtype=float)
    for col, lab in enumerate(uniq):
        members = np.flatnonzero(labels == lab)
        sub_cov = matrix[np.ix_(members, members)]
        sub_mu = None if mean is None else mean[members]
        intra[members, col] = _solve(sub_cov, sub_mu)
    reduced_cov = intra.T @ matrix @ intra
    reduced_mu = None if mean is None else intra.T @ mean
    inter = _solve(reduced_cov, reduced_mu)
    weights = intra @ inter
    total = float(weights.sum())
    return weights / total if total > 0.0 else _equal_weights(n)


# ---------------------------------------------------------------------------
# Wasserstein DRO mean-variance (Blanchet-Chen-Zhou 2022, p = q = 2)
# ---------------------------------------------------------------------------


def wasserstein_dro_objective(cov: Any, weights: Any, *, radius: float) -> float:
    """Robust standard deviation ``sqrt(w'Sw) + sqrt(radius) * ||w||_2``."""
    raw = np.asarray(cov, dtype=float)
    r = float(radius)
    if not np.isfinite(r) or r < 0.0:
        raise ValueError("Wasserstein radius must be finite and non-negative")
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not np.all(np.isfinite(raw)):
        raise ValueError("covariance must be a finite square matrix")
    matrix = _ensure_psd(raw)
    w = np.asarray(weights, dtype=float)
    if w.size != matrix.shape[0] or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and match covariance dimensions")
    eps = math.sqrt(r)
    return math.sqrt(max(0.0, float(w @ matrix @ w))) + eps * float(np.linalg.norm(w))


def wasserstein_dro_weights(
    cov: Any,
    *,
    mu: Any | None = None,
    radius: float = 0.0,
    target_return: float | None = None,
    max_iter: int = 20_000,
    tol: float = 1e-13,
) -> np.ndarray:
    """Exact BCZ (2022) type-2 Wasserstein DRO mean-variance on the long-only simplex.

    ``eps = sqrt(radius)``.  Minimise ``sqrt(w'Sw) + eps*||w||_2`` subject to
    ``w >= 0``, ``sum(w) = 1`` and -- only when both ``mu`` and ``target_return``
    are given -- the robust-mean constraint ``mu'w - eps*||w||_2 >= target_return``.
    Convex objective / convex feasible set; solved by projected gradient with
    Armijo backtracking, the target constraint by dual bisection.  Raises
    ``ValueError`` when no simplex point meets the target.  ``radius = 0`` with a
    non-binding target reproduces :func:`long_only_min_variance`.  ``radius`` is
    in squared-return units (daily panels: ~1e-5..1e-4).
    """
    raw = np.asarray(cov, dtype=float)
    r = float(radius)
    if not np.isfinite(r) or r < 0.0:
        raise ValueError("Wasserstein radius must be finite and non-negative")
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not np.all(np.isfinite(raw)):
        raise ValueError("covariance must be a finite square matrix")
    matrix = _ensure_psd(raw)
    n = int(matrix.shape[0])
    mean = None if mu is None else np.asarray(mu, dtype=float).reshape(-1)
    if mean is not None and (mean.size != n or not np.all(np.isfinite(mean))):
        raise ValueError("expected returns must be finite and match covariance dimensions")
    if target_return is not None and not np.isfinite(float(target_return)):
        raise ValueError("target_return must be finite")
    if target_return is not None and mean is None:
        raise ValueError("target_return requires expected returns")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and positive")
    if n == 0:
        return np.zeros(0, dtype=float)
    eps = math.sqrt(r)
    if n == 1:
        if target_return is not None and float(mean[0]) - eps < float(target_return) - 1e-12:
            raise ValueError("infeasible target_return for singleton portfolio")
        return np.ones(1, dtype=float)
    radius_zero_weights: np.ndarray | None = None
    if r == 0.0:
        radius_zero_weights = long_only_min_variance(matrix, max_iter=max_iter, tol=tol)
        if min_variance_kkt_residual(matrix, radius_zero_weights) > 1e-7:
            raise RuntimeError("radius-zero Wasserstein solve lacks a min-variance KKT certificate")
        if (
            target_return is None
            or float(mean @ radius_zero_weights) >= float(target_return) - 1e-12
        ):
            return radius_zero_weights
    step0 = _lipschitz_step(matrix + eps * eps * np.eye(n)) * max(1.0, math.sqrt(n))

    def _sigma(w: np.ndarray) -> float:
        return math.sqrt(max(1e-300, float(w @ matrix @ w)))

    def _norm(w: np.ndarray) -> float:
        return max(1e-300, float(np.linalg.norm(w)))

    def _risk(w: np.ndarray) -> float:
        return _sigma(w) + eps * _norm(w)

    def _risk_grad(w: np.ndarray) -> np.ndarray:
        return matrix @ w / _sigma(w) + eps * w / _norm(w)

    def _robust_mean(w: np.ndarray) -> float:
        return float(mean @ w) - eps * float(np.linalg.norm(w))

    def _robust_mean_grad(w: np.ndarray) -> np.ndarray:
        return mean - eps * w / _norm(w)

    def project(v: np.ndarray) -> np.ndarray:
        return project_to_simplex(v, total=1.0)

    start = radius_zero_weights if radius_zero_weights is not None else _equal_weights(n)

    def _certified_solve(
        objective: Callable[[np.ndarray], float],
        gradient: Callable[[np.ndarray], np.ndarray],
        initial: np.ndarray,
        step: float,
    ) -> np.ndarray:
        solved = _projected_gradient(
            objective, gradient, project, initial, step0=step, max_iter=max_iter, tol=tol
        )
        cert_step = min(1.0, step)
        grad = gradient(solved)
        residual = float(np.linalg.norm(project(solved - cert_step * grad) - solved))
        residual /= cert_step * max(1.0, float(np.linalg.norm(grad)))
        if not np.isfinite(residual) or residual > max(1e-8, math.sqrt(tol)):
            raise RuntimeError("Wasserstein solver did not reach a projected-KKT certificate")
        return solved

    unconstrained = _certified_solve(_risk, _risk_grad, start, step0)
    if mean is None or target_return is None:
        return unconstrained
    target = float(target_return)
    if _robust_mean(unconstrained) >= target - 1e-12:
        return unconstrained
    # Feasibility: maximise the concave robust mean over the simplex.
    best_mean_w = _certified_solve(
        lambda w: -_robust_mean(w),
        lambda w: -_robust_mean_grad(w),
        start,
        1.0 / max(1e-12, float(np.abs(mean).max()) + eps),
    )
    if _robust_mean(best_mean_w) < target - 1e-9:
        raise ValueError(
            f"infeasible target_return={target:.6g}: max robust mean on the simplex is "
            f"{_robust_mean(best_mean_w):.6g}"
        )

    def _solve(lam: float) -> np.ndarray:
        return _certified_solve(
            lambda w: _risk(w) - lam * _robust_mean(w),
            lambda w: _risk_grad(w) - lam * _robust_mean_grad(w),
            unconstrained,
            step0 / (1.0 + lam),
        )

    lam_lo, lam_hi = 0.0, 1.0
    w_hi = _solve(lam_hi)
    for _ in range(80):
        if _robust_mean(w_hi) >= target - 1e-12:
            break
        lam_lo, lam_hi = lam_hi, lam_hi * 2.0
        w_hi = _solve(lam_hi)
    else:  # pragma: no cover - the feasibility check above should prevent this
        raise RuntimeError("Wasserstein target dual search did not converge")
    best = w_hi
    for _ in range(60):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        w_mid = _solve(lam_mid)
        if _robust_mean(w_mid) >= target - 1e-12:
            lam_hi, best = lam_mid, w_mid
        else:
            lam_lo = lam_mid
        if lam_hi - lam_lo <= 1e-12 * max(1.0, lam_hi):
            break
    primal_residual = max(0.0, target - _robust_mean(best))
    complementarity = lam_hi * abs(_robust_mean(best) - target)
    certificate_tol = max(1e-8, math.sqrt(tol))
    if primal_residual > certificate_tol or complementarity > certificate_tol:
        raise RuntimeError("Wasserstein target solve lacks primal/dual complementarity")
    return best


# ---------------------------------------------------------------------------
# Graph-risk heuristic (full-graph inverse eigenvector centrality)
# ---------------------------------------------------------------------------


def graph_inverse_centrality_weights(cov: Any, *, floor: float = 1e-6) -> np.ndarray:
    """Full-graph inverse-centrality graph-risk heuristic (permutation-equivariant).

    Adjacency ``A_ij = max(|rho_ij|, floor)`` for ``i != j`` and ``A_ii = 0``;
    ``z`` = Perron eigenvector of ``A`` (largest eigenvalue, taken elementwise
    absolute and normalised to sum 1); weights are the exact minimiser of
    ``sum_i z_i * w_i^2`` on the simplex, i.e. ``w_i ~ 1 / z_i``.  De-weights the
    most central (most connected) names.  Not a reproduction of MST/PMFG
    peripherality portfolios.
    """
    floor_value = float(floor)
    if not np.isfinite(floor_value) or floor_value <= 0.0:
        raise ValueError("graph adjacency floor must be finite and positive")
    matrix = _ensure_psd(cov)
    n = int(matrix.shape[0])
    if n == 0:
        return np.zeros(0, dtype=float)
    if n <= 2:
        return _equal_weights(n)
    corr = _cov_to_corr(matrix)
    adjacency = np.maximum(np.abs(corr), floor_value)
    np.fill_diagonal(adjacency, 0.0)
    try:
        evals, evecs = np.linalg.eigh(adjacency)
    except np.linalg.LinAlgError:
        return _equal_weights(n)
    perron = np.abs(evecs[:, int(np.argmax(evals))])
    total = float(perron.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _equal_weights(n)
    z = np.maximum(perron / total, 1e-12)
    weights = 1.0 / z
    return weights / float(weights.sum())


# ---------------------------------------------------------------------------
# Registered plugin wrappers (same contract as optimizers_extra)
# ---------------------------------------------------------------------------


def _prep(ids: list[str], returns_matrix: Any, cov_window: int | None, *, estimator: str = "lw"):
    ids = list(ids)
    matrix = _clean_matrix(returns_matrix)
    if not ids or matrix.shape[1] != len(ids):
        return None, None
    if estimator == "sample":
        rows = matrix if cov_window is None else matrix[-int(cov_window) :]
        if rows.shape[0] < 2:
            return None, None
        centered = rows - rows.mean(axis=0)
        cov = (centered.T @ centered) / float(rows.shape[0])  # MLE / n divisor
    else:
        cov, _ = ledoit_wolf_shrunk_covariance(matrix, window=cov_window)
    if cov.shape != (len(ids), len(ids)):
        return None, None
    return matrix, cov


@register("portfolio", "HRPDendrogram")
class HRPDendrogramPortfolio:
    """Dendrogram HRP (optionally box-constrained) -- see :func:`hrp_dendrogram_weights`."""

    def __init__(
        self,
        *,
        linkage_method: str = "single",
        lower: float | Sequence[float] | None = None,
        upper_bound: float | Sequence[float] | None = None,
        cov_window: int | None = None,
    ) -> None:
        self.linkage_method = str(linkage_method)
        self.lower = lower
        self.upper_bound = upper_bound
        self.cov_window = cov_window

    def allocate(
        self, ids: list[str], returns_matrix: Any, *, upper: dict[str, float] | None = None
    ) -> dict[str, float]:
        _matrix, cov = _prep(ids, returns_matrix, self.cov_window)
        if cov is None:
            return {}
        bounds = None
        if self.lower is not None or self.upper_bound is not None:
            bounds = {
                "lower": 0.0 if self.lower is None else self.lower,
                "upper": 1.0 if self.upper_bound is None else self.upper_bound,
            }
        if bounds is not None and upper is not None:
            lo, hi = _resolve_bounds(bounds, len(ids))
            hi = np.minimum(hi, np.asarray([upper.get(sid, 1.0) for sid in ids], dtype=float))
            bounds = (lo, hi)
        weights = hrp_dendrogram_weights(cov, linkage_method=self.linkage_method, bounds=bounds)
        return (
            dict(zip(ids, weights))
            if bounds is not None
            else _apply_upper_bounds(list(ids), weights, upper)
        )


@register("portfolio", "HERC")
class HERCPortfolio:
    """Hierarchical Equal Risk Contribution -- see :func:`herc_weights`."""

    def __init__(
        self,
        *,
        linkage_method: str = "ward",
        n_clusters: int | None = None,
        max_clusters: int = 10,
        cov_window: int | None = None,
    ) -> None:
        self.linkage_method = str(linkage_method)
        self.n_clusters = n_clusters
        self.max_clusters = int(max_clusters)
        self.cov_window = cov_window

    def allocate(
        self, ids: list[str], returns_matrix: Any, *, upper: dict[str, float] | None = None
    ) -> dict[str, float]:
        _matrix, cov = _prep(ids, returns_matrix, self.cov_window)
        if cov is None:
            return {}
        weights = herc_weights(
            cov,
            linkage_method=self.linkage_method,
            n_clusters=self.n_clusters,
            max_clusters=self.max_clusters,
        )
        return _apply_upper_bounds(list(ids), weights, upper)


@register("portfolio", "NCO")
class NCOPortfolio:
    """Nested Clustered Optimization -- see :func:`nco_weights`."""

    def __init__(
        self,
        *,
        use_mean: bool = False,
        linkage_method: str = "single",
        n_clusters: int | None = None,
        max_clusters: int = 10,
        cov_window: int | None = None,
    ) -> None:
        self.use_mean = bool(use_mean)
        self.linkage_method = str(linkage_method)
        self.n_clusters = n_clusters
        self.max_clusters = int(max_clusters)
        self.cov_window = cov_window

    def allocate(
        self, ids: list[str], returns_matrix: Any, *, upper: dict[str, float] | None = None
    ) -> dict[str, float]:
        matrix, cov = _prep(ids, returns_matrix, self.cov_window)
        if cov is None:
            return {}
        mu = matrix.mean(axis=0) if self.use_mean and matrix.shape[0] > 0 else None
        weights = nco_weights(
            cov,
            mu=mu,
            linkage_method=self.linkage_method,
            n_clusters=self.n_clusters,
            max_clusters=self.max_clusters,
        )
        return _apply_upper_bounds(list(ids), weights, upper)


@register("portfolio", "WassersteinDRO")
class WassersteinDROPortfolio:
    """BCZ Wasserstein DRO mean-variance -- see :func:`wasserstein_dro_weights`.

    ``radius`` is in squared-return units and defaults to 0 (= long-only
    min-variance) so a caller must state the ambiguity radius explicitly;
    ``target_return`` (per-bar) turns on the robust-mean constraint using the
    sample mean.  Covariance defaults to the MLE (``n`` divisor) sample matrix
    as in the paper; ``cov_estimator="ledoit_wolf"`` opts into shrinkage.
    """

    def __init__(
        self,
        *,
        radius: float = 0.0,
        target_return: float | None = None,
        cov_estimator: str = "sample",
        max_iter: int = 20_000,
        cov_window: int | None = None,
    ) -> None:
        self.radius = float(radius)
        self.target_return = None if target_return is None else float(target_return)
        self.cov_estimator = str(cov_estimator)
        self.max_iter = int(max_iter)
        self.cov_window = cov_window

    def allocate(
        self, ids: list[str], returns_matrix: Any, *, upper: dict[str, float] | None = None
    ) -> dict[str, float]:
        raw = np.asarray(returns_matrix, dtype=float)
        if raw.ndim not in (1, 2) or not np.all(np.isfinite(raw)):
            raise ValueError("Wasserstein returns must contain only finite values")
        estimator = "sample" if self.cov_estimator == "sample" else "lw"
        matrix, cov = _prep(ids, returns_matrix, self.cov_window, estimator=estimator)
        if cov is None:
            return {}
        mu = matrix.mean(axis=0) if self.target_return is not None else None
        weights = wasserstein_dro_weights(
            cov,
            mu=mu,
            radius=self.radius,
            target_return=self.target_return,
            max_iter=self.max_iter,
        )
        allocated = _apply_upper_bounds(list(ids), weights, upper)
        if self.target_return is not None:
            capped = np.asarray([allocated[sid] for sid in ids], dtype=float)
            robust_mean = float(mu @ capped) - math.sqrt(self.radius) * float(
                np.linalg.norm(capped)
            )
            if robust_mean < self.target_return - 1e-10:
                raise ValueError("upper bounds make the Wasserstein target_return infeasible")
        return allocated


@register("portfolio", "GraphInverseCentrality")
class GraphInverseCentralityPortfolio:
    """Full-graph inverse-centrality graph-risk heuristic -- see :func:`graph_inverse_centrality_weights`."""

    def __init__(self, *, floor: float = 1e-6, cov_window: int | None = None) -> None:
        self.floor = float(floor)
        self.cov_window = cov_window

    def allocate(
        self, ids: list[str], returns_matrix: Any, *, upper: dict[str, float] | None = None
    ) -> dict[str, float]:
        _matrix, cov = _prep(ids, returns_matrix, self.cov_window)
        if cov is None:
            return {}
        weights = graph_inverse_centrality_weights(cov, floor=self.floor)
        return _apply_upper_bounds(list(ids), weights, upper)
