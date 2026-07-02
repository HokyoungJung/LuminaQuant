"""Eigen-free effective number of independent trials.

The multiple-testing penalty in the Deflated Sharpe Ratio assumes ``num_trials``
*independent* strategy trials. When a research family shares highly correlated
candidates the raw candidate count over-states the breadth of the search, which
inflates the penalty and over-rejects genuine edges. We therefore collapse the
candidate return-correlation matrix ``C`` to an *effective* number of independent
trials via the participation ratio.

Eigen-free identity
-------------------
For a symmetric positive-semidefinite ``C`` with eigenvalues ``lambda_i``::

    N_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2
          = (trace C)^2 / ||C||_F^2
          = (trace C)^2 / sum_ij C_ij^2

The second line drops the eigenvalues entirely: ``trace C = sum_i lambda_i`` and
``||C||_F^2 = sum_i lambda_i^2 = sum_ij C_ij^2`` for a symmetric matrix. No
eigendecomposition (``eigh``/``svd``/scipy) is required. For a correlation matrix
the unit diagonal gives ``trace C = S`` (the candidate count), so::

    N_eff = S^2 / sum_ij C_ij^2

which is exactly ``1`` when every candidate is perfectly correlated (rank-1) and
exactly ``S`` when every candidate is mutually orthogonal (identity).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "candidate_return_correlation_matrix",
    "effective_number_of_trials",
    "raw_number_of_trials",
]


def raw_number_of_trials(correlation_matrix: np.ndarray) -> int:
    """Raw candidate count = leading dimension of the correlation matrix.

    This is an *upper bound* on the effective breadth and must only ever be used
    for logging / audit; it must never be fed to the multiple-testing penalty as
    the independent-trial count.
    """
    c = np.asarray(correlation_matrix, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("correlation_matrix must be a square 2-D array")
    return int(c.shape[0])


def effective_number_of_trials(
    correlation_matrix: np.ndarray,
    *,
    raw_n: int | None = None,
) -> float:
    """Participation-ratio effective number of independent trials (eigen-free).

    Parameters
    ----------
    correlation_matrix:
        Square candidate return-correlation matrix ``C`` (unit diagonal expected
        for a genuine correlation matrix, but not required by the arithmetic).
    raw_n:
        Optional explicit upper clamp. Defaults to ``C.shape[0]``.

    Returns:
    -------
    float
        ``N_eff`` clamped to ``[1, raw_n]``. Computed as
        ``(trace C)^2 / sum_ij C_ij^2`` with no eigendecomposition.
    """
    c = np.asarray(correlation_matrix, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("correlation_matrix must be a square 2-D array")
    s = c.shape[0]
    upper = float(s if raw_n is None else int(raw_n))
    if s == 0:
        return 1.0
    if upper < 1.0:
        upper = 1.0

    trace = float(np.trace(c))
    frob_sq = float(np.sum(c * c))
    if not np.isfinite(frob_sq) or frob_sq <= 0.0:
        return 1.0

    n_eff = (trace * trace) / frob_sq
    if not np.isfinite(n_eff):
        return 1.0
    return float(min(upper, max(1.0, n_eff)))


def candidate_return_correlation_matrix(returns_matrix: np.ndarray) -> np.ndarray:
    """Deterministic candidate return-correlation matrix (eigen-free helper).

    ``returns_matrix`` has shape ``(n_candidates, n_periods)``. Non-finite / zero
    -variance rows are treated as uncorrelated (their off-diagonal entries are set
    to ``0`` and the diagonal is forced to ``1``) so that the matrix stays a valid
    correlation matrix for :func:`effective_number_of_trials`.
    """
    r = np.asarray(returns_matrix, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns_matrix must be a 2-D (n_candidates, n_periods) array")
    n = r.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.ones((1, 1), dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(r)
    c = np.asarray(c, dtype=float)
    c = np.where(np.isfinite(c), c, 0.0)
    np.fill_diagonal(c, 1.0)
    # Symmetrize defensively against floating-point asymmetry.
    return 0.5 * (c + c.T)
