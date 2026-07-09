"""Cross-sectional Gram-Schmidt residualization (pure Python, never-raise).

THEORY / PURPOSE.  Many cross-sectional signals load partially on axes another
factor already trades (momentum, nearness-to-high, size, ...).  To isolate the
part of a target signal that is orthogonal to a set of nuisance regressors, one
regresses the target on the regressors (with an intercept) and keeps the OLS
residual.  With several regressors the numerically stable, allocation-free way
to do this is sequential Gram-Schmidt: orthogonalize each regressor against the
ones already accepted, drop any regressor whose residual norm collapses to zero
(perfectly collinear / degenerate), and project the (demeaned) target onto the
resulting orthogonal basis, subtracting each projection in turn.  Because the
basis vectors are mutually orthogonal, the sequential subtraction yields exactly
the multiple-regression residual of the target on ``span({regressors, 1})``.

This is a CROSS-SECTIONAL primitive: every input list is one observation per
symbol, aligned by position, for a single decision bar.  It is not a rolling
time-series transform.

The function is pure (``math`` only, no numpy / scipy / sklearn), deterministic
(fixed-order reductions), and never raises: degenerate, mismatched, empty, or
non-finite input yields ``None`` rather than an exception.
"""

from __future__ import annotations

import math

_EPS = 1e-12


def _dot(a: list[float], b: list[float]) -> float:
    """Fixed-order dot product of two equal-length vectors."""
    return math.fsum(x * y for x, y in zip(a, b, strict=False))


def cross_sectional_residualize(
    target: list[float],
    regressors: list[list[float]] | None,
    *,
    eps: float = _EPS,
) -> list[float] | None:
    """Return the cross-sectional OLS residual of ``target`` on ``regressors``.

    ``target`` is a per-symbol vector (one value per symbol, aligned by
    position) and ``regressors`` is a list of same-length per-symbol vectors.
    An intercept is always included (every vector, including the target, is
    demeaned before projection), so the result is the residual of the target on
    ``span({1, *regressors})``.

    The regressors are orthogonalized via sequential Gram-Schmidt; a regressor
    whose residual squared-norm is ``<= eps`` (perfectly collinear with the ones
    already accepted, or constant) is skipped rather than inverted, so the
    routine is stable under collinearity.  When ``regressors`` is empty (or
    ``None``) the demeaned target is returned unchanged.

    Returns ``None`` (never raises) when ``target`` is empty, when any regressor
    length differs from ``len(target)``, or when any input value is non-finite.

    Args:
        target: Per-symbol target vector to residualize.
        regressors: List of per-symbol regressor vectors (may be empty/None).
        eps: Squared-norm floor below which a regressor is treated as
            degenerate/collinear and dropped.

    Returns:
        The residual vector aligned with ``target``, or ``None`` when unavailable.
    """
    if not isinstance(target, (list, tuple)):
        return None
    n = len(target)
    if n == 0:
        return None
    target_f = [float(value) for value in target]
    if any(not math.isfinite(value) for value in target_f):
        return None

    reg_list: list[list[float]] = []
    if regressors:
        for reg in regressors:
            if not isinstance(reg, (list, tuple)) or len(reg) != n:
                return None
            reg_f = [float(value) for value in reg]
            if any(not math.isfinite(value) for value in reg_f):
                return None
            reg_list.append(reg_f)

    eps_f = max(0.0, float(eps))

    # Demean the target (intercept term).
    mean_t = math.fsum(target_f) / float(n)
    resid = [value - mean_t for value in target_f]

    # Build a mutually-orthogonal basis from the demeaned regressors and project
    # the target residual onto each accepted basis vector in turn.
    basis: list[tuple[list[float], float]] = []
    for reg_f in reg_list:
        mean_r = math.fsum(reg_f) / float(n)
        vec = [value - mean_r for value in reg_f]
        for basis_vec, basis_norm2 in basis:
            coef = _dot(vec, basis_vec) / basis_norm2
            vec = [vi - coef * bi for vi, bi in zip(vec, basis_vec, strict=False)]
        norm2 = _dot(vec, vec)
        if norm2 <= eps_f:
            # Degenerate / collinear regressor: contributes no new direction.
            continue
        basis.append((vec, norm2))
        coef = _dot(resid, vec) / norm2
        resid = [ri - coef * vi for ri, vi in zip(resid, vec, strict=False)]

    if any(not math.isfinite(value) for value in resid):
        return None
    return resid


__all__ = ["cross_sectional_residualize"]
