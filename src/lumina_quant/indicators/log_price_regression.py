"""Trailing log-price regression primitives for the OLS-basis alpha sleeves.

Two families of pure numerics share the same trailing-log-price machinery and
are gathered here so the regression is built once, cleanly, and consumed by the
cross-sectional OLS-basis sleeves (trend-quality and path-convexity):

- **Signed trend QUALITY** -- the ordinary-least-squares fit of ``log(close)``
  on a bar-index grid over a trailing window, expressed as ``sign(slope) * R^2``
  (a directional goodness-of-fit).  A smooth, near-linear trend scores near
  ``+/-1``; a jumpy or choppy move of the same net magnitude scores low.  Built
  on the existing pure :func:`ts_regression_slope` / :func:`ts_regression_rsquared`
  primitives (``rolling_stats``); a closed-form slope t-statistic is provided as
  an alternative score.

- **Orthonormal path CONVEXITY** -- the coefficient of the trailing log-price
  vector on the *quadratic* member of a discrete orthonormal polynomial basis
  ``(p0=level, p1=linear, p2=quadratic)`` built by closed-form Gram-Schmidt over
  the fixed index grid.  Because ``p2`` is orthogonal to ``p1`` by construction,
  this curvature score carries ZERO loading on the trailing-return level: adding
  any linear trend ``a + b*t`` to the log-price path leaves the score unchanged.
  That orthogonality is the whole point -- it is a curvature signal that a
  first-order momentum sleeve cannot span.

The module is pure (``math`` / ``functools`` only -- no numpy, scipy, sklearn, or
statsmodels), deterministic (fixed-order reductions, per-window basis cache), and
returns ``None`` on degenerate input rather than raising.
"""

from __future__ import annotations

import math
from functools import lru_cache

from lumina_quant.indicators.rolling_stats import (
    ts_regression_rsquared,
    ts_regression_slope,
)

_EPS = 1e-12


def trailing_log_closes(closes, window: int) -> list[float] | None:
    """Return ``log`` of the trailing ``window`` closes, or ``None`` if degenerate.

    Requires at least ``window`` samples and strictly positive, finite closes in
    the trailing window (a non-positive or non-finite close makes the log
    undefined, so the whole window is rejected -- never-raise skip).
    """
    window_i = int(window)
    if window_i < 2:
        return None
    tail = list(closes)[-window_i:]
    if len(tail) < window_i:
        return None
    out: list[float] = []
    for value in tail:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            return None
        out.append(math.log(parsed))
    return out


@lru_cache(maxsize=64)
def orthonormal_polynomial_basis(
    window: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None:
    """Return orthonormal discrete polynomials ``(p0, p1, p2)`` over ``0..window-1``.

    Closed-form Gram-Schmidt over ``{1, t, t^2}`` on the integer grid, normalized
    to unit length.  Deterministic and cached per window length (the grid never
    changes).  Returns ``None`` when ``window < 3`` (no room for a quadratic
    basis vector) or the quadratic residual is degenerate.
    """
    n = int(window)
    if n < 3:
        return None
    grid = [float(i) for i in range(n)]

    def _dot(a: list[float], b: list[float]) -> float:
        return sum(a[i] * b[i] for i in range(n))

    v0 = [1.0] * n
    norm0 = math.sqrt(_dot(v0, v0))
    if norm0 <= _EPS:
        return None
    p0 = [value / norm0 for value in v0]

    proj_t = _dot(grid, p0)
    u1 = [grid[i] - proj_t * p0[i] for i in range(n)]
    norm1 = math.sqrt(_dot(u1, u1))
    if norm1 <= _EPS:
        return None
    p1 = [value / norm1 for value in u1]

    t2 = [value * value for value in grid]
    proj0 = _dot(t2, p0)
    proj1 = _dot(t2, p1)
    u2 = [t2[i] - proj0 * p0[i] - proj1 * p1[i] for i in range(n)]
    norm2 = math.sqrt(_dot(u2, u2))
    if norm2 <= _EPS:
        return None
    p2 = [value / norm2 for value in u2]

    return tuple(p0), tuple(p1), tuple(p2)


def log_price_trend_fit(closes, *, window: int) -> tuple[float, float] | None:
    """Return ``(slope, r2)`` of ``log(close)`` regressed on the bar index.

    ``None`` on degenerate input (short history, non-positive close, flat fit).
    """
    y = trailing_log_closes(closes, window)
    if y is None:
        return None
    x = [float(i) for i in range(len(y))]
    slope = ts_regression_slope(x, y)
    r2 = ts_regression_rsquared(x, y)
    if slope is None or r2 is None:
        return None
    return float(slope), float(r2)


def signed_trend_quality(closes, *, window: int) -> tuple[float, float, float] | None:
    """Return ``(signed_r2, r2, slope)`` for the trailing log-price OLS fit.

    ``signed_r2 = sign(slope) * r2`` -- a directional trend-quality score in
    ``[-1, 1]``.  ``None`` on degenerate input.
    """
    fit = log_price_trend_fit(closes, window=window)
    if fit is None:
        return None
    slope, r2 = fit
    sign = 1.0 if slope > 0.0 else (-1.0 if slope < 0.0 else 0.0)
    return sign * r2, r2, slope


def trend_slope_t_stat(r2: float, n: int) -> float | None:
    """Return the (non-negative) slope t-statistic from ``R^2`` and sample size.

    Closed form ``t = sqrt((n - 2) * R^2 / (1 - R^2))`` (no scipy).  A perfect
    fit (``R^2 -> 1``) saturates the denominator rather than diverging.  ``None``
    when ``n <= 2`` or the inputs are non-finite.
    """
    r2f = float(r2)
    n_i = int(n)
    if n_i <= 2 or not math.isfinite(r2f):
        return None
    r2f = max(0.0, min(1.0, r2f))
    denom = max(1.0 - r2f, _EPS)
    t_squared = (n_i - 2) * r2f / denom
    if t_squared < 0.0 or not math.isfinite(t_squared):
        return None
    return math.sqrt(t_squared)


def orthonormal_path_convexity(closes, *, window: int, vol_floor: float = 1e-6) -> float | None:
    """Return the vol-normalized quadratic (curvature) coefficient of log price.

    Projects the trailing log-price vector onto the orthonormal quadratic basis
    member ``p2`` and divides by the rolling standard deviation of the window's
    log returns (floored at ``vol_floor``) so the score is scale-free and
    cross-sectionally comparable.  Because ``p2`` is orthogonal to the linear
    basis member, the RAW projection -- and, since a constant return shift does
    not change the return variance, the normalized score too -- is invariant to
    adding any linear trend to the log-price path (zero first-order loading).
    ``None`` on degenerate input.
    """
    y = trailing_log_closes(closes, window)
    if y is None:
        return None
    basis = orthonormal_polynomial_basis(window)
    if basis is None:
        return None
    _p0, _p1, p2 = basis
    n = len(y)
    c2 = sum(p2[i] * y[i] for i in range(n))
    returns = [y[i] - y[i - 1] for i in range(1, n)]
    if len(returns) < 2:
        return None
    mean_ret = sum(returns) / float(len(returns))
    variance = sum((r - mean_ret) ** 2 for r in returns) / float(len(returns) - 1)
    std = math.sqrt(variance) if variance > 0.0 else 0.0
    std = max(std, float(vol_floor))
    result = c2 / std
    return result if math.isfinite(result) else None


__all__ = [
    "log_price_trend_fit",
    "orthonormal_path_convexity",
    "orthonormal_polynomial_basis",
    "signed_trend_quality",
    "trailing_log_closes",
    "trend_slope_t_stat",
]
