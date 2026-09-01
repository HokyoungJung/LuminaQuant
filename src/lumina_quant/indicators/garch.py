"""Deterministic GARCH(1,1) conditional-volatility primitives (pure Python).

Method absorbed from the precious-metal research repo's ARMA-GARCH extreme
band notebook (``GARCH (1).ipynb``): the volatility-clustering recursion
``sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}`` used there
via R's ``fGarch`` is re-implemented dependency-free.  Instead of a numerical
MLE optimizer (non-deterministic across platforms), the fit uses VARIANCE
TARGETING — ``omega = s^2 * (1 - alpha - beta)`` with ``s^2`` the sample
variance, so the long-run variance always matches the window — plus an
exhaustive deterministic grid search over ``(alpha, beta)`` maximizing the
Gaussian quasi log-likelihood.  Same inputs always give the same parameters.

The point of the conditional (rather than rolling/realized) variance is that
it re-prices risk IMMEDIATELY after a shock, which is what makes standardized
innovations ``r_t / sigma_{t|t-1}`` a sharper breakout/exhaustion yardstick
than a plain rolling z-score.
"""

from __future__ import annotations

import math

_MIN_OBS = 32
_STATIONARY_CAP = 0.998

# Deterministic (alpha, beta) search grids: alpha 0.02..0.30, beta 0.60..0.98.
_ALPHA_GRID = tuple(round(0.02 * step, 6) for step in range(1, 16))
_BETA_GRID = tuple(round(0.60 + 0.02 * step, 6) for step in range(0, 20))


def _finite_returns(values) -> list[float] | None:
    """Return ``values`` as floats, or ``None`` when empty/non-finite."""
    series = [float(value) for value in list(values)]
    if not series or not all(math.isfinite(value) for value in series):
        return None
    return series


def _sample_variance(values: list[float]) -> float | None:
    """Return the (ddof=1) sample variance, or ``None`` when degenerate."""
    count = len(values)
    if count < 2:
        return None
    center = sum(values) / float(count)
    # d * d (not d ** 2): float ** raises OverflowError on huge-but-finite
    # inputs, while multiplication overflows to inf and hits the guard below.
    variance = 0.0
    for value in values:
        delta = value - center
        variance += delta * delta
    variance /= float(count - 1)
    if variance <= 1e-18 or not math.isfinite(variance):
        return None
    return variance


def garch11_fit(returns) -> tuple[float, float, float] | None:
    """Return deterministic variance-targeted GARCH(1,1) ``(omega, alpha, beta)``.

    Fits by exhaustive grid search over ``(alpha, beta)`` (stationarity
    ``alpha + beta <= 0.998`` enforced) with ``omega`` pinned by variance
    targeting, maximizing the Gaussian quasi log-likelihood of the demeaned
    returns.  Ties resolve to the first grid point scanned, so the result is
    reproducible bit-for-bit.  Returns ``None`` on fewer than 32 observations,
    non-finite input, or a flat window.

    Args:
        returns: Trailing per-bar returns (oldest first), e.g. log returns.

    Returns:
        ``(omega, alpha, beta)`` or ``None`` when unavailable.
    """
    series = _finite_returns(returns)
    if series is None or len(series) < _MIN_OBS:
        return None
    variance = _sample_variance(series)
    if variance is None:
        return None
    center = sum(series) / float(len(series))
    eps_sq = [(value - center) * (value - center) for value in series]
    if not all(math.isfinite(value) for value in eps_sq):
        return None

    best: tuple[float, float, float] | None = None
    best_ll = -math.inf
    for alpha in _ALPHA_GRID:
        for beta in _BETA_GRID:
            persistence = alpha + beta
            if persistence > _STATIONARY_CAP:
                continue
            omega = variance * (1.0 - persistence)
            sigma_sq = variance
            log_lik = 0.0
            for value in eps_sq:
                if sigma_sq <= 1e-18:
                    log_lik = -math.inf
                    break
                log_lik -= math.log(sigma_sq) + value / sigma_sq
                sigma_sq = omega + alpha * value + beta * sigma_sq
            if log_lik > best_ll:
                best_ll = log_lik
                best = (omega, alpha, beta)
    if best is None or not math.isfinite(best_ll):
        return None
    return best


def garch11_next_variance(returns, *, omega, alpha, beta) -> float | None:
    """Return the one-step-ahead GARCH(1,1) variance forecast for the returns.

    Runs the recursion ``sigma^2_{t+1} = omega + alpha * eps^2_t + beta *
    sigma^2_t`` over the demeaned trailing returns (initialized at the sample
    variance) and returns the variance forecast for the NEXT bar.  Returns
    ``None`` on invalid parameters (non-finite, negative, or non-stationary)
    or degenerate history.
    """
    omega_val = float(omega)
    alpha_val = float(alpha)
    beta_val = float(beta)
    if not all(math.isfinite(value) for value in (omega_val, alpha_val, beta_val)):
        return None
    if omega_val <= 0.0 or alpha_val < 0.0 or beta_val < 0.0:
        return None
    if alpha_val + beta_val > _STATIONARY_CAP:
        return None
    series = _finite_returns(returns)
    if series is None:
        return None
    variance = _sample_variance(series)
    if variance is None:
        return None
    center = sum(series) / float(len(series))
    sigma_sq = variance
    for value in series:
        eps = value - center
        sigma_sq = omega_val + alpha_val * eps * eps + beta_val * sigma_sq
    if sigma_sq <= 0.0 or not math.isfinite(sigma_sq):
        return None
    return sigma_sq


__all__ = [
    "garch11_fit",
    "garch11_next_variance",
]
