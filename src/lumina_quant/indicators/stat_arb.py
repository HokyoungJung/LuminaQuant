"""Statistical-arbitrage primitives: 2-state Kalman hedge ratio and PCA residual s-scores.

Independent adaptations of textbook stat-arb machinery popularised in Korean
retail-quant education (pairs trading / market-microstructure lineage) and in
the primary literature:

* Online Kalman-filter regression ``y_t = beta_t * x_t + alpha_t + eps`` with a
  random-walk state (Chan-style). Unlike the scalar RLS in ``rolling_stats``,
  the state is 2-dimensional (slope + intercept) with an explicit process-noise
  ``delta`` and observation-noise ``obs_noise``; the standardized innovation
  ``e_t / sqrt(S_t)`` is the tradable z-score.
* Avellaneda & Lee (2010) PCA residual s-scores: regress each asset's returns on
  the top-``k`` eigenportfolios, cumulate the residual, fit an OU/AR(1) and
  report ``s = -m / sigma_eq`` (the auxiliary process is centred by
  construction when the regression carries an intercept).

Both are pure numpy/math (no scipy/sklearn) and never raise; degenerate input
returns ``None``.  Hypothesis primitives only -- no performance claim.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .common import safe_float


@dataclass(frozen=True, slots=True)
class KalmanHedgeState:
    """Posterior state of the 2-state (slope, intercept) Kalman regression."""

    beta: float
    alpha: float
    p00: float
    p01: float
    p11: float
    innovation: float | None = None
    innovation_var: float | None = None
    updates: int = 0

    @property
    def innovation_z(self) -> float | None:
        """Standardized innovation ``e / sqrt(S)`` (``None`` before the first update)."""
        if self.innovation is None or self.innovation_var is None or self.innovation_var <= 0.0:
            return None
        return self.innovation / math.sqrt(self.innovation_var)

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "beta": self.beta,
            "alpha": self.alpha,
            "p00": self.p00,
            "p01": self.p01,
            "p11": self.p11,
            "innovation": self.innovation,
            "innovation_var": self.innovation_var,
            "updates": self.updates,
        }

    @classmethod
    def from_dict(cls, payload) -> KalmanHedgeState | None:
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                beta=float(payload["beta"]),
                alpha=float(payload["alpha"]),
                p00=float(payload["p00"]),
                p01=float(payload["p01"]),
                p11=float(payload["p11"]),
                innovation=safe_float(payload.get("innovation")),
                innovation_var=safe_float(payload.get("innovation_var")),
                updates=max(0, int(payload.get("updates", 0))),
            )
        except KeyError, TypeError, ValueError:
            return None


def kalman_hedge_initial_state(*, init_var: float = 1.0) -> KalmanHedgeState:
    """Diffuse prior: ``beta = alpha = 0`` with variance ``init_var`` on both."""
    variance = max(1e-12, float(init_var))
    return KalmanHedgeState(beta=0.0, alpha=0.0, p00=variance, p01=0.0, p11=variance)


def kalman_hedge_ratio_step(
    state: KalmanHedgeState | None,
    y,
    x,
    *,
    delta: float = 1e-4,
    obs_noise: float = 1e-3,
    init_var: float = 1.0,
) -> KalmanHedgeState | None:
    """One predict/update step of the random-walk regression Kalman filter.

    ``delta`` maps to process noise ``Q = delta / (1 - delta) * I`` (Chan's
    parameterisation); ``obs_noise`` is the observation variance ``R``.  Returns
    the new posterior, or ``None`` when ``y``/``x`` are non-finite (the caller
    keeps the old state).
    """
    y_f, x_f = safe_float(y), safe_float(x)
    if y_f is None or x_f is None:
        return None
    if state is None:
        state = kalman_hedge_initial_state(init_var=init_var)
    delta_f = min(0.999_999, max(0.0, float(delta)))
    q = delta_f / (1.0 - delta_f) if delta_f > 0.0 else 0.0
    r = max(1e-18, float(obs_noise))
    # predict
    p00 = state.p00 + q
    p01 = state.p01
    p11 = state.p11 + q
    # innovation with H = [x, 1]
    predicted = state.beta * x_f + state.alpha
    innovation = y_f - predicted
    s = x_f * x_f * p00 + 2.0 * x_f * p01 + p11 + r
    if not math.isfinite(s) or s <= 0.0:
        return None
    k0 = (x_f * p00 + p01) / s
    k1 = (x_f * p01 + p11) / s
    beta = state.beta + k0 * innovation
    alpha = state.alpha + k1 * innovation
    # P = P - K H P
    hp0 = x_f * p00 + p01
    hp1 = x_f * p01 + p11
    n00 = p00 - k0 * hp0
    n01 = p01 - k0 * hp1
    n11 = p11 - k1 * hp1
    if not all(math.isfinite(v) for v in (beta, alpha, n00, n01, n11)):
        return None
    return KalmanHedgeState(
        beta=beta,
        alpha=alpha,
        p00=max(0.0, n00),
        p01=n01,
        p11=max(0.0, n11),
        innovation=innovation,
        innovation_var=s,
        updates=state.updates + 1,
    )


def kalman_hedge_ratio(
    ys, xs, *, delta: float = 1e-4, obs_noise: float = 1e-3, init_var: float = 1.0
) -> KalmanHedgeState | None:
    """Run the filter over aligned sequences and return the final posterior.

    ``None`` if fewer than two valid observations were processed.
    """
    state: KalmanHedgeState | None = None
    for y, x in zip(list(ys), list(xs)):
        nxt = kalman_hedge_ratio_step(
            state, y, x, delta=delta, obs_noise=obs_noise, init_var=init_var
        )
        if nxt is not None:
            state = nxt
    if state is None or state.updates < 2:
        return None
    return state


def kalman_spread(state: KalmanHedgeState | None, y, x) -> float | None:
    """A-posteriori spread ``y - beta*x - alpha`` under ``state``."""
    y_f, x_f = safe_float(y), safe_float(x)
    if state is None or y_f is None or x_f is None:
        return None
    value = y_f - state.beta * x_f - state.alpha
    return value if math.isfinite(value) else None


def _ar1_fit(x: np.ndarray) -> tuple[float, float, float] | None:
    """OLS ``x[t+1] = a + b x[t] + zeta``; returns ``(a, b, var_zeta)`` or ``None``."""
    if x.shape[0] < 8:
        return None
    lag, lead = x[:-1], x[1:]
    lag_c = lag - lag.mean()
    denom = float(lag_c @ lag_c)
    if denom <= 1e-18:
        return None
    b = float(lag_c @ (lead - lead.mean())) / denom
    a = float(lead.mean() - b * lag.mean())
    resid = lead - (a + b * lag)
    dof = max(1, lead.shape[0] - 2)
    var = float(resid @ resid) / dof
    if not all(math.isfinite(v) for v in (a, b, var)):
        return None
    return a, b, var


def pca_residual_sscores(
    returns_rows: Sequence[Sequence[float]],
    *,
    n_factors: int = 1,
    max_half_life_bars: float | None = None,
    min_rows: int = 30,
) -> list[float | None]:
    """Avellaneda-Lee s-scores for every column of a ``T x N`` returns panel.

    Steps: standardize columns -> eigen-decompose the correlation matrix (numpy
    ``eigh``) -> top-``n_factors`` eigenportfolios (loadings / vol) -> per-asset
    OLS on ``[1, factors]`` -> cumulative residual -> AR(1) fit -> ``s = -m /
    sigma_eq`` with ``m = a/(1-b)``, ``sigma_eq = sqrt(var/(1-b^2))``.

    A column gets ``None`` when its variance is zero, the AR(1) slope is outside
    ``(0, 1)`` (no mean reversion), or its implied half-life exceeds
    ``max_half_life_bars`` (default: half the panel length).  The whole result is
    ``None``-filled when the panel is too short (< ``min_rows``), non-finite, or
    narrower than ``n_factors + 2`` columns.
    """
    try:
        panel = np.asarray([[float(v) for v in row] for row in returns_rows], dtype=float)
    except TypeError, ValueError:
        return []
    if panel.ndim != 2 or panel.size == 0:
        return []
    rows, cols = panel.shape
    out: list[float | None] = [None] * cols
    k = max(1, int(n_factors))
    if rows < max(8, int(min_rows)) or cols < k + 2 or not np.all(np.isfinite(panel)):
        return out
    mean = panel.mean(axis=0)
    std = panel.std(axis=0, ddof=1)
    valid = std > 1e-12
    if int(valid.sum()) < k + 2:
        return out
    sub = panel[:, valid]
    z = (sub - mean[valid]) / std[valid]
    corr = (z.T @ z) / float(rows - 1)
    try:
        evals, evecs = np.linalg.eigh(corr)
    except np.linalg.LinAlgError:
        return out
    order = np.argsort(evals)[::-1][:k]
    loadings = evecs[:, order] / std[valid][:, None]  # eigenportfolio weights
    factors = sub @ loadings  # T x k factor returns
    design = np.column_stack([np.ones(rows), factors])
    hl_cap = float(max_half_life_bars) if max_half_life_bars else rows / 2.0
    valid_idx = np.flatnonzero(valid)
    for pos, col in enumerate(valid_idx):
        y = sub[:, pos]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        cum = np.cumsum(resid)
        fit = _ar1_fit(cum)
        if fit is None:
            continue
        a, b, var = fit
        if b <= 0.0 or b >= 1.0 or var <= 0.0:
            continue
        half_life = -math.log(2.0) / math.log(b)
        if not math.isfinite(half_life) or half_life > hl_cap:
            continue
        m = a / (1.0 - b)
        sigma_eq = math.sqrt(var / (1.0 - b * b))
        if sigma_eq <= 0.0 or not math.isfinite(sigma_eq):
            continue
        s = -m / sigma_eq
        if math.isfinite(s):
            out[int(col)] = float(s)
    return out


__all__ = [
    "KalmanHedgeState",
    "kalman_hedge_initial_state",
    "kalman_hedge_ratio",
    "kalman_hedge_ratio_step",
    "kalman_spread",
    "pca_residual_sscores",
]
