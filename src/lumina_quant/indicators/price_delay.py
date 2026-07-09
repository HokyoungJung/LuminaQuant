"""Hou-Moskowitz price-delay share (Dimson-beta nonsynchronicity numeric).

``price_delay_share`` measures how much of a symbol's systematic return
variation is explained by the *lagged* benchmark returns rather than the
contemporaneous benchmark return -- the Hou & Moskowitz (2005, RFS) ``D1``
delay statistic, a sign-free, scale-free ``[0, 1]`` characteristic.

Two nested market-model regressions are estimated over a trailing window of
aligned ``(asset, benchmark)`` log returns:

- RESTRICTED  ``r_i,t = a + b0 * r_m,t``
- FULL        ``r_i,t = a + b0 * r_m,t + sum_{n=1..L} b_n * r_m,t-n``

on the SAME sample rows (the last ``L`` rows the lagged columns consume are
dropped from BOTH designs so the two ``R^2`` values are directly comparable),
solved by ordinary least squares via the normal equations
(``np.linalg.solve`` on the Gram matrix -- the pure-numpy pattern already used
by ``advanced_alpha.cross_leadlag_spillover``).

The delay share is::

    D1 = clip(1 - R2_restricted / R2_full, 0, 1)

A name whose returns load only on the *contemporaneous* benchmark (instantly
priced) has ``R2_restricted ~= R2_full`` and ``D1 ~= 0``; a name whose returns
load on *lagged* benchmark moves (slow diffusion) has ``R2_restricted ~= 0``
while ``R2_full`` stays high, so ``D1 -> 1``.

The optional ``score_mode="lag_weighted"`` returns the lag-weighted coefficient
share ``D2 = sum_n (n * |b_n|) / (|b0| + sum_n |b_n|)`` from the full model
instead of the ``R^2`` ratio.

Design guards (never raise, ``None`` on undefined):

- ``None`` when fewer than ``min_obs`` usable regression rows survive after the
  lag trim, or the histories are too short / non-finite.
- ``None`` when ``R2_full <= 0`` -- a coin with NO systematic loading has an
  UNDEFINED delay share and must never be admitted (the pure-idiosyncratic
  exclusion guard); likewise on a singular Gram matrix.

This is a pure numeric (numpy only): no I/O, no global state, deterministic
(fixed-order reductions), and it never raises.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

_EPS = 1e-12
_SCORE_MODES = ("d1", "lag_weighted")


def _finite_floats(values: Sequence[float] | object) -> list[float]:
    """Coerce an iterable to a list of finite floats, dropping anything else."""
    if not isinstance(values, (list, tuple, np.ndarray)):
        try:
            values = list(values)  # type: ignore[arg-type]
        except Exception:
            return []
    out: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except TypeError, ValueError:
            continue
        if math.isfinite(parsed):
            out.append(parsed)
    return out


def _ols_r2_beta(design: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Return ``(R^2, beta)`` for OLS of ``y`` on ``design`` (intercept included).

    Solves the normal equations via ``np.linalg.solve`` on the Gram matrix.
    Returns ``None`` on a singular system or a degenerate (zero-variance) ``y``.
    """
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    if sst <= _EPS:
        return None
    gram = design.T @ design
    try:
        beta = np.linalg.solve(gram, design.T @ y)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    resid = y - design @ beta
    sse = float(resid @ resid)
    r2 = 1.0 - (sse / sst)
    if not math.isfinite(r2):
        return None
    return r2, beta


def price_delay_share(
    asset_returns: Sequence[float] | object,
    bench_returns: Sequence[float] | object,
    *,
    lags: int = 5,
    min_obs: int = 30,
    min_r2: float = 0.10,
    score_mode: str = "d1",
) -> float | None:
    """Return the Hou-Moskowitz ``D1`` price-delay share in ``[0, 1]`` or ``None``.

    ``asset_returns`` / ``bench_returns`` are trailing log-return series (the
    caller aligns them to the same trailing bars).  ``lags`` is the number of
    benchmark lags in the full model; ``min_obs`` is the minimum surviving
    regression sample; ``min_r2`` is the ADJUSTED full-model ``R^2`` floor below
    which the symbol carries NO usable systematic loading and the delay share is
    undefined (the pure-idiosyncratic exclusion guard -- the degrees-of-freedom
    penalty drives a spurious finite-sample fit below it); ``score_mode`` is
    ``"d1"`` (the ``R^2`` ratio) or
    ``"lag_weighted"`` (the lag-weighted coefficient share).
    """
    lag_order = max(1, int(lags))
    r2_floor = max(_EPS, float(min_r2))
    mode = str(score_mode).lower()
    if mode not in _SCORE_MODES:
        mode = "d1"

    asset = _finite_floats(asset_returns)
    bench = _finite_floats(bench_returns)
    count = min(len(asset), len(bench))
    if count < lag_order + 2:
        return None
    a = np.asarray(asset[-count:], dtype=float)
    m = np.asarray(bench[-count:], dtype=float)

    rows = count - lag_order
    if rows < max(int(min_obs), lag_order + 2):
        return None

    y = a[lag_order:]
    contemp = m[lag_order:]
    ones = np.ones(rows, dtype=float)

    restricted = np.column_stack((ones, contemp))
    full_cols = [ones, contemp]
    for k in range(1, lag_order + 1):
        full_cols.append(m[lag_order - k : count - k])
    full = np.column_stack(full_cols)

    r2_full = _ols_r2_beta(full, y)
    if r2_full is None:
        return None
    r2f, beta_full = r2_full
    # Systematic-loading guard on the ADJUSTED full-model R^2: a pure-idiosyncratic
    # coin overfits (k / n) worth of spurious raw R^2 with many lag regressors, so
    # the raw floor cannot exclude it -- the degrees-of-freedom penalty drives that
    # spurious fit toward zero while a genuine loading survives it.
    n_predictors = full.shape[1] - 1  # exclude the intercept column
    dof = rows - n_predictors - 1
    if dof <= 0:
        return None
    adjusted_r2f = 1.0 - ((1.0 - r2f) * (rows - 1) / dof)
    if not math.isfinite(adjusted_r2f) or adjusted_r2f <= r2_floor:
        # No usable systematic loading -> delay share is undefined, not zero
        # (a pure-idiosyncratic coin must never be admitted to either tail).
        return None

    if mode == "lag_weighted":
        # beta_full = [intercept, b0(contemporaneous), b1, ..., bL].
        b0 = abs(float(beta_full[1]))
        lag_coeffs = [abs(float(beta_full[2 + idx])) for idx in range(lag_order)]
        denom = b0 + sum(lag_coeffs)
        if denom <= _EPS:
            return None
        weighted = sum(float(idx + 1) * lag_coeffs[idx] for idx in range(lag_order))
        return float(max(0.0, min(1.0, weighted / denom)))

    r2_restricted = _ols_r2_beta(restricted, y)
    if r2_restricted is None:
        return None
    r2r = max(0.0, r2_restricted[0])
    delay = 1.0 - (r2r / r2f)
    return float(max(0.0, min(1.0, delay)))


__all__ = ["price_delay_share"]
