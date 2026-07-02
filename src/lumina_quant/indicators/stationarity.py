"""Stationarity screening primitives: ADF t-statistic and AR(1) half-life.

Method absorbed from the precious-metal research repo's from-scratch
cointegration toolkit (``python_adf_test`` in ``VECM.ipynb`` /
``VAR_VECM_python_ver.ipynb``): an augmented Dickey-Fuller regression solved
with plain least squares plus tabulated critical values, so mean-reversion
sleeves can require statistical evidence that a trailing window actually
reverts before fading it (the source repo's own docs flag the absence of such
quantitative screening as its top follow-up).  Everything here is pure Python
(no numpy/scipy/statsmodels) and operates on the TRAILING window the caller
passes, so there is no full-sample lookahead.

The ADF regression with ``lags = k`` is::

    dy_t = c + gamma * y_{t-1} + sum_{i=1..k} phi_i * dy_{t-i} + e_t

and the reported statistic is ``t(gamma) = gamma_hat / se(gamma_hat)``.  More
negative means stronger rejection of a unit root (i.e. stationary /
mean-reverting).  Critical values are the standard asymptotic constant-only
Dickey-Fuller values.
"""

from __future__ import annotations

import math

# Asymptotic Dickey-Fuller critical values, constant-only regression.
_ADF_CRITICAL_VALUES = {
    "1%": -3.43,
    "5%": -2.86,
    "10%": -2.57,
}


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Solve ``matrix @ x = rhs`` via Gaussian elimination with partial pivoting."""
    size = len(matrix)
    aug = [[*row, rhs[idx]] for idx, row in enumerate(matrix)]
    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) <= 1e-12:
            return None
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for row in range(col + 1, size):
            factor = aug[row][col] / pivot
            if factor == 0.0:
                continue
            for k in range(col, size + 1):
                aug[row][k] -= factor * aug[col][k]
    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        acc = aug[row][size] - sum(aug[row][k] * solution[k] for k in range(row + 1, size))
        solution[row] = acc / aug[row][row]
    if not all(math.isfinite(value) for value in solution):
        return None
    return solution


def adf_t_statistic(values, *, lags: int = 0) -> float | None:
    """Return the augmented Dickey-Fuller t-statistic over the trailing window.

    Constant-only ADF regression with ``lags`` lagged difference terms; more
    negative output means stronger evidence of stationarity.  Returns ``None``
    on insufficient (< ``lags + 8`` usable rows) or non-finite history, or a
    degenerate regression (flat window / singular normal equations).

    Args:
        values: Trailing level series (oldest first), e.g. log closes.
        lags: Number of lagged difference regressors (``>= 0``).

    Returns:
        The t-statistic of the unit-root coefficient, or ``None``.
    """
    lag_count = max(0, int(lags))
    series = [float(value) for value in list(values)]
    if not all(math.isfinite(value) for value in series):
        return None
    diffs = [series[idx] - series[idx - 1] for idx in range(1, len(series))]
    # Row t regresses diffs[t] on [1, series[t], diffs[t-1], ..., diffs[t-lags]].
    rows = len(diffs) - lag_count
    n_params = 2 + lag_count
    if rows < n_params + 6:
        return None
    design: list[list[float]] = []
    target: list[float] = []
    for t in range(lag_count, len(diffs)):
        row = [1.0, series[t]]
        for i in range(1, lag_count + 1):
            row.append(diffs[t - i])
        design.append(row)
        target.append(diffs[t])

    xtx = [
        [sum(design[r][i] * design[r][j] for r in range(rows)) for j in range(n_params)]
        for i in range(n_params)
    ]
    xty = [sum(design[r][i] * target[r] for r in range(rows)) for i in range(n_params)]
    coeffs = _solve_linear_system(xtx, xty)
    if coeffs is None:
        return None
    residual_ss = 0.0
    for r in range(rows):
        fitted = sum(design[r][i] * coeffs[i] for i in range(n_params))
        residual = target[r] - fitted
        residual_ss += residual * residual
    dof = rows - n_params
    if dof <= 0:
        return None
    sigma_sq = residual_ss / float(dof)
    if sigma_sq <= 0.0 or not math.isfinite(sigma_sq):
        return None
    # var(gamma_hat) = sigma^2 * [(X'X)^-1]_{gamma,gamma}; gamma is column 1.
    unit = [0.0] * n_params
    unit[1] = 1.0
    inverse_col = _solve_linear_system(xtx, unit)
    if inverse_col is None:
        return None
    gamma_var = sigma_sq * inverse_col[1]
    if gamma_var <= 0.0 or not math.isfinite(gamma_var):
        return None
    t_stat = coeffs[1] / math.sqrt(gamma_var)
    return t_stat if math.isfinite(t_stat) else None


def adf_critical_value(significance) -> float | None:
    """Return the constant-only asymptotic ADF critical value for ``significance``.

    Accepted keys: ``"1%"``, ``"5%"``, ``"10%"`` — returns ``None`` otherwise.
    """
    return _ADF_CRITICAL_VALUES.get(str(significance))


def ar1_half_life(values) -> float | None:
    """Return the AR(1) mean-reversion half-life in bars over the trailing window.

    Fits ``(y_t - mean) = phi * (y_{t-1} - mean)`` by least squares and maps
    ``phi`` to ``-ln(2) / ln(phi)``.  Returns ``None`` on insufficient or
    non-finite history, a degenerate regressor, or a non-mean-reverting fit
    (``phi <= 0`` or ``phi >= 1``).
    """
    series = [float(value) for value in list(values)]
    count = len(series)
    if count < 8 or not all(math.isfinite(value) for value in series):
        return None
    center = sum(series) / float(count)
    demeaned = [value - center for value in series]
    denom = sum(value * value for value in demeaned[:-1])
    if denom <= 1e-12:
        return None
    phi = sum(demeaned[t] * demeaned[t - 1] for t in range(1, count)) / denom
    if not math.isfinite(phi) or phi <= 0.0 or phi >= 1.0:
        return None
    half_life = -math.log(2.0) / math.log(phi)
    if not math.isfinite(half_life) or half_life <= 0.0:
        return None
    return half_life


__all__ = [
    "adf_critical_value",
    "adf_t_statistic",
    "ar1_half_life",
]
