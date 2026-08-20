"""HAR-RV (heterogeneous autoregressive realized-variance) primitives.

Corsi's HAR-RV model as a reusable, dependency-light indicator: aggregate
intraday squared log returns into UTC-day realized variance, build the lagged
1/5/22-day block-mean design (cumsum-based, loop-free), fit by OLS
(``numpy.linalg.lstsq``), and return the one-step-ahead variance forecast under
the repo's latest-scalar ``float | None`` never-raise contract.  Annualization
goes exclusively through :mod:`lumina_quant.indicators.annualization`.

EXTERNAL PRIOR
--------------
- Corsi (2009) -- HAR-RV is the standard, hard-to-beat RV forecast benchmark
  across asset classes.
- Andersen-Bollerslev-Diebold-Labys (2003) -- realized-variance measurement.
- Repo ledger: design-brief section 6 lists HAR-RV as census-verified missing;
  grep confirms zero HAR code anywhere.

EXPECTED NULL (pre-registered, verbatim)
----------------------------------------
"As an indicator there is no trading null; the honest null is that HAR adds
nothing over the existing ewma_volatility/garch11 for the repo's downstream
uses -- V-DIAG's baseline-vs-candidate design measures exactly this at the
program level."

FALSIFYING MEASUREMENT (pre-registered, verbatim)
-------------------------------------------------
"Unit-level: on a synthetic log-RV series generated from a known HAR
data-generating process, fitted coefficients must recover truth within
tolerance and the forecast must beat a naive lag-1 forecast on MSE; on
white-noise RV it must NOT beat naive (guards against implementation bias).
Consumer-level: V-DIAG baseline QLIKE using this estimator must match a
hand-rolled reference implementation to 1e-12 on fixtures."

All entry points are pure (no I/O), deterministic, and never raise on
adversarial input: degenerate series propagate ``None`` (scalar contract) or
empty tuples (series builders).  No scipy/sklearn/statsmodels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lumina_quant.indicators.annualization import (
    annualize_per_bar_vol,
    bars_per_year_from_spacing,
)

__all__ = [
    "DEFAULT_HAR_LAGS",
    "LOG_RV_EPS",
    "daily_realized_variance",
    "har_annualized_vol_forecast",
    "har_design",
    "har_fit",
    "har_rv_forecast",
    "log_rv_transform",
]

# Pre-registered constants (Corsi 2009 daily/weekly/monthly cascade).
DEFAULT_HAR_LAGS: tuple[int, ...] = (1, 5, 22)
# Floor applied before the log transform so zero-RV days stay finite.
LOG_RV_EPS: float = 1e-18
# Minimum usable design rows before an OLS fit is attempted (4 params * 8).
_MIN_FIT_ROWS: int = 32
_SECONDS_PER_DAY: int = 86_400


def _finite_float(value: Any) -> float | None:
    """Best-effort float coercion returning ``None`` on non-finite/garbage."""
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def daily_realized_variance(
    closes: Any,
    epoch_seconds: Any,
    *,
    min_intraday_returns: int = 12,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Aggregate intraday squared log returns into UTC-day realized variance.

    A squared log return joins UTC day ``D`` (``epoch // 86400``) only when BOTH
    endpoints of the return fall inside ``D`` (strictly intraday; the cross-day
    overnight return is excluded by construction).  Days carrying fewer than
    ``min_intraday_returns`` intraday returns are dropped as unreliable RV
    estimates (pre-registered default 12).

    Returns ``(day_ids, rv_values)`` as aligned ascending tuples; ``((), ())``
    on empty/degenerate/non-iterable input.  Never raises.
    """
    try:
        close_list = list(closes)
        epoch_list = list(epoch_seconds)
    except TypeError:
        return (), ()
    floor_returns = max(1, int(min_intraday_returns)) if min_intraday_returns else 1

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    prev_close: float | None = None
    prev_day: int | None = None
    for raw_close, raw_epoch in zip(close_list, epoch_list, strict=False):
        close = _finite_float(raw_close)
        epoch = _finite_float(raw_epoch)
        if close is None or epoch is None or close <= 0.0:
            prev_close = None
            prev_day = None
            continue
        day = int(epoch // _SECONDS_PER_DAY)
        if prev_close is not None and prev_day == day:
            log_ret = math.log(close / prev_close)
            sums[day] = sums.get(day, 0.0) + log_ret * log_ret
            counts[day] = counts.get(day, 0) + 1
        prev_close = close
        prev_day = day

    days = sorted(day for day, count in counts.items() if count >= floor_returns)
    if not days:
        return (), ()
    return tuple(days), tuple(float(sums[day]) for day in days)


def _clean_rv_array(rv_values: Any, *, eps: float) -> np.ndarray | None:
    """Return the RV series floored at ``eps`` as a 1-D array, or ``None``.

    Non-finite entries and negative variances make the series unusable and
    propagate ``None`` (fail-closed); never raises.
    """
    try:
        arr = np.asarray(list(rv_values), dtype=float)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)) or bool(np.any(arr < 0.0)):
        return None
    try:
        floor = float(eps) if math.isfinite(float(eps)) and float(eps) > 0.0 else LOG_RV_EPS
    except TypeError, ValueError:
        floor = LOG_RV_EPS
    return np.maximum(arr, floor)


def log_rv_transform(rv_values: Any, *, eps: float = LOG_RV_EPS) -> np.ndarray | None:
    """Return ``log(max(rv, eps))`` as a 1-D float array, or ``None``.

    Non-finite entries and negative variances make the series unusable and
    propagate ``None`` (fail-closed); never raises.
    """
    cleaned = _clean_rv_array(rv_values, eps=eps)
    if cleaned is None:
        return None
    return np.log(cleaned)


def har_design(
    log_rv: Any,
    *,
    lags: tuple[int, ...] = DEFAULT_HAR_LAGS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the HAR block-mean design ``(X, y)`` from a log-RV series.

    Row ``t`` (for ``t`` in ``[max(lags), n)``) predicts ``log_rv[t]`` from an
    intercept plus, per lag ``k``, the trailing block mean
    ``mean(log_rv[t-k:t])`` -- computed via one cumulative sum, no Python
    loops over rows.  Returns ``None`` when the series is shorter than
    ``max(lags) + 1``, non-finite, or the lag tuple is degenerate.  Never
    raises.
    """
    try:
        arr = np.asarray(list(log_rv), dtype=float)
        lag_list = sorted({int(k) for k in lags})
    except Exception:
        return None
    if arr.ndim != 1 or not lag_list or lag_list[0] < 1:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    n = int(arr.size)
    max_lag = lag_list[-1]
    if n <= max_lag:
        return None

    csum = np.concatenate(([0.0], np.cumsum(arr)))
    rows = np.arange(max_lag, n)
    columns = [np.ones(rows.size, dtype=float)]
    for k in lag_list:
        columns.append((csum[rows] - csum[rows - k]) / float(k))
    design = np.column_stack(columns)
    target = arr[rows]
    return design, target


def _har_forecast_row(arr: np.ndarray, lags: list[int]) -> np.ndarray:
    """Return the next-step regressor row ``[1, block means ending at n]``."""
    n = int(arr.size)
    row = [1.0]
    for k in lags:
        row.append(float(np.mean(arr[n - k : n])))
    return np.asarray(row, dtype=float)


def har_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """OLS coefficients via ``np.linalg.lstsq`` (min-norm on rank deficiency).

    Returns ``None`` on shape mismatch, non-finite output, or solver failure;
    never raises.
    """
    try:
        if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
            return None
        if design.shape[0] < design.shape[1]:
            return None
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    except Exception:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    return coef


def har_rv_forecast(
    rv_values: Any,
    *,
    lags: tuple[int, ...] = DEFAULT_HAR_LAGS,
    log_space: bool = True,
    eps: float = LOG_RV_EPS,
    min_fit_rows: int = _MIN_FIT_ROWS,
) -> float | None:
    """One-step-ahead HAR realized-variance forecast (latest scalar).

    Fits OLS on the lagged 1/5/22 block means of the (log-transformed, when
    ``log_space``) daily RV series and forecasts the NEXT day's variance.  The
    forecast is returned on the VARIANCE scale (``exp`` of the log-space
    prediction).  Returns ``None`` on fewer than ``max(lags) + min_fit_rows``
    usable observations, negative/non-finite inputs, or solver failure.  Never
    raises.
    """
    if log_space:
        series = log_rv_transform(rv_values, eps=eps)
    else:
        series = _clean_rv_array(rv_values, eps=eps)
    if series is None:
        return None
    built = har_design(series, lags=lags)
    if built is None:
        return None
    design, target = built
    if design.shape[0] < max(int(min_fit_rows), design.shape[1]):
        return None
    coef = har_fit(design, target)
    if coef is None:
        return None
    lag_list = sorted({int(k) for k in lags})
    prediction = float(np.dot(_har_forecast_row(series, lag_list), coef))
    if not math.isfinite(prediction):
        return None
    forecast = math.exp(prediction) if log_space else max(0.0, prediction)
    if not math.isfinite(forecast) or forecast < 0.0:
        return None
    return float(forecast)


def har_annualized_vol_forecast(
    rv_values: Any,
    *,
    times: Any = None,
    bars_per_year: float | None = None,
    lags: tuple[int, ...] = DEFAULT_HAR_LAGS,
    log_space: bool = True,
    eps: float = LOG_RV_EPS,
) -> float | None:
    """Annualized volatility from the HAR one-step variance forecast.

    ``sqrt(forecast)`` is annualized exclusively through
    :mod:`lumina_quant.indicators.annualization`: pass ``bars_per_year``
    directly or provide the observation epoch ``times`` so the cadence is
    inferred via ``bars_per_year_from_spacing`` (365.25-day year).  ``None``
    propagates from every degenerate leg; never raises.
    """
    forecast = har_rv_forecast(rv_values, lags=lags, log_space=log_space, eps=eps)
    if forecast is None:
        return None
    per_bar_vol = math.sqrt(forecast)
    bpy = bars_per_year
    if bpy is None and times is not None:
        bpy = bars_per_year_from_spacing(times)
    return annualize_per_bar_vol(per_bar_vol, bpy)
