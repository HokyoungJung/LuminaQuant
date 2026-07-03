"""Hill (1975) tail-index estimator: extreme-value-theory tail SHAPE, not level.

THEORY.  Under extreme-value theory (EVT / Pickands-Balkema-de Haan), the
upper tail of many heavy-tailed distributions is asymptotically a power law:
``P(X > x) ~ x^(-alpha)`` for large ``x``.  Hill (1975) estimates the tail
exponent ``alpha`` directly from the ``k`` largest order statistics of a
positive sample, with NO distributional assumption beyond regularly-varying
tails.  A SMALLER ``alpha`` means a FATTER (heavier) tail; a LARGER ``alpha``
means a thinner one.  Kelly & Jiang (2014, "Tail Risk and Asset Prices") show
that a fattening common tail-risk factor (their ``alpha`` moving lower)
predicts elevated crash risk cross-sectionally and carries a return premium --
i.e. the SHAPE of the loss tail, not its current level or recent extremes,
carries forward-looking information.

This module estimates the tail-index EXPONENT of the loss-return distribution
(a continuous shape parameter recovered from many order statistics).  This is
DISTINCT from :mod:`lumina_quant.indicators.rare_event`, which scores how
extreme a SINGLE observation is relative to its own history (a per-bar rarity
percentile, not a distributional shape estimate); the two answer different
questions and can be used together without redundancy.

Functions:
- :func:`hill_tail_index`: the Hill (1975) tail-index estimator over the ``k``
  largest exceedances of a positive-valued sample.
- :func:`tail_index_ratio`: the ratio of a short-horizon Hill estimate to a
  long-horizon baseline, i.e. whether the tail is fattening (``< 1``) or
  thinning (``>= 1``) relative to its own recent history -- a regime signal
  rather than a point estimate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def _finite_positive(values: Iterable[float]) -> list[float]:
    """Return the finite, strictly-positive floats in ``values`` (order preserved)."""
    out: list[float] = []
    for raw in values:
        try:
            value = float(raw)
        except TypeError, ValueError:
            continue
        if math.isfinite(value) and value > 0.0:
            out.append(value)
    return out


def hill_tail_index(values: Iterable[float], *, k: int) -> float | None:
    """Return the Hill (1975) tail-index estimator over the ``k`` largest values.

    ``values`` must be a POSITIVE-valued sample (the caller passes, e.g., the
    absolute value of negative returns so the estimator reads the LOSS tail).
    Writing the order statistics in descending order ``x_(1) >= x_(2) >= ... >=
    x_(n)``, the estimator is::

        alpha_hat = k / sum(ln(x_(i) / x_(k+1)) for i in 1..k)

    i.e. the reciprocal of the mean log-excess of the ``k`` largest values over
    the ``(k+1)``-th largest (the threshold).  Smaller ``alpha_hat`` means a
    fatter (heavier) tail.

    Guards (never raises): requires ``k >= 2``; requires at least ``k + 1``
    finite, strictly-positive values; requires the threshold ``x_(k+1) > 0``
    (guaranteed by the positivity filter, kept explicit for clarity); and
    requires a finite, well-defined result (returns ``None`` on a degenerate
    log-sum, e.g. all top-``k`` values tied with the threshold).

    Args:
        values: A positive-valued sample (e.g. absolute loss returns).
        k: Number of upper order statistics to use in the estimator.

    Returns:
        The Hill tail-index estimate ``alpha_hat``, or ``None`` when unavailable.
    """
    k_int = int(k)
    if k_int < 2:
        return None
    sample = _finite_positive(values)
    if len(sample) < k_int + 1:
        return None
    ordered = sorted(sample, reverse=True)
    threshold = ordered[k_int]  # x_(k+1): the (k+1)-th largest order statistic.
    if threshold <= 0.0:
        return None
    log_sum = math.fsum(math.log(x / threshold) for x in ordered[:k_int])
    if log_sum <= 0.0:
        return None
    alpha_hat = k_int / log_sum
    return alpha_hat if math.isfinite(alpha_hat) else None


def tail_index_ratio(
    short_window_values: Iterable[float],
    long_window_values: Iterable[float],
    *,
    k_short: int,
    k_long: int,
) -> float | None:
    """Return the short-horizon Hill tail index relative to a long-horizon baseline.

    ``ratio = hill_tail_index(short_window_values, k=k_short) /
    hill_tail_index(long_window_values, k=k_long)``.  A ratio ``< 1`` means the
    recent tail index has FALLEN below its own baseline, i.e. the loss tail is
    FATTENING (heavier); a ratio ``>= 1`` means it is stable or thinning.  This
    is a regime comparison (self-relative), not an absolute tail-index level,
    so it is comparable across symbols with structurally different baseline
    tail shapes.

    Guards (never raises): ``None`` when either estimate is unavailable, when
    the long-horizon baseline is non-positive, or when the ratio is non-finite.

    Args:
        short_window_values: Positive-valued sample over the recent window.
        long_window_values: Positive-valued sample over the full baseline window.
        k_short: Upper order statistics count for the short-horizon estimate.
        k_long: Upper order statistics count for the long-horizon estimate.

    Returns:
        The tail-index ratio, or ``None`` when unavailable.
    """
    hill_short = hill_tail_index(short_window_values, k=k_short)
    hill_long = hill_tail_index(long_window_values, k=k_long)
    if hill_short is None or hill_long is None or hill_long <= 0.0:
        return None
    ratio = hill_short / hill_long
    return ratio if math.isfinite(ratio) else None


__all__ = [
    "hill_tail_index",
    "tail_index_ratio",
]
