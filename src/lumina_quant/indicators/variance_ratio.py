"""Lo-MacKinlay variance-ratio primitives (indicator-level extraction).

Fills the census-verified indicator gap "Lo-MacKinlay variance-ratio
(indicator-level)".  :func:`variance_ratio` is EXTRACTED VERBATIM
(parity-locked) from the previously private ``_variance_ratio`` in
``strategies/cusum_varratio_alpha_sleeves.py`` -- the documented BIASED
overlapping estimator stays the DEFAULT so the consuming
``VarianceRatioTrendRiderStrategy`` signals are byte-identical: its
finite-sample downward bias is deliberately conservative for
``VR >= 1 + threshold`` persistence gates (it raises, never lowers, the bar).
Opt-in additions for future stat-gated consumers:

- ``variance_ratio(returns, k, unbiased=True)`` -- the Lo-MacKinlay unbiased
  estimator with ``m = k * (n - k + 1) * (1 - k / n)`` in the k-period
  denominator and the ``n - 1`` sample denominator in the 1-period leg.
- :func:`variance_ratio_zstat` -- the Lo-MacKinlay test statistic on the
  unbiased ratio; heteroskedasticity-robust by default (their z*(q) with the
  ``delta_j`` White-style weights), homoskedastic-iid variant opt-in.

External prior (pre-registered): Lo & MacKinlay (1988, RFS) "Stock Market
Prices Do Not Follow Random Walks" -- the variance-ratio test with
heteroskedasticity-robust asymptotics.

EXPECTED NULL (verbatim, pre-registered): "No new alpha claim -- this is a
dedup/extraction. Null at consumer level is already priced:
VarianceRatioTrendRider's evidence is whatever it is; extraction must not
change it by one byte."

FALSIFYING MEASUREMENT (verbatim, pre-registered): "Parity test: on pinned
fixture return vectors, indicator-level variance_ratio(returns,k) must equal
the current strategy-private _variance_ratio to exact float equality, and the
refactored VarianceRatioTrendRiderStrategy must produce byte-identical signals
on the existing data-free sleeve tests. Any diff = revert. For the new z-stat:
golden values against hand-computed Lo-MacKinlay small-sample cases."

Conventions: pure stdlib (no numpy/scipy/statsmodels), latest-scalar
``float | None``, never-raise (garbage/short/degenerate input propagates
``None``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from lumina_quant.indicators.alpha_features import finite_floats

_EPS = 1e-12

__all__ = [
    "variance_ratio",
    "variance_ratio_zstat",
]


def variance_ratio(
    returns: Iterable[Any] | None, k: int, *, unbiased: bool = False
) -> float | None:
    """Return the Lo-MacKinlay variance ratio ``Var(k-ret) / (k * Var(1-ret))``.

    Uses overlapping ``k``-period sums.  ``~1`` random walk, ``>1`` persistent,
    ``<1`` mean-reverting.  ``None`` when there are too few points or the
    one-period variance is degenerate.  The DEFAULT (``unbiased=False``) is the
    BIASED overlapping estimator (it divides ``Var(k)`` by the simple count
    ``n-k+1`` rather than the Lo-MacKinlay unbiased ``k(n-k+1)(1-k/n)``
    denominator), so a true random walk reads slightly BELOW 1 in finite
    samples; that bias is conservative for a ``VR >= 1 + threshold``
    persistence gate (it raises, never lowers, the bar).

    PARITY LOCK: the default path is extracted VERBATIM from the previously
    strategy-private ``_variance_ratio`` in
    ``strategies/cusum_varratio_alpha_sleeves.py`` (which now imports this
    function); on the finite float lists that sleeve produces the output is
    bit-for-bit identical (the only addition is the never-raise
    ``finite_floats`` coercion, a no-op on such input).

    ``unbiased=True`` (opt-in, for future stat-gated consumers) applies the
    Lo-MacKinlay (1988) small-sample corrections: the 1-period variance uses
    the ``n - 1`` sample denominator and the overlapping k-period variance uses
    ``m = k * (n - k + 1) * (1 - k / n)``.
    """
    values = finite_floats(returns)
    n = len(values)
    try:
        kk = max(2, int(k))
    except Exception:
        return None
    if n < kk + 1:
        return None
    mean = sum(values) / n
    sum_sq = sum((r - mean) * (r - mean) for r in values)
    var1 = sum_sq / n
    if var1 <= _EPS:
        return None
    acc = 0.0
    count = 0
    target = kk * mean
    for i in range(n - kk + 1):
        s = math.fsum(values[i : i + kk])
        acc += (s - target) * (s - target)
        count += 1
    if count <= 0:
        return None
    if not unbiased:
        var_k = acc / count
        return var_k / (kk * var1)
    # Lo-MacKinlay unbiased denominators (opt-in).
    m = float(kk) * float(n - kk + 1) * (1.0 - float(kk) / float(n))
    if m <= _EPS or n < 2:
        return None
    var1_unbiased = sum_sq / float(n - 1)
    if var1_unbiased <= _EPS:
        return None
    return (acc / m) / var1_unbiased


def variance_ratio_zstat(
    returns: Iterable[Any] | None, k: int, *, heteroskedastic: bool = True
) -> float | None:
    """Lo-MacKinlay z-statistic for ``VR(k) = 1`` (random walk), on the unbiased ratio.

    ``heteroskedastic=True`` (default) is the robust ``z*(k)`` of Lo-MacKinlay
    (1988): ``(VR(k) - 1) / sqrt(theta(k))`` with
    ``theta(k) = sum_{j=1}^{k-1} (2(k-j)/k)^2 * delta_j`` and
    ``delta_j = sum_t (r_t-mu)^2 (r_{t-j}-mu)^2 / (sum_t (r_t-mu)^2)^2``.
    ``heteroskedastic=False`` is the iid variant with asymptotic variance
    ``phi(k) = 2(2k-1)(k-1) / (3kn)``.  Positive z favors persistence, negative
    favors mean reversion.  ``None`` on short/degenerate input, never raises.
    """
    values = finite_floats(returns)
    n = len(values)
    try:
        kk = max(2, int(k))
    except Exception:
        return None
    if n < kk + 1:
        return None
    vr = variance_ratio(values, kk, unbiased=True)
    if vr is None:
        return None
    if not heteroskedastic:
        phi = 2.0 * (2.0 * kk - 1.0) * (kk - 1.0) / (3.0 * kk * n)
        if phi <= _EPS:
            return None
        return (vr - 1.0) / math.sqrt(phi)
    mean = sum(values) / n
    dev_sq = [(r - mean) * (r - mean) for r in values]
    total_sq = math.fsum(dev_sq)
    if total_sq <= _EPS:
        return None
    theta = 0.0
    for j in range(1, kk):
        delta_num = math.fsum(dev_sq[t] * dev_sq[t - j] for t in range(j, n))
        delta_j = delta_num / (total_sq * total_sq)
        weight = 2.0 * (kk - j) / kk
        theta += weight * weight * delta_j
    if theta <= _EPS:
        return None
    return (vr - 1.0) / math.sqrt(theta)
