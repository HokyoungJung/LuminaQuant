"""Cross-sectional turnover-share and z-extremeness primitives (pure Python).

Method absorbed from the Reference-Price repo (``index_maker.ipynb`` /
``analysis.ipynb``), which builds trade-flow indices from customs data: an
item's share of the day's total traded amount (``amt_ratio_index``), dense
ranks of that share with rolling rank persistence, a bounded CDF extremeness
map ``2 * |Phi(z) - 0.5|`` of a rolling z-score, and a top-N share-weighted
composite price index.  Translated to a perp universe the "traded amount"
becomes per-bar quote (dollar) volume, so these helpers quantify where the
universe's turnover is concentrating and how anomalous a flow reading is.

These are stateless building blocks for factor/research pipelines: the caller
supplies one cross-section (or trailing history) at a time, so causality is
the caller's window discipline, same as the other indicator modules.
"""

from __future__ import annotations

import math

_SQRT_2 = math.sqrt(2.0)


def cross_sectional_share(value, total) -> float | None:
    """Return ``value / total`` as a cross-sectional share, or ``None`` if invalid.

    Both inputs must be finite and non-negative with ``total > 0``; the result
    is clamped to ``[0, 1]``.
    """
    value_f = float(value)
    total_f = float(total)
    if not (math.isfinite(value_f) and math.isfinite(total_f)):
        return None
    if value_f < 0.0 or total_f <= 0.0:
        return None
    return max(0.0, min(1.0, value_f / total_f))


def dense_rank_desc(values, target) -> int | None:
    """Return the 1-based dense rank of ``target`` among ``values`` (1 = largest).

    Mirrors pandas ``rank(ascending=False, method="dense")`` for one member of
    the cross-section.  Returns ``None`` on empty/non-finite input or a
    non-finite target.
    """
    target_f = float(target)
    series = [float(value) for value in list(values)]
    if not series or not math.isfinite(target_f):
        return None
    if not all(math.isfinite(value) for value in series):
        return None
    distinct_above = {value for value in series if value > target_f}
    return 1 + len(distinct_above)


def normal_cdf(z) -> float | None:
    """Return the standard normal CDF ``Phi(z)``, or ``None`` on non-finite input.

    This is the PUBLIC, ``None``-contract Gaussian CDF for the indicator layer.
    Private siblings with a bare-float contract (NaN/inf propagate) exist in
    ``rare_event._normal_cdf`` and ``strategy_factory.research_metrics._norm_cdf``
    (the latter is the research layer's documented source of truth, reused by
    ``research.options_math``); they are intentionally left untouched to keep
    their non-finite behaviour byte-identical.  New indicator-layer callers
    should use this function.
    """
    z_val = float(z)
    if not math.isfinite(z_val):
        return None
    return 0.5 * (1.0 + math.erf(z_val / _SQRT_2))


def cdf_extremeness(z) -> float | None:
    """Return ``2 * |Phi(z) - 0.5|`` in ``[0, 1]`` — how extreme a z-score is.

    ``0`` means the observation sits at its rolling median, ``1`` means an
    extreme tail reading in either direction; the Reference-Price repo uses
    this as a bounded anomaly gauge (``cdf_index``).  ``None`` on non-finite
    input.
    """
    cdf = normal_cdf(z)
    if cdf is None:
        return None
    return max(0.0, min(1.0, 2.0 * abs(cdf - 0.5)))


def share_weighted_return(shares, returns) -> float | None:
    """Return the share-weighted average return of a (top-N) cross-section.

    Implements one step of the Reference-Price top-N composite index: member
    returns weighted by their turnover shares renormalized within the basket
    (``sum(share_i * ret_i) / sum(share_i)``).  Returns ``None`` on length
    mismatch, empty/non-finite input, negative shares, or an all-zero share
    basket.
    """
    share_list = [float(value) for value in list(shares)]
    return_list = [float(value) for value in list(returns)]
    if not share_list or len(share_list) != len(return_list):
        return None
    if not all(math.isfinite(value) and value >= 0.0 for value in share_list):
        return None
    if not all(math.isfinite(value) for value in return_list):
        return None
    total = sum(share_list)
    if total <= 0.0:
        return None
    weighted = sum(share * ret for share, ret in zip(share_list, return_list, strict=False))
    result = weighted / total
    return result if math.isfinite(result) else None


__all__ = [
    "cdf_extremeness",
    "cross_sectional_share",
    "dense_rank_desc",
    "normal_cdf",
    "share_weighted_return",
]
