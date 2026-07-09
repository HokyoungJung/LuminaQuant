"""Grinblatt-Han (2005) turnover-decay reference price and capital-gains overhang.

Pure-Python, dependency-free helpers for the disposition / capital-gains-overhang
family.  The Grinblatt-Han reference price ``R`` estimates the price at which the
CURRENT holder cohort acquired its inventory: recent high-volume bars claim
ownership, and older prices decay geometrically as inventory turns over.

``grinblatt_han_reference_price`` is deliberately NOT a rolling VWAP.  A rolling
VWAP ``sum(P*V)/sum(V)`` is invariant to the time ORDERING of the (price,
volume) pairs; the Grinblatt-Han recursion is order-sensitive because it
survival-discounts older prices by the turnover of every more-recent bar.  Two
price/volume paths with identical rolling VWAP but different volume ordering
therefore yield different reference prices -- the property that separates the
capital-gains-overhang axis from a VWAP-reversion alias.

Both helpers are ``None``-propagating, deterministic (``math``-module arithmetic
only), and never raise on degenerate input (empty / short / non-positive volume /
non-finite / ``close <= 0``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

_EPS = 1e-12
_DEFAULT_V_CLIP_MAX = 0.999


def _finite(value: Any) -> float | None:
    """Coerce to a finite float or ``None`` (never raises)."""
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def grinblatt_han_reference_price(
    prices: Sequence[float] | Iterable[float],
    volumes: Sequence[float] | Iterable[float],
    window: int,
    *,
    skip_recent: int = 1,
    v_clip_max: float = _DEFAULT_V_CLIP_MAX,
) -> float | None:
    """Return the Grinblatt-Han turnover-decay reference price ``R``.

    For the trailing ``window`` bars (after excluding the most recent
    ``skip_recent`` bars), each bar's turnover fraction is
    ``v_n = clip(volume_n / total_window_volume, 0, v_clip_max)``.  The
    survival weight of a bar is its turnover times the probability that the
    inventory bought there has NOT been turned over by any more-recent bar::

        w_{t-n} = v_{t-n} * prod_{tau=1..n-1} (1 - v_{t-n+tau})

    computed here newest-to-oldest via a running ``(1 - v)`` survival product.
    ``R = sum(w * price) / sum(w)``.

    Returns ``None`` when fewer than 2 usable bars remain, the window volume is
    non-positive, or any used input is non-finite.
    """
    win = int(window)
    if win < 2:
        return None
    price_list = list(prices)
    volume_list = list(volumes)
    aligned = min(len(price_list), len(volume_list))
    if aligned < 2:
        return None
    skip = max(0, int(skip_recent))
    end = aligned - skip  # exclusive index of the last bar folded into R
    if end < 2:
        return None
    start = max(0, end - win)
    used_prices: list[float] = []
    used_volumes: list[float] = []
    for idx in range(start, end):
        price = _finite(price_list[idx])
        volume = _finite(volume_list[idx])
        if price is None or volume is None:
            return None
        used_prices.append(price)
        used_volumes.append(max(0.0, volume))
    if len(used_prices) < 2:
        return None
    total_volume = math.fsum(used_volumes)
    if total_volume <= _EPS:
        return None
    clip = _finite(v_clip_max)
    clip = _DEFAULT_V_CLIP_MAX if clip is None else min(max(clip, 0.0), 0.999999)
    turnovers = [min(clip, max(0.0, volume / total_volume)) for volume in used_volumes]
    # Survival weights: the newest bar keeps its full turnover (empty product);
    # each older bar is discounted by (1 - v) of every more-recent bar.
    weights = [0.0] * len(used_prices)
    survival = 1.0
    for idx in range(len(used_prices) - 1, -1, -1):
        weights[idx] = turnovers[idx] * survival
        survival *= 1.0 - turnovers[idx]
    weight_sum = math.fsum(weights)
    if weight_sum <= _EPS:
        return None
    weighted_price = math.fsum(
        price * weight for price, weight in zip(used_prices, weights, strict=True)
    )
    reference = weighted_price / weight_sum
    return float(reference) if math.isfinite(reference) else None


def capital_gains_overhang(close: Any, ref: Any) -> float | None:
    """Return the unrealized capital-gains overhang ``(close - ref) / close``.

    Positive overhang => price is above the current cohort's cost basis
    (holders sitting on gains); negative => underwater holders.  ``None`` when
    ``close <= 0`` or either input is non-finite / ``None``.
    """
    current = _finite(close)
    if current is None or current <= 0.0:
        return None
    reference = _finite(ref)
    if reference is None:
        return None
    return float((current - reference) / current)


__all__ = ["capital_gains_overhang", "grinblatt_han_reference_price"]
