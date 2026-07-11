"""Pure volatility-annualization helpers (canonical home).

The alpha-pool sizing-discipline audit (2026-07-10) found a systematic
horizon mismatch across sleeve vol targets: an annualized-scale
``target_vol`` (e.g. ``0.20``) compared against a PER-BAR realized-vol
estimate pins the de-risk scalar at its clamp and the throttle never
engages.  The fix is to annualize the per-bar estimate before the
comparison, with the bar cadence inferred deterministically from the
observed bar spacing (median of recent epoch deltas) so a sleeve needs no
timeframe parameter and behaves correctly on 1h/4h/1d feeds alike.

Every function is pure, deterministic and never raises: degenerate input
(fewer than two usable timestamps, non-positive spacing, non-finite vol)
propagates ``None`` so callers can fall back to a conservative unity
scalar (never inflating leverage).

``_SECONDS_PER_YEAR`` uses the 365.25-day year the first landed fix
(``external_alpha_sleeves``) verified against; the ~0.07% delta vs a
365-day year is far below any decision threshold in the sleeves.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

_SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0

__all__ = [
    "annualize_per_bar_vol",
    "bars_per_year_from_spacing",
    "median_bar_spacing_seconds",
]


def median_bar_spacing_seconds(times: Any) -> float | None:
    """Median positive spacing (seconds) between consecutive distinct epochs.

    ``times`` is any iterable of epoch-seconds-like values; non-finite and
    non-numeric entries are skipped.  Returns ``None`` when fewer than two
    usable timestamps or no positive spacing exists.
    """
    epochs: list[float] = []
    try:
        for raw in times:
            try:
                value = float(raw)
            except TypeError, ValueError:
                continue
            if math.isfinite(value):
                epochs.append(value)
    except TypeError:
        return None
    if len(epochs) < 2:
        return None
    epochs.sort()
    deltas = [b - a for a, b in pairwise(epochs) if b - a > 0.0]
    if not deltas:
        return None
    deltas.sort()
    mid = len(deltas) // 2
    if len(deltas) % 2:
        return deltas[mid]
    return 0.5 * (deltas[mid - 1] + deltas[mid])


def bars_per_year_from_spacing(times: Any) -> float | None:
    """Bars-per-year inferred from observed bar spacing (365.25-day year)."""
    spacing = median_bar_spacing_seconds(times)
    if spacing is None or spacing <= 0.0:
        return None
    return _SECONDS_PER_YEAR / spacing


def annualize_per_bar_vol(per_bar_vol: float, bars_per_year: float | None) -> float | None:
    """``per_bar_vol * sqrt(bars_per_year)``; ``None``-propagating, never raises."""
    if bars_per_year is None:
        return None
    try:
        vol = float(per_bar_vol)
        bpy = float(bars_per_year)
    except TypeError, ValueError:
        return None
    if not math.isfinite(vol) or not math.isfinite(bpy) or bpy <= 0.0 or vol < 0.0:
        return None
    return vol * math.sqrt(bpy)
