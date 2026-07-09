"""Da-Gurun-Warachka frog-in-the-pan information discreteness numeric.

``information_discreteness`` is a magnitude-BLIND sign census of a symbol's
formation-window daily returns -- Da, Gurun & Warachka (2014, RFS) "Frog in the
Pan"::

    ID = sign(PRET) * (pct_neg - pct_pos)

over a ``formation_bars`` daily-return window that EXCLUDES the most recent
``skip_bars`` bars (the skip-week), where ``pct_pos`` / ``pct_neg`` are the
fractions of formation days whose return is above ``+zero_eps`` / below
``-zero_eps`` (flat days dilute BOTH), and ``PRET`` is the formation window's
cumulative return (excluding the skip bars).

Interpretation:

- CONTINUOUS information (a smooth grind: many small same-direction days) ->
  one side of the sign census dominates -> ``pct_neg - pct_pos`` is large in
  magnitude with the opposite sign of ``PRET`` -> ID strongly NEGATIVE.
- DISCRETE / jump-driven information (a flat drift punctuated by a single large
  jump that sets ``PRET``'s sign while the many small days lean the other way)
  -> ID near ``0`` or POSITIVE.

The statistic is deliberately magnitude-blind: it counts the SIGN of each day,
never the size, which is exactly what separates it from path-length efficiency
(Kaufman ER), fit quality (signed-R^2), or plain momentum/vol (magnitude only).

Pure Python (``math`` only): deterministic, no I/O, never raises; ``None`` on
short history or a degenerate (flat / non-finite) window.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_ZERO_EPS = 1e-6


def _sign(value: float, eps: float) -> int:
    """Return ``+1`` / ``-1`` / ``0`` for a value against a dead-band ``eps``."""
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def information_discreteness(
    closes: Sequence[float] | object,
    *,
    formation_bars: int = 56,
    skip_bars: int = 7,
    zero_eps: float = _ZERO_EPS,
) -> float | None:
    """Return the frog-in-the-pan information-discreteness score, or ``None``.

    ``closes`` is a trailing close series.  The formation window is the
    ``formation_bars`` daily returns ending ``skip_bars`` bars before the last
    close.  Returns ``None`` when the history is too short for the window or the
    window has no directional content (``PRET`` inside the dead band).
    """
    form = max(1, int(formation_bars))
    skip = max(0, int(skip_bars))
    eps = abs(float(zero_eps))

    if not isinstance(closes, (list, tuple)):
        try:
            closes = list(closes)  # type: ignore[arg-type]
        except Exception:
            return None
    prices: list[float] = []
    for value in closes:
        try:
            parsed = float(value)
        except TypeError, ValueError:
            return None
        if not math.isfinite(parsed) or parsed <= 0.0:
            return None
        prices.append(parsed)

    # Drop the skip window, then require formation_bars + 1 closes for the
    # formation-window returns.
    work = prices[: len(prices) - skip] if skip > 0 else prices
    if len(work) < form + 1:
        return None
    window = work[-(form + 1) :]

    pos = 0
    neg = 0
    for idx in range(1, len(window)):
        prev = window[idx - 1]
        curr = window[idx]
        if prev <= 0.0:
            return None
        ret = (curr / prev) - 1.0
        sign = _sign(ret, eps)
        if sign > 0:
            pos += 1
        elif sign < 0:
            neg += 1

    pret = (window[-1] / window[0]) - 1.0
    pret_sign = _sign(pret, eps)
    if pret_sign == 0:
        return None

    total = float(form)
    pct_pos = pos / total
    pct_neg = neg / total
    score = float(pret_sign) * (pct_neg - pct_pos)
    if not math.isfinite(score):
        return None
    return float(max(-1.0, min(1.0, score)))


__all__ = ["information_discreteness"]
