"""Trailing-window periodogram cycle extraction (pure Python, no FFT library).

Method absorbed from the precious-metal research repo's ``rfft_periodogram.py``
(demean -> real FFT -> amplitude spectrum -> dominant period = argmax
amplitude), re-implemented causally: the caller passes a TRAILING window so no
full-sample lookahead is possible, the series is linearly detrended (a trend
otherwise leaks power into the longest scanned period), and the discrete
Fourier projection is evaluated directly per candidate period so the module
needs neither numpy nor scipy.

For a candidate period ``P`` (in bars) with angular frequency ``w = 2*pi/P``,
the complex projection over the ``N``-bar window is ``c = sum_t x_t *
exp(-i*w*t)``; for ``x_t ~ A*cos(w*t + phi)`` this gives amplitude
``A ~ 2*|c|/N`` and phase ``phi = arg(c)``, so the cycle position at the
LATEST bar is ``psi = (w*(N-1) + phi) mod 2*pi`` (``0`` = crest, ``pi`` =
trough, rising while ``pi < psi < 2*pi``).
"""

from __future__ import annotations

import math

_TWO_PI = 2.0 * math.pi


def _linear_detrend(values: list[float]) -> list[float] | None:
    """Return ``values`` minus its least-squares line, or ``None`` if degenerate."""
    count = len(values)
    if count < 2:
        return None
    mean_t = (count - 1) / 2.0
    mean_x = sum(values) / float(count)
    var_t = sum((t - mean_t) ** 2 for t in range(count))
    if var_t <= 0.0:
        return None
    cov_tx = sum((t - mean_t) * (values[t] - mean_x) for t in range(count))
    slope = cov_tx / var_t
    intercept = mean_x - slope * mean_t
    detrended = [values[t] - (intercept + slope * t) for t in range(count)]
    if not all(math.isfinite(value) for value in detrended):
        return None
    return detrended


def dominant_cycle(
    values, *, min_period: int, max_period: int
) -> tuple[float, float, float, float] | None:
    """Return ``(period, amplitude, phase, purity)`` of the strongest trailing cycle.

    Scans integer periods ``min_period..min(max_period, N // 2)`` (so at least
    two full cycles fit in the window) over the linearly detrended input and
    picks the period with the largest amplitude ``2*|c|/N`` (deterministic
    tie-break toward the SHORTER period).  ``phase`` is the cycle position at
    the latest bar in ``[0, 2*pi)`` — ``0`` = crest, ``pi`` = trough, rising
    while ``pi < phase < 2*pi``.  ``purity`` is the dominant period's share of
    the total scanned squared amplitude in ``(0, 1]``.  Returns ``None`` on
    insufficient/non-finite history, a degenerate (flat) window, or an empty
    scan range.

    Args:
        values: Trailing series (oldest first), e.g. log closes.
        min_period: Shortest candidate cycle length in bars (``>= 2``).
        max_period: Longest candidate cycle length in bars.

    Returns:
        ``(period, amplitude, phase, purity)`` or ``None`` when unavailable.
    """
    min_p = max(2, int(min_period))
    max_p = int(max_period)
    series = [float(value) for value in list(values)]
    count = len(series)
    if count < 2 * min_p or not all(math.isfinite(value) for value in series):
        return None
    top_p = min(max_p, count // 2)
    if top_p < min_p:
        return None
    detrended = _linear_detrend(series)
    if detrended is None:
        return None

    best_period = 0
    best_amp_sq = 0.0
    best_real = 0.0
    best_imag = 0.0
    total_amp_sq = 0.0
    for period in range(min_p, top_p + 1):
        omega = _TWO_PI / float(period)
        real = 0.0
        imag = 0.0
        for t, value in enumerate(detrended):
            angle = omega * t
            real += value * math.cos(angle)
            imag -= value * math.sin(angle)
        amplitude = 2.0 * math.hypot(real, imag) / float(count)
        amp_sq = amplitude * amplitude
        total_amp_sq += amp_sq
        if amp_sq > best_amp_sq:
            best_amp_sq = amp_sq
            best_period = period
            best_real = real
            best_imag = imag
    if best_period <= 0 or total_amp_sq <= 0.0:
        return None
    amplitude = math.sqrt(best_amp_sq)
    phase = (
        _TWO_PI / float(best_period) * float(count - 1) + math.atan2(best_imag, best_real)
    ) % _TWO_PI
    purity = best_amp_sq / total_amp_sq
    if not (math.isfinite(amplitude) and math.isfinite(phase) and math.isfinite(purity)):
        return None
    return (float(best_period), amplitude, phase, min(1.0, purity))


def cycle_phase_fraction(phase) -> float | None:
    """Return the cycle phase normalized to ``[0, 1)`` (``0`` crest, ``0.5`` trough)."""
    parsed = float(phase)
    if not math.isfinite(parsed):
        return None
    return (parsed % _TWO_PI) / _TWO_PI


__all__ = [
    "cycle_phase_fraction",
    "dominant_cycle",
]
