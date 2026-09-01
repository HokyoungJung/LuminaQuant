"""Deterministic forecast/signal ensemble-combination primitives (pure Python).

Method absorbed from the DeepLearning repo's online ensemble engine
(``ensemble_strategies/components/weights.py`` and the oracle/hybrid model
variants): inverse-error weighting ``w_i = (1/(err_i + eps)) / sum``,
temperature softmax over scores, a prior/adaptive weight blend, the
disagreement coefficient (dispersion gate) ``std / |mean|`` used by
``run_disagreement_switching`` to fall back to equal weights when models
agree, and the rolling direction hit-rate quality gate from ``metric/``.

These are stateless building blocks for combining multiple sleeve signals or
bridge forecasts; the caller owns the rolling windows, so causality is the
caller's window discipline.
"""

from __future__ import annotations

import math


def _finite_list(values) -> list[float] | None:
    """Return ``values`` as floats, or ``None`` when empty/non-finite."""
    series = [float(value) for value in list(values)]
    if not series or not all(math.isfinite(value) for value in series):
        return None
    return series


def inverse_error_weights(errors, *, epsilon: float = 1e-8) -> list[float] | None:
    """Return weights proportional to ``1 / (error + epsilon)``, summing to 1.

    Errors must be finite and non-negative (e.g. rolling MAPE/MAE per model).
    Returns ``None`` on empty/invalid input.
    """
    series = _finite_list(errors)
    eps = max(1e-12, float(epsilon))
    if series is None or any(value < 0.0 for value in series):
        return None
    inverses = [1.0 / (value + eps) for value in series]
    total = sum(inverses)
    if total <= 0.0 or not math.isfinite(total):
        return None
    return [value / total for value in inverses]


def softmax_weights(scores, *, temperature: float = 1.0) -> list[float] | None:
    """Return a numerically stable softmax over ``scores`` (higher = heavier).

    Returns ``None`` on empty/non-finite input or a non-positive temperature.
    """
    series = _finite_list(scores)
    temp = float(temperature)
    if series is None or temp <= 0.0 or not math.isfinite(temp):
        return None
    peak = max(series)
    exps = [math.exp((value - peak) / temp) for value in series]
    total = sum(exps)
    if total <= 0.0 or not math.isfinite(total):
        return None
    return [value / total for value in exps]


def blend_weights(primary, secondary, *, ratio) -> list[float] | None:
    """Return ``ratio * primary + (1 - ratio) * secondary``, renormalized to 1.

    The DeepLearning hybrid models blend a prior (default-model) weight vector
    with adaptive inverse-error weights this way.  Both vectors must be the
    same length with finite non-negative entries and ``ratio`` in ``[0, 1]``;
    returns ``None`` otherwise or when the blend sums to zero.
    """
    ratio_val = float(ratio)
    if not math.isfinite(ratio_val) or ratio_val < 0.0 or ratio_val > 1.0:
        return None
    primary_list = _finite_list(primary)
    secondary_list = _finite_list(secondary)
    if primary_list is None or secondary_list is None:
        return None
    if len(primary_list) != len(secondary_list):
        return None
    if any(value < 0.0 for value in primary_list + secondary_list):
        return None
    blended = [
        ratio_val * left + (1.0 - ratio_val) * right
        for left, right in zip(primary_list, secondary_list, strict=False)
    ]
    total = sum(blended)
    if total <= 0.0 or not math.isfinite(total):
        return None
    return [value / total for value in blended]


def disagreement_coefficient(values) -> float | None:
    """Return the dispersion gate ``std(values) / |mean(values)|`` (ddof=1).

    Low output means the ensemble members agree (the DeepLearning
    disagreement-switching rule trusts consensus below ~0.02); high output
    means they diverge.  Returns ``None`` on fewer than two finite values or a
    near-zero mean (dispersion is undefined relative to no consensus level).
    """
    series = _finite_list(values)
    if series is None or len(series) < 2:
        return None
    center = sum(series) / float(len(series))
    if abs(center) <= 1e-12:
        return None
    # d * d (not d ** 2): float ** raises OverflowError on huge-but-finite
    # inputs, while multiplication overflows to inf and hits the guard below.
    variance = 0.0
    for value in series:
        delta = value - center
        variance += delta * delta
    variance /= float(len(series) - 1)
    if variance < 0.0 or not math.isfinite(variance):
        return None
    result = math.sqrt(variance) / abs(center)
    return result if math.isfinite(result) else None


def direction_hit_rate(predicted, realized, *, deadband: float = 0.0) -> float | None:
    """Return the sign-agreement rate between predicted and realized returns.

    Pairs where ``|realized| <= deadband`` are excluded from scoring (the
    DeepLearning ``metric/`` module's epsilon deadband), so the rate reflects
    only decidable outcomes.  Returns ``None`` on length mismatch, empty or
    non-finite input, or when no pair survives the deadband.
    """
    predicted_list = _finite_list(predicted)
    realized_list = _finite_list(realized)
    band = float(deadband)
    if predicted_list is None or realized_list is None or band < 0.0:
        return None
    if len(predicted_list) != len(realized_list):
        return None
    hits = 0
    scored = 0
    for pred, real in zip(predicted_list, realized_list, strict=False):
        if abs(real) <= band:
            continue
        scored += 1
        if (pred > 0.0 and real > 0.0) or (pred < 0.0 and real < 0.0):
            hits += 1
    if scored == 0:
        return None
    return hits / float(scored)


__all__ = [
    "blend_weights",
    "direction_hit_rate",
    "disagreement_coefficient",
    "inverse_error_weights",
    "softmax_weights",
]
