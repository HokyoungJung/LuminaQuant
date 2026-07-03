"""Deterministic tests for the Hill (1975) tail-index indicator primitives.

Covers :func:`hill_tail_index` recovering a known Pareto tail exponent from a
synthetic sample generated deterministically via inverse-CDF sampling on a
hand-rolled LCG (bit-for-bit reproducible); its guards (short input, ``k``
too big for the sample, an empty loss-tail sample, non-finite input); and
:func:`tail_index_ratio` correctly flagging a fattening vs. thinning tail
regime plus its own guards.  No backtest is run.
"""

from __future__ import annotations

import math

from lumina_quant.indicators.tail_index import hill_tail_index, tail_index_ratio


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield state / 2**31


def _pareto_sample(seed: int, n: int, alpha: float) -> list[float]:
    """Deterministic Pareto(alpha) sample via inverse-CDF on LCG uniforms.

    ``x = (1 - u) ** (-1 / alpha)`` for ``u ~ Uniform(0, 1)`` is the standard
    Pareto(1, alpha) inverse-CDF (minimum value 1).
    """
    gen = _lcg(seed)
    out = []
    for _ in range(n):
        u = min(next(gen), 0.999999999)
        out.append((1.0 - u) ** (-1.0 / alpha))
    return out


# --------------------------------------------------------------------------- #
# hill_tail_index
# --------------------------------------------------------------------------- #
def test_hill_tail_index_recovers_known_pareto_alpha() -> None:
    alpha = 2.5
    sample = _pareto_sample(seed=7, n=2000, alpha=alpha)
    estimate = hill_tail_index(sample, k=200)
    assert estimate is not None
    assert abs(estimate - alpha) / alpha < 0.15, estimate


def test_hill_tail_index_recovers_different_alpha() -> None:
    alpha = 4.0
    sample = _pareto_sample(seed=23, n=2500, alpha=alpha)
    estimate = hill_tail_index(sample, k=250)
    assert estimate is not None
    assert abs(estimate - alpha) / alpha < 0.15, estimate


def test_hill_tail_index_fatter_tail_gives_smaller_alpha() -> None:
    # Smaller alpha == fatter (heavier) tail; a lower-alpha Pareto sample must
    # recover a smaller Hill estimate than a higher-alpha one from the same
    # generator family and sample size.
    fat = _pareto_sample(seed=5, n=2000, alpha=1.5)
    thin = _pareto_sample(seed=5, n=2000, alpha=5.0)
    hill_fat = hill_tail_index(fat, k=200)
    hill_thin = hill_tail_index(thin, k=200)
    assert hill_fat is not None and hill_thin is not None
    assert hill_fat < hill_thin, (hill_fat, hill_thin)


def test_hill_tail_index_guards_short_input() -> None:
    # Fewer than k+1 values.
    assert hill_tail_index([5.0, 4.0, 3.0], k=3) is None


def test_hill_tail_index_guards_k_too_small() -> None:
    assert hill_tail_index([5.0, 4.0, 3.0, 2.0, 1.0], k=1) is None
    assert hill_tail_index([5.0, 4.0, 3.0, 2.0, 1.0], k=0) is None
    assert hill_tail_index([5.0, 4.0, 3.0, 2.0, 1.0], k=-2) is None


def test_hill_tail_index_guards_empty_loss_tail() -> None:
    # An all-positive-return tape has no losses: the caller passes an empty
    # sample, which must be a graceful None (never raise, never divide by zero).
    assert hill_tail_index([], k=5) is None


def test_hill_tail_index_guards_non_finite() -> None:
    assert hill_tail_index([float("nan")] * 20, k=5) is None
    assert hill_tail_index([float("inf")] * 20, k=5) is None
    # Mixed finite/non-finite: non-finite entries are dropped, finite ones remain.
    mixed = [float("nan"), 10.0, 9.0, 8.0, 7.0, float("inf"), -1.0, 6.0, 5.0]
    result = hill_tail_index(mixed, k=3)
    assert result is not None and math.isfinite(result)


def test_hill_tail_index_rejects_non_positive_values() -> None:
    # Zeros/negatives are dropped by the positivity filter; too few remain.
    assert hill_tail_index([0.0, -1.0, -2.0, 5.0, 4.0], k=3) is None


# --------------------------------------------------------------------------- #
# tail_index_ratio
# --------------------------------------------------------------------------- #
def test_tail_index_ratio_flags_fattening_regime() -> None:
    baseline = _pareto_sample(seed=11, n=2000, alpha=3.0)
    recent_fat = _pareto_sample(seed=13, n=500, alpha=1.5)
    ratio = tail_index_ratio(recent_fat, baseline, k_short=100, k_long=200)
    assert ratio is not None
    assert ratio < 1.0, ratio


def test_tail_index_ratio_flags_thinning_regime() -> None:
    baseline = _pareto_sample(seed=11, n=2000, alpha=3.0)
    recent_thin = _pareto_sample(seed=17, n=500, alpha=5.0)
    ratio = tail_index_ratio(recent_thin, baseline, k_short=100, k_long=200)
    assert ratio is not None
    assert ratio > 1.0, ratio


def test_tail_index_ratio_guards() -> None:
    baseline = _pareto_sample(seed=11, n=2000, alpha=3.0)
    assert tail_index_ratio([], baseline, k_short=100, k_long=200) is None
    assert tail_index_ratio(baseline, [], k_short=100, k_long=200) is None
    assert tail_index_ratio([], [], k_short=2, k_long=2) is None
    assert tail_index_ratio(baseline, baseline, k_short=1, k_long=200) is None
