"""Deterministic contract tests for the behavioral-value indicators.

Covers ``salience_theory_value`` (BGS 2012 salience distortion) and
``prospect_theory_value`` (TK-1992 sorted-histogram valuation): hand-computed
values, the co-timing permutation lock (the mechanical proof that ST is a
co-timing functional while MAX/skew are alignment-invariant), sign properties
implied by loss aversion, the latest-scalar ``float | None`` never-raise
contract, and two-call determinism.  Data-free and seed-free (fixed vectors).
"""

from __future__ import annotations

import math

from lumina_quant.indicators.behavioral_value import (
    prospect_theory_value,
    salience_theory_value,
)

_CANONICAL_ST = {"theta": 0.1, "delta": 0.7}
_CANONICAL_PT = {"alpha": 0.88, "lam": 2.25, "gamma_gain": 0.61, "gamma_loss": 0.69}


def _skewness(values: list[float]) -> float:
    """Population Fisher-Pearson skewness (test-local reference implementation)."""
    n = len(values)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    std = var**0.5
    return sum((v - mu) ** 3 for v in values) / n / std**3


def _tk_weight(p: float, gamma: float) -> float:
    return p**gamma / (p**gamma + (1.0 - p) ** gamma) ** (1.0 / gamma)


# --------------------------------------------------------------------------- #
# salience_theory_value
# --------------------------------------------------------------------------- #


def test_salience_hand_computed_three_days() -> None:
    returns = [0.10, -0.02, 0.01]
    market = [0.0, 0.0, 0.0]
    # sigma = |r| / (|r| + 0.1) -> already sorted descending -> ranks 0,1,2.
    weights = [1.0, 0.7, 0.49]
    total = sum(weights)
    expected = (weights[0] * 0.10 + weights[1] * -0.02 + weights[2] * 0.01) / total - sum(
        returns
    ) / 3.0
    got = salience_theory_value(returns, market)
    assert got is not None and abs(got - expected) < 1e-12


def test_salience_uniform_delta_is_zero_distortion() -> None:
    returns = [0.03, -0.05, 0.02, 0.01, -0.04]
    market = [0.0] * 5
    got = salience_theory_value(returns, market, delta=1.0)
    assert got is not None and abs(got) < 1e-15


def test_salience_co_timing_permutation_lock() -> None:
    """Permuting the MARKET-day alignment moves ST; MAX/skew are frozen.

    The own-return vector is bit-identical between the two calls, so any
    alignment-invariant functional of it (MAX, skewness, momentum) is frozen
    by construction; only the co-timing input changed.
    """
    returns = [0.001 * (-1.0) ** i for i in range(18)] + [0.05, -0.002, 0.003]
    market = [0.0] * 18 + [0.05, 0.0, 0.0]  # the big own-day is a market day
    permuted = [0.05] + [0.0] * 20  # market spike moved to a different day
    frozen_max = max(returns)
    frozen_skew = _skewness(returns)
    st_aligned = salience_theory_value(returns, market)
    st_permuted = salience_theory_value(returns, permuted)
    assert st_aligned is not None and st_permuted is not None
    assert abs(st_aligned - st_permuted) > 1e-6
    # Alignment-invariant lottery controls are bit-identical after both calls.
    assert max(returns) == frozen_max and _skewness(returns) == frozen_skew


def test_salience_tail_alignment_and_pairwise_nan_drop() -> None:
    base = salience_theory_value([0.02, -0.01, 0.03], [0.0, 0.0, 0.0])
    # Longer market series is tail-trimmed to the common length.
    longer = salience_theory_value([0.02, -0.01, 0.03], [9.9, 0.0, 0.0, 0.0])
    assert base is not None and longer is not None and abs(base - longer) < 1e-15
    # A non-finite pair is dropped pairwise, not fatal.
    with_nan = salience_theory_value([0.02, float("nan"), -0.01, 0.03], [0.0, 0.0, 0.0, 0.0])
    assert with_nan is not None


def test_salience_never_raises_on_degenerate() -> None:
    cases = (
        (None, None),
        ([], []),
        ([0.1], [0.0]),
        ("abc", "abc"),
        ([float("nan")] * 5, [0.0] * 5),
        ([[1.0, 2.0]], [[3.0, 4.0]]),
        ([0.1, 0.2], None),
        ({"a": 1}, [0.0, 0.0]),
    )
    for returns, market in cases:
        assert salience_theory_value(returns, market) is None
    # Explicitly invalid parameters fail closed rather than being defaulted or
    # clamped into a different model.
    returns = [0.02, -0.03, 0.01, 0.05, -0.01]
    market = [0.01, -0.01, 0.0, 0.02, 0.0]
    for theta, delta in (
        (-1.0, 0.7),
        (0.1, 0.0),
        (float("nan"), 0.7),
        (None, None),
        ("abc", "abc"),
    ):
        assert salience_theory_value(returns, market, theta=theta, delta=delta) is None


# --------------------------------------------------------------------------- #
# prospect_theory_value
# --------------------------------------------------------------------------- #


def test_prospect_hand_computed_two_outcomes() -> None:
    # Sorted outcomes [-0.1, 0.2], equal 1/2 probabilities:
    # pi_loss = w-(1/2) - w-(0) ; pi_gain = w+(1/2) - w+(0).
    expected = _tk_weight(0.5, 0.69) * (-2.25 * 0.1**0.88) + _tk_weight(0.5, 0.61) * (0.2**0.88)
    got = prospect_theory_value([0.2, -0.1])
    assert got is not None and abs(got - expected) < 1e-12


def test_prospect_sign_properties() -> None:
    all_gains = prospect_theory_value([0.01] * 10 + [0.05] * 10)
    all_losses = prospect_theory_value([-0.01] * 10 + [-0.05] * 10)
    symmetric = prospect_theory_value([0.05, -0.05] * 45)
    assert all_gains is not None and all_gains > 0.0
    assert all_losses is not None and all_losses < 0.0
    # Loss aversion (lam=2.25): a symmetric gamble has strictly negative value.
    assert symmetric is not None and symmetric < 0.0


def test_prospect_jackpot_monotonicity() -> None:
    base = [0.001, -0.001] * 20
    with_jackpot = [*base, 0.15]
    without = [*base, 0.001]
    tk_jackpot = prospect_theory_value(with_jackpot)
    tk_plain = prospect_theory_value(without)
    assert tk_jackpot is not None and tk_plain is not None
    assert tk_jackpot > tk_plain


def test_prospect_never_raises_on_degenerate() -> None:
    for bad in (None, [], [0.1], "abc", [float("nan")] * 5, [[1.0], [2.0]], {"a": 1}):
        assert prospect_theory_value(bad) is None
    assert prospect_theory_value([float("inf"), 0.1, -0.1]) is not None  # inf dropped
    returns = [0.02, -0.03, 0.01, 0.05, -0.01]
    for params in (
        {"alpha": -2.0},
        {"lam": -5.0},
        {"gamma_gain": 0.0, "gamma_loss": 100.0},
        {"alpha": None, "lam": None, "gamma_gain": None, "gamma_loss": None},
        {"alpha": "abc", "lam": "abc", "gamma_gain": "abc", "gamma_loss": "abc"},
    ):
        assert prospect_theory_value(returns, **params) is None


# --------------------------------------------------------------------------- #
# determinism / canonical defaults
# --------------------------------------------------------------------------- #


def test_two_call_determinism() -> None:
    returns = [math.sin(0.7 * i) * 0.02 for i in range(60)]
    market = [math.cos(0.3 * i) * 0.01 for i in range(60)]
    assert salience_theory_value(returns, market) == salience_theory_value(returns, market)
    assert prospect_theory_value(returns) == prospect_theory_value(returns)


def test_canonical_constants_are_the_defaults() -> None:
    returns = [0.02, -0.03, 0.01, 0.05, -0.01, -0.04, 0.03]
    market = [0.01, -0.01, 0.0, 0.02, 0.0, -0.02, 0.01]
    assert salience_theory_value(returns, market) == salience_theory_value(
        returns, market, **_CANONICAL_ST
    )
    assert prospect_theory_value(returns) == prospect_theory_value(returns, **_CANONICAL_PT)
