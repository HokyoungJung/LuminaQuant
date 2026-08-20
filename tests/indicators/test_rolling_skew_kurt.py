"""Deterministic tests for rolling skewness / excess-kurtosis indicators.

Parity-locks the canonical ``rolling_skewness`` against the two formerly
strategy-private ``_skewness`` copies: the golden values below were pinned
from BOTH pre-refactor implementations (``skew_innovation_alpha_sleeves`` and
``cross_sectional_anomaly_alpha_sleeves``), which agreed bit-exactly on every
fixture, and the post-refactor aliases are asserted to BE the canonical
function.  All randomness is a seeded inline LCG (never the ``random``
module); every assertion is bit-reproducible.
"""

from __future__ import annotations

import math

from lumina_quant.indicators.rolling_stats import rolling_excess_kurtosis, rolling_skewness
from lumina_quant.strategies.cross_sectional_anomaly_alpha_sleeves import (
    _skewness as anomaly_skewness,
)
from lumina_quant.strategies.skew_innovation_alpha_sleeves import (
    _skewness as innovation_skewness,
)

# --------------------------------------------------------------------------- #
# LCG (deterministic, no `random` module)
# --------------------------------------------------------------------------- #


def _lcg_stream(seed: int):
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# parity locks (goldens pinned from BOTH pre-refactor _skewness copies)
# --------------------------------------------------------------------------- #

_PARITY_FIXTURES = [
    ([0.0, 0.0, 0.0, 0.0, 10.0], 1.5),
    ([1.0, 2.0, 3.0, 4.0, 100.0], 1.4975367033335198),
    ([-0.03, 0.01, 0.02, -0.005, 0.04, -0.02, 0.015], -0.057943812424212995),
    ([0.5, -1.25, 2.75, 0.125, -0.375, 1.5], 0.3903810734299788),
]


def test_skewness_parity_goldens_bit_exact() -> None:
    for values, golden in _PARITY_FIXTURES:
        assert rolling_skewness(values) == golden


def test_sleeve_aliases_are_the_canonical_function() -> None:
    # skew_innovation's private alias IS the canonical indicator (its original
    # recipe was bit-exact to the fsum canonical form). cross_sectional_anomaly
    # deliberately KEEPS its own verbatim plain-sum copy (adversarial-review
    # W3): the fsum recipe drifts by one ULP on ~10% of windows, and a silent
    # last-ULP change to the registered LotterySkewness scoring path is the
    # class of default-numerics change this repo requires to be gated.
    assert innovation_skewness is rolling_skewness
    assert anomaly_skewness is not rolling_skewness


# --------------------------------------------------------------------------- #
# skewness correctness / guards
# --------------------------------------------------------------------------- #


def test_skewness_symmetric_input_is_zero() -> None:
    assert rolling_skewness([-2.0, -1.0, 0.0, 1.0, 2.0]) == 0.0


def test_skewness_sign_convention() -> None:
    right = rolling_skewness([0.0, 0.0, 0.0, 0.0, 10.0])
    left = rolling_skewness([0.0, -10.0, 0.0, 0.0, 0.0])
    assert right is not None and right > 0.0
    assert left is not None and left < 0.0


def test_skewness_degenerate_inputs_return_none() -> None:
    assert rolling_skewness([1.0, 2.0]) is None  # < 3 samples
    assert rolling_skewness([3.0, 3.0, 3.0, 3.0]) is None  # zero dispersion
    assert rolling_skewness([]) is None
    assert rolling_skewness([float("nan"), float("inf"), 1.0]) is None  # 1 finite


def test_skewness_drops_non_finite_samples() -> None:
    base = [1.0, 2.0, 3.0]
    assert rolling_skewness([*base, float("nan")]) == rolling_skewness(base)
    assert rolling_skewness([float("inf"), *base]) == rolling_skewness(base)


def test_skewness_window_semantics() -> None:
    values = [5.0, -1.0, 0.0, 1.0, 4.0, 9.0]
    assert rolling_skewness(values, window=3) == rolling_skewness(values[-3:])
    assert rolling_skewness(values, window=100) == rolling_skewness(values)
    assert rolling_skewness(values, window=0) is None
    assert rolling_skewness(values, window=-4) is None
    assert rolling_skewness(values, window="x") is None


def test_skewness_never_raises_on_garbage() -> None:
    assert rolling_skewness(None) is None
    assert rolling_skewness(42) is None
    assert rolling_skewness(object()) is None
    assert rolling_skewness(["a", "b", "c"]) is None
    assert rolling_skewness([1.0, "b", None, 2.0]) is None  # 2 finite -> None
    # Huge-but-finite values overflow float ** and must fail closed, not raise.
    assert rolling_skewness([1e200, -1e200, 0.0, 5.0]) is None


def test_skewness_two_run_determinism() -> None:
    stream = _lcg_stream(20260820)
    values = [(next(stream) - 0.5) * 0.2 for _ in range(64)]
    assert rolling_skewness(values) == rolling_skewness(list(values))


# --------------------------------------------------------------------------- #
# excess kurtosis
# --------------------------------------------------------------------------- #


def test_excess_kurtosis_hand_golden() -> None:
    # values [0,0,0,0,10]: mean 2, m2 = 80/5 = 16, m4 = 4160/5 = 832,
    # kurtosis = 832/256 - 3 = 0.25 exactly.
    assert rolling_excess_kurtosis([0.0, 0.0, 0.0, 0.0, 10.0]) == 0.25


def test_excess_kurtosis_uniform_sample_is_platykurtic() -> None:
    stream = _lcg_stream(7)
    values = [next(stream) for _ in range(4000)]
    kurt = rolling_excess_kurtosis(values)
    assert kurt is not None
    assert abs(kurt - (-1.2)) < 0.12  # uniform excess kurtosis = -6/5


def test_excess_kurtosis_normal_like_sample_is_near_zero() -> None:
    stream = _lcg_stream(99)
    # Irwin-Hall(12) - 6 approximates a standard normal (excess kurt -0.1).
    values = [sum(next(stream) for _ in range(12)) - 6.0 for _ in range(4000)]
    kurt = rolling_excess_kurtosis(values)
    assert kurt is not None
    assert abs(kurt) < 0.25


def test_excess_kurtosis_fat_tails_are_positive() -> None:
    values = [0.1, -0.1] * 30 + [8.0]
    kurt = rolling_excess_kurtosis(values)
    assert kurt is not None
    assert kurt > 5.0


def test_excess_kurtosis_degenerate_inputs_return_none() -> None:
    assert rolling_excess_kurtosis([1.0, 2.0, 3.0]) is None  # < 4 samples
    assert rolling_excess_kurtosis([2.0] * 10) is None  # zero dispersion
    assert rolling_excess_kurtosis([]) is None
    assert rolling_excess_kurtosis(None) is None
    assert rolling_excess_kurtosis(object()) is None
    assert rolling_excess_kurtosis([1e200, -1e200, 0.0, 5.0]) is None


def test_excess_kurtosis_window_semantics_and_nan_filter() -> None:
    values = [9.0, 1.0, 2.0, 3.0, 4.0, 50.0]
    assert rolling_excess_kurtosis(values, window=5) == rolling_excess_kurtosis(values[-5:])
    with_nan = [1.0, float("nan"), 2.0, 3.0, 4.0, 50.0]
    assert rolling_excess_kurtosis(with_nan) == rolling_excess_kurtosis([1.0, 2.0, 3.0, 4.0, 50.0])
    assert math.isfinite(rolling_excess_kurtosis(values))
