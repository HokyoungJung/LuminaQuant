"""Deterministic tests for the VPIN / Bulk Volume Classification primitives.

Covers :func:`bulk_volume_buy_fraction` against a hand-computed ``erf``
reference and its guards (non-positive/non-finite sigma, non-finite/
non-numeric price_change); :func:`vpin_from_buckets` mean-absolute-imbalance
math and its guards (empty/degenerate input); and
:func:`accumulate_volume_bucket` single- and multi-bucket-per-call closing
math, carry-across-calls behaviour, and its never-raise guards (non-positive
bucket size, negative/non-finite volume, adversarial ``state`` payloads).  No
backtest is run.
"""

from __future__ import annotations

import math

from lumina_quant.indicators.vpin import (
    accumulate_volume_bucket,
    bulk_volume_buy_fraction,
    vpin_from_buckets,
)


# --------------------------------------------------------------------------- #
# bulk_volume_buy_fraction
# --------------------------------------------------------------------------- #
def test_bulk_volume_buy_fraction_neutral_at_zero() -> None:
    assert bulk_volume_buy_fraction(0.0, 1.0) == 0.5


def test_bulk_volume_buy_fraction_matches_erf_reference() -> None:
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2))), computed independently here as the
    # ground truth for a known dP/sigma ratio.
    for dp, sigma in ((1.0, 1.0), (-2.0, 4.0), (3.5, 2.0)):
        z = dp / sigma
        expected = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        got = bulk_volume_buy_fraction(dp, sigma)
        assert got is not None
        assert math.isclose(got, expected, rel_tol=1e-12), (dp, sigma, expected, got)


def test_bulk_volume_buy_fraction_symmetric_around_neutral() -> None:
    up = bulk_volume_buy_fraction(1.0, 1.0)
    down = bulk_volume_buy_fraction(-1.0, 1.0)
    assert up is not None and down is not None
    assert math.isclose(up + down, 1.0, rel_tol=1e-12)
    assert up > 0.5 > down


def test_bulk_volume_buy_fraction_bounded_in_unit_interval() -> None:
    for dp in (1e6, -1e6, 1.0, -1.0):
        result = bulk_volume_buy_fraction(dp, 0.5)
        assert result is not None
        assert 0.0 <= result <= 1.0


def test_bulk_volume_buy_fraction_guards_sigma() -> None:
    assert bulk_volume_buy_fraction(1.0, 0.0) is None
    assert bulk_volume_buy_fraction(1.0, -2.0) is None
    assert bulk_volume_buy_fraction(1.0, float("nan")) is None
    assert bulk_volume_buy_fraction(1.0, float("inf")) is None


def test_bulk_volume_buy_fraction_guards_price_change_and_types() -> None:
    assert bulk_volume_buy_fraction(float("nan"), 1.0) is None
    assert bulk_volume_buy_fraction(float("inf"), 1.0) is None
    assert bulk_volume_buy_fraction(float("-inf"), 1.0) is None
    assert bulk_volume_buy_fraction("not-a-number", 1.0) is None
    assert bulk_volume_buy_fraction(None, 1.0) is None
    assert bulk_volume_buy_fraction(object(), 1.0) is None
    assert bulk_volume_buy_fraction(1.0, "not-a-number") is None
    assert bulk_volume_buy_fraction(1.0, None) is None


# --------------------------------------------------------------------------- #
# vpin_from_buckets
# --------------------------------------------------------------------------- #
def test_vpin_from_buckets_mean_absolute_imbalance() -> None:
    imbalances = [0.5, -0.5, 0.25, -0.25]
    assert vpin_from_buckets(imbalances) == 0.375


def test_vpin_from_buckets_all_zero_is_zero() -> None:
    assert vpin_from_buckets([0.0, 0.0, 0.0]) == 0.0


def test_vpin_from_buckets_filters_non_finite_and_non_numeric() -> None:
    mixed = [0.5, -0.5, "bad", None, float("nan"), float("inf"), 0.25, -0.25]
    # Only the four finite numeric entries (0.5, -0.5, 0.25, -0.25) survive.
    assert vpin_from_buckets(mixed) == 0.375


def test_vpin_from_buckets_guards_empty_and_degenerate() -> None:
    assert vpin_from_buckets([]) is None
    assert vpin_from_buckets(["bad", None, float("nan"), float("inf")]) is None


# --------------------------------------------------------------------------- #
# accumulate_volume_bucket
# --------------------------------------------------------------------------- #
def test_accumulate_volume_bucket_single_close_balanced() -> None:
    state, completed = accumulate_volume_bucket(
        None, buy_volume=1.0, sell_volume=1.0, bucket_size=2.0
    )
    assert completed == [0.0]
    assert state == (0.0, 0.0)


def test_accumulate_volume_bucket_single_close_imbalanced() -> None:
    # buy=3, sell=1 in one bucket of size 4 -> imbalance = (3-1)/4 = 0.5.
    state, completed = accumulate_volume_bucket(
        None, buy_volume=3.0, sell_volume=1.0, bucket_size=4.0
    )
    assert completed == [0.5]
    assert state == (0.0, 0.0)


def test_accumulate_volume_bucket_multiple_buckets_in_one_call() -> None:
    # A single heavy call (buy=3, sell=1, total=4) against a bucket_size of 2
    # closes TWO buckets; the overall 3:1 buy/sell split is applied uniformly
    # to each (imbalance 0.5 each), and the carry drains to exactly zero.
    state, completed = accumulate_volume_bucket(
        None, buy_volume=3.0, sell_volume=1.0, bucket_size=2.0
    )
    assert completed == [0.5, 0.5]
    assert state == (0.0, 0.0)


def test_accumulate_volume_bucket_carries_state_across_calls() -> None:
    bucket_size = 10.0
    state = None
    state, completed = accumulate_volume_bucket(
        state, buy_volume=3.0, sell_volume=2.0, bucket_size=bucket_size
    )
    assert completed == []
    assert state == (3.0, 2.0)
    state, completed = accumulate_volume_bucket(
        state, buy_volume=4.0, sell_volume=1.0, bucket_size=bucket_size
    )
    # Total accumulated across both calls: buy=7, sell=3, total=10 -> closes.
    assert len(completed) == 1
    assert math.isclose(completed[0], 0.4, rel_tol=1e-12)
    assert state == (0.0, 0.0)


def test_accumulate_volume_bucket_guards_nonpositive_bucket_size() -> None:
    prior = (1.0, 2.0)
    for bad_size in (0.0, -5.0, float("nan"), float("inf")):
        state, completed = accumulate_volume_bucket(
            prior, buy_volume=5.0, sell_volume=5.0, bucket_size=bad_size
        )
        assert completed == []
        assert state == prior


def test_accumulate_volume_bucket_guards_negative_and_nonfinite_volume() -> None:
    # Negative/non-finite volume inputs are treated as zero, not raised.
    state, completed = accumulate_volume_bucket(
        None, buy_volume=-5.0, sell_volume=-3.0, bucket_size=2.0
    )
    assert completed == []
    assert state == (0.0, 0.0)
    state, completed = accumulate_volume_bucket(
        None, buy_volume=float("nan"), sell_volume=float("inf"), bucket_size=2.0
    )
    assert completed == []
    assert state == (0.0, 0.0)


def test_accumulate_volume_bucket_guards_adversarial_state() -> None:
    # A non-tuple, wrong-length, or non-numeric-element state must never
    # raise; it is treated as a fresh (0.0, 0.0) accumulator.
    for bad_state in ("garbage", 12345, [], (1.0,), ("x", "y"), None):
        state, completed = accumulate_volume_bucket(
            bad_state, buy_volume=1.0, sell_volume=1.0, bucket_size=2.0
        )
        assert completed == [0.0]
        assert state == (0.0, 0.0)
