"""Unit tests for the Grinblatt-Han reference-price indicator.

Covers the load-bearing NON-equivalence to a rolling VWAP (FIXTURE C: identical
``rolling_vwap`` but different reference price under a volume re-ordering), the
turnover-decay weighting behaviour, the capital-gains-overhang sign, and the
never-raise / None-propagating degenerate guards.  Deterministic (no ``random``;
a small seeded LCG shapes the longer path).
"""

from __future__ import annotations

import math

from lumina_quant.indicators.reference_price import (
    capital_gains_overhang,
    grinblatt_han_reference_price,
)
from lumina_quant.indicators.vwap import rolling_vwap


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# FIXTURE C -- not a rolling-VWAP alias (the load-bearing indicator property)
# --------------------------------------------------------------------------- #


def test_fixture_c_identical_vwap_but_different_reference_under_reordering() -> None:
    # The SAME (price, volume) pairs presented in two chronological orders:
    # heavy volume early (at 100) vs heavy volume late (at 100).  A rolling VWAP
    # is a permutation-invariant sum(P*V)/sum(V), so both orderings share it
    # exactly; the Grinblatt-Han recursion survival-discounts older prices by the
    # turnover of every more-recent bar, so the reference price differs.
    pairs = [(100.0, 900.0), (110.0, 100.0)]
    early_heavy_p = [p for p, _v in pairs]
    early_heavy_v = [v for _p, v in pairs]
    late_heavy_p = [p for p, _v in reversed(pairs)]
    late_heavy_v = [v for _p, v in reversed(pairs)]

    vwap_a = rolling_vwap(early_heavy_p, early_heavy_v, 2)
    vwap_b = rolling_vwap(late_heavy_p, late_heavy_v, 2)
    assert vwap_a is not None and vwap_b is not None
    assert math.isclose(vwap_a, vwap_b, rel_tol=1e-12, abs_tol=0.0)

    ref_a = grinblatt_han_reference_price(early_heavy_p, early_heavy_v, 2, skip_recent=0)
    ref_b = grinblatt_han_reference_price(late_heavy_p, late_heavy_v, 2, skip_recent=0)
    assert ref_a is not None and ref_b is not None
    # Different reference despite identical VWAP -- the recursion is NOT a VWAP.
    assert abs(ref_a - ref_b) > 1e-6


def test_reference_is_pulled_toward_the_high_volume_cost_basis() -> None:
    # 60 heavy-volume accumulation bars at 100, then a thin drift up to 130: the
    # reference (cost basis) stays near 100, far below the last price.
    prices = [100.0]
    volumes = [1.0]
    gen = _lcg_stream(11)
    for _ in range(60):
        prices.append(100.0 * (1.0 + (next(gen) - 0.5) * 0.004))
        volumes.append(1000.0)
    for i in range(10):
        prices.append(100.0 + 30.0 * (i + 1) / 10.0)
        volumes.append(1.0)
    ref = grinblatt_han_reference_price(prices, volumes, 80, skip_recent=1)
    assert ref is not None
    assert abs(ref - 100.0) < 5.0  # anchored on the heavy-volume cost basis
    cgo = capital_gains_overhang(prices[-1], ref)
    assert cgo is not None and cgo > 0.15  # large positive overhang


def test_reference_equals_price_on_constant_series() -> None:
    prices = [100.0] * 10
    volumes = [1000.0] * 10
    ref = grinblatt_han_reference_price(prices, volumes, 8, skip_recent=1)
    assert ref is not None
    assert math.isclose(ref, 100.0, rel_tol=1e-12, abs_tol=1e-9)


def test_skip_recent_excludes_latest_bars_from_reference() -> None:
    # A late spike bar is excluded from the reference by skip_recent so it does
    # not contaminate the cost basis (only the overhang numerator uses the close).
    prices = [100.0] * 20 + [500.0]
    volumes = [1000.0] * 20 + [1000.0]
    ref_skip = grinblatt_han_reference_price(prices, volumes, 21, skip_recent=1)
    ref_noskip = grinblatt_han_reference_price(prices, volumes, 21, skip_recent=0)
    assert ref_skip is not None and ref_noskip is not None
    assert ref_skip < ref_noskip  # skipping the 500 spike keeps the basis near 100
    assert math.isclose(ref_skip, 100.0, rel_tol=1e-12, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# capital_gains_overhang sign / None propagation
# --------------------------------------------------------------------------- #


def test_capital_gains_overhang_sign() -> None:
    assert capital_gains_overhang(130.0, 100.0) == (130.0 - 100.0) / 130.0
    below = capital_gains_overhang(80.0, 100.0)
    assert below is not None and below < 0.0
    assert capital_gains_overhang(100.0, 100.0) == 0.0


def test_capital_gains_overhang_degenerate_returns_none() -> None:
    assert capital_gains_overhang(0.0, 100.0) is None
    assert capital_gains_overhang(-5.0, 100.0) is None
    assert capital_gains_overhang(float("nan"), 100.0) is None
    assert capital_gains_overhang(100.0, None) is None
    assert capital_gains_overhang(100.0, float("inf")) is None


# --------------------------------------------------------------------------- #
# never-raise / None on degenerate input
# --------------------------------------------------------------------------- #


def test_reference_none_on_degenerate_input() -> None:
    assert grinblatt_han_reference_price([], [], 8) is None
    assert grinblatt_han_reference_price([100.0], [10.0], 8) is None  # < 2 bars
    assert grinblatt_han_reference_price([100.0, 101.0], [10.0, 10.0], 1) is None  # window < 2
    # non-positive total volume
    assert grinblatt_han_reference_price([100.0, 101.0, 102.0], [0.0, 0.0, 0.0], 8) is None
    # skip_recent consumes all but < 2 bars
    assert (
        grinblatt_han_reference_price([100.0, 101.0, 102.0], [1.0, 1.0, 1.0], 8, skip_recent=2)
        is None
    )
    # non-finite input in the used window
    assert (
        grinblatt_han_reference_price(
            [100.0, float("nan"), 102.0], [1.0, 1.0, 1.0], 8, skip_recent=0
        )
        is None
    )
    assert (
        grinblatt_han_reference_price(
            [100.0, 101.0, 102.0], [1.0, float("inf"), 1.0], 8, skip_recent=0
        )
        is None
    )


def test_reference_deterministic_two_calls() -> None:
    prices = [100.0, 101.0, 99.0, 103.0, 98.0, 105.0]
    volumes = [10.0, 50.0, 5.0, 80.0, 3.0, 40.0]
    first = grinblatt_han_reference_price(prices, volumes, 5, skip_recent=1)
    second = grinblatt_han_reference_price(prices, volumes, 5, skip_recent=1)
    assert first == second
