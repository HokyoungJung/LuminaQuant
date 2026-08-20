"""Deterministic tests for the Lo-MacKinlay variance-ratio indicator.

Covers the PARITY LOCK of the default (BIASED overlapping) estimator against a
verbatim inline copy of the pre-extraction strategy-private ``_variance_ratio``
(exact float equality on seeded LCG return vectors and on the sleeve's own
fixtures); the import alias in the consuming rider sleeve; the opt-in unbiased
estimator via its closed-form relation to the biased one
(``VR_u = VR_b * (n - 1) / (n - k)``); the heteroskedasticity-robust and iid
z-statistics against independent inline references and hand-derived signs; and
the never-raise guards.  No backtest is run; no data files are read.
"""

from __future__ import annotations

import math

import pytest

from lumina_quant.indicators.variance_ratio import variance_ratio, variance_ratio_zstat
from lumina_quant.strategies.cusum_varratio_alpha_sleeves import _variance_ratio

_EPS = 1e-12


def _lcg(seed: int):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2**31)
        yield state / 2**31


def _lcg_returns(seed: int, n: int, scale: float = 0.01) -> list[float]:
    gen = _lcg(seed)
    return [(next(gen) - 0.5) * scale for _ in range(n)]


# --------------------------------------------------------------------------- #
# Verbatim PRE-EXTRACTION reference (copied from the previously private
# ``_variance_ratio`` in strategies/cusum_varratio_alpha_sleeves.py) -- the
# parity oracle for the BIASED default.
# --------------------------------------------------------------------------- #
def _reference_variance_ratio(returns: list[float], k: int) -> float | None:
    n = len(returns)
    kk = max(2, int(k))
    if n < kk + 1:
        return None
    mean = sum(returns) / n
    var1 = sum((r - mean) * (r - mean) for r in returns) / n
    if var1 <= _EPS:
        return None
    acc = 0.0
    count = 0
    target = kk * mean
    for i in range(n - kk + 1):
        s = math.fsum(returns[i : i + kk])
        acc += (s - target) * (s - target)
        count += 1
    if count <= 0:
        return None
    var_k = acc / count
    return var_k / (kk * var1)


# --------------------------------------------------------------------------- #
# Parity lock (default biased estimator)
# --------------------------------------------------------------------------- #
def test_parity_with_pre_extraction_reference_exact() -> None:
    for seed in (1, 42, 20260820):
        for n in (5, 16, 64, 257):
            returns = _lcg_returns(seed, n)
            for k in (2, 3, 4, 8):
                got = variance_ratio(returns, k)
                want = _reference_variance_ratio(returns, k)
                assert got == want, (seed, n, k)


def test_parity_on_sleeve_fixtures_exact() -> None:
    # The exact fixtures the existing sleeve test exercises.
    alternating = [0.01 if i % 2 == 0 else -0.01 for i in range(64)]
    block = [0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01]
    persistent = block * 8
    for returns in (alternating, persistent):
        assert variance_ratio(returns, 2) == _reference_variance_ratio(returns, 2)


def test_sleeve_import_alias_is_the_indicator_function() -> None:
    # The rider sleeve's ``_variance_ratio`` IS the extracted indicator (the
    # refactor moved the function, it did not fork it), so sleeve signals are
    # byte-identical by construction.
    assert _variance_ratio is variance_ratio


# --------------------------------------------------------------------------- #
# Regime behavior + guards (biased default)
# --------------------------------------------------------------------------- #
def test_regimes_and_guards() -> None:
    alternating = [0.01 if i % 2 == 0 else -0.01 for i in range(64)]
    vr_mr = variance_ratio(alternating, 2)
    assert vr_mr is not None and vr_mr < 1.0
    # Perfect +/- alternation: every 2-period sum is exactly zero => VR == 0.
    assert vr_mr == pytest.approx(0.0, abs=1e-15)
    block = [0.01] * 4 + [-0.01] * 4
    vr_pers = variance_ratio(block * 8, 2)
    assert vr_pers is not None and vr_pers > 1.0
    # Too few points / degenerate variance / garbage k => None, never raise.
    assert variance_ratio([0.01, 0.01], 4) is None
    assert variance_ratio([0.01] * 32, 2) is None  # constant returns, var1 == 0
    assert variance_ratio([], 2) is None
    assert variance_ratio(None, 2) is None
    assert variance_ratio(_lcg_returns(3, 32), "x") is None  # type: ignore[arg-type]


def test_garbage_entries_dropped_never_raise() -> None:
    clean = _lcg_returns(9, 32)
    dirty = ["x", *clean[:16], None, *clean[16:], float("nan"), float("inf")]
    assert variance_ratio(dirty, 4) == variance_ratio(clean, 4)


def test_determinism_two_calls() -> None:
    returns = _lcg_returns(11, 128)
    assert variance_ratio(returns, 4) == variance_ratio(list(returns), 4)
    assert variance_ratio_zstat(returns, 4) == variance_ratio_zstat(list(returns), 4)


# --------------------------------------------------------------------------- #
# Opt-in unbiased estimator
# --------------------------------------------------------------------------- #
def test_unbiased_closed_form_relation_to_biased() -> None:
    # var_k legs differ by (n-k+1) vs k(n-k+1)(1-k/n) and var1 legs by n vs
    # n-1, which collapses to VR_unbiased = VR_biased * (n - 1) / (n - k).
    for seed in (5, 77):
        for n in (16, 64, 200):
            returns = _lcg_returns(seed, n)
            for k in (2, 4, 8):
                biased = variance_ratio(returns, k)
                unbiased = variance_ratio(returns, k, unbiased=True)
                assert biased is not None and unbiased is not None
                assert unbiased == pytest.approx(biased * (n - 1) / (n - k), rel=1e-12), (
                    seed,
                    n,
                    k,
                )


def test_unbiased_guards() -> None:
    assert variance_ratio([0.01, -0.01], 2, unbiased=True) is None  # n < k + 1
    assert variance_ratio([0.01] * 32, 2, unbiased=True) is None  # degenerate


# --------------------------------------------------------------------------- #
# Lo-MacKinlay z-statistics (opt-in)
# --------------------------------------------------------------------------- #
def _reference_zstat_heteroskedastic(returns: list[float], k: int) -> float | None:
    """Independent inline reference: Lo-MacKinlay (1988) robust z*(k)."""
    n = len(returns)
    if n < k + 1:
        return None
    vr = variance_ratio(returns, k, unbiased=True)
    if vr is None:
        return None
    mu = sum(returns) / n
    dev_sq = [(r - mu) ** 2 for r in returns]
    denom = math.fsum(dev_sq) ** 2
    if denom <= 0.0:
        return None
    theta = 0.0
    for j in range(1, k):
        num = math.fsum(dev_sq[t] * dev_sq[t - j] for t in range(j, n))
        theta += (2.0 * (k - j) / k) ** 2 * (num / denom)
    if theta <= 0.0:
        return None
    return (vr - 1.0) / math.sqrt(theta)


def test_zstat_heteroskedastic_matches_reference() -> None:
    for seed in (13, 202):
        for n in (32, 128):
            returns = _lcg_returns(seed, n)
            for k in (2, 4):
                got = variance_ratio_zstat(returns, k)
                want = _reference_zstat_heteroskedastic(returns, k)
                assert got is not None and want is not None
                assert got == pytest.approx(want, rel=1e-10), (seed, n, k)


def test_zstat_homoskedastic_matches_phi_formula() -> None:
    returns = _lcg_returns(17, 96)
    for k in (2, 4, 8):
        vr = variance_ratio(returns, k, unbiased=True)
        assert vr is not None
        n = len(returns)
        phi = 2.0 * (2.0 * k - 1.0) * (k - 1.0) / (3.0 * k * n)
        want = (vr - 1.0) / math.sqrt(phi)
        got = variance_ratio_zstat(returns, k, heteroskedastic=False)
        assert got == pytest.approx(want, rel=1e-12)


def test_zstat_signs_on_known_regimes() -> None:
    # Persistent blocks => VR > 1 => positive z; alternating => VR < 1 =>
    # negative z (matches Lo-MacKinlay's persistence/mean-reversion reading).
    block = [0.01] * 4 + [-0.01] * 4
    z_pers = variance_ratio_zstat(block * 16, 2)
    assert z_pers is not None and z_pers > 0.0
    alternating = [0.01 if i % 2 == 0 else -0.01 for i in range(128)]
    z_mr = variance_ratio_zstat(alternating, 2)
    assert z_mr is not None and z_mr < 0.0


def test_zstat_guards_never_raise() -> None:
    assert variance_ratio_zstat([0.01, -0.01], 2) is None
    assert variance_ratio_zstat([0.01] * 64, 2) is None  # degenerate variance
    assert variance_ratio_zstat([], 2) is None
    assert variance_ratio_zstat(None, 2) is None
    assert variance_ratio_zstat(_lcg_returns(3, 64), "x") is None  # type: ignore[arg-type]
    assert variance_ratio_zstat(["x", None], 2) is None
