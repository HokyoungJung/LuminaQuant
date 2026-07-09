"""Deterministic tests for the pure ``cross_sectional_residualize`` primitive.

Direct import only.  Covers the load-bearing property (a target collinear with a
regressor residualizes to ~0), orthogonality of the residual to every regressor,
the no-regressor demean identity, degenerate/collinear-regressor skipping, the
never-raise / None contract on bad input, and run-to-run determinism.
"""

from __future__ import annotations

import math

from lumina_quant.indicators.cross_sectional_residualize import cross_sectional_residualize


def _lcg_stream(seed: int):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / float(0x7FFFFFFF)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _demean(v: list[float]) -> list[float]:
    m = sum(v) / len(v)
    return [x - m for x in v]


def test_target_collinear_with_regressor_residualizes_to_zero() -> None:
    near = [1.0, -0.5, 0.3, -1.2, 0.4, 0.0, 0.7, -0.6]
    mom = [0.2, 0.9, -0.4, 0.1, -0.8, 0.5, -0.1, 0.3]
    # target is an affine function of ``near`` -> residual must vanish.
    target = [2.0 * x + 3.0 for x in near]
    resid = cross_sectional_residualize(target, [mom, near])
    assert resid is not None
    assert max(abs(v) for v in resid) < 1e-9


def test_residual_orthogonal_to_every_regressor() -> None:
    gen = _lcg_stream(7)
    n = 12
    target = [next(gen) for _ in range(n)]
    r1 = [next(gen) for _ in range(n)]
    r2 = [next(gen) for _ in range(n)]
    resid = cross_sectional_residualize(target, [r1, r2])
    assert resid is not None
    # Residual is mean-zero and orthogonal to each (demeaned) regressor.
    assert abs(sum(resid)) < 1e-9
    assert abs(_dot(resid, _demean(r1))) < 1e-9
    assert abs(_dot(resid, _demean(r2))) < 1e-9


def test_no_regressors_returns_demeaned_target() -> None:
    target = [1.0, 2.0, 3.0, 4.0]
    for regs in (None, []):
        resid = cross_sectional_residualize(target, regs)
        assert resid is not None
        assert resid == [x - 2.5 for x in target]


def test_degenerate_regressor_is_skipped() -> None:
    target = [0.5, -0.2, 0.9, -0.7, 0.1]
    good = [1.0, 2.0, -1.0, 0.5, -0.3]
    constant = [4.0, 4.0, 4.0, 4.0, 4.0]  # zero variance -> dropped
    collinear = [2.0 * x for x in good]  # collinear with ``good`` -> dropped
    resid_all = cross_sectional_residualize(target, [good, constant, collinear])
    resid_one = cross_sectional_residualize(target, [good])
    assert resid_all is not None and resid_one is not None
    # The constant + collinear regressors add no new direction: identical result.
    assert max(abs(a - b) for a, b in zip(resid_all, resid_one, strict=False)) < 1e-12


def test_bad_input_returns_none_never_raises() -> None:
    assert cross_sectional_residualize([], [[1.0]]) is None
    assert cross_sectional_residualize([1.0, 2.0], [[1.0]]) is None  # length mismatch
    assert cross_sectional_residualize([1.0, float("nan")], None) is None
    assert cross_sectional_residualize([1.0, 2.0], [[1.0, float("inf")]]) is None
    assert cross_sectional_residualize("not a list", None) is None  # type: ignore[arg-type]


def test_determinism_two_runs_identical() -> None:
    gen = _lcg_stream(101)
    n = 20
    target = [next(gen) for _ in range(n)]
    r1 = [next(gen) for _ in range(n)]
    r2 = [next(gen) for _ in range(n)]
    first = cross_sectional_residualize(target, [r1, r2])
    second = cross_sectional_residualize(target, [r1, r2])
    assert first == second
    assert first is not None and all(math.isfinite(v) for v in first)
