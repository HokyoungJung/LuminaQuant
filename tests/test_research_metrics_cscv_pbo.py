"""Parity and performance-guard tests for the CSCV-PBO sufficient-statistics
fast path.

``cscv_pbo`` (``lumina_quant.strategy_factory.research_metrics``) used to
re-slice and re-scan the full ``(n_candidates, n_periods)`` return matrix
inside every one of its ``C(S, S/2)`` in-sample/out-of-sample partitions,
which made it combinatorially expensive at the canonical ``n_splits=16``
(measured ~83s @ 256 candidates x 2000 periods). The module now precomputes
per-candidate, per-block sufficient statistics (finite count / sum / sum of
squares) once up front and sums those across each partition instead of
reslicing, and looks up the out-of-sample rank of the in-sample winner
directly instead of materializing a full tie-grouped rank array.

This file pins that the fast path's OUTPUT never changes: ``_reference_cscv_pbo``
below is a self-contained copy of the original reslice-and-recompute algorithm
(built from the module's own retained reference-oracle helpers,
``_slice_sharpes`` / ``_ascending_average_rank`` -- no longer on ``cscv_pbo``'s
hot path, kept for exactly this purpose), and every case asserts it against
the current ``cscv_pbo``.
"""

from __future__ import annotations

import logging
import math
import time
from itertools import combinations

import numpy as np
import pytest

from lumina_quant.strategy_factory import research_metrics
from lumina_quant.strategy_factory.research_metrics import (
    _ascending_average_rank,
    _slice_sharpes,
    cscv_pbo,
)


def _reference_cscv_pbo(returns_matrix: np.ndarray, *, n_splits: int = 8) -> float:
    """Reslice-and-recompute oracle: the pre-optimization ``cscv_pbo`` algorithm.

    Rebuilt here (rather than imported) so this test pins *behavior*, not
    implementation -- it keeps working even if the retained
    ``_slice_sharpes`` / ``_ascending_average_rank`` helpers are ever removed
    from the module, in which case this copy becomes the last word on the
    legacy semantics.
    """
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be a 2-D (n_candidates, n_periods) array")
    n_cand, n_periods = int(m.shape[0]), int(m.shape[1])
    if n_cand < 2 or n_periods < 4:
        return 1.0

    s = int(n_splits)
    if s < 2:
        s = 2
    if s % 2 == 1:
        s -= 1
    while s >= 2 and (n_periods // s) < 2:
        s -= 2
    if s < 2:
        return 1.0

    block_size = n_periods // s
    blocks = [np.arange(b * block_size, (b + 1) * block_size) for b in range(s)]
    half = s // 2

    le_zero = 0
    total = 0
    for is_blocks in combinations(range(s), half):
        is_set = frozenset(is_blocks)
        is_cols = np.concatenate([blocks[b] for b in range(s) if b in is_set])
        oos_cols = np.concatenate([blocks[b] for b in range(s) if b not in is_set])
        is_perf = _slice_sharpes(m[:, is_cols])
        oos_perf = _slice_sharpes(m[:, oos_cols])
        best = int(np.argmax(is_perf))
        oos_ranks = _ascending_average_rank(oos_perf)
        omega = float(oos_ranks[best]) / (n_cand + 1.0)
        omega = min(1.0 - 1e-12, max(1e-12, omega))
        lam = math.log(omega / (1.0 - omega))
        total += 1
        if lam <= 0.0:
            le_zero += 1
    if total == 0:
        return 1.0
    return float(le_zero / total)


def _assert_parity(mat: np.ndarray, n_splits: int) -> None:
    expected = _reference_cscv_pbo(mat, n_splits=n_splits)
    actual = cscv_pbo(mat, n_splits=n_splits)
    # In practice this is bit-identical across every case exercised below: PBO
    # is a ratio of two partition COUNTS, so the fast path only diverges from
    # the oracle if a floating-point difference somewhere flips a partition's
    # lambda<=0 classification. rtol/atol<=1e-9 is the documented fallback
    # tolerance in case that ever happens on an input/platform this suite
    # doesn't cover.
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("n_splits", [8, 10, 12, 16])
@pytest.mark.parametrize("seed", range(4))
def test_cscv_pbo_matches_reference_oracle_on_random_matrices(seed: int, n_splits: int) -> None:
    # Shapes are kept modest (not the 256x2000 scale used for the timing
    # comparison) because the reference oracle below is deliberately the
    # unoptimized O(n_candidates * n_periods)-per-partition algorithm; the
    # 256x2000 case is covered for parity separately in the module's docstring
    # benchmark and is exercised here purely for speed in
    # test_cscv_pbo_s16_is_fast_at_256x2000.
    rng = np.random.default_rng(seed)
    n_cand = int(rng.integers(6, 24))
    n_periods = int(rng.integers(200, 600))
    scale = float(rng.choice([0.001, 0.01, 0.05, 1.0]))
    mat = rng.standard_normal((n_cand, n_periods)) * scale
    _assert_parity(mat, n_splits)


@pytest.mark.parametrize("n_splits", [8, 10, 12, 16])
def test_cscv_pbo_matches_reference_oracle_with_ties(n_splits: int) -> None:
    rng = np.random.default_rng(99)
    base = rng.standard_normal((10, 400)) * 0.01
    mat = np.vstack([base, base[0:1], base[2:3]])  # duplicate rows -> exact ties
    _assert_parity(mat, n_splits)


@pytest.mark.parametrize("n_splits", [8, 10, 12, 16])
def test_cscv_pbo_matches_reference_oracle_with_nans(n_splits: int) -> None:
    rng = np.random.default_rng(7)
    mat = rng.standard_normal((20, 500)) * 0.01
    mat[rng.random(mat.shape) < 0.02] = np.nan
    _assert_parity(mat, n_splits)


def test_cscv_pbo_s16_is_fast_at_256x2000() -> None:
    """Regression guard: S=16 (the canonical CSCV split count) must stay fast.

    Before the sufficient-statistics fast path this call took ~83s at
    256x2000; a generous 8s ceiling (well above the ~1s observed locally)
    catches a reintroduced O(n_candidates * n_periods)-per-partition recompute
    without making the suite flaky on slower CI hardware.
    """
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((256, 2000)) * 0.01
    t0 = time.perf_counter()
    cscv_pbo(mat, n_splits=16)
    assert time.perf_counter() - t0 < 8.0


def test_cscv_pbo_warns_once_for_large_partition_counts(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(research_metrics, "_cscv_large_partition_warned", False)
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((16, 200)) * 0.01

    with caplog.at_level(logging.WARNING, logger=research_metrics.__name__):
        cscv_pbo(mat, n_splits=18)  # C(18, 9) = 48620 > threshold
        cscv_pbo(mat, n_splits=18)  # second call must NOT warn again

    warnings = [r for r in caplog.records if "partitions" in r.message]
    assert len(warnings) == 1


def test_cscv_pbo_small_n_splits_does_not_warn(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(research_metrics, "_cscv_large_partition_warned", False)
    rng = np.random.default_rng(2)
    mat = rng.standard_normal((16, 200)) * 0.01

    with caplog.at_level(logging.WARNING, logger=research_metrics.__name__):
        cscv_pbo(mat, n_splits=8)

    assert not any("partitions" in r.message for r in caplog.records)
