"""B3 known-answer tests for the survivorship gate.

Coverage map (mirrors the B3 blocker spec):

(a) exact-scalar effective-trials KAT: k perfect duplicates + m independent
    collapse to N_eff = (k + m)^2 / (k^2 + m), plus the identity / all-ones
    endpoints;
(b) dispersion-term KAT: the *empirical* variance_across_trials genuinely moves
    the Deflated Sharpe Ratio (it is not a no-op);
(c) middle-correlation near-duplicate calibration: at an intermediate correlation
    both DSR inputs (N_eff and dispersion) individually move the output, and the
    gate is calibrated -- a true edge is not over-rejected and a null is not
    under-rejected (a calibration statement, not mere monotonicity);
(d) endpoints: reject a known null at realistic breadth, accept a genuine edge at
    N = 1;
(e) N_eff never exceeds raw_N;
(f) N_eff / dispersion / DSR verdict / PBO / SPA are byte-reproducible across
    thread counts.

No scipy / sklearn / eigendecomposition; only ``research_metrics`` primitives.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lumina_quant.research.alpha_candidate_ledger import (
    AlphaCandidateRecord,
    ResearchLedger,
)
from lumina_quant.research.effective_trials import (
    candidate_return_correlation_matrix,
    effective_number_of_trials,
    raw_number_of_trials,
)
from lumina_quant.research.survivorship import (
    empirical_variance_across_trials,
    evaluate_survivorship_gate,
)
from lumina_quant.strategy_factory.research_metrics import deflated_sharpe_ratio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_N = 512


# ---------------------------------------------------------------------------
# Deterministic fixtures shared by the calibration tests.
# ---------------------------------------------------------------------------


def _series(target_sharpe: float, *, seed: int, n: int = _N) -> np.ndarray:
    """Return series whose per-period Sharpe equals ``target_sharpe`` exactly."""
    raw = np.random.default_rng(seed).standard_normal(n)
    raw = (raw - raw.mean()) / np.std(raw, ddof=1)
    return 0.01 * (raw + target_sharpe)


def _correlated_family(correlation: float, *, seed: int, n: int = _N) -> np.ndarray:
    """8-candidate return matrix with a shared factor at the given correlation."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    idio_scale = float(np.sqrt(max(0.0, 1.0 - correlation**2)))
    rows = [correlation * base + idio_scale * rng.standard_normal(n) for _ in range(8)]
    return np.vstack(rows)


# Canonical family Sharpe cross-sections (per-period scale).
_FAM = np.array([0.0, 0.10, -0.05, 0.15, 0.08, -0.02, 0.12, 0.05])
_FAM_TIGHT = np.array([0.0, 0.05, -0.02, 0.07, 0.04, -0.01, 0.06, 0.03])


# ---------------------------------------------------------------------------
# (a) exact-scalar effective-trials KAT.
# ---------------------------------------------------------------------------


def test_effective_trials_exact_scalar_kat():
    k, m = 5, 3
    c = np.zeros((k + m, k + m), dtype=float)
    c[:k, :k] = 1.0  # k perfectly duplicated candidates
    for i in range(k, k + m):
        c[i, i] = 1.0  # m mutually independent candidates
    expected = (k + m) ** 2 / (k**2 + m)
    assert expected == 64.0 / 28.0
    assert effective_number_of_trials(c) == expected

    # Endpoints: identity -> S independent trials; all-ones -> a single trial.
    assert effective_number_of_trials(np.eye(8)) == 8.0
    assert effective_number_of_trials(np.ones((8, 8))) == 1.0


# ---------------------------------------------------------------------------
# (e) N_eff never exceeds raw_N.
# ---------------------------------------------------------------------------


def test_n_eff_never_exceeds_raw_n():
    for seed in (1, 17, 101, 2026):
        for corr in (0.02, 0.4, 0.75, 0.98):
            mat = _correlated_family(corr, seed=seed)
            c = candidate_return_correlation_matrix(mat)
            raw = raw_number_of_trials(c)
            n_eff = effective_number_of_trials(c)
            assert 1.0 <= n_eff <= float(raw)


# ---------------------------------------------------------------------------
# (b) dispersion-term KAT: empirical variance_across_trials moves the DSR.
# ---------------------------------------------------------------------------


def test_dispersion_term_changes_dsr():
    candidate = _series(0.18, seed=303)
    var_wide = empirical_variance_across_trials(_FAM)
    var_tight = empirical_variance_across_trials(_FAM_TIGHT)
    assert var_wide > var_tight > 0.0

    dsr_wide = deflated_sharpe_ratio(
        candidate, num_trials=6, variance_across_trials=var_wide
    )
    dsr_tight = deflated_sharpe_ratio(
        candidate, num_trials=6, variance_across_trials=var_tight
    )
    # Wider trial-Sharpe dispersion -> higher multiple-testing benchmark ->
    # strictly lower confidence. This proves the dispersion term is load bearing.
    assert dsr_wide != dsr_tight
    assert dsr_wide < dsr_tight


# ---------------------------------------------------------------------------
# (c) middle-correlation near-duplicate calibration.
# ---------------------------------------------------------------------------


def test_middle_correlation_calibration():
    c_high = candidate_return_correlation_matrix(_correlated_family(0.95, seed=11))
    c_mid = candidate_return_correlation_matrix(_correlated_family(0.60, seed=11))
    c_low = candidate_return_correlation_matrix(_correlated_family(0.05, seed=11))

    n_high = effective_number_of_trials(c_high)
    n_mid = effective_number_of_trials(c_mid)
    n_low = effective_number_of_trials(c_low)
    # The middle correlation lands strictly between the two extremes: it is a
    # genuine intermediate breadth, not a collapsed endpoint.
    assert n_high < n_mid < n_low
    assert 1.0 < n_mid < 8.0

    candidate = _series(0.18, seed=303)
    var_mid = empirical_variance_across_trials(_FAM)
    var_alt = empirical_variance_across_trials(_FAM_TIGHT)

    base = deflated_sharpe_ratio(
        candidate, num_trials=n_mid, variance_across_trials=var_mid
    )
    # Perturb DSR input 1 (N_eff) with dispersion held fixed.
    moved_by_neff = deflated_sharpe_ratio(
        candidate, num_trials=n_low, variance_across_trials=var_mid
    )
    # Perturb DSR input 2 (dispersion) with N_eff held fixed.
    moved_by_dispersion = deflated_sharpe_ratio(
        candidate, num_trials=n_mid, variance_across_trials=var_alt
    )
    # Both inputs individually change the output at the middle operating point.
    assert moved_by_neff != base
    assert moved_by_dispersion != base
    assert moved_by_neff < base  # more independent trials -> lower confidence
    assert moved_by_dispersion > base  # tighter dispersion -> higher confidence

    # Calibration (not monotonicity): at the middle breadth a true edge survives
    # and a null is rejected.
    edge = _series(0.35, seed=101)
    null = _series(0.05, seed=202)
    edge_verdict = evaluate_survivorship_gate(
        edge, family_sharpes=_FAM, correlation_matrix=c_mid
    )
    null_verdict = evaluate_survivorship_gate(
        null, family_sharpes=_FAM, correlation_matrix=c_mid
    )
    assert edge_verdict.passed is True
    assert null_verdict.passed is False
    # Both verdicts consumed a real intermediate N_eff and empirical dispersion.
    assert edge_verdict.n_eff == n_mid
    assert edge_verdict.variance_across_trials == var_mid
    assert edge_verdict.raw_n == 8


# ---------------------------------------------------------------------------
# (d) endpoints: reject null at realistic breadth, accept edge at N = 1.
# ---------------------------------------------------------------------------


def test_reject_null_at_breadth_and_accept_edge_at_n1():
    edge = _series(0.35, seed=101)
    null = _series(0.05, seed=202)

    # Accept a genuine edge at N = 1 (single trial, identity correlation).
    identity = np.eye(1, dtype=float)
    edge_n1 = evaluate_survivorship_gate(
        edge, family_sharpes=_FAM, correlation_matrix=identity
    )
    assert edge_n1.n_eff == 1.0
    assert edge_n1.raw_n == 1
    assert edge_n1.passed is True

    # Reject a known null at realistic breadth (near-orthogonal 8-candidate family).
    c_low = candidate_return_correlation_matrix(_correlated_family(0.05, seed=11))
    null_breadth = evaluate_survivorship_gate(
        null, family_sharpes=_FAM, correlation_matrix=c_low
    )
    assert null_breadth.n_eff > 4.0
    assert null_breadth.passed is False
    # The genuine edge still survives even at that realistic breadth.
    edge_breadth = evaluate_survivorship_gate(
        edge, family_sharpes=_FAM, correlation_matrix=c_low
    )
    assert edge_breadth.passed is True


# ---------------------------------------------------------------------------
# Ledger provenance: frozen record cannot mutate and round-trips through JSONL.
# ---------------------------------------------------------------------------


def test_alpha_candidate_ledger_is_frozen_and_roundtrips(tmp_path):
    record = AlphaCandidateRecord(
        candidate_id="cand-001",
        formula="rank(close) - rank(open)",
        seed=20260701,
        ic=0.041,
        icir=0.83,
        turnover=0.22,
        sharpe=0.35,
        deflated_sharpe=0.999,
        n_eff=4.4333,
        dispersion=0.00508,
        spa_pvalue=0.03,
        pbo=0.1,
        raw_n=8,
        passed=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.deflated_sharpe = 0.0  # type: ignore[misc]

    ledger = ResearchLedger(tmp_path / "alpha_ledger.jsonl")
    ledger.append(record)
    ledger.append(
        AlphaCandidateRecord(
            candidate_id="cand-002",
            formula="ts_mean(volume, 5)",
            seed=20260701,
            ic=0.005,
            icir=0.1,
            turnover=0.5,
            sharpe=0.05,
            deflated_sharpe=0.11,
            n_eff=7.9,
            dispersion=0.00508,
            spa_pvalue=0.42,
            pbo=0.6,
            raw_n=8,
            passed=False,
        )
    )
    rows = ledger.read_all()
    assert [r["candidate_id"] for r in rows] == ["cand-001", "cand-002"]
    assert ledger.query(passed=True) == [rows[0]]
    assert ledger.summary()["survived_count"] == 1
    # Provenance never leaks a mutable-alias placeholder default.
    assert rows[0]["cost_ab"] == {}


# ---------------------------------------------------------------------------
# (f) thread-count byte reproducibility of the whole gate.
# ---------------------------------------------------------------------------

_THREAD_SNIPPET = """
import json, sys
import numpy as np
from lumina_quant.research.effective_trials import (
    candidate_return_correlation_matrix,
)
from lumina_quant.research.survivorship import evaluate_survivorship_gate


def series(t, seed, n=512):
    r = np.random.default_rng(seed).standard_normal(n)
    r = (r - r.mean()) / np.std(r, ddof=1)
    return 0.01 * (r + t)


def family(corr, seed, n=512):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    s = float(np.sqrt(max(0.0, 1.0 - corr * corr)))
    return np.vstack([corr * base + s * rng.standard_normal(n) for _ in range(8)])


fam = np.array([0.0, 0.10, -0.05, 0.15, 0.08, -0.02, 0.12, 0.05])
c_mid = candidate_return_correlation_matrix(family(0.60, 11))
edge = evaluate_survivorship_gate(series(0.35, 101), family_sharpes=fam, correlation_matrix=c_mid)
null = evaluate_survivorship_gate(series(0.05, 202), family_sharpes=fam, correlation_matrix=c_mid)
sys.stdout.write(json.dumps({"edge": edge.to_dict(), "null": null.to_dict()}, sort_keys=True))
"""


def _run_with_threads(threads: int) -> str:
    env = dict(os.environ)
    env["POLARS_MAX_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    proc = subprocess.run(
        [sys.executable, "-c", _THREAD_SNIPPET],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise AssertionError(f"threads={threads} failed: {proc.stderr}")
    return proc.stdout


def test_thread_count_byte_reproducibility():
    out_1 = _run_with_threads(1)
    out_2 = _run_with_threads(2)
    out_4 = _run_with_threads(4)
    assert out_1 == out_2
    assert out_1 == out_4
    payload = json.loads(out_1)
    # Sanity: the byte-identical verdict is the calibrated one.
    assert payload["edge"]["passed"] is True
    assert payload["null"]["passed"] is False
