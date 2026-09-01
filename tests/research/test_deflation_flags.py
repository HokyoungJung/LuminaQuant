"""Flag tests for the DEFLATION lane (report defects #3, #4, minor approx_pbo).

Every fix is opt-in and defaults OFF. These tests pin BOTH sides of the contract:

* flag OFF -> byte-identical to the legacy math (the default path is unchanged);
* flag ON  -> the corrected, stricter / family-wise behavior.

Covered flags (all default OFF):
  (a) single_correlation_discount -- remove the N_eff double-discount so the DSR
      benchmark uses the raw family count once (the empirical dispersion already
      carries the ~(1-rho)V correlation compression);
  (b) hac_inference -- HAC/Newey-West correct the iid degrees of freedom behind
      the DSR studentization;
  (c) cscv_pbo -- replace the single-series approx_pbo heuristic with a family-wise
      CSCV probability of backtest overfitting.

No scipy / sklearn / eigendecomposition; only research primitives.
"""

from __future__ import annotations

import numpy as np

from lumina_quant.research.alpha_search import build_synthetic_search_panel, run_alpha_search
from lumina_quant.research.effective_trials import (
    candidate_return_correlation_matrix,
    effective_number_of_trials,
    raw_number_of_trials,
)
from lumina_quant.research.survivorship import evaluate_survivorship_gate
from lumina_quant.strategy_factory.research_metrics import (
    _hac_inflation_factor,
    cscv_pbo,
    deflated_sharpe_ratio,
)

_N = 512
_FAM = np.array([0.0, 0.10, -0.05, 0.15, 0.08, -0.02, 0.12, 0.05])


def _series(target_sharpe: float, *, seed: int, n: int = _N) -> np.ndarray:
    raw = np.random.default_rng(seed).standard_normal(n)
    raw = (raw - raw.mean()) / np.std(raw, ddof=1)
    return 0.01 * (raw + target_sharpe)


def _correlated_family(correlation: float, *, seed: int, n: int = _N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    idio = float(np.sqrt(max(0.0, 1.0 - correlation**2)))
    return np.vstack([correlation * base + idio * rng.standard_normal(n) for _ in range(8)])


def _ar1(phi: float, *, seed: int, n: int = _N, drift: float = 0.3) -> np.ndarray:
    """Positively autocorrelated AR(1) return series with a positive mean."""
    rng = np.random.default_rng(seed)
    innov = rng.standard_normal(n)
    x = np.empty(n, dtype=float)
    x[0] = innov[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + innov[t]
    return 0.01 * (x / np.std(x) + drift)


# ---------------------------------------------------------------------------
# (a) single_correlation_discount.
# ---------------------------------------------------------------------------


def test_single_correlation_discount_off_is_byte_identical():
    c_mid = candidate_return_correlation_matrix(_correlated_family(0.60, seed=11))
    cand = _series(0.35, seed=101)
    legacy = evaluate_survivorship_gate(cand, family_sharpes=_FAM, correlation_matrix=c_mid)
    explicit_off = evaluate_survivorship_gate(
        cand,
        family_sharpes=_FAM,
        correlation_matrix=c_mid,
        single_correlation_discount=False,
        hac_inference=False,
        pbo_override=None,
    )
    assert legacy.to_dict() == explicit_off.to_dict()


def test_single_correlation_discount_on_uses_raw_count_and_is_stricter():
    c_mid = candidate_return_correlation_matrix(_correlated_family(0.60, seed=11))
    raw_n = raw_number_of_trials(c_mid)
    n_eff = effective_number_of_trials(c_mid, raw_n=raw_n)
    # The fixture must actually exercise the double-discount: N_eff < raw_n.
    assert 1.0 < n_eff < float(raw_n)

    cand = _series(0.35, seed=101)
    off = evaluate_survivorship_gate(cand, family_sharpes=_FAM, correlation_matrix=c_mid)
    on = evaluate_survivorship_gate(
        cand,
        family_sharpes=_FAM,
        correlation_matrix=c_mid,
        single_correlation_discount=True,
    )
    # Applying the multiple-testing count once (raw_n) raises the benchmark, so a
    # correlated candidate's deflated Sharpe can only fall (stricter gate).
    assert on.deflated_sharpe < off.deflated_sharpe
    # The audit fields (N_eff / raw_n) are unchanged -- only the DSR moved.
    assert on.n_eff == off.n_eff
    assert on.raw_n == off.raw_n


def test_single_correlation_discount_matches_raw_count_dsr_exactly():
    c_mid = candidate_return_correlation_matrix(_correlated_family(0.60, seed=11))
    raw_n = raw_number_of_trials(c_mid)
    var = float(np.var(np.sort(_FAM), ddof=1))
    cand = _series(0.35, seed=101)
    expected = deflated_sharpe_ratio(cand, num_trials=float(raw_n), variance_across_trials=var)
    on = evaluate_survivorship_gate(
        cand,
        family_sharpes=_FAM,
        correlation_matrix=c_mid,
        single_correlation_discount=True,
    )
    assert on.deflated_sharpe == expected


# ---------------------------------------------------------------------------
# (b) hac_inference.
# ---------------------------------------------------------------------------


def test_deflated_sharpe_hac_off_is_byte_identical():
    cand = _series(0.18, seed=303)
    legacy = deflated_sharpe_ratio(cand, num_trials=6, variance_across_trials=0.005)
    explicit_off = deflated_sharpe_ratio(
        cand, num_trials=6, variance_across_trials=0.005, hac_inference=False
    )
    assert legacy == explicit_off


def test_hac_inflation_factor_properties():
    ar = _ar1(0.6, seed=0)
    vif_ar = _hac_inflation_factor(ar - ar.mean())
    # Positive serial dependence inflates the long-run variance well above iid.
    assert vif_ar > 1.0

    # Strong negative autocorrelation is floored at 1.0 (never a laxer gate).
    alt = 0.01 * ((-1.0) ** np.arange(_N) + 0.3)
    assert _hac_inflation_factor(alt - alt.mean()) == 1.0

    # Degenerate / too-short inputs are a strict no-op.
    assert _hac_inflation_factor(np.zeros(3)) == 1.0
    assert _hac_inflation_factor(np.zeros(64)) == 1.0


def test_hac_inference_on_lowers_positive_z_dsr():
    ar = _ar1(0.6, seed=0)
    off = deflated_sharpe_ratio(ar, num_trials=2, variance_across_trials=1e-4)
    on = deflated_sharpe_ratio(ar, num_trials=2, variance_across_trials=1e-4, hac_inference=True)
    # The candidate sits above the benchmark (z > 0); shrinking the effective
    # sample size for autocorrelation pulls the confidence down.
    assert 0.5 < on < off


def test_gate_hac_off_is_byte_identical():
    c_mid = candidate_return_correlation_matrix(_correlated_family(0.60, seed=11))
    cand = _series(0.35, seed=101)
    legacy = evaluate_survivorship_gate(cand, family_sharpes=_FAM, correlation_matrix=c_mid)
    off = evaluate_survivorship_gate(
        cand, family_sharpes=_FAM, correlation_matrix=c_mid, hac_inference=False
    )
    assert legacy.to_dict() == off.to_dict()


# ---------------------------------------------------------------------------
# (c) cscv_pbo primitive.
# ---------------------------------------------------------------------------


def test_cscv_pbo_is_bounded_and_deterministic():
    mat = _correlated_family(0.2, seed=7)
    a = cscv_pbo(mat)
    b = cscv_pbo(mat)
    assert a == b
    assert 0.0 <= a <= 1.0


def test_cscv_pbo_degenerate_inputs_return_one():
    # Fewer than two candidates, or too few observations, are undecidable -> 1.0.
    assert cscv_pbo(np.zeros((1, 100))) == 1.0
    assert cscv_pbo(np.zeros((5, 3))) == 1.0
    assert cscv_pbo(np.zeros((6, 100))) == 1.0  # zero-variance family


def test_cscv_pbo_dominant_signal_is_low():
    rng = np.random.default_rng(1)
    dom_row = 0.01 * (np.ones(400) * 0.5 + 0.1 * rng.standard_normal(400))
    noise = [0.01 * np.random.default_rng(100 + i).standard_normal(400) for i in range(7)]
    dom = np.vstack([dom_row, *noise])
    # A candidate that dominates in every block is the in-sample winner AND ranks
    # top out-of-sample -> logit > 0 on every split -> PBO at the floor.
    assert cscv_pbo(dom) < 0.5


def test_cscv_pbo_rejects_non_2d():
    import pytest

    with pytest.raises(ValueError):
        cscv_pbo(np.zeros(100))


# ---------------------------------------------------------------------------
# run_alpha_search integration.
# ---------------------------------------------------------------------------


def test_alpha_search_all_flags_off_is_byte_identical():
    panel = build_synthetic_search_panel()
    legacy = run_alpha_search(panel, max_candidates=40)
    explicit_off = run_alpha_search(
        panel,
        max_candidates=40,
        single_correlation_discount=False,
        hac_inference=False,
        cscv_pbo=False,
    )
    assert legacy.records == explicit_off.records
    assert legacy.promoted_ids == explicit_off.promoted_ids
    assert legacy.n_eff == explicit_off.n_eff
    assert legacy.dispersion == explicit_off.dispersion


def test_alpha_search_cscv_pbo_shares_one_family_value():
    panel = build_synthetic_search_panel()
    base = run_alpha_search(panel, max_candidates=40)
    costed = run_alpha_search(panel, max_candidates=40, cscv_pbo=True)
    # Every record carries the single family-wise CSCV PBO (a family property).
    family_pbos = {round(rec["pbo"], 12) for rec in costed.records}
    assert len(family_pbos) == 1
    # ... which differs from the per-candidate approx_pbo cross-section by default.
    base_pbos = {round(rec["pbo"], 12) for rec in base.records}
    assert len(base_pbos) > 1
    # Deterministic under the flag.
    again = run_alpha_search(panel, max_candidates=40, cscv_pbo=True)
    assert costed.records == again.records


def test_alpha_search_single_correlation_discount_tightens_gate():
    panel = build_synthetic_search_panel()
    off = run_alpha_search(panel, max_candidates=40)
    on = run_alpha_search(panel, max_candidates=40, single_correlation_discount=True)
    # Applying the trial-count discount once raises the benchmark, so no deflated
    # Sharpe rises and the promoted set cannot grow.
    off_dsr = {rec["candidate_id"]: rec["deflated_sharpe"] for rec in off.records}
    for rec in on.records:
        assert rec["deflated_sharpe"] <= off_dsr[rec["candidate_id"]] + 1e-12
    assert set(on.promoted_ids).issubset(set(off.promoted_ids))
    # Deterministic under the flag.
    again = run_alpha_search(panel, max_candidates=40, single_correlation_discount=True)
    assert on.records == again.records
