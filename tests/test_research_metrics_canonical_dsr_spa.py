"""Tests for the canonical DSR / SPA implementations and the opt-in hard gate.

Covers the audit fix that replaced the simplified DSR/SPA proxies with the
canonical Bailey & Lopez de Prado Deflated Sharpe Ratio and a stationary
block-bootstrap reality-check p-value, and made them bindable via a hard gate in
``select_diversified_shortlist`` (default off => existing behavior preserved).
"""

from __future__ import annotations

import math

import numpy as np

from lumina_quant.strategy_factory import research_metrics as rm
from lumina_quant.strategy_factory.selection import (
    passes_dsr_spa_hard_gate,
    select_diversified_shortlist,
)


def _make_returns(*, mean: float, sd: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = (x - x.mean()) / x.std(ddof=1)  # exactly mean 0, sd 1
    return x * sd + mean


# --------------------------------------------------------------------------- #
# expected_max_sharpe (multiple-testing benchmark)
# --------------------------------------------------------------------------- #


def test_expected_max_sharpe_single_trial_is_zero() -> None:
    assert rm.expected_max_sharpe(variance_across_trials=0.01, num_trials=1) == 0.0


def test_expected_max_sharpe_monotonic_increasing_in_trials() -> None:
    vals = [
        rm.expected_max_sharpe(variance_across_trials=0.01, num_trials=k)
        for k in (1, 2, 10, 50, 200)
    ]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    assert vals[-1] > vals[1] > 0.0


def test_expected_max_sharpe_matches_canonical_closed_form() -> None:
    # Hand-compute E[max SR] from the canonical formula with explicit inputs.
    var = 0.01
    k = 10
    gamma = 0.5772156649015329
    z1 = rm._norm_ppf(1.0 - 1.0 / k)
    z2 = rm._norm_ppf(1.0 - 1.0 / (k * math.e))
    expected = math.sqrt(var) * ((1.0 - gamma) * z1 + gamma * z2)
    assert rm.expected_max_sharpe(variance_across_trials=var, num_trials=k) == expected


# --------------------------------------------------------------------------- #
# _norm_ppf / _norm_cdf round-trip (scipy-free normal helpers)
# --------------------------------------------------------------------------- #


def test_norm_ppf_cdf_round_trip() -> None:
    for p in (0.05, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999):
        assert abs(rm._norm_cdf(rm._norm_ppf(p)) - p) < 1e-6


# --------------------------------------------------------------------------- #
# Canonical Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #


def test_deflated_sharpe_monotonic_decreasing_in_num_trials() -> None:
    returns = _make_returns(mean=0.001, sd=0.01, n=400, seed=11)
    vals = [rm.deflated_sharpe_ratio(returns, num_trials=k) for k in (1, 2, 5, 20, 100)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    assert vals[0] > vals[-1]


def test_deflated_sharpe_monotonic_increasing_in_sharpe() -> None:
    low = _make_returns(mean=0.0005, sd=0.01, n=400, seed=21)
    high = _make_returns(mean=0.003, sd=0.01, n=400, seed=21)
    dsr_low = rm.deflated_sharpe_ratio(low, num_trials=10)
    dsr_high = rm.deflated_sharpe_ratio(high, num_trials=10)
    assert dsr_high > dsr_low


def test_deflated_sharpe_matches_hand_computed_value() -> None:
    # Construct a series with a known per-period Sharpe and supply Var(SR) so the
    # whole computation is reproducible by hand from the canonical formula.
    series = _make_returns(mean=0.15, sd=1.0, n=257, seed=3)
    var_sr = 0.02
    num_trials = 8

    mu = float(np.mean(series))
    sigma = float(np.std(series, ddof=1))
    sharpe = mu / sigma
    n = float(series.size)
    centered = series - mu
    skew = float(np.mean(centered**3)) / sigma**3
    kurt = float(np.mean(centered**4)) / sigma**4

    sr0 = rm.expected_max_sharpe(variance_across_trials=var_sr, num_trials=num_trials)
    denom = math.sqrt(max(1e-8, 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe**2))
    z = (sharpe - sr0) * math.sqrt(n - 1.0) / denom
    hand = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    impl = rm.deflated_sharpe_ratio(series, num_trials=num_trials, variance_across_trials=var_sr)
    assert abs(impl - hand) < 1e-9


def test_deflated_sharpe_returns_zero_for_short_or_flat_series() -> None:
    assert rm.deflated_sharpe_ratio(np.zeros(8), num_trials=4) == 0.0
    assert rm.deflated_sharpe_ratio(np.full(64, 0.01), num_trials=4) == 0.0


def test_deflated_sharpe_does_not_divide_penalty_by_series_length() -> None:
    # Regression guard for the old proxy that divided expected-max-Sharpe by
    # sqrt(n): the multiple-testing penalty must NOT shrink as the series grows.
    var_sr = 0.02
    short = rm.expected_max_sharpe(variance_across_trials=var_sr, num_trials=20)
    # E[max SR] is purely a function of trials/variance, independent of n.
    assert short > 0.0
    long_returns = _make_returns(mean=0.0008, sd=0.01, n=2000, seed=31)
    short_returns = _make_returns(mean=0.0008, sd=0.01, n=200, seed=31)
    dsr_long = rm.deflated_sharpe_ratio(long_returns, num_trials=20)
    dsr_short = rm.deflated_sharpe_ratio(short_returns, num_trials=20)
    # More evidence (longer series) with the same per-period edge yields a
    # stronger (or equal), never weaker, deflation confidence.
    assert dsr_long >= dsr_short


# --------------------------------------------------------------------------- #
# SPA / reality-check p-value
# --------------------------------------------------------------------------- #


def test_spa_pvalue_in_unit_interval() -> None:
    returns = _make_returns(mean=0.001, sd=0.01, n=300, seed=5)
    p = rm.spa_like_pvalue(returns)
    assert 0.0 <= p <= 1.0


def test_spa_pvalue_rejects_pure_noise() -> None:
    # Across many pure-noise series the p-value should be far from significant
    # (centred near uniform), with essentially no false rejections at 5%.
    pvals = []
    for s in range(40):
        noise = _make_returns(mean=0.0, sd=0.01, n=300, seed=100 + s)
        pvals.append(rm.spa_like_pvalue(noise, seed=s + 1))
    false_rejections = sum(1 for p in pvals if p < 0.05)
    assert false_rejections == 0
    assert (sum(pvals) / len(pvals)) > 0.2


def test_spa_pvalue_flags_strong_signal() -> None:
    signal = _make_returns(mean=0.004, sd=0.01, n=400, seed=7)
    assert rm.spa_like_pvalue(signal) < 0.05


def test_spa_pvalue_non_positive_mean_returns_one() -> None:
    losing = _make_returns(mean=-0.001, sd=0.01, n=300, seed=9)
    assert rm.spa_like_pvalue(losing) == 1.0


def test_spa_pvalue_is_deterministic_for_fixed_seed() -> None:
    returns = _make_returns(mean=0.0015, sd=0.01, n=300, seed=13)
    assert rm.spa_like_pvalue(returns, seed=42) == rm.spa_like_pvalue(returns, seed=42)


# --------------------------------------------------------------------------- #
# Hard gate in selection (opt-in; default no-op)
# --------------------------------------------------------------------------- #


def _candidate(name: str, *, deflated_sharpe: float, spa_pvalue: float) -> dict:
    return {
        "name": name,
        "symbols": [f"{name.upper()}/USDT"],
        "oos": {
            "return": 0.05,
            "sharpe": 2.0,
            "pbo": 0.1,
            "turnover": 0.1,
            "mdd": 0.05,
            "trades": 30,
            "deflated_sharpe": deflated_sharpe,
            "spa_pvalue": spa_pvalue,
        },
        "hurdle_fields": {"oos": {"score": 10.0, "excess_return": 0.05, "pass": True}},
    }


def test_hard_gate_is_noop_when_disabled() -> None:
    weak = _candidate("alpha", deflated_sharpe=0.0, spa_pvalue=0.99)
    # Default params: gate disabled => candidate passes.
    assert passes_dsr_spa_hard_gate(weak, mode="oos") is True


def test_hard_gate_rejects_failing_dsr_when_enabled() -> None:
    weak = _candidate("alpha", deflated_sharpe=0.10, spa_pvalue=0.10)
    params = {"dsr_spa_hard_gate": 1.0, "dsr_gate_floor": 0.5, "spa_gate_ceiling": 0.2}
    assert passes_dsr_spa_hard_gate(weak, mode="oos", robust_score_params=params) is False


def test_hard_gate_rejects_failing_spa_when_enabled() -> None:
    weak = _candidate("alpha", deflated_sharpe=0.9, spa_pvalue=0.40)
    params = {"dsr_spa_hard_gate": 1.0, "dsr_gate_floor": 0.5, "spa_gate_ceiling": 0.2}
    assert passes_dsr_spa_hard_gate(weak, mode="oos", robust_score_params=params) is False


def test_hard_gate_accepts_passing_candidate_when_enabled() -> None:
    strong = _candidate("alpha", deflated_sharpe=0.9, spa_pvalue=0.05)
    params = {"dsr_spa_hard_gate": 1.0, "dsr_gate_floor": 0.5, "spa_gate_ceiling": 0.2}
    assert passes_dsr_spa_hard_gate(strong, mode="oos", robust_score_params=params) is True


def test_select_shortlist_gate_disabled_keeps_failing_candidate() -> None:
    candidates = [
        _candidate("strong", deflated_sharpe=0.9, spa_pvalue=0.02),
        _candidate("weak", deflated_sharpe=0.05, spa_pvalue=0.95),
    ]
    shortlist = select_diversified_shortlist(candidates, mode="oos")
    names = {row["name"] for row in shortlist}
    assert names == {"strong", "weak"}


def test_select_shortlist_gate_enabled_rejects_failing_candidate() -> None:
    candidates = [
        _candidate("strong", deflated_sharpe=0.9, spa_pvalue=0.02),
        _candidate("weak", deflated_sharpe=0.05, spa_pvalue=0.95),
    ]
    params = {"dsr_spa_hard_gate": 1.0, "dsr_gate_floor": 0.5, "spa_gate_ceiling": 0.2}
    shortlist = select_diversified_shortlist(candidates, mode="oos", robust_score_params=params)
    names = {row["name"] for row in shortlist}
    assert names == {"strong"}
