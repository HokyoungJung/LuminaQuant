"""Tests for the cost-realism A/B and RiskManager go-live parity (C6a)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lumina_quant.research.alpha_search import build_synthetic_search_panel, run_alpha_search
from lumina_quant.research.cost_realism import (
    FLAT_REGIME,
    REALISTIC_REGIME,
    CostRegime,
    apply_cost_drag,
    check_risk_gate_parity,
    cost_ab_report,
    one_side_cost_fraction,
    roundtrip_cost_fraction,
)


def _variable_returns(mean: float, amp: float = 0.001, n: int = 120) -> np.ndarray:
    """Deterministic return series with fixed mean and non-zero variance."""
    return mean + amp * np.sin(np.arange(n, dtype=np.float64) * 0.3)


# --- cost regimes ----------------------------------------------------------


def test_realistic_regime_costs_more_than_flat():
    flat = one_side_cost_fraction(FLAT_REGIME)
    real = one_side_cost_fraction(REALISTIC_REGIME)
    assert real > flat > 0.0
    assert roundtrip_cost_fraction(REALISTIC_REGIME) == 2.0 * real


def test_sqrt_impact_scales_with_participation():
    lo = REALISTIC_REGIME.slippage_at(0.01)
    hi = REALISTIC_REGIME.slippage_at(0.25)
    assert hi > lo
    # flat regime ignores participation entirely
    assert FLAT_REGIME.slippage_at(0.01) == FLAT_REGIME.slippage_at(0.25)


def test_apply_cost_drag_reduces_and_is_deterministic():
    gross = _variable_returns(0.002)
    net = apply_cost_drag(gross, turnover=0.1, regime=REALISTIC_REGIME)
    assert net.shape == gross.shape
    assert float(np.mean(net)) < float(np.mean(gross))
    net2 = apply_cost_drag(gross, turnover=0.1, regime=REALISTIC_REGIME)
    assert np.array_equal(net, net2)


def test_funding_adds_drag_only_when_periods_positive():
    gross = _variable_returns(0.002)
    no_funding = apply_cost_drag(gross, turnover=0.05, regime=REALISTIC_REGIME)
    with_funding = apply_cost_drag(
        gross, turnover=0.05, regime=REALISTIC_REGIME, funding_periods_per_step=3.0
    )
    assert float(np.mean(with_funding)) < float(np.mean(no_funding))


# --- cost_ab_report --------------------------------------------------------


def test_cost_ab_report_all_float_and_deterministic():
    gross = _variable_returns(0.002)
    r1 = cost_ab_report(gross, turnover=0.05)
    r2 = cost_ab_report(gross, turnover=0.05)
    assert r1 == r2
    assert all(isinstance(v, float) for v in r1.values())
    assert set(r1) >= {"gross_sharpe", "net_sharpe_realistic", "survives", "sharpe_decay_ratio"}


def test_strong_low_turnover_candidate_survives():
    gross = _variable_returns(0.002)  # strong mean, low turnover
    report = cost_ab_report(gross, turnover=0.02)
    assert report["survives"] == 1.0
    assert report["net_sharpe_realistic"] > 0.0
    # flat costs less than realistic
    assert report["net_sharpe_flat"] >= report["net_sharpe_realistic"]


def test_marginal_high_turnover_candidate_does_not_survive():
    gross = _variable_returns(0.0002)  # tiny edge
    report = cost_ab_report(gross, turnover=1.5)  # churns the book every period
    assert report["survives"] == 0.0
    assert report["net_sharpe_realistic"] < report["gross_sharpe"]


def test_negative_gross_never_survives():
    gross = _variable_returns(-0.001)
    report = cost_ab_report(gross, turnover=0.01)
    assert report["survives"] == 0.0


# --- alpha_search integration (additive, default OFF) ----------------------


def test_alpha_search_cost_gate_off_is_byte_identical():
    panel = build_synthetic_search_panel()
    a = run_alpha_search(panel, max_candidates=40)
    b = run_alpha_search(panel, max_candidates=40)
    assert a.records == b.records
    # default path leaves cost_ab empty (byte-identical to pre-cost-gate behaviour)
    assert all(row["cost_ab"] == {} for row in a.records)


def test_alpha_search_cost_gate_on_only_removes_survivors():
    panel = build_synthetic_search_panel()
    baseline = run_alpha_search(panel, max_candidates=40)
    costed = run_alpha_search(panel, max_candidates=40, cost_ab_enabled=True)
    # cost gate can only shrink the promoted set, never grow it
    assert set(costed.promoted_ids) <= set(baseline.promoted_ids)
    # every record now carries a populated cost_ab with a survives verdict
    assert all("survives" in row["cost_ab"] for row in costed.records)
    assert any(row["cost_ab"] for row in costed.records)


def test_alpha_search_cost_gate_on_is_deterministic():
    panel = build_synthetic_search_panel()
    a = run_alpha_search(panel, max_candidates=40, cost_ab_enabled=True)
    b = run_alpha_search(panel, max_candidates=40, cost_ab_enabled=True)
    assert a.records == b.records
    assert a.promoted_ids == b.promoted_ids


# --- RiskManager go-live parity -------------------------------------------


def _order(quantity: float, *, reduce_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(quantity=quantity, reduce_only=reduce_only)


def test_risk_gate_parity_accepts_reasonable_order_identically():
    cfg = SimpleNamespace(MAX_ORDER_VALUE=5000.0)
    parity = check_risk_gate_parity(_order(1.0), 100.0, risk_config=cfg)
    assert parity.backtest_ok is True
    assert parity.live_ok is True
    assert parity.parity is True


def test_risk_gate_parity_rejects_oversized_order_identically():
    cfg = SimpleNamespace(MAX_ORDER_VALUE=5000.0)
    parity = check_risk_gate_parity(_order(100.0), 100.0, risk_config=cfg)  # 10_000 > 5_000
    assert parity.backtest_ok is False
    assert parity.live_ok is False
    assert parity.parity is True
    assert parity.reason_backtest == parity.reason_live
