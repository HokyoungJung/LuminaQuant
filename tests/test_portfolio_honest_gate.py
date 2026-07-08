"""Portfolio-layer honest gate + cross-run trial accountant (data-free).

Covers the linchpin of the honest measurement foundation
(performance_lever_measurement 2026-07-08):

* ``evaluate_weighted_portfolio`` is BYTE-IDENTICAL with the gate OFF (the shipped
  default) -- no new key, same dict.
* ``count_config_grid_trials`` is the cross-run trial accountant (product of grid
  axes).
* The honesty proof: a synthetic "best-of-many" config cell -- a return stream
  selected as the MAX in-sample Sharpe over ``N`` noise trials -- is REJECTED once
  the portfolio DSR is deflated by ``num_trials=N``, while a genuine single-config
  stream (``num_trials=1``) with real edge PASSES. The accountant is shown to be
  load-bearing: the very same best-of-many stream would PASS at ``num_trials=1``.

No market data is needed; every stream is synthetic and seeded.
"""

from __future__ import annotations

import numpy as np

from lumina_quant import portfolio_followup_rules as P

# The strict-research profile floors (configs/profiles/*.yaml).
FLOORS = {
    "dsr_gate_floor": 0.90,
    "spa_gate_ceiling": 0.05,
    "pbo_gate_ceiling": 0.50,
}


def test_evaluate_weighted_portfolio_off_path_is_byte_identical():
    rows = [
        {
            "_saved_weight": 0.6,
            "return_streams": {
                "oos": [
                    {"datetime": f"2025-03-{d:02d}T00:00:00Z", "v": 0.002} for d in range(1, 20)
                ]
            },
        },
        {
            "_saved_weight": 0.4,
            "return_streams": {
                "oos": [
                    {"datetime": f"2025-03-{d:02d}T00:00:00Z", "v": 0.001} for d in range(1, 20)
                ]
            },
        },
    ]
    base = P.evaluate_weighted_portfolio(rows)
    off = P.evaluate_weighted_portfolio(rows, honest_gate=False, num_trials=1)
    assert base == off
    assert "portfolio_honest_gate" not in base

    on = P.evaluate_weighted_portfolio(rows, honest_gate=True, num_trials=1, **FLOORS)
    # The ON path is a strict superset: every OFF key is present and unchanged.
    assert "portfolio_honest_gate" in on
    for key, value in base.items():
        assert on[key] == value


def test_count_config_grid_trials_counts_every_cell():
    assert P.count_config_grid_trials(None) == 1
    assert P.count_config_grid_trials([]) == 1
    assert P.count_config_grid_trials({}) == 1
    # band x min_hold x allocator x regime_threshold x order_policy x funding_window
    axes = {
        "band_min_hold": 6,
        "allocator": 3,
        "regime_threshold": 4,
        "order_policy": 2,
        "funding_window": 2,
    }
    assert P.count_config_grid_trials(axes) == 6 * 3 * 4 * 2 * 2
    # Non-positive / non-int axes are ignored, never counted as a cell.
    assert P.count_config_grid_trials([5, 0, 3, None]) == 15


def test_portfolio_honest_gate_no_op_at_shipped_defaults():
    # Default floors (0.0 / 1.0 / 1.0) + num_trials=1 never reject.
    rng = np.random.default_rng(3)
    stream = 0.001 + 0.01 * rng.standard_normal(200)
    report = P.portfolio_honest_gate_report(stream)
    assert report["passed"] is True
    assert report["reject_reasons"] == []
    assert report["num_trials"] == 1


def _genuine_single_config_stream() -> np.ndarray:
    rng = np.random.default_rng(7)
    return 0.0015 + 0.004 * rng.standard_normal(300)


def _best_of_many_stream(num_trials: int) -> np.ndarray:
    """Select the MAX in-sample-Sharpe stream over ``num_trials`` pure-noise trials."""
    rng = np.random.default_rng(101)
    best_stream: np.ndarray | None = None
    best_sharpe = -np.inf
    for _ in range(num_trials):
        candidate = 0.006 * rng.standard_normal(300)
        sharpe = float(candidate.mean() / (candidate.std() + 1e-12))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_stream = candidate
    assert best_stream is not None
    return best_stream


def test_genuine_single_config_stream_passes():
    report = P.portfolio_honest_gate_report(_genuine_single_config_stream(), num_trials=1, **FLOORS)
    assert report["passed"] is True, report
    assert report["deflated_sharpe"] >= FLOORS["dsr_gate_floor"]
    assert report["spa_pvalue"] <= FLOORS["spa_gate_ceiling"]
    assert report["pbo"] <= FLOORS["pbo_gate_ceiling"]


def test_best_of_many_is_rejected_once_deflated_by_num_trials():
    num_trials = 60
    stream = _best_of_many_stream(num_trials)
    deflated = P.portfolio_honest_gate_report(stream, num_trials=num_trials, **FLOORS)
    assert deflated["passed"] is False, deflated
    assert "dsr_below_floor" in deflated["reject_reasons"]
    assert deflated["num_trials"] == num_trials


def test_trial_accountant_is_load_bearing():
    # The SAME best-of-many stream that the accountant rejects would PASS the DSR
    # floor if it were (dishonestly) scored as a single trial. This proves the
    # cross-run accountant -- not the raw Sharpe -- is what stops config-grid
    # overfitting.
    num_trials = 60
    stream = _best_of_many_stream(num_trials)
    naive = P.portfolio_honest_gate_report(stream, num_trials=1, **FLOORS)
    deflated = P.portfolio_honest_gate_report(stream, num_trials=num_trials, **FLOORS)
    assert naive["deflated_sharpe"] >= FLOORS["dsr_gate_floor"]
    assert deflated["deflated_sharpe"] < FLOORS["dsr_gate_floor"]
    assert naive["passed"] is True
    assert deflated["passed"] is False


def _dated_oos_rows(values: np.ndarray, weight: float = 1.0) -> list[dict]:
    stream = [
        {"datetime": f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}T00:00:00Z", "v": float(v)}
        for i, v in enumerate(values)
    ]
    return [{"_saved_weight": weight, "return_streams": {"oos": stream}}]


def test_uplift_gate_reports_marginal_edge_stream():
    n = 120
    rng = np.random.default_rng(19)
    baseline_daily = 0.0005 + 0.003 * rng.standard_normal(n)
    uplift_daily = 0.0015 + 0.002 * rng.standard_normal(n)
    off_rows = _dated_oos_rows(baseline_daily)
    on_rows = _dated_oos_rows(baseline_daily + uplift_daily)

    report = P.portfolio_uplift_gate_report(on_rows, off_rows, num_trials=1, **FLOORS)
    assert report["stream"] == "uplift"
    assert report["aligned_days"] == n
    # A real positive marginal edge clears the honest gate at a single config.
    assert report["passed"] is True, report

    # A zero uplift (ON == OFF) has no edge -> rejected.
    flat = P.portfolio_uplift_gate_report(off_rows, off_rows, num_trials=1, **FLOORS)
    assert flat["passed"] is False
