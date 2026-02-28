from __future__ import annotations

from lumina_quant.calibration.cost_calibration import calibrate_impact_coefficients


def test_calibration_uses_realized_impact_and_reports_error_reduction():
    fills = [
        {
            "strategy": "trend_momentum",
            "impact_bps": 5.0,
            "realized_impact_bps": 6.0,
            "realized_slippage_bps": 20.0,
        },
        {
            "strategy": "trend_momentum",
            "impact_bps": 10.0,
            "realized_impact_bps": 12.0,
            "realized_slippage_bps": 22.0,
        },
    ]
    out = calibrate_impact_coefficients(fills, {"trend_momentum": 1.0})
    payload = out["trend_momentum"]
    assert payload["new_k"] > payload["old_k"]
    assert payload["observations"] == 2
    assert payload["mae_after_bps"] <= payload["mae_before_bps"]
    assert payload["error_reduction_pct"] >= 0.0

