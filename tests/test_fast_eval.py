from __future__ import annotations

import numpy as np
from lumina_quant.optimization.fast_eval import evaluate_metrics_numba


def test_evaluate_metrics_numba_returns_finite_tuple():
    totals = np.asarray([10000.0, 10100.0, 9900.0, 10250.0, 10300.0], dtype=np.float64)
    sharpe, cagr, max_dd = evaluate_metrics_numba(totals, 252)
    assert np.isfinite(float(sharpe))
    assert np.isfinite(float(cagr))
    assert np.isfinite(float(max_dd))
    assert float(max_dd) >= 0.0
