from __future__ import annotations

import numpy as np
from lumina_quant.optimization.fast_eval import NUMBA_AVAILABLE, evaluate_metrics_numba
from lumina_quant.services.portfolio import PortfolioPerformanceService


class _Config:
    ANNUAL_PERIODS = 252


def test_fast_eval_parity_with_service_path():
    totals = [10000.0, 10010.0, 10050.0, 9950.0, 10100.0, 10200.0]
    fast = PortfolioPerformanceService.build_fast_stats(metric_totals=totals, config=_Config)

    s, c, d = evaluate_metrics_numba(np.asarray(totals, dtype=np.float64), 252)
    if NUMBA_AVAILABLE:
        assert abs(float(fast["sharpe"]) - float(s)) < 1e-9
        assert abs(float(fast["cagr"]) - float(c)) < 1e-9
        assert abs(float(fast["max_drawdown"]) - float(d)) < 1e-9
    else:
        assert abs(float(fast["cagr"]) - float(c)) < 1e-9
        assert abs(float(fast["max_drawdown"]) - float(d)) < 1e-9
        assert abs(float(fast["sharpe"]) - float(s)) < 1.0
