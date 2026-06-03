from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import run_alpha_zoo_69_asset_clean_oos_gate as clean_gate
from scripts.research import run_alpha_zoo_69_asset_per_asset_dynamic_pool as dynamic
from scripts.research import run_alpha_zoo_69_asset_walkforward_monitor as walkforward


def _synthetic_context() -> dynamic.DynamicContext:
    index = pd.date_range("2025-01-01", periods=96, freq="h")
    returns = np.zeros((3, len(index)), dtype=float)
    # Two rows compete for AAA. Row 0 has positive trailing performance;
    # row 1 has negative trailing performance and should be skipped by floor.
    returns[0, :48] = 0.002
    returns[0, 48:] = 0.001
    returns[1, :48] = -0.002
    returns[1, 48:] = 0.003
    returns[2, :] = 0.001
    positions = np.where(np.abs(returns) > 0.0, 1.0, 0.0)
    rows = (
        {"source_row_index": 0, "symbol": "AAA", "fit_score": 0.2},
        {"source_row_index": 1, "symbol": "AAA", "fit_score": 0.1},
        {"source_row_index": 2, "symbol": "BBB", "fit_score": 0.2},
    )
    return dynamic.DynamicContext(
        index=pd.DatetimeIndex(index),
        returns=returns,
        positions=positions,
        notionals=np.array([0.1, 0.1, 0.2], dtype=float),
        rows=rows,
        row_symbols=("AAA", "AAA", "BBB"),
        universe_symbols=("AAA", "BBB"),
        fit_scores=np.array([0.2, 0.1, 0.2], dtype=float),
        candidate_pool_policy={"candidate_pool_symbol_count": 2},
    )


def test_dynamic_weights_select_one_row_per_symbol_and_apply_gross_cap() -> None:
    context = _synthetic_context()
    weights, log = dynamic.dynamic_weights(
        context,
        dynamic.SelectorParams(
            lookback_days=1,
            rebalance_days=1,
            top_n=2,
            target_gross=1.0,
            min_trailing_return=0.0,
            fit_weight=0.0,
            vol_penalty=0.0,
            max_symbol_gross=0.3,
        ),
    )

    assert log
    assert all(len(row["active_symbols"]) <= 2 for row in log)
    assert all(row["active_symbols"].count("AAA") <= 1 for row in log)
    gross = np.sum(np.abs(weights) * context.notionals[:, None], axis=0)
    assert float(gross.max()) <= 0.600001  # two symbols * 0.3 cap
    assert np.any(weights[0] > 0.0)
    assert not np.any(weights[1, 24:48] > 0.0)


def test_evaluate_dynamic_selector_reports_negative_wfo_fold() -> None:
    context = _synthetic_context()
    windows = clean_gate.GateWindows(
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")),
        validation=(pd.Timestamp("2025-01-02 01:00"), pd.Timestamp("2025-01-03")),
        locked_oos=(pd.Timestamp("2025-01-03 01:00"), pd.Timestamp("2025-01-04")),
    )
    folds = (
        walkforward.WalkForwardFold(
            "synthetic_negative_validation",
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")),
            (pd.Timestamp("2025-01-02 01:00"), pd.Timestamp("2025-01-02 12:00")),
            (pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-03 12:00")),
        ),
    )
    evaluation = dynamic.evaluate_dynamic_selector(
        context,
        dynamic.SelectorParams(
            lookback_days=1,
            rebalance_days=1,
            top_n=1,
            target_gross=0.5,
            min_trailing_return=-1.0,
            fit_weight=0.0,
            vol_penalty=0.0,
            max_symbol_gross=0.5,
        ),
        windows,
        folds=folds,
    )

    assert set(evaluation["clean_metrics"]) == {"train", "validation", "locked_oos"}
    assert evaluation["walkforward_summary"]["fold_count"] == 1
    assert "all_validation_and_oos_positive" in evaluation["walkforward_summary"]
