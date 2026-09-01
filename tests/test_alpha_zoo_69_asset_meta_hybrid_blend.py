from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research import run_alpha_zoo_69_asset_clean_oos_gate as clean_gate
from scripts.research import run_alpha_zoo_69_asset_meta_hybrid_blend as meta
from scripts.research import run_alpha_zoo_69_asset_walkforward_monitor as walkforward


def test_blend_return_series_aligns_and_weights() -> None:
    dynamic = pd.Series([0.02, 0.02], index=pd.to_datetime(["2025-01-01", "2025-01-02"]))
    core = pd.Series([0.01, 0.03], index=pd.to_datetime(["2025-01-02", "2025-01-03"]))

    blended = meta.blend_return_series(dynamic, core, 0.75)

    assert list(blended.index) == list(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))
    np.testing.assert_allclose(blended.to_numpy(), [0.015, 0.0175, 0.0075])


def test_candidate_score_prefers_all_positive_walkforward_candidate() -> None:
    passing = {
        "dynamic_weight": 0.7,
        "max_oos_mdd_gate": 0.2,
        "clean_metrics": {"locked_oos": {"total_return": 0.1}},
        "walkforward_summary": {
            "all_validation_and_oos_positive": True,
            "min_validation_return": 0.01,
            "min_oos_return": 0.05,
            "max_oos_mdd": 0.08,
        },
    }
    failing = {
        **passing,
        "clean_metrics": {"locked_oos": {"total_return": 0.2}},
        "walkforward_summary": {
            **passing["walkforward_summary"],
            "all_validation_and_oos_positive": False,
        },
    }

    assert meta.candidate_score(passing) > 0.0
    assert meta.candidate_score(failing) < -1e8


def test_evaluate_return_series_reports_walkforward_summary() -> None:
    index = pd.date_range("2025-01-01", "2026-05-06 23:00", freq="h")
    returns = pd.Series(0.0001, index=index)
    windows = clean_gate.GateWindows(
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31 23:00")),
        validation=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:00")),
        locked_oos=(pd.Timestamp("2026-03-01"), pd.Timestamp("2026-05-06 23:00")),
    )

    result = meta.evaluate_return_series(returns, windows, folds=walkforward.DEFAULT_FOLDS)

    assert set(result["clean_metrics"]) == {"train", "validation", "locked_oos"}
    assert result["walkforward_summary"]["fold_count"] == len(walkforward.DEFAULT_FOLDS)
    assert result["walkforward_summary"]["all_validation_and_oos_positive"] is True
