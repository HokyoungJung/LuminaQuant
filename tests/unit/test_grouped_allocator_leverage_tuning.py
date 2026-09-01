import importlib.util
import sys
from pathlib import Path

import pandas as pd

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_grouped_allocator_leverage_tuning.py"
)
SPEC = importlib.util.spec_from_file_location("run_grouped_allocator_leverage_tuning", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_apply_state_leverage_marks_liquidation_and_stops_segment() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01", tz="UTC"),
                "split_group": "train",
                "state": "incumbent",
                "base_return": 0.0,
            },
            {
                "date": pd.Timestamp("2025-01-02", tz="UTC"),
                "split_group": "train",
                "state": "incumbent",
                "base_return": -0.5,
            },
            {
                "date": pd.Timestamp("2025-01-03", tz="UTC"),
                "split_group": "train",
                "state": "incumbent",
                "base_return": 0.1,
            },
            {
                "date": pd.Timestamp("2025-01-04", tz="UTC"),
                "split_group": "train",
                "state": "autoresearch_55_45",
                "base_return": 0.1,
            },
        ]
    )
    tuned, liquidations = MODULE._apply_state_leverage(
        frame,
        leverage_by_state={"incumbent": 3, "blend_85_15": 1, "autoresearch_55_45": 2},
    )
    assert liquidations["incumbent"] == 1
    returns = tuned["leveraged_return"].tolist()
    assert returns[1] <= -0.98
    assert returns[2] == 0.0
    assert returns[3] == 0.2


def test_objective_uses_train_and_val_metrics() -> None:
    metrics = {
        "train": {
            "sharpe": 1.0,
            "sortino": 1.5,
            "calmar": 2.0,
            "total_return": 0.1,
            "max_drawdown": 0.05,
            "volatility": 0.2,
        },
        "val": {
            "sharpe": 2.0,
            "sortino": 2.5,
            "calmar": 3.0,
            "total_return": 0.2,
            "max_drawdown": 0.04,
            "volatility": 0.15,
        },
    }
    value = MODULE._objective(metrics)
    assert value > 0.0
