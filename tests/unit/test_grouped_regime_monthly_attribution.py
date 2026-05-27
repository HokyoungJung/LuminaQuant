import importlib.util
import sys
from pathlib import Path

import pandas as pd

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_grouped_regime_monthly_attribution.py"
)
SPEC = importlib.util.spec_from_file_location("run_grouped_regime_monthly_attribution", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_monthly_return_table_compounds_by_month() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-02-01"], utc=True),
            "split_group": ["train", "train", "val"],
            "incumbent": [0.1, 0.1, 0.0],
            "blend_85_15": [0.0, 0.0, 0.0],
            "autoresearch_55_45": [0.0, 0.0, 0.0],
            "hard_allocator": [0.0, 0.0, 0.0],
            "soft_allocator": [0.0, 0.0, 0.0],
        }
    )
    monthly = MODULE._monthly_return_table(
        frame,
        value_columns=[
            "incumbent",
            "blend_85_15",
            "autoresearch_55_45",
            "hard_allocator",
            "soft_allocator",
        ],
    )
    jan = monthly.loc[monthly["month"].eq("2025-01")].iloc[0]
    assert abs(float(jan["incumbent"]) - 0.21) < 1e-12
    assert jan["winner"] == "incumbent"
