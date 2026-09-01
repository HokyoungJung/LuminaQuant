from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "scripts" / "research" / "optuna_tune_profit_moonshot_calendar.py"
SPEC = importlib.util.spec_from_file_location("optuna_tune_profit_moonshot_calendar", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load optuna_tune_profit_moonshot_calendar module")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from scripts.research import replay_profit_moonshot_fresh_start as FRESH


def test_optuna_calendar_spec_uses_locked_trx_calendar_contract() -> None:
    spec = MODULE._spec_from_params(
        FRESH,
        {
            "short_symbol": "ETHUSDT",
            "threshold": 0.015,
            "hold_bars": 120,
            "long_scale": 6.2,
            "short_scale": 10.0,
            "take_profit_pct": 0.018,
        },
        trial_number=7,
    )

    assert spec.family == "calendar_rotation"
    assert spec.lookback_bars == 168
    assert spec.calendar_long_symbol == "TRXUSDT"
    assert spec.calendar_short_symbol == "ETHUSDT"
    assert spec.calendar_long_months == (3, 4, 5)
    assert spec.calendar_short_months == (1, 2)
    assert spec.long_allocation_scale == 6.2
    assert spec.short_allocation_scale == 10.0
    assert spec.take_profit_pct == 0.018


def test_optuna_objective_is_train_validation_only() -> None:
    result = {
        "split_results": {
            "train": {"metrics": {"total_return": 0.02, "sharpe": 1.5, "max_drawdown": 0.01}},
            "val": {"metrics": {"total_return": 0.01, "sharpe": 2.0, "max_drawdown": 0.005}},
            "oos": {"metrics": {"total_return": -0.99, "sharpe": -50.0, "max_drawdown": 0.9}},
        }
    }
    better_oos_only = {
        "split_results": {
            "train": {"metrics": {"total_return": 0.02, "sharpe": 1.5, "max_drawdown": 0.01}},
            "val": {"metrics": {"total_return": 0.01, "sharpe": 2.0, "max_drawdown": 0.005}},
            "oos": {"metrics": {"total_return": 99.0, "sharpe": 500.0, "max_drawdown": 0.0}},
        }
    }

    assert MODULE._objective_score(result) == MODULE._objective_score(better_oos_only)
