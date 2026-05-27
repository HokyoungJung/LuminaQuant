from pathlib import Path
import importlib.util
import sys

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "run_profit_moonshot_cadence_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("run_profit_moonshot_cadence_sweep", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
cadence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cadence
SPEC.loader.exec_module(cadence)


def test_cached_negative_equity_metrics_are_capped_with_raw_diagnostics() -> None:
    metrics = cadence._normalize_metrics_payload(
        {
            "total_return": -1.1158936570801055,
            "max_drawdown": 1.757880422159876,
            "sharpe": 0.0120102809288089,
        },
        final_equity=-1158.936570801055,
    )

    assert metrics["equity_breach_observed"] is True
    assert metrics["total_return"] == -1.0
    assert metrics["max_drawdown"] == 1.0
    assert metrics["sharpe"] == 0.0
    assert metrics["raw_total_return"] == -1.1158936570801055
    assert metrics["raw_max_drawdown"] == 1.757880422159876


def test_full_gate_rejects_equity_breach_even_after_capping() -> None:
    pass_gate, reasons = cadence._full_gate(
        {
            "train": {
                "status": "completed",
                "metrics": {
                    "total_return": -1.1158936570801055,
                    "max_drawdown": 1.757880422159876,
                },
                "trade_count": 10,
                "liquidation_count": 0,
                "final_equity": -1158.936570801055,
            },
            "val": {
                "status": "completed",
                "metrics": {"total_return": 0.01, "max_drawdown": 0.001, "sharpe": 0.5},
                "trade_count": 10,
                "liquidation_count": 0,
                "final_equity": 10100.0,
            },
            "oos": {
                "status": "completed",
                "metrics": {"total_return": 0.02, "max_drawdown": 0.001, "sharpe": 1.5},
                "trade_count": 10,
                "liquidation_count": 0,
                "final_equity": 10200.0,
            },
        }
    )

    assert pass_gate is False
    assert "train_equity_breach_observed" in reasons
