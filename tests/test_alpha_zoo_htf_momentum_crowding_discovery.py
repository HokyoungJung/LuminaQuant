from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_htf_momentum_crowding_discovery.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_htf_momentum_crowding_discovery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _gate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "train_return": 0.10,
        "validation_return": 0.05,
        "locked_oos_return": 0.03,
        "validation_mdd": 0.05,
        "train_trade_event_count": 100,
        "validation_trade_event_count": 40,
        "locked_oos_trade_event_count": 25,
        "train_validation_return_ratio": 2.0,
        "locked_oos_liquidation_count": 0,
        "locked_oos_account_wipeout_count": 0,
        "family": "htf_trend_persistence",
        "decision": "paper_testnet_candidate_after_fill_preflight",
        "model_id": "synthetic",
        "train_return_per_turnover_proxy_bps": 12.0,
        "validation_return_per_turnover_proxy_bps": 12.0,
        "locked_oos_return_per_turnover_proxy_bps": 12.0,
    }
    row.update(overrides)
    return row


def test_candidate_score_ignores_locked_oos_return() -> None:
    high_oos = {"train_return": 0.1, "validation_return": 0.05, "validation_mdd": 0.01, "locked_oos_return": 1.0}
    low_oos = {"train_return": 0.1, "validation_return": 0.05, "validation_mdd": 0.01, "locked_oos_return": -1.0}

    assert MODULE._candidate_score(high_oos) == MODULE._candidate_score(low_oos)


def test_gate_marks_sample_pass_as_shadow_when_execution_efficiency_fails() -> None:
    row = MODULE._gate_candidate(_gate_row(validation_return_per_turnover_proxy_bps=2.0))

    assert row["backtest_sample_gate_pass"] is True
    assert row["execution_efficiency_proxy_gate_pass"] is False
    assert row["paper_candidate_gate_pass"] is False
    assert row["decision"] == "validation_alpha_shadow_until_execution_efficiency"
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False


def test_gate_allows_paper_testnet_only_when_sample_and_efficiency_pass() -> None:
    row = MODULE._gate_candidate(_gate_row())

    assert row["backtest_sample_gate_pass"] is True
    assert row["execution_efficiency_proxy_gate_pass"] is True
    assert row["paper_candidate_gate_pass"] is True
    assert row["decision"] == "paper_testnet_candidate_after_fill_preflight"
    assert row["ready_for_paper"] is True
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False


def test_simulate_symbol_charges_round_trip_cost_on_entry_and_exit() -> None:
    bars = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=4, freq="1h"),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    sim = MODULE.simulate_symbol(bars, np.array([1.0, 1.0, 0.0, 0.0]), leverage=1.0, allocation_fraction=1.0)

    assert sim.returns.sum() == -0.001
    assert not sim.liquidation_flags.any()


def test_summary_never_enables_real_money() -> None:
    rows = [MODULE._gate_candidate(_gate_row())]
    summary = MODULE._summary(rows)

    assert summary["paper_candidate_gate_pass_count"] == 1
    assert summary["ready_for_real"] is False
    assert summary["real_money_execution"] is False


def test_selected_output_rows_keeps_gate_pass_beyond_top_n() -> None:
    high_validation_reject = MODULE._gate_candidate(
        _gate_row(
            model_id="high-validation-reject",
            train_return=-0.01,
            validation_return=0.20,
            locked_oos_return=0.05,
            train_validation_return_ratio=-0.05,
        )
    )
    lower_score_sample_pass = MODULE._gate_candidate(
        _gate_row(
            model_id="lower-score-sample-pass",
            train_return=0.03,
            validation_return=0.03,
            locked_oos_return=0.03,
            validation_mdd=0.05,
        )
    )
    ranked = MODULE._rank_rows([high_validation_reject, lower_score_sample_pass])

    selected = MODULE._selected_output_rows(ranked, top_n=1)

    assert [row["model_id"] for row in selected] == ["high-validation-reject", "lower-score-sample-pass"]
    assert selected[1]["backtest_sample_gate_pass"] is True
