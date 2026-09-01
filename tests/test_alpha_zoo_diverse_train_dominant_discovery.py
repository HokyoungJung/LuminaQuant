from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_diverse_train_dominant_discovery.py"
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_diverse_train_dominant_discovery", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _gate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "train_return": 0.08,
        "validation_return": 0.04,
        "locked_oos_return": 0.03,
        "validation_mdd": 0.05,
        "train_trade_event_count": 100,
        "validation_trade_event_count": 40,
        "locked_oos_trade_event_count": 25,
        "train_validation_return_ratio": 2.0,
        "locked_oos_liquidation_count": 0,
        "locked_oos_account_wipeout_count": 0,
        "family": "stateful_momentum_hysteresis",
        "decision": "paper_testnet_candidate_after_fill_preflight",
        "model_id": "synthetic",
        "train_return_per_turnover_proxy_bps": 12.0,
        "validation_return_per_turnover_proxy_bps": 12.0,
        "locked_oos_return_per_turnover_proxy_bps": 12.0,
    }
    row.update(overrides)
    return row


def test_candidate_score_ignores_locked_oos_return() -> None:
    high_oos = {
        "train_return": 0.08,
        "validation_return": 0.04,
        "validation_mdd": 0.01,
        "locked_oos_return": 1.0,
    }
    low_oos = {
        "train_return": 0.08,
        "validation_return": 0.04,
        "validation_mdd": 0.01,
        "locked_oos_return": -1.0,
    }

    assert MODULE._candidate_score(high_oos) == MODULE._candidate_score(low_oos)


def test_gate_rejects_train_below_validation_even_when_other_guards_pass() -> None:
    row = MODULE._gate_candidate(
        _gate_row(train_return=0.03, validation_return=0.04, train_validation_return_ratio=0.75)
    )

    assert row["train_dominant_sample_gate_pass"] is False
    assert row["paper_candidate_gate_pass"] is False
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False
    assert any("below_validation_return" in reason for reason in row["rejection_reasons"])


def test_gate_allows_paper_testnet_only_when_train_dominant_and_efficiency_pass() -> None:
    row = MODULE._gate_candidate(_gate_row())

    assert row["train_dominant_sample_gate_pass"] is True
    assert row["execution_efficiency_proxy_gate_pass"] is True
    assert row["paper_candidate_gate_pass"] is True
    assert row["decision"] == "paper_testnet_candidate_after_fill_preflight"
    assert row["ready_for_paper"] is True
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False


def test_stateful_signal_holds_until_exit() -> None:
    idx = pd.RangeIndex(5)
    signal = MODULE._stateful_signal(
        pd.Series([False, True, False, False, False], index=idx),
        pd.Series([False, False, False, True, False], index=idx),
        side="long_only",
    )

    assert signal.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_selected_output_rows_keeps_gate_pass_beyond_top_n() -> None:
    high_validation_reject = MODULE._gate_candidate(
        _gate_row(
            model_id="high-validation-reject",
            train_return=0.05,
            validation_return=0.20,
            train_validation_return_ratio=0.25,
        )
    )
    lower_score_gate_pass = MODULE._gate_candidate(
        _gate_row(
            model_id="lower-score-gate-pass",
            train_return=0.05,
            validation_return=0.03,
            validation_mdd=0.05,
        )
    )
    ranked = MODULE._rank_rows([high_validation_reject, lower_score_gate_pass])

    selected = MODULE._selected_output_rows(ranked, top_n=1)

    assert [row["model_id"] for row in selected] == [
        "high-validation-reject",
        "lower-score-gate-pass",
    ]
    assert selected[1]["train_dominant_sample_gate_pass"] is True
