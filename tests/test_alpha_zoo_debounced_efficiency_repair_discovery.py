from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "research" / "run_alpha_zoo_debounced_efficiency_repair_discovery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_debounced_efficiency_repair_discovery", MODULE_PATH
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
        "family": "debounced_momentum_hysteresis_efficiency_repair",
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
        "train_return_per_turnover_proxy_bps": 11.0,
        "validation_return_per_turnover_proxy_bps": 12.0,
        "validation_trade_event_count": 40,
    }
    low_oos = dict(high_oos, locked_oos_return=-1.0)

    assert MODULE._candidate_score(high_oos) == MODULE._candidate_score(low_oos)


def test_gate_rejects_train_below_validation_as_spike() -> None:
    row = MODULE._gate_candidate(
        _gate_row(train_return=0.03, validation_return=0.04, train_validation_return_ratio=0.75)
    )

    assert row["train_dominant_sample_gate_pass"] is False
    assert row["paper_candidate_gate_pass"] is False
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False
    assert any("below_validation_return" in reason for reason in row["rejection_reasons"])


def test_gate_requires_all_split_return_per_turnover_above_10bps() -> None:
    row = MODULE._gate_candidate(_gate_row(locked_oos_return_per_turnover_proxy_bps=9.99))

    assert row["train_dominant_sample_gate_pass"] is True
    assert row["execution_efficiency_proxy_gate_pass"] is False
    assert row["primary_10bps_promotion_gate_pass"] is False
    assert any(
        "locked_oos_return_per_turnover_proxy_bps_9.990" in r for r in row["rejection_reasons"]
    )


def test_gate_allows_paper_testnet_only_when_all_strict_gates_pass() -> None:
    row = MODULE._gate_candidate(_gate_row())

    assert row["train_dominant_sample_gate_pass"] is True
    assert row["execution_efficiency_proxy_gate_pass"] is True
    assert row["paper_candidate_gate_pass"] is True
    assert row["primary_10bps_promotion_gate_pass"] is True
    assert row["decision"] == "paper_testnet_candidate_after_fill_preflight"
    assert row["ready_for_paper"] is True
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False


def test_debounced_state_signal_enforces_min_hold_and_cooldown() -> None:
    idx = pd.RangeIndex(6)
    signal = MODULE._debounced_state_signal(
        pd.Series([False, True, False, False, True, True], index=idx),
        pd.Series([False, False, True, True, False, False], index=idx),
        side="long_only",
        min_hold_bars=2,
        cooldown_bars=2,
    )

    assert signal.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]


def test_handoff_and_no_promotion_artifacts_are_fail_closed() -> None:
    handoff = MODULE._paper_testnet_handoff([])
    shortlist = MODULE._no_promotion_shortlist(
        [MODULE._gate_candidate(_gate_row(model_id="shadow"))], limit=1
    )

    assert handoff["status"] == "no_paper_candidates"
    assert handoff["ready_for_real"] is False
    assert handoff["real_money_execution"] is False
    assert shortlist["status"] == "no_new_paper_promotion_shadow_shortlist"
    assert shortlist["ready_for_real"] is False
    assert shortlist["real_money_execution"] is False
    assert shortlist["baseline_lanes_preserved"]


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
        _gate_row(model_id="lower-score-gate-pass", train_return=0.05, validation_return=0.03)
    )
    ranked = MODULE._rank_rows([high_validation_reject, lower_score_gate_pass])

    selected = MODULE._selected_output_rows(ranked, top_n=1)

    assert [row["model_id"] for row in selected] == [
        "high-validation-reject",
        "lower-score-gate-pass",
    ]
    assert selected[1]["paper_candidate_gate_pass"] is True
