from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_10bps_full_retune.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_10bps_full_retune", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_candidate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_row(
    *,
    candidate_name: str = "alpha_zoo_test_candidate",
    leverage: float = 9.0,
    allocation_fraction: float = 0.125,
    train_return: float = 0.62,
    validation_return: float = 0.44,
    locked_oos_return: float = 0.18,
    train_sharpe: float = 2.4,
    validation_sharpe: float = 1.8,
    locked_oos_sharpe: float = 1.2,
    train_sortino: float = 3.0,
    validation_sortino: float = 2.2,
    locked_oos_sortino: float = 1.5,
    train_smart_sortino: float = 2.7,
    validation_smart_sortino: float = 2.0,
    locked_oos_smart_sortino: float = 1.4,
    train_calmar: float = 4.1,
    validation_calmar: float = 3.0,
    locked_oos_calmar: float = 2.0,
    locked_oos_mdd: float = 0.08,
) -> dict[str, object]:
    return {
        "candidate_name": candidate_name,
        "leverage": leverage,
        "allocation_fraction": allocation_fraction,
        "train_return": train_return,
        "validation_return": validation_return,
        "locked_oos_return": locked_oos_return,
        "train_sharpe": train_sharpe,
        "validation_sharpe": validation_sharpe,
        "locked_oos_sharpe": locked_oos_sharpe,
        "train_sortino": train_sortino,
        "validation_sortino": validation_sortino,
        "locked_oos_sortino": locked_oos_sortino,
        "train_smart_sortino": train_smart_sortino,
        "validation_smart_sortino": validation_smart_sortino,
        "locked_oos_smart_sortino": locked_oos_smart_sortino,
        "train_calmar": train_calmar,
        "validation_calmar": validation_calmar,
        "locked_oos_calmar": locked_oos_calmar,
        "train_mdd": 0.04,
        "validation_mdd": 0.05,
        "locked_oos_mdd": locked_oos_mdd,
        "locked_oos_trade_count": 18,
        "locked_oos_liquidation_count": 0,
        "total_account_wipeout_count": 0,
    }


def _normalized_cost_model(payload: dict[str, object]) -> str:
    explicit = payload.get("cost_model")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    evidence = dict(payload.get("execution_cost_evidence") or {})
    if (
        bool(evidence.get("promotion_uses_primary_cost_only"))
        and float(payload.get("round_trip_slippage_fee_bps_primary") or 0.0) == 10.0
    ):
        return "round_trip_all_in"
    return ""


def test_build_payload_persists_10bps_report_only_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_csv = tmp_path / "candidate.csv"
    cost_validation_json = tmp_path / "cost_validation.json"
    _write_candidate_csv(candidate_csv, [_candidate_row()])
    cost_validation_json.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(MODULE, "_rss_mib", lambda: 80.0)

    payload = MODULE.build_payload(
        SimpleNamespace(
            candidate_csv=str(candidate_csv),
            cost_validation_json=str(cost_validation_json),
            output_dir=str(tmp_path / "out"),
        )
    )

    assert payload["artifact_kind"] == "alpha_zoo_10bps_full_retune"
    assert payload["round_trip_slippage_fee_bps_primary"] == pytest.approx(10.0)
    assert payload["split_manifest"]["timestamp_index_hash"] == MODULE.EXPECTED_TIMESTAMP_INDEX_HASH
    assert payload["split_manifest"]["split_contract"] == MODULE.SPLIT_CONTRACT
    assert payload["locked_oos_contamination_audit"] == {
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
    }
    assert payload["candidate_policy"]["promotion_cost_bps"] == pytest.approx(10.0)
    assert payload["candidate_policy"]["prior_oos_bucket_rows_are_shadow_only"] is True
    assert payload["candidate_policy"]["requires_train_validation_regeneration"] is True
    assert payload["candidate_policy"][
        "train_must_exceed_validation_and_locked_oos_metrics"
    ] == list(MODULE.METRIC_DOMINANCE_KEYS)
    assert payload["execution_cost_evidence"] == {
        "diagnostic_only": True,
        "primary_round_trip_slippage_fee_bps": 10.0,
        "symbols": list(MODULE.SYMBOLS),
        "promotion_uses_primary_cost_only": True,
    }
    assert _normalized_cost_model(payload) == "round_trip_all_in"
    assert payload["memory_summary"] == {
        "peak_rss_mib": pytest.approx(80.0),
        "limit_mib": pytest.approx(MODULE.MEMORY_LIMIT_MIB),
        "pass_under_8gb": True,
        "guard_status": "pass",
        "pass_fail_reason": "peak_rss_under_limit",
    }
    assert payload["live_promotable_10bps_model_id"] is None
    assert payload["live_promotable_10bps_count"] == 0
    assert len(payload["candidate_model_metrics"]) == 3
    assert {row["split"] for row in payload["candidate_model_metrics"]} == set(MODULE.SPLIT_ORDER)
    assert all(
        row["round_trip_slippage_fee_bps"] == pytest.approx(10.0)
        for row in payload["candidate_model_metrics"]
    )
    assert all(
        row["candidate_universe_uses_locked_oos_bucket"] is True
        for row in payload["candidate_model_metrics"]
    )
    assert all(row["shadow_only"] is True for row in payload["candidate_model_metrics"])
    assert all(
        row["regenerated_train_validation_only"] is False
        for row in payload["candidate_model_metrics"]
    )
    assert all(row["calendar_primary"] is False for row in payload["candidate_model_metrics"])
    assert all(row["promotion_gate_pass"] is False for row in payload["candidate_model_metrics"])


def test_promotion_gate_requires_train_validation_and_locked_oos_dominance() -> None:
    promotable_row = {
        "round_trip_slippage_fee_bps": 10.0,
        "candidate_universe_uses_locked_oos_bucket": False,
        "shadow_only": False,
        "regenerated_train_validation_only": True,
        "calendar_primary": False,
        "train_total_return": 0.62,
        "validation_total_return": 0.44,
        "locked_oos_total_return": 0.18,
        "train_sharpe": 2.4,
        "validation_sharpe": 1.8,
        "locked_oos_sharpe": 1.2,
        "train_sortino": 3.0,
        "validation_sortino": 2.2,
        "locked_oos_sortino": 1.5,
        "train_smart_sortino": 2.7,
        "validation_smart_sortino": 2.0,
        "locked_oos_smart_sortino": 1.4,
        "train_calmar": 4.1,
        "validation_calmar": 3.0,
        "locked_oos_calmar": 2.0,
        "validation_mdd": 0.05,
        "locked_oos_mdd": 0.08,
        "validation_trade_count": 16,
        "locked_oos_trade_count": 18,
        "validation_liquidation_count": 0,
        "locked_oos_liquidation_count": 0,
        "validation_account_wipeout_count": 0,
        "locked_oos_account_wipeout_count": 0,
        "minimum_margin_buffer": 1.0,
        "locked_oos_minimum_margin_buffer": 1.0,
        "total_account_wipeout_count": 0,
        "promotion_liquidation_count": 0,
    }

    gate = MODULE.promotion_gate(promotable_row)
    assert gate == {"promotion_gate_pass": True, "promotion_gate_reasons": []}

    calendar_poisoned = dict(promotable_row, calendar_primary=True)
    assert (
        "calendar_or_date_rule_forbidden"
        in MODULE.promotion_gate(calendar_poisoned)["promotion_gate_reasons"]
    )

    locked_oos_poisoned = dict(promotable_row, candidate_universe_uses_locked_oos_bucket=True)
    assert (
        "candidate_universe_uses_locked_oos_bucket"
        in MODULE.promotion_gate(locked_oos_poisoned)["promotion_gate_reasons"]
    )

    dominance_poisoned = dict(promotable_row, train_total_return=0.01, validation_total_return=0.02)
    assert (
        "train_return_not_above_validation"
        in MODULE.promotion_gate(dominance_poisoned)["promotion_gate_reasons"]
    )
