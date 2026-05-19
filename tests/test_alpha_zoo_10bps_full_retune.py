from __future__ import annotations

import csv
import importlib.util
import json
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


def _split_manifest() -> dict[str, object]:
    return {
        "split_contract": {
            "train": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:00:00Z"},
            "validation": {
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-03-31T23:00:00Z",
            },
            "locked_oos": {
                "start": "2026-04-01T00:00:00Z",
                "end": "2026-05-17T10:00:00Z",
            },
        },
        "timestamp_index_hash": MODULE.EXPECTED_TIMESTAMP_INDEX_HASH,
    }


def _source_metric_rows() -> list[dict[str, object]]:
    split_values = {
        "train": {
            "total_return": 0.62,
            "max_drawdown": 0.04,
            "sharpe": 2.4,
            "sortino": 3.0,
            "smart_sortino": 2.7,
            "calmar": 4.1,
        },
        "validation": {
            "total_return": 0.44,
            "max_drawdown": 0.05,
            "sharpe": 1.8,
            "sortino": 2.2,
            "smart_sortino": 2.0,
            "calmar": 3.0,
        },
        "locked_oos": {
            "total_return": 0.18,
            "max_drawdown": 0.08,
            "sharpe": 1.2,
            "sortino": 1.5,
            "smart_sortino": 1.4,
            "calmar": 2.0,
        },
    }
    rows: list[dict[str, object]] = []
    for split, values in split_values.items():
        rows.append(
            {
                "model_id": "shadow_reference_model",
                "model_kind": "individual_seed",
                "role": "historical_10bps_reference",
                "candidate_name": "alpha_zoo_test_candidate",
                "leverage": 9.0,
                "allocation_fraction": 0.125,
                "round_trip_slippage_fee_bps": 10.0,
                "split": split,
                "return_mdd": 10.0,
                "trade_event_count": 12,
                "active_return_hours": 12,
                "liquidation_count": 0,
                "account_wipeout_count": 0,
                "minimum_margin_buffer": 1.0,
                **values,
            }
        )
    return rows


def _normalized_cost_model(payload: dict[str, object]) -> str:
    explicit = payload.get("cost_model")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    evidence = dict(payload.get("execution_cost_evidence") or {})
    if (
        float(
            evidence.get(
                "primary_round_trip_cost_bps",
                evidence.get("primary_round_trip_slippage_fee_bps", 0.0),
            )
            or 0.0
        )
        == 10.0
        and float(payload.get("round_trip_slippage_fee_bps_primary") or 0.0) == 10.0
    ):
        return "round_trip_all_in"
    return ""


def test_build_payload_persists_10bps_report_only_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_csv = tmp_path / "candidate.csv"
    cost_validation_json = tmp_path / "cost_validation.json"
    source_live_json = tmp_path / "source_live.json"
    _write_candidate_csv(candidate_csv, [_candidate_row()])
    cost_validation_json.write_text(
        json.dumps(
            {
                "split_manifest": _split_manifest(),
                "selection_policy": {
                    "post_hoc_seed_basket_uses_leaderboard_oos_buckets_by_request": True
                },
                "model_cost_metrics": _source_metric_rows(),
            }
        ),
        encoding="utf-8",
    )
    source_live_json.write_text(
        json.dumps(
            {
                "selection": {
                    "strategy_summaries": [
                        {
                            "candidate_name": "alpha_zoo_test_candidate",
                            "candidate_source": "unit",
                            "params": {"entry_threshold": 1.2, "max_hold_bars": 36},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "_rss_mib", lambda: 80.0)

    payload = MODULE.build_payload(
        SimpleNamespace(
            source_candidate_csv=str(candidate_csv),
            source_cost_json=str(cost_validation_json),
            source_cost_metrics_csv=str(tmp_path / "missing_metric_rows.csv"),
            source_live_json=str(source_live_json),
            output_dir=str(tmp_path / "out"),
            fresh_candidate_limit=0,
            hybrid_seed_count=0,
            n_trials=0,
            top_n=10,
            allow_split_hash_drift=False,
        )
    )

    assert payload["artifact_kind"] == "alpha_zoo_10bps_full_retune"
    assert payload["round_trip_slippage_fee_bps_primary"] == pytest.approx(10.0)
    assert payload["split_manifest"]["timestamp_index_hash"] == MODULE.EXPECTED_TIMESTAMP_INDEX_HASH
    assert payload["split_manifest"]["split_contract"] == _split_manifest()["split_contract"]
    audit = payload["locked_oos_contamination_audit"]
    assert audit["uses_locked_oos_for_objective"] is False
    assert audit["uses_locked_oos_for_selection"] is False
    assert audit["uses_locked_oos_for_pruning"] is False
    assert audit["uses_locked_oos_for_parameter_fitting"] is False
    assert audit["locked_oos_role"] == "gate_report_only_after_candidate_freeze"
    selection_policy = payload["selection_policy"]
    assert selection_policy["optimization_input_splits"] == ["train", "validation"]
    assert selection_policy["uses_locked_oos_for_selection"] is False
    fresh_contract = payload["method_contract"]["fresh_train_validation_retune"]
    assert fresh_contract["selection_inputs"] == ["train", "validation"]
    assert fresh_contract["uses_locked_oos_for_selection"] is False
    assert fresh_contract["candidate_rows_selected_before_oos_gate"] == 0
    assert fresh_contract["evaluated_10bps_streams"] == 0
    assert fresh_contract["evaluated_trade_filter_variants"] == 0
    assert fresh_contract["selected_trade_filter_variants"] == 0
    assert fresh_contract["trade_filter_gate_pass_count"] == 0
    assert fresh_contract["trade_filter_selection_inputs"] == ["train", "validation"]
    assert fresh_contract["trade_filter_locked_oos_role"] == "gate_report_only_after_variant_freeze"
    assert fresh_contract["skipped_candidate_names"] == []
    assert payload["execution_cost_evidence"]["diagnostic_only"] is True
    assert payload["execution_cost_evidence"]["primary_round_trip_cost_bps"] == pytest.approx(
        10.0
    )
    assert payload["execution_cost_evidence"]["symbols"] == list(MODULE.EXECUTION_COST_SYMBOLS)
    assert _normalized_cost_model(payload) == "round_trip_all_in"
    assert payload["memory_summary"]["peak_rss_mib"] == pytest.approx(80.0)
    assert payload["memory_summary"]["limit_mib"] == pytest.approx(MODULE.MEMORY_LIMIT_MIB)
    assert payload["memory_summary"]["pass_under_8gb"] is True
    assert payload["memory_summary"]["guard_status"] == "pass"
    assert "below limit_mib" in payload["memory_summary"]["pass_fail_reason"]
    assert payload["live_promotable_10bps_model_id"] is None
    assert payload["no_10bps_live_ready_model"] is True
    assert payload["candidate_universe_summary"]["live_promotable_count"] == 0
    assert payload["candidate_universe_summary"]["fresh_train_validation_model_count"] == 0
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
    assert all(
        row["primary_10bps_promotion_gate_pass"] is False
        for row in payload["candidate_model_metrics"]
    )


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
