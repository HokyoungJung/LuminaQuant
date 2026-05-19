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

EXPECTED_SPLIT_CONTRACT = {
    "train": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T23:00:00Z"},
    "validation": {"start": "2026-01-01T00:00:00Z", "end": "2026-03-31T23:00:00Z"},
    "locked_oos": {"start": "2026-04-01T00:00:00Z", "end": "2026-05-17T10:00:00Z"},
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(
    *,
    model_id: str = "candidate_a",
    cost_bps: float = 10.0,
) -> list[dict[str, object]]:
    split_rows = [
        ("train", 0.62, 0.04, 2.4, 3.0, 2.7, 4.1, 20),
        ("validation", 0.44, 0.05, 1.8, 2.2, 2.0, 3.0, 16),
        ("locked_oos", 0.18, 0.08, 1.2, 1.5, 1.4, 2.0, 18),
    ]
    return [
        {
            "model_id": model_id,
            "model_kind": "seed",
            "role": "source_cost_validation",
            "candidate_name": "alpha_zoo_test_candidate",
            "leverage": 9.0,
            "allocation_fraction": 0.125,
            "round_trip_slippage_fee_bps": cost_bps,
            "split": split,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "smart_sortino": smart_sortino,
            "calmar": calmar,
            "return_mdd": total_return / max_drawdown,
            "trade_event_count": trade_count,
            "active_return_hours": trade_count,
            "liquidation_count": 0,
            "account_wipeout_count": 0,
            "minimum_margin_buffer": 1.0,
        }
        for split, total_return, max_drawdown, sharpe, sortino, smart_sortino, calmar, trade_count in split_rows
    ]


def _split_metrics() -> dict[str, dict[str, object]]:
    return MODULE._split_metrics_from_rows(_metric_rows())


def test_build_payload_persists_10bps_report_only_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cost_json = tmp_path / "source_cost.json"
    source_cost_metrics_csv = tmp_path / "source_cost_metrics.csv"
    source_live_json = tmp_path / "source_live.json"
    source_candidate_csv = tmp_path / "missing_candidates.csv"
    _write_json(
        source_cost_json,
        {
            "split_manifest": {
                "split_contract": EXPECTED_SPLIT_CONTRACT,
                "timestamp_index_hash": MODULE.EXPECTED_TIMESTAMP_INDEX_HASH,
            },
            "selection_policy": {
                "post_hoc_seed_basket_uses_leaderboard_oos_buckets_by_request": True
            },
        },
    )
    _write_csv(source_cost_metrics_csv, _metric_rows())
    _write_json(
        source_live_json,
        {
            "selection": {
                "strategy_summaries": [
                    {
                        "candidate_name": "alpha_zoo_test_candidate",
                        "candidate_source": "unit_fixture",
                        "params": {
                            "entry_threshold": 1.2,
                            "exit_threshold": 0.5,
                            "max_hold_bars": 24,
                        },
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(MODULE, "_rss_mib", lambda: 80.0)

    payload = MODULE.build_payload(
        SimpleNamespace(
            source_cost_json=str(source_cost_json),
            source_cost_metrics_csv=str(source_cost_metrics_csv),
            source_live_json=str(source_live_json),
            source_candidate_csv=str(source_candidate_csv),
            output_dir=str(tmp_path / "out"),
            n_trials=80,
            top_n=50,
            allow_split_hash_drift=False,
        )
    )

    assert payload["artifact_kind"] == "alpha_zoo_10bps_full_retune"
    assert payload["real_money_execution"] is False
    assert payload["round_trip_slippage_fee_bps_primary"] == pytest.approx(10.0)
    assert payload["split_manifest"]["timestamp_index_hash"] == MODULE.EXPECTED_TIMESTAMP_INDEX_HASH
    assert payload["split_manifest"]["split_contract"] == EXPECTED_SPLIT_CONTRACT
    assert payload["locked_oos_contamination_audit"]["uses_locked_oos_for_objective"] is False
    assert payload["locked_oos_contamination_audit"]["uses_locked_oos_for_selection"] is False
    assert payload["locked_oos_contamination_audit"]["uses_locked_oos_for_pruning"] is False
    assert (
        payload["locked_oos_contamination_audit"]["uses_locked_oos_for_parameter_fitting"] is False
    )
    assert payload["locked_oos_contamination_audit"]["locked_oos_role"] == (
        "gate_report_only_after_candidate_freeze"
    )
    assert payload["execution_cost_evidence"]["diagnostic_only"] is True
    assert set(payload["execution_cost_evidence"]["symbols"]) == set(MODULE.EXECUTION_COST_SYMBOLS)
    assert payload["memory_summary"]["limit_mib"] == pytest.approx(MODULE.MEMORY_LIMIT_MIB)
    assert payload["memory_summary"]["guard_status"] == "pass"
    assert payload["memory_summary"]["pass_under_8gb"] is True
    assert payload["live_promotable_10bps_model_id"] is None
    assert payload["no_10bps_live_ready_model"] is True
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
    assert all(row["live_promotable_10bps"] is False for row in payload["candidate_model_metrics"])
    assert all(
        "candidate_universe_uses_locked_oos_bucket" in row["primary_10bps_promotion_gate_reasons"]
        for row in payload["candidate_model_metrics"]
    )
    assert all(row["calendar_primary"] is False for row in payload["candidate_variant_inventory"])


def test_primary_promotion_gate_requires_train_validation_and_locked_oos_dominance() -> None:
    gate = MODULE.primary_promotion_gate(
        _split_metrics(),
        cost_bps=10.0,
        candidate_universe_uses_locked_oos_bucket=False,
        regenerated_train_validation_only=True,
        promotability_scope="live_candidate",
    )
    assert gate["live_promotable_10bps"] is True
    assert gate["primary_10bps_promotion_gate_reasons"] == []

    locked_oos_poisoned = MODULE.primary_promotion_gate(
        _split_metrics(),
        cost_bps=10.0,
        candidate_universe_uses_locked_oos_bucket=True,
        regenerated_train_validation_only=False,
        promotability_scope="shadow_only",
    )
    assert (
        "candidate_universe_uses_locked_oos_bucket"
        in locked_oos_poisoned["primary_10bps_promotion_gate_reasons"]
    )
    assert (
        "shadow_only_historical_oos_bucket_lineage"
        in locked_oos_poisoned["primary_10bps_promotion_gate_reasons"]
    )

    dominance_poisoned = _split_metrics()
    dominance_poisoned["train"]["total_return"] = 0.01
    assert (
        "train_total_return_not_above_validation"
        in MODULE.primary_promotion_gate(
            dominance_poisoned,
            cost_bps=10.0,
            candidate_universe_uses_locked_oos_bucket=False,
            regenerated_train_validation_only=True,
            promotability_scope="live_candidate",
        )["primary_10bps_promotion_gate_reasons"]
    )

    assert MODULE._variant_has_calendar_rule({"month_filter": [1, 2]}) is True
