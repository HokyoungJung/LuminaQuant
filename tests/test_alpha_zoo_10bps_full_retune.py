from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_10bps_full_retune.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_10bps_full_retune", RUNNER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _split_row(
    model_id: str,
    split: str,
    *,
    total_return: float,
    max_drawdown: float,
    sharpe: float,
    sortino: float,
    smart_sortino: float,
    calmar: float,
    trade_count: int = 12,
    liquidation_count: int = 0,
    account_wipeout_count: int = 0,
    minimum_margin_buffer: float = 100.0,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_kind": "alpha_zoo_candidate_reference",
        "role": "shadow_reference_prior_candidate_csv",
        "candidate_name": "alpha_zoo_fast_residual",
        "leverage": 7.0,
        "allocation_fraction": 0.15,
        "round_trip_slippage_fee_bps": 10.0,
        "split": split,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "smart_sortino": smart_sortino,
        "calmar": calmar,
        "return_mdd": total_return / max_drawdown if max_drawdown else 0.0,
        "trade_event_count": trade_count,
        "active_return_hours": 8,
        "liquidation_count": liquidation_count,
        "account_wipeout_count": account_wipeout_count,
        "minimum_margin_buffer": minimum_margin_buffer,
        "candidate_universe_uses_locked_oos_bucket": True,
        "shadow_only": True,
        "regenerated_train_validation_only": False,
        "calendar_primary": False,
        "split_gate_pass": False,
        "split_gate_reasons": "shadow_only_not_live_promotable",
        "promotion_gate_pass": False,
        "promotion_gate_reasons": "shadow_only_not_live_promotable",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_cost_payload() -> dict[str, object]:
    return {
        "split_manifest": {
            "timestamp_index_hash": MODULE.EXPECTED_TIMESTAMP_INDEX_HASH,
        },
        "selection_policy": {
            "post_hoc_seed_basket_uses_leaderboard_oos_buckets_by_request": True,
        },
    }


def _source_live_payload() -> dict[str, object]:
    return {
        "selection": {
            "strategy_summaries": [
                {
                    "candidate_name": "alpha_zoo_fast_residual",
                    "candidate_source": "live_notional_risk_aligned_alpha_zoo_20260518",
                    "params": {
                        "entry_threshold": 1.0,
                        "max_hold_bars": 12,
                        "month_filter": [1, 2],
                    },
                }
            ]
        }
    }


def test_promotion_gate_requires_10bps_train_validation_frozen_candidate() -> None:
    gate = MODULE._promotion_gate_result(
        {
            "train": {
                "total_return": 0.30,
                "max_drawdown": 0.04,
                "sharpe": 2.0,
                "sortino": 2.5,
                "smart_sortino": 2.2,
                "calmar": 3.0,
            },
            "validation": {
                "total_return": 0.20,
                "max_drawdown": 0.03,
                "sharpe": 1.5,
                "sortino": 2.0,
                "smart_sortino": 1.8,
                "calmar": 2.4,
            },
            "locked_oos": {
                "total_return": 0.10,
                "max_drawdown": 0.02,
                "sharpe": 1.1,
                "sortino": 1.4,
                "smart_sortino": 1.2,
                "calmar": 1.8,
            },
        },
        cost_bps=10.0,
        candidate_universe_uses_locked_oos_bucket=False,
        regenerated_train_validation_only=True,
        promotability_scope="live_candidate",
        strict_zero_liquidation=True,
    )

    assert gate["primary_10bps_promotion_gate_pass"] is True
    assert gate["primary_10bps_promotion_gate_reasons"] == []
    assert gate["live_promotable_10bps"] is True


def test_promotion_gate_rejects_non_10bps_cost() -> None:
    gate = MODULE._promotion_gate_result(
        {
            "train": {
                "total_return": 0.30,
                "max_drawdown": 0.04,
                "sharpe": 2.0,
                "sortino": 2.5,
                "smart_sortino": 2.2,
                "calmar": 3.0,
            },
            "validation": {
                "total_return": 0.20,
                "max_drawdown": 0.03,
                "sharpe": 1.5,
                "sortino": 2.0,
                "smart_sortino": 1.8,
                "calmar": 2.4,
            },
            "locked_oos": {
                "total_return": 0.10,
                "max_drawdown": 0.02,
                "sharpe": 1.1,
                "sortino": 1.4,
                "smart_sortino": 1.2,
                "calmar": 1.8,
            },
        },
        cost_bps=5.0,
        candidate_universe_uses_locked_oos_bucket=False,
        regenerated_train_validation_only=True,
        promotability_scope="live_candidate",
        strict_zero_liquidation=True,
    )

    assert gate["primary_10bps_promotion_gate_pass"] is False
    assert "primary_cost_not_10bps" in gate["primary_10bps_promotion_gate_reasons"]


def test_variant_inventory_rejects_calendar_rules() -> None:
    row = MODULE._variant_row(
        candidate_name="alpha_zoo_fast_residual",
        variant_name="calendar_reference_rejected",
        params={"month_filter": [1, 2], "entry_threshold": 1.0},
        source="source_live_aligned",
    )

    assert row["calendar_primary"] is True
    assert row["variant_gate_pass"] is False
    assert row["calendar_rule_rejection_reasons"] == ["month_filter"]


def test_build_payload_writes_required_artifacts_and_ranks_shadow_candidates(tmp_path: Path) -> None:
    cost_json = tmp_path / "source_cost.json"
    live_json = tmp_path / "source_live.json"
    candidate_csv = tmp_path / "candidates.csv"
    metrics_csv = tmp_path / "source_metrics.csv"

    cost_json.write_text(json.dumps(_source_cost_payload()), encoding="utf-8")
    live_json.write_text(json.dumps(_source_live_payload()), encoding="utf-8")

    rows = [
        _split_row(
            "model_a",
            "train",
            total_return=0.30,
            max_drawdown=0.05,
            sharpe=1.8,
            sortino=2.1,
            smart_sortino=2.0,
            calmar=3.0,
        ),
        _split_row(
            "model_a",
            "validation",
            total_return=0.20,
            max_drawdown=0.04,
            sharpe=1.5,
            sortino=1.8,
            smart_sortino=1.7,
            calmar=2.5,
        ),
        _split_row(
            "model_a",
            "locked_oos",
            total_return=0.10,
            max_drawdown=0.03,
            sharpe=1.2,
            sortino=1.4,
            smart_sortino=1.3,
            calmar=2.0,
        ),
    ]
    _write_csv(metrics_csv, rows)
    candidate_csv.write_text(
        "candidate_name,leverage,allocation_fraction\nalpha_zoo_fast_residual,7,0.15\n",
        encoding="utf-8",
    )

    args = MODULE.parse_args(
        [
            "--source-cost-json",
            str(cost_json),
            "--source-cost-metrics-csv",
            str(metrics_csv),
            "--source-live-json",
            str(live_json),
            "--source-candidate-csv",
            str(candidate_csv),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    payload = MODULE.build_payload(args)
    output_paths = MODULE.write_outputs(payload, Path(args.output_dir))

    assert payload["round_trip_slippage_fee_bps_primary"] == 10.0
    assert payload["method_contract"]["historical_top_bucket_rows_shadow_only"] is True
    assert payload["candidate_universe_summary"]["model_count"] == 1
    assert payload["candidate_universe_summary"]["live_promotable_count"] == 0
    assert payload["no_10bps_live_ready_model"] is True
    assert payload["best_shadow_10bps_model"]["model_id"] == "model_a"
    assert payload["locked_oos_contamination_audit"] == {
        "uses_locked_oos_for_objective": False,
        "uses_locked_oos_for_pruning": False,
        "uses_locked_oos_for_selection": False,
        "uses_locked_oos_for_parameter_fitting": False,
        "locked_oos_role": "gate_report_only_after_candidate_freeze",
        "evidence": [
            "train_validation_score reads train and validation metrics only",
            "primary promotion gate reads locked-OOS only after model rows are frozen",
            "historical OOS/top-bucket seed-union rows are marked shadow_only",
        ],
    }
    assert len(payload["candidate_variant_inventory"]) >= 1
    assert len(payload["tuned_seed_selection"]["rows"]) == 1
    assert payload["tuned_seed_selection"]["rows"][0]["model_id"] == "model_a"

    expected_files = [
        "latest_json",
        "latest_markdown",
        "candidate_model_metrics_csv",
        "candidate_variant_inventory_csv",
        "tuned_seed_selection_json",
        "tuned_seed_selection_csv",
        "hybrid_weights_csv",
        "execution_cost_evidence_json",
    ]
    for key in expected_files:
        assert Path(output_paths[key]).exists(), key


def test_build_payload_includes_execution_cost_symbol_set(tmp_path: Path) -> None:
    cost_json = tmp_path / "source_cost.json"
    live_json = tmp_path / "source_live.json"
    candidate_csv = tmp_path / "candidates.csv"
    metrics_csv = tmp_path / "source_metrics.csv"

    cost_json.write_text(json.dumps(_source_cost_payload()), encoding="utf-8")
    live_json.write_text(json.dumps(_source_live_payload()), encoding="utf-8")
    _write_csv(
        metrics_csv,
        [
            _split_row(
                "model_a",
                "train",
                total_return=0.30,
                max_drawdown=0.05,
                sharpe=1.8,
                sortino=2.1,
                smart_sortino=2.0,
                calmar=3.0,
            ),
            _split_row(
                "model_a",
                "validation",
                total_return=0.20,
                max_drawdown=0.04,
                sharpe=1.5,
                sortino=1.8,
                smart_sortino=1.7,
                calmar=2.5,
            ),
            _split_row(
                "model_a",
                "locked_oos",
                total_return=0.10,
                max_drawdown=0.03,
                sharpe=1.2,
                sortino=1.4,
                smart_sortino=1.3,
                calmar=2.0,
            ),
        ],
    )
    candidate_csv.write_text(
        "candidate_name,leverage,allocation_fraction\nalpha_zoo_fast_residual,7,0.15\n",
        encoding="utf-8",
    )

    args = MODULE.parse_args(
        [
            "--source-cost-json",
            str(cost_json),
            "--source-cost-metrics-csv",
            str(metrics_csv),
            "--source-live-json",
            str(live_json),
            "--source-candidate-csv",
            str(candidate_csv),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    payload = MODULE.build_payload(args)

    assert payload["execution_cost_evidence"]["diagnostic_only"] is True
    assert set(payload["execution_cost_evidence"]["symbols"]) == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "TRXUSDT",
    }
