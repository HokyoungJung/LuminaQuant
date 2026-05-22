from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_paper_fill_efficiency_gate.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_zoo_paper_fill_efficiency_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _candidate_row(*, ready_for_paper: bool = False) -> dict[str, object]:
    return {
        "model_id": "candidate_pass" if ready_for_paper else "candidate_reject",
        "candidate_name": "alpha_zoo_quality_single_pair",
        "train_return": 0.20,
        "validation_return": 0.05,
        "locked_oos_return": 0.03,
        "train_trade_event_count": 100,
        "validation_trade_event_count": 40,
        "locked_oos_trade_event_count": 25,
        "validation_mdd": 0.05,
        "train_validation_return_ratio": 4.0,
        "primary_10bps_promotion_gate_pass": ready_for_paper,
        "execution_efficiency_proxy_gate_pass": ready_for_paper,
        "ready_for_paper": ready_for_paper,
        "ready_for_real": False,
        "real_money_execution": False,
        "rejection_reasons": [] if ready_for_paper else ["validation_return_below_threshold"],
    }


def _write_sources(
    tmp_path: Path,
    *,
    candidates: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    sample = tmp_path / "sample_guarded.json"
    sample.write_text(
        json.dumps(
            {
                "artifact_kind": "alpha_zoo_sample_guarded_alpha_discovery",
                "real_money_execution": False,
                "decision": {"status": "no_new_paper_promotion_shadow_shortlist"},
                "sample_guarded_candidates": candidates if candidates is not None else [_candidate_row()],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monitoring = tmp_path / "monitoring.json"
    monitoring.write_text(
        json.dumps(
            {
                "artifact_kind": "alpha_zoo_paper_forward_monitoring_contract",
                "status": "pending_paper_forward_fills",
                "real_money_execution": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sample, monitoring


def _fill_row(*, pnl: float, spread: float = 2.0, all_in_cost: float = 5.0) -> dict[str, object]:
    return {
        "env": "paper",
        "symbol": "BTC/USDT",
        "side": "BUY",
        "notional": 1000.0,
        "realized_pnl_quote": pnl,
        "spread_bps_at_submit": spread,
        "all_in_cost_bps": all_in_cost,
        "timeout_flag": False,
        "cancel_flag": False,
        "partial_fill_flag": False,
        "liquidation_count": 0,
        "account_wipeout_count": 0,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_paper_fill_efficiency_gate_fails_closed_without_fills(tmp_path: Path) -> None:
    sample, monitoring = _write_sources(tmp_path)
    payload = MODULE.build_payload(
        MODULE.parse_args(
            [
                "--sample-guarded-json",
                str(sample),
                "--monitoring-contract-json",
                str(monitoring),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )

    summary = payload["actual_fill_efficiency_summary"]
    assert payload["artifact_kind"] == "alpha_zoo_paper_fill_efficiency_gate"
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["paper_execution_allowed"] is False
    assert summary["status"] == "pending_paper_testnet_fill_telemetry"
    assert summary["actual_fill_efficiency_gate_pass"] is False
    assert "missing_paper_testnet_fill_telemetry" in summary["rejection_reasons"]
    fallback = payload["backtest_fallback_cut"]
    assert fallback["status"] == "no_backtest_fallback_promotion"
    assert fallback["backtest_fallback_pass_count"] == 0
    assert fallback["fallback_candidates"][0]["decision"] == "no_promotion"
    assert payload["gate_policy"]["backtest_fallback_policy"]["real_money_gate"] == "always_false"
    assert (
        payload["gate_policy"]["backtest_fallback_policy"]["locked_oos_role"]
        == "gate_report_only_after_train_validation_profile_freeze"
    )
    assert Path(payload["output_paths"]["latest_json"]).exists()
    assert Path(payload["output_paths"]["timestamped_json"]).exists()
    assert Path(payload["output_paths"]["paper_fill_efficiency_decisions_csv"]).exists()
    assert Path(payload["output_paths"]["backtest_fallback_candidates_csv"]).exists()


def test_paper_fill_efficiency_gate_passes_with_realized_edge_above_spread_multiple(tmp_path: Path) -> None:
    sample, monitoring = _write_sources(tmp_path)
    fills = tmp_path / "fills.jsonl"
    _write_jsonl(fills, [_fill_row(pnl=2.0) for _ in range(30)])

    payload = MODULE.build_payload(
        MODULE.parse_args(
            [
                "--sample-guarded-json",
                str(sample),
                "--monitoring-contract-json",
                str(monitoring),
                "--fill-jsonl",
                str(fills),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )

    summary = payload["actual_fill_efficiency_summary"]
    assert summary["fill_count"] == 30
    assert summary["avg_bbo_spread_bps"] == pytest.approx(2.0)
    assert summary["return_per_turnover_threshold_bps"] == pytest.approx(10.0)
    assert summary["realized_return_per_turnover_bps"] == pytest.approx(20.0)
    assert summary["actual_fill_efficiency_gate_pass"] is True
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False


def test_backtest_fallback_can_cut_to_backtest_paper_review_without_fills(tmp_path: Path) -> None:
    sample, monitoring = _write_sources(tmp_path, candidates=[_candidate_row(ready_for_paper=True)])

    payload = MODULE.build_payload(
        MODULE.parse_args(
            [
                "--sample-guarded-json",
                str(sample),
                "--monitoring-contract-json",
                str(monitoring),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )

    summary = payload["actual_fill_efficiency_summary"]
    fallback = payload["backtest_fallback_cut"]
    assert summary["status"] == "pending_paper_testnet_fill_telemetry"
    assert summary["actual_fill_efficiency_gate_pass"] is False
    assert fallback["status"] == "backtest_fallback_candidates_found"
    assert fallback["backtest_fallback_pass_count"] == 1
    assert fallback["fallback_candidates"][0]["decision"] == "paper_testnet_review_only"
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False


def test_paper_fill_efficiency_gate_rejects_low_return_per_turnover(tmp_path: Path) -> None:
    sample, monitoring = _write_sources(tmp_path)
    fills = tmp_path / "fills.jsonl"
    _write_jsonl(fills, [_fill_row(pnl=0.5) for _ in range(30)])

    summary = MODULE.build_payload(
        MODULE.parse_args(
            [
                "--sample-guarded-json",
                str(sample),
                "--monitoring-contract-json",
                str(monitoring),
                "--fill-jsonl",
                str(fills),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    )["actual_fill_efficiency_summary"]

    assert summary["realized_return_per_turnover_bps"] == pytest.approx(5.0)
    assert summary["return_per_turnover_threshold_bps"] == pytest.approx(10.0)
    assert summary["actual_fill_efficiency_gate_pass"] is False
    assert "return_per_turnover_bps_5.000_not_above_10.000" in summary["rejection_reasons"]


def test_paper_fill_efficiency_gate_rejects_real_money_source(tmp_path: Path) -> None:
    sample, monitoring = _write_sources(tmp_path)
    sample.write_text('{"real_money_execution": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unexpectedly allows real-money"):
        MODULE.build_payload(
            MODULE.parse_args(
                [
                    "--sample-guarded-json",
                    str(sample),
                    "--monitoring-contract-json",
                    str(monitoring),
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )
        )
