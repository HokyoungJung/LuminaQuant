from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_7x_paper_forward_preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_7x_paper_forward_preflight", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_round_trip_cost_summary_compares_realized_cost_to_10bps_assumption() -> None:
    summary = MODULE.summarize_round_trip_costs(
        [
            {"realized_fee_bps": 3.0, "realized_slippage_bps": 4.0},
            {"all_in_round_trip_bps": 10.0},
            {"realized_fee_bps": 5.0, "realized_slippage_bps": 7.0},
        ],
    )

    assert summary["observed_round_trips"] == 3
    assert summary["mean_all_in_round_trip_bps"] == pytest.approx(29.0 / 3.0)
    assert summary["p95_all_in_round_trip_bps"] == pytest.approx(11.8)
    assert summary["cost_status"] == "pass"

    failing = MODULE.summarize_round_trip_costs([{"all_in_round_trip_bps": 16.0}])
    assert failing["cost_status"] == "fail"


def test_realized_bps_requires_positive_notional() -> None:
    assert MODULE.realized_bps(5.0, 10_000.0) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="positive"):
        MODULE.realized_bps(1.0, 0.0)


def test_cost_summary_rejects_malformed_fill_costs() -> None:
    with pytest.raises(ValueError, match="realized_fee_bps"):
        MODULE.summarize_round_trip_costs(
            [{"realized_fee_bps": "bad", "realized_slippage_bps": 1.0}]
        )

    with pytest.raises(ValueError, match="all_in_round_trip_bps"):
        MODULE.summarize_round_trip_costs([{"all_in_round_trip_bps": float("nan")}])


def test_build_paper_forward_bundle_from_frozen_10bps_sources(tmp_path: Path) -> None:
    payload = MODULE.build_payload(
        MODULE.parse_args(
            [
                "--output-dir",
                str(tmp_path),
            ]
        )
    )

    assert payload["artifact_kind"] == "alpha_zoo_7x_paper_forward_preflight_bundle"
    # Paper readiness was intentionally demoted (see "Demote weak validation Alpha
    # Zoo exposure before paper trials"); the frozen 10bps evidence now yields
    # ready_for_paper=False, so this asserts the current demoted gate state.
    assert payload["ready_for_paper"] is False
    assert payload["ready_for_real"] is False
    assert payload["real_money_execution"] is False
    assert payload["locked_oos_governance"]["uses_locked_oos_for_selection"] is False

    rows = {row["role"]: row for row in payload["side_by_side_profiles"]}
    active = rows["active"]
    balanced = rows["balanced"]
    assert active["model_id"] == MODULE.ACTIVE_MODEL_ID
    assert balanced["model_id"] == MODULE.BALANCED_MODEL_ID
    assert active["paper_equivalent_sizing"]["expected_replay_notional"] == pytest.approx(14_000.0)
    assert active["paper_equivalent_sizing"]["live_notional"] == pytest.approx(14_000.0)
    assert balanced["paper_equivalent_sizing"]["expected_replay_notional"] == pytest.approx(
        10_500.0
    )
    assert balanced["paper_equivalent_sizing"]["live_notional"] == pytest.approx(10_500.0)
    assert active["preflight"]["status"]["ready_for_paper"] is False
    assert active["preflight"]["status"]["ready_for_real"] is False
    assert balanced["preflight"]["status"]["ready_for_paper"] is False
    assert balanced["preflight"]["status"]["ready_for_real"] is False

    active_decision = Path(payload["output_paths"]["active_decision"])
    balanced_decision = Path(payload["output_paths"]["balanced_decision"])
    monitoring_json = Path(payload["output_paths"]["monitoring_contract_json"])
    monitoring_csv = Path(payload["output_paths"]["monitoring_contract_csv"])
    assert active_decision.exists()
    assert balanced_decision.exists()
    assert monitoring_json.exists()
    assert monitoring_csv.exists()
    assert "active," in monitoring_csv.read_text(encoding="utf-8")
    assert "balanced," in monitoring_csv.read_text(encoding="utf-8")
