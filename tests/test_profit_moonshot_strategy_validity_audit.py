from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FINAL_PATH = ROOT / "scripts" / "research" / "write_profit_moonshot_live_final_selection.py"
FINAL_SPEC = importlib.util.spec_from_file_location(
    "write_profit_moonshot_live_final_selection_for_validity_tests", FINAL_PATH
)
assert FINAL_SPEC is not None and FINAL_SPEC.loader is not None
FINAL = importlib.util.module_from_spec(FINAL_SPEC)
sys.modules[FINAL_SPEC.name] = FINAL
FINAL_SPEC.loader.exec_module(FINAL)

AUDIT_PATH = ROOT / "scripts" / "research" / "audit_profit_moonshot_strategy_validity.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_profit_moonshot_strategy_validity", AUDIT_PATH
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = AUDIT
AUDIT_SPEC.loader.exec_module(AUDIT)


def _validity_for(sleeves: list[str]) -> dict:
    return FINAL._strategy_validity(
        kind="candidate_portfolio",
        name="candidate",
        raw={"sleeves": sleeves},
        source_artifact="candidate.json",
        candidate_derived=True,
        benchmark_only=False,
    )


def test_calendar_primary_sleeves_fail_even_inside_hybrid_aliases() -> None:
    validity = _validity_for(
        [
            "candidate_hybrid_input_01_fresh_portfolio_train_val_monthly_return_budget_fresh_calendar_trx_takeprofit_sethusdt_thr180_h168",
            "fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all",
        ]
    )

    assert validity["pass"] is False
    assert validity["primary_signal_type"] == "calendar_primary"
    assert "calendar_primary_alpha_unsupported" in validity["rejection_reasons"]


def test_calendar_parameter_fields_fail_even_when_sleeves_are_stateful() -> None:
    validity = FINAL._strategy_validity(
        kind="candidate_portfolio",
        name="stateful_with_fixed_calendar_params",
        raw={
            "sleeves": ["fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all"],
            "params": {"month_filter": [1, 2], "risk": {"entry_hours": [2]}},
        },
        source_artifact="candidate.json",
        candidate_derived=True,
        benchmark_only=False,
    )

    assert validity["pass"] is False
    assert validity["primary_signal_type"] == "calendar_timed_state_signal"
    assert "calendar_entry_rule_unsupported" in validity["rejection_reasons"]
    assert "calendar_month_entry_rule" in validity["rejection_reasons"]
    assert "calendar_hour_entry_rule" in validity["rejection_reasons"]
    assert validity["calendar_rule_param_refs"] == [
        "params.month_filter",
        "params.risk.entry_hours",
    ]


def test_dynamic_state_signal_families_are_not_overblocked_by_secondary_time_filters() -> None:
    dynamic_sleeves = [
        "fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all",
        "fresh_funding_carry_fade_lb168_thr40_h120_asia",
        "fresh_flow_exhaustion_lb72_thr90_h48_us",
        "fresh_adaptive_trend_lb96_thr120_h72",
        "fresh_cross_sectional_sharpe_lb336_top1_h120",
        "fresh_compression_breakout_lb48_thr80_h24",
    ]

    for sleeve in dynamic_sleeves:
        validity = _validity_for([sleeve])
        assert validity["pass"] is True
        assert validity["primary_signal_type"] == "state_signal"
        assert validity["rejection_reasons"] == []


def test_missing_candidate_sleeves_fail_closed() -> None:
    validity = _validity_for([])

    assert validity["pass"] is False
    assert "strategy_source_row_missing_sleeves" in validity["rejection_reasons"]


def test_closure_manifest_records_required_roles_and_missing_optional_sources(
    tmp_path: Path,
) -> None:
    final_json = tmp_path / "final.json"
    final_md = tmp_path / "final.md"
    liquidation_json = tmp_path / "liquidation.json"
    candidate_json = tmp_path / "candidate.json"
    hybrid_json = tmp_path / "candidate_hybrid.json"
    merged_csv = tmp_path / "merged.csv"
    for path, payload in [
        (liquidation_json, {"artifact_kind": "liquidation", "rows": [{"name": "liq"}]}),
        (candidate_json, {"artifact_kind": "candidate", "rows": [{"name": "cand"}]}),
        (hybrid_json, {"artifact_kind": "hybrid", "rows": [{"name": "hybrid"}]}),
    ]:
        path.write_text(json.dumps(payload), encoding="utf-8")
    final_md.write_text("# final\n\nrow\n", encoding="utf-8")
    merged_csv.write_text("name,score\none,1\n", encoding="utf-8")

    row = {
        "name": "dynamic_candidate",
        "kind": "candidate_portfolio",
        "candidate_derived": True,
        "benchmark_only": False,
        "source_artifact": str(candidate_json),
        "sleeves": ["fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all"],
        "decision_gates": {"deployable_candidate": True},
        "strategy_validity": _validity_for(
            ["fresh_pair_resid_revert_spread_lb24_z150_h120_sc10_st100_tp400_all"]
        ),
        "rejection_reasons": [],
    }
    final_payload = {
        "rows": [row],
        "source_artifacts": {
            "liquidation_json": str(liquidation_json),
            "candidate_portfolio_json": str(candidate_json),
            "candidate_hybrid_json": str(hybrid_json),
        },
    }
    final_json.write_text(json.dumps(final_payload), encoding="utf-8")

    payload = AUDIT.build_strategy_validity_audit_payload(
        final_selection_payload=final_payload,
        source_artifacts={
            "final_selection_json": str(final_json),
            "final_selection_md": str(final_md),
            "liquidation_json": str(liquidation_json),
            "candidate_portfolio_json": str(candidate_json),
            "candidate_hybrid_json": str(hybrid_json),
            "merged_candidate_csv": str(merged_csv),
            "current_base": "",
            "passing_artifacts": "",
        },
    )

    roles = {entry["source_role"] for entry in payload["closure_manifest"]}
    assert {
        "final_selection_json",
        "final_selection_md",
        "liquidation_validation",
        "candidate_portfolio",
        "candidate_hybrid",
        "merged_candidate_csv",
        "per_row_per_sleeve_sources",
    }.issubset(roles)
    assert payload["status"] == "pass"
    assert payload["summary"]["deployable_valid_count"] == 1
    assert payload["source_pool_summary"]["available"] is True
    assert payload["source_pool_summary"]["row_count"] == 1
    assert {"source_role": "current_base", "reason": "not_provided"} in payload[
        "missing_optional_sources"
    ]
    assert {"source_role": "passing_artifacts", "reason": "not_provided"} in payload[
        "missing_optional_sources"
    ]
