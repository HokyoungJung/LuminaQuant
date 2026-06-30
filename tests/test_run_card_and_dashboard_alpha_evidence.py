from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pandas as pd
import pytest

from lumina_quant.alpha_zoo.evidence import AlphaEvidenceThresholds, alpha_benchmark_evidence
import lumina_quant.dashboard.alpha_evidence_service as alpha_evidence_service
from lumina_quant.dashboard.alpha_evidence_service import build_alpha_evidence_payload
import lumina_quant.live.order_gateway as order_gateway
from lumina_quant.research.run_card import (
    RunCardRealityGateError,
    assert_reality_gates_pass,
    build_research_run_card,
    stable_payload_hash,
    write_run_card,
)


def _passing_run_card() -> dict:
    card = build_research_run_card(
        run_id="unit-run",
        execution_mode="paper",
        strategy_name="UnitStrategy",
        config={"timeframe": "1m"},
        candidate={"candidate_id": "unit", "params": {"x": 1}},
        data_manifest={"source": "unit", "rows": 100},
        source_refs=("unit-test",),
        cost_model={"fee_bps": 2.0, "slippage_bps": 1.0},
        funding_model={"funding_rate_bps": 0.1},
        data_integrity={"passed": True},
        selection_policy={"uses_locked_oos_for_selection": False},
        parity_checks={"passed": True},
        performance_budget={"observed_regression_ratio": 0.01, "max_regression_ratio": 0.10},
        artifacts={"evidence": {"classification": "alive"}},
    )
    return card.to_dict()


def test_run_card_enforces_reality_gates_and_strict_json(tmp_path) -> None:
    payload = _passing_run_card()

    assert all(payload["reality_gates"].values())
    assert len(payload["run_card_hash"]) == 64

    path = write_run_card(tmp_path / "run-card.json", payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["artifact_kind"] == "lumina_research_run_card"
    assert loaded["reality_gates"]["gate_no_real_money_run"] is True


def test_run_card_blocks_real_mode_and_oos_leakage() -> None:
    card = build_research_run_card(
        run_id="bad-run",
        execution_mode="real",
        strategy_name="UnitStrategy",
        config={},
        candidate={},
        data_manifest={},
        source_refs=("unit-test",),
        cost_model={"fee_bps": 2.0},
        funding_model={"funding_rate_bps": 0.1},
        data_integrity={"passed": True},
        selection_policy={"uses_locked_oos_for_selection": True},
        parity_checks={"passed": True},
        performance_budget={"observed_regression_ratio": 0.01, "max_regression_ratio": 0.10},
    ).to_dict()

    with pytest.raises(RunCardRealityGateError) as exc_info:
        assert_reality_gates_pass(card)

    assert "gate_no_real_money_run" in str(exc_info.value)
    assert "gate_no_oos_leakage" in str(exc_info.value)


def test_run_card_consumes_alpha_evidence_oos_leakage_flag() -> None:
    rows = []
    for idx, split in enumerate(("train", "train", "locked_oos", "locked_oos")):
        timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=idx)
        for rank, symbol in enumerate(("A", "B", "C", "D"), start=1):
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "split": split,
                    "signal": float(rank),
                    "forward_return": float(rank) / 100.0,
                }
            )
    evidence = alpha_benchmark_evidence(
        pd.DataFrame(rows),
        factor="signal",
        selection_splits=("train", "locked_oos"),
        thresholds=AlphaEvidenceThresholds(min_periods=1, min_abs_t_stat=1.0),
    )
    card = build_research_run_card(
        run_id="oos-evidence-run",
        execution_mode="paper",
        strategy_name="UnitStrategy",
        config={},
        candidate={"evidence_hash": evidence["evidence_hash"]},
        data_manifest={"source": "unit"},
        source_refs=("unit-test",),
        cost_model={"fee_bps": 2.0},
        funding_model={"funding_rate_bps": 0.1},
        data_integrity={"passed": True},
        selection_policy=evidence,
        parity_checks={"passed": True},
        performance_budget={"observed_regression_ratio": 0.01, "max_regression_ratio": 0.10},
    ).to_dict()

    assert evidence["uses_locked_oos_for_selection"] is True
    assert card["reality_gates"]["gate_no_oos_leakage"] is False
    with pytest.raises(RunCardRealityGateError, match="gate_no_oos_leakage"):
        assert_reality_gates_pass(card)


def test_run_card_fails_closed_when_required_gates_or_inputs_are_missing(tmp_path) -> None:
    with pytest.raises(RunCardRealityGateError) as empty_exc:
        assert_reality_gates_pass({})
    assert "gate_backtest_live_parity" in str(empty_exc.value)

    card = build_research_run_card(
        run_id="missing-input-run",
        execution_mode="",
        strategy_name="UnitStrategy",
        config={},
        candidate={},
        data_manifest={},
        source_refs=("unit-test",),
        cost_model={"fee_bps": 2.0},
        funding_model={"funding_rate_bps": 0.1},
        data_integrity={"passed": True},
        selection_policy={},
        parity_checks={"passed": True},
        performance_budget={},
    ).to_dict()

    assert card["reality_gates"]["gate_no_oos_leakage"] is False
    assert card["reality_gates"]["gate_performance_budget"] is False
    assert card["reality_gates"]["gate_no_real_money_run"] is False
    with pytest.raises(RunCardRealityGateError):
        write_run_card(tmp_path / "blocked.json", card)


def test_write_run_card_rejects_nan_even_for_mapping_payload(tmp_path) -> None:
    payload = _passing_run_card()
    payload["bad"] = float("nan")

    with pytest.raises(ValueError):
        write_run_card(tmp_path / "nan.json", payload)


def test_dashboard_alpha_evidence_payload_is_read_only() -> None:
    run_card = _passing_run_card()
    payload = build_alpha_evidence_payload(
        evidence=[{"factor": "unit", "classification": "alive"}],
        run_cards=[run_card],
        live_readiness={"recommended_action": "paper_shadow_only"},
    )

    assert payload["advisory_only"] is True
    assert payload["real_money_execution_enabled"] is False
    assert payload["summary"]["alpha_count"] == 1
    assert payload["summary"]["classifications"] == {"alive": 1}
    assert payload["summary"]["reality_gates"]["gate_no_real_money_run"]["passed"] is True


def test_alpha_evidence_dashboard_does_not_cross_order_gateway_readiness_boundary() -> None:
    service_source = inspect.getsource(alpha_evidence_service)
    order_gateway_source = inspect.getsource(order_gateway)
    route_source = Path(
        "apps/dashboard_web/app/api/python/dashboard/alpha-evidence/route.ts"
    ).read_text(encoding="utf-8")

    assert "OrderGateway" not in service_source
    assert "lumina_quant.live" not in service_source
    assert "execute_order" not in service_source
    assert "cancel_order" not in service_source
    assert "readiness_policy" not in service_source
    assert "enforce_live_readiness" not in service_source

    assert "readiness_policy" not in order_gateway_source
    assert "enforce_live_readiness" not in order_gateway_source

    assert "POST" not in route_source
    assert "OrderGateway" not in route_source


def test_alpha_benchmark_loop_stays_within_performance_budget() -> None:
    rows = []
    max_seconds = 2.0
    for idx in range(48):
        split = "train" if idx < 24 else "validation" if idx < 36 else "locked_oos"
        timestamp = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=idx)
        for rank in range(96):
            signal = float(rank)
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": f"S{rank:03d}",
                    "split": split,
                    "signal": signal,
                    "forward_return": signal / 1000.0,
                }
            )

    start = time.perf_counter()
    evidence = alpha_benchmark_evidence(pd.DataFrame(rows), factor="signal")
    elapsed = time.perf_counter() - start
    performance_budget = {
        "observed_seconds": elapsed,
        "max_seconds": max_seconds,
        "observed_regression_ratio": elapsed / max_seconds,
        "max_regression_ratio": 1.0,
    }
    card = build_research_run_card(
        run_id="perf-budget-run",
        execution_mode="paper",
        strategy_name="UnitStrategy",
        config={},
        candidate={"evidence_hash": evidence["evidence_hash"]},
        data_manifest={"rows": len(rows)},
        source_refs=("unit-test",),
        cost_model={"fee_bps": 2.0},
        funding_model={"funding_rate_bps": 0.1},
        data_integrity={"passed": True},
        selection_policy=evidence,
        parity_checks={"passed": True},
        performance_budget=performance_budget,
    ).to_dict()

    assert elapsed <= max_seconds
    assert evidence["pass"] is True
    assert card["reality_gates"]["gate_performance_budget"] is True


def test_stable_payload_hash_is_deterministic() -> None:
    assert stable_payload_hash({"b": 2, "a": 1}) == stable_payload_hash({"a": 1, "b": 2})
