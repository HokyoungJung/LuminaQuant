"""Tests for the go-live promotion pipeline bridge (audit P1/P2).

Covers:
* ``scripts/ops/write_real_money_attestation.py`` — refuses positive flags without
  embedded+verified references; emits a clean referenced artifact when evidence is
  present;
* ``scripts/research/write_strategy_factory_live_decision.py`` — turns a
  candidate_research winner into a ``promote_candidate`` decision, fail-closed on
  hard-rejects / unregistered strategies;
* readiness_policy end-to-end: a referenced attestation unlocks ``ready_for_real``
  while a decision cannot self-attest; ``ready_for_full`` additionally requires
  recorded canary evidence or an explicit override; missing artifacts raise
  ``LiveReadinessBlockedError`` (not a raw ``FileNotFoundError``).
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import lumina_quant.live.readiness_policy as readiness_policy
from lumina_quant.live.readiness_policy import (
    LiveReadinessBlockedError,
    build_live_readiness_payload,
    enforce_live_readiness_from_files,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATTEST = _load(ROOT / "scripts" / "ops" / "write_real_money_attestation.py", "wr_attest")
FACTORY = _load(
    ROOT / "scripts" / "research" / "write_strategy_factory_live_decision.py",
    "wr_factory",
)


# --------------------------------------------------------------------------- #
# write_real_money_attestation                                                  #
# --------------------------------------------------------------------------- #
def _evidence(tmp_path: Path) -> dict[str, Path]:
    paper = tmp_path / "paper_stats.json"
    paper.write_text(json.dumps({"trades": 42, "net_return": 0.03}), encoding="utf-8")
    slippage = tmp_path / "fill_slippage.json"
    slippage.write_text(json.dumps({"median_slippage_bps": 3.0}), encoding="utf-8")
    lineage = tmp_path / "decision_lineage.json"
    lineage.write_text(json.dumps({"decision": "promote_candidate"}), encoding="utf-8")
    return {"paper": paper, "slippage": slippage, "lineage": lineage}


def test_attestation_refuses_positive_flags_without_evidence() -> None:
    with pytest.raises(ATTEST.AttestationRefused) as exc:
        ATTEST.build_attestation(assert_ready_for_real=True)
    reasons = " ".join(exc.value.reasons)
    assert "operator_id" in reasons
    assert "evidence is required" in reasons


def test_attestation_refuses_without_operator_id(tmp_path: Path) -> None:
    ev = _evidence(tmp_path)
    with pytest.raises(ATTEST.AttestationRefused):
        ATTEST.build_attestation(
            operator_id="",
            paper_stats=ev["paper"],
            fill_slippage_summary=ev["slippage"],
            decision=ev["lineage"],
            assert_ready_for_real=True,
            assert_real_execution_allowed=True,
            assert_real_money_execution=True,
        )


def test_attestation_refuses_missing_evidence_file(tmp_path: Path) -> None:
    ev = _evidence(tmp_path)
    with pytest.raises(ATTEST.AttestationRefused) as exc:
        ATTEST.build_attestation(
            operator_id="op-1",
            paper_stats=tmp_path / "nope.json",
            fill_slippage_summary=ev["slippage"],
            decision=ev["lineage"],
            assert_ready_for_real=True,
            assert_real_execution_allowed=True,
            assert_real_money_execution=True,
        )
    assert any("not found" in r for r in exc.value.reasons)


def test_attestation_emits_clean_artifact_with_verified_evidence(tmp_path: Path) -> None:
    ev = _evidence(tmp_path)
    payload = ATTEST.build_attestation(
        operator_id="op-1",
        paper_stats=ev["paper"],
        fill_slippage_summary=ev["slippage"],
        decision=ev["lineage"],
        assert_ready_for_real=True,
        assert_real_execution_allowed=True,
        assert_real_money_execution=True,
    )
    assert payload["ready_for_real"] is True
    assert payload["real_execution_allowed"] is True
    assert payload["real_money_execution"] is True
    assert payload["canary_execution_recorded"] is False
    # Evidence is embedded + verified (sha256 present).
    assert payload["evidence"]["paper_stats"]["sha256"]
    assert payload["evidence"]["decision_lineage"]["decision"] == "promote_candidate"


def test_attestation_canary_recorded_requires_canary_run(tmp_path: Path) -> None:
    ev = _evidence(tmp_path)
    with pytest.raises(ATTEST.AttestationRefused):
        ATTEST.build_attestation(
            operator_id="op-1",
            paper_stats=ev["paper"],
            fill_slippage_summary=ev["slippage"],
            decision=ev["lineage"],
            assert_ready_for_real=True,
            assert_real_execution_allowed=True,
            assert_real_money_execution=True,
            record_canary_evidence=True,  # no canary_run supplied -> refuse
        )
    canary = tmp_path / "canary_run.json"
    canary.write_text(json.dumps({"fills": 10, "pnl": 1.2}), encoding="utf-8")
    payload = ATTEST.build_attestation(
        operator_id="op-1",
        paper_stats=ev["paper"],
        fill_slippage_summary=ev["slippage"],
        decision=ev["lineage"],
        canary_run=canary,
        assert_ready_for_real=True,
        assert_real_execution_allowed=True,
        assert_real_money_execution=True,
        record_canary_evidence=True,
    )
    assert payload["canary_execution_recorded"] is True
    assert payload["evidence"]["canary_run"]["verified"] is True


def test_attestation_main_refuses_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "attest.json"
    code = ATTEST.main(["--assert-ready-for-real", "--output", str(out)])
    assert code == 2
    assert not out.exists()


def test_attestation_main_writes_on_success(tmp_path: Path) -> None:
    ev = _evidence(tmp_path)
    out = tmp_path / "attest.json"
    code = ATTEST.main(
        [
            "--operator-id",
            "op-1",
            "--paper-stats",
            str(ev["paper"]),
            "--fill-slippage-summary",
            str(ev["slippage"]),
            "--decision",
            str(ev["lineage"]),
            "--assert-ready-for-real",
            "--assert-real-execution-allowed",
            "--assert-real-money-execution",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["ready_for_real"] is True


# --------------------------------------------------------------------------- #
# write_strategy_factory_live_decision                                          #
# --------------------------------------------------------------------------- #
def _research(candidate: dict) -> dict:
    return {"schema_version": "1", "candidates": [candidate]}


def test_factory_promotes_registered_candidate(tmp_path: Path) -> None:
    research = _research(
        {
            "candidate_id": "cand-1",
            "name": "abnormal_return_continuation_1h_v1",
            "strategy_class": "AbnormalReturnContinuationStrategy",
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "params": {"lookback_bars": 24},
            "strategy_timeframe": "1h",
            "pass": True,
            "hard_reject": False,
        }
    )
    research_path = tmp_path / "candidate_research.json"
    research_path.write_text(json.dumps(research), encoding="utf-8")

    decision = FACTORY.build_strategy_factory_decision(
        research=research,
        research_path=research_path,
        candidate_id="cand-1",
    )
    assert decision["decision"] == "promote_candidate"
    assert decision["strategy_name"] == "AbnormalReturnContinuationStrategy"
    assert decision["symbols"] == ["BTC/USDT", "ETH/USDT"]
    # The emitted decision is runtime-compatible via the explicit strategy_name (P2).
    assert (
        readiness_policy._decision_runtime_compatible(
            decision_allowed=True,
            decision_keep=False,
            selected_reference=decision["selected_mode"],
            strategy_name=decision["strategy_name"],
        )
        is True
    )


def test_factory_refuses_hard_rejected_candidate(tmp_path: Path) -> None:
    research = _research(
        {
            "candidate_id": "cand-2",
            "name": "abnormal_x",
            "strategy_class": "AbnormalReturnContinuationStrategy",
            "hard_reject": True,
            "hard_reject_reasons": ["oos_negative"],
        }
    )
    with pytest.raises(FACTORY.DecisionRefused):
        FACTORY.build_strategy_factory_decision(
            research=research,
            research_path=tmp_path / "r.json",
            candidate_id="cand-2",
        )


def test_factory_refuses_unregistered_strategy(tmp_path: Path) -> None:
    research = _research(
        {
            "candidate_id": "cand-3",
            "name": "made_up",
            "strategy_class": "TotallyNotARegisteredStrategy",
            "pass": True,
            "hard_reject": False,
        }
    )
    with pytest.raises(FACTORY.DecisionRefused) as exc:
        FACTORY.build_strategy_factory_decision(
            research=research,
            research_path=tmp_path / "r.json",
            candidate_id="cand-3",
        )
    assert "not in the live" in str(exc.value)


def test_factory_refuses_missing_candidate(tmp_path: Path) -> None:
    research = _research({"candidate_id": "cand-1", "name": "x"})
    with pytest.raises(FACTORY.DecisionRefused):
        FACTORY.build_strategy_factory_decision(
            research=research,
            research_path=tmp_path / "r.json",
            candidate_id="does-not-exist",
        )


def test_factory_wires_attestation_reference(tmp_path: Path) -> None:
    research = _research(
        {
            "candidate_id": "cand-1",
            "name": "abnormal_1h",
            "strategy_class": "AbnormalReturnContinuationStrategy",
            "pass": True,
            "hard_reject": False,
        }
    )
    attestation = tmp_path / "attest.json"
    attestation.write_text("{}", encoding="utf-8")
    decision = FACTORY.build_strategy_factory_decision(
        research=research,
        research_path=tmp_path / "r.json",
        candidate_id="cand-1",
        attestation_path=attestation,
    )
    assert decision["strategy_params"]["real_money_attestation_artifact_path"] == str(
        attestation.resolve()
    )


# --------------------------------------------------------------------------- #
# readiness_policy end-to-end: attestation unlocks real, full needs canary       #
# --------------------------------------------------------------------------- #
def _real_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "storage:",
                '  postgres_dsn: "postgresql://demo"',
                "live:",
                '  mode: "real"',
                "  testnet: false",
                "  require_real_enable_flag: true",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _fresh_refresh(tmp_path: Path) -> Path:
    fresh_cutoff = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "completed",
                "collection_cutoff_utc": fresh_cutoff,
                "feature_results": [{"last_timestamp_utc": fresh_cutoff}],
            }
        ),
        encoding="utf-8",
    )
    return refresh


def _decision_with_attestation(tmp_path: Path, attestation_body: dict) -> Path:
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps(attestation_body), encoding="utf-8")
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "decision": "selected_live_mode",
                "selected_mode": "MovingAverageCrossStrategy",
                "candidate_key": "moving_average_cross",
                "strategy_params": {"real_money_attestation_artifact_path": str(attestation)},
            }
        ),
        encoding="utf-8",
    )
    return decision


def test_e2e_full_requires_canary_evidence(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LUMINA_ENABLE_LIVE_REAL", "true")
    config_path = _real_config(tmp_path)
    refresh = _fresh_refresh(tmp_path)
    decision = _decision_with_attestation(
        tmp_path,
        {
            "ready_for_real": True,
            "real_execution_allowed": True,
            "real_money_execution": True,
            "clean_promotion_eligible": True,
        },
    )
    payload = build_live_readiness_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )
    # real is reachable; full is NOT (no recorded canary evidence).
    assert payload["status"]["ready_for_real"] is True
    assert payload["status"]["ready_for_full"] is False

    # full stage enforcement blocks.
    with pytest.raises(LiveReadinessBlockedError):
        enforce_live_readiness_from_files(
            mode="real",
            config_path=config_path,
            refresh_json=refresh,
            decision_json=decision,
            stale_minutes=10_000,
            go_live_stage="full",
        )


def test_e2e_full_unlocked_by_recorded_canary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LUMINA_ENABLE_LIVE_REAL", "true")
    config_path = _real_config(tmp_path)
    refresh = _fresh_refresh(tmp_path)
    decision = _decision_with_attestation(
        tmp_path,
        {
            "ready_for_real": True,
            "real_execution_allowed": True,
            "real_money_execution": True,
            "clean_promotion_eligible": True,
            "canary_execution_recorded": True,
        },
    )
    payload = build_live_readiness_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )
    assert payload["status"]["ready_for_real"] is True
    assert payload["status"]["ready_for_full"] is True
    assert payload["checks"]["canary_evidence_recorded"] is True


def test_e2e_full_unlocked_by_explicit_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LUMINA_ENABLE_LIVE_REAL", "true")
    monkeypatch.setenv("LUMINA_ALLOW_FULL_WITHOUT_CANARY", "1")
    config_path = _real_config(tmp_path)
    refresh = _fresh_refresh(tmp_path)
    decision = _decision_with_attestation(
        tmp_path,
        {
            "ready_for_real": True,
            "real_execution_allowed": True,
            "real_money_execution": True,
            "clean_promotion_eligible": True,
        },
    )
    payload = build_live_readiness_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )
    assert payload["status"]["ready_for_full"] is True
    assert payload["checks"]["full_stage_override"] is True


def test_missing_refresh_artifact_raises_blocked_error(tmp_path: Path) -> None:
    config_path = _real_config(tmp_path)
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps({"decision": "keep_incumbent"}), encoding="utf-8")
    with pytest.raises(LiveReadinessBlockedError) as exc:
        build_live_readiness_payload(
            config_path=config_path,
            refresh_json=tmp_path / "missing_refresh.json",
            decision_json=decision,
            stale_minutes=10_000,
        )
    assert exc.value.payload.get("error") == "missing_or_unreadable_refresh_artifact"


def test_missing_decision_artifact_raises_blocked_error(tmp_path: Path) -> None:
    config_path = _real_config(tmp_path)
    refresh = _fresh_refresh(tmp_path)
    with pytest.raises(LiveReadinessBlockedError) as exc:
        build_live_readiness_payload(
            config_path=config_path,
            refresh_json=refresh,
            decision_json=tmp_path / "missing_decision.json",
            stale_minutes=10_000,
        )
    assert exc.value.payload.get("error") == "missing_or_unreadable_decision_artifact"
