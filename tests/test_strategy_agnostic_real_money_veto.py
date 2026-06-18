"""Regression tests for the strategy-agnostic real-money veto (fix/audit-hardening).

Covers the round-1 fail-open holes in
``lumina_quant.live.readiness_policy._strategy_agnostic_real_money_veto_checks``:

* a self-attesting decision must NOT override a referenced PAPER-ONLY / not-ready
  artifact (any()-vs-all() flaw);
* blocking/positive flags encoded as ``'true'``/1 (or ``'false'``/0) must be
  honored (strict ``is True``/``is False`` miss string/int encodings);
* absent positive flags keep the veto active (fail-closed);
* nested selected/strategy profiles are inspected so a dirty selected profile
  cannot be masked by a clean top-level payload.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import lumina_quant.live.readiness_policy as readiness_policy
from lumina_quant.live.readiness_policy import (
    _as_bool_flag,
    _strategy_agnostic_real_money_veto_checks,
)


# --------------------------------------------------------------------------- #
# _as_bool_flag coercion                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        (1, True),
        ("yes", True),
        ("on", True),
        (False, False),
        ("false", False),
        ("False", False),
        ("0", False),
        (0, False),
        ("no", False),
        ("off", False),
        (None, None),
        ("", None),
        ("maybe", None),
        (2, None),
        (1.0, None),
    ],
)
def test_as_bool_flag_coercion(value: object, expected: bool | None) -> None:
    assert _as_bool_flag(value) is expected


# --------------------------------------------------------------------------- #
# any()-vs-all(): self-attesting decision cannot override a dirty artifact      #
# --------------------------------------------------------------------------- #
def test_self_attesting_decision_cannot_override_paper_only_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "paper_only.json"
    artifact.write_text(
        json.dumps(
            {
                "ready_for_real": True,
                "real_money_execution": True,
                "real_execution_allowed": True,
                "paper_testnet_only": True,  # dirty: referenced artifact is paper-only
            }
        ),
        encoding="utf-8",
    )
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        # Self-attesting clean decision must NOT win over the dirty artifact.
        "ready_for_real": True,
        "real_money_execution": True,
        "real_execution_allowed": True,
        "clean_promotion_eligible": True,
        "strategy_params": {"selected_artifact_path": str(artifact)},
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is True
    assert checks["artifact_paper_testnet_only"] is True


def test_self_attesting_decision_cannot_override_not_ready_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "not_ready.json"
    artifact.write_text(
        json.dumps(
            {
                "ready_for_real": False,  # dirty: explicit negation
                "real_money_execution": True,
                "real_execution_allowed": True,
            }
        ),
        encoding="utf-8",
    )
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "ready_for_real": True,
        "real_money_execution": True,
        "real_execution_allowed": True,
        "clean_promotion_eligible": True,
        "strategy_params": {"selected_artifact_path": str(artifact)},
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is True


# --------------------------------------------------------------------------- #
# string/int encodings: blocking flags as 'true'/1 must block                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("blocking_value", ["true", "True", 1, "1", "yes", "on"])
def test_paper_only_string_or_int_encoding_blocks(blocking_value: object) -> None:
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "ready_for_real": True,
        "real_money_execution": True,
        "real_execution_allowed": True,
        "clean_promotion_eligible": True,
        "paper_testnet_only": blocking_value,  # encoded blocker
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is True
    assert checks["artifact_paper_testnet_only"] is True


@pytest.mark.parametrize("blocking_value", ["true", 1, "1"])
def test_governance_blocker_string_or_int_encoding_blocks(blocking_value: object) -> None:
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "ready_for_real": True,
        "real_money_execution": True,
        "real_execution_allowed": True,
        "clean_promotion_eligible": True,
        "post_oos_research_variant": blocking_value,  # encoded governance blocker
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is True
    assert "post_oos_research_variant" in checks["artifact_governance_veto_reasons"]


@pytest.mark.parametrize("blocking_value", ["false", 0, "0", "False"])
def test_clean_promotion_eligible_false_encoding_blocks(blocking_value: object) -> None:
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "ready_for_real": True,
        "real_money_execution": True,
        "real_execution_allowed": True,
        "clean_promotion_eligible": blocking_value,  # encoded falsy blocker
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is True
    assert "clean_promotion_eligible_false" in checks["artifact_governance_veto_reasons"]


# --------------------------------------------------------------------------- #
# positive flags as 'true'/1 with no blockers => veto False                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("positive_value", ["true", "True", 1, "1", "yes", "on"])
def test_positive_flags_string_or_int_encoding_clear_veto(positive_value: object) -> None:
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "ready_for_real": positive_value,
        "real_money_execution": positive_value,
        "real_execution_allowed": positive_value,
        "clean_promotion_eligible": positive_value,
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is False
    assert checks["artifact_ready_for_real"] is True
    assert checks["artifact_governance_veto_reasons"] == []


# --------------------------------------------------------------------------- #
# absent positive flags keep the veto active (fail-closed)                      #
# --------------------------------------------------------------------------- #
def test_absent_positive_flags_keep_veto() -> None:
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "candidate_key": "moving_average_cross",
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is True
    assert checks["artifact_ready_for_real"] is False


# --------------------------------------------------------------------------- #
# nested selected/strategy profile cannot be masked by clean top-level          #
# --------------------------------------------------------------------------- #
def test_dirty_selected_profile_blocks_clean_top_level() -> None:
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "ready_for_real": True,
        "real_money_execution": True,
        "real_execution_allowed": True,
        "clean_promotion_eligible": True,
        "selected_profile": {
            "ready_for_real": True,
            "real_money_execution": True,
            "paper_testnet_candidate": True,  # dirty nested profile
        },
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is True
    assert checks["artifact_paper_testnet_only"] is True


def test_dirty_strategy_profiles_list_blocks() -> None:
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "ready_for_real": True,
        "real_money_execution": True,
        "real_execution_allowed": True,
        "clean_promotion_eligible": True,
        "strategy_profiles": [
            {"ready_for_real": True},
            {"post_oos_research_variant": True},  # dirty nested governance blocker
        ],
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is True
    assert "post_oos_research_variant" in checks["artifact_governance_veto_reasons"]


def test_clean_nested_profile_does_not_block() -> None:
    decision = {
        "decision": "selected_live_mode",
        "selected_mode": "MovingAverageCrossStrategy",
        "ready_for_real": True,
        "real_money_execution": True,
        "real_execution_allowed": True,
        "clean_promotion_eligible": True,
        "selected_profile": {
            "ready_for_real": True,
            "real_money_execution": True,
            "paper_testnet_candidate": False,
        },
    }

    checks = _strategy_agnostic_real_money_veto_checks(decision)

    assert checks["artifact_real_money_veto"] is False


# --------------------------------------------------------------------------- #
# end-to-end regression via build_live_readiness_payload                        #
# --------------------------------------------------------------------------- #
def _write_real_mode_config(tmp_path: Path) -> Path:
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


def _write_fresh_refresh(tmp_path: Path) -> Path:
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


def test_end_to_end_self_attesting_decision_with_paper_only_artifact_blocks_real(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = _write_real_mode_config(tmp_path)
    refresh = _write_fresh_refresh(tmp_path)
    artifact = tmp_path / "paper_only.json"
    artifact.write_text(
        json.dumps(
            {
                "ready_for_real": True,
                "real_money_execution": True,
                "real_execution_allowed": True,
                "paper_testnet_only": True,
            }
        ),
        encoding="utf-8",
    )
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "decision": "selected_live_mode",
                "selected_mode": "MovingAverageCrossStrategy",
                "candidate_key": "moving_average_cross",
                "ready_for_real": True,
                "real_money_execution": True,
                "real_execution_allowed": True,
                "clean_promotion_eligible": True,
                "strategy_params": {"selected_artifact_path": str(artifact)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LUMINA_ENABLE_LIVE_REAL", "true")

    payload = readiness_policy.build_live_readiness_payload(
        config_path=config_path,
        refresh_json=refresh,
        decision_json=decision,
        stale_minutes=10_000,
    )

    assert payload["checks"]["artifact_real_money_veto"] is True
    assert payload["checks"]["artifact_paper_testnet_only"] is True
    assert payload["status"]["ready_for_real"] is False
    assert payload["recommended_action"] == "block_until_preflight_gaps_closed"
