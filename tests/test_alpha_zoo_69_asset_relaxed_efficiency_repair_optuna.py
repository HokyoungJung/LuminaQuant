from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_relaxed_efficiency_repair_optuna as module


def _base_row(**overrides):
    row = {
        "symbol": "CRCLUSDT",
        "asset_group": "tradfi_equity",
        "train_return": 0.08,
        "validation_return": 0.04,
        "train_mdd": 0.04,
        "validation_mdd": 0.03,
        "train_return_per_turnover_proxy_bps": 55.0,
        "validation_return_per_turnover_proxy_bps": 60.0,
        "train_trade_event_count": 100,
        "validation_trade_event_count": 35,
        "dominant_anchor_abs_corr": 0.10,
        "gross_notional_fraction": 2.0,
        "train_liquidation_count": 0,
        "validation_liquidation_count": 0,
        "train_account_wipeout_count": 0,
        "validation_account_wipeout_count": 0,
        "train_return_stress_15bps_proxy": 0.04,
        "validation_return_stress_15bps_proxy": 0.03,
        "train_return_stress_20bps_proxy": 0.03,
        "validation_return_stress_20bps_proxy": 0.02,
    }
    row.update(overrides)
    return row


def test_material_positive_dominance_is_relaxed_under_mdd_guard() -> None:
    spec = module.RELAXED_PROFILE_SPECS[0]
    row = _base_row(train_return=0.03, validation_return=0.08)

    reasons = module._candidate_repair_reasons(row, spec)

    assert "train_below_validation_spike_risk" not in reasons
    assert (
        "train_below_validation_relaxed_material_positive_mdd_guarded"
        in module._candidate_relaxations(row, spec)
    )


def test_dominance_relaxation_does_not_override_10bps_rpt_gate() -> None:
    spec = module.RELAXED_PROFILE_SPECS[0]
    row = _base_row(
        train_return=0.03,
        validation_return=0.08,
        train_return_per_turnover_proxy_bps=8.0,
    )

    reasons = module._candidate_repair_reasons(row, spec)

    assert any(reason.startswith("train_rpt_8.000_not_above_10bps") for reason in reasons)


def test_tradfi_low_train_sample_can_be_warning_when_mdd_and_rpt_pass() -> None:
    spec = module.RELAXED_PROFILE_SPECS[0]
    row = _base_row(train_trade_event_count=1, validation_trade_event_count=12)

    reasons = module._candidate_repair_reasons(row, spec)

    assert not any(reason.startswith("train_events_1_below") for reason in reasons)
    assert "low_train_sample_relaxed_tradfi_mdd_guarded" in module._candidate_relaxations(row, spec)


def test_non_material_crypto_low_sample_stays_rejected() -> None:
    spec = module.RELAXED_PROFILE_SPECS[1]
    row = _base_row(
        symbol="ETHUSDT",
        asset_group="crypto_core",
        train_return=0.01,
        validation_return=0.03,
        train_trade_event_count=1,
        validation_trade_event_count=10,
    )

    reasons = module._candidate_repair_reasons(row, spec)

    assert any(reason.startswith("train_events_1_below") for reason in reasons)


def test_selection_reasons_relax_dominance_and_gross_only_when_mdd_ok() -> None:
    ok = _base_row(train_return=0.05, validation_return=0.09, gross_notional_fraction=9.0)
    high_mdd = {**ok, "validation_mdd": 0.25}

    assert module._selection_reasons(ok, max_gross=8.0) == []
    assert any(
        reason.startswith("validation_mdd_0.2500_above")
        for reason in module._selection_reasons(high_mdd, max_gross=8.0)
    )


def test_relaxed_hybrid_row_never_enables_real_flags() -> None:
    row = _base_row(profile_id="hybrid", train_return=0.05, validation_return=0.06)

    out = module._apply_relaxed_hybrid_row_fields(row)

    assert out["ready_for_real"] is False
    assert out["real_money_execution"] is False
    assert out["real_execution_allowed"] is False
    assert out["ready_for_paper"] is True
