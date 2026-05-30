from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_efficiency_repair_optuna as module
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69


def _base_row(**overrides):
    row = {
        "train_return": 0.20,
        "validation_return": 0.10,
        "train_mdd": 0.04,
        "validation_mdd": 0.03,
        "train_return_per_turnover_proxy_bps": 55.0,
        "validation_return_per_turnover_proxy_bps": 60.0,
        "train_trade_event_count": 100,
        "validation_trade_event_count": 35,
        "dominant_anchor_abs_corr": 0.10,
    }
    row.update(overrides)
    return row


def test_candidate_live_efficiency_score_penalizes_validation_spike_and_low_rpt() -> None:
    spec = module.EFFICIENCY_PROFILE_SPECS[0]

    robust = module._candidate_live_efficiency_score(_base_row(), spec)
    spiky = module._candidate_live_efficiency_score(
        _base_row(train_return=0.05, validation_return=0.30), spec
    )
    low_rpt = module._candidate_live_efficiency_score(
        _base_row(
            train_return_per_turnover_proxy_bps=8.0,
            validation_return_per_turnover_proxy_bps=9.0,
        ),
        spec,
    )

    assert robust > spiky
    assert robust > low_rpt


def test_stress_metrics_subtract_incremental_cost_from_return_and_rpt() -> None:
    metrics = {
        "train_return": 0.20,
        "validation_return": 0.10,
        "train_return_per_turnover_proxy_bps": 50.0,
        "validation_return_per_turnover_proxy_bps": 40.0,
    }

    stressed = module._stress_metrics(metrics, {"train": 20.0, "validation": 10.0})

    assert stressed["train_return_stress_15bps_proxy"] == pytest.approx(0.19)
    assert stressed["validation_return_stress_20bps_proxy"] == pytest.approx(0.09)
    assert stressed["train_return_per_turnover_stress_20bps_proxy"] == pytest.approx(40.0)
    assert stressed["validation_return_per_turnover_stress_15bps_proxy"] == pytest.approx(35.0)


def test_selection_reasons_require_positive_20bps_stress() -> None:
    row = {
        "train_return": 0.05,
        "validation_return": 0.04,
        "train_mdd": 0.01,
        "validation_mdd": 0.01,
        "gross_notional_fraction": 2.0,
        "train_return_per_turnover_proxy_bps": 30.0,
        "validation_return_per_turnover_proxy_bps": 30.0,
        "train_liquidation_count": 0,
        "validation_liquidation_count": 0,
        "train_account_wipeout_count": 0,
        "validation_account_wipeout_count": 0,
        "train_return_stress_15bps_proxy": 0.01,
        "validation_return_stress_15bps_proxy": 0.01,
        "train_return_stress_20bps_proxy": -0.001,
        "validation_return_stress_20bps_proxy": 0.01,
        "low_efficiency_notional_share": 0.0,
        "low_sample_notional_share": 0.0,
    }

    reasons = module._selection_reasons(row)

    assert "train_return_stress_20bps_proxy_not_positive" in reasons


def test_weighted_turnover_events_use_selected_positive_multipliers() -> None:
    def stream(model_id: str, events: int, notional: float) -> broad69.CandidateStream:
        row = {
            "model_id": model_id,
            "notional_fraction": notional,
            "train_trade_event_count": events,
            "validation_trade_event_count": events // 2,
        }
        idx = pd.date_range("2026-01-01", periods=2, freq="h")
        returns = pd.Series([0.0, 0.0], index=idx)
        return broad69.CandidateStream(row=row, returns=returns, position=returns)

    turnover, events = module._weighted_turnover_events(
        [stream("a", 10, 0.5), stream("b", 20, 0.25)], np.array([0.2, 0.0])
    )

    assert turnover["train"] == pytest.approx(1.0)
    assert turnover["validation"] == pytest.approx(0.5)
    assert events["train"] == 10
    assert events["validation"] == 5


def test_source_rows_without_train_data_are_not_eligible_for_repair() -> None:
    report = {
        "symbols": {
            "OLDUSDT": {"timeframes": {"1h": {"train_eligible": True}}},
            "NEWUSDT": {"timeframes": {"1h": {"train_eligible": False}}},
        }
    }

    assert module._source_row_train_eligible({"symbol": "OLDUSDT", "timeframe": "1h"}, report)
    assert not module._source_row_train_eligible({"symbol": "NEWUSDT", "timeframe": "1h"}, report)


def test_efficiency_allocation_filters_rejected_sleeves() -> None:
    idx = pd.date_range("2026-01-01", periods=2, freq="h")

    def stream(model_id: str, reasons: list[str]) -> broad69.CandidateStream:
        row = {
            "model_id": model_id,
            "efficiency_repair_reasons": reasons,
            "live_efficiency_score": 1.0,
        }
        returns = pd.Series([0.0, 0.0], index=idx)
        return broad69.CandidateStream(row=row, returns=returns, position=returns)

    allocatable = module._allocatable_efficiency_streams(
        [stream("valid", []), stream("validation_only", ["train_events_0_below_20"])]
    )

    assert [item.row["model_id"] for item in allocatable] == ["valid"]
