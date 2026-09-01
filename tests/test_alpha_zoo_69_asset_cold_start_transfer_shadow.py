from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_cold_start_transfer_shadow as module
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69


def _stream(
    *,
    symbol: str,
    model_id: str,
    donor_score: float,
    shadow_score: float,
    validation_return: float,
    validation_rpt: float = 50.0,
    validation_mdd: float = 0.02,
) -> broad69.CandidateStream:
    index = pd.date_range("2026-04-04 04:00", periods=4, freq="h")
    row = {
        "model_id": model_id,
        "symbol": symbol,
        "asset_group": broad69._asset_group(symbol),
        "family": "cross_sectional_momentum_rank",
        "timeframe": "1h",
        "notional_fraction": 0.20,
        "donor_selection_score": donor_score,
        "donor_quality_score": donor_score / 10.0,
        "shadow_validation_score": shadow_score,
        "train_trade_event_count": 0,
        "validation_trade_event_count": 4,
        "train_return": 0.0,
        "validation_return": validation_return,
        "train_mdd": 0.0,
        "validation_mdd": validation_mdd,
        "train_return_per_turnover_proxy_bps": None,
        "validation_return_per_turnover_proxy_bps": validation_rpt,
    }
    returns = pd.Series([validation_return / 4] * 4, index=index)
    position = pd.Series([1.0, 1.0, 0.0, 0.0], index=index)
    return broad69.CandidateStream(row=row, returns=returns, position=position)


def test_primary_shadow_selection_uses_donor_score_not_target_validation_score() -> None:
    low_target_high_donor = _stream(
        symbol="AAPLUSDT",
        model_id="donor_frozen_winner",
        donor_score=100.0,
        shadow_score=-5.0,
        validation_return=-0.05,
    )
    high_target_low_donor = _stream(
        symbol="MSFTUSDT",
        model_id="target_oracle_winner",
        donor_score=1.0,
        shadow_score=100.0,
        validation_return=0.50,
    )

    selected, mult = module._select_shadow_sleeves(
        [high_target_low_donor, low_target_high_donor],
        max_sleeves=1,
        max_gross=0.2,
        selection_policy="primary",
        use_target_validation_score=False,
    )

    assert len(mult) == 1
    assert selected[0].row["model_id"] == "donor_frozen_winner"
    assert selected[0].row["primary_shadow_selected"] is True


def test_oracle_shadow_selection_is_explicitly_validation_diagnostic() -> None:
    negative = _stream(
        symbol="AAPLUSDT",
        model_id="negative",
        donor_score=100.0,
        shadow_score=1000.0,
        validation_return=-0.05,
    )
    positive = _stream(
        symbol="MSFTUSDT",
        model_id="positive",
        donor_score=1.0,
        shadow_score=10.0,
        validation_return=0.10,
    )

    selected, _ = module._select_shadow_sleeves(
        [negative, positive],
        max_sleeves=2,
        max_gross=0.4,
        selection_policy="oracle",
        use_target_validation_score=True,
    )

    assert [stream.row["model_id"] for stream in selected] == ["positive"]
    assert positive.row["oracle_shadow_selected"] is True


def test_shadow_portfolio_never_enables_paper_or_real_flags() -> None:
    windows = broad69.SplitWindows(
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-04-04 03:00")),
        validation=(pd.Timestamp("2026-04-04 04:00"), pd.Timestamp("2026-05-30 03:00")),
    )
    stream = _stream(
        symbol="AAPLUSDT",
        model_id="aapl_shadow",
        donor_score=10.0,
        shadow_score=10.0,
        validation_return=0.10,
    )

    row = module._portfolio_metrics(
        portfolio_id="shadow",
        selection_policy="donor_frozen_primary",
        streams=[stream],
        multipliers=np.array([1.0]),
        windows=windows,
    )

    assert row["target_validation_pnl_used_for_selection"] is False
    assert row["ready_for_paper"] is False
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False
    assert row["real_execution_allowed"] is False
    assert row["train_return"] == pytest.approx(0.0)
    assert "target_symbols_have_no_train_rows" in row["rejection_reasons"]


def test_donor_quality_penalizes_validation_spikes_and_low_execution_efficiency() -> None:
    robust = {
        "train_return": 0.20,
        "validation_return": 0.10,
        "train_mdd": 0.02,
        "validation_mdd": 0.02,
        "train_return_per_turnover_proxy_bps": 60.0,
        "validation_return_per_turnover_proxy_bps": 50.0,
        "train_trade_event_count": 80,
        "validation_trade_event_count": 20,
        "profile_objective_score": 1.0,
    }
    spiky = {**robust, "train_return": 0.03, "validation_return": 0.30}
    low_rpt = {
        **robust,
        "train_return_per_turnover_proxy_bps": 4.0,
        "validation_return_per_turnover_proxy_bps": 5.0,
    }

    assert module._donor_quality_score(robust) > module._donor_quality_score(spiky)
    assert module._donor_quality_score(robust) > module._donor_quality_score(low_rpt)


def test_donor_similarity_rejects_targets_without_enough_validation_coverage() -> None:
    report = {
        "symbols": {
            "AAPLUSDT": {
                "timeframes": {
                    "1h": {
                        "train_rows": 0,
                        "validation_rows": module.MIN_TARGET_VALIDATION_ROWS - 1,
                    }
                }
            }
        }
    }
    donor = {
        "symbol": "NVDAUSDT",
        "profile_id": "profile",
        "timeframe": "1h",
        "optuna_params": {"timeframe": "1h", "lookback_bars": 12},
        "donor_quality_score": 10.0,
    }

    score = module._donor_similarity_score(
        target_symbol="AAPLUSDT",
        target_profile_id="profile",
        donor_row=donor,
        train_eligibility=report,
    )

    assert score == float("-inf")
