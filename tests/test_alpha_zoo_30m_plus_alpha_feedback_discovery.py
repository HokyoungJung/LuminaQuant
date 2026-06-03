from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "research" / "run_alpha_zoo_30m_plus_alpha_feedback_discovery.py"
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_30m_plus_alpha_feedback_discovery", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _gate_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "train_return": 0.08,
        "validation_return": 0.04,
        "locked_oos_return": 0.03,
        "validation_mdd": 0.05,
        "train_trade_event_count": 100,
        "validation_trade_event_count": 40,
        "locked_oos_trade_event_count": 25,
        "train_validation_return_ratio": 2.0,
        "locked_oos_liquidation_count": 0,
        "locked_oos_account_wipeout_count": 0,
        "family": "volatility_adjusted_trend_persistence",
        "symbol": "ETHUSDT",
        "timeframe": "30m",
        "decision": "paper_testnet_candidate_after_fill_preflight",
        "model_id": "synthetic",
        "train_return_per_turnover_proxy_bps": 12.0,
        "validation_return_per_turnover_proxy_bps": 12.0,
        "locked_oos_return_per_turnover_proxy_bps": 12.0,
        "feature_backed": False,
        "feature_coverage": {},
    }
    row.update(overrides)
    return row


def test_validate_timeframes_enforces_30m_floor() -> None:
    assert MODULE._validate_timeframes(["30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]) == (
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    )

    with pytest.raises(ValueError, match="below 30m"):
        MODULE._validate_timeframes(["15m"])

    assert MODULE._timeframe_hours("1d") == 24.0
    assert MODULE._pandas_rule("1d") == "1D"


def test_native_30m_loader_builds_base_before_resampling(tmp_path: Path) -> None:
    symbol_dir = tmp_path / "TESTUSDT"
    symbol_dir.mkdir()
    dt = pd.date_range("2025-01-01", periods=90, freq="1min")
    prices = np.arange(100.0, 190.0)
    pl.DataFrame(
        {
            "datetime": dt,
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices + 0.5,
            "volume": np.ones(len(dt)),
        }
    ).write_parquet(symbol_dir / "2025-test.parquet")

    base = MODULE._load_symbol_base_30m("TESTUSDT", data_root=tmp_path)
    bars = MODULE.load_requested_bars(["TESTUSDT"], timeframes=("30m", "1h"), data_root=tmp_path)

    assert MODULE.BAR_CONSTRUCTION == "native_1s_to_30m_base_then_requested_timeframe"
    assert len(base) == 3
    assert base["datetime"].diff().dropna().eq(pd.Timedelta(minutes=30)).all()
    assert len(bars[("TESTUSDT", "30m")]) > len(bars[("TESTUSDT", "1h")])
    assert bars[("TESTUSDT", "30m")].iloc[0]["close"] == 129.5


def test_candidate_score_ignores_locked_oos_and_feature_coverage() -> None:
    row = {
        "train_return": 0.08,
        "validation_return": 0.04,
        "validation_mdd": 0.01,
        "validation_trade_event_count": 40,
        "train_return_per_turnover_proxy_bps": 11.0,
        "validation_return_per_turnover_proxy_bps": 12.0,
        "locked_oos_return": 1.0,
        "feature_coverage": {"locked_oos": 1.0},
    }
    changed_oos = dict(row, locked_oos_return=-1.0, feature_coverage={"locked_oos": 0.0})

    assert MODULE._candidate_score(row) == MODULE._candidate_score(changed_oos)


def test_gate_rejects_validation_spike_and_exact_10bps_rpt() -> None:
    spike = MODULE._gate_candidate(
        _gate_row(train_return=0.03, validation_return=0.04, train_validation_return_ratio=0.75)
    )
    exact_cost = MODULE._gate_candidate(_gate_row(validation_return_per_turnover_proxy_bps=10.0))

    assert spike["train_dominant_sample_gate_pass"] is False
    assert spike["paper_candidate_gate_pass"] is False
    assert any("below_validation_return" in reason for reason in spike["rejection_reasons"])
    assert exact_cost["execution_efficiency_proxy_gate_pass"] is False
    assert exact_cost["primary_10bps_promotion_gate_pass"] is False
    assert any(
        "validation_return_per_turnover_proxy_bps_10.000" in r
        for r in exact_cost["rejection_reasons"]
    )
    assert exact_cost["ready_for_real"] is False
    assert exact_cost["real_money_execution"] is False


def test_feature_coverage_policy_fails_closed_including_locked_oos_gate() -> None:
    row = MODULE._gate_candidate(
        _gate_row(
            feature_backed=True,
            feature_coverage={"train": 0.90, "validation": 0.85, "locked_oos": 0.50},
        )
    )

    assert MODULE.MIN_TRAIN_FEATURE_COVERAGE == 0.80
    assert MODULE.MIN_VALIDATION_FEATURE_COVERAGE == 0.80
    assert MODULE.MIN_LOCKED_OOS_FEATURE_COVERAGE == 0.80
    assert MODULE.max_asof_feature_age_hours("4h") == 24.0
    assert MODULE.max_asof_feature_age_hours("6h") == 36.0
    assert row["train_dominant_sample_gate_pass"] is True
    assert row["feature_coverage_gate_pass"] is False
    assert row["paper_candidate_gate_pass"] is False
    assert "locked_oos_feature_coverage_0.500_below_0.80" in row["rejection_reasons"]


def test_attach_features_marks_stale_or_missing_points_invalid() -> None:
    bars = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 12:00", "2025-01-02 12:00"]
            ),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    features = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01 00:00"]),
            "funding_rate": [0.0001],
            "open_interest": [100.0],
            "taker_buy_quote_volume": [60.0],
            "taker_sell_quote_volume": [40.0],
        }
    )

    attached = MODULE._attach_features_with_age(bars, features, timeframe="4h")

    assert attached["feature_valid"].tolist() == [True, True, False]
    assert attached.loc[0, "taker_buy_sell_imbalance"] == pytest.approx(0.2)


def test_handoff_summary_and_selected_rows_are_fail_closed() -> None:
    paper = MODULE._gate_candidate(_gate_row(model_id="paper"))
    reject = MODULE._gate_candidate(
        _gate_row(
            model_id="reject",
            train_return=0.05,
            validation_return=0.20,
            train_validation_return_ratio=0.25,
        )
    )
    ranked = MODULE._rank_rows([reject, paper])
    selected = MODULE._selected_output_rows(ranked, top_n=1)
    handoff = MODULE._paper_testnet_handoff([paper])
    empty_handoff = MODULE._paper_testnet_handoff([])
    summary = MODULE._summary(ranked)

    assert [row["model_id"] for row in selected] == ["reject", "paper"]
    assert handoff["status"] == "paper_testnet_candidates_available"
    assert handoff["ready_for_real"] is False
    assert handoff["real_money_execution"] is False
    assert handoff["preflight"]["required_mode"] == "paper_or_testnet_only"
    assert empty_handoff["status"] == "no_paper_candidates"
    assert summary["ready_for_real"] is False
    assert summary["real_money_execution"] is False
