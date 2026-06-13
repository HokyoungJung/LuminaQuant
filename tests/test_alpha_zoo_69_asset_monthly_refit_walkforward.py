from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import run_alpha_zoo_69_asset_monthly_refit_walkforward as module


def _candidate(
    label: str,
    *,
    family: str = "profile_optuna",
    daily_return: float = 0.01,
) -> module.CandidateResult:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    return module.CandidateResult(
        family=family,
        candidate_label=label,
        source_profile_id=label,
        row={
            "profile_id": label,
            "selection_reasons": [],
            "uses_locked_oos_for_selection": False,
        },
        returns=pd.Series(daily_return, index=index, dtype=float),
    )


def test_validate_timeframes_exact_30m_to_1d() -> None:
    assert module._validate_timeframes_30m_to_1d(
        ["30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1h"]
    ) == ("30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        module._validate_timeframes_30m_to_1d(["15m"])
    with pytest.raises(ValueError, match="unsupported timeframe"):
        module._validate_timeframes_30m_to_1d(["2d"])


def test_monthly_fold_boundaries_do_not_overlap() -> None:
    folds = module.build_monthly_folds(
        train_start=pd.Timestamp("2025-01-01"),
        first_oos_start=pd.Timestamp("2025-09-01"),
        latest_data=pd.Timestamp("2025-10-15"),
        bar_minutes=30,
    )
    assert folds[0].train[1] < folds[0].validation[0]
    assert folds[0].validation[1] < folds[0].locked_oos[0]
    assert folds[0].refit_at == folds[0].locked_oos[0]


def test_aggregate_rows_recomputes_extended_metrics() -> None:
    rows = []
    for idx, value in enumerate([0.10, -0.05, 0.03], start=1):
        rows.append(
            {
                "candidate_label": "c",
                "family": "f",
                "clean_promotion_eligible": True,
                "train": {"total_return": 0.2, "mdd": 0.05},
                "validation": {"total_return": 0.1, "mdd": 0.04},
                "locked_oos": {"total_return": value, "mdd": 0.06 + idx / 1000},
                "ready_for_paper": True,
            }
        )
    agg = module._aggregate_rows(rows)[0]
    assert agg["compounded_oos_return"] == pytest.approx((1.10 * 0.95 * 1.03) - 1.0)
    assert agg["positive_oos_folds"] == 2
    assert agg["max_loss_streak"] == 1
    assert agg["gain_loss_ratio"] > 0.0
    assert agg["annualized_oos_return_approx"] != 0.0
    assert agg["monthly_volatility"] > 0.0
    assert "monthly_skew" in agg
    assert "profit_factor" in agg
    assert "omega_0" in agg
    assert "monthly_equity_mdd" in agg
    assert agg["clean_promotion_eligible"] is True


def test_aggregate_rows_rank_by_compounded_return_before_hit_count() -> None:
    def row(label: str, value: float) -> dict[str, object]:
        return {
            "candidate_label": label,
            "family": "f",
            "clean_promotion_eligible": True,
            "train": {"total_return": 0.2, "mdd": 0.05},
            "validation": {"total_return": 0.1, "mdd": 0.04},
            "locked_oos": {"total_return": value, "mdd": 0.03},
            "ready_for_paper": True,
        }

    ranked = module._aggregate_rows(
        [
            row("steady", 0.02),
            row("steady", 0.02),
            row("steady", 0.02),
            row("selective_high_comp", 0.20),
            row("selective_high_comp", 0.0),
            row("selective_high_comp", 0.0),
        ]
    )

    assert ranked[0]["candidate_label"] == "selective_high_comp"
    assert ranked[0]["positive_oos_folds"] < ranked[1]["positive_oos_folds"]


def test_timeframe_coverage_summary_reports_one_day() -> None:
    summary = module._timeframe_coverage_summary(
        {
            "timeframes": {
                "1d": {
                    "BTCUSDT": {
                        "rows": 3,
                        "earliest": "2025-01-01T00:00:00",
                        "latest": "2025-01-03T00:00:00",
                    },
                    "NEWUSDT": {"rows": 0, "earliest": None, "latest": None},
                }
            }
        }
    )
    assert summary["1d"]["symbols_with_rows"] == 1
    assert summary["1d"]["symbols_without_rows"] == 1
    assert "complete_bucket_policy" in summary["1d"]


def test_load_all_bars_marks_missing_new_symbols_without_failing(tmp_path: Path) -> None:
    data_root = tmp_path / "exchange=binance"
    btc_dir = data_root / "symbol=BTCUSDT" / "timeframe=1m" / "date=2025-01-01"
    btc_dir.mkdir(parents=True)
    datetimes = pd.date_range("2025-01-01", periods=30, freq="1min")
    pl.DataFrame(
        {
            "datetime": datetimes.to_pydatetime().tolist(),
            "open": [100.0] * len(datetimes),
            "high": [101.0] * len(datetimes),
            "low": [99.0] * len(datetimes),
            "close": [100.5] * len(datetimes),
            "volume": [10.0] * len(datetimes),
        }
    ).write_parquet(btc_dir / "part.parquet")

    bars, coverage = module.broad69.load_all_bars(
        ("BTCUSDT", "ANTHROPICUSDT"),
        data_root=data_root,
        timeframes=("30m",),
    )

    assert not bars[("BTCUSDT", "30m")].empty
    assert bars[("ANTHROPICUSDT", "30m")].empty
    assert coverage["requested_symbol_count"] == 2
    assert coverage["loaded_symbol_count"] == 1
    assert coverage["missing_symbols"] == ["ANTHROPICUSDT"]
    assert coverage["timeframes"]["30m"]["ANTHROPICUSDT"]["missing_reason"] == (
        "missing_direct_1m_parquet"
    )


def test_cost_model_is_pinned_to_10bps_round_trip() -> None:
    assert module.DEFAULT_SLIPPAGE_BPS == 10.0
    assert module.broad69.PRIMARY_ROUND_TRIP_COST_BPS == 10.0

    bars = pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
        }
    )
    signal = pd.Series([0.0, 1.0, 1.0, 0.0], dtype=float).to_numpy()
    result = module.broad69.simulate_symbol(
        bars,
        signal,
        integer_leverage=1,
        allocation_fraction=1.0,
        round_trip_cost_bps=module.DEFAULT_SLIPPAGE_BPS,
    )

    # Two half-turnover transitions, 0->1 and 1->0, equal one full 10bps round trip.
    assert result.returns.sum() == pytest.approx(-0.001)


def test_leaf_strategy_material_filter_rejects_nested_hybrid_inputs() -> None:
    leaf = _candidate("profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna")
    hybrid = _candidate("cross_candidate_hybrid:hybrid_v3_5", family="cross_candidate_hybrid")
    selector = _candidate(
        "dynamic_conviction_switch:t0.90_risk_capped_fallback",
        family="dynamic_conviction_switch",
    )
    selected_optuna = _candidate("profile_optuna:selected_optuna")

    assert module._leaf_strategy_material_candidate(leaf) is True
    assert module._leaf_strategy_material_candidate(hybrid) is False
    assert module._leaf_strategy_material_candidate(selector) is False
    assert module._leaf_strategy_material_candidate(selected_optuna) is False


def test_strict_calm_leaf_selector_is_clean_and_does_not_peek_oos() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    calm_bad_oos = pd.Series(0.003, index=index, dtype=float)
    calm_bad_oos.loc["2025-03-01":"2025-03-31"] = -0.020
    lower_validation_good_oos = pd.Series(0.001, index=index, dtype=float)
    lower_validation_good_oos.loc["2025-03-01":"2025-03-31"] = 0.050
    nested = pd.Series(0.020, index=index, dtype=float)
    candidates = [
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label=(
                "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
            ),
            source_profile_id="strict_balanced",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=calm_bad_oos,
        ),
        module.CandidateResult(
            family="profile_optuna",
            candidate_label="profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna",
            source_profile_id="profile_growth",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=lower_validation_good_oos,
        ),
        module.CandidateResult(
            family="dynamic_conviction_switch",
            candidate_label="dynamic_conviction_switch:t0.90_risk_capped_fallback",
            source_profile_id="nested_selector",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=nested,
        ),
    ]

    selected = module._strict_calm_leaf_selector_candidates(candidates, fold)
    row = module._evaluate_candidate(selected[0], fold)

    assert row["family"] == "strict_calm_leaf_selector"
    assert row["selected_candidate_label"] == (
        "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
    )
    assert row["uses_locked_oos_for_selection"] is False
    assert row["clean_promotion_eligible"] is True
    assert row["nested_hybrid_dependency"] is False
    assert "dynamic_conviction_switch:t0.90_risk_capped_fallback" not in row["final_weights"]
    assert row["locked_oos"]["total_return"] < 0.0


def test_leaf_strategy_material_filter_rejects_calendar_primary_inputs() -> None:
    def candidate_with_row(label: str, row: dict[str, object], family: str = "profile_optuna"):
        candidate = _candidate(label, family=family)
        candidate.row.update(row)
        return candidate

    clean_monthly_refit = _candidate("profile_optuna:monthly_refit_validation_cadence")
    clean_monthly_refit.row.update(
        {
            "profile_id": "monthly_refit_validation_cadence",
            "profile_kind": "leaf_momentum_profile",
            "protocol_note": "monthly_refit is a cadence, not calendar alpha",
        }
    )
    calendar_family = candidate_with_row(
        "calendar_rotation:btc_may_short",
        {"profile_id": "calendar_rotation_btc_may_short"},
        family="calendar_rotation",
    )
    calendar_flag = candidate_with_row(
        "profile_optuna:flagged_calendar_primary",
        {"calendar_primary": "true"},
    )
    calendar_params = candidate_with_row(
        "profile_optuna:fixed_month_entry",
        {"params_json": '{"entry_months": [5], "lookback": 20}'},
    )
    rejected_calendar = candidate_with_row(
        "profile_optuna:rejected_calendar_rule",
        {"rejection_reasons": ["calendar_fixed_month_alpha"]},
    )

    assert module._leaf_strategy_material_candidate(clean_monthly_refit) is True
    assert module._leaf_strategy_material_candidate(calendar_family) is False
    assert module._leaf_strategy_material_candidate(calendar_flag) is False
    assert module._leaf_strategy_material_candidate(calendar_params) is False
    assert module._leaf_strategy_material_candidate(rejected_calendar) is False


def test_clean_material_filters_reject_disguised_nested_row_references() -> None:
    disguised_leaf = _candidate(
        "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna",
        family="profile_optuna",
    )
    disguised_leaf.row["final_weights"] = {"cross_candidate_hybrid:hybrid_v3_5": 1.0}

    assert module._leaf_strategy_material_candidate(disguised_leaf) is False
    assert module._clean_source_candidate(disguised_leaf) is False
    assert module._clean_downstream_candidate(disguised_leaf) is False


def test_dynamic_conviction_switch_uses_train_validation_only() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    aggressive_label = "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna"
    aggressive_returns = pd.Series(0.04, index=index, dtype=float)
    aggressive_returns.loc["2025-03-01":"2025-03-31"] = -0.03
    fallback_returns = pd.Series(0.001, index=index, dtype=float)
    candidates = [
        module.CandidateResult(
            family="profile_optuna",
            candidate_label=aggressive_label,
            source_profile_id="aggressive",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=aggressive_returns,
        ),
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label="strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna",
            source_profile_id="fallback",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=fallback_returns,
        ),
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label="strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna",
            source_profile_id="fallback_growth",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=fallback_returns,
        ),
    ]

    switched = module._dynamic_conviction_switch_candidates(candidates, fold)

    assert switched
    assert all(candidate.row["uses_locked_oos_for_selection"] is False for candidate in switched)
    assert switched[0].row["selected_candidate_label"] == aggressive_label
    # The selector still picks the high train/validation leaf candidate even
    # though its locked OOS is deliberately worse than fallback in this fixture.
    assert module._period_metrics(switched[0].returns, fold.locked_oos)["total_return"] < 0.0


def test_dynamic_conviction_switch_emits_fallback_when_aggressive_pool_missing() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    fallback_label = "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
    fallback_returns = pd.Series(0.001, index=index, dtype=float)
    candidates = [
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label=fallback_label,
            source_profile_id="fallback",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=fallback_returns,
        ),
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label=(
                "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna"
            ),
            source_profile_id="fallback_growth",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=fallback_returns,
        ),
    ]

    switched = module._dynamic_conviction_switch_candidates(candidates, fold)

    assert len(switched) == 72
    assert all(candidate.row["aggressive_candidate_label"] is None for candidate in switched)
    assert all(
        candidate.row["selected_candidate_label"] == fallback_label for candidate in switched
    )
    assert all(set(candidate.row["final_weights"]) == {fallback_label} for candidate in switched)
    assert any(candidate.row.get("risk_scale", 1.0) > 1.0 for candidate in switched)
    assert any("_val_ret02_calmar80_gate" in candidate.candidate_label for candidate in switched)
    assert all(candidate.row["uses_locked_oos_for_selection"] is False for candidate in switched)


def test_dynamic_conviction_switch_keeps_scaled_gate_cash_folds_when_gate_fails() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    fallback_label = "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
    fallback_returns = pd.Series(0.001, index=index, dtype=float)
    fallback_returns.loc["2025-02-01":"2025-02-28"] = 0.0
    candidates = [
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label=fallback_label,
            source_profile_id="fallback",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=fallback_returns,
        ),
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label=(
                "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna"
            ),
            source_profile_id="fallback_growth",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=fallback_returns,
        ),
    ]

    switched = module._dynamic_conviction_switch_candidates(candidates, fold)
    scaled_cash = [
        candidate
        for candidate in switched
        if candidate.candidate_label.endswith("_val_ret02_calmar80_gate_val_mdd20_scaled")
    ]

    assert len(scaled_cash) == 8
    assert all(
        candidate.row["selected_candidate_label"] == "cash_validation_strength_guard"
        for candidate in scaled_cash
    )
    assert all(candidate.row["final_weights"] == {} for candidate in scaled_cash)
    assert all(
        module._period_metrics(candidate.returns, fold.locked_oos)["total_return"] == 0.0
        for candidate in scaled_cash
    )
    mdd30_scaled_cash = [
        candidate
        for candidate in switched
        if candidate.candidate_label.endswith("_val_ret02_calmar80_gate_val_mdd30_scaled")
    ]
    assert len(mdd30_scaled_cash) == 8
    assert all(candidate.row["target_validation_mdd"] == 0.30 for candidate in mdd30_scaled_cash)


def test_lagged_shadow_leaf_router_uses_only_prior_completed_oos_and_leaf_sources() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    relaxed_label = (
        "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna"
    )
    strict_label = "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
    relaxed_prior_winner_bad_current_oos = pd.Series(0.003, index=index, dtype=float)
    relaxed_prior_winner_bad_current_oos.loc["2025-03-01":"2025-03-31"] = -0.020
    strict_current_oos_winner = pd.Series(0.002, index=index, dtype=float)
    strict_current_oos_winner.loc["2025-03-01":"2025-03-31"] = 0.020
    nested = pd.Series(0.050, index=index, dtype=float)
    candidates = [
        module.CandidateResult(
            family="relaxed_efficiency",
            candidate_label=relaxed_label,
            source_profile_id=relaxed_label,
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=relaxed_prior_winner_bad_current_oos,
        ),
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label=strict_label,
            source_profile_id=strict_label,
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=strict_current_oos_winner,
        ),
        module.CandidateResult(
            family="dynamic_conviction_switch",
            candidate_label="dynamic_conviction_switch:t0.90_risk_capped_fallback",
            source_profile_id="nested_selector",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=nested,
        ),
    ]

    routed = module._lagged_shadow_leaf_router_candidates(
        candidates,
        fold,
        prior_completed_returns={
            relaxed_label: [0.00, -0.01, 0.12, 0.10],
            strict_label: [0.03, 0.02, 0.01, 0.00],
        },
        prior_completed_fold_ids=("2024-11", "2024-12", "2025-01", "2025-02"),
    )
    row = module._evaluate_candidate(routed[0], fold)

    assert row["candidate_label"] == module.LAGGED_SHADOW_LEAF_ROUTER_LABEL
    assert row["selected_candidate_label"] == relaxed_label
    assert row["router_branch"] == "lagged_shadow_leaf"
    assert row["online_update_cutoff_fold"] == "2025-02"
    assert row["lagged_shadow_history_tail"] == [0.12, 0.10]
    assert row["uses_locked_oos_for_selection"] is False
    assert row["current_fold_oos_used_for_weighting"] is False
    assert row["clean_promotion_eligible"] is False
    assert row["nested_hybrid_dependency"] is False
    assert "dynamic_conviction_switch:t0.90_risk_capped_fallback" not in row["final_weights"]
    # Current OOS is deliberately worse than the strict leaf, proving the
    # router did not choose by same-month locked OOS.
    assert row["locked_oos"]["total_return"] < 0.0
    scaled = [
        module._evaluate_candidate(candidate, fold)
        for candidate in routed
        if candidate.candidate_label.endswith("_lag_val_mdd20_cap150")
    ]
    assert len(scaled) == 1
    assert scaled[0]["risk_scale"] > 1.0
    assert scaled[0]["lagged_shadow_scale_applied"] is True
    assert scaled[0]["uses_locked_oos_for_selection"] is False
    assert scaled[0]["nested_hybrid_dependency"] is False
    preregistered = [
        module._evaluate_candidate(candidate, fold)
        for candidate in routed
        if candidate.candidate_label == module.PREREGISTERED_LAGGED_LEAF_ROUTER_LABEL
    ]
    assert len(preregistered) == 1
    assert preregistered[0]["selected_candidate_label"] == relaxed_label
    assert preregistered[0]["router_branch"] == "pre_registered_lagged_plus_validation_leaf"
    assert preregistered[0]["lagged_shadow_avg_window"] == 1
    assert preregistered[0]["lagged_shadow_history_tail"] == [0.10]
    assert preregistered[0]["lagged_shadow_validation_weight"] == pytest.approx(0.25)
    assert preregistered[0]["uses_locked_oos_for_selection"] is False
    assert preregistered[0]["current_fold_oos_used_for_weighting"] is False
    assert preregistered[0]["nested_hybrid_dependency"] is False
    assert preregistered[0]["clean_promotion_eligible"] is False


def test_preregistered_lagged_leaf_router_replay_uses_prior_leaf_history() -> None:
    strict_label = "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
    relaxed_label = (
        "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna"
    )

    def row(
        fold_id: str,
        label: str,
        family: str,
        *,
        train_return: float,
        train_mdd: float,
        validation_return: float,
        validation_mdd: float,
        oos_return: float,
    ) -> dict[str, object]:
        return {
            "fold_id": fold_id,
            "candidate_label": label,
            "source_profile_id": label,
            "profile_kind": "leaf",
            "family": family,
            "clean_promotion_eligible": True,
            "selection_reasons": [],
            "uses_locked_oos_for_selection": False,
            "same_month_self_feeding": False,
            "current_fold_oos_used_for_weighting": False,
            "post_oos_research_variant": False,
            "requires_fresh_forward_shadow": False,
            "train": {"total_return": train_return, "mdd": train_mdd},
            "validation": {
                "total_return": validation_return,
                "mdd": validation_mdd,
                "calmar": validation_return / max(validation_mdd, 0.01),
            },
            "locked_oos": {"total_return": oos_return, "mdd": 0.03},
        }

    rows: list[dict[str, object]] = []
    for idx, fold_id in enumerate(["2025-01", "2025-02", "2025-03", "2025-04"], start=1):
        rows.extend(
            [
                row(
                    fold_id,
                    strict_label,
                    "strict_efficiency",
                    train_return=0.10,
                    train_mdd=0.10,
                    validation_return=0.05,
                    validation_mdd=0.08,
                    oos_return=0.01 * idx,
                ),
                row(
                    fold_id,
                    relaxed_label,
                    "relaxed_efficiency",
                    train_return=0.10,
                    train_mdd=0.10,
                    validation_return=0.05,
                    validation_mdd=0.08,
                    oos_return=0.00 if idx < 4 else 0.10,
                ),
            ]
        )
    rows.extend(
        [
            row(
                "2025-05",
                strict_label,
                "strict_efficiency",
                train_return=0.10,
                train_mdd=0.10,
                validation_return=0.05,
                validation_mdd=0.08,
                oos_return=-0.25,
            ),
            row(
                "2025-05",
                relaxed_label,
                "relaxed_efficiency",
                train_return=0.20,
                train_mdd=0.15,
                validation_return=0.30,
                validation_mdd=0.12,
                oos_return=0.42,
            ),
        ]
    )

    replayed = module._append_preregistered_lagged_leaf_router_rows(rows)
    router_rows = [
        item
        for item in replayed
        if item["candidate_label"] == module.PREREGISTERED_LAGGED_LEAF_ROUTER_LABEL
    ]

    assert len(router_rows) == 5
    assert router_rows[-1]["selected_candidate_label"] == relaxed_label
    assert router_rows[-1]["router_branch"] == "pre_registered_lagged_plus_validation_leaf"
    assert router_rows[-1]["lagged_shadow_history_count"] == 4
    assert router_rows[-1]["lagged_shadow_history_tail"] == [0.10]
    assert router_rows[-1]["locked_oos"]["total_return"] == pytest.approx(0.42)
    assert router_rows[-1]["uses_locked_oos_for_selection"] is False
    assert router_rows[-1]["post_oos_research_variant"] is True
    assert router_rows[-1]["requires_fresh_forward_shadow"] is True
    assert router_rows[-1]["nested_hybrid_dependency"] is False


def test_dynamic_aware_hybrid_is_disabled_for_nested_hybrid_materials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dynamic-aware nested hybrid optimizer should not run")

    monkeypatch.setattr(module.optuna_hybrid, "_run_optuna", fail_if_called)
    candidates = [
        _candidate(
            "dynamic_conviction_switch:t0.90_risk_capped_fallback",
            family="dynamic_conviction_switch",
            daily_return=0.010,
        ),
        _candidate(
            "cross_candidate_hybrid:hybrid_v3_6_train_validation_fit",
            family="cross_candidate_hybrid",
            daily_return=0.006,
        ),
        _candidate(
            "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna",
            family="strict_efficiency",
            daily_return=0.002,
        ),
    ]

    out = module._dynamic_aware_hybrid_candidates(candidates, fold, hybrid_trials=8, seed=7)

    assert out == []


def test_meta_portfolio_candidates_use_leaf_materials_only() -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    nested_label = "validation_selector:top_clean"
    candidates = [
        _candidate(
            "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna", daily_return=0.010
        ),
        _candidate(
            "profile_optuna:balanced_mdd12_gross5_69_asset_profile_optuna", daily_return=0.008
        ),
        _candidate(nested_label, family="validation_selector", daily_return=0.050),
    ]

    out = module._meta_portfolio_candidates(candidates, fold)

    assert out
    assert all(nested_label not in candidate.row.get("final_weights", {}) for candidate in out)


def test_fixed_risk_enhanced_blend_is_disabled_for_nested_materials() -> None:
    candidates = [
        _candidate(
            "dynamic_conviction_switch:t0.85_risk_capped_fallback",
            family="dynamic_conviction_switch",
            daily_return=0.010,
        ),
        _candidate(
            "dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit",
            family="dynamic_aware_hybrid",
            daily_return=0.004,
        ),
    ]

    assert module._fixed_risk_enhanced_blend_candidates(candidates) == []


def test_fixed_relaxed_dynamic_blend_is_disabled_for_nested_hybrids() -> None:
    index = pd.date_range("2025-01-01", "2025-01-05", freq="1D")
    candidates = [
        module.CandidateResult(
            family="relaxed_efficiency",
            candidate_label="relaxed_efficiency:hybrid_v3_5",
            source_profile_id="relaxed",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=pd.Series(0.01, index=index, dtype=float),
        ),
        module.CandidateResult(
            family="dynamic_aware_hybrid",
            candidate_label="dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit",
            source_profile_id="dynamic",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=pd.Series(0.01, index=index, dtype=float),
        ),
    ]

    assert module._fixed_relaxed_dynamic_blend_candidates(candidates) == []


def test_teacher_leaf_blend_uses_validation_only_leaf_material() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    high_validation_bad_oos = pd.Series(0.004, index=index, dtype=float)
    high_validation_bad_oos.loc["2025-02-01":"2025-02-28"] = 0.012
    high_validation_bad_oos.loc["2025-03-01":"2025-03-31"] = -0.020
    lower_validation_good_oos = pd.Series(0.003, index=index, dtype=float)
    lower_validation_good_oos.loc["2025-02-01":"2025-02-28"] = 0.004
    lower_validation_good_oos.loc["2025-03-01":"2025-03-31"] = 0.010
    nested = pd.Series(0.050, index=index, dtype=float)
    candidates = [
        module.CandidateResult(
            family="relaxed_efficiency",
            candidate_label=(
                "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna"
            ),
            source_profile_id="relaxed_growth",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=high_validation_bad_oos,
        ),
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label=(
                "strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna"
            ),
            source_profile_id="strict_balanced",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=lower_validation_good_oos,
        ),
        module.CandidateResult(
            family="dynamic_conviction_switch",
            candidate_label="dynamic_conviction_switch:t0.90_risk_capped_fallback",
            source_profile_id="nested_selector",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=nested,
        ),
    ]

    blended = module._teacher_leaf_blend_candidates(candidates, fold)

    assert blended
    assert all(
        "dynamic_conviction_switch:t0.90_risk_capped_fallback" not in candidate.row["final_weights"]
        for candidate in blended
    )
    assert any(
        "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna"
        in candidate.row["final_weights"]
        for candidate in blended
    )
    assert all(candidate.row["uses_locked_oos_for_selection"] is False for candidate in blended)
    # The validation-led blend can still lose in the locked OOS fixture,
    # proving the construction did not choose by current OOS.
    assert any(
        module._period_metrics(candidate.returns, fold.locked_oos)["total_return"] < 0.0
        for candidate in blended
    )


def test_teacher_leaf_blend_evaluates_as_clean_non_nested_candidate() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    candidates = [
        module.CandidateResult(
            family="relaxed_efficiency",
            candidate_label=(
                "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna"
            ),
            source_profile_id="relaxed_growth",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=pd.Series(0.004, index=index, dtype=float),
        ),
        module.CandidateResult(
            family="profile_optuna",
            candidate_label="profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna",
            source_profile_id="profile_growth",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=pd.Series(0.003, index=index, dtype=float),
        ),
    ]

    row = module._evaluate_candidate(
        module._teacher_leaf_blend_candidates(candidates, fold)[0], fold
    )

    assert row["family"] == "teacher_leaf_blend"
    assert row["clean_promotion_eligible"] is True
    assert row["nested_hybrid_dependency"] is False
    assert row["uses_locked_oos_for_selection"] is False
    assert row["current_fold_oos_used_for_weighting"] is False
    assert set(row["final_weights"]) == {
        "relaxed_efficiency:growth_mdd20_gross8_69_asset_relaxed_efficiency_repair_optuna",
        "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna",
    }


def _tradfi_source_stream(
    *,
    symbol: str,
    model_id: str,
    cash_session_return: float,
    asset_group: str = "tradfi_equity",
    timeframe: str = "30m",
    dominant_anchor: str = "us_equity_beta_spy",
    dominant_anchor_abs_corr: float = 0.25,
) -> module.broad69.CandidateStream:
    index = pd.date_range("2025-01-01", "2025-03-31 23:30", freq="30min")
    returns = pd.Series(0.0, index=index, dtype=float)
    returns.iloc[module._us_equity_cash_session_mask(index)] = cash_session_return
    row = {
        "model_id": model_id,
        "family": "cross_sectional_momentum_rank",
        "symbol": symbol,
        "asset_group": asset_group,
        "timeframe": timeframe,
        "side": "long",
        "notional_fraction": 0.10,
        "validation_return_per_turnover_proxy_bps": 30.0,
        "dominant_anchor": dominant_anchor,
        "dominant_anchor_abs_corr": dominant_anchor_abs_corr,
    }
    return module.broad69.CandidateStream(
        row=row,
        returns=returns,
        position=pd.Series(1.0, index=index, dtype=float),
    )


def test_tradfi_us_equity_session_switch_is_clean_and_cash_session_masked() -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    streams = [
        _tradfi_source_stream(
            symbol="SPYUSDT",
            asset_group="tradfi_etf_index",
            model_id="spy_cash_session",
            cash_session_return=0.00020,
        ),
        _tradfi_source_stream(
            symbol="NVDAUSDT",
            model_id="nvda_cash_session",
            cash_session_return=0.00018,
            dominant_anchor="tech_growth_qqq",
            dominant_anchor_abs_corr=0.30,
        ),
        _tradfi_source_stream(
            symbol="AAPLUSDT",
            model_id="aapl_cash_session",
            cash_session_return=0.00016,
            dominant_anchor="tech_growth_qqq",
            dominant_anchor_abs_corr=0.35,
        ),
        _tradfi_source_stream(
            symbol="BTCUSDT",
            asset_group="crypto_core",
            model_id="btc_should_be_ignored",
            cash_session_return=0.00200,
            dominant_anchor="crypto_beta_btc",
            dominant_anchor_abs_corr=0.10,
        ),
    ]

    candidates = module._tradfi_us_equity_session_switch_candidates(streams, fold)
    by_label = {candidate.candidate_label: candidate for candidate in candidates}

    assert set(by_label) == {
        "tradfi_us_equity_session_switch:cash_session_top8_mdd15",
        "tradfi_us_equity_session_switch:cash_session_top12_mdd20",
        "tradfi_us_equity_session_switch:cash_session_beta_guard_mdd12",
    }
    candidate = by_label["tradfi_us_equity_session_switch:cash_session_top8_mdd15"]
    row = module._evaluate_candidate(candidate, fold)

    assert row["clean_promotion_eligible"] is True
    assert row["uses_locked_oos_for_selection"] is False
    assert row["post_oos_research_variant"] is False
    assert row["requires_fresh_forward_shadow"] is False
    assert row["ready_for_paper"] is True
    assert row["ready_for_real"] is False
    assert row["real_money_execution"] is False
    assert "BTCUSDT" not in row["selected_symbols"]
    assert set(row["selected_symbols"]) <= {"SPYUSDT", "NVDAUSDT", "AAPLUSDT"}
    assert row["selection_inputs"] == [
        "train",
        "validation",
        "us_equity_market_structure_priors",
    ]
    assert row["tradfi_us_equity_controls"]["overnight_gap_policy"] == (
        "no_non_cash_session_return_exposure_in_research_proxy"
    )
    assert candidate.returns.loc[pd.Timestamp("2025-02-03 03:00:00")] == 0.0
    assert module._leaf_strategy_material_candidate(candidate) is False


def test_tradfi_us_equity_session_switch_cash_guards_when_no_tradfi_signal() -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    streams = [
        _tradfi_source_stream(
            symbol="BTCUSDT",
            asset_group="crypto_core",
            model_id="btc_only",
            cash_session_return=0.001,
            dominant_anchor="crypto_beta_btc",
            dominant_anchor_abs_corr=0.10,
        )
    ]

    candidates = module._tradfi_us_equity_session_switch_candidates(streams, fold)

    assert len(candidates) == 3
    assert all(candidate.returns.sum() == 0.0 for candidate in candidates)
    assert all(
        candidate.row["selected_candidate_label"]
        == "cash_no_eligible_tradfi_us_equity_session_signal"
        for candidate in candidates
    )
    assert all(module._evaluate_candidate(candidate, fold)["clean_promotion_eligible"] is True for candidate in candidates)


def test_asset_timeframe_leverage_family_marks_clean_train_validation_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )

    def make_profile_stream(
        profile_id: str, daily_return: float
    ) -> module.grid_hybrid.ProfileStream:
        return module.grid_hybrid.ProfileStream(
            profile_id=profile_id,
            candidate_tier="paper_testnet_individual_sleeve_first_candidate",
            leverage_map={"BTCUSDT": 3},
            gross_notional_fraction=1.2,
            asset_gross_notional_fraction={"BTCUSDT": 1.2},
            selected_model_ids=(profile_id,),
            returns=pd.Series(daily_return, index=index, dtype=float),
            turnover_by_split=dict.fromkeys(module.grid_hybrid.ilp.SPLIT_ORDER, 1.0),
            trade_events_by_split=dict.fromkeys(module.grid_hybrid.ilp.SPLIT_ORDER, 10),
            liquidation_count_by_split=dict.fromkeys(module.grid_hybrid.ilp.SPLIT_ORDER, 0),
        )

    def fake_tune(*, spec, candidate_streams, windows, n_trials, seed):
        profile_id = str(spec["profile_id"])
        stream = make_profile_stream(profile_id, 0.002 if "balanced" in profile_id else 0.003)
        row = {
            "profile_id": profile_id,
            "profile_kind": spec["profile_kind"],
            "train_return": 0.08,
            "validation_return": 0.05,
            "train_mdd": 0.02,
            "validation_mdd": 0.02,
            "gross_notional_fraction": 1.2,
            "selected_sleeve_count": spec["min_sleeves"],
            "concentration": {"top_symbol_share": 0.10, "top_asset_group_share": 0.20},
            "selection_reasons": [],
            "final_weights": {profile_id: 1.0},
            "asset_gross_notional_fraction": {"BTCUSDT": 1.2},
            "rebalance_policy": spec["rebalance_policy"],
            "leverage_tuning_policy": spec["leverage_tuning_policy"],
        }
        selected = [
            {
                "parent_profile_id": profile_id,
                "symbol": "BTCUSDT",
                "timeframe": "1h",
                "integer_leverage": 3,
                "sleeve_multiplier": 0.5,
                "weighted_notional_fraction": 0.6,
            }
        ]
        return stream, row, selected

    def fake_run_optuna(
        streams, *, version, n_trials, seed, fit_splits, warmup_splits, require_locked_oos_gate
    ):
        weights = {streams[0].profile_id: 1.0}
        return SimpleNamespace(
            row={
                "profile_id": f"asset_tf_fake_{version}",
                "train_return": 0.08,
                "validation_return": 0.05,
                "train_mdd": 0.02,
                "validation_mdd": 0.02,
                "gross_notional_fraction": 1.2,
                "selection_reasons": [],
                "weights": weights,
                "final_weights": weights,
            },
            returns=streams[0].returns,
        )

    def fake_static(streams, *, n_trials, seed):
        weights = {stream.profile_id: 1.0 / float(len(streams)) for stream in streams}
        return {
            "profile_id": "asset_tf_static",
            "train_return": 0.08,
            "validation_return": 0.05,
            "train_mdd": 0.02,
            "validation_mdd": 0.02,
            "gross_notional_fraction": 1.2,
            "selection_reasons": [],
            "final_weights": weights,
        }

    monkeypatch.setattr(module, "tune_individual_robust_profile", fake_tune)
    monkeypatch.setattr(module.optuna_hybrid, "_run_optuna", fake_run_optuna)
    monkeypatch.setattr(
        module.optuna_hybrid, "_choose_selected_optuna_result", lambda items: items[0]
    )
    monkeypatch.setattr(module.profile69, "optimize_static_profile_blend", fake_static)

    candidates, aux = module._run_asset_timeframe_leverage_family(
        fold=fold,
        candidate_streams=[],
        hybrid_trials=2,
        seed=11,
    )

    assert aux["profile_row_count"] == len(module.ASSET_TIMEFRAME_LEVERAGE_PROFILE_SPECS)
    assert {
        "asset_timeframe_leverage:asset_tf_leverage_balanced_mdd12_gross4_core16",
        "asset_timeframe_leverage:hybrid_v3_5",
    }.issubset({candidate.candidate_label for candidate in candidates})
    evaluated = [module._evaluate_candidate(candidate, fold) for candidate in candidates]
    assert all(row["uses_locked_oos_for_selection"] is False for row in evaluated)
    assert all(row["clean_promotion_eligible"] is True for row in evaluated)
    assert all(
        row["rebalance_policy"] == "monthly_refit_signal_level_position_updates"
        for row in evaluated
    )
    assert all(
        row["leverage_tuning_policy"]
        == "train_validation_only_source_integer_leverage_plus_post_allocation_multiplier"
        for row in evaluated
    )


def test_validation_selector_ignores_locked_oos_outcome() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    high_validation_bad_oos = pd.Series(0.010, index=index, dtype=float)
    high_validation_bad_oos.loc["2025-03-01":"2025-03-31"] = -0.020
    low_validation_good_oos = pd.Series(0.002, index=index, dtype=float)
    low_validation_good_oos.loc["2025-03-01":"2025-03-31"] = 0.020
    candidates = [
        module.CandidateResult(
            family="profile_optuna",
            candidate_label="high_validation_bad_oos",
            source_profile_id="high_validation_bad_oos",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=high_validation_bad_oos,
        ),
        module.CandidateResult(
            family="profile_optuna",
            candidate_label="low_validation_good_oos",
            source_profile_id="low_validation_good_oos",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=low_validation_good_oos,
        ),
    ]

    selected = module._validation_selector_candidates(candidates, fold)

    assert selected
    by_label = {candidate.candidate_label: candidate for candidate in selected}
    assert (
        by_label["validation_selector:validation_calmar_mdd12"].row["selected_candidate_label"]
        == "high_validation_bad_oos"
    )
    assert (
        by_label["validation_selector:validation_utility_mdd15"].row["selected_candidate_label"]
        == "high_validation_bad_oos"
    )
    assert all(candidate.row["uses_locked_oos_for_selection"] is False for candidate in selected)
    assert (
        module._period_metrics(
            by_label["validation_selector:validation_calmar_mdd12"].returns,
            fold.locked_oos,
        )["total_return"]
        < 0.0
    )


def test_validation_selector_excludes_post_oos_research_variants() -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    post_oos = _candidate("post_oos_research", daily_return=0.05)
    post_oos.row["post_oos_research_variant"] = True
    post_oos.row["requires_fresh_forward_shadow"] = True
    clean = _candidate("clean_candidate", daily_return=0.002)

    selected = module._validation_selector_candidates([post_oos, clean], fold)

    assert selected
    assert all(
        candidate.row["selected_candidate_label"] == "clean_candidate" for candidate in selected
    )


def test_mdd30_risk_scaled_candidates_use_leaf_sources_only() -> None:
    source_label = "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna"
    nested_label = "dynamic_conviction_switch:t0.85_risk_capped_fallback"
    candidates = [
        _candidate(source_label, family="profile_optuna", daily_return=0.010),
        _candidate(nested_label, family="dynamic_conviction_switch", daily_return=0.020),
    ]
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )

    out = module._mdd30_high_volatility_candidates(candidates, fold)

    labels = {item.candidate_label for item in out}
    assert "mdd30_risk_scaled:profile_growth_x1_50" in labels
    assert all(nested_label not in candidate.row.get("final_weights", {}) for candidate in out)
    scaled = next(
        item for item in out if item.candidate_label == "mdd30_risk_scaled:profile_growth_x1_50"
    )
    assert scaled.row["source_candidate_label"] == source_label
    assert scaled.row["uses_locked_oos_for_selection"] is False
    assert scaled.row["current_fold_oos_used_for_weighting"] is False
    assert scaled.row["post_oos_research_variant"] is True
    assert scaled.row["requires_fresh_forward_shadow"] is True
    assert scaled.row["risk_scale"] == pytest.approx(1.50)


def test_mdd30_high_vol_gate_can_pick_bad_oos_from_validation_only() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    aggressive_label = "profile_optuna:aggressive_mdd30_gross10_69_asset_profile_optuna"
    breakout_bad_oos = pd.Series(0.010, index=index, dtype=float)
    breakout_bad_oos.loc["2025-03-01":"2025-03-31"] = -0.020
    defensive_good_oos = pd.Series(0.001, index=index, dtype=float)
    defensive_good_oos.loc["2025-03-01":"2025-03-31"] = 0.002
    candidates = [
        module.CandidateResult(
            family="profile_optuna",
            candidate_label=aggressive_label,
            source_profile_id="breakout_bad_oos",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=breakout_bad_oos,
        ),
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label="strict_efficiency:balanced_mdd12_gross5_69_asset_efficiency_repair_optuna",
            source_profile_id="defensive_good_oos",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=defensive_good_oos,
        ),
        module.CandidateResult(
            family="strict_efficiency",
            candidate_label="strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna",
            source_profile_id="defensive_growth",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=defensive_good_oos,
        ),
    ]

    gated = module._mdd30_high_volatility_candidates(candidates, fold)
    selected = next(
        item
        for item in gated
        if item.candidate_label == "mdd30_high_vol_gate:validation_breakout_or_defensive_scaled"
    )

    assert selected.row["selected_candidate_label"] == aggressive_label
    assert selected.row["uses_locked_oos_for_selection"] is False
    assert selected.row["high_vol_breakout"] is True
    # OOS is deliberately worse, proving the gate did not peek at locked OOS.
    assert module._period_metrics(selected.returns, fold.locked_oos)["total_return"] < 0.0


def test_bridge_protocol_manifest_hash_matches_frozen_file() -> None:
    manifest_path = module.DEFAULT_BRIDGE_PROTOCOL_MANIFEST
    loaded = module._load_bridge_protocol_manifest(manifest_path)
    expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    assert loaded["_sha256"] == expected_hash
    assert loaded["_path"] == str(manifest_path.resolve())
    assert loaded["deployable_expert_roster"]
    assert loaded["allowed_pre_oos_features"]
    assert loaded["fixed_grids"]
    assert loaded["objective_utility"]
    assert loaded["fallback_rules"]
    assert loaded["negative_controls"]


def test_protocol_freeze_report_forbids_post_oos_expansion() -> None:
    manifest = module._load_bridge_protocol_manifest(module.DEFAULT_BRIDGE_PROTOCOL_MANIFEST)
    freeze_report = module._protocol_freeze_report(manifest)

    assert freeze_report["bridge_protocol_manifest_present"] is True
    assert freeze_report["bridge_protocol_manifest_sha256"] == manifest["_sha256"]
    assert freeze_report["frozen_before_first_oos_evaluation"] is True
    assert freeze_report["post_oos_expansion_allowed"] is False
    assert freeze_report["oos_used_for_protocol_expansion"] is False


def test_hybrid_bridge_refuses_nested_dynamic_and_hybrid_inputs() -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    manifest = module._load_bridge_protocol_manifest(module.DEFAULT_BRIDGE_PROTOCOL_MANIFEST)
    bridge_candidates = module._hybrid_assimilated_dynamic_candidates(
        [
            _candidate(
                "dynamic_conviction_switch:t0.90_risk_capped_fallback",
                family="dynamic_conviction_switch",
                daily_return=0.010,
            ),
            _candidate("cross_candidate_hybrid:hybrid_v3_5", family="cross_candidate_hybrid"),
            _candidate(
                "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna", daily_return=0.008
            ),
        ],
        fold,
        prior_completed_utilities={"2025-02": [0.02]},
        bridge_manifest=manifest,
    )

    assert bridge_candidates == []
    assert (
        module._online_weight_audit(
            [{"month": "2025-03", "utility_months_used": ["2025-01", "2025-02"]}]
        )["fully_lagged_online_weights"]
        is True
    )
    assert (
        module._online_weight_audit([{"month": "2025-03", "utility_months_used": ["2025-03"]}])[
            "fully_lagged_online_weights"
        ]
        is False
    )


def test_hybrid_bridge_eligible_pool_excludes_nested_and_post_oos_research_variants() -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    post_oos = _candidate(
        "mdd30_risk_scaled:profile_growth_x1_50", family="mdd30_risk_scaled", daily_return=0.05
    )
    post_oos.row["post_oos_research_variant"] = True
    post_oos.row["requires_fresh_forward_shadow"] = True
    clean_leaf = _candidate(
        "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna", daily_return=0.008
    )

    eligible = module._bridge_eligible_candidates(
        [
            _candidate(
                "dynamic_conviction_switch:t0.90_risk_capped_fallback",
                family="dynamic_conviction_switch",
                daily_return=0.010,
            ),
            _candidate("cross_candidate_hybrid:hybrid_v3_5", family="cross_candidate_hybrid"),
            clean_leaf,
            post_oos,
        ],
        fold,
    )

    assert [candidate.candidate_label for candidate, _ in eligible] == [clean_leaf.candidate_label]


def test_dynamic_switch_rows_do_not_self_feed_same_month_oos_or_oracle_inputs() -> None:
    clean_rows = [
        {
            "fold_id": "2025-03",
            "family": "dynamic_conviction_switch",
            "candidate_label": "dynamic_conviction_switch:t0.90_risk_capped_fallback",
            "selection_inputs": ["train", "validation"],
        }
    ]
    poisoned_rows = [
        {
            **clean_rows[0],
            "feature_inputs": ["same_fold_selected_label", "locked_oos_return", "oracle_rank"],
        }
    ]

    assert module._dynamic_self_feed_audit(clean_rows)["no_same_month_dynamic_self_feeding"] is True
    poisoned_audit = module._dynamic_self_feed_audit(poisoned_rows)
    assert poisoned_audit["no_same_month_dynamic_self_feeding"] is False
    assert poisoned_audit["violations"][0]["fold_id"] == "2025-03"


def test_metric_reconciliation_recomputes_report_aggregates_from_fold_rows() -> None:
    payload = {
        "fold_candidate_rows": [
            {
                "candidate_label": "candidate_a",
                "family": "f",
                "clean_promotion_eligible": True,
                "train": {"total_return": 0.02, "mdd": 0.01},
                "validation": {"total_return": 0.03, "mdd": 0.01},
                "locked_oos": {"total_return": 0.10, "mdd": 0.02},
            },
            {
                "candidate_label": "candidate_a",
                "family": "f",
                "clean_promotion_eligible": True,
                "train": {"total_return": 0.02, "mdd": 0.01},
                "validation": {"total_return": 0.03, "mdd": 0.01},
                "locked_oos": {"total_return": -0.05, "mdd": 0.03},
            },
        ],
    }
    payload["aggregate_rankings"] = module._aggregate_rows(payload["fold_candidate_rows"])

    report = module._metric_reconciliation_report(payload)

    assert report == {"metrics_reconciled": True, "mismatches": [], "candidate_count": 1}


def test_promotability_hard_stop_blocks_non_threshold_result() -> None:
    blocked = module._promotability_decision(
        {"compounded_oos_return": 0.20, "max_oos_mdd": 0.16, "min_oos_return": -0.05}
    )
    allowed = module._promotability_decision(
        {"compounded_oos_return": 0.54, "max_oos_mdd": 0.18, "min_oos_return": -0.02}
    )

    assert blocked["promotable"] is False
    assert blocked["promotion_hard_stop_pass"] is False
    assert (
        blocked["if_false_recommendation"]
        == "paper_shadow_only_further_uplift_would_be_oos_mining_risk"
    )
    assert allowed["promotable"] is True
    assert allowed["promotion_hard_stop_reasons"]


def test_promotability_blocks_post_oos_research_variant_even_if_metrics_are_high() -> None:
    blocked = module._promotability_decision(
        {
            "clean_promotion_eligible": False,
            "compounded_oos_return": 0.60,
            "max_oos_mdd": 0.10,
            "min_oos_return": -0.01,
        }
    )

    assert blocked["promotable"] is False
    assert blocked["promotion_hard_stop_pass"] is False
    assert blocked["promotion_hard_stop_reasons"] == ["blocked_non_clean_research_variant"]
    assert blocked["if_false_recommendation"] == "fresh_forward_shadow_required_before_promotion"


def test_recompute_payload_marks_downstream_post_oos_dependencies_non_clean() -> None:
    base_row = {
        "fold_id": "2025-03",
        "family": "mdd30_risk_scaled",
        "candidate_label": "mdd30_risk_scaled:dyn085_x1_50",
        "post_oos_research_variant": True,
        "requires_fresh_forward_shadow": True,
        "clean_promotion_eligible": False,
        "train": {"total_return": 0.10, "mdd": 0.05},
        "validation": {"total_return": 0.10, "mdd": 0.05},
        "locked_oos": {"total_return": 0.10, "mdd": 0.05},
    }
    contaminated_bridge = {
        "fold_id": "2025-03",
        "family": "hybrid_oracle_bridge",
        "candidate_label": "hybrid_oracle_bridge:uses_mdd30",
        "post_oos_research_variant": False,
        "requires_fresh_forward_shadow": False,
        "clean_promotion_eligible": True,
        "final_weights": {"mdd30_risk_scaled:dyn085_x1_50": 1.0},
        "bridge_inputs": ["mdd30_risk_scaled:dyn085_x1_50"],
        "train": {"total_return": 0.10, "mdd": 0.05},
        "validation": {"total_return": 0.10, "mdd": 0.05},
        "locked_oos": {"total_return": 0.20, "mdd": 0.05},
    }

    recomputed = module._recompute_payload_from_existing(
        {"fold_candidate_rows": [base_row, contaminated_bridge]}
    )
    rows = {row["candidate_label"]: row for row in recomputed["fold_candidate_rows"]}
    bridge = rows["hybrid_oracle_bridge:uses_mdd30"]

    assert bridge["post_oos_research_variant"] is True
    assert bridge["requires_fresh_forward_shadow"] is True
    assert bridge["clean_promotion_eligible"] is False
    assert recomputed["metric_reconciliation"]["metrics_reconciled"] is True


def test_recompute_payload_separates_raw_clean_and_demoted_rankings() -> None:
    clean_row = {
        "fold_id": "2025-03",
        "family": "relaxed_efficiency",
        "candidate_label": "relaxed_efficiency:growth_leaf",
        "clean_promotion_eligible": True,
        "train": {"total_return": 0.10, "mdd": 0.05},
        "validation": {"total_return": 0.08, "mdd": 0.04},
        "locked_oos": {"total_return": 0.05, "mdd": 0.03},
    }
    raw_winner_nested = {
        "fold_id": "2025-03",
        "family": "meta_portfolio",
        "candidate_label": "meta_portfolio:raw_winner",
        "clean_promotion_eligible": True,
        "final_weights": {"cross_candidate_hybrid:hybrid_v3_5": 1.0},
        "train": {"total_return": 0.20, "mdd": 0.05},
        "validation": {"total_return": 0.18, "mdd": 0.04},
        "locked_oos": {"total_return": 0.30, "mdd": 0.03},
    }

    recomputed = module._recompute_payload_from_existing(
        {"fold_candidate_rows": [clean_row, raw_winner_nested]}
    )

    assert recomputed["aggregate_rankings"][0]["candidate_label"] == "meta_portfolio:raw_winner"
    assert recomputed["aggregate_rankings"][0]["clean_promotion_eligible"] is False
    assert recomputed["clean_promotion_rankings"][0]["candidate_label"] == (
        "relaxed_efficiency:growth_leaf"
    )
    assert recomputed["demoted_nested_or_historical_rankings"][0]["candidate_label"] == (
        "meta_portfolio:raw_winner"
    )
    assert recomputed["demoted_nested_or_historical_rankings"][0]["non_clean_reasons"] == [
        "nested_hybrid_dependency"
    ]


def test_row_level_leaf_selectors_are_oos_clean_non_nested_shadow_rows() -> None:
    leaf = {
        "fold_id": "2025-03",
        "family": "profile_optuna",
        "candidate_label": "profile_optuna:growth_leaf",
        "source_profile_id": "profile_optuna:growth_leaf",
        "profile_kind": "leaf_momentum_profile",
        "selection_reasons": [],
        "clean_promotion_eligible": True,
        "uses_locked_oos_for_selection": False,
        "train": {"total_return": 0.40, "mdd": 0.10, "calmar": 4.0},
        "validation": {"total_return": 0.12, "mdd": 0.05, "calmar": 2.4},
        "locked_oos": {"total_return": 0.07, "mdd": 0.03},
    }
    nested = {
        **leaf,
        "family": "cross_candidate_hybrid",
        "candidate_label": "cross_candidate_hybrid:hybrid_v3_5",
        "source_profile_id": "cross_candidate_hybrid:hybrid_v3_5",
        "profile_kind": "hybrid",
        "validation": {"total_return": 0.50, "mdd": 0.02, "calmar": 25.0},
        "locked_oos": {"total_return": 0.50, "mdd": 0.02},
    }

    augmented = module._augment_payload_with_row_level_leaf_selectors(
        {"fold_candidate_rows": [leaf, nested]}
    )
    selector_rows = [
        row
        for row in augmented["fold_candidate_rows"]
        if row["family"] == "row_level_leaf_selector"
    ]

    assert len(selector_rows) == len(module.ROW_LEVEL_LEAF_SELECTOR_SPECS)
    assert {row["selected_candidate_label"] for row in selector_rows} == {
        "profile_optuna:growth_leaf"
    }
    assert all(row["uses_locked_oos_for_selection"] is False for row in selector_rows)
    assert all(row["mechanically_oos_clean"] is True for row in selector_rows)
    assert all(row["nested_hybrid_dependency"] is False for row in selector_rows)
    assert all(row["clean_promotion_eligible"] is False for row in selector_rows)
    assert all(row["locked_oos"]["total_return"] == pytest.approx(0.07) for row in selector_rows)
    assert augmented["metric_reconciliation"]["metrics_reconciled"] is True
    assert any(
        row["candidate_label"] == "row_level_leaf_selector:validation_calmar_mdd20"
        and row["non_clean_reasons"]
        == [
            "post_oos_research_variant",
            "requires_fresh_forward_shadow",
        ]
        for row in augmented["demoted_nested_or_historical_rankings"]
    )


def test_non_leaf_reference_detector_covers_portfolio_families_and_selected_tokens() -> None:
    forbidden = [
        "cross_candidate_hybrid:hybrid_v3_5",
        "meta_portfolio:validation_balanced",
        "dynamic_conviction_switch:t0.90_risk_capped_fallback",
        "tradfi_us_equity_session_switch:cash_session_top8_mdd15",
        "validation_selector:top_clean",
        "mdd30_high_vol_gate:breakout_x1_50",
        "profile_optuna:selected_optuna",
        "individual_robust:selected_train_validation_legal",
        "relaxed_efficiency:hybrid_v3_5",
    ]
    allowed_leaf = "profile_optuna:growth_mdd20_gross8_69_asset_profile_optuna"

    assert all(module._candidate_label_is_non_leaf_reference(label) for label in forbidden)
    assert module._candidate_label_is_non_leaf_reference(allowed_leaf) is False


def test_recompute_payload_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "walkforward.json"
    source.write_text(json.dumps({"fold_candidate_rows": []}), "utf-8")
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"

    recomputed = module._recompute_payload_from_existing(
        json.loads(source.read_text("utf-8")),
        source_path=source,
        output_json=output_json,
        output_md=output_md,
    )

    provenance = recomputed["recompute_provenance"]
    assert provenance["source_json_path"] == str(source.resolve())
    assert provenance["source_json_sha256"] == module._file_sha256(source)
    assert provenance["recomputed_from_existing_rows"] is True
    assert provenance["fresh_optuna_rerun"] is False
    assert provenance["output_paths"] == {
        "json": str(output_json.resolve()),
        "markdown": str(output_md.resolve()),
    }


def test_markdown_renders_clean_and_demoted_sections() -> None:
    payload = module._recompute_payload_from_existing(
        {
            "generated_at_utc": "2026-06-04T00:00:00Z",
            "data_coverage": {"global_latest_utc": "2026-06-04T00:00:00"},
            "timeframes": ["30m"],
            "cost_model": {"slippage_bps": 10.0},
            "trial_policy": {"asset_trials": 1, "profile_trials": 1, "hybrid_trials": 1},
            "folds": [
                {
                    "fold_id": "2025-03",
                    "refit_at": "2025-03-01T00:00:00",
                    "train": {"start": "2025-01-01T00:00:00", "end": "2025-01-31T00:00:00"},
                    "validation": {
                        "start": "2025-02-01T00:00:00",
                        "end": "2025-02-28T00:00:00",
                    },
                    "locked_oos": {
                        "start": "2025-03-01T00:00:00",
                        "end": "2025-03-31T00:00:00",
                    },
                }
            ],
            "fold_candidate_rows": [
                {
                    "fold_id": "2025-03",
                    "family": "relaxed_efficiency",
                    "candidate_label": "relaxed_efficiency:growth_leaf",
                    "source_profile_id": "growth_leaf",
                    "clean_promotion_eligible": True,
                    "train": {"total_return": 0.10, "mdd": 0.05},
                    "validation": {"total_return": 0.08, "mdd": 0.04},
                    "locked_oos": {"total_return": 0.05, "mdd": 0.03},
                },
                {
                    "fold_id": "2025-03",
                    "family": "fixed_relaxed_dynamic_blend",
                    "candidate_label": "fixed_relaxed_dynamic_blend:relaxed60_dynamic40",
                    "source_profile_id": "nested",
                    "final_weights": {"dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit": 1.0},
                    "train": {"total_return": 0.20, "mdd": 0.05},
                    "validation": {"total_return": 0.18, "mdd": 0.04},
                    "locked_oos": {"total_return": 0.30, "mdd": 0.03},
                },
            ],
        }
    )

    markdown = module._render_markdown(payload)

    assert "## Raw aggregate ranking (diagnostic only)" in markdown
    assert "## Clean-promotion ranking (current recommendation set)" in markdown
    assert "## Demoted nested/historical ranking" in markdown
    assert "Best clean candidate monthly OOS detail" in markdown
