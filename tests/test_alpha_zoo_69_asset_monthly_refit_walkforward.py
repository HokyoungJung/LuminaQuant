from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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


def test_dynamic_conviction_switch_uses_train_validation_only() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    aggressive_returns = pd.Series(0.04, index=index, dtype=float)
    aggressive_returns.loc["2025-03-01":"2025-03-31"] = -0.03
    fallback_returns = pd.Series(0.001, index=index, dtype=float)
    candidates = [
        module.CandidateResult(
            family="cross_candidate_hybrid",
            candidate_label="cross_candidate_hybrid:hybrid_v3_5",
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
    assert switched[0].row["selected_candidate_label"] == "cross_candidate_hybrid:hybrid_v3_5"
    # The selector still picks the high train/validation candidate even though
    # its locked OOS is deliberately worse than fallback in this fixture.
    assert module._period_metrics(switched[0].returns, fold.locked_oos)["total_return"] < 0.0


def test_dynamic_aware_hybrid_absorbs_dynamic_as_clean_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    calls: list[dict[str, object]] = []

    def fake_run_optuna(
        streams, *, version, n_trials, seed, fit_splits, warmup_splits, require_locked_oos_gate
    ):
        labels = [stream.profile_id for stream in streams]
        calls.append(
            {
                "labels": labels,
                "version": version,
                "fit_splits": tuple(fit_splits),
                "require_locked_oos_gate": require_locked_oos_gate,
            }
        )
        returns = sum(stream.returns for stream in streams) / float(len(streams))
        weights = {label: 1.0 / float(len(labels)) for label in labels}
        return SimpleNamespace(
            row={
                "profile_id": f"fake_{version}_{'_'.join(fit_splits)}",
                "selection_reasons": [],
                "weights": weights,
                "final_weights": weights,
            },
            returns=returns,
        )

    monkeypatch.setattr(module.optuna_hybrid, "_run_optuna", fake_run_optuna)
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
        _candidate(
            "strict_efficiency:growth_mdd20_gross8_69_asset_efficiency_repair_optuna",
            family="strict_efficiency",
            daily_return=0.001,
        ),
    ]

    out = module._dynamic_aware_hybrid_candidates(candidates, fold, hybrid_trials=8, seed=7)

    assert len(out) == 4
    assert {item.candidate_label for item in out} == {
        "dynamic_aware_hybrid:hybrid_v3_5",
        "dynamic_aware_hybrid:hybrid_v3_6",
        "dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit",
        "dynamic_aware_hybrid:hybrid_v3_6_train_validation_fit",
    }
    assert all(candidate.row["uses_locked_oos_for_selection"] is False for candidate in out)
    assert all(candidate.row["same_month_self_feeding"] is False for candidate in out)
    assert all(candidate.row["current_fold_oos_used_for_weighting"] is False for candidate in out)
    assert all(
        "dynamic_conviction_switch:t0.90_risk_capped_fallback"
        in candidate.row["dynamic_input_labels"]
        for candidate in out
    )
    assert all(call["require_locked_oos_gate"] is False for call in calls)
    assert all(
        "dynamic_conviction_switch:t0.90_risk_capped_fallback" in call["labels"] for call in calls
    )


def test_fixed_risk_enhanced_blend_is_research_only_without_oos_selection() -> None:
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

    out = module._fixed_risk_enhanced_blend_candidates(candidates)

    assert {
        "risk_enhanced_blend:dyn085_70_aware_v36tv_30",
        "risk_enhanced_blend:dyn085_60_aware_v36tv_40",
        "risk_enhanced_blend:dyn085_50_aware_v36tv_50",
    }.issubset({candidate.candidate_label for candidate in out})
    candidate = next(
        item
        for item in out
        if item.candidate_label == "risk_enhanced_blend:dyn085_70_aware_v36tv_30"
    )
    assert candidate.row["uses_locked_oos_for_selection"] is False
    assert candidate.row["current_fold_oos_used_for_weighting"] is False
    assert candidate.row["post_oos_research_variant"] is True
    assert candidate.row["requires_fresh_forward_shadow"] is True
    assert candidate.row["ready_for_real"] is False
    assert (
        module._period_metrics(
            candidate.returns,
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        )["total_return"]
        > module._period_metrics(
            candidates[1].returns,
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        )["total_return"]
    )


def test_fixed_relaxed_dynamic_blend_uses_exact_streams_and_is_shadow_only() -> None:
    index = pd.date_range("2025-01-01", "2025-01-05", freq="1D")
    relaxed_returns = pd.Series([0.10, -0.04, 0.03, -0.02, 0.01], index=index, dtype=float)
    dynamic_returns = pd.Series([0.02, -0.01, 0.01, 0.00, 0.01], index=index, dtype=float)
    candidates = [
        module.CandidateResult(
            family="relaxed_efficiency",
            candidate_label="relaxed_efficiency:hybrid_v3_5",
            source_profile_id="relaxed",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=relaxed_returns,
        ),
        module.CandidateResult(
            family="dynamic_aware_hybrid",
            candidate_label="dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit",
            source_profile_id="dynamic",
            row={"selection_reasons": [], "uses_locked_oos_for_selection": False},
            returns=dynamic_returns,
        ),
    ]

    out = module._fixed_relaxed_dynamic_blend_candidates(candidates)

    assert {
        "fixed_relaxed_dynamic_blend:relaxed40_dynamic60",
        "fixed_relaxed_dynamic_blend:relaxed50_dynamic50",
        "fixed_relaxed_dynamic_blend:relaxed60_dynamic40",
    }.issubset({candidate.candidate_label for candidate in out})
    blend = next(
        candidate
        for candidate in out
        if candidate.candidate_label == "fixed_relaxed_dynamic_blend:relaxed50_dynamic50"
    )
    pd.testing.assert_series_equal(
        blend.returns, relaxed_returns * 0.5 + dynamic_returns * 0.5, check_freq=False
    )
    assert blend.row["uses_locked_oos_for_selection"] is False
    assert blend.row["current_fold_oos_used_for_weighting"] is False
    assert blend.row["post_oos_research_variant"] is True
    assert blend.row["requires_fresh_forward_shadow"] is True
    assert blend.row["ready_for_real"] is False


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

    def make_profile_stream(profile_id: str, daily_return: float) -> module.grid_hybrid.ProfileStream:
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
    monkeypatch.setattr(module.optuna_hybrid, "_choose_selected_optuna_result", lambda items: items[0])
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


def test_mdd30_risk_scaled_candidates_use_no_locked_oos_selection() -> None:
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
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )

    out = module._mdd30_high_volatility_candidates(candidates, fold)

    scaled = next(item for item in out if item.candidate_label == "mdd30_risk_scaled:dyn085_x1_50")
    assert scaled.row["uses_locked_oos_for_selection"] is False
    assert scaled.row["current_fold_oos_used_for_weighting"] is False
    assert scaled.row["post_oos_research_variant"] is True
    assert scaled.row["requires_fresh_forward_shadow"] is True
    assert scaled.row["risk_scale"] == pytest.approx(1.50)
    assert (
        module._period_metrics(scaled.returns, fold.validation)["total_return"]
        > module._period_metrics(
            candidates[0].returns,
            fold.validation,
        )["total_return"]
    )


def test_mdd30_high_vol_gate_can_pick_bad_oos_from_validation_only() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31", freq="1D")
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    breakout_bad_oos = pd.Series(0.010, index=index, dtype=float)
    breakout_bad_oos.loc["2025-03-01":"2025-03-31"] = -0.020
    defensive_good_oos = pd.Series(0.001, index=index, dtype=float)
    defensive_good_oos.loc["2025-03-01":"2025-03-31"] = 0.002
    candidates = [
        module.CandidateResult(
            family="profile_optuna",
            candidate_label="profile_optuna:selected_optuna",
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

    assert selected.row["selected_candidate_label"] == "profile_optuna:selected_optuna"
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


def test_hybrid_bridge_hedge_weights_use_only_prior_completed_month_utility() -> None:
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
            _candidate("profile_optuna:selected_optuna", daily_return=0.008),
        ],
        fold,
        prior_completed_utilities={"2025-02": [0.02]},
        bridge_manifest=manifest,
    )

    hedge = next(
        item
        for item in bridge_candidates
        if item.candidate_label == "hybrid_oracle_bridge:hybrid_assimilated_dynamic_v1_hedge"
    )
    assert hedge.row["bridge_assimilation_mode"] == "fully_lagged_hedge_validation_blend"
    assert hedge.row["online_update_cutoff_fold"] == "2025-02"
    assert hedge.row["current_fold_oos_used_for_weighting"] is False
    assert hedge.row["uses_locked_oos_for_selection"] is False
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


def test_hybrid_bridge_excludes_post_oos_research_variants_from_clean_pool() -> None:
    fold = module.MonthlyFold(
        fold_id="2025-03",
        refit_at=pd.Timestamp("2025-03-01"),
        train=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31")),
        validation=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-28")),
        locked_oos=(pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31")),
    )
    manifest = module._load_bridge_protocol_manifest(module.DEFAULT_BRIDGE_PROTOCOL_MANIFEST)
    post_oos = _candidate(
        "mdd30_risk_scaled:dyn085_x1_50", family="mdd30_risk_scaled", daily_return=0.05
    )
    post_oos.row["post_oos_research_variant"] = True
    post_oos.row["requires_fresh_forward_shadow"] = True
    bridge_candidates = module._hybrid_assimilated_dynamic_candidates(
        [
            _candidate(
                "dynamic_conviction_switch:t0.90_risk_capped_fallback",
                family="dynamic_conviction_switch",
                daily_return=0.010,
            ),
            _candidate("cross_candidate_hybrid:hybrid_v3_5", family="cross_candidate_hybrid"),
            _candidate("profile_optuna:selected_optuna", daily_return=0.008),
            post_oos,
        ],
        fold,
        prior_completed_utilities={},
        bridge_manifest=manifest,
    )

    assert bridge_candidates
    assert all(
        "mdd30_risk_scaled:dyn085_x1_50" not in candidate.row["bridge_inputs"]
        for candidate in bridge_candidates
    )
    assert all(
        "mdd30_risk_scaled:dyn085_x1_50" not in candidate.row["final_weights"]
        for candidate in bridge_candidates
    )


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
