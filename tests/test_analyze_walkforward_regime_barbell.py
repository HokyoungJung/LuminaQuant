from __future__ import annotations

import math

from scripts.research.analyze_walkforward_regime_barbell import (
    DEFAULT_CRISIS_FALLBACK_LABEL,
    RegimeContext,
    SelectorSpec,
    _aggregate_rows,
    _methodology_validation_audit,
    build_selector_rows,
    select_bull_sleeve,
)


def _block(ret: float, mdd: float = 0.05) -> dict[str, float | str]:
    return {
        "start": "2026-01-01T00:00:00",
        "end": "2026-01-31T23:30:00",
        "total_return": ret,
        "mdd": mdd,
    }


def _row(
    label: str,
    fold: str,
    *,
    train: float,
    val: float,
    oos: float,
    family: str = "individual_robust",
) -> dict:
    return {
        "candidate_label": label,
        "fold_id": fold,
        "family": family,
        "clean_promotion_eligible": True,
        "uses_locked_oos_for_selection": False,
        "current_fold_oos_used_for_weighting": False,
        "same_month_self_feeding": False,
        "train": _block(train, 0.08),
        "validation": _block(val, 0.06),
        "locked_oos": _block(oos, 0.07),
    }


def _regime(fold: str, decision: str) -> RegimeContext:
    return RegimeContext(
        fold_id=fold,
        validation_btc_return=0.04 if decision == "BULL" else -0.04,
        last_validation_month_btc_return=0.03 if decision == "BULL" else -0.03,
        validation_crypto_breadth=0.7 if decision == "BULL" else 0.3,
        last_validation_month_crypto_breadth=0.7 if decision == "BULL" else 0.3,
        validation_btc_ma_gap=0.01,
        oos_btc_return=None,
        decision=decision,
        reason="unit_test_regime",
    )


def test_bull_selection_uses_train_validation_not_oos() -> None:
    low_val_high_oos = _row("bull_oos_mined", "2026-01", train=0.05, val=0.001, oos=0.50)
    high_val_low_oos = _row("bull_tv_selected", "2026-01", train=0.08, val=0.06, oos=-0.01)

    selected = select_bull_sleeve(
        [low_val_high_oos, high_val_low_oos],
        clean_only=True,
        min_validation_return=0.0,
        max_validation_mdd=0.25,
    )

    assert selected is not None
    assert selected["candidate_label"] == "bull_tv_selected"


def test_selector_rows_freeze_decision_before_oos() -> None:
    folds = ["2026-01", "2026-02"]
    crisis_rows = [
        _row(
            DEFAULT_CRISIS_FALLBACK_LABEL,
            "2026-01",
            train=0.10,
            val=0.02,
            oos=-0.02,
            family="lagged_shadow_leaf_router",
        ),
        _row(
            DEFAULT_CRISIS_FALLBACK_LABEL,
            "2026-02",
            train=0.10,
            val=0.02,
            oos=0.10,
            family="lagged_shadow_leaf_router",
        ),
    ]
    bull_rows = [
        _row("bull", "2026-01", train=0.12, val=0.07, oos=0.08),
        _row("bull", "2026-02", train=0.12, val=0.07, oos=-0.04),
    ]
    rows_by_fold = {
        fold: [row for row in [*crisis_rows, *bull_rows] if row["fold_id"] == fold]
        for fold in folds
    }
    spec = SelectorSpec(
        label="selector:test",
        crisis_label=DEFAULT_CRISIS_FALLBACK_LABEL,
        bull_clean_only=True,
        bull_weight=0.60,
        mixed_bull_weight=0.40,
        bear_bull_weight=0.10,
        min_bull_validation_return=0.0,
        max_bull_validation_mdd=0.25,
    )

    rows = build_selector_rows(
        fold_order=folds,
        rows_by_fold=rows_by_fold,
        regimes={"2026-01": _regime("2026-01", "BULL"), "2026-02": _regime("2026-02", "BEAR")},
        spec=spec,
    )

    assert rows[0]["barbell_weights"] == {"bull": 0.6, "crisis": 0.4, "cash": 0.0}
    assert rows[1]["barbell_weights"] == {"bull": 0.1, "crisis": 0.9, "cash": 0.0}
    assert rows[0]["uses_locked_oos_for_selection"] is False
    assert rows[0]["current_fold_oos_used_for_weighting"] is False
    assert rows[0]["selection_inputs"] == ["train", "validation", "lagged_market_regime_features"]
    assert math.isclose(rows[0]["locked_oos"]["total_return"], 0.4 * -0.02 + 0.6 * 0.08)
    assert math.isclose(rows[1]["locked_oos"]["total_return"], 0.9 * 0.10 + 0.1 * -0.04)


def test_aggregate_rows_reports_positive_folds_and_comp() -> None:
    rows = [
        _row("selector", "2026-01", train=0.0, val=0.0, oos=0.10),
        _row("selector", "2026-02", train=0.0, val=0.0, oos=-0.05),
        _row("selector", "2026-03", train=0.0, val=0.0, oos=0.02),
    ]

    aggregate = _aggregate_rows("selector", rows)

    assert aggregate["positive_oos_folds"] == 2
    assert math.isclose(aggregate["compounded_oos_return"], (1.10 * 0.95 * 1.02) - 1.0)
    assert aggregate["min_oos_return"] == -0.05
    assert aggregate["ready_for_real"] is False


def test_methodology_audit_separates_decision_wf_from_method_wf() -> None:
    audit = _methodology_validation_audit(
        [
            SelectorSpec(
                label="selector:post_oos_recency",
                crisis_label=DEFAULT_CRISIS_FALLBACK_LABEL,
                bull_clean_only=True,
                bull_weight=0.55,
                mixed_bull_weight=0.30,
                bear_bull_weight=0.0,
                min_bull_validation_return=0.0,
                max_bull_validation_mdd=0.25,
                post_oos_research_variant=True,
                recency_halflife_folds=2.0,
            )
        ]
    )

    assert audit["per_fold_selector_decisions_walk_forward_valid"] is True
    assert audit["current_fold_oos_used_by_selector_decisions"] is False
    assert audit["prior_completed_oos_used_by_recency_weighting"] is True
    assert audit["selector_family_and_hyperparameters_chosen_after_source_oos_review"] is True
    assert audit["nested_walk_forward_methodology_validated"] is False
