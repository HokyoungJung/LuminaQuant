from __future__ import annotations

from pathlib import Path

import pytest

from scripts.research import run_same_stem_post_oos_update_walkforward as module


def _row(
    fold_id: str,
    label: str,
    *,
    validation_return: float,
    validation_mdd: float,
    oos_return: float,
) -> dict[str, object]:
    return {
        "fold_id": fold_id,
        "candidate_label": label,
        "family": label.split(":", 1)[0],
        "source_profile_id": label,
        "profile_kind": "fixture_variant",
        "train": {"total_return": 0.20, "mdd": 0.10},
        "validation": {
            "total_return": validation_return,
            "mdd": validation_mdd,
            "calmar": validation_return / validation_mdd,
        },
        "locked_oos": {"total_return": oos_return, "mdd": 0.03},
        "uses_locked_oos_for_selection": False,
        "current_fold_oos_used_for_weighting": False,
        "same_month_self_feeding": False,
        "post_oos_research_variant": False,
        "requires_fresh_forward_shadow": False,
        "ready_for_paper": True,
        "ready_for_real": False,
        "real_money_execution": False,
    }


def test_normalized_strategy_stem_collapses_hybrid_variants() -> None:
    assert module.normalized_strategy_stem("relaxed_efficiency:hybrid_v3_5") == "hybrid_v3_5"
    assert (
        module.normalized_strategy_stem("cross_candidate_hybrid:hybrid_v3_6_train_validation_fit")
        == "hybrid_v3_6"
    )
    assert (
        module.normalized_strategy_stem("fixed_relaxed_dynamic_blend:relaxed70_dynamic30")
        == "fixed_relaxed_dynamic_blend"
    )


def test_replay_same_stem_uses_prior_oos_not_current_fold_winner() -> None:
    relaxed = "relaxed_efficiency:hybrid_v3_5"
    robust = "individual_robust:hybrid_v3_5"
    rows = [
        _row("2025-01", relaxed, validation_return=0.20, validation_mdd=0.10, oos_return=-0.20),
        _row("2025-01", robust, validation_return=0.10, validation_mdd=0.10, oos_return=0.10),
        # Current OOS strongly favors relaxed, but prior completed OOS favors robust.
        _row("2025-02", relaxed, validation_return=0.20, validation_mdd=0.10, oos_return=0.50),
        _row("2025-02", robust, validation_return=0.10, validation_mdd=0.10, oos_return=-0.05),
    ]

    replayed = module.replay_same_stem_post_oos_updates(rows, target_stems=("hybrid_v3_5",))

    assert [row["fold_id"] for row in replayed] == ["2025-01", "2025-02"]
    assert replayed[0]["selection_score_mode"] == "bootstrap_validation_calmar"
    assert replayed[0]["selected_candidate_label"] == relaxed
    assert replayed[1]["selection_score_mode"] == "prior_completed_post_oos_calmar"
    assert replayed[1]["selected_candidate_label"] == robust
    assert replayed[1]["locked_oos"]["total_return"] == pytest.approx(-0.05)
    assert replayed[1]["uses_locked_oos_for_selection"] is False
    assert replayed[1]["current_fold_oos_used_for_weighting"] is False
    assert replayed[1]["same_month_self_feeding"] is False
    assert replayed[1]["post_oos_research_variant"] is True
    assert replayed[1]["requires_fresh_forward_shadow"] is True
    assert replayed[1]["real_money_execution"] is False


def test_build_payload_blocks_promotion_and_round_trips_outputs(tmp_path: Path) -> None:
    rows = [
        _row(
            "2025-01",
            "relaxed_efficiency:hybrid_v3_5",
            validation_return=0.20,
            validation_mdd=0.10,
            oos_return=0.10,
        ),
        _row(
            "2025-01",
            "individual_robust:hybrid_v3_5",
            validation_return=0.10,
            validation_mdd=0.10,
            oos_return=0.00,
        ),
        _row(
            "2025-02",
            "relaxed_efficiency:hybrid_v3_5",
            validation_return=0.20,
            validation_mdd=0.10,
            oos_return=0.05,
        ),
        _row(
            "2025-02",
            "individual_robust:hybrid_v3_5",
            validation_return=0.10,
            validation_mdd=0.10,
            oos_return=0.20,
        ),
    ]
    source_path = tmp_path / "source.json"
    source_path.write_text('{"fold_candidate_rows": []}\n', "utf-8")

    payload = module.build_payload(
        {"fold_candidate_rows": rows},
        source_path=source_path,
        target_stems=("hybrid_v3_5",),
    )

    assert payload["status"] == "research_shadow_only_no_execution"
    assert payload["hard_gates"]["promotion_eligible"] is False
    assert payload["leak_checks"]["replay_uses_locked_oos_for_current_fold_selection_true"] == 0
    assert payload["leak_checks"]["replay_real_money_execution_true"] == 0
    assert payload["aggregate_rankings"][0]["candidate_label"] == (
        "same_stem_post_oos_update:hybrid_v3_5_lagged_top1_calmar"
    )
    assert payload["aggregate_rankings"][0]["requires_fresh_forward_shadow"] is True

    outputs = module.write_outputs(payload, output_dir=tmp_path, stem="same_stem_probe")
    assert Path(outputs["json"]).exists()
    markdown = Path(outputs["markdown"]).read_text("utf-8")
    assert "current fold OOS is hidden" in markdown
    assert "H3.5 monthly selected variants" in markdown
