import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "write_portfolio_live_readiness_decision.py"
)
SPEC = importlib.util.spec_from_file_location(
    "write_portfolio_live_readiness_decision", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_live_readiness_decision_prefers_current_switch_mode(tmp_path: Path) -> None:
    switch_path = tmp_path / "switch.json"
    switch_path.write_text(
        json.dumps(
            {
                "recommended_mode": {
                    "mode": "hybrid_guarded_mode",
                    "allocation": {"hybrid_online_portfolio": 1.0},
                },
                "rationale": ["older", "Promote the current hybrid mode."],
                "current_market_state": {
                    "favored_group": "mixed",
                    "pair_liquidity_state": "normal",
                },
            }
        ),
        encoding="utf-8",
    )
    max_perf_path = tmp_path / "max_perf.json"
    max_perf_path.write_text(
        json.dumps({"winner": {"status": "retained_incumbent"}}), encoding="utf-8"
    )

    payload = MODULE.build_live_readiness_decision(
        switch_path=switch_path,
        max_perf_path=max_perf_path,
    )

    assert payload["decision"] == "selected_live_mode"
    assert payload["selected_mode"] == "hybrid_guarded_mode"
    assert payload["candidate_key"] == "hybrid_guarded_mode"
    assert payload["selection_basis"] == "current_operating_switch"


def test_build_live_readiness_decision_supports_production_guarded_switch_mode(
    tmp_path: Path,
) -> None:
    switch_path = tmp_path / "switch.json"
    switch_path.write_text(
        json.dumps(
            {
                "recommended_mode": {
                    "mode": "production_guarded_mode",
                    "allocation": {"production_guarded_portfolio": 1.0},
                },
                "rationale": [
                    "older",
                    "Use the current operating switch recommendation: production_guarded_mode.",
                ],
                "current_market_state": {
                    "favored_group": "mixed",
                    "pair_liquidity_state": "strong",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_live_readiness_decision(
        switch_path=switch_path,
        max_perf_path=tmp_path / "missing_max_perf.json",
    )

    assert payload["decision"] == "selected_live_mode"
    assert payload["selected_mode"] == "production_guarded_mode"
    assert payload["candidate_key"] == "production_guarded_mode"
    assert payload["decision_reason"].endswith("production_guarded_mode.")


def test_build_live_readiness_decision_falls_back_to_max_perf_when_switch_missing(
    tmp_path: Path,
) -> None:
    max_perf_path = tmp_path / "max_perf.json"
    max_perf_path.write_text(
        json.dumps(
            {
                "selection_basis": "locked_oos_robustness_gates",
                "winner": {
                    "status": "promoted_challenger",
                    "candidate_key": "challenger_55_45",
                    "label": "55/45 challenger",
                    "reason": "Promote the challenger.",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_live_readiness_decision(
        switch_path=tmp_path / "missing_switch.json",
        max_perf_path=max_perf_path,
    )

    assert payload["decision"] == "promote_candidate"
    assert payload["candidate_key"] == "challenger_55_45"
    assert payload["selection_basis"] == "locked_oos_robustness_gates"


def test_build_live_readiness_decision_can_prefer_promotion_review(tmp_path: Path) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "status": "promotion_ready_with_review",
                "recommendation": "promote_candidate_after_manual_review",
                "review_target": str(
                    tmp_path / "strict_autoresearch_1x_practical_shadow_latest.json"
                ),
                "selection_basis": "strict_autoresearch_1x_practical_candidate_review",
                "current_live_default": "production_guarded_mode",
            }
        ),
        encoding="utf-8",
    )
    switch_path = tmp_path / "switch.json"
    switch_path.write_text(
        json.dumps(
            {
                "recommended_mode": {
                    "mode": "production_guarded_mode",
                    "allocation": {"production_guarded_portfolio": 1.0},
                },
                "rationale": ["keep current"],
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_live_readiness_decision(
        review_path=review_path,
        switch_path=switch_path,
        max_perf_path=tmp_path / "missing.json",
        prefer_review=True,
    )

    assert payload["decision"] == "promote_candidate"
    assert payload["candidate_key"] == "strict_autoresearch_1x_practical_shadow_latest"
    assert payload["selected_mode"] == "strict_autoresearch_practical_mode"
    assert payload["candidate_mode"] == "strict_autoresearch_practical_mode"
    assert payload["selection_basis"] == "strict_autoresearch_1x_practical_candidate_review"


def test_build_live_readiness_decision_can_fallback_to_review_when_switch_missing(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "status": "shadow_only_pending_followup",
                "recommendation": "keep_shadow",
                "review_target": str(tmp_path / "candidate.json"),
                "selection_basis": "portfolio_promotion_review",
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_live_readiness_decision(
        review_path=review_path,
        switch_path=tmp_path / "missing_switch.json",
        max_perf_path=tmp_path / "missing_max_perf.json",
    )

    assert payload["decision"] == "keep_incumbent"
    assert payload["candidate_key"] == ""
    assert payload["selection_basis"] == "portfolio_promotion_review"


def test_build_live_readiness_decision_uses_explicit_candidate_mode_from_review(
    tmp_path: Path,
) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            {
                "status": "promotion_ready_with_review",
                "recommendation": "promote_candidate_after_manual_review",
                "review_target": str(tmp_path / "candidate.json"),
                "candidate_key": "production_guarded_40_state_vwap_pair_25_cash_35",
                "candidate_mode": "production_guarded_state_vwap_pair_mode",
                "selected_mode": "production_guarded_state_vwap_pair_mode",
                "selection_basis": "dense_pairs_state_vwap_overlay_candidate_review",
                "current_live_default": "production_guarded_mode",
            }
        ),
        encoding="utf-8",
    )

    payload = MODULE.build_live_readiness_decision(
        review_path=review_path,
        switch_path=tmp_path / "missing_switch.json",
        max_perf_path=tmp_path / "missing_max_perf.json",
    )

    assert payload["decision"] == "promote_candidate"
    assert payload["candidate_key"] == "production_guarded_40_state_vwap_pair_25_cash_35"
    assert payload["selected_mode"] == "production_guarded_state_vwap_pair_mode"
    assert payload["candidate_mode"] == "production_guarded_state_vwap_pair_mode"
    assert payload["selection_basis"] == "dense_pairs_state_vwap_overlay_candidate_review"
