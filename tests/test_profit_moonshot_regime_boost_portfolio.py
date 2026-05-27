from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path("scripts/research/run_state_distilled_regime_boost_portfolio.py")
SPEC = importlib.util.spec_from_file_location(
    "run_state_distilled_regime_boost_portfolio", MODULE_PATH
)
assert SPEC and SPEC.loader
regime_boost = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = regime_boost
SPEC.loader.exec_module(regime_boost)


def test_strategy_validity_fails_closed_on_calendar_and_oos_selection() -> None:
    card = regime_boost.validate_strategy_card(
        {
            "calendar_primary": False,
            "feature_fields": ["btc_ret_168h", "hour"],
            "source_coverage": {"panel": "real"},
            "uses_locked_oos_for_selection": True,
        }
    )

    assert card["strategy_valid"] is False
    assert "calendar_fields_present" in card["rejection_reasons"]
    assert "locked_oos_used_for_selection" in card["rejection_reasons"]


def test_dynamic_leverage_is_asset_vol_targeted_and_capped_at_25x() -> None:
    cfg = regime_boost.VolTargetLeverageConfig(
        max_effective_leverage=25.0, target_annual_volatility=2.5
    )

    low_vol = regime_boost.effective_dynamic_leverage(
        confidence=0.95,
        vol_ratio=0.50,
        long_term_vol=0.01,
        regime="bull",
        config=cfg,
        requested_max_leverage=25.0,
    )
    high_vol = regime_boost.effective_dynamic_leverage(
        confidence=0.95,
        vol_ratio=0.50,
        long_term_vol=2.0,
        regime="bull",
        config=cfg,
        requested_max_leverage=25.0,
    )
    stress = regime_boost.effective_dynamic_leverage(
        confidence=0.95,
        vol_ratio=0.50,
        long_term_vol=0.01,
        regime="stress",
        config=cfg,
        requested_max_leverage=25.0,
    )

    assert low_vol == 25.0
    assert 1.0 <= high_vol < low_vol
    assert stress == cfg.stress_leverage


def test_grid_cap_is_deterministic_and_hard_limited() -> None:
    cfg = regime_boost.RegimeBoostConfig(
        selection=regime_boost.SelectionConfig(grid_limit=10_000, hard_grid_cap=256)
    )
    configs, meta = regime_boost.enforce_grid_cap(regime_boost.grid_product(cfg), cfg.selection)

    assert meta["evaluated_count"] == min(meta["product_space_size"], 256)
    assert len(configs) == meta["evaluated_count"]
    assert meta["skipped_pruned_count"] == meta["product_space_size"] - meta["evaluated_count"]
    assert "search_space_hash" in meta


def test_selection_ignores_locked_oos_poison_candidate() -> None:
    rows = [
        {
            "name": "good_train_validation",
            "selection_score": 1.0,
            "uses_locked_oos_for_selection": False,
            "locked_oos_total_return": -0.50,
        },
        {
            "name": "oos_only_winner_forbidden",
            "selection_score": -1.0,
            "uses_locked_oos_for_selection": False,
            "locked_oos_total_return": 99.0,
        },
    ]

    selected = regime_boost.select_candidate_from_rows(rows)

    assert selected["name"] == "good_train_validation"


def test_neutral_pair_fit_uses_train_validation_only_even_with_oos_poison() -> None:
    rows = []
    for split, sol_ret, bnb_ret in [
        ("train", 0.01, -0.01),
        ("validation", 0.02, -0.02),
        ("locked_oos", 10.0, -10.0),
    ]:
        rows.append(
            {
                "split": split,
                "next_returns": {
                    "ETHUSDT": 0.001,
                    "SOLUSDT": sol_ret,
                    "BNBUSDT": bnb_ret,
                    "TRXUSDT": 0.002,
                },
            }
        )

    fit = regime_boost.fit_neutral_pair_overlay(rows, regime_boost.NeutralPairOverlayConfig())

    assert fit["uses_locked_oos_for_pair_fit"] is False
    assert fit["fit_splits"] == ["train", "validation"]
    assert fit["as_of_policy"] == "train_validation_lagged_features_only_frozen_before_locked_oos"


def test_strict_promotion_fails_on_liquidation_buffer_or_oos_mdd_but_not_return_mdd_ratio() -> None:
    positive_metrics = {
        "total_return": 0.01,
        "sharpe": 1.0,
        "sortino": 1.0,
        "smart_sortino": 1.0,
        "calmar": 1.0,
    }
    metrics = {
        "train": {
            "liquidation_count": 0,
            "minimum_margin_buffer": 1.0,
            "return_mdd": -100.0,
            **positive_metrics,
        },
        "validation": {
            "liquidation_count": 0,
            "minimum_margin_buffer": 1.0,
            "return_mdd": -100.0,
            **positive_metrics,
        },
        "locked_oos": {
            "liquidation_count": 0,
            "minimum_margin_buffer": 1.0,
            "max_drawdown": 0.10,
            "return_mdd": -100.0,
            **positive_metrics,
        },
    }
    promoted, reasons = regime_boost.strict_lane_promoted(metrics, regime_boost.SelectionConfig())
    assert promoted is True
    assert reasons == []

    metrics["locked_oos"]["max_drawdown"] = 0.251
    promoted, reasons = regime_boost.strict_lane_promoted(metrics, regime_boost.SelectionConfig())
    assert promoted is False
    assert "locked_oos_mdd_gt_25pct" in reasons

    metrics["locked_oos"]["max_drawdown"] = 0.10
    metrics["validation"]["liquidation_count"] = 1
    promoted, reasons = regime_boost.strict_lane_promoted(metrics, regime_boost.SelectionConfig())
    assert promoted is False
    assert "validation_liquidation_count_positive" in reasons


def test_freeze_hash_sidecar_and_locked_oos_gate_reference_identical_params(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"selection_score": 1.0}) + "\n")
    selected_config = regime_boost.dataclass_to_dict(regime_boost.RegimeBoostConfig())
    selected = {
        "config": selected_config,
        "selection_score": 1.0,
        "train": {"total_return": 0.1},
        "validation": {"total_return": 0.2},
    }
    freeze_payload = regime_boost.build_freeze_payload(
        selected=selected,
        ledger_path=ledger,
        input_manifest=[],
        grid_meta={"evaluated_count": 1},
        pair_fit={"uses_locked_oos_for_pair_fit": False},
    )

    freeze_path, sidecar_path, freeze_hash = regime_boost.write_freeze_artifacts(
        tmp_path, freeze_payload
    )
    sidecar = json.loads(sidecar_path.read_text())
    frozen = json.loads(freeze_path.read_text())

    assert "freeze_artifact_hash" not in frozen
    assert sidecar["freeze_artifact_hash"] == freeze_hash
    assert frozen["candidate_freeze_before_locked_oos_gate"] is True
    assert frozen["locked_oos_metrics_visible_during_selection"] is False
