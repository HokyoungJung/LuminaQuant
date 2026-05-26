from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from lumina_quant.portfolio.optimizer_core import (
    LOCKED_OOS_OBJECTIVE_POLICY,
    StreamCache,
    apply_caps,
    asset_exposure,
    build_portfolio_returns,
    build_portfolio_stream,
    canonical_split,
    metals_exposure,
    objective_policy_payload,
    project_simplex_with_upper_bounds,
)

ROOT = Path(__file__).resolve().parents[1]
HYBRID_MODULE_PATH = ROOT / "scripts" / "research" / "optuna_tune_hybrid_online_portfolio.py"
HYBRID_TUNING_MODULE_PATH = ROOT / "scripts" / "research" / "tune_hybrid_online_portfolio.py"
SPEC = importlib.util.spec_from_file_location("optuna_tune_hybrid_online_portfolio", HYBRID_MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load optuna_tune_hybrid_online_portfolio module")
hybrid_optuna = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hybrid_optuna
SPEC.loader.exec_module(hybrid_optuna)

TUNING_SPEC = importlib.util.spec_from_file_location("tune_hybrid_online_portfolio", HYBRID_TUNING_MODULE_PATH)
if TUNING_SPEC is None or TUNING_SPEC.loader is None:
    raise RuntimeError("Failed to load tune_hybrid_online_portfolio module")
hybrid_tuning = importlib.util.module_from_spec(TUNING_SPEC)
sys.modules[TUNING_SPEC.name] = hybrid_tuning
TUNING_SPEC.loader.exec_module(hybrid_tuning)


def test_stream_cache_aggregates_timestamps_and_reuses_aligned_returns() -> None:
    rows = {
        "a": {
            "candidate_id": "a",
            "return_streams": {
                "val": [
                    {"t": "2026-01-02T00:00:00Z", "v": 0.02},
                    {"t": "2026-01-01T00:00:00Z", "v": 0.01},
                    {"t": "2026-01-01T00:00:00Z", "v": 0.005},
                ]
            },
        },
        "b": {
            "candidate_id": "b",
            "return_streams": {
                "validation": [
                    {"timestamp": "2026-01-02T00:00:00+00:00", "v": 0.03},
                    {"timestamp": "2026-01-03T00:00:00Z", "v": 0.04},
                ]
            },
        },
    }
    cache = StreamCache()

    stream = build_portfolio_stream({"a": 0.5, "b": 0.25}, rows, split="validation", cache=cache)

    assert [point["t"] for point in stream] == [
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        "2026-01-03T00:00:00Z",
    ]
    assert [point["v"] for point in stream] == pytest.approx([0.0075, 0.0175, 0.01])
    assert build_portfolio_returns({"a": 0.5, "b": 0.25}, rows, split="val", cache=cache).tolist() == pytest.approx(
        [0.0075, 0.0175, 0.01]
    )
    assert build_portfolio_returns({"b": 1.0}, rows, split="val", cache=StreamCache()).tolist() == pytest.approx(
        [0.03, 0.04]
    )
    assert canonical_split("test_oos", default="val") == "oos"


def test_objective_policy_payload_labels_locked_oos_contract() -> None:
    payload = objective_policy_payload("locked_train_val", oos_is_objective_input=False)

    assert payload["objective_policy"] == LOCKED_OOS_OBJECTIVE_POLICY
    assert payload["oos_is_objective_input"] is False
    assert payload["locked_oos_label"] == "locked_oos_report_only"


def test_apply_caps_enforces_strategy_asset_and_metals_exposure() -> None:
    records = {
        "trend_btc": {"candidate_id": "trend_btc", "family": "trend", "symbols": ["BTC/USDT"]},
        "trend_eth": {"candidate_id": "trend_eth", "family": "trend", "symbols": ["ETH/USDT"]},
        "metal_mix": {"candidate_id": "metal_mix", "family": "macro", "symbols": ["XAU/USDT", "BTC/USDT"]},
    }
    weights, caps = apply_caps(
        {"trend_btc": 0.80, "trend_eth": 0.10, "metal_mix": 0.10},
        records=records,
        max_strategy=0.50,
        max_family=0.60,
        max_asset=0.55,
        max_metals=0.10,
    )

    assert caps["max_strategy"] == pytest.approx(0.50)
    assert all(weight <= 0.500001 for weight in weights.values())
    assert weights["trend_btc"] + weights["trend_eth"] <= 0.600001
    assert asset_exposure(weights, records, "BTC/USDT") <= 0.550001
    assert metals_exposure(weights, records) <= 0.100001


def test_apply_caps_reserves_cash_when_asset_cap_limits_all_candidates() -> None:
    records = {
        f"btc_{idx}": {"candidate_id": f"btc_{idx}", "family": f"family_{idx}", "symbols": ["BTC/USDT"]}
        for idx in range(3)
    }

    weights, caps = apply_caps(
        dict.fromkeys(records, 1.0),
        records=records,
        max_strategy=0.50,
        max_family=1.0,
        max_asset=0.20,
        max_metals=1.0,
    )

    assert caps["max_asset"] == pytest.approx(0.20)
    assert caps["active_weight"] == pytest.approx(0.20)
    assert caps["cash_reserve_weight"] == pytest.approx(0.80)
    assert sum(weights.values()) == pytest.approx(0.20)
    assert asset_exposure(weights, records, "BTC/USDT") <= 0.200001


def test_project_simplex_with_upper_bounds_preserves_capacity_and_preferences() -> None:
    weights = project_simplex_with_upper_bounds(
        {"a": 10.0, "b": 1.0, "c": 1.0},
        upper={"a": 0.50, "b": 0.40, "c": 0.40},
        target_sum=1.0,
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] == pytest.approx(0.50)
    assert weights["b"] == pytest.approx(weights["c"])


def test_hybrid_optuna_locked_profile_ignores_oos_metrics() -> None:
    base = {
        "scenarios": {
            "refreshed_latest_tail": {
                "split_metrics": {
                    "train": {"total_return": 0.02, "sharpe": 1.4, "max_drawdown": 0.02},
                    "val": {"total_return": 0.03, "sharpe": 1.6, "max_drawdown": 0.01},
                    "oos": {"total_return": -0.95, "sharpe": -40.0, "max_drawdown": 0.9},
                }
            },
            "historical_saved_baseline": {
                "split_metrics": {
                    "oos": {"total_return": -0.5, "sharpe": -10.0, "max_drawdown": 0.7}
                }
            },
        },
        "readiness": {"beats_cash_refreshed": True, "pair_cap_respected": True},
    }
    better_oos_only = deepcopy(base)
    better_oos_only["scenarios"]["refreshed_latest_tail"]["split_metrics"]["oos"] = {
        "total_return": 99.0,
        "sharpe": 500.0,
        "max_drawdown": 0.0,
    }
    better_oos_only["scenarios"]["historical_saved_baseline"]["split_metrics"]["oos"] = {
        "total_return": 88.0,
        "sharpe": 400.0,
        "max_drawdown": 0.0,
    }

    assert hybrid_optuna._objective_policy_for_profile("locked_train_val")["objective_policy"] == LOCKED_OOS_OBJECTIVE_POLICY
    assert hybrid_optuna._objective_from_payload(base, profile="locked_train_val") == pytest.approx(
        hybrid_optuna._objective_from_payload(better_oos_only, profile="locked_train_val")
    )
    assert hybrid_tuning._objective(base, profile="locked_train_val") == pytest.approx(
        hybrid_tuning._objective(better_oos_only, profile="locked_train_val")
    )
    assert hybrid_tuning._objective_policy_for_profile("locked_train_val")["objective_policy"] == LOCKED_OOS_OBJECTIVE_POLICY
    assert hybrid_optuna._objective_from_payload(base, profile="live_guarded") != pytest.approx(
        hybrid_optuna._objective_from_payload(better_oos_only, profile="live_guarded")
    )


def test_hybrid_optuna_search_policy_records_locked_oos_non_use() -> None:
    payload = hybrid_optuna._search_policy_for_profile("locked_train_val", n_trials=0)

    assert payload["search_method"] == "optuna"
    assert payload["requested_trials"] == 1
    assert payload["direction"] == "maximize"
    assert payload["sampler"] == "TPESampler"
    assert payload["seed"] == 42
    assert payload["selection_policy"] == "best_objective_desc"
    assert payload["selection_inputs"] == ["train", "validation"]
    assert payload["search_space_source"].endswith("::_build_config")
    assert payload["objective_policy"]["objective_policy"] == LOCKED_OOS_OBJECTIVE_POLICY
    assert payload["uses_locked_oos_for_selection"] is False
    assert payload["uses_locked_oos_for_objective"] is False
    assert payload["uses_locked_oos_for_pruning"] is False
    assert payload["uses_locked_oos_for_parameter_fitting"] is False
