from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts" / "research" / "run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_alpha_zoo_integer_leverage_optuna_hybrid_decision", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _dates() -> pd.DatetimeIndex:
    train = pd.date_range("2025-01-01 00:00:00", periods=40, freq="h")
    validation = pd.date_range("2026-01-01 00:00:00", periods=30, freq="h")
    oos = pd.date_range("2026-04-01 00:00:00", periods=24, freq="h")
    return pd.DatetimeIndex([*train, *validation, *oos])


def _stream(profile_id: str, train_ret: float, val_ret: float, oos_ret: float) -> object:
    index = _dates()
    values = [train_ret] * 40 + [val_ret] * 30 + [oos_ret] * 24
    return MODULE.grid_hybrid.ProfileStream(
        profile_id=profile_id,
        candidate_tier="test",
        leverage_map={profile_id: 1},
        gross_notional_fraction=1.0,
        asset_gross_notional_fraction={profile_id: 1.0},
        selected_model_ids=(profile_id,),
        returns=pd.Series(values, index=index, dtype=float),
        turnover_by_split={"train": 1.0, "validation": 1.0, "locked_oos": 1.0},
        trade_events_by_split={"train": 40, "validation": 30, "locked_oos": 24},
        liquidation_count_by_split={"train": 0, "validation": 0, "locked_oos": 0},
    )


def test_objective_score_ignores_locked_oos_report_only_fields() -> None:
    row = {
        "train_return": 0.30,
        "validation_return": 0.12,
        "train_mdd": 0.05,
        "validation_mdd": 0.04,
        "train_return_per_turnover_proxy_bps": 30.0,
        "validation_return_per_turnover_proxy_bps": 25.0,
        "gross_notional_fraction": 2.0,
        "locked_oos_return_report_only": -0.95,
        "locked_oos_mdd_report_only": 0.99,
        "locked_oos_return_per_turnover_proxy_bps_report_only": -200.0,
    }
    changed_oos = dict(row)
    changed_oos.update(
        {
            "locked_oos_return_report_only": 2.50,
            "locked_oos_mdd_report_only": 0.01,
            "locked_oos_return_per_turnover_proxy_bps_report_only": 400.0,
        }
    )

    assert MODULE._objective_score(row) == pytest.approx(MODULE._objective_score(changed_oos))


def test_v35_v36_run_model_keeps_real_money_disabled_and_cost_contract() -> None:
    streams = [
        _stream("balanced", 0.004, 0.003, 0.001),
        _stream("growth", 0.006, 0.004, 0.0015),
        _stream("aggressive", 0.007, 0.005, 0.0012),
    ]

    for version, profile_id in (("v3_5", MODULE.V35_PROFILE_ID), ("v3_6", MODULE.V36_PROFILE_ID)):
        result = MODULE._run_model(
            streams, MODULE.HybridParams(), version=version, profile_id=profile_id
        )

        assert result.row["ready_for_real"] is False
        assert result.row["real_money_execution"] is False
        assert (
            result.row["train_return_per_turnover_proxy_bps"]
            > MODULE.ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS
        )
        assert (
            result.row["validation_return_per_turnover_proxy_bps"]
            > MODULE.ilp.RETURN_PER_TURNOVER_THRESHOLD_BPS
        )
        assert MODULE.ilp.PRIMARY_ROUND_TRIP_COST_BPS == 10.0
        assert set(result.row["average_weights_train_validation"]) == {
            "balanced",
            "growth",
            "aggressive",
        }
        assert sum(result.row["average_weights_train_validation"].values()) == pytest.approx(1.0)


def test_standard_live_refit_fits_train_only_and_disables_oos_gate() -> None:
    streams = [
        _stream("balanced", 0.004, 0.003, -0.01),
        _stream("growth", 0.006, 0.004, -0.02),
        _stream("aggressive", 0.007, 0.005, -0.03),
    ]
    split_windows = {
        "train": ("2025-01-01 00:00:00", "2025-01-02 15:00:00"),
        "validation": ("2026-01-01 00:00:00", "2026-01-02 05:00:00"),
        "locked_oos": ("2026-04-02 00:00:00", "2026-04-01 23:00:00"),
    }

    with MODULE._split_window_context(split_windows):
        result = MODULE._run_model(
            streams,
            MODULE.HybridParams(),
            version="v3_5",
            profile_id=MODULE.V35_PROFILE_ID,
            fit_splits=("train",),
            require_locked_oos_gate=False,
        )

    assert result.row["fit_splits"] == ["train"]
    assert result.row["warmup_splits"] == ["train"]
    assert result.row["warmup_policy"] == "warmup_ratio_applies_to_train_split_only"
    assert result.row["final_refit"] is False
    assert result.row["locked_oos_gate_required"] is False
    assert result.row["test_set_policy"] == "disabled_for_live_final_refit_no_test_set_reserved"
    assert result.row["report_only_gate_reasons"] == []
    assert result.row["ready_for_real"] is False
    assert result.row["real_money_execution"] is False
    assert result.row["locked_oos_trade_event_count_report_only"] == 0


def test_final_refit_records_train_validation_fit_inputs() -> None:
    streams = [
        _stream("balanced", 0.004, 0.003, 0.001),
        _stream("growth", 0.006, 0.004, 0.0015),
        _stream("aggressive", 0.007, 0.005, 0.0012),
    ]

    result = MODULE._run_model(
        streams,
        MODULE.HybridParams(),
        version="v3_5",
        profile_id=MODULE.V35_PROFILE_ID,
        fit_splits=("train", "validation"),
        final_refit=True,
        require_locked_oos_gate=False,
    )

    assert result.row["fit_splits"] == ["train", "validation"]
    assert result.row["warmup_splits"] == ["train"]
    assert result.row["final_refit"] is True
    assert result.row["test_set_policy"] == "disabled_for_live_final_refit_no_test_set_reserved"


def test_learn_params_uses_train_warmup_even_when_fit_includes_validation() -> None:
    train = [[0.02, -0.01, -0.01]] * 30
    validation = [[-0.01, 0.05, -0.01]] * 30
    returns = pd.DataFrame([*train, *validation]).to_numpy(dtype=float)
    params = MODULE.HybridParams(warmup_ratio=0.5)

    learned = MODULE._learn_params(
        returns,
        params,
        opt_indices=pd.Index(range(60)).to_numpy(dtype=int),
        warmup_indices=pd.Index(range(30)).to_numpy(dtype=int),
    )

    assert learned.default_idx == 0


def test_all_exposed_hybrid_params_are_in_optuna_config() -> None:
    assert set(MODULE._trial_params_from_hybrid(MODULE.HybridParams())) == set(
        MODULE.HYBRID_OPTUNA_CONFIG
    )


def test_selected_optuna_sort_key_does_not_prefer_oos_spike() -> None:
    row_a = {
        "profile_id": "a",
        "train_return": 0.30,
        "validation_return": 0.10,
        "train_mdd": 0.05,
        "validation_mdd": 0.04,
        "train_return_per_turnover_proxy_bps": 30.0,
        "validation_return_per_turnover_proxy_bps": 25.0,
        "gross_notional_fraction": 2.0,
        "locked_oos_return_report_only": -0.90,
        "locked_oos_liquidation_count_report_only": 0,
        "locked_oos_account_wipeout_count_report_only": 0,
        "locked_oos_trade_event_count_report_only": 24,
        "locked_oos_return_per_turnover_proxy_bps_report_only": -90.0,
    }
    row_b = dict(row_a)
    row_b.update(
        {
            "profile_id": "b",
            "train_return": 0.20,
            "validation_return": 0.08,
            "locked_oos_return_report_only": 5.0,
            "locked_oos_return_per_turnover_proxy_bps_report_only": 500.0,
        }
    )
    fake_a = MODULE.OptunaModelResult(
        row=row_a,
        returns=pd.Series(dtype=float),
        weights=pd.DataFrame(),
        allocations=[],
        learned_params=MODULE.LearnedParams(0, 0, 0, 0.5, 0.1, 0),
        params=MODULE.HybridParams(),
    )
    fake_b = MODULE.OptunaModelResult(
        row=row_b,
        returns=pd.Series(dtype=float),
        weights=pd.DataFrame(),
        allocations=[],
        learned_params=MODULE.LearnedParams(0, 0, 0, 0.5, 0.1, 0),
        params=MODULE.HybridParams(),
    )

    selected = MODULE._choose_selected_optuna([fake_a, fake_b])

    assert selected["profile_id"] == "a"


def test_methodology_contract_names_v35_v36_and_optuna() -> None:
    params = MODULE.HybridParams()

    assert params.max_single_weight == pytest.approx(0.78)
    assert MODULE.V35_PROFILE_ID.startswith("hybrid_v3_5_optuna")
    assert MODULE.V36_PROFILE_ID.startswith("hybrid_v3_6_optuna")
    assert MODULE.ARTIFACT_KIND.endswith("optuna_hybrid_decision")
