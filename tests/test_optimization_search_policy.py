from __future__ import annotations

from pathlib import Path

import pytest

from lumina_quant.optimization.search_policy import (
    build_bounded_grid_combinations,
    optimization_search_policy_payload,
    run_optuna_study,
    suggest_params_from_optuna_config,
)


class FakeTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def suggest_int(self, name, low, high, **kwargs):
        self.calls.append(("int", name))
        assert kwargs == {"step": 2}
        return int(low)

    def suggest_float(self, name, low, high, step=None):
        self.calls.append(("float", name))
        assert step == 0.5
        return float(high)

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name))
        return choices[-1]


def test_bounded_grid_requires_justification_and_records_truncation_metadata() -> None:
    with pytest.raises(ValueError, match="justification"):
        build_bounded_grid_combinations({"a": [1]}, justification="")

    result = build_bounded_grid_combinations(
        {"a": [1, 2], "b": ["x", "y"]},
        max_combinations=3,
        justification="finite integer leverage policy enumeration",
    )

    assert result.combinations == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
    ]
    assert result.metadata["search_method"] == "bounded_grid"
    assert result.metadata["raw_candidate_count"] == 4
    assert result.metadata["candidate_count"] == 3
    assert result.metadata["truncated"] is True
    assert (
        result.metadata["bounded_grid_justification"]
        == "finite integer leverage policy enumeration"
    )
    assert result.metadata["uses_locked_oos_for_selection"] is False
    assert result.metadata["uses_locked_oos_for_objective"] is False


def test_suggest_params_from_optuna_config_uses_canonical_schema() -> None:
    trial = FakeTrial()

    params = suggest_params_from_optuna_config(
        trial,
        {
            "window": {"type": "int", "low": 2, "high": 10, "step": 2},
            "threshold": {"type": "float", "low": 0.0, "high": 2.0, "step": 0.5},
            "side": {"type": "categorical", "choices": ["long", "short"]},
            "ignored": {"type": "unsupported", "low": 1, "high": 2},
        },
    )

    assert params == {"window": 2, "threshold": 2.0, "side": "short"}
    assert trial.calls == [("int", "window"), ("float", "threshold"), ("categorical", "side")]


def test_optuna_policy_payload_defaults_to_locked_oos_report_only_flags() -> None:
    payload = optimization_search_policy_payload(
        search_method="optuna",
        objective_policy={"objective_policy": "locked_train_val"},
        selection_inputs=("train", "validation"),
    )

    assert payload["search_method"] == "optuna"
    assert payload["objective_policy"] == {"objective_policy": "locked_train_val"}
    assert payload["selection_inputs"] == ["train", "validation"]
    assert payload["uses_locked_oos_for_selection"] is False
    assert payload["uses_locked_oos_for_objective"] is False
    assert payload["uses_locked_oos_for_pruning"] is False
    assert payload["uses_locked_oos_for_parameter_fitting"] is False


def test_run_optuna_study_centralizes_seeded_study_creation() -> None:
    optuna = pytest.importorskip("optuna")

    def objective(trial) -> float:
        value = trial.suggest_float("x", 0.0, 1.0)
        return -abs(value - 0.5)

    study = run_optuna_study(
        optuna_module=optuna,
        objective=objective,
        n_trials=3,
        seed=7,
        enqueue_trials=[{"x": 0.5}],
        show_progress_bar=False,
    )

    assert len(study.trials) == 3
    assert study.best_params == {"x": 0.5}
    assert study.best_value == pytest.approx(-0.0)


def test_migrated_optimization_surfaces_use_shared_search_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    migrated_paths = [
        root / "src" / "lumina_quant" / "cli" / "optimize.py",
        root / "scripts" / "research" / "optuna_tune_hybrid_online_portfolio.py",
    ]

    for path in migrated_paths:
        source = path.read_text(encoding="utf-8")
        assert "run_optuna_study" in source
        assert ".create_study(" not in source
    cli_source = migrated_paths[0].read_text(encoding="utf-8")
    assert "build_bounded_grid_combinations" in cli_source
    assert "itertools.product" not in cli_source
