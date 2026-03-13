from __future__ import annotations

from lumina_quant.strategies import registry as strategy_registry


def test_public_sample_strategy_schema_and_defaults_exist():
    schema = strategy_registry.get_strategy_param_schema("PublicSampleStrategy")
    defaults = strategy_registry.get_default_strategy_params("PublicSampleStrategy")
    assert schema == {
        "decision_cadence_seconds": {"type": "int", "min": 1, "max": 3600}
    }
    assert defaults["decision_cadence_seconds"] == 20


def test_canonical_param_naming_scheme():
    canonical = strategy_registry.get_strategy_canonical_param_names("PublicSampleStrategy")
    assert canonical["decision_cadence_seconds"] == "public_sample.decision_cadence_seconds"


def test_resolve_strategy_params_coerces_known_values_and_keeps_unknown():
    resolved = strategy_registry.resolve_strategy_params(
        "PublicSampleStrategy",
        {
            "decision_cadence_seconds": "9",
            "custom_note": "keep-me",
        },
    )
    assert resolved["decision_cadence_seconds"] == 9
    assert resolved["custom_note"] == "keep-me"


def test_resolve_optuna_grid_configs_filter_unknown_params():
    optuna = strategy_registry.resolve_optuna_config(
        "PublicSampleStrategy",
        {
            "n_trials": 77,
            "params": {
                "decision_cadence_seconds": {"type": "int", "low": 6, "high": 18},
                "unknown": {"type": "float", "low": 0.1, "high": 0.2},
            },
        },
    )
    grid = strategy_registry.resolve_grid_config(
        "PublicSampleStrategy",
        {
            "params": {
                "decision_cadence_seconds": [8, 10, 12],
                "unknown": [1, 2, 3],
            }
        },
    )

    assert optuna["n_trials"] == 77
    assert "decision_cadence_seconds" in optuna["params"]
    assert "unknown" not in optuna["params"]
    assert grid["params"]["decision_cadence_seconds"] == [8, 10, 12]
    assert "unknown" not in grid["params"]


def test_resolve_unknown_public_strategy_name_falls_back_to_sample_defaults():
    resolved = strategy_registry.resolve_strategy_params(
        "RsiStrategy",
        {
            "decision_cadence_seconds": 9,
        },
    )
    assert resolved["decision_cadence_seconds"] == 9
