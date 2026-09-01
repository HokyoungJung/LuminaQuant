from __future__ import annotations

from lumina_quant.strategies import registry as strategy_registry


def test_strategy_param_schema_and_defaults_exist_for_rsi():
    schema = strategy_registry.get_strategy_param_schema("RsiStrategy")
    defaults = strategy_registry.get_default_strategy_params("RsiStrategy")
    assert "rsi_period" in schema
    assert "oversold" in schema
    assert "overbought" in schema
    assert defaults["rsi_period"] == 14
    assert defaults["oversold"] == 30.0
    assert defaults["overbought"] == 70.0


def test_canonical_param_naming_scheme():
    canonical = strategy_registry.get_strategy_canonical_param_names("MovingAverageCrossStrategy")
    assert canonical["short_window"] == "moving_average_cross.short_window"
    assert canonical["long_window"] == "moving_average_cross.long_window"


def test_resolve_strategy_params_coerces_known_values_and_keeps_unknown():
    resolved = strategy_registry.resolve_strategy_params(
        "RsiStrategy",
        {
            "rsi_period": "9",
            "oversold": "-20",
            "overbought": "130",
            "allow_short": "false",
            "custom_note": "keep-me",
        },
    )
    assert resolved["rsi_period"] == 9
    assert resolved["oversold"] == 1.0
    assert resolved["overbought"] == 99.0
    assert resolved["allow_short"] is False
    assert resolved["custom_note"] == "keep-me"


def test_resolve_optuna_grid_configs_filter_unknown_params():
    optuna = strategy_registry.resolve_optuna_config(
        "RsiStrategy",
        {
            "n_trials": 77,
            "params": {
                "rsi_period": {"type": "int", "low": 6, "high": 18},
                "unknown": {"type": "float", "low": 0.1, "high": 0.2},
            },
        },
    )
    grid = strategy_registry.resolve_grid_config(
        "RsiStrategy",
        {
            "params": {
                "rsi_period": [8, 10, 12],
                "unknown": [1, 2, 3],
            }
        },
    )

    assert optuna["n_trials"] == 77
    assert "rsi_period" in optuna["params"]
    assert "unknown" not in optuna["params"]
    assert grid["params"]["rsi_period"] == [8, 10, 12]
    assert "unknown" not in grid["params"]


def test_resolve_strategy_params_does_not_use_canonical_alias_keys():
    resolved = strategy_registry.resolve_strategy_params(
        "RsiStrategy",
        {
            "rsi.rsi_period": 9,
        },
    )
    assert resolved["rsi_period"] == 14
    assert resolved["rsi.rsi_period"] == 9
