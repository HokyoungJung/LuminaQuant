import json

from lumina_quant.cli import main
from lumina_quant.config import load_public_pipeline_config
from lumina_quant.strategy_loader import DEFAULT_STRATEGY_REF

SAMPLE_CONFIG = "sample_configs/public_sample_pipeline.toml"
CUSTOM_STRATEGY = "tests.fixtures.custom_strategy:ThresholdLongStrategy"


def test_public_sample_config_loads_tuning_inputs() -> None:
    config = load_public_pipeline_config(SAMPLE_CONFIG)
    assert config.backtest.data_path == "sample_data/sample_ohlcv.csv"
    assert config.backtest.strategy_ref == DEFAULT_STRATEGY_REF
    assert config.backtest.strategy_params == {"fast_window": 3, "slow_window": 8}
    assert config.optimization.method == "optuna"
    assert set(config.optimization.search_space) == {"fast_window", "slow_window"}
    assert config.optimization.n_trials == 16
    assert config.optimization.sampler_seed == 7


def test_cli_optimize_defaults_to_optuna(capsys) -> None:
    assert main(["optimize", "--data", "sample_data/sample_ohlcv.csv", "--n-trials", "4"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["objective"] == "generic_score"
    assert payload["method"] == "optuna"
    assert payload["candidate_count"] > 0


def test_cli_optimize_uses_config(capsys) -> None:
    assert main(["optimize", "--config", SAMPLE_CONFIG]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["objective"] == "generic_score"
    assert payload["method"] == "optuna"
    assert payload["candidate_count"] > 0


def test_cli_arguments_override_config(capsys) -> None:
    assert main(
        [
            "optimize",
            "--config",
            SAMPLE_CONFIG,
            "--method",
            "grid",
            "--fast-grid",
            "2",
            "--slow-grid",
            "6",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "grid"
    assert payload["candidate_count"] == 1
    assert payload["best"]["fast_window"] == 2
    assert payload["best"]["slow_window"] == 6


def test_cli_loads_external_strategy_path(capsys) -> None:
    assert main(
        [
            "backtest",
            "--data",
            "sample_data/sample_ohlcv.csv",
            "--strategy",
            CUSTOM_STRATEGY,
            "--strategy-param",
            "start_after=2",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trade_count"] > 0
