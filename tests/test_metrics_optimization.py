from lumina_quant.config import SearchParam
from lumina_quant.metrics import compute_metrics
from lumina_quant.pipeline import run_backtest_pipeline, run_optimization_pipeline

CUSTOM_STRATEGY = "tests.fixtures.custom_strategy:ThresholdLongStrategy"


def test_metrics_summary_has_generic_score() -> None:
    result = run_backtest_pipeline("sample_data/sample_ohlcv.csv")
    metrics = compute_metrics(result)
    assert metrics.trade_count == result.trade_count
    assert metrics.total_return == result.total_return
    assert metrics.score < metrics.total_return


def test_grid_optimization_returns_best_candidate() -> None:
    result = run_optimization_pipeline(
        "sample_data/sample_ohlcv.csv",
        method="grid",
        fast_grid="2,3",
        slow_grid="6,8",
    )
    assert result.objective == "generic_score"
    assert result.method == "grid"
    assert result.best in result.candidates
    assert len(result.candidates) == 4


def test_optuna_optimization_returns_best_candidate() -> None:
    result = run_optimization_pipeline(
        "sample_data/sample_ohlcv.csv",
        method="optuna",
        fast_grid="2,3",
        slow_grid="6,8",
        n_trials=8,
        sampler_seed=11,
    )
    assert result.objective == "generic_score"
    assert result.method == "optuna"
    assert result.best in result.candidates
    assert len(result.candidates) > 0
    assert all(candidate.trial_number is not None for candidate in result.candidates)


def test_custom_strategy_can_use_pipeline_and_optuna_search_space() -> None:
    backtest = run_backtest_pipeline(
        "sample_data/sample_ohlcv.csv",
        strategy_ref=CUSTOM_STRATEGY,
        strategy_params={"start_after": 2},
    )
    assert backtest.trade_count > 0

    optimized = run_optimization_pipeline(
        "sample_data/sample_ohlcv.csv",
        strategy_ref=CUSTOM_STRATEGY,
        method="optuna",
        search_space={"start_after": SearchParam(kind="int", low=1, high=3, step=1)},
        n_trials=6,
        sampler_seed=3,
    )
    assert optimized.strategy_ref == CUSTOM_STRATEGY
    assert optimized.method == "optuna"
    assert "start_after" in optimized.best.params
