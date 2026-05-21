from lumina_quant.metrics import compute_metrics
from lumina_quant.pipeline import run_backtest_pipeline, run_optimization_pipeline


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
