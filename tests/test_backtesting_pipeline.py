from lumina_quant.data import load_ohlcv_csv
from lumina_quant.pipeline import run_backtest_pipeline
from lumina_quant.sample_strategy import MovingAverageCrossStrategy


def test_sample_data_loads() -> None:
    bars = load_ohlcv_csv("sample_data/sample_ohlcv.csv")
    assert len(bars) == 30
    assert bars[0].close == 100.8


def test_sample_strategy_validates_windows() -> None:
    strategy = MovingAverageCrossStrategy(fast_window=3, slow_window=8)
    assert strategy.fast_window == 3
    assert strategy.slow_window == 8


def test_backtest_pipeline_returns_deterministic_summary() -> None:
    result = run_backtest_pipeline("sample_data/sample_ohlcv.csv", fast_window=3, slow_window=8)
    assert result.initial_cash == 10_000.0
    assert result.trade_count > 0
    assert result.final_equity > 0.0
    assert 0.0 <= result.max_drawdown < 1.0
    assert len(result.equity_curve) == 30
