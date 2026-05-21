"""Composable public pipeline helpers."""

from __future__ import annotations

from lumina_quant.backtesting import BacktestResult, run_backtest
from lumina_quant.data import load_ohlcv_csv
from lumina_quant.live import PaperLiveResult, run_paper_live
from lumina_quant.optimization import OptimizationResult, run_grid_optimization
from lumina_quant.sample_strategy import MovingAverageCrossStrategy


def build_sample_strategy(fast_window: int = 3, slow_window: int = 8) -> MovingAverageCrossStrategy:
    return MovingAverageCrossStrategy(fast_window=fast_window, slow_window=slow_window)


def run_backtest_pipeline(
    data_path: str,
    *,
    fast_window: int = 3,
    slow_window: int = 8,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> BacktestResult:
    bars = load_ohlcv_csv(data_path)
    strategy = build_sample_strategy(fast_window=fast_window, slow_window=slow_window)
    return run_backtest(bars, strategy, initial_cash=initial_cash, fee_bps=fee_bps)


def run_paper_live_pipeline(
    data_path: str,
    *,
    fast_window: int = 3,
    slow_window: int = 8,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> PaperLiveResult:
    bars = load_ohlcv_csv(data_path)
    strategy = build_sample_strategy(fast_window=fast_window, slow_window=slow_window)
    return run_paper_live(bars, strategy, initial_cash=initial_cash, fee_bps=fee_bps)


def run_optimization_pipeline(
    data_path: str,
    *,
    fast_grid: str = "2,3,4",
    slow_grid: str = "6,8,10",
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> OptimizationResult:
    return run_grid_optimization(
        data_path,
        fast_grid=fast_grid,
        slow_grid=slow_grid,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
    )
