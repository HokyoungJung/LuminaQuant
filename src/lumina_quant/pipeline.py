"""Composable public pipeline helpers."""

from __future__ import annotations

from typing import Any

from lumina_quant.backtesting import BacktestResult, run_backtest
from lumina_quant.config import SearchParam
from lumina_quant.data import load_ohlcv_csv
from lumina_quant.live import PaperLiveResult, run_paper_live
from lumina_quant.optimization import (
    OptimizationResult,
    run_grid_optimization,
    run_optuna_optimization,
)
from lumina_quant.sample_strategy import MovingAverageCrossStrategy
from lumina_quant.strategy_loader import DEFAULT_STRATEGY_REF, StrategyProtocol, build_strategy


def build_sample_strategy(fast_window: int = 3, slow_window: int = 8) -> MovingAverageCrossStrategy:
    return MovingAverageCrossStrategy(fast_window=fast_window, slow_window=slow_window)


def build_pipeline_strategy(
    strategy_ref: str = DEFAULT_STRATEGY_REF,
    strategy_params: dict[str, Any] | None = None,
    *,
    fast_window: int | None = None,
    slow_window: int | None = None,
) -> StrategyProtocol:
    params = dict(strategy_params or {})
    if fast_window is not None:
        params["fast_window"] = fast_window
    if slow_window is not None:
        params["slow_window"] = slow_window
    return build_strategy(strategy_ref, params)


def run_backtest_pipeline(
    data_path: str,
    *,
    strategy_ref: str = DEFAULT_STRATEGY_REF,
    strategy_params: dict[str, Any] | None = None,
    fast_window: int | None = None,
    slow_window: int | None = None,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> BacktestResult:
    bars = load_ohlcv_csv(data_path)
    strategy = build_pipeline_strategy(
        strategy_ref,
        strategy_params,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    return run_backtest(bars, strategy, initial_cash=initial_cash, fee_bps=fee_bps)


def run_paper_live_pipeline(
    data_path: str,
    *,
    strategy_ref: str = DEFAULT_STRATEGY_REF,
    strategy_params: dict[str, Any] | None = None,
    fast_window: int | None = None,
    slow_window: int | None = None,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> PaperLiveResult:
    bars = load_ohlcv_csv(data_path)
    strategy = build_pipeline_strategy(
        strategy_ref,
        strategy_params,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    return run_paper_live(bars, strategy, initial_cash=initial_cash, fee_bps=fee_bps)


def run_optimization_pipeline(
    data_path: str,
    *,
    strategy_ref: str = DEFAULT_STRATEGY_REF,
    strategy_params: dict[str, Any] | None = None,
    method: str = "optuna",
    fast_grid: str = "2,3,4",
    slow_grid: str = "6,8,10",
    search_space: dict[str, SearchParam] | None = None,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
    n_trials: int = 16,
    sampler_seed: int = 7,
) -> OptimizationResult:
    if method == "grid":
        return run_grid_optimization(
            data_path,
            strategy_ref=strategy_ref,
            strategy_params=strategy_params,
            fast_grid=fast_grid,
            slow_grid=slow_grid,
            search_space=search_space,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
        )
    if method == "optuna":
        return run_optuna_optimization(
            data_path,
            strategy_ref=strategy_ref,
            strategy_params=strategy_params,
            fast_grid=fast_grid,
            slow_grid=slow_grid,
            search_space=search_space,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            n_trials=n_trials,
            sampler_seed=sampler_seed,
        )
    raise ValueError("optimization method must be 'grid' or 'optuna'")
