"""Chunked backtest runner to avoid full-range in-memory datasets."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.data_windowed_parquet import HistoricParquetWindowedDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio


def iter_chunk_windows(
    *,
    start_date: datetime,
    end_date: datetime,
    chunk_days: int,
) -> list[tuple[datetime, datetime]]:
    if chunk_days <= 0:
        return [(start_date, end_date)]

    windows: list[tuple[datetime, datetime]] = []
    cursor = start_date
    delta = timedelta(days=int(chunk_days))
    while cursor <= end_date:
        chunk_end = min(end_date, cursor + delta - timedelta(microseconds=1))
        windows.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(microseconds=1)
    return windows


def _capture_backtest_state(
    backtest: Backtest,
    *,
    record_history: bool,
    track_metrics: bool,
    record_trades: bool,
) -> dict[str, Any]:
    strategy_state: dict[str, Any] = {}
    get_strategy_state = getattr(backtest.strategy, "get_state", None)
    if callable(get_strategy_state):
        raw = get_strategy_state()
        if isinstance(raw, dict):
            strategy_state = raw

    portfolio_state = {}
    get_portfolio_state = getattr(backtest.portfolio, "get_state", None)
    if callable(get_portfolio_state):
        raw = get_portfolio_state()
        if isinstance(raw, dict):
            portfolio_state = raw

    execution_state = {}
    get_execution_state = getattr(backtest.execution_handler, "get_state", None)
    if callable(get_execution_state):
        raw = get_execution_state()
        if isinstance(raw, dict):
            execution_state = raw

    carry: dict[str, Any] = {
        "strategy_state": strategy_state,
        "portfolio_state": portfolio_state,
        "execution_state": execution_state,
        "engine_state": (
            backtest.get_engine_state()
            if callable(getattr(backtest, "get_engine_state", None))
            else {}
        ),
        "trade_count": int(getattr(backtest.portfolio, "trade_count", 0)),
        "active_orders": deepcopy(getattr(backtest.execution_handler, "active_orders", [])),
        "order_seq": int(getattr(backtest.execution_handler, "_order_seq", 0)),
    }
    if bool(record_history):
        carry["all_positions"] = getattr(backtest.portfolio, "all_positions", [])
        carry["all_holdings"] = getattr(backtest.portfolio, "all_holdings", [])
    if bool(record_trades):
        carry["trades"] = getattr(backtest.portfolio, "trades", [])
    if bool(track_metrics):
        carry["metric_totals"] = getattr(backtest.portfolio, "_metric_totals", [])
        carry["metric_benchmarks"] = getattr(backtest.portfolio, "_metric_benchmarks", [])
    return carry


def _restore_backtest_state(backtest: Backtest, carry: dict[str, Any]) -> None:
    if not carry:
        return

    set_strategy_state = getattr(backtest.strategy, "set_state", None)
    if callable(set_strategy_state):
        set_strategy_state(dict(carry.get("strategy_state", {})))

    set_portfolio_state = getattr(backtest.portfolio, "set_state", None)
    if callable(set_portfolio_state):
        set_portfolio_state(dict(carry.get("portfolio_state", {})))

    set_execution_state = getattr(backtest.execution_handler, "set_state", None)
    if callable(set_execution_state):
        set_execution_state(dict(carry.get("execution_state", {})))

    set_engine_state = getattr(backtest, "set_engine_state", None)
    if callable(set_engine_state):
        set_engine_state(dict(carry.get("engine_state", {})))

    if "all_positions" in carry:
        backtest.portfolio.all_positions = carry["all_positions"]
    if "all_holdings" in carry:
        backtest.portfolio.all_holdings = carry["all_holdings"]
    if "trades" in carry:
        backtest.portfolio.trades = carry["trades"]
    if "trade_count" in carry:
        backtest.portfolio.trade_count = int(carry["trade_count"])
    if "metric_totals" in carry:
        backtest.portfolio._metric_totals = carry["metric_totals"]
    if "metric_benchmarks" in carry:
        backtest.portfolio._metric_benchmarks = carry["metric_benchmarks"]

    if hasattr(backtest.execution_handler, "active_orders"):
        backtest.execution_handler.active_orders = deepcopy(carry.get("active_orders", []))
    if hasattr(backtest.execution_handler, "_order_seq"):
        backtest.execution_handler._order_seq = int(carry.get("order_seq", 0))


def run_backtest_chunked(
    *,
    csv_dir: str,
    symbol_list: list[str],
    start_date: datetime,
    end_date: datetime,
    strategy_cls,
    strategy_params: dict[str, Any] | None,
    data_loader: Callable[[datetime, datetime], dict[str, Any]],
    chunk_days: int,
    strategy_timeframe: str,
    data_handler_cls=None,
    execution_handler_cls=SimulatedExecutionHandler,
    portfolio_cls=Portfolio,
    backtest_mode: str = "windowed",
    data_handler_kwargs: dict[str, Any] | None = None,
    record_history: bool = True,
    track_metrics: bool = True,
    record_trades: bool = True,
) -> Backtest:
    """Execute one logical backtest by loading bounded chunks sequentially."""
    mode_token = str(backtest_mode or "windowed").strip().lower()
    if mode_token not in {"windowed", "legacy_batch", "legacy_1s"}:
        mode_token = "windowed"

    selected_data_handler_cls = data_handler_cls
    if selected_data_handler_cls is None:
        selected_data_handler_cls = (
            HistoricParquetWindowedDataHandler
            if mode_token == "windowed"
            else HistoricCSVDataHandler
        )

    handler_kwargs = dict(data_handler_kwargs or {})

    windows = iter_chunk_windows(
        start_date=start_date,
        end_date=end_date,
        chunk_days=max(1, int(chunk_days)),
    )

    carry: dict[str, Any] = {}
    final_backtest: Backtest | None = None

    for chunk_start, chunk_end in windows:
        chunk_data = data_loader(chunk_start, chunk_end)
        if not chunk_data:
            continue

        backtest = Backtest(
            csv_dir=csv_dir,
            symbol_list=symbol_list,
            start_date=chunk_start,
            end_date=chunk_end,
            data_handler_cls=selected_data_handler_cls,
            execution_handler_cls=execution_handler_cls,
            portfolio_cls=portfolio_cls,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params or {},
            data_dict=chunk_data,
            data_handler_kwargs=handler_kwargs,
            record_history=record_history,
            track_metrics=track_metrics,
            record_trades=record_trades,
            strategy_timeframe=str(strategy_timeframe),
        )

        if carry:
            _restore_backtest_state(backtest, carry)

        backtest.simulate_trading(output=False)
        carry = _capture_backtest_state(
            backtest,
            record_history=bool(record_history),
            track_metrics=bool(track_metrics),
            record_trades=bool(record_trades),
        )
        final_backtest = backtest

    if final_backtest is not None:
        return final_backtest

    empty = Backtest(
        csv_dir=csv_dir,
        symbol_list=symbol_list,
        start_date=start_date,
        end_date=end_date,
        data_handler_cls=selected_data_handler_cls,
        execution_handler_cls=execution_handler_cls,
        portfolio_cls=portfolio_cls,
        strategy_cls=strategy_cls,
        strategy_params=strategy_params or {},
        data_dict={},
        data_handler_kwargs=handler_kwargs,
        record_history=record_history,
        track_metrics=track_metrics,
        record_trades=record_trades,
        strategy_timeframe=str(strategy_timeframe),
    )
    empty.simulate_trading(output=False)
    return empty
