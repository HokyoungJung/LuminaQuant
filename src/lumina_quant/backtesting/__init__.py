"""Backtesting package exports.

This module intentionally avoids eager imports to prevent circular-import
side effects when lightweight helper modules (e.g. cli_contract) are imported.
"""

__all__ = [
    "Backtest",
    "DataHandler",
    "ExecutionHandler",
    "HistoricCSVDataHandler",
    "HistoricParquetWindowedDataHandler",
    "Portfolio",
    "SimulatedExecutionHandler",
]


def __getattr__(name: str):
    if name == "Backtest":
        from lumina_quant.backtesting.backtest import Backtest as _Backtest

        return _Backtest
    if name in {"DataHandler", "HistoricCSVDataHandler"}:
        from lumina_quant.backtesting.data import DataHandler as _DataHandler
        from lumina_quant.backtesting.data import HistoricCSVDataHandler as _HistoricCSVDataHandler

        return _DataHandler if name == "DataHandler" else _HistoricCSVDataHandler
    if name == "HistoricParquetWindowedDataHandler":
        from lumina_quant.backtesting.data_windowed_parquet import (
            HistoricParquetWindowedDataHandler as _HistoricParquetWindowedDataHandler,
        )

        return _HistoricParquetWindowedDataHandler
    if name in {"ExecutionHandler", "SimulatedExecutionHandler"}:
        from lumina_quant.backtesting.execution_sim import ExecutionHandler as _ExecutionHandler
        from lumina_quant.backtesting.execution_sim import (
            SimulatedExecutionHandler as _SimulatedExecutionHandler,
        )

        return _ExecutionHandler if name == "ExecutionHandler" else _SimulatedExecutionHandler
    if name == "Portfolio":
        from lumina_quant.backtesting.portfolio_backtest import Portfolio as _Portfolio

        return _Portfolio
    raise AttributeError(name)
