"""Mode-based system assembly shared by backtest/live entrypoints."""

from __future__ import annotations


def build_system(mode: str) -> dict[str, object]:
    selected = str(mode or "").strip().lower()
    if selected == "backtest":
        from lumina_quant.backtesting.backtest import Backtest
        from lumina_quant.backtesting.data import HistoricCSVDataHandler
        from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
        from lumina_quant.backtesting.portfolio_backtest import Portfolio

        return {
            "mode": "backtest",
            "engine_cls": Backtest,
            "data_handler_cls": HistoricCSVDataHandler,
            "execution_handler_cls": SimulatedExecutionHandler,
            "portfolio_cls": Portfolio,
        }

    if selected in {"live", "sandbox"}:
        from lumina_quant.backtesting.portfolio_backtest import Portfolio
        from lumina_quant.live.execution_live import LiveExecutionHandler
        from lumina_quant.live.trader import LiveTrader

        data_handler_cls = None
        if selected == "live":
            from lumina_quant.live.data_poll import LiveDataHandler

            data_handler_cls = LiveDataHandler
        else:
            from lumina_quant.live.data_poll import LiveDataHandler

            data_handler_cls = LiveDataHandler

        return {
            "mode": selected,
            "engine_cls": LiveTrader,
            "data_handler_cls": data_handler_cls,
            "execution_handler_cls": LiveExecutionHandler,
            "portfolio_cls": Portfolio,
        }

    raise ValueError(f"Unsupported system mode: {mode}")
