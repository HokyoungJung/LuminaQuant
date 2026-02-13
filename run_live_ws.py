import argparse
import os

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import LiveConfig
from lumina_quant.live.execution_live import LiveExecutionHandler
from lumina_quant.live.trader import LiveTrader
from strategies.rsi_strategy import RsiStrategy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LuminaQuant live trader (WebSocket).")
    parser.add_argument(
        "--enable-live-real",
        action="store_true",
        help="Explicitly allow real trading mode.",
    )
    args = parser.parse_args()
    if args.enable_live_real:
        os.environ["LUMINA_ENABLE_LIVE_REAL"] = "true"

    from lumina_quant.live.data_ws import BinanceWebSocketDataHandler

    LiveConfig.validate()

    trader = LiveTrader(
        symbol_list=LiveConfig.SYMBOLS,
        data_handler_cls=BinanceWebSocketDataHandler,  # Using WS Handler
        execution_handler_cls=LiveExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=RsiStrategy,
    )

    trader.run()
