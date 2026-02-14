import argparse
import os

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import LiveConfig
from lumina_quant.live.execution_live import LiveExecutionHandler
from lumina_quant.live.trader import LiveTrader
from strategies.moving_average import MovingAverageCrossStrategy
from strategies.rsi_strategy import RsiStrategy

STRATEGY_MAP = {
    "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
    "RsiStrategy": RsiStrategy,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LuminaQuant live trader (WebSocket).")
    parser.add_argument(
        "--enable-live-real",
        action="store_true",
        help="Explicitly allow real trading mode.",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_MAP.keys()),
        default="RsiStrategy",
        help="Strategy class to run in live mode.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional external run_id for audit trail correlation.",
    )
    parser.add_argument(
        "--stop-file",
        default="",
        help="Optional stop-file path for graceful shutdown signal.",
    )
    args = parser.parse_args()
    if args.enable_live_real:
        os.environ["LUMINA_ENABLE_LIVE_REAL"] = "true"

    from lumina_quant.live.data_ws import BinanceWebSocketDataHandler

    LiveConfig.validate()
    strategy_cls = STRATEGY_MAP.get(args.strategy, RsiStrategy)

    trader = LiveTrader(
        symbol_list=LiveConfig.SYMBOLS,
        data_handler_cls=BinanceWebSocketDataHandler,  # Using WS Handler
        execution_handler_cls=LiveExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=strategy_cls,
        strategy_name=args.strategy,
        stop_file=args.stop_file,
        external_run_id=args.run_id,
    )

    trader.run()
