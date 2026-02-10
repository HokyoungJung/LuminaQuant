from lumina_quant.live_trader import LiveTrader
from lumina_quant.live_data_ws import BinanceWebSocketDataHandler
from lumina_quant.live_execution import LiveExecutionHandler
from lumina_quant.portfolio import Portfolio
from lumina_quant.config import LiveConfig
from strategies.rsi_strategy import RsiStrategy

if __name__ == "__main__":
    LiveConfig.validate()

    trader = LiveTrader(
        symbol_list=LiveConfig.SYMBOLS,
        data_handler_cls=BinanceWebSocketDataHandler,  # Using WS Handler
        execution_handler_cls=LiveExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=RsiStrategy,
    )

    trader.run()
