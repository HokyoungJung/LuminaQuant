import argparse
import os

from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import LiveConfig
from lumina_quant.live.data_poll import LiveDataHandler
from lumina_quant.live.execution_live import LiveExecutionHandler
from lumina_quant.live.trader import LiveTrader
from strategies.moving_average import MovingAverageCrossStrategy


def main():
    parser = argparse.ArgumentParser(description="Run LuminaQuant live trader.")
    parser.add_argument(
        "--enable-live-real",
        action="store_true",
        help="Explicitly allow real trading mode.",
    )
    args = parser.parse_args()
    if args.enable_live_real:
        os.environ["LUMINA_ENABLE_LIVE_REAL"] = "true"

    # 1. Check Configuration
    print("=== Quants Agent Live Trader ===")
    LiveConfig.validate()
    print(f"Mode: {'TESTNET/PAPER' if LiveConfig.IS_TESTNET else 'REAL TRADING'}")
    print(f"Exchange: {LiveConfig.EXCHANGE}")

    # Note: Exchange-specific keys might need check based on config.EXCHANGE_ID
    # For now, we assume binance-like environment vars if usage is binance.

    # 2. Setup
    symbol_list = LiveConfig.SYMBOLS  # e.g. ['BTC/USDT'] from config.yaml
    print(f"Trading Symbols: {symbol_list}")

    # 3. Initialize Trader
    try:
        trader = LiveTrader(
            symbol_list=symbol_list,
            data_handler_cls=LiveDataHandler,
            execution_handler_cls=LiveExecutionHandler,
            portfolio_cls=Portfolio,
            strategy_cls=MovingAverageCrossStrategy,
        )

        # 4. Run
        print("Starting engine... Press Ctrl+C to stop.")
        trader.run()

    except KeyboardInterrupt:
        print("\nStopping trader...")
    except Exception as e:
        print(f"\nCritical Error: {e}")


if __name__ == "__main__":
    main()
