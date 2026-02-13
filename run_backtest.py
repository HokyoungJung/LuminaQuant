import os
import json
from datetime import datetime
from lumina_quant.backtest import Backtest
from lumina_quant.data import HistoricCSVDataHandler
from lumina_quant.execution import SimulatedExecutionHandler
from lumina_quant.portfolio import Portfolio
from lumina_quant.config import BacktestConfig, BaseConfig, OptimizationConfig


# ==========================================
# CONFIGURATION FROM YAML
# ==========================================

# 1. Strategy Selection
from strategies.moving_average import MovingAverageCrossStrategy
from strategies.rsi_strategy import RsiStrategy

# Map string name to class
STRATEGY_MAP = {
    "RsiStrategy": RsiStrategy,
    "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
}
strategy_name = OptimizationConfig.STRATEGY_NAME
STRATEGY_CLASS = STRATEGY_MAP.get(strategy_name, RsiStrategy)


# 2. Strategy Parameters
STRATEGY_PARAMS = {}
# Provide some defaults if nothing else works
if strategy_name == "RsiStrategy":
    STRATEGY_PARAMS = {"rsi_period": 14, "oversold": 30, "overbought": 70}
elif strategy_name == "MovingAverageCrossStrategy":
    STRATEGY_PARAMS = {"short_window": 10, "long_window": 30}


# Try loading optimized
param_path = os.path.join(
    "best_optimized_parameters", strategy_name, "best_params.json"
)

if os.path.exists(param_path):
    try:
        with open(param_path, "r") as f:
            loaded_params = json.load(f)
        print(f"[OK] Loaded Optimized Params from {param_path}")
        STRATEGY_PARAMS = loaded_params
    except Exception as e:
        print(f"[WARN] Failed to load optimized params: {e}")
else:
    print(f"[INFO] Optimized params not found at {param_path}. Using Defaults.")


# 3. Data Settings
CSV_DIR = "data"
SYMBOL_LIST = BaseConfig.SYMBOLS

# 4. Dates
try:
    START_DATE = datetime.strptime(BacktestConfig.START_DATE, "%Y-%m-%d")
except Exception:
    START_DATE = datetime(2024, 1, 1)

# ==========================================
# EXECUTION (Do not modify generally)
# ==========================================


def run():
    print("------------------------------------------------")
    print(f"Running Backtest for {SYMBOL_LIST}")
    print(f"Strategy: {STRATEGY_CLASS.__name__}")
    print(f"Params: {STRATEGY_PARAMS}")
    print("------------------------------------------------")

    # Initialize Backtest
    backtest = Backtest(
        csv_dir=CSV_DIR,
        symbol_list=SYMBOL_LIST,
        start_date=START_DATE,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=STRATEGY_CLASS,
        strategy_params=STRATEGY_PARAMS,
    )

    backtest.simulate_trading()


if __name__ == "__main__":
    run()
