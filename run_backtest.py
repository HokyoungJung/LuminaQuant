import argparse
import json
import os
import uuid
from datetime import datetime

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import BacktestConfig, BaseConfig, OptimizationConfig
from lumina_quant.market_data import load_data_dict_from_db
from lumina_quant.utils.audit_store import AuditStore

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
param_path = os.path.join("best_optimized_parameters", strategy_name, "best_params.json")

if os.path.exists(param_path):
    try:
        with open(param_path) as f:
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

try:
    END_DATE = (
        datetime.strptime(BacktestConfig.END_DATE, "%Y-%m-%d") if BacktestConfig.END_DATE else None
    )
except Exception:
    END_DATE = None

MARKET_DB_PATH = getattr(BaseConfig, "MARKET_DATA_SQLITE_PATH", BaseConfig.STORAGE_SQLITE_PATH)
MARKET_DB_EXCHANGE = getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance")

# ==========================================
# EXECUTION (Do not modify generally)
# ==========================================


def _load_data_dict(data_source, market_db_path, market_exchange):
    source = str(data_source).strip().lower()
    if source == "csv":
        return None

    data_dict = load_data_dict_from_db(
        market_db_path,
        exchange=market_exchange,
        symbol_list=SYMBOL_LIST,
        timeframe=BaseConfig.TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    if data_dict:
        print(
            f"[INFO] Loaded {len(data_dict)}/{len(SYMBOL_LIST)} symbols from DB "
            f"{market_db_path} (exchange={market_exchange}, timeframe={BaseConfig.TIMEFRAME})."
        )
        return data_dict
    if source == "db":
        raise RuntimeError(
            "No market data found in DB for requested symbols/timeframe. "
            "Run scripts/sync_binance_ohlcv.py first or switch to --data-source csv."
        )
    return None


def run(
    data_source="auto",
    market_db_path=MARKET_DB_PATH,
    market_exchange=MARKET_DB_EXCHANGE,
    run_id="",
):
    print("------------------------------------------------")
    print(f"Running Backtest for {SYMBOL_LIST}")
    print(f"Strategy: {STRATEGY_CLASS.__name__}")
    print(f"Params: {STRATEGY_PARAMS}")
    print("------------------------------------------------")

    backtest_run_id = str(run_id or "").strip() or str(uuid.uuid4())
    audit_store = AuditStore(getattr(BaseConfig, "STORAGE_SQLITE_PATH", "logs/lumina_quant.db"))
    audit_store.start_run(
        mode="backtest",
        metadata={
            "symbols": list(SYMBOL_LIST),
            "strategy": STRATEGY_CLASS.__name__,
            "params": STRATEGY_PARAMS,
            "data_source": str(data_source),
            "market_db_path": str(market_db_path),
            "market_exchange": str(market_exchange),
        },
        run_id=backtest_run_id,
    )

    try:
        data_dict = _load_data_dict(data_source, market_db_path, market_exchange)

        # Initialize Backtest
        backtest = Backtest(
            csv_dir=CSV_DIR,
            symbol_list=SYMBOL_LIST,
            start_date=START_DATE,
            end_date=END_DATE,
            data_handler_cls=HistoricCSVDataHandler,
            execution_handler_cls=SimulatedExecutionHandler,
            portfolio_cls=Portfolio,
            strategy_cls=STRATEGY_CLASS,
            strategy_params=STRATEGY_PARAMS,
            data_dict=data_dict,
        )

        backtest.simulate_trading()
        audit_store.end_run(
            backtest_run_id,
            status="COMPLETED",
            metadata={
                "final_equity": float(backtest.portfolio.current_holdings.get("total", 0.0)),
            },
        )
    except Exception as exc:
        audit_store.end_run(
            backtest_run_id,
            status="FAILED",
            metadata={"error": str(exc)},
        )
        raise
    finally:
        audit_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LuminaQuant backtest.")
    parser.add_argument(
        "--data-source",
        choices=["auto", "csv", "db"],
        default="auto",
        help="Market data source (auto: DB first then CSV fallback).",
    )
    parser.add_argument(
        "--market-db-path",
        default=MARKET_DB_PATH,
        help="SQLite path for market OHLCV data.",
    )
    parser.add_argument(
        "--market-exchange",
        default=MARKET_DB_EXCHANGE,
        help="Exchange key used in OHLCV DB rows.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional external run_id for audit trail correlation.",
    )
    args = parser.parse_args()
    run(
        data_source=args.data_source,
        market_db_path=args.market_db_path,
        market_exchange=args.market_exchange,
        run_id=args.run_id,
    )
