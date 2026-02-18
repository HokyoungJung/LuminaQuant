import argparse
import json
import os
import uuid
from datetime import datetime

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import BacktestConfig, BaseConfig, LiveConfig, OptimizationConfig
from lumina_quant.data_collector import auto_collect_market_data
from lumina_quant.market_data import load_data_dict_from_db
from lumina_quant.utils.audit_store import AuditStore
from strategies import (
    DEFAULT_STRATEGY_NAME,
    get_default_strategy_params,
    get_strategy_map,
    resolve_strategy_class,
)

# ==========================================
# CONFIGURATION FROM YAML
# ==========================================
# 1. Strategy Selection
STRATEGY_MAP = get_strategy_map()
requested_strategy_name = str(OptimizationConfig.STRATEGY_NAME or "").strip()
STRATEGY_CLASS = resolve_strategy_class(requested_strategy_name, default_name=DEFAULT_STRATEGY_NAME)
strategy_name = STRATEGY_CLASS.__name__


# 2. Strategy Parameters
STRATEGY_PARAMS = get_default_strategy_params(strategy_name)


# Try loading optimized
param_path = os.path.join("best_optimized_parameters", strategy_name, "best_params.json")
meta_path = os.path.join("best_optimized_parameters", strategy_name, "best_params.meta.json")

if os.path.exists(param_path):
    try:
        with open(param_path) as f:
            loaded_params = json.load(f)
        print(f"[OK] Loaded Optimized Params from {param_path}")
        STRATEGY_PARAMS = loaded_params
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as meta_file:
                    meta = json.load(meta_file)
                selection_basis = str(meta.get("selection_basis", "")).strip().lower()
                if selection_basis == "validation_only":
                    print(
                        "[INFO] Parameter provenance: validation-only selection with locked OOS holdout."
                    )
                else:
                    print(
                        "[WARN] Parameter provenance metadata exists but selection basis is not "
                        "'validation_only'."
                    )
            except Exception as meta_err:
                print(f"[WARN] Failed to parse params metadata: {meta_err}")
        else:
            print(
                "[WARN] No parameter provenance metadata file found. "
                "Consider re-running optimize.py with strict OOS settings."
            )
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

MARKET_DB_PATH = BaseConfig.MARKET_DATA_SQLITE_PATH
MARKET_DB_EXCHANGE = BaseConfig.MARKET_DATA_EXCHANGE
BASE_TIMEFRAME = str(os.getenv("LQ_BASE_TIMEFRAME", "1s") or "1s").strip().lower()
AUTO_COLLECT_DB = str(os.getenv("LQ_AUTO_COLLECT_DB", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# ==========================================
# EXECUTION (Do not modify generally)
# ==========================================


def _load_data_dict(
    data_source,
    market_db_path,
    market_exchange,
    *,
    base_timeframe,
    auto_collect_db=True,
):
    source = str(data_source).strip().lower()
    if source == "csv":
        return None

    if source in {"auto", "db"} and auto_collect_db:
        try:
            sync_rows = auto_collect_market_data(
                symbol_list=list(SYMBOL_LIST),
                timeframe=str(base_timeframe),
                db_path=str(market_db_path),
                exchange_id=str(market_exchange),
                market_type=str(LiveConfig.MARKET_TYPE),
                since_dt=START_DATE,
                until_dt=END_DATE,
                api_key=str(LiveConfig.BINANCE_API_KEY or ""),
                secret_key=str(LiveConfig.BINANCE_SECRET_KEY or ""),
                testnet=bool(LiveConfig.IS_TESTNET),
                limit=1000,
                max_batches=100000,
                retries=3,
                base_wait_sec=0.5,
            )

            def _safe_int(value):
                try:
                    return int(value)
                except Exception:
                    return 0

            upserted = sum(_safe_int(item.get("upserted_rows", 0)) for item in sync_rows)
            fetched = sum(_safe_int(item.get("fetched_rows", 0)) for item in sync_rows)
            print(
                f"[INFO] Auto collector checked DB coverage for {len(sync_rows)} symbols "
                f"(fetched={fetched}, upserted={upserted})."
            )
        except Exception as exc:
            if source == "db":
                raise RuntimeError(f"DB auto-collect failed: {exc}") from exc
            print(f"[WARN] DB auto-collect failed; continuing with fallback behavior: {exc}")

    data_dict = load_data_dict_from_db(
        market_db_path,
        exchange=market_exchange,
        symbol_list=SYMBOL_LIST,
        timeframe=str(base_timeframe),
        start_date=START_DATE,
        end_date=END_DATE,
    )
    if data_dict:
        missing = [symbol for symbol in SYMBOL_LIST if symbol not in data_dict]
        print(
            f"[INFO] Loaded {len(data_dict)}/{len(SYMBOL_LIST)} symbols from DB "
            f"{market_db_path} (exchange={market_exchange}, timeframe={base_timeframe})."
        )
        if missing:
            print(f"[WARN] Symbols still missing in DB after load: {missing}")
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
    base_timeframe=BASE_TIMEFRAME,
    auto_collect_db=AUTO_COLLECT_DB,
    run_id="",
):
    print("------------------------------------------------")
    print(f"Running Backtest for {SYMBOL_LIST}")
    print(f"Strategy: {STRATEGY_CLASS.__name__}")
    print(f"Params: {STRATEGY_PARAMS}")
    print("------------------------------------------------")

    backtest_run_id = str(run_id or "").strip() or str(uuid.uuid4())
    audit_store = AuditStore(BaseConfig.STORAGE_SQLITE_PATH)
    audit_store.start_run(
        mode="backtest",
        metadata={
            "symbols": list(SYMBOL_LIST),
            "strategy": STRATEGY_CLASS.__name__,
            "params": STRATEGY_PARAMS,
            "data_source": str(data_source),
            "market_db_path": str(market_db_path),
            "market_exchange": str(market_exchange),
            "base_timeframe": str(base_timeframe),
            "strategy_timeframe": str(BaseConfig.TIMEFRAME),
            "auto_collect_db": bool(auto_collect_db),
        },
        run_id=backtest_run_id,
    )

    try:
        data_dict = _load_data_dict(
            data_source,
            market_db_path,
            market_exchange,
            base_timeframe=str(base_timeframe),
            auto_collect_db=bool(auto_collect_db),
        )

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
            strategy_timeframe=str(BaseConfig.TIMEFRAME),
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
        "--base-timeframe",
        default=BASE_TIMEFRAME,
        help="Collection/backtest source timeframe. Use the minimum resolution (recommended: 1s).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional external run_id for audit trail correlation.",
    )
    parser.add_argument(
        "--no-auto-collect-db",
        action="store_true",
        help="Disable automatic DB market-data collection before loading.",
    )
    args = parser.parse_args()
    run(
        data_source=args.data_source,
        market_db_path=args.market_db_path,
        market_exchange=args.market_exchange,
        base_timeframe=args.base_timeframe,
        auto_collect_db=(not bool(args.no_auto_collect_db) and bool(AUTO_COLLECT_DB)),
        run_id=args.run_id,
    )
