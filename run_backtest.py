import argparse
import json
import os
import uuid
from datetime import datetime
from types import SimpleNamespace

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import BacktestConfig, BaseConfig, OptimizationConfig
from lumina_quant.market_data import load_data_dict_from_db, normalize_timeframe_token
from lumina_quant.utils.audit_store import AuditStore
from strategies import registry as strategy_registry

# ==========================================
# CONFIGURATION FROM YAML
# ==========================================
# 1. Strategy Selection
STRATEGY_MAP = strategy_registry.get_strategy_map()
requested_strategy_name = str(OptimizationConfig.STRATEGY_NAME or "").strip()
STRATEGY_CLASS = strategy_registry.resolve_strategy_class(
    requested_strategy_name,
    default_name=strategy_registry.DEFAULT_STRATEGY_NAME,
)
strategy_name = STRATEGY_CLASS.__name__


# 2. Strategy Parameters
STRATEGY_PARAMS = strategy_registry.get_default_strategy_params(strategy_name)


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
MARKET_DB_BACKEND = BaseConfig.STORAGE_BACKEND
MARKET_INFLUX_URL = BaseConfig.INFLUX_URL
MARKET_INFLUX_ORG = BaseConfig.INFLUX_ORG
MARKET_INFLUX_BUCKET = BaseConfig.INFLUX_BUCKET
MARKET_INFLUX_TOKEN_ENV = BaseConfig.INFLUX_TOKEN_ENV


def _normalize_timeframe_or_default(value, default):
    token = str(value or "").strip()
    if not token:
        return str(default)
    try:
        return normalize_timeframe_token(token)
    except Exception:
        return str(default)


BASE_TIMEFRAME = _normalize_timeframe_or_default(os.getenv("LQ_BASE_TIMEFRAME", "1s"), "1s")
AUTO_COLLECT_DB = str(os.getenv("LQ_AUTO_COLLECT_DB", "0")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _enforce_1s_base_timeframe(value: str) -> str:
    token = _normalize_timeframe_or_default(value, "1s")
    if token != "1s":
        print(
            f"[WARN] base_timeframe '{token}' overridden to '1s' for intrabar backtest execution."
        )
    return "1s"


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
    _ = auto_collect_db
    if source == "csv":
        return None

    data_dict = load_data_dict_from_db(
        market_db_path,
        exchange=market_exchange,
        symbol_list=SYMBOL_LIST,
        timeframe=str(base_timeframe),
        start_date=START_DATE,
        end_date=END_DATE,
        backend=str(MARKET_DB_BACKEND),
        influx_url=str(MARKET_INFLUX_URL),
        influx_org=str(MARKET_INFLUX_ORG),
        influx_bucket=str(MARKET_INFLUX_BUCKET),
        influx_token_env=str(MARKET_INFLUX_TOKEN_ENV),
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
            "Provide a prebuilt DB dataset or switch to --data-source csv."
        )
    return None


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _persist_backtest_audit_rows(audit_store, run_id, backtest):
    equity_rows = 0
    fill_rows = 0

    try:
        equity_curve = getattr(backtest.portfolio, "equity_curve", None)
        if equity_curve is None:
            backtest.portfolio.create_equity_curve_dataframe()
            equity_curve = getattr(backtest.portfolio, "equity_curve", None)
        if equity_curve is not None:
            for row in equity_curve.iter_rows(named=True):
                total = _safe_float(row.get("total"))
                if total is None:
                    continue
                cash = _safe_float(row.get("cash"))
                metadata = {}
                benchmark = _safe_float(row.get("benchmark_price"))
                funding = _safe_float(row.get("funding"))
                if benchmark is not None:
                    metadata["benchmark_price"] = benchmark
                if funding is not None:
                    metadata["funding"] = funding
                audit_store.log_equity(
                    run_id,
                    timeindex=row.get("datetime"),
                    total=total,
                    cash=cash,
                    metadata=metadata,
                )
                equity_rows += 1
    except Exception as exc:
        print(f"[WARN] Failed to persist equity rows to audit DB: {exc}")

    try:
        trades = list(getattr(backtest.portfolio, "trades", []) or [])
        for idx, trade in enumerate(trades):
            symbol = str(trade.get("symbol") or "").strip()
            side = str(trade.get("direction") or "").strip().upper()
            quantity = _safe_float(trade.get("quantity"))
            if not symbol or side not in {"BUY", "SELL"} or quantity is None or quantity <= 0.0:
                continue
            fill_event = SimpleNamespace(
                timeindex=trade.get("datetime"),
                symbol=symbol,
                direction=side,
                quantity=quantity,
                fill_cost=_safe_float(trade.get("fill_cost")),
                commission=_safe_float(trade.get("commission")) or 0.0,
                client_order_id=f"bt-{run_id}-{idx:06d}",
                order_id=None,
                status="FILLED",
                metadata={"source": "backtest_trade_log"},
            )
            audit_store.log_fill(run_id, fill_event)
            fill_rows += 1
    except Exception as exc:
        print(f"[WARN] Failed to persist fill rows to audit DB: {exc}")

    return {"equity_rows": equity_rows, "fill_rows": fill_rows}


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
    timeframe_token = _enforce_1s_base_timeframe(str(base_timeframe))
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
            "base_timeframe": str(timeframe_token),
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
            base_timeframe=str(timeframe_token),
            auto_collect_db=bool(auto_collect_db),
        )

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
        persisted_counts = _persist_backtest_audit_rows(audit_store, backtest_run_id, backtest)
        audit_store.end_run(
            backtest_run_id,
            status="COMPLETED",
            metadata={
                "final_equity": float(backtest.portfolio.current_holdings.get("total", 0.0)),
                **persisted_counts,
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
        base_timeframe=_normalize_timeframe_or_default(args.base_timeframe, "1s"),
        auto_collect_db=(not bool(args.no_auto_collect_db) and bool(AUTO_COLLECT_DB)),
        run_id=args.run_id,
    )
