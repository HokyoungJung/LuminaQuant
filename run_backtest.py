import argparse
import json
import os
import uuid
from datetime import datetime
from types import SimpleNamespace

from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.chunked_runner import run_backtest_chunked
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import BacktestConfig, BaseConfig, LiveConfig, OptimizationConfig
from lumina_quant.market_data import load_data_dict_from_db, normalize_timeframe_token
from lumina_quant.parquet_market_data import (
    is_parquet_market_data_store,
    load_data_dict_from_parquet,
)
from lumina_quant.utils.audit_store import AuditStore

try:
    from lumina_quant.data_collector import auto_collect_market_data
except Exception:
    def auto_collect_market_data(**_kwargs):  # type: ignore[no-redef]
        return []


try:
    from strategies import registry as strategy_registry
except Exception:
    class _UnavailableStrategy:
        def __init__(self, *_args, **_kwargs):
            pass

        def calculate_signals(self, _event):
            return None

    class _PublicStrategyRegistryFallback:
        DEFAULT_STRATEGY_NAME = "UnavailableStrategy"

        @staticmethod
        def get_strategy_map():
            return {"UnavailableStrategy": _UnavailableStrategy}

        @staticmethod
        def resolve_strategy_class(_requested_name, default_name=None):
            _ = default_name
            return _UnavailableStrategy

        @staticmethod
        def get_default_strategy_params(_strategy_name):
            return {}

        @staticmethod
        def resolve_strategy_params(_strategy_name, params):
            return dict(params or {})

    strategy_registry = _PublicStrategyRegistryFallback()

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

STRATEGY_PARAMS = strategy_registry.resolve_strategy_params(strategy_name, STRATEGY_PARAMS)


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

MARKET_DB_PATH = BaseConfig.MARKET_DATA_PARQUET_PATH
MARKET_DB_EXCHANGE = BaseConfig.MARKET_DATA_EXCHANGE
MARKET_DB_BACKEND = BaseConfig.STORAGE_BACKEND


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


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_optional_bool(name):
    raw = os.getenv(name)
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None
    return token.lower() not in {"0", "false", "no", "off"}


def _env_int(name, default):
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


BT_CHUNK_DAYS = max(1, _env_int("LQ_BT_CHUNK_DAYS", 7))
BT_CHUNK_WARMUP_BARS = max(0, _env_int("LQ_BT_CHUNK_WARMUP_BARS", 0))


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
    if source == "csv":
        return None

    use_parquet = is_parquet_market_data_store(
        str(market_db_path),
        backend=str(MARKET_DB_BACKEND),
    )

    if source in {"auto", "db"} and auto_collect_db and not use_parquet:
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
    elif source in {"auto", "db"} and auto_collect_db and use_parquet:
        print("[INFO] Auto collector skipped for parquet market-data backend.")

    if use_parquet:
        data_dict = load_data_dict_from_parquet(
            str(market_db_path),
            exchange=str(market_exchange),
            symbol_list=list(SYMBOL_LIST),
            timeframe=str(base_timeframe),
            start_date=START_DATE,
            end_date=END_DATE,
            chunk_days=BT_CHUNK_DAYS,
            warmup_bars=BT_CHUNK_WARMUP_BARS,
        )
    else:
        data_dict = load_data_dict_from_db(
            market_db_path,
            exchange=market_exchange,
            symbol_list=SYMBOL_LIST,
            timeframe=str(base_timeframe),
            start_date=START_DATE,
            end_date=END_DATE,
            backend=str(MARKET_DB_BACKEND),
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


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_float_or(value, default):
    parsed = _safe_float(value)
    if parsed is None:
        return float(default)
    return float(parsed)


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


def _resolve_execution_profile(*, low_memory=None, persist_output=None):
    resolved_low_memory = (
        _env_bool("LQ_BACKTEST_LOW_MEMORY", False) if low_memory is None else bool(low_memory)
    )
    resolved_persist_output = persist_output
    if resolved_persist_output is None:
        resolved_persist_output = _env_optional_bool("LQ_BACKTEST_PERSIST_OUTPUT")
    if resolved_persist_output is None:
        resolved_persist_output = (
            False
            if resolved_low_memory
            else bool(getattr(BacktestConfig, "PERSIST_OUTPUT", True))
        )
    return {
        "low_memory": bool(resolved_low_memory),
        "record_history": not bool(resolved_low_memory),
        "track_metrics": True,
        "record_trades": not bool(resolved_low_memory),
        "persist_output": bool(resolved_persist_output),
    }


def _print_low_memory_stats(backtest):
    fast_stats = {}
    try:
        fast_stats = dict(backtest.portfolio.output_summary_stats_fast() or {})
    except Exception as exc:
        fast_stats = {"status": f"error: {exc}"}

    final_equity = _safe_float(getattr(backtest.portfolio, "current_holdings", {}).get("total"))
    trade_count = int(getattr(backtest.portfolio, "trade_count", 0))
    print("[INFO] Low-memory mode enabled (history/trade logs disabled).")
    print(
        "[INFO] Backtest summary: "
        f"final_equity={final_equity if final_equity is not None else 0.0:.4f}, "
        f"trade_count={trade_count}, "
        f"sharpe={_safe_float_or(fast_stats.get('sharpe'), 0.0):.4f}, "
        f"cagr={_safe_float_or(fast_stats.get('cagr'), 0.0):.6f}, "
        f"max_drawdown={_safe_float_or(fast_stats.get('max_drawdown'), 0.0):.6f}"
    )
    return fast_stats


def _persist_low_memory_outputs(backtest, persist_output):
    if not bool(persist_output):
        return
    backtest.portfolio.create_equity_curve_dataframe()
    backtest.portfolio.output_trade_log(os.path.join("data", "trades.csv"))
    backtest.portfolio.save_equity_curve(os.path.join("data", "equity.csv"))


def run(
    data_source="auto",
    market_db_path=MARKET_DB_PATH,
    market_exchange=MARKET_DB_EXCHANGE,
    base_timeframe=BASE_TIMEFRAME,
    auto_collect_db=AUTO_COLLECT_DB,
    run_id="",
    low_memory=None,
    persist_output=None,
):
    print("------------------------------------------------")
    print(f"Running Backtest for {SYMBOL_LIST}")
    print(f"Strategy: {STRATEGY_CLASS.__name__}")
    print(f"Params: {STRATEGY_PARAMS}")
    print("------------------------------------------------")

    backtest_run_id = str(run_id or "").strip() or str(uuid.uuid4())
    timeframe_token = _enforce_1s_base_timeframe(str(base_timeframe))
    audit_store = AuditStore(BaseConfig.POSTGRES_DSN)
    execution_profile = _resolve_execution_profile(
        low_memory=low_memory,
        persist_output=persist_output,
    )
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
            **execution_profile,
        },
        run_id=backtest_run_id,
    )

    try:
        use_parquet = is_parquet_market_data_store(
            str(market_db_path),
            backend=str(MARKET_DB_BACKEND),
        )
        source_token = str(data_source).strip().lower()
        use_chunked_runner = bool(use_parquet and source_token in {"auto", "db"})
        data_dict = None

        if not use_chunked_runner:
            data_dict = _load_data_dict(
                data_source,
                market_db_path,
                market_exchange,
                base_timeframe=str(timeframe_token),
                auto_collect_db=bool(auto_collect_db),
            )

        if use_chunked_runner:
            if END_DATE is None:
                raise RuntimeError(
                    "Chunked backtest requires an explicit END_DATE when using parquet backend."
                )

            def _chunk_loader(chunk_start, chunk_end):
                return load_data_dict_from_parquet(
                    str(market_db_path),
                    exchange=str(market_exchange),
                    symbol_list=list(SYMBOL_LIST),
                    timeframe=str(timeframe_token),
                    start_date=chunk_start,
                    end_date=chunk_end,
                    chunk_days=max(1, int(BT_CHUNK_DAYS)),
                    warmup_bars=0,
                )

            backtest = run_backtest_chunked(
                csv_dir=CSV_DIR,
                symbol_list=list(SYMBOL_LIST),
                start_date=START_DATE,
                end_date=END_DATE,
                strategy_cls=STRATEGY_CLASS,
                strategy_params=STRATEGY_PARAMS,
                data_loader=_chunk_loader,
                chunk_days=max(1, int(BT_CHUNK_DAYS)),
                strategy_timeframe=str(BaseConfig.TIMEFRAME),
                data_handler_cls=HistoricCSVDataHandler,
                execution_handler_cls=SimulatedExecutionHandler,
                portfolio_cls=Portfolio,
                record_history=bool(execution_profile["record_history"]),
                track_metrics=bool(execution_profile["track_metrics"]),
                record_trades=bool(execution_profile["record_trades"]),
            )
            if bool(execution_profile["low_memory"]):
                _persist_low_memory_outputs(backtest, execution_profile["persist_output"])
                _print_low_memory_stats(backtest)
            else:
                backtest._output_performance(
                    persist_output=bool(execution_profile["persist_output"]),
                    verbose=True,
                )
        else:
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
                record_history=bool(execution_profile["record_history"]),
                track_metrics=bool(execution_profile["track_metrics"]),
                record_trades=bool(execution_profile["record_trades"]),
                strategy_timeframe=str(BaseConfig.TIMEFRAME),
            )
            if bool(execution_profile["low_memory"]):
                backtest.simulate_trading(output=False)
                _persist_low_memory_outputs(backtest, execution_profile["persist_output"])
                _print_low_memory_stats(backtest)
            else:
                backtest.simulate_trading(
                    persist_output=bool(execution_profile["persist_output"]),
                )

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
        help="Market data parquet root path.",
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
    low_memory_group = parser.add_mutually_exclusive_group()
    low_memory_group.add_argument(
        "--low-memory",
        dest="low_memory",
        action="store_true",
        help=(
            "Use low-memory execution profile (record_history=False, record_trades=False, "
            "track_metrics=True)."
        ),
    )
    low_memory_group.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Explicitly disable low-memory profile even if LQ_BACKTEST_LOW_MEMORY is set.",
    )
    parser.set_defaults(low_memory=None)
    persist_group = parser.add_mutually_exclusive_group()
    persist_group.add_argument(
        "--persist-output",
        dest="persist_output",
        action="store_true",
        help="Force writing CSV outputs (equity/trades).",
    )
    persist_group.add_argument(
        "--no-persist-output",
        dest="persist_output",
        action="store_false",
        help="Force disabling CSV outputs (equity/trades).",
    )
    parser.set_defaults(persist_output=None)
    args = parser.parse_args()
    run(
        data_source=args.data_source,
        market_db_path=args.market_db_path,
        market_exchange=args.market_exchange,
        base_timeframe=_normalize_timeframe_or_default(args.base_timeframe, "1s"),
        auto_collect_db=(not bool(args.no_auto_collect_db) and bool(AUTO_COLLECT_DB)),
        run_id=args.run_id,
        low_memory=args.low_memory,
        persist_output=args.persist_output,
    )
