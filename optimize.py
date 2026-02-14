import argparse
import itertools
import json
import math
import multiprocessing
import os
import sys
import uuid
from datetime import datetime

import polars as pl

# Engine Imports
from lumina_quant.backtesting.backtest import Backtest
from lumina_quant.backtesting.data import HistoricCSVDataHandler
from lumina_quant.backtesting.execution_sim import SimulatedExecutionHandler
from lumina_quant.backtesting.portfolio_backtest import Portfolio
from lumina_quant.config import BacktestConfig, BaseConfig, OptimizationConfig
from lumina_quant.market_data import load_data_dict_from_db, resolve_symbol_csv_path
from lumina_quant.optimization.storage import save_optimization_rows
from lumina_quant.optimization.walkers import build_walk_forward_splits
from lumina_quant.utils.audit_store import AuditStore

# Strategy Imports
from strategies.moving_average import MovingAverageCrossStrategy
from strategies.rsi_strategy import RsiStrategy

# Optuna Import
try:
    import optuna
    from optuna.trial import TrialState

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TrialState = None
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not found. Run 'pip install optuna'.")

# ==========================================
# CONFIGURATION FROM YAML
# ==========================================

# 1. Select Method
OPTIMIZATION_METHOD = OptimizationConfig.METHOD

# 2. Select Strategy
STRATEGY_MAP = {
    "RsiStrategy": RsiStrategy,
    "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
}
strategy_name = OptimizationConfig.STRATEGY_NAME
STRATEGY_CLASS = STRATEGY_MAP.get(strategy_name, RsiStrategy)

# 3. Data Settings
CSV_DIR = "data"
SYMBOL_LIST = BaseConfig.SYMBOLS
MARKET_DB_PATH = getattr(BaseConfig, "MARKET_DATA_SQLITE_PATH", BaseConfig.STORAGE_SQLITE_PATH)
MARKET_DB_EXCHANGE = getattr(BaseConfig, "MARKET_DATA_EXCHANGE", "binance")

# 4. Optimization Settings
GRID_PARAMS = OptimizationConfig.GRID_CONFIG.get("params", {})
OPTUNA_CONFIG = OptimizationConfig.OPTUNA_CONFIG.get("params", {})
OPTUNA_TRIALS = int(OptimizationConfig.OPTUNA_CONFIG.get("n_trials", 20))
MAX_WORKERS = int(getattr(OptimizationConfig, "MAX_WORKERS", 4))


# 5. Data Splitting Settings (WFA)
try:
    BASE_START = datetime.strptime(BacktestConfig.START_DATE, "%Y-%m-%d")

except Exception as e:
    print(f"Error parsing dates from config: {e}. Using defaults.")
    BASE_START = datetime(2023, 1, 1)

# Global Data Cache for Multiprocessing (Copy-on-Write)
DATA_DICT = {}


def load_all_data(
    csv_dir,
    symbol_list,
    *,
    data_source="auto",
    market_db_path=None,
    market_exchange="binance",
    timeframe="1m",
):
    """
    Load all data into memory once.
    Priority: DB (when selected/available) then CSV fallback.
    """
    source = str(data_source).strip().lower()
    data = {}

    if source in {"db", "auto"} and market_db_path:
        db_data = load_data_dict_from_db(
            market_db_path,
            exchange=market_exchange,
            symbol_list=symbol_list,
            timeframe=timeframe,
        )
        if db_data:
            data.update(db_data)
            print(
                f"Loaded {len(db_data)}/{len(symbol_list)} symbols from DB "
                f"{market_db_path} (exchange={market_exchange}, timeframe={timeframe})."
            )
        elif source == "db":
            print(
                f"Warning: no OHLCV rows found in DB {market_db_path} for "
                f"exchange={market_exchange}, timeframe={timeframe}."
            )

    if source == "db":
        return data

    print(f"Loading CSV fallback for missing symbols from {csv_dir}...")
    for s in symbol_list:
        if s in data:
            continue
        csv_path = resolve_symbol_csv_path(csv_dir, s)
        try:
            if os.path.exists(csv_path):
                df = pl.read_csv(csv_path, try_parse_dates=True)
                # Optimization: select only needed columns and sort once
                required_cols = ["datetime", "open", "high", "low", "close", "volume"]
                if all(c in df.columns for c in required_cols):
                    df = df.select(required_cols).sort("datetime")
                    data[s] = df
            else:
                print(f"Warning: {csv_path} not found.")
        except Exception as e:
            print(f"Error loading {s}: {e}")
    return data


def _data_datetime_range(data_dict):
    starts = []
    ends = []
    for df in data_dict.values():
        if df is None or "datetime" not in df.columns or df.height == 0:
            continue
        starts.append(df["datetime"].min())
        ends.append(df["datetime"].max())
    if not starts or not ends:
        return None, None
    # Use intersection across symbols for robust multi-asset walk-forward windows.
    return max(starts), min(ends)


def _build_data_aware_split(data_start, data_end):
    total_seconds = (data_end - data_start).total_seconds()
    if total_seconds <= 0:
        return None

    train_end = data_start + (data_end - data_start) * 0.7
    val_end = train_end + (data_end - data_start) * 0.15

    if train_end <= data_start or val_end <= train_end or data_end <= val_end:
        return None

    return {
        "fold": 1,
        "train_start": data_start,
        "train_end": train_end,
        "val_start": train_end,
        "val_end": val_end,
        "test_start": val_end,
        "test_end": data_end,
    }


def _filter_valid_splits(splits, data_start, data_end):
    valid = []
    for split in splits:
        if split["train_start"] < data_start:
            continue
        if split["test_end"] > data_end:
            continue
        valid.append(split)
    return valid


# ==========================================
# IMPLEMENTATION
# ==========================================


def _execute_backtest(
    strategy_cls, params, csv_dir, symbol_list, start_date, end_date, data_dict=None
):
    """
    Core execution logic shared by Grid and Optuna.
    Use provided data_dict if available, otherwise fallback (or use global DATA_DICT safely).
    """
    try:
        # Use passed data_dict or fall back to global DATA_DICT
        # In multiprocessing (fork), global might be accessible or passed explicitly.
        # For simplicity, we prioritize passing it if possible, or assume it's available via COW.
        current_data = data_dict if data_dict is not None else DATA_DICT

        backtest = Backtest(
            csv_dir=csv_dir,
            symbol_list=symbol_list,
            start_date=start_date,  # Specific Start
            end_date=end_date,  # Specific End
            data_handler_cls=HistoricCSVDataHandler,
            execution_handler_cls=SimulatedExecutionHandler,
            portfolio_cls=Portfolio,
            strategy_cls=strategy_cls,
            strategy_params=params,
            data_dict=current_data,
            record_history=False,
            track_metrics=True,
            record_trades=False,
        )
        backtest.simulate_trading(output=False)
        stats = backtest.portfolio.output_summary_stats_fast()
        no_data = stats.get("status") != "ok"

        sharpe = float(stats.get("sharpe", -999.0))
        if not math.isfinite(sharpe):
            sharpe = -999.0
        if no_data:
            sharpe = -999.0

        cagr_pct = float(stats.get("cagr", 0.0)) * 100.0
        mdd_pct = float(stats.get("max_drawdown", 0.0)) * 100.0

        return {
            "params": params,
            "sharpe": sharpe,
            "cagr": f"{cagr_pct:.4f}",
            "mdd": f"{mdd_pct:.4f}",
            "num_trades": int(getattr(backtest.portfolio, "trade_count", 0)),
            "no_data": no_data,
        }
    except Exception as e:
        # print(f"Backtest Error: {e}")
        return {"params": params, "error": str(e), "sharpe": -999.0}


def run_single_backtest_train(args):
    # Unpack including data_dict if we decide to pass it explictly,
    # OR rely on global DATA_DICT if using 'fork' start method (Linux/Mac).
    # Windows uses 'spawn', so globals are NOT shared. We MUST pass data or reload.
    # Passing Polars DF via pickling is fast.
    strategy_cls, params, csv_dir, symbol_list, start_date, end_date = args

    # On Windows, DATA_DICT will be empty in the child process unless initialized.
    # However, passing the entire dict in args can be heavy if huge.
    # But for reasonable datasets (< few GB), it's faster than independent I/O.
    # Wait, 'spawn' pickles the args.
    # Let's try attempting to read global DATA_DICT. If empty, reload?
    # No, that defeats the purpose.
    # Best practice for Windows MP: Pass the data in args if it fits in memory.

    # NOTE: To fix "Global variable not shared" on Windows, we need to handle it.
    # But Pool.map pickles arguments.
    # Let's modify GridOptimizer to include data_dict in args?
    # Or just rely on 'csv_dir' loading if data_dict is empty?
    # Actually, we can use 'initializer' in Pool to set the global variable in workers.

    return _execute_backtest(
        strategy_cls,
        params,
        csv_dir,
        symbol_list,
        start_date,
        end_date,
        data_dict=DATA_DICT,
    )


def pool_initializer(shared_data):
    global DATA_DICT
    DATA_DICT = shared_data


class GridSearchOptimizer:
    def __init__(self, strategy_cls, param_grid, csv_dir, symbol_list, start_date, end_date):
        self.strategy_cls = strategy_cls
        self.param_grid = param_grid
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date

    def generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def run(self, max_workers=4):
        combinations = self.generate_param_combinations()
        print(f"Starting Grid Search (Train Phase) with {len(combinations)} combinations...")

        pool_args = [
            (
                self.strategy_cls,
                params,
                self.csv_dir,
                self.symbol_list,
                self.start_date,
                self.end_date,
            )
            for params in combinations
        ]

        worker_count = max(1, int(max_workers))
        if worker_count == 1 or len(pool_args) <= 1:
            results = [run_single_backtest_train(args) for args in pool_args]
        else:
            # Use explicit spawn context for cross-platform consistency.
            ctx = multiprocessing.get_context("spawn")
            chunksize = max(1, len(pool_args) // max(1, worker_count * 4))
            with ctx.Pool(
                processes=worker_count,
                initializer=pool_initializer,
                initargs=(DATA_DICT,),
            ) as pool:
                results = pool.map(run_single_backtest_train, pool_args, chunksize)

        valid_results = [r for r in results if "error" not in r]
        sorted_results = sorted(valid_results, key=lambda x: x["sharpe"], reverse=True)
        return sorted_results


class OptunaOptimizer:
    def __init__(self, strategy_cls, optuna_config, csv_dir, symbol_list, start_date, end_date):
        self.strategy_cls = strategy_cls
        self.optuna_config = optuna_config
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date

    def objective(self, trial):
        params = {}
        for key, conf in self.optuna_config.items():
            p_type = conf.get("type")
            if p_type == "int":
                params[key] = trial.suggest_int(key, conf["low"], conf["high"])
            elif p_type == "float":
                step = conf.get("step", None)
                params[key] = trial.suggest_float(key, conf["low"], conf["high"], step=step)
            elif p_type == "categorical":
                params[key] = trial.suggest_categorical(key, conf["choices"])

        # Train on Train Set specific date range
        # Here we are in the main process (usually), so DATA_DICT is available.
        # Optuna usually runs sequentially or with its own parallel backend.
        # If running simple sequential optuna:
        result = _execute_backtest(
            self.strategy_cls,
            params,
            self.csv_dir,
            self.symbol_list,
            self.start_date,
            self.end_date,
            data_dict=DATA_DICT,
        )
        return result["sharpe"]

    def run(self, n_trials=20, n_jobs=1):
        if not OPTUNA_AVAILABLE or optuna is None or TrialState is None:
            print("Error: Optuna not installed.")
            return []

        study = optuna.create_study(direction="maximize")
        worker_count = max(1, int(n_jobs))
        study.optimize(self.objective, n_trials=n_trials, n_jobs=worker_count)

        best_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)

        params_list = []
        for t in best_trials[:10]:
            if t.state != TrialState.COMPLETE:
                continue
            params_list.append({"params": t.params, "sharpe": t.value, "cagr": "N/A", "mdd": "N/A"})
        return params_list


def _apply_fold_metadata(rows, split):
    out = []
    for row in rows:
        r = dict(row)
        r["fold"] = split["fold"]
        r["train_start"] = split["train_start"].date().isoformat()
        r["train_end"] = split["train_end"].date().isoformat()
        r["val_start"] = split["val_start"].date().isoformat()
        r["val_end"] = split["val_end"].date().isoformat()
        r["test_start"] = split["test_start"].date().isoformat()
        r["test_end"] = split["test_end"].date().isoformat()
        out.append(r)
    return out


def run_walk_forward_fold(split):
    fold = split["fold"]
    train_start = split["train_start"]
    train_end = split["train_end"]
    val_start = split["val_start"]
    val_end = split["val_end"]
    test_start = split["test_start"]
    test_end = split["test_end"]

    print(f"\n=== FOLD {fold} TRAIN [{train_start.date()} ~ {train_end.date()}] ===")
    if OPTIMIZATION_METHOD == "GRID":
        optimizer = GridSearchOptimizer(
            STRATEGY_CLASS, GRID_PARAMS, CSV_DIR, SYMBOL_LIST, train_start, train_end
        )
        train_results = optimizer.run(max_workers=MAX_WORKERS)
    elif OPTIMIZATION_METHOD == "OPTUNA":
        optimizer = OptunaOptimizer(
            STRATEGY_CLASS, OPTUNA_CONFIG, CSV_DIR, SYMBOL_LIST, train_start, train_end
        )
        train_results = optimizer.run(n_trials=OPTUNA_TRIALS, n_jobs=MAX_WORKERS)
    else:
        raise ValueError(f"Unknown optimization method: {OPTIMIZATION_METHOD}")

    if not train_results:
        return None

    best_candidate = train_results[0]
    print(
        f"[Fold {fold} Train] Top Candidate: Params={best_candidate['params']} | Sharpe={best_candidate['sharpe']:.4f}"
    )

    print(f"=== FOLD {fold} VALIDATION [{val_start.date()} ~ {val_end.date()}] ===")
    val_candidates = []
    limit = min(3, len(train_results))
    for cand in train_results[:limit]:
        res = _execute_backtest(
            STRATEGY_CLASS,
            cand["params"],
            CSV_DIR,
            SYMBOL_LIST,
            val_start,
            val_end,
            data_dict=DATA_DICT,
        )
        res["train_sharpe"] = cand["sharpe"]
        divergence = abs(float(cand["sharpe"]) - float(res["sharpe"]))
        penalty_factor = OptimizationConfig.OVERFIT_PENALTY
        res["robustness_score"] = float(res["sharpe"]) - (divergence * penalty_factor)
        if res.get("no_data"):
            res["robustness_score"] = -999.0
        val_candidates.append(res)
        print(
            f"[Fold {fold} Val] Params={cand['params']} -> Val Sharpe={res['sharpe']:.4f} (Train={cand['sharpe']:.4f})"
        )

    val_candidates.sort(key=lambda x: x["robustness_score"], reverse=True)
    final_best = val_candidates[0]
    print(f"[Fold {fold} Validation] Selected Params: {final_best['params']}")

    print(f"=== FOLD {fold} TEST [{test_start.date()} ~ {test_end.date()}] ===")
    test_res = _execute_backtest(
        STRATEGY_CLASS,
        final_best["params"],
        CSV_DIR,
        SYMBOL_LIST,
        test_start,
        test_end,
        data_dict=DATA_DICT,
    )
    print(
        f"[Fold {fold} Test] Sharpe={test_res['sharpe']:.4f} | CAGR={test_res['cagr']} | MaxDD={test_res['mdd']}"
    )

    return {
        "fold": fold,
        "split": split,
        "train_results": _apply_fold_metadata(train_results, split),
        "best_candidate": dict(best_candidate),
        "val_candidates": _apply_fold_metadata(val_candidates, split),
        "selected": dict(final_best),
        "test_result": _apply_fold_metadata([test_res], split)[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LuminaQuant walk-forward optimization.")
    parser.add_argument(
        "--folds",
        type=int,
        default=OptimizationConfig.WALK_FORWARD_FOLDS,
        help="Number of walk-forward folds.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=OPTUNA_TRIALS,
        help="Optuna trial count per fold when OPTUNA is selected.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Worker process count for grid search.",
    )
    parser.add_argument(
        "--save-best-params",
        action="store_true",
        help="Write winning params into best_optimized_parameters/<strategy>/best_params.json.",
    )
    parser.add_argument(
        "--data-source",
        choices=["auto", "csv", "db"],
        default="auto",
        help="Market data source for optimization (auto: DB then CSV fallback).",
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

    run_id = str(args.run_id or "").strip() or str(uuid.uuid4())
    db_path = getattr(BaseConfig, "STORAGE_SQLITE_PATH", "logs/lumina_quant.db")
    audit_store = AuditStore(db_path)
    audit_store.start_run(
        mode="optimize",
        metadata={
            "strategy": STRATEGY_CLASS.__name__,
            "method": str(OPTIMIZATION_METHOD),
            "folds": int(args.folds),
            "n_trials": int(args.n_trials),
            "max_workers": int(args.max_workers),
            "data_source": str(args.data_source),
            "market_db_path": str(args.market_db_path),
            "market_exchange": str(args.market_exchange),
        },
        run_id=run_id,
    )

    final_status = "FAILED"
    final_metadata = {}

    OPTUNA_TRIALS = args.n_trials
    MAX_WORKERS = max(1, int(args.max_workers))
    persist_best_params = bool(args.save_best_params or OptimizationConfig.PERSIST_BEST_PARAMS)

    try:
        # Load Data Once
        DATA_DICT = load_all_data(
            CSV_DIR,
            SYMBOL_LIST,
            data_source=args.data_source,
            market_db_path=args.market_db_path,
            market_exchange=args.market_exchange,
            timeframe=BaseConfig.TIMEFRAME,
        )
        splits = build_walk_forward_splits(
            BASE_START,
            args.folds,
        )
        data_start, data_end = _data_datetime_range(DATA_DICT)
        if data_start is None or data_end is None:
            print("No usable datetime range found in loaded data.")
            sys.exit(1)

        valid_splits = _filter_valid_splits(splits, data_start, data_end)
        if not valid_splits:
            fallback_split = _build_data_aware_split(data_start, data_end)
            if fallback_split is None:
                print(
                    "Could not build a valid walk-forward split from current data range. "
                    "Check your dataset coverage and config dates."
                )
                sys.exit(1)
            valid_splits = [fallback_split]
            print(
                "[INFO] Default walk-forward windows were outside available data range. "
                "Using one data-aware fallback split."
            )

        if not valid_splits:
            print("No walk-forward splits generated.")
            sys.exit(1)

        fold_reports = []
        for split in valid_splits:
            try:
                report = run_walk_forward_fold(split)
                if report is None:
                    print(f"[Fold {split['fold']}] No valid optimization results.")
                    continue
                fold_reports.append(report)
                save_optimization_rows(
                    db_path,
                    run_id,
                    f"fold_{split['fold']}_train",
                    report["train_results"],
                )
                save_optimization_rows(
                    db_path,
                    run_id,
                    f"fold_{split['fold']}_validation",
                    report["val_candidates"],
                )
                save_optimization_rows(
                    db_path,
                    run_id,
                    f"fold_{split['fold']}_test",
                    [report["test_result"]],
                )
            except Exception as e:
                print(f"[Fold {split['fold']}] Failed: {e}")

        if not fold_reports:
            print("No valid fold report generated.")
            sys.exit(1)

        # Select overall winner by test sharpe, then robustness score.
        fold_reports.sort(
            key=lambda r: (
                0 if r["test_result"].get("no_data") else 1,
                float(r["test_result"].get("sharpe", -999.0)),
                float(r["selected"].get("robustness_score", -999.0)),
                int(r["test_result"].get("num_trades", 0)),
            ),
            reverse=True,
        )
        winner = fold_reports[0]

        print("\n>>>> FINAL WALK-FORWARD REPORT <<<<")
        for report in fold_reports:
            f = report["fold"]
            print(
                f"Fold {f}: Train={report['best_candidate']['sharpe']:.4f} | "
                f"Val={report['selected']['sharpe']:.4f} | "
                f"Test={report['test_result']['sharpe']:.4f}"
            )

        print(
            f"\nSelected Fold: {winner['fold']} | Params: {winner['selected']['params']} | "
            f"Test Sharpe: {winner['test_result']['sharpe']:.4f}"
        )

        if persist_best_params:
            # Save best parameters from winning fold (opt-in artifact).
            strategy_name = STRATEGY_CLASS.__name__
            save_dir = os.path.join("best_optimized_parameters", strategy_name)
            os.makedirs(save_dir, exist_ok=True)
            best_params_file = os.path.join(save_dir, "best_params.json")
            with open(best_params_file, "w") as f:
                json.dump(winner["selected"]["params"], f, indent=4)
            print(f"[Artifact] Best Parameters saved to '{best_params_file}'")
        else:
            print("[Artifact] best_params.json export skipped (pure-compute mode).")

        final_status = "COMPLETED"
        final_metadata = {
            "selected_fold": int(winner["fold"]),
            "selected_params": winner["selected"].get("params", {}),
            "selected_test_sharpe": float(winner["test_result"].get("sharpe", -999.0)),
        }
    except SystemExit as exc:
        code = int(exc.code or 0)
        final_status = "COMPLETED" if code == 0 else "FAILED"
        final_metadata = {"exit_code": code}
        raise
    except Exception as exc:
        final_status = "FAILED"
        final_metadata = {"error": str(exc)}
        raise
    finally:
        audit_store.end_run(run_id, status=final_status, metadata=final_metadata)
        audit_store.close()
