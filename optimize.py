import itertools
import multiprocessing
import sys
import os
import json
import uuid
import sqlite3
import polars as pl
from datetime import datetime, timezone

# Engine Imports
from lumina_quant.backtest import Backtest
from lumina_quant.data import HistoricCSVDataHandler
from lumina_quant.execution import SimulatedExecutionHandler
from lumina_quant.portfolio import Portfolio
from lumina_quant.config import BacktestConfig, OptimizationConfig, BaseConfig

# Strategy Imports
from strategies.moving_average import MovingAverageCrossStrategy
from strategies.rsi_strategy import RsiStrategy

# Optuna Import
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
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

# 4. Optimization Settings
GRID_PARAMS = OptimizationConfig.GRID_CONFIG.get("params", {})
OPTUNA_CONFIG = OptimizationConfig.OPTUNA_CONFIG.get("params", {})
OPTUNA_TRIALS = int(OptimizationConfig.OPTUNA_CONFIG.get("n_trials", 20))


# 5. Data Splitting Settings (WFA)
try:
    BASE_START = datetime.strptime(BacktestConfig.START_DATE, "%Y-%m-%d")

except Exception as e:
    print(f"Error parsing dates from config: {e}. Using defaults.")
    BASE_START = datetime(2023, 1, 1)

# Global Data Cache for Multiprocessing (Copy-on-Write)
DATA_DICT = {}


def add_months(dt, months):
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    day = min(
        dt.day,
        [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1],
    )
    return datetime(year, month, day)


def build_walk_forward_splits(
    base_start,
    folds,
    train_months=12,
    val_months=6,
    test_months=6,
    step_months=6,
):
    splits = []
    cursor = base_start
    for i in range(folds):
        train_start = cursor
        train_end = add_months(train_start, train_months)
        val_start = train_end
        val_end = add_months(val_start, val_months)
        test_start = val_end
        test_end = add_months(test_start, test_months)
        splits.append(
            {
                "fold": i + 1,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        cursor = add_months(cursor, step_months)
    return splits


def save_optimization_rows(db_path, run_id, stage, rows):
    """
    Persist optimization artifacts into SQLite.
    """
    if not rows:
        return
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            created_at TEXT NOT NULL,
            params_json TEXT NOT NULL,
            sharpe REAL,
            cagr TEXT,
            mdd TEXT,
            train_sharpe REAL,
            robustness_score REAL,
            extra_json TEXT
        )
        """
    )

    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        cur.execute(
            """
            INSERT INTO optimization_results(
                run_id, stage, created_at, params_json, sharpe, cagr, mdd, train_sharpe, robustness_score, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stage,
                now,
                json.dumps(row.get("params", {})),
                float(row.get("sharpe", 0.0)) if row.get("sharpe") is not None else None,
                str(row.get("cagr")) if row.get("cagr") is not None else None,
                str(row.get("mdd")) if row.get("mdd") is not None else None,
                float(row.get("train_sharpe", 0.0))
                if row.get("train_sharpe") is not None
                else None,
                float(row.get("robustness_score", 0.0))
                if row.get("robustness_score") is not None
                else None,
                json.dumps({k: v for k, v in row.items() if k not in {"params", "sharpe", "cagr", "mdd", "train_sharpe", "robustness_score"}}),
            ),
        )
    conn.commit()
    conn.close()


def load_all_data(csv_dir, symbol_list):
    """
    Loads all data into memory using Polars once.
    """
    print(f"Loading data for {len(symbol_list)} symbols from {csv_dir}...")
    data = {}
    for s in symbol_list:
        candidates = [
            os.path.join(csv_dir, f"{s}.csv"),
            os.path.join(csv_dir, f"{s.replace('/', '')}.csv"),
            os.path.join(csv_dir, f"{s.replace('/', '_')}.csv"),
            os.path.join(csv_dir, f"{s.replace('/', '-')}.csv"),
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
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
        )
        backtest.simulate_trading(output=False)
        backtest.portfolio.create_equity_curve_dataframe()

        stats = backtest.portfolio.output_summary_stats()
        stats_dict = {k: v for k, v in stats}
        no_data = stats_dict.get("Status") == "Not enough data"

        # We prioritize Sharpe Ratio for optimization
        sharpe = float(stats_dict.get("Sharpe Ratio", 0.0))
        if sharpe != sharpe:
            sharpe = -999.0  # Handle NaN
        if no_data:
            sharpe = -999.0

        return {
            "params": params,
            "sharpe": sharpe,
            "cagr": stats_dict.get("CAGR", "0.0%").strip("%"),
            "mdd": stats_dict.get("Max Drawdown", "0.0%").strip("%"),
            "num_trades": len(backtest.portfolio.trades),
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
    def __init__(
        self, strategy_cls, param_grid, csv_dir, symbol_list, start_date, end_date
    ):
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
        print(
            f"Starting Grid Search (Train Phase) with {len(combinations)} combinations..."
        )

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

        # Use initializer to share data efficiently on Windows/Spawn
        with multiprocessing.Pool(
            processes=max_workers, initializer=pool_initializer, initargs=(DATA_DICT,)
        ) as pool:
            results = pool.map(run_single_backtest_train, pool_args)

        valid_results = [r for r in results if "error" not in r]
        sorted_results = sorted(valid_results, key=lambda x: x["sharpe"], reverse=True)
        return sorted_results


class OptunaOptimizer:
    def __init__(
        self, strategy_cls, optuna_config, csv_dir, symbol_list, start_date, end_date
    ):
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
                params[key] = trial.suggest_float(
                    key, conf["low"], conf["high"], step=step
                )
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

    def run(self, n_trials=20):
        if not OPTUNA_AVAILABLE:
            print("Error: Optuna not installed.")
            return []

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        best_trials = sorted(
            study.trials, key=lambda t: t.value if t.value else -999, reverse=True
        )

        params_list = []
        for t in best_trials[:10]:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            params_list.append(
                {"params": t.params, "sharpe": t.value, "cagr": "N/A", "mdd": "N/A"}
            )
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

    print(
        f"\n=== FOLD {fold} TRAIN [{train_start.date()} ~ {train_end.date()}] ==="
    )
    if OPTIMIZATION_METHOD == "GRID":
        optimizer = GridSearchOptimizer(
            STRATEGY_CLASS, GRID_PARAMS, CSV_DIR, SYMBOL_LIST, train_start, train_end
        )
        train_results = optimizer.run(max_workers=4)
    elif OPTIMIZATION_METHOD == "OPTUNA":
        optimizer = OptunaOptimizer(
            STRATEGY_CLASS, OPTUNA_CONFIG, CSV_DIR, SYMBOL_LIST, train_start, train_end
        )
        train_results = optimizer.run(n_trials=OPTUNA_TRIALS)
    else:
        raise ValueError(f"Unknown optimization method: {OPTIMIZATION_METHOD}")

    if not train_results:
        return None

    best_candidate = train_results[0]
    print(
        f"[Fold {fold} Train] Top Candidate: Params={best_candidate['params']} | Sharpe={best_candidate['sharpe']:.4f}"
    )

    print(
        f"=== FOLD {fold} VALIDATION [{val_start.date()} ~ {val_end.date()}] ==="
    )
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
    run_id = str(uuid.uuid4())
    db_path = getattr(BaseConfig, "STORAGE_SQLITE_PATH", "logs/lumina_quant.db")

    # Load Data Once
    DATA_DICT = load_all_data(CSV_DIR, SYMBOL_LIST)
    splits = build_walk_forward_splits(
        BASE_START,
        OptimizationConfig.WALK_FORWARD_FOLDS,
    )
    if not splits:
        print("No walk-forward splits generated.")
        sys.exit(1)

    fold_reports = []
    for split in splits:
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

    # Save Best Parameters from winning fold.
    strategy_name = STRATEGY_CLASS.__name__
    save_dir = os.path.join("best_optimized_parameters", strategy_name)
    os.makedirs(save_dir, exist_ok=True)
    best_params_file = os.path.join(save_dir, "best_params.json")
    with open(best_params_file, "w") as f:
        json.dump(winner["selected"]["params"], f, indent=4)
    print(f"[Artifact] Best Parameters saved to '{best_params_file}'")
