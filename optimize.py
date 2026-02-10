import itertools
import multiprocessing
import sys
import os
import polars as pl
from datetime import datetime

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
    TRAIN_START = datetime.strptime(BacktestConfig.START_DATE, "%Y-%m-%d")
    TRAIN_END = datetime(TRAIN_START.year + 1, 1, 1)
    VAL_START = TRAIN_END
    VAL_END = datetime(VAL_START.year, 7, 1)
    TEST_START = VAL_END
    TEST_END = datetime(TEST_START.year + 1, 1, 1)

except Exception as e:
    print(f"Error parsing dates from config: {e}. Using defaults.")
    TRAIN_START = datetime(2023, 1, 1)
    TRAIN_END = datetime(2024, 1, 1)
    VAL_START = datetime(2024, 1, 1)
    VAL_END = datetime(2024, 6, 1)
    TEST_START = datetime(2024, 6, 1)
    TEST_END = datetime(2025, 1, 1)

# Global Data Cache for Multiprocessing (Copy-on-Write)
DATA_DICT = {}


def load_all_data(csv_dir, symbol_list):
    """
    Loads all data into memory using Polars once.
    """
    print(f"Loading data for {len(symbol_list)} symbols from {csv_dir}...")
    data = {}
    for s in symbol_list:
        csv_path = os.path.join(csv_dir, f"{s}.csv")
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
        backtest.simulate_trading()

        stats = backtest.portfolio.output_summary_stats()
        stats_dict = {k: v for k, v in stats}

        # We prioritize Sharpe Ratio for optimization
        sharpe = float(stats_dict.get("Sharpe Ratio", 0.0))
        if sharpe != sharpe:
            sharpe = -999.0  # Handle NaN

        return {
            "params": params,
            "sharpe": sharpe,
            "cagr": stats_dict.get("CAGR", "0.0%").strip("%"),
            "mdd": stats_dict.get("Max Drawdown", "0.0%").strip("%"),
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


if __name__ == "__main__":
    # Load Data Once
    DATA_DICT = load_all_data(CSV_DIR, SYMBOL_LIST)

    print(f"=== PHASE 1: TRAINING [{TRAIN_START.date()} ~ {TRAIN_END.date()}] ===")
    if OPTIMIZATION_METHOD == "GRID":
        optimizer = GridSearchOptimizer(
            STRATEGY_CLASS, GRID_PARAMS, CSV_DIR, SYMBOL_LIST, TRAIN_START, TRAIN_END
        )
        train_results = optimizer.run(max_workers=4)

    elif OPTIMIZATION_METHOD == "OPTUNA":
        optimizer = OptunaOptimizer(
            STRATEGY_CLASS, OPTUNA_CONFIG, CSV_DIR, SYMBOL_LIST, TRAIN_START, TRAIN_END
        )
        # Optuna jobs=-1 can parallelize, but we need to handle data sharing if so.
        # For now, keep it single threaded or handle via Optuna's mechanism.
        # Simple run() above is single process.
        train_results = optimizer.run(n_trials=OPTUNA_TRIALS)

    else:
        print(f"Unknown method: {OPTIMIZATION_METHOD}")
        sys.exit(1)

    if not train_results:
        print("No valid results found in optimization.")
        sys.exit(0)

    print("\n[Train] Top Candidate:")
    best_candidate = train_results[0]
    print(
        f"Params: {best_candidate['params']} | Sharpe: {best_candidate['sharpe']:.4f}"
    )

    # ==========================================
    # VALIDATION PHASE
    # ==========================================
    print(f"\n=== PHASE 2: VALIDATION [{VAL_START.date()} ~ {VAL_END.date()}] ===")
    print("Verifying Top 3 Candidates on Validation Set...")

    val_candidates = []
    # Test top 3 or less
    limit = min(3, len(train_results))
    for cand in train_results[:limit]:
        res = _execute_backtest(
            STRATEGY_CLASS,
            cand["params"],
            CSV_DIR,
            SYMBOL_LIST,
            VAL_START,
            VAL_END,
            data_dict=DATA_DICT,
        )
        res["train_sharpe"] = cand["sharpe"]
        val_candidates.append(res)
        print(
            f"Params: {cand['params']} -> Val Sharpe: {res['sharpe']:.4f} (Train: {cand['sharpe']:.4f})"
        )

    # Robustness Check & Ranking
    for c in val_candidates:
        train_s = float(c["train_sharpe"])
        val_s = float(c["sharpe"])

        # Robustness Score
        divergence = abs(train_s - val_s)
        penalty_factor = 0.5
        c["robustness_score"] = val_s - (divergence * penalty_factor)

    # Sort by Robustness Score instead of raw Sharpe
    val_candidates.sort(key=lambda x: x["robustness_score"], reverse=True)
    final_best = val_candidates[0]

    print(f"\n[Validation] Selected Best Robust Params: {final_best['params']}")

    # ==========================================
    # TEST PHASE (FINAL)
    # ==========================================
    print(f"\n=== PHASE 3: FINAL TEST [{TEST_START.date()} ~ {TEST_END.date()}] ===")
    print("Running Final Simulation on Unseen Test Data...")

    test_res = _execute_backtest(
        STRATEGY_CLASS,
        final_best["params"],
        CSV_DIR,
        SYMBOL_LIST,
        TEST_START,
        TEST_END,
        data_dict=DATA_DICT,
    )

    print("\n>>>> FINAL REPORT <<<<")
    print(f"Best Params: {final_best['params']}")
    print(f"Train Sharpe : {best_candidate['sharpe']:.4f}")
    print(f"Val Sharpe   : {final_best['sharpe']:.4f}")
    print(f"Test Sharpe  : {test_res['sharpe']:.4f}")
    print(f"Test CAGR    : {test_res['cagr']}")
    print(f"Test MaxDD   : {test_res['mdd']}")

    # Overfitting Check
    train_score = float(best_candidate["sharpe"])
    test_score = float(test_res["sharpe"])

    if test_score < train_score * 0.5:
        print("\n[WARNING] Test performance dropped significantly (>50%) vs Train.")
        print("This suggests OVERFITTING. Consider simpler logic or fewer parameters.")
    else:
        print("\n[SUCCESS] Strategy appears robust across datasets.")

    # Save Best Parameters
    import json
    import os

    strategy_name = STRATEGY_CLASS.__name__
    # Create directory: best_optimized_parameters/<StrategyName>
    save_dir = os.path.join("best_optimized_parameters", strategy_name)
    os.makedirs(save_dir, exist_ok=True)

    best_params_file = os.path.join(save_dir, "best_params.json")

    with open(best_params_file, "w") as f:
        json.dump(final_best["params"], f, indent=4)
    print(f"\n[Artifact] Best Parameters saved to '{best_params_file}'")
