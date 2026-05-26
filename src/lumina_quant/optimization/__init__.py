"""Optimization package exports."""

from lumina_quant.optimization.fast_eval import NUMBA_AVAILABLE, evaluate_metrics_numba
from lumina_quant.optimization.frozen_dataset import FrozenDataset, build_frozen_dataset
from lumina_quant.optimization.native_backend import (
    NATIVE_BACKEND_NAME,
    evaluate_metrics_backend,
)
from lumina_quant.optimization.search_policy import (
    BoundedGridResult,
    build_bounded_grid_combinations,
    optimization_search_policy_payload,
    run_optuna_study,
    suggest_params_from_optuna_config,
)
from lumina_quant.optimization.storage import save_optimization_rows
from lumina_quant.optimization.threading_control import configure_numba_threads
from lumina_quant.optimization.walkers import add_months, build_walk_forward_splits

__all__ = [
    "NATIVE_BACKEND_NAME",
    "NUMBA_AVAILABLE",
    "BoundedGridResult",
    "FrozenDataset",
    "add_months",
    "build_bounded_grid_combinations",
    "build_frozen_dataset",
    "build_walk_forward_splits",
    "configure_numba_threads",
    "evaluate_metrics_backend",
    "evaluate_metrics_numba",
    "optimization_search_policy_payload",
    "run_optuna_study",
    "save_optimization_rows",
    "suggest_params_from_optuna_config",
]
