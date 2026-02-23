"""Optimization package exports."""

from lumina_quant.optimization.fast_eval import NUMBA_AVAILABLE, evaluate_metrics_numba
from lumina_quant.optimization.frozen_dataset import FrozenDataset, build_frozen_dataset
from lumina_quant.optimization.native_backend import (
    NATIVE_BACKEND_NAME,
    evaluate_metrics_backend,
)
from lumina_quant.optimization.storage import save_optimization_rows
from lumina_quant.optimization.threading_control import configure_numba_threads
from lumina_quant.optimization.walkers import add_months, build_walk_forward_splits

__all__ = [
    "NATIVE_BACKEND_NAME",
    "NUMBA_AVAILABLE",
    "FrozenDataset",
    "add_months",
    "build_frozen_dataset",
    "build_walk_forward_splits",
    "configure_numba_threads",
    "evaluate_metrics_backend",
    "evaluate_metrics_numba",
    "save_optimization_rows",
]
