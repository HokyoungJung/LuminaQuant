from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from lumina_quant.alpha_zoo import native_hybrid_optuna_backend as native_backend

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts" / "research" / "run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py"
)
SPEC = importlib.util.spec_from_file_location(
    "alpha_zoo_optuna_hybrid_native_test_runner", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_python_backend_mode_returns_none() -> None:
    returns = np.ones((8, 3), dtype=np.float64) * 0.001

    assert (
        native_backend.evaluate_hybrid_portfolio_native(
            returns,
            version="v3_5",
            start_idx=0,
            mape_window=4,
            bias_window=3,
            short_vol_window=3,
            bias_correction_alpha=1.0,
            bias_combine_ratio=0.25,
            max_single_weight=0.78,
            default_idx=0,
            high_vol_best_idx=1,
            default_weight_ratio=0.5,
            high_vol_threshold=0.001,
            high_vol_weight_boost=0.1,
            backend="python",
        )
        is None
    )


def test_native_backend_matches_python_when_available() -> None:
    if not native_backend.native_backend_available():
        pytest.skip("Rust hybrid Optuna backend is not built in this environment")

    rng = np.random.default_rng(20260527)
    returns = rng.normal(0.00005, 0.002, size=(256, 3)).astype(np.float64)
    params = RUNNER.HybridParams(mape_window=25, bias_window=10, short_vol_window=7)
    learned = RUNNER.LearnedParams(
        high_vol_threshold=0.001,
        default_idx=0,
        high_vol_best_idx=1,
        default_weight_ratio=0.5,
        high_vol_weight_boost=0.1,
        cv_score=0.0,
    )

    previous = os.environ.get(native_backend.HYBRID_OPTUNA_BACKEND_ENV)
    try:
        os.environ[native_backend.HYBRID_OPTUNA_BACKEND_ENV] = "python"
        py_returns, py_weights, py_allocations = RUNNER._portfolio_returns_for_params(
            returns,
            params=params,
            learned=learned,
            version="v3_6",
            start_idx=0,
            allocation_stride=24,
        )
        os.environ[native_backend.HYBRID_OPTUNA_BACKEND_ENV] = "rust"
        rust_returns, rust_weights, rust_allocations = RUNNER._portfolio_returns_for_params(
            returns,
            params=params,
            learned=learned,
            version="v3_6",
            start_idx=0,
            allocation_stride=24,
        )
    finally:
        if previous is None:
            os.environ.pop(native_backend.HYBRID_OPTUNA_BACKEND_ENV, None)
        else:
            os.environ[native_backend.HYBRID_OPTUNA_BACKEND_ENV] = previous

    assert np.max(np.abs(py_returns - rust_returns)) <= 1e-8
    assert np.max(np.abs(py_weights - rust_weights)) <= 1e-8
    assert py_allocations[-1]["default_idx"] == rust_allocations[-1]["default_idx"]
    assert py_allocations[-1]["weights"] == pytest.approx(rust_allocations[-1]["weights"])
