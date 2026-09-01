from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lumina_quant.alpha_zoo import native_alpha_fold_backend as native_backend
from scripts.research import run_alpha_zoo_69_asset_optuna_hybrid_refit as broad69


def _bars() -> pd.DataFrame:
    close = np.array([100.0, 101.5, 101.0, 103.0, 98.0, 99.0], dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * np.array([1.01, 1.02, 1.01, 1.03, 1.02, 1.01]),
            "low": close * np.array([0.99, 0.98, 0.99, 0.97, 0.98, 0.99]),
            "close": close,
        }
    )


def test_python_alpha_fold_backend_matches_reference_simulator() -> None:
    bars = _bars()
    signal = np.array([0.0, 1.0, 1.0, -1.0, -1.0, 0.0], dtype=float)

    expected = broad69.simulate_symbol(
        bars,
        signal,
        integer_leverage=3,
        allocation_fraction=0.1,
        round_trip_cost_bps=10.0,
    )
    returns, liquidation, wipeout = native_backend.simulate_symbol_arrays(
        bars["close"].to_numpy(dtype=float),
        bars["high"].to_numpy(dtype=float),
        bars["low"].to_numpy(dtype=float),
        signal,
        integer_leverage=3,
        allocation_fraction=0.1,
        round_trip_cost_bps=10.0,
        backend="python",
    )

    assert returns == pytest.approx(expected.returns)
    assert liquidation.tolist() == expected.liquidation_flags.tolist()
    assert wipeout.tolist() == expected.account_wipeout_flags.tolist()


def test_rust_alpha_fold_backend_matches_reference_when_available() -> None:
    if not native_backend.native_backend_available():
        pytest.skip("Rust alpha-fold backend is not built in this environment")
    bars = _bars()
    signal = np.array([1.0, 1.0, 0.0, -1.0, -1.0, 0.0], dtype=float)

    expected = broad69.simulate_symbol(
        bars,
        signal,
        integer_leverage=4,
        allocation_fraction=0.2,
        round_trip_cost_bps=10.0,
    )
    returns, liquidation, wipeout = native_backend.simulate_symbol_arrays(
        bars["close"].to_numpy(dtype=float),
        bars["high"].to_numpy(dtype=float),
        bars["low"].to_numpy(dtype=float),
        signal,
        integer_leverage=4,
        allocation_fraction=0.2,
        round_trip_cost_bps=10.0,
        backend="rust",
    )

    assert np.max(np.abs(returns - expected.returns)) <= 1e-12
    assert liquidation.tolist() == expected.liquidation_flags.tolist()
    assert wipeout.tolist() == expected.account_wipeout_flags.tolist()
