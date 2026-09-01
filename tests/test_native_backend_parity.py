"""Unconditional CPU-vs-pyo3 numeric parity tests for Rust kernels.

These tests assert that every pyo3 Rust kernel agrees with its Python/CPU
reference implementation at rtol=1e-8 (matching the repo golden tolerance).
They run unconditionally: if the native extension is unavailable the tests
skip with an explicit reason rather than silently passing.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

_RNG_SEED = 42
_ANNUAL_PERIODS = 252
_SERIES_LEN = 500
_RTOL = 1e-8
_ATOL = 1e-12  # absolute floor for near-zero values


def _equity_curve(n: int = _SERIES_LEN, seed: int = _RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.005, n)
    return np.ascontiguousarray((1.0 + returns).cumprod() * 10_000.0, dtype=np.float64)


def _symbol_arrays(
    n: int = 50, seed: int = _RNG_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = np.ascontiguousarray(
        (1.0 + rng.normal(0.0, 0.005, n)).cumprod() * 100.0, dtype=np.float64
    )
    high = np.ascontiguousarray(close * (1.0 + rng.uniform(0.001, 0.015, n)), dtype=np.float64)
    low = np.ascontiguousarray(close * (1.0 - rng.uniform(0.001, 0.015, n)), dtype=np.float64)
    raw_signal = rng.choice([-1.0, 0.0, 1.0], size=n)
    signal = np.ascontiguousarray(raw_signal, dtype=np.float64)
    return close, high, low, signal


# ── evaluate_metrics (Sharpe / CAGR / MaxDD) ─────────────────────────────────


class TestEvaluateMetricsParity:
    """pyo3 evaluate_metrics must match the Python reference at rtol=1e-8."""

    @pytest.fixture(autouse=True)
    def _require_pyo3(self) -> None:
        try:
            import lumina_quant._compute as _lq_compute  # type: ignore[attr-defined]

            assert hasattr(_lq_compute, "evaluate_metrics")
        except Exception as exc:
            pytest.skip(f"lumina_quant._compute.evaluate_metrics unavailable: {exc}")

    def _py_result(self, series: np.ndarray, annual_periods: int) -> tuple[float, float, float]:
        from lumina_quant.optimization.native_backend import _evaluate_python

        return _evaluate_python(series, annual_periods)

    def _pyo3_result(self, series: np.ndarray, annual_periods: int) -> tuple[float, float, float]:
        from lumina_quant._compute import evaluate_metrics  # type: ignore[attr-defined]

        raw = evaluate_metrics(np.ascontiguousarray(series, dtype=np.float64), annual_periods)
        return float(raw[0]), float(raw[1]), float(raw[2])

    def test_sharpe_parity(self) -> None:
        series = _equity_curve()
        py_s, _, _ = self._py_result(series, _ANNUAL_PERIODS)
        pyo3_s, _, _ = self._pyo3_result(series, _ANNUAL_PERIODS)
        assert abs(pyo3_s - py_s) <= _RTOL * max(abs(py_s), _ATOL), (
            f"Sharpe parity failed: pyo3={pyo3_s:.12g} python={py_s:.12g} "
            f"diff={abs(pyo3_s - py_s):.3e}"
        )

    def test_cagr_parity(self) -> None:
        series = _equity_curve()
        _, py_c, _ = self._py_result(series, _ANNUAL_PERIODS)
        _, pyo3_c, _ = self._pyo3_result(series, _ANNUAL_PERIODS)
        assert abs(pyo3_c - py_c) <= _RTOL * max(abs(py_c), _ATOL), (
            f"CAGR parity failed: pyo3={pyo3_c:.12g} python={py_c:.12g} "
            f"diff={abs(pyo3_c - py_c):.3e}"
        )

    def test_max_drawdown_parity(self) -> None:
        series = _equity_curve()
        _, _, py_d = self._py_result(series, _ANNUAL_PERIODS)
        _, _, pyo3_d = self._pyo3_result(series, _ANNUAL_PERIODS)
        assert abs(pyo3_d - py_d) <= _RTOL * max(abs(py_d), _ATOL), (
            f"MaxDD parity failed: pyo3={pyo3_d:.12g} python={py_d:.12g} "
            f"diff={abs(pyo3_d - py_d):.3e}"
        )

    def test_triple_parity_short_series(self) -> None:
        series = _equity_curve(n=10)
        py_out = self._py_result(series, _ANNUAL_PERIODS)
        pyo3_out = self._pyo3_result(series, _ANNUAL_PERIODS)
        for i, label in enumerate(("sharpe", "cagr", "max_dd")):
            assert abs(pyo3_out[i] - py_out[i]) <= _RTOL * max(abs(py_out[i]), _ATOL), (
                f"{label} parity failed for short series: "
                f"pyo3={pyo3_out[i]:.12g} python={py_out[i]:.12g}"
            )

    def test_via_backend_selection_details(self) -> None:
        """backend_selection_details reports pyo3_available=True when built."""
        from lumina_quant.optimization.native_backend import backend_selection_details

        details = backend_selection_details()
        assert details["pyo3_available"] == "True", (
            "pyo3 extension is importable but backend_selection_details reports False"
        )


# ── simulate_symbol_fold (alpha-fold) ─────────────────────────────────────────


class TestSimulateSymbolFoldParity:
    """pyo3 simulate_symbol_fold must match the Python reference at rtol=1e-8."""

    @pytest.fixture(autouse=True)
    def _require_pyo3(self) -> None:
        try:
            import lumina_quant._compute as _lq_compute  # type: ignore[attr-defined]

            assert hasattr(_lq_compute, "simulate_symbol_fold")
        except Exception as exc:
            pytest.skip(f"lumina_quant._compute.simulate_symbol_fold unavailable: {exc}")

    def _py_simulate(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        signal: np.ndarray,
        integer_leverage: int,
        allocation_fraction: float,
        round_trip_cost_bps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from lumina_quant.alpha_zoo.native_alpha_fold_backend import (
            _simulate_symbol_arrays_python,
        )

        return _simulate_symbol_arrays_python(
            close,
            high,
            low,
            signal,
            integer_leverage=integer_leverage,
            allocation_fraction=allocation_fraction,
            round_trip_cost_bps=round_trip_cost_bps,
        )

    def _pyo3_simulate(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        signal: np.ndarray,
        integer_leverage: int,
        allocation_fraction: float,
        round_trip_cost_bps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from lumina_quant._compute import simulate_symbol_fold  # type: ignore[attr-defined]

        raw = simulate_symbol_fold(
            np.ascontiguousarray(close, dtype=np.float64),
            np.ascontiguousarray(high, dtype=np.float64),
            np.ascontiguousarray(low, dtype=np.float64),
            np.ascontiguousarray(signal, dtype=np.float64),
            int(integer_leverage),
            float(allocation_fraction),
            float(round_trip_cost_bps),
        )
        return raw[0], raw[1].astype(bool), raw[2].astype(bool)

    def test_returns_parity(self) -> None:
        close, high, low, signal = _symbol_arrays()
        py_r, _, _ = self._py_simulate(close, high, low, signal, 3, 0.1, 10.0)
        pyo3_r, _, _ = self._pyo3_simulate(close, high, low, signal, 3, 0.1, 10.0)
        max_diff = float(np.max(np.abs(py_r - pyo3_r)))
        ref_scale = float(np.max(np.abs(py_r)))
        assert max_diff <= _RTOL * max(ref_scale, _ATOL), (
            f"simulate_symbol_fold returns parity failed: max_diff={max_diff:.3e}"
        )

    def test_liquidation_flags_parity(self) -> None:
        close, high, low, signal = _symbol_arrays()
        _, py_liq, _ = self._py_simulate(close, high, low, signal, 5, 0.2, 5.0)
        _, pyo3_liq, _ = self._pyo3_simulate(close, high, low, signal, 5, 0.2, 5.0)
        assert np.array_equal(py_liq, pyo3_liq), (
            "simulate_symbol_fold liquidation flags mismatch between Python and pyo3"
        )

    def test_wipeout_flags_parity(self) -> None:
        close, high, low, signal = _symbol_arrays()
        _, _, py_wipe = self._py_simulate(close, high, low, signal, 5, 0.2, 5.0)
        _, _, pyo3_wipe = self._pyo3_simulate(close, high, low, signal, 5, 0.2, 5.0)
        assert np.array_equal(py_wipe, pyo3_wipe), (
            "simulate_symbol_fold wipeout flags mismatch between Python and pyo3"
        )

    def test_via_simulate_symbol_arrays_auto_backend(self) -> None:
        """simulate_symbol_arrays in auto mode must use pyo3 when available."""
        from lumina_quant.alpha_zoo.native_alpha_fold_backend import (
            native_backend_available,
            simulate_symbol_arrays,
        )

        if not native_backend_available():
            pytest.skip("native alpha-fold backend not available")

        close, high, low, signal = _symbol_arrays()
        auto_r, _auto_liq, _auto_wipe = simulate_symbol_arrays(
            close,
            high,
            low,
            signal,
            integer_leverage=3,
            allocation_fraction=0.1,
            round_trip_cost_bps=10.0,
            backend="auto",
        )
        py_r, _py_liq, _py_wipe = simulate_symbol_arrays(
            close,
            high,
            low,
            signal,
            integer_leverage=3,
            allocation_fraction=0.1,
            round_trip_cost_bps=10.0,
            backend="python",
        )
        max_diff = float(np.max(np.abs(auto_r - py_r)))
        ref_scale = float(np.max(np.abs(py_r)))
        assert max_diff <= _RTOL * max(ref_scale, _ATOL), (
            f"auto vs python returns parity failed: max_diff={max_diff:.3e}"
        )
