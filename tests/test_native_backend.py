from __future__ import annotations

import lumina_quant.optimization.native_backend as native_backend
import numpy as np
import pytest
from lumina_quant.optimization.native_backend import (
    NATIVE_BACKEND_NAME,
    backend_selection_details,
    evaluate_metrics_backend,
)


def test_native_backend_fallback_path_returns_finite_values():
    totals = np.asarray([10000.0, 10020.0, 9980.0, 10050.0, 10100.0], dtype=np.float64)
    sharpe, cagr, max_dd = evaluate_metrics_backend(totals, 252)
    assert isinstance(NATIVE_BACKEND_NAME, str)
    details = backend_selection_details()
    assert details["mode"] in {"python", "numba", "native"}
    assert np.isfinite(float(sharpe))
    assert np.isfinite(float(cagr))
    assert np.isfinite(float(max_dd))


def test_backend_details_has_pyo3_available_key():
    details = backend_selection_details()
    assert "pyo3_available" in details
    # When lumina_quant._compute is installed, pyo3 should be available
    assert details["pyo3_available"] in {"True", "False"}


def test_evaluate_metrics_backend_falls_back_when_pyo3_none(monkeypatch):
    monkeypatch.setattr(native_backend, "_PYO3_FN", None)
    monkeypatch.setattr(native_backend, "_BACKEND_MODE", "python")
    totals = np.asarray([10000.0, 10020.0, 9980.0, 10050.0], dtype=np.float64)
    sharpe, cagr, max_dd = evaluate_metrics_backend(totals, 252)
    assert np.isfinite(float(sharpe))
    assert np.isfinite(float(cagr))
    assert np.isfinite(float(max_dd))
