from __future__ import annotations

import lumina_quant.optimization.native_backend as native_backend
import numpy as np
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


def test_discover_candidates_handles_macos_naming(monkeypatch, tmp_path):
    c_path = tmp_path / "native" / "c_metrics" / "build" / "liblumina_metrics.dylib"
    r_path = tmp_path / "native" / "rust_metrics" / "target" / "release" / "liblumina_metrics.dylib"
    c_path.parent.mkdir(parents=True, exist_ok=True)
    r_path.parent.mkdir(parents=True, exist_ok=True)
    c_path.write_bytes(b"x")
    r_path.write_bytes(b"x")

    monkeypatch.delenv("LQ_NATIVE_METRICS_DLL", raising=False)
    monkeypatch.setattr(native_backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native_backend, "_project_root", lambda: tmp_path)

    found = native_backend._discover_dll_candidates()
    labels = {name for name, _ in found}
    assert labels == {"c", "rust"}
