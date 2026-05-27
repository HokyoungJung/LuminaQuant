from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_optuna_hybrid_backend.py"
SPEC = importlib.util.spec_from_file_location("benchmark_optuna_hybrid_backend", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_parser_defaults_are_bounded_for_memory() -> None:
    args = MODULE._build_parser().parse_args([])

    assert args.rows <= 50_000
    assert args.columns == 3
    assert args.evals >= 1
    assert args.backend == "both"


def test_synthetic_returns_are_deterministic_and_finite() -> None:
    first = MODULE.generate_synthetic_returns(rows=16, columns=3)
    second = MODULE.generate_synthetic_returns(rows=16, columns=3)

    assert first.shape == (16, 3)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_require_rust_returns_skip_payload_when_native_missing(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.native_backend, "native_backend_available", lambda: False)
    monkeypatch.setattr(
        MODULE.native_backend,
        "hybrid_optuna_backend_diagnostics",
        lambda _requested=None: {
            "requested_backend": "rust",
            "resolved_backend": "python",
            "native_available": False,
            "native_library_path": None,
            "native_load_error": "missing test lib",
        },
    )

    result = MODULE.run_benchmark(
        rows=16,
        columns=3,
        evals=1,
        version="v3_5",
        backend="both",
        require_rust=True,
    )

    assert result.status == "rust_unavailable"
    assert result.python is None
    assert result.rust is None
    assert result.parity.reason == "missing test lib"
