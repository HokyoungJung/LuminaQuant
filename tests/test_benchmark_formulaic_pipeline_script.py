from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "scripts" / "benchmark_formulaic_pipeline.py"
    spec = importlib.util.spec_from_file_location("benchmark_formulaic_pipeline_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load benchmark_formulaic_pipeline module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_synthetic_payload_has_expected_keys():
    payload = MODULE._synthetic_ohlcv(400)
    assert {"opens", "highs", "lows", "closes", "volumes", "vwaps"} <= set(payload)
    assert len(payload["closes"]) == 400


def test_summarize_computes_throughput():
    summary = MODULE._summarize(
        name="unit",
        iterations=2,
        samples=[0.5, 1.0],
        units_per_iter=10.0,
        extra={"ok": 1},
    )
    assert summary.name == "unit"
    assert summary.median_seconds > 0.0
    assert summary.throughput > 0.0
