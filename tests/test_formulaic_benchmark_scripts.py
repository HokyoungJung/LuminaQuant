from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(name: str, rel_path: str):
    root = Path(__file__).resolve().parent.parent
    module_path = root / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_indicators_parser_defaults():
    module = _load_module("benchmark_indicators_script", "scripts/benchmark_indicators.py")
    args = module._build_parser().parse_args([])
    assert args.rows >= 300
    assert args.iters >= 1
    assert args.output.endswith(".json")


def test_benchmark_formulaic_alpha_parser_defaults():
    module = _load_module("benchmark_formulaic_alpha_script", "scripts/benchmark_formulaic_alpha.py")
    args = module._build_parser().parse_args([])
    assert args.alpha_start == 1
    assert args.alpha_end == 101
    assert args.backend in {"auto", "numpy", "polars"}


def test_benchmark_backtest_loop_parser_defaults():
    module = _load_module("benchmark_backtest_loop_script", "scripts/benchmark_backtest_loop.py")
    args = module._build_parser().parse_args([])
    assert args.iters >= 1
    assert args.warmup >= 0
