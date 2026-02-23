from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

_OPTIMIZE_PATH = Path(__file__).resolve().parents[1] / "optimize.py"
sys.path.insert(0, str(_OPTIMIZE_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("optimize_module", _OPTIMIZE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
optimize = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(optimize)


def test_resolve_topk_bounds():
    topk = optimize._resolve_topk(100)
    assert 1 <= topk <= 100
    assert topk >= optimize.TWO_STAGE_MIN_TOPK
    assert topk <= optimize.TWO_STAGE_MAX_TOPK


def test_resolve_prefilter_window_bounds():
    start = datetime(2024, 1, 1)
    end = datetime(2024, 2, 1)
    fast_start, fast_end = optimize._resolve_prefilter_window(start, end)

    assert fast_start == start
    assert start < fast_end <= end
