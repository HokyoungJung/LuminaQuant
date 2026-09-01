from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from lumina_quant.core.market_window_contract import (
    build_market_window_event,
    market_window_event_payload,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_market_window_contract.py"
SPEC = importlib.util.spec_from_file_location("benchmark_market_window_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_trusted_market_window_path_matches_generic_payload() -> None:
    bars = MODULE.generate_normalized_bars(symbols=2, window_seconds=8)
    event_time = bars["S0/USDT"][-1][0]

    generic = build_market_window_event(
        time=event_time,
        window_seconds=8,
        bars_1s=bars,
        event_time_watermark_ms=event_time,
        lag_ms=0,
        is_stale=False,
    )
    trusted = build_market_window_event(
        time=event_time,
        window_seconds=8,
        bars_1s=bars,
        event_time_watermark_ms=event_time,
        lag_ms=0,
        is_stale=False,
        bars_1s_already_normalized=True,
    )

    assert market_window_event_payload(trusted) == market_window_event_payload(generic)


def test_market_window_benchmark_payload_is_bounded_and_passing() -> None:
    result = MODULE.run_benchmark(symbols=2, window_seconds=8, ticks=64, evals=2)

    assert result.status == "pass"
    assert result.builder.payloads_match is True
    assert result.builder.generic_seconds_per_eval > 0.0
    assert result.builder.trusted_seconds_per_eval > 0.0
    assert result.aggregator.ticks == 64
    assert result.aggregator.symbols == 2
