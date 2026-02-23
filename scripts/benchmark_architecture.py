"""Architecture benchmark suite for optimization and parity components."""

from __future__ import annotations

import argparse
import time

import numpy as np
from lumina_quant.events import MarketEvent
from lumina_quant.message_bus import MessageBus
from lumina_quant.optimization.fast_eval import evaluate_metrics_numba
from lumina_quant.replay import assert_monotonic_event_order, stable_event_sort
from lumina_quant.runtime_cache import RuntimeCache


def benchmark_metric_kernel(bars: int, evals: int) -> float:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.001, size=max(2, int(bars))).astype(np.float64)
    totals = (1.0 + returns).cumprod() * 10_000.0
    evaluate_metrics_numba(totals, 252)
    start = time.perf_counter()
    for _ in range(max(1, int(evals))):
        evaluate_metrics_numba(totals, 252)
    elapsed = max(1e-9, time.perf_counter() - start)
    return float(evals) / elapsed


def benchmark_message_bus(messages: int) -> float:
    bus = MessageBus()
    sink = {"count": 0}

    def _handler(payload) -> None:
        _ = payload
        sink["count"] += 1

    bus.subscribe("event.MARKET", _handler)
    start = time.perf_counter()
    for idx in range(max(1, int(messages))):
        bus.publish("event.MARKET", idx)
    elapsed = max(1e-9, time.perf_counter() - start)
    return float(messages) / elapsed


def benchmark_runtime_cache(updates: int) -> float:
    cache = RuntimeCache()
    start = time.perf_counter()
    for idx in range(max(1, int(updates))):
        cache.update_market(
            "BTC/USDT",
            {
                "time": idx,
                "timestamp_ns": idx * 1_000_000,
                "sequence": idx,
                "close": 100.0 + float(idx),
                "volume": 1.0,
            },
        )
        cache.update_order_state(f"OID-{idx}", {"state": "OPEN", "symbol": "BTC/USDT"})
    elapsed = max(1e-9, time.perf_counter() - start)
    return float(updates) / elapsed


def benchmark_replay_sort(events: int) -> float:
    items = [
        MarketEvent(
            time=idx,
            symbol="BTC/USDT",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1.0,
            timestamp_ns=int(events - idx),
            sequence=idx,
        )
        for idx in range(max(1, int(events)))
    ]
    start = time.perf_counter()
    ordered = stable_event_sort(items)
    assert_monotonic_event_order(ordered)
    elapsed = max(1e-9, time.perf_counter() - start)
    return float(events) / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark architecture components")
    parser.add_argument("--bars", type=int, default=20_000)
    parser.add_argument("--evals", type=int, default=3_000)
    parser.add_argument("--messages", type=int, default=200_000)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--events", type=int, default=100_000)
    args = parser.parse_args()

    metric_eps = benchmark_metric_kernel(args.bars, args.evals)
    bus_eps = benchmark_message_bus(args.messages)
    cache_eps = benchmark_runtime_cache(args.updates)
    replay_eps = benchmark_replay_sort(args.events)

    print("benchmark_architecture")
    print(f"metric_eval_per_sec={metric_eps:.2f}")
    print(f"bus_publish_per_sec={bus_eps:.2f}")
    print(f"cache_update_per_sec={cache_eps:.2f}")
    print(f"replay_sort_events_per_sec={replay_eps:.2f}")


if __name__ == "__main__":
    main()
