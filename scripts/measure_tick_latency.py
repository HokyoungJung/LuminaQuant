"""Measure per-tick latency proxies for the live order path and replay checker.

Three measurements (none require a live exchange connection):

1. live_signal_per_tick — per-SINGLE-tick call latency for
   evaluate_debounced_state_native + evaluate_trailing_state_native via the
   pyo3 backend (lumina_quant._compute).  Falls back to Python if the native
   extension is unavailable.  Reports median/p95/p99 across --iters calls.

2. paper_order_path — per-transition latency through OrderStateMachine.transition()
   (SUBMITTED→ACKED then ACKED→FILLED), the innermost hot path of the paper-mode
   order lifecycle.  Pure Python dict lookup, no I/O.
   Reports median/p95/p99 across --iters calls.

3. replay_path_proxy — per-event call latency for ShadowLiveRunner.run() with a
   single synthetic event pair.  Measures stable_event_sort + comparison overhead.
   Labelled: "parity-checker overhead; NOT signal or order latency".
   Reports median/p95/p99 across --iters calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(samples: list[float], pct: float) -> float:
    """Return the p-th percentile (0-100) of sorted samples."""
    if not samples:
        return float("nan")
    sorted_s = sorted(samples)
    k = (pct / 100.0) * (len(sorted_s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(sorted_s) - 1)
    frac = k - lo
    return sorted_s[lo] * (1.0 - frac) + sorted_s[hi] * frac


def _stats(samples_s: list[float]) -> dict[str, float]:
    ms = [s * 1000.0 for s in samples_s]
    return {
        "median_ms": statistics.median(ms),
        "mean_ms": statistics.fmean(ms),
        "p95_ms": _percentile(ms, 95),
        "p99_ms": _percentile(ms, 99),
        "min_ms": min(ms),
        "max_ms": max(ms),
        "n": len(ms),
    }


# ---------------------------------------------------------------------------
# Measurement 1: live signal per-tick latency
# ---------------------------------------------------------------------------


def _measure_live_signal_per_tick(iters: int) -> dict:
    """Time a single-tick call through debounced + trailing signal kernels."""
    from lumina_quant.alpha_zoo import native_live_signal_backend as nb

    rng = np.random.default_rng(42)
    close = rng.uniform(90.0, 110.0, size=1).astype(np.float64)
    atr = np.array([1.5], dtype=np.float64)
    true_arr = np.array([True], dtype=bool)
    false_arr = np.array([False], dtype=bool)

    backend = nb.describe_live_signal_backend("auto")
    resolved = "rust" if backend.startswith("rust:") else "python"

    debounced_samples: list[float] = []
    trailing_samples: list[float] = []

    for _ in range(iters):
        t0 = perf_counter()
        nb.evaluate_debounced_state_native(
            true_arr,
            false_arr,
            false_arr,
            false_arr,
            side="long_only",
            min_hold_bars=1,
            cooldown_bars=0,
        )
        debounced_samples.append(perf_counter() - t0)

        t0 = perf_counter()
        nb.evaluate_trailing_state_native(
            close,
            true_arr,
            false_arr,
            false_arr,
            false_arr,
            atr,
            side="long_only",
            min_hold_bars=1,
            cooldown_bars=0,
            trail_atr_mult=2.0,
        )
        trailing_samples.append(perf_counter() - t0)

    combined = [d + t for d, t in zip(debounced_samples, trailing_samples)]
    return {
        "backend": resolved,
        "backend_description": backend,
        "native_available": nb.native_backend_available(),
        "native_load_error": nb._PYO3_LOAD_ERROR or None,
        "iters": iters,
        "debounced": _stats(debounced_samples),
        "trailing": _stats(trailing_samples),
        "combined_debounced_plus_trailing": _stats(combined),
    }


# ---------------------------------------------------------------------------
# Measurement 2: paper order path (OrderStateMachine transitions)
# ---------------------------------------------------------------------------


def _measure_paper_order_path(iters: int) -> dict:
    """Time OrderStateMachine.transition() — the innermost paper-mode order step."""
    from lumina_quant.live.order_state_machine import (
        STATE_ACKED,
        STATE_FILLED,
        STATE_SUBMITTED,
        OrderStateMachine,
    )

    submit_to_ack: list[float] = []
    ack_to_fill: list[float] = []

    for _ in range(iters):
        t0 = perf_counter()
        OrderStateMachine.transition(STATE_SUBMITTED, STATE_ACKED)
        submit_to_ack.append(perf_counter() - t0)

        t0 = perf_counter()
        OrderStateMachine.transition(STATE_ACKED, STATE_FILLED)
        ack_to_fill.append(perf_counter() - t0)

    combined = [a + b for a, b in zip(submit_to_ack, ack_to_fill)]
    return {
        "note": (
            "OrderStateMachine.transition() only — pure Python dict lookup, "
            "no network, no exchange. Full tick→order latency includes I/O and "
            "is measured in Phase 5 against testnet."
        ),
        "iters": iters,
        "submit_to_ack": _stats(submit_to_ack),
        "ack_to_fill": _stats(ack_to_fill),
        "full_transition_chain": _stats(combined),
    }


# ---------------------------------------------------------------------------
# Measurement 3: replay path proxy (ShadowLiveRunner)
# ---------------------------------------------------------------------------


@dataclass
class _SyntheticEvent:
    timestamp_ns: int
    sequence: int


def _measure_replay_path_proxy(iters: int) -> dict:
    """Time ShadowLiveRunner.run() with a single synthetic event."""
    from lumina_quant.live.shadow_live_runner import ShadowLiveRunner

    runner = ShadowLiveRunner(divergence_tolerance=1e-9)
    event = _SyntheticEvent(timestamp_ns=1_700_000_000_000_000_000, sequence=1)

    samples: list[float] = []
    for _ in range(iters):
        t0 = perf_counter()
        runner.run(baseline_events=[event], candidate_events=[event])
        samples.append(perf_counter() - t0)

    return {
        "note": (
            "ShadowLiveRunner.run() parity-checker overhead on a single synthetic "
            "event — stable_event_sort + timestamp/sequence comparison only. "
            "NOT signal or order latency. Proxy for minimum replay-path overhead."
        ),
        "iters": iters,
        **_stats(samples),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure per-tick latency proxies (no live exchange required)."
    )
    parser.add_argument(
        "--iters", type=int, default=10_000, help="Iterations per measurement (default: 10000)."
    )
    parser.add_argument(
        "--output",
        default="docs/perf/data/bench_tick_latency.json",
        help="Output JSON path (must stay outside the frozen baseline/ dir).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    iters = max(100, int(args.iters))

    print(f"Measuring live signal per-tick latency ({iters} iters)...")
    live_signal = _measure_live_signal_per_tick(iters)
    print(
        f"  backend={live_signal['backend']}  "
        f"combined_median={live_signal['combined_debounced_plus_trailing']['median_ms']:.4f}ms"
    )

    print(f"Measuring paper order path latency ({iters} iters)...")
    paper_order = _measure_paper_order_path(iters)
    print(f"  chain_median={paper_order['full_transition_chain']['median_ms']:.4f}ms")

    print(f"Measuring replay path proxy ({iters} iters)...")
    replay_proxy = _measure_replay_path_proxy(iters)
    print(f"  proxy_median={replay_proxy['median_ms']:.4f}ms")

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "iters": iters,
        "live_signal_per_tick": live_signal,
        "paper_order_path": paper_order,
        "replay_path_proxy": replay_proxy,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
