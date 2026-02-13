"""Compare benchmark snapshots and validate relative SLO targets."""

from __future__ import annotations

import argparse
import json


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _pct_improvement(old: float, new: float, *, lower_is_better: bool) -> float:
    if old == 0:
        return 0.0
    if lower_is_better:
        return ((old - new) / old) * 100.0
    return ((new - old) / old) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark snapshots.")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark JSON.")
    parser.add_argument("--candidate", required=True, help="Candidate benchmark JSON.")
    parser.add_argument("--min-runtime-improvement", type=float, default=30.0)
    parser.add_argument("--min-memory-improvement", type=float, default=20.0)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)

    runtime_gain = _pct_improvement(
        float(baseline["median_seconds"]),
        float(candidate["median_seconds"]),
        lower_is_better=True,
    )
    memory_gain = _pct_improvement(
        float(baseline["median_peak_tracemalloc_mb"]),
        float(candidate["median_peak_tracemalloc_mb"]),
        lower_is_better=True,
    )

    print(f"Runtime improvement: {runtime_gain:.2f}%")
    print(f"Memory improvement: {memory_gain:.2f}%")

    if runtime_gain < args.min_runtime_improvement:
        print("Runtime SLO failed.")
        return 2
    if memory_gain < args.min_memory_improvement:
        print("Memory SLO failed.")
        return 3
    print("Performance SLO passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
