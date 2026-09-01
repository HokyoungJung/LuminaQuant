#!/usr/bin/env python3
"""Gate checker for MARKET_WINDOW parity rollout metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_snapshot(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot at {path} must be a JSON object.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MARKET_WINDOW rollout gates.")
    parser.add_argument("--baseline", required=True, help="Baseline snapshot JSON path.")
    parser.add_argument("--canary", required=True, help="Canary snapshot JSON path.")
    parser.add_argument("--max-p95-payload-bytes", type=float, required=True)
    parser.add_argument("--max-queue-lag-increase-pct", type=float, required=True)
    parser.add_argument("--max-fail-fast-incidents", type=int, required=True)
    args = parser.parse_args()

    baseline = _load_snapshot(args.baseline)
    canary = _load_snapshot(args.canary)

    baseline_lag = float(baseline.get("p95_queue_lag_ms", 0.0) or 0.0)
    canary_lag = float(canary.get("p95_queue_lag_ms", 0.0) or 0.0)
    if baseline_lag <= 0.0:
        queue_lag_increase_pct = 0.0 if canary_lag <= 0.0 else 100.0
    else:
        queue_lag_increase_pct = ((canary_lag - baseline_lag) / baseline_lag) * 100.0

    canary_payload = float(canary.get("p95_payload_bytes", 0.0) or 0.0)
    canary_fail_fast = int(canary.get("fail_fast_incidents", 0) or 0)

    checks = {
        "payload_p95_ok": canary_payload <= float(args.max_p95_payload_bytes),
        "queue_lag_ok": queue_lag_increase_pct <= float(args.max_queue_lag_increase_pct),
        "fail_fast_ok": canary_fail_fast <= int(args.max_fail_fast_incidents),
    }
    passed = all(checks.values())

    decision = {
        "passed": bool(passed),
        "checks": checks,
        "baseline": {
            "p95_payload_bytes": float(baseline.get("p95_payload_bytes", 0.0) or 0.0),
            "p95_queue_lag_ms": baseline_lag,
            "fail_fast_incidents": int(baseline.get("fail_fast_incidents", 0) or 0),
        },
        "canary": {
            "p95_payload_bytes": canary_payload,
            "p95_queue_lag_ms": canary_lag,
            "fail_fast_incidents": canary_fail_fast,
        },
        "thresholds": {
            "max_p95_payload_bytes": float(args.max_p95_payload_bytes),
            "max_queue_lag_increase_pct": float(args.max_queue_lag_increase_pct),
            "max_fail_fast_incidents": int(args.max_fail_fast_incidents),
        },
        "derived": {
            "queue_lag_increase_pct": float(queue_lag_increase_pct),
        },
    }
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
