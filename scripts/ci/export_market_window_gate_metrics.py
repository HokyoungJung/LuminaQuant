#!/usr/bin/env python3
"""Export MARKET_WINDOW rollout gate metrics snapshot from ndjson logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_bool(value: str) -> bool:
    token = str(value or "").strip().lower()
    return token in {"1", "true", "yes", "on"}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _load_rows(input_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not input_path.exists():
        return rows
    for line in input_path.read_text(encoding="utf-8").splitlines():
        raw = str(line).strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(payload)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MARKET_WINDOW rollout gate metrics.")
    parser.add_argument("--input", required=True, help="Input metrics ndjson path.")
    parser.add_argument("--output", required=True, help="Output JSON snapshot path.")
    parser.add_argument("--window-hours", type=int, default=24, help="Window size in hours.")
    parser.add_argument(
        "--require-flag",
        default="true",
        help="Filter rows by parity_v2_enabled boolean (true/false).",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.input))
    require_flag = _parse_bool(args.require_flag)
    filtered = [row for row in rows if bool(row.get("parity_v2_enabled")) is require_flag]
    max_ts = max((int(row.get("timestamp_ms", 0) or 0) for row in filtered), default=0)
    window_ms = max(1, int(args.window_hours)) * 3_600_000
    cutoff_ts = max(0, int(max_ts - window_ms))
    window_rows = [row for row in filtered if int(row.get("timestamp_ms", 0) or 0) >= cutoff_ts]

    payload_bytes = [float(row.get("payload_bytes", 0) or 0) for row in window_rows]
    queue_lag_ms = [float(row.get("queue_lag_ms", 0) or 0) for row in window_rows]
    fail_fast_incidents = sum(
        1 for row in window_rows if bool(row.get("fail_fast_incident", False))
    )

    snapshot = {
        "input_path": str(args.input),
        "require_flag": bool(require_flag),
        "window_hours": int(args.window_hours),
        "sample_count": len(window_rows),
        "window_start_timestamp_ms": int(cutoff_ts),
        "window_end_timestamp_ms": int(max_ts),
        "p95_payload_bytes": float(_percentile(payload_bytes, 0.95)),
        "p95_queue_lag_ms": float(_percentile(queue_lag_ms, 0.95)),
        "fail_fast_incidents": int(fail_fast_incidents),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
