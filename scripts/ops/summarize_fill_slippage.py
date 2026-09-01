from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, float(pct))) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _derive_slippage_bps(payload: dict[str, Any]) -> float | None:
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("realized_slippage_bps") is not None:
        try:
            return float(metadata["realized_slippage_bps"])
        except Exception:
            return None
    reference = (
        metadata.get("reference_price")
        or metadata.get("expected_mid_price")
        or metadata.get("decision_price")
    )
    fill_price = payload.get("fill_price") or metadata.get("fill_price")
    side = str(payload.get("side") or metadata.get("side") or "").upper()
    try:
        reference = float(reference)
        fill_price = float(fill_price)
    except Exception:
        return None
    if reference <= 0.0 or fill_price <= 0.0:
        return None
    if side == "BUY":
        return ((fill_price - reference) / reference) * 10_000.0
    return ((reference - fill_price) / reference) * 10_000.0


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symbol[str(row.get("symbol") or "UNKNOWN")].append(row)

    def _section(items: list[dict[str, Any]]) -> dict[str, Any]:
        slips = [
            value for value in (_derive_slippage_bps(item) for item in items) if value is not None
        ]
        metadata_rows = [dict(item.get("metadata") or {}) for item in items]
        timeout_count = sum(1 for m in metadata_rows if bool(m.get("timeout_flag")))
        partial_fill_count = sum(1 for m in metadata_rows if bool(m.get("partial_fill_flag")))
        cancel_count = sum(1 for m in metadata_rows if bool(m.get("cancel_flag")))
        missing_slippage_count = len(items) - len(slips)
        return {
            "count": len(items),
            "missing_slippage_count": missing_slippage_count,
            "median_slippage_bps": (median(slips) if slips else None),
            "p90_slippage_bps": _percentile(slips, 0.90),
            "p95_slippage_bps": _percentile(slips, 0.95),
            "max_slippage_bps": (max(slips) if slips else None),
            "timeout_count": timeout_count,
            "partial_fill_count": partial_fill_count,
            "cancel_count": cancel_count,
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "artifact_kind": "fill_slippage_summary",
        "overall": _section(rows),
        "by_symbol": {symbol: _section(items) for symbol, items in sorted(by_symbol.items())},
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        rows.append(json.loads(token))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize fill slippage from JSONL exports.")
    parser.add_argument("--input-jsonl", required=True)
    args = parser.parse_args(argv)

    rows = _read_jsonl(Path(args.input_jsonl))
    payload = build_summary(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
