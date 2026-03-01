"""Export diagnostics for advanced factor sleeves from candidate research output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _stream_to_array(stream: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_safe_float(row.get("v"), 0.0) for row in stream], dtype=float)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    n = min(x.size, y.size)
    if n < 8:
        return 0.0
    xa = x[-n:]
    ya = y[-n:]
    sx = float(np.std(xa, ddof=1))
    sy = float(np.std(ya, ddof=1))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    value = float(np.corrcoef(xa, ya)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export factor diagnostics from candidate research report.")
    parser.add_argument("--report", default="reports/candidate_research_latest.json")
    parser.add_argument("--output-dir", default="reports")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report_path = Path(args.report).resolve()
    if not report_path.exists():
        raise RuntimeError(f"Report file not found: {report_path}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = [dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)]
    if not candidates:
        raise RuntimeError("No candidates found in report.")

    by_family: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_timeframe: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    oos_streams: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for row in candidates:
        family = str(row.get("family") or "other")
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "")
        oos = dict(row.get("oos") or {})

        by_family[family]["count"] += 1
        by_family[family]["avg_sharpe_sum"] += _safe_float(oos.get("sharpe"), 0.0)
        by_family[family]["avg_return_sum"] += _safe_float(oos.get("return"), 0.0)
        by_family[family]["avg_dsr_sum"] += _safe_float(oos.get("deflated_sharpe"), 0.0)

        by_timeframe[timeframe]["count"] += 1
        by_timeframe[timeframe]["avg_sharpe_sum"] += _safe_float(oos.get("sharpe"), 0.0)
        by_timeframe[timeframe]["avg_return_sum"] += _safe_float(oos.get("return"), 0.0)

        cid = str(row.get("candidate_id") or row.get("name") or "")
        stream = list((row.get("return_streams") or {}).get("oos") or [])
        if cid and stream:
            oos_streams[cid] = _stream_to_array(stream)
            labels[cid] = str(row.get("name") or cid)

    family_summary = {}
    for family, stats in by_family.items():
        count = max(1.0, stats["count"])
        family_summary[family] = {
            "count": int(stats["count"]),
            "avg_oos_sharpe": float(stats["avg_sharpe_sum"] / count),
            "avg_oos_return": float(stats["avg_return_sum"] / count),
            "avg_deflated_sharpe": float(stats["avg_dsr_sum"] / count),
        }

    timeframe_summary = {}
    for timeframe, stats in by_timeframe.items():
        count = max(1.0, stats["count"])
        timeframe_summary[timeframe] = {
            "count": int(stats["count"]),
            "avg_oos_sharpe": float(stats["avg_sharpe_sum"] / count),
            "avg_oos_return": float(stats["avg_return_sum"] / count),
        }

    # Top correlation pairs across candidate OOS streams.
    corr_pairs: list[dict[str, Any]] = []
    ids = list(oos_streams.keys())
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            corr = _corr(oos_streams[left], oos_streams[right])
            corr_pairs.append(
                {
                    "left_id": left,
                    "left_name": labels.get(left, left),
                    "right_id": right,
                    "right_name": labels.get(right, right),
                    "corr": float(corr),
                }
            )
    corr_pairs.sort(key=lambda row: abs(float(row["corr"])), reverse=True)

    diagnostics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_report": str(report_path),
        "family_summary": dict(sorted(family_summary.items())),
        "timeframe_summary": dict(sorted(timeframe_summary.items())),
        "top_correlation_pairs": corr_pairs[:50],
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"factor_diagnostics_{stamp}.json"
    latest = output_dir / "factor_diagnostics_latest.json"

    out_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    latest.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Saved latest: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
