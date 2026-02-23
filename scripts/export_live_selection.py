"""Export deployable parameter selection from sweep/report artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _pick_best_candidate(report: dict, mode: str) -> dict | None:
    candidates = list(report.get("candidates", []) or [])
    if not candidates:
        return None

    hurdle_key = "val" if str(mode).strip().lower() == "live" else "oos"
    passing = [
        row
        for row in candidates
        if bool((((row.get("hurdle_fields") or {}).get(hurdle_key)) or {}).get("pass", False))
    ]
    if passing:
        return max(
            passing,
            key=lambda row: _safe_float(
                (((row.get("hurdle_fields") or {}).get(hurdle_key)) or {}).get("score"),
                -1e9,
            ),
        )

    best_excess = max(
        candidates,
        key=lambda row: _safe_float(
            (((row.get("hurdle_fields") or {}).get(hurdle_key)) or {}).get("excess_return"),
            -1e9,
        ),
    )
    return best_excess


def main() -> None:
    parser = argparse.ArgumentParser(description="Export live deployment selection JSON.")
    parser.add_argument(
        "--sweep-report",
        required=True,
        help="Path to reports/timeframe_sweep_*.json",
    )
    parser.add_argument(
        "--out-dir",
        default="best_optimized_parameters/live",
        help="Output directory for deployment selection file.",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep_report)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep report not found: {sweep_path}")

    with sweep_path.open(encoding="utf-8") as fp:
        sweep = json.load(fp)

    ranked = list(sweep.get("ranked", []) or [])
    if not ranked:
        raise RuntimeError("No successful ranked entries in sweep report.")

    best_entry = ranked[0]
    report_path = Path(str(best_entry.get("report_path", "")))
    if not report_path.exists():
        raise FileNotFoundError(f"Best timeframe report missing: {report_path}")

    with report_path.open(encoding="utf-8") as fp:
        report = json.load(fp)

    mode = str((report.get("split") or {}).get("mode", sweep.get("mode", "oos")))
    best_candidate = _pick_best_candidate(report, mode)
    if best_candidate is None:
        raise RuntimeError("No candidate entries in best timeframe report.")

    export = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "sweep_report": str(sweep_path),
        "best_timeframe": best_entry.get("timeframe"),
        "base_timeframe": sweep.get("base_timeframe"),
        "exchange": sweep.get("exchange"),
        "strategy_set": sweep.get("strategy_set"),
        "selected_report": str(report_path),
        "selected_candidate": {
            "name": best_candidate.get("name"),
            "symbols": best_candidate.get("symbols"),
            "params": best_candidate.get("params"),
            "train_metrics": best_candidate.get("train"),
            "val_metrics": best_candidate.get("val"),
            "oos_metrics": best_candidate.get("oos"),
            "hurdle": best_candidate.get("hurdle"),
            "hurdle_fields": best_candidate.get("hurdle_fields"),
        },
        "ensemble": report.get("ensemble"),
        "benchmark": report.get("benchmark"),
        "split": report.get("split"),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"live_selection_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(export, fp, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
