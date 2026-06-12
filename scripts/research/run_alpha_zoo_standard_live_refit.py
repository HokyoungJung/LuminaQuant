#!/usr/bin/env python3
"""Run the standard Alpha Zoo live-refit policy.

Policy:
- refresh/data coverage is measured from local 1s OHLCV parquet;
- validation is the latest 8 weeks of complete bars;
- Optuna fits/learns only on train during selection;
- after train/validation selection freezes the best hybrid, learned state is
  final-refit on train+validation for the frozen live artifact;
- no locked test/OOS set is reserved or used for live final refit.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumina_quant.alpha_zoo.live_training_policy import (  # noqa: E402
    STANDARD_VALIDATION_WEEKS,
    compute_standard_live_training_plan,
    format_utc,
)
from lumina_quant.alpha_zoo.optuna_hybrid_config import WATCH_SYMBOLS  # noqa: E402
from scripts.research import run_alpha_zoo_integer_leverage_optuna_hybrid_decision as optuna_hybrid  # noqa: E402

DEFAULT_OUTPUT_DIR = optuna_hybrid.DEFAULT_STANDARD_LIVE_REFIT_OUTPUT_DIR
DEFAULT_PRIOR_ARTIFACT = (
    optuna_hybrid.DEFAULT_OUTPUT_DIR
    / "alpha_zoo_integer_leverage_optuna_hybrid_decision_latest.json"
)


def _json_safe(value: Any) -> Any:
    return optuna_hybrid._json_safe(value)


def _symbol_coverage(data_root: Path, symbol: str) -> dict[str, Any]:
    symbol_root = data_root / symbol
    files = sorted(symbol_root.glob("*.parquet"))
    if not files:
        return {"symbol": symbol, "file_count": 0, "row_count": 0, "error": "missing_files"}
    latest = None
    earliest = None
    row_count = 0
    errors = 0
    for path in files:
        try:
            frame = (
                pl.scan_parquet(str(path))
                .select(
                    pl.col("datetime").min().alias("start"),
                    pl.col("datetime").max().alias("end"),
                    pl.len().alias("rows"),
                )
                .collect()
            )
        except Exception:
            errors += 1
            continue
        start = frame["start"][0]
        end = frame["end"][0]
        rows = int(frame["rows"][0] or 0)
        if start is not None:
            earliest = start if earliest is None or start < earliest else earliest
        if end is not None:
            latest = end if latest is None or end > latest else latest
        row_count += rows
    return {
        "symbol": symbol,
        "file_count": len(files),
        "scanned_file_count": len(files) - errors,
        "error_count": errors,
        "row_count": row_count,
        "earliest": None if earliest is None else format_utc(earliest),
        "latest": None if latest is None else format_utc(latest),
    }


def _data_coverage(data_root: Path, symbols: Sequence[str]) -> dict[str, Any]:
    rows = [_symbol_coverage(data_root, symbol) for symbol in symbols]
    latest_values = [row.get("latest") for row in rows if row.get("latest")]
    if not latest_values:
        raise ValueError(f"no local 1s OHLCV coverage found under {data_root}")
    latest_datetimes = [
        datetime.fromisoformat(str(value).replace("Z", "+00:00")) for value in latest_values
    ]
    common_end = min(latest_datetimes)
    return {
        "data_root": str(data_root),
        "symbols": rows,
        "common_data_end_utc": format_utc(common_end),
        "coverage_policy": "minimum_latest_timestamp_across_watch_symbols",
        "runner_peak_rss_mib_at_coverage": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
    }


def _comparison_delta(
    new_row: Mapping[str, Any], old_row: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not old_row:
        return {}
    keys = [
        "train_return",
        "validation_return",
        "train_mdd",
        "validation_mdd",
        "train_return_per_turnover_proxy_bps",
        "validation_return_per_turnover_proxy_bps",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        try:
            out[f"delta_{key}"] = float(new_row.get(key) or 0.0) - float(old_row.get(key) or 0.0)
        except (TypeError, ValueError):
            out[f"delta_{key}"] = None
    return out


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    symbols = tuple(item.strip().upper() for item in str(args.symbols).split(",") if item.strip())
    coverage = _data_coverage(data_root, symbols)
    data_end = str(args.data_end_utc).strip() or str(coverage["common_data_end_utc"])
    plan = compute_standard_live_training_plan(
        data_end_utc=data_end,
        train_start_utc=str(args.train_start_utc),
        validation_weeks=int(args.validation_weeks),
        warmup_ratio=float(args.warmup_ratio),
        bar_minutes=int(args.bar_minutes),
    )
    coverage["standard_live_training_plan"] = plan.as_payload()
    integer_artifact = Path(args.integer_portfolio_artifact).expanduser().resolve()
    prior_artifact = Path(args.prior_artifact).expanduser().resolve()
    payload = optuna_hybrid.build_payload_from_inputs(
        integer_payload=optuna_hybrid.ilp._load_json(integer_artifact),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        integer_artifact_path=integer_artifact,
        data_root=data_root,
        feature_root=Path(args.feature_root).expanduser().resolve(),
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        write_outputs=True,
        split_windows=plan.split_windows(),
        standard_live_refit=True,
        final_refit=True,
        data_coverage=coverage,
        prior_artifact_path=prior_artifact,
    )
    old = payload.get("previous_selected_profile_comparison")
    evidence = payload.get("selection_evidence_profile") or {}
    payload["train_validation_comparison_to_previous_selected"] = _comparison_delta(
        evidence if isinstance(evidence, Mapping) else {}, old if isinstance(old, Mapping) else None
    )
    # Re-write after adding the small comparison delta.
    latest = Path(payload["output_paths"]["latest_json"])
    latest.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    timestamped = Path(payload["output_paths"]["timestamped_json"])
    timestamped.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integer-portfolio-artifact",
        default=str(optuna_hybrid.DEFAULT_INTEGER_PORTFOLIO_ARTIFACT),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data-root", default=str(optuna_hybrid.ilp.DEFAULT_DATA_ROOT))
    parser.add_argument("--feature-root", default=str(optuna_hybrid.ilp.DEFAULT_FEATURE_ROOT))
    parser.add_argument("--symbols", default=",".join(WATCH_SYMBOLS))
    parser.add_argument("--train-start-utc", default="2025-01-01T00:00:00Z")
    parser.add_argument("--data-end-utc", default="")
    parser.add_argument("--validation-weeks", type=int, default=STANDARD_VALIDATION_WEEKS)
    parser.add_argument("--bar-minutes", type=int, default=60)
    parser.add_argument("--warmup-ratio", type=float, default=0.60)
    parser.add_argument("--n-trials", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--prior-artifact", default=str(DEFAULT_PRIOR_ARTIFACT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    payload = build_payload(parse_args(argv))
    evidence = payload.get("selection_evidence_profile") or {}
    selected = payload["selected_optuna_hybrid_profile"]
    print(
        json.dumps(
            _json_safe(
                {
                    "output_paths": payload["output_paths"],
                    "standard_live_training_plan": payload["data_coverage"].get(
                        "standard_live_training_plan"
                    ),
                    "selection_evidence_profile": evidence,
                    "selected_final_refit_profile": selected,
                    "ready_for_real": payload["ready_for_real"],
                    "real_money_execution": payload["real_money_execution"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
