"""Run the focused production-safe carry/trend retune lane.

This is a reproducible, low-memory wrapper around ``run_research_candidates.py`` for
just the production-marked carry/trend factor-rotation candidates. It keeps the
execution contract explicit (sequential, thread-capped, exact split, manifest-driven)
without reopening the full article pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GROUP_ROOT = (
    ROOT
    / "var"
    / "reports"
    / "exact_window_backtests"
    / "followup_status"
    / "portfolio_incumbent_autoresearch_grouped"
)
DEFAULT_MANIFEST = (
    GROUP_ROOT
    / "carry_trend_production_retune_current"
    / "carry_trend_production_manifest_latest.json"
)
DEFAULT_OUTPUT_DIR = GROUP_ROOT / "carry_trend_production_retune_current" / "research_run"
DEFAULT_SCORE_CONFIG = ROOT / "configs" / "score_config.example.json"
DEFAULT_TIMEFRAMES = ["1h", "4h"]
DEFAULT_BASE_TIMEFRAME = "1m"

LOW_MEMORY_ENV = {
    "POLARS_MAX_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "LQ_BACKTEST_LOW_MEMORY": "1",
    "LQ_AUTO_COLLECT_DB": "0",
    "PYTHONUNBUFFERED": "1",
}


def build_command(
    *,
    manifest: Path,
    output_dir: Path,
    score_config: Path,
    symbols: list[str],
    timeframes: list[str],
    base_timeframe: str,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    oos_start: str,
    oos_end: str,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_research_candidates.py"),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--base-timeframe",
        str(base_timeframe),
        "--timeframes",
        *[str(item) for item in timeframes],
        "--symbols",
        *[str(item) for item in symbols],
        "--train-start",
        str(train_start),
        "--train-end",
        str(train_end),
        "--validation-start",
        str(validation_start),
        "--validation-end",
        str(validation_end),
        "--oos-start",
        str(oos_start),
        "--oos-end",
        str(oos_end),
        "--skip-coverage-rebuild",
        "--score-config",
        str(score_config),
    ]


def run_retune(**kwargs: Any) -> subprocess.CompletedProcess[str]:
    command = build_command(**kwargs)
    env = os.environ.copy()
    env.update(LOW_MEMORY_ENV)
    return subprocess.run(
        command, cwd=str(ROOT), env=env, text=True, capture_output=True, check=False
    )


def _load_manifest_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _manifest_symbols_and_timeframes(path: Path) -> tuple[list[str], list[str]]:
    payload = _load_manifest_payload(path)
    candidates = [
        dict(row) for row in list(payload.get("candidates") or []) if isinstance(row, dict)
    ]
    symbol_order: list[str] = []
    timeframe_order: list[str] = []
    for row in candidates:
        for symbol in list(row.get("symbols") or []):
            token = str(symbol).strip()
            if token and token not in symbol_order:
                symbol_order.append(token)
        timeframe = str(row.get("strategy_timeframe") or row.get("timeframe") or "").strip()
        if timeframe and timeframe not in timeframe_order:
            timeframe_order.append(timeframe)
    return symbol_order, timeframe_order


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-config", type=Path, default=DEFAULT_SCORE_CONFIG)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframes", nargs="+", default=None)
    parser.add_argument("--base-timeframe", default=DEFAULT_BASE_TIMEFRAME)
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--validation-start", default="2026-01-01")
    parser.add_argument("--validation-end", default="2026-02-28")
    parser.add_argument("--oos-start", default="2026-03-01")
    parser.add_argument("--oos-end", default="2026-04-14")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    manifest_symbols, manifest_timeframes = _manifest_symbols_and_timeframes(manifest_path)
    result = run_retune(
        manifest=manifest_path,
        output_dir=Path(args.output_dir).resolve(),
        score_config=Path(args.score_config).resolve(),
        symbols=[str(item) for item in list(args.symbols or manifest_symbols)],
        timeframes=[
            str(item) for item in list(args.timeframes or manifest_timeframes or DEFAULT_TIMEFRAMES)
        ],
        base_timeframe=str(args.base_timeframe),
        train_start=str(args.train_start),
        train_end=str(args.train_end),
        validation_start=str(args.validation_start),
        validation_end=str(args.validation_end),
        oos_start=str(args.oos_start),
        oos_end=str(args.oos_end),
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
