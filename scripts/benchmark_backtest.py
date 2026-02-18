"""Benchmark backtest runtime and memory usage.

This script creates a reproducible benchmark snapshot for the current codebase.
It is intended for regression gates in CI and local optimization loops.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumina_quant.backtest import Backtest
from lumina_quant.data import HistoricCSVDataHandler
from lumina_quant.execution import SimulatedExecutionHandler
from lumina_quant.portfolio import Portfolio
from strategies import get_default_strategy_params, get_strategy_map

try:
    import psutil
except ImportError:
    psutil = None

STRATEGY_MAP = get_strategy_map()


def _read_yaml(path: str) -> dict[str, Any]:
    """Read a YAML file and return the parsed dictionary."""
    with open(path, encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _load_best_params(strategy_name: str) -> dict[str, Any]:
    """Load optimized params if present, otherwise return defaults."""
    params = get_default_strategy_params(strategy_name)
    param_path = os.path.join(
        "best_optimized_parameters",
        strategy_name,
        "best_params.json",
    )
    meta_path = os.path.join(
        "best_optimized_parameters",
        strategy_name,
        "best_params.meta.json",
    )
    if not os.path.exists(param_path):
        return params
    try:
        with open(param_path, encoding="utf-8") as file:
            loaded = json.load(file)
        if isinstance(loaded, dict):
            params.update(loaded)
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as meta_file:
                    meta = json.load(meta_file)
                basis = str(meta.get("selection_basis", "")).strip().lower()
                if basis != "validation_only":
                    print(
                        "[WARN] best_params metadata indicates non validation-only selection basis "
                        f"for {strategy_name}."
                    )
            except (OSError, ValueError, TypeError):
                print(f"[WARN] Failed to parse best_params metadata for {strategy_name}.")
        else:
            print(f"[WARN] best_params metadata missing for {strategy_name}.")
    except (OSError, ValueError, TypeError):
        pass
    return params


def _get_rss_mb() -> float | None:
    """Return current resident set size in MB when available."""
    if psutil is None:
        return None
    try:
        process = psutil.Process(os.getpid())
    except OSError:
        return None
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


@dataclass(slots=True)
class BenchmarkSample:
    """Single iteration benchmark sample."""

    seconds: float
    peak_tracemalloc_mb: float
    peak_rss_mb: float | None
    bars_processed: int
    bars_per_sec: float


@dataclass(slots=True)
class BenchmarkSummary:
    """Summary statistics for benchmark iterations."""

    strategy: str
    symbols: list[str]
    iterations: int
    warmup: int
    seed: int
    generated_at_utc: str
    python: str
    samples: list[dict[str, Any]]
    median_seconds: float
    mean_seconds: float
    median_bars_per_sec: float
    mean_bars_per_sec: float
    median_peak_tracemalloc_mb: float
    max_peak_rss_mb: float | None
    comparison: dict[str, Any] | None = None


def _load_snapshot(path: str) -> dict[str, Any] | None:
    """Load a previous benchmark snapshot for comparison."""
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError, TypeError):
        return None
    return None


def _build_comparison(current: BenchmarkSummary, previous: dict[str, Any]) -> dict[str, Any]:
    """Build comparison metrics between current and previous benchmark snapshots."""
    prev_median_seconds = float(previous.get("median_seconds", 0.0) or 0.0)
    prev_median_bps = float(previous.get("median_bars_per_sec", 0.0) or 0.0)

    curr_median_seconds = float(current.median_seconds)
    curr_median_bps = float(current.median_bars_per_sec)

    sec_delta_pct = 0.0
    bps_delta_pct = 0.0
    speedup_factor = 0.0

    if prev_median_seconds > 0.0:
        sec_delta_pct = ((curr_median_seconds - prev_median_seconds) / prev_median_seconds) * 100.0
        speedup_factor = (
            prev_median_seconds / curr_median_seconds if curr_median_seconds > 0 else 0.0
        )
    if prev_median_bps > 0.0:
        bps_delta_pct = ((curr_median_bps - prev_median_bps) / prev_median_bps) * 100.0

    return {
        "previous_snapshot": previous.get("generated_at_utc"),
        "previous_median_seconds": prev_median_seconds,
        "current_median_seconds": curr_median_seconds,
        "median_seconds_delta_pct": sec_delta_pct,
        "previous_median_bars_per_sec": prev_median_bps,
        "current_median_bars_per_sec": curr_median_bps,
        "median_bars_per_sec_delta_pct": bps_delta_pct,
        "speedup_factor": speedup_factor,
    }


def _run_once(
    *,
    strategy_name: str,
    symbols: list[str],
    start_date: datetime,
    params: dict[str, Any],
    record_history: bool,
) -> BenchmarkSample:
    """Execute one backtest and collect timing/memory metrics."""
    strategy_cls = STRATEGY_MAP[strategy_name]
    rss_before = _get_rss_mb()
    tracemalloc.start()
    started = time.perf_counter()

    backtest = Backtest(
        csv_dir="data",
        symbol_list=symbols,
        start_date=start_date,
        data_handler_cls=HistoricCSVDataHandler,
        execution_handler_cls=SimulatedExecutionHandler,
        portfolio_cls=Portfolio,
        strategy_cls=strategy_cls,
        strategy_params=params,
        record_history=record_history,
        track_metrics=record_history,
        record_trades=record_history,
    )
    backtest.simulate_trading(output=False)
    ended = time.perf_counter()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    bars_processed = int(backtest.market_events)
    seconds = ended - started
    bars_per_sec = float(bars_processed) / seconds if seconds > 0 else 0.0
    rss_after = _get_rss_mb()
    peak_rss_mb = None
    if rss_before is not None and rss_after is not None:
        peak_rss_mb = max(rss_before, rss_after)

    return BenchmarkSample(
        seconds=seconds,
        peak_tracemalloc_mb=peak / (1024.0 * 1024.0),
        peak_rss_mb=peak_rss_mb,
        bars_processed=bars_processed,
        bars_per_sec=bars_per_sec,
    )


def build_benchmark_summary(args: argparse.Namespace) -> BenchmarkSummary:
    """Run warmup + benchmark loops and build a serialized summary."""
    config_data = _read_yaml(args.config)
    strategy_name = args.strategy or str(
        (config_data.get("optimization", {}) or {}).get("strategy", "RsiStrategy")
    )
    if strategy_name not in STRATEGY_MAP:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    config_symbols = (config_data.get("trading", {}) or {}).get("symbols", ["BTC/USDT"])
    symbols = (
        [token.strip() for token in args.symbols.split(",") if token.strip()]
        if args.symbols
        else list(config_symbols)
    )
    if not symbols:
        raise ValueError("No symbols specified for benchmark.")

    start_date_raw = (config_data.get("backtest", {}) or {}).get("start_date", "2024-01-01")
    start_date = datetime.strptime(str(start_date_raw), "%Y-%m-%d")

    params = _load_best_params(strategy_name)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    for _ in range(args.warmup):
        _run_once(
            strategy_name=strategy_name,
            symbols=symbols,
            start_date=start_date,
            params=params,
            record_history=args.record_history,
        )

    samples: list[BenchmarkSample] = []
    for _ in range(args.iters):
        sample = _run_once(
            strategy_name=strategy_name,
            symbols=symbols,
            start_date=start_date,
            params=params,
            record_history=args.record_history,
        )
        samples.append(sample)

    seconds_list = [sample.seconds for sample in samples]
    bps_list = [sample.bars_per_sec for sample in samples]
    tracemalloc_list = [sample.peak_tracemalloc_mb for sample in samples]
    rss_candidates = [sample.peak_rss_mb for sample in samples if sample.peak_rss_mb is not None]
    max_peak_rss_mb = max(rss_candidates) if rss_candidates else None

    return BenchmarkSummary(
        strategy=strategy_name,
        symbols=symbols,
        iterations=args.iters,
        warmup=args.warmup,
        seed=args.seed,
        generated_at_utc=datetime.now(UTC).isoformat(),
        python=sys.version.split()[0],
        samples=[asdict(sample) for sample in samples],
        median_seconds=statistics.median(seconds_list),
        mean_seconds=statistics.fmean(seconds_list),
        median_bars_per_sec=statistics.median(bps_list),
        mean_bars_per_sec=statistics.fmean(bps_list),
        median_peak_tracemalloc_mb=statistics.median(tracemalloc_list),
        max_peak_rss_mb=max_peak_rss_mb,
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Benchmark LuminaQuant backtest runtime.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols override, e.g. BTC/USDT,ETH/USDT.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=sorted(STRATEGY_MAP.keys()),
        help="Strategy class name override.",
    )
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed value.")
    parser.add_argument(
        "--record-history",
        action="store_true",
        help="Keep full portfolio history during the benchmark run.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("reports", "benchmarks", "baseline_snapshot.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--compare-to",
        default="",
        help="Optional path to previous benchmark JSON snapshot for delta report.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the benchmark and write a summary JSON artifact."""
    args = _parse_args()
    summary = build_benchmark_summary(args)
    previous = _load_snapshot(args.compare_to)
    if previous is not None:
        summary.comparison = _build_comparison(summary, previous)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)

    print(json.dumps(asdict(summary), indent=2))
    print(f"Saved benchmark snapshot: {args.output}")


if __name__ == "__main__":
    main()
