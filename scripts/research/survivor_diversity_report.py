#!/usr/bin/env python
"""Survivor-diversity diagnostic (DATA-PC handoff).

Runs on the DATA PC where real funding-charged returns exist. Given the set of
SURVIVOR sleeves (net-Sharpe > 0 under the funding-correct realistic cost model),
it reports how much genuine diversification the survivors actually offer, so the
deferred allocation levers (HRP/ERC, vol overlay, ...) are only pursued when the
upside is REAL rather than a correlated illusion:

* survivor count;
* the pairwise return-correlation matrix;
* the number of genuinely LOW-CORRELATION clusters (|corr| < ``--low-corr-threshold``,
  default 0.3) -- reusing ``portfolio.optimizer_core.cluster_by_correlation``;
* the CRASH-period correlation (correlation conditioned on benchmark deep-drawdown
  bars -- does trend + carry still de-correlate when everything sells off?);
* a VERDICT: diversification upside BOUNDED (< 3 low-corr clusters OR high
  crash-correlation) vs REAL (>= 3 low-corr clusters AND low crash-correlation).

The survivor net returns are read from a CSV / parquet / JSON path so the script is
runnable on the data PC. When the inputs are GROSS, ``--apply-cost-drag`` nets them
with the realistic cost regime via ``research.cost_realism.apply_cost_drag`` before
the diagnostic. No statistics are reinvented: correlation/clustering come from the
existing portfolio machinery; cost-drag from the existing cost-realism machinery.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lumina_quant.portfolio.optimizer_core import cluster_by_correlation
from lumina_quant.research.cost_realism import REALISTIC_REGIME, apply_cost_drag

_LOW_CORR_THRESHOLD_DEFAULT = 0.30
_CRASH_DRAWDOWN_THRESHOLD_DEFAULT = -0.10
_CRASH_CORR_CEILING_DEFAULT = 0.60
_MIN_LOW_CORR_CLUSTERS_DEFAULT = 3


def _as_matrix(returns_by_id: Mapping[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    """Stack the survivor return series into a ``(n_survivors, n_bars)`` matrix.

    Series are truncated to the shortest common length so the correlation is
    computed over a shared, index-aligned window.
    """
    ids = list(returns_by_id.keys())
    series = [np.asarray(returns_by_id[key], dtype=float).reshape(-1) for key in ids]
    if not series:
        return ids, np.zeros((0, 0), dtype=float)
    length = min(int(arr.size) for arr in series)
    matrix = np.asarray([arr[:length] for arr in series], dtype=float)
    return ids, matrix


def _correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Signed Pearson correlation matrix; zero-variance rows correlate at 0."""
    n = int(matrix.shape[0])
    if n == 0 or matrix.shape[1] < 2:
        return np.zeros((n, n), dtype=float)
    corr = np.corrcoef(matrix)
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _mean_offdiagonal_correlation(matrix: np.ndarray) -> float:
    """Mean of the upper-triangle (i < j) pairwise correlations."""
    n = int(matrix.shape[0])
    if n < 2 or matrix.shape[1] < 2:
        return 0.0
    corr = _correlation_matrix(matrix)
    iu = np.triu_indices(n, k=1)
    values = corr[iu]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _benchmark_crash_mask(
    matrix: np.ndarray,
    *,
    benchmark: Sequence[float] | None,
    crash_drawdown_threshold: float,
) -> np.ndarray:
    """Boolean mask of benchmark DEEP-DRAWDOWN bars.

    The benchmark defaults to the equal-weight survivor portfolio when none is
    supplied. Crash bars are those whose running-peak drawdown is at/below
    ``crash_drawdown_threshold``. When too few (< 8) such bars exist the mask falls
    back to the worst-decile benchmark-return bars so a crash correlation can still
    be estimated.
    """
    n_bars = int(matrix.shape[1])
    if n_bars < 2:
        return np.zeros(n_bars, dtype=bool)
    if benchmark is None:
        bench = matrix.mean(axis=0)
    else:
        bench = np.asarray(benchmark, dtype=float).reshape(-1)[:n_bars]
    if bench.size < n_bars:
        bench = np.concatenate([bench, np.zeros(n_bars - bench.size, dtype=float)])

    equity = np.cumprod(1.0 + bench)
    running_peak = np.maximum.accumulate(equity)
    drawdown = np.where(running_peak > 0.0, equity / running_peak - 1.0, 0.0)
    mask = drawdown <= float(crash_drawdown_threshold)
    if int(mask.sum()) >= 8:
        return mask
    # Fallback: worst-decile bars by raw benchmark return.
    k = max(8, round(0.10 * n_bars))
    k = min(k, n_bars)
    worst = np.argsort(bench)[:k]
    fallback = np.zeros(n_bars, dtype=bool)
    fallback[worst] = True
    return fallback


def analyze_survivor_diversity(
    returns_by_id: Mapping[str, Sequence[float]],
    *,
    benchmark: Sequence[float] | None = None,
    low_corr_threshold: float = _LOW_CORR_THRESHOLD_DEFAULT,
    crash_drawdown_threshold: float = _CRASH_DRAWDOWN_THRESHOLD_DEFAULT,
    crash_corr_ceiling: float = _CRASH_CORR_CEILING_DEFAULT,
    min_low_corr_clusters: int = _MIN_LOW_CORR_CLUSTERS_DEFAULT,
) -> dict[str, Any]:
    """Diagnose survivor diversification. Pure/data-free; returns a report dict."""
    ids, matrix = _as_matrix(returns_by_id)
    survivor_count = len(ids)

    corr = _correlation_matrix(matrix)
    correlation_matrix = {
        ids[i]: {ids[j]: float(corr[i, j]) for j in range(survivor_count)}
        for i in range(survivor_count)
    }

    # Low-correlation clustering reuses the repo's correlation clusterer: series that
    # correlate at |corr| >= threshold merge, so the CLUSTER COUNT is the number of
    # genuinely low-correlation (independent) bets.
    stream_map: dict[str, list[dict[str, Any]]] = {
        ids[i]: [{"v": float(v)} for v in matrix[i]] for i in range(survivor_count)
    }
    clusters = cluster_by_correlation(ids, stream_map, threshold=float(low_corr_threshold))
    low_corr_cluster_count = len(clusters)

    crash_mask = _benchmark_crash_mask(
        matrix, benchmark=benchmark, crash_drawdown_threshold=crash_drawdown_threshold
    )
    crash_bar_count = int(crash_mask.sum())
    if survivor_count >= 2 and crash_bar_count >= 8:
        crash_period_correlation = _mean_offdiagonal_correlation(matrix[:, crash_mask])
        crash_corr_measurable = True
    else:
        crash_period_correlation = 0.0
        crash_corr_measurable = False

    reasons: list[str] = []
    if low_corr_cluster_count < int(min_low_corr_clusters):
        reasons.append("insufficient_low_correlation_clusters")
    if crash_corr_measurable and crash_period_correlation >= float(crash_corr_ceiling):
        reasons.append("high_crash_period_correlation")
    verdict = "BOUNDED" if reasons else "REAL"

    return {
        "survivor_count": survivor_count,
        "survivor_ids": ids,
        "bars": int(matrix.shape[1]) if matrix.size else 0,
        "low_corr_threshold": float(low_corr_threshold),
        "low_corr_cluster_count": low_corr_cluster_count,
        "clusters": [list(cluster) for cluster in clusters],
        "correlation_matrix": correlation_matrix,
        "mean_pairwise_correlation": _mean_offdiagonal_correlation(matrix),
        "crash_drawdown_threshold": float(crash_drawdown_threshold),
        "crash_bar_count": crash_bar_count,
        "crash_corr_measurable": crash_corr_measurable,
        "crash_period_correlation": crash_period_correlation,
        "crash_corr_ceiling": float(crash_corr_ceiling),
        "min_low_corr_clusters": int(min_low_corr_clusters),
        "verdict": verdict,
        "verdict_reasons": reasons,
    }


# --------------------------------------------------------------------------- IO
def _load_returns(
    path: Path, *, benchmark_col: str
) -> tuple[dict[str, list[float]], list[float] | None]:
    """Load survivor returns (and optional benchmark) from CSV / parquet / JSON."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        raw = payload.get("returns") if isinstance(payload.get("returns"), Mapping) else payload
        returns = {str(k): [float(x) for x in v] for k, v in raw.items() if k != benchmark_col}
        bench = payload.get("benchmark") or raw.get(benchmark_col)
        benchmark = [float(x) for x in bench] if isinstance(bench, Sequence) else None
        return returns, benchmark
    if suffix in {".parquet", ".pq"}:
        import polars as pl

        frame = pl.read_parquet(path)
        columns = {col: [float(x) for x in frame[col].to_list()] for col in frame.columns}
    elif suffix == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            columns = {name: [] for name in fieldnames}
            for record in reader:
                for name in fieldnames:
                    cell = record.get(name, "")
                    columns[name].append(float(cell) if str(cell).strip() else 0.0)
    else:
        raise ValueError(f"unsupported input suffix: {suffix!r} (use .csv/.parquet/.json)")

    benchmark = columns.pop(benchmark_col, None)
    return columns, benchmark


def _maybe_apply_cost_drag(
    returns: dict[str, list[float]], *, turnover: float, funding_periods_per_step: float
) -> dict[str, list[float]]:
    """Net GROSS survivor returns with the realistic cost regime (reused machinery)."""
    netted: dict[str, list[float]] = {}
    for key, series in returns.items():
        drag = apply_cost_drag(
            np.asarray(series, dtype=float),
            turnover=turnover,
            regime=REALISTIC_REGIME,
            funding_periods_per_step=funding_periods_per_step,
        )
        netted[key] = [float(x) for x in drag]
    return netted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="CSV/parquet/JSON of survivor net returns"
    )
    parser.add_argument(
        "--benchmark-col", default="benchmark", help="Column/key holding the benchmark series"
    )
    parser.add_argument("--low-corr-threshold", type=float, default=_LOW_CORR_THRESHOLD_DEFAULT)
    parser.add_argument(
        "--crash-drawdown-threshold", type=float, default=_CRASH_DRAWDOWN_THRESHOLD_DEFAULT
    )
    parser.add_argument("--crash-corr-ceiling", type=float, default=_CRASH_CORR_CEILING_DEFAULT)
    parser.add_argument("--min-low-corr-clusters", type=int, default=_MIN_LOW_CORR_CLUSTERS_DEFAULT)
    parser.add_argument(
        "--apply-cost-drag",
        action="store_true",
        help="Treat inputs as GROSS and net them with the realistic cost regime first",
    )
    parser.add_argument(
        "--turnover", type=float, default=1.0, help="Per-bar turnover for --apply-cost-drag"
    )
    parser.add_argument(
        "--funding-periods-per-step",
        type=float,
        default=0.0,
        help="Funding intervals per bar for --apply-cost-drag (e.g. 3.0 eight-hour intervals per daily bar)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional path to write the JSON report"
    )
    args = parser.parse_args(argv)

    returns, benchmark = _load_returns(args.input, benchmark_col=args.benchmark_col)
    if args.apply_cost_drag:
        returns = _maybe_apply_cost_drag(
            returns,
            turnover=args.turnover,
            funding_periods_per_step=args.funding_periods_per_step,
        )

    report = analyze_survivor_diversity(
        returns,
        benchmark=benchmark,
        low_corr_threshold=args.low_corr_threshold,
        crash_drawdown_threshold=args.crash_drawdown_threshold,
        crash_corr_ceiling=args.crash_corr_ceiling,
        min_low_corr_clusters=args.min_low_corr_clusters,
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
