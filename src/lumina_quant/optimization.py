"""Generic parameter-grid optimizer for the public sample strategy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from lumina_quant.backtesting import BacktestResult, run_backtest
from lumina_quant.data import load_ohlcv_csv
from lumina_quant.metrics import MetricsSummary, compute_metrics
from lumina_quant.sample_strategy import MovingAverageCrossStrategy


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    fast_window: int
    slow_window: int
    metrics: MetricsSummary
    result: BacktestResult

    def to_dict(self) -> dict[str, object]:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "metrics": self.metrics.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    objective: str
    best: OptimizationCandidate
    candidates: list[OptimizationCandidate]

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "best": self.best.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_count": len(self.candidates),
        }


def _parse_int_grid(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def run_grid_optimization(
    data_path: str,
    *,
    fast_grid: str = "2,3,4",
    slow_grid: str = "6,8,10",
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> OptimizationResult:
    """Search generic moving-average windows using a deterministic score."""
    bars = load_ohlcv_csv(data_path)
    candidates: list[OptimizationCandidate] = []
    for fast_window, slow_window in product(_parse_int_grid(fast_grid), _parse_int_grid(slow_grid)):
        if slow_window <= fast_window:
            continue
        strategy = MovingAverageCrossStrategy(fast_window=fast_window, slow_window=slow_window)
        result = run_backtest(bars, strategy, initial_cash=initial_cash, fee_bps=fee_bps)
        candidates.append(
            OptimizationCandidate(
                fast_window=fast_window,
                slow_window=slow_window,
                metrics=compute_metrics(result),
                result=result,
            )
        )
    if not candidates:
        raise ValueError("no valid fast/slow window candidates")
    best = max(candidates, key=lambda item: (item.metrics.score, item.metrics.total_return))
    return OptimizationResult(objective="generic_score", best=best, candidates=candidates)
