"""Generic parameter optimizers for the public sample strategy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import optuna
from optuna.trial import TrialState

from lumina_quant.backtesting import BacktestResult, run_backtest
from lumina_quant.data import load_ohlcv_csv
from lumina_quant.metrics import MetricsSummary, compute_metrics
from lumina_quant.models import Bar
from lumina_quant.sample_strategy import MovingAverageCrossStrategy


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    fast_window: int
    slow_window: int
    metrics: MetricsSummary
    result: BacktestResult
    trial_number: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "metrics": self.metrics.to_dict(),
            "result": self.result.to_dict(),
            "trial_number": self.trial_number,
        }


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    objective: str
    method: str
    best: OptimizationCandidate
    candidates: list[OptimizationCandidate]

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "method": self.method,
            "best": self.best.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_count": len(self.candidates),
        }


def _parse_int_grid(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def _evaluate_candidate(
    bars: list[Bar],
    *,
    fast_window: int,
    slow_window: int,
    initial_cash: float,
    fee_bps: float,
    trial_number: int | None = None,
) -> OptimizationCandidate:
    strategy = MovingAverageCrossStrategy(fast_window=fast_window, slow_window=slow_window)
    result = run_backtest(bars, strategy, initial_cash=initial_cash, fee_bps=fee_bps)
    return OptimizationCandidate(
        fast_window=fast_window,
        slow_window=slow_window,
        metrics=compute_metrics(result),
        result=result,
        trial_number=trial_number,
    )


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
        candidates.append(
            _evaluate_candidate(
                bars,
                fast_window=fast_window,
                slow_window=slow_window,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
            )
        )
    if not candidates:
        raise ValueError("no valid fast/slow window candidates")
    best = max(candidates, key=lambda item: (item.metrics.score, item.metrics.total_return))
    return OptimizationResult(
        objective="generic_score",
        method="grid",
        best=best,
        candidates=candidates,
    )


def run_optuna_optimization(
    data_path: str,
    *,
    fast_grid: str = "2,3,4",
    slow_grid: str = "6,8,10",
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
    n_trials: int = 16,
    sampler_seed: int = 7,
) -> OptimizationResult:
    """Run Optuna over the same public-safe sample strategy search space."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    bars = load_ohlcv_csv(data_path)
    fast_values = _parse_int_grid(fast_grid)
    slow_values = _parse_int_grid(slow_grid)
    candidates: list[OptimizationCandidate] = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        fast_window = int(trial.suggest_categorical("fast_window", fast_values))
        slow_window = int(trial.suggest_categorical("slow_window", slow_values))
        if slow_window <= fast_window:
            trial.set_user_attr("rejection_reason", "slow_window_must_exceed_fast_window")
            raise optuna.TrialPruned()
        candidate = _evaluate_candidate(
            bars,
            fast_window=fast_window,
            slow_window=slow_window,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            trial_number=trial.number,
        )
        trial.set_user_attr("total_return", candidate.metrics.total_return)
        trial.set_user_attr("max_drawdown", candidate.metrics.max_drawdown)
        trial.set_user_attr("trade_count", candidate.metrics.trade_count)
        candidates.append(candidate)
        return candidate.metrics.score

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    complete_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not complete_trials or not candidates:
        raise ValueError("optuna produced no complete public sample candidates")
    best = max(candidates, key=lambda item: (item.metrics.score, item.metrics.total_return))
    return OptimizationResult(
        objective="generic_score",
        method="optuna",
        best=best,
        candidates=sorted(candidates, key=lambda item: item.trial_number or -1),
    )
