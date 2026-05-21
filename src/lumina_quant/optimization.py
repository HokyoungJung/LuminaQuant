"""Generic parameter optimizers for public strategies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import optuna
from optuna.trial import TrialState

from lumina_quant.backtesting import BacktestResult, run_backtest
from lumina_quant.config import SearchParam
from lumina_quant.data import load_ohlcv_csv
from lumina_quant.metrics import MetricsSummary, compute_metrics
from lumina_quant.models import Bar
from lumina_quant.strategy_loader import DEFAULT_STRATEGY_REF, build_strategy


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    params: dict[str, Any]
    metrics: MetricsSummary
    result: BacktestResult
    trial_number: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "params": dict(self.params),
            "metrics": self.metrics.to_dict(),
            "result": self.result.to_dict(),
            "trial_number": self.trial_number,
        }
        if "fast_window" in self.params:
            payload["fast_window"] = self.params["fast_window"]
        if "slow_window" in self.params:
            payload["slow_window"] = self.params["slow_window"]
        return payload


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    objective: str
    method: str
    strategy_ref: str
    best: OptimizationCandidate
    candidates: list[OptimizationCandidate]

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "method": self.method,
            "strategy_ref": self.strategy_ref,
            "best": self.best.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_count": len(self.candidates),
        }


def _parse_int_grid(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def _default_search_space(fast_grid: str, slow_grid: str) -> dict[str, SearchParam]:
    return {
        "fast_window": SearchParam(kind="categorical", choices=tuple(_parse_int_grid(fast_grid))),
        "slow_window": SearchParam(kind="categorical", choices=tuple(_parse_int_grid(slow_grid))),
    }


def _evaluate_candidate(
    bars: list[Bar],
    *,
    strategy_ref: str,
    params: dict[str, Any],
    initial_cash: float,
    fee_bps: float,
    trial_number: int | None = None,
) -> OptimizationCandidate:
    strategy = build_strategy(strategy_ref, params)
    result = run_backtest(bars, strategy, initial_cash=initial_cash, fee_bps=fee_bps)
    return OptimizationCandidate(
        params=dict(params),
        metrics=compute_metrics(result),
        result=result,
        trial_number=trial_number,
    )


def run_grid_optimization(
    data_path: str,
    *,
    strategy_ref: str = DEFAULT_STRATEGY_REF,
    strategy_params: dict[str, Any] | None = None,
    fast_grid: str = "2,3,4",
    slow_grid: str = "6,8,10",
    search_space: dict[str, SearchParam] | None = None,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
) -> OptimizationResult:
    """Search public strategy parameters using deterministic grid enumeration."""
    bars = load_ohlcv_csv(data_path)
    fixed_params = dict(strategy_params or {})
    space = search_space or _default_search_space(fast_grid, slow_grid)
    candidates: list[OptimizationCandidate] = []
    for sampled_params in _grid_parameter_product(space):
        params = {**fixed_params, **sampled_params}
        try:
            candidates.append(
                _evaluate_candidate(
                    bars,
                    strategy_ref=strategy_ref,
                    params=params,
                    initial_cash=initial_cash,
                    fee_bps=fee_bps,
                )
            )
        except (TypeError, ValueError):
            continue
    if not candidates:
        raise ValueError("no valid optimization candidates")
    best = max(candidates, key=lambda item: (item.metrics.score, item.metrics.total_return))
    return OptimizationResult(
        objective="generic_score",
        method="grid",
        strategy_ref=strategy_ref,
        best=best,
        candidates=candidates,
    )


def run_optuna_optimization(
    data_path: str,
    *,
    strategy_ref: str = DEFAULT_STRATEGY_REF,
    strategy_params: dict[str, Any] | None = None,
    fast_grid: str = "2,3,4",
    slow_grid: str = "6,8,10",
    search_space: dict[str, SearchParam] | None = None,
    initial_cash: float = 10_000.0,
    fee_bps: float = 1.0,
    n_trials: int = 16,
    sampler_seed: int = 7,
) -> OptimizationResult:
    """Run Optuna over a public-safe strategy search space."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    bars = load_ohlcv_csv(data_path)
    fixed_params = dict(strategy_params or {})
    space = search_space or _default_search_space(fast_grid, slow_grid)
    candidates: list[OptimizationCandidate] = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params, **_suggest_params(trial, space)}
        try:
            candidate = _evaluate_candidate(
                bars,
                strategy_ref=strategy_ref,
                params=params,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                trial_number=trial.number,
            )
        except (TypeError, ValueError) as exc:
            trial.set_user_attr("rejection_reason", str(exc))
            raise optuna.TrialPruned() from exc
        trial.set_user_attr("total_return", candidate.metrics.total_return)
        trial.set_user_attr("max_drawdown", candidate.metrics.max_drawdown)
        trial.set_user_attr("trade_count", candidate.metrics.trade_count)
        candidates.append(candidate)
        return candidate.metrics.score

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    complete_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not complete_trials or not candidates:
        raise ValueError("optuna produced no complete public candidates")
    best = max(candidates, key=lambda item: (item.metrics.score, item.metrics.total_return))
    return OptimizationResult(
        objective="generic_score",
        method="optuna",
        strategy_ref=strategy_ref,
        best=best,
        candidates=sorted(
            candidates,
            key=lambda item: -1 if item.trial_number is None else item.trial_number,
        ),
    )


def _suggest_params(trial: optuna.Trial, search_space: dict[str, SearchParam]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if spec.kind == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec.low),
                int(spec.high),
                step=int(spec.step or 1),
            )
        elif spec.kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec.low),
                float(spec.high),
                step=float(spec.step) if spec.step is not None else None,
            )
        elif spec.kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec.choices))
        else:
            raise ValueError(f"unsupported search parameter type: {spec.kind}")
    return params


def _grid_parameter_product(search_space: dict[str, SearchParam]) -> list[dict[str, Any]]:
    names = list(search_space)
    values = [_grid_values(search_space[name]) for name in names]
    return [dict(zip(names, items, strict=True)) for items in product(*values)]


def _grid_values(spec: SearchParam) -> list[Any]:
    if spec.kind == "categorical":
        return list(spec.choices)
    if spec.kind == "int":
        low = int(spec.low)
        high = int(spec.high)
        step = int(spec.step or 1)
        return list(range(low, high + 1, step))
    if spec.kind == "float":
        low = float(spec.low)
        high = float(spec.high)
        step = float(spec.step or 1.0)
        values: list[float] = []
        current = low
        while current <= high + 1e-12:
            values.append(current)
            current += step
        return values
    raise ValueError(f"unsupported search parameter type: {spec.kind}")
