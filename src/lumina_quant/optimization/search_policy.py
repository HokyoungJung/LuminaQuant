"""Shared optimization search policy helpers.

This module keeps low-level search-loop mechanics in one place so CLI and
research runners do not grow divergent Optuna/grid implementations.  It does
not own domain scoring; callers still provide their objective functions and
candidate evaluation logic.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


LOCKED_OOS_SEARCH_FLAGS: dict[str, bool] = {
    "uses_locked_oos_for_selection": False,
    "uses_locked_oos_for_objective": False,
    "uses_locked_oos_for_pruning": False,
    "uses_locked_oos_for_parameter_fitting": False,
}


@dataclass(frozen=True, slots=True)
class BoundedGridResult:
    """Deterministic grid combinations plus audit metadata."""

    combinations: list[dict[str, Any]]
    metadata: dict[str, Any]


def optimization_search_policy_payload(
    *,
    search_method: str,
    objective_policy: str | Mapping[str, Any] | None = None,
    selection_inputs: Sequence[str] = ("train", "validation"),
    bounded_grid_justification: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical optimization/search metadata for artifacts and logs."""
    method = str(search_method).strip().lower()
    payload: dict[str, Any] = {
        "search_method": method,
        "selection_inputs": [str(item) for item in selection_inputs],
        **LOCKED_OOS_SEARCH_FLAGS,
    }
    if objective_policy is not None:
        payload["objective_policy"] = (
            dict(objective_policy)
            if isinstance(objective_policy, Mapping)
            else str(objective_policy)
        )
    if bounded_grid_justification:
        payload["bounded_grid_justification"] = str(bounded_grid_justification)
    if extra:
        payload.update(dict(extra))
    return payload


def build_bounded_grid_combinations(
    param_grid: Mapping[str, Sequence[Any]],
    *,
    max_combinations: int | None = None,
    justification: str,
) -> BoundedGridResult:
    """Return deterministic cartesian combinations with a required audit reason.

    Grid search remains valid for small deterministic enumerations, but the
    caller must provide a justification and optional cap instead of open-coding
    ``itertools.product`` in each runner.
    """
    reason = str(justification).strip()
    if not reason:
        raise ValueError("bounded grid search requires a justification")

    keys = [str(key) for key in param_grid]
    values_by_key = [list(param_grid[key]) for key in param_grid]
    raw_count = 1
    for values in values_by_key:
        raw_count *= len(values)

    limit = None if max_combinations is None else max(0, int(max_combinations))
    combinations: list[dict[str, Any]] = []
    for idx, combo in enumerate(itertools.product(*values_by_key)):
        if limit is not None and idx >= limit:
            break
        combinations.append(dict(zip(keys, combo, strict=True)))

    metadata = optimization_search_policy_payload(
        search_method="bounded_grid",
        bounded_grid_justification=reason,
        extra={
            "candidate_count": len(combinations),
            "raw_candidate_count": raw_count,
            "max_combinations": limit,
            "truncated": limit is not None and raw_count > limit,
            "param_keys": keys,
        },
    )
    return BoundedGridResult(combinations=combinations, metadata=metadata)


def suggest_params_from_optuna_config(
    trial: Any, optuna_config: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Suggest a parameter dictionary from the canonical Optuna config schema."""
    params: dict[str, Any] = {}
    for key, conf in optuna_config.items():
        p_type = conf.get("type")
        if p_type == "int":
            kwargs: dict[str, Any] = {}
            if "step" in conf:
                kwargs["step"] = conf["step"]
            params[key] = trial.suggest_int(key, conf["low"], conf["high"], **kwargs)
        elif p_type == "float":
            step = conf.get("step", None)
            params[key] = trial.suggest_float(key, conf["low"], conf["high"], step=step)
        elif p_type == "categorical":
            params[key] = trial.suggest_categorical(key, conf["choices"])
    return params


def run_optuna_study(
    *,
    optuna_module: Any,
    objective: Callable[[Any], float],
    n_trials: int,
    direction: str = "maximize",
    seed: int | None = None,
    sampler: Any | None = None,
    enqueue_trials: Sequence[Mapping[str, Any]] | None = None,
    n_jobs: int = 1,
    show_progress_bar: bool = False,
) -> Any:
    """Create and optimize an Optuna study through the shared policy path."""
    if optuna_module is None:
        raise RuntimeError("Optuna is required for optuna search")
    resolved_sampler = sampler
    if resolved_sampler is None and seed is not None:
        resolved_sampler = optuna_module.samplers.TPESampler(seed=int(seed))

    study = optuna_module.create_study(direction=str(direction), sampler=resolved_sampler)
    for trial_params in list(enqueue_trials or []):
        study.enqueue_trial(dict(trial_params))
    study.optimize(
        objective,
        n_trials=max(1, int(n_trials)),
        n_jobs=max(1, int(n_jobs)),
        show_progress_bar=bool(show_progress_bar),
    )
    return study
