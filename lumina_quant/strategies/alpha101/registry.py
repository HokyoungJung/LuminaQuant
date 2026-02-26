"""Alpha101 registry for callable specs and tunable parameter metadata."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

from lumina_quant.tuning.param_registry import HyperParam, ParamRegistry

from .compiler import CompiledAlphaFormula

_ALPHA_ID_MIN = 1
_ALPHA_ID_MAX = 101
_SCHEMA_PARAM_RE = re.compile(r"[^a-zA-Z0-9_]+")

ALPHA101_PARAM_REGISTRY = ParamRegistry()


def _load_formula_module() -> Any:
    return importlib.import_module("lumina_quant.indicators.formulaic_definitions")


def _load_formula_specs() -> dict[int, Any]:
    module = _load_formula_module()
    loaded_specs = getattr(module, "ALPHA_FUNCTION_SPECS", None)
    if not isinstance(loaded_specs, Mapping):
        loaded_specs = getattr(module, "ALPHA_SPECS", None)

    if isinstance(loaded_specs, Mapping):
        specs: dict[int, Any] = {}
        for key, spec in loaded_specs.items():
            specs[int(key)] = spec
        if specs:
            return specs

    raise RuntimeError("ALPHA_FUNCTION_SPECS missing from formulaic_definitions.py")


def _spec_compiled(spec: Any) -> CompiledAlphaFormula | None:
    compiled = getattr(spec, "compiled", None)
    if isinstance(compiled, CompiledAlphaFormula):
        return compiled
    return None


def _spec_callable(spec: Any) -> Callable[..., float | None] | None:
    callable_obj = getattr(spec, "callable", None)
    return callable_obj if callable(callable_obj) else None


def _spec_tunable_constants(spec: Any) -> dict[str, float] | None:
    constants = getattr(spec, "tunable_constants", None)
    if not isinstance(constants, Mapping):
        return None
    out: dict[str, float] = {}
    for key, value in constants.items():
        out[str(key)] = float(value)
    return out


ALPHA_FUNCTION_SPECS = _load_formula_specs()


def _normalize_alpha_id(alpha_id: int) -> int:
    alpha_int = int(alpha_id)
    if alpha_int < _ALPHA_ID_MIN or alpha_int > _ALPHA_ID_MAX:
        raise ValueError(f"Unknown Alpha101 id: {alpha_id}")
    return alpha_int


def list_alpha_ids() -> tuple[int, ...]:
    return tuple(sorted(int(alpha_id) for alpha_id in ALPHA_FUNCTION_SPECS))


def missing_alpha_ids() -> tuple[int, ...]:
    available = set(list_alpha_ids())
    return tuple(alpha_id for alpha_id in range(_ALPHA_ID_MIN, _ALPHA_ID_MAX + 1) if alpha_id not in available)


@lru_cache(maxsize=256)
def get_compiled_formula(alpha_id: int) -> CompiledAlphaFormula:
    alpha_int = _normalize_alpha_id(alpha_id)
    spec = ALPHA_FUNCTION_SPECS.get(alpha_int)
    if spec is None:
        raise ValueError(f"Unknown Alpha101 id: {alpha_id}")

    compiled = _spec_compiled(spec)
    if compiled is None:
        raise RuntimeError(
            "Compiled IR is not exposed by this code-native spec. "
            "Use evaluate_alpha/get_alpha_callable/list_tunable_params instead."
        )
    return compiled


def get_formula(alpha_id: int) -> str:
    _ = _normalize_alpha_id(alpha_id)
    raise RuntimeError(
        "Raw formula strings are intentionally removed. "
        "Use callable specs from lumina_quant.indicators.formulaic_definitions."
    )


def evaluate_alpha(
    alpha_id: int,
    *,
    context,
    rank_window: int = 20,
    param_overrides: Mapping[str, float] | None = None,
    vector_backend: str = "auto",
) -> float | None:
    alpha_int = _normalize_alpha_id(alpha_id)
    spec = ALPHA_FUNCTION_SPECS.get(alpha_int)
    if spec is None:
        raise ValueError(f"Unknown Alpha101 id: {alpha_id}")

    alpha_callable = _spec_callable(spec)
    if alpha_callable is None:
        raise RuntimeError(f"Alpha101 spec {alpha_int} does not expose callable")

    return alpha_callable(
        context=context,
        rank_window=rank_window,
        param_overrides=param_overrides,
        param_registry=ALPHA101_PARAM_REGISTRY,
        vector_backend=vector_backend,
    )


def list_tunable_params(alpha_id: int | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    if alpha_id is None:
        formula_ids = list_alpha_ids()
    else:
        formula_ids = (_normalize_alpha_id(alpha_id),)

    for alpha_key in formula_ids:
        spec = ALPHA_FUNCTION_SPECS[alpha_key]
        constants = _spec_tunable_constants(spec)
        if constants is None:
            continue
        out.update(constants)
    return out


def set_param_overrides(overrides: Mapping[str, float]) -> None:
    ALPHA101_PARAM_REGISTRY.update(overrides)


def clear_param_overrides(prefix: str = "alpha101.") -> None:
    ALPHA101_PARAM_REGISTRY.clear_prefix(prefix=prefix)


def _search_bounds(default: float) -> tuple[float, float]:
    value = float(default)
    if abs(value) <= 1e-12:
        return -1.0, 1.0
    if value > 0.0:
        low = value * 0.5
        high = value * 1.5
    else:
        low = value * 1.5
        high = value * 0.5
    if abs(high - low) <= 1e-12:
        low = value - 1.0
        high = value + 1.0
    return float(low), float(high)


def _schema_param_name(param_key: str) -> str:
    token = _SCHEMA_PARAM_RE.sub("_", str(param_key)).strip("_").lower()
    token = token or "alpha101_param"
    if token[0].isdigit():
        token = f"p_{token}"
    return token


def build_optuna_search_space(alpha_id: int | None = None, *, n_trials: int = 20) -> dict[str, Any]:
    defaults = list_tunable_params(alpha_id=alpha_id)
    schema: dict[str, HyperParam] = {}
    key_map: dict[str, str] = {}

    for param_key, default in sorted(defaults.items()):
        schema_name = _schema_param_name(param_key)
        suffix = 1
        unique_name = schema_name
        while unique_name in schema:
            suffix += 1
            unique_name = f"{schema_name}_{suffix}"
        low, high = _search_bounds(default)
        key_map[unique_name] = param_key
        schema[unique_name] = HyperParam.floating(
            name=unique_name,
            default=float(default),
            low=low,
            high=high,
            description=f"Tunable Alpha101 constant for {param_key}",
        )

    if not schema:
        return {"n_trials": max(1, int(n_trials)), "params": {}}

    local_registry = ParamRegistry()
    local_registry.register("Alpha101Formula", schema, optuna_trials=max(1, int(n_trials)))
    optuna_cfg = local_registry.default_optuna_config("Alpha101Formula")

    raw_params = optuna_cfg.get("params", {})
    params: dict[str, Any] = {}
    if isinstance(raw_params, Mapping):
        for schema_key, spec in raw_params.items():
            mapped_key = key_map.get(str(schema_key), str(schema_key))
            params[mapped_key] = spec

    return {
        "n_trials": int(optuna_cfg.get("n_trials", max(1, int(n_trials)))),
        "params": params,
    }


def get_alpha_callable(alpha_id: int) -> Callable[..., float | None]:
    alpha_int = _normalize_alpha_id(alpha_id)

    def _call(
        *,
        context,
        rank_window: int = 20,
        param_overrides: Mapping[str, float] | None = None,
        vector_backend: str = "auto",
    ) -> float | None:
        return evaluate_alpha(
            alpha_int,
            context=context,
            rank_window=rank_window,
            param_overrides=param_overrides,
            vector_backend=vector_backend,
        )

    _call.__name__ = f"alpha_{alpha_int:03d}_callable"
    return _call


def get_all_alpha_callables() -> dict[int, Callable[..., float | None]]:
    return {alpha_id: get_alpha_callable(alpha_id) for alpha_id in list_alpha_ids()}
