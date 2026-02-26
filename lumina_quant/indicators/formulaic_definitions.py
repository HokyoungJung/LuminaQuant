"""Alpha101 callable definitions (code-native spec layer)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from lumina_quant.indicators import formulaic_operators as ops
from lumina_quant.strategies.alpha101.formula_sources import ALPHA_PROGRAM_DEFINITIONS


def _last_finite_value(series: pd.Series) -> float | None:
    for value in reversed(series.to_list()):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
    return None


def _to_series(value, index: pd.Index) -> pd.Series:
    return ops.to_series(value, index)


def _resolve_constant(
    *,
    key: str,
    default: float,
    param_overrides: Mapping[str, float] | None,
    param_registry,
) -> float:
    if param_overrides is not None and key in param_overrides:
        try:
            return float(param_overrides[key])
        except (TypeError, ValueError):
            return float(default)
    if param_registry is None:
        return float(default)
    return float(param_registry.get(key, default))


def _build_env(context: dict[str, pd.Series], *, index: pd.Index, rank_window: int) -> dict[str, object]:
    return {
        **context,
        "abs": np.abs,
        "log": np.log,
        "sign": np.sign,
        "rank": lambda s: ops.rank_series(s, index=index, window=max(2, int(rank_window))),
        "ts_rank": lambda s, w: ops.ts_rank_series(s, w, index=index),
        "ts_sum": lambda s, w: ops.ts_sum_series(s, w, index=index),
        "ts_stddev": lambda s, w: ops.ts_stddev_series(s, w, index=index),
        "ts_corr": lambda left, right, window: ops.ts_corr_series(left, right, window, index=index),
        "ts_cov": lambda left, right, window: ops.ts_cov_series(left, right, window, index=index),
        "ts_min": lambda s, w: ops.ts_min_series(s, w, index=index),
        "ts_max": lambda s, w: ops.ts_max_series(s, w, index=index),
        "ts_product": lambda s, w: ops.ts_product_series(s, w, index=index),
        "delay": lambda s, p=1: ops.delay_series(s, p, index=index),
        "delta": lambda s, p=1: ops.delta_series(s, p, index=index),
        "ts_argmax": lambda s, w: ops.ts_argmax_series(s, w, index=index),
        "ts_argmin": lambda s, w: ops.ts_argmin_series(s, w, index=index),
        "decay_linear": lambda s, p=10: ops.decay_linear_series(s, p, index=index),
        "scale": lambda s, a=1.0: ops.scale_series(
            s,
            rank_window=int(rank_window),
            index=index,
            a=a,
        ),
        "signed_power": lambda s, p: ops.signed_power_series(s, p, index=index),
        "where": lambda cond, left, right: ops.where_series(cond, left, right, index=index),
        "indneutralize": lambda s, g: ops.indneutralize_series(s, g, index=index),
        "max": np.maximum,
        "min": np.minimum,
    }


@lru_cache(maxsize=256)
def _build_callable(alpha_id: int) -> Callable[..., float | None]:
    definition = ALPHA_PROGRAM_DEFINITIONS.get(int(alpha_id))
    if definition is None:
        raise ValueError(f"Unknown Alpha101 id: {alpha_id}")

    def _call(
        *,
        context,
        rank_window: int = 20,
        param_overrides: Mapping[str, float] | None = None,
        param_registry=None,
        vector_backend: str = "auto",
    ) -> float | None:
        _ = vector_backend
        if not isinstance(context, dict) or not context:
            return None
        index = next(iter(context.values())).index
        env = _build_env(context, index=index, rank_window=int(rank_window))

        def const(key: str, default: float) -> float:
            return _resolve_constant(
                key=key,
                default=default,
                param_overrides=param_overrides,
                param_registry=param_registry,
            )

        result = definition.program(env, const)
        result_series = _to_series(result, index).replace([np.inf, -np.inf], np.nan)
        if result_series.empty:
            return None
        latest = _last_finite_value(result_series.dropna())
        if latest is None:
            return None
        latest_float = float(latest)
        return latest_float if math.isfinite(latest_float) else None

    _call.__name__ = f"alpha_{int(alpha_id):03d}_compiled_callable"
    _call.__doc__ = f"Generated code-native Alpha101 callable for alpha_id={int(alpha_id)}."
    return _call


@dataclass(frozen=True, slots=True)
class AlphaFunctionSpec:
    """Callable Alpha spec backed by generated code + tunable constants."""

    alpha_id: int

    @property
    def callable(self) -> Callable[..., float | None]:
        return _build_callable(self.alpha_id)

    @property
    def tunable_constants(self) -> dict[str, float]:
        definition = ALPHA_PROGRAM_DEFINITIONS.get(int(self.alpha_id))
        if definition is None:
            return {}
        return dict(definition.tunable_constants)

    def metadata(self) -> dict[str, object]:
        return {
            "alpha_id": int(self.alpha_id),
            "constant_defaults": self.tunable_constants,
        }


def _build_alpha_specs() -> dict[int, AlphaFunctionSpec]:
    return {
        alpha_id: AlphaFunctionSpec(alpha_id=alpha_id)
        for alpha_id in sorted(ALPHA_PROGRAM_DEFINITIONS)
    }


ALPHA_FUNCTION_SPECS: dict[int, AlphaFunctionSpec] = _build_alpha_specs()
ALPHA_SPECS = ALPHA_FUNCTION_SPECS


def get_alpha_function_spec(alpha_id: int) -> AlphaFunctionSpec:
    alpha_int = int(alpha_id)
    spec = ALPHA_FUNCTION_SPECS.get(alpha_int)
    if spec is None:
        raise ValueError(f"Unknown Alpha101 id: {alpha_id}")
    return spec


def get_all_alpha_function_specs() -> dict[int, AlphaFunctionSpec]:
    return dict(ALPHA_FUNCTION_SPECS)


def list_alpha_tunable_constants(alpha_id: int | None = None) -> dict[str, float]:
    specs = (
        [get_alpha_function_spec(alpha_id)]
        if alpha_id is not None
        else [ALPHA_FUNCTION_SPECS[key] for key in sorted(ALPHA_FUNCTION_SPECS)]
    )
    result: dict[str, float] = {}
    for spec in specs:
        result.update(spec.tunable_constants)
    return result


__all__ = [
    "ALPHA_FUNCTION_SPECS",
    "ALPHA_SPECS",
    "AlphaFunctionSpec",
    "get_all_alpha_function_specs",
    "get_alpha_function_spec",
    "list_alpha_tunable_constants",
]
